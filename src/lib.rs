//! fastscape-rs v0.2: Landscape evolution with nonlinear pedogenesis and BDI agents.
//!
//! Integrates 5 solutions:
//! 1. Spatially variable K_d via operator splitting (erosion module)
//! 2. On-the-fly K_f from soil state (erosion + pedogenesis modules)
//! 3. ε-regularized dR/dt with MPRK integration (pedogenesis module)
//! 4. Influence maps for BDI agent → SoA bridging (agents + grid modules)
//! 5. Multi-scale sub-stepping with biological sub-cycle (this module)

pub mod grid;
pub mod graph;
pub mod flow;
pub mod erosion;
pub mod climate;
pub mod integrators;
pub mod pedogenesis;
pub mod agents;

use grid::TerrainGrid;
use erosion::{StreamPowerParams, XiQParams, DiffusionParams};
use climate::{LinearTheoryParams, LfpmParams, EbmParams};
use pedogenesis::PedogenesisParams;
use agents::{AgentParams, AgentPopulation};

#[derive(Clone, Debug)]
pub enum ClimateModel {
    Uniform { rate: f64 },
    LinearTheory(LinearTheoryParams),
    Lfpm(LfpmParams),
}

#[derive(Clone, Debug)]
pub enum ErosionModel {
    StreamPower(StreamPowerParams),
    XiQ(XiQParams),
}

#[derive(Clone, Debug)]
pub struct SimulationConfig {
    pub rows: usize,
    pub cols: usize,
    pub dx: f64,
    pub dy: f64,
    pub total_time: f64,
    pub dt_geomorph: f64,
    pub climate: ClimateModel,
    pub erosion: ErosionModel,
    pub ebm: EbmParams,
    pub diffusion: DiffusionParams,
    pub pedogenesis: PedogenesisParams,
    pub agents: AgentParams,
    pub enable_biogeochem: bool,
    pub enable_pedogenesis: bool,
    pub enable_agents: bool,
    pub init_elevation: f64,
    pub init_perturbation: f64,
    pub seed: u64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            rows: 256,
            cols: 256,
            dx: 500.0,
            dy: 500.0,
            total_time: 5_000_000.0,
            dt_geomorph: 10_000.0,
            climate: ClimateModel::LinearTheory(LinearTheoryParams::default()),
            erosion: ErosionModel::StreamPower(StreamPowerParams::default()),
            ebm: EbmParams::default(),
            diffusion: DiffusionParams::default(),
            pedogenesis: PedogenesisParams::default(),
            agents: AgentParams::default(),
            enable_biogeochem: false,
            enable_pedogenesis: false,
            enable_agents: false,
            init_elevation: 0.0,
            init_perturbation: 1.0,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StepDiagnostics {
    pub time: f64,
    pub mean_elevation: f64,
    pub max_elevation: f64,
    pub mean_precipitation: f64,
    pub mean_erosion_rate: f64,
    pub total_sediment_flux: f64,
    pub mean_soil_state: f64,
    pub max_soil_state: f64,
    pub agent_count: usize,
}

/// Run the full coupled simulation.
///
/// Per macro timestep:
///   1. Biological sub-cycle: agents run representative_steps, accumulate influence (Sol #4+#5)
///   2. Climate → Fill → Route → Erode(K_f(S), Sol #2) → Diffuse(K_d(S), Sol #1)
///   3. Pedogenesis dS/dt with ε-regularization (Sol #3) via MPRK22
///   4. Biogeochemistry via MPRK22
pub fn run_simulation(config: &SimulationConfig) -> (TerrainGrid, Vec<StepDiagnostics>) {
    let mut grid = TerrainGrid::new(config.rows, config.cols, config.dx, config.dy);
    grid.init_random_perturbation(config.init_elevation, config.init_perturbation, config.seed);

    if config.enable_biogeochem {
        for val in grid.soil_carbon.iter_mut() { *val = 5.0; }
        for val in grid.soil_nitrogen.iter_mut() { *val = 0.5; }
    }

    if config.enable_pedogenesis {
        let mut rng_state = config.seed.wrapping_add(999);
        for val in grid.soil_state.iter_mut() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = ((rng_state >> 33) as f64) / (u32::MAX as f64);
            *val = 0.1 + 0.05 * (r - 0.5);
        }
    }

    let mut population = if config.enable_agents {
        Some(AgentPopulation::new(
            &config.agents, config.rows, config.cols,
            config.seed.wrapping_add(7777),
        ))
    } else {
        None
    };

    let n_steps = (config.total_time / config.dt_geomorph).ceil() as usize;
    let output_interval = (n_steps / 50).max(1);
    let mut diagnostics = Vec::new();
    let mut time = 0.0;

    println!("=== fastscape-rs v0.2: Pedogenesis + BDI Agents ===");
    println!("Grid: {}x{} cells, dx={:.0}m", config.rows, config.cols, config.dx);
    println!("Time: {:.0} yr total, dt_geomorph={:.0} yr ({} steps)",
        config.total_time, config.dt_geomorph, n_steps);
    if config.enable_pedogenesis {
        println!("Pedogenesis: ENABLED (Phillips, eps={:.3})", config.pedogenesis.epsilon);
    }
    if config.enable_agents {
        println!("BDI Agents: ENABLED ({} initial, dt_bio={:.0}yr, {} repr. steps)",
            config.agents.initial_count, config.agents.dt_bio, config.agents.representative_steps);
    }
    println!("Diffusion: K_d={:.4}, beta={:.2}, S_c={}",
        config.diffusion.k_d, config.diffusion.beta_diffusivity,
        if config.diffusion.s_c.is_finite() { format!("{:.2}", config.diffusion.s_c) }
        else { "inf (linear)".into() });
    println!();

    for step in 0..n_steps {
        // === Phase A: Biological Sub-Cycle (Solutions #4 + #5) ===
        if let Some(ref mut pop) = population {
            agents::run_biological_subcycle(
                &mut grid, pop, &config.agents, config.dt_geomorph,
            );
        }

        // === Phase B: Geomorphological Step ===
        match &config.climate {
            ClimateModel::Uniform { rate } => { grid.precipitation.fill(*rate); }
            ClimateModel::LinearTheory(params) => {
                climate::linear_theory_precipitation(&mut grid, params);
            }
            ClimateModel::Lfpm(params) => {
                climate::lfpm_precipitation(&mut grid, params);
            }
        }

        climate::compute_temperature_pet(&mut grid, &config.ebm);
        flow::priority_flood_fill(&mut grid);

        let routing = flow::route_flow_dinf(&grid);
        flow::accumulate_flow(&mut grid, &routing);
        flow::compute_slopes(&mut grid);

        match &config.erosion {
            ErosionModel::StreamPower(params) => {
                erosion::implicit_spl_erode(&mut grid, &routing, params, config.dt_geomorph);
            }
            ErosionModel::XiQ(params) => {
                erosion::implicit_xi_q_erode(&mut grid, &routing, params, config.dt_geomorph);
            }
        }

        erosion::hillslope_diffusion(&mut grid, &config.diffusion, config.dt_geomorph);

        if config.enable_pedogenesis {
            pedogenesis::integrate_pedogenesis(&mut grid, &config.pedogenesis, config.dt_geomorph);
        }

        if config.enable_biogeochem {
            integrators::integrate_soil_biogeochemistry(&mut grid, config.dt_geomorph);
        }

        time += config.dt_geomorph;

        if step % output_interval == 0 || step == n_steps - 1 {
            let agent_count = population.as_ref().map_or(0, |p| p.agents.len());
            let diag = collect_diagnostics(&grid, time, agent_count);
            println!(
                "  Step {:5}/{}: t={:.2}Myr | h_mean={:.1}m | h_max={:.1}m | S_mean={:.3} | agents={}",
                step + 1, n_steps, time / 1e6,
                diag.mean_elevation, diag.max_elevation,
                diag.mean_soil_state, diag.agent_count,
            );
            diagnostics.push(diag);
        }
    }

    println!("\nSimulation complete. Final time: {:.2} Myr", time / 1e6);
    (grid, diagnostics)
}

fn collect_diagnostics(grid: &TerrainGrid, time: f64, agent_count: usize) -> StepDiagnostics {
    let n = grid.len() as f64;
    StepDiagnostics {
        time,
        mean_elevation: grid.elevation.iter().sum::<f64>() / n,
        max_elevation: grid.elevation.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        mean_precipitation: grid.precipitation.iter().sum::<f64>() / n,
        mean_erosion_rate: grid.erosion_rate.iter().map(|e| e.abs()).sum::<f64>() / n,
        total_sediment_flux: grid.sediment_flux.iter().sum::<f64>(),
        mean_soil_state: grid.soil_state.iter().sum::<f64>() / n,
        max_soil_state: grid.soil_state.iter().cloned().fold(0.0f64, f64::max),
        agent_count,
    }
}

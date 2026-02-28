//! fastscape-rs v0.3: Corrected biogeomorphic coupling.
//!
//! Fixes applied:
//! 1. Temporal Rupture → Markov Chain land-use regimes (no agent sub-stepping)
//! 2. Erodibility Coupling → Tool & Cover (Sklar & Dietrich 2004), constant K_f
//! 3. Hillslope Diffusion → Fully implicit ADI with variable K_d (no operator split)
//! 4. Pedogenesis Singularity → Michaelis-Menten kinetics (no ε-regularization)
//! 5. Spatial Mismatch → Fisher-KPP reaction-diffusion for population density

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
use agents::LandUseParams;

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
    pub land_use: LandUseParams,
    pub enable_biogeochem: bool,
    pub enable_pedogenesis: bool,
    pub enable_land_use: bool,
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
            land_use: LandUseParams::default(),
            enable_biogeochem: false,
            enable_pedogenesis: false,
            enable_land_use: false,
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
    pub mean_population: f64,
    pub regime_counts: [usize; 4],
}

/// Run the full coupled simulation (v0.3).
///
/// Per macro timestep:
///   1. Land-use dynamics: Fisher-KPP diffusion + Markov regime transitions (Fix #1+#5)
///   2. Climate → Fill → Route → Erode(Tool & Cover, Fix #2) → Diffuse(variable K_d ADI, Fix #3)
///   3. Pedogenesis dS/dt with Michaelis-Menten (Fix #4) via MPRK22
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

    if config.enable_land_use {
        agents::initialize_population(&mut grid, &config.land_use, config.seed.wrapping_add(7777));
    }

    let n_steps = (config.total_time / config.dt_geomorph).ceil() as usize;
    let output_interval = (n_steps / 50).max(1);
    let mut diagnostics = Vec::new();
    let mut time = 0.0;

    println!("=== fastscape-rs v0.3: Corrected Biogeomorphic Coupling ===");
    println!("Grid: {}x{} cells, dx={:.0}m", config.rows, config.cols, config.dx);
    println!("Time: {:.0} yr total, dt_geomorph={:.0} yr ({} steps)",
        config.total_time, config.dt_geomorph, n_steps);
    println!("Fixes applied:");
    println!("  #1: Markov Chain land-use regimes (no temporal extrapolation)");
    println!("  #2: Tool & Cover bedrock erosion (constant K_f)");
    println!("  #3: Fully implicit ADI variable K_d (no operator splitting)");
    println!("  #4: Michaelis-Menten pedogenesis (no ε-regularization)");
    println!("  #5: Fisher-KPP population diffusion (no discrete pathfinding)");
    if config.enable_pedogenesis {
        println!("Pedogenesis: ENABLED (Michaelis-Menten K_m={:.3})", config.pedogenesis.k_m);
    }
    if config.enable_land_use {
        println!("Land-Use: ENABLED (Fisher-KPP r={:.3}, D={:.0})",
            config.land_use.growth_rate, config.land_use.diffusion_coeff);
    }
    println!("Diffusion: K_d={:.4}, beta={:.2}, S_c={}",
        config.diffusion.k_d, config.diffusion.beta_diffusivity,
        if config.diffusion.s_c.is_finite() { format!("{:.2}", config.diffusion.s_c) }
        else { "inf (linear)".into() });
    println!();

    for step in 0..n_steps {
        // === Phase A: Land-Use Dynamics (Fixes #1 + #5) ===
        if config.enable_land_use {
            agents::update_land_use(
                &mut grid, &config.land_use, config.dt_geomorph,
                config.seed, step,
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

        // Fix #2: Tool & Cover erosion (K_f constant, sediment cover modulates)
        match &config.erosion {
            ErosionModel::StreamPower(params) => {
                erosion::implicit_spl_erode(&mut grid, &routing, params, config.dt_geomorph);
            }
            ErosionModel::XiQ(params) => {
                erosion::implicit_xi_q_erode(&mut grid, &routing, params, config.dt_geomorph);
            }
        }

        // Fix #3: Fully implicit variable K_d diffusion
        erosion::hillslope_diffusion(&mut grid, &config.diffusion, config.dt_geomorph);

        // Fix #4: Michaelis-Menten pedogenesis
        if config.enable_pedogenesis {
            pedogenesis::integrate_pedogenesis(&mut grid, &config.pedogenesis, config.dt_geomorph);
        }

        if config.enable_biogeochem {
            integrators::integrate_soil_biogeochemistry(&mut grid, config.dt_geomorph);
        }

        time += config.dt_geomorph;

        if step % output_interval == 0 || step == n_steps - 1 {
            let diag = collect_diagnostics(&grid, time);
            println!(
                "  Step {:5}/{}: t={:.2}Myr | h_mean={:.1}m | h_max={:.1}m | S_mean={:.3} | pop={:.2} | regimes=[P:{} G:{} I:{} D:{}]",
                step + 1, n_steps, time / 1e6,
                diag.mean_elevation, diag.max_elevation,
                diag.mean_soil_state, diag.mean_population,
                diag.regime_counts[0], diag.regime_counts[1],
                diag.regime_counts[2], diag.regime_counts[3],
            );
            diagnostics.push(diag);
        }
    }

    println!("\nSimulation complete. Final time: {:.2} Myr", time / 1e6);
    (grid, diagnostics)
}

fn collect_diagnostics(grid: &TerrainGrid, time: f64) -> StepDiagnostics {
    let n = grid.len() as f64;
    let mut regime_counts = [0usize; 4];
    for &lu in grid.land_use.iter() {
        let idx = (lu as usize).min(3);
        regime_counts[idx] += 1;
    }

    StepDiagnostics {
        time,
        mean_elevation: grid.elevation.iter().sum::<f64>() / n,
        max_elevation: grid.elevation.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        mean_precipitation: grid.precipitation.iter().sum::<f64>() / n,
        mean_erosion_rate: grid.erosion_rate.iter().map(|e| e.abs()).sum::<f64>() / n,
        total_sediment_flux: grid.sediment_flux.iter().sum::<f64>(),
        mean_soil_state: grid.soil_state.iter().sum::<f64>() / n,
        max_soil_state: grid.soil_state.iter().cloned().fold(0.0f64, f64::max),
        mean_population: grid.population_density.iter().sum::<f64>() / n,
        regime_counts,
    }
}

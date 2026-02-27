//! fastscape-rs: High-performance landscape evolution model with reduced-complexity climate coupling.
//!
//! Architecture:
//! - SoA memory layout (grid module) for SIMD-friendly cache utilization
//! - CSR flow graph (graph module) eliminating million-allocation Vec<Vec<usize>>
//! - D-infinity flow routing (flow module) for grid-artifact-free hydrology
//! - Implicit O(n) SPL + ξ-q erosion-deposition solver (erosion module)
//! - Linear Theory / LFPM / EBM climate models (climate module)
//! - MPRK positivity-preserving integrators (integrators module)
//!
//! Temporal coupling uses operator splitting: steady-state climate → flow routing → erosion

pub mod grid;
pub mod graph;
pub mod flow;
pub mod erosion;
pub mod climate;
pub mod integrators;

use grid::TerrainGrid;
use erosion::{StreamPowerParams, XiQParams};
use climate::{LinearTheoryParams, LfpmParams, EbmParams};

/// Which climate model to use for precipitation.
#[derive(Clone, Debug)]
pub enum ClimateModel {
    /// Uniform precipitation (no climate coupling).
    Uniform { rate: f64 },
    /// Linear Theory of Orographic Precipitation (Smith & Barstad 2004). O(N log N).
    LinearTheory(LinearTheoryParams),
    /// Linear Feedback Precipitation Model (Hergarten & Robl 2022). O(N).
    Lfpm(LfpmParams),
}

/// Which erosion model to use.
#[derive(Clone, Debug)]
pub enum ErosionModel {
    /// Pure detachment-limited Stream Power Law.
    StreamPower(StreamPowerParams),
    /// ξ-q under-capacity erosion-deposition (Davy & Lague).
    XiQ(XiQParams),
}

/// Full simulation configuration.
#[derive(Clone, Debug)]
pub struct SimulationConfig {
    // Grid
    pub rows: usize,
    pub cols: usize,
    pub dx: f64,
    pub dy: f64,

    // Time
    pub total_time: f64,     // Total simulation time [yr]
    pub dt_geomorph: f64,    // Geomorphic timestep [yr]

    // Models
    pub climate: ClimateModel,
    pub erosion: ErosionModel,
    pub ebm: EbmParams,

    // Hillslope
    pub k_d: f64,            // Hillslope diffusivity [m²/yr]

    // Biogeochemistry
    pub enable_biogeochem: bool,

    // Initialization
    pub init_elevation: f64,
    pub init_perturbation: f64,
    pub seed: u64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            rows: 256,
            cols: 256,
            dx: 500.0,  // 500m cell spacing
            dy: 500.0,
            total_time: 5_000_000.0, // 5 Myr
            dt_geomorph: 10_000.0,   // 10 kyr timesteps
            climate: ClimateModel::LinearTheory(LinearTheoryParams::default()),
            erosion: ErosionModel::StreamPower(StreamPowerParams::default()),
            ebm: EbmParams::default(),
            k_d: 0.01,
            enable_biogeochem: false,
            init_elevation: 0.0,
            init_perturbation: 1.0,
            seed: 42,
        }
    }
}

/// Diagnostic statistics collected at each output step.
#[derive(Clone, Debug)]
pub struct StepDiagnostics {
    pub time: f64,
    pub mean_elevation: f64,
    pub max_elevation: f64,
    pub mean_precipitation: f64,
    pub mean_erosion_rate: f64,
    pub total_sediment_flux: f64,
}

/// Run the full coupled landscape-climate simulation.
///
/// Operator splitting loop per timestep:
/// 1. Compute steady-state climate fields from current topography
/// 2. Compute temperature and PET via EBM
/// 3. Route flow (D∞) and accumulate discharge
/// 4. Apply erosion (implicit O(n) SPL or ξ-q)
/// 5. Apply hillslope diffusion
/// 6. Optionally integrate soil biogeochemistry (MPRK22)
pub fn run_simulation(config: &SimulationConfig) -> (TerrainGrid, Vec<StepDiagnostics>) {
    let mut grid = TerrainGrid::new(config.rows, config.cols, config.dx, config.dy);
    grid.init_random_perturbation(config.init_elevation, config.init_perturbation, config.seed);

    // Initialize soil pools if biogeochem enabled
    if config.enable_biogeochem {
        for val in grid.soil_carbon.iter_mut() {
            *val = 5.0; // 5 kg C/m²
        }
        for val in grid.soil_nitrogen.iter_mut() {
            *val = 0.5; // 0.5 kg N/m²
        }
    }

    let n_steps = (config.total_time / config.dt_geomorph).ceil() as usize;
    let output_interval = (n_steps / 50).max(1); // ~50 output points
    let mut diagnostics = Vec::new();
    let mut time = 0.0;

    println!("=== fastscape-rs Simulation ===");
    println!("Grid: {}×{} cells, dx={:.0}m", config.rows, config.cols, config.dx);
    println!("Time: {:.0} yr total, dt={:.0} yr ({} steps)", config.total_time, config.dt_geomorph, n_steps);
    println!("Climate: {:?}", std::mem::discriminant(&config.climate));
    println!("Erosion: {:?}", std::mem::discriminant(&config.erosion));
    println!();

    for step in 0..n_steps {
        // === 1. Climate: steady-state precipitation from current topography ===
        match &config.climate {
            ClimateModel::Uniform { rate } => {
                grid.precipitation.fill(*rate);
            }
            ClimateModel::LinearTheory(params) => {
                climate::linear_theory_precipitation(&mut grid, params);
            }
            ClimateModel::Lfpm(params) => {
                climate::lfpm_precipitation(&mut grid, params);
            }
        }

        // === 2. Temperature and PET via EBM ===
        climate::compute_temperature_pet(&mut grid, &config.ebm);

        // === 3. Flow routing (D∞) and discharge accumulation ===
        let routing = flow::route_flow_dinf(&grid);
        flow::accumulate_flow(&mut grid, &routing);
        flow::compute_slopes(&mut grid);

        // === 4. Erosion (implicit O(n)) ===
        match &config.erosion {
            ErosionModel::StreamPower(params) => {
                erosion::implicit_spl_erode(&mut grid, &routing, params, config.dt_geomorph);
            }
            ErosionModel::XiQ(params) => {
                erosion::implicit_xi_q_erode(&mut grid, &routing, params, config.dt_geomorph);
            }
        }

        // === 5. Hillslope diffusion ===
        erosion::hillslope_diffusion(&mut grid, config.k_d, config.dt_geomorph);

        // === 6. Soil biogeochemistry (MPRK22) ===
        if config.enable_biogeochem {
            integrators::integrate_soil_biogeochemistry(&mut grid, config.dt_geomorph);
        }

        time += config.dt_geomorph;

        // Collect diagnostics
        if step % output_interval == 0 || step == n_steps - 1 {
            let diag = collect_diagnostics(&grid, time);
            println!(
                "  Step {:5}/{}: t={:.2} Myr | h_mean={:.1}m | h_max={:.1}m | P_mean={:.3} m/yr",
                step + 1, n_steps,
                time / 1e6,
                diag.mean_elevation,
                diag.max_elevation,
                diag.mean_precipitation,
            );
            diagnostics.push(diag);
        }
    }

    println!("\nSimulation complete. Final time: {:.2} Myr", time / 1e6);
    (grid, diagnostics)
}

fn collect_diagnostics(grid: &TerrainGrid, time: f64) -> StepDiagnostics {
    let n = grid.len() as f64;
    let mean_elevation = grid.elevation.iter().sum::<f64>() / n;
    let max_elevation = grid.elevation.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_precipitation = grid.precipitation.iter().sum::<f64>() / n;
    let mean_erosion_rate = grid.erosion_rate.iter().map(|e| e.abs()).sum::<f64>() / n;
    let total_sediment_flux = grid.sediment_flux.iter().sum::<f64>();

    StepDiagnostics {
        time,
        mean_elevation,
        max_elevation,
        mean_precipitation,
        mean_erosion_rate,
        total_sediment_flux,
    }
}

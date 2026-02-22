/// World Simulation Orchestrator
///
/// Wires together all solver subsystems into a single epoch step:
///
///   Heightmap ──► FastScape (geology) ──► Climate (thermostat + orographic + carbon)
///       │                │                      │
///       │                ▼                      ▼
///       │         Pedogenesis ◄──────── Biome/veg data
///       │           │   │
///       │           ▼   ▼
///       │      kf_mod  kd_mod ──► fed back into FastScape
///       │
///       └──► Nutrients (biogeochem P/K cycling)
///                  │
///                  ▼
///            growth_multipliers ──► fed back into Pedogenesis + Climate
///
/// The LOD manager sits above this: the world simulation operates on the
/// global 2.5D heightmap. The SVO is instantiated separately for the local zone.

use crate::climate::{ClimateConfig, ClimateEngine, WindParams};
use crate::compute::activity_mask::ActivityProcessor;
use crate::geology::{FastScapeSolver, SplParams, TectonicForcing};
use crate::soil::{
    BiogeochemSolver, ColumnEnv, NutrientColumn,
    PedogenesisParams, PedogenesisSolver, PedogenesisState,
};

/// Configuration for the entire world simulation.
#[derive(Debug, Clone)]
pub struct WorldConfig {
    pub grid_width: usize,
    pub grid_height: usize,
    pub cell_size_m: f32,
    pub sea_level: f32,
    pub dt_years: f64,
    pub spl_params: SplParams,
    pub climate_config: ClimateConfig,
    pub wind_params: WindParams,
    pub tectonic: TectonicForcing,
    /// Base erodibility coefficient (before pedogenesis modulation).
    pub kf_base: f32,
    /// Base diffusion coefficient (before pedogenesis modulation).
    pub kd_base: f32,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            grid_width: 256,
            grid_height: 256,
            cell_size_m: 1000.0,
            sea_level: 0.0,
            dt_years: 1000.0,
            spl_params: SplParams::default(),
            climate_config: ClimateConfig::default(),
            wind_params: WindParams::default(),
            tectonic: TectonicForcing::new(0.001),
            kf_base: 1e-5,
            kd_base: 0.01,
        }
    }
}

/// Snapshot of per-epoch diagnostics.
#[derive(Debug, Clone)]
pub struct EpochReport {
    pub epoch: u64,
    pub dt_years: f64,
    pub max_elevation: f32,
    pub mean_elevation: f32,
    pub co2_ppm: f32,
    pub temperature_c: f32,
    pub mean_soil_s: f32,
    pub mean_p_labile: f32,
    pub mean_k_exch: f32,
    pub total_erosion_flux: f32,
    pub climate_summary: String,
}

impl std::fmt::Display for EpochReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[Epoch {:>4}] dt={:.0}yr  elev={:.0}/{:.0}m  CO₂={:.0}ppm  T={:.1}°C  \
             soil={:.3}  P={:.1}  K={:.1}  erosion_flux={:.1}",
            self.epoch, self.dt_years, self.mean_elevation, self.max_elevation,
            self.co2_ppm, self.temperature_c,
            self.mean_soil_s, self.mean_p_labile, self.mean_k_exch,
            self.total_erosion_flux,
        )
    }
}

/// The full world simulation state.
pub struct WorldSimulation {
    pub config: WorldConfig,

    // --- Core terrain data (the "global heightmap" data) ---
    pub elevations: Vec<f32>,
    pub delta_h: Vec<f32>,
    pub uplift: Vec<f32>,
    pub kf: Vec<f32>,
    pub kd: Vec<f32>,

    // --- Solver engines ---
    pub fastscape: FastScapeSolver,
    pub climate: ClimateEngine,
    pub pedo_solver: PedogenesisSolver,
    pub pedo_states: Vec<PedogenesisState>,
    pub pedo_params: Vec<PedogenesisParams>,
    pub biogeo: BiogeochemSolver,
    pub nutrient_columns: Vec<NutrientColumn>,
    pub activity: ActivityProcessor,
    pub activity_mask: Vec<u32>,

    // --- Tracking ---
    pub epoch: u64,
}

impl WorldSimulation {
    /// Initialize the world from config + an initial elevation array.
    pub fn new(config: WorldConfig, initial_elevations: Vec<f32>) -> Self {
        let w = config.grid_width;
        let h = config.grid_height;
        let n = w * h;
        assert_eq!(initial_elevations.len(), n, "Elevation array size mismatch");

        // Geology
        let mut spl = config.spl_params.clone();
        spl.cell_size = config.cell_size_m;
        let fastscape = FastScapeSolver::new(w, h, spl);
        let uplift = config.tectonic.uplift_array(w, h, config.cell_size_m);
        let kf = vec![config.kf_base; n];
        let kd = vec![config.kd_base; n];

        // Climate
        let climate = ClimateEngine::new(
            w, h, config.climate_config.clone(),
            config.wind_params.clone(), config.cell_size_m,
        );

        // Pedogenesis
        let pedo_solver = PedogenesisSolver::new(w, h);
        let pedo_states = pedo_solver.initialize(&initial_elevations, config.sea_level);
        let pedo_params = vec![PedogenesisParams::default(); n];

        // Nutrients
        let biogeo = BiogeochemSolver::new(w, h);
        let nutrient_columns: Vec<NutrientColumn> = initial_elevations
            .iter()
            .map(|&elev| {
                if elev > config.sea_level + 500.0 {
                    NutrientColumn::new_young_mineral(0.25)
                } else if elev > config.sea_level {
                    NutrientColumn::new_alluvial(0.35)
                } else {
                    NutrientColumn::new_tropical(0.4) // ocean floor placeholder
                }
            })
            .collect();

        // Activity mask
        let activity = ActivityProcessor::new(w, h);
        let activity_mask = activity.all_land_active(&initial_elevations, config.sea_level);

        Self {
            elevations: initial_elevations,
            delta_h: vec![0.0; n],
            uplift,
            kf,
            kd,
            fastscape,
            climate,
            pedo_solver,
            pedo_states,
            pedo_params,
            biogeo,
            nutrient_columns,
            activity,
            activity_mask,
            epoch: 0,
            config,
        }
    }

    /// Run one epoch step: geology → climate → pedogenesis → nutrients → feedback.
    pub fn step_epoch(&mut self) -> EpochReport {
        let dt = self.config.dt_years;
        let dt_f32 = dt as f32;
        let n = self.config.grid_width * self.config.grid_height;

        // ===================================================================
        // 1. GEOLOGY: FastScape erosion + uplift + diffusion
        // ===================================================================
        self.delta_h = self.fastscape.step_epoch(
            &mut self.elevations,
            &self.uplift,
            &self.kf,
            &self.kd,
            dt_f32,
        );

        // ===================================================================
        // 2. CLIMATE: thermostat + orographic precipitation + carbon cycle
        // ===================================================================
        // Compute plant biomass proxy from nutrient growth multipliers.
        // IMPORTANT: pass the per-cell MEAN, not the grid-wide sum.
        // The thermostat coefficients expect a normalized biomass value.
        let growth_mults = self.biogeo.growth_multipliers(&self.nutrient_columns);
        let land_cells = self.elevations.iter()
            .filter(|&&e| e > self.config.sea_level).count().max(1);
        let plant_biomass: f32 = self.climate.columns.iter().enumerate()
            .filter(|(i, _)| self.elevations[*i] > self.config.sea_level)
            .map(|(i, col)| {
                col.biome.veg_cover() * growth_mults.get(i).copied().unwrap_or(0.5)
            }).sum::<f32>() / land_cells as f32;

        // Animal respiration scales with biomass: herbivores consume ~10-15%
        // of net primary production, respiring most of it back as CO₂.
        let animal_respiration = plant_biomass * 0.12;

        // Sub-step the climate to prevent overshoot at large dt.
        // The thermostat is an ODE; stepping 5000yr at once overshoots equilibrium.
        let climate_substeps = ((dt / 500.0).ceil() as usize).max(1);
        let climate_sub_dt = dt / climate_substeps as f64;
        for _ in 0..climate_substeps {
            self.climate.step_epoch(
                &self.elevations, &self.delta_h,
                plant_biomass, animal_respiration, climate_sub_dt,
            );
        }

        // ===================================================================
        // 3. PEDOGENESIS: soil formation ↔ erosion feedback
        // ===================================================================
        // Build pedogenesis environments from climate + erosion data.
        let pedo_envs = self.pedo_solver.build_envs(
            &growth_mults, &self.delta_h, self.config.cell_size_m,
        );

        let (kf_mod, kd_mod) = self.pedo_solver.step_epoch(
            &mut self.pedo_states,
            &self.pedo_params,
            &pedo_envs,
            &vec![self.config.kf_base; n],
            &vec![self.config.kd_base; n],
            dt,
        );

        // Feed pedogenesis-modulated erodibility back for next epoch.
        self.kf = kf_mod;
        self.kd = kd_mod;

        // ===================================================================
        // 4. NUTRIENTS: biogeochemical P/K cycling
        // ===================================================================
        let nutrient_envs: Vec<ColumnEnv> = (0..n)
            .map(|i| {
                let col = &self.climate.columns[i];
                let elev = self.elevations[i];
                let dh = self.delta_h[i];
                let flooded = dh > 0.05 && elev < self.config.sea_level + 50.0;
                ColumnEnv {
                    temp_c: col.temp_c,
                    moisture: col.moisture,
                    runoff_m_yr: col.runoff_m,
                    flooded,
                    flood_p_input: if flooded { 8.0 } else { 0.0 },
                    delta_h_m: dh,
                    veg_cover: col.biome.veg_cover(),
                }
            })
            .collect();

        self.biogeo.step_epoch(&mut self.nutrient_columns, &nutrient_envs, dt);

        // Update activity mask (cells may sink below sea level).
        self.activity_mask = self.activity.all_land_active(&self.elevations, self.config.sea_level);

        // ===================================================================
        // 5. DIAGNOSTICS
        // ===================================================================
        self.epoch += 1;

        let mean_soil = self.pedo_solver.mean_s(&self.pedo_states);
        let mean_p = self.biogeo.mean_surface_p_labile(&self.nutrient_columns, &self.activity_mask);
        let mean_k = self.biogeo.mean_surface_k_exch(&self.nutrient_columns, &self.activity_mask);
        let total_erosion = self.fastscape.total_erosion_flux(&self.delta_h, dt_f32);

        EpochReport {
            epoch: self.epoch,
            dt_years: dt,
            max_elevation: self.fastscape.max_elevation(&self.elevations),
            mean_elevation: self.fastscape.mean_elevation(&self.elevations),
            co2_ppm: self.climate.thermostat.state.atmospheric_co2_ppm,
            temperature_c: self.climate.thermostat.state.global_mean_temp_c,
            mean_soil_s: mean_soil,
            mean_p_labile: mean_p,
            mean_k_exch: mean_k,
            total_erosion_flux: total_erosion,
            climate_summary: self.climate.summary(),
        }
    }

    /// Run multiple epochs, returning reports.
    pub fn run(&mut self, n_epochs: usize) -> Vec<EpochReport> {
        (0..n_epochs).map(|_| self.step_epoch()).collect()
    }

    /// Elder God CO₂ injection (direct climate manipulation).
    pub fn inject_co2(&mut self, delta_ppm: f32) {
        self.climate.thermostat.elder_god_co2_injection(delta_ppm);
    }

    /// Update tectonic forcing (e.g., add a new hotspot mid-simulation).
    pub fn update_tectonics(&mut self, tectonic: TectonicForcing) {
        self.config.tectonic = tectonic.clone();
        self.uplift = tectonic.uplift_array(
            self.config.grid_width, self.config.grid_height, self.config.cell_size_m,
        );
    }
}

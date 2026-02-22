/// World Simulation Orchestrator — Hydrologically Coupled
///
/// Epoch pipeline (reordered for proper water coupling):
///
///   ┌─────────────────────────────────────────────────────────────────┐
///   │ 1. CLIMATE (orographic)                                        │
///   │    Elevation → precipitation, PET, temperature per cell         │
///   │                     │                                          │
///   │ 2. GEOLOGY (FastScape with precip-weighted drainage)           │
///   │    Precipitation → drainage accumulation → erosion/deposition  │
///   │                     │                                          │
///   │ 3. HYDROLOGY                                                   │
///   │    Precip + PET + drainage + slopes →                          │
///   │      runoff, soil moisture, water table, flood detection       │
///   │                     │                                          │
///   │ 4. CARBON CYCLE (runoff-coupled weathering)                    │
///   │    Runoff + moisture + erosion → CO₂ drawdown → thermostat     │
///   │                     │                                          │
///   │ 5. PEDOGENESIS (hydro-driven erosion intensity)                │
///   │    Soil moisture + erosion → soil S → kf/kd modulation         │
///   │                     │                                          │
///   │ 6. NUTRIENTS (hydro-driven moisture, runoff, floods)           │
///   │    Real runoff + TWI moisture + flood P → P/K cycling          │
///   │                     │                                          │
///   │ 7. FEEDBACK                                                    │
///   │    growth_multipliers → pedogenesis + climate next epoch       │
///   │    kf/kd → FastScape next epoch                                │
///   └─────────────────────────────────────────────────────────────────┘

use crate::climate::{ClimateConfig, ClimateEngine, WindParams};
use crate::compute::activity_mask::ActivityProcessor;
use crate::geology::{FastScapeSolver, SplParams, TectonicForcing};
use crate::hydrology::{HydrologySolver, HydrologyConfig};
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
    pub kf_base: f32,
    pub kd_base: f32,
    pub hydrology_config: HydrologyConfig,
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
            hydrology_config: HydrologyConfig::default(),
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
    // Hydrology diagnostics
    pub mean_runoff: f32,
    pub mean_soil_moisture: f32,
    pub flooded_cells: usize,
    pub max_discharge: f32,
    pub mean_water_table: f32,
}

impl std::fmt::Display for EpochReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[Epoch {:>4}] dt={:.0}yr  elev={:.0}/{:.0}m  CO₂={:.0}ppm  T={:.1}°C  \
             soil={:.3}  P={:.1}  K={:.1}  erosion={:.1}  \
             runoff={:.3}m  moist={:.2}  flood={}  Q_max={:.0}  WT={:.1}m",
            self.epoch, self.dt_years, self.mean_elevation, self.max_elevation,
            self.co2_ppm, self.temperature_c,
            self.mean_soil_s, self.mean_p_labile, self.mean_k_exch,
            self.total_erosion_flux,
            self.mean_runoff, self.mean_soil_moisture,
            self.flooded_cells, self.max_discharge, self.mean_water_table,
        )
    }
}

/// The full world simulation state.
pub struct WorldSimulation {
    pub config: WorldConfig,

    // --- Core terrain data ---
    pub elevations: Vec<f32>,
    pub delta_h: Vec<f32>,
    pub uplift: Vec<f32>,
    pub kf: Vec<f32>,
    pub kd: Vec<f32>,

    // --- Solver engines ---
    pub fastscape: FastScapeSolver,
    pub climate: ClimateEngine,
    pub hydrology: HydrologySolver,
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

        // Hydrology
        let hydrology = HydrologySolver::new(
            w, h, config.cell_size_m, config.hydrology_config.clone(),
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
                    NutrientColumn::new_tropical(0.4)
                }
            })
            .collect();

        let activity = ActivityProcessor::new(w, h);
        let activity_mask = activity.all_land_active(&initial_elevations, config.sea_level);

        Self {
            elevations: initial_elevations,
            delta_h: vec![0.0; n],
            uplift, kf, kd,
            fastscape, climate, hydrology,
            pedo_solver, pedo_states, pedo_params,
            biogeo, nutrient_columns,
            activity, activity_mask,
            epoch: 0, config,
        }
    }

    /// Run one fully-coupled epoch step.
    pub fn step_epoch(&mut self) -> EpochReport {
        let dt = self.config.dt_years;
        let dt_f32 = dt as f32;
        let n = self.config.grid_width * self.config.grid_height;
        let sea = self.config.sea_level;

        // ===================================================================
        // 1. CLIMATE: orographic precipitation + temperature
        //    (Run first so we have per-cell precip for geology & hydrology)
        // ===================================================================
        let growth_mults = self.biogeo.growth_multipliers(&self.nutrient_columns);
        let land_cells = self.elevations.iter().filter(|&&e| e > sea).count().max(1);
        let plant_biomass: f32 = self.climate.columns.iter().enumerate()
            .filter(|(i, _)| self.elevations[*i] > sea)
            .map(|(i, col)| {
                col.biome.veg_cover() * growth_mults.get(i).copied().unwrap_or(0.5)
            }).sum::<f32>() / land_cells as f32;
        let animal_respiration = plant_biomass * 0.12;

        // Run orographic + lapse rate to get per-cell precipitation and PET
        // (but don't update thermostat yet — need hydro runoff for that)
        let oro = self.climate.orographic.compute(&self.elevations);
        let base_temp = self.climate.thermostat.state.global_mean_temp_c;

        // Collect per-cell precipitation and PET
        let mut precipitation = vec![0.0f32; n];
        let mut pet = vec![0.0f32; n];
        for i in 0..n {
            let elev = self.elevations[i];
            let precip = oro.precipitation[i];
            let temp = self.climate.lapse.local_temp(base_temp, elev);
            precipitation[i] = precip;
            pet[i] = self.climate.lapse.pet(temp);

            // Update climate columns with current orographic data
            let runoff = self.climate.lapse.runoff(precip, temp, elev);
            let moisture = (runoff / precip.max(1e-6)).clamp(0.0, 1.0);
            let biome = crate::climate::Biome::classify(temp, precip, elev);
            self.climate.columns[i] = crate::climate::ColumnClimate {
                temp_c: temp, precipitation_m: precip,
                runoff_m: runoff, moisture, biome,
            };
        }

        // ===================================================================
        // 2. GEOLOGY: FastScape with precipitation-weighted drainage
        //    (Wet slopes erode faster than dry slopes)
        // ===================================================================
        self.delta_h = self.fastscape.step_epoch_with_precip(
            &mut self.elevations,
            &self.uplift,
            &self.kf,
            &self.kd,
            &precipitation,
            dt_f32,
        );

        // ===================================================================
        // 3. HYDROLOGY: compute water table, soil moisture, floods, runoff
        //    from drainage network + precipitation + PET
        // ===================================================================
        self.hydrology.compute(
            &precipitation,
            &pet,
            &self.elevations,
            self.fastscape.drainage_area(),
            self.fastscape.slopes(),
            self.fastscape.receivers(),
            &self.delta_h,
            sea,
        );

        // Extract hydro arrays for downstream consumers
        let hydro_runoff: Vec<f32> = self.hydrology.cells.iter().map(|c| c.runoff_m).collect();
        let hydro_moisture: Vec<f32> = self.hydrology.cells.iter().map(|c| c.soil_moisture).collect();

        // ===================================================================
        // 4. CARBON CYCLE: runoff-coupled silicate weathering → thermostat
        //    (Wet climates weather faster, drawing down more CO₂)
        // ===================================================================
        let climate_substeps = ((dt / 500.0).ceil() as usize).max(1);
        let climate_sub_dt = dt / climate_substeps as f64;
        for _ in 0..climate_substeps {
            let fluxes = self.climate.carbon.compute_fluxes_hydro(
                &self.elevations, &self.delta_h,
                &hydro_runoff, &hydro_moisture,
                self.config.grid_width, self.config.grid_height,
                &self.climate.thermostat.state,
            );
            self.climate.thermostat.step_epoch(
                plant_biomass, animal_respiration,
                fluxes.silicate_drawdown, fluxes.organic_burial,
                fluxes.anthropogenic_flush, climate_sub_dt,
            );
        }

        // ===================================================================
        // 5. PEDOGENESIS: soil formation ↔ erosion, using hydro moisture
        // ===================================================================
        let pedo_envs: Vec<crate::soil::PedoEnv> = (0..n).map(|i| {
            let gm = growth_mults.get(i).copied().unwrap_or(0.5);
            let dh = self.delta_h[i];
            let hydro = &self.hydrology.cells[i];
            // Erosion intensity from actual delta_h, amplified by runoff
            let erosion_intensity = (dh.abs() / self.config.cell_size_m * 1000.0).min(1.0);
            // Vegetation cover driven by moisture availability
            let veg_cover = (gm * 1.2 * hydro.soil_moisture.max(0.1)).min(1.0);
            crate::soil::PedoEnv {
                veg_cover,
                growth_multiplier: gm * hydro.soil_moisture.max(0.2),
                erosion_intensity,
            }
        }).collect();

        let (kf_mod, kd_mod) = self.pedo_solver.step_epoch(
            &mut self.pedo_states,
            &self.pedo_params,
            &pedo_envs,
            &vec![self.config.kf_base; n],
            &vec![self.config.kd_base; n],
            dt,
        );
        self.kf = kf_mod;
        self.kd = kd_mod;

        // ===================================================================
        // 6. NUTRIENTS: P/K cycling driven by real hydrology
        //    (Soil moisture, runoff, flood status all from hydrology solver)
        // ===================================================================
        let nutrient_envs: Vec<ColumnEnv> = (0..n).map(|i| {
            let col = &self.climate.columns[i];
            let hydro = &self.hydrology.cells[i];
            ColumnEnv {
                temp_c: col.temp_c,
                moisture: hydro.soil_moisture,         // from TWI + water balance
                runoff_m_yr: hydro.runoff_m,           // from Budyko curve
                flooded: hydro.flooded,                // from discharge threshold
                flood_p_input: hydro.flood_p_input,    // from upstream erosion
                delta_h_m: self.delta_h[i],
                veg_cover: col.biome.veg_cover(),
            }
        }).collect();

        self.biogeo.step_epoch(&mut self.nutrient_columns, &nutrient_envs, dt);
        self.activity_mask = self.activity.all_land_active(&self.elevations, sea);

        // ===================================================================
        // 7. DIAGNOSTICS
        // ===================================================================
        self.epoch += 1;

        EpochReport {
            epoch: self.epoch,
            dt_years: dt,
            max_elevation: self.fastscape.max_elevation(&self.elevations),
            mean_elevation: self.fastscape.mean_elevation(&self.elevations),
            co2_ppm: self.climate.thermostat.state.atmospheric_co2_ppm,
            temperature_c: self.climate.thermostat.state.global_mean_temp_c,
            mean_soil_s: self.pedo_solver.mean_s(&self.pedo_states),
            mean_p_labile: self.biogeo.mean_surface_p_labile(&self.nutrient_columns, &self.activity_mask),
            mean_k_exch: self.biogeo.mean_surface_k_exch(&self.nutrient_columns, &self.activity_mask),
            total_erosion_flux: self.fastscape.total_erosion_flux(&self.delta_h, dt_f32),
            climate_summary: self.climate.summary(),
            mean_runoff: self.hydrology.mean_runoff(),
            mean_soil_moisture: self.hydrology.mean_soil_moisture(),
            flooded_cells: self.hydrology.flooded_count(),
            max_discharge: self.hydrology.max_discharge(),
            mean_water_table: self.hydrology.mean_water_table(),
        }
    }

    pub fn run(&mut self, n_epochs: usize) -> Vec<EpochReport> {
        (0..n_epochs).map(|_| self.step_epoch()).collect()
    }

    pub fn inject_co2(&mut self, delta_ppm: f32) {
        self.climate.thermostat.elder_god_co2_injection(delta_ppm);
    }

    pub fn update_tectonics(&mut self, tectonic: TectonicForcing) {
        self.config.tectonic = tectonic.clone();
        self.uplift = tectonic.uplift_array(
            self.config.grid_width, self.config.grid_height, self.config.cell_size_m,
        );
    }
}

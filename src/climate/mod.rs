// ============================================================================
// Climate Engine — Phase 3: Proserpina Principle + Geological Carbon Thermostat
//
// Implements the three-layer climate model from the design document:
//
//   1. GLOBAL THERMOSTAT — atmospheric_carbon variable drives base temperature
//      via the Proserpina Principle (producers cool, consumers warm).
//      Carbonate-silicate weathering feedback: tectonic uplift → erosion →
//      silicate weathering → CO₂ drawdown → glaciation.
//
//   2. LOCAL TEMPERATURE — environmental lapse rate: −6.5°C / 1000 m
//      Creates vertical biome stratification from terrain heightmap.
//
//   3. LOCAL RAINFALL — orographic forcing via Linear Feedback Precipitation
//      Model (LFPM 1.0): wind vector ray-cast over heightmap, moisture
//      depletes on windward slopes, rain shadow on leeward.
//      Runoff output feeds directly into FastScape SPL as drainage area A.
//
// Integration points:
//   - Receives Δh from FastScape → updates carbonate-silicate weathering flux
//   - Receives plant/animal biomass totals from BDI agents (Phase 3)
//   - Outputs: temperature[], rainfall[], runoff[], atmospheric_carbon
//   - Runoff[] → FastScape SplParams.drainage overrides
//   - Temperature[] → BiogeochemSolver ColumnEnv.temp_c
//   - Rainfall[]    → BiogeochemSolver ColumnEnv.moisture
// ============================================================================

pub mod thermostat;
pub mod lapse_rate;
pub mod orographic;
pub mod carbon_cycle;

pub use thermostat::{ClimateState, ProserpinaThermostat, ClimateConfig};
pub use lapse_rate::LapseRateEngine;
pub use orographic::{OrographicEngine, WindParams, OrographicOutput};
pub use carbon_cycle::{CarbonCycleEngine, CarbonFluxes, SilicateWeatheringParams};

use crate::compute::activity_mask::ActivityProcessor;

/// Full per-column climate output — passed to terrain and biogeochem solvers.
#[derive(Debug, Clone)]
pub struct ColumnClimate {
    /// Local mean annual temperature [°C]
    pub temp_c: f32,
    /// Annual precipitation [m/yr]
    pub precipitation_m: f32,
    /// Annual runoff [m/yr] — precipitation minus evapotranspiration
    pub runoff_m: f32,
    /// Soil moisture as fraction of field capacity (0–1)
    pub moisture: f32,
    /// Biome classification derived from temp + precip
    pub biome: Biome,
}

/// Coarse biome classification for gameplay/visual feedback
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Biome {
    Ocean          = 0,
    TropicalForest = 1,
    TemperateForest= 2,
    Grassland      = 3,
    Desert         = 4,
    Tundra         = 5,
    Alpine         = 6,
    Glacier        = 7,
}

impl Biome {
    /// Classify biome from temperature and precipitation
    pub fn classify(temp_c: f32, precip_m: f32, elevation_m: f32) -> Self {
        if elevation_m < 0.0 { return Biome::Ocean; }
        if temp_c < -10.0    { return Biome::Glacier; }
        if temp_c < 0.0 && elevation_m > 1500.0 { return Biome::Alpine; }
        if temp_c < 5.0      { return Biome::Tundra; }
        if precip_m < 0.25   { return Biome::Desert; }
        if precip_m < 0.6 && temp_c < 20.0 { return Biome::Grassland; }
        if temp_c > 22.0 && precip_m > 1.5  { return Biome::TropicalForest; }
        Biome::TemperateForest
    }

    /// Vegetation cover fraction (0–1) for pedogenesis and biogeochem
    pub fn veg_cover(&self) -> f32 {
        match self {
            Biome::TropicalForest  => 0.95,
            Biome::TemperateForest => 0.80,
            Biome::Grassland       => 0.55,
            Biome::Desert          => 0.05,
            Biome::Tundra          => 0.30,
            Biome::Alpine          => 0.15,
            Biome::Glacier         => 0.0,
            Biome::Ocean           => 0.0,
        }
    }

    /// Photosynthesis rate modifier (affects Proserpina CO₂ drawdown)
    pub fn photosynthesis_rate(&self) -> f32 {
        match self {
            Biome::TropicalForest  => 1.0,
            Biome::TemperateForest => 0.65,
            Biome::Grassland       => 0.35,
            Biome::Desert          => 0.05,
            Biome::Tundra          => 0.15,
            Biome::Alpine          => 0.10,
            Biome::Glacier         => 0.0,
            Biome::Ocean           => 0.0,
        }
    }

    /// Name for diagnostics
    pub fn name(&self) -> &'static str {
        match self {
            Biome::TropicalForest  => "Tropical Forest",
            Biome::TemperateForest => "Temperate Forest",
            Biome::Grassland       => "Grassland",
            Biome::Desert          => "Desert",
            Biome::Tundra          => "Tundra",
            Biome::Alpine          => "Alpine",
            Biome::Glacier         => "Glacier",
            Biome::Ocean           => "Ocean",
        }
    }
}

/// Top-level climate driver — orchestrates all sub-engines each epoch.
pub struct ClimateEngine {
    pub thermostat:  ProserpinaThermostat,
    pub lapse:       LapseRateEngine,
    pub orographic:  OrographicEngine,
    pub carbon:      CarbonCycleEngine,
    pub grid_width:  usize,
    pub grid_height: usize,
    /// Cached per-column outputs from last epoch
    pub columns: Vec<ColumnClimate>,
}

impl ClimateEngine {
    pub fn new(
        grid_width: usize,
        grid_height: usize,
        config: ClimateConfig,
        wind: WindParams,
        cell_size_m: f32,
    ) -> Self {
        let n = grid_width * grid_height;
        let thermostat  = ProserpinaThermostat::new(config.clone());
        let lapse       = LapseRateEngine::new(config.clone());
        let orographic  = OrographicEngine::new(grid_width, grid_height, wind, cell_size_m);
        let carbon      = CarbonCycleEngine::new(SilicateWeatheringParams::default());

        // Default climate column (will be overwritten on first step_epoch)
        let default_col = ColumnClimate {
            temp_c: 15.0,
            precipitation_m: 0.8,
            runoff_m: 0.4,
            moisture: 0.6,
            biome: Biome::TemperateForest,
        };

        Self {
            thermostat,
            lapse,
            orographic,
            carbon,
            grid_width,
            grid_height,
            columns: vec![default_col; n],
        }
    }

    /// Full climate update for one geological epoch.
    ///
    /// Inputs:
    ///   elevations — current heightmap [m]
    ///   delta_h    — elevation changes from FastScape this epoch [m]
    ///   plant_biomass_total  — total plant voxel/agent mass (Proserpina drawdown)
    ///   animal_respiration   — total animal respiration (Proserpina emission)
    ///   dt_years   — epoch timestep [years]
    pub fn step_epoch(
        &mut self,
        elevations: &[f32],
        delta_h: &[f32],
        plant_biomass_total: f32,
        animal_respiration: f32,
        dt_years: f64,
    ) {
        // 1. Update carbon cycle (silicate weathering from erosion, biotic fluxes)
        let fluxes = self.carbon.compute_fluxes(
            elevations,
            delta_h,
            self.grid_width,
            self.grid_height,
            &self.thermostat.state,
        );

        // 2. Advance global thermostat
        self.thermostat.step_epoch(
            plant_biomass_total,
            animal_respiration,
            fluxes.silicate_drawdown,
            fluxes.organic_burial,
            fluxes.anthropogenic_flush,
            dt_years,
        );

        // 3. Compute orographic rainfall
        let oro = self.orographic.compute(elevations);

        // 4. Build per-column climate from global base temp + lapse + orographic
        let base_temp = self.thermostat.state.global_mean_temp_c;
        for idx in 0..(self.grid_width * self.grid_height) {
            let elev = elevations[idx];
            let precip = oro.precipitation[idx];

            let temp = self.lapse.local_temp(base_temp, elev);
            let runoff = self.lapse.runoff(precip, temp, elev);
            let moisture = (runoff / precip.max(1e-6)).clamp(0.0, 1.0);
            let biome = Biome::classify(temp, precip, elev);

            self.columns[idx] = ColumnClimate {
                temp_c: temp,
                precipitation_m: precip,
                runoff_m: runoff,
                moisture,
                biome,
            };
        }
    }

    /// Emit a diagnostic summary line
    pub fn summary(&self) -> String {
        self.thermostat.state.summary()
    }

    /// Count biome distribution across active land cells
    pub fn biome_counts(&self, elevations: &[f32], sea_level: f32) -> [(Biome, usize); 8] {
        let mut counts = [0usize; 8];
        for (i, col) in self.columns.iter().enumerate() {
            if elevations[i] > sea_level {
                counts[col.biome as usize] += 1;
            }
        }
        [
            (Biome::Ocean,           counts[0]),
            (Biome::TropicalForest,  counts[1]),
            (Biome::TemperateForest, counts[2]),
            (Biome::Grassland,       counts[3]),
            (Biome::Desert,          counts[4]),
            (Biome::Tundra,          counts[5]),
            (Biome::Alpine,          counts[6]),
            (Biome::Glacier,         counts[7]),
        ]
    }
}

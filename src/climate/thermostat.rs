// ============================================================================
// Proserpina Thermostat — Global Climate Engine
//
// Implements Gregory Retallack's Proserpina Principle:
//   "Producers (plants) cool the planet; consumers (animals, humans) warm it."
//
// The simulation maintains a single global variable: atmospheric_carbon [ppm].
// This drives global base temperature via a logarithmic forcing function
// (matches real-world radiative forcing: ΔT = λ × ln(C / C₀)).
//
// Every geological epoch tick:
//   1. Plant voxels/agents draw down atmospheric_carbon (photosynthesis)
//   2. Animal respiration + organic decay + fossil fuel burning add to it
//   3. Silicate weathering (from FastScape erosion) draws it down further
//   4. Organic carbon burial sequesters it long-term
//   5. Anthropogenic erosion causes "Carbon Flush" — ancient SOM oxidizes
//
// The resulting endogenous greenhouse/ice-age cycles emerge purely from
// population and tectonic dynamics — no GCM required.
// ============================================================================

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Pre-industrial baseline CO₂ [ppm] — equilibrium reference
pub const C0_BASELINE_PPM: f32 = 280.0;

/// Climate sensitivity: °C per doubling of CO₂ (IPCC best estimate ~3°C)
pub const LAMBDA_SENSITIVITY: f32 = 3.0;

/// Global mean temperature at baseline CO₂ [°C]
pub const T0_BASELINE_C: f32 = 14.0;

/// Minimum CO₂ before extreme glaciation [ppm]
pub const C_MIN_PPM: f32 = 150.0;

/// Maximum CO₂ before runaway greenhouse [ppm]
pub const C_MAX_PPM: f32 = 8000.0;

/// Ice-age threshold [°C] — global mean below this triggers glaciation
pub const ICE_AGE_THRESHOLD_C: f32 = 8.0;

/// Hothouse threshold [°C] — global mean above this triggers heat crisis
pub const HOTHOUSE_THRESHOLD_C: f32 = 22.0;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateConfig {
    /// Initial atmospheric CO₂ [ppm]
    pub initial_co2_ppm: f32,
    /// Climate sensitivity λ [°C per doubling]
    pub lambda: f32,
    /// Baseline global temperature at C₀ [°C]
    pub t0_baseline: f32,
    /// Plant photosynthesis CO₂ uptake coefficient [ppm per unit biomass per year]
    pub plant_uptake_coeff: f32,
    /// Animal respiration CO₂ emission coefficient [ppm per unit biomass per year]
    pub animal_respiration_coeff: f32,
    /// Organic matter decay baseline rate [yr⁻¹] — adds to CO₂
    pub decay_rate: f32,
    /// Volcanic outgassing rate [ppm/yr] — geological CO₂ background source
    pub volcanic_outgas_rate: f32,
    /// Prevailing wind direction (radians from east, clockwise)
    pub wind_angle_rad: f32,
    /// Prevailing wind speed [m/s] — affects orographic moisture transport
    pub wind_speed_ms: f32,
}

impl Default for ClimateConfig {
    fn default() -> Self {
        Self {
            initial_co2_ppm:          C0_BASELINE_PPM,
            lambda:                    LAMBDA_SENSITIVITY,
            t0_baseline:               T0_BASELINE_C,
            plant_uptake_coeff:        0.002,
            animal_respiration_coeff:  0.001,
            decay_rate:                0.005,
            volcanic_outgas_rate:      0.01, // ~0.01 ppm/yr (Berner 2004)
            wind_angle_rad:            0.0,  // westerly by default
            wind_speed_ms:             8.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Climate State
// ---------------------------------------------------------------------------

/// The global climate state — one instance per simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateState {
    /// Current atmospheric CO₂ [ppm]
    pub atmospheric_co2_ppm: f32,
    /// Derived global mean temperature [°C]
    pub global_mean_temp_c: f32,
    /// Cumulative silicate weathering drawdown [ppm total]
    pub cumulative_silicate_drawdown: f32,
    /// Cumulative organic burial sequestration [ppm total]
    pub cumulative_organic_burial: f32,
    /// Whether we're currently in an ice age
    pub in_ice_age: bool,
    /// Whether we're in a hothouse state
    pub in_hothouse: bool,
    /// Epoch count for tracking
    pub epoch: u64,
}

impl ClimateState {
    pub fn new(initial_co2_ppm: f32, t0_baseline: f32, lambda: f32) -> Self {
        let temp = t0_from_co2(initial_co2_ppm, t0_baseline, lambda);
        Self {
            atmospheric_co2_ppm: initial_co2_ppm,
            global_mean_temp_c: temp,
            cumulative_silicate_drawdown: 0.0,
            cumulative_organic_burial: 0.0,
            in_ice_age: temp < ICE_AGE_THRESHOLD_C,
            in_hothouse: temp > HOTHOUSE_THRESHOLD_C,
            epoch: 0,
        }
    }

    pub fn summary(&self) -> String {
        let phase = if self.in_ice_age {
            "❄ ICE AGE"
        } else if self.in_hothouse {
            "🔥 HOTHOUSE"
        } else {
            "🌿 Interglacial"
        };
        format!(
            "CO₂={:.0}ppm  T={:.1}°C  {}  SilicateΣ={:.1}ppm  BurialΣ={:.1}ppm",
            self.atmospheric_co2_ppm,
            self.global_mean_temp_c,
            phase,
            self.cumulative_silicate_drawdown,
            self.cumulative_organic_burial,
        )
    }
}

/// Logarithmic temperature forcing: ΔT = λ × log₂(C / C₀)
pub fn t0_from_co2(co2_ppm: f32, t0_baseline: f32, lambda: f32) -> f32 {
    if co2_ppm <= 0.0 {
        return t0_baseline - lambda * 10.0; // deep freeze
    }
    let forcing = lambda * (co2_ppm / C0_BASELINE_PPM).log2();
    t0_baseline + forcing
}

// ---------------------------------------------------------------------------
// Proserpina Thermostat Engine
// ---------------------------------------------------------------------------

/// Drives the global CO₂/temperature cycle each geological epoch.
pub struct ProserpinaThermostat {
    pub config: ClimateConfig,
    pub state:  ClimateState,
}

impl ProserpinaThermostat {
    pub fn new(config: ClimateConfig) -> Self {
        let state = ClimateState::new(
            config.initial_co2_ppm,
            config.t0_baseline,
            config.lambda,
        );
        Self { config, state }
    }

    /// Advance one geological epoch.
    ///
    /// Parameters (all as totals for the domain per year):
    ///   plant_biomass       — sum of plant agent mass / photosynthetic voxels
    ///   animal_respiration  — sum of animal agent metabolic output
    ///   silicate_drawdown   — CO₂ removed by silicate weathering [ppm/yr]
    ///   organic_burial      — CO₂ sequestered by sediment burial [ppm/yr]
    ///   anthropogenic_flush — CO₂ released by exposed SOM oxidation [ppm/yr]
    ///   dt_years            — epoch duration [years]
    pub fn step_epoch(
        &mut self,
        plant_biomass: f32,
        animal_respiration: f32,
        silicate_drawdown: f32,      // [ppm/yr] — from CarbonCycleEngine
        organic_burial: f32,         // [ppm/yr] — from high-erosion zones
        anthropogenic_flush: f32,    // [ppm/yr] — from deforestation/tillage
        dt_years: f64,
    ) {
        let dt = dt_years as f32;
        let c = &self.config;

        // ---- SOURCES (add CO₂) ----
        // Volcanic outgassing (constant background)
        let volcanic = c.volcanic_outgas_rate * dt;

        // Animal respiration + organic decay
        let respiration = (c.animal_respiration_coeff * animal_respiration
                         + c.decay_rate * plant_biomass * 0.1) // partial decay of plant matter
                         * dt;

        // Carbon flush from erosion-exposed ancient SOM
        let flush = anthropogenic_flush * dt;

        // ---- SINKS (remove CO₂) ----
        // Photosynthesis (temperature-scaled — plants are less productive in cold/hot extremes)
        let temp_scale = photosynthesis_temp_scale(self.state.global_mean_temp_c);
        let photosynthesis = c.plant_uptake_coeff * plant_biomass * temp_scale * dt;

        // Silicate weathering (from FastScape erosion exposing fresh bedrock)
        let silicate = silicate_drawdown * dt;

        // Organic carbon burial (high-erosion zones bury OM before it decomposes)
        let burial = organic_burial * dt;

        // ---- NET CO₂ CHANGE ----
        let delta_co2 = volcanic + respiration + flush - photosynthesis - silicate - burial;

        self.state.atmospheric_co2_ppm =
            (self.state.atmospheric_co2_ppm + delta_co2).clamp(C_MIN_PPM, C_MAX_PPM);

        // ---- UPDATE DERIVED TEMPERATURE ----
        self.state.global_mean_temp_c = t0_from_co2(
            self.state.atmospheric_co2_ppm,
            self.config.t0_baseline,
            self.config.lambda,
        );

        // ---- UPDATE ACCUMULATORS ----
        self.state.cumulative_silicate_drawdown += silicate;
        self.state.cumulative_organic_burial    += burial;

        // ---- PHASE TRANSITIONS ----
        self.state.in_ice_age   = self.state.global_mean_temp_c < ICE_AGE_THRESHOLD_C;
        self.state.in_hothouse  = self.state.global_mean_temp_c > HOTHOUSE_THRESHOLD_C;

        self.state.epoch += 1;
    }

    /// Elder God: directly inject or remove CO₂ [ppm]
    /// Positive = greenhouse forcing, negative = cooling
    pub fn elder_god_co2_injection(&mut self, delta_ppm: f32) {
        self.state.atmospheric_co2_ppm =
            (self.state.atmospheric_co2_ppm + delta_ppm).clamp(C_MIN_PPM, C_MAX_PPM);
        self.state.global_mean_temp_c = t0_from_co2(
            self.state.atmospheric_co2_ppm,
            self.config.t0_baseline,
            self.config.lambda,
        );
        println!(
            "[ELDER GOD] CO₂ injection {:+.1}ppm → {:.0}ppm, T={:.1}°C",
            delta_ppm,
            self.state.atmospheric_co2_ppm,
            self.state.global_mean_temp_c,
        );
    }
}

/// Temperature scaling for photosynthesis rate (peaked around 15–25°C)
/// Models the Gaussian-like optimum of C3 photosynthesis.
fn photosynthesis_temp_scale(temp_c: f32) -> f32 {
    // Optimum at 18°C, half-rate at -5°C and 40°C
    let opt = 18.0_f32;
    let width = 20.0_f32;
    let t = -(temp_c - opt).powi(2) / (2.0 * width * width);
    t.exp().clamp(0.01, 1.0) // minimum 1% even in extreme cold/heat
}

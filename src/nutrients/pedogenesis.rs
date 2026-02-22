// ============================================================================
// Pedogenesis Solver — Phase 2: Critical Zone
//
// Implements Phillips' nonlinear soil development model:
//   dS/dt = dP/dt − dR/dt
//
// Where:
//   dP/dt = c1 × exp(−k1 × S)    progressive pedogenesis (self-limiting)
//   dR/dt = c2 × exp(−k2 / S)    regressive pedogenesis (erosion-driven)
//   S ∈ [0, 1]                    degree of soil development
//
// S feeds back into FastScape via K_f and K_d modulation:
//   K_f(S) = K_f_base × exp(−β_f × S)    (high S → more cohesion → lower K_f)
//   K_d(S) = K_d_base × (1 + β_d × S)   (high S → heavier soil → higher K_d)
//
// This closes the materialist loop: biology → soil state → physical erosion.
// ============================================================================

use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Pedogenesis parameters
// ---------------------------------------------------------------------------

/// Maximum progressive pedogenesis rate (per year) under ideal conditions.
/// Driven by biological input (vegetation, fauna, root activity).
pub const C1_DEFAULT: f32 = 0.002;
/// Progressive feedback coefficient: rate slows as soil matures.
pub const K1_DEFAULT: f32 = 3.0;

/// Maximum regressive pedogenesis rate (per year) — erosion/disturbance ceiling.
pub const C2_DEFAULT: f32 = 0.001;
/// Regressive feedback: bare young soil is most vulnerable (low S → high R).
pub const K2_DEFAULT: f32 = 0.05;

/// K_f (erodibility) modulation: how strongly S suppresses erodibility.
pub const BETA_F: f32 = 2.5;
/// K_d (diffusivity) modulation: how strongly S amplifies diffusivity.
pub const BETA_D: f32 = 1.8;

// ---------------------------------------------------------------------------
// Pedogenesis state per column
// ---------------------------------------------------------------------------

/// Soil development state for one column.
/// S ∈ [0, 1] where 0 = bare parent material, 1 = fully mature soil profile.
#[derive(Debug, Clone, Copy)]
pub struct PedogenesisState {
    /// Degree of soil development (0–1)
    pub s: f32,
    /// Current progressive rate dP/dt (yr⁻¹)
    pub dp_dt: f32,
    /// Current regressive rate dR/dt (yr⁻¹)
    pub dr_dt: f32,
    /// Lyapunov divergence accumulator (diagnostic — tracks chaos potential)
    pub lyapunov_acc: f32,
}

impl PedogenesisState {
    pub fn new(initial_s: f32) -> Self {
        Self {
            s: initial_s.clamp(0.001, 0.999),
            dp_dt: 0.0,
            dr_dt: 0.0,
            lyapunov_acc: 0.0,
        }
    }

    /// Compute instantaneous rates given current S and drivers
    pub fn compute_rates(&mut self, c1: f32, k1: f32, c2: f32, k2: f32) {
        self.dp_dt = c1 * (-k1 * self.s).exp();
        self.dr_dt = c2 * (-k2 / self.s.max(1e-4)).exp();
    }

    /// Forward-Euler integration for one timestep
    pub fn step(&mut self, params: &PedogenesisParams, env: &PedoEnv, dt_years: f64) {
        let dt = dt_years as f32;

        // Scale progressive rate by vegetation cover and nutrient availability
        let c1_eff = params.c1 * env.veg_cover * env.growth_multiplier.max(0.1);
        // Scale regressive rate by erosion intensity (from FastScape SPL)
        let c2_eff = params.c2 * (1.0 + env.erosion_intensity * 5.0);

        self.compute_rates(c1_eff, params.k1, c2_eff, params.k2);

        let ds_dt = self.dp_dt - self.dr_dt;
        self.s = (self.s + ds_dt * dt).clamp(0.001, 0.999);

        // Lyapunov accumulator: tracks local divergence rate
        // Positive when dp/dt > dr/dt (stable maturation)
        // Negative when regressive forces dominate (chaotic reset)
        self.lyapunov_acc += (self.dp_dt - self.dr_dt).abs() * dt * 0.001;
    }

    /// Erodibility K_f modulated by soil development state.
    /// High S → more cohesion → lower erodibility.
    pub fn modulated_kf(&self, kf_base: f32) -> f32 {
        kf_base * (-BETA_F * self.s).exp()
    }

    /// Diffusivity K_d modulated by soil development state.
    /// High S → heavier, water-retentive soil → higher diffusivity (creep).
    pub fn modulated_kd(&self, kd_base: f32) -> f32 {
        kd_base * (1.0 + BETA_D * self.s)
    }
}

impl Default for PedogenesisState {
    fn default() -> Self {
        Self::new(0.1) // Start as young, barely-developed soil
    }
}

// ---------------------------------------------------------------------------
// Pedogenesis parameters (column-level constants set by parent material / climate)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct PedogenesisParams {
    pub c1: f32,
    pub k1: f32,
    pub c2: f32,
    pub k2: f32,
}

impl Default for PedogenesisParams {
    fn default() -> Self {
        Self {
            c1: C1_DEFAULT,
            k1: K1_DEFAULT,
            c2: C2_DEFAULT,
            k2: K2_DEFAULT,
        }
    }
}

impl PedogenesisParams {
    /// Tropical wet parameters: fast progression but high regressive risk
    pub fn tropical() -> Self {
        Self { c1: 0.003, k1: 3.5, c2: 0.0015, k2: 0.04 }
    }

    /// Cold/arid parameters: slow progression, low regressive pressure
    pub fn cold_arid() -> Self {
        Self { c1: 0.0008, k1: 2.5, c2: 0.0005, k2: 0.08 }
    }

    /// Basaltic volcanic: fast initial development due to reactive minerals
    pub fn volcanic() -> Self {
        Self { c1: 0.004, k1: 4.0, c2: 0.001, k2: 0.03 }
    }
}

// ---------------------------------------------------------------------------
// Environmental inputs to pedogenesis (per column, per epoch)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct PedoEnv {
    /// Vegetation cover fraction (0=bare, 1=full canopy). Drives progressive rate.
    pub veg_cover: f32,
    /// Liebig growth multiplier from biogeochem (0–1). Nutrient feedback.
    pub growth_multiplier: f32,
    /// Erosion intensity from FastScape (normalized 0–1).
    /// High values boost regressive rate (landscape reset).
    pub erosion_intensity: f32,
}

impl Default for PedoEnv {
    fn default() -> Self {
        Self { veg_cover: 0.7, growth_multiplier: 0.6, erosion_intensity: 0.0 }
    }
}

// ---------------------------------------------------------------------------
// Grid-level pedogenesis solver
// ---------------------------------------------------------------------------

pub struct PedogenesisSolver {
    pub grid_width:  usize,
    pub grid_height: usize,
}

impl PedogenesisSolver {
    pub fn new(w: usize, h: usize) -> Self {
        Self { grid_width: w, grid_height: h }
    }

    /// Initialize grid states from elevation (young soil on steep terrain,
    /// mature soil on old stable lowlands).
    pub fn initialize(
        &self,
        elevations: &[f32],
        sea_level: f32,
    ) -> Vec<PedogenesisState> {
        let n = self.grid_width * self.grid_height;
        (0..n).map(|i| {
            let elev = elevations.get(i).copied().unwrap_or(0.0);
            // Lowlands (stable): higher S; highlands (active erosion): lower S
            let s = if elev <= sea_level {
                0.05
            } else {
                let norm = (elev / 2000.0).min(1.0);
                0.5 - norm * 0.35  // 0.15 at high peaks, 0.5 in lowlands
            };
            PedogenesisState::new(s)
        }).collect()
    }

    /// Step all columns in parallel.
    /// Returns arrays of modulated K_f and K_d for the next FastScape pass.
    pub fn step_epoch(
        &self,
        states:   &mut Vec<PedogenesisState>,
        params:   &[PedogenesisParams],
        envs:     &[PedoEnv],
        kf_base:  &[f32],
        kd_base:  &[f32],
        dt_years: f64,
    ) -> (Vec<f32>, Vec<f32>) {
        // Step all states in parallel
        states.par_iter_mut()
              .zip(params.par_iter())
              .zip(envs.par_iter())
              .for_each(|((state, param), env)| {
                  state.step(param, env, dt_years);
              });

        // Compute modulated K_f and K_d arrays
        let kf_mod: Vec<f32> = states.iter()
            .zip(kf_base.iter())
            .map(|(s, &kf)| s.modulated_kf(kf))
            .collect();

        let kd_mod: Vec<f32> = states.iter()
            .zip(kd_base.iter())
            .map(|(s, &kd)| s.modulated_kd(kd))
            .collect();

        (kf_mod, kd_mod)
    }

    /// Build per-column environment from terrain + biogeochem output.
    pub fn build_envs(
        &self,
        growth_multipliers: &[f32],
        delta_h: &[f32],
        cell_size_m: f32,
    ) -> Vec<PedoEnv> {
        let n = self.grid_width * self.grid_height;
        (0..n).map(|i| {
            let gm = growth_multipliers.get(i).copied().unwrap_or(0.5);
            let dh = delta_h.get(i).copied().unwrap_or(0.0);

            // Erosion intensity: |Δh| per cell relative to some reference scale
            let erosion_intensity = (dh.abs() / cell_size_m * 1000.0).min(1.0);

            // Vegetation cover scales with nutrient availability (proxy for now)
            let veg_cover = (gm * 1.2).min(1.0);

            PedoEnv { veg_cover, growth_multiplier: gm, erosion_intensity }
        }).collect()
    }

    /// Diagnostic: mean soil development index across all columns
    pub fn mean_s(&self, states: &[PedogenesisState]) -> f32 {
        let sum: f32 = states.iter().map(|s| s.s).sum();
        sum / states.len().max(1) as f32
    }

    /// Count of columns in regressive state (dr_dt > dp_dt)
    pub fn regressive_count(&self, states: &[PedogenesisState]) -> usize {
        states.iter().filter(|s| s.dr_dt > s.dp_dt).count()
    }
}

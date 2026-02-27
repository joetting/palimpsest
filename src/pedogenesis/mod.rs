//! Pedogenesis: Phillips' nonlinear progressive-regressive soil dynamics.
//!
//! Implements dS/dt = dP/dt - dR/dt where:
//!   dP/dt = c1 * exp(-k1 * S)   (progressive: self-limiting enrichment)
//!   dR/dt = c2 * exp(-k2 / (S + ε))  (regressive: erosion/degradation)
//!
//! **Solution #3**: ε-regularization prevents the dR/dt singularity at S→0.
//!   Additionally, for S < S_min, the regressive rate linearly decays to zero
//!   as a piecewise safety net. The MPRK integrator handles the actual
//!   near-zero dynamics via the Patankar trick.
//!
//! **Solution #4**: Biological contributions from the influence map (read buffer)
//!   are added to the progressive/regressive rates before integration.
//!
//! **Solution #2**: The soil state S modulates K_f and K_d on-the-fly via
//!   pure functions exposed here — no extra arrays stored.

use crate::grid::TerrainGrid;
use crate::integrators::{PdsRates, mprk22_step};

/// Parameters governing pedogenesis dynamics.
#[derive(Clone, Debug)]
pub struct PedogenesisParams {
    /// Maximum progressive rate c1 [1/yr]
    pub c1: f64,
    /// Progressive feedback coefficient k1 [dimensionless]
    pub k1: f64,
    /// Maximum regressive rate c2 [1/yr]
    pub c2: f64,
    /// Regressive feedback coefficient k2 [dimensionless]
    pub k2: f64,
    /// Regularization parameter ε to prevent singularity at S=0 (Solution #3)
    pub epsilon: f64,
    /// Minimum S below which regressive rate linearly decays (piecewise safety)
    pub s_min: f64,
    /// Coupling strength: how much erosion rate feeds into dR/dt
    pub erosion_coupling: f64,
    /// Coupling strength: how much bio-progressive influence amplifies dP/dt
    pub bio_progressive_coupling: f64,
    /// Coupling strength: how much bio-regressive influence amplifies dR/dt
    pub bio_regressive_coupling: f64,
}

impl Default for PedogenesisParams {
    fn default() -> Self {
        Self {
            c1: 0.005,
            k1: 0.1,
            c2: 0.003,
            k2: 0.5,
            epsilon: 0.01,
            s_min: 0.05,
            erosion_coupling: 1.0,
            bio_progressive_coupling: 1.0,
            bio_regressive_coupling: 1.0,
        }
    }
}

/// Compute the progressive pedogenesis rate dP/dt for a given soil state.
///
/// dP/dt = c1 * exp(-k1 * S) + bio_progressive_input
///
/// Self-limiting: as S grows, progressive rate decays exponentially.
#[inline]
fn progressive_rate(s: f64, params: &PedogenesisParams, bio_input: f64) -> f64 {
    let base = params.c1 * (-params.k1 * s).exp();
    (base + params.bio_progressive_coupling * bio_input).max(0.0)
}

/// Compute the regressive pedogenesis rate dR/dt for a given soil state.
///
/// dR/dt = c2 * exp(-k2 / (S + ε)) + erosion_contribution + bio_regressive_input
///
/// **Solution #3**: The ε term prevents divide-by-zero. For S < s_min,
/// the exponential term is linearly tapered to zero.
#[inline]
fn regressive_rate(
    s: f64,
    params: &PedogenesisParams,
    erosion_rate: f64,
    bio_input: f64,
) -> f64 {
    let exp_term = if s > params.s_min {
        // Normal regime: full exponential
        params.c2 * (-params.k2 / (s + params.epsilon)).exp()
    } else {
        // Piecewise linear taper near zero (Solution #3 safety net)
        let ratio = (s / params.s_min).max(0.0);
        let full_val = params.c2 * (-params.k2 / (params.s_min + params.epsilon)).exp();
        full_val * ratio
    };

    let erosion_contrib = params.erosion_coupling * erosion_rate.abs();
    let bio_contrib = params.bio_regressive_coupling * bio_input;

    (exp_term + erosion_contrib + bio_contrib).max(0.0)
}

/// Compute effective erodibility K_f as a pure function of soil state S.
///
/// **Solution #2**: K_f is NOT stored as an array. It is computed on-the-fly
/// during the erosion kernel. Higher soil development → lower erodibility
/// (more cohesive, root-bound soil resists fluvial incision).
///
/// K_f_eff = K_f_base / (1 + α * S)
///
/// where α controls the sensitivity of erodibility to soil development.
#[inline]
pub fn effective_erodibility(k_f_base: f64, soil_state: f64, alpha: f64) -> f64 {
    k_f_base / (1.0 + alpha * soil_state)
}

/// Compute effective diffusivity K_d as a pure function of soil state S.
///
/// **Solution #2**: K_d is NOT stored as an array. Computed on-the-fly.
/// Higher soil development → higher diffusivity (deep, weathered soil is
/// susceptible to gravitational creep and mass wasting).
///
/// K_d_eff = K_d_base * (1 + β * S)
#[inline]
pub fn effective_diffusivity(k_d_base: f64, soil_state: f64, beta: f64) -> f64 {
    k_d_base * (1.0 + beta * soil_state)
}

/// Integrate pedogenesis across the entire grid using MPRK22.
///
/// This reads:
/// - Current soil_state S
/// - Erosion rate from the geomorphology pass (drives dR/dt)
/// - Biological influence maps (read buffers, from agent sub-cycling)
/// - Temperature and precipitation (modulate rates)
///
/// And writes:
/// - Updated soil_state S
pub fn integrate_pedogenesis(
    grid: &mut TerrainGrid,
    params: &PedogenesisParams,
    dt: f64,
) {
    let rows = grid.rows;
    let cols = grid.cols;

    for r in 0..rows {
        for c in 0..cols {
            if grid.is_boundary(r, c) { continue; }

            let erosion = grid.erosion_rate[[r, c]];
            let bio_p = grid.bio_progressive_read[[r, c]];
            let bio_r = grid.bio_regressive_read[[r, c]];

            // Climate modulation: warmer + wetter = faster pedogenesis
            let temp = grid.temperature[[r, c]];
            let precip = grid.precipitation[[r, c]];
            let climate_factor = (2.0_f64.powf((temp - 15.0) / 10.0).max(0.1))
                * (precip / 1.0).min(2.0).max(0.1);

            let mut state = vec![grid.soil_state[[r, c]]];

            // Capture parameters for the closure
            let p = params.clone();
            let rate_fn = move |s: &[f64]| -> PdsRates {
                let s_val = s[0];
                let prod = progressive_rate(s_val, &p, bio_p) * climate_factor;
                let dest = regressive_rate(s_val, &p, erosion, bio_r);
                PdsRates {
                    production: vec![prod],
                    destruction: vec![dest],
                }
            };

            mprk22_step(&mut state, &rate_fn, dt);

            grid.soil_state[[r, c]] = state[0];
        }
    }
}

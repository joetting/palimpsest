//! Pedogenesis: Phillips' nonlinear progressive-regressive soil dynamics.
//!
//! **Fix #4 (Pedogenesis Singularity)**: Replaces the unphysical ε-regularized
//! exponential dR/dt = c2 * exp(-k2 / (S + ε)) with Michaelis-Menten kinetics:
//!   dR/dt = c2 * S / (S + K_m)
//! As S → 0, dR/dt → 0 smoothly and physically. No ε-hacks, no piecewise safety
//! nets. The MPRK integrator naturally tracks the state to zero (bare bedrock ridges).
//!
//! **Fix #1 integration**: Land-use regime rates replace BDI influence maps.
//! Progressive/regressive contributions come directly from the LandUseRegime enum.
//!
//! **Fix #2 integration**: Removed `effective_erodibility()`. K_f is now strictly
//! constant (lithological). The erosion module handles the Tool & Cover effect via
//! `tool_cover_factor()` using sediment thickness.

use crate::grid::{TerrainGrid, LandUseRegime};
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
    /// Half-saturation constant K_m for Michaelis-Menten depletion [dimensionless]
    /// (Fix #4: replaces epsilon + s_min)
    pub k_m: f64,
    /// Coupling strength: how much erosion rate feeds into dR/dt
    pub erosion_coupling: f64,
    /// Coupling strength for land-use regime progressive contribution
    pub regime_progressive_coupling: f64,
    /// Coupling strength for land-use regime regressive contribution
    pub regime_regressive_coupling: f64,
}

impl Default for PedogenesisParams {
    fn default() -> Self {
        Self {
            c1: 0.005,
            k1: 0.1,
            c2: 0.003,
            k_m: 0.05,    // Half-saturation: at S = K_m, dR/dt = c2/2
            erosion_coupling: 1.0,
            regime_progressive_coupling: 1.0,
            regime_regressive_coupling: 1.0,
        }
    }
}

/// Progressive pedogenesis rate dP/dt.
///
/// dP/dt = c1 * exp(-k1 * S) + regime_progressive
/// Self-limiting: as S grows, progressive rate decays exponentially.
#[inline]
fn progressive_rate(s: f64, params: &PedogenesisParams, regime_rate: f64) -> f64 {
    let base = params.c1 * (-params.k1 * s).exp();
    (base + params.regime_progressive_coupling * regime_rate).max(0.0)
}

/// Regressive pedogenesis rate dR/dt — Michaelis-Menten kinetics (Fix #4).
///
/// dR/dt = c2 * S / (S + K_m) + erosion_contribution + regime_regressive
///
/// As S → 0, dR/dt → 0 smoothly and physically. No singularity. The MPRK22
/// integrator can track the state down to zero without artificial thresholding.
#[inline]
fn regressive_rate(
    s: f64,
    params: &PedogenesisParams,
    erosion_rate: f64,
    regime_rate: f64,
) -> f64 {
    // Michaelis-Menten depletion: naturally zero at S=0
    let mm_term = params.c2 * s / (s + params.k_m);
    let erosion_contrib = params.erosion_coupling * erosion_rate.abs();
    let regime_contrib = params.regime_regressive_coupling * regime_rate;

    (mm_term + erosion_contrib + regime_contrib).max(0.0)
}

/// Tool & Cover factor for bedrock erosion (Fix #2).
///
/// Sklar & Dietrich (2004): E_b = K_f * A^m * S^n * exp(-H_sed / H_*)
/// where H_sed is the physical sediment thickness and H_* is the bedrock
/// roughness length scale.
///
/// This REPLACES the old `effective_erodibility()` which conflated bedrock
/// strength with sediment cover. K_f is now strictly constant (lithological).
#[inline]
pub fn tool_cover_factor(sediment_thickness: f64, h_star: f64) -> f64 {
    (-sediment_thickness / h_star).exp()
}

/// Compute effective diffusivity K_d as a pure function of soil state S.
/// Higher soil development → higher diffusivity (deep, weathered soil creeps more).
#[inline]
pub fn effective_diffusivity(k_d_base: f64, soil_state: f64, beta: f64) -> f64 {
    k_d_base * (1.0 + beta * soil_state)
}

/// Integrate pedogenesis across the entire grid using MPRK22.
///
/// Reads: soil_state, erosion_rate, land_use regime, temperature, precipitation
/// Writes: updated soil_state
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
            let regime = LandUseRegime::from_u8(grid.land_use[[r, c]]);
            let regime_prog = regime.progressive_rate();
            let regime_regr = regime.regressive_rate();

            // Climate modulation: warmer + wetter = faster pedogenesis
            let temp = grid.temperature[[r, c]];
            let precip = grid.precipitation[[r, c]];
            let climate_factor = (2.0_f64.powf((temp - 15.0) / 10.0).max(0.1))
                * (precip / 1.0).min(2.0).max(0.1);

            let mut state = vec![grid.soil_state[[r, c]]];

            let p = params.clone();
            let rate_fn = move |s: &[f64]| -> PdsRates {
                let s_val = s[0];
                let prod = progressive_rate(s_val, &p, regime_prog) * climate_factor;
                let dest = regressive_rate(s_val, &p, erosion, regime_regr);
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

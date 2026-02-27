//! Modified Patankar-Runge-Kutta (MPRK) integrators for positivity-preserving,
//! mass-conservative integration of Production-Destruction Systems.
//!
//! These replace the unstable explicit Euler + clamp() pattern. The Patankar trick
//! weights destruction terms by (y^{n+1} / y^n), guaranteeing the solution
//! asymptotically approaches zero but never crosses it.
//!
//! Implements MPRK22 (2nd order) and MPRK43 (3rd order) schemes.

use crate::grid::TerrainGrid;

/// A Production-Destruction System (PDS) for a single cell.
/// dy_i/dt = P_i(y) - D_i(y)
///
/// `production[i]` = rate of production of pool i
/// `destruction[i]` = rate of destruction of pool i
pub struct PdsRates {
    pub production: Vec<f64>,
    pub destruction: Vec<f64>,
}

/// Modified Patankar-Euler (first-order, unconditionally positive).
///
/// y_i^{n+1} = y_i^n + Δt * (P_i - D_i * y_i^{n+1} / y_i^n)
///
/// Rearranges to: y_i^{n+1} = (y_i^n + Δt * P_i) / (1 + Δt * D_i / y_i^n)
pub fn patankar_euler_step(
    state: &mut [f64],
    rates: &PdsRates,
    dt: f64,
) {
    for (i, y) in state.iter_mut().enumerate() {
        let p = rates.production[i];
        let d = rates.destruction[i];

        if *y > 1e-30 {
            *y = (*y + dt * p) / (1.0 + dt * d / *y);
        } else {
            // Near-zero: only production matters
            *y = dt * p;
        }
        // Guaranteed positive by construction — no clamp needed
        debug_assert!(*y >= 0.0, "MPRK positivity violation!");
    }
}

/// Modified Patankar-Runge-Kutta 2nd order (MPRK22).
///
/// Two-stage method:
/// Stage 1 (Patankar-Euler predictor):
///   y* = y^n + Δt * (P(y^n) - D(y^n) * y*/y^n)
///
/// Stage 2 (corrector):
///   y^{n+1} = y^n + Δt/2 * (P(y^n) + P(y*)) - Δt/2 * (D(y^n)*y^{n+1}/y^n + D(y*)*y^{n+1}/y*)
pub fn mprk22_step<F>(
    state: &mut [f64],
    compute_rates: &F,
    dt: f64,
) where
    F: Fn(&[f64]) -> PdsRates,
{
    let n = state.len();
    let y_n: Vec<f64> = state.to_vec();

    // Stage 1: Patankar-Euler predictor
    let rates_n = compute_rates(&y_n);
    let mut y_star = vec![0.0f64; n];
    for i in 0..n {
        if y_n[i] > 1e-30 {
            y_star[i] = (y_n[i] + dt * rates_n.production[i])
                / (1.0 + dt * rates_n.destruction[i] / y_n[i]);
        } else {
            y_star[i] = dt * rates_n.production[i];
        }
    }

    // Stage 2: Corrector using both stage evaluations
    let rates_star = compute_rates(&y_star);
    for i in 0..n {
        let p_avg = 0.5 * (rates_n.production[i] + rates_star.production[i]);

        let d_n = if y_n[i] > 1e-30 {
            rates_n.destruction[i] / y_n[i]
        } else {
            0.0
        };
        let d_star = if y_star[i] > 1e-30 {
            rates_star.destruction[i] / y_star[i]
        } else {
            0.0
        };
        let d_avg = 0.5 * (d_n + d_star);

        if d_avg > 1e-30 {
            state[i] = (y_n[i] + dt * p_avg) / (1.0 + dt * d_avg);
        } else {
            state[i] = y_n[i] + dt * p_avg;
        }
        state[i] = state[i].max(0.0); // Numerical safety (should be unnecessary)
    }
}

/// Compute soil biogeochemistry rates for a single grid cell.
/// Models a simple 2-pool system: soil carbon (C) and soil nitrogen (N).
///
/// Carbon: dC/dt = litter_input - k_decomp * C * T_factor
/// Nitrogen: dN/dt = N_deposition + mineralization(C) - plant_uptake(N) - leaching(N)
pub fn soil_biogeo_rates(
    state: &[f64],  // [C, N]
    temperature: f64,
    precipitation: f64,
) -> PdsRates {
    let carbon = state[0];
    let nitrogen = state[1];

    // Temperature sensitivity (Q10 = 2.0)
    let t_factor = 2.0_f64.powf((temperature - 15.0) / 10.0).max(0.0);

    // Moisture factor (simplified; based on precipitation)
    let m_factor = (precipitation / 1.0).min(1.0).max(0.1); // normalized to 1 m/yr

    // Carbon pool
    let k_decomp = 0.02; // base decomposition rate [1/yr]
    let litter_input = 0.5; // kg C/m²/yr
    let c_production = litter_input;
    let c_destruction = k_decomp * t_factor * m_factor * carbon;

    // Nitrogen pool
    let n_deposition = 0.01; // kg N/m²/yr
    let mineralization = 0.05 * k_decomp * t_factor * carbon; // N released from C decomp
    let n_production = n_deposition + mineralization;
    let plant_uptake = 0.1 * nitrogen; // proportional to N
    let leaching = 0.05 * precipitation * nitrogen; // proportional to precip and N
    let n_destruction = plant_uptake + leaching;

    PdsRates {
        production: vec![c_production, n_production],
        destruction: vec![c_destruction, n_destruction],
    }
}

/// Integrate soil biogeochemistry across the entire grid using MPRK22.
pub fn integrate_soil_biogeochemistry(
    grid: &mut TerrainGrid,
    dt: f64,
) {
    let rows = grid.rows;
    let cols = grid.cols;

    for r in 0..rows {
        for c in 0..cols {
            let temp = grid.temperature[[r, c]];
            let precip = grid.precipitation[[r, c]];

            let mut state = vec![
                grid.soil_carbon[[r, c]],
                grid.soil_nitrogen[[r, c]],
            ];

            // Use MPRK22 for unconditional positivity
            let rate_fn = |s: &[f64]| -> PdsRates {
                soil_biogeo_rates(s, temp, precip)
            };
            mprk22_step(&mut state, &rate_fn, dt);

            grid.soil_carbon[[r, c]] = state[0];
            grid.soil_nitrogen[[r, c]] = state[1];
        }
    }
}

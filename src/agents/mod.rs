//! Land-Use Regime Dynamics: Markov Chain transitions + Fisher-KPP population diffusion.
//!
//! **Fix #1 (Temporal Rupture)**: Replaces the BDI agent sub-cycle with macroscopic
//! land-use regime transitions evaluated directly at the dt_geomorph timescale.
//! No more "representative sub-step" multiplier extrapolating 100 years across 10,000.
//! Grid cells transition between {Pristine, Grazed, IntensiveAgriculture, Degraded}
//! using transition matrices driven by climate, soil capacity, and population pressure.
//!
//! **Fix #5 (Spatial Mismatch)**: Replaces discrete A*-style agent pathfinding with
//! Fisher-KPP reaction-diffusion for abstract population density. Population "diffuses"
//! along carrying-capacity gradients like a gas, appropriate for the 500m grid scale.

use crate::grid::{TerrainGrid, LandUseRegime};
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Parameters for the land-use regime system.
#[derive(Clone, Debug)]
pub struct LandUseParams {
    /// Intrinsic population growth rate r in Fisher-KPP: dN/dt = r*N*(1-N/K) [1/yr]
    pub growth_rate: f64,
    /// Population diffusion coefficient D [m²/yr] (Fix #5: replaces agent movement)
    pub diffusion_coeff: f64,
    /// Base carrying capacity K_0 [dimensionless] — scaled by local fertility
    pub base_carrying_capacity: f64,
    /// Population threshold for Pristine → Grazed transition
    pub threshold_graze: f64,
    /// Population threshold for Grazed → IntensiveAgriculture transition
    pub threshold_intensive: f64,
    /// Soil state threshold below which land degrades
    pub soil_degradation_threshold: f64,
    /// Recovery rate: probability of Degraded → Pristine per dt when population is low
    pub recovery_rate: f64,
    /// Catastrophic depletion: probability multiplier for sudden degradation events
    pub catastrophic_prob: f64,
    /// Initial population seed density
    pub initial_density: f64,
    /// Number of initial seed locations
    pub seed_count: usize,
}

impl Default for LandUseParams {
    fn default() -> Self {
        Self {
            growth_rate: 0.03,           // ~3% per year intrinsic growth
            diffusion_coeff: 5000.0,     // m²/yr — slow spread at 500m scale
            base_carrying_capacity: 1.0,
            threshold_graze: 0.15,
            threshold_intensive: 0.5,
            soil_degradation_threshold: 0.1,
            recovery_rate: 0.001,        // Slow natural recovery
            catastrophic_prob: 0.01,     // 1% chance per transition step of catastrophic event
            initial_density: 0.0,
            seed_count: 10,
        }
    }
}

/// Seed initial population centers on the grid.
pub fn initialize_population(grid: &mut TerrainGrid, params: &LandUseParams, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let rows = grid.rows;
    let cols = grid.cols;

    for _ in 0..params.seed_count {
        let r = rng.gen_range(rows / 4..3 * rows / 4);
        let c = rng.gen_range(cols / 4..3 * cols / 4);
        // Seed near fertile, flat, well-watered locations
        let fertility = (grid.soil_state[[r, c]] * 0.5
            + grid.precipitation[[r, c]] * 0.3
            + (1.0 - grid.slope[[r, c]].min(1.0)) * 0.2)
            .max(0.01);
        grid.population_density[[r, c]] = 0.3 * fertility;
    }
}

/// Run the full land-use dynamics for one geomorphological timestep.
///
/// This replaces `run_biological_subcycle`. Operations:
/// 1. Fisher-KPP reaction-diffusion for population density (Fix #5)
/// 2. Markov Chain regime transitions based on population + soil state (Fix #1)
///
/// Both operate directly at dt_geomorph — no sub-stepping or temporal extrapolation.
pub fn update_land_use(
    grid: &mut TerrainGrid,
    params: &LandUseParams,
    dt: f64,
    seed: u64,
    step: usize,
) {
    // === Phase 1: Fisher-KPP reaction-diffusion for population density ===
    fisher_kpp_step(grid, params, dt);

    // === Phase 2: Markov Chain regime transitions ===
    markov_regime_transitions(grid, params, dt, seed.wrapping_add(step as u64));
}

/// Fisher-KPP reaction-diffusion: dN/dt = D∇²N + r*N*(1 - N/K(x,y))
///
/// **Fix #5**: Population density diffuses outward along carrying-capacity gradients.
/// At 500m grid spacing, this represents demographic pressure spreading across
/// the landscape, not individual agents pathfinding.
///
/// Uses operator splitting: diffusion (implicit ADI) then reaction (analytical).
fn fisher_kpp_step(grid: &mut TerrainGrid, params: &LandUseParams, dt: f64) {
    let rows = grid.rows;
    let cols = grid.cols;
    let dx2 = grid.dx * grid.dx;
    let dy2 = grid.dy * grid.dy;

    // --- Diffusion: implicit ADI with carrying-capacity-weighted diffusion ---
    // D_eff(x,y) = D * K(x,y) / K_0 — population spreads faster toward fertile areas
    let d_base = params.diffusion_coeff;

    // Half-step 1: implicit in x
    let mut intermediate = grid.population_density.clone();
    for r in 1..rows - 1 {
        let mut a = vec![0.0f64; cols];
        let mut b = vec![0.0f64; cols];
        let mut c_coef = vec![0.0f64; cols];
        let mut d_rhs = vec![0.0f64; cols];

        for c_idx in 1..cols - 1 {
            let k_local = local_carrying_capacity(grid, params, r, c_idx);
            let d_eff = d_base * (k_local / params.base_carrying_capacity).max(0.01);
            let rx = d_eff * dt / (2.0 * dx2);
            let ry = d_eff * dt / (2.0 * dy2);

            a[c_idx] = -rx;
            b[c_idx] = 1.0 + 2.0 * rx;
            c_coef[c_idx] = -rx;
            d_rhs[c_idx] = grid.population_density[[r, c_idx]]
                + ry * (grid.population_density[[r + 1, c_idx]]
                    - 2.0 * grid.population_density[[r, c_idx]]
                    + grid.population_density[[r - 1, c_idx]]);
        }
        // Boundary: zero-flux (Neumann)
        b[0] = 1.0; d_rhs[0] = grid.population_density[[r, 0]];
        b[cols - 1] = 1.0; d_rhs[cols - 1] = grid.population_density[[r, cols - 1]];

        let result = thomas_solve(&a, &b, &c_coef, &d_rhs);
        for c_idx in 0..cols {
            intermediate[[r, c_idx]] = result[c_idx].max(0.0);
        }
    }

    // Half-step 2: implicit in y
    for c_idx in 1..cols - 1 {
        let mut a = vec![0.0f64; rows];
        let mut b = vec![0.0f64; rows];
        let mut c_coef = vec![0.0f64; rows];
        let mut d_rhs = vec![0.0f64; rows];

        for r in 1..rows - 1 {
            let k_local = local_carrying_capacity(grid, params, r, c_idx);
            let d_eff = d_base * (k_local / params.base_carrying_capacity).max(0.01);
            let rx = d_eff * dt / (2.0 * dx2);
            let ry = d_eff * dt / (2.0 * dy2);

            a[r] = -ry;
            b[r] = 1.0 + 2.0 * ry;
            c_coef[r] = -ry;
            d_rhs[r] = intermediate[[r, c_idx]]
                + rx * (intermediate[[r, c_idx + 1]]
                    - 2.0 * intermediate[[r, c_idx]]
                    + intermediate[[r, c_idx - 1]]);
        }
        b[0] = 1.0; d_rhs[0] = intermediate[[0, c_idx]];
        b[rows - 1] = 1.0; d_rhs[rows - 1] = intermediate[[rows - 1, c_idx]];

        let result = thomas_solve(&a, &b, &c_coef, &d_rhs);
        for r in 0..rows {
            grid.population_density[[r, c_idx]] = result[r].max(0.0);
        }
    }

    // --- Reaction: logistic growth N_new = K / (1 + ((K-N)/N)*exp(-r*dt)) ---
    // Analytical solution to dN/dt = r*N*(1-N/K) over dt
    for r in 1..rows - 1 {
        for c in 1..cols - 1 {
            let n = grid.population_density[[r, c]];
            if n < 1e-12 { continue; }

            let k = local_carrying_capacity(grid, params, r, c);
            if k < 1e-12 {
                // Uninhabitable: population decays
                grid.population_density[[r, c]] = n * (-0.1 * dt).exp();
                continue;
            }

            // Analytical logistic solution
            let ratio = (k - n) / n;
            if ratio > 0.0 {
                grid.population_density[[r, c]] =
                    k / (1.0 + ratio * (-params.growth_rate * dt).exp());
            } else {
                // Over carrying capacity: decay toward K
                grid.population_density[[r, c]] =
                    k / (1.0 + ratio * (-params.growth_rate * dt).exp());
            }
            grid.population_density[[r, c]] = grid.population_density[[r, c]].max(0.0);
        }
    }
}

/// Local carrying capacity K(x,y) — a function of soil state, precipitation, slope.
/// This replaces the agent's "fertility perception" with a proper grid-scale metric.
#[inline]
fn local_carrying_capacity(grid: &TerrainGrid, params: &LandUseParams, r: usize, c: usize) -> f64 {
    let soil = grid.soil_state[[r, c]].max(0.0);
    let precip = grid.precipitation[[r, c]].max(0.0);
    let slope = grid.slope[[r, c]];

    // Steep slopes reduce capacity; fertile, wet land increases it
    let slope_penalty = (1.0 - (slope / 0.5).min(1.0)).max(0.0);
    let moisture_factor = (precip / 0.5).min(2.0);
    let soil_factor = soil / (soil + 0.2); // Michaelis-Menten-style saturation

    params.base_carrying_capacity * soil_factor * moisture_factor * slope_penalty
}

/// Markov Chain regime transitions evaluated at the geomorphological timescale.
///
/// **Fix #1**: Instead of running thousands of agent sub-steps, evaluate state
/// transitions directly using transition probabilities that depend on:
/// - Local population density (from Fisher-KPP)
/// - Current soil state
/// - Climate conditions
///
/// Catastrophic degradation is applied as a punctuated event, not a smeared average.
fn markov_regime_transitions(
    grid: &mut TerrainGrid,
    params: &LandUseParams,
    dt: f64,
    seed: u64,
) {
    let rows = grid.rows;
    let cols = grid.cols;
    let mut rng = StdRng::seed_from_u64(seed);

    for r in 1..rows - 1 {
        for c in 1..cols - 1 {
            let pop = grid.population_density[[r, c]];
            let soil = grid.soil_state[[r, c]];
            let current = LandUseRegime::from_u8(grid.land_use[[r, c]]);

            let new_regime = match current {
                LandUseRegime::Pristine => {
                    if pop > params.threshold_graze {
                        // Population pressure triggers grazing
                        LandUseRegime::Grazed
                    } else {
                        LandUseRegime::Pristine
                    }
                }
                LandUseRegime::Grazed => {
                    if pop > params.threshold_intensive {
                        LandUseRegime::IntensiveAgriculture
                    } else if pop < params.threshold_graze * 0.3 {
                        // Population collapse → recovery toward pristine
                        let recovery_prob = 1.0 - (-params.recovery_rate * dt).exp();
                        if rng.gen::<f64>() < recovery_prob {
                            LandUseRegime::Pristine
                        } else {
                            LandUseRegime::Grazed
                        }
                    } else {
                        LandUseRegime::Grazed
                    }
                }
                LandUseRegime::IntensiveAgriculture => {
                    if soil < params.soil_degradation_threshold {
                        // Soil collapse → degradation (catastrophic punctuated event)
                        LandUseRegime::Degraded
                    } else if rng.gen::<f64>() < params.catastrophic_prob * dt {
                        // Stochastic catastrophic event (drought, pest, war)
                        // Apply as sudden depletion step
                        grid.soil_state[[r, c]] *= 0.3; // Catastrophic soil loss
                        LandUseRegime::Degraded
                    } else if pop < params.threshold_graze * 0.5 {
                        // Abandonment
                        let recovery_prob = 1.0 - (-params.recovery_rate * 0.5 * dt).exp();
                        if rng.gen::<f64>() < recovery_prob {
                            LandUseRegime::Grazed
                        } else {
                            LandUseRegime::IntensiveAgriculture
                        }
                    } else {
                        LandUseRegime::IntensiveAgriculture
                    }
                }
                LandUseRegime::Degraded => {
                    if pop < params.threshold_graze * 0.1 && soil > params.soil_degradation_threshold * 2.0 {
                        // Slow natural recovery (80-260 years for forest regrowth)
                        let recovery_prob = 1.0 - (-params.recovery_rate * 0.3 * dt).exp();
                        if rng.gen::<f64>() < recovery_prob {
                            LandUseRegime::Pristine
                        } else {
                            LandUseRegime::Degraded
                        }
                    } else {
                        LandUseRegime::Degraded
                    }
                }
            };

            grid.land_use[[r, c]] = new_regime as u8;
        }
    }
}

/// Thomas algorithm for tridiagonal systems (shared with erosion module).
fn thomas_solve(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = d.len();
    let mut cp = vec![0.0f64; n];
    let mut dp = vec![0.0f64; n];

    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];

    for i in 1..n {
        let m = b[i] - a[i] * cp[i - 1];
        if m.abs() < 1e-30 {
            cp[i] = 0.0;
            dp[i] = 0.0;
        } else {
            cp[i] = if i < n - 1 { c[i] / m } else { 0.0 };
            dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
        }
    }

    let mut x = vec![0.0f64; n];
    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    x
}

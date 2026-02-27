//! Erosion solvers with soil-state-coupled erodibility and diffusivity.
//!
//! **Solution #2 (On-the-fly K_f)**: Erodibility is computed as a pure function
//! of soil_state S at each cell during the erosion kernel. No extra K_f array.
//!   K_f_eff = K_f_base / (1 + α * S)
//!
//! **Solution #1 (Operator-Split Variable K_d)**: The implicit ADI solver uses
//! a constant K_d_base for unconditional stability. The soil-induced variance
//! ΔK_d(S) is applied as an explicit correction after the implicit step.
//!   K_d_eff = K_d_base * (1 + β * S)
//!   Explicit correction: Δh = dt * ΔK_d(S) * ∇²h

use crate::grid::TerrainGrid;
use crate::flow::FlowRoutingResult;
use crate::pedogenesis;

/// Parameters for the Stream Power Law erosion model.
#[derive(Clone, Debug)]
pub struct StreamPowerParams {
    /// Bedrock erodibility coefficient K_f [m^(1-2m) yr^(-1)]
    pub k_f: f64,
    /// Sediment erodibility (typically 5-10× bedrock) [same units]
    pub k_f_sed: f64,
    /// Drainage area (discharge) exponent m
    pub m: f64,
    /// Slope exponent n
    pub n: f64,
    /// Tectonic uplift rate U [m/yr]
    pub uplift_rate: f64,
    /// Soil-state sensitivity for erodibility: K_f / (1 + alpha * S)
    pub alpha_erodibility: f64,
}

impl Default for StreamPowerParams {
    fn default() -> Self {
        Self {
            k_f: 1e-5,
            k_f_sed: 5e-5,
            m: 0.5,
            n: 1.0,
            uplift_rate: 1e-3,
            alpha_erodibility: 0.5,
        }
    }
}

/// Parameters for the ξ-q under-capacity sediment transport model.
#[derive(Clone, Debug)]
pub struct XiQParams {
    pub k_f: f64,
    pub k_f_sed: f64,
    pub m: f64,
    pub n: f64,
    pub v_s: f64,
    pub d_star: f64,
    pub uplift_rate: f64,
    pub alpha_erodibility: f64,
}

impl Default for XiQParams {
    fn default() -> Self {
        Self {
            k_f: 1e-5,
            k_f_sed: 5e-5,
            m: 0.5,
            n: 1.0,
            v_s: 1.0,
            d_star: 1.0,
            uplift_rate: 1e-3,
            alpha_erodibility: 0.5,
        }
    }
}

/// Implicit O(n) SPL solver with soil-state-modulated erodibility (Solution #2).
pub fn implicit_spl_erode(
    grid: &mut TerrainGrid,
    routing: &FlowRoutingResult,
    params: &StreamPowerParams,
    dt: f64,
) {
    let n_nodes = grid.len();
    let dx = grid.dx;
    let cell_area = grid.dx * grid.dy;

    let mut elev: Vec<f64> = grid.elevation.iter().copied().collect();
    let mut bedrock: Vec<f64> = grid.bedrock.iter().copied().collect();
    let mut sediment: Vec<f64> = grid.sediment.iter().copied().collect();
    let discharge: Vec<f64> = grid.discharge.iter().copied().collect();
    let soil_state: Vec<f64> = grid.soil_state.iter().copied().collect();

    // Nutrient tracking
    let mut c_conc: Vec<f64> = vec![0.0; n_nodes];
    let mut n_conc: Vec<f64> = vec![0.0; n_nodes];
    let mut c_flux: Vec<f64> = vec![0.0; n_nodes];
    let mut n_flux: Vec<f64> = vec![0.0; n_nodes];

    for i in 0..n_nodes {
        let (r, c) = grid.grid_index(i);
        let sed_thick = sediment[i].max(1e-10);
        c_conc[i] = grid.soil_carbon[[r, c]] / sed_thick;
        n_conc[i] = grid.soil_nitrogen[[r, c]] / sed_thick;
    }

    // Apply uplift to interior nodes
    for i in 0..n_nodes {
        let (r, c) = grid.grid_index(i);
        if !grid.is_boundary(r, c) {
            bedrock[i] += dt * params.uplift_rate;
            elev[i] = bedrock[i] + sediment[i];
        }
    }

    // Process nodes in stack order
    if (params.n - 1.0).abs() < 1e-10 {
        // Linear case
        for &node in routing.stack.iter() {
            let recv = routing.receivers[node];
            if recv == node { continue; }

            let q_m = discharge[node].powf(params.m);
            let h_old = elev[node];
            let h_recv = elev[recv];

            // **Solution #2**: Compute K_f on-the-fly from soil state
            let k_eff = if sediment[node] > 0.01 {
                pedogenesis::effective_erodibility(
                    params.k_f_sed, soil_state[node], params.alpha_erodibility)
            } else {
                pedogenesis::effective_erodibility(
                    params.k_f, soil_state[node], params.alpha_erodibility)
            };

            let factor = k_eff * q_m / dx;
            let h_new = (h_old + dt * factor * h_recv) / (1.0 + dt * factor);
            let h_new = h_new.max(h_recv);

            let dh = h_old - h_new;

            if dh > 0.0 {
                let sed_eroded = dh.min(sediment[node]);
                let rock_eroded = (dh - sed_eroded).max(0.0);
                sediment[node] -= sed_eroded;
                bedrock[node] -= rock_eroded;

                let c_mobilized = sed_eroded * c_conc[node];
                let n_mobilized = sed_eroded * n_conc[node];

                let frac = routing.flow_fraction[node];
                c_flux[recv] += c_mobilized * frac * cell_area;
                n_flux[recv] += n_mobilized * frac * cell_area;
                let recv2 = routing.receivers_secondary[node];
                if recv2 != node && recv2 != recv {
                    c_flux[recv2] += c_mobilized * (1.0 - frac) * cell_area;
                    n_flux[recv2] += n_mobilized * (1.0 - frac) * cell_area;
                }
            }

            elev[node] = bedrock[node] + sediment[node];
            let (r, c) = grid.grid_index(node);
            grid.erosion_rate[[r, c]] = dh / dt;
        }
    } else {
        // Nonlinear case: adaptive Newton-Raphson
        for &node in routing.stack.iter() {
            let recv = routing.receivers[node];
            if recv == node { continue; }

            let q_m = discharge[node].powf(params.m);
            let h_recv = elev[recv];
            let h_old = elev[node];

            let k_eff = if sediment[node] > 0.01 {
                pedogenesis::effective_erodibility(
                    params.k_f_sed, soil_state[node], params.alpha_erodibility)
            } else {
                pedogenesis::effective_erodibility(
                    params.k_f, soil_state[node], params.alpha_erodibility)
            };

            let mut h = h_old;
            let mut converged = false;

            for iter in 0..50 {
                let slope = ((h - h_recv) / dx).max(0.0);
                let s_n = slope.powf(params.n);
                let f = h - h_old + dt * k_eff * q_m * s_n;
                let df = 1.0 + dt * k_eff * q_m * params.n
                    * slope.powf(params.n - 1.0) / dx;
                let dh = f / df;
                h -= dh;

                if dh.abs() < 1e-12 {
                    converged = true;
                    break;
                }

                if iter > 10 && dh.abs() > 0.5 * (h_old - h_recv) {
                    h = 0.5 * (h + h_old);
                }
            }

            if !converged {
                let factor = k_eff * q_m / dx;
                h = (h_old + dt * factor * h_recv) / (1.0 + dt * factor);
            }

            h = h.max(h_recv);
            let dh = h_old - h;

            if dh > 0.0 {
                let sed_eroded = dh.min(sediment[node]);
                let rock_eroded = (dh - sed_eroded).max(0.0);
                sediment[node] -= sed_eroded;
                bedrock[node] -= rock_eroded;

                let frac = routing.flow_fraction[node];
                c_flux[recv] += sed_eroded * c_conc[node] * frac * cell_area;
                n_flux[recv] += sed_eroded * n_conc[node] * frac * cell_area;
            }

            elev[node] = bedrock[node] + sediment[node];
        }
    }

    // Write back
    for i in 0..n_nodes {
        let (r, c) = grid.grid_index(i);
        grid.elevation[[r, c]] = elev[i];
        grid.bedrock[[r, c]] = bedrock[i];
        grid.sediment[[r, c]] = sediment[i];

        let deposited_c = c_flux[i] / cell_area;
        let deposited_n = n_flux[i] / cell_area;
        grid.soil_carbon[[r, c]] += deposited_c;
        grid.soil_nitrogen[[r, c]] += deposited_n;

        if grid.erosion_rate[[r, c]] > 0.0 {
            let eroded_frac = (grid.erosion_rate[[r, c]] * dt / sediment[i].max(0.01)).min(1.0);
            grid.soil_carbon[[r, c]] *= (1.0 - eroded_frac).max(0.0);
            grid.soil_nitrogen[[r, c]] *= (1.0 - eroded_frac).max(0.0);
        }
    }
}

/// ξ-q under-capacity erosion-deposition solver with soil-state coupling.
pub fn implicit_xi_q_erode(
    grid: &mut TerrainGrid,
    routing: &FlowRoutingResult,
    params: &XiQParams,
    dt: f64,
) {
    let n_nodes = grid.len();
    let dx = grid.dx;
    let cell_area = grid.dx * grid.dy;

    let mut elev: Vec<f64> = grid.elevation.iter().copied().collect();
    let mut bedrock: Vec<f64> = grid.bedrock.iter().copied().collect();
    let mut sediment: Vec<f64> = grid.sediment.iter().copied().collect();
    let discharge: Vec<f64> = grid.discharge.iter().copied().collect();
    let soil_state: Vec<f64> = grid.soil_state.iter().copied().collect();
    let mut sed_flux = vec![0.0f64; n_nodes];

    for i in 0..n_nodes {
        let (r, c) = grid.grid_index(i);
        if !grid.is_boundary(r, c) {
            bedrock[i] += dt * params.uplift_rate;
            elev[i] = bedrock[i] + sediment[i];
        }
    }

    for &node in routing.stack.iter() {
        let recv = routing.receivers[node];
        if recv == node { continue; }

        let q_w = discharge[node];
        let q_m = q_w.powf(params.m);
        let xi = ((q_w * params.d_star) / params.v_s).max(1e-10);
        let incoming_sed = sed_flux[node];

        // Solution #2: on-the-fly K_f
        let k_eff = if sediment[node] > 0.01 {
            pedogenesis::effective_erodibility(
                params.k_f_sed, soil_state[node], params.alpha_erodibility)
        } else {
            pedogenesis::effective_erodibility(
                params.k_f, soil_state[node], params.alpha_erodibility)
        };

        if (params.n - 1.0).abs() < 1e-10 {
            let h_recv = elev[recv];
            let erosion_coeff = k_eff * q_m / dx;
            let depo_coeff = 1.0 / xi;

            let h_new = (elev[node] + dt * (erosion_coeff * h_recv + incoming_sed / xi))
                / (1.0 + dt * (erosion_coeff + depo_coeff));

            let slope = ((h_new - h_recv) / dx).max(0.0);
            let bedrock_erosion = k_eff * q_m * slope;
            let deposition = incoming_sed / xi;

            let dh = elev[node] - h_new;
            if dh > 0.0 {
                let sed_eroded = dh.min(sediment[node]);
                sediment[node] -= sed_eroded;
                bedrock[node] -= (dh - sed_eroded).max(0.0);
            } else {
                sediment[node] += (-dh).max(0.0);
            }

            let outgoing_sed = (incoming_sed + bedrock_erosion * cell_area
                - deposition * cell_area).max(0.0);

            let frac = routing.flow_fraction[node];
            sed_flux[recv] += outgoing_sed * frac;
            let recv2 = routing.receivers_secondary[node];
            if recv2 != node && recv2 != recv {
                sed_flux[recv2] += outgoing_sed * (1.0 - frac);
            }

            elev[node] = bedrock[node] + sediment[node];
            let (r, c) = grid.grid_index(node);
            grid.erosion_rate[[r, c]] = dh / dt;
        } else {
            let h_recv = elev[recv];
            let mut h = elev[node];
            let mut converged = false;

            for iter in 0..50 {
                let slope = ((h - h_recv) / dx).max(1e-30);
                let s_n = slope.powf(params.n);
                let erosion = k_eff * q_m * s_n;
                let deposition = incoming_sed / xi;

                let f = h - elev[node] + dt * (erosion - deposition);
                let df = 1.0 + dt * k_eff * q_m * params.n
                    * slope.powf(params.n - 1.0) / dx;
                let dh = f / df;
                h -= dh;

                if dh.abs() < 1e-12 {
                    converged = true;
                    break;
                }
                if iter > 10 && dh.abs() > 0.5 * (elev[node] - h_recv).abs() {
                    h = 0.5 * (h + elev[node]);
                }
            }

            if !converged {
                let factor = k_eff * q_m / dx;
                h = (elev[node] + dt * (factor * h_recv + incoming_sed / xi))
                    / (1.0 + dt * (factor + 1.0 / xi));
            }

            h = h.max(h_recv);
            let total_dh = elev[node] - h;
            if total_dh > 0.0 {
                let sed_eroded = total_dh.min(sediment[node]);
                sediment[node] -= sed_eroded;
                bedrock[node] -= (total_dh - sed_eroded).max(0.0);
            } else {
                sediment[node] += (-total_dh).max(0.0);
            }

            let slope = ((bedrock[node] + sediment[node] - h_recv) / dx).max(0.0);
            let outgoing = (incoming_sed + k_eff * q_m * slope.powf(params.n) * cell_area
                - (incoming_sed / xi) * cell_area).max(0.0);

            let frac = routing.flow_fraction[node];
            sed_flux[recv] += outgoing * frac;
            let recv2 = routing.receivers_secondary[node];
            if recv2 != node && recv2 != recv {
                sed_flux[recv2] += outgoing * (1.0 - frac);
            }

            elev[node] = bedrock[node] + sediment[node];
        }
    }

    for (i, _) in elev.iter().enumerate() {
        let (r, c) = grid.grid_index(i);
        grid.elevation[[r, c]] = bedrock[i] + sediment[i];
        grid.bedrock[[r, c]] = bedrock[i];
        grid.sediment[[r, c]] = sediment[i];
        grid.sediment_flux[[r, c]] = sed_flux[i];
    }
}

/// Hillslope diffusion parameters.
#[derive(Clone, Debug)]
pub struct DiffusionParams {
    /// Base linear diffusivity K_d [m²/yr]
    pub k_d: f64,
    /// Critical slope S_c for nonlinear diffusion [m/m] (Roering et al., 1999)
    pub s_c: f64,
    /// Soil-state sensitivity for diffusivity: K_d * (1 + beta * S) (Solution #1)
    pub beta_diffusivity: f64,
}

impl Default for DiffusionParams {
    fn default() -> Self {
        Self {
            k_d: 0.01,
            s_c: f64::INFINITY,
            beta_diffusivity: 0.3,
        }
    }
}

/// Hillslope diffusion with operator-split variable K_d (Solution #1).
///
/// Two-phase approach:
/// 1. Implicit ADI with constant K_d_base → unconditionally stable
/// 2. Explicit correction for ΔK_d(S) = K_d_base * β * S → soil-modulated creep
///
/// The explicit correction is stable as long as β*S is moderate relative to the
/// base (physically reasonable: soil development modulates but doesn't dominate).
pub fn hillslope_diffusion(
    grid: &mut TerrainGrid,
    params: &DiffusionParams,
    dt: f64,
) {
    if params.s_c.is_finite() {
        nonlinear_hillslope_diffusion(grid, params, dt);
    } else {
        // Phase 1: Implicit ADI with constant K_d_base
        implicit_adi_diffusion(grid, params.k_d, dt);

        // Phase 2: Explicit correction for soil-induced variance (Solution #1)
        if params.beta_diffusivity > 0.0 {
            explicit_variable_kd_correction(grid, params, dt);
        }
    }
}

/// Implicit ADI linear diffusion with constant K_d (unconditionally stable).
fn implicit_adi_diffusion(grid: &mut TerrainGrid, k_d: f64, dt: f64) {
    let rows = grid.rows;
    let cols = grid.cols;
    let dx2 = grid.dx * grid.dx;
    let dy2 = grid.dy * grid.dy;

    let rx = k_d * dt / (2.0 * dx2);
    let ry = k_d * dt / (2.0 * dy2);

    // Half-step 1: implicit in x-direction
    let mut intermediate = grid.elevation.clone();
    for r in 1..rows - 1 {
        let mut a_coef = vec![0.0f64; cols];
        let mut b_coef = vec![0.0f64; cols];
        let mut c_coef = vec![0.0f64; cols];
        let mut d_rhs  = vec![0.0f64; cols];

        for c in 1..cols - 1 {
            a_coef[c] = -rx;
            b_coef[c] = 1.0 + 2.0 * rx;
            c_coef[c] = -rx;
            d_rhs[c] = grid.elevation[[r, c]]
                + ry * (grid.elevation[[r + 1, c]] - 2.0 * grid.elevation[[r, c]] + grid.elevation[[r - 1, c]]);
        }
        b_coef[0] = 1.0;
        d_rhs[0] = grid.elevation[[r, 0]];
        b_coef[cols - 1] = 1.0;
        d_rhs[cols - 1] = grid.elevation[[r, cols - 1]];

        let result = thomas_solve(&a_coef, &b_coef, &c_coef, &d_rhs);
        for c in 0..cols {
            intermediate[[r, c]] = result[c];
        }
    }

    // Half-step 2: implicit in y-direction
    for c in 1..cols - 1 {
        let mut a_coef = vec![0.0f64; rows];
        let mut b_coef = vec![0.0f64; rows];
        let mut c_coef_arr = vec![0.0f64; rows];
        let mut d_rhs  = vec![0.0f64; rows];

        for r in 1..rows - 1 {
            a_coef[r] = -ry;
            b_coef[r] = 1.0 + 2.0 * ry;
            c_coef_arr[r] = -ry;
            d_rhs[r] = intermediate[[r, c]]
                + rx * (intermediate[[r, c + 1]] - 2.0 * intermediate[[r, c]] + intermediate[[r, c - 1]]);
        }
        b_coef[0] = 1.0;
        d_rhs[0] = intermediate[[0, c]];
        b_coef[rows - 1] = 1.0;
        d_rhs[rows - 1] = intermediate[[rows - 1, c]];

        let result = thomas_solve(&a_coef, &b_coef, &c_coef_arr, &d_rhs);
        for r in 0..rows {
            grid.elevation[[r, c]] = result[r];
        }
    }

    update_sediment_from_diffusion(grid);
}

/// Explicit correction for spatially variable K_d modulated by soil state (Solution #1).
///
/// ΔK_d(r,c) = K_d_base * β * S(r,c)
/// Δh(r,c) = dt * ΔK_d(r,c) * ∇²h(r,c)
///
/// This is applied AFTER the stable implicit ADI pass. The correction is small
/// relative to the base diffusion (β*S << 1 in most cells), so explicit treatment
/// is numerically safe. For extreme cases, adaptive sub-stepping is used.
fn explicit_variable_kd_correction(
    grid: &mut TerrainGrid,
    params: &DiffusionParams,
    dt: f64,
) {
    let rows = grid.rows;
    let cols = grid.cols;
    let dx2 = grid.dx * grid.dx;
    let dy2 = grid.dy * grid.dy;

    // Estimate max effective ΔK_d for CFL check
    let max_s = grid.soil_state.iter().cloned().fold(0.0f64, f64::max);
    let max_delta_kd = params.k_d * params.beta_diffusivity * max_s;
    let cfl_dt = 0.25 * dx2.min(dy2) / max_delta_kd.max(1e-30);
    let n_sub = ((dt / cfl_dt).ceil() as usize).max(1).min(100);
    let sub_dt = dt / n_sub as f64;

    for _ in 0..n_sub {
        let elev_snapshot = grid.elevation.clone();

        for r in 1..rows - 1 {
            for c in 1..cols - 1 {
                let s = grid.soil_state[[r, c]];
                let delta_kd = params.k_d * params.beta_diffusivity * s;

                if delta_kd < 1e-30 { continue; }

                let laplacian =
                    (elev_snapshot[[r + 1, c]] + elev_snapshot[[r - 1, c]]
                        - 2.0 * elev_snapshot[[r, c]]) / dy2
                    + (elev_snapshot[[r, c + 1]] + elev_snapshot[[r, c - 1]]
                        - 2.0 * elev_snapshot[[r, c]]) / dx2;

                grid.elevation[[r, c]] += sub_dt * delta_kd * laplacian;
            }
        }
    }

    update_sediment_from_diffusion(grid);
}

/// Nonlinear Roering et al. (1999) hillslope diffusion (explicit, sub-stepped).
fn nonlinear_hillslope_diffusion(grid: &mut TerrainGrid, params: &DiffusionParams, dt: f64) {
    let rows = grid.rows;
    let cols = grid.cols;
    let dx = grid.dx;
    let dy = grid.dy;
    let k_d = params.k_d;
    let s_c = params.s_c;
    let s_c2 = s_c * s_c;

    let max_slope = grid.slope.iter().cloned().fold(0.0f64, f64::max);
    let ratio = (max_slope / s_c).min(0.99);
    let k_eff_max = k_d / (1.0 - ratio * ratio).powi(2);
    let max_dt = 0.2 * (dx * dx).min(dy * dy) / k_eff_max.max(1e-30);
    let n_substeps = ((dt / max_dt).ceil() as usize).max(1);
    let sub_dt = dt / n_substeps as f64;

    for _ in 0..n_substeps {
        let elev_old = grid.elevation.clone();
        for r in 1..rows - 1 {
            for c in 1..cols - 1 {
                let sx = (elev_old[[r, c + 1]] - elev_old[[r, c - 1]]) / (2.0 * dx);
                let sy = (elev_old[[r + 1, c]] - elev_old[[r - 1, c]]) / (2.0 * dy);
                let s_mag = (sx * sx + sy * sy).sqrt();

                // Include soil-state modulation in nonlinear diffusivity too
                let soil_s = grid.soil_state[[r, c]];
                let base_k = pedogenesis::effective_diffusivity(k_d, soil_s, params.beta_diffusivity);

                let ratio_sq = (s_mag * s_mag / s_c2).min(0.99);
                let k_eff = base_k / (1.0 - ratio_sq);

                let laplacian =
                    (elev_old[[r + 1, c]] + elev_old[[r - 1, c]] - 2.0 * elev_old[[r, c]]) / (dy * dy)
                    + (elev_old[[r, c + 1]] + elev_old[[r, c - 1]] - 2.0 * elev_old[[r, c]]) / (dx * dx);

                grid.elevation[[r, c]] += sub_dt * k_eff * laplacian;
            }
        }
    }

    update_sediment_from_diffusion(grid);
}

/// Update sediment layer after diffusion has modified elevation.
fn update_sediment_from_diffusion(grid: &mut TerrainGrid) {
    for r in 0..grid.rows {
        for c in 0..grid.cols {
            let new_elev = grid.elevation[[r, c]];
            let old_surface = grid.bedrock[[r, c]] + grid.sediment[[r, c]];
            let dh = new_elev - old_surface;
            if dh > 0.0 {
                grid.sediment[[r, c]] += dh;
            } else {
                let sed_removed = (-dh).min(grid.sediment[[r, c]]);
                grid.sediment[[r, c]] -= sed_removed;
            }
            grid.elevation[[r, c]] = grid.bedrock[[r, c]] + grid.sediment[[r, c]];
        }
    }
}

/// Thomas algorithm for solving tridiagonal systems.
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

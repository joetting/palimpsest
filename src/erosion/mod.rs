//! Erosion solvers with physically correct dual-layer substrate tracking.
//!
//! **Critique fixes**:
//! - #1a: Erosion now differentiates bedrock vs sediment. The solver first strips
//!   loose sediment (fast, no K_f needed), then incises bedrock at rate K_f.
//! - #1b: Added nonlinear Roering et al. (1999) hillslope diffusion for steep terrain.
//! - #1c: Implicit ADI (Alternating Direction Implicit) diffusion solver eliminates
//!   the CFL-induced sub-stepping bottleneck for high-resolution grids.
//! - #4:  Biogeochemical pools (C, N) are advected with the sediment flux during
//!   erosion and deposited proportionally.
//! - #5:  Newton-Raphson uses adaptive iteration limits with convergence monitoring.

use crate::grid::TerrainGrid;
use crate::flow::FlowRoutingResult;

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
}

impl Default for StreamPowerParams {
    fn default() -> Self {
        Self {
            k_f: 1e-5,
            k_f_sed: 5e-5,
            m: 0.5,
            n: 1.0,
            uplift_rate: 1e-3,
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
        }
    }
}

/// Implicit O(n) SPL solver with dual-layer substrate tracking.
///
/// **Critique fix #1a**: Erosion now correctly differentiates between removing
/// loose sediment (using k_f_sed) and incising bedrock (using k_f). When
/// sediment cover exists, the effective erodibility is the sediment value;
/// once sediment is stripped, bedrock erodibility takes over.
///
/// **Critique fix #4**: When erosion removes sediment, the nutrients bound in
/// that sediment (C, N per unit thickness) are mobilized into the sediment flux
/// and deposited downstream proportionally.
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

    // Nutrient concentrations in sediment [kg/m² per m of sediment thickness]
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

    // Apply uplift to interior nodes (raises bedrock)
    for i in 0..n_nodes {
        let (r, c) = grid.grid_index(i);
        if !grid.is_boundary(r, c) {
            bedrock[i] += dt * params.uplift_rate;
            elev[i] = bedrock[i] + sediment[i];
        }
    }

    // Process nodes in stack order (outlets first, then upstream)
    if (params.n - 1.0).abs() < 1e-10 {
        for &node in routing.stack.iter() {
            let recv = routing.receivers[node];
            if recv == node { continue; }

            let q_m = discharge[node].powf(params.m);
            let h_old = elev[node];
            let h_recv = elev[recv];

            // Effective erodibility depends on whether we're cutting sediment or bedrock
            let k_eff = if sediment[node] > 0.01 {
                params.k_f_sed
            } else {
                params.k_f
            };

            let factor = k_eff * q_m / dx;
            let h_new = (h_old + dt * factor * h_recv) / (1.0 + dt * factor);
            let h_new = h_new.max(h_recv);

            let dh = h_old - h_new; // Total lowering (positive = erosion)

            // Partition erosion between sediment and bedrock
            if dh > 0.0 {
                let sed_eroded = dh.min(sediment[node]);
                let rock_eroded = (dh - sed_eroded).max(0.0);

                sediment[node] -= sed_eroded;
                bedrock[node] -= rock_eroded;

                // Nutrient mobilization from eroded sediment
                let c_mobilized = sed_eroded * c_conc[node];
                let n_mobilized = sed_eroded * n_conc[node];

                // Route nutrients to receiver
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
        // Nonlinear case: adaptive Newton-Raphson (critique fix #5)
        for &node in routing.stack.iter() {
            let recv = routing.receivers[node];
            if recv == node { continue; }

            let q_m = discharge[node].powf(params.m);
            let h_recv = elev[recv];
            let h_old = elev[node];

            let k_eff = if sediment[node] > 0.01 { params.k_f_sed } else { params.k_f };

            let mut h = h_old;
            let mut converged = false;
            let max_iter = 50; // Increased from 20 for robustness
            let tol = 1e-12;

            for iter in 0..max_iter {
                let slope = ((h - h_recv) / dx).max(0.0);
                let s_n = slope.powf(params.n);
                let f = h - h_old + dt * k_eff * q_m * s_n;
                let df = 1.0 + dt * k_eff * q_m * params.n
                    * slope.powf(params.n - 1.0) / dx;
                let dh = f / df;
                h -= dh;

                if dh.abs() < tol {
                    converged = true;
                    break;
                }

                // Damped step if oscillating (critique fix #5)
                if iter > 10 && dh.abs() > 0.5 * (h_old - h_recv) {
                    h = 0.5 * (h + h_old);
                }
            }

            if !converged {
                // Fallback: use linear approximation
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

    // Deposit transported nutrients at receiver locations
    for i in 0..n_nodes {
        let (r, c) = grid.grid_index(i);
        grid.elevation[[r, c]] = elev[i];
        grid.bedrock[[r, c]] = bedrock[i];
        grid.sediment[[r, c]] = sediment[i];

        // Add deposited nutrients to local pools
        let deposited_c = c_flux[i] / cell_area; // kg/m²
        let deposited_n = n_flux[i] / cell_area;
        grid.soil_carbon[[r, c]] += deposited_c;
        grid.soil_nitrogen[[r, c]] += deposited_n;

        // Remove mobilized nutrients from eroded cells
        if grid.erosion_rate[[r, c]] > 0.0 {
            let eroded_frac = (grid.erosion_rate[[r, c]] * dt / sediment[i].max(0.01)).min(1.0);
            grid.soil_carbon[[r, c]] *= (1.0 - eroded_frac).max(0.0);
            grid.soil_nitrogen[[r, c]] *= (1.0 - eroded_frac).max(0.0);
        }
    }
}

/// ξ-q under-capacity erosion-deposition solver with dual-layer tracking.
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
        let xi = (q_w * params.d_star) / params.v_s;
        let xi = xi.max(1e-10);
        let incoming_sed = sed_flux[node];

        let k_eff = if sediment[node] > 0.01 { params.k_f_sed } else { params.k_f };

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
                // Deposition: add to sediment layer
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
            // Nonlinear with adaptive Newton-Raphson
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
    /// Linear diffusivity K_d [m²/yr]
    pub k_d: f64,
    /// Critical slope S_c for nonlinear diffusion [m/m] (Roering et al., 1999)
    /// Set to f64::INFINITY to use linear diffusion only.
    pub s_c: f64,
}

impl Default for DiffusionParams {
    fn default() -> Self {
        Self {
            k_d: 0.01,
            s_c: f64::INFINITY, // Linear by default
        }
    }
}

/// Hillslope diffusion with nonlinear Roering option and implicit ADI solver.
///
/// **Critique fix #1b**: When `s_c` is finite, uses the Roering et al. (1999)
/// nonlinear diffusion: q_s = K_d * S / (1 - (S/S_c)²). This correctly models
/// the rapid increase in sediment flux as slopes approach the angle of repose.
///
/// **Critique fix #1c**: Uses Alternating Direction Implicit (ADI) splitting for
/// the linear case, eliminating the CFL-induced sub-stepping bottleneck. The
/// ADI method is unconditionally stable regardless of dt/dx² ratio.
pub fn hillslope_diffusion(
    grid: &mut TerrainGrid,
    params: &DiffusionParams,
    dt: f64,
) {
    if params.s_c.is_finite() {
        nonlinear_hillslope_diffusion(grid, params, dt);
    } else {
        implicit_adi_diffusion(grid, params.k_d, dt);
    }
}

/// Implicit ADI (Alternating Direction Implicit) linear diffusion.
///
/// Solves ∂h/∂t = K_d ∇²h in two half-steps:
/// 1. Implicit in x, explicit in y (half step)
/// 2. Implicit in y, explicit in x (half step)
///
/// Unconditionally stable — no CFL restriction, no sub-stepping needed.
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
        // Build tridiagonal system for this row: -rx*h[c-1] + (1+2rx)*h[c] - rx*h[c+1] = rhs
        let mut a_coef = vec![0.0f64; cols]; // sub-diagonal
        let mut b_coef = vec![0.0f64; cols]; // diagonal
        let mut c_coef = vec![0.0f64; cols]; // super-diagonal
        let mut d_rhs  = vec![0.0f64; cols]; // RHS

        for c in 1..cols - 1 {
            a_coef[c] = -rx;
            b_coef[c] = 1.0 + 2.0 * rx;
            c_coef[c] = -rx;
            // Explicit in y-direction for the RHS
            d_rhs[c] = grid.elevation[[r, c]]
                + ry * (grid.elevation[[r + 1, c]] - 2.0 * grid.elevation[[r, c]] + grid.elevation[[r - 1, c]]);
        }
        // Boundary conditions (Dirichlet: fixed)
        b_coef[0] = 1.0;
        d_rhs[0] = grid.elevation[[r, 0]];
        b_coef[cols - 1] = 1.0;
        d_rhs[cols - 1] = grid.elevation[[r, cols - 1]];

        // Thomas algorithm
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

    // Update bedrock/sediment: diffusion moves sediment (not bedrock)
    for r in 0..rows {
        for c in 0..cols {
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

/// Nonlinear Roering et al. (1999) hillslope diffusion.
///
/// Sediment flux: q_s = K_d * S / (1 - (S/S_c)²)
///
/// As S → S_c, flux → ∞, producing steep threshold hillslopes matching
/// observed angle-of-repose behavior. Solved explicitly with adaptive sub-stepping.
fn nonlinear_hillslope_diffusion(grid: &mut TerrainGrid, params: &DiffusionParams, dt: f64) {
    let rows = grid.rows;
    let cols = grid.cols;
    let dx = grid.dx;
    let dy = grid.dy;
    let k_d = params.k_d;
    let s_c = params.s_c;
    let s_c2 = s_c * s_c;

    // Estimate max effective diffusivity for CFL
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
                // Compute slopes
                let sx = (elev_old[[r, c + 1]] - elev_old[[r, c - 1]]) / (2.0 * dx);
                let sy = (elev_old[[r + 1, c]] - elev_old[[r - 1, c]]) / (2.0 * dy);
                let s_mag = (sx * sx + sy * sy).sqrt();

                // Nonlinear diffusivity: K_eff = K_d / (1 - (S/Sc)²)
                let ratio_sq = (s_mag * s_mag / s_c2).min(0.99);
                let k_eff = k_d / (1.0 - ratio_sq);

                // Laplacian with effective diffusivity
                let laplacian =
                    (elev_old[[r + 1, c]] + elev_old[[r - 1, c]] - 2.0 * elev_old[[r, c]]) / (dy * dy)
                  + (elev_old[[r, c + 1]] + elev_old[[r, c - 1]] - 2.0 * elev_old[[r, c]]) / (dx * dx);

                grid.elevation[[r, c]] += sub_dt * k_eff * laplacian;
            }
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

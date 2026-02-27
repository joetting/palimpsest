//! Erosion solvers: implicit O(n) Stream Power Law, ξ-q sediment transport, and hillslope diffusion.
//!
//! The implicit solver processes nodes in stack order (outlets → ridges), solving for the
//! new elevation at each node in a single upstream pass. This achieves O(n) complexity
//! with unconditional numerical stability regardless of timestep size.

use crate::grid::TerrainGrid;
use crate::flow::FlowRoutingResult;

/// Parameters for the Stream Power Law erosion model.
#[derive(Clone, Debug)]
pub struct StreamPowerParams {
    /// Bedrock erodibility coefficient K_f [m^(1-2m) yr^(-1)]
    pub k_f: f64,
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
            m: 0.5,
            n: 1.0,
            uplift_rate: 1e-3,  // 1 mm/yr
        }
    }
}

/// Parameters for the ξ-q under-capacity sediment transport model (Davy & Lague 2009).
#[derive(Clone, Debug)]
pub struct XiQParams {
    /// Bedrock erodibility K_f
    pub k_f: f64,
    /// Discharge exponent m
    pub m: f64,
    /// Slope exponent n
    pub n: f64,
    /// Net settling velocity v_s [m/yr]
    pub v_s: f64,
    /// Vertical distribution parameter d* [dimensionless]
    pub d_star: f64,
    /// Tectonic uplift rate U [m/yr]
    pub uplift_rate: f64,
}

impl Default for XiQParams {
    fn default() -> Self {
        Self {
            k_f: 1e-5,
            m: 0.5,
            n: 1.0,
            v_s: 1.0,
            d_star: 1.0,
            uplift_rate: 1e-3,
        }
    }
}

/// Implicit O(n) Stream Power Law solver.
///
/// Solves: h_new = h_old + dt * U - dt * K_f * Q^m * S^n
///
/// For the linear case (n=1), processes nodes from outlets to ridges in stack order.
/// At each node, the implicit equation is solved directly:
///   h_i^{t+1} = (h_i^t + dt*U + dt*K_f*Q^m * h_recv^{t+1} / dx) / (1 + dt*K_f*Q^m / dx)
///
/// This is unconditionally stable: no CFL restriction on dt.
pub fn implicit_spl_erode(
    grid: &mut TerrainGrid,
    routing: &FlowRoutingResult,
    params: &StreamPowerParams,
    dt: f64,
) {
    let n_nodes = grid.len();
    let dx = grid.dx;

    // Flatten elevation for in-place update
    let mut elev: Vec<f64> = grid.elevation.iter().copied().collect();
    let discharge: Vec<f64> = grid.discharge.iter().copied().collect();

    // Apply uplift to interior nodes
    for i in 0..n_nodes {
        let (r, c) = grid.grid_index(i);
        if !grid.is_boundary(r, c) {
            elev[i] += dt * params.uplift_rate;
        }
    }

    // Process nodes in stack order (outlets first, then upstream)
    // For n=1, direct solve. For n≠1, Newton-Raphson.
    if (params.n - 1.0).abs() < 1e-10 {
        // === Linear case (n=1): direct implicit solve ===
        for &node in routing.stack.iter() {
            let recv = routing.receivers[node];
            if recv == node {
                continue; // Base level / outlet
            }

            let q_m = discharge[node].powf(params.m);
            let factor = params.k_f * q_m / dx;

            // Implicit: h_node = (h_old + dt*factor*h_recv) / (1 + dt*factor)
            elev[node] = (elev[node] + dt * factor * elev[recv]) / (1.0 + dt * factor);
        }
    } else {
        // === Nonlinear case (n≠1): localized Newton-Raphson ===
        for &node in routing.stack.iter() {
            let recv = routing.receivers[node];
            if recv == node {
                continue;
            }

            let q_m = discharge[node].powf(params.m);
            let h_recv = elev[recv];
            let h_old = elev[node];

            // Newton-Raphson: solve F(h) = h - h_old + dt*K_f*Q^m * ((h - h_recv)/dx)^n = 0
            let mut h = h_old;
            for _ in 0..20 {
                let slope = ((h - h_recv) / dx).max(0.0);
                let s_n = slope.powf(params.n);
                let f = h - h_old + dt * params.k_f * q_m * s_n;
                let df = 1.0 + dt * params.k_f * q_m * params.n
                    * slope.powf(params.n - 1.0) / dx;
                let dh = f / df;
                h -= dh;
                if dh.abs() < 1e-10 {
                    break;
                }
            }
            elev[node] = h.max(h_recv); // Can't erode below receiver
        }
    }

    // Write back
    for (i, &val) in elev.iter().enumerate() {
        let (r, c) = grid.grid_index(i);
        grid.elevation[[r, c]] = val;
    }
}

/// ξ-q under-capacity erosion-deposition solver (Davy & Lague 2009).
///
/// ∂h/∂t = U - K_f * q_w^m * |S|^n + q_s / ξ(q_w)
///
/// where ξ = (q_w * d*) / v_s is the deposition length.
///
/// Implicit O(n) solver using reverse stack traversal (Gailleton et al. 2024).
/// For n=1: direct solve. For n≠1: localized Newton-Raphson per node.
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
    let discharge: Vec<f64> = grid.discharge.iter().copied().collect();
    let mut sed_flux = vec![0.0f64; n_nodes]; // Sediment flux q_s

    // Apply uplift
    for i in 0..n_nodes {
        let (r, c) = grid.grid_index(i);
        if !grid.is_boundary(r, c) {
            elev[i] += dt * params.uplift_rate;
        }
    }

    // Process from outlets upstream
    for &node in routing.stack.iter() {
        let recv = routing.receivers[node];
        if recv == node {
            continue;
        }

        let q_w = discharge[node];
        let q_m = q_w.powf(params.m);

        // Deposition length: ξ = (q_w * d*) / v_s
        let xi = (q_w * params.d_star) / params.v_s;
        let xi = xi.max(1e-10); // Prevent division by zero

        // Incoming sediment flux from all donors
        let incoming_sed: f64 = sed_flux[node]; // Already accumulated

        if (params.n - 1.0).abs() < 1e-10 {
            // Linear case: direct implicit solve
            let h_recv = elev[recv];
            let erosion_coeff = params.k_f * q_m / dx;
            let depo_coeff = 1.0 / xi;

            // Implicit equation:
            // h = (h_old + dt*(erosion_coeff*h_recv + incoming_sed/xi)) / (1 + dt*(erosion_coeff + 1/xi))
            let h_new = (elev[node] + dt * (erosion_coeff * h_recv + incoming_sed / xi))
                / (1.0 + dt * (erosion_coeff + depo_coeff));

            // Compute erosion and deposition
            let slope = ((h_new - h_recv) / dx).max(0.0);
            let bedrock_erosion = params.k_f * q_m * slope;
            let deposition = incoming_sed / xi;
            let net_erosion = bedrock_erosion - deposition;

            // Update sediment flux to receiver
            let outgoing_sed = incoming_sed + bedrock_erosion * cell_area - deposition * cell_area;
            let outgoing_sed = outgoing_sed.max(0.0);

            let frac = routing.flow_fraction[node];
            sed_flux[recv] += outgoing_sed * frac;
            let recv2 = routing.receivers_secondary[node];
            if recv2 != node && recv2 != recv {
                sed_flux[recv2] += outgoing_sed * (1.0 - frac);
            }

            elev[node] = h_new;

            // Track erosion rate
            let (r, c) = grid.grid_index(node);
            grid.erosion_rate[[r, c]] = net_erosion;
        } else {
            // Nonlinear: Newton-Raphson
            let h_recv = elev[recv];
            let mut h = elev[node];

            for _ in 0..20 {
                let slope = ((h - h_recv) / dx).max(1e-30);
                let s_n = slope.powf(params.n);
                let erosion = params.k_f * q_m * s_n;
                let deposition = incoming_sed / xi;

                let f = h - elev[node] + dt * (erosion - deposition);
                let df = 1.0 + dt * params.k_f * q_m * params.n
                    * slope.powf(params.n - 1.0) / dx;
                let dh = f / df;
                h -= dh;
                if dh.abs() < 1e-10 {
                    break;
                }
            }

            elev[node] = h.max(h_recv);

            // Compute outgoing sediment
            let slope = ((elev[node] - h_recv) / dx).max(0.0);
            let bedrock_erosion = params.k_f * q_m * slope.powf(params.n);
            let outgoing = (incoming_sed + bedrock_erosion * cell_area
                - (incoming_sed / xi) * cell_area).max(0.0);

            let frac = routing.flow_fraction[node];
            sed_flux[recv] += outgoing * frac;
            let recv2 = routing.receivers_secondary[node];
            if recv2 != node && recv2 != recv {
                sed_flux[recv2] += outgoing * (1.0 - frac);
            }
        }
    }

    // Write back elevation and sediment flux
    for (i, &val) in elev.iter().enumerate() {
        let (r, c) = grid.grid_index(i);
        grid.elevation[[r, c]] = val;
    }
    for (i, &val) in sed_flux.iter().enumerate() {
        let (r, c) = grid.grid_index(i);
        grid.sediment_flux[[r, c]] = val;
    }
}

/// Linear hillslope diffusion: ∂h/∂t = K_d ∇²h
///
/// Explicit finite-difference with 5-point stencil (stable for dt < dx²/(4*K_d)).
/// For very large K_d or dt, consider implicit ADI; for typical LEM parameters this is fine.
pub fn hillslope_diffusion(
    grid: &mut TerrainGrid,
    k_d: f64,
    dt: f64,
) {
    let rows = grid.rows;
    let cols = grid.cols;
    let dx2 = grid.dx * grid.dx;
    let dy2 = grid.dy * grid.dy;

    // CFL check
    let max_dt = 0.25 * dx2.min(dy2) / k_d;
    let n_substeps = ((dt / max_dt).ceil() as usize).max(1);
    let sub_dt = dt / n_substeps as f64;

    for _ in 0..n_substeps {
        let elev_old = grid.elevation.clone();
        for r in 1..rows - 1 {
            for c in 1..cols - 1 {
                let laplacian =
                    (elev_old[[r + 1, c]] + elev_old[[r - 1, c]] - 2.0 * elev_old[[r, c]]) / dy2
                  + (elev_old[[r, c + 1]] + elev_old[[r, c - 1]] - 2.0 * elev_old[[r, c]]) / dx2;
                grid.elevation[[r, c]] += sub_dt * k_d * laplacian;
            }
        }
    }
}

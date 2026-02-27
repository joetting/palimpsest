//! Flow routing algorithms: D-infinity (Tarboton) and D8.
//!
//! D∞ computes flow direction as a continuous angle on 8 triangular facets,
//! partitioning flow proportionally between the 2 bounding cells. This eliminates
//! the grid-aligned artifact valleys of D8 while avoiding over-dispersion of pure MFD.
//!
//! Also computes topological stack ordering (highest to lowest) for the implicit O(n) solver.

use crate::grid::TerrainGrid;
use crate::graph::CsrFlowGraph;

/// 8 neighbor offsets: (drow, dcol) for N, NE, E, SE, S, SW, W, NW
const D8_OFFSETS: [(i32, i32); 8] = [
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0),  (1, -1), (0, -1), (-1, -1),
];

/// Result of flow routing computation.
pub struct FlowRoutingResult {
    /// Primary receiver for each node (steepest descent neighbor).
    pub receivers: Vec<usize>,
    /// For D∞: secondary receiver (the other cell bounding the steepest facet).
    /// Same as primary if D8 mode or flow is axis-aligned.
    pub receivers_secondary: Vec<usize>,
    /// Fraction of flow to primary receiver [0, 1]. Secondary gets (1 - fraction).
    pub flow_fraction: Vec<f64>,
    /// Topological stack: nodes ordered from outlet (lowest) to ridge (highest).
    /// Processing in reverse order gives the upstream→downstream traversal needed by the solver.
    pub stack: Vec<usize>,
    /// CSR donor graph (who flows *into* each node).
    pub donor_graph: CsrFlowGraph,
}

/// Compute D-infinity flow routing (Tarboton 1997).
///
/// For each interior cell, evaluates 8 triangular facets formed by the 3×3 neighborhood.
/// The steepest downslope facet defines a continuous angle, and flow is partitioned
/// proportionally between the two cells bounding that facet.
pub fn route_flow_dinf(grid: &TerrainGrid) -> FlowRoutingResult {
    let n = grid.len();
    let rows = grid.rows;
    let cols = grid.cols;
    let elev = &grid.elevation;

    let mut receivers = vec![0usize; n];
    let mut receivers_secondary = vec![0usize; n];
    let mut flow_fraction = vec![1.0f64; n];

    // Initialize: each node is its own receiver (base level / pit)
    for i in 0..n {
        receivers[i] = i;
        receivers_secondary[i] = i;
    }

    // For each cell, evaluate 8 triangular facets
    // Facet k is formed by neighbor k and neighbor (k+1)%8
    for r in 0..rows {
        for c in 0..cols {
            let idx = grid.flat_index(r, c);
            let h0 = elev[[r, c]];

            if grid.is_boundary(r, c) {
                // Boundary cells: receiver = self (fixed base level)
                receivers[idx] = idx;
                receivers_secondary[idx] = idx;
                flow_fraction[idx] = 1.0;
                continue;
            }

            let mut max_slope = 0.0f64;
            let mut best_primary = idx;
            let mut best_secondary = idx;
            let mut best_fraction = 1.0f64;

            for k in 0..8 {
                let k_next = (k + 1) % 8;
                let (dr1, dc1) = D8_OFFSETS[k];
                let (dr2, dc2) = D8_OFFSETS[k_next];

                let r1 = r as i32 + dr1;
                let c1 = c as i32 + dc1;
                let r2 = r as i32 + dr2;
                let c2 = c as i32 + dc2;

                // Bounds check
                if r1 < 0 || r1 >= rows as i32 || c1 < 0 || c1 >= cols as i32 {
                    continue;
                }
                if r2 < 0 || r2 >= rows as i32 || c2 < 0 || c2 >= cols as i32 {
                    continue;
                }

                let r1 = r1 as usize;
                let c1 = c1 as usize;
                let r2 = r2 as usize;
                let c2 = c2 as usize;
                let idx1 = grid.flat_index(r1, c1);
                let idx2 = grid.flat_index(r2, c2);

                let h1 = elev[[r1, c1]];
                let h2 = elev[[r2, c2]];

                // Distance to each neighbor
                let dist1 = ((dr1 as f64 * grid.dy).powi(2) + (dc1 as f64 * grid.dx).powi(2)).sqrt();
                let dist2 = ((dr2 as f64 * grid.dy).powi(2) + (dc2 as f64 * grid.dx).powi(2)).sqrt();

                // Slopes along the two edges of the facet
                let s1 = (h0 - h1) / dist1;
                let s2 = (h0 - h2) / dist2;

                // Steepest direction on this triangular facet
                // The facet subtends an angular range; find the angle of steepest descent
                // within this facet using planar geometry.

                // Angular span of the facet
                let _angle1 = (dr1 as f64).atan2(dc1 as f64);
                let _angle2 = (dr2 as f64).atan2(dc2 as f64);

                // Simple approach: steepest gradient on the planar triangle
                // If both neighbors are downslope, partition proportionally
                if s1 <= 0.0 && s2 <= 0.0 {
                    continue; // Both neighbors are upslope; skip this facet
                }

                // Effective slope: maximum of the planar interpolation
                let slope_mag;
                let frac;

                if s1 > 0.0 && s2 > 0.0 {
                    // Both downslope: compute steepest direction on the facet
                    slope_mag = (s1 * s1 + s2 * s2).sqrt();
                    // Partition: fraction to neighbor 1 proportional to its slope
                    frac = s1 / (s1 + s2);
                } else if s1 > 0.0 {
                    slope_mag = s1;
                    frac = 1.0;
                } else {
                    slope_mag = s2;
                    frac = 0.0;
                }

                if slope_mag > max_slope {
                    max_slope = slope_mag;
                    best_primary = idx1;
                    best_secondary = idx2;
                    best_fraction = frac;
                }
            }

            receivers[idx] = best_primary;
            receivers_secondary[idx] = best_secondary;
            flow_fraction[idx] = best_fraction;
        }
    }

    // Build donor graph via CSR from receivers
    let donor_graph = CsrFlowGraph::from_receivers(&receivers);

    // Compute topological stack ordering using Braun & Willett algorithm.
    // Process from base level upward (BFS from outlets).
    let stack = compute_stack_order(&receivers, n);

    FlowRoutingResult {
        receivers,
        receivers_secondary,
        flow_fraction,
        stack,
        donor_graph,
    }
}

/// Compute topological stack order: outlets first, then upstream nodes.
/// This is the Braun & Willett (2013) ordering used by the implicit O(n) solver.
fn compute_stack_order(receivers: &[usize], n: usize) -> Vec<usize> {
    // Build donor lists
    let mut donors: Vec<Vec<usize>> = vec![vec![]; n];
    for (i, &recv) in receivers.iter().enumerate() {
        if recv != i {
            donors[recv].push(i);
        }
    }

    // BFS/DFS from base-level nodes (self-receivers)
    let mut stack = Vec::with_capacity(n);
    let mut visited = vec![false; n];

    // Start with outlets (nodes that receive to themselves)
    let mut queue: Vec<usize> = (0..n).filter(|&i| receivers[i] == i).collect();
    for &node in &queue {
        visited[node] = true;
    }
    stack.extend_from_slice(&queue);

    // Process queue
    let mut head = 0;
    while head < queue.len() {
        let node = queue[head];
        head += 1;
        for &donor in &donors[node] {
            if !visited[donor] {
                visited[donor] = true;
                stack.push(donor);
                queue.push(donor);
            }
        }
    }

    stack
}

/// Compute drainage area and effective discharge using the stack ordering.
/// Traverses upstream → downstream (reverse stack), accumulating area and precipitation.
pub fn accumulate_flow(
    grid: &mut TerrainGrid,
    routing: &FlowRoutingResult,
) {
    let n = grid.len();
    let cell_area = grid.dx * grid.dy;

    // Reset
    for val in grid.drainage_area.iter_mut() {
        *val = cell_area;
    }
    for val in grid.discharge.iter_mut() {
        *val = 0.0;
    }

    // Initialize discharge with local precipitation × cell area
    let precip_flat: Vec<f64> = grid.precipitation.iter().copied().collect();
    let mut discharge_flat = vec![0.0f64; n];
    let mut area_flat = vec![cell_area; n];

    for i in 0..n {
        discharge_flat[i] = precip_flat[i] * cell_area;
    }

    // Traverse from ridges to outlets (reverse stack order)
    for &node in routing.stack.iter().rev() {
        let recv1 = routing.receivers[node];
        let recv2 = routing.receivers_secondary[node];
        let frac = routing.flow_fraction[node];

        if recv1 != node {
            // D∞ partitioning: split flow between primary and secondary receiver
            area_flat[recv1] += area_flat[node] * frac;
            discharge_flat[recv1] += discharge_flat[node] * frac;

            if recv2 != node && recv2 != recv1 {
                area_flat[recv2] += area_flat[node] * (1.0 - frac);
                discharge_flat[recv2] += discharge_flat[node] * (1.0 - frac);
            }
        }
    }

    // Write back to grid
    for (i, val) in area_flat.iter().enumerate() {
        let (r, c) = grid.grid_index(i);
        grid.drainage_area[[r, c]] = *val;
    }
    for (i, val) in discharge_flat.iter().enumerate() {
        let (r, c) = grid.grid_index(i);
        grid.discharge[[r, c]] = *val;
    }
}

/// Compute local slope at each cell using central differences.
pub fn compute_slopes(grid: &mut TerrainGrid) {
    let rows = grid.rows;
    let cols = grid.cols;
    let dx = grid.dx;
    let dy = grid.dy;

    for r in 0..rows {
        for c in 0..cols {
            if grid.is_boundary(r, c) {
                grid.slope[[r, c]] = 0.0;
                continue;
            }
            let dhdx = (grid.elevation[[r, c + 1]] - grid.elevation[[r, c - 1]]) / (2.0 * dx);
            let dhdy = (grid.elevation[[r + 1, c]] - grid.elevation[[r - 1, c]]) / (2.0 * dy);
            grid.slope[[r, c]] = (dhdx * dhdx + dhdy * dhdy).sqrt();
        }
    }
}

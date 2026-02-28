//! Flow routing with Priority-Flood depression filling and D-infinity partitioning.

use crate::grid::TerrainGrid;
use crate::graph::CsrFlowGraph;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

const D8_OFFSETS: [(i32, i32); 8] = [
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0),  (1, -1), (0, -1), (-1, -1),
];

#[derive(PartialEq)]
struct FloodEntry {
    elevation: f64,
    index: usize,
}

impl Eq for FloodEntry {}

impl PartialOrd for FloodEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.elevation.partial_cmp(&self.elevation)
    }
}

impl Ord for FloodEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub struct FlowRoutingResult {
    pub receivers: Vec<usize>,
    pub receivers_secondary: Vec<usize>,
    pub flow_fraction: Vec<f64>,
    pub stack: Vec<usize>,
    pub donor_graph: CsrFlowGraph,
}

pub fn priority_flood_fill(grid: &mut TerrainGrid) {
    let rows = grid.rows;
    let cols = grid.cols;
    let n = grid.len();

    let mut visited = vec![false; n];
    let mut filled: Vec<f64> = grid.elevation.iter().copied().collect();
    let mut heap = BinaryHeap::new();

    for r in 0..rows {
        for c in 0..cols {
            if grid.is_boundary(r, c) {
                let idx = grid.flat_index(r, c);
                visited[idx] = true;
                heap.push(FloodEntry { elevation: filled[idx], index: idx });
            }
        }
    }

    while let Some(entry) = heap.pop() {
        let (r, c) = grid.grid_index(entry.index);
        for &(dr, dc) in D8_OFFSETS.iter() {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr < 0 || nr >= rows as i32 || nc < 0 || nc >= cols as i32 { continue; }
            let ni = grid.flat_index(nr as usize, nc as usize);
            if visited[ni] { continue; }
            visited[ni] = true;
            filled[ni] = filled[ni].max(entry.elevation);
            heap.push(FloodEntry { elevation: filled[ni], index: ni });
        }
    }

    for (i, &val) in filled.iter().enumerate() {
        let (r, c) = grid.grid_index(i);
        grid.lake_level[[r, c]] = val;
    }
}

pub fn route_flow_dinf(grid: &TerrainGrid) -> FlowRoutingResult {
    let n = grid.len();
    let rows = grid.rows;
    let cols = grid.cols;
    let elev = &grid.lake_level;

    let mut receivers = vec![0usize; n];
    let mut receivers_secondary = vec![0usize; n];
    let mut flow_fraction = vec![1.0f64; n];

    for i in 0..n { receivers[i] = i; receivers_secondary[i] = i; }

    for r in 0..rows {
        for c in 0..cols {
            let idx = grid.flat_index(r, c);
            let h0 = elev[[r, c]];

            if grid.is_boundary(r, c) {
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

                if r1 < 0 || r1 >= rows as i32 || c1 < 0 || c1 >= cols as i32 { continue; }
                if r2 < 0 || r2 >= rows as i32 || c2 < 0 || c2 >= cols as i32 { continue; }

                let r1 = r1 as usize; let c1 = c1 as usize;
                let r2 = r2 as usize; let c2 = c2 as usize;
                let idx1 = grid.flat_index(r1, c1);
                let idx2 = grid.flat_index(r2, c2);

                let h1 = elev[[r1, c1]];
                let h2 = elev[[r2, c2]];

                let dist1 = ((dr1 as f64 * grid.dy).powi(2) + (dc1 as f64 * grid.dx).powi(2)).sqrt();
                let dist2 = ((dr2 as f64 * grid.dy).powi(2) + (dc2 as f64 * grid.dx).powi(2)).sqrt();

                let s1 = (h0 - h1) / dist1;
                let s2 = (h0 - h2) / dist2;

                if s1 <= 0.0 && s2 <= 0.0 { continue; }

                let (slope_mag, frac) = if s1 > 0.0 && s2 > 0.0 {
                    ((s1 * s1 + s2 * s2).sqrt(), s1 / (s1 + s2))
                } else if s1 > 0.0 {
                    (s1, 1.0)
                } else {
                    (s2, 0.0)
                };

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

    let donor_graph = CsrFlowGraph::from_receivers(&receivers);
    let stack = compute_stack_order(&receivers, n);

    FlowRoutingResult { receivers, receivers_secondary, flow_fraction, stack, donor_graph }
}

fn compute_stack_order(receivers: &[usize], n: usize) -> Vec<usize> {
    let mut donors: Vec<Vec<usize>> = vec![vec![]; n];
    for (i, &recv) in receivers.iter().enumerate() {
        if recv != i { donors[recv].push(i); }
    }

    let mut stack = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    let mut queue: Vec<usize> = (0..n).filter(|&i| receivers[i] == i).collect();
    for &node in &queue { visited[node] = true; }
    stack.extend_from_slice(&queue);

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

pub fn accumulate_flow(grid: &mut TerrainGrid, routing: &FlowRoutingResult) {
    let n = grid.len();
    let cell_area = grid.dx * grid.dy;

    let precip_flat: Vec<f64> = grid.precipitation.iter().copied().collect();
    let mut discharge_flat = vec![0.0f64; n];
    let mut area_flat = vec![cell_area; n];

    for i in 0..n { discharge_flat[i] = precip_flat[i] * cell_area; }

    for &node in routing.stack.iter().rev() {
        let recv1 = routing.receivers[node];
        let recv2 = routing.receivers_secondary[node];
        let frac = routing.flow_fraction[node];

        if recv1 != node {
            area_flat[recv1] += area_flat[node] * frac;
            discharge_flat[recv1] += discharge_flat[node] * frac;
            if recv2 != node && recv2 != recv1 {
                area_flat[recv2] += area_flat[node] * (1.0 - frac);
                discharge_flat[recv2] += discharge_flat[node] * (1.0 - frac);
            }
        }
    }

    for (i, val) in area_flat.iter().enumerate() {
        let (r, c) = grid.grid_index(i);
        grid.drainage_area[[r, c]] = *val;
    }
    for (i, val) in discharge_flat.iter().enumerate() {
        let (r, c) = grid.grid_index(i);
        grid.discharge[[r, c]] = *val;
    }
}

pub fn compute_slopes(grid: &mut TerrainGrid) {
    let rows = grid.rows;
    let cols = grid.cols;
    let dx = grid.dx;
    let dy = grid.dy;

    for r in 0..rows {
        for c in 0..cols {
            if grid.is_boundary(r, c) { grid.slope[[r, c]] = 0.0; continue; }
            let dhdx = (grid.elevation[[r, c + 1]] - grid.elevation[[r, c - 1]]) / (2.0 * dx);
            let dhdy = (grid.elevation[[r + 1, c]] - grid.elevation[[r - 1, c]]) / (2.0 * dy);
            grid.slope[[r, c]] = (dhdx * dhdx + dhdy * dhdy).sqrt();
        }
    }
}

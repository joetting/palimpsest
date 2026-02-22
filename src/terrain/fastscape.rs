// ============================================================================
// FastScape Landscape Evolution Model — Phase 2: The Hydraulic Computer
//
// Implements the Stream Power Law (SPL):
//   ∂h/∂t = U - Kf·A^m·S^n + Kd·∇²h
//
// Where:
//   h  = topographic elevation [m]
//   U  = tectonic uplift rate [m/yr]
//   Kf = erodibility coefficient [m^(1-2m) yr^-1]
//   A  = drainage area [m²] (proxy for discharge)
//   S  = local slope [m/m]
//   m  = drainage area exponent (typically 0.5)
//   n  = slope exponent (typically 1.0)
//   Kd = hillslope diffusivity [m² yr^-1]
//
// FastScape achieves O(N) unconditionally stable implicit integration via
// topological stack-order traversal (ridges → outlets).
// ============================================================================

use crate::ecs::world::World;

// ---------------------------------------------------------------------------
// SPL Parameters
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SplParams {
    /// Drainage area exponent m
    pub m: f32,
    /// Slope exponent n  
    pub n: f32,
    /// Cell size [m]
    pub cell_size: f32,
    /// Rainfall rate [m/yr] — scales drainage area to discharge
    pub rainfall: f32,
    /// Sea level [m] — boundary condition (fixed outlet)
    pub sea_level: f32,
    /// Maximum Newton-Raphson iterations for n ≠ 1
    pub max_nr_iters: usize,
    /// Newton-Raphson convergence tolerance
    pub nr_tol: f32,
    /// Sediment transport coefficient G (for deposition in transport-limited)
    pub g_sediment: f32,
}

impl Default for SplParams {
    fn default() -> Self {
        Self {
            m:            0.5,
            n:            1.0,  // Linear case: explicit implicit solution exists
            cell_size:    1000.0, // 1 km cells
            rainfall:     1.0,
            sea_level:    0.0,
            max_nr_iters: 50,
            nr_tol:       1e-4,
            g_sediment:   1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// FastScape Solver
// ---------------------------------------------------------------------------

pub struct FastScapeSolver {
    pub params: SplParams,
    width:  usize,
    height: usize,
    /// Pre-allocated work arrays
    receivers:   Vec<usize>,   // downstream receiver for each node
    donors:      Vec<Vec<usize>>, // upstream donors for each node
    stack:       Vec<usize>,   // topologically sorted order
    drainage:    Vec<f32>,     // drainage area [m²]
    slope:       Vec<f32>,     // slope to receiver [m/m]
}

impl FastScapeSolver {
    pub fn new(width: usize, height: usize, params: SplParams) -> Self {
        let n = width * height;
        Self {
            params,
            width,
            height,
            receivers: vec![usize::MAX; n],
            donors:    vec![Vec::new(); n],
            stack:     Vec::with_capacity(n),
            drainage:  vec![0.0; n],
            slope:     vec![0.0; n],
        }
    }

    pub fn n_nodes(&self) -> usize {
        self.width * self.height
    }

    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    #[inline]
    fn coords(&self, idx: usize) -> (usize, usize) {
        (idx % self.width, idx / self.width)
    }

    fn is_boundary(&self, x: usize, y: usize) -> bool {
        x == 0 || y == 0 || x == self.width - 1 || y == self.height - 1
    }

    // -----------------------------------------------------------------------
    // Step 1: Flow Routing — find single-flow-direction receivers
    // -----------------------------------------------------------------------

    /// For each node, find the steepest-descent neighbor (D8 routing).
    /// Boundary nodes are their own outlets (fixed base level = sea level).
    pub fn route_flow(&mut self, h: &[f32]) {
        let n = self.n_nodes();
        let cell = self.params.cell_size;
        let diag = cell * std::f32::consts::SQRT_2;

        for i in 0..n {
            self.receivers[i] = usize::MAX;
            self.donors[i].clear();
        }

        for y in 0..self.height {
            for x in 0..self.width {
                let i = self.idx(x, y);

                if self.is_boundary(x, y) {
                    // Boundary = outlet (fixed to sea level)
                    self.receivers[i] = usize::MAX;
                    continue;
                }

                let hi = h[i];
                let mut best_slope: f32 = 0.0;
                let mut best_j = usize::MAX;

                // D8: check all 8 neighbors
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 { continue; }
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx < 0 || ny < 0
                            || nx >= self.width as i32
                            || ny >= self.height as i32 {
                            continue;
                        }
                        let j = self.idx(nx as usize, ny as usize);
                        let dist = if dx != 0 && dy != 0 { diag } else { cell };
                        let s = (hi - h[j]) / dist;
                        if s > best_slope {
                            best_slope = s;
                            best_j = j;
                        }
                    }
                }

                self.receivers[i] = best_j;
                self.slope[i] = best_slope;

                if best_j != usize::MAX {
                    self.donors[best_j].push(i);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 2: Stack Order — topological sort from outlets to ridges
    // -----------------------------------------------------------------------

    /// Build a topologically sorted array: indices ordered from outlets (base)
    /// to ridges (peaks). FastScape's implicit solver requires this ordering
    /// to propagate solutions downstream → upstream.
    pub fn compute_stack(&mut self) {
        let n = self.n_nodes();
        self.stack.clear();
        self.stack.reserve(n);

        // Start from all outlets (boundary nodes and nodes with no upstream receiver)
        let mut queue = std::collections::VecDeque::new();
        for i in 0..n {
            if self.receivers[i] == usize::MAX {
                queue.push_back(i);
            }
        }

        // BFS from outlets, building stack in reverse topological order
        let mut visited = vec![false; n];
        while let Some(i) = queue.pop_front() {
            if visited[i] { continue; }
            visited[i] = true;
            self.stack.push(i);
            for &donor in &self.donors[i].clone() {
                if !visited[donor] {
                    queue.push_back(donor);
                }
            }
        }

        // stack is currently outlets→ridges; for implicit solve we want ridges→outlets
        self.stack.reverse();
    }

    // -----------------------------------------------------------------------
    // Step 3: Drainage Area — accumulate upstream area downstream
    // -----------------------------------------------------------------------

    /// Compute drainage area by traversing stack from ridges → outlets,
    /// accumulating cell area through the receiver tree.
    pub fn accumulate_drainage(&mut self) {
        let cell_area = self.params.cell_size * self.params.cell_size;
        // Initialize: each cell drains itself
        for a in self.drainage.iter_mut() {
            *a = cell_area;
        }
        // Traverse ridges → outlets (stack order, reversed)
        // Our stack was reversed, so iterate in original push order = ridges first
        for &i in &self.stack {
            let r = self.receivers[i];
            if r != usize::MAX {
                let ai = self.drainage[i];
                self.drainage[r] += ai;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 4: Implicit Fluvial Incision Solver
    // -----------------------------------------------------------------------

    /// Solve the implicit SPL for channel erosion.
    ///
    /// For n=1 (linear case), exact implicit solution:
    ///   h_new[i] = (h_old[i] + Kf·A^m·S_ref·Δt·h_new[r]) / (1 + Kf·A^m·S_ref·Δt / cell)
    ///
    /// For n≠1, Newton-Raphson iteration is used.
    /// Traversal is outlet → ridge (outlets first, so receivers are solved before donors).
    pub fn solve_incision(&self, h: &mut Vec<f32>, uplift: &[f32], kf: &[f32], dt: f32) {
        let n = self.params.n;
        let m = self.params.m;
        let cell = self.params.cell_size;

        // Apply uplift first
        for i in 0..h.len() {
            h[i] += uplift[i] * dt;
        }

        // Traverse outlets → ridges (reverse of our stack which is ridges→outlets)
        for &i in self.stack.iter().rev() {
            let r = self.receivers[i];
            if r == usize::MAX {
                // Boundary: fix to sea level
                h[i] = self.params.sea_level.max(h[i]);
                continue;
            }

            let a = self.drainage[i];
            let ki = kf[i];
            let hr = h[r]; // receiver already solved (outlet-first order)

            // Distance to receiver
            let (xi, yi) = self.coords(i);
            let (xr, yr) = self.coords(r);
            let dx = (xi as i32 - xr as i32).unsigned_abs() as f32;
            let dy = (yi as i32 - yr as i32).unsigned_abs() as f32;
            let dist = if dx > 0.0 && dy > 0.0 { cell * std::f32::consts::SQRT_2 } else { cell };

            // Characteristic erosion coefficient F = Kf·A^m·Δt / dist
            let f_coeff = ki * a.powf(m) * dt / dist;

            let h_old = h[i];

            if (n - 1.0).abs() < 1e-6 {
                // Linear case: exact implicit solution
                h[i] = (h_old + f_coeff * hr) / (1.0 + f_coeff);
            } else {
                // Nonlinear: Newton-Raphson
                let mut h_new = h_old;
                for _ in 0..self.params.max_nr_iters {
                    let slope = ((h_new - hr) / dist).max(0.0);
                    let erosion = ki * a.powf(m) * slope.powf(n) * dt;
                    let f_val = h_new - h_old + erosion - uplift[i] * dt;
                    let df = 1.0 + ki * a.powf(m) * n * slope.powf(n - 1.0) * dt / dist;
                    let delta = f_val / df;
                    h_new -= delta;
                    if delta.abs() < self.params.nr_tol { break; }
                }
                h[i] = h_new.max(hr); // can't erode below receiver
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 5: Hillslope Diffusion (explicit)
    // -----------------------------------------------------------------------

    /// Apply hillslope diffusion: ∂h/∂t = Kd·∇²h
    /// Uses explicit finite differences — stable when Kd·Δt/cell² < 0.5
    pub fn apply_diffusion(&self, h: &mut Vec<f32>, kd: &[f32], dt: f32) {
        let w = self.width;
        let hgt = self.height;
        let cell2 = self.params.cell_size * self.params.cell_size;
        let h_old = h.clone();

        for y in 1..hgt - 1 {
            for x in 1..w - 1 {
                let i = self.idx(x, y);
                let laplacian = h_old[self.idx(x + 1, y)]
                    + h_old[self.idx(x - 1, y)]
                    + h_old[self.idx(x, y + 1)]
                    + h_old[self.idx(x, y - 1)]
                    - 4.0 * h_old[i];
                h[i] += kd[i] * laplacian / cell2 * dt;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Full Epoch Step
    // -----------------------------------------------------------------------

    /// Run one geological epoch timestep on the height field.
    ///
    /// Returns the elevation delta array (Δh) for nutrient piggybacking.
    pub fn step_epoch(
        &mut self,
        h: &mut Vec<f32>,
        uplift: &[f32],
        kf: &[f32],
        kd: &[f32],
        dt: f32,
    ) -> Vec<f32> {
        let h_before = h.clone();

        // 1. Route flow
        self.route_flow(h);
        // 2. Topological sort
        self.compute_stack();
        // 3. Drainage area
        self.accumulate_drainage();
        // 4. Implicit channel incision
        self.solve_incision(h, uplift, kf, dt);
        // 5. Explicit hillslope diffusion
        self.apply_diffusion(h, kd, dt);

        // Compute elevation deltas for downstream use (nutrient transport, etc.)
        h.iter().zip(h_before.iter()).map(|(new, old)| new - old).collect()
    }

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    /// Maximum elevation in the grid
    pub fn max_elevation(&self, h: &[f32]) -> f32 {
        h.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Mean elevation
    pub fn mean_elevation(&self, h: &[f32]) -> f32 {
        h.iter().sum::<f32>() / h.len() as f32
    }

    /// Total erosion flux [m³/yr]
    pub fn total_erosion_flux(&self, deltas: &[f32], dt: f32) -> f32 {
        let cell_area = self.params.cell_size * self.params.cell_size;
        deltas.iter().filter(|&&d| d < 0.0).map(|&d| -d * cell_area / dt).sum()
    }

    /// Export drainage area array
    pub fn drainage_area(&self) -> &[f32] {
        &self.drainage
    }

    /// Export receiver array
    pub fn receivers(&self) -> &[usize] {
        &self.receivers
    }
}

// ---------------------------------------------------------------------------
// Tectonic Forcing Interface
// ---------------------------------------------------------------------------

/// Uplift configuration for "Elder God" player interventions
#[derive(Debug, Clone)]
pub struct TectonicForcing {
    /// Baseline uplift rate [m/yr] applied uniformly
    pub base_uplift: f32,
    /// Regional uplift overrides: (x, y, radius, rate)
    pub hotspots: Vec<UpliftHotspot>,
}

#[derive(Debug, Clone)]
pub struct UpliftHotspot {
    pub cx: f32,
    pub cy: f32,
    pub radius: f32,
    pub rate: f32,   // [m/yr]
}

impl TectonicForcing {
    pub fn new(base_uplift: f32) -> Self {
        Self { base_uplift, hotspots: Vec::new() }
    }

    pub fn add_hotspot(&mut self, cx: f32, cy: f32, radius: f32, rate: f32) {
        self.hotspots.push(UpliftHotspot { cx, cy, radius, rate });
    }

    /// Generate the uplift array for the current configuration
    pub fn uplift_array(&self, width: usize, height: usize, cell_size: f32) -> Vec<f32> {
        let n = width * height;
        let mut u = vec![self.base_uplift; n];
        for hs in &self.hotspots {
            for y in 0..height {
                for x in 0..width {
                    let px = x as f32 * cell_size;
                    let py = y as f32 * cell_size;
                    let dist = ((px - hs.cx).powi(2) + (py - hs.cy).powi(2)).sqrt();
                    if dist <= hs.radius {
                        // Gaussian falloff
                        let t = dist / hs.radius;
                        let w = (-3.0 * t * t).exp();
                        u[y * width + x] += hs.rate * w;
                    }
                }
            }
        }
        u
    }
}

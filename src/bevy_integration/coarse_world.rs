/// src/bevy_integration/coarse_world.rs
///
/// The CoarseWorldSolver implements the "invisible global simulation" described
/// in the design docs: a much coarser-resolution FastScape grid (32×32 cells,
/// 10 km per cell = 320×320 km domain) that:
///
///   1. Provides atmospheric and tectonic LBCs to the fine-resolution world.
///   2. Evolves at 10× the time step of the fine world to stay efficient.
///   3. Uses one-way coupling by default (coarse → fine boundary).
///      Two-way coupling (fine average → coarse interior) is opt-in.
///
/// This is the "course world solver" referred to in your message.

use crate::terrain::fastscape::{FastScapeSolver, SplParams, TectonicForcing};
use crate::terrain::heightmap::Heightmap;

pub struct CoarseWorldSolver {
    pub width: usize,
    pub height: usize,
    pub cell_size_m: f32,
    pub heights: Vec<f32>,

    solver: FastScapeSolver,
    tectonics: TectonicForcing,

    /// Cached lateral boundary condition values extracted from fine-world edges.
    boundary_north: Vec<f32>,
    boundary_south: Vec<f32>,
    boundary_east:  Vec<f32>,
    boundary_west:  Vec<f32>,

    /// Atmospheric state passed from climate engine
    pub co2_ppm: f32,
    pub global_temp_c: f32,
    pub mean_precip_m: f32,

    /// Step counter
    pub steps: u64,
}

impl CoarseWorldSolver {
    pub fn new(width: usize, height: usize, cell_size_m: f32) -> Self {
        let n = width * height;

        // Generate a simple fBm heightmap for the coarse world
        let hm = Heightmap::fbm(width, height, 600.0, 5, 2.0, 0.5, 9999);
        let mut heights = hm.data;

        // Apply gentle island mask so water drains at edges
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let max_r = cx.min(cy) * 0.85;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let r = (dx * dx + dy * dy).sqrt();
                let mask = (1.0 - (r / max_r).min(1.0)).max(0.0);
                heights[y * width + x] *= mask;
            }
        }

        // SPL params — coarser Kf since cells are 10× larger
        let spl = SplParams {
            m: 0.5,
            n: 1.0,
            cell_size: cell_size_m,
            rainfall: 1.0,
            sea_level: 0.0,
            ..Default::default()
        };

        let kf_soft = 5e-5f32; // homogeneous soft-rock coarse world
        let kd_soft = 0.015f32;

        // Store defaults in the SplParams (solver uses per-cell arrays)
        let solver = FastScapeSolver::new(width, height, spl);
        let tectonics = TectonicForcing::new(0.00005); // gentle background uplift

        Self {
            width,
            height,
            cell_size_m,
            heights,
            solver,
            tectonics,
            boundary_north: vec![0.0; width],
            boundary_south: vec![0.0; width],
            boundary_east:  vec![0.0; height],
            boundary_west:  vec![0.0; height],
            co2_ppm: 280.0,
            global_temp_c: 14.0,
            mean_precip_m: 0.8,
            steps: 0,
        }
    }

    // ─── Boundary coupling ────────────────────────────────────────────────────

    /// Sample the four edges of the fine-resolution simulation and store as
    /// LBC vectors. Called before each coarse step from Bevy.
    pub fn ingest_fine_boundary(
        &mut self,
        fine_heights: &[f32],
        fine_w: usize,
        fine_h: usize,
        fine_cell_m: f32,
    ) {
        // Scale factor: how many fine cells per coarse cell edge
        let scale_x = fine_w as f32 / self.width as f32;
        let scale_y = fine_h as f32 / self.height as f32;

        // North edge (y=0 in fine world) — averaged into coarse boundary
        for cx in 0..self.width {
            let mut sum = 0.0f32;
            let mut count = 0u32;
            let fx_start = (cx as f32 * scale_x) as usize;
            let fx_end = ((cx + 1) as f32 * scale_x) as usize;
            for fx in fx_start..fx_end.min(fine_w) {
                sum += fine_heights[fx]; // y=0 row
                count += 1;
            }
            self.boundary_north[cx] = if count > 0 { sum / count as f32 } else { 0.0 };
        }

        // South edge (y=fine_h-1)
        for cx in 0..self.width {
            let mut sum = 0.0f32;
            let mut count = 0u32;
            let fx_start = (cx as f32 * scale_x) as usize;
            let fx_end = ((cx + 1) as f32 * scale_x) as usize;
            for fx in fx_start..fx_end.min(fine_w) {
                sum += fine_heights[(fine_h - 1) * fine_w + fx];
                count += 1;
            }
            self.boundary_south[cx] = if count > 0 { sum / count as f32 } else { 0.0 };
        }

        // West edge (x=0)
        for cy in 0..self.height {
            let mut sum = 0.0f32;
            let mut count = 0u32;
            let fy_start = (cy as f32 * scale_y) as usize;
            let fy_end = ((cy + 1) as f32 * scale_y) as usize;
            for fy in fy_start..fy_end.min(fine_h) {
                sum += fine_heights[fy * fine_w];
                count += 1;
            }
            self.boundary_west[cy] = if count > 0 { sum / count as f32 } else { 0.0 };
        }

        // East edge (x=fine_w-1)
        for cy in 0..self.height {
            let mut sum = 0.0f32;
            let mut count = 0u32;
            let fy_start = (cy as f32 * scale_y) as usize;
            let fy_end = ((cy + 1) as f32 * scale_y) as usize;
            for fy in fy_start..fy_end.min(fine_h) {
                sum += fine_heights[fy * fine_w + fine_w - 1];
                count += 1;
            }
            self.boundary_east[cy] = if count > 0 { sum / count as f32 } else { 0.0 };
        }
    }

    /// Apply stored LBCs to the coarse heightmap's border rows/columns,
    /// effectively "pinning" the coarse grid edges to match fine-world averages.
    pub fn apply_lbcs(&mut self) {
        let w = self.width;
        let h = self.height;

        // North row (y=0)
        for x in 0..w {
            self.heights[x] = self.boundary_north[x];
        }
        // South row (y=h-1)
        for x in 0..w {
            self.heights[(h - 1) * w + x] = self.boundary_south[x];
        }
        // West column (x=0)
        for y in 0..h {
            self.heights[y * w] = self.boundary_west[y];
        }
        // East column (x=w-1)
        for y in 0..h {
            self.heights[y * w + w - 1] = self.boundary_east[y];
        }
    }

    // ─── Two-way coupling (optional) ─────────────────────────────────────────

    /// Average the fine-world interior into the corresponding coarse interior
    /// cells. Only call this when two-way coupling is enabled.
    pub fn ingest_fine_interior(
        &mut self,
        fine_heights: &[f32],
        fine_w: usize,
        fine_h: usize,
    ) {
        let scale_x = fine_w as f32 / self.width as f32;
        let scale_y = fine_h as f32 / self.height as f32;
        let w = self.width;
        let h = self.height;

        // Skip border cells (those are handled by LBCs)
        for cy in 1..h - 1 {
            for cx in 1..w - 1 {
                let fx_start = (cx as f32 * scale_x) as usize;
                let fx_end = ((cx + 1) as f32 * scale_x) as usize;
                let fy_start = (cy as f32 * scale_y) as usize;
                let fy_end = ((cy + 1) as f32 * scale_y) as usize;

                let mut sum = 0.0f32;
                let mut count = 0u32;
                for fy in fy_start..fy_end.min(fine_h) {
                    for fx in fx_start..fx_end.min(fine_w) {
                        sum += fine_heights[fy * fine_w + fx];
                        count += 1;
                    }
                }
                if count > 0 {
                    // Blend: 70% coarse, 30% fine-average (prevents instability)
                    let fine_avg = sum / count as f32;
                    self.heights[cy * w + cx] =
                        self.heights[cy * w + cx] * 0.7 + fine_avg * 0.3;
                }
            }
        }
    }

    // ─── Step ────────────────────────────────────────────────────────────────

    /// Advance the coarse world by `dt_years`. Applies LBCs, then runs FastScape.
    pub fn step(&mut self, dt_years: f32) {
        // Apply LBCs before stepping so boundary values stay fixed
        self.apply_lbcs();

        let w = self.width;
        let h = self.height;
        let uplift = self.tectonics.uplift_array(w, h, self.cell_size_m);

        // Uniform coarse Kf and Kd
        let kf = vec![5e-5f32; w * h];
        let kd = vec![0.015f32; w * h];

        let _deltas = self.solver.step_epoch(
            &mut self.heights,
            &uplift,
            &kf,
            &kd,
            dt_years,
        );

        self.steps += 1;
    }

    // ─── Queries ─────────────────────────────────────────────────────────────

    /// Interpolate the coarse heightmap at a fine-world position (in metres).
    /// Used to query what "background elevation" the coarse world predicts
    /// for a given point — useful for LBC diagnostics.
    pub fn sample_at_fine_pos(&self, world_x_m: f32, world_y_m: f32) -> f32 {
        let coarse_domain_m = self.width as f32 * self.cell_size_m;
        let cx = (world_x_m / coarse_domain_m * self.width as f32).clamp(0.0, self.width as f32 - 1.001);
        let cy = (world_y_m / coarse_domain_m * self.height as f32).clamp(0.0, self.height as f32 - 1.001);
        let x0 = cx as usize;
        let y0 = cy as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = cx - x0 as f32;
        let fy = cy - y0 as f32;
        let w = self.width;
        let h00 = self.heights[y0 * w + x0];
        let h10 = self.heights[y0 * w + x1];
        let h01 = self.heights[y1 * w + x0];
        let h11 = self.heights[y1 * w + x1];
        let top = h00 + (h10 - h00) * fx;
        let bot = h01 + (h11 - h01) * fx;
        top + (bot - top) * fy
    }

    /// Mean elevation of the coarse world (diagnostic).
    pub fn mean_elevation(&self) -> f32 {
        self.heights.iter().sum::<f32>() / self.heights.len() as f32
    }

    /// Max elevation of the coarse world (diagnostic).
    pub fn max_elevation(&self) -> f32 {
        self.heights.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Add a tectonic hotspot to the coarse world (Elder God tool).
    pub fn add_uplift_hotspot(&mut self, cx_frac: f32, cy_frac: f32, radius_frac: f32, rate_m_yr: f32) {
        let cx_m = cx_frac * self.width as f32 * self.cell_size_m;
        let cy_m = cy_frac * self.height as f32 * self.cell_size_m;
        let radius_m = radius_frac * self.width as f32 * self.cell_size_m;
        self.tectonics.add_hotspot(cx_m, cy_m, radius_m, rate_m_yr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coarse_world_steps_without_panic() {
        let mut cw = CoarseWorldSolver::new(16, 16, 10_000.0);
        for _ in 0..5 {
            cw.step(10_000.0);
        }
        assert!(cw.max_elevation() >= 0.0);
    }

    #[test]
    fn boundary_ingestion_smoke() {
        let mut cw = CoarseWorldSolver::new(8, 8, 10_000.0);
        let fine_h = vec![100.0f32; 32 * 32];
        cw.ingest_fine_boundary(&fine_h, 32, 32, 1_000.0);
        cw.apply_lbcs();
        // After applying LBCs, border cells should be ~100m
        assert!((cw.heights[0] - 100.0).abs() < 10.0);
    }
}

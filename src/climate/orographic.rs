// ============================================================================
// Orographic Precipitation Engine — LFPM 1.0
//
// Implements the Linear Feedback Precipitation Model (Smith & Barstad 2004,
// simplified to LFPM 1.0 for O(N) performance in voxel engines).
//
// Algorithm:
//   1. Cast wind vectors across the elevation grid from the windward edge.
//   2. Each wind ray carries a moisture value Qv [m].
//   3. As the ray encounters rising terrain, it is forced to ascend
//      (orographic lifting). Ascending moist air cools, condensing water
//      into cloud water Qc.
//   4. Precipitation falls where Qc exceeds a threshold (collision efficiency).
//   5. On the leeward side, descending air warms and re-evaporates — creating
//      a dry rain shadow.
//
// The two tracked components:
//   Qv — Vapor:       moisture carried by the wind parcel [m equivalent depth]
//   Qc — Cloud Water: condensed moisture ready to precipitate [m]
//
// Linear feedback interactions:
//   dQv/ds = −α × Qv × (dh/ds)⁺  — condensation on ascent (positive slope)
//   dQc/ds = +α × Qv × (dh/ds)⁺ − β × Qc   — accumulation minus fallout
//   P[x]   = β × Qc                           — precipitation rate at each cell
//
// Where:
//   s — along-wind distance
//   α — condensation coefficient [m⁻¹]
//   β — fallout rate [km⁻¹]
//   (dh/ds)⁺ = max(0, dh/ds) — only upslope contributes
// ============================================================================

use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Wind parameters
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WindParams {
    /// Wind direction angle [radians from east, anticlockwise]
    /// 0 = eastward, π/2 = northward, π = westward, 3π/2 = southward
    pub angle_rad: f32,
    /// Baseline moisture carried by the wind into the domain [m/yr equivalent]
    pub incoming_moisture_m: f32,
    /// Condensation coefficient α [per meter of ascent]
    pub alpha_condensation: f32,
    /// Cloud water fallout rate β [per km of travel]
    pub beta_fallout_per_km: f32,
    /// Background precipitation floor (even on lee sides) [m/yr]
    pub background_precip_m: f32,
}

impl Default for WindParams {
    fn default() -> Self {
        Self {
            angle_rad:           std::f32::consts::PI,  // westerly (wind from west)
            incoming_moisture_m: 2.0,  // generous maritime moisture
            alpha_condensation:  0.005, // 0.5% moisture loss per metre of ascent
            beta_fallout_per_km: 0.15, // 15% of cloud water falls per km
            background_precip_m: 0.05, // 50mm/yr background (even deserts get something)
        }
    }
}

impl WindParams {
    /// Unit vector of wind travel direction (dx, dy) per step
    pub fn wind_dir(&self) -> (f32, f32) {
        (self.angle_rad.cos(), self.angle_rad.sin())
    }
}

// ---------------------------------------------------------------------------
// Orographic output
// ---------------------------------------------------------------------------

/// Per-column precipitation output from the orographic solver.
pub struct OrographicOutput {
    /// Annual precipitation [m/yr] per column
    pub precipitation: Vec<f32>,
    /// Wind vapor remaining at each column (for diagnostics)
    pub vapor:         Vec<f32>,
    /// Cloud water at each column (for diagnostics)
    pub cloud_water:   Vec<f32>,
    pub grid_width:    usize,
    pub grid_height:   usize,
}

impl OrographicOutput {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        Self {
            precipitation: vec![0.0; n],
            vapor:         vec![0.0; n],
            cloud_water:   vec![0.0; n],
            grid_width: width,
            grid_height: height,
        }
    }

    pub fn mean_precipitation(&self) -> f32 {
        self.precipitation.iter().sum::<f32>() / self.precipitation.len() as f32
    }

    pub fn max_precipitation(&self) -> f32 {
        self.precipitation.iter().cloned().fold(0.0_f32, f32::max)
    }
}

// ---------------------------------------------------------------------------
// Orographic Engine
// ---------------------------------------------------------------------------

pub struct OrographicEngine {
    pub grid_width:  usize,
    pub grid_height: usize,
    pub wind:        WindParams,
    pub cell_size_m: f32,
}

impl OrographicEngine {
    pub fn new(width: usize, height: usize, wind: WindParams, cell_size_m: f32) -> Self {
        Self {
            grid_width: width,
            grid_height: height,
            wind,
            cell_size_m,
        }
    }

    /// Compute orographic precipitation for the given elevation grid.
    ///
    /// Uses parallel ray casting: each row/column of entry points spawns one ray
    /// that is marched across the grid in the wind direction.
    pub fn compute(&self, elevations: &[f32]) -> OrographicOutput {
        let mut output = OrographicOutput::new(self.grid_width, self.grid_height);
        let w = self.grid_width;
        let h = self.grid_height;

        // Wind direction unit vector (travel direction of the air parcel)
        let (wdx, wdy) = self.wind.wind_dir();

        // We march rays from the windward edge. Determine which edges to seed.
        // Rays enter from:
        //   wdx > 0: left edge (x=0), sweeping right
        //   wdx < 0: right edge (x=w-1), sweeping left
        //   wdy > 0: bottom edge (y=0), sweeping up
        //   wdy < 0: top edge (y=h-1), sweeping down
        //
        // Each cell on the windward edges is a ray origin.
        // For diagonal winds, we seed BOTH edges and average.

        let n_rays_horizontal = h;  // one per row
        let n_rays_vertical   = w;  // one per column

        // Collect all (origin, precip/vapor/cloud) outputs and splat onto grid
        // Parallel rays using rayon
        let alpha = self.wind.alpha_condensation;
        let beta  = self.wind.beta_fallout_per_km;
        let qv0   = self.wind.incoming_moisture_m;
        let background = self.wind.background_precip_m;
        let cell_km = self.cell_size_m / 1000.0; // cell size in km (for beta)

        // We'll use a contribution count array so we can average multiple-ray hits
        let mut contrib_count = vec![0u32; w * h];

        // ---- Horizontal rays (enter from left if wdx>0, right if wdx<0) ----
        if wdx.abs() > 0.05 {
            let ray_results: Vec<(Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>)> = (0..n_rays_horizontal)
                .into_par_iter()
                .map(|row_y| {
                    let start_x: i32 = if wdx >= 0.0 { 0 } else { (w as i32) - 1 };
                    let x0 = start_x as f32;
                    let y0 = row_y as f32;
                    march_ray(
                        x0, y0, wdx, wdy,
                        w, h,
                        elevations,
                        qv0, alpha, beta, cell_km, background,
                    )
                })
                .collect();

            for (indices, precips, vapors, clouds) in ray_results {
                for (k, &idx) in indices.iter().enumerate() {
                    output.precipitation[idx] += precips[k];
                    output.vapor[idx]          += vapors[k];
                    output.cloud_water[idx]    += clouds[k];
                    contrib_count[idx]         += 1;
                }
            }
        }

        // ---- Vertical rays (enter from bottom if wdy>0, top if wdy<0) ----
        if wdy.abs() > 0.05 {
            let ray_results: Vec<(Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>)> = (0..n_rays_vertical)
                .into_par_iter()
                .map(|col_x| {
                    let start_y: i32 = if wdy >= 0.0 { 0 } else { (h as i32) - 1 };
                    let x0 = col_x as f32;
                    let y0 = start_y as f32;
                    march_ray(
                        x0, y0, wdx, wdy,
                        w, h,
                        elevations,
                        qv0, alpha, beta, cell_km, background,
                    )
                })
                .collect();

            for (indices, precips, vapors, clouds) in ray_results {
                for (k, &idx) in indices.iter().enumerate() {
                    output.precipitation[idx] += precips[k];
                    output.vapor[idx]          += vapors[k];
                    output.cloud_water[idx]    += clouds[k];
                    contrib_count[idx]         += 1;
                }
            }
        }

        // Normalize by contribution count
        for idx in 0..(w * h) {
            let c = contrib_count[idx].max(1) as f32;
            output.precipitation[idx] /= c;
            output.vapor[idx]          /= c;
            output.cloud_water[idx]    /= c;

            // Ensure minimum background precipitation everywhere
            output.precipitation[idx] = output.precipitation[idx].max(background);
        }

        output
    }

    /// Update wind parameters (Elder God intervention or seasonal cycle)
    pub fn set_wind(&mut self, wind: WindParams) {
        self.wind = wind;
    }

    /// Rotate wind direction by delta_radians (seasonal shifting)
    pub fn rotate_wind(&mut self, delta_rad: f32) {
        self.wind.angle_rad = (self.wind.angle_rad + delta_rad)
            .rem_euclid(2.0 * std::f32::consts::PI);
    }
}

// ---------------------------------------------------------------------------
// Ray march — the core LFPM 1.0 solver for a single wind ray
// ---------------------------------------------------------------------------

/// March a single wind ray across the elevation grid, computing LFPM precipitation.
///
/// Returns (cell_indices, precipitation_values, vapor_values, cloud_values)
/// for every cell the ray passes through.
fn march_ray(
    x0: f32, y0: f32,
    wdx: f32, wdy: f32,
    width: usize, height: usize,
    elevations: &[f32],
    qv0: f32,         // incoming vapor [m/yr]
    alpha: f32,        // condensation coefficient [per m of ascent]
    beta: f32,         // fallout rate [per km of travel]
    cell_km: f32,      // cell size in km
    background: f32,   // floor precipitation
) -> (Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut indices = Vec::new();
    let mut precips = Vec::new();
    let mut vapors  = Vec::new();
    let mut clouds  = Vec::new();

    let mut qv = qv0;   // vapor
    let mut qc = 0.0_f32; // cloud water
    let mut prev_elev = 0.0_f32;

    // Step the ray cell by cell using Bresenham-style DDA
    let mut x = x0;
    let mut y = y0;

    // Normalize step so we move roughly one cell per iteration
    let step_scale = 1.0 / wdx.abs().max(wdy.abs()).max(1e-6);
    let sx = wdx * step_scale;
    let sy = wdy * step_scale;

    let max_steps = (width + height) * 2;

    for _ in 0..max_steps {
        let ix = x.round() as i32;
        let iy = y.round() as i32;
        if ix < 0 || iy < 0 || ix >= width as i32 || iy >= height as i32 {
            break;
        }
        let idx = iy as usize * width + ix as usize;
        let elev = elevations[idx];

        // Orographic ascent: positive when terrain rises in wind direction
        let dh = (elev - prev_elev).max(0.0); // only upslope (positive)

        // Condensation: ascending air loses vapor → gains cloud water
        let condensation = alpha * qv * dh;
        let condensation = condensation.min(qv); // can't condense more than available
        qv = (qv - condensation).max(0.0);
        qc += condensation;

        // Precipitation fallout from cloud water
        let fallout = beta * cell_km * qc;
        let fallout = fallout.min(qc);
        qc = (qc - fallout).max(0.0);

        // Total precipitation at this cell: fallout + background floor
        let precip = fallout.max(background);

        indices.push(idx);
        precips.push(precip);
        vapors.push(qv);
        clouds.push(qc);

        prev_elev = elev;
        x += sx;
        y += sy;
    }

    (indices, precips, vapors, clouds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orographic_creates_rain_shadow() {
        // Simple 10x1 grid with a mountain in the middle
        let width = 20;
        let height = 1;
        let mut elevs = vec![0.0f32; width];
        // Mountain at x=8..12
        for x in 7..13 {
            elevs[x] = 2000.0;
        }

        let wind = WindParams {
            angle_rad: 0.0, // eastward wind (enters from left)
            incoming_moisture_m: 2.0,
            ..WindParams::default()
        };
        let engine = OrographicEngine::new(width, height, wind, 1000.0);
        let out = engine.compute(&elevs);

        // Windward (left) side should have more rain than leeward (right)
        let windward_precip: f32 = out.precipitation[3..7].iter().sum::<f32>() / 4.0;
        let leeward_precip:  f32 = out.precipitation[13..17].iter().sum::<f32>() / 4.0;
        assert!(
            windward_precip > leeward_precip,
            "Expected windward > leeward, got {:.3} vs {:.3}",
            windward_precip, leeward_precip
        );
    }
}

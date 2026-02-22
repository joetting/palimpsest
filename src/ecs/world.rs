// ============================================================================
// ECS World — Archetype-based entity storage
//
// Stores TerrainColumn entities in a flat Vec (SoA-style) for cache-friendly
// iteration during compute-heavy geological passes.
// ============================================================================

use super::components::{
    ActivityMask, Entity, GridPos, MaterialId, TerrainColumn, UpdateCohortId,
};

pub struct World {
    /// Flat array of all terrain columns, indexed by flat grid position.
    /// Layout: columns[y * width + x]
    pub columns: Vec<TerrainColumn>,
    pub width: u32,
    pub height: u32,
    pub activity: ActivityMask,
    next_entity: u32,
}

impl World {
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width * height) as usize;
        let mut activity = ActivityMask::new(width, height);
        activity.fill_active();
        Self {
            columns: Vec::with_capacity(n),
            width,
            height,
            activity,
            next_entity: 0,
        }
    }

    /// Spawn entity, returning its ID
    pub fn spawn(&mut self) -> Entity {
        let e = Entity(self.next_entity);
        self.next_entity += 1;
        e
    }

    /// Initialize the world grid from a height function.
    ///
    /// height_fn: (x, y) -> elevation in meters
    /// mat_fn:    (elevation) -> MaterialId
    /// num_cohorts: how many update cohorts to distribute columns into
    pub fn initialize_grid<F, M>(
        &mut self,
        height_fn: F,
        mat_fn: M,
        num_cohorts: u8,
    ) where
        F: Fn(u32, u32) -> f32,
        M: Fn(f32) -> MaterialId,
    {
        let w = self.width;
        let h = self.height;
        self.columns.clear();
        self.columns.reserve((w * h) as usize);

        for y in 0..h {
            for x in 0..w {
                let elev = height_fn(x, y);
                let mat = mat_fn(elev);
                let cohort = ((y * w + x) % num_cohorts as u32) as u8;
                let col = TerrainColumn::new(GridPos::new(x, y), elev, mat, cohort);
                self.columns.push(col);
            }
        }
    }

    /// Get a column by grid coordinates
    #[inline]
    pub fn column(&self, x: u32, y: u32) -> &TerrainColumn {
        &self.columns[(y * self.width + x) as usize]
    }

    /// Get a mutable column by grid coordinates
    #[inline]
    pub fn column_mut(&mut self, x: u32, y: u32) -> &mut TerrainColumn {
        &mut self.columns[(y * self.width + x) as usize]
    }

    /// Get column by flat index
    #[inline]
    pub fn column_by_index(&self, idx: usize) -> &TerrainColumn {
        &self.columns[idx]
    }

    #[inline]
    pub fn column_by_index_mut(&mut self, idx: usize) -> &mut TerrainColumn {
        &mut self.columns[idx]
    }

    /// Number of active columns
    pub fn active_count(&self) -> usize {
        self.activity.active_count()
    }

    /// Total columns
    pub fn total_columns(&self) -> usize {
        self.columns.len()
    }

    /// Extract a flat elevation array for GPU upload (SoA layout)
    pub fn elevation_array(&self) -> Vec<f32> {
        self.columns.iter().map(|c| c.elevation.0).collect()
    }

    /// Extract a flat erodibility array for GPU upload
    pub fn erodibility_array(&self) -> Vec<f32> {
        self.columns.iter().map(|c| c.erodibility.0).collect()
    }

    /// Extract drainage area array
    pub fn drainage_area_array(&self) -> Vec<f32> {
        self.columns.iter().map(|c| c.drainage.0).collect()
    }

    /// Apply elevation deltas from the solver back to columns
    pub fn apply_elevation_deltas(&mut self, deltas: &[f32]) {
        assert_eq!(deltas.len(), self.columns.len());
        for (col, &dh) in self.columns.iter_mut().zip(deltas.iter()) {
            col.elevation.0 += dh;
            // Clamp to sea floor
            if col.elevation.0 < 0.0 {
                col.elevation.0 = 0.0;
            }
            // Sync the top layer thickness
            let e = col.elevation.0;
            if col.layers.count > 0 {
                let top = col.layers.count - 1;
                // Simple: adjust top layer to match new surface
                let base = col.layers.base[top];
                col.layers.thickness[top] = (e - base).max(0.0);
            }
        }
    }

    /// Iterate all columns with their flat index (for parallel work)
    pub fn columns_with_index(&self) -> impl Iterator<Item = (usize, &TerrainColumn)> {
        self.columns.iter().enumerate()
    }

    /// Get the 4-connected neighbors of a column (wrapping disabled — boundaries return None)
    pub fn neighbors_4(&self, x: u32, y: u32) -> [Option<(u32, u32)>; 4] {
        let w = self.width;
        let h = self.height;
        [
            if y > 0     { Some((x, y - 1)) } else { None }, // N
            if y < h - 1 { Some((x, y + 1)) } else { None }, // S
            if x > 0     { Some((x - 1, y)) } else { None }, // W
            if x < w - 1 { Some((x + 1, y)) } else { None }, // E
        ]
    }

    /// Get the 8-connected neighbors
    pub fn neighbors_8(&self, x: u32, y: u32) -> Vec<(u32, u32)> {
        let w = self.width;
        let h = self.height;
        let mut result = Vec::with_capacity(8);
        let x = x as i32;
        let y = y as i32;
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                if dx == 0 && dy == 0 { continue; }
                let nx = x + dx;
                let ny = y + dy;
                if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                    result.push((nx as u32, ny as u32));
                }
            }
        }
        result
    }

    /// Compute the slope from column (x,y) to neighbor (nx,ny) [m/m]
    pub fn slope_to(&self, x: u32, y: u32, nx: u32, ny: u32, cell_size: f32) -> f32 {
        let dh = self.column(x, y).elevation.0 - self.column(nx, ny).elevation.0;
        let dist = if x != nx && y != ny {
            cell_size * std::f32::consts::SQRT_2
        } else {
            cell_size
        };
        dh / dist
    }
}

/// 2.5D Global Heightmap
///
/// The entire planetary surface as a 2D grid of HeightmapCells.
/// Supports periodic (toroidal) boundary conditions for the "closed sphere" model.
/// Designed for the FastScape O(N) geological solver to operate on.

use rayon::prelude::*;

use crate::common::{EngineConfig, IVec2};

/// Per-cell data for the global 2.5D simulation.
/// Packed for cache-friendliness: 48 bytes per cell.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct HeightmapCell {
    /// Terrain elevation in world units.
    pub elevation: f32,
    /// Bedrock elevation (below which is impenetrable).
    pub bedrock: f32,
    /// Soil/sediment depth above bedrock.
    pub sediment_depth: f32,
    /// Water table height (for hydrology).
    pub water_level: f32,
    /// Accumulated nutrient value (e.g., phosphorus concentration).
    pub nutrient: f32,
    /// Surface temperature (for climate/thermostat).
    pub temperature: f32,
    /// Flow accumulation from drainage solver (Stream Power Law).
    pub flow_accumulation: f32,
    /// Drainage direction encoded as index offset (for FastScape).
    pub drainage_receiver: i32,
    /// Biome/region classification tag.
    pub biome_id: u16,
    /// Flags (e.g., is_ocean, is_glaciated, has_settlement).
    pub flags: u16,
}

impl Default for HeightmapCell {
    fn default() -> Self {
        Self {
            elevation: 0.0,
            bedrock: 0.0,
            sediment_depth: 0.0,
            water_level: 0.0,
            nutrient: 0.0,
            temperature: 20.0,
            flow_accumulation: 0.0,
            drainage_receiver: -1,
            biome_id: 0,
            flags: 0,
        }
    }
}

impl HeightmapCell {
    /// Surface height = bedrock + sediment.
    pub fn surface_height(&self) -> f32 {
        self.bedrock + self.sediment_depth
    }

    // --- Flag helpers ---
    pub const FLAG_OCEAN: u16 = 1 << 0;
    pub const FLAG_GLACIATED: u16 = 1 << 1;
    pub const FLAG_SETTLEMENT: u16 = 1 << 2;
    pub const FLAG_ACTIVE_ZONE: u16 = 1 << 3;

    pub fn has_flag(&self, flag: u16) -> bool {
        self.flags & flag != 0
    }

    pub fn set_flag(&mut self, flag: u16) {
        self.flags |= flag;
    }

    pub fn clear_flag(&mut self, flag: u16) {
        self.flags &= !flag;
    }
}

/// The global heightmap: a flat Vec storing a `resolution x resolution` grid
/// with periodic (toroidal) boundary conditions.
pub struct Heightmap {
    /// Side length of the square grid.
    pub resolution: u32,
    /// The cell data, stored in row-major order.
    pub cells: Vec<HeightmapCell>,
    /// World-space size of each cell (meters per cell).
    pub cell_size: f64,
}

impl Heightmap {
    /// Create a new heightmap from engine config, all cells zeroed.
    pub fn new(config: &EngineConfig) -> Self {
        let res = config.global_resolution;
        let total = (res as usize) * (res as usize);
        Self {
            resolution: res,
            cells: vec![HeightmapCell::default(); total],
            cell_size: config.voxel_size,
        }
    }

    /// Create from an existing elevation array (e.g., loaded from disk).
    pub fn from_elevations(resolution: u32, elevations: &[f32], cell_size: f64) -> Self {
        let total = (resolution as usize) * (resolution as usize);
        assert_eq!(
            elevations.len(),
            total,
            "Elevation data size mismatch: expected {}, got {}",
            total,
            elevations.len()
        );
        let cells: Vec<HeightmapCell> = elevations
            .iter()
            .map(|&elev| HeightmapCell {
                elevation: elev,
                bedrock: elev,
                sediment_depth: 0.0,
                ..Default::default()
            })
            .collect();
        Self {
            resolution,
            cells,
            cell_size,
        }
    }

    // -----------------------------------------------------------------------
    // Indexing with periodic (toroidal) boundary conditions
    // -----------------------------------------------------------------------

    /// Wrap coordinate to [0, resolution) for periodic boundaries.
    #[inline]
    pub fn wrap(&self, coord: i32) -> u32 {
        let r = self.resolution as i32;
        ((coord % r + r) % r) as u32
    }

    /// Convert 2D grid coords to flat index, with wrapping.
    #[inline]
    pub fn index(&self, x: i32, y: i32) -> usize {
        let wx = self.wrap(x);
        let wy = self.wrap(y);
        (wy as usize) * (self.resolution as usize) + (wx as usize)
    }

    /// Get cell reference at (x, y) with periodic wrapping.
    #[inline]
    pub fn get(&self, x: i32, y: i32) -> &HeightmapCell {
        &self.cells[self.index(x, y)]
    }

    /// Get mutable cell reference at (x, y) with periodic wrapping.
    #[inline]
    pub fn get_mut(&mut self, x: i32, y: i32) -> &mut HeightmapCell {
        let idx = self.index(x, y);
        &mut self.cells[idx]
    }

    /// Get the 4 Von Neumann neighbors (N, E, S, W) with periodic wrapping.
    pub fn neighbors_4(&self, x: i32, y: i32) -> [IVec2; 4] {
        [
            IVec2::new(x, y - 1), // North
            IVec2::new(x + 1, y), // East
            IVec2::new(x, y + 1), // South
            IVec2::new(x - 1, y), // West
        ]
    }

    /// Get the 8 Moore neighbors with periodic wrapping.
    pub fn neighbors_8(&self, x: i32, y: i32) -> [IVec2; 8] {
        [
            IVec2::new(x - 1, y - 1),
            IVec2::new(x, y - 1),
            IVec2::new(x + 1, y - 1),
            IVec2::new(x - 1, y),
            IVec2::new(x + 1, y),
            IVec2::new(x - 1, y + 1),
            IVec2::new(x, y + 1),
            IVec2::new(x + 1, y + 1),
        ]
    }

    // -----------------------------------------------------------------------
    // Parallel operations via Rayon
    // -----------------------------------------------------------------------

    /// Apply a function to every cell in parallel. Read-only.
    pub fn par_for_each<F>(&self, f: F)
    where
        F: Fn(u32, u32, &HeightmapCell) + Sync + Send,
    {
        let res = self.resolution;
        self.cells.par_iter().enumerate().for_each(|(idx, cell)| {
            let x = (idx % res as usize) as u32;
            let y = (idx / res as usize) as u32;
            f(x, y, cell);
        });
    }

    /// Apply a mutable function to every cell in parallel.
    /// Each cell is independent so this is safe with rayon.
    pub fn par_for_each_mut<F>(&mut self, f: F)
    where
        F: Fn(u32, u32, &mut HeightmapCell) + Sync + Send,
    {
        let res = self.resolution;
        self.cells
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, cell)| {
                let x = (idx % res as usize) as u32;
                let y = (idx / res as usize) as u32;
                f(x, y, cell);
            });
    }

    /// Parallel reduction: compute a single value from all cells.
    pub fn par_reduce<T, Map, Reduce>(&self, identity: T, map: Map, reduce: Reduce) -> T
    where
        T: Send + Sync + Clone,
        Map: Fn(&HeightmapCell) -> T + Sync + Send,
        Reduce: Fn(T, T) -> T + Sync + Send,
    {
        self.cells
            .par_iter()
            .map(map)
            .reduce(|| identity.clone(), reduce)
    }

    // -----------------------------------------------------------------------
    // Solver interface stubs (you will provide actual solver code)
    // -----------------------------------------------------------------------

    /// Extract a flat f32 elevation array for FFI / solver input.
    pub fn elevations(&self) -> Vec<f32> {
        self.cells.iter().map(|c| c.elevation).collect()
    }

    /// Extract a flat f32 sediment array.
    pub fn sediments(&self) -> Vec<f32> {
        self.cells.iter().map(|c| c.sediment_depth).collect()
    }

    /// Write back elevation data from a solver output.
    pub fn set_elevations(&mut self, data: &[f32]) {
        assert_eq!(data.len(), self.cells.len());
        self.cells
            .iter_mut()
            .zip(data.iter())
            .for_each(|(cell, &elev)| {
                cell.elevation = elev;
            });
    }

    /// Write back drainage receivers from solver output.
    pub fn set_drainage_receivers(&mut self, data: &[i32]) {
        assert_eq!(data.len(), self.cells.len());
        self.cells
            .iter_mut()
            .zip(data.iter())
            .for_each(|(cell, &recv)| {
                cell.drainage_receiver = recv;
            });
    }

    /// Write back flow accumulation from solver output.
    pub fn set_flow_accumulation(&mut self, data: &[f32]) {
        assert_eq!(data.len(), self.cells.len());
        self.cells
            .iter_mut()
            .zip(data.iter())
            .for_each(|(cell, &flow)| {
                cell.flow_accumulation = flow;
            });
    }

    // -----------------------------------------------------------------------
    // Region extraction for local zone instantiation
    // -----------------------------------------------------------------------

    /// Extract a rectangular subregion of the heightmap for local 3D instantiation.
    /// Returns cells in row-major order for the subregion [x0..x0+width, y0..y0+height].
    /// Coordinates wrap at boundaries.
    pub fn extract_region(
        &self,
        x0: i32,
        y0: i32,
        width: u32,
        height: u32,
    ) -> Vec<HeightmapCell> {
        let mut region = Vec::with_capacity((width as usize) * (height as usize));
        for dy in 0..height as i32 {
            for dx in 0..width as i32 {
                region.push(*self.get(x0 + dx, y0 + dy));
            }
        }
        region
    }

    // -----------------------------------------------------------------------
    // Utility / diagnostics
    // -----------------------------------------------------------------------

    /// Total number of cells.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Min/max elevation across the entire map.
    pub fn elevation_range(&self) -> (f32, f32) {
        self.cells.iter().fold((f32::MAX, f32::MIN), |(lo, hi), c| {
            (lo.min(c.elevation), hi.max(c.elevation))
        })
    }

    /// Sum of all nutrient values (conservation check for the Proserpina Principle).
    pub fn total_nutrient(&self) -> f64 {
        self.par_reduce(
            0.0_f64,
            |c| c.nutrient as f64,
            |a, b| a + b,
        )
    }

    /// Sum of all sediment (mass conservation check).
    pub fn total_sediment(&self) -> f64 {
        self.par_reduce(
            0.0_f64,
            |c| c.sediment_depth as f64,
            |a, b| a + b,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(res: u32) -> EngineConfig {
        EngineConfig {
            global_resolution: res,
            ..Default::default()
        }
    }

    #[test]
    fn test_periodic_wrapping() {
        let hm = Heightmap::new(&test_config(16));
        // Positive wrap
        assert_eq!(hm.wrap(16), 0);
        assert_eq!(hm.wrap(17), 1);
        // Negative wrap
        assert_eq!(hm.wrap(-1), 15);
        assert_eq!(hm.wrap(-17), 15);
    }

    #[test]
    fn test_get_set() {
        let mut hm = Heightmap::new(&test_config(8));
        hm.get_mut(3, 4).elevation = 42.0;
        assert_eq!(hm.get(3, 4).elevation, 42.0);
        // Periodic: (3 + 8, 4 + 8) should alias to (3, 4)
        assert_eq!(hm.get(11, 12).elevation, 42.0);
    }

    #[test]
    fn test_parallel_mutation() {
        let mut hm = Heightmap::new(&test_config(64));
        hm.par_for_each_mut(|x, y, cell| {
            cell.elevation = (x as f32) + (y as f32) * 0.01;
        });
        let cell = hm.get(10, 20);
        assert!((cell.elevation - 10.2).abs() < 0.001);
    }

    #[test]
    fn test_extract_region_wrapping() {
        let mut hm = Heightmap::new(&test_config(8));
        // Mark cell (7, 7) — this should be accessible via wrapping from (-1, -1)
        hm.get_mut(7, 7).elevation = 99.0;
        let region = hm.extract_region(-1, -1, 3, 3);
        // region[0] = cell(-1, -1) = cell(7, 7)
        assert_eq!(region[0].elevation, 99.0);
    }

    #[test]
    fn test_nutrient_conservation() {
        let mut hm = Heightmap::new(&test_config(32));
        hm.par_for_each_mut(|_, _, cell| {
            cell.nutrient = 1.0;
        });
        let total = hm.total_nutrient();
        assert!((total - 1024.0).abs() < 0.01); // 32*32 = 1024
    }
}

/// Activity mask: a bitfield that marks which cells are "active" for
/// expensive per-cell computation (nutrient cycling, etc.).
pub mod activity_mask {
    pub struct ActivityProcessor {
        pub grid_width: usize,
        pub grid_height: usize,
    }

    impl ActivityProcessor {
        pub fn new(grid_width: usize, grid_height: usize) -> Self {
            Self { grid_width, grid_height }
        }

        /// Create a mask where all cells above sea_level are active.
        pub fn all_land_active(&self, elevations: &[f32], sea_level: f32) -> Vec<u32> {
            let n = self.grid_width * self.grid_height;
            let words = (n + 31) / 32;
            let mut mask = vec![0u32; words];
            for i in 0..n {
                if elevations.get(i).copied().unwrap_or(0.0) > sea_level {
                    mask[i / 32] |= 1 << (i % 32);
                }
            }
            mask
        }

        /// Count active cells.
        pub fn active_count(mask: &[u32]) -> usize {
            mask.iter().map(|w| w.count_ones() as usize).sum()
        }
    }
}

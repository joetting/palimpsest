/// src/bevy_integration/render_resources.rs
///
/// `TerrainRenderData` is a plain Bevy Resource that acts as the CPU-side
/// "render staging buffer". The mesh system reads it whenever `dirty` is true.
/// Keeping it separate from the simulation resource means the render thread
/// never blocks the simulation and vice versa.

use bevy::prelude::*;

/// All data the terrain mesh + overlay systems need.
#[derive(Resource, Default)]
pub struct TerrainRenderData {
    // Fine-world grid
    pub width:      usize,
    pub height:     usize,
    pub cell_size_m: f32,
    pub heights:    Vec<f32>,
    pub biomes:     Vec<u8>,   // one byte per column: Biome as u8
    pub sea_level:  f32,

    // Climate summary
    pub co2_ppm:    f32,
    pub temp_c:     f32,
    pub in_ice_age: bool,

    // Coarse world (for debug split-screen or LBC visualisation)
    pub coarse_heights: Vec<f32>,
    pub coarse_width:   usize,
    pub coarse_height:  usize,

    /// Set true by the upload system; cleared by the mesh-rebuild system.
    pub dirty: bool,
}

impl TerrainRenderData {
    /// Map a column index to a world-space XZ position in metres.
    pub fn world_pos(&self, idx: usize) -> Vec3 {
        let x = (idx % self.width) as f32 * self.cell_size_m;
        let z = (idx / self.width) as f32 * self.cell_size_m;
        let y = self.heights.get(idx).copied().unwrap_or(0.0);
        Vec3::new(x, y, z)
    }

    /// Biome colour for vertex painting.
    /// Colours match the `Biome` enum ordering in climate/mod.rs.
    pub fn biome_color(biome_u8: u8) -> Color {
        match biome_u8 {
            0 => Color::srgb(0.10, 0.25, 0.65), // Ocean
            1 => Color::srgb(0.08, 0.55, 0.12), // TropicalForest
            2 => Color::srgb(0.27, 0.55, 0.20), // TemperateForest
            3 => Color::srgb(0.70, 0.78, 0.30), // Grassland
            4 => Color::srgb(0.88, 0.78, 0.50), // Desert
            5 => Color::srgb(0.62, 0.78, 0.72), // Tundra
            6 => Color::srgb(0.55, 0.55, 0.60), // Alpine
            7 => Color::srgb(0.92, 0.96, 1.00), // Glacier
            _ => Color::srgb(0.5, 0.5, 0.5),
        }
    }

    /// Height normalised to [0,1] for colour scaling.
    pub fn normalized_height(&self, idx: usize) -> f32 {
        let h = self.heights.get(idx).copied().unwrap_or(0.0);
        let max_h = self.heights.iter().cloned().fold(1.0f32, f32::max);
        (h / max_h).clamp(0.0, 1.0)
    }
}

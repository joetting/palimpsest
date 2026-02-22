/// LOD Streaming System
///
/// Bridges the 2.5D global heightmap and the local 3D SVO.
/// - Instantiates heightmap regions into full 3D voxels as the player moves.
/// - Compresses distant regions back into heightmap-only representation.
/// - Manages the "active zone" ring around the player.

use crate::common::{EngineConfig, IVec2, Vec3, VoxelMaterial};
use crate::heightmap::Heightmap;
use crate::svo::SparseVoxelOctree;

/// Tracks the player's position and the active local zone.
pub struct LodManager {
    /// Current center of the active zone in heightmap grid coordinates.
    pub active_center: IVec2,
    /// Radius of the active zone in heightmap cells.
    pub active_radius: u32,
    /// The live SVO for the local zone.
    pub local_svo: Option<SparseVoxelOctree>,
    /// Engine configuration.
    config: EngineConfig,
    /// How far (in cells) the player must move before triggering a re-center.
    pub hysteresis: u32,
}

impl LodManager {
    pub fn new(config: &EngineConfig) -> Self {
        let active_radius = (config.local_radius / config.voxel_size) as u32;
        Self {
            active_center: IVec2::new(0, 0),
            active_radius,
            local_svo: None,
            config: config.clone(),
            hysteresis: active_radius / 4, // re-center when player drifts 25%
        }
    }

    /// Called every frame (or at intervals). Returns true if the SVO was rebuilt.
    pub fn update(&mut self, player_world_pos: &Vec3, heightmap: &Heightmap) -> bool {
        let player_cell = IVec2::new(
            (player_world_pos.x / heightmap.cell_size) as i32,
            (player_world_pos.z / heightmap.cell_size) as i32,
        );

        let dx = (player_cell.x - self.active_center.x).unsigned_abs();
        let dy = (player_cell.y - self.active_center.y).unsigned_abs();

        // Only rebuild if player has moved beyond the hysteresis threshold,
        // or if no SVO exists yet.
        if self.local_svo.is_some() && dx <= self.hysteresis && dy <= self.hysteresis {
            return false;
        }

        // Re-center and rebuild.
        self.active_center = player_cell;
        self.rebuild_svo(heightmap);
        true
    }

    /// Instantiate the heightmap region around `active_center` into a 3D SVO.
    fn rebuild_svo(&mut self, heightmap: &Heightmap) {
        let center = self.active_center;
        let _radius = self.active_radius as i32;
        let cell_size = heightmap.cell_size;

        // World-space origin of the SVO = center of active zone.
        let origin = Vec3::new(
            center.x as f64 * cell_size,
            0.0, // Y origin at sea level; heightmap provides vertical data.
            center.y as f64 * cell_size,
        );

        // Build the SVO by sampling the heightmap for each voxel position.
        let svo = SparseVoxelOctree::from_sample(
            &self.config,
            origin,
            |wx, wy, wz| {
                // Convert world position to heightmap cell.
                let hx = (wx / cell_size).floor() as i32;
                let hy = (wz / cell_size).floor() as i32;

                let cell = heightmap.get(hx, hy);
                let surface = cell.surface_height() as f64;
                let bedrock = cell.bedrock as f64;
                let water = cell.water_level as f64;

                // Generate material based on vertical position.
                if wy < bedrock {
                    VoxelMaterial::BEDROCK
                } else if wy < surface {
                    // Between bedrock and surface = soil/sediment.
                    // Could layer materials by depth here.
                    if wy < bedrock + (surface - bedrock) * 0.3 {
                        VoxelMaterial::ROCK
                    } else {
                        VoxelMaterial::SOIL
                    }
                } else if wy < water {
                    VoxelMaterial::WATER
                } else {
                    VoxelMaterial::AIR
                }
            },
        );

        self.local_svo = Some(svo);
    }

    /// Compress the current SVO state back into the heightmap.
    /// Call this before discarding the SVO (e.g., when the player moves away).
    /// This captures any modifications the player or agents made to the terrain.
    pub fn compress_to_heightmap(&self, heightmap: &mut Heightmap) {
        let svo = match &self.local_svo {
            Some(s) => s,
            None => return,
        };

        let cell_size = heightmap.cell_size;
        let bounds = svo.bounds();

        // For each heightmap cell in the active zone, scan the SVO column
        // to find the new surface height and update the heightmap.
        let x_start = ((bounds.min.x / cell_size).floor()) as i32;
        let x_end = ((bounds.max.x / cell_size).ceil()) as i32;
        let z_start = ((bounds.min.z / cell_size).floor()) as i32;
        let z_end = ((bounds.max.z / cell_size).ceil()) as i32;

        for hz in z_start..z_end {
            for hx in x_start..x_end {
                let wx = (hx as f64 + 0.5) * cell_size;
                let wz = (hz as f64 + 0.5) * cell_size;

                // Scan downward from top of SVO to find highest solid voxel.
                let mut highest_solid = bounds.min.y;
                let voxel_step = cell_size / (1 << svo.max_depth()) as f64;

                let mut wy = bounds.max.y - voxel_step * 0.5;
                while wy >= bounds.min.y {
                    let mat = svo.get_material(&Vec3::new(wx, wy, wz));
                    if mat.is_solid() {
                        highest_solid = wy + voxel_step * 0.5;
                        break;
                    }
                    wy -= voxel_step;
                }

                let cell = heightmap.get_mut(hx, hz);
                cell.elevation = highest_solid as f32;
                cell.sediment_depth = (highest_solid as f32 - cell.bedrock).max(0.0);
            }
        }
    }

    /// Get an immutable reference to the local SVO (if loaded).
    pub fn svo(&self) -> Option<&SparseVoxelOctree> {
        self.local_svo.as_ref()
    }

    /// Get a mutable reference to the local SVO.
    pub fn svo_mut(&mut self) -> Option<&mut SparseVoxelOctree> {
        self.local_svo.as_mut()
    }

    /// Check whether a world position is within the active 3D zone.
    pub fn is_in_active_zone(&self, pos: &Vec3) -> bool {
        match &self.local_svo {
            Some(svo) => svo.bounds().contains_point(pos),
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_env() -> (EngineConfig, Heightmap) {
        let config = EngineConfig {
            global_resolution: 64,
            svo_max_depth: 4,
            voxel_size: 1.0,
            local_radius: 8.0,
            lod_tiers: 2,
        };
        let mut hm = Heightmap::new(&config);
        // Create a simple terrain: flat at elevation 10, bedrock at 0.
        hm.par_for_each_mut(|_x, _y, cell| {
            cell.bedrock = 0.0;
            cell.elevation = 10.0;
            cell.sediment_depth = 10.0;
        });
        (config, hm)
    }

    #[test]
    fn test_initial_build() {
        let (config, hm) = make_test_env();
        let mut lod = LodManager::new(&config);
        let rebuilt = lod.update(&Vec3::new(32.0, 5.0, 32.0), &hm);
        assert!(rebuilt);
        assert!(lod.svo().is_some());

        let svo = lod.svo().unwrap();
        // Below surface should be solid.
        assert!(svo
            .get_material(&Vec3::new(32.0, 2.0, 32.0))
            .is_solid());
        // Above surface should be air.
        assert!(svo
            .get_material(&Vec3::new(32.0, 12.0, 32.0))
            .is_air());
    }

    #[test]
    fn test_hysteresis() {
        let (config, hm) = make_test_env();
        let mut lod = LodManager::new(&config);
        // First update: always rebuilds.
        lod.update(&Vec3::new(32.0, 5.0, 32.0), &hm);
        // Tiny move: should NOT rebuild.
        let rebuilt = lod.update(&Vec3::new(32.5, 5.0, 32.5), &hm);
        assert!(!rebuilt);
    }

    #[test]
    fn test_compress_roundtrip() {
        let (config, mut hm) = make_test_env();
        let mut lod = LodManager::new(&config);
        lod.update(&Vec3::new(32.0, 5.0, 32.0), &hm);

        // Modify terrain in the SVO: dig a hole.
        if let Some(svo) = lod.svo_mut() {
            svo.set_material(&Vec3::new(32.0, 9.0, 32.0), VoxelMaterial::AIR);
        }

        // Compress back to heightmap.
        lod.compress_to_heightmap(&mut hm);

        // The elevation at (32, 32) should have decreased.
        let cell = hm.get(32, 32);
        assert!(cell.elevation < 10.0 || cell.sediment_depth < 10.0);
    }
}

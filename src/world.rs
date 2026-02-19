use std::collections::HashMap;
use noise::{NoiseFn, Perlin, Fbm};
use crate::camera::Camera;
use crate::chunk::{Chunk, Voxel, CHUNK_SIZE, build_mesh, Vertex};

pub const RENDER_DIST: i32 = 6; // chunks

#[derive(Default, Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub struct ChunkPos(pub i32, pub i32, pub i32);

pub struct ChunkMesh {
    pub vertices: Vec<Vertex>,
    pub indices:  Vec<u32>,
}

pub struct World {
    pub chunks:  HashMap<ChunkPos, Chunk>,
    pub meshes:  HashMap<ChunkPos, ChunkMesh>,
    noise: Fbm<Perlin>,
}

impl World {
    pub fn new() -> Self {
        let mut noise = Fbm::<Perlin>::new(42);
        noise.octaves = 6;
        noise.frequency = 0.5;
        noise.lacunarity = 2.0;
        noise.persistence = 0.5;

        Self {
            chunks: HashMap::new(),
            meshes: HashMap::new(),
            noise,
        }
    }

    pub fn update(&mut self, camera: &Camera) {
        let cam_chunk = ChunkPos(
            (camera.position.x / CHUNK_SIZE as f32).floor() as i32,
            (camera.position.y / CHUNK_SIZE as f32).floor() as i32,
            (camera.position.z / CHUNK_SIZE as f32).floor() as i32,
        );

        // Load chunks in range
        for dz in -RENDER_DIST..=RENDER_DIST {
            for dy in -2..=2i32 {
                for dx in -RENDER_DIST..=RENDER_DIST {
                    let pos = ChunkPos(
                        cam_chunk.0 + dx,
                        cam_chunk.1 + dy,
                        cam_chunk.2 + dz,
                    );
                    if !self.chunks.contains_key(&pos) {
                        let chunk = self.generate_chunk(pos);
                        self.chunks.insert(pos, chunk);
                    }
                }
            }
        }

        // Build dirty meshes
        let dirty_positions: Vec<ChunkPos> = self.chunks.iter()
            .filter(|(_, c)| c.dirty)
            .map(|(p, _)| *p)
            .collect();

        for pos in dirty_positions {
            let world_origin = [
                pos.0 as f32 * CHUNK_SIZE as f32,
                pos.1 as f32 * CHUNK_SIZE as f32,
                pos.2 as f32 * CHUNK_SIZE as f32,
            ];
            let chunk = &self.chunks[&pos];
            let (verts, idxs) = build_mesh(chunk, world_origin);
            self.meshes.insert(pos, ChunkMesh { vertices: verts, indices: idxs });
            self.chunks.get_mut(&pos).unwrap().dirty = false;
        }

        // Unload distant chunks
        let to_remove: Vec<ChunkPos> = self.chunks.keys()
            .filter(|p| {
                let dx = (p.0 - cam_chunk.0).abs();
                let dz = (p.2 - cam_chunk.2).abs();
                dx > RENDER_DIST + 2 || dz > RENDER_DIST + 2
            })
            .cloned()
            .collect();

        for pos in to_remove {
            self.chunks.remove(&pos);
            self.meshes.remove(&pos);
        }
    }

    fn generate_chunk(&self, pos: ChunkPos) -> Chunk {
        let mut chunk = Chunk::empty();
        let cs = CHUNK_SIZE as i32;

        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let wx = (pos.0 * cs + lx as i32) as f64;
                let wz = (pos.2 * cs + lz as i32) as f64;

                // Sample multi-scale terrain height
                let scale = 0.006;
                let h_norm = self.noise.get([wx * scale, wz * scale]);

                // Map noise to world height - tall, dramatic terrain
                let terrain_h = 48.0 + h_norm * 50.0;

                // Secondary detail layer
                let detail_scale = 0.025;
                let detail = self.noise.get([wx * detail_scale + 100.0, wz * detail_scale + 100.0]) * 8.0;
                let terrain_h = terrain_h + detail;

                for ly in 0..CHUNK_SIZE {
                    let wy = pos.1 * cs + ly as i32;
                    let wy_f = wy as f64;

                    if wy_f > terrain_h {
                        continue; // air
                    }

                    let depth = terrain_h - wy_f;
                    let voxel = if depth < 1.0 {
                        // Vary surface material by slope / height
                        if terrain_h > 85.0 {
                            Voxel::Rock
                        } else if terrain_h < 20.0 {
                            Voxel::Sand
                        } else {
                            Voxel::Grass
                        }
                    } else if depth < 4.0 {
                        if terrain_h < 20.0 { Voxel::Sand } else { Voxel::Soil }
                    } else {
                        Voxel::Stone
                    };

                    chunk.set(lx, ly, lz, voxel);
                }
            }
        }

        chunk
    }
}

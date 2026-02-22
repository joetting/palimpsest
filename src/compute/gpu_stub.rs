// ============================================================================
// GPU Compute Pipeline — Phase 1 CPU Stub
//
// Provides the interface and WGSL shader templates for the WGPU integration
// in Phase 2. Phase 1 runs the same logic on CPU with rayon parallelism,
// so the architecture is correct before GPU binding is added.
//
// Structure-of-Arrays (SoA) layout matches what GPU compute shaders expect:
//   - All elevations packed together
//   - All erodibilities packed together
//   etc.
//
// This ensures zero-cost port to actual WGPU storage buffers.
// ============================================================================

use rayon::prelude::*;

/// The SoA terrain buffer — mirrors what would be GPU storage buffers
#[derive(Debug)]
pub struct TerrainBuffers {
    pub elevation:    Vec<f32>,
    pub erodibility:  Vec<f32>,
    pub diffusivity:  Vec<f32>,
    pub uplift:       Vec<f32>,
    pub drainage:     Vec<f32>,
    pub sediment:     Vec<f32>,
    pub width:        usize,
    pub height:       usize,
}

impl TerrainBuffers {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        Self {
            elevation:   vec![0.0; n],
            erodibility: vec![1e-5; n],
            diffusivity: vec![0.01; n],
            uplift:      vec![0.0; n],
            drainage:    vec![1.0; n],
            sediment:    vec![0.0; n],
            width,
            height,
        }
    }

    pub fn n_elements(&self) -> usize {
        self.elevation.len()
    }

    /// Pack elevation + erodibility into vec4 pairs for GPU upload simulation.
    /// On real GPU: packed as vec4<f32> arrays for coalesced memory access.
    pub fn pack_vec4_terrain_a(&self) -> Vec<[f32; 4]> {
        self.elevation.iter().zip(self.erodibility.iter())
            .zip(self.diffusivity.iter().zip(self.uplift.iter()))
            .map(|((e, kf), (kd, u))| [*e, *kf, *kd, *u])
            .collect()
    }

    /// WGSL shader template for terrain update — documents GPU implementation
    pub fn wgsl_terrain_update_shader() -> &'static str {
        r#"
// WGSL Compute Shader: terrain_update.wgsl
// Phase 2 GPU implementation of SPL apply step
//
// Invoked AFTER CPU FastScape solver computes Δh:
//   dispatch_workgroups(ceil(width/8), ceil(height/8), 1)

struct TerrainA {
    data: array<vec4<f32>>,  // [elevation, erodibility, diffusivity, uplift]
}

struct TerrainB {
    data: array<vec4<f32>>,  // [drainage, sediment, water, activity_f32]
}

struct DeltaBuffer {
    deltas: array<f32>,       // Δh from CPU FastScape solver
}

@group(0) @binding(0) var<storage, read_write> terrain_a: TerrainA;
@group(0) @binding(1) var<storage, read_write> terrain_b: TerrainB;
@group(0) @binding(2) var<storage, read> deltas: DeltaBuffer;
@group(0) @binding(3) var<storage, read> activity_mask: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn apply_terrain_deltas(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let x = wg_id.x * 8u + local_id.x;
    let y = wg_id.y * 8u + local_id.y;
    let width: u32 = /* push_constant */ 512u;
    
    if x >= width { return; }
    
    let idx = y * width + x;
    
    // Bitmask early-out — skip inactive columns
    let mask_word = activity_mask[idx / 32u];
    if ((mask_word >> (idx % 32u)) & 1u) == 0u { return; }
    
    // Apply elevation delta
    var a = terrain_a.data[idx];
    a.x = max(0.0, a.x + deltas.deltas[idx]);
    terrain_a.data[idx] = a;
}
"#
    }
}

/// CPU-parallel compute pipeline (mirrors GPU dispatch structure)
pub struct ComputePipeline {
    pub buffers: TerrainBuffers,
    /// Activity bitmask for sparse skipping
    pub activity: Vec<u32>,
}

impl ComputePipeline {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        let n_words = (n + 31) / 32;
        let activity = vec![u32::MAX; n_words]; // all active initially
        Self {
            buffers: TerrainBuffers::new(width, height),
            activity,
        }
    }

    /// Apply elevation deltas in parallel (GPU: compute shader dispatch)
    pub fn apply_elevation_deltas(&mut self, deltas: &[f32]) {
        let activity = &self.activity;
        self.buffers.elevation
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, elev)| {
                // Bitmask check
                let word = i / 32;
                let bit = i % 32;
                if (activity.get(word).copied().unwrap_or(0) >> bit) & 1 == 0 {
                    return;
                }
                *elev = (*elev + deltas[i]).max(0.0);
            });
    }

    /// Parallel diffusion step (matches GPU workgroup_size(8,8,1) pattern)
    pub fn parallel_diffusion_apply(&mut self, kd: &[f32], laplacian: &[f32], dt: f32) {
        let n = self.buffers.elevation.len();
        let activity = &self.activity;
        self.buffers.elevation
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, elev)| {
                let word = i / 32;
                let bit = i % 32;
                if (activity.get(word).copied().unwrap_or(0) >> bit) & 1 == 0 {
                    return;
                }
                if i < kd.len() && i < laplacian.len() {
                    *elev += kd[i] * laplacian[i] * dt;
                }
            });
    }

    /// Deactivate columns below sea level with no agents
    pub fn update_activity_mask(&mut self, sea_level: f32) {
        let elev = &self.buffers.elevation;
        let width_words = (elev.len() + 31) / 32;
        self.activity = vec![0u32; width_words];
        for (i, &e) in elev.iter().enumerate() {
            if e > sea_level {
                let word = i / 32;
                let bit = i % 32;
                self.activity[word] |= 1 << bit;
            }
        }
    }

    pub fn active_count(&self) -> usize {
        self.activity.iter().map(|w| w.count_ones() as usize).sum()
    }
}

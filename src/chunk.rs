use bytemuck::{Pod, Zeroable};

pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_AREA: usize = CHUNK_SIZE * CHUNK_SIZE;
pub const CHUNK_VOL:  usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

/// Material types - will expand with geology later
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Voxel {
    Air   = 0,
    Stone = 1,
    Soil  = 2,
    Grass = 3,
    Sand  = 4,
    Rock  = 5,
}

impl Voxel {
    pub fn is_solid(self) -> bool {
        !matches!(self, Voxel::Air)
    }

    /// RGB color for this material
    pub fn color(self) -> [f32; 3] {
        match self {
            Voxel::Air   => [0.0, 0.0, 0.0],
            Voxel::Stone => [0.55, 0.53, 0.50],
            Voxel::Soil  => [0.48, 0.34, 0.22],
            Voxel::Grass => [0.30, 0.55, 0.20],
            Voxel::Sand  => [0.82, 0.76, 0.55],
            Voxel::Rock  => [0.42, 0.40, 0.38],
        }
    }
}

pub struct Chunk {
    pub voxels: Box<[Voxel; CHUNK_VOL]>,
    pub dirty: bool,
}

impl Chunk {
    pub fn empty() -> Self {
        Self {
            voxels: Box::new([Voxel::Air; CHUNK_VOL]),
            dirty: true,
        }
    }

    #[inline]
    pub fn idx(x: usize, y: usize, z: usize) -> usize {
        x + y * CHUNK_SIZE + z * CHUNK_AREA
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        self.voxels[Self::idx(x, y, z)]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, v: Voxel) {
        self.voxels[Self::idx(x, y, z)] = v;
        self.dirty = true;
    }
}

// ─── Vertex layout ─────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color:    [f32; 3],
    pub normal:   [f32; 3],
}

impl Vertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Float32x3, // position
        1 => Float32x3, // color
        2 => Float32x3, // normal
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

// ─── Greedy mesh builder ────────────────────────────────────────────────────

const FACES: [([f32; 3], [[f32; 3]; 4]); 6] = [
    // +X
    ([1.0, 0.0, 0.0], [[1.0,0.0,0.0],[1.0,1.0,0.0],[1.0,1.0,1.0],[1.0,0.0,1.0]]),
    // -X
    ([-1.0,0.0,0.0], [[0.0,0.0,1.0],[0.0,1.0,1.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]),
    // +Y
    ([0.0, 1.0, 0.0], [[0.0,1.0,0.0],[0.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,0.0]]),
    // -Y
    ([0.0,-1.0, 0.0], [[0.0,0.0,1.0],[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,0.0,1.0]]),
    // +Z
    ([0.0, 0.0, 1.0], [[1.0,0.0,1.0],[1.0,1.0,1.0],[0.0,1.0,1.0],[0.0,0.0,1.0]]),
    // -Z
    ([0.0, 0.0,-1.0], [[0.0,0.0,0.0],[0.0,1.0,0.0],[1.0,1.0,0.0],[1.0,0.0,0.0]]),
];

const NEIGHBOR_OFFSETS: [[i32; 3]; 6] = [
    [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
];

pub fn build_mesh(chunk: &Chunk, chunk_world: [f32; 3]) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32>  = Vec::new();
    let cs = CHUNK_SIZE as i32;

    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let vox = chunk.get(x, y, z);
                if !vox.is_solid() { continue; }

                let color = vox.color();

                for (face_idx, (normal, corners)) in FACES.iter().enumerate() {
                    let nb = NEIGHBOR_OFFSETS[face_idx];
                    let nx = x as i32 + nb[0];
                    let ny = y as i32 + nb[1];
                    let nz = z as i32 + nb[2];

                    // Expose face if neighbor is out of bounds or air
                    let exposed = if nx < 0 || ny < 0 || nz < 0
                                  || nx >= cs || ny >= cs || nz >= cs {
                        true
                    } else {
                        !chunk.get(nx as usize, ny as usize, nz as usize).is_solid()
                    };

                    if !exposed { continue; }

                    // Simple ambient occlusion shading by face direction
                    let ao = match face_idx {
                        2 => 1.0_f32,  // top
                        3 => 0.45,     // bottom
                        0 | 1 => 0.75, // sides X
                        _ => 0.65,     // sides Z
                    };

                    let shaded = [color[0]*ao, color[1]*ao, color[2]*ao];
                    let base_idx = verts.len() as u32;

                    for c in corners {
                        verts.push(Vertex {
                            position: [
                                chunk_world[0] + x as f32 + c[0],
                                chunk_world[1] + y as f32 + c[1],
                                chunk_world[2] + z as f32 + c[2],
                            ],
                            color: shaded,
                            normal: *normal,
                        });
                    }

                    // Two triangles per face
                    indices.extend_from_slice(&[
                        base_idx, base_idx+1, base_idx+2,
                        base_idx, base_idx+2, base_idx+3,
                    ]);
                }
            }
        }
    }

    (verts, indices)
}

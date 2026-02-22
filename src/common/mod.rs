/// Common types and constants for the voxel engine.

/// World-space coordinate (integer voxel grid position).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IVec3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl IVec3 {
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };
}

/// Floating-point world position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn distance_sq(&self, other: &Vec3) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }

    pub fn distance(&self, other: &Vec3) -> f64 {
        self.distance_sq(other).sqrt()
    }
}

/// 2D grid coordinate for the heightmap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IVec2 {
    pub x: i32,
    pub y: i32,
}

impl IVec2 {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// Material/voxel type packed into a u16 for memory efficiency.
/// Upper 8 bits: material class, lower 8 bits: material variant/metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct VoxelMaterial(pub u16);

impl VoxelMaterial {
    pub const AIR: Self = Self(0);
    pub const BEDROCK: Self = Self(1);
    pub const SOIL: Self = Self(2);
    pub const ROCK: Self = Self(3);
    pub const WATER: Self = Self(4);
    pub const SEDIMENT: Self = Self(5);

    /// Material class (upper 8 bits).
    pub const fn class(&self) -> u8 {
        (self.0 >> 8) as u8
    }

    /// Material variant (lower 8 bits).
    pub const fn variant(&self) -> u8 {
        (self.0 & 0xFF) as u8
    }

    pub const fn from_class_variant(class: u8, variant: u8) -> Self {
        Self(((class as u16) << 8) | (variant as u16))
    }

    pub const fn is_air(&self) -> bool {
        self.0 == 0
    }

    pub const fn is_solid(&self) -> bool {
        self.0 != 0 && self.0 != Self::WATER.0
    }
}

/// Axis-Aligned Bounding Box for spatial queries.
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn contains_point(&self, p: &Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    pub fn center(&self) -> Vec3 {
        Vec3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    pub fn size(&self) -> Vec3 {
        Vec3::new(
            self.max.x - self.min.x,
            self.max.y - self.min.y,
            self.max.z - self.min.z,
        )
    }
}

/// Configuration for the engine, set at initialization.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Global heightmap resolution (e.g., 1024 = 1024x1024 grid).
    pub global_resolution: u32,
    /// Maximum depth of the SVO (determines finest voxel resolution).
    /// A depth of 8 = 256^3, depth of 10 = 1024^3.
    pub svo_max_depth: u8,
    /// Size of a single voxel at the finest SVO level, in world units.
    pub voxel_size: f64,
    /// Radius (in world units) around the player where full 3D voxels are instantiated.
    pub local_radius: f64,
    /// Number of LOD tiers for the transition zone between local and global.
    pub lod_tiers: u8,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            global_resolution: 1024,
            svo_max_depth: 8, // 256^3 local zone by default
            voxel_size: 1.0,
            local_radius: 256.0,
            lod_tiers: 4,
        }
    }
}

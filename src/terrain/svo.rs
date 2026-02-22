// ============================================================================
// Sparse Voxel Octree (SVO) — Phase 1 Stub
//
// Full implementation in Phase 2+. For now this provides the interface
// that the compute pipeline will target, plus a simple dense fallback.
//
// The SVO allows O(log n) skipping of inactive regions:
//   - Deep ocean (below sea level with no active agents)
//   - Bare bedrock (no soil, no biological activity)
//   - Empty air
//
// This is the "Body without Organs" data structure — the underlying
// potential space before stratification occurs.
// ============================================================================

use crate::ecs::components::MaterialId;

/// A node in the octree. Either a leaf (holds a single material)
/// or an internal node (holds 8 children).
#[derive(Debug, Clone)]
pub enum SvoNode {
    /// Homogeneous node — all children are the same material
    Leaf(MaterialId),
    /// Mixed node — children vary
    Internal(Box<[SvoNode; 8]>),
    /// Empty air (most common case — gets culled immediately)
    Empty,
}

impl SvoNode {
    pub fn is_empty(&self) -> bool {
        matches!(self, SvoNode::Empty)
    }

    pub fn is_homogeneous(&self) -> bool {
        matches!(self, SvoNode::Leaf(_) | SvoNode::Empty)
    }
}

/// Phase 1 stub: wraps a flat dense array with an SVO interface.
/// Real octree construction deferred to Phase 2 GPU integration.
pub struct SparseVoxelOctree {
    pub width:  u32,
    pub height: u32,
    pub depth:  u32,
    /// Dense fallback storage for Phase 1
    voxels: Vec<MaterialId>,
}

impl SparseVoxelOctree {
    pub fn new_dense(width: u32, height: u32, depth: u32) -> Self {
        let n = (width * height * depth) as usize;
        Self {
            width, height, depth,
            voxels: vec![MaterialId::Air; n],
        }
    }

    pub fn get(&self, x: u32, y: u32, z: u32) -> MaterialId {
        let idx = (z * self.height * self.width + y * self.width + x) as usize;
        self.voxels.get(idx).copied().unwrap_or(MaterialId::Air)
    }

    pub fn set(&mut self, x: u32, y: u32, z: u32, mat: MaterialId) {
        let idx = (z * self.height * self.width + y * self.width + x) as usize;
        if idx < self.voxels.len() {
            self.voxels[idx] = mat;
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.voxels.len() * std::mem::size_of::<MaterialId>()
    }

    /// Count of non-air voxels
    pub fn active_voxel_count(&self) -> usize {
        self.voxels.iter().filter(|&&m| m != MaterialId::Air).count()
    }
}

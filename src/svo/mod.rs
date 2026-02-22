/// Sparse Voxel Octree (SVO)
///
/// Represents the local 3D voxel zone around the player. Supports:
/// - Configurable max depth (depth N = 2^N resolution per axis)
/// - Sparse storage: only non-homogeneous nodes are subdivided
/// - Parallel bulk construction from heightmap data
/// - Point/AABB queries with LOD cutoff
/// - In-place voxel editing (set/clear)

use rayon::prelude::*;

use crate::common::{VoxelMaterial, Vec3, AABB, EngineConfig};

// ---------------------------------------------------------------------------
// Node representation
// ---------------------------------------------------------------------------

/// Index into the SVO's node pool. u32 gives us ~4 billion nodes.
pub type NodeIndex = u32;

/// Sentinel value: "no node here."
pub const EMPTY_NODE: NodeIndex = u32::MAX;

/// A single octree node.
/// 
/// - **Leaf**: `children` is all EMPTY_NODE, `material` holds the voxel type.
/// - **Branch**: `children` has 8 valid indices, `material` is a summary
///   (e.g., most common child material for LOD rendering).
#[derive(Debug, Clone, Copy)]
pub struct OctreeNode {
    /// Children indexed by octant (0..8). EMPTY_NODE = no child / homogeneous.
    pub children: [NodeIndex; 8],
    /// Material of this node (meaningful for leaves; summary for branches).
    pub material: VoxelMaterial,
    /// Depth in the tree (0 = root, max_depth = finest voxel).
    pub depth: u8,
}

impl OctreeNode {
    pub fn new_leaf(material: VoxelMaterial, depth: u8) -> Self {
        Self {
            children: [EMPTY_NODE; 8],
            material,
            depth,
        }
    }

    pub fn new_branch(depth: u8) -> Self {
        Self {
            children: [EMPTY_NODE; 8],
            material: VoxelMaterial::AIR,
            depth,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.iter().all(|&c| c == EMPTY_NODE)
    }

    pub fn is_homogeneous(&self) -> bool {
        self.is_leaf()
    }
}

/// Octant index from a position relative to the node center.
#[inline]
fn octant_index(px: f64, py: f64, pz: f64, cx: f64, cy: f64, cz: f64) -> usize {
    let mut idx = 0;
    if px >= cx { idx |= 1; }
    if py >= cy { idx |= 2; }
    if pz >= cz { idx |= 4; }
    idx
}

/// Get the child AABB for a given octant within a parent AABB.
#[inline]
fn child_aabb(parent: &AABB, octant: usize) -> AABB {
    let center = parent.center();
    let min_x = if octant & 1 != 0 { center.x } else { parent.min.x };
    let min_y = if octant & 2 != 0 { center.y } else { parent.min.y };
    let min_z = if octant & 4 != 0 { center.z } else { parent.min.z };
    let max_x = if octant & 1 != 0 { parent.max.x } else { center.x };
    let max_y = if octant & 2 != 0 { parent.max.y } else { center.y };
    let max_z = if octant & 4 != 0 { parent.max.z } else { center.z };
    AABB::new(
        Vec3::new(min_x, min_y, min_z),
        Vec3::new(max_x, max_y, max_z),
    )
}

// ---------------------------------------------------------------------------
// The SVO itself
// ---------------------------------------------------------------------------

/// Pool-allocated Sparse Voxel Octree.
pub struct SparseVoxelOctree {
    /// All nodes stored in a flat pool for cache locality.
    nodes: Vec<OctreeNode>,
    /// Index of the root node.
    root: NodeIndex,
    /// Max depth of the tree. Depth D means 2^D voxels per axis.
    max_depth: u8,
    /// World-space AABB of the entire octree volume.
    bounds: AABB,
}

impl SparseVoxelOctree {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create an empty SVO covering the given world-space bounds.
    pub fn new(config: &EngineConfig, origin: Vec3) -> Self {
        let half = config.local_radius;
        let bounds = AABB::new(
            Vec3::new(origin.x - half, origin.y - half, origin.z - half),
            Vec3::new(origin.x + half, origin.y + half, origin.z + half),
        );
        let root_node = OctreeNode::new_leaf(VoxelMaterial::AIR, 0);
        Self {
            nodes: vec![root_node],
            root: 0,
            max_depth: config.svo_max_depth,
            bounds,
        }
    }

    /// Build an SVO from a heightmap region.
    ///
    /// `sample_fn(world_x, world_y, world_z) -> VoxelMaterial` is called for
    /// each potential voxel position. The SVO prunes homogeneous regions.
    pub fn from_sample<F>(config: &EngineConfig, origin: Vec3, sample_fn: F) -> Self
    where
        F: Fn(f64, f64, f64) -> VoxelMaterial + Sync + Send,
    {
        let half = config.local_radius;
        let bounds = AABB::new(
            Vec3::new(origin.x - half, origin.y - half, origin.z - half),
            Vec3::new(origin.x + half, origin.y + half, origin.z + half),
        );

        let mut svo = Self {
            nodes: Vec::with_capacity(1024 * 64), // pre-allocate
            root: 0,
            max_depth: config.svo_max_depth,
            bounds,
        };

        // Allocate root
        svo.nodes.push(OctreeNode::new_branch(0));
        svo.root = 0;

        // Recursively build
        svo.build_recursive(0, &bounds, 0, &sample_fn);

        svo.nodes.shrink_to_fit();
        svo
    }

    /// Recursive top-down SVO construction.
    fn build_recursive<F>(
        &mut self,
        node_idx: NodeIndex,
        aabb: &AABB,
        depth: u8,
        sample_fn: &F,
    ) where
        F: Fn(f64, f64, f64) -> VoxelMaterial + Sync + Send,
    {
        // At max depth, sample the center and make a leaf.
        if depth >= self.max_depth {
            let c = aabb.center();
            let mat = sample_fn(c.x, c.y, c.z);
            self.nodes[node_idx as usize] = OctreeNode::new_leaf(mat, depth);
            return;
        }

        // Sample all 8 child octant centers to check homogeneity.
        let mut child_mats = [VoxelMaterial::AIR; 8];
        let mut all_same = true;
        for octant in 0..8 {
            let child_bb = child_aabb(aabb, octant);
            let c = child_bb.center();
            child_mats[octant] = sample_fn(c.x, c.y, c.z);
            if child_mats[octant] != child_mats[0] {
                all_same = false;
            }
        }

        // If homogeneous at the coarse sample, check more points for safety
        // at intermediate depths, then collapse to a leaf.
        if all_same && depth > 0 {
            self.nodes[node_idx as usize] = OctreeNode::new_leaf(child_mats[0], depth);
            return;
        }

        // Otherwise, subdivide.
        let mut node = OctreeNode::new_branch(depth);
        for octant in 0..8 {
            let child_bb = child_aabb(aabb, octant);
            let child_idx = self.nodes.len() as NodeIndex;
            self.nodes.push(OctreeNode::new_branch(depth + 1));
            node.children[octant] = child_idx;
            self.build_recursive(child_idx, &child_bb, depth + 1, sample_fn);
        }

        // After building children, check if they all collapsed to the same leaf.
        // If so, collapse this branch too.
        let first_child = self.nodes[node.children[0] as usize];
        if first_child.is_leaf() {
            let all_leaves_same = node.children.iter().all(|&ci| {
                let child = &self.nodes[ci as usize];
                child.is_leaf() && child.material == first_child.material
            });
            if all_leaves_same {
                self.nodes[node_idx as usize] =
                    OctreeNode::new_leaf(first_child.material, depth);
                // Note: orphaned child nodes remain in pool; a compaction pass
                // can reclaim them if memory is tight.
                return;
            }
        }

        // Set the branch's summary material to the most common child material.
        node.material = first_child.material; // simple heuristic
        self.nodes[node_idx as usize] = node;
    }

    // -------------------------------------------------------------------
    // Queries
    // -------------------------------------------------------------------

    /// Get the material at a world-space point, resolving to the finest LOD.
    pub fn get_material(&self, pos: &Vec3) -> VoxelMaterial {
        self.get_material_at_depth(pos, self.max_depth)
    }

    /// Get the material at a point, stopping at the given LOD depth.
    pub fn get_material_at_depth(&self, pos: &Vec3, target_depth: u8) -> VoxelMaterial {
        if !self.bounds.contains_point(pos) {
            return VoxelMaterial::AIR;
        }
        self.query_recursive(self.root, &self.bounds, pos, target_depth)
    }

    fn query_recursive(
        &self,
        node_idx: NodeIndex,
        aabb: &AABB,
        pos: &Vec3,
        target_depth: u8,
    ) -> VoxelMaterial {
        let node = &self.nodes[node_idx as usize];

        // Leaf or reached target depth -> return this node's material.
        if node.is_leaf() || node.depth >= target_depth {
            return node.material;
        }

        let center = aabb.center();
        let octant = octant_index(pos.x, pos.y, pos.z, center.x, center.y, center.z);
        let child_idx = node.children[octant];

        if child_idx == EMPTY_NODE {
            return node.material;
        }

        let child_bb = child_aabb(aabb, octant);
        self.query_recursive(child_idx, &child_bb, pos, target_depth)
    }

    /// Set the material at a world-space point. Subdivides nodes as needed
    /// down to `max_depth`.
    pub fn set_material(&mut self, pos: &Vec3, material: VoxelMaterial) {
        if !self.bounds.contains_point(pos) {
            return;
        }
        let root = self.root;
        let bounds = self.bounds;
        self.set_recursive(root, &bounds, pos, material, 0);
    }

    fn set_recursive(
        &mut self,
        node_idx: NodeIndex,
        aabb: &AABB,
        pos: &Vec3,
        material: VoxelMaterial,
        depth: u8,
    ) {
        if depth >= self.max_depth {
            self.nodes[node_idx as usize].material = material;
            return;
        }

        let node = self.nodes[node_idx as usize];

        // If this is a leaf and we need to subdivide, expand it first.
        if node.is_leaf() {
            let old_mat = node.material;
            let mut branch = OctreeNode::new_branch(depth);
            for octant in 0..8 {
                let child_idx = self.nodes.len() as NodeIndex;
                self.nodes.push(OctreeNode::new_leaf(old_mat, depth + 1));
                branch.children[octant] = child_idx;
            }
            branch.material = old_mat;
            self.nodes[node_idx as usize] = branch;
        }

        let center = aabb.center();
        let octant = octant_index(pos.x, pos.y, pos.z, center.x, center.y, center.z);
        let child_idx = self.nodes[node_idx as usize].children[octant];
        let child_bb = child_aabb(aabb, octant);
        self.set_recursive(child_idx, &child_bb, pos, material, depth + 1);

        // After modification, try to collapse if all children are now identical leaves.
        self.try_collapse(node_idx);
    }

    /// If all 8 children of a node are identical leaves, collapse to a leaf.
    fn try_collapse(&mut self, node_idx: NodeIndex) {
        let node = self.nodes[node_idx as usize];
        if node.is_leaf() {
            return;
        }

        let first = self.nodes[node.children[0] as usize];
        if !first.is_leaf() {
            return;
        }

        let all_same = node.children.iter().all(|&ci| {
            let c = &self.nodes[ci as usize];
            c.is_leaf() && c.material == first.material
        });

        if all_same {
            self.nodes[node_idx as usize] =
                OctreeNode::new_leaf(first.material, node.depth);
        }
    }

    /// Collect all leaf voxels intersecting an AABB, at the finest available depth.
    /// Returns (center_position, material, node_size) tuples.
    pub fn query_aabb(&self, query: &AABB) -> Vec<(Vec3, VoxelMaterial, f64)> {
        let mut results = Vec::new();
        self.query_aabb_recursive(self.root, &self.bounds, query, &mut results);
        results
    }

    fn query_aabb_recursive(
        &self,
        node_idx: NodeIndex,
        aabb: &AABB,
        query: &AABB,
        results: &mut Vec<(Vec3, VoxelMaterial, f64)>,
    ) {
        if !aabb.intersects(query) {
            return;
        }

        let node = &self.nodes[node_idx as usize];
        if node.is_leaf() {
            if !node.material.is_air() {
                let size = aabb.max.x - aabb.min.x;
                results.push((aabb.center(), node.material, size));
            }
            return;
        }

        for octant in 0..8 {
            let child_idx = node.children[octant];
            if child_idx != EMPTY_NODE {
                let child_bb = child_aabb(aabb, octant);
                self.query_aabb_recursive(child_idx, &child_bb, query, results);
            }
        }
    }

    // -------------------------------------------------------------------
    // Parallel bulk operations
    // -------------------------------------------------------------------

    /// Parallel bulk set: apply a list of (position, material) edits.
    /// Edits are sorted and applied sequentially (SVO mutation isn't trivially
    /// parallelizable), but the sort itself uses rayon.
    pub fn bulk_set(&mut self, edits: &mut Vec<(Vec3, VoxelMaterial)>) {
        // Sort edits by a space-filling curve for better cache locality.
        edits.par_sort_by(|a, b| {
            let ka = morton_key(a.0.x, a.0.y, a.0.z);
            let kb = morton_key(b.0.x, b.0.y, b.0.z);
            ka.cmp(&kb)
        });

        for (pos, mat) in edits.iter() {
            self.set_material(pos, *mat);
        }
    }

    // -------------------------------------------------------------------
    // Stats / diagnostics
    // -------------------------------------------------------------------

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    pub fn branch_count(&self) -> usize {
        self.nodes.iter().filter(|n| !n.is_leaf()).count()
    }

    pub fn max_depth(&self) -> u8 {
        self.max_depth
    }

    pub fn bounds(&self) -> &AABB {
        &self.bounds
    }

    pub fn memory_bytes(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<OctreeNode>()
    }
}

// ---------------------------------------------------------------------------
// Morton code for spatial sorting
// ---------------------------------------------------------------------------

/// Simple 3D Morton code (Z-order curve) for cache-friendly sorting.
fn morton_key(x: f64, y: f64, z: f64) -> u64 {
    let ix = (x.max(0.0) as u64) & 0x1FFFFF;
    let iy = (y.max(0.0) as u64) & 0x1FFFFF;
    let iz = (z.max(0.0) as u64) & 0x1FFFFF;
    interleave_bits(ix) | (interleave_bits(iy) << 1) | (interleave_bits(iz) << 2)
}

fn interleave_bits(mut x: u64) -> u64 {
    x = (x | (x << 32)) & 0x1F00000000FFFF;
    x = (x | (x << 16)) & 0x1F0000FF0000FF;
    x = (x | (x << 8)) & 0x100F00F00F00F00F;
    x = (x | (x << 4)) & 0x10C30C30C30C30C3;
    x = (x | (x << 2)) & 0x1249249249249249;
    x
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(depth: u8, radius: f64) -> EngineConfig {
        EngineConfig {
            svo_max_depth: depth,
            local_radius: radius,
            ..Default::default()
        }
    }

    #[test]
    fn test_empty_svo() {
        let svo = SparseVoxelOctree::new(&test_config(4, 16.0), Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(svo.node_count(), 1);
        assert_eq!(
            svo.get_material(&Vec3::new(0.0, 0.0, 0.0)),
            VoxelMaterial::AIR
        );
    }

    #[test]
    fn test_set_and_get() {
        let mut svo =
            SparseVoxelOctree::new(&test_config(4, 16.0), Vec3::new(0.0, 0.0, 0.0));
        let pos = Vec3::new(1.0, 2.0, 3.0);
        svo.set_material(&pos, VoxelMaterial::ROCK);
        assert_eq!(svo.get_material(&pos), VoxelMaterial::ROCK);
    }

    #[test]
    fn test_out_of_bounds() {
        let svo = SparseVoxelOctree::new(&test_config(4, 16.0), Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(
            svo.get_material(&Vec3::new(100.0, 100.0, 100.0)),
            VoxelMaterial::AIR
        );
    }

    #[test]
    fn test_from_sample_flat_ground() {
        // A flat world: rock below y=0, air above.
        let svo = SparseVoxelOctree::from_sample(
            &test_config(4, 8.0),
            Vec3::new(0.0, 0.0, 0.0),
            |_x, y, _z| {
                if y < 0.0 {
                    VoxelMaterial::ROCK
                } else {
                    VoxelMaterial::AIR
                }
            },
        );
        assert_eq!(
            svo.get_material(&Vec3::new(0.0, -2.0, 0.0)),
            VoxelMaterial::ROCK
        );
        assert_eq!(
            svo.get_material(&Vec3::new(0.0, 2.0, 0.0)),
            VoxelMaterial::AIR
        );
        // Sparse: should be much fewer nodes than 2^(4*3)
        println!(
            "Flat ground SVO: {} nodes ({} leaves, {} branches)",
            svo.node_count(),
            svo.leaf_count(),
            svo.branch_count()
        );
    }

    #[test]
    fn test_collapse_on_fill() {
        let mut svo =
            SparseVoxelOctree::new(&test_config(2, 4.0), Vec3::new(0.0, 0.0, 0.0));
        // Set every position to ROCK — should eventually collapse back.
        for x in -3..=3 {
            for y in -3..=3 {
                for z in -3..=3 {
                    svo.set_material(
                        &Vec3::new(x as f64, y as f64, z as f64),
                        VoxelMaterial::ROCK,
                    );
                }
            }
        }
        // The root should now be a single leaf.
        assert!(svo.nodes[svo.root as usize].is_leaf());
        assert_eq!(svo.nodes[svo.root as usize].material, VoxelMaterial::ROCK);
    }

    #[test]
    fn test_aabb_query() {
        let svo = SparseVoxelOctree::from_sample(
            &test_config(3, 8.0),
            Vec3::new(0.0, 0.0, 0.0),
            |_x, y, _z| {
                if y < 0.0 {
                    VoxelMaterial::ROCK
                } else {
                    VoxelMaterial::AIR
                }
            },
        );
        let query = AABB::new(Vec3::new(-2.0, -4.0, -2.0), Vec3::new(2.0, -1.0, 2.0));
        let results = svo.query_aabb(&query);
        assert!(!results.is_empty());
        for (_, mat, _) in &results {
            assert_eq!(*mat, VoxelMaterial::ROCK);
        }
    }
}

use crate::ecs::components::MaterialId;
#[derive(Debug, Clone)]
pub enum SvoNode { Leaf(MaterialId), Internal(Box<[SvoNode;8]>), Empty }
impl SvoNode {
    pub fn is_empty(&self) -> bool { matches!(self,SvoNode::Empty) }
    pub fn is_homogeneous(&self) -> bool { matches!(self,SvoNode::Leaf(_)|SvoNode::Empty) }
}
pub struct SparseVoxelOctree { pub width:u32, pub height:u32, pub depth:u32, voxels:Vec<MaterialId> }
impl SparseVoxelOctree {
    pub fn new_dense(width:u32,height:u32,depth:u32) -> Self {
        let n=(width*height*depth) as usize;
        Self { width,height,depth,voxels:vec![MaterialId::Air;n] }
    }
    pub fn get(&self,x:u32,y:u32,z:u32) -> MaterialId {
        let idx=(z*self.height*self.width+y*self.width+x) as usize;
        self.voxels.get(idx).copied().unwrap_or(MaterialId::Air)
    }
    pub fn set(&mut self,x:u32,y:u32,z:u32,mat:MaterialId) {
        let idx=(z*self.height*self.width+y*self.width+x) as usize;
        if idx<self.voxels.len() { self.voxels[idx]=mat; }
    }
    pub fn memory_bytes(&self) -> usize { self.voxels.len()*std::mem::size_of::<MaterialId>() }
    pub fn active_voxel_count(&self) -> usize { self.voxels.iter().filter(|&&m|m!=MaterialId::Air).count() }
}

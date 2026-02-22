use rayon::prelude::*;
#[derive(Debug)]
pub struct TerrainBuffers {
    pub elevation:Vec<f32>, pub erodibility:Vec<f32>, pub diffusivity:Vec<f32>,
    pub uplift:Vec<f32>, pub drainage:Vec<f32>, pub sediment:Vec<f32>,
    pub width:usize, pub height:usize,
}
impl TerrainBuffers {
    pub fn new(width:usize,height:usize) -> Self {
        let n=width*height;
        Self { elevation:vec![0.0;n],erodibility:vec![1e-5;n],diffusivity:vec![0.01;n],
            uplift:vec![0.0;n],drainage:vec![1.0;n],sediment:vec![0.0;n],width,height }
    }
    pub fn n_elements(&self) -> usize { self.elevation.len() }
    pub fn pack_vec4_terrain_a(&self) -> Vec<[f32;4]> {
        self.elevation.iter().zip(self.erodibility.iter())
            .zip(self.diffusivity.iter().zip(self.uplift.iter()))
            .map(|((e,kf),(kd,u))|[*e,*kf,*kd,*u]).collect()
    }
    pub fn wgsl_terrain_update_shader() -> &'static str { "" }
}
pub struct ComputePipeline { pub buffers: TerrainBuffers, pub activity: Vec<u32> }
impl ComputePipeline {
    pub fn new(width:usize,height:usize) -> Self {
        let n=width*height; let n_words=(n+31)/32;
        Self { buffers:TerrainBuffers::new(width,height), activity:vec![u32::MAX;n_words] }
    }
    pub fn apply_elevation_deltas(&mut self,deltas:&[f32]) {
        let activity=&self.activity;
        self.buffers.elevation.par_iter_mut().enumerate().for_each(|(i,elev)|{
            let word=i/32; let bit=i%32;
            if (activity.get(word).copied().unwrap_or(0)>>bit)&1==0 { return; }
            *elev=(*elev+deltas[i]).max(0.0);
        });
    }
    pub fn parallel_diffusion_apply(&mut self,kd:&[f32],laplacian:&[f32],dt:f32) {
        let n=self.buffers.elevation.len(); let activity=&self.activity;
        self.buffers.elevation.par_iter_mut().enumerate().for_each(|(i,elev)|{
            let word=i/32; let bit=i%32;
            if (activity.get(word).copied().unwrap_or(0)>>bit)&1==0 { return; }
            if i<kd.len()&&i<laplacian.len() { *elev+=kd[i]*laplacian[i]*dt; }
        });
    }
    pub fn update_activity_mask(&mut self,sea_level:f32) {
        let elev=&self.buffers.elevation;
        let width_words=(elev.len()+31)/32;
        self.activity=vec![0u32;width_words];
        for (i,&e) in elev.iter().enumerate() {
            if e>sea_level { let word=i/32; let bit=i%32; self.activity[word]|=1<<bit; }
        }
    }
    pub fn active_count(&self) -> usize { self.activity.iter().map(|w|w.count_ones() as usize).sum() }
}

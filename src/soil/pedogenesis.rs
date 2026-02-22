use rayon::prelude::*;
pub const C1_DEFAULT:f32=0.002; pub const K1_DEFAULT:f32=3.0;
pub const C2_DEFAULT:f32=0.001; pub const K2_DEFAULT:f32=0.05;
pub const BETA_F:f32=2.5; pub const BETA_D:f32=1.8;
#[derive(Debug, Clone, Copy)]
pub struct PedogenesisState {
    pub s:f32, pub dp_dt:f32, pub dr_dt:f32, pub lyapunov_acc:f32,
}
impl PedogenesisState {
    pub fn new(initial_s:f32) -> Self { Self{s:initial_s.clamp(0.001,0.999),dp_dt:0.0,dr_dt:0.0,lyapunov_acc:0.0} }
    pub fn compute_rates(&mut self,c1:f32,k1:f32,c2:f32,k2:f32) {
        self.dp_dt=c1*(-k1*self.s).exp(); self.dr_dt=c2*(-k2/self.s.max(1e-4)).exp();
    }
    pub fn step(&mut self,params:&PedogenesisParams,env:&PedoEnv,dt_years:f64) {
        let dt=dt_years as f32;
        let c1_eff=params.c1*env.veg_cover*env.growth_multiplier.max(0.1);
        let c2_eff=params.c2*(1.0+env.erosion_intensity*5.0);
        
        // --- NEW SUB-STEPPING LOGIC ---
        let max_dt = 10.0; // Maximum years per integration step
        let steps = (dt / max_dt).ceil() as usize;
        let step_dt = dt / (steps.max(1) as f32);

        for _ in 0..steps {
            self.compute_rates(c1_eff,params.k1,c2_eff,params.k2);
            let ds_dt=self.dp_dt-self.dr_dt;
            self.s=(self.s+ds_dt*step_dt).clamp(0.001,0.999);
            self.lyapunov_acc+=(self.dp_dt-self.dr_dt).abs()*step_dt*0.001;
        }
        // ------------------------------
    }
    pub fn modulated_kf(&self,kf_base:f32) -> f32 { kf_base*(-BETA_F*self.s).exp() }
    pub fn modulated_kd(&self,kd_base:f32) -> f32 { kd_base*(1.0+BETA_D*self.s) }
}
impl Default for PedogenesisState { fn default() -> Self { Self::new(0.1) } }
#[derive(Debug, Clone, Copy)]
pub struct PedogenesisParams { pub c1:f32,pub k1:f32,pub c2:f32,pub k2:f32 }
impl Default for PedogenesisParams {
    fn default() -> Self { Self{c1:C1_DEFAULT,k1:K1_DEFAULT,c2:C2_DEFAULT,k2:K2_DEFAULT} }
}
impl PedogenesisParams {
    pub fn tropical() -> Self { Self{c1:0.003,k1:3.5,c2:0.0015,k2:0.04} }
    pub fn cold_arid() -> Self { Self{c1:0.0008,k1:2.5,c2:0.0005,k2:0.08} }
    pub fn volcanic() -> Self { Self{c1:0.004,k1:4.0,c2:0.001,k2:0.03} }
}
#[derive(Debug, Clone, Copy)]
pub struct PedoEnv { pub veg_cover:f32,pub growth_multiplier:f32,pub erosion_intensity:f32 }
impl Default for PedoEnv { fn default() -> Self { Self{veg_cover:0.7,growth_multiplier:0.6,erosion_intensity:0.0} } }
pub struct PedogenesisSolver { pub grid_width:usize,pub grid_height:usize }
impl PedogenesisSolver {
    pub fn new(w:usize,h:usize) -> Self { Self{grid_width:w,grid_height:h} }
    pub fn initialize(&self,elevations:&[f32],sea_level:f32) -> Vec<PedogenesisState> {
        let n=self.grid_width*self.grid_height;
        (0..n).map(|i|{
            let elev=elevations.get(i).copied().unwrap_or(0.0);
            let s=if elev<=sea_level{0.05}else{let norm=(elev/2000.0).min(1.0);0.5-norm*0.35};
            PedogenesisState::new(s)
        }).collect()
    }
    pub fn step_epoch(&self,states:&mut Vec<PedogenesisState>,params:&[PedogenesisParams],envs:&[PedoEnv],kf_base:&[f32],kd_base:&[f32],dt_years:f64) -> (Vec<f32>,Vec<f32>) {
        states.par_iter_mut().zip(params.par_iter()).zip(envs.par_iter())
            .for_each(|((state,param),env)|{state.step(param,env,dt_years);});
        let kf_mod=states.iter().zip(kf_base.iter()).map(|(s,&kf)|s.modulated_kf(kf)).collect();
        let kd_mod=states.iter().zip(kd_base.iter()).map(|(s,&kd)|s.modulated_kd(kd)).collect();
        (kf_mod,kd_mod)
    }
    pub fn build_envs(&self,growth_multipliers:&[f32],delta_h:&[f32],cell_size_m:f32) -> Vec<PedoEnv> {
        let n=self.grid_width*self.grid_height;
        (0..n).map(|i|{
            let gm=growth_multipliers.get(i).copied().unwrap_or(0.5);
            let dh=delta_h.get(i).copied().unwrap_or(0.0);
            let erosion_intensity=(dh.abs()/cell_size_m*1000.0).min(1.0);
            let veg_cover=(gm*1.2).min(1.0);
            PedoEnv{veg_cover,growth_multiplier:gm,erosion_intensity}
        }).collect()
    }
    pub fn mean_s(&self,states:&[PedogenesisState]) -> f32 { states.iter().map(|s|s.s).sum::<f32>()/states.len().max(1) as f32 }
    pub fn regressive_count(&self,states:&[PedogenesisState]) -> usize { states.iter().filter(|s|s.dr_dt>s.dp_dt).count() }
}

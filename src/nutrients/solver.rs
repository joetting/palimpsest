use rayon::prelude::*;
use crate::nutrients::pools::{
    NutrientColumn, NutrientLayer,
    K_W_P, PHI_P, K_OCC, K_MIN_P, K_LIT_P, K_UPTAKE_P, K_LEACH_P, P_TECTONIC_INPUT,
    K_W_K, K_FIX_BASE, K_REL, K_UPTAKE_K, K_LIT_K, K_LEACH_K, defac,
};
#[derive(Debug, Clone, Copy)]
pub struct ColumnEnv {
    pub temp_c:f32, pub moisture:f32, pub runoff_m_yr:f32,
    pub flooded:bool, pub flood_p_input:f32, pub delta_h_m:f32, pub veg_cover:f32,
}
impl Default for ColumnEnv {
    fn default() -> Self {
        Self{temp_c:15.0,moisture:0.6,runoff_m_yr:0.4,flooded:false,flood_p_input:0.0,delta_h_m:0.0,veg_cover:0.8}
    }
}
#[inline(always)]
fn analytical_equilibrium(a:f32,b:f32,k_forward:f32,k_backward:f32,dt:f32) -> (f32,f32) {
    let total=a+b; let denom=k_forward+k_backward;
    if denom<1e-12 { return (a,b); }
    let a_eq=k_backward/denom*total;
    let alpha=1.0-(-(denom*dt)).exp();
    let a_new=a+alpha*(a_eq-a);
    let b_new=total-a_new;
    (a_new,b_new)
}
fn step_fast_reactions(layer:&mut NutrientLayer,clay_frac:f32,dt_half:f32) {
    let k_fix=K_FIX_BASE*clay_frac;
    let (k_exch_new,k_fixed_new)=analytical_equilibrium(layer.k_exch,layer.k_fixed,k_fix,K_REL,dt_half);
    layer.k_exch=k_exch_new; layer.k_fixed=k_fixed_new;
    let k_ads=K_OCC*10.0*clay_frac; let k_des=K_OCC*2.0;
    let (p_labile_new,_)=analytical_equilibrium(layer.p_labile,0.0,k_ads*0.1,k_des*0.1,dt_half);
    layer.p_labile=p_labile_new.max(0.0);
}
fn step_slow_reactions(layer:&mut NutrientLayer,clay_frac:f32,root_dens:f32,env:&ColumnEnv,dt:f64) {
    let dt=dt as f32;
    let d=defac(env.temp_c,env.moisture); let runoff=env.runoff_m_yr; let roots=root_dens;
    let dp_mineral=P_TECTONIC_INPUT-K_W_P*layer.p_mineral*d;
    let weathered_p=K_W_P*layer.p_mineral*d;
    let occlusion_flux=K_OCC*layer.p_labile*clay_frac;
    let k_min_return=(K_MIN_P*layer.p_organic*d).min(layer.p_organic/dt.max(1e-6));
    let p_leach_flux=K_LEACH_P*layer.p_labile*runoff;
    let flood_p=if env.flooded{env.flood_p_input}else{0.0};
    let canopy_p_max_flux=K_LEACH_P*layer.p_labile*env.runoff_m_yr;
    let canopy_p_trap=(env.veg_cover*0.05).min(canopy_p_max_flux*0.5);
    let dk_mineral=-K_W_K*layer.k_mineral*d;
    let weathered_k=K_W_K*layer.k_mineral*d;
    let k_leach_flux=K_LEACH_K*layer.k_exch*runoff;
    let k_fix_slow=K_FIX_BASE*0.1*layer.k_exch*clay_frac;
    let k_rel_slow=K_REL*0.1*layer.k_fixed;
    let dk_fixed=k_fix_slow-k_rel_slow;
    let p_veg_decay_factor=(-K_LIT_P*dt).exp();
    let p_lit_return=layer.p_veg*(1.0-p_veg_decay_factor);
    let p_up_actual=(K_UPTAKE_P*layer.p_labile*roots*d*dt).min(layer.p_labile*0.9);
    layer.p_veg=(layer.p_veg*p_veg_decay_factor+p_up_actual).max(0.0);
    let k_veg_decay_factor=(-K_LIT_K*dt).exp();
    let k_lit_return=layer.k_veg*(1.0-k_veg_decay_factor);
    let k_up_actual=(K_UPTAKE_K*layer.k_exch*roots*d*dt).min(layer.k_exch*0.9);
    layer.k_veg=(layer.k_veg*k_veg_decay_factor+k_up_actual).max(0.0);
    layer.p_mineral=(layer.p_mineral+dp_mineral*dt).max(0.0);
    let p_out=(occlusion_flux+p_leach_flux)*dt+p_up_actual;
    let p_in=(weathered_p*PHI_P+k_min_return+flood_p/dt.max(1e-6)+canopy_p_trap)*dt;
    layer.p_labile=(layer.p_labile-p_out.min(layer.p_labile)+p_in).max(0.0);
    layer.p_occluded+=occlusion_flux*dt;
    let p_min_actual=(k_min_return*dt).min(layer.p_organic);
    layer.p_organic=(layer.p_organic+p_lit_return-p_min_actual).max(0.0);
    layer.k_mineral=(layer.k_mineral+dk_mineral*dt).max(0.0);
    let k_out_total=(k_leach_flux+k_fix_slow)*dt+k_up_actual;
    let k_in_total=(weathered_k+k_rel_slow)*dt+k_lit_return;
    layer.k_exch=(layer.k_exch-k_out_total.min(layer.k_exch)+k_in_total).max(0.0);
    layer.k_fixed=(layer.k_fixed+dk_fixed*dt).max(0.0);
    layer.clamp_non_negative();
}
fn step_vertical_leaching(column:&mut NutrientColumn,env:&ColumnEnv,dt:f64) {
    let dt=dt as f32; let runoff=env.runoff_m_yr;
    let mut p_leachate=0.0f32; let mut k_leachate=0.0f32;
    // Capture initial surface values before loop mutates them (Bug 3 fix)
    let init_surface_p = column.layers[0].p_labile;
    let init_surface_k = column.layers[0].k_exch;
    for i in 0..8 {
        let layer=&mut column.layers[i];
        let cec_retention=(column.clay_frac*column.cec/40.0).clamp(0.0,0.90);
        let p_retained=p_leachate*cec_retention; let k_retained=k_leachate*cec_retention;
        let p_add=p_retained.min(layer.p_labile*0.5); let k_add=k_retained.min(layer.k_exch*0.5);
        layer.p_labile=(layer.p_labile+p_add).max(0.0); layer.k_exch=(layer.k_exch+k_add).max(0.0);
        p_leachate=(p_leachate-p_retained)*(1.0-K_LEACH_P*runoff*dt).max(0.0);
        k_leachate=(k_leachate-k_retained)*(1.0-K_LEACH_K*runoff*dt).max(0.0);
        p_leachate=p_leachate.min(init_surface_p);
        k_leachate=k_leachate.min(init_surface_k);
    }
}
fn step_erosion_transport(column:&mut NutrientColumn,delta_h_m:f32) {
    if delta_h_m.abs()<1e-4 { return; }
    let layer_thickness_m=0.25f32;
    if delta_h_m>0.0 {
        let layers=column.layers;
        let mean_p_labile=layers.iter().map(|l|l.p_labile).sum::<f32>()/8.0;
        let mean_k_exch=layers.iter().map(|l|l.k_exch).sum::<f32>()/8.0;
        let mean_p_organic=layers.iter().map(|l|l.p_organic).sum::<f32>()/8.0;
        let deposit_frac=(delta_h_m/layer_thickness_m).min(1.0);
        let top=&mut column.layers[0];
        top.p_labile+=mean_p_labile*deposit_frac*0.5; top.k_exch+=mean_k_exch*deposit_frac*0.5;
        top.p_organic+=mean_p_organic*deposit_frac*0.3;
    } else {
        let eroded_frac=((-delta_h_m)/layer_thickness_m).min(1.0);
        let top=&mut column.layers[0];
        top.p_labile=(top.p_labile*(1.0-eroded_frac)).max(0.0);
        top.k_exch=(top.k_exch*(1.0-eroded_frac)).max(0.0);
        top.p_organic=(top.p_organic*(1.0-eroded_frac)).max(0.0);
        top.p_veg=(top.p_veg*(1.0-eroded_frac*0.5)).max(0.0);
    }
}
pub fn update_column(column:&mut NutrientColumn,env:&ColumnEnv,dt_years:f64) {
    let dt_half=dt_years*0.5;
    for layer in &mut column.layers { step_fast_reactions(layer,column.clay_frac,dt_half as f32); }
    step_erosion_transport(column,env.delta_h_m);
    step_vertical_leaching(column,env,dt_years);
    for layer in &mut column.layers { step_fast_reactions(layer,column.clay_frac,dt_half as f32); }
    let n_substeps=((dt_years/10.0).ceil() as usize).max(1);
    let dt_sub=dt_years/n_substeps as f64;
    for _ in 0..n_substeps {
        for (i,layer) in column.layers.iter_mut().enumerate() {
            step_slow_reactions(layer,column.clay_frac,column.root_density[i],env,dt_sub);
        }
    }
}
pub struct BiogeochemSolver { pub grid_width:usize, pub grid_height:usize }
impl BiogeochemSolver {
    pub fn new(grid_width:usize,grid_height:usize) -> Self { Self{grid_width,grid_height} }
    pub fn step_epoch(&self,columns:&mut Vec<NutrientColumn>,envs:&[ColumnEnv],dt_years:f64) {
        assert_eq!(columns.len(),envs.len());
        columns.par_iter_mut().zip(envs.par_iter()).for_each(|(col,env)|{update_column(col,env,dt_years);});
    }
    pub fn build_default_envs(&self,elevations:&[f32],sea_level:f32,delta_h:&[f32],activity_mask:&[u32]) -> Vec<ColumnEnv> {
        let n=self.grid_width*self.grid_height;
        (0..n).map(|i|{
            let elev=elevations.get(i).copied().unwrap_or(0.0);
            let dh=delta_h.get(i).copied().unwrap_or(0.0);
            if elev<=sea_level { return ColumnEnv{temp_c:4.0,moisture:1.0,runoff_m_yr:0.0,flooded:false,flood_p_input:0.0,delta_h_m:0.0,veg_cover:0.0}; }
            let word=activity_mask.get(i/32).copied().unwrap_or(0);
            let active=(word>>(i%32))&1==1;
            if !active { return ColumnEnv::default(); }
            let temp_c=(20.0-elev*0.006).max(-5.0);
            let moisture=if elev<500.0{0.7}else if elev<1500.0{0.5}else{0.3};
            let runoff=moisture*0.6;
            let veg_cover=if elev>sea_level+5.0&&elev<2500.0{0.8}else{0.1};
            let flooded=dh>0.05&&elev<sea_level+50.0;
            let flood_p_input=if flooded{8.0}else{0.0};
            ColumnEnv{temp_c,moisture,runoff_m_yr:runoff,flooded,flood_p_input,delta_h_m:dh,veg_cover}
        }).collect()
    }
    pub fn growth_multipliers(&self,columns:&[NutrientColumn]) -> Vec<f32> {
        columns.iter().map(|c|c.column_growth_multiplier()).collect()
    }
    pub fn mean_surface_p_labile(&self,columns:&[NutrientColumn],activity_mask:&[u32]) -> f32 {
        let mut sum=0.0f32; let mut count=0u32;
        for (i,col) in columns.iter().enumerate() {
            let word=activity_mask.get(i/32).copied().unwrap_or(0);
            if (word>>(i%32))&1==1 { sum+=col.surface_p_labile(); count+=1; }
        }
        if count==0{0.0}else{sum/count as f32}
    }
    pub fn mean_surface_k_exch(&self,columns:&[NutrientColumn],activity_mask:&[u32]) -> f32 {
        let mut sum=0.0f32; let mut count=0u32;
        for (i,col) in columns.iter().enumerate() {
            let word=activity_mask.get(i/32).copied().unwrap_or(0);
            if (word>>(i%32))&1==1 { sum+=col.surface_k_exch(); count+=1; }
        }
        if count==0{0.0}else{sum/count as f32}
    }
}

use serde::{Deserialize, Serialize};
pub const K_W_P:f32=0.005; pub const PHI_P:f32=0.3; pub const K_OCC:f32=0.0001;
pub const K_MIN_P:f32=1.0; pub const K_LIT_P:f32=0.2; pub const K_UPTAKE_P:f32=2.0;
pub const K_LEACH_P:f32=0.005; pub const P_TECTONIC_INPUT:f32=0.001;
pub const K_W_K:f32=0.001; pub const K_FIX_BASE:f32=0.01; pub const K_REL:f32=0.001;
pub const K_UPTAKE_K:f32=4.0; pub const K_LIT_K:f32=0.7; pub const K_LEACH_K:f32=0.01;
pub const K_HALF_P:f32=5.0; pub const K_HALF_K:f32=50.0;
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NutrientLayer {
    pub p_mineral:f32, pub p_labile:f32, pub p_occluded:f32, pub p_organic:f32,
    pub k_mineral:f32, pub k_exch:f32, pub k_fixed:f32, pub k_veg:f32, pub p_veg:f32,
}
impl NutrientLayer {
    pub fn young_mineral_soil() -> Self {
        Self{p_mineral:800.0,p_labile:15.0,p_occluded:20.0,p_organic:120.0,k_mineral:15_000.0,k_exch:200.0,k_fixed:600.0,k_veg:80.0,p_veg:40.0}
    }
    pub fn old_tropical_soil() -> Self {
        Self{p_mineral:40.0,p_labile:2.0,p_occluded:350.0,p_organic:80.0,k_mineral:3_000.0,k_exch:60.0,k_fixed:200.0,k_veg:30.0,p_veg:8.0}
    }
    pub fn alluvial_floodplain() -> Self {
        Self{p_mineral:400.0,p_labile:25.0,p_occluded:60.0,p_organic:250.0,k_mineral:12_000.0,k_exch:380.0,k_fixed:800.0,k_veg:100.0,p_veg:50.0}
    }
    pub fn clamp_non_negative(&mut self) {
        self.p_mineral=self.p_mineral.max(0.0); self.p_labile=self.p_labile.max(0.0);
        self.p_occluded=self.p_occluded.max(0.0); self.p_organic=self.p_organic.max(0.0);
        self.k_mineral=self.k_mineral.max(0.0); self.k_exch=self.k_exch.max(0.0);
        self.k_fixed=self.k_fixed.max(0.0); self.k_veg=self.k_veg.max(0.0); self.p_veg=self.p_veg.max(0.0);
    }
    pub fn f_p(&self) -> f32 { self.p_labile/(self.p_labile+K_HALF_P) }
    pub fn f_k(&self) -> f32 { self.k_exch/(self.k_exch+K_HALF_K) }
    pub fn growth_multiplier(&self) -> f32 { self.f_p().min(self.f_k()) }
    pub fn total_p(&self) -> f32 { self.p_mineral+self.p_labile+self.p_occluded+self.p_organic+self.p_veg }
    pub fn total_k(&self) -> f32 { self.k_mineral+self.k_exch+self.k_fixed+self.k_veg }
}
impl Default for NutrientLayer { fn default() -> Self { Self::young_mineral_soil() } }
#[inline(always)]
pub fn defac(temp_celsius:f32,moisture_fraction:f32) -> f32 {
    let temp_factor=(0.07*(temp_celsius-20.0)).exp();
    let moist_factor=moisture_fraction.min(1.0).max(0.0);
    temp_factor*moist_factor
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NutrientColumn {
    pub layers:[NutrientLayer;8], pub clay_frac:f32,
    pub root_density:[f32;8], pub cec:f32,
}
impl NutrientColumn {
    pub fn new_young_mineral(clay_frac:f32) -> Self {
        let cec=clay_frac*35.0+5.0;
        Self{layers:[NutrientLayer::young_mineral_soil();8],clay_frac,root_density:[0.35,0.25,0.18,0.10,0.06,0.03,0.02,0.01],cec}
    }
    pub fn new_tropical(clay_frac:f32) -> Self {
        let cec=clay_frac*20.0+3.0;
        Self{layers:[NutrientLayer::old_tropical_soil();8],clay_frac,root_density:[0.40,0.28,0.15,0.08,0.05,0.02,0.01,0.01],cec}
    }
    pub fn new_alluvial(clay_frac:f32) -> Self {
        let cec=clay_frac*40.0+8.0;
        Self{layers:[NutrientLayer::alluvial_floodplain();8],clay_frac,root_density:[0.35,0.25,0.18,0.10,0.06,0.03,0.02,0.01],cec}
    }
    pub fn surface_p_labile(&self) -> f32 {
        (0..3).map(|i|self.layers[i].p_labile*self.root_density[i]).sum::<f32>()
            /self.root_density[..3].iter().sum::<f32>().max(1e-6)
    }
    pub fn surface_k_exch(&self) -> f32 {
        (0..3).map(|i|self.layers[i].k_exch*self.root_density[i]).sum::<f32>()
            /self.root_density[..3].iter().sum::<f32>().max(1e-6)
    }
    pub fn column_growth_multiplier(&self) -> f32 {
        (0..8).map(|i|self.layers[i].growth_multiplier()*self.root_density[i]).sum::<f32>()
    }
}
impl Default for NutrientColumn { fn default() -> Self { Self::new_young_mineral(0.3) } }

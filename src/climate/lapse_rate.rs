use crate::climate::thermostat::ClimateConfig;
pub const ELR: f32 = 0.0065;
pub const FROST_LINE_M: f32 = 2154.0;
pub const REFERENCE_ELEVATION_M: f32 = 0.0;
pub const PT_ALPHA: f32 = 1.26;
pub const ET_REFERENCE_M_YR: f32 = 0.60;
pub struct LapseRateEngine { pub elr:f32, pub et_reference:f32 }
impl LapseRateEngine {
    pub fn new(_config:ClimateConfig) -> Self { Self { elr:ELR, et_reference:ET_REFERENCE_M_YR } }
    #[inline] pub fn local_temp(&self,base_temp_c:f32,elevation_m:f32) -> f32 {
        base_temp_c-self.elr*(elevation_m-REFERENCE_ELEVATION_M).max(0.0)
    }
    #[inline] pub fn pet(&self,temp_c:f32) -> f32 {
        if temp_c<=0.0 { return 0.02; }
        let temp_scale=(temp_c/15.0).max(0.0).min(3.0);
        self.et_reference*PT_ALPHA*temp_scale
    }
    #[inline] pub fn runoff(&self,precip_m:f32,temp_c:f32,_elevation_m:f32) -> f32 {
        let snow_fraction=if temp_c<0.0{1.0}else{0.0};
        let liquid_precip=precip_m*(1.0-snow_fraction);
        let pet=self.pet(temp_c); let aet=pet.min(liquid_precip);
        (liquid_precip-aet).max(0.0)
    }
    #[inline] pub fn soil_moisture(&self,precip_m:f32,temp_c:f32) -> f32 {
        let pet=self.pet(temp_c);
        if pet<1e-6 { return 1.0; }
        let aridity=pet/precip_m.max(1e-6);
        (1.0/(1.0+aridity*aridity)).clamp(0.0,1.0)
    }
    pub fn snowline_elevation(&self,base_temp_c:f32) -> f32 {
        if base_temp_c<=0.0 { return 0.0; } base_temp_c/self.elr
    }
    pub fn freeze_thaw_intensity(&self,base_temp_c:f32,elevation_m:f32) -> f32 {
        let local_temp=self.local_temp(base_temp_c,elevation_m);
        let sigma=5.0f32;
        (-(local_temp/sigma).powi(2)).exp()
    }
}

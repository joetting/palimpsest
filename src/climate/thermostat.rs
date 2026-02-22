use serde::{Deserialize, Serialize};
pub const C0_BASELINE_PPM: f32 = 280.0;
pub const LAMBDA_SENSITIVITY: f32 = 3.0;
pub const T0_BASELINE_C: f32 = 14.0;
pub const C_MIN_PPM: f32 = 150.0;
pub const C_MAX_PPM: f32 = 8000.0;
pub const ICE_AGE_THRESHOLD_C: f32 = 8.0;
pub const HOTHOUSE_THRESHOLD_C: f32 = 22.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateConfig {
    pub initial_co2_ppm: f32, pub lambda: f32, pub t0_baseline: f32,
    pub plant_uptake_coeff: f32, pub animal_respiration_coeff: f32,
    pub decay_rate: f32, pub volcanic_outgas_rate: f32,
    pub wind_angle_rad: f32, pub wind_speed_ms: f32,
}
impl Default for ClimateConfig {
    fn default() -> Self {
        Self { initial_co2_ppm:C0_BASELINE_PPM,lambda:LAMBDA_SENSITIVITY,t0_baseline:T0_BASELINE_C,
            plant_uptake_coeff:0.002,animal_respiration_coeff:0.001,decay_rate:0.005,
            volcanic_outgas_rate:0.01,wind_angle_rad:0.0,wind_speed_ms:8.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateState {
    pub atmospheric_co2_ppm: f32, pub global_mean_temp_c: f32,
    pub cumulative_silicate_drawdown: f32, pub cumulative_organic_burial: f32,
    pub in_ice_age: bool, pub in_hothouse: bool, pub epoch: u64,
}
impl ClimateState {
    pub fn new(initial_co2_ppm:f32,t0_baseline:f32,lambda:f32) -> Self {
        let temp=t0_from_co2(initial_co2_ppm,t0_baseline,lambda);
        Self { atmospheric_co2_ppm:initial_co2_ppm,global_mean_temp_c:temp,
            cumulative_silicate_drawdown:0.0,cumulative_organic_burial:0.0,
            in_ice_age:temp<ICE_AGE_THRESHOLD_C,in_hothouse:temp>HOTHOUSE_THRESHOLD_C,epoch:0 }
    }
    pub fn summary(&self) -> String {
        let phase=if self.in_ice_age{"❄ ICE AGE"}else if self.in_hothouse{"🔥 HOTHOUSE"}else{"🌿 Interglacial"};
        format!("CO₂={:.0}ppm  T={:.1}°C  {}  SilicateΣ={:.1}ppm  BurialΣ={:.1}ppm",
            self.atmospheric_co2_ppm,self.global_mean_temp_c,phase,
            self.cumulative_silicate_drawdown,self.cumulative_organic_burial)
    }
}
pub fn t0_from_co2(co2_ppm:f32,t0_baseline:f32,lambda:f32) -> f32 {
    if co2_ppm<=0.0 { return t0_baseline-lambda*10.0; }
    let forcing=lambda*(co2_ppm/C0_BASELINE_PPM).log2();
    t0_baseline+forcing
}
pub struct ProserpinaThermostat { pub config:ClimateConfig, pub state:ClimateState }
impl ProserpinaThermostat {
    pub fn new(config:ClimateConfig) -> Self {
        let state=ClimateState::new(config.initial_co2_ppm,config.t0_baseline,config.lambda);
        Self { config,state }
    }
    pub fn step_epoch(&mut self,plant_biomass:f32,animal_respiration:f32,silicate_drawdown:f32,
        organic_burial:f32,anthropogenic_flush:f32,dt_years:f64) {
        let dt=dt_years as f32; let c=&self.config;
        let volcanic=c.volcanic_outgas_rate*dt;
        let respiration=(c.animal_respiration_coeff*animal_respiration+c.decay_rate*plant_biomass*0.1)*dt;
        let flush=anthropogenic_flush*dt;
        let temp_scale=photosynthesis_temp_scale(self.state.global_mean_temp_c);
        let photosynthesis=c.plant_uptake_coeff*plant_biomass*temp_scale*dt;
        let silicate=silicate_drawdown*dt;
        let burial=organic_burial*dt;
        let delta_co2=volcanic+respiration+flush-photosynthesis-silicate-burial;
        self.state.atmospheric_co2_ppm=(self.state.atmospheric_co2_ppm+delta_co2).clamp(C_MIN_PPM,C_MAX_PPM);
        self.state.global_mean_temp_c=t0_from_co2(self.state.atmospheric_co2_ppm,self.config.t0_baseline,self.config.lambda);
        self.state.cumulative_silicate_drawdown+=silicate;
        self.state.cumulative_organic_burial+=burial;
        self.state.in_ice_age=self.state.global_mean_temp_c<ICE_AGE_THRESHOLD_C;
        self.state.in_hothouse=self.state.global_mean_temp_c>HOTHOUSE_THRESHOLD_C;
        self.state.epoch+=1;
    }
    pub fn elder_god_co2_injection(&mut self,delta_ppm:f32) {
        self.state.atmospheric_co2_ppm=(self.state.atmospheric_co2_ppm+delta_ppm).clamp(C_MIN_PPM,C_MAX_PPM);
        self.state.global_mean_temp_c=t0_from_co2(self.state.atmospheric_co2_ppm,self.config.t0_baseline,self.config.lambda);
        println!("[ELDER GOD] CO₂ injection {:+.1}ppm → {:.0}ppm, T={:.1}°C",delta_ppm,self.state.atmospheric_co2_ppm,self.state.global_mean_temp_c);
    }
}
fn photosynthesis_temp_scale(temp_c:f32) -> f32 {
    let opt=18.0f32; let width=20.0f32;
    let t=-(temp_c-opt).powi(2)/(2.0*width*width);
    t.exp().clamp(0.01,1.0)
}

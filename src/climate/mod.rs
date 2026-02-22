pub mod thermostat;
pub mod lapse_rate;
pub mod orographic;
pub mod carbon_cycle;
pub use thermostat::{ClimateState, ProserpinaThermostat, ClimateConfig};
pub use lapse_rate::LapseRateEngine;
pub use orographic::{OrographicEngine, WindParams, OrographicOutput};
pub use carbon_cycle::{CarbonCycleEngine, CarbonFluxes, SilicateWeatheringParams};
use crate::compute::activity_mask::ActivityProcessor;

#[derive(Debug, Clone)]
pub struct ColumnClimate {
    pub temp_c:f32, pub precipitation_m:f32, pub runoff_m:f32,
    pub moisture:f32, pub biome:Biome,
}
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Biome {
    Ocean=0,TropicalForest=1,TemperateForest=2,Grassland=3,
    Desert=4,Tundra=5,Alpine=6,Glacier=7,
}
impl Biome {
    pub fn classify(temp_c:f32,precip_m:f32,elevation_m:f32) -> Self {
        if elevation_m<0.0{return Biome::Ocean;}
        if temp_c < -10.0{return Biome::Glacier;}
        if temp_c < 0.0&&elevation_m>1500.0{return Biome::Alpine;}
        if temp_c < 5.0{return Biome::Tundra;}
        if precip_m<0.25{return Biome::Desert;}
        if precip_m<0.6&&temp_c<20.0{return Biome::Grassland;}
        if temp_c>22.0&&precip_m>1.5{return Biome::TropicalForest;}
        Biome::TemperateForest
    }
    pub fn veg_cover(&self) -> f32 {
        match self {
            Biome::TropicalForest=>0.95,Biome::TemperateForest=>0.80,Biome::Grassland=>0.55,
            Biome::Desert=>0.05,Biome::Tundra=>0.30,Biome::Alpine=>0.15,
            Biome::Glacier|Biome::Ocean=>0.0,
        }
    }
    pub fn photosynthesis_rate(&self) -> f32 {
        match self {
            Biome::TropicalForest=>1.0,Biome::TemperateForest=>0.65,Biome::Grassland=>0.35,
            Biome::Desert=>0.05,Biome::Tundra=>0.15,Biome::Alpine=>0.10,
            Biome::Glacier|Biome::Ocean=>0.0,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Biome::TropicalForest=>"Tropical Forest",Biome::TemperateForest=>"Temperate Forest",
            Biome::Grassland=>"Grassland",Biome::Desert=>"Desert",Biome::Tundra=>"Tundra",
            Biome::Alpine=>"Alpine",Biome::Glacier=>"Glacier",Biome::Ocean=>"Ocean",
        }
    }
}
pub struct ClimateEngine {
    pub thermostat:ProserpinaThermostat, pub lapse:LapseRateEngine,
    pub orographic:OrographicEngine, pub carbon:CarbonCycleEngine,
    pub grid_width:usize, pub grid_height:usize,
    pub columns:Vec<ColumnClimate>,
}
impl ClimateEngine {
    pub fn new(grid_width:usize,grid_height:usize,config:ClimateConfig,wind:WindParams,cell_size_m:f32) -> Self {
        let n=grid_width*grid_height;
        let thermostat=ProserpinaThermostat::new(config.clone());
        let lapse=LapseRateEngine::new(config.clone());
        let orographic=OrographicEngine::new(grid_width,grid_height,wind,cell_size_m);
        let carbon=CarbonCycleEngine::new(SilicateWeatheringParams {
            erosion_amplification: 0.5,
            base_rate_ppm_yr_km2: 0.0001,
            ..SilicateWeatheringParams::default()
        });
        let default_col=ColumnClimate{temp_c:15.0,precipitation_m:0.8,runoff_m:0.4,moisture:0.6,biome:Biome::TemperateForest};
        Self { thermostat,lapse,orographic,carbon,grid_width,grid_height,columns:vec![default_col;n] }
    }
    pub fn step_epoch(&mut self,elevations:&[f32],delta_h:&[f32],plant_biomass_total:f32,animal_respiration:f32,dt_years:f64) {
        let fluxes=self.carbon.compute_fluxes(elevations,delta_h,self.grid_width,self.grid_height,&self.thermostat.state);
        self.thermostat.step_epoch(plant_biomass_total,animal_respiration,fluxes.silicate_drawdown,fluxes.organic_burial,fluxes.anthropogenic_flush,dt_years);
        let oro=self.orographic.compute(elevations);
        let base_temp=self.thermostat.state.global_mean_temp_c;
        for idx in 0..(self.grid_width*self.grid_height) {
            let elev=elevations[idx]; let precip=oro.precipitation[idx];
            let temp=self.lapse.local_temp(base_temp,elev);
            let runoff=self.lapse.runoff(precip,temp,elev);
            let moisture=(runoff/precip.max(1e-6)).clamp(0.0,1.0);
            let biome=Biome::classify(temp,precip,elev);
            self.columns[idx]=ColumnClimate{temp_c:temp,precipitation_m:precip,runoff_m:runoff,moisture,biome};
        }
    }
    pub fn summary(&self) -> String { self.thermostat.state.summary() }
    pub fn biome_counts(&self,elevations:&[f32],sea_level:f32) -> [(Biome,usize);8] {
        let mut counts=[0usize;8];
        for (i,col) in self.columns.iter().enumerate() {
            if elevations[i]>sea_level { counts[col.biome as usize]+=1; }
        }
        [(Biome::Ocean,counts[0]),(Biome::TropicalForest,counts[1]),(Biome::TemperateForest,counts[2]),
         (Biome::Grassland,counts[3]),(Biome::Desert,counts[4]),(Biome::Tundra,counts[5]),
         (Biome::Alpine,counts[6]),(Biome::Glacier,counts[7])]
    }
}

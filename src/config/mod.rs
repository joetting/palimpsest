use crate::ecs::world::World;
use crate::ecs::scheduler::{SimulationClock, StrangRunner, TemporalConfig};
use crate::terrain::fastscape::{FastScapeSolver, SplParams, TectonicForcing};
use crate::terrain::heightmap::Heightmap;
use crate::compute::gpu_stub::ComputePipeline;
use crate::ecs::components::MaterialId;
use crate::climate::{ClimateEngine, ClimateConfig, WindParams};
use crate::climate::carbon_cycle::SilicateWeatheringParams;
use crate::ecs::components::Erodibility;

pub struct SimulationConfig {
    pub grid_width:usize, pub grid_height:usize, pub cell_size_m:f32,
    pub max_elevation_m:f32, pub sea_level_m:f32, pub geo_dt_years:f64,
    pub seed:u64, pub num_cohorts:u8,
    pub climate:Option<ClimateConfig>, pub wind:Option<WindParams>,
}
impl Default for SimulationConfig {
    fn default() -> Self {
        Self { grid_width:128,grid_height:128,cell_size_m:1_000.0,max_elevation_m:400.0,
            sea_level_m:0.0,geo_dt_years:100.0,seed:42,num_cohorts:10,climate:None,wind:None }
    }
}
pub struct DeepTimeSimulation {
    pub world:World, pub solver:FastScapeSolver, pub compute:ComputePipeline,
    pub clock:SimulationClock, pub tectonics:TectonicForcing, pub climate:ClimateEngine,
    pub config:SimulationConfig, pub heights:Vec<f32>,
}
impl DeepTimeSimulation {
    pub fn new(config:SimulationConfig) -> Self {
        let w=config.grid_width; let h=config.grid_height;
        let mut hm=Heightmap::fbm(w,h,config.max_elevation_m,7,2.0,0.5,config.seed);
        hm.apply_island_mask(0.15);
        let stats=hm.stats(); println!("Initial heightmap: {}",stats);
        let mut world=World::new(w as u32,h as u32);
        world.initialize_grid(
            |x,y|hm.get(x as usize,y as usize),
            |elev|{if elev<=config.sea_level_m+10.0{MaterialId::Sediment}else if elev<200.0{MaterialId::SoilA}else if elev<800.0{MaterialId::SoftRock}else{MaterialId::Bedrock}},
            config.num_cohorts,
        );
        let spl_params=SplParams{m:0.5,n:1.0,cell_size:config.cell_size_m,rainfall:1.0,sea_level:config.sea_level_m,..Default::default()};
        let solver=FastScapeSolver::new(w,h,spl_params);
        let mut compute=ComputePipeline::new(w,h);
        compute.buffers.elevation=hm.data.clone();
        compute.update_activity_mask(config.sea_level_m);
        let temporal=TemporalConfig{geo_dt_years:config.geo_dt_years,social_ticks_per_geo_epoch:500,..Default::default()};
        let clock=SimulationClock::new(temporal);
        let tectonics=TectonicForcing::new(0.0001);
        let climate_config=config.climate.clone().unwrap_or_default();
        let wind_params=config.wind.clone().unwrap_or_default();
        let mut climate=ClimateEngine::new(w,h,climate_config,wind_params,config.cell_size_m);
        climate.carbon.set_cell_size_m(config.cell_size_m);
        let heights=hm.data;
        let mut sim=Self{world,solver,compute,clock,tectonics,climate,config,heights};
        let n=sim.config.grid_width*sim.config.grid_height;
        let delta_h_zero=vec![0.0f32;n];
        sim.climate.step_epoch(&sim.heights,&delta_h_zero,100.0,10.0,0.0);
        sim
    }
    pub fn step_geological_epoch(&mut self) {
        let dt=self.clock.geo_dt();
        let w=self.config.grid_width; let h=self.config.grid_height;
        let uplift=self.tectonics.uplift_array(w,h,self.config.cell_size_m);
        let kf:Vec<f32>=self.world.columns.iter().map(|c|c.erodibility.0).collect();
        let kd:Vec<f32>=self.world.columns.iter().map(|c|c.diffusivity.0).collect();
        let deltas=self.solver.step_epoch(&mut self.heights,&uplift,&kf,&kd,dt);
        self.world.apply_elevation_deltas(&deltas);
        self.compute.buffers.elevation=self.heights.clone();
        self.compute.update_activity_mask(self.config.sea_level_m);
        let plant_biomass=self.count_land_cells() as f32*50.0;
        let animal_resp=plant_biomass*0.1;
        self.climate.step_epoch(&self.heights,&deltas,plant_biomass,animal_resp,self.clock.geo_dt() as f64);
        self.sync_climate_to_terrain();
    }
    fn sync_climate_to_terrain(&mut self) {
        let base_temp=self.climate.thermostat.state.global_mean_temp_c;
        for (i,col) in self.world.columns.iter_mut().enumerate() {
            let climate_col=&self.climate.columns[i];
            let rain_kf_mult=(climate_col.precipitation_m/1.0).powf(0.4).clamp(0.5,3.0);
            let ft_kf_mult=self.climate.lapse.freeze_thaw_intensity(base_temp,col.elevation.0);
            let ft_bonus=1.0+ft_kf_mult*2.0;
            col.erodibility.0=Erodibility::for_material(col.layers.surface_material()).0*rain_kf_mult*ft_bonus;
        }
    }
    fn count_land_cells(&self) -> usize { self.heights.iter().filter(|&&h|h>self.config.sea_level_m).count() }
    pub fn step_frame(&mut self,dt_s:f64) {
        let result=self.clock.advance_social_tick(dt_s);
        if result.fire_geo { self.step_geological_epoch(); }
    }
    pub fn add_uplift_hotspot(&mut self,center_x:u32,center_y:u32,radius_cells:f32,rate_m_yr:f32) {
        let cx=center_x as f32*self.config.cell_size_m;
        let cy=center_y as f32*self.config.cell_size_m;
        let radius_m=radius_cells*self.config.cell_size_m;
        self.tectonics.add_hotspot(cx,cy,radius_m,rate_m_yr);
        println!("[ELDER GOD] Uplift hotspot at ({},{}) r={}km @{:.2}mm/yr",center_x,center_y,radius_cells as u32,rate_m_yr*1000.0);
        let radius_km=radius_cells*self.config.cell_size_m/1000.0;
        let drawdown=self.climate.carbon.estimate_uplift_drawdown(radius_km,rate_m_yr);
        println!("          ↳ Projected CO₂ drawdown over 500kyr: ~{:.1} ppm",drawdown);
    }
    pub fn elder_god_co2_injection(&mut self,delta_ppm:f32) { self.climate.thermostat.elder_god_co2_injection(delta_ppm); }
    pub fn elder_god_rotate_wind(&mut self,delta_rad:f32) {
        self.climate.orographic.rotate_wind(delta_rad);
        let angle_deg=self.climate.orographic.wind.angle_rad.to_degrees();
        println!("[ELDER GOD] Wind rotated → {:.0}°",angle_deg);
    }
    pub fn print_stats(&self) {
        let max_h=self.solver.max_elevation(&self.heights);
        let mean_h=self.solver.mean_elevation(&self.heights);
        let active=self.compute.active_count();
        let total=self.config.grid_width*self.config.grid_height;
        println!("  {} | MaxH={:.0}m MeanH={:.0}m | Active={}/{} ({:.1}% skip)",
            self.clock.summary(),max_h,mean_h,active,total,(1.0-active as f32/total as f32)*100.0);
        println!("  Climate: {}",self.climate.summary());
    }
    pub fn print_biome_distribution(&self) {
        let biome_counts=self.climate.biome_counts(&self.heights,self.config.sea_level_m);
        let land_total:usize=biome_counts.iter().skip(1).map(|(_,c)|c).sum();
        println!("  Biome distribution ({} land cells):",land_total);
        for (biome,count) in &biome_counts {
            if *count>0 {
                let pct=*count as f32/land_total.max(1) as f32*100.0;
                println!("    {:20} {:5} cells ({:.1}%)",biome.name(),count,pct);
            }
        }
    }
}

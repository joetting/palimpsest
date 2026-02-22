use rayon::prelude::*;
use crate::climate::thermostat::ClimateState;
#[derive(Debug, Clone)]
pub struct SilicateWeatheringParams {
    pub base_rate_ppm_yr_km2:f32, pub q10:f32, pub erosion_amplification:f32,
    pub soil_shield_factor:f32, pub organic_burial_efficiency:f32,
    pub carbon_flush_fraction:f32, pub reference_temp_c:f32,
}
impl Default for SilicateWeatheringParams {
    fn default() -> Self {
        Self { base_rate_ppm_yr_km2:0.0003,q10:2.0,erosion_amplification:5.0,
            soil_shield_factor:0.8,organic_burial_efficiency:0.45,
            carbon_flush_fraction:0.30,reference_temp_c:15.0 }
    }
}
#[derive(Debug, Clone, Default)]
pub struct CarbonFluxes {
    pub silicate_drawdown:f32, pub organic_burial:f32, pub anthropogenic_flush:f32,
    pub active_erosion_cells:usize, pub mean_erosion_rate_mm_yr:f32, pub tectonic_contribution:f32,
}
impl CarbonFluxes {
    pub fn summary(&self) -> String {
        format!("Silicate↓={:.4}ppm/yr  OrgBurial↓={:.4}ppm/yr  CFlush↑={:.4}ppm/yr  ErosionCells={}  MeanErosion={:.3}mm/yr",
            self.silicate_drawdown,self.organic_burial,self.anthropogenic_flush,
            self.active_erosion_cells,self.mean_erosion_rate_mm_yr)
    }
}
pub struct CarbonCycleEngine { pub params:SilicateWeatheringParams, cell_area_km2:f32 }
impl CarbonCycleEngine {
    pub fn new(params:SilicateWeatheringParams) -> Self { Self { params, cell_area_km2:1.0 } }
    pub fn set_cell_size_m(&mut self,cell_size_m:f32) { self.cell_area_km2=(cell_size_m/1000.0).powi(2); }
    pub fn compute_fluxes(&self,elevations:&[f32],delta_h:&[f32],grid_width:usize,grid_height:usize,climate:&ClimateState) -> CarbonFluxes {
        let n=grid_width*grid_height;
        let temp=climate.global_mean_temp_c;
        let q10_scale=self.params.q10.powf((temp-self.params.reference_temp_c)/10.0);
        let (total_silicate,total_burial,total_flush,erosion_count,erosion_sum)=
            (0..n).into_par_iter().map(|i|{
                let elev=elevations.get(i).copied().unwrap_or(0.0);
                let dh=delta_h.get(i).copied().unwrap_or(0.0);
                if elev<=0.0 { return (0.0f32,0.0f32,0.0f32,0usize,0.0f32); }
                let erosion_m=(-dh).max(0.0); let deposition_m=dh.max(0.0);
                let erosion_amplifier=1.0+self.params.erosion_amplification*erosion_m;
                let silicate_rate=self.params.base_rate_ppm_yr_km2*self.cell_area_km2*q10_scale*erosion_amplifier;
                let burial=if deposition_m>0.01 {
                    let som_density_kg_m3=20.0;
                    let buried_som=deposition_m*som_density_kg_m3*self.cell_area_km2*1e6;
                    let domain_mass_scale=1e-12;
                    buried_som*self.params.organic_burial_efficiency*domain_mass_scale
                } else { 0.0 };
                let flush=if erosion_m>0.001 {
                    let ancient_som=erosion_m*10.0*self.cell_area_km2*1e6;
                    let domain_mass_scale=1e-12;
                    ancient_som*self.params.carbon_flush_fraction*domain_mass_scale
                } else { 0.0 };
                let erosion_active=if erosion_m>1e-4{1}else{0};
                (silicate_rate,burial,flush,erosion_active,erosion_m*1000.0)
            }).reduce(||(0.0f32,0.0f32,0.0f32,0usize,0.0f32),|a,b|(a.0+b.0,a.1+b.1,a.2+b.2,a.3+b.3,a.4+b.4));
        let mean_erosion=if erosion_count>0{erosion_sum/erosion_count as f32}else{0.0};
        
        // --- NEW SCALING LOGIC ---
        let earth_area_km2 = crate::math::constants::EARTH_CONTINENTAL_AREA_M2 as f32 / 1_000_000.0;
        let grid_area_km2 = (n as f32) * self.cell_area_km2;
        let area_scaling = earth_area_km2 / grid_area_km2.max(1.0);

        CarbonFluxes { 
            silicate_drawdown: total_silicate / area_scaling,
            organic_burial: total_burial / area_scaling,
            anthropogenic_flush: total_flush / area_scaling,
            active_erosion_cells: erosion_count,
            mean_erosion_rate_mm_yr: mean_erosion,
            tectonic_contribution: (total_silicate * 0.6) / area_scaling 
        }
    }
    pub fn estimate_uplift_drawdown(&self,radius_km:f32,rate_m_yr:f32) -> f32 {
        let area_km2=std::f32::consts::PI*radius_km*radius_km;
        let enhanced_rate=self.params.base_rate_ppm_yr_km2*area_km2*10.0*rate_m_yr;
        enhanced_rate*500_000.0
    }

    /// Compute fluxes with per-cell runoff coupling.
    /// Silicate weathering scales with runoff (wet climates weather faster)
    /// and with soil moisture (dry soils don't weather).
    pub fn compute_fluxes_hydro(
        &self, elevations: &[f32], delta_h: &[f32],
        runoff: &[f32], soil_moisture: &[f32],
        grid_width: usize, grid_height: usize,
        climate: &ClimateState,
    ) -> CarbonFluxes {
        let n = grid_width * grid_height;
        let temp = climate.global_mean_temp_c;
        let q10_scale = self.params.q10.powf((temp - self.params.reference_temp_c) / 10.0);

        let (total_silicate, total_burial, total_flush, erosion_count, erosion_sum) =
            (0..n).into_par_iter().map(|i| {
                let elev = elevations.get(i).copied().unwrap_or(0.0);
                let dh = delta_h.get(i).copied().unwrap_or(0.0);
                if elev <= 0.0 { return (0.0f32, 0.0f32, 0.0f32, 0usize, 0.0f32); }

                let erosion_m = (-dh).max(0.0);
                let deposition_m = dh.max(0.0);
                let cell_runoff = runoff.get(i).copied().unwrap_or(0.3);
                let cell_moisture = soil_moisture.get(i).copied().unwrap_or(0.5);

                // Runoff amplifier: weathering rate ~ runoff^0.65 (West 2012)
                // Normalized to reference runoff of 0.4 m/yr
                let runoff_amplifier = (cell_runoff / 0.4).max(0.01).powf(0.65);

                // Moisture factor: very dry soil doesn't weather
                let moisture_factor = cell_moisture.clamp(0.05, 1.0);

                let erosion_amplifier = 1.0 + self.params.erosion_amplification * erosion_m;

                let silicate_rate = self.params.base_rate_ppm_yr_km2
                    * self.cell_area_km2
                    * q10_scale
                    * erosion_amplifier
                    * runoff_amplifier
                    * moisture_factor;

                let burial = if deposition_m > 0.01 {
                    let som_density_kg_m3 = 20.0;
                    let buried_som = deposition_m * som_density_kg_m3 * self.cell_area_km2 * 1e6;
                    let domain_mass_scale = 1e-12;
                    buried_som * self.params.organic_burial_efficiency * domain_mass_scale
                } else { 0.0 };

                let flush = if erosion_m > 0.001 {
                    let ancient_som = erosion_m * 10.0 * self.cell_area_km2 * 1e6;
                    let domain_mass_scale = 1e-12;
                    ancient_som * self.params.carbon_flush_fraction * domain_mass_scale
                } else { 0.0 };

                let erosion_active = if erosion_m > 1e-4 { 1 } else { 0 };
                (silicate_rate, burial, flush, erosion_active, erosion_m * 1000.0)
            }).reduce(
                || (0.0f32, 0.0f32, 0.0f32, 0usize, 0.0f32),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4),
            );

        let mean_erosion = if erosion_count > 0 { erosion_sum / erosion_count as f32 } else { 0.0 };

        let earth_area_km2 = crate::math::constants::EARTH_CONTINENTAL_AREA_M2 as f32 / 1_000_000.0;
        let grid_area_km2 = (n as f32) * self.cell_area_km2;
        let area_scaling = earth_area_km2 / grid_area_km2.max(1.0);

        CarbonFluxes {
            silicate_drawdown: total_silicate / area_scaling,
            organic_burial: total_burial / area_scaling,
            anthropogenic_flush: total_flush / area_scaling,
            active_erosion_cells: erosion_count,
            mean_erosion_rate_mm_yr: mean_erosion,
            tectonic_contribution: (total_silicate * 0.6) / area_scaling,
        }
    }
}

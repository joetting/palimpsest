/// Diagnostic: trace one epoch of CO2 fluxes to find the imbalance.
use crate::world::WorldSimulation;

pub fn trace_co2_budget(world: &mut WorldSimulation) {
    let dt = world.config.dt_years;
    let dt_f32 = dt as f32;
    let n = world.config.grid_width * world.config.grid_height;
    let c = &world.climate.thermostat.config;
    let state = &world.climate.thermostat.state;
    
    let growth_mults = world.biogeo.growth_multipliers(&world.nutrient_columns);
    let land_cells = world.elevations.iter()
        .filter(|&&e| e > world.config.sea_level).count().max(1);
    let plant_biomass: f32 = world.climate.columns.iter().enumerate()
        .filter(|(i, _)| world.elevations[*i] > world.config.sea_level)
        .map(|(i, col)| {
            col.biome.veg_cover() * growth_mults.get(i).copied().unwrap_or(0.5)
        }).sum::<f32>() / land_cells as f32;
    let animal_respiration = plant_biomass * 0.12;

    // Use hydro-coupled fluxes if available
    let hydro_runoff: Vec<f32> = world.hydrology.cells.iter().map(|c| c.runoff_m).collect();
    let hydro_moisture: Vec<f32> = world.hydrology.cells.iter().map(|c| c.soil_moisture).collect();
    let fluxes = world.climate.carbon.compute_fluxes_hydro(
        &world.elevations, &world.delta_h,
        &hydro_runoff, &hydro_moisture,
        world.config.grid_width, world.config.grid_height,
        state,
    );
    
    // Thermostat flux terms (same math as step_epoch)
    let volcanic = c.volcanic_outgas_rate * dt_f32;
    let respiration = (c.animal_respiration_coeff * animal_respiration 
                       + c.decay_rate * plant_biomass * 0.1) * dt_f32;
    let flush = fluxes.anthropogenic_flush * dt_f32;
    
    let temp_scale = {
        let opt = 18.0f32; let width = 20.0f32;
        let t = -(state.global_mean_temp_c - opt).powi(2) / (2.0 * width * width);
        t.exp().clamp(0.01, 1.0)
    };
    let photosynthesis = c.plant_uptake_coeff * plant_biomass * temp_scale * dt_f32;
    let silicate = fluxes.silicate_drawdown * dt_f32;
    let burial = fluxes.organic_burial * dt_f32;
    
    let delta_co2 = volcanic + respiration + flush - photosynthesis - silicate - burial;
    
    println!("=== CO₂ BUDGET TRACE (epoch {}) ===", world.epoch);
    println!("  Current CO₂:    {:.1} ppm", state.atmospheric_co2_ppm);
    println!("  Current T:       {:.1} °C", state.global_mean_temp_c);
    println!("  dt:              {:.0} years", dt);
    println!("  plant_biomass:   {:.2} (sum of veg_cover * growth_mult)", plant_biomass);
    println!("  animal_resp:     {:.4}", animal_respiration);
    println!("  temp_scale:      {:.4} (photosynthesis efficiency)", temp_scale);
    println!("  --- SOURCES (add CO₂) ---");
    println!("  volcanic:        {:+.4} ppm  (rate={:.4}/yr * dt={:.0})", volcanic, c.volcanic_outgas_rate, dt);
    println!("  respiration:     {:+.4} ppm  (animal_coeff*resp + decay*biomass*0.1)*dt", respiration);
    println!("  carbon_flush:    {:+.4} ppm  (erosion re-exposes ancient C)", flush);
    println!("  --- SINKS (remove CO₂) ---");
    println!("  photosynthesis:  {:+.4} ppm  (uptake_coeff*biomass*temp_scale*dt)", -photosynthesis);
    println!("  silicate_weath:  {:+.4} ppm  (weathering drawdown*dt)", -silicate);
    println!("  organic_burial:  {:+.4} ppm  (burial*dt)", -burial);
    println!("  --- NET ---");
    println!("  delta_CO₂:       {:+.4} ppm", delta_co2);
    println!("  projected_CO₂:   {:.1} ppm (clamped to [{}, {}])", 
        (state.atmospheric_co2_ppm + delta_co2).clamp(150.0, 8000.0), 150, 8000);
    println!("  cell_area_km2:   {:.4}", (world.config.cell_size_m / 1000.0).powi(2));
    println!("  grid_area_km2:   {:.0}", n as f32 * (world.config.cell_size_m / 1000.0).powi(2));
    println!("  earth_area_km2:  {:.0}", crate::math::constants::EARTH_CONTINENTAL_AREA_M2 as f32 / 1e6);
    println!("  area_ratio:      {:.2}", 
        (crate::math::constants::EARTH_CONTINENTAL_AREA_M2 as f32 / 1e6) / 
        (n as f32 * (world.config.cell_size_m / 1000.0).powi(2)));
    println!("  carbon fluxes:   {}", fluxes.summary());
    println!("===============================\n");
}

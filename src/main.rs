mod ecs;
mod terrain;
mod compute;
mod math;
mod config;
mod nutrients;
mod climate;

use config::{DeepTimeSimulation, SimulationConfig};
use climate::{ClimateConfig, WindParams};
use nutrients::pools::NutrientColumn;
use nutrients::solver::{BiogeochemSolver, ColumnEnv, update_column};
use nutrients::pedogenesis::{PedogenesisSolver, PedogenesisParams, PedogenesisState};
use image::{ImageBuffer, Rgb};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  DEEP TIME ENGINE — Phase 3: Climate Engine");
    println!("  Proserpina Thermostat + Orographic Rainfall + CO₂ Feedback");
    println!("═══════════════════════════════════════════════════════════════\n");

    let climate_config = ClimateConfig {
        initial_co2_ppm: 500.0, lambda: 3.0, t0_baseline: 18.0,
        plant_uptake_coeff: 0.002, animal_respiration_coeff: 0.001,
        volcanic_outgas_rate: 0.035, wind_angle_rad: std::f32::consts::PI,
        wind_speed_ms: 8.0, ..ClimateConfig::default()
    };
    let wind_params = WindParams {
        angle_rad: std::f32::consts::PI, incoming_moisture_m: 4.5,
        alpha_condensation: 0.008, beta_fallout_per_km: 0.15, background_precip_m: 0.4,
    };
    let config = SimulationConfig {
        grid_width: 128, grid_height: 128, cell_size_m: 1_000.0,
        max_elevation_m: 400.0, sea_level_m: 0.0, geo_dt_years: 100.0,
        seed: 1984, num_cohorts: 10,
        climate: Some(climate_config), wind: Some(wind_params),
    };

    let mut sim = DeepTimeSimulation::new(config);
    let n = sim.config.grid_width * sim.config.grid_height;

    println!("[PHASE 2/3] Initializing nutrient and pedogenesis columns...");
    let mut nutrient_columns: Vec<NutrientColumn> = (0..n).map(|i| {
        let elev = sim.heights.get(i).copied().unwrap_or(0.0);
        let climate_col = &sim.climate.columns[i];
        match climate_col.biome {
            climate::Biome::TropicalForest => NutrientColumn::new_tropical(0.45),
            climate::Biome::Ocean => NutrientColumn::new_alluvial(0.5),
            _ => if elev < 100.0 { NutrientColumn::new_alluvial(0.45) } else { NutrientColumn::new_young_mineral(0.3) },
        }
    }).collect();

    let pedo_solver = PedogenesisSolver::new(sim.config.grid_width, sim.config.grid_height);
    let mut pedo_states = pedo_solver.initialize(&sim.heights, sim.config.sea_level_m);
    let pedo_params: Vec<PedogenesisParams> = (0..n).map(|i| {
        let elev = sim.heights.get(i).copied().unwrap_or(0.0);
        if elev < 200.0 { PedogenesisParams::tropical() }
        else if elev > 1500.0 { PedogenesisParams::cold_arid() }
        else { PedogenesisParams::default() }
    }).collect();

    let kf_base: Vec<f32> = sim.world.columns.iter().map(|c| c.erodibility.0).collect();
    let kd_base: Vec<f32> = sim.world.columns.iter().map(|c| c.diffusivity.0).collect();
    let biogeo_solver = BiogeochemSolver::new(sim.config.grid_width, sim.config.grid_height);

    println!("\n[INITIAL STATE]");
    sim.print_stats();
    sim.print_biome_distribution();
    print_nutrient_stats(&nutrient_columns, &pedo_states, &sim.compute.activity);

    println!("\n[ELDER GOD] Generating a central mountain ridge...");
    let hill_centers = [(40u32,64u32),(64,64),(88,64)];
    for (x,y) in hill_centers.iter() { sim.add_uplift_hotspot(*x,*y,12.0,0.00015); }

    println!("\n[SIMULATION] Coupled terrain + climate + biogeochem + pedogenesis\n");
    println!("  {:<8} {:<9} {:<9} {:<10} {:<8} {:<8} {:<8} {:<10} {:<8}",
        "Epoch","Time","MaxH(m)","MeanH(m)","CO₂(ppm)","T(°C)","P_lab","SoilDev","Regress%");
    println!("  {}","-".repeat(90));

    let n_epochs = 40;
    let dt = sim.config.geo_dt_years;
    let mut last_heights = sim.heights.clone();

    for epoch in 0..n_epochs {
        for _ in 0..500 {
            let result = sim.clock.advance_social_tick(1.0/60.0);
            if result.fire_geo { sim.step_geological_epoch(); break; }
        }
        let delta_h: Vec<f32> = sim.heights.iter().zip(last_heights.iter()).map(|(c,p)|c-p).collect();
        last_heights = sim.heights.clone();

        let bio_envs: Vec<ColumnEnv> = (0..n).map(|i| {
            let elev = sim.heights.get(i).copied().unwrap_or(0.0);
            let dh = delta_h.get(i).copied().unwrap_or(0.0);
            let cc = &sim.climate.columns[i];
            if elev <= sim.config.sea_level_m {
                return ColumnEnv { temp_c:4.0,moisture:1.0,runoff_m_yr:0.0,flooded:false,flood_p_input:0.0,delta_h_m:0.0,veg_cover:0.0 };
            }
            let flooded = dh>0.05 && elev<sim.config.sea_level_m+50.0;
            ColumnEnv { temp_c:cc.temp_c,moisture:cc.moisture,runoff_m_yr:cc.runoff_m,
                flooded,flood_p_input:if flooded{8.0}else{0.0},delta_h_m:dh,veg_cover:cc.biome.veg_cover() }
        }).collect();
        biogeo_solver.step_epoch(&mut nutrient_columns, &bio_envs, dt);

        let growth_mults = biogeo_solver.growth_multipliers(&nutrient_columns);
        let pedo_envs = pedo_solver.build_envs(&growth_mults, &delta_h, sim.config.cell_size_m);
        let (kf_mod,kd_mod) = pedo_solver.step_epoch(&mut pedo_states,&pedo_params,&pedo_envs,&kf_base,&kd_base,dt);
        for (i,col) in sim.world.columns.iter_mut().enumerate() {
            col.erodibility.0 = kf_mod.get(i).copied().unwrap_or(col.erodibility.0);
            col.diffusivity.0 = kd_mod.get(i).copied().unwrap_or(col.diffusivity.0);
        }

        if epoch%8==0 || epoch==n_epochs-1 {
            let max_h = sim.solver.max_elevation(&sim.heights);
            let mean_h = sim.solver.mean_elevation(&sim.heights);
            let co2 = sim.climate.thermostat.state.atmospheric_co2_ppm;
            let temp = sim.climate.thermostat.state.global_mean_temp_c;
            let mean_p = biogeo_solver.mean_surface_p_labile(&nutrient_columns, &sim.compute.activity);
            let mean_s = pedo_solver.mean_s(&pedo_states);
            let reg_pct = pedo_solver.regressive_count(&pedo_states) as f32/pedo_states.len() as f32*100.0;
            let elapsed_ky = sim.clock.geo_elapsed_years/1000.0;
            let phase_icon = if sim.climate.thermostat.state.in_ice_age{"❄"} else if sim.climate.thermostat.state.in_hothouse{"🔥"} else{"🌿"};
            println!("  {:<8} {:<9} {:<9.1} {:<10.1} {:<8.1} {:<8.1} {:<8.2} {:<10.3} {:<8.1} {}",
                format!("#{}", sim.clock.geo_epoch),format!("{:.1}ka",elapsed_ky),
                max_h,mean_h,co2,temp,mean_p,mean_s,reg_pct,phase_icon);
        }

        if epoch==n_epochs/4 {
            println!("\n  [ELDER GOD] Major orogenic event — CO₂ drawdown incoming...");
            sim.add_uplift_hotspot(32,96,25.0,0.0008); println!();
        }
        if epoch==n_epochs/2 {
            println!("\n  [ELDER GOD] Volcanic superplume — CO₂ injection +150 ppm!");
            sim.elder_god_co2_injection(150.0);
            println!("  [CLIMATE]  → {}",sim.climate.thermostat.state.summary()); println!();
        }
        if epoch==n_epochs*3/4 {
            println!("\n  [ELDER GOD] Shifting wind patterns eastward...");
            sim.elder_god_rotate_wind(std::f32::consts::PI*0.25); println!();
        }
    }

    println!("\n[FINAL STATE]");
    sim.print_stats();
    sim.print_biome_distribution();
    print_nutrient_stats(&nutrient_columns, &pedo_states, &sim.compute.activity);

    println!("\n  Carbon Cycle Summary:");
    println!("  Cumulative silicate drawdown : {:.2} ppm",sim.climate.thermostat.state.cumulative_silicate_drawdown);
    println!("  Cumulative organic burial    : {:.2} ppm",sim.climate.thermostat.state.cumulative_organic_burial);

    println!("\n[MAYA DEMO] Deforestation collapse → P depletion → yield crisis");
    demo_maya_collapse();

    println!("\n════════════════════════════════════════════════════════════════");
    println!("  PHASE 3 COMPLETE — Climate Engine integrated");
    println!("  Proserpina Thermostat | LFPM Orographic | CO₂ Weathering Loop");
    println!("════════════════════════════════════════════════════════════════\n");
}

fn print_nutrient_stats(columns:&[NutrientColumn],pedo:&[PedogenesisState],mask:&[u32]) {
    let mut sum_p=0.0f32; let mut sum_k=0.0f32; let mut sum_oc=0.0f32; let mut sum_s=0.0f32; let mut count=0u32;
    for i in 0..columns.len() {
        let word=mask.get(i/32).copied().unwrap_or(0);
        if (word>>(i%32))&1==1 {
            sum_p+=columns[i].surface_p_labile(); sum_k+=columns[i].surface_k_exch();
            sum_oc+=columns[i].layers[0].p_occluded; sum_s+=pedo.get(i).map(|s|s.s).unwrap_or(0.0); count+=1;
        }
    }
    let c=count.max(1) as f32;
    println!("  Active columns   : {}",count);
    println!("  Mean P_labile    : {:.2} mg/kg",sum_p/c);
    println!("  Mean K_exch      : {:.1} mg/kg",sum_k/c);
    println!("  Mean P_occluded  : {:.1} mg/kg",sum_oc/c);
    println!("  Mean soil dev S  : {:.3}",sum_s/c);
}

fn demo_maya_collapse() {
    let mut col = NutrientColumn::new_alluvial(0.4);
    println!("  Baseline: P_labile={:.1}, K_exch={:.0}, growth={:.2}",col.surface_p_labile(),col.surface_k_exch(),col.column_growth_multiplier());
    for cycle in 1..=3 {
        let farm=ColumnEnv{temp_c:26.0,moisture:0.8,runoff_m_yr:0.7,flooded:false,flood_p_input:0.0,delta_h_m:0.0,veg_cover:0.05};
        update_column(&mut col,&farm,50.0);
        let fallow=ColumnEnv{veg_cover:0.4,..farm};
        update_column(&mut col,&fallow,20.0);
        let gm=col.column_growth_multiplier();
        println!("  Cycle {:}: P={:.1}, K={:.0}, growth={:.2}  → {}",cycle,col.surface_p_labile(),col.surface_k_exch(),gm,
            match gm { g if g>0.6=>"Productive", g if g>0.4=>"Stressed", g if g>0.2=>"CRISIS", _=>"COLLAPSE" });
    }
}

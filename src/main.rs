// ============================================================================
// Deep Time Engine — Phase 3: Climate Engine Integration
// Proserpina Thermostat + Orographic Rainfall + Geological Carbon Thermostat
// ============================================================================

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

    // ── Climate Config ────────────────────────────────────────────────────
    let climate_config = ClimateConfig {
        initial_co2_ppm:          280.0,   // pre-industrial baseline
        lambda:                    3.0,    // 3°C per CO₂ doubling (IPCC)
        t0_baseline:               14.0,   // global mean at baseline
        plant_uptake_coeff:        0.002,
        animal_respiration_coeff:  0.001,
        volcanic_outgas_rate:      0.012,
        wind_angle_rad:            std::f32::consts::PI, // westerly
        wind_speed_ms:             8.0,
        ..ClimateConfig::default()
    };

    // Westerly wind from the west (π radians = wind travels eastward)
    let wind_params = WindParams {
        angle_rad:           std::f32::consts::PI,
        incoming_moisture_m: 2.0,
        alpha_condensation:  0.005,
        beta_fallout_per_km: 0.18,
        background_precip_m: 0.04,
    };

    // ── Simulation Config ─────────────────────────────────────────────────
    let config = SimulationConfig {
        grid_width:      128,
        grid_height:     128,
        cell_size_m:     1_000.0,
        max_elevation_m: 400.0,
        sea_level_m:     0.0,
        geo_dt_years:    100.0,
        seed:            1984,
        num_cohorts:     10,
        climate:         Some(climate_config),
        wind:            Some(wind_params),
    };

    let mut sim = DeepTimeSimulation::new(config);
    let n = sim.config.grid_width * sim.config.grid_height;

    // ── Phase 2: Nutrient + Pedogenesis init (carried over) ───────────────
    println!("[PHASE 2/3] Initializing nutrient and pedogenesis columns...");
    let mut nutrient_columns: Vec<NutrientColumn> = (0..n).map(|i| {
        let elev = sim.heights.get(i).copied().unwrap_or(0.0);
        let climate_col = &sim.climate.columns[i];
        // Initialise nutrient type based on climate + elevation
        match climate_col.biome {
            climate::Biome::TropicalForest  => NutrientColumn::new_tropical(0.45),
            climate::Biome::Ocean           => NutrientColumn::new_alluvial(0.5),
            _                               =>
                if elev < 100.0 { NutrientColumn::new_alluvial(0.45) }
                else             { NutrientColumn::new_young_mineral(0.3) },
        }
    }).collect();

    let pedo_solver = PedogenesisSolver::new(sim.config.grid_width, sim.config.grid_height);
    let mut pedo_states = pedo_solver.initialize(&sim.heights, sim.config.sea_level_m);
    let pedo_params: Vec<PedogenesisParams> = (0..n).map(|i| {
        let elev = sim.heights.get(i).copied().unwrap_or(0.0);
        if elev < 200.0       { PedogenesisParams::tropical() }
        else if elev > 1500.0 { PedogenesisParams::cold_arid() }
        else                   { PedogenesisParams::default() }
    }).collect();

    let kf_base: Vec<f32> = sim.world.columns.iter().map(|c| c.erodibility.0).collect();
    let kd_base: Vec<f32> = sim.world.columns.iter().map(|c| c.diffusivity.0).collect();
    let biogeo_solver = BiogeochemSolver::new(sim.config.grid_width, sim.config.grid_height);

    // ── Initial state printout ─────────────────────────────────────────────
    println!("\n[INITIAL STATE]");
    sim.print_stats();
    sim.print_biome_distribution();
    print_nutrient_stats(&nutrient_columns, &pedo_states, &sim.compute.activity);

    // ── Elder God setup ───────────────────────────────────────────────────
    println!("\n[ELDER GOD] Generating rolling hill field...");
    let hill_centers = [
        (30u32, 30u32), (98, 30),
        (64, 64),
        (30, 98), (98, 98),
        (50, 90), (90, 50),
    ];
    for (x, y) in hill_centers.iter() {
        sim.add_uplift_hotspot(*x, *y, 20.0, 0.0003);
    }

    // ── Main simulation loop ───────────────────────────────────────────────
    println!("\n[SIMULATION] Coupled terrain + climate + biogeochem + pedogenesis\n");
    println!("  {:<8} {:<9} {:<9} {:<10} {:<8} {:<8} {:<8} {:<10} {:<8}",
        "Epoch", "Time", "MaxH(m)", "MeanH(m)", "CO₂(ppm)", "T(°C)", "P_lab", "SoilDev", "Regress%");
    println!("  {}", "-".repeat(90));

    let n_epochs = 40;
    let dt = sim.config.geo_dt_years;
    let mut last_heights = sim.heights.clone();

    for epoch in 0..n_epochs {
        // ── Terrain step ──────────────────────────────────────────────────
        for _ in 0..500 {
            let result = sim.clock.advance_social_tick(1.0 / 60.0);
            if result.fire_geo { sim.step_geological_epoch(); break; }
        }

        let delta_h: Vec<f32> = sim.heights.iter()
            .zip(last_heights.iter())
            .map(|(c, p)| c - p)
            .collect();
        last_heights = sim.heights.clone();

        // ── Biogeochem step — now uses climate temp/moisture ──────────────
        let bio_envs: Vec<ColumnEnv> = (0..n).map(|i| {
            let elev = sim.heights.get(i).copied().unwrap_or(0.0);
            let dh   = delta_h.get(i).copied().unwrap_or(0.0);
            let cc   = &sim.climate.columns[i];
            if elev <= sim.config.sea_level_m {
                return ColumnEnv {
                    temp_c: 4.0, moisture: 1.0,
                    runoff_m_yr: 0.0, flooded: false,
                    flood_p_input: 0.0, delta_h_m: 0.0,
                    veg_cover: 0.0,
                };
            }
            let flooded = dh > 0.05 && elev < sim.config.sea_level_m + 50.0;
            ColumnEnv {
                temp_c:       cc.temp_c,
                moisture:     cc.moisture,
                runoff_m_yr:  cc.runoff_m,
                flooded,
                flood_p_input: if flooded { 8.0 } else { 0.0 },
                delta_h_m:    dh,
                veg_cover:    cc.biome.veg_cover(),
            }
        }).collect();
        biogeo_solver.step_epoch(&mut nutrient_columns, &bio_envs, dt);

        // ── Pedogenesis step ──────────────────────────────────────────────
        let growth_mults = biogeo_solver.growth_multipliers(&nutrient_columns);
        let pedo_envs = pedo_solver.build_envs(&growth_mults, &delta_h, sim.config.cell_size_m);
        let (kf_mod, kd_mod) = pedo_solver.step_epoch(
            &mut pedo_states, &pedo_params, &pedo_envs, &kf_base, &kd_base, dt,
        );
        for (i, col) in sim.world.columns.iter_mut().enumerate() {
            col.erodibility.0 = kf_mod.get(i).copied().unwrap_or(col.erodibility.0);
            col.diffusivity.0  = kd_mod.get(i).copied().unwrap_or(col.diffusivity.0);
        }

        // ── Reporting & PNG export ────────────────────────────────────────
        if epoch % 8 == 0 || epoch == n_epochs - 1 {
            let max_h  = sim.solver.max_elevation(&sim.heights);
            let mean_h = sim.solver.mean_elevation(&sim.heights);
            let co2    = sim.climate.thermostat.state.atmospheric_co2_ppm;
            let temp   = sim.climate.thermostat.state.global_mean_temp_c;
            let mean_p = biogeo_solver.mean_surface_p_labile(&nutrient_columns, &sim.compute.activity);
            let mean_s = pedo_solver.mean_s(&pedo_states);
            let reg_pct = pedo_solver.regressive_count(&pedo_states) as f32
                          / pedo_states.len() as f32 * 100.0;
            let elapsed_ky = sim.clock.geo_elapsed_years / 1000.0;

            // Climate phase indicator
            let phase_icon = if sim.climate.thermostat.state.in_ice_age { "❄" }
                             else if sim.climate.thermostat.state.in_hothouse { "🔥" }
                             else { "🌿" };

            println!("  {:<8} {:<9} {:<9.1} {:<10.1} {:<8.1} {:<8.1} {:<8.2} {:<10.3} {:<8.1} {}",
                format!("#{}", sim.clock.geo_epoch),
                format!("{:.1}ka", elapsed_ky),
                max_h, mean_h, co2, temp, mean_p, mean_s, reg_pct, phase_icon,
            );

            // Export PNGs
            let w = sim.config.grid_width;
            let h = sim.config.grid_height;

            // Terrain
            export_grid_to_png(&format!("terrain_epoch_{:03}.png", epoch),
                w, h, &sim.heights, sim.config.sea_level_m, 2000.0, true);

            // Soil development
            let soil_array: Vec<f32> = pedo_states.iter().map(|p| p.s).collect();
            export_grid_to_png(&format!("soil_epoch_{:03}.png", epoch),
                w, h, &soil_array, 0.0, 1.0, false);

            // Temperature heatmap
            let temp_array: Vec<f32> = sim.climate.columns.iter().map(|c| c.temp_c).collect();
            export_grid_to_png_diverging(&format!("temp_epoch_{:03}.png", epoch),
                w, h, &temp_array, -10.0, 35.0);

            // Rainfall heatmap
            let rain_array: Vec<f32> = sim.climate.columns.iter().map(|c| c.precipitation_m).collect();
            export_grid_to_png(&format!("rainfall_epoch_{:03}.png", epoch),
                w, h, &rain_array, 0.0, 4.0, false);

            // Biome map
            export_biome_png(&format!("biome_epoch_{:03}.png", epoch),
                w, h, &sim.climate.columns);
        }

        // ── Mid-simulation events ─────────────────────────────────────────
        if epoch == n_epochs / 4 {
            println!("\n  [ELDER GOD] Major orogenic event — CO₂ drawdown incoming...");
            sim.add_uplift_hotspot(32, 96, 25.0, 0.0008);
            println!();
        }

        if epoch == n_epochs / 2 {
            println!("\n  [ELDER GOD] Volcanic superplume — CO₂ injection +150 ppm!");
            sim.elder_god_co2_injection(150.0);
            println!("  [CLIMATE]  → {}", sim.climate.thermostat.state.summary());
            println!();
        }

        if epoch == n_epochs * 3 / 4 {
            println!("\n  [ELDER GOD] Shifting wind patterns eastward...");
            sim.elder_god_rotate_wind(std::f32::consts::PI * 0.25); // 45° rotation
            println!();
        }
    }

    // ── Final state ────────────────────────────────────────────────────────
    println!("\n[FINAL STATE]");
    sim.print_stats();
    sim.print_biome_distribution();
    print_nutrient_stats(&nutrient_columns, &pedo_states, &sim.compute.activity);

    println!("\n  Carbon Cycle Summary:");
    println!("  Cumulative silicate drawdown : {:.2} ppm",
        sim.climate.thermostat.state.cumulative_silicate_drawdown);
    println!("  Cumulative organic burial    : {:.2} ppm",
        sim.climate.thermostat.state.cumulative_organic_burial);

    // ── Maya collapse demo (unchanged) ────────────────────────────────────
    println!("\n[MAYA DEMO] Deforestation collapse → P depletion → yield crisis");
    demo_maya_collapse();

    println!("\n════════════════════════════════════════════════════════════════");
    println!("  PHASE 3 COMPLETE — Climate Engine integrated");
    println!("  Proserpina Thermostat | LFPM Orographic | CO₂ Weathering Loop");
    println!("════════════════════════════════════════════════════════════════\n");
}

// ── Diagnostics helpers ───────────────────────────────────────────────────

fn print_nutrient_stats(
    columns: &[NutrientColumn],
    pedo:    &[PedogenesisState],
    mask:    &[u32],
) {
    let mut sum_p = 0.0_f32; let mut sum_k = 0.0_f32;
    let mut sum_oc = 0.0_f32; let mut sum_s = 0.0_f32;
    let mut count = 0_u32;
    for i in 0..columns.len() {
        let word = mask.get(i / 32).copied().unwrap_or(0);
        if (word >> (i % 32)) & 1 == 1 {
            sum_p  += columns[i].surface_p_labile();
            sum_k  += columns[i].surface_k_exch();
            sum_oc += columns[i].layers[0].p_occluded;
            sum_s  += pedo.get(i).map(|s| s.s).unwrap_or(0.0);
            count  += 1;
        }
    }
    let c = count.max(1) as f32;
    println!("  Active columns   : {}", count);
    println!("  Mean P_labile    : {:.2} mg/kg", sum_p / c);
    println!("  Mean K_exch      : {:.1} mg/kg", sum_k / c);
    println!("  Mean P_occluded  : {:.1} mg/kg", sum_oc / c);
    println!("  Mean soil dev S  : {:.3}", sum_s / c);
}

fn demo_maya_collapse() {
    let mut col = NutrientColumn::new_alluvial(0.4);
    println!("  Baseline: P_labile={:.1}, K_exch={:.0}, growth={:.2}",
        col.surface_p_labile(), col.surface_k_exch(), col.column_growth_multiplier());

    for cycle in 1..=3 {
        let farm = ColumnEnv {
            temp_c: 26.0, moisture: 0.8, runoff_m_yr: 0.7,
            flooded: false, flood_p_input: 0.0, delta_h_m: 0.0, veg_cover: 0.05,
        };
        update_column(&mut col, &farm, 50.0);

        let fallow = ColumnEnv { veg_cover: 0.4, ..farm };
        update_column(&mut col, &fallow, 20.0);

        let gm = col.column_growth_multiplier();
        println!("  Cycle {:}: P={:.1}, K={:.0}, growth={:.2}  → {}",
            cycle, col.surface_p_labile(), col.surface_k_exch(), gm,
            match gm {
                g if g > 0.6 => "Productive",
                g if g > 0.4 => "Stressed",
                g if g > 0.2 => "CRISIS",
                _             => "COLLAPSE",
            }
        );
    }
}

// ── PNG export helpers ────────────────────────────────────────────────────

fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }

fn lerp_rgb(c1: [u8; 3], c2: [u8; 3], t: f32) -> Rgb<u8> {
    Rgb([
        lerp(c1[0] as f32, c2[0] as f32, t) as u8,
        lerp(c1[1] as f32, c2[1] as f32, t) as u8,
        lerp(c1[2] as f32, c2[2] as f32, t) as u8,
    ])
}

fn get_viridis(norm: f32) -> Rgb<u8> {
    let t = norm.clamp(0.0, 1.0);
    let stops: &[(f32, [u8; 3])] = &[
        (0.00, [68, 1, 84]),
        (0.25, [59, 82, 139]),
        (0.50, [33, 145, 140]),
        (0.75, [94, 201, 98]),
        (1.00, [253, 231, 37]),
    ];
    for i in 0..stops.len() - 1 {
        let (t1, c1) = stops[i];
        let (t2, c2) = stops[i + 1];
        if t <= t2 {
            let local_t = (t - t1) / (t2 - t1);
            return lerp_rgb(c1, c2, local_t);
        }
    }
    Rgb(stops[stops.len()-1].1)
}

fn get_terrain_color(norm: f32) -> Rgb<u8> {
    let t = norm.clamp(0.0, 1.0);
    let stops: &[(f32, [u8; 3])] = &[
        (0.00, [194, 178, 128]),
        (0.15, [34, 139, 34]),
        (0.40, [85, 107, 47]),
        (0.70, [120, 120, 120]),
        (1.00, [240, 240, 255]),
    ];
    for i in 0..stops.len() - 1 {
        let (t1, c1) = stops[i];
        let (t2, c2) = stops[i + 1];
        if t <= t2 {
            let lt = (t - t1) / (t2 - t1);
            return lerp_rgb(c1, c2, lt);
        }
    }
    Rgb(stops[stops.len()-1].1)
}

fn export_grid_to_png(
    filename: &str, width: usize, height: usize,
    data: &[f32], sea_level: f32, max_val: f32, is_terrain: bool,
) {
    let mut img = ImageBuffer::new(width as u32, height as u32);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let idx = (y as usize) * width + (x as usize);
        let val = data[idx];
        *pixel = if is_terrain {
            if val <= sea_level {
                let d = ((sea_level - val) / 100.0).clamp(0.0, 1.0);
                lerp_rgb([0, 15, 80], [0, 105, 200], 1.0 - d)
            } else {
                let norm = ((val - sea_level) / (max_val - sea_level).max(1.0)).clamp(0.0, 1.0);
                get_terrain_color(norm)
            }
        } else {
            let norm = (val / max_val.max(0.001)).clamp(0.0, 1.0);
            get_viridis(norm)
        };
    }
    if let Err(e) = img.save(filename) { eprintln!("PNG save error {}: {}", filename, e); }
}

/// Diverging colormap: blue (cold) → white (neutral) → red (hot)
fn export_grid_to_png_diverging(
    filename: &str, width: usize, height: usize,
    data: &[f32], min_val: f32, max_val: f32,
) {
    let mut img = ImageBuffer::new(width as u32, height as u32);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let idx = (y as usize) * width + (x as usize);
        let val = data[idx];
        let norm = ((val - min_val) / (max_val - min_val)).clamp(0.0, 1.0);
        *pixel = if norm < 0.5 {
            let t = norm * 2.0;
            lerp_rgb([30, 100, 200], [240, 240, 255], t)
        } else {
            let t = (norm - 0.5) * 2.0;
            lerp_rgb([240, 240, 255], [200, 50, 30], t)
        };
    }
    if let Err(e) = img.save(filename) { eprintln!("PNG save error {}: {}", filename, e); }
}

fn export_biome_png(filename: &str, width: usize, height: usize, columns: &[climate::ColumnClimate]) {
    use climate::Biome;
    let biome_colors: &[(Biome, [u8; 3])] = &[
        (Biome::Ocean,           [30, 80, 180]),
        (Biome::TropicalForest,  [0, 120, 30]),
        (Biome::TemperateForest, [60, 160, 60]),
        (Biome::Grassland,       [180, 210, 80]),
        (Biome::Desert,          [230, 200, 100]),
        (Biome::Tundra,          [180, 210, 210]),
        (Biome::Alpine,          [200, 200, 220]),
        (Biome::Glacier,         [240, 248, 255]),
    ];
    let mut img = ImageBuffer::new(width as u32, height as u32);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let idx = (y as usize) * width + (x as usize);
        let biome = columns.get(idx).map(|c| c.biome).unwrap_or(Biome::Ocean);
        let color = biome_colors.iter()
            .find(|(b, _)| *b == biome)
            .map(|(_, c)| *c)
            .unwrap_or([128, 128, 128]);
        *pixel = Rgb(color);
    }
    if let Err(e) = img.save(filename) { eprintln!("PNG save error {}: {}", filename, e); }
}

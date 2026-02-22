// ============================================================================
// Deep Time Engine — Phase 2: Critical Zone
//
// Phase 1 (complete): ECS, FastScape, Strang, bitmask, cohorts, tectonic forcing
// Phase 2 (this build): 9-pool P-K biogeochemistry + Phillips pedogenesis
// ============================================================================

mod ecs;
mod terrain;
mod compute;
mod math;
mod config;
mod nutrients;

use config::{DeepTimeSimulation, SimulationConfig};
use nutrients::pools::NutrientColumn;
use nutrients::solver::{BiogeochemSolver, ColumnEnv, update_column};
use nutrients::pedogenesis::{PedogenesisSolver, PedogenesisParams, PedogenesisState};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  DEEP TIME ENGINE — Phase 2: Critical Zone");
    println!("  Coupled P-K Biogeochemistry + Phillips Pedogenesis");
    println!("═══════════════════════════════════════════════════════════════\n");

    let config = SimulationConfig {
        grid_width:      128,
        grid_height:     128,
        cell_size_m:     1_000.0,
        max_elevation_m: 2_500.0,
        sea_level_m:     0.0,
        geo_dt_years:    100.0,
        seed:            1984,
        num_cohorts:     10,
    };

    let mut sim = DeepTimeSimulation::new(config);
    let n = sim.config.grid_width * sim.config.grid_height;

    // -------------------------------------------------------------------------
    // Phase 2: Initialize nutrient columns
    // -------------------------------------------------------------------------
    println!("[PHASE 2] Initializing {} nutrient columns (×8 layers ×9 pools)...", n);
    let mem_mb = n * 8 * 9 * 4 / (1024 * 1024);
    println!("  Memory footprint: ~{} MB", mem_mb);

    let mut nutrient_columns: Vec<NutrientColumn> = (0..n).map(|i| {
        let elev = sim.heights.get(i).copied().unwrap_or(0.0);
        if elev <= sim.config.sea_level_m {
            NutrientColumn::new_alluvial(0.5)
        } else if elev < 100.0 {
            NutrientColumn::new_alluvial(0.45)
        } else if elev < 800.0 {
            NutrientColumn::new_young_mineral(0.3)
        } else {
            NutrientColumn::new_young_mineral(0.15)
        }
    }).collect();

    // -------------------------------------------------------------------------
    // Phase 2: Initialize pedogenesis
    // -------------------------------------------------------------------------
    println!("[PHASE 2] Initializing Phillips pedogenesis (dS/dt = dP/dt - dR/dt)...");
    let pedo_solver = PedogenesisSolver::new(sim.config.grid_width, sim.config.grid_height);
    let mut pedo_states = pedo_solver.initialize(&sim.heights, sim.config.sea_level_m);
    let pedo_params: Vec<PedogenesisParams> = (0..n).map(|i| {
        let elev = sim.heights.get(i).copied().unwrap_or(0.0);
        if elev < 200.0      { PedogenesisParams::tropical() }
        else if elev > 1500.0 { PedogenesisParams::cold_arid() }
        else                  { PedogenesisParams::default() }
    }).collect();

    let kf_base: Vec<f32> = sim.world.columns.iter().map(|c| c.erodibility.0).collect();
    let kd_base: Vec<f32> = sim.world.columns.iter().map(|c| c.diffusivity.0).collect();
    let biogeo_solver = BiogeochemSolver::new(sim.config.grid_width, sim.config.grid_height);

    println!("\n[INITIAL STATE]");
    print_nutrient_stats(&nutrient_columns, &pedo_states, &sim.compute.activity);

    // -------------------------------------------------------------------------
    // Coupled simulation loop
    // -------------------------------------------------------------------------
    println!("\n[ELDER GOD] Adding tectonic hotspot (64,64)...");
    sim.add_uplift_hotspot(64, 64, 25.0, 0.002);

    println!("\n[SIMULATION] Coupled terrain + biogeochemistry + pedogenesis\n");
    println!("  {:<8} {:<9} {:<9} {:<10} {:<10} {:<10} {:<10} {:<8}",
        "Epoch", "Time", "MaxH(m)", "MeanH(m)", "P_labile", "K_exch", "SoilDev", "Regress%");
    println!("  {}", "-".repeat(80));

    let n_epochs = 40;
    let dt = sim.config.geo_dt_years;
    let mut last_heights = sim.heights.clone();

    for epoch in 0..n_epochs {
        // Phase 1: terrain step
        for _ in 0..500 {
            let result = sim.clock.advance_social_tick(1.0 / 60.0);
            if result.fire_geo { sim.step_geological_epoch(); break; }
        }

        // Δh from terrain step
        let delta_h: Vec<f32> = sim.heights.iter()
            .zip(last_heights.iter())
            .map(|(c, p)| c - p)
            .collect();
        last_heights = sim.heights.clone();

        // Phase 2a: biogeochemistry
        let bio_envs = biogeo_solver.build_default_envs(
            &sim.heights, sim.config.sea_level_m, &delta_h, &sim.compute.activity,
        );
        biogeo_solver.step_epoch(&mut nutrient_columns, &bio_envs, dt);

        // Phase 2b: growth multipliers → pedogenesis
        let growth_mults = biogeo_solver.growth_multipliers(&nutrient_columns);
        let pedo_envs = pedo_solver.build_envs(&growth_mults, &delta_h, sim.config.cell_size_m);
        let (kf_mod, kd_mod) = pedo_solver.step_epoch(
            &mut pedo_states, &pedo_params, &pedo_envs, &kf_base, &kd_base, dt,
        );

        // Phase 2c: feed S-modulated Kf/Kd back to terrain solver
        for (i, col) in sim.world.columns.iter_mut().enumerate() {
            col.erodibility.0 = kf_mod.get(i).copied().unwrap_or(col.erodibility.0);
            col.diffusivity.0  = kd_mod.get(i).copied().unwrap_or(col.diffusivity.0);
        }

        if epoch % 8 == 0 || epoch == n_epochs - 1 {
            let max_h  = sim.solver.max_elevation(&sim.heights);
            let mean_h = sim.solver.mean_elevation(&sim.heights);
            let mean_p = biogeo_solver.mean_surface_p_labile(&nutrient_columns, &sim.compute.activity);
            let mean_k = biogeo_solver.mean_surface_k_exch(&nutrient_columns, &sim.compute.activity);
            let mean_s  = pedo_solver.mean_s(&pedo_states);
            let reg_pct = pedo_solver.regressive_count(&pedo_states) as f32
                          / pedo_states.len() as f32 * 100.0;
            let elapsed_ky = sim.clock.geo_elapsed_years / 1000.0;

            println!("  {:<8} {:<9} {:<9.1} {:<10.1} {:<10.2} {:<10.1} {:<10.3} {:<8.1}",
                format!("#{}", sim.clock.geo_epoch),
                format!("{:.1}ka", elapsed_ky),
                max_h, mean_h, mean_p, mean_k, mean_s, reg_pct,
            );
        }

        if epoch == n_epochs / 2 {
            println!("\n  [ELDER GOD] Orogenic event! Erosion spike → soil reset → P loss...");
            sim.add_uplift_hotspot(32, 96, 40.0, 0.005);
            println!();
        }
    }

    println!("\n[FINAL STATE]");
    print_nutrient_stats(&nutrient_columns, &pedo_states, &sim.compute.activity);

    // -------------------------------------------------------------------------
    // Maya collapse demo
    // -------------------------------------------------------------------------
    println!("\n[MAYA DEMO] Deforestation collapse → P depletion → yield crisis");
    demo_maya_collapse();

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("\n================================================================");
    println!("  PHASE 2 COMPLETE — Critical Zone Architecture");
    println!("================================================================\n");
    println!("  PHASE 1 (terrain):");
    println!("  [x] ECS World            {} entities, SoA GPU-ready", sim.world.total_columns());
    println!("  [x] FastScape Solver     O(N) implicit SPL, Newton-Raphson fixed");
    println!("  [x] Strang Splitting     Social|Geo|Bio multirate schedules");
    println!("  [x] Activity Bitmask     sparse GPU skip");
    println!("  [x] Tectonic Forcing     {} active hotspots\n", sim.tectonics.hotspots.len());
    println!("  PHASE 2 (biogeochemistry + pedogenesis):");
    println!("  [x] 9-pool P-K ODE       Buendía P-cycle + K biocycling pump analogue");
    println!("  [x] DEFAC scalar         Q10 temperature × moisture limitation");
    println!("  [x] Strang ODE solver    analytical equilibrium fast + Euler slow");
    println!("  [x] Vertical leaching    8-layer clay-retained cascade");
    println!("  [x] Alluvial P renewal   flood-deposited P (Nile/Maya mechanic)");
    println!("  [x] FastScape Δh piggyback erosion/deposition carries nutrients");
    println!("  [x] Phillips pedogenesis dS/dt = c1·exp(-k1·S) - c2·exp(-k2/S)");
    println!("  [x] Lyapunov tracking    soil divergence accumulator per column");
    println!("  [x] S → Kf/Kd feedback  mature soil resists incision, promotes creep");
    println!("  [x] Growth multiplier    Liebig min(f_P, f_K) for BDI agents\n");
    println!("  Total sim time:  {}", sim.clock.summary());
    println!("  Geo epochs run:  {}", sim.clock.geo_epoch);
    println!("\n  NEXT -> Phase 3: Flesh & Genes");
    println!("         BDI agents, Lotka-Volterra ecology, faunalturbation,");
    println!("         plant-soil feedback, yield crisis → collapse cascade");
    println!("================================================================");
}

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
    println!("  Mean P_labile    : {:.2} mg/kg  (plant-accessible inorganic P)", sum_p / c);
    println!("  Mean K_exch      : {:.1} mg/kg  (exchangeable K on CEC sites)", sum_k / c);
    println!("  Mean P_occluded  : {:.1} mg/kg  (Fe/Al-bound irreversible sink)", sum_oc / c);
    println!("  Mean soil dev S  : {:.3}        (0=bare parent, 1=mature)", sum_s / c);
}

fn demo_maya_collapse() {
    let mut col = NutrientColumn::new_alluvial(0.4);
    println!("  Baseline (alluvial floodplain): P_labile={:.1}, K_exch={:.0}, growth={:.2}",
        col.surface_p_labile(), col.surface_k_exch(), col.column_growth_multiplier());

    for cycle in 1..=3 {
        // 50 yr cultivation: bare soil, high runoff, no canopy P trapping
        let farm = ColumnEnv {
            temp_c: 26.0, moisture: 0.8, runoff_m_yr: 0.7,
            flooded: false, flood_p_input: 0.0, delta_h_m: 0.0, veg_cover: 0.05,
        };
        update_column(&mut col, &farm, 50.0);

        // 20 yr fallow: partial regrowth
        let fallow = ColumnEnv { veg_cover: 0.4, ..farm };
        update_column(&mut col, &fallow, 20.0);

        let gm = col.column_growth_multiplier();
        println!("  Cycle {:}: P_labile={:.1}, K_exch={:.0}, growth={:.2}  → {}",
            cycle,
            col.surface_p_labile(), col.surface_k_exch(), gm,
            match gm {
                g if g > 0.6 => "Productive",
                g if g > 0.4 => "Stressed",
                g if g > 0.2 => "CRISIS",
                _             => "COLLAPSE",
            }
        );
    }

    // Flood renewal — simulate river reconnection / dam removal
    let flood = ColumnEnv {
        flooded: true, flood_p_input: 12.0, veg_cover: 0.3,
        runoff_m_yr: 0.5, temp_c: 26.0, moisture: 0.9, delta_h_m: 0.1,
    };
    update_column(&mut col, &flood, 100.0);
    let gm = col.column_growth_multiplier();
    println!("  After 100yr flood renewal: P_labile={:.1}, K_exch={:.0}, growth={:.2}  → {}",
        col.surface_p_labile(), col.surface_k_exch(), gm,
        if gm > 0.4 { "Recovery underway" } else { "Still collapsed" }
    );
}

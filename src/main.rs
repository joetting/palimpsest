// ============================================================================
// Deep Time Engine — Phase 1: The "Body Without Organs"
//
// A materialist voxel simulation engine built in Rust.
// Based on Manuel DeLanda's "A Thousand Years of Nonlinear History"
//
// Phase 1 establishes:
//   ✓ SVO/ECS Data Structure (World, TerrainColumn entities, Components)
//   ✓ FastScape Landscape Evolution (O(N) implicit SPL solver)
//   ✓ Strang Operator Splitting (multirate scheduler)
//   ✓ Activity Bitmask (sparse computation)
//   ✓ Temporal Cohort Spreading (staggered updates)
//   ✓ GPU Compute Pipeline Architecture (CPU stub + WGSL templates)
//   ✓ Tectonic Forcing Interface (Elder God interventions)
// ============================================================================

mod ecs;
mod terrain;
mod compute;
mod math;
mod config;

use config::{DeepTimeSimulation, SimulationConfig};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  DEEP TIME ENGINE — Phase 1: The Body Without Organs");
    println!("  A Rust-based materialist voxel simulation");
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

    println!("\n[PHASE 1] ECS World initialized");
    println!("  Grid: {}x{} columns ({} entities)",
        sim.config.grid_width, sim.config.grid_height,
        sim.world.total_columns());
    println!("  Cell size: {} km", sim.config.cell_size_m / 1000.0);
    println!("  Domain: {}x{} km",
        sim.config.grid_width as f32 * sim.config.cell_size_m / 1000.0,
        sim.config.grid_height as f32 * sim.config.cell_size_m / 1000.0);

    println!("\n[PHASE 1] Component system demo -- sample columns:");
    for pos in [(0u32, 0u32), (32, 32), (64, 64), (96, 96), (127, 127)] {
        let col = sim.world.column(pos.0, pos.1);
        println!("  ({:3},{:3}): h={:6.1}m  mat={:?}  Kf={:.2e}  Kd={:.3}  cohort={}",
            pos.0, pos.1,
            col.elevation.0,
            col.layers.surface_material(),
            col.erodibility.0,
            col.diffusivity.0,
            col.cohort.0,
        );
    }

    println!("\n[PHASE 1] Activity bitmask:");
    println!("  Active columns: {}", sim.compute.active_count());
    println!("  Skip ratio: {:.1}%",
        (1.0 - sim.compute.active_count() as f32
            / (sim.config.grid_width * sim.config.grid_height) as f32) * 100.0);

    let cohort_size = sim.world.total_columns() / sim.config.num_cohorts as usize;
    println!("\n[PHASE 1] Temporal cohort spreading:");
    println!("  Cohorts: {}  (~{} columns each)", sim.config.num_cohorts, cohort_size);
    println!("  CPU load per frame: ~{:.1}%", 100.0 / sim.config.num_cohorts as f32);

    println!("\n[ELDER GOD] Adding tectonic hotspot -- mountain-building event");
    sim.add_uplift_hotspot(64, 64, 25.0, 0.002);

    println!("\n[FASTSCAPE] Running geological simulation...\n");
    println!("  {:<12} {:<10} {:<10} {:<10} {:<10}",
        "Epoch", "Time", "MaxH(m)", "MeanH(m)", "Active%");
    println!("  {}", "-".repeat(60));

    let n_epochs = 20;
    let social_ticks_per_epoch = 500;

    for epoch in 0..n_epochs {
        for _ in 0..social_ticks_per_epoch {
            let result = sim.clock.advance_social_tick(1.0 / 60.0);
            if result.fire_geo {
                sim.step_geological_epoch();
                break;
            }
        }

        if epoch % 4 == 0 || epoch == n_epochs - 1 {
            let max_h = sim.solver.max_elevation(&sim.heights);
            let mean_h = sim.solver.mean_elevation(&sim.heights);
            let active_pct = sim.compute.active_count() as f32
                / (sim.config.grid_width * sim.config.grid_height) as f32 * 100.0;
            let elapsed_ky = sim.clock.geo_elapsed_years / 1000.0;

            println!("  {:<12} {:<10} {:<10.1} {:<10.1} {:<10.1}",
                format!("#{}", sim.clock.geo_epoch),
                format!("{:.1} ka", elapsed_ky),
                max_h, mean_h, active_pct);
        }
    }

    println!("\n[ELDER GOD] Triggering major orogenic event...");
    sim.add_uplift_hotspot(32, 96, 40.0, 0.005);

    println!("\n[FASTSCAPE] Continuing with active orogeny...\n");
    println!("  {:<12} {:<10} {:<10} {:<10} {:<10}",
        "Epoch", "Time", "MaxH(m)", "MeanH(m)", "Active%");
    println!("  {}", "-".repeat(60));

    for epoch in 0..n_epochs {
        for _ in 0..social_ticks_per_epoch {
            let result = sim.clock.advance_social_tick(1.0 / 60.0);
            if result.fire_geo {
                sim.step_geological_epoch();
                break;
            }
        }

        if epoch % 4 == 0 || epoch == n_epochs - 1 {
            let max_h = sim.solver.max_elevation(&sim.heights);
            let mean_h = sim.solver.mean_elevation(&sim.heights);
            let active_pct = sim.compute.active_count() as f32
                / (sim.config.grid_width * sim.config.grid_height) as f32 * 100.0;
            let elapsed_ky = sim.clock.geo_elapsed_years / 1000.0;

            println!("  {:<12} {:<10} {:<10.1} {:<10.1} {:<10.1}",
                format!("#{}", sim.clock.geo_epoch),
                format!("{:.1} ka", elapsed_ky),
                max_h, mean_h, active_pct);
        }
    }

    println!("\n================================================================");
    println!("  PHASE 1 COMPLETE -- Architecture Status");
    println!("================================================================");
    println!();
    println!("  [x] ECS World            {} entities, SoA layout ready for GPU", sim.world.total_columns());
    println!("  [x] FastScape Solver     D8 routing -> stack sort -> implicit SPL");
    println!("  [x] Strang Splitting     Social|Geo|Bio multirate schedules");
    println!("  [x] Activity Bitmask     {:.1}% columns skippable",
        (1.0 - sim.compute.active_count() as f32
            / (sim.config.grid_width * sim.config.grid_height) as f32) * 100.0);
    println!("  [x] Cohort Spreading     {} cohorts, ~{:.0}% CPU load/frame",
        sim.config.num_cohorts, 100.0 / sim.config.num_cohorts as f32);
    println!("  [x] Tectonic Forcing     {} active hotspots", sim.tectonics.hotspots.len());
    println!("  [x] GPU Pipeline         WGSL shaders templated, SoA layout correct");
    println!();
    println!("  Total sim time:  {}", sim.clock.summary());
    println!("  Geo epochs run:  {}", sim.clock.geo_epoch);
    println!("  Social ticks:    {}", sim.clock.social_tick);
    println!();
    println!("  NEXT -> Phase 2: Pedogenesis & Biogeochemistry (P-K cycle)");
    println!("         Phase 3: Biological agents (BDI + Lotka-Volterra)");
    println!("         Phase 4: WGPU GPU integration (actual compute shaders)");
    println!("================================================================");
}

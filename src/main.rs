pub mod common;
pub mod heightmap;
pub mod lod;
pub mod svo;
pub mod math;
pub mod ecs;
pub mod compute;
pub mod climate;
pub mod geology;
pub mod soil;
pub mod hydrology;
pub mod world;
pub mod diag;
pub mod viz;

use common::{EngineConfig, Vec3};
use geology::TectonicForcing;
use lod::LodManager;
use world::{WorldConfig, WorldSimulation};

use std::path::Path;
use std::time::Instant;

fn main() {
    println!("=== Voxel Engine: Full Solver Integration ===\n");

    let output_dir = Path::new("output");
    std::fs::create_dir_all(output_dir).ok();

    // ------------------------------------------------------------------
    // 1. Configure world simulation
    // ------------------------------------------------------------------
    let grid_w: usize = 128;
    let grid_h: usize = 128;
    let cell_size_m: f32 = 1000.0;

    let mut tectonic = TectonicForcing::new(0.001);
    let center_x = (grid_w as f32 / 2.0) * cell_size_m;
    let center_y = (grid_h as f32 / 2.0) * cell_size_m;
    tectonic.add_hotspot(center_x, center_y, 30_000.0, 0.005);
    // Secondary ridge
    tectonic.add_hotspot(center_x * 0.3, center_y * 0.7, 20_000.0, 0.003);

    let world_config = WorldConfig {
        grid_width: grid_w,
        grid_height: grid_h,
        cell_size_m,
        sea_level: 0.0,
        dt_years: 5000.0,
        spl_params: geology::SplParams {
            cell_size: cell_size_m,
            m: 0.5,
            n: 1.0,
            rainfall: 1.0,
            sea_level: 0.0,
            ..Default::default()
        },
        climate_config: climate::ClimateConfig::default(),
        wind_params: climate::WindParams::default(),
        tectonic,
        kf_base: 1e-5,
        kd_base: 0.01,
        hydrology_config: crate::hydrology::HydrologyConfig::default(),
    };

    // ------------------------------------------------------------------
    // 2. Generate initial terrain
    // ------------------------------------------------------------------
    let n = grid_w * grid_h;
    let initial_elevations: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i % grid_w) as f32;
            let y = (i / grid_w) as f32;
            let fx = x / grid_w as f32 * std::f32::consts::PI * 4.0;
            let fy = y / grid_h as f32 * std::f32::consts::PI * 4.0;
            50.0 + 30.0 * fx.sin() * fy.cos() + 10.0 * (fx * 2.5).sin()
        })
        .collect();

    // ------------------------------------------------------------------
    // 3. Initialize
    // ------------------------------------------------------------------
    let t = Instant::now();
    let mut world = WorldSimulation::new(world_config, initial_elevations);
    println!(
        "World initialized in {:.1}ms: {}x{} grid ({} cells, {:.0}km x {:.0}km)",
        t.elapsed().as_secs_f64() * 1000.0,
        grid_w, grid_h, n,
        grid_w as f32 * cell_size_m / 1000.0,
        grid_h as f32 * cell_size_m / 1000.0,
    );
    println!("  Climate: {}", world.climate.summary());

    // Render initial state
    viz::render_epoch(&world, None, output_dir, 0);
    println!("  Rendered epoch 0 PNGs");

    // ------------------------------------------------------------------
    // 4. Run geological epochs with PNG output
    // ------------------------------------------------------------------
    let n_epochs = 20;
    let render_interval = 5; // render every N epochs
    println!(
        "\nRunning {} epochs ({:.1} Myr total), rendering every {}...\n",
        n_epochs,
        n_epochs as f64 * world.config.dt_years / 1e6,
        render_interval,
    );

    let t = Instant::now();
    for i in 1..=n_epochs {
        let report = world.step_epoch();
        println!("{}", report);

        if i % render_interval == 0 || i == n_epochs {
            viz::render_epoch(&world, None, output_dir, world.epoch);
        }
    }
    let total_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "\n{} epochs in {:.0}ms ({:.1}ms/epoch)",
        n_epochs, total_ms, total_ms / n_epochs as f64,
    );

    // ------------------------------------------------------------------
    // 5. Elder God CO₂ injection
    // ------------------------------------------------------------------
    println!("\n--- Elder God CO₂ injection: +1500 ppm ---");
    world.inject_co2(1500.0);
    for _ in 0..5 {
        let report = world.step_epoch();
        println!("{}", report);
    }
    viz::render_epoch(&world, None, output_dir, world.epoch);

    // ------------------------------------------------------------------
    // 6. Biome distribution
    // ------------------------------------------------------------------
    let biomes = world.climate.biome_counts(&world.elevations, world.config.sea_level);
    println!("\nBiome distribution:");
    for (biome, count) in &biomes {
        if *count > 0 {
            println!("  {:>20}: {} cells ({:.1}%)", biome.name(), count, *count as f64 / n as f64 * 100.0);
        }
    }

    // ------------------------------------------------------------------
    // 7. Local 3D zone: SVO instantiation + rendering
    // ------------------------------------------------------------------
    println!("\n--- Local 3D Zone ---");

    let engine_config = EngineConfig {
        global_resolution: grid_w as u32,
        svo_max_depth: 6, // 64^3
        voxel_size: cell_size_m as f64,
        local_radius: 16.0 * cell_size_m as f64,
        lod_tiers: 3,
    };

    // Build the heightmap adapter from world state
    let mut hm = crate::heightmap::Heightmap::new(&engine_config);
    hm.par_for_each_mut(|x, y, cell| {
        let idx = (y as usize) * grid_w + (x as usize).min(grid_w - 1);
        let elev = world.elevations.get(idx).copied().unwrap_or(0.0);
        cell.elevation = elev;
        cell.bedrock = elev * 0.7;
        cell.sediment_depth = elev * 0.3;
        if let Some(col) = world.climate.columns.get(idx) {
            cell.temperature = col.temp_c;
        }
        if let Some(nc) = world.nutrient_columns.get(idx) {
            cell.nutrient = nc.surface_p_labile();
        }
    });

    let player_pos = Vec3::new(center_x as f64, 200.0, center_y as f64);
    let mut lod_manager = LodManager::new(&engine_config);

    let t = Instant::now();
    lod_manager.update(&player_pos, &hm);
    let svo_ms = t.elapsed().as_secs_f64() * 1000.0;

    if let Some(svo_ref) = lod_manager.svo() {
        println!(
            "SVO: {} nodes ({} leaves, {} branches), {:.1} KB, built in {:.1}ms",
            svo_ref.node_count(), svo_ref.leaf_count(), svo_ref.branch_count(),
            svo_ref.memory_bytes() as f64 / 1024.0, svo_ms,
        );

        // Render SVO slices
        viz::render_epoch(&world, Some(svo_ref), output_dir, world.epoch);
        println!("  SVO slices rendered");
    }

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    println!("\n=== Final State ===");
    println!("  Climate: {}", world.climate.summary());
    println!("  Max elevation: {:.0}m", world.fastscape.max_elevation(&world.elevations));
    println!("  Mean soil S: {:.3}", world.pedo_solver.mean_s(&world.pedo_states));

    // List output files
    let mut files: Vec<_> = std::fs::read_dir(output_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    files.sort();
    println!("\n  Output PNGs ({} files):", files.len());
    for f in &files {
        println!("    {}", f);
    }

    println!("\n=== Done. ===");
}

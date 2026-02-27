use fastscape_rs::*;
use fastscape_rs::erosion::{StreamPowerParams, DiffusionParams};
use fastscape_rs::climate::{LinearTheoryParams, EbmParams};
use fastscape_rs::pedogenesis::PedogenesisParams;
use fastscape_rs::agents::AgentParams;
use image::{ImageBuffer, Rgb};
use std::path::Path;
use rand::Rng;
use std::f64::consts::FRAC_PI_4;

fn main() {
    println!("================================================================");
    println!("  fastscape-rs v0.2");
    println!("  Nonlinear Pedogenesis + BDI Agents + Multi-Scale Coupling");
    println!("  Solutions: Variable K_d | On-the-fly K_f | eps-regularized dR/dt");
    println!("             Influence Maps | Temporal Sub-Stepping");
    println!("================================================================");
    println!();

    let out_dir = Path::new("output");
    std::fs::create_dir_all(out_dir).expect("Failed to create output directory");

    let mut rng = rand::thread_rng();
    let random_wind_dir = rng.gen_range(-FRAC_PI_4..FRAC_PI_4);

    let lt_params = LinearTheoryParams {
        p0: 0.5,
        wind_speed: 15.0,
        wind_direction: random_wind_dir,
        hw: 3000.0,
        tau_c: 1500.0,
        tau_f: 1500.0,
        ..Default::default()
    };

    let spl_params = StreamPowerParams {
        k_f: 2e-5,
        k_f_sed: 1e-4,
        m: 0.5,
        n: 1.0,
        uplift_rate: 5e-4,
        alpha_erodibility: 0.5, // Solution #2: K_f / (1 + 0.5*S)
    };

    let config = SimulationConfig {
        rows: 128,
        cols: 128,
        dx: 500.0,
        dy: 500.0,
        total_time: 2_000_000.0,
        dt_geomorph: 5_000.0,
        climate: ClimateModel::LinearTheory(lt_params),
        erosion: ErosionModel::StreamPower(spl_params),
        ebm: EbmParams::default(),
        diffusion: DiffusionParams {
            k_d: 0.01,
            s_c: 0.8,
            beta_diffusivity: 0.3, // Solution #1: K_d * (1 + 0.3*S)
        },
        pedogenesis: PedogenesisParams {
            c1: 0.005,
            k1: 0.1,
            c2: 0.003,
            k2: 0.5,
            epsilon: 0.01,       // Solution #3
            s_min: 0.05,         // Solution #3
            erosion_coupling: 1.0,
            bio_progressive_coupling: 1.0,
            bio_regressive_coupling: 1.0,
        },
        agents: AgentParams {
            initial_count: 300,
            dt_bio: 1.0,
            representative_steps: 100, // Solution #5: 100 repr. steps per macro dt
            perception_radius: 3,
            move_probability: 0.3,
            grazing_intensity: 0.002,
            decomposition_intensity: 0.003,
            farming_progressive: 0.002,
            farming_regressive: 0.001,
            forester_progressive: 0.004,
            hunger_threshold: 0.3,
            energy_recovery: 0.1,
            energy_cost: 0.02,
        },
        enable_biogeochem: true,
        enable_pedogenesis: true,
        enable_agents: true,
        init_elevation: 0.0,
        init_perturbation: 1.0,
        seed: 12345,
    };

    let (grid, diagnostics) = run_simulation(&config);

    println!("\nGenerating PNG outputs...");

    save_elevation_png(&grid, &out_dir.join("elevation.png"));
    println!("  -> elevation.png");

    save_precipitation_png(&grid, &out_dir.join("precipitation.png"));
    println!("  -> precipitation.png");

    save_slope_png(&grid, &out_dir.join("slope.png"));
    println!("  -> slope.png");

    save_drainage_png(&grid, &out_dir.join("drainage_area.png"));
    println!("  -> drainage_area.png");

    save_temperature_png(&grid, &out_dir.join("temperature.png"));
    println!("  -> temperature.png");

    save_soil_carbon_png(&grid, &out_dir.join("soil_carbon.png"));
    println!("  -> soil_carbon.png");

    save_soil_state_png(&grid, &out_dir.join("soil_state.png"));
    println!("  -> soil_state.png (pedodiversity)");

    save_timeseries_png(&diagnostics, &out_dir.join("timeseries.png"));
    println!("  -> timeseries.png");

    println!("\n--- Final Statistics ---");
    if let Some(d) = diagnostics.last() {
        println!("  Mean elevation:     {:.1} m", d.mean_elevation);
        println!("  Max elevation:      {:.1} m", d.max_elevation);
        println!("  Mean precipitation: {:.4} m/yr", d.mean_precipitation);
        println!("  Mean erosion rate:  {:.6} m/yr", d.mean_erosion_rate);
        println!("  Mean soil state S:  {:.4}", d.mean_soil_state);
        println!("  Max soil state S:   {:.4}", d.max_soil_state);
        println!("  Living agents:      {}", d.agent_count);
    }

    println!("\nAll outputs saved to ./output/");
}

// ============================================================================
// Colormaps
// ============================================================================

fn lerp3(a: (f64, f64, f64), b: (f64, f64, f64), t: f64) -> (f64, f64, f64) {
    (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t, a.2 + (b.2 - a.2) * t)
}

fn terrain_colormap(t: f64) -> Rgb<u8> {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.0001 { (30.0, 60.0, 120.0) }
    else if t < 0.15 { lerp3((30.0, 100.0, 50.0), (60.0, 140.0, 60.0), t / 0.15) }
    else if t < 0.4 { lerp3((60.0, 140.0, 60.0), (180.0, 160.0, 80.0), (t - 0.15) / 0.25) }
    else if t < 0.7 { lerp3((180.0, 160.0, 80.0), (140.0, 100.0, 60.0), (t - 0.4) / 0.3) }
    else if t < 0.9 { lerp3((140.0, 100.0, 60.0), (180.0, 180.0, 180.0), (t - 0.7) / 0.2) }
    else { lerp3((180.0, 180.0, 180.0), (255.0, 255.0, 255.0), (t - 0.9) / 0.1) };
    Rgb([r as u8, g as u8, b as u8])
}

fn precip_colormap(t: f64) -> Rgb<u8> {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.3 { lerp3((255.0, 250.0, 240.0), (180.0, 210.0, 240.0), t / 0.3) }
    else if t < 0.7 { lerp3((180.0, 210.0, 240.0), (50.0, 100.0, 200.0), (t - 0.3) / 0.4) }
    else { lerp3((50.0, 100.0, 200.0), (10.0, 20.0, 120.0), (t - 0.7) / 0.3) };
    Rgb([r as u8, g as u8, b as u8])
}

fn temperature_colormap(t: f64) -> Rgb<u8> {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 { lerp3((30.0, 60.0, 180.0), (240.0, 240.0, 245.0), t / 0.5) }
    else { lerp3((240.0, 240.0, 245.0), (200.0, 30.0, 30.0), (t - 0.5) / 0.5) };
    Rgb([r as u8, g as u8, b as u8])
}

fn carbon_colormap(t: f64) -> Rgb<u8> {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 { lerp3((40.0, 30.0, 20.0), (60.0, 120.0, 40.0), t / 0.5) }
    else { lerp3((60.0, 120.0, 40.0), (150.0, 220.0, 80.0), (t - 0.5) / 0.5) };
    Rgb([r as u8, g as u8, b as u8])
}

/// Soil state colormap: low S = bare rock (grey) → high S = rich soil (deep brown/green)
fn soil_state_colormap(t: f64) -> Rgb<u8> {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.2 {
        lerp3((160.0, 160.0, 160.0), (140.0, 110.0, 70.0), t / 0.2)
    } else if t < 0.5 {
        lerp3((140.0, 110.0, 70.0), (80.0, 60.0, 30.0), (t - 0.2) / 0.3)
    } else if t < 0.8 {
        lerp3((80.0, 60.0, 30.0), (40.0, 90.0, 30.0), (t - 0.5) / 0.3)
    } else {
        lerp3((40.0, 90.0, 30.0), (20.0, 140.0, 50.0), (t - 0.8) / 0.2)
    };
    Rgb([r as u8, g as u8, b as u8])
}

// ============================================================================
// PNG Save Functions
// ============================================================================

fn save_elevation_png(grid: &grid::TerrainGrid, path: &Path) {
    let (rows, cols) = (grid.rows, grid.cols);
    let min = grid.elevation.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = grid.elevation.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-10);
    let img = ImageBuffer::from_fn(cols as u32, rows as u32, |x, y| {
        let t = (grid.elevation[[y as usize, x as usize]] - min) / range;
        terrain_colormap(t)
    });
    img.save(path).expect("Failed to save elevation PNG");
}

fn save_precipitation_png(grid: &grid::TerrainGrid, path: &Path) {
    let (rows, cols) = (grid.rows, grid.cols);
    let min = grid.precipitation.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = grid.precipitation.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-10);
    let img = ImageBuffer::from_fn(cols as u32, rows as u32, |x, y| {
        let t = (grid.precipitation[[y as usize, x as usize]] - min) / range;
        precip_colormap(t)
    });
    img.save(path).expect("Failed to save precipitation PNG");
}

fn save_slope_png(grid: &grid::TerrainGrid, path: &Path) {
    let (rows, cols) = (grid.rows, grid.cols);
    let max = grid.slope.iter().cloned().fold(0.0f64, f64::max).max(1e-10);
    let img = ImageBuffer::from_fn(cols as u32, rows as u32, |x, y| {
        let t = (grid.slope[[y as usize, x as usize]] / max).clamp(0.0, 1.0);
        let v = (t * 255.0) as u8;
        Rgb([v, v, v])
    });
    img.save(path).expect("Failed to save slope PNG");
}

fn save_drainage_png(grid: &grid::TerrainGrid, path: &Path) {
    let (rows, cols) = (grid.rows, grid.cols);
    let log_min = grid.drainage_area.iter().cloned()
        .filter(|&v| v > 0.0).fold(f64::INFINITY, f64::min).ln();
    let log_max = grid.drainage_area.iter().cloned()
        .fold(f64::NEG_INFINITY, f64::max).max(1.0).ln();
    let log_range = (log_max - log_min).max(1e-10);
    let img = ImageBuffer::from_fn(cols as u32, rows as u32, |x, y| {
        let val = grid.drainage_area[[y as usize, x as usize]].max(1.0).ln();
        let t = ((val - log_min) / log_range).clamp(0.0, 1.0);
        let (r, g, b) = if t < 0.5 { lerp3((5.0, 5.0, 20.0), (20.0, 80.0, 180.0), t / 0.5) }
        else { lerp3((20.0, 80.0, 180.0), (180.0, 240.0, 255.0), (t - 0.5) / 0.5) };
        Rgb([r as u8, g as u8, b as u8])
    });
    img.save(path).expect("Failed to save drainage area PNG");
}

fn save_temperature_png(grid: &grid::TerrainGrid, path: &Path) {
    let (rows, cols) = (grid.rows, grid.cols);
    let min = grid.temperature.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = grid.temperature.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-10);
    let img = ImageBuffer::from_fn(cols as u32, rows as u32, |x, y| {
        let t = (grid.temperature[[y as usize, x as usize]] - min) / range;
        temperature_colormap(t)
    });
    img.save(path).expect("Failed to save temperature PNG");
}

fn save_soil_carbon_png(grid: &grid::TerrainGrid, path: &Path) {
    let (rows, cols) = (grid.rows, grid.cols);
    let min = grid.soil_carbon.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = grid.soil_carbon.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-10);
    let img = ImageBuffer::from_fn(cols as u32, rows as u32, |x, y| {
        let t = (grid.soil_carbon[[y as usize, x as usize]] - min) / range;
        carbon_colormap(t)
    });
    img.save(path).expect("Failed to save soil carbon PNG");
}

fn save_soil_state_png(grid: &grid::TerrainGrid, path: &Path) {
    let (rows, cols) = (grid.rows, grid.cols);
    let min = grid.soil_state.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = grid.soil_state.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-10);
    let img = ImageBuffer::from_fn(cols as u32, rows as u32, |x, y| {
        let t = (grid.soil_state[[y as usize, x as usize]] - min) / range;
        soil_state_colormap(t)
    });
    img.save(path).expect("Failed to save soil state PNG");
}

fn save_timeseries_png(diagnostics: &[StepDiagnostics], path: &Path) {
    let w: u32 = 800;
    let h: u32 = 400;
    let margin: u32 = 50;
    let plot_w = w - 2 * margin;
    let plot_h = h - 2 * margin;

    let mut img = ImageBuffer::from_pixel(w, h, Rgb([255u8, 255, 255]));

    if diagnostics.is_empty() {
        img.save(path).expect("Failed to save timeseries PNG");
        return;
    }

    let t_max = diagnostics.last().map(|d| d.time).unwrap_or(1.0);
    let h_max_val = diagnostics.iter().map(|d| d.max_elevation).fold(0.0f64, f64::max).max(1.0);

    // Axes
    for x in margin..=(w - margin) { img.put_pixel(x, h - margin, Rgb([0, 0, 0])); }
    for y in margin..=(h - margin) { img.put_pixel(margin, y, Rgb([0, 0, 0])); }

    // Max elevation (red)
    let mut prev: Option<(u32, u32)> = None;
    for d in diagnostics {
        let px = margin + ((d.time / t_max) * plot_w as f64) as u32;
        let py = h - margin - ((d.max_elevation / h_max_val) * plot_h as f64).min(plot_h as f64) as u32;
        if let Some((px0, py0)) = prev { draw_line(&mut img, px0, py0, px, py, Rgb([200, 40, 40])); }
        prev = Some((px, py));
    }

    // Mean elevation (blue)
    prev = None;
    for d in diagnostics {
        let px = margin + ((d.time / t_max) * plot_w as f64) as u32;
        let py = h - margin - ((d.mean_elevation / h_max_val) * plot_h as f64).min(plot_h as f64) as u32;
        if let Some((px0, py0)) = prev { draw_line(&mut img, px0, py0, px, py, Rgb([40, 80, 200])); }
        prev = Some((px, py));
    }

    // Mean soil state (brown, scaled separately)
    let s_max = diagnostics.iter().map(|d| d.max_soil_state).fold(0.0f64, f64::max).max(1e-10);
    prev = None;
    for d in diagnostics {
        let px = margin + ((d.time / t_max) * plot_w as f64) as u32;
        let py = h - margin - ((d.mean_soil_state / s_max) * plot_h as f64).min(plot_h as f64) as u32;
        if let Some((px0, py0)) = prev { draw_line(&mut img, px0, py0, px, py, Rgb([120, 80, 30])); }
        prev = Some((px, py));
    }

    // Legend
    for dx in 0..12u32 {
        for dy in 0..6u32 {
            img.put_pixel(margin + 10 + dx, margin + 10 + dy, Rgb([200, 40, 40]));
            img.put_pixel(margin + 10 + dx, margin + 22 + dy, Rgb([40, 80, 200]));
            img.put_pixel(margin + 10 + dx, margin + 34 + dy, Rgb([120, 80, 30]));
        }
    }

    img.save(path).expect("Failed to save timeseries PNG");
}

fn draw_line(img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, x0: u32, y0: u32, x1: u32, y1: u32, color: Rgb<u8>) {
    let (w, h) = (img.width(), img.height());
    let dx = (x1 as i32 - x0 as i32).abs();
    let dy = -(y1 as i32 - y0 as i32).abs();
    let sx: i32 = if x0 < x1 { 1 } else { -1 };
    let sy: i32 = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0 as i32;
    let mut y = y0 as i32;
    loop {
        if x >= 0 && x < w as i32 && y >= 0 && y < h as i32 {
            img.put_pixel(x as u32, y as u32, color);
            if y + 1 < h as i32 { img.put_pixel(x as u32, (y + 1) as u32, color); }
        }
        if x == x1 as i32 && y == y1 as i32 { break; }
        let e2 = 2 * err;
        if e2 >= dy { err += dy; x += sx; }
        if e2 <= dx { err += dx; y += sy; }
    }
}

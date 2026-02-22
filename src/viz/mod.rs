/// Visualization: renders simulation state to PNGs.
///
/// Produces two scales of output per epoch:
/// - **Global** (heightmap-resolution): elevation, biome, CO₂/temperature, soil-S, nutrients
/// - **Local** (SVO cross-section): voxel material slice at player position

use image::{Rgb, RgbImage};
use std::path::Path;

use crate::climate::Biome;
use crate::world::WorldSimulation;

// ---------------------------------------------------------------------------
// Color maps
// ---------------------------------------------------------------------------

fn elevation_color(elev: f32, sea_level: f32, max_elev: f32) -> Rgb<u8> {
    if elev <= sea_level {
        // Ocean: deep blue → light blue by depth
        let d = ((sea_level - elev) / 200.0).clamp(0.0, 1.0);
        Rgb([
            (30.0 + 40.0 * (1.0 - d)) as u8,
            (60.0 + 80.0 * (1.0 - d)) as u8,
            (120.0 + 100.0 * (1.0 - d)) as u8,
        ])
    } else {
        let t = ((elev - sea_level) / (max_elev - sea_level).max(1.0)).clamp(0.0, 1.0);
        if t < 0.2 {
            // Low: green
            let s = t / 0.2;
            Rgb([(50.0 + 30.0 * s) as u8, (120.0 + 60.0 * s) as u8, (40.0 + 20.0 * s) as u8])
        } else if t < 0.5 {
            // Mid: yellow-brown
            let s = (t - 0.2) / 0.3;
            Rgb([(80.0 + 100.0 * s) as u8, (180.0 - 60.0 * s) as u8, (60.0 - 20.0 * s) as u8])
        } else if t < 0.8 {
            // High: grey rock
            let s = (t - 0.5) / 0.3;
            Rgb([(180.0 + 40.0 * s) as u8, (120.0 + 60.0 * s) as u8, (40.0 + 100.0 * s) as u8])
        } else {
            // Peak: white snow
            let s = (t - 0.8) / 0.2;
            Rgb([(220.0 + 35.0 * s) as u8, (180.0 + 75.0 * s) as u8, (140.0 + 115.0 * s) as u8])
        }
    }
}

fn biome_color(biome: &Biome) -> Rgb<u8> {
    match biome {
        Biome::Ocean => Rgb([30, 60, 150]),
        Biome::TropicalForest => Rgb([0, 100, 0]),
        Biome::TemperateForest => Rgb([34, 139, 34]),
        Biome::Grassland => Rgb([154, 205, 50]),
        Biome::Desert => Rgb([210, 180, 100]),
        Biome::Tundra => Rgb([176, 196, 222]),
        Biome::Alpine => Rgb([139, 137, 137]),
        Biome::Glacier => Rgb([230, 240, 255]),
    }
}

fn scalar_color(val: f32, min: f32, max: f32, cold: Rgb<u8>, hot: Rgb<u8>) -> Rgb<u8> {
    let t = ((val - min) / (max - min).max(1e-6)).clamp(0.0, 1.0);
    Rgb([
        (cold.0[0] as f32 + t * (hot.0[0] as f32 - cold.0[0] as f32)) as u8,
        (cold.0[1] as f32 + t * (hot.0[1] as f32 - cold.0[1] as f32)) as u8,
        (cold.0[2] as f32 + t * (hot.0[2] as f32 - cold.0[2] as f32)) as u8,
    ])
}

fn soil_s_color(s: f32) -> Rgb<u8> {
    // 0 = bare rock (grey) → 1 = deep soil (dark brown)
    let t = s.clamp(0.0, 1.0);
    Rgb([
        (180.0 - 120.0 * t) as u8,
        (170.0 - 100.0 * t) as u8,
        (160.0 - 130.0 * t) as u8,
    ])
}

fn nutrient_color(val: f32, max_val: f32) -> Rgb<u8> {
    let t = (val / max_val.max(1.0)).clamp(0.0, 1.0);
    // Black → green → yellow
    if t < 0.5 {
        let s = t / 0.5;
        Rgb([0, (200.0 * s) as u8, 0])
    } else {
        let s = (t - 0.5) / 0.5;
        Rgb([(255.0 * s) as u8, 200, 0])
    }
}

// ---------------------------------------------------------------------------
// Global-scale PNG renders
// ---------------------------------------------------------------------------

pub fn render_global_elevation(world: &WorldSimulation, path: &Path) {
    let w = world.config.grid_width;
    let h = world.config.grid_height;
    let max_elev = world.elevations.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sea = world.config.sea_level;

    let mut img = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let color = elevation_color(world.elevations[idx], sea, max_elev);
            img.put_pixel(x as u32, y as u32, color);
        }
    }
    img.save(path).expect("Failed to save elevation PNG");
}

pub fn render_global_biome(world: &WorldSimulation, path: &Path) {
    let w = world.config.grid_width;
    let h = world.config.grid_height;

    let mut img = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let color = if world.elevations[idx] <= world.config.sea_level {
                biome_color(&Biome::Ocean)
            } else {
                biome_color(&world.climate.columns[idx].biome)
            };
            img.put_pixel(x as u32, y as u32, color);
        }
    }
    img.save(path).expect("Failed to save biome PNG");
}

pub fn render_global_temperature(world: &WorldSimulation, path: &Path) {
    let w = world.config.grid_width;
    let h = world.config.grid_height;

    let mut img = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let temp = world.climate.columns[idx].temp_c;
            let color = scalar_color(
                temp, -20.0, 40.0,
                Rgb([0, 0, 200]),   // cold: blue
                Rgb([200, 30, 0]),  // hot: red
            );
            img.put_pixel(x as u32, y as u32, color);
        }
    }
    img.save(path).expect("Failed to save temperature PNG");
}

pub fn render_global_precipitation(world: &WorldSimulation, path: &Path) {
    let w = world.config.grid_width;
    let h = world.config.grid_height;

    let mut img = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let precip = world.climate.columns[idx].precipitation_m;
            let color = scalar_color(
                precip, 0.0, 3.0,
                Rgb([240, 220, 180]),  // dry: tan
                Rgb([0, 30, 180]),     // wet: deep blue
            );
            img.put_pixel(x as u32, y as u32, color);
        }
    }
    img.save(path).expect("Failed to save precipitation PNG");
}

pub fn render_global_soil(world: &WorldSimulation, path: &Path) {
    let w = world.config.grid_width;
    let h = world.config.grid_height;

    let mut img = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let s = world.pedo_states[idx].s;
            let color = if world.elevations[idx] <= world.config.sea_level {
                Rgb([30, 60, 150]) // ocean
            } else {
                soil_s_color(s)
            };
            img.put_pixel(x as u32, y as u32, color);
        }
    }
    img.save(path).expect("Failed to save soil PNG");
}

pub fn render_global_nutrients(world: &WorldSimulation, path: &Path) {
    let w = world.config.grid_width;
    let h = world.config.grid_height;
    let max_p = world.nutrient_columns.iter()
        .map(|c| c.surface_p_labile())
        .fold(0.0f32, f32::max)
        .max(1.0);

    let mut img = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let color = if world.elevations[idx] <= world.config.sea_level {
                Rgb([30, 60, 150])
            } else {
                nutrient_color(world.nutrient_columns[idx].surface_p_labile(), max_p)
            };
            img.put_pixel(x as u32, y as u32, color);
        }
    }
    img.save(path).expect("Failed to save nutrient PNG");
}

pub fn render_global_erosion(world: &WorldSimulation, path: &Path) {
    let w = world.config.grid_width;
    let h = world.config.grid_height;
    let max_abs = world.delta_h.iter().map(|d| d.abs()).fold(0.0f32, f32::max).max(0.001);

    let mut img = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let dh = world.delta_h[idx];
            let color = if dh < -0.0001 {
                // Erosion: red
                let t = (-dh / max_abs).clamp(0.0, 1.0);
                Rgb([(80.0 + 175.0 * t) as u8, (80.0 * (1.0 - t)) as u8, (80.0 * (1.0 - t)) as u8])
            } else if dh > 0.0001 {
                // Deposition: blue
                let t = (dh / max_abs).clamp(0.0, 1.0);
                Rgb([(80.0 * (1.0 - t)) as u8, (80.0 * (1.0 - t)) as u8, (80.0 + 175.0 * t) as u8])
            } else {
                Rgb([80, 80, 80]) // neutral
            };
            img.put_pixel(x as u32, y as u32, color);
        }
    }
    img.save(path).expect("Failed to save erosion PNG");
}

// ---------------------------------------------------------------------------
// Local SVO cross-section render
// ---------------------------------------------------------------------------

pub fn render_local_svo_slice(
    svo: &crate::svo::SparseVoxelOctree,
    slice_y: f64,
    path: &Path,
) {
    let bounds = svo.bounds();
    let res = 1usize << (svo.max_depth() as usize); // pixels = finest voxel res
    let res = res.min(512); // cap at 512px
    let step_x = (bounds.max.x - bounds.min.x) / res as f64;
    let step_z = (bounds.max.z - bounds.min.z) / res as f64;

    let mut img = RgbImage::new(res as u32, res as u32);
    for pz in 0..res {
        for px in 0..res {
            let wx = bounds.min.x + (px as f64 + 0.5) * step_x;
            let wz = bounds.min.z + (pz as f64 + 0.5) * step_z;
            let pos = crate::common::Vec3::new(wx, slice_y, wz);
            let mat = svo.get_material(&pos);
            let color = material_color(mat);
            img.put_pixel(px as u32, pz as u32, color);
        }
    }
    img.save(path).expect("Failed to save SVO slice PNG");
}

/// Render a vertical cross-section (X-Y plane at fixed Z).
pub fn render_local_svo_cross_section(
    svo: &crate::svo::SparseVoxelOctree,
    slice_z: f64,
    path: &Path,
) {
    let bounds = svo.bounds();
    let res_x = (1usize << (svo.max_depth() as usize)).min(512);
    let res_y = (1usize << (svo.max_depth() as usize)).min(512);
    let step_x = (bounds.max.x - bounds.min.x) / res_x as f64;
    let step_y = (bounds.max.y - bounds.min.y) / res_y as f64;

    let mut img = RgbImage::new(res_x as u32, res_y as u32);
    for py in 0..res_y {
        for px in 0..res_x {
            let wx = bounds.min.x + (px as f64 + 0.5) * step_x;
            // Flip Y so ground is at bottom
            let wy = bounds.max.y - (py as f64 + 0.5) * step_y;
            let pos = crate::common::Vec3::new(wx, wy, slice_z);
            let mat = svo.get_material(&pos);
            let color = material_color(mat);
            img.put_pixel(px as u32, py as u32, color);
        }
    }
    img.save(path).expect("Failed to save SVO cross-section PNG");
}

fn material_color(mat: crate::common::VoxelMaterial) -> Rgb<u8> {
    match mat {
        m if m == crate::common::VoxelMaterial::AIR => Rgb([200, 220, 240]),
        m if m == crate::common::VoxelMaterial::BEDROCK => Rgb([60, 60, 60]),
        m if m == crate::common::VoxelMaterial::ROCK => Rgb([130, 120, 110]),
        m if m == crate::common::VoxelMaterial::SOIL => Rgb([100, 70, 40]),
        m if m == crate::common::VoxelMaterial::WATER => Rgb([30, 80, 180]),
        m if m == crate::common::VoxelMaterial::SEDIMENT => Rgb([180, 160, 100]),
        _ => Rgb([255, 0, 255]), // unknown: magenta
    }
}

// ---------------------------------------------------------------------------
// Batch render: all maps for one epoch
// ---------------------------------------------------------------------------

pub fn render_epoch(
    world: &WorldSimulation,
    svo: Option<&crate::svo::SparseVoxelOctree>,
    output_dir: &Path,
    epoch: u64,
) {
    std::fs::create_dir_all(output_dir).ok();

    // Global maps
    render_global_elevation(world, &output_dir.join(format!("epoch_{:04}_elevation.png", epoch)));
    render_global_biome(world, &output_dir.join(format!("epoch_{:04}_biome.png", epoch)));
    render_global_temperature(world, &output_dir.join(format!("epoch_{:04}_temperature.png", epoch)));
    render_global_precipitation(world, &output_dir.join(format!("epoch_{:04}_precip.png", epoch)));
    render_global_soil(world, &output_dir.join(format!("epoch_{:04}_soil.png", epoch)));
    render_global_nutrients(world, &output_dir.join(format!("epoch_{:04}_nutrients.png", epoch)));
    render_global_erosion(world, &output_dir.join(format!("epoch_{:04}_erosion.png", epoch)));

    // Local SVO slices (if loaded)
    if let Some(svo) = svo {
        let center = svo.bounds().center();
        // Horizontal slice at terrain mid-height
        render_local_svo_slice(svo, center.y, &output_dir.join(format!("epoch_{:04}_svo_horiz.png", epoch)));
        // Vertical cross-section through center
        render_local_svo_cross_section(svo, center.z, &output_dir.join(format!("epoch_{:04}_svo_vert.png", epoch)));
    }
}

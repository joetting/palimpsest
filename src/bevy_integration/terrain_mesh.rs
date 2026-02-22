/// src/bevy_integration/terrain_mesh.rs
///
/// Builds a Bevy mesh from the `TerrainRenderData` resource every time
/// `dirty` is set. Uses a flat quad grid (one quad per voxel column) with
/// vertex colours driven by biome type.
///
/// Design notes:
///  - The mesh is a single `Mesh` asset, rebuilt on `dirty`. For 128×128 this
///    is ~16k quads — fast enough to rebuild each epoch without GPU stutter.
///  - For Phase 4 (real WGPU compute shaders), this placeholder will be
///    replaced by a GPU-driven heightmap with tessellation.

use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;

use super::render_resources::TerrainRenderData;

// ─── Plugin ──────────────────────────────────────────────────────────────────

pub struct TerrainMeshPlugin;

impl Plugin for TerrainMeshPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_systems(Startup, setup_terrain_mesh)
            .add_systems(Update, rebuild_terrain_mesh);
    }
}

// ─── Components ──────────────────────────────────────────────────────────────

/// Marker component placed on the terrain mesh entity.
#[derive(Component)]
pub struct TerrainMeshMarker;

/// Marker for the coarse-world debug mesh.
#[derive(Component)]
pub struct CoarseMeshMarker;

// ─── Setup ───────────────────────────────────────────────────────────────────

fn setup_terrain_mesh(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Spawn an empty mesh entity; it will be populated by rebuild_terrain_mesh.
    let mesh_handle = meshes.add(Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    ));
    let mat_handle = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        unlit: true, // we colour by vertex, skip PBR until Phase 4
        ..default()
    });

    commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(mat_handle),
        Transform::default(),
        TerrainMeshMarker,
    ));

    // Coarse world mesh (smaller, offset to the side for debug view)
    let coarse_mesh = meshes.add(Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    ));
    let coarse_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 1.0, 1.0, 0.6),
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    commands.spawn((
        Mesh3d(coarse_mesh),
        MeshMaterial3d(coarse_mat),
        // Offset to the right of the fine world for side-by-side debug
        Transform::from_xyz(200_000.0, 0.0, 0.0),
        CoarseMeshMarker,
    ));

    // Directional light
    commands.spawn((
        DirectionalLight {
            illuminance: 15_000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_4,
            0.0,
        )),
    ));

    // Camera — positioned to look down at the terrain from the south-west
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(64_000.0, 80_000.0, -30_000.0)
            .looking_at(Vec3::new(64_000.0, 0.0, 64_000.0), Vec3::Y),
    ));
}

// ─── Rebuild ─────────────────────────────────────────────────────────────────

fn rebuild_terrain_mesh(
    mut render_data: ResMut<TerrainRenderData>,
    terrain_query: Query<&Mesh3d, With<TerrainMeshMarker>>,
    coarse_query: Query<&Mesh3d, With<CoarseMeshMarker>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if !render_data.dirty {
        return;
    }
    render_data.dirty = false;

    // Fine-world mesh
    if let Ok(mesh_component) = terrain_query.get_single() {
        if let Some(mesh) = meshes.get_mut(&mesh_component.0) {
            build_heightmap_mesh(
                mesh,
                &render_data.heights,
                &render_data.biomes,
                render_data.width,
                render_data.height,
                render_data.cell_size_m,
                render_data.sea_level,
                1.0, // height scale
            );
        }
    }

    // Coarse-world debug mesh (scaled so it's roughly the same screen size)
    if let Ok(coarse_component) = coarse_query.get_single() {
        if let Some(mesh) = meshes.get_mut(&coarse_component.0) {
            let coarse_biomes = vec![0u8; render_data.coarse_heights.len()];
            let cw = render_data.coarse_width;
            let ch = render_data.coarse_height;
            // Scale cell size so coarse world appears the same footprint as fine
            let coarse_cell =
                render_data.cell_size_m * render_data.width as f32 / cw as f32;
            build_heightmap_mesh(
                mesh,
                &render_data.coarse_heights,
                &coarse_biomes,
                cw,
                ch,
                coarse_cell,
                render_data.sea_level,
                0.5, // compress height so coarse world is visually distinct
            );
        }
    }
}

/// Core mesh builder. Generates a quad grid where each column is one quad.
/// Vertex colours are biome-based. Sea cells are rendered flat at sea level.
fn build_heightmap_mesh(
    mesh: &mut Mesh,
    heights: &[f32],
    biomes: &[u8],
    width: usize,
    height: usize,
    cell_size: f32,
    sea_level: f32,
    height_scale: f32,
) {
    let n_verts = width * height;
    let n_quads = (width - 1) * (height - 1);

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n_verts);
    let mut normals:   Vec<[f32; 3]> = Vec::with_capacity(n_verts);
    let mut colors:    Vec<[f32; 4]> = Vec::with_capacity(n_verts);
    let mut uvs:       Vec<[f32; 2]> = Vec::with_capacity(n_verts);
    let mut indices:   Vec<u32>       = Vec::with_capacity(n_quads * 6);

    // Vertices
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let h = heights.get(idx).copied().unwrap_or(0.0);
            let y_pos = (h.max(sea_level) - sea_level) * height_scale;
            let biome = biomes.get(idx).copied().unwrap_or(0);

            let color = if h <= sea_level {
                // Deeper water → darker blue
                let depth_frac = ((sea_level - h) / 200.0).clamp(0.0, 1.0);
                [0.05, 0.10 + 0.20 * (1.0 - depth_frac), 0.55 + 0.25 * (1.0 - depth_frac), 1.0]
            } else {
                let c = TerrainRenderData::biome_color(biome);
                let [r, g, b, a] = c.to_srgba().to_f32_array();
                // Shade by elevation: higher = lighter
                let elevation_frac = (h / 2000.0).clamp(0.0, 1.0);
                let shade = 0.8 + 0.2 * elevation_frac;
                [r * shade, g * shade, b * shade, a]
            };

            positions.push([x as f32 * cell_size, y_pos, y as f32 * cell_size]);
            normals.push([0.0, 1.0, 0.0]); // computed below
            colors.push(color);
            uvs.push([x as f32 / width as f32, y as f32 / height as f32]);
        }
    }

    // Indices (two triangles per quad)
    for y in 0..(height - 1) {
        for x in 0..(width - 1) {
            let i = (y * width + x) as u32;
            let r = 1u32; // right
            let d = width as u32; // down
            // Triangle 1: TL, BL, TR
            indices.push(i);
            indices.push(i + d);
            indices.push(i + r);
            // Triangle 2: TR, BL, BR
            indices.push(i + r);
            indices.push(i + d);
            indices.push(i + d + r);
        }
    }

    // Recompute smooth normals from cross-product of neighbours
    let mut computed_normals = vec![[0.0f32; 3]; n_verts];
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let i = y * width + x;
            let p = Vec3::from(positions[i]);
            let px = Vec3::from(positions[i + 1]);
            let py = Vec3::from(positions[i + width]);
            let dx = px - p;
            let dy = py - p;
            let n = dx.cross(dy).normalize_or_zero();
            computed_normals[i] = n.into();
        }
    }
    // Fill border normals with up
    for y in [0, height - 1] {
        for x in 0..width {
            computed_normals[y * width + x] = [0.0, 1.0, 0.0];
        }
    }
    for x in [0, width - 1] {
        for y in 0..height {
            computed_normals[y * width + x] = [0.0, 1.0, 0.0];
        }
    }

    // Write into mesh
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, computed_normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
}

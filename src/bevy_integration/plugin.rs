/// src/bevy_integration/plugin.rs
///
/// The "DeepTimePlugin" bridges our custom simulation core with Bevy's ECS.
/// Architecture philosophy:
///   - The simulation (`DeepTimeSimulation`) lives as a Bevy `Resource`.
///   - Bevy systems read/write it and push data to GPU-side render resources.
///   - The coarse-world solver runs as a separate resource providing LBCs.
///   - NO game logic moves into Bevy components yet — agents come in Phase 5.

use bevy::prelude::*;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};

use crate::config::{DeepTimeSimulation, SimulationConfig};
use crate::climate::{ClimateConfig, WindParams};
use super::coarse_world::CoarseWorldSolver;
use super::terrain_mesh::TerrainMeshPlugin;
use super::render_resources::TerrainRenderData;
use super::ui_overlay::StratometerUiPlugin;

// ─── Resources ───────────────────────────────────────────────────────────────

/// Wraps the full simulation. Bevy treats it as a plain resource.
#[derive(Resource)]
pub struct SimResource(pub DeepTimeSimulation);

/// Wraps the coarse (global) world solver.
#[derive(Resource)]
pub struct CoarseWorldResource(pub CoarseWorldSolver);

/// Simulation timing control exposed to Bevy.
#[derive(Resource, Default)]
pub struct SimControl {
    /// When false, geological ticks are paused.
    pub running: bool,
    /// Speed multiplier: how many social ticks to advance per Bevy frame.
    pub ticks_per_frame: u32,
    /// Elder God tool selection.
    pub active_tool: ElderGodTool,
}

#[derive(Default, PartialEq, Eq, Clone, Copy, Debug)]
pub enum ElderGodTool {
    #[default]
    None,
    UpliftHotspot,
    Co2Injection,
    WindRotate,
}

// ─── Events ──────────────────────────────────────────────────────────────────

/// Fired when a geological epoch completes — used to trigger mesh rebuild.
#[derive(Event)]
pub struct GeologicalEpochCompleted {
    pub epoch: u64,
}

/// Fired when the coarse world finishes a low-res step.
#[derive(Event)]
pub struct CoarseWorldUpdated;

// ─── Plugin ──────────────────────────────────────────────────────────────────

pub struct DeepTimePlugin;

impl Plugin for DeepTimePlugin {
    fn build(&self, app: &mut App) {
        // Build and insert the fine-resolution simulation
        let sim = build_default_simulation();
        let coarse = CoarseWorldSolver::new(32, 32, 10_000.0);

        app
            // Resources
            .insert_resource(SimResource(sim))
            .insert_resource(CoarseWorldResource(coarse))
            .insert_resource(SimControl {
                running: true,
                ticks_per_frame: 10,
                ..default()
            })
            .insert_resource(TerrainRenderData::default())

            // Events
            .add_event::<GeologicalEpochCompleted>()
            .add_event::<CoarseWorldUpdated>()

            // Sub-plugins
            .add_plugins(TerrainMeshPlugin)
            .add_plugins(StratometerUiPlugin)
            .add_plugins(FrameTimeDiagnosticsPlugin)

            // Systems — ordered within a custom set
            .configure_sets(
                Update,
                (
                    SimSet::Tick,
                    SimSet::CoarseWorld,
                    SimSet::UploadRender,
                )
                    .chain(),
            )
            .add_systems(Update, (
                tick_simulation.in_set(SimSet::Tick),
                tick_coarse_world.in_set(SimSet::CoarseWorld),
                upload_terrain_to_render.in_set(SimSet::UploadRender),
            ));
    }
}

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
enum SimSet {
    Tick,
    CoarseWorld,
    UploadRender,
}

// ─── Systems ─────────────────────────────────────────────────────────────────

/// Advances the fine-resolution simulation by `ticks_per_frame` social ticks.
/// Fires `GeologicalEpochCompleted` whenever a geological epoch boundary is crossed.
fn tick_simulation(
    mut sim_res: ResMut<SimResource>,
    control: Res<SimControl>,
    time: Res<Time>,
    mut epoch_events: EventWriter<GeologicalEpochCompleted>,
) {
    if !control.running {
        return;
    }

    let sim = &mut sim_res.0;
    let dt = time.delta_secs_f64() / control.ticks_per_frame as f64;

    for _ in 0..control.ticks_per_frame {
        // step_frame handles the Strang-split accumulator internally
        sim.step_frame(dt);
    }

    // Fire the event only when the epoch counter actually advanced this frame.
    // We detect this by checking if step_frame triggered a geo tick internally.
    // Since step_frame calls advance_social_tick in a loop, we check the epoch
    // after vs before — the simplest approach is to re-expose the tick result.
    // For now we use a conservative approach: emit once per 500 social ticks
    // (which is one geo epoch by default TemporalConfig).
    // TODO Phase 4: expose TickResult from step_frame so we don't poll.
    let epoch = sim.clock.geo_epoch;
    if epoch > 0 && sim.clock.social_tick % sim.clock.config.social_ticks_per_geo_epoch == 0 {
        epoch_events.send(GeologicalEpochCompleted { epoch });
    }
}

/// Advances the coarse world every N geological epochs.
/// The coarse world provides lateral boundary conditions to the fine sim.
fn tick_coarse_world(
    mut coarse_res: ResMut<CoarseWorldResource>,
    sim_res: Res<SimResource>,
    mut epoch_events: EventReader<GeologicalEpochCompleted>,
    mut coarse_events: EventWriter<CoarseWorldUpdated>,
) {
    // Only run when an epoch just completed
    if epoch_events.is_empty() {
        return;
    }
    epoch_events.clear();

    let sim = &sim_res.0;
    let epoch = sim.clock.geo_epoch;

    // Coarse world runs every 10 fine-world epochs
    if epoch % 10 != 0 {
        return;
    }

    let coarse = &mut coarse_res.0;

    // Feed boundary conditions from the fine world edges
    coarse.ingest_fine_boundary(
        &sim.heights,
        sim.config.grid_width,
        sim.config.grid_height,
        sim.config.cell_size_m,
    );

    // Step the coarse world by 10 * fine_dt (same total time)
    let coarse_dt = sim.config.geo_dt_years * 10.0;
    coarse.step(coarse_dt as f32);

    coarse_events.send(CoarseWorldUpdated);
}

/// Copies heightmap + climate data into `TerrainRenderData` for the mesh system.
fn upload_terrain_to_render(
    sim_res: Res<SimResource>,
    coarse_res: Res<CoarseWorldResource>,
    mut render_data: ResMut<TerrainRenderData>,
    mut epoch_events: EventReader<GeologicalEpochCompleted>,
) {
    // Only re-upload when terrain actually changed
    if epoch_events.is_empty() {
        return;
    }
    epoch_events.clear();

    let sim = &sim_res.0;
    let w = sim.config.grid_width;
    let h = sim.config.grid_height;

    render_data.width = w;
    render_data.height = h;
    render_data.cell_size_m = sim.config.cell_size_m;
    render_data.heights = sim.heights.clone();
    render_data.sea_level = sim.config.sea_level_m;
    render_data.co2_ppm = sim.climate.thermostat.state.atmospheric_co2_ppm;
    render_data.temp_c = sim.climate.thermostat.state.global_mean_temp_c;
    render_data.in_ice_age = sim.climate.thermostat.state.in_ice_age;
    render_data.dirty = true;

    // Biome colours for vertex colouring
    render_data.biomes = sim
        .climate
        .columns
        .iter()
        .map(|c| c.biome as u8)
        .collect();

    // Coarse world overlay (for debug visualisation)
    render_data.coarse_heights = coarse_res.0.heights.clone();
    render_data.coarse_width = coarse_res.0.width;
    render_data.coarse_height = coarse_res.0.height;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn build_default_simulation() -> DeepTimeSimulation {
    let climate_config = ClimateConfig {
        initial_co2_ppm: 500.0,
        lambda: 3.0,
        t0_baseline: 18.0,
        plant_uptake_coeff: 0.002,
        animal_respiration_coeff: 0.001,
        volcanic_outgas_rate: 0.035,
        wind_angle_rad: std::f32::consts::PI,
        wind_speed_ms: 8.0,
        ..ClimateConfig::default()
    };
    let wind_params = WindParams {
        angle_rad: std::f32::consts::PI,
        incoming_moisture_m: 4.5,
        alpha_condensation: 0.008,
        beta_fallout_per_km: 0.15,
        background_precip_m: 0.4,
    };
    let config = SimulationConfig {
        grid_width: 128,
        grid_height: 128,
        cell_size_m: 1_000.0,
        max_elevation_m: 400.0,
        sea_level_m: 0.0,
        geo_dt_years: 100.0,
        seed: 1984,
        num_cohorts: 10,
        climate: Some(climate_config),
        wind: Some(wind_params),
    };
    DeepTimeSimulation::new(config)
}

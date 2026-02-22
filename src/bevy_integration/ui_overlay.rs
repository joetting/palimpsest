/// src/bevy_integration/ui_overlay.rs
///
/// The "Stratometer" UI — a lightweight Bevy UI overlay showing:
///   - CO₂ / temperature / ice age phase
///   - Current epoch and elapsed geological time
///   - Coarse world max elevation (LBC diagnostic)
///   - FPS counter
///   - Elder God tool selector (keyboard shortcuts)
///
/// All text is plain Bevy UI — no egui dependency yet. Phase 6 will add
/// the full Kolmogorov-entropy heatmap using a custom render pass.

use bevy::prelude::*;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use super::plugin::{SimResource, CoarseWorldResource, SimControl, ElderGodTool};

pub struct StratometerUiPlugin;

impl Plugin for StratometerUiPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_systems(Startup, setup_ui)
            .add_systems(Update, (
                update_stratometer_text,
                handle_keyboard_input,
            ));
    }
}

// ─── Marker components ───────────────────────────────────────────────────────

#[derive(Component)] struct ClimateText;
#[derive(Component)] struct EpochText;
#[derive(Component)] struct CoarseText;
#[derive(Component)] struct FpsText;
#[derive(Component)] struct ToolText;

// ─── Setup ───────────────────────────────────────────────────────────────────

fn setup_ui(mut commands: Commands) {
    // Root node — top-left panel
    commands.spawn(Node {
        position_type: PositionType::Absolute,
        top:  Val::Px(10.0),
        left: Val::Px(10.0),
        flex_direction: FlexDirection::Column,
        row_gap: Val::Px(4.0),
        padding: UiRect::all(Val::Px(8.0)),
        ..default()
    })
    .with_children(|parent| {
        let style = TextFont {
            font_size: 14.0,
            ..default()
        };
        let dim = TextColor(Color::srgba(0.9, 0.9, 0.9, 0.9));

        parent.spawn((Text::new("🌿 Interglacial | CO₂: 280 ppm | T: 14.0°C"), style.clone(), dim.clone(), ClimateText));
        parent.spawn((Text::new("Epoch #0 | 0.0 ka"), style.clone(), dim.clone(), EpochText));
        parent.spawn((Text::new("Coarse world: max 0m mean 0m | Step 0"), style.clone(), dim.clone(), CoarseText));
        parent.spawn((Text::new("FPS: --"), style.clone(), dim.clone(), FpsText));
        parent.spawn((Text::new("[SPACE] pause | [U] uplift | [C] CO₂ | [W] wind | [↑↓] speed"), style.clone(), TextColor(Color::srgba(0.7, 0.9, 0.7, 0.8)), ToolText));
    });
}

// ─── Update ──────────────────────────────────────────────────────────────────

fn update_stratometer_text(
    sim_res: Res<SimResource>,
    coarse_res: Res<CoarseWorldResource>,
    control: Res<SimControl>,
    diagnostics: Res<DiagnosticsStore>,
    mut climate_q:  Query<&mut Text, (With<ClimateText>,  Without<EpochText>, Without<CoarseText>, Without<FpsText>, Without<ToolText>)>,
    mut epoch_q:    Query<&mut Text, (With<EpochText>,    Without<ClimateText>)>,
    mut coarse_q:   Query<&mut Text, (With<CoarseText>,   Without<ClimateText>, Without<EpochText>)>,
    mut fps_q:      Query<&mut Text, (With<FpsText>,      Without<ClimateText>, Without<EpochText>, Without<CoarseText>)>,
    mut tool_q:     Query<&mut Text, (With<ToolText>,     Without<ClimateText>, Without<EpochText>, Without<CoarseText>, Without<FpsText>)>,
) {
    let sim = &sim_res.0;
    let state = &sim.climate.thermostat.state;

    // Helper: update a single-section Text component
    // In Bevy 0.15, Text::new() creates one section; mutate via sections[0].value
    macro_rules! set_text {
        ($query:expr, $val:expr) => {
            if let Ok(mut t) = $query.get_single_mut() {
                if !t.sections.is_empty() {
                    t.sections[0].value = $val;
                }
            }
        };
    }

    // Climate line
    let phase = if state.in_ice_age { "❄ ICE AGE" }
                else if state.in_hothouse { "🔥 HOTHOUSE" }
                else { "🌿 Interglacial" };
    set_text!(climate_q, format!(
        "{} | CO₂: {:.0} ppm | T: {:.1}°C",
        phase, state.atmospheric_co2_ppm, state.global_mean_temp_c
    ));

    // Epoch line
    let geo_ky = sim.clock.geo_elapsed_years / 1000.0;
    let paused = if control.running { "" } else { " [PAUSED]" };
    set_text!(epoch_q, format!(
        "Epoch #{} | {:.1} ka | ×{} speed{}",
        sim.clock.geo_epoch, geo_ky, control.ticks_per_frame, paused
    ));

    // Coarse world line
    {
        let cw = &coarse_res.0;
        set_text!(coarse_q, format!(
            "Coarse world: max {:.0}m mean {:.0}m | Step {}",
            cw.max_elevation(), cw.mean_elevation(), cw.steps
        ));
    }

    // FPS
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    set_text!(fps_q, format!("FPS: {:.0}", fps));

    // Tool hint
    let tool = match control.active_tool {
        ElderGodTool::None          => "No tool active",
        ElderGodTool::UpliftHotspot => "▲ Uplift Hotspot — click terrain to place",
        ElderGodTool::Co2Injection  => "💨 CO₂ Injection — press +/- to inject/remove",
        ElderGodTool::WindRotate    => "🌬 Wind Rotate — press ←/→ to rotate wind",
    };
    set_text!(tool_q, format!("[SPACE] pause | [U/C/W] tools | [↑↓] speed | Tool: {}", tool));
}

// ─── Keyboard input ──────────────────────────────────────────────────────────

fn handle_keyboard_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut control: ResMut<SimControl>,
    mut sim_res: ResMut<SimResource>,
) {
    // Pause / resume
    if keys.just_pressed(KeyCode::Space) {
        control.running = !control.running;
    }

    // Speed
    if keys.just_pressed(KeyCode::ArrowUp) {
        control.ticks_per_frame = (control.ticks_per_frame * 2).min(128);
    }
    if keys.just_pressed(KeyCode::ArrowDown) {
        control.ticks_per_frame = (control.ticks_per_frame / 2).max(1);
    }

    // Tool selection
    if keys.just_pressed(KeyCode::KeyU) {
        control.active_tool = if control.active_tool == ElderGodTool::UpliftHotspot {
            ElderGodTool::None
        } else {
            ElderGodTool::UpliftHotspot
        };
    }
    if keys.just_pressed(KeyCode::KeyC) {
        control.active_tool = if control.active_tool == ElderGodTool::Co2Injection {
            ElderGodTool::None
        } else {
            ElderGodTool::Co2Injection
        };
    }
    if keys.just_pressed(KeyCode::KeyW) {
        control.active_tool = if control.active_tool == ElderGodTool::WindRotate {
            ElderGodTool::None
        } else {
            ElderGodTool::WindRotate
        };
    }

    // CO₂ injection (active when tool selected)
    if control.active_tool == ElderGodTool::Co2Injection {
        if keys.just_pressed(KeyCode::Equal) {
            sim_res.0.elder_god_co2_injection(50.0);
        }
        if keys.just_pressed(KeyCode::Minus) {
            sim_res.0.elder_god_co2_injection(-50.0);
        }
    }

    // Wind rotation (active when tool selected)
    if control.active_tool == ElderGodTool::WindRotate {
        if keys.just_pressed(KeyCode::ArrowRight) {
            sim_res.0.elder_god_rotate_wind(std::f32::consts::PI * 0.125);
        }
        if keys.just_pressed(KeyCode::ArrowLeft) {
            sim_res.0.elder_god_rotate_wind(-std::f32::consts::PI * 0.125);
        }
    }
}

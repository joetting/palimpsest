/// src/bevy_main.rs
///
/// Entry point for the Bevy-integrated build (`cargo run --bin bevy_sim`).
/// The original `cargo run --bin deep_time_engine` still works unchanged.
///
/// Architecture:
///   App
///   ├── DefaultPlugins (window, input, asset, render)
///   └── DeepTimePlugin
///       ├── SimResource         (fine 128×128 simulation)
///       ├── CoarseWorldResource (coarse 32×32 LBC solver)
///       ├── TerrainMeshPlugin   (heightmap mesh builder)
///       └── StratometerUiPlugin (HUD overlay)

// Declare all modules exactly as existing main.rs does — we share the codebase.
mod ecs;
mod terrain;
mod compute;
mod math;
mod config;
mod nutrients;
mod climate;

// New Bevy integration layer
mod bevy_integration;

use bevy::prelude::*;
use bevy_integration::DeepTimePlugin;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Deep Time Engine — Bevy Phase".to_string(),
                    resolution: (1280.0, 720.0).into(),
                    ..default()
                }),
                ..default()
            }),
        )
        .add_plugins(DeepTimePlugin)
        .run();
}

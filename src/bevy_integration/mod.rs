/// src/bevy_integration/mod.rs

pub mod plugin;
pub mod coarse_world;
pub mod render_resources;
pub mod terrain_mesh;
pub mod ui_overlay;

pub use plugin::{DeepTimePlugin, SimResource, CoarseWorldResource, SimControl, ElderGodTool};
pub use coarse_world::CoarseWorldSolver;
pub use render_resources::TerrainRenderData;

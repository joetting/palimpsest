// ============================================================================
// Simulation State — Top-level container for the Deep Time Engine
//
// Connects ECS World, FastScape solver, compute pipeline, and scheduler
// into a coherent simulation loop.
// ============================================================================

use crate::ecs::world::World;
use crate::ecs::scheduler::{SimulationClock, StrangRunner, TemporalConfig};
use crate::terrain::fastscape::{FastScapeSolver, SplParams, TectonicForcing};
use crate::terrain::heightmap::Heightmap;
use crate::compute::gpu_stub::ComputePipeline;
use crate::ecs::components::MaterialId;

pub struct SimulationConfig {
    pub grid_width:     usize,
    pub grid_height:    usize,
    pub cell_size_m:    f32,
    pub max_elevation_m: f32,
    pub sea_level_m:    f32,
    pub geo_dt_years:   f64,
    pub seed:           u64,
    pub num_cohorts:    u8,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            grid_width:      128,
            grid_height:     128,
            cell_size_m:     1_000.0,   // 1 km per cell
            max_elevation_m: 3_000.0,   // max 3 km
            sea_level_m:     0.0,
            geo_dt_years:    100.0,     // 100-year epochs
            seed:            42,
            num_cohorts:     10,
        }
    }
}

pub struct DeepTimeSimulation {
    pub world:    World,
    pub solver:   FastScapeSolver,
    pub compute:  ComputePipeline,
    pub clock:    SimulationClock,
    pub tectonics: TectonicForcing,
    pub config:   SimulationConfig,
    /// Raw height array (mirrors world elevations, owned by solver pipeline)
    pub heights:  Vec<f32>,
}

impl DeepTimeSimulation {
    pub fn new(config: SimulationConfig) -> Self {
        let w = config.grid_width;
        let h = config.grid_height;

        // Generate initial terrain via fBm
        let heightmap = Heightmap::fbm(
            w, h,
            config.max_elevation_m,
            7,    // octaves
            2.0,  // lacunarity
            0.5,  // persistence
            config.seed,
        );

        // Apply island mask so boundaries drain naturally
        let mut hm = heightmap;
        hm.apply_island_mask(0.15);

        let stats = hm.stats();
        println!("Initial heightmap: {}", stats);

        // Initialize ECS world
        let mut world = World::new(w as u32, h as u32);
        world.initialize_grid(
            |x, y| hm.get(x as usize, y as usize),
            |elev| {
                if elev <= config.sea_level_m + 10.0 { MaterialId::Sediment }
                else if elev < 200.0 { MaterialId::SoilA }
                else if elev < 800.0 { MaterialId::SoftRock }
                else { MaterialId::Bedrock }
            },
            config.num_cohorts,
        );

        // FastScape solver
        let spl_params = SplParams {
            m:            0.5,
            n:            1.0,
            cell_size:    config.cell_size_m,
            rainfall:     1.0,
            sea_level:    config.sea_level_m,
            ..Default::default()
        };
        let solver = FastScapeSolver::new(w, h, spl_params);

        // Compute pipeline
        let mut compute = ComputePipeline::new(w, h);
        compute.buffers.elevation = hm.data.clone();
        compute.update_activity_mask(config.sea_level_m);

        // Scheduler clock
        let temporal = TemporalConfig {
            geo_dt_years:               config.geo_dt_years,
            social_ticks_per_geo_epoch: 500,
            ..Default::default()
        };
        let clock = SimulationClock::new(temporal);

        // Tectonic forcing — gentle default
        let tectonics = TectonicForcing::new(0.0001); // 0.1 mm/yr background

        let heights = hm.data;

        Self { world, solver, compute, clock, tectonics, config, heights }
    }

    /// Run one geological epoch (called by the Strang geological schedule)
    pub fn step_geological_epoch(&mut self) {
        let dt = self.clock.geo_dt();
        let w = self.config.grid_width;
        let h = self.config.grid_height;

        // Build uplift array from tectonic forcing
        let uplift = self.tectonics.uplift_array(w, h, self.config.cell_size_m);

        // Erodibility and diffusivity arrays from world components
        let kf: Vec<f32> = self.world.columns.iter().map(|c| c.erodibility.0).collect();
        let kd: Vec<f32> = self.world.columns.iter().map(|c| c.diffusivity.0).collect();

        // Run FastScape epoch — returns Δh array
        let deltas = self.solver.step_epoch(
            &mut self.heights,
            &uplift,
            &kf,
            &kd,
            dt,
        );

        // Apply deltas back to the ECS world
        self.world.apply_elevation_deltas(&deltas);

        // Sync compute pipeline buffers
        self.compute.buffers.elevation = self.heights.clone();

        // Update activity mask (ocean columns become inactive)
        self.compute.update_activity_mask(self.config.sea_level_m);
    }

    /// Advance the simulation by one "frame" (social micro-tick)
    pub fn step_frame(&mut self, dt_s: f64) {
        let result = self.clock.advance_social_tick(dt_s);
        if result.fire_geo {
            self.step_geological_epoch();
        }
    }

    /// "Elder God" intervention: add a tectonic hotspot at world coordinates
    pub fn add_uplift_hotspot(
        &mut self,
        center_x: u32,
        center_y: u32,
        radius_cells: f32,
        rate_m_yr: f32,
    ) {
        let cx = center_x as f32 * self.config.cell_size_m;
        let cy = center_y as f32 * self.config.cell_size_m;
        let radius_m = radius_cells * self.config.cell_size_m;
        self.tectonics.add_hotspot(cx, cy, radius_m, rate_m_yr);
        println!(
            "Uplift hotspot added at ({},{}) r={}km @{:.2}mm/yr",
            center_x, center_y,
            radius_cells as u32,
            rate_m_yr * 1000.0
        );
    }

    /// Simulation diagnostics
    pub fn print_stats(&self) {
        let max_h = self.solver.max_elevation(&self.heights);
        let mean_h = self.solver.mean_elevation(&self.heights);
        let active = self.compute.active_count();
        let total = self.config.grid_width * self.config.grid_height;

        println!(
            "  {} | MaxH={:.0}m MeanH={:.0}m | Active={}/{} ({:.1}% skip)",
            self.clock.summary(),
            max_h, mean_h,
            active, total,
            (1.0 - active as f32 / total as f32) * 100.0
        );
    }
}

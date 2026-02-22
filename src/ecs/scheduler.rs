// ============================================================================
// Multirate Scheduler — Strang Operator Splitting
//
// Implements the schedule hierarchy required for Deep Time:
//
//   Social Schedule  (high-freq, FixedUpdate style)
//   Geological Schedule (low-freq epoch-tick, accumulator-triggered)
//
// Strang splitting sequence:
//   1. Social Half-Step  (Δt/2)
//   2. Geological Full-Step  (Δt_geo)
//   3. Social Half-Step  (Δt/2)
//
// This gives O(Δt²) accuracy and is time-reversible.
// ============================================================================

use std::time::Duration;

/// Labels for the different schedules — analogous to Bevy's ScheduleLabel
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScheduleLabel {
    /// High-frequency social micro-tick (seconds–minutes)
    SocialMicroTick,
    /// Geological epoch-tick (100–10,000 years)
    GeologicalEpoch,
    /// Biological update (years–decades)
    BiologicalCycle,
    /// Full Strang step (orchestrator)
    StrangStep,
}

/// Configuration for temporal scaling
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    /// Social time per real second [sim-seconds / real-second]
    pub social_time_scale: f64,
    /// Geological timestep [years per epoch-tick]
    pub geo_dt_years: f64,
    /// Biological timestep [years per bio-tick]
    pub bio_dt_years: f64,
    /// How many social ticks accumulate before triggering a geo epoch
    pub social_ticks_per_geo_epoch: u64,
    /// Number of cohorts for staggered social updates
    pub social_cohorts: u8,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            social_time_scale:          1.0,
            geo_dt_years:               100.0,  // 100-year epochs
            bio_dt_years:               1.0,    // Annual bio updates
            social_ticks_per_geo_epoch: 1000,   // 1000 social ticks = 1 geo epoch
            social_cohorts:             10,     // 10% of agents update per frame
        }
    }
}

/// Tracks accumulated simulation time across all scales
#[derive(Debug, Clone)]
pub struct SimulationClock {
    /// Total elapsed social-scale time [seconds]
    pub social_elapsed_s: f64,
    /// Total elapsed geological time [years]
    pub geo_elapsed_years: f64,
    /// Total elapsed biological time [years]
    pub bio_elapsed_years: f64,
    /// Social tick counter
    pub social_tick: u64,
    /// Geological epoch counter
    pub geo_epoch: u64,
    /// Biological cycle counter
    pub bio_cycle: u64,
    /// Accumulator for triggering geo epochs
    social_tick_accumulator: u64,
    /// Accumulator for biological cycles
    social_tick_bio_accumulator: u64,
    pub config: TemporalConfig,
}

impl SimulationClock {
    pub fn new(config: TemporalConfig) -> Self {
        Self {
            social_elapsed_s:             0.0,
            geo_elapsed_years:            0.0,
            bio_elapsed_years:            0.0,
            social_tick:                  0,
            geo_epoch:                    0,
            bio_cycle:                    0,
            social_tick_accumulator:      0,
            social_tick_bio_accumulator:  0,
            config,
        }
    }

    /// Advance one social micro-tick of duration `dt_s` seconds.
    /// Returns which other schedules should fire this tick.
    pub fn advance_social_tick(&mut self, dt_s: f64) -> TickResult {
        self.social_elapsed_s += dt_s * self.config.social_time_scale;
        self.social_tick += 1;
        self.social_tick_accumulator += 1;
        self.social_tick_bio_accumulator += 1;

        let fire_geo = self.social_tick_accumulator >= self.config.social_ticks_per_geo_epoch;
        if fire_geo {
            self.social_tick_accumulator = 0;
            self.geo_elapsed_years += self.config.geo_dt_years;
            self.geo_epoch += 1;
        }

        // Bio: one cycle per ~365 social ticks (roughly yearly)
        let ticks_per_bio = (self.config.social_ticks_per_geo_epoch as f64
            / (self.config.geo_dt_years / self.config.bio_dt_years)) as u64;
        let fire_bio = self.social_tick_bio_accumulator >= ticks_per_bio.max(1);
        if fire_bio {
            self.social_tick_bio_accumulator = 0;
            self.bio_elapsed_years += self.config.bio_dt_years;
            self.bio_cycle += 1;
        }

        TickResult { fire_geo, fire_bio }
    }

    /// Current geological delta-t [years]
    pub fn geo_dt(&self) -> f32 {
        self.config.geo_dt_years as f32
    }

    /// Current biological delta-t [years]
    pub fn bio_dt(&self) -> f32 {
        self.config.bio_dt_years as f32
    }

    /// Which social cohort should update this tick (round-robin)
    pub fn active_cohort(&self) -> u8 {
        (self.social_tick % self.config.social_cohorts as u64) as u8
    }

    /// Human-readable time summary
    pub fn summary(&self) -> String {
        let geo_ky = self.geo_elapsed_years / 1000.0;
        let geo_my = self.geo_elapsed_years / 1_000_000.0;
        if geo_my >= 0.1 {
            format!(
                "Epoch {} | {:.2} Ma | Social tick {} | Bio cycle {}",
                self.geo_epoch, geo_my, self.social_tick, self.bio_cycle
            )
        } else {
            format!(
                "Epoch {} | {:.1} ka | Social tick {} | Bio cycle {}",
                self.geo_epoch, geo_ky, self.social_tick, self.bio_cycle
            )
        }
    }
}

/// Result of advancing one social tick
#[derive(Debug, Clone, Copy)]
pub struct TickResult {
    pub fire_geo: bool,
    pub fire_bio: bool,
}

/// A schedule is a named collection of system callbacks.
/// In Phase 1 this is a simple trait-object list; Phase 4+ would use
/// Bevy's proper scheduling graph.
pub type SystemFn<State> = Box<dyn Fn(&mut State) + Send + Sync>;

pub struct Schedule<State> {
    pub label: ScheduleLabel,
    systems: Vec<SystemFn<State>>,
}

impl<State> Schedule<State> {
    pub fn new(label: ScheduleLabel) -> Self {
        Self { label, systems: Vec::new() }
    }

    pub fn add_system<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&mut State) + Send + Sync + 'static,
    {
        self.systems.push(Box::new(f));
        self
    }

    pub fn run(&self, state: &mut State) {
        for system in &self.systems {
            system(state);
        }
    }
}

/// Strang splitting runner — coordinates Social|Geo|Bio schedules
pub struct StrangRunner<State> {
    pub social_schedule:  Schedule<State>,
    pub geo_schedule:     Schedule<State>,
    pub bio_schedule:     Schedule<State>,
    pub clock:            SimulationClock,
}

impl<State> StrangRunner<State> {
    pub fn new(config: TemporalConfig) -> Self {
        Self {
            social_schedule: Schedule::new(ScheduleLabel::SocialMicroTick),
            geo_schedule:    Schedule::new(ScheduleLabel::GeologicalEpoch),
            bio_schedule:    Schedule::new(ScheduleLabel::BiologicalCycle),
            clock:           SimulationClock::new(config),
        }
    }

    /// Run one "wall-clock frame" worth of simulation.
    ///
    /// Strang sequence:
    ///   - Social half-step  
    ///   - [If epoch fires] Geological full-step
    ///   - Social half-step
    pub fn step(&mut self, state: &mut State, dt_s: f64) {
        // Social Half-Step A
        self.social_schedule.run(state);

        let result = self.clock.advance_social_tick(dt_s);

        // Geological Full-Step (only when epoch accumulator fires)
        if result.fire_geo {
            self.geo_schedule.run(state);
        }

        // Biological Cycle
        if result.fire_bio {
            self.bio_schedule.run(state);
        }

        // Social Half-Step B  
        // (For full Strang accuracy, social systems would split their
        //  delta-time as Δt/2 | Δt/2. In this implementation the social
        //  schedule runs as a single tick per frame, approximating the split.)
        self.social_schedule.run(state);
    }

    pub fn clock(&self) -> &SimulationClock {
        &self.clock
    }
}

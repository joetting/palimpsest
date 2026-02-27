//! Biological agents: BDI architecture with influence map accumulation.
//!
//! **Solution #4 (Influence Maps)**: Agents never directly mutate TerrainGrid arrays.
//! Instead, they write intentions to the `bio_progressive_write` and `bio_regressive_write`
//! influence maps. The main physics loop consumes the read buffers in a single SIMD-friendly
//! vectorized pass during pedogenesis integration.
//!
//! **Solution #5 (Multi-Scale Sub-Stepping)**: The agent sub-cycle runs at dt_bio (e.g. 1 yr)
//! inside each macro geomorphological timestep (e.g. 10,000 yr). Agent impacts accumulate
//! into the influence map over many sub-steps. After the biological sub-cycle completes,
//! the accumulated influence is averaged and consumed by pedogenesis as a bulk condition.
//!
//! Agents use bounded rationality: zero global knowledge, decisions based only on local
//! observation within a limited perception radius.

use crate::grid::TerrainGrid;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// A single BDI agent entity.
#[derive(Clone, Debug)]
pub struct BdiAgent {
    /// Grid position (row, col)
    pub row: usize,
    pub col: usize,
    /// Agent type determines interaction mode
    pub kind: AgentKind,
    /// Internal energy / satiation level [0, 1]
    pub energy: f64,
    /// Persistence: higher = more stubborn, resists behavioral shifts
    pub persistence: f64,
    /// Age in biological sub-steps
    pub age: u64,
}

/// Agent archetypes that produce different pedological effects.
#[derive(Clone, Debug, Copy, PartialEq)]
pub enum AgentKind {
    /// Herbivore: grazes vegetation, can cause overgrazing → regressive spike
    Herbivore,
    /// Decomposer: breaks down litter → progressive enrichment
    Decomposer,
    /// Farmer: intensive agriculture, moderate progressive + high regressive if overworked
    Farmer,
    /// Forester: plants trees, strong progressive, low regressive
    Forester,
}

/// Parameters controlling the agent sub-cycle.
#[derive(Clone, Debug)]
pub struct AgentParams {
    /// Number of initial agents to spawn
    pub initial_count: usize,
    /// Biological sub-step duration [yr]
    pub dt_bio: f64,
    /// Number of representative sub-steps per macro timestep (accelerated forcing)
    /// Rather than running the full dt_geomorph / dt_bio steps, we run this many
    /// and extrapolate. This implements the accelerated forcing approach.
    pub representative_steps: usize,
    /// Perception radius in grid cells
    pub perception_radius: usize,
    /// Movement probability per sub-step
    pub move_probability: f64,
    /// Grazing intensity (regressive contribution per herbivore per step)
    pub grazing_intensity: f64,
    /// Decomposition intensity (progressive contribution per decomposer per step)
    pub decomposition_intensity: f64,
    /// Farming progressive intensity
    pub farming_progressive: f64,
    /// Farming regressive intensity (soil compaction, nutrient depletion)
    pub farming_regressive: f64,
    /// Forester progressive intensity
    pub forester_progressive: f64,
    /// Energy threshold below which agent seeks food
    pub hunger_threshold: f64,
    /// Energy recovery per step when on fertile soil
    pub energy_recovery: f64,
    /// Energy cost per step
    pub energy_cost: f64,
}

impl Default for AgentParams {
    fn default() -> Self {
        Self {
            initial_count: 500,
            dt_bio: 1.0,
            representative_steps: 200,
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
        }
    }
}

/// The agent population managed as a flat Vec (cache-friendly iteration).
pub struct AgentPopulation {
    pub agents: Vec<BdiAgent>,
    rng: StdRng,
}

impl AgentPopulation {
    /// Spawn agents randomly across the grid interior.
    pub fn new(params: &AgentParams, rows: usize, cols: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut agents = Vec::with_capacity(params.initial_count);

        let kinds = [AgentKind::Herbivore, AgentKind::Decomposer,
                     AgentKind::Farmer, AgentKind::Forester];

        for i in 0..params.initial_count {
            let r = rng.gen_range(1..rows - 1);
            let c = rng.gen_range(1..cols - 1);
            let kind = kinds[i % kinds.len()];

            agents.push(BdiAgent {
                row: r,
                col: c,
                kind,
                energy: 0.5 + rng.gen::<f64>() * 0.5,
                persistence: 0.3 + rng.gen::<f64>() * 0.7,
                age: 0,
            });
        }

        Self { agents, rng }
    }

    /// Execute one biological sub-step for all agents.
    ///
    /// Agents observe local conditions (bounded rationality), update desires,
    /// form intentions, act, and write their effects to the influence map WRITE buffer.
    fn step(
        &mut self,
        grid: &mut TerrainGrid,
        params: &AgentParams,
    ) {
        let rows = grid.rows;
        let cols = grid.cols;

        for agent in self.agents.iter_mut() {
            agent.age += 1;

            // === BELIEFS: Observe local environment ===
            let local_soil = grid.soil_state[[agent.row, agent.col]];
            let local_precip = grid.precipitation[[agent.row, agent.col]];
            let local_slope = grid.slope[[agent.row, agent.col]];
            let local_carbon = grid.soil_carbon[[agent.row, agent.col]];

            // Fertility perception: rich soil + water + organic matter
            let fertility = (local_soil * 0.3 + local_precip * 0.3 + local_carbon * 0.01)
                .min(1.0);

            // === DESIRES: Update internal state ===
            agent.energy -= params.energy_cost;
            if fertility > 0.3 && agent.energy < 1.0 {
                agent.energy = (agent.energy + params.energy_recovery * fertility).min(1.0);
            }

            // === INTENTIONS: Decide action based on beliefs + desires ===

            // Movement: if hungry and local conditions poor, seek better ground
            let should_move = agent.energy < params.hunger_threshold
                || self.rng.gen::<f64>() < params.move_probability;

            if should_move {
                // Look within perception radius for best cell
                let mut best_r = agent.row;
                let mut best_c = agent.col;
                let mut best_score = f64::NEG_INFINITY;
                let rad = params.perception_radius as i32;

                for dr in -rad..=rad {
                    for dc in -rad..=rad {
                        let nr = agent.row as i32 + dr;
                        let nc = agent.col as i32 + dc;
                        if nr < 1 || nr >= (rows - 1) as i32
                            || nc < 1 || nc >= (cols - 1) as i32 { continue; }
                        let nr = nr as usize;
                        let nc = nc as usize;

                        let score = match agent.kind {
                            AgentKind::Herbivore => {
                                // Seek fertile, flat ground
                                grid.soil_state[[nr, nc]] * 2.0
                                    + grid.soil_carbon[[nr, nc]] * 0.5
                                    - grid.slope[[nr, nc]] * 5.0
                            }
                            AgentKind::Decomposer => {
                                // Seek high-carbon, moist areas
                                grid.soil_carbon[[nr, nc]]
                                    + grid.precipitation[[nr, nc]] * 0.5
                            }
                            AgentKind::Farmer => {
                                // Seek flat, fertile, well-watered land
                                grid.soil_state[[nr, nc]] * 3.0
                                    + grid.precipitation[[nr, nc]]
                                    - grid.slope[[nr, nc]] * 10.0
                            }
                            AgentKind::Forester => {
                                // Seek moderate slope, some existing soil
                                grid.soil_state[[nr, nc]]
                                    + grid.precipitation[[nr, nc]]
                                    - (grid.slope[[nr, nc]] - 0.1).abs() * 3.0
                            }
                        };

                        // Add noise for bounded rationality
                        let noisy_score = score + (self.rng.gen::<f64>() - 0.5) * 0.5;
                        if noisy_score > best_score {
                            best_score = noisy_score;
                            best_r = nr;
                            best_c = nc;
                        }
                    }
                }

                agent.row = best_r;
                agent.col = best_c;
            }

            // === ACTION: Write effects to influence map (Solution #4) ===
            // Agents NEVER touch elevation, bedrock, etc. directly.
            // They only write to the influence map WRITE buffers.

            let r = agent.row;
            let c = agent.col;

            match agent.kind {
                AgentKind::Herbivore => {
                    // Grazing: moderate progressive (manure) + regressive if overgrazed
                    let overgrazing = if local_soil < 0.2 { 2.0 } else { 0.5 };
                    grid.bio_progressive_write[[r, c]] += params.grazing_intensity * 0.3;
                    grid.bio_regressive_write[[r, c]] += params.grazing_intensity * overgrazing;
                }
                AgentKind::Decomposer => {
                    // Pure progressive enrichment (nutrient cycling)
                    grid.bio_progressive_write[[r, c]] += params.decomposition_intensity;
                }
                AgentKind::Farmer => {
                    // Dual: enrichment from cultivation + degradation from compaction
                    grid.bio_progressive_write[[r, c]] += params.farming_progressive;
                    grid.bio_regressive_write[[r, c]] += params.farming_regressive;

                    // Overworked soil: if farming on degraded land, regressive spikes
                    if local_soil < 0.3 {
                        grid.bio_regressive_write[[r, c]] += params.farming_regressive * 2.0;
                    }
                }
                AgentKind::Forester => {
                    // Strong progressive from root systems and litter
                    grid.bio_progressive_write[[r, c]] += params.forester_progressive;
                }
            }

            // Kill agents with no energy (starvation)
            if agent.energy <= 0.0 {
                agent.energy = 0.0;
            }
        }

        // Remove dead agents
        self.agents.retain(|a| a.energy > 0.0);

        // Occasional reproduction if population is healthy
        let new_agents: Vec<BdiAgent> = self.agents.iter()
            .filter(|a| a.energy > 0.8 && a.age > 10)
            .filter(|_| self.rng.gen::<f64>() < 0.01)
            .map(|parent| {
                let dr: i32 = self.rng.gen_range(-2..=2);
                let dc: i32 = self.rng.gen_range(-2..=2);
                let nr = (parent.row as i32 + dr).clamp(1, (rows - 2) as i32) as usize;
                let nc = (parent.col as i32 + dc).clamp(1, (cols - 2) as i32) as usize;
                BdiAgent {
                    row: nr,
                    col: nc,
                    kind: parent.kind,
                    energy: 0.5,
                    persistence: parent.persistence,
                    age: 0,
                }
            })
            .collect();

        self.agents.extend(new_agents);
    }
}

/// Run the biological sub-cycle for one macro timestep.
///
/// **Solution #5 (Multi-Scale Sub-Stepping)**:
/// Instead of running dt_geomorph / dt_bio steps (e.g. 10,000 steps for a 10kyr
/// macro step at 1yr bio resolution), we run `representative_steps` (e.g. 200)
/// and normalize the accumulated influence. This implements accelerated forcing:
/// the biological system reaches quasi-steady state quickly, and its statistical
/// average is extrapolated over the geological timescale.
///
/// The normalization factor converts accumulated influence to a per-year rate
/// that the pedogenesis integrator can consume over the full dt_geomorph.
pub fn run_biological_subcycle(
    grid: &mut TerrainGrid,
    population: &mut AgentPopulation,
    params: &AgentParams,
    dt_geomorph: f64,
) {
    // Clear write buffers before accumulation
    grid.bio_progressive_write.fill(0.0);
    grid.bio_regressive_write.fill(0.0);

    // Run representative sub-steps
    let n_steps = params.representative_steps;
    for _ in 0..n_steps {
        population.step(grid, params);
    }

    // Normalize: convert accumulated totals to per-year rates.
    //
    // The influence maps now contain the sum of all agent contributions over
    // `n_steps` biological sub-steps. To convert to a rate [1/yr]:
    //   rate = total_accumulated / (n_steps * dt_bio)
    //
    // This rate will be consumed by pedogenesis over dt_geomorph.
    let normalization = 1.0 / (n_steps as f64 * params.dt_bio);

    for val in grid.bio_progressive_write.iter_mut() {
        *val *= normalization;
    }
    for val in grid.bio_regressive_write.iter_mut() {
        *val *= normalization;
    }

    // Swap buffers: move accumulated writes → read buffer for physics consumption
    grid.swap_influence_buffers();
}

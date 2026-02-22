// ============================================================================
// Biogeochemistry Solver — Phase 2: Critical Zone
//
// Implements the 9-pool P-K ODE system from Buendía et al. (2010) extended
// with 3 K pools (Walker-Syers + K biocycling pump).
//
// Numerical strategy: Strang operator splitting
//   1. Half-step FAST reactions  — analytical equilibrium (unconditionally stable)
//   2. Full-step TRANSPORT       — erosion Δh → nutrient piggybacking
//   3. Half-step FAST reactions  — repeat
//   4. Full-step SLOW reactions  — forward Euler (k × Δt ≪ 1, stable)
//
// At Δt = 100 yr:
//   Fast (K exchange, P sorption): analytical snap-to-eq → no CFL constraint
//   Medium (P occlusion, k_occ×Δt = 0.01): forward Euler, stable
//   Slow (mineral weathering, k_w×Δt = 0.5): forward Euler, stable
// ============================================================================

use rayon::prelude::*;
use crate::nutrients::pools::{
    NutrientColumn, NutrientLayer,
    K_W_P, PHI_P, K_OCC, K_MIN_P, K_LIT_P, K_UPTAKE_P, K_LEACH_P, P_TECTONIC_INPUT,
    K_W_K, K_FIX_BASE, K_REL, K_UPTAKE_K, K_LIT_K, K_LEACH_K,
    defac,
};

// ---------------------------------------------------------------------------
// Per-column environmental inputs (set from the terrain/climate state)
// ---------------------------------------------------------------------------

/// Environmental inputs for one column's biogeochemistry update.
#[derive(Debug, Clone, Copy)]
pub struct ColumnEnv {
    /// Mean annual temperature (°C) — drives DEFAC
    pub temp_c: f32,
    /// Soil moisture as fraction of field capacity (0–1)
    pub moisture: f32,
    /// Annual runoff depth (m/yr) — drives leaching
    pub runoff_m_yr: f32,
    /// Whether this column is flooded this epoch (alluvial P renewal)
    pub flooded: bool,
    /// P delivered by flood deposit (mg/kg) — zero if not flooded
    pub flood_p_input: f32,
    /// Sediment eroded (+) or deposited (−) from FastScape (m)
    pub delta_h_m: f32,
    /// Fraction of vegetation cover remaining (0=bare, 1=forest)
    /// Drives progressive pedogenesis and canopy P-trapping
    pub veg_cover: f32,
}

impl Default for ColumnEnv {
    fn default() -> Self {
        Self {
            temp_c: 15.0,
            moisture: 0.6,
            runoff_m_yr: 0.4,
            flooded: false,
            flood_p_input: 0.0,
            delta_h_m: 0.0,
            veg_cover: 0.8,
        }
    }
}

// ---------------------------------------------------------------------------
// Analytical equilibrium substep (fast reactions, half-step)
//
// Applies the analytical solution to: dA/dt = -k_f A + k_b B
//   A_eq = k_b / (k_f + k_b) × (A + B)
//   α    = 1 − exp(−(k_f + k_b) × dt)
//   A_new = A + α × (A_eq − A)
// This is unconditionally stable for any Δt.
// ---------------------------------------------------------------------------

#[inline(always)]
fn analytical_equilibrium(a: f32, b: f32, k_forward: f32, k_backward: f32, dt: f32) -> (f32, f32) {
    let total = a + b;
    let denom = k_forward + k_backward;
    if denom < 1e-12 {
        return (a, b);
    }
    let a_eq = k_backward / denom * total;
    let alpha = 1.0 - (-(denom * dt)).exp();
    let a_new = a + alpha * (a_eq - a);
    let b_new = total - a_new;
    (a_new, b_new)
}

// ---------------------------------------------------------------------------
// Fast-reaction half-step (Strang steps 1 & 3)
// Handles K exchange↔fixed and P labile↔sorption equilibrium
// ---------------------------------------------------------------------------

fn step_fast_reactions(layer: &mut NutrientLayer, clay_frac: f32, dt_half: f32) {
    // K exchangeable ↔ K fixed  (clay interlayer fixation)
    // k_forward = K_FIX_BASE × clay_frac  (drying causes fixation)
    // k_backward = K_REL  (wetting slowly releases)
    let k_fix = K_FIX_BASE * clay_frac;
    let (k_exch_new, k_fixed_new) = analytical_equilibrium(
        layer.k_exch, layer.k_fixed,
        k_fix, K_REL,
        dt_half,
    );
    layer.k_exch  = k_exch_new;
    layer.k_fixed = k_fixed_new;

    // P labile ↔ P sorbed (fast surface adsorption onto Fe/Al oxides)
    // Treated as equilibrium with P_occluded acting as the slow sink;
    // the very fast adsorption pre-equilibration is captured here.
    // k_forward = k_occ * clay_frac (sorption to clay surfaces)
    // k_backward = K_OCC (slow release — much smaller than adsorption rate)
    // Note: the irreversible occlusion component is in the slow step.
    let k_ads = K_OCC * 10.0 * clay_frac; // fast reversible sorption
    let k_des = K_OCC * 2.0;               // partial reversibility
    let (p_labile_new, _p_sorbed) = analytical_equilibrium(
        layer.p_labile, 0.0, // no distinct fast-sorbed pool; pre-equilibrate labile only
        k_ads * 0.1, k_des * 0.1, // small correction since occluded is the main sink
        dt_half,
    );
    // Clamp: only a small fraction moves in half-step
    layer.p_labile = p_labile_new.max(0.0);
}

// ---------------------------------------------------------------------------
// Slow-reaction full-step (Strang step 4) — forward Euler
// Handles mineral weathering, occlusion, litterfall, uptake, leaching
// ---------------------------------------------------------------------------

fn step_slow_reactions(
    layer: &mut NutrientLayer,
    clay_frac: f32,
    root_dens: f32,
    env: &ColumnEnv,
    dt: f64,
) {
    let dt = dt as f32;
    let d = defac(env.temp_c, env.moisture);
    let runoff = env.runoff_m_yr;
    let roots = root_dens;

    // ---- PHOSPHORUS ----

    // Mineral P: tectonic input and apatite dissolution
    let dp_mineral = P_TECTONIC_INPUT - K_W_P * layer.p_mineral * d;
    let weathered_p = K_W_P * layer.p_mineral * d;

    // Labile P: receives weathered fraction, mineralization, loses to
    // occlusion, leaching, plant uptake, and gains from flood/dust
    let occlusion_flux     = K_OCC * layer.p_labile * clay_frac;
    let mineralization_flux = K_MIN_P * layer.p_organic * d;
    let p_leach_flux       = K_LEACH_P * layer.p_labile * runoff;
    let p_uptake_flux      = K_UPTAKE_P * layer.p_labile * roots * d;
    let flood_p            = if env.flooded { env.flood_p_input } else { 0.0 };
    // Canopy trapping of atmospheric P from volcanic ash/dust (limited flux)
    // Scaled so it can at most compensate leaching losses, not grow unboundedly
    let canopy_p_max_flux  = K_LEACH_P * layer.p_labile * env.runoff_m_yr; // bounded by leach rate
    let canopy_p_trap      = (env.veg_cover * 0.05).min(canopy_p_max_flux * 0.5); // mg/kg/yr

    // Organic P mineralization return to labile
    // Mineralization: capped to what the organic pool actually contains in this sub-step
    let k_min_return = (K_MIN_P * layer.p_organic * d).min(layer.p_organic / dt.max(1e-6));

    // (dp_labile, dp_organic, dp_veg are computed analytically below)
    let _ = p_uptake_flux; // superseded by analytical veg step

    // ---- POTASSIUM ----

    // Mineral K: feldspar/mica dissolution
    let dk_mineral = -K_W_K * layer.k_mineral * d;
    let weathered_k = K_W_K * layer.k_mineral * d;

    // Exchangeable K: gains from weathering + K release from fixed pool,
    // loses to fixation (handled analytically above, but residual slow drift),
    // leaching, and plant uptake. Gains from litterfall.
    let k_leach_flux   = K_LEACH_K * layer.k_exch * runoff;
    let k_uptake_flux  = K_UPTAKE_K * layer.k_exch * roots * d;
    let litterfall_k   = K_LIT_K * layer.k_veg;
    let k_fix_slow     = K_FIX_BASE * 0.1 * layer.k_exch * clay_frac; // residual slow fixation
    let k_rel_slow     = K_REL * 0.1 * layer.k_fixed;                  // residual slow release

    let dk_fixed = k_fix_slow - k_rel_slow;

    // ---- APPLY EULER STEP (flux-clamped to prevent pool exhaustion at large Δt) ----
    // Vegetation pools use analytical exponential decay for the veg→soil return flux
    // to avoid overshoot when dt × k_lit >> 1.

    // P veg: analytical decay then uptake refill
    // P_veg_new = P_veg × exp(-k_lit_P × dt) + uptake_fraction
    let p_veg_decay_factor = (-K_LIT_P * dt).exp();
    let p_lit_return = layer.p_veg * (1.0 - p_veg_decay_factor); // actual return to organic
    let p_up_actual = (K_UPTAKE_P * layer.p_labile * roots * d * dt)
                       .min(layer.p_labile * 0.9); // can't take >90% of labile in one step
    layer.p_veg = (layer.p_veg * p_veg_decay_factor + p_up_actual).max(0.0);

    // K veg: analytical decay then uptake refill
    let k_veg_decay_factor = (-K_LIT_K * dt).exp();
    let k_lit_return = layer.k_veg * (1.0 - k_veg_decay_factor); // actual return to exch
    let k_up_actual = (K_UPTAKE_K * layer.k_exch * roots * d * dt)
                       .min(layer.k_exch * 0.9);
    layer.k_veg = (layer.k_veg * k_veg_decay_factor + k_up_actual).max(0.0);

    // P mineral: geological depletion only
    layer.p_mineral = (layer.p_mineral + dp_mineral * dt).max(0.0);

    // P labile: use actual litterfall returns computed above
    let p_out = (occlusion_flux + p_leach_flux) * dt + p_up_actual;
    let p_in  = (weathered_p * PHI_P + k_min_return + flood_p / dt.max(1e-6) + canopy_p_trap) * dt;
    layer.p_labile = (layer.p_labile - p_out.min(layer.p_labile) + p_in).max(0.0);

    // P occluded: only grows
    layer.p_occluded += occlusion_flux * dt;

    // P organic: gains from litter return, loses to mineralization (clamped)
    let p_min_actual = (k_min_return * dt).min(layer.p_organic);
    let p_organic_new = layer.p_organic + p_lit_return - p_min_actual;
    layer.p_organic = p_organic_new.max(0.0);

    // K mineral: only depletes
    layer.k_mineral = (layer.k_mineral + dk_mineral * dt).max(0.0);

    // K exch: gains from weathering + litter return, loses to leach + uptake + fixation
    let k_out_total = (k_leach_flux + k_fix_slow) * dt + k_up_actual;
    let k_in_total  = (weathered_k + k_rel_slow) * dt + k_lit_return;
    layer.k_exch = (layer.k_exch - k_out_total.min(layer.k_exch) + k_in_total).max(0.0);

    // K fixed
    layer.k_fixed = (layer.k_fixed + dk_fixed * dt).max(0.0);

    layer.clamp_non_negative();
}

// ---------------------------------------------------------------------------
// Vertical leaching cascade (top → bottom layer)
// Dissolved P and K carry over from upper to lower layers proportionally
// to clay retention capacity.
// ---------------------------------------------------------------------------

fn step_vertical_leaching(column: &mut NutrientColumn, env: &ColumnEnv, dt: f64) {
    let dt = dt as f32;
    let runoff = env.runoff_m_yr;

    // Leachate cascades downward; retained fraction ∝ clay × CEC.
    // Each layer only passes forward the UNRETAINED portion of incoming leachate —
    // the layer's own pool is already leached by step_slow_reactions.
    // This prevents the cascade from amplifying geometrically.
    let mut p_leachate: f32 = 0.0;
    let mut k_leachate: f32 = 0.0;

    for i in 0..8 {
        let layer = &mut column.layers[i];
        let cec_retention = (column.clay_frac * column.cec / 40.0).clamp(0.0, 0.90);

        // How much of the incoming leachate does this layer retain?
        let p_retained = p_leachate * cec_retention;
        let k_retained = k_leachate * cec_retention;

        // Clamp retention to available pool headroom (prevent over-addition)
        let p_add = p_retained.min(layer.p_labile * 0.5);
        let k_add = k_retained.min(layer.k_exch   * 0.5);

        layer.p_labile = (layer.p_labile + p_add).max(0.0);
        layer.k_exch   = (layer.k_exch   + k_add).max(0.0);

        // Only the unretained portion passes to the next layer
        p_leachate = (p_leachate - p_retained) * (1.0 - K_LEACH_P * runoff * dt).max(0.0);
        k_leachate = (k_leachate - k_retained) * (1.0 - K_LEACH_K * runoff * dt).max(0.0);

        // Safety: leachate cannot exceed what was already removed from top layer
        p_leachate = p_leachate.min(column.layers[0].p_labile);
        k_leachate = k_leachate.min(column.layers[0].k_exch);
        // Last layer: remainder exits to groundwater
    }
}

// ---------------------------------------------------------------------------
// Erosion/deposition transport (Strang step 2)
// Nutrients piggyback on FastScape Δh sediment flux.
// ---------------------------------------------------------------------------

fn step_erosion_transport(column: &mut NutrientColumn, delta_h_m: f32) {
    if delta_h_m.abs() < 1e-4 {
        return;
    }

    let layer_thickness_m: f32 = 0.25; // 25 cm per layer

    if delta_h_m > 0.0 {
        // DEPOSITION: incoming sediment enriches top layer from below
        // Sediment arriving is assumed to carry average column concentrations
        let layers = column.layers;
        let mean_p_labile: f32  = layers.iter().map(|l| l.p_labile).sum::<f32>()  / 8.0;
        let mean_k_exch: f32    = layers.iter().map(|l| l.k_exch).sum::<f32>()    / 8.0;
        let mean_p_organic: f32 = layers.iter().map(|l| l.p_organic).sum::<f32>() / 8.0;

        let deposit_frac = (delta_h_m / layer_thickness_m).min(1.0);
        let top = &mut column.layers[0];
        top.p_labile  += mean_p_labile  * deposit_frac * 0.5; // partial dilution
        top.k_exch    += mean_k_exch    * deposit_frac * 0.5;
        top.p_organic += mean_p_organic * deposit_frac * 0.3;

    } else {
        // EROSION: removes from top layers proportional to erosion depth
        let eroded_frac = ((-delta_h_m) / layer_thickness_m).min(1.0);
        let top = &mut column.layers[0];
        top.p_labile   = (top.p_labile   * (1.0 - eroded_frac)).max(0.0);
        top.k_exch     = (top.k_exch     * (1.0 - eroded_frac)).max(0.0);
        top.p_organic  = (top.p_organic  * (1.0 - eroded_frac)).max(0.0);
        top.p_veg      = (top.p_veg      * (1.0 - eroded_frac * 0.5)).max(0.0);
    }
}

// ---------------------------------------------------------------------------
// Main solver: full Strang-split update for one column
// ---------------------------------------------------------------------------

/// Full Strang-split biogeochemistry update for one soil column.
///
/// Call once per geological epoch (Δt = 100 yr).
pub fn update_column(column: &mut NutrientColumn, env: &ColumnEnv, dt_years: f64) {
    let dt_half = dt_years * 0.5;

    // ---- STEP 1: Half-step fast reactions ----
    for layer in &mut column.layers {
        step_fast_reactions(layer, column.clay_frac, dt_half as f32);
    }

    // ---- STEP 2: Full-step transport (erosion/deposition + vertical leaching) ----
    step_erosion_transport(column, env.delta_h_m);
    step_vertical_leaching(column, env, dt_years);

    // ---- STEP 3: Half-step fast reactions ----
    for layer in &mut column.layers {
        step_fast_reactions(layer, column.clay_frac, dt_half as f32);
    }

    // ---- STEP 4: Full-step slow reactions, sub-stepped for stability ----
    // Fast biological rates (k_min=1.0/yr, k_uptake=2-4/yr) are stiff at Δt=100yr.
    // Sub-step into 10-year increments — cheap (8 layers × 10 sub-steps × ~20 ops).
    let n_substeps: usize = ((dt_years / 10.0).ceil() as usize).max(1);
    let dt_sub = dt_years / n_substeps as f64;
    for _ in 0..n_substeps {
        for (i, layer) in column.layers.iter_mut().enumerate() {
            step_slow_reactions(layer, column.clay_frac, column.root_density[i], env, dt_sub);
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel grid update — processes all columns in parallel via rayon
// ---------------------------------------------------------------------------

/// Biogeochemistry solver operating on a flat grid of nutrient columns.
pub struct BiogeochemSolver {
    pub grid_width:  usize,
    pub grid_height: usize,
}

impl BiogeochemSolver {
    pub fn new(grid_width: usize, grid_height: usize) -> Self {
        Self { grid_width, grid_height }
    }

    /// Update all columns in parallel.
    /// `columns`  — flat Vec of NutrientColumn (width × height)
    /// `envs`     — parallel Vec of ColumnEnv (one per column, pre-built)
    /// `dt_years` — geological timestep (typically 100 yr)
    pub fn step_epoch(
        &self,
        columns: &mut Vec<NutrientColumn>,
        envs:    &[ColumnEnv],
        dt_years: f64,
    ) {
        assert_eq!(columns.len(), envs.len());

        columns.par_iter_mut()
               .zip(envs.par_iter())
               .for_each(|(col, env)| {
                   update_column(col, env, dt_years);
               });
    }

    /// Build default environments from terrain state.
    /// In Phase 3 this will be driven by actual climate and BDI agent states.
    pub fn build_default_envs(
        &self,
        elevations:  &[f32],
        sea_level:   f32,
        delta_h:     &[f32],
        activity_mask: &[u32],
    ) -> Vec<ColumnEnv> {
        let n = self.grid_width * self.grid_height;
        (0..n).map(|i| {
            let elev = elevations.get(i).copied().unwrap_or(0.0);
            let dh   = delta_h.get(i).copied().unwrap_or(0.0);

            // Skip ocean columns
            if elev <= sea_level {
                return ColumnEnv {
                    temp_c: 4.0,
                    moisture: 1.0,
                    runoff_m_yr: 0.0,
                    flooded: false,
                    flood_p_input: 0.0,
                    delta_h_m: 0.0,
                    veg_cover: 0.0,
                };
            }

            // Check activity mask bit
            let word = activity_mask.get(i / 32).copied().unwrap_or(0);
            let active = (word >> (i % 32)) & 1 == 1;
            if !active {
                return ColumnEnv::default();
            }

            // Simple elevation-based defaults (Phase 3 will add real climate)
            let temp_c = (20.0 - elev * 0.006).max(-5.0); // lapse rate 6°C/km
            let moisture = if elev < 500.0 { 0.7 } else if elev < 1500.0 { 0.5 } else { 0.3 };
            let runoff = moisture * 0.6;
            let veg_cover = if elev > sea_level + 5.0 && elev < 2500.0 { 0.8 } else { 0.1 };

            // Floodplain: low-elevation deposition zones get periodic P renewal
            let flooded = dh > 0.05 && elev < sea_level + 50.0;
            let flood_p_input = if flooded { 8.0 } else { 0.0 };

            ColumnEnv {
                temp_c,
                moisture,
                runoff_m_yr: runoff,
                flooded,
                flood_p_input,
                delta_h_m: dh,
                veg_cover,
            }
        }).collect()
    }

    /// Compute grid of growth multipliers (0–1) for BDI agents / yield calc
    pub fn growth_multipliers(&self, columns: &[NutrientColumn]) -> Vec<f32> {
        columns.iter().map(|c| c.column_growth_multiplier()).collect()
    }

    /// Diagnostic: mean surface P_labile across all active columns
    pub fn mean_surface_p_labile(&self, columns: &[NutrientColumn], activity_mask: &[u32]) -> f32 {
        let mut sum = 0.0_f32;
        let mut count = 0_u32;
        for (i, col) in columns.iter().enumerate() {
            let word = activity_mask.get(i / 32).copied().unwrap_or(0);
            if (word >> (i % 32)) & 1 == 1 {
                sum += col.surface_p_labile();
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { sum / count as f32 }
    }

    /// Diagnostic: mean surface K_exch across all active columns
    pub fn mean_surface_k_exch(&self, columns: &[NutrientColumn], activity_mask: &[u32]) -> f32 {
        let mut sum = 0.0_f32;
        let mut count = 0_u32;
        for (i, col) in columns.iter().enumerate() {
            let word = activity_mask.get(i / 32).copied().unwrap_or(0);
            if (word >> (i % 32)) & 1 == 1 {
                sum += col.surface_k_exch();
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { sum / count as f32 }
    }
}

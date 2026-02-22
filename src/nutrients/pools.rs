// ============================================================================
// Nutrient Pools — Phase 2: Critical Zone
//
// 9 state variables per soil layer, modelling the coupled P-K biogeochemical
// system from Buendía et al. (2010) + K analogue.
//
// Pool layout (matches GPU SoA vec4 packing in buf A + B):
//   Buffer A: [P_labile, P_occluded, P_mineral, P_organic]
//   Buffer B: [K_exch,   K_fixed,    K_mineral,  K_veg]  +  P_veg piggybacked
//
// All concentrations are in mg/kg dry soil.
// All rate constants are in yr⁻¹ unless noted.
// ============================================================================

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants — calibrated from literature (see research doc)
// ---------------------------------------------------------------------------

/// P weathering rate constant (yr⁻¹) — Buendía et al. 2010
pub const K_W_P: f32           = 0.005;
/// Fraction of weathered P that enters labile pool (vs secondary minerals)
pub const PHI_P: f32           = 0.3;
/// P occlusion rate (yr⁻¹) — essentially irreversible Fe/Al binding
pub const K_OCC: f32           = 0.0001;
/// P mineralization from organic pool (yr⁻¹)
pub const K_MIN_P: f32         = 1.0;
/// P litterfall rate (yr⁻¹) — ~50% resorption so 50% returned to soil
pub const K_LIT_P: f32         = 0.2;
/// P plant uptake rate (yr⁻¹)
pub const K_UPTAKE_P: f32      = 2.0;
/// P leaching rate per unit runoff (dimensionless fraction × runoff yr⁻¹)
pub const K_LEACH_P: f32       = 0.005;
/// Tectonic P input (mg/kg/yr) — very slow geological renewal
pub const P_TECTONIC_INPUT: f32 = 0.001;

/// K weathering rate (yr⁻¹) — derived from field feldspar/mica dissolution
pub const K_W_K: f32           = 0.001;
/// K fixation rate in clay interlayers (yr⁻¹) — clay-content dependent
pub const K_FIX_BASE: f32      = 0.01;
/// K release from fixed pool (yr⁻¹) — Sparks 1987
pub const K_REL: f32           = 0.001;
/// K plant uptake (yr⁻¹) — higher than P because K is more demanded
pub const K_UPTAKE_K: f32      = 4.0;
/// K litterfall (yr⁻¹) — ~10-30% resorption so 70-90% returned (K is "wasted")
pub const K_LIT_K: f32         = 0.7;
/// K leaching rate per unit runoff
pub const K_LEACH_K: f32       = 0.01;

/// Michaelis-Menten half-saturation for P limitation (mg/kg)
pub const K_HALF_P: f32        = 5.0;
/// Michaelis-Menten half-saturation for K limitation (mg/kg)
pub const K_HALF_K: f32        = 50.0;

// ---------------------------------------------------------------------------
// NutrientLayer — one soil layer's complete P+K state (9 pools)
// ---------------------------------------------------------------------------

/// 9-pool coupled P-K state for one soil layer.
/// Packed for GPU SoA layout: buf A gets P pools, buf B gets K pools + K_veg.
/// P_veg is stored alongside K_veg in buf B (two f16-packed or separate slot).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NutrientLayer {
    // ---- PHOSPHORUS (4 pools) ----
    /// Primary mineral P (apatite). Source of all P. Geological depletion only.
    pub p_mineral:  f32,   // mg/kg  typical: 100–2000
    /// Labile inorganic P. Plant-accessible. Fast sorption/desorption target.
    pub p_labile:   f32,   // mg/kg  typical: 1–50
    /// Occluded P. Fe/Al-bound. Effectively irreversible sink.
    pub p_occluded: f32,   // mg/kg  typical: 10–500
    /// Organic P (in living/dead biomass and SOM).
    pub p_organic:  f32,   // mg/kg  typical: 50–800

    // ---- POTASSIUM (4 pools) ----
    /// Structural K in feldspars/micas. Primary geological source.
    pub k_mineral:  f32,   // mg/kg  typical: 5000–40000
    /// Exchangeable K on clay/OM CEC sites. Rapidly plant-accessible.
    pub k_exch:     f32,   // mg/kg  typical: 50–500
    /// Fixed K in clay interlayers (2:1 minerals). Slowly released.
    pub k_fixed:    f32,   // mg/kg  typical: 100–2000
    /// K in vegetation (the biocycling pump).
    pub k_veg:      f32,   // mg/kg  flux-dependent

    // ---- VEGETATION P (piggybacked) ----
    /// P in living vegetation (litterfall return closes the cycle).
    pub p_veg:      f32,   // mg/kg  flux-dependent
}

impl NutrientLayer {
    /// Default initialisation for mineral-rich young soil (post-glacial / basalt)
    pub fn young_mineral_soil() -> Self {
        Self {
            p_mineral:  800.0,
            p_labile:   15.0,
            p_occluded: 20.0,
            p_organic:  120.0,
            k_mineral:  15_000.0,
            k_exch:     200.0,
            k_fixed:    600.0,
            k_veg:      80.0,
            p_veg:      40.0,
        }
    }

    /// Default for old, deeply-weathered tropical soil (Oxisol-like)
    pub fn old_tropical_soil() -> Self {
        Self {
            p_mineral:  40.0,
            p_labile:   2.0,
            p_occluded: 350.0,
            p_organic:  80.0,
            k_mineral:  3_000.0,
            k_exch:     60.0,
            k_fixed:    200.0,
            k_veg:      30.0,
            p_veg:      8.0,
        }
    }

    /// Default for alluvial floodplain soil (flood-renewed P)
    pub fn alluvial_floodplain() -> Self {
        Self {
            p_mineral:  400.0,
            p_labile:   25.0,     // elevated by seasonal flood deposit
            p_occluded: 60.0,
            p_organic:  250.0,
            k_mineral:  12_000.0, // K-rich from feldspar-rich alluvium
            k_exch:     380.0,
            k_fixed:    800.0,
            k_veg:      100.0,
            p_veg:      50.0,
        }
    }

    /// Clamp all pools to non-negative (mass conservation)
    pub fn clamp_non_negative(&mut self) {
        self.p_mineral  = self.p_mineral.max(0.0);
        self.p_labile   = self.p_labile.max(0.0);
        self.p_occluded = self.p_occluded.max(0.0);
        self.p_organic  = self.p_organic.max(0.0);
        self.k_mineral  = self.k_mineral.max(0.0);
        self.k_exch     = self.k_exch.max(0.0);
        self.k_fixed    = self.k_fixed.max(0.0);
        self.k_veg      = self.k_veg.max(0.0);
        self.p_veg      = self.p_veg.max(0.0);
    }

    /// Plant-available P index (Michaelis-Menten)
    pub fn f_p(&self) -> f32 {
        self.p_labile / (self.p_labile + K_HALF_P)
    }

    /// Plant-available K index (Michaelis-Menten)
    pub fn f_k(&self) -> f32 {
        self.k_exch / (self.k_exch + K_HALF_K)
    }

    /// Liebig minimum growth multiplier (0–1)
    pub fn growth_multiplier(&self) -> f32 {
        self.f_p().min(self.f_k())
    }

    /// Total P (mass conservation diagnostic)
    pub fn total_p(&self) -> f32 {
        self.p_mineral + self.p_labile + self.p_occluded + self.p_organic + self.p_veg
    }

    /// Total K (mass conservation diagnostic)
    pub fn total_k(&self) -> f32 {
        self.k_mineral + self.k_exch + self.k_fixed + self.k_veg
    }
}

impl Default for NutrientLayer {
    fn default() -> Self {
        Self::young_mineral_soil()
    }
}

// ---------------------------------------------------------------------------
// DEFAC scalar — Q₁₀ temperature × moisture limitation
// ---------------------------------------------------------------------------

/// Decomposition factor (DEFAC): dimensionless rate multiplier applied to all
/// biological reactions. Combines temperature and moisture effects.
///
/// DEFAC = exp(0.07 × (T - 20)) × min(1.0, moisture / field_capacity)
///
/// This gives Q₁₀ ≈ 2 for temperature and a linear moisture ramp up to
/// field capacity, matching CENTURY model parameterization.
#[inline(always)]
pub fn defac(temp_celsius: f32, moisture_fraction: f32) -> f32 {
    let temp_factor = (0.07 * (temp_celsius - 20.0)).exp();
    let moist_factor = moisture_fraction.min(1.0).max(0.0);
    temp_factor * moist_factor
}

/// Soil column nutrient state: 8 layers per column (matching LayeredColumn)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NutrientColumn {
    pub layers: [NutrientLayer; 8],
    /// Clay fraction (0–1) for this column. Drives K fixation & P occlusion.
    pub clay_frac: f32,
    /// Root density weighting per layer (sums to 1.0). Top-heavy.
    pub root_density: [f32; 8],
    /// Cation exchange capacity (cmol/kg) for K retention scaling.
    pub cec: f32,
}

impl NutrientColumn {
    pub fn new_young_mineral(clay_frac: f32) -> Self {
        let cec = clay_frac * 35.0 + 5.0; // simple linear CEC from clay
        // Root density: exponential decline with depth
        let root_density = {
            let weights = [0.35_f32, 0.25, 0.18, 0.10, 0.06, 0.03, 0.02, 0.01];
            weights
        };
        Self {
            layers: [NutrientLayer::young_mineral_soil(); 8],
            clay_frac,
            root_density,
            cec,
        }
    }

    pub fn new_tropical(clay_frac: f32) -> Self {
        let cec = clay_frac * 20.0 + 3.0; // Oxisols have lower CEC
        let root_density = [0.40_f32, 0.28, 0.15, 0.08, 0.05, 0.02, 0.01, 0.01];
        Self {
            layers: [NutrientLayer::old_tropical_soil(); 8],
            clay_frac,
            root_density,
            cec,
        }
    }

    pub fn new_alluvial(clay_frac: f32) -> Self {
        let cec = clay_frac * 40.0 + 8.0;
        let root_density = [0.35_f32, 0.25, 0.18, 0.10, 0.06, 0.03, 0.02, 0.01];
        Self {
            layers: [NutrientLayer::alluvial_floodplain(); 8],
            clay_frac,
            root_density,
            cec,
        }
    }

    /// Weighted surface P_labile (top 3 layers, root-density weighted)
    pub fn surface_p_labile(&self) -> f32 {
        (0..3).map(|i| self.layers[i].p_labile * self.root_density[i]).sum::<f32>()
            / self.root_density[..3].iter().sum::<f32>().max(1e-6)
    }

    /// Weighted surface K_exch (top 3 layers)
    pub fn surface_k_exch(&self) -> f32 {
        (0..3).map(|i| self.layers[i].k_exch * self.root_density[i]).sum::<f32>()
            / self.root_density[..3].iter().sum::<f32>().max(1e-6)
    }

    /// Liebig growth multiplier for the whole column (root-weighted)
    pub fn column_growth_multiplier(&self) -> f32 {
        (0..8)
            .map(|i| self.layers[i].growth_multiplier() * self.root_density[i])
            .sum::<f32>()
    }
}

impl Default for NutrientColumn {
    fn default() -> Self {
        Self::new_young_mineral(0.3)
    }
}

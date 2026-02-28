//! Struct-of-Arrays (SoA) terrain grid for maximum cache utilization and SIMD auto-vectorization.
//!
//! **v0.3 Changes**:
//! - Removed BDI influence map double-buffering (Fix #1: agents replaced by land-use regimes)
//! - Added land_use_state for Markov Chain regime transitions (Fix #1)
//! - Added population_density for Fisher-KPP reaction-diffusion (Fix #5)
//! - Added sediment_thickness alias clarity for Tool & Cover (Fix #2)

use ndarray::Array2;

/// Land-use regime states for Markov Chain transitions (Fix #1).
/// Each cell is in exactly one macroscopic state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LandUseRegime {
    /// Pristine ecosystem — full canopy, tight nutrient cycling
    Pristine = 0,
    /// Lightly grazed — moderate progressive, low regressive
    Grazed = 1,
    /// Intensive agriculture — high regressive, moderate progressive from cultivation
    IntensiveAgriculture = 2,
    /// Degraded — soil stripped, minimal biological activity
    Degraded = 3,
}

impl LandUseRegime {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Pristine,
            1 => Self::Grazed,
            2 => Self::IntensiveAgriculture,
            3 => Self::Degraded,
            _ => Self::Pristine,
        }
    }

    /// Progressive pedogenesis rate contribution for this regime [1/yr]
    pub fn progressive_rate(&self) -> f64 {
        match self {
            Self::Pristine => 0.004,           // Strong: full litter, root systems
            Self::Grazed => 0.002,              // Moderate: manure + reduced litter
            Self::IntensiveAgriculture => 0.001, // Low: cultivation disrupts structure
            Self::Degraded => 0.0002,           // Minimal: sparse pioneer vegetation
        }
    }

    /// Regressive pedogenesis rate contribution for this regime [1/yr]
    pub fn regressive_rate(&self) -> f64 {
        match self {
            Self::Pristine => 0.0005,            // Minimal: protected by canopy
            Self::Grazed => 0.001,               // Low: some compaction, reduced cover
            Self::IntensiveAgriculture => 0.003,  // High: compaction, nutrient export, tillage
            Self::Degraded => 0.002,             // Moderate: exposed but less active disturbance
        }
    }
}

/// SoA terrain grid — each field is a contiguous f64 array for optimal SIMD vectorization.
pub struct TerrainGrid {
    pub rows: usize,
    pub cols: usize,
    pub dx: f64,
    pub dy: f64,

    // === Topography (dual-layer: surface = bedrock + sediment) ===
    pub elevation: Array2<f64>,
    pub bedrock: Array2<f64>,
    /// Physical sediment/soil thickness H_sed [m] — used for Tool & Cover (Fix #2)
    pub sediment: Array2<f64>,

    // === Hydrology ===
    pub drainage_area: Array2<f64>,
    pub discharge: Array2<f64>,
    pub slope: Array2<f64>,
    pub lake_level: Array2<f64>,

    // === Climate fields ===
    pub precipitation: Array2<f64>,
    pub temperature: Array2<f64>,
    pub evapotranspiration: Array2<f64>,

    // === Biogeochemistry ===
    pub soil_carbon: Array2<f64>,
    pub soil_nitrogen: Array2<f64>,
    pub carbon_flux: Array2<f64>,
    pub nitrogen_flux: Array2<f64>,

    // === Erosion/Deposition tracking ===
    pub sediment_flux: Array2<f64>,
    pub erosion_rate: Array2<f64>,

    // === Pedogenesis state (Fix #4: Michaelis-Menten, no ε-regularization) ===
    pub soil_state: Array2<f64>,

    // === Land-Use Regime (Fix #1: Markov Chain, replaces BDI agents) ===
    /// Current land-use regime per cell
    pub land_use: Array2<u8>,

    // === Population Density (Fix #5: Fisher-KPP reaction-diffusion) ===
    /// Abstract population density per cell [dimensionless, 0..carrying_capacity]
    pub population_density: Array2<f64>,
}

impl TerrainGrid {
    pub fn new(rows: usize, cols: usize, dx: f64, dy: f64) -> Self {
        Self {
            rows, cols, dx, dy,
            elevation: Array2::zeros((rows, cols)),
            bedrock: Array2::zeros((rows, cols)),
            sediment: Array2::zeros((rows, cols)),
            drainage_area: Array2::zeros((rows, cols)),
            discharge: Array2::zeros((rows, cols)),
            slope: Array2::zeros((rows, cols)),
            lake_level: Array2::zeros((rows, cols)),
            precipitation: Array2::zeros((rows, cols)),
            temperature: Array2::zeros((rows, cols)),
            evapotranspiration: Array2::zeros((rows, cols)),
            soil_carbon: Array2::zeros((rows, cols)),
            soil_nitrogen: Array2::zeros((rows, cols)),
            carbon_flux: Array2::zeros((rows, cols)),
            nitrogen_flux: Array2::zeros((rows, cols)),
            sediment_flux: Array2::zeros((rows, cols)),
            erosion_rate: Array2::zeros((rows, cols)),
            soil_state: Array2::zeros((rows, cols)),
            land_use: Array2::zeros((rows, cols)),  // All start as Pristine
            population_density: Array2::zeros((rows, cols)),
        }
    }

    pub fn init_random_perturbation(&mut self, base_elevation: f64, amplitude: f64, seed: u64) {
        let mut state = seed;
        for val in self.elevation.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = ((state >> 33) as f64) / (u32::MAX as f64);
            *val = base_elevation + amplitude * (r - 0.5);
        }
        self.bedrock.assign(&self.elevation);
        self.lake_level.assign(&self.elevation);
    }

    #[inline]
    pub fn len(&self) -> usize { self.rows * self.cols }

    #[inline]
    pub fn flat_index(&self, row: usize, col: usize) -> usize { row * self.cols + col }

    #[inline]
    pub fn grid_index(&self, idx: usize) -> (usize, usize) { (idx / self.cols, idx % self.cols) }

    #[inline]
    pub fn is_boundary(&self, row: usize, col: usize) -> bool {
        row == 0 || col == 0 || row == self.rows - 1 || col == self.cols - 1
    }

    pub fn sync_surface(&mut self) {
        for r in 0..self.rows {
            for c in 0..self.cols {
                self.elevation[[r, c]] = self.bedrock[[r, c]] + self.sediment[[r, c]];
            }
        }
    }
}

//! Struct-of-Arrays (SoA) terrain grid for maximum cache utilization and SIMD auto-vectorization.
//!
//! Each physics pass (climate, hydrology, geomorphology) accesses only the arrays it needs,
//! achieving near-100% cache line utilization vs the ~8% of an AoS layout.

use ndarray::Array2;

/// SoA terrain grid — each field is a contiguous f64 array for optimal SIMD vectorization.
pub struct TerrainGrid {
    pub rows: usize,
    pub cols: usize,
    pub dx: f64,
    pub dy: f64,

    // === Topography (dual-layer: surface = bedrock + sediment) ===
    /// Surface elevation h(x,y) = bedrock + sediment [m]
    pub elevation: Array2<f64>,
    /// Bedrock elevation [m]
    pub bedrock: Array2<f64>,
    /// Sediment thickness [m]
    pub sediment: Array2<f64>,

    // === Hydrology ===
    /// Upstream contributing drainage area [m²]
    pub drainage_area: Array2<f64>,
    /// Effective discharge Q_eff = ∫P(x,y)dA [m³/yr]
    pub discharge: Array2<f64>,
    /// Local topographic slope magnitude |∇h|
    pub slope: Array2<f64>,
    /// Lake level after depression filling [m]. Equals elevation if not in a lake.
    pub lake_level: Array2<f64>,

    // === Climate fields ===
    /// Spatial precipitation P(x,y) [m/yr]
    pub precipitation: Array2<f64>,
    /// Surface temperature T(x,y) [°C]
    pub temperature: Array2<f64>,
    /// Potential evapotranspiration [m/yr]
    pub evapotranspiration: Array2<f64>,

    // === Biogeochemistry (MPRK-integrated, sediment-coupled) ===
    /// Soil organic carbon [kg/m²]
    pub soil_carbon: Array2<f64>,
    /// Soil nitrogen pool [kg/m²]
    pub soil_nitrogen: Array2<f64>,
    /// Sediment-bound carbon flux being transported [kg/m/yr]
    pub carbon_flux: Array2<f64>,
    /// Sediment-bound nitrogen flux being transported [kg/m/yr]
    pub nitrogen_flux: Array2<f64>,

    // === Erosion/Deposition tracking ===
    /// Sediment flux q(x,y) [m²/yr] for ξ-q model
    pub sediment_flux: Array2<f64>,
    /// Local erosion rate [m/yr] (positive = erosion, negative = deposition)
    pub erosion_rate: Array2<f64>,
}

impl TerrainGrid {
    pub fn new(rows: usize, cols: usize, dx: f64, dy: f64) -> Self {
        Self {
            rows,
            cols,
            dx,
            dy,
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
        }
    }

    /// Initialize with stochastic perturbation on a flat surface.
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
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    #[inline]
    pub fn flat_index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    #[inline]
    pub fn grid_index(&self, idx: usize) -> (usize, usize) {
        (idx / self.cols, idx % self.cols)
    }

    #[inline]
    pub fn is_boundary(&self, row: usize, col: usize) -> bool {
        row == 0 || col == 0 || row == self.rows - 1 || col == self.cols - 1
    }

    /// Enforce the surface constraint: elevation = bedrock + sediment.
    /// Call after any operation that modifies bedrock or sediment independently.
    pub fn sync_surface(&mut self) {
        for r in 0..self.rows {
            for c in 0..self.cols {
                self.elevation[[r, c]] = self.bedrock[[r, c]] + self.sediment[[r, c]];
            }
        }
    }
}

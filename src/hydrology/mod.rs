/// Hydrology Module
///
/// Bridges the gap between orographic precipitation and downstream consumers
/// (nutrients, pedogenesis, carbon cycle) by computing per-cell:
///
/// - **Discharge**: accumulated flow from FastScape's drainage network
/// - **Water table depth**: estimated from drainage position + topographic wetness
/// - **Soil moisture**: physically-based from precipitation, PET, and drainage position
/// - **Flood status**: from discharge exceeding bankfull threshold
/// - **Runoff**: net water leaving each cell after infiltration
///
/// This replaces the disconnected proxy values that were hardcoded in the
/// nutrient solver and pedogenesis environments.

use rayon::prelude::*;

/// Per-cell hydrological state computed each epoch.
#[derive(Debug, Clone, Copy)]
pub struct CellHydrology {
    /// Precipitation falling on this cell (m/yr), from orographic engine.
    pub precipitation_m: f32,
    /// Local runoff generated at this cell (m/yr) = precip - ET - infiltration.
    pub runoff_m: f32,
    /// Accumulated upstream discharge at this cell (m³/yr).
    pub discharge_m3: f32,
    /// Topographic Wetness Index: ln(A / tan(slope)), dimensionless.
    /// High TWI = wet valley bottoms. Low TWI = dry ridgelines.
    pub wetness_index: f32,
    /// Estimated water table depth below surface (m). 0 = saturated.
    pub water_table_depth_m: f32,
    /// Soil moisture fraction [0, 1] derived from TWI + precipitation.
    pub soil_moisture: f32,
    /// Whether this cell is currently flooded (discharge > bankfull threshold).
    pub flooded: bool,
    /// Flood-delivered nutrient P input (kg/m² from upstream erosion).
    pub flood_p_input: f32,
    /// Specific discharge (discharge per unit width, m²/yr) for erosion coupling.
    pub specific_discharge: f32,
}

impl Default for CellHydrology {
    fn default() -> Self {
        Self {
            precipitation_m: 0.0,
            runoff_m: 0.0,
            discharge_m3: 0.0,
            wetness_index: 0.0,
            water_table_depth_m: 10.0,
            soil_moisture: 0.3,
            flooded: false,
            flood_p_input: 0.0,
            specific_discharge: 0.0,
        }
    }
}

/// Configuration for the hydrology solver.
#[derive(Debug, Clone)]
pub struct HydrologyConfig {
    /// Bankfull discharge threshold (m³/yr). Cells exceeding this are "flooded".
    /// Scales with cell size: larger cells = larger rivers = higher threshold.
    pub bankfull_discharge_m3: f32,
    /// Maximum water table depth (m) in well-drained uplands.
    pub max_water_table_depth: f32,
    /// Infiltration capacity (fraction of precipitation that infiltrates).
    pub infiltration_fraction: f32,
    /// P concentration in floodwater (kg/m³) — enriched sediment from upstream.
    pub flood_p_concentration: f32,
}

impl Default for HydrologyConfig {
    fn default() -> Self {
        Self {
            bankfull_discharge_m3: 1e9, // ~30 m³/s equivalent for 1km cells
            max_water_table_depth: 30.0,
            infiltration_fraction: 0.4,
            flood_p_concentration: 0.005,
        }
    }
}

/// The hydrology solver.
pub struct HydrologySolver {
    pub config: HydrologyConfig,
    pub grid_width: usize,
    pub grid_height: usize,
    pub cell_size_m: f32,
    /// Per-cell hydrology state, updated each epoch.
    pub cells: Vec<CellHydrology>,
}

impl HydrologySolver {
    pub fn new(grid_width: usize, grid_height: usize, cell_size_m: f32, config: HydrologyConfig) -> Self {
        let n = grid_width * grid_height;
        Self {
            config,
            grid_width,
            grid_height,
            cell_size_m,
            cells: vec![CellHydrology::default(); n],
        }
    }

    /// Compute hydrology for the current epoch.
    ///
    /// Inputs come from:
    /// - `precipitation`: per-cell from OrographicEngine (m/yr)
    /// - `pet`: per-cell potential evapotranspiration from LapseRateEngine (m/yr)
    /// - `elevations`: current terrain
    /// - `drainage_area`: from FastScape's flow accumulation (m²)
    /// - `slope`: per-cell surface slope (from FastScape or computed here)
    /// - `receivers`: FastScape's drainage receiver indices
    /// - `delta_h`: erosion/deposition from last geology step
    /// - `sea_level`: current sea level
    pub fn compute(
        &mut self,
        precipitation: &[f32],
        pet: &[f32],
        elevations: &[f32],
        drainage_area: &[f32],
        slopes: &[f32],
        _receivers: &[usize],
        delta_h: &[f32],
        sea_level: f32,
    ) {
        let n = self.grid_width * self.grid_height;
        let cell_area = self.cell_size_m * self.cell_size_m;
        let cell_width = self.cell_size_m;
        let bankfull = self.config.bankfull_discharge_m3;
        let max_wtd = self.config.max_water_table_depth;
        let _infil_frac = self.config.infiltration_fraction;
        let flood_p_conc = self.config.flood_p_concentration;

        // Parallel computation of per-cell hydrology
        self.cells = (0..n).into_par_iter().map(|i| {
            let elev = elevations.get(i).copied().unwrap_or(0.0);
            let precip = precipitation.get(i).copied().unwrap_or(0.0);
            let pet_i = pet.get(i).copied().unwrap_or(0.3);
            let area = drainage_area.get(i).copied().unwrap_or(cell_area);
            let slope = slopes.get(i).copied().unwrap_or(0.001).max(1e-4);
            let _dh = delta_h.get(i).copied().unwrap_or(0.0);

            // Underwater cells
            if elev <= sea_level {
                return CellHydrology {
                    precipitation_m: precip,
                    water_table_depth_m: 0.0,
                    soil_moisture: 1.0,
                    ..Default::default()
                };
            }

            // --- Runoff: Budyko-style water balance ---
            // AET = min(PET, precip) with smooth transition
            let aridity = pet_i / precip.max(0.01);
            // Budyko curve: AET/P = 1 + PET/P - (1 + (PET/P)^w)^(1/w), w=2
            let aet = if aridity < 0.01 {
                0.0
            } else {
                let w = 2.0f32;
                let ratio = 1.0 + aridity - (1.0 + aridity.powf(w)).powf(1.0 / w);
                (ratio * precip).max(0.0)
            };
            let runoff = (precip - aet).max(0.0);

            // --- Discharge: from FastScape drainage accumulation ---
            // drainage_area already includes upstream contributing area
            let discharge = area * runoff.max(0.01); // m² * m/yr = m³/yr

            // --- Topographic Wetness Index ---
            // TWI = ln(A / tan(slope))
            let tan_slope = slope.max(1e-4);
            let twi = (area / tan_slope).ln().max(0.0);

            // --- Water table depth ---
            // Inversely related to TWI: valleys (high TWI) = shallow water table
            // Ridges (low TWI) = deep water table
            let twi_norm = (twi / 15.0).clamp(0.0, 1.0); // TWI ~0-15 typical range
            let water_table = max_wtd * (1.0 - twi_norm * 0.9);

            // --- Soil moisture ---
            // Combine water balance (precip/PET) with landscape position (TWI)
            let climate_moisture = (1.0 - aet / precip.max(0.01)).clamp(0.0, 1.0);
            let position_moisture = twi_norm;
            // Weighted: 60% climate-driven, 40% landscape-position
            let soil_moisture = (0.6 * climate_moisture + 0.4 * position_moisture).clamp(0.0, 1.0);

            // --- Flood detection ---
            // A cell floods when upstream discharge exceeds bankfull capacity
            let flooded = discharge > bankfull && elev < sea_level + 200.0;

            // --- Flood P input ---
            // Floodwaters carry P from upstream erosion
            let flood_p = if flooded {
                // Volume of floodwater above bankfull, per cell area
                let excess_discharge = discharge - bankfull;
                let flood_depth = excess_discharge / cell_area; // m/yr of flooding
                flood_depth * flood_p_conc // kg/m²
            } else {
                0.0
            };

            // --- Specific discharge (for erosion coupling) ---
            let specific_discharge = discharge / cell_width;

            CellHydrology {
                precipitation_m: precip,
                runoff_m: runoff,
                discharge_m3: discharge,
                wetness_index: twi,
                water_table_depth_m: water_table,
                soil_moisture,
                flooded,
                flood_p_input: flood_p,
                specific_discharge,
            }
        }).collect();
    }

    // --- Diagnostics ---

    pub fn mean_runoff(&self) -> f32 {
        let sum: f32 = self.cells.iter().map(|c| c.runoff_m).sum();
        sum / self.cells.len().max(1) as f32
    }

    pub fn mean_soil_moisture(&self) -> f32 {
        let sum: f32 = self.cells.iter().map(|c| c.soil_moisture).sum();
        sum / self.cells.len().max(1) as f32
    }

    pub fn flooded_count(&self) -> usize {
        self.cells.iter().filter(|c| c.flooded).count()
    }

    pub fn max_discharge(&self) -> f32 {
        self.cells.iter().map(|c| c.discharge_m3).fold(0.0f32, f32::max)
    }

    pub fn mean_water_table(&self) -> f32 {
        let sum: f32 = self.cells.iter().map(|c| c.water_table_depth_m).sum();
        sum / self.cells.len().max(1) as f32
    }
}

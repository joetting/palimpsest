// ============================================================================
// Geological Carbon Cycle Engine — Carbonate-Silicate Feedback
//
// Implements the Walker-Kasting CO₂ thermostat:
//
//   Physical erosion → exposes fresh silicate bedrock
//   → Carbonic acid weathering: CaSiO₃ + CO₂ + H₂O → CaCO₃ + SiO₂ + H₂O
//   → CO₂ drawn from atmosphere into soluble bicarbonate (HCO₃⁻)
//   → Transported to ocean → buried as carbonate sediment
//   → Net: CO₂ removed from atmosphere
//
// Tectonic uplifts create steep slopes → mass wasting → fresh rock exposure
// → accelerated CO₂ drawdown → global cooling → potential ice age.
//
// The "Carbon Flush" inverse:
//   Agricultural/deforestation erosion → ancient topsoil SOM exposed to O₂
//   → Rapid oxidation → CO₂ release → warming pulse
//
// Organic carbon burial (high-erosion, low-O₂ floodplains):
//   Fast burial under silt → organic matter preserved before decomposition
//   → Long-term sequestration (the origin of fossil fuels)
// ============================================================================

use rayon::prelude::*;
use crate::climate::thermostat::ClimateState;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SilicateWeatheringParams {
    /// Baseline silicate weathering rate at 15°C, zero erosion [ppm CO₂/yr/km²]
    pub base_rate_ppm_yr_km2: f32,
    /// Q10 temperature coefficient — rate doubles every 10°C (standard for geochemistry)
    pub q10: f32,
    /// Erosion amplification factor — how much erosion multiplies weathering rate
    /// (erosion exposes fresh unweathered minerals; typical factor 2–10×)
    pub erosion_amplification: f32,
    /// Soil shield factor — deep, mature soil inhibits weathering by blocking acid contact
    /// Range [0, 1]: 1 = no shielding, 0 = fully shielded
    pub soil_shield_factor: f32,
    /// Organic burial efficiency: fraction of eroded SOM that gets buried (not oxidized)
    /// In high-energy deposition zones this can be 0.4–0.6
    pub organic_burial_efficiency: f32,
    /// Carbon flush coefficient: fraction of SOM exposed by erosion that rapidly oxidizes
    pub carbon_flush_fraction: f32,
    /// Reference temperature for Q10 scaling [°C]
    pub reference_temp_c: f32,
}

impl Default for SilicateWeatheringParams {
    fn default() -> Self {
        Self {
            base_rate_ppm_yr_km2:     0.0003, // ~0.3 ppb CO₂/yr per km² of continent
            q10:                       2.0,
            erosion_amplification:     5.0,
            soil_shield_factor:        0.8,   // typical mature soil shields ~80% of bedrock
            organic_burial_efficiency: 0.45,
            carbon_flush_fraction:     0.30,  // 30% of exposed SOM rapidly oxidizes
            reference_temp_c:          15.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Carbon flux output
// ---------------------------------------------------------------------------

/// Net carbon flux outputs for one epoch, passed to the thermostat.
#[derive(Debug, Clone, Default)]
pub struct CarbonFluxes {
    /// Total CO₂ drawn down by silicate weathering [ppm/yr]
    pub silicate_drawdown: f32,
    /// Total CO₂ sequestered by organic carbon burial [ppm/yr]
    pub organic_burial: f32,
    /// Total CO₂ released by anthropogenic / erosion carbon flush [ppm/yr]
    pub anthropogenic_flush: f32,
    /// Diagnostic: number of active erosion cells
    pub active_erosion_cells: usize,
    /// Diagnostic: mean erosion rate [mm/yr]
    pub mean_erosion_rate_mm_yr: f32,
    /// Diagnostic: tectonic contribution (uplift-driven weathering) [ppm/yr]
    pub tectonic_contribution: f32,
}

impl CarbonFluxes {
    pub fn summary(&self) -> String {
        format!(
            "Silicate↓={:.4}ppm/yr  OrgBurial↓={:.4}ppm/yr  CFlush↑={:.4}ppm/yr  ErosionCells={}  MeanErosion={:.3}mm/yr",
            self.silicate_drawdown,
            self.organic_burial,
            self.anthropogenic_flush,
            self.active_erosion_cells,
            self.mean_erosion_rate_mm_yr,
        )
    }
}

// ---------------------------------------------------------------------------
// Carbon Cycle Engine
// ---------------------------------------------------------------------------

pub struct CarbonCycleEngine {
    pub params: SilicateWeatheringParams,
    /// Domain area per cell [km²]
    cell_area_km2: f32,
}

impl CarbonCycleEngine {
    pub fn new(params: SilicateWeatheringParams) -> Self {
        Self {
            params,
            cell_area_km2: 1.0, // default 1 km² cells; updated from terrain config
        }
    }

    pub fn set_cell_size_m(&mut self, cell_size_m: f32) {
        self.cell_area_km2 = (cell_size_m / 1000.0).powi(2);
    }

    /// Compute carbon fluxes for one geological epoch.
    ///
    /// erosion_delta_h: Δh per column from FastScape (negative = eroded) [m]
    /// Returns CarbonFluxes to feed into the Proserpina thermostat.
    pub fn compute_fluxes(
        &self,
        elevations:  &[f32],
        delta_h:     &[f32],
        grid_width:  usize,
        grid_height: usize,
        climate:     &ClimateState,
    ) -> CarbonFluxes {
        let n = grid_width * grid_height;
        let temp = climate.global_mean_temp_c;

        // Q10 temperature scaling: rate multiplier relative to reference
        let q10_scale = self.q10.powf((temp - self.params.reference_temp_c) / 10.0);

        // Parallel reduction across all cells
        let (
            total_silicate, total_burial, total_flush,
            erosion_count, erosion_sum
        ) = (0..n).into_par_iter().map(|i| {
            let elev = elevations.get(i).copied().unwrap_or(0.0);
            let dh   = delta_h.get(i).copied().unwrap_or(0.0);

            // Only land cells above sea level
            if elev <= 0.0 {
                return (0.0_f32, 0.0_f32, 0.0_f32, 0_usize, 0.0_f32);
            }

            let erosion_m = (-dh).max(0.0); // positive = eroded
            let deposition_m = dh.max(0.0); // positive = deposited

            // ---- SILICATE WEATHERING ----
            // Erosion exposes fresh minerals; rate scales with erosion intensity
            let erosion_amplifier = 1.0 + self.params.erosion_amplification * erosion_m * 100.0;
            let silicate_rate = self.params.base_rate_ppm_yr_km2
                * self.cell_area_km2
                * q10_scale
                * erosion_amplifier;

            // ---- ORGANIC CARBON BURIAL ----
            // Rapid deposition buries SOM before oxidation in O₂
            let burial = if deposition_m > 0.01 {
                // Buried organic matter estimated from erosion flux × soil OM density (~2% OM)
                let som_density_kg_m3 = 20.0; // 2% OM in 1000 kg/m³ soil
                let buried_som = deposition_m * som_density_kg_m3 * self.cell_area_km2 * 1e6;
                // Convert to ppm CO₂ equivalent (very rough; ~0.0001 ppm per kg per domain)
                let domain_mass_scale = 1e-12;
                buried_som * self.params.organic_burial_efficiency * domain_mass_scale
            } else {
                0.0
            };

            // ---- CARBON FLUSH ----
            // Erosion of topsoil exposes ancient SOM to oxidation
            let flush = if erosion_m > 0.001 {
                // Ancient SOM concentration in deep soils (~1% by mass)
                let ancient_som = erosion_m * 10.0 * self.cell_area_km2 * 1e6; // kg
                let domain_mass_scale = 1e-12;
                ancient_som * self.params.carbon_flush_fraction * domain_mass_scale
            } else {
                0.0
            };

            let erosion_active = if erosion_m > 1e-4 { 1 } else { 0 };
            (silicate_rate, burial, flush, erosion_active, erosion_m * 1000.0) // mm
        }).reduce(
            || (0.0_f32, 0.0_f32, 0.0_f32, 0_usize, 0.0_f32),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4),
        );

        let mean_erosion = if erosion_count > 0 {
            erosion_sum / erosion_count as f32
        } else {
            0.0
        };

        CarbonFluxes {
            silicate_drawdown:    total_silicate,
            organic_burial:       total_burial,
            anthropogenic_flush:  total_flush,
            active_erosion_cells: erosion_count,
            mean_erosion_rate_mm_yr: mean_erosion,
            tectonic_contribution: total_silicate * 0.6, // ~60% from tectonic-driven erosion
        }
    }

    /// Estimate the CO₂ drawdown from a single Elder God tectonic uplift event.
    ///
    /// Used for immediate feedback when the player triggers an orogenic event.
    /// Returns estimated ppm drawdown over the subsequent ~1 Myr.
    pub fn estimate_uplift_drawdown(&self, radius_km: f32, rate_m_yr: f32) -> f32 {
        // Very rough: uplift creates ~10× normal erosion flux for ~500 kyr
        let area_km2 = std::f32::consts::PI * radius_km * radius_km;
        let enhanced_rate = self.params.base_rate_ppm_yr_km2 * area_km2 * 10.0 * rate_m_yr;
        enhanced_rate * 500_000.0 // over 500 kyr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::climate::thermostat::ClimateState;
    use crate::climate::thermostat::C0_BASELINE_PPM;

    #[test]
    fn test_erosion_increases_drawdown() {
        let engine = CarbonCycleEngine::new(SilicateWeatheringParams::default());
        let state = ClimateState::new(C0_BASELINE_PPM, 14.0, 3.0);
        let elevs = vec![500.0f32; 4];

        // No erosion
        let delta_h_none = vec![0.0f32; 4];
        let flux_none = engine.compute_fluxes(&elevs, &delta_h_none, 2, 2, &state);

        // Heavy erosion: -2m per epoch
        let delta_h_heavy = vec![-2.0f32; 4];
        let flux_heavy = engine.compute_fluxes(&elevs, &delta_h_heavy, 2, 2, &state);

        assert!(
            flux_heavy.silicate_drawdown > flux_none.silicate_drawdown,
            "Erosion should increase silicate drawdown"
        );
    }
}

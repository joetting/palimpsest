// ============================================================================
// Environmental Lapse Rate Engine
//
// Calculates local temperature and moisture from global base values using the
// environmental lapse rate (ELR): approximately −6.5°C per 1,000 m of elevation.
//
// This creates vertical biome stratification without any fluid dynamics:
//   - Tropical forest at sea-level lowlands
//   - Deciduous forest at mid-elevation slopes
//   - Treeless alpine tundra near summits
//   - Permafrost/glacier above the frost line
//
// Also computes evapotranspiration (ET) using a Priestley-Taylor approximation,
// allowing the runoff = precipitation − ET feedback for FastScape drainage area.
// ============================================================================

use crate::climate::thermostat::ClimateConfig;

/// Environmental lapse rate [°C / m]
pub const ELR: f32 = 0.0065; // 6.5°C per 1000 m

/// Frost line elevation offset [m] — elevation at which T = 0°C above base_temp=14°C
/// Purely derived: 14.0 / 0.0065 ≈ 2154 m
pub const FROST_LINE_M: f32 = 2154.0;

/// Reference elevation for lapse rate calculation [m] (sea level)
pub const REFERENCE_ELEVATION_M: f32 = 0.0;

/// Priestley-Taylor α coefficient (dimensionless, ≈1.26 for humid conditions)
pub const PT_ALPHA: f32 = 1.26;

/// Reference potential evapotranspiration [m/yr] at 15°C (Penman approximation)
pub const ET_REFERENCE_M_YR: f32 = 0.60;

pub struct LapseRateEngine {
    pub elr: f32, // °C / m
    pub et_reference: f32,
}

impl LapseRateEngine {
    pub fn new(_config: ClimateConfig) -> Self {
        Self {
            elr: ELR,
            et_reference: ET_REFERENCE_M_YR,
        }
    }

    /// Local temperature at given elevation given global base temperature.
    ///
    /// T_local = T_global_base − ELR × (elevation − reference_elevation)
    #[inline]
    pub fn local_temp(&self, base_temp_c: f32, elevation_m: f32) -> f32 {
        base_temp_c - self.elr * (elevation_m - REFERENCE_ELEVATION_M).max(0.0)
    }

    /// Potential evapotranspiration (PET) [m/yr] at given temperature.
    ///
    /// Simplified Priestley-Taylor: PET scales with temperature above 0°C.
    /// Below 0°C: frozen ground, evaporation near zero.
    #[inline]
    pub fn pet(&self, temp_c: f32) -> f32 {
        if temp_c <= 0.0 {
            return 0.02; // minimal sublimation from ice/snow
        }
        // Linear scaling: reference ET at 15°C, doubles by ~35°C
        let temp_scale = (temp_c / 15.0).max(0.0).min(3.0);
        self.et_reference * PT_ALPHA * temp_scale
    }

    /// Actual evapotranspiration (AET) = min(PET, precipitation).
    /// Runoff = precipitation - AET
    #[inline]
    pub fn runoff(&self, precip_m: f32, temp_c: f32, elevation_m: f32) -> f32 {
        // At very high elevation, precipitation falls as snow (no runoff until melt)
        let snow_fraction = if temp_c < 0.0 { 1.0 } else { 0.0 };
        let liquid_precip = precip_m * (1.0 - snow_fraction);

        let pet = self.pet(temp_c);
        let aet = pet.min(liquid_precip);
        (liquid_precip - aet).max(0.0)
    }

    /// Soil moisture as fraction of field capacity [0–1].
    /// Computed from the ratio of precipitation to PET (Budyko aridity index).
    #[inline]
    pub fn soil_moisture(&self, precip_m: f32, temp_c: f32) -> f32 {
        let pet = self.pet(temp_c);
        if pet < 1e-6 {
            return 1.0; // frozen/polar — soil is saturated (ice)
        }
        let aridity = pet / precip_m.max(1e-6); // Budyko aridity index
        // Moisture = 1 / (1 + aridity) — hyperbolic decline from wet to dry
        (1.0 / (1.0 + aridity * aridity)).clamp(0.0, 1.0)
    }

    /// Elevation of the snowline for a given base temperature [m].
    /// Above snowline, mean annual temperature < 0°C.
    pub fn snowline_elevation(&self, base_temp_c: f32) -> f32 {
        if base_temp_c <= 0.0 {
            return 0.0; // entire terrain is frozen
        }
        base_temp_c / self.elr
    }

    /// Freeze-thaw cycle intensity at a given elevation (0–1).
    /// Drives mechanical frost weathering — increases erodibility.
    /// Peaks at the freeze-thaw boundary (around 0°C mean annual temp).
    pub fn freeze_thaw_intensity(&self, base_temp_c: f32, elevation_m: f32) -> f32 {
        let local_temp = self.local_temp(base_temp_c, elevation_m);
        // Maximum intensity near 0°C, Gaussian decay on both sides
        let sigma = 5.0_f32; // ±5°C half-width
        (-(local_temp / sigma).powi(2)).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::climate::thermostat::ClimateConfig;

    #[test]
    fn test_lapse_rate_correct() {
        let engine = LapseRateEngine::new(ClimateConfig::default());
        // At 1000m above sea level with 14°C base: should be 14 - 6.5 = 7.5°C
        let t = engine.local_temp(14.0, 1000.0);
        assert!((t - 7.5).abs() < 0.01, "Expected 7.5°C at 1000m, got {}", t);
    }

    #[test]
    fn test_snowline() {
        let engine = LapseRateEngine::new(ClimateConfig::default());
        // With base 14°C, snowline should be at ~2154m
        let sl = engine.snowline_elevation(14.0);
        assert!((sl - 2153.8).abs() < 5.0, "Expected ~2154m snowline, got {}", sl);
    }

    #[test]
    fn test_runoff_non_negative() {
        let engine = LapseRateEngine::new(ClimateConfig::default());
        let r = engine.runoff(0.3, 20.0, 100.0);
        assert!(r >= 0.0, "Runoff should be non-negative");
    }
}

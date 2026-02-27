//! Reduced-complexity climate models for landscape-climate co-evolution.
//!
//! Three models implemented:
//! 1. Linear Theory (LT) of Orographic Precipitation (Smith & Barstad 2004) — O(N log N) via FFT
//! 2. Linear Feedback Precipitation Model (LFPM 1.0, Hergarten & Robl 2022) — O(N) block-tridiagonal
//! 3. Zero-dimensional Energy Balance Model (EBM) — O(N) pointwise
//!
//! All models are steady-state: given current topography, they instantly compute spatial
//! precipitation/temperature fields without tracking transient atmospheric dynamics.

use ndarray::Array2;
use crate::grid::TerrainGrid;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

// ============================================================================
// 1. LINEAR THEORY OF OROGRAPHIC PRECIPITATION (Smith & Barstad 2004)
// ============================================================================

/// Parameters for the Linear Theory orographic precipitation model.
#[derive(Clone, Debug)]
pub struct LinearTheoryParams {
    /// Background precipitation rate P₀ [m/yr]
    pub p0: f64,
    /// Wind speed |U| [m/s]
    pub wind_speed: f64,
    /// Wind direction (radians, 0 = west-to-east)
    pub wind_direction: f64,
    /// Moist adiabatic lapse rate Γ_m [°C/m]
    pub lapse_rate: f64,
    /// Reference saturation vapor density at sea level [kg/m³]
    pub rho_sref: f64,
    /// Moist layer depth H_w [m]
    pub hw: f64,
    /// Cloud water conversion time τ_c [s]
    pub tau_c: f64,
    /// Hydrometeor fallout time τ_f [s]
    pub tau_f: f64,
    /// Moist stability frequency N_m [1/s] (squared; <0 for unstable)
    pub nm_sq: f64,
}

impl Default for LinearTheoryParams {
    fn default() -> Self {
        Self {
            p0: 1.0,          // 1 m/yr background
            wind_speed: 10.0,  // 10 m/s
            wind_direction: 0.0, // Westerly
            lapse_rate: 6.5e-3,
            rho_sref: 7.5e-3,
            hw: 2500.0,
            tau_c: 1000.0,
            tau_f: 1000.0,
            nm_sq: 0.005 * 0.005,
        }
    }
}

/// Compute orographic precipitation using the Linear Theory (Smith & Barstad 2004).
///
/// Uses FFT to convert the topographic forcing into the wavenumber domain,
/// applies the transfer function, and inverse FFTs to get spatial precipitation.
/// Complexity: O(N log N) where N = rows × cols.
pub fn linear_theory_precipitation(
    grid: &mut TerrainGrid,
    params: &LinearTheoryParams,
) {
    let rows = grid.rows;
    let cols = grid.cols;
    let dx = grid.dx;
    let dy = grid.dy;

    // Wind components
    let u_wind = params.wind_speed * params.wind_direction.cos();
    let v_wind = params.wind_speed * params.wind_direction.sin();

    // Moisture scale factor C_w = rho_sref * Gamma_m / rho_water
    let c_w = params.rho_sref * params.lapse_rate;

    // Advection distances
    let _tau_c_u = params.tau_c * params.wind_speed;
    let _tau_f_u = params.tau_f * params.wind_speed;

    // We need to do 2D FFT. Since rustfft only does 1D, we do row-by-row then col-by-col.
    let mut planner = FftPlanner::<f64>::new();
    let fft_cols = planner.plan_fft_forward(cols);
    let fft_rows = planner.plan_fft_forward(rows);
    let ifft_cols = planner.plan_fft_inverse(cols);
    let ifft_rows = planner.plan_fft_inverse(rows);

    // Create complex elevation field
    let mut field: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            field[r * cols + c] = Complex::new(grid.elevation[[r, c]], 0.0);
        }
    }

    // Forward FFT: rows
    let mut scratch = vec![Complex::new(0.0, 0.0); rows.max(cols)];
    for r in 0..rows {
        let start = r * cols;
        let row_slice = &mut field[start..start + cols];
        fft_cols.process_with_scratch(row_slice, &mut scratch[..cols]);
    }

    // Forward FFT: columns (need to extract/insert column data)
    let mut col_buf = vec![Complex::new(0.0, 0.0); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = field[r * cols + c];
        }
        fft_rows.process_with_scratch(&mut col_buf, &mut scratch[..rows]);
        for r in 0..rows {
            field[r * cols + c] = col_buf[r];
        }
    }

    // Apply transfer function in wavenumber domain
    let norm = 1.0 / (rows * cols) as f64;
    for kr in 0..rows {
        for kc in 0..cols {
            // Wavenumber components
            let k = if kc <= cols / 2 {
                2.0 * PI * kc as f64 / (cols as f64 * dx)
            } else {
                2.0 * PI * (kc as f64 - cols as f64) / (cols as f64 * dx)
            };
            let l = if kr <= rows / 2 {
                2.0 * PI * kr as f64 / (rows as f64 * dy)
            } else {
                2.0 * PI * (kr as f64 - rows as f64) / (rows as f64 * dy)
            };

            // Skip DC component
            if kr == 0 && kc == 0 {
                field[0] = Complex::new(0.0, 0.0);
                continue;
            }

            // Intrinsic frequency σ = U·k + V·l
            let sigma = u_wind * k + v_wind * l;

            // Vertical wavenumber (simplified)
            let k_mag_sq = k * k + l * l;
            let m_sq = if k_mag_sq > 1e-30 {
                (params.nm_sq - sigma * sigma) * k_mag_sq / (sigma * sigma + 1e-30)
            } else {
                0.0
            };

            // Transfer function numerator: C_w * i * σ * (vertical structure)
            let numerator = Complex::new(0.0, c_w * sigma * params.hw);

            // Denominator: (1 - i*m*H_w) * (1 + i*σ*τ_c) * (1 + i*σ*τ_f)
            let m_vert = if m_sq > 0.0 { m_sq.sqrt() } else { 0.0 };
            let d1 = Complex::new(1.0, -m_vert * params.hw);
            let d2 = Complex::new(1.0, sigma * params.tau_c);
            let d3 = Complex::new(1.0, sigma * params.tau_f);
            let denominator = d1 * d2 * d3;

            let denom_mag = denominator.norm_sqr();
            if denom_mag > 1e-30 {
                let transfer = numerator / denominator;
                field[kr * cols + kc] = field[kr * cols + kc] * transfer;
            } else {
                field[kr * cols + kc] = Complex::new(0.0, 0.0);
            }
        }
    }

    // Inverse FFT: columns
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = field[r * cols + c];
        }
        ifft_rows.process_with_scratch(&mut col_buf, &mut scratch[..rows]);
        for r in 0..rows {
            field[r * cols + c] = col_buf[r];
        }
    }

    // Inverse FFT: rows
    for r in 0..rows {
        let start = r * cols;
        let row_slice = &mut field[start..start + cols];
        ifft_cols.process_with_scratch(row_slice, &mut scratch[..cols]);
    }

    // Write precipitation = P₀ + orographic component (clamped ≥ 0)
    for r in 0..rows {
        for c in 0..cols {
            let orog = field[r * cols + c].re * norm;
            grid.precipitation[[r, c]] = (params.p0 + orog).max(0.0);
        }
    }
}

// ============================================================================
// 2. LINEAR FEEDBACK PRECIPITATION MODEL (LFPM 1.0, Hergarten & Robl 2022)
// ============================================================================

/// Parameters for LFPM 1.0.
#[derive(Clone, Debug)]
pub struct LfpmParams {
    /// Background vapor flux F_v0 [m²/yr]
    pub fv0: f64,
    /// Condensation length scale L_c [m]
    pub l_c: f64,
    /// Fallout length scale L_f [m]
    pub l_f: f64,
    /// Transversal dispersion length scale L_d [m]
    pub l_d: f64,
    /// Sea-level re-evaporation coefficient β₀
    pub beta_0: f64,
    /// Thermodynamic scale height H₀ [m]
    pub h0: f64,
    /// Advection direction (radians, 0 = along x-axis)
    pub advection_direction: f64,
}

impl Default for LfpmParams {
    fn default() -> Self {
        Self {
            fv0: 500.0,
            l_c: 50_000.0,  // 50 km
            l_f: 5_000.0,   // 5 km
            l_d: 10_000.0,  // 10 km
            beta_0: 0.8,
            h0: 2500.0,
            advection_direction: 0.0,
        }
    }
}

/// Compute precipitation using LFPM 1.0 — block-tridiagonal O(N) solver.
///
/// Solves two coupled advection-dispersion equations for vapor flux F_v and cloud
/// water flux F_c on the Fastscape grid using an upwind scheme along the advection
/// axis and implicit finite differences for transversal dispersion.
///
/// The key geomorphic coupling is via β = β₀ exp(-H/H₀): as topography rises,
/// re-evaporation drops, trapping more moisture as precipitation.
pub fn lfpm_precipitation(
    grid: &mut TerrainGrid,
    params: &LfpmParams,
) {
    let rows = grid.rows;
    let cols = grid.cols;
    let dx = grid.dx;
    let dy = grid.dy;

    // For simplicity, assume advection along the x-axis (column direction).
    // The advection_direction parameter rotates the grid conceptually.
    // In a full implementation, you'd rotate the coordinate system.

    let mut fv = Array2::from_elem((rows, cols), params.fv0); // Vapor flux
    let mut fc = Array2::zeros((rows, cols));                   // Cloud water flux

    // Upwind sweep along advection axis (column by column, left to right)
    for c in 1..cols {
        // For each column, solve the implicit transversal dispersion system
        // This is a tridiagonal system in the row direction for each (Fv, Fc) pair

        let d_coeff = params.l_d / (dy * dy); // Dispersion coefficient

        for r in 0..rows {
            // Altitude-dependent re-evaporation
            let h = grid.elevation[[r, c]].max(0.0);
            let beta = params.beta_0 * (-h / params.h0).exp();

            // Upwind advection: d(Fv)/dx ≈ (Fv[c] - Fv[c-1]) / dx
            let fv_prev = fv[[r, c - 1]];
            let fc_prev: f64 = fc[[r, c - 1]];

            // Condensation/evaporation exchange term
            let exchange = (fv_prev - beta * fc_prev) / params.l_c;

            // Fallout rate
            let fallout = fc_prev / params.l_f;

            // Update vapor flux (upwind)
            fv[[r, c]] = fv_prev - dx * exchange;

            // Update cloud water flux (upwind + fallout)
            fc[[r, c]] = fc_prev + dx * (exchange - fallout);

            // Ensure non-negative
            fv[[r, c]] = fv[[r, c]].max(0.0);
            fc[[r, c]] = fc[[r, c]].max(0.0);
        }

        // Apply transversal dispersion (implicit tridiagonal in y-direction)
        // Thomas algorithm for both Fv and Fc
        if rows > 2 {
            solve_tridiagonal_dispersion_inplace(
                &mut fv, c, rows, d_coeff * dx, // effective dispersion per x-step
            );
            solve_tridiagonal_dispersion_inplace(
                &mut fc, c, rows, d_coeff * dx,
            );
        }
    }

    // Precipitation = fallout of cloud water: P = F_c / L_f
    for r in 0..rows {
        for c in 0..cols {
            let precip = fc[[r, c]] / params.l_f;
            grid.precipitation[[r, c]] = precip.max(0.0);
        }
    }
}

/// Thomas algorithm for implicit transversal dispersion on a single column.
/// Solves: -d*f[r-1] + (1+2d)*f[r] - d*f[r+1] = f[r] (current value)
fn solve_tridiagonal_dispersion_inplace(
    field: &mut Array2<f64>,
    col: usize,
    rows: usize,
    d: f64,
) {
    if d.abs() < 1e-30 {
        return; // No dispersion
    }

    let a = -d;           // sub-diagonal
    let b = 1.0 + 2.0 * d; // main diagonal
    let c_diag = -d;       // super-diagonal

    let mut cp = vec![0.0f64; rows];
    let mut dp = vec![0.0f64; rows];

    // Forward sweep
    cp[0] = c_diag / b;
    dp[0] = field[[0, col]] / b;

    for r in 1..rows {
        let m = b - a * cp[r - 1];
        if m.abs() < 1e-30 {
            continue;
        }
        cp[r] = if r < rows - 1 { c_diag / m } else { 0.0 };
        dp[r] = (field[[r, col]] - a * dp[r - 1]) / m;
    }

    // Back substitution
    field[[rows - 1, col]] = dp[rows - 1];
    for r in (0..rows - 1).rev() {
        field[[r, col]] = dp[r] - cp[r] * field[[r + 1, col]];
    }
}

// ============================================================================
// 3. ZERO-DIMENSIONAL ENERGY BALANCE MODEL (EBM)
// ============================================================================

/// Parameters for the simple EBM temperature/evapotranspiration model.
#[derive(Clone, Debug)]
pub struct EbmParams {
    /// Sea-level temperature T₀ [°C]
    pub t_sea_level: f64,
    /// Environmental lapse rate [°C/m]
    pub lapse_rate: f64,
    /// Top-of-atmosphere solar radiation Q [W/m²]
    pub solar_radiation: f64,
    /// Base albedo (snow-free)
    pub albedo_base: f64,
    /// Snow albedo
    pub albedo_snow: f64,
    /// Temperature threshold for snow [°C]
    pub snow_threshold: f64,
    /// Stefan-Boltzmann linearization coefficients: I_out = A + B*T
    pub sb_a: f64,
    pub sb_b: f64,
    /// PET Hargreaves coefficient [m/yr per °C]
    pub pet_coeff: f64,
}

impl Default for EbmParams {
    fn default() -> Self {
        Self {
            t_sea_level: 20.0,
            lapse_rate: 6.5e-3,
            solar_radiation: 342.0,
            albedo_base: 0.3,
            albedo_snow: 0.8,
            snow_threshold: 0.0,
            sb_a: 204.0,
            sb_b: 2.17,
            pet_coeff: 0.025,
        }
    }
}

/// Compute surface temperature and potential evapotranspiration.
///
/// Uses elevation-dependent lapse rate for temperature and Hargreaves-type PET.
/// O(N) pointwise calculation — no spatial coupling needed.
pub fn compute_temperature_pet(
    grid: &mut TerrainGrid,
    params: &EbmParams,
) {
    let rows = grid.rows;
    let cols = grid.cols;

    for r in 0..rows {
        for c in 0..cols {
            let h = grid.elevation[[r, c]].max(0.0);

            // Temperature via lapse rate
            let t = params.t_sea_level - params.lapse_rate * h;
            grid.temperature[[r, c]] = t;

            // Albedo feedback (ice/snow threshold)
            let albedo = if t < params.snow_threshold {
                params.albedo_snow
            } else {
                params.albedo_base
            };

            // Equilibrium temperature from EBM: Q(1-α) = A + B*T_eq
            // T_eq = (Q*(1-α) - A) / B
            let t_eq = (params.solar_radiation * (1.0 - albedo) - params.sb_a) / params.sb_b;

            // Blend: use lapse-rate T primarily, EBM adjusts for energy balance
            let t_final = 0.7 * t + 0.3 * t_eq; // Weighted blend
            grid.temperature[[r, c]] = t_final;

            // Potential evapotranspiration (Hargreaves-type, simplified)
            // PET increases with temperature, zero below freezing
            let pet = if t_final > 0.0 {
                params.pet_coeff * t_final
            } else {
                0.0
            };
            grid.evapotranspiration[[r, c]] = pet;
        }
    }
}

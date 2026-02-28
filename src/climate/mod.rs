//! Reduced-complexity climate models with corrected boundary treatments.

use ndarray::Array2;
use crate::grid::TerrainGrid;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct LinearTheoryParams {
    pub p0: f64,
    pub wind_speed: f64,
    pub wind_direction: f64,
    pub lapse_rate: f64,
    pub rho_sref: f64,
    pub hw: f64,
    pub tau_c: f64,
    pub tau_f: f64,
    pub nm_sq: f64,
}

impl Default for LinearTheoryParams {
    fn default() -> Self {
        Self {
            p0: 1.0, wind_speed: 10.0, wind_direction: 0.0,
            lapse_rate: 6.5e-3, rho_sref: 7.5e-3, hw: 2500.0,
            tau_c: 1000.0, tau_f: 1000.0, nm_sq: 0.005 * 0.005,
        }
    }
}

pub fn linear_theory_precipitation(grid: &mut TerrainGrid, params: &LinearTheoryParams) {
    let rows = grid.rows;
    let cols = grid.cols;
    let dx = grid.dx;
    let dy = grid.dy;

    let u_wind = params.wind_speed * params.wind_direction.cos();
    let v_wind = params.wind_speed * params.wind_direction.sin();
    let c_w = params.rho_sref * params.lapse_rate;

    let pad_rows = rows.next_power_of_two();
    let pad_cols = cols.next_power_of_two();

    let mut planner = FftPlanner::<f64>::new();
    let fft_cols = planner.plan_fft_forward(pad_cols);
    let fft_rows = planner.plan_fft_forward(pad_rows);
    let ifft_cols = planner.plan_fft_inverse(pad_cols);
    let ifft_rows = planner.plan_fft_inverse(pad_rows);

    let mut field = vec![Complex::new(0.0, 0.0); pad_rows * pad_cols];
    for r in 0..rows {
        let wr = 0.5 * (1.0 - (2.0 * PI * r as f64 / rows as f64).cos());
        for c in 0..cols {
            let wc = 0.5 * (1.0 - (2.0 * PI * c as f64 / cols as f64).cos());
            field[r * pad_cols + c] = Complex::new(grid.elevation[[r, c]] * wr * wc, 0.0);
        }
    }

    let mut scratch = vec![Complex::new(0.0, 0.0); pad_rows.max(pad_cols)];
    for r in 0..pad_rows {
        let start = r * pad_cols;
        fft_cols.process_with_scratch(&mut field[start..start + pad_cols], &mut scratch[..pad_cols]);
    }

    let mut col_buf = vec![Complex::new(0.0, 0.0); pad_rows];
    for c in 0..pad_cols {
        for r in 0..pad_rows { col_buf[r] = field[r * pad_cols + c]; }
        fft_rows.process_with_scratch(&mut col_buf, &mut scratch[..pad_rows]);
        for r in 0..pad_rows { field[r * pad_cols + c] = col_buf[r]; }
    }

    let norm = 1.0 / (pad_rows * pad_cols) as f64;
    for kr in 0..pad_rows {
        for kc in 0..pad_cols {
            let k = if kc <= pad_cols / 2 {
                2.0 * PI * kc as f64 / (pad_cols as f64 * dx)
            } else {
                2.0 * PI * (kc as f64 - pad_cols as f64) / (pad_cols as f64 * dx)
            };
            let l = if kr <= pad_rows / 2 {
                2.0 * PI * kr as f64 / (pad_rows as f64 * dy)
            } else {
                2.0 * PI * (kr as f64 - pad_rows as f64) / (pad_rows as f64 * dy)
            };

            if kr == 0 && kc == 0 { field[0] = Complex::new(0.0, 0.0); continue; }

            let sigma = u_wind * k + v_wind * l;
            let k_mag_sq = k * k + l * l;
            let m_sq = if k_mag_sq > 1e-30 {
                (params.nm_sq - sigma * sigma) * k_mag_sq / (sigma * sigma + 1e-30)
            } else { 0.0 };

            let numerator = Complex::new(0.0, c_w * sigma * params.hw);
            let m_vert = if m_sq > 0.0 { m_sq.sqrt() } else { 0.0 };
            let d1 = Complex::new(1.0, -m_vert * params.hw);
            let d2 = Complex::new(1.0, sigma * params.tau_c);
            let d3 = Complex::new(1.0, sigma * params.tau_f);
            let denominator = d1 * d2 * d3;

            let denom_mag = denominator.norm_sqr();
            if denom_mag > 1e-30 {
                field[kr * pad_cols + kc] = field[kr * pad_cols + kc] * numerator / denominator;
            } else {
                field[kr * pad_cols + kc] = Complex::new(0.0, 0.0);
            }
        }
    }

    for c in 0..pad_cols {
        for r in 0..pad_rows { col_buf[r] = field[r * pad_cols + c]; }
        ifft_rows.process_with_scratch(&mut col_buf, &mut scratch[..pad_rows]);
        for r in 0..pad_rows { field[r * pad_cols + c] = col_buf[r]; }
    }

    for r in 0..rows {
        let start = r * cols;
        let row_slice = &mut field[start..start + cols];
        ifft_cols.process_with_scratch(row_slice, &mut scratch[..cols]);
    }

    for r in 0..rows {
        for c in 0..cols {
            let orog = field[r * pad_cols + c].re * norm;
            grid.precipitation[[r, c]] = (params.p0 + orog).max(0.0);
        }
    }
}

#[derive(Clone, Debug)]
pub struct LfpmParams {
    pub fv0: f64, pub l_c: f64, pub l_f: f64, pub l_d: f64,
    pub beta_0: f64, pub h0: f64, pub advection_direction: f64,
}

impl Default for LfpmParams {
    fn default() -> Self {
        Self {
            fv0: 500.0, l_c: 50_000.0, l_f: 5_000.0, l_d: 10_000.0,
            beta_0: 0.8, h0: 2500.0, advection_direction: 0.0,
        }
    }
}

pub fn lfpm_precipitation(grid: &mut TerrainGrid, params: &LfpmParams) {
    let rows = grid.rows;
    let cols = grid.cols;
    let dx = grid.dx;
    let dy = grid.dy;

    let cos_a = params.advection_direction.cos();
    let sin_a = params.advection_direction.sin();

    let mut fv = Array2::from_elem((rows, cols), params.fv0);
    let mut fc: Array2<f64> = Array2::zeros((rows, cols));

    let sweep_x = cos_a.abs() >= sin_a.abs();

    if sweep_x {
        let sign = if cos_a >= 0.0 { 1i32 } else { -1i32 };
        let col_order: Vec<usize> = if sign > 0 { (1..cols).collect() }
        else { (0..cols - 1).rev().collect() };
        let d_coeff = params.l_d / (dy * dy);

        for &c in &col_order {
            let c_prev = (c as i32 - sign) as usize;
            for r in 0..rows {
                let h = grid.elevation[[r, c]].max(0.0);
                let beta = params.beta_0 * (-h / params.h0).exp();
                let fv_prev = fv[[r, c_prev]];
                let fc_prev: f64 = fc[[r, c_prev]];
                let exchange = (fv_prev - beta * fc_prev) / params.l_c;
                let fallout = fc_prev / params.l_f;
                let cross_contrib = if sin_a.abs() > 0.01 && r > 0 && r < rows - 1 {
                    let fv_cross = if sin_a > 0.0 { fv[[r - 1, c]] } else { fv[[r + 1, c]] };
                    sin_a.abs() * (fv_cross - fv_prev) / dy * 0.1
                } else { 0.0 };
                fv[[r, c]] = (fv_prev - dx * exchange + cross_contrib).max(0.0);
                fc[[r, c]] = (fc_prev + dx * (exchange - fallout)).max(0.0);
            }
            if rows > 2 {
                solve_tridiagonal_dispersion_inplace(&mut fv, c, rows, d_coeff * dx);
                solve_tridiagonal_dispersion_inplace(&mut fc, c, rows, d_coeff * dx);
            }
        }
    } else {
        let sign = if sin_a >= 0.0 { 1i32 } else { -1i32 };
        let row_order: Vec<usize> = if sign > 0 { (1..rows).collect() }
        else { (0..rows - 1).rev().collect() };
        let d_coeff = params.l_d / (dx * dx);

        for &r in &row_order {
            let r_prev = (r as i32 - sign) as usize;
            for c in 0..cols {
                let h = grid.elevation[[r, c]].max(0.0);
                let beta = params.beta_0 * (-h / params.h0).exp();
                let fv_prev = fv[[r_prev, c]];
                let fc_prev: f64 = fc[[r_prev, c]];
                let exchange = (fv_prev - beta * fc_prev) / params.l_c;
                let fallout = fc_prev / params.l_f;
                fv[[r, c]] = (fv_prev - dy * exchange).max(0.0);
                fc[[r, c]] = (fc_prev + dy * (exchange - fallout)).max(0.0);
            }
            if cols > 2 {
                solve_tridiagonal_dispersion_row_inplace(&mut fv, r, cols, d_coeff * dy);
                solve_tridiagonal_dispersion_row_inplace(&mut fc, r, cols, d_coeff * dy);
            }
        }
    }

    for r in 0..rows {
        for c in 0..cols {
            grid.precipitation[[r, c]] = (fc[[r, c]] / params.l_f).max(0.0);
        }
    }
}

fn solve_tridiagonal_dispersion_inplace(field: &mut Array2<f64>, col: usize, rows: usize, d: f64) {
    if d.abs() < 1e-30 { return; }
    let a = -d; let b = 1.0 + 2.0 * d; let c_diag = -d;
    let mut cp = vec![0.0f64; rows];
    let mut dp = vec![0.0f64; rows];
    cp[0] = c_diag / b;
    dp[0] = field[[0, col]] / b;
    for r in 1..rows {
        let m = b - a * cp[r - 1];
        if m.abs() < 1e-30 { continue; }
        cp[r] = if r < rows - 1 { c_diag / m } else { 0.0 };
        dp[r] = (field[[r, col]] - a * dp[r - 1]) / m;
    }
    field[[rows - 1, col]] = dp[rows - 1];
    for r in (0..rows - 1).rev() { field[[r, col]] = dp[r] - cp[r] * field[[r + 1, col]]; }
}

fn solve_tridiagonal_dispersion_row_inplace(field: &mut Array2<f64>, row: usize, cols: usize, d: f64) {
    if d.abs() < 1e-30 { return; }
    let a = -d; let b = 1.0 + 2.0 * d; let c_diag = -d;
    let mut cp = vec![0.0f64; cols];
    let mut dp = vec![0.0f64; cols];
    cp[0] = c_diag / b;
    dp[0] = field[[row, 0]] / b;
    for c in 1..cols {
        let m = b - a * cp[c - 1];
        if m.abs() < 1e-30 { continue; }
        cp[c] = if c < cols - 1 { c_diag / m } else { 0.0 };
        dp[c] = (field[[row, c]] - a * dp[c - 1]) / m;
    }
    field[[row, cols - 1]] = dp[cols - 1];
    for c in (0..cols - 1).rev() { field[[row, c]] = dp[c] - cp[c] * field[[row, c + 1]]; }
}

#[derive(Clone, Debug)]
pub struct EbmParams {
    pub t_sea_level: f64, pub lapse_rate: f64, pub solar_radiation: f64,
    pub albedo_base: f64, pub albedo_snow: f64, pub snow_threshold: f64,
    pub sb_a: f64, pub sb_b: f64, pub pet_coeff: f64, pub max_iterations: usize,
}

impl Default for EbmParams {
    fn default() -> Self {
        Self {
            t_sea_level: 20.0, lapse_rate: 6.5e-3, solar_radiation: 342.0,
            albedo_base: 0.3, albedo_snow: 0.8, snow_threshold: 0.0,
            sb_a: 204.0, sb_b: 2.17, pet_coeff: 0.025, max_iterations: 5,
        }
    }
}

pub fn compute_temperature_pet(grid: &mut TerrainGrid, params: &EbmParams) {
    let rows = grid.rows;
    let cols = grid.cols;
    for r in 0..rows {
        for c in 0..cols {
            let h = grid.elevation[[r, c]].max(0.0);
            let mut t = params.t_sea_level - params.lapse_rate * h;
            let width = 2.0;
            for _ in 0..params.max_iterations {
                let sigmoid = 1.0 / (1.0 + (-(t - params.snow_threshold) / width).exp());
                let albedo = params.albedo_snow + (params.albedo_base - params.albedo_snow) * sigmoid;
                let f = params.solar_radiation * (1.0 - albedo) - params.sb_a - params.sb_b * (t + params.lapse_rate * h);
                let dsigmoid = sigmoid * (1.0 - sigmoid) / width;
                let dalbedo = (params.albedo_base - params.albedo_snow) * dsigmoid;
                let df = -params.solar_radiation * dalbedo - params.sb_b;
                if df.abs() < 1e-30 { break; }
                let dt_step = -f / df;
                t += dt_step;
                if dt_step.abs() < 0.01 { break; }
            }
            grid.temperature[[r, c]] = t;
            grid.evapotranspiration[[r, c]] = if t > 0.0 { params.pet_coeff * t } else { 0.0 };
        }
    }
}

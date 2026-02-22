pub mod constants {
    pub const SECONDS_PER_YEAR: f64 = 31_557_600.0;
    pub const DEFAULT_CELL_SIZE_M: f32 = 1_000.0;
    pub const EARTH_CONTINENTAL_AREA_M2: f64 = 1.489e14;
    pub const MEAN_EROSION_RATE: f32 = 0.0001;
    pub const DEFAULT_RAINFALL: f32 = 1.0;
}
#[inline] pub fn clamp(v: f32, lo: f32, hi: f32) -> f32 { v.max(lo).min(hi) }
#[inline] pub fn sigmoid(x: f32, alpha: f32, threshold: f32) -> f32 { 1.0 / (1.0 + (-alpha * (x - threshold)).exp()) }
#[inline] pub fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }
#[inline] pub fn m_yr_to_mm_yr(rate: f32) -> f32 { rate * 1000.0 }
pub fn laplacian_2d(h: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut lap = vec![0.0f32; h.len()];
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let i = y * width + x;
            lap[i] = h[i + 1] + h[i - 1] + h[i + width] + h[i - width] - 4.0 * h[i];
        }
    }
    lap
}
pub fn gradient_magnitude(h: &[f32], width: usize, height: usize, cell_size: f32) -> Vec<f32> {
    let mut grad = vec![0.0f32; h.len()];
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let i = y * width + x;
            let dhdx = (h[i + 1] - h[i - 1]) / (2.0 * cell_size);
            let dhdy = (h[i + width] - h[i - width]) / (2.0 * cell_size);
            grad[i] = (dhdx * dhdx + dhdy * dhdy).sqrt();
        }
    }
    grad
}

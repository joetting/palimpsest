// ============================================================================
// Heightmap Generation — procedural terrain initialization
//
// Uses fractal Brownian motion (fBm) to create geologically plausible
// initial conditions before FastScape begins carving the landscape.
// ============================================================================

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng as SmallRng;

pub struct Heightmap {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

impl Heightmap {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0.0; width * height],
            width,
            height,
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        self.data[y * self.width + x] = v;
    }

    /// Generate using fractional Brownian motion — the "Body without Organs"
    /// initial state before geological history begins carving it.
    ///
    /// amplitude: max height in meters
    /// octaves:   number of noise frequencies (higher = more detail)
    /// lacunarity: frequency multiplier per octave (typically 2.0)
    /// persistence: amplitude decay per octave (typically 0.5)
    pub fn fbm(
        width: usize,
        height: usize,
        amplitude: f32,
        octaves: u32,
        lacunarity: f32,
        persistence: f32,
        seed: u64,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut hm = Self::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let nx = x as f32 / width as f32;
                let ny = y as f32 / height as f32;
                let v = fbm_noise(nx, ny, octaves, lacunarity, persistence, &mut rng);
                hm.set(x, y, v * amplitude);
            }
        }

        // Normalize to [0, amplitude]
        let min = hm.data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = hm.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max - min).max(1e-6);
        for v in hm.data.iter_mut() {
            *v = (*v - min) / range * amplitude;
        }

        hm
    }

    /// Combine two heightmaps with a blend factor
    pub fn blend(&self, other: &Heightmap, t: f32) -> Heightmap {
        assert_eq!(self.data.len(), other.data.len());
        let data = self.data.iter().zip(other.data.iter())
            .map(|(a, b)| a * (1.0 - t) + b * t)
            .collect();
        Heightmap { data, width: self.width, height: self.height }
    }

    /// Apply a radial mask — useful for creating island-style terrains
    pub fn apply_island_mask(&mut self, sea_fraction: f32) {
        let cx = self.width as f32 / 2.0;
        let cy = self.height as f32 / 2.0;
        let max_r = (cx.min(cy)) * (1.0 - sea_fraction).sqrt();

        for y in 0..self.height {
            for x in 0..self.width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let r = (dx * dx + dy * dy).sqrt();
                let mask = (1.0 - (r / max_r).min(1.0)).max(0.0);
                let v = self.get(x, y);
                self.set(x, y, v * mask);
            }
        }
    }

    /// Statistics
    pub fn stats(&self) -> HeightmapStats {
        let min = self.data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = self.data.iter().sum::<f32>() / self.data.len() as f32;
        let variance = self.data.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
            / self.data.len() as f32;
        HeightmapStats { min, max, mean, std_dev: variance.sqrt() }
    }
}

pub struct HeightmapStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std_dev: f32,
}

impl std::fmt::Display for HeightmapStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "min={:.1}m  max={:.1}m  mean={:.1}m  σ={:.1}m",
            self.min, self.max, self.mean, self.std_dev
        )
    }
}

// ---------------------------------------------------------------------------
// Internal noise functions (value noise + fBm)
// ---------------------------------------------------------------------------

fn smooth(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0) // Ken Perlin's quintic
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Simple lattice value noise (no external crate required for Phase 1)
fn value_noise(x: f32, y: f32, rng: &mut SmallRng) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = x - x.floor();
    let yf = y - y.floor();

    // Hash lattice corners deterministically from their coordinates
    let hash = |px: i32, py: i32| -> f32 {
        let mut h = (px.wrapping_mul(374761393)).wrapping_add(py.wrapping_mul(668265263));
        h = h.wrapping_add(h.wrapping_shr(13));
        h = h.wrapping_mul(1274126177i32);
        h = h.wrapping_add(h.wrapping_shr(16));
        (h as f32) / (i32::MAX as f32) * 0.5 + 0.5
    };

    let v00 = hash(xi,     yi    );
    let v10 = hash(xi + 1, yi    );
    let v01 = hash(xi,     yi + 1);
    let v11 = hash(xi + 1, yi + 1);

    let u = smooth(xf);
    let v = smooth(yf);

    lerp(lerp(v00, v10, u), lerp(v01, v11, u), v)
}

fn fbm_noise(x: f32, y: f32, octaves: u32, lacunarity: f32, persistence: f32, rng: &mut SmallRng) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = 1.0f32;
    let mut max_value = 0.0f32;

    for _ in 0..octaves {
        value += value_noise(x * frequency, y * frequency, rng) * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    value / max_value
}

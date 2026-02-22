// ============================================================================
// ECS Components — Phase 1: The "Body Without Organs" (Core Architecture)
//
// Each voxel *column* is an entity. Components represent either:
//   - Properties: actualized, directly observable values (density, material)
//   - Capacities: relational potentials triggered by interaction (erodibility)
// ============================================================================

use serde::{Deserialize, Serialize};

/// Unique entity identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity(pub u32);

// ---------------------------------------------------------------------------
// GRID POSITION
// ---------------------------------------------------------------------------

/// 2D grid position (column-based; Z is handled by LayeredColumn)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GridPos {
    pub x: u32,
    pub y: u32,
}

impl GridPos {
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }

    /// Flat index into a width×height grid
    pub fn flat_index(&self, width: u32) -> usize {
        (self.y * width + self.x) as usize
    }
}

// ---------------------------------------------------------------------------
// MATERIAL SYSTEM
// ---------------------------------------------------------------------------

/// Fundamental material types that voxels can be.
/// DeLanda: these are the "lavas and magmas" before stratification.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialId {
    Air          = 0,
    Bedrock      = 1,  // Granite/basalt — high resistance
    SoftRock     = 2,  // Limestone/sandstone — moderate resistance
    RegolithSand = 3,  // Loose sand/gravel — low resistance
    SoilA        = 4,  // Shallow A-horizon soil
    SoilB        = 5,  // B-horizon (illuviation)
    SoilC        = 6,  // C-horizon (weathered parent material)
    Water        = 7,
    Sediment     = 8,  // Transported and deposited material
    Organic      = 9,  // Accumulated organic matter
}

impl Default for MaterialId {
    fn default() -> Self {
        MaterialId::Air
    }
}

// ---------------------------------------------------------------------------
// INTRINSIC PROPERTIES (actualized values)
// ---------------------------------------------------------------------------

/// Physical density of the material in kg/m³
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Density(pub f32);

impl Default for Density {
    fn default() -> Self {
        Density(2700.0) // granite default
    }
}

impl Density {
    pub fn for_material(mat: MaterialId) -> Self {
        match mat {
            MaterialId::Bedrock      => Density(2700.0),
            MaterialId::SoftRock     => Density(2400.0),
            MaterialId::RegolithSand => Density(1600.0),
            MaterialId::SoilA        => Density(1200.0),
            MaterialId::SoilB        => Density(1400.0),
            MaterialId::SoilC        => Density(1600.0),
            MaterialId::Water        => Density(1000.0),
            MaterialId::Sediment     => Density(1500.0),
            MaterialId::Organic      => Density(800.0),
            MaterialId::Air          => Density(1.2),
        }
    }
}

// ---------------------------------------------------------------------------
// RELATIONAL CAPACITIES (triggered by interaction)
// ---------------------------------------------------------------------------

/// Erodibility Kf — capacity to be eroded by water [m^(1-2m) yr^-1]
/// This is the Kf in the Stream Power Law: ∂h/∂t = U - Kf·A^m·S^n + Kd·∇²h
/// DeLanda: a capacity that becomes actual only when water acts upon it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Erodibility(pub f32);

impl Erodibility {
    pub fn for_material(mat: MaterialId) -> Self {
        match mat {
            MaterialId::Bedrock      => Erodibility(1e-6),  // Very resistant
            MaterialId::SoftRock     => Erodibility(5e-5),  // Moderate
            MaterialId::RegolithSand => Erodibility(1e-3),  // Easily eroded
            MaterialId::SoilA        => Erodibility(5e-4),
            MaterialId::SoilB        => Erodibility(2e-4),
            MaterialId::SoilC        => Erodibility(3e-4),
            MaterialId::Sediment     => Erodibility(8e-4),
            _                        => Erodibility(0.0),
        }
    }
}

/// Hillslope diffusivity Kd — capacity for creep/diffusion [m² yr^-1]
/// Represents the rate at which elevation gradients smooth via soil creep.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Diffusivity(pub f32);

impl Diffusivity {
    pub fn for_material(mat: MaterialId) -> Self {
        match mat {
            MaterialId::Bedrock      => Diffusivity(0.001),
            MaterialId::SoftRock     => Diffusivity(0.01),
            MaterialId::RegolithSand => Diffusivity(0.05),
            MaterialId::SoilA        => Diffusivity(0.03),
            MaterialId::SoilB        => Diffusivity(0.02),
            MaterialId::SoilC        => Diffusivity(0.015),
            MaterialId::Sediment     => Diffusivity(0.04),
            _                        => Diffusivity(0.0),
        }
    }
}

// ---------------------------------------------------------------------------
// TERRAIN HEIGHT & LAYERED COLUMN
// ---------------------------------------------------------------------------

/// Topographic elevation of a column [meters above datum]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Elevation(pub f32);

/// Drainage area flowing through this column [m²]
/// Calculated by FastScape flow routing; proxy for local discharge.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DrainageArea(pub f32);

impl Default for DrainageArea {
    fn default() -> Self {
        DrainageArea(1.0) // minimum: each cell drains itself
    }
}

/// Tectonic uplift rate for this column [m yr^-1]
/// The player's primary Elder God intervention: U in the SPL.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UpliftRate(pub f32);

impl Default for UpliftRate {
    fn default() -> Self {
        UpliftRate(0.0)
    }
}

/// A vertical stack of material segments for 3D cave/overhang support.
/// Each column stores up to MAX_LAYERS distinct material layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayeredColumn {
    /// Bottom elevation of each layer segment [m]
    pub base: [f32; Self::MAX_LAYERS],
    /// Thickness of each layer [m]
    pub thickness: [f32; Self::MAX_LAYERS],
    /// Material of each layer
    pub material: [MaterialId; Self::MAX_LAYERS],
    /// How many layers are active
    pub count: usize,
}

impl LayeredColumn {
    pub const MAX_LAYERS: usize = 8;

    /// Create a column with a single layer of given material up to elevation h
    pub fn single(base: f32, thickness: f32, mat: MaterialId) -> Self {
        let mut col = Self {
            base: [0.0; Self::MAX_LAYERS],
            thickness: [0.0; Self::MAX_LAYERS],
            material: [MaterialId::Air; Self::MAX_LAYERS],
            count: 1,
        };
        col.base[0] = base;
        col.thickness[0] = thickness;
        col.material[0] = mat;
        col
    }

    /// Surface elevation = base + thickness of topmost active layer
    pub fn surface_elevation(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        let i = self.count - 1;
        self.base[i] + self.thickness[i]
    }

    /// Topmost material
    pub fn surface_material(&self) -> MaterialId {
        if self.count == 0 {
            return MaterialId::Air;
        }
        self.material[self.count - 1]
    }

    /// Remove thickness from the top of the column (erosion)
    pub fn erode_top(&mut self, amount: f32) -> f32 {
        let mut remaining = amount;
        while remaining > 0.0 && self.count > 0 {
            let i = self.count - 1;
            if self.thickness[i] > remaining {
                self.thickness[i] -= remaining;
                remaining = 0.0;
            } else {
                remaining -= self.thickness[i];
                self.count -= 1;
            }
        }
        amount - remaining // actually eroded
    }

    /// Deposit material on top of the column
    pub fn deposit_top(&mut self, thickness: f32, mat: MaterialId) {
        if self.count == 0 {
            self.base[0] = 0.0;
            self.thickness[0] = thickness;
            self.material[0] = mat;
            self.count = 1;
            return;
        }
        let top = self.count - 1;
        if self.material[top] == mat {
            // Merge with same material
            self.thickness[top] += thickness;
        } else if self.count < Self::MAX_LAYERS {
            let new_base = self.base[top] + self.thickness[top];
            let i = self.count;
            self.base[i] = new_base;
            self.thickness[i] = thickness;
            self.material[i] = mat;
            self.count += 1;
        } else {
            // Column full — merge into top layer (graceful degradation)
            self.thickness[top] += thickness;
        }
    }
}

// ---------------------------------------------------------------------------
// FLOW ROUTING
// ---------------------------------------------------------------------------

/// Index of the downstream (receiver) column for flow routing.
/// u32::MAX signals an outlet/boundary.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FlowReceiver(pub u32);

impl FlowReceiver {
    pub const OUTLET: u32 = u32::MAX;

    pub fn is_outlet(&self) -> bool {
        self.0 == Self::OUTLET
    }
}

/// Position in the topologically sorted stack order (FastScape).
/// Lower values = closer to outlets; higher = ridge nodes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StackOrder(pub u32);

// ---------------------------------------------------------------------------
// SPARSE COMPUTATION MASK
// ---------------------------------------------------------------------------

/// One bit per column stored in a packed u32 array.
/// Bit = 0 means this column is inactive (deep ocean, bare glacier, etc.)
/// and can be skipped during compute passes — critical for GPU efficiency.
pub struct ActivityMask {
    words: Vec<u32>,
    pub width: u32,
    pub height: u32,
}

impl ActivityMask {
    pub fn new(width: u32, height: u32) -> Self {
        let n_cols = (width * height) as usize;
        let n_words = (n_cols + 31) / 32;
        Self {
            words: vec![0u32; n_words],
            width,
            height,
        }
    }

    /// Mark all columns active by default
    pub fn fill_active(&mut self) {
        self.words.fill(u32::MAX);
    }

    pub fn set_active(&mut self, x: u32, y: u32, active: bool) {
        let bit = (y * self.width + x) as usize;
        let word = bit / 32;
        let shift = bit % 32;
        if active {
            self.words[word] |= 1 << shift;
        } else {
            self.words[word] &= !(1 << shift);
        }
    }

    pub fn is_active(&self, x: u32, y: u32) -> bool {
        let bit = (y * self.width + x) as usize;
        let word = bit / 32;
        let shift = bit % 32;
        (self.words[word] >> shift) & 1 == 1
    }

    pub fn active_count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Raw bitmask words for GPU upload
    pub fn as_words(&self) -> &[u32] {
        &self.words
    }
}

// ---------------------------------------------------------------------------
// TEMPORAL COHORT (for staggered updates)
// ---------------------------------------------------------------------------

/// Divides agents/columns into cohorts for staggered update scheduling.
/// Only 1/N cohorts update per frame, distributing CPU load.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UpdateCohortId(pub u8);

impl UpdateCohortId {
    pub fn should_update(&self, frame: u64, num_cohorts: u8) -> bool {
        (frame % num_cohorts as u64) == self.0 as u64
    }
}

// ---------------------------------------------------------------------------
// SEDIMENT FLUX
// ---------------------------------------------------------------------------

/// Outgoing sediment flux from this column [m³ yr^-1].
/// Accumulated during FastScape's downstream traversal.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SedimentFlux(pub f32);

// ---------------------------------------------------------------------------
// WATER FLOW
// ---------------------------------------------------------------------------

/// Volumetric water flow rate [m³ yr^-1]; computed from drainage area × rainfall
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct WaterFlow(pub f32);

// ---------------------------------------------------------------------------
// COMPONENT BUNDLES (convenience groupings)
// ---------------------------------------------------------------------------

/// Everything needed to represent a terrain column for Phase 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainColumn {
    pub pos:          GridPos,
    pub elevation:    Elevation,
    pub layers:       LayeredColumn,
    pub erodibility:  Erodibility,
    pub diffusivity:  Diffusivity,
    pub uplift:       UpliftRate,
    pub drainage:     DrainageArea,
    pub receiver:     FlowReceiver,
    pub stack_order:  StackOrder,
    pub sediment:     SedimentFlux,
    pub water:        WaterFlow,
    pub cohort:       UpdateCohortId,
}

impl TerrainColumn {
    pub fn new(pos: GridPos, base_elevation: f32, mat: MaterialId, cohort: u8) -> Self {
        let layers = LayeredColumn::single(0.0, base_elevation, mat);
        Self {
            pos,
            elevation:   Elevation(base_elevation),
            layers,
            erodibility: Erodibility::for_material(mat),
            diffusivity: Diffusivity::for_material(mat),
            uplift:      UpliftRate::default(),
            drainage:    DrainageArea::default(),
            receiver:    FlowReceiver(FlowReceiver::OUTLET),
            stack_order: StackOrder(0),
            sediment:    SedimentFlux::default(),
            water:       WaterFlow::default(),
            cohort:      UpdateCohortId(cohort),
        }
    }
}

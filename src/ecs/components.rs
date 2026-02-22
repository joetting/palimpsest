use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GridPos { pub x: u32, pub y: u32 }
impl GridPos {
    pub fn new(x: u32, y: u32) -> Self { Self { x, y } }
    pub fn flat_index(&self, width: u32) -> usize { (self.y * width + self.x) as usize }
}
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialId {
    Air=0,Bedrock=1,SoftRock=2,RegolithSand=3,SoilA=4,SoilB=5,SoilC=6,Water=7,Sediment=8,Organic=9,
}
impl Default for MaterialId { fn default() -> Self { MaterialId::Air } }
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Density(pub f32);
impl Default for Density { fn default() -> Self { Density(2700.0) } }
impl Density {
    pub fn for_material(mat: MaterialId) -> Self {
        match mat {
            MaterialId::Bedrock=>Density(2700.0),MaterialId::SoftRock=>Density(2400.0),
            MaterialId::RegolithSand=>Density(1600.0),MaterialId::SoilA=>Density(1200.0),
            MaterialId::SoilB=>Density(1400.0),MaterialId::SoilC=>Density(1600.0),
            MaterialId::Water=>Density(1000.0),MaterialId::Sediment=>Density(1500.0),
            MaterialId::Organic=>Density(800.0),MaterialId::Air=>Density(1.2),
        }
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Erodibility(pub f32);
impl Erodibility {
    pub fn for_material(mat: MaterialId) -> Self {
        match mat {
            MaterialId::Bedrock=>Erodibility(1e-6),MaterialId::SoftRock=>Erodibility(5e-5),
            MaterialId::RegolithSand=>Erodibility(1e-3),MaterialId::SoilA=>Erodibility(5e-4),
            MaterialId::SoilB=>Erodibility(2e-4),MaterialId::SoilC=>Erodibility(3e-4),
            MaterialId::Sediment=>Erodibility(8e-4),_=>Erodibility(0.0),
        }
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Diffusivity(pub f32);
impl Diffusivity {
    pub fn for_material(mat: MaterialId) -> Self {
        match mat {
            MaterialId::Bedrock=>Diffusivity(0.001),MaterialId::SoftRock=>Diffusivity(0.01),
            MaterialId::RegolithSand=>Diffusivity(0.05),MaterialId::SoilA=>Diffusivity(0.03),
            MaterialId::SoilB=>Diffusivity(0.02),MaterialId::SoilC=>Diffusivity(0.015),
            MaterialId::Sediment=>Diffusivity(0.04),_=>Diffusivity(0.0),
        }
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Elevation(pub f32);
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DrainageArea(pub f32);
impl Default for DrainageArea { fn default() -> Self { DrainageArea(1.0) } }
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UpliftRate(pub f32);
impl Default for UpliftRate { fn default() -> Self { UpliftRate(0.0) } }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayeredColumn {
    pub base: [f32; Self::MAX_LAYERS],
    pub thickness: [f32; Self::MAX_LAYERS],
    pub material: [MaterialId; Self::MAX_LAYERS],
    pub count: usize,
}
impl LayeredColumn {
    pub const MAX_LAYERS: usize = 8;
    pub fn single(base: f32, thickness: f32, mat: MaterialId) -> Self {
        let mut col = Self { base: [0.0; Self::MAX_LAYERS], thickness: [0.0; Self::MAX_LAYERS], material: [MaterialId::Air; Self::MAX_LAYERS], count: 1 };
        col.base[0]=base; col.thickness[0]=thickness; col.material[0]=mat; col
    }
    pub fn surface_elevation(&self) -> f32 {
        if self.count==0 { return 0.0; }
        let i=self.count-1; self.base[i]+self.thickness[i]
    }
    pub fn surface_material(&self) -> MaterialId {
        if self.count==0 { return MaterialId::Air; } self.material[self.count-1]
    }
    pub fn erode_top(&mut self, amount: f32) -> f32 {
        let mut remaining=amount;
        while remaining>0.0&&self.count>0 {
            let i=self.count-1;
            if self.thickness[i]>remaining { self.thickness[i]-=remaining; remaining=0.0; }
            else { remaining-=self.thickness[i]; self.count-=1; }
        }
        amount-remaining
    }
    pub fn deposit_top(&mut self, thickness: f32, mat: MaterialId) {
        if self.count==0 { self.base[0]=0.0; self.thickness[0]=thickness; self.material[0]=mat; self.count=1; return; }
        let top=self.count-1;
        if self.material[top]==mat { self.thickness[top]+=thickness; }
        else if self.count<Self::MAX_LAYERS {
            let new_base=self.base[top]+self.thickness[top];
            let i=self.count; self.base[i]=new_base; self.thickness[i]=thickness; self.material[i]=mat; self.count+=1;
        } else { self.thickness[top]+=thickness; }
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FlowReceiver(pub u32);
impl FlowReceiver {
    pub const OUTLET: u32 = u32::MAX;
    pub fn is_outlet(&self) -> bool { self.0==Self::OUTLET }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StackOrder(pub u32);
pub struct ActivityMask { words: Vec<u32>, pub width: u32, pub height: u32 }
impl ActivityMask {
    pub fn new(width: u32, height: u32) -> Self {
        let n_cols=(width*height) as usize; let n_words=(n_cols+31)/32;
        Self { words: vec![0u32;n_words], width, height }
    }
    pub fn fill_active(&mut self) { self.words.fill(u32::MAX); }
    pub fn set_active(&mut self, x: u32, y: u32, active: bool) {
        let bit=(y*self.width+x) as usize; let word=bit/32; let shift=bit%32;
        if active { self.words[word]|=1<<shift; } else { self.words[word]&=!(1<<shift); }
    }
    pub fn is_active(&self, x: u32, y: u32) -> bool {
        let bit=(y*self.width+x) as usize; let word=bit/32; let shift=bit%32;
        (self.words[word]>>shift)&1==1
    }
    pub fn active_count(&self) -> usize { self.words.iter().map(|w|w.count_ones() as usize).sum() }
    pub fn as_words(&self) -> &[u32] { &self.words }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UpdateCohortId(pub u8);
impl UpdateCohortId {
    pub fn should_update(&self, frame: u64, num_cohorts: u8) -> bool {
        (frame%num_cohorts as u64)==self.0 as u64
    }
}
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SedimentFlux(pub f32);
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct WaterFlow(pub f32);
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainColumn {
    pub pos: GridPos, pub elevation: Elevation, pub layers: LayeredColumn,
    pub erodibility: Erodibility, pub diffusivity: Diffusivity, pub uplift: UpliftRate,
    pub drainage: DrainageArea, pub receiver: FlowReceiver, pub stack_order: StackOrder,
    pub sediment: SedimentFlux, pub water: WaterFlow, pub cohort: UpdateCohortId,
}
impl TerrainColumn {
    pub fn new(pos: GridPos, base_elevation: f32, mat: MaterialId, cohort: u8) -> Self {
        let layers=LayeredColumn::single(0.0,base_elevation,mat);
        Self { pos, elevation: Elevation(base_elevation), layers,
            erodibility: Erodibility::for_material(mat), diffusivity: Diffusivity::for_material(mat),
            uplift: UpliftRate::default(), drainage: DrainageArea::default(),
            receiver: FlowReceiver(FlowReceiver::OUTLET), stack_order: StackOrder(0),
            sediment: SedimentFlux::default(), water: WaterFlow::default(), cohort: UpdateCohortId(cohort),
        }
    }
}

use super::components::{ActivityMask, Entity, GridPos, MaterialId, TerrainColumn, UpdateCohortId};
pub struct World {
    pub columns: Vec<TerrainColumn>,
    pub width: u32, pub height: u32,
    pub activity: ActivityMask,
    next_entity: u32,
}
impl World {
    pub fn new(width: u32, height: u32) -> Self {
        let mut activity=ActivityMask::new(width,height);
        activity.fill_active();
        Self { columns: Vec::with_capacity((width*height) as usize), width, height, activity, next_entity: 0 }
    }
    pub fn spawn(&mut self) -> Entity { let e=Entity(self.next_entity); self.next_entity+=1; e }
    pub fn initialize_grid<F,M>(&mut self, height_fn: F, mat_fn: M, num_cohorts: u8)
    where F: Fn(u32,u32)->f32, M: Fn(f32)->MaterialId {
        let (w,h)=(self.width,self.height);
        self.columns.clear(); self.columns.reserve((w*h) as usize);
        for y in 0..h { for x in 0..w {
            let elev=height_fn(x,y); let mat=mat_fn(elev);
            let cohort=((y*w+x)%num_cohorts as u32) as u8;
            self.columns.push(TerrainColumn::new(GridPos::new(x,y),elev,mat,cohort));
        }}
    }
    #[inline] pub fn column(&self,x:u32,y:u32) -> &TerrainColumn { &self.columns[(y*self.width+x) as usize] }
    #[inline] pub fn column_mut(&mut self,x:u32,y:u32) -> &mut TerrainColumn { &mut self.columns[(y*self.width+x) as usize] }
    #[inline] pub fn column_by_index(&self,idx:usize) -> &TerrainColumn { &self.columns[idx] }
    #[inline] pub fn column_by_index_mut(&mut self,idx:usize) -> &mut TerrainColumn { &mut self.columns[idx] }
    pub fn active_count(&self) -> usize { self.activity.active_count() }
    pub fn total_columns(&self) -> usize { self.columns.len() }
    pub fn elevation_array(&self) -> Vec<f32> { self.columns.iter().map(|c|c.elevation.0).collect() }
    pub fn erodibility_array(&self) -> Vec<f32> { self.columns.iter().map(|c|c.erodibility.0).collect() }
    pub fn drainage_area_array(&self) -> Vec<f32> { self.columns.iter().map(|c|c.drainage.0).collect() }
    pub fn apply_elevation_deltas(&mut self, deltas: &[f32]) {
        assert_eq!(deltas.len(),self.columns.len());
        for (col,&dh) in self.columns.iter_mut().zip(deltas.iter()) {
            col.elevation.0+=dh;
            if col.elevation.0<0.0 { col.elevation.0=0.0; }
            let e=col.elevation.0;
            if col.layers.count>0 {
                let top=col.layers.count-1; let base=col.layers.base[top];
                col.layers.thickness[top]=(e-base).max(0.0);
            }
        }
    }
    pub fn columns_with_index(&self) -> impl Iterator<Item=(usize,&TerrainColumn)> { self.columns.iter().enumerate() }
    pub fn neighbors_4(&self,x:u32,y:u32) -> [Option<(u32,u32)>;4] {
        let (w,h)=(self.width,self.height);
        [ if y>0{Some((x,y-1))}else{None}, if y<h-1{Some((x,y+1))}else{None},
          if x>0{Some((x-1,y))}else{None}, if x<w-1{Some((x+1,y))}else{None} ]
    }
    pub fn neighbors_8(&self,x:u32,y:u32) -> Vec<(u32,u32)> {
        let (w,h)=(self.width as i32,self.height as i32);
        let (xi,yi)=(x as i32,y as i32); let mut result=Vec::with_capacity(8);
        for dy in -1i32..=1 { for dx in -1i32..=1 {
            if dx==0&&dy==0 { continue; }
            let (nx,ny)=(xi+dx,yi+dy);
            if nx>=0&&nx<w&&ny>=0&&ny<h { result.push((nx as u32,ny as u32)); }
        }}
        result
    }
    pub fn slope_to(&self,x:u32,y:u32,nx:u32,ny:u32,cell_size:f32) -> f32 {
        let dh=self.column(x,y).elevation.0-self.column(nx,ny).elevation.0;
        let dist=if x!=nx&&y!=ny{cell_size*std::f32::consts::SQRT_2}else{cell_size};
        dh/dist
    }
}

use rayon::prelude::*;
#[derive(Debug, Clone)]
pub struct WindParams {
    pub angle_rad:f32, pub incoming_moisture_m:f32,
    pub alpha_condensation:f32, pub beta_fallout_per_km:f32, pub background_precip_m:f32,
}
impl Default for WindParams {
    fn default() -> Self {
        Self { angle_rad:std::f32::consts::PI,incoming_moisture_m:2.0,
            alpha_condensation:0.005,beta_fallout_per_km:0.15,background_precip_m:0.05 }
    }
}
impl WindParams {
    pub fn wind_dir(&self) -> (f32,f32) { (self.angle_rad.cos(),self.angle_rad.sin()) }
}
pub struct OrographicOutput {
    pub precipitation:Vec<f32>, pub vapor:Vec<f32>, pub cloud_water:Vec<f32>,
    pub grid_width:usize, pub grid_height:usize,
}
impl OrographicOutput {
    pub fn new(width:usize,height:usize) -> Self {
        let n=width*height;
        Self { precipitation:vec![0.0;n],vapor:vec![0.0;n],cloud_water:vec![0.0;n],grid_width:width,grid_height:height }
    }
    pub fn mean_precipitation(&self) -> f32 { self.precipitation.iter().sum::<f32>()/self.precipitation.len() as f32 }
    pub fn max_precipitation(&self) -> f32 { self.precipitation.iter().cloned().fold(0.0f32,f32::max) }
}
pub struct OrographicEngine { pub grid_width:usize,pub grid_height:usize,pub wind:WindParams,pub cell_size_m:f32 }
impl OrographicEngine {
    pub fn new(width:usize,height:usize,wind:WindParams,cell_size_m:f32) -> Self {
        Self { grid_width:width,grid_height:height,wind,cell_size_m }
    }
    pub fn compute(&self,elevations:&[f32]) -> OrographicOutput {
        let mut output=OrographicOutput::new(self.grid_width,self.grid_height);
        let w=self.grid_width; let h=self.grid_height;
        let (wdx,wdy)=self.wind.wind_dir();
        let alpha=self.wind.alpha_condensation; let beta=self.wind.beta_fallout_per_km;
        let qv0=self.wind.incoming_moisture_m; let background=self.wind.background_precip_m;
        let cell_km=self.cell_size_m/1000.0;
        let mut contrib_count=vec![0u32;w*h];
        if wdx.abs()>0.05 {
            let ray_results:Vec<(Vec<usize>,Vec<f32>,Vec<f32>,Vec<f32>)>=(0..h).into_par_iter().map(|row_y|{
                let start_x:i32=if wdx>=0.0{0}else{(w as i32)-1};
                march_ray(start_x as f32,row_y as f32,wdx,wdy,w,h,elevations,qv0,alpha,beta,cell_km,background)
            }).collect();
            for (indices,precips,vapors,clouds) in ray_results {
                for (k,&idx) in indices.iter().enumerate() {
                    output.precipitation[idx]+=precips[k]; output.vapor[idx]+=vapors[k];
                    output.cloud_water[idx]+=clouds[k]; contrib_count[idx]+=1;
                }
            }
        }
        if wdy.abs()>0.05 {
            let ray_results:Vec<(Vec<usize>,Vec<f32>,Vec<f32>,Vec<f32>)>=(0..w).into_par_iter().map(|col_x|{
                let start_y:i32=if wdy>=0.0{0}else{(h as i32)-1};
                march_ray(col_x as f32,start_y as f32,wdx,wdy,w,h,elevations,qv0,alpha,beta,cell_km,background)
            }).collect();
            for (indices,precips,vapors,clouds) in ray_results {
                for (k,&idx) in indices.iter().enumerate() {
                    output.precipitation[idx]+=precips[k]; output.vapor[idx]+=vapors[k];
                    output.cloud_water[idx]+=clouds[k]; contrib_count[idx]+=1;
                }
            }
        }
        for idx in 0..(w*h) {
            let c=contrib_count[idx].max(1) as f32;
            output.precipitation[idx]/=c; output.vapor[idx]/=c; output.cloud_water[idx]/=c;
            output.precipitation[idx]=output.precipitation[idx].max(background);
        }
        output
    }
    pub fn set_wind(&mut self,wind:WindParams) { self.wind=wind; }
    pub fn rotate_wind(&mut self,delta_rad:f32) {
        self.wind.angle_rad=(self.wind.angle_rad+delta_rad).rem_euclid(2.0*std::f32::consts::PI);
    }
}
fn march_ray(x0:f32,y0:f32,wdx:f32,wdy:f32,width:usize,height:usize,elevations:&[f32],
    qv0:f32,alpha:f32,beta:f32,cell_km:f32,background:f32) -> (Vec<usize>,Vec<f32>,Vec<f32>,Vec<f32>) {
    let mut indices=Vec::new(); let mut precips=Vec::new();
    let mut vapors=Vec::new(); let mut clouds=Vec::new();
    let mut qv=qv0; let mut qc=0.0f32; let mut prev_elev=0.0f32;
    let mut x=x0; let mut y=y0;
    let step_scale=1.0/wdx.abs().max(wdy.abs()).max(1e-6);
    let sx=wdx*step_scale; let sy=wdy*step_scale;
    let max_steps=(width+height)*2;
    for _ in 0..max_steps {
        let ix=x.round() as i32; let iy=y.round() as i32;
        if ix<0||iy<0||ix>=width as i32||iy>=height as i32 { break; }
        let idx=iy as usize*width+ix as usize;
        let elev=elevations[idx];
        let dh=(elev-prev_elev).max(0.0);
        let condensation=(alpha*qv*dh).min(qv);
        qv=(qv-condensation).max(0.0); qc+=condensation;
        let fallout=(beta*cell_km*qc).min(qc);
        qc=(qc-fallout).max(0.0);
        let precip=fallout.max(background);
        indices.push(idx); precips.push(precip); vapors.push(qv); clouds.push(qc);
        prev_elev=elev; x+=sx; y+=sy;
    }
    (indices,precips,vapors,clouds)
}

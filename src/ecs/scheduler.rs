#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScheduleLabel { SocialMicroTick, GeologicalEpoch, BiologicalCycle, StrangStep }
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    pub social_time_scale: f64, pub geo_dt_years: f64, pub bio_dt_years: f64,
    pub social_ticks_per_geo_epoch: u64, pub social_cohorts: u8,
}
impl Default for TemporalConfig {
    fn default() -> Self {
        Self { social_time_scale:1.0,geo_dt_years:100.0,bio_dt_years:1.0,social_ticks_per_geo_epoch:1000,social_cohorts:10 }
    }
}
#[derive(Debug, Clone)]
pub struct SimulationClock {
    pub social_elapsed_s: f64, pub geo_elapsed_years: f64, pub bio_elapsed_years: f64,
    pub social_tick: u64, pub geo_epoch: u64, pub bio_cycle: u64,
    social_tick_accumulator: u64, social_tick_bio_accumulator: u64,
    pub config: TemporalConfig,
}
impl SimulationClock {
    pub fn new(config: TemporalConfig) -> Self {
        Self { social_elapsed_s:0.0,geo_elapsed_years:0.0,bio_elapsed_years:0.0,
            social_tick:0,geo_epoch:0,bio_cycle:0,social_tick_accumulator:0,
            social_tick_bio_accumulator:0,config }
    }
    pub fn advance_social_tick(&mut self, dt_s: f64) -> TickResult {
        self.social_elapsed_s+=dt_s*self.config.social_time_scale;
        self.social_tick+=1; self.social_tick_accumulator+=1; self.social_tick_bio_accumulator+=1;
        let fire_geo=self.social_tick_accumulator>=self.config.social_ticks_per_geo_epoch;
        if fire_geo { self.social_tick_accumulator=0; self.geo_elapsed_years+=self.config.geo_dt_years; self.geo_epoch+=1; }
        let ticks_per_bio=((self.config.social_ticks_per_geo_epoch as f64/(self.config.geo_dt_years/self.config.bio_dt_years)) as u64).max(1);
        let fire_bio=self.social_tick_bio_accumulator>=ticks_per_bio;
        if fire_bio { self.social_tick_bio_accumulator=0; self.bio_elapsed_years+=self.config.bio_dt_years; self.bio_cycle+=1; }
        TickResult { fire_geo, fire_bio }
    }
    pub fn geo_dt(&self) -> f32 { self.config.geo_dt_years as f32 }
    pub fn bio_dt(&self) -> f32 { self.config.bio_dt_years as f32 }
    pub fn active_cohort(&self) -> u8 { (self.social_tick%self.config.social_cohorts as u64) as u8 }
    pub fn summary(&self) -> String {
        let geo_ky=self.geo_elapsed_years/1000.0;
        let geo_my=self.geo_elapsed_years/1_000_000.0;
        if geo_my>=0.1 { format!("Epoch {} | {:.2} Ma | Social tick {} | Bio cycle {}",self.geo_epoch,geo_my,self.social_tick,self.bio_cycle) }
        else { format!("Epoch {} | {:.1} ka | Social tick {} | Bio cycle {}",self.geo_epoch,geo_ky,self.social_tick,self.bio_cycle) }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct TickResult { pub fire_geo: bool, pub fire_bio: bool }
pub type SystemFn<State> = Box<dyn Fn(&mut State) + Send + Sync>;
pub struct Schedule<State> { pub label: ScheduleLabel, systems: Vec<SystemFn<State>> }
impl<State> Schedule<State> {
    pub fn new(label: ScheduleLabel) -> Self { Self { label, systems: Vec::new() } }
    pub fn add_system<F: Fn(&mut State)+Send+Sync+'static>(&mut self, f: F) -> &mut Self { self.systems.push(Box::new(f)); self }
    pub fn run(&self, state: &mut State) { for system in &self.systems { system(state); } }
}
pub struct StrangRunner<State> {
    pub social_schedule: Schedule<State>, pub geo_schedule: Schedule<State>,
    pub bio_schedule: Schedule<State>, pub clock: SimulationClock,
}
impl<State> StrangRunner<State> {
    pub fn new(config: TemporalConfig) -> Self {
        Self { social_schedule:Schedule::new(ScheduleLabel::SocialMicroTick),
            geo_schedule:Schedule::new(ScheduleLabel::GeologicalEpoch),
            bio_schedule:Schedule::new(ScheduleLabel::BiologicalCycle),
            clock:SimulationClock::new(config) }
    }
    pub fn step(&mut self, state: &mut State, dt_s: f64) {
        self.social_schedule.run(state);
        let result=self.clock.advance_social_tick(dt_s);
        if result.fire_geo { self.geo_schedule.run(state); }
        if result.fire_bio { self.bio_schedule.run(state); }
        self.social_schedule.run(state);
    }
    pub fn clock(&self) -> &SimulationClock { &self.clock }
}

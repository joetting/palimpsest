/// Minimal ECS component stubs required by solver code.
pub mod components {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    #[repr(u16)]
    pub enum MaterialId {
        #[default]
        Air = 0,
        Bedrock = 1,
        Soil = 2,
        Rock = 3,
        Water = 4,
        Sediment = 5,
        Sand = 6,
        Clay = 7,
        Humus = 8,
        Road = 9,
    }
}

// ============================================================================
// ECS Query Helpers — Parallel iteration over component slices
// ============================================================================

use super::components::UpdateCohortId;
use super::world::World;

/// Returns the flat indices of all active columns for a given cohort
pub fn cohort_indices(world: &World, active_cohort: u8) -> Vec<usize> {
    world
        .columns_with_index()
        .filter(|(_, col)| {
            world.activity.is_active(col.pos.x, col.pos.y)
                && col.cohort.0 == active_cohort
        })
        .map(|(i, _)| i)
        .collect()
}

/// Returns ALL active column indices regardless of cohort
pub fn all_active_indices(world: &World) -> Vec<usize> {
    world
        .columns_with_index()
        .filter(|(_, col)| world.activity.is_active(col.pos.x, col.pos.y))
        .map(|(i, _)| i)
        .collect()
}

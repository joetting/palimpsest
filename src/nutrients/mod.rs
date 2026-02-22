// ============================================================================
// Nutrients — Phase 2: Critical Zone
//
// Coupled P-K biogeochemistry and Phillips pedogenesis.
//
// Modules:
//   pools       — 9-pool NutrientLayer / NutrientColumn data structures
//   solver      — Strang-split ODE solver (fast analytical + slow Euler)
//   pedogenesis — Phillips dS/dt = dP/dt - dR/dt with K_f/K_d feedback
// ============================================================================

pub mod pools;
pub mod solver;
pub mod pedogenesis;

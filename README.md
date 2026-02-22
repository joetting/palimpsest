# Deep Time Engine — Phase 1: The Body Without Organs

A materialist voxel simulation engine in Rust, grounded in Manuel DeLanda's
*A Thousand Years of Nonlinear History* and the FastScape landscape evolution
algorithm.

## What's Built

### Architecture (all compiles and runs)

```
src/
├── ecs/
│   ├── components.rs   — ECS components: properties vs. relational capacities
│   │                     MaterialId, Density, Erodibility, Diffusivity,
│   │                     LayeredColumn (8-layer 3D cave support),
│   │                     ActivityMask (bitmask for GPU sparse skipping),
│   │                     UpdateCohortId (staggered update cohorts)
│   ├── world.rs        — Flat SoA entity storage (GPU-ready layout)
│   │                     128×128 = 16,384 column entities
│   ├── scheduler.rs    — Strang operator splitting multirate scheduler
│   │                     Social | Geological | Biological schedules
│   │                     SimulationClock with accumulator-triggered epoch ticks
│   └── query.rs        — Cohort-filtered parallel query helpers
│
├── terrain/
│   ├── fastscape.rs    — O(N) implicit FastScape SPL solver
│   │                     ∂h/∂t = U − Kf·A^m·S^n + Kd·∇²h
│   │                     D8 flow routing → topological stack sort →
│   │                     drainage accumulation → implicit channel incision
│   │                     (Newton-Raphson for n≠1) → explicit hillslope diffusion
│   │                     TectonicForcing with Gaussian uplift hotspots
│   ├── heightmap.rs    — fBm fractal terrain generation (7 octaves)
│   │                     Island mask for natural drainage boundaries
│   └── svo.rs          — Sparse Voxel Octree interface (dense fallback Phase 1)
│
├── compute/
│   ├── gpu_stub.rs     — TerrainBuffers in SoA layout (vec4 packing for GPU)
│   │                     WGSL compute shader templates (terrain_update.wgsl)
│   │                     rayon parallel CPU fallback matching dispatch pattern
│   │                     Activity bitmask for sparse compute skipping
│   └── activity_mask.rs — ActivityProcessor with 32-bit packed bitmask words
│
├── math/mod.rs         — Geological constants, Laplacian, gradient utilities
└── config/mod.rs       — DeepTimeSimulation top-level orchestrator
```

## Sample Output

```
Initial heightmap: min=0.0m  max=1327.2m  mean=252.0m  σ=318.0m

[PHASE 1] ECS World initialized
  Grid: 128x128 columns (16384 entities)
  Cell size: 1 km | Domain: 128×128 km

[PHASE 1] Component system demo:
  ( 64, 64): h=1327.2m  mat=Bedrock  Kf=1.00e-6  Kd=0.001  cohort=6
  ( 32, 32): h= 311.3m  mat=SoftRock Kf=5.00e-5  Kd=0.010  cohort=8

[PHASE 1] Activity bitmask: 33.4% of ocean columns skippable

[ELDER GOD] Uplift hotspot added at (64,64) r=25km @2.00mm/yr

  Epoch #1  |  0.1 ka  | MaxH=1327.4m | MeanH=251.2m
  Epoch #20 |  2.0 ka  | MaxH=1331.3m | MeanH=239.3m

[ELDER GOD] Triggering major orogenic event...
  Uplift hotspot added at (32,96) r=40km @5.00mm/yr

  Epoch #40 |  4.0 ka  | MaxH=1335.4m | MeanH=230.6m

[x] ECS World         16384 entities, SoA layout ready for GPU
[x] FastScape Solver  D8 routing → stack sort → implicit SPL
[x] Strang Splitting  Social|Geo|Bio multirate schedules
[x] Activity Bitmask  33.4% columns skippable
[x] Cohort Spreading  10 cohorts, ~10% CPU load/frame
[x] Tectonic Forcing  2 active hotspots
[x] GPU Pipeline      WGSL shaders templated, SoA layout correct
```

## Key Design Decisions

**Why SoA layout?**  
All elevations contiguous, all erodibilities contiguous — matches WGPU
storage buffer access patterns. `vec4<f32>` packing gives 3–5× GPU speedup
vs. AoS when adjacent threads read the same field.

**Why implicit SPL?**  
Explicit solvers require Δt < cell/(Kf·A^m·S^(n-1)) — at 100 yr epochs
and geological Kf values this CFL condition forces impossibly small steps.
FastScape's implicit scheme is unconditionally stable; Δt = 100,000 yr
is numerically safe.

**Why Strang splitting?**  
Social ticks (seconds) vs. geo epochs (100 yr) gives stiffness ratio κ ≈ 10⁹.
Strang gives O(Δt²) accuracy and time-reversibility, essential for million-
iteration geological runs. Stiff operators (geo) go last in each Strang step.

**Why cohort spreading?**  
16,384 entities × 60 fps = 983,040 updates/sec. With 10 cohorts: 98,304/sec.
Agent behavioral inertia means 6 Hz per-agent update captures macro-level
Karatani exchange mode emergent behavior.

## Running

```bash
cargo run --release
```

## Phase Roadmap

- **Phase 1 ✓** — Body Without Organs: ECS, FastScape, Strang, bitmask
- **Phase 2** — Critical Zone: P-K biogeochemistry (9-pool ODE, Buendía model)
- **Phase 3** — Flesh & Genes: BDI agents, Lotka-Volterra, faunalturbation
- **Phase 4** — WGPU: real compute shaders replacing CPU stubs
- **Phase 5** — Urban Exoskeleton: Karatani modes, stigmergic mineralization
- **Phase 6** — Stratometer UI: Kolmogorov entropy overlay, Elder God tools

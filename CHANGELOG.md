# Changelog

## 0.4.0 (Unreleased)

### Added

- Full Torch Ewald/PME APIs support energy-derived forces, charge
  gradients, and strain-first virials, including second-order force/stress
  losses.
- 2D slab (Yeh-Berkowitz) correction for Ewald and PME summation, exposed as
  `compute_slab_correction` and a `slab_correction=` keyword on the Ewald/PME
  entry points, with both Torch and JAX bindings.
- Torch slab correction participates in autograd when inputs require
  gradients.
- Full JAX Ewald/PME energy-only calls support first-order gradients for
  positions, charges, and row-vector displacement virials.
- JAX PME reciprocal higher-order support is limited to tested position and
  charge scalar losses. PME cell/stress/strain higher-order derivatives remain
  unsupported.
- Torch Ewald accepts `miller_bounds` for k-vector generation.
- Torch/JAX PME accept precomputed `cell_inv_t`, `volume`, and B-spline
  moduli where supported.
- `compute_bspline_moduli_1d` is exported from the top-level Torch and JAX
  electrostatics namespaces for PME precompute workflows.
- Electrostatics autograd documents `positions`, `charges`, and `cell` as
  the only gradient targets. Setup values such as `alpha` are constants, and
  cell-derived reciprocal caches are static metadata assumed to correspond to
  the current cell.
- Higher-order electrostatics support is exposed through framework autograd on
  scalar losses; no public Hessian or Jacobian tensor/function APIs were added.

### Fixed

- Naive PBC neighbor wrapping now leaves non-periodic axes unwrapped when
  per-axis `pbc` flags are supplied, fixing partial-PBC and non-periodic
  systems (#104).
- Fixed Torch Ewald gradients for non-uniform per-atom energy cotangents
  (`torch.autograd.grad(..., grad_outputs=w)`).
- JAX electrostatics no longer import the removed `jax.custom_transpose`. The
  Ewald/PME real- and reciprocal-space and slab HVP transpose rules are
  migrated to `jax.custom_vjp` (a stable API), restoring importability on
  current JAX (0.10+) while preserving the second-order (force/stress-loss)
  derivatives.
- Coupled the FIRE2 variable-cell updates so positions and cell degrees of
  freedom advance consistently during constrained/variable-cell relaxation.
- Neighbor-list launchers now reject unbatched methods when batch metadata is
  supplied, instead of silently producing incorrect lists.
- **MTK NPT/NPH cell propagation**: kernels wrote `V·(P − P_ext)/W`
  (strain-rate units) into `cell_velocity` while consumers read it as
  `ḣ = dh/dt`, costing a factor of cell length in the cell response.
  `cell_velocity` is now the strain rate `ε̇ = p_g/W` everywhere and
  the cell update is `h_new = h + dt · ε̇ · h`.
- **MTK velocity-half-step coupling**: isotropic kernels used
  `α = 1 + 1/(3N_atoms)` instead of the canonical
  `α = 1 + 1/N_atoms` (ASE `IsotropicMTKNPT._integrate_p`).
  Anisotropic and triclinic kernels used `(1 + 1/N_atoms)·ε̇`,
  which only matches ASE `MTKNPT._integrate_p` for uniform strain;
  replaced with `ε̇ + Tr(ε̇)/(3·N)·I` (canonical trace correction).
- **MTK barostat half-step thermostat coupling**: NPT
  cell-velocity-update kernels applied `−η̇₁·ε̇` inline, mixing the
  pressure/kinetic driving operator with NHC drag. Removed; callers
  apply barostat-NHC coupling separately, matching ASE and TorchSim.

### Deprecated

- Direct-output flags on full Torch and JAX Ewald/PME APIs are deprecated for
  differentiable training: `compute_forces`, `compute_virial`,
  `compute_charge_gradients`, and `hybrid_forces`. They remain available and keep
  the existing tuple order. Component `compute_forces=True` remains available for
  no-autograd MD/inference use; component charge-gradient, virial, and hybrid
  direct outputs warn as legacy training-style outputs.
- `nvalchemiops.neighbors.zero_array` now emits a `DeprecationWarning` and
  forwards to `array.zero_()`. Call `array.zero_()` directly.
- `cells_inv` argument on `compute_cell_kinetic_energy`,
  `npt_velocity_half_step{,_out}`, `npt_position_update{,_out}`,
  `nph_velocity_half_step{,_out}`, `nph_position_update{,_out}`,
  `run_npt_step`, and `run_nph_step`. Kernels consume
  `cell_velocities` directly as the strain rate `ε̇ = p_g/W`. Passing
  `cells_inv` emits a `DeprecationWarning`; the argument will be
  removed in a future release.
- `volumes` argument on `compute_cell_kinetic_energy`,
  `npt_velocity_half_step{,_out}`, and `nph_velocity_half_step{,_out}`.
  Kernels consume `cell_velocities` directly as the strain rate and
  no longer need a volume fallback. Passing `volumes` emits a
  `DeprecationWarning`; the argument will be removed in a future release.

### Breaking Changes

- `cell_velocities` now stores the strain rate `ε̇ = p_g/W`, not
  `ḣ = dh/dt`. Kernel signatures unchanged.
- `npt_barostat_half_step{,_aniso,_triclinic}` drop the `eta_dots`
  argument; thermostat coupling is now a separate Trotter operator.
- The internal `make_outer_neigh_offsets` helper was removed.

### Added (neighbors)

- **Pair potentials evaluated inline**: neighbor kernels now accept a
  user-supplied `pair_fn` callback (with `pair_params`, `pair_energies`,
  `pair_forces` buffers) that computes per-pair energy and force as pairs
  are enumerated, so Lennard-Jones–style potentials no longer require a
  separate pass over the neighbor list.
- **Per-pair vectors and distances on demand**: `return_vectors` and
  `return_distances` keyword arguments return the separation vectors
  `r_ij` and Euclidean distances `|r_ij|` alongside the neighbor matrix,
  avoiding a manual recomputation downstream.
- **Cluster-pair tile algorithm**: a new CUDA strategy for large
  fully-periodic float32 systems. `neighbor_list` auto-selects it when
  it is eligible; pass `method="cluster_tile"` (or
  `"batch_cluster_tile"`) to force it. Supports dual cutoff in
  matrix format.
- **Partial rebuild for batched workflows**: callers can pass
  `rebuild_flags` to re-enumerate only the systems whose atoms have
  moved enough to need a fresh list; unchanged systems keep their
  previous output. Supported for matrix and segmented-COO outputs in
  both the JAX and PyTorch bindings.
- **JAX CUDA graph replay**: JAX neighbor-list builders accept a
  `graph_mode` keyword (`GraphMode`) to capture and replay the build as a
  CUDA graph, reducing per-step launch overhead in MD loops.

### Changed (neighbors)

- Restructured `nvalchemiops/neighbors/` into per-strategy subpackages:
  `naive/`, `cell_list/`, `cluster_tile/`, `rebuild/`. Public launchers
  live under `*/launchers.py`; strategy selection lives under
  `*/dispatch.py`.
- The flat compatibility modules `nvalchemiops.neighbors.{naive_dual_cutoff,
  batch_naive, batch_cell_list, batch_naive_dual_cutoff, rebuild_detection}`
  continue to re-export the new entry points with `DeprecationWarning`.
  (Note: `nvalchemiops.neighbors.naive` and `nvalchemiops.neighbors.cell_list`
  are now the canonical subpackages, not deprecated shims.)

### Added (electrostatics)

- Higher-order (multipole) electrostatics for charges, dipoles, and quadrupoles
  (l = 0, 1, 2): direct-k Ewald (`multipole_ewald_summation`), particle-mesh
  Ewald (`multipole_particle_mesh_ewald`), reciprocal- and real-space entry
  points, electrostatic feature extraction (`multipole_electrostatic_features`),
  and an SCF cache/step API for repeated evaluations on a fixed cell. Provided as
  Warp kernels and `nvalchemiops.torch` bindings, single-system and batched, with
  energies, forces, moment gradients, stress, and force-loss (`create_graph`)
  training; the forward and first-order backward are `torch.compile`-compatible.

### Added (segment ops)

- Differentiable segment operations: backward kernels for the segment-op
  reductions enable autograd through `nvalchemiops.segment_ops`, with Torch
  (`nvalchemiops.torch.segment_ops`) and JAX (`nvalchemiops.jax.segment_ops`)
  bindings, an autograd example (`examples/02_segment_ops_autograd.py`), user
  guide docs, and benchmarks.

### Changed

- DFT-D3 dispersion kernels optimized for improved performance.
- Loosened the PyTorch version requirement to widen compatible installs.
- Updated the CUDA backend extras (`torch-cu12`/`jax-cu12` and related
  optional dependencies).

## 0.3.0 - 2026-XX-XX

### Breaking Changes

- **PyTorch is now an optional dependency**: The previous PyTorch-based functionality
has been moved to a separate `nvalchemiops.torch` namespace. See the hosted documentation
for a detailed migration guide. Previous imports should still be supported, however
will issue deprecation warnings. The old interfaces will be removed in an upcoming
release.

### Added

- Framework-agnostic Warp kernel layer for all modules (neighbors, electrostatics,
  dispersion, math/spline) that operates directly on `warp.array` objects. A best
  effort to have interfaces that mirror their framework bindings is made, however
  due to differences in functionalities this may not always be possible.
- Thin PyTorch bindings in `nvalchemiops.torch.*` that wrap the Warp kernels.
- Deprecation warnings for old import paths to guide migration.
- JAX bindings in `nvalchemiops.jax.*` that wrap the Warp kernels, providing
  support for neighbor lists, DFT-D3 dispersion, electrostatics (Coulomb, Ewald,
  PME), and splines with `jax.jit` compatibility.
- GPU-accelerated molecular dynamics integrators with single-system and batched modes:
Velocity Verlet (NVE), Langevin (NVT), Nosé-Hoover Chain (NVT), NPT, NPH, and
Velocity Rescaling
- FIRE (Fast Inertial Relaxation Engine) geometry optimizer with adaptive timestep,
variable cell optimization, and cell filtering for constrained optimization
- Lennard-Jones potential with GPU-accelerated energy, force, and virial computation
integrated with neighbor lists
- Batch processing utilities (`nvalchemiops.batch_utils`) with support for both
`batch_idx` (ragged arrays) and `atom_ptr` (CSR format) including operations:
`batch_sum`, `batch_mean`, `batch_max`, `batch_min`, `batch_scale`,
`batch_normalize`, `batch_gather`, `batch_scatter`
- Cell manipulation utilities including volume calculation, inverse, wrapping,
alignment, and transformation
- SHAKE and RATTLE constraint algorithms for bond length constraints and rigid
molecules

## 0.2.0 - 2025-12-19

### Added

- Methods/kernels for computing electrostatic interactions
  - Includes direct Coulomb, Ewald, and particle mesh Ewald methods.
  - Some supporting math routines including spherical harmonics, spline
  evaluation, and Gaussian basis.
- New scripts in the `examples/electrostatics` folder that demonstrate
the new electrostatics interface.

### Changed

- Default behavior for `estimate_max_neighbors` is now more sensible
  - The default `atomic_density` value is changed from 0.5 to 0.35, which
  should provide better estimates of the maximum number of neighbors for
  most systems.
  - The rounding value has now been changed from the nearest power of 2
  to the nearest multiple of 16, which means the padding in neighbor
  matrices will be significantly lower and more realistic, as the prior
  behavior tended to significantly overpredict the maximum neighbor count.

### Fixed

- Issue #2 and #3 duplicate neighbors appearing in cell and batched cell lists.

## 0.1.0 - 2025-12-05

First release of the package

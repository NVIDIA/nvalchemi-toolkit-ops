# Changelog

## Unreleased

### Fixed

- **MTK NPT/NPH cell-propagation state-variable mismatch**: the
  cell-velocity-update kernel wrote a strain-rate-units acceleration
  (`V·(P − P_ext)/W`) into `cell_velocity` while drag, position, and
  cell-KE helpers consumed it as `ḣ = dh/dt` and the cell update did
  `h_new = h + dt·h_dot` (treating it as `dh/dt`). The mismatch cost a
  factor of cell length in the cell response and suppressed volume
  fluctuations by ~40× vs ASE `MTKNPT` / TorchSim
  `npt_nose_hoover_isotropic` in `nvalchemi-toolkit` PR #90's empirical
  octane comparison. Treats `cell_velocity` as the strain rate
  `ε̇ = p_g/W` consistently and updates the cell as
  `h_new = h + dt · ε̇ · h`.
- **MTK velocity-half-step modular-invariance coupling factor**: the
  12 NPT/NPH velocity-half-step kernels used `α = 1 + 1/(3N_atoms)`
  instead of the canonical `α = 1 + d/N_f = 1 + 1/N_atoms`
  (ASE `IsotropicMTKNPT._integrate_p`, TorchSim
  `_npt_nose_hoover_isotropic_inner_step`).

### Breaking Changes

- `cell_velocities` semantics: callers must now store the strain rate
  `ε̇ = p_g/W`, not the absolute derivative `ḣ = dh/dt`. Kernel
  signatures are unchanged.

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

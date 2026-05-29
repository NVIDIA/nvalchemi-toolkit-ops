<!-- markdownlint-disable MD013 -->

# Change Log

## Unreleased

### Neighbors subpackage layout

The `nvalchemiops.neighbors` package was restructured from flat modules into
per-strategy subpackages: `naive/`, `cell_list/`, `cluster_tile/`, and
`rebuild/`.  Public launchers live under `*/launchers.py`; strategy selection
lives under `*/dispatch.py`.

The flat compatibility modules
(`nvalchemiops.neighbors.{naive_dual_cutoff, batch_naive, batch_cell_list,
batch_naive_dual_cutoff, rebuild_detection}`) continue to re-export the new
entry points and emit `DeprecationWarning`.  `nvalchemiops.neighbors.naive`
and `nvalchemiops.neighbors.cell_list` are now the canonical subpackages (not
deprecated shims).  New code should import directly from the subpackages.

### Added: neighbor-list features

- **Pair potentials evaluated inline.** Neighbor kernels accept a
  user-supplied `pair_fn` callback (with `pair_params`, `pair_energies`,
  `pair_forces` buffers) that returns per-pair energy and force as pairs
  are enumerated, so Lennard-Jones–style potentials no longer require a
  separate pass over the neighbor list.
- **Per-pair vectors and distances on demand.** Pass `return_vectors=True`
  and/or `return_distances=True` to get the separation vectors `r_ij` and
  Euclidean distances `|r_ij|` alongside the neighbor matrix, avoiding a
  manual recomputation downstream.
- **Cluster-pair tile algorithm.** A new CUDA strategy is available under
  `nvalchemiops.neighbors.cluster_tile`, with framework bindings exposed
  by `nvalchemiops.{jax,torch}.neighbors`. `neighbor_list` selects it
  automatically for fully-periodic float32 CUDA inputs with at least
  2000 average atoms per system; pass `method="cluster_tile"` (or
  `"batch_cluster_tile"`) to force it. Dual cutoff is supported in
  matrix format.
- **Partial rebuild for batched workflows.** Pass `rebuild_flags` to
  re-enumerate only the systems whose atoms have moved enough to need a
  fresh list; unchanged systems keep their previous output. Supported
  for matrix and segmented-COO outputs in both the JAX and PyTorch
  bindings.

## Version 0.3.0

### Breaking Changes

- **PyTorch is now an optional dependency**: Core codebase consists of framework-agnostic `warp-lang` kernels with PyTorch bindings in separate namespace (`nvalchemiops.torch.*`). You can install the minimum supported version of PyTorch via `uv pip install nvalchemiops[torch]`.
- **Naive PBC cached metadata changed**: public Torch and JAX naive neighbor-list workflows now cache `shift_range_per_dimension`, `num_shifts_per_system`, and `max_shifts_per_system`. `shift_offset` and `total_shifts` are no longer part of the public API for cached naive-PBC inputs.

### Migration Guide

```{tip}
If PyTorch is detected in the environment, existing imports will continue
to work for the next few minor version increments, but will emit warnings
to remind users to update import paths (shown below).
```

- Core modules comprise the pure `warp-lang` kernels and launchers.
- **PyTorch neighbor lists**: Change `nvalchemiops.neighborlist.neighbor_list`  to `nvalchemiops.torch.neighbors.neighbor_list`
- **DFT-D3**: Change `from nvalchemiops.interactions.dispersion import dftd3` to `from nvalchemiops.torch.interactions.dispersion import dftd3`
- **Coulomb**: Change `from nvalchemiops.interactions.electrostatics import coulomb_energy` to `from nvalchemiops.torch.interactions.electrostatics import coulomb_energy`
- **Ewald**: Change `from nvalchemiops.interactions.electrostatics import ewald_summation` to `from nvalchemiops.torch.interactions.electrostatics import ewald_summation`
- **PME**: Change `from nvalchemiops.interactions.electrostatics import particle_mesh_ewald` to `from nvalchemiops.torch.interactions.electrostatics import particle_mesh_ewald`
- **Utility functions**: `estimate_cell_list_sizes` and `estimate_batch_cell_list_sizes` are now imported directly from `nvalchemiops.torch.neighbors` (previously `nvalchemiops.neighborlist.neighbor_utils`)

## Version 0.2.0

- Bug fixes associated with neighbor list computation.
- Added electrostatics interface.

## Version 0.1.0

- Initial public beta release of `nvalchemiops`.

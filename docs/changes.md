<!-- markdownlint-disable MD013 -->

# Change Log

## v0.4.0 (Unreleased)

### Electrostatics energy-derivative migration

- Full PyTorch Ewald/PME APIs support energy-derived forces, charge
  gradients, and strain-first virials for differentiable training, including
  second-order force/stress losses.
- Full JAX Ewald/PME energy-only APIs support gradients for positions, charges,
  and strain-first virials. JAX PME reciprocal position and charge losses use
  the native PME mesh Hessian-vector product path.
- Direct-output flags on full Ewald/PME APIs remain functional but are
  deprecated for differentiable training. Component `compute_forces=True`
  remains available for no-autograd MD/inference use; component charge-gradient,
  virial, and hybrid direct outputs warn as legacy training-style outputs.
- Slab corrections participate in energy-derived full-API gradients while the
  standalone explicit-output slab kernels remain available for forward direct
  outputs.
- Electrostatics gradients are defined only for positions, charges, and cell.
  Setup values such as alpha are constants, and cell-derived reciprocal caches
  are static metadata assumed to correspond to the current cell.
- `compute_bspline_moduli_1d` is exported from the top-level PyTorch and JAX
  electrostatics namespaces for PME precompute workflows.

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

### Neighbor-list helper and dependency changes

- `nvalchemiops.neighbors.zero_array` is **deprecated**: it now emits a
  `DeprecationWarning` and forwards to `array.zero_()`.  Call `array.zero_()`
  directly.
- The internal `make_outer_neigh_offsets` helper was **removed** (it was
  unused outside the old launchers).
- The minimum `warp-lang` requirement is raised to **`>= 1.13`** (lock file
  pinned to `1.13.0`).

### Added: neighbor-list features

- **Pair potentials evaluated inline.** Single-cutoff neighbor kernels accept a
  user-supplied Warp `pair_fn` with a per-atom `pair_params` table and return
  per-pair energy and force as pairs are enumerated, so Lennard-Jones–style
  potentials no longer require a separate pass over the neighbor list. The
  optional `pair_energies` / `pair_forces` buffers are auto-allocated and
  returned, matrix- or COO-shaped to match the output format, and are
  forward-only (the kernels do not backpropagate through the functor). Wired
  across Warp, PyTorch, and JAX (JAX specifics below).
- **Per-pair vectors and distances on demand.** Pass `return_vectors=True`
  and/or `return_distances=True` to get the separation vectors `r_ij` and
  Euclidean distances `|r_ij|` alongside the neighbor matrix, avoiding a
  manual recomputation downstream.  With `return_neighbor_list=True`, the
  `naive` / `cell_list` paths return them as flat per-pair COO arrays aligned
  with the neighbor list.  These outputs (and `pair_fn`) combine with
  `half_fill=True` on the `naive` / `cell_list` Torch and JAX paths, with correct
  per-pair gradients; they are unsupported with `rebuild_flags` (skipped systems
  return stale cached geometry).
- **Cluster-pair tile algorithm.** A new CUDA strategy is available under
  `nvalchemiops.neighbors.cluster_tile`, with framework bindings exposed
  by `nvalchemiops.{jax,torch}.neighbors`. `method=None` can now select it
  for feasible CUDA float32 fully-periodic workloads with compatible output
  options and contiguous batch metadata. Dual cutoff is supported in matrix
  format for explicit cluster-tile calls.
- **Partial rebuild for batched workflows.** Pass `rebuild_flags` to
  re-enumerate only the systems whose atoms have moved enough to need a
  fresh list; unchanged systems keep their previous output. Supported
  for matrix and segmented-COO outputs in both the JAX and PyTorch
  bindings.
- **JAX cell-list / naive feature parity.** The JAX bindings now honor `half_fill`
  and `fill_value` for `cell_list` / `batch_cell_list` (previously silently
  ignored), and the fine-grained strategies `cell_list_pair_centric` and
  `naive_tile` run their dedicated CUDA kernels (via `jax_callable`) instead of
  falling back to `atom_centric` / scalar, so `suggest` / `estimate` names
  round-trip through `method=`. `pair_fn` is wired through all JAX paths
  (`jax_kernel` for naive/cell-list, `jax_callable` for the tile paths;
  cluster-tile is fp32- and eager-cutoff only, with matrix and COO pair outputs —
  COO is eager-only); `target_indices` +
  pair outputs work on `cell_list` / `batch_cell_list` (compact `num_targets`
  rows; COO source index is the compact row, matching Torch). JAX `pair_centric`
  sizes its launch from the host and requires `graph_mode="none"`, raising a
  clear error under `jax.jit` with a traced radius; only `pair_centric` +
  `target_indices` is rejected (identical results via `atom_centric`).

### Changed: automatic method selection

- **`neighbor_list(method=None)` now dispatches on a geometry cost model**
  instead of average atom count. The naive↔cell-list crossover is governed by
  the number of cutoff-sized cells `V / cutoff**3` (atom count and density
  cancel out of the per-system ratio), so large high-cutoff systems are no
  longer mis-routed to the `O(N^2)` naive path. The selector also compares a
  guarded cluster-tile cost for feasible CUDA float32 periodic cases. The cost
  estimate is exposed publicly (Warp, PyTorch and JAX) as
  `estimate_neighbor_list_costs` (feasible strategies + relative cost, sorted) and
  `suggest_neighbor_list_method` (the cheapest); call one once on per-system
  geometry (`batch_ptr`, `cell`, `pbc`, `cutoff`) and pass the returned strategy
  name explicitly to avoid repeated auto-dispatch syncs. The returned
  fine-grained names (`naive_tile`, `cell_list_pair_centric`, `cluster_tile`, …,
  plus `batch_` variants) are accepted directly as `method=`. The crossover
  constants are env-overridable (`NVALCHEMI_NEIGHLIST_CELL_SHELL`,
  `NVALCHEMI_NEIGHLIST_CELL_SETUP`).
- **Cell-less COO auto dispatch return arity is method-dependent.** When
  `method=None` and `return_neighbor_list=True` with no input `cell`, the chosen
  method is honored as-is: `naive` returns a 2-tuple `(neighbor_list,
  neighbor_ptr)` (non-periodic, no shifts) and `cell_list`/`batch_cell_list`
  synthesize a non-PBC cell and return a 3-tuple `(neighbor_list, neighbor_ptr,
  shifts)` with zeroed shifts. Pass an explicit `cell`+`pbc` (or use the
  neighbor-matrix format) for a stable 3-tuple. *Caveat:* for a large
  non-periodic system without a cell, `cell_list` synthesizes a degenerate
  single-cell grid (\(O(N^2)\) COO conversion) — pass an explicit `cell` sized
  to the bounding box for large non-periodic systems.

### Fixed

- **JAX `naive` PBC pair-output paths dropped non-zero periodic images.** The
  JAX `naive_neighbor_list` pair-output path (`return_distances` /
  `return_vectors`, and now `pair_fn`) launched its periodic kernel with the
  shift axis pinned to 1, so when `cutoff` exceeded half the cell width (R>1)
  every non-zero periodic image was silently dropped — yielding too few
  neighbors and incorrect per-pair distances/vectors/forces relative to the
  PyTorch binding. The launch now enumerates all shifts (`max_shifts`), matching
  PyTorch and the analytic neighbor set in the multi-image regime. The
  single-cutoff `cutoff < half-cell` (R==1) case is unchanged.
- **JAX per-pair distance/vector higher-order gradients were wrong for losses
  nonlinear in distance.** The JAX neighbor-list autograd returned the *detached*
  Warp-kernel distances/vectors and re-attached only a first-order gradient via a
  `custom_vjp`, so the Hessian / Hessian-vector-product was incorrect (~45% off)
  whenever the downstream loss was nonlinear in the returned distances (e.g.
  `(distances**2).sum()`); first-order gradients (forces) were unaffected. The
  geometry is now reconstructed as a live, differentiable pure-JAX function of
  positions/cell, so gradients of all orders are exact (matching PyTorch and the
  analytic Hessian). Affects all JAX `return_distances`/`return_vectors` bindings
  (`naive`, `cell_list`, `cluster_tile`, batched).

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

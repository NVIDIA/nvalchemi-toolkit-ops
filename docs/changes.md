<!-- markdownlint-disable MD013 -->

# Change Log

## v0.4.1 - 2026-08-03

### Added

- Monopole Torch and JAX Ewald, PME, and slab entry points accept keyword-only
  `energy_reduction="atom" | "system"` (default `"atom"`). `"atom"` returns
  per-atom energies `(N,)`; `"system"` returns per-system totals `(B,)`.
  Direct-output fields (forces, charge gradients, virials) keep their existing
  shapes. Torch eager atom mode may synchronize once per participating component
  when a materialized uniform cotangent is proven by value inspection; system
  mode is structurally sync-free for arbitrary `(B,)` loss weights. JAX adds
  API/layout parity only; underlying Warp kernels remain atom-buffer-oriented.
- PyTorch cluster-tile selective calls can append caller-owned tile state with
  `return_state=True`, without changing the default neighbor-list return arity.

### Changed

- Improved JAX neighbor-list import performance by deferring dtype-specific
  direct naive and cell-list Warp wrapper registration until first use.
  Cluster-tile graph callbacks now use bundled callback/preload registrations
  with lazy direct kernels for naive and cell-list paths. Public behavior is
  unchanged.

### Fixed

- Fixed Torch PME and Ewald energy gradients for connected charge, position, and
  cell inputs. Non-uniform or weighted energy losses and `create_graph=True`
  higher-order derivatives no longer double-count upstream chain-rule terms.
- Torch `ewald_reciprocal_space` now preserves graph-connected reciprocal
  vectors for cell/strain autograd, restoring the physical reciprocal Ewald
  virial when vectors are regenerated from the differentiable cell.
- Torch Ewald, PME, and slab backward paths now compile when an explicit
  single-system batch (`batch_idx=zeros(N)`) is supplied. Reciprocal PME
  compiled gradients are also correct when a compiled function is reused across
  mesh sizes.
- Torch DFT-D3 custom operators now zero caller-owned energy, forces,
  coordination-number, and virial buffers before empty-system or zero-edge
  early returns, so reused output tensors cannot retain stale values.
- JAX DFT-D3 CSR calls with atoms but no edges return zero-filled per-atom
  forces and coordination numbers with shapes `(N, 3)` and `(N,)`, matching
  the neighbor-matrix contract and preserving per-system energy and virial axes.
- JAX cell-list builds now derive search radii from their realized grids,
  preventing missed neighbors when static capacity changes the constructed
  grid. Batched `capacity_strategy="geometry"` preserves promoted grids for
  all non-empty systems by reserving an equal per-system capacity; volume-based
  sizing remains the default. Fused Warp graph calls with explicit
  `max_total_cells` now require an explicit `neighbor_search_radius`.
- Unbatched JAX naive dual-cutoff PBC neighbor lists now populate both cutoff
  outputs when using the default `wrap_positions=True`. Previously this path
  wrapped positions but skipped the fill kernel, leaving zero counts and padded
  matrices.
- Batched PyTorch cluster-tile segmented COO validates fixed topology, offsets,
  counts, and tile-state capacities before launching Warp kernels.
- Single-system Torch and JAX segmented cluster-tile COO now require one exact
  physical interval, bound writes by output capacity, fail closed for malformed
  offsets, and cap compiled/JIT active counts to writable capacity. Batched
  per-system physical subsegments remain supported.
- Compiled unified PyTorch cluster-tile dispatch now rejects tensor-valued PBC
  rather than treating it as fully periodic. Eagerly validate PBC and compile the
  direct single-system fixed-state route instead.
- JAX cluster-tile empty selective rebuilds now preserve false-flag state and
  clear true-flag pair and tile counts while retaining fixed-capacity storage.

## v0.4.0 - 2026-07-13

### Added

- Full Torch Ewald/PME APIs support energy-derived forces, charge gradients,
  and strain-first virials, including second-order force/stress losses.
- Full JAX Ewald/PME energy-only calls support first-order gradients for
  positions, charges, and row-vector displacement virials. JAX PME reciprocal
  higher-order support is limited to tested position and charge scalar losses;
  PME cell/stress/strain higher-order derivatives remain unsupported.
- 2D slab correction is exposed through `compute_slab_correction` and the
  high-level Ewald/PME `slab_correction=` keyword in Torch and JAX.
- Higher-order multipole electrostatics for charges, dipoles, and quadrupoles
  (`l = 0, 1, 2`) are available through Torch/Warp direct-k Ewald, PME,
  feature extraction, and SCF cache/step APIs.
- Differentiable segment operations are available through
  `nvalchemiops.torch.segment_ops` and `nvalchemiops.jax.segment_ops`.
- Neighbor-list APIs now include inline `pair_fn` potentials, optional per-pair
  vectors/distances, a cluster-pair tile strategy, partial rebuild flags, and
  public strategy cost/suggestion helpers.
- `compute_bspline_moduli_1d` is exported from the top-level Torch and JAX
  electrostatics namespaces for PME precompute workflows.

### Changed

- Direct-output flags on full Ewald/PME APIs remain functional but are
  deprecated for differentiable training. Use energy-only calls plus framework
  autograd for forces, charge gradients, and virials in training workflows.
- `neighbor_list(method=None)` now uses a geometry cost model and can select
  fine-grained strategies such as `naive_tile`, `cell_list_pair_centric`, and
  `cluster_tile` when eligible.
- The `nvalchemiops.neighbors` package was restructured into per-strategy
  subpackages (`naive/`, `cell_list/`, `cluster_tile/`, `rebuild/`). Flat
  compatibility modules continue to re-export with `DeprecationWarning`.
- DFT-D3 dispersion kernels were optimized for improved performance.
- PyTorch version requirements were loosened and CUDA backend extras were
  updated for CUDA 12/13 install workflows.
- The minimum `warp-lang` requirement is now `>= 1.13`.

### Fixed

- Fixed an issue with JAX `naive` PBC pair-output paths that dropped non-zero
  periodic images. The  JAX `naive_neighbor_list` pair-output path (`return_distances` /
  `return_vectors`, and now `pair_fn`) launched its periodic kernel with the
  shift axis pinned to 1, so when `cutoff` exceeded half the cell width (R>1)
  every non-zero periodic image was silently dropped — yielding too few
  neighbors and incorrect per-pair distances/vectors/forces relative to the
  PyTorch binding. The launch now enumerates all shifts (`max_shifts`), matching
  PyTorch and the analytic neighbor set in the multi-image regime. The
  single-cutoff `cutoff < half-cell` (R==1) case is unchanged.
- Fixed an issue with JAX per-pair distance/vector higher-order gradients.
  The JAX neighbor-list autograd returned the *detached*
  Warp-kernel distances/vectors and re-attached only a first-order gradient via a
  `custom_vjp`, so the Hessian / Hessian-vector-product was incorrect (~45% off)
  whenever the downstream loss was nonlinear in the returned distances (e.g.
  `(distances**2).sum()`); first-order gradients (forces) were unaffected. The
  geometry is now reconstructed as a live, differentiable pure-JAX function of
  positions/cell, so gradients of all orders are exact (matching PyTorch and the
  analytic Hessian). Affects all JAX `return_distances`/`return_vectors` bindings
  (`naive`, `cell_list`, `cluster_tile`, batched).
- Fixed DFT-D3 forces and virials with S5 smoothing. When smoothing was
  active, the CN-chain `dE/dCN` used the unswitched pair energy, so CN-chain
  forces and virials did not exactly match the gradient of the switched energy.
  Only runs with S5 smoothing enabled were affected; the default (smoothing
  disabled) was already correct and is unchanged.
- Naive PBC neighbor wrapping now leaves non-periodic axes unwrapped when
  per-axis `pbc` flags are supplied.
- Fixed Torch Ewald gradients for non-uniform per-atom energy cotangents.
- JAX electrostatics no longer imports the removed `jax.custom_transpose`;
  transpose rules use stable `jax.custom_vjp` paths.
- FIRE2 variable-cell updates now advance positions and cell degrees of
  freedom consistently during constrained/variable-cell relaxation.
- Neighbor-list launchers now reject unbatched methods when batch metadata is
  supplied.
- MTK NPT/NPH cell propagation, velocity half-step coupling, and barostat
  half-step thermostat coupling now match the intended strain-rate formulation.
- JAX `naive` PBC pair-output paths enumerate all periodic images in the
  multi-image regime.
- JAX per-pair distance/vector outputs are reconstructed as live differentiable
  geometry, fixing higher-order gradients for nonlinear distance losses.

### Deprecated and Removed

- `compute_forces`, `compute_virial`, `compute_charge_gradients`, and
  `hybrid_forces` direct-output flags on full Ewald/PME APIs are deprecated for
  differentiable training.
- `nvalchemiops.neighbors.zero_array` is deprecated; call `array.zero_()`
  directly.
- `cells_inv` and `volumes` dynamics arguments listed in `CHANGELOG.md` are
  deprecated.
- `cell_velocities` now stores the strain rate `ε̇ = p_g/W`, not
  `ḣ = dh/dt`.
- `npt_barostat_half_step{,_aniso,_triclinic}` drop the `eta_dots` argument.
- The internal `make_outer_neigh_offsets` helper was removed.

## Version 0.3.0 - 2026-03-16

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

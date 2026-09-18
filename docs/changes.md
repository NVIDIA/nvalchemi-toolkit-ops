<!-- markdownlint-disable MD013 -->

# Change Log

## Unreleased

### Added

- FourierD3, a particle-mesh evaluation of the DFT-D3(BJ) dispersion correction. Where
  `dftd3` sums pair interactions in real space, `fourier_dftd3` evaluates the same
  correction on a mesh in `O(N log N)` with **no real-space cutoff on the dispersion sum**;
  the only real-space cutoff remaining is the short coordination-number list a machine-learned
  force field already builds. This matters because a `1/r^6` interaction summed over three
  dimensions leaves a truncation error decaying only as `1/r^3`, so converging the real-space
  form to sub-meV/atom requires cutoffs at which the neighbour list dominates the simulation
  step. Available as `nvalchemiops.interactions.dispersion._fourier_dftd3` (Warp component
  launchers), and as `fourier_dftd3` in both the Torch and JAX dispersion modules. Energy,
  forces and the virial are supported, in float32 and float64, batched, with either neighbour
  format. `dftd3` is unchanged and remains the right choice for open boundary conditions and
  small molecules.
- `FourierD3Parameters` in the Torch and JAX dispersion modules, holding the low-rank
  decomposition of Grimme's reference tensor for the species present. The decomposition is
  independent of the damping parameters, so one instance is valid for every functional; the
  damping values are call-time arguments and are never stored alongside derived quantities.
- `decompose_c6_reference` in `nvalchemiops.interactions.dispersion._c6_decomposition`, a
  host-side helper that performs that decomposition and caches it on the content of the
  reference tables together with every option that can change the result.
- `batch_spline_spread_channels` and `batch_spline_gather_channels` in
  `nvalchemiops.math.spline`, Warp-level launchers for the multi-channel B-spline kernels.
  Both key the cell lookup and the mesh slab off one per-atom index, so a caller partitioning
  atoms by something other than the system alone can pass a composite index and have each atom
  touch only its own slab.

- FourierD3 owns its B-spline spread and gather rather than reusing the shared ones from
  `nvalchemiops.math.spline`. The shared spread skips stencil points whose interpolation
  weight falls at or below `1e-8`, which is a sound efficiency measure for a value but not
  for a gradient: dropping a stencil point removes its contribution to the force as well, and
  for an atom near a mesh cell boundary a quarter of the stencil can fall below the
  threshold. Keeping every point costs a few extra atomic adds and makes the interpolation
  the exact B-spline that the gather differentiates. PME is unaffected.

- `FourierD3Setup` in the Torch dispersion module, holding the cell- and mesh-derived
  quantities that do not change between steps. Passing it to `fourier_dftd3` skips a matrix
  inversion and a set of spline moduli per call, worth 3.1x at 8,000 atoms, and is required
  for
  `torch.compile(mode="reduce-overhead")` because `torch.linalg.inv` cannot be recorded into
  a CUDA graph. Warp launches are now bound to PyTorch's current stream without an entry
  synchronisation, which graph capture also forbids.

- `rank_chunk_size` on `fourier_dftd3` in both the Torch and JAX bindings, capping how many
  rank slots are resident on the mesh at once. The mesh and its transforms dominate the
  workspace and scale as `num_systems * n_species * rank * nx * ny * nz`, so many species or
  a large retained rank can exhaust device memory on a fine mesh. Every stage after the
  coordination number is a sum over rank slots with no coupling between them, so chunking
  leaves the result unchanged to round-off; the cost is one extra spread, transform pair and
  gather per chunk. Defaults to `None`, the single-pass behaviour. Host-static, so it is safe
  under `jax.jit` and `torch.compile`.

### Changed

- The minimum `warp-lang` requirement is now `>= 1.16.0`, raised from `>= 1.13.0`. The JAX
  FourierD3 binding passes `block_dim` to `warp.jax_experimental.jax_kernel` to launch its
  block-per-atom coordination-number and reciprocal-space kernels, and that argument was
  added to `jax_kernel` in Warp 1.16.0. FourierD3 is the only component that uses it; on an
  older Warp the JAX binding fails at import with a `TypeError`.

### Notes

- FourierD3 uses a modified coordination-number function that decays to zero at the neighbour
  list cutoff, where the standard D3 function tends to a non-zero constant. That modification
  is what makes the coordination numbers independent of the list used to build them, so the
  two functions are not identical; in practice the difference is small. On diamond with the
  published tables, FourierD3 agrees to 5.9e-06 relative with `dftd3` extrapolated to an
  infinite cutoff, while reading only a 6 Angstrom list, where `dftd3` at that same cutoff is
  still 10% short.
- `rcov` follows the same convention as `dftd3`: the shipped table already folds in Grimme's
  4/3 scale, so the counting function crosses one half at a separation equal to the sum of the
  two tabulated radii. Pass `dftd3` and `fourier_dftd3` the same table.
- `r_cut` must equal the radius the neighbour list was built with, and has no default, because
  the reference parameters are conventionally in atomic units and a value meant as 6 Angstrom
  would otherwise act silently as 6 Bohr.

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

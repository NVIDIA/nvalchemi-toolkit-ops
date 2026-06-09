<!-- markdownlint-disable MD025 -->

(migration_guide)=

# Migration Guide

This guide lists user-visible migrations by release.

## v0.4.0 (Unreleased): Electrostatics

### Energy-Derivative Training

For full Ewald/PME APIs, prefer deriving training quantities from the returned
energy tensor instead of requesting direct outputs. On the full APIs, each flag
below remains functional but emits `DeprecationWarning`; component APIs such as
`ewald_real_space`, `ewald_reciprocal_space`, and `pme_reciprocal_space` keep
direct outputs for no-autograd MD/inference loops.

| Direct-output flag | Energy-derived replacement |
|--------------------|----------------------------|
| `compute_forces=True` | `forces = -grad(E.sum(), positions)` |
| `compute_virial=True` | `grad_u = grad(E.sum(), displacement)` with the row-vector displacement recipe; `virial = -grad_u`, `stress = grad_u / V` |
| `compute_charge_gradients=True` | `dEdq = grad(E.sum(), charges)` |
| `hybrid_forces=True` | Keep `charges = charge_model(positions)` in the graph and derive forces from energy |

Torch full Ewald/PME supports first- and second-order energy derivatives for
force/stress training. When a loss mixes forces **and** stress, take both from a
single `grad(E.sum(), (positions, strain), create_graph=True)` call rather than two
separate `grad` calls -- this runs the reciprocal double-backward once instead
of twice (see {ref}`energy-derivative-contract`).
This support is exposed through standard autograd on scalar losses; the
electrostatics APIs do not expose public Hessian or Jacobian tensors/functions.

JAX full Ewald/PME supports first-order energy derivatives for positions,
charges, and row-vector displacement virials using the same per-system
energy-cotangent reducer as Torch. Higher-order JAX support is limited to tested
position and charge scalar losses. JAX PME stress/cell/strain, alpha, and
precomputed-metadata higher-order paths are unsupported until implemented and
tested. JAX direct-output flags remain functional for compatibility in v0.4.0
but are deprecated for differentiable training.

### Precomputed Electrostatics Metadata

Advanced callers can precompute setup-only metadata and pass it to the Ewald/PME
entry points instead of regenerating it inside hot loops.

| Surface | Precomputed inputs |
|---------|--------------------|
| Ewald reciprocal | `k_vectors`, `miller_bounds` |
| PME reciprocal | `cell_inv_t`, `volume`, `k_vectors`, `k_squared`, `moduli_x`, `moduli_y`, `moduli_z` |
| PME B-spline helpers | `compute_bspline_moduli_1d(...)` |

These inputs are caches, not differentiable parameters. `alpha`, cutoffs,
mesh controls, batch metadata, neighbor topology, and PME B-spline moduli are
treated as constants even if supplied as grad-bearing tensors. Cell-derived
caches such as `k_vectors`, `k_squared`, `volume`, and `cell_inv_t` remain
accepted when cell derivatives are requested, but they are static metadata
assumed to correspond to the current `cell`.

For fixed-cell loops, build metadata once from a detached or stopped-gradient
cell and reuse it while the cell is unchanged:

```python
# Torch Ewald fixed-cell loop.
with torch.no_grad():
    k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)
for positions in trajectory:
    energy = ewald_summation(..., cell=cell, k_vectors=k_vectors)
```

```python
# JAX PME fixed-cell loop.
cell_static = jax.lax.stop_gradient(cell)
cell_inv_t = jnp.linalg.inv(cell_static).transpose(0, 2, 1)
volume = jnp.abs(jnp.linalg.det(cell_static))
reciprocal_cell = 2.0 * jnp.pi * jnp.linalg.inv(cell_static)
k_vectors, k_squared = generate_k_vectors_pme(
    cell_static, mesh_dimensions, reciprocal_cell=reciprocal_cell
)
mesh_nx, mesh_ny, mesh_nz = mesh_dimensions
moduli_x = compute_bspline_moduli_1d(
    jnp.fft.fftfreq(mesh_nx, d=1.0 / mesh_nx), mesh_nx, spline_order
)
moduli_y = compute_bspline_moduli_1d(
    jnp.fft.fftfreq(mesh_ny, d=1.0 / mesh_ny), mesh_ny, spline_order
)
moduli_z = compute_bspline_moduli_1d(
    jnp.fft.rfftfreq(mesh_nz, d=1.0 / mesh_nz), mesh_nz, spline_order
)
for positions in trajectory:
    energy = particle_mesh_ewald(
        positions, charges, cell,
        k_vectors=k_vectors,
        k_squared=k_squared,
        cell_inv_t=cell_inv_t,
        volume=volume,
        moduli_x=moduli_x,
        moduli_y=moduli_y,
        moduli_z=moduli_z,
        mesh_dimensions=mesh_dimensions,
        spline_order=spline_order,
    )
```

If the cell changes and cell-gradient correctness matters, regenerate the
cell-derived metadata for that cell or omit the cache so the wrapper computes it
internally. For `jax.jit`, `miller_bounds`, `mesh_dimensions`, spline order, and
other shape controls must be concrete static values.

## v0.3.0: PyTorch Namespace Migration

Starting with version 0.3.0, PyTorch is now an optional dependency. The previous
PyTorch-based functionality has been moved to a separate `nvalchemiops.torch`
namespace. This section provides a mapping of old import paths to new ones.

### Import Path Changes

| Old Import Path | New Import Path |
|-----------------|-----------------|
| `from nvalchemiops.interactions.dispersion import dftd3` | `from nvalchemiops.torch.interactions.dispersion import dftd3` |
| `from nvalchemiops.interactions.dispersion import D3Parameters` | `from nvalchemiops.torch.interactions.dispersion import D3Parameters` |
| `from nvalchemiops.neighbors import neighbor_list` | `from nvalchemiops.torch.neighbors import neighbor_list` |
| `from nvalchemiops.neighbors import estimate_max_neighbors` | `from nvalchemiops.torch.neighbors.neighbor_utils import estimate_max_neighbors` |
| `from nvalchemiops.neighborlist import neighbor_list` | `from nvalchemiops.torch.neighbors import neighbor_list` |
| `from nvalchemiops.neighborlist import cell_list` | `from nvalchemiops.torch.neighbors import cell_list` |

### Backwards Compatibility

The old import paths will continue to work but will emit `DeprecationWarning`
messages. They will be removed in a future release.

## Naive PBC Metadata Changes

Advanced callers that precompute periodic metadata for naive neighbor-list
methods should update cached arguments as follows:

| Old Cached Inputs | New Cached Inputs |
|-------------------|-------------------|
| `shift_range_per_dimension`, `shift_offset`, `total_shifts` | `shift_range_per_dimension`, `num_shifts_per_system`, `max_shifts_per_system` |

The public Torch and JAX APIs now decode periodic shifts on-the-fly inside the
neighbor kernels. Materialized shift buffers and `shift_offset` / `total_shifts`
are no longer part of the public naive-PBC workflow.

## Warp Kernels

If you need direct access to the underlying Warp kernels (without PyTorch),
use the non-torch namespaces:

- `nvalchemiops.neighbors` - Warp neighbor list kernels
- `nvalchemiops.interactions.dispersion` - Warp dispersion kernels
- `nvalchemiops.interactions.electrostatics` - Warp electrostatics kernels
- `nvalchemiops.math` - Warp math and spline kernels

These modules comprise both targeted kernels as well as end-to-end launchers where
possible, which run the full workflow based on `warp.array`s.

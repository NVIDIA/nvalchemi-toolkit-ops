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
| `compute_virial=True` | `virial = -grad(E.sum(), strain)` with the strain-first recipe |
| `compute_charge_gradients=True` | `dEdq = grad(E.sum(), charges)` |
| `hybrid_forces=True` | Keep `charges = charge_model(positions)` in the graph and derive forces from energy |

Torch full Ewald/PME supports first- and second-order energy derivatives for
force/stress training. When a loss mixes forces **and** stress, take both from a
single `grad(E.sum(), (positions, strain), create_graph=True)` call rather than two
separate `grad` calls -- this runs the reciprocal double-backward once instead
of twice (see {ref}`energy-derivative-contract`).

JAX full Ewald/PME supports first-order energy derivatives for positions,
charges, and strain-consistent cell gradients using the same per-system
energy-cotangent reducer as Torch. JAX PME higher-order derivatives raise
`NotImplementedError` until a native PME Hessian-vector product is available.
JAX direct-output flags remain functional during the transition but are
deprecated for differentiable training.

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

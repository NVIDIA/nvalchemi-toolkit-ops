<!-- markdownlint-disable MD025 -->

(migration_guide)=

# Migration Guide (v0.3.0)

Starting with version 0.3.0, PyTorch is now an optional dependency. The previous
PyTorch-based functionality has been moved to a separate `nvalchemiops.torch`
namespace. This guide provides a mapping of old import paths to new ones.

## Import Path Changes

| Old Import Path | New Import Path |
|-----------------|-----------------|
| `from nvalchemiops.interactions.dispersion import dftd3` | `from nvalchemiops.torch.interactions.dispersion import dftd3` |
| `from nvalchemiops.interactions.dispersion import D3Parameters` | `from nvalchemiops.torch.interactions.dispersion import D3Parameters` |
| `from nvalchemiops.neighbors import neighbor_list` | `from nvalchemiops.torch.neighbors import neighbor_list` |
| `from nvalchemiops.neighbors import estimate_max_neighbors` | `from nvalchemiops.torch.neighbors.neighbor_utils import estimate_max_neighbors` |
| `from nvalchemiops.neighborlist import neighbor_list` | `from nvalchemiops.torch.neighbors import neighbor_list` |
| `from nvalchemiops.neighborlist import cell_list` | `from nvalchemiops.torch.neighbors import cell_list` |

## Backwards Compatibility

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

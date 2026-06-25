<!-- markdownlint-disable MD013 -->

<!-- markdownlint-disable MD013 -->

(neighborlist_userguide)=

# Neighbor Lists

Neighbor lists enumerate atom pairs within a cutoff distance. ALCHEMI Toolkit-Ops
provides GPU-accelerated neighbor list algorithms via
[NVIDIA Warp](https://nvidia.github.io/warp/) with bindings for both PyTorch and JAX.

```{tip}
Start with the unified `neighbor_list` function
({func}`~nvalchemiops.torch.neighbors.neighbor_list` for PyTorch,
{func}`~nvalchemiops.jax.neighbors.neighbor_list` for JAX).
It automatically selects the best algorithm for your system size and handles
both single and batched inputs.
```

## Why Neighbor Lists Matter for Performance

Neighbor list construction can dominate runtime when called repeatedly:

- **Naive algorithms scale as \(O(N^2)\)**: Checking all atom pairs becomes
  prohibitive for systems with a large number of atoms. The "~2000 atoms"
  figures used below are illustrative only — the actual `naive`/`cell_list`
  crossover is decided per system by the geometry cost model (see the
  Method Dispatch section), not a fixed atom count.
- **Repeated construction**: callers that rebuild lists every step pay this
  cost on each call
- **Memory bandwidth**: Large neighbor matrices can bottleneck GPU throughput

ALCHEMI Toolkit-Ops addresses these costs with O(N) cell-list algorithms, a
cluster-pair tile algorithm for large fully-periodic float32 inputs, efficient
batch processing for heterogeneous inputs, and memory layouts optimized for
GPU access patterns. See [performance considerations](nl_performance) for
guidance.

## Quick Start

The `neighbor_list` function provides a unified interface that automatically
dispatches to the optimal algorithm based on system size and whether batch
indices are provided.

::::::{tab-set}

:::::{tab-item} PyTorch
:sync: pytorch

::::{tab-set}

:::{tab-item} Single + Large
:sync: single-large

Single system with >2000 atoms

```python
from nvalchemiops.torch.neighbors import neighbor_list

neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, method="cell_list"
)
```

Dispatches to {func}`~nvalchemiops.torch.neighbors.cell_list` --- \(O(N)\) algorithm
using spatial decomposition.
:::

:::{tab-item} Single + Small
:sync: single-small

Single system with <2000 atoms

```python
from nvalchemiops.torch.neighbors import neighbor_list

neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, method="naive"
)
```

Dispatches to {func}`~nvalchemiops.torch.neighbors.naive_neighbor_list` --- \(O(N^2)\)
algorithm with lower overhead.
:::

:::{tab-item} Batch + Large
:sync: batch-large

Multiple systems with >2000 atoms each

```python
from nvalchemiops.torch.neighbors import neighbor_list

neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cells, pbc=pbc,
    batch_idx=batch_idx, method="batch_cell_list"
)
```

Dispatches to {func}`~nvalchemiops.torch.neighbors.batch_cell_list` --- \(O(N)\)
algorithm for heterogeneous batches.
:::

:::{tab-item} Batch + Small
:sync: batch-small

Multiple systems with <2000 atoms each

```python
from nvalchemiops.torch.neighbors import neighbor_list

neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cells, pbc=pbc,
    batch_idx=batch_idx, method="batch_naive"
)
```

Dispatches to {func}`~nvalchemiops.torch.neighbors.batch_naive_neighbor_list` ---
\(O(N^2)\) algorithm for batched small systems.
:::

::::

:::::

:::::{tab-item} JAX
:sync: jax

::::{tab-set}

:::{tab-item} Single + Large
:sync: single-large

Single system with >2000 atoms

```python
from nvalchemiops.jax.neighbors import neighbor_list

neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, method="cell_list"
)
```

Dispatches to {func}`~nvalchemiops.jax.neighbors.cell_list` --- \(O(N)\) algorithm
using spatial decomposition.
:::

:::{tab-item} Single + Small
:sync: single-small

Single system with <2000 atoms

```python
from nvalchemiops.jax.neighbors import neighbor_list

neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, method="naive"
)
```

Dispatches to {func}`~nvalchemiops.jax.neighbors.naive_neighbor_list` --- \(O(N^2)\)
algorithm with lower overhead.
:::

:::{tab-item} Batch + Large
:sync: batch-large

Multiple systems with >2000 atoms each

```python
from nvalchemiops.jax.neighbors import neighbor_list

neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cells, pbc=pbc,
    batch_idx=batch_idx, method="batch_cell_list"
)
```

Dispatches to {func}`~nvalchemiops.jax.neighbors.batch_cell_list` --- \(O(N)\)
algorithm for heterogeneous batches.
:::

:::{tab-item} Batch + Small
:sync: batch-small

Multiple systems with <2000 atoms each

```python
from nvalchemiops.jax.neighbors import neighbor_list

neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cells, pbc=pbc,
    batch_idx=batch_idx, method="batch_naive"
)
```

Dispatches to {func}`~nvalchemiops.jax.neighbors.batch_naive_neighbor_list` ---
\(O(N^2)\) algorithm for batched small systems.
:::

::::

:::::

::::::

```{note}
When `method` is not specified, `neighbor_list` automatically selects by
comparing the estimated work of `naive`, `cell_list`, and `cluster_tile`,
computed from per-system geometry (atom counts and cell / bounding-box volumes)
rather than atom count alone.  The `naive`↔`cell_list` crossover is governed by the number of
cutoff-sized cells `V / cutoff**3` (atom count and density cancel out of the
per-system ratio): small or dense systems use `naive`, larger sparse systems
use `cell_list`.  This avoids routing large high-cutoff systems to the
\(O(N^2)\) path.  Auto-dispatch also considers `cluster_tile` for feasible
CUDA float32 fully-periodic workloads with compatible outputs and contiguous
batch metadata.  The same estimate is exposed publicly via
`suggest_neighbor_list_method` / `estimate_neighbor_list_costs` (see
[Estimating and Running a Strategy Explicitly](#estimating-and-running-a-strategy-explicitly));
call one once on per-system
geometry (`batch_ptr`, `cell`, `pbc`, `cutoff`) and reuse the returned strategy
name explicitly when repeated calls should avoid auto-dispatch syncs.  The
crossover constants are env-overridable (`NVALCHEMI_NEIGHLIST_CELL_SHELL`,
`NVALCHEMI_NEIGHLIST_CELL_SETUP`)---benchmark your workload and recalibrate if
needed.
```

## Data Formats

ALCHEMI Toolkit-Ops supports two output formats for neighbor data:

Neighbor Matrix (default)
: Fixed-size array of shape `(num_atoms, max_neighbors)` where each row
  contains the neighbor indices for that atom, padded with a fill value.
  Returns `(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)`.

Neighbor List (COO format)
: Sparse array of shape `(2, num_pairs)` containing `[source_atoms, target_atoms]`.
  Returns `(neighbor_list, neighbor_ptr, neighbor_list_shifts)` where
  `neighbor_ptr` is a CSR-style pointer array. The first set of atoms (nominally
  `source_atoms`) is guaranteed to be sorted.

### When to Use Each Format

**Neighbor Matrix** is preferred when:

- Using `torch.compile` or `jax.jit` (fixed memory layout avoids graph breaks)
- Systems have dense, uniform neighbor distributions
- Cache-friendly access patterns are important

**Neighbor List (COO)** is preferred when:

- Integrating with graph neural network libraries (PyG, DGL)
- Systems are sparse with highly variable neighbors per atom
- Memory efficiency is critical

### Switching Formats

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
# Get COO format directly
neighbor_list_coo, neighbor_ptr, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, return_neighbor_list=True
)

# Or convert from matrix format
from nvalchemiops.torch.neighbors.neighbor_utils import get_neighbor_list_from_neighbor_matrix

neighbor_list_coo, neighbor_ptr, shifts_coo = get_neighbor_list_from_neighbor_matrix(
    neighbor_matrix, num_neighbors, neighbor_matrix_shifts, fill_value=num_atoms
)
```

:::

:::{tab-item} JAX
:sync: jax

```python
# Get COO format directly
neighbor_list_coo, neighbor_ptr, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, return_neighbor_list=True
)

# Or convert from matrix format
from nvalchemiops.jax.neighbors.neighbor_utils import get_neighbor_list_from_neighbor_matrix

neighbor_list_coo, neighbor_ptr, shifts_coo = get_neighbor_list_from_neighbor_matrix(
    neighbor_matrix, num_neighbors, neighbor_matrix_shifts, fill_value=num_atoms
)
```

:::

::::

```{warning}
Setting `return_neighbor_list=True` incurs a conversion overhead. If you need
both formats, compute the matrix format first and convert as needed.
```

## Method Dispatch

### Method and Strategy

`method` is the high-level `neighbor_list(...)` selector. A family method such as
`method="naive"` or `method="cell_list"` chooses the neighbor-list algorithm family
and lets that family choose its direct strategy automatically. A strategy-pinned
method such as `method="naive_tile"` or `method="cell_list_pair_centric"` chooses
both the algorithm family and the direct strategy.

`neighbor_list(..., method="naive")` does not resolve to `"scalar"` or `"tile"` in
the high-level dispatcher. It forwards `strategy="auto"` to the direct naive
implementation, where the scalar/tile strategy is selected.

`cluster_tile` and `batch_cluster_tile` are complete high-level methods with a
single implementation. There is no `strategy` choice available for them.

`strategy` is only for direct algorithm calls such as `naive_neighbor_list(...)` or
`cell_list(...)`. For direct naive calls, `strategy` selects `"auto"`, `"scalar"`,
or `"tile"`. For direct cell-list calls, `strategy` selects `"auto"`,
`"atom_centric"`, or `"pair_centric"`.

Use `method=` when calling `neighbor_list(...)`. Use `strategy=` only when calling
a direct algorithm function.

Strategy-pinned high-level methods:

```python
neighbor_list(positions, cutoff, method="naive_tile")
neighbor_list(positions, cutoff, cell=cell, pbc=pbc, method="cell_list_pair_centric")
```

Direct algorithm strategy:

```python
naive_neighbor_list(positions, cutoff, strategy="tile")
cell_list(positions, cutoff, cell=cell, pbc=pbc, strategy="pair_centric")
```

When `method=None`, `neighbor_list` selects an algorithm using the following
logic:

1. If `cutoff2` is provided, choose the dual-cutoff naive method.
2. Otherwise, build per-system geometry (`batch_ptr`, `cell`, `pbc`); cell-less
   inputs synthesize a bounding-box cell purely for the cost estimate.
3. Compare the guarded geometry cost of `"naive"`, `"cell_list"`, and
   `"cluster_tile"` (the last only when its CUDA / float32 / fully-periodic
   guards pass); choose the lowest-cost feasible base method.
4. If `batch_idx` or `batch_ptr` is provided for more than one system, prepend
   `"batch_"` to the method.

The chosen method is honored as-is.  For a **cell-less** COO call
(`return_neighbor_list=True` with no `cell`) the return arity is therefore
method-dependent: `"naive"` returns a 2-tuple `(neighbor_list, neighbor_ptr)`
(non-periodic, no shifts), while `"cell_list"` synthesizes a non-PBC cell and
returns a 3-tuple `(neighbor_list, neighbor_ptr, shifts)` with zeroed shifts.
Pass an explicit `cell`+`pbc` (or use the matrix format) for a stable 3-tuple.

(estimating-and-running-a-strategy-explicitly)=

### Estimating and running a strategy explicitly

The same cost model is exposed as a public estimation API on all three backends
(`nvalchemiops.neighbors`, `nvalchemiops.torch.neighbors`,
`nvalchemiops.jax.neighbors`).  `estimate_neighbor_list_costs` returns every
feasible strategy with its relative estimated cost (lower is faster), sorted
cheapest-first, and `suggest_neighbor_list_method` returns just the top name:

```python
from nvalchemiops.torch.neighbors import (
    estimate_neighbor_list_costs,
    suggest_neighbor_list_method,
)

estimate_neighbor_list_costs(batch_ptr, cell, pbc, cutoff=6.0)
# -> [("cell_list_pair_centric", 2.1e6), ("naive_tile", 3.2e6), ...]

method = suggest_neighbor_list_method(batch_ptr, cell, pbc, cutoff=6.0)
# -> e.g. "cell_list_pair_centric"  (a "batch_..." name when num_systems > 1)
```

The strategy names are the fine-grained, directly-runnable paths
(`naive_tile`, `naive_scalar`, `cell_list_pair_centric`,
`cell_list_atom_centric`, `cluster_tile`, plus `batch_` variants).  `suggest`
and `estimate` **synchronize on the host** -- they launch a tiny selector kernel
on the device and read its result back, so call them outside `torch.compile` /
`jax.jit`.  The returned name is accepted directly as `method=`, so the compiled
neighbor build runs without a graph break:

```python
neighbor_list(positions, cutoff, cell=cell, pbc=pbc, method=method)
```

#### Interpreting the cost estimate

The costs are **relative**, in arbitrary units: only their ordering matters, so
compare them to each other, not to a wall-clock time.  Each value is a closed
form derived from the geometry (atom counts, cell volume, cutoff, periodic
images) that approximates the dominant kernel work for that strategy -- the
candidate pairs scanned, the neighbors written, and the per-launch overhead.
A strategy that fails a feasibility guard (for example `cluster_tile` on a
non-periodic or non-float32 input) is omitted from the result entirely rather
than returned with a large cost.

```{note}
The estimate is a hardware-independent model of *algorithmic* work; it does not
measure your GPU.  The true crossover between strategies shifts with the device
(memory bandwidth, occupancy, launch overhead), so on a given machine the
predicted best strategy may be marginally slower than a close runner-up.  The
ranking is reliable for the large gaps that matter (avoiding an \(O(N^2)\) blow-up
on a big system); for cases where the top costs are within a small factor,
benchmark the top few candidates on your target hardware and pass the winner as
`method=` explicitly.  Two calibration constants are env-overridable:
`NVALCHEMI_NEIGHLIST_CELL_SHELL` (default `27.0`, the cell-list neighbor-shell work
multiplier — roughly the `3x3x3` stencil of cells scanned per atom) and
`NVALCHEMI_NEIGHLIST_CELL_SETUP` (default `4096.0`, the cell-list build/setup cost
floor).  Raising `CELL_SETUP` biases the model toward `naive` for smaller systems;
lowering it favors `cell_list`.
```

### Available Methods

`method=` accepts the family method names below and the strategy-pinned method
names returned by `suggest_neighbor_list_method` / `estimate_neighbor_list_costs`.
Family methods resolve to a default direct strategy (`"naive"` → scalar,
`"cell_list"` → atom-centric); prefix any name with `batch_` for multi-system
batched inputs.

| Method | Algorithm | Use Case |
|--------|-----------|----------|
| `"naive"`, `"naive_scalar"` | \(O(N^2)\) scalar pairwise | Small single systems |
| `"naive_tile"` | \(O(N^2)\) tiled CUDA kernel | Small single systems on GPU |
| `"cell_list"`, `"cell_list_atom_centric"` | \(O(N)\) spatial decomposition, one thread per atom | Large single systems |
| `"cell_list_pair_centric"` | \(O(N)\) cell list, one thread per candidate pair | Large, high-parallelism systems |
| `"cluster_tile"` | Cluster-pair tile (CUDA, float32, fully periodic) | Large single systems on GPU |
| `"naive_dual_cutoff"` | \(O(N^2)\) with two cutoffs | Two-cutoff queries |
| `"batch_*"` | Per-system batched form of any of the above (e.g. `"batch_cell_list"`, `"batch_cluster_tile"`, `"batch_naive_dual_cutoff"`) | Batched systems |

Method names that do not start with `batch_` refer to single-system algorithms.
When `batch_idx` or `batch_ptr` (batch metadata) is supplied, those explicit
method names are treated as aliases for the corresponding `batch_*` methods.
For example, `method="naive"` is dispatched as `method="batch_naive"` when batch
metadata is provided.

Override automatic selection by passing the `method` parameter:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
# Force cell_list on a small system for testing
from nvalchemiops.torch.neighbors import neighbor_list

neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, method="cell_list"
)
```

:::

:::{tab-item} JAX
:sync: jax

```python
# Force cell_list on a small system for testing
from nvalchemiops.jax.neighbors import neighbor_list

neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, method="cell_list"
)
```

:::

::::

(naive-algorithm)=

## Naive Algorithm

The naive algorithm enumerates every $N(N-1)/2$ atom pair, computes the Euclidean
distance under the active periodic boundary conditions, and keeps pairs within the
cutoff. With no spatial data structure it has the lowest setup overhead, which makes
it the right choice for small single systems (illustratively below ~2000 atoms; the
cost model decides) and for batches of small heterogeneous systems via `batch_naive`.
It supports periodic boundaries (with or without pre-wrapped positions), half-fill,
inline pair-potential evaluation through `pair_fn`, and — through the separate
`naive_dual_cutoff` variant — dual cutoff.

(cell-list-algorithm)=

## Cell-List Algorithm

The cell-list algorithm bins atoms into spatial cells aligned to the simulation box
and enumerates pairs only between neighboring cells, scaling as $O(N)$ for roughly
uniform neighbor counts. It is the default when the cost model estimates it cheaper
than `naive`. It supports periodic boundaries with arbitrary (including triclinic)
cells, half-fill, partial lists (`target_indices`), and inline pair-potential
evaluation through `pair_fn`; build and query are separate launchers so the bin
structure can be cached across steps (see [Build/Query Separation](#build-query-separation)).
It has no
dual-cutoff variant — use `naive_dual_cutoff` for two-cutoff queries.

The query has two CUDA kernel strategies, `atom_centric` (default) and `pair_centric`,
chosen with `strategy="auto"` or pinned explicitly; both produce identical pair sets
(only per-row ordering differs) and are available on PyTorch and JAX. On JAX,
`pair_centric` is bound through `jax_callable`, sizes its launch from the host, and
requires `graph_mode="none"` (it raises under `jax.jit` with a traced radius; use
`atom_centric` there).

(cluster-pair-tile-algorithm)=

## Cluster-Pair Tile Algorithm

The cluster-pair tile algorithm is a CUDA-only build strategy that groups atoms into
Morton-sorted tiles and queries pairs cooperatively per tile, targeting large
fully-periodic float32 systems where the cell-list build overhead is unfavorable.
`neighbor_list(method=None)` auto-selects it when eligible; force it with
`method="cluster_tile"` / `"batch_cluster_tile"`.

It requires float32 positions on a CUDA device, a provided `cell` with `pbc` true on
all three axes, `half_fill=False`, and no `target_indices`; unsupported output
combinations raise a clear `ValueError` or `NotImplementedError`. Build and query are
separate launchers
({func}`~nvalchemiops.neighbors.cluster_tile.build_cluster_tile_list` /
{func}`~nvalchemiops.neighbors.cluster_tile.query_cluster_tile`, with bindings under
`nvalchemiops.{jax,torch}.neighbors.cluster_tile`), so the tiles can be cached across
steps; batched workflows accept `rebuild_flags` to re-enumerate only systems whose
atoms moved beyond the skin distance. Dual cutoff is supported in matrix format but
cannot be combined with pair-potential outputs.

(nl_performance)=

## Performance Tuning

### Key Parameters

`max_neighbors`
: Maximum neighbors per atom; determines the width of `neighbor_matrix`.
  Auto-estimated if not provided. Pass this value explicitly to `neighbor_list`
  calls if you have an accurate value to reduce memory requirements as well
  as improve kernel performance. The `estimate_max_neighbors()` method will
  otherwise provide a **very** conservative estimate based on atomic
  density.

`atomic_density`
: Atomic density in atoms per unit volume, used by `estimate_max_neighbors()`.
  Default is 0.2. Increase for dense systems to avoid truncated neighbor lists.

`safety_factor`
: Multiplier applied to the neighbor estimate. Default is 1.0. Provides
  headroom for density fluctuations.

`max_nbins`
: Maximum number of spatial cells for cell list decomposition (the
  `max_total_cells` cap). Defaults to 524288 for single systems and 8192 per system
  for batched inputs. Limits memory usage for very large simulation boxes.

`wrap_positions`
: Controls whether positions are wrapped into the primary cell before neighbor
  search. Default is `True`. Set to `False` when positions are already wrapped
  (e.g. after an integration step that keeps coordinates inside the box) to skip
  two GPU kernel launches per call.
  Only applies to naive methods; cell list methods handle wrapping internally.

`shift_range_per_dimension`, `num_shifts_per_system`, `max_shifts_per_system`
: Optional cached naive-PBC metadata for advanced workflows. Use
  `compute_naive_num_shifts()` to compute these values outside repeated calls,
  especially for JAX where `max_shifts_per_system` must be concrete outside
  `jax.jit`. Older `shift_offset` and `total_shifts` inputs are no longer part
  of the public Torch/JAX API.

### Estimation Utilities

The {func}`~nvalchemiops.neighbors.neighbor_utils.estimate_max_neighbors` function estimates
the maximum number of neighbors $n$ any atom could have based on the cutoff sphere
volume ($r$) and atomic density $\rho$, with an additional safety factor ($S$):

$$
n = S \times \rho \times \frac{4}{3} \pi r^3
$$

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.neighbors import estimate_cell_list_sizes

max_neighbors = estimate_max_neighbors(
    cutoff,
    atomic_density=0.15,
    safety_factor=1.0
)

max_total_cells, neighbor_search_radius = estimate_cell_list_sizes(
    cell, pbc, cutoff
)
```

:::

:::{tab-item} JAX
:sync: jax

```python
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.jax.neighbors import estimate_cell_list_sizes

max_neighbors = estimate_max_neighbors(
    cutoff,
    atomic_density=0.15,
    safety_factor=1.0
)

max_total_cells, neighbor_search_radius, _ = estimate_cell_list_sizes(
    positions, cell, cutoff, pbc=pbc, buffer_factor=1.5
)
```

```{note}
The JAX `estimate_cell_list_sizes` takes `positions` as its first argument
(to infer array sizes) and uses a `buffer_factor` parameter instead of
`max_nbins`. It also returns a 3-tuple.
This function is **not** compatible with `jax.jit` because it derives
concrete array sizes from traced data.
```

:::

::::

**Setting `atomic_density`**: This should reflect the expected atomic density of
your system in atoms per unit volume (using the same length units as `cutoff`).
If set too low, the neighbor matrix may be too narrow and a
`NeighborOverflowError` will be raised at runtime. If set too high, memory is
wasted on unused columns.

**Setting `safety_factor`**: This multiplier provides headroom for local density
fluctuations (e.g., atoms clustering in one region). The default of 1.0 is
typically sufficient for systems with reasonably uniform density (e.g. standard
public datasets). Increase it for systems with significant density variation
where atoms may cluster in one region.

```{tip}
Users should check the "convergence" of the neighbor list computation by checking
the respective array containing the number of neighbors per atom, against
the maximum estimated number of neighbors. For optimal performance these
two factors should be close: if the actual number of neighbors per atom is
low relative to the estimated number, the allocated neighbor matrix will
be very sparse and memory inefficient (i.e. most elements will be padding).
If the actual number exceeds the estimate, neighborhoods will be truncated
and there is no guarantee that the nearest neighbors are included.
```

### Pre-allocation for Repeated Calculations

Pre-allocating output arrays avoids repeated memory allocation overhead when
computing neighbor lists repeatedly across calls.

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

Pre-allocation also enables `torch.compile` compatibility by ensuring fixed
tensor shapes.

```python
import torch
from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors

num_atoms = positions.shape[0]
max_neighbors = estimate_max_neighbors(cutoff, atomic_density=0.15)

# Pre-allocate tensors
neighbor_matrix = torch.full(
    (num_atoms, max_neighbors), num_atoms, dtype=torch.int32, device="cuda"
)
neighbor_matrix_shifts = torch.zeros(
    (num_atoms, max_neighbors, 3), dtype=torch.int32, device="cuda"
)
num_neighbors = torch.zeros(num_atoms, dtype=torch.int32, device="cuda")

# Pass pre-allocated tensors
neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc,
    neighbor_matrix=neighbor_matrix,
    neighbor_matrix_shifts=neighbor_matrix_shifts,
    num_neighbors=num_neighbors,
    fill_value=num_atoms
)
```

For cell list methods, you can also pre-allocate the spatial data structures:

```python
from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.torch.neighbors.cell_list import estimate_cell_list_sizes
from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list

max_total_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)

(
    cells_per_dimension, neighbor_search_radius,
    atom_periodic_shifts, atom_to_cell_mapping,
    atoms_per_cell_count, cell_atom_start_indices, cell_atom_list
) = allocate_cell_list(num_atoms, max_total_cells, neighbor_search_radius, device)

neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc,
    cells_per_dimension=cells_per_dimension,
    neighbor_search_radius=neighbor_search_radius,
    atom_periodic_shifts=atom_periodic_shifts,
    atom_to_cell_mapping=atom_to_cell_mapping,
    atoms_per_cell_count=atoms_per_cell_count,
    cell_atom_start_indices=cell_atom_start_indices,
    cell_atom_list=cell_atom_list
)
```

:::

:::{tab-item} JAX
:sync: jax

JAX returns new arrays rather than mutating inputs in place. For fixed
`jax.jit` layouts, pass size controls such as `max_neighbors` and
`max_total_cells` as static ints; on APIs that accept caller-owned arrays, pass
pre-shaped arrays to define the returned buffer layout and allow XLA donation or
reuse. With `target_indices`, those arrays must have compact `num_targets` rows.

```python
from nvalchemiops.jax.neighbors import neighbor_list
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors

num_atoms = positions.shape[0]
max_neighbors = estimate_max_neighbors(cutoff, atomic_density=0.15)

# Pass max_neighbors (a static int) to fix the output width for jax.jit.
neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions,
    cutoff,
    cell=cell,
    pbc=pbc,
    max_neighbors=max_neighbors,
    fill_value=num_atoms,
)
```

For cell-list methods, also pass `max_total_cells` so the cell grid is statically
sized (derive it with `estimate_cell_list_sizes`):

```python
from nvalchemiops.jax.neighbors import estimate_cell_list_sizes, neighbor_list
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors

max_total_cells, _radius, _ = estimate_cell_list_sizes(
    positions, cell, cutoff, pbc=pbc
)
max_neighbors = estimate_max_neighbors(cutoff)

neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions,
    cutoff,
    cell=cell,
    pbc=pbc,
    method="cell_list",
    max_neighbors=max_neighbors,
    max_total_cells=max_total_cells,
)
```

:::

::::

```{warning}
If `max_neighbors` is too small, neighbors beyond that limit are silently
dropped. Monitor `num_neighbors.max()` (PyTorch) or `jnp.max(num_neighbors)`
(JAX) against your `max_neighbors` setting to detect truncation.
```

## Usage Patterns

### Basic Single System

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
import torch
from nvalchemiops.torch.neighbors import neighbor_list

# Create atomic system
positions = torch.rand(1000, 3, device="cuda") * 20.0
cell = torch.eye(3, device="cuda").unsqueeze(0) * 20.0
pbc = torch.tensor([True, True, True], device="cuda")
cutoff = 5.0

# Compute neighbors (automatic method selection)
neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc
)

print(f"Average neighbors: {num_neighbors.float().mean():.1f}")
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.neighbors import neighbor_list

# Create atomic system
key = jax.random.PRNGKey(0)
positions = jax.random.uniform(key, (1000, 3), dtype=jnp.float32) * 20.0
cell = jnp.eye(3, dtype=jnp.float32)[None, ...] * 20.0
pbc = jnp.array([[True, True, True]])
cutoff = 5.0

# Compute neighbors (automatic method selection)
neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc
)

print(f"Average neighbors: {jnp.mean(num_neighbors.astype(jnp.float32)):.1f}")
```

:::

::::

### Batch Processing

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
import torch
from nvalchemiops.torch.neighbors import neighbor_list

# Three systems of different sizes
positions = torch.cat([
    torch.rand(100, 3, device="cuda"),   # System 0
    torch.rand(150, 3, device="cuda"),   # System 1
    torch.rand(80, 3, device="cuda"),    # System 2
])

batch_idx = torch.cat([
    torch.zeros(100, dtype=torch.int32, device="cuda"),
    torch.ones(150, dtype=torch.int32, device="cuda"),
    torch.full((80,), 2, dtype=torch.int32, device="cuda"),
])

cells = torch.stack([
    torch.eye(3, device="cuda") * 10.0,
    torch.eye(3, device="cuda") * 12.0,
    torch.eye(3, device="cuda") * 8.0,
])

pbc = torch.tensor([
    [True, True, True],
    [True, True, False],
    [False, False, False],
], device="cuda")

neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff=5.0, cell=cells, pbc=pbc, batch_idx=batch_idx
)
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.neighbors import neighbor_list

# Three systems of different sizes
key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
positions = jnp.concatenate([
    jax.random.uniform(k1, (100, 3), dtype=jnp.float32),   # System 0
    jax.random.uniform(k2, (150, 3), dtype=jnp.float32),   # System 1
    jax.random.uniform(k3, (80, 3), dtype=jnp.float32),    # System 2
])

batch_idx = jnp.concatenate([
    jnp.zeros(100, dtype=jnp.int32),
    jnp.ones(150, dtype=jnp.int32),
    jnp.full((80,), 2, dtype=jnp.int32),
])

batch_ptr = jnp.array([0, 100, 250, 330], dtype=jnp.int32)

cells = jnp.stack([
    jnp.eye(3, dtype=jnp.float32) * 10.0,
    jnp.eye(3, dtype=jnp.float32) * 12.0,
    jnp.eye(3, dtype=jnp.float32) * 8.0,
])

pbc = jnp.array([
    [True, True, True],
    [True, True, False],
    [False, False, False],
])

neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff=5.0, cell=cells, pbc=pbc,
    batch_idx=batch_idx, batch_ptr=batch_ptr
)
```

:::

::::

### Half-Fill Mode

Store only half of neighbor pairs to avoid double-counting in symmetric
calculations:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
# Full: stores both (i,j) and (j,i)
neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, half_fill=False
)

# Half: stores only (i,j) where i < j (or with non-zero periodic shift)
neighbor_matrix_half, num_neighbors_half, shifts_half = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, half_fill=True
)

# half_fill=True produces ~50% of the pairs
```

:::

:::{tab-item} JAX
:sync: jax

```python
# Full: stores both (i,j) and (j,i)
neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, half_fill=False
)

# Half: stores only (i,j) where i < j (or with non-zero periodic shift)
neighbor_matrix_half, num_neighbors_half, shifts_half = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, half_fill=True
)

# half_fill=True produces ~50% of the pairs
```

```{note}
In JAX, `half_fill` and `fill_value` are supported by `naive`, `batch_naive`,
`cell_list`, and `batch_cell_list` (the cell-list paths use `graph_mode="none"`
for `half_fill`).  The `naive` tiled kernel (`strategy="tile"`) is
CUDA-only and opt-in; JAX `naive` auto-selection still uses the scalar kernel.
```

:::

::::

(build-query-separation)=

### Build/Query Separation

Separate building and querying allows caching the spatial data structure
across repeated calls when the cell-list bins remain valid:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
from nvalchemiops.torch.neighbors.cell_list import (
    build_cell_list, query_cell_list, estimate_cell_list_sizes
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list, estimate_max_neighbors
)

# Setup (once)
max_total_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)
cell_list_cache = allocate_cell_list(num_atoms, max_total_cells, neighbor_search_radius, device)

max_neighbors = estimate_max_neighbors(cutoff)
neighbor_matrix = torch.full((num_atoms, max_neighbors), -1, dtype=torch.int32, device=device)
neighbor_shifts = torch.zeros((num_atoms, max_neighbors, 3), dtype=torch.int32, device=device)
num_neighbors = torch.zeros(num_atoms, dtype=torch.int32, device=device)

# Repeated-query loop
for step in range(num_steps):
    # Build cell list (expensive, done when atoms change cells)
    build_cell_list(positions, cutoff, cell, pbc, *cell_list_cache)

    # Query neighbors (cheaper)
    neighbor_matrix.fill_(-1)
    neighbor_shifts.zero_()
    num_neighbors.zero_()
    query_cell_list(
        positions, cutoff, cell, pbc, *cell_list_cache,
        neighbor_matrix, neighbor_shifts, num_neighbors
    )

    forces = compute_forces(positions, neighbor_matrix, num_neighbors, ...)
    positions = integrate(positions, forces, dt)
```

:::

:::{tab-item} JAX
:sync: jax

```python
from nvalchemiops.jax.neighbors import (
    build_cell_list, query_cell_list, estimate_cell_list_sizes
)
from nvalchemiops.jax.neighbors.neighbor_utils import allocate_cell_list
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors

# Setup (once, outside jit)
max_total_cells, neighbor_search_radius, _ = estimate_cell_list_sizes(
    positions, cell, cutoff, pbc=pbc
)
cell_list_cache = allocate_cell_list(num_atoms, max_total_cells, neighbor_search_radius)

max_neighbors = estimate_max_neighbors(cutoff)

# Repeated-query loop (JAX returns new arrays each step; no in-place mutation)
for step in range(num_steps):
    # Build cell list (expensive, done when atoms change cells)
    cell_list_cache = build_cell_list(
        positions, cutoff, cell, pbc, *cell_list_cache
    )

    # Query neighbors (cheaper)
    (
        cells_per_dimension, neighbor_search_radius,
        atom_periodic_shifts, atom_to_cell_mapping,
        atoms_per_cell_count, cell_atom_start_indices, cell_atom_list
    ) = cell_list_cache

    neighbor_matrix, num_neighbors, neighbor_shifts = query_cell_list(
        positions, cutoff, cell, pbc,
        cells_per_dimension, atom_periodic_shifts, atom_to_cell_mapping,
        atoms_per_cell_count, cell_atom_start_indices, cell_atom_list,
        neighbor_search_radius, max_neighbors=max_neighbors
    )

    forces = compute_forces(positions, neighbor_matrix, num_neighbors, ...)
    positions = integrate(positions, forces, dt)
```

```{note}
JAX follows a functional paradigm: `build_cell_list` and `query_cell_list`
return new arrays rather than mutating buffers in-place. Reassign the
returned values each step.
```

:::

::::

### Rebuild Detection with Skin Distance

Avoid rebuilding neighbor lists every step by using a skin distance:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
from nvalchemiops.torch.neighbors.cell_list import (
    build_cell_list, query_cell_list, estimate_cell_list_sizes
)
from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list
from nvalchemiops.torch.neighbors.rebuild_detection import cell_list_needs_rebuild

cutoff = 5.0
skin_distance = 1.0
effective_cutoff = cutoff + skin_distance

# Build with effective cutoff (includes skin)
max_total_cells, neighbor_search_radius = estimate_cell_list_sizes(
    cell, pbc, effective_cutoff
)
cell_list_cache = allocate_cell_list(num_atoms, max_total_cells, neighbor_search_radius, device)

(
    cells_per_dimension, neighbor_search_radius,
    atom_periodic_shifts, atom_to_cell_mapping,
    atoms_per_cell_count, cell_atom_start_indices, cell_atom_list
) = cell_list_cache

build_cell_list(positions, effective_cutoff, cell, pbc, *cell_list_cache)

for step in range(num_steps):
    positions = integrate(positions, forces, dt)

    # Check if any atom moved to a different cell
    needs_rebuild = cell_list_needs_rebuild(
        positions, atom_to_cell_mapping, cells_per_dimension, cell, pbc
    )

    if needs_rebuild.item():
        build_cell_list(positions, effective_cutoff, cell, pbc, *cell_list_cache)

    # Query with actual cutoff (not effective)
    query_cell_list(positions, cutoff, cell, pbc, *cell_list_cache, ...)
```

:::

:::{tab-item} JAX
:sync: jax

```python
from nvalchemiops.jax.neighbors import (
    build_cell_list, query_cell_list, estimate_cell_list_sizes
)
from nvalchemiops.jax.neighbors.neighbor_utils import allocate_cell_list
from nvalchemiops.jax.neighbors.rebuild_detection import cell_list_needs_rebuild

cutoff = 5.0
skin_distance = 1.0
effective_cutoff = cutoff + skin_distance

# Build with effective cutoff (includes skin)
max_total_cells, neighbor_search_radius, _ = estimate_cell_list_sizes(
    positions, cell, effective_cutoff, pbc=pbc
)
cell_list_cache = allocate_cell_list(num_atoms, max_total_cells, neighbor_search_radius)

(
    cells_per_dimension, neighbor_search_radius,
    atom_periodic_shifts, atom_to_cell_mapping,
    atoms_per_cell_count, cell_atom_start_indices, cell_atom_list
) = cell_list_cache

cell_list_cache = build_cell_list(
    positions, effective_cutoff, cell, pbc, *cell_list_cache
)
(
    cells_per_dimension, neighbor_search_radius,
    atom_periodic_shifts, atom_to_cell_mapping,
    atoms_per_cell_count, cell_atom_start_indices, cell_atom_list
) = cell_list_cache

for step in range(num_steps):
    positions = integrate(positions, forces, dt)

    # Check if any atom moved to a different cell
    needs_rebuild = cell_list_needs_rebuild(
        positions, atom_to_cell_mapping, cells_per_dimension, cell, pbc
    )

    if needs_rebuild.item():
        cell_list_cache = build_cell_list(
            positions, effective_cutoff, cell, pbc, *cell_list_cache
        )
        (
            cells_per_dimension, neighbor_search_radius,
            atom_periodic_shifts, atom_to_cell_mapping,
            atoms_per_cell_count, cell_atom_start_indices, cell_atom_list
        ) = cell_list_cache

    # Query with actual cutoff (not effective)
    neighbor_matrix, num_neighbors, neighbor_shifts = query_cell_list(
        positions, cutoff, cell, pbc,
        cells_per_dimension, atom_periodic_shifts, atom_to_cell_mapping,
        atoms_per_cell_count, cell_atom_start_indices, cell_atom_list,
        neighbor_search_radius
    )
```

:::

::::

### Selective Rebuild (`rebuild_flags`)

In batched workflows, `rebuild_flags` re-enumerates only the systems that need a
fresh list and **preserves the previous output for the rest** — the skip happens on
the GPU with no host sync. Combine it with rebuild detection
(`batch_neighbor_list_needs_rebuild` / `batch_cell_list_needs_rebuild`) so only the
systems whose atoms crossed the skin distance are recomputed:

```python
from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.torch.neighbors.rebuild_detection import (
    batch_cell_list_needs_rebuild,
)

rebuild_flags = batch_cell_list_needs_rebuild(...)  # (num_systems,) bool

# Reuse the previous step's output buffers; only flagged systems are rewritten.
neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cells, pbc=pbc, batch_idx=batch_idx,
    rebuild_flags=rebuild_flags,
    neighbor_matrix=neighbor_matrix,
    num_neighbors=num_neighbors,
    neighbor_matrix_shifts=shifts,
)
```

Systems with `rebuild_flags[i] == False` keep their existing rows from the passed-in
buffers, so reuse the previous step's output arrays. Supported for matrix and
segmented-COO outputs in both the PyTorch and JAX `batch_naive` / `batch_cell_list`
paths (single-system paths take a whole-system flag of shape `(1,)`). It is not
combined with differentiable per-pair geometry.

### Dual Cutoff

Compute two neighbor lists with different cutoffs simultaneously:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
from nvalchemiops.torch.neighbors import neighbor_list

cutoff1, cutoff2 = 3.0, 6.0

(
    neighbor_matrix1, num_neighbors1, shifts1,
    neighbor_matrix2, num_neighbors2, shifts2
) = neighbor_list(
    positions, cutoff1, cutoff2=cutoff2, cell=cell, pbc=pbc
)

# neighbor_matrix1: neighbors within cutoff1
# neighbor_matrix2: neighbors within cutoff2 (superset of cutoff1)
```

:::

:::{tab-item} JAX
:sync: jax

```python
from nvalchemiops.jax.neighbors import neighbor_list

cutoff1, cutoff2 = 3.0, 6.0

(
    neighbor_matrix1, num_neighbors1, shifts1,
    neighbor_matrix2, num_neighbors2, shifts2
) = neighbor_list(
    positions, cutoff1, cutoff2=cutoff2, cell=cell, pbc=pbc
)

# neighbor_matrix1: neighbors within cutoff1
# neighbor_matrix2: neighbors within cutoff2 (superset of cutoff1)
```

:::

::::

### Partial Neighbor Lists (`target_indices`)

Pass `target_indices` (an `int32` array of atom indices) to build neighbors only for a
subset of *central* atoms. Output rows are **compact**: there are `num_targets` rows
and row `r` corresponds to atom `target_indices[r]`. In COO output the source index
`nl[0]` is the compact row in `[0, num_targets)` (map it back through `target_indices`):

```python
from nvalchemiops.torch.neighbors import neighbor_list

target_indices = torch.tensor([0, 5, 9], dtype=torch.int32, device="cuda")
nm, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, target_indices=target_indices,
)
# nm has 3 rows; row r holds the neighbors of atom target_indices[r].
```

Supported on the `naive` / `cell_list` paths and their batched forms across Warp,
PyTorch, and JAX, including low-level JAX cell-list query wrappers; `cluster_tile`
does not support `target_indices`. On JAX, `cell_list` `target_indices` runs through
the `atom_centric` strategy (`pair_centric` plus `target_indices` is rejected;
identical results are available via `atom_centric`).

### Per-Pair Distances and Vectors

Pass `return_distances=True` and/or `return_vectors=True` to get the per-pair
separation distances `|r_ij|` and displacement vectors `r_ij` alongside the neighbor
matrix, avoiding a manual recompute downstream. Each flag appends one array to the
return tuple, in the order *distances, then vectors*:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
from nvalchemiops.torch.neighbors import neighbor_list

nm, num_neighbors, shifts, distances, vectors = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc,
    return_distances=True, return_vectors=True,
)
# distances: (n_atoms, max_neighbors)      |r_ij| per slot
# vectors:   (n_atoms, max_neighbors, 3)   r_ij per slot
```

:::

:::{tab-item} JAX
:sync: jax

```python
from nvalchemiops.jax.neighbors import neighbor_list

nm, num_neighbors, shifts, distances, vectors = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc,
    return_distances=True, return_vectors=True,
)
```

:::

::::

The default matrix format returns `distances` with shape `(n_atoms, max_neighbors)`
and `vectors` with shape `(n_atoms, max_neighbors, 3)`, slot-aligned with
`neighbor_matrix`. With the COO format (`return_neighbor_list=True`) the `naive` and
`cell_list` paths repack them into flat per-pair arrays `(num_pairs,)` and
`(num_pairs, 3)` that index-align with the returned neighbor list. The returned
`distances` / `vectors` are differentiable with respect to `positions` (and `cell`)
on both the PyTorch and JAX paths (each emitted pair's geometry is reconstructed
live from its indices and shift), so they can flow straight into a loss without
re-deriving geometry.

### Inline Pair Potentials with `pair_fn`

Supply a Warp `pair_fn` to evaluate a pairwise potential *as neighbors are enumerated*,
filling `pair_energies` / `pair_forces` in the same pass — no second loop over the
neighbor list. `pair_fn` is a `wp.Function` taking the separation vector, distance, a
per-atom `pair_params` table, and the pair indices, and returning `(energy, force)`:

```python
import warp as wp

@wp.func
def lj_pair_fn(
    r_ij: wp.vec3f,
    distance: wp.float32,
    pair_params: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
):
    epsilon = wp.sqrt(pair_params[i, 0] * pair_params[j, 0])
    sigma = 0.5 * (pair_params[i, 1] + pair_params[j, 1])
    sr = sigma / distance
    sr2 = sr * sr
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    energy = 4.0 * epsilon * (sr12 - sr6)
    force = (24.0 * epsilon * (sr6 - 2.0 * sr12) / (distance * distance)) * r_ij
    return energy, force
```

Pass `pair_fn` with its per-atom `pair_params` table. The `pair_energies` /
`pair_forces` outputs are **optional**: like `neighbor_matrix`, they are allocated for
you when omitted and appended to the return tuple — matrix-shaped in matrix output, or
flat COO `(num_pairs,)` / `(num_pairs, 3)` aligned with the neighbor list when
`return_neighbor_list=True`. (If you do pass buffers, they are also filled in place.)
See `examples/neighbors/06_pair_outputs_lj.py` for a complete, validated
Lennard-Jones example, including combination with `target_indices`.

```{note}
`pair_fn` is supported on the **Warp, PyTorch, and JAX** paths — `naive`, `cell_list`,
`cluster_tile`, and their batched forms.  The JAX bindings build a per-`pair_fn`
callable at call time that closes over the `wp.Function` (cached by `pair_fn`
identity): a `jax_kernel` over the specialized naive / cell-list kernel, and a
`jax_callable` over the Warp `query_cluster_tile` launcher for the tile paths.
cluster-tile pair outputs are fp32-only and support both matrix and COO output
(COO packs the matrix result and is eager-only — its pair count is data-dependent,
so a traced call raises; use `format="matrix"` under `jax.jit`).  `pair_energies` /
`pair_forces` are **forward-only**
outputs (the Warp kernels are registered with `enable_backward=False`); use
`return_distances` / `return_vectors` for differentiable geometry. Differentiating a
loss through `pair_energies` / `pair_forces` returns a **zero** gradient under JAX
(they are `stop_gradient`'d). Under JAX the energy/force buffers are always
auto-allocated and returned (functional arrays cannot be filled in place), and — like
the differentiable-geometry path — a traced (jit'd) cutoff is not yet supported.

On JAX, `naive` / `batch_naive` and `cell_list` / `batch_cell_list` support
`target_indices` (partial neighbor lists) combined with pair outputs: the
compact output has `num_targets` rows (row `r` → atom `target_indices[r]`), and
in COO mode the source index `nl[0]` is the compact row in `[0, num_targets)`
(mapped back via `target_indices`), matching the Torch contract.

For PyTorch `torch.compile(fullgraph=True)`, pass a pre-specialized wrapper from
`nvalchemiops.torch.neighbors.compile_pair_fn(pair_fn)` instead of the raw
`wp.Function`. The compiled wrapper registers fixed-shape matrix custom ops for
Torch `naive`, `batch_naive`, `cell_list`, and `batch_cell_list`, including
compact `target_indices` rows on the naive paths. Raw `wp.Function` pair outputs
remain eager-only under fullgraph, and COO pair-output packing remains outside
the compiled matrix path.
```

---

This concludes the high-level documentation for neighbor lists: you should now
be able to integrate `nvalchemiops` routines for your neighbor list requirements,
and consult the API reference for [PyTorch](../../modules/torch/neighbors)
, [JAX](../../modules/jax/neighbors), and [Warp](../../modules/warp/neighbors) for further details.

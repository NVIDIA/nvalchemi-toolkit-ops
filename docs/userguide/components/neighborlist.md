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

- **Naive algorithms scale as $O(N^2)$**: Checking all atom pairs becomes
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

Dispatches to {func}`~nvalchemiops.torch.neighbors.cell_list` --- $O(N)$ algorithm
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

Dispatches to {func}`~nvalchemiops.torch.neighbors.naive_neighbor_list` --- $O(N^2)$
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

Dispatches to {func}`~nvalchemiops.torch.neighbors.batch_cell_list` --- $O(N)$
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
$O(N^2)$ algorithm for batched small systems.
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

Dispatches to {func}`~nvalchemiops.jax.neighbors.cell_list` --- $O(N)$ algorithm
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

Dispatches to {func}`~nvalchemiops.jax.neighbors.naive_neighbor_list` --- $O(N^2)$
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

Dispatches to {func}`~nvalchemiops.jax.neighbors.batch_cell_list` --- $O(N)$
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
$O(N^2)$ algorithm for batched small systems.
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
$O(N^2)$ path.  Auto-dispatch also considers `cluster_tile` for feasible
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

For JAX, compile a method-specific neighbor function with this fixed matrix
layout. The unified `neighbor_list(...)` dispatcher is eager. Compact COO output
has a data-dependent pair count and is produced eagerly. Direct naive and
cell-list APIs accept `coo_capacity` for padded fixed-capacity COO with raw-count
recovery metadata. Cluster-tile APIs expose fixed segmented COO state.

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

```{note}
With PyTorch >=2.10, exact COO conversion supports
`torch.compile(fullgraph=True)` when the edge count changes. Exact sizing via
`nonzero` may synchronize the host. Capacity overflow raises
`NeighborOverflowError` in eager execution and an asynchronous runtime error
in compiled execution.
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
`jax.jit`. On JAX, use the result to select the corresponding direct method and
close the resulting strategy over a compiled wrapper.
`neighbor_list(..., method=method)` is an eager convenience call:

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
ranking is reliable for the large gaps that matter (avoiding an $O(N^2)$ blow-up
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
| `"naive"`, `"naive_scalar"` | $O(N^2)$ scalar pairwise | Small single systems |
| `"naive_tile"` | $O(N^2)$ tiled CUDA kernel | Small single systems on GPU |
| `"cell_list"`, `"cell_list_atom_centric"` | $O(N)$ spatial decomposition, one thread per atom | Large single systems |
| `"cell_list_pair_centric"` | $O(N)$ cell list, one thread per candidate pair | Large, high-parallelism systems |
| `"cluster_tile"` | Cluster-pair tile (CUDA, float32, fully periodic) | Large single systems on GPU |
| `"naive_dual_cutoff"` | $O(N^2)$ with two cutoffs | Two-cutoff queries |
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
(only per-row ordering differs) and are available on PyTorch and JAX. On the fast
path, `pair_centric` schedules one CUDA block per `(source_cell, neighbor offset)`;
when that uncoarsened launch would exceed the Warp one-dimensional limit, the
launcher transparently coarsens multiple logical blocks per CUDA block under the
same `pair_centric` strategy. On JAX,
`pair_centric` is bound through `jax_callable`, sizes its launch from the host, and
requires `graph_mode="none"`. Compiled single-system calls provide
`pair_centric_n_outer`; compiled batched calls provide `pair_centric_total_cells`,
`pair_centric_n_outer`, and `pair_centric_r_max`.

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

(cluster-tile-buffer-capacity)=

### Tile-buffer capacity

Cluster-tile construction divides each system into groups of at most 32 atoms.
A system with $N$ atoms has $\lceil N/32\rceil$ groups. Groups do not
cross system boundaries, so a compact batch containing systems of sizes $N_i$
has $g=\sum_i\lceil N_i/32\rceil$ groups in total.

The build stores discovered tile pairs in one buffer shared by all row groups.
If `max_tiles_per_group` is $m$, a single-system build with $g$ groups reserves
$C=g\,\min(g,m)$ records. A compact Torch batch reserves

$$
C=\sum_i g_i\,\min(g_i,m)
$$

records in one buffer pooled across all systems. This formula determines only
the total capacity; it does not impose per-system quotas. Segmented batches use
the same per-system terms but assign each system a fixed interval. Compact JAX
batches retain their fixed-shape $C=G\,\min(G,m)$ allocation, where
$G=\sum_i g_i$. Each record contains two `int32` group indices, so the
tile-index arrays use $8C$ bytes for one system. A compact batch also records
the system index and uses $12C$ bytes. Other scratch buffers and the neighbor
output do not depend on $m$.

#### Choosing a capacity

`cluster_tile_neighbor_list` and `batch_cluster_tile_neighbor_list` construct
the tile list and query the neighbor output in the same call. During eager
execution, they call
{func}`~nvalchemiops.neighbors.cluster_tile.estimate_max_tiles_per_group` when
`max_tiles_per_group` is `None`. The estimator uses each system's atom count,
cell volume, and cutoff; its `safety` parameter adds headroom for uneven density
or changing geometries.

Available capacity choices are:

- Before execution, use the estimator for a heuristic based on the current
  geometry.
- After an eager overflow, use the reported tile count to calculate the exact
  requirement for that geometry.
- For a geometry-independent single-system bound, require capacity of at least
  $g(g+1)/2$ and use `max_tiles_per_group=ceil((g + 1) / 2)`.
- For a compact Torch batch, the conservative geometry-independent shared
  factor $m=\max_i\lceil(g_i+1)/2\rceil$ gives every system enough contribution
  for its dense upper triangle while keeping the resulting buffer pooled.
- For a compact JAX batch, require total capacity of at least
  $\sum_i g_i(g_i+1)/2$. With $G=\sum_i g_i$, the minimum shared factor is
  $\left\lceil\sum_i g_i(g_i+1)/(2G)\right\rceil$ for a nonempty batch.
- For a segmented batch, require each segment to hold at least
  $g_i(g_i+1)/2$ records. A geometry-independent shared factor is
  $\max_i\lceil(g_i+1)/2\rceil$.

Dual-cutoff sizing uses the larger of `cutoff` and `cutoff2`. Use that outer
cutoff when calling the estimator directly. JAX cluster-tile APIs require
`cutoff2 >= cutoff`, while Torch cluster-tile APIs accept either ordering.

#### Recovering from eager overflow

`TileBufferOverflow` reports the required tile-pair count as
`error.num_tiles` and the allocated capacity as `error.max_tiles`. For a compact
single-system build, the exact retry value for that geometry is

$$
m_{\mathrm{retry}} = \left\lceil\frac{\mathtt{error.num\_tiles}}{g}\right\rceil,
$$

where $g$ is the system's group count. For a compact Torch batch, choose the
smallest positive integer $m$ satisfying

$$
\sum_i g_i\,\min(g_i,m) \ge \mathtt{error.num\_tiles}.
$$

The capacity remains pooled: one system may consume more than its individual
term as long as the batch's total count fits. For a compact JAX batch, use the
single-buffer retry
$m_{\mathrm{retry}}=\lceil\mathtt{error.num\_tiles}/G\rceil$ with the total
group count $G$.

A segmented batch gives each system its own interval in the tile buffer. System
$i$ has capacity `tile_offsets[i + 1] - tile_offsets[i]` and reports its required
count in `tile_counts[i]`. An eager `TileBufferOverflow` identifies only the
first overflowing system and its required count. Resize that segment or increase
the shared factor and retry; another system can fail next. Once all counts are
available, for example from a compiled lower-level build, the exact shared value
for that geometry is

$$
m_{\mathrm{retry}} =
\max_{i:\,g_i>0}\left\lceil\frac{\mathtt{tile\_counts}[i]}{g_i}\right\rceil.
$$

If resizing changes `tile_offsets`, initialize replacement state and mark every
system for rebuild. State laid out with the old offsets cannot be retained under
the new layout. For a trajectory, retain the largest observed requirement and
add headroom for later geometries. `TileBufferOverflow` reports exhaustion of
the intermediate tile-pair buffer. `NeighborOverflowError` reports that the
final matrix or COO neighbor output is too small.

#### Compiled PyTorch direct APIs

The direct PyTorch cluster-tile functions support
`torch.compile(fullgraph=True)` for tile, matrix, and nonselective exact COO
output. Matrix support includes dual cutoffs. Matrix and exact COO vectors and
distances remain differentiable. Exact COO requires PyTorch 2.10 or newer and
may return a different pair count on each call. Pair callbacks remain
eager-only.

Exact COO is counted and written directly into source-owned CSR rows without a
matrix intermediate. For source atom `i`, entries
`neighbor_ptr[i]:neighbor_ptr[i + 1]` in `neighbor_list` all belong to `i`.
Atomic writes leave pair order within each row unspecified. Shifts and any
requested vectors, distances, energies, or forces use the same pair order.
Topology, shifts, and requested geometry are returned with the exact active
pair length and do not alias reusable capacity buffers. Compact calls append
distances before vectors: requesting distances returns
`(pairs, ptr, shifts, distances)`, requesting vectors returns
`(pairs, ptr, shifts, vectors)`, and requesting both returns
`(pairs, ptr, shifts, distances, vectors)`. Optional caller-owned geometry
buffers hold the active prefix and may be reused; their inactive tails are
unspecified. These buffers are non-differentiable value snapshots and must not
require gradients; build losses from the returned exact geometry. Pair
callbacks keep their existing topology-only return and write aligned callback
outputs into caller-owned buffers.

A compiled single-system call may allocate its scratch internally when
`max_tiles_per_group` is a positive static integer. Batched compiled calls
should allocate once with `allocate_batch_cluster_tile_list` and pass the
caller-owned scratch tuple; selective matrix calls additionally require fixed
output buffers and tile segment metadata. Eager calls retain structured
`TileBufferOverflow` and `NeighborOverflowError` exceptions. Compiled capacity
failures are asynchronous device runtime errors.

```python
import torch

from nvalchemiops.torch.neighbors import cluster_tile_neighbor_list


@torch.compile(fullgraph=True)
def compiled_matrix(positions, cell):
    return cluster_tile_neighbor_list(
        positions,
        cutoff,
        cell,
        format="matrix",
        max_neighbors=max_neighbors,
        max_tiles_per_group=max_tiles_per_group,
        return_distances=True,
    )
```

#### Prepared PyTorch execution

Use `prepare_cluster_tile` when repeated calls have the same atom count,
single or batched partition, dtype, device, output format, and capacities.
Preparation owns the fixed-capacity scratch and output buffers. Execution uses
the current positions and cell through the matching direct API with the
prepared state:

```python
import torch

from nvalchemiops.torch.neighbors import (
    cluster_tile_neighbor_list,
    prepare_cluster_tile,
)

state = prepare_cluster_tile(
    positions,
    cutoff,
    cell,
    format="matrix",
    max_neighbors=max_neighbors,
    max_tiles_per_group=max_tiles_per_group,
    return_distances=True,
)

@torch.compile(fullgraph=True)
def compiled_neighbors(current_positions, current_cell):
    # Capture state as a closure constant; do not pass it as a graph input.
    return cluster_tile_neighbor_list(
        current_positions,
        cell=current_cell,
        state=state,
    )
```

For a batched state, pass `batch_ptr` to `prepare_cluster_tile` and execute it
with `batch_cluster_tile_neighbor_list(..., cell_batch=current_cells, state=state)`.

Prepared execution supports single and batched tile, matrix, dual-cutoff
matrix topology, and nonselective exact COO output. Dual-cutoff prepared state
does not support vectors or distances. Preparation rejects that combination
before allocating storage. The other formats preserve the corresponding direct
function's return tuple. Matrix topology and tile results borrow state-owned
storage and a later call overwrites them. State-owned matrix geometry buffers
are also borrowed, non-differentiable snapshots. When autograd reconstruction
is needed, the function returns fresh differentiable geometry and writes
matching detached values to those buffers; build losses from the returned
geometry. Without reconstruction, returned matrix geometry aliases the state
buffers. Exact COO topology, shifts, and requested geometry are newly sized on
each call. Reusable exact-COO capacity buffers are available as
`state.neighbor_vectors` and `state.neighbor_distances`; only the active prefix
matching the returned pair count is defined, and it is a detached snapshot of
the returned geometry.

Preparation avoids reallocating the fixed scratch and output buffers, but
execution may still allocate temporary tensors and exact-sized COO results. It
is not an allocation-free API. Finish backward before reusing the same state,
and copy every borrowed result that must survive that reuse. A second state owns
distinct storage.

Preparation fixes the atom count, batch partition, shape, dtype, and device.
For batches, it caches atom/system and padded-layout mappings derived only from
that partition. Morton ordering, sorted coordinates, cell inverses, and group
bounds are recomputed from the current positions and cells for nonselective
executions and selected systems in selective executions. An all-false eager or
ordinary compiled selective call may preserve them without rebuilding.
Execution rejects mismatches before launching kernels. Prepared pair callbacks,
energies, forces, and caller-provided buffers are not supported.

`ClusterTileState` is prepared configuration and reusable borrowed storage, not
the neighbor-list result. Each call returns the same tuple as the corresponding
unprepared method-specific function. With `state=`, the call ignores `cutoff`,
`cutoff2`, `format`, `max_neighbors`, `max_pairs`, `fill_value`,
`max_tiles_per_group`, `return_vectors`, `return_distances`, and `pair_fn`, even
when they differ from the prepared configuration. The batched API also ignores
`batch_ptr`. Change these settings by preparing another state. `positions` and
`cell` or `cell_batch` remain required on every call. `rebuild_flags` remains
active for selective states. Prepared pair callbacks and `pair_params` are not
supported. Supplying explicit scratch, output, segment, or inverse-cell buffers
raises `ValueError`, as does `return_state=True`.

Set `selective=True` during preparation to rebuild matrix topology only for
selected systems. Selective prepared execution supports single and batched
matrix output, including dual cutoffs. It does not support tile or COO output,
vectors, distances, or pair callbacks. Each execution requires a Boolean
`rebuild_flags` tensor on the prepared device with one value per system. A true
flag rebuilds that system. A false flag preserves its initialized neighbor
matrix, counts, and shifts byte-for-byte; preserving a system before its first
successful rebuild raises an error. Outside CUDA Graph capture, an all-false
eager or ordinary compiled call preserves all existing state and may skip the
inverse, sorting, and build work.

An eager call marks every selected system uninitialized before rebuilding it
and marks it initialized only after the complete call succeeds. If an eager
rebuild fails, for example because the matrix capacity is too small, a later
call cannot preserve any system selected by the failed call. Retry those
systems with true flags after changing the geometry, or prepare a new state
with sufficient capacity.

##### CUDA Graph capture

A warmed `torch.compile(fullgraph=True)` prepared callable can be captured with
`torch.cuda.CUDAGraph` when it returns matrix topology. This supports single and
batched state, one or two cutoffs, and selective or nonselective execution. For
selective state, initialize every system before capture. Warm the compiled
all-true path on a side stream and synchronize that stream before capture.

Keep the prepared state object and the positions, cells, rebuild flags, output,
and scratch tensors at the same addresses. Shapes, dtypes, devices, capacities,
the batch partition, and the output format must remain fixed. Update positions,
cells, and flags by copying into the existing tensors. Do not execute or replay
the same mutable state concurrently.

Selective replay accepts different flag values without recapture. The captured
graph always records the complete inverse, Morton sort, metadata update,
tile-build, query, and tail sequence; device-side flags decide which systems are
updated on each replay. Therefore an all-false replay preserves topology but
does not skip sorting or reduce the recorded launch sequence. Outside graph
capture, eager all-false calls return immediately and ordinary compiled
all-false calls skip the build after reading the flags on the host.

Capture is not supported for a direct eager prepared call, exact COO output,
tile output, pair geometry, or backward execution. A device assertion during
replay does not leave the prepared state recoverable. CUDA Graph-private
temporary allocations may occur; the guarantee is stable prepared/public
storage and no forbidden host synchronization during capture.

#### Compiled JAX

JAX fixes array shapes while tracing a transformed or compiled function. When a
call allocates either single-system tile-index array, or any of the batched row,
column, and system arrays, `max_tiles_per_group` determines their shapes and must
be a positive static Python integer. Close over it or mark the argument static
with `static_argnames`. Complete caller-supplied tile-index storage already
fixes capacity and does not require the factor. Omitting only the fixed-size
`num_tiles` buffer does not change that rule. An explicitly supplied factor is
always validated, but it never resizes caller-owned arrays.

The runtime tile count cannot be converted to a Python value inside the compiled
region, so an undersized bound does not raise `TileBufferOverflow` there.
Ordinary compiled convenience calls should therefore use a known-sufficient
bound; the geometry-independent bounds above are the safest default. An
adaptive workflow should compile the lower-level build, return its tile
counters, and check capacity on the host before invoking the query:

- For a compact buffer, require
  `int(num_tiles[0]) <= tile_row_group.shape[0]`.
- For segmented buffers, require
  `tile_counts <= tile_offsets[1:] - tile_offsets[:-1]` element by element.

Do not consume neighbor output when either check fails. `num_neighbors` reports
the final-output requirement and cannot detect tile pairs omitted by an
undersized intermediate buffer.

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

`max_tiles_per_group`
: Sets the capacity of the tile-pair buffer shared by all row groups. For $g$
  32-atom groups, a value $m$ reserves $g\,\min(g,m)$ records. The tile-index
  arrays use 8 bytes per record for one system and 12 bytes per record for a
  compact batch. The combined build/query functions estimate the value during
  eager execution when they allocate storage and it is `None`. A transformed
  or compiled JAX call that allocates tile-index storage requires a positive
  static Python integer; complete caller-supplied tile-index arrays do not.

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
: Naive-PBC launch metadata computed by `compute_naive_num_shifts()`. All three
  values are required for every JAX PBC call under `jax.jit`; eager calls may
  compute them internally. They must correspond to the call's `cell`, `pbc`,
  and static cutoff. Recompute the values and specialize the compiled function
  when any of those inputs changes.

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

max_total_cells, _cells_per_dimension, neighbor_search_radius = estimate_cell_list_sizes(
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
If set too low, the neighbor matrix may be too narrow. Matrix output keeps its
fixed width but reports the required per-atom counts, which callers must compare
with that width before consuming the result. Eager compact COO conversion raises
`NeighborOverflowError` when those counts exceed the matrix width. Fixed-capacity
COO instead returns the raw required count for each row plus a scalar
`metadata_valid` flag; its pointer describes only the stored prefix. When
metadata is valid, compare each pointer difference with its raw count to find
incomplete rows. When metadata is invalid, every returned count is `-1` and the
launch metadata must be refreshed before retrying. These rules apply to the
naive and cell-list matrix/COO outputs. Cluster-tile methods use their documented
tile and segmented-COO capacity contracts. If `atomic_density` is set too high,
memory is wasted on unused columns.

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

JAX returns new arrays rather than mutating inputs in place. The unified
`neighbor_list(...)` API is eager: it may choose a method, allocate, or inspect
host values. For `jax.jit`, call an existing method-specific function with
static allocation controls and fixed-shape buffers. With `target_indices`, those
arrays must have compact `num_targets` rows.

```python
import jax
import jax.numpy as jnp

from nvalchemiops.jax.neighbors import (
    compute_naive_num_shifts,
    naive_neighbor_list,
)
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors

num_atoms = positions.shape[0]
max_neighbors = estimate_max_neighbors(cutoff, atomic_density=0.15)
shift_range, num_shifts, max_shifts = compute_naive_num_shifts(cell, cutoff, pbc)

neighbor_matrix = jnp.full((num_atoms, max_neighbors), num_atoms, dtype=jnp.int32)
num_neighbors = jnp.zeros((num_atoms,), dtype=jnp.int32)
neighbor_matrix_shifts = jnp.zeros(
    (num_atoms, max_neighbors, 3), dtype=jnp.int32
)


@jax.jit
def compiled_naive(positions, neighbor_matrix, num_neighbors, shifts):
    return naive_neighbor_list(
        positions,
        cutoff,
        cell=cell,
        pbc=pbc,
        max_neighbors=max_neighbors,
        neighbor_matrix=neighbor_matrix,
        num_neighbors=num_neighbors,
        neighbor_matrix_shifts=shifts,
        shift_range_per_dimension=shift_range,
        num_shifts_per_system=num_shifts,
        max_shifts_per_system=max_shifts,
    )

neighbor_matrix, num_neighbors, shifts = compiled_naive(
    positions,
    neighbor_matrix,
    num_neighbors,
    neighbor_matrix_shifts,
)

# A count larger than the matrix width means the caller must grow capacity and
# recompile from eager code.
assert int(jnp.max(num_neighbors)) <= max_neighbors
```

This wrapper closes over `cell`, `pbc`, `cutoff`, and the shift metadata derived
from them. If the cell or boundary conditions change, call
`compute_naive_num_shifts()` again outside `jax.jit` and create a specialization
with the matching values.

For cell-list methods, estimate capacity outside JIT and call `cell_list`
directly. Atom-centric queries need the fixed allocation values. Pair-centric
queries additionally need a static launch size derived from the same concrete
search radius:

```python
from nvalchemiops.jax.neighbors import (
    cell_list,
    compute_batch_pair_centric_n_outer,
    estimate_cell_list_sizes,
)
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors

max_total_cells, _cells_per_dimension, neighbor_search_radius = estimate_cell_list_sizes(
    positions, cell, cutoff, pbc=pbc
)
max_neighbors = estimate_max_neighbors(cutoff)
launch_radius = tuple(int(value) for value in neighbor_search_radius)
pair_centric_n_outer = compute_batch_pair_centric_n_outer(launch_radius, False)


@jax.jit
def compiled_cell_list(positions):
    return cell_list(
        positions,
        cutoff,
        cell,
        pbc,
        max_neighbors=max_neighbors,
        max_total_cells=max_total_cells,
        neighbor_search_radius=neighbor_search_radius,
        strategy="pair_centric",
        pair_centric_n_outer=pair_centric_n_outer,
    )


neighbor_matrix, num_neighbors, shifts = compiled_cell_list(positions)
```

The cell-list wrapper likewise closes over the geometry used for capacity and
radius estimation. A search radius can instead be a runtime JAX array, but it
must describe the current cell grid; pair-centric fixed-COO calls report whether
their static launch metadata still matches through `metadata_valid`.

For `batch_cell_list`, compute the corresponding metadata from the arrays
returned by `estimate_batch_cell_list_sizes`: `pair_centric_total_cells` is the
sum of `prod(cells_per_dimension, axis=1)`, `pair_centric_r_max` is the
per-axis maximum search radius, and `pair_centric_n_outer` is computed from that
maximum. Close these exact values over the compiled call.

The static batch values must describe one safe launch:
`pair_centric_n_outer` must match `pair_centric_r_max`, and
`pair_centric_total_cells` cannot exceed the allocated cell-list capacity. These
relationships are checked before the pair-centric CUDA query is launched. The actual
cell count and search radius are runtime JAX arrays, so a compiled call can receive
geometry that no longer matches its static launch metadata. For fixed COO that
invalidates the whole launch's count metadata: `metadata_valid` is false and
every returned count is `-1`. Leave the compiled region, recompute sizing
metadata for the new geometry, and compile or retry with those values.

Fixed-capacity COO uses the same direct methods. The returned arrays keep a
static leading capacity; `neighbor_ptr[-1]` is clipped to that capacity, and
the raw row counts tell the eager caller whether to grow `max_neighbors`,
`coo_capacity`, or both. `metadata_valid` describes launch-metadata completeness
for this call only; it does not certify initialization or coordinate freshness
of caller-retained buffers, and it is not a persistent or sticky prepared-state
validity flag:

```python
coo_capacity = num_atoms * max_neighbors


@jax.jit
def compiled_coo(positions):
    return naive_neighbor_list(
        positions,
        cutoff,
        cell=cell,
        pbc=pbc,
        max_neighbors=max_neighbors,
        return_neighbor_list=True,
        coo_capacity=coo_capacity,
        shift_range_per_dimension=shift_range,
        num_shifts_per_system=num_shifts,
        max_shifts_per_system=max_shifts,
    )


(
    neighbor_list_coo,
    neighbor_ptr,
    shifts_coo,
    num_neighbors,
    metadata_valid,
) = compiled_coo(
    positions,
)
if not bool(metadata_valid):
    raise RuntimeError("refresh pair-centric launch metadata outside jax.jit")
stored_counts = neighbor_ptr[1:] - neighbor_ptr[:-1]
if bool(jnp.any(stored_counts != num_neighbors)):
    required_max_neighbors = int(jnp.max(num_neighbors, initial=0))
    required_coo_capacity = int(jnp.sum(num_neighbors))
    raise RuntimeError(
        f"grow neighbor capacity outside jax.jit; max_neighbors must be at "
        f"least {required_max_neighbors} and coo_capacity must be at least "
        f"{required_coo_capacity}"
    )
num_pairs = int(neighbor_ptr[-1])
neighbor_list_coo = neighbor_list_coo[:, :num_pairs]
shifts_coo = shifts_coo[:num_pairs]
```

For batched full-row output, row `r` belongs to `batch_idx[r]`. With
`target_indices`, row `r` instead belongs to `batch_idx[target_indices[r]]`.
Group raw row counts by that ownership: the per-system matrix-width requirement
is the maximum owned-row count, and the per-system pair requirement is their
sum. A globally packed retry still needs `sum(num_neighbors)` COO columns.
Pointer differences retain mid-row truncation, including the case where one
system stores a complete row and the next stores only part of a row.

The naive dual-cutoff APIs return two complete fixed-COO groups, each with
independent counts and validity. They require the second cutoff to be greater
than or equal to the first. Cluster-tile dual-matrix calls instead retain their
matrix contract and require `cutoff2 >= cutoff`.

Treat `cutoff` as a static specialization input: pass a Python scalar closed
over the compiled function, and specialize another function when the cutoff
changes. Search-radius arrays can be JAX arrays because kernels consume them as
device data, while their allocation and pair-centric launch metadata are fixed
outside `jax.jit`.

:::

::::

```{warning}
If `max_neighbors` is too small, entries beyond the matrix width cannot be returned,
but `num_neighbors` retains the required count. Monitor `num_neighbors.max()`
(PyTorch) or `jnp.max(num_neighbors)` (JAX) against `max_neighbors` and retry from
eager code when the capacity is insufficient.
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
max_total_cells, _cells_per_dimension, neighbor_search_radius = estimate_cell_list_sizes(
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
max_total_cells, _cells_per_dimension, neighbor_search_radius = estimate_cell_list_sizes(
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

Cluster-tile selective rebuilds also require the tile-list buffers. JAX returns that
state as part of every selective result. Zero-atom calls keep the same arity as
nonempty calls: selective matrix output contains the matrix triple plus tile state
(and both matrix triples for dual cutoff), while segmented COO remains the normal
seven-array single-system or ten-array batched tuple.

PyTorch keeps the existing return arity by default. To receive the tile state, select
the cluster-tile method explicitly and pass `return_state=True` with
`rebuild_flags`:

- Single-system matrix output appends `(num_tiles, tile_row_group,
  tile_col_group)`, yielding six tensors for one cutoff and nine for two.
- Single-system segmented COO returns `(neighbor_list, pair_offsets,
  pair_counts, neighbor_list_shifts)` and appends `(num_tiles,
  tile_row_group, tile_col_group)` with `return_state=True`, yielding seven
  tensors. The fixed buffers have shapes `(2, max_pairs)`, `(2,)`, `(1,)`, and
  `(max_pairs, 3)`; `pair_offsets` is `[0, max_pairs]`, and
  `pair_counts[0]` gives the active prefix. Values after that prefix are
  inactive.
- Batched matrix output appends `(tile_offsets, tile_counts, num_tiles,
  tile_row_group, tile_col_group, tile_system)`, yielding nine tensors for one
  cutoff and twelve for two.
- Batched segmented COO appends the same six state tensors to its four topology
  outputs, yielding ten tensors.

```python
from nvalchemiops.torch.neighbors import neighbor_list

(
    neighbor_matrix,
    num_neighbors,
    shifts,
    tile_offsets,
    tile_counts,
    num_tiles,
    tile_row_group,
    tile_col_group,
    tile_system,
) = neighbor_list(
    positions,
    cutoff,
    cell=cells,
    pbc=pbc,
    batch_ptr=batch_ptr,
    method="batch_cluster_tile",
    rebuild_flags=rebuild_flags,
    return_state=True,
)
```

For a single-system fixed-capacity COO workflow, retain the public topology
buffers and tile state from the previous step:

```python
(
    neighbor_list_coo,
    pair_offsets,
    pair_counts,
    shifts_coo,
    num_tiles,
    tile_row_group,
    tile_col_group,
) = neighbor_list(
    positions,
    cutoff,
    cell=cell,
    pbc=pbc,
    method="cluster_tile",
    return_neighbor_list=True,
    rebuild_flags=rebuild_flags,  # shape (1,)
    neighbor_list=neighbor_list_coo,
    pair_offsets=pair_offsets,  # tensor([0, max_pairs], dtype=torch.int32)
    pair_counts=pair_counts,
    neighbor_list_shifts=shifts_coo,
    num_tiles=num_tiles,
    tile_row_group=tile_row_group,
    tile_col_group=tile_col_group,
    return_state=True,
)
```

Pass the returned topology buffers and state back on the next call. `return_state`
is intentionally rejected for auto-selected and non-cluster-tile methods, and it
requires `rebuild_flags`. Caller-supplied state tensors are returned by identity:
the suffix aliases the same buffers, which the next selective call updates in place.
When only the public state suffix is supplied, temporary sorting scratch is allocated
internally without replacing those state buffers.

Use two phases for PyTorch selective COO. First make an eager bootstrap call with
every flag true; it may allocate omitted topology and tile buffers and returns the
reusable state. Every later eager call containing a false flag must supply all fixed
topology and tile-state buffers, otherwise it raises `ValueError` before a kernel
launch. Single-system Torch and JAX COO state has one segment:
`pair_offsets` must stay `[0, physical_capacity]`, and `pair_counts` has one value.

```python
# Eager bootstrap: omit every persistent buffer.
state = cluster_tile_neighbor_list(
    positions, cutoff, cell, format="coo", return_state=True,
    rebuild_flags=torch.ones(1, dtype=torch.bool, device=positions.device),
)
```

The direct single-system `cluster_tile_neighbor_list` route also supports
`torch.compile(fullgraph=True)` for this steady-state phase. Bootstrap and validate
capacities eagerly, then compile a function that accepts fixed-shape buffers and
`rebuild_flags`; compiled calls preserve false-flag data and rebuild true-flag data.
False flags retain the fixed launch graph, so they do not promise zero preprocessing
or kernel launches. The unified `neighbor_list(..., method="cluster_tile")` dispatch
is unsupported under compilation because it cannot validate tensor-valued `pbc`
without host synchronization. Compiled calls also omit
host-synchronized offset and overflow diagnostics, so retain the eagerly validated
capacities. The Warp COO query enforces physical output-buffer bounds as defense in
depth, but mutated or malformed metadata remains unsupported. Compiled Torch and
JIT-compiled JAX cannot synchronize to raise a data-dependent metadata error. If
caller-provided single-system offsets no longer equal `[0, physical_capacity]`, a
true rebuild is suppressed before it writes and the returned active count is zero.
For valid offsets, an overflowed count is capped at writable capacity. A false
rebuild retains valid prior buffers and counts; malformed metadata still returns a
zero active count without changing those buffers. This prevents a returned count
from naming unwritten entries; it does not make mutated metadata supported. Batched
cluster-tile fullgraph support is not provided.

```python
@torch.compile(fullgraph=True)
def reuse(flags, neighbor_list, pair_offsets, pair_counts, shifts, num_tiles, row, col):
    return cluster_tile_neighbor_list(
        positions, cutoff, cell, format="coo", return_state=True,
        rebuild_flags=flags, neighbor_list=neighbor_list, pair_offsets=pair_offsets,
        pair_counts=pair_counts, neighbor_list_shifts=shifts, num_tiles=num_tiles,
        tile_row_group=row, tile_col_group=col,
    )
```

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

For PyTorch cluster-tile methods, geometry output buffers are
non-differentiable write targets and must not require gradients. When matrix
geometry must be reconstructed for autograd, the returned distances and vectors
are fresh differentiable tensors; any supplied buffers receive detached
snapshots of the same values. Without reconstruction, the returned geometry
continues to be the supplied or internally allocated buffers. Build losses from
the returned tensors rather than from reusable output buffers.

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
(compact COO packs the matrix result eagerly because its pair count is
data-dependent; use `format="matrix"` under `jax.jit`). Naive and cell-list
methods can instead use `coo_capacity` to return padded fixed-capacity COO,
with geometry and pair outputs aligned to the same valid prefix. `pair_energies` /
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

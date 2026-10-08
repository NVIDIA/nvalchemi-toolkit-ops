<!-- markdownlint-disable MD013 -->

# Advanced configuration

Most applications can start with the calls in the {doc}`main guide <neighborlist>`.
Use this page when you need to control storage, tune a cell grid, work with native
tiles, or combine cutoffs. For compiled execution and reuse, see
{doc}`neighborlist_repeated`.

## Choose storage and capacity together

A matrix has one row per source and `max_neighbors` slots per row. It is simple
to use and has fixed shapes, but a few crowded neighborhoods can force a wide
allocation for every atom. Compact COO stores only active pairs and suits edge
consumers, at the cost of packing and a data-dependent pair count.

| Layout | Valid data | Useful when |
| --- | --- | --- |
| Matrix | First `counts[i]` slots of row `i` | Fixed-shape consumers, per-source reductions |
| Compact COO | All returned columns; `ptr` delimits source rows | Eager edge/graph consumers |
| Fixed-capacity JAX COO | Prefix ending at `ptr[-1]`; inspect required counts and validity | Compiled naive/cell-list edge consumers |
| Segmented COO | Active prefix of each system's reserved segment | Selective rebuilds on supported routes |
| Native cluster tiles | Active tile records and their sorted atom groups | Consumers designed to query tiles directly |

(neighbor-list-capacity-estimation)=

### Estimate max_neighbors

`estimate_max_neighbors` uses a cutoff-sphere volume and an assumed atomic
number density to suggest an initial capacity per atom:

$$
M \approx \rho\frac{4\pi r^3}{3}.
$$

By default, positive cutoffs reserve at least 16 slots, rounded up to a multiple
of 16. When distances are in Å, supply density in **atoms/Å³**. Converting typical
mass densities to atom counts gives about **0.10** for liquid water near room
temperature ([NIST water density](https://www.nist.gov/file/40736)) and **0.08**
for crystalline α-quartz ([Handbook of Mineralogy](https://handbookofmineralogy.org/pdfs/quartz.pdf)).
Count atoms, rather than molecules or formula units.

Use a higher input density to allow for expected local variation. The `0.15`
below is a conservative example above those averages:

```python
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors

width = estimate_max_neighbors(cutoff, atomic_density=0.15)
```

An average density does not bound every local neighborhood. Check required
counts on representative structures and allow for expected changes.
`atomic_density` during preparation estimates omitted widths; explicit
capacities take precedence.

:::{note}
Density is a storage-estimation input, not a check of the system's physical
density. A low estimate can allocate too few neighbor slots. Before using a
matrix, compare its required counts with its row width. Some direct matrix
calls, as well as Torch prepared naive/cell calls, return excessive counts
without raising an exception.

For JAX prepared execution, call `check_neighbor_list_state` on the successor
state before using the results; it raises `NeighborOverflowError` for
insufficient neighbor capacity. Cluster intermediate storage has separate
`TileBufferOverflow` checks; see {ref}`cluster-tile-buffer-capacity`.

Increase the explicit capacity, or the density estimate when capacity is
inferred. Rebuild the list or prepare a new state, then check it before use.
See {doc}`neighborlist_repeated` for compiled checks and reuse.
:::

(trim-neighbor-matrix)=

### Trim unused matrix columns in Torch

A matrix may reserve more columns than the current neighborhoods need. After
checking that every required count fits the allocated width, trim all rows to
`K = counts.max()`. Keep the image shifts and any distances or vectors aligned
with the same slice. Rows with fewer than `K` neighbors still need a valid mask.

This continues the main guide's Torch matrix example:

```python
if torch.any(counts > matrix.shape[1]):
    raise RuntimeError("Increase neighbor capacity and rebuild before trimming")

@torch.compile(fullgraph=True)
def trim_matrix(matrix, counts, shifts):
    K = counts.max()
    matrix_view = matrix[:, :K]
    shift_view = shifts[:, :K]
    valid = torch.arange(matrix_view.shape[1], device=matrix.device)[None, :] < counts[:, None]
    return matrix_view, shift_view, valid

matrix_view, shift_view, valid = trim_matrix(matrix, counts, shifts)
```

Tensor-bound slicing works with `torch.compile(fullgraph=True)` in PyTorch 2.14;
see the [compile documentation](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html).
It needs no explicit `.item()`, but obtaining the width can still synchronize
CUDA with the host. Measure the consumer savings together with that cost.
The slice is a view that retains the original allocation, may be noncontiguous,
and may borrow prepared-state storage. It reduces padded consumer work rather
than freeing the reserved buffer.

Before gathering positions, replace padded neighbor IDs with a valid index,
then mask their contribution. When present, slice distances and vectors to the
returned width with `distances[:, :matrix_view.shape[1]]` and
`vectors[:, :matrix_view.shape[1]]`. In JAX `jit`, keep fixed shapes and
masks; an exact width determined by runtime counts cannot be returned as a
standard dynamically sized array.

### Fixed-capacity COO in JAX

Pass a static `coo_capacity` for compiled naive or cell-list COO. Capacity bounds
the total stored pairs as well as the per-row width. The topology tuple appends
required row counts and `metadata_valid`; optional geometry follows those fields.
The pointer delimits what was actually stored, while required counts tell you
whether row or total capacity was sufficient.

This continues the main guide's JAX setup:

```python
from nvalchemiops.jax.neighbors import (
    check_neighbor_list_state, neighbor_list, prepare_neighbor_list,
)

state = prepare_neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc,
    method="naive", strategy="scalar", max_neighbors=32,
    return_neighbor_list=True, coo_capacity=1024,
)

@jax.jit
def build_edges(current_positions, current_cell, current_state):
    return neighbor_list(current_positions, cell=current_cell, state=current_state)

(edges, ptr, shifts, required_counts, metadata_valid), state = build_edges(
    positions, cell, state,
)
check_neighbor_list_state(state)
active_pairs = int(ptr[-1])
active_edges = edges[:, :active_pairs]  # host-side inspection after the check
```

The unused tail is padding. Do not send all capacity columns to an edge consumer
as if they were valid. Within a compiled consumer, use the pointer and a valid
prefix mask without creating a data-dependent shape. If metadata is invalid,
refresh the configuration; its required counts cannot be trusted. For direct
calls, also compare required counts with pointer differences after leaving
`jax.jit`. The prepared-state check gathers the applicable failure status.

### Segmented COO

Selective COO reserves a fixed segment for each system. `pair_offsets[b]` marks
its start and `pair_counts[b]` gives its active length. Consume
`[pair_offsets[b], pair_offsets[b] + pair_counts[b])`; the rest of the segment is
inactive. A false rebuild flag retains the previous segment.

Keep offsets and all persistent buffers together. Resizing segments changes the
state layout and requires a new state initialized with every system rebuilt.
Single-system selective cluster COO has one segment with offsets
`[0, physical_capacity]`. Exact tuple names differ between direct Torch and JAX
calls; prepared state manages the persistent storage.

(cluster-tile-buffer-capacity)=

## Size cluster-tile storage

Cluster construction first stores candidate tile pairs, then queries atom pairs.
Its intermediate tile capacity is separate from the final matrix width or COO
pair capacity. Sizing only the final output is insufficient.

For a nonempty system with $g=\lceil N/32\rceil$ groups,
`max_tiles_per_group=m` reserves $g\min(g,m)$ tile records. For a compact Torch
batch, the pooled capacity is the sum of those contributions over systems.
Increasing `m` up to the group count provides more room for candidate tiles at
the cost of memory. Eager combined calls estimate it when omitted. Under a
transformation that needs a static allocation shape, supply a positive Python
integer or complete caller-owned tile arrays.

`TileBufferOverflow` reports insufficient intermediate storage in checked eager
paths. Increase the tile capacity and rebuild. Compiled calls cannot always turn
a device count into a Python exception. Validate the intended capacities eagerly;
for a workflow that adapts capacities, use a separate lower-level build, check its
counts on the host, and query only a successful build. Selective segmented batches
may need successive retries as different systems overflow.

### Native tile output

Use the direct cluster API for `format="tile"`; the unified function exposes
matrix and COO. For a single Torch system:

```python
from nvalchemiops.torch.neighbors import cluster_tile_neighbor_list

num_tiles, tile_rows, tile_cols, atom_order, x, y, z = cluster_tile_neighbor_list(
    positions, cutoff, cell, format="tile", max_tiles_per_group=1,
)
```

This small fixture has one group; choose capacity for your actual system.
Tile row/column records refer to sorted groups, and `atom_order` maps their atom
slots back to input IDs. Sorted coordinates and padded lanes require a tile-aware
consumer; the records are not an ordinary atom-pair edge list. Request matrix or
COO when your next operation expects those layouts. Native tiles do not include
per-pair geometry or pair-function arrays.

Torch compact cluster COO with a pair function writes callback buffers in place
and returns its topology tuple, unlike naive/cell-list calls that append pair
arrays. Use matrix output or prepared result properties for a straightforward
callback consumer, and consult the direct signature before unpacking a
cluster-specific result.

## Tune a cell grid

Cell lists trade bin count against the number of atoms checked per bin. A finer
grid can reduce occupancy but requires a wider stencil and more bin work. A
coarser grid saves grid storage but increases candidate comparisons. `max_nbins`
caps grid storage; reducing it can increase occupancy. Compare build and query
together on the coordinate distribution you will use.

`min_cells_per_dimension` is a lower bound in configured sizing, not a requested
exact grid. Query coverage also depends on cutoff and cell geometry, especially
for skewed cells. If you pass a prebuilt grid to queries with several cutoffs,
configure its search radius for the largest cutoff. See
{ref}`build-query-separation` for a complete example.

(configured-and-adaptive-sizing)=

### Configured and adaptive sizing

For eligible pair-centric queries, consider adaptive sizing for systems with
approximately uniform spatial density, including batches with similar atom
counts, cell shapes and volumes. It uses cell geometry
and atom population to balance cell visits against estimated work per cell,
which can reduce build/query cost compared with the configured grid. The cost
model assumes roughly uniform bin occupancy; clustered distributions or very
different systems in a batch can change the benefit. Compare both policies on
your actual inputs.

`grid_policy="configured"` remains the default. Set `grid_policy="adaptive"`
explicitly to opt in. With `neighbor_list(method="cell_list", ...)`, adaptive
sizing also applies when the automatic cell-list strategy selects pair-centric
execution; you do not need to pin that strategy to use the policy.

For an eager full-list CUDA query, the call is:

```python
matrix, counts, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc,
    method="cell_list_pair_centric", max_neighbors=32,
    grid_policy="adaptive",
)
```

Adaptive sizing is limited to eligible full-list pair-centric routes. It does
not combine with selected rows, half fill, selective rebuilds, or fixed-cell
execution. Prepared states require configured sizing. Torch compiled adaptive
calls require complete cell-list workspace; JAX adaptive selection is eager,
with additional restrictions on explicit launch metadata. Use the backend API
for those workspace and metadata arguments. Grid selection is reused within
the same construction, not cached across calls. Every construction still bins
the current atoms and enumerates neighbors; it does not reuse a Verlet topology.

### Prepared free-boundary spans

When preparation synthesizes a nonperiodic cell from exemplar coordinates,
`span_margin` reserves growth in the coordinate spans. The synthesized box
recenters as the system translates, so a rigid translation alone does not
exhaust the margin. Choose it from the extent your workload can reach, and
re-prepare after a span failure. It is unrelated to the displacement margin of buffered topology.
An explicit appropriate cell gives you direct control over the domain.

## Query two cutoffs together

Two lists can serve a short-range MLIP supplemented by explicit long-range
physics, as in [DPLR](https://arxiv.org/abs/2112.13327). Here a **6 Å** list
illustrates local MLIP environments, while a **12 Å** list illustrates the
real-space part of Ewald/PME electrostatics. The reciprocal-space contribution
is evaluated separately; a finite neighbor list alone does not provide the
complete long-range interaction. See the
[LAMMPS KSpace documentation](https://doc.lammps.org/stable/kspace_style.html).
These cutoffs are illustrative; the model and physics determine the values.

The `"naive_dual_cutoff"` method selects the **scalar** implementation and emits
two lists in one search; no extra `strategy` argument is needed. Direct search
is a useful option when the cutoffs cover much of the system's physical domain.
Keep `cutoff2 >= cutoff`; equal cutoffs are valid. Each list has its own capacity:

```python
matrix1, counts1, shifts1, matrix2, counts2, shifts2 = neighbor_list(
    positions, 6.0, cutoff2=12.0, cell=cell, pbc=pbc,
    method="naive_dual_cutoff", max_neighbors1=128, max_neighbors2=1024,
)
```

The widths `128` and `1024` are illustrative allocations for this example, not
capacity guarantees for water or another material. Estimate and check each
width for your actual structures. Doubling the cutoff from 6 to 12 multiplies
the cutoff-sphere volume by eight at fixed density.

Cluster tiles also support two cutoffs in matrix output, with fully periodic
float32 input. Dual cluster output cannot combine with per-pair geometry or a
pair function. There is no dual cell-list or tiled-naive implementation; a
unified dual request uses a compatible scalar naive or cluster route.

Prepared width names differ: Torch uses `max_neighbors` and `max_neighbors2`;
JAX also accepts `max_neighbors1` for the first width. Preparation estimates
omitted capacities, but each still needs enough room for its cutoff. Each
consumer uses its own geometric neighborhood; the lists do not define the
interaction or model.

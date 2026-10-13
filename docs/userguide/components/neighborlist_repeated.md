<!-- markdownlint-disable MD013 -->

(prepared_neighbor_lists)=

# Repeated and compiled calculations

When coordinates change at each step, neighbor-list setup can become a noticeable
part of the calculation. Prepared state fixes the method and storage; compilation
reduces host overhead on eligible routes. Reusing a grid, bins, or buffered
topology can save other work, but each has a different validity condition.

Start with the {doc}`main guide <neighborlist>` for method choice and output use.
The snippets here continue its CUDA FCC setup.

## Prepare once and compile the calculation

Preparation runs eagerly. It fixes atom count and ordering, batch membership,
dtype/device, cutoffs, PBC pattern, selected rows, output format, and capacities.
Positions and applicable cell values can change later. Prepare a new state when
a fixed item changes; execution-time configuration does not override it.

Inspect `state.method` and `state.strategy` to see the resolved route.
Automatic JAX preparation may choose a compilation-eligible equivalent instead
of the eager selector's route. Explicit method and strategy pin your choice.

The following returns a radial feature computed from current geometry, so the
compiled function includes both the neighbor search and a small consumer:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
import torch
from nvalchemiops.torch.neighbors import neighbor_list, prepare_neighbor_list

state = prepare_neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc,
    method="naive", strategy="scalar", max_neighbors=32,
    return_distances=True,
)
if not state.supports_compilation:
    raise RuntimeError(state.compilation_blocker)

@torch.compile(fullgraph=True)
def radial_features(current_positions, current_cell):
    matrix, counts, shifts, distances = neighbor_list(
        current_positions, cell=current_cell, state=state,
    )
    valid = torch.arange(matrix.shape[1], device=current_positions.device)[None, :] < counts[:, None]
    weights = torch.where(valid, (1.0 - distances / cutoff).clamp_min(0) ** 2, 0.0)
    return weights.sum(dim=1)

features = radial_features(positions, cell)
```

Torch captures the mutable state in the closure. Each execution updates its
results. Finish consumption and backward before reusing it; clone results that
must survive the next call. Preparation alone does not produce initialized
neighborhoods.

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.neighbors import (
    check_neighbor_list_state, neighbor_list, prepare_neighbor_list,
)

state = prepare_neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc,
    method="naive", strategy="scalar", max_neighbors=32,
    return_distances=True,
)
if not state.supports_compilation:
    raise RuntimeError(state.compilation_blocker)

@jax.jit
def radial_features(current_positions, current_cell, current_state):
    (matrix, counts, shifts, distances), next_state = neighbor_list(
        current_positions, cell=current_cell, state=current_state,
    )
    valid = jnp.arange(matrix.shape[1])[None, :] < counts[:, None]
    weights = jnp.where(valid, jnp.maximum(1.0 - distances / cutoff, 0) ** 2, 0.0)
    return weights.sum(axis=1), next_state

features, state = radial_features(positions, cell, state)
check_neighbor_list_state(state)  # host check before using the result
```

JAX passes immutable state into the call and returns a successor. Check the
successor outside `jax.jit`; an invalid system stays invalid until you prepare
a new state. If you enable state donation, consume results first and do not use
aliases retained from the donated state.

:::

::::

`supports_compilation` describes route eligibility, not a successful compilation
in your environment. Warm up the actual function, then compare its result with
the eager calculation before timing it.

### Which routes can compile?

Automatic selection and host-side capacity inspection belong outside compiled
code. Use prepared state or a direct method API with fixed configuration for
compiled work; these are the documented JAX compilation boundaries. Pinned Torch unified calls can reach static
naive/cell-list routes, subject to the underlying method's buffer and metadata
requirements.

| Prepared configuration | Torch `fullgraph=True` | JAX `jit` |
| --- | --- | --- |
| Scalar naive; topology or geometry | Eligible | Eligible |
| Tiled naive; topology, including selected rows | Eligible | Eligible |
| Atom-centric cell list; topology or geometry | Eligible | Eligible |
| Pair-centric cell list; topology | Eligible | Eligible with static launch metadata |
| Pair-centric cell list; geometry | Eligible | Eager only |
| Cluster-tile matrix, including two cutoffs | Eligible | Single system eligible; explicit batch route eager only |
| Pair function | Prepared callbacks eager only | Route dependent; atom-centric/naive static routes can compile |

Validate selected atom indices before compiling a partial naive query with
`torch.compile` or `jax.jit`: each `target_indices` entry must be in
`[0, num_atoms)`. Compiled execution does not perform the eager bounds check.

COO eligibility depends on its layout. JAX exact-size compact COO requires eager
shape compaction; fixed-capacity or segmented COO provides a static alternative
on supported routes. Torch prepared batched selective cluster COO is eager-only.
See {doc}`neighborlist_advanced` and the
{func}`Torch preparation <nvalchemiops.torch.neighbors.prepare_neighbor_list>` /
{func}`JAX preparation <nvalchemiops.jax.neighbors.prepare_neighbor_list>`
references for the configuration you use.

Torch direct naive and cell-list **fixed-matrix** pair-output calls can use
`compile_pair_fn(pair_fn)` to specialize a callback for compilation. This is a
different path from prepared callbacks; raw Warp functions and COO callback
packing remain outside that compiled matrix path.

For direct JAX periodic naive calls under `jax.jit`, precompute all three launch
metadata values with `compute_naive_num_shifts`: `shift_range_per_dimension`,
`num_shifts_per_system`, and `max_shifts_per_system`. They must match the static
cutoff, cell, and PBC. Preparation handles the supported route's metadata for you.

## Know what is being reused

| Reused object | Work it can save | What still changes |
| --- | --- | --- |
| Prepared state | Selection, allocation, fixed launch setup | Neighborhood enumeration and current geometry |
| Fixed-cell geometry | Cell inversion and grid geometry | Atom bin membership and neighbor topology |
| Built cell bins | Spatial bin construction while membership remains valid | Cutoff query and current pair geometry |
| Buffered topology | Neighbor enumeration while the displacement bound holds | Geometry and physical-cutoff filtering |

### Fixed cells and free boundaries

With `fixed_cell=True`, prepared execution caches applicable cell geometry. Keep
the cell values fixed; supplying a changed cell is not supported. Atom membership
and topology are still rebuilt as needed by the execution route. This is useful
for repeated calculations at fixed volume, but is not a Verlet neighbor cache.

With `fixed_cell=False`, positions and applicable cell values may change. The
prepared storage and periodic search coverage must still be sufficient. A
smaller or more skewed cell can require re-preparation; prepared execution
rejects insufficient coverage rather than extending fixed buffers.

For a free-boundary cell-list route without an explicit cell, preparation creates
a bounding cell from the exemplar coordinate spans. `span_margin` reserves room
for later configurations to grow in extent; the synthesized box can recenter
as the system translates. It bounds the span, not an absolute position or a
Verlet displacement. Re-prepare if the current spans exceed that capacity. See
{doc}`neighborlist_advanced` for these settings.

(build-query-separation)=

### Build bins once, then query several cutoffs

Separate construction and query when several consumers use the **same current
coordinates**. Build a grid and stencil suitable for the largest cutoff, then
query smaller cutoffs from those bins. Increasing the query cutoff beyond that
coverage is insufficient: you must rebuild/reconfigure the search stencil.

Here is a complete Torch example. The lower-level JAX functions have functional
return values; use their API signatures rather than treating them as in-place
Torch calls.

```python
from nvalchemiops.torch.neighbors.cell_list import (
    build_cell_list, estimate_cell_list_sizes, query_cell_list,
)
from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list

box = cell[None, :, :]
max_cells, radius = estimate_cell_list_sizes(box, pbc, cutoff)
bins = allocate_cell_list(len(positions), max_cells, radius, device)
build_cell_list(positions, cutoff, box, pbc, *bins)

matrix = torch.empty((len(positions), 32), dtype=torch.int32, device=device)
shifts = torch.empty((len(positions), 32, 3), dtype=torch.int32, device=device)
counts = torch.zeros(len(positions), dtype=torch.int32, device=device)
for query_cutoff in (2.9, 3.0):
    counts.zero_()  # low-level atom-centric queries accumulate into counts
    query_cell_list(
        positions, query_cutoff, box, pbc, *bins,
        matrix, shifts, counts, strategy="atom_centric",
    )
    print(query_cutoff, int(counts.sum()))
```

Consume each result before querying into the same buffers again. To reuse bins
across changed coordinates, `cell_list_needs_rebuild` detects changes in bin
coordinates. Rebuild when membership, cell, PBC, or stored periodic representation
changes. This check is not a displacement test for a buffered topology.

### Keep buffered topology while atoms move a little

A buffered list includes neighbors out to `physical_cutoff + skin`. At each step,
recompute geometry from current coordinates and filter at the physical cutoff.
For a fixed cell, a conservative condition is that every atom moves less than
`skin / 2` from its position at the last build: then a pair can close by less than
`skin` before you rebuild. This is the usual
[neighbor-skin tradeoff](https://docs.lammps.org/neighbor.html): more stored pairs
can reduce rebuild frequency, but add consumer work at every step.

The toolkit provides `neighbor_list_needs_rebuild` and its batched counterpart
for displacement checks. Retain reference coordinates from the **last actual
build**, and refresh them only when rebuilding. These helpers do not turn every
`neighbor_list` call into a cached topology calculation.

When consuming retained indices and shifts, reconstruct vectors using the
current positions and current source-system cell. Keep a consistent periodic
representation: wrapping an atom can change the image shift needed for an old
edge. Cell deformation or a representation change can require a rebuild even
when a simple Cartesian displacement check looks small. A minimum-image
endpoint displacement alone cannot detect an arbitrarily long excursion between
checks.

Saved geometry and callback arrays can belong to the previous build unless
the route explicitly reconstructs current geometry. Reusing topology is
therefore different from reusing saved distances, vectors, or pair outputs. Filter fresh distances at the physical cutoff before evaluating
the current interaction.

## Rebuild selected systems in a batch

Prepare with `selective=True` when some systems can retain their old topology.
`rebuild_flags` has one Boolean entry per system. A true entry rebuilds that
system; a false entry preserves its previous output. Initialize every system
before retaining it.

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
state = prepare_neighbor_list(
    batch_positions, cutoff, cell=cells, pbc=batch_pbc, batch_ptr=batch_ptr,
    method="naive", strategy="scalar", max_neighbors=32, selective=True,
)
flags = torch.ones(2, dtype=torch.bool, device=device)
first = neighbor_list(batch_positions, cell=cells, state=state, rebuild_flags=flags)
flags = torch.tensor([True, False], dtype=torch.bool, device=device)
next_result = neighbor_list(batch_positions, cell=cells, state=state, rebuild_flags=flags)
```

:::

:::{tab-item} JAX
:sync: jax

```python
state = prepare_neighbor_list(
    batch_positions, cutoff, cell=cells, pbc=batch_pbc, batch_ptr=batch_ptr,
    method="naive", strategy="scalar", max_neighbors=32, selective=True,
)
flags = jax.device_put(jnp.ones(2, jnp.bool_), gpu)
first, state = neighbor_list(batch_positions, cell=cells, state=state, rebuild_flags=flags)
check_neighbor_list_state(state)
flags = jax.device_put(jnp.array([True, False]), gpu)
next_result, state = neighbor_list(batch_positions, cell=cells, state=state, rebuild_flags=flags)
check_neighbor_list_state(state)
```

:::

::::

This example keeps the coordinates unchanged to show preservation. Your
application must decide which systems are valid to retain. A buffered topology
needs its displacement and periodic-representation conditions; unchanged bins
alone do not justify retaining old pair geometry. JAX prepared selective states reject geometry and pair outputs; Torch naive
selective calls also reject them. Torch cell-list calls accept geometry with selective flags, but skipped rows
retain saved geometry, even if the current positions require gradients.
Reconstruct current geometry explicitly before using retained topology for a
new calculation. The topology still needs your validity check.

Prepared cluster state owns its topology, tile records, and sorting storage.
If you use direct selective cluster APIs, retain **all** returned state and fixed
output buffers, and bootstrap eagerly with every flag true. See the backend API
for the matrix and segmented-COO tuples. A false flag preserves results but does
not promise that every route skips all setup or kernel launches.

### CUDA Graph replay in Torch

A compilation-eligible prepared cluster matrix route can be used in a CUDA Graph
when its configuration and tensor addresses remain fixed. Warm up before capture,
initialize retained systems, and copy new positions, applicable cells, and flags
into the captured input tensors before replay. Consume borrowed results before
the next replay. Validate capacity eagerly first; a compiled device assertion
requires restarting the process.

Compilation, CUDA Graph capture, and topology reuse solve different costs.
Choose them according to the repeated work your application actually performs,
then measure rebuild and reuse steps separately as described in the
{ref}`performance guidance <nl_performance>`.

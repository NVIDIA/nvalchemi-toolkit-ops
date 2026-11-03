<!-- markdownlint-disable MD013 -->

(neighborlist_userguide)=

# Neighbor Lists

A neighbor list is a data structure that records each atom's neighbors within a
cutoff and, in periodic systems, their image information.
ALCHEMI Toolkit-Ops builds CUDA neighbor lists in NVIDIA Warp, with PyTorch and
JAX interfaces. Features include:

- **Three algorithm families:** naive searches ($O(N^2)$), cell lists ($O(N)$
  with bounded bin occupancy and search neighborhoods),
  and cluster tiles (approximately $O(N)$ local query work, plus group-pair
  construction).
  Select automatically or pin a strategy; the method sections explain the
  scaling conditions.
- **Floating-point types:** float32 and float64 for naive and cell lists;
  cluster tiles require float32. Enable JAX x64 mode to use float64.
- **Batching:** single systems or batches of independent systems, with periodic,
  partially periodic, or free boundaries.
- **Selective rebuilds:** rebuild chosen systems in a batch while preserving
  the others' existing lists.
- **Partial queries:** find neighbors for selected source atoms within a system.
- **Differentiable geometry:** distances and displacement vectors, treating the
  neighbor pairs as fixed during differentiation.
- **User-defined pair potentials:** evaluate Warp pair functions during neighbor
  enumeration; their energy and force outputs are forward-only.
- **Repeated and compiled work:** prepared state, `torch.compile` and `jax.jit`
  for eligible configurations, and supported Torch CUDA Graph replay.
- **Query modes:** full or half lists and searches with two cutoffs.
- **Output layouts:** padded matrices, COO edge lists, or native cluster tiles.

Support varies by method, backend, and configuration; see the
[feature table](#match-the-method-to-your-request) and
{doc}`neighborlist_repeated`. Lists describe geometry; your model defines bonds
and exclusions.

## Output formats

Neighbor lists store atom IDs in two ordinary layouts:

- **Padded matrix:** a matrix with dimensions number of atoms × maximum number
  of neighbors per atom, set by `max_neighbors`. See
  {ref}`Estimate max_neighbors <neighbor-list-capacity-estimation>` to choose
  an initial capacity.
  For an atom with K neighbors, the first K entries in its row contain the
  neighbors' indices in arbitrary order. The remaining entries are padding.
  This is the default layout.
- **List (COO):** two rows and one column per neighbor pair. The first row holds
  source atom IDs; the second holds destination atom IDs. Compact COO stores
  only active pairs.

Periodic neighbors also carry aligned integer shifts identifying the
destination's image in units of the cell vectors. Row counts mark the valid
matrix entries. For compact COO, pointers delimit each source's pairs;
differences between successive pointers give its neighbor count.

A full list gives each atom its complete neighborhood, with symmetric pairs in
both directions. A half list keeps one representative of each pair and needs a
consumer that accounts for both atoms. See [additional features](#additional-features)
for distances, vectors, and matrix/COO consumers. Native cluster tiles and fixed
COO layouts are covered in {doc}`neighborlist_advanced`.

## Quick start

For this small system, scalar naive search avoids building a spatial index.
The call explicitly selects that strategy and requests a full list in the
default padded matrix layout.

For your own structures, see
{ref}`Estimate max_neighbors <neighbor-list-capacity-estimation>` to choose an
initial matrix capacity and check that it is sufficient.

Here is an ideal FCC lattice with 32 atoms in an 8.1 Å cubic cell. The 3 Å cutoff
includes the first shell of 12 neighbors per atom. This is a geometric fixture,
without a potential or a claim about equilibrium.

:::{note}
The unified call handles eager dispatch and allocation for convenience; this
example does not provide the best performance. Compare
compatible methods using the {ref}`practical performance guidance <nl_performance>`.
For repeated calls or compilation, see {doc}`neighborlist_repeated`.
:::

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
import torch
from nvalchemiops.torch.neighbors import neighbor_list

device = torch.device("cuda")
basis = torch.tensor(
    [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    dtype=torch.float32, device=device,
)
axis = torch.arange(2, device=device)
grid = torch.cartesian_prod(axis, axis, axis)
positions = 4.05 * (grid[:, None, :] + basis[None, :, :]).reshape(-1, 3)
cell = 8.1 * torch.eye(3, dtype=torch.float32, device=device)
pbc = torch.ones(3, dtype=torch.bool, device=device)
cutoff = 3.0

matrix, counts, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc,
    method="naive_scalar", half_fill=False, max_neighbors=32,
)
print("counts:", counts[:4].tolist())  # [12, 12, 12, 12]
print("neighbors of atom 0:", matrix[0, :int(counts[0])].tolist())
print("aligned image shifts:", shifts[0, :int(counts[0])].tolist())
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.neighbors import neighbor_list

gpu = jax.devices("gpu")[0]
with jax.default_device(gpu):
    basis = jnp.array(
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
        dtype=jnp.float32,
    )
    axis = jnp.arange(2)
    grid = jnp.stack(
        jnp.meshgrid(axis, axis, axis, indexing="ij"), axis=-1,
    ).reshape(-1, 3)
    positions = 4.05 * (grid[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    cell = 8.1 * jnp.eye(3, dtype=jnp.float32)
    pbc = jnp.ones(3, dtype=jnp.bool_)
    cutoff = 3.0

    matrix, counts, shifts = neighbor_list(
        positions, cutoff, cell=cell, pbc=pbc,
        method="naive_scalar", half_fill=False, max_neighbors=32,
    )
print("counts:", counts[:4].tolist())  # [12, 12, 12, 12]
print("neighbors of atom 0:", matrix[0, :int(counts[0])].tolist())
print("aligned image shifts:", shifts[0, :int(counts[0])].tolist())
```

:::

::::

Each atom has 12 neighbors, so the first 12 entries of each matrix row are
valid. Every printed neighbor ID is paired with the shift in the same slot.
A shift of `[0, 0, 0]` means the original image; `[-1, 0, 0]` translates the
destination by one cell vector in the negative first direction. Neighbor order
can vary.

To see the same neighborhood as a compact COO list, use either backend's
`neighbor_list` from above:

```python
edges, ptr, image_shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc,
    method="naive_scalar", half_fill=False, max_neighbors=32,
    return_neighbor_list=True,
)
print("first row offsets:", ptr[:4].tolist())  # [0, 12, 24, 36]
print("total pairs:", int(ptr[-1]))  # 384
```

The offsets put atom 0's pairs in columns 0–11, atom 1's in 12–23, and so on.
The difference between successive offsets is the source's neighbor count.
`image_shifts` aligns with the pair columns in `edges`; matrix counts are not
returned in this layout.

For periodic boundaries, supply cell vectors as rows and a Boolean `pbc` mask
for the periodic directions. For free boundaries, omit both on supported routes.
Matrix storage needs sufficient `max_neighbors` capacity; see
{ref}`Estimate max_neighbors <neighbor-list-capacity-estimation>` for an initial
width. Increase it and rebuild if a returned count exceeds the row width.
Unified eager calls allocate working storage for you; reusable buffers and
prepared state are covered in {doc}`neighborlist_repeated`.

The examples in this guide target CUDA. CPU execution is available for
compatibility on supported routes; cluster tiles and pair-centric cell lists
require CUDA.

## Choose a method

The three families offer five strategies. They find the same pair/image
relationships when settings match and capacity is sufficient, but differ in
candidate work, setup, and supported outputs. See the
[feature table](#match-the-method-to-your-request) for compatible requests.

(naive-algorithm)=

### Naive

Naive search checks all atom pairs in a nonperiodic system, or candidate
periodic images followed by a distance test. It needs no spatial index, making
it useful when the cutoff covers much of the system's physical domain, or when
querying only a few selected source atoms. The scalar strategy checks pairs
individually and supports geometry and pair functions; the tiled strategy
shares coordinate tiles between CUDA threads and provides topology alone. Full
queries require
$O(N^2)$ candidate checks; $T$ selected sources require about $O(TN)$, assuming
a bounded image range.

(cell-list-algorithm)=

### Cell lists

Cell lists bin atoms in space and search nearby bins, reducing distant
candidate checks in larger systems with local neighborhoods. Atom-centric
search assigns work to source atoms; pair-centric search distributes work over
candidate bin pairs so threads can cooperate on their atoms. At fixed density
and cutoff, bounded bin occupancy and search neighborhoods allow approximately
$O(N)$ work; crowded bins or a system-wide cutoff reduce that advantage.
Pair-centric lists require CUDA, and JAX supports only full lists on that route.

(cluster-pair-tile-algorithm)=

### Cluster tiles

Cluster tiles spatially sort atoms into groups of 32, reject distant group pairs
using their bounds, and check atom pairs in the remaining tiles. Shortlist them
for large fully periodic float32 CUDA systems with a local cutoff. At fixed
density and cutoff, compact groups with bounded neighborhoods can keep
accepted tile work roughly linear; sorting and group-pair construction still
add cost. This route requires an explicit cell and full lists. Tile-aware consumers can avoid conversion to
matrix or COO; see {ref}`cluster-tile-buffer-capacity` for native output.

(estimating-and-running-a-strategy-explicitly)=

## Let the Toolkit-Ops choose

When `method` is omitted, the ALCHEMI Toolkit-Ops uses internal heuristics to choose a
compatible strategy. It estimates work from atom count, cutoff, density,
periodic images, dtype, and requested outputs; it does not time or tune your
calculation. To inspect a suggestion, print the returned method and then run
that exact choice. This example reuses the single-system FCC setup:

:::{note}
Heuristic estimates do not always identify the fastest compatible strategy.
For performance-critical applications, benchmark compatible methods individually
on your actual workload, including setup and required outputs, then pin the measured
choice. See the {ref}`performance guidance <nl_performance>`.
:::

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
from nvalchemiops.torch.neighbors import suggest_neighbor_list_method

batch_ptr = torch.tensor([0, len(positions)], dtype=torch.int32, device=device)
method = suggest_neighbor_list_method(
    batch_ptr, cell, pbc, cutoff=cutoff, positions=positions, half_fill=False,
)
print("Selected method:", method)
matrix, counts, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc,
    method=method, half_fill=False, max_neighbors=32,
)
```

:::

:::{tab-item} JAX
:sync: jax

```python
from nvalchemiops.jax.neighbors import suggest_neighbor_list_method

batch_ptr = jax.device_put(jnp.array([0, len(positions)], jnp.int32), gpu)
method = suggest_neighbor_list_method(
    batch_ptr, cell, pbc, cutoff=cutoff, positions=positions, half_fill=False,
)
print("Selected method:", method)
matrix, counts, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc,
    method=method, half_fill=False, max_neighbors=32,
)
```

:::

::::

Pass the same output requests to the suggestion and calculation. Fine-grained
names such as `"naive_scalar"` pin a strategy; family names such as `"naive"`
leave strategy selection automatic. Keep heuristic selection outside compiled
or repeated calculations.

## Batch systems

Keep each system's atoms contiguous and concatenate their coordinates. A
`batch_ptr` marks the boundaries; atom IDs in the output refer to the concatenated
array. No pairs are emitted between different systems.

This batches two copies of the Quick Start fixture:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
n = len(positions)
batch_positions = torch.cat([positions, positions])
cells = torch.stack([cell, cell])
batch_pbc = torch.stack([pbc, pbc])
batch_ptr = torch.tensor([0, n, 2 * n], dtype=torch.int32, device=device)
matrix, counts, shifts = neighbor_list(
    batch_positions, cutoff, cell=cells, pbc=batch_pbc, batch_ptr=batch_ptr,
    method="naive_scalar", max_neighbors=32,
)
```

:::

:::{tab-item} JAX
:sync: jax

```python
n = len(positions)
batch_positions = jnp.concatenate([positions, positions])
cells = jnp.stack([cell, cell])
batch_pbc = jnp.stack([pbc, pbc])
batch_ptr = jax.device_put(jnp.array([0, n, 2 * n], jnp.int32), gpu)
matrix, counts, shifts = neighbor_list(
    batch_positions, cutoff, cell=cells, pbc=batch_pbc, batch_ptr=batch_ptr,
    method="naive_scalar", max_neighbors=32,
)
```

:::

::::

Single-system method names resolve to their batched versions when batch metadata
is supplied. If you also pass `batch_idx`, it must agree with `batch_ptr`.
If your inputs are interleaved, sort every per-atom array consistently and remap
selected atom IDs. Cluster tiles require contiguous system ownership. When
reconstructing periodic geometry yourself, use the cell of the **source system**.

## Prepare a state for repeated calls

`NeighborListState` groups the resolved method, configuration, storage, and
execution diagnostics. `prepare_neighbor_list` performs setup; the subsequent
call calculates the neighborhoods. Use this convenience when atom order, batch
membership, cutoff, and output configuration stay fixed while coordinates change.
Torch state manages reusable buffers; JAX passes a functional state and returns
a successor.

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
from nvalchemiops.torch.neighbors import prepare_neighbor_list

state = prepare_neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc,
    method="naive", strategy="scalar", max_neighbors=32,
)
matrix, counts, shifts = neighbor_list(positions, cell=cell, state=state)
```

Torch updates the state in place. Results can borrow its storage, so finish
using them, including any backward pass, before the next call.

:::

:::{tab-item} JAX
:sync: jax

```python
from nvalchemiops.jax.neighbors import (
    check_neighbor_list_state, prepare_neighbor_list,
)

state = prepare_neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc,
    method="naive", strategy="scalar", max_neighbors=32,
)
(matrix, counts, shifts), state = neighbor_list(positions, cell=cell, state=state)
check_neighbor_list_state(state)
```

JAX returns a successor state. Pass it to the next call and check its status on
the host. Consume results before donating their state to another call.

:::

::::

Prepared state is convenient, but it is not necessarily the fastest route.
JAX status updates can add latency, especially for small searches, and its host
checker may synchronize with the device. Compare prepared and direct calls with
the same method and outputs, including the checks needed before consuming
results. Preparation does not skip rebuilding neighborhoods. See
{doc}`neighborlist_repeated` for compilation and reuse.

## Additional features

### Distances and vectors

Request geometry when a consumer needs it. With row cell vectors, the
separation vector is

$$
\mathbf{r}_{ij}=\mathbf{x}_j-\mathbf{x}_i+\mathbf{s}_{ij}\,\mathbf{C}.
$$

Here, $\mathbf{x}_i$ and $\mathbf{x}_j$ are the Cartesian positions of source
atom $i$ and neighbor atom $j$. $\mathbf{C}$ is the $3\times3$ cell matrix for
their system, with the three cell vectors stored as rows. The three-component
integer shift $\mathbf{s}_{ij}$ specifies how many of each cell vector to add
to atom $j$'s position to select its periodic image. Thus $\mathbf{r}_{ij}$
points from atom $i$ to that image of atom $j$; its length is the interatomic
distance. For free boundaries, the shift is zero.

A large cutoff can include several images of the same atom; keep their shifts
when identifying pairs. Distances and vectors are differentiable in positions and cell while the
emitted pair/image selection is treated as fixed. The discrete decision to
include a pair is not differentiated.

Here we differentiate a directional feature of atom 0 in the Quick Start
system. A radial weight decreases toward the cutoff, while the x component of
the separation vector adds directional information. The gradients describe how
this feature changes with positions and cell. The JAX example uses prepared geometry state to keep allocation
and image-range metadata outside differentiation. Its local precision context
avoids reduced precision in float32 cell-shift products without changing the
global default.

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
x = positions.detach().clone().requires_grad_(True)
c = cell.detach().clone().requires_grad_(True)
matrix, counts, shifts, distances, vectors = neighbor_list(
    x, cutoff, cell=c, pbc=pbc,
    method="naive_scalar", half_fill=False, max_neighbors=32,
    return_distances=True, return_vectors=True,
)
valid = torch.arange(matrix.shape[1], device=device)[None, :] < counts[:, None]
weights = torch.where(valid, (1.0 - distances / cutoff).clamp_min(0) ** 2, 0.0)
directional_feature = (weights * (1.0 + vectors[..., 0] / cutoff)).sum(dim=1)
loss = directional_feature[0]
position_grad, cell_grad = torch.autograd.grad(loss, (x, c))
print(position_grad[1])
print(cell_grad)
```

:::

:::{tab-item} JAX
:sync: jax

```python
geometry_state = prepare_neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc,
    method="naive", strategy="scalar", max_neighbors=32,
    return_distances=True, return_vectors=True,
)

def source_feature(x, c):
    (matrix, counts, shifts, distances, vectors), next_state = neighbor_list(
        x, cell=c, state=geometry_state,
    )
    valid = jnp.arange(matrix.shape[1])[None, :] < counts[:, None]
    weights = jnp.where(valid, jnp.maximum(1.0 - distances / cutoff, 0) ** 2, 0.0)
    directional_feature = (weights * (1.0 + vectors[..., 0] / cutoff)).sum(axis=1)
    return directional_feature[0], next_state

with jax.default_matmul_precision("highest"):
    (loss, geometry_state), (position_grad, cell_grad) = jax.value_and_grad(
        source_feature, argnums=(0, 1), has_aux=True,
    )(positions, cell)
check_neighbor_list_state(geometry_state)
print(position_grad[1])
print(cell_grad)
```

:::

::::

Use the valid-slot mask for every reduction and gather: padding is storage, not
an extra neighbor. Apply the same JAX precision context to other periodic
float32 distance/vector calls, including partial or batched queries.
For naive/cell-list calls, requested outputs append in the order **distances,
vectors, pair energies, pair forces**. See {doc}`neighborlist_advanced` for
additional layout rules.

### Pair functions

A Warp `pair_fn` evaluates an interaction as neighbors are emitted and returns
its energy and force on the source atom. For a symmetric pair interaction,
`half_fill=True` emits one representative: sum its energy once and scatter the
force to the source and its negative to the destination. Compact COO makes
those two atom IDs explicit.

This overlap penalty, $E(r)=\tfrac12 k(3-r)^2$ inside a cutoff of 3, illustrates
the callback. Choose the interaction and parameters for your application.
Define the callback at module scope:

```python
import warp as wp

@wp.func
def overlap_pair(
    vector: wp.vec3f,
    distance: wp.float32,
    params: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
):
    strength = 0.5 * (params[i, 0] + params[j, 0])
    overlap = 3.0 - distance
    energy = 0.5 * strength * overlap * overlap
    force = wp.vec3f(0.0)
    if distance > 0.0:
        force = (-strength * overlap / distance) * vector
    return energy, force
```

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
pair_params = torch.ones((len(positions), 1), dtype=torch.float32, device=device)
edges, ptr, image_shifts, pair_energies, pair_forces = neighbor_list(
    positions, 3.0, cell=cell, pbc=pbc,
    method="naive_scalar", half_fill=True, max_neighbors=32,
    return_neighbor_list=True, pair_fn=overlap_pair, pair_params=pair_params,
)
sources, destinations = edges.long()
energy = pair_energies.sum()
forces = torch.zeros_like(positions)
forces.index_add_(0, sources, pair_forces)
forces.index_add_(0, destinations, -pair_forces)
```

:::

:::{tab-item} JAX
:sync: jax

```python
pair_params = jax.device_put(jnp.ones((len(positions), 1), jnp.float32), gpu)
edges, ptr, image_shifts, pair_energies, pair_forces = neighbor_list(
    positions, 3.0, cell=cell, pbc=pbc,
    method="naive_scalar", half_fill=True, max_neighbors=32,
    return_neighbor_list=True, pair_fn=overlap_pair, pair_params=pair_params,
)
sources, destinations = edges
energy = pair_energies.sum()
forces = jnp.zeros_like(positions).at[sources].add(pair_forces)
forces = forces.at[destinations].add(-pair_forces)
```

:::

::::

A **full** list (`half_fill=False`) contains both $(i,j,\mathbf{s})$ and $(j,i,-\mathbf{s})$, so a
symmetric interaction's total energy for compact COO is
**`0.5 * pair_energies.sum()`**. For a padded matrix, apply the same `0.5`
factor to the sum over valid entries.
Its valid source-force row sums already give each atom's force: do not halve them.
For full COO, scatter only the source contributions. The `0.5` inside the
potential defines a single pair's energy; the additional factor for a full list
corrects double enumeration. Cluster tiles require full lists, and per-source
environment queries need complete neighborhoods rather than half lists.

Callback outputs are **forward-only**, including stopped gradients in JAX.
Use returned distances or vectors and an ordinary Torch/JAX expression when
you need autodiff, as above. See the
{ref}`pair-function contract <warp-neighbor-pair-function-contract>` for dtypes
and parameters, and {doc}`neighborlist_repeated` for compilation limits.

### Partial neighbor lists

Suppose a Monte Carlo proposal moves several atoms together in one system.
You already have `current_positions`, `trial_positions`, and the IDs of the moved
atoms. Query both configurations for those source atoms using **full fill**.
When the selected set is small, naive search can avoid constructing a spatial
index for the entire system.

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
selected = torch.tensor([0, 5, 9], dtype=torch.int32, device=device)
```

:::

:::{tab-item} JAX
:sync: jax

```python
selected = jax.device_put(jnp.array([0, 5, 9], jnp.int32), gpu)
```

:::

::::

The query arguments are the same in either backend. For periodic JAX distances
and vectors, wrap both calls in the local precision context shown in the
[geometry example](#distances-and-vectors):

```python
options = dict(
    cell=cell, pbc=pbc, method="naive_scalar", max_neighbors=32,
    target_indices=selected, half_fill=False,
    return_distances=True, return_vectors=True,
)
old_matrix, old_counts, old_shifts, old_distances, old_vectors = neighbor_list(
    current_positions, cutoff, **options,
)
new_matrix, new_counts, new_shifts, new_distances, new_vectors = neighbor_list(
    trial_positions, cutoff, **options,
)
```

Outputs have three compact source rows. Row `r` belongs to `selected[r]`;
destinations remain IDs into the **full** coordinate array. Use `counts[r]` to
find valid slots. Distances and vectors are aligned with those slots and belong
to their respective configuration.

For inspection, compare pair/image identities, not slot order:

```python
def pair_images(matrix, counts, shifts, selected):
    keys = set()
    for row, source in enumerate(selected.tolist()):
        for slot in range(int(counts[row])):
            destination = int(matrix[row, slot])
            image = tuple(shifts[row, slot].tolist())
            keys.add((source, destination, *image))
    return keys

old_pairs = pair_images(old_matrix, old_counts, old_shifts, selected)
new_pairs = pair_images(new_matrix, new_counts, new_shifts, selected)
removed = old_pairs - new_pairs
added = new_pairs - old_pairs
```

This host-side comparison is for understanding changed neighborhoods, not a GPU
inner loop. Pairs present in both sets can still have changed geometry. Keep the
same cell and coordinate/image convention when comparing keys.

For topology alone, use `method="naive_tile"`; geometry requests require scalar
naive. With partial COO output, remap compact sources with `selected[edges[0]]`;
`edges[1]` already contains original destination IDs. Choose unique, in-range
`int32` source IDs for this workflow.

Partial naive queries do not support `rebuild_flags`, with either scalar or
tiled execution.

Selected rows describe neighborhoods around the moved atoms. They do not by
themselves define a general local energy update: a many-body or message-passing
model can depend on other affected environments. The required update depends on
the model; [MACE](https://arxiv.org/abs/2206.07697), for example, combines
higher-order local information through message passing.

### Selective rebuilds

Selective rebuilding operates on **systems in a batch**, while partial lists
operate on **source atoms**. Prepare a selective topology-only state and first
build every system. Subsequent flags choose which systems to rebuild; the
others keep their existing rows. Here atom 0 of the first system moves, so that
system is rebuilt while the unchanged second system retains its list.

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
selective_state = prepare_neighbor_list(
    batch_positions, cutoff, cell=cells, pbc=batch_pbc, batch_ptr=batch_ptr,
    method="batch_naive", strategy="scalar", max_neighbors=32, selective=True,
)
flags = torch.ones(2, dtype=torch.bool, device=device)
matrix, counts, shifts = neighbor_list(
    batch_positions, cell=cells, state=selective_state, rebuild_flags=flags,
)
updated_positions = batch_positions.clone()
updated_positions[0, 0] += 0.25
flags = torch.tensor([True, False], dtype=torch.bool, device=device)
matrix, counts, shifts = neighbor_list(
    updated_positions, cell=cells, state=selective_state, rebuild_flags=flags,
)
```

:::

:::{tab-item} JAX
:sync: jax

```python
selective_state = prepare_neighbor_list(
    batch_positions, cutoff, cell=cells, pbc=batch_pbc, batch_ptr=batch_ptr,
    method="batch_naive", strategy="scalar", max_neighbors=32, selective=True,
)
flags = jax.device_put(jnp.ones(2, jnp.bool_), gpu)
(matrix, counts, shifts), selective_state = neighbor_list(
    batch_positions, cell=cells, state=selective_state, rebuild_flags=flags,
)
check_neighbor_list_state(selective_state)
updated_positions = batch_positions.at[0, 0].add(0.25)
flags = jax.device_put(jnp.array([True, False], jnp.bool_), gpu)
(matrix, counts, shifts), selective_state = neighbor_list(
    updated_positions, cell=cells, state=selective_state, rebuild_flags=flags,
)
check_neighbor_list_state(selective_state)
```

:::

::::

Initialize all systems before preserving any rows. This scalar-naive selective
route provides topology, without geometry or pair callbacks. Torch results can
borrow state storage; clone a result if it must survive the next call. JAX
returns a successor state and needs a host-side status check. See
{doc}`neighborlist_repeated` for state validity and reuse details.

## Match the method to your request

The table below covers eager CUDA routes. Matrix and COO use the unified
interface; native cluster tiles require a direct call or prepared tile state.
Explicit route support can be broader than automatic-selector eligibility.
Compilation has its own boundaries in {doc}`neighborlist_repeated`.

| Explicit strategy | Selected sources | Half fill | Distances/vectors and pair functions | Layouts | Two cutoffs |
| --- | --- | --- | --- | --- | --- |
| Scalar naive | Yes | Yes | Yes | Matrix, COO | Scalar dual route |
| Tiled naive | Yes, topology only | Yes | No | Matrix, COO | No tiled dual route |
| Atom-centric cell list | Yes | Yes | Yes | Matrix, COO | No cell-list dual route |
| Pair-centric cell list | Torch only | Torch only | Yes | Matrix, COO | No cell-list dual route |
| Cluster tile | No | No | Yes, one cutoff | Matrix, COO, native tiles¹ | Matrix only |

¹ Native tiles contain topology; geometry and pair-function outputs require
matrix or COO. Cluster tiles require fully periodic float32 input. Partial naive
queries cannot combine with selective rebuilds.

The selector uses a narrower eligibility set than some explicit APIs. For
example, an explicitly chosen Torch pair-centric cell list can handle requests
that automatic selection excludes. If you need a specific supported route,
pin it and check the backend API. Passing `cutoff2` does not create a dual
cell-list or tiled-naive algorithm; see {doc}`neighborlist_advanced` for direct
calls and output contracts.

(nl_performance)=

## Get good performance on your workload

A useful comparison includes the work you will actually repeat. For a local
ML potential, that may be building a graph, packing edge geometry, and evaluating
the model. For short-range interactions, it may be construction plus pair
calculation. A fast search can lose its advantage if the next operation requires
an expensive layout conversion.

1. **Shortlist compatible routes.** Use the method descriptions and feature
   table. Start with scalar naive for small systems or selected rows, then try
   tiled naive for topology or spatial methods for larger local neighborhoods.
   Use automatic selection as a suggestion. For eligible full-list pair-centric
   calls, benchmark adaptive against configured grid sizing for systems with
   nearly uniform atomic density and batches of similar systems. See
   {ref}`configured and adaptive sizing <configured-and-adaptive-sizing>` for
   the restrictions; prepared states require configured sizing.
2. **Choose matrix capacity.** Use
   {ref}`Estimate max_neighbors <neighbor-list-capacity-estimation>` for an
   initial `max_neighbors` value. Allow room for the largest neighborhood you
   expect and check required counts on representative configurations. An average
   density does not bound local crowding; excessively wide matrices cost memory
   and consumer work.
3. **Compare equal work.** Keep geometry, cutoff, periodicity, fill mode, precision,
   and required outputs the same. Include setup, binning or sorting, query,
   packing, and the consumer where they belong in your application.
4. **Warm up and wait for CUDA.** Exclude first-time Warp compilation and
   Torch/JAX tracing unless startup is the quantity you need. Synchronize Torch
   with `torch.cuda.synchronize()` or wait on JAX results with
   `jax.block_until_ready(...)` around timed work.
5. **Separate rebuild and reuse steps.** Measure the frequency your application
   needs. A cached grid, reused bins, and a buffered neighbor topology save
   different work; see {doc}`neighborlist_repeated`.
   Move preparation and workspace setup outside the hot loop, keep capacities
   stable, and reuse caller-owned outputs and workspace where the route supports
   them. The {ref}`separate build/query example <build-query-separation>` shows
   Torch buffers in use. In JAX, keep shapes static under `jax.jit`; optional
   buffer donation can reuse storage, but do not retain or use donated aliases.
6. **Pin the measured choice.** Use a fine-grained method name or prepare an
   explicit method/strategy. Revisit it when the workload or hardware changes.

Choose the layout your consumer needs and avoid repeated
conversions or copies. For Torch matrix consumers, consider
{ref}`trimming unused columns <trim-neighbor-matrix>` to reduce padded work,
including its synchronization cost in your timing. See
{doc}`neighborlist_advanced` for storage, grid tuning, and capacity handling.

```{toctree}
:maxdepth: 1

neighborlist_repeated
neighborlist_advanced
```

For exact signatures, see the [PyTorch](../../modules/torch/neighbors.rst),
[JAX](../../modules/jax/neighbors.rst), and
[Warp](../../modules/warp/neighbors.rst) API references.

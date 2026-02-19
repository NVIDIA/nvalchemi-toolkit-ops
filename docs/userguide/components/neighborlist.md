<!-- markdownlint-disable MD013 -->

<!-- markdownlint-disable MD013 -->

(neighborlist_userguide)=

# Neighbor Lists

Neighbor lists identify atom pairs within a cutoff distance---the foundation for
all forms of interatomic interactions including but not limited to: machine-learned
interatomic potentials, dispersion corrections, and so on. ALCHEMI Toolkit-Ops provides
GPU-accelerated neighbor list algorithms via [NVIDIA Warp](https://nvidia.github.io/warp/)
with full `torch.compile` support.

```{tip}
Start with the unified {func}`~nvalchemiops.torch.neighbors.neighbor_list` function.
It automatically selects the best algorithm for your system size and handles
both single and batched inputs.
```

## Why Neighbor Lists Matter for Performance

Neighbor list construction can dominate runtime in atomistic foundation models:

- **Naive algorithms scale as \(O(N^2)\)**: Checking all atom pairs becomes
  prohibitive for systems with a large number of atoms (approx. 2000 atoms,
  but depends on structure and hardware)
- **Repeated construction**: Training loops and MD simulations rebuild neighbor
  lists frequently---every step or every few steps
- **Memory bandwidth**: Large neighbor matrices can bottleneck GPU throughput

ALCHEMI Toolkit-Ops addresses these challenges with O(N) cell list algorithms,
efficient batch processing for heterogeneous datasets, and memory layouts
optimized for GPU access patterns. See [performance considerations](nl_performance)
for guidance.

## Quick Start

The {func}`~nvalchemiops.torch.neighbors.neighbor_list` function provides a unified
interface that automatically dispatches to the optimal algorithm based on system
size and whether batch indices are provided.

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

Dispatches to {func}`~nvalchemiops.torch.neighbors.unbatched.cell_list` --- \(O(N)\) algorithm
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

Dispatches to {func}`~nvalchemiops.torch.neighbors.unbatched.naive_neighbor_list` --- \(O(N^2)\)
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

Dispatches to {func}`~nvalchemiops.torch.neighbors.batched.batch_cell_list` --- \(O(N)\)
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

Dispatches to {func}`~nvalchemiops.torch.neighbors.batched.batch_naive_neighbor_list` ---
\(O(N^2)\) algorithm for batched small systems.
:::

::::

```{note}
When `method` is not specified, `neighbor_list` automatically selects based on
average system size (greater than 2000 atoms per system) and whether `batch_idx` is provided.
The crossover point depends on system density and cutoff radius---benchmark
your workload to find the optimal threshold.
```

## Data Formats

ALCHEMI Toolkit-Ops supports two output formats for neighbor data:

Neighbor Matrix (default)
: Fixed-size tensor of shape `(num_atoms, max_neighbors)` where each row
  contains the neighbor indices for that atom, padded with a fill value.
  Returns `(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)`.

Neighbor List (COO format)
: Sparse tensor of shape `(2, num_pairs)` containing `[source_atoms, target_atoms]`.
  Returns `(neighbor_list, neighbor_ptr, neighbor_list_shifts)` where
  `neighbor_ptr` is a CSR-style pointer array. The first set of atoms (nominally
  `source_atoms`) is guaranteed to be sorted.

### When to Use Each Format

**Neighbor Matrix** is preferred when:

- Using `torch.compile` (fixed memory layout avoids graph breaks)
- Systems have dense, uniform neighbor distributions
- Cache-friendly access patterns are important

**Neighbor List (COO)** is preferred when:

- Integrating with graph neural network libraries (PyG, DGL)
- Systems are sparse with highly variable neighbors per atom
- Memory efficiency is critical

### Switching Formats

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

```{warning}
Setting `return_neighbor_list=True` incurs a conversion overhead. If you need
both formats, compute the matrix format first and convert as needed.
```

## Method Dispatch

When `method=None`, {func}`~nvalchemiops.torch.neighbors.neighbor_list` selects
an algorithm using the following logic:

1. If `cutoff2` is provided, then dual cutoff method
2. If average atoms per system `>= 2000`, then `"cell_list"`
3. Otherwise, `"naive"` ($N^2$ scaling algorithm)
4. If `batch_idx` or `batch_ptr` is provided, then prepend `"batch_"` to the method

### Available Methods

| Method | Algorithm | Use Case |
|--------|-----------|----------|
| `"naive"` | \(O(N^2)\) pairwise | Small single systems (<2000 atoms) |
| `"cell_list"` | \(O(N)\) spatial decomposition | Large single systems |
| `"batch_naive"` | \(O(N^2)\) per system | Batched small systems |
| `"batch_cell_list"` | \(O(N)\) per system | Batched large systems |
| `"naive_dual_cutoff"` | \(O(N^2)\) with two cutoffs | Multi-range potentials |
| `"batch_naive_dual_cutoff"` | Batched dual cutoff | Batched multi-range |

Override automatic selection by passing the `method` parameter:

```python
# Force cell_list on a small system for testing
neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, method="cell_list"
)
```

## Performance Tuning

(nl_performance)=

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
: Maximum number of spatial cells for cell list decomposition. Default is 1000.
  Limits memory usage for very large simulation boxes.

### Estimation Utilities

The {func}`~nvalchemiops.neighbors.neighbor_utils.estimate_max_neighbors` function estimates
the maximum number of neighbors $n$ any atom could have based on the cutoff sphere
volume ($r$) and atomic density $\rho$, with an additional safety factor ($S$):

$$
n = S \times \rho \times \frac{4}{3} \pi r^3
$$

```python
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.neighbors.unbatched import estimate_cell_list_sizes

max_neighbors = estimate_max_neighbors(
    cutoff,
    atomic_density=0.15,
    safety_factor=1.0
)

max_total_cells, neighbor_search_radius = estimate_cell_list_sizes(
    cell, pbc, cutoff, max_nbins=1000
)
```

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
the respective tensor containing the number of neighbors per atom, against
the maximum estimated number of neighbors. For optimal performance these
two factors should be close: if the actual number of neighbors per atom is
low relative to the estimated number, the allocated neighbor matrix will
be very sparse and memory inefficient (i.e. most elements will be padding).
If the actual number exceeds the estimate, neighborhoods will be truncated
and there is no guarantee that the nearest neighbors are included.
```

### Pre-allocation for Repeated Calculations

Pre-allocating output tensors avoids repeated memory allocation overhead when
computing neighbor lists in a loop (e.g., during MD simulation or training).
This also enables `torch.compile` compatibility by ensuring fixed tensor shapes.

```python
import torch
from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.torch.neighbors.neighbor_utils import estimate_max_neighbors

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
from nvalchemiops.torch.neighbors.unbatched import estimate_cell_list_sizes
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

```{warning}
If `max_neighbors` is too small, neighbors beyond that limit are silently
dropped. Monitor `num_neighbors.max()` against your `max_neighbors` setting
to detect truncation.
```

## Usage Patterns

### Basic Single System

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

### Batch Processing

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

### Half-Fill Mode

Store only half of neighbor pairs to avoid double-counting in symmetric
calculations:

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

### Build/Query Separation for MD Workflows

For molecular dynamics, separate building and querying allows caching the
spatial data structure:

```python
from nvalchemiops.torch.neighbors.unbatched import (
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

# MD loop
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

### Rebuild Detection with Skin Distance

Avoid rebuilding neighbor lists every step by using a skin distance:

```python
from nvalchemiops.torch.neighbors.unbatched import (
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

### Dual Cutoff

Compute two neighbor lists with different cutoffs simultaneously:

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

---

This concludes the high-level documentation for neighbor lists: you should now
be able to integrate `nvalchemiops` routines for your neighbor list requirements,
and consult the API reference for [PyTorch](../../modules/torch/neighbors)
and [Warp](../../modules/warp/neighbors) for further details.

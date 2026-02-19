(introduction_guide)=

# Introduction to ALCHEMI Toolkit-Ops

ALCHEMI Toolkit-Ops provides GPU-accelerated algorithms for atomistic simulations,
computational chemistry, and graph neural networks. Built on
[NVIDIA Warp](https://nvidia.github.io/warp/), it delivers high-performance
primitives for neighbor list construction, dispersion corrections, and related
operations on both single systems and batched datasets.

```{note}
If you need a quick way to get started, see the [User Guide](../index) for
installation and a working code example.
```

## When to Use ALCHEMI Toolkit-Ops

This package is designed for GPU-accelerated workflows in computational chemistry
and machine learning. Common use cases include:

Density Functional Theory
: Add DFT-D3(BJ) dispersion corrections for improved accuracy in weakly bound
  systems. Batch calculations enable high-throughput processing of molecular
  databases, and accurate forces support geometry optimization.

Graph Neural Networks
: Generate edge connectivity with neighbor lists in COO format. Batch processing
  supports high-throughput training on molecular datasets with heterogeneous
  system sizes.

Molecular Dynamics
: Construct neighbor lists for short-range interactions and dispersion forces.
  Skin distance optimization and rebuild detection reduce expensive list
  reconstructions during long simulations.

High-Throughput Screening
: Process large molecular databases with batch computation. Evaluate energies
  and forces for conformer analysis, virtual screening, and property prediction.

Method Development
: Access low-level Warp kernels for custom algorithm development with profiling
  and memory estimation utilities.

## Design Principles

ALCHEMI Toolkit-Ops prioritizes performance, correctness, and usability:

1. All algorithms are GPU-accelerated via NVIDIA Warp kernels
2. High-level APIs handle algorithm selection automatically; low-level kernels
   remain accessible for custom workflows
3. Outputs are PyTorch tensors with full `torch.compile` compatibility
4. Both dense (neighbor matrix) and sparse (COO) formats are supported

## Core Components

### Neighbor Finding

The {func}`~nvalchemiops.torch.neighbors.neighbor_list` function provides a unified
interface that automatically selects between algorithms based on system size
and whether batch indices are provided. It returns either a dense neighbor
matrix or sparse COO list, with consistent behavior across all modes.

```{tip}
The crossover point between cell list and naive algorithms depends on system
density and cutoff radius. We encourage users to benchmark the performance on
their workload and develop an intuition for which algorithm to use under what
circumstances.
```

Choose the right parameters based on your use case:

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

Dispatches to {func}`~nvalchemiops.torch.neighbors.unbatched.cell_list` — O(N) algorithm
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

Dispatches to {func}`~nvalchemiops.torch.neighbors.unbatched.naive_neighbor_list` — O(N²)
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

Dispatches to {func}`~nvalchemiops.torch.neighbors.batched.batch_cell_list` — O(N)
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

Dispatches to {func}`~nvalchemiops.torch.neighbors.batched.batch_naive_neighbor_list` —
O(N²) algorithm for batched small systems.
:::

::::

```{note}
When `method` is not specified, `neighbor_list` automatically selects based on
average system size (≥2000 atoms per system → cell list) and whether `batch_idx` is provided.
```

For advanced workflows, {func}`~nvalchemiops.torch.neighbors.build_cell_list` and
{func}`~nvalchemiops.torch.neighbors.query_cell_list` allow caching spatial data
structures between queries. Dual-cutoff variants are available for multi-range
potentials.

### Rebuild Detection

For molecular dynamics workflows,
{func}`~nvalchemiops.torch.neighbors.cell_list_needs_rebuild` and related functions
minimize expensive cell list reconstructions by detecting when atoms have moved
beyond a skin distance threshold.

### Dispersion Corrections

The {func}`~nvalchemiops.torch.interactions.dispersion.dftd3`
function computes DFT-D3(BJ) dispersion energy and forces with
environment-dependent C6 coefficients based on coordination numbers. It supports
both neighbor formats, periodic and non-periodic systems, and batched computation.

```{tip}
The damping parameters (`a1`, `a2`, `s8`) are functional-dependent. Common values
can be found in the [Grimme group DFT-D3 documentation](https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/bj_damping).
Positions and parameters must use consistent units (atomic units recommended).
```

Choose the right parameters based on your system type:

::::{tab-set}

:::{tab-item} Molecule
:sync: molecule

Non-periodic molecular system

```python
from nvalchemiops.torch.interactions.dispersion import dftd3
from nvalchemiops.torch.neighbors import neighbor_list

# Build neighbor list (positions in Bohr)
neighbors, neighbor_ptr, _ = neighbor_list(
    positions, cutoff=40.0, return_neighbor_list=True
)

# Compute D3 correction (PBE functional)
energy, forces, coord_num = dftd3(
    positions, numbers, neighbor_list=neighbors,
    a1=0.3981, a2=4.4211, s8=0.7875, d3_params=d3_params
)

```

:::

:::{tab-item} Periodic
:sync: periodic

Single periodic system (crystal, surface)

```python
from nvalchemiops.torch.interactions.dispersion import dftd3
from nvalchemiops.torch.neighbors import neighbor_list

# Build neighbor list with PBC
neighbors, neighbor_ptr, shifts = neighbor_list(
    positions, cutoff=40.0, cell=cell, pbc=pbc, return_neighbor_list=True
)

# Compute D3 correction with periodic shifts
energy, forces, coord_num = dftd3(
    positions, numbers, neighbor_list=neighbors,
    a1=0.3981, a2=4.4211, s8=0.7875, d3_params=d3_params,
    cell=cell, unit_shifts=shifts
)
```

:::

:::{tab-item} Batch
:sync: batch

Multiple systems processed simultaneously

```python
from nvalchemiops.torch.interactions.dispersion import dftd3
from nvalchemiops.torch.neighbors import neighbor_list

# Build batched neighbor list
neighbors, neighbor_ptr, shifts = neighbor_list(
    positions, cutoff=40.0, cell=cells, pbc=pbc,
    batch_idx=batch_idx, return_neighbor_list=True
)

# Compute D3 correction for all systems
energy, forces, coord_num = dftd3(
    positions, numbers, neighbor_list=neighbors,
    a1=0.3981, a2=4.4211, s8=0.7875, d3_params=d3_params,
    cell=cells, unit_shifts=shifts, batch_idx=batch_idx
)
```

Returns per-system energies with shape `(num_systems,)`.
:::

::::

```{note}
DFT-D3 parameters must be provided via the `d3_params` argument (as a
{class}`~nvalchemiops.torch.interactions.dispersion.D3Parameters` instance or
dict). See the [dispersion documentation](../components/dispersion) for parameter
setup and loading from standard reference files.
```

### Electrostatic Interactions

The {func}`~nvalchemiops.torch.interactions.electrostatics.ewald_summation` and
{func}`~nvalchemiops.torch.interactions.electrostatics.particle_mesh_ewald` functions
compute long-range Coulomb interactions in periodic systems. Both methods split the
slowly-converging $1/r$ potential into real-space (short-range) and reciprocal-space
(long-range) components, with automatic parameter estimation based on target accuracy.

```{tip}
For systems with <5000 atoms, use Ewald summation. For larger systems, PME provides
$O(N \log N)$ scaling via FFT-accelerated reciprocal-space calculations.
```

Choose the right method based on your system size:

::::{tab-set}

:::{tab-item} Ewald
:sync: ewald

Small to medium systems (<5000 atoms)

```python
from nvalchemiops.torch.interactions.electrostatics import ewald_summation
from nvalchemiops.torch.neighbors import neighbor_list

# Build neighbor list
neighbors, neighbor_ptr, shifts = neighbor_list(
    positions, cutoff=10.0, cell=cell, pbc=pbc, return_neighbor_list=True
)

# Compute electrostatics (parameters estimated automatically)
energies, forces = ewald_summation(
    positions, charges, cell, neighbor_list=neighbors,
    neighbor_ptr=neighbor_ptr, neighbor_shifts=shifts,
    accuracy=1e-6, compute_forces=True
)
```

Uses explicit k-vector summation in reciprocal space — $O(N^2)$ scaling.
:::

:::{tab-item} PME
:sync: pme

Large systems (>5000 atoms)

```python
from nvalchemiops.torch.interactions.electrostatics import particle_mesh_ewald
from nvalchemiops.torch.neighbors import neighbor_list

# Build neighbor list
neighbors, neighbor_ptr, shifts = neighbor_list(
    positions, cutoff=10.0, cell=cell, pbc=pbc, return_neighbor_list=True
)

# Compute electrostatics with FFT acceleration
energies, forces = particle_mesh_ewald(
    positions, charges, cell, neighbor_list=neighbors,
    neighbor_ptr=neighbor_ptr, neighbor_shifts=shifts,
    accuracy=1e-6, compute_forces=True
)
```

Uses FFT-based reciprocal-space calculation — $O(N \log N)$ scaling.
:::

:::{tab-item} Batch
:sync: batch

Multiple systems processed simultaneously

```python
from nvalchemiops.torch.interactions.electrostatics import ewald_summation
from nvalchemiops.torch.neighbors import neighbor_list

# Build batched neighbor list
neighbors, neighbor_ptr, shifts = neighbor_list(
    positions, cutoff=10.0, cell=cells, pbc=pbc,
    batch_idx=batch_idx, return_neighbor_list=True
)

# Batched electrostatics
energies, forces = ewald_summation(
    positions, charges, cell=cells, neighbor_list=neighbors,
    neighbor_ptr=neighbor_ptr, neighbor_shifts=shifts,
    batch_idx=batch_idx, accuracy=1e-6, compute_forces=True
)
```

Returns per-atom energies; sum by system using `batch_idx`.
:::

::::

## Ecosystem Integration

ALCHEMI Toolkit-Ops integrates with the scientific Python ecosystem:

### PyTorch

All inputs and outputs are PyTorch tensors with automatic CPU/GPU handling.
Custom operators are registered for `torch.compile` compatibility, and tensors
maintain gradients for automatic differentiation where applicable.

### NVIDIA Tools

Kernels are implemented in [NVIDIA Warp](https://nvidia.github.io/warp/) with
GPU memory layouts optimized for NVIDIA architectures. Standard profiling tools
like [Nsight Systems](https://developer.nvidia.com/nsight-systems) work out of
the box.

### Computational Chemistry

DFT-D3 parameters follow the standard format from the
[Grimme group](https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/),
ensuring compatibility with established workflows.

## What's Next?

1. Follow the [installation guide](install) to set up your environment
2. Read the [neighbor list documentation](../components/neighborlist) for
   spatial algorithm details
3. Explore [DFT-D3 dispersion corrections](../components/dispersion) for
   energy and force calculations
4. Learn about [electrostatic interactions](../components/electrostatics) for
   Ewald summation and PME calculations
5. Check the `examples/` directory for complete working scripts

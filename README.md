<!-- markdownlint-disable MD033 MD007 -->

# NVIDIA ALCHEMI Toolkit-Ops

[![PyPI version](https://badge.fury.io/py/nvalchemi-toolkit-ops.svg)](https://badge.fury.io/py/nvalchemi-toolkit-ops)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![codecov](https://codecov.io/gh/NVIDIA/nvalchemi-toolkit-ops/branch/main/graph/badge.svg)](https://codecov.io/gh/NVIDIA/nvalchemi-toolkit-ops)
[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue)](https://nvidia.github.io/nvalchemi-toolkit-ops/)

## High-performance NVIDIA Warp primitives for computational chemistry

NVIDIA ALCHEMI Toolkit-Ops is a collection of GPU-optimized, batched primitives for
accelerating atomistic simulations. High performance compute kernels are written
in [NVIDIA `warp-lang`](https://github.com/NVIDIA/warp).

### Key Features

- Neighborhood computation, including naive $O(N^2)$ and cell list $O(N)$
implementations
- Dispersion corrections via Becke-Johnson damped DFT-D3
- Electrostatic interactions, including Ewald and particle mesh Ewald algorithms

Kernels are naturally intended to be highly scalable (>100,000 atoms) and generally
optimized for high throughput operations (on the order of several microseconds per
atom) on GPUs, with batching support.

### Use Cases

There are currently three primary use cases where we imagine `nvalchemi-toolkit-ops` to
fit into the ecosystem:

- Library maintainers and developers are encouraged to benchmark and explore
integrating functionality like neighbor list computation to accelerate
existing workflows;
- Researchers and model developers ideally should be able to rely on
this package (and not implement their own!) for neighbor list computation,
interatomic interactions, and so on during method development;
- Engineers looking to build applications that involve molecular dynamics,
interatomic potentials, and the like can take advantage of optimized and
maintained low-level kernels. `warp-lang` kernels should be sufficiently
modular to allow for a high degree of flexibility and reusability.

The combination of being GPU-first and batched should enable the kernels contained
in `nvalchemi-toolkit-ops` to be ready for a wide range of research and production
applications.

### Example Snippets

We encourage interested readers to browse our hosted
[documentation](https://nvidia.github.io/nvalchemi-toolkit-ops). Below are some
short snippets that highlight our straightforward API and use cases:

<details>
<summary>Neighbor list in a 2D unit cell with 50,000 atoms</summary>

This example uses PyTorch:

```python
import torch
from nvalchemiops.neighborlist import neighbor_list

torch.set_default_dtype(torch.float32)
torch.set_default_device(torch.device("cuda"))

NUM_ATOMS = 50_000
# arbitrarily scale positions
positions = torch.randn((NUM_ATOMS, 3)) * 10.0
cell = torch.eye(3, dtype=torch.float32).unsqueeze(0)
pbc = torch.tensor([True, True, False], dtype=torch.bool)
cutoff = 6.0
# use padded matrix representation for neighbors, optimal for
# compiled applications that need constant shapes
neighbor_matrix, num_neighbors, shift_matrix = neighbor_list(
    positions,
    cutoff,
    cell=cell,
    pbc=pbc,
    method="cell_list"
)
# ...or pass `return_neighbor_list=True` for the familiar COO
# `edge_index` format. `method` will also automatically determine
# neighbor algorithm based off system size
edge_index, neighbor_ptr, shifts = neighbor_list(
    positions,
    cutoff,
    cell=cell,
    pbc=pbc,
    return_neighbor_list=True
)

```

</details>

<details>
<summary>DFT-D3(BJ) corrections on a batch of molecules</summary>

This example assumes you already have concatenated a set of molecules
into combined tensors, and have computed some form of neighborhood
using the `neighbor_list` API. Here, we'll demonstrate using the
matrix representation:

```python
import torch
from nvalchemiops.interactions.dispersion import dftd3

# the following parameters need to be constructed ahead of time
positions = ...  # [num_atoms, 3]
atomic_numbers = ...  # [num_atoms]
cell = ...  # [num_systems, 3, 3]
pbc = ...  # [num_systems, 3]
batch_idx = ...  # [num_atoms]
batch_ptr = ...  # [num_systems + 1]
# construct neighbor matrix
neighbor_matrix, num_neighbors, shift_matrix = neighbor_list(
    positions,
    cutoff=...,  # on the order of ~20 Angstroms
    cell=cell,
    pbc=pbc,
    batch_idx=batch_idx,
    batch_ptr=batch_ptr
)
# DFT-D3 parameters need to be provided, which comprises reference C6 parameters.
# Refer to the user documentation to see the expected structure and data source.
d3_params = ...
# pass everything to the functional interface
d3_energies, d3_forces, coord_nums, d3_virials = dftd3(
    positions=positions,
    numbers=atomic_numbers,
    neighbor_matrix=neighbor_matrix,
    neighbor_matrix_shifts=shift_matrix,
    batch_idx=batch_idx,
    # functional specific DFT-D3 parameters
    a1=..., a2=..., s8=...,
    d3_params=d3_params,
    compute_virial=True
)
```

</details>

<details>
<summary>Electrostatics via particle mesh Ewald</summary>

This example shows how to compute the per-atom and system energies
as well as the forces using the particle mesh Ewald interface.


```python
import torch
from nvalchemiops.interactions.electrostatics import particle_mesh_ewald

# the following parameters need to be constructed ahead of time
positions = ...  # [num_atoms, 3]
atomic_numbers = ...  # [num_atoms]
cell = ...  # [num_systems, 3, 3]
pbc = ...  # [num_systems, 3]
atomic_charges = ... # [num_atoms]
# construct neighbor matrix
neighbor_matrix, num_neighbors, shift_matrix = neighbor_list(
    positions,
    cutoff=...,  # on the order of ~20 Angstroms
    cell=cell,
    pbc=pbc,
)
# call PME, using automatic parameter tuning
atom_energies, atom_forces = particle_mesh_ewald(
    positions=positions,
    charges=atomic_charges,
    cell=cell,
    neighbor_matrix=neighbor_matrix,
    neighbor_matrix_shifts=shift_matrix,
    accuracy=1e-6
)
system_energy = atom_energies.sum()
```

</details>

## Roadmap

Currently, we have the following features that are being implemented, or are planned
to be implemented soon:

- A variety of widely used pair-potentials, including
    - Ziebler-Biersack-Littmark (ZBL)
    - Lennard-Jones
    - Quantum Drude Oscillator
- DFT-D4 dispersion corrections
- Primitives typically used with machine learned interatomic potentials, such as:
    - Polynomial basis functions
    - (Segmented) graph operations
    - Mixed precision, e.g. higher precision accumulation
- JAX wrappers

## Contributions & Disclaimers

Currently, NVIDIA ALCHEMI Toolkit-Ops is undergoing a public beta, where we are
soliciting feedback from the community. During this time, direct code contributions are
not accepted as our first priority will be to define and provide a stable API, which
is/will be subject to change. Feature requests, discussions, and general feedback are
welcome and encouraged via
[Github Issues](https://www.github.com/NVIDIA/nvalchemi-toolkit-ops).

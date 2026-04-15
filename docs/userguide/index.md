<!-- markdownlint-disable MD014 -->

(userguide)=

# User Guide

Welcome to the ALCHEMI Toolkit-Ops user guide: this side of the documentation
is to provide a high-level and conceptual understanding of the philosophy
and supported features in `nvalchemiops`.

## Quick Start

The quickest way to install ALCHEMI Toolkit-Ops:

```bash
$ pip install nvalchemi-toolkit-ops
```

To install ALCHEMI Toolkit-Ops with a deep-learning backend:

::::{tab-set}

:::{tab-item} PyTorch
:sync: torch

```bash
$ pip install 'nvalchemi-toolkit-ops[torch]'
```

:::

:::{tab-item} JAX
:sync: jax

```bash
$ pip install 'nvalchemi-toolkit-ops[jax]'
```

:::

::::

```{tip}
Running on **NVIDIA DGX Spark**? The Blackwell GPU requires CUDA 13 wheels for
PyTorch. See the [CUDA 13 installation notes](about/install.md#cuda-13-installation)
before proceeding.
```

Make sure it is importable:

```bash
$ python -c "import nvalchemiops; print(nvalchemiops.__version__)"
```

Try out some of the API; a good place to start is to compute
the neighbor matrix (or equivalently, list):

::::{tab-set}

:::{tab-item} PyTorch
:sync: torch

```python
import torch
from nvalchemiops.torch.neighbors import cell_list

# Create atomic system data
positions = torch.randn(1000, 3, device='cuda') * 25.0  # 1000 atoms
cell = torch.eye(3, device='cuda').unsqueeze(0) * 25.0  # 25x25x25 unit cell
pbc = torch.tensor([True, True, True], device='cuda')  # PBC
cutoff = 2.5  # Cutoff radius in Angstroms

# Compute neighbor matrix (default format)
neighbor_matrix, num_neighbors, shifts = cell_list(
    positions, cutoff, cell, pbc
)

# Or get neighbor list (COO format) for graph neural networks
neighbor_list, neighbor_ptr, shifts = cell_list(
    positions, cutoff, cell, pbc, return_neighbor_list=True
)
source_indices = neighbor_list[0]
target_indices = neighbor_list[1]

print(f"Found {neighbor_list.shape[1]} neighbor pairs")
# neighbor_ptr is a CSR-style pointer; compute num_neighbors from it
num_neighbors = neighbor_ptr[1:] - neighbor_ptr[:-1]
print(f"Average neighbors per atom: {num_neighbors.float().mean():.1f}")
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.neighbors import cell_list

# Create atomic system data
positions = jnp.array(jax.random.normal(jax.random.key(0), (1000, 3))) * 25.0
cell = jnp.eye(3).reshape(1, 3, 3) * 25.0  # 25x25x25 unit cell
pbc = jnp.array([True, True, True])  # PBC
cutoff = 2.5  # Cutoff radius in Angstroms

# Compute neighbor matrix (default format)
neighbor_matrix, num_neighbors, shifts = cell_list(
    positions, cutoff, cell, pbc
)

# Or get neighbor list (COO format) for graph neural networks
neighbor_list, neighbor_ptr, shifts = cell_list(
    positions, cutoff, cell, pbc, return_neighbor_list=True
)
source_indices = neighbor_list[0]
target_indices = neighbor_list[1]

print(f"Found {neighbor_list.shape[1]} neighbor pairs")
# neighbor_ptr is a CSR-style pointer; compute num_neighbors from it
num_neighbors = neighbor_ptr[1:] - neighbor_ptr[:-1]
print(f"Average neighbors per atom: {jnp.mean(num_neighbors.astype(jnp.float32)):.1f}")
```

:::

::::

See the [PyTorch API Reference](../modules/torch/neighbors.rst) and
[JAX API Reference](../modules/jax/neighbors.rst) for the full API documentation.

## About

- [Install](about/install)
- [Introduction](about/intro)
- [Conventions](about/conventions)

## Core Components

- [NeighborLists](components/neighborlist)
- [Electrostatics](components/electrostatics)
- [Dispersion Corrections](components/dispersion)
- [Dynamics](components/dynamics)

## Advanced Usage

```{toctree}
:caption: About
:maxdepth: 1
:hidden:

about/install
about/intro
about/conventions
about/migration
about/faq

```

```{toctree}
:caption: Core Components
:maxdepth: 1
:hidden:

components/neighborlist
components/electrostatics
components/dispersion
components/dynamics
```

```{toctree}
:caption: Advanced Usage
:maxdepth: 1
:hidden:

```

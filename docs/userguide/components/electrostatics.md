<!-- markdownlint-disable MD013 MD049 -->

(electrostatics_userguide)=

# Electrostatic Interactions

Electrostatic interactions arise from Coulombic forces between charged particles.
In periodic systems, the $1/r$ potential decays slowly, requiring special techniques
to handle the conditionally convergent lattice sum. ALCHEMI Toolkit-Ops provides
GPU-accelerated implementations of Ewald summation, two-dimensional slab
correction, Particle Mesh Ewald (PME), and Damped Shifted Force (DSF) electrostatics
via [NVIDIA Warp](https://nvidia.github.io/warp/). PyTorch and JAX bindings support
energy autograd where documented. Direct-output flags remain available for legacy
compatibility and component-level MD/inference workflows, but full Ewald/PME
training should derive forces, charge gradients, and stress from the returned
energy tensor.

```{tip}
For periodic systems, start with
{func}`~nvalchemiops.torch.interactions.electrostatics.ewald_summation` (PyTorch) /
{func}`~nvalchemiops.jax.interactions.electrostatics.ewald_summation` (JAX) or
{func}`~nvalchemiops.torch.interactions.electrostatics.particle_mesh_ewald` (PyTorch) /
{func}`~nvalchemiops.jax.interactions.electrostatics.particle_mesh_ewald` (JAX). For non-periodic
systems or large-scale simulations, consider approximate
{func}`~nvalchemiops.torch.interactions.electrostatics.dsf_coulomb` (PyTorch only) which provides $O(N)$ scaling
with smooth force continuity at the cutoff.
For slab-like systems with two periodic directions, use Ewald or PME with
`slab_correction=True` and `pbc=...` in PyTorch or JAX.
```

## Overview of Available Methods

ALCHEMI Toolkit-Ops provides electrostatics modules for point charges:

| Method | Scaling | Best For |
|--------|---------|----------|
| **Ewald Summation** | $O(N^2)$ | Small/medium systems (<5000 atoms), 2D slabs |
| **Particle Mesh Ewald** | $O(N \log N)$ | Large periodic systems |
| **Damped Shifted Force (DSF)** | $O(N)$ | Large systems, non-periodic |
| **Direct Coulomb** | $O(N^2)$ | Non-periodic or as real-space component |
| **Ewald Multipole** | $O(N^2)$ | Multipolar systems, small/medium |
| **PME Multipole** | $O(N \log N)$ | Multipolar systems, large |

All methods support:

- Single-system and batched calculations
- Periodic boundary conditions
- Automatic differentiation (see per-method details below)
- Both neighbor list (COO) and neighbor matrix formats

| Method | Position gradients | Charge gradients | Cell gradients |
|--------|--------------------|------------------|----------------|
| Ewald / PME | Autograd | Autograd | Autograd |
| Direct Coulomb | Autograd | Autograd | Autograd |
| DSF | Analytical forces | Analytical (straight-through) | Analytical virial (PBC) |

```{important}
For MLIP training with the full Ewald/PME APIs, call the function without
direct-output flags and derive forces, stress, and charge gradients from the
returned energy tensor. The full-api flags `compute_forces`,
`compute_charge_gradients`, `compute_virial`, and `hybrid_forces` are deprecated
and emit `DeprecationWarning`; component APIs such as `ewald_real_space`,
`ewald_reciprocal_space`, and `pme_reciprocal_space` keep direct outputs as
no-autograd MD/inference paths. See {ref}`energy-derivative-contract`
for the full migration recipe and performance guidance.

JAX Ewald/PME energy autograd supports first-order derivatives for positions,
charges, and strain-first virials. Higher-order JAX support is limited to tested
position and charge scalar losses; PME cell/stress/strain higher-order
derivatives are unsupported. There are no public Hessian or Jacobian APIs.
```

Torch and JAX electrostatics support `float32` and `float64` point-charge Ewald
and PME inputs. Keep positions, charges, cells, `alpha`, and precomputed metadata
in a consistent dtype within each call. The examples use `float64` because
reciprocal-space electrostatics and gradient checks are accuracy sensitive;
`float32` is supported when throughput is the priority.

## Quick Start

:::::::{tab-set}

::::::{tab-item} Ewald Summation
:sync: ewald

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
import torch

from nvalchemiops.torch.interactions.electrostatics import ewald_summation
from nvalchemiops.torch.neighbors import neighbor_list

positions = positions.detach().requires_grad_(True)

# Build neighbor list
neighbor_list_coo, neighbor_ptr, neighbor_shifts = neighbor_list(
    positions, cutoff=10.0, cell=cell, pbc=pbc, return_neighbor_list=True
)

# Compute electrostatics (parameters estimated automatically)
energies = ewald_summation(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    accuracy=5e-4,  # Target accuracy for parameter estimation
)
forces = -torch.autograd.grad(energies.sum(), positions)[0]
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.interactions.electrostatics import ewald_summation
from nvalchemiops.jax.neighbors import neighbor_list

# Build neighbor list
neighbor_list_coo, neighbor_ptr, neighbor_shifts = neighbor_list(
    positions, cutoff=10.0, cell=cell, pbc=pbc, return_neighbor_list=True
)

# Compute electrostatics (parameters estimated automatically)
def total_energy(pos):
    energies = ewald_summation(
        positions=pos,
        charges=charges,
        cell=cell,
        neighbor_list=neighbor_list_coo,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
        accuracy=5e-4,  # Target accuracy for parameter estimation
    )
    return jnp.sum(energies)

forces = -jax.grad(total_energy)(positions)
```

:::

::::

::::::

::::::{tab-item} Particle Mesh Ewald
:sync: pme

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
import torch

from nvalchemiops.torch.interactions.electrostatics import particle_mesh_ewald
from nvalchemiops.torch.neighbors import neighbor_list

positions = positions.detach().requires_grad_(True)

# Build neighbor list
neighbor_list_coo, neighbor_ptr, neighbor_shifts = neighbor_list(
    positions, cutoff=10.0, cell=cell, pbc=pbc, return_neighbor_list=True
)

# Compute electrostatics (parameters estimated automatically)
energies = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    accuracy=5e-4,
)
forces = -torch.autograd.grad(energies.sum(), positions)[0]
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.interactions.electrostatics import particle_mesh_ewald
from nvalchemiops.jax.neighbors import neighbor_list

# Build neighbor list
neighbor_list_coo, neighbor_ptr, neighbor_shifts = neighbor_list(
    positions, cutoff=10.0, cell=cell, pbc=pbc, return_neighbor_list=True
)

# Compute electrostatics (parameters estimated automatically)
def total_energy(pos):
    energies = particle_mesh_ewald(
        positions=pos,
        charges=charges,
        cell=cell,
        neighbor_list=neighbor_list_coo,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
        accuracy=5e-4,
    )
    return jnp.sum(energies)

forces = -jax.grad(total_energy)(positions)
```

:::

::::

::::::

::::::{tab-item} DSF Coulomb
:sync: dsf

```{note}
DSF Coulomb bindings are currently available for PyTorch only. See [JAX electrostatics API](../../modules/jax/electrostatics) for available JAX functions.
```

```python
from nvalchemiops.torch.interactions.electrostatics import dsf_coulomb
from nvalchemiops.torch.neighbors import neighbor_list

# Build full neighbor list
neighbor_list_coo, neighbor_ptr, neighbor_shifts = neighbor_list(
    positions, cutoff=10.0, cell=cell, pbc=pbc, return_neighbor_list=True
)

# Compute DSF electrostatics
energies, forces = dsf_coulomb(
    positions=positions,
    charges=charges,
    cutoff=10.0,
    alpha=0.2,  # Damping parameter (0.0 for undamped shifted-force)
    cell=cell,
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
    unit_shifts=neighbor_shifts,
    compute_forces=True,
)
```

::::::

::::::{tab-item} Direct Coulomb
:sync: coulomb

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
from nvalchemiops.torch.interactions.electrostatics import coulomb_energy_forces
from nvalchemiops.torch.neighbors import neighbor_list

# Build neighbor list
neighbor_list_coo, neighbor_ptr, neighbor_shifts = neighbor_list(
    positions, cutoff=10.0, cell=cell, pbc=pbc, return_neighbor_list=True
)

# Undamped Coulomb (alpha=0) or damped for Ewald real-space (alpha>0)
energies, forces = coulomb_energy_forces(
    positions=positions,
    charges=charges,
    cell=cell,
    cutoff=10.0,
    alpha=0.0,  # Set to >0 for damped (Ewald real-space)
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
)
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.interactions.electrostatics import coulomb_energy_forces
from nvalchemiops.jax.neighbors import neighbor_list

# Build neighbor list
neighbor_list_coo, neighbor_ptr, neighbor_shifts = neighbor_list(
    positions, cutoff=10.0, cell=cell, pbc=pbc, return_neighbor_list=True
)

# Undamped Coulomb (alpha=0) or damped for Ewald real-space (alpha>0)
energies, forces = coulomb_energy_forces(
    positions=positions,
    charges=charges,
    cell=cell,
    cutoff=10.0,
    alpha=0.0,  # Set to >0 for damped (Ewald real-space)
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
)
```

:::

::::

::::::

:::::::

## Data Formats

### Tensor Specifications

The table below lists out the general syntax and expected shapes for tensors used
in the electrostatics code. When possible to do so, we encourage developers and
users to align their variable naming to what is shown here for ease of debugging
and consistency.

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `positions` | `(N, 3)` | `float64/float32` | Atomic coordinates |
| `charges` | `(N,)` | `float64/float32` | Atomic partial charges |
| `cell` | `(1, 3, 3)` or `(B, 3, 3)` | `float64/float32` | Unit cell lattice vectors (rows) |
| `pbc` | `(1, 3)` or `(B, 3)` | `bool` | Periodic boundary conditions per axis |
| `batch_idx` | `(N,)` | `int32` | System index for each atom (batched only) |
| `alpha` | `float` or `(B,)` tensor | `float64/float32` | Ewald splitting parameter |

### Output Data Types

Internal reductions use `float64` where needed for numerical stability. Full
framework API energy outputs follow the input floating-point precision unless a
specific component documents a `float64` output. Forces and virials match the
input precision; charge gradients are `float64` for direct electrostatics
component outputs that accumulate charge potentials.

### Neighbor Representations

The electrostatics functions accept neighbors in two formats:

**Neighbor List (COO)**: Shape `(2, num_pairs)` where row 0 contains source indices
and row 1 contains target indices. Each pair is listed once. Provide with
`neighbor_list` and `neighbor_shifts` arguments.

**Neighbor Matrix**: Shape `(N, max_neighbors)` where each row contains neighbor
indices for that atom, padded with `fill_value`. Provide with `neighbor_matrix`
and `neighbor_matrix_shifts` arguments.

```{tip}
See the [neighbor list documentation](neighborlist_userguide) for API usage and performance considerations
when deciding between COO and matrix representations.
```

## Ewald Summation

### Mathematical Background

The Ewald method splits the slowly-converging Coulomb sum into four components:

```{math}
E_{\text{total}} = E_{\text{real}} + E_{\text{reciprocal}} - E_{\text{self}} - E_{\text{background}}
```

**Real-Space (Short-Range)**:

```{math}
E_{\text{real}} = \frac{1}{2} \sum_{i \neq j} q_i q_j \frac{\text{erfc}(\alpha r_{ij})}{ r_{ij}}
```

The complementary error function $\text{erfc}(\alpha r)$ rapidly damps interactions beyond
approximately $r \approx 3/\alpha$, confining contributions to a local neighborhood.

**Reciprocal-Space (Long-Range)**:

```{math}
E_{\text{reciprocal}} = \frac{1}{2V} \sum_{\mathbf{k} \neq 0}
\frac{4\pi}{k^2} \exp\left(-\frac{k^2}{4\alpha^2}\right) |S(\mathbf{k})|^2
```

where the structure factor is:

```{math}
S(\mathbf{k}) = \sum_j q_j \exp(i\mathbf{k} \cdot \mathbf{r}_j)
```

**Self-Energy Correction**:

```{math}
E_{\text{self}} = \frac{\alpha}{\sqrt{\pi}} \sum_i q_i^2
```

Removes the spurious self-interaction introduced by the Gaussian charge distribution.

**Background Correction** (for non-neutral systems):

```{math}
E_{\text{background}} = \frac{\pi}{2\alpha^2 V} Q_{\text{total}}^2
```

### Usage Examples

The full Ewald API returns per-atom energy by default. Derive training forces
from that energy; direct-output flags are legacy compatibility outputs and emit
`DeprecationWarning`. Snippets in this section that still request
`compute_forces=True`, `compute_charge_gradients=True`, or `compute_virial=True`
show the legacy direct-output tuple contract.

#### Explicit Parameters

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
import torch

from nvalchemiops.torch.interactions.electrostatics import ewald_summation

positions = positions.detach().requires_grad_(True)
energies = ewald_summation(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=0.3,        # Ewald splitting parameter
    k_cutoff=8.0,     # Reciprocal-space cutoff in inverse length
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
)
forces = -torch.autograd.grad(energies.sum(), positions)[0]
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.interactions.electrostatics import ewald_summation

def total_energy(pos):
    energies = ewald_summation(
        positions=pos,
        charges=charges,
        cell=cell,
        alpha=0.3,        # Ewald splitting parameter
        k_cutoff=8.0,     # Reciprocal-space cutoff in inverse length
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
    )
    return jnp.sum(energies)

forces = -jax.grad(total_energy)(positions)
```

:::

::::

#### Automatic Parameter Estimation

When `alpha` or `k_cutoff` are not provided, they are estimated based on `accuracy`:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
energies, forces = ewald_summation(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    accuracy=1e-6,  # Target relative error
    compute_forces=True,
)
```

:::

:::{tab-item} JAX
:sync: jax

```python
energies, forces = ewald_summation(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    accuracy=1e-6,  # Target relative error
    compute_forces=True,
)
```

:::

::::

The estimation uses the Kolafa-Perram formula:

```{math}
\eta = \left(\frac{V^2}{N}\right)^{1/6} / \sqrt{2\pi}
```

```{math}
\alpha = \frac{1}{2 \cdot \eta}, \quad
r_{\text{cutoff}} = \sqrt{-2 \ln \varepsilon} \cdot \eta, \quad
k_{\text{cutoff}} = \sqrt{-2 \ln \varepsilon} / \eta
```

```{tip}
Refer to the [Parameter Estimation](parameter-estimation) section for API usage.
```

#### 2D Slab Correction

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

For slab-like systems with two periodic directions and one non-periodic direction,
PyTorch Ewald can add the Yeh-Berkowitz / Ballenegger-Arnold-Cerdà slab
correction. Pass `slab_correction=True` and a boolean `pbc` tensor with exactly
one `False` entry; that entry marks the non-periodic axis:

```python
import torch

from nvalchemiops.torch.interactions.electrostatics import ewald_summation
from nvalchemiops.torch.neighbors import neighbor_list

pbc_slab = torch.tensor([[True, True, False]], dtype=torch.bool, device=positions.device)

# The neighbor list controls real-space periodic images. For this slab setup,
# use the same T/T/F periodicity and a cell with enough vacuum along z.
neighbor_list_coo, neighbor_ptr, neighbor_shifts = neighbor_list(
    positions,
    cutoff=5.0,
    cell=cell,
    pbc=pbc_slab,
    return_neighbor_list=True,
)

energies, forces = ewald_summation(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=0.3,
    k_cutoff=8.0,
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    pbc=pbc_slab,
    slab_correction=True,
    compute_forces=True,
)
```

:::

:::{tab-item} JAX
:sync: jax

JAX Ewald supports the same explicit-output slab correction. Pass
`slab_correction=True` and request forces, charge gradients, or virials with the
usual flags:

```python
import jax.numpy as jnp

from nvalchemiops.jax.interactions.electrostatics import ewald_summation
from nvalchemiops.jax.neighbors import neighbor_list

pbc_slab = jnp.array([[True, True, False]], dtype=jnp.bool_)

# The neighbor list controls real-space periodic images. For this slab setup,
# use the same T/T/F periodicity and a cell with enough vacuum along z.
neighbor_list_coo, neighbor_ptr, neighbor_shifts = neighbor_list(
    positions,
    cutoff=5.0,
    cell=cell,
    pbc=pbc_slab,
    return_neighbor_list=True,
)

energies, forces, charge_grads = ewald_summation(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=0.3,
    k_cutoff=8.0,
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    pbc=pbc_slab,
    slab_correction=True,
    compute_forces=True,
    compute_charge_gradients=True,
)
```

:::

::::

```{tip}
For batched slab simulations, pass `pbc` as an explicit contiguous `(B, 3)`
tensor so each system carries its own slab geometry.
```

For an orthorhombic slab with non-periodic $z$ direction, total charge
$Q = \sum_i q_i$, dipole moment $M_z = \sum_i q_i z_i$, second moment
$M_{z^2} = \sum_i q_i z_i^2$, box length $L_z$, and volume $V$, the correction is:

```{math}
E_\mathrm{slab}
= \frac{2\pi}{V}
\left(M_z^2 - Q M_{z^2} - \frac{Q^2 L_z^2}{12}\right)
```

The per-atom contribution used by the slab kernels is:

```{math}
e_i
= \frac{2\pi}{V} q_i
\left[
z_i M_z
- \frac{1}{2}\left(M_{z^2} + Q z_i^2\right)
- \frac{Q L_z^2}{12}
\right]
```

with force:

```{math}
\mathbf{F}_i^\mathrm{slab}
= -\frac{4\pi}{V} q_i \left(M_z - Q z_i\right)\hat{\mathbf{z}}.
```

For neutral systems ($Q=0$), this reduces to the Yeh-Berkowitz slab correction,
$E_\mathrm{slab}=2\pi M_z^2/V$. For triclinic cells, Toolkit-Ops uses the
normal-following form: replace $z_i$ by the projected coordinate
$\mathbf{r}_i\cdot\hat{\mathbf{n}}$ and $L_z$ by the projected cell height.

#### Separating real- and reciprocal-space

When either components are required individually, the following code can
be used instead of the high level wrapper to compute the contributions
directly:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
from nvalchemiops.torch.interactions.electrostatics import (
    ewald_real_space, ewald_reciprocal_space, generate_k_vectors_ewald_summation,
)

# Real-space only (short-range, damped Coulomb)
alpha = torch.tensor([0.3], dtype=positions.dtype, device=positions.device)
real_energies, real_forces = ewald_real_space(
    positions, charges, cell, alpha=alpha,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
)

# Reciprocal-space only (long-range, smooth)
k_vectors = generate_k_vectors_ewald_summation(cell.detach(), k_cutoff=8.0)
recip_energies, recip_forces = ewald_reciprocal_space(
    positions, charges, cell, k_vectors, alpha, compute_forces=True,
)
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
from nvalchemiops.jax.interactions.electrostatics import (
    ewald_real_space,
    ewald_reciprocal_space,
    generate_k_vectors_ewald_summation,
)

# Real-space only (short-range, damped Coulomb)
real_energies, real_forces = ewald_real_space(
    positions, charges, cell, alpha=0.3,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
)

# Reciprocal-space only (long-range, smooth)
k_vectors = generate_k_vectors_ewald_summation(
    jax.lax.stop_gradient(cell), k_cutoff=8.0
)
recip_energies, recip_forces = ewald_reciprocal_space(
    positions, charges, cell, k_vectors, alpha=0.3, compute_forces=True,
)
```

:::

::::

```{note}
The sum of real and reciprocal components gives the Ewald energy.
The self-energy and background corrections are embedded within
the reciprocal energy.
```

When using the Ewald component functions for slab-like systems, add the slab
correction explicitly after computing the 3D-periodic real- and reciprocal-space
parts:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
from nvalchemiops.torch.interactions.electrostatics import compute_slab_correction

slab_energies, slab_forces = compute_slab_correction(
    positions=positions,
    charges=charges,
    cell=cell,
    pbc=pbc_slab,
    compute_forces=True,
)

ewald_slab_energies = real_energies + recip_energies + slab_energies
ewald_slab_forces = real_forces + recip_forces + slab_forces
```

:::

:::{tab-item} JAX
:sync: jax

```python
from nvalchemiops.jax.interactions.electrostatics import compute_slab_correction

slab_energies, slab_forces = compute_slab_correction(
    positions=positions,
    charges=charges,
    cell=cell,
    pbc=pbc_slab,
    compute_forces=True,
)

ewald_slab_energies = real_energies + recip_energies + slab_energies
ewald_slab_forces = real_forces + recip_forces + slab_forces
```

:::

::::

## Particle Mesh Ewald (PME)

### Mathematical Background

For very large atomic systems, the particle mesh Ewald (PME) algorithm
provides substantial improvements in computational performance over
conventional Ewald summation. PME accelerates the reciprocal-space sum
using fast Fourier transforms by:

1. **Charge Assignment**: Spread charges onto a mesh using B-spline interpolation
2. **Forward FFT**: Transform charge mesh to reciprocal space
3. **Convolution**: Multiply by Green's function in k-space
4. **Inverse FFT**: Transform back to get potentials/electric field
5. **Force Interpolation**: Gather forces at atomic positions

The B-spline interpolation is corrected by per-axis modulus tables in the PME
influence function. For a nonzero reciprocal grid vector, Toolkit-Ops uses the
convolution kernel

```{math}
G(\mathbf{k}) =
\frac{4\pi}{V}
\frac{\exp\left(-k^2 / 4\alpha^2\right)}{k^2}
\frac{1}{M_x(k_x) M_y(k_y) M_z(k_z)}
```

where $M_x$, $M_y$, and $M_z$ are the one-dimensional B-spline modulus tables for
the chosen spline order. The reciprocal energy keeps the usual final one-half
factor from $E = \frac{1}{2}\sum_i q_i \phi_i$.

### Usage Examples

The full PME API follows the same contract as full Ewald: use energy autograd
for training derivatives, and reserve direct-output flags for legacy migration
checks. The reciprocal component API, `pme_reciprocal_space`, remains the
direct-output escape hatch for no-autograd MD/inference loops. Snippets in this
section that still request full-API direct outputs show compatibility behavior
and emit `DeprecationWarning`.

For JAX PME under `jax.jit` or other JAX transformations, pass explicit
`mesh_dimensions` when `cell`, `alpha`, or batch metadata are traced. For JAX
Ewald under `jax.jit`, pass `miller_bounds` as a concrete static tuple or build
`k_vectors` outside jit; dynamic `miller_bounds` changes the reciprocal array
shape and is not traceable.
`mesh_spacing` and `accuracy`-based mesh sizing need concrete mesh setup values.

#### Basic Usage

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
import torch

from nvalchemiops.torch.interactions.electrostatics import particle_mesh_ewald

positions = positions.detach().requires_grad_(True)
energies = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=0.3,
    mesh_dimensions=(32, 32, 32),  # FFT mesh size
    spline_order=4,                 # B-spline order (4 = cubic)
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
)
forces = -torch.autograd.grad(energies.sum(), positions)[0]
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp

from nvalchemiops.jax.interactions.electrostatics import particle_mesh_ewald

def total_energy(pos):
    energies = particle_mesh_ewald(
        positions=pos,
        charges=charges,
        cell=cell,
        alpha=0.3,
        mesh_dimensions=(32, 32, 32),  # FFT mesh size
        spline_order=4,                 # B-spline order (4 = cubic)
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
    )
    return jnp.sum(energies)

forces = -jax.grad(total_energy)(positions)
```

:::

::::

#### Precomputed PME Moduli

For fixed mesh dimensions and spline order, precompute the one-dimensional
B-spline modulus tables once and pass them to PME:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
from nvalchemiops.torch.interactions.electrostatics import (
    compute_bspline_moduli_1d,
    particle_mesh_ewald,
)

mesh_dimensions = (32, 32, 32)
spline_order = 4
miller_x = torch.fft.fftfreq(
    mesh_dimensions[0], d=1.0 / mesh_dimensions[0],
    device=positions.device, dtype=positions.dtype,
)
miller_y = torch.fft.fftfreq(
    mesh_dimensions[1], d=1.0 / mesh_dimensions[1],
    device=positions.device, dtype=positions.dtype,
)
miller_z = torch.fft.rfftfreq(
    mesh_dimensions[2], d=1.0 / mesh_dimensions[2],
    device=positions.device, dtype=positions.dtype,
)

energies = particle_mesh_ewald(
    positions, charges, cell,
    alpha=0.3,
    mesh_dimensions=mesh_dimensions,
    spline_order=spline_order,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    moduli_x=compute_bspline_moduli_1d(miller_x, mesh_dimensions[0], spline_order),
    moduli_y=compute_bspline_moduli_1d(miller_y, mesh_dimensions[1], spline_order),
    moduli_z=compute_bspline_moduli_1d(miller_z, mesh_dimensions[2], spline_order),
)
```

:::

:::{tab-item} JAX
:sync: jax

```python
from nvalchemiops.jax.interactions.electrostatics import (
    compute_bspline_moduli_1d,
    particle_mesh_ewald,
)

mesh_dimensions = (32, 32, 32)
spline_order = 4
miller_x = jnp.fft.fftfreq(mesh_dimensions[0], d=1.0 / mesh_dimensions[0])
miller_y = jnp.fft.fftfreq(mesh_dimensions[1], d=1.0 / mesh_dimensions[1])
miller_z = jnp.fft.rfftfreq(mesh_dimensions[2], d=1.0 / mesh_dimensions[2])

energies = particle_mesh_ewald(
    positions, charges, cell,
    alpha=0.3,
    mesh_dimensions=mesh_dimensions,
    spline_order=spline_order,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    moduli_x=compute_bspline_moduli_1d(miller_x, mesh_dimensions[0], spline_order),
    moduli_y=compute_bspline_moduli_1d(miller_y, mesh_dimensions[1], spline_order),
    moduli_z=compute_bspline_moduli_1d(miller_z, mesh_dimensions[2], spline_order),
)
```

:::

::::

#### Mesh Spacing

Instead of explicit mesh dimensions, specify mesh spacing:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
energies = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=0.3,
    mesh_spacing=0.5,  # Angstrom (or your length unit)
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
)
forces = -torch.autograd.grad(energies.sum(), positions)[0]
```

:::

:::{tab-item} JAX
:sync: jax

```python
energies = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=0.3,
    mesh_spacing=0.5,  # Angstrom (or your length unit)
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
)
forces = -jax.grad(lambda pos: particle_mesh_ewald(
    pos, charges, cell,
    alpha=0.3,
    mesh_spacing=0.5,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
).sum())(positions)
```

:::

::::

#### Automatic Parameter Estimation

Similar to the Ewald summation interface, PME accepts an `accuracy` parameter
that can be used to automatically determine sensible $\alpha$ and mesh:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
energies = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    accuracy=4e-5,  # Estimates alpha and mesh dimensions
)
forces = -torch.autograd.grad(energies.sum(), positions)[0]
```

:::

:::{tab-item} JAX
:sync: jax

```python
energies = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    accuracy=4e-5,  # Estimates alpha and mesh dimensions
)
forces = -jax.grad(lambda pos: particle_mesh_ewald(
    pos, charges, cell,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    accuracy=4e-5,
).sum())(positions)
```

:::

::::

```{note}
We encourage users to properly benchmark performance gains
afforded by `accuracy` on their systems of interest. The lower
the value of `accuracy` the more precise, at the cost of higher
computational requirements.
```

#### 2D Slab Correction with PME

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

For slab-like systems with two periodic directions, full PyTorch PME supports the
same slab correction as Ewald. Pass `slab_correction=True` and a boolean `pbc`
tensor with exactly one `False` entry; that entry marks the non-periodic axis:

```python
import torch

from nvalchemiops.torch.interactions.electrostatics import particle_mesh_ewald
from nvalchemiops.torch.neighbors import neighbor_list

pbc_slab = torch.tensor([[True, True, False]], dtype=torch.bool, device=positions.device)

# The neighbor list controls real-space periodic images. For this slab setup,
# use the same T/T/F periodicity and a cell with enough vacuum along z.
neighbor_list_coo, neighbor_ptr, neighbor_shifts = neighbor_list(
    positions,
    cutoff=5.0,
    cell=cell,
    pbc=pbc_slab,
    return_neighbor_list=True,
)

energies, forces = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=0.3,
    mesh_dimensions=(32, 32, 32),
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    pbc=pbc_slab,
    slab_correction=True,
    compute_forces=True,
)
```

The full PME interface adds the same slab correction described in the Ewald
section to the 3D-periodic real-space and PME reciprocal-space terms.

When using the PME reciprocal-space component directly, add the slab correction
explicitly:

```python
from nvalchemiops.torch.interactions.electrostatics import (
    compute_slab_correction,
    ewald_real_space,
    pme_reciprocal_space,
)

alpha = torch.tensor([0.3], dtype=positions.dtype, device=positions.device)

real_energies, real_forces = ewald_real_space(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=alpha,
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
)

pme_reciprocal_energies, pme_reciprocal_forces = pme_reciprocal_space(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=alpha,
    mesh_dimensions=(32, 32, 32),
    compute_forces=True,
)

slab_energies, slab_forces = compute_slab_correction(
    positions=positions,
    charges=charges,
    cell=cell,
    pbc=pbc_slab,
    compute_forces=True,
)

pme_slab_energies = real_energies + pme_reciprocal_energies + slab_energies
pme_slab_forces = real_forces + pme_reciprocal_forces + slab_forces
```

:::

:::{tab-item} JAX
:sync: jax

For legacy migration checks, full JAX PME still accepts the same slab correction
and explicit-output flags. The snippet below shows that compatibility tuple. New
differentiable training code should omit these flags and differentiate the
returned energy:

```python
import jax
import jax.numpy as jnp

from nvalchemiops.jax.interactions.electrostatics import particle_mesh_ewald
from nvalchemiops.jax.neighbors import neighbor_list

pbc_slab = jnp.array([[True, True, False]], dtype=jnp.bool_)

# The neighbor list controls real-space periodic images. For this slab setup,
# use the same T/T/F periodicity and a cell with enough vacuum along z.
neighbor_list_coo, neighbor_ptr, neighbor_shifts = neighbor_list(
    positions,
    cutoff=5.0,
    cell=cell,
    pbc=pbc_slab,
    return_neighbor_list=True,
)

energies, forces, charge_grads = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=0.3,
    mesh_dimensions=(32, 32, 32),
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    pbc=pbc_slab,
    slab_correction=True,
    compute_forces=True,
    compute_charge_gradients=True,
)
```

The full JAX PME interface adds the same slab correction described in the Ewald
section; component-wise PME composition uses `compute_slab_correction(...)` in
the same way as the PyTorch snippet above.

:::

::::

### PME vs Ewald: When to Use Each

| Criterion | Ewald | PME |
|-----------|-------|-----|
| System size | $<5000$ atoms | Any size |
| Scaling | $O(N^2)$ | $O(N \log N)$ |
| Setup overhead | Lower | Higher (FFT setup) |
| Accuracy control | `k_cutoff` | Mesh resolution |
| Memory | Low | Mesh memory $(n_x \times n_y \times n_z)$ |

For small systems, direct Ewald may be faster due to lower overhead. For large
systems, PME's $O(N \log N)$ scaling provides substantial speedup.

## Damped Shifted Force (DSF)

```{note}
DSF Coulomb bindings are currently available for PyTorch only. See [JAX electrostatics API](../../modules/jax/electrostatics) for available JAX functions.
```

### Motivation

Standard truncation of the $1/r$ Coulomb potential at a cutoff radius introduces
two fundamental problems in molecular simulations:

- **Charge imbalance**: The truncation sphere is generally not charge-neutral,
  causing long-range potential oscillations and systematic errors in thermodynamic
  properties.
- **Force discontinuities**: Atoms crossing the cutoff boundary experience
  instantaneous jumps in force, injecting energy into the system and violating
  energy conservation during molecular dynamics.

The Damped Shifted Force (DSF) method, introduced by Fennell and Gezelter (2006),
solves both problems through a pairwise, real-space $\mathcal{O}(N)$ electrostatic
summation technique. The core idea (building on the earlier Wolf summation) is that
the neglected environment beyond the cutoff can be approximated from local structure:
a neutralizing "image charge" is placed on the surface of the cutoff sphere for every
charge within it. A shifted-force construction then ensures both the potential energy
and the force smoothly vanish at the cutoff radius $R_c$.

```{tip}
DSF is particularly well-suited for non-periodic systems (clusters, droplets,
interfaces) and extremely large systems where the $\mathcal{O}(N)$ scaling
provides significant speedups over Ewald-based methods.
```

### Mathematical Background

#### Shifted-Force Construction

For a generic pair potential $v(r)$, the shifted-force form ensures both the
potential and its derivative (force) vanish at the cutoff:

```{math}
V_{\text{SF}}(r) = v(r) - v(R_c) - v'(R_c)(r - R_c), \quad r \le R_c
```

This guarantees $V_{\text{SF}}(R_c) = 0$ and $F_{\text{SF}}(R_c) = -V'_{\text{SF}}(R_c) = 0$.

For DSF, the base kernel is the damped Coulomb interaction
$v(r) = \text{erfc}(\alpha r) / r$, where the complementary error function
screens the interaction similarly to the real-space part of Ewald summation.

#### DSF Pair Potential

The potential energy for a pair of charges $i$ and $j$ at distance $r_{ij} \le R_c$:

```{math}
V_{\text{DSF}}(r_{ij}) = q_i q_j \left[ \frac{\text{erfc}(\alpha r_{ij})}{r_{ij}}
- \frac{\text{erfc}(\alpha R_c)}{R_c}
+ \left( \frac{\text{erfc}(\alpha R_c)}{R_c^2}
+ \frac{2\alpha}{\sqrt{\pi}} \frac{e^{-\alpha^2 R_c^2}}{R_c}
\right)(r_{ij} - R_c) \right]
```

For $r_{ij} > R_c$, $V_{\text{DSF}}(r_{ij}) = 0$.

The three terms have clear physical interpretations:

- **Damped Coulomb** ($\text{erfc}(\alpha r)/r$): The screened interaction between the charges.
- **Potential shift** ($-\text{erfc}(\alpha R_c)/R_c$): Charge neutralization on the cutoff sphere, ensuring $V(R_c) = 0$.
- **Force shift** (linear in $r - R_c$): Ensures the derivative (force) also vanishes at $R_c$, preventing energy drift.

#### DSF Force

The force between charges at distance $r_{ij} \le R_c$:

```{math}
\mathbf{F}_{\text{DSF}}(r_{ij}) = q_i q_j \left[ \left(
\frac{\text{erfc}(\alpha r_{ij})}{r_{ij}^2}
+ \frac{2\alpha}{\sqrt{\pi}} \frac{e^{-\alpha^2 r_{ij}^2}}{r_{ij}}
\right) - \left(
\frac{\text{erfc}(\alpha R_c)}{R_c^2}
+ \frac{2\alpha}{\sqrt{\pi}} \frac{e^{-\alpha^2 R_c^2}}{R_c}
\right) \right] \frac{\mathbf{r}_{ij}}{r_{ij}}
```

The subtracted constant ensures the force magnitude is exactly zero at $r_{ij} = R_c$.

#### Self-Energy Correction

Each charge interacts with its own neutralizing image charge on the cutoff sphere.
This self-energy must be subtracted:

```{math}
U_i^{\text{self}} = -\left(
\frac{\text{erfc}(\alpha R_c)}{2 R_c}
+ \frac{\alpha}{\sqrt{\pi}} \right) q_i^2
```

#### Total System Energy

The total DSF electrostatic energy is:

```{math}
U_{\text{elec}} = \frac{1}{2} \sum_{i} \sum_{j \neq i} V_{\text{DSF}}(r_{ij}) + \sum_i U_i^{\text{self}}
```

```{note}
The implementation assumes a **full neighbor list** where each pair $(i, j)$ appears
in both directions. The factor of $1/2$ accounts for this double counting.
```

### Usage Examples

#### Basic Energy and Forces

```python
from nvalchemiops.torch.interactions.electrostatics import dsf_coulomb
from nvalchemiops.torch.neighbors import neighbor_list

# Build full neighbor list
neighbor_list_coo, neighbor_ptr, neighbor_shifts = neighbor_list(
    positions, cutoff=10.0, cell=cell, pbc=pbc, return_neighbor_list=True
)

# Compute DSF energy and forces
energy, forces = dsf_coulomb(
    positions=positions,
    charges=charges,
    cutoff=10.0,
    alpha=0.2,
    cell=cell,
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
    unit_shifts=neighbor_shifts,
    compute_forces=True,
)
```

#### With Periodic Boundary Conditions and Virial

```python
energy, forces, virial = dsf_coulomb(
    positions=positions,
    charges=charges,
    cutoff=10.0,
    alpha=0.2,
    cell=cell,
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
    unit_shifts=neighbor_shifts,
    compute_forces=True,
    compute_virial=True,
)
# energy: (num_systems,), dtype=float64
# forces: (num_atoms, 3), dtype matches input
# virial: (num_systems, 3, 3), dtype matches input
```

#### Using Neighbor Matrix Format

```python
from nvalchemiops.torch.neighbors import cell_list

# Build neighbor matrix
neighbor_matrix, num_neighbors, shifts = cell_list(
    positions, cutoff=10.0, cell=cell, pbc=pbc
)

energy, forces = dsf_coulomb(
    positions=positions,
    charges=charges,
    cutoff=10.0,
    alpha=0.2,
    cell=cell,
    neighbor_matrix=neighbor_matrix,
    neighbor_matrix_shifts=shifts,
    compute_forces=True,
)
```

#### Charge Gradients for MLIP Training

For machine learning interatomic potentials (MLIPs) with geometry-dependent charges,
DSF supports charge gradient computation through PyTorch autograd:

```python
# Charges predicted by a neural network (requires_grad flows from the model)
charges = charge_model(positions, atomic_numbers)

energy, forces = dsf_coulomb(
    positions=positions,
    charges=charges,
    cutoff=12.0,
    alpha=0.2,
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
)

# Backpropagate through charges
loss = (energy - ref_energy).pow(2).sum()
loss.backward()
# charges.grad now contains dE/dq * dloss/dE
```

```{note}
Charge gradients ($\partial E / \partial q_i$) are computed analytically by the
Warp kernel and propagated through PyTorch autograd via a "straight-through trick."
The returned ``energy`` tensor is **not** differentiable with respect to ``positions``
or ``cell`` through autograd -- forces and virials are computed analytically by the kernel.
```

#### Batched Calculations

```python
import torch
from nvalchemiops.torch.interactions.electrostatics import dsf_coulomb

# Concatenate atoms from multiple systems
positions = torch.cat([pos_sys0, pos_sys1])
charges = torch.cat([charges_sys0, charges_sys1])

# System index for each atom
batch_idx = torch.cat([
    torch.zeros(len(pos_sys0), dtype=torch.int32),
    torch.ones(len(pos_sys1), dtype=torch.int32),
]).to(positions.device)

energy, forces = dsf_coulomb(
    positions=positions,
    charges=charges,
    cutoff=10.0,
    alpha=0.2,
    batch_idx=batch_idx,
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
    num_systems=2,
)
# energy: (2,) -- per-system energies
# forces: (N, 3) -- per-atom forces
```

#### Undamped Shifted-Force Coulomb (alpha=0)

Setting $\alpha = 0$ reduces DSF to a shifted-force bare Coulomb interaction
(since $\text{erfc}(0) = 1$ and $e^0 = 1$):

```python
energy, forces = dsf_coulomb(
    positions=positions,
    charges=charges,
    cutoff=12.0,
    alpha=0.0,  # Undamped: shifted-force 1/r
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
)
```

### Parameter Guidance

The accuracy of the DSF method is controlled by two parameters:

| Parameter | Typical Range | Guidance |
|-----------|---------------|----------|
| $R_c$ (cutoff) | 10--15 | 12 is a common standard; 15 recommended for higher precision |
| $\alpha$ (damping) | 0.0--0.25 | Controls convergence vs. accuracy trade-off |

**Damping parameter regimes:**

- $\alpha = 0.0$ (undamped): Best for structural properties (RDFs) and absolute
  force magnitudes. Simplest form; no erfc damping overhead.
- $\alpha \approx 0.2\text{--}0.25$: Best for long-time dynamics, collective motions,
  and dielectric properties. Accelerates convergence with cutoff but over-damping
  should be avoided.

```{important}
A practical convergence heuristic is to monitor $\text{erfc}(\alpha R_c)$:

- **Most applications**: $\text{erfc}(\alpha R_c) < 10^{-3}$ is adequate.
  For example, $\alpha = 0.2$ and $R_c = 12$ gives
  $\text{erfc}(2.4) \approx 5 \times 10^{-4}$.
- **High precision**: $\text{erfc}(\alpha R_c) < 10^{-5}$ is recommended.
  For example, $\alpha = 0.2$ and $R_c = 15$ gives
  $\text{erfc}(3.0) \approx 2 \times 10^{-5}$.
```

### When to Use DSF

| Criterion | DSF | Ewald | PME |
|-----------|-----|-------|-----|
| Scaling | $O(N)$ | $O(N^2)$ | $O(N \log N)$ |
| Periodicity required | No | Yes | Yes |
| Force continuity at cutoff | Yes | Depends on cutoff | Depends on cutoff |
| Self-energy correction | Built-in | Separate term | Separate term |
| Best for | Large systems, clusters, non-periodic | Small periodic systems | Large periodic systems |
| Charge gradients (dE/dq) | Analytic, via straight-through | Via autograd | Via autograd |

**Choose DSF when:**

- The system is **non-periodic** (clusters, droplets, interfaces) where Ewald/PME
  would require artificial periodic boundary conditions.
- The system is **extremely large** and the $O(N)$ scaling provides significant
  speedups and memory savings over PME.
- Training **MLIPs with geometry-dependent charges** where analytic $\partial E / \partial q$
  is needed for backpropagation.

**Choose Ewald/PME when:**

- High accuracy of long-range electrostatics is critical for the target property
  (e.g., dielectric constants, free energies of solvation).
- The system is periodic and relatively small ($< 5000$ atoms), where Ewald's
  lower overhead may be advantageous.

### Applicability and Limitations

**Applicability:**

- Large-scale MD simulations with approximate Coulomb
- Non-periodic and partially periodic systems (clusters, droplets, surfaces, interfaces)

**Limitations:**

- **Dielectric properties**: May slightly underestimate the dielectric constant
  in some liquids if the cutoff is too small or damping too high. Typical cutoffs
  of 12--15 provide adequate accuracy for most systems.
- **Molecular torques**: Over-damping ($\alpha > 0.3$) can degrade the accuracy of
  torques in molecular systems. Keep $\alpha \le 0.25$ for molecular simulations.
- **Low-frequency phonons**: In crystal lattices, undamped DSF may deviate slightly
  from Ewald results for very low-frequency modes, though $\alpha \approx 0.2$
  typically resolves this.

### Software Ecosystem

The DSF method is widely implemented and validated across major simulation packages,
including LAMMPS (`pair_style coul/dsf`), OpenMD, DL\_POLY, Cassandra, JAX-MD,
and CP2K. This broad adoption provides extensive cross-validation of the method
and its parameters.

### References

- Fennell, C. J.; Gezelter, J. D. (2006). "Is the Ewald summation still necessary?
  Pairwise alternatives to the accepted standard for long-range electrostatics."
  *J. Chem. Phys.* 124, 234104.
  [DOI: 10.1063/1.2206581](https://doi.org/10.1063/1.2206581)

- Wolf, D.; Keblinski, P.; Phillpot, S. R.; Eggebrecht, J. (1999). "Exact method
  for the simulation of Coulombic systems by spherically truncated, pairwise r-1
  summation." *J. Chem. Phys.* 110, 8254.
  [DOI: 10.1063/1.478738](https://doi.org/10.1063/1.478738)

## Batched Calculations

All electrostatics functions support batched calculations for evaluating multiple
independent systems simultaneously. For most use cases (except for very large systems)
batching is the optimal way to amortize GPU utilization.

The API for electrostatics only needs minor modification to support batches of
systems: users must provide a `batch_idx` tensor to both the initial neighbor
list computation as well as to either the
{func}`~nvalchemiops.torch.interactions.electrostatics.ewald_summation` and
{func}`~nvalchemiops.torch.interactions.electrostatics.particle_mesh_ewald` methods.
While $\alpha$ can be specified independently for each system within a batch, the
mesh dimensions must be the same for all systems (although each system has its own mesh grid).

Example code to perform a batched Ewald calculation:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
import torch
from nvalchemiops.torch.interactions.electrostatics import ewald_summation
from nvalchemiops.torch.neighbors import neighbor_list

# Concatenate atoms from multiple systems
positions = torch.cat([pos_system0, pos_system1, pos_system2])
charges = torch.cat([charges_system0, charges_system1, charges_system2])

# Assign each atom to its system
batch_idx = torch.cat([
    torch.zeros(len(pos_system0), dtype=torch.int32),
    torch.ones(len(pos_system1), dtype=torch.int32),
    torch.full((len(pos_system2),), 2, dtype=torch.int32),
]).to(positions.device)

# Stack cells (B, 3, 3)
cells = torch.stack([cell0, cell1, cell2])
pbc = torch.tensor([[True, True, True]] * 3, device=positions.device)

# Build batched neighbor list
neighbor_list_coo, neighbor_ptr, neighbor_shifts = neighbor_list(
    positions, cutoff=10.0, cell=cells, pbc=pbc,
    batch_idx=batch_idx, method="batch_naive", return_neighbor_list=True
)

# Per-system alpha values (optional)
alphas = torch.tensor([0.3, 0.35, 0.3], dtype=torch.float64, device=positions.device)

# Batched calculation
energies, forces = ewald_summation(
    positions=positions,
    charges=charges,
    cell=cells,
    alpha=alphas,  # Per-system or single value
    k_cutoff=8.0,
    batch_idx=batch_idx,
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
)

# energies: (total_atoms,) - per-atom energies
# Sum per system:
energy_per_system = torch.zeros(3, device=positions.device)
energy_per_system.scatter_add_(0, batch_idx.long(), energies)
```

Batch mode uses one shared set of Miller indices for the reciprocal-space
calculation. If `k_cutoff` is supplied per system, either directly or via
`estimate_ewald_parameters`, `nvalchemiops` uses the maximum cutoff across the
batch to build that shared set.

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.interactions.electrostatics import ewald_summation
from nvalchemiops.jax.neighbors import neighbor_list

# Concatenate atoms from multiple systems
positions = jnp.concatenate([pos_system0, pos_system1, pos_system2])
charges = jnp.concatenate([charges_system0, charges_system1, charges_system2])

# Assign each atom to its system
batch_idx = jnp.concatenate([
    jnp.zeros(len(pos_system0), dtype=jnp.int32),
    jnp.ones(len(pos_system1), dtype=jnp.int32),
    jnp.full((len(pos_system2),), 2, dtype=jnp.int32),
])

# Stack cells (B, 3, 3)
cells = jnp.stack([cell0, cell1, cell2])
pbc = jnp.array([[True, True, True]] * 3)

# Build batched neighbor list
neighbor_list_coo, neighbor_ptr, neighbor_shifts = neighbor_list(
    positions, cutoff=10.0, cell=cells, pbc=pbc,
    batch_idx=batch_idx, method="batch_naive", return_neighbor_list=True
)

# Per-system alpha values (optional)
alphas = jnp.array([0.3, 0.35, 0.3], dtype=jnp.float64)

# Batched calculation
energies, forces = ewald_summation(
    positions=positions,
    charges=charges,
    cell=cells,
    alpha=alphas,  # Per-system or single value
    k_cutoff=8.0,
    batch_idx=batch_idx,
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
)

# energies: (total_atoms,) - per-atom energies
# Sum per system using segment_sum:
energy_per_system = jax.ops.segment_sum(energies, batch_idx, num_segments=3)
```

:::

::::

## Autograd Support

Ewald and PME support automatic differentiation for gradients with respect to
positions, charges, and cell parameters. DSF supports autograd for charge
gradients only; forces and virials are computed analytically by the Warp kernel
(see the DSF Coulomb section above for details). This enables:

- Geometry and lattice parameter optimization
- Integration (and training) with machine learning force fields
- Sensitivity analysis

### Position Gradients (Forces)

The code snippet shows how the electrostatics interface in `nvalchemiops` can
be used with the autograd interface to arrive at the same derivatives
of energy with respect to atomic positions (forces).

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
positions.requires_grad_(True)
energies, explicit_forces = ewald_summation(
    positions, charges, cell, alpha=0.3, k_cutoff=8.0,
    neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
    compute_forces=True,
)

# Autograd forces should match explicit forces
total_energy = energies.sum()
total_energy.backward()
autograd_forces = -positions.grad

assert torch.allclose(autograd_forces, explicit_forces, rtol=1e-5)
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.interactions.electrostatics import ewald_summation

# Define energy function for differentiation
def energy_fn(positions):
    energies = ewald_summation(
        positions, charges, cell, alpha=0.3, k_cutoff=8.0,
        neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
    )
    return jnp.sum(energies)

# Compute explicit forces from the function
_, explicit_forces = ewald_summation(
    positions, charges, cell, alpha=0.3, k_cutoff=8.0,
    neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
    compute_forces=True,
)

# Autograd forces should match explicit forces
autograd_forces = -jax.grad(energy_fn)(positions)

assert jnp.allclose(autograd_forces, explicit_forces, rtol=1e-5)
```

:::

::::

Note, however, that this is only to show that gradient flow works through
the `ewald_summation` call: if only the forces are required, users should just
use the `explicit_forces` directly _without_ autograd for computational
efficiency.

### Charge Gradients

Similar to the positions gradients above, we can compute the gradient of the
energy with respect to atomic charges in the following way:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
charges.requires_grad_(True)
energies = ewald_summation(
    positions, charges, cell, alpha=0.3, k_cutoff=8.0,
    neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
    compute_forces=False,  # disable forces for performance
)

total_energy = energies.sum()
total_energy.backward()
charge_gradients = charges.grad  # dE/dq
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.interactions.electrostatics import ewald_summation

def energy_fn(charges):
    energies = ewald_summation(
        positions, charges, cell, alpha=0.3, k_cutoff=8.0,
        neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
        compute_forces=False,
    )
    return jnp.sum(energies)

charge_gradients = jax.grad(energy_fn)(charges)  # dE/dq
```

:::

::::

For a batch of samples, you may need to use the autograd interface more
explicitly:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
charges.requires_grad_(True)
energies = ewald_summation(...)
energy_per_system = torch.zeros(3, device=positions.device)
# scatter add based on the system index mapping
energy_per_system.scatter_add_(0, batch_idx.long(), energies)
# now compute the derivatives
(charge_gradients,) = torch.autograd.grad(
  outputs=[energy_per_system],
  inputs=[charges],
  grad_outputs=torch.ones_like(energy_per_system),
)
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.interactions.electrostatics import ewald_summation

def batch_energy_fn(charges):
    energies = ewald_summation(
        positions, charges, cell, alpha=0.3, k_cutoff=8.0,
        neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
        batch_idx=batch_idx,
        compute_forces=False,
    )
    # Sum per system using segment_sum
    energy_per_system = jax.ops.segment_sum(energies, batch_idx, num_segments=3)
    return jnp.sum(energy_per_system)

charge_gradients = jax.grad(batch_energy_fn)(charges)
```

:::

::::

### Geometry-Dependent Charges (Hybrid Mode)

```{important}
`hybrid_forces=True` is **deprecated** and emits a `DeprecationWarning`. The
recommended `q(R)` path keeps `charges = charge_model(positions)` in the autograd
graph and derives the full force from energy; see
{ref}`energy-derivative-contract`. The section
below documents the legacy behavior for callers still on the old flag.
```

When charges depend on atomic positions -- as in machine-learned interatomic
potentials (MLIPs) with learned charge models (`q = q(R)`) -- computing total forces requires two contributions:

- **Fixed-charge positional forces** `F = -dE/dR|_q`, computed analytically by
  the Ewald/PME kernel (`compute_forces=True`)
- **Charge chain-rule forces** `-(dE/dq)(dq/dR)`, computed via PyTorch autograd through the charge model

The legacy `hybrid_forces=True` path computes both contributions without adding
the fixed-charge positional term twice. In standard mode, `energy.backward()`
already includes both position and charge terms, so adding explicit forces would
**double-count** the positional contribution. `hybrid_forces=True` detaches
positions and cell from the autograd graph and makes energy differentiable only
through the charges via a straight-through estimator.

```{important}
Do not combine explicit forces (`compute_forces=True`) with full autograd
forces (`-torch.autograd.grad(energy, positions)`) in standard mode -- this
double-counts the positional term `dE/dR|_q`. During migration, use
`hybrid_forces=True` only for legacy direct-output code that still needs explicit
fixed-charge forces plus autograd charge gradients.
```

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
import torch
from nvalchemiops.torch.interactions.electrostatics import particle_mesh_ewald

positions.requires_grad_(True)

# Uniform scaling tensor (identity) for computing the charge virial.
# dE/d(scaling) through the charge path gives the charge contribution
# to the virial, i.e. the energy derivative w.r.t. strain.
scaling = torch.eye(3, dtype=positions.dtype, device=positions.device,
                    requires_grad=True)
positions_scaled = positions @ scaling
cell_scaled = cell @ scaling

# Geometry-dependent charges from scaled positions
q = charge_model(positions_scaled, Z)

# hybrid_forces=True: explicit forces + virial are analytical (forward-only),
# energy is differentiable w.r.t. charges only (via straight-through trick)
energies, direct_forces, direct_virial = particle_mesh_ewald(
    positions_scaled, q, cell_scaled,
    neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
    compute_forces=True, compute_virial=True,
    hybrid_forces=True,
)

# Differentiate energy w.r.t. positions and scaling.
# In hybrid mode only the charge pathway is in the autograd graph.
dE_dpos, dE_dscaling = torch.autograd.grad(
    energies.sum(), [positions, scaling],
)

total_forces = direct_forces - dE_dpos
total_virial = direct_virial.squeeze(0) - dE_dscaling  # W = -dE/dε
```

:::

::::

```{note}
**When not to use hybrid mode:** If the training loss involves forces or
virial directly (e.g., `loss = ||F - F_ref||^2 + ||sigma - sigma_ref||^2`),
use standard mode instead.  In hybrid mode, forces and virial are forward-only
and do not propagate gradients back to model parameters.
```

```{note}
**DSF comparison:** DSF (`dsf_coulomb`) always operates in hybrid mode --
positions are never in the autograd graph, so explicit forces and autograd
charge-chain-rule forces are always complementary without any extra flag.
```

```{note}
**JAX:** Full JAX Ewald/PME calls follow the same first-order energy-derivative
contract for new code. Direct-output and `hybrid_forces` flags are kept as
deprecated compatibility outputs during migration.
```

### Virial / Stress

```{important}
`compute_virial=True` on the full `ewald_summation` / `particle_mesh_ewald` APIs
is **deprecated** and emits a `DeprecationWarning`. For MLIP training, use the
strain-first energy derivative documented in
{ref}`energy-derivative-contract`:
`grad_u = torch.autograd.grad(E.sum(), displacement)[0]`,
`virial = -grad_u`, and `stress = grad_u / V`. That virial equals the direct
output below. The section below documents the legacy direct-virial behavior.
```

Both Ewald and PME provide explicit virial computation via `compute_virial=True`.
Those direct virials are kept for compatibility, MD/inference loops, and
migration checks. For differentiable stress training, derive virials from the
scalar energy with the strain-first recipe instead of training on the
direct-output tensor.

**Convention:**

- Real-space: $W_\text{real} = -\sum_{i<j} \mathbf{r}_{ij} \otimes \mathbf{F}_{ij}$,
  where $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$ and $\mathbf{F}_{ij}$ is the force on atom $i$ due to atom $j$.
- Reciprocal-space: $W_\text{recip}(k) = E(k) \left[\delta_{ab} - \frac{2 k_a k_b}{k^2}\left(1 + \frac{k^2}{4\alpha^2}\right)\right]$
- Stress (tensile-positive Cauchy stress): $\sigma = -W / V$ where $V = |\det(\mathbf{C})|$
- The virial convention is validated against finite-difference strain derivatives
  of the row-vector affine displacement energy
  ($R' = R(I + u)$, $C' = C(I + u)$, $W_{ab} = -\partial E / \partial u_{ab}$)
  in the test suite.

See {ref}`conventions` for the project-wide virial and stress definitions used by all
interaction modules.

**Ewald summation with virial:**

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
energies, forces, virial = ewald_summation(
    positions, charges, cell,
    neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
    compute_forces=True,
    compute_virial=True,
)

# Single system: virial shape (1, 3, 3)
volume = torch.abs(torch.linalg.det(cell))          # scalar
stress = -virial.squeeze(0) / volume                 # (3, 3)

# Batch: virial shape (B, 3, 3)
volume = torch.abs(torch.linalg.det(cell))           # (B,)
stress = -virial / volume[:, None, None]             # (B, 3, 3)
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.interactions.electrostatics import ewald_summation

energies, forces, virial = ewald_summation(
    positions, charges, cell,
    neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
    compute_forces=True,
    compute_virial=True,
)

# Single system: virial shape (1, 3, 3)
volume = jnp.abs(jnp.linalg.det(cell))          # scalar
stress = -virial.squeeze(0) / volume             # (3, 3)

# Batch: virial shape (B, 3, 3)
volume = jnp.abs(jnp.linalg.det(cell))           # (B,)
stress = -virial / volume[:, None, None]         # (B, 3, 3)
```

:::

::::

**PME with virial:**

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
energies, forces, virial = particle_mesh_ewald(
    positions, charges, cell,
    neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
    compute_forces=True,
    compute_virial=True,
)

# Single system
volume = torch.abs(torch.linalg.det(cell))
stress = -virial.squeeze(0) / volume                 # (3, 3)

# Batch
volume = torch.abs(torch.linalg.det(cell))           # (B,)
stress = -virial / volume[:, None, None]             # (B, 3, 3)
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.interactions.electrostatics import particle_mesh_ewald

energies, forces, virial = particle_mesh_ewald(
    positions, charges, cell,
    neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
    compute_forces=True,
    compute_virial=True,
)

# Single system
volume = jnp.abs(jnp.linalg.det(cell))
stress = -virial.squeeze(0) / volume                 # (3, 3)

# Batch
volume = jnp.abs(jnp.linalg.det(cell))           # (B,)
stress = -virial / volume[:, None, None]             # (B, 3, 3)
```

:::

::::

**MLIP training loss example:**

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
strain = torch.zeros(3, 3, dtype=positions.dtype, device=positions.device)
strain.requires_grad_(True)
deformation = torch.eye(3, dtype=positions.dtype, device=positions.device) + strain
positions_s = positions @ deformation
cell_s = cell @ deformation

energies = ewald_summation(
    positions_s, charges, cell_s,
    neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
)
total_energy = energies.sum()
grad_pos, grad_strain = torch.autograd.grad(
    total_energy,
    (positions_s, strain),
    create_graph=True,
)
forces = -grad_pos
virial = -grad_strain

# Compute stress (single system shown; for batch use volume[:, None, None])
volume = torch.abs(torch.linalg.det(cell_s.squeeze(0)))
pred_stress = grad_strain / volume

loss = (
    w_energy * (total_energy - E_target) ** 2
    + w_forces * (forces - F_target).pow(2).sum()
    + w_stress * (pred_stress - stress_target).pow(2).sum()
)
loss.backward()
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.interactions.electrostatics import ewald_summation

def energy_from_strain(positions, charges, cell, strain):
    deformation = jnp.eye(3, dtype=positions.dtype) + strain
    positions_s = positions @ deformation
    cell_s = cell @ deformation
    return jnp.sum(ewald_summation(
        positions_s, charges, cell_s,
        neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
    ))

def loss_fn(positions, charges, cell):
    strain = jnp.zeros((3, 3), dtype=positions.dtype)
    energy, (grad_pos, grad_strain) = jax.value_and_grad(
        lambda pos, eps: energy_from_strain(pos, charges, cell, eps),
        argnums=(0, 1),
    )(
        positions,
        strain,
    )
    forces = -grad_pos
    virial = -grad_strain

    # Compute stress (single system shown; for batch use volume[:, None, None])
    volume = jnp.abs(jnp.linalg.det(cell.squeeze(0)))
    pred_stress = grad_strain / volume

    return (
        w_energy * (energy - E_target) ** 2
        + w_forces * jnp.sum((forces - F_target) ** 2)
        + w_stress * jnp.sum((pred_stress - stress_target) ** 2)
    )

# Compute loss and gradients simultaneously.
loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(positions, charges, cell)
pos_grad, charge_grad, cell_grad = grads
```

For second-order force or charge-gradient losses in JAX, use energy autograd.
JAX PME reciprocal position and charge losses use the native PME mesh HVP path.

:::

::::

:::{note}
Direct virials are compatibility outputs. Stress-loss training should use the
strain-first energy derivative above.
:::

:::{tip}
For quick inference or debugging you can also obtain an approximate stress via
cell gradients followed by reading the gradient divided by volume. In PyTorch use
`cell.requires_grad_(True)` followed by `energy.backward()` and reading
`cell.grad / volume`. In JAX use `jax.grad` with respect to the cell parameter.
This shortcut is **not** recommended for MLIP training; use the strain-first
energy derivative contract above for training stress losses.
:::

(energy-derivative-contract)=

## Energy-Derivative Contract

For differentiable energy evaluation, **energy is the only differentiable output
of the full
{func}`~nvalchemiops.torch.interactions.electrostatics.ewald_summation` and
{func}`~nvalchemiops.torch.interactions.electrostatics.particle_mesh_ewald`
APIs, with matching first-order support on the full JAX Ewald/PME APIs**.
Forces, virial/stress, and charge gradients are derivatives of that energy.
With no direct-output flags set, the call returns the per-atom energy tensor only.

Only `positions`, `charges`, and `cell` are differentiable inputs in this
contract. Setup values such as `alpha`, cutoffs, accuracy, mesh spacing or
dimensions, spline order, PBC/slab flags, batch metadata, neighbor topology,
Miller/grid indices, and PME B-spline moduli are constants. Gradients are not
reported for those setup values.

Precomputed numerical metadata is treated as setup state, not as a
differentiable parameter. `k_vectors`, `k_squared`, `volume`, `cell_inv_t`,
reciprocal-cell metadata, and slab-geometry caches remain accepted when
differentiating with respect to `cell`, but they are static metadata assumed
to correspond to the current `cell`. Precomputed structure
factors, charge meshes, and total-charge caches must be omitted from any public
API that would use them while differentiating with respect to `positions` or
`charges`.

Neighbor-list differentiation is fixed-topology differentiation. The gradient
includes pair displacements and periodic image terms such as `shift @ cell`, but
does not differentiate the discrete event of a pair entering or leaving the
neighbor list.

Torch supports the second-order force/stress-loss paths used in training. JAX
higher-order support is limited to tested position and charge scalar losses.
JAX PME stress/cell/strain, alpha, and precomputed-metadata higher-order
derivatives are unsupported until implemented and tested, including high-level
`particle_mesh_ewald(..., slab_correction=True)` calls. Energy-returning Ewald,
PME, and slab paths support non-uniform per-atom losses such as `loss = (weights
* energies).sum()` for positions, charges, and supported cell derivatives.
Precomputed static caches still do not recover the derivative of how those
caches were generated; omit the cache when that derivative is part of the
intended loss.
Second-order support means differentiating scalar losses through these energy
paths with Torch/JAX autograd. Electrostatics does not expose public Hessian or
Jacobian tensors/functions.

### Fixed-Cell Metadata Recipes

For fixed-cell Ewald/PME loops, precompute reciprocal metadata once from a cell
that is detached from autograd, then reuse it while that cell is unchanged.

```python
with torch.no_grad():
    k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

for positions in trajectory:
    energy = ewald_summation(
        positions, charges, cell,
        k_vectors=k_vectors,
        neighbor_list=nl,
        neighbor_ptr=nl_ptr,
        neighbor_shifts=shifts,
    )
```

```python
cell_static = jax.lax.stop_gradient(cell)
cell_inv_t = jnp.linalg.inv(cell_static).transpose(0, 2, 1)
volume = jnp.abs(jnp.linalg.det(cell_static))
reciprocal_cell = 2.0 * jnp.pi * jnp.linalg.inv(cell_static)
k_vectors, k_squared = generate_k_vectors_pme(
    cell_static, mesh_dimensions, reciprocal_cell=reciprocal_cell
)
mesh_nx, mesh_ny, mesh_nz = mesh_dimensions
miller_x = jnp.fft.fftfreq(mesh_nx, d=1.0 / mesh_nx)
miller_y = jnp.fft.fftfreq(mesh_ny, d=1.0 / mesh_ny)
miller_z = jnp.fft.rfftfreq(mesh_nz, d=1.0 / mesh_nz)
moduli_x = compute_bspline_moduli_1d(miller_x, mesh_nx, spline_order)
moduli_y = compute_bspline_moduli_1d(miller_y, mesh_ny, spline_order)
moduli_z = compute_bspline_moduli_1d(miller_z, mesh_nz, spline_order)

for positions in trajectory:
    energy = particle_mesh_ewald(
        positions, charges, cell,
        k_vectors=k_vectors,
        k_squared=k_squared,
        cell_inv_t=cell_inv_t,
        volume=volume,
        moduli_x=moduli_x,
        moduli_y=moduli_y,
        moduli_z=moduli_z,
        mesh_dimensions=mesh_dimensions,
        spline_order=spline_order,
        neighbor_list=nl,
        neighbor_ptr=nl_ptr,
        neighbor_shifts=shifts,
    )
```

If the cell changes and cell gradients are part of the loss, regenerate
cell-derived metadata for that cell or omit the cache so the wrapper rebuilds it
internally. The cached tensors are setup metadata; they do not carry derivatives
of the metadata-generation step.

```{important}
The direct-output flags `compute_forces`, `compute_virial`,
`compute_charge_gradients`, and `hybrid_forces` on the full
`ewald_summation` / `particle_mesh_ewald` APIs are **deprecated** and emit a
`DeprecationWarning`. They remain functional for compatibility in v0.4.0, but
differentiable training code should use the energy-derivative recipes below. The
lower-level component functions (`ewald_real_space`, `ewald_reciprocal_space`,
`pme_reciprocal_space`) keep their direct-force outputs as no-autograd
MD/inference paths and do not warn. Full-API calls that still pass
`compute_forces=True` etc. emit a `DeprecationWarning`.
```

The examples below use `particle_mesh_ewald`; `ewald_summation` follows the same
contract. A complete runnable script is in
{doc}`/examples/electrostatics/07_pme_energy_derivative_training`.

### Force Evaluation

```python
positions = positions.detach().requires_grad_(True)
energy = particle_mesh_ewald(
    positions, charges, cell,
    neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
)  # returns the per-atom energy tensor only

forces = -torch.autograd.grad(energy.sum(), positions)[0]  # (N, 3)
```

`energy.sum()` is the scalar total energy whose derivative is the full force for
the graph that produced `energy`.

### Force-Loss Training

For Torch training on a force loss, build the force with `create_graph=True` so
the later `loss.backward()` can differentiate through it (double-backward):

```python
positions = positions.detach().requires_grad_(True)
energy = particle_mesh_ewald(positions, charges, cell, ...)

forces = -torch.autograd.grad(energy.sum(), positions, create_graph=True)[0]
loss = force_loss(forces, target_forces)
loss.backward()
```

### Geometry-Dependent Charges (`q(R)`)

When charges are predicted from positions by a learned model, keep
`charges = charge_model(positions)` **in the autograd graph**. The full force then
includes both the fixed-charge term and the charge-model chain-rule term
$\frac{\partial E}{\partial q}\frac{\partial q}{\partial R}$:

```python
positions = positions.detach().requires_grad_(True)
charges = charge_model(positions)              # stays connected to positions
energy = particle_mesh_ewald(positions, charges, cell, ...)

forces = -torch.autograd.grad(energy.sum(), positions, create_graph=True)[0]
```

```{important}
This replaces the deprecated `hybrid_forces=True` path. A legacy direct force
(`compute_forces=True`) is the fixed-charge partial $-\partial E/\partial R|_q$
and does not include the $\partial E/\partial q \cdot \partial q/\partial R$
term for `q(R)` models -- which is why direct force output on the full API is
deprecated.
```

### Virial and Stress (Strain-First)

`strain` is not a PME/Ewald argument. Build a differentiable row-vector
displacement tensor, deform positions and cell by `I + strain`, and let autograd
map gradients from the deformed inputs back to strain. The virial is
$W = -\partial E/\partial u$ and tensile-positive stress is
$\partial E/\partial u / V$:

```python
positions = positions.detach().requires_grad_(True)
num_systems = cell.shape[0]
strain = torch.zeros(
    num_systems, 3, 3, device=positions.device, dtype=positions.dtype,
    requires_grad=True,
)
eye = torch.eye(3, device=positions.device, dtype=positions.dtype).unsqueeze(0)
deform = eye + strain

# batch_idx maps each atom to its system (all zeros for a single system)
positions_s = torch.einsum("ni,nij->nj", positions, deform[batch_idx])
cell_s = torch.einsum("bij,bjk->bik", cell, deform)

energy = particle_mesh_ewald(positions_s, charges, cell_s, ...)
grad_strain = torch.autograd.grad(
    energy.sum(), strain,
    create_graph=True,   # keep for stress-loss training; omit for evaluation
)[0]                                                       # (num_systems, 3, 3)
virial = -grad_strain

volume = torch.abs(torch.linalg.det(cell_s))               # (num_systems,)
stress = grad_strain / volume[:, None, None]               # (num_systems, 3, 3)
```

This `virial` matches the (deprecated) `compute_virial=True` direct output -- both
are $-\partial E/\partial u$. The stress uses the project-wide tensile-positive
Cauchy convention $\sigma = -W/V = \partial E/\partial u / V$; see
{ref}`conventions`. For stress-loss training, build the loss from `stress` and
call `loss.backward()`.

### Combined Force + Stress Loss (Performance)

When a single training loss mixes **both** forces and stress, take them from **one**
`torch.autograd.grad` call over `(positions, strain)` together -- not two separate
calls:

The position argument in that combined call chooses the force coordinate frame.
Use `positions_s` for deformed-coordinate force targets, or use the undeformed
reference `positions` if the target forces are defined in the reference frame.
The runnable derivative-training example uses the reference-frame variant.

```python
# Preferred: one combined grad call -> one double-backward.
# This variant returns deformed-coordinate forces.
grad_pos, grad_strain = torch.autograd.grad(
    energy.sum(), (positions_s, strain), create_graph=True,
)
forces = -grad_pos
virial = -grad_strain
stress = grad_strain / volume[:, None, None]
```

Each `create_graph=True` `grad` call builds its own first-derivative graph node, and
`loss.backward()` runs the reciprocal second-derivative (an $O(K\cdot N)$ kernel)
**once per node**. Computing forces and virial in two separate `grad` calls
therefore doubles the reciprocal double-backward work; combining them in one
call avoids duplicate reciprocal double-backward work. The gradients are identical
either way -- this is purely a performance choice.

### `torch.compile` Compatibility

Direct-output Ewald/PME calls without framework autograd can be wrapped in
`torch.compile(fullgraph=True)` when all shape-determining metadata is static
and precomputed outside the compiled function. This is useful for no-autograd
MD/inference loops and for benchmarking the deprecated direct-output migration
path.

Energy-autograd training callables that contain `torch.autograd.grad` are not
treated as a `torch.compile` fast path in this release: Dynamo does not trace the
complete force/stress loss callable as one full graph. Compile only the
energy-forward function when that is useful for an application, and keep the
force, stress, and
double-backward training step in eager PyTorch. Benchmark CSV rows label this
difference explicitly with `derivative_contract` and `workload`.

### Charge Gradients

$\partial E/\partial q$ is an ordinary gradient of the energy w.r.t. charges:

```python
charges = charges.detach().requires_grad_(True)
energy = particle_mesh_ewald(positions, charges, cell, ...)

charge_grad = torch.autograd.grad(
    energy.sum(), charges,
    create_graph=True,   # keep for charge-gradient-loss training
)[0]                                                                   # (N,)
```

(electrostatics-migration)=

### Migration From Deprecated Flags

Each deprecated direct-output flag maps to an energy-autograd replacement. The
deprecated flags remain available for compatibility in v0.4.0 but emit a
`DeprecationWarning`.

| Deprecated flag | Replacement |
|-----------------|-------------|
| `compute_forces=True` | `forces = -torch.autograd.grad(E.sum(), positions)[0]` |
| `compute_virial=True` | `grad_u = torch.autograd.grad(E.sum(), displacement)[0]` with the row-vector displacement recipe; `virial = -grad_u`, `stress = grad_u / V` |
| `compute_charge_gradients=True` | `dEdq = torch.autograd.grad(E.sum(), charges)[0]` |
| `hybrid_forces=True` | Keep `charges = charge_model(positions)` in the graph; derive the force from energy (full `q(R)` force) |

```{note}
These deprecations apply to the **full** APIs only. `ewald_real_space`,
`ewald_reciprocal_space`, and `pme_reciprocal_space` retain their direct-force
outputs for no-autograd MD/inference loops and do not warn. They are not part of
the differentiable training contract.
```

```{note}
JAX full Ewald/PME follows the same first-order energy-derivative contract
for positions, charges, and row-vector displacement virials. Higher-order JAX
support is limited to tested position and charge scalar losses; PME reciprocal
terms use the native PME mesh HVP path. JAX PME stress/cell/strain, alpha, and
precomputed-metadata higher-order paths are
unsupported until implemented and tested. JAX direct-output flags remain
functional for compatibility in v0.4.0 but are deprecated for differentiable
training.
```

(parameter-estimation)=

## Parameter Estimation

ALCHEMI Toolkit-Ops provides functions to estimate sensible parameters based on
desired accuracy threshold with two functions that share some functionality,
but target the Ewald and PME algorithms respectively.

### Ewald Parameters

The function {func}`~nvalchemiops.torch.interactions.electrostatics.estimate_ewald_parameters`
(PyTorch) / {func}`~nvalchemiops.jax.interactions.electrostatics.estimate_ewald_parameters`
(JAX) is used to estimate $\alpha$ and cutoffs for real- and reciprocal-space specifically
for the **Ewald** algorithm:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
from nvalchemiops.torch.interactions.electrostatics import estimate_ewald_parameters

params = estimate_ewald_parameters(
    positions=positions,
    cell=cell,
    batch_idx=None,  # or provide for batched systems
    accuracy=1e-6,
)

print(f"alpha = {params.alpha.item():.4f}")
print(f"r_cutoff = {params.real_space_cutoff.item():.4f}")
print(f"k_cutoff = {params.reciprocal_space_cutoff.item():.4f}")
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.interactions.electrostatics import estimate_ewald_parameters

params = estimate_ewald_parameters(
    positions=positions,
    cell=cell,
    batch_idx=None,  # or provide for batched systems
    accuracy=1e-6,
)

print(f"alpha = {params.alpha:.4f}")
print(f"r_cutoff = {params.real_space_cutoff:.4f}")
print(f"k_cutoff = {params.reciprocal_space_cutoff:.4f}")
```

:::

::::

This method returns an `EwaldParameters` dataclass, which
is a light data structure that holds parameters used for the Ewald algorithm.

### PME Parameters

The function {func}`~nvalchemiops.torch.interactions.electrostatics.estimate_pme_parameters`
(PyTorch) / {func}`~nvalchemiops.jax.interactions.electrostatics.estimate_pme_parameters`
(JAX) is used to estimate $\alpha$, the real-space cutoff, and mesh specifications specifically
for the PME algorithm; the value of $\alpha$ is determined the same way as for Ewald.

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
from nvalchemiops.torch.interactions.electrostatics import estimate_pme_parameters

params = estimate_pme_parameters(
    positions=positions,
    cell=cell,
    batch_idx=None,
    accuracy=1e-6,
)

print(f"alpha = {params.alpha.item():.4f}")
print(f"Mesh: {params.mesh_dimensions}")
print(f"r_cutoff = {params.real_space_cutoff.item():.4f}")
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.interactions.electrostatics import estimate_pme_parameters

params = estimate_pme_parameters(
    positions=positions,
    cell=cell,
    batch_idx=None,
    accuracy=1e-6,
)

print(f"alpha = {params.alpha:.4f}")
print(f"Mesh: {params.mesh_dimensions}")
print(f"r_cutoff = {params.real_space_cutoff:.4f}")
```

:::

::::

This method returns a `PMEParameters` dataclass, which
is a light data structure that holds parameters used for the particle-mesh Ewald algorithm.
For batched inputs, `estimate_pme_parameters` intentionally returns one shared
real-space cutoff and one shared $\alpha$ for the whole batch. The shared values
are computed from the median atom count and median cell volume, while
`mesh_spacing` remains per-system because it depends on each cell length. Pass
`real_space_cutoff=` when a simulation needs to pin the neighbor-list cutoff
instead of using this median-system heuristic.

## Units

The electrostatics functions are unit-agnostic; they work in whatever consistent
unit system you provide. Common conventions:

| Unit System | Positions | Energy | Charge |
|-------------|-----------|--------|--------|
| Atomic units | Bohr | Hartree | e |
| eV-Angstrom | Angstrom | eV | e |
| LAMMPS "real" | Angstrom | kcal/mol | e |

```{important}
Ensure consistency between your position units, cell units, and cutoff values.
The `alpha` parameter has units of inverse length.
```

For atomic units (Bohr/Hartree), no additional constants are needed. For other
unit systems, you may need to multiply energies by a Coulomb constant:

```python
# eV-Angstrom: k_e ~ 14.3996 eV*Angstrom
# The functions assume k_e = 1 (atomic units)
```

## Theory Background

### The Ewald Splitting

The Coulomb potential $1/r$ is split into short-range and long-range components
using a Gaussian screening function:

```{math}
\frac{1}{r} = \frac{\text{erfc}(\alpha r)}{r} + \frac{\text{erf}(\alpha r)}{r}
```

- The $\text{erfc}$ term decays exponentially and is computed in real space
- The $\text{erf}$ term is smooth and computed efficiently in reciprocal space

The splitting parameter $\alpha$ controls the balance:

- Large $\alpha$: More work in reciprocal space, fewer k-vectors needed
- Small $\alpha$: More work in real space, larger neighbor cutoff needed

### Charge Neutrality

For periodic systems, overall charge neutrality is required for the electrostatic
energy to be well-defined. Non-neutral systems include a background correction:

```{math}
E_{\text{background}} = \frac{\pi}{2\alpha^2 V} Q_{\text{total}}^2
```

This term represents the interaction of the charged system with a uniform
neutralizing background.

### B-Spline Interpolation (PME)

PME uses cardinal B-splines of order $p$ for charge assignment:

- Order 1: Nearest-grid-point (NGP)
- Order 2: Cloud-in-cell (CIC)
- Order 3: Triangular-shaped cloud (TSC)
- Order 4: Cubic B-spline (recommended)
- Order 5: Quartic B-spline
- Order 6: Quintic B-spline

Higher spline orders provide better accuracy but spread charges over more grid
points. Orders 1-6 are supported; order 4 (cubic) is the standard choice,
balancing accuracy and efficiency.

## Troubleshooting

### Common Issues

**Energy not converging with k_cutoff**:
The reciprocal-space energy should converge as `k_cutoff` increases. If it doesn't,
check that your cell is properly defined (lattice vectors as rows) and that the
volume is computed correctly.

**Force discontinuities**:
Ensure the real-space cutoff is compatible with your neighbor list cutoff. The
neighbor list should include all pairs within the damping range of $\text{erfc}(\alpha r)$.

**NaN or Inf values**:

- Check for overlapping atoms (r -> 0)
- Verify cell volume is positive
- Ensure charges are finite

**Memory issues with large meshes**:
PME mesh memory scales as $n_x \times n_y \times n_z$. For very large cells, consider using
coarser mesh spacing. It may also be worth comparing compute requirements between Ewald
and PME algorithms.

### Validation

```{note}
The validation example below uses `torchpme`, which is a PyTorch-specific package.
JAX users can validate against reference implementations in their ecosystem or
compare against the PyTorch results for equivalent inputs.
```

You can validate PME results against reference implementations like `torchpme`. Here's a simple example
comparing reciprocal-space energies:

```python
import torch
import math
from nvalchemiops.torch.interactions.electrostatics import pme_reciprocal_space

# Create a simple dipole system
device = torch.device("cuda")
dtype = torch.float64
cell_size = 10.0
separation = 2.0

# Two charges separated along x-axis
center = cell_size / 2
positions = torch.tensor(
    [
        [center - separation / 2, center, center],
        [center + separation / 2, center, center],
    ],
    dtype=dtype,
    device=device,
)
charges = torch.tensor([1.0, -1.0], dtype=dtype, device=device)
cell = torch.eye(3, dtype=dtype, device=device) * cell_size

# PME parameters
alpha = 0.3
mesh_spacing = 0.5
mesh_dims = (20, 20, 20)

# Compute reciprocal-space energy
energy = pme_reciprocal_space(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=alpha,
    mesh_dimensions=mesh_dims,
    spline_order=4,
    compute_forces=False,
)

print(f"Reciprocal-space energy: {energy.sum().item():.6f}")

# Optional: Compare with torchpme if available
try:
    from torchpme import PMECalculator
    from torchpme.potentials import CoulombPotential

    # torchpme uses sigma where Gaussian is exp(-r**2/(2 * sigma**2))
    # Standard Ewald uses exp(-alpha**2 * r**2), so sigma = 1/(2**0.5 * alpha)
    smearing = 1.0 / (math.sqrt(2.0) * alpha)
    potential = CoulombPotential(smearing=smearing).to(device=device, dtype=dtype)

    calculator = PMECalculator(
        potential=potential,
        mesh_spacing=mesh_spacing,
        interpolation_nodes=4,
        full_neighbor_list=True,
        prefactor=1.0,
    ).to(device=device, dtype=dtype)

    charges_pme = charges.unsqueeze(1)
    reciprocal_potential = calculator._compute_kspace(charges_pme, cell, positions)
    torchpme_energy = (reciprocal_potential * charges_pme).sum()

    print(f"TorchPME energy: {torchpme_energy.item():.6f}")
    print(f"Relative difference: {abs(energy.sum() - torchpme_energy) / abs(torchpme_energy):.2e}")

except ImportError:
    print("torchpme not available for comparison")
```

For more comprehensive validation examples, including:

- Crystal structure systems (CsCl, wurtzite, zincblende)
- Gradient validation against numerical finite differences
- Batch processing consistency checks
- Conservation law tests (momentum, translation invariance)

See the unit tests at `test/interactions/electrostatics/` in the repository.

## Further Reading

- Ewald, P. P. (1921). "Die Berechnung optischer und elektrostatischer Gitterpotentiale."
  *Ann. Phys.* 369, 253-287.
  [DOI: 10.1002/andp.19213690304](https://doi.org/10.1002/andp.19213690304)

- Darden, T.; York, D.; Pedersen, L. (1993). "Particle mesh Ewald: An N*log(N)
  method for Ewald sums in large systems." J. Chem. Phys. 98, 10089.
  [DOI: 10.1063/1.464397](https://doi.org/10.1063/1.464397)

- Essmann, U.; Perera, L.; Berkowitz, M. L.; Darden, T.; Lee, H.; Pedersen, L. G.
  (1995). "A smooth particle mesh Ewald method." *J. Chem. Phys.* 103, 8577.
  [DOI: 10.1063/1.470117](https://doi.org/10.1063/1.470117)

- Yeh, I.-C.; Berkowitz, M. L. (1999). "Ewald summation for systems with slab
  geometry." *J. Chem. Phys.* 111, 3155-3162.
  [DOI: 10.1063/1.479595](https://doi.org/10.1063/1.479595)

- Ballenegger, V.; Arnold, A.; Cerdà, J. J. (2009). "Simulations of non-neutral
  slab systems with long-range electrostatic interactions in two-dimensional
  periodic boundary conditions." *J. Chem. Phys.* 131, 094107.
  [DOI: 10.1063/1.3216473](https://doi.org/10.1063/1.3216473)

- Kolafa, J.; Perram, J. W. (1992). "Cutoff Errors in the Ewald Summation Formulae
  for Point Charge Systems." *Mol. Sim.* 9, 351-368.
  [DOI: 10.1080/08927029208049126](https://doi.org/10.1080/08927029208049126)

- Sagui, C.; Darden, T. A. (1999). "Molecular Dynamics Simulations of Biomolecules:
  Long-Range Electrostatic Effects." *Annu. Rev. Biophys. Biomol. Struct.* 28, 155-179.
  [DOI: 10.1146/annurev.biophys.28.1.155](https://doi.org/10.1146/annurev.biophys.28.1.155)

- Fennell, C. J.; Gezelter, J. D. (2006). "Is the Ewald summation still necessary?
  Pairwise alternatives to the accepted standard for long-range electrostatics."
  *J. Chem. Phys.* 124, 234104.
  [DOI: 10.1063/1.2206581](https://doi.org/10.1063/1.2206581)

- Wolf, D.; Keblinski, P.; Phillpot, S. R.; Eggebrecht, J. (1999). "Exact method
  for the simulation of Coulombic systems by spherically truncated, pairwise r-1
  summation." *J. Chem. Phys.* 110, 8254.
  [DOI: 10.1063/1.478738](https://doi.org/10.1063/1.478738)

---

For detailed API documentation, see the [PyTorch API](../../modules/torch/electrostatics), [JAX API](../../modules/jax/electrostatics), and [Warp API](../../modules/warp/electrostatics) references.

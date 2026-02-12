<!-- markdownlint-disable MD013 MD049 -->

(electrostatics_userguide)=

# Electrostatic Interactions

Electrostatic interactions arise from Coulombic forces between charged particles.
In periodic systems, the $1/r$ potential decays slowly, requiring special techniques
to handle the conditionally convergent lattice sum. ALCHEMI Toolkit-Ops provides
GPU-accelerated implementations of Ewald summation and Particle Mesh Ewald (PME)
via [NVIDIA Warp](https://nvidia.github.io/warp/), with full PyTorch autograd support
for machine learning applications.

```{tip}
For most applications, start with {func}`~nvalchemiops.torch.interactions.electrostatics.ewald_summation`
or {func}`~nvalchemiops.torch.interactions.electrostatics.particle_mesh_ewald`. These unified APIs
automatically handle parameter estimation and dispatch to optimized kernels based on your input.
```

## Overview of Available Methods

ALCHEMI Toolkit-Ops provides electrostatics modules for point charges:

| Method | Scaling | Best For |
|--------|---------|----------|
| **Ewald Summation** | $O(N^2)$ | Small/medium systems (<5000 atoms) |
| **Particle Mesh Ewald** | $O(N \log N)$ | Large systems |
| **Direct Coulomb** | $O(N \times \text{pairs})$ | Non-periodic or as real-space component |
| **Ewald Multipole** | $O(N^2)$ | Multipolar systems, small/medium |
| **PME Multipole** | $O(N \log N)$ | Multipolar systems, large |

All methods support:

- Single-system and batched calculations
- Periodic boundary conditions
- Automatic differentiation (positions, charges, cell)
- Both neighbor list (COO) and neighbor matrix formats

## Quick Start

::::{tab-set}

:::{tab-item} Ewald Summation
:sync: ewald

```python
from nvalchemiops.torch.interactions.electrostatics import ewald_summation
from nvalchemiops.torch.neighbors import neighbor_list

# Build neighbor list
neighbor_list_coo, neighbor_ptr, neighbor_shifts = neighbor_list(
    positions, cutoff=10.0, cell=cell, pbc=pbc, return_neighbor_list=True
)

# Compute electrostatics (parameters estimated automatically)
energies, forces = ewald_summation(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    accuracy=5e-4,  # Target accuracy for parameter estimation
    compute_forces=True,
)
```

:::

:::{tab-item} Particle Mesh Ewald
:sync: pme

```python
from nvalchemiops.torch.interactions.electrostatics import particle_mesh_ewald
from nvalchemiops.torch.neighbors import neighbor_list

# Build neighbor list
neighbor_list_coo, neighbor_ptr, neighbor_shifts = neighbor_list(
    positions, cutoff=10.0, cell=cell, pbc=pbc, return_neighbor_list=True
)

# Compute electrostatics (parameters estimated automatically)
energies, forces = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    accuracy=5e-4,
    compute_forces=True,
)
```

:::

:::{tab-item} Direct Coulomb
:sync: coulomb

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

::::

## Data Formats

### Tensor Specifications

The table below lists out the general syntax and expected shapes for tensors used
in the electrostatics code. When possible to do so, we encourage developers and
users to align their variable naming to what is shown here for ease of debugging
and consistency.

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `positions` | `(N, 3)` | `float64` | Atomic coordinates |
| `charges` | `(N,)` | `float64` | Atomic partial charges |
| `cell` | `(1, 3, 3)` or `(B, 3, 3)` | `float64` | Unit cell lattice vectors (rows) |
| `pbc` | `(1, 3)` or `(B, 3)` | `bool` | Periodic boundary conditions per axis |
| `batch_idx` | `(N,)` | `int32` | System index for each atom (batched only) |
| `alpha` | `float` or `(B,)` tensor | `float64` | Ewald splitting parameter |

### Output Data Types

Energies are always computed and returned in `float64` for numerical stability
during accumulation. Forces, virial, and charge gradients match the input
precision -- `float32` when positions are `float32`, `float64` when positions
are `float64`.

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

#### Explicit Parameters

```python
from nvalchemiops.torch.interactions.electrostatics import ewald_summation

energies, forces = ewald_summation(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=0.3,        # Ewald splitting parameter
    k_cutoff=8.0,     # Reciprocal-space cutoff in inverse length
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
)
```

#### Automatic Parameter Estimation

When `alpha` or `k_cutoff` are not provided, they are estimated based on `accuracy`:

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

#### Separating real- and reciprocal-space

When either components are required individually, the following code can
be used instead of the high level wrapper to compute the contributions
directly:

```python
from nvalchemiops.torch.interactions.electrostatics import ewald_real_space, ewald_reciprocal_space

# Real-space only (short-range, damped Coulomb)
real_energies, real_forces = ewald_real_space(
    positions, charges, cell, alpha=0.3,
    neighbor_list=neighbor_list, neighbor_shifts=neighbor_shifts,
)

# Reciprocal-space only (long-range, smooth)
recip_energies, recip_forces = ewald_reciprocal_space(
    positions, charges, cell, alpha=0.3, k_cutoff=8.0,
)
```

```{note}
The sum of real and reciprocal components gives the Ewald energy.
The self-energy and background corrections are embedded within
the reciprocal energy.
```

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

The B-spline interpolation introduces errors corrected by the influence function:

```{math}
G(\mathbf{k}) = \frac{2\pi}{V} \cdot \frac{\exp(-k^2 / 4\alpha^2)}{k^2} \cdot \frac{1}{C^{2p}(\mathbf{k})}
```

where $C(\mathbf{k})$ is the B-spline correction factor and $p$ is the spline order.

### Usage Examples

#### Basic Usage

```python
from nvalchemiops.torch.interactions.electrostatics import particle_mesh_ewald

energies, forces = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=0.3,
    mesh_dimensions=(32, 32, 32),  # FFT mesh size
    spline_order=4,                 # B-spline order (4 = cubic)
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
)
```

#### Mesh Spacing

Instead of explicit mesh dimensions, specify mesh spacing:

```python
energies, forces = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=0.3,
    mesh_spacing=0.5,  # Angstrom (or your length unit)
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
)
```

#### Automatic Parameter Estimation

Similar to the Ewald summation interface, PME accepts an `accuracy` parameter
that can be used to automatically determine sensible $\alpha$ and mesh:

```python
energies, forces = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    accuracy=4e-5,  # Estimates alpha and mesh dimensions
    compute_forces=True,
)
```

```{note}
We encourage users to properly benchmark performance gains
afforded by `accuracy` on their systems of interest. The lower
the value of `accuracy` the more precise, at the cost of higher
computational requirements.
```

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

## Multipole Electrostatics

ALCHEMI Toolkit-Ops supports multipolar charge distributions beyond point charges.
Multipole electrostatics extends standard Ewald and PME to include:

| Order L | Name | Components | Physical Meaning |
|---------|------|------------|------------------|
| 0 | Monopole | 1 | Net charge |
| 1 | Dipole | 3 | Charge separation |
| 2 | Quadrupole | 5 | Charge distribution shape |

For maximum angular momentum $L_\text{max}=2$, each
atom has **9 multipole coefficients**.

### Multipole Coefficient Layout

The multipole coefficients follow spherical harmonic ordering:

```python
# Shape: (N, 9) where N is number of atoms
# Channel layout:
#   [0]: q^{0,0}   - monopole (charge)
#   [1]: q^{1,-1}  - dipole y-component
#   [2]: q^{1,0}   - dipole z-component
#   [3]: q^{1,+1}  - dipole x-component
#   [4]: q^{2,-2}  - quadrupole xy
#   [5]: q^{2,-1}  - quadrupole yz
#   [6]: q^{2,0}   - quadrupole 3z²-r²
#   [7]: q^{2,+1}  - quadrupole xz
#   [8]: q^{2,+2}  - quadrupole x²-y²

import torch
multipoles = torch.zeros((num_atoms, 9), dtype=torch.float64)
multipoles[:, 0] = charges          # Set monopoles (same as point charges)
multipoles[:, 1:4] = dipoles        # Set dipole moments
multipoles[:, 4:9] = quadrupoles    # Set quadrupole moments
```

### Ewald Multipole

Explicit k-vector Ewald summation for multipoles:

```python
from nvalchemiops.torch.interactions.electrostatics import ewald_multipole_summation

energies = ewald_multipole_summation(
    positions=positions,        # (N, 3)
    multipoles=multipoles,      # (N, 9)
    cell=cell,
    neighbor_list=neighbor_list,
    neighbor_shifts=neighbor_shifts,
    accuracy=1e-6,
)
```

For the reciprocal-space only (no real-space, useful for ML applications):

```python
from nvalchemiops.torch.interactions.electrostatics import ewald_multipole_reciprocal_space
from nvalchemiops.torch.interactions.electrostatics.k_vectors import generate_k_vectors_ewald_summation

k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)
alpha = torch.tensor([0.5], dtype=torch.float64, device=cell.device)

energies = ewald_multipole_reciprocal_space(
    positions, multipoles, cell, k_vectors, alpha,
)

# With response field (gradient w.r.t. multipoles)
energies, response = ewald_multipole_reciprocal_space(
    positions, multipoles, cell, k_vectors, alpha,
    compute_response=True,
)
```

### PME Multipole

FFT-based multipole electrostatics with $O(N \log N)$ scaling:

```python
from nvalchemiops.torch.interactions.electrostatics import pme_multipole_summation

energies = pme_multipole_summation(
    positions=positions,
    multipoles=multipoles,
    cell=cell,
    neighbor_list=neighbor_list,
    neighbor_shifts=neighbor_shifts,
    accuracy=1e-6,  # Estimates alpha and mesh
)

# Or with explicit parameters:
energies = pme_multipole_summation(
    positions=positions,
    multipoles=multipoles,
    cell=cell,
    neighbor_list=neighbor_list,
    neighbor_shifts=neighbor_shifts,
    alpha=0.3,
    mesh_dimensions=(32, 32, 32),
    spline_order=4,
)
```

### Mathematical Background

The multipole charge density is represented using Gaussian Type Orbitals (GTOs):

```{math}
\rho_i(\mathbf{r}) = \sum_{l,m} q_i^{lm} \cdot \phi_{lm}(\mathbf{r} - \mathbf{r}_i, \sigma_i)
```

where the GTO width $\sigma$ is related to the Ewald parameter $\alpha$ by:

```{math}
\sigma = \frac{1}{2\alpha}
```

**Real-space** uses damped T-tensors (interaction tensors derived from $\text{erfc}(\alpha r)/r)$:

- Monopole-Monopole: $q_i T^0 q_j$
- Monopole-Dipole: $q_i T^1 \cdot \mu_j$
- Dipole-Dipole: $\mu_i \cdot T^2 \cdot \mu_j$
- And higher-order terms...

**Reciprocal-space** uses Fourier-transformed GTOs:

```{math}
\tilde{\phi}_{lm}(\mathbf{k}, \sigma) = (-i)^l Y_{lm}(\hat{\mathbf{k}}) \cdot e^{-k^2 \sigma^2 / 2}
```

**Self-energy correction**:

```{math}
E_{\text{self}} = \sum_{lm} C_l (q_i^{lm})^2 \alpha^{2l+1}
```

where $C_l$ are l-dependent constants.

### Use Cases

Multipole electrostatics are useful for:

- **Polarizable force fields**: Dipole moments from induced polarization
- **Machine learning potentials**: Higher-order features beyond point charges
- **Coarse-grained models**: Representing charge distributions of molecular groups
- **Quantum chemistry interfaces**: Using multipole moments from quantum calculations

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

## Autograd Support

All electrostatics functions support automatic differentiation for gradients
with respect to positions, charges, and cell parameters. This enables:

- Geometry and lattice parameter optimization
- Integration (and training) with machine learning force fields
- Sensitivity analysis

### Position Gradients (Forces)

The code snippet shows how the electrostatics interface in `nvalchemiops` can
be used with the PyTorch `autograd` interface to arrive at the same derivatives
of energy with respect to atomic positions (forces).

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

Note, however, that this is only to show that gradient flow works through
the `ewald_summation` call: if only the forces are required, users should just
use the `explicit_forces` directly _without_ `autograd` for computational
efficiency.

### Charge Gradients

Similar to the positions gradients above, we can compute the gradient of the
energy with respect to atomic charges in the following way:

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

For a batch of samples, you may need to use the `autograd` interface more
explicitly:

```python
charges.requires_grad_(True)
energy_per_atom = ewald_summation(...)
energy_per_system = torch.zeros(3, device=positions.device)
# scatter add based on the system index mapping
energy_per_system.scatter_add_(0, batch_idx.long(), energies)
# now compute the derivatives
(charge_gradients, _) = torch.autograd.grad(
  outputs=[energy_per_system,]
  inputs=[charges,]
  grad_outputs=torch.ones_like(charges)
)
```

### Virial / Stress

Both Ewald and PME provide explicit virial computation via `compute_virial=True`.
The virial is differentiable by default: when `compute_virial=True` and inputs
require gradients, stress-based losses automatically back-propagate to model parameters.

**Convention:**

- Real-space: $W_\text{real} = -\frac{1}{2} \sum_{i<j} \mathbf{r}_{ij} \otimes \mathbf{F}_{ij}$,
  where $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$ and $\mathbf{F}_{ij}$ is the force on atom $i$ due to atom $j$.
- Reciprocal-space: $W_\text{recip}(k) = E(k) \left[\delta_{ab} - \frac{2 k_a k_b}{k^2}\left(1 + \frac{k^2}{4\alpha^2}\right)\right]$
- Stress: $\sigma = W / V$ where $V = |\det(\mathbf{C})|$
- The virial convention is validated against finite-difference strain derivatives
  of the energy ($W_{ab} = -\partial E / \partial \varepsilon_{ab}$) in the test suite.

**Ewald summation with virial:**

```python
energies, forces, virial = ewald_summation(
    positions, charges, cell,
    neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
    compute_forces=True,
    compute_virial=True,
)

# Single system: virial shape (1, 3, 3)
volume = torch.abs(torch.linalg.det(cell))          # scalar
stress = virial.squeeze(0) / volume                  # (3, 3)

# Batch: virial shape (B, 3, 3)
volume = torch.abs(torch.linalg.det(cell))           # (B,)
stress = virial / volume[:, None, None]              # (B, 3, 3)
```

**PME with virial:**

```python
energies, forces, virial = particle_mesh_ewald(
    positions, charges, cell,
    neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
    compute_forces=True,
    compute_virial=True,
)

# Single system
volume = torch.abs(torch.linalg.det(cell))
stress = virial.squeeze(0) / volume                  # (3, 3)

# Batch
volume = torch.abs(torch.linalg.det(cell))           # (B,)
stress = virial / volume[:, None, None]              # (B, 3, 3)
```

**MLIP training loss example:**

```python
# Forward pass with explicit outputs
energies, forces, virial = ewald_summation(
    positions, charges, cell,
    neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
    compute_forces=True,
    compute_virial=True,
)

# Compute stress (single system shown; for batch use volume[:, None, None])
volume = torch.abs(torch.linalg.det(cell))
pred_stress = virial.squeeze(0) / volume

loss = (
    w_energy * (energies.sum() - E_target) ** 2
    + w_forces * (forces - F_target).pow(2).sum()
    + w_stress * (pred_stress - stress_target).pow(2).sum()
)
loss.backward()  # Stress-loss gradients flow automatically with compute_virial=True
```

:::{note}
When `compute_virial=True` and inputs track gradients, the virial automatically
participates in the autograd graph. Stress-based losses back-propagate to model
parameters without any additional flags.
:::

:::{tip}
For quick inference or debugging you can also obtain an approximate stress via
`cell.requires_grad_(True)` followed by `energy.backward()` and reading
`cell.grad / volume`. This is first-order only (no higher-order gradients
through the Warp bridge) and is **not** recommended for MLIP training.
:::

## Parameter Estimation

ALCHEMI Toolkit-Ops provides functions to estimate sensible parameters based on
desired accuracy threshold with two functions that share some functionality,
but target the Ewald and PME algorithms respectively.

### Ewald Parameters

The function {func}`~nvalchemiops.torch.interactions.electrostatics.estimate_ewald_parameters`
is used to estimate $\alpha$ and cutoffs for real- and reciprocal-space specifically
for the **Ewald** algorithm:

```python
from nvalchemiops.torch.interactions.electrostatics import estimate_ewald_parameters

params = estimate_ewald_parameters(
    positions=positions,
    cell=cell,
    batch_idx=None,  # or provide for batched systems
    accuracy=1e-6,
)

print(f"α = {params.alpha.item():.4f}")
print(f"r_cutoff = {params.real_space_cutoff.item():.4f}")
print(f"k_cutoff = {params.reciprocal_space_cutoff.item():.4f}")
```

This method returns {func}`~nvalchemiops.torch.interactions.electrostatics.EwaldParameters`, which
is a light data structure that holds parameters used for the Ewald algorithm.

### PME Parameters

The function {func}`~nvalchemiops.torch.interactions.electrostatics.estimate_pme_parameters`
is used to estimate $\alpha$, the real-space cutoff, and mesh specifications specifically
for the PME algorithm; the value of $\alpha$ is determined the same way as for Ewald.

```python
from nvalchemiops.torch.interactions.electrostatics import estimate_pme_parameters

params = estimate_pme_parameters(
    positions=positions,
    cell=cell,
    batch_idx=None,
    accuracy=1e-6,
)

print(f"α = {params.alpha.item():.4f}")
print(f"Mesh: {params.mesh_dimensions}")
print(f"r_cutoff = {params.real_space_cutoff.item():.4f}")
```

This method returns {func}`~nvalchemiops.torch.interactions.electrostatics.PMEParameters`, which
is a light data structure that holds parameters used for the particle-mesh Ewald algorithm.

## Units

The electrostatics functions are unit-agnostic; they work in whatever consistent
unit system you provide. Common conventions:

| Unit System | Positions | Energy | Charge |
|-------------|-----------|--------|--------|
| Atomic units | Bohr | Hartree | e |
| eV-Angstrom | Å | eV | e |
| LAMMPS "real" | Å | kcal/mol | e |

```{important}
Ensure consistency between your position units, cell units, and cutoff values.
The `alpha` parameter has units of inverse length.
```

For atomic units (Bohr/Hartree), no additional constants are needed. For other
unit systems, you may need to multiply energies by a Coulomb constant:

```python
# eV-Angstrom: k_e ~ 14.3996 eV·Å
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
- Higher orders: Smoother but wider support

Higher spline orders provide better accuracy but spread charges over more grid
points. Order 4 (cubic) is the standard choice, balancing accuracy and efficiency.

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

- Check for overlapping atoms (r → 0)
- Verify cell volume is positive
- Ensure charges are finite

**Memory issues with large meshes**:
PME mesh memory scales as $n_x \times n_y \times n_z$. For very large cells, consider using
coarser mesh spacing. It may also be worth comparing compute requirements between Ewald
and PME algorithms.

### Validation

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

## References

- Ewald, P. P. (1921). "Die Berechnung optischer und elektrostatischer Gitterpotentiale."
  *Ann. Phys.* 369, 253-287.
  [DOI: 10.1002/andp.19213690304](https://doi.org/10.1002/andp.19213690304)

- Darden, T.; York, D.; Pedersen, L. (1993). "Particle mesh Ewald: An N⋅log(N)
  method for Ewald sums in large systems." *J. Chem. Phys.* 98, 10089.
  [DOI: 10.1063/1.464397](https://doi.org/10.1063/1.464397)

- Essmann, U.; Perera, L.; Berkowitz, M. L.; Darden, T.; Lee, H.; Pedersen, L. G.
  (1995). "A smooth particle mesh Ewald method." *J. Chem. Phys.* 103, 8577.
  [DOI: 10.1063/1.470117](https://doi.org/10.1063/1.470117)

- Kolafa, J.; Perram, J. W. (1992). "Cutoff Errors in the Ewald Summation Formulae
  for Point Charge Systems." *Mol. Sim.* 9, 351-368.
  [DOI: 10.1080/08927029208049126](https://doi.org/10.1080/08927029208049126)

- Sagui, C.; Darden, T. A. (1999). "Molecular Dynamics Simulations of Biomolecules:
  Long-Range Electrostatic Effects." *Annu. Rev. Biophys. Biomol. Struct.* 28, 155-179.
  [DOI: 10.1146/annurev.biophys.28.1.155](https://doi.org/10.1146/annurev.biophys.28.1.155)

---

For detailed API documentation, see the [PyTorch API](../../modules/torch/electrostatics) and [Warp API](../../modules/warp/electrostatics) references.

<!-- markdownlint-disable MD013 -->

(dispersion_userguide)=

# DFT-D3(BJ) Dispersion Correction

Dispersion corrections account for van der Waals interactions that standard DFT
functionals underestimate. ALCHEMI Toolkit-Ops provides GPU-accelerated DFT-D3
with Becke-Johnson damping via [NVIDIA Warp](https://nvidia.github.io/warp/),
supporting batched computation, periodic systems, and full `torch.compile`
compatibility.

```{tip}
The current implementation computes two-body terms only (C6 and C8). Three-body
Axilrod-Teller-Muto (ATM/C9) contributions are not included.
```

## Why Dispersion Correction Matters

Standard DFT functionals systematically underestimate long-range dispersion
due to their local or semi-local nature. This leads to significant errors in:

- Binding energies of weakly bound complexes
- Molecular crystal structures and lattice energies
- Conformational energies of flexible molecules
- Adsorption energies on surfaces

DFT-D3 adds an empirical pairwise correction with environment-dependent C6
coefficients that adapt based on coordination numbers. The Becke-Johnson damping
variant provides improved short-range behavior for molecular geometries.

## Quick Start

```{important}
**DFT-D3 parameters must be explicitly provided** via `d3_params` (a
{class}`~nvalchemiops.interactions.dispersion.dftd3.D3Parameters` instance or
dictionary). See [Parameter Setup](#parameter-setup) for details.

**Unit consistency is required**: Standard D3 parameters use atomic units---
Bohr for lengths, Hartree for energies. Unit mismatches may cause the neighbor list
calculation to run out of memory (e.g. cutoff in Bohr, but positions are in
Angstroms) or yield an unconverged estimate of the dispersion corrections (i.e.
cutoff in Angstrom, positions in Bohr). See [Units](dispersion_units) for guidance
on units for each parameter.
```

::::{tab-set}

:::{tab-item} Neighbor Matrix (Dense)
:sync: matrix

```python
from nvalchemiops.interactions.dispersion.dftd3 import dftd3
from nvalchemiops.neighborlist import neighbor_list

# Build neighbor matrix; 50 Bohr cutoff
neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff=50.0, cell=cell, pbc=pbc
)

# Compute dispersion correction (PBE functional)
energy, forces, coord_num = dftd3(
    positions=positions,           # [num_atoms, 3] in Bohr
    numbers=numbers,               # [num_atoms] atomic numbers
    neighbor_matrix=neighbor_matrix,
    a1=0.3981, a2=4.4211, s8=0.7875,
    d3_params=d3_params,
)
```

:::

:::{tab-item} Neighbor List (Sparse COO)
:sync: coo

```python
from nvalchemiops.interactions.dispersion.dftd3 import dftd3
from nvalchemiops.neighborlist import neighbor_list

# Build neighbor list in COO format; 50 Bohr cutoff
neighbor_list_coo, neighbor_list_ptr, unit_shifts = neighbor_list(
    positions, cutoff=50.0, cell=cell, pbc=pbc, return_neighbor_list=True
)

# Compute dispersion correction (PBE functional)
energy, forces, coord_num = dftd3(
    positions=positions,           # [num_atoms, 3] in Bohr
    numbers=numbers,               # [num_atoms] atomic numbers
    neighbor_list=neighbor_list_coo,  # [2, num_pairs]
    neighbor_ptr=neighbor_list_ptr,
    a1=0.3981, a2=4.4211, s8=0.7875,
    d3_params=d3_params,
)
```

:::

::::

### Periodic Boundary Conditions

For periodic systems, provide the cell and shift tensors:

::::{tab-set}

:::{tab-item} Neighbor Matrix (Dense)
:sync: matrix

```python
energy, forces, coord_num, virial = dftd3(
    positions=positions,
    numbers=numbers,
    neighbor_matrix=neighbor_matrix,
    neighbor_matrix_shifts=neighbor_matrix_shifts,   # [num_atoms, max_neighbors, 3]
    cell=cell,                                       # [num_systems, 3, 3]
    a1=0.3981, a2=4.4211, s8=0.7875,
    d3_params=d3_params,
    compute_virial=True                              # also compute virial
)
```

:::

:::{tab-item} Neighbor List (Sparse COO)
:sync: coo

```python
energy, forces, coord_num = dftd3(
    positions=positions,
    numbers=numbers,
    neighbor_list=neighbor_list_coo,
    unit_shifts=unit_shifts,           # [num_pairs, 3]
    cell=cell,                         # [num_systems, 3, 3]
    a1=0.3981, a2=4.4211, s8=0.7875,
    d3_params=d3_params,
)
```

:::

::::

Use `batch_idx` to specify multi-system batches, mapping each atom to its system.

## Data Formats

### Neighbor Representations

ALCHEMI Toolkit-Ops supports two neighbor formats that produce identical results:

Neighbor Matrix (default)
: Dense tensor of shape `[num_atoms, max_neighbors]`. Each row contains neighbor
  indices for that atom, padded with `fill_value` (typically `num_atoms`).
  Use with `neighbor_matrix` and `neighbor_matrix_shifts` arguments.

Neighbor List (COO)
: Sparse tensor of shape `[2, num_pairs]` where row 0 contains source indices
  and row 1 contains target indices. No padding needed.
  Use with `neighbor_list` and `unit_shifts` arguments.

### When to Use Each Format

We refer the reader for a more in-depth discussion in the
[neighbor list documentation](./neighborlist.md), but briefly:

**Neighbor Matrix** is preferred when:

- Using `torch.compile` (fixed memory layout avoids graph breaks)
- Systems have dense, uniform neighbor distributions

**Neighbor List (COO)** is preferred when:

- Integrating with graph neural network libraries (PyG, DGL)
- Systems are sparse with highly variable neighbors per atom
- Memory efficiency is critical

```{note}
The neighbor representation should be **symmetric** (bidirectional): if atom `j`
is a neighbor of atom `i`, then atom `i` should also be a neighbor of atom `j`.
```

### Units

(dispersion_units)=

While the DFT-D3 kernels themselves are unit-agnostic, the reference parameters themselves
typically use atomic units. The table below lists quantities and their conversions from
conventional units that are more commonly encountered in computational chemistry
and materials science:

| Quantity | Unit | Conversion from SI |
|----------|------|-------------------|
| Positions | Bohr | $(\text{Å} \times 1.8897259886)$ |
| Energy (output) | Hartree | $(\times 27.211 \rightarrow \text{eV})$ |
| Forces (output) | Hartree/Bohr | $(\times 51.422 \rightarrow \text{eV/Å})$ |
| Virial (output) | Hartree | $(\times 27.211 \rightarrow \text{eV})$ |
| `a2` parameter | Bohr | Same as positions |
| Cutoffs | Bohr | Same as positions |
| C6 coefficients | (Hartree $\cdot$ Bohr$^6$) | — |

Coordination numbers are dimensionless.

```{tip}
Use `scipy.constants` for precise conversions:
`scipy.constants.value('atomic unit of length')` for Bohr,
`scipy.constants.value('Hartree energy in eV')` for Hartree.
```

### Data Types

| Tensor | Dtype | Notes |
|--------|-------|-------|
| Positions | `float32` or `float64` | FP64 used for distance vectors only |
| Cell | `float32` or `float64` | Same format as Positions |
| Atomic numbers | `int32` | |
| Neighbor indices | `int32` | |
| Energy (output) | `float32` | Always |
| Forces (output) | `float32` | Always |
| Virial (output) | `float32` | Always |
| Coordination numbers (output) | `float32` | Always |

## Parameter Setup

DFT-D3 requires reference parameters that **must be explicitly provided**.
Three options are available:

### Option 1: D3Parameters Dataclass (Recommended)

```python
from nvalchemiops.interactions.dispersion.dftd3 import D3Parameters

d3_params = D3Parameters(
    rcov=covalent_radii,      # [max_Z+1] in Bohr
    r4r2=r4r2_values,         # [max_Z+1] <r^4>/<r^2> expectation values
    c6ab=c6_reference,        # [max_Z+1, max_Z+1, 5, 5] C6 grid
    cn_ref=coord_num_ref,     # [max_Z+1, max_Z+1, 5, 5] CN grid
)

energy, forces, coord_num = dftd3(
    positions, numbers, neighbor_matrix,
    a1=0.3981, a2=4.4211, s8=0.7875,
    d3_params=d3_params,
)
```

### Option 2: Dictionary

```python
d3_params = {
    "rcov": covalent_radii,
    "r4r2": r4r2_values,
    "c6ab": c6_reference,
    "cn_ref": coord_num_ref,
}
```

### Option 3: Individual Parameters

```python
energy, forces, coord_num = dftd3(
    positions, numbers, neighbor_matrix,
    a1=0.3981, a2=4.4211, s8=0.7875,
    covalent_radii=covalent_radii,
    r4r2=r4r2_values,
    c6_reference=c6_reference,
    coord_num_ref=coord_num_ref,
)
```

### Obtaining Parameters

Parameter files are available from the
[Grimme group website](https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/).
See `examples/interactions/utils.py` for loading utilities.

### Functional-Specific Damping Parameters

Common BJ damping parameters (`a1`, `a2`, `s8`):

| Functional | a1 | a2 (Bohr) | s8 |
|------------|----|-----------|----|
| PBE | 0.3981 | 4.4211 | 0.7875 |
| PBE0 | 0.4145 | 4.8593 | 1.2177 |
| B3LYP | 0.3981 | 4.4211 | 1.9889 |

See the [DFT-D3 parameters page](https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/)
for a complete list.

## Performance Tuning

### Key Parameters

`s5_smoothing_on` / `s5_smoothing_off`
: Enable smooth cutoff with the S5 switching function. The function provides
  $C^2$ continuity at both boundaries, preventing force discontinuities.

### torch.compile Compatibility

Both neighbor formats support `torch.compile` for JIT optimization:

```python
compiled_dftd3 = torch.compile(dftd3)

energy, forces, coord_num = compiled_dftd3(
    positions=positions,
    numbers=numbers,
    neighbor_list=neighbor_list_coo,
    a1=0.3981, a2=4.4211, s8=0.7875,
    d3_params=d3_params,
    num_systems=num_systems  # will introduce CUDA graph break if not provided
)
```

### Precision Notes

- Input positions and unit cell can be `float64` for large unit cells or systems far from origin
- Intermediate calculations use `float32` (sufficient for dispersion accuracy)
- All outputs are `float32`

## Usage Patterns

### Single Molecule (Non-Periodic)

```python
import torch
from nvalchemiops.interactions.dispersion.dftd3 import dftd3
from nvalchemiops.neighborlist import neighbor_list

ANGSTROM_TO_BOHR = 1.8897259886
HARTREE_TO_EV = 27.211386245981

# Water molecule (Ångström → Bohr)
positions_ang = torch.tensor([
    [0.0000,  0.0000,  0.1173],  # O
    [0.0000,  0.7572, -0.4692],  # H
    [0.0000, -0.7572, -0.4692],  # H
], dtype=torch.float32, device="cuda")
positions = positions_ang * ANGSTROM_TO_BOHR

numbers = torch.tensor([8, 1, 1], dtype=torch.int32, device="cuda")

# Non-periodic: use large box with pbc=False
cell = torch.eye(3, dtype=torch.float32, device="cuda").unsqueeze(0) * 100.0
pbc = torch.tensor([False, False, False], device="cuda")

neighbor_matrix, num_neighbors, _ = neighbor_list(
    positions, cutoff=50.0, cell=cell, pbc=pbc
)

# d3_params loaded from file (see examples/interactions/utils.py)
energy, forces, coord_num = dftd3(
    positions=positions,
    numbers=numbers,
    neighbor_matrix=neighbor_matrix,
    a1=0.3981, a2=4.4211, s8=0.7875,
    d3_params=d3_params,
)

print(f"Dispersion energy: {energy[0].item():.6f} Ha")
print(f"                   {energy[0].item() * HARTREE_TO_EV:.6f} eV")
```

### Batched Periodic Crystals

```python
import torch
from nvalchemiops.interactions.dispersion.dftd3 import dftd3
from nvalchemiops.neighborlist import neighbor_list

ANGSTROM_TO_BOHR = 1.8897259886

# Batch of 4 systems, 8 atoms each
BATCH_SIZE = 4
atoms_per_system = 8
total_atoms = BATCH_SIZE * atoms_per_system

positions_ang = torch.randn(total_atoms, 3, device="cuda")
positions = positions_ang * ANGSTROM_TO_BOHR
numbers = torch.randint(1, 20, (total_atoms,), dtype=torch.int32, device="cuda")

# Per-system cells (Å → Bohr)
cells = torch.eye(3, device="cuda").unsqueeze(0).repeat(BATCH_SIZE, 1, 1) * 10.0
cells = cells * ANGSTROM_TO_BOHR

pbc = torch.tensor([True, True, True], device="cuda").unsqueeze(0).repeat(BATCH_SIZE, 1)

batch_idx = torch.repeat_interleave(
    torch.arange(BATCH_SIZE, dtype=torch.int32, device="cuda"),
    atoms_per_system
)
batch_ptr = torch.arange(0, total_atoms + 1, atoms_per_system, dtype=torch.int32, device="cuda")

# Build neighbors
neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff=20.0 * ANGSTROM_TO_BOHR,
    cell=cells, pbc=pbc, batch_idx=batch_idx, batch_ptr=batch_ptr,
    method="batch_cell_list"
)

energy, forces, coord_num, virial = dftd3(
    positions=positions,
    numbers=numbers,
    neighbor_matrix=neighbor_matrix,
    neighbor_matrix_shifts=shifts,
    cell=cells,
    batch_idx=batch_idx,
    a1=0.4145, a2=4.8593, s8=1.2177,  # PBE0
    d3_params=d3_params,
    compute_virial=True
)

print(f"Energies per system (Ha): {energy}")
```

### Smooth Cutoff

```python
energy, forces, coord_num = dftd3(
    positions=positions,
    numbers=numbers,
    neighbor_matrix=neighbor_matrix,
    a1=0.3981, a2=4.4211, s8=0.7875,
    d3_params=d3_params,
    s5_smoothing_on=40.0,   # Start transition at 40 Bohr
    s5_smoothing_off=50.0,  # Complete cutoff at 50 Bohr
)
```

## Theory Background

This section summarizes the DFT-D3 method [^grimme2010] with Becke-Johnson
damping [^grimme2011]. For complete derivations and benchmarks, see the
original publications.

### Dispersion Energy Expression

The DFT-D3 energy with Becke-Johnson damping:

```{math}
E_{\text{disp}} = -\frac{1}{2} \sum_{i \neq j} S_w(r_{ij}) \left[
\frac{s_6 C_6^{ij}}{r_{ij}^6 + f_{\text{damp},6}^6} +
\frac{s_8 C_8^{ij}}{r_{ij}^8 + f_{\text{damp},8}^8}
\right]
```

where:

- $r_{ij}$: interatomic distance
- $C_6^{ij}, C_8^{ij}$: dispersion coefficients
- $s_6, s_8$: functional-dependent scaling factors
- $f_{\text{damp}}$: Becke-Johnson damping function
- $S_w(r)$: optional smooth switching function

### Coordination-Dependent C6

A key innovation in DFT-D3 [^grimme2010] is environment-dependent C6 coefficients.
The C6 coefficient adapts to the local chemical environment via Gaussian
interpolation over reference coordination numbers:

```{math}
C_6(\text{CN}_i, \text{CN}_j) =
\frac{\sum_{pq} C_6^{\text{ref}}[p,q] \cdot L_{pq}}{\sum_{pq} L_{pq}}
```

Coordination numbers use a geometric counting function:

```{math}
\text{CN}_i = \sum_{j \neq i} \frac{1}{1 + \exp\left[k_1 \left(\frac{r_{\text{cov}}}{r_{ij}} - 1\right)\right]}
```

### Becke-Johnson Damping

The BJ damping function [^grimme2011] prevents unphysical short-range behavior
while providing improved accuracy for equilibrium geometries:

```{math}
f_{\text{damp}} = a_1 \sqrt{C_8^{ij}/C_6^{ij}} + a_2
```

### Force Calculation

Forces include both direct and coordination-dependent contributions via chain rule:

```{math}
\mathbf{F}_i = -\sum_j \left[\left.\frac{\partial E_{ij}}{\partial \mathbf{r}_i}
\right|_{\text{CN}} +
\left(\sum_k \frac{\partial E_{ik}}{\partial \text{CN}_i}\right)
\frac{\partial \text{CN}_i}{\partial \mathbf{r}_i}\right]
```

This decomposition into direct and chain-rule terms is central to the multi-pass
kernel architecture described in [Implementation Details](#implementation-details).

## Implementation Details

### Why Multi-Pass Kernels?

The force calculation requires computing the chain-rule contribution:

```{math}
\mathbf{F}_{\text{chain},i} = -\left(\sum_k \frac{\partial E_{ik}}{\partial \text{CN}_i}\right)
\frac{\partial \text{CN}_i}{\partial \mathbf{r}_i}
```

The term $\sum_k \partial E_{ik}/\partial \text{CN}_i$ must be accumulated over
**all pairs** involving atom $i$ before computing the CN-dependent forces. This
creates a data dependency that prevents computing direct and chain-rule forces
in a single pass. The multi-pass architecture separates these computations:

Pass 0 (PBC only)
: Convert integer unit cell shifts to Cartesian coordinates using the lattice
  vectors. This is performed once and reused in subsequent passes.

Pass 1
: Compute coordination numbers $\text{CN}_i$ for all atoms using the geometric
  counting function. These are needed for C6 interpolation in Pass 2.

Pass 2
: For each atom pair: interpolate C6 coefficients, compute damped dispersion
  energy, compute direct forces $\mathbf{F}_{\text{direct}}$, and **accumulate**
  $\partial E/\partial \text{CN}_i$ into a per-atom buffer.

Pass 3
: Using the accumulated $\partial E/\partial \text{CN}$ values from Pass 2,
  compute the chain-rule force contribution and add it to the direct forces.

### Neighbor Format Dispatch

The {func}`~nvalchemiops.interactions.dispersion.dftd3.dftd3`
function dispatches to different kernel implementations based on the neighbor
representation format:

- **Neighbor matrix** (`neighbor_matrix` argument): Dispatches to
  `_dftd3_nm_op`, which launches kernels that iterate over a
  dense `[num_atoms, max_neighbors]` array.

- **Neighbor list** (`neighbor_list` + `neighbor_ptr` arguments): Dispatches to
  `_dftd3_nl_op`, which launches kernels that
  use CSR (Compressed Sparse Row) format for memory-efficient sparse traversal.

Both paths execute the same four-pass algorithm and produce identical results.
The choice affects memory layout and access patterns but not numerical output.

### Precision Handling

The kernels support both `float32` and `float64` input positions through Warp's
overload mechanism. At module load time, kernel overloads are registered for
each precision:

- `float32` positions use `wp.vec3f` vectors and `wp.mat33f` matrices
- `float64` positions use `wp.vec3d` vectors and `wp.mat33d` matrices

Distance vectors are computed at the input precision, but all dispersion
calculations (C6 interpolation, damping, energy/force accumulation) use
`float32` for efficiency. Outputs are always `float32`.

### Numerical Stability

- **Log-sum-exp trick**: The Gaussian-weighted C6 interpolation computes
  $\sum_k w_k C_6^{(k)} / \sum_k w_k$ where $w_k = \exp(\text{arg}_k)$. To
  prevent overflow, all exponentials are computed relative to the maximum
  argument: $\exp(\text{arg}_k - \text{arg}_{\max})$.

- **Threshold skipping**: Contributions with $\exp(\text{arg}) < 10^{-12}$
  (relative to max) are skipped to avoid unnecessary computation.

## References

[^grimme2010]: Grimme, S.; Antony, J.; Ehrlich, S.; Krieg, H. "A consistent and
    accurate ab initio parametrization of density functional dispersion
    correction (DFT-D) for the 94 elements H-Pu." _J. Chem. Phys._ **2010**,
    _132_, 154104. [DOI: 10.1063/1.3382344](https://doi.org/10.1063/1.3382344)

[^grimme2011]: Grimme, S.; Ehrlich, S.; Goerigk, L. "Effect of the damping
    function in dispersion corrected density functional theory."
    _J. Comput. Chem._ **2011**, _32_, 1456-1465.
    [DOI: 10.1002/jcc.21759](https://doi.org/10.1002/jcc.21759)

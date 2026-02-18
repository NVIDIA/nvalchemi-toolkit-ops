# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DFT-D3(BJ) Dispersion Correction - Warp Kernel Implementation

This module implements the DFT-D3 dispersion correction with Becke-Johnson (BJ)
damping as Warp GPU/CPU kernels. The implementation provides efficient computation
of dispersion energies and forces using a multi-pass algorithm with support for
periodic boundary conditions, batched systems, and smooth cutoff functions.

For detailed theory, usage examples, and parameter setup, see the
:doc:`DFT-D3 User Guide </userguide/components/dispersion>`.

Multi-Pass Kernel Architecture
-------------------------------
The implementation uses four kernel passes to efficiently handle the chain rule
dependency in force calculations:

1. **Pass 0 (_compute_cartesian_shifts)**: [PBC only] Convert unit cell shifts
   to Cartesian coordinates

2. **Pass 1 (_cn_kernel)**: Compute coordination numbers using geometric counting
   function

3. **Pass 2 (_direct_forces_and_dE_dCN_kernel)**: Compute C6 interpolation,
   dispersion energy, direct forces, and accumulate :math:`\\partial E/\\partial \\text{CN}`

4. **Pass 3 (_cn_forces_contrib_kernel)**: Add CN-dependent force contribution
   using precomputed :math:`\\partial E/\\partial \\text{CN}` values

Warp Launchers (Framework-Agnostic)
------------------------------------
This module provides four framework-agnostic warp launcher functions that accept
warp arrays directly, with distinct signatures based on neighbor format and PBC support.
These are called by framework-specific wrappers (PyTorch, JAX) after converting
framework tensors to warp arrays:

**Neighbor Matrix Format:**

- ``dftd3_matrix`` - Non-periodic systems (no PBC parameters)
- ``dftd3_matrix_pbc`` - Periodic systems (requires cell and neighbor_matrix_shifts)

**Neighbor List (CSR) Format:**

- ``dftd3`` - Non-periodic systems (no PBC parameters)
- ``dftd3_pbc`` - Periodic systems (requires cell and unit_shifts)

.. code-block:: python

    from nvalchemiops.interactions.dispersion._dftd3 import (
        dftd3_matrix,
        dftd3_matrix_pbc,
        dftd3,
        dftd3_pbc,
    )

    # Neighbor matrix format - non-periodic
    dftd3_matrix(
        positions=positions_wp,  # warp array
        numbers=numbers_wp,
        neighbor_matrix=neighbor_matrix_wp,
        covalent_radii=covalent_radii_wp,
        r4r2=r4r2_wp,
        c6_reference=c6_reference_wp,
        coord_num_ref=coord_num_ref_wp,
        a1=0.3981, a2=4.4211, s8=1.9889,
        coord_num=coord_num_wp,  # pre-allocated output
        forces=forces_wp,  # pre-allocated output
        energy=energy_wp,  # pre-allocated output
        virial=virial_wp,  # pre-allocated output (not computed for non-PBC)
        vec_dtype=wp.vec3f,
    )

    # Neighbor matrix format - periodic (PBC)
    dftd3_matrix_pbc(
        positions=positions_wp,
        numbers=numbers_wp,
        neighbor_matrix=neighbor_matrix_wp,
        cell=cell_wp,  # REQUIRED for PBC
        neighbor_matrix_shifts=shifts_wp,  # REQUIRED for PBC
        covalent_radii=covalent_radii_wp,
        r4r2=r4r2_wp,
        c6_reference=c6_reference_wp,
        coord_num_ref=coord_num_ref_wp,
        a1=0.3981, a2=4.4211, s8=1.9889,
        coord_num=coord_num_wp,
        forces=forces_wp,
        energy=energy_wp,
        virial=virial_wp,
        vec_dtype=wp.vec3f,
        compute_virial=True,  # Optional: enable virial computation
    )

    # Neighbor list format (CSR) - non-periodic
    dftd3(
        positions=positions_wp,
        numbers=numbers_wp,
        idx_j=idx_j_wp,
        neighbor_ptr=neighbor_ptr_wp,
        covalent_radii=covalent_radii_wp,
        r4r2=r4r2_wp,
        c6_reference=c6_reference_wp,
        coord_num_ref=coord_num_ref_wp,
        a1=0.3981, a2=4.4211, s8=1.9889,
        coord_num=coord_num_wp,
        forces=forces_wp,
        energy=energy_wp,
        virial=virial_wp,  # pre-allocated output (not computed for non-PBC)
        vec_dtype=wp.vec3f,
    )

    # Neighbor list format (CSR) - periodic (PBC)
    dftd3_pbc(
        positions=positions_wp,
        numbers=numbers_wp,
        idx_j=idx_j_wp,
        neighbor_ptr=neighbor_ptr_wp,
        cell=cell_wp,  # REQUIRED for PBC
        unit_shifts=unit_shifts_wp,  # REQUIRED for PBC
        covalent_radii=covalent_radii_wp,
        r4r2=r4r2_wp,
        c6_reference=c6_reference_wp,
        coord_num_ref=coord_num_ref_wp,
        a1=0.3981, a2=4.4211, s8=1.9889,
        coord_num=coord_num_wp,
        forces=forces_wp,
        energy=energy_wp,
        virial=virial_wp,
        vec_dtype=wp.vec3f,
        compute_virial=True,  # Optional: enable virial computation
    )

PyTorch Interface
-----------------
For PyTorch integration, use the high-level wrapper in the torch namespace:

.. code-block:: python

    from nvalchemiops.torch.interactions.dispersion import dftd3, D3Parameters

    # Using neighbor matrix format
    energy, forces, coord_num = dftd3(
        positions, numbers,
        neighbor_matrix=neighbor_matrix,
        a1=0.3981, a2=4.4211, s8=1.9889,
        d3_params=d3_params,  # D3Parameters instance or dict
        cell=cell,  # Optional for PBC
        neighbor_matrix_shifts=neighbor_matrix_shifts,  # Optional for PBC
    )

    # Using neighbor list format (sparse COO)
    energy, forces, coord_num = dftd3(
        positions, numbers,
        neighbor_list=neighbor_list,  # shape (2, num_pairs)
        a1=0.3981, a2=4.4211, s8=1.9889,
        d3_params=d3_params,
        cell=cell,  # Optional for PBC
        unit_shifts=unit_shifts,  # Optional for PBC, shape (num_pairs, 3)
    )

Data Structure Requirements
---------------------------
**Neighbor Formats**

The implementation supports two neighbor representation formats:

1. **Neighbor Matrix Format** (dense): `[num_atoms, max_neighbors]` where
   `neighbor_matrix[i, k]` is the k-th neighbor of atom i. Padding entries use
   values >= `fill_value` (typically `num_atoms`).

2. **Neighbor List Format** (sparse COO): `[2, num_pairs]` where row 0 contains
   source atom indices and row 1 contains target atom indices. No padding needed.

Both formats can be generated by :func:`nvalchemiops.neighborlist.neighbor_list` using
the `return_neighbor_list` parameter.

**Parameter Arrays**

- `covalent_radii`: `[max_Z+1]` float32
- `r4r2`: `[max_Z+1]` float32
- `c6_reference`: `[max_Z+1, max_Z+1, 5, 5]` float32
- `coord_num_ref`: `[max_Z+1, max_Z+1, 5, 5]` float32

Index 0 reserved for padding; valid atomic numbers 1 to max_Z.

**Periodic Boundary Conditions**

- `cell`: `[num_systems, 3, 3]` lattice vectors (row format)
- For neighbor matrix: `neighbor_matrix_shifts`: `[num_atoms, max_neighbors, 3]` int32 unit cell shifts
- For neighbor list: `unit_shifts`: `[num_pairs, 3]` int32 unit cell shifts

Units
-----
Kernels are **unit-agnostic** but require consistency. Standard Grimme group
parameters use **atomic units (Bohr, Hartree)**, which is recommended:

- Positions, covalent radii, `a2`, cutoffs: Bohr
- Energy output: Hartree
- Forces output: Hartree/Bohr
- Parameter `k1`: 1/Bohr

Technical Notes
---------------
- Supports float32 and float64 positions and cell. Outputs are always float32
- **Two-body only**: Axilrod-Teller-Muto (C9) three-body terms not included

See Also
--------
:class:`D3Parameters` : Dataclass for parameter validation and management
:func:`dftd3` : Main PyTorch interface function
:doc:`/userguide/components/dispersion` : Complete user guide with theory and examples
"""

from __future__ import annotations

from typing import Any

import warp as wp

__all__ = [
    # Warp launchers (framework-agnostic public API)
    "dftd3_matrix",
    "dftd3_matrix_pbc",
    "dftd3",
    "dftd3_pbc",
    # Kernel overload dictionaries (for framework bindings)
    "_compute_cartesian_shifts_matrix_overload",
    "_cn_kernel_matrix_overload",
    "_direct_forces_and_dE_dCN_kernel_matrix_overload",
    "_cn_forces_contrib_kernel_matrix_overload",
    "_compute_cartesian_shifts_overload",
    "_cn_kernel_overload",
    "_direct_forces_and_dE_dCN_kernel_overload",
    "_cn_forces_contrib_kernel_overload",
]

# ==============================================================================
# Helper Functions
# ==============================================================================


@wp.func
def _s5_switch(
    r: wp.float32,
    r_on: wp.float32,
    r_off: wp.float32,
    inv_w: wp.float32,
) -> tuple[wp.float32, wp.float32]:
    """
    C² smooth switching function for cutoff smoothing.

    This function provides a smooth transition from 1 to 0 over the interval
    [r_on, r_off]. The switching polynomial S5(t) has continuous first and
    second derivatives at the boundaries.

    Parameters
    ----------
    r : float32
        Distance between atoms
    r_on : float32
        Distance where switching begins
    r_off : float32
        Distance where switching completes
    inv_w : float32
        Precomputed 1/(r_off - r_on) for efficiency

    Returns
    -------
    Sw : float32
        Switching function value Sw(r) ∈ [0, 1]
    dSw_dr : float32
        Derivative of switching function with respect to r

    Notes
    -----
    The switching function is defined as:

    .. math::

        S_w(r) = \\begin{cases}
        1 & \\text{if } r \\leq r_{\\text{on}} \\\\
        1 - S_5(t) & \\text{if } r_{\\text{on}} < r < r_{\\text{off}} \\\\
        0 & \\text{if } r \\geq r_{\\text{off}}
        \\end{cases}

    where :math:`t = (r - r_{\\text{on}})/(r_{\\text{off}} - r_{\\text{on}}) \\in (0,1)` and

    .. math::

        S_5(t) = 10t^3 - 15t^4 + 6t^5

    The derivative is:

    .. math::

        \\frac{dS_5}{dt} = 30t^2 - 60t^3 + 30t^4

        \\frac{dS_w}{dr} = -\\frac{dS_5}{dt} \\cdot \\frac{1}{r_{\\text{off}} - r_{\\text{on}}}

    This ensures :math:`C^2` continuity (continuous function, first, and second derivatives)
    at both :math:`r_{\\text{on}}` and :math:`r_{\\text{off}}` boundaries.

    See Also
    --------
    :func:`_direct_forces_and_dE_dCN_kernel_matrix` : Uses this switching function
    for cutoff smoothing (neighbor matrix)
    :func:`_direct_forces_and_dE_dCN_kernel` : Uses this switching function
    for cutoff smoothing (neighbor list)
    """
    if r_off <= r_on:
        # disabled or degenerate: no switching
        return 1.0, 0.0
    if r <= r_on:
        return 1.0, 0.0
    if r >= r_off:
        return 0.0, 0.0
    t = (r - r_on) * inv_w  # t in (0,1)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    switch = 1.0 - (10.0 * t3 - 15.0 * t4 + 6.0 * t5)
    dSdt = -30.0 * t2 + 60.0 * t3 - 30.0 * t4  # NOSONAR (S125) "math formula"
    dSw_dr = dSdt * inv_w  # NOSONAR (S125) "math formula"
    return switch, dSw_dr


@wp.func
def _c6ab_interpolate(
    cn_i: wp.float32,
    cn_j: wp.float32,
    c6ab_mat: wp.array2d(dtype=wp.float32),
    cnref_i_mat: wp.array2d(dtype=wp.float32),
    cnref_j_mat: wp.array2d(dtype=wp.float32),
    k3: wp.float32,
) -> tuple[wp.float32, wp.float32, wp.float32]:
    """
    Interpolate C6 coefficient and CN derivatives using Gaussian weighting.

    This function performs Gaussian interpolation over a 5x5 reference grid
    to compute the environment-dependent C6 coefficient for an atom pair,
    along with derivatives with respect to coordination numbers.

    Parameters
    ----------
    cn_i : float32
        Coordination number of atom i
    cn_j : float32
        Coordination number of atom j
    c6ab_mat : wp.array2d(dtype=float32)
        C6 reference values [5, 5] for this element pair
    cnref_i_mat : wp.array2d(dtype=float32)
        CN reference grid [5, 5] for atom i
    cnref_j_mat : wp.array2d(dtype=float32)
        CN reference grid [5, 5] for atom j
    k3 : float32
        Gaussian width parameter (typically -4.0)

    Returns
    -------
    c6_ij : float32
        Interpolated C6 coefficient
    dC6_dCNi : float32
        Derivative :math:`\\partial C_6/\\partial \text{CN}_i`
    dC6_dCNj : float32
        Derivative :math:`\\partial C_6/\\partial \text{CN}_j`

    Notes
    -----
    The Gaussian weights are:

    .. math::

        L_{pq} = \\exp\\left(-k_3 \\left[(\\text{CN}_i - \\text{CN}_{\\text{ref},i}[p,q])^2 +
        (\\text{CN}_j - \\text{CN}_{\\text{ref},j}[p,q])^2\\right]\\right)

    The interpolated C6 and derivatives are:

    .. math::

        C_6 = \\frac{\\sum_{pq} C_6^{\\text{ref}}[p,q] L_{pq}}{\\sum_{pq} L_{pq}}

        \\frac{\\partial C_6}{\\partial \\text{CN}_i} = \\frac{2k_3}{w} (z_{d_i} - C_6 w_{d_i})

    where accumulators :math:`w`, :math:`z`, :math:`w_{d_i}`, :math:`z_{d_i}` are
    computed in a single pass over the 5x5 grid.

    See Also
    --------
    :func:`_direct_forces_and_dE_dCN_kernel_matrix` : Calls this function for C6
    coefficient interpolation (neighbor matrix)
    :func:`_direct_forces_and_dE_dCN_kernel` : Calls this function for C6
    coefficient interpolation (neighbor list)
    """
    # log-sum-exp trick to avoid numerical instability
    max_exp = wp.float(-1e20)
    for p in range(5):
        for q in range(5):
            c6_val = c6ab_mat[p, q]
            if c6_val == 0.0:  # NOSONAR (S1244) "gpu kernel"
                continue
            cnref_i = cnref_i_mat[p, q]
            cnref_j = cnref_j_mat[q, p]
            di = cn_i - cnref_i
            dj = cn_j - cnref_j
            exp_arg = k3 * (di * di + dj * dj)
            if exp_arg > max_exp:
                max_exp = exp_arg

    w = float(0.0)
    z = float(0.0)
    w_di = float(0.0)
    w_dj = float(0.0)
    z_di = float(0.0)
    z_dj = float(0.0)

    for p in range(5):
        for q in range(5):
            c6_val = c6ab_mat[p, q]
            if c6_val == 0.0:  # NOSONAR (S1244) "gpu kernel"
                continue
            cnref_i = cnref_i_mat[p, q]
            cnref_j = cnref_j_mat[q, p]  # Note transpose indexing
            di = cn_i - cnref_i
            dj = cn_j - cnref_j
            # Compute exponent argument and skip negligible contributions
            exp_arg = k3 * (di * di + dj * dj) - max_exp
            if exp_arg < -12.0:
                continue
            L = wp.exp(exp_arg)
            w += L
            z += c6_val * L
            w_di += L * di
            w_dj += L * dj
            z_di += c6_val * L * di
            z_dj += c6_val * L * dj

    eps_w = 1e-12
    if w > eps_w:
        w_inv = 1.0 / w
        c6_ij = z * w_inv
        s_i = z_di - c6_ij * w_di
        s_j = z_dj - c6_ij * w_dj
        k3_w_w_inv = (2.0 * k3) * w_inv
        dC6_dCNi = k3_w_w_inv * s_i  # NOSONAR (S125) "math formula"
        dC6_dCNj = k3_w_w_inv * s_j  # NOSONAR (S125) "math formula"
        return c6_ij, dC6_dCNi, dC6_dCNj
    else:
        return 0.0, 0.0, 0.0


@wp.func
def _compute_distance_vector_pbc(
    pos_i: Any,
    pos_j: Any,
    cartesian_shift: Any,
    periodic: bool,
    compute_vectors: bool,
) -> tuple[wp.float32, wp.float32, wp.vec3f, wp.vec3f]:
    """
    Compute distance with optional PBC and vector outputs.

    Parameters
    ----------
    pos_i, pos_j : vec3
        Atomic positions
    cartesian_shift : vec3
        PBC shift (ignored if periodic=False)
    periodic : bool
        Apply PBC shift
    compute_vectors : bool
        If True, compute r_hat; if False, r_hat returns zero vector

    Returns
    -------
    r : float32
        Distance
    r_inv : float32
        Inverse distance (0 if r < 1e-12)
    r_hat : vec3f
        Unit vector (zero vec if compute_vectors=False or r < 1e-12)
    r_ij : vec3f
        Distance vector (always returned)
    """
    if periodic:
        r_ij_native = (pos_j - pos_i) + cartesian_shift
    else:
        r_ij_native = pos_j - pos_i

    r_ij = wp.vec3f(
        wp.float32(r_ij_native[0]),
        wp.float32(r_ij_native[1]),
        wp.float32(r_ij_native[2]),
    )
    r = wp.length(r_ij)

    if r < 1e-12:
        return r, wp.float32(0.0), wp.vec3f(0.0, 0.0, 0.0), r_ij

    r_inv = 1.0 / r

    if compute_vectors:
        r_hat = r_ij * r_inv
        return r, r_inv, r_hat, r_ij
    else:
        return r, r_inv, wp.vec3f(0.0, 0.0, 0.0), r_ij


@wp.func
def _cn_counting(
    r_inv: wp.float32,
    rcov_i: wp.float32,
    rcov_j: wp.float32,
    k1: wp.float32,
    compute_derivative: bool,
) -> tuple[wp.float32, wp.float32]:
    """
    Compute CN counting function with optional derivative.

    Parameters
    ----------
    r_inv : float32
        Inverse distance
    rcov_i, rcov_j : float32
        Covalent radii
    k1 : float32
        Steepness parameter
    compute_derivative : bool
        If True, compute dCN_dr; if False, returns None

    Returns
    -------
    f_cn : float32
        Counting function value
    dCN_dr : float32
        Derivative (zero if compute_derivative=False)
    """
    rcov_ij = rcov_i + rcov_j
    rcov_r_inv = rcov_ij * r_inv
    f_cn = 1.0 / (1.0 + wp.exp(-k1 * (rcov_r_inv - 1.0)))

    if compute_derivative:
        dCN_dr = -f_cn * (1.0 - f_cn) * k1 * rcov_r_inv * r_inv  # NOSONAR (S125)
        return f_cn, dCN_dr
    else:
        return f_cn, wp.float32(0.0)


@wp.func
def _bj_damping(
    r: wp.float32,
    r4r2_i: wp.float32,
    r4r2_j: wp.float32,
    a1: wp.float32,
    a2: wp.float32,
    s6: wp.float32,
    s8: wp.float32,
) -> tuple[wp.float32, wp.float32, wp.float32, wp.float32, wp.float32, wp.float32]:
    """
    Compute Becke-Johnson damping.

    Returns
    -------
    damp_sum, r4r2_ij, r6, r4, den6_inv, den8_inv : float32
    """
    r4r2_ij = 3.0 * r4r2_i * r4r2_j
    r0 = a1 * wp.sqrt(r4r2_ij) + a2

    r2 = r * r
    r4 = r2 * r2
    r6 = r4 * r2
    r8 = r4 * r4

    r0_2 = r0 * r0
    r0_4 = r0_2 * r0_2
    r0_6 = r0_4 * r0_2
    r0_8 = r0_4 * r0_4

    den6 = r6 + r0_6
    den8 = r8 + r0_8
    den6_inv = 1.0 / den6
    den8_inv = 1.0 / den8

    damp_6 = s6 * den6_inv
    damp_8 = s8 * r4r2_ij * den8_inv
    damp_sum = damp_6 + damp_8

    return damp_sum, r4r2_ij, r6, r4, den6_inv, den8_inv


@wp.func
def _dispersion_energy_force(
    c6_ij: wp.float32,
    r: wp.float32,
    r_hat: wp.vec3f,
    damp_sum: wp.float32,
    r4r2_ij: wp.float32,
    r6: wp.float32,
    r4: wp.float32,
    den6_inv: wp.float32,
    den8_inv: wp.float32,
    s6: wp.float32,
    s8: wp.float32,
    s5_smoothing_on: wp.float32,
    s5_smoothing_off: wp.float32,
    inv_w: wp.float32,
) -> tuple[wp.float32, wp.vec3f]:
    """
    Compute dispersion energy and direct force with S5 switching.

    Returns
    -------
    e_ij_sw : float32
        Smoothed energy
    F_direct : vec3f
        Direct force vector
    """
    e_ij = -c6_ij * damp_sum

    r5 = r4 * r
    r7 = r6 * r
    dD6_dr = -6.0 * s6 * r5 * den6_inv * den6_inv  # NOSONAR (S125) "math formula"
    dD8_dr = -8.0 * s8 * r4r2_ij * r7 * den8_inv * den8_inv  # NOSONAR (S125)
    dE_dr_direct = -c6_ij * (dD6_dr + dD8_dr)  # NOSONAR (S125) "math formula"

    sw, dsw_dr = _s5_switch(r, s5_smoothing_on, s5_smoothing_off, inv_w)
    e_ij_sw = e_ij * sw
    dE_dr_direct_sw = sw * dE_dr_direct + e_ij * dsw_dr  # NOSONAR (S125)

    F_direct = dE_dr_direct_sw * r_hat  # NOSONAR (S125) "math formula"

    return e_ij_sw, F_direct


@wp.func
def _unit_shift_to_cartesian(
    unit_shift: wp.vec3i,
    cell_mat: Any,
) -> Any:
    """Convert integer unit cell shift to Cartesian coordinates."""
    unit_shift_float = type(cell_mat[0])(
        type(cell_mat[0, 0])(unit_shift[0]),
        type(cell_mat[0, 0])(unit_shift[1]),
        type(cell_mat[0, 0])(unit_shift[2]),
    )
    return unit_shift_float * cell_mat


# ==============================================================================
# Kernels
# ==============================================================================


@wp.kernel(enable_backward=False)
def _compute_cartesian_shifts_matrix(
    cell: wp.array(dtype=Any),
    unit_shifts: wp.array2d(dtype=wp.vec3i),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    batch_idx: wp.array(dtype=wp.int32),
    fill_value: wp.int32,
    cartesian_shifts: wp.array2d(dtype=Any),
):
    """
    Convert unit cell shifts to Cartesian coordinates for periodic boundaries.

    For each neighbor in the neighbor matrix, this kernel computes the Cartesian
    shift vector that should be applied to atom j's position to obtain its
    periodic image closest to atom i.

    Parameters
    ----------
    cell : wp.array3d(dtype=float32)
        Unit cell lattice vectors [num_systems, 3, 3]. Convention: cell[s, i, :]
        is the i-th lattice vector for system s (row vectors). Units should match
        position coordinates.
    unit_shifts : wp.array2d(dtype=vec3i)
        Integer unit cell shifts [num_atoms, max_neighbors] as vec3i
    neighbor_matrix : wp.array2d(dtype=int32)
        Neighbor indices [num_atoms, max_neighbors]. See module docstring
        for more details.
    batch_idx : wp.array(dtype=int32)
        System index [num_atoms] for each atom
    fill_value : int32
        Value indicating padding in neighbor_matrix (typically num_atoms)
    cartesian_shifts : wp.array2d(dtype=vec3f)
        Output: Cartesian shift vectors [num_atoms, max_neighbors] as vec3 in same
        units as cell vectors

    Notes
    -----
    The Cartesian shift is computed as:

    .. math::

        \\mathbf{s} = n_a \\mathbf{a} + n_b \\mathbf{b} + n_c \\mathbf{c}

    where :math:`\\mathbf{a}, \\mathbf{b}, \\mathbf{c}` are lattice vectors
    and :math:`n_a, n_b, n_c` are integer shifts. The system ID is obtained
    from atom i's batch index.

    Launch with dim=(num_atoms, max_neighbors) (one thread per atom-neighbor pair).

    See Also
    --------
    :func:`_cn_kernel_matrix` : Pass 1 - Uses computed Cartesian shifts for PBC (neighbor matrix)
    :func:`_cn_kernel` : Pass 1 - Uses computed Cartesian shifts for PBC (neighbor list)
    :func:`_direct_forces_and_dE_dCN_kernel_matrix` : Pass 2 - Uses computed
    Cartesian shifts for PBC (neighbor matrix)
    :func:`_direct_forces_and_dE_dCN_kernel` : Pass 2 - Uses computed
    Cartesian shifts for PBC (neighbor list)
    :func:`_cn_forces_contrib_kernel_matrix` : Pass 3 - Uses computed Cartesian shifts for PBC (neighbor matrix)
    :func:`_cn_forces_contrib_kernel` : Pass 3 - Uses computed Cartesian shifts for PBC (neighbor list)
    :func:`dftd3` : High-level wrapper that orchestrates all passes
    """
    atom_i, neighbor_idx = wp.tid()
    max_neighbors = neighbor_matrix.shape[1]

    if neighbor_idx >= max_neighbors:
        return

    atom_j = neighbor_matrix[atom_i, neighbor_idx]
    if atom_j >= fill_value:
        return

    system_id = batch_idx[atom_i]

    cell_mat = cell[system_id]
    unit_shift = unit_shifts[atom_i, neighbor_idx]
    cartesian_shifts[atom_i, neighbor_idx] = _unit_shift_to_cartesian(
        unit_shift, cell_mat
    )


@wp.kernel(enable_backward=False)
def _cn_kernel_matrix(
    positions: wp.array(dtype=Any),
    numbers: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    cartesian_shifts: wp.array2d(dtype=Any),
    covalent_radii: wp.array(dtype=wp.float32),
    k1: wp.float32,
    fill_value: wp.int32,
    periodic: bool,
    coord_num: wp.array(dtype=wp.float32),
):
    """
    Compute coordination numbers using geometric counting function.

    This kernel computes the coordination number (CN) for each atom based on
    a smooth counting function that depends on interatomic distances and
    covalent radii. Supports periodic boundary conditions via Cartesian shifts.

    Algorithm
    ---------
    For each atom i, iterate over its neighbors (neighbor matrix format) and accumulate:

    .. math::

        \\text{CN}_i = \\sum_{j \\in \\text{neighbors}(i)} f(r_{ij})

        f(r) = \\frac{1}{1 + \\exp\\left[k_1\\left(\\frac{r_{\\text{cov}}}{r} - 1\\right)\\right]}

    where :math:`r_{\\text{cov}} = r_{\\text{cov}}[Z_i] + r_{\\text{cov}}[Z_j]`.
    The counting function smoothly transitions from 1 (bonded) to 0 (non-bonded).

    Parameters
    ----------
    positions : wp.array(dtype=vec3f)
        Atomic coordinates [num_atoms]
    numbers : wp.array(dtype=int32)
        Atomic numbers [num_atoms]
    neighbor_matrix : wp.array2d(dtype=int32)
        Neighbor indices [num_atoms, max_neighbors]. See module docstring
        for more details.
    cartesian_shifts : wp.array2d(dtype=vec3f)
        Cartesian shifts [num_atoms, max_neighbors] as vec3 for PBC (ignored if periodic=False), in same units as positions
    covalent_radii : wp.array(dtype=float32)
        Covalent radii [max_Z+1] indexed by atomic number, in same units as positions
    k1 : float32
        Steepness parameter for counting function (typically 16.0 1/Bohr)
    fill_value : int32
        Value indicating padding in neighbor_matrix (typically num_atoms)
    periodic : bool
        If True, apply PBC using cartesian_shifts; if False, non-periodic
    coord_num : wp.array(dtype=float32)
        Output: coordination numbers [num_atoms] (dimensionless)

    Notes
    -----
    - Launch with dim=num_atoms (one thread per atom)
    - Each thread iterates over all neighbors and accumulates CN in a local register
    - Padding atoms indicated by numbers[i] == 0 are skipped
    - Neighbor entries with j >= fill_value are padding and are skipped

    See Also
    --------
    :func:`_compute_cartesian_shifts_matrix` : Pass 0 - Computes Cartesian shifts for PBC
    :func:`_direct_forces_and_dE_dCN_kernel_matrix` : Pass 2 - Uses coordination numbers
    computed here
    :func:`dftd3` : High-level wrapper that orchestrates all passes
    """
    atom_i = wp.tid()
    if atom_i >= numbers.shape[0]:
        return
    # skip padding atoms
    if numbers[atom_i] == 0:
        return

    max_neighbors = neighbor_matrix.shape[1]
    pos_i = positions[atom_i]
    rcov_i = covalent_radii[numbers[atom_i]]

    # Accumulate coordination number in local register
    cn_acc = wp.float32(0.0)

    for neighbor_idx in range(max_neighbors):
        atom_j = neighbor_matrix[atom_i, neighbor_idx]
        if atom_j >= fill_value:
            continue
        # skip padding
        if numbers[atom_j] == 0:
            continue

        # Compute distance with optional PBC shift
        r, r_inv, r_hat, r_ij = _compute_distance_vector_pbc(
            pos_i,
            positions[atom_j],
            cartesian_shifts[atom_i, neighbor_idx],
            periodic,
            False,
        )
        if r < 1e-12:
            continue

        # Compute coordination number contribution
        f_cn, dCN_dr = _cn_counting(
            r_inv, rcov_i, covalent_radii[numbers[atom_j]], k1, False
        )
        cn_acc += f_cn

    # Write final coordination number once
    coord_num[atom_i] = cn_acc


@wp.kernel(enable_backward=False)
def _direct_forces_and_dE_dCN_kernel_matrix(  # NOSONAR (S1542) "math formula"
    positions: wp.array(dtype=Any),
    numbers: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    cartesian_shifts: wp.array2d(dtype=Any),
    coord_num: wp.array(dtype=wp.float32),
    r4r2: wp.array(dtype=wp.float32),
    c6_reference: wp.array4d(dtype=wp.float32),
    coord_num_ref: wp.array4d(dtype=wp.float32),
    k3: wp.float32,
    a1: wp.float32,
    a2: wp.float32,
    s6: wp.float32,
    s8: wp.float32,
    s5_smoothing_on: wp.float32,
    s5_smoothing_off: wp.float32,
    inv_w: wp.float32,
    fill_value: wp.int32,
    periodic: bool,
    batch_idx: wp.array(dtype=wp.int32),
    compute_virial: bool,
    dE_dCN: wp.array(dtype=wp.float32),  # NOSONAR (S125) "math formula"
    forces: wp.array(dtype=wp.vec3f),
    energy: wp.array(dtype=wp.float32),
    virial: wp.array(dtype=Any),
):
    """
    Pass 2: Compute direct forces, energy, and accumulate dE/dCN per atom.

    Computes dispersion energy and forces at constant CN, and accumulates
    dE/dCN contributions for each atom for use in Pass 3.

    Parameters
    ----------
    positions : wp.array(dtype=vec3f)
        Atomic coordinates [num_atoms]
    numbers : wp.array(dtype=int32)
        Atomic numbers [num_atoms]
    neighbor_matrix : wp.array2d(dtype=int32)
        Neighbor indices [num_atoms, max_neighbors]. See module docstring
        for more details.
    cartesian_shifts : wp.array2d(dtype=vec3f)
        Cartesian shifts [num_atoms, max_neighbors] as vec3 for PBC, in same units as positions
    coord_num : wp.array(dtype=float32)
        Coordination numbers [num_atoms] from Pass 1 (dimensionless)
    r4r2 : wp.array(dtype=float32)
        <r⁴>/<r²> expectation values [max_Z+1] (dimensionless)
    c6_reference : wp.array4d(dtype=float32)
        C6 reference [max_Z+1, max_Z+1, 5, 5] in energy x distance^6 units
    coord_num_ref : wp.array4d(dtype=float32)
        CN reference grid [max_Z+1, max_Z+1, 5, 5] (dimensionless)
    k3 : float32
        CN interpolation width (typically -4.0, dimensionless)
    a1, a2 : float32
        Becke-Johnson damping parameters (a1 dimensionless, a2 in distance units)
    s6, s8 : float32
        Scaling factors for C6 and C8 terms (dimensionless)
    s5_smoothing_on, s5_smoothing_off : float32
        S5 switching radii in same units as positions
    inv_w : float32
        Precomputed 1/(s5_off - s5_on) in inverse distance units
    fill_value : int32
        Value indicating padding in neighbor_matrix (typically num_atoms)
    periodic : bool
        If True, apply PBC using cartesian_shifts
    batch_idx : wp.array(dtype=int32)
        System index [num_atoms]
    dE_dCN : wp.array(dtype=float32)
        Output: accumulated dE/dCN [num_atoms] in energy units
    forces : wp.array(dtype=vec3f)
        Output: direct forces [num_atoms] in energy/distance units
    energy : wp.array(dtype=float32)
        Output: system energies [num_systems] in energy units

    Notes
    -----
    - Launch with dim=num_atoms (one thread per atom)
    - Each thread iterates over all neighbors and accumulates results in local registers
    - Direct forces are F = :math:`-(\\partial E/\\partial r)|_\text{CN}`, without chain rule term
    - dE_dCN[i] = :math:`\\sum_j \\partial E_{ij}/\\partial \text{CN}_i` accumulated over all pairs containing atom i
    - Neighbor entries with j >= fill_value are padding and are skipped

    See Also
    --------
    :func:`_compute_cartesian_shifts_matrix` : Pass 0 - Computes Cartesian shifts for PBC
    :func:`_cn_kernel_matrix` : Pass 1 - Computes coordination numbers used here
    :func:`_cn_forces_contrib_kernel_matrix` : Pass 3 - Uses dE/dCN values accumulated here
    :func:`_c6ab_interpolate` : Called to interpolate C6 coefficients
    :func:`_s5_switch` : Called for cutoff smoothing
    :func:`dftd3` : High-level wrapper that orchestrates all passes
    """
    atom_i = wp.tid()
    if atom_i >= numbers.shape[0]:
        return
    # skip padding atoms
    if numbers[atom_i] == 0:
        return

    max_neighbors = neighbor_matrix.shape[1]
    pos_i = positions[atom_i]
    cn_i = coord_num[atom_i]
    z_i = numbers[atom_i]
    r4r2_i = r4r2[z_i]

    # Accumulate in local registers (using float64 for better precision)
    F_acc = wp.vec3d()  # NOSONAR (S117) "math formula"
    dE_dCN_acc = wp.float32(0.0)  # NOSONAR (S117) "math formula"
    energy_acc = wp.float64(0.0)

    # Initialize virial accumulator
    if compute_virial:
        virial_acc = wp.mat33d()

    for neighbor_idx in range(max_neighbors):
        atom_j = neighbor_matrix[atom_i, neighbor_idx]
        if atom_j >= fill_value:
            continue
        # skip padding atoms
        if numbers[atom_j] == 0:
            continue

        # Geometry
        r, r_inv, r_hat, r_ij = _compute_distance_vector_pbc(
            pos_i,
            positions[atom_j],
            cartesian_shifts[atom_i, neighbor_idx],
            periodic,
            True,
        )
        if r < 1e-12:
            continue

        cn_j = coord_num[atom_j]
        z_j = numbers[atom_j]

        # C6 interpolation
        c6ab_mat = c6_reference[z_i, z_j]
        cnref_i_mat = coord_num_ref[z_i, z_j]
        cnref_j_mat = coord_num_ref[z_j, z_i]

        c6_ij, dC6_dCNi, dC6_dCNj = _c6ab_interpolate(  # NOSONAR (S125) "math formula"
            cn_i, cn_j, c6ab_mat, cnref_i_mat, cnref_j_mat, k3
        )
        if c6_ij < 1e-12:
            continue

        # BJ damping
        damp_sum, r4r2_ij, r6, r4, den6_inv, den8_inv = _bj_damping(
            r, r4r2_i, r4r2[z_j], a1, a2, s6, s8
        )

        # Energy and direct force
        e_ij_sw, F_direct = _dispersion_energy_force(
            c6_ij,
            r,
            r_hat,
            damp_sum,
            r4r2_ij,
            r6,
            r4,
            den6_inv,
            den8_inv,
            s6,
            s8,
            s5_smoothing_on,
            s5_smoothing_off,
            inv_w,
        )

        # Accumulate in registers
        F_acc += wp.vec3d(F_direct)  # NOSONAR (S117) "math formula"
        energy_acc += wp.float64(e_ij_sw)
        dE_dCN_acc += -damp_sum * dC6_dCNi  # NOSONAR (S117) "math formula"

        # Accumulate virial if requested
        if compute_virial:
            virial_acc += wp.mat33d(wp.outer(F_direct, r_ij))

    # Write final results once (atomic only for shared batch array)
    # Convert from float64 accumulation to float32 output
    forces[atom_i] = wp.vec3f(F_acc)
    dE_dCN[atom_i] = dE_dCN_acc
    wp.atomic_add(energy, batch_idx[atom_i], 0.5 * wp.float32(energy_acc))

    # Add virial contribution with -0.5 scaling for correct sign and double counting
    if compute_virial:
        wp.atomic_add(virial, batch_idx[atom_i], -0.5 * wp.mat33f(virial_acc))


@wp.kernel(enable_backward=False)
def _cn_forces_contrib_kernel_matrix(
    positions: wp.array(dtype=Any),
    numbers: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    cartesian_shifts: wp.array2d(dtype=Any),
    covalent_radii: wp.array(dtype=wp.float32),
    dE_dCN: wp.array(dtype=wp.float32),  # NOSONAR (S125) "math formula"
    k1: wp.float32,
    fill_value: wp.int32,
    periodic: bool,
    batch_idx: wp.array(dtype=wp.int32),
    compute_virial: bool,
    forces: wp.array(dtype=wp.vec3f),
    virial: wp.array(dtype=Any),
):
    """
    Pass 3: Add CN-dependent force contribution.

    Adds the CN-dependent term to forces computed in Pass 2. Computes
    distances and CN derivatives without repeating C6 interpolation and
    damping calculations.

    Parameters
    ----------
    positions : wp.array(dtype=vec3f)
        Atomic coordinates [num_atoms] in consistent distance units
    numbers : wp.array(dtype=int32)
        Atomic numbers [num_atoms]
    neighbor_matrix : wp.array2d(dtype=int32)
        Neighbor indices [num_atoms, max_neighbors]. See module docstring
        for more details.
    cartesian_shifts : wp.array2d(dtype=vec3f)
        Cartesian shifts [num_atoms, max_neighbors] as vec3 for PBC, in same units as positions
    covalent_radii : wp.array(dtype=float32)
        Covalent radii [max_Z+1] in same units as positions
    dE_dCN : wp.array(dtype=float32)
        Precomputed dE/dCN [num_atoms] from Pass 2 in energy units
    k1 : float32
        CN counting steepness in inverse distance units (typically 16.0 1/Bohr)
    fill_value : int32
        Value indicating padding in neighbor_matrix (typically num_atoms)
    periodic : bool
        If True, apply PBC using cartesian_shifts
    forces : wp.array(dtype=vec3f)
        Input/Output: add chain term to direct forces [num_atoms] in energy/distance units

    Notes
    -----
    - Launch with dim=num_atoms (one thread per atom)
    - Each thread iterates over all neighbors and accumulates results in local registers
    - Skips C6 interpolation and damping calculations
    - Uses precomputed dE_dCN[i] = :math:`\\sum_k \\partial E_{ik}/\\partial \text{CN}_i` from all pairs
    - Neighbor entries with j >= fill_value are padding and are skipped

    See Also
    --------
    :func:`_compute_cartesian_shifts_matrix` : Pass 0 - Computes Cartesian shifts for PBC
    :func:`_direct_forces_and_dE_dCN_kernel_matrix` : Pass 2 - Computes dE/dCN values used here
    :func:`dftd3` : High-level wrapper that orchestrates all passes
    """
    atom_i = wp.tid()
    if atom_i >= numbers.shape[0]:
        return
    # skip padding atoms
    if numbers[atom_i] == 0:
        return

    max_neighbors = neighbor_matrix.shape[1]
    dE_dCN_i = dE_dCN[atom_i]  # NOSONAR (S125) "math formula"
    pos_i = positions[atom_i]
    rcov_i = covalent_radii[numbers[atom_i]]

    # Accumulate force in local register (using float64 for better precision)
    F_chain_acc = wp.vec3d()  # NOSONAR (S117) "math formula"

    # Initialize virial accumulator
    if compute_virial:
        virial_chain_acc = wp.mat33d()

    for neighbor_idx in range(max_neighbors):
        atom_j = neighbor_matrix[atom_i, neighbor_idx]
        if atom_j >= fill_value:
            continue
        if numbers[atom_j] == 0:
            continue

        # Distance
        r, r_inv, r_hat, r_ij = _compute_distance_vector_pbc(
            pos_i,
            positions[atom_j],
            cartesian_shifts[atom_i, neighbor_idx],
            periodic,
            True,
        )
        if r < 1e-12:
            continue

        # CN derivative
        f_cn, dCN_dr = _cn_counting(
            r_inv, rcov_i, covalent_radii[numbers[atom_j]], k1, True
        )

        # CN-dependent force contribution
        dE_dCN_j = dE_dCN[atom_j]  # NOSONAR (S125) "math formula"
        dE_dr_chain = (dE_dCN_i + dE_dCN_j) * dCN_dr  # NOSONAR (S125) "math formula"
        F_chain = dE_dr_chain * r_hat  # NOSONAR (S125) "math formula"

        F_chain_acc += wp.vec3d(F_chain)

        # Accumulate virial if requested
        if compute_virial:
            virial_chain_acc += wp.mat33d(wp.outer(F_chain, r_ij))

    # Add accumulated force to existing forces (direct read-modify-write)
    # Convert from float64 accumulation to float32 output
    forces[atom_i] = forces[atom_i] + wp.vec3f(F_chain_acc)

    # Add virial contribution with -0.5 scaling for correct sign and double counting
    if compute_virial:
        wp.atomic_add(virial, batch_idx[atom_i], -0.5 * wp.mat33f(virial_chain_acc))


# ==============================================================================
# Neighbor List Kernels
# ==============================================================================


@wp.kernel(enable_backward=False)
def _compute_cartesian_shifts(
    cell: wp.array(dtype=Any),
    unit_shifts: wp.array(dtype=wp.vec3i),
    neighbor_ptr: wp.array(dtype=wp.int32),
    batch_idx: wp.array(dtype=wp.int32),
    cartesian_shifts: wp.array(dtype=Any),
):
    """
    Convert unit cell shifts to Cartesian coordinates for CSR neighbor lists.

    For each edge in the CSR neighbor list, this kernel computes the Cartesian
    shift vector that should be applied to the destination atom's position.

    Parameters
    ----------
    cell : wp.array(dtype=mat33)
        Unit cell lattice vectors [num_systems, 3, 3]. Convention: cell[s, i, :]
        is the i-th lattice vector for system s (row vectors). Units should match
        position coordinates.
    unit_shifts : wp.array(dtype=vec3i)
        Integer unit cell shifts [num_edges] as vec3i
    neighbor_ptr : wp.array(dtype=int32)
        CSR row pointers [num_atoms+1]
    batch_idx : wp.array(dtype=int32)
        System index [num_atoms] for each atom
    cartesian_shifts : wp.array(dtype=vec3)
        Output: Cartesian shift vectors [num_edges] as vec3 in same units as cell vectors

    Notes
    -----
    Launch with dim=num_atoms (one thread per atom). Each thread processes all edges
    for that atom using the CSR pointers.

    See Also
    --------
    :func:`_cn_kernel` : Uses computed Cartesian shifts for PBC
    :func:`_direct_forces_and_dE_dCN_kernel` : Uses computed Cartesian shifts for PBC
    :func:`_cn_forces_contrib_kernel` : Uses computed Cartesian shifts for PBC
    """
    atom_i = wp.tid()

    # Get number of atoms from batch_idx size
    if atom_i >= batch_idx.shape[0]:
        return

    system_id = batch_idx[atom_i]
    cell_mat = cell[system_id]

    # Get range of edges for this atom
    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]

    # Convert all unit shifts for this atom's neighbors to Cartesian
    for edge_idx in range(j_range_start, j_range_end):
        unit_shift = unit_shifts[edge_idx]
        cartesian_shifts[edge_idx] = _unit_shift_to_cartesian(unit_shift, cell_mat)


@wp.kernel(enable_backward=False)
def _cn_kernel(
    positions: wp.array(dtype=Any),
    numbers: wp.array(dtype=wp.int32),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    cartesian_shifts: wp.array(dtype=Any),
    covalent_radii: wp.array(dtype=wp.float32),
    k1: wp.float32,
    periodic: bool,
    coord_num: wp.array(dtype=wp.float32),
):
    """
    Compute coordination numbers using CSR neighbor list format.

    Parameters
    ----------
    positions : wp.array(dtype=vec3)
        Atomic coordinates [num_atoms]
    numbers : wp.array(dtype=int32)
        Atomic numbers [num_atoms]
    idx_j : wp.array(dtype=int32)
        Destination atom indices [num_edges] in CSR format
    neighbor_ptr : wp.array(dtype=int32)
        CSR row pointers [num_atoms+1]
    cartesian_shifts : wp.array(dtype=vec3)
        Cartesian shifts [num_edges] as vec3 for PBC
    covalent_radii : wp.array(dtype=float32)
        Covalent radii [max_Z+1] indexed by atomic number
    k1 : float32
        Steepness parameter for counting function
    periodic : bool
        If True, apply PBC using cartesian_shifts
    coord_num : wp.array(dtype=float32)
        Output: coordination numbers [num_atoms]

    Notes
    -----
    Launch with dim=num_atoms (one thread per atom).
    """
    atom_i = wp.tid()
    if atom_i >= numbers.shape[0]:
        return

    # skip padding atoms
    if numbers[atom_i] == 0:
        return

    pos_i = positions[atom_i]
    rcov_i = covalent_radii[numbers[atom_i]]

    # Accumulate coordination number in local register
    cn_acc = wp.float32(0.0)

    # Iterate over neighbors using CSR pointers
    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_range_start, j_range_end):
        atom_j = idx_j[edge_idx]

        # skip padding atoms
        if numbers[atom_j] == 0:
            continue

        # Compute distance with optional PBC shift
        r, r_inv, r_hat, r_ij = _compute_distance_vector_pbc(
            pos_i, positions[atom_j], cartesian_shifts[edge_idx], periodic, False
        )
        if r < 1e-12:
            continue

        # Compute coordination number contribution
        f_cn, dCN_dr = _cn_counting(
            r_inv, rcov_i, covalent_radii[numbers[atom_j]], k1, False
        )
        cn_acc += f_cn

    # Write final coordination number once
    coord_num[atom_i] = cn_acc


@wp.kernel(enable_backward=False)
def _direct_forces_and_dE_dCN_kernel(  # NOSONAR (S1542) "math formula"
    positions: wp.array(dtype=Any),
    numbers: wp.array(dtype=wp.int32),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    cartesian_shifts: wp.array(dtype=Any),
    coord_num: wp.array(dtype=wp.float32),
    r4r2: wp.array(dtype=wp.float32),
    c6_reference: wp.array4d(dtype=wp.float32),
    coord_num_ref: wp.array4d(dtype=wp.float32),
    k3: wp.float32,
    a1: wp.float32,
    a2: wp.float32,
    s6: wp.float32,
    s8: wp.float32,
    s5_smoothing_on: wp.float32,
    s5_smoothing_off: wp.float32,
    inv_w: wp.float32,
    periodic: bool,
    batch_idx: wp.array(dtype=wp.int32),
    compute_virial: bool,
    dE_dCN: wp.array(dtype=wp.float32),  # NOSONAR (S125) "math formula"
    forces: wp.array(dtype=wp.vec3f),
    energy: wp.array(dtype=wp.float32),
    virial: wp.array(dtype=Any),
):
    """
    Pass 2: Compute direct forces, energy, and accumulate dE/dCN using
    CSR neighbor list.

    Notes
    -----
    Launch with dim=num_atoms (one thread per atom).
    """
    atom_i = wp.tid()
    if atom_i >= numbers.shape[0]:
        return

    # skip padding atoms
    if numbers[atom_i] == 0:
        return

    pos_i = positions[atom_i]
    cn_i = coord_num[atom_i]
    z_i = numbers[atom_i]
    r4r2_i = r4r2[z_i]

    # Accumulate in local registers (using float64 for better precision)
    F_acc = wp.vec3d()  # NOSONAR (S117) "math formula"
    dE_dCN_acc = wp.float32(0.0)  # NOSONAR (S117) "math formula"
    energy_acc = wp.float64(0.0)

    # Initialize virial accumulator
    if compute_virial:
        virial_acc = wp.mat33d()

    # Iterate over neighbors using CSR pointers
    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_range_start, j_range_end):
        atom_j = idx_j[edge_idx]

        # skip padding atoms
        if numbers[atom_j] == 0:
            continue

        # Geometry
        r, r_inv, r_hat, r_ij = _compute_distance_vector_pbc(
            pos_i, positions[atom_j], cartesian_shifts[edge_idx], periodic, True
        )
        if r < 1e-12:
            continue

        cn_j = coord_num[atom_j]
        z_j = numbers[atom_j]

        # C6 interpolation
        c6ab_mat = c6_reference[z_i, z_j]
        cnref_i_mat = coord_num_ref[z_i, z_j]
        cnref_j_mat = coord_num_ref[z_j, z_i]

        c6_ij, dC6_dCNi, dC6_dCNj = _c6ab_interpolate(  # NOSONAR (S125) "math formula"
            cn_i, cn_j, c6ab_mat, cnref_i_mat, cnref_j_mat, k3
        )
        if c6_ij < 1e-12:
            continue

        # BJ damping
        damp_sum, r4r2_ij, r6, r4, den6_inv, den8_inv = _bj_damping(
            r, r4r2_i, r4r2[z_j], a1, a2, s6, s8
        )

        # Energy and direct force
        e_ij_sw, F_direct = _dispersion_energy_force(
            c6_ij,
            r,
            r_hat,
            damp_sum,
            r4r2_ij,
            r6,
            r4,
            den6_inv,
            den8_inv,
            s6,
            s8,
            s5_smoothing_on,
            s5_smoothing_off,
            inv_w,
        )

        # Accumulate in registers
        F_acc += wp.vec3d(F_direct)
        energy_acc += wp.float64(e_ij_sw)
        dE_dCN_acc += -damp_sum * dC6_dCNi  # NOSONAR (S117) "math formula"

        # Accumulate virial if requested
        if compute_virial:
            virial_acc += wp.mat33d(wp.outer(F_direct, r_ij))

    # Write final results once (atomic only for shared batch array)
    # Convert from float64 accumulation to float32 output
    forces[atom_i] = wp.vec3f(F_acc)
    dE_dCN[atom_i] = wp.float32(dE_dCN_acc)
    wp.atomic_add(energy, batch_idx[atom_i], 0.5 * wp.float32(energy_acc))

    # Add virial contribution with -0.5 scaling for correct sign and double counting
    if compute_virial:
        wp.atomic_add(virial, batch_idx[atom_i], -0.5 * wp.mat33f(virial_acc))


@wp.kernel(enable_backward=False)
def _cn_forces_contrib_kernel(
    positions: wp.array(dtype=Any),
    numbers: wp.array(dtype=wp.int32),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    cartesian_shifts: wp.array(dtype=Any),
    covalent_radii: wp.array(dtype=wp.float32),
    dE_dCN: wp.array(dtype=wp.float32),  # NOSONAR (S125) "math formula"
    k1: wp.float32,
    periodic: bool,
    batch_idx: wp.array(dtype=wp.int32),
    compute_virial: bool,
    forces: wp.array(dtype=wp.vec3f),
    virial: wp.array(dtype=Any),
):
    """
    Pass 3: Add CN-dependent force contribution using CSR neighbor list.

    Notes
    -----
    Launch with dim=num_atoms (one thread per atom).
    """
    atom_i = wp.tid()
    if atom_i >= numbers.shape[0]:
        return

    # skip padding atoms
    if numbers[atom_i] == 0:
        return

    dE_dCN_i = dE_dCN[atom_i]  # NOSONAR (S125) "math formula"
    pos_i = positions[atom_i]
    rcov_i = covalent_radii[numbers[atom_i]]

    # Accumulate force in local register (using float64 for better precision)
    F_chain_acc = wp.vec3d()  # NOSONAR (S117) "math formula"

    # Initialize virial accumulator
    if compute_virial:
        virial_chain_acc = wp.mat33d()

    # Iterate over neighbors using CSR pointers
    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_range_start, j_range_end):
        atom_j = idx_j[edge_idx]

        if numbers[atom_j] == 0:
            continue

        # Distance
        r, r_inv, r_hat, r_ij = _compute_distance_vector_pbc(
            pos_i, positions[atom_j], cartesian_shifts[edge_idx], periodic, True
        )
        if r < 1e-12:
            continue

        # CN derivative
        f_cn, dCN_dr = _cn_counting(
            r_inv, rcov_i, covalent_radii[numbers[atom_j]], k1, True
        )

        # CN-dependent force contribution
        dE_dCN_j = dE_dCN[atom_j]  # NOSONAR (S125) "math formula"
        dE_dr_chain = (dE_dCN_i + dE_dCN_j) * dCN_dr  # NOSONAR (S125) "math formula"
        F_chain = dE_dr_chain * r_hat  # NOSONAR (S125) "math formula"

        F_chain_acc += wp.vec3d(F_chain)

        # Accumulate virial if requested
        if compute_virial:
            virial_chain_acc += wp.mat33d(wp.outer(F_chain, r_ij))

    # Add accumulated force to existing forces (direct read-modify-write)
    # Convert from float64 accumulation to float32 output
    forces[atom_i] = forces[atom_i] + wp.vec3f(F_chain_acc)

    # Add virial contribution with -0.5 scaling for correct sign and double counting
    if compute_virial:
        wp.atomic_add(virial, batch_idx[atom_i], -0.5 * wp.mat33f(virial_chain_acc))


# ==============================================================================
# Kernel Overload Registration
# ==============================================================================

# Type constants for overload generation
T = [wp.float32, wp.float64]
V = [wp.vec3f, wp.vec3d]
M = [wp.mat33f, wp.mat33d]

# Overload dictionaries keyed by scalar type
# Neighbor matrix format (dense)
_compute_cartesian_shifts_matrix_overload = {}
_cn_kernel_matrix_overload = {}
_direct_forces_and_dE_dCN_kernel_matrix_overload = {}
_cn_forces_contrib_kernel_matrix_overload = {}

# Neighbor list kernel overload dictionaries (CSR format) - default naming convention
_compute_cartesian_shifts_overload = {}
_cn_kernel_overload = {}
_direct_forces_and_dE_dCN_kernel_overload = {}
_cn_forces_contrib_kernel_overload = {}

# Register overloads for all kernel variants
for t, v, m in zip(T, V, M):
    # Neighbor matrix format (dense)
    _compute_cartesian_shifts_matrix_overload[t] = wp.overload(
        _compute_cartesian_shifts_matrix,
        [
            wp.array(dtype=m),
            wp.array2d(dtype=wp.vec3i),
            wp.array2d(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.int32,
            wp.array2d(dtype=v),
        ],
    )
    _cn_kernel_matrix_overload[t] = wp.overload(
        _cn_kernel_matrix,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=v),
            wp.array(dtype=wp.float32),
            wp.float32,
            wp.int32,
            wp.bool,
            wp.array(dtype=wp.float32),
        ],
    )
    _direct_forces_and_dE_dCN_kernel_matrix_overload[t] = wp.overload(
        _direct_forces_and_dE_dCN_kernel_matrix,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=v),
            wp.array(dtype=wp.float32),
            wp.array(dtype=wp.float32),
            wp.array4d(dtype=wp.float32),
            wp.array4d(dtype=wp.float32),
            wp.float32,
            wp.float32,
            wp.float32,
            wp.float32,
            wp.float32,
            wp.float32,
            wp.float32,
            wp.float32,
            wp.int32,
            wp.bool,
            wp.array(dtype=wp.int32),
            wp.bool,
            wp.array(dtype=wp.float32),
            wp.array(dtype=wp.vec3f),
            wp.array(dtype=wp.float32),
            wp.array(dtype=wp.mat33f),
        ],
    )
    _cn_forces_contrib_kernel_matrix_overload[t] = wp.overload(
        _cn_forces_contrib_kernel_matrix,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=v),
            wp.array(dtype=wp.float32),
            wp.array(dtype=wp.float32),
            wp.float32,
            wp.int32,
            wp.bool,
            wp.array(dtype=wp.int32),
            wp.bool,
            wp.array(dtype=wp.vec3f),
            wp.array(dtype=wp.mat33f),
        ],
    )
    # Neighbor list kernel overloads (CSR format) - default naming convention
    _compute_cartesian_shifts_overload[t] = wp.overload(
        _compute_cartesian_shifts,
        [
            wp.array(dtype=m),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
        ],
    )
    _cn_kernel_overload[t] = wp.overload(
        _cn_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
            wp.array(dtype=wp.float32),
            wp.float32,
            wp.bool,
            wp.array(dtype=wp.float32),
        ],
    )
    _direct_forces_and_dE_dCN_kernel_overload[t] = wp.overload(
        _direct_forces_and_dE_dCN_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
            wp.array(dtype=wp.float32),
            wp.array(dtype=wp.float32),
            wp.array4d(dtype=wp.float32),
            wp.array4d(dtype=wp.float32),
            wp.float32,
            wp.float32,
            wp.float32,
            wp.float32,
            wp.float32,
            wp.float32,
            wp.float32,
            wp.float32,
            wp.bool,
            wp.array(dtype=wp.int32),
            wp.bool,
            wp.array(dtype=wp.float32),
            wp.array(dtype=wp.vec3f),
            wp.array(dtype=wp.float32),
            wp.array(dtype=wp.mat33f),
        ],
    )
    _cn_forces_contrib_kernel_overload[t] = wp.overload(
        _cn_forces_contrib_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
            wp.array(dtype=wp.float32),
            wp.array(dtype=wp.float32),
            wp.float32,
            wp.bool,
            wp.array(dtype=wp.int32),
            wp.bool,
            wp.array(dtype=wp.vec3f),
            wp.array(dtype=wp.mat33f),
        ],
    )


# ==============================================================================
# Warp Launchers (Framework-Agnostic)
# ==============================================================================


def dftd3_matrix(
    positions: wp.array,
    numbers: wp.array,
    neighbor_matrix: wp.array,
    covalent_radii: wp.array,
    r4r2: wp.array,
    c6_reference: wp.array,
    coord_num_ref: wp.array,
    a1: float,
    a2: float,
    s8: float,
    coord_num: wp.array,
    forces: wp.array,
    energy: wp.array,
    virial: wp.array,
    batch_idx: wp.array,
    cartesian_shifts: wp.array,
    dE_dCN: wp.array,
    wp_dtype: type,
    device: str,
    k1: float = 16.0,
    k3: float = -4.0,
    s6: float = 1.0,
    s5_smoothing_on: float = 0.0,
    s5_smoothing_off: float = 0.0,
    fill_value: int | None = None,
) -> None:
    """
    Launch DFT-D3(BJ) dispersion calculation using neighbor matrix format (non-periodic).

    This is a framework-agnostic warp launcher for non-periodic (non-PBC) systems
    that accepts warp arrays directly and orchestrates the multi-pass kernel execution
    for DFT-D3(BJ) dispersion energy, forces, and coordination number calculations.
    Framework-specific wrappers (PyTorch, JAX) handle tensor-to-warp conversion and
    call this function.

    For periodic systems, use :func:`dftd3_matrix_pbc` instead.

    Multi-Pass Algorithm
    ---------------------
    1. **Pass 1**: Compute coordination numbers using geometric counting function
    2. **Pass 2**: Compute direct forces, energy, and accumulate dE/dCN
    3. **Pass 3**: Add CN-dependent force contribution using chain rule

    Parameters
    ----------
    positions : wp.array(dtype=vec3f or vec3d), shape [num_atoms]
        Atomic coordinates in consistent distance units (typically Bohr). Supports
        both float32 (vec3f) and float64 (vec3d) precision.
    numbers : wp.array(dtype=int32), shape [num_atoms]
        Atomic numbers
    neighbor_matrix : wp.array2d(dtype=int32), shape [num_atoms, max_neighbors]
        Neighbor indices. Padding entries have values >= fill_value.
    covalent_radii : wp.array(dtype=float32), shape [max_Z+1]
        Covalent radii indexed by atomic number, in same units as positions
    r4r2 : wp.array(dtype=float32), shape [max_Z+1]
        <r⁴>/<r²> expectation values for C8 computation (dimensionless)
    c6_reference : wp.array4d(dtype=float32), shape [max_Z+1, max_Z+1, 5, 5]
        C6 reference values in energy x distance^6 units
    coord_num_ref : wp.array4d(dtype=float32), shape [max_Z+1, max_Z+1, 5, 5]
        CN reference grid (dimensionless)
    a1 : float
        Becke-Johnson damping parameter 1 (functional-dependent, dimensionless)
    a2 : float
        Becke-Johnson damping parameter 2 (functional-dependent), in same units as positions
    s8 : float
        C8 term scaling factor (functional-dependent, dimensionless)
    coord_num : wp.array(dtype=float32), shape [num_atoms]
        OUTPUT: Coordination numbers (dimensionless). Must be pre-allocated and zeroed.
    forces : wp.array(dtype=vec3f), shape [num_atoms]
        OUTPUT: Atomic forces in energy/distance units. Must be pre-allocated and zeroed.
    energy : wp.array(dtype=float32), shape [num_systems]
        OUTPUT: Dispersion energy in energy units. Must be pre-allocated and zeroed.
    virial : wp.array(dtype=mat33f), shape [num_systems]
        OUTPUT: Virial tensor (not computed for non-periodic systems). Must be
        pre-allocated but will not be modified.
    batch_idx : wp.array(dtype=int32), shape [num_atoms]
        Batch indices mapping each atom to its system index.
    cartesian_shifts : wp.array(dtype=vec3f or vec3d), shape [num_atoms, max_neighbors]
        SCRATCH: Pre-allocated buffer for Cartesian shift vectors.
        Values are not used for non-periodic systems, but the array must
        still be provided with shape matching neighbor_matrix.
        Must be pre-allocated by caller.
    dE_dCN : wp.array(dtype=float32), shape [num_atoms]
        SCRATCH: Pre-allocated buffer for chain rule dE/dCN intermediate.
        Must be pre-allocated and zeroed by caller.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64) matching positions dtype.
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    k1 : float, optional
        CN counting function steepness parameter, in inverse distance units
        (typically 16.0 1/Bohr). Default: 16.0
    k3 : float, optional
        CN interpolation Gaussian width parameter (typically -4.0, dimensionless).
        Default: -4.0
    s6 : float, optional
        C6 term scaling factor (typically 1.0, dimensionless). Default: 1.0
    s5_smoothing_on : float, optional
        Distance where S5 switching begins, in same units as positions. Default: 0.0
    s5_smoothing_off : float, optional
        Distance where S5 switching completes, in same units as positions. Default: 0.0
    fill_value : int or None, optional
        Value indicating padding in neighbor_matrix. If None, inferred from num_atoms.

    Returns
    -------
    None
        All outputs are written to pre-allocated arrays (coord_num, forces, energy).

    Notes
    -----
    - All output arrays must be pre-allocated and zeroed by the caller
    - Supports float32 and float64 positions; outputs always float32
    - Padding atoms indicated by numbers[i] == 0 are skipped
    - **Two-body only**: Three-body Axilrod-Teller-Muto terms not included
    - Unit consistency required: standard D3 parameters use atomic units
      (Bohr for distances, Hartree for energy)
    - Virial is NOT computed for non-periodic systems (use dftd3_matrix_pbc for PBC)

    See Also
    --------
    dftd3_matrix_pbc : Neighbor matrix format with PBC support
    dftd3 : Neighbor list (CSR) format, non-periodic
    dftd3_pbc : Neighbor list (CSR) format with PBC support
    """
    # Get number of atoms from positions array
    num_atoms = positions.shape[0]

    # Set fill_value if not provided
    if fill_value is None:
        fill_value = num_atoms

    # Handle empty case
    if num_atoms == 0:
        return

    # Precompute inv_w for S5 switching
    if s5_smoothing_off > s5_smoothing_on:
        inv_w = 1.0 / (s5_smoothing_off - s5_smoothing_on)
    else:
        inv_w = 0.0

    periodic = False

    # Pass 1: Compute coordination numbers
    wp.launch(
        kernel=_cn_kernel_matrix_overload[wp_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            numbers,
            neighbor_matrix,
            cartesian_shifts,
            covalent_radii,
            wp.float32(k1),
            wp.int32(fill_value),
            periodic,
        ],
        outputs=[coord_num],
        device=device,
    )

    # Pass 2: Compute direct forces, energy, and accumulate dE/dCN
    # compute_virial=False for non-periodic systems
    wp.launch(
        kernel=_direct_forces_and_dE_dCN_kernel_matrix_overload[wp_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            numbers,
            neighbor_matrix,
            cartesian_shifts,
            coord_num,
            r4r2,
            c6_reference,
            coord_num_ref,
            wp.float32(k3),
            wp.float32(a1),
            wp.float32(a2),
            wp.float32(s6),
            wp.float32(s8),
            wp.float32(s5_smoothing_on),
            wp.float32(s5_smoothing_off),
            wp.float32(inv_w),
            wp.int32(fill_value),
            periodic,
            batch_idx,
            False,  # compute_virial=False for non-periodic
        ],
        outputs=[dE_dCN, forces, energy, virial],
        device=device,
    )

    # Pass 3: Add CN-dependent force contribution
    wp.launch(
        kernel=_cn_forces_contrib_kernel_matrix_overload[wp_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            numbers,
            neighbor_matrix,
            cartesian_shifts,
            covalent_radii,
            dE_dCN,
            wp.float32(k1),
            wp.int32(fill_value),
            periodic,
            batch_idx,
            False,  # compute_virial=False for non-periodic
        ],
        outputs=[forces, virial],
        device=device,
    )


def dftd3_matrix_pbc(
    positions: wp.array,
    numbers: wp.array,
    neighbor_matrix: wp.array,
    cell: wp.array,
    neighbor_matrix_shifts: wp.array,
    covalent_radii: wp.array,
    r4r2: wp.array,
    c6_reference: wp.array,
    coord_num_ref: wp.array,
    a1: float,
    a2: float,
    s8: float,
    coord_num: wp.array,
    forces: wp.array,
    energy: wp.array,
    virial: wp.array,
    batch_idx: wp.array,
    cartesian_shifts: wp.array,
    dE_dCN: wp.array,
    wp_dtype: type,
    device: str,
    k1: float = 16.0,
    k3: float = -4.0,
    s6: float = 1.0,
    s5_smoothing_on: float = 0.0,
    s5_smoothing_off: float = 0.0,
    fill_value: int | None = None,
    compute_virial: bool = False,
) -> None:
    """
    Launch DFT-D3(BJ) dispersion calculation using neighbor matrix format with PBC.

    This is a framework-agnostic warp launcher for periodic boundary condition (PBC)
    systems that accepts warp arrays directly and orchestrates the multi-pass kernel
    execution for DFT-D3(BJ) dispersion energy, forces, virial, and coordination
    number calculations. Framework-specific wrappers (PyTorch, JAX) handle
    tensor-to-warp conversion and call this function.

    For non-periodic systems, use :func:`dftd3_matrix` instead.

    Multi-Pass Algorithm
    ---------------------
    1. **Pass 0**: Convert unit cell shifts to Cartesian coordinates
    2. **Pass 1**: Compute coordination numbers using geometric counting function
    3. **Pass 2**: Compute direct forces, energy, and accumulate dE/dCN
    4. **Pass 3**: Add CN-dependent force contribution using chain rule

    Parameters
    ----------
    positions : wp.array(dtype=vec3f or vec3d), shape [num_atoms]
        Atomic coordinates in consistent distance units (typically Bohr). Supports
        both float32 (vec3f) and float64 (vec3d) precision.
    numbers : wp.array(dtype=int32), shape [num_atoms]
        Atomic numbers
    neighbor_matrix : wp.array2d(dtype=int32), shape [num_atoms, max_neighbors]
        Neighbor indices. Padding entries have values >= fill_value.
    cell : wp.array(dtype=mat33f or mat33d), shape [num_systems]
        Unit cell lattice vectors for PBC, in same dtype/units as positions.
        Convention: cell[s, i, :] is the i-th lattice vector for system s (row vectors).
    neighbor_matrix_shifts : wp.array2d(dtype=vec3i), shape [num_atoms, max_neighbors]
        Integer unit cell shifts for PBC. shift[i, k] is the shift for the k-th
        neighbor of atom i.
    covalent_radii : wp.array(dtype=float32), shape [max_Z+1]
        Covalent radii indexed by atomic number, in same units as positions
    r4r2 : wp.array(dtype=float32), shape [max_Z+1]
        <r⁴>/<r²> expectation values for C8 computation (dimensionless)
    c6_reference : wp.array4d(dtype=float32), shape [max_Z+1, max_Z+1, 5, 5]
        C6 reference values in energy x distance^6 units
    coord_num_ref : wp.array4d(dtype=float32), shape [max_Z+1, max_Z+1, 5, 5]
        CN reference grid (dimensionless)
    a1 : float
        Becke-Johnson damping parameter 1 (functional-dependent, dimensionless)
    a2 : float
        Becke-Johnson damping parameter 2 (functional-dependent), in same units as positions
    s8 : float
        C8 term scaling factor (functional-dependent, dimensionless)
    coord_num : wp.array(dtype=float32), shape [num_atoms]
        OUTPUT: Coordination numbers (dimensionless). Must be pre-allocated and zeroed.
    forces : wp.array(dtype=vec3f), shape [num_atoms]
        OUTPUT: Atomic forces in energy/distance units. Must be pre-allocated and zeroed.
    energy : wp.array(dtype=float32), shape [num_systems]
        OUTPUT: Dispersion energy in energy units. Must be pre-allocated and zeroed.
    virial : wp.array(dtype=mat33f), shape [num_systems]
        OUTPUT: Virial tensor in energy units. Must be pre-allocated and zeroed.
        Only computed if compute_virial=True.
    batch_idx : wp.array(dtype=int32), shape [num_atoms]
        Batch indices mapping each atom to its system index.
    cartesian_shifts : wp.array(dtype=vec3f or vec3d), shape [num_atoms, max_neighbors]
        SCRATCH: Pre-allocated buffer for Cartesian shift vectors.
        Populated by Pass 0 from unit cell shifts. Must be pre-allocated
        with shape matching neighbor_matrix.
    dE_dCN : wp.array(dtype=float32), shape [num_atoms]
        SCRATCH: Pre-allocated buffer for chain rule dE/dCN intermediate.
        Must be pre-allocated and zeroed by caller.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64) matching positions dtype.
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    k1 : float, optional
        CN counting function steepness parameter, in inverse distance units
        (typically 16.0 1/Bohr). Default: 16.0
    k3 : float, optional
        CN interpolation Gaussian width parameter (typically -4.0, dimensionless).
        Default: -4.0
    s6 : float, optional
        C6 term scaling factor (typically 1.0, dimensionless). Default: 1.0
    s5_smoothing_on : float, optional
        Distance where S5 switching begins, in same units as positions. Default: 0.0
    s5_smoothing_off : float, optional
        Distance where S5 switching completes, in same units as positions. Default: 0.0
    fill_value : int or None, optional
        Value indicating padding in neighbor_matrix. If None, inferred from num_atoms.
    compute_virial : bool, optional
        If True, compute virial tensor. Default: False

    Returns
    -------
    None
        All outputs are written to pre-allocated arrays (coord_num, forces, energy, virial).

    Notes
    -----
    - All output arrays must be pre-allocated and zeroed by the caller
    - Supports float32 and float64 positions/cell; outputs always float32
    - Padding atoms indicated by numbers[i] == 0 are skipped
    - **Two-body only**: Three-body Axilrod-Teller-Muto terms not included
    - Unit consistency required: standard D3 parameters use atomic units
      (Bohr for distances, Hartree for energy)
    - Virial tensor is computed when compute_virial=True

    See Also
    --------
    dftd3_matrix : Neighbor matrix format, non-periodic
    dftd3 : Neighbor list (CSR) format, non-periodic
    dftd3_pbc : Neighbor list (CSR) format with PBC support
    """
    # Get number of atoms from positions array
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1] if num_atoms > 0 else 0

    # Set fill_value if not provided
    if fill_value is None:
        fill_value = num_atoms

    # Handle empty case
    if num_atoms == 0:
        return

    # Precompute inv_w for S5 switching
    if s5_smoothing_off > s5_smoothing_on:
        inv_w = 1.0 / (s5_smoothing_off - s5_smoothing_on)
    else:
        inv_w = 0.0

    # Pass 0: Compute cartesian shifts from unit cell shifts
    periodic = True

    wp.launch(
        kernel=_compute_cartesian_shifts_matrix_overload[wp_dtype],
        dim=(num_atoms, max_neighbors),
        inputs=[
            cell,
            neighbor_matrix_shifts,
            neighbor_matrix,
            batch_idx,
            wp.int32(fill_value),
        ],
        outputs=[cartesian_shifts],
        device=device,
    )

    # Pass 1: Compute coordination numbers
    wp.launch(
        kernel=_cn_kernel_matrix_overload[wp_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            numbers,
            neighbor_matrix,
            cartesian_shifts,
            covalent_radii,
            wp.float32(k1),
            wp.int32(fill_value),
            periodic,
        ],
        outputs=[coord_num],
        device=device,
    )

    # Pass 2: Compute direct forces, energy, and accumulate dE/dCN
    wp.launch(
        kernel=_direct_forces_and_dE_dCN_kernel_matrix_overload[wp_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            numbers,
            neighbor_matrix,
            cartesian_shifts,
            coord_num,
            r4r2,
            c6_reference,
            coord_num_ref,
            wp.float32(k3),
            wp.float32(a1),
            wp.float32(a2),
            wp.float32(s6),
            wp.float32(s8),
            wp.float32(s5_smoothing_on),
            wp.float32(s5_smoothing_off),
            wp.float32(inv_w),
            wp.int32(fill_value),
            periodic,
            batch_idx,
            compute_virial,
        ],
        outputs=[dE_dCN, forces, energy, virial],
        device=device,
    )

    # Pass 3: Add CN-dependent force contribution
    wp.launch(
        kernel=_cn_forces_contrib_kernel_matrix_overload[wp_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            numbers,
            neighbor_matrix,
            cartesian_shifts,
            covalent_radii,
            dE_dCN,
            wp.float32(k1),
            wp.int32(fill_value),
            periodic,
            batch_idx,
            compute_virial,
        ],
        outputs=[forces, virial],
        device=device,
    )


def dftd3(
    positions: wp.array,
    numbers: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    covalent_radii: wp.array,
    r4r2: wp.array,
    c6_reference: wp.array,
    coord_num_ref: wp.array,
    a1: float,
    a2: float,
    s8: float,
    coord_num: wp.array,
    forces: wp.array,
    energy: wp.array,
    virial: wp.array,
    batch_idx: wp.array,
    cartesian_shifts: wp.array,
    dE_dCN: wp.array,
    wp_dtype: type,
    device: str,
    k1: float = 16.0,
    k3: float = -4.0,
    s6: float = 1.0,
    s5_smoothing_on: float = 0.0,
    s5_smoothing_off: float = 0.0,
) -> None:
    """
    Launch DFT-D3(BJ) dispersion calculation using neighbor list (CSR) format (non-periodic).

    This is a framework-agnostic warp launcher for non-periodic (non-PBC) systems
    that accepts warp arrays directly and orchestrates the multi-pass kernel execution
    for DFT-D3(BJ) dispersion energy, forces, and coordination number calculations
    using CSR (Compressed Sparse Row) neighbor list format. Framework-specific
    wrappers (PyTorch, JAX) handle tensor-to-warp conversion and call this function.

    For periodic systems, use :func:`dftd3_pbc` instead.

    Multi-Pass Algorithm
    ---------------------
    1. **Pass 1**: Compute coordination numbers using geometric counting function
    2. **Pass 2**: Compute direct forces, energy, and accumulate dE/dCN
    3. **Pass 3**: Add CN-dependent force contribution using chain rule

    Parameters
    ----------
    positions : wp.array(dtype=vec3f or vec3d), shape [num_atoms]
        Atomic coordinates in consistent distance units (typically Bohr). Supports
        both float32 (vec3f) and float64 (vec3d) precision.
    numbers : wp.array(dtype=int32), shape [num_atoms]
        Atomic numbers
    idx_j : wp.array(dtype=int32), shape [num_edges]
        Destination atom indices in CSR format
    neighbor_ptr : wp.array(dtype=int32), shape [num_atoms+1]
        CSR row pointers where neighbor_ptr[i]:neighbor_ptr[i+1]
        gives the range of neighbors for atom i
    covalent_radii : wp.array(dtype=float32), shape [max_Z+1]
        Covalent radii indexed by atomic number, in same units as positions
    r4r2 : wp.array(dtype=float32), shape [max_Z+1]
        <r⁴>/<r²> expectation values for C8 computation (dimensionless)
    c6_reference : wp.array4d(dtype=float32), shape [max_Z+1, max_Z+1, 5, 5]
        C6 reference values in energy x distance^6 units
    coord_num_ref : wp.array4d(dtype=float32), shape [max_Z+1, max_Z+1, 5, 5]
        CN reference grid (dimensionless)
    a1 : float
        Becke-Johnson damping parameter 1 (functional-dependent, dimensionless)
    a2 : float
        Becke-Johnson damping parameter 2 (functional-dependent), in same units as positions
    s8 : float
        C8 term scaling factor (functional-dependent, dimensionless)
    coord_num : wp.array(dtype=float32), shape [num_atoms]
        OUTPUT: Coordination numbers (dimensionless). Must be pre-allocated and zeroed.
    forces : wp.array(dtype=vec3f), shape [num_atoms]
        OUTPUT: Atomic forces in energy/distance units. Must be pre-allocated and zeroed.
    energy : wp.array(dtype=float32), shape [num_systems]
        OUTPUT: Dispersion energy in energy units. Must be pre-allocated and zeroed.
    virial : wp.array(dtype=mat33f), shape [num_systems]
        OUTPUT: Virial tensor (not computed for non-periodic systems). Must be
        pre-allocated but will not be modified.
    batch_idx : wp.array(dtype=int32), shape [num_atoms]
        Batch indices mapping each atom to its system index.
    cartesian_shifts : wp.array(dtype=vec3f or vec3d), shape [num_edges]
        SCRATCH: Pre-allocated buffer for Cartesian shift vectors.
        Values are not used for non-periodic systems, but the array must
        still be provided with length matching idx_j.
        Must be pre-allocated by caller.
    dE_dCN : wp.array(dtype=float32), shape [num_atoms]
        SCRATCH: Pre-allocated buffer for chain rule dE/dCN intermediate.
        Must be pre-allocated and zeroed by caller.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64) matching positions dtype.
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    k1 : float, optional
        CN counting function steepness parameter, in inverse distance units
        (typically 16.0 1/Bohr). Default: 16.0
    k3 : float, optional
        CN interpolation Gaussian width parameter (typically -4.0, dimensionless).
        Default: -4.0
    s6 : float, optional
        C6 term scaling factor (typically 1.0, dimensionless). Default: 1.0
    s5_smoothing_on : float, optional
        Distance where S5 switching begins, in same units as positions. Default: 0.0
    s5_smoothing_off : float, optional
        Distance where S5 switching completes, in same units as positions. Default: 0.0

    Returns
    -------
    None
        All outputs are written to pre-allocated arrays (coord_num, forces, energy).
        Virial is not computed for non-periodic systems.

    Notes
    -----
    - All output arrays must be pre-allocated and zeroed by the caller
    - Supports float32 and float64 positions; outputs always float32
    - Padding atoms indicated by numbers[i] == 0 are skipped
    - **Two-body only**: Three-body Axilrod-Teller-Muto terms not included
    - Unit consistency required: standard D3 parameters use atomic units
      (Bohr for distances, Hartree for energy)
    - CSR format is more memory-efficient for sparse neighbor lists
    - Virial is NOT computed for non-periodic systems (use dftd3_pbc for PBC)

    See Also
    --------
    dftd3_pbc : Neighbor list (CSR) format with PBC support
    dftd3_matrix : Neighbor matrix format, non-periodic
    dftd3_matrix_pbc : Neighbor matrix format with PBC support
    """
    # Get number of atoms and edges
    num_atoms = positions.shape[0]
    num_edges = idx_j.shape[0]

    # Handle empty case
    if num_atoms == 0 or num_edges == 0:
        return

    # Precompute inv_w for S5 switching
    if s5_smoothing_off > s5_smoothing_on:
        inv_w = 1.0 / (s5_smoothing_off - s5_smoothing_on)
    else:
        inv_w = 0.0

    periodic = False

    # Pass 1: Compute coordination numbers
    wp.launch(
        kernel=_cn_kernel_overload[wp_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            numbers,
            idx_j,
            neighbor_ptr,
            cartesian_shifts,
            covalent_radii,
            wp.float32(k1),
            periodic,
        ],
        outputs=[coord_num],
        device=device,
    )

    # Pass 2: Compute direct forces, energy, and accumulate dE/dCN
    # compute_virial=False for non-periodic systems
    wp.launch(
        kernel=_direct_forces_and_dE_dCN_kernel_overload[wp_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            numbers,
            idx_j,
            neighbor_ptr,
            cartesian_shifts,
            coord_num,
            r4r2,
            c6_reference,
            coord_num_ref,
            wp.float32(k3),
            wp.float32(a1),
            wp.float32(a2),
            wp.float32(s6),
            wp.float32(s8),
            wp.float32(s5_smoothing_on),
            wp.float32(s5_smoothing_off),
            wp.float32(inv_w),
            periodic,
            batch_idx,
            False,  # compute_virial=False for non-periodic
        ],
        outputs=[dE_dCN, forces, energy, virial],
        device=device,
    )

    # Pass 3: Add CN-dependent force contribution
    wp.launch(
        kernel=_cn_forces_contrib_kernel_overload[wp_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            numbers,
            idx_j,
            neighbor_ptr,
            cartesian_shifts,
            covalent_radii,
            dE_dCN,
            wp.float32(k1),
            periodic,
            batch_idx,
            False,  # compute_virial=False for non-periodic
        ],
        outputs=[forces, virial],
        device=device,
    )


def dftd3_pbc(
    positions: wp.array,
    numbers: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    cell: wp.array,
    unit_shifts: wp.array,
    covalent_radii: wp.array,
    r4r2: wp.array,
    c6_reference: wp.array,
    coord_num_ref: wp.array,
    a1: float,
    a2: float,
    s8: float,
    coord_num: wp.array,
    forces: wp.array,
    energy: wp.array,
    virial: wp.array,
    batch_idx: wp.array,
    cartesian_shifts: wp.array,
    dE_dCN: wp.array,
    wp_dtype: type,
    device: str,
    k1: float = 16.0,
    k3: float = -4.0,
    s6: float = 1.0,
    s5_smoothing_on: float = 0.0,
    s5_smoothing_off: float = 0.0,
    compute_virial: bool = False,
) -> None:
    """
    Launch DFT-D3(BJ) dispersion calculation using neighbor list (CSR) format with PBC.

    This is a framework-agnostic warp launcher for periodic boundary condition (PBC)
    systems that accepts warp arrays directly and orchestrates the multi-pass kernel
    execution for DFT-D3(BJ) dispersion energy, forces, virial, and coordination
    number calculations using CSR (Compressed Sparse Row) neighbor list format.
    Framework-specific wrappers (PyTorch, JAX) handle tensor-to-warp conversion
    and call this function.

    For non-periodic systems, use :func:`dftd3` instead.

    Multi-Pass Algorithm
    ---------------------
    1. **Pass 0**: Convert unit cell shifts to Cartesian coordinates
    2. **Pass 1**: Compute coordination numbers using geometric counting function
    3. **Pass 2**: Compute direct forces, energy, and accumulate dE/dCN
    4. **Pass 3**: Add CN-dependent force contribution using chain rule

    Parameters
    ----------
    positions : wp.array(dtype=vec3f or vec3d), shape [num_atoms]
        Atomic coordinates in consistent distance units (typically Bohr). Supports
        both float32 (vec3f) and float64 (vec3d) precision.
    numbers : wp.array(dtype=int32), shape [num_atoms]
        Atomic numbers
    idx_j : wp.array(dtype=int32), shape [num_edges]
        Destination atom indices in CSR format
    neighbor_ptr : wp.array(dtype=int32), shape [num_atoms+1]
        CSR row pointers where neighbor_ptr[i]:neighbor_ptr[i+1]
        gives the range of neighbors for atom i
    cell : wp.array(dtype=mat33f or mat33d), shape [num_systems]
        Unit cell lattice vectors for PBC, in same dtype/units as positions.
        Convention: cell[s, i, :] is the i-th lattice vector for system s (row vectors).
    unit_shifts : wp.array(dtype=vec3i), shape [num_edges]
        Integer unit cell shifts for PBC. shift[e] is the shift for edge e.
    covalent_radii : wp.array(dtype=float32), shape [max_Z+1]
        Covalent radii indexed by atomic number, in same units as positions
    r4r2 : wp.array(dtype=float32), shape [max_Z+1]
        <r⁴>/<r²> expectation values for C8 computation (dimensionless)
    c6_reference : wp.array4d(dtype=float32), shape [max_Z+1, max_Z+1, 5, 5]
        C6 reference values in energy x distance^6 units
    coord_num_ref : wp.array4d(dtype=float32), shape [max_Z+1, max_Z+1, 5, 5]
        CN reference grid (dimensionless)
    a1 : float
        Becke-Johnson damping parameter 1 (functional-dependent, dimensionless)
    a2 : float
        Becke-Johnson damping parameter 2 (functional-dependent), in same units as positions
    s8 : float
        C8 term scaling factor (functional-dependent, dimensionless)
    coord_num : wp.array(dtype=float32), shape [num_atoms]
        OUTPUT: Coordination numbers (dimensionless). Must be pre-allocated and zeroed.
    forces : wp.array(dtype=vec3f), shape [num_atoms]
        OUTPUT: Atomic forces in energy/distance units. Must be pre-allocated and zeroed.
    energy : wp.array(dtype=float32), shape [num_systems]
        OUTPUT: Dispersion energy in energy units. Must be pre-allocated and zeroed.
    virial : wp.array(dtype=mat33f), shape [num_systems]
        OUTPUT: Virial tensor in energy units. Must be pre-allocated and zeroed.
        Only computed if compute_virial=True.
    batch_idx : wp.array(dtype=int32), shape [num_atoms]
        Batch indices mapping each atom to its system index.
    cartesian_shifts : wp.array(dtype=vec3f or vec3d), shape [num_edges]
        SCRATCH: Pre-allocated buffer for Cartesian shift vectors.
        Populated by Pass 0 from unit cell shifts. Must be pre-allocated
        with length matching idx_j.
    dE_dCN : wp.array(dtype=float32), shape [num_atoms]
        SCRATCH: Pre-allocated buffer for chain rule dE/dCN intermediate.
        Must be pre-allocated and zeroed by caller.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64) matching positions dtype.
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    k1 : float, optional
        CN counting function steepness parameter, in inverse distance units
        (typically 16.0 1/Bohr). Default: 16.0
    k3 : float, optional
        CN interpolation Gaussian width parameter (typically -4.0, dimensionless).
        Default: -4.0
    s6 : float, optional
        C6 term scaling factor (typically 1.0, dimensionless). Default: 1.0
    s5_smoothing_on : float, optional
        Distance where S5 switching begins, in same units as positions. Default: 0.0
    s5_smoothing_off : float, optional
        Distance where S5 switching completes, in same units as positions. Default: 0.0
    compute_virial : bool, optional
        If True, compute virial tensor. Default: False

    Returns
    -------
    None
        All outputs are written to pre-allocated arrays (coord_num, forces, energy, virial).

    Notes
    -----
    - All output arrays must be pre-allocated and zeroed by the caller
    - Supports float32 and float64 positions/cell; outputs always float32
    - Padding atoms indicated by numbers[i] == 0 are skipped
    - **Two-body only**: Three-body Axilrod-Teller-Muto terms not included
    - Unit consistency required: standard D3 parameters use atomic units
      (Bohr for distances, Hartree for energy)
    - Virial tensor is computed when compute_virial=True

    See Also
    --------
    dftd3 : Neighbor list (CSR) format, non-periodic
    dftd3_matrix : Neighbor matrix format, non-periodic
    dftd3_matrix_pbc : Neighbor matrix format with PBC support
    """
    # Get number of atoms and edges
    num_atoms = positions.shape[0]
    num_edges = idx_j.shape[0]

    # Handle empty case
    if num_atoms == 0 or num_edges == 0:
        return

    # Precompute inv_w for S5 switching
    if s5_smoothing_off > s5_smoothing_on:
        inv_w = 1.0 / (s5_smoothing_off - s5_smoothing_on)
    else:
        inv_w = 0.0

    # Pass 0: Compute cartesian shifts from unit cell shifts
    periodic = True

    wp.launch(
        kernel=_compute_cartesian_shifts_overload[wp_dtype],
        dim=num_atoms,
        inputs=[
            cell,
            unit_shifts,
            neighbor_ptr,
            batch_idx,
        ],
        outputs=[cartesian_shifts],
        device=device,
    )

    # Pass 1: Compute coordination numbers
    wp.launch(
        kernel=_cn_kernel_overload[wp_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            numbers,
            idx_j,
            neighbor_ptr,
            cartesian_shifts,
            covalent_radii,
            wp.float32(k1),
            periodic,
        ],
        outputs=[coord_num],
        device=device,
    )

    # Pass 2: Compute direct forces, energy, and accumulate dE/dCN
    wp.launch(
        kernel=_direct_forces_and_dE_dCN_kernel_overload[wp_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            numbers,
            idx_j,
            neighbor_ptr,
            cartesian_shifts,
            coord_num,
            r4r2,
            c6_reference,
            coord_num_ref,
            wp.float32(k3),
            wp.float32(a1),
            wp.float32(a2),
            wp.float32(s6),
            wp.float32(s8),
            wp.float32(s5_smoothing_on),
            wp.float32(s5_smoothing_off),
            wp.float32(inv_w),
            periodic,
            batch_idx,
            compute_virial,
        ],
        outputs=[dE_dCN, forces, energy, virial],
        device=device,
    )

    # Pass 3: Add CN-dependent force contribution
    wp.launch(
        kernel=_cn_forces_contrib_kernel_overload[wp_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            numbers,
            idx_j,
            neighbor_ptr,
            cartesian_shifts,
            covalent_radii,
            dE_dCN,
            wp.float32(k1),
            periodic,
            batch_idx,
            compute_virial,
        ],
        outputs=[forces, virial],
        device=device,
    )

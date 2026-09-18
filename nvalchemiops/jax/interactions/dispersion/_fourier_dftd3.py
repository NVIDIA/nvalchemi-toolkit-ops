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

r"""
JAX binding for FourierD3.

Evaluates the periodic DFT-D3(BJ) dispersion correction by particle-mesh summation, with no
real-space cutoff on the dispersion sum itself. See
:mod:`nvalchemiops.interactions.dispersion._fourier_dftd3` for the method.

This layer supplies the two Fourier transforms, which Warp cannot perform on a full mesh, and
drives the Warp launchers around them through ``warp.jax_experimental.jax_kernel``.

The kernels are launched with ``enable_backward=False``, matching the real-space
:func:`~nvalchemiops.jax.interactions.dispersion.dftd3`: forces and the virial are explicit
outputs rather than quantities recovered by differentiating the energy.

Units
-----
Every length must share one system: ``positions``, ``cell``, ``rcov``, ``r_cut`` and
``mesh_spacing``. The DFT-D3 reference parameters are conventionally atomic units, so
``r_cut`` has no default.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from warp.jax_experimental import jax_kernel

from nvalchemiops.interactions.dispersion._c6_decomposition import (
    decompose_c6_reference,
)
from nvalchemiops.interactions.dispersion._fourier_dftd3 import (
    FD3_CN_BLOCK_SIZE,
    FD3_KSPACE_BLOCK_SIZE,
    _fd3_cn_forces_kernel_overload,
    _fd3_cn_forces_matrix_kernel_overload,
    _fd3_cn_kernel_overload,
    _fd3_cn_matrix_kernel_overload,
    _fd3_cn_sensitivity_kernel_overload,
    _fd3_coefficients_kernel_overload,
    _fd3_gather_and_force_kernel_overload,
    _fd3_kspace_kernel_overload,
    _fd3_self_energy_kernel_overload,
    _fd3_spread_kernel_overload,
)

__all__ = [
    "FourierD3Parameters",
    "fourier_dftd3",
]

# warp.jax_experimental.ffi hardcodes a block dimension of 256, so any kernel whose launch is
# expressed per atom must be given that as its second launch dimension.
JAX_BLOCK_DIM = 256


def _normalize_dtype(dtype):
    """Resolve a JAX dtype to the key used for kernel dispatch."""
    if dtype == jnp.float32 or str(dtype) == "float32":
        return jnp.float32
    if dtype == jnp.float64 or str(dtype) == "float64":
        return jnp.float64
    raise ValueError(f"Unsupported dtype for FourierD3 positions: {dtype}")


def _make_jax_kernels(overloads, num_outputs, in_out_argnames=None, block_dim=None):
    """Wrap a dtype-keyed set of Warp overloads as JAX kernels."""
    jax_to_wp = {jnp.float32: wp.float32, jnp.float64: wp.float64}
    extra = {} if in_out_argnames is None else {"in_out_argnames": in_out_argnames}
    if block_dim is not None:
        extra["block_dim"] = block_dim
    return {
        jax_dtype: jax_kernel(
            overloads[wp_dtype],
            num_outputs=num_outputs,
            enable_backward=False,
            **extra,
        )
        for jax_dtype, wp_dtype in jax_to_wp.items()
    }


_coordination_kernels = _make_jax_kernels(
    _fd3_cn_kernel_overload, 1, block_dim=FD3_CN_BLOCK_SIZE
)
_coordination_matrix_kernels = _make_jax_kernels(
    _fd3_cn_matrix_kernel_overload, 1, block_dim=FD3_CN_BLOCK_SIZE
)
_coefficient_kernels = _make_jax_kernels(_fd3_coefficients_kernel_overload, 2)
_spread_kernels = _make_jax_kernels(_fd3_spread_kernel_overload, 1, ["mesh"])
_kspace_kernels = _make_jax_kernels(
    _fd3_kspace_kernel_overload,
    3,
    ["energy", "cotangent", "virial"],
    block_dim=FD3_KSPACE_BLOCK_SIZE,
)
_gather_kernels = _make_jax_kernels(
    _fd3_gather_and_force_kernel_overload, 2, ["d_energy_d_c6", "forces"]
)
_self_energy_kernels = _make_jax_kernels(
    _fd3_self_energy_kernel_overload, 2, ["energy", "d_energy_d_c6"]
)
_sensitivity_kernels = _make_jax_kernels(_fd3_cn_sensitivity_kernel_overload, 1)
_cn_forces_kernels = _make_jax_kernels(
    _fd3_cn_forces_kernel_overload,
    2,
    ["forces", "virial"],
    block_dim=FD3_CN_BLOCK_SIZE,
)
_cn_forces_matrix_kernels = _make_jax_kernels(
    _fd3_cn_forces_matrix_kernel_overload,
    2,
    ["forces", "virial"],
    block_dim=FD3_CN_BLOCK_SIZE,
)


@dataclass
class FourierD3Parameters:
    """Separable dispersion coefficients for the species in a system.

    The JAX counterpart of
    :class:`nvalchemiops.torch.interactions.dispersion.FourierD3Parameters`, and like it it
    stores no damping parameters: those are call-time arguments, so one instance is valid for
    every functional parametrisation.

    Attributes
    ----------
    rcov : jax.Array, shape (max_z + 1,)
        Covalent radii indexed by atomic number.
    sqrt_q : jax.Array, shape (n_species,)
        Square root of the quadrupole-to-dipole ratio, per species channel.
    cnref : jax.Array, shape (n_species, n_ref)
        Reference coordination numbers, negative in unused slots.
    v_q : jax.Array, shape (n_species, n_ref, rank)
        Eigenvectors of the decomposed reference tensor.
    eigs : jax.Array, shape (rank,)
        Eigenvalues, which may be negative.
    species_map : jax.Array, shape (max_z + 1,), dtype=int32
        Atomic number to channel index; ``-1`` where a species is not covered.
    max_relative_error : float
        Largest relative error the truncated decomposition makes.
    """

    rcov: jax.Array
    sqrt_q: jax.Array
    cnref: jax.Array
    v_q: jax.Array
    eigs: jax.Array
    species_map: jax.Array
    max_relative_error: float

    @property
    def rank(self) -> int:
        """Number of retained rank slots."""
        return int(self.eigs.shape[0])

    @property
    def n_species(self) -> int:
        """Number of species channels."""
        return int(self.cnref.shape[0])

    @classmethod
    def from_tables(
        cls,
        rcov,
        r4r2,
        c6ab,
        cn_ref,
        species,
        tol: float = 1e-4,
        max_rank: int | None = None,
        dtype=jnp.float64,
    ) -> FourierD3Parameters:
        """Decompose Grimme's reference tables for a given set of species.

        Parameters mirror the Torch implementation; see that class for details.
        """
        decomposition = decompose_c6_reference(
            np.asarray(c6ab, dtype=np.float64),
            np.asarray(cn_ref, dtype=np.float64),
            species,
            tol=tol,
            max_rank=max_rank,
        )
        return cls(
            rcov=jnp.asarray(rcov, dtype=dtype),
            sqrt_q=jnp.asarray(np.asarray(r4r2)[decomposition.species], dtype=dtype),
            cnref=jnp.asarray(decomposition.cnref, dtype=dtype),
            v_q=jnp.asarray(decomposition.v_q, dtype=dtype),
            eigs=jnp.asarray(decomposition.eigs, dtype=dtype),
            species_map=jnp.asarray(decomposition.species_map, dtype=jnp.int32),
            max_relative_error=decomposition.max_relative_error,
        )


def _next_fft_friendly(size):
    """Smallest size at least ``size`` whose only prime factors are 2, 3, 5 and 7.

    cuFFT has specialised radix kernels for these factors and falls back to Bluestein's
    algorithm otherwise, and the difference is not marginal -- a prime or large-factor edge can
    cost several times what the next friendly size does.

    Only a mesh derived from ``mesh_spacing`` is rounded. An explicit ``mesh_dimensions`` is a
    number the caller chose and is passed through exactly, even when it is a poor size.
    """
    candidate = max(1, int(size))
    while True:
        remainder = candidate
        for prime in (2, 3, 5, 7):
            while remainder % prime == 0:
                remainder //= prime
        if remainder == 1:
            return candidate
        candidate += 1


def _check_mesh_supports_stencil(mesh, spline_order, origin):
    """Refuse a mesh too short to hold the interpolation stencil.

    Every axis must hold at least ``spline_order`` nodes. The stencil is that wide and wraps
    periodically, so on a shorter axis two stencil points land on the same node and the
    interpolation stops being the B-spline the gather differentiates. Equality is fine: the
    stencil then covers each node exactly once.
    """
    if min(mesh) < spline_order:
        raise ValueError(
            f"{origin} gives mesh {mesh}, but every axis must hold at least "
            f"spline_order = {spline_order} nodes. The interpolation stencil is that wide "
            f"and wraps periodically, so a shorter axis would visit a node twice."
        )
    return mesh


def _resolve_mesh(mesh_dimensions, mesh_spacing, cells, spline_order):
    """Settle the mesh size, requiring exactly one of the two ways of asking for it.

    Either route is held to the same minimum, ``spline_order`` nodes per axis.
    """
    if (mesh_dimensions is None) == (mesh_spacing is None):
        raise ValueError(
            "Provide exactly one of mesh_dimensions or mesh_spacing. There is no "
            "accuracy-based default for FourierD3."
        )
    if mesh_dimensions is not None:
        if len(mesh_dimensions) != 3 or any(int(n) < 1 for n in mesh_dimensions):
            raise ValueError(
                f"mesh_dimensions must be three positive integers, got {mesh_dimensions}."
            )
        return _check_mesh_supports_stencil(
            tuple(int(n) for n in mesh_dimensions), spline_order, "mesh_dimensions"
        )
    if mesh_spacing <= 0.0:
        raise ValueError(f"mesh_spacing must be positive, got {mesh_spacing}.")
    try:
        lengths = np.linalg.norm(np.asarray(cells), axis=-1).max(axis=0)
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        raise ValueError(
            "mesh_spacing reads the cell lengths, which is not possible inside jax.jit. "
            "Pass mesh_dimensions explicitly when tracing."
        ) from None
    return _check_mesh_supports_stencil(
        tuple(_next_fft_friendly(np.ceil(length / mesh_spacing)) for length in lengths),
        spline_order,
        f"mesh_spacing = {mesh_spacing}",
    )


def _reject_half_filled(
    neighbor_list,
    unit_shifts,
    neighbor_matrix,
    neighbor_matrix_shifts,
    fill_value,
    n_atoms,
):
    """Reject a neighbour list that does not hold both directions of every pair.

    FourierD3 accumulates each atom's coordination number, and the chain rule from it, out
    of that atom's own row only; the reverse edge is walked by the other atom's block. Both
    orientations therefore have to be present, which is what the neighbour builders produce
    unless asked for ``half_fill=True``.

    In a full directed list every edge is cancelled by its reverse, so ``source - target``
    and the image shifts each sum to exactly zero. The test is sound as a rejection -- a
    valid list can never trip it -- without being complete. Reading the sums is impossible
    while tracing, so under ``jax.jit`` the check is skipped rather than failing.
    """
    if neighbor_matrix is not None:
        valid = jnp.asarray(neighbor_matrix) < fill_value
        rows = jnp.arange(n_atoms, dtype=jnp.int64)[:, None]
        balance = jnp.where(valid, rows - jnp.asarray(neighbor_matrix), 0).sum()
        drift = jnp.where(valid[..., None], jnp.asarray(neighbor_matrix_shifts), 0).sum(
            axis=(0, 1)
        )
    else:
        balance = (neighbor_list[0] - neighbor_list[1]).sum()
        drift = jnp.asarray(unit_shifts).sum(axis=0)
    try:
        residual = int(abs(balance)) + int(jnp.abs(drift).sum())
    except (
        jax.errors.ConcretizationTypeError,
        jax.errors.TracerArrayConversionError,
    ):
        return
    if residual != 0:
        raise ValueError(
            "The neighbour list does not hold both directions of every pair. FourierD3 "
            "builds each atom's coordination number from its own row, so a half-filled "
            "list omits contributions and yields wrong energies and non-conservative "
            "forces. Rebuild it with half_fill=False."
        )


def fourier_dftd3(
    positions,
    numbers,
    a1: float,
    a2: float,
    s8: float,
    *,
    fd3_params: FourierD3Parameters,
    cell,
    r_cut: float,
    mesh_dimensions: tuple[int, int, int] | None = None,
    mesh_spacing: float | None = None,
    neighbor_matrix=None,
    neighbor_matrix_shifts=None,
    neighbor_list=None,
    neighbor_ptr=None,
    unit_shifts=None,
    fill_value: int | None = None,
    s6: float = 1.0,
    spline_order: int = 4,
    exact_moduli: bool = True,
    batch_idx=None,
    compute_virial: bool = False,
    num_systems: int | None = None,
):
    r"""Evaluate the DFT-D3(BJ) dispersion correction by particle-mesh summation.

    The JAX counterpart of
    :func:`nvalchemiops.torch.interactions.dispersion.fourier_dftd3`; the arguments and their
    meanings are the same, minus ``device`` and the precomputed ``setup``, which exists in the
    Torch binding to keep ``torch.linalg.inv`` out of a CUDA graph and has no counterpart
    here because ``jax.jit`` already hoists it.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic positions.
    numbers : jax.Array, shape (N,)
        Atomic numbers. Zero marks a padding atom.
    a1, a2, s8 : float
        Becke-Johnson damping parameters.
    fd3_params : FourierD3Parameters
        Separable coefficients covering every species present.
    cell : jax.Array, shape (3, 3), (1, 3, 3) or (B, 3, 3)
        Lattice vectors as rows. Required: FourierD3 is periodic.
    r_cut : float
        Coordination-number cutoff, in the same length unit as ``positions``. Must equal the
        radius the neighbour list was built with.
    mesh_dimensions, mesh_spacing
        Exactly one is required. ``mesh_spacing`` reads cell lengths and so cannot be used
        inside ``jax.jit``. A mesh derived from it is rounded **up** to a size whose only
        prime factors are 2, 3, 5 and 7, keeping the transform on cuFFT's radix kernels; the
        result is never coarser than the spacing asked for. An explicit ``mesh_dimensions``
        is used exactly as given.
    neighbor_matrix, neighbor_matrix_shifts, neighbor_list, neighbor_ptr, unit_shifts
        Exactly one neighbour format, with its matching lattice images.

        It must hold **both directions of every pair**, which is what the neighbour builders
        produce by default. FourierD3 accumulates each atom's coordination number from its
        own row alone, so a list built with ``half_fill=True`` loses half of every atom's
        coordination and yields wrong energies and non-conservative forces. Such a list is
        rejected rather than used, except while tracing, where the values cannot be read.
    fill_value : int, optional
        Padding sentinel for the dense format. Defaults to the atom count.
    s6 : float, default=1.0
        Sixth-order scaling.
    spline_order : int, default=4
        B-spline interpolation order, from 2 to 6. Accuracy at a fixed mesh improves with
        order: measured against a converged reference, orders 2 to 5 land roughly three
        orders of magnitude apart each way, so raising the order buys more than refining the
        mesh does. Order 3 is noticeably noisier than its neighbours; prefer an even order
        unless you have measured otherwise.
    exact_moduli : bool, default=True
        Which B-spline attenuation to divide out. ``True`` uses the discrete modulus, the
        magnitude of the DFT of the spline coefficients, which is what interpolation on a
        finite mesh actually applies and what the gather differentiates. ``False`` uses the
        continuous ``sinc(m / N) ** spline_order``, the convention the in-repo PME uses,
        retained so results can be reproduced against it. Host-static, so it selects a branch
        at trace time and is safe under ``jax.jit``.
    batch_idx : jax.Array, shape (N,), optional
        System index per atom.
    compute_virial : bool, default=False
        Whether to return the virial.
    num_systems : int, optional
        Number of systems. Inferred from ``cell`` when omitted.

    Returns
    -------
    energy : jax.Array, shape (num_systems,)
    forces : jax.Array, shape (N, 3)
    virial : jax.Array, shape (num_systems, 3, 3)
        Returned only when ``compute_virial`` is set. Follows the repository convention in
        ``docs/userguide/about/conventions.md``: the **negative** derivative of the energy
        with respect to the affine displacement, :math:`W = -\partial E/\partial u`. The
        tensile-positive Cauchy stress is :math:`\sigma = -W/V`.

    Notes
    -----
    The returned ``energy`` is **not differentiable**. The kernels are launched with
    ``enable_backward=False`` and no VJP or JVP rule is registered on top of them, so
    ``jax.grad`` of the energy raises ``ValueError: ... cannot be differentiated`` rather than
    reproducing ``forces``. Use the returned ``forces`` and ``virial``, which are analytic
    derivatives of the same energy. This matches the real-space
    :func:`~nvalchemiops.jax.interactions.dispersion.dftd3`, and it is what keeps the call
    traceable under :func:`jax.jit`.
    """
    matrix_given = neighbor_matrix is not None
    list_given = neighbor_list is not None
    if matrix_given and list_given:
        raise ValueError(
            "Cannot provide both neighbor_matrix and neighbor_list. "
            "Please provide only one neighbor representation format."
        )
    if not matrix_given and not list_given:
        raise ValueError("Must provide either neighbor_matrix or neighbor_list.")
    if matrix_given and unit_shifts is not None:
        raise ValueError(
            "unit_shifts is for neighbor_list format. "
            "Use neighbor_matrix_shifts for neighbor_matrix format."
        )
    if list_given and neighbor_matrix_shifts is not None:
        raise ValueError(
            "neighbor_matrix_shifts is for neighbor_matrix format. "
            "Use unit_shifts for neighbor_list format."
        )
    if cell is None:
        raise ValueError("cell is required: FourierD3 evaluates a periodic sum.")
    if spline_order < 2 or spline_order > 6:
        raise ValueError(f"spline_order must be between 2 and 6, got {spline_order}.")

    dtype = _normalize_dtype(positions.dtype)
    positions = jnp.asarray(positions, dtype=dtype)
    cells = jnp.asarray(cell, dtype=dtype).reshape(-1, 3, 3)
    n_atoms = positions.shape[0]
    if num_systems is None:
        num_systems = cells.shape[0]
    if batch_idx is None:
        batch_idx = jnp.zeros(n_atoms, dtype=jnp.int32)
    batch_idx = batch_idx.astype(jnp.int32)
    numbers = numbers.astype(jnp.int32)
    if fill_value is None:
        fill_value = n_atoms

    params = fd3_params
    rcov = jnp.asarray(params.rcov, dtype=dtype)
    cnref = jnp.asarray(params.cnref, dtype=dtype)
    v_q = jnp.asarray(params.v_q, dtype=dtype)
    eigs = jnp.asarray(params.eigs, dtype=dtype)
    sqrt_q = jnp.asarray(params.sqrt_q, dtype=dtype)
    species_index = params.species_map[numbers].astype(jnp.int32)

    _reject_half_filled(
        neighbor_list,
        unit_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        fill_value,
        n_atoms,
    )

    mesh_nx, mesh_ny, mesh_nz = _resolve_mesh(
        mesh_dimensions, mesh_spacing, cells, spline_order
    )
    n_species, rank = params.n_species, params.rank

    if n_atoms == 0:
        # Nothing to spread, so the mesh, its transforms and the reciprocal sum would all be
        # zero. This sits after every argument check rather than before: an empty batch has
        # to reject a bad mesh or a missing parameter exactly as a populated one does, or the
        # mistake stays hidden until a later batch happens to contain an atom. What it does
        # skip is the work -- a mesh of num_systems * n_species * rank * nx * ny * nz, the
        # transforms over it, and handing Warp zero-length arrays, which it cannot wrap.
        # ``n_atoms`` is a shape, so the branch is taken at trace time and is safe under jit.
        energy = jnp.zeros(num_systems, dtype=dtype)
        forces = jnp.zeros((0, 3), dtype=dtype)
        if compute_virial:
            return energy, forces, jnp.zeros((num_systems, 3, 3), dtype=dtype)
        return energy, forces
    n_groups = num_systems * n_species

    # Padding atoms, and any species the decomposition does not cover, keep a negative group
    # so that every kernel's guard fires. Folding the system index in first would make a
    # padding atom in system 1 or later land on a valid slab belonging to an earlier system.
    group_idx = jnp.where(
        species_index < 0, -1, batch_idx * n_species + species_index
    ).astype(jnp.int32)
    cell_inv_t = jnp.swapaxes(jnp.linalg.inv(cells), -1, -2)
    cell_inv_grouped = jnp.repeat(cell_inv_t, n_species, axis=0)

    if matrix_given:
        shifts = jnp.asarray(neighbor_matrix_shifts, dtype=dtype)
        if cells.shape[0] == 1:
            cartesian_shifts = shifts @ cells[0]
        else:
            # A row holds one atom's neighbours and shifts by that atom's own lattice.
            cartesian_shifts = shifts @ cells[batch_idx]
        neighbours = jnp.asarray(neighbor_matrix, dtype=jnp.int32)
    else:
        shifts = jnp.asarray(unit_shifts, dtype=dtype)
        if cells.shape[0] == 1:
            # Every edge shares one cell; the general path would gather a 3x3 per edge.
            cartesian_shifts = shifts @ cells[0]
        else:
            edge_system = batch_idx[neighbor_list[0].astype(jnp.int32)]
            cartesian_shifts = jnp.einsum("ij,ijk->ik", shifts, cells[edge_system])
        neighbours = neighbor_list[1].astype(jnp.int32)

    # Pass 1: coordination numbers.
    if matrix_given:
        (coordination,) = _coordination_matrix_kernels[dtype](
            positions,
            numbers,
            neighbours,
            cartesian_shifts,
            rcov,
            float(r_cut),
            int(fill_value),
            int(FD3_CN_BLOCK_SIZE),
            launch_dims=(n_atoms, FD3_CN_BLOCK_SIZE),
            output_dims={"coord_num": (n_atoms,)},
        )
    else:
        (coordination,) = _coordination_kernels[dtype](
            positions,
            numbers,
            neighbours,
            jnp.asarray(neighbor_ptr, dtype=jnp.int32),
            cartesian_shifts,
            rcov,
            float(r_cut),
            int(FD3_CN_BLOCK_SIZE),
            launch_dims=(n_atoms, FD3_CN_BLOCK_SIZE),
            output_dims={"coord_num": (n_atoms,)},
        )

    # Pass 2: separable coefficients.
    c6, dc6_dcn = _coefficient_kernels[dtype](
        coordination,
        species_index,
        cnref,
        v_q,
        launch_dims=(n_atoms,),
        output_dims={"c6": (n_atoms, rank), "dc6_dcn": (n_atoms, rank)},
    )

    # Pass 3: spread onto the (system, species, rank) mesh.
    mesh = jnp.zeros((n_groups * rank, mesh_nx, mesh_ny, mesh_nz), dtype=dtype)
    (mesh,) = _spread_kernels[dtype](
        positions,
        c6,
        group_idx,
        cell_inv_grouped,
        int(spline_order),
        int(rank),
        mesh,
        launch_dims=(n_atoms, spline_order**3),
    )

    # Pass 4: forward transform.
    mesh_fft = jnp.fft.rfftn(mesh, axes=(-3, -2, -1))
    mesh_fft_pairs = jnp.stack([mesh_fft.real, mesh_fft.imag], axis=-1).astype(dtype)

    miller_x = jnp.fft.fftfreq(mesh_nx, d=1.0 / mesh_nx).astype(dtype)
    miller_y = jnp.fft.fftfreq(mesh_ny, d=1.0 / mesh_ny).astype(dtype)
    miller_z = jnp.fft.rfftfreq(mesh_nz, d=1.0 / mesh_nz).astype(dtype)
    moduli = [
        _bspline_moduli(m, n, spline_order, exact_moduli, dtype)
        for m, n in ((miller_x, mesh_nx), (miller_y, mesh_ny), (miller_z, mesh_nz))
    ]
    volumes = jnp.abs(jnp.linalg.det(cells)).astype(dtype)
    k_matrix = (2.0 * jnp.pi * jnp.linalg.inv(cells)).astype(dtype)

    # Pass 5: reciprocal-space contraction. The bin count is padded so that a block of the
    # reduction never spans two systems.
    num_bins = mesh_nx * mesh_ny * (mesh_nz // 2 + 1)
    padded_bins = -(-num_bins // FD3_KSPACE_BLOCK_SIZE) * FD3_KSPACE_BLOCK_SIZE
    energy_init = jnp.zeros(num_systems, dtype=dtype)
    virial_init = jnp.zeros((num_systems, 3, 3), dtype=dtype)
    cotangent_init = jnp.zeros_like(mesh_fft_pairs)
    energy, cotangent, virial = _kspace_kernels[dtype](
        mesh_fft_pairs,
        k_matrix,
        moduli[0],
        moduli[1],
        moduli[2],
        volumes,
        sqrt_q,
        eigs,
        float(s6),
        float(s8),
        float(a1),
        float(a2),
        int(mesh_nx),
        int(mesh_ny),
        int(mesh_nz),
        int(num_bins),
        int(FD3_KSPACE_BLOCK_SIZE),
        int(n_species),
        int(rank),
        bool(compute_virial),
        energy_init,
        cotangent_init,
        virial_init,
        launch_dims=(num_systems, padded_bins),
    )

    # Pass 6: inverse transform, unnormalised so that it is the adjoint of the forward one.
    potential = jnp.fft.irfftn(
        cotangent[..., 0] + 1j * cotangent[..., 1],
        s=(mesh_nx, mesh_ny, mesh_nz),
        axes=(-3, -2, -1),
        norm="forward",
    ).astype(dtype)

    # Pass 7: gather the coefficient derivative and the direct mesh force.
    d_energy_d_c6, forces = _gather_kernels[dtype](
        potential,
        positions,
        c6,
        group_idx,
        cell_inv_grouped,
        int(spline_order),
        int(rank),
        jnp.zeros((n_atoms, rank), dtype=dtype),
        jnp.zeros((n_atoms, 3), dtype=dtype),
        launch_dims=(n_atoms,),
    )

    # Pass 8: self-energy, before the chain rule.
    energy, d_energy_d_c6 = _self_energy_kernels[dtype](
        c6,
        species_index,
        batch_idx,
        sqrt_q,
        eigs,
        float(s6),
        float(s8),
        float(a1),
        float(a2),
        energy,
        d_energy_d_c6,
        launch_dims=(n_atoms,),
    )

    # Pass 9: contract to dE/dCN, then chain through the real-space edges.
    (sensitivity,) = _sensitivity_kernels[dtype](
        d_energy_d_c6,
        dc6_dcn,
        launch_dims=(n_atoms,),
        output_dims={"d_energy_d_cn": (n_atoms,)},
    )
    if matrix_given:
        forces, virial = _cn_forces_matrix_kernels[dtype](
            sensitivity,
            positions,
            numbers,
            neighbours,
            cartesian_shifts,
            rcov,
            float(r_cut),
            int(fill_value),
            batch_idx,
            int(FD3_CN_BLOCK_SIZE),
            bool(compute_virial),
            forces,
            virial,
            launch_dims=(n_atoms, FD3_CN_BLOCK_SIZE),
        )
    else:
        forces, virial = _cn_forces_kernels[dtype](
            sensitivity,
            positions,
            numbers,
            neighbours,
            jnp.asarray(neighbor_ptr, dtype=jnp.int32),
            cartesian_shifts,
            rcov,
            float(r_cut),
            batch_idx,
            int(FD3_CN_BLOCK_SIZE),
            bool(compute_virial),
            forces,
            virial,
            launch_dims=(n_atoms, FD3_CN_BLOCK_SIZE),
        )

    if compute_virial:
        return energy, forces, virial
    return energy, forces


def _cardinal_bspline(u, order):
    """Cardinal B-spline of the given order, by the Cox-de Boor recursion."""
    if order == 1:
        return jnp.where((u >= 0.0) & (u < 1.0), 1.0, 0.0)
    lower = _cardinal_bspline(u, order - 1)
    shifted = _cardinal_bspline(u - 1.0, order - 1)
    return (u * lower + (float(order) - u) * shifted) / float(order - 1)


def _bspline_moduli(miller, mesh_size, spline_order, exact, dtype):
    """B-spline attenuation for one mesh axis.

    With ``exact`` set, the magnitude of the DFT of the spline coefficients, which is what
    interpolation on a finite mesh actually applies. The Nyquist bin of an even mesh can
    vanish, which would divide by zero during deconvolution, so it is replaced by the mean of
    its neighbours.

    Otherwise ``sinc(m / N) ** spline_order``, the continuous transform of the spline and the
    convention the in-repo PME uses. It is the cheaper approximation and is retained so that
    results can be reproduced against PME; the discrete form is the default because it is
    what the gather actually differentiates.
    """
    if not exact:
        return (jnp.sinc(miller / mesh_size) ** spline_order).astype(dtype)

    nodes = jnp.arange(spline_order, dtype=dtype) + 1.0
    coefficients = (
        jnp.zeros(mesh_size, dtype=dtype)
        .at[:spline_order]
        .set(_cardinal_bspline(nodes, spline_order))
    )
    modulus = jnp.abs(jnp.fft.fft(coefficients))
    if mesh_size % 2 == 0:
        half = mesh_size // 2
        modulus = modulus.at[half].set(0.5 * (modulus[half - 1] + modulus[half + 1]))
    return modulus[jnp.round(miller).astype(jnp.int32) % mesh_size].astype(dtype)

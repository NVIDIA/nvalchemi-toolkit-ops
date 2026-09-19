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

Particle-mesh DFT-D3(BJ) with no real-space cutoff on the dispersion sum. See
:mod:`nvalchemiops.interactions.dispersion._fourier_dftd3` for the method.

This layer supplies the two Fourier transforms, which Warp cannot perform on a full mesh, and
drives the Warp launchers through ``warp.jax_kernel``. The coordination-number and
reciprocal-space passes reduce within a block, which Warp's CPU backend cannot do, so their
launches narrow to one thread per item there; see :func:`_blocked`. Results agree between
backends.

Kernels run with
``enable_backward=False``, matching :func:`~nvalchemiops.jax.interactions.dispersion.dftd3`:
forces and the virial are explicit outputs, not derivatives of the energy.

Units
-----
``positions``, ``cell``, ``rcov``, ``r_cut`` and ``mesh_spacing`` must share one length unit.
D3 parameters are conventionally atomic units, so ``r_cut`` has no default.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from warp import jax_kernel

from nvalchemiops.interactions.dispersion._c6_decomposition import (
    decompose_c6_reference,
)
from nvalchemiops.interactions.dispersion._fourier_dftd3 import (
    FD3_CN_BLOCK_SIZE,
    FD3_KSPACE_BLOCK_SIZE,
    _check_mesh_supports_stencil,
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
    _next_fft_friendly,
)

__all__ = [
    "FourierD3Parameters",
    "fourier_dftd3",
]

_ELEMENTWISE_BLOCK_DIM = 256
"""Threads per block for the passes with no block-level cooperation.

These kernels are one thread per atom, or per atom and stencil point, with no reduction
across the block, so the size only has to keep the machine busy. Stated explicitly rather
than left to ``jax_kernel``'s default, which is this same value but not part of its
contract. The coordination-number and reciprocal-space passes reduce within a block and set
their own sizes.
"""


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
_coefficient_kernels = _make_jax_kernels(
    _fd3_coefficients_kernel_overload, 2, block_dim=_ELEMENTWISE_BLOCK_DIM
)
_spread_kernels = _make_jax_kernels(
    _fd3_spread_kernel_overload, 1, ["mesh"], block_dim=_ELEMENTWISE_BLOCK_DIM
)
_kspace_kernels = _make_jax_kernels(
    _fd3_kspace_kernel_overload,
    3,
    ["energy", "cotangent", "virial"],
    block_dim=FD3_KSPACE_BLOCK_SIZE,
)
_gather_kernels = _make_jax_kernels(
    _fd3_gather_and_force_kernel_overload,
    2,
    ["d_energy_d_c6", "forces"],
    block_dim=_ELEMENTWISE_BLOCK_DIM,
)
_self_energy_kernels = _make_jax_kernels(
    _fd3_self_energy_kernel_overload,
    2,
    ["energy", "d_energy_d_c6"],
    block_dim=_ELEMENTWISE_BLOCK_DIM,
)
_sensitivity_kernels = _make_jax_kernels(
    _fd3_cn_sensitivity_kernel_overload, 1, block_dim=_ELEMENTWISE_BLOCK_DIM
)
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


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["rcov", "sqrt_q", "cn_ref", "v_q", "eigs", "species_map"],
    meta_fields=["max_relative_error"],
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
    cn_ref : jax.Array, shape (n_species, n_ref)
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
    cn_ref: jax.Array
    v_q: jax.Array
    eigs: jax.Array
    species_map: jax.Array
    max_relative_error: float

    def __post_init__(self):
        """Validate dtypes and ranks, matching the Torch container.

        This also runs when the pytree is rebuilt inside ``jax.jit``, where the fields are
        tracers. Only ``dtype`` and ``ndim`` are read, which tracers carry, so the checks
        hold while tracing as well as eagerly.
        """
        fields = {
            "rcov": self.rcov,
            "sqrt_q": self.sqrt_q,
            "cn_ref": self.cn_ref,
            "v_q": self.v_q,
            "eigs": self.eigs,
        }
        for name, array in fields.items():
            if array.dtype not in (jnp.float32, jnp.float64):
                raise TypeError(
                    f"{name} must be float32 or float64, got {array.dtype}."
                )
        if self.species_map.dtype not in (jnp.int32, jnp.int64):
            raise TypeError(
                f"species_map must be int32 or int64, got {self.species_map.dtype}."
            )
        fields["species_map"] = self.species_map
        ranks = {
            "rcov": 1,
            "sqrt_q": 1,
            "eigs": 1,
            "species_map": 1,
            "cn_ref": 2,
            "v_q": 3,
        }
        for name, expected in ranks.items():
            array = fields[name]
            if array.ndim != expected:
                raise ValueError(
                    f"{name} must be {expected}D, got shape {tuple(array.shape)}."
                )

    @property
    def rank(self) -> int:
        """Number of retained rank slots."""
        return int(self.eigs.shape[0])

    @property
    def n_species(self) -> int:
        """Number of species channels."""
        return int(self.cn_ref.shape[0])

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
            cn_ref=jnp.asarray(decomposition.cn_ref, dtype=dtype),
            v_q=jnp.asarray(decomposition.v_q, dtype=dtype),
            eigs=jnp.asarray(decomposition.eigs, dtype=dtype),
            species_map=jnp.asarray(decomposition.species_map, dtype=jnp.int32),
            max_relative_error=decomposition.max_relative_error,
        )


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


def _blocked(call, gpu_block, reference):
    """Run a block-per-item kernel with the block width its backend can actually provide.

    These passes launch one block per atom or per bin, stride the block's threads through
    that item's work and reduce at the end. Warp's CPU backend has no block launches, so a
    wide launch there leaves every partial sum but the first unwritten and the result is
    silently wrong. One thread per item, striding by one, is correct on CPU and is what the
    Warp launchers behind the Torch binding already do.

    Two paths, because neither alone covers both. Eagerly ``reference`` carries a concrete
    device and the width follows it directly, which is the only thing that respects an
    explicit ``jax.device_put``. While tracing it carries nothing, so both widths are staged
    and :func:`jax.lax.platform_dependent` resolves the choice at lowering, where the
    compilation target is known.

    ``call`` takes a block width and returns the kernel's outputs.
    """
    try:
        platforms = {device.platform for device in reference.devices()}
    except (AttributeError, jax.errors.ConcretizationTypeError):
        return jax.lax.platform_dependent(
            cpu=lambda: call(1),
            default=lambda: call(gpu_block),
        )
    return call(gpu_block if platforms & {"gpu", "cuda", "rocm"} else 1)


def _reject_out_of_range_batch(batch_idx, num_systems):
    """Reject batch indices that point at no system.

    Out of range, an atom lands on a mesh slab belonging to no system and its contribution
    vanishes with no error: a whole-system ``batch_idx`` of 5 against one system returns
    exactly zero energy.

    Returns ``(multiplier, batch_idx)``. Eagerly the values can be read, so this raises and
    the pair passes through unchanged. While tracing they cannot, and ``jax.jit`` is the
    documented path, so the indices are clamped into range and the multiplier is NaN.
    """
    if batch_idx.shape[0] == 0:
        return 1.0, batch_idx
    try:
        lowest = int(np.asarray(batch_idx).min())
        highest = int(np.asarray(batch_idx).max())
    except (
        jax.errors.ConcretizationTypeError,
        jax.errors.TracerArrayConversionError,
    ):
        # Tracing: the values cannot be read, so this is handled on device in two parts.
        # The clamp keeps the kernels in bounds -- an index past the last system otherwise
        # reaches the grouped cell and the mesh out of range, which is an illegal access
        # rather than a wrong number. The multiplier then poisons the result, so the clamp
        # cannot quietly turn bad input into a plausible answer.
        in_range = (batch_idx >= 0) & (batch_idx < num_systems)
        guard = jnp.where(jnp.all(in_range), 1.0, jnp.nan)
        return guard, jnp.clip(batch_idx, 0, num_systems - 1)
    if lowest < 0 or highest >= num_systems:
        raise ValueError(
            f"batch_idx values span [{lowest}, {highest}], outside "
            f"[0, {num_systems}) for {num_systems} system(s)."
        )
    return 1.0, batch_idx


def _reject_uncovered_species(species_index, numbers):
    """Reject atoms whose element the decomposition does not cover.

    ``species_map`` marks both padding and uncovered elements with ``-1``, and the mesh
    grouping below treats every ``-1`` as padding. A real element missing from
    ``fourier_d3_params`` would therefore be dropped from the sum with no error at all, which is
    the worst kind of wrong: a plausible energy that is quietly missing atoms.

    Atomic number zero is padding and is skipped by design; only a real element that the
    decomposition misses is an error.

    Returns a multiplier for the results: ``1.0`` normally, NaN when tracing found an
    uncovered element. Eagerly the mask can be read, so this raises instead and names the
    elements. Under ``jax.jit`` it cannot, and merely skipping would leave the documented
    execution path returning a finite, plausible energy quietly missing atoms. Poisoning the
    result keeps that visible without the host synchronisation tracing forbids. This is the
    one check here that cannot simply be skipped, because its failure is silent corruption
    rather than an exception.
    """
    uncovered = (species_index < 0) & (numbers != 0)
    try:
        missing = np.unique(np.asarray(numbers)[np.asarray(uncovered)]).tolist()
    except (
        jax.errors.ConcretizationTypeError,
        jax.errors.TracerArrayConversionError,
    ):
        return jnp.where(jnp.any(uncovered), jnp.nan, 1.0)
    if missing:
        raise ValueError(
            f"Atomic numbers {missing} are not covered by fourier_d3_params. Rebuild the "
            f"decomposition with every species present in the system."
        )
    return 1.0


def fourier_dftd3(
    positions,
    numbers,
    a1: float,
    a2: float,
    s8: float,
    *,
    fourier_d3_params: FourierD3Parameters,
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
    rank_chunk_size: int | None = None,
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
    fourier_d3_params : FourierD3Parameters
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
        coordination and yields wrong energies and non-conservative forces. This is not
        checked: the values are unreadable under ``jax.jit``, which is the documented path,
        and the builders' default already satisfies it.
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
    rank_chunk_size : int, optional
        Number of rank slots to hold on the mesh at once. ``None``, the default, stages all
        ``rank`` slots in a single pass, which is fastest. The mesh and its transforms
        dominate the workspace and scale as
        ``num_systems * n_species * rank * nx * ny * nz``, so a system with many species or a
        large retained rank can exceed device memory on a fine mesh. Setting this to ``k``
        caps the resident slots at ``k`` and reduces that allocation by roughly ``rank / k``,
        at the cost of one extra spread, forward and inverse transform, and gather per chunk.
        Only those reciprocal stages are chunked; the self-energy and the coordination chain
        rule need every slot at once and run once afterwards. The per-atom ``(N, rank)``
        coefficient arrays stay at full width throughout. The result is unchanged to
        round-off: every reciprocal stage is a sum over slots with no coupling between them.
        Host-static, so it is safe under ``jax.jit`` -- but the loop is **unrolled at trace
        time**, so the traced graph and the compile time grow linearly in
        ``rank / rank_chunk_size``. Prefer the largest chunk that fits.
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
    ``fourier_d3_params`` must cover every element present. Eagerly this raises and names the
    missing elements; under ``jax.jit`` the mask cannot be read back, so the energy, forces
    and virial come back NaN rather than silently omitting those atoms.

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
    if matrix_given:
        if unit_shifts is not None:
            raise ValueError(
                "unit_shifts is for neighbor_list format. "
                "Use neighbor_matrix_shifts for neighbor_matrix format."
            )
        if neighbor_matrix_shifts is None:
            raise ValueError(
                "neighbor_matrix_shifts is required: FourierD3 is periodic, so every "
                "neighbour needs its lattice image."
            )
    else:
        if neighbor_matrix_shifts is not None:
            raise ValueError(
                "neighbor_matrix_shifts is for neighbor_matrix format. "
                "Use unit_shifts for neighbor_list format."
            )
        if neighbor_ptr is None:
            raise ValueError("neighbor_ptr is required alongside neighbor_list.")
        if unit_shifts is None:
            raise ValueError(
                "unit_shifts is required: FourierD3 is periodic, so every neighbour needs "
                "its lattice image."
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
    elif num_systems != cells.shape[0]:
        raise ValueError(
            f"num_systems is {num_systems} but cell holds {cells.shape[0]} system(s). "
            "The energy and virial are shaped from num_systems while the mesh is built "
            "from the cells, so a mismatch silently pads the result with zeros."
        )
    if batch_idx is None:
        batch_idx = jnp.zeros(n_atoms, dtype=jnp.int32)
    elif batch_idx.shape[0] != n_atoms:
        raise ValueError(
            f"batch_idx has {batch_idx.shape[0]} entries but there are {n_atoms} atoms."
        )
    batch_idx = batch_idx.astype(jnp.int32)
    batch_guard, batch_idx = _reject_out_of_range_batch(batch_idx, num_systems)
    numbers = numbers.astype(jnp.int32)
    if fill_value is None:
        fill_value = n_atoms

    params = fourier_d3_params
    covalent_radii = jnp.asarray(params.rcov, dtype=dtype)
    cn_ref = jnp.asarray(params.cn_ref, dtype=dtype)
    v_q = jnp.asarray(params.v_q, dtype=dtype)
    eigs = jnp.asarray(params.eigs, dtype=dtype)
    sqrt_q = jnp.asarray(params.sqrt_q, dtype=dtype)
    species_index = params.species_map[numbers].astype(jnp.int32)

    covered = _reject_uncovered_species(species_index, numbers)

    mesh_nx, mesh_ny, mesh_nz = _resolve_mesh(
        mesh_dimensions, mesh_spacing, cells, spline_order
    )
    n_species, rank = params.n_species, params.rank

    if n_atoms == 0:
        # Nothing to spread, so mesh, transforms and reciprocal sum are all zero. Placed
        # after every argument check, not before: an empty batch must reject a bad mesh just
        # as a populated one does. ``n_atoms`` is a shape, so this branches at trace time.
        energy = jnp.zeros(num_systems, dtype=dtype)
        forces = jnp.zeros((0, 3), dtype=dtype)
        if compute_virial:
            return energy, forces, jnp.zeros((num_systems, 3, 3), dtype=dtype)
        return energy, forces
    n_groups = num_systems * n_species

    # Padding, and species the decomposition misses, keep a negative group so every guard
    # fires. Folding the system index in first would land them on an earlier system's slab.
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

        def _coordination(block):
            return _coordination_matrix_kernels[dtype](
                positions,
                numbers,
                neighbours,
                cartesian_shifts,
                covalent_radii,
                float(r_cut),
                int(fill_value),
                int(block),
                launch_dims=(n_atoms, block),
                output_dims={"coord_num": (n_atoms,)},
            )

    else:

        def _coordination(block):
            return _coordination_kernels[dtype](
                positions,
                numbers,
                neighbours,
                jnp.asarray(neighbor_ptr, dtype=jnp.int32),
                cartesian_shifts,
                covalent_radii,
                float(r_cut),
                int(block),
                launch_dims=(n_atoms, block),
                output_dims={"coord_num": (n_atoms,)},
            )

    (coordination,) = _blocked(_coordination, FD3_CN_BLOCK_SIZE, positions)

    # Pass 2: separable coefficients.
    c6, dc6_dcn = _coefficient_kernels[dtype](
        coordination,
        species_index,
        cn_ref,
        v_q,
        launch_dims=(n_atoms,),
        output_dims={"c6": (n_atoms, rank), "dc6_dcn": (n_atoms, rank)},
    )

    miller_x = jnp.fft.fftfreq(mesh_nx, d=1.0 / mesh_nx).astype(dtype)
    miller_y = jnp.fft.fftfreq(mesh_ny, d=1.0 / mesh_ny).astype(dtype)
    miller_z = jnp.fft.rfftfreq(mesh_nz, d=1.0 / mesh_nz).astype(dtype)
    moduli = [
        _bspline_moduli(m, n, spline_order, exact_moduli, dtype)
        for m, n in ((miller_x, mesh_nx), (miller_y, mesh_ny), (miller_z, mesh_nz))
    ]
    volumes = jnp.abs(jnp.linalg.det(cells)).astype(dtype)
    k_matrix = (2.0 * jnp.pi * jnp.linalg.inv(cells)).astype(dtype)

    # Slots do not couple, so a chunk can go through spread, transform, contraction and
    # gather alone and be added in. The mesh dominates allocation and scales with resident
    # slots: this trades extra passes for peak memory.
    energy_total = jnp.zeros(num_systems, dtype=dtype)
    forces_total = jnp.zeros((n_atoms, 3), dtype=dtype)
    virial_total = jnp.zeros((num_systems, 3, 3), dtype=dtype)
    d_energy_d_c6_chunks = []

    # A non-positive size yields no chunks at all, so the loop below never runs and the
    # result stays at its zero initialisation. Checked rather than left to ``range``, which
    # rejects a step of zero but silently produces nothing for a negative one.
    slots = rank if rank_chunk_size is None else rank_chunk_size
    if not isinstance(slots, int) or isinstance(slots, bool):
        raise TypeError(
            f"rank_chunk_size must be an int or None, got {type(rank_chunk_size).__name__}."
            " It sets the number of kernel launches, so it cannot be an array or a traced"
            " value."
        )
    if slots < 1:
        raise ValueError(f"rank_chunk_size must be at least 1, got {rank_chunk_size}.")
    slots = min(slots, rank)
    for slot_start in range(0, rank, slots):
        slot_count = min(slots, rank - slot_start)
        c6_chunk = c6[:, slot_start : slot_start + slot_count]
        eigs_chunk = eigs[slot_start : slot_start + slot_count]

        # Pass 3: spread onto the (system, species, rank) mesh.
        mesh = jnp.zeros(
            (n_groups * slot_count, mesh_nx, mesh_ny, mesh_nz), dtype=dtype
        )
        (mesh,) = _spread_kernels[dtype](
            positions,
            c6_chunk,
            group_idx,
            cell_inv_grouped,
            int(spline_order),
            int(slot_count),
            mesh,
            launch_dims=(n_atoms, spline_order**3),
        )

        # Pass 4: forward transform.
        mesh_fft = jnp.fft.rfftn(mesh, axes=(-3, -2, -1))
        mesh_fft_pairs = jnp.stack([mesh_fft.real, mesh_fft.imag], axis=-1).astype(
            dtype
        )

        # Pass 5: reciprocal-space contraction. The bin count is padded so that a block of the
        # reduction never spans two systems.
        num_bins = mesh_nx * mesh_ny * (mesh_nz // 2 + 1)
        energy_init = jnp.zeros(num_systems, dtype=dtype)
        virial_init = jnp.zeros((num_systems, 3, 3), dtype=dtype)
        cotangent_init = jnp.zeros_like(mesh_fft_pairs)

        def _contract(block):
            # Padded so that a block of the reduction never spans two systems. At a width of
            # one there is nothing to pad.
            padded_bins = -(-num_bins // block) * block
            return _kspace_kernels[dtype](
                mesh_fft_pairs,
                k_matrix,
                moduli[0],
                moduli[1],
                moduli[2],
                volumes,
                sqrt_q,
                eigs_chunk,
                float(s6),
                float(s8),
                float(a1),
                float(a2),
                int(mesh_nx),
                int(mesh_ny),
                int(mesh_nz),
                int(num_bins),
                int(block),
                int(n_species),
                int(slot_count),
                bool(compute_virial),
                energy_init,
                cotangent_init,
                virial_init,
                launch_dims=(num_systems, padded_bins),
            )

        energy, cotangent, virial = _blocked(
            _contract, FD3_KSPACE_BLOCK_SIZE, positions
        )

        # Pass 6: inverse transform, unnormalised so that it is the adjoint of the forward one.
        potential = jnp.fft.irfftn(
            cotangent[..., 0] + 1j * cotangent[..., 1],
            s=(mesh_nx, mesh_ny, mesh_nz),
            axes=(-3, -2, -1),
            norm="forward",
        ).astype(dtype)

        # Pass 7: gather. The only stage that reads the mesh, so the only one in the loop.
        d_energy_d_c6_chunk, forces = _gather_kernels[dtype](
            potential,
            positions,
            c6_chunk,
            group_idx,
            cell_inv_grouped,
            int(spline_order),
            int(slot_count),
            jnp.zeros((n_atoms, slot_count), dtype=dtype),
            forces_total,
            launch_dims=(n_atoms,),
        )

        energy_total = energy_total + energy
        forces_total = forces
        virial_total = virial_total + virial
        d_energy_d_c6_chunks.append(d_energy_d_c6_chunk)

    # Reassembled at full width -- the chain rule below needs every slot at once. Per-atom,
    # not per-mesh-point, so it does not undermine the bound.
    d_energy_d_c6 = (
        d_energy_d_c6_chunks[0]
        if len(d_energy_d_c6_chunks) == 1
        else jnp.concatenate(d_energy_d_c6_chunks, axis=1)
    )

    # Pass 8: self-energy, before the chain rule. Needs every slot, and never touches the
    # mesh, so it runs once after the chunks rather than inside them.
    energy_total, d_energy_d_c6 = _self_energy_kernels[dtype](
        c6,
        species_index,
        batch_idx,
        sqrt_q,
        eigs,
        float(s6),
        float(s8),
        float(a1),
        float(a2),
        energy_total,
        d_energy_d_c6,
        launch_dims=(n_atoms,),
    )

    # Pass 9: contract to dE/dCN, then chain through the real-space edges. This walks the
    # whole neighbour list, which is what makes running it per chunk the expensive mistake.
    (sensitivity,) = _sensitivity_kernels[dtype](
        d_energy_d_c6,
        dc6_dcn,
        launch_dims=(n_atoms,),
        output_dims={"d_energy_d_cn": (n_atoms,)},
    )
    if matrix_given:

        def _chain(block):
            return _cn_forces_matrix_kernels[dtype](
                sensitivity,
                positions,
                numbers,
                neighbours,
                cartesian_shifts,
                covalent_radii,
                float(r_cut),
                int(fill_value),
                batch_idx,
                int(block),
                bool(compute_virial),
                forces_total,
                virial_total,
                launch_dims=(n_atoms, block),
            )

    else:

        def _chain(block):
            return _cn_forces_kernels[dtype](
                sensitivity,
                positions,
                numbers,
                neighbours,
                jnp.asarray(neighbor_ptr, dtype=jnp.int32),
                cartesian_shifts,
                covalent_radii,
                float(r_cut),
                batch_idx,
                int(block),
                bool(compute_virial),
                forces_total,
                virial_total,
                launch_dims=(n_atoms, block),
            )

    forces_total, virial_total = _blocked(_chain, FD3_CN_BLOCK_SIZE, positions)

    # 1.0 unless tracing found an element the decomposition misses; see
    # :func:`_reject_uncovered_species`.
    covered = covered * batch_guard
    energy_total = energy_total * covered
    forces_total = forces_total * covered
    if compute_virial:
        return energy_total, forces_total, virial_total * covered
    return energy_total, forces_total


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

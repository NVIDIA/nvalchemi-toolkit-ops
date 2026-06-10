# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# LMAX={0,1,2} real-space cell-gradient kernels.
#
# The pair energy depends on the unit cell through the periodic shift
# ``periodic_shift = cell^T @ shift_vec`` (with ``shift_vec ∈ Z^3``),
# ``r_vec = pos_j - pos_i + periodic_shift``, ``r = |r_vec|``. The cell
# gradient is ``∂E/∂cell[a, b] = Σ_pairs (∂E_pair/∂r_vec[b]) * shift_vec[a]``;
# each pair contributes one outer-product term that the kernel atomically
# scatters into a per-system 3x3 ``grad_cell`` array.
#
# ``half_neighbor_list``: scale = 0.5 for a full list (each pair processed
# twice; shift_vec and dPE_dr_j both flip sign across directions, so
# shift_vec ⊗ ∂E/∂r_vec is invariant and the two directions sum to one
# outer product per pair) and scale = 1.0 for a half list.

from __future__ import annotations

from typing import Any

import warp as wp

from nvalchemiops.interactions.electrostatics.multipole_ewald_kernels import (
    _dipole_pair_contribution_fused,
    _gto_ewald_ab,
    _gto_ewald_monopole_pair_terms_fused,
    _quadrupole_pair_contribution_fused,
)

_QUADRUPOLE_CELL_GRAD_KERNEL_CACHE: dict = {}
_QUADRUPOLE_CELL_GRAD_OVERLOAD_CACHE: dict = {}
_QUADRUPOLE_CELL_GRAD_SCALE_CACHE: dict = {}


def _make_scale_array(half_neighbor_list: bool, device: str):
    """Return a 1-element scale array cached per (device, list convention).

    Parameters
    ----------
    half_neighbor_list : bool
        ``True`` selects scale 1.0 (single direction per pair); ``False``
        selects 0.5 (full list, two directions summing to one outer product).
    device : str
        Warp device for the cached array.

    Returns
    -------
    warp.array
        Single-element ``float64`` array holding the per-direction scale.
    """
    key = (str(device), bool(half_neighbor_list))
    if key not in _QUADRUPOLE_CELL_GRAD_SCALE_CACHE:
        value = 1.0 if half_neighbor_list else 0.5
        _QUADRUPOLE_CELL_GRAD_SCALE_CACHE[key] = wp.array(
            [value],
            dtype=wp.float64,
            device=device,
        )
    return _QUADRUPOLE_CELL_GRAD_SCALE_CACHE[key]


# =============================================================================
# Kernel factory
# =============================================================================


def _make_quadrupole_cell_grad_kernel(storage: str, is_batch: bool):
    """Build the uninstantiated LMAX=2 cell-grad kernel for the given storage/batch.

    Parameters
    ----------
    storage : str
        Neighbor-list storage layout; only ``"csr"`` is supported.
    is_batch : bool
        ``True`` returns the batched kernel (per-system ``cells`` indexed by
        ``atom_batch_idx``); ``False`` returns the single-system kernel.

    Returns
    -------
    warp.Kernel
        The uninstantiated (``dtype=Any``) cell-gradient kernel, to be typed
        via :func:`wp.overload`.

    Raises
    ------
    ValueError
        If ``storage`` is not ``"csr"``.
    """
    if storage == "csr" and not is_batch:
        return _build_kernel_csr_single()
    if storage == "csr" and is_batch:
        return _build_kernel_csr_batched()
    raise ValueError(f"unsupported (storage={storage!r}, is_batch={is_batch})")


def _build_kernel_csr_single():
    def _kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        dipoles: wp.array(dtype=Any),
        quadrupoles: wp.array(dtype=Any),
        cell: wp.array(dtype=Any),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        sigma: wp.array(dtype=Any),
        alpha: wp.array(dtype=Any),
        per_direction_scale: wp.array(dtype=wp.float64),
        grad_energies: wp.array(dtype=wp.float64),
        grad_cell: wp.array(dtype=Any),
    ):
        r"""LMAX=2 single-system real-space cell-gradient scatter kernel.

        For each atom ``i`` (one thread), walk its CSR neighbor slice and
        accumulate the per-pair outer product
        :math:`\text{shift\_vec} \otimes \partial E_{pair}/\partial r_{vec}`
        into the per-system :math:`3\times3` ``grad_cell``, weighted by the
        per-direction ``scale`` and the symmetric energy weight
        :math:`(\text{grad\_energies}[i] + \text{grad\_energies}[j])/2`.

        Launch Grid
        -----------
        ``dim = n_atoms``. One thread per atom ``i``; the thread loops over
        ``neighbor_ptr[i] .. neighbor_ptr[i+1]``.

        Parameters
        ----------
        positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Cartesian atom positions.
        charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
            Per-atom monopole charges.
        dipoles : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Per-atom dipole moments.
        quadrupoles : wp.array, shape (N,), dtype=wp.mat33f or wp.mat33d
            Per-atom (symmetric) Cartesian quadrupole tensors.
        cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
            Single-system unit cell (rows are lattice vectors); transposed
            to map integer ``unit_shifts`` to the Cartesian periodic shift.
        idx_j : wp.array, shape (nnz,), dtype=wp.int32
            CSR column indices (neighbor atom ``j`` of each pair).
        neighbor_ptr : wp.array, shape (N + 1,), dtype=wp.int32
            CSR row pointers into ``idx_j`` / ``unit_shifts``.
        unit_shifts : wp.array, shape (nnz,), dtype=wp.vec3i
            Integer lattice shift :math:`\in \mathbb{Z}^3` for each pair.
        sigma : wp.array, shape (1,), dtype=wp.float32 or wp.float64
            Gaussian charge width for the GTO-Ewald real-space split.
        alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
            Ewald splitting parameter.
        per_direction_scale : wp.array, shape (1,), dtype=wp.float64
            Per-direction scale (1.0 half list, 0.5 full list).
        grad_energies : wp.array, shape (N,), dtype=wp.float64
            Upstream per-atom energy weights; pass ``ones(N)`` for the plain
            :math:`\partial E_{total}/\partial \text{cell}`.
        grad_cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
            OUTPUT: per-system :math:`3\times3` cell gradient, accumulated
            atomically.
        """
        atom_i = wp.tid()
        sigma_ = wp.float64(sigma[0])
        alpha_ = wp.float64(alpha[0])
        ab = _gto_ewald_ab(sigma_, alpha_)
        a_coef = ab[0]
        b_coef = ab[1]
        gc_template = grad_cell[0]
        scale = per_direction_scale[0]
        ge_i = grad_energies[atom_i]
        cell_t = wp.transpose(cell[0])

        pos_i = positions[atom_i]
        qi = wp.float64(charges[atom_i])
        mu_i_n = dipoles[atom_i]
        mu_i = wp.vec3d(
            wp.float64(mu_i_n[0]), wp.float64(mu_i_n[1]), wp.float64(mu_i_n[2])
        )
        Q_i_n = quadrupoles[atom_i]
        Q_i = wp.mat33d(
            wp.float64(Q_i_n[0, 0]),
            wp.float64(Q_i_n[0, 1]),
            wp.float64(Q_i_n[0, 2]),
            wp.float64(Q_i_n[1, 0]),
            wp.float64(Q_i_n[1, 1]),
            wp.float64(Q_i_n[1, 2]),
            wp.float64(Q_i_n[2, 0]),
            wp.float64(Q_i_n[2, 1]),
            wp.float64(Q_i_n[2, 2]),
        )

        k_start = neighbor_ptr[atom_i]
        k_end = neighbor_ptr[atom_i + 1]
        for k in range(k_start, k_end):
            j = idx_j[k]
            shift_vec = unit_shifts[k]
            qj = wp.float64(charges[j])
            pos_j = positions[j]
            mu_j_n = dipoles[j]
            mu_j = wp.vec3d(
                wp.float64(mu_j_n[0]), wp.float64(mu_j_n[1]), wp.float64(mu_j_n[2])
            )
            Q_j_n = quadrupoles[j]
            Q_j = wp.mat33d(
                wp.float64(Q_j_n[0, 0]),
                wp.float64(Q_j_n[0, 1]),
                wp.float64(Q_j_n[0, 2]),
                wp.float64(Q_j_n[1, 0]),
                wp.float64(Q_j_n[1, 1]),
                wp.float64(Q_j_n[1, 2]),
                wp.float64(Q_j_n[2, 0]),
                wp.float64(Q_j_n[2, 1]),
                wp.float64(Q_j_n[2, 2]),
            )

            periodic_shift = cell_t * type(pos_i)(
                type(pos_i[0])(shift_vec[0]),
                type(pos_i[0])(shift_vec[1]),
                type(pos_i[0])(shift_vec[2]),
            )
            sep = pos_j - pos_i + periodic_shift
            distance = wp.float64(wp.length(sep))
            if distance > wp.float64(1e-8):
                r_vec = wp.vec3d(
                    wp.float64(sep[0]), wp.float64(sep[1]), wp.float64(sep[2])
                )
                contrib = _quadrupole_pair_contribution_fused(
                    r_vec,
                    distance,
                    qi,
                    mu_i,
                    Q_i,
                    qj,
                    mu_j,
                    Q_j,
                    a_coef,
                    b_coef,
                )
                dr = contrib.dPE_dr_j
                sh0 = wp.float64(shift_vec[0])
                sh1 = wp.float64(shift_vec[1])
                sh2 = wp.float64(shift_vec[2])
                # Per-atom weighting (ge[i]+ge[j])/2; grad_energies = ones(N)
                # recovers the uniform scale·shift⊗dr emit.
                w_cell = scale * wp.float64(0.5) * (ge_i + grad_energies[j])
                s_dr0 = w_cell * dr[0]
                s_dr1 = w_cell * dr[1]
                s_dr2 = w_cell * dr[2]
                wp.atomic_add(
                    grad_cell,
                    0,
                    type(gc_template)(
                        type(gc_template[0, 0])(sh0 * s_dr0),
                        type(gc_template[0, 0])(sh0 * s_dr1),
                        type(gc_template[0, 0])(sh0 * s_dr2),
                        type(gc_template[0, 0])(sh1 * s_dr0),
                        type(gc_template[0, 0])(sh1 * s_dr1),
                        type(gc_template[0, 0])(sh1 * s_dr2),
                        type(gc_template[0, 0])(sh2 * s_dr0),
                        type(gc_template[0, 0])(sh2 * s_dr1),
                        type(gc_template[0, 0])(sh2 * s_dr2),
                    ),
                )

    _kernel.__name__ = "_multipole_real_space_quadrupole_csr_cell_grad_kernel"
    return wp.kernel(_kernel, enable_backward=False)


def _build_kernel_csr_batched():
    def _kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        dipoles: wp.array(dtype=Any),
        quadrupoles: wp.array(dtype=Any),
        cells: wp.array(dtype=Any),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        atom_batch_idx: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        sigma: wp.array(dtype=Any),
        alpha: wp.array(dtype=Any),
        per_direction_scale: wp.array(dtype=wp.float64),
        grad_energies: wp.array(dtype=wp.float64),
        grad_cell: wp.array(dtype=Any),
    ):
        r"""Batched LMAX=2 real-space cell-gradient scatter kernel.

        Batched analog of the single-system kernel: each atom ``i`` maps to a
        system ``b = atom_batch_idx[i]`` whose cell is ``cells[b]``, and the
        per-pair outer product is scattered atomically into ``grad_cell[b]``.

        Launch Grid
        -----------
        ``dim = n_atoms`` (total atoms across all batched systems). One thread
        per atom ``i``.

        Parameters
        ----------
        positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Cartesian atom positions (all systems concatenated).
        charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
            Per-atom monopole charges.
        dipoles : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Per-atom dipole moments.
        quadrupoles : wp.array, shape (N,), dtype=wp.mat33f or wp.mat33d
            Per-atom (symmetric) Cartesian quadrupole tensors.
        cells : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
            Per-system unit cells, indexed by ``atom_batch_idx``.
        idx_j : wp.array, shape (nnz,), dtype=wp.int32
            CSR column indices (neighbor atom ``j`` of each pair).
        neighbor_ptr : wp.array, shape (N + 1,), dtype=wp.int32
            CSR row pointers into ``idx_j`` / ``unit_shifts``.
        atom_batch_idx : wp.array, shape (N,), dtype=wp.int32
            System index ``b`` for each atom.
        unit_shifts : wp.array, shape (nnz,), dtype=wp.vec3i
            Integer lattice shift :math:`\in \mathbb{Z}^3` for each pair.
        sigma : wp.array, shape (1,), dtype=wp.float32 or wp.float64
            Gaussian charge width for the GTO-Ewald real-space split.
        alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
            Ewald splitting parameter.
        per_direction_scale : wp.array, shape (1,), dtype=wp.float64
            Per-direction scale (1.0 half list, 0.5 full list).
        grad_energies : wp.array, shape (N,), dtype=wp.float64
            Upstream per-atom energy weights; pass ``ones(N)`` for the plain
            :math:`\partial E_{total}/\partial \text{cell}`.
        grad_cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
            OUTPUT: per-system :math:`3\times3` cell gradients, accumulated
            atomically.
        """
        atom_i = wp.tid()
        b = atom_batch_idx[atom_i]
        sigma_ = wp.float64(sigma[0])
        alpha_ = wp.float64(alpha[0])
        ab = _gto_ewald_ab(sigma_, alpha_)
        a_coef = ab[0]
        b_coef = ab[1]
        gc_template = grad_cell[0]
        scale = per_direction_scale[0]
        ge_i = grad_energies[atom_i]
        cell_t = wp.transpose(cells[b])

        pos_i = positions[atom_i]
        qi = wp.float64(charges[atom_i])
        mu_i_n = dipoles[atom_i]
        mu_i = wp.vec3d(
            wp.float64(mu_i_n[0]), wp.float64(mu_i_n[1]), wp.float64(mu_i_n[2])
        )
        Q_i_n = quadrupoles[atom_i]
        Q_i = wp.mat33d(
            wp.float64(Q_i_n[0, 0]),
            wp.float64(Q_i_n[0, 1]),
            wp.float64(Q_i_n[0, 2]),
            wp.float64(Q_i_n[1, 0]),
            wp.float64(Q_i_n[1, 1]),
            wp.float64(Q_i_n[1, 2]),
            wp.float64(Q_i_n[2, 0]),
            wp.float64(Q_i_n[2, 1]),
            wp.float64(Q_i_n[2, 2]),
        )

        k_start = neighbor_ptr[atom_i]
        k_end = neighbor_ptr[atom_i + 1]
        for k in range(k_start, k_end):
            j = idx_j[k]
            shift_vec = unit_shifts[k]
            qj = wp.float64(charges[j])
            pos_j = positions[j]
            mu_j_n = dipoles[j]
            mu_j = wp.vec3d(
                wp.float64(mu_j_n[0]), wp.float64(mu_j_n[1]), wp.float64(mu_j_n[2])
            )
            Q_j_n = quadrupoles[j]
            Q_j = wp.mat33d(
                wp.float64(Q_j_n[0, 0]),
                wp.float64(Q_j_n[0, 1]),
                wp.float64(Q_j_n[0, 2]),
                wp.float64(Q_j_n[1, 0]),
                wp.float64(Q_j_n[1, 1]),
                wp.float64(Q_j_n[1, 2]),
                wp.float64(Q_j_n[2, 0]),
                wp.float64(Q_j_n[2, 1]),
                wp.float64(Q_j_n[2, 2]),
            )

            periodic_shift = cell_t * type(pos_i)(
                type(pos_i[0])(shift_vec[0]),
                type(pos_i[0])(shift_vec[1]),
                type(pos_i[0])(shift_vec[2]),
            )
            sep = pos_j - pos_i + periodic_shift
            distance = wp.float64(wp.length(sep))
            if distance > wp.float64(1e-8):
                r_vec = wp.vec3d(
                    wp.float64(sep[0]), wp.float64(sep[1]), wp.float64(sep[2])
                )
                contrib = _quadrupole_pair_contribution_fused(
                    r_vec,
                    distance,
                    qi,
                    mu_i,
                    Q_i,
                    qj,
                    mu_j,
                    Q_j,
                    a_coef,
                    b_coef,
                )
                dr = contrib.dPE_dr_j
                sh0 = wp.float64(shift_vec[0])
                sh1 = wp.float64(shift_vec[1])
                sh2 = wp.float64(shift_vec[2])
                w_cell = scale * wp.float64(0.5) * (ge_i + grad_energies[j])
                s_dr0 = w_cell * dr[0]
                s_dr1 = w_cell * dr[1]
                s_dr2 = w_cell * dr[2]
                wp.atomic_add(
                    grad_cell,
                    b,
                    type(gc_template)(
                        type(gc_template[0, 0])(sh0 * s_dr0),
                        type(gc_template[0, 0])(sh0 * s_dr1),
                        type(gc_template[0, 0])(sh0 * s_dr2),
                        type(gc_template[0, 0])(sh1 * s_dr0),
                        type(gc_template[0, 0])(sh1 * s_dr1),
                        type(gc_template[0, 0])(sh1 * s_dr2),
                        type(gc_template[0, 0])(sh2 * s_dr0),
                        type(gc_template[0, 0])(sh2 * s_dr1),
                        type(gc_template[0, 0])(sh2 * s_dr2),
                    ),
                )

    _kernel.__name__ = "_batch_multipole_real_space_quadrupole_csr_cell_grad_kernel"
    return wp.kernel(_kernel, enable_backward=False)


def _sig_csr_single(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=v),
        wp.array(dtype=m),
        wp.array(dtype=m),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.vec3i),
        wp.array(dtype=t),
        wp.array(dtype=t),
        wp.array(dtype=wp.float64),  # per_direction_scale
        wp.array(dtype=wp.float64),  # grad_energies
        wp.array(dtype=m),
    ]


def _sig_csr_batched(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=v),
        wp.array(dtype=m),
        wp.array(dtype=m),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.vec3i),
        wp.array(dtype=t),
        wp.array(dtype=t),
        wp.array(dtype=wp.float64),  # per_direction_scale
        wp.array(dtype=wp.float64),  # grad_energies
        wp.array(dtype=m),
    ]


_SIG_BUILDERS = {
    ("csr", False): _sig_csr_single,
    ("csr", True): _sig_csr_batched,
}


def _get_overload(storage: str, is_batch: bool, vec_dtype, scalar_dtype):
    kernel_key = (storage, is_batch)
    if kernel_key not in _QUADRUPOLE_CELL_GRAD_KERNEL_CACHE:
        _QUADRUPOLE_CELL_GRAD_KERNEL_CACHE[kernel_key] = (
            _make_quadrupole_cell_grad_kernel(storage, is_batch)
        )
    kernel = _QUADRUPOLE_CELL_GRAD_KERNEL_CACHE[kernel_key]
    overload_key = (storage, is_batch, vec_dtype)
    if overload_key not in _QUADRUPOLE_CELL_GRAD_OVERLOAD_CACHE:
        sig = _SIG_BUILDERS[(storage, is_batch)](vec_dtype, scalar_dtype)
        _QUADRUPOLE_CELL_GRAD_OVERLOAD_CACHE[overload_key] = wp.overload(kernel, sig)
    return _QUADRUPOLE_CELL_GRAD_OVERLOAD_CACHE[overload_key]


# =============================================================================
# Public launchers
# =============================================================================


def multipole_real_space_quadrupole_csr_cell_grad(
    positions,
    charges,
    dipoles,
    quadrupoles,
    cell,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma,
    alpha,
    grad_energies,
    grad_cell,
    *,
    device: str,
    half_neighbor_list: bool = False,
):
    r"""CSR single-system LMAX=2 cell-gradient launcher.

    Per-pair shift scatters are weighted by
    ``(grad_energies[i] + grad_energies[j])/2``; pass
    ``grad_energies = ones(N)`` to recover the uniform-weight
    :math:`\partial E_{total}/\partial \text{cell}` emit.

    Parameters
    ----------
    positions : torch.Tensor / warp.array, shape (N,), vec3 dtype
        Cartesian atom positions.
    charges : shape (N,), scalar dtype
        Per-atom monopole charges.
    dipoles : shape (N,), vec3 dtype
        Per-atom dipole moments.
    quadrupoles : shape (N,), mat33 dtype
        Per-atom symmetric Cartesian quadrupole tensors.
    cell : shape (1,), mat33 dtype
        Single-system unit cell.
    idx_j : shape (nnz,), int32
        CSR neighbor column indices.
    neighbor_ptr : shape (N + 1,), int32
        CSR row pointers.
    unit_shifts : shape (nnz,), vec3i
        Integer lattice shifts per pair.
    sigma : shape (1,), scalar dtype
        Gaussian charge width.
    alpha : shape (1,), scalar dtype
        Ewald splitting parameter.
    grad_energies : shape (N,), float64
        Upstream per-atom energy weights.
    grad_cell : shape (1,), mat33 dtype
        OUTPUT: per-system :math:`3\times3` cell gradient (atomically
        accumulated; not zeroed by this launcher).
    device : str, keyword-only
        Warp device for the launch.
    half_neighbor_list : bool, keyword-only, default False
        ``True`` for a half neighbor list (scale 1.0); ``False`` for a full
        list (scale 0.5).
    """
    overload = _get_overload("csr", False, positions.dtype, charges.dtype)
    n_atoms = positions.shape[0]
    scale_arr = _make_scale_array(half_neighbor_list, device)
    wp.launch(
        overload,
        dim=n_atoms,
        inputs=[
            positions,
            charges,
            dipoles,
            quadrupoles,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma,
            alpha,
            scale_arr,
            grad_energies,
            grad_cell,
        ],
        device=device,
    )


def batch_multipole_real_space_quadrupole_csr_cell_grad(
    positions,
    charges,
    dipoles,
    quadrupoles,
    cells,
    idx_j,
    neighbor_ptr,
    atom_batch_idx,
    unit_shifts,
    sigma,
    alpha,
    grad_energies,
    grad_cell,
    *,
    device: str,
    half_neighbor_list: bool = False,
):
    r"""Batched CSR LMAX=2 cell-gradient launcher.

    Per-pair shift scatters weighted by
    ``(grad_energies[i] + grad_energies[j])/2``.

    Parameters
    ----------
    positions : shape (N,), vec3 dtype
        Cartesian atom positions (all systems concatenated).
    charges : shape (N,), scalar dtype
        Per-atom monopole charges.
    dipoles : shape (N,), vec3 dtype
        Per-atom dipole moments.
    quadrupoles : shape (N,), mat33 dtype
        Per-atom symmetric Cartesian quadrupole tensors.
    cells : shape (B,), mat33 dtype
        Per-system unit cells, indexed by ``atom_batch_idx``.
    idx_j : shape (nnz,), int32
        CSR neighbor column indices.
    neighbor_ptr : shape (N + 1,), int32
        CSR row pointers.
    atom_batch_idx : shape (N,), int32
        System index for each atom.
    unit_shifts : shape (nnz,), vec3i
        Integer lattice shifts per pair.
    sigma : shape (1,), scalar dtype
        Gaussian charge width.
    alpha : shape (1,), scalar dtype
        Ewald splitting parameter.
    grad_energies : shape (N,), float64
        Upstream per-atom energy weights.
    grad_cell : shape (B,), mat33 dtype
        OUTPUT: per-system :math:`3\times3` cell gradients (atomic).
    device : str, keyword-only
        Warp device for the launch.
    half_neighbor_list : bool, keyword-only, default False
        ``True`` half list (scale 1.0); ``False`` full list (scale 0.5).
    """
    overload = _get_overload("csr", True, positions.dtype, charges.dtype)
    n_atoms = positions.shape[0]
    scale_arr = _make_scale_array(half_neighbor_list, device)
    wp.launch(
        overload,
        dim=n_atoms,
        inputs=[
            positions,
            charges,
            dipoles,
            quadrupoles,
            cells,
            idx_j,
            neighbor_ptr,
            atom_batch_idx,
            unit_shifts,
            sigma,
            alpha,
            scale_arr,
            grad_energies,
            grad_cell,
        ],
        device=device,
    )


# =============================================================================
# LMAX=1 cell gradient
# =============================================================================

_DIPOLE_CELL_GRAD_KERNEL_CACHE: dict = {}
_DIPOLE_CELL_GRAD_OVERLOAD_CACHE: dict = {}


def _make_dipole_cell_grad_kernel(storage: str, is_batch: bool):
    """Build the uninstantiated LMAX=1 cell-grad kernel for the storage/batch.

    Parameters
    ----------
    storage : str
        Neighbor-list storage layout; only ``"csr"`` is supported.
    is_batch : bool
        ``True`` returns the batched kernel; ``False`` the single-system one.

    Returns
    -------
    warp.Kernel
        The uninstantiated dipole cell-gradient kernel.

    Raises
    ------
    ValueError
        If ``storage`` is not ``"csr"``.
    """
    if storage != "csr":
        raise ValueError(f"unsupported storage {storage!r} (csr only)")
    return (
        _build_dipole_kernel_csr_batched()
        if is_batch
        else _build_dipole_kernel_csr_single()
    )


def _build_dipole_kernel_csr_single():
    def _kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        dipoles: wp.array(dtype=Any),
        cell: wp.array(dtype=Any),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        sigma: wp.array(dtype=Any),
        alpha: wp.array(dtype=Any),
        per_direction_scale: wp.array(dtype=wp.float64),
        grad_cell: wp.array(dtype=Any),
    ):
        r"""LMAX=1 single-system real-space cell-gradient scatter kernel.

        Same scatter pattern as the LMAX=2 kernel but using
        ``_dipole_pair_contribution_fused.dPE_dr_j`` (no quadrupole channel
        and no per-atom ``grad_energies`` weighting). One thread per atom ``i``
        accumulates :math:`\text{shift\_vec} \otimes \partial E_{pair}/\partial r_{vec}`
        (scaled by ``per_direction_scale``) into the per-system ``grad_cell``.

        Launch Grid
        -----------
        ``dim = n_atoms``. One thread per atom ``i``.

        Parameters
        ----------
        positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Cartesian atom positions.
        charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
            Per-atom monopole charges.
        dipoles : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Per-atom dipole moments.
        cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
            Single-system unit cell (transposed to map ``unit_shifts``).
        idx_j : wp.array, shape (nnz,), dtype=wp.int32
            CSR column indices (neighbor atom ``j``).
        neighbor_ptr : wp.array, shape (N + 1,), dtype=wp.int32
            CSR row pointers.
        unit_shifts : wp.array, shape (nnz,), dtype=wp.vec3i
            Integer lattice shifts :math:`\in \mathbb{Z}^3`.
        sigma : wp.array, shape (1,), dtype=wp.float32 or wp.float64
            Gaussian charge width.
        alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
            Ewald splitting parameter.
        per_direction_scale : wp.array, shape (1,), dtype=wp.float64
            Per-direction scale (1.0 half list, 0.5 full list).
        grad_cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
            OUTPUT: per-system :math:`3\times3` cell gradient (atomic).
        """
        atom_i = wp.tid()
        sigma_ = wp.float64(sigma[0])
        alpha_ = wp.float64(alpha[0])
        ab = _gto_ewald_ab(sigma_, alpha_)
        a_coef = ab[0]
        b_coef = ab[1]
        gc_template = grad_cell[0]
        scale = per_direction_scale[0]
        cell_t = wp.transpose(cell[0])

        pos_i = positions[atom_i]
        qi = wp.float64(charges[atom_i])
        mu_i_n = dipoles[atom_i]
        mu_i = wp.vec3d(
            wp.float64(mu_i_n[0]), wp.float64(mu_i_n[1]), wp.float64(mu_i_n[2])
        )

        k_start = neighbor_ptr[atom_i]
        k_end = neighbor_ptr[atom_i + 1]
        for k in range(k_start, k_end):
            j = idx_j[k]
            shift_vec = unit_shifts[k]
            qj = wp.float64(charges[j])
            pos_j = positions[j]
            mu_j_n = dipoles[j]
            mu_j = wp.vec3d(
                wp.float64(mu_j_n[0]), wp.float64(mu_j_n[1]), wp.float64(mu_j_n[2])
            )

            periodic_shift = cell_t * type(pos_i)(
                type(pos_i[0])(shift_vec[0]),
                type(pos_i[0])(shift_vec[1]),
                type(pos_i[0])(shift_vec[2]),
            )
            sep = pos_j - pos_i + periodic_shift
            distance = wp.float64(wp.length(sep))
            if distance > wp.float64(1e-8):
                r_vec = wp.vec3d(
                    wp.float64(sep[0]), wp.float64(sep[1]), wp.float64(sep[2])
                )
                contrib = _dipole_pair_contribution_fused(
                    r_vec,
                    distance,
                    qi,
                    mu_i,
                    qj,
                    mu_j,
                    a_coef,
                    b_coef,
                )
                dr = contrib.dPE_dr_j
                sh0 = wp.float64(shift_vec[0])
                sh1 = wp.float64(shift_vec[1])
                sh2 = wp.float64(shift_vec[2])
                s_dr0 = scale * dr[0]
                s_dr1 = scale * dr[1]
                s_dr2 = scale * dr[2]
                wp.atomic_add(
                    grad_cell,
                    0,
                    type(gc_template)(
                        type(gc_template[0, 0])(sh0 * s_dr0),
                        type(gc_template[0, 0])(sh0 * s_dr1),
                        type(gc_template[0, 0])(sh0 * s_dr2),
                        type(gc_template[0, 0])(sh1 * s_dr0),
                        type(gc_template[0, 0])(sh1 * s_dr1),
                        type(gc_template[0, 0])(sh1 * s_dr2),
                        type(gc_template[0, 0])(sh2 * s_dr0),
                        type(gc_template[0, 0])(sh2 * s_dr1),
                        type(gc_template[0, 0])(sh2 * s_dr2),
                    ),
                )

    _kernel.__name__ = "_multipole_real_space_dipole_csr_cell_grad_kernel"
    return wp.kernel(_kernel, enable_backward=False)


def _build_dipole_kernel_csr_batched():
    def _kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        dipoles: wp.array(dtype=Any),
        cells: wp.array(dtype=Any),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        atom_batch_idx: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        sigma: wp.array(dtype=Any),
        alpha: wp.array(dtype=Any),
        per_direction_scale: wp.array(dtype=wp.float64),
        grad_cell: wp.array(dtype=Any),
    ):
        r"""Batched LMAX=1 real-space cell-gradient scatter kernel.

        Batched analog of the single-system dipole kernel: atom ``i`` maps to
        system ``b = atom_batch_idx[i]`` with cell ``cells[b]``; the per-pair
        outer product is scattered into ``grad_cell[b]``.

        Launch Grid
        -----------
        ``dim = n_atoms`` (all systems concatenated). One thread per atom ``i``.

        Parameters
        ----------
        positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Cartesian atom positions (all systems).
        charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
            Per-atom monopole charges.
        dipoles : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Per-atom dipole moments.
        cells : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
            Per-system unit cells, indexed by ``atom_batch_idx``.
        idx_j : wp.array, shape (nnz,), dtype=wp.int32
            CSR column indices (neighbor atom ``j``).
        neighbor_ptr : wp.array, shape (N + 1,), dtype=wp.int32
            CSR row pointers.
        atom_batch_idx : wp.array, shape (N,), dtype=wp.int32
            System index ``b`` for each atom.
        unit_shifts : wp.array, shape (nnz,), dtype=wp.vec3i
            Integer lattice shifts :math:`\in \mathbb{Z}^3`.
        sigma : wp.array, shape (1,), dtype=wp.float32 or wp.float64
            Gaussian charge width.
        alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
            Ewald splitting parameter.
        per_direction_scale : wp.array, shape (1,), dtype=wp.float64
            Per-direction scale (1.0 half list, 0.5 full list).
        grad_cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
            OUTPUT: per-system :math:`3\times3` cell gradients (atomic).
        """
        atom_i = wp.tid()
        b = atom_batch_idx[atom_i]
        sigma_ = wp.float64(sigma[0])
        alpha_ = wp.float64(alpha[0])
        ab = _gto_ewald_ab(sigma_, alpha_)
        a_coef = ab[0]
        b_coef = ab[1]
        gc_template = grad_cell[0]
        scale = per_direction_scale[0]
        cell_t = wp.transpose(cells[b])

        pos_i = positions[atom_i]
        qi = wp.float64(charges[atom_i])
        mu_i_n = dipoles[atom_i]
        mu_i = wp.vec3d(
            wp.float64(mu_i_n[0]), wp.float64(mu_i_n[1]), wp.float64(mu_i_n[2])
        )

        k_start = neighbor_ptr[atom_i]
        k_end = neighbor_ptr[atom_i + 1]
        for k in range(k_start, k_end):
            j = idx_j[k]
            shift_vec = unit_shifts[k]
            qj = wp.float64(charges[j])
            pos_j = positions[j]
            mu_j_n = dipoles[j]
            mu_j = wp.vec3d(
                wp.float64(mu_j_n[0]), wp.float64(mu_j_n[1]), wp.float64(mu_j_n[2])
            )

            periodic_shift = cell_t * type(pos_i)(
                type(pos_i[0])(shift_vec[0]),
                type(pos_i[0])(shift_vec[1]),
                type(pos_i[0])(shift_vec[2]),
            )
            sep = pos_j - pos_i + periodic_shift
            distance = wp.float64(wp.length(sep))
            if distance > wp.float64(1e-8):
                r_vec = wp.vec3d(
                    wp.float64(sep[0]), wp.float64(sep[1]), wp.float64(sep[2])
                )
                contrib = _dipole_pair_contribution_fused(
                    r_vec,
                    distance,
                    qi,
                    mu_i,
                    qj,
                    mu_j,
                    a_coef,
                    b_coef,
                )
                dr = contrib.dPE_dr_j
                sh0 = wp.float64(shift_vec[0])
                sh1 = wp.float64(shift_vec[1])
                sh2 = wp.float64(shift_vec[2])
                s_dr0 = scale * dr[0]
                s_dr1 = scale * dr[1]
                s_dr2 = scale * dr[2]
                wp.atomic_add(
                    grad_cell,
                    b,
                    type(gc_template)(
                        type(gc_template[0, 0])(sh0 * s_dr0),
                        type(gc_template[0, 0])(sh0 * s_dr1),
                        type(gc_template[0, 0])(sh0 * s_dr2),
                        type(gc_template[0, 0])(sh1 * s_dr0),
                        type(gc_template[0, 0])(sh1 * s_dr1),
                        type(gc_template[0, 0])(sh1 * s_dr2),
                        type(gc_template[0, 0])(sh2 * s_dr0),
                        type(gc_template[0, 0])(sh2 * s_dr1),
                        type(gc_template[0, 0])(sh2 * s_dr2),
                    ),
                )

    _kernel.__name__ = "_batch_multipole_real_space_dipole_csr_cell_grad_kernel"
    return wp.kernel(_kernel, enable_backward=False)


def _sig_dipole_csr_single(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=v),
        wp.array(dtype=m),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.vec3i),
        wp.array(dtype=t),
        wp.array(dtype=t),
        wp.array(dtype=wp.float64),
        wp.array(dtype=m),
    ]


def _sig_dipole_csr_batched(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=v),
        wp.array(dtype=m),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.vec3i),
        wp.array(dtype=t),
        wp.array(dtype=t),
        wp.array(dtype=wp.float64),
        wp.array(dtype=m),
    ]


_DIPOLE_SIG_BUILDERS = {
    ("csr", False): _sig_dipole_csr_single,
    ("csr", True): _sig_dipole_csr_batched,
}


def _get_dipole_overload(storage, is_batch, vec_dtype, scalar_dtype):
    kernel_key = (storage, is_batch)
    if kernel_key not in _DIPOLE_CELL_GRAD_KERNEL_CACHE:
        _DIPOLE_CELL_GRAD_KERNEL_CACHE[kernel_key] = _make_dipole_cell_grad_kernel(
            storage, is_batch
        )
    kernel = _DIPOLE_CELL_GRAD_KERNEL_CACHE[kernel_key]
    overload_key = (storage, is_batch, vec_dtype)
    if overload_key not in _DIPOLE_CELL_GRAD_OVERLOAD_CACHE:
        sig = _DIPOLE_SIG_BUILDERS[(storage, is_batch)](vec_dtype, scalar_dtype)
        _DIPOLE_CELL_GRAD_OVERLOAD_CACHE[overload_key] = wp.overload(kernel, sig)
    return _DIPOLE_CELL_GRAD_OVERLOAD_CACHE[overload_key]


def multipole_real_space_dipole_csr_cell_grad(
    positions,
    charges,
    dipoles,
    cell,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma,
    alpha,
    grad_cell,
    *,
    device: str,
    half_neighbor_list: bool = False,
):
    r"""CSR single-system LMAX=1 cell-gradient launcher.

    Parameters
    ----------
    positions : shape (N,), vec3 dtype
        Cartesian atom positions.
    charges : shape (N,), scalar dtype
        Per-atom monopole charges.
    dipoles : shape (N,), vec3 dtype
        Per-atom dipole moments.
    cell : shape (1,), mat33 dtype
        Single-system unit cell.
    idx_j : shape (nnz,), int32
        CSR neighbor column indices.
    neighbor_ptr : shape (N + 1,), int32
        CSR row pointers.
    unit_shifts : shape (nnz,), vec3i
        Integer lattice shifts per pair.
    sigma : shape (1,), scalar dtype
        Gaussian charge width.
    alpha : shape (1,), scalar dtype
        Ewald splitting parameter.
    grad_cell : shape (1,), mat33 dtype
        OUTPUT: per-system :math:`3\times3` cell gradient (atomic).
    device : str, keyword-only
        Warp device for the launch.
    half_neighbor_list : bool, keyword-only, default False
        ``True`` half list (scale 1.0); ``False`` full list (scale 0.5).
    """
    overload = _get_dipole_overload("csr", False, positions.dtype, charges.dtype)
    n_atoms = positions.shape[0]
    scale_arr = _make_scale_array(half_neighbor_list, device)
    wp.launch(
        overload,
        dim=n_atoms,
        inputs=[
            positions,
            charges,
            dipoles,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma,
            alpha,
            scale_arr,
            grad_cell,
        ],
        device=device,
    )


def batch_multipole_real_space_dipole_csr_cell_grad(
    positions,
    charges,
    dipoles,
    cells,
    idx_j,
    neighbor_ptr,
    atom_batch_idx,
    unit_shifts,
    sigma,
    alpha,
    grad_cell,
    *,
    device: str,
    half_neighbor_list: bool = False,
):
    r"""Batched CSR LMAX=1 cell-gradient launcher.

    Parameters
    ----------
    positions : shape (N,), vec3 dtype
        Cartesian atom positions (all systems).
    charges : shape (N,), scalar dtype
        Per-atom monopole charges.
    dipoles : shape (N,), vec3 dtype
        Per-atom dipole moments.
    cells : shape (B,), mat33 dtype
        Per-system unit cells, indexed by ``atom_batch_idx``.
    idx_j : shape (nnz,), int32
        CSR neighbor column indices.
    neighbor_ptr : shape (N + 1,), int32
        CSR row pointers.
    atom_batch_idx : shape (N,), int32
        System index for each atom.
    unit_shifts : shape (nnz,), vec3i
        Integer lattice shifts per pair.
    sigma : shape (1,), scalar dtype
        Gaussian charge width.
    alpha : shape (1,), scalar dtype
        Ewald splitting parameter.
    grad_cell : shape (B,), mat33 dtype
        OUTPUT: per-system :math:`3\times3` cell gradients (atomic).
    device : str, keyword-only
        Warp device for the launch.
    half_neighbor_list : bool, keyword-only, default False
        ``True`` half list (scale 1.0); ``False`` full list (scale 0.5).
    """
    overload = _get_dipole_overload("csr", True, positions.dtype, charges.dtype)
    n_atoms = positions.shape[0]
    scale_arr = _make_scale_array(half_neighbor_list, device)
    wp.launch(
        overload,
        dim=n_atoms,
        inputs=[
            positions,
            charges,
            dipoles,
            cells,
            idx_j,
            neighbor_ptr,
            atom_batch_idx,
            unit_shifts,
            sigma,
            alpha,
            scale_arr,
            grad_cell,
        ],
        device=device,
    )


# =============================================================================
# LMAX=0 cell gradient
# =============================================================================
#
# LMAX=0 inlines the qq position gradient
# ``dPE/dr_j = -qi*qj*(a_scalar/r)*r_vec``, with ``a_scalar`` from
# ``_gto_ewald_monopole_pair_terms_fused``.

_MONOPOLE_CELL_GRAD_KERNEL_CACHE: dict = {}
_MONOPOLE_CELL_GRAD_OVERLOAD_CACHE: dict = {}


def _make_monopole_cell_grad_kernel(storage: str, is_batch: bool):
    """Build the uninstantiated LMAX=0 cell-grad kernel for the storage/batch.

    Parameters
    ----------
    storage : str
        Neighbor-list storage layout; only ``"csr"`` is supported.
    is_batch : bool
        ``True`` returns the batched kernel; ``False`` the single-system one.

    Returns
    -------
    warp.Kernel
        The uninstantiated monopole cell-gradient kernel.

    Raises
    ------
    ValueError
        If ``storage`` is not ``"csr"``.
    """
    if storage != "csr":
        raise ValueError(f"unsupported storage {storage!r} (csr only)")
    return (
        _build_monopole_kernel_csr_batched()
        if is_batch
        else _build_monopole_kernel_csr_single()
    )


def _build_monopole_kernel_csr_single():
    def _kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        cell: wp.array(dtype=Any),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        sigma: wp.array(dtype=Any),
        alpha: wp.array(dtype=Any),
        per_direction_scale: wp.array(dtype=wp.float64),
        grad_cell: wp.array(dtype=Any),
    ):
        r"""LMAX=0 single-system real-space cell-gradient scatter kernel.

        Inlines the charge-charge position gradient
        :math:`\partial E_{pair}/\partial r_j = -q_i q_j (a/r)\, r_{vec}`
        (with ``a`` the GTO-Ewald monopole radial term) and scatters the
        per-pair outer product :math:`\text{shift\_vec} \otimes \partial E_{pair}/\partial r_{vec}`
        (scaled by ``per_direction_scale``) into the per-system ``grad_cell``.

        Launch Grid
        -----------
        ``dim = n_atoms``. One thread per atom ``i``.

        Parameters
        ----------
        positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Cartesian atom positions.
        charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
            Per-atom monopole charges.
        cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
            Single-system unit cell (transposed to map ``unit_shifts``).
        idx_j : wp.array, shape (nnz,), dtype=wp.int32
            CSR column indices (neighbor atom ``j``).
        neighbor_ptr : wp.array, shape (N + 1,), dtype=wp.int32
            CSR row pointers.
        unit_shifts : wp.array, shape (nnz,), dtype=wp.vec3i
            Integer lattice shifts :math:`\in \mathbb{Z}^3`.
        sigma : wp.array, shape (1,), dtype=wp.float32 or wp.float64
            Gaussian charge width.
        alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
            Ewald splitting parameter.
        per_direction_scale : wp.array, shape (1,), dtype=wp.float64
            Per-direction scale (1.0 half list, 0.5 full list).
        grad_cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
            OUTPUT: per-system :math:`3\times3` cell gradient (atomic).
        """
        atom_i = wp.tid()
        sigma_ = wp.float64(sigma[0])
        alpha_ = wp.float64(alpha[0])
        ab = _gto_ewald_ab(sigma_, alpha_)
        a_coef = ab[0]
        b_coef = ab[1]
        gc_template = grad_cell[0]
        scale = per_direction_scale[0]
        cell_t = wp.transpose(cell[0])

        pos_i = positions[atom_i]
        qi = wp.float64(charges[atom_i])

        k_start = neighbor_ptr[atom_i]
        k_end = neighbor_ptr[atom_i + 1]
        for k in range(k_start, k_end):
            j = idx_j[k]
            shift_vec = unit_shifts[k]
            qj = wp.float64(charges[j])
            pos_j = positions[j]

            periodic_shift = cell_t * type(pos_i)(
                type(pos_i[0])(shift_vec[0]),
                type(pos_i[0])(shift_vec[1]),
                type(pos_i[0])(shift_vec[2]),
            )
            sep = pos_j - pos_i + periodic_shift
            distance = wp.float64(wp.length(sep))
            if distance > wp.float64(1e-8):
                radial = _gto_ewald_monopole_pair_terms_fused(distance, a_coef, b_coef)
                inv_r = wp.float64(1.0) / distance
                coeff = -qi * qj * radial.a_scalar * inv_r
                dr0 = coeff * wp.float64(sep[0])
                dr1 = coeff * wp.float64(sep[1])
                dr2 = coeff * wp.float64(sep[2])
                sh0 = wp.float64(shift_vec[0])
                sh1 = wp.float64(shift_vec[1])
                sh2 = wp.float64(shift_vec[2])
                s_dr0 = scale * dr0
                s_dr1 = scale * dr1
                s_dr2 = scale * dr2
                wp.atomic_add(
                    grad_cell,
                    0,
                    type(gc_template)(
                        type(gc_template[0, 0])(sh0 * s_dr0),
                        type(gc_template[0, 0])(sh0 * s_dr1),
                        type(gc_template[0, 0])(sh0 * s_dr2),
                        type(gc_template[0, 0])(sh1 * s_dr0),
                        type(gc_template[0, 0])(sh1 * s_dr1),
                        type(gc_template[0, 0])(sh1 * s_dr2),
                        type(gc_template[0, 0])(sh2 * s_dr0),
                        type(gc_template[0, 0])(sh2 * s_dr1),
                        type(gc_template[0, 0])(sh2 * s_dr2),
                    ),
                )

    _kernel.__name__ = "_multipole_real_space_monopole_csr_cell_grad_kernel"
    return wp.kernel(_kernel, enable_backward=False)


def _build_monopole_kernel_csr_batched():
    def _kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        cells: wp.array(dtype=Any),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        atom_batch_idx: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        sigma: wp.array(dtype=Any),
        alpha: wp.array(dtype=Any),
        per_direction_scale: wp.array(dtype=wp.float64),
        grad_cell: wp.array(dtype=Any),
    ):
        r"""Batched LMAX=0 real-space cell-gradient scatter kernel.

        Batched analog of the single-system monopole kernel: atom ``i`` maps
        to system ``b = atom_batch_idx[i]`` with cell ``cells[b]``; the inlined
        charge-charge :math:`\text{shift\_vec} \otimes \partial E_{pair}/\partial r_{vec}`
        outer product is scattered into ``grad_cell[b]``.

        Launch Grid
        -----------
        ``dim = n_atoms`` (all systems concatenated). One thread per atom ``i``.

        Parameters
        ----------
        positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Cartesian atom positions (all systems).
        charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
            Per-atom monopole charges.
        cells : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
            Per-system unit cells, indexed by ``atom_batch_idx``.
        idx_j : wp.array, shape (nnz,), dtype=wp.int32
            CSR column indices (neighbor atom ``j``).
        neighbor_ptr : wp.array, shape (N + 1,), dtype=wp.int32
            CSR row pointers.
        atom_batch_idx : wp.array, shape (N,), dtype=wp.int32
            System index ``b`` for each atom.
        unit_shifts : wp.array, shape (nnz,), dtype=wp.vec3i
            Integer lattice shifts :math:`\in \mathbb{Z}^3`.
        sigma : wp.array, shape (1,), dtype=wp.float32 or wp.float64
            Gaussian charge width.
        alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
            Ewald splitting parameter.
        per_direction_scale : wp.array, shape (1,), dtype=wp.float64
            Per-direction scale (1.0 half list, 0.5 full list).
        grad_cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
            OUTPUT: per-system :math:`3\times3` cell gradients (atomic).
        """
        atom_i = wp.tid()
        b = atom_batch_idx[atom_i]
        sigma_ = wp.float64(sigma[0])
        alpha_ = wp.float64(alpha[0])
        ab = _gto_ewald_ab(sigma_, alpha_)
        a_coef = ab[0]
        b_coef = ab[1]
        gc_template = grad_cell[0]
        scale = per_direction_scale[0]
        cell_t = wp.transpose(cells[b])

        pos_i = positions[atom_i]
        qi = wp.float64(charges[atom_i])

        k_start = neighbor_ptr[atom_i]
        k_end = neighbor_ptr[atom_i + 1]
        for k in range(k_start, k_end):
            j = idx_j[k]
            shift_vec = unit_shifts[k]
            qj = wp.float64(charges[j])
            pos_j = positions[j]

            periodic_shift = cell_t * type(pos_i)(
                type(pos_i[0])(shift_vec[0]),
                type(pos_i[0])(shift_vec[1]),
                type(pos_i[0])(shift_vec[2]),
            )
            sep = pos_j - pos_i + periodic_shift
            distance = wp.float64(wp.length(sep))
            if distance > wp.float64(1e-8):
                radial = _gto_ewald_monopole_pair_terms_fused(distance, a_coef, b_coef)
                inv_r = wp.float64(1.0) / distance
                coeff = -qi * qj * radial.a_scalar * inv_r
                dr0 = coeff * wp.float64(sep[0])
                dr1 = coeff * wp.float64(sep[1])
                dr2 = coeff * wp.float64(sep[2])
                sh0 = wp.float64(shift_vec[0])
                sh1 = wp.float64(shift_vec[1])
                sh2 = wp.float64(shift_vec[2])
                s_dr0 = scale * dr0
                s_dr1 = scale * dr1
                s_dr2 = scale * dr2
                wp.atomic_add(
                    grad_cell,
                    b,
                    type(gc_template)(
                        type(gc_template[0, 0])(sh0 * s_dr0),
                        type(gc_template[0, 0])(sh0 * s_dr1),
                        type(gc_template[0, 0])(sh0 * s_dr2),
                        type(gc_template[0, 0])(sh1 * s_dr0),
                        type(gc_template[0, 0])(sh1 * s_dr1),
                        type(gc_template[0, 0])(sh1 * s_dr2),
                        type(gc_template[0, 0])(sh2 * s_dr0),
                        type(gc_template[0, 0])(sh2 * s_dr1),
                        type(gc_template[0, 0])(sh2 * s_dr2),
                    ),
                )

    _kernel.__name__ = "_batch_multipole_real_space_monopole_csr_cell_grad_kernel"
    return wp.kernel(_kernel, enable_backward=False)


def _sig_monopole_csr_single(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=m),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.vec3i),
        wp.array(dtype=t),
        wp.array(dtype=t),
        wp.array(dtype=wp.float64),
        wp.array(dtype=m),
    ]


def _sig_monopole_csr_batched(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=m),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.vec3i),
        wp.array(dtype=t),
        wp.array(dtype=t),
        wp.array(dtype=wp.float64),
        wp.array(dtype=m),
    ]


_MONOPOLE_SIG_BUILDERS = {
    ("csr", False): _sig_monopole_csr_single,
    ("csr", True): _sig_monopole_csr_batched,
}


def _get_monopole_overload(storage, is_batch, vec_dtype, scalar_dtype):
    kernel_key = (storage, is_batch)
    if kernel_key not in _MONOPOLE_CELL_GRAD_KERNEL_CACHE:
        _MONOPOLE_CELL_GRAD_KERNEL_CACHE[kernel_key] = _make_monopole_cell_grad_kernel(
            storage, is_batch
        )
    kernel = _MONOPOLE_CELL_GRAD_KERNEL_CACHE[kernel_key]
    overload_key = (storage, is_batch, vec_dtype)
    if overload_key not in _MONOPOLE_CELL_GRAD_OVERLOAD_CACHE:
        sig = _MONOPOLE_SIG_BUILDERS[(storage, is_batch)](vec_dtype, scalar_dtype)
        _MONOPOLE_CELL_GRAD_OVERLOAD_CACHE[overload_key] = wp.overload(kernel, sig)
    return _MONOPOLE_CELL_GRAD_OVERLOAD_CACHE[overload_key]


def multipole_real_space_monopole_csr_cell_grad(
    positions,
    charges,
    cell,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma,
    alpha,
    grad_cell,
    *,
    device: str,
    half_neighbor_list: bool = False,
):
    r"""CSR single-system LMAX=0 cell-gradient launcher.

    Parameters
    ----------
    positions : shape (N,), vec3 dtype
        Cartesian atom positions.
    charges : shape (N,), scalar dtype
        Per-atom monopole charges.
    cell : shape (1,), mat33 dtype
        Single-system unit cell.
    idx_j : shape (nnz,), int32
        CSR neighbor column indices.
    neighbor_ptr : shape (N + 1,), int32
        CSR row pointers.
    unit_shifts : shape (nnz,), vec3i
        Integer lattice shifts per pair.
    sigma : shape (1,), scalar dtype
        Gaussian charge width.
    alpha : shape (1,), scalar dtype
        Ewald splitting parameter.
    grad_cell : shape (1,), mat33 dtype
        OUTPUT: per-system :math:`3\times3` cell gradient (atomic).
    device : str, keyword-only
        Warp device for the launch.
    half_neighbor_list : bool, keyword-only, default False
        ``True`` half list (scale 1.0); ``False`` full list (scale 0.5).
    """
    overload = _get_monopole_overload("csr", False, positions.dtype, charges.dtype)
    n_atoms = positions.shape[0]
    scale_arr = _make_scale_array(half_neighbor_list, device)
    wp.launch(
        overload,
        dim=n_atoms,
        inputs=[
            positions,
            charges,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma,
            alpha,
            scale_arr,
            grad_cell,
        ],
        device=device,
    )


def batch_multipole_real_space_monopole_csr_cell_grad(
    positions,
    charges,
    cells,
    idx_j,
    neighbor_ptr,
    atom_batch_idx,
    unit_shifts,
    sigma,
    alpha,
    grad_cell,
    *,
    device: str,
    half_neighbor_list: bool = False,
):
    r"""Batched CSR LMAX=0 cell-gradient launcher.

    Parameters
    ----------
    positions : shape (N,), vec3 dtype
        Cartesian atom positions (all systems).
    charges : shape (N,), scalar dtype
        Per-atom monopole charges.
    cells : shape (B,), mat33 dtype
        Per-system unit cells, indexed by ``atom_batch_idx``.
    idx_j : shape (nnz,), int32
        CSR neighbor column indices.
    neighbor_ptr : shape (N + 1,), int32
        CSR row pointers.
    atom_batch_idx : shape (N,), int32
        System index for each atom.
    unit_shifts : shape (nnz,), vec3i
        Integer lattice shifts per pair.
    sigma : shape (1,), scalar dtype
        Gaussian charge width.
    alpha : shape (1,), scalar dtype
        Ewald splitting parameter.
    grad_cell : shape (B,), mat33 dtype
        OUTPUT: per-system :math:`3\times3` cell gradients (atomic).
    device : str, keyword-only
        Warp device for the launch.
    half_neighbor_list : bool, keyword-only, default False
        ``True`` half list (scale 1.0); ``False`` full list (scale 0.5).
    """
    overload = _get_monopole_overload("csr", True, positions.dtype, charges.dtype)
    n_atoms = positions.shape[0]
    scale_arr = _make_scale_array(half_neighbor_list, device)
    wp.launch(
        overload,
        dim=n_atoms,
        inputs=[
            positions,
            charges,
            cells,
            idx_j,
            neighbor_ptr,
            atom_batch_idx,
            unit_shifts,
            sigma,
            alpha,
            scale_arr,
            grad_cell,
        ],
        device=device,
    )

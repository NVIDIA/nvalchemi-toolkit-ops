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
PyTorch Adapter for FIRE2 Optimizer
====================================

Thin wrapper that accepts PyTorch tensors, allocates scratch buffers via
PyTorch's CUDA caching allocator, and calls the pure-Warp FIRE2 kernels.

Entry points:

- :func:`fire2_step_coord` -- coordinate-only optimization.
- :func:`fire2_step_coord_cell` -- variable-cell optimization
  (coordinates + cell DOFs).  Packs atomic and cell DOFs into an
  interleaved extended velocity/force representation, runs FIRE2
  mixing on the generalized DOFs, then applies a coupled atomic/cell
  position update.
- :func:`fire2_step_extended` -- run FIRE2 directly on caller-managed
  extended arrays (no per-step pack/unpack overhead).

Modifies inputs in-place. Scratch buffers and static metadata
(``atom_ptr``, ``ext_atom_ptr``, ``ext_batch_idx``) can be passed in
for reuse across steps, or left as ``None`` to allocate internally
each call.  See :func:`fire2_step_coord_cell` docstring for
pre-computation recipes.
"""

from __future__ import annotations

import warnings

import torch
import warp as wp

from nvalchemiops.batch_utils import atom_ptr_to_batch_idx, batch_idx_to_atom_ptr
from nvalchemiops.dynamics.optimizers.fire2 import (
    fire2_reduce,
    fire2_step,
    fire2_update,
)
from nvalchemiops.dynamics.utils.cell_filter import (
    _apply_fire2_coord_cell_step,
    _fire2_coord_cell_clamp_apply,
    _fire2_coord_cell_compute_max_norm,
    extend_atom_ptr,
    pack_forces_with_cell,
    pack_velocities_with_cell,
    unpack_velocities_with_cell,
)

# Torch dtype -> Warp dtype mappings
_TORCH_TO_WP_VEC = {torch.float32: wp.vec3f, torch.float64: wp.vec3d}
_TORCH_TO_WP_MAT = {torch.float32: wp.mat33f, torch.float64: wp.mat33d}


def _alloc_or_zero(
    buf: torch.Tensor | None, size: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Return a zeroed buffer, allocating one if *buf* is ``None``."""
    if buf is None:
        return torch.zeros(size, dtype=dtype, device=device)
    buf.zero_()
    return buf


def _coord_cell_ext_metadata(
    batch_idx: torch.Tensor,
    M: int,
    device: torch.device,
    wp_device,
    *,
    atom_ptr: torch.Tensor | None,
    ext_atom_ptr: torch.Tensor | None,
    ext_batch_idx: torch.Tensor | None,
):
    """Return the packed-layout metadata for variable-cell FIRE2.

    Computes ``atom_ptr`` / ``ext_atom_ptr`` / ``ext_batch_idx`` (as Warp arrays)
    for the interleaved [atoms, 2 cell rows] layout, filling any not supplied.
    """
    N = batch_idx.shape[0]
    N_ext = N + 2 * M
    wp_bidx = wp.from_torch(batch_idx.detach(), dtype=wp.int32)
    if atom_ptr is None:
        atom_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
        atom_counts = torch.zeros(M, dtype=torch.int32, device=device)
        batch_idx_to_atom_ptr(
            wp_bidx,
            wp.from_torch(atom_counts, dtype=wp.int32),
            wp.from_torch(atom_ptr, dtype=wp.int32),
        )
    wp_atom_ptr = wp.from_torch(atom_ptr, dtype=wp.int32)
    if ext_atom_ptr is None:
        ext_atom_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
        extend_atom_ptr(
            wp_atom_ptr,
            wp.from_torch(ext_atom_ptr, dtype=wp.int32),
            device=wp_device,
        )
    wp_ext_atom_ptr = wp.from_torch(ext_atom_ptr, dtype=wp.int32)
    if ext_batch_idx is None:
        ext_batch_idx = torch.empty(N_ext, dtype=torch.int32, device=device)
        atom_ptr_to_batch_idx(
            wp_ext_atom_ptr,
            wp.from_torch(ext_batch_idx, dtype=wp.int32),
        )
    wp_ext_batch_idx = wp.from_torch(ext_batch_idx, dtype=wp.int32)
    return atom_ptr, wp_atom_ptr, wp_ext_atom_ptr, wp_ext_batch_idx


def _coord_cell_mix_impl(
    positions,
    velocities,
    forces,
    cell,
    cell_velocities,
    cell_force,
    batch_idx,
    alpha,
    dt,
    nsteps_inc,
    *,
    atom_ptr,
    ext_atom_ptr,
    ext_velocities,
    ext_forces,
    ext_batch_idx,
    vf,
    v_sumsq,
    f_sumsq,
    max_norm,
    delaystep,
    dtgrow,
    dtshrink,
    alphashrink,
    alpha0,
    tmax,
    tmin,
    cell_force_scale,
    compute_reductions,
    ext_positions,
):
    """Shared front half of the variable-cell FIRE2 step.

    Packs atomic + cell DOFs, runs the FIRE2 reduction (gated by
    ``compute_reductions``) and velocity mix on the packed DOFs, and unpacks the
    mixed velocities. Does **not** apply positions/cell. Returns the Warp
    handles the apply phase needs, or ``None`` for an empty system.
    """
    dtype = positions.dtype
    device = positions.device
    N = positions.shape[0]
    M = alpha.shape[0]
    N_ext = N + 2 * M
    vec_type = _TORCH_TO_WP_VEC[dtype]
    mat_type = _TORCH_TO_WP_MAT[dtype]
    wp_device = wp.device_from_torch(device)

    if cell_force_scale <= 0.0:
        raise ValueError("cell_force_scale must be positive")
    if ext_positions is not None:
        warnings.warn(
            "fire2_step_coord_cell(..., ext_positions=...) is deprecated and ignored.",
            DeprecationWarning,
            stacklevel=3,
        )
    if N == 0:
        for buf in (vf, v_sumsq, f_sumsq, max_norm):
            if buf is not None:
                buf.zero_()
        return None

    if ext_velocities is None:
        ext_velocities = torch.empty(N_ext, 3, dtype=dtype, device=device)
    if ext_forces is None:
        ext_forces = torch.empty(N_ext, 3, dtype=dtype, device=device)

    if compute_reductions:
        vf = _alloc_or_zero(vf, M, dtype, device)
        v_sumsq = _alloc_or_zero(v_sumsq, M, dtype, device)
        f_sumsq = _alloc_or_zero(f_sumsq, M, dtype, device)
    elif vf is None or v_sumsq is None or f_sumsq is None:
        raise ValueError(
            "vf, v_sumsq, f_sumsq must be provided when compute_reductions=False"
        )
    max_norm = _alloc_or_zero(max_norm, M, dtype, device)

    wp_pos = wp.from_torch(positions.detach(), dtype=vec_type)
    wp_vel = wp.from_torch(velocities.detach(), dtype=vec_type)
    wp_forces = wp.from_torch(forces.detach(), dtype=vec_type)
    wp_cell = wp.from_torch(cell.detach(), dtype=mat_type)
    wp_cell_vel = wp.from_torch(cell_velocities.detach(), dtype=mat_type)
    wp_bidx = wp.from_torch(batch_idx.detach(), dtype=wp.int32)
    wp_ext_vel = wp.from_torch(ext_velocities, dtype=vec_type)
    wp_ext_forces = wp.from_torch(ext_forces, dtype=vec_type)

    atom_ptr, wp_atom_ptr, wp_ext_atom_ptr, wp_ext_batch_idx = _coord_cell_ext_metadata(
        batch_idx,
        M,
        device,
        wp_device,
        atom_ptr=atom_ptr,
        ext_atom_ptr=ext_atom_ptr,
        ext_batch_idx=ext_batch_idx,
    )

    atom_counts = atom_ptr[1:] - atom_ptr[:-1]
    if torch.any(atom_counts <= 0).item():
        raise ValueError("fire2_step_coord_cell requires at least one atom per system")
    cell_force_divisor = atom_counts.to(dtype=dtype).reshape(M, 1, 1) * cell_force_scale
    cell_force_work = (cell_force.detach() / cell_force_divisor).contiguous()
    wp_cell_force = wp.from_torch(cell_force_work, dtype=mat_type)

    if M == 1:
        pack_velocities_with_cell(wp_vel, wp_cell_vel, wp_ext_vel, device=wp_device)
        pack_forces_with_cell(wp_forces, wp_cell_force, wp_ext_forces, device=wp_device)
    else:
        pack_velocities_with_cell(
            wp_vel,
            wp_cell_vel,
            wp_ext_vel,
            wp_atom_ptr,
            wp_ext_atom_ptr,
            device=wp_device,
            batch_idx=wp_bidx,
        )
        pack_forces_with_cell(
            wp_forces,
            wp_cell_force,
            wp_ext_forces,
            wp_atom_ptr,
            wp_ext_atom_ptr,
            device=wp_device,
            batch_idx=wp_bidx,
        )

    wp_dt = wp.from_torch(dt.detach())
    wp_vf = wp.from_torch(vf)
    wp_max_norm = wp.from_torch(max_norm)
    fire2_update(
        wp_ext_vel,
        wp_ext_forces,
        wp_ext_batch_idx,
        wp.from_torch(alpha.detach()),
        wp_dt,
        wp.from_torch(nsteps_inc.detach(), dtype=wp.int32),
        wp_vf,
        wp.from_torch(v_sumsq),
        wp.from_torch(f_sumsq),
        wp_max_norm,
        delaystep=delaystep,
        dtgrow=dtgrow,
        dtshrink=dtshrink,
        alphashrink=alphashrink,
        alpha0=alpha0,
        tmax=tmax,
        tmin=tmin,
        compute_max_norm=False,
        compute_reductions=compute_reductions,
    )

    if M == 1:
        unpack_velocities_with_cell(
            wp_ext_vel, wp_vel, wp_cell_vel, num_atoms=N, device=wp_device
        )
    else:
        unpack_velocities_with_cell(
            wp_ext_vel,
            wp_vel,
            wp_cell_vel,
            atom_ptr=wp_atom_ptr,
            ext_atom_ptr=wp_ext_atom_ptr,
            device=wp_device,
            batch_idx=wp_bidx,
        )

    return (
        wp_pos,
        wp_vel,
        wp_cell,
        wp_cell_vel,
        wp_dt,
        wp_vf,
        wp_ext_batch_idx,
        wp_ext_atom_ptr,
        wp_max_norm,
        wp_device,
    )


def fire2_step_coord(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    batch_idx: torch.Tensor,
    alpha: torch.Tensor,
    dt: torch.Tensor,
    nsteps_inc: torch.Tensor,
    *,
    vf: torch.Tensor | None = None,
    v_sumsq: torch.Tensor | None = None,
    f_sumsq: torch.Tensor | None = None,
    max_norm: torch.Tensor | None = None,
    delaystep: int = 60,
    dtgrow: float = 1.05,
    dtshrink: float = 0.75,
    alphashrink: float = 0.985,
    alpha0: float = 0.09,
    tmax: float = 0.08,
    tmin: float = 0.005,
    maxstep: float = 0.1,
    compute_reductions: bool = True,
) -> None:
    """FIRE2 coordinate-only optimization step.

    Converts PyTorch tensors to Warp arrays (zero-copy) and delegates to
    the pure-Warp :func:`~nvalchemiops.dynamics.optimizers.fire2.fire2_step`.

    Modifies *positions*, *velocities*, *alpha*, *dt*, and *nsteps_inc*
    in-place.

    Parameters
    ----------
    positions : Tensor, shape (N, 3), dtype float32/float64
        Atomic positions.
    velocities : Tensor, shape (N, 3), dtype float32/float64
        Atomic velocities.
    forces : Tensor, shape (N, 3), dtype float32/float64
        Forces on atoms (read-only).
    batch_idx : Tensor, shape (N,), dtype int32
        Sorted system index per atom.  Must be non-decreasing;
        segmented reductions rely on contiguous atom ranges.
    alpha : Tensor, shape (M,), dtype float32/float64
        FIRE2 mixing parameter (one per system).
    dt : Tensor, shape (M,), dtype float32/float64
        Per-system timestep.
    nsteps_inc : Tensor, shape (M,), dtype int32
        Consecutive positive-power step counter.
    vf, v_sumsq, f_sumsq, max_norm : Tensor, shape (M,), optional
        Scratch buffers for per-system reductions.  Allocated and zeroed
        if ``None``; zeroed in-place if provided.  Pre-allocate and pass
        them in tight loops to avoid repeated allocation::

            M = alpha.shape[0]
            vf = torch.empty(M, dtype=positions.dtype,
                             device=positions.device)
            v_sumsq = torch.empty_like(vf)
            f_sumsq = torch.empty_like(vf)
            max_norm = torch.empty_like(vf)

    delaystep, dtgrow, dtshrink, alphashrink, alpha0, tmax, tmin, maxstep
        FIRE2 hyperparameters.  See
        :func:`~nvalchemiops.dynamics.optimizers.fire2.fire2_step` for
        defaults and descriptions.
    compute_reductions : bool, default True
        If True, recompute ``vf``/``v_sumsq``/``f_sumsq`` internally. If False,
        use the caller-supplied values in those buffers instead of recomputing
        them (they must be provided and are not zeroed); the ``maxstep`` clamp
        still uses this call's internally-computed ``max_norm``.

    Notes
    -----
    Default hyperparameters (from the underlying Warp implementation):
    ``delaystep=60``, ``dtgrow=1.05``, ``dtshrink=0.75``,
    ``alphashrink=0.985``, ``alpha0=0.09``, ``tmax=0.08``,
    ``tmin=0.005``, ``maxstep=0.1``.
    See :func:`nvalchemiops.dynamics.optimizers.fire2.fire2_step` for
    full descriptions.

    For variable-cell optimization (coordinates + cell DOFs), use
    :func:`fire2_step_coord_cell` instead.

    Examples
    --------
    Minimal single-step call:

    >>> fire2_step_coord(
    ...     positions, velocities, forces,
    ...     batch_idx, alpha, dt, nsteps_inc,
    ... )

    Tight optimization loop with pre-allocated scratch buffers:

    >>> M = alpha.shape[0]
    >>> vf = torch.empty(M, dtype=positions.dtype, device=positions.device)
    >>> v_sumsq = torch.empty_like(vf)
    >>> f_sumsq = torch.empty_like(vf)
    >>> max_norm = torch.empty_like(vf)
    >>> for step in range(num_steps):
    ...     fire2_step_coord(
    ...         positions, velocities, forces,
    ...         batch_idx, alpha, dt, nsteps_inc,
    ...         vf=vf, v_sumsq=v_sumsq,
    ...         f_sumsq=f_sumsq, max_norm=max_norm,
    ...     )
    """
    dtype = positions.dtype
    device = positions.device
    M = alpha.shape[0]
    vec_type = _TORCH_TO_WP_VEC[dtype]

    # Scratch buffers: allocate/zero when recomputing; require and preserve
    # them when the caller supplies precomputed reductions.
    if compute_reductions:
        vf = _alloc_or_zero(vf, M, dtype, device)
        v_sumsq = _alloc_or_zero(v_sumsq, M, dtype, device)
        f_sumsq = _alloc_or_zero(f_sumsq, M, dtype, device)
    elif vf is None or v_sumsq is None or f_sumsq is None:
        raise ValueError(
            "vf, v_sumsq, f_sumsq must be provided when compute_reductions=False"
        )
    max_norm = _alloc_or_zero(max_norm, M, dtype, device)

    if positions.shape[0] == 0:
        return

    # Delegate to the Warp-level fire2_step
    fire2_step(
        wp.from_torch(positions.detach(), dtype=vec_type),
        wp.from_torch(velocities.detach(), dtype=vec_type),
        wp.from_torch(forces.detach(), dtype=vec_type),
        wp.from_torch(batch_idx.detach(), dtype=wp.int32),
        wp.from_torch(alpha.detach()),
        wp.from_torch(dt.detach()),
        wp.from_torch(nsteps_inc.detach(), dtype=wp.int32),
        wp.from_torch(vf),
        wp.from_torch(v_sumsq),
        wp.from_torch(f_sumsq),
        wp.from_torch(max_norm),
        delaystep=delaystep,
        dtgrow=dtgrow,
        dtshrink=dtshrink,
        alphashrink=alphashrink,
        alpha0=alpha0,
        tmax=tmax,
        tmin=tmin,
        maxstep=maxstep,
        compute_reductions=compute_reductions,
    )


def fire2_step_coord_cell(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    cell: torch.Tensor,
    cell_velocities: torch.Tensor,
    cell_force: torch.Tensor,
    batch_idx: torch.Tensor,
    alpha: torch.Tensor,
    dt: torch.Tensor,
    nsteps_inc: torch.Tensor,
    *,
    atom_ptr: torch.Tensor | None = None,
    ext_atom_ptr: torch.Tensor | None = None,
    ext_positions: torch.Tensor | None = None,
    ext_velocities: torch.Tensor | None = None,
    ext_forces: torch.Tensor | None = None,
    ext_batch_idx: torch.Tensor | None = None,
    vf: torch.Tensor | None = None,
    v_sumsq: torch.Tensor | None = None,
    f_sumsq: torch.Tensor | None = None,
    max_norm: torch.Tensor | None = None,
    delaystep: int = 60,
    dtgrow: float = 1.05,
    dtshrink: float = 0.75,
    alphashrink: float = 0.985,
    alpha0: float = 0.09,
    tmax: float = 0.08,
    tmin: float = 0.005,
    maxstep: float = 0.1,
    cell_force_scale: float = 1.0,
    compute_reductions: bool = True,
) -> None:
    """FIRE2 variable-cell optimization step.

    Performs a FIRE2 step on both atomic coordinates and cell degrees of
    freedom. Internally packs atomic + cell velocity/force DOFs into an
    **interleaved layout** (each system's atoms followed by its 2 cell
    vec3s), runs the FIRE2 reduction + mixing phase on those generalized
    DOFs, unpacks the mixed velocities, and then applies the physically
    coupled atomic/cell update directly on the caller's coordinate and cell
    tensors.

    The cell must be pre-aligned to upper-triangular form via
    :func:`nvalchemiops.dynamics.utils.cell_filter.align_cell` before
    the first call.

    Modifies *positions*, *velocities*, *cell*, *cell_velocities*,
    *alpha*, *dt*, and *nsteps_inc* in-place.

    Parameters
    ----------
    positions : Tensor, shape (N, 3), dtype float32/float64
        Atomic positions.
    velocities : Tensor, shape (N, 3), dtype float32/float64
        Atomic velocities.
    forces : Tensor, shape (N, 3), dtype float32/float64
        Forces on atoms (read-only).
    cell : Tensor, shape (M, 3, 3), dtype float32/float64
        Cell matrices (upper-triangular from ``align_cell()``).
    cell_velocities : Tensor, shape (M, 3, 3), dtype float32/float64
        Cell velocity matrices.
    cell_force : Tensor, shape (M, 3, 3), dtype float32/float64
        Raw cell force matrices from ``stress_to_cell_force()`` (read-only).
        These are divided by ``atoms_per_system * cell_force_scale`` before
        FIRE2 velocity mixing.
    batch_idx : Tensor, shape (N,), dtype int32
        Sorted system index per atom.
    alpha : Tensor, shape (M,), dtype float32/float64
        FIRE2 mixing parameter.
    dt : Tensor, shape (M,), dtype float32/float64
        Per-system timestep.
    nsteps_inc : Tensor, shape (M,), dtype int32
        Consecutive positive-power counter.
    atom_ptr : Tensor, shape (M+1,), dtype int32, optional
        CSR-style atom pointers derived from *batch_idx*.  If ``None``,
        computed internally each call via
        :func:`~nvalchemiops.batch_utils.batch_idx_to_atom_ptr`.
        Pre-compute once and pass in tight loops to avoid repeated
        allocation.  See *Notes* for how to compute.
    ext_atom_ptr : Tensor, shape (M+1,), dtype int32, optional
        Extended atom pointers (accounts for 2 cell DOFs per system).
        If ``None``, computed from *atom_ptr* each call via
        :func:`~nvalchemiops.dynamics.utils.cell_filter.extend_atom_ptr`.
        See *Notes* for how to compute.
    ext_positions : Tensor, shape (N+2M, 3), optional
        Deprecated and ignored.
        ``ext_positions`` is accepted only for backward compatibility.
        The coupled variable-cell FIRE2 path no longer updates packed
        positions directly.

    ext_velocities, ext_forces : Tensor, shape (N+2M, 3), optional
        Pre-allocated extended working arrays for the FIRE2 generalized-DOF
        reduction and mixing phase. Allocated if ``None``; contents are
        overwritten each call.
    ext_batch_idx : Tensor, shape (N+2M,), dtype int32, optional
        Pre-computed extended batch index (sorted, matching interleaved
        pack layout).  If ``None``, computed from *ext_atom_ptr* each
        call via
        :func:`~nvalchemiops.batch_utils.atom_ptr_to_batch_idx`.
        If provided, assumed correct and reused without recomputation.
        See *Notes* for how to compute.
    vf, v_sumsq, f_sumsq, max_norm : Tensor, shape (M,), optional
        Scratch buffers for reductions.  Allocated and zeroed if ``None``;
        zeroed in-place if provided. In this coupled cell adapter,
        ``max_norm`` is the final physical Cartesian atomic displacement norm,
        recomputed after cell motion is coupled back to the atoms.
    delaystep, dtgrow, dtshrink, alphashrink, alpha0, tmax, tmin, maxstep
        FIRE2 hyperparameters.
    cell_force_scale : float, default=1.0
        Extra positive multiplier for stress-derived cell-force normalization.
        Cell forces are divided by ``atoms_per_system * cell_force_scale``.
    compute_reductions : bool, default True
        If True, recompute ``vf``/``v_sumsq``/``f_sumsq`` internally. If False,
        use the caller-supplied values instead (they must be provided and are
        not zeroed). These reductions are over the **generalized (atom + cell)
        DOFs** of each system, so a caller assembling them across a partition
        must include the (replicated) cell contribution exactly once. The
        ``maxstep`` clamp still uses this call's internally-recomputed
        ``max_norm`` (the physical Cartesian displacement after cell coupling).

    Notes
    -----
    Default hyperparameters (from the underlying Warp implementation):
    ``delaystep=60``, ``dtgrow=1.05``, ``dtshrink=0.75``,
    ``alphashrink=0.985``, ``alpha0=0.09``, ``tmax=0.08``,
    ``tmin=0.005``, ``maxstep=0.1``.
    See :func:`nvalchemiops.dynamics.optimizers.fire2.fire2_step` for
    full descriptions.

    The high-level variable-cell adapter normalizes raw stress-derived cell
    forces by the number of atoms in each system. ``cell_force_scale`` is an
    extra multiplier on top of that per-system atom-count normalization.

    **Pre-computing static metadata for tight loops**

    When *batch_idx* does not change between steps (fixed system sizes),
    *atom_ptr*, *ext_atom_ptr*, and *ext_batch_idx* are constant and can
    be pre-computed once to eliminate per-step allocation and kernel
    launches::

        import warp as wp
        from nvalchemiops.batch_utils import (
            atom_ptr_to_batch_idx,
            batch_idx_to_atom_ptr,
        )
        from nvalchemiops.dynamics.utils.cell_filter import extend_atom_ptr

        N, M = positions.shape[0], alpha.shape[0]
        N_ext = N + 2 * M
        device = positions.device

        # 1) atom_ptr from batch_idx  (CSR pointers into atom array)
        atom_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
        atom_counts = torch.zeros(M, dtype=torch.int32, device=device)
        batch_idx_to_atom_ptr(
            wp.from_torch(batch_idx, dtype=wp.int32),
            wp.from_torch(atom_counts, dtype=wp.int32),
            wp.from_torch(atom_ptr, dtype=wp.int32),
        )

        # 2) ext_atom_ptr  (CSR pointers into extended array,
        #    each system's range grows by 2 for the cell DOFs)
        ext_atom_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
        extend_atom_ptr(
            wp.from_torch(atom_ptr, dtype=wp.int32),
            wp.from_torch(ext_atom_ptr, dtype=wp.int32),
        )

        # 3) ext_batch_idx  (sorted system index for extended array)
        ext_batch_idx = torch.empty(N_ext, dtype=torch.int32, device=device)
        atom_ptr_to_batch_idx(
            wp.from_torch(ext_atom_ptr, dtype=wp.int32),
            wp.from_torch(ext_batch_idx, dtype=wp.int32),
        )

    Then pass all three on every step::

        fire2_step_coord_cell(
            ...,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            ext_batch_idx=ext_batch_idx,
        )

    **Extended array layout (interleaved)**

    The mixing phase places each system's cell DOFs immediately after its
    atoms::

        [sys0_atom0, ..., sys0_atomK, sys0_cell_row0, sys0_cell_row1,
         sys1_atom0, ..., sys1_atomJ, sys1_cell_row0, sys1_cell_row1, ...]

    This ensures that *ext_batch_idx* is sorted (all DOFs for system 0
    precede all DOFs for system 1, etc.), which is required by
    ``fire2_step``'s segmented reductions.

    Examples
    --------
    Minimal single-step call (all buffers allocated internally):

    >>> fire2_step_coord_cell(
    ...     positions, velocities, forces,
    ...     cell, cell_velocities, cell_force,
    ...     batch_idx, alpha, dt, nsteps_inc,
    ... )

    Tight optimization loop with pre-allocated buffers:

    >>> # Pre-compute static metadata once
    >>> atom_ptr = ...   # see Notes
    >>> ext_atom_ptr = ...
    >>> ext_batch_idx = ...
    >>> N_ext = positions.shape[0] + 2 * alpha.shape[0]
    >>> ext_vel = torch.empty(N_ext, 3, dtype=positions.dtype,
    ...                       device=positions.device)
    >>> ext_forces = torch.empty_like(ext_vel)
    >>> M = alpha.shape[0]
    >>> vf = torch.empty(M, dtype=positions.dtype, device=positions.device)
    >>> v_sumsq = torch.empty_like(vf)
    >>> f_sumsq = torch.empty_like(vf)
    >>> max_norm = torch.empty_like(vf)
    >>> for step in range(num_steps):
    ...     fire2_step_coord_cell(
    ...         positions, velocities, forces,
    ...         cell, cell_velocities, cell_force,
    ...         batch_idx, alpha, dt, nsteps_inc,
    ...         atom_ptr=atom_ptr,
    ...         ext_atom_ptr=ext_atom_ptr,
    ...         ext_velocities=ext_vel,
    ...         ext_forces=ext_forces,
    ...         ext_batch_idx=ext_batch_idx,
    ...         vf=vf, v_sumsq=v_sumsq,
    ...         f_sumsq=f_sumsq, max_norm=max_norm,
    ...     )
    """
    res = _coord_cell_mix_impl(
        positions,
        velocities,
        forces,
        cell,
        cell_velocities,
        cell_force,
        batch_idx,
        alpha,
        dt,
        nsteps_inc,
        atom_ptr=atom_ptr,
        ext_atom_ptr=ext_atom_ptr,
        ext_velocities=ext_velocities,
        ext_forces=ext_forces,
        ext_batch_idx=ext_batch_idx,
        vf=vf,
        v_sumsq=v_sumsq,
        f_sumsq=f_sumsq,
        max_norm=max_norm,
        delaystep=delaystep,
        dtgrow=dtgrow,
        dtshrink=dtshrink,
        alphashrink=alphashrink,
        alpha0=alpha0,
        tmax=tmax,
        tmin=tmin,
        cell_force_scale=cell_force_scale,
        compute_reductions=compute_reductions,
        ext_positions=ext_positions,
    )
    if res is None:
        return
    (
        wp_pos,
        wp_vel,
        wp_cell,
        wp_cell_vel,
        wp_dt,
        wp_vf,
        wp_ext_batch_idx,
        wp_ext_atom_ptr,
        wp_max_norm,
        wp_device,
    ) = res
    # Couple + clamp + apply the affine cell update directly on positions/cells.
    _apply_fire2_coord_cell_step(
        wp_pos,
        wp_vel,
        wp_cell,
        wp_cell_vel,
        wp_dt,
        wp_vf,
        wp_ext_batch_idx,
        wp_ext_atom_ptr,
        wp_max_norm,
        maxstep=maxstep,
        device=wp_device,
    )


def fire2_step_coord_cell_mix(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    cell: torch.Tensor,
    cell_velocities: torch.Tensor,
    cell_force: torch.Tensor,
    batch_idx: torch.Tensor,
    alpha: torch.Tensor,
    dt: torch.Tensor,
    nsteps_inc: torch.Tensor,
    *,
    atom_ptr: torch.Tensor | None = None,
    ext_atom_ptr: torch.Tensor | None = None,
    ext_velocities: torch.Tensor | None = None,
    ext_forces: torch.Tensor | None = None,
    ext_batch_idx: torch.Tensor | None = None,
    vf: torch.Tensor | None = None,
    v_sumsq: torch.Tensor | None = None,
    f_sumsq: torch.Tensor | None = None,
    max_norm: torch.Tensor | None = None,
    delaystep: int = 60,
    dtgrow: float = 1.05,
    dtshrink: float = 0.75,
    alphashrink: float = 0.985,
    alpha0: float = 0.09,
    tmax: float = 0.08,
    tmin: float = 0.005,
    cell_force_scale: float = 1.0,
    compute_reductions: bool = True,
) -> None:
    """Phase 1 of the coupled variable-cell FIRE2 step: reduce + mix only.

    Packs atomic + cell DOFs, runs the FIRE2 reduction (gated by
    ``compute_reductions``) and velocity mixing on the generalized DOFs, and
    unpacks the mixed velocities back into ``velocities`` / ``cell_velocities``.
    Updates ``alpha``, ``dt``, ``nsteps_inc`` in-place. It does **not** apply the
    position/cell step — pair it with :func:`fire2_step_coord_cell_couple` and
    :func:`fire2_step_coord_cell_apply` so the displacement clamp can use a
    ``max_norm`` post-processed between the couple and apply phases.

    See :func:`fire2_step_coord_cell` for parameter descriptions.
    """
    _coord_cell_mix_impl(
        positions,
        velocities,
        forces,
        cell,
        cell_velocities,
        cell_force,
        batch_idx,
        alpha,
        dt,
        nsteps_inc,
        atom_ptr=atom_ptr,
        ext_atom_ptr=ext_atom_ptr,
        ext_velocities=ext_velocities,
        ext_forces=ext_forces,
        ext_batch_idx=ext_batch_idx,
        vf=vf,
        v_sumsq=v_sumsq,
        f_sumsq=f_sumsq,
        max_norm=max_norm,
        delaystep=delaystep,
        dtgrow=dtgrow,
        dtshrink=dtshrink,
        alphashrink=alphashrink,
        alpha0=alpha0,
        tmax=tmax,
        tmin=tmin,
        cell_force_scale=cell_force_scale,
        compute_reductions=compute_reductions,
        ext_positions=None,
    )


def fire2_step_coord_cell_couple(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    cell: torch.Tensor,
    cell_velocities: torch.Tensor,
    dt: torch.Tensor,
    vf: torch.Tensor,
    batch_idx: torch.Tensor,
    max_norm: torch.Tensor,
    *,
    atom_ptr: torch.Tensor | None = None,
    ext_atom_ptr: torch.Tensor | None = None,
    ext_batch_idx: torch.Tensor | None = None,
) -> None:
    """Phase 2 of the coupled variable-cell FIRE2 step: measure ``max_norm``.

    Recomputes each atom's cell-coupled step from the mixed velocities (output
    of :func:`fire2_step_coord_cell_mix`) and writes the per-system maximum
    physical Cartesian displacement norm into ``max_norm`` (zeroed first).
    Positions and cell are **not** moved, so a caller can post-process
    ``max_norm`` before :func:`fire2_step_coord_cell_apply`.
    """
    device = positions.device
    M = dt.shape[0]
    if positions.shape[0] == 0:
        max_norm.zero_()
        return
    vec_type = _TORCH_TO_WP_VEC[positions.dtype]
    mat_type = _TORCH_TO_WP_MAT[cell.dtype]
    wp_device = wp.device_from_torch(device)
    _, _, wp_ext_atom_ptr, wp_ext_batch_idx = _coord_cell_ext_metadata(
        batch_idx,
        M,
        device,
        wp_device,
        atom_ptr=atom_ptr,
        ext_atom_ptr=ext_atom_ptr,
        ext_batch_idx=ext_batch_idx,
    )
    _fire2_coord_cell_compute_max_norm(
        wp.from_torch(positions.detach(), dtype=vec_type),
        wp.from_torch(velocities.detach(), dtype=vec_type),
        wp.from_torch(cell.detach(), dtype=mat_type),
        wp.from_torch(cell_velocities.detach(), dtype=mat_type),
        wp.from_torch(dt.detach()),
        wp.from_torch(vf),
        wp_ext_batch_idx,
        wp_ext_atom_ptr,
        wp.from_torch(max_norm),
        device=wp_device,
    )


def fire2_step_coord_cell_apply(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    cell: torch.Tensor,
    cell_velocities: torch.Tensor,
    dt: torch.Tensor,
    vf: torch.Tensor,
    batch_idx: torch.Tensor,
    max_norm: torch.Tensor,
    *,
    maxstep: float = 0.1,
    atom_ptr: torch.Tensor | None = None,
    ext_atom_ptr: torch.Tensor | None = None,
    ext_batch_idx: torch.Tensor | None = None,
) -> None:
    """Phase 3 of the coupled variable-cell FIRE2 step: clamp + apply.

    Recomputes the same cell-coupled step as
    :func:`fire2_step_coord_cell_couple`, clamps it by
    ``min(1, maxstep / max_norm[s])`` using the supplied ``max_norm``, and writes
    ``positions``, ``cell``, ``cell_velocities`` (uphill zeroing), and the clamped
    ``dt`` in-place.
    """
    device = positions.device
    M = dt.shape[0]
    if positions.shape[0] == 0:
        return
    vec_type = _TORCH_TO_WP_VEC[positions.dtype]
    mat_type = _TORCH_TO_WP_MAT[cell.dtype]
    wp_device = wp.device_from_torch(device)
    _, _, wp_ext_atom_ptr, wp_ext_batch_idx = _coord_cell_ext_metadata(
        batch_idx,
        M,
        device,
        wp_device,
        atom_ptr=atom_ptr,
        ext_atom_ptr=ext_atom_ptr,
        ext_batch_idx=ext_batch_idx,
    )
    _fire2_coord_cell_clamp_apply(
        wp.from_torch(positions.detach(), dtype=vec_type),
        wp.from_torch(velocities.detach(), dtype=vec_type),
        wp.from_torch(cell.detach(), dtype=mat_type),
        wp.from_torch(cell_velocities.detach(), dtype=mat_type),
        wp.from_torch(dt.detach()),
        wp.from_torch(vf),
        wp_ext_batch_idx,
        wp_ext_atom_ptr,
        wp.from_torch(max_norm),
        maxstep,
        device=wp_device,
    )


def fire2_compute_extended_reductions(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    cell: torch.Tensor,
    cell_velocities: torch.Tensor,
    cell_force: torch.Tensor,
    batch_idx: torch.Tensor,
    dt: torch.Tensor,
    *,
    atom_ptr: torch.Tensor | None = None,
    ext_atom_ptr: torch.Tensor | None = None,
    ext_velocities: torch.Tensor | None = None,
    ext_forces: torch.Tensor | None = None,
    ext_batch_idx: torch.Tensor | None = None,
    cell_force_scale: float = 1.0,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Split the variable-cell FIRE2 reductions into atom and cell parts.

    Returns ``(atom_partial, cell_term)``, each a ``(vf, v_sumsq, f_sumsq)``
    triple of ``(M,)`` tensors, such that ``atom_partial + cell_term`` equals the
    reduction :func:`fire2_step_coord_cell_mix` computes over the packed
    generalized DOFs. The atom part is a plain per-atom partition (combine it
    across a caller-side partition), while the cell part is replicated (add it
    exactly once)::

        atom, cell_t = fire2_compute_extended_reductions(..., dt)
        vf = allreduce(SUM, atom[0]) + cell_t[0]   # and v_sumsq, f_sumsq
        fire2_step_coord_cell_mix(..., vf=vf, ..., compute_reductions=False)

    Uses the same ``v + f*dt`` half-step as the internal reduction, so the
    combined result is bit-parity with recomputing it in one pass.
    """
    dtype = positions.dtype
    device = positions.device
    N = positions.shape[0]
    M = cell.shape[0]
    N_ext = N + 2 * M
    vec_type = _TORCH_TO_WP_VEC[dtype]
    mat_type = _TORCH_TO_WP_MAT[dtype]
    wp_device = wp.device_from_torch(device)

    if cell_force_scale <= 0.0:
        raise ValueError("cell_force_scale must be positive")

    def _triple():
        return (
            torch.zeros(M, dtype=dtype, device=device),
            torch.zeros(M, dtype=dtype, device=device),
            torch.zeros(M, dtype=dtype, device=device),
        )

    atom_partial = _triple()
    cell_term = _triple()
    if N == 0:
        return atom_partial, cell_term

    wp_dt = wp.from_torch(dt.detach())
    wp_bidx = wp.from_torch(batch_idx.detach(), dtype=wp.int32)

    # atom partial: the FIRE2 half-step reduction over the atom arrays directly.
    fire2_reduce(
        wp.from_torch(velocities.detach(), dtype=vec_type),
        wp.from_torch(forces.detach(), dtype=vec_type),
        wp_dt,
        wp_bidx,
        wp.from_torch(atom_partial[0]),
        wp.from_torch(atom_partial[1]),
        wp.from_torch(atom_partial[2]),
    )

    # total: pack the extended (atom + cell) DOFs and reduce over them.
    if ext_velocities is None:
        ext_velocities = torch.empty(N_ext, 3, dtype=dtype, device=device)
    if ext_forces is None:
        ext_forces = torch.empty(N_ext, 3, dtype=dtype, device=device)
    wp_vel = wp.from_torch(velocities.detach(), dtype=vec_type)
    wp_forces = wp.from_torch(forces.detach(), dtype=vec_type)
    wp_cell_vel = wp.from_torch(cell_velocities.detach(), dtype=mat_type)
    wp_ext_vel = wp.from_torch(ext_velocities, dtype=vec_type)
    wp_ext_forces = wp.from_torch(ext_forces, dtype=vec_type)

    atom_ptr, wp_atom_ptr, wp_ext_atom_ptr, wp_ext_batch_idx = _coord_cell_ext_metadata(
        batch_idx,
        M,
        device,
        wp_device,
        atom_ptr=atom_ptr,
        ext_atom_ptr=ext_atom_ptr,
        ext_batch_idx=ext_batch_idx,
    )
    atom_counts = atom_ptr[1:] - atom_ptr[:-1]
    if torch.any(atom_counts <= 0).item():
        raise ValueError(
            "fire2_compute_extended_reductions requires at least one atom per system"
        )
    cell_force_divisor = atom_counts.to(dtype=dtype).reshape(M, 1, 1) * cell_force_scale
    cell_force_work = (cell_force.detach() / cell_force_divisor).contiguous()
    wp_cell_force = wp.from_torch(cell_force_work, dtype=mat_type)

    if M == 1:
        pack_velocities_with_cell(wp_vel, wp_cell_vel, wp_ext_vel, device=wp_device)
        pack_forces_with_cell(wp_forces, wp_cell_force, wp_ext_forces, device=wp_device)
    else:
        pack_velocities_with_cell(
            wp_vel,
            wp_cell_vel,
            wp_ext_vel,
            wp_atom_ptr,
            wp_ext_atom_ptr,
            device=wp_device,
            batch_idx=wp_bidx,
        )
        pack_forces_with_cell(
            wp_forces,
            wp_cell_force,
            wp_ext_forces,
            wp_atom_ptr,
            wp_ext_atom_ptr,
            device=wp_device,
            batch_idx=wp_bidx,
        )

    total = _triple()
    fire2_reduce(
        wp_ext_vel,
        wp_ext_forces,
        wp_dt,
        wp_ext_batch_idx,
        wp.from_torch(total[0]),
        wp.from_torch(total[1]),
        wp.from_torch(total[2]),
    )

    # cell term = total (atom + cell) - atom partial.
    for i in range(3):
        cell_term[i].copy_(total[i] - atom_partial[i])

    return atom_partial, cell_term


def fire2_step_extended(
    ext_positions: torch.Tensor,
    ext_velocities: torch.Tensor,
    ext_forces: torch.Tensor,
    ext_batch_idx: torch.Tensor,
    alpha: torch.Tensor,
    dt: torch.Tensor,
    nsteps_inc: torch.Tensor,
    *,
    vf: torch.Tensor | None = None,
    v_sumsq: torch.Tensor | None = None,
    f_sumsq: torch.Tensor | None = None,
    max_norm: torch.Tensor | None = None,
    delaystep: int = 60,
    dtgrow: float = 1.05,
    dtshrink: float = 0.75,
    alphashrink: float = 0.985,
    alpha0: float = 0.09,
    tmax: float = 0.08,
    tmin: float = 0.005,
    maxstep: float = 0.1,
) -> None:
    """Run FIRE2 directly on pre-packed extended arrays (no pack/unpack).

    This is a lower-level API for callers that maintain persistent extended
    arrays (positions + cell DOFs interleaved).  The caller is responsible
    for packing data into the extended layout before the first call and
    unpacking results after the last call (or as needed).

    This eliminates the per-step pack/unpack overhead that
    ``fire2_step_coord_cell`` incurs.

    Unlike :func:`fire2_step_coord_cell`, this function treats the packed DOFs
    exactly as provided. It does not add the affine atomic remap implied by
    variable-cell motion, so it should be used only when the caller explicitly
    wants generic packed-DOF FIRE2 behavior.

    Parameters
    ----------
    ext_positions : torch.Tensor, shape (N_ext, 3)
        Extended position array (atoms + cell DOFs interleaved).
    ext_velocities : torch.Tensor, shape (N_ext, 3)
        Extended velocity array.
    ext_forces : torch.Tensor, shape (N_ext, 3)
        Extended force array.
    ext_batch_idx : torch.Tensor, shape (N_ext,), dtype=int32
        System index for each element in the extended arrays.
    alpha : torch.Tensor, shape (M,)
        FIRE2 mixing parameter per system.
    dt : torch.Tensor, shape (M,)
        Timestep per system.
    nsteps_inc : torch.Tensor, shape (M,), dtype=int32
        Consecutive positive-power step counter per system.
    vf, v_sumsq, f_sumsq, max_norm : torch.Tensor or None
        Per-system scratch buffers, shape (M,). Allocated internally if None.
    delaystep, dtgrow, dtshrink, alphashrink, alpha0, tmax, tmin, maxstep :
        FIRE2 hyperparameters.  See ``fire2_step_coord_cell`` for details.

    Notes
    -----
    Modifies ``ext_positions``, ``ext_velocities``, ``alpha``, ``dt``,
    and ``nsteps_inc`` in-place.
    """
    dtype = ext_positions.dtype
    device = ext_positions.device
    M = alpha.shape[0]
    vec_type = _TORCH_TO_WP_VEC[dtype]

    # Reduction scratch buffers
    vf = _alloc_or_zero(vf, M, dtype, device)
    v_sumsq = _alloc_or_zero(v_sumsq, M, dtype, device)
    f_sumsq = _alloc_or_zero(f_sumsq, M, dtype, device)
    max_norm = _alloc_or_zero(max_norm, M, dtype, device)

    if ext_positions.shape[0] == 0:
        return

    fire2_step(
        wp.from_torch(ext_positions.detach(), dtype=vec_type),
        wp.from_torch(ext_velocities.detach(), dtype=vec_type),
        wp.from_torch(ext_forces.detach(), dtype=vec_type),
        wp.from_torch(ext_batch_idx.detach(), dtype=wp.int32),
        wp.from_torch(alpha.detach()),
        wp.from_torch(dt.detach()),
        wp.from_torch(nsteps_inc.detach(), dtype=wp.int32),
        wp.from_torch(vf),
        wp.from_torch(v_sumsq),
        wp.from_torch(f_sumsq),
        wp.from_torch(max_norm),
        delaystep=delaystep,
        dtgrow=dtgrow,
        dtshrink=dtshrink,
        alphashrink=alphashrink,
        alpha0=alpha0,
        tmax=tmax,
        tmin=tmin,
        maxstep=maxstep,
    )

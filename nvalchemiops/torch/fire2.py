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
  interleaved extended array, runs FIRE2, and unpacks results.
- :func:`fire2_step_extended` -- run FIRE2 directly on caller-managed
  extended arrays (no per-step pack/unpack overhead).

Modifies inputs in-place. Scratch buffers and static metadata
(``atom_ptr``, ``ext_atom_ptr``, ``ext_batch_idx``) can be passed in
for reuse across steps, or left as ``None`` to allocate internally
each call.  See :func:`fire2_step_coord_cell` docstring for
pre-computation recipes.
"""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.batch_utils import atom_ptr_to_batch_idx, batch_idx_to_atom_ptr
from nvalchemiops.dynamics.optimizers.fire2 import fire2_step
from nvalchemiops.dynamics.utils.cell_filter import (
    extend_atom_ptr,
    pack_forces_with_cell,
    pack_positions_with_cell,
    pack_velocities_with_cell,
    unpack_positions_with_cell,
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

    Notes
    -----
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

    # Scratch buffers: allocate if not provided, zero if provided
    vf = _alloc_or_zero(vf, M, dtype, device)
    v_sumsq = _alloc_or_zero(v_sumsq, M, dtype, device)
    f_sumsq = _alloc_or_zero(f_sumsq, M, dtype, device)
    max_norm = _alloc_or_zero(max_norm, M, dtype, device)

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
) -> None:
    """FIRE2 variable-cell optimization step.

    Performs a FIRE2 step on both atomic coordinates and cell degrees of
    freedom.  Internally packs atomic + cell DOFs into extended arrays
    using an **interleaved layout** (each system's atoms followed by its
    2 cell vec3s), runs the 3-kernel FIRE2 algorithm, and unpacks
    results back.

    The cell must be pre-aligned to upper-triangular form via
    ``align_cell()`` before the first call.

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
        Cell force matrices from ``stress_to_cell_force()`` (read-only).
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
    ext_positions, ext_velocities, ext_forces : Tensor, shape (N+2M, 3), optional
        Pre-allocated extended working arrays.  Allocated if ``None``;
        contents are overwritten each call.  Providing them avoids
        repeated allocation in tight loops.
    ext_batch_idx : Tensor, shape (N+2M,), dtype int32, optional
        Pre-computed extended batch index (sorted, matching interleaved
        pack layout).  If ``None``, computed from *ext_atom_ptr* each
        call via
        :func:`~nvalchemiops.batch_utils.atom_ptr_to_batch_idx`.
        If provided, assumed correct and reused without recomputation.
        See *Notes* for how to compute.
    vf, v_sumsq, f_sumsq, max_norm : Tensor, shape (M,), optional
        Scratch buffers for reductions.  Allocated and zeroed if ``None``;
        zeroed in-place if provided.
    delaystep, dtgrow, dtshrink, alphashrink, alpha0, tmax, tmin, maxstep
        FIRE2 hyperparameters.

    Notes
    -----
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

    The packing places each system's cell DOFs immediately after its
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
    >>> ext_pos = torch.empty(N_ext, 3, dtype=positions.dtype,
    ...                       device=positions.device)
    >>> ext_vel = torch.empty_like(ext_pos)
    >>> ext_forces = torch.empty_like(ext_pos)
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
    ...         ext_positions=ext_pos,
    ...         ext_velocities=ext_vel,
    ...         ext_forces=ext_forces,
    ...         ext_batch_idx=ext_batch_idx,
    ...         vf=vf, v_sumsq=v_sumsq,
    ...         f_sumsq=f_sumsq, max_norm=max_norm,
    ...     )
    """
    dtype = positions.dtype
    device = positions.device
    N = positions.shape[0]
    M = alpha.shape[0]
    N_ext = N + 2 * M
    vec_type = _TORCH_TO_WP_VEC[dtype]
    mat_type = _TORCH_TO_WP_MAT[dtype]
    wp_device = wp.device_from_torch(device)

    # --- Allocate extended working arrays if not provided ---
    if ext_positions is None:
        ext_positions = torch.empty(N_ext, 3, dtype=dtype, device=device)
    if ext_velocities is None:
        ext_velocities = torch.empty(N_ext, 3, dtype=dtype, device=device)
    if ext_forces is None:
        ext_forces = torch.empty(N_ext, 3, dtype=dtype, device=device)

    # --- Reduction scratch buffers: allocate or zero ---
    vf = _alloc_or_zero(vf, M, dtype, device)
    v_sumsq = _alloc_or_zero(v_sumsq, M, dtype, device)
    f_sumsq = _alloc_or_zero(f_sumsq, M, dtype, device)
    max_norm = _alloc_or_zero(max_norm, M, dtype, device)

    # --- Convert inputs to Warp arrays for pack/unpack kernels ---
    wp_pos = wp.from_torch(positions.detach(), dtype=vec_type)
    wp_vel = wp.from_torch(velocities.detach(), dtype=vec_type)
    wp_forces = wp.from_torch(forces.detach(), dtype=vec_type)
    wp_cell = wp.from_torch(cell.detach(), dtype=mat_type)
    wp_cell_vel = wp.from_torch(cell_velocities.detach(), dtype=mat_type)
    wp_cell_force = wp.from_torch(cell_force.detach(), dtype=mat_type)
    wp_bidx = wp.from_torch(batch_idx.detach(), dtype=wp.int32)

    wp_ext_pos = wp.from_torch(ext_positions, dtype=vec_type)
    wp_ext_vel = wp.from_torch(ext_velocities, dtype=vec_type)
    wp_ext_forces = wp.from_torch(ext_forces, dtype=vec_type)

    # --- Compute atom_ptr / ext_atom_ptr if not provided ---
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

    # --- Pack into extended arrays ---
    if M == 1:
        pack_positions_with_cell(
            wp_pos,
            wp_cell,
            wp_ext_pos,
            device=wp_device,
        )
        pack_velocities_with_cell(
            wp_vel,
            wp_cell_vel,
            wp_ext_vel,
            device=wp_device,
        )
        pack_forces_with_cell(
            wp_forces,
            wp_cell_force,
            wp_ext_forces,
            device=wp_device,
        )
    else:
        pack_positions_with_cell(
            wp_pos,
            wp_cell,
            wp_ext_pos,
            wp_atom_ptr,
            wp_ext_atom_ptr,
            device=wp_device,
            batch_idx=wp_bidx,
        )
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

    # --- Extended batch_idx: compute sorted index from ext_atom_ptr if not provided ---
    if ext_batch_idx is None:
        ext_batch_idx = torch.empty(N_ext, dtype=torch.int32, device=device)
        atom_ptr_to_batch_idx(
            wp_ext_atom_ptr,
            wp.from_torch(ext_batch_idx, dtype=wp.int32),
        )

    # --- Delegate to the Warp-level fire2_step on extended arrays ---
    fire2_step(
        wp_ext_pos,
        wp_ext_vel,
        wp_ext_forces,
        wp.from_torch(ext_batch_idx, dtype=wp.int32),
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

    # --- Unpack extended arrays back to original tensors ---
    if M == 1:
        unpack_positions_with_cell(
            wp_ext_pos,
            wp_pos,
            wp_cell,
            num_atoms=N,
            device=wp_device,
        )
        unpack_velocities_with_cell(
            wp_ext_vel,
            wp_vel,
            wp_cell_vel,
            num_atoms=N,
            device=wp_device,
        )
    else:
        unpack_positions_with_cell(
            wp_ext_pos,
            wp_pos,
            wp_cell,
            atom_ptr=wp_atom_ptr,
            ext_atom_ptr=wp_ext_atom_ptr,
            device=wp_device,
            batch_idx=wp_bidx,
        )
        unpack_velocities_with_cell(
            wp_ext_vel,
            wp_vel,
            wp_cell_vel,
            atom_ptr=wp_atom_ptr,
            ext_atom_ptr=wp_ext_atom_ptr,
            device=wp_device,
            batch_idx=wp_bidx,
        )


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

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

"""Registered slab correction energy chain."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import warp as wp

from nvalchemiops.torch._warp_op_helpers import register_warp_op_chain
from nvalchemiops.torch.interactions.electrostatics._util import (
    _is_sync_free_uniform_cotangent,
)
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = ["register_slab_ops", "slab_correction_energy"]

_SLAB_CHAIN: dict[str, object] | None = None
_SLAB_OPS_REGISTERED = False


def _wp_from_torch(tensor: torch.Tensor, dtype):
    """Convert a tensor to Warp without allocating unused Warp gradients."""
    return wp.from_torch(tensor.detach().contiguous(), dtype=dtype, requires_grad=False)


def _scoped_stream(device: torch.device):
    """Bind Warp launches to PyTorch's current CUDA stream."""
    if device.type != "cuda":
        return nullcontext()
    return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device)))


def _per_system_cotangent(
    grad_energy_atom: torch.Tensor,
    batch_idx: torch.Tensor,
    num_systems: int,
) -> torch.Tensor:
    """Reduce atom energy cotangents to per-system means."""
    if grad_energy_atom.numel() == 0:
        return torch.zeros(
            num_systems, device=grad_energy_atom.device, dtype=torch.float64
        )

    idx = batch_idx.to(device=grad_energy_atom.device, dtype=torch.long)
    sums = torch.zeros(num_systems, device=grad_energy_atom.device, dtype=torch.float64)
    sums = sums.index_add(0, idx, grad_energy_atom.reshape(-1).to(torch.float64))
    counts = torch.bincount(idx, minlength=num_systems).to(torch.float64)
    return sums / counts.clamp_min(1.0)


def _cotangent_per_system_uniform(
    grad_energy_atom: torch.Tensor,
    batch_idx: torch.Tensor,
    num_systems: int,
) -> bool:
    """Whether the per-atom energy cotangent is constant within each system."""
    if _is_sync_free_uniform_cotangent(grad_energy_atom):
        return True
    g = grad_energy_atom.reshape(-1)
    if g.numel() == 0:
        return True
    if g.numel() == 1 or g.numel() == num_systems:
        return True
    if g.numel() != batch_idx.numel():
        return False
    if g.is_cuda:
        return False
    idx = batch_idx.to(device=g.device, dtype=torch.long)
    g64 = g.to(torch.float64)
    sys_max = torch.full(
        (num_systems,), float("-inf"), dtype=torch.float64, device=g.device
    ).scatter_reduce(0, idx, g64, reduce="amax", include_self=False)
    return bool(torch.all(g64 == sys_max.index_select(0, idx)).item())


def _distribute_system_values(
    system_values: torch.Tensor,
    batch_idx: torch.Tensor,
    num_atoms: int,
) -> torch.Tensor:
    """Distribute per-system values to per-atom cotangents by atom count."""
    if num_atoms == 0:
        return torch.zeros(0, device=system_values.device, dtype=torch.float64)

    idx = batch_idx.to(device=system_values.device, dtype=torch.long)
    counts = torch.bincount(idx, minlength=system_values.shape[0]).to(torch.float64)
    values = system_values / counts.clamp_min(1.0)
    return values.index_select(0, idx)


def _run_moments(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute slab moments with the shared Warp reduction kernel."""
    from nvalchemiops.interactions.electrostatics.slab_kernels import (
        slab_reduce_moments,
    )

    num_systems = cell.shape[0]
    mz = torch.zeros(num_systems, 3, device=positions.device, dtype=torch.float64)
    mz2 = torch.zeros_like(mz)
    qtotal = torch.zeros(num_systems, device=positions.device, dtype=torch.float64)
    if positions.shape[0] == 0:
        return mz, mz2, qtotal

    wp_dtype = get_wp_dtype(positions.dtype)
    device = wp.device_from_torch(positions.device)
    with _scoped_stream(positions.device):
        slab_reduce_moments(
            _wp_from_torch(positions, get_wp_vec_dtype(positions.dtype)),
            _wp_from_torch(charges.to(positions.dtype), wp_dtype),
            _wp_from_torch(batch_idx, wp.int32),
            _wp_from_torch(pbc.contiguous(), wp.bool),
            _wp_from_torch(cell, get_wp_mat_dtype(positions.dtype)),
            _wp_from_torch(mz, wp.float64),
            _wp_from_torch(mz2, wp.float64),
            _wp_from_torch(qtotal, wp.float64),
            wp_dtype=wp_dtype,
            device=device,
        )
    return mz, mz2, qtotal


def _run_geometry(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute caller-owned slab geometry buffers with the shared Warp kernel."""
    from nvalchemiops.interactions.electrostatics.slab_kernels import (
        slab_precompute_geometry,
    )

    num_systems = cell.shape[0]
    slab_axis = torch.zeros(num_systems, device=positions.device, dtype=torch.int32)
    slab_normal = torch.zeros(
        num_systems, 3, device=positions.device, dtype=torch.float64
    )
    slab_volume = torch.zeros(num_systems, device=positions.device, dtype=torch.float64)
    slab_height_sq = torch.zeros_like(slab_volume)
    if num_systems == 0:
        return slab_axis, slab_normal, slab_volume, slab_height_sq

    wp_dtype = get_wp_dtype(positions.dtype)
    device = wp.device_from_torch(positions.device)
    with _scoped_stream(positions.device):
        slab_precompute_geometry(
            _wp_from_torch(pbc.contiguous(), wp.bool),
            _wp_from_torch(cell, get_wp_mat_dtype(positions.dtype)),
            _wp_from_torch(slab_axis, wp.int32),
            _wp_from_torch(slab_normal, wp.vec3d),
            _wp_from_torch(slab_volume, wp.float64),
            _wp_from_torch(slab_height_sq, wp.float64),
            wp_dtype=wp_dtype,
            device=device,
        )
    return slab_axis, slab_normal, slab_volume, slab_height_sq


def _slab_backward_values(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    grad_system: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute literal slab energy gradients via Warp kernels."""
    from nvalchemiops.interactions.electrostatics.slab_kernels import (
        slab_correction_backward,
    )

    num_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    grad_positions = torch.zeros(
        num_atoms, 3, device=positions.device, dtype=positions.dtype
    )
    grad_charges = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    grad_cell = torch.zeros_like(cell, dtype=positions.dtype)
    if num_systems == 0:
        return grad_positions, grad_charges, grad_cell

    mz, mz2, qtotal = _run_moments(positions, charges, cell, pbc, batch_idx)
    slab_axis, slab_normal, slab_volume, slab_height_sq = _run_geometry(
        positions, cell, pbc
    )
    grad_normal = torch.zeros(
        num_systems, 3, device=positions.device, dtype=torch.float64
    )

    wp_dtype = get_wp_dtype(positions.dtype)
    device = wp.device_from_torch(positions.device)
    with _scoped_stream(positions.device):
        slab_correction_backward(
            _wp_from_torch(positions, get_wp_vec_dtype(positions.dtype)),
            _wp_from_torch(charges.to(positions.dtype), wp_dtype),
            _wp_from_torch(batch_idx, wp.int32),
            _wp_from_torch(pbc.contiguous(), wp.bool),
            _wp_from_torch(cell, get_wp_mat_dtype(positions.dtype)),
            _wp_from_torch(mz, wp.float64),
            _wp_from_torch(mz2, wp.float64),
            _wp_from_torch(qtotal, wp.float64),
            _wp_from_torch(slab_axis, wp.int32),
            _wp_from_torch(slab_normal, wp.vec3d),
            _wp_from_torch(slab_volume, wp.float64),
            _wp_from_torch(slab_height_sq, wp.float64),
            _wp_from_torch(grad_system, wp.float64),
            _wp_from_torch(grad_positions, get_wp_vec_dtype(positions.dtype)),
            _wp_from_torch(grad_charges, wp.float64),
            _wp_from_torch(grad_normal, wp.vec3d),
            _wp_from_torch(grad_cell, get_wp_mat_dtype(positions.dtype)),
            wp_dtype=wp_dtype,
            device=device,
        )
    return grad_positions, grad_charges, grad_cell


def _slab_weighted_backward_values(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    grad_energy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute weighted slab VJP values with Warp kernels."""
    from nvalchemiops.interactions.electrostatics.slab_kernels import (
        _slab_correction_weighted_backward,
    )

    num_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    grad_positions = torch.zeros(
        num_atoms, 3, device=positions.device, dtype=positions.dtype
    )
    grad_charges = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    grad_cell = torch.zeros_like(cell, dtype=positions.dtype)
    if num_systems == 0:
        return grad_positions, grad_charges, grad_cell

    mz, mz2, qtotal = _run_moments(positions, charges, cell, pbc, batch_idx)
    slab_axis, slab_normal, slab_volume, slab_height_sq = _run_geometry(
        positions, cell, pbc
    )
    weighted_mz = torch.zeros_like(mz)
    weighted_mz2 = torch.zeros_like(mz2)
    weighted_qtotal = torch.zeros_like(qtotal)
    grad_normal = torch.zeros(
        num_systems, 3, device=positions.device, dtype=torch.float64
    )

    wp_dtype = get_wp_dtype(positions.dtype)
    device = wp.device_from_torch(positions.device)
    with _scoped_stream(positions.device):
        _slab_correction_weighted_backward(
            _wp_from_torch(positions, get_wp_vec_dtype(positions.dtype)),
            _wp_from_torch(charges.to(positions.dtype), wp_dtype),
            _wp_from_torch(grad_energy.reshape(-1).to(torch.float64), wp.float64),
            _wp_from_torch(batch_idx, wp.int32),
            _wp_from_torch(pbc.contiguous(), wp.bool),
            _wp_from_torch(cell, get_wp_mat_dtype(positions.dtype)),
            _wp_from_torch(mz, wp.float64),
            _wp_from_torch(mz2, wp.float64),
            _wp_from_torch(qtotal, wp.float64),
            _wp_from_torch(slab_axis, wp.int32),
            _wp_from_torch(slab_normal, wp.vec3d),
            _wp_from_torch(slab_volume, wp.float64),
            _wp_from_torch(slab_height_sq, wp.float64),
            _wp_from_torch(weighted_mz, wp.float64),
            _wp_from_torch(weighted_mz2, wp.float64),
            _wp_from_torch(weighted_qtotal, wp.float64),
            _wp_from_torch(grad_positions, get_wp_vec_dtype(positions.dtype)),
            _wp_from_torch(grad_charges, wp.float64),
            _wp_from_torch(grad_normal, wp.vec3d),
            _wp_from_torch(grad_cell, get_wp_mat_dtype(positions.dtype)),
            wp_dtype=wp_dtype,
            device=device,
        )
    return grad_positions, grad_charges, grad_cell


def _slab_weighted_double_backward_values(
    h_pos: torch.Tensor,
    h_charge: torch.Tensor,
    h_cell: torch.Tensor,
    grad_energy: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute weighted slab HVP values with Warp kernels."""
    from nvalchemiops.interactions.electrostatics.slab_kernels import (
        _slab_correction_weighted_double_backward,
    )

    num_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    grad_grad_energy = torch.zeros(
        num_atoms, device=positions.device, dtype=torch.float64
    )
    grad_positions = torch.zeros_like(positions)
    grad_charges = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    grad_cell = torch.zeros_like(cell)
    if num_systems == 0:
        return grad_grad_energy, grad_positions, grad_charges, grad_cell

    mz, mz2, qtotal = _run_moments(positions, charges, cell, pbc, batch_idx)
    slab_axis, slab_normal, slab_volume, slab_height_sq = _run_geometry(
        positions, cell, pbc
    )
    weighted_mz = torch.zeros_like(mz)
    weighted_mz2 = torch.zeros_like(mz2)
    weighted_qtotal = torch.zeros_like(qtotal)
    dmz = torch.zeros_like(mz)
    dmz2 = torch.zeros_like(mz2)
    dqtotal = torch.zeros_like(qtotal)
    d_weighted_mz = torch.zeros_like(mz)
    d_weighted_mz2 = torch.zeros_like(mz2)
    d_weighted_qtotal = torch.zeros_like(qtotal)
    dnormal = torch.zeros(num_systems, 3, device=positions.device, dtype=torch.float64)
    dvolume = torch.zeros(num_systems, device=positions.device, dtype=torch.float64)
    dheight_sq = torch.zeros_like(dvolume)
    grad_normal = torch.zeros_like(dnormal)
    h_grad_normal = torch.zeros_like(dnormal)

    wp_dtype = get_wp_dtype(positions.dtype)
    device = wp.device_from_torch(positions.device)
    with _scoped_stream(positions.device):
        _slab_correction_weighted_double_backward(
            _wp_from_torch(positions, get_wp_vec_dtype(positions.dtype)),
            _wp_from_torch(charges.to(positions.dtype), wp_dtype),
            _wp_from_torch(grad_energy.reshape(-1).to(torch.float64), wp.float64),
            _wp_from_torch(
                h_pos.to(positions.dtype), get_wp_vec_dtype(positions.dtype)
            ),
            _wp_from_torch(h_charge.to(torch.float64), wp.float64),
            _wp_from_torch(h_cell.to(cell.dtype), get_wp_mat_dtype(positions.dtype)),
            _wp_from_torch(batch_idx, wp.int32),
            _wp_from_torch(pbc.contiguous(), wp.bool),
            _wp_from_torch(cell, get_wp_mat_dtype(positions.dtype)),
            _wp_from_torch(mz, wp.float64),
            _wp_from_torch(mz2, wp.float64),
            _wp_from_torch(qtotal, wp.float64),
            _wp_from_torch(slab_axis, wp.int32),
            _wp_from_torch(slab_normal, wp.vec3d),
            _wp_from_torch(slab_volume, wp.float64),
            _wp_from_torch(slab_height_sq, wp.float64),
            _wp_from_torch(weighted_mz, wp.float64),
            _wp_from_torch(weighted_mz2, wp.float64),
            _wp_from_torch(weighted_qtotal, wp.float64),
            _wp_from_torch(dmz, wp.float64),
            _wp_from_torch(dmz2, wp.float64),
            _wp_from_torch(dqtotal, wp.float64),
            _wp_from_torch(d_weighted_mz, wp.float64),
            _wp_from_torch(d_weighted_mz2, wp.float64),
            _wp_from_torch(d_weighted_qtotal, wp.float64),
            _wp_from_torch(dnormal, wp.vec3d),
            _wp_from_torch(dvolume, wp.float64),
            _wp_from_torch(dheight_sq, wp.float64),
            _wp_from_torch(grad_normal, wp.vec3d),
            _wp_from_torch(h_grad_normal, wp.vec3d),
            _wp_from_torch(grad_grad_energy, wp.float64),
            _wp_from_torch(grad_positions, get_wp_vec_dtype(positions.dtype)),
            _wp_from_torch(grad_charges, wp.float64),
            _wp_from_torch(grad_cell, get_wp_mat_dtype(positions.dtype)),
            wp_dtype=wp_dtype,
            device=device,
        )
    return grad_grad_energy, grad_positions, grad_charges, grad_cell


def _slab_forward_launch(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    from nvalchemiops.interactions.electrostatics.slab_kernels import (
        _launch_slab_correction,
    )

    num_atoms = positions.shape[0]
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    if num_atoms == 0:
        return energies

    mz, mz2, qtotal = _run_moments(positions, charges, cell, pbc, batch_idx)
    slab_axis, slab_normal, slab_volume, slab_height_sq = _run_geometry(
        positions, cell, pbc
    )
    energy_in = torch.zeros_like(energies)

    wp_dtype = get_wp_dtype(positions.dtype)
    device = wp.device_from_torch(positions.device)
    with _scoped_stream(positions.device):
        _launch_slab_correction(
            positions=_wp_from_torch(positions, get_wp_vec_dtype(positions.dtype)),
            charges=_wp_from_torch(charges.to(positions.dtype), wp_dtype),
            batch_idx=_wp_from_torch(batch_idx, wp.int32),
            pbc=_wp_from_torch(pbc.contiguous(), wp.bool),
            cell=_wp_from_torch(cell, get_wp_mat_dtype(positions.dtype)),
            mz=_wp_from_torch(mz, wp.float64),
            mz2=_wp_from_torch(mz2, wp.float64),
            qtotal=_wp_from_torch(qtotal, wp.float64),
            slab_axis=_wp_from_torch(slab_axis, wp.int32),
            slab_normal=_wp_from_torch(slab_normal, wp.vec3d),
            slab_volume=_wp_from_torch(slab_volume, wp.float64),
            slab_height_sq=_wp_from_torch(slab_height_sq, wp.float64),
            energy_in=_wp_from_torch(energy_in, wp.float64),
            energy_out=_wp_from_torch(energies, wp.float64),
            wp_dtype=wp_dtype,
            device=device,
        )
    return energies


def _slab_backward_launch(
    grad_energy: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _cotangent_per_system_uniform(grad_energy, batch_idx, cell.shape[0]):
        return _slab_weighted_backward_values(
            positions,
            charges,
            cell,
            pbc,
            batch_idx,
            grad_energy,
        )
    grad_system = _per_system_cotangent(grad_energy, batch_idx, cell.shape[0])
    return _slab_backward_values(positions, charges, cell, pbc, batch_idx, grad_system)


def _slab_double_backward_launch(
    h_pos: torch.Tensor,
    h_charge: torch.Tensor,
    h_cell: torch.Tensor,
    grad_energy: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _cotangent_per_system_uniform(grad_energy, batch_idx, cell.shape[0]):
        return _slab_weighted_double_backward_values(
            h_pos,
            h_charge,
            h_cell,
            grad_energy,
            positions,
            charges,
            cell,
            pbc,
            batch_idx,
        )

    num_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    grad_system = _per_system_cotangent(grad_energy, batch_idx, num_systems)
    ones_system = torch.ones(num_systems, device=positions.device, dtype=torch.float64)

    base_pos, base_q, base_cell = _slab_backward_values(
        positions, charges, cell, pbc, batch_idx, ones_system
    )
    atom_dot = (base_pos.to(torch.float64) * h_pos.to(torch.float64)).sum(dim=1)
    atom_dot = atom_dot + base_q * h_charge.reshape(-1).to(torch.float64)
    system_dot = torch.zeros(num_systems, device=positions.device, dtype=torch.float64)
    if num_atoms > 0:
        system_dot = system_dot.index_add(
            0,
            batch_idx.to(device=positions.device, dtype=torch.long),
            atom_dot,
        )
    system_dot = system_dot + (
        base_cell.to(torch.float64) * h_cell.to(torch.float64)
    ).sum(dim=(1, 2))
    grad_grad_energy = _distribute_system_values(system_dot, batch_idx, num_atoms)

    grad_positions = torch.zeros_like(positions)
    grad_charges = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    grad_cell = torch.zeros_like(cell)
    if num_systems == 0:
        return grad_grad_energy, grad_positions, grad_charges, grad_cell

    from nvalchemiops.interactions.electrostatics.slab_kernels import (
        slab_correction_double_backward,
    )

    mz, mz2, qtotal = _run_moments(positions, charges, cell, pbc, batch_idx)
    slab_axis, slab_normal, slab_volume, slab_height_sq = _run_geometry(
        positions, cell, pbc
    )
    dmz = torch.zeros_like(mz)
    dmz2 = torch.zeros_like(mz2)
    dqtotal = torch.zeros_like(qtotal)
    dnormal = torch.zeros(num_systems, 3, device=positions.device, dtype=torch.float64)
    dvolume = torch.zeros(num_systems, device=positions.device, dtype=torch.float64)
    dheight_sq = torch.zeros_like(dvolume)
    grad_normal = torch.zeros_like(dnormal)
    h_grad_normal = torch.zeros_like(dnormal)

    wp_dtype = get_wp_dtype(positions.dtype)
    device = wp.device_from_torch(positions.device)
    with _scoped_stream(positions.device):
        slab_correction_double_backward(
            _wp_from_torch(positions, get_wp_vec_dtype(positions.dtype)),
            _wp_from_torch(charges.to(positions.dtype), wp_dtype),
            _wp_from_torch(
                h_pos.to(positions.dtype), get_wp_vec_dtype(positions.dtype)
            ),
            _wp_from_torch(h_charge.to(torch.float64), wp.float64),
            _wp_from_torch(h_cell.to(cell.dtype), get_wp_mat_dtype(positions.dtype)),
            _wp_from_torch(batch_idx, wp.int32),
            _wp_from_torch(pbc.contiguous(), wp.bool),
            _wp_from_torch(cell, get_wp_mat_dtype(positions.dtype)),
            _wp_from_torch(mz, wp.float64),
            _wp_from_torch(mz2, wp.float64),
            _wp_from_torch(qtotal, wp.float64),
            _wp_from_torch(slab_axis, wp.int32),
            _wp_from_torch(slab_normal, wp.vec3d),
            _wp_from_torch(slab_volume, wp.float64),
            _wp_from_torch(slab_height_sq, wp.float64),
            _wp_from_torch(grad_system, wp.float64),
            _wp_from_torch(dmz, wp.float64),
            _wp_from_torch(dmz2, wp.float64),
            _wp_from_torch(dqtotal, wp.float64),
            _wp_from_torch(dnormal, wp.vec3d),
            _wp_from_torch(dvolume, wp.float64),
            _wp_from_torch(dheight_sq, wp.float64),
            _wp_from_torch(grad_normal, wp.vec3d),
            _wp_from_torch(h_grad_normal, wp.vec3d),
            _wp_from_torch(grad_positions, get_wp_vec_dtype(positions.dtype)),
            _wp_from_torch(grad_charges, wp.float64),
            _wp_from_torch(grad_cell, get_wp_mat_dtype(positions.dtype)),
            wp_dtype=wp_dtype,
            device=device,
        )
    return grad_grad_energy, grad_positions, grad_charges, grad_cell


def _slab_forward_fake(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    del charges, cell, pbc, batch_idx
    return torch.empty(positions.shape[0], device=positions.device, dtype=torch.float64)


def _slab_backward_fake(
    grad_energy: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del grad_energy, charges, pbc, batch_idx
    return (
        torch.empty_like(positions),
        torch.empty(positions.shape[0], device=positions.device, dtype=torch.float64),
        torch.empty_like(cell),
    )


def _slab_double_backward_fake(
    h_pos: torch.Tensor,
    h_charge: torch.Tensor,
    h_cell: torch.Tensor,
    grad_energy: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del h_pos, h_charge, h_cell, grad_energy, charges, pbc, batch_idx
    return (
        torch.empty(positions.shape[0], device=positions.device, dtype=torch.float64),
        torch.empty_like(positions),
        torch.empty(positions.shape[0], device=positions.device, dtype=torch.float64),
        torch.empty_like(cell),
    )


def register_slab_ops() -> None:
    """Register the slab-correction Torch custom-op chain once."""
    global _SLAB_CHAIN, _SLAB_OPS_REGISTERED
    if _SLAB_OPS_REGISTERED:
        return

    _SLAB_CHAIN = register_warp_op_chain(
        name="nvalchemiops::slab_correction_energy",
        forward=_slab_forward_launch,
        backward=_slab_backward_launch,
        double_backward=_slab_double_backward_launch,
        diff_input_positions=(0, 1, 2),
        n_forward_inputs=5,
        second_order_diff_positions=(0, 1, 2, 3),
        n_backward_inputs=6,
        forward_fake=_slab_forward_fake,
        backward_fake=_slab_backward_fake,
        double_backward_fake=_slab_double_backward_fake,
        double_backward_return_arity=4,
    )
    _SLAB_OPS_REGISTERED = True


def slab_correction_energy(*args, **kwargs):
    """Call the registered slab-correction energy op."""
    register_slab_ops()
    if _SLAB_CHAIN is None:
        raise RuntimeError("Slab correction op registration failed")
    return _SLAB_CHAIN["forward"](*args, **kwargs)

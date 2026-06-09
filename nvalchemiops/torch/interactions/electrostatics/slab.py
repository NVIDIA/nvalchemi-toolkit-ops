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

"""PyTorch bindings for the Yeh-Berkowitz / Ballenegger slab correction."""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.interactions.electrostatics.slab_kernels import (
    _launch_slab_correction,
    _slab_reduce_moments_kernel_overload,
    slab_precompute_geometry,
)
from nvalchemiops.torch.autograd import (
    needs_grad,
    warp_from_torch,
)
from nvalchemiops.torch.interactions.electrostatics._registration import (
    ensure_electrostatics_ops_registered,
)
from nvalchemiops.torch.interactions.electrostatics._slab_chain import (
    slab_correction_energy as _slab_correction_energy_autograd,
)
from nvalchemiops.torch.interactions.electrostatics._util import (
    _build_electrostatic_result,
)
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = ["compute_slab_correction"]


def _prepare_cell(cell: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Ensure cell is 3D (B, 3, 3) and return number of systems."""
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    return cell, cell.shape[0]


def _prepare_pbc_for_slab(
    pbc: torch.Tensor | None,
    num_systems: int,
    device: torch.device,
) -> torch.Tensor:
    """Validate and normalize the pbc tensor for slab correction.

    Parameters
    ----------
    pbc : torch.Tensor or None
        Per-system periodic boundary conditions. Accepted shapes:
        - None: raises (slab correction requires explicit pbc).
        - (3,) bool: accepted only for a single system.
        - (num_systems, 3) bool: used as-is.
        Batched slab correction requires explicit per-system pbc because slab
        geometry can differ across systems.
    num_systems : int
        Number of systems in the batch.
    device : torch.device
        Target device for the output tensor.

    Returns
    -------
    torch.Tensor, shape (num_systems, 3), dtype=bool
        Validated pbc tensor.
    """
    if pbc is None:
        raise ValueError(
            "slab_correction=True requires an explicit `pbc` argument. "
            "Pass a (3,) or (B, 3) bool tensor with True for periodic "
            "directions and False for the non-periodic (slab) direction."
        )
    if pbc.dtype != torch.bool:
        raise ValueError(f"`pbc` must be a bool tensor, got dtype={pbc.dtype}.")
    if pbc.dim() == 1:
        if pbc.shape[0] != 3:
            raise ValueError(
                f"`pbc` of shape (3,) expected, got shape {tuple(pbc.shape)}."
            )
        if num_systems != 1:
            raise ValueError(
                "Batched slab correction requires `pbc` shape (B, 3); got "
                "shape (3,). Pass per-system pbc explicitly."
            )
        pbc = pbc.unsqueeze(0).contiguous()
    elif pbc.dim() == 2:
        if pbc.shape != (num_systems, 3):
            raise ValueError(
                f"`pbc` of shape ({num_systems}, 3) expected, got "
                f"shape {tuple(pbc.shape)}."
            )
    else:
        raise ValueError(
            f"`pbc` must be 1D (3,) or 2D (B, 3), got {pbc.dim()}D tensor."
        )
    return pbc.to(device=device, dtype=torch.bool)


@torch.library.custom_op(
    "nvalchemiops::_slab_correction_direct",
    mutates_args=(
        "slab_axis",
        "slab_normal",
        "slab_volume",
        "slab_height_sq",
        "mz",
        "mz2",
        "qtotal",
        "energy_in",
        "slab_energies",
        "slab_forces",
        "slab_charge_grads",
        "slab_virial",
    ),
)
def _slab_correction_direct_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    compute_forces: bool,
    compute_charge_gradients: bool,
    compute_virial: bool,
    slab_axis: torch.Tensor,
    slab_normal: torch.Tensor,
    slab_volume: torch.Tensor,
    slab_height_sq: torch.Tensor,
    mz: torch.Tensor,
    mz2: torch.Tensor,
    qtotal: torch.Tensor,
    energy_in: torch.Tensor,
    slab_energies: torch.Tensor,
    slab_forces: torch.Tensor,
    slab_charge_grads: torch.Tensor,
    slab_virial: torch.Tensor,
) -> None:
    """Fill slab direct-output tensors through a Dynamo-visible custom-op boundary."""
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    device = wp.device_from_torch(positions.device)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=False)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=False)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32, requires_grad=False)
    wp_pbc = warp_from_torch(pbc.contiguous(), wp.bool, requires_grad=False)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=False)
    wp_slab_axis = warp_from_torch(slab_axis, wp.int32, requires_grad=False)
    wp_slab_normal = warp_from_torch(slab_normal, wp.vec3d, requires_grad=False)
    wp_slab_volume = warp_from_torch(slab_volume, wp.float64, requires_grad=False)
    wp_slab_height_sq = warp_from_torch(slab_height_sq, wp.float64, requires_grad=False)

    slab_axis.zero_()
    slab_normal.zero_()
    slab_volume.zero_()
    slab_height_sq.zero_()
    mz.zero_()
    mz2.zero_()
    qtotal.zero_()
    energy_in.zero_()
    slab_energies.zero_()
    if compute_forces:
        slab_forces.zero_()
    if compute_charge_gradients:
        slab_charge_grads.zero_()
    if compute_virial:
        slab_virial.zero_()

    slab_precompute_geometry(
        wp_pbc,
        wp_cell,
        wp_slab_axis,
        wp_slab_normal,
        wp_slab_volume,
        wp_slab_height_sq,
        wp_scalar,
        device,
    )

    if num_atoms == 0:
        return

    wp_mz = warp_from_torch(mz, wp.float64, requires_grad=False)
    wp_mz2 = warp_from_torch(mz2, wp.float64, requires_grad=False)
    wp_qtotal = warp_from_torch(qtotal, wp.float64, requires_grad=False)
    wp_energy_in = warp_from_torch(energy_in, wp.float64, requires_grad=False)
    wp_slab_energies = warp_from_torch(slab_energies, wp.float64, requires_grad=False)
    wp_slab_forces = (
        warp_from_torch(slab_forces, wp_vec, requires_grad=False)
        if compute_forces
        else None
    )
    wp_slab_charge_grads = (
        warp_from_torch(slab_charge_grads, wp.float64, requires_grad=False)
        if compute_charge_gradients
        else None
    )
    wp_slab_virial = (
        warp_from_torch(slab_virial, wp_mat, requires_grad=False)
        if compute_virial
        else None
    )

    wp.launch(
        _slab_reduce_moments_kernel_overload[wp_scalar],
        dim=num_atoms,
        inputs=[
            wp_positions,
            wp_charges,
            wp_batch_idx,
            wp_pbc,
            wp_cell,
            wp_mz,
            wp_mz2,
            wp_qtotal,
        ],
        device=device,
    )

    _launch_slab_correction(
        positions=wp_positions,
        charges=wp_charges,
        batch_idx=wp_batch_idx,
        pbc=wp_pbc,
        cell=wp_cell,
        mz=wp_mz,
        mz2=wp_mz2,
        qtotal=wp_qtotal,
        slab_axis=wp_slab_axis,
        slab_normal=wp_slab_normal,
        slab_volume=wp_slab_volume,
        slab_height_sq=wp_slab_height_sq,
        energy_in=wp_energy_in,
        energy_out=wp_slab_energies,
        wp_dtype=wp_scalar,
        forces=wp_slab_forces,
        charge_grads=wp_slab_charge_grads,
        virial=wp_slab_virial,
        compute_forces=compute_forces,
        compute_charge_gradients=compute_charge_gradients,
        compute_virial=compute_virial,
        device=device,
    )


@_slab_correction_direct_op.register_fake
def _(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    compute_forces: bool,
    compute_charge_gradients: bool,
    compute_virial: bool,
    slab_axis: torch.Tensor,
    slab_normal: torch.Tensor,
    slab_volume: torch.Tensor,
    slab_height_sq: torch.Tensor,
    mz: torch.Tensor,
    mz2: torch.Tensor,
    qtotal: torch.Tensor,
    energy_in: torch.Tensor,
    slab_energies: torch.Tensor,
    slab_forces: torch.Tensor,
    slab_charge_grads: torch.Tensor,
    slab_virial: torch.Tensor,
) -> None:
    return


def _run_slab_correction_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Run the Ewald-style slab custom-op implementation."""
    num_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    input_dtype = positions.dtype
    slab_axis = torch.empty(num_systems, device=cell.device, dtype=torch.int32)
    slab_normal = torch.empty(num_systems, 3, device=cell.device, dtype=torch.float64)
    slab_volume = torch.empty(num_systems, device=cell.device, dtype=torch.float64)
    slab_height_sq = torch.empty_like(slab_volume)
    mz_t = torch.empty(num_systems, 3, device=positions.device, dtype=torch.float64)
    mz2_t = torch.empty_like(mz_t)
    qtotal_t = torch.empty(num_systems, device=positions.device, dtype=torch.float64)
    energy_in_t = torch.empty(num_atoms, device=positions.device, dtype=torch.float64)
    slab_energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)

    output_tensors: list[torch.Tensor] = [slab_energies]

    if compute_forces:
        slab_forces = torch.zeros(
            num_atoms, 3, device=positions.device, dtype=input_dtype
        )
        output_tensors.append(slab_forces)
    else:
        slab_forces = torch.empty(0, 3, device=positions.device, dtype=input_dtype)

    if compute_charge_gradients:
        slab_charge_grads = torch.zeros(
            num_atoms, device=positions.device, dtype=torch.float64
        )
        output_tensors.append(slab_charge_grads)
    else:
        slab_charge_grads = torch.empty(0, device=positions.device, dtype=torch.float64)

    if compute_virial:
        slab_virial = torch.zeros(
            num_systems, 3, 3, device=positions.device, dtype=input_dtype
        )
        output_tensors.append(slab_virial)
    else:
        slab_virial = torch.empty(0, 3, 3, device=positions.device, dtype=input_dtype)

    _slab_correction_direct_op(
        positions,
        charges,
        cell,
        pbc,
        batch_idx,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
        slab_axis,
        slab_normal,
        slab_volume,
        slab_height_sq,
        mz_t,
        mz2_t,
        qtotal_t,
        energy_in_t,
        slab_energies,
        slab_forces,
        slab_charge_grads,
        slab_virial,
    )

    return tuple(output_tensors)


def compute_slab_correction(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Yeh-Berkowitz slab correction for 2D periodic electrostatics, with the
    Ballenegger et al. (2009) Eq. 29 extension for non-neutral systems.

    Returns the slab-correction contribution (per-atom energy and optionally
    per-atom force, charge gradient, and per-system virial). The caller adds
    these to the corresponding 3D Ewald/PME quantities; in normal usage the
    correction is invoked through ``ewald_summation(..., slab_correction=True)``
    or ``particle_mesh_ewald(..., slab_correction=True)``.

    Background-charge convention
    ----------------------------
    For systems with net charge :math:`Q \\ne 0`, the formula corresponds to a
    uniform-volume neutralizing background (the same convention used by
    standard 3D Ewald). This matches LAMMPS and Ballenegger et al. (2009)
    Eq. 29.

    Cell geometry
    -------------
    Orthorhombic and triclinic cells are supported. For triclinic slab
    systems, the slab normal follows the plane spanned by the two periodic
    cell vectors.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates.
    charges : torch.Tensor, shape (N,)
        Atomic charges.
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrices.
    pbc : torch.Tensor, shape (3,) or (B, 3), dtype=bool
        Per-system periodic boundary conditions. True for periodic directions,
        False for the non-periodic (slab) direction. Systems whose pbc is not
        slab-like (i.e., has anything other than exactly one False entry)
        contribute zero. A (3,) tensor is accepted only for single-system
        calls; batched calls require explicit (B, 3) per-system pbc.
    batch_idx : torch.Tensor, shape (N,), dtype=int32, optional
        System index for each atom. Defaults to all zeros (single system). When
        provided, atoms must be grouped by system: ``batch_idx`` must be
        contiguous, nondecreasing, and use system IDs ``0..B-1``.
    compute_forces : bool, default=False
        If True, return per-atom forces.
    compute_charge_gradients : bool, default=False
        If True, return per-atom charge gradients dE_slab/dq_i.
    compute_virial : bool, default=False
        If True, return per-system virial tensor using the normal-following
        affine strain convention W = E_slab * (I - 2 n n^T).

    Returns
    -------
    energies : torch.Tensor, shape (N,), dtype=float64
        Per-atom slab correction energy.
    forces : torch.Tensor, shape (N, 3), dtype matches positions, optional
        Per-atom slab force (only returned if compute_forces=True).
    charge_grads : torch.Tensor, shape (N,), dtype=float64, optional
        Per-atom slab charge gradient (only if compute_charge_gradients=True).
    virial : torch.Tensor, shape (B, 3, 3), dtype matches positions, optional
        Per-system slab virial (only if compute_virial=True).

    Examples
    --------
    Standalone correction for an orthorhombic slab with vacuum along z::

        >>> pbc_slab = torch.tensor([[True, True, False]], device=positions.device)
        >>> slab_energy, slab_forces = compute_slab_correction(
        ...     positions, charges, cell, pbc_slab, compute_forces=True
        ... )
        >>> corrected_energy = ewald_energy + slab_energy

    Triclinic cells use the normal to the periodic plane::

        >>> triclinic_energy, triclinic_forces = compute_slab_correction(
        ...     positions, charges, triclinic_cell, pbc_slab, compute_forces=True
        ... )
    """
    ensure_electrostatics_ops_registered()
    cell, num_systems = _prepare_cell(cell)
    pbc = _prepare_pbc_for_slab(pbc, num_systems, positions.device)

    if batch_idx is None:
        batch_idx = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=positions.device
        )

    grad_enabled = needs_grad(positions, charges, cell)
    direct_positions = positions.detach() if grad_enabled else positions
    direct_charges = charges.detach() if grad_enabled else charges
    direct_cell = cell.detach() if grad_enabled else cell

    if compute_forces or compute_charge_gradients or compute_virial:
        direct_outputs = _run_slab_correction_op(
            direct_positions,
            direct_charges,
            direct_cell,
            pbc,
            batch_idx,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
        )
        out_idx = 0
        energies = direct_outputs[out_idx]
        out_idx += 1
        forces = direct_outputs[out_idx] if compute_forces else None
        out_idx += int(compute_forces)
        charge_grads = direct_outputs[out_idx] if compute_charge_gradients else None
        out_idx += int(compute_charge_gradients)
        virial = direct_outputs[out_idx] if compute_virial else None
        if grad_enabled:
            energies = _slab_correction_energy_autograd(
                positions, charges, cell, pbc, batch_idx
            )
        return _build_electrostatic_result(
            energies,
            forces,
            charge_grads,
            virial,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )

    if not grad_enabled:
        return _run_slab_correction_op(positions, charges, cell, pbc, batch_idx)[0]

    return _slab_correction_energy_autograd(positions, charges, cell, pbc, batch_idx)

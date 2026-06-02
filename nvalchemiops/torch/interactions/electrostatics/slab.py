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
)
from nvalchemiops.torch.autograd import (
    OutputSpec,
    WarpAutogradContextManager,
    attach_for_backward,
    needs_grad,
    warp_custom_op,
    warp_from_torch,
)
from nvalchemiops.torch.interactions.electrostatics._util import (
    ElectrostaticOutputs,
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


def _slab_energy_output() -> OutputSpec:
    return OutputSpec("slab_energies", wp.float64, lambda pos, *_: (pos.shape[0],))


def _slab_forces_output() -> OutputSpec:
    return OutputSpec(
        "slab_forces",
        lambda pos, *_: get_wp_vec_dtype(pos.dtype),
        lambda pos, *_: (pos.shape[0], 3),
    )


def _slab_charge_grads_output() -> OutputSpec:
    return OutputSpec("slab_charge_grads", wp.float64, lambda pos, *_: (pos.shape[0],))


def _slab_virial_output() -> OutputSpec:
    return OutputSpec(
        "slab_virial",
        lambda pos, *_: get_wp_mat_dtype(pos.dtype),
        lambda pos, charges, cell, *_: (cell.shape[0], 3, 3),
    )


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
    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)
    need_force_kernel = compute_forces or compute_charge_gradients or compute_virial
    virial_grad = needs_grad_flag and compute_virial

    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_pbc = warp_from_torch(pbc.contiguous(), wp.bool)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)

    mz_t = torch.zeros(num_systems, 3, device=positions.device, dtype=torch.float64)
    mz2_t = torch.zeros(num_systems, 3, device=positions.device, dtype=torch.float64)
    qtotal_t = torch.zeros(num_systems, device=positions.device, dtype=torch.float64)
    wp_mz = warp_from_torch(mz_t, wp.float64, requires_grad=needs_grad_flag)
    wp_mz2 = warp_from_torch(mz2_t, wp.float64, requires_grad=needs_grad_flag)
    wp_qtotal = warp_from_torch(qtotal_t, wp.float64, requires_grad=needs_grad_flag)

    energy_in_t = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_energy_in = warp_from_torch(
        energy_in_t, wp.float64, requires_grad=needs_grad_flag
    )

    slab_energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_slab_energies = warp_from_torch(
        slab_energies, wp.float64, requires_grad=needs_grad_flag
    )

    output_tensors: list[torch.Tensor] = [slab_energies]
    backward_kw = {
        "slab_energies": wp_slab_energies,
        "positions": wp_positions,
        "charges": wp_charges,
        "cell": wp_cell,
    }

    wp_slab_forces = None
    if need_force_kernel:
        slab_forces = torch.zeros(
            num_atoms, 3, device=positions.device, dtype=input_dtype
        )
        wp_slab_forces = warp_from_torch(
            slab_forces, wp_vec, requires_grad=needs_grad_flag
        )
        output_tensors.append(slab_forces)
        backward_kw["slab_forces"] = wp_slab_forces

    wp_slab_charge_grads = None
    if compute_charge_gradients:
        slab_charge_grads = torch.zeros(
            num_atoms, device=positions.device, dtype=torch.float64
        )
        wp_slab_charge_grads = warp_from_torch(
            slab_charge_grads, wp.float64, requires_grad=needs_grad_flag
        )
        output_tensors.append(slab_charge_grads)
        backward_kw["slab_charge_grads"] = wp_slab_charge_grads

    wp_slab_virial = None
    if need_force_kernel:
        slab_virial = torch.zeros(
            num_systems, 3, 3, device=positions.device, dtype=input_dtype
        )
        wp_slab_virial = warp_from_torch(slab_virial, wp_mat, requires_grad=virial_grad)
        output_tensors.append(slab_virial)
        if virial_grad:
            backward_kw["slab_virial"] = wp_slab_virial

    with WarpAutogradContextManager(needs_grad_flag) as tape:
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

    if needs_grad_flag:
        attach_for_backward(slab_energies, tape=tape, **backward_kw)
        stable_outputs = [slab_energies]
        stable_outputs.extend(tensor.clone() for tensor in output_tensors[1:])
        return tuple(stable_outputs)

    return tuple(output_tensors)


@warp_custom_op(
    name="alchemiops::_slab_correction_energy",
    outputs=[_slab_energy_output()],
    grad_arrays=["slab_energies", "positions", "charges", "cell"],
)
def _slab_correction_energy_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Compute slab correction energy only."""
    return _run_slab_correction_op(positions, charges, cell, pbc, batch_idx)[0]


@warp_custom_op(
    name="alchemiops::_slab_correction_energy_forces",
    outputs=[_slab_energy_output(), _slab_forces_output(), _slab_virial_output()],
    grad_arrays=[
        "slab_energies",
        "slab_forces",
        "slab_virial",
        "positions",
        "charges",
        "cell",
    ],
)
def _slab_correction_energy_forces_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute slab correction energy, forces, and optionally virial."""
    return _run_slab_correction_op(
        positions,
        charges,
        cell,
        pbc,
        batch_idx,
        compute_forces=True,
        compute_virial=compute_virial,
    )


@warp_custom_op(
    name="alchemiops::_slab_correction_energy_forces_charge_grad",
    outputs=[
        _slab_energy_output(),
        _slab_forces_output(),
        _slab_charge_grads_output(),
        _slab_virial_output(),
    ],
    grad_arrays=[
        "slab_energies",
        "slab_forces",
        "slab_charge_grads",
        "slab_virial",
        "positions",
        "charges",
        "cell",
    ],
)
def _slab_correction_energy_forces_charge_grad_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute slab correction energy, forces, charge gradients, and optionally virial."""
    return _run_slab_correction_op(
        positions,
        charges,
        cell,
        pbc,
        batch_idx,
        compute_forces=True,
        compute_charge_gradients=True,
        compute_virial=compute_virial,
    )


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
        System index for each atom. Defaults to all zeros (single system).
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
    cell, num_systems = _prepare_cell(cell)
    pbc = _prepare_pbc_for_slab(pbc, num_systems, positions.device)

    if batch_idx is None:
        batch_idx = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=positions.device
        )

    if compute_charge_gradients:
        energies, forces, charge_grads, virial = (
            _slab_correction_energy_forces_charge_grad_op(
                positions,
                charges,
                cell,
                pbc,
                batch_idx,
                compute_virial=compute_virial,
            )
        )
        return _build_electrostatic_result(
            ElectrostaticOutputs(energies, forces, charge_grads, virial),
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )

    if compute_forces or compute_virial:
        energies, forces, virial = _slab_correction_energy_forces_op(
            positions,
            charges,
            cell,
            pbc,
            batch_idx,
            compute_virial=compute_virial,
        )
        return _build_electrostatic_result(
            ElectrostaticOutputs(energies, forces, virial=virial),
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )

    return _slab_correction_energy_op(positions, charges, cell, pbc, batch_idx)

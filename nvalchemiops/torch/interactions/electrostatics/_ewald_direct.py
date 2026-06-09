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

"""Direct (non-autograd) factory-kernel outputs for the legacy Ewald flags.

The public ``compute_forces`` / ``compute_charge_gradients`` / ``compute_virial`` /
``hybrid_forces`` flags return the same legacy tuples as before -- explicit forces
(``-dE/dR``), charge gradients (``dE/dq``) and row-vector displacement virial
(``W = -dE/dstrain``) computed directly from the factory ``order="forward"``
kernels (``_DerivState.E_F`` / ``E_F_dQ`` + ``cell_grad`` for the virial).
These match the hand-written kernels bit-exactly with the legacy kernels and run
tape-free. The energy itself
is produced and connected to autograd separately by the explicit chains; this module
only fills the extra (forward-only) tuple slots.
"""

from __future__ import annotations

import math

import torch
import warp as wp

from nvalchemiops.interactions.electrostatics._factory_common import _DerivState
from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    BATCH_BLOCK_SIZE,
    REAL_SPACE_TILED_BLOCK_DIM,
)
from nvalchemiops.interactions.electrostatics.ewald_real_factory import (
    alloc_ewald_real_sentinels,
    get_ewald_real_kernel,
)
from nvalchemiops.interactions.electrostatics.ewald_recip_factory import (
    alloc_ewald_recip_sentinels,
    get_ewald_recip_kernel,
)
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = [
    "real_space_direct",
    "reciprocal_space_direct",
]

_PI = math.pi


def _wp(tensor: torch.Tensor, dtype):
    return wp.from_torch(tensor.detach().contiguous(), dtype=dtype, requires_grad=False)


def _scoped_stream(device: torch.device):
    if device.type != "cuda":
        from contextlib import nullcontext

        return nullcontext()
    return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device)))


# ===========================================================================
# Real-space direct outputs (forces / charge_gradients / virial)
# ===========================================================================


@torch.library.custom_op(
    "nvalchemiops::_ewald_real_space_direct",
    mutates_args=("forces", "charge_grads", "virial"),
)
def _real_space_direct_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor | None,
    idx_j: torch.Tensor | None,
    neighbor_ptr: torch.Tensor | None,
    neighbor_shifts: torch.Tensor | None,
    neighbor_matrix: torch.Tensor | None,
    neighbor_matrix_shifts: torch.Tensor | None,
    mask_value: int,
    want_charge_grad: bool,
    want_virial: bool,
    forces: torch.Tensor,
    charge_grads: torch.Tensor,
    virial: torch.Tensor,
) -> None:
    """Fill direct real-space forces, charge gradients, and virial."""
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    batched = batch_idx is not None
    use_matrix = neighbor_matrix is not None

    empty = (
        (neighbor_matrix.shape[0] == 0 or neighbor_matrix.shape[1] == 0)
        if use_matrix
        else (idx_j is None or idx_j.shape[0] == 0)
    )
    if num_atoms == 0 or empty:
        return

    deriv_state = _DerivState.E_F_dQ if want_charge_grad else _DerivState.E_F
    sentinels = alloc_ewald_real_sentinels(wp_scalar, device)
    if use_matrix:
        nbr = (
            sentinels["idx_j"],
            sentinels["neighbor_ptr"],
            sentinels["unit_shifts"],
            _wp(neighbor_matrix, wp.int32),
            _wp(neighbor_matrix_shifts, wp.vec3i),
        )
    else:
        nbr = (
            _wp(idx_j, wp.int32),
            _wp(neighbor_ptr, wp.int32),
            _wp(neighbor_shifts, wp.vec3i),
            sentinels["neighbor_matrix"],
            sentinels["unit_shifts_matrix"],
        )
    kernel = get_ewald_real_kernel(
        wp_scalar,
        batched=batched,
        neighbor_input="matrix" if use_matrix else "list",
        deriv_state=deriv_state,
        cell_grad=want_virial,
        order="forward",
        tiled=use_matrix,
    )
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_batch = _wp(batch_idx, wp.int32) if batched else sentinels["batch_id"]
    wp_cg = (
        _wp(charge_grads, wp.float64)
        if want_charge_grad
        else sentinels["charge_gradients"]
    )
    wp_virial = _wp(virial, wp_mat) if want_virial else sentinels["virial"]
    launch_inputs = [
        _wp(positions, wp_vec),
        _wp(charges, wp_scalar),
        _wp(cell, wp_mat),
        wp_batch,
        *nbr,
        int(mask_value),
        _wp(alpha, wp_scalar),
        _wp(energies, wp.float64),
        _wp(forces, wp_vec),
        wp_cg,
        wp_virial,
    ]
    with _scoped_stream(positions.device):
        # Neighbor-matrix uses the cooperative-block (tiled) kernel; CSR stays
        # one-thread-per-atom.
        if use_matrix:
            wp.launch_tiled(
                kernel,
                dim=[num_atoms],
                inputs=launch_inputs,
                block_dim=REAL_SPACE_TILED_BLOCK_DIM,
                device=device,
            )
        else:
            wp.launch(kernel, dim=[num_atoms], inputs=launch_inputs, device=device)
    return


@_real_space_direct_op.register_fake
def _(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor | None,
    idx_j: torch.Tensor | None,
    neighbor_ptr: torch.Tensor | None,
    neighbor_shifts: torch.Tensor | None,
    neighbor_matrix: torch.Tensor | None,
    neighbor_matrix_shifts: torch.Tensor | None,
    mask_value: int,
    want_charge_grad: bool,
    want_virial: bool,
    forces: torch.Tensor,
    charge_grads: torch.Tensor,
    virial: torch.Tensor,
) -> None:
    return


def real_space_direct(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    *,
    batch_idx: torch.Tensor | None,
    idx_j: torch.Tensor | None,
    neighbor_ptr: torch.Tensor | None,
    neighbor_shifts: torch.Tensor | None,
    neighbor_matrix: torch.Tensor | None,
    neighbor_matrix_shifts: torch.Tensor | None,
    mask_value: int,
    want_charge_grad: bool,
    want_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (forces, charge_gradients, virial) directly from the real-space forward kernel.

    Returns physical forces (``-dE/dR``), ``dE/dq`` and the strain virial ``W``;
    slots not requested are still allocated (zeros) so callers can index uniformly.
    """
    num_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=positions.dtype)
    charge_grads = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    virial = torch.zeros(
        num_systems, 3, 3, device=positions.device, dtype=positions.dtype
    )
    _real_space_direct_op(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        idx_j,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        mask_value,
        want_charge_grad,
        want_virial,
        forces,
        charge_grads,
        virial,
    )
    return forces, charge_grads, virial


# ===========================================================================
# Reciprocal-space direct outputs (forces / charge_gradients / virial)
# ===========================================================================


@torch.library.custom_op(
    "nvalchemiops::_ewald_reciprocal_space_direct",
    mutates_args=(
        "energies",
        "forces",
        "charge_grads",
        "virial",
        "cos_kr",
        "sin_kr",
        "real_sf",
        "imag_sf",
        "total_charges",
    ),
)
def _reciprocal_space_direct_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors_2d: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor | None,
    atom_start: torch.Tensor | None,
    atom_end: torch.Tensor | None,
    want_charge_grad: bool,
    want_virial: bool,
    energies: torch.Tensor,
    forces: torch.Tensor,
    charge_grads: torch.Tensor,
    virial: torch.Tensor,
    cos_kr: torch.Tensor,
    sin_kr: torch.Tensor,
    real_sf: torch.Tensor,
    imag_sf: torch.Tensor,
    total_charges: torch.Tensor,
) -> None:
    """Fill direct reciprocal-space energy, forces, charge gradients, and virial."""
    num_atoms = positions.shape[0]
    num_k = k_vectors_2d.shape[-2]
    num_systems = cell.shape[0]
    input_dtype = positions.dtype
    device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    batched = batch_idx is not None

    if num_atoms == 0:
        return

    if num_k > 0:
        deriv_state = _DerivState.E_F_dQ if want_charge_grad else _DerivState.E_F
        bundle = get_ewald_recip_kernel(
            wp_scalar,
            batched=batched,
            deriv_state=deriv_state,
            cell_grad=want_virial,
            order="forward",
        )
        s = alloc_ewald_recip_sentinels(wp_scalar, device)
        cos_kr_wp, sin_kr_wp, real_sf_wp, imag_sf_wp = _fill(
            bundle,
            positions,
            charges,
            cell,
            k_vectors_2d,
            alpha,
            batch_idx,
            atom_start,
            atom_end,
            num_k,
            num_systems,
            num_atoms,
            wp_scalar,
            wp_vec,
            wp_mat,
            device,
            cos_kr,
            sin_kr,
            real_sf,
            imag_sf,
            total_charges,
        )
        batch_id = (
            _wp(batch_idx, wp.int32)
            if batched
            else wp.empty((0,), dtype=wp.int32, device=device)
        )
        wp_cg = (
            _wp(charge_grads, wp.float64) if want_charge_grad else s["charge_gradients"]
        )
        with _scoped_stream(positions.device):
            wp.launch(
                bundle.compute,
                dim=num_atoms,
                inputs=[
                    _wp(charges, wp_scalar),
                    batch_id,
                    _wp(k_vectors_2d, wp_vec),
                    cos_kr_wp,
                    sin_kr_wp,
                    real_sf_wp,
                    imag_sf_wp,
                    s["grad_energy"],
                    _wp(energies, wp.float64),
                    _wp(forces, wp_vec),
                    wp_cg,
                ],
                device=device,
            )
            if want_virial:
                volume = torch.abs(torch.det(cell.to(torch.float64))).reshape(
                    num_systems
                )
                wp_vol = _wp(volume, wp.float64)
                wp_virial = _wp(virial, wp_mat)
                if batched:
                    wp.launch(
                        bundle.virial,
                        dim=(num_k, num_systems),
                        inputs=[
                            _wp(k_vectors_2d, wp_vec),
                            _wp(alpha, wp_scalar),
                            wp_vol,
                            real_sf_wp,
                            imag_sf_wp,
                            wp_virial,
                        ],
                        device=device,
                    )
                else:
                    # The reused single-system virial kernel takes 1D k_vectors / S(k).
                    kv_1d = _wp(k_vectors_2d.reshape(num_k, 3), wp_vec)
                    wp.launch(
                        bundle.virial,
                        dim=num_k,
                        inputs=[
                            kv_1d,
                            _wp(alpha, wp_scalar),
                            wp_vol,
                            real_sf_wp.reshape((num_k,)),
                            imag_sf_wp.reshape((num_k,)),
                            wp_virial,
                        ],
                        device=device,
                    )

    # Charge-gradient self + background corrections (Torch; matches the legacy op):
    #   dE_self/dq_i = 2 alpha q_i / sqrt(pi);  dE_bg/dq_i = pi (Q_tot / V) / alpha^2.
    if want_charge_grad:
        charges64 = charges.to(torch.float64)
        if batched:
            bidx = batch_idx
            alpha_atom = alpha.to(torch.float64).index_select(0, bidx)
            vol = torch.abs(torch.linalg.det(cell.to(torch.float64)))  # (S,)
            q_over_v = torch.zeros(
                num_systems, dtype=torch.float64, device=positions.device
            )
            q_over_v = q_over_v.index_add(0, bidx, charges64) / vol
            q_over_v_atom = q_over_v.index_select(0, bidx)
        else:
            alpha_atom = alpha.to(torch.float64)[0]
            vol = torch.abs(torch.det(cell[0].to(torch.float64)))
            q_over_v_atom = charges64.sum() / vol
        self_grad = 2.0 * alpha_atom / math.sqrt(_PI) * charges64
        bg_grad = _PI / (alpha_atom * alpha_atom) * q_over_v_atom
        charge_grads.sub_(self_grad + bg_grad)

    # Virial background correction: W_bg = -E_bg I.
    if want_virial:
        if batched:
            q_tot = torch.zeros(num_systems, dtype=input_dtype, device=positions.device)
            q_tot = q_tot.index_add(0, batch_idx, charges.to(input_dtype))
            vol = torch.abs(torch.linalg.det(cell)).to(input_dtype)
            alpha_b = alpha.to(input_dtype)
            e_bg = _PI * q_tot**2 / (2.0 * alpha_b**2 * vol)
            eye = torch.eye(3, device=positions.device, dtype=input_dtype)
            virial.sub_(e_bg[:, None, None] * eye)
        else:
            q_tot = charges.sum().to(input_dtype)
            vol = torch.abs(torch.det(cell[0].to(input_dtype)))
            alpha_v = alpha.to(input_dtype).squeeze()
            e_bg = _PI * q_tot**2 / (2.0 * alpha_v**2 * vol)
            eye = torch.eye(3, device=positions.device, dtype=input_dtype)
            virial.sub_(e_bg * eye)

    return


@_reciprocal_space_direct_op.register_fake
def _(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors_2d: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor | None,
    atom_start: torch.Tensor | None,
    atom_end: torch.Tensor | None,
    want_charge_grad: bool,
    want_virial: bool,
    energies: torch.Tensor,
    forces: torch.Tensor,
    charge_grads: torch.Tensor,
    virial: torch.Tensor,
    cos_kr: torch.Tensor,
    sin_kr: torch.Tensor,
    real_sf: torch.Tensor,
    imag_sf: torch.Tensor,
    total_charges: torch.Tensor,
) -> None:
    return


def reciprocal_space_direct(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors_2d: torch.Tensor,
    alpha: torch.Tensor,
    *,
    batch_idx: torch.Tensor | None,
    atom_start: torch.Tensor | None,
    atom_end: torch.Tensor | None,
    want_charge_grad: bool,
    want_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute k-space energy, forces, charge gradients, and virial directly.

    The returned energy is the reciprocal k-space sum only. Callers that expose
    reciprocal energies must still apply the self/background corrections.
    Forces are ``-dE/dR`` from the k-sum; charge gradients are the full reciprocal
    ``dE/dq`` -- the k-sum potential ``phi`` minus the self / background derivatives
    ``2 alpha q_i / sqrt(pi)`` and ``pi (Q_tot / V) / alpha^2`` (the legacy
    ``compute_charge_gradients`` op applies these corrections to the kernel ``phi``);
    virial is ``W`` from the reused k-major virial kernel minus the background
    ``-E_bg I`` term -- matching the legacy reciprocal tuple bit-for-bit.
    """
    num_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=positions.dtype)
    charge_grads = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    virial = torch.zeros(
        num_systems, 3, 3, device=positions.device, dtype=positions.dtype
    )
    cos_kr = torch.empty(
        k_vectors_2d.shape[-2],
        num_atoms,
        device=positions.device,
        dtype=torch.float64,
    )
    sin_kr = torch.empty_like(cos_kr)
    real_sf = torch.zeros(
        num_systems,
        k_vectors_2d.shape[-2],
        device=positions.device,
        dtype=torch.float64,
    )
    imag_sf = torch.zeros_like(real_sf)
    total_charges = torch.zeros(
        num_systems,
        device=positions.device,
        dtype=torch.float64,
    )
    _reciprocal_space_direct_op(
        positions,
        charges,
        cell,
        k_vectors_2d,
        alpha,
        batch_idx,
        atom_start,
        atom_end,
        want_charge_grad,
        want_virial,
        energies,
        forces,
        charge_grads,
        virial,
        cos_kr,
        sin_kr,
        real_sf,
        imag_sf,
        total_charges,
    )
    return energies, forces, charge_grads, virial


def _fill(
    bundle,
    positions,
    charges,
    cell,
    k_vectors_2d,
    alpha,
    batch_idx,
    atom_start,
    atom_end,
    num_k,
    num_systems,
    num_atoms,
    wp_scalar,
    wp_vec,
    wp_mat,
    device,
    cos_kr,
    sin_kr,
    real_sf,
    imag_sf,
    total_charges,
):
    """Run the reused hand-written structure-factor fill kernel (2D (S, K) output)."""
    batched = batch_idx is not None
    real_sf.zero_()
    imag_sf.zero_()
    total_charges.zero_()
    wp_cos_kr = _wp(cos_kr, wp.float64)
    wp_sin_kr = _wp(sin_kr, wp.float64)
    wp_cell = _wp(cell, wp_mat)
    wp_alpha = _wp(alpha, wp_scalar)
    wp_pos = _wp(positions, wp_vec)
    wp_chg = _wp(charges, wp_scalar)
    with _scoped_stream(positions.device):
        if batched:
            wp_as = _wp(atom_start, wp.int32)
            wp_ae = _wp(atom_end, wp.int32)
            max_atoms = int((atom_end - atom_start).max().item()) if num_atoms else 0
            max_blocks = max((max_atoms + BATCH_BLOCK_SIZE - 1) // BATCH_BLOCK_SIZE, 1)
            wp.launch(
                bundle.fill,
                dim=(num_k, num_systems, max_blocks),
                inputs=[
                    wp_pos,
                    wp_chg,
                    _wp(k_vectors_2d, wp_vec),
                    wp_cell,
                    wp_alpha,
                    wp_as,
                    wp_ae,
                    _wp(total_charges, wp.float64),
                    wp_cos_kr,
                    wp_sin_kr,
                    _wp(real_sf, wp.float64),
                    _wp(imag_sf, wp.float64),
                ],
                device=device,
            )
        else:
            kv_1d = _wp(k_vectors_2d.reshape(num_k, 3), wp_vec)
            wp.launch(
                bundle.fill,
                dim=num_k,
                inputs=[
                    wp_pos,
                    wp_chg,
                    kv_1d,
                    wp_cell,
                    wp_alpha,
                    _wp(total_charges.reshape(1), wp.float64),
                    wp_cos_kr,
                    wp_sin_kr,
                    _wp(real_sf.reshape(num_k), wp.float64),
                    _wp(imag_sf.reshape(num_k), wp.float64),
                ],
                device=device,
            )
    return (
        _wp(cos_kr, wp.float64),
        _wp(sin_kr, wp.float64),
        _wp(real_sf, wp.float64),
        _wp(imag_sf, wp.float64),
    )

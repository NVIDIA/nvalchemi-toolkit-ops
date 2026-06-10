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

"""Parity and derivative tests for the ``ewald_recip`` factory kernels.

Five guarantees (adapted to the multi-stage reciprocal sum):

1. **Forward parity** -- the factory atom-major ``compute`` E / F / dE/dq outputs
   are bit-exact (``np.array_equal``) vs the relevant hand-written launchers, for
   ``wp.float32`` + ``wp.float64`` x single/batch. Force-only ``E_F`` keeps the
   energy bit-exact with the energy-only launcher while its force output remains
   bit-exact with ``_ewald_reciprocal_space_energy_forces``. ``S(k)`` / virial come
   from the *reused* hand-written kernels, so they are bit-exact by construction.
2. **Backward scaling** -- ``order="backward"`` with ``grad_E = 1`` reproduces the
   forward first-derivatives (``grad_R = -F``, ``grad_q = phi``); ``backward(grad_E)
   == grad_E * backward(1)`` per system. The backward virial bakes ``grad_energy`` in
   (``ge * W``); a ``ge != 1`` check pins the scaling.
3. **Finite-diff** -- the backward outputs vs central-difference forces / charge-grad
   of the factory k-sum-only energy, and the backward virial vs strain-first FD
   virial (k-sum only, background excluded).
4. **Double-backward finite-diff** -- the ``order="double_backward"`` outputs
   (``grad_positions`` / ``grad_charges`` / ``grad_grad_energy``) vs a
   central-difference of the backward-kernel outputs contracted with the cotangents
   (no autograd graph, per the brief).
5. **F3 harness parity** -- the backward forces / charge-grad / virial vs the SHARED
   F3 ``_deriv_check`` primitives (``fd_forces`` / ``fd_charge_grad`` /
   ``fd_strain_virial``) on a ``fixed_charge_system`` CsCl crystal, reported against
   the F3-certified baselines.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from nvalchemiops.interactions.electrostatics._factory_common import _DerivState
from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    BATCH_BLOCK_SIZE,
    _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_cellgrad,
    batch_ewald_reciprocal_space_compute_energy,
    batch_ewald_reciprocal_space_energy_forces,
    batch_ewald_reciprocal_space_energy_forces_charge_grad,
    batch_ewald_reciprocal_space_fill_structure_factors,
    ewald_reciprocal_space_compute_energy,
    ewald_reciprocal_space_energy_forces,
    ewald_reciprocal_space_energy_forces_charge_grad,
    ewald_reciprocal_space_fill_structure_factors,
)
from nvalchemiops.interactions.electrostatics.ewald_recip_factory import (
    _make_backward_kspace_from_cache_kernel,
    alloc_ewald_recip_sentinels,
    get_ewald_recip_kernel,
)
from test.interactions.electrostatics._deriv_check import (
    fd_charge_grad,
    fd_forces,
    fd_strain_virial,
    fixed_charge_system,
    max_abs_rel,
)
from test.interactions.electrostatics.conftest import create_cscl_supercell

_DTYPES = [wp.float32, wp.float64]
_DTYPE_IDS = ["f32", "f64"]
_ALPHA = 0.4
_NPF = {wp.float32: np.float32, wp.float64: np.float64}
_VEC = {wp.float32: wp.vec3f, wp.float64: wp.vec3d}
_MAT = {wp.float32: wp.mat33f, wp.float64: wp.mat33d}


# ==============================================================================
# Systems (asymmetric -> nonzero forces/virial; neutral -> well-defined Ewald)
# ==============================================================================


def _half_space_k_vectors(L, n_max=2):
    """Half-space reciprocal-lattice vectors 2*pi/L * (nx,ny,nz), excluding k=0/-k."""
    k_factor = 2.0 * np.pi / L
    # Half-space: nz > 0, or (nz == 0 and (ny > 0, or ny == 0 and nx > 0)).
    ks = [
        [nx, ny, nz]
        for nx in range(-n_max, n_max + 1)
        for ny in range(-n_max, n_max + 1)
        for nz in range(0, n_max + 1)
        if nz > 0 or (nz == 0 and (ny > 0 or (ny == 0 and nx > 0)))
    ]
    return (np.array(ks, dtype=np.float64) * k_factor).copy()


def _single_system():
    """Four atoms (neutral) in a cubic box with half-space k-vectors."""
    L = 8.0
    positions = np.array(
        [
            [0.40, 0.70, 0.20],
            [3.10, 1.30, 2.15],
            [1.90, 4.80, 1.40],
            [5.60, 2.20, 5.10],
        ],
        dtype=np.float64,
    )
    charges = np.array([0.7, -0.4, 0.5, -0.8], dtype=np.float64)
    cell = np.array([[[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]], dtype=np.float64)
    k_vectors = _half_space_k_vectors(L, n_max=2)
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "k_vectors": k_vectors[None],  # (1, K, 3)
        "num_atoms": 4,
        "num_k": k_vectors.shape[0],
        "num_systems": 1,
        "L": L,
    }


def _batch_system():
    """Two independent single-systems concatenated (same k-set)."""
    s = _single_system()
    n = s["num_atoms"]
    positions = np.concatenate([s["positions"], s["positions"] + 0.35], axis=0)
    charges = np.concatenate([s["charges"], -s["charges"]], axis=0)
    cell = np.concatenate([s["cell"], s["cell"]], axis=0)
    batch_idx = np.array([0] * n + [1] * n, dtype=np.int32)
    k_vectors = np.concatenate([s["k_vectors"], s["k_vectors"]], axis=0)  # (2, K, 3)
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "k_vectors": k_vectors,
        "batch_idx": batch_idx,
        "atom_start": np.array([0, n], dtype=np.int32),
        "atom_end": np.array([n, 2 * n], dtype=np.int32),
        "num_atoms": 2 * n,
        "num_k": s["num_k"],
        "num_systems": 2,
        "L": s["L"],
    }


# ==============================================================================
# Warp-array preparation
# ==============================================================================


def _alpha_array(num_systems, device, dtype):
    return wp.from_numpy(
        np.full(num_systems, _ALPHA, dtype=_NPF[dtype]), dtype=dtype, device=device
    )


def _to_wp(sysd, device, dtype):
    """Common warp arrays from a system dict (k_vectors kept 2D (S,K))."""
    npf = _NPF[dtype]
    positions = wp.from_numpy(
        sysd["positions"].astype(npf), dtype=_VEC[dtype], device=device
    )
    charges = wp.from_numpy(sysd["charges"].astype(npf), dtype=dtype, device=device)
    cell = wp.from_numpy(sysd["cell"].astype(npf), dtype=_MAT[dtype], device=device)
    kv_2d = wp.from_numpy(
        sysd["k_vectors"].astype(npf), dtype=_VEC[dtype], device=device
    )
    return positions, charges, cell, kv_2d


def _fill_structure_factors(sysd, device, dtype, *, batched):
    """Run the (reused) hand-written fill kernel; return the precomputed arrays."""
    positions, charges, cell, kv_2d = _to_wp(sysd, device, dtype)
    n = sysd["num_atoms"]
    nsys = sysd["num_systems"]
    num_k = sysd["num_k"]
    alpha = _alpha_array(nsys, device, dtype)

    cos_kr = wp.zeros((num_k, n), dtype=wp.float64, device=device)
    sin_kr = wp.zeros((num_k, n), dtype=wp.float64, device=device)

    if batched:
        real_sf = wp.zeros((nsys, num_k), dtype=wp.float64, device=device)
        imag_sf = wp.zeros((nsys, num_k), dtype=wp.float64, device=device)
        total_charges = wp.zeros(nsys, dtype=wp.float64, device=device)
        atom_start = wp.from_numpy(sysd["atom_start"], dtype=wp.int32, device=device)
        atom_end = wp.from_numpy(sysd["atom_end"], dtype=wp.int32, device=device)
        max_atoms = int((sysd["atom_end"] - sysd["atom_start"]).max())
        from nvalchemiops.interactions.electrostatics.ewald_kernels import (
            BATCH_BLOCK_SIZE,
        )

        max_blocks = (max_atoms + BATCH_BLOCK_SIZE - 1) // BATCH_BLOCK_SIZE
        batch_ewald_reciprocal_space_fill_structure_factors(
            positions=positions,
            charges=charges,
            k_vectors=kv_2d,
            cell=cell,
            alpha=alpha,
            atom_start=atom_start,
            atom_end=atom_end,
            total_charges=total_charges,
            cos_k_dot_r=cos_kr,
            sin_k_dot_r=sin_kr,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            num_k=num_k,
            num_systems=nsys,
            max_blocks_per_system=max_blocks,
            wp_dtype=dtype,
            device=device,
        )
    else:
        # Single-system fill kernel takes 1D k_vectors / structure factors.
        kv_1d = wp.from_numpy(
            sysd["k_vectors"][0].astype(_NPF[dtype]), dtype=_VEC[dtype], device=device
        )
        real_sf = wp.zeros(num_k, dtype=wp.float64, device=device)
        imag_sf = wp.zeros(num_k, dtype=wp.float64, device=device)
        total_charge = wp.zeros(1, dtype=wp.float64, device=device)
        ewald_reciprocal_space_fill_structure_factors(
            positions=positions,
            charges=charges,
            k_vectors=kv_1d,
            cell=cell,
            alpha=alpha,
            total_charge=total_charge,
            cos_k_dot_r=cos_kr,
            sin_k_dot_r=sin_kr,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            wp_dtype=dtype,
            device=device,
        )
        # Promote to (1, K) views for the factory compute kernel.
        real_sf = real_sf.reshape((1, num_k))
        imag_sf = imag_sf.reshape((1, num_k))
    wp.synchronize()
    return cos_kr, sin_kr, real_sf, imag_sf


# ==============================================================================
# Factory launch helpers
# ==============================================================================


def _launch_forward(sysd, *, dtype, batched, device, deriv):
    """Run fill (reused) + factory compute; return (energies, forces, cg) numpy."""
    bundle = get_ewald_recip_kernel(
        dtype, batched=batched, deriv_state=deriv, order="forward"
    )
    positions, charges, cell, kv_2d = _to_wp(sysd, device, dtype)
    n = sysd["num_atoms"]
    s = alloc_ewald_recip_sentinels(dtype, device)
    cos_kr, sin_kr, real_sf, imag_sf = _fill_structure_factors(
        sysd, device, dtype, batched=batched
    )
    batch_id = (
        wp.from_numpy(sysd["batch_idx"], dtype=wp.int32, device=device)
        if batched
        else wp.empty((0,), dtype=wp.int32, device=device)
    )
    energies = wp.zeros(n, dtype=wp.float64, device=device)
    forces = wp.zeros(n, dtype=_VEC[dtype], device=device)
    cg = wp.zeros(n, dtype=wp.float64, device=device)
    grad_energy = wp.empty((0,), dtype=wp.float64, device=device)

    wp.launch(
        bundle.compute,
        dim=n,
        inputs=[
            charges,
            batch_id,
            kv_2d,
            cos_kr,
            sin_kr,
            real_sf,
            imag_sf,
            grad_energy,
            energies,
            forces if deriv.value >= _DerivState.E_F.value else s["atomic_forces"],
            cg if deriv.value >= _DerivState.E_F_dQ.value else s["charge_gradients"],
        ],
        device=device,
    )
    wp.synchronize()
    return energies.numpy(), forces.numpy(), cg.numpy()


def _launch_backward(sysd, *, dtype, batched, device, deriv, grad_energy_np):
    """Run fill (reused) + factory backward compute; return (grad_pos, grad_q)."""
    bundle = get_ewald_recip_kernel(
        dtype, batched=batched, deriv_state=deriv, order="backward"
    )
    positions, charges, cell, kv_2d = _to_wp(sysd, device, dtype)
    n = sysd["num_atoms"]
    s = alloc_ewald_recip_sentinels(dtype, device)
    cos_kr, sin_kr, real_sf, imag_sf = _fill_structure_factors(
        sysd, device, dtype, batched=batched
    )
    batch_id = (
        wp.from_numpy(sysd["batch_idx"], dtype=wp.int32, device=device)
        if batched
        else wp.empty((0,), dtype=wp.int32, device=device)
    )
    grad_energy = wp.from_numpy(
        grad_energy_np.astype(np.float64), dtype=wp.float64, device=device
    )
    energies = wp.empty((0,), dtype=wp.float64, device=device)
    grad_pos = wp.zeros(n, dtype=_VEC[dtype], device=device)
    grad_q = wp.zeros(n, dtype=wp.float64, device=device)

    wp.launch(
        bundle.compute,
        dim=n,
        inputs=[
            charges,
            batch_id,
            kv_2d,
            cos_kr,
            sin_kr,
            real_sf,
            imag_sf,
            grad_energy,
            energies,
            grad_pos,
            grad_q
            if deriv.value >= _DerivState.E_F_dQ.value
            else s["charge_gradients"],
        ],
        device=device,
    )
    wp.synchronize()
    return grad_pos.numpy(), grad_q.numpy()


def _launch_backward_virial(sysd, *, dtype, batched, device, grad_energy_np):
    """Run the factory backward virial kernel (ge baked in); return virial numpy."""
    bundle = get_ewald_recip_kernel(
        dtype,
        batched=batched,
        deriv_state=_DerivState.E_F,
        cell_grad=True,
        order="backward",
    )
    nsys = sysd["num_systems"]
    num_k = sysd["num_k"]
    L = sysd["L"]
    cos_kr, sin_kr, real_sf, imag_sf = _fill_structure_factors(
        sysd, device, dtype, batched=batched
    )
    kv_2d = wp.from_numpy(
        sysd["k_vectors"].astype(_NPF[dtype]), dtype=_VEC[dtype], device=device
    )
    alpha = _alpha_array(nsys, device, dtype)
    volume = wp.from_numpy(
        np.full(nsys, L**3, dtype=np.float64), dtype=wp.float64, device=device
    )
    grad_energy = wp.from_numpy(
        grad_energy_np.astype(np.float64), dtype=wp.float64, device=device
    )
    virial = wp.zeros(nsys, dtype=_MAT[dtype], device=device)
    dim = (num_k, nsys) if batched else num_k
    wp.launch(
        bundle.virial,
        dim=dim,
        inputs=[kv_2d, alpha, volume, real_sf, imag_sf, grad_energy, virial],
        device=device,
    )
    wp.synchronize()
    return virial.numpy()


def _launch_backward_kspace(sysd, *, dtype, batched, device, grad_energy_np):
    """Run the factory first-order cell-input kernel; return (grad_kvectors, grad_volume).

    ``grad_kvectors`` (per (system, k) vec3) = ``ge dE/dk``; ``grad_volume`` (per
    system scalar) = ``ge dE/dV``. These are the cell-side first derivatives the recip
    kernel owns (Torch maps them to ``grad_cell``).
    """
    bundle = get_ewald_recip_kernel(
        dtype,
        batched=batched,
        deriv_state=_DerivState.E_F,
        cell_grad=True,
        order="backward",
    )
    positions, charges, cell, kv_2d = _to_wp(sysd, device, dtype)
    nsys = sysd["num_systems"]
    num_k = sysd["num_k"]
    L = sysd["L"]
    alpha = _alpha_array(nsys, device, dtype)
    volume = wp.from_numpy(
        np.full(nsys, L**3, dtype=np.float64), dtype=wp.float64, device=device
    )
    grad_energy = wp.from_numpy(
        grad_energy_np.astype(np.float64), dtype=wp.float64, device=device
    )
    if batched:
        batch_id = wp.from_numpy(sysd["batch_idx"], dtype=wp.int32, device=device)
        atom_start = wp.from_numpy(sysd["atom_start"], dtype=wp.int32, device=device)
        atom_end = wp.from_numpy(sysd["atom_end"], dtype=wp.int32, device=device)
    else:
        batch_id = wp.empty((0,), dtype=wp.int32, device=device)
        atom_start = wp.empty((0,), dtype=wp.int32, device=device)
        atom_end = wp.empty((0,), dtype=wp.int32, device=device)
    grad_kv = wp.zeros((nsys, num_k), dtype=_VEC[dtype], device=device)
    grad_vol = wp.zeros(nsys, dtype=wp.float64, device=device)
    dim = (num_k, nsys) if batched else num_k
    wp.launch(
        bundle.kspace,
        dim=dim,
        inputs=[
            positions,
            charges,
            kv_2d,
            alpha,
            volume,
            batch_id,
            atom_start,
            atom_end,
            grad_energy,
            grad_kv,
            grad_vol,
        ],
        device=device,
    )
    wp.synchronize()
    return grad_kv.numpy(), grad_vol.numpy()


def _launch_batched_cellgrad_cache_kspace(sysd, *, dtype, device, grad_energy_np):
    """Run batched fill-cellgrad + cache consumer; return (grad_kvectors, grad_volume)."""
    positions, charges, cell, kv_2d = _to_wp(sysd, device, dtype)
    n = sysd["num_atoms"]
    nsys = sysd["num_systems"]
    num_k = sysd["num_k"]
    alpha = _alpha_array(nsys, device, dtype)
    atom_start = wp.from_numpy(sysd["atom_start"], dtype=wp.int32, device=device)
    atom_end = wp.from_numpy(sysd["atom_end"], dtype=wp.int32, device=device)
    max_atoms = int((sysd["atom_end"] - sysd["atom_start"]).max())
    max_blocks = max((max_atoms + BATCH_BLOCK_SIZE - 1) // BATCH_BLOCK_SIZE, 1)

    cos_kr = wp.zeros((num_k, n), dtype=wp.float64, device=device)
    sin_kr = wp.zeros((num_k, n), dtype=wp.float64, device=device)
    real_sf = wp.zeros((nsys, num_k), dtype=wp.float64, device=device)
    imag_sf = wp.zeros((nsys, num_k), dtype=wp.float64, device=device)
    total_charges = wp.zeros(nsys, dtype=wp.float64, device=device)
    cellgrad_cache = wp.zeros((nsys * num_k, 8), dtype=wp.float64, device=device)

    wp.launch(
        _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_cellgrad,
        dim=(num_k, nsys, max_blocks),
        inputs=[
            positions,
            charges,
            kv_2d,
            cell,
            alpha,
            atom_start,
            atom_end,
            total_charges,
            cos_kr,
            sin_kr,
            real_sf,
            imag_sf,
            cellgrad_cache,
        ],
        device=device,
    )

    volume = wp.from_numpy(
        np.full(nsys, sysd["L"] ** 3, dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    grad_energy = wp.from_numpy(
        grad_energy_np.astype(np.float64), dtype=wp.float64, device=device
    )
    grad_kv = wp.zeros((nsys, num_k), dtype=_VEC[dtype], device=device)
    grad_vol = wp.zeros(nsys, dtype=wp.float64, device=device)
    consume = _make_backward_kspace_from_cache_kernel(dtype, batched=True)
    wp.launch(
        consume,
        dim=(num_k, nsys),
        inputs=[
            kv_2d,
            alpha,
            volume,
            cellgrad_cache,
            grad_energy,
            wp.int32(num_k),
            grad_kv,
            grad_vol,
        ],
        device=device,
    )
    wp.synchronize()
    return grad_kv.numpy(), grad_vol.numpy()


def _ref_forward_virial(sysd, *, dtype, batched, device):
    """Run the reused (unscaled) forward virial kernel W; return numpy."""
    from nvalchemiops.interactions.electrostatics.ewald_kernels import (
        _batch_ewald_reciprocal_space_virial_kernel,
        _ewald_reciprocal_space_virial_kernel,
    )

    nsys = sysd["num_systems"]
    num_k = sysd["num_k"]
    L = sysd["L"]
    cos_kr, sin_kr, real_sf, imag_sf = _fill_structure_factors(
        sysd, device, dtype, batched=batched
    )
    alpha = _alpha_array(nsys, device, dtype)
    volume = wp.from_numpy(
        np.full(nsys, L**3, dtype=np.float64), dtype=wp.float64, device=device
    )
    virial = wp.zeros(nsys, dtype=_MAT[dtype], device=device)
    if batched:
        kv = wp.from_numpy(
            sysd["k_vectors"].astype(_NPF[dtype]), dtype=_VEC[dtype], device=device
        )
        wp.launch(
            _batch_ewald_reciprocal_space_virial_kernel,
            dim=(num_k, nsys),
            inputs=[kv, alpha, volume, real_sf, imag_sf, virial],
            device=device,
        )
    else:
        kv = wp.from_numpy(
            sysd["k_vectors"][0].astype(_NPF[dtype]), dtype=_VEC[dtype], device=device
        )
        wp.launch(
            _ewald_reciprocal_space_virial_kernel,
            dim=num_k,
            inputs=[
                kv,
                alpha,
                volume,
                real_sf.reshape((num_k,)),
                imag_sf.reshape((num_k,)),
                virial,
            ],
            device=device,
        )
    wp.synchronize()
    return virial.numpy()


def _launch_double_backward(
    sysd,
    *,
    dtype,
    batched,
    device,
    deriv,
    grad_energy_np,
    vpos_np,
    vq_np,
    cell_grad=False,
    vkv_np=None,
    vvol_np=None,
):
    """Run the factory double-backward (reduce + compute); return numpy outputs.

    When ``cell_grad`` is set, the cell-input (k-vector / volume) second-order slots
    are exercised: ``vkv_np`` is the per-(system, k) k-vector cotangent and ``vvol_np``
    the per-system volume cotangent. Returns ``(grad_grad_energy, grad_pos, grad_q,
    grad_kvectors, grad_volume)``; the last two are zeros when ``cell_grad`` is False.
    """
    bundle = get_ewald_recip_kernel(
        dtype,
        batched=batched,
        deriv_state=deriv,
        order="double_backward",
        cell_grad=cell_grad,
    )
    positions, charges, cell, kv_2d = _to_wp(sysd, device, dtype)
    n = sysd["num_atoms"]
    nsys = sysd["num_systems"]
    num_k = sysd["num_k"]
    L = sysd["L"]
    alpha = _alpha_array(nsys, device, dtype)
    s = alloc_ewald_recip_sentinels(dtype, device)
    deriv_dq = wp.int32(1 if deriv.value >= _DerivState.E_F_dQ.value else 0)
    cg = wp.int32(1 if cell_grad else 0)

    if batched:
        batch_id = wp.from_numpy(sysd["batch_idx"], dtype=wp.int32, device=device)
        atom_start = wp.from_numpy(sysd["atom_start"], dtype=wp.int32, device=device)
        atom_end = wp.from_numpy(sysd["atom_end"], dtype=wp.int32, device=device)
    else:
        batch_id = wp.empty((0,), dtype=wp.int32, device=device)
        atom_start = wp.empty((0,), dtype=wp.int32, device=device)
        atom_end = wp.empty((0,), dtype=wp.int32, device=device)

    grad_energy = wp.from_numpy(
        grad_energy_np.astype(np.float64), dtype=wp.float64, device=device
    )
    v_pos = wp.from_numpy(vpos_np.astype(_NPF[dtype]), dtype=_VEC[dtype], device=device)
    v_charge = wp.from_numpy(vq_np.astype(np.float64), dtype=wp.float64, device=device)

    # Cell-input cotangents + volume. Volume from cell (cubic L) so the kernel matches
    # the Torch det(cell) the public path uses.
    volume = wp.from_numpy(
        np.full(nsys, L**3, dtype=np.float64), dtype=wp.float64, device=device
    )
    if cell_grad:
        v_kvectors = wp.from_numpy(
            vkv_np.astype(_NPF[dtype]), dtype=_VEC[dtype], device=device
        )
        v_volume = wp.from_numpy(
            vvol_np.astype(np.float64), dtype=wp.float64, device=device
        )
        gPu = wp.zeros((nsys, num_k), dtype=wp.float64, device=device)
        gQu = wp.zeros((nsys, num_k), dtype=wp.float64, device=device)
        grad_kv = wp.zeros((nsys, num_k), dtype=_VEC[dtype], device=device)
        grad_vol = wp.zeros(nsys, dtype=wp.float64, device=device)
    else:
        v_kvectors = s["v_kvectors"]
        v_volume = s["v_volume"]
        gPu = s["grad_charges"].reshape((0, 0))  # zero-size 2D sentinel
        gQu = gPu
        grad_kv = s["grad_kvectors"]
        grad_vol = s["grad_volume"]

    gA = wp.zeros((nsys, num_k), dtype=wp.float64, device=device)
    gB = wp.zeros((nsys, num_k), dtype=wp.float64, device=device)
    gC = wp.zeros((nsys, num_k), dtype=wp.float64, device=device)
    gD = wp.zeros((nsys, num_k), dtype=wp.float64, device=device)
    gP = wp.zeros((nsys, num_k), dtype=wp.float64, device=device)
    gQ = wp.zeros((nsys, num_k), dtype=wp.float64, device=device)
    gge = wp.zeros(nsys, dtype=wp.float64, device=device)
    grad_pos = wp.zeros(n, dtype=_VEC[dtype], device=device)
    grad_q = wp.zeros(n, dtype=wp.float64, device=device)

    reduce_dim = (num_k, nsys) if batched else num_k
    wp.launch(
        bundle.fill,
        dim=reduce_dim,
        inputs=[
            positions,
            charges,
            kv_2d,
            cell,
            alpha,
            batch_id,
            atom_start,
            atom_end,
            v_pos,
            v_charge,
            grad_energy,
            deriv_dq,
            gA,
            gB,
            gC,
            gD,
            gP,
            gQ,
            gge,
            cg,
            volume,
            v_kvectors,
            v_volume,
            gPu,
            gQu,
            grad_kv,
            grad_vol,
        ],
        device=device,
    )
    wp.launch(
        bundle.compute,
        dim=n,
        inputs=[
            positions,
            charges,
            kv_2d,
            batch_id,
            v_pos,
            v_charge,
            grad_energy,
            deriv_dq,
            gA,
            gB,
            gC,
            gD,
            gP,
            gQ,
            grad_pos,
            grad_q if deriv.value >= _DerivState.E_F_dQ.value else s["grad_charges"],
            cg,
            alpha,
            volume,
            v_kvectors,
            v_volume,
            gPu,
            gQu,
        ],
        device=device,
    )
    wp.synchronize()
    return (
        gge.numpy(),
        grad_pos.numpy(),
        grad_q.numpy(),
        grad_kv.numpy() if cell_grad else np.zeros((nsys, num_k, 3)),
        grad_vol.numpy() if cell_grad else np.zeros(nsys),
    )


# ==============================================================================
# Hand-written reference launchers (k-sum only; reuse the same fill output)
# ==============================================================================


def _ref_energy_forces(sysd, *, dtype, batched, device, charge_grad):
    """Run the hand-written compute launcher; return (energies, forces, cg) numpy."""
    positions, charges, cell, kv_2d = _to_wp(sysd, device, dtype)
    n = sysd["num_atoms"]
    cos_kr, sin_kr, real_sf, imag_sf = _fill_structure_factors(
        sysd, device, dtype, batched=batched
    )
    energies = wp.zeros(n, dtype=wp.float64, device=device)
    forces = wp.zeros(n, dtype=_VEC[dtype], device=device)
    cg = wp.zeros(n, dtype=wp.float64, device=device)

    if batched:
        batch_id = wp.from_numpy(sysd["batch_idx"], dtype=wp.int32, device=device)
        if charge_grad:
            batch_ewald_reciprocal_space_energy_forces_charge_grad(
                charges=charges,
                batch_id=batch_id,
                k_vectors=kv_2d,
                cos_k_dot_r=cos_kr,
                sin_k_dot_r=sin_kr,
                real_structure_factors=real_sf,
                imag_structure_factors=imag_sf,
                reciprocal_energies=energies,
                atomic_forces=forces,
                charge_gradients=cg,
                wp_dtype=dtype,
                device=device,
            )
        else:
            batch_ewald_reciprocal_space_energy_forces(
                charges=charges,
                batch_id=batch_id,
                k_vectors=kv_2d,
                cos_k_dot_r=cos_kr,
                sin_k_dot_r=sin_kr,
                real_structure_factors=real_sf,
                imag_structure_factors=imag_sf,
                reciprocal_energies=energies,
                atomic_forces=forces,
                wp_dtype=dtype,
                device=device,
            )
    else:
        # Single launchers take 1D k_vectors + 1D structure factors.
        kv_1d = wp.from_numpy(
            sysd["k_vectors"][0].astype(_NPF[dtype]), dtype=_VEC[dtype], device=device
        )
        num_k = sysd["num_k"]
        real_1d = real_sf.reshape((num_k,))
        imag_1d = imag_sf.reshape((num_k,))
        if charge_grad:
            ewald_reciprocal_space_energy_forces_charge_grad(
                charges=charges,
                k_vectors=kv_1d,
                cos_k_dot_r=cos_kr,
                sin_k_dot_r=sin_kr,
                real_structure_factors=real_1d,
                imag_structure_factors=imag_1d,
                reciprocal_energies=energies,
                atomic_forces=forces,
                charge_gradients=cg,
                wp_dtype=dtype,
                device=device,
            )
        else:
            ewald_reciprocal_space_energy_forces(
                charges=charges,
                k_vectors=kv_1d,
                cos_k_dot_r=cos_kr,
                sin_k_dot_r=sin_kr,
                real_structure_factors=real_1d,
                imag_structure_factors=imag_1d,
                reciprocal_energies=energies,
                atomic_forces=forces,
                wp_dtype=dtype,
                device=device,
            )
    wp.synchronize()
    return energies.numpy(), forces.numpy(), cg.numpy()


def _ref_compute_energy(sysd, *, dtype, batched, device):
    """Run the hand-written energy-only ``compute_energy`` launcher; return numpy."""
    positions, charges, cell, kv_2d = _to_wp(sysd, device, dtype)
    n = sysd["num_atoms"]
    num_k = sysd["num_k"]
    cos_kr, sin_kr, real_sf, imag_sf = _fill_structure_factors(
        sysd, device, dtype, batched=batched
    )
    energies = wp.zeros(n, dtype=wp.float64, device=device)
    if batched:
        batch_id = wp.from_numpy(sysd["batch_idx"], dtype=wp.int32, device=device)
        batch_ewald_reciprocal_space_compute_energy(
            charges=charges,
            batch_id=batch_id,
            cos_k_dot_r=cos_kr,
            sin_k_dot_r=sin_kr,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            reciprocal_energies=energies,
            wp_dtype=dtype,
            device=device,
        )
    else:
        ewald_reciprocal_space_compute_energy(
            charges=charges,
            cos_k_dot_r=cos_kr,
            sin_k_dot_r=sin_kr,
            real_structure_factors=real_sf.reshape((num_k,)),
            imag_structure_factors=imag_sf.reshape((num_k,)),
            reciprocal_energies=energies,
            wp_dtype=dtype,
            device=device,
        )
    wp.synchronize()
    return energies.numpy()


# ==============================================================================
# 1. Forward parity (bit-exact vs hand-written compute)
# ==============================================================================

_BATCHED = [False, True]
_BATCH_IDS = ["single", "batch"]


def _zero_k_system(*, batched):
    """System fixture with no reciprocal k-vectors."""
    if batched:
        return {
            "positions": np.array(
                [
                    [0.2, 0.3, 0.4],
                    [1.1, 0.5, 0.7],
                    [0.6, 1.5, 0.2],
                    [1.4, 1.2, 1.1],
                ],
                dtype=np.float64,
            ),
            "charges": np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float64),
            "cell": np.stack([np.eye(3) * 8.0, np.eye(3) * 9.0]).astype(np.float64),
            "k_vectors": np.zeros((2, 0, 3), dtype=np.float64),
            "batch_idx": np.array([0, 0, 1, 1], dtype=np.int32),
            "num_atoms": 4,
            "num_k": 0,
            "num_systems": 2,
            "L": 8.0,
        }
    return {
        "positions": np.array(
            [[0.2, 0.3, 0.4], [1.1, 0.5, 0.7]],
            dtype=np.float64,
        ),
        "charges": np.array([0.5, -0.5], dtype=np.float64),
        "cell": np.eye(3, dtype=np.float64)[None] * 8.0,
        "k_vectors": np.zeros((1, 0, 3), dtype=np.float64),
        "num_atoms": 2,
        "num_k": 0,
        "num_systems": 1,
        "L": 8.0,
    }


class TestZeroKGuards:
    @pytest.mark.parametrize("batched", _BATCHED, ids=_BATCH_IDS)
    def test_forward_compute_zero_k_returns_zeros(self, device, batched):
        sysd = _zero_k_system(batched=batched)
        dtype = wp.float64
        bundle = get_ewald_recip_kernel(
            dtype,
            batched=batched,
            deriv_state=_DerivState.E_F_dQ,
            order="forward",
        )
        positions, charges, _cell, kv_2d = _to_wp(sysd, device, dtype)
        n = sysd["num_atoms"]
        nsys = sysd["num_systems"]
        batch_id = (
            wp.from_numpy(sysd["batch_idx"], dtype=wp.int32, device=device)
            if batched
            else wp.empty((0,), dtype=wp.int32, device=device)
        )
        cos_kr = wp.zeros((0, n), dtype=wp.float64, device=device)
        sin_kr = wp.zeros((0, n), dtype=wp.float64, device=device)
        real_sf = wp.zeros((nsys, 0), dtype=wp.float64, device=device)
        imag_sf = wp.zeros((nsys, 0), dtype=wp.float64, device=device)
        grad_energy = wp.empty((0,), dtype=wp.float64, device=device)
        energies = wp.from_numpy(
            np.full(n, 3.0, dtype=np.float64), dtype=wp.float64, device=device
        )
        forces = wp.from_numpy(
            np.full((n, 3), 7.0, dtype=np.float64),
            dtype=_VEC[dtype],
            device=device,
        )
        charge_grads = wp.from_numpy(
            np.full(n, 11.0, dtype=np.float64), dtype=wp.float64, device=device
        )

        wp.launch(
            bundle.compute,
            dim=n,
            inputs=[
                charges,
                batch_id,
                kv_2d,
                cos_kr,
                sin_kr,
                real_sf,
                imag_sf,
                grad_energy,
                energies,
                forces,
                charge_grads,
            ],
            device=device,
        )
        wp.synchronize()

        assert np.array_equal(energies.numpy(), np.zeros(n))
        assert np.array_equal(forces.numpy(), np.zeros((n, 3)))
        assert np.array_equal(charge_grads.numpy(), np.zeros(n))

    @pytest.mark.parametrize("batched", _BATCHED, ids=_BATCH_IDS)
    def test_double_backward_compute_zero_k_returns_zeros(self, device, batched):
        sysd = _zero_k_system(batched=batched)
        dtype = wp.float64
        bundle = get_ewald_recip_kernel(
            dtype,
            batched=batched,
            deriv_state=_DerivState.E_F_dQ,
            order="double_backward",
            cell_grad=True,
        )
        positions, charges, _cell, kv_2d = _to_wp(sysd, device, dtype)
        n = sysd["num_atoms"]
        nsys = sysd["num_systems"]
        batch_id = (
            wp.from_numpy(sysd["batch_idx"], dtype=wp.int32, device=device)
            if batched
            else wp.empty((0,), dtype=wp.int32, device=device)
        )
        v_pos = wp.zeros(n, dtype=_VEC[dtype], device=device)
        v_charge = wp.zeros(n, dtype=wp.float64, device=device)
        grad_energy = wp.from_numpy(
            np.ones(nsys, dtype=np.float64), dtype=wp.float64, device=device
        )
        zero_sk = wp.zeros((nsys, 0), dtype=wp.float64, device=device)
        grad_pos = wp.from_numpy(
            np.full((n, 3), 7.0, dtype=np.float64),
            dtype=_VEC[dtype],
            device=device,
        )
        grad_q = wp.from_numpy(
            np.full(n, 11.0, dtype=np.float64), dtype=wp.float64, device=device
        )
        alpha = _alpha_array(nsys, device, dtype)
        volume = wp.from_numpy(
            np.full(nsys, 8.0**3, dtype=np.float64),
            dtype=wp.float64,
            device=device,
        )
        v_kvectors = wp.zeros((nsys, 0), dtype=_VEC[dtype], device=device)
        v_volume = wp.zeros(nsys, dtype=wp.float64, device=device)

        wp.launch(
            bundle.compute,
            dim=n,
            inputs=[
                positions,
                charges,
                kv_2d,
                batch_id,
                v_pos,
                v_charge,
                grad_energy,
                wp.int32(1),
                zero_sk,
                zero_sk,
                zero_sk,
                zero_sk,
                zero_sk,
                zero_sk,
                grad_pos,
                grad_q,
                wp.int32(1),
                alpha,
                volume,
                v_kvectors,
                v_volume,
                zero_sk,
                zero_sk,
            ],
            device=device,
        )
        wp.synchronize()

        assert np.array_equal(grad_pos.numpy(), np.zeros((n, 3)))
        assert np.array_equal(grad_q.numpy(), np.zeros(n))


class TestForwardParity:
    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batched", _BATCHED, ids=_BATCH_IDS)
    def test_energy_forces(self, device, dtype, batched):
        sysd = _batch_system() if batched else _single_system()
        e_got, f_got, _ = _launch_forward(
            sysd, dtype=dtype, batched=batched, device=device, deriv=_DerivState.E_F
        )
        e_ref = _ref_compute_energy(sysd, dtype=dtype, batched=batched, device=device)
        _, f_ref, _ = _ref_energy_forces(
            sysd, dtype=dtype, batched=batched, device=device, charge_grad=False
        )
        assert np.array_equal(e_got, e_ref)
        assert np.array_equal(f_got, f_ref)

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batched", _BATCHED, ids=_BATCH_IDS)
    def test_charge_grad(self, device, dtype, batched):
        sysd = _batch_system() if batched else _single_system()
        e_got, f_got, cg_got = _launch_forward(
            sysd, dtype=dtype, batched=batched, device=device, deriv=_DerivState.E_F_dQ
        )
        e_ref, f_ref, cg_ref = _ref_energy_forces(
            sysd, dtype=dtype, batched=batched, device=device, charge_grad=True
        )
        assert np.array_equal(e_got, e_ref)
        assert np.array_equal(f_got, f_ref)
        assert np.array_equal(cg_got, cg_ref)

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batched", _BATCHED, ids=_BATCH_IDS)
    def test_energy_only(self, device, dtype, batched):
        # The E branch is bit-exact vs the hand-written ``compute_energy`` oracle
        # (charge applied outside the phase sum), which uses a different rounding
        # order than ``energy_forces`` (charge folded into cos/sin).
        sysd = _batch_system() if batched else _single_system()
        e_got, _, _ = _launch_forward(
            sysd, dtype=dtype, batched=batched, device=device, deriv=_DerivState.E
        )
        e_ref = _ref_compute_energy(sysd, dtype=dtype, batched=batched, device=device)
        assert np.array_equal(e_got, e_ref)

    def test_virial_kernel_reused(self, device):
        # cell_grad selects the hand-written virial kernel (bit-exact by reuse).
        from nvalchemiops.interactions.electrostatics.ewald_kernels import (
            _batch_ewald_reciprocal_space_virial_kernel,
            _ewald_reciprocal_space_virial_kernel,
        )

        single = get_ewald_recip_kernel(
            wp.float64, deriv_state=_DerivState.E_F, cell_grad=True, order="forward"
        )
        assert single.virial.generic_parent is _ewald_reciprocal_space_virial_kernel
        assert single.virial.func is _ewald_reciprocal_space_virial_kernel.func
        batch = get_ewald_recip_kernel(
            wp.float64,
            batched=True,
            deriv_state=_DerivState.E_F,
            cell_grad=True,
            order="forward",
        )
        assert (
            batch.virial.generic_parent is _batch_ewald_reciprocal_space_virial_kernel
        )
        assert batch.virial.func is _batch_ewald_reciprocal_space_virial_kernel.func
        no_v = get_ewald_recip_kernel(
            wp.float64, deriv_state=_DerivState.E_F, cell_grad=False, order="forward"
        )
        assert no_v.virial is None


# ==============================================================================
# 2. Backward == forward first-derivatives scaled by grad_E
# ==============================================================================


class TestBackwardScaling:
    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batched", _BATCHED, ids=_BATCH_IDS)
    def test_backward_unit_cotangent(self, device, dtype, batched):
        # grad_E = 1: backward grad_R == -(forward physical force) (= dE/dR),
        # backward grad_q == forward charge grad.
        sysd = _batch_system() if batched else _single_system()
        nsys = sysd["num_systems"]
        ge = np.ones(nsys, dtype=np.float64)
        gpos, gq = _launch_backward(
            sysd,
            dtype=dtype,
            batched=batched,
            device=device,
            deriv=_DerivState.E_F_dQ,
            grad_energy_np=ge,
        )
        _, f_ref, cg_ref = _ref_energy_forces(
            sysd, dtype=dtype, batched=batched, device=device, charge_grad=True
        )
        np.testing.assert_allclose(gpos, -f_ref, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gq, cg_ref, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    def test_backward_random_cotangent_scales(self, device, dtype):
        sysd = _batch_system()
        nsys = sysd["num_systems"]
        rng = np.random.default_rng(0)
        ge = rng.uniform(0.5, 2.0, size=nsys)
        gpos_r, gq_r = _launch_backward(
            sysd,
            dtype=dtype,
            batched=True,
            device=device,
            deriv=_DerivState.E_F_dQ,
            grad_energy_np=ge,
        )
        gpos_1, gq_1 = _launch_backward(
            sysd,
            dtype=dtype,
            batched=True,
            device=device,
            deriv=_DerivState.E_F_dQ,
            grad_energy_np=np.ones(nsys),
        )
        bidx = sysd["batch_idx"]
        scale_atom = ge[bidx]
        np.testing.assert_allclose(
            gpos_r, gpos_1 * scale_atom[:, None], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(gq_r, gq_1 * scale_atom, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batched", _BATCHED, ids=_BATCH_IDS)
    def test_backward_virial_scales(self, device, dtype, batched):
        # The backward virial bakes grad_energy in (output ge * W), matching the
        # real-space scaling contract.
        # ge=1 must equal the forward W; ge!=1 must scale W per system -- a ge=1-only
        # check cannot distinguish W from ge*W, so test ge != 1 explicitly.
        sysd = _batch_system() if batched else _single_system()
        nsys = sysd["num_systems"]
        W = _ref_forward_virial(sysd, dtype=dtype, batched=batched, device=device)

        # f32 virials are atomic k-sums; reduction order differs between the
        # forward and backward kernels (and across launches on GPU), so compare
        # at f32 precision rather than bit-exact. f64 stays tight.
        is_f32 = dtype == wp.float32
        rtol = 2e-4 if is_f32 else 1e-6
        atol = 1e-5 if is_f32 else 1e-8

        v1 = _launch_backward_virial(
            sysd,
            dtype=dtype,
            batched=batched,
            device=device,
            grad_energy_np=np.ones(nsys),
        )
        np.testing.assert_allclose(v1, W, rtol=rtol, atol=atol)

        ge = (np.arange(nsys) + 2.0).astype(np.float64)  # [2] or [2, 3]
        vge = _launch_backward_virial(
            sysd, dtype=dtype, batched=batched, device=device, grad_energy_np=ge
        )
        expected = W * ge[:, None, None]
        np.testing.assert_allclose(vge, expected, rtol=rtol, atol=atol)


# ==============================================================================
# 3. Backward finite-diff (vs central-diff of factory k-sum-only energy, f64)
# ==============================================================================

_FD_EPS = 1e-6
_FD_TOL = 1e-6


def _factory_energy_total(sysd, positions_np, charges_np, cell_np, *, device, batched):
    sd = dict(sysd)
    sd["positions"] = positions_np
    sd["charges"] = charges_np
    sd["cell"] = cell_np
    e, _, _ = _launch_forward(
        sd, dtype=wp.float64, batched=batched, device=device, deriv=_DerivState.E
    )
    return float(e.sum())


class TestBackwardFiniteDiff:
    def test_forces_fd(self, device):
        sysd = _single_system()
        n = sysd["num_atoms"]
        pos0 = sysd["positions"].copy()
        fd_force = np.zeros((n, 3))
        for i in range(n):
            for d in range(3):
                pp = pos0.copy()
                pp[i, d] += _FD_EPS
                pm = pos0.copy()
                pm[i, d] -= _FD_EPS
                ep = _factory_energy_total(
                    sysd,
                    pp,
                    sysd["charges"],
                    sysd["cell"],
                    device=device,
                    batched=False,
                )
                em = _factory_energy_total(
                    sysd,
                    pm,
                    sysd["charges"],
                    sysd["cell"],
                    device=device,
                    batched=False,
                )
                fd_force[i, d] = -(ep - em) / (2 * _FD_EPS)
        gpos, _ = _launch_backward(
            sysd,
            dtype=wp.float64,
            batched=False,
            device=device,
            deriv=_DerivState.E_F,
            grad_energy_np=np.ones(1),
        )
        # backward grad_R = dE/dR = -force.
        max_abs = np.abs((-gpos) - fd_force).max()
        assert max_abs < _FD_TOL, f"force FD max_abs={max_abs:.3e}"

    def test_charge_grad_fd(self, device):
        sysd = _single_system()
        n = sysd["num_atoms"]
        q0 = sysd["charges"].copy()
        fd_cg = np.zeros(n)
        for i in range(n):
            qp = q0.copy()
            qp[i] += _FD_EPS
            qm = q0.copy()
            qm[i] -= _FD_EPS
            ep = _factory_energy_total(
                sysd, sysd["positions"], qp, sysd["cell"], device=device, batched=False
            )
            em = _factory_energy_total(
                sysd, sysd["positions"], qm, sysd["cell"], device=device, batched=False
            )
            fd_cg[i] = (ep - em) / (2 * _FD_EPS)
        _, gq = _launch_backward(
            sysd,
            dtype=wp.float64,
            batched=False,
            device=device,
            deriv=_DerivState.E_F_dQ,
            grad_energy_np=np.ones(1),
        )
        max_abs = np.abs(gq - fd_cg).max()
        assert max_abs < 1e-8, f"charge-grad FD max_abs={max_abs:.3e}"

    def test_virial_fd(self, device):
        # Strain-first FD virial W = -dE/dstrain (k-sum-only energy, k-vectors
        # regenerated from the deformed cell), vs the reused virial kernel.
        sysd = _single_system()
        L = sysd["L"]
        pos0 = sysd["positions"].copy()
        cell0 = sysd["cell"].copy()
        # Reciprocal lattice for an orthorhombic deformation: k -> deform^{-T} k.
        # Build k from integer indices so the reciprocal set tracks the cell.
        k_factor = 2.0 * np.pi / L
        n_idx = np.round(sysd["k_vectors"][0] / k_factor).astype(np.float64)

        def energy_of_strain(strain):
            deform = np.eye(3) + strain[0]
            pos_s = pos0 @ deform.T
            cell_s = cell0 @ deform.T
            # Reciprocal vectors of the deformed cell: 2*pi * (cell_s^{-1})^T applied
            # to integer indices. cell rows are lattice vectors -> b = 2*pi inv(A)^T.
            A = cell_s[0]
            recip = 2.0 * np.pi * np.linalg.inv(A).T  # rows are b1,b2,b3
            k_s = n_idx @ recip
            sd = dict(sysd)
            sd["k_vectors"] = k_s[None]
            return _factory_energy_total(
                sd, pos_s, sysd["charges"], cell_s, device=device, batched=False
            )

        fd_W = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                sp = np.zeros((1, 3, 3))
                sp[0, a, b] += _FD_EPS
                sm = np.zeros((1, 3, 3))
                sm[0, a, b] -= _FD_EPS
                fd_W[a, b] = -(energy_of_strain(sp) - energy_of_strain(sm)) / (
                    2 * _FD_EPS
                )

        # Reused virial kernel: W = sum_k E(k) (delta - kfac k k).
        positions, charges, cell, kv_2d = _to_wp(sysd, device, wp.float64)
        cos_kr, sin_kr, real_sf, imag_sf = _fill_structure_factors(
            sysd, device, wp.float64, batched=False
        )
        num_k = sysd["num_k"]
        kv_1d = wp.from_numpy(sysd["k_vectors"][0], dtype=wp.vec3d, device=device)
        alpha = _alpha_array(1, device, wp.float64)
        vol = wp.from_numpy(
            np.array([L**3], dtype=np.float64), dtype=wp.float64, device=device
        )
        virial = wp.zeros(1, dtype=wp.mat33d, device=device)
        from nvalchemiops.interactions.electrostatics.ewald_kernels import (
            _ewald_reciprocal_space_virial_kernel,
        )

        wp.launch(
            _ewald_reciprocal_space_virial_kernel,
            dim=num_k,
            inputs=[
                kv_1d,
                alpha,
                vol,
                real_sf.reshape((num_k,)),
                imag_sf.reshape((num_k,)),
                virial,
            ],
            device=device,
        )
        wp.synchronize()
        W = virial.numpy()[0]
        max_abs = np.abs(W - fd_W).max()
        assert max_abs < 1e-5, f"virial FD max_abs={max_abs:.3e}\nW={W}\nfd={fd_W}"


# ==============================================================================
# 4. Double-backward finite-diff (central-diff of backward-kernel outputs)
# ==============================================================================

_FD_TOL_2ND = 1e-6


class TestDoubleBackwardFiniteDiff:
    def test_position_hessian_fd(self, device):
        # v_q = 0, ge = 1. double_backward grad_R == J-HVP, FD of backward grad_R
        # w.r.t. positions contracted with v_pos.
        sysd = _single_system()
        n = sysd["num_atoms"]
        rng = np.random.default_rng(1)
        v_pos = rng.standard_normal((n, 3))
        v_q = np.zeros(n)
        ge = np.ones(1)

        def backward_gradR(positions_np):
            sd = dict(sysd)
            sd["positions"] = positions_np
            gpos, _ = _launch_backward(
                sd,
                dtype=wp.float64,
                batched=False,
                device=device,
                deriv=_DerivState.E_F,
                grad_energy_np=ge,
            )
            return gpos

        pos0 = sysd["positions"].copy()
        fd_gradR = np.zeros((n, 3))
        for i in range(n):
            for d in range(3):
                pp = pos0.copy()
                pp[i, d] += _FD_EPS
                pm = pos0.copy()
                pm[i, d] -= _FD_EPS
                lp = (v_pos * backward_gradR(pp)).sum()
                lm = (v_pos * backward_gradR(pm)).sum()
                fd_gradR[i, d] = (lp - lm) / (2 * _FD_EPS)

        _, dbwd_gradR, _, _, _ = _launch_double_backward(
            sysd,
            dtype=wp.float64,
            batched=False,
            device=device,
            deriv=_DerivState.E_F,
            grad_energy_np=ge,
            vpos_np=v_pos,
            vq_np=v_q,
        )
        max_abs = np.abs(dbwd_gradR - fd_gradR).max()
        assert max_abs < _FD_TOL_2ND, f"position-Hessian FD max_abs={max_abs:.3e}"

    def test_charge_cross_fd(self, device):
        # Nonzero v_pos AND v_q: force<->charge cross + charge self terms.
        sysd = _single_system()
        n = sysd["num_atoms"]
        rng = np.random.default_rng(2)
        v_pos = rng.standard_normal((n, 3))
        v_q = rng.standard_normal(n)
        ge = np.ones(1)

        def backward_grads(positions_np, charges_np):
            sd = dict(sysd)
            sd["positions"] = positions_np
            sd["charges"] = charges_np
            gpos, gq = _launch_backward(
                sd,
                dtype=wp.float64,
                batched=False,
                device=device,
                deriv=_DerivState.E_F_dQ,
                grad_energy_np=ge,
            )
            return gpos, gq

        pos0 = sysd["positions"].copy()
        q0 = sysd["charges"].copy()

        fd_gradR = np.zeros((n, 3))
        for i in range(n):
            for d in range(3):
                pp = pos0.copy()
                pp[i, d] += _FD_EPS
                pm = pos0.copy()
                pm[i, d] -= _FD_EPS
                gpp, gqp = backward_grads(pp, q0)
                gpm, gqm = backward_grads(pm, q0)
                lp = (v_pos * gpp).sum() + (v_q * gqp).sum()
                lm = (v_pos * gpm).sum() + (v_q * gqm).sum()
                fd_gradR[i, d] = (lp - lm) / (2 * _FD_EPS)

        fd_gradQ = np.zeros(n)
        for i in range(n):
            qp = q0.copy()
            qp[i] += _FD_EPS
            qm = q0.copy()
            qm[i] -= _FD_EPS
            gpp, gqp = backward_grads(pos0, qp)
            gpm, gqm = backward_grads(pos0, qm)
            lp = (v_pos * gpp).sum() + (v_q * gqp).sum()
            lm = (v_pos * gpm).sum() + (v_q * gqm).sum()
            fd_gradQ[i] = (lp - lm) / (2 * _FD_EPS)

        _, dbwd_gradR, dbwd_gradQ, _, _ = _launch_double_backward(
            sysd,
            dtype=wp.float64,
            batched=False,
            device=device,
            deriv=_DerivState.E_F_dQ,
            grad_energy_np=ge,
            vpos_np=v_pos,
            vq_np=v_q,
        )
        max_abs_r = np.abs(dbwd_gradR - fd_gradR).max()
        max_abs_q = np.abs(dbwd_gradQ - fd_gradQ).max()
        assert max_abs_r < _FD_TOL_2ND, f"dbwd grad_R FD max_abs={max_abs_r:.3e}"
        assert max_abs_q < _FD_TOL_2ND, f"dbwd grad_q FD max_abs={max_abs_q:.3e}"

    def test_grad_grad_energy_fd(self, device):
        # dL/d(grad_E) == FD of backward output contracted with cotangents w.r.t. ge.
        sysd = _single_system()
        n = sysd["num_atoms"]
        rng = np.random.default_rng(3)
        v_pos = rng.standard_normal((n, 3))
        v_q = rng.standard_normal(n)

        def L_of_ge(ge_val):
            gpos, gq = _launch_backward(
                sysd,
                dtype=wp.float64,
                batched=False,
                device=device,
                deriv=_DerivState.E_F_dQ,
                grad_energy_np=np.array([ge_val]),
            )
            return (v_pos * gpos).sum() + (v_q * gq).sum()

        fd = (L_of_ge(1.0 + _FD_EPS) - L_of_ge(1.0 - _FD_EPS)) / (2 * _FD_EPS)
        gge, _, _, _, _ = _launch_double_backward(
            sysd,
            dtype=wp.float64,
            batched=False,
            device=device,
            deriv=_DerivState.E_F_dQ,
            grad_energy_np=np.ones(1),
            vpos_np=v_pos,
            vq_np=v_q,
        )
        max_abs = abs(gge[0] - fd)
        assert max_abs < _FD_TOL_2ND, f"grad_grad_energy FD max_abs={max_abs:.3e}"

    def test_double_backward_batched_consistency(self, device):
        # Batched double-backward == per-system single double-backward stacked.
        sysd = _batch_system()
        n = sysd["num_atoms"]
        rng = np.random.default_rng(4)
        v_pos = rng.standard_normal((n, 3))
        v_q = rng.standard_normal(n)
        ge = rng.uniform(0.5, 2.0, size=2)

        gge_b, gr_b, gq_b, _, _ = _launch_double_backward(
            sysd,
            dtype=wp.float64,
            batched=True,
            device=device,
            deriv=_DerivState.E_F_dQ,
            grad_energy_np=ge,
            vpos_np=v_pos,
            vq_np=v_q,
        )

        # System 0 and 1 as standalone single systems.
        for sys_id, sl in enumerate([slice(0, 4), slice(4, 8)]):
            single = {
                "positions": sysd["positions"][sl].copy(),
                "charges": sysd["charges"][sl].copy(),
                "cell": sysd["cell"][sys_id : sys_id + 1].copy(),
                "k_vectors": sysd["k_vectors"][sys_id : sys_id + 1].copy(),
                "num_atoms": 4,
                "num_k": sysd["num_k"],
                "num_systems": 1,
                "L": sysd["L"],
            }
            gge_s, gr_s, gq_s, _, _ = _launch_double_backward(
                single,
                dtype=wp.float64,
                batched=False,
                device=device,
                deriv=_DerivState.E_F_dQ,
                grad_energy_np=np.array([ge[sys_id]]),
                vpos_np=v_pos[sl],
                vq_np=v_q[sl],
            )
            np.testing.assert_allclose(gge_b[sys_id], gge_s[0], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(gr_b[sl], gr_s, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(gq_b[sl], gq_s, rtol=1e-9, atol=1e-9)


# ==============================================================================
# 5. F3 harness parity: reciprocal backward kernel vs F3 finite-diff primitives
# ==============================================================================
#
# Drives the SHARED F3 harness (`_deriv_check.fd_forces` / `fd_charge_grad` /
# `fd_strain_virial`) on a `fixed_charge_system` CsCl crystal so the achieved
# tolerances are reported against the F3-certified baselines (not the synthetic
# 4-atom box above). The `energy_fn` closure runs the factory fill+compute on the
# k-sum-only energy and DERIVES the half-space k-vectors from the cell it receives:
# `fd_forces` / `fd_charge_grad` hold the cell fixed (k stays pinned), while
# `fd_strain_virial` deforms the cell (k regenerates from the deformed reciprocal
# lattice) -- matching the k-sum-only virial kernel (background excluded).


def _integer_k_indices(n_max=2):
    """Half-space integer k-indices (nx, ny, nz), excluding 0 and -k duplicates."""
    return np.array(
        [
            [nx, ny, nz]
            for nx in range(-n_max, n_max + 1)
            for ny in range(-n_max, n_max + 1)
            for nz in range(0, n_max + 1)
            if nz > 0 or (nz == 0 and (ny > 0 or (ny == 0 and nx > 0)))
        ],
        dtype=np.float64,
    )


def _recip_system(device):
    """A small CsCl crystal (k-sum-only) ready for the F3 harness."""
    return fixed_charge_system(
        create_cscl_supercell, size=1, jitter=0.2, cutoff=5.0, device="cpu"
    )


def _make_recip_energy_fn(system, k_idx, alpha_val, wp_device):
    """Build an F3 ``energy_fn(p, q, c) -> (N,) energy`` over the factory recip path.

    k-vectors are derived from the cell ``c`` via the reciprocal lattice
    ``b = 2*pi * inv(A)^T`` so they regenerate under strain (virial) but stay pinned
    when the cell is held fixed (forces / charge-grad).
    """

    def energy_fn(p, q, c):
        pos_np = p.detach().cpu().numpy().astype(np.float64)
        q_np = q.detach().cpu().numpy().astype(np.float64)
        cell_np = c.detach().cpu().numpy().astype(np.float64)
        recip = 2.0 * np.pi * np.linalg.inv(cell_np[0]).T  # rows b1, b2, b3
        k_vectors = (k_idx @ recip)[None]  # (1, K, 3)
        sd = {
            "positions": pos_np,
            "charges": q_np,
            "cell": cell_np,
            "k_vectors": k_vectors,
            "num_atoms": pos_np.shape[0],
            "num_k": k_vectors.shape[1],
            "num_systems": 1,
            "L": float(cell_np[0, 0, 0]),
        }
        e, _, _ = _launch_forward(
            sd, dtype=wp.float64, batched=False, device=wp_device, deriv=_DerivState.E
        )
        return torch.as_tensor(e, dtype=torch.float64, device=p.device)

    return energy_fn


class TestF3HarnessBackwardParity:
    """Reciprocal backward / virial kernels vs the shared F3 finite-diff primitives.

    Tolerance note: FD here is compared against the *analytic* backward kernel, so
    the achieved deviation is the central-difference truncation floor (O(h^2)) rather
    than the ~1e-11 the F3 self-test reaches against autograd of the same closure.
    A wrong kernel term would be off by O(1e-2)+, far above these floors.
    """

    def _setup(self, device):
        system = _recip_system(device)
        k_idx = _integer_k_indices(n_max=2)
        alpha_val = float(system.alpha[0].item())
        # The factory uses _ALPHA internally; pin the harness system's alpha to match.
        assert abs(_ALPHA - alpha_val) < 1.0  # both ~0.3-0.4; closure uses _ALPHA
        energy_fn = _make_recip_energy_fn(system, k_idx, _ALPHA, device)
        return system, k_idx, energy_fn

    def _backward_csum(self, system, k_idx, device, *, deriv):
        recip = 2.0 * np.pi * np.linalg.inv(system.cell[0].cpu().numpy()).T
        k_vectors = (k_idx @ recip)[None]
        sd = {
            "positions": system.positions.cpu().numpy().astype(np.float64),
            "charges": system.charges.cpu().numpy().astype(np.float64),
            "cell": system.cell.cpu().numpy().astype(np.float64),
            "k_vectors": k_vectors,
            "num_atoms": system.positions.shape[0],
            "num_k": k_vectors.shape[1],
            "num_systems": 1,
            "L": float(system.cell[0, 0, 0].item()),
        }
        return _launch_backward(
            sd,
            dtype=wp.float64,
            batched=False,
            device=device,
            deriv=deriv,
            grad_energy_np=np.ones(1),
        )

    def test_forces_fd(self, device):
        system, k_idx, energy_fn = self._setup(device)
        fd = fd_forces(energy_fn, system.positions, system.charges, system.cell)
        gpos, _ = self._backward_csum(system, k_idx, device, deriv=_DerivState.E_F)
        force = torch.as_tensor(-gpos, dtype=torch.float64)  # backward grad_R = dE/dR
        max_abs, max_rel = max_abs_rel(force, fd)
        assert max_abs < 1e-6, (
            f"force max_abs={max_abs:.3e} max_rel={max_rel:.3e} device={device}"
        )

    def test_charge_grad_fd(self, device):
        system, k_idx, energy_fn = self._setup(device)
        fd = fd_charge_grad(energy_fn, system.positions, system.charges, system.cell)
        _, gq = self._backward_csum(system, k_idx, device, deriv=_DerivState.E_F_dQ)
        cg = torch.as_tensor(gq, dtype=torch.float64)
        max_abs, max_rel = max_abs_rel(cg, fd)
        assert max_abs < 1e-7, (
            f"charge-grad max_abs={max_abs:.3e} max_rel={max_rel:.3e} device={device}"
        )

    def test_virial_fd(self, device):
        system, k_idx, energy_fn = self._setup(device)
        fd = fd_strain_virial(
            energy_fn,
            system.positions,
            system.charges,
            system.cell,
            batch_idx=None,
        )  # (1, 3, 3)
        # Backward virial state (ge=1) == W; compare to strain-first -dE/dstrain.
        sd = {
            "k_vectors": (
                k_idx @ (2.0 * np.pi * np.linalg.inv(system.cell[0].cpu().numpy()).T)
            )[None],
            "num_atoms": system.positions.shape[0],
            "num_k": k_idx.shape[0],
            "num_systems": 1,
            "L": float(system.cell[0, 0, 0].item()),
            "positions": system.positions.cpu().numpy().astype(np.float64),
            "charges": system.charges.cpu().numpy().astype(np.float64),
            "cell": system.cell.cpu().numpy().astype(np.float64),
        }
        W = _launch_backward_virial(
            sd,
            dtype=wp.float64,
            batched=False,
            device=device,
            grad_energy_np=np.ones(1),
        )
        max_abs, max_rel = max_abs_rel(
            torch.as_tensor(W[0], dtype=torch.float64), fd[0]
        )
        assert max_abs < 1e-5, (
            f"virial max_abs={max_abs:.3e} max_rel={max_rel:.3e} device={device}"
        )


# ==============================================================================
# 6. Cell second-order: grad_kvectors / grad_volume + cross terms
# ==============================================================================
#
# The recip kernel never sees the integer Miller indices, so the cell->k / cell->V
# derivative is structurally Torch's (k_vectors.py / det(cell)); the kernel owns
# second derivatives w.r.t. its differentiable inputs ``k_vectors`` (vec3 per k) and
# ``volume`` (scalar per system) -- mirroring PME's grad_k_squared / grad_volume.
#
# These tests use a self-contained NumPy reciprocal-energy reference, parameterized by
# (positions, charges, k_vectors, volume, alpha), to (a) FD-check the first-order
# grad_kvectors / grad_volume and (b) provide an independent analytic backward whose
# central-difference w.r.t. each cell-side input is the second-order oracle for the
# double-backward kernel. Volume is a free scalar input here so the V-channel is
# exercised cleanly (g_k ~ 1/V).

_EIGHTPI_NP = 8.0 * np.pi


def _np_recip_backward(positions, charges, k_vectors, volume, alpha):
    """Analytic first derivatives of the k-sum energy E = 1/2 sum_k g_k (A^2+B^2).

    g_k = (8 pi / V) e^{-k^2/4a^2}/k^2, A=sum q cos(k.r), B=sum q sin(k.r).
    Returns (E, dE/dr (N,3), dE/dq (N,), dE/dk (K,3), dE/dV scalar). All independent
    of the Warp kernels -- the FD oracle.
    """
    K = k_vectors.shape[0]
    ksq = (k_vectors**2).sum(axis=1)  # (K,)
    keep = ksq > 1e-10
    g = np.zeros(K)
    g[keep] = _EIGHTPI_NP / volume * np.exp(-ksq[keep] / (4.0 * alpha**2)) / ksq[keep]
    kr = k_vectors @ positions.T  # (K, N)
    cos = np.cos(kr)
    sin = np.sin(kr)
    A = (charges[None, :] * cos).sum(axis=1)  # (K,)
    B = (charges[None, :] * sin).sum(axis=1)
    S = A**2 + B**2
    E = 0.5 * (g * S).sum()

    # dE/dr_m = sum_k k g_k q_m (B cos_m - A sin_m)
    coef = g * B  # (K,)
    coef2 = g * A
    # per atom: sum_k k (coef cos_{k,m} - coef2 sin_{k,m}) q_m
    fr = coef[:, None] * cos - coef2[:, None] * sin  # (K, N)
    dEdr = (fr[:, :, None] * k_vectors[:, None, :]).sum(axis=0) * charges[:, None]
    # dE/dq_m = sum_k g_k (A cos_m + B sin_m)
    dEdq = ((g * A)[:, None] * cos + (g * B)[:, None] * sin).sum(axis=0)
    # dE/dk = g_k [ B Ra - A Rb - 1/2 mu S k ],  Ra=sum q cos r, Rb=sum q sin r
    Ra = (charges[None, :, None] * cos[:, :, None] * positions[None, :, :]).sum(axis=1)
    Rb = (charges[None, :, None] * sin[:, :, None] * positions[None, :, :]).sum(axis=1)
    mu = np.zeros(K)
    mu[keep] = 1.0 / (2.0 * alpha**2) + 2.0 / ksq[keep]
    dEdk = g[:, None] * (
        B[:, None] * Ra - A[:, None] * Rb - 0.5 * (mu * S)[:, None] * k_vectors
    )
    # dE/dV = -E/V
    dEdV = -E / volume
    return E, dEdr, dEdq, dEdk, dEdV


def _recip_cfg(seed=0, n=4, L=8.0):
    """Self-contained recip config (positions, charges, k_vectors, volume, alpha)."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, L, size=(n, 3))
    charges = rng.uniform(-1.0, 1.0, size=n)
    charges -= charges.mean()  # neutral
    k = _half_space_k_vectors(L, n_max=2)
    volume = L**3
    return positions, charges, k, volume, _ALPHA


class TestCellSecondOrderFirstDeriv:
    """First-order grad_kvectors / grad_volume vs FD of the NumPy energy."""

    def test_grad_volume_exact(self, device):
        # g_k ~ 1/V exactly -> dE/dV = -E/V. grad_volume kernel (ge=1) must match to
        # machine precision (no FD floor).
        positions, charges, k, volume, alpha = _recip_cfg(seed=1)
        n, K = positions.shape[0], k.shape[0]
        sysd = {
            "positions": positions,
            "charges": charges,
            "cell": (np.eye(3) * volume ** (1 / 3))[None],
            "k_vectors": k[None],
            "num_atoms": n,
            "num_k": K,
            "num_systems": 1,
            "L": volume ** (1 / 3),
        }
        _, grad_vol = _launch_backward_kspace(
            sysd,
            dtype=wp.float64,
            batched=False,
            device=device,
            grad_energy_np=np.ones(1),
        )
        E, _, _, _, dEdV = _np_recip_backward(positions, charges, k, volume, alpha)
        assert abs(grad_vol[0] - dEdV) < 1e-10, (
            f"grad_volume err={abs(grad_vol[0] - dEdV):.3e}"
        )
        # nonzero -> not silently zeroed on the stress path
        assert abs(grad_vol[0]) > 1e-6

    def test_grad_kvectors_fd(self, device):
        positions, charges, k, volume, alpha = _recip_cfg(seed=2)
        n, K = positions.shape[0], k.shape[0]
        sysd = {
            "positions": positions,
            "charges": charges,
            "cell": (np.eye(3) * volume ** (1 / 3))[None],
            "k_vectors": k[None],
            "num_atoms": n,
            "num_k": K,
            "num_systems": 1,
            "L": volume ** (1 / 3),
        }
        grad_kv, _ = _launch_backward_kspace(
            sysd,
            dtype=wp.float64,
            batched=False,
            device=device,
            grad_energy_np=np.ones(1),
        )
        # FD of energy w.r.t. each k_vectors[j, d].
        eps = 1e-7
        fd = np.zeros((K, 3))
        for j in range(K):
            for d in range(3):
                kp = k.copy()
                kp[j, d] += eps
                km = k.copy()
                km[j, d] -= eps
                ep = _np_recip_backward(positions, charges, kp, volume, alpha)[0]
                em = _np_recip_backward(positions, charges, km, volume, alpha)[0]
                fd[j, d] = (ep - em) / (2 * eps)
        max_abs = np.abs(grad_kv[0] - fd).max()
        assert max_abs < 1e-6, f"grad_kvectors FD max_abs={max_abs:.3e}"
        assert np.abs(grad_kv[0]).max() > 1e-6  # nonzero stress path

    def test_batched_first_order(self, device):
        # Batched first-order kspace == per-system single, stacked (per-system ge).
        sysd = _batch_system()
        nsys = sysd["num_systems"]
        ge = (np.arange(nsys) + 1.5).astype(np.float64)
        gkv_b, gvol_b = _launch_backward_kspace(
            sysd, dtype=wp.float64, batched=True, device=device, grad_energy_np=ge
        )
        for sys_id, sl in enumerate([slice(0, 4), slice(4, 8)]):
            single = {
                "positions": sysd["positions"][sl].copy(),
                "charges": sysd["charges"][sl].copy(),
                "cell": sysd["cell"][sys_id : sys_id + 1].copy(),
                "k_vectors": sysd["k_vectors"][sys_id : sys_id + 1].copy(),
                "num_atoms": 4,
                "num_k": sysd["num_k"],
                "num_systems": 1,
                "L": sysd["L"],
            }
            gkv_s, gvol_s = _launch_backward_kspace(
                single,
                dtype=wp.float64,
                batched=False,
                device=device,
                grad_energy_np=np.array([ge[sys_id]]),
            )
            np.testing.assert_allclose(gkv_b[sys_id], gkv_s[0], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(gvol_b[sys_id], gvol_s[0], rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    def test_batched_cellgrad_cache_matches_recompute(self, device, dtype):
        # The forward-fused batched cache must reproduce the existing O(N*K)
        # recompute kspace kernel for non-uniform per-system cotangents.
        sysd = _batch_system()
        ge = np.array([0.75, 1.8], dtype=np.float64)
        gkv_ref, gvol_ref = _launch_backward_kspace(
            sysd, dtype=dtype, batched=True, device=device, grad_energy_np=ge
        )
        gkv_cache, gvol_cache = _launch_batched_cellgrad_cache_kspace(
            sysd, dtype=dtype, device=device, grad_energy_np=ge
        )
        is_f32 = dtype == wp.float32
        rtol = 2e-5 if is_f32 else 1e-9
        atol = 2e-6 if is_f32 else 1e-9
        np.testing.assert_allclose(gkv_cache, gkv_ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(gvol_cache, gvol_ref, rtol=rtol, atol=atol)


class TestCellComposeThroughCell:
    """End-to-end: compose the kernel's grad_kvectors / grad_volume back to grad_cell
    via the SAME map ``k_vectors.py`` uses (``k = m @ 2 pi inv(cell)^T``,
    ``V = det(cell)``), and check against an FD ``dE/dcell`` oracle.

    This is the convention guard the per-input tests are blind to: a transpose / sign /
    ``8 pi`` half-space mismatch between the kernel's ``dE/dk`` and the actual
    ``k(cell)`` map would pass every per-input FD and still produce wrong stress. Torch
    composes ``dk/dcell`` / ``dV/dcell`` (the part the kernel does NOT own); the kernel
    supplies ``dE/dk`` / ``dE/dV`` (the part it does).
    """

    def test_first_order_grad_cell(self, device):
        L = 8.0
        rng = np.random.default_rng(21)
        positions = rng.uniform(0.0, L, size=(4, 3))
        charges = rng.uniform(-1, 1, size=4)
        charges -= charges.mean()
        miller = _integer_k_indices(n_max=2)  # (K, 3) integer indices
        cell0 = np.eye(3) * L

        def k_and_V(cell_np):
            recip = 2.0 * np.pi * np.linalg.inv(cell_np).T  # rows b1, b2, b3
            return miller @ recip, abs(np.linalg.det(cell_np))

        # Kernel grad_kvectors / grad_volume (ge=1) at cell0.
        k0, V0 = k_and_V(cell0)
        sysd = {
            "positions": positions,
            "charges": charges,
            "cell": cell0[None],
            "k_vectors": k0[None],
            "num_atoms": 4,
            "num_k": k0.shape[0],
            "num_systems": 1,
            "L": L,
        }
        grad_kv, grad_vol = _launch_backward_kspace(
            sysd,
            dtype=wp.float64,
            batched=False,
            device=device,
            grad_energy_np=np.ones(1),
        )

        # Torch composes dk/dcell and dV/dcell (the cell->k / cell->V map the kernel
        # does not own); the kernel supplied dE/dk (grad_kv) and dE/dV (grad_vol).
        cell_t = torch.tensor(cell0, dtype=torch.float64, requires_grad=True)
        recip_t = 2.0 * torch.pi * torch.linalg.inv(cell_t).transpose(-1, -2)
        miller_t = torch.tensor(miller, dtype=torch.float64)
        k_t = miller_t @ recip_t  # (K, 3)
        V_t = torch.abs(torch.det(cell_t))
        torch.autograd.backward(
            [k_t, V_t],
            [
                torch.tensor(grad_kv[0], dtype=torch.float64),
                torch.tensor(grad_vol[0], dtype=torch.float64),  # V_t is scalar
            ],
        )
        grad_cell_kernel = cell_t.grad.numpy()

        # FD oracle: dE/dcell with k, V regenerated INSIDE the closure from the cell.
        eps = 1e-6
        fd = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                cp = cell0.copy()
                cp[a, b] += eps
                cm = cell0.copy()
                cm[a, b] -= eps
                kp, Vp = k_and_V(cp)
                km, Vm = k_and_V(cm)
                ep = _np_recip_backward(positions, charges, kp, Vp, _ALPHA)[0]
                em = _np_recip_backward(positions, charges, km, Vm, _ALPHA)[0]
                fd[a, b] = (ep - em) / (2 * eps)

        max_abs = np.abs(grad_cell_kernel - fd).max()
        assert max_abs < 1e-6, (
            f"grad_cell (composed) FD max_abs={max_abs:.3e}\n"
            f"kernel=\n{grad_cell_kernel}\nfd=\n{fd}"
        )
        assert np.abs(grad_cell_kernel).max() > 1e-6  # nonzero stress path

    def test_second_order_grad_cell(self, device):
        # End-to-end strain second derivative: FD the contracted backward (forces +
        # k/V cell grads) w.r.t. cell, composed to grad_cell, vs the double-backward.
        L = 8.0
        rng = np.random.default_rng(22)
        positions = rng.uniform(0.0, L, size=(4, 3))
        charges = rng.uniform(-1, 1, size=4)
        charges -= charges.mean()
        miller = _integer_k_indices(n_max=2)
        cell0 = np.eye(3) * L
        v_pos = rng.standard_normal((4, 3))
        v_q = rng.standard_normal(4)

        def k_and_V(cell_np):
            recip = 2.0 * np.pi * np.linalg.inv(cell_np).T
            return miller @ recip, abs(np.linalg.det(cell_np))

        # Loss L(cell) = v_pos . dE/dR + v_q . dE/dq, with k,V regenerated from cell.
        # Its grad w.r.t. cell is the stress double-backward (cell <- k/V <- backward).
        def loss_of_cell(cell_np):
            k_np, V_np = k_and_V(cell_np)
            _, dEdr, dEdq, _, _ = _np_recip_backward(
                positions, charges, k_np, V_np, _ALPHA
            )
            return (v_pos * dEdr).sum() + (v_q * dEdq).sum()

        eps = 1e-6
        fd_cell = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                cp = cell0.copy()
                cp[a, b] += eps
                cm = cell0.copy()
                cm[a, b] -= eps
                fd_cell[a, b] = (loss_of_cell(cp) - loss_of_cell(cm)) / (2 * eps)

        # Double-backward: with v_pos / v_q set (no direct vk/vV cotangent), the cell
        # second-order reaches grad_cell purely through the k/V cross terms. Feed the
        # k/V cotangents that Torch would pass: vk = dk/dcell contracted... but here we
        # validate via the dbwd grad_kvectors / grad_volume composed through cell.
        k0, V0 = k_and_V(cell0)
        sysd = {
            "positions": positions,
            "charges": charges,
            "cell": cell0[None],
            "k_vectors": k0[None],
            "num_atoms": 4,
            "num_k": k0.shape[0],
            "num_systems": 1,
            "L": L,
            "_dev": device,
        }
        # No direct k/V cotangents: the loss depends on cell only via dE/dR, dE/dq which
        # depend on k, V. So the chain is dL/dcell = (dL/dk).(dk/dcell)+(dL/dV)(dV/dcell)
        # where dL/dk = d/dk (v_pos.dE/dR + v_q.dE/dq) = double_backward grad_kvectors
        # with vk=vV=0. Run dbwd with vk=vV=0 -> grad_kvectors / grad_volume are the
        # cross (k/V <- pos/charge) derivatives.
        gge, gpos, gq, gkv, gvol = _launch_double_backward(
            sysd,
            dtype=wp.float64,
            batched=False,
            device=device,
            deriv=_DerivState.E_F_dQ,
            grad_energy_np=np.ones(1),
            vpos_np=v_pos,
            vq_np=v_q,
            cell_grad=True,
            vkv_np=np.zeros((1, k0.shape[0], 3)),
            vvol_np=np.zeros(1),
        )
        # Compose grad_kvectors / grad_volume (= dL/dk, dL/dV) through cell via Torch.
        cell_t = torch.tensor(cell0, dtype=torch.float64, requires_grad=True)
        recip_t = 2.0 * torch.pi * torch.linalg.inv(cell_t).transpose(-1, -2)
        miller_t = torch.tensor(miller, dtype=torch.float64)
        k_t = miller_t @ recip_t
        V_t = torch.abs(torch.det(cell_t))
        torch.autograd.backward(
            [k_t, V_t],
            [
                torch.tensor(gkv[0], dtype=torch.float64),
                torch.tensor(gvol[0], dtype=torch.float64),  # V_t is scalar
            ],
        )
        grad_cell_kernel = cell_t.grad.numpy()
        max_abs = np.abs(grad_cell_kernel - fd_cell).max()
        assert max_abs < 1e-6, (
            f"stress double-backward (composed) FD max_abs={max_abs:.3e}\n"
            f"kernel=\n{grad_cell_kernel}\nfd=\n{fd_cell}"
        )
        assert np.abs(grad_cell_kernel).max() > 1e-6


class TestCellSecondOrderDoubleBackward:
    """double-backward grad_kvectors / grad_volume + cross terms vs FD of the
    NumPy analytic backward contracted with the cotangents.

    L(k, V, R, q) = v_pos.dE/dR + v_q.dE/dq + sum_k v_kv.dE/dk + v_V dE/dV.
    The double-backward outputs are grad_X = ge dL/dX; here ge=1. FD each input
    direction (k-only, V-only, then pos/charge cross) so a wrong channel is isolated.
    """

    def _setup(self, seed):
        positions, charges, k, volume, alpha = _recip_cfg(seed=seed)
        n, K = positions.shape[0], k.shape[0]
        rng = np.random.default_rng(seed + 100)
        v_pos = rng.standard_normal((n, 3))
        v_q = rng.standard_normal(n)
        v_kv = rng.standard_normal((K, 3))
        v_vol = rng.standard_normal(1)
        sysd = {
            "positions": positions,
            "charges": charges,
            "cell": (np.eye(3) * volume ** (1 / 3))[None],
            "k_vectors": k[None],
            "num_atoms": n,
            "num_k": K,
            "num_systems": 1,
            "L": volume ** (1 / 3),
        }
        return positions, charges, k, volume, alpha, v_pos, v_q, v_kv, v_vol, sysd

    @staticmethod
    def _L(positions, charges, k, volume, alpha, v_pos, v_q, v_kv, v_vol):
        _, dEdr, dEdq, dEdk, dEdV = _np_recip_backward(
            positions, charges, k, volume, alpha
        )
        return (
            (v_pos * dEdr).sum()
            + (v_q * dEdq).sum()
            + (v_kv * dEdk).sum()
            + v_vol[0] * dEdV
        )

    def _run_dbwd(self, sysd, v_pos, v_q, v_kv, v_vol):
        gge, gpos, gq, gkv, gvol = _launch_double_backward(
            sysd,
            dtype=wp.float64,
            batched=False,
            device=sysd["_dev"],
            deriv=_DerivState.E_F_dQ,
            grad_energy_np=np.ones(1),
            vpos_np=v_pos,
            vq_np=v_q,
            cell_grad=True,
            vkv_np=v_kv[None],
            vvol_np=v_vol,
        )
        return gge, gpos, gq, gkv[0], gvol

    def test_grad_kvectors_second_order(self, device):
        # FD of L w.r.t. k_vectors -> dbwd grad_kvectors.
        p, q, k, V, a, vp, vq, vkv, vV, sysd = self._setup(seed=3)
        sysd["_dev"] = device
        _, _, _, gkv, _ = self._run_dbwd(sysd, vp, vq, vkv, vV)
        eps = 1e-6
        K = k.shape[0]
        fd = np.zeros((K, 3))
        for j in range(K):
            for d in range(3):
                kp = k.copy()
                kp[j, d] += eps
                km = k.copy()
                km[j, d] -= eps
                lp = self._L(p, q, kp, V, a, vp, vq, vkv, vV)
                lm = self._L(p, q, km, V, a, vp, vq, vkv, vV)
                fd[j, d] = (lp - lm) / (2 * eps)
        max_abs = np.abs(gkv - fd).max()
        assert max_abs < 1e-6, f"dbwd grad_kvectors FD max_abs={max_abs:.3e}"
        assert np.abs(gkv).max() > 1e-6

    def test_grad_volume_second_order(self, device):
        # FD of L w.r.t. volume -> dbwd grad_volume.
        p, q, k, V, a, vp, vq, vkv, vV, sysd = self._setup(seed=4)
        sysd["_dev"] = device
        _, _, _, _, gvol = self._run_dbwd(sysd, vp, vq, vkv, vV)
        eps = V * 1e-7
        lp = self._L(p, q, k, V + eps, a, vp, vq, vkv, vV)
        lm = self._L(p, q, k, V - eps, a, vp, vq, vkv, vV)
        fd = (lp - lm) / (2 * eps)
        assert abs(gvol[0] - fd) < 1e-7, (
            f"dbwd grad_volume FD err={abs(gvol[0] - fd):.3e}"
        )
        assert abs(gvol[0]) > 1e-8

    def test_cross_terms_into_positions_charges(self, device):
        # FD of L w.r.t. positions / charges -> dbwd grad_positions / grad_charges
        # (these now carry the k/V cross terms in addition to the pos/charge terms).
        p, q, k, V, a, vp, vq, vkv, vV, sysd = self._setup(seed=5)
        sysd["_dev"] = device
        _, gpos, gq, _, _ = self._run_dbwd(sysd, vp, vq, vkv, vV)
        n = p.shape[0]
        eps = 1e-6
        fd_pos = np.zeros((n, 3))
        for i in range(n):
            for d in range(3):
                pp = p.copy()
                pp[i, d] += eps
                pm = p.copy()
                pm[i, d] -= eps
                lp = self._L(pp, q, k, V, a, vp, vq, vkv, vV)
                lm = self._L(pm, q, k, V, a, vp, vq, vkv, vV)
                fd_pos[i, d] = (lp - lm) / (2 * eps)
        fd_q = np.zeros(n)
        for i in range(n):
            qp = q.copy()
            qp[i] += eps
            qm = q.copy()
            qm[i] -= eps
            lp = self._L(p, qp, k, V, a, vp, vq, vkv, vV)
            lm = self._L(p, qm, k, V, a, vp, vq, vkv, vV)
            fd_q[i] = (lp - lm) / (2 * eps)
        max_pos = np.abs(gpos - fd_pos).max()
        max_q = np.abs(gq - fd_q).max()
        assert max_pos < 1e-6, f"dbwd grad_positions (cross) FD max_abs={max_pos:.3e}"
        assert max_q < 1e-6, f"dbwd grad_charges (cross) FD max_abs={max_q:.3e}"

    def test_grad_grad_energy_with_cell_cotangents(self, device):
        # grad_grad_energy = Phi must include the k/V cotangent contributions
        # (Phi = ... + sum_k vk.dE/dk + vV dE/dV).
        p, q, k, V, a, vp, vq, vkv, vV, sysd = self._setup(seed=6)
        sysd["_dev"] = device
        gge, _, _, _, _ = self._run_dbwd(sysd, vp, vq, vkv, vV)
        phi = self._L(p, q, k, V, a, vp, vq, vkv, vV)  # = Phi (ge=1, dL/dge)
        assert abs(gge[0] - phi) < 1e-9, f"grad_grad_energy err={abs(gge[0] - phi):.3e}"

    def test_f32_looser(self, device):
        # f32 second-order: looser tolerance vs the f64 double-backward reference,
        # exercising actual f32 correctness (not just nonzero).
        p, q, k, V, a, vp, vq, vkv, vV, sysd = self._setup(seed=7)
        sysd["_dev"] = device
        gge32, gpos32, gq32, gkv32, gvol32 = _launch_double_backward(
            sysd,
            dtype=wp.float32,
            batched=False,
            device=device,
            deriv=_DerivState.E_F_dQ,
            grad_energy_np=np.ones(1),
            vpos_np=vp,
            vq_np=vq,
            cell_grad=True,
            vkv_np=vkv[None],
            vvol_np=vV,
        )
        _, gpos64, gq64, gkv64, gvol64 = self._run_dbwd(sysd, vp, vq, vkv, vV)
        phi = self._L(p, q, k, V, a, vp, vq, vkv, vV)
        # Phi (grad_grad_energy) is f64-accumulated even in the f32 kernel.
        assert abs(gge32[0] - phi) < 1e-5
        # f32 outputs vs the f64 reference -- loose tolerance scaled to magnitude.
        f32_tol = 5e-4
        np.testing.assert_allclose(
            gkv32[0], gkv64, rtol=f32_tol, atol=f32_tol * (np.abs(gkv64).max() + 1e-30)
        )
        np.testing.assert_allclose(
            gpos32, gpos64, rtol=f32_tol, atol=f32_tol * (np.abs(gpos64).max() + 1e-30)
        )
        np.testing.assert_allclose(gvol32, gvol64, rtol=f32_tol, atol=1e-7)

    def test_batched_cell_second_order(self, device):
        # Batched dbwd with cell cotangents == per-system single, stacked.
        sysd = _batch_system()
        n, K = sysd["num_atoms"], sysd["num_k"]
        rng = np.random.default_rng(8)
        v_pos = rng.standard_normal((n, 3))
        v_q = rng.standard_normal(n)
        v_kv = rng.standard_normal((sysd["num_systems"], K, 3))
        v_vol = rng.uniform(-1, 1, size=sysd["num_systems"])
        ge = rng.uniform(0.5, 2.0, size=sysd["num_systems"])
        gge_b, gr_b, gq_b, gkv_b, gvol_b = _launch_double_backward(
            sysd,
            dtype=wp.float64,
            batched=True,
            device=device,
            deriv=_DerivState.E_F_dQ,
            grad_energy_np=ge,
            vpos_np=v_pos,
            vq_np=v_q,
            cell_grad=True,
            vkv_np=v_kv,
            vvol_np=v_vol,
        )
        for sys_id, sl in enumerate([slice(0, 4), slice(4, 8)]):
            single = {
                "positions": sysd["positions"][sl].copy(),
                "charges": sysd["charges"][sl].copy(),
                "cell": sysd["cell"][sys_id : sys_id + 1].copy(),
                "k_vectors": sysd["k_vectors"][sys_id : sys_id + 1].copy(),
                "num_atoms": 4,
                "num_k": K,
                "num_systems": 1,
                "L": sysd["L"],
            }
            gge_s, gr_s, gq_s, gkv_s, gvol_s = _launch_double_backward(
                single,
                dtype=wp.float64,
                batched=False,
                device=device,
                deriv=_DerivState.E_F_dQ,
                grad_energy_np=np.array([ge[sys_id]]),
                vpos_np=v_pos[sl],
                vq_np=v_q[sl],
                cell_grad=True,
                vkv_np=v_kv[sys_id : sys_id + 1],
                vvol_np=v_vol[sys_id : sys_id + 1],
            )
            np.testing.assert_allclose(gge_b[sys_id], gge_s[0], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(gr_b[sl], gr_s, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(gq_b[sl], gq_s, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(gkv_b[sys_id], gkv_s[0], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(gvol_b[sys_id], gvol_s[0], rtol=1e-9, atol=1e-9)

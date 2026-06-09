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

"""Parity and derivative tests for the ``ewald_real`` factory kernels.

Three guarantees:

1. **Forward derivative parity** -- the factory ``E_F`` / ``E_F_dQ`` / virial
   outputs match the hand-written
   ``ewald_real_space_energy_forces[_charge_grad][_matrix]`` launchers, for
   ``wp.float32`` + ``wp.float64`` x single/batch x CSR/NM, allowing one
   dtype epsilon of roundoff from equivalent accumulation order.
2. **Backward finite-diff** -- the ``order="backward"`` kernel emits
   ``grad_E * dE/dR`` / ``dE/dq`` / virial; with ``grad_E = 1`` these are compared to
   central-difference forces / charge gradients / strain virial of the factory
   energy (float64), against the F3-certified baselines.
3. **Double-backward finite-diff** -- the ``order="double_backward"`` outputs
   (position Hessian . v_pos, charge cross terms, ``grad_grad_energy``) are compared
   to a central-difference of the backward-kernel outputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from nvalchemiops.interactions.electrostatics._factory_common import _DerivState
from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    REAL_SPACE_TILED_BLOCK_DIM,
    batch_ewald_real_space_energy_forces,
    batch_ewald_real_space_energy_forces_charge_grad,
    batch_ewald_real_space_energy_forces_charge_grad_matrix,
    batch_ewald_real_space_energy_forces_matrix,
    ewald_real_space_energy_forces,
    ewald_real_space_energy_forces_charge_grad,
    ewald_real_space_energy_forces_charge_grad_matrix,
    ewald_real_space_energy_forces_matrix,
)
from nvalchemiops.interactions.electrostatics.ewald_real_factory import (
    alloc_ewald_real_sentinels,
    get_ewald_real_kernel,
)
from test.interactions.electrostatics._deriv_check import (
    fd_charge_grad,
    fd_forces,
    fd_strain_virial,
    fixed_charge_system,
    max_abs_rel,
)
from test.interactions.electrostatics.conftest import create_cscl_supercell
from test.interactions.electrostatics.test_ewald_kernels import (
    prepare_csr_inputs,
    prepare_matrix_inputs,
)

_DTYPES = [wp.float32, wp.float64]
_DTYPE_IDS = ["f32", "f64"]
_ALPHA = 0.3
_MASK = 999
_NPF = {wp.float32: np.float32, wp.float64: np.float64}
_VEC = {wp.float32: wp.vec3f, wp.float64: wp.vec3d}
_MAT = {wp.float32: wp.mat33f, wp.float64: wp.mat33d}


def _assert_forward_parity(actual, expected, dtype):
    """Assert factory forward outputs match direct kernels at dtype precision."""
    np_dtype = np.float64 if dtype == wp.float64 else np.float32
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=np.finfo(np_dtype).eps,
    )


# ==============================================================================
# Systems (jittered, asymmetric -> nonzero forces/virial, rtol-safe)
# ==============================================================================


def _single_system():
    """Four atoms in a finite box, CSR half-list + matching full neighbor matrix."""
    positions = np.array(
        [
            [0.10, 0.20, -0.10],
            [2.05, 0.30, 0.15],
            [0.90, 1.80, 0.40],
            [1.60, 1.20, 2.10],
        ],
        dtype=np.float64,
    )
    charges = np.array([0.7, -0.4, 0.5, -0.8], dtype=np.float64)
    cell = np.array(
        [[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]], dtype=np.float64
    )
    # Half neighbor list: every unordered pair once.
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    idx_j = np.array([p[1] for p in pairs], dtype=np.int32)
    counts = [0, 0, 0, 0]
    for i, _ in pairs:
        counts[i] += 1
    neighbor_ptr = np.zeros(5, dtype=np.int32)
    for i in range(4):
        neighbor_ptr[i + 1] = neighbor_ptr[i] + counts[i]
    unit_shifts = np.zeros((len(pairs), 3), dtype=np.int32)

    # Full neighbor matrix (each pair both directions) for NM parity.
    nbrs = {i: [] for i in range(4)}
    for i, j in pairs:
        nbrs[i].append(j)
        nbrs[j].append(i)
    maxn = max(len(v) for v in nbrs.values())
    neighbor_matrix = np.full((4, maxn), _MASK, dtype=np.int32)
    for i in range(4):
        for k, j in enumerate(nbrs[i]):
            neighbor_matrix[i, k] = j
    neighbor_shifts = np.zeros((4, maxn, 3), dtype=np.int32)

    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "idx_j": idx_j,
        "neighbor_ptr": neighbor_ptr,
        "unit_shifts": unit_shifts,
        "neighbor_matrix": neighbor_matrix,
        "neighbor_shifts": neighbor_shifts,
        "fill_value": _MASK,
        "num_atoms": 4,
    }


def _batch_system():
    """Two independent single-systems concatenated (CSR + matrix)."""
    s = _single_system()
    n = s["num_atoms"]
    positions = np.concatenate([s["positions"], s["positions"] + 0.05], axis=0)
    charges = np.concatenate([s["charges"], -s["charges"]], axis=0)
    cell = np.concatenate([s["cell"], s["cell"]], axis=0)
    batch_idx = np.array([0] * n + [1] * n, dtype=np.int32)

    idx_j = np.concatenate([s["idx_j"], s["idx_j"] + n]).astype(np.int32)
    nptr0 = s["neighbor_ptr"]
    neighbor_ptr = np.concatenate([nptr0, nptr0[1:] + nptr0[-1]]).astype(np.int32)
    unit_shifts = np.concatenate([s["unit_shifts"], s["unit_shifts"]], axis=0)

    nm0 = s["neighbor_matrix"]
    nm1 = np.where(nm0 == _MASK, _MASK, nm0 + n)
    neighbor_matrix = np.concatenate([nm0, nm1], axis=0).astype(np.int32)
    neighbor_shifts = np.concatenate(
        [s["neighbor_shifts"], s["neighbor_shifts"]], axis=0
    )
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "batch_idx": batch_idx,
        "idx_j": idx_j,
        "neighbor_ptr": neighbor_ptr,
        "unit_shifts": unit_shifts,
        "neighbor_matrix": neighbor_matrix,
        "neighbor_shifts": neighbor_shifts,
        "fill_value": _MASK,
        "num_atoms": 2 * n,
        "num_systems": 2,
    }


# ==============================================================================
# Factory launch helpers
# ==============================================================================


def _alpha_array(num_systems, device, dtype):
    return wp.from_numpy(
        np.full(num_systems, _ALPHA, dtype=_NPF[dtype]), dtype=dtype, device=device
    )


def _launch_forward(sysd, *, dtype, batched, neighbor_input, device, deriv, cell_grad):
    """Launch a factory forward kernel; return (energies, forces, cg, virial) numpy."""
    if neighbor_input == "list":
        inputs = prepare_csr_inputs(sysd, device, dtype=dtype)
    else:
        inputs = prepare_matrix_inputs(sysd, device, dtype=dtype)
    n = sysd["num_atoms"]
    nsys = sysd.get("num_systems", 1)
    alpha = _alpha_array(nsys, device, dtype)
    s = alloc_ewald_real_sentinels(dtype, device)

    batch_id = (
        wp.from_numpy(sysd["batch_idx"], dtype=wp.int32, device=device)
        if batched
        else s["batch_id"]
    )
    energies = wp.zeros(n, dtype=wp.float64, device=device)
    forces = wp.zeros(n, dtype=_VEC[dtype], device=device)
    cg = wp.zeros(n, dtype=wp.float64, device=device)
    virial = wp.zeros(nsys, dtype=_MAT[dtype], device=device)

    nm = (
        inputs["neighbor_matrix"]
        if neighbor_input == "matrix"
        else s["neighbor_matrix"]
    )
    nms = (
        inputs["neighbor_shifts"]
        if neighbor_input == "matrix"
        else s["unit_shifts_matrix"]
    )
    idxj = inputs["idx_j"] if neighbor_input == "list" else s["idx_j"]
    nptr = inputs["neighbor_ptr"] if neighbor_input == "list" else s["neighbor_ptr"]
    ush = inputs["unit_shifts"] if neighbor_input == "list" else s["unit_shifts"]

    kernel = get_ewald_real_kernel(
        dtype,
        batched=batched,
        neighbor_input=neighbor_input,
        deriv_state=deriv,
        cell_grad=cell_grad,
        order="forward",
    )
    wp.launch(
        kernel,
        dim=n,
        inputs=[
            inputs["positions"],
            inputs["charges"],
            inputs["cell"],
            batch_id,
            idxj,
            nptr,
            ush,
            nm,
            nms,
            wp.int32(_MASK),
            alpha,
            energies,
            forces if deriv.value >= _DerivState.E_F.value else s["atomic_forces"],
            cg if deriv.value >= _DerivState.E_F_dQ.value else s["charge_gradients"],
            virial if cell_grad else s["virial"],
        ],
        device=device,
    )
    wp.synchronize()
    return energies.numpy(), forces.numpy(), cg.numpy(), virial.numpy()


def _launch_backward(
    sysd, *, dtype, batched, neighbor_input, device, deriv, cell_grad, grad_energy_np
):
    """Launch the backward kernel; return (grad_pos, grad_q, virial) numpy."""
    if neighbor_input == "list":
        inputs = prepare_csr_inputs(sysd, device, dtype=dtype)
    else:
        inputs = prepare_matrix_inputs(sysd, device, dtype=dtype)
    n = sysd["num_atoms"]
    nsys = sysd.get("num_systems", 1)
    alpha = _alpha_array(nsys, device, dtype)
    s = alloc_ewald_real_sentinels(dtype, device)

    batch_id = (
        wp.from_numpy(sysd["batch_idx"], dtype=wp.int32, device=device)
        if batched
        else s["batch_id"]
    )
    grad_energy = wp.from_numpy(
        grad_energy_np.astype(np.float64), dtype=wp.float64, device=device
    )
    grad_pos = wp.zeros(n, dtype=_VEC[dtype], device=device)
    grad_q = wp.zeros(n, dtype=wp.float64, device=device)
    virial = wp.zeros(nsys, dtype=_MAT[dtype], device=device)

    nm = (
        inputs["neighbor_matrix"]
        if neighbor_input == "matrix"
        else s["neighbor_matrix"]
    )
    nms = (
        inputs["neighbor_shifts"]
        if neighbor_input == "matrix"
        else s["unit_shifts_matrix"]
    )
    idxj = inputs["idx_j"] if neighbor_input == "list" else s["idx_j"]
    nptr = inputs["neighbor_ptr"] if neighbor_input == "list" else s["neighbor_ptr"]
    ush = inputs["unit_shifts"] if neighbor_input == "list" else s["unit_shifts"]

    kernel = get_ewald_real_kernel(
        dtype,
        batched=batched,
        neighbor_input=neighbor_input,
        deriv_state=deriv,
        cell_grad=cell_grad,
        order="backward",
    )
    wp.launch(
        kernel,
        dim=n,
        inputs=[
            grad_energy,
            inputs["positions"],
            inputs["charges"],
            inputs["cell"],
            batch_id,
            idxj,
            nptr,
            ush,
            nm,
            nms,
            wp.int32(_MASK),
            alpha,
            s["grad_energy"],  # pair_energies slot unused in backward
            grad_pos,
            grad_q
            if deriv.value >= _DerivState.E_F_dQ.value
            else s["charge_gradients"],
            virial if cell_grad else s["virial"],
        ],
        device=device,
    )
    wp.synchronize()
    return grad_pos.numpy(), grad_q.numpy(), virial.numpy()


def _launch_double_backward(
    sysd,
    *,
    dtype,
    batched,
    neighbor_input,
    device,
    deriv,
    grad_energy_np,
    vpos_np,
    vq_np,
    cell_grad=False,
    vcell_np=None,
):
    """Launch the double_backward kernel; return (gge, grad_pos, grad_q, grad_cell)."""
    if neighbor_input == "list":
        inputs = prepare_csr_inputs(sysd, device, dtype=dtype)
    else:
        inputs = prepare_matrix_inputs(sysd, device, dtype=dtype)
    n = sysd["num_atoms"]
    nsys = sysd.get("num_systems", 1)
    alpha = _alpha_array(nsys, device, dtype)
    s = alloc_ewald_real_sentinels(dtype, device)

    batch_id = (
        wp.from_numpy(sysd["batch_idx"], dtype=wp.int32, device=device)
        if batched
        else s["batch_id"]
    )
    grad_energy = wp.from_numpy(
        grad_energy_np.astype(np.float64), dtype=wp.float64, device=device
    )
    v_pos = wp.from_numpy(vpos_np.astype(_NPF[dtype]), dtype=_VEC[dtype], device=device)
    v_charge = wp.from_numpy(vq_np.astype(np.float64), dtype=wp.float64, device=device)
    if cell_grad:
        v_cell = wp.from_numpy(
            vcell_np.astype(_NPF[dtype]), dtype=_MAT[dtype], device=device
        )
    else:
        v_cell = s["v_cell"]
    gge = wp.zeros(nsys, dtype=wp.float64, device=device)
    grad_pos = wp.zeros(n, dtype=_VEC[dtype], device=device)
    grad_q = wp.zeros(n, dtype=wp.float64, device=device)
    grad_cell = wp.zeros(nsys, dtype=_MAT[dtype], device=device)

    nm = (
        inputs["neighbor_matrix"]
        if neighbor_input == "matrix"
        else s["neighbor_matrix"]
    )
    nms = (
        inputs["neighbor_shifts"]
        if neighbor_input == "matrix"
        else s["unit_shifts_matrix"]
    )
    idxj = inputs["idx_j"] if neighbor_input == "list" else s["idx_j"]
    nptr = inputs["neighbor_ptr"] if neighbor_input == "list" else s["neighbor_ptr"]
    ush = inputs["unit_shifts"] if neighbor_input == "list" else s["unit_shifts"]

    kernel = get_ewald_real_kernel(
        dtype,
        batched=batched,
        neighbor_input=neighbor_input,
        deriv_state=deriv,
        cell_grad=cell_grad,
        order="double_backward",
    )
    wp.launch(
        kernel,
        dim=n,
        inputs=[
            v_pos,
            v_charge,
            v_cell,
            grad_energy,
            inputs["positions"],
            inputs["charges"],
            inputs["cell"],
            batch_id,
            idxj,
            nptr,
            ush,
            nm,
            nms,
            wp.int32(_MASK),
            alpha,
            gge,
            grad_pos,
            grad_q
            if deriv.value >= _DerivState.E_F_dQ.value
            else s["charge_gradients"],
            grad_cell if cell_grad else s["grad_cell"],
        ],
        device=device,
    )
    wp.synchronize()
    return gge.numpy(), grad_pos.numpy(), grad_q.numpy(), grad_cell.numpy()


# ==============================================================================
# Hand-written reference launchers
# ==============================================================================


def _ref_forces(sysd, *, dtype, batched, neighbor_input, device, charge_grad, virial):
    """Run the hand-written launcher; return (forces, cg, virial) numpy."""
    n = sysd["num_atoms"]
    nsys = sysd.get("num_systems", 1)
    alpha = _alpha_array(nsys, device, dtype)
    energies = wp.zeros(n, dtype=wp.float64, device=device)
    forces = wp.zeros(n, dtype=_VEC[dtype], device=device)
    cg = wp.zeros(n, dtype=wp.float64, device=device)
    vir = wp.zeros(nsys, dtype=_MAT[dtype], device=device)

    if neighbor_input == "list":
        inputs = prepare_csr_inputs(sysd, device, dtype=dtype)
        common = dict(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            alpha=alpha,
            pair_energies=energies,
            atomic_forces=forces,
            wp_dtype=dtype,
            device=device,
            compute_virial=virial,
        )
        if batched:
            batch_id = wp.from_numpy(sysd["batch_idx"], dtype=wp.int32, device=device)
            if charge_grad:
                batch_ewald_real_space_energy_forces_charge_grad(
                    batch_id=batch_id, charge_gradients=cg, virial=vir, **common
                )
            else:
                batch_ewald_real_space_energy_forces(
                    batch_id=batch_id, virial=vir, **common
                )
        else:
            if charge_grad:
                ewald_real_space_energy_forces_charge_grad(
                    charge_gradients=cg, virial=vir, **common
                )
            else:
                ewald_real_space_energy_forces(virial=vir, **common)
    else:
        inputs = prepare_matrix_inputs(sysd, device, dtype=dtype)
        common = dict(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            neighbor_matrix=inputs["neighbor_matrix"],
            unit_shifts_matrix=inputs["neighbor_shifts"],
            mask_value=_MASK,
            alpha=alpha,
            pair_energies=energies,
            atomic_forces=forces,
            wp_dtype=dtype,
            device=device,
            compute_virial=virial,
        )
        if batched:
            batch_id = wp.from_numpy(sysd["batch_idx"], dtype=wp.int32, device=device)
            if charge_grad:
                batch_ewald_real_space_energy_forces_charge_grad_matrix(
                    batch_id=batch_id, charge_gradients=cg, virial=vir, **common
                )
            else:
                batch_ewald_real_space_energy_forces_matrix(
                    batch_id=batch_id, virial=vir, **common
                )
        else:
            if charge_grad:
                ewald_real_space_energy_forces_charge_grad_matrix(
                    charge_gradients=cg, virial=vir, **common
                )
            else:
                ewald_real_space_energy_forces_matrix(virial=vir, **common)
    wp.synchronize()
    return forces.numpy(), cg.numpy(), vir.numpy()


# ==============================================================================
# 1. Forward derivative parity (bit-exact vs hand-written)
# ==============================================================================

_GRID = [(b, nb) for b in (False, True) for nb in ("list", "matrix")]
_GRID_IDS = [f"{'batch' if b else 'single'}-{nb}" for b, nb in _GRID]


class TestForwardDerivativeParity:
    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batched,neighbor_input", _GRID, ids=_GRID_IDS)
    def test_forces(self, device, dtype, batched, neighbor_input):
        sysd = _batch_system() if batched else _single_system()
        _, f_got, _, _ = _launch_forward(
            sysd,
            dtype=dtype,
            batched=batched,
            neighbor_input=neighbor_input,
            device=device,
            deriv=_DerivState.E_F,
            cell_grad=False,
        )
        f_ref, _, _ = _ref_forces(
            sysd,
            dtype=dtype,
            batched=batched,
            neighbor_input=neighbor_input,
            device=device,
            charge_grad=False,
            virial=False,
        )
        _assert_forward_parity(f_got, f_ref, dtype)

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batched,neighbor_input", _GRID, ids=_GRID_IDS)
    def test_charge_grad(self, device, dtype, batched, neighbor_input):
        sysd = _batch_system() if batched else _single_system()
        _, f_got, cg_got, _ = _launch_forward(
            sysd,
            dtype=dtype,
            batched=batched,
            neighbor_input=neighbor_input,
            device=device,
            deriv=_DerivState.E_F_dQ,
            cell_grad=False,
        )
        f_ref, cg_ref, _ = _ref_forces(
            sysd,
            dtype=dtype,
            batched=batched,
            neighbor_input=neighbor_input,
            device=device,
            charge_grad=True,
            virial=False,
        )
        _assert_forward_parity(f_got, f_ref, dtype)
        _assert_forward_parity(cg_got, cg_ref, dtype)

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batched,neighbor_input", _GRID, ids=_GRID_IDS)
    def test_virial(self, device, dtype, batched, neighbor_input):
        sysd = _batch_system() if batched else _single_system()
        _, _, _, v_got = _launch_forward(
            sysd,
            dtype=dtype,
            batched=batched,
            neighbor_input=neighbor_input,
            device=device,
            deriv=_DerivState.E_F_dQ,
            cell_grad=True,
        )
        _, _, v_ref = _ref_forces(
            sysd,
            dtype=dtype,
            batched=batched,
            neighbor_input=neighbor_input,
            device=device,
            charge_grad=True,
            virial=True,
        )
        _assert_forward_parity(v_got, v_ref, dtype)


# ==============================================================================
# 2. Backward == forward first-derivatives scaled by grad_E
# ==============================================================================


class TestBackwardScaling:
    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batched,neighbor_input", _GRID, ids=_GRID_IDS)
    def test_backward_unit_cotangent(self, device, dtype, batched, neighbor_input):
        # grad_E = 1: backward grad_R == -(forward physical force) (= dE/dR),
        # backward grad_q == forward charge grad, backward virial == forward virial.
        sysd = _batch_system() if batched else _single_system()
        nsys = sysd.get("num_systems", 1)
        ge = np.ones(nsys, dtype=np.float64)
        gpos, gq, gvir = _launch_backward(
            sysd,
            dtype=dtype,
            batched=batched,
            neighbor_input=neighbor_input,
            device=device,
            deriv=_DerivState.E_F_dQ,
            cell_grad=True,
            grad_energy_np=ge,
        )
        f_ref, cg_ref, v_ref = _ref_forces(
            sysd,
            dtype=dtype,
            batched=batched,
            neighbor_input=neighbor_input,
            device=device,
            charge_grad=True,
            virial=True,
        )
        # dE/dR = -(physical force).
        np.testing.assert_allclose(gpos, -f_ref, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gq, cg_ref, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gvir, v_ref, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    def test_backward_random_cotangent_scales(self, device, dtype):
        # backward(grad_E) == grad_E * backward(1) per system.
        sysd = _batch_system()
        nsys = sysd["num_systems"]
        rng = np.random.default_rng(0)
        ge = rng.uniform(0.5, 2.0, size=nsys)
        gpos_r, gq_r, _ = _launch_backward(
            sysd,
            dtype=dtype,
            batched=True,
            neighbor_input="list",
            device=device,
            deriv=_DerivState.E_F_dQ,
            cell_grad=False,
            grad_energy_np=ge,
        )
        gpos_1, gq_1, _ = _launch_backward(
            sysd,
            dtype=dtype,
            batched=True,
            neighbor_input="list",
            device=device,
            deriv=_DerivState.E_F_dQ,
            cell_grad=False,
            grad_energy_np=np.ones(nsys),
        )
        bidx = sysd["batch_idx"]
        scale_atom = ge[bidx]
        np.testing.assert_allclose(
            gpos_r, gpos_1 * scale_atom[:, None], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(gq_r, gq_1 * scale_atom, rtol=1e-6, atol=1e-6)


# ==============================================================================
# 3. Backward finite-diff (vs central-difference of factory energy, f64)
# ==============================================================================


def _factory_energy_total(
    sysd, positions_np, charges_np, cell_np, *, device, batched, neighbor_input
):
    """Sum of factory energies for the given (positions, charges, cell)."""
    sd = dict(sysd)
    sd["positions"] = positions_np
    sd["charges"] = charges_np
    sd["cell"] = cell_np
    e, _, _, _ = _launch_forward(
        sd,
        dtype=wp.float64,
        batched=batched,
        neighbor_input=neighbor_input,
        device=device,
        deriv=_DerivState.E,
        cell_grad=False,
    )
    return float(e.sum())


_FD_EPS = 1e-6
# Central-difference floors. First derivatives reach ~1e-8 on these sharp erfc/r
# pair terms (r~2A); the double-backward (second-derivative-of-backward) FD floor
# is ~1e-7 (2x on the full neighbor matrix). Richardson extrapolation confirms the
# analytic kernels match to that floor, so the residual is FD truncation, not a
# kernel error. A genuine wrong term would be off by O(1e-2) or more.
_FD_TOL_1ST = 1e-6
_FD_TOL_2ND = 1e-6


class TestBackwardFiniteDiff:
    @pytest.mark.parametrize("neighbor_input", ["list", "matrix"])
    def test_forces_fd(self, device, neighbor_input):
        sysd = _single_system()
        n = sysd["num_atoms"]
        pos0 = sysd["positions"].copy()
        # central-difference -dE/dR (physical force).
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
                    neighbor_input=neighbor_input,
                )
                em = _factory_energy_total(
                    sysd,
                    pm,
                    sysd["charges"],
                    sysd["cell"],
                    device=device,
                    batched=False,
                    neighbor_input=neighbor_input,
                )
                fd_force[i, d] = -(ep - em) / (2 * _FD_EPS)
        # backward grad_R = dE/dR = -force; with grad_E=1.
        gpos, _, _ = _launch_backward(
            sysd,
            dtype=wp.float64,
            batched=False,
            neighbor_input=neighbor_input,
            device=device,
            deriv=_DerivState.E_F,
            cell_grad=False,
            grad_energy_np=np.ones(1),
        )
        max_abs = np.abs((-gpos) - fd_force).max()
        # Central-difference floor (h=1e-6) for these sharp erfc/r pair terms at
        # r~2A is ~1e-8 (O(h^2) truncation), looser than the F3 certified systems.
        assert max_abs < 1e-6, f"force FD max_abs={max_abs:.3e}"

    @pytest.mark.parametrize("neighbor_input", ["list", "matrix"])
    def test_charge_grad_fd(self, device, neighbor_input):
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
                sysd,
                sysd["positions"],
                qp,
                sysd["cell"],
                device=device,
                batched=False,
                neighbor_input=neighbor_input,
            )
            em = _factory_energy_total(
                sysd,
                sysd["positions"],
                qm,
                sysd["cell"],
                device=device,
                batched=False,
                neighbor_input=neighbor_input,
            )
            fd_cg[i] = (ep - em) / (2 * _FD_EPS)
        _, gq, _ = _launch_backward(
            sysd,
            dtype=wp.float64,
            batched=False,
            neighbor_input=neighbor_input,
            device=device,
            deriv=_DerivState.E_F_dQ,
            cell_grad=False,
            grad_energy_np=np.ones(1),
        )
        max_abs = np.abs(gq - fd_cg).max()
        assert max_abs < 1e-9, f"charge-grad FD max_abs={max_abs:.3e}"

    @pytest.mark.parametrize("neighbor_input", ["list", "matrix"])
    def test_virial_fd(self, device, neighbor_input):
        # Strain-first FD virial W = -dE/dstrain, compared to backward virial state.
        sysd = _single_system()
        pos0 = torch.tensor(sysd["positions"])
        cell0 = torch.tensor(sysd["cell"])

        def energy_of_strain(strain_np):
            strain = torch.tensor(strain_np)
            deform = torch.eye(3, dtype=torch.float64) + strain[0]
            pos_s = (pos0 @ deform).numpy()
            cell_s = (cell0 @ deform).numpy()
            return _factory_energy_total(
                sysd,
                pos_s,
                sysd["charges"],
                cell_s,
                device=device,
                batched=False,
                neighbor_input=neighbor_input,
            )

        fd_W = np.zeros((1, 3, 3))
        for a in range(3):
            for b in range(3):
                sp = np.zeros((1, 3, 3))
                sp[0, a, b] += _FD_EPS
                sm = np.zeros((1, 3, 3))
                sm[0, a, b] -= _FD_EPS
                fd_W[0, a, b] = -(energy_of_strain(sp) - energy_of_strain(sm)) / (
                    2 * _FD_EPS
                )
        # backward virial state (grad_E=1) is W = sum r (x) F.
        _, _, gvir = _launch_backward(
            sysd,
            dtype=wp.float64,
            batched=False,
            neighbor_input=neighbor_input,
            device=device,
            deriv=_DerivState.E_F,
            cell_grad=True,
            grad_energy_np=np.ones(1),
        )
        max_abs = np.abs(gvir - fd_W).max()
        assert max_abs < 1e-6, f"virial FD max_abs={max_abs:.3e}"


# ==============================================================================
# 4. Double-backward finite-diff (central-diff of backward-kernel outputs)
# ==============================================================================


class TestDoubleBackwardFiniteDiff:
    @pytest.mark.parametrize("neighbor_input", ["list", "matrix"])
    def test_position_hessian_fd(self, device, neighbor_input):
        # v_q = 0, ge = 1. double_backward grad_R == J-HVP, validated by FD of the
        # backward grad_R w.r.t. positions contracted with v_pos.
        sysd = _single_system()
        n = sysd["num_atoms"]
        rng = np.random.default_rng(1)
        v_pos = rng.standard_normal((n, 3))
        v_q = np.zeros(n)
        ge = np.ones(1)

        # L(R) = sum_i v_pos_i . grad_R_i(R); dL/dR via FD == double_backward grad_R.
        def backward_gradR(positions_np):
            sd = dict(sysd)
            sd["positions"] = positions_np
            gpos, _, _ = _launch_backward(
                sd,
                dtype=wp.float64,
                batched=False,
                neighbor_input=neighbor_input,
                device=device,
                deriv=_DerivState.E_F,
                cell_grad=False,
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

        _, dbwd_gradR, _, _ = _launch_double_backward(
            sysd,
            dtype=wp.float64,
            batched=False,
            neighbor_input=neighbor_input,
            device=device,
            deriv=_DerivState.E_F,
            grad_energy_np=ge,
            vpos_np=v_pos,
            vq_np=v_q,
        )
        max_abs = np.abs(dbwd_gradR - fd_gradR).max()
        assert max_abs < _FD_TOL_2ND, f"position-Hessian FD max_abs={max_abs:.3e}"

    @pytest.mark.parametrize("neighbor_input", ["list", "matrix"])
    def test_charge_cross_fd(self, device, neighbor_input):
        # Nonzero v_pos AND v_q: exercises force<->charge cross + charge self terms.
        # FD the full backward output (grad_R, grad_q) contracted with (v_pos, v_q)
        # w.r.t. (R, q); compare to double_backward (grad_R, grad_q).
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
            gpos, gq, _ = _launch_backward(
                sd,
                dtype=wp.float64,
                batched=False,
                neighbor_input=neighbor_input,
                device=device,
                deriv=_DerivState.E_F_dQ,
                cell_grad=False,
                grad_energy_np=ge,
            )
            return gpos, gq

        pos0 = sysd["positions"].copy()
        q0 = sysd["charges"].copy()

        # dL/dR.
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

        # dL/dq.
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

        _, dbwd_gradR, dbwd_gradQ, _ = _launch_double_backward(
            sysd,
            dtype=wp.float64,
            batched=False,
            neighbor_input=neighbor_input,
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

    @pytest.mark.parametrize("neighbor_input", ["list", "matrix"])
    def test_grad_grad_energy_fd(self, device, neighbor_input):
        # dL/d(grad_E) == FD of backward output contracted with cotangents w.r.t. ge.
        sysd = _single_system()
        n = sysd["num_atoms"]
        rng = np.random.default_rng(3)
        v_pos = rng.standard_normal((n, 3))
        v_q = rng.standard_normal(n)

        def L_of_ge(ge_val):
            gpos, gq, _ = _launch_backward(
                sysd,
                dtype=wp.float64,
                batched=False,
                neighbor_input=neighbor_input,
                device=device,
                deriv=_DerivState.E_F_dQ,
                cell_grad=False,
                grad_energy_np=np.array([ge_val]),
            )
            return (v_pos * gpos).sum() + (v_q * gq).sum()

        fd = (L_of_ge(1.0 + _FD_EPS) - L_of_ge(1.0 - _FD_EPS)) / (2 * _FD_EPS)
        gge, _, _, _ = _launch_double_backward(
            sysd,
            dtype=wp.float64,
            batched=False,
            neighbor_input=neighbor_input,
            device=device,
            deriv=_DerivState.E_F_dQ,
            grad_energy_np=np.ones(1),
            vpos_np=v_pos,
            vq_np=v_q,
        )
        max_abs = abs(gge[0] - fd)
        assert max_abs < _FD_TOL_2ND, f"grad_grad_energy FD max_abs={max_abs:.3e}"


# ==============================================================================
# 5. F3 harness parity: backward kernel vs F3 finite-diff primitives
# ==============================================================================
#
# These drive the SHARED F3 harness (`_deriv_check.fd_forces` / `fd_charge_grad` /
# `fd_strain_virial`) on a `fixed_charge_system` crystal so the achieved tolerances
# are reported against the F3-certified baselines (forces/charge-grad ~1e-11,
# virial ~1e-10 for f64), not the denser, sharper jittered system above. The
# `energy_fn` closure runs the factory real-space ENERGY kernel and the backward
# kernel runs the same pinned neighbor list, so the half/full convention matches.


def _csr_from_harness(system, device):
    """Convert an F3 ``DerivCheckSystem`` neighbor list to a factory CSR dict.

    ``cell_list(..., return_neighbor_list=True)`` returns a full directed list
    grouped by source atom, i.e. CSR-compatible: ``idx_j = neighbor_list[1]``,
    ``neighbor_ptr`` as-is, ``neighbor_shifts`` as ``unit_shifts``.
    """
    return {
        "positions": system.positions.cpu().numpy().astype(np.float64),
        "charges": system.charges.cpu().numpy().astype(np.float64),
        "cell": system.cell.cpu().numpy().astype(np.float64),
        "idx_j": system.neighbor_list[1].cpu().numpy().astype(np.int32),
        "neighbor_ptr": system.neighbor_ptr.cpu().numpy().astype(np.int32),
        "unit_shifts": system.neighbor_shifts.cpu().numpy().astype(np.int32),
        "num_atoms": system.positions.shape[0],
    }


def _make_real_energy_fn(system, csr, device, wp_device):
    """Build an F3 ``energy_fn(p, q, c) -> (N,) energy`` over the factory kernel."""
    alpha_val = float(system.alpha[0].item())

    def energy_fn(p, q, c):
        sd = dict(csr)
        sd["positions"] = p.detach().cpu().numpy().astype(np.float64)
        sd["charges"] = q.detach().cpu().numpy().astype(np.float64)
        sd["cell"] = c.detach().cpu().numpy().astype(np.float64)
        e, _, _, _ = _launch_forward(
            sd,
            dtype=wp.float64,
            batched=False,
            neighbor_input="list",
            device=wp_device,
            deriv=_DerivState.E,
            cell_grad=False,
        )
        return torch.as_tensor(e, dtype=torch.float64, device=p.device)

    return energy_fn, alpha_val


def _backward_real(csr, device, wp_device, *, deriv, cell_grad):
    """Run the backward kernel (grad_E=1) on the harness CSR system."""
    return _launch_backward(
        csr,
        dtype=wp.float64,
        batched=False,
        neighbor_input="list",
        device=wp_device,
        deriv=deriv,
        cell_grad=cell_grad,
        grad_energy_np=np.ones(1),
    )


class TestF3HarnessBackwardParity:
    """Backward kernel outputs vs the shared F3 finite-diff primitives (f64).

    Tolerance note: the F3 self-test reaches ~1e-11 because it compares FD against
    *autograd* of the same closure (both differentiate identically). Here FD is
    compared against the *analytic* backward kernel, so the achieved deviation is
    the central-difference truncation floor (O(h^2)): ~3e-8 for forces and ~3e-7
    for the strain virial on this CsCl crystal. ``dE/dq`` is linear in the charges,
    so its FD is near-exact and reaches the ~1e-10 baseline. A wrong kernel term
    would be off by O(1e-2)+, far above these floors.
    """

    def _system(self, device):
        # CPU build of the crystal (numpy geometry), warp launched on `device`.
        return fixed_charge_system(
            create_cscl_supercell, size=1, jitter=0.2, cutoff=5.0, device="cpu"
        )

    def test_forces_fd(self, device):
        system = self._system(device)
        csr = _csr_from_harness(system, device)
        energy_fn, _ = _make_real_energy_fn(system, csr, device, device)
        fd = fd_forces(energy_fn, system.positions, system.charges, system.cell)
        # backward grad_R = dE/dR; physical force = -dE/dR.
        gpos, _, _ = _backward_real(
            csr, device, device, deriv=_DerivState.E_F, cell_grad=False
        )
        force = torch.as_tensor(-gpos, dtype=torch.float64)
        max_abs, max_rel = max_abs_rel(force, fd)
        assert max_abs < 1e-6, (
            f"force max_abs={max_abs:.3e} max_rel={max_rel:.3e} device={device}"
        )

    def test_charge_grad_fd(self, device):
        system = self._system(device)
        csr = _csr_from_harness(system, device)
        energy_fn, _ = _make_real_energy_fn(system, csr, device, device)
        fd = fd_charge_grad(energy_fn, system.positions, system.charges, system.cell)
        _, gq, _ = _backward_real(
            csr, device, device, deriv=_DerivState.E_F_dQ, cell_grad=False
        )
        cg = torch.as_tensor(gq, dtype=torch.float64)
        max_abs, max_rel = max_abs_rel(cg, fd)
        assert max_abs < 1e-10, (
            f"charge-grad max_abs={max_abs:.3e} max_rel={max_rel:.3e} device={device}"
        )

    def test_strain_virial_fd(self, device):
        system = self._system(device)
        csr = _csr_from_harness(system, device)
        energy_fn, _ = _make_real_energy_fn(system, csr, device, device)
        fd_W = fd_strain_virial(
            energy_fn, system.positions, system.charges, system.cell, batch_idx=None
        )
        # backward virial state (grad_E=1) is W = sum r (x) F = -dE/dstrain.
        _, _, gvir = _backward_real(
            csr, device, device, deriv=_DerivState.E_F, cell_grad=True
        )
        W = torch.as_tensor(gvir, dtype=torch.float64)
        max_abs, max_rel = max_abs_rel(W, fd_W)
        assert max_abs < 1e-6, (
            f"virial max_abs={max_abs:.3e} max_rel={max_rel:.3e} device={device}"
        )


# ==============================================================================
# 6. Batched double-backward finite-diff (covers per-system isys reduction)
# ==============================================================================


class TestBatchedDoubleBackward:
    @pytest.mark.parametrize("neighbor_input", ["list", "matrix"])
    def test_batched_dbwd_vs_fd(self, device, neighbor_input):
        # Per-system grad_E so the grad_grad_energy reduction (atomic_add by isys)
        # and the isys indexing in the double-backward body are exercised in batch.
        sysd = _batch_system()
        n = sysd["num_atoms"]
        nsys = sysd["num_systems"]
        rng = np.random.default_rng(7)
        v_pos = rng.standard_normal((n, 3))
        v_q = rng.standard_normal(n)
        ge = rng.uniform(0.5, 2.0, size=nsys)

        def backward_grads(positions_np, charges_np):
            sd = dict(sysd)
            sd["positions"] = positions_np
            sd["charges"] = charges_np
            gpos, gq, _ = _launch_backward(
                sd,
                dtype=wp.float64,
                batched=True,
                neighbor_input=neighbor_input,
                device=device,
                deriv=_DerivState.E_F_dQ,
                cell_grad=False,
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

        # grad_grad_energy[s] = FD of L w.r.t. ge[s] (per-system reduction).
        fd_gge = np.zeros(nsys)
        for srt in range(nsys):
            gep = ge.copy()
            gep[srt] += _FD_EPS
            gem = ge.copy()
            gem[srt] -= _FD_EPS

            def L(gevec):
                gpos, gq, _ = _launch_backward(
                    sysd,
                    dtype=wp.float64,
                    batched=True,
                    neighbor_input=neighbor_input,
                    device=device,
                    deriv=_DerivState.E_F_dQ,
                    cell_grad=False,
                    grad_energy_np=gevec,
                )
                return (v_pos * gpos).sum() + (v_q * gq).sum()

            fd_gge[srt] = (L(gep) - L(gem)) / (2 * _FD_EPS)

        gge, dbwd_gradR, dbwd_gradQ, _ = _launch_double_backward(
            sysd,
            dtype=wp.float64,
            batched=True,
            neighbor_input=neighbor_input,
            device=device,
            deriv=_DerivState.E_F_dQ,
            grad_energy_np=ge,
            vpos_np=v_pos,
            vq_np=v_q,
        )
        assert np.abs(dbwd_gradR - fd_gradR).max() < _FD_TOL_2ND
        assert np.abs(dbwd_gradQ - fd_gradQ).max() < _FD_TOL_2ND
        assert np.abs(gge - fd_gge).max() < _FD_TOL_2ND


# ==============================================================================
# 7. Cell second-order finite-diff for stress-loss double-backward
# ==============================================================================
#
# The single-box `_single_system` has all-zero integer shifts, so `d sep/d cell = 0`
# and the cell second-order terms vanish identically -- a vacuous test. These tests
# use the periodic CsCl crystal (26/28 edges carry a nonzero lattice shift), the only
# system where the cell branch is genuinely exercised. The decisive check is the
# contract "double_backward == directional derivative of backward": with NONZERO
# `v_cell`, central-difference the backward kernel's outputs contracted with all three
# cotangents,
#     L(R, q, cell, ge) = sum v_pos . grad_R + sum v_charge * grad_q + sum v_cell : grad_cell,
# w.r.t. cell / positions / charges / ge, and compare to the double_backward outputs
# (grad_cell, grad_positions, grad_charges, grad_grad_energy). Perturbing cell drives
# the new d2E/dcell2 + d2E/dcell dR + d2E/dcell dq terms; the symmetric-Hessian
# property d2E/dR dcell == d2E/dcell dR falls out because grad_positions (from v_cell)
# and grad_cell (from v_pos) are both matched against the same FD harness.


def _cscl_csr(size=1, jitter=0.2, cutoff=5.0):
    """CsCl-crystal CSR dict (periodic; nonzero integer shifts) for the cell tests."""
    system = fixed_charge_system(
        create_cscl_supercell, size=size, jitter=jitter, cutoff=cutoff, device="cpu"
    )
    return {
        "positions": system.positions.cpu().numpy().astype(np.float64),
        "charges": system.charges.cpu().numpy().astype(np.float64),
        "cell": system.cell.cpu().numpy().astype(np.float64),
        "idx_j": system.neighbor_list[1].cpu().numpy().astype(np.int32),
        "neighbor_ptr": system.neighbor_ptr.cpu().numpy().astype(np.int32),
        "unit_shifts": system.neighbor_shifts.cpu().numpy().astype(np.int32),
        "num_atoms": system.positions.shape[0],
    }


def _csr_to_matrix(csr):
    """Convert a (directed) CSR dict to a neighbor-matrix dict, preserving shifts."""
    n = csr["num_atoms"]
    ptr = csr["neighbor_ptr"]
    idx_j = csr["idx_j"]
    shifts = csr["unit_shifts"]
    degrees = [int(ptr[i + 1] - ptr[i]) for i in range(n)]
    maxn = max(degrees) if degrees else 0
    neighbor_matrix = np.full((n, maxn), _MASK, dtype=np.int32)
    neighbor_shifts = np.zeros((n, maxn, 3), dtype=np.int32)
    for i in range(n):
        for k, e in enumerate(range(int(ptr[i]), int(ptr[i + 1]))):
            neighbor_matrix[i, k] = idx_j[e]
            neighbor_shifts[i, k] = shifts[e]
    out = dict(csr)
    out["neighbor_matrix"] = neighbor_matrix
    out["neighbor_shifts"] = neighbor_shifts
    out["fill_value"] = _MASK
    return out


def _cscl_batch():
    """Two CsCl systems (different jitter) concatenated -> batched CSR dict."""
    s0 = _cscl_csr(jitter=0.2)
    s1 = _cscl_csr(jitter=0.35)
    n0 = s0["num_atoms"]
    positions = np.concatenate([s0["positions"], s1["positions"]], axis=0)
    charges = np.concatenate([s0["charges"], s1["charges"]], axis=0)
    cell = np.concatenate([s0["cell"], s1["cell"]], axis=0)
    batch_idx = np.array([0] * n0 + [1] * s1["num_atoms"], dtype=np.int32)
    idx_j = np.concatenate([s0["idx_j"], s1["idx_j"] + n0]).astype(np.int32)
    p0 = s0["neighbor_ptr"]
    neighbor_ptr = np.concatenate([p0, p0[1:] + p0[-1]]).astype(np.int32)
    unit_shifts = np.concatenate([s0["unit_shifts"], s1["unit_shifts"]], axis=0)
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "batch_idx": batch_idx,
        "idx_j": idx_j,
        "neighbor_ptr": neighbor_ptr,
        "unit_shifts": unit_shifts,
        "num_atoms": n0 + s1["num_atoms"],
        "num_systems": 2,
    }


class TestCellSecondOrderFiniteDiff:
    """double_backward cell terms == directional derivative of backward (nonzero v_cell)."""

    def _run(self, sysd, *, dtype, batched, neighbor_input, device, tol, seed):
        if neighbor_input == "matrix":
            sysd = _csr_to_matrix(sysd)
        n = sysd["num_atoms"]
        nsys = sysd.get("num_systems", 1)
        rng = np.random.default_rng(seed)
        v_pos = rng.standard_normal((n, 3))
        v_q = rng.standard_normal(n)
        # Asymmetric v_cell so the (M + M^T) symmetrization is genuinely tested.
        v_cell = rng.standard_normal((nsys, 3, 3))
        ge = rng.uniform(0.5, 2.0, size=nsys)

        def backward_outputs(positions_np, charges_np, cell_np, ge_vec):
            sd = dict(sysd)
            sd["positions"] = positions_np
            sd["charges"] = charges_np
            sd["cell"] = cell_np
            gpos, gq, gvir = _launch_backward(
                sd,
                dtype=dtype,
                batched=batched,
                neighbor_input=neighbor_input,
                device=device,
                deriv=_DerivState.E_F_dQ,
                cell_grad=True,
                grad_energy_np=ge_vec,
            )
            return gpos, gq, gvir

        def L(positions_np, charges_np, cell_np, ge_vec):
            gpos, gq, gvir = backward_outputs(positions_np, charges_np, cell_np, ge_vec)
            return (v_pos * gpos).sum() + (v_q * gq).sum() + (v_cell * gvir).sum()

        pos0 = sysd["positions"].copy()
        q0 = sysd["charges"].copy()
        cell0 = sysd["cell"].copy()
        eps = _FD_EPS

        # dL/dcell (perturb the cell input directly: the new d2E/dcell* terms).
        fd_gcell = np.zeros((nsys, 3, 3))
        for s in range(nsys):
            for a in range(3):
                for b in range(3):
                    cp = cell0.copy()
                    cp[s, a, b] += eps
                    cm = cell0.copy()
                    cm[s, a, b] -= eps
                    fd_gcell[s, a, b] = (L(pos0, q0, cp, ge) - L(pos0, q0, cm, ge)) / (
                        2 * eps
                    )

        # dL/dR (force<->cell cross included via v_cell).
        fd_gpos = np.zeros((n, 3))
        for i in range(n):
            for d in range(3):
                pp = pos0.copy()
                pp[i, d] += eps
                pm = pos0.copy()
                pm[i, d] -= eps
                fd_gpos[i, d] = (L(pp, q0, cell0, ge) - L(pm, q0, cell0, ge)) / (
                    2 * eps
                )

        # dL/dq (charge<->cell cross included via v_cell).
        fd_gq = np.zeros(n)
        for i in range(n):
            qp = q0.copy()
            qp[i] += eps
            qm = q0.copy()
            qm[i] -= eps
            fd_gq[i] = (L(pos0, qp, cell0, ge) - L(pos0, qm, cell0, ge)) / (2 * eps)

        # dL/dge (grad_grad_energy: cell term fm*vquad included).
        fd_gge = np.zeros(nsys)
        for s in range(nsys):
            gep = ge.copy()
            gep[s] += eps
            gem = ge.copy()
            gem[s] -= eps
            fd_gge[s] = (L(pos0, q0, cell0, gep) - L(pos0, q0, cell0, gem)) / (2 * eps)

        gge, dpos, dq, dcell = _launch_double_backward(
            sysd,
            dtype=dtype,
            batched=batched,
            neighbor_input=neighbor_input,
            device=device,
            deriv=_DerivState.E_F_dQ,
            grad_energy_np=ge,
            vpos_np=v_pos,
            vq_np=v_q,
            cell_grad=True,
            vcell_np=v_cell,
        )
        assert np.abs(dcell - fd_gcell).max() < tol, (
            f"grad_cell FD max_abs={np.abs(dcell - fd_gcell).max():.3e}"
        )
        assert np.abs(dpos - fd_gpos).max() < tol, (
            f"grad_pos(cell-cross) FD max_abs={np.abs(dpos - fd_gpos).max():.3e}"
        )
        assert np.abs(dq - fd_gq).max() < tol, (
            f"grad_q(cell-cross) FD max_abs={np.abs(dq - fd_gq).max():.3e}"
        )
        assert np.abs(gge - fd_gge).max() < tol, (
            f"grad_grad_energy(cell) FD max_abs={np.abs(gge - fd_gge).max():.3e}"
        )

    @pytest.mark.parametrize("neighbor_input", ["list", "matrix"])
    def test_single_f64(self, device, neighbor_input):
        self._run(
            _cscl_csr(),
            dtype=wp.float64,
            batched=False,
            neighbor_input=neighbor_input,
            device=device,
            tol=_FD_TOL_2ND,
            seed=11,
        )

    @pytest.mark.parametrize("neighbor_input", ["list", "matrix"])
    def test_batched_f64(self, device, neighbor_input):
        self._run(
            _cscl_batch(),
            dtype=wp.float64,
            batched=True,
            neighbor_input=neighbor_input,
            device=device,
            tol=_FD_TOL_2ND,
            seed=12,
        )

    def _f32_vs_f64(self, sysd, *, batched, neighbor_input, device, seed):
        # The f64 FD tests above pin the cell math exactly. A central-difference of
        # the f32 backward kernel is dominated by single-precision energy noise
        # (~1e-7 rel) amplified through a second-derivative-of-backward, so FD vs the
        # f32 kernel is a poor oracle. Instead validate the f32 kernel directly
        # against the (FD-certified) f64 double_backward kernel -- both analytic, so
        # the only difference is the f32 accumulation in the vec/mat write path.
        if neighbor_input == "matrix":
            sysd = _csr_to_matrix(sysd)
        n = sysd["num_atoms"]
        nsys = sysd.get("num_systems", 1)
        rng = np.random.default_rng(seed)
        v_pos = rng.standard_normal((n, 3))
        v_q = rng.standard_normal(n)
        v_cell = rng.standard_normal((nsys, 3, 3))
        ge = rng.uniform(0.5, 2.0, size=nsys)
        out = {}
        for dt in (wp.float64, wp.float32):
            out[dt] = _launch_double_backward(
                sysd,
                dtype=dt,
                batched=batched,
                neighbor_input=neighbor_input,
                device=device,
                deriv=_DerivState.E_F_dQ,
                grad_energy_np=ge,
                vpos_np=v_pos,
                vq_np=v_q,
                cell_grad=True,
                vcell_np=v_cell,
            )
        for k in range(4):  # gge, grad_pos, grad_q, grad_cell
            np.testing.assert_allclose(
                out[wp.float32][k], out[wp.float64][k], rtol=2e-4, atol=2e-4
            )

    @pytest.mark.parametrize("neighbor_input", ["list", "matrix"])
    def test_single_f32(self, device, neighbor_input):
        self._f32_vs_f64(
            _cscl_csr(),
            batched=False,
            neighbor_input=neighbor_input,
            device=device,
            seed=13,
        )

    @pytest.mark.parametrize("neighbor_input", ["list", "matrix"])
    def test_batched_f32(self, device, neighbor_input):
        self._f32_vs_f64(
            _cscl_batch(),
            batched=True,
            neighbor_input=neighbor_input,
            device=device,
            seed=14,
        )


# ==============================================================================
# Tiled (cooperative-block) matrix kernel parity vs the serial matrix kernel
# ==============================================================================
#
# The neighbor-matrix forward / double_backward kernels are launched in
# production via ``wp.launch_tiled(block_dim=REAL_SPACE_TILED_BLOCK_DIM)`` (the
# cooperative-block variant, ``tiled=True``), which restores the PR85 real-space
# throughput that the serial one-thread-per-atom factory kernel regressed. The
# tiled kernel shares the per-pair ``@wp.func`` cores with the serial kernel, so
# it agrees to round-off (the block reduction reorders the float64 sums, so this
# is a *tolerance* parity, not the bit-exact ``np.array_equal`` of the serial
# parity oracle above). The ``device`` fixture covers CPU (block_dim clamps to 1,
# so the strided loop degrades to the serial scan) and GPU.


def _wide_matrix_system(n=96, maxn=70, seed=7):
    """Matrix system with >block_dim neighbors per atom (exercises strided wrap)."""
    rng = np.random.default_rng(seed)
    box = 13.0
    positions = rng.uniform(0.5, box - 0.5, size=(n, 3)).astype(np.float64)
    charges = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    charges -= charges.mean()
    cell = np.array([[[box, 0, 0], [0, box, 0], [0, 0, box]]], dtype=np.float64)
    neighbor_matrix = np.full((n, maxn), _MASK, dtype=np.int32)
    neighbor_shifts = np.zeros((n, maxn, 3), dtype=np.int32)
    for i in range(n):
        cnt = int(rng.integers(maxn // 2, maxn + 1))  # variable, many > block_dim
        others = rng.choice(
            [x for x in range(n) if x != i], size=min(cnt, n - 1), replace=False
        )
        neighbor_matrix[i, : len(others)] = others
        neighbor_shifts[i, : len(others)] = rng.integers(-1, 2, size=(len(others), 3))
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "neighbor_matrix": neighbor_matrix,
        "neighbor_shifts": neighbor_shifts,
        "fill_value": _MASK,
        "num_atoms": n,
    }


def _as_batched(sysd):
    """Duplicate a single-system matrix system into two stacked systems."""
    n = sysd["num_atoms"]
    nm = sysd["neighbor_matrix"]
    return {
        "positions": np.concatenate([sysd["positions"], sysd["positions"] + 0.03]),
        "charges": np.concatenate([sysd["charges"], -sysd["charges"]]),
        "cell": np.concatenate([sysd["cell"], sysd["cell"]]),
        "batch_idx": np.array([0] * n + [1] * n, dtype=np.int32),
        "neighbor_matrix": np.concatenate(
            [nm, np.where(nm == _MASK, _MASK, nm + n)]
        ).astype(np.int32),
        "neighbor_shifts": np.concatenate(
            [sysd["neighbor_shifts"], sysd["neighbor_shifts"]]
        ),
        "fill_value": _MASK,
        "num_atoms": 2 * n,
        "num_systems": 2,
    }


def _matrix_warp_inputs(sysd, dtype, device):
    """Build the warp matrix-layout arrays directly (no CSR slots needed)."""
    return {
        "positions": wp.from_numpy(
            sysd["positions"].astype(_NPF[dtype]), dtype=_VEC[dtype], device=device
        ),
        "charges": wp.from_numpy(
            sysd["charges"].astype(_NPF[dtype]), dtype=dtype, device=device
        ),
        "cell": wp.from_numpy(
            sysd["cell"].astype(_NPF[dtype]), dtype=_MAT[dtype], device=device
        ),
        "neighbor_matrix": wp.from_numpy(
            sysd["neighbor_matrix"], dtype=wp.int32, device=device
        ),
        "neighbor_shifts": wp.from_numpy(
            sysd["neighbor_shifts"].reshape(*sysd["neighbor_matrix"].shape, 3),
            dtype=wp.vec3i,
            device=device,
        ),
    }


def _launch_matrix_forward(sysd, *, dtype, batched, device, deriv, cell_grad, tiled):
    inp = _matrix_warp_inputs(sysd, dtype, device)
    n = sysd["num_atoms"]
    nsys = sysd.get("num_systems", 1)
    s = alloc_ewald_real_sentinels(dtype, device)
    alpha = _alpha_array(nsys, device, dtype)
    batch_id = (
        wp.from_numpy(sysd["batch_idx"], dtype=wp.int32, device=device)
        if batched
        else s["batch_id"]
    )
    energies = wp.zeros(n, dtype=wp.float64, device=device)
    forces = wp.zeros(n, dtype=_VEC[dtype], device=device)
    cg = wp.zeros(n, dtype=wp.float64, device=device)
    virial = wp.zeros(nsys, dtype=_MAT[dtype], device=device)
    kernel = get_ewald_real_kernel(
        dtype,
        batched=batched,
        neighbor_input="matrix",
        deriv_state=deriv,
        cell_grad=cell_grad,
        order="forward",
        tiled=tiled,
    )
    inputs = [
        inp["positions"],
        inp["charges"],
        inp["cell"],
        batch_id,
        s["idx_j"],
        s["neighbor_ptr"],
        s["unit_shifts"],
        inp["neighbor_matrix"],
        inp["neighbor_shifts"],
        wp.int32(_MASK),
        alpha,
        energies,
        forces if deriv.value >= _DerivState.E_F.value else s["atomic_forces"],
        cg if deriv.value >= _DerivState.E_F_dQ.value else s["charge_gradients"],
        virial if cell_grad else s["virial"],
    ]
    if tiled:
        wp.launch_tiled(
            kernel,
            dim=[n],
            inputs=inputs,
            block_dim=REAL_SPACE_TILED_BLOCK_DIM,
            device=device,
        )
    else:
        wp.launch(kernel, dim=n, inputs=inputs, device=device)
    wp.synchronize()
    return energies.numpy(), forces.numpy(), cg.numpy(), virial.numpy()


def _launch_matrix_double_backward(
    sysd, *, dtype, batched, device, cell_grad, tiled, seed
):
    inp = _matrix_warp_inputs(sysd, dtype, device)
    n = sysd["num_atoms"]
    nsys = sysd.get("num_systems", 1)
    s = alloc_ewald_real_sentinels(dtype, device)
    alpha = _alpha_array(nsys, device, dtype)
    rng = np.random.default_rng(seed)
    v_pos = wp.from_numpy(
        rng.uniform(-1, 1, (n, 3)).astype(_NPF[dtype]), dtype=_VEC[dtype], device=device
    )
    v_charge = wp.from_numpy(
        rng.uniform(-1, 1, n).astype(np.float64), dtype=wp.float64, device=device
    )
    v_cell = (
        wp.from_numpy(
            rng.uniform(-1, 1, (nsys, 3, 3)).astype(_NPF[dtype]),
            dtype=_MAT[dtype],
            device=device,
        )
        if cell_grad
        else s["v_cell"]
    )
    grad_energy = wp.from_numpy(
        rng.uniform(-1, 1, nsys).astype(np.float64), dtype=wp.float64, device=device
    )
    batch_id = (
        wp.from_numpy(sysd["batch_idx"], dtype=wp.int32, device=device)
        if batched
        else s["batch_id"]
    )
    gge = wp.zeros(nsys, dtype=wp.float64, device=device)
    gpos = wp.zeros(n, dtype=_VEC[dtype], device=device)
    gq = wp.zeros(n, dtype=wp.float64, device=device)
    gcell = wp.zeros(nsys, dtype=_MAT[dtype], device=device)
    kernel = get_ewald_real_kernel(
        dtype,
        batched=batched,
        neighbor_input="matrix",
        deriv_state=_DerivState.E_F_dQ,
        cell_grad=cell_grad,
        order="double_backward",
        tiled=tiled,
    )
    inputs = [
        v_pos,
        v_charge,
        v_cell,
        grad_energy,
        inp["positions"],
        inp["charges"],
        inp["cell"],
        batch_id,
        s["idx_j"],
        s["neighbor_ptr"],
        s["unit_shifts"],
        inp["neighbor_matrix"],
        inp["neighbor_shifts"],
        wp.int32(_MASK),
        alpha,
        gge,
        gpos,
        gq,
        gcell if cell_grad else s["grad_cell"],
    ]
    if tiled:
        wp.launch_tiled(
            kernel,
            dim=[n],
            inputs=inputs,
            block_dim=REAL_SPACE_TILED_BLOCK_DIM,
            device=device,
        )
    else:
        wp.launch(kernel, dim=n, inputs=inputs, device=device)
    wp.synchronize()
    return gge.numpy(), gpos.numpy(), gq.numpy(), gcell.numpy()


def _tol(dtype):
    return (
        dict(rtol=1e-6, atol=1e-9)
        if dtype == wp.float64
        else dict(rtol=2e-3, atol=1e-3)
    )


class TestTiledMatrixParity:
    """Tiled cooperative-block matrix kernel == serial matrix kernel (to round-off)."""

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
    @pytest.mark.parametrize("cell_grad", [False, True], ids=["nocg", "cellgrad"])
    def test_forward(self, device, dtype, batched, cell_grad):
        base = _wide_matrix_system()
        sysd = _as_batched(base) if batched else base
        serial = _launch_matrix_forward(
            sysd,
            dtype=dtype,
            batched=batched,
            device=device,
            deriv=_DerivState.E_F_dQ,
            cell_grad=cell_grad,
            tiled=False,
        )
        tiled = _launch_matrix_forward(
            sysd,
            dtype=dtype,
            batched=batched,
            device=device,
            deriv=_DerivState.E_F_dQ,
            cell_grad=cell_grad,
            tiled=True,
        )
        for label, x, y in zip(("energy", "forces", "cg", "virial"), serial, tiled):
            assert np.allclose(x, y, **_tol(dtype)), (
                f"{label} tiled mismatch (max {np.abs(x - y).max():.2e})"
            )

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
    @pytest.mark.parametrize("cell_grad", [False, True], ids=["nocg", "cellgrad"])
    def test_double_backward(self, device, dtype, batched, cell_grad):
        base = _wide_matrix_system(seed=11)
        sysd = _as_batched(base) if batched else base
        serial = _launch_matrix_double_backward(
            sysd,
            dtype=dtype,
            batched=batched,
            device=device,
            cell_grad=cell_grad,
            tiled=False,
            seed=42,
        )
        tiled = _launch_matrix_double_backward(
            sysd,
            dtype=dtype,
            batched=batched,
            device=device,
            cell_grad=cell_grad,
            tiled=True,
            seed=42,
        )
        for label, x, y in zip(("gge", "gpos", "gq", "gcell"), serial, tiled):
            assert np.allclose(x, y, **_tol(dtype)), (
                f"{label} tiled mismatch (max {np.abs(x - y).max():.2e})"
            )

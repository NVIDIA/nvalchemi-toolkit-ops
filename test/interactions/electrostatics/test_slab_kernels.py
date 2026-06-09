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
Unit tests for slab correction Warp kernel launchers.

Tests cover:
- Moment reduction correctness (M_z, M_z2, Q_total)
- Per-atom energy, force, charge-gradient, and virial correctness
- Float32 and float64 output paths
- Optional-output launch paths
- Non-periodic axis selection
- Triclinic projected-normal geometry
- Mixed-axis and mixed-pbc batches, including 3D no-op systems

These tests use Warp arrays directly and do not require PyTorch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from nvalchemiops.interactions.electrostatics.slab_kernels import (
    slab_correction,
    slab_correction_backward,
    slab_correction_double_backward,
    slab_precompute_geometry,
    slab_reduce_moments,
)

PI = math.pi
TWOPI = 2.0 * PI
FOURPI = 4.0 * PI


# ==============================================================================
# Helpers
# ==============================================================================


def _axis_to_pbc(axis: int) -> np.ndarray:
    """Convert a non-periodic axis index (0, 1, or 2) to a (3,) pbc bool array."""
    pbc = np.array([True, True, True], dtype=np.bool_)
    pbc[axis] = False
    return pbc


def _slab_normal(cell, axis):
    """Return the periodic-plane normal for a cell with row-vector lattice."""
    normal = np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])
    return normal / np.linalg.norm(normal)


def analytical_slab_correction(positions, charges, cell, axis):
    """Compute slab correction using numpy for reference.

    Returns per-atom energies, forces (N,3), charge_grads, and per-atom virial
    matrices using the normal-following triclinic geometry.
    """
    normal = _slab_normal(cell, axis)
    z = positions @ normal
    q = charges
    V = abs(np.linalg.det(cell))
    L2 = np.dot(cell[axis], normal) ** 2

    M = np.sum(q * z)
    M2 = np.sum(q * z**2)
    Q = np.sum(q)

    # Per-atom energy
    bracket = z * M - 0.5 * (M2 + Q * z**2) - Q / 12.0 * L2
    energies = (TWOPI / V) * q * bracket

    # Per-atom force along slab normal
    forces = (-(FOURPI / V) * q * (M - Q * z))[:, None] * normal[None, :]

    # Per-atom charge gradient
    charge_grads = (FOURPI / V) * bracket

    projector = np.eye(3) - 2.0 * np.outer(normal, normal)
    virial_per_atom = energies[:, None, None] * projector[None, :, :]

    return energies, forces, charge_grads, virial_per_atom


def _make_warp_arrays(system, wp_dtype, device="cpu"):
    """Convert a single-system test dict to Warp arrays for kernel calls.

    `system` must contain: positions, charges, cell, axis, num_atoms.
    Returns a dict of Warp arrays + helpful metadata.
    """
    if wp_dtype == wp.float32:
        np_dtype = np.float32
        vec_dtype = wp.vec3f
        mat_dtype = wp.mat33f
    else:
        np_dtype = np.float64
        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

    N = system["num_atoms"]
    cell_np = system["cell"]
    axis = system["axis"]

    positions = wp.array(
        system["positions"].astype(np_dtype), dtype=vec_dtype, device=device
    )
    charges = wp.array(
        system["charges"].astype(np_dtype), dtype=wp_dtype, device=device
    )
    batch_idx = wp.zeros(N, dtype=wp.int32, device=device)

    # pbc as (1, 3) bool array (single system)
    pbc_np = _axis_to_pbc(axis)[None, :]  # (1, 3)
    pbc = wp.array(pbc_np, dtype=wp.bool, device=device)

    # Cell as (1,) array of mat33; kernel computes volume / lengths internally.
    cell_arr = wp.array(
        cell_np[None, :, :].astype(np_dtype), dtype=mat_dtype, device=device
    )

    # Moment arrays (float64, zero-initialized). mz, mz2 are (B, 3)
    # with projected M and M2 stored in the non-periodic axis slot.
    mz = wp.zeros((1, 3), dtype=wp.float64, device=device)
    mz2 = wp.zeros((1, 3), dtype=wp.float64, device=device)
    qtotal = wp.zeros(1, dtype=wp.float64, device=device)
    slab_axis = wp.zeros(1, dtype=wp.int32, device=device)
    slab_normal = wp.zeros(1, dtype=wp.vec3d, device=device)
    slab_volume = wp.zeros(1, dtype=wp.float64, device=device)
    slab_height_sq = wp.zeros(1, dtype=wp.float64, device=device)

    # Output arrays
    energy_in = wp.zeros(N, dtype=wp.float64, device=device)
    energy_out = wp.zeros(N, dtype=wp.float64, device=device)
    forces = wp.zeros(N, dtype=vec_dtype, device=device)
    charge_grads = wp.zeros(N, dtype=wp.float64, device=device)
    virial = wp.zeros(1, dtype=mat_dtype, device=device)

    return {
        "positions": positions,
        "charges": charges,
        "batch_idx": batch_idx,
        "pbc": pbc,
        "cell": cell_arr,
        "axis": axis,
        "mz": mz,
        "mz2": mz2,
        "qtotal": qtotal,
        "slab_axis": slab_axis,
        "slab_normal": slab_normal,
        "slab_volume": slab_volume,
        "slab_height_sq": slab_height_sq,
        "energy_in": energy_in,
        "energy_out": energy_out,
        "forces": forces,
        "charge_grads": charge_grads,
        "virial": virial,
        "wp_dtype": wp_dtype,
    }


def _run_kernels(
    w,
    *,
    compute_forces=True,
    compute_charge_gradients=True,
    compute_virial=True,
):
    """Launch both kernels with kwargs from _make_warp_arrays output."""
    slab_reduce_moments(
        positions=w["positions"],
        charges=w["charges"],
        batch_idx=w["batch_idx"],
        pbc=w["pbc"],
        cell=w["cell"],
        mz=w["mz"],
        mz2=w["mz2"],
        qtotal=w["qtotal"],
        wp_dtype=w["wp_dtype"],
    )
    slab_precompute_geometry(
        pbc=w["pbc"],
        cell=w["cell"],
        slab_axis=w["slab_axis"],
        slab_normal=w["slab_normal"],
        slab_volume=w["slab_volume"],
        slab_height_sq=w["slab_height_sq"],
        wp_dtype=w["wp_dtype"],
    )
    slab_correction(
        positions=w["positions"],
        charges=w["charges"],
        batch_idx=w["batch_idx"],
        pbc=w["pbc"],
        cell=w["cell"],
        mz=w["mz"],
        mz2=w["mz2"],
        qtotal=w["qtotal"],
        slab_axis=w["slab_axis"],
        slab_normal=w["slab_normal"],
        slab_volume=w["slab_volume"],
        slab_height_sq=w["slab_height_sq"],
        energy_in=w["energy_in"],
        energy_out=w["energy_out"],
        forces=w["forces"],
        charge_grads=w["charge_grads"],
        virial=w["virial"],
        wp_dtype=w["wp_dtype"],
        compute_forces=compute_forces,
        compute_charge_gradients=compute_charge_gradients,
        compute_virial=compute_virial,
    )
    wp.synchronize()


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def slab_system_z():
    """4-atom slab system, non-periodic along z.

    Positions and charges chosen so M_z, M_z2, Q are all nonzero
    to exercise all terms in the correction formula.

    Cell: 10 x 10 x 30 (large z for vacuum gap).
    """
    positions = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [2.0, 3.0, 1.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )
    charges = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float64)
    cell = np.array(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 30.0]],
        dtype=np.float64,
    )
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "axis": 2,
        "num_atoms": 4,
    }


# ==============================================================================
# Test 1: Moment reduction correctness
# ==============================================================================


class TestMomentReduction:
    """Test that moment reduction kernel computes M_z, M_z2, Q correctly."""

    @pytest.mark.parametrize("wp_dtype", [wp.float32, wp.float64])
    def test_moments(self, slab_system_z, wp_dtype, device):
        w = _make_warp_arrays(slab_system_z, wp_dtype, device)

        slab_reduce_moments(
            positions=w["positions"],
            charges=w["charges"],
            batch_idx=w["batch_idx"],
            pbc=w["pbc"],
            cell=w["cell"],
            mz=w["mz"],
            mz2=w["mz2"],
            qtotal=w["qtotal"],
            wp_dtype=wp_dtype,
        )
        wp.synchronize()

        # Expected values from numpy
        axis = slab_system_z["axis"]
        z = slab_system_z["positions"] @ _slab_normal(slab_system_z["cell"], axis)
        q = slab_system_z["charges"]
        expected_mz = np.sum(q * z)
        expected_mz2 = np.sum(q * z**2)
        expected_q = np.sum(q)

        rtol = 1e-5 if wp_dtype == wp.float32 else 1e-12
        np.testing.assert_allclose(w["mz"].numpy()[0, axis], expected_mz, rtol=rtol)
        np.testing.assert_allclose(w["mz2"].numpy()[0, axis], expected_mz2, rtol=rtol)
        np.testing.assert_allclose(w["qtotal"].numpy()[0], expected_q, rtol=rtol)

    def test_moments_3d_periodic_zero(self, slab_system_z, device):
        """For pbc=[T, T, T] (3D periodic), no contribution to moments."""
        w = _make_warp_arrays(slab_system_z, wp.float64, device)
        # Override pbc with all True
        pbc_3d = wp.array(
            np.array([[True, True, True]], dtype=np.bool_),
            dtype=wp.bool,
            device=device,
        )
        slab_reduce_moments(
            positions=w["positions"],
            charges=w["charges"],
            batch_idx=w["batch_idx"],
            pbc=pbc_3d,
            cell=w["cell"],
            mz=w["mz"],
            mz2=w["mz2"],
            qtotal=w["qtotal"],
            wp_dtype=wp.float64,
        )
        wp.synchronize()
        np.testing.assert_allclose(w["mz"].numpy()[0], 0.0, atol=1e-15)
        np.testing.assert_allclose(w["mz2"].numpy()[0], 0.0, atol=1e-15)
        np.testing.assert_allclose(w["qtotal"].numpy()[0], 0.0, atol=1e-15)


# ==============================================================================
# Test 2: Per-atom energy correctness
# ==============================================================================


class TestSlabOutputs:
    """Test slab correction outputs against the analytical formula."""

    @pytest.mark.parametrize("wp_dtype", [wp.float32, wp.float64])
    def test_orthogonal_outputs(self, slab_system_z, wp_dtype, device):
        w = _make_warp_arrays(slab_system_z, wp_dtype, device)
        _run_kernels(w)

        expected_e, expected_f, expected_cg, expected_v = analytical_slab_correction(
            slab_system_z["positions"],
            slab_system_z["charges"],
            slab_system_z["cell"],
            slab_system_z["axis"],
        )

        rtol = 1e-5 if wp_dtype == wp.float32 else 1e-12
        np.testing.assert_allclose(w["energy_out"].numpy(), expected_e, rtol=rtol)
        np.testing.assert_allclose(
            w["forces"].numpy(), expected_f, rtol=rtol, atol=1e-15
        )
        np.testing.assert_allclose(w["charge_grads"].numpy(), expected_cg, rtol=rtol)
        np.testing.assert_allclose(
            w["virial"].numpy()[0], expected_v.sum(axis=0), rtol=rtol, atol=1e-15
        )


# ==============================================================================
# Test 2b: Optional output flags
# ==============================================================================


class TestSlabOutputFlags:
    """Test that the correction kernel only writes requested optional outputs."""

    def test_energy_only_skips_optional_outputs(self, slab_system_z, device):
        """Energy-only mode leaves force, charge-grad, and virial outputs untouched."""
        w = _make_warp_arrays(slab_system_z, wp.float64, device)
        _run_kernels(
            w,
            compute_forces=False,
            compute_charge_gradients=False,
            compute_virial=False,
        )

        expected_e, _, _, _ = analytical_slab_correction(
            slab_system_z["positions"],
            slab_system_z["charges"],
            slab_system_z["cell"],
            slab_system_z["axis"],
        )

        np.testing.assert_allclose(w["energy_out"].numpy(), expected_e, rtol=1e-12)
        np.testing.assert_allclose(w["forces"].numpy(), 0.0, atol=1e-15)
        np.testing.assert_allclose(w["charge_grads"].numpy(), 0.0, atol=1e-15)
        np.testing.assert_allclose(w["virial"].numpy(), 0.0, atol=1e-15)

    def test_forces_and_charge_grads_skip_virial(self, slab_system_z, device):
        """Force/charge-gradient mode leaves virial output untouched."""
        w = _make_warp_arrays(slab_system_z, wp.float64, device)
        _run_kernels(
            w,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=False,
        )

        _, expected_f, expected_cg, _ = analytical_slab_correction(
            slab_system_z["positions"],
            slab_system_z["charges"],
            slab_system_z["cell"],
            slab_system_z["axis"],
        )

        np.testing.assert_allclose(
            w["forces"].numpy(), expected_f, rtol=1e-12, atol=1e-15
        )
        np.testing.assert_allclose(w["charge_grads"].numpy(), expected_cg, rtol=1e-12)
        np.testing.assert_allclose(w["virial"].numpy(), 0.0, atol=1e-15)

    def test_charge_grads_do_not_mutate_disabled_forces(self, slab_system_z, device):
        """Charge-gradient mode does not write force output when forces are disabled."""
        w = _make_warp_arrays(slab_system_z, wp.float64, device)
        sentinel = np.full((slab_system_z["positions"].shape[0], 3), 5.0)
        w["forces"] = wp.array(sentinel, dtype=wp.vec3d, device=device)

        _run_kernels(
            w,
            compute_forces=False,
            compute_charge_gradients=True,
            compute_virial=False,
        )

        expected_e, _, expected_cg, _ = analytical_slab_correction(
            slab_system_z["positions"],
            slab_system_z["charges"],
            slab_system_z["cell"],
            slab_system_z["axis"],
        )

        np.testing.assert_allclose(w["energy_out"].numpy(), expected_e, rtol=1e-12)
        np.testing.assert_allclose(w["charge_grads"].numpy(), expected_cg, rtol=1e-12)
        np.testing.assert_allclose(w["forces"].numpy(), sentinel, atol=0.0)
        np.testing.assert_allclose(w["virial"].numpy(), 0.0, atol=1e-15)

    def test_virial_does_not_mutate_disabled_forces(self, slab_system_z, device):
        """Virial mode does not write force output when forces are disabled."""
        w = _make_warp_arrays(slab_system_z, wp.float64, device)
        sentinel = np.full((slab_system_z["positions"].shape[0], 3), 5.0)
        w["forces"] = wp.array(sentinel, dtype=wp.vec3d, device=device)

        _run_kernels(
            w,
            compute_forces=False,
            compute_charge_gradients=False,
            compute_virial=True,
        )

        expected_e, _, _, expected_v = analytical_slab_correction(
            slab_system_z["positions"],
            slab_system_z["charges"],
            slab_system_z["cell"],
            slab_system_z["axis"],
        )

        np.testing.assert_allclose(w["energy_out"].numpy(), expected_e, rtol=1e-12)
        np.testing.assert_allclose(
            w["virial"].numpy()[0], expected_v.sum(axis=0), rtol=1e-12, atol=1e-15
        )
        np.testing.assert_allclose(w["forces"].numpy(), sentinel, atol=0.0)
        np.testing.assert_allclose(w["charge_grads"].numpy(), 0.0, atol=1e-15)

    def test_charge_grads_and_virial_do_not_mutate_disabled_forces(
        self, slab_system_z, device
    ):
        """Charge-gradient plus virial mode does not write force output."""
        w = _make_warp_arrays(slab_system_z, wp.float64, device)
        sentinel = np.full((slab_system_z["positions"].shape[0], 3), 5.0)
        w["forces"] = wp.array(sentinel, dtype=wp.vec3d, device=device)

        _run_kernels(
            w,
            compute_forces=False,
            compute_charge_gradients=True,
            compute_virial=True,
        )

        expected_e, _, expected_cg, expected_v = analytical_slab_correction(
            slab_system_z["positions"],
            slab_system_z["charges"],
            slab_system_z["cell"],
            slab_system_z["axis"],
        )

        np.testing.assert_allclose(w["energy_out"].numpy(), expected_e, rtol=1e-12)
        np.testing.assert_allclose(w["charge_grads"].numpy(), expected_cg, rtol=1e-12)
        np.testing.assert_allclose(
            w["virial"].numpy()[0], expected_v.sum(axis=0), rtol=1e-12, atol=1e-15
        )
        np.testing.assert_allclose(w["forces"].numpy(), sentinel, atol=0.0)


# ==============================================================================
# Test 3: Triclinic projected-normal geometry
# ==============================================================================


class TestTriclinicGeometry:
    """Test slab correction for tilted periodic planes."""

    def test_triclinic_outputs(self, device):
        """Triclinic energy/force/charge-grad/virial match the reference."""
        positions = np.array(
            [[1.0, 2.0, 3.0], [4.0, 1.5, 6.0], [2.0, 3.5, 7.5]],
            dtype=np.float64,
        )
        charges = np.array([1.0, -0.5, 0.3], dtype=np.float64)
        cell = np.array(
            [[9.0, 0.0, 0.0], [2.0, 8.0, 1.5], [0.5, 0.2, 25.0]],
            dtype=np.float64,
        )
        system = {
            "positions": positions,
            "charges": charges,
            "cell": cell,
            "axis": 2,
            "num_atoms": 3,
        }

        w = _make_warp_arrays(system, wp.float64, device)
        _run_kernels(w)

        expected_e, expected_f, expected_cg, expected_v = analytical_slab_correction(
            positions, charges, cell, 2
        )
        normal = _slab_normal(cell, 2)

        np.testing.assert_allclose(
            w["mz"].numpy()[0, 2], np.sum(charges * (positions @ normal)), rtol=1e-12
        )
        np.testing.assert_allclose(
            w["mz2"].numpy()[0, 2],
            np.sum(charges * (positions @ normal) ** 2),
            rtol=1e-12,
        )
        np.testing.assert_allclose(w["energy_out"].numpy(), expected_e, rtol=1e-12)
        np.testing.assert_allclose(
            w["forces"].numpy(), expected_f, rtol=1e-12, atol=1e-15
        )
        np.testing.assert_allclose(w["charge_grads"].numpy(), expected_cg, rtol=1e-12)
        np.testing.assert_allclose(
            w["virial"].numpy()[0], expected_v.sum(axis=0), rtol=1e-12, atol=1e-15
        )


# ==============================================================================
# Test 4: Non-periodic axis selection
# ==============================================================================


class TestAxisSelection:
    """Test that the correction works for all three axis choices."""

    @pytest.mark.parametrize("axis", [0, 1])
    def test_axis(self, axis, device):
        """Rotate the same physical system so the non-periodic axis changes."""
        # Base system: slab along z
        base_positions = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [2.0, 3.0, 1.0]],
            dtype=np.float64,
        )
        base_charges = np.array([1.0, -1.0, 0.5], dtype=np.float64)
        base_cell = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 30.0]],
            dtype=np.float64,
        )

        # Rotate: cycle axes so that 'axis' becomes the non-periodic one
        perm = [(axis + 1) % 3, (axis + 2) % 3, axis]
        positions = base_positions[:, perm]
        cell = base_cell[np.ix_(perm, perm)]

        system = {
            "positions": positions,
            "charges": base_charges,
            "cell": cell,
            "axis": axis,
            "num_atoms": 3,
        }

        w = _make_warp_arrays(system, wp.float64, device)
        _run_kernels(w)

        expected_e, expected_f, _, _ = analytical_slab_correction(
            positions, base_charges, cell, axis
        )

        np.testing.assert_allclose(w["energy_out"].numpy(), expected_e, rtol=1e-12)
        np.testing.assert_allclose(
            w["forces"].numpy(), expected_f, rtol=1e-12, atol=1e-15
        )


# ==============================================================================
# Test 5: Mixed-axis and mixed-pbc batches
# ==============================================================================


class TestMixedAxisBatch:
    """Test a batch where different systems have different non-periodic axes."""

    def test_mixed_axes_and_3d_pbc(self, device):
        """Mixed slab axes compute correctly while 3D-pbc systems stay zero."""
        wp_dtype = wp.float64

        # System A: slab in z
        pos_a = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 5.0]], dtype=np.float64)
        q_a = np.array([1.0, -1.0], dtype=np.float64)
        cell_a = np.array(
            [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 24.0]], dtype=np.float64
        )

        # System B: triclinic slab in y
        pos_b = np.array(
            [[1.0, 3.0, 0.5], [2.0, 7.0, 1.5], [3.0, 1.0, 2.5]], dtype=np.float64
        )
        q_b = np.array([0.5, -0.3, 0.2], dtype=np.float64)
        cell_b = np.array(
            [[12.0, 0.8, 1.0], [0.2, 36.0, 1.0], [1.5, 0.4, 12.0]],
            dtype=np.float64,
        )

        # System C: fully periodic, so slab correction must be zero.
        pos_c = np.array(
            [[1.0, 2.0, 0.5], [2.0, 3.0, 1.5], [3.0, 1.0, 2.5]], dtype=np.float64
        )
        q_c = np.array([0.5, -0.3, 0.2], dtype=np.float64)
        cell_c = np.array(
            [[10.0, 0.5, 0.0], [0.0, 11.0, 0.4], [0.2, 0.0, 12.0]],
            dtype=np.float64,
        )

        e_ref_a, f_ref_a, cg_ref_a, v_ref_a = analytical_slab_correction(
            pos_a, q_a, cell_a, 2
        )
        e_ref_b, f_ref_b, cg_ref_b, v_ref_b = analytical_slab_correction(
            pos_b, q_b, cell_b, 1
        )

        batch_pos = np.concatenate([pos_a, pos_b, pos_c], axis=0)
        batch_q = np.concatenate([q_a, q_b, q_c])
        batch_idx_np = np.array([0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        total_atoms = len(batch_q)

        wp_positions = wp.array(batch_pos, dtype=wp.vec3d, device=device)
        wp_charges = wp.array(batch_q, dtype=wp.float64, device=device)
        wp_batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)

        pbc_np = np.array(
            [[True, True, False], [True, False, True], [True, True, True]],
            dtype=np.bool_,
        )
        wp_pbc = wp.array(pbc_np, dtype=wp.bool, device=device)

        cell_batch = np.stack([cell_a, cell_b, cell_c], axis=0).astype(np.float64)
        wp_cell = wp.array(cell_batch, dtype=wp.mat33d, device=device)

        wp_mz = wp.zeros((3, 3), dtype=wp.float64, device=device)
        wp_mz2 = wp.zeros((3, 3), dtype=wp.float64, device=device)
        wp_qtotal = wp.zeros(3, dtype=wp.float64, device=device)
        wp_slab_axis = wp.zeros(3, dtype=wp.int32, device=device)
        wp_slab_normal = wp.zeros(3, dtype=wp.vec3d, device=device)
        wp_slab_volume = wp.zeros(3, dtype=wp.float64, device=device)
        wp_slab_height_sq = wp.zeros(3, dtype=wp.float64, device=device)
        wp_energy_in = wp.zeros(total_atoms, dtype=wp.float64, device=device)
        wp_energy_out = wp.zeros(total_atoms, dtype=wp.float64, device=device)
        wp_forces = wp.zeros(total_atoms, dtype=wp.vec3d, device=device)
        wp_charge_grads = wp.zeros(total_atoms, dtype=wp.float64, device=device)
        wp_virial = wp.zeros(3, dtype=wp.mat33d, device=device)

        slab_reduce_moments(
            wp_positions,
            wp_charges,
            wp_batch_idx,
            wp_pbc,
            wp_cell,
            wp_mz,
            wp_mz2,
            wp_qtotal,
            wp_dtype,
        )
        slab_precompute_geometry(
            wp_pbc,
            wp_cell,
            wp_slab_axis,
            wp_slab_normal,
            wp_slab_volume,
            wp_slab_height_sq,
            wp_dtype,
        )
        slab_correction(
            wp_positions,
            wp_charges,
            wp_batch_idx,
            wp_pbc,
            wp_cell,
            wp_mz,
            wp_mz2,
            wp_qtotal,
            wp_slab_axis,
            wp_slab_normal,
            wp_slab_volume,
            wp_slab_height_sq,
            wp_energy_in,
            wp_energy_out,
            wp_forces,
            wp_charge_grads,
            wp_virial,
            wp_dtype,
        )
        wp.synchronize()

        e_out = wp_energy_out.numpy()
        f_out = wp_forces.numpy()
        cg_out = wp_charge_grads.numpy()
        v_out = wp_virial.numpy()
        mz_out = wp_mz.numpy()
        mz2_out = wp_mz2.numpy()
        qtotal_out = wp_qtotal.numpy()

        a_atoms = slice(0, 2)
        b_atoms = slice(2, 5)
        c_atoms = slice(5, 8)

        # System A slab energies match the analytical slab reference.
        np.testing.assert_allclose(e_out[a_atoms], e_ref_a, rtol=1e-12)
        # System A slab forces match the analytical slab reference.
        np.testing.assert_allclose(f_out[a_atoms], f_ref_a, rtol=1e-12, atol=1e-15)
        # System A slab charge gradients match the analytical slab reference.
        np.testing.assert_allclose(cg_out[a_atoms], cg_ref_a, rtol=1e-12)
        # System A slab virial matches the analytical slab reference.
        np.testing.assert_allclose(v_out[0], v_ref_a.sum(axis=0), rtol=1e-12)

        # System B triclinic slab energies match the analytical slab reference.
        np.testing.assert_allclose(e_out[b_atoms], e_ref_b, rtol=1e-12)
        # System B triclinic slab forces match the analytical slab reference.
        np.testing.assert_allclose(f_out[b_atoms], f_ref_b, rtol=1e-12, atol=1e-15)
        # System B triclinic slab charge gradients match the analytical slab reference.
        np.testing.assert_allclose(cg_out[b_atoms], cg_ref_b, rtol=1e-12)
        # System B triclinic slab virial matches the analytical slab reference.
        np.testing.assert_allclose(v_out[1], v_ref_b.sum(axis=0), rtol=1e-12)

        # Fully periodic system energies remain unchanged by slab correction.
        np.testing.assert_allclose(e_out[c_atoms], 0.0, rtol=0, atol=0)
        # Fully periodic system forces remain unchanged by slab correction.
        np.testing.assert_allclose(f_out[c_atoms], 0.0, rtol=0, atol=0)
        # Fully periodic system charge gradients remain unchanged by slab correction.
        np.testing.assert_allclose(cg_out[c_atoms], 0.0, rtol=0, atol=0)
        # Fully periodic system virial remains unchanged by slab correction.
        np.testing.assert_allclose(v_out[2], 0.0, rtol=0, atol=0)
        # Fully periodic system moments are skipped by slab reduction.
        np.testing.assert_allclose(mz_out[2], 0.0, rtol=0, atol=0)
        # Fully periodic system squared moments are skipped by slab reduction.
        np.testing.assert_allclose(mz2_out[2], 0.0, rtol=0, atol=0)
        # Fully periodic system total charge is skipped by slab reduction.
        np.testing.assert_allclose(qtotal_out[2], 0.0, rtol=0, atol=0)


# ==============================================================================
# Analytic slab HVP kernels
# ==============================================================================


def _types_for_wp_dtype(wp_dtype):
    """Return matching NumPy, vector, and matrix dtypes for a Warp scalar dtype."""
    if wp_dtype == wp.float32:
        return np.float32, wp.vec3f, wp.mat33f
    return np.float64, wp.vec3d, wp.mat33d


def _run_backward_np(
    positions,
    charges,
    cell,
    batch_idx,
    pbc,
    grad_system,
    wp_dtype,
    device,
):
    """Run first-order slab backward and return NumPy outputs."""
    np_dtype, vec_dtype, mat_dtype = _types_for_wp_dtype(wp_dtype)
    num_atoms = charges.shape[0]
    num_systems = cell.shape[0]
    wp_positions = wp.array(positions.astype(np_dtype), dtype=vec_dtype, device=device)
    wp_charges = wp.array(charges.astype(np_dtype), dtype=wp_dtype, device=device)
    wp_batch_idx = wp.array(batch_idx.astype(np.int32), dtype=wp.int32, device=device)
    wp_pbc = wp.array(pbc.astype(np.bool_), dtype=wp.bool, device=device)
    wp_cell = wp.array(cell.astype(np_dtype), dtype=mat_dtype, device=device)
    wp_mz = wp.zeros((num_systems, 3), dtype=wp.float64, device=device)
    wp_mz2 = wp.zeros((num_systems, 3), dtype=wp.float64, device=device)
    wp_qtotal = wp.zeros(num_systems, dtype=wp.float64, device=device)
    wp_slab_axis = wp.zeros(num_systems, dtype=wp.int32, device=device)
    wp_slab_normal = wp.zeros(num_systems, dtype=wp.vec3d, device=device)
    wp_slab_volume = wp.zeros(num_systems, dtype=wp.float64, device=device)
    wp_slab_height_sq = wp.zeros(num_systems, dtype=wp.float64, device=device)
    wp_grad_system = wp.array(
        grad_system.astype(np.float64), dtype=wp.float64, device=device
    )
    wp_grad_positions = wp.zeros(num_atoms, dtype=vec_dtype, device=device)
    wp_grad_charges = wp.zeros(num_atoms, dtype=wp.float64, device=device)
    wp_grad_normal = wp.zeros(num_systems, dtype=wp.vec3d, device=device)
    wp_grad_cell = wp.zeros(num_systems, dtype=mat_dtype, device=device)

    slab_reduce_moments(
        wp_positions,
        wp_charges,
        wp_batch_idx,
        wp_pbc,
        wp_cell,
        wp_mz,
        wp_mz2,
        wp_qtotal,
        wp_dtype,
    )
    slab_precompute_geometry(
        wp_pbc,
        wp_cell,
        wp_slab_axis,
        wp_slab_normal,
        wp_slab_volume,
        wp_slab_height_sq,
        wp_dtype,
    )
    slab_correction_backward(
        wp_positions,
        wp_charges,
        wp_batch_idx,
        wp_pbc,
        wp_cell,
        wp_mz,
        wp_mz2,
        wp_qtotal,
        wp_slab_axis,
        wp_slab_normal,
        wp_slab_volume,
        wp_slab_height_sq,
        wp_grad_system,
        wp_grad_positions,
        wp_grad_charges,
        wp_grad_normal,
        wp_grad_cell,
        wp_dtype,
    )
    wp.synchronize()
    return (
        wp_grad_positions.numpy(),
        wp_grad_charges.numpy(),
        wp_grad_cell.numpy(),
        wp_mz.numpy(),
        wp_mz2.numpy(),
        wp_qtotal.numpy(),
    )


def _run_double_backward_np(
    positions,
    charges,
    cell,
    h_positions,
    h_charges,
    h_cell,
    batch_idx,
    pbc,
    grad_system,
    wp_dtype,
    device,
):
    """Run analytic slab HVP kernels and return NumPy outputs plus moment tangents."""
    np_dtype, vec_dtype, mat_dtype = _types_for_wp_dtype(wp_dtype)
    num_atoms = charges.shape[0]
    num_systems = cell.shape[0]
    wp_positions = wp.array(positions.astype(np_dtype), dtype=vec_dtype, device=device)
    wp_charges = wp.array(charges.astype(np_dtype), dtype=wp_dtype, device=device)
    wp_h_positions = wp.array(
        h_positions.astype(np_dtype), dtype=vec_dtype, device=device
    )
    wp_h_charges = wp.array(
        h_charges.astype(np.float64), dtype=wp.float64, device=device
    )
    wp_h_cell = wp.array(h_cell.astype(np_dtype), dtype=mat_dtype, device=device)
    wp_batch_idx = wp.array(batch_idx.astype(np.int32), dtype=wp.int32, device=device)
    wp_pbc = wp.array(pbc.astype(np.bool_), dtype=wp.bool, device=device)
    wp_cell = wp.array(cell.astype(np_dtype), dtype=mat_dtype, device=device)
    wp_mz = wp.zeros((num_systems, 3), dtype=wp.float64, device=device)
    wp_mz2 = wp.zeros((num_systems, 3), dtype=wp.float64, device=device)
    wp_qtotal = wp.zeros(num_systems, dtype=wp.float64, device=device)
    wp_slab_axis = wp.zeros(num_systems, dtype=wp.int32, device=device)
    wp_slab_normal = wp.zeros(num_systems, dtype=wp.vec3d, device=device)
    wp_slab_volume = wp.zeros(num_systems, dtype=wp.float64, device=device)
    wp_slab_height_sq = wp.zeros(num_systems, dtype=wp.float64, device=device)
    wp_grad_system = wp.array(
        grad_system.astype(np.float64), dtype=wp.float64, device=device
    )
    wp_dmz = wp.zeros((num_systems, 3), dtype=wp.float64, device=device)
    wp_dmz2 = wp.zeros((num_systems, 3), dtype=wp.float64, device=device)
    wp_dqtotal = wp.zeros(num_systems, dtype=wp.float64, device=device)
    wp_dnormal = wp.zeros(num_systems, dtype=wp.vec3d, device=device)
    wp_dvolume = wp.zeros(num_systems, dtype=wp.float64, device=device)
    wp_dheight_sq = wp.zeros(num_systems, dtype=wp.float64, device=device)
    wp_grad_normal = wp.zeros(num_systems, dtype=wp.vec3d, device=device)
    wp_h_grad_normal = wp.zeros(num_systems, dtype=wp.vec3d, device=device)
    wp_grad_positions = wp.zeros(num_atoms, dtype=vec_dtype, device=device)
    wp_grad_charges = wp.zeros(num_atoms, dtype=wp.float64, device=device)
    wp_grad_cell = wp.zeros(num_systems, dtype=mat_dtype, device=device)

    slab_reduce_moments(
        wp_positions,
        wp_charges,
        wp_batch_idx,
        wp_pbc,
        wp_cell,
        wp_mz,
        wp_mz2,
        wp_qtotal,
        wp_dtype,
    )
    slab_precompute_geometry(
        wp_pbc,
        wp_cell,
        wp_slab_axis,
        wp_slab_normal,
        wp_slab_volume,
        wp_slab_height_sq,
        wp_dtype,
    )
    slab_correction_double_backward(
        wp_positions,
        wp_charges,
        wp_h_positions,
        wp_h_charges,
        wp_h_cell,
        wp_batch_idx,
        wp_pbc,
        wp_cell,
        wp_mz,
        wp_mz2,
        wp_qtotal,
        wp_slab_axis,
        wp_slab_normal,
        wp_slab_volume,
        wp_slab_height_sq,
        wp_grad_system,
        wp_dmz,
        wp_dmz2,
        wp_dqtotal,
        wp_dnormal,
        wp_dvolume,
        wp_dheight_sq,
        wp_grad_normal,
        wp_h_grad_normal,
        wp_grad_positions,
        wp_grad_charges,
        wp_grad_cell,
        wp_dtype,
    )
    wp.synchronize()
    return (
        wp_grad_positions.numpy(),
        wp_grad_charges.numpy(),
        wp_grad_cell.numpy(),
        wp_dmz.numpy(),
        wp_dmz2.numpy(),
        wp_dqtotal.numpy(),
    )


def _make_hvp_case(axis):
    """Build a small slab HVP case with nonzero position, charge, and cell terms."""
    positions = np.array(
        [[0.3, 1.2, 2.1], [2.0, 0.7, 4.5], [1.3, 2.2, 0.9]],
        dtype=np.float64,
    )
    charges = np.array([0.7, -1.2, 0.4], dtype=np.float64)
    cells = np.array(
        [
            [[24.0, 0.3, 0.5], [0.7, 8.0, 0.1], [0.2, 0.5, 7.0]],
            [[8.0, 0.5, 0.1], [0.4, 26.0, 0.2], [1.0, 0.3, 7.5]],
            [[8.0, 0.1, 0.2], [1.0, 7.0, 0.4], [0.3, 0.2, 21.0]],
        ],
        dtype=np.float64,
    )
    h_positions = np.array(
        [[0.2, -0.1, 0.3], [-0.4, 0.5, -0.2], [0.1, 0.2, -0.3]],
        dtype=np.float64,
    )
    h_charges = np.array([0.3, -0.2, 0.5], dtype=np.float64)
    h_cell = np.array(
        [
            [0.03, -0.02, 0.01],
            [-0.01, 0.04, -0.03],
            [0.02, 0.01, -0.02],
        ],
        dtype=np.float64,
    )
    return (
        positions,
        charges,
        cells[axis][None, :, :],
        h_positions,
        h_charges,
        h_cell[None, :, :],
        np.zeros(positions.shape[0], dtype=np.int32),
        _axis_to_pbc(axis)[None, :],
        np.array([1.25], dtype=np.float64),
    )


class TestAnalyticSlabHvp:
    """Analytic slab HVP kernels match finite differences of first backward."""

    @pytest.mark.parametrize("wp_dtype", [wp.float32, wp.float64])
    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_hvp_matches_backward_finite_difference(self, axis, wp_dtype, device):
        """Position, charge, cell, and moment HVP terms match finite difference."""
        (
            positions,
            charges,
            cell,
            h_positions,
            h_charges,
            h_cell,
            batch_idx,
            pbc,
            grad_system,
        ) = _make_hvp_case(axis)
        eps = 5.0e-3 if wp_dtype == wp.float32 else 1.0e-6

        actual = _run_double_backward_np(
            positions,
            charges,
            cell,
            h_positions,
            h_charges,
            h_cell,
            batch_idx,
            pbc,
            grad_system,
            wp_dtype,
            device,
        )
        plus = _run_backward_np(
            positions + eps * h_positions,
            charges + eps * h_charges,
            cell + eps * h_cell,
            batch_idx,
            pbc,
            grad_system,
            wp_dtype,
            device,
        )
        minus = _run_backward_np(
            positions - eps * h_positions,
            charges - eps * h_charges,
            cell - eps * h_cell,
            batch_idx,
            pbc,
            grad_system,
            wp_dtype,
            device,
        )
        expected = tuple((p - m) / (2.0 * eps) for p, m in zip(plus, minus))

        rtol = 8.0e-3 if wp_dtype == wp.float32 else 3.0e-7
        atol = 8.0e-5 if wp_dtype == wp.float32 else 3.0e-9
        for actual_part, expected_part in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                actual_part,
                expected_part,
                rtol=rtol,
                atol=atol,
            )

    def test_batched_hvp_skips_3d_pbc_system(self, device):
        """Batched HVP handles mixed slab axes and leaves 3D rows at zero."""
        case0 = _make_hvp_case(2)
        case1 = _make_hvp_case(1)
        positions = np.concatenate([case0[0], case1[0] + 0.4], axis=0)
        charges = np.concatenate([case0[1], case1[1]], axis=0)
        cell = np.concatenate([case0[2], case1[2], case0[2] * 0.8], axis=0)
        h_positions = np.concatenate([case0[3], -case1[3]], axis=0)
        h_charges = np.concatenate([case0[4], -case1[4]], axis=0)
        h_cell = np.concatenate([case0[5], case1[5], case0[5]], axis=0)
        batch_idx = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        pbc = np.array(
            [[True, True, False], [True, False, True], [True, True, True]],
            dtype=np.bool_,
        )
        grad_system = np.array([0.7, -1.1, 2.0], dtype=np.float64)

        actual = _run_double_backward_np(
            positions,
            charges,
            cell,
            h_positions,
            h_charges,
            h_cell,
            batch_idx,
            pbc,
            grad_system,
            wp.float64,
            device,
        )
        eps = 1.0e-6
        plus = _run_backward_np(
            positions + eps * h_positions,
            charges + eps * h_charges,
            cell + eps * h_cell,
            batch_idx,
            pbc,
            grad_system,
            wp.float64,
            device,
        )
        minus = _run_backward_np(
            positions - eps * h_positions,
            charges - eps * h_charges,
            cell - eps * h_cell,
            batch_idx,
            pbc,
            grad_system,
            wp.float64,
            device,
        )
        expected = tuple((p - m) / (2.0 * eps) for p, m in zip(plus, minus))
        for actual_part, expected_part in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                actual_part,
                expected_part,
                rtol=3.0e-7,
                atol=3.0e-9,
            )
        np.testing.assert_allclose(actual[2][2], 0.0, rtol=0, atol=0)
        np.testing.assert_allclose(actual[3][2], 0.0, rtol=0, atol=0)
        np.testing.assert_allclose(actual[4][2], 0.0, rtol=0, atol=0)
        np.testing.assert_allclose(actual[5][2], 0.0, rtol=0, atol=0)

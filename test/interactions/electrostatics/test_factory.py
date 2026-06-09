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

"""Tests for shared electrostatics factory validation and sentinels.

Two guarantees are checked:

1. **Bit-exact parity** -- the factory ``ewald_real`` energy kernel produces the
   exact same per-atom float64 energy array as the hand-written launchers, for
   ``wp.float32`` and ``wp.float64``, single and batched, CSR (list) and
   neighbor-matrix inputs.
2. **Dead-branch elimination** -- a Python compile-time axis (``BATCHED`` /
   ``NEIGHBOR_MATRIX``) that is ``False`` produces generated source with the
   unused branch removed. This is the same codegen mechanism the ``deriv_state``
   branches rely on.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pytest
import warp as wp

from nvalchemiops.interactions.electrostatics._factory_common import (
    _DerivState,
    get_backward_scale_kernel,
)
from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    batch_ewald_real_space_energy,
    batch_ewald_real_space_energy_matrix,
    ewald_real_space_energy,
    ewald_real_space_energy_matrix,
)
from nvalchemiops.interactions.electrostatics.ewald_real_factory import (
    _ewald_real_module_name,
    alloc_ewald_real_sentinels,
    get_ewald_real_kernel,
    make_ewald_real_kernel,
)

# Reuse the warp-array prep helpers from the kernel test module so the factory is
# exercised against the exact same input encoding the hand-written launchers use.
# The system fixtures are defined locally below (importing fixtures across test
# modules shadows the import name); the prep helpers are plain functions.
from test.interactions.electrostatics.test_ewald_kernels import (
    make_alpha_array,
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


@pytest.fixture(scope="session")
def two_atom_system():
    """Two opposite charges along x at r=3, large cell, CSR half neighbor list."""
    return {
        "positions": np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64),
        "charges": np.array([1.0, -1.0], dtype=np.float64),
        "cell": np.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        ),
        "idx_j": np.array([1], dtype=np.int32),
        "neighbor_ptr": np.array([0, 1, 1], dtype=np.int32),
        "unit_shifts": np.array([[0, 0, 0]], dtype=np.int32),
        "num_atoms": 2,
    }


@pytest.fixture(scope="session")
def two_atom_matrix_system():
    """Two-atom system in neighbor-matrix format (full neighbor list)."""
    return {
        "positions": np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64),
        "charges": np.array([1.0, -1.0], dtype=np.float64),
        "cell": np.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        ),
        "neighbor_matrix": np.array([[1, _MASK], [0, _MASK]], dtype=np.int32),
        "neighbor_shifts": np.array(
            [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]], dtype=np.int32
        ),
        "fill_value": _MASK,
        "num_atoms": 2,
    }


@pytest.fixture(scope="session")
def batch_two_systems():
    """Two independent two-atom systems, CSR half neighbor list."""
    return {
        "positions": np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        "charges": np.array([1.0, -1.0, 2.0, -1.0], dtype=np.float64),
        "cell": np.array(
            [
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            ],
            dtype=np.float64,
        ),
        "idx_j": np.array([1, 3], dtype=np.int32),
        "neighbor_ptr": np.array([0, 1, 1, 2, 2], dtype=np.int32),
        "unit_shifts": np.zeros((2, 3), dtype=np.int32),
        "batch_idx": np.array([0, 0, 1, 1], dtype=np.int32),
        "num_atoms": 4,
    }


@pytest.fixture(scope="session")
def batch_two_systems_matrix():
    """Two independent two-atom systems, neighbor-matrix format."""
    return {
        "positions": np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        "charges": np.array([1.0, -1.0, 2.0, -1.0], dtype=np.float64),
        "cell": np.array(
            [
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            ],
            dtype=np.float64,
        ),
        "neighbor_matrix": np.array(
            [[1, _MASK], [0, _MASK], [3, _MASK], [2, _MASK]], dtype=np.int32
        ),
        "neighbor_shifts": np.zeros((4, 2, 3), dtype=np.int32),
        "batch_idx": np.array([0, 0, 1, 1], dtype=np.int32),
        "fill_value": _MASK,
        "num_atoms": 4,
    }


def _launch_factory_energy(
    *,
    wp_dtype,
    batched,
    neighbor_input,
    device,
    positions,
    charges,
    cell,
    alpha,
    num_atoms,
    batch_id=None,
    idx_j=None,
    neighbor_ptr=None,
    unit_shifts=None,
    neighbor_matrix=None,
    unit_shifts_matrix=None,
):
    """Launch the factory energy kernel, filling inactive slots with sentinels."""
    out = wp.zeros(num_atoms, dtype=wp.float64, device=device)
    s = alloc_ewald_real_sentinels(wp_dtype, device)
    kernel = get_ewald_real_kernel(
        wp_dtype,
        batched=batched,
        neighbor_input=neighbor_input,
        deriv_state=_DerivState.E,
    )
    wp.launch(
        kernel,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cell,
            batch_id if batch_id is not None else s["batch_id"],
            idx_j if idx_j is not None else s["idx_j"],
            neighbor_ptr if neighbor_ptr is not None else s["neighbor_ptr"],
            unit_shifts if unit_shifts is not None else s["unit_shifts"],
            neighbor_matrix if neighbor_matrix is not None else s["neighbor_matrix"],
            unit_shifts_matrix
            if unit_shifts_matrix is not None
            else s["unit_shifts_matrix"],
            wp.int32(_MASK),
            alpha,
            out,
            s["atomic_forces"],
            s["charge_gradients"],
            s["virial"],
        ],
        device=device,
    )
    wp.synchronize()
    return out.numpy()


# ==============================================================================
# Bit-exact parity: factory energy kernel == hand-written energy launcher
# ==============================================================================


class TestFactoryEnergyParity:
    """Factory ``ewald_real`` energy kernel must match the hand-written kernels."""

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    def test_csr_single(self, device, two_atom_system, dtype):
        inputs = prepare_csr_inputs(two_atom_system, device, dtype=dtype)
        alpha = make_alpha_array(_ALPHA, device, dtype=dtype)
        n = two_atom_system["num_atoms"]

        ref = wp.zeros(n, dtype=wp.float64, device=device)
        ewald_real_space_energy(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            alpha=alpha,
            pair_energies=ref,
            wp_dtype=dtype,
            device=device,
        )

        got = _launch_factory_energy(
            wp_dtype=dtype,
            batched=False,
            neighbor_input="list",
            device=device,
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            alpha=alpha,
            num_atoms=n,
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
        )
        assert np.array_equal(got, ref.numpy())

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    def test_matrix_single(self, device, two_atom_matrix_system, dtype):
        inputs = prepare_matrix_inputs(two_atom_matrix_system, device, dtype=dtype)
        alpha = make_alpha_array(_ALPHA, device, dtype=dtype)
        n = two_atom_matrix_system["num_atoms"]
        mask = two_atom_matrix_system["fill_value"]

        ref = wp.zeros(n, dtype=wp.float64, device=device)
        ewald_real_space_energy_matrix(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            neighbor_matrix=inputs["neighbor_matrix"],
            unit_shifts_matrix=inputs["neighbor_shifts"],
            mask_value=mask,
            alpha=alpha,
            pair_energies=ref,
            wp_dtype=dtype,
            device=device,
        )

        # The factory uses a fixed sentinel mask; reuse the fixture's fill_value.
        out = wp.zeros(n, dtype=wp.float64, device=device)
        s = alloc_ewald_real_sentinels(dtype, device)
        kernel = get_ewald_real_kernel(
            dtype, batched=False, neighbor_input="matrix", deriv_state=_DerivState.E
        )
        wp.launch(
            kernel,
            dim=n,
            inputs=[
                inputs["positions"],
                inputs["charges"],
                inputs["cell"],
                s["batch_id"],
                s["idx_j"],
                s["neighbor_ptr"],
                s["unit_shifts"],
                inputs["neighbor_matrix"],
                inputs["neighbor_shifts"],
                wp.int32(mask),
                alpha,
                out,
                s["atomic_forces"],
                s["charge_gradients"],
                s["virial"],
            ],
            device=device,
        )
        wp.synchronize()
        assert np.array_equal(out.numpy(), ref.numpy())

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    def test_csr_batch(self, device, batch_two_systems, dtype):
        inputs = prepare_csr_inputs(batch_two_systems, device, dtype=dtype)
        batch_id = wp.from_numpy(
            batch_two_systems["batch_idx"], dtype=wp.int32, device=device
        )
        # Per-system alpha (B,)
        npf = np.float64 if dtype == wp.float64 else np.float32
        alpha = wp.from_numpy(
            np.array([_ALPHA, _ALPHA], dtype=npf), dtype=dtype, device=device
        )
        n = batch_two_systems["num_atoms"]

        ref = wp.zeros(n, dtype=wp.float64, device=device)
        batch_ewald_real_space_energy(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            batch_id=batch_id,
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            alpha=alpha,
            pair_energies=ref,
            wp_dtype=dtype,
            device=device,
        )

        got = _launch_factory_energy(
            wp_dtype=dtype,
            batched=True,
            neighbor_input="list",
            device=device,
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            alpha=alpha,
            num_atoms=n,
            batch_id=batch_id,
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
        )
        assert np.array_equal(got, ref.numpy())

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    def test_matrix_batch(self, device, batch_two_systems_matrix, dtype):
        inputs = prepare_matrix_inputs(batch_two_systems_matrix, device, dtype=dtype)
        batch_id = wp.from_numpy(
            batch_two_systems_matrix["batch_idx"], dtype=wp.int32, device=device
        )
        npf = np.float64 if dtype == wp.float64 else np.float32
        alpha = wp.from_numpy(
            np.array([_ALPHA, _ALPHA], dtype=npf), dtype=dtype, device=device
        )
        n = batch_two_systems_matrix["num_atoms"]
        mask = batch_two_systems_matrix["fill_value"]

        ref = wp.zeros(n, dtype=wp.float64, device=device)
        batch_ewald_real_space_energy_matrix(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            batch_id=batch_id,
            neighbor_matrix=inputs["neighbor_matrix"],
            unit_shifts_matrix=inputs["neighbor_shifts"],
            mask_value=mask,
            alpha=alpha,
            pair_energies=ref,
            wp_dtype=dtype,
            device=device,
        )

        out = wp.zeros(n, dtype=wp.float64, device=device)
        s = alloc_ewald_real_sentinels(dtype, device)
        kernel = get_ewald_real_kernel(
            dtype, batched=True, neighbor_input="matrix", deriv_state=_DerivState.E
        )
        wp.launch(
            kernel,
            dim=n,
            inputs=[
                inputs["positions"],
                inputs["charges"],
                inputs["cell"],
                batch_id,
                s["idx_j"],
                s["neighbor_ptr"],
                s["unit_shifts"],
                inputs["neighbor_matrix"],
                inputs["neighbor_shifts"],
                wp.int32(mask),
                alpha,
                out,
                s["atomic_forces"],
                s["charge_gradients"],
                s["virial"],
            ],
            device=device,
        )
        wp.synchronize()
        assert np.array_equal(out.numpy(), ref.numpy())


class TestFactoryCudaSentinels:
    """CUDA canaries for inactive factory slots backed by zero-size sentinels."""

    @staticmethod
    def _cuda_device() -> str:
        if not wp.is_cuda_available():
            pytest.skip("CUDA not available")
        return "cuda:0"

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    def test_matrix_energy_ignores_csr_and_derivative_sentinels(
        self,
        two_atom_matrix_system,
        dtype,
    ):
        """Matrix energy specialization does not touch inactive CSR/output slots."""
        device = self._cuda_device()
        inputs = prepare_matrix_inputs(two_atom_matrix_system, device, dtype=dtype)
        alpha = make_alpha_array(_ALPHA, device, dtype=dtype)
        sentinels = alloc_ewald_real_sentinels(dtype, device)
        n = two_atom_matrix_system["num_atoms"]

        out = wp.zeros(n, dtype=wp.float64, device=device)
        kernel = get_ewald_real_kernel(
            dtype,
            batched=False,
            neighbor_input="matrix",
            deriv_state=_DerivState.E,
        )
        wp.launch(
            kernel,
            dim=n,
            inputs=[
                inputs["positions"],
                inputs["charges"],
                inputs["cell"],
                sentinels["batch_id"],
                sentinels["idx_j"],
                sentinels["neighbor_ptr"],
                sentinels["unit_shifts"],
                inputs["neighbor_matrix"],
                inputs["neighbor_shifts"],
                wp.int32(two_atom_matrix_system["fill_value"]),
                alpha,
                out,
                sentinels["atomic_forces"],
                sentinels["charge_gradients"],
                sentinels["virial"],
            ],
            device=device,
        )
        wp.synchronize()
        assert np.isfinite(out.numpy()).all()

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    def test_csr_force_charge_ignores_matrix_and_virial_sentinels(
        self,
        two_atom_system,
        dtype,
    ):
        """CSR E/F/dQ specialization does not touch inactive matrix/virial slots."""
        device = self._cuda_device()
        inputs = prepare_csr_inputs(two_atom_system, device, dtype=dtype)
        alpha = make_alpha_array(_ALPHA, device, dtype=dtype)
        sentinels = alloc_ewald_real_sentinels(dtype, device)
        n = two_atom_system["num_atoms"]

        out = wp.zeros(n, dtype=wp.float64, device=device)
        forces = wp.zeros(n, dtype=_VEC[dtype], device=device)
        charge_grads = wp.zeros(n, dtype=wp.float64, device=device)
        kernel = get_ewald_real_kernel(
            dtype,
            batched=False,
            neighbor_input="list",
            deriv_state=_DerivState.E_F_dQ,
            cell_grad=False,
        )
        wp.launch(
            kernel,
            dim=n,
            inputs=[
                inputs["positions"],
                inputs["charges"],
                inputs["cell"],
                sentinels["batch_id"],
                inputs["idx_j"],
                inputs["neighbor_ptr"],
                inputs["unit_shifts"],
                sentinels["neighbor_matrix"],
                sentinels["unit_shifts_matrix"],
                wp.int32(_MASK),
                alpha,
                out,
                forces,
                charge_grads,
                sentinels["virial"],
            ],
            device=device,
        )
        wp.synchronize()
        assert np.isfinite(out.numpy()).all()
        assert np.isfinite(forces.numpy()).all()
        assert np.isfinite(charge_grads.numpy()).all()


# ==============================================================================
# Cache + axis validation
# ==============================================================================


class TestFactoryCacheAndValidation:
    def test_cache_returns_same_object(self):
        a = get_ewald_real_kernel(wp.float64, batched=False, neighbor_input="list")
        b = get_ewald_real_kernel(wp.float64, batched=False, neighbor_input="list")
        assert a is b

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError):
            get_ewald_real_kernel(wp.float16)

    def test_derivative_axes_now_supported(self):
        # Force-bearing derivative states support the first- and second-order
        # factory kernels.
        assert make_ewald_real_kernel(wp.float64, deriv_state=_DerivState.E_F)
        assert make_ewald_real_kernel(
            wp.float64, deriv_state=_DerivState.E_F, order="backward"
        )
        assert make_ewald_real_kernel(
            wp.float64, deriv_state=_DerivState.E_F, order="double_backward"
        )
        assert make_ewald_real_kernel(
            wp.float64, deriv_state=_DerivState.E_F, cell_grad=True
        )

    def test_unsupported_order_raises(self):
        with pytest.raises(NotImplementedError):
            make_ewald_real_kernel(wp.float64, order="bogus")

    def test_cell_grad_with_energy_only_raises_value_error(self):
        # Permanently invalid combination: no force terms to sum into the virial.
        with pytest.raises(ValueError):
            make_ewald_real_kernel(
                wp.float64, cell_grad=True, deriv_state=_DerivState.E
            )

    @pytest.mark.parametrize("deriv_state", [_DerivState.E, _DerivState.E_dQ])
    @pytest.mark.parametrize("order", ["backward", "double_backward"])
    def test_derivative_order_without_force_terms_raises_value_error(
        self, deriv_state, order
    ):
        # backward / double_backward need force-bearing factory roots.
        with pytest.raises(ValueError, match="E_F, E_F_dQ"):
            make_ewald_real_kernel(wp.float64, order=order, deriv_state=deriv_state)

    def test_unsupported_component_raises(self):
        with pytest.raises(NotImplementedError):
            get_ewald_real_kernel(wp.float64, component="ewald_recip")


class TestBackwardScaleKernel:
    """Shared first-backward scale kernel matches the cache-scaling contract."""

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
    def test_positions_and_charges(self, device, dtype, batched):
        """Scale atom-major ``dE/dR`` / ``dE/dq`` by per-system cotangents."""
        num_atoms = 4
        dEdR = np.array(
            [
                [0.1, -0.2, 0.3],
                [-0.4, 0.5, -0.6],
                [0.7, 0.8, -0.9],
                [-1.0, 1.1, 1.2],
            ],
            dtype=_NPF[dtype],
        )
        dEdq = np.array([0.25, -0.5, 0.75, -1.0], dtype=np.float64)
        if batched:
            grad_energy = np.array([1.5, -0.25], dtype=np.float64)
            batch_idx = np.array([0, 0, 1, 1], dtype=np.int32)
            atom_scale = grad_energy[batch_idx]
        else:
            grad_energy = np.array([1.5], dtype=np.float64)
            batch_idx = np.empty((0,), dtype=np.int32)
            atom_scale = np.full(num_atoms, grad_energy[0], dtype=np.float64)

        out_pos = wp.zeros(num_atoms, dtype=_VEC[dtype], device=device)
        out_q = wp.zeros(num_atoms, dtype=wp.float64, device=device)
        kernel = get_backward_scale_kernel(
            dtype,
            batched=batched,
            scale_positions=True,
            scale_charges=True,
        )
        wp.launch(
            kernel,
            dim=num_atoms,
            inputs=[
                wp.from_numpy(grad_energy, dtype=wp.float64, device=device),
                wp.from_numpy(batch_idx, dtype=wp.int32, device=device),
                wp.from_numpy(dEdR, dtype=_VEC[dtype], device=device),
                wp.from_numpy(dEdq, dtype=wp.float64, device=device),
                out_pos,
                out_q,
                wp.int32(num_atoms),
            ],
            device=device,
        )
        wp.synchronize()

        np.testing.assert_allclose(
            out_pos.numpy(), (atom_scale[:, None] * dEdR).astype(_NPF[dtype])
        )
        np.testing.assert_allclose(out_q.numpy(), atom_scale * dEdq)


# ==============================================================================
# Named + documented generated kernels (R1)
# ==============================================================================


class TestFactoryKernelNames:
    """Generated kernels carry a descriptive name + a "Specialization" docstring.

    R1 mirrors the neighbor-list naming convention: each specialization gets a
    descriptive ``__name__`` / Warp ``key`` encoding its specialization axes, plus a contract
    docstring extended with a "Specialization" section. ``module.name`` (which the
    dead-branch test keys on) stays the per-spec ``module=`` and is unaffected.
    """

    def test_name_encodes_axes(self):
        kernel = make_ewald_real_kernel(
            wp.float64,
            batched=True,
            neighbor_input="matrix",
            deriv_state=_DerivState.E_F_dQ,
            cell_grad=True,
            order="forward",
        )
        name = kernel.__name__
        # base + every axis token + dtype are all present in the name.
        for token in (
            "ewald_real_forward",
            "e_f_dq",
            "cellgrad",
            "batch",
            "matrix",
            "f64",
        ):
            assert token in name, f"{token!r} missing from {name!r}"
        # The Warp identity key is set to the same descriptive name.
        assert kernel.key == name
        # The per-spec module name (dead-branch test anchor) is left untouched.
        assert kernel.module.name == "ewald_real_f64_batch_matrix"

    def test_f32_single_list_name(self):
        kernel = make_ewald_real_kernel(
            wp.float32,
            batched=False,
            neighbor_input="list",
            deriv_state=_DerivState.E,
            order="forward",
        )
        name = kernel.__name__
        for token in ("ewald_real_forward", "single", "list", "f32"):
            assert token in name, f"{token!r} missing from {name!r}"
        assert "f64" not in name

    def test_docstring_has_specialization_section(self):
        kernel = make_ewald_real_kernel(
            wp.float64,
            batched=False,
            neighbor_input="list",
            deriv_state=_DerivState.E_F,
            order="backward",
        )
        doc = kernel.__doc__
        assert "Specialization" in doc
        # The section lists the specialization axes with their values.
        assert "dtype = f64" in doc
        assert "neighbor_input = list" in doc
        assert "deriv_state = E_F" in doc
        assert "order = backward" in doc

    def test_backward_double_backward_names_differ(self):
        bwd = make_ewald_real_kernel(
            wp.float64, deriv_state=_DerivState.E_F, order="backward"
        )
        dbwd = make_ewald_real_kernel(
            wp.float64, deriv_state=_DerivState.E_F, order="double_backward"
        )
        assert bwd.__name__ != dbwd.__name__
        assert "backward" in bwd.__name__
        assert "double_backward" in dbwd.__name__


# ==============================================================================
# Dead-branch elimination
# ==============================================================================


def _generated_source_for(kernel: wp.Kernel) -> str:
    """Compile ``kernel`` for CPU and return its generated C++ source.

    Each specialization is built with a deterministic ``module=`` name, so the
    cache directory is located by name (not mtime-newest), making the check
    robust to concurrent compilations elsewhere in the suite.
    """
    module = kernel.module
    module.load("cpu")
    cache_dir = wp.config.kernel_cache_dir
    # Warp names a spec's cache dir ``wp_<module_name>_<hexhash>``. Match only dirs
    # whose trailing segment is a bare hex hash, so a base module name (e.g.
    # ``ewald_real_f64_single_list``) does not also pick up the derivative-order
    # specializations that append ``_backward`` / ``_double_backward`` to the name.
    prefix = f"wp_{module.name}_"
    matches = [
        p
        for p in glob.glob(os.path.join(cache_dir, f"{prefix}*"))
        if re.fullmatch(r"[0-9a-fA-F]+", os.path.basename(p)[len(prefix) :])
    ]
    assert matches, f"no cache dir for module {module.name!r} under {cache_dir}"
    # Several content-hash dirs can share one module name: the energy-only and the
    # E_F / E_F_dQ forward specializations all build into this module, and CPU vs
    # GPU builds emit different artifacts (CPU -> ``.cpp``, GPU -> ``.cu`` / ``.ptx``).
    # Pick the newest dir that actually holds a generated ``.cpp`` (CPU source).
    matches.sort(key=os.path.getmtime, reverse=True)
    cpp = None
    for d in matches:
        found = glob.glob(os.path.join(d, "*.cpp"))
        if found:
            cpp = found
            break
    assert cpp, f"no generated .cpp under any {module.name!r} cache dir"
    with open(cpp[0]) as fh:
        return fh.read()


def _body_reads(src: str, param: str) -> int:
    """Count actual in-body array reads of ``param`` in generated Warp source.

    Every kernel parameter appears in the argument struct + load preamble + the
    (unused) adjoint regardless of dead-code elimination, so a substring check on
    the bare name is not a reliable signal. Warp emits ``wp::address(var_<name>,
    ...)`` only where the array is actually indexed in the live body; this count
    drops to zero when the owning branch is dead-eliminated.

    The trailing comma in the match pattern keeps ``unit_shifts`` from also
    matching ``unit_shifts_matrix`` reads.

    Note: the ``wp::address(var_*`` marker is coupled to the Warp codegen output
    format (validated against warp 1.14). If a future Warp version changes the
    generated C++ symbol convention, update this marker.
    """
    return src.count(f"wp::address(var_{param},")


class TestDeadBranchElimination:
    """A ``False`` compile-time axis must remove its branch from generated code.

    This is the same compile-time-constant mechanism the ``deriv_state`` branches
    rely on: flipping a Python constant drops the corresponding
    kernel body. The check inspects in-body array reads (``wp::address(var_*``),
    not the argument-struct boilerplate, which is always present.
    """

    def test_single_csr_drops_batched_and_matrix_branches(self):
        # Single-system CSR: BATCHED=False, NEIGHBOR_MATRIX=False.
        kernel = make_ewald_real_kernel(
            wp.float64, batched=False, neighbor_input="list"
        )
        src = _generated_source_for(kernel)
        # Dead branches: batched (batch_id) and neighbor-matrix path.
        assert _body_reads(src, "batch_id") == 0
        assert _body_reads(src, "neighbor_matrix") == 0
        assert _body_reads(src, "unit_shifts_matrix") == 0
        # Active CSR path keeps its neighbor-list traversal.
        assert _body_reads(src, "neighbor_ptr") > 0
        assert _body_reads(src, "idx_j") > 0

    def test_batched_matrix_keeps_those_branches(self):
        # Batched neighbor-matrix: BATCHED=True, NEIGHBOR_MATRIX=True.
        kernel = make_ewald_real_kernel(
            wp.float64, batched=True, neighbor_input="matrix"
        )
        src = _generated_source_for(kernel)
        # Active branches.
        assert _body_reads(src, "batch_id") > 0
        assert _body_reads(src, "neighbor_matrix") > 0
        assert _body_reads(src, "unit_shifts_matrix") > 0
        # Dead CSR-only path.
        assert _body_reads(src, "neighbor_ptr") == 0
        assert _body_reads(src, "idx_j") == 0

    def test_module_names_are_distinct_per_spec(self):
        assert _ewald_real_module_name(wp.float64, False, "list") != (
            _ewald_real_module_name(wp.float64, True, "matrix")
        )

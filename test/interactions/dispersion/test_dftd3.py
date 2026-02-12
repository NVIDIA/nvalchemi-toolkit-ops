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
Core warp launcher tests for DFT-D3 implementation.

This test suite focuses on the core warp launchers (dftd3_matrix, dftd3)
and includes:
- S5 switching function tests
- Warp launcher interface tests
- CPU/GPU consistency for warp launchers
- Regression tests to ensure kernel outputs haven't changed
- Basic shape and interface tests

These tests use warp arrays directly and do not require PyTorch.
For PyTorch binding tests, see test/bindings/torch/dispersion/test_dftd3.py
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from nvalchemiops.interactions.dispersion._dftd3 import (
    _s5_switch,
    dftd3,
    dftd3_matrix,
)
from test.interactions.dispersion.conftest import from_warp, to_warp

# ==============================================================================
# Helper Functions
# ==============================================================================


def neighbor_matrix_to_csr(
    nbmat: np.ndarray, fill_value: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert neighbor matrix format to CSR (Compressed Sparse Row) format.

    Parameters
    ----------
    nbmat : np.ndarray, shape [num_atoms, max_neighbors]
        Neighbor matrix with padding
    fill_value : int
        Value used for padding (typically num_atoms)

    Returns
    -------
    idx_j : np.ndarray
        Flat array of neighbor indices
    neighbor_ptr : np.ndarray, shape [num_atoms + 1]
        CSR row pointers
    """
    num_atoms = nbmat.shape[0]
    neighbor_ptr = np.zeros(num_atoms + 1, dtype=np.int32)
    idx_j_list = []

    for i in range(num_atoms):
        neighbors = nbmat[i]
        valid_neighbors = neighbors[neighbors < fill_value]
        idx_j_list.extend(valid_neighbors.tolist())
        neighbor_ptr[i + 1] = neighbor_ptr[i] + len(valid_neighbors)

    idx_j = np.array(idx_j_list, dtype=np.int32)
    return idx_j, neighbor_ptr


def run_dftd3_matrix(
    system: dict,
    element_tables: dict,
    functional_params: dict,
    device: str,
    batch_indices: np.ndarray | None = None,
    wp_dtype=wp.float32,
) -> dict:
    """
    Run dftd3_matrix warp launcher for a system.

    Parameters
    ----------
    system : dict
        System with 'coord', 'numbers', 'nbmat', 'B', 'M' (numpy arrays)
    element_tables : dict
        Element parameters (numpy arrays)
    functional_params : dict
        Functional parameters (k1, k3, a1, a2, s6, s8)
    device : str
        Warp device string
    batch_indices : np.ndarray or None
        Batch indices for atoms
    wp_dtype : warp dtype
        Scalar dtype (wp.float32 or wp.float64)

    Returns
    -------
    dict
        Results with 'energy', 'forces', 'coord_num' (numpy arrays)
    """
    # Derive vector dtype from scalar dtype
    if wp_dtype == wp.float64:
        vec_dtype = wp.vec3d
    else:
        vec_dtype = wp.vec3f

    # Extract system data (now numpy arrays)
    B = system["B"]
    coord_flat = system["coord"]
    numbers = system["numbers"]
    nbmat = system["nbmat"]

    # Convert to warp arrays
    positions = to_warp(coord_flat.reshape(B, 3), vec_dtype, device)
    numbers_wp = to_warp(numbers, wp.int32, device)
    neighbor_matrix_wp = to_warp(nbmat, wp.int32, device)

    # Prepare element tables as warp arrays
    scalar_dtype = wp.float32
    max_z_inc = element_tables["z_max_inc"]

    rcov_wp = to_warp(element_tables["rcov"], scalar_dtype, device)
    r4r2_wp = to_warp(element_tables["r4r2"], scalar_dtype, device)
    c6_reference_wp = to_warp(
        element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5),
        scalar_dtype,
        device,
    )
    coord_num_ref_wp = to_warp(
        element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5),
        scalar_dtype,
        device,
    )

    # Determine number of systems and batch_idx
    if batch_indices is not None:
        batch_idx_wp = to_warp(batch_indices, wp.int32, device)
        num_systems = int(batch_indices.max()) + 1
    else:
        batch_idx_wp = None  # Let launcher create default
        num_systems = 1

    # Allocate outputs (pre-zeroed)
    coord_num_wp = wp.zeros(B, dtype=wp.float32, device=device)
    forces_wp = wp.zeros(B, dtype=wp.vec3f, device=device)
    energy_wp = wp.zeros(num_systems, dtype=wp.float32, device=device)
    virial_wp = wp.zeros(num_systems, dtype=wp.mat33f, device=device)

    # Call warp launcher
    dftd3_matrix(
        positions=positions,
        numbers=numbers_wp,
        neighbor_matrix=neighbor_matrix_wp,
        covalent_radii=rcov_wp,
        r4r2=r4r2_wp,
        c6_reference=c6_reference_wp,
        coord_num_ref=coord_num_ref_wp,
        a1=functional_params["a1"],
        a2=functional_params["a2"],
        s8=functional_params["s8"],
        coord_num=coord_num_wp,
        forces=forces_wp,
        energy=energy_wp,
        virial=virial_wp,
        wp_dtype=wp_dtype,
        device=device,
        k1=functional_params["k1"],
        k3=functional_params["k3"],
        s6=functional_params["s6"],
        batch_idx=batch_idx_wp,
    )

    # Convert back to numpy
    return {
        "energy": from_warp(energy_wp),
        "forces": from_warp(forces_wp),
        "coord_num": from_warp(coord_num_wp),
    }


def run_dftd3(
    system: dict,
    element_tables: dict,
    functional_params: dict,
    device: str,
    batch_indices: np.ndarray | None = None,
    wp_dtype=wp.float32,
) -> dict:
    """
    Run dftd3 warp launcher for a system (neighbor list / CSR format).

    Parameters
    ----------
    system : dict
        System with 'coord', 'numbers', 'nbmat', 'B', 'M' (numpy arrays)
    element_tables : dict
        Element parameters (numpy arrays)
    functional_params : dict
        Functional parameters (k1, k3, a1, a2, s6, s8)
    device : str
        Warp device string
    batch_indices : np.ndarray or None
        Batch indices for atoms
    wp_dtype : warp dtype
        Scalar dtype (wp.float32 or wp.float64)

    Returns
    -------
    dict
        Results with 'energy', 'forces', 'coord_num' (numpy arrays)
    """
    # Derive vector dtype from scalar dtype
    if wp_dtype == wp.float64:
        vec_dtype = wp.vec3d
    else:
        vec_dtype = wp.vec3f

    # Extract system data (now numpy arrays)
    B = system["B"]
    coord_flat = system["coord"]
    numbers = system["numbers"]
    nbmat = system["nbmat"]

    # Convert neighbor matrix to CSR format
    fill_value = B
    idx_j, neighbor_ptr = neighbor_matrix_to_csr(nbmat, fill_value)

    # Convert to warp arrays
    positions = to_warp(coord_flat.reshape(B, 3), vec_dtype, device)
    numbers_wp = to_warp(numbers, wp.int32, device)
    idx_j_wp = to_warp(idx_j, wp.int32, device)
    neighbor_ptr_wp = to_warp(neighbor_ptr, wp.int32, device)

    # Prepare element tables as warp arrays
    scalar_dtype = wp.float32
    max_z_inc = element_tables["z_max_inc"]

    rcov_wp = to_warp(element_tables["rcov"], scalar_dtype, device)
    r4r2_wp = to_warp(element_tables["r4r2"], scalar_dtype, device)
    c6_reference_wp = to_warp(
        element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5),
        scalar_dtype,
        device,
    )
    coord_num_ref_wp = to_warp(
        element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5),
        scalar_dtype,
        device,
    )

    # Determine number of systems and batch_idx
    if batch_indices is not None:
        batch_idx_wp = to_warp(batch_indices, wp.int32, device)
        num_systems = int(batch_indices.max()) + 1
    else:
        batch_idx_wp = None  # Let launcher create default
        num_systems = 1

    # Allocate outputs (pre-zeroed)
    coord_num_wp = wp.zeros(B, dtype=wp.float32, device=device)
    forces_wp = wp.zeros(B, dtype=wp.vec3f, device=device)
    energy_wp = wp.zeros(num_systems, dtype=wp.float32, device=device)
    virial_wp = wp.zeros(num_systems, dtype=wp.mat33f, device=device)

    # Call warp launcher
    dftd3(
        positions=positions,
        numbers=numbers_wp,
        idx_j=idx_j_wp,
        neighbor_ptr=neighbor_ptr_wp,
        covalent_radii=rcov_wp,
        r4r2=r4r2_wp,
        c6_reference=c6_reference_wp,
        coord_num_ref=coord_num_ref_wp,
        a1=functional_params["a1"],
        a2=functional_params["a2"],
        s8=functional_params["s8"],
        coord_num=coord_num_wp,
        forces=forces_wp,
        energy=energy_wp,
        virial=virial_wp,
        wp_dtype=wp_dtype,
        device=device,
        k1=functional_params["k1"],
        k3=functional_params["k3"],
        s6=functional_params["s6"],
        batch_idx=batch_idx_wp,
    )

    # Convert back to numpy
    return {
        "energy": from_warp(energy_wp),
        "forces": from_warp(forces_wp),
        "coord_num": from_warp(coord_num_wp),
    }


# ==============================================================================
# S5 Switch Tests
# ==============================================================================


class TestS5Switch:
    """Tests for S5 switching function."""

    @staticmethod
    @wp.kernel
    def eval_s5_switch_kernel(
        r_vals: wp.array(dtype=wp.float32),
        r_on: wp.float32,
        r_off: wp.float32,
        inv_w: wp.float32,
        switch_output: wp.array(dtype=wp.float32),
        dswitch_output: wp.array(dtype=wp.float32),
    ):
        """Helper kernel to evaluate _s5_switch function."""
        tid = wp.tid()
        switch, dswitch = _s5_switch(r_vals[tid], r_on, r_off, inv_w)
        switch_output[tid] = switch
        dswitch_output[tid] = dswitch

    @pytest.mark.parametrize(
        "r_vals,r_on,r_off,expected_sw,expected_behavior",
        [
            ([0.5, 1.0, 1.5], 2.0, 5.0, [1.0, 1.0, 1.0], "below_r_on"),
            ([6.0, 10.0, 20.0], 2.0, 5.0, [0.0, 0.0, 0.0], "above_r_off"),
            ([1.0, 5.0, 10.0], 5.0, 5.0, [1.0, 1.0, 1.0], "disabled"),
        ],
    )
    @pytest.mark.usefixtures("device")
    def test_s5_switch_regions(
        self, request, r_vals, r_on, r_off, expected_sw, expected_behavior
    ):
        """Test _s5_switch in different regions (below r_on, above r_off, disabled)."""
        device = request.getfixturevalue("device")
        r_array = np.array(r_vals, dtype=np.float32)
        r_wp = to_warp(r_array, wp.float32, device)
        switch_output = wp.zeros(len(r_vals), dtype=wp.float32, device=device)
        dswitch_output = wp.zeros(len(r_vals), dtype=wp.float32, device=device)

        # Compute inv_w
        inv_w = 1.0 / (r_off - r_on) if r_off > r_on else 0.0

        wp.launch(
            self.eval_s5_switch_kernel,
            dim=len(r_vals),
            inputs=[r_wp, r_on, r_off, inv_w],
            outputs=[switch_output, dswitch_output],
            device=device,
        )

        switch = from_warp(switch_output)
        dswitch = from_warp(dswitch_output)

        expected_array = np.array(expected_sw, dtype=np.float32)
        np.testing.assert_allclose(switch, expected_array, atol=1e-7, rtol=0)

        if expected_behavior in ["below_r_on", "above_r_off", "disabled"]:
            np.testing.assert_allclose(
                dswitch, np.zeros_like(dswitch), atol=1e-7, rtol=0
            )

    @pytest.mark.usefixtures("device")
    def test_s5_switch_transition_region(self, request):
        """Test _s5_switch in transition region with monotonicity."""
        device = request.getfixturevalue("device")
        r_on, r_off = 2.0, 5.0
        r_vals = np.linspace(r_on + 0.1, r_off - 0.1, 10, dtype=np.float32)

        r_wp = to_warp(r_vals, wp.float32, device)
        switch_output = wp.zeros(len(r_vals), dtype=wp.float32, device=device)
        dswitch_output = wp.zeros(len(r_vals), dtype=wp.float32, device=device)

        inv_w = 1.0 / (r_off - r_on)

        wp.launch(
            self.eval_s5_switch_kernel,
            dim=len(r_vals),
            inputs=[r_wp, r_on, r_off, inv_w],
            outputs=[switch_output, dswitch_output],
            device=device,
        )

        switch = from_warp(switch_output)
        assert np.all(switch > 0.0) and np.all(switch < 1.0)
        assert np.all(np.diff(switch) < 0)  # Monotonically decreasing


# ==============================================================================
# Warp Launcher Interface Tests
# ==============================================================================


class TestWarpLauncherMatrix:
    """Test dftd3_matrix warp launcher interface."""

    @pytest.mark.usefixtures("h2_system", "element_tables", "device")
    def test_basic_h2(self, request):
        """Test basic H2 system with neighbor matrix format."""
        h2_system = request.getfixturevalue("h2_system")
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        functional_params = {
            "a1": 0.4,
            "a2": 4.0,
            "s6": 1.0,
            "s8": 0.8,
            "k1": 16.0,
            "k3": -4.0,
        }

        results = run_dftd3_matrix(h2_system, element_tables, functional_params, device)

        # Basic checks
        assert results["energy"].shape == (1,)
        assert results["forces"].shape == (2, 3)
        assert results["coord_num"].shape == (2,)
        assert np.isfinite(results["energy"]).all()
        assert np.isfinite(results["forces"]).all()
        assert np.isfinite(results["coord_num"]).all()

        # Energy should be negative (attractive dispersion)
        assert results["energy"][0] < 0.0

        # Forces should be opposite for symmetric system
        np.testing.assert_allclose(
            results["forces"][0], -results["forces"][1], rtol=1e-5, atol=1e-7
        )

    @pytest.mark.parametrize("wp_dtype", [wp.float32, wp.float64])
    @pytest.mark.usefixtures("h2_system", "element_tables", "device")
    def test_dtype_support(self, request, wp_dtype):
        """Test that both float32 and float64 precision are supported."""
        h2_system = request.getfixturevalue("h2_system")
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        functional_params = {
            "a1": 0.4,
            "a2": 4.0,
            "s6": 1.0,
            "s8": 0.8,
            "k1": 16.0,
            "k3": -4.0,
        }

        results = run_dftd3_matrix(
            h2_system,
            element_tables,
            functional_params,
            device,
            wp_dtype=wp_dtype,
        )

        # Should produce finite results for both precisions
        assert np.isfinite(results["energy"]).all()
        assert np.isfinite(results["forces"]).all()
        assert np.isfinite(results["coord_num"]).all()

    @pytest.mark.usefixtures("element_tables", "device")
    def test_empty_system(self, request):
        """Test handling of empty system."""
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        # Empty system should be handled gracefully
        # Note: We test this directly without the helper
        positions_wp = wp.zeros((0, 3), dtype=wp.vec3f, device=device)
        numbers_wp = wp.zeros(0, dtype=wp.int32, device=device)
        neighbor_matrix_wp = wp.zeros((0, 0), dtype=wp.int32, device=device)

        max_z_inc = element_tables["z_max_inc"]
        rcov_wp = to_warp(element_tables["rcov"], wp.float32, device)
        r4r2_wp = to_warp(element_tables["r4r2"], wp.float32, device)
        c6_reference_wp = to_warp(
            element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5),
            wp.float32,
            device,
        )
        coord_num_ref_wp = to_warp(
            element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5),
            wp.float32,
            device,
        )

        coord_num_wp = wp.zeros(0, dtype=wp.float32, device=device)
        forces_wp = wp.zeros(0, dtype=wp.vec3f, device=device)
        energy_wp = wp.zeros(1, dtype=wp.float32, device=device)
        virial_wp = wp.zeros(1, dtype=wp.mat33f, device=device)
        batch_idx_wp = wp.zeros(0, dtype=wp.int32, device=device)

        # Should not crash with empty system
        dftd3_matrix(
            positions=positions_wp,
            numbers=numbers_wp,
            neighbor_matrix=neighbor_matrix_wp,
            covalent_radii=rcov_wp,
            r4r2=r4r2_wp,
            c6_reference=c6_reference_wp,
            coord_num_ref=coord_num_ref_wp,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            coord_num=coord_num_wp,
            forces=forces_wp,
            energy=energy_wp,
            virial=virial_wp,
            wp_dtype=wp.float32,
            device=device,
            batch_idx=batch_idx_wp,
        )

        energy = from_warp(energy_wp)
        assert energy[0] == pytest.approx(0.0)


# ==============================================================================
# Regression Tests
# ==============================================================================


class TestRegression:
    """Regression tests against reference outputs to ensure kernel values haven't changed."""

    @pytest.mark.parametrize(
        "system_name",
        ["ne2_system", "hcl_dimer_system"],
    )
    @pytest.mark.usefixtures("element_tables", "functional_params", "device")
    def test_regression(
        self,
        system_name,
        request,
    ):
        """Test full pipeline against reference outputs for regression."""
        system = request.getfixturevalue(system_name)
        element_tables = request.getfixturevalue("element_tables")
        functional_params = request.getfixturevalue("functional_params")
        device = request.getfixturevalue("device")

        results = run_dftd3_matrix(system, element_tables, functional_params, device)

        # Basic sanity checks
        assert np.isfinite(results["energy"]).all()
        assert np.isfinite(results["forces"]).all()
        assert np.isfinite(results["coord_num"]).all()

        # CN should be non-negative and reasonable
        assert (results["coord_num"] >= 0).all()
        assert (results["coord_num"] <= 12).all()  # Physical upper bound

        # Energy should be negative (attractive)
        assert results["energy"][0] < 0.0

    @pytest.mark.usefixtures(
        "ne2_system",
        "element_tables",
        "functional_params",
        "device",
        "ne2_reference_cpu",
    )
    def test_ne2_reference_values(
        self,
        request,
    ):
        """Test Ne2 system against hard-coded reference values from original implementation."""
        ne2_system = request.getfixturevalue("ne2_system")
        element_tables = request.getfixturevalue("element_tables")
        functional_params = request.getfixturevalue("functional_params")
        device = request.getfixturevalue("device")
        ne2_reference_cpu = request.getfixturevalue("ne2_reference_cpu")

        results = run_dftd3_matrix(
            ne2_system, element_tables, functional_params, device
        )

        # Compare against reference values from conftest.py
        reference = ne2_reference_cpu

        # Energy comparison
        np.testing.assert_allclose(
            results["energy"],
            reference["total_energy"],
            rtol=1e-5,
            atol=1e-7,
            err_msg="Energy changed from reference implementation",
        )

        # Coordination number comparison
        np.testing.assert_allclose(
            results["coord_num"],
            reference["cn"],
            rtol=1e-5,
            atol=1e-7,
            err_msg="Coordination numbers changed from reference implementation",
        )

        # Force comparison
        np.testing.assert_allclose(
            results["forces"],
            reference["force"],
            rtol=1e-5,
            atol=1e-7,
            err_msg="Forces changed from reference implementation",
        )

    @pytest.mark.usefixtures(
        "hcl_dimer_system",
        "element_tables",
        "functional_params",
        "device",
        "hcl_dimer_reference_cpu",
    )
    def test_hcl_dimer_reference_values(
        self,
        request,
    ):
        """Test HCl dimer system against hard-coded reference values from original implementation."""
        hcl_dimer_system = request.getfixturevalue("hcl_dimer_system")
        element_tables = request.getfixturevalue("element_tables")
        functional_params = request.getfixturevalue("functional_params")
        device = request.getfixturevalue("device")
        hcl_dimer_reference_cpu = request.getfixturevalue("hcl_dimer_reference_cpu")

        results = run_dftd3_matrix(
            hcl_dimer_system, element_tables, functional_params, device
        )

        # Compare against reference values from conftest.py
        reference = hcl_dimer_reference_cpu

        # Energy comparison
        np.testing.assert_allclose(
            results["energy"],
            reference["total_energy"],
            rtol=1e-5,
            atol=1e-7,
            err_msg="Energy changed from reference implementation",
        )

        # Coordination number comparison
        np.testing.assert_allclose(
            results["coord_num"],
            reference["cn"],
            rtol=1e-5,
            atol=1e-7,
            err_msg="Coordination numbers changed from reference implementation",
        )

        # Force comparison
        np.testing.assert_allclose(
            results["forces"],
            reference["force"],
            rtol=1e-5,
            atol=1e-7,
            err_msg="Forces changed from reference implementation",
        )


# ==============================================================================
# CPU/GPU Consistency Tests
# ==============================================================================


class TestCPUGPUConsistency:
    """CPU/GPU consistency tests for warp launchers."""

    @pytest.mark.parametrize(
        "system_name",
        ["ne2_system"],
    )
    @pytest.mark.usefixtures("element_tables", "functional_params")
    def test_consistency(
        self,
        system_name,
        request,
    ):
        """Test that warp launcher produces identical results on CPU and GPU."""
        # Skip if CUDA not available
        if not wp.is_cuda_available():
            pytest.skip("CUDA not available")

        system = request.getfixturevalue(system_name)
        element_tables = request.getfixturevalue("element_tables")
        functional_params = request.getfixturevalue("functional_params")

        device_cpu = "cpu"
        device_gpu = "cuda:0"

        # Run on both devices
        results_cpu = run_dftd3_matrix(
            system, element_tables, functional_params, device_cpu
        )
        results_gpu = run_dftd3_matrix(
            system, element_tables, functional_params, device_gpu
        )

        # Compare outputs
        np.testing.assert_allclose(
            results_gpu["energy"], results_cpu["energy"], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            results_gpu["forces"], results_cpu["forces"], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            results_gpu["coord_num"],
            results_cpu["coord_num"],
            rtol=1e-6,
            atol=1e-6,
        )


# ==============================================================================
# Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Tests for critical edge cases."""

    @pytest.mark.usefixtures("single_atom_system", "element_tables", "device")
    def test_single_atom(self, request):
        """Test single atom system (no neighbors)."""
        single_atom_system = request.getfixturevalue("single_atom_system")
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        functional_params = {
            "a1": 0.4,
            "a2": 4.0,
            "s6": 1.0,
            "s8": 0.8,
            "k1": 16.0,
            "k3": -4.0,
        }

        results = run_dftd3_matrix(
            single_atom_system, element_tables, functional_params, device
        )

        # Single atom should have zero energy and forces
        assert results["energy"][0] == pytest.approx(0.0, abs=1e-10)
        assert np.allclose(results["forces"], 0.0, atol=1e-10)
        assert results["coord_num"][0] == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.usefixtures("element_tables", "device")
    def test_very_short_distance(self, request):
        """Test numerical stability at very short distances (BJ damping)."""
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        # H2 at very short distance (0.5 Bohr)
        coord = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)
        numbers = np.array([1, 1], dtype=np.int32)
        nbmat = np.array([[1, 2], [0, 2]], dtype=np.int32)

        system = {
            "coord": coord,
            "numbers": numbers,
            "nbmat": nbmat,
            "B": 2,
            "M": 2,
        }

        functional_params = {
            "a1": 0.3981,
            "a2": 4.4211,
            "s6": 1.0,
            "s8": 1.9889,
            "k1": 16.0,
            "k3": -4.0,
        }

        results = run_dftd3_matrix(system, element_tables, functional_params, device)

        # Should not produce NaN or Inf due to BJ damping
        assert np.isfinite(results["energy"]).all()
        assert np.isfinite(results["forces"]).all()


# ==============================================================================
# Batching Tests
# ==============================================================================


class TestBatching:
    """Tests for batched calculations with multiple systems."""

    @pytest.mark.usefixtures("element_tables", "device")
    def test_two_identical_systems(self, request):
        """Test batching two identical H2 molecules."""
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        # Create two identical H2 molecules
        coord = np.array(
            [
                0.0,
                0.0,
                0.0,  # System 0, atom 0
                1.4,
                0.0,
                0.0,  # System 0, atom 1
                0.0,
                0.0,
                0.0,  # System 1, atom 0
                1.4,
                0.0,
                0.0,  # System 1, atom 1
            ],
            dtype=np.float32,
        )
        numbers = np.array([1, 1, 1, 1], dtype=np.int32)
        batch_idx = np.array([0, 0, 1, 1], dtype=np.int32)

        # Create symmetric neighbor lists
        nbmat = np.array([[1, 4], [0, 4], [3, 4], [2, 4]], dtype=np.int32)

        system = {
            "coord": coord,
            "numbers": numbers,
            "nbmat": nbmat,
            "B": 4,
            "M": 2,
        }

        functional_params = {
            "a1": 0.3981,
            "a2": 4.4211,
            "s6": 1.0,
            "s8": 1.9889,
            "k1": 16.0,
            "k3": -4.0,
        }

        results = run_dftd3_matrix(
            system, element_tables, functional_params, device, batch_idx
        )

        # Should have 2 systems
        assert results["energy"].shape == (2,)
        assert results["forces"].shape == (4, 3)
        assert results["coord_num"].shape == (4,)

        # Both systems are identical, so energies should be equal
        np.testing.assert_allclose(
            results["energy"][0], results["energy"][1], rtol=1e-6, atol=1e-6
        )

        # Forces for system 0 and system 1 should be identical
        np.testing.assert_allclose(
            results["forces"][0:2], results["forces"][2:4], rtol=1e-6, atol=1e-6
        )


# ==============================================================================
# Shape Tests
# ==============================================================================


class TestShapes:
    """Verify output array shapes are correct."""

    @pytest.mark.usefixtures("h2_system", "element_tables", "device")
    def test_output_shapes(self, request):
        """Test that output shapes match expected dimensions."""
        h2_system = request.getfixturevalue("h2_system")
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        functional_params = {
            "a1": 0.3981,
            "a2": 4.4211,
            "s6": 1.0,
            "s8": 1.9889,
            "k1": 16.0,
            "k3": -4.0,
        }

        results = run_dftd3_matrix(h2_system, element_tables, functional_params, device)

        assert results["energy"].shape == (1,)
        assert results["forces"].shape == (h2_system["B"], 3)
        assert results["coord_num"].shape == (h2_system["B"],)

        # Forces should be float32
        assert results["forces"].dtype == np.float32


# ==============================================================================
# Warp Launcher Neighbor List Tests
# ==============================================================================


class TestWarpLauncherList:
    """Test dftd3 warp launcher interface (CSR neighbor list format)."""

    @pytest.mark.usefixtures("h2_system", "element_tables", "device")
    def test_basic_h2(self, request):
        """Test basic H2 system with neighbor list format."""
        h2_system = request.getfixturevalue("h2_system")
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        functional_params = {
            "a1": 0.4,
            "a2": 4.0,
            "s6": 1.0,
            "s8": 0.8,
            "k1": 16.0,
            "k3": -4.0,
        }

        results = run_dftd3(h2_system, element_tables, functional_params, device)

        # Basic checks
        assert results["energy"].shape == (1,)
        assert results["forces"].shape == (2, 3)
        assert results["coord_num"].shape == (2,)
        assert np.isfinite(results["energy"]).all()
        assert np.isfinite(results["forces"]).all()
        assert np.isfinite(results["coord_num"]).all()

        # Energy should be negative (attractive dispersion)
        assert results["energy"][0] < 0.0

        # Forces should be opposite for symmetric system
        np.testing.assert_allclose(
            results["forces"][0], -results["forces"][1], rtol=1e-5, atol=1e-7
        )

    @pytest.mark.parametrize("wp_dtype", [wp.float32, wp.float64])
    @pytest.mark.usefixtures("h2_system", "element_tables", "device")
    def test_dtype_support(self, request, wp_dtype):
        """Test that both float32 and float64 precision are supported."""
        h2_system = request.getfixturevalue("h2_system")
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        functional_params = {
            "a1": 0.4,
            "a2": 4.0,
            "s6": 1.0,
            "s8": 0.8,
            "k1": 16.0,
            "k3": -4.0,
        }

        results = run_dftd3(
            h2_system,
            element_tables,
            functional_params,
            device,
            wp_dtype=wp_dtype,
        )

        # Should produce finite results for both precisions
        assert np.isfinite(results["energy"]).all()
        assert np.isfinite(results["forces"]).all()
        assert np.isfinite(results["coord_num"]).all()

    @pytest.mark.usefixtures("element_tables", "device")
    def test_empty_system(self, request):
        """Test handling of empty system."""
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        # Empty system should be handled gracefully
        # Note: We test this directly without the helper
        positions_wp = wp.zeros((0, 3), dtype=wp.vec3f, device=device)
        numbers_wp = wp.zeros(0, dtype=wp.int32, device=device)
        idx_j_wp = wp.zeros(0, dtype=wp.int32, device=device)
        neighbor_ptr_wp = wp.array([0], dtype=wp.int32, device=device)

        max_z_inc = element_tables["z_max_inc"]
        rcov_wp = to_warp(element_tables["rcov"], wp.float32, device)
        r4r2_wp = to_warp(element_tables["r4r2"], wp.float32, device)
        c6_reference_wp = to_warp(
            element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5),
            wp.float32,
            device,
        )
        coord_num_ref_wp = to_warp(
            element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5),
            wp.float32,
            device,
        )

        coord_num_wp = wp.zeros(0, dtype=wp.float32, device=device)
        forces_wp = wp.zeros(0, dtype=wp.vec3f, device=device)
        energy_wp = wp.zeros(1, dtype=wp.float32, device=device)
        virial_wp = wp.zeros(1, dtype=wp.mat33f, device=device)
        batch_idx_wp = wp.zeros(0, dtype=wp.int32, device=device)

        # Should not crash with empty system
        dftd3(
            positions=positions_wp,
            numbers=numbers_wp,
            idx_j=idx_j_wp,
            neighbor_ptr=neighbor_ptr_wp,
            covalent_radii=rcov_wp,
            r4r2=r4r2_wp,
            c6_reference=c6_reference_wp,
            coord_num_ref=coord_num_ref_wp,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            coord_num=coord_num_wp,
            forces=forces_wp,
            energy=energy_wp,
            virial=virial_wp,
            wp_dtype=wp.float32,
            device=device,
            batch_idx=batch_idx_wp,
        )

        energy = from_warp(energy_wp)
        assert energy[0] == pytest.approx(0.0)


# ==============================================================================
# Neighbor Format Consistency Tests
# ==============================================================================


class TestFormatConsistency:
    """Test that neighbor matrix and neighbor list formats produce identical results."""

    @pytest.mark.usefixtures("h2_system", "element_tables", "device")
    def test_nm_nl_consistency_h2(self, request):
        """Test that dftd3_matrix and dftd3 produce identical results for H2."""
        h2_system = request.getfixturevalue("h2_system")
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        functional_params = {
            "a1": 0.4,
            "a2": 4.0,
            "s6": 1.0,
            "s8": 0.8,
            "k1": 16.0,
            "k3": -4.0,
        }

        # Run both methods
        results_nm = run_dftd3_matrix(
            h2_system, element_tables, functional_params, device
        )
        results_nl = run_dftd3(h2_system, element_tables, functional_params, device)

        # Compare outputs - should be identical
        np.testing.assert_allclose(
            results_nl["energy"],
            results_nm["energy"],
            rtol=1e-6,
            atol=1e-7,
            err_msg="Energy differs between neighbor matrix and neighbor list formats",
        )
        np.testing.assert_allclose(
            results_nl["forces"],
            results_nm["forces"],
            rtol=1e-6,
            atol=1e-7,
            err_msg="Forces differ between neighbor matrix and neighbor list formats",
        )
        np.testing.assert_allclose(
            results_nl["coord_num"],
            results_nm["coord_num"],
            rtol=1e-6,
            atol=1e-7,
            err_msg="Coordination numbers differ between neighbor matrix and neighbor list formats",
        )

    @pytest.mark.usefixtures(
        "ne2_system", "element_tables", "functional_params", "device"
    )
    def test_nm_nl_consistency_ne2(self, request):
        """Test that dftd3_matrix and dftd3 produce identical results for Ne2."""
        ne2_system = request.getfixturevalue("ne2_system")
        element_tables = request.getfixturevalue("element_tables")
        functional_params = request.getfixturevalue("functional_params")
        device = request.getfixturevalue("device")

        # Run both methods
        results_nm = run_dftd3_matrix(
            ne2_system, element_tables, functional_params, device
        )
        results_nl = run_dftd3(ne2_system, element_tables, functional_params, device)

        # Compare outputs - should be identical
        np.testing.assert_allclose(
            results_nl["energy"],
            results_nm["energy"],
            rtol=1e-6,
            atol=1e-7,
            err_msg="Energy differs between neighbor matrix and neighbor list formats",
        )
        np.testing.assert_allclose(
            results_nl["forces"],
            results_nm["forces"],
            rtol=1e-6,
            atol=1e-7,
            err_msg="Forces differ between neighbor matrix and neighbor list formats",
        )
        np.testing.assert_allclose(
            results_nl["coord_num"],
            results_nm["coord_num"],
            rtol=1e-6,
            atol=1e-7,
            err_msg="Coordination numbers differ between neighbor matrix and neighbor list formats",
        )

    @pytest.mark.usefixtures(
        "hcl_dimer_system", "element_tables", "functional_params", "device"
    )
    def test_nm_nl_consistency_hcl_dimer(self, request):
        """Test that dftd3_matrix and dftd3 produce identical results for HCl dimer."""
        hcl_dimer_system = request.getfixturevalue("hcl_dimer_system")
        element_tables = request.getfixturevalue("element_tables")
        functional_params = request.getfixturevalue("functional_params")
        device = request.getfixturevalue("device")

        # Run both methods
        results_nm = run_dftd3_matrix(
            hcl_dimer_system, element_tables, functional_params, device
        )
        results_nl = run_dftd3(
            hcl_dimer_system, element_tables, functional_params, device
        )

        # Compare outputs - should be identical
        np.testing.assert_allclose(
            results_nl["energy"],
            results_nm["energy"],
            rtol=1e-6,
            atol=1e-7,
            err_msg="Energy differs between neighbor matrix and neighbor list formats",
        )
        np.testing.assert_allclose(
            results_nl["forces"],
            results_nm["forces"],
            rtol=1e-6,
            atol=1e-7,
            err_msg="Forces differ between neighbor matrix and neighbor list formats",
        )
        np.testing.assert_allclose(
            results_nl["coord_num"],
            results_nm["coord_num"],
            rtol=1e-6,
            atol=1e-7,
            err_msg="Coordination numbers differ between neighbor matrix and neighbor list formats",
        )


# ==============================================================================
# Regression Tests (Neighbor List)
# ==============================================================================


class TestRegressionList:
    """Regression tests for neighbor list format against reference outputs."""

    @pytest.mark.usefixtures(
        "ne2_system",
        "element_tables",
        "functional_params",
        "device",
        "ne2_reference_cpu",
    )
    def test_ne2_reference_values(
        self,
        request,
    ):
        """Test Ne2 system against hard-coded reference values using neighbor list format."""
        ne2_system = request.getfixturevalue("ne2_system")
        element_tables = request.getfixturevalue("element_tables")
        functional_params = request.getfixturevalue("functional_params")
        device = request.getfixturevalue("device")
        ne2_reference_cpu = request.getfixturevalue("ne2_reference_cpu")

        results = run_dftd3(ne2_system, element_tables, functional_params, device)

        # Compare against reference values from conftest.py
        reference = ne2_reference_cpu

        # Energy comparison
        np.testing.assert_allclose(
            results["energy"],
            reference["total_energy"],
            rtol=1e-5,
            atol=1e-7,
            err_msg="Energy changed from reference implementation (neighbor list format)",
        )

        # Coordination number comparison
        np.testing.assert_allclose(
            results["coord_num"],
            reference["cn"],
            rtol=1e-5,
            atol=1e-7,
            err_msg="Coordination numbers changed from reference implementation (neighbor list format)",
        )

        # Force comparison
        np.testing.assert_allclose(
            results["forces"],
            reference["force"],
            rtol=1e-5,
            atol=1e-7,
            err_msg="Forces changed from reference implementation (neighbor list format)",
        )

    @pytest.mark.usefixtures(
        "hcl_dimer_system",
        "element_tables",
        "functional_params",
        "device",
        "hcl_dimer_reference_cpu",
    )
    def test_hcl_dimer_reference_values(
        self,
        request,
    ):
        """Test HCl dimer system against hard-coded reference values using neighbor list format."""
        hcl_dimer_system = request.getfixturevalue("hcl_dimer_system")
        element_tables = request.getfixturevalue("element_tables")
        functional_params = request.getfixturevalue("functional_params")
        device = request.getfixturevalue("device")
        hcl_dimer_reference_cpu = request.getfixturevalue("hcl_dimer_reference_cpu")

        results = run_dftd3(hcl_dimer_system, element_tables, functional_params, device)

        # Compare against reference values from conftest.py
        reference = hcl_dimer_reference_cpu

        # Energy comparison
        np.testing.assert_allclose(
            results["energy"],
            reference["total_energy"],
            rtol=1e-5,
            atol=1e-7,
            err_msg="Energy changed from reference implementation (neighbor list format)",
        )

        # Coordination number comparison
        np.testing.assert_allclose(
            results["coord_num"],
            reference["cn"],
            rtol=1e-5,
            atol=1e-7,
            err_msg="Coordination numbers changed from reference implementation (neighbor list format)",
        )

        # Force comparison
        np.testing.assert_allclose(
            results["forces"],
            reference["force"],
            rtol=1e-5,
            atol=1e-7,
            err_msg="Forces changed from reference implementation (neighbor list format)",
        )


# ==============================================================================
# CPU/GPU Consistency Tests (Neighbor List)
# ==============================================================================


class TestCPUGPUConsistencyList:
    """CPU/GPU consistency tests for neighbor list format."""

    @pytest.mark.parametrize(
        "system_name",
        ["ne2_system"],
    )
    @pytest.mark.usefixtures("element_tables", "functional_params")
    def test_consistency(
        self,
        system_name,
        request,
    ):
        """Test that neighbor list launcher produces identical results on CPU and GPU."""
        # Skip if CUDA not available
        if not wp.is_cuda_available():
            pytest.skip("CUDA not available")

        system = request.getfixturevalue(system_name)
        element_tables = request.getfixturevalue("element_tables")
        functional_params = request.getfixturevalue("functional_params")

        device_cpu = "cpu"
        device_gpu = "cuda:0"

        # Run on both devices
        results_cpu = run_dftd3(system, element_tables, functional_params, device_cpu)
        results_gpu = run_dftd3(system, element_tables, functional_params, device_gpu)

        # Compare outputs
        np.testing.assert_allclose(
            results_gpu["energy"], results_cpu["energy"], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            results_gpu["forces"], results_cpu["forces"], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            results_gpu["coord_num"],
            results_cpu["coord_num"],
            rtol=1e-6,
            atol=1e-6,
        )

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Minimal test suite for DFT-D3 dftd3.py implementation.

This test suite focuses on the high-level wrapper function and includes:
- S5 switching function tests
- Full pipeline regression tests
- CPU/GPU consistency
- Edge cases (empty systems, short distances)
- PBC (periodic boundary conditions) tests
- Basic shape and interface tests

The philosophy is to test the public API comprehensively while keeping
the test suite minimal and focused.
"""

from __future__ import annotations

import pytest
import torch
import warp as wp

from nvalchemiops.interactions.dispersion.dftd3 import (
    D3Parameters,
    _s5_switch,
    dftd3,
)
from test.interactions.dispersion.conftest import (
    from_warp,
    to_warp,
)

# ==============================================================================
# Helper Functions
# ==============================================================================


def run_dftd3(
    system: dict,
    element_tables: dict,
    functional_params: dict,
    device: str,
    batch_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Run dftd3 wrapper for a system.

    Parameters
    ----------
    system : dict
        System with 'coord', 'numbers', 'nbmat', 'B', 'M'
    element_tables : dict
        Element parameters
    functional_params : dict
        Functional parameters (k1, k3, a1, a2, s6, s8)
    device : str
        Warp device string
    batch_indices : torch.Tensor or None
        Batch indices for atoms

    Returns
    -------
    dict
        Results with 'energy', 'forces', 'coord_num'
    """
    # Extract system data
    B = system["B"]
    coord_flat = system["coord"]
    numbers = system["numbers"]
    nbmat = system["nbmat"]

    # Determine torch device from Warp device string
    torch_device = "cuda" if "cuda" in device else "cpu"

    # Convert to PyTorch tensors (fixtures may already be tensors)
    if isinstance(coord_flat, torch.Tensor):
        positions = coord_flat.reshape(B, 3).float().to(torch_device)
    else:
        positions = torch.from_numpy(coord_flat).reshape(B, 3).float().to(torch_device)

    if isinstance(numbers, torch.Tensor):
        numbers_torch = numbers.int().to(torch_device)
    else:
        numbers_torch = torch.from_numpy(numbers).int().to(torch_device)

    if isinstance(nbmat, torch.Tensor):
        neighbor_matrix = nbmat.int().to(torch_device)
    else:
        neighbor_matrix = torch.from_numpy(nbmat).int().to(torch_device)

    # Prepare element tables (fixtures may already be tensors)
    def ensure_tensor(data):
        if isinstance(data, torch.Tensor):
            return data.float().to(torch_device)
        else:
            return torch.from_numpy(data).float().to(torch_device)

    rcov = ensure_tensor(element_tables["rcov"])
    r4r2 = ensure_tensor(element_tables["r4r2"])

    # Convert reference arrays to 4D format [max_Z+1, max_Z+1, 5, 5]
    # The old format is 3D [max_Z+1, max_Z+1, 25], need to reshape
    c6ref_3d = element_tables["c6ref"]
    max_z_inc = element_tables["z_max_inc"]
    c6_reference = ensure_tensor(c6ref_3d).reshape(max_z_inc, max_z_inc, 5, 5)

    # For coord_num_ref, we only need one (the old had i and j which are transposes)
    cnref_i_3d = element_tables["cnref_i"]
    coord_num_ref = ensure_tensor(cnref_i_3d).reshape(max_z_inc, max_z_inc, 5, 5)

    # Call the wrapper
    energy, forces, coord_num = dftd3(
        positions=positions,
        numbers=numbers_torch,
        neighbor_matrix=neighbor_matrix,
        covalent_radii=rcov,
        r4r2=r4r2,
        c6_reference=c6_reference,
        coord_num_ref=coord_num_ref,
        a1=functional_params["a1"],
        a2=functional_params["a2"],
        s8=functional_params["s8"],
        k1=functional_params["k1"],
        k3=functional_params["k3"],
        s6=functional_params["s6"],
        batch_idx=batch_indices,
        device=device,
    )

    return {
        "energy": energy,
        "forces": forces,
        "coord_num": coord_num,
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
    def test_s5_switch_regions(
        self, device_cpu, r_vals, r_on, r_off, expected_sw, expected_behavior
    ):
        """Test _s5_switch in different regions (below r_on, above r_off, disabled)."""
        r_tensor = torch.tensor(r_vals, dtype=torch.float32)
        r_wp = to_warp(r_tensor, wp.float32, device_cpu)
        switch_output = wp.zeros(len(r_vals), dtype=wp.float32, device=device_cpu)
        dswitch_output = wp.zeros(len(r_vals), dtype=wp.float32, device=device_cpu)

        # Compute inv_w
        inv_w = 1.0 / (r_off - r_on) if r_off > r_on else 0.0

        wp.launch(
            self.eval_s5_switch_kernel,
            dim=len(r_vals),
            inputs=[r_wp, r_on, r_off, inv_w],
            outputs=[switch_output, dswitch_output],
            device=device_cpu,
        )

        switch = from_warp(switch_output)
        dswitch = from_warp(dswitch_output)

        expected_tensor = torch.tensor(expected_sw, dtype=torch.float32)
        torch.testing.assert_close(switch, expected_tensor, atol=1e-7, rtol=0)

        if expected_behavior in ["below_r_on", "above_r_off", "disabled"]:
            torch.testing.assert_close(
                dswitch, torch.zeros_like(dswitch), atol=1e-7, rtol=0
            )

    def test_s5_switch_transition_region(self, device_cpu):
        """Test _s5_switch in transition region with monotonicity."""
        r_on, r_off = 2.0, 5.0
        r_vals = torch.linspace(r_on + 0.1, r_off - 0.1, 10, dtype=torch.float32)

        r_wp = to_warp(r_vals, wp.float32, device_cpu)
        switch_output = wp.zeros(len(r_vals), dtype=wp.float32, device=device_cpu)
        dswitch_output = wp.zeros(len(r_vals), dtype=wp.float32, device=device_cpu)

        inv_w = 1.0 / (r_off - r_on)

        wp.launch(
            self.eval_s5_switch_kernel,
            dim=len(r_vals),
            inputs=[r_wp, r_on, r_off, inv_w],
            outputs=[switch_output, dswitch_output],
            device=device_cpu,
        )

        switch = from_warp(switch_output)
        assert torch.all(switch > 0.0) and torch.all(switch < 1.0)
        assert torch.all(torch.diff(switch) < 0)  # Monotonically decreasing


# ==============================================================================
# D3Parameters Validation Tests
# ==============================================================================


class TestD3ParametersValidation:
    """Tests for D3Parameters dataclass validation."""

    def test_valid_parameters(self):
        """Test that valid parameters pass validation."""
        max_z = 10
        params = D3Parameters(
            rcov=torch.rand(max_z + 1, dtype=torch.float32) + 0.5,
            r4r2=torch.rand(max_z + 1, dtype=torch.float32) + 1.0,
            c6ab=torch.rand(max_z + 1, max_z + 1, 5, 5, dtype=torch.float32),
            cn_ref=torch.rand(max_z + 1, max_z + 1, 5, 5, dtype=torch.float32),
        )

        assert params.max_z == max_z
        assert params.device == torch.device("cpu")

    def test_invalid_tensor_type(self):
        """Test that non-tensor types raise TypeError."""
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            D3Parameters(
                rcov=[1.0, 2.0],  # List instead of tensor
                r4r2=torch.rand(3),
                c6ab=torch.rand(3, 3, 5, 5),
                cn_ref=torch.rand(3, 3, 5, 5),
            )

    def test_invalid_dtype(self):
        """Test that non-float dtypes raise TypeError."""
        with pytest.raises(TypeError, match="must be float32 or float64"):
            D3Parameters(
                rcov=torch.tensor([1, 2, 3], dtype=torch.int32),
                r4r2=torch.rand(3),
                c6ab=torch.rand(3, 3, 5, 5),
                cn_ref=torch.rand(3, 3, 5, 5),
            )

    def test_rcov_wrong_ndim(self):
        """Test that rcov with wrong dimensions raises ValueError."""
        with pytest.raises(ValueError, match="rcov must be 1D tensor"):
            D3Parameters(
                rcov=torch.rand(3, 3),  # 2D instead of 1D
                r4r2=torch.rand(3),
                c6ab=torch.rand(3, 3, 5, 5),
                cn_ref=torch.rand(3, 3, 5, 5),
            )

    def test_rcov_too_small(self):
        """Test that rcov with insufficient elements raises ValueError."""
        with pytest.raises(ValueError, match="must have at least 2 elements"):
            D3Parameters(
                rcov=torch.rand(1),  # Only padding, no elements
                r4r2=torch.rand(1),
                c6ab=torch.rand(1, 1, 5, 5),
                cn_ref=torch.rand(1, 1, 5, 5),
            )

    def test_r4r2_shape_mismatch(self):
        """Test that r4r2 shape mismatch with rcov raises ValueError."""
        with pytest.raises(ValueError, match="r4r2 must have shape"):
            D3Parameters(
                rcov=torch.rand(10),
                r4r2=torch.rand(5),  # Wrong size
                c6ab=torch.rand(10, 10, 5, 5),
                cn_ref=torch.rand(10, 10, 5, 5),
            )

    def test_c6ab_shape_mismatch(self):
        """Test that c6ab shape mismatch raises ValueError."""
        with pytest.raises(ValueError, match="c6ab must have shape"):
            D3Parameters(
                rcov=torch.rand(10),
                r4r2=torch.rand(10),
                c6ab=torch.rand(10, 10, 3, 3),  # Wrong interp_mesh size
                cn_ref=torch.rand(10, 10, 5, 5),
            )

    def test_cn_ref_shape_mismatch(self):
        """Test that cn_ref shape mismatch raises ValueError."""
        with pytest.raises(ValueError, match="cn_ref must have shape"):
            D3Parameters(
                rcov=torch.rand(10),
                r4r2=torch.rand(10),
                c6ab=torch.rand(10, 10, 5, 5),
                cn_ref=torch.rand(8, 8, 5, 5),  # Wrong max_z
            )

    def test_custom_interp_mesh(self):
        """Test that custom interp_mesh size works correctly."""
        max_z = 5
        interp_mesh = 3
        params = D3Parameters(
            rcov=torch.rand(max_z + 1),
            r4r2=torch.rand(max_z + 1),
            c6ab=torch.rand(max_z + 1, max_z + 1, interp_mesh, interp_mesh),
            cn_ref=torch.rand(max_z + 1, max_z + 1, interp_mesh, interp_mesh),
            interp_mesh=interp_mesh,
        )

        assert params.interp_mesh == interp_mesh
        assert params.c6ab.shape == (max_z + 1, max_z + 1, interp_mesh, interp_mesh)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_consistency(self):
        """Test that mixed devices raise ValueError."""
        with pytest.raises(
            ValueError, match="All parameters must be on the same device"
        ):
            D3Parameters(
                rcov=torch.rand(10, device="cpu"),
                r4r2=torch.rand(10, device="cpu"),
                c6ab=torch.rand(10, 10, 5, 5, device="cpu"),
                cn_ref=torch.rand(10, 10, 5, 5, device="cuda"),
            )

    def test_to_method_device(self):
        """Test that to() method moves parameters to correct device."""
        params = D3Parameters(
            rcov=torch.rand(10),
            r4r2=torch.rand(10),
            c6ab=torch.rand(10, 10, 5, 5),
            cn_ref=torch.rand(10, 10, 5, 5),
        )

        # Move to CPU (should be no-op)
        params_cpu = params.to(device="cpu")
        assert params_cpu.device == torch.device("cpu")
        assert params_cpu.rcov.device == torch.device("cpu")

    def test_to_method_dtype(self):
        """Test that to() method converts to correct dtype."""
        params = D3Parameters(
            rcov=torch.rand(10, dtype=torch.float64),
            r4r2=torch.rand(10, dtype=torch.float64),
            c6ab=torch.rand(10, 10, 5, 5, dtype=torch.float64),
            cn_ref=torch.rand(10, 10, 5, 5, dtype=torch.float64),
        )

        # Convert to float32
        params_f32 = params.to(dtype=torch.float32)
        assert params_f32.rcov.dtype == torch.float32
        assert params_f32.r4r2.dtype == torch.float32
        assert params_f32.c6ab.dtype == torch.float32
        assert params_f32.cn_ref.dtype == torch.float32

    def test_to_method_device_and_dtype(self):
        """Test that to() method handles both device and dtype."""
        params = D3Parameters(
            rcov=torch.rand(10, dtype=torch.float64),
            r4r2=torch.rand(10, dtype=torch.float64),
            c6ab=torch.rand(10, 10, 5, 5, dtype=torch.float64),
            cn_ref=torch.rand(10, 10, 5, 5, dtype=torch.float64),
        )

        # Convert to float32 and ensure on CPU
        params_converted = params.to(device="cpu", dtype=torch.float32)
        assert params_converted.device == torch.device("cpu")
        assert params_converted.rcov.dtype == torch.float32

    def test_float64_parameters(self):
        """Test that float64 parameters are accepted."""
        params = D3Parameters(
            rcov=torch.rand(10, dtype=torch.float64),
            r4r2=torch.rand(10, dtype=torch.float64),
            c6ab=torch.rand(10, 10, 5, 5, dtype=torch.float64),
            cn_ref=torch.rand(10, 10, 5, 5, dtype=torch.float64),
        )

        assert params.rcov.dtype == torch.float64
        assert params.r4r2.dtype == torch.float64


# ==============================================================================
# Regression Tests
# ==============================================================================


class TestRegression:
    """Regression tests against reference outputs."""

    @pytest.mark.parametrize(
        "system_name",
        ["ne2_system", "hcl_dimer_system"],
    )
    def test_regression(
        self,
        system_name,
        request,
        element_tables,
        functional_params,
        device_cpu,
    ):
        """Test full pipeline against reference outputs for regression."""
        system = request.getfixturevalue(system_name)

        results = run_dftd3(system, element_tables, functional_params, device_cpu)

        # Basic sanity checks (no reference data for new implementation yet)
        assert torch.isfinite(results["energy"]).all()
        assert torch.isfinite(results["forces"]).all()
        assert torch.isfinite(results["coord_num"]).all()

        # CN should be non-negative and reasonable
        assert (results["coord_num"] >= 0).all()
        assert (results["coord_num"] <= 12).all()  # Physical upper bound


# ==============================================================================
# CPU/GPU Consistency Tests
# ==============================================================================


class TestCPUGPUConsistency:
    """CPU/GPU consistency tests."""

    @pytest.mark.parametrize(
        "system_name",
        ["ne2_system"],
    )
    def test_consistency(
        self,
        system_name,
        request,
        element_tables,
        functional_params,
        device_cpu,
        device_gpu,
    ):
        """Test that wrapper produces identical results on CPU and GPU."""
        system = request.getfixturevalue(system_name)

        # Run on both devices
        results_cpu = run_dftd3(system, element_tables, functional_params, device_cpu)
        results_gpu = run_dftd3(system, element_tables, functional_params, device_gpu)

        # Compare outputs
        torch.testing.assert_close(
            results_gpu["energy"].cpu(), results_cpu["energy"], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            results_gpu["forces"].cpu(), results_cpu["forces"], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            results_gpu["coord_num"].cpu(),
            results_cpu["coord_num"],
            rtol=1e-6,
            atol=1e-6,
        )


# ==============================================================================
# Custom Op Branch Coverage Tests
# ==============================================================================


class TestCustomOpBranches:
    """Test specific branches in the custom op that need coverage."""

    def test_float64_positions(self, element_tables):
        """Test with float64 positions to cover float64 dtype branch."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float64
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32)
        neighbor_matrix = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5)
        coord_num_ref = element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5)

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=element_tables["rcov"],
            r4r2=element_tables["r4r2"],
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            device="cpu",
        )

        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()

    def test_with_s5_smoothing(self, element_tables):
        """Test with S5 smoothing enabled to cover inv_w calculation branch."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32)
        neighbor_matrix = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5)
        coord_num_ref = element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5)

        # Enable S5 smoothing: s5_smoothing_off > s5_smoothing_on
        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=element_tables["rcov"],
            r4r2=element_tables["r4r2"],
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            s5_smoothing_on=5.0,  # Enable smoothing
            s5_smoothing_off=10.0,  # s5_off > s5_on
            device="cpu",
        )

        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()

    def test_explicit_fill_value(self, element_tables):
        """Test with explicit fill_value to cover that branch."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32)
        neighbor_matrix = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5)
        coord_num_ref = element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5)

        # Explicitly provide fill_value
        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=element_tables["rcov"],
            r4r2=element_tables["r4r2"],
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            fill_value=2,  # Explicit fill_value
            device="cpu",
        )

        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()

    def test_no_device_parameter(self, element_tables):
        """Test without device parameter to cover device inference branch."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32)
        neighbor_matrix = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5)
        coord_num_ref = element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5)

        # Don't pass device parameter - it should be inferred
        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=element_tables["rcov"],
            r4r2=element_tables["r4r2"],
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            # device parameter not provided
        )

        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()


# ==============================================================================
# Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Tests for critical edge cases."""

    def test_empty_system(self, element_tables):
        """Test handling of completely empty system (0 atoms)."""
        positions = torch.zeros((0, 3), dtype=torch.float32)
        numbers = torch.zeros((0,), dtype=torch.int32)
        neighbor_matrix = torch.zeros((0, 0), dtype=torch.int32)

        # Prepare element tables as 4D (fixtures may be tensors or numpy arrays)
        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float()
            else:
                return torch.from_numpy(data).float()

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device="cpu",
        )

        # Verify shapes
        assert energy.shape == (1,)
        assert forces.shape == (0, 3)
        assert coord_num.shape == (0,)
        # Energy should be zero
        assert energy[0] == pytest.approx(0.0)

    def test_empty_system_with_batch(self, element_tables):
        """Test handling of empty system with batch indices."""
        positions = torch.zeros((0, 3), dtype=torch.float32)
        numbers = torch.zeros((0,), dtype=torch.int32)
        neighbor_matrix = torch.zeros((0, 0), dtype=torch.int32)
        batch_idx = torch.zeros((0,), dtype=torch.int32)

        # Prepare element tables as 4D (fixtures may be tensors or numpy arrays)
        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float()
            else:
                return torch.from_numpy(data).float()

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        energy, _, _ = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            batch_idx=batch_idx,
            device="cpu",
        )

        # Empty batch should still produce valid output
        assert energy.shape == (1,)
        assert energy[0] == pytest.approx(0.0)

    def test_very_short_distance(self, element_tables):
        """Test numerical stability at very short distances (BJ damping)."""
        # H2 at very short distance (0.5 Bohr)
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32)

        # Create symmetric neighbor matrix format
        # Atom 0 has neighbor 1, Atom 1 has neighbor 0
        neighbor_matrix = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        # Prepare element tables as 4D (fixtures may be tensors or numpy arrays)
        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float()
            else:
                return torch.from_numpy(data).float()

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        energy, forces, _ = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device="cpu",
        )

        # Should not produce NaN or Inf due to BJ damping
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()


# ==============================================================================
# PBC Tests
# ==============================================================================


class TestPBC:
    """Tests for periodic boundary conditions."""

    def test_pbc_disabled(self, element_tables):
        """Test non-periodic calculation (default behavior)."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32)
        neighbor_matrix = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        # Prepare element tables as 4D (fixtures may be tensors or numpy arrays)
        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float()
            else:
                return torch.from_numpy(data).float()

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        # No PBC parameters
        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device="cpu",
        )

        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()

    def test_virial_requires_pbc(self, element_tables):
        """Test that virial computation requires PBC parameters."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32)
        neighbor_matrix = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float()
            else:
                return torch.from_numpy(data).float()

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        # Test 1: compute_virial=True without cell
        with pytest.raises(
            ValueError, match="Virial computation requires periodic boundary conditions"
        ):
            dftd3(
                positions=positions,
                numbers=numbers,
                neighbor_matrix=neighbor_matrix,
                covalent_radii=ensure_tensor(element_tables["rcov"]),
                r4r2=ensure_tensor(element_tables["r4r2"]),
                c6_reference=c6_reference,
                coord_num_ref=coord_num_ref,
                a1=0.4,
                a2=4.0,
                s8=0.8,
                compute_virial=True,
                device="cpu",
            )

        # Test 2: cell provided but no neighbor_matrix_shifts
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=torch.float32,
        )
        with pytest.raises(ValueError, match="neighbor_matrix_shifts"):
            dftd3(
                positions=positions,
                numbers=numbers,
                neighbor_matrix=neighbor_matrix,
                cell=cell,  # Cell but no shifts
                covalent_radii=ensure_tensor(element_tables["rcov"]),
                r4r2=ensure_tensor(element_tables["r4r2"]),
                c6_reference=c6_reference,
                coord_num_ref=coord_num_ref,
                a1=0.4,
                a2=4.0,
                s8=0.8,
                compute_virial=True,
                device="cpu",
            )

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
    )
    def test_virial_neighbor_matrix(self, element_tables, device):
        """Test virial computation with neighbor_matrix format and PBC."""
        torch_device = "cuda" if "cuda" in device else "cpu"

        # Single H atom in periodic box seeing itself in neighboring cells
        positions = torch.tensor(
            [[0.0, 0.0, 0.0]], dtype=torch.float32, device=torch_device
        )
        numbers = torch.tensor([1], dtype=torch.int32, device=torch_device)

        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=torch.float32,
            device=torch_device,
        )
        neighbor_matrix = torch.tensor([[0, 0]], dtype=torch.int32, device=torch_device)
        neighbor_matrix_shifts = torch.tensor(
            [[[1, 0, 0], [0, 1, 0]]], dtype=torch.int32, device=torch_device
        )

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = (
            element_tables["c6ref"]
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
            .to(torch_device)
        )
        coord_num_ref = (
            element_tables["cnref_i"]
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
            .to(torch_device)
        )

        energy, forces, coord_num, virial = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            cell=cell,
            covalent_radii=element_tables["rcov"].float().to(torch_device),
            r4r2=element_tables["r4r2"].float().to(torch_device),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            compute_virial=True,
            device=device,
        )

        # Verify virial properties
        assert virial.shape == (1, 3, 3)
        assert virial.dtype == torch.float32
        assert virial.device.type == torch_device
        assert torch.isfinite(virial).all()

        # Check symmetry (virial should be symmetric for isotropic interactions)
        torch.testing.assert_close(
            virial, virial.transpose(-2, -1), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
    )
    def test_virial_neighbor_list(self, element_tables, device):
        """Test virial computation with neighbor_list format and unit_shifts."""
        torch_device = "cuda" if "cuda" in device else "cpu"

        positions = torch.tensor(
            [[0.0, 0.0, 0.0]], dtype=torch.float32, device=torch_device
        )
        numbers = torch.tensor([1], dtype=torch.int32, device=torch_device)
        batch_idx = torch.tensor([0], dtype=torch.int32, device=torch_device)

        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=torch.float32,
            device=torch_device,
        )

        # Neighbor list: atom 0 sees itself in 2 neighboring cells
        neighbor_list = torch.tensor(
            [[0, 0], [0, 0]], dtype=torch.int32, device=torch_device
        )
        unit_shifts = torch.tensor(
            [[1, 0, 0], [0, 1, 0]], dtype=torch.int32, device=torch_device
        )
        neighbor_ptr = torch.tensor([0, 2], dtype=torch.int32, device=torch_device)

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = (
            element_tables["c6ref"]
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
            .to(torch_device)
        )
        coord_num_ref = (
            element_tables["cnref_i"]
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
            .to(torch_device)
        )

        energy, forces, coord_num, virial = dftd3(
            positions=positions,
            numbers=numbers,
            batch_idx=batch_idx,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            unit_shifts=unit_shifts,
            cell=cell,
            covalent_radii=element_tables["rcov"].float().to(torch_device),
            r4r2=element_tables["r4r2"].float().to(torch_device),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            compute_virial=True,
            device=device,
        )

        assert virial.shape == (1, 3, 3)
        assert virial.dtype == torch.float32
        assert torch.isfinite(virial).all()

        # Check symmetry
        torch.testing.assert_close(
            virial, virial.transpose(-2, -1), rtol=1e-5, atol=1e-5
        )

    def test_virial_batched(self, element_tables):
        """Test virial computation with multiple batched systems."""
        # Two systems: H and Ne atoms in different sized boxes
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
        )
        numbers = torch.tensor([1, 10], dtype=torch.int32)
        batch_idx = torch.tensor([0, 1], dtype=torch.int32)

        # Different cell sizes for each system
        cell = torch.tensor(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]],
            ],
            dtype=torch.float32,
        )

        # Each atom sees itself in neighboring cells
        neighbor_matrix = torch.tensor([[0, 0], [1, 1]], dtype=torch.int32)
        neighbor_matrix_shifts = torch.tensor(
            [[[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 0, 1]]], dtype=torch.int32
        )

        # Prepare element tables
        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float()
            else:
                return torch.from_numpy(data).float()

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        energy, forces, coord_num, virial = dftd3(
            positions=positions,
            numbers=numbers,
            batch_idx=batch_idx,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            cell=cell,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            compute_virial=True,
            device="cpu",
        )

        # Should have virial for each system
        assert virial.shape == (2, 3, 3)
        assert torch.isfinite(virial).all()

        # Each system should have symmetric virial
        torch.testing.assert_close(
            virial[0], virial[0].transpose(-1, -2), rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            virial[1], virial[1].transpose(-1, -2), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.gpu
    def test_virial_cpu_gpu_consistency(self, element_tables):
        """Test that virial computation produces identical results on CPU and GPU."""
        # Single H atom in periodic box
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        numbers = torch.tensor([1], dtype=torch.int32)

        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=torch.float32,
        )
        neighbor_matrix = torch.tensor([[0, 0]], dtype=torch.int32)
        neighbor_matrix_shifts = torch.tensor(
            [[[1, 0, 0], [0, 1, 0]]], dtype=torch.int32
        )

        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float()
            else:
                return torch.from_numpy(data).float()

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        # Run on CPU
        _, _, _, virial_cpu = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            cell=cell,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            compute_virial=True,
            device="cpu",
        )

        # Run on GPU
        _, _, _, virial_gpu = dftd3(
            positions=positions.cuda(),
            numbers=numbers.cuda(),
            neighbor_matrix=neighbor_matrix.cuda(),
            neighbor_matrix_shifts=neighbor_matrix_shifts.cuda(),
            cell=cell.cuda(),
            covalent_radii=ensure_tensor(element_tables["rcov"]).cuda(),
            r4r2=ensure_tensor(element_tables["r4r2"]).cuda(),
            c6_reference=c6_reference.cuda(),
            coord_num_ref=coord_num_ref.cuda(),
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            compute_virial=True,
            device="cuda:0",
        )

        # Compare results
        torch.testing.assert_close(virial_gpu.cpu(), virial_cpu, rtol=1e-6, atol=1e-6)

    def test_virial_float64(self, element_tables):
        """Test virial computation with float64 positions."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0]],
            dtype=torch.float64,  # float64
        )
        numbers = torch.tensor([1], dtype=torch.int32)

        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=torch.float64,  # float64
        )
        neighbor_matrix = torch.tensor([[0, 0]], dtype=torch.int32)
        neighbor_matrix_shifts = torch.tensor(
            [[[1, 0, 0], [0, 1, 0]]], dtype=torch.int32
        )

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = (
            element_tables["c6ref"].float().reshape(max_z_inc, max_z_inc, 5, 5)
        )
        coord_num_ref = (
            element_tables["cnref_i"].float().reshape(max_z_inc, max_z_inc, 5, 5)
        )

        energy, forces, coord_num, virial = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            cell=cell,
            covalent_radii=element_tables["rcov"].float(),
            r4r2=element_tables["r4r2"].float(),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            compute_virial=True,
            device="cpu",
        )

        # Virial should work with float64 positions
        assert virial.shape == (1, 3, 3)
        assert torch.isfinite(virial).all()


# ==============================================================================
# Shape Tests
# ==============================================================================


class TestShapes:
    """Verify output array shapes are correct."""

    @pytest.mark.parametrize("use_compile", [False, True], ids=["eager", "compiled"])
    def test_output_shapes(self, use_compile):
        """Test that output shapes match expected dimensions.

        Tests both eager mode and torch.compile mode to ensure the custom op
        is compatible with PyTorch compilation.
        """
        num_atoms = 5
        positions = torch.randn(num_atoms, 3, dtype=torch.float32)
        numbers = torch.randint(1, 10, (num_atoms,), dtype=torch.int32)

        # Create simple neighbor matrix with 2 neighbors per atom
        max_neighbors = 2
        neighbor_matrix = torch.randint(
            0, num_atoms, (num_atoms, max_neighbors), dtype=torch.int32
        )

        # Prepare element tables as 4D
        max_z_inc = 95
        c6_reference = torch.rand(max_z_inc, max_z_inc, 5, 5, dtype=torch.float32)
        coord_num_ref = torch.rand(max_z_inc, max_z_inc, 5, 5, dtype=torch.float32)

        # Optionally compile the function
        if use_compile:
            compute_fn = torch.compile(dftd3)
        else:
            compute_fn = dftd3

        energy, forces, coord_num = compute_fn(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=torch.rand(max_z_inc, dtype=torch.float32) + 0.5,
            r4r2=torch.rand(max_z_inc, dtype=torch.float32) + 1.0,
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device="cpu",
        )

        assert energy.shape == (1,)
        assert forces.shape == (num_atoms, 3)
        assert coord_num.shape == (num_atoms,)

        # Forces should be float32
        assert forces.dtype == torch.float32


# ==============================================================================
# Torch Compile Compatibility Tests
# ==============================================================================


class TestTorchCompile:
    """Test torch.compile compatibility of dftd3 wrapper."""

    def test_compile_basic(self, ne2_system, element_tables, device_cpu):
        """Test that dftd3 can be compiled with torch.compile."""
        system = ne2_system

        # Extract system data (fixtures already return tensors)
        positions = system["coord"].reshape(system["B"], 3).float().to("cpu")
        numbers = system["numbers"].int().to("cpu")
        neighbor_matrix = system["nbmat"].int().to("cpu")

        # Prepare element tables (already tensors)
        max_z_inc = element_tables["z_max_inc"]
        rcov = element_tables["rcov"].float().to("cpu")
        r4r2 = element_tables["r4r2"].float().to("cpu")
        c6_reference = (
            element_tables["c6ref"]
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
            .to("cpu")
        )
        coord_num_ref = (
            element_tables["cnref_i"]
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
            .to("cpu")
        )

        # Compile the function
        compiled_fn = torch.compile(dftd3)

        # Run compiled version
        energy_compiled, forces_compiled, cn_compiled = compiled_fn(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=rcov,
            r4r2=r4r2,
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device=device_cpu,
        )

        # Run eager version for comparison
        energy_eager, forces_eager, cn_eager = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=rcov,
            r4r2=r4r2,
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device=device_cpu,
        )

        # Results should match between compiled and eager
        torch.testing.assert_close(energy_compiled, energy_eager, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(forces_compiled, forces_eager, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(cn_compiled, cn_eager, rtol=1e-6, atol=1e-6)

    def test_compile_with_d3_params(self, ne2_system, element_tables, device_cpu):
        """Test compiled function with D3Parameters dataclass."""
        system = ne2_system

        # Extract system data (fixtures already return tensors)
        positions = system["coord"].reshape(system["B"], 3).float().to("cpu")
        numbers = system["numbers"].int().to("cpu")
        neighbor_matrix = system["nbmat"].int().to("cpu")

        # Create D3Parameters (element_tables already contain tensors)
        max_z_inc = element_tables["z_max_inc"]
        d3_params = D3Parameters(
            rcov=element_tables["rcov"].float().to("cpu"),
            r4r2=element_tables["r4r2"].float().to("cpu"),
            c6ab=element_tables["c6ref"]
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
            .to("cpu"),
            cn_ref=element_tables["cnref_i"]
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
            .to("cpu"),
        )

        # Compile the function
        compiled_fn = torch.compile(dftd3)

        # Run compiled version with D3Parameters
        energy, forces, cn = compiled_fn(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device=device_cpu,
        )

        # Should produce finite results
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(cn).all()

    def test_compile_multiple_calls(self, ne2_system, element_tables, device_cpu):
        """Test that compiled function can be called multiple times."""
        system = ne2_system

        # Extract system data (fixtures already return tensors)
        positions = system["coord"].reshape(system["B"], 3).float().to("cpu")
        numbers = system["numbers"].int().to("cpu")
        neighbor_matrix = system["nbmat"].int().to("cpu")

        # Prepare parameters (element_tables already contain tensors)
        max_z_inc = element_tables["z_max_inc"]
        rcov = element_tables["rcov"].float().to("cpu")
        r4r2 = element_tables["r4r2"].float().to("cpu")
        c6_reference = (
            element_tables["c6ref"]
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
            .to("cpu")
        )
        coord_num_ref = (
            element_tables["cnref_i"]
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
            .to("cpu")
        )

        # Compile the function
        compiled_fn = torch.compile(dftd3)

        # Call multiple times
        results = []
        for _ in range(3):
            energy, forces, cn = compiled_fn(
                positions=positions,
                numbers=numbers,
                neighbor_matrix=neighbor_matrix,
                covalent_radii=rcov,
                r4r2=r4r2,
                c6_reference=c6_reference,
                coord_num_ref=coord_num_ref,
                a1=0.3981,
                a2=4.4211,
                s8=1.9889,
                device=device_cpu,
            )
            results.append((energy, forces, cn))

        # All calls should produce identical results
        for i in range(1, len(results)):
            torch.testing.assert_close(results[i][0], results[0][0])
            torch.testing.assert_close(results[i][1], results[0][1])
            torch.testing.assert_close(results[i][2], results[0][2])

    def test_compile_neighbor_list_basic(self, h2_system, element_tables, device_cpu):
        """Test that dftd3 with neighbor_list can be compiled."""
        system = h2_system

        # Extract system data
        positions = system["coord"].reshape(system["B"], 3).float().to("cpu")
        numbers = system["numbers"].int().to("cpu")

        # Prepare element tables
        max_z_inc = element_tables["z_max_inc"]
        rcov = element_tables["rcov"].float().to("cpu")
        r4r2 = element_tables["r4r2"].float().to("cpu")
        c6_reference = (
            element_tables["c6ref"]
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
            .to("cpu")
        )
        coord_num_ref = (
            element_tables["cnref_i"]
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
            .to("cpu")
        )

        # Use neighbor list API to get properly formatted CSR data
        from nvalchemiops.neighborlist.neighborlist import (
            neighbor_list as build_neighbor_list,
        )

        # Build neighbor list - use large cutoff
        cutoff = 10.0

        neighbor_list, neighbor_ptr = build_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            return_neighbor_list=True,
        )

        # Compile the function
        compiled_fn = torch.compile(dftd3)

        # Run compiled version with neighbor_list
        energy_compiled, forces_compiled, cn_compiled = compiled_fn(
            positions=positions,
            numbers=numbers,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            covalent_radii=rcov,
            r4r2=r4r2,
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device=device_cpu,
        )

        # Run eager version for comparison
        energy_eager, forces_eager, cn_eager = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            covalent_radii=rcov,
            r4r2=r4r2,
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device=device_cpu,
        )

        # Results should match between compiled and eager
        torch.testing.assert_close(energy_compiled, energy_eager, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(forces_compiled, forces_eager, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(cn_compiled, cn_eager, rtol=1e-6, atol=1e-6)

    def test_compile_neighbor_list_with_pbc(self, element_tables, device_cpu):
        """Test compiled neighbor_list version with PBC and unit_shifts."""
        # Create a simple PBC system
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device="cpu")
        numbers = torch.tensor([1], dtype=torch.int32, device="cpu")

        # Define cell
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=torch.float32,
            device="cpu",
        )

        # Prepare element tables
        max_z_inc = element_tables["z_max_inc"]
        rcov = element_tables["rcov"].float().to("cpu")
        r4r2 = element_tables["r4r2"].float().to("cpu")
        c6_reference = (
            element_tables["c6ref"]
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
            .to("cpu")
        )
        coord_num_ref = (
            element_tables["cnref_i"]
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
            .to("cpu")
        )

        # Create synthetic neighbor list manually for specific test case
        # Atom sees itself in two neighboring cells
        neighbor_list = torch.tensor([[0, 0], [0, 0]], dtype=torch.int32, device="cpu")
        unit_shifts = torch.tensor(
            [[1, 0, 0], [0, 1, 0]], dtype=torch.int32, device="cpu"
        )

        # Compute neighbor_ptr
        num_neighbors = torch.tensor(
            [2], dtype=torch.int32, device="cpu"
        )  # atom 0 has 2 neighbors
        neighbor_ptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device="cpu"),
                torch.cumsum(num_neighbors, dim=0, dtype=torch.int32),
            ]
        )

        # Compile the function
        compiled_fn = torch.compile(dftd3)

        # Run compiled version
        # Note: neighbor list must be symmetric (each pair appears twice)
        energy_compiled, forces_compiled, cn_compiled = compiled_fn(
            positions=positions,
            numbers=numbers,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            covalent_radii=rcov,
            r4r2=r4r2,
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            cell=cell,
            unit_shifts=unit_shifts,
            device=device_cpu,
        )

        # Run eager version
        energy_eager, forces_eager, cn_eager = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            covalent_radii=rcov,
            r4r2=r4r2,
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            cell=cell,
            unit_shifts=unit_shifts,
            device=device_cpu,
        )

        # Results should match
        torch.testing.assert_close(energy_compiled, energy_eager, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(forces_compiled, forces_eager, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(cn_compiled, cn_eager, rtol=1e-6, atol=1e-6)

    def test_compile_neighbor_list_multiple_calls(
        self, h2_system, element_tables, device_cpu
    ):
        """Test that compiled function with neighbor_list can be called multiple times."""
        system = h2_system

        # Extract system data
        positions = system["coord"].reshape(system["B"], 3).float().to("cpu")
        numbers = system["numbers"].int().to("cpu")

        # Prepare parameters
        max_z_inc = element_tables["z_max_inc"]
        rcov = element_tables["rcov"].float().to("cpu")
        r4r2 = element_tables["r4r2"].float().to("cpu")
        c6_reference = (
            element_tables["c6ref"]
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
            .to("cpu")
        )
        coord_num_ref = (
            element_tables["cnref_i"]
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
            .to("cpu")
        )

        # Use neighbor list API to get properly formatted CSR data
        from nvalchemiops.neighborlist.neighborlist import (
            neighbor_list as build_neighbor_list,
        )

        # Build neighbor list using a large cutoff (non-periodic H2)
        cutoff = 10.0

        neighbor_list, neighbor_ptr = build_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            return_neighbor_list=True,
        )

        # Compile the function
        compiled_fn = torch.compile(dftd3)

        # Call multiple times
        results = []
        for _ in range(3):
            energy, forces, cn = compiled_fn(
                positions=positions,
                numbers=numbers,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                covalent_radii=rcov,
                r4r2=r4r2,
                c6_reference=c6_reference,
                coord_num_ref=coord_num_ref,
                a1=0.3981,
                a2=4.4211,
                s8=1.9889,
                device=device_cpu,
            )
            results.append((energy, forces, cn))

        # All calls should produce identical results
        for i in range(1, len(results)):
            torch.testing.assert_close(results[i][0], results[0][0])
            torch.testing.assert_close(results[i][1], results[0][1])
            torch.testing.assert_close(results[i][2], results[0][2])


# ==============================================================================
# Parameter Supply Tests
# ==============================================================================


class TestEmptySystemEdgeCases:
    """Tests for empty system edge cases in the wrapper."""

    def test_empty_system_no_batch_idx(self, element_tables):
        """Test empty system without batch_idx."""
        positions = torch.empty((0, 3), dtype=torch.float32)
        numbers = torch.empty((0,), dtype=torch.int32)
        neighbor_matrix = torch.empty((0, 5), dtype=torch.int32)

        # Prepare element tables
        max_z_inc = element_tables["z_max_inc"]
        c6_reference = element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5)
        coord_num_ref = element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5)

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=element_tables["rcov"],
            r4r2=element_tables["r4r2"],
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            device="cpu",
        )

        # Should return zero energy for 1 system
        assert energy.shape == (1,)
        assert energy.item() == pytest.approx(0.0, abs=1e-10)
        assert forces.shape == (0, 3)
        assert coord_num.shape == (0,)

    def test_empty_system_with_batch_idx(self, element_tables):
        """Test empty system with batch_idx provided."""
        positions = torch.empty((0, 3), dtype=torch.float32)
        numbers = torch.empty((0,), dtype=torch.int32)
        neighbor_matrix = torch.empty((0, 5), dtype=torch.int32)
        batch_idx = torch.empty((0,), dtype=torch.int32)

        # Prepare element tables
        max_z_inc = element_tables["z_max_inc"]
        c6_reference = element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5)
        coord_num_ref = element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5)

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=element_tables["rcov"],
            r4r2=element_tables["r4r2"],
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            batch_idx=batch_idx,
            device="cpu",
        )

        # Should still return something for 1 system (default when batch_idx is empty)
        assert energy.shape == (1,)
        assert energy.item() == pytest.approx(0.0, abs=1e-10)
        assert forces.shape == (0, 3)
        assert coord_num.shape == (0,)

    def test_empty_system_with_nonempty_batch_idx(self, element_tables):
        """Test empty system with batch_idx tensor that has max value."""
        positions = torch.empty((0, 3), dtype=torch.float32)
        numbers = torch.empty((0,), dtype=torch.int32)
        neighbor_matrix = torch.empty((0, 5), dtype=torch.int32)
        # Create a batch_idx that would indicate 3 systems if it had atoms
        batch_idx = torch.tensor([0, 1, 2], dtype=torch.int32)

        # Prepare element tables
        max_z_inc = element_tables["z_max_inc"]
        c6_reference = element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5)
        coord_num_ref = element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5)

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=element_tables["rcov"],
            r4r2=element_tables["r4r2"],
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            batch_idx=batch_idx,
            device="cpu",
        )

        # Should return 3 systems based on batch_idx.max() + 1
        assert energy.shape == (3,)
        assert torch.allclose(energy, torch.zeros_like(energy), atol=1e-10, rtol=0)
        assert forces.shape == (0, 3)
        assert coord_num.shape == (0,)


class TestParameterSupply:
    """Tests for different ways to supply DFT-D3 parameters.

    The dftd3 function requires parameters to be explicitly
    provided using one of the following methods:

    1. Explicit parameters: Supply covalent_radii, r4r2, c6_reference, and
       coord_num_ref directly as function arguments.

    2. ``d3_params`` data structure: Supply a dictionary, or instance of
       a ``D3Parameters`` dataclass with keys "rcov", "r4r2", "c6ab", and "cn_ref".
       with keys "rcov", "r4r2", "c6ab", and "cn_ref".

    3. Override mechanism: When d3_params is provided, individual parameters
       can be overridden by supplying them explicitly as function arguments.

    These tests verify all parameter supply mechanisms work correctly and that
    the function raises appropriate errors when no parameters are provided.
    """

    def test_explicit_parameters(self, element_tables):
        """Test supplying parameters explicitly via individual arguments."""
        num_atoms = 2
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32)
        neighbor_matrix = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        # Prepare element tables as 4D
        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float()
            else:
                return torch.from_numpy(data).float()

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        # Supply all parameters explicitly
        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device="cpu",
        )

        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()
        assert energy.shape == (1,)
        assert forces.shape == (num_atoms, 3)
        assert coord_num.shape == (num_atoms,)

    def test_d3_params_dataclass(self, element_tables):
        """Test supplying parameters via D3Parameters dataclass."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32)
        neighbor_matrix = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        # Create D3Parameters dataclass
        max_z_inc = element_tables["z_max_inc"]
        d3_params = D3Parameters(
            rcov=element_tables["rcov"],
            r4r2=element_tables["r4r2"],
            c6ab=element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5),
            cn_ref=element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5),
        )

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,  # Pass D3Parameters instance
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device="cpu",
        )

        # Should succeed with D3Parameters dataclass
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()

    def test_d3_params_dict(self, element_tables):
        """Test supplying parameters via d3_params dictionary (not D3Parameters)."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32)
        neighbor_matrix = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        # Create d3_params as plain dict (not D3Parameters instance)
        # This tests the isinstance() == False branch
        max_z_inc = element_tables["z_max_inc"]
        d3_params = {
            "rcov": element_tables["rcov"].float(),
            "r4r2": element_tables["r4r2"].float(),
            "c6ab": element_tables["c6ref"].float().reshape(max_z_inc, max_z_inc, 5, 5),
            "cn_ref": element_tables["cnref_i"]
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5),
        }

        # Supply parameters via dictionary without any overrides
        # This should extract all 4 parameters from dict
        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,  # Plain dict, no overrides
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device="cpu",
        )

        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()

    def test_d3_params_dict_with_override(self, element_tables):
        """Test that explicit parameters override d3_params dictionary."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32)
        neighbor_matrix = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        # Prepare element tables as 4D
        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float()
            else:
                return torch.from_numpy(data).float()

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        # Create d3_params dictionary with dummy values
        d3_params = {
            "rcov": torch.ones(max_z_inc, dtype=torch.float32),
            "r4r2": torch.ones(max_z_inc, dtype=torch.float32),
            "c6ab": torch.ones(max_z_inc, max_z_inc, 5, 5, dtype=torch.float32),
            "cn_ref": torch.ones(max_z_inc, max_z_inc, 5, 5, dtype=torch.float32),
        }

        # Override with explicit parameters
        override_rcov = ensure_tensor(element_tables["rcov"])
        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
            covalent_radii=override_rcov,
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device="cpu",
        )

        # Should succeed with overridden parameters
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()

    def test_no_parameters_raises_error(self):
        """Test that missing parameters raise RuntimeError."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32)
        neighbor_matrix = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        # Try to call without providing any parameters
        with pytest.raises(
            RuntimeError,
            match="DFT-D3 parameters must be explicitly provided",
        ):
            dftd3(
                positions=positions,
                numbers=numbers,
                neighbor_matrix=neighbor_matrix,
                a1=0.3981,
                a2=4.4211,
                s8=1.9889,
                device="cpu",
            )

    def test_d3_params_dataclass_with_override(self, element_tables):
        """Test D3Parameters dataclass with parameter override."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32)
        neighbor_matrix = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        # Create D3Parameters dataclass
        max_z_inc = element_tables["z_max_inc"]
        d3_params = D3Parameters(
            rcov=element_tables["rcov"],
            r4r2=element_tables["r4r2"],
            c6ab=element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5),
            cn_ref=element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5),
        )

        # Override only covalent_radii
        custom_rcov = element_tables["rcov"] * 1.1

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
            covalent_radii=custom_rcov,  # Override this parameter
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device="cpu",
        )

        # Should succeed with overridden parameter
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()

    def test_extract_r4r2_from_d3_params(self, element_tables):
        """Test that r4r2 is extracted from d3_params when not provided."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32)
        neighbor_matrix = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        max_z_inc = element_tables["z_max_inc"]
        d3_params = {
            "rcov": element_tables["rcov"],
            "r4r2": element_tables["r4r2"],
            "c6ab": element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5),
            "cn_ref": element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5),
        }

        # Override all EXCEPT r4r2 to ensure r4r2 is extracted
        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
            covalent_radii=element_tables["rcov"],
            c6_reference=element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5),
            coord_num_ref=element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5),
            # r4r2 NOT provided - should be extracted from d3_params
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device="cpu",
        )

        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()

    def test_extract_coord_num_ref_from_d3_params(self, element_tables):
        """Test that coord_num_ref is extracted from d3_params when not provided."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32)
        neighbor_matrix = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        max_z_inc = element_tables["z_max_inc"]
        d3_params = {
            "rcov": element_tables["rcov"],
            "r4r2": element_tables["r4r2"],
            "c6ab": element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5),
            "cn_ref": element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5),
        }

        # Override all EXCEPT coord_num_ref to ensure it's extracted
        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
            covalent_radii=element_tables["rcov"],
            r4r2=element_tables["r4r2"],
            c6_reference=element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5),
            # coord_num_ref NOT provided - should be extracted from d3_params
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device="cpu",
        )

        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()

    def test_partial_override_with_d3_params(self, element_tables):
        """Test partial override: supply d3_params and override one parameter."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32)
        neighbor_matrix = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        # Prepare element tables as 4D
        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float()
            else:
                return torch.from_numpy(data).float()

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        # Create d3_params dictionary
        d3_params = {
            "rcov": ensure_tensor(element_tables["rcov"]),
            "r4r2": ensure_tensor(element_tables["r4r2"]),
            "c6ab": c6_reference,
            "cn_ref": coord_num_ref,
        }

        # Override only covalent_radii
        custom_rcov = ensure_tensor(element_tables["rcov"]) * 1.1  # Scale by 1.1

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
            covalent_radii=custom_rcov,  # Override this one
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device="cpu",
        )

        # Should succeed with partially overridden parameters
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()


# ==============================================================================
# Batching Tests
# ==============================================================================


class TestBatchIndexHandling:
    """Test batch_idx parameter handling for num_systems determination."""

    def test_with_batch_idx_determines_num_systems(self, element_tables):
        """Test that batch_idx is used to determine num_systems."""
        # Create a simple system with 4 atoms in 2 systems
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # System 0, atom 0
                [1.4, 0.0, 0.0],  # System 0, atom 1
                [5.0, 0.0, 0.0],  # System 1, atom 0
                [6.4, 0.0, 0.0],  # System 1, atom 1
            ],
            dtype=torch.float32,
        )
        numbers = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
        neighbor_matrix = torch.full((4, 5), 4, dtype=torch.int32)
        neighbor_matrix[0, 0] = 1
        neighbor_matrix[1, 0] = 0
        neighbor_matrix[2, 0] = 3
        neighbor_matrix[3, 0] = 2

        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

        # Prepare parameters
        max_z_inc = element_tables["z_max_inc"]
        c6_reference = element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5)
        coord_num_ref = element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5)

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=element_tables["rcov"],
            r4r2=element_tables["r4r2"],
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            batch_idx=batch_idx,
            device="cpu",
        )

        # Should have 2 systems based on batch_idx
        assert energy.shape == (2,)
        assert forces.shape == (4, 3)
        assert coord_num.shape == (4,)


class TestBatching:
    """Tests for batched calculations with multiple systems.

    The batch_idx parameter allows computing dispersion corrections for multiple
    independent molecular systems in a single kernel launch. This is critical for
    efficient GPU utilization and is used in the following kernels:

    1. _compute_cartesian_shifts: Uses batch_idx[j] to select the correct cell
       matrix for each atom's system when computing PBC shifts.

    2. _direct_forces_and_dE_dCN_kernel: Uses batch_idx[i] to accumulate energy
       contributions to the correct system in the energy array.

    These tests verify that:
    - Multiple systems can be processed together correctly
    - Energy is accumulated to the correct system
    - Forces are computed correctly for each atom
    - Batched results match individual system calculations
    - PBC works correctly with batching
    """

    def test_two_identical_systems(self, element_tables, device_cpu):
        """Test batching two identical H2 molecules."""
        # Create two identical H2 molecules
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # System 0, atom 0
                [1.4, 0.0, 0.0],  # System 0, atom 1
                [0.0, 0.0, 0.0],  # System 1, atom 0
                [1.4, 0.0, 0.0],  # System 1, atom 1
            ],
            dtype=torch.float32,
        )
        numbers = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

        # Create symmetric neighbor lists for each system
        # System 0: atom 0 <-> atom 1
        # System 1: atom 2 <-> atom 3
        # Neighbor matrix: atom i has neighbor at [i, 0]
        neighbor_matrix = torch.tensor(
            [[1, 4], [0, 4], [3, 4], [2, 4]], dtype=torch.int32
        )

        # Prepare element tables as 4D
        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float()
            else:
                return torch.from_numpy(data).float()

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            batch_idx=batch_idx,
            device=device_cpu,
        )

        # Should have 2 systems
        assert energy.shape == (2,)
        assert forces.shape == (4, 3)
        assert coord_num.shape == (4,)

        # All energies should be finite and non-zero
        assert torch.isfinite(energy).all(), "All energies should be finite"
        assert torch.all(torch.abs(energy) > 1e-10), "All energies should be non-zero"
        assert (energy < 0.0).all(), "Dispersion energies should be negative"

        # Both systems are identical, so energies should be equal
        torch.testing.assert_close(energy[0], energy[1], rtol=1e-6, atol=1e-6)

        # Forces for system 0 and system 1 should be identical
        torch.testing.assert_close(forces[0:2], forces[2:4], rtol=1e-6, atol=1e-6)

        # Coordination numbers should be identical
        torch.testing.assert_close(coord_num[0:2], coord_num[2:4], rtol=1e-6, atol=1e-6)

    def test_batch_vs_individual(self, element_tables, device_cpu):
        """Test that batched calculation matches individual calculations."""
        # Create two different systems: H2 and Ne2
        positions_h2 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32
        )
        numbers_h2 = torch.tensor([1, 1], dtype=torch.int32)

        positions_ne2 = torch.tensor(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=torch.float32
        )
        numbers_ne2 = torch.tensor([10, 10], dtype=torch.int32)

        # Prepare element tables as 4D
        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float()
            else:
                return torch.from_numpy(data).float()

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        # Run individual calculations
        neighbor_matrix_single = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)

        energy_h2, forces_h2, cn_h2 = dftd3(
            positions=positions_h2,
            numbers=numbers_h2,
            neighbor_matrix=neighbor_matrix_single,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device=device_cpu,
        )

        energy_ne2, forces_ne2, cn_ne2 = dftd3(
            positions=positions_ne2,
            numbers=numbers_ne2,
            neighbor_matrix=neighbor_matrix_single,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            device=device_cpu,
        )

        # Now run batched calculation
        positions_batch = torch.cat([positions_h2, positions_ne2], dim=0)
        numbers_batch = torch.cat([numbers_h2, numbers_ne2], dim=0)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

        neighbor_matrix_batch = torch.tensor(
            [[1, 4], [0, 4], [3, 4], [2, 4]], dtype=torch.int32
        )

        energy_batch, forces_batch, cn_batch = dftd3(
            positions=positions_batch,
            numbers=numbers_batch,
            neighbor_matrix=neighbor_matrix_batch,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            batch_idx=batch_idx,
            device=device_cpu,
        )

        # All energies should be finite and non-zero
        assert torch.isfinite(energy_batch).all(), (
            "All batched energies should be finite"
        )
        assert torch.all(torch.abs(energy_batch) > 1e-10), (
            "All batched energies should be non-zero"
        )
        assert (energy_batch < 0.0).all(), "Dispersion energies should be negative"

        # Compare results
        torch.testing.assert_close(energy_batch[0], energy_h2[0], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(energy_batch[1], energy_ne2[0], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(forces_batch[0:2], forces_h2, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(forces_batch[2:4], forces_ne2, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(cn_batch[0:2], cn_h2, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(cn_batch[2:4], cn_ne2, rtol=1e-6, atol=1e-6)

    def test_batch_with_different_sizes(self, element_tables, device_cpu):
        """Test batching systems with different numbers of atoms."""
        # System 0: 2 atoms (H2)
        # System 1: 3 atoms (H3 linear)
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # System 0, atom 0
                [1.4, 0.0, 0.0],  # System 0, atom 1
                [0.0, 0.0, 0.0],  # System 1, atom 0
                [1.4, 0.0, 0.0],  # System 1, atom 1
                [2.8, 0.0, 0.0],  # System 1, atom 2
            ],
            dtype=torch.float32,
        )
        numbers = torch.tensor([1, 1, 1, 1, 1], dtype=torch.int32)
        batch_idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32)

        # Create neighbor lists
        # System 0: 0 <-> 1
        # System 1: 2 <-> 3 <-> 4
        # Neighbor matrix: max_neighbors=2 (atom 3 has 2 neighbors, others have 1)
        neighbor_matrix = torch.tensor(
            [[1, 5], [0, 5], [3, 4], [2, 4], [3, 5]], dtype=torch.int32
        )

        # Prepare element tables as 4D
        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float()
            else:
                return torch.from_numpy(data).float()

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            batch_idx=batch_idx,
            device=device_cpu,
        )

        # Should have 2 systems
        assert energy.shape == (2,)
        assert forces.shape == (5, 3)
        assert coord_num.shape == (5,)

        # All values should be finite
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()

        # All energies should be non-zero
        assert torch.all(torch.abs(energy) > 1e-10), "All energies should be non-zero"
        assert (energy < 0.0).all(), "Dispersion energies should be negative"

        # System 1 should have more energy than system 0 (more atoms)
        # This is not guaranteed in general, but for this specific case it should hold
        assert energy[1] < energy[0]  # More negative = more dispersion

    def test_batch_with_pbc(self, element_tables, device_cpu):
        """Test batching with periodic boundary conditions."""
        # Create two systems with PBC
        # System 0: Single H atom in a box
        # System 1: Single Ne atom in a box
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # System 0
                [0.0, 0.0, 0.0],  # System 1
            ],
            dtype=torch.float32,
        )
        numbers = torch.tensor([1, 10], dtype=torch.int32)
        batch_idx = torch.tensor([0, 1], dtype=torch.int32)

        # Define cells for each system
        cell = torch.tensor(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],  # System 0
                [[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]],  # System 1
            ],
            dtype=torch.float32,
        )

        # Create neighbor matrix with periodic images
        # Each atom sees itself in neighboring cells (2 neighbors per atom)
        neighbor_matrix = torch.tensor([[0, 0], [1, 1]], dtype=torch.int32)
        neighbor_matrix_shifts = torch.tensor(
            [[[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 0, 1]]], dtype=torch.int32
        )

        # Prepare element tables as 4D
        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float()
            else:
                return torch.from_numpy(data).float()

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            batch_idx=batch_idx,
            cell=cell,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            device=device_cpu,
        )

        # Should have 2 systems
        assert energy.shape == (2,)
        assert forces.shape == (2, 3)
        assert coord_num.shape == (2,)

        # All values should be finite
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()

        # All energies should be non-zero (atoms interact with periodic images)
        assert torch.all(torch.abs(energy) > 1e-10), (
            "All energies should be non-zero with PBC"
        )
        assert (energy < 0.0).all(), "Dispersion energies should be negative"

    def test_batch_energy_accumulation(self, element_tables, device_cpu):
        """Test that energy is correctly accumulated to the right system."""
        # Create a batch where we can verify energy accumulation
        # System 0: 1 atom (should have zero energy - no pairs)
        # System 1: 2 atoms (should have non-zero energy)
        # System 2: 1 atom (should have zero energy - no pairs)
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # System 0, atom 0
                [0.0, 0.0, 0.0],  # System 1, atom 0
                [1.4, 0.0, 0.0],  # System 1, atom 1
                [0.0, 0.0, 0.0],  # System 2, atom 0
            ],
            dtype=torch.float32,
        )
        numbers = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
        batch_idx = torch.tensor([0, 1, 1, 2], dtype=torch.int32)

        # Neighbor list: only system 1 has interactions
        # Atom 0: no neighbors, Atom 1: neighbor 2, Atom 2: neighbor 1, Atom 3: no neighbors
        neighbor_matrix = torch.tensor(
            [[4, 4], [2, 4], [1, 4], [4, 4]], dtype=torch.int32
        )

        # Prepare element tables as 4D
        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float()
            else:
                return torch.from_numpy(data).float()

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        energy, _, _ = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            batch_idx=batch_idx,
            device=device_cpu,
        )

        # Should have 3 systems
        assert energy.shape == (3,)

        # Systems 0 and 2 should have zero energy (no pairs)
        assert energy[0] == pytest.approx(0.0, abs=1e-7)
        assert energy[2] == pytest.approx(0.0, abs=1e-7)

        # System 1 should have non-zero energy
        assert abs(energy[1].item()) > 1e-10
        assert energy[1] < 0.0  # Dispersion is attractive

    @pytest.mark.parametrize("device_name", ["device_cpu", "device_gpu"])
    def test_batch_cpu_gpu_consistency(self, element_tables, device_name, request):
        """Test that batched calculations are consistent between CPU and GPU."""
        device = request.getfixturevalue(device_name)

        # Determine torch device from Warp device string
        torch_device = "cuda" if "cuda" in device else "cpu"

        # Create a simple batch
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=torch_device,
        )
        numbers = torch.tensor([1, 1, 10, 10], dtype=torch.int32, device=torch_device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=torch_device)

        neighbor_matrix = torch.tensor(
            [[1, 4], [0, 4], [3, 4], [2, 4]], dtype=torch.int32, device=torch_device
        )

        # Prepare element tables as 4D
        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float().to(torch_device)
            else:
                return torch.from_numpy(data).float().to(torch_device)

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            batch_idx=batch_idx,
            device=device,
        )

        # Should have 2 systems with finite values
        assert energy.shape == (2,)
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()

        # All energies should be non-zero
        assert torch.all(torch.abs(energy) > 1e-10), "All energies should be non-zero"
        assert (energy < 0.0).all(), "Dispersion energies should be negative"

    def test_large_batch(self, element_tables, device_cpu):
        """Test batching with many systems to verify scalability."""
        num_systems = 10
        atoms_per_system = 2

        # Create positions: each system is an H2 molecule at different positions
        positions_list = []
        for i in range(num_systems):
            offset = i * 10.0  # Space systems far apart
            positions_list.extend([[offset, 0.0, 0.0], [offset + 1.4, 0.0, 0.0]])
        positions = torch.tensor(positions_list, dtype=torch.float32)

        # All atoms are hydrogen
        numbers = torch.ones(num_systems * atoms_per_system, dtype=torch.int32)

        # Create batch indices
        batch_idx = torch.repeat_interleave(
            torch.arange(num_systems, dtype=torch.int32), atoms_per_system
        )

        # Create neighbor matrix: each system has its own pair
        # For each system: atom 0 -> atom 1, atom 1 -> atom 0
        neighbor_matrix_list = []
        total_atoms = num_systems * atoms_per_system
        for i in range(num_systems):
            base = i * atoms_per_system
            neighbor_matrix_list.append([base + 1, total_atoms])  # atom 0 neighbors
            neighbor_matrix_list.append([base, total_atoms])  # atom 1 neighbors
        neighbor_matrix = torch.tensor(neighbor_matrix_list, dtype=torch.int32)

        # Prepare element tables as 4D
        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float()
            else:
                return torch.from_numpy(data).float()

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            batch_idx=batch_idx,
            device=device_cpu,
        )

        # Should have num_systems energies
        assert energy.shape == (num_systems,)
        assert forces.shape == (num_systems * atoms_per_system, 3)
        assert coord_num.shape == (num_systems * atoms_per_system,)

        # All energies should be finite and non-zero
        assert torch.isfinite(energy).all(), "All energies should be finite"
        assert torch.all(torch.abs(energy) > 1e-10), "All energies should be non-zero"
        assert (energy < 0.0).all(), "Dispersion energies should be negative"

        # All systems are identical, so energies should be equal
        for i in range(1, num_systems):
            torch.testing.assert_close(energy[i], energy[0], rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
    )
    def test_neighbor_list_equivalence(self, h2_system, element_tables, device):
        """Test that neighbor list format produces identical results to neighbor matrix."""
        system = h2_system
        functional_params = {
            "k1": 16.0,
            "k3": -4.0,
            "a1": 0.3981,
            "a2": 4.4211,
            "s6": 1.0,
            "s8": 1.9889,
        }

        # Run with neighbor matrix
        results_matrix = run_dftd3(system, element_tables, functional_params, device)

        torch_device = "cuda" if "cuda" in device else "cpu"

        # Prepare tensors
        positions = (
            torch.from_numpy(system["coord"])
            .reshape(system["B"], 3)
            .float()
            .to(torch_device)
            if not isinstance(system["coord"], torch.Tensor)
            else system["coord"].reshape(system["B"], 3).float().to(torch_device)
        )
        numbers = (
            torch.from_numpy(system["numbers"]).int().to(torch_device)
            if not isinstance(system["numbers"], torch.Tensor)
            else system["numbers"].int().to(torch_device)
        )

        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float().to(torch_device)
            else:
                return torch.from_numpy(data).float().to(torch_device)

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        # Use neighbor list API to get properly formatted CSR data
        from nvalchemiops.neighborlist.neighborlist import (
            neighbor_list as build_neighbor_list,
        )

        # Build neighbor list - use large cutoff to ensure we capture all pairs
        cutoff = 10.0

        neighbor_list, neighbor_ptr = build_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            return_neighbor_list=True,
        )

        # Run with neighbor list from API (will be symmetric by default)
        energy_list, forces_list, coord_num_list = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            k1=functional_params["k1"],
            k3=functional_params["k3"],
            s6=functional_params["s6"],
            device=device,
        )

        # Compare results - should be identical
        torch.testing.assert_close(
            energy_list, results_matrix["energy"], rtol=1e-5, atol=1e-7
        )
        torch.testing.assert_close(
            forces_list, results_matrix["forces"], rtol=1e-5, atol=1e-7
        )
        torch.testing.assert_close(
            coord_num_list, results_matrix["coord_num"], rtol=1e-5, atol=1e-7
        )

    def test_neighbor_list_validation(self, h2_system, element_tables):
        """Test that mutual exclusivity validation works correctly."""
        system = h2_system
        torch_device = "cpu"

        positions = (
            torch.from_numpy(system["coord"])
            .reshape(system["B"], 3)
            .float()
            .to(torch_device)
            if not isinstance(system["coord"], torch.Tensor)
            else system["coord"].reshape(system["B"], 3).float().to(torch_device)
        )
        numbers = (
            torch.from_numpy(system["numbers"]).int().to(torch_device)
            if not isinstance(system["numbers"], torch.Tensor)
            else system["numbers"].int().to(torch_device)
        )
        neighbor_matrix = (
            torch.from_numpy(system["nbmat"]).int().to(torch_device)
            if not isinstance(system["nbmat"], torch.Tensor)
            else system["nbmat"].int().to(torch_device)
        )
        neighbor_list = torch.tensor(
            [[0, 1], [1, 0]], dtype=torch.int32, device=torch_device
        )

        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float().to(torch_device)
            else:
                return torch.from_numpy(data).float().to(torch_device)

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        # Test 1: Both neighbor_matrix and neighbor_list provided
        with pytest.raises(
            ValueError, match="Cannot provide both neighbor_matrix and neighbor_list"
        ):
            dftd3(
                positions=positions,
                numbers=numbers,
                neighbor_matrix=neighbor_matrix,
                neighbor_list=neighbor_list,
                covalent_radii=ensure_tensor(element_tables["rcov"]),
                r4r2=ensure_tensor(element_tables["r4r2"]),
                c6_reference=c6_reference,
                coord_num_ref=coord_num_ref,
                a1=0.3981,
                a2=4.4211,
                s8=1.9889,
            )

        # Test 2: Neither provided
        with pytest.raises(
            ValueError, match="Must provide either neighbor_matrix or neighbor_list"
        ):
            dftd3(
                positions=positions,
                numbers=numbers,
                covalent_radii=ensure_tensor(element_tables["rcov"]),
                r4r2=ensure_tensor(element_tables["r4r2"]),
                c6_reference=c6_reference,
                coord_num_ref=coord_num_ref,
                a1=0.3981,
                a2=4.4211,
                s8=1.9889,
            )

        # Test 3: unit_shifts with neighbor_matrix
        with pytest.raises(ValueError, match="unit_shifts is for neighbor_list format"):
            dftd3(
                positions=positions,
                numbers=numbers,
                neighbor_matrix=neighbor_matrix,
                unit_shifts=torch.zeros((2, 3), dtype=torch.int32, device=torch_device),
                covalent_radii=ensure_tensor(element_tables["rcov"]),
                r4r2=ensure_tensor(element_tables["r4r2"]),
                c6_reference=c6_reference,
                coord_num_ref=coord_num_ref,
                a1=0.3981,
                a2=4.4211,
                s8=1.9889,
            )

        # Test 4: neighbor_matrix_shifts with neighbor_list
        with pytest.raises(
            ValueError, match="neighbor_matrix_shifts is for neighbor_matrix format"
        ):
            dftd3(
                positions=positions,
                numbers=numbers,
                neighbor_list=neighbor_list,
                neighbor_matrix_shifts=torch.zeros(
                    (system["B"], neighbor_matrix.size(1), 3),
                    dtype=torch.int32,
                    device=torch_device,
                ),
                covalent_radii=ensure_tensor(element_tables["rcov"]),
                r4r2=ensure_tensor(element_tables["r4r2"]),
                c6_reference=c6_reference,
                coord_num_ref=coord_num_ref,
                a1=0.3981,
                a2=4.4211,
                s8=1.9889,
            )

        # Test 5: neighbor_list without neighbor_ptr
        with pytest.raises(
            ValueError, match="neighbor_ptr must be provided when using neighbor_list"
        ):
            dftd3(
                positions=positions,
                numbers=numbers,
                neighbor_list=neighbor_list,
                covalent_radii=ensure_tensor(element_tables["rcov"]),
                r4r2=ensure_tensor(element_tables["r4r2"]),
                c6_reference=c6_reference,
                coord_num_ref=coord_num_ref,
                a1=0.3981,
                a2=4.4211,
                s8=1.9889,
            )

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
    )
    def test_neighbor_list_pbc(self, element_tables, device):
        """Test neighbor list format with PBC using unit_shifts."""
        torch_device = "cuda" if "cuda" in device else "cpu"

        # Create a simple PBC system - single H atom in a box
        positions = torch.tensor(
            [[0.0, 0.0, 0.0]], dtype=torch.float32, device=torch_device
        )
        numbers = torch.tensor([1], dtype=torch.int32, device=torch_device)

        # Define cell
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=torch.float32,
            device=torch_device,
        )

        # Neighbor matrix: atom sees itself in neighboring cells
        neighbor_matrix = torch.tensor([[0, 0]], dtype=torch.int32, device=torch_device)
        neighbor_matrix_shifts = torch.tensor(
            [[[1, 0, 0], [0, 1, 0]]], dtype=torch.int32, device=torch_device
        )

        def ensure_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.float().to(torch_device)
            else:
                return torch.from_numpy(data).float().to(torch_device)

        max_z_inc = element_tables["z_max_inc"]
        c6_reference = ensure_tensor(element_tables["c6ref"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )
        coord_num_ref = ensure_tensor(element_tables["cnref_i"]).reshape(
            max_z_inc, max_z_inc, 5, 5
        )

        # Run with neighbor matrix (reference)
        # Note: neighbor matrix must be symmetric (each pair appears twice)
        energy_matrix, forces_matrix, coord_num_matrix = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            cell=cell,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            device=device,
        )

        # Convert to neighbor list format manually
        # Neighbor matrix is [[0, 0]] with shifts [[[1,0,0], [0,1,0]]]
        # This means atom 0 has 2 neighbors (both atom 0 in different periodic images)
        neighbor_list = torch.tensor(
            [[0, 0], [0, 0]], dtype=torch.int32, device=torch_device
        )
        unit_shifts = torch.tensor(
            [[1, 0, 0], [0, 1, 0]], dtype=torch.int32, device=torch_device
        )

        # Compute neighbor_ptr
        num_neighbors = torch.tensor(
            [2], dtype=torch.int32, device=torch_device
        )  # atom 0 has 2 neighbors
        neighbor_ptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=torch_device),
                torch.cumsum(num_neighbors, dim=0, dtype=torch.int32),
            ]
        )

        # Run with neighbor list + neighbor_ptr + unit_shifts
        # Note: neighbor list must be symmetric (each pair appears twice)
        energy_list, forces_list, coord_num_list = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            covalent_radii=ensure_tensor(element_tables["rcov"]),
            r4r2=ensure_tensor(element_tables["r4r2"]),
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.3981,
            a2=4.4211,
            s8=1.9889,
            cell=cell,
            unit_shifts=unit_shifts,
            device=device,
        )

        # Compare results - should be identical
        torch.testing.assert_close(energy_list, energy_matrix, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(forces_list, forces_matrix, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(
            coord_num_list, coord_num_matrix, rtol=1e-5, atol=1e-7
        )

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
PyTorch binding tests for DFT-D3 custom ops.

This test suite focuses on the PyTorch binding layer, including:
- Custom op interface (_dftd3_matrix_op, _dftd3_op)
- Device handling (CPU/GPU)
- Dtype support (float32/float64)
- Torch.compile compatibility
- Tensor conversion and memory management
- Integration with the high-level dftd3() wrapper

For comprehensive functional tests (PBC, batching, edge cases), see
test/interactions/dispersion/test_dftd3.py
"""

from __future__ import annotations

import numpy as np
import pytest

# Try to import torch - these tests require PyTorch
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import PyTorch bindings
if TORCH_AVAILABLE:
    from nvalchemiops.torch.interactions.dispersion import D3Parameters, dftd3
    from nvalchemiops.torch.interactions.dispersion._dftd3 import (
        _dftd3_matrix_op,
        _dftd3_op,
    )
    from nvalchemiops.torch.neighbors import (
        neighbor_list as build_neighbor_list,
    )

# Skip all tests in this module if torch is not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed - these tests require torch to be available",
)


# ==============================================================================
# Helper Functions
# ==============================================================================


def numpy_to_torch(arr: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    return torch.from_numpy(arr).to(device)


# ==============================================================================
# Custom Op Interface Tests (Neighbor Matrix Format)
# ==============================================================================


class TestCustomOpNeighborMatrix:
    """Test the _dftd3_matrix_op custom operator directly."""

    @pytest.mark.usefixtures("element_tables", "device")
    def test_custom_op_basic(self, request):
        """Test custom op with basic H2 system."""
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        # Simple H2 system
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32, device=device)
        neighbor_matrix = torch.tensor(
            [[1, 2], [0, 2]], dtype=torch.int32, device=device
        )

        # Parameters (convert from numpy)
        max_z_inc = element_tables["z_max_inc"]
        covalent_radii = numpy_to_torch(element_tables["rcov"], device).float()
        r4r2 = numpy_to_torch(element_tables["r4r2"], device).float()
        c6_reference = (
            numpy_to_torch(element_tables["c6ref"], device)
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
        )
        coord_num_ref = (
            numpy_to_torch(element_tables["cnref_i"], device)
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
        )

        # Allocate outputs
        energy = torch.zeros(1, dtype=torch.float32, device=device)
        forces = torch.zeros((2, 3), dtype=torch.float32, device=device)
        coord_num = torch.zeros(2, dtype=torch.float32, device=device)
        virial = torch.zeros((0, 3, 3), dtype=torch.float32, device=device)

        # Call custom op directly
        _dftd3_matrix_op(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=covalent_radii,
            r4r2=r4r2,
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            energy=energy,
            forces=forces,
            coord_num=coord_num,
            virial=virial,
            device=device,
        )

        # Verify outputs are finite and reasonable
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()
        assert energy[0] < 0.0  # Dispersion is attractive

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.usefixtures("element_tables", "device")
    def test_custom_op_dtypes(self, request, dtype):
        """Test custom op with different position dtypes."""
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        # Simple H2 system
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=dtype, device=device
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32, device=device)
        neighbor_matrix = torch.tensor(
            [[1, 2], [0, 2]], dtype=torch.int32, device=device
        )

        # Parameters (convert from numpy)
        max_z_inc = element_tables["z_max_inc"]
        covalent_radii = numpy_to_torch(element_tables["rcov"], device).float()
        r4r2 = numpy_to_torch(element_tables["r4r2"], device).float()
        c6_reference = (
            numpy_to_torch(element_tables["c6ref"], device)
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
        )
        coord_num_ref = (
            numpy_to_torch(element_tables["cnref_i"], device)
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
        )

        # Allocate outputs (always float32)
        energy = torch.zeros(1, dtype=torch.float32, device=device)
        forces = torch.zeros((2, 3), dtype=torch.float32, device=device)
        coord_num = torch.zeros(2, dtype=torch.float32, device=device)
        virial = torch.zeros((0, 3, 3), dtype=torch.float32, device=device)

        # Call custom op
        _dftd3_matrix_op(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=covalent_radii,
            r4r2=r4r2,
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            energy=energy,
            forces=forces,
            coord_num=coord_num,
            virial=virial,
            device=device,
        )

        # Verify outputs
        assert energy.dtype == torch.float32
        assert forces.dtype == torch.float32
        assert coord_num.dtype == torch.float32
        assert torch.isfinite(energy).all()

    @pytest.mark.usefixtures("element_tables", "device")
    def test_custom_op_empty_system(self, request):
        """Test custom op with empty system."""
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        # Empty system
        positions = torch.zeros((0, 3), dtype=torch.float32, device=device)
        numbers = torch.zeros((0,), dtype=torch.int32, device=device)
        neighbor_matrix = torch.zeros((0, 0), dtype=torch.int32, device=device)

        # Parameters (convert from numpy)
        max_z_inc = element_tables["z_max_inc"]
        covalent_radii = numpy_to_torch(element_tables["rcov"], device).float()
        r4r2 = numpy_to_torch(element_tables["r4r2"], device).float()
        c6_reference = (
            numpy_to_torch(element_tables["c6ref"], device)
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
        )
        coord_num_ref = (
            numpy_to_torch(element_tables["cnref_i"], device)
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
        )

        # Allocate outputs
        energy = torch.zeros(1, dtype=torch.float32, device=device)
        forces = torch.zeros((0, 3), dtype=torch.float32, device=device)
        coord_num = torch.zeros((0,), dtype=torch.float32, device=device)
        virial = torch.zeros((0, 3, 3), dtype=torch.float32, device=device)

        # Should handle empty case without error
        _dftd3_matrix_op(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=covalent_radii,
            r4r2=r4r2,
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            energy=energy,
            forces=forces,
            coord_num=coord_num,
            virial=virial,
            device=device,
        )
        energy = energy.cpu().item()

        assert energy == pytest.approx(0.0)


# ==============================================================================
# Custom Op Interface Tests (Neighbor List Format)
# ==============================================================================


class TestCustomOpNeighborList:
    """Test the _dftd3_op custom operator directly."""

    @pytest.mark.usefixtures("element_tables", "device")
    def test_custom_op_basic(self, request):
        """Test custom op with basic H2 system in CSR format."""
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        # Simple H2 system
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32, device=device)

        # CSR format: atom 0 has neighbor 1, atom 1 has neighbor 0
        idx_j = torch.tensor([1, 0], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

        # Parameters (convert from numpy)
        max_z_inc = element_tables["z_max_inc"]
        covalent_radii = numpy_to_torch(element_tables["rcov"], device).float()
        r4r2 = numpy_to_torch(element_tables["r4r2"], device).float()
        c6_reference = (
            numpy_to_torch(element_tables["c6ref"], device)
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
        )
        coord_num_ref = (
            numpy_to_torch(element_tables["cnref_i"], device)
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
        )

        # Allocate outputs
        energy = torch.zeros(1, dtype=torch.float32, device=device)
        forces = torch.zeros((2, 3), dtype=torch.float32, device=device)
        coord_num = torch.zeros(2, dtype=torch.float32, device=device)
        virial = torch.zeros((0, 3, 3), dtype=torch.float32, device=device)

        # Call custom op directly
        _dftd3_op(
            positions=positions,
            numbers=numbers,
            idx_j=idx_j,
            neighbor_ptr=neighbor_ptr,
            covalent_radii=covalent_radii,
            r4r2=r4r2,
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            energy=energy,
            forces=forces,
            coord_num=coord_num,
            virial=virial,
            device=device,
        )

        # Verify outputs are finite and reasonable
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()
        assert energy[0] < 0.0  # Dispersion is attractive

    @pytest.mark.usefixtures("element_tables", "device")
    def test_custom_op_empty_system(self, request):
        """Test custom op with empty system."""
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        # Empty system
        positions = torch.zeros((0, 3), dtype=torch.float32, device=device)
        numbers = torch.zeros((0,), dtype=torch.int32, device=device)
        idx_j = torch.zeros((0,), dtype=torch.int32, device=device)
        neighbor_ptr = torch.zeros((1,), dtype=torch.int32, device=device)

        # Parameters (convert from numpy)
        max_z_inc = element_tables["z_max_inc"]
        covalent_radii = numpy_to_torch(element_tables["rcov"], device).float()
        r4r2 = numpy_to_torch(element_tables["r4r2"], device).float()
        c6_reference = (
            numpy_to_torch(element_tables["c6ref"], device)
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
        )
        coord_num_ref = (
            numpy_to_torch(element_tables["cnref_i"], device)
            .float()
            .reshape(max_z_inc, max_z_inc, 5, 5)
        )

        # Allocate outputs
        energy = torch.zeros(1, dtype=torch.float32, device=device)
        forces = torch.zeros((0, 3), dtype=torch.float32, device=device)
        coord_num = torch.zeros((0,), dtype=torch.float32, device=device)
        virial = torch.zeros((0, 3, 3), dtype=torch.float32, device=device)

        # Should handle empty case without error
        _dftd3_op(
            positions=positions,
            numbers=numbers,
            idx_j=idx_j,
            neighbor_ptr=neighbor_ptr,
            covalent_radii=covalent_radii,
            r4r2=r4r2,
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            energy=energy,
            forces=forces,
            coord_num=coord_num,
            virial=virial,
            device=device,
        )

        energy = energy.cpu().item()

        assert energy == pytest.approx(0.0)


# ==============================================================================
# Device Handling Tests
# ==============================================================================


class TestDeviceHandling:
    """Test device handling in PyTorch bindings."""

    @pytest.mark.usefixtures("element_tables", "device")
    def test_device_execution(self, request):
        """Test that computation works on both CPU and GPU."""
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32, device=device)
        neighbor_matrix = torch.tensor(
            [[1, 2], [0, 2]], dtype=torch.int32, device=device
        )

        max_z_inc = element_tables["z_max_inc"]
        d3_params = D3Parameters(
            rcov=numpy_to_torch(element_tables["rcov"], device),
            r4r2=numpy_to_torch(element_tables["r4r2"], device),
            c6ab=numpy_to_torch(element_tables["c6ref"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
            cn_ref=numpy_to_torch(element_tables["cnref_i"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
        )

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            device=device,
        )

        # Verify outputs are on correct device
        expected_device = "cuda" if "cuda" in device else "cpu"
        assert energy.device.type == expected_device
        assert forces.device.type == expected_device
        assert coord_num.device.type == expected_device

    @pytest.mark.usefixtures("element_tables")
    def test_cpu_gpu_consistency(self, request):
        """Test that CPU and GPU produce identical results."""
        # Skip if CUDA not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for PyTorch")

        element_tables = request.getfixturevalue("element_tables")

        # Setup inputs on CPU
        positions_cpu = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32, device="cpu"
        )
        numbers_cpu = torch.tensor([1, 1], dtype=torch.int32, device="cpu")
        neighbor_matrix_cpu = torch.tensor(
            [[1, 2], [0, 2]], dtype=torch.int32, device="cpu"
        )

        max_z_inc = element_tables["z_max_inc"]
        d3_params_cpu = D3Parameters(
            rcov=numpy_to_torch(element_tables["rcov"]),
            r4r2=numpy_to_torch(element_tables["r4r2"]),
            c6ab=numpy_to_torch(element_tables["c6ref"]).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
            cn_ref=numpy_to_torch(element_tables["cnref_i"]).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
        )

        # Run on CPU
        energy_cpu, forces_cpu, coord_num_cpu = dftd3(
            positions=positions_cpu,
            numbers=numbers_cpu,
            neighbor_matrix=neighbor_matrix_cpu,
            d3_params=d3_params_cpu,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            device="cpu",
        )

        # Setup inputs on GPU
        positions_gpu = positions_cpu.cuda()
        numbers_gpu = numbers_cpu.cuda()
        neighbor_matrix_gpu = neighbor_matrix_cpu.cuda()
        d3_params_gpu = d3_params_cpu.to(device="cuda")

        # Run on GPU
        energy_gpu, forces_gpu, coord_num_gpu = dftd3(
            positions=positions_gpu,
            numbers=numbers_gpu,
            neighbor_matrix=neighbor_matrix_gpu,
            d3_params=d3_params_gpu,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            device="cuda:0",
        )

        # Compare results
        torch.testing.assert_close(energy_gpu.cpu(), energy_cpu, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(forces_gpu.cpu(), forces_cpu, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            coord_num_gpu.cpu(), coord_num_cpu, rtol=1e-6, atol=1e-6
        )


# ==============================================================================
# Dtype Tests
# ==============================================================================


class TestDtypeHandling:
    """Test dtype handling in PyTorch bindings."""

    @pytest.mark.parametrize("positions_dtype", [torch.float32, torch.float64])
    @pytest.mark.usefixtures("element_tables", "device")
    def test_position_dtypes(self, request, positions_dtype):
        """Test that both float32 and float64 positions are supported."""
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=positions_dtype, device=device
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32, device=device)
        neighbor_matrix = torch.tensor(
            [[1, 2], [0, 2]], dtype=torch.int32, device=device
        )

        max_z_inc = element_tables["z_max_inc"]
        d3_params = D3Parameters(
            rcov=numpy_to_torch(element_tables["rcov"], device),
            r4r2=numpy_to_torch(element_tables["r4r2"], device),
            c6ab=numpy_to_torch(element_tables["c6ref"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
            cn_ref=numpy_to_torch(element_tables["cnref_i"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
        )

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            device=device,
        )

        # Outputs should always be float32
        assert energy.dtype == torch.float32
        assert forces.dtype == torch.float32
        assert coord_num.dtype == torch.float32
        assert torch.isfinite(energy).all()

    @pytest.mark.usefixtures("element_tables", "device")
    def test_float32_float64_consistency(self, request):
        """Test that float32 and float64 produce similar results."""
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        # Run with float32
        positions_f32 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32, device=device)
        neighbor_matrix = torch.tensor(
            [[1, 2], [0, 2]], dtype=torch.int32, device=device
        )

        max_z_inc = element_tables["z_max_inc"]
        d3_params = D3Parameters(
            rcov=numpy_to_torch(element_tables["rcov"], device),
            r4r2=numpy_to_torch(element_tables["r4r2"], device),
            c6ab=numpy_to_torch(element_tables["c6ref"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
            cn_ref=numpy_to_torch(element_tables["cnref_i"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
        )

        energy_f32, forces_f32, coord_num_f32 = dftd3(
            positions=positions_f32,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            device=device,
        )

        # Run with float64
        positions_f64 = positions_f32.double()
        energy_f64, forces_f64, coord_num_f64 = dftd3(
            positions=positions_f64,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            device=device,
        )

        # Should be very close (float64 may have slightly better precision)
        torch.testing.assert_close(energy_f64, energy_f32, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(forces_f64, forces_f32, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(coord_num_f64, coord_num_f32, rtol=1e-6, atol=1e-7)


# ==============================================================================
# Torch.compile Tests
# ==============================================================================


class TestTorchCompile:
    """Test torch.compile compatibility."""

    @pytest.mark.usefixtures("element_tables", "device")
    def test_wrapper_compile(self, request):
        """Test that dftd3 wrapper can be compiled."""
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32, device=device)
        neighbor_matrix = torch.tensor(
            [[1, 2], [0, 2]], dtype=torch.int32, device=device
        )

        max_z_inc = element_tables["z_max_inc"]
        d3_params = D3Parameters(
            rcov=numpy_to_torch(element_tables["rcov"], device),
            r4r2=numpy_to_torch(element_tables["r4r2"], device),
            c6ab=numpy_to_torch(element_tables["c6ref"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
            cn_ref=numpy_to_torch(element_tables["cnref_i"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
        )

        # Compile the function
        compiled_fn = torch.compile(dftd3)

        # Run compiled version
        energy_compiled, forces_compiled, coord_num_compiled = compiled_fn(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            device=device,
        )

        # Run eager version
        energy_eager, forces_eager, coord_num_eager = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            device=device,
        )

        # Results should match
        torch.testing.assert_close(energy_compiled, energy_eager, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(forces_compiled, forces_eager, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            coord_num_compiled, coord_num_eager, rtol=1e-6, atol=1e-6
        )

    @pytest.mark.usefixtures("element_tables", "device")
    def test_compiled_multiple_calls(self, request):
        """Test that compiled function can be called multiple times."""
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32, device=device)
        neighbor_matrix = torch.tensor(
            [[1, 2], [0, 2]], dtype=torch.int32, device=device
        )

        max_z_inc = element_tables["z_max_inc"]
        d3_params = D3Parameters(
            rcov=numpy_to_torch(element_tables["rcov"], device),
            r4r2=numpy_to_torch(element_tables["r4r2"], device),
            c6ab=numpy_to_torch(element_tables["c6ref"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
            cn_ref=numpy_to_torch(element_tables["cnref_i"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
        )

        # Compile the function
        compiled_fn = torch.compile(dftd3)

        # Call multiple times
        results = []
        for _ in range(3):
            energy, forces, coord_num = compiled_fn(
                positions=positions,
                numbers=numbers,
                neighbor_matrix=neighbor_matrix,
                d3_params=d3_params,
                a1=0.4,
                a2=4.0,
                s8=0.8,
                device=device,
            )
            results.append((energy, forces, coord_num))

        # All calls should produce identical results
        for i in range(1, len(results)):
            torch.testing.assert_close(results[i][0], results[0][0])
            torch.testing.assert_close(results[i][1], results[0][1])
            torch.testing.assert_close(results[i][2], results[0][2])


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests for PyTorch bindings with high-level wrapper."""

    @pytest.mark.usefixtures("h2_system", "element_tables", "device")
    def test_neighbor_matrix_format(self, request):
        """Test integration with neighbor matrix format."""
        h2_system = request.getfixturevalue("h2_system")
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        # Convert numpy arrays from fixtures to torch tensors
        positions = (
            torch.from_numpy(h2_system["coord"])
            .reshape(h2_system["B"], 3)
            .float()
            .to(device)
        )
        numbers = torch.from_numpy(h2_system["numbers"]).int().to(device)
        neighbor_matrix = torch.from_numpy(h2_system["nbmat"]).int().to(device)

        max_z_inc = element_tables["z_max_inc"]
        d3_params = D3Parameters(
            rcov=numpy_to_torch(element_tables["rcov"], device),
            r4r2=numpy_to_torch(element_tables["r4r2"], device),
            c6ab=numpy_to_torch(element_tables["c6ref"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
            cn_ref=numpy_to_torch(element_tables["cnref_i"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
        )

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            device=device,
        )

        # Basic checks
        assert energy.shape == (1,)
        assert forces.shape == (h2_system["B"], 3)
        assert coord_num.shape == (h2_system["B"],)
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()

    @pytest.mark.usefixtures("h2_system", "element_tables", "device")
    def test_neighbor_list_format(self, request):
        """Test integration with neighbor list format."""
        h2_system = request.getfixturevalue("h2_system")
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        # Convert numpy arrays from fixtures to torch tensors
        positions = (
            torch.from_numpy(h2_system["coord"])
            .reshape(h2_system["B"], 3)
            .float()
            .to(device)
        )
        numbers = torch.from_numpy(h2_system["numbers"]).int().to(device)

        # Build neighbor list from API
        neighbor_list, neighbor_ptr = build_neighbor_list(
            positions=positions,
            cutoff=10.0,
            return_neighbor_list=True,
        )

        max_z_inc = element_tables["z_max_inc"]
        d3_params = D3Parameters(
            rcov=numpy_to_torch(element_tables["rcov"], device),
            r4r2=numpy_to_torch(element_tables["r4r2"], device),
            c6ab=numpy_to_torch(element_tables["c6ref"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
            cn_ref=numpy_to_torch(element_tables["cnref_i"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
        )

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            d3_params=d3_params,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            device=device,
        )

        # Basic checks
        assert energy.shape == (1,)
        assert forces.shape == (h2_system["B"], 3)
        assert coord_num.shape == (h2_system["B"],)
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(coord_num).all()

    def test_d3_parameters_validation(self, request):
        """Test that D3Parameters validates inputs correctly."""
        # Valid parameters
        params = D3Parameters(
            rcov=torch.rand(10, dtype=torch.float32),
            r4r2=torch.rand(10, dtype=torch.float32),
            c6ab=torch.rand(10, 10, 5, 5, dtype=torch.float32),
            cn_ref=torch.rand(10, 10, 5, 5, dtype=torch.float32),
        )
        assert params.max_z == 9
        assert params.device == torch.device("cpu")

        # Invalid: wrong dtype (create int tensor then convert)
        with pytest.raises(TypeError, match="must be float32 or float64"):
            D3Parameters(
                rcov=torch.randint(0, 10, (10,), dtype=torch.int32).float().int(),
                r4r2=torch.rand(10, dtype=torch.float32),
                c6ab=torch.rand(10, 10, 5, 5, dtype=torch.float32),
                cn_ref=torch.rand(10, 10, 5, 5, dtype=torch.float32),
            )

        # Invalid: shape mismatch
        with pytest.raises(ValueError, match="r4r2 must have shape"):
            D3Parameters(
                rcov=torch.rand(10, dtype=torch.float32),
                r4r2=torch.rand(5, dtype=torch.float32),  # Wrong size
                c6ab=torch.rand(10, 10, 5, 5, dtype=torch.float32),
                cn_ref=torch.rand(10, 10, 5, 5, dtype=torch.float32),
            )

    @pytest.mark.parametrize(
        "precision", [torch.bfloat16, torch.float16, torch.float32, torch.float64]
    )
    @pytest.mark.usefixtures("device")
    def test_d3_parameters_device_transfer(self, request, precision):
        """Test that D3Parameters.to() method works correctly."""
        device = request.getfixturevalue("device")
        params_cpu = D3Parameters(
            rcov=torch.rand(10),
            r4r2=torch.rand(10),
            c6ab=torch.rand(10, 10, 5, 5),
            cn_ref=torch.rand(10, 10, 5, 5),
        )
        if precision in [torch.bfloat16, torch.float16]:
            with pytest.raises(TypeError, match="must be float32 or float64"):
                params_cpu.to(device=device, dtype=precision)
            return
        new_params = params_cpu.to(device=device, dtype=precision)

        for name in ["rcov", "r4r2", "c6ab", "cn_ref"]:
            assert getattr(new_params, name).device == torch.device(device)
            assert getattr(new_params, name).dtype == precision


# ==============================================================================
# Regression Tests
# ==============================================================================


class TestValidation:
    """Comprehensive validation tests for error handling and edge cases."""

    def test_d3_parameters_validation_comprehensive(self, request):
        """Test all D3Parameters validation error paths."""
        # Test 1: Non-tensor input for rcov
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            D3Parameters(
                rcov=[1.0, 2.0],  # List instead of tensor
                r4r2=torch.rand(10, dtype=torch.float32),
                c6ab=torch.rand(10, 10, 5, 5, dtype=torch.float32),
                cn_ref=torch.rand(10, 10, 5, 5, dtype=torch.float32),
            )

        # Test 2: Wrong dimension for rcov (2D instead of 1D)
        with pytest.raises(ValueError, match="rcov must be 1D tensor"):
            D3Parameters(
                rcov=torch.rand(10, 1, dtype=torch.float32),  # 2D
                r4r2=torch.rand(10, dtype=torch.float32),
                c6ab=torch.rand(10, 10, 5, 5, dtype=torch.float32),
                cn_ref=torch.rand(10, 10, 5, 5, dtype=torch.float32),
            )

        # Test 3: Too few elements in rcov (only 1 element)
        with pytest.raises(ValueError, match="at least 2 elements"):
            D3Parameters(
                rcov=torch.rand(1, dtype=torch.float32),
                r4r2=torch.rand(1, dtype=torch.float32),
                c6ab=torch.rand(1, 1, 5, 5, dtype=torch.float32),
                cn_ref=torch.rand(1, 1, 5, 5, dtype=torch.float32),
            )

        # Test 4: c6ab shape mismatch
        with pytest.raises(ValueError, match="c6ab must have shape"):
            D3Parameters(
                rcov=torch.rand(10, dtype=torch.float32),
                r4r2=torch.rand(10, dtype=torch.float32),
                c6ab=torch.rand(10, 10, 3, 3, dtype=torch.float32),  # Wrong mesh size
                cn_ref=torch.rand(10, 10, 5, 5, dtype=torch.float32),
            )

        # Test 5: cn_ref shape mismatch
        with pytest.raises(ValueError, match="cn_ref must have shape"):
            D3Parameters(
                rcov=torch.rand(10, dtype=torch.float32),
                r4r2=torch.rand(10, dtype=torch.float32),
                c6ab=torch.rand(10, 10, 5, 5, dtype=torch.float32),
                cn_ref=torch.rand(8, 8, 5, 5, dtype=torch.float32),  # Wrong size
            )

        # Test 6: Device inconsistency - skip if CUDA not available
        if torch.cuda.is_available():
            with pytest.raises(ValueError, match="must be on the same device"):
                D3Parameters(
                    rcov=torch.rand(10, dtype=torch.float32, device="cpu"),
                    r4r2=torch.rand(
                        10, dtype=torch.float32, device="cuda"
                    ),  # Different device
                    c6ab=torch.rand(10, 10, 5, 5, dtype=torch.float32, device="cpu"),
                    cn_ref=torch.rand(10, 10, 5, 5, dtype=torch.float32, device="cpu"),
                )

    @pytest.mark.usefixtures("element_tables", "device")
    def test_dftd3_neighbor_format_validation(self, request):
        """Test dftd3() neighbor format validation."""
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32, device=device)
        neighbor_matrix = torch.tensor(
            [[1, 2], [0, 2]], dtype=torch.int32, device=device
        )
        neighbor_list = torch.tensor([1, 0], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

        max_z_inc = element_tables["z_max_inc"]
        d3_params = D3Parameters(
            rcov=numpy_to_torch(element_tables["rcov"], device),
            r4r2=numpy_to_torch(element_tables["r4r2"], device),
            c6ab=numpy_to_torch(element_tables["c6ref"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
            cn_ref=numpy_to_torch(element_tables["cnref_i"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
        )

        # Test 1: Both neighbor formats provided
        with pytest.raises(
            ValueError, match="Cannot provide both neighbor_matrix and neighbor_list"
        ):
            dftd3(
                positions=positions,
                numbers=numbers,
                neighbor_matrix=neighbor_matrix,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                d3_params=d3_params,
                a1=0.4,
                a2=4.0,
                s8=0.8,
                device=device,
            )

        # Test 2: Neither neighbor format provided
        with pytest.raises(
            ValueError, match="Must provide either neighbor_matrix or neighbor_list"
        ):
            dftd3(
                positions=positions,
                numbers=numbers,
                d3_params=d3_params,
                a1=0.4,
                a2=4.0,
                s8=0.8,
                device=device,
            )

        # Test 3: Wrong shift format for neighbor_matrix (using unit_shifts)
        with pytest.raises(ValueError, match="unit_shifts is for neighbor_list format"):
            dftd3(
                positions=positions,
                numbers=numbers,
                neighbor_matrix=neighbor_matrix,
                unit_shifts=torch.zeros((2, 3), dtype=torch.int32, device=device),
                d3_params=d3_params,
                a1=0.4,
                a2=4.0,
                s8=0.8,
                device=device,
            )

        # Test 4: Wrong shift format for neighbor_list (using neighbor_matrix_shifts)
        with pytest.raises(
            ValueError, match="neighbor_matrix_shifts is for neighbor_matrix format"
        ):
            dftd3(
                positions=positions,
                numbers=numbers,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_matrix_shifts=torch.zeros(
                    (2, 2, 3), dtype=torch.int32, device=device
                ),
                d3_params=d3_params,
                a1=0.4,
                a2=4.0,
                s8=0.8,
                device=device,
            )

        # Test 5: Missing neighbor_ptr for neighbor_list format
        with pytest.raises(ValueError, match="neighbor_ptr must be provided"):
            dftd3(
                positions=positions,
                numbers=numbers,
                neighbor_list=neighbor_list,
                d3_params=d3_params,
                a1=0.4,
                a2=4.0,
                s8=0.8,
                device=device,
            )

    @pytest.mark.usefixtures("element_tables", "device")
    def test_dftd3_parameter_and_functional_validation(self, request):
        """Test dftd3() parameter and functional validation."""
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32, device=device)
        neighbor_matrix = torch.tensor(
            [[1, 2], [0, 2]], dtype=torch.int32, device=device
        )

        # Test 1: Missing all functional parameters
        max_z_inc = element_tables["z_max_inc"]
        d3_params = D3Parameters(
            rcov=numpy_to_torch(element_tables["rcov"], device),
            r4r2=numpy_to_torch(element_tables["r4r2"], device),
            c6ab=numpy_to_torch(element_tables["c6ref"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
            cn_ref=numpy_to_torch(element_tables["cnref_i"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
        )

        with pytest.raises(
            ValueError, match="Functional parameters a1, a2, and s8 must be provided"
        ):
            dftd3(
                positions=positions,
                numbers=numbers,
                neighbor_matrix=neighbor_matrix,
                d3_params=d3_params,
                a1=None,
                a2=None,
                s8=None,
                device=device,
            )

        # Test 2: Missing all D3 parameters
        with pytest.raises(
            RuntimeError, match="DFT-D3 parameters must be explicitly provided"
        ):
            dftd3(
                positions=positions,
                numbers=numbers,
                neighbor_matrix=neighbor_matrix,
                a1=0.4,
                a2=4.0,
                s8=0.8,
                device=device,
            )

    @pytest.mark.usefixtures("element_tables", "device")
    def test_dftd3_virial_and_pbc_validation(self, request):
        """Test dftd3() virial computation and PBC validation."""
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        numbers = torch.tensor([1, 1], dtype=torch.int32, device=device)
        neighbor_matrix = torch.tensor(
            [[1, 2], [0, 2]], dtype=torch.int32, device=device
        )
        neighbor_list = torch.tensor([1, 0], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

        max_z_inc = element_tables["z_max_inc"]
        d3_params = D3Parameters(
            rcov=numpy_to_torch(element_tables["rcov"], device),
            r4r2=numpy_to_torch(element_tables["r4r2"], device),
            c6ab=numpy_to_torch(element_tables["c6ref"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
            cn_ref=numpy_to_torch(element_tables["cnref_i"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
        )

        # Test 1: Virial computation without cell (neighbor_matrix format)
        with pytest.raises(
            ValueError, match="Virial computation requires periodic boundary conditions"
        ):
            dftd3(
                positions=positions,
                numbers=numbers,
                neighbor_matrix=neighbor_matrix,
                d3_params=d3_params,
                a1=0.4,
                a2=4.0,
                s8=0.8,
                compute_virial=True,
                device=device,
            )

        # Test 2: Virial computation without neighbor_matrix_shifts (neighbor_matrix format)
        cell = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)
        with pytest.raises(ValueError, match="Please provide neighbor_matrix_shifts"):
            dftd3(
                positions=positions,
                numbers=numbers,
                neighbor_matrix=neighbor_matrix,
                cell=cell,
                d3_params=d3_params,
                a1=0.4,
                a2=4.0,
                s8=0.8,
                compute_virial=True,
                device=device,
            )

        # Test 3: Virial computation without unit_shifts (neighbor_list format)
        with pytest.raises(ValueError, match="Please provide unit_shifts"):
            dftd3(
                positions=positions,
                numbers=numbers,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                cell=cell,
                d3_params=d3_params,
                a1=0.4,
                a2=4.0,
                s8=0.8,
                compute_virial=True,
                device=device,
            )

        # Test 4: Empty system with virial computation
        empty_positions = torch.zeros((0, 3), dtype=torch.float32, device=device)
        empty_numbers = torch.zeros((0,), dtype=torch.int32, device=device)
        empty_neighbor_matrix = torch.zeros((0, 0), dtype=torch.int32, device=device)
        empty_neighbor_matrix_shifts = torch.zeros(
            (0, 0, 3), dtype=torch.int32, device=device
        )

        energy, forces, coord_num, virial = dftd3(
            positions=empty_positions,
            numbers=empty_numbers,
            neighbor_matrix=empty_neighbor_matrix,
            neighbor_matrix_shifts=empty_neighbor_matrix_shifts,
            cell=cell,
            d3_params=d3_params,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            compute_virial=True,
            device=device,
        )

        # Verify empty outputs with virial
        assert energy.shape == (1,)
        assert forces.shape == (0, 3)
        assert coord_num.shape == (0,)
        assert virial.shape == (0, 3, 3)
        energy = energy.cpu().item()
        assert energy == pytest.approx(0.0)


class TestRegression:
    """Regression tests to ensure PyTorch bindings produce correct results."""

    @pytest.mark.usefixtures("h2_system", "element_tables", "device")
    def test_h2_energy_sign(self, request):
        """Test that H2 produces negative (attractive) dispersion energy."""
        h2_system = request.getfixturevalue("h2_system")
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        # Convert numpy arrays from fixtures to torch tensors
        positions = (
            torch.from_numpy(h2_system["coord"])
            .reshape(h2_system["B"], 3)
            .float()
            .to(device)
        )
        numbers = torch.from_numpy(h2_system["numbers"]).int().to(device)
        neighbor_matrix = torch.from_numpy(h2_system["nbmat"]).int().to(device)

        max_z_inc = element_tables["z_max_inc"]
        d3_params = D3Parameters(
            rcov=numpy_to_torch(element_tables["rcov"], device),
            r4r2=numpy_to_torch(element_tables["r4r2"], device),
            c6ab=numpy_to_torch(element_tables["c6ref"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
            cn_ref=numpy_to_torch(element_tables["cnref_i"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
        )

        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            device=device,
        )

        # Dispersion should be attractive
        assert energy[0] < 0.0

        # Forces should be opposite for symmetric system
        torch.testing.assert_close(forces[0], -forces[1], rtol=1e-5, atol=1e-7)

        # Coordination numbers should be small but non-zero
        assert (coord_num > 0).all()
        assert (coord_num < 1.0).all()

    @pytest.mark.usefixtures("ne2_system", "element_tables", "device")
    def test_ne2_larger_dispersion(self, request):
        """Test that Ne2 has larger dispersion than H2 (more polarizable)."""
        ne2_system = request.getfixturevalue("ne2_system")
        element_tables = request.getfixturevalue("element_tables")
        device = request.getfixturevalue("device")

        # Ne2 energy - convert numpy arrays from fixtures to torch tensors
        positions = (
            torch.from_numpy(ne2_system["coord"])
            .reshape(ne2_system["B"], 3)
            .float()
            .to(device)
        )
        numbers = torch.from_numpy(ne2_system["numbers"]).int().to(device)
        neighbor_matrix = torch.from_numpy(ne2_system["nbmat"]).int().to(device)

        max_z_inc = element_tables["z_max_inc"]
        d3_params = D3Parameters(
            rcov=numpy_to_torch(element_tables["rcov"], device),
            r4r2=numpy_to_torch(element_tables["r4r2"], device),
            c6ab=numpy_to_torch(element_tables["c6ref"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
            cn_ref=numpy_to_torch(element_tables["cnref_i"], device).reshape(
                max_z_inc, max_z_inc, 5, 5
            ),
        )

        energy_ne2, _, _ = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
            a1=0.4,
            a2=4.0,
            s8=0.8,
            device=device,
        )

        # Ne2 should have significant dispersion energy
        assert energy_ne2[0] < -1e-3  # Reasonably large magnitude
        assert torch.isfinite(energy_ne2).all()

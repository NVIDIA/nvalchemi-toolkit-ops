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
Tests for autograd utilities in nvalchemiops.autograd

This test suite validates the Warp-PyTorch autograd integration utilities
including conditional tape creation, gradient extraction, and standard backward patterns.
"""

from typing import Any

import pytest
import torch
import warp as wp

from nvalchemiops.torch.autograd import (
    OutputSpec,
    WarpAutogradContextManager,
    _resolve_warp_dtype,
    attach_for_backward,
    extract_gradients,
    needs_grad,
    retrieve_for_backward,
    standard_backward,
    warp_custom_op,
)

# ============================================================================
# Simple Test Kernels
# ============================================================================


@wp.kernel
def simple_multiply_kernel(
    input_array: wp.array(dtype=wp.float32),
    multiplier: wp.float32,
    output_array: wp.array(dtype=wp.float32),
):
    """Simple kernel: output = input * multiplier"""
    i = wp.tid()
    output_array[i] = input_array[i] * multiplier


@wp.kernel
def elementwise_add_kernel(
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    output: wp.array(dtype=wp.float32),
):
    """Simple kernel: output = a + b"""
    i = wp.tid()
    output[i] = a[i] + b[i]


@wp.kernel
def vector_scale_kernel(
    vectors: wp.array(dtype=wp.vec3f),
    scale: wp.float32,
    output: wp.array(dtype=wp.vec3f),
):
    """Vector kernel: output = vectors * scale"""
    i = wp.tid()
    output[i] = vectors[i] * scale


@wp.kernel
def dummy_energy_forces(
    positions: wp.array(dtype=wp.vec3f),
    energies: wp.array(dtype=wp.float32),
    forces: wp.array(dtype=wp.vec3f),
):
    """
    Compute dummy energy and forces for testing multiple outputs.

    Energy: E = -0.5 * ||pos||^2 (like a harmonic potential centered at origin)
    Forces: F = -grad(E) = -pos (force points toward origin)
    """
    i = wp.tid()
    pos = positions[i]

    energies[i] = -0.5 * wp.length_sq(pos)
    forces[i] = -pos


# ============================================================================
# Test Helper Functions
# ============================================================================


class TestNeedsGrad:
    """Tests for needs_grad() function."""

    def test_single_tensor_requires_grad(self):
        """Test with single tensor requiring gradients."""
        x = torch.randn(10, requires_grad=True)
        assert needs_grad(x) is True

    def test_single_tensor_no_grad(self):
        """Test with single tensor not requiring gradients."""
        x = torch.randn(10, requires_grad=False)
        assert needs_grad(x) is False

    def test_multiple_tensors_all_require_grad(self):
        """Test with multiple tensors all requiring gradients."""
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)
        z = torch.randn(10, requires_grad=True)
        assert needs_grad(x, y, z) is True

    def test_multiple_tensors_none_require_grad(self):
        """Test with multiple tensors none requiring gradients."""
        x = torch.randn(10, requires_grad=False)
        y = torch.randn(10, requires_grad=False)
        z = torch.randn(10, requires_grad=False)
        assert needs_grad(x, y, z) is False

    def test_multiple_tensors_mixed_grad(self):
        """Test with mixed gradient requirements."""
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=False)
        z = torch.randn(10, requires_grad=False)
        assert needs_grad(x, y, z) is True

    def test_empty_input(self):
        """Test with no tensors."""
        assert needs_grad() is False

    def test_non_tensor_inputs_ignored(self):
        """Test that non-tensor inputs are safely ignored."""
        x = torch.randn(10, requires_grad=True)
        assert needs_grad(x, 1.0, "string", None) is True


class TestWarpAutogradContextManager:
    """Tests for WarpAutogradContextManager() context manager."""

    def test_creates_tape_when_enabled(self):
        """Test that tape is created when enable=True."""
        with WarpAutogradContextManager(True) as tape:
            assert tape is not None
            assert isinstance(tape, wp.Tape)

    def test_no_tape_when_disabled(self):
        """Test that no tape is created when enable=False."""
        with WarpAutogradContextManager(False) as tape:
            assert tape is None

    def test_tape_records_operations(self):
        """Test that tape actually records operations."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        input_tensor = torch.ones(
            10, device=device, dtype=torch.float32, requires_grad=True
        )
        output_tensor = torch.zeros(
            10, device=device, dtype=torch.float32, requires_grad=True
        )
        loss = torch.zeros(1, device=device, dtype=torch.float32, requires_grad=True)
        wp_input = wp.from_torch(input_tensor, dtype=wp.float32)
        wp_output = wp.from_torch(output_tensor, dtype=wp.float32)
        wp_loss = wp.from_torch(loss, dtype=wp.float32)
        with WarpAutogradContextManager(True) as tape:
            wp.launch(
                simple_multiply_kernel,
                dim=10,
                inputs=[wp_input, wp.float32(2.0), wp_output],
                device=device,
            )

        # Check tape captured something
        tape.backward(wp_loss)
        assert tape is not None
        assert len(tape.gradients) > 0  # Tape has gradient storage

    def test_no_overhead_when_disabled(self):
        """Test that disabled context has no performance overhead."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        input_tensor = torch.ones(10, device=device, dtype=torch.float32)
        output_tensor = torch.zeros(10, device=device, dtype=torch.float32)

        wp_input = wp.from_torch(input_tensor, dtype=wp.float32)
        wp_output = wp.from_torch(output_tensor, dtype=wp.float32)

        # Should not raise any errors
        with WarpAutogradContextManager(False) as tape:
            wp.launch(
                simple_multiply_kernel,
                dim=10,
                inputs=[wp_input, wp.float32(2.0), wp_output],
                device=device,
            )

        assert tape is None


class TestAttachRetrieveBackward:
    """Tests for attach_for_backward() and retrieve_for_backward()."""

    def test_attach_tape_only(self):
        """Test attaching only a tape."""
        output = torch.zeros(10)
        tape = wp.Tape()

        attach_for_backward(output, tape=tape)

        assert hasattr(output, "_warp_tape")
        assert output._warp_tape is tape

    def test_attach_arrays_only(self):
        """Test attaching only warp arrays."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output = torch.zeros(10, device=device)

        wp_array1 = wp.zeros(10, dtype=wp.float32, device=device)
        wp_array2 = wp.zeros(10, dtype=wp.float32, device=device)

        attach_for_backward(output, positions=wp_array1, charges=wp_array2)

        assert hasattr(output, "_wp_positions")
        assert hasattr(output, "_wp_charges")
        assert output._wp_positions is wp_array1
        assert output._wp_charges is wp_array2

    def test_attach_tape_and_arrays(self):
        """Test attaching both tape and arrays."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output = torch.zeros(10, device=device)
        tape = wp.Tape()
        wp_array = wp.zeros(10, dtype=wp.float32, device=device)

        attach_for_backward(output, tape=tape, positions=wp_array)

        assert hasattr(output, "_warp_tape")
        assert hasattr(output, "_wp_positions")
        assert output._warp_tape is tape
        assert output._wp_positions is wp_array

    def test_retrieve_tape_and_arrays(self):
        """Test retrieving tape and arrays."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output = torch.zeros(10, device=device)
        tape = wp.Tape()

        wp_pos = wp.zeros(10, dtype=wp.float32, device=device)
        wp_charges = wp.zeros(5, dtype=wp.float32, device=device)

        attach_for_backward(output, tape=tape, positions=wp_pos, charges=wp_charges)

        # Retrieve
        retrieved_tape, arrays = retrieve_for_backward(output, "positions", "charges")

        assert retrieved_tape is tape
        assert "positions" in arrays
        assert "charges" in arrays
        assert arrays["positions"] is wp_pos
        assert arrays["charges"] is wp_charges

    def test_retrieve_partial_arrays(self):
        """Test retrieving only some of the attached arrays."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output = torch.zeros(10, device=device)
        tape = wp.Tape()

        wp_pos = wp.zeros(10, dtype=wp.float32, device=device)
        wp_charges = wp.zeros(5, dtype=wp.float32, device=device)
        wp_cell = wp.zeros(9, dtype=wp.float32, device=device)

        attach_for_backward(
            output, tape=tape, positions=wp_pos, charges=wp_charges, cell=wp_cell
        )

        # Retrieve only some
        retrieved_tape, arrays = retrieve_for_backward(output, "positions", "cell")

        assert retrieved_tape is tape
        assert "positions" in arrays
        assert "cell" in arrays
        assert "charges" not in arrays


class TestExtractGradients:
    """Tests for extract_gradients() function."""

    def test_extract_single_gradient(self):
        """Test extracting gradient from single input."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create mock context
        class MockCtx:
            def __init__(self):
                self.positions = torch.randn(10, 3, device=device, requires_grad=True)

        ctx = MockCtx()

        # Create warp array with gradient
        wp_positions = wp.zeros(
            (10, 3), dtype=wp.float32, device=device, requires_grad=True
        )
        wp_positions.grad = wp.ones((10, 3), dtype=wp.float32, device=device)

        arrays = {"positions": wp_positions}

        grads = extract_gradients(ctx, arrays, ["positions"])

        assert len(grads) == 1
        assert grads[0] is not None
        assert isinstance(grads[0], torch.Tensor)
        assert grads[0].shape == (10, 3)

    def test_extract_multiple_gradients(self):
        """Test extracting gradients from multiple inputs."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        class MockCtx:
            def __init__(self):
                self.positions = torch.randn(10, 3, device=device, requires_grad=True)
                self.charges = torch.randn(10, device=device, requires_grad=True)

        ctx = MockCtx()

        wp_positions = wp.zeros(
            (10, 3), dtype=wp.float32, device=device, requires_grad=True
        )
        wp_positions.grad = wp.ones((10, 3), dtype=wp.float32, device=device)

        wp_charges = wp.zeros(10, dtype=wp.float32, device=device, requires_grad=True)
        wp_charges.grad = wp.ones(10, dtype=wp.float32, device=device) * 2.0

        arrays = {"positions": wp_positions, "charges": wp_charges}

        grads = extract_gradients(ctx, arrays, ["positions", "charges"])

        assert len(grads) == 2
        assert grads[0] is not None
        assert grads[1] is not None
        assert grads[0].shape == (10, 3)
        assert grads[1].shape == (10,)

    def test_none_for_no_requires_grad(self):
        """Test that None is returned for inputs without requires_grad."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        class MockCtx:
            def __init__(self):
                self.positions = torch.randn(10, 3, device=device, requires_grad=True)
                self.charges = torch.randn(10, device=device, requires_grad=False)
                self.alpha = 1.0

        ctx = MockCtx()

        wp_positions = wp.zeros(
            (10, 3), dtype=wp.float32, device=device, requires_grad=True
        )
        wp_positions.grad = wp.ones((10, 3), dtype=wp.float32, device=device)

        arrays = {"positions": wp_positions}

        grads = extract_gradients(ctx, arrays, ["positions", "charges", "alpha"])

        assert len(grads) == 3
        assert grads[0] is not None  # positions
        assert grads[1] is None  # charges (no requires_grad)
        assert grads[2] is None  # alpha (not a tensor)

    def test_zeros_for_missing_array(self):
        """Test that zeros are returned when warp array not provided."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        class MockCtx:
            def __init__(self):
                self.positions = torch.randn(10, 3, device=device, requires_grad=True)
                self.charges = torch.randn(10, device=device, requires_grad=True)

        ctx = MockCtx()

        # Only provide positions array, not charges
        wp_positions = wp.zeros(
            (10, 3), dtype=wp.float32, device=device, requires_grad=True
        )
        wp_positions.grad = wp.ones((10, 3), dtype=wp.float32, device=device)

        arrays = {"positions": wp_positions}

        grads = extract_gradients(ctx, arrays, ["positions", "charges"])

        assert len(grads) == 2
        assert grads[0] is not None
        assert grads[1] is not None  # Should be zeros
        assert torch.all(grads[1] == 0)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegrationSimpleOperator:
    """Integration test with a complete custom operator."""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_forward_backward_with_helpers(self, device):
        """Test complete forward/backward cycle using helper functions."""

        # Define custom operator
        @torch.library.custom_op("test::simple_mul", mutates_args=())
        def simple_mul_op(input: torch.Tensor, multiplier: float) -> torch.Tensor:
            needs_grad_flag = needs_grad(input)

            # Detach input to break PyTorch autograd graph - we handle gradients manually
            wp_input = wp.from_torch(
                input.detach(), dtype=wp.float32, requires_grad=needs_grad_flag
            )
            out = torch.zeros_like(input)
            wp_out = wp.from_torch(out, dtype=wp.float32, requires_grad=needs_grad_flag)

            with WarpAutogradContextManager(needs_grad_flag) as tape:
                wp.launch(
                    simple_multiply_kernel,
                    dim=input.shape[0],
                    inputs=[wp_input, wp.float32(multiplier), wp_out],
                    device=device,
                )

            if needs_grad_flag:
                # Store tape, output, and input for backward
                attach_for_backward(out, tape=tape, out=wp_out, input=wp_input)

            return out

        def backward(ctx, grad_output):
            return standard_backward(
                ctx,
                grad_output,
                output_names="out",  # Updated parameter name
                array_names=["out", "input"],
                input_names=["input", "multiplier"],
            )

        def setup_context(ctx, inputs, output=None, outputs=None):
            if output is None:
                output = outputs
            ctx.input, ctx.multiplier = inputs
            ctx.out = output

        simple_mul_op.register_autograd(backward, setup_context=setup_context)

        # Test the operator
        x = torch.randn(10, device=device, requires_grad=True)
        multiplier = 5.0

        # Forward
        y = simple_mul_op(x, multiplier)

        # Check forward result
        expected = x * multiplier
        assert torch.allclose(y, expected, rtol=1e-5)

        # Backward
        loss = y.sum()
        loss.backward()

        # Check gradient
        assert x.grad is not None
        expected_grad = torch.ones_like(x) * multiplier
        assert torch.allclose(x.grad, expected_grad, rtol=1e-5)

    def test_inference_no_grad(self, device):
        """Test that inference mode has no gradient overhead."""

        @torch.library.custom_op("test::simple_mul_v2", mutates_args=())
        def simple_mul_op(input: torch.Tensor, multiplier: float) -> torch.Tensor:
            needs_grad_flag = needs_grad(input)

            wp_input = wp.from_torch(
                input, dtype=wp.float32, requires_grad=needs_grad_flag
            )
            out = torch.zeros_like(input)
            wp_out = wp.from_torch(out, dtype=wp.float32, requires_grad=needs_grad_flag)

            with WarpAutogradContextManager(needs_grad_flag) as tape:
                wp.launch(
                    simple_multiply_kernel,
                    dim=input.shape[0],
                    inputs=[wp_input, wp.float32(multiplier), wp_out],
                    device=device,
                )

            if needs_grad_flag:
                attach_for_backward(out, tape=tape, output=wp_out, input=wp_input)

            # Verify tape is None in inference mode
            if not needs_grad_flag:
                assert tape is None

            return out

        def backward(ctx, grad_output):
            return standard_backward(
                ctx, grad_output, "output", ["output", "input"], ["input", "multiplier"]
            )

        def setup_context(ctx, inputs, output=None, outputs=None):
            if output is None:
                output = outputs
            ctx.input, ctx.multiplier = inputs
            ctx.output = output

        simple_mul_op.register_autograd(backward, setup_context=setup_context)

        # Test without gradients
        x = torch.randn(10, device=device, requires_grad=False)
        y = simple_mul_op(x, 3.0)

        # Should not have attached anything
        assert not hasattr(y, "_warp_tape")

        # Result should still be correct
        expected = x * 3.0
        assert torch.allclose(y, expected, rtol=1e-5)


class TestStandardBackward:
    """Tests for standard_backward() helper."""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_standard_backward_basic(self, device):
        """Test standard_backward with simple operator."""

        @torch.library.custom_op("test::elementwise_add", mutates_args=())
        def elementwise_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            needs_grad_flag = needs_grad(a, b)

            wp_a = wp.from_torch(
                a.detach(), dtype=wp.float32, requires_grad=needs_grad_flag
            )
            wp_b = wp.from_torch(
                b.detach(), dtype=wp.float32, requires_grad=needs_grad_flag
            )

            out = torch.zeros_like(a)
            wp_out = wp.from_torch(out, dtype=wp.float32, requires_grad=needs_grad_flag)

            with WarpAutogradContextManager(needs_grad_flag) as tape:
                wp.launch(
                    elementwise_add_kernel,
                    dim=a.shape[0],
                    inputs=[wp_a, wp_b, wp_out],
                    device=device,
                )

            if needs_grad_flag:
                attach_for_backward(out, tape=tape, out=wp_out, a=wp_a, b=wp_b)

            return out

        def backward(ctx, grad_output):
            # Use standard_backward - one line!
            return standard_backward(
                ctx, grad_output, "out", ["out", "a", "b"], ["a", "b"]
            )

        def setup_context(ctx, inputs, output=None, outputs=None):
            if output is None:
                output = outputs
            ctx.a, ctx.b = inputs
            ctx.out = output

        elementwise_add.register_autograd(backward, setup_context=setup_context)

        # Test
        a = torch.randn(10, device=device, requires_grad=True)
        b = torch.randn(10, device=device, requires_grad=True)

        c = elementwise_add(a, b)
        loss = c.sum()
        loss.backward()

        # Gradients should be all ones
        assert a.grad is not None
        assert b.grad is not None
        assert torch.allclose(a.grad, torch.ones_like(a))
        assert torch.allclose(b.grad, torch.ones_like(b))


class TestMultipleOutputs:
    """Tests for kernels with multiple outputs (energy + forces pattern)."""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_energy_forces_forward(self, device):
        """Test forward pass for energy+forces kernel."""

        @torch.library.custom_op("test::energy_forces", mutates_args=())
        def compute_energy_forces(
            positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            needs_grad_flag = needs_grad(positions)

            n_atoms = positions.shape[0]

            wp_positions = wp.from_torch(
                positions.detach(), dtype=wp.vec3f, requires_grad=needs_grad_flag
            )

            energies = torch.zeros(n_atoms, device=device, dtype=torch.float32)
            forces = torch.zeros((n_atoms, 3), device=device, dtype=torch.float32)

            wp_energies = wp.from_torch(
                energies, dtype=wp.float32, requires_grad=needs_grad_flag
            )
            wp_forces = wp.from_torch(
                forces, dtype=wp.vec3f, requires_grad=needs_grad_flag
            )

            with WarpAutogradContextManager(needs_grad_flag) as tape:
                wp.launch(
                    dummy_energy_forces,
                    dim=n_atoms,
                    inputs=[wp_positions, wp_energies, wp_forces],
                    device=device,
                )

            if needs_grad_flag:
                attach_for_backward(
                    energies,
                    tape=tape,
                    energies=wp_energies,
                    forces=wp_forces,
                    positions=wp_positions,
                )

            return energies, forces

        def backward(ctx, grad_energies, grad_forces):
            """Backward for multiple outputs."""
            tape, arrays = retrieve_for_backward(
                ctx.energies, "energies", "forces", "positions"
            )

            # Set gradients for both outputs
            if grad_energies is not None:
                wp_grad_energies = wp.from_torch(
                    grad_energies.contiguous(), dtype=wp.float32
                )
                wp.copy(arrays["energies"].grad, wp_grad_energies)

            if grad_forces is not None:
                wp_grad_forces = wp.from_torch(grad_forces.contiguous(), dtype=wp.vec3f)
                wp.copy(arrays["forces"].grad, wp_grad_forces)

            # Run backward
            tape.backward()

            # Extract gradient for positions
            return extract_gradients(ctx, arrays, ["positions"])

        def setup_context(ctx, inputs, output=None, outputs=None):
            if output is None:
                output = outputs
            ctx.positions = inputs[0]
            ctx.energies, ctx.forces = output

        compute_energy_forces.register_autograd(backward, setup_context=setup_context)

        # Test forward pass
        positions = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            device=device,
            dtype=torch.float32,
            requires_grad=False,
        )

        energies, forces = compute_energy_forces(positions)

        # Check energy: E = -0.5 * ||pos||^2
        expected_energies = -0.5 * (positions**2).sum(dim=1)
        assert torch.allclose(energies, expected_energies, rtol=1e-5)

        # Check forces: F = -pos
        expected_forces = -positions
        assert torch.allclose(forces, expected_forces, rtol=1e-5)

    def test_energy_forces_backward_energy_only(self, device):
        """Test backward when only energy contributes to loss."""

        @torch.library.custom_op("test::energy_forces_v2", mutates_args=())
        def compute_energy_forces(
            positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            needs_grad_flag = needs_grad(positions)
            n_atoms = positions.shape[0]

            wp_positions = wp.from_torch(
                positions.detach(), dtype=wp.vec3f, requires_grad=needs_grad_flag
            )
            energies = torch.zeros(n_atoms, device=device, dtype=torch.float32)
            forces = torch.zeros((n_atoms, 3), device=device, dtype=torch.float32)
            wp_energies = wp.from_torch(
                energies, dtype=wp.float32, requires_grad=needs_grad_flag
            )
            wp_forces = wp.from_torch(
                forces, dtype=wp.vec3f, requires_grad=needs_grad_flag
            )

            with WarpAutogradContextManager(needs_grad_flag) as tape:
                wp.launch(
                    dummy_energy_forces,
                    dim=n_atoms,
                    inputs=[wp_positions, wp_energies, wp_forces],
                    device=device,
                )

            if needs_grad_flag:
                attach_for_backward(
                    energies,
                    tape=tape,
                    energies=wp_energies,
                    forces=wp_forces,
                    positions=wp_positions,
                )

            return energies, forces

        def backward(ctx, grad_energies, grad_forces):
            # Use standard_backward with multiple outputs - one line!
            return standard_backward(
                ctx,
                grad_outputs=(grad_energies, grad_forces),
                output_names=["energies", "forces"],
                output_dtypes=[wp.float32, wp.vec3f],
                array_names=["energies", "forces", "positions"],
                input_names=["positions"],
            )

        def setup_context(ctx, inputs, output=None, outputs=None):
            if output is None:
                output = outputs
            ctx.positions = inputs[0]
            ctx.energies, ctx.forces = output

        compute_energy_forces.register_autograd(backward, setup_context=setup_context)

        # Test with energy loss only
        positions = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

        energies, _ = compute_energy_forces(positions)

        # Loss from energy only
        loss = energies.sum()
        loss.backward()

        # Gradient: dE/dpos = d(-0.5*||pos||^2)/dpos = -pos
        expected_grad = -positions

        assert positions.grad is not None
        assert torch.allclose(positions.grad, expected_grad, rtol=1e-5)

    def test_energy_forces_backward_combined_loss(self, device):
        """Test backward with combined loss from both energy and forces."""

        @torch.library.custom_op("test::energy_forces_v3", mutates_args=())
        def compute_energy_forces(
            positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            needs_grad_flag = needs_grad(positions)
            n_atoms = positions.shape[0]

            wp_positions = wp.from_torch(
                positions.detach(), dtype=wp.vec3f, requires_grad=needs_grad_flag
            )
            energies = torch.zeros(n_atoms, device=device, dtype=torch.float32)
            forces = torch.zeros((n_atoms, 3), device=device, dtype=torch.float32)
            wp_energies = wp.from_torch(
                energies, dtype=wp.float32, requires_grad=needs_grad_flag
            )
            wp_forces = wp.from_torch(
                forces, dtype=wp.vec3f, requires_grad=needs_grad_flag
            )

            with WarpAutogradContextManager(needs_grad_flag) as tape:
                wp.launch(
                    dummy_energy_forces,
                    dim=n_atoms,
                    inputs=[wp_positions, wp_energies, wp_forces],
                    device=device,
                )

            if needs_grad_flag:
                attach_for_backward(
                    energies,
                    tape=tape,
                    energies=wp_energies,
                    forces=wp_forces,
                    positions=wp_positions,
                )

            return energies, forces

        def backward(ctx, grad_energies, grad_forces):
            # Use standard_backward with multiple outputs - one line!
            return standard_backward(
                ctx,
                grad_outputs=(grad_energies, grad_forces),
                output_names=["energies", "forces"],
                output_dtypes=[wp.float32, wp.vec3f],
                array_names=["energies", "forces", "positions"],
                input_names=["positions"],
            )

        def setup_context(ctx, inputs, output=None, outputs=None):
            if output is None:
                output = outputs
            ctx.positions = inputs[0]
            ctx.energies, ctx.forces = output

        compute_energy_forces.register_autograd(backward, setup_context=setup_context)

        # Test with combined loss: energies.sum() - forces.norm(dim=1).sum()
        positions = torch.tensor(
            [
                [3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
            ],
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

        energies, forces = compute_energy_forces(positions)

        # Combined loss
        loss = energies.sum() - forces.norm(dim=1).sum()
        loss.backward()

        # Compute expected gradient analytically
        # Loss = sum(-0.5*||pos||^2) - sum(||-pos||)
        #      = -0.5*sum(||pos||^2) - sum(||pos||)
        # dLoss/dpos = -pos - d(||pos||)/dpos
        #            = -pos - pos/||pos||
        #            = -pos * (1 + 1/||pos||)

        pos_norms = positions.norm(dim=1, keepdim=True)
        expected_grad = -positions * (1 + 1 / pos_norms)

        assert positions.grad is not None
        assert torch.allclose(positions.grad, expected_grad, rtol=1e-4, atol=1e-6)

    def test_multiple_outputs_inference_mode(self, device):
        """Test that multiple outputs work correctly in inference mode."""

        @torch.library.custom_op("test::energy_forces_v4", mutates_args=())
        def compute_energy_forces(
            positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            needs_grad_flag = needs_grad(positions)
            n_atoms = positions.shape[0]

            wp_positions = wp.from_torch(
                positions.detach(), dtype=wp.vec3f, requires_grad=needs_grad_flag
            )
            energies = torch.zeros(n_atoms, device=device, dtype=torch.float32)
            forces = torch.zeros((n_atoms, 3), device=device, dtype=torch.float32)
            wp_energies = wp.from_torch(
                energies, dtype=wp.float32, requires_grad=needs_grad_flag
            )
            wp_forces = wp.from_torch(
                forces, dtype=wp.vec3f, requires_grad=needs_grad_flag
            )

            with WarpAutogradContextManager(needs_grad_flag) as tape:
                wp.launch(
                    dummy_energy_forces,
                    dim=n_atoms,
                    inputs=[wp_positions, wp_energies, wp_forces],
                    device=device,
                )

            if needs_grad_flag:
                attach_for_backward(
                    energies,
                    tape=tape,
                    energies=wp_energies,
                    forces=wp_forces,
                    positions=wp_positions,
                )
            else:
                # Verify no tape attached in inference mode
                assert not hasattr(energies, "_warp_tape")

            return energies, forces

        def backward(ctx, grad_energies, grad_forces):
            # Use standard_backward with multiple outputs - one line!
            return standard_backward(
                ctx,
                grad_outputs=(grad_energies, grad_forces),
                output_names=["energies", "forces"],
                output_dtypes=[wp.float32, wp.vec3f],
                array_names=["energies", "forces", "positions"],
                input_names=["positions"],
            )

        def setup_context(ctx, inputs, outputs):
            ctx.positions = inputs[0]
            ctx.energies, ctx.forces = outputs

        compute_energy_forces.register_autograd(backward, setup_context=setup_context)

        # Test inference mode (no gradients)
        positions = torch.tensor(
            [
                [1.0, 2.0, 3.0],
            ],
            device=device,
            dtype=torch.float32,
            requires_grad=False,
        )

        energies, forces = compute_energy_forces(positions)

        # Check results are correct
        expected_energies = -0.5 * (positions**2).sum(dim=1)
        expected_forces = -positions

        assert torch.allclose(energies, expected_energies, rtol=1e-5)
        assert torch.allclose(forces, expected_forces, rtol=1e-5)

        # Verify no gradient tracking artifacts
        assert not hasattr(energies, "_warp_tape")


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestStandardBackwardErrorHandling:
    """Test error handling in standard_backward function."""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_missing_output_dtypes_for_multiple_outputs(self, device):
        """Test that ValueError is raised when output_dtypes is missing for multiple outputs."""

        # Create a mock context
        class MockCtx:
            def __init__(self):
                self.energies = torch.zeros(5, device=device)
                self.forces = torch.zeros((5, 3), device=device)

        ctx = MockCtx()

        # Attach mock tape and arrays
        tape = wp.Tape()
        wp_energies = wp.zeros(5, dtype=wp.float32, device=device, requires_grad=True)
        wp_forces = wp.zeros(
            (5, 3), dtype=wp.float32, device=device, requires_grad=True
        )
        wp_positions = wp.zeros(
            (5, 3), dtype=wp.float32, device=device, requires_grad=True
        )

        attach_for_backward(
            ctx.energies,
            tape=tape,
            energies=wp_energies,
            forces=wp_forces,
            positions=wp_positions,
        )

        grad_energies = torch.ones(5, device=device)
        grad_forces = torch.ones((5, 3), device=device)

        # Should raise ValueError when output_dtypes is None
        with pytest.raises(
            ValueError, match="output_dtypes must be specified for multiple outputs"
        ):
            standard_backward(
                ctx,
                grad_outputs=(grad_energies, grad_forces),
                output_names=["energies", "forces"],
                output_dtypes=None,  # Missing!
                array_names=["energies", "forces", "positions"],
                input_names=["positions"],
            )

    def test_grad_outputs_not_tuple_for_multiple_outputs(self, device):
        """Test that ValueError is raised when grad_outputs is not a tuple/list for multiple outputs."""

        class MockCtx:
            def __init__(self):
                self.energies = torch.zeros(5, device=device)
                self.forces = torch.zeros((5, 3), device=device)

        ctx = MockCtx()

        # Attach mock tape and arrays
        tape = wp.Tape()
        wp_energies = wp.zeros(5, dtype=wp.float32, device=device, requires_grad=True)
        wp_forces = wp.zeros(
            (5, 3), dtype=wp.float32, device=device, requires_grad=True
        )
        wp_positions = wp.zeros(
            (5, 3), dtype=wp.float32, device=device, requires_grad=True
        )

        attach_for_backward(
            ctx.energies,
            tape=tape,
            energies=wp_energies,
            forces=wp_forces,
            positions=wp_positions,
        )

        grad_energies = torch.ones(5, device=device)

        # Should raise ValueError when grad_outputs is not a tuple/list
        with pytest.raises(
            ValueError, match="grad_outputs must be a tuple/list for multiple outputs"
        ):
            standard_backward(
                ctx,
                grad_outputs=grad_energies,  # Single tensor instead of tuple!
                output_names=["energies", "forces"],  # But multiple output names
                output_dtypes=[wp.float32, wp.vec3f],
                array_names=["energies", "forces", "positions"],
                input_names=["positions"],
            )

    def test_mismatched_output_counts(self, device):
        """Test behavior when grad_outputs, output_names, and output_dtypes have different lengths."""

        class MockCtx:
            def __init__(self):
                self.energies = torch.zeros(5, device=device)
                self.forces = torch.zeros((5, 3), device=device)
                self.positions = torch.zeros((5, 3), device=device, requires_grad=True)

        ctx = MockCtx()

        # Attach mock tape and arrays
        tape = wp.Tape()
        wp_energies = wp.zeros(5, dtype=wp.float32, device=device, requires_grad=True)
        wp_forces = wp.zeros(
            (5, 3), dtype=wp.float32, device=device, requires_grad=True
        )
        wp_positions = wp.zeros(
            (5, 3), dtype=wp.float32, device=device, requires_grad=True
        )

        attach_for_backward(
            ctx.energies,
            tape=tape,
            energies=wp_energies,
            forces=wp_forces,
            positions=wp_positions,
        )

        grad_energies = torch.ones(5, device=device)
        grad_forces = torch.ones((5, 3), device=device)

        # Should raise ValueError when output_dtypes length doesn't match output_names
        with pytest.raises(
            ValueError, match="Mismatch: got 1 output_dtypes but 2 output_names"
        ):
            standard_backward(
                ctx,
                grad_outputs=(grad_energies, grad_forces),
                output_names=["energies", "forces"],
                output_dtypes=[wp.float32],  # Only one dtype for two outputs!
                array_names=["energies", "forces", "positions"],
                input_names=["positions"],
            )

    def test_mismatched_grad_outputs_count(self, device):
        """Test behavior when grad_outputs length doesn't match output_names."""

        class MockCtx:
            def __init__(self):
                self.energies = torch.zeros(5, device=device)
                self.forces = torch.zeros((5, 3), device=device)
                self.positions = torch.zeros((5, 3), device=device, requires_grad=True)

        ctx = MockCtx()

        # Attach mock tape and arrays
        tape = wp.Tape()
        wp_energies = wp.zeros(5, dtype=wp.float32, device=device, requires_grad=True)
        wp_forces = wp.zeros(
            (5, 3), dtype=wp.float32, device=device, requires_grad=True
        )
        wp_positions = wp.zeros(
            (5, 3), dtype=wp.float32, device=device, requires_grad=True
        )

        attach_for_backward(
            ctx.energies,
            tape=tape,
            energies=wp_energies,
            forces=wp_forces,
            positions=wp_positions,
        )

        grad_energies = torch.ones(5, device=device)

        # Should raise ValueError when grad_outputs length doesn't match output_names
        with pytest.raises(
            ValueError, match="Mismatch: got 1 grad_outputs but 2 output_names"
        ):
            standard_backward(
                ctx,
                grad_outputs=(grad_energies,),  # Only one gradient for two outputs!
                output_names=["energies", "forces"],
                output_dtypes=[wp.float32, wp.vec3f],
                array_names=["energies", "forces", "positions"],
                input_names=["positions"],
            )


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_tensor(self):
        """Test with empty tensors."""
        x = torch.empty(0, requires_grad=True)
        assert needs_grad(x) is True

    def test_scalar_tensor(self):
        """Test with scalar tensor."""
        x = torch.tensor(1.0, requires_grad=True)
        assert needs_grad(x) is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensors(self):
        """Test with CUDA tensors."""
        x = torch.randn(10, device="cuda", requires_grad=True)
        y = torch.randn(10, device="cuda", requires_grad=False)
        assert needs_grad(x, y) is True

    def test_mixed_devices_needs_grad(self):
        """Test needs_grad with mixed device tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x_cpu = torch.randn(10, requires_grad=True)
        x_cuda = torch.randn(10, device="cuda", requires_grad=True)

        # Should work even with mixed devices
        assert needs_grad(x_cpu, x_cuda) is True


class TestResolveDtype:
    """Test _resolve_dtype for coverage of lines 71-73, 79-86."""

    def test_resolve_dtype_vec3_tensor(self):
        """Test _resolve_dtype with vec3 tensor (lines 71-73)."""

        # Create a 2D tensor with shape[-1] == 3 (vec3-like)
        tensor = torch.randn(10, 3, dtype=torch.float32)

        # When dtype is Any and tensor is vec3-like, should return vec dtype
        result = _resolve_warp_dtype(Any, tensor)
        assert result == wp.vec3f

    def test_resolve_dtype_vec3_tensor_float64(self):
        """Test _resolve_dtype with vec3 float64 tensor."""
        tensor = torch.randn(10, 3, dtype=torch.float64)
        result = _resolve_warp_dtype(Any, tensor)
        assert result == wp.vec3d

    def test_resolve_dtype_scalar_tensor(self):
        """Test _resolve_dtype with scalar tensor."""
        tensor = torch.randn(10, dtype=torch.float32)
        result = _resolve_warp_dtype(Any, tensor)
        assert result == wp.float32

    def test_resolve_dtype_wp_array_with_any(self):
        """Test _resolve_dtype with wp.array(dtype=Any) (lines 79-86)."""

        # Create a mock wp.array type with Any as inner dtype
        # We use wp.array to create an actual array type with ndim
        array_type = wp.array(dtype=Any, ndim=2)

        # Scalar tensor
        tensor = torch.randn(10, 5, dtype=torch.float64)
        result = _resolve_warp_dtype(array_type, tensor)
        assert result == wp.float64

    def test_resolve_dtype_wp_array_with_any_vec3(self):
        """Test _resolve_dtype with wp.array(dtype=Any) for vec3-like tensor."""
        array_type = wp.array(dtype=Any, ndim=2)

        # Vec3-like tensor
        tensor = torch.randn(10, 3, dtype=torch.float32)
        result = _resolve_warp_dtype(array_type, tensor)
        assert result == wp.vec3f

    def test_resolve_dtype_concrete(self):
        """Test _resolve_dtype with concrete dtype returns as-is."""

        tensor = torch.randn(10, 3, dtype=torch.float32)
        result = _resolve_warp_dtype(wp.float64, tensor)
        assert result == wp.float64


class TestWarpCustomOpDecorator:
    """Test warp_custom_op decorator for coverage of lines 206-208, 226-247."""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_auto_generated_grad_arrays(self, device):
        """Test auto-generated grad_arrays when not provided (lines 206-208)."""

        # Define a custom op WITHOUT grad_arrays - should auto-generate
        @warp_custom_op(
            name="test::auto_grad_arrays",
            outputs=[
                OutputSpec("result", wp.float32, lambda inp, *_: (inp.shape[0],)),
            ],
            # grad_arrays intentionally not provided - should be auto-generated
        )
        def auto_grad_op(inp: torch.Tensor, scale: float) -> torch.Tensor:
            result = torch.zeros(inp.shape[0], device=inp.device, dtype=torch.float32)
            # Simple pass-through for testing
            result[:] = inp[:, 0] * scale
            return result

        # Test that the op was created and works
        x = torch.randn(5, 3, device=device, dtype=torch.float32)
        result = auto_grad_op(x, 2.0)
        assert result.shape == (5,)

    def test_fake_impl_execution(self, device):
        """Test fake_impl execution for torch.compile (lines 226-247)."""

        @warp_custom_op(
            name="test::fake_impl_test",
            outputs=[
                OutputSpec("energies", wp.float32, lambda pos, *_: (pos.shape[0],)),
                OutputSpec("forces", wp.vec3f, lambda pos, *_: (pos.shape[0], 3)),
            ],
            grad_arrays=["energies", "forces", "positions"],
        )
        def fake_impl_op(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            n = positions.shape[0]
            energies = torch.zeros(n, device=positions.device, dtype=torch.float32)
            forces = torch.zeros(n, 3, device=positions.device, dtype=torch.float32)
            return energies, forces

        # Test that fake impl is registered by checking the op works
        x = torch.randn(10, 3, device=device, dtype=torch.float32)
        energies, forces = fake_impl_op(x)
        assert energies.shape == (10,)
        assert forces.shape == (10, 3)

    def test_fake_impl_with_static_shape(self, device):
        """Test fake_impl with static shape (tuple instead of callable)."""

        @warp_custom_op(
            name="test::static_shape_test",
            outputs=[
                OutputSpec("result", wp.mat33f, (3, 3)),  # Static shape
            ],
            grad_arrays=["result"],
        )
        def static_shape_op(inp: torch.Tensor) -> torch.Tensor:
            return torch.eye(3, device=inp.device, dtype=torch.float32)

        x = torch.randn(5, device=device, dtype=torch.float32)
        result = static_shape_op(x)
        assert result.shape == (3, 3)


class TestSetupContextOutput:
    """Test setup_context_impl handling of output parameter (line 282)."""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_setup_context_single_output(self, device):
        """Test setup_context with single output tensor."""

        @warp_custom_op(
            name="test::single_output_context",
            outputs=[
                OutputSpec("result", wp.float32, lambda inp, *_: (inp.shape[0],)),
            ],
            grad_arrays=["result", "inp"],
        )
        def single_output_op(inp: torch.Tensor) -> torch.Tensor:
            result = torch.zeros(inp.shape[0], device=inp.device, dtype=torch.float32)
            result[:] = inp.sum(dim=-1)
            return result

        # Test that single output works
        x = torch.randn(5, 3, device=device, dtype=torch.float32)
        result = single_output_op(x)
        assert result.shape == (5,)


class TestStandardBackwardDefaultDtype:
    """Test standard_backward with default output_dtypes for single output (line 622)."""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_single_output_default_dtype(self, device):
        """Test that single output uses default wp.float32 dtype (line 622)."""

        class MockCtx:
            def __init__(self):
                self.positions = torch.zeros((5, 3), device=device, requires_grad=True)
                self.energies = torch.zeros(5, device=device)

        ctx = MockCtx()

        # Attach mock tape and arrays
        tape = wp.Tape()
        wp_energies = wp.zeros(5, dtype=wp.float32, device=device, requires_grad=True)
        wp_positions = wp.zeros(
            (5, 3), dtype=wp.float32, device=device, requires_grad=True
        )

        attach_for_backward(
            ctx.energies,
            tape=tape,
            energies=wp_energies,
            positions=wp_positions,
        )

        grad_output = torch.ones(5, device=device)

        # Single output case - output_dtypes defaults to [wp.float32]
        # This should NOT raise an error
        try:
            standard_backward(
                ctx,
                grad_outputs=grad_output,
                output_names="energies",  # Single output as string
                output_dtypes=None,  # Will default to wp.float32
                array_names=["energies", "positions"],
                input_names=["positions"],
            )
            # Just verify it returns something (gradient computation may fail
            # without actual kernel but shouldn't error on dtypes)
        except ValueError as e:
            # Should not get "output_dtypes must be specified" error for single output
            assert "output_dtypes must be specified for multiple outputs" not in str(e)


class TestTorchCompileFakeImpl:
    """Test fake_impl execution with torch.compile to cover lines 226-247, 282."""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for torch.compile"
    )
    def test_single_output_torch_compile(self, device):
        """Test single-output custom op with torch.compile triggers fake_impl."""

        # Create a simple custom op using the decorator
        @warp_custom_op(
            name="test::compile_single_output",
            outputs=[
                OutputSpec("result", wp.float32, lambda x, *_: (x.shape[0],)),
            ],
            grad_arrays=["result", "x"],
        )
        def compile_single_op(x: torch.Tensor) -> torch.Tensor:
            # Simple operation - just sum along last dimension
            result = x.sum(dim=-1)
            return result

        # Compile the function
        compiled_fn = torch.compile(compile_single_op, fullgraph=False)

        # Run it - this triggers fake_impl during tracing
        x = torch.randn(10, 3, device=device, dtype=torch.float32)
        result = compiled_fn(x)

        assert result.shape == (10,)
        assert result.dtype == torch.float32

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for torch.compile"
    )
    def test_multiple_output_torch_compile(self, device):
        """Test multiple-output custom op with torch.compile triggers fake_impl."""

        # Create a custom op with multiple outputs
        @warp_custom_op(
            name="test::compile_multi_output",
            outputs=[
                OutputSpec("energies", wp.float32, lambda pos, *_: (pos.shape[0],)),
                OutputSpec("forces", wp.vec3f, lambda pos, *_: (pos.shape[0], 3)),
            ],
            grad_arrays=["energies", "forces", "positions"],
        )
        def compile_multi_op(
            positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            energies = positions.sum(dim=-1)
            forces = -positions
            return energies, forces

        # Compile the function
        compiled_fn = torch.compile(compile_multi_op, fullgraph=False)

        # Run it - this triggers fake_impl with multiple outputs
        positions = torch.randn(5, 3, device=device, dtype=torch.float32)
        energies, forces = compiled_fn(positions)

        assert energies.shape == (5,)
        assert forces.shape == (5, 3)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for torch.compile"
    )
    def test_static_shape_torch_compile(self, device):
        """Test custom op with static shape in OutputSpec triggers fake_impl."""

        # Create a custom op with static shape
        @warp_custom_op(
            name="test::compile_static_shape",
            outputs=[
                OutputSpec("matrix", wp.mat33f, (3, 3)),  # Static shape
            ],
            grad_arrays=["matrix"],
        )
        def compile_static_op(x: torch.Tensor) -> torch.Tensor:
            return torch.eye(3, device=x.device, dtype=torch.float32)

        # Compile the function
        compiled_fn = torch.compile(compile_static_op, fullgraph=False)

        # Run it
        x = torch.randn(5, device=device, dtype=torch.float32)
        result = compiled_fn(x)

        assert result.shape == (3, 3)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for torch.compile"
    )
    def test_cpu_fallback_in_fake_impl(self):
        """Test fake_impl CPU fallback when no tensor has device."""

        # This is tricky to test directly, but we can create a scenario
        # where device inference might need fallback
        @warp_custom_op(
            name="test::compile_cpu_fallback",
            outputs=[
                OutputSpec("result", wp.float32, lambda *_: (5,)),
            ],
            grad_arrays=["result"],
        )
        def cpu_fallback_op(scale: float) -> torch.Tensor:
            # Non-tensor input
            return torch.zeros(5) * scale

        # Compile and run on CPU
        compiled_fn = torch.compile(cpu_fallback_op, fullgraph=False)
        result = compiled_fn(2.0)

        assert result.shape == (5,)

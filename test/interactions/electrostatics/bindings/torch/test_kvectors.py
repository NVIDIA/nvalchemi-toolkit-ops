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
Tests for k-vector generation utilities.

Tests cover:
- Ewald summation k-vector generation
- PME k-vector generation
- Comparison against torchpme reference implementations
- Batch support for multiple systems
"""

from importlib import import_module

import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
    generate_k_vectors_pme,
)

try:
    _ = import_module("ase")
    HAS_ASE = True
except ModuleNotFoundError:
    HAS_ASE = False

try:
    _ = import_module("torchpme")
    HAS_TORCHPME = True
    from torchpme.lib.kvectors import _generate_kvectors as _generate_kvectors_torchpme
except ModuleNotFoundError:
    HAS_TORCHPME = False

# Import test utilities for crystal structure generation
from .test_utils import (
    create_cscl_supercell,
    create_wurtzite_system,
    create_zincblende_system,
)


def generate_kvectors_for_pme_reference(cell, mesh_dimensions):
    """Generate k-vectors using torchpme as reference for PME."""
    ns = torch.tensor(mesh_dimensions).to(cell.device)
    kvectors = _generate_kvectors_torchpme(cell, ns, for_ewald=False)
    return kvectors


###########################################################################################
########################### Ewald K-Vector Tests ##########################################
###########################################################################################


class TestKVectorsEwald:
    """Test k-vector generation for Ewald summation."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_output_shape_single_system(self, device):
        """Test output shape for single system."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        # Should be (K, 3) for single system (squeezed)
        assert k_vectors.ndim == 2
        assert k_vectors.shape[1] == 3
        assert k_vectors.shape[0] > 0

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_output_shape_batch(self, device):
        """Test output shape for batch of systems."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(3, -1, -1)
            * 10.0
        )
        k_vectors = generate_k_vectors_ewald_summation(cell.contiguous(), k_cutoff=8.0)

        # Should be (B, K, 3) for batch
        assert k_vectors.ndim == 3
        assert k_vectors.shape[0] == 3
        assert k_vectors.shape[2] == 3

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_larger_cutoff_more_vectors(self, device):
        """Test that larger cutoff produces more k-vectors."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0

        k_vectors_small = generate_k_vectors_ewald_summation(cell, k_cutoff=5.0)
        k_vectors_large = generate_k_vectors_ewald_summation(cell, k_cutoff=10.0)

        assert k_vectors_large.shape[0] > k_vectors_small.shape[0]


###########################################################################################
########################### PME K-Vector Tests ############################################
###########################################################################################


class TestKVectorsPME:
    """Test k-vector generation for PME."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_output_shapes(self, device):
        """Test output shapes for PME k-vectors."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        mesh_dims = (16, 16, 16)

        k_vectors, k_squared_safe = generate_k_vectors_pme(cell, mesh_dims)

        # k_vectors shape: (nx, ny, nz/2+1, 3)
        assert k_vectors.shape == (16, 16, 9, 3)
        # k_squared_safe shape: (nx, ny, nz/2+1)
        assert k_squared_safe.shape == (16, 16, 9)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_k_squared_positive(self, device):
        """Test that k_squared_safe is always positive (avoids division by zero)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        mesh_dims = (16, 16, 16)

        _, k_squared_safe = generate_k_vectors_pme(cell, mesh_dims)

        assert (k_squared_safe > 0).all(), "k_squared_safe should always be positive"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_k_zero_has_safe_value(self, device):
        """Test that k=0 has a safe non-zero k² value."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        mesh_dims = (16, 16, 16)

        k_vectors, k_squared_safe = generate_k_vectors_pme(cell, mesh_dims)

        # k=0 is at index [0, 0, 0]
        k_zero = k_vectors[0, 0, 0]
        k_sq_zero = k_squared_safe[0, 0, 0]

        assert torch.norm(k_zero, dim=0) < 1e-10, "k[0,0,0] should be zero"
        assert k_sq_zero > 0, "k_squared_safe[0,0,0] should be non-zero for safety"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("mesh_dims", [(8, 8, 8), (16, 16, 16), (32, 32, 32)])
    def test_different_mesh_sizes(self, device, mesh_dims):
        """Test different mesh dimensions."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0

        k_vectors, k_squared_safe = generate_k_vectors_pme(cell, mesh_dims)

        nx, ny, nz = mesh_dims
        expected_shape = (nx, ny, nz // 2 + 1)

        assert k_vectors.shape[:3] == expected_shape
        assert k_squared_safe.shape == expected_shape

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_rectangular_mesh(self, device):
        """Test non-cubic mesh dimensions."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        cell = torch.diag(torch.tensor([10.0, 15.0, 20.0], device=device)).unsqueeze(0)
        mesh_dims = (16, 24, 32)

        k_vectors, k_squared_safe = generate_k_vectors_pme(cell, mesh_dims)

        nx, ny, nz = mesh_dims
        expected_shape = (nx, ny, nz // 2 + 1)

        assert k_vectors.shape[:3] == expected_shape
        assert k_squared_safe.shape == expected_shape

    @pytest.mark.skipif(not HAS_TORCHPME, reason="torchpme not installed")
    @pytest.mark.skipif(not HAS_TORCHPME, reason="torchpme not available")
    @pytest.mark.parametrize("size", [1, 2, 3])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    @pytest.mark.parametrize("mesh_dims", [(16, 16, 16), (32, 32, 32)])
    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_k_vectors_match_torchpme(self, size, system_fn, mesh_dims, device):
        """Test k-vector generation matches torchpme reference for PME."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }
        system = system_fns[system_fn](size)
        cell = torch.tensor(system.cell, dtype=torch.float64, device=device)

        k_vectors, _ = generate_k_vectors_pme(cell.unsqueeze(0), mesh_dims)
        k_vectors_reference = generate_kvectors_for_pme_reference(cell, mesh_dims)

        # Reshape for comparison - torchpme returns (nx, ny, nz, 3) but we use rfft
        # so we have (nx, ny, nz/2+1, 3)
        _, _, nz = mesh_dims
        k_vectors_ref_rfft = k_vectors_reference[:, :, : nz // 2 + 1, :]

        assert torch.allclose(k_vectors, k_vectors_ref_rfft, atol=1e-4, rtol=1e-4)


###########################################################################################
########################### Gradient Tests ################################################
###########################################################################################


class TestKVectorGradients:
    """Test that k-vector generation supports autograd through cell."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_ewald_k_vectors_have_gradients(self, device):
        """Test that Ewald k-vectors flow gradients through cell."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        cell = cell.clone().requires_grad_(True)

        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        # Sum to create scalar for backward
        loss = k_vectors.sum()
        loss.backward()

        assert cell.grad is not None
        assert torch.isfinite(cell.grad).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_pme_k_vectors_have_gradients(self, device):
        """Test that PME k-vectors flow gradients through cell."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        cell = cell.clone().requires_grad_(True)

        k_vectors, k_squared_safe = generate_k_vectors_pme(cell, (16, 16, 16))

        # Sum to create scalar for backward
        loss = k_vectors.sum() + k_squared_safe.sum()
        loss.backward()

        assert cell.grad is not None
        assert torch.isfinite(cell.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

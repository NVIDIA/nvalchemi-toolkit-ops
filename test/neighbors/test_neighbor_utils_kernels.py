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

"""Tests for neighbor utility warp launchers."""

import pytest
import torch
import warp as wp

from nvalchemiops.neighbors.neighbor_utils import (
    compute_naive_num_shifts,
    zero_array,
)
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype

from .test_utils import create_simple_cubic_system

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda:0")
dtypes = [torch.float32, torch.float64]


@pytest.mark.parametrize("device", devices)
class TestNeighborUtilsWpLaunchers:
    """Test the public launcher API for neighbor utilities."""

    @pytest.mark.parametrize("dtype", dtypes)
    def test_compute_naive_num_shifts(self, device, dtype):
        """Test compute_naive_num_shifts launcher."""
        _, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        cutoff = 1.5

        # Prepare output arrays
        shift_range_per_dimension = torch.zeros(3, dtype=torch.int32, device=device)
        num_shifts = torch.zeros(1, dtype=torch.int32, device=device)

        # Convert to warp arrays
        wp_dtype = get_wp_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)

        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool)
        wp_shift_range = wp.from_torch(shift_range_per_dimension, dtype=wp.vec3i)
        wp_num_shifts = wp.from_torch(num_shifts, dtype=wp.int32)

        # Call launcher
        compute_naive_num_shifts(
            wp_cell,
            cutoff,
            wp_pbc,
            wp_num_shifts,
            wp_shift_range,
            wp_dtype,
            device,
        )

        # Verify results
        assert torch.all(shift_range_per_dimension >= 0), (
            "Shift ranges should be non-negative"
        )
        assert num_shifts.item() > 0, "Should have at least one shift"

    def test_zero_array(self, device):
        """Test zero_array launcher."""
        # Test data with non-zero values
        test_array = torch.full((100,), 42, dtype=torch.int32, device=device)

        # Convert to warp array
        wp_array = wp.from_torch(test_array, dtype=wp.int32)

        # Call launcher
        zero_array(wp_array, device)

        # Should be all zeros
        assert torch.all(test_array == 0), "Array should be zeroed"

    def test_zero_array_empty(self, device):
        """Test zero_array with empty array."""
        test_array = torch.empty(0, dtype=torch.int32, device=device)

        # Convert to warp array
        wp_array = wp.from_torch(test_array, dtype=wp.int32)

        # Call launcher - should handle gracefully
        zero_array(wp_array, device)

        assert test_array.shape == (0,)

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

"""Shared pytest fixtures for torch neighbors binding tests."""

from importlib import import_module

import pytest
import torch

# =============================================================================
# External dependency checks
# =============================================================================

# Check if vesin is available for consistency checks
try:
    _ = import_module("vesin")
    VESIN_AVAILABLE = True
except ModuleNotFoundError:
    VESIN_AVAILABLE = False

# Pytest marker for tests that require vesin
requires_vesin = pytest.mark.skipif(
    not VESIN_AVAILABLE, reason="`vesin` required for consistency checks."
)

# =============================================================================
# Device configuration
# =============================================================================

# Build list of available devices
AVAILABLE_DEVICES = ["cpu"]
if torch.cuda.is_available():
    AVAILABLE_DEVICES.append("cuda:0")


# =============================================================================
# Core fixtures - device and dtype
# =============================================================================


@pytest.fixture(params=AVAILABLE_DEVICES, ids=lambda d: d.replace(":", "_"))
def device(request):
    """Fixture providing test devices (cpu, cuda:0 if available).

    Returns
    -------
    str
        Device string for torch tensors
    """
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def dtype(request):
    """Fixture providing torch dtypes for testing.

    Returns
    -------
    torch.dtype
        The torch dtype (float32 or float64)
    """
    return request.param


# =============================================================================
# Common parameter fixtures
# =============================================================================


@pytest.fixture(params=[False, True], ids=["full_fill", "half_fill"])
def half_fill(request):
    """Fixture for half_fill parameter in neighbor list functions.

    half_fill=True: Only store each pair once (i < j)
    half_fill=False: Store both (i,j) and (j,i) pairs

    Returns
    -------
    bool
        Whether to use half-fill mode
    """
    return request.param


@pytest.fixture(params=[False, True], ids=["no_pbc", "pbc"])
def pbc_flag(request):
    """Fixture for periodic boundary condition flag.

    Returns
    -------
    bool
        Whether to use periodic boundary conditions
    """
    return request.param


@pytest.fixture(params=[False, True], ids=["matrix", "list"])
def return_neighbor_list(request):
    """Fixture for return format selection.

    return_neighbor_list=False: Return neighbor matrix (N, max_neighbors)
    return_neighbor_list=True: Return COO format neighbor list (2, num_pairs)

    Returns
    -------
    bool
        Whether to return neighbor list format
    """
    return request.param


@pytest.fixture(params=[False, True], ids=["no_preallocate", "preallocate"])
def preallocate(request):
    """Fixture for pre-allocation mode.

    preallocate=True: Caller provides output tensors
    preallocate=False: Function allocates output tensors

    Returns
    -------
    bool
        Whether to pre-allocate output tensors
    """
    return request.param


# =============================================================================
# Helper fixtures for common test scenarios
# =============================================================================


@pytest.fixture
def simple_system(device, dtype):
    """Fixture providing a simple cubic test system.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (positions, cell, pbc) for a simple 8-atom cubic system
    """
    from ...test_utils import create_simple_cubic_system

    return create_simple_cubic_system(
        num_atoms=8, cell_size=2.0, dtype=dtype, device=device
    )


@pytest.fixture
def random_system(device, dtype):
    """Fixture providing a random test system.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (positions, cell, pbc) for a random 20-atom system
    """
    from ...test_utils import create_random_system

    return create_random_system(
        num_atoms=20, cell_size=5.0, dtype=dtype, device=device, seed=42
    )

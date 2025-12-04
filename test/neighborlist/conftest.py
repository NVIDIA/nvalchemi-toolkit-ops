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

"""Pytest configuration for neighbor list tests."""

import pytest
import torch
import warp as wp


def pytest_configure(config):
    """Configure pytest for neighbor list tests."""
    # Add custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow (performance tests)")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "warp: marks tests that require Warp")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark GPU tests
        if "cuda" in item.name.lower() or "gpu" in item.name.lower():
            item.add_marker(pytest.mark.gpu)

        # Mark slow tests
        if "performance" in item.name.lower() or "stress" in item.name.lower():
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session", autouse=True)
def setup_warp():
    """Initialize Warp if available."""
    # Initialize Warp with appropriate settings
    wp.init()

    # Set device preferences
    if torch.cuda.is_available():
        wp.set_device("cuda:0")

    yield

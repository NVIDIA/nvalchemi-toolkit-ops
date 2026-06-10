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

"""Coverage for deprecated core neighbor compatibility facades."""

import importlib
import sys

import pytest


@pytest.mark.parametrize(
    ("module_name", "export_name"),
    [
        ("nvalchemiops.neighbors.batch_cell_list", "batch_query_cell_list"),
        ("nvalchemiops.neighbors.batch_naive", "batch_naive_neighbor_matrix"),
        (
            "nvalchemiops.neighbors.batch_naive_dual_cutoff",
            "batch_naive_neighbor_matrix_dual_cutoff",
        ),
        (
            "nvalchemiops.neighbors.naive_dual_cutoff",
            "naive_neighbor_matrix_dual_cutoff",
        ),
    ],
)
def test_deprecated_neighbor_facades_warn_and_export(module_name, export_name):
    """Deprecated import paths still warn and expose their launcher symbols."""
    sys.modules.pop(module_name, None)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        module = importlib.import_module(module_name)

    assert export_name in module.__all__
    assert hasattr(module, export_name)

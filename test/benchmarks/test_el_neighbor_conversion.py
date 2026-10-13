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

"""Exact packed-row conversion for the electrostatics benchmark setup."""

import pytest
import torch

from benchmarks.interactions.electrostatics.benchmark_electrostatics_suite import (
    _el_build_nl,
    _torch_neighbor_matrix_indices_chunked,
    _torch_neighbor_shift_matrix_to_list_chunked,
)
from nvalchemiops.torch.neighbors import batch_cell_list
from nvalchemiops.torch.neighbors.neighbor_utils import (
    get_neighbor_list_from_neighbor_matrix,
)


class TestElNeighborConversion:
    """Conversion preserves the public atom-centric row and image order."""

    @pytest.mark.parametrize("counts", [[], [0, 0], [2, 0, 1], [3, 1, 2]])
    def test_packed_rows(self, counts):
        """Indices and shifts retain the same complete CSR ordering."""
        width, fill_value = 3, 999
        counts = torch.tensor(counts, dtype=torch.int32)
        matrix = torch.full((len(counts), width), fill_value, dtype=torch.int32)
        shifts = torch.arange(len(counts) * width * 3, dtype=torch.int32).reshape(
            len(counts), width, 3
        )
        for row, count in enumerate(counts):
            matrix[row, :count] = torch.arange(count, dtype=torch.int32)
        expected = get_neighbor_list_from_neighbor_matrix(
            matrix, counts, neighbor_shift_matrix=shifts, fill_value=fill_value
        )
        pairs, ptr = _torch_neighbor_matrix_indices_chunked(
            matrix, counts, fill_value=fill_value
        )
        del matrix
        compact_shifts = _torch_neighbor_shift_matrix_to_list_chunked(
            shifts, counts, ptr
        )
        for actual, wanted in zip((pairs, ptr, compact_shifts), expected, strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

    def test_overflow(self):
        """Required row counts above matrix capacity raise before copying."""
        with pytest.raises(ValueError, match="smaller than observed"):
            _torch_neighbor_matrix_indices_chunked(
                torch.ones((1, 2), dtype=torch.int32),
                torch.tensor([3], dtype=torch.int32),
                fill_value=999,
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("cutoff", [1.00000012, 18.0])
    def test_public_float64_images(self, cutoff):
        """Setup keeps float64 cutoff boundaries and repeated periodic images."""
        positions = torch.tensor(
            [[0, 0, 0], [1, 2**-11, 0]], device="cuda", dtype=torch.float64
        )
        cell = torch.eye(3, device="cuda", dtype=torch.float64)[None] * 16
        pbc = torch.ones((1, 3), device="cuda", dtype=torch.bool)
        batch_idx = torch.zeros(2, device="cuda", dtype=torch.int32)
        matrix, counts, shifts = batch_cell_list(
            positions=positions,
            cutoff=cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            max_neighbors=64,
            strategy="atom_centric",
            atom_centric_path="direct",
            return_neighbor_list=False,
        )
        expected = get_neighbor_list_from_neighbor_matrix(
            matrix, counts, neighbor_shift_matrix=shifts, fill_value=2
        )
        actual = _el_build_nl(positions, cell, pbc, batch_idx, cutoff, "torch", None)
        for value, wanted in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, wanted, rtol=0, atol=0)
        assert actual[0].shape[1] >= 2

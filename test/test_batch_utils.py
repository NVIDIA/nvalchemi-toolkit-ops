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

"""Tests for nvalchemiops.batch_utils."""

import numpy as np
import pytest
import warp as wp

from nvalchemiops.batch_utils import (
    atom_ptr_to_batch_idx,
    atoms_per_system_from_atom_ptr,
    atoms_per_system_from_batch_idx,
    batch_idx_to_atom_ptr,
    create_atom_ptr,
    create_batch_idx,
)


@pytest.fixture(scope="module")
def device():
    return wp.get_device()


class TestBatchIndex:
    """Tests for batch_utils utilities (strict no-allocation API)."""

    def test_create_atom_ptr(self, device):
        atom_counts_np = np.array([10, 15, 12], dtype=np.int32)
        atom_counts = wp.array(atom_counts_np, device=device)
        atom_ptr = wp.zeros(4, dtype=wp.int32, device=device)

        create_atom_ptr(atom_counts, atom_ptr)
        wp.synchronize()

        expected = np.array([0, 10, 25, 37], dtype=np.int32)
        np.testing.assert_array_equal(atom_ptr.numpy(), expected)

    def test_create_batch_idx(self, device):
        atom_counts_np = np.array([3, 2, 4], dtype=np.int32)
        atom_counts = wp.array(atom_counts_np, device=device)
        atom_ptr = wp.zeros(4, dtype=wp.int32, device=device)
        create_atom_ptr(atom_counts, atom_ptr)

        batch_idx = wp.zeros(9, dtype=wp.int32, device=device)
        create_batch_idx(atom_ptr, batch_idx)
        wp.synchronize()

        expected = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int32)
        np.testing.assert_array_equal(batch_idx.numpy(), expected)

    def test_roundtrip_batch_idx_to_atom_ptr(self, device):
        # Build batch_idx from known counts
        atom_counts_np = np.array([5, 10, 8, 3], dtype=np.int32)
        total = int(atom_counts_np.sum())  # 26
        M = len(atom_counts_np)

        atom_counts = wp.array(atom_counts_np, device=device)
        atom_ptr_src = wp.zeros(M + 1, dtype=wp.int32, device=device)
        create_atom_ptr(atom_counts, atom_ptr_src)
        batch_idx = wp.zeros(total, dtype=wp.int32, device=device)
        create_batch_idx(atom_ptr_src, batch_idx)

        # Convert back
        scratch_counts = wp.zeros(M, dtype=wp.int32, device=device)
        atom_ptr_dst = wp.zeros(M + 1, dtype=wp.int32, device=device)
        batch_idx_to_atom_ptr(batch_idx, scratch_counts, atom_ptr_dst)
        wp.synchronize()

        expected = np.array([0, 5, 15, 23, 26], dtype=np.int32)
        np.testing.assert_array_equal(atom_ptr_dst.numpy(), expected)

    def test_roundtrip_atom_ptr_to_batch_idx(self, device):
        atom_counts_np = np.array([3, 2, 4], dtype=np.int32)
        M = len(atom_counts_np)
        total = int(atom_counts_np.sum())  # 9

        atom_counts = wp.array(atom_counts_np, device=device)
        atom_ptr = wp.zeros(M + 1, dtype=wp.int32, device=device)
        create_atom_ptr(atom_counts, atom_ptr)

        batch_idx = wp.zeros(total, dtype=wp.int32, device=device)
        atom_ptr_to_batch_idx(atom_ptr, batch_idx)
        wp.synchronize()

        expected = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int32)
        np.testing.assert_array_equal(batch_idx.numpy(), expected)

    def test_atoms_per_system_from_ptr(self, device):
        atom_counts_np = np.array([10, 15, 12], dtype=np.int32)
        M = len(atom_counts_np)

        atom_counts_in = wp.array(atom_counts_np, device=device)
        atom_ptr = wp.zeros(M + 1, dtype=wp.int32, device=device)
        create_atom_ptr(atom_counts_in, atom_ptr)

        result = wp.zeros(M, dtype=wp.int32, device=device)
        atoms_per_system_from_atom_ptr(atom_ptr, result)
        wp.synchronize()

        np.testing.assert_array_equal(result.numpy(), atom_counts_np)

    def test_atoms_per_system_from_batch_idx(self, device):
        atom_counts_np = np.array([3, 2, 4], dtype=np.int32)
        M = len(atom_counts_np)
        total = int(atom_counts_np.sum())

        atom_counts_in = wp.array(atom_counts_np, device=device)
        atom_ptr = wp.zeros(M + 1, dtype=wp.int32, device=device)
        create_atom_ptr(atom_counts_in, atom_ptr)
        batch_idx = wp.zeros(total, dtype=wp.int32, device=device)
        create_batch_idx(atom_ptr, batch_idx)

        result = wp.zeros(M, dtype=wp.int32, device=device)
        atoms_per_system_from_batch_idx(batch_idx, result)
        wp.synchronize()

        np.testing.assert_array_equal(result.numpy(), atom_counts_np)


class TestBoundedSegmentOffsets:
    """Check generic row/tile metadata independently of matrix multiplication."""

    @pytest.mark.parametrize(
        "groups", [0, 4, 32, 33, 64, 65, 128, 129, 256, 257, 512, 513, 1024]
    )
    @pytest.mark.parametrize("capacity", [0, 2048, 2**30 + 17])
    @pytest.mark.parametrize(
        "pattern",
        [
            [0, -1, -2147483648, 0],
            [65535, 65536, 65537, 1],
            [2147483647, 2147483647, -2147483648, 0],
        ],
        ids=["empty", "limb_carry", "wide_overflow"],
    )
    def test_cooperative_fused_prefixes(self, device, groups, capacity, pattern):
        """Cooperative limb scans must match wide arithmetic on strided inputs."""
        import nvalchemiops.batch_utils as batch_utils

        values = np.resize(np.asarray(pattern, np.int32), groups)
        storage = wp.array(np.repeat(values, 2), device=device)
        lengths = storage[::2] if groups else storage
        outputs = [
            wp.array(np.full(groups + 1, -99, np.int32), device=device)
            for _ in range(3)
        ]
        bucket = max(32, 1 << (groups - 1).bit_length())
        wp.launch_tiled(
            batch_utils._segment_offsets_cooperative[bucket],
            dim=(1,),
            inputs=[
                lengths,
                capacity,
                32,
                False,
                outputs[0],
                outputs[1],
                256,
                outputs[2],
                0,
            ],
            block_dim=min(bucket, 128),
            device=device,
        )
        rows = np.concatenate(
            (
                [0],
                np.minimum(np.cumsum(np.maximum(values.astype(np.int64), 0)), capacity),
            )
        )
        clipped = np.diff(rows)
        tiles = np.concatenate(([0], np.cumsum((clipped + 31) // 32)))
        extras = np.concatenate(
            ([0], np.cumsum(np.maximum((clipped + 255) // 256 - 1, 0)))
        )
        for actual, expected in zip(outputs, (rows, tiles, extras), strict=True):
            np.testing.assert_array_equal(actual.numpy(), expected)

    @pytest.mark.parametrize("groups", [32, 64, 128, 129, 257, 513])
    def test_cooperative_adaptive_chunks(self, device, groups):
        """The final clipped tile prefix must drive live chunk sizes in every bucket."""
        import nvalchemiops.batch_utils as batch_utils

        capacity = 131072
        bucket = max(32, 1 << (groups - 1).bit_length())
        lengths = wp.empty(groups, dtype=wp.int32, device=device)
        outputs = [
            wp.empty(groups + 1, dtype=wp.int32, device=device) for _ in range(3)
        ]
        for values, expected_extras in (
            ([25000] * 4, [0, 31, 62, 93, 124]),
            ([0] * 4, [0] * 5),
            ([2147483647, -1, 1, 0], [0, 127, 127, 127, 127]),
        ):
            values = np.pad(np.asarray(values, np.int32), (0, groups - 4))
            lengths.assign(values)
            wp.launch_tiled(
                batch_utils._segment_offsets_cooperative[bucket],
                dim=(1,),
                inputs=[
                    lengths,
                    capacity,
                    32,
                    False,
                    outputs[0],
                    outputs[1],
                    1024,
                    outputs[2],
                    512,
                ],
                block_dim=min(bucket, 128),
                device=device,
            )
            rows = np.concatenate(
                (
                    [0],
                    np.minimum(
                        np.cumsum(np.maximum(values.astype(np.int64), 0)), capacity
                    ),
                )
            )
            tiles = np.concatenate(([0], np.cumsum((np.diff(rows) + 31) // 32)))
            extras = np.pad(expected_extras, (0, groups - 4), mode="edge")
            for actual, expected in zip(outputs, (rows, tiles, extras), strict=True):
                np.testing.assert_array_equal(actual.numpy(), expected)

    @pytest.mark.parametrize("groups", [4, 32, 64, 128, 129, 1024, 1025])
    @pytest.mark.parametrize("chunk_rows", [256, 512])
    @pytest.mark.parametrize(
        "values,expected_rows,expected_tiles,extras256,extras512,extras512_small",
        [
            (
                [0, 513, 33, 0],
                [0, 0, 513, 546, 546],
                [0, 0, 17, 19, 19],
                [0, 0, 2, 2, 2],
                [0, 0, 1, 1, 1],
                [0, 0, 2, 2, 2],
            ),
            (
                [-1, 513, 64, 2147483647],
                [0, 0, 513, 577, 2048],
                [0, 0, 17, 19, 65],
                [0, 0, 2, 2, 7],
                [0, 0, 1, 1, 3],
                [0, 0, 1, 1, 3],
            ),
            (
                [-2147483648, 513, 2147483647, 2147483647],
                [0, 0, 513, 2048, 2048],
                [0, 0, 17, 65, 65],
                [0, 0, 2, 7, 7],
                [0, 0, 1, 3, 3],
                [0, 0, 1, 3, 3],
            ),
            (
                [0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ),
        ],
        ids=["padding", "capacity_clamp", "int32_overflow", "empty"],
    )
    def test_fused_backward_schedules(
        self,
        device,
        groups,
        chunk_rows,
        values,
        expected_rows,
        expected_tiles,
        extras256,
        extras512,
        extras512_small,
    ):
        """One bounded row scan must feed both independently specified work prefixes."""
        import nvalchemiops.segment_mm as segmm

        values = np.pad(np.asarray(values, np.int32), (0, groups - 4))
        lengths = wp.array(np.repeat(values, 2), device=device)[::2]
        extras = wp.array(np.full(groups + 1, -99, np.int32), device=device)
        rows, tiles = segmm._segment_offsets(
            lengths, 2048, chunk_rows=chunk_rows, chunk_offsets=extras
        )
        expected_extras = extras256 if chunk_rows == 256 else extras512
        if groups == 4 and chunk_rows == 512:
            expected_extras = extras512_small
        for actual, expected in (
            (rows, expected_rows),
            (tiles, expected_tiles),
            (extras, expected_extras),
        ):
            expected = np.pad(
                np.asarray(expected, np.int32), (0, groups - 4), mode="edge"
            )
            np.testing.assert_array_equal(actual.numpy(), expected)

    @pytest.mark.parametrize("tile_rows", [7, 32])
    @pytest.mark.parametrize("groups", [0, 4, 128, 129, 1024])
    @pytest.mark.parametrize("skip_first_tile", [False, True])
    def test_clipped_wide_prefixes(self, device, tile_rows, groups, skip_first_tile):
        """Both paths clamp before narrowing and count tiles from clipped lengths."""
        from nvalchemiops.batch_utils import (
            _segment_counts,
            _segment_offsets_clamped,
            _segment_offsets_small,
        )

        values = np.zeros(groups, dtype=np.int32)
        if groups:
            values[:4] = [-1, 8, 2**31 - 1, 2**31 - 1]
        # The first real segment owns eight rows; the next is clipped to nine.
        expected_rows = np.zeros(groups + 1, dtype=np.int32)
        expected_tiles = np.zeros(groups + 1, dtype=np.int32)
        if groups:
            expected_rows[2] = 8
            expected_rows[3:] = 17
            expected_tiles[2] = 2 if tile_rows == 7 else 1
            expected_tiles[3:] = 4 if tile_rows == 7 else 2
            if skip_first_tile:
                expected_tiles[2] -= 1
                expected_tiles[3:] -= 2
        storage = wp.array(np.repeat(values, 2), device=device)
        lengths = storage[::2] if groups else storage
        rows = wp.empty(groups + 1, dtype=wp.int32, device=device)
        tiles = wp.empty(groups + 1, dtype=wp.int32, device=device)
        prefix = wp.empty(groups + 1, dtype=wp.int64, device=device)
        wp.launch(_segment_counts, groups + 1, inputs=[lengths, prefix], device=device)
        wp.utils.array_scan(prefix, prefix)
        wp.launch(
            _segment_offsets_clamped,
            groups + 1,
            inputs=[prefix, 17, tile_rows, skip_first_tile, rows, tiles, 0, tiles, 0],
            device=device,
        )
        wp.utils.array_scan(tiles, tiles)
        np.testing.assert_array_equal(rows.numpy(), expected_rows)
        np.testing.assert_array_equal(tiles.numpy(), expected_tiles)
        wp.launch(
            _segment_offsets_small,
            groups + 1,
            inputs=[lengths, 17, tile_rows, skip_first_tile, rows, tiles, 0, tiles, 0],
            device=device,
        )
        np.testing.assert_array_equal(rows.numpy(), expected_rows)
        np.testing.assert_array_equal(tiles.numpy(), expected_tiles)

    def test_zero_padding_preserves_active_rows(self, device):
        """The reusable padding kernel must not erase previously computed rows."""
        from nvalchemiops.batch_utils import _zero_padding_overloads

        rows = wp.array([0, 0, 2, 4, 4], dtype=wp.int32, device=device)
        out = wp.array(np.full((7, 3), 7, dtype=np.float32), device=device)
        wp.launch(
            _zero_padding_overloads[wp.float32],
            out.shape,
            inputs=[rows, out],
            device=device,
        )
        np.testing.assert_array_equal(out.numpy()[:4], 7)
        np.testing.assert_array_equal(out.numpy()[4:], 0)

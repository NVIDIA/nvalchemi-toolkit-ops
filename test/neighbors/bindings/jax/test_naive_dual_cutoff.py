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

"""Tests for JAX bindings of naive dual cutoff neighbor list methods."""

from __future__ import annotations

from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.neighbors import (
    naive_neighbor_list,
    naive_neighbor_list_dual_cutoff,
)

from .conftest import create_simple_cubic_system_jax, requires_gpu

pytestmark = requires_gpu

dual_module = import_module("nvalchemiops.jax.neighbors.naive_dual_cutoff")


def _active_neighbor_shift_rows(
    neighbor_matrix: jax.Array,
    shifts: jax.Array,
    counts: jax.Array,
    atom_index: int,
) -> list[tuple[int, int, int, int]]:
    """Return sorted active ``(neighbor, sx, sy, sz)`` rows for one atom."""
    count = int(np.asarray(counts[atom_index]))
    rows = np.concatenate(
        (
            np.asarray(neighbor_matrix[atom_index, :count])[:, None],
            np.asarray(shifts[atom_index, :count]),
        ),
        axis=1,
    )
    return sorted(tuple(row) for row in rows.tolist())


class TestNaiveDualCutoffCorrectness:
    """Test correctness of naive dual cutoff neighbor list."""

    def test_matrix_format_no_pbc(self):
        """Test dual cutoff neighbor list in matrix format without PBC."""
        positions, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=jnp.float32
        )

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1,
                cutoff2,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
            )
        )

        # Verify output shapes and types
        assert neighbor_matrix1.shape == (8, max_neighbors1)
        assert neighbor_matrix2.shape == (8, max_neighbors2)
        assert num_neighbors1.shape == (8,)
        assert num_neighbors2.shape == (8,)
        assert neighbor_matrix1.dtype == jnp.int32
        assert neighbor_matrix2.dtype == jnp.int32
        assert num_neighbors1.dtype == jnp.int32
        assert num_neighbors2.dtype == jnp.int32

        # Verify neighbor counts are reasonable
        assert jnp.all(num_neighbors1 >= 0)
        assert jnp.all(num_neighbors2 >= 0)
        assert jnp.all(num_neighbors1 <= max_neighbors1)
        assert jnp.all(num_neighbors2 <= max_neighbors2)
        # Larger cutoff should find at least as many neighbors
        assert jnp.all(num_neighbors2 >= num_neighbors1)

    def test_matrix_format_with_pbc(self):
        """Test dual cutoff neighbor list in matrix format with PBC."""
        positions, cell, pbc = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=jnp.float32
        )

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        (
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix_shifts1,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        ) = naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            pbc=pbc,
            cell=cell,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
        )

        # Verify output shapes and types
        assert neighbor_matrix1.shape == (8, max_neighbors1)
        assert neighbor_matrix2.shape == (8, max_neighbors2)
        assert neighbor_matrix_shifts1.shape == (8, max_neighbors1, 3)
        assert neighbor_matrix_shifts2.shape == (8, max_neighbors2, 3)
        assert num_neighbors1.shape == (8,)
        assert num_neighbors2.shape == (8,)

        # Verify dtypes
        assert neighbor_matrix1.dtype == jnp.int32
        assert neighbor_matrix2.dtype == jnp.int32
        assert num_neighbors1.dtype == jnp.int32
        assert num_neighbors2.dtype == jnp.int32
        assert neighbor_matrix_shifts1.dtype == jnp.int32
        assert neighbor_matrix_shifts2.dtype == jnp.int32

        # Verify neighbor counts
        assert jnp.all(num_neighbors1 >= 0)
        assert jnp.all(num_neighbors2 >= 0)
        assert jnp.all(num_neighbors2 >= num_neighbors1)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize(
        "wrap_kwargs",
        [
            pytest.param({}, id="default_wrapped"),
            pytest.param({"wrap_positions": False}, id="prewrapped"),
        ],
    )
    def test_pbc_wrap_modes_match_single_cutoff(
        self,
        dtype,
        wrap_kwargs,
    ):
        """Both PBC wrap modes match independent single-cutoff results."""
        positions, cell, pbc = create_simple_cubic_system_jax(
            num_atoms=8,
            cell_size=2.0,
            dtype=dtype,
        )
        cutoff1 = 1.1
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        (
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix_shifts1,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        ) = naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            cell=cell,
            pbc=pbc,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            **wrap_kwargs,
        )
        reference1 = naive_neighbor_list(
            positions,
            cutoff=cutoff1,
            cell=cell,
            pbc=pbc,
            max_neighbors=max_neighbors1,
            **wrap_kwargs,
        )
        reference2 = naive_neighbor_list(
            positions,
            cutoff=cutoff2,
            cell=cell,
            pbc=pbc,
            max_neighbors=max_neighbors2,
            **wrap_kwargs,
        )

        assert jnp.any(num_neighbors1 > 0)
        assert jnp.any(num_neighbors2 > 0)
        assert jnp.array_equal(num_neighbors1, reference1[1])
        assert jnp.array_equal(num_neighbors2, reference2[1])
        for atom_index in range(positions.shape[0]):
            assert _active_neighbor_shift_rows(
                neighbor_matrix1,
                neighbor_matrix_shifts1,
                num_neighbors1,
                atom_index,
            ) == _active_neighbor_shift_rows(
                reference1[0],
                reference1[2],
                reference1[1],
                atom_index,
            )
            assert _active_neighbor_shift_rows(
                neighbor_matrix2,
                neighbor_matrix_shifts2,
                num_neighbors2,
                atom_index,
            ) == _active_neighbor_shift_rows(
                reference2[0],
                reference2[2],
                reference2[1],
                atom_index,
            )


class TestNaiveDualCutoffEdgeCases:
    """Test edge cases for naive dual cutoff neighbor list."""

    def test_single_atom(self):
        """Test with single atom (should have no neighbors)."""
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)

        cutoff1 = 1.0
        cutoff2 = 1.5

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1,
                cutoff2,
                max_neighbors1=10,
                max_neighbors2=10,
            )
        )

        assert int(num_neighbors1[0]) == 0
        assert int(num_neighbors2[0]) == 0

    def test_identical_cutoffs(self):
        """Test with identical cutoffs (both lists should match)."""
        positions, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=jnp.float32
        )

        cutoff = 1.2
        max_neighbors = 20

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions,
                cutoff,
                cutoff,
                max_neighbors1=max_neighbors,
                max_neighbors2=max_neighbors,
            )
        )

        # Neighbor counts should be identical
        assert jnp.all(num_neighbors1 == num_neighbors2)

    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    @pytest.mark.parametrize("with_pbc", [True, False])
    def test_zero_cutoffs_fast_path(self, return_neighbor_list, with_pbc):
        """``cutoff1 <= 0 and cutoff2 <= 0`` returns 0-pair tuples — 4 shape
        variants × {return_list, !return_list} × {pbc, !pbc}."""
        positions, cell, pbc = create_simple_cubic_system_jax(
            num_atoms=4, cell_size=2.0, dtype=jnp.float32
        )
        kwargs = dict(max_neighbors1=10, max_neighbors2=15)
        if with_pbc:
            kwargs["cell"] = cell
            kwargs["pbc"] = pbc.squeeze(0)
        result = naive_neighbor_list_dual_cutoff(
            positions,
            0.0,
            0.0,
            return_neighbor_list=return_neighbor_list,
            **kwargs,
        )
        N = positions.shape[0]
        if return_neighbor_list:
            if with_pbc:
                # (nlist1, nptr1, shifts1, nlist2, nptr2, shifts2)
                assert len(result) == 6
                nl1, np1, sh1, nl2, np2, sh2 = result
                assert nl1.shape == (2, 0)
                assert np1.shape == (N + 1,)
                assert sh1.shape == (0, 3)
                assert nl2.shape == (2, 0)
                assert np2.shape == (N + 1,)
                assert sh2.shape == (0, 3)
            else:
                # (nlist1, nptr1, nlist2, nptr2)
                assert len(result) == 4
                nl1, np1, nl2, np2 = result
                assert nl1.shape == (2, 0)
                assert np1.shape == (N + 1,)
                assert nl2.shape == (2, 0)
                assert np2.shape == (N + 1,)
        else:
            if with_pbc:
                # (nm1, nn1, shifts1, nm2, nn2, shifts2)
                assert len(result) == 6
                nm1, nn1, sh1, nm2, nn2, sh2 = result
                assert int(nn1.sum()) == 0 and int(nn2.sum()) == 0
            else:
                # (nm1, nn1, nm2, nn2)
                assert len(result) == 4
                nm1, nn1, nm2, nn2 = result
                assert int(nn1.sum()) == 0 and int(nn2.sum()) == 0


# ==============================================================================
# Tests: return_neighbor_list=True (COO format)
# ==============================================================================


class TestDualCutoffListFormat:
    """Test dual cutoff neighbor list in COO list format."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_unbatched_list_format_no_pbc(self, dtype):
        """Test unbatched dual cutoff in list format without PBC."""
        positions, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=dtype
        )
        cutoff1 = 1.0
        cutoff2 = 1.5

        neighbor_list1, neighbor_ptr1, neighbor_list2, neighbor_ptr2 = (
            naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1,
                cutoff2,
                max_neighbors1=15,
                max_neighbors2=25,
                return_neighbor_list=True,
            )
        )

        # Verify COO format shapes
        assert neighbor_list1.shape[0] == 2
        assert neighbor_list2.shape[0] == 2
        assert neighbor_ptr1.shape == (9,)
        assert neighbor_ptr2.shape == (9,)
        # Larger cutoff should find at least as many pairs
        assert neighbor_list2.shape[1] >= neighbor_list1.shape[1]

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_unbatched_list_format_with_pbc(self, dtype):
        """Test unbatched dual cutoff in list format with PBC."""
        positions, cell, pbc = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=dtype
        )
        cutoff1 = 1.0
        cutoff2 = 1.5

        (
            neighbor_list1,
            neighbor_ptr1,
            neighbor_shifts1,
            neighbor_list2,
            neighbor_ptr2,
            neighbor_shifts2,
        ) = naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            cell=cell,
            pbc=pbc,
            max_neighbors1=15,
            max_neighbors2=25,
            return_neighbor_list=True,
        )

        # Verify COO format shapes
        assert neighbor_list1.shape[0] == 2
        assert neighbor_list2.shape[0] == 2
        assert neighbor_ptr1.shape == (9,)
        assert neighbor_ptr2.shape == (9,)
        assert neighbor_shifts1.shape[0] == neighbor_list1.shape[1]
        assert neighbor_shifts2.shape[0] == neighbor_list2.shape[1]
        assert neighbor_shifts1.shape[1] == 3
        assert neighbor_shifts2.shape[1] == 3


class TestNaiveDualCutoffJIT:
    """Smoke tests for naive_neighbor_list_dual_cutoff with jax.jit."""

    def test_jit_no_pbc(self):
        """Test dual cutoff without PBC works with jax.jit."""
        positions, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=jnp.float32
        )

        @jax.jit
        def jitted_dual(positions):
            return naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1=1.0,
                cutoff2=1.5,
                max_neighbors1=15,
                max_neighbors2=25,
            )

        nm1, nn1, nm2, nn2 = jitted_dual(positions)

        assert nm1.shape == (8, 15)
        assert nm2.shape == (8, 25)
        assert nn1.shape == (8,)
        assert nn2.shape == (8,)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
class TestNaiveDualCutoffSelectiveRebuildFlags:
    """Test selective rebuild (rebuild_flags) for naive_neighbor_list_dual_cutoff JAX."""

    def test_wrapped_pbc_false_flag_preserves_both_cutoffs(self, dtype):
        """Wrapped PBC false flags preserve both matrices, counts, and shifts."""
        positions, cell, pbc = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=dtype
        )
        positions = positions.at[0, 0].add(2.0)
        kwargs = {
            "cell": cell,
            "pbc": pbc,
            "max_neighbors1": 15,
            "max_neighbors2": 25,
        }
        reference = naive_neighbor_list_dual_cutoff(positions, 1.1, 1.5, **kwargs)
        preserved = naive_neighbor_list_dual_cutoff(
            positions,
            1.1,
            1.5,
            neighbor_matrix1=reference[0],
            num_neighbors1=reference[1],
            neighbor_matrix_shifts1=reference[2],
            neighbor_matrix2=reference[3],
            num_neighbors2=reference[4],
            neighbor_matrix_shifts2=reference[5],
            rebuild_flags=jnp.zeros(1, dtype=jnp.bool_),
            **kwargs,
        )

        for got, expected in zip(preserved, reference):
            assert jnp.array_equal(got, expected)

    def test_wrapped_pbc_true_flag_rebuilds_both_cutoffs(self, dtype):
        """Wrapped PBC true flags rebuild both cutoff matrices and shifts."""
        positions, cell, pbc = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=dtype
        )
        positions = positions.at[0, 0].add(2.0)
        kwargs = {
            "cell": cell,
            "pbc": pbc,
            "max_neighbors1": 15,
            "max_neighbors2": 25,
        }
        reference = naive_neighbor_list_dual_cutoff(positions, 1.1, 1.5, **kwargs)
        rebuilt = naive_neighbor_list_dual_cutoff(
            positions,
            1.1,
            1.5,
            neighbor_matrix1=jnp.zeros_like(reference[0]),
            num_neighbors1=jnp.zeros_like(reference[1]),
            neighbor_matrix_shifts1=jnp.zeros_like(reference[2]),
            neighbor_matrix2=jnp.zeros_like(reference[3]),
            num_neighbors2=jnp.zeros_like(reference[4]),
            neighbor_matrix_shifts2=jnp.zeros_like(reference[5]),
            rebuild_flags=jnp.ones(1, dtype=jnp.bool_),
            **kwargs,
        )

        for matrix_index, count_index, shifts_index in ((0, 1, 2), (3, 4, 5)):
            assert jnp.array_equal(rebuilt[count_index], reference[count_index])
            for atom_index, count in enumerate(reference[count_index].tolist()):
                rebuilt_pairs = jnp.concatenate(
                    [
                        rebuilt[matrix_index][atom_index, :count, None],
                        rebuilt[shifts_index][atom_index, :count],
                    ],
                    axis=1,
                )
                reference_pairs = jnp.concatenate(
                    [
                        reference[matrix_index][atom_index, :count, None],
                        reference[shifts_index][atom_index, :count],
                    ],
                    axis=1,
                )
                assert {tuple(row) for row in rebuilt_pairs.tolist()} == {
                    tuple(row) for row in reference_pairs.tolist()
                }

    def test_no_rebuild_preserves_data(self, dtype):
        """Flag=False: neighbor data should remain unchanged."""
        positions, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=dtype
        )
        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        # Initial full build
        nm1, nn1, nm2, nn2 = naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
        )

        saved_nn1 = jnp.array(nn1)
        saved_nn2 = jnp.array(nn2)

        # Selective rebuild with flag=False
        rebuild_flags = jnp.zeros(1, dtype=jnp.bool_)
        nm1b, nn1b, nm2b, nn2b = naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            neighbor_matrix1=nm1,
            neighbor_matrix2=nm2,
            num_neighbors1=nn1,
            num_neighbors2=nn2,
            rebuild_flags=rebuild_flags,
        )

        assert jnp.all(nn1b == saved_nn1), "nn1 must be unchanged when flag=False"
        assert jnp.all(nn2b == saved_nn2), "nn2 must be unchanged when flag=False"

    def test_rebuild_updates_data(self, dtype):
        """Flag=True: result should match a fresh full rebuild."""
        positions, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=dtype
        )
        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        # Reference: full build
        _, nn1_ref, _, nn2_ref = naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
        )

        # Selective rebuild with flag=True
        nm1_stale = jnp.full((8, max_neighbors1), 99, dtype=jnp.int32)
        nm2_stale = jnp.full((8, max_neighbors2), 99, dtype=jnp.int32)
        nn1_stale = jnp.full((8,), 99, dtype=jnp.int32)
        nn2_stale = jnp.full((8,), 99, dtype=jnp.int32)

        rebuild_flags = jnp.ones(1, dtype=jnp.bool_)
        _, nn1b, _, nn2b = naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            neighbor_matrix1=nm1_stale,
            neighbor_matrix2=nm2_stale,
            num_neighbors1=nn1_stale,
            num_neighbors2=nn2_stale,
            rebuild_flags=rebuild_flags,
        )

        assert jnp.all(nn1b == nn1_ref), "nn1 should match full rebuild when flag=True"
        assert jnp.all(nn2b == nn2_ref), "nn2 should match full rebuild when flag=True"


class TestRegistrationLaziness:
    """Regression tests for lazy direct dual-cutoff naive registry construction."""

    @staticmethod
    def _direct_registrations():
        """Return every direct dual-cutoff lazy registration."""
        return tuple(dual_module._DIRECT_NAIVE_DUAL_KERNELS.values())

    @pytest.fixture(autouse=True)
    def _restore_direct_caches(self):
        """Restore process-global direct caches after each laziness test."""
        snapshots = [
            (registration, dict(registration._cache))
            for registration in self._direct_registrations()
        ]
        try:
            yield
        finally:
            for registration, cache in snapshots:
                registration._cache.clear()
                registration._cache.update(cache)

    def _clear_direct_caches(self) -> None:
        """Clear lazy direct dual-cutoff wrapper caches before each check."""
        for registration in self._direct_registrations():
            registration._cache.clear()

    def test_direct_no_pbc_caches_one_wrapper(self) -> None:
        """Direct scalar no-PBC should register exactly one dtype wrapper."""
        self._clear_direct_caches()
        positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32)
        naive_neighbor_list_dual_cutoff(
            positions,
            0.75,
            1.0,
            max_neighbors1=4,
            max_neighbors2=4,
        )

        assert len(dual_module._DIRECT_NAIVE_DUAL_KERNELS[("none", False)]._cache) == 1
        for key, registration in dual_module._DIRECT_NAIVE_DUAL_KERNELS.items():
            if key != ("none", False):
                assert len(registration._cache) == 0

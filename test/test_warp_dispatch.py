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
Unit tests for generic warp_dispatch primitives.

Tests cover:
- register_overloads: default/custom dtype pairs, custom key_fn
- build_dispatch_table: composite keys, dtype iteration, missing-key error
- dispatch: successful launch, KeyError on missing key
- validate_out_array: shape/dtype/device checks
"""

from enum import Enum
from typing import Any

import numpy as np
import pytest
import warp as wp

from nvalchemiops.warp_dispatch import (
    DEFAULT_DTYPE_PAIRS,
    build_dispatch_table,
    dispatch,
    register_overloads,
    validate_out_array,
)

DEVICE = "cuda:0"


# ==============================================================================
# Tests for DEFAULT_DTYPE_PAIRS
# ==============================================================================


class TestDefaultDtypePairs:
    """Tests for the DEFAULT_DTYPE_PAIRS constant."""

    def test_contains_float32_and_float64(self):
        assert (wp.vec3f, wp.float32) in DEFAULT_DTYPE_PAIRS
        assert (wp.vec3d, wp.float64) in DEFAULT_DTYPE_PAIRS

    def test_length(self):
        assert len(DEFAULT_DTYPE_PAIRS) == 2


# ==============================================================================
# Tests for register_overloads
# ==============================================================================


@wp.kernel
def _generic_test_kernel(
    a: wp.array(dtype=Any),
    b: wp.array(dtype=Any),
):
    """Generic kernel for overload registration tests."""
    _i = wp.tid()
    pass


class TestRegisterOverloads:
    """Tests for register_overloads."""

    def test_default_dtype_pairs(self):
        overloads = register_overloads(
            _generic_test_kernel,
            lambda v, t: [wp.array(dtype=v), wp.array(dtype=t)],
        )
        assert wp.vec3f in overloads
        assert wp.vec3d in overloads
        assert len(overloads) == 2

    def test_custom_dtype_pairs(self):
        overloads = register_overloads(
            _generic_test_kernel,
            lambda v, t: [wp.array(dtype=v), wp.array(dtype=t)],
            dtype_pairs=((wp.vec3f, wp.float32),),
        )
        assert wp.vec3f in overloads
        assert wp.vec3d not in overloads
        assert len(overloads) == 1

    def test_custom_key_fn(self):
        overloads = register_overloads(
            _generic_test_kernel,
            lambda v, t: [wp.array(dtype=v), wp.array(dtype=t)],
            key_fn=lambda v, t: (v, t),
        )
        assert (wp.vec3f, wp.float32) in overloads
        assert (wp.vec3d, wp.float64) in overloads
        assert len(overloads) == 2

    def test_overload_values_are_not_none(self):
        overloads = register_overloads(
            _generic_test_kernel,
            lambda v, t: [wp.array(dtype=v), wp.array(dtype=t)],
        )
        for key, value in overloads.items():
            assert value is not None, f"Overload for {key} is None"


# ==============================================================================
# Tests for build_dispatch_table
# ==============================================================================


class _TestAxis(Enum):
    """Test dispatch axis."""

    MODE_A = "mode_a"
    MODE_B = "mode_b"


@wp.kernel
def _table_kernel_a(data: wp.array(dtype=Any)):
    i = wp.tid()
    data[i] = data[i]


@wp.kernel
def _table_kernel_b(
    data: wp.array(dtype=Any),
    extra: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    data[i] = data[i]


class TestBuildDispatchTable:
    """Tests for build_dispatch_table."""

    def test_default_dtype_pairs(self):
        table = build_dispatch_table(
            {
                _TestAxis.MODE_A: (
                    _table_kernel_a,
                    lambda v, t: [wp.array(dtype=v)],
                ),
            }
        )
        assert (_TestAxis.MODE_A, wp.vec3f) in table
        assert (_TestAxis.MODE_A, wp.vec3d) in table
        assert len(table) == 2

    def test_multiple_axes(self):
        table = build_dispatch_table(
            {
                _TestAxis.MODE_A: (
                    _table_kernel_a,
                    lambda v, t: [wp.array(dtype=v)],
                ),
                _TestAxis.MODE_B: (
                    _table_kernel_b,
                    lambda v, t: [wp.array(dtype=v), wp.array(dtype=wp.int32)],
                ),
            }
        )
        assert (_TestAxis.MODE_A, wp.vec3f) in table
        assert (_TestAxis.MODE_A, wp.vec3d) in table
        assert (_TestAxis.MODE_B, wp.vec3f) in table
        assert (_TestAxis.MODE_B, wp.vec3d) in table
        assert len(table) == 4

    def test_custom_dtype_pairs(self):
        table = build_dispatch_table(
            {
                _TestAxis.MODE_A: (
                    _table_kernel_a,
                    lambda v, t: [wp.array(dtype=v)],
                ),
            },
            dtype_pairs=((wp.vec3f, wp.float32),),
        )
        assert (_TestAxis.MODE_A, wp.vec3f) in table
        assert (_TestAxis.MODE_A, wp.vec3d) not in table
        assert len(table) == 1

    def test_string_axis_keys(self):
        """Axis keys can be any hashable, not just enums."""
        table = build_dispatch_table(
            {
                "naive": (
                    _table_kernel_a,
                    lambda v, t: [wp.array(dtype=v)],
                ),
                "cell_list": (
                    _table_kernel_a,
                    lambda v, t: [wp.array(dtype=v)],
                ),
            }
        )
        assert ("naive", wp.vec3f) in table
        assert ("cell_list", wp.vec3d) in table

    def test_tuple_axis_keys(self):
        """Axis keys can be tuples of enum values for multi-axis dispatch."""
        table = build_dispatch_table(
            {
                (_TestAxis.MODE_A, _TestAxis.MODE_B): (
                    _table_kernel_a,
                    lambda v, t: [wp.array(dtype=v)],
                ),
            }
        )
        assert ((_TestAxis.MODE_A, _TestAxis.MODE_B), wp.vec3f) in table

    def test_values_are_not_none(self):
        table = build_dispatch_table(
            {
                _TestAxis.MODE_A: (
                    _table_kernel_a,
                    lambda v, t: [wp.array(dtype=v)],
                ),
            }
        )
        for key, value in table.items():
            assert value is not None, f"Overload for {key} is None"


# ==============================================================================
# Tests for dispatch
# ==============================================================================


@wp.kernel
def _dispatch_add_one_kernel(data: wp.array(dtype=wp.vec3f)):
    i = wp.tid()
    data[i] = data[i] + wp.vec3f(1.0, 1.0, 1.0)


@wp.kernel
def _dispatch_add_two_kernel(data: wp.array(dtype=wp.vec3f)):
    i = wp.tid()
    data[i] = data[i] + wp.vec3f(2.0, 2.0, 2.0)


# Build a simple table with non-generic (concrete) kernels for dispatch tests.
_dispatch_test_table = {
    ("add_one", wp.vec3f): _dispatch_add_one_kernel,
    ("add_two", wp.vec3f): _dispatch_add_two_kernel,
}


class TestDispatch:
    """Tests for the dispatch function."""

    def test_successful_launch(self):
        data = wp.zeros(4, dtype=wp.vec3f, device=DEVICE)
        dispatch(
            _dispatch_test_table,
            ("add_one", wp.vec3f),
            dim=4,
            inputs=[data],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)
        result = data.numpy()
        np.testing.assert_array_equal(result, np.full((4, 3), 1.0))

    def test_dispatch_selects_correct_kernel(self):
        data = wp.zeros(4, dtype=wp.vec3f, device=DEVICE)
        dispatch(
            _dispatch_test_table,
            ("add_two", wp.vec3f),
            dim=4,
            inputs=[data],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)
        result = data.numpy()
        np.testing.assert_array_equal(result, np.full((4, 3), 2.0))

    def test_missing_key_raises_key_error(self):
        data = wp.zeros(4, dtype=wp.vec3f, device=DEVICE)
        with pytest.raises(KeyError, match="No kernel registered for dispatch key"):
            dispatch(
                _dispatch_test_table,
                ("nonexistent", wp.vec3f),
                dim=4,
                inputs=[data],
                device=DEVICE,
            )

    def test_missing_key_error_lists_available_keys(self):
        data = wp.zeros(4, dtype=wp.vec3f, device=DEVICE)
        with pytest.raises(KeyError, match="Available keys"):
            dispatch(
                _dispatch_test_table,
                ("nonexistent", wp.vec3f),
                dim=4,
                inputs=[data],
                device=DEVICE,
            )

    def test_dispatch_with_outputs(self):
        """dispatch passes outputs kwarg to wp.launch."""
        inp = wp.zeros(4, dtype=wp.vec3f, device=DEVICE)
        dispatch(
            _dispatch_test_table,
            ("add_one", wp.vec3f),
            dim=4,
            inputs=[inp],
            outputs=[],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)
        result = inp.numpy()
        np.testing.assert_array_equal(result, np.full((4, 3), 1.0))


# ==============================================================================
# Tests for validate_out_array
# ==============================================================================


class TestValidateOutArray:
    """Tests for validate_out_array."""

    def test_valid(self):
        ref = wp.zeros(10, dtype=wp.vec3f, device=DEVICE)
        out = wp.zeros(10, dtype=wp.vec3f, device=DEVICE)
        validate_out_array(out, ref, "test_out")

    def test_shape_mismatch(self):
        ref = wp.zeros(10, dtype=wp.vec3f, device=DEVICE)
        out = wp.zeros(5, dtype=wp.vec3f, device=DEVICE)
        with pytest.raises(ValueError, match="test_out shape mismatch"):
            validate_out_array(out, ref, "test_out")

    def test_dtype_mismatch(self):
        ref = wp.zeros(10, dtype=wp.vec3f, device=DEVICE)
        out = wp.zeros(10, dtype=wp.vec3d, device=DEVICE)
        with pytest.raises(ValueError, match="test_out dtype mismatch"):
            validate_out_array(out, ref, "test_out")

    def test_device_mismatch(self):
        ref = wp.zeros(10, dtype=wp.vec3f, device=DEVICE)
        out = wp.zeros(10, dtype=wp.vec3f, device="cpu")
        with pytest.raises(ValueError, match="test_out device mismatch"):
            validate_out_array(out, ref, "test_out")

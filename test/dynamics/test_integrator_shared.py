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
Unit tests for dynamics-specific launch helpers.

Tests cover:
- ExecutionMode and KernelFamily types
- resolve_execution_mode: all 3 modes + mutual exclusivity error
- launch_family: correct kernel selection per mode, unsupported mode error
- dispatch_family: end-to-end dispatch
- build_family_dict: family construction

Generic primitives (register_overloads, validate_out_array, build_dispatch_table,
dispatch) are tested in ``test/test_warp_dispatch.py``.
"""

from typing import Any

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.utils.launch_helpers import (
    ExecutionMode,
    KernelFamily,
    build_family_dict,
    dispatch_family,
    launch_family,
    resolve_execution_mode,
)

DEVICE = "cuda:0"


# ==============================================================================
# Tests for ExecutionMode
# ==============================================================================


class TestExecutionMode:
    """Tests for ExecutionMode enum."""

    def test_values(self):
        assert ExecutionMode.SINGLE.value == "single"
        assert ExecutionMode.BATCH_IDX.value == "batch_idx"
        assert ExecutionMode.ATOM_PTR.value == "atom_ptr"

    def test_members(self):
        assert len(ExecutionMode) == 3


# ==============================================================================
# Tests for KernelFamily
# ==============================================================================


class TestKernelFamily:
    """Tests for KernelFamily dataclass."""

    def test_all_fields(self):
        family = KernelFamily(single="s", batch_idx="b", atom_ptr="p")
        assert family.single == "s"
        assert family.batch_idx == "b"
        assert family.atom_ptr == "p"

    def test_defaults(self):
        family = KernelFamily(single="s")
        assert family.single == "s"
        assert family.batch_idx is None
        assert family.atom_ptr is None

    def test_frozen(self):
        family = KernelFamily(single="s")
        with pytest.raises(AttributeError):
            family.single = "other"


# ==============================================================================
# Tests for resolve_execution_mode
# ==============================================================================


class TestResolveExecutionMode:
    """Tests for resolve_execution_mode."""

    def test_single_mode(self):
        assert resolve_execution_mode(None, None) is ExecutionMode.SINGLE

    def test_batch_idx_mode(self):
        batch_idx = wp.array([0, 0, 1, 1], dtype=wp.int32, device=DEVICE)
        assert resolve_execution_mode(batch_idx, None) is ExecutionMode.BATCH_IDX

    def test_atom_ptr_mode(self):
        atom_ptr = wp.array([0, 2, 4], dtype=wp.int32, device=DEVICE)
        assert resolve_execution_mode(None, atom_ptr) is ExecutionMode.ATOM_PTR

    def test_mutual_exclusivity(self):
        batch_idx = wp.array([0, 0, 1, 1], dtype=wp.int32, device=DEVICE)
        atom_ptr = wp.array([0, 2, 4], dtype=wp.int32, device=DEVICE)
        with pytest.raises(ValueError, match="Provide batch_idx OR atom_ptr, not both"):
            resolve_execution_mode(batch_idx, atom_ptr)


# ==============================================================================
# Tests for launch_family
# ==============================================================================


# Simple kernels for testing dispatch
@wp.kernel
def _test_single_kernel(output: wp.array(dtype=wp.float32)):
    i = wp.tid()
    output[i] = 1.0


@wp.kernel
def _test_batch_kernel(output: wp.array(dtype=wp.float32)):
    i = wp.tid()
    output[i] = 2.0


@wp.kernel
def _test_ptr_kernel(output: wp.array(dtype=wp.float32)):
    i = wp.tid()
    output[i] = 3.0


class TestLaunchFamily:
    """Tests for launch_family."""

    def test_single_mode_launches_single_kernel(self):
        output = wp.zeros(4, dtype=wp.float32, device=DEVICE)
        family = KernelFamily(
            single=_test_single_kernel,
            batch_idx=_test_batch_kernel,
            atom_ptr=_test_ptr_kernel,
        )
        launch_family(
            family,
            mode=ExecutionMode.SINGLE,
            dim=4,
            inputs_single=[output],
            inputs_batch=[output],
            inputs_ptr=[output],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)
        np.testing.assert_array_equal(output.numpy(), [1.0, 1.0, 1.0, 1.0])

    def test_batch_idx_mode_launches_batch_kernel(self):
        output = wp.zeros(4, dtype=wp.float32, device=DEVICE)
        family = KernelFamily(
            single=_test_single_kernel,
            batch_idx=_test_batch_kernel,
            atom_ptr=_test_ptr_kernel,
        )
        launch_family(
            family,
            mode=ExecutionMode.BATCH_IDX,
            dim=4,
            inputs_single=[output],
            inputs_batch=[output],
            inputs_ptr=[output],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)
        np.testing.assert_array_equal(output.numpy(), [2.0, 2.0, 2.0, 2.0])

    def test_atom_ptr_mode_launches_ptr_kernel(self):
        output = wp.zeros(4, dtype=wp.float32, device=DEVICE)
        family = KernelFamily(
            single=_test_single_kernel,
            batch_idx=_test_batch_kernel,
            atom_ptr=_test_ptr_kernel,
        )
        launch_family(
            family,
            mode=ExecutionMode.ATOM_PTR,
            dim=4,
            inputs_single=[output],
            inputs_batch=[output],
            inputs_ptr=[output],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)
        np.testing.assert_array_equal(output.numpy(), [3.0, 3.0, 3.0, 3.0])

    def test_unsupported_atom_ptr_raises(self):
        family = KernelFamily(single=_test_single_kernel, atom_ptr=None)
        with pytest.raises(ValueError, match="atom_ptr mode not supported"):
            launch_family(
                family,
                mode=ExecutionMode.ATOM_PTR,
                dim=4,
                inputs_single=[],
                inputs_ptr=[],
                device=DEVICE,
            )

    def test_unsupported_batch_idx_raises(self):
        family = KernelFamily(single=_test_single_kernel, batch_idx=None)
        with pytest.raises(ValueError, match="batch_idx mode not supported"):
            launch_family(
                family,
                mode=ExecutionMode.BATCH_IDX,
                dim=4,
                inputs_single=[],
                inputs_batch=[],
                device=DEVICE,
            )


# ==============================================================================
# Tests for dispatch_family
# ==============================================================================


@wp.kernel
def _dispatch_write_kernel(
    data: wp.array(dtype=wp.vec3f),
    out: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    out[i] = data[i] + wp.vec3f(1.0, 1.0, 1.0)


@wp.kernel
def _dispatch_write_batch_kernel(
    data: wp.array(dtype=wp.vec3f),
    batch_idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    out[i] = data[i] + wp.vec3f(2.0, 2.0, 2.0)


@wp.kernel
def _dispatch_write_ptr_kernel(
    data: wp.array(dtype=wp.vec3f),
    atom_ptr: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec3f),
):
    sys_id = wp.tid()
    start = atom_ptr[sys_id]
    end = atom_ptr[sys_id + 1]
    for i in range(start, end):
        out[i] = data[i] + wp.vec3f(3.0, 3.0, 3.0)


_dispatch_test_families = {
    wp.vec3f: KernelFamily(
        single=_dispatch_write_kernel,
        batch_idx=_dispatch_write_batch_kernel,
        atom_ptr=_dispatch_write_ptr_kernel,
    ),
}


class TestDispatchFamily:
    """End-to-end tests for dispatch_family."""

    def test_single_mode(self):
        data = wp.zeros(4, dtype=wp.vec3f, device=DEVICE)
        out = wp.zeros(4, dtype=wp.vec3f, device=DEVICE)
        dispatch_family(
            _dispatch_test_families,
            data,
            inputs_single=[data, out],
            inputs_batch=[data, None, out],
            inputs_ptr=[data, None, out],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)
        result = out.numpy()
        np.testing.assert_array_equal(result, np.full((4, 3), 1.0))

    def test_batch_idx_mode(self):
        data = wp.zeros(4, dtype=wp.vec3f, device=DEVICE)
        out = wp.zeros(4, dtype=wp.vec3f, device=DEVICE)
        batch_idx = wp.array([0, 0, 1, 1], dtype=wp.int32, device=DEVICE)
        dispatch_family(
            _dispatch_test_families,
            data,
            batch_idx=batch_idx,
            inputs_single=[data, out],
            inputs_batch=[data, batch_idx, out],
            inputs_ptr=[data, None, out],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)
        result = out.numpy()
        np.testing.assert_array_equal(result, np.full((4, 3), 2.0))

    def test_atom_ptr_mode(self):
        data = wp.zeros(4, dtype=wp.vec3f, device=DEVICE)
        out = wp.zeros(4, dtype=wp.vec3f, device=DEVICE)
        atom_ptr = wp.array([0, 2, 4], dtype=wp.int32, device=DEVICE)
        dispatch_family(
            _dispatch_test_families,
            data,
            atom_ptr=atom_ptr,
            inputs_single=[data, out],
            inputs_batch=[data, None, out],
            inputs_ptr=[data, atom_ptr, out],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)
        result = out.numpy()
        np.testing.assert_array_equal(result, np.full((4, 3), 3.0))

    def test_device_inferred_from_primary_array(self):
        """When device=None, dispatch_family infers from primary_array."""
        data = wp.zeros(4, dtype=wp.vec3f, device=DEVICE)
        out = wp.zeros(4, dtype=wp.vec3f, device=DEVICE)
        dispatch_family(
            _dispatch_test_families,
            data,
            device=None,
            inputs_single=[data, out],
        )
        wp.synchronize_device(DEVICE)
        result = out.numpy()
        np.testing.assert_array_equal(result, np.full((4, 3), 1.0))

    def test_unknown_dtype_raises_key_error(self):
        """dispatch_family raises KeyError when primary dtype not in family_dict."""
        data = wp.zeros(4, dtype=wp.vec3d, device=DEVICE)
        out = wp.zeros(4, dtype=wp.vec3d, device=DEVICE)
        with pytest.raises(KeyError):
            dispatch_family(
                _dispatch_test_families,
                data,
                inputs_single=[data, out],
                device=DEVICE,
            )


# ==============================================================================
# Tests for build_family_dict
# ==============================================================================


@wp.kernel
def _bfd_single_kernel(a: wp.array(dtype=Any), b: wp.array(dtype=Any)):
    pass


@wp.kernel
def _bfd_batch_kernel(
    a: wp.array(dtype=Any), idx: wp.array(dtype=wp.int32), b: wp.array(dtype=Any)
):
    pass


@wp.kernel
def _bfd_ptr_kernel(
    a: wp.array(dtype=Any), ptr: wp.array(dtype=wp.int32), b: wp.array(dtype=Any)
):
    pass


class TestBuildFamilyDict:
    """Tests for build_family_dict."""

    def test_default_dtype_keys(self):
        families = build_family_dict(
            _bfd_single_kernel,
            lambda v, t: [wp.array(dtype=v), wp.array(dtype=t)],
            _bfd_batch_kernel,
            lambda v, t: [
                wp.array(dtype=v),
                wp.array(dtype=wp.int32),
                wp.array(dtype=t),
            ],
            _bfd_ptr_kernel,
            lambda v, t: [
                wp.array(dtype=v),
                wp.array(dtype=wp.int32),
                wp.array(dtype=t),
            ],
        )
        assert wp.vec3f in families
        assert wp.vec3d in families
        assert len(families) == 2

    def test_custom_dtype_pairs(self):
        families = build_family_dict(
            _bfd_single_kernel,
            lambda v, t: [wp.array(dtype=v), wp.array(dtype=t)],
            _bfd_batch_kernel,
            lambda v, t: [
                wp.array(dtype=v),
                wp.array(dtype=wp.int32),
                wp.array(dtype=t),
            ],
            _bfd_ptr_kernel,
            lambda v, t: [
                wp.array(dtype=v),
                wp.array(dtype=wp.int32),
                wp.array(dtype=t),
            ],
            dtype_pairs=((wp.vec3f, wp.float32),),
        )
        assert wp.vec3f in families
        assert wp.vec3d not in families
        assert len(families) == 1

    def test_returns_kernel_families(self):
        families = build_family_dict(
            _bfd_single_kernel,
            lambda v, t: [wp.array(dtype=v), wp.array(dtype=t)],
            _bfd_batch_kernel,
            lambda v, t: [
                wp.array(dtype=v),
                wp.array(dtype=wp.int32),
                wp.array(dtype=t),
            ],
            _bfd_ptr_kernel,
            lambda v, t: [
                wp.array(dtype=v),
                wp.array(dtype=wp.int32),
                wp.array(dtype=t),
            ],
        )
        for family in families.values():
            assert isinstance(family, KernelFamily)
            assert family.single is not None
            assert family.batch_idx is not None
            assert family.atom_ptr is not None

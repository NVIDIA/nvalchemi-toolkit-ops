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
Unit tests for nvalchemiops.types module.

This test suite validates the dtype conversion functions that map
PyTorch dtypes to Warp dtypes.

Tests cover:
- float16/float32/float64 scalar dtype conversions
- float16/float32/float64 vec3 dtype conversions
- float16/float32/float64 mat33 dtype conversions
- Unsupported dtype error handling
"""

import pytest
import torch
import warp as wp

from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype


class TestGetWpDtype:
    """Test get_wp_dtype function."""

    def test_float16(self):
        """Test float16 dtype conversion."""
        assert get_wp_dtype(torch.float16) == wp.float16

    def test_float32(self):
        """Test float32 dtype conversion."""
        assert get_wp_dtype(torch.float32) == wp.float32

    def test_float64(self):
        """Test float64 dtype conversion."""
        assert get_wp_dtype(torch.float64) == wp.float64

    def test_unsupported_dtype_int32(self):
        """Test that int32 raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_dtype(torch.int32)

    def test_unsupported_dtype_int64(self):
        """Test that int64 raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_dtype(torch.int64)

    def test_unsupported_dtype_bool(self):
        """Test that bool raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_dtype(torch.bool)

    def test_unsupported_dtype_complex64(self):
        """Test that complex64 raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_dtype(torch.complex64)

    def test_unsupported_dtype_bfloat16(self):
        """Test that bfloat16 raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_dtype(torch.bfloat16)


class TestGetWpVecDtype:
    """Test get_wp_vec_dtype function."""

    def test_float16(self):
        """Test float16 vec dtype conversion."""
        assert get_wp_vec_dtype(torch.float16) == wp.vec3h

    def test_float32(self):
        """Test float32 vec dtype conversion."""
        assert get_wp_vec_dtype(torch.float32) == wp.vec3f

    def test_float64(self):
        """Test float64 vec dtype conversion."""
        assert get_wp_vec_dtype(torch.float64) == wp.vec3d

    def test_unsupported_dtype_int32(self):
        """Test that int32 raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_vec_dtype(torch.int32)

    def test_unsupported_dtype_int64(self):
        """Test that int64 raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_vec_dtype(torch.int64)

    def test_unsupported_dtype_bool(self):
        """Test that bool raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_vec_dtype(torch.bool)

    def test_unsupported_dtype_complex64(self):
        """Test that complex64 raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_vec_dtype(torch.complex64)

    def test_unsupported_dtype_bfloat16(self):
        """Test that bfloat16 raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_vec_dtype(torch.bfloat16)


class TestGetWpMatDtype:
    """Test get_wp_mat_dtype function."""

    def test_float16(self):
        """Test float16 mat dtype conversion."""
        assert get_wp_mat_dtype(torch.float16) == wp.mat33h

    def test_float32(self):
        """Test float32 mat dtype conversion."""
        assert get_wp_mat_dtype(torch.float32) == wp.mat33f

    def test_float64(self):
        """Test float64 mat dtype conversion."""
        assert get_wp_mat_dtype(torch.float64) == wp.mat33d

    def test_unsupported_dtype_int32(self):
        """Test that int32 raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_mat_dtype(torch.int32)

    def test_unsupported_dtype_int64(self):
        """Test that int64 raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_mat_dtype(torch.int64)

    def test_unsupported_dtype_bool(self):
        """Test that bool raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_mat_dtype(torch.bool)

    def test_unsupported_dtype_complex64(self):
        """Test that complex64 raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_mat_dtype(torch.complex64)

    def test_unsupported_dtype_bfloat16(self):
        """Test that bfloat16 raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_mat_dtype(torch.bfloat16)


class TestDtypeConsistency:
    """Test consistency across dtype conversion functions."""

    @pytest.mark.parametrize(
        "torch_dtype", [torch.float16, torch.float32, torch.float64]
    )
    def test_all_functions_accept_same_dtypes(self, torch_dtype):
        """Test that all three functions accept the same supported dtypes."""
        # All should succeed without raising
        get_wp_dtype(torch_dtype)
        get_wp_vec_dtype(torch_dtype)
        get_wp_mat_dtype(torch_dtype)

    @pytest.mark.parametrize(
        "torch_dtype",
        [torch.int32, torch.int64, torch.bool, torch.complex64, torch.bfloat16],
    )
    def test_all_functions_reject_same_dtypes(self, torch_dtype):
        """Test that all three functions reject the same unsupported dtypes."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_dtype(torch_dtype)
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_vec_dtype(torch_dtype)
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_wp_mat_dtype(torch_dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

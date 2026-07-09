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
"""PyTorch dtype and device conversion utilities for Warp interop."""

from __future__ import annotations

import torch
import warp as wp


def get_wp_dtype(dtype: torch.dtype):
    """Get the Warp scalar dtype corresponding to a PyTorch dtype.

    Parameters
    ----------
    dtype : torch.dtype
        PyTorch floating-point dtype. Supported values: ``torch.float32``,
        ``torch.float64``, ``torch.float16``.

    Returns
    -------
    wp.type
        Corresponding Warp scalar dtype (``wp.float32``, ``wp.float64``, or
        ``wp.float16``).

    Raises
    ------
    ValueError
        If ``dtype`` is not one of the supported floating-point dtypes.
    """
    if dtype == torch.float32:
        return wp.float32
    elif dtype == torch.float64:
        return wp.float64
    elif dtype == torch.float16:
        return wp.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_wp_vec_dtype(dtype: torch.dtype):
    """Get the Warp 3-vector dtype corresponding to a PyTorch dtype.

    Parameters
    ----------
    dtype : torch.dtype
        PyTorch floating-point dtype. Supported values: ``torch.float32``,
        ``torch.float64``, ``torch.float16``.

    Returns
    -------
    wp.type
        Corresponding Warp vec3 dtype (``wp.vec3f``, ``wp.vec3d``, or
        ``wp.vec3h``).

    Raises
    ------
    ValueError
        If ``dtype`` is not one of the supported floating-point dtypes.
    """
    if dtype == torch.float32:
        return wp.vec3f
    elif dtype == torch.float64:
        return wp.vec3d
    elif dtype == torch.float16:
        return wp.vec3h
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_wp_mat_dtype(dtype: torch.dtype):
    """Get the Warp 3x3 matrix dtype corresponding to a PyTorch dtype.

    Parameters
    ----------
    dtype : torch.dtype
        PyTorch floating-point dtype. Supported values: ``torch.float32``,
        ``torch.float64``, ``torch.float16``.

    Returns
    -------
    wp.type
        Corresponding Warp mat33 dtype (``wp.mat33f``, ``wp.mat33d``, or
        ``wp.mat33h``).

    Raises
    ------
    ValueError
        If ``dtype`` is not one of the supported floating-point dtypes.
    """
    if dtype == torch.float32:
        return wp.mat33f
    elif dtype == torch.float64:
        return wp.mat33d
    elif dtype == torch.float16:
        return wp.mat33h
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

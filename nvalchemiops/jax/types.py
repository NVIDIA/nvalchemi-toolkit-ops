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
"""JAX dtype and device conversion utilities for Warp interop."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import warp as wp


def get_warp_device(jax_device: jax.Device) -> str:
    """Convert a JAX device to a Warp device string.

    Parameters
    ----------
    jax_device : jax.Device
        A JAX device object.

    Returns
    -------
    str
        The corresponding Warp device string ("cpu" or "cuda:N").

    Examples
    --------
    >>> import jax
    >>> from nvalchemiops.jax.types import get_warp_device
    >>> get_warp_device(jax.devices("cpu")[0])
    'cpu'
    """
    if jax_device.platform == "gpu":
        # JAX GPU devices have an id attribute for multi-GPU systems
        device_id = getattr(jax_device, "id", 0)
        return f"cuda:{device_id}"
    else:
        return "cpu"


def get_warp_device_from_array(arr: jax.Array) -> str:
    """Get the Warp device string from a JAX array.

    Parameters
    ----------
    arr : jax.Array
        A JAX array.

    Returns
    -------
    str
        The corresponding Warp device string.

    Notes
    -----
    Warp JAX bindings only support CUDA devices. This function always returns
    "cuda" without querying the array device to maintain jax.jit compatibility.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from nvalchemiops.jax.types import get_warp_device_from_array
    >>> arr = jnp.zeros(10)
    >>> device = get_warp_device_from_array(arr)
    """
    return "cuda"


def get_wp_dtype(dtype: jnp.dtype):
    """Get the warp dtype for a given JAX dtype.

    Parameters
    ----------
    dtype : jnp.dtype
        JAX array dtype (e.g., jnp.float32, jnp.float64).

    Returns
    -------
    wp.dtype
        Corresponding Warp dtype.

    Raises
    ------
    ValueError
        If the dtype is not supported.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from nvalchemiops.jax.types import get_wp_dtype
    >>> get_wp_dtype(jnp.float32)
    float32
    >>> get_wp_dtype(jnp.float64)
    float64
    """
    if dtype == jnp.float32:
        return wp.float32
    elif dtype == jnp.float64:
        return wp.float64
    elif dtype == jnp.float16:
        return wp.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_wp_vec_dtype(dtype: jnp.dtype):
    """Get the warp vec3 dtype for a given JAX dtype.

    Parameters
    ----------
    dtype : jnp.dtype
        JAX array dtype (e.g., jnp.float32, jnp.float64).

    Returns
    -------
    wp.dtype
        Corresponding Warp vec3 dtype (vec3f, vec3d, or vec3h).

    Raises
    ------
    ValueError
        If the dtype is not supported.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from nvalchemiops.jax.types import get_wp_vec_dtype
    >>> get_wp_vec_dtype(jnp.float32)
    vec3f
    >>> get_wp_vec_dtype(jnp.float64)
    vec3d
    """
    if dtype == jnp.float32:
        return wp.vec3f
    elif dtype == jnp.float64:
        return wp.vec3d
    elif dtype == jnp.float16:
        return wp.vec3h
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_wp_mat_dtype(dtype: jnp.dtype):
    """Get the warp mat33 dtype for a given JAX dtype.

    Parameters
    ----------
    dtype : jnp.dtype
        JAX array dtype (e.g., jnp.float32, jnp.float64).

    Returns
    -------
    wp.dtype
        Corresponding Warp mat33 dtype (mat33f, mat33d, or mat33h).

    Raises
    ------
    ValueError
        If the dtype is not supported.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from nvalchemiops.jax.types import get_wp_mat_dtype
    >>> get_wp_mat_dtype(jnp.float32)
    mat33f
    >>> get_wp_mat_dtype(jnp.float64)
    mat33d
    """
    if dtype == jnp.float32:
        return wp.mat33f
    elif dtype == jnp.float64:
        return wp.mat33d
    elif dtype == jnp.float16:
        return wp.mat33h
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

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

"""Lazy dtype-indexed ``jax_kernel`` wrappers, shared by the JAX bindings.

Importing this module builds no Warp/JAX FFI wrappers. The wrapper for a dtype is created on
first lookup, and the Warp kernel behind it still compiles on first launch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax.numpy as jnp
import warp as wp
from warp import jax_kernel

__all__: list[str] = []

_JAX_TO_WP = {jnp.float32: wp.float32, jnp.float64: wp.float64}


class _LazyJaxKernels:
    """Lazy ``{jnp.float32 | jnp.float64 -> jax_kernel}`` mapping.

    ``resolve`` maps a Warp scalar dtype to the specialized kernel for it, so the same class
    serves both a prebuilt overload dict and a kernel factory.
    """

    def __init__(
        self,
        resolve: Callable[[Any], Any],
        num_outputs: int,
        in_out_argnames: Sequence[str] | None,
        block_dim: int | None,
    ) -> None:
        self._resolve = resolve
        self._num_outputs = num_outputs
        self._in_out_argnames = in_out_argnames
        self._block_dim = block_dim
        self._cache: dict = {}

    def __getitem__(self, jax_dtype):
        if jax_dtype not in self._cache:
            self._cache[jax_dtype] = jax_kernel(
                self._resolve(_JAX_TO_WP[jax_dtype]),
                num_outputs=self._num_outputs,
                in_out_argnames=self._in_out_argnames,
                block_dim=self._block_dim,
                enable_backward=False,
            )
        return self._cache[jax_dtype]

    def __contains__(self, jax_dtype) -> bool:
        return jax_dtype in _JAX_TO_WP


def make_jax_kernels(
    wp_overload_dict: dict,
    num_outputs: int,
    in_out_argnames: Sequence[str] | None = None,
    block_dim: int | None = None,
) -> _LazyJaxKernels:
    """Return a lazy ``{jax_dtype -> jax_kernel}`` mapping over Warp overloads.

    Parameters
    ----------
    wp_overload_dict : dict
        Warp kernel overloads keyed by ``wp.float32`` / ``wp.float64``.
    num_outputs : int
        Number of output arrays the kernel returns.
    in_out_argnames : sequence of str, optional
        Names of in-place output arguments.
    block_dim : int, optional
        Threads per block. Required by kernels that reduce within a block; left to
        ``jax_kernel``'s own default when omitted.

    Returns
    -------
    _LazyJaxKernels
        Subscript with ``jnp.float32`` / ``jnp.float64``.
    """
    return _LazyJaxKernels(
        wp_overload_dict.__getitem__, num_outputs, in_out_argnames, block_dim
    )


def make_jax_kernel_factory(
    wp_kernel_factory: Callable[[Any], Any],
    num_outputs: int,
    in_out_argnames: Sequence[str] | None = None,
    block_dim: int | None = None,
) -> _LazyJaxKernels:
    """Return a lazy mapping for kernels built on demand by ``wp_kernel_factory``.

    As :func:`make_jax_kernels`, but ``wp_kernel_factory`` takes a Warp scalar dtype and
    returns the specialized :class:`warp.Kernel` for it.
    """
    return _LazyJaxKernels(wp_kernel_factory, num_outputs, in_out_argnames, block_dim)

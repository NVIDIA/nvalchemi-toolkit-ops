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

"""Lazy dtype-indexed ``jax_kernel`` wrapper mapping.

Importing this module does not construct Warp/JAX FFI wrappers. The wrapper for
a dtype is created on first lookup, and the underlying Warp kernel still compiles
on first launch.
"""

from __future__ import annotations

import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import jax_kernel

__all__: list[str] = []


class _LazyJaxKernels:
    """Lazy ``{jnp.float32 | jnp.float64 -> jax_kernel}`` mapping.

    Provides the ``_jax_kernel_by_dtype[dtype]`` lookup shape used by the
    electrostatics wrappers.
    """

    _JAX_TO_WP = {jnp.float32: wp.float32, jnp.float64: wp.float64}

    def __init__(
        self,
        wp_overload_dict: dict,
        num_outputs: int,
        in_out_argnames: list[str],
    ) -> None:
        self._wp_overload_dict = wp_overload_dict
        self._num_outputs = num_outputs
        self._in_out_argnames = in_out_argnames
        self._cache: dict = {}

    def __getitem__(self, jax_dtype):
        if jax_dtype not in self._cache:
            wp_dtype = self._JAX_TO_WP[jax_dtype]
            self._cache[jax_dtype] = jax_kernel(
                self._wp_overload_dict[wp_dtype],
                num_outputs=self._num_outputs,
                in_out_argnames=self._in_out_argnames,
                enable_backward=False,
            )
        return self._cache[jax_dtype]

    def __contains__(self, jax_dtype) -> bool:
        return jax_dtype in self._JAX_TO_WP


def _make_jax_kernels(
    wp_overload_dict: dict,
    num_outputs: int,
    in_out_argnames: list[str],
) -> _LazyJaxKernels:
    """Return a lazy ``{jax_dtype -> jax_kernel}`` mapping.

    Parameters
    ----------
    wp_overload_dict : dict
        Warp kernel overload dictionary keyed by ``wp.float32`` /
        ``wp.float64``.
    num_outputs : int
        Number of output arrays the kernel returns.
    in_out_argnames : list of str
        Names of in-place output arguments.

    Returns
    -------
    _LazyJaxKernels
        Subscript with ``jnp.float32`` / ``jnp.float64`` to obtain a
        :func:`warp.jax_experimental.jax_kernel` instance. The wrapper
        for each dtype is built lazily on first access.
    """
    return _LazyJaxKernels(wp_overload_dict, num_outputs, in_out_argnames)


class _LazyJaxKernelFactory:
    """Lazy ``{jnp.float32 | jnp.float64 -> jax_kernel}`` mapping from a factory."""

    _JAX_TO_WP = _LazyJaxKernels._JAX_TO_WP

    def __init__(
        self,
        wp_kernel_factory,
        num_outputs: int,
        in_out_argnames: list[str],
    ) -> None:
        self._wp_kernel_factory = wp_kernel_factory
        self._num_outputs = num_outputs
        self._in_out_argnames = in_out_argnames
        self._cache: dict = {}

    def __getitem__(self, jax_dtype):
        if jax_dtype not in self._cache:
            wp_dtype = self._JAX_TO_WP[jax_dtype]
            self._cache[jax_dtype] = jax_kernel(
                self._wp_kernel_factory(wp_dtype),
                num_outputs=self._num_outputs,
                in_out_argnames=self._in_out_argnames,
                enable_backward=False,
            )
        return self._cache[jax_dtype]

    def __contains__(self, jax_dtype) -> bool:
        return jax_dtype in self._JAX_TO_WP


def _make_jax_kernel_factory(
    wp_kernel_factory,
    num_outputs: int,
    in_out_argnames: list[str],
) -> _LazyJaxKernelFactory:
    """Return a lazy dtype mapping for kernels produced by ``wp_kernel_factory``.

    Parameters
    ----------
    wp_kernel_factory : Callable
        Function accepting a Warp scalar dtype and returning one specialized
        :class:`warp.Kernel`.
    num_outputs : int
        Number of output arrays the kernel returns.
    in_out_argnames : list of str
        Names of in-place output arguments.

    Returns
    -------
    _LazyJaxKernelFactory
        Subscript with ``jnp.float32`` / ``jnp.float64`` to obtain a lazy
        :func:`warp.jax_experimental.jax_kernel` wrapper.
    """
    return _LazyJaxKernelFactory(wp_kernel_factory, num_outputs, in_out_argnames)

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
Warp Dispatch Core
==================

Generic, module-agnostic primitives for Warp kernel dispatch:

- **Overload registration**: :func:`register_overloads` registers
  ``wp.overload`` variants across dtype pairs.
- **Dispatch tables**: :func:`build_dispatch_table` maps composite
  axis keys x dtype pairs to overloaded kernels.
- **Dispatch**: :func:`dispatch` looks up a kernel in a dispatch
  table and launches it via ``wp.launch``.
- **Validation**: :func:`validate_out_array` checks output arrays
  against a reference for shape, dtype, and device.

These helpers do **not** generate or modify Warp kernels at runtime.
All overloads are registered eagerly at module import time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import warp as wp

__all__ = [
    "DEFAULT_DTYPE_PAIRS",
    "build_dispatch_table",
    "dispatch",
    "register_overloads",
    "validate_out_array",
]

# =============================================================================
# Constants
# =============================================================================

#: Default dtype pairs: ``(vec3_type, scalar_type)``.
DEFAULT_DTYPE_PAIRS: tuple[tuple[type, type], ...] = (
    (wp.vec3f, wp.float32),
    (wp.vec3d, wp.float64),
)

# =============================================================================
# Overload Registration
# =============================================================================


def register_overloads(
    kernel: Any,
    signature_builder: Callable,
    dtype_pairs: tuple[tuple, ...] = DEFAULT_DTYPE_PAIRS,
    *,
    key_fn: Callable | None = None,
    dtypes: list | tuple | None = None,
) -> dict:
    """Register ``wp.overload`` variants for each dtype or dtype pair.

    This function supports two modes:

    **Pair mode** (default): iterates over ``dtype_pairs`` of
    ``(vec_dtype, scalar_dtype)`` tuples, calling
    ``signature_builder(vec_dtype, scalar_dtype)``.

    **Flat mode**: when ``dtypes`` is provided, iterates over a flat
    sequence of types, calling ``signature_builder(dtype)`` with a
    single argument.  ``dtype_pairs`` is ignored in this mode.

    Parameters
    ----------
    kernel : warp kernel
        The generic ``@wp.kernel`` function to overload.
    signature_builder : callable
        In pair mode: ``(vec_dtype, scalar_dtype) -> list``.
        In flat mode: ``(dtype,) -> list``.
    dtype_pairs : tuple of (vec_dtype, scalar_dtype) tuples
        Dtype pairs to register (pair mode).  Defaults to
        ``(vec3f, float32)`` and ``(vec3d, float64)``.
        Ignored when ``dtypes`` is provided.
    key_fn : callable or None
        In pair mode: ``(vec_dtype, scalar_dtype) -> key``.
        In flat mode: ``(dtype,) -> key``.
        Defaults to ``vec_dtype`` (pair mode) or ``dtype`` (flat mode).
    dtypes : list or tuple or None
        Flat sequence of types for flat mode.  When provided, pair mode
        is skipped and ``signature_builder`` receives a single argument.

    Returns
    -------
    dict
        Mapping from key to the overloaded kernel handle.

    Examples
    --------
    Pair mode (vec/scalar pairs)::

        overloads = register_overloads(
            _velocity_rescale_kernel,
            lambda v, t: [wp.array(dtype=v), wp.array(dtype=t)],
        )

    Flat mode (single dtype list)::

        overloads = register_overloads(
            _segmented_sum_kernel,
            lambda t: [wp.array(dtype=t), wp.array(dtype=wp.int32),
                       wp.array(dtype=t), wp.int32, wp.int32],
            dtypes=[wp.float32, wp.float64, wp.vec3f, wp.vec3d],
        )
    """
    if dtypes is not None:
        if key_fn is None:
            key_fn = lambda t: t  # noqa: E731
        out: dict = {}
        for dt in dtypes:
            out[key_fn(dt)] = wp.overload(kernel, signature_builder(dt))
        return out

    # Pair mode (original behavior)
    if key_fn is None:
        key_fn = lambda v, t: v  # noqa: E731

    out: dict = {}
    for vec_dtype, scalar_dtype in dtype_pairs:
        sig = signature_builder(vec_dtype, scalar_dtype)
        key = key_fn(vec_dtype, scalar_dtype)
        out[key] = wp.overload(kernel, sig)
    return out


# =============================================================================
# Dispatch Table
# =============================================================================


def build_dispatch_table(
    entries: dict[Any, tuple],
    dtype_pairs: tuple[tuple, ...] = DEFAULT_DTYPE_PAIRS,
) -> dict:
    """Build a dispatch table mapping ``(axis_key, dtype_key)`` to overloaded kernels.

    This is the generic form of overload registration for multi-axis
    dispatch.  Each entry maps an arbitrary axis key (enum value,
    tuple of enum values, string, etc.) to a ``(kernel, signature_builder)``
    pair.  The table is the Cartesian product of axis keys and dtype pairs.

    Parameters
    ----------
    entries : dict
        ``{axis_key: (kernel, signature_builder)}`` where *axis_key* is
        any hashable value and *signature_builder* is
        ``(vec_dtype, scalar_dtype) -> list``.
    dtype_pairs : tuple of (vec_dtype, scalar_dtype) tuples
        Dtype pairs to register.  Defaults to :data:`DEFAULT_DTYPE_PAIRS`.

    Returns
    -------
    dict
        ``{(axis_key, vec_dtype): overloaded_kernel}`` mapping.

    Examples
    --------
    Build a table dispatching over execution mode::

        from enum import Enum

        class Mode(Enum):
            SINGLE = "single"
            BATCHED = "batched"

        table = build_dispatch_table({
            Mode.SINGLE: (_single_kernel, lambda v, t: [wp.array(dtype=v)]),
            Mode.BATCHED: (_batch_kernel, lambda v, t: [wp.array(dtype=v), wp.array(dtype=wp.int32)]),
        })
        # table[(Mode.SINGLE, wp.vec3f)] -> overloaded single kernel for float32
    """
    table: dict = {}
    for axis_key, (kernel, sig_builder) in entries.items():
        for vec_dtype, scalar_dtype in dtype_pairs:
            table[(axis_key, vec_dtype)] = wp.overload(
                kernel, sig_builder(vec_dtype, scalar_dtype)
            )
    return table


# =============================================================================
# Dispatch
# =============================================================================


def dispatch(
    table: dict,
    key: tuple,
    *,
    dim: int,
    inputs: list,
    device: str,
    outputs: list | None = None,
) -> None:
    """Look up a kernel in a dispatch table and launch it.

    Parameters
    ----------
    table : dict
        Dispatch table mapping composite keys to overloaded kernels,
        typically built by :func:`build_dispatch_table`.
    key : tuple
        Composite dispatch key, e.g. ``(axis_value, vec_dtype)``.
    dim : int
        Launch grid dimension.
    inputs : list
        Kernel input arguments.
    device : str
        Warp device string.
    outputs : list or None
        Optional kernel output arguments.

    Raises
    ------
    KeyError
        If *key* is not found in the dispatch table.  The error
        message lists all available keys.
    """
    try:
        kernel = table[key]
    except KeyError:
        raise KeyError(
            f"No kernel registered for dispatch key {key!r}. "
            f"Available keys: {sorted(table.keys(), key=str)}"
        ) from None
    wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs or [], device=device)


# =============================================================================
# Validation
# =============================================================================


def validate_out_array(
    out: wp.array,
    reference: wp.array,
    name: str,
) -> None:
    """Validate that an output array matches a reference in shape, dtype, device.

    Parameters
    ----------
    out : wp.array
        Output array to validate.
    reference : wp.array
        Reference array whose shape, dtype, and device are expected.
    name : str
        Human-readable name for error messages.

    Raises
    ------
    ValueError
        If shape, dtype, or device do not match.
    """
    if out.shape != reference.shape:
        raise ValueError(
            f"{name} shape mismatch: expected {reference.shape}, got {out.shape}"
        )
    if out.dtype != reference.dtype:
        raise ValueError(
            f"{name} dtype mismatch: expected {reference.dtype}, got {out.dtype}"
        )
    if str(out.device) != str(reference.device):
        raise ValueError(
            f"{name} device mismatch: expected {reference.device}, got {out.device}"
        )

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
Launch Helpers for Dynamics Kernels
====================================

Dynamics-specific dispatch wrappers built on top of
:mod:`nvalchemiops.warp_dispatch` generic primitives.

This module provides:

- :class:`ExecutionMode` -- the three dynamics execution modes
  (single / batch_idx / atom_ptr).
- :class:`KernelFamily` -- container grouping per-mode kernel variants.
- :func:`resolve_execution_mode` -- pick mode from optional batch arrays.
- :func:`launch_family` -- launch the right variant from a KernelFamily.
- :func:`dispatch_family` -- end-to-end: resolve mode, infer
  device/dtype/dim, launch.
- :func:`build_family_dict` -- build ``{vec_dtype: KernelFamily}`` dicts
  with eager overload registration.

Generic utilities (:func:`register_overloads`, :data:`DEFAULT_DTYPE_PAIRS`)
are re-exported from :mod:`nvalchemiops.warp_dispatch` for backward
compatibility.

Design Rule
-----------
Warp interface functions must receive ALL arrays pre-allocated.
Output arrays are required, not optional.  Output/scratch arrays that
require zeroing are zeroed internally by the function before use.
Only Torch interface functions may accept optional arguments and
allocate as needed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import warp as wp

# Re-export generic primitives from warp_dispatch for backward compatibility.
from nvalchemiops.warp_dispatch import (
    DEFAULT_DTYPE_PAIRS,
    register_overloads,
)

# =============================================================================
# Types
# =============================================================================


class ExecutionMode(Enum):
    """Execution mode for batched vs single-system kernel dispatch.

    Three modes are supported:

    - ``SINGLE``: One system, no batch metadata. Scalar arrays have shape ``(1,)``.
    - ``BATCH_IDX``: Multiple systems, each atom tagged with a system index.
      Kernel launches with ``dim=num_atoms``.
    - ``ATOM_PTR``: Multiple systems, CSR-style pointer array defines atom ranges.
      Kernel launches with ``dim=num_systems``.
    """

    SINGLE = "single"
    BATCH_IDX = "batch_idx"
    ATOM_PTR = "atom_ptr"


@dataclass(frozen=True)
class KernelFamily:
    """Container for per-mode kernel overloads.

    Groups the three execution-mode variants of a single logical operation
    so that dispatch helpers can select the correct kernel by mode.

    Parameters
    ----------
    single : object
        Kernel (or overloaded kernel) for single-system mode.
    batch_idx : object or None
        Kernel for ``batch_idx`` mode. None if not supported.
    atom_ptr : object or None
        Kernel for ``atom_ptr`` (CSR pointer) mode. None if not supported.
    """

    single: object
    batch_idx: object | None = None
    atom_ptr: object | None = None


# =============================================================================
# Dispatch
# =============================================================================


def resolve_execution_mode(
    batch_idx: wp.array | None,
    atom_ptr: wp.array | None,
) -> ExecutionMode:
    """Determine execution mode from optional batch metadata arrays.

    Parameters
    ----------
    batch_idx : wp.array or None
        Per-atom system index array (``batch_idx`` mode).
    atom_ptr : wp.array or None
        CSR-style atom pointer array (``atom_ptr`` mode).

    Returns
    -------
    ExecutionMode
        The resolved execution mode.

    Raises
    ------
    ValueError
        If both ``batch_idx`` and ``atom_ptr`` are provided.
    """
    if batch_idx is not None and atom_ptr is not None:
        raise ValueError("Provide batch_idx OR atom_ptr, not both")
    if atom_ptr is not None:
        return ExecutionMode.ATOM_PTR
    if batch_idx is not None:
        return ExecutionMode.BATCH_IDX
    return ExecutionMode.SINGLE


def launch_family(
    family: KernelFamily,
    *,
    mode: ExecutionMode,
    dim: int,
    inputs_single: list,
    inputs_batch: list | None = None,
    inputs_ptr: list | None = None,
    device: str,
) -> None:
    """Launch the correct kernel variant from a :class:`KernelFamily`.

    Selects the kernel matching *mode* and launches it with the
    appropriate inputs and grid dimension.

    Parameters
    ----------
    family : KernelFamily
        Container holding ``single``, ``batch_idx``, and ``atom_ptr``
        kernel variants.
    mode : ExecutionMode
        Resolved execution mode (from :func:`resolve_execution_mode`).
    dim : int
        Launch grid dimension.  For ``SINGLE`` and ``BATCH_IDX`` this is
        typically ``num_atoms``; for ``ATOM_PTR`` it is ``num_systems``.
    inputs_single : list
        Kernel input list for single-system mode.
    inputs_batch : list or None
        Kernel input list for ``batch_idx`` mode.
    inputs_ptr : list or None
        Kernel input list for ``atom_ptr`` mode.
    device : str
        Warp device string.

    Raises
    ------
    ValueError
        If the requested mode's kernel is ``None`` in the family.
    """
    if mode is ExecutionMode.ATOM_PTR:
        kernel = family.atom_ptr
        if kernel is None:
            raise ValueError("atom_ptr mode not supported for this kernel family")
        wp.launch(kernel, dim=dim, inputs=inputs_ptr, device=device)
    elif mode is ExecutionMode.BATCH_IDX:
        kernel = family.batch_idx
        if kernel is None:
            raise ValueError("batch_idx mode not supported for this kernel family")
        wp.launch(kernel, dim=dim, inputs=inputs_batch, device=device)
    else:
        wp.launch(family.single, dim=dim, inputs=inputs_single, device=device)


# =============================================================================
# High-Level Dispatch
# =============================================================================


def dispatch_family(
    family_dict: dict,
    primary_array: wp.array,
    *,
    batch_idx: wp.array | None = None,
    atom_ptr: wp.array | None = None,
    device: str | None = None,
    inputs_single: list,
    inputs_batch: list | None = None,
    inputs_ptr: list | None = None,
) -> None:
    """Resolve mode, infer device/dtype/dim, and launch the correct kernel.

    This combines :func:`resolve_execution_mode`, device inference,
    dtype-based family lookup, dimension calculation, and
    :func:`launch_family` into a single call.

    Parameters
    ----------
    family_dict : dict
        Mapping from dtype (e.g. ``wp.vec3f``) to :class:`KernelFamily`.
    primary_array : wp.array
        The main input array.  Used for device inference (when *device*
        is ``None``), dtype-based family lookup, and default ``dim``.
    batch_idx, atom_ptr : wp.array or None
        Optional batch metadata (mutually exclusive).
    device : str or None
        Warp device.  If ``None``, inferred from *primary_array*.
    inputs_single, inputs_batch, inputs_ptr : list
        Kernel input lists forwarded to :func:`launch_family`.
    """
    mode = resolve_execution_mode(batch_idx, atom_ptr)
    if device is None:
        device = primary_array.device
    family = family_dict[primary_array.dtype]
    n = primary_array.shape[0]
    dim = atom_ptr.shape[0] - 1 if mode is ExecutionMode.ATOM_PTR else n
    launch_family(
        family,
        mode=mode,
        dim=dim,
        inputs_single=inputs_single,
        inputs_batch=inputs_batch,
        inputs_ptr=inputs_ptr,
        device=device,
    )


# =============================================================================
# Family Construction
# =============================================================================


def build_family_dict(
    single_kernel,
    single_sig: Callable,
    batch_kernel,
    batch_sig: Callable,
    ptr_kernel,
    ptr_sig: Callable,
    dtype_pairs: tuple[tuple, ...] = DEFAULT_DTYPE_PAIRS,
) -> dict:
    """Build a ``{vec_dtype: KernelFamily}`` dict with eager overload registration.

    A convenience wrapper that calls :func:`~nvalchemiops.warp_dispatch.register_overloads`
    for each execution-mode kernel, then packs the results into
    :class:`KernelFamily` instances.

    Parameters
    ----------
    single_kernel, batch_kernel, ptr_kernel
        Generic ``@wp.kernel`` functions for each execution mode.
    single_sig, batch_sig, ptr_sig : callable
        Signature builders ``(vec_dtype, scalar_dtype) -> list``.
    dtype_pairs : tuple
        Dtype pairs to register.  Defaults to ``DEFAULT_DTYPE_PAIRS``.

    Returns
    -------
    dict
        ``{vec_dtype: KernelFamily}`` mapping.
    """
    single_overloads = register_overloads(single_kernel, single_sig, dtype_pairs)
    batch_overloads = register_overloads(batch_kernel, batch_sig, dtype_pairs)
    ptr_overloads = register_overloads(ptr_kernel, ptr_sig, dtype_pairs)

    return {
        vec_dtype: KernelFamily(
            single=single_overloads[vec_dtype],
            batch_idx=batch_overloads[vec_dtype],
            atom_ptr=ptr_overloads[vec_dtype],
        )
        for vec_dtype, _ in dtype_pairs
    }

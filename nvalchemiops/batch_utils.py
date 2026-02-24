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

"""Index and pointer conversion utilities for batched / segmented arrays.

Provides conversions between the two standard batching representations:

- **batch_idx** (sorted segment index): ``shape (N,), dtype int32``
  where ``batch_idx[i]`` is the system that atom *i* belongs to.
- **atom_ptr** (CSR-style pointer): ``shape (M+1,), dtype int32``
  where system *s* owns atoms ``[atom_ptr[s], atom_ptr[s+1])``.

All public functions follow a strict no-allocation policy: callers must
pre-allocate (and, where noted, zero-initialize) every output and scratch
array.
"""

from __future__ import annotations

import warp as wp

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _create_atom_ptr_kernel(
    atom_counts: wp.array(dtype=wp.int32),
    atom_ptr: wp.array(dtype=wp.int32),
):
    """Sequential prefix sum: atom_ptr[i+1] = atom_ptr[i] + atom_counts[i].

    Launch with ``dim=1`` (single thread).
    """
    tid = wp.tid()
    if tid == 0:
        atom_ptr[0] = wp.int32(0)
        for i in range(atom_counts.shape[0]):
            atom_ptr[i + 1] = atom_ptr[i] + atom_counts[i]


@wp.kernel(enable_backward=False)
def _create_batch_idx_kernel(
    atom_ptr: wp.array(dtype=wp.int32),
    batch_idx: wp.array(dtype=wp.int32),
):
    """Assign system index to every atom from atom_ptr.

    Launch with ``dim=num_systems``.
    """
    sys = wp.tid()
    a0 = atom_ptr[sys]
    a1 = atom_ptr[sys + 1]
    for i in range(a0, a1):
        batch_idx[i] = wp.int32(sys)


@wp.kernel(enable_backward=False)
def _atoms_per_system_from_batch_idx_kernel(
    batch_idx: wp.array(dtype=wp.int32),
    atom_counts: wp.array(dtype=wp.int32),
):
    """Count atoms per system from batch_idx using atomics.

    Launch with ``dim=total_atoms``.  ``atom_counts`` must be zero-initialized.
    """
    i = wp.tid()
    sys = batch_idx[i]
    wp.atomic_add(atom_counts, sys, wp.int32(1))


@wp.kernel(enable_backward=False)
def _atoms_per_system_from_ptr_kernel(
    atom_ptr: wp.array(dtype=wp.int32),
    atom_counts: wp.array(dtype=wp.int32),
):
    """Compute atoms per system by differencing adjacent atom_ptr entries.

    Launch with ``dim=num_systems``.
    """
    sys = wp.tid()
    atom_counts[sys] = atom_ptr[sys + 1] - atom_ptr[sys]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_atom_ptr(
    atom_counts: wp.array,
    atom_ptr: wp.array,
) -> None:
    """Build CSR-style pointer array from per-system atom counts.

    ``atom_ptr[0] = 0; atom_ptr[s+1] = atom_ptr[s] + atom_counts[s]``

    Parameters
    ----------
    atom_counts : wp.array(dtype=int32), shape (M,)
        Number of atoms in each system.
    atom_ptr : wp.array(dtype=int32), shape (M+1,)
        Output pointer array.  Need not be zero-initialized.
    """
    wp.launch(
        _create_atom_ptr_kernel,
        dim=1,
        inputs=[atom_counts, atom_ptr],
        device=atom_counts.device,
    )


def create_batch_idx(
    atom_ptr: wp.array,
    batch_idx: wp.array,
) -> None:
    """Fill a sorted segment-index array from a CSR pointer array.

    For each system *s*, sets ``batch_idx[i] = s`` for
    ``i in [atom_ptr[s], atom_ptr[s+1])``.

    Parameters
    ----------
    atom_ptr : wp.array(dtype=int32), shape (M+1,)
        CSR-style pointer array.
    batch_idx : wp.array(dtype=int32), shape (N,)
        Output segment indices.
    """
    num_systems = atom_ptr.shape[0] - 1
    if num_systems == 0:
        return
    wp.launch(
        _create_batch_idx_kernel,
        dim=num_systems,
        inputs=[atom_ptr, batch_idx],
        device=atom_ptr.device,
    )


def atoms_per_system_from_batch_idx(
    batch_idx: wp.array,
    atom_counts: wp.array,
) -> None:
    """Count atoms per system from a sorted segment-index array.

    ``atom_counts`` must be zero-initialized by the caller.

    Parameters
    ----------
    batch_idx : wp.array(dtype=int32), shape (N,)
        Sorted segment indices.
    atom_counts : wp.array(dtype=int32), shape (M,)
        Output atom counts. Must be zero-initialized.
    """
    N = batch_idx.shape[0]
    if N == 0:
        return
    wp.launch(
        _atoms_per_system_from_batch_idx_kernel,
        dim=N,
        inputs=[batch_idx, atom_counts],
        device=batch_idx.device,
    )


def atoms_per_system_from_atom_ptr(
    atom_ptr: wp.array,
    atom_counts: wp.array,
) -> None:
    """Compute atoms per system by differencing adjacent atom_ptr entries.

    Parameters
    ----------
    atom_ptr : wp.array(dtype=int32), shape (M+1,)
        CSR-style pointer array.
    atom_counts : wp.array(dtype=int32), shape (M,)
        Output atom counts.
    """
    M = atom_counts.shape[0]
    if M == 0:
        return
    wp.launch(
        _atoms_per_system_from_ptr_kernel,
        dim=M,
        inputs=[atom_ptr, atom_counts],
        device=atom_ptr.device,
    )


def batch_idx_to_atom_ptr(
    batch_idx: wp.array,
    atom_counts: wp.array,
    atom_ptr: wp.array,
) -> None:
    """Convert batch_idx to atom_ptr representation.

    ``atom_counts`` is used as scratch space and is overwritten.
    It must be zero-initialized by the caller.

    Parameters
    ----------
    batch_idx : wp.array(dtype=int32), shape (N,)
        Sorted segment indices.
    atom_counts : wp.array(dtype=int32), shape (M,)
        Scratch for intermediate counts. Must be zero-initialized.
    atom_ptr : wp.array(dtype=int32), shape (M+1,)
        Output CSR-style pointer array.
    """
    atoms_per_system_from_batch_idx(batch_idx, atom_counts)
    create_atom_ptr(atom_counts, atom_ptr)


def atom_ptr_to_batch_idx(
    atom_ptr: wp.array,
    batch_idx: wp.array,
) -> None:
    """Convert atom_ptr to batch_idx representation.

    Parameters
    ----------
    atom_ptr : wp.array(dtype=int32), shape (M+1,)
        CSR-style pointer array.
    batch_idx : wp.array(dtype=int32), shape (N,)
        Output sorted segment indices.
    """
    create_batch_idx(atom_ptr, batch_idx)

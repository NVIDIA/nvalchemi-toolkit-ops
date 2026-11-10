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

from typing import Any

import warp as wp

from nvalchemiops.warp_dispatch import register_overloads

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


@wp.func
def _bounded_segment_length(value: wp.int32) -> wp.int32:
    """Return an unchecked nonnegative length within the int32 index range."""
    return wp.max(value, 0)


@wp.func
def _bounded_segment_length(value: wp.int64) -> wp.int32:  # noqa: F811
    """Bound a wide unchecked count before scans and int32 indexing."""
    return int(wp.clamp(value, wp.int64(0), wp.int64(2147483647)))


@wp.kernel(enable_backward=False)
def _segment_counts(
    seglen: wp.array(dtype=Any),
    row_counts: wp.array(dtype=wp.int64),
):
    """Write nonnegative row counts for an overflow-safe inclusive device scan.

    Parameters
    ----------
    seglen : wp.array, shape (R,), dtype int32 or int64
        Runtime segment lengths. Negative values are treated as zero.
    row_counts : wp.array, shape (R + 1,), dtype int64
        OUTPUT: Pre-allocated row counts with a leading zero, followed by
        the nonnegative lengths.

    Notes
    -----
    Thread launch: ``R + 1`` threads, one per output element.
    Modifies: every element of ``row_counts``.

    Apply an inclusive scan to ``row_counts`` before passing it to
    ``_segment_offsets_clamped``.

    See Also
    --------
    _segment_offsets_clamped : Convert scanned row counts to bounded offsets.
    """
    index = wp.tid()
    if index == 0:
        row_counts[0] = wp.int64(0)
    else:
        # Bounded counts keep scans over int32-sized segment lists within int64.
        row_counts[index] = wp.int64(_bounded_segment_length(seglen[index - 1]))


register_overloads(
    _segment_counts,
    lambda dtype: [wp.array(dtype=dtype), wp.array(dtype=wp.int64)],
    dtypes=(wp.int32, wp.int64),
)


@wp.func
def _segment_chunk_rows(active_rows: int, extra_slots: int, minimum_rows: int):
    """Fit active SegMM rows into fixed extra slots using 32-row chunk multiples."""
    required = wp.max(active_rows - 1, 0) / (extra_slots + 1) + 1
    return wp.max(minimum_rows, (required + 31) / 32 * 32)


@wp.func
def _checked_segment_length(value: Any, capacity: int):
    """Bound wide counts and encode invalid lengths as capacity plus one."""
    raw = wp.int64(value)
    ceiling = wp.int64(capacity) + wp.int64(1)
    if raw < wp.int64(0):
        return ceiling
    return wp.min(raw, ceiling)


@wp.kernel(enable_backward=False)
def _segment_counts_checked(
    seglen: wp.array(dtype=Any),
    row_counts: wp.array(dtype=wp.int64),
    capacity: int,
):
    """Prepare counts whose scanned total also detects invalid segment lengths.

    Negative or oversized lengths contribute ``capacity + 1``. Each count
    fits this bound, so an int64 scan is safe for int32-sized segment counts.

    Notes
    -----
    Thread launch: ``R + 1`` threads, one per output element.
    Modifies: every element of caller-owned ``row_counts``.
    """
    index = wp.tid()
    if index == 0:
        row_counts[0] = wp.int64(0)
    else:
        row_counts[index] = _checked_segment_length(seglen[index - 1], capacity)


register_overloads(
    _segment_counts_checked,
    lambda dtype: [wp.array(dtype=dtype), wp.array(dtype=wp.int64), wp.int32],
    dtypes=(wp.int32, wp.int64),
)


def _register_segment_offset_overloads(kernel: wp.Kernel) -> None:
    """Register length types before CUDA graphs can retain the module executable."""
    register_overloads(
        kernel,
        lambda dtype: [
            wp.array(dtype=dtype),
            wp.int32,
            wp.int32,
            wp.bool,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.int32,
            wp.array(dtype=wp.int32),
            wp.int32,
            wp.array(dtype=wp.int32),
        ],
        dtypes=(wp.int32, wp.int64),
    )


def _make_segment_offsets_small(with_chunks: bool, validate: bool = False) -> wp.Kernel:
    """Specialize the small prefix builder with an optional second work schedule."""

    @wp.kernel(enable_backward=False)
    def _segment_offsets_small(
        seglen: wp.array(dtype=Any),
        capacity: int,
        tile_rows: int,
        skip_first_tile: bool,
        row_offsets: wp.array(dtype=wp.int32),
        tile_offsets: wp.array(dtype=wp.int32),
        chunk_rows: int,
        chunk_offsets: wp.array(dtype=wp.int32),
        minimum_chunk_rows: int,
        validation: wp.array(dtype=wp.int32),
    ):
        """Build capacity-bounded prefix arrays directly for small segment counts.

        Parameters
        ----------
        seglen : wp.array, shape (R,), dtype int32 or int64
            Runtime segment lengths. Unchecked calls treat negative values as zero.
        capacity : int
            Maximum number of active rows, stored as an int32 scalar.
        tile_rows : int
            Positive number of rows in each work tile.
        skip_first_tile : bool
            Exclude each nonempty segment's first tile from the work prefix.
        row_offsets : wp.array, shape (R + 1,), dtype int32
            OUTPUT: Pre-allocated cumulative row offsets, bounded by ``capacity``.
        tile_offsets : wp.array, shape (R + 1,), dtype int32
            OUTPUT: Pre-allocated cumulative counts of ``tile_rows``-row tiles.
            Both offset arrays start with zero.

        chunk_rows : int
            Positive row count for the fused additional-chunk schedule. Ignored
            by the ordinary specialization.
        chunk_offsets : wp.array, shape (R + 1,), dtype int32
            OUTPUT: Additional-chunk counts or prefixes in the fused specialization.
            Unused by the ordinary specialization.
        minimum_chunk_rows : int
            Positive floor for adapting the chunk schedule to active rows, or zero
            to retain its static size and skip the total-row pass.
        validation : wp.array, shape (1,), dtype int32, or None
            OUTPUT: One for valid lengths, zero otherwise, in the checked
            specialization. Unused in the unchecked specialization.

        Notes
        -----
        Thread launch: ``R + 1`` threads, one per prefix element.
        Each thread sums its preceding segments, giving quadratic total work.
        Callers choose a bounded segment count appropriate to their launch overhead.
        Modifies: every element of ``row_offsets`` and ``tile_offsets``; the fused
        specialization also overwrites every element of ``chunk_offsets``.
        The checked specialization also overwrites ``validation[0]``.

        See Also
        --------
        _segment_offsets_clamped : Build offsets from scanned row counts.
        """
        index = wp.tid()
        if minimum_chunk_rows > 0:
            total = int(0)
            for group in range(seglen.shape[0]):
                total += wp.min(
                    _bounded_segment_length(seglen[group]), capacity - total
                )
            if wp.static(with_chunks):
                chunk_rows = _segment_chunk_rows(
                    total, (capacity - 1) / chunk_rows, minimum_chunk_rows
                )
            else:
                tile_rows = _segment_chunk_rows(
                    total, (capacity - 1) / tile_rows, minimum_chunk_rows
                )
        rows = int(0)
        tiles = int(0)
        chunks = int(0)
        valid = int(1)
        for group in range(index):
            if wp.static(validate):
                raw = wp.int64(seglen[group])
                if raw < wp.int64(0) or raw > wp.int64(capacity - rows):
                    valid = 0
                length = int(
                    wp.min(wp.max(raw, wp.int64(0)), wp.int64(capacity - rows))
                )
            else:
                length = wp.min(_bounded_segment_length(seglen[group]), capacity - rows)
            rows += length
            count = length / tile_rows + int(length % tile_rows != 0)
            if skip_first_tile:
                count = wp.max(count - 1, 0)
            tiles += count
            if wp.static(with_chunks):
                chunk_count = length / chunk_rows + int(length % chunk_rows != 0)
                chunks += wp.max(chunk_count - 1, 0)
        if wp.static(validate):
            if index == seglen.shape[0]:
                validation[0] = valid
        row_offsets[index] = rows
        tile_offsets[index] = tiles
        if wp.static(with_chunks):
            chunk_offsets[index] = chunks

    _register_segment_offset_overloads(_segment_offsets_small)
    return _segment_offsets_small


_segment_offsets_small = _make_segment_offsets_small(False)
_segment_offsets_small_fused = _make_segment_offsets_small(True)
_segment_offsets_small_checked = _make_segment_offsets_small(False, True)
_segment_offsets_small_fused_checked = _make_segment_offsets_small(True, True)


def _make_segment_offsets_clamped(
    with_chunks: bool, validate: bool = False
) -> wp.Kernel:
    """Specialize bounded count construction with an optional chunk-count output."""

    @wp.kernel(enable_backward=False)
    def _segment_offsets_clamped(
        row_prefix: wp.array(dtype=wp.int64),
        capacity: int,
        tile_rows: int,
        skip_first_tile: bool,
        row_offsets: wp.array(dtype=wp.int32),
        tile_offsets: wp.array(dtype=wp.int32),
        chunk_rows: int,
        chunk_offsets: wp.array(dtype=wp.int32),
        minimum_chunk_rows: int,
        validation: wp.array(dtype=wp.int32),
    ):
        """Clamp wide row prefixes to capacity and count tiles over the clipped ranges.

        Parameters
        ----------
        row_prefix : wp.array, shape (R + 1,), dtype int64
            Inclusive scan of nonnegative segment lengths with a leading zero.
        capacity : int
            Maximum number of active rows, stored as an int32 scalar.
        tile_rows : int
            Positive number of rows in each work tile.
        skip_first_tile : bool
            Exclude each nonempty segment's first tile from the work prefix.
        row_offsets : wp.array, shape (R + 1,), dtype int32
            OUTPUT: Pre-allocated cumulative row offsets, bounded by ``capacity``.
        tile_offsets : wp.array, shape (R + 1,), dtype int32
            OUTPUT: Pre-allocated per-segment counts of ``tile_rows``-row tiles,
            preceded by zero. Apply an inclusive scan before using as offsets.

        chunk_rows : int
            Positive row count for the fused additional-chunk schedule. Ignored
            by the ordinary specialization.
        chunk_offsets : wp.array, shape (R + 1,), dtype int32
            OUTPUT: Additional-chunk counts or prefixes in the fused specialization.
            Unused by the ordinary specialization.
        minimum_chunk_rows : int
            Positive floor for adapting the chunk schedule to active rows, or zero
            to retain its static size.
        validation : wp.array, shape (1,), dtype int32, or None
            OUTPUT: Validity of the checked count prefix. Unused in the
            unchecked specialization.

        Notes
        -----
        Thread launch: ``R + 1`` threads, one per prefix element.
        Modifies: every element of ``row_offsets`` and ``tile_offsets``; the fused
        specialization also overwrites every element of ``chunk_offsets``.
        The checked specialization also overwrites ``validation[0]``.

        See Also
        --------
        _segment_counts : Prepare nonnegative lengths for the row-prefix scan.
        _segment_offsets_small : Build row and tile offsets directly for small inputs.
        """
        index = wp.tid()
        if wp.static(validate):
            if index == 0:
                validation[0] = int(
                    row_prefix[row_prefix.shape[0] - 1] <= wp.int64(capacity)
                )
        if minimum_chunk_rows > 0:
            total = int(wp.min(row_prefix[row_prefix.shape[0] - 1], wp.int64(capacity)))
            if wp.static(with_chunks):
                chunk_rows = _segment_chunk_rows(
                    total, (capacity - 1) / chunk_rows, minimum_chunk_rows
                )
            else:
                tile_rows = _segment_chunk_rows(
                    total, (capacity - 1) / tile_rows, minimum_chunk_rows
                )
        row = int(wp.min(row_prefix[index], wp.int64(capacity)))
        row_offsets[index] = row
        if index == 0:
            tile_offsets[0] = 0
            if wp.static(with_chunks):
                chunk_offsets[0] = 0
        else:
            previous = int(wp.min(row_prefix[index - 1], wp.int64(capacity)))
            length = row - previous
            count = length / tile_rows + int(length % tile_rows != 0)
            if skip_first_tile:
                count = wp.max(count - 1, 0)
            tile_offsets[index] = count
            if wp.static(with_chunks):
                chunk_count = length / chunk_rows + int(length % chunk_rows != 0)
                chunk_offsets[index] = wp.max(chunk_count - 1, 0)

    return _segment_offsets_clamped


_segment_offsets_clamped = _make_segment_offsets_clamped(False)
_segment_offsets_clamped_fused = _make_segment_offsets_clamped(True)
_segment_offsets_clamped_checked = _make_segment_offsets_clamped(False, True)
_segment_offsets_clamped_fused_checked = _make_segment_offsets_clamped(True, True)


@wp.func
def _segment_length_low_limb(value: int):
    """Return the low 16 bits of a nonnegative segment length."""
    return wp.max(value, 0) & 65535


@wp.func
def _segment_length_high_limb(value: int):
    """Return the high 16 bits of a nonnegative segment length."""
    return wp.max(value, 0) >> 16


@wp.func
def _checked_length_low_limb(value: wp.int64):
    """Extract the low limb of a checked count, including the overflow sentinel."""
    return int(value & wp.int64(65535))


@wp.func
def _checked_length_high_limb(value: wp.int64):
    """Extract the high limb without narrowing capacity plus one to int32."""
    return int(value >> wp.int64(16))


@wp.func
def _segment_clamped_scan(low: int, high: int, capacity: int):
    """Reconstruct an exact wide prefix before narrowing its bounded value."""
    prefix = wp.int64(high) * wp.int64(65536) + wp.int64(low)
    return int(wp.min(prefix, wp.int64(capacity)))


@wp.func
def _segment_scanned_length(low: int, high: int, length: int, capacity: int):
    """Recover a clipped segment length from its inclusive prefix and input."""
    prefix = wp.int64(high) * wp.int64(65536) + wp.int64(low)
    previous = prefix - wp.int64(wp.max(length, 0))
    return int(
        wp.min(prefix, wp.int64(capacity)) - wp.min(previous, wp.int64(capacity))
    )


@wp.func
def _checked_scanned_length(low: int, high: int, length: wp.int64, capacity: int):
    """Recover bounded ranges from the normalized counts used by the scan."""
    prefix = wp.int64(high) * wp.int64(65536) + wp.int64(low)
    previous = prefix - length
    return int(
        wp.min(prefix, wp.int64(capacity)) - wp.min(previous, wp.int64(capacity))
    )


@wp.func
def _segment_work_count(length: int, tile_rows: int, skip_first: int):
    """Count work tiles without overflowing a rounded-up row numerator."""
    count = length / tile_rows + int(length % tile_rows != 0)
    if skip_first != 0:
        count = wp.max(count - 1, 0)
    return count


def _make_segment_offsets_cooperative(bucket: int, validate: bool = False) -> wp.Kernel:
    """Specialize one-block fused prefixes for a power-of-two segment bucket."""

    @wp.kernel(enable_backward=False)
    def _segment_offsets_cooperative_kernel(
        seglen: wp.array(dtype=Any),
        capacity: int,
        tile_rows: int,
        skip_first_tile: bool,
        row_offsets: wp.array(dtype=wp.int32),
        tile_offsets: wp.array(dtype=wp.int32),
        chunk_rows: int,
        chunk_offsets: wp.array(dtype=wp.int32),
        minimum_chunk_rows: int,
        validation: wp.array(dtype=wp.int32),
    ):
        """Build all three bounded prefixes with four cooperative int32 scans.

        Parameters
        ----------
        seglen : wp.array, shape (R,), dtype int32 or int64
            Segment lengths, with R <= bucket. Unchecked calls treat negative
            values as zero.
        capacity : int
            Nonnegative upper bound for the cumulative active row count.
        tile_rows : int
            Positive row count per work tile.
        skip_first_tile : bool
            Whether the tile prefix excludes each segment's first tile.
        row_offsets : wp.array, shape (R + 1,), dtype int32
            OUTPUT: Capacity-bounded cumulative rows with a leading zero.
        tile_offsets : wp.array, shape (R + 1,), dtype int32
            OUTPUT: Cumulative work-tile counts with a leading zero.
        chunk_rows : int
            Positive static chunk-size ceiling for the additional-chunk prefix.
        chunk_offsets : wp.array, shape (R + 1,), dtype int32
            OUTPUT: Cumulative chunks beyond each segment's first, with a
            leading zero.
        minimum_chunk_rows : int
            Positive lower bound for adapting chunks to the active row total,
            or zero to retain the static chunk size.
        validation : wp.array, shape (1,), dtype int32, or None
            OUTPUT: One for valid lengths, zero otherwise, in the checked
            specialization. Unused in the unchecked specialization.

        Notes
        -----
        At bucket <= 1024, sums of the low and high 16-bit limbs fit int32
        independently. Wide reconstruction and capacity clamping happen inside
        ``wp.tile_map`` before the bounded work counts are scanned.

        Thread launch: one tiled block with ``block_dim=min(bucket, 128)``.
        Modifies: every element of ``row_offsets``, ``tile_offsets``, and
        ``chunk_offsets``.
        The checked specialization also overwrites ``validation[0]``.
        """
        _, lane = wp.tid()
        raw_lengths = wp.tile_load(seglen, shape=(bucket,), offset=(0,))
        if wp.static(validate):
            lengths = wp.tile_map(_checked_segment_length, raw_lengths, capacity)
            low = wp.tile_scan_inclusive(wp.tile_map(_checked_length_low_limb, lengths))
            high = wp.tile_scan_inclusive(
                wp.tile_map(_checked_length_high_limb, lengths)
            )
            clipped = wp.tile_map(_checked_scanned_length, low, high, lengths, capacity)
            total_low = wp.tile_extract(low, bucket - 1)
            total_high = wp.tile_extract(high, bucket - 1)
            if lane == 0:
                total = wp.int64(total_high) * wp.int64(65536) + wp.int64(total_low)
                validation[0] = int(total <= wp.int64(capacity))
        else:
            lengths = wp.tile_map(_bounded_segment_length, raw_lengths)
            low = wp.tile_scan_inclusive(wp.tile_map(_segment_length_low_limb, lengths))
            high = wp.tile_scan_inclusive(
                wp.tile_map(_segment_length_high_limb, lengths)
            )
            clipped = wp.tile_map(_segment_scanned_length, low, high, lengths, capacity)
        rows = wp.tile_map(_segment_clamped_scan, low, high, capacity)
        if minimum_chunk_rows > 0:
            chunk_rows = _segment_chunk_rows(
                wp.tile_extract(rows, bucket - 1),
                (capacity - 1) / chunk_rows,
                minimum_chunk_rows,
            )
        tiles = wp.tile_scan_inclusive(
            wp.tile_map(_segment_work_count, clipped, tile_rows, int(skip_first_tile))
        )
        chunks = wp.tile_scan_inclusive(
            wp.tile_map(_segment_work_count, clipped, chunk_rows, 1)
        )
        wp.tile_store(row_offsets, rows, offset=(1,))
        wp.tile_store(tile_offsets, tiles, offset=(1,))
        wp.tile_store(chunk_offsets, chunks, offset=(1,))
        if lane == 0:
            row_offsets[0] = 0
            tile_offsets[0] = 0
            chunk_offsets[0] = 0

    _register_segment_offset_overloads(_segment_offsets_cooperative_kernel)
    return _segment_offsets_cooperative_kernel


_segment_offsets_cooperative = {
    bucket: _make_segment_offsets_cooperative(bucket)
    for bucket in (32, 64, 128, 256, 512, 1024)
}
_segment_offsets_cooperative_checked = {
    bucket: _make_segment_offsets_cooperative(bucket, True)
    for bucket in (32, 64, 128, 256, 512, 1024)
}


@wp.func
def _segment_offset_difference(previous: int, following: int):
    """Ignore negative differences from the zero-filled tail of a prefix tile."""
    return wp.max(following - previous, 0)


def _make_segment_chunk_offsets(bucket: int) -> wp.Kernel:
    """Specialize a chunk-prefix scan over precomputed bounded row offsets."""

    @wp.kernel(enable_backward=False)
    def _segment_chunk_offsets_kernel(
        row_offsets: wp.array(dtype=wp.int32),
        chunk_rows: int,
        extra_slots: int,
        chunk_offsets: wp.array(dtype=wp.int32),
    ):
        """Write an adaptive additional-chunk schedule into caller-owned storage.

        Parameters
        ----------
        row_offsets : wp.array, shape (R + 1,), dtype int32
            Nondecreasing, capacity-bounded row prefixes with a leading zero.
            R must be at most bucket.
        chunk_rows : int
            Positive static chunk-size ceiling; larger ceilings adapt to the
            active rows using the same rule as the weight-gradient kernel.
        extra_slots : int
            Nonnegative fixed workspace budget beyond the first chunk per segment.
        chunk_offsets : wp.array, shape (R + 1,), dtype int32
            OUTPUT: Cumulative additional-chunk counts with a leading zero.

        Notes
        -----
        Thread launch: one tiled block with ``block_dim=min(bucket, 128)``.
        Modifies: every ``chunk_offsets`` element, including the leading zero.
        """
        _, lane = wp.tid()
        groups = row_offsets.shape[0] - 1
        minimum_rows = int(256)
        if groups >= 32:
            minimum_rows = 512
        if chunk_rows > minimum_rows:
            chunk_rows = _segment_chunk_rows(
                row_offsets[groups], extra_slots, minimum_rows
            )
        previous = wp.tile_load(row_offsets, shape=(bucket,), offset=(0,))
        following = wp.tile_load(row_offsets, shape=(bucket,), offset=(1,))
        lengths = wp.tile_map(_segment_offset_difference, previous, following)
        chunks = wp.tile_scan_inclusive(
            wp.tile_map(_segment_work_count, lengths, chunk_rows, 1)
        )
        wp.tile_store(chunk_offsets, chunks, offset=(1,))
        if lane == 0:
            chunk_offsets[0] = 0

    return _segment_chunk_offsets_kernel


_segment_chunk_offsets = {
    bucket: _make_segment_chunk_offsets(bucket)
    for bucket in (32, 64, 128, 256, 512, 1024)
}


@wp.kernel(enable_backward=False)
def _zero_padding(
    row_offsets: wp.array(dtype=wp.int32),
    out: wp.array2d(dtype=Any),
):
    """Zero output rows outside the active segments without reading input padding.

    Parameters
    ----------
    row_offsets : wp.array, shape (R + 1,), dtype int32
        Cumulative active row counts; the last entry is the total active count.
    out : wp.array, shape (capacity, N), dtype float32 or bfloat16
        OUTPUT: Pre-allocated output buffer whose padding rows are zeroed.
        Active rows are unchanged.

    Notes
    -----
    Thread launch: ``(capacity, N)``, one thread per output element.
    Modifies: rows at or beyond the total active row count in ``out``.

    See Also
    --------
    _segment_offsets_clamped : Build bounded row offsets from wide prefixes.
    """
    row, column = wp.tid()
    if row >= row_offsets[row_offsets.shape[0] - 1]:
        out[row, column] = out.dtype(0)


_zero_padding_overloads = register_overloads(
    _zero_padding,
    lambda dtype: [wp.array(dtype=wp.int32), wp.array2d(dtype=dtype)],
    dtypes=(wp.float32, wp.bfloat16),
)

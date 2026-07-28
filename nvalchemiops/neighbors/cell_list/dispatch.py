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


"""Cell-list strategy-selection and pair-centric launch sizing."""

import os
from typing import Literal

__all__ = [
    "PAIR_CENTRIC_MAX_LINEAR_LAUNCH",
    "is_pair_centric_launch_safe",
    "is_pair_centric_parallelism_sufficient",
    "pair_centric_launch_size",
    "select_batch_cell_list_strategy",
    "select_cell_list_strategy",
    "compute_batch_pair_centric_n_outer",
]

PAIR_CENTRIC_MAX_LINEAR_LAUNCH = 2**31 - 1
"""Maximum safe one-dimensional Warp launch size for pair-centric kernels."""


def compute_batch_pair_centric_n_outer(
    R_max: tuple[int, int, int], half_fill: bool
) -> int:
    """Compute the number of non-self outer cell offsets at the given radius.

    Used to size the batched pair-centric launch grid.  Operates on the
    cross-system maximum ``R_max``; blocks targeting systems with
    smaller per-axis radii early-return when their decoded offset is
    out-of-range.

    Parameters
    ----------
    R_max : tuple[int, int, int]
        Cross-system maximum per-axis neighbor search radius.
    half_fill : bool
        If True, use the half-shell offset count; otherwise the full
        shell minus the self entry.

    Returns
    -------
    int
        Number of non-self outer cell offsets to enumerate.

    See Also
    --------
    batch_query_cell_list_pair_centric_sorted : Consumer that sizes its launch
        grid by ``(total_cells x (n_outer + 1))``.
    """
    Rx, Ry, Rz = int(R_max[0]), int(R_max[1]), int(R_max[2])
    if half_fill:
        return Rx * (2 * Ry + 1) * (2 * Rz + 1) + Ry * (2 * Rz + 1) + Rz
    return (2 * Rx + 1) * (2 * Ry + 1) * (2 * Rz + 1) - 1


def _pair_centric_logical_block_count(total_cells: int, n_outer: int) -> int:
    """Return the number of logical (source cell, offset) blocks.

    Parameters
    ----------
    total_cells : int
        Number of source cells in the launch.
    n_outer : int
        Number of non-self neighbor-cell offsets.

    Returns
    -------
    int
        Total logical block count ``total_cells * (n_outer + 1)``.
    """
    return int(total_cells) * (int(n_outer) + 1)


def _pair_centric_max_cuda_blocks(
    block_dim: int = 64,
    *,
    max_launch_size: int = PAIR_CENTRIC_MAX_LINEAR_LAUNCH,
) -> int:
    """Return the maximum CUDA blocks that fit in one Warp launch.

    Parameters
    ----------
    block_dim : int, default=64
        Threads per CUDA block.
    max_launch_size : int, default=PAIR_CENTRIC_MAX_LINEAR_LAUNCH
        Maximum allowed one-dimensional Warp launch size.

    Returns
    -------
    int
        Maximum CUDA blocks ``max_launch_size // block_dim``, at least 1.

    Raises
    ------
    ValueError
        If ``block_dim <= 0`` or ``max_launch_size < block_dim``.
    """
    block_dim_int = int(block_dim)
    max_launch_int = int(max_launch_size)
    if block_dim_int <= 0:
        raise ValueError(f"block_dim must be positive, got {block_dim_int}")
    if max_launch_int < block_dim_int:
        raise ValueError(
            f"max_launch_size ({max_launch_int}) must be >= block_dim ({block_dim_int})"
        )
    return max(1, max_launch_int // block_dim_int)


def _pair_centric_work_per_cuda_block(
    total_cells: int,
    n_outer: int,
    block_dim: int = 64,
    *,
    max_launch_size: int = PAIR_CENTRIC_MAX_LINEAR_LAUNCH,
) -> int:
    """Return how many logical blocks each CUDA block processes when coarsening.

    Parameters
    ----------
    total_cells : int
        Number of source cells in the launch.
    n_outer : int
        Number of non-self neighbor-cell offsets.
    block_dim : int, default=64
        Threads per CUDA block.
    max_launch_size : int, default=PAIR_CENTRIC_MAX_LINEAR_LAUNCH
        Maximum allowed one-dimensional Warp launch size.

    Returns
    -------
    int
        Logical blocks per CUDA block.  Returns 1 when the uncoarsened
        fast path fits within ``max_launch_size``.

    Raises
    ------
    ValueError
        If ``block_dim <= 0`` or ``max_launch_size < block_dim``.
    """
    logical_blocks = _pair_centric_logical_block_count(total_cells, n_outer)
    max_cuda_blocks = _pair_centric_max_cuda_blocks(
        block_dim,
        max_launch_size=max_launch_size,
    )
    return max(1, (logical_blocks + max_cuda_blocks - 1) // max_cuda_blocks)


def _pair_centric_coarsened_launch_size(
    total_cells: int,
    n_outer: int,
    block_dim: int = 64,
    *,
    max_launch_size: int = PAIR_CENTRIC_MAX_LINEAR_LAUNCH,
) -> int:
    """Return the one-dimensional Warp launch size after coarsening.

    Parameters
    ----------
    total_cells : int
        Number of source cells in the launch.
    n_outer : int
        Number of non-self neighbor-cell offsets.
    block_dim : int, default=64
        Threads per CUDA block.
    max_launch_size : int, default=PAIR_CENTRIC_MAX_LINEAR_LAUNCH
        Maximum allowed one-dimensional Warp launch size.

    Returns
    -------
    int
        Total launch dimension ``ceil(logical_blocks / work_per_cuda_block)
        * block_dim``.

    Raises
    ------
    ValueError
        If ``block_dim <= 0`` or ``max_launch_size < block_dim``.
    """
    logical_blocks = _pair_centric_logical_block_count(total_cells, n_outer)
    work_per_cuda_block = _pair_centric_work_per_cuda_block(
        total_cells,
        n_outer,
        block_dim,
        max_launch_size=max_launch_size,
    )
    cuda_blocks = (logical_blocks + work_per_cuda_block - 1) // work_per_cuda_block
    return cuda_blocks * int(block_dim)


def pair_centric_launch_size(
    total_cells: int,
    n_outer: int,
    block_dim: int = 64,
) -> int:
    """Return the uncoarsened linear launch size for a pair-centric query.

    This describes the fast-path launch shape (one logical block per CUDA
    block).  Oversized launches use
    :func:`_pair_centric_coarsened_launch_size` instead.

    Parameters
    ----------
    total_cells : int
        Number of source cells in the launch.
    n_outer : int
        Number of non-self neighbor-cell offsets.
    block_dim : int, default=64
        Threads per logical source-cell/offset block.

    Returns
    -------
    int
        Total one-dimensional Warp launch size for the uncoarsened fast path.
    """
    return int(total_cells) * (int(n_outer) + 1) * int(block_dim)


def is_pair_centric_launch_safe(
    total_cells: int,
    n_outer: int,
    block_dim: int = 64,
    *,
    max_launch_size: int = PAIR_CENTRIC_MAX_LINEAR_LAUNCH,
) -> bool:
    """Return whether the uncoarsened pair-centric fast path fits the launch limit.

    A ``False`` result does not mean pair-centric is infeasible; the
    launcher may transparently coarsen logical blocks into fewer CUDA
    blocks via :func:`_pair_centric_work_per_cuda_block`.

    Parameters
    ----------
    total_cells : int
        Number of source cells in the launch.
    n_outer : int
        Number of non-self neighbor-cell offsets.
    block_dim : int, default=64
        Threads per logical source-cell/offset block.
    max_launch_size : int, default=PAIR_CENTRIC_MAX_LINEAR_LAUNCH
        Maximum allowed one-dimensional launch size.

    Returns
    -------
    bool
        ``True`` if the uncoarsened fast-path launch size is safe.
    """
    return pair_centric_launch_size(total_cells, n_outer, block_dim) <= int(
        max_launch_size
    )


def is_pair_centric_parallelism_sufficient(
    total_atoms: int,
    total_cells: int,
    n_outer: int,
    block_dim: int = 256,
) -> bool:
    """Return whether pair-centric exposes at least atom-kernel block count.

    Parameters
    ----------
    total_atoms : int
        Number of central atoms in the query.
    total_cells : int
        Number of source cells in the pair-centric launch.
    n_outer : int
        Number of non-self neighbor-cell offsets.
    block_dim : int, default=256
        Threads per atom-centric CUDA block.  The atom-centric query
        kernel is launched with ``wp.launch(..., dim=natom)`` and no
        explicit ``block_dim``, so it runs at Warp's default block size
        (256); ``atom_blocks`` must use the same divisor for the
        like-for-like block-count comparison below.

    Returns
    -------
    bool
        True when the pair-centric grid has at least as many logical
        blocks as an atom-centric one-thread-per-atom launch.
    """
    block_dim_int = max(int(block_dim), 1)
    atom_blocks = max(1, (int(total_atoms) + block_dim_int - 1) // block_dim_int)
    pair_blocks = int(total_cells) * (int(n_outer) + 1)
    return pair_blocks >= atom_blocks


def select_cell_list_strategy(
    natom: int, cutoff: float
) -> Literal["atom_centric", "pair_centric"]:
    """Select ``"atom_centric"`` or ``"pair_centric"`` for the given (N, cutoff).

    Sync-free: takes Python ints / floats, no GPU reads.  Note this applies to
    the strategy *decision* only; once ``"pair_centric"`` is chosen, the Torch
    launcher materializes launch metadata (``n_outer`` per axis, and for the
    batched path ``R_max`` / ``total_cells``) via scalar tensor readbacks, which
    do synchronize to host.  ``"atom_centric"`` avoids those reads.

    Pair-centric wins iff any of:
      1. ``cutoff >= 8  AND N <= 65536``
      2. ``cutoff >= 6  AND N <=  8192``
      3. ``cutoff >= 4  AND N <=  1024``

    To pin a strategy, pass it explicitly (``cell_list(..., strategy=...)`` or a
    fine-grained ``method=`` name) rather than via an environment variable.
    """
    n = int(natom)
    c = float(cutoff)
    if (
        (c >= 8.0 and n <= 65536)
        or (c >= 6.0 and n <= 8192)
        or (c >= 4.0 and n <= 1024)
    ):
        return "pair_centric"
    return "atom_centric"


# Three-clause rule (see ``select_batch_cell_list_strategy``).  Every
# pair-centric clause is gated by avg_aps >= 4096 so high-cutoff
# many-tiny-system batches stay atom-centric.
#   1. cutoff >= 8 - high-cutoff, reasonably-sized systems.
#   2. cutoff >= 6 AND total_atoms <= 65_536 - small-/medium-N MLIP
#      regime with reasonably-sized systems.
#   3. cutoff >= 6 AND num_systems <= 8 - few-large-systems regime,
#      where cell-level parallelism dominates atomic contention.
_BATCH_PAIR_STRATEGY_DEFAULTS = {
    "NVALCHEMI_NEIGHLIST_BATCH_PAIR_CUTOFF_FLOOR": 8.0,
    "NVALCHEMI_NEIGHLIST_BATCH_PAIR_TOTAL_CAP": 65_536,
    "NVALCHEMI_NEIGHLIST_BATCH_PAIR_AVG_APS_FLOOR": 4096,
    "NVALCHEMI_NEIGHLIST_BATCH_PAIR_NSYS_CAP": 8,
}


def select_batch_cell_list_strategy(
    total_atoms: int,
    num_systems: int,
    cutoff: float,
) -> Literal["atom_centric", "pair_centric"]:
    """Select ``"atom_centric"`` or ``"pair_centric"`` for the batch path.

    Selects pair-centric when ``avg_atoms_per_system`` is at least
    ``NVALCHEMI_NEIGHLIST_BATCH_PAIR_AVG_APS_FLOOR`` (default 4096) and
    *any* of the three clauses holds:

    1. ``cutoff >= NVALCHEMI_NEIGHLIST_BATCH_PAIR_CUTOFF_FLOOR`` (default 8.0).
    2. ``cutoff >= 6`` AND
       ``total_atoms <= NVALCHEMI_NEIGHLIST_BATCH_PAIR_TOTAL_CAP`` (default 65_536).
    3. ``cutoff >= 6`` AND
       ``num_systems <= NVALCHEMI_NEIGHLIST_BATCH_PAIR_NSYS_CAP`` (default 8)
       AND ``total_atoms > TOTAL_CAP`` (few-large-systems regime).

    To pin a strategy, pass it explicitly rather than via an environment
    variable.
    """

    def _i(name: str) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return int(_BATCH_PAIR_STRATEGY_DEFAULTS[name])
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return int(_BATCH_PAIR_STRATEGY_DEFAULTS[name])

    def _f(name: str) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return float(_BATCH_PAIR_STRATEGY_DEFAULTS[name])
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(_BATCH_PAIR_STRATEGY_DEFAULTS[name])

    cutoff_floor = _f("NVALCHEMI_NEIGHLIST_BATCH_PAIR_CUTOFF_FLOOR")
    total_cap = _i("NVALCHEMI_NEIGHLIST_BATCH_PAIR_TOTAL_CAP")
    avg_aps_floor = _i("NVALCHEMI_NEIGHLIST_BATCH_PAIR_AVG_APS_FLOOR")
    nsys_cap = _i("NVALCHEMI_NEIGHLIST_BATCH_PAIR_NSYS_CAP")

    avg_aps = total_atoms // max(num_systems, 1)
    if avg_aps < avg_aps_floor:
        return "atom_centric"
    if cutoff >= cutoff_floor:
        return "pair_centric"
    if cutoff >= 6.0 and total_atoms <= total_cap:
        return "pair_centric"
    # Few-large-systems clause: require total_atoms above the same cap so
    # the per-call setup overhead (gather + cell_to_system map + R_max
    # .item()) doesn't dominate for tiny batches.
    if cutoff >= 6.0 and num_systems <= nsys_cap and total_atoms > total_cap:
        return "pair_centric"
    return "atom_centric"

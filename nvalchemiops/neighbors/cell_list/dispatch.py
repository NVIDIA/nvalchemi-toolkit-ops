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


def pair_centric_launch_size(
    total_cells: int,
    n_outer: int,
    block_dim: int = 64,
) -> int:
    """Return the linear launch size for a pair-centric cell-list query.

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
        Total one-dimensional Warp launch size.
    """
    return int(total_cells) * (int(n_outer) + 1) * int(block_dim)


def is_pair_centric_launch_safe(
    total_cells: int,
    n_outer: int,
    block_dim: int = 64,
    *,
    max_launch_size: int = PAIR_CENTRIC_MAX_LINEAR_LAUNCH,
) -> bool:
    """Return whether a pair-centric launch fits the safe linear launch limit.

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
        ``True`` if the pair-centric launch size is safe.
    """
    return pair_centric_launch_size(total_cells, n_outer, block_dim) <= int(
        max_launch_size
    )


# Strategy envar: "auto" (default), "atom_centric", or "pair_centric".
_STRATEGY_ENV = "NVALCHEMI_NEIGHLIST_STRATEGY"


def select_cell_list_strategy(
    natom: int, cutoff: float
) -> Literal["atom_centric", "pair_centric"]:
    """Select ``"atom_centric"`` or ``"pair_centric"`` for the given (N, cutoff).

    Sync-free: takes Python ints / floats, no GPU reads.

    Pair-centric wins iff any of:
      1. ``cutoff >= 8  AND N <= 65536``
      2. ``cutoff >= 6  AND N <=  8192``
      3. ``cutoff >= 4  AND N <=  1024``

    Override via env var ``NVALCHEMI_NEIGHLIST_STRATEGY`` =
    ``"auto"`` (default) / ``"atom_centric"`` / ``"pair_centric"``.

    Calibrated on GB10 sm_121.  Cross-GPU sensitivity is real but
    bounded (<= ~35 % wallclock penalty per cell, <= ~3 % mean) - recalibrate
    by sweeping ``benchmark_neighborlist.py --methods
    cell_list_atom_centric cell_list_pair_centric`` and editing the rule
    above, or pin a single variant via the env var.
    """
    override = os.environ.get(_STRATEGY_ENV, "auto").strip()
    if override in ("atom_centric", "pair_centric"):
        return override
    n = int(natom)
    c = float(cutoff)
    if (
        (c >= 8.0 and n <= 65536)
        or (c >= 6.0 and n <= 8192)
        or (c >= 4.0 and n <= 1024)
    ):
        return "pair_centric"
    return "atom_centric"


# Three-clause rule (see ``select_batch_cell_list_strategy``):
#   1. cutoff >= 8 - pair-centric dominates almost everywhere here.
#   2. cutoff >= 6 AND total_atoms <= 65_536 AND avg_aps >= 4096 -
#      small-/medium-N MLIP regime with reasonably-sized systems.  The
#      avg_aps floor avoids picking pair-centric for many-tiny-systems
#      shapes where its setup overhead dominates the kernel speedup.
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

    Selects pair-centric when *any* of the three clauses holds:

    1. ``cutoff >= NVALCHEMI_NEIGHLIST_BATCH_PAIR_CUTOFF_FLOOR`` (default 8.0).
    2. ``cutoff >= 6`` AND
       ``total_atoms <= NVALCHEMI_NEIGHLIST_BATCH_PAIR_TOTAL_CAP`` (default 65_536)
       AND ``avg_atoms_per_system >= NVALCHEMI_NEIGHLIST_BATCH_PAIR_AVG_APS_FLOOR``
       (default 4096).
    3. ``cutoff >= 6`` AND
       ``num_systems <= NVALCHEMI_NEIGHLIST_BATCH_PAIR_NSYS_CAP`` (default 8)
       AND ``total_atoms > TOTAL_CAP`` (few-large-systems regime).

    Override via env var ``NVALCHEMI_NEIGHLIST_STRATEGY`` =
    ``"auto"`` (default) / ``"atom_centric"`` / ``"pair_centric"``.

    Calibrated on Blackwell GB10.  The thresholds are env-tunable; the
    cost of picking wrong is bounded (<= ~20 % mean wallclock penalty on
    cross-GPU measurements).  Recalibrate by sweeping
    ``benchmark_neighborlist.py --methods batch_cell_list_atom_centric
    batch_cell_list_pair_centric`` and resetting the env vars above.
    """
    override = os.environ.get(_STRATEGY_ENV, "auto").strip()
    if override in ("atom_centric", "pair_centric"):
        return override

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

    if cutoff >= cutoff_floor:
        return "pair_centric"
    avg_aps = total_atoms // max(num_systems, 1)
    if cutoff >= 6.0 and total_atoms <= total_cap and avg_aps >= avg_aps_floor:
        return "pair_centric"
    # Few-large-systems clause: require total_atoms above the same cap so
    # the per-call setup overhead (gather + cell_to_system map + R_max
    # .item()) doesn't dominate for tiny batches.
    if cutoff >= 6.0 and num_systems <= nsys_cap and total_atoms > total_cap:
        return "pair_centric"
    return "atom_centric"

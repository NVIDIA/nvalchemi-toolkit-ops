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


"""Shared helpers for frontend neighbor-list auto dispatch."""

import os
from collections.abc import Iterable
from functools import lru_cache
from typing import Literal

import warp as wp

from nvalchemiops.neighbors.neighbor_utils import DTYPE_INFO_ALL, empty_sentinel

__all__ = [
    "AUTO_BASE_DEFAULTS",
    "DEFAULT_BATCH_MAX_NBINS",
    "DEFAULT_SINGLE_MAX_NBINS",
    "FEATURE_BATCHED",
    "FEATURE_CUDA",
    "FEATURE_POSITIONS_FLOAT32",
    "NEIGHBOR_LIST_STRATEGIES",
    "auto_base_constants",
    "finalize_neighbor_list_method",
    "get_select_neighbor_list_method_cost_kernel",
    "neighbor_list_strategy_run_args",
    "optional_outputs_mask",
    "estimate_neighbor_list_costs",
    "suggest_neighbor_list_method",
]

NeighborListMethod = Literal["naive", "cell_list", "cluster_tile"]
CellListStrategy = Literal["atom_centric", "pair_centric"]
AtomCentricPath = Literal["direct", "sorted"]
NaiveStrategy = Literal["scalar", "tile"]

# Cost-model constants (env-overridable, NVALCHEMI_NEIGHLIST_* style).
#
# The selector compares three families using quantities computed from the same
# geometry kernel:
#   naive       ~ per-system direct candidate work, image count, output work
#   cell_list   ~ cell-grid candidate work plus expected output work
#   cluster_tile ~ fully-periodic tile output work when semantically compatible
#
# These are intentionally coarse: the selector is a routing guard, not a timing
# predictor.  It uses volume and expected pairs so density/cutoff regimes drive
# the decision rather than raw atom-count thresholds.
AUTO_BASE_DEFAULTS = {
    "NVALCHEMI_NEIGHLIST_CELL_SHELL": 27.0,
    "NVALCHEMI_NEIGHLIST_CELL_SETUP": 4096.0,
}

DEFAULT_SINGLE_MAX_NBINS = 524288
"""Default single-system cell-list cell cap."""

DEFAULT_BATCH_MAX_NBINS = 8192
"""Default per-system batched cell-list cell cap."""

FEATURE_CUDA = 1 << 0
"""Selector feature flag: the active device supports CUDA kernels."""

FEATURE_POSITIONS_FLOAT32 = 1 << 1
"""Selector feature flag: positions are float32, required by cluster-tile auto."""

FEATURE_BATCHED = 1 << 2
"""Selector feature flag: the call routes through a batched frontend."""

_MAX_WARP_LINEAR_LAUNCH = 2**31 - 1
_EPS = 1.0e-30
_INF = 3.0e30
_SPHERE_VOLUME_FACTOR = 4.1887902047863905

_OPTION_CUTOFF2 = 1 << 0
_OPTION_HALF_FILL = 1 << 1
_OPTION_RETURN_NEIGHBOR_LIST = 1 << 2
_OPTION_TARGET_INDICES = 1 << 3
_OPTION_RETURN_VECTORS = 1 << 4
_OPTION_RETURN_DISTANCES = 1 << 5
_OPTION_USE_PAIR_FN = 1 << 6
_OPTION_REBUILD_FLAGS = 1 << 8
_OPTION_WRAP_POSITIONS_FALSE = 1 << 9

_OPTION_BITS = {
    "cutoff2": _OPTION_CUTOFF2,
    "half_fill": _OPTION_HALF_FILL,
    "return_neighbor_list": _OPTION_RETURN_NEIGHBOR_LIST,
    "target_indices": _OPTION_TARGET_INDICES,
    "return_vectors": _OPTION_RETURN_VECTORS,
    "return_distances": _OPTION_RETURN_DISTANCES,
    "use_pair_fn": _OPTION_USE_PAIR_FN,
    "rebuild_flags": _OPTION_REBUILD_FLAGS,
    "wrap_positions_false": _OPTION_WRAP_POSITIONS_FALSE,
}

_OPTION_ALIASES = {
    "compute_vectors": "return_vectors",
    "vectors": "return_vectors",
    "neighbor_vectors": "return_vectors",
    "compute_distances": "return_distances",
    "distances": "return_distances",
    "neighbor_distances": "return_distances",
    "pair_fn": "use_pair_fn",
    "pair_params": "use_pair_fn",
    "pair_energies": "use_pair_fn",
    "pair_forces": "use_pair_fn",
    "wrap_positions": "wrap_positions_false",
}

_FLAG_NAMES = (
    "invalid_input",
    "naive_unsafe",
    "cell_list_unsafe",
    "cluster_tile_unsupported_options",
    "cluster_tile_unsupported_geometry",
    "cluster_tile_noncontiguous_batch",
    "cluster_tile_image_multiplicity",
    "pair_centric_unsafe",
    "naive_tile_unsafe",
)


# Fine-grained, directly-runnable neighbor-list strategies.  These are the names
# returned by :func:`estimate_neighbor_list_costs` / :func:`suggest_neighbor_list_method`
# and accepted by the public ``neighbor_list(method=...)`` frontends; each maps
# 1:1 to a base method plus its sub-options.
NEIGHBOR_LIST_STRATEGIES = (
    "naive_scalar",
    "naive_tile",
    "cell_list_atom_centric",
    "cell_list_pair_centric",
    "cluster_tile",
)

# name -> (base_method, native_strategy, cell_list_strategy, atom_centric_path).
_STRATEGY_RUN_ARGS: dict[str, tuple[str, str, str, str]] = {
    "naive_scalar": ("naive", "scalar", "auto", "auto"),
    "naive_tile": ("naive", "tile", "auto", "auto"),
    "cell_list_atom_centric": ("cell_list", "auto", "atom_centric", "direct"),
    "cell_list_pair_centric": ("cell_list", "auto", "pair_centric", "sorted"),
    "cluster_tile": ("cluster_tile", "auto", "auto", "auto"),
}


def neighbor_list_strategy_run_args(strategy: str) -> tuple[str, str, str, str]:
    """Resolve a strategy name to its run arguments.

    Parameters
    ----------
    strategy : str
        A name from :data:`NEIGHBOR_LIST_STRATEGIES`, optionally ``batch_``
        prefixed.

    Returns
    -------
    tuple of str
        ``(method, native_strategy, cell_list_strategy, atom_centric_path)``.
        The ``batch_`` prefix, if present, is carried onto ``method``.
    """
    batched = strategy.startswith("batch_")
    base = strategy[len("batch_") :] if batched else strategy
    if base not in _STRATEGY_RUN_ARGS:
        raise ValueError(f"unknown neighbor-list strategy {strategy!r}")
    method, native_strategy, cell_list_strategy, atom_centric_path = _STRATEGY_RUN_ARGS[
        base
    ]
    if batched:
        method = "batch_" + method
    return method, native_strategy, cell_list_strategy, atom_centric_path


def _auto_env_float(name: str) -> float:
    raw = os.environ.get(name)
    try:
        return float(raw) if raw is not None else float(AUTO_BASE_DEFAULTS[name])
    except (TypeError, ValueError):
        return float(AUTO_BASE_DEFAULTS[name])


def auto_base_constants() -> tuple[float, float]:
    """Return the current auto-dispatch constants.

    Returns
    -------
    shell : float
        Multiplicative estimate of cell-list neighbor-shell work.
    setup : float
        Per-system setup floor for cell-list work.

    Notes
    -----
    The values are read from ``NVALCHEMI_NEIGHLIST_CELL_SHELL`` and
    ``NVALCHEMI_NEIGHLIST_CELL_SETUP`` when present, otherwise from
    :data:`AUTO_BASE_DEFAULTS`.
    """
    return (
        _auto_env_float("NVALCHEMI_NEIGHLIST_CELL_SHELL"),
        _auto_env_float("NVALCHEMI_NEIGHLIST_CELL_SETUP"),
    )


def _wp_scalar_from_cell_dtype(cell_dtype: type) -> type:
    """Return the Warp scalar dtype for a mat33 cell dtype."""
    for scalar_dtype, (_, mat_dtype) in DTYPE_INFO_ALL.items():
        if cell_dtype == mat_dtype:
            return scalar_dtype
    raise ValueError(f"Unsupported cell dtype: {cell_dtype!r}")


def _safe_int(value) -> int:
    """Convert host scalar-like values to ``int``."""
    return int(value.item() if hasattr(value, "item") else value)


def _safe_float(value) -> float:
    """Convert host scalar-like values to ``float``."""
    return float(value.item() if hasattr(value, "item") else value)


def _option_is_active(value: object) -> bool:
    """Return whether a public-style selector option should set a bit."""
    if value is None or value is False:
        return False
    return True


def optional_outputs_mask(
    optional_outputs: Iterable[str] | None = None,
    **public_kwargs: object,
) -> int:
    """Encode public neighbor-list option names into a selector bitmask.

    Parameters
    ----------
    optional_outputs : iterable of str, optional
        Public-style option names.  Supported names include ``"cutoff2"``,
        ``"half_fill"``, ``"return_neighbor_list"``, ``"target_indices"``,
        ``"return_vectors"``, ``"return_distances"``, ``"use_pair_fn"``, and
        ``"rebuild_flags"``.  Aliases matching common public buffers such as
        ``"neighbor_vectors"`` and ``"pair_fn"`` are accepted.
    **public_kwargs
        Public neighbor-list keyword names mapped to their values.  Truthy or
        non-``None`` values set the corresponding feasibility bit.  For
        ``wrap_positions``, a value of ``False`` sets
        ``"wrap_positions_false"``.

    Returns
    -------
    int
        Bitmask consumed by the Warp selector kernel.
    """
    mask = 0
    for name in optional_outputs or ():
        canonical = _OPTION_ALIASES.get(str(name), str(name))
        try:
            mask |= _OPTION_BITS[canonical]
        except KeyError as exc:
            raise ValueError(f"Unknown optional output option: {name!r}") from exc

    for name, value in public_kwargs.items():
        canonical = _OPTION_ALIASES.get(name, name)
        if canonical == "wrap_positions_false":
            if value is False:
                mask |= _OPTION_BITS[canonical]
            continue
        if canonical not in _OPTION_BITS:
            raise ValueError(f"Unknown optional output option: {name!r}")
        if _option_is_active(value):
            mask |= _OPTION_BITS[canonical]
    return int(mask)


def finalize_neighbor_list_method(costs, flags) -> list[tuple[str, float]]:
    """Reduce selector costs and flags to feasible strategies, cheapest-first.

    Parameters
    ----------
    costs : array-like, shape (5,)
        ``(naive_scalar, naive_tile, cell_atom, cell_pair, cluster_tile)``.
    flags : array-like, shape (9,)
        Invalid-input and per-strategy feasibility flags.

    Returns
    -------
    list of (str, float)
        Feasible strategy names (from :data:`NEIGHBOR_LIST_STRATEGIES`) paired
        with their relative estimated cost (lower is faster), sorted ascending.

    Raises
    ------
    ValueError
        If the selector reported invalid input.
    RuntimeError
        If no strategy satisfies the safety and feasibility guards.
    """
    raw_flags = [_safe_int(flags[i]) for i in range(min(len(flags), len(_FLAG_NAMES)))]
    # Pad to the full flag set so a short ``flags`` argument cannot IndexError on
    # the fixed indices used below; a missing flag defaults to 0 (feasible).
    while len(raw_flags) < len(_FLAG_NAMES):
        raw_flags.append(0)
    if raw_flags[0] != 0:
        raise ValueError("invalid batch_ptr, cutoff, or max_nbins for auto dispatch")

    raw_costs = [_safe_float(costs[i]) for i in range(min(len(costs), 5))]
    while len(raw_costs) < 5:
        raw_costs.append(_INF)

    naive_ok = raw_flags[1] == 0
    cell_ok = raw_flags[2] == 0
    cluster_ok = (
        raw_flags[3] == 0
        and raw_flags[4] == 0
        and raw_flags[5] == 0
        and raw_flags[6] == 0
    )

    candidates: list[tuple[str, float]] = []
    if naive_ok:
        candidates.append(("naive_scalar", raw_costs[0]))
        if raw_flags[8] == 0:
            candidates.append(("naive_tile", raw_costs[1]))
    if cell_ok:
        candidates.append(("cell_list_atom_centric", raw_costs[2]))
        if raw_flags[7] == 0:
            candidates.append(("cell_list_pair_centric", raw_costs[3]))
    if cluster_ok:
        candidates.append(("cluster_tile", raw_costs[4]))

    feasible = [(name, cost) for name, cost in candidates if cost < _INF * 0.5]
    if not feasible:
        raise RuntimeError(
            "No neighbor-list method fits the current safety and feasibility guards"
        )
    feasible.sort(key=lambda item: item[1])
    return feasible


@lru_cache(maxsize=None)
def get_select_neighbor_list_method_cost_kernel(wp_dtype: type) -> wp.Kernel:
    """Return the shared guarded auto-dispatch cost kernel.

    Parameters
    ----------
    wp_dtype : type
        Warp scalar dtype for cell matrices, usually ``wp.float32`` or
        ``wp.float64``.

    Returns
    -------
    wp.Kernel
        Kernel computing method costs, sub-option costs, and feasibility flags.
    """
    if wp_dtype not in DTYPE_INFO_ALL:
        raise ValueError(f"Unsupported dtype: {wp_dtype!r}")
    _, mat_dtype = DTYPE_INFO_ALL[wp_dtype]

    @wp.kernel(enable_backward=False)
    def _kernel(
        batch_ptr: wp.array(dtype=wp.int32),
        batch_idx: wp.array(dtype=wp.int32),
        batch_idx_is_provided: bool,
        cell: wp.array(dtype=mat_dtype),
        pbc_single: wp.array(dtype=wp.bool),
        pbc_batch: wp.array2d(dtype=wp.bool),
        pbc_is_batched: bool,
        cutoff: wp.float32,
        shell: wp.float32,
        setup: wp.float32,
        max_nbins: wp.int32,
        max_launch_size: wp.int64,
        option_mask: wp.int32,
        feature_mask: wp.int32,
        target_count: wp.int32,
        costs: wp.array(dtype=wp.float32),
        flags: wp.array(dtype=wp.int32),
    ) -> None:
        """Compute dispatch costs, sub-option costs, and safety flags.

        Parameters
        ----------
        batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
            Cumulative per-system atom counts.
        batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
            Optional dense system ids.  Zero-size sentinel when absent.
        batch_idx_is_provided : bool
            Whether ``batch_idx`` should be validated for contiguous layout.
        cell : wp.array, shape (num_systems,), dtype=wp.mat33*
            Per-system cell matrices.
        pbc_single : wp.array, shape (3,), dtype=wp.bool
            Shared PBC flags. Zero-size sentinel when ``pbc_is_batched=True``.
        pbc_batch : wp.array, shape (num_systems, 3), dtype=wp.bool
            Per-system PBC flags. Zero-size sentinel when
            ``pbc_is_batched=False``.
        pbc_is_batched : bool
            Selects ``pbc_batch`` vs ``pbc_single``.
        cutoff : float
            Neighbor-list cutoff.
        shell : float
            Cell-list shell work multiplier.
        setup : float
            Cell-list per-system setup cost.
        max_nbins : int
            Cell-list per-system cap used by the selected frontend.
        max_launch_size : int
            Conservative one-dimensional launch-size guard.
        option_mask : int
            Public-option feasibility bits from :func:`optional_outputs_mask`.
        feature_mask : int
            Device/dtype/frontend feature bits.
        target_count : int
            Number of targeted source rows when ``target_indices`` is active.
            Zero means "all atoms".
        costs : wp.array, shape (5,), dtype=wp.float32
            OUTPUT: ``(naive_scalar, naive_tile, cell_atom, cell_pair,
            cluster_tile)``.
        flags : wp.array, shape (9,), dtype=wp.int32
            OUTPUT: invalid input and method/sub-option feasibility flags.

        Returns
        -------
        None
            The kernel has no return value; results are written in place to the
            ``costs`` and ``flags`` output arrays.

        Notes
        -----
        - Thread launch: ``max(num_systems, total_atoms when batch_idx is
          supplied)``.
        - Modifies: ``costs`` and ``flags`` by atomic updates.

        See Also
        --------
        finalize_neighbor_list_method : Reduce these costs/flags to a method.
        optional_outputs_mask : Builds the ``option_mask`` consumed here.
        """
        tid = wp.tid()
        num_systems = wp.int32(cell.shape[0])

        has_cuda = (feature_mask & wp.int32(FEATURE_CUDA)) != wp.int32(0)
        is_float32 = (feature_mask & wp.int32(FEATURE_POSITIONS_FLOAT32)) != wp.int32(0)
        is_batched = (feature_mask & wp.int32(FEATURE_BATCHED)) != wp.int32(0)

        has_cutoff2 = (option_mask & wp.int32(_OPTION_CUTOFF2)) != wp.int32(0)
        half_fill = (option_mask & wp.int32(_OPTION_HALF_FILL)) != wp.int32(0)
        return_list = (
            option_mask & wp.int32(_OPTION_RETURN_NEIGHBOR_LIST)
        ) != wp.int32(0)
        has_target_indices = (
            option_mask & wp.int32(_OPTION_TARGET_INDICES)
        ) != wp.int32(0)
        has_vectors = (option_mask & wp.int32(_OPTION_RETURN_VECTORS)) != wp.int32(0)
        has_distances = (option_mask & wp.int32(_OPTION_RETURN_DISTANCES)) != wp.int32(
            0
        )
        use_pair_fn = (option_mask & wp.int32(_OPTION_USE_PAIR_FN)) != wp.int32(0)
        has_rebuild_flags = (option_mask & wp.int32(_OPTION_REBUILD_FLAGS)) != wp.int32(
            0
        )
        wrap_positions_false = (
            option_mask & wp.int32(_OPTION_WRAP_POSITIONS_FALSE)
        ) != wp.int32(0)
        has_pair_outputs = (
            has_target_indices or has_vectors or has_distances or use_pair_fn
        )

        total_atoms = batch_ptr[num_systems]
        if tid == 0:
            if cutoff <= wp.float32(0.0) or max_nbins <= wp.int32(0):
                wp.atomic_max(flags, 0, wp.int32(1))
            if batch_ptr[0] != wp.int32(0) or total_atoms < wp.int32(0):
                wp.atomic_max(flags, 0, wp.int32(1))
            if has_cutoff2:
                wp.atomic_max(flags, 2, wp.int32(1))

        if batch_idx_is_provided and tid < total_atoms:
            atom_system = batch_idx[tid]
            if atom_system < wp.int32(0) or atom_system >= num_systems:
                wp.atomic_max(flags, 0, wp.int32(1))
                wp.atomic_max(flags, 5, wp.int32(1))
            else:
                if tid < batch_ptr[atom_system] or tid >= batch_ptr[atom_system + 1]:
                    wp.atomic_max(flags, 5, wp.int32(1))

        if tid >= num_systems:
            return

        start = batch_ptr[tid]
        end = batch_ptr[tid + 1]
        n_atoms = end - start
        if n_atoms < wp.int32(0):
            wp.atomic_max(flags, 0, wp.int32(1))
            n_atoms = wp.int32(0)

        cell_i = cell[tid]
        cell_volume = (
            cell_i[0, 0] * (cell_i[1, 1] * cell_i[2, 2] - cell_i[1, 2] * cell_i[2, 1])
            - cell_i[0, 1] * (cell_i[1, 0] * cell_i[2, 2] - cell_i[1, 2] * cell_i[2, 0])
            + cell_i[0, 2] * (cell_i[1, 0] * cell_i[2, 1] - cell_i[1, 1] * cell_i[2, 0])
        )
        volume = wp.float32(wp.abs(cell_volume))

        pbc_x = wp.bool(False)
        pbc_y = wp.bool(False)
        pbc_z = wp.bool(False)
        if pbc_is_batched:
            pbc_x = pbc_batch[tid, 0]
            pbc_y = pbc_batch[tid, 1]
            pbc_z = pbc_batch[tid, 2]
        else:
            pbc_x = pbc_single[0]
            pbc_y = pbc_single[1]
            pbc_z = pbc_single[2]
        any_pbc = pbc_x or pbc_y or pbc_z
        all_pbc = pbc_x and pbc_y and pbc_z

        n_float = wp.float32(n_atoms)
        active_atoms = n_atoms
        if has_target_indices:
            active_atoms = wp.int32(
                wp.ceil(
                    wp.float32(n_atoms)
                    * wp.float32(max(target_count, wp.int32(0)))
                    / wp.float32(max(total_atoms, wp.int32(1)))
                )
            )
            active_atoms = min(active_atoms, n_atoms)
        active_float = wp.float32(active_atoms)
        n_sq = n_float * n_float
        # Naive candidate count: each of ``active`` source atoms is tested against
        # all ``n`` atoms, so active_n_work ~ N^2 distance checks. n_pairs_cap =
        # n*(n-1) is the exact upper bound on real pairs.
        active_n_work = active_float * n_float
        n_pairs_cap = wp.max(
            n_float * wp.float32(max(n_atoms - wp.int32(1), 0)), wp.float32(0.0)
        )

        if volume <= wp.float32(_EPS):
            wp.atomic_max(flags, 2, wp.int32(1))
            wp.atomic_max(flags, 4, wp.int32(1))
            if any_pbc:
                wp.atomic_max(flags, 1, wp.int32(1))
            wp.atomic_add(costs, 0, n_sq)
            wp.atomic_add(costs, 1, n_sq)
            wp.atomic_add(costs, 2, wp.float32(_INF))
            wp.atomic_add(costs, 3, wp.float32(_INF))
            wp.atomic_add(costs, 4, wp.float32(_INF))
            return

        inverse_cell_transpose = wp.transpose(wp.inverse(cell_i))
        face_distance_x = wp.float32(1.0) / wp.float32(
            wp.length(inverse_cell_transpose[0])
        )
        face_distance_y = wp.float32(1.0) / wp.float32(
            wp.length(inverse_cell_transpose[1])
        )
        face_distance_z = wp.float32(1.0) / wp.float32(
            wp.length(inverse_cell_transpose[2])
        )
        min_face_distance = wp.min(
            face_distance_x, wp.min(face_distance_y, face_distance_z)
        )
        if min_face_distance <= wp.float32(_EPS):
            wp.atomic_max(flags, 2, wp.int32(1))
            wp.atomic_max(flags, 4, wp.int32(1))
            return

        shift_x = wp.int32(0)
        shift_y = wp.int32(0)
        shift_z = wp.int32(0)
        if any_pbc:
            if pbc_x:
                shift_x = wp.int32(
                    wp.ceil(wp.float32(wp.length(inverse_cell_transpose[0])) * cutoff)
                )
            if pbc_y:
                shift_y = wp.int32(
                    wp.ceil(wp.float32(wp.length(inverse_cell_transpose[1])) * cutoff)
                )
            if pbc_z:
                shift_z = wp.int32(
                    wp.ceil(wp.float32(wp.length(inverse_cell_transpose[2])) * cutoff)
                )

        shift_count = (
            wp.int64(shift_x) * wp.int64(2 * shift_y + 1) * wp.int64(2 * shift_z + 1)
            + wp.int64(shift_y) * wp.int64(2 * shift_z + 1)
            + wp.int64(shift_z)
            + wp.int64(1)
        )
        if shift_count <= wp.int64(0):
            wp.atomic_max(flags, 1, wp.int32(1))
            shift_count = wp.int64(1)
        if shift_count > max_launch_size:
            wp.atomic_max(flags, 1, wp.int32(1))
        if shift_count * wp.int64(max(total_atoms, wp.int32(1))) > max_launch_size:
            wp.atomic_max(flags, 1, wp.int32(1))
        if (
            shift_count
            * wp.int64(max(n_atoms, wp.int32(1)))
            * wp.int64(max(num_systems, wp.int32(1)))
            > max_launch_size
        ):
            wp.atomic_max(flags, 1, wp.int32(1))

        # Naive viability bound: above ~8e9 candidate pairs
        # (active_n_work * shift_count) naive stops being competitive with
        # cell_list, so mark it unsafe.
        if active_n_work * wp.float32(shift_count) > wp.float32(8.0e9):
            wp.atomic_max(flags, 1, wp.int32(1))

        cutoff_volume = cutoff * cutoff * cutoff
        # Floor against float32 underflow for tiny cutoffs so the cell-cost
        # division below (volume / cutoff_volume) cannot produce inf/NaN.
        cutoff_volume = wp.max(cutoff_volume, wp.float32(_EPS))
        density = n_float / wp.max(volume, wp.float32(_EPS))
        # Expected output work, shared by every method: neighbors per atom
        # (density * cutoff-sphere volume) times the active source atoms,
        # capped by the exact pair count for non-periodic and halved for half_fill.
        expected_neighbors = density * wp.float32(_SPHERE_VOLUME_FACTOR) * cutoff_volume
        expected_pairs = active_float * expected_neighbors
        if not any_pbc:
            expected_pairs = wp.min(expected_pairs, wp.min(n_pairs_cap, active_n_work))
        if half_fill:
            expected_pairs = expected_pairs * wp.float32(0.5)

        # Fixed per-system launch/allocation overhead (higher when batched).
        naive_setup = wp.float32(2048.0)
        if is_batched:
            naive_setup = wp.float32(20000.0)
        # Naive cost = per-candidate scan + per-pair output write + setup.
        # Scan weight: scalar 0.35 (global loads) vs tile 0.010 (shared-memory
        # reuse). Output write weight: 2.0 scalar vs 1.5 tile.
        scalar_cost = (
            wp.float32(0.35) * active_n_work * wp.float32(shift_count)
            + wp.float32(2.0) * expected_pairs
            + naive_setup
        )
        tile_cost = (
            wp.float32(0.010) * active_n_work * wp.float32(shift_count)
            + wp.float32(1.5) * expected_pairs
            + naive_setup
        )
        if (
            (not has_cuda)
            or has_pair_outputs
            or (any_pbc and is_batched and wrap_positions_false)
        ):
            wp.atomic_max(flags, 8, wp.int32(1))
        wp.atomic_add(costs, 0, scalar_cost)
        wp.atomic_add(costs, 1, tile_cost)

        cells_per_dimension = wp.vec3i(0, 0, 0)
        ADAPTIVE_MIN_CELLS = wp.int32(4)
        # Clamp the float->int cast so an extreme face_distance/cutoff ratio
        # cannot overflow int32; the halving loop below reduces the count to
        # <= max_nbins regardless, so capping per axis here is lossless.
        max_cells_f = wp.float32(max_nbins)
        cells_per_dimension[0] = max(
            wp.int32(wp.min(face_distance_x / cutoff, max_cells_f)), 1
        )
        cells_per_dimension[1] = max(
            wp.int32(wp.min(face_distance_y / cutoff, max_cells_f)), 1
        )
        cells_per_dimension[2] = max(
            wp.int32(wp.min(face_distance_z / cutoff, max_cells_f)), 1
        )

        for dim in range(3):
            pbc_dim = pbc_z
            if dim == 0:
                pbc_dim = pbc_x
            elif dim == 1:
                pbc_dim = pbc_y
            if pbc_dim or cells_per_dimension[dim] > wp.int32(1):
                while cells_per_dimension[dim] < ADAPTIVE_MIN_CELLS:
                    cells_per_dimension[dim] = cells_per_dimension[dim] * wp.int32(2)

        # Accumulate the cell-count product in int64 so a large grid cannot wrap
        # around int32 and slip past the guards below; the halving loop reduces
        # it to <= max_nbins, which always fits int32.
        total_cells_i64 = (
            wp.int64(cells_per_dimension[0])
            * wp.int64(cells_per_dimension[1])
            * wp.int64(cells_per_dimension[2])
        )
        while total_cells_i64 > wp.int64(max_nbins):
            for dim in range(3):
                cells_per_dimension[dim] = max(
                    cells_per_dimension[dim] // wp.int32(2), wp.int32(1)
                )
            total_cells_i64 = (
                wp.int64(cells_per_dimension[0])
                * wp.int64(cells_per_dimension[1])
                * wp.int64(cells_per_dimension[2])
            )
        total_cells_i32 = wp.int32(total_cells_i64)
        if total_cells_i32 <= wp.int32(0):
            wp.atomic_max(flags, 2, wp.int32(1))
            total_cells_i32 = wp.int32(1)
        if (
            wp.int64(total_cells_i32) * wp.int64(max(num_systems, wp.int32(1)))
            > max_launch_size
        ):
            wp.atomic_max(flags, 2, wp.int32(1))

        # cost_cells = grid cell count (volume / cutoff^3), clamped to
        # [1, max_nbins]. grid_work is the pruned candidate count: naive
        # active_n_work scanned over ``shell`` stencil cells (default 27 ~ 3x3x3).
        cost_cells = volume / cutoff_volume
        cost_cells = wp.max(cost_cells, wp.float32(1.0))
        cost_cells = wp.min(cost_cells, wp.float32(max_nbins))
        grid_work = shell * active_n_work / cost_cells
        # Atom-centric: 2.0 per-candidate distance test + 0.8 output write,
        # over the cell-grid build/sort floor ``setup``.
        cell_atom_cost = (
            setup + wp.float32(2.0) * grid_work + wp.float32(0.8) * expected_pairs
        )

        radius_x = wp.int32(
            wp.ceil(cutoff * wp.float32(cells_per_dimension[0]) / face_distance_x)
        )
        radius_y = wp.int32(
            wp.ceil(cutoff * wp.float32(cells_per_dimension[1]) / face_distance_y)
        )
        radius_z = wp.int32(
            wp.ceil(cutoff * wp.float32(cells_per_dimension[2]) / face_distance_z)
        )
        radius_x = max(radius_x, wp.int32(0))
        radius_y = max(radius_y, wp.int32(0))
        radius_z = max(radius_z, wp.int32(0))
        n_outer = (
            radius_x
            * (wp.int32(2) * radius_y + wp.int32(1))
            * (wp.int32(2) * radius_z + wp.int32(1))
            + radius_y * (wp.int32(2) * radius_z + wp.int32(1))
            + radius_z
        )
        if not half_fill:
            n_outer = (wp.int32(2) * radius_x + wp.int32(1)) * (
                wp.int32(2) * radius_y + wp.int32(1)
            ) * (wp.int32(2) * radius_z + wp.int32(1)) - wp.int32(1)
        # Pair-centric launches one 64-thread block per (cell, neighbor-offset);
        # n_outer = stencil offsets (forward half only when half_fill).
        # pair_launch ~ total threads; pair_blocks ~ block count.
        pair_launch = (
            wp.int64(total_cells_i32) * wp.int64(n_outer + wp.int32(1)) * wp.int64(64)
        )
        atom_blocks = wp.int64(max(n_atoms, wp.int32(1)) + wp.int32(63)) // wp.int64(64)
        pair_blocks = wp.int64(total_cells_i32) * wp.int64(n_outer + wp.int32(1))
        if (not has_cuda) or pair_launch > max_launch_size or pair_blocks < atom_blocks:
            wp.atomic_max(flags, 7, wp.int32(1))
        # Pair-centric: cheaper per pair (0.55 write, 0.5 scan) but pays a
        # per-block launch term (0.025 * pair_launch) for its many small blocks.
        cell_pair_cost = (
            setup
            + wp.float32(0.55) * expected_pairs
            + wp.float32(0.025) * wp.float32(pair_launch)
            + wp.float32(0.5) * grid_work
        )
        wp.atomic_add(costs, 2, cell_atom_cost)
        wp.atomic_add(costs, 3, cell_pair_cost)

        if (
            half_fill
            or has_target_indices
            or use_pair_fn
            or has_rebuild_flags
            or (has_cutoff2 and (return_list or has_vectors or has_distances))
        ):
            wp.atomic_max(flags, 3, wp.int32(1))
        if (not has_cuda) or (not is_float32) or (not all_pbc):
            wp.atomic_max(flags, 4, wp.int32(1))
        if cutoff > wp.float32(0.5) * min_face_distance:
            wp.atomic_max(flags, 6, wp.int32(1))
        if is_batched and batch_idx_is_provided:
            # The per-atom validation above sets flag 5 when the dense labels
            # do not match the contiguous ranges implied by batch_ptr.
            pass

        cluster_large_enough = wp.bool(False)
        if is_batched:
            cluster_large_enough = total_atoms >= wp.int32(4096)
        else:
            cluster_large_enough = expected_neighbors >= wp.float32(512.0)
        if not cluster_large_enough:
            wp.atomic_max(flags, 4, wp.int32(1))

        # Cluster-tile one-time pipeline floor (Morton encode, sort, cluster
        # build, tile scan), added once per batch rather than per-system.
        if tid == 0:
            wp.atomic_add(costs, 4, wp.float32(1200000.0))
        # Per-atom build: 32 * n (radix sort + cluster assignment, modeled
        # linearly over the large-N range cluster_tile runs in); the
        # 0.35 * expected_pairs term is the tile query output write.
        cluster_cost = (
            wp.float32(0.35) * expected_pairs + wp.float32(32.0) * n_float + setup
        )
        wp.atomic_add(costs, 4, cluster_cost)

    return _kernel


def _validate_selector_inputs(
    batch_ptr: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    batch_idx: wp.array | None,
) -> int:
    """Validate shared selector inputs and return ``num_systems``."""
    if batch_ptr.dtype != wp.int32:
        raise ValueError("batch_ptr must be a wp.array with dtype=wp.int32")
    if batch_ptr.ndim != 1:
        raise ValueError("batch_ptr must be a 1-D wp.array")
    if batch_ptr.shape[0] < 2:
        return 0
    num_systems = int(batch_ptr.shape[0] - 1)
    if cell.shape[0] != num_systems:
        raise ValueError("cell must contain one matrix per system")
    if pbc.dtype != wp.bool:
        raise ValueError("pbc must be a wp.array with dtype=wp.bool")
    if pbc.ndim not in (1, 2):
        raise ValueError("pbc must have shape (3,) or (num_systems, 3)")
    if pbc.ndim == 1 and pbc.shape[0] != 3:
        raise ValueError("pbc must have shape (3,) or (num_systems, 3)")
    if pbc.ndim == 2 and (pbc.shape[0] != num_systems or pbc.shape[1] != 3):
        raise ValueError("pbc must have shape (3,) or (num_systems, 3)")
    device = str(batch_ptr.device)
    if str(cell.device) != device or str(pbc.device) != device:
        raise ValueError("batch_ptr, cell, and pbc must be on the same device")
    if batch_idx is not None:
        if batch_idx.dtype != wp.int32 or batch_idx.ndim != 1:
            raise ValueError("batch_idx must be a 1-D wp.array with dtype=wp.int32")
        if str(batch_idx.device) != device:
            raise ValueError("batch_idx must be on the same device as batch_ptr")
    if not float(cutoff) > 0.0:
        raise ValueError("cutoff must be positive")
    return num_systems


def _default_feature_mask(batch_ptr: wp.array, cell: wp.array, num_systems: int) -> int:
    """Return feature bits inferred by the shared Warp selector."""
    mask = 0
    if "cuda" in str(batch_ptr.device).lower():
        mask |= FEATURE_CUDA
    if _wp_scalar_from_cell_dtype(cell.dtype) == wp.float32:
        mask |= FEATURE_POSITIONS_FLOAT32
    if num_systems > 1:
        mask |= FEATURE_BATCHED
    return mask


def estimate_neighbor_list_costs(
    batch_ptr: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    *,
    batch_idx: wp.array | None = None,
    max_nbins: int | None = None,
    max_launch_size: int = _MAX_WARP_LINEAR_LAUNCH,
    optional_outputs: Iterable[str] | None = None,
    option_mask: int = 0,
    feature_mask: int | None = None,
    target_count: int | None = None,
) -> list[tuple[str, float]]:
    """Report feasible neighbor-list strategies and their estimated cost.

    Parameters
    ----------
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts. The final entry is the total atom count.
    cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Per-system cell matrices. Non-periodic callers should pass a
        synthesized bounding-box cell.
    pbc : wp.array, shape (3,) or (num_systems, 3), dtype=wp.bool
        Shared or per-system periodic-boundary flags.
    cutoff : float
        Neighbor cutoff.  For dual-cutoff routing, pass the larger cutoff.
    batch_idx : wp.array, optional
        Dense per-atom batch ids.  When provided, the selector validates that
        the labels match the contiguous ranges implied by ``batch_ptr`` before
        allowing auto cluster-tile.
    max_nbins : int, optional
        Per-system cell-list cap. Defaults to the single-system cap when
        ``num_systems == 1`` and the batched cap otherwise.
    max_launch_size : int, default=2**31 - 1
        Conservative launch-size guard used by current neighbor-list kernels.
    optional_outputs : iterable of str, optional
        Public-style neighbor-list option names, encoded with
        :func:`optional_outputs_mask`.
    option_mask : int, default=0
        Pre-encoded option bits to OR with ``optional_outputs``.
    feature_mask : int, optional
        Device/dtype/frontend feature bits.  When omitted, CUDA and cell dtype
        are inferred from the Warp arrays.
    target_count : int, optional
        Number of source rows requested by ``target_indices``.  When omitted,
        the selector scores all atoms.  Frontends should pass
        ``len(target_indices)`` when the public ``target_indices`` kwarg is
        active.

    Returns
    -------
    list of (str, float)
        Feasible strategies (from :data:`NEIGHBOR_LIST_STRATEGIES`) and their
        relative estimated cost (lower is faster), sorted cheapest-first.
        Batched inputs (``num_systems > 1``) return ``batch_`` prefixed names.

    Notes
    -----
    The returned costs are *relative* (arbitrary units): only their ordering is
    meaningful, so compare them to each other, not to a wall-clock time.  The
    model approximates algorithmic work (candidate pairs, neighbors written,
    launch overhead) and is **hardware-independent** -- the true crossover
    between strategies shifts with the device, so when the top costs are within a
    small factor the predicted best may be marginally slower than a close
    runner-up; benchmark the top few on your hardware in that case.

    This launches one Warp kernel over systems (and over atoms when validating
    ``batch_idx`` contiguity) and reads back five costs plus nine flags, so it
    is **host-only**: call it outside ``torch.compile`` / ``jax.jit`` and pass
    the chosen name as an explicit ``method=`` to run compiled.
    """
    num_systems = _validate_selector_inputs(batch_ptr, cell, pbc, cutoff, batch_idx)
    if num_systems == 0:
        return [("cell_list_atom_centric", 0.0)]

    if max_nbins is None:
        max_nbins = (
            DEFAULT_SINGLE_MAX_NBINS if num_systems == 1 else DEFAULT_BATCH_MAX_NBINS
        )
    if int(max_nbins) <= 0:
        raise ValueError("max_nbins must be positive")

    device = batch_ptr.device
    costs = wp.zeros(5, dtype=wp.float32, device=device)
    flags = wp.zeros(len(_FLAG_NAMES), dtype=wp.int32, device=device)
    shell, setup = auto_base_constants()
    pbc_is_batched = pbc.ndim == 2
    pbc_single = pbc if not pbc_is_batched else empty_sentinel(1, wp.bool, device)
    pbc_batch = pbc if pbc_is_batched else empty_sentinel(2, wp.bool, device)
    batch_idx_is_provided = batch_idx is not None
    batch_idx_arg = (
        batch_idx if batch_idx is not None else empty_sentinel(1, wp.int32, device)
    )
    wp_dtype = _wp_scalar_from_cell_dtype(cell.dtype)
    feature_bits = _default_feature_mask(batch_ptr, cell, num_systems)
    if feature_mask is not None:
        feature_bits |= int(feature_mask)
    options = int(option_mask) | optional_outputs_mask(optional_outputs)
    if options & _OPTION_TARGET_INDICES and target_count is None:
        raise ValueError(
            "target_count is required when target_indices is included in "
            "optional_outputs"
        )
    target_count_int = 0 if target_count is None else int(target_count)
    launch_dim = max(
        num_systems, int(batch_idx.shape[0]) if batch_idx is not None else 0
    )

    wp.launch(
        get_select_neighbor_list_method_cost_kernel(wp_dtype),
        dim=launch_dim,
        inputs=[
            batch_ptr,
            batch_idx_arg,
            batch_idx_is_provided,
            cell,
            pbc_single,
            pbc_batch,
            pbc_is_batched,
            wp.float32(float(cutoff)),
            wp.float32(float(shell)),
            wp.float32(float(setup)),
            wp.int32(int(max_nbins)),
            wp.int64(int(max_launch_size)),
            wp.int32(options),
            wp.int32(feature_bits),
            wp.int32(target_count_int),
            costs,
            flags,
        ],
        device=device,
    )

    strategies = finalize_neighbor_list_method(costs.numpy(), flags.numpy())
    if num_systems > 1:
        strategies = [("batch_" + name, cost) for name, cost in strategies]
    return strategies


def suggest_neighbor_list_method(*args, **kwargs) -> str:
    """Return the cheapest feasible neighbor-list strategy name.

    Thin wrapper over :func:`estimate_neighbor_list_costs` that returns only the
    top-ranked strategy name.  Accepts the same arguments and shares the same
    host-only synchronization caveat (call outside ``torch.compile`` /
    ``jax.jit``).
    """
    return estimate_neighbor_list_costs(*args, **kwargs)[0][0]

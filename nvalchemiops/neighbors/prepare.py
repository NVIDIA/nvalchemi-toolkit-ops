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


"""Framework-agnostic planning for neighbor-list buffer preparation.

The Torch and JAX ``prepare_neighbor_list_method`` functions differ only in how
they allocate arrays. Everything that decides *what* to allocate -- output
capacity, which buffers a strategy needs, how dual-cutoff buffers are named --
lives here so the two bindings cannot drift. The capacity rule in particular is
calibrated against measurement and is the one piece most expensive to duplicate.
"""

from __future__ import annotations

from dataclasses import dataclass

from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors

__all__ = [
    "_CELL_LIST_STATE_KEYS",
    "_BufferSpec",
    "_MIN_EXTENT",
    "_density_from_volume",
    "_neighbor_capacity",
    "_output_buffer_specs",
    "_split_batched_method",
    "_strategy_scratch_kind",
]

# Scratch returned by ``allocate_cell_list``, in order.
_CELL_LIST_STATE_KEYS = (
    "cells_per_dimension",
    "neighbor_search_radius",
    "atom_periodic_shifts",
    "atom_to_cell_mapping",
    "atoms_per_cell_count",
    "cell_atom_start_indices",
    "cell_atom_list",
)

# Poisson spread allowance on the busiest row, in units of sqrt(mean).
_CAPACITY_SIGMA = 6.0
# Matches the default of ``estimate_max_neighbors``.
_DEFAULT_DENSITY = 0.2
# Floor on a per-axis extent when the density falls back to the bounding box,
# so a flat or single-atom system cannot produce a zero volume.
_MIN_EXTENT = 1e-6


@dataclass(frozen=True)
class _BufferSpec:
    """One output buffer a strategy needs, described without allocating it.

    Attributes
    ----------
    key : str
        Keyword name the neighbor-list function expects.
    shape : tuple of int
        Buffer shape.
    fill : int or None
        Constant to fill with; ``None`` means zeros.
    """

    key: str
    shape: tuple[int, ...]
    fill: int | None = None


def _density_from_volume(total_atoms: int, volume: float) -> float:
    """Return mean number density, falling back to a neutral default.

    Parameters
    ----------
    total_atoms : int
        Number of atoms in the system.
    volume : float
        Enclosing volume; non-positive values fall back to the default.

    Returns
    -------
    float
        Atoms per unit volume.
    """
    if total_atoms <= 0 or volume <= 0.0:
        return _DEFAULT_DENSITY
    return total_atoms / volume


def _neighbor_capacity(
    radius: float,
    atomic_density: float,
    requested: int | None = None,
) -> int:
    """Return rows to allocate per atom for a cutoff of ``radius``.

    Parameters
    ----------
    radius : float
        Cutoff the buffer must hold neighbours for.
    atomic_density : float
        Mean number density of the system.
    requested : int, optional
        Caller-supplied capacity; returned unchanged when given.

    Returns
    -------
    int
        Capacity, rounded up to a multiple of 16.

    Notes
    -----
    This is :func:`estimate_max_neighbors` with an allowance for how far the
    busiest row sits above the mean, which the mean estimate alone does not
    carry. It is still an estimate, not a guarantee: callers should compare the
    resulting counts against the capacity and re-prepare when it is exceeded.

    See Also
    --------
    estimate_max_neighbors : The underlying mean estimate.
    """
    if requested is not None:
        return int(requested)
    return estimate_max_neighbors(
        radius,
        atomic_density=atomic_density,
        fluctuation_sigma=_CAPACITY_SIGMA,
    )


def _output_buffer_specs(
    num_rows: int,
    max_neighbors: int,
    periodic: bool,
    suffix: str = "",
) -> tuple[_BufferSpec, ...]:
    """Describe the matrix outputs for one cutoff.

    Parameters
    ----------
    num_rows : int
        Rows in the neighbour matrix: atoms, or targets when restricted.
    max_neighbors : int
        Columns in the neighbour matrix.
    periodic : bool
        Whether shifts are produced; free boundaries have none.
    suffix : str, optional
        ``""`` for single-cutoff methods, ``"1"`` / ``"2"`` for dual-cutoff
        methods, which name their buffers ``neighbor_matrix1`` and so on.

    Returns
    -------
    tuple of _BufferSpec
        Buffers to allocate. ``fill`` is the caller's fill value for the
        matrix and ``None`` (zeros) elsewhere.
    """
    specs = [
        _BufferSpec(f"neighbor_matrix{suffix}", (num_rows, max_neighbors), fill=0),
        _BufferSpec(f"num_neighbors{suffix}", (num_rows,)),
    ]
    if periodic:
        specs.append(
            _BufferSpec(f"neighbor_matrix_shifts{suffix}", (num_rows, max_neighbors, 3))
        )
    return tuple(specs)


def _split_batched_method(method: str) -> tuple[str, bool]:
    """Split a strategy name into its base name and batched flag.

    Parameters
    ----------
    method : str
        Strategy name, optionally ``batch_`` prefixed.

    Returns
    -------
    base : str
        Name without the prefix.
    batched : bool
        Whether the name carried the prefix.
    """
    if method.startswith("batch_"):
        return method[len("batch_") :], True
    return method, False


def _strategy_scratch_kind(method: str, periodic: bool) -> str:
    """Return which family of scratch buffers a strategy needs.

    Parameters
    ----------
    method : str
        Strategy name, optionally ``batch_`` prefixed.
    periodic : bool
        Whether the call runs with periodic boundaries.

    Returns
    -------
    str
        ``"cell_list"``, ``"naive_periodic"``, or ``"none"``. Aperiodic naive
        and cluster_tile need no strategy-specific scratch.
    """
    base, _ = _split_batched_method(method)
    if base.startswith("cell_list"):
        return "cell_list"
    if base.startswith("naive") and periodic:
        return "naive_periodic"
    return "none"

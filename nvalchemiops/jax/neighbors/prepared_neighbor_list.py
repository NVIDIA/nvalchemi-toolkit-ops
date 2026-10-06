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

"""Prepared immutable state for supported JAX neighbor-list routes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from nvalchemiops.jax.neighbors._registration import (
    _lazy_cell_inverse_kernel,
    _lazy_cluster_geometry_kernel,
)

__all__ = [
    "NeighborListState",
    "check_neighbor_list_state",
    "prepare_neighbor_list",
]
_TOKEN = object()
_CALLER_OWNED = frozenset(
    {
        "atom_periodic_shifts",
        "atom_to_cell_mapping",
        "atoms_per_cell_count",
        "batch_idx_sorted",
        "batch_ptr_padded",
        "cell_atom_list",
        "cell_atom_start_indices",
        "cell_offsets",
        "cells_per_dimension",
        "group_ctr_x",
        "group_ctr_y",
        "group_ctr_z",
        "group_ext_x",
        "group_ext_y",
        "group_ext_z",
        "group_ptr",
        "group_system",
        "inv_cell_buffer",
        "inv_cell_batch",
        "morton_codes",
        "neighbor_distances",
        "neighbor_list",
        "neighbor_list_shifts",
        "neighbor_matrix",
        "neighbor_matrix1",
        "neighbor_matrix2",
        "neighbor_matrix_shifts",
        "neighbor_matrix_shifts1",
        "neighbor_matrix_shifts2",
        "neighbor_vectors",
        "neighbor_search_radius",
        "num_tiles",
        "num_neighbors",
        "num_neighbors1",
        "num_neighbors2",
        "pair_energies",
        "pair_counts",
        "pair_counter",
        "pair_forces",
        "pair_offsets",
        "per_atom_cell_offsets_buffer",
        "positions_wrapped_buffer",
        "sort_inv",
        "sorted_atom_index",
        "sorted_pos_x",
        "sorted_pos_y",
        "sorted_pos_z",
        "sorted_positions",
        "sorted_shifts",
        "tile_col_group",
        "tile_counts",
        "tile_offsets",
        "tile_row_group",
        "tile_system",
    }
)
_CONFIG_KEYS = frozenset(
    {
        "atom_centric_path",
        "coo_capacity",
        "format",
        "max_atoms_per_system",
        "max_neighbors",
        "max_neighbors1",
        "max_neighbors2",
        "max_shifts_per_system",
        "max_tiles_per_group",
        "max_total_cells",
        "num_shifts_per_system",
        "pair_fn",
        "pair_centric_n_outer",
        "pair_centric_r_max",
        "pair_centric_total_cells",
        "pair_params",
        "return_distances",
        "return_vectors",
        "shift_range_per_dimension",
        "span_margin",
        "strategy",
        "target_indices",
    }
)
_EXECUTION_KEYS = _CONFIG_KEYS | {"rebuild_flags"}
_CELL_LEAF = 0
_PBC_LEAF = 1
_BATCH_IDX_LEAF = 2
_BATCH_PTR_LEAF = 3
_FIXED_CELL_GEOMETRY_LEAF = 4
_RESULT_LEAF_OFFSET = 5
_RESULT_LEAF_COUNT = 12
_SHIFT_RANGE_LEAF = -18
_NUM_SHIFTS_LEAF = -17
_TARGET_INDICES_LEAF = -16
_SPAN_CAPACITY_LEAF = -15
_SPAN_AXIS_LEAF = -14
_SPAN_REQUIRED_LEAF = -13
_INITIALIZED_LEAF = -12
_VALID_LEAF = -11
_FAILURE_CODE_LEAF = -10
_REQUIRED_LEAF = -9
_CAPACITY_LEAF = -8
_FIRST_FAILURE_LEAF = -7
_PRIVATE_MATRIX1_LEAF = -6
_PRIVATE_NUM1_LEAF = -5
_PRIVATE_SHIFT1_LEAF = -4
_PRIVATE_MATRIX2_LEAF = -3
_PRIVATE_NUM2_LEAF = -2
_PRIVATE_SHIFT2_LEAF = -1
_FIXED_CELL_INVERSE_REGISTRATION = _lazy_cell_inverse_kernel()
_FIXED_CLUSTER_GEOMETRY_REGISTRATION = _lazy_cluster_geometry_kernel()
_CUTOFF_GROUP_RESULT_NAMES = frozenset(
    {
        "neighbor_matrix",
        "num_neighbors",
        "neighbor_matrix_shifts",
        "neighbor_list",
        "neighbor_ptr",
        "neighbor_list_shifts",
        "metadata_valid",
    }
)


def _result_property(name: str) -> property:
    """Create a documented read-only view of an applicable result leaf."""

    def get(state: NeighborListState) -> jax.Array | None:
        possible_base = name[:-1] if name.endswith(("1", "2")) else name
        is_grouped = possible_base in _CUTOFF_GROUP_RESULT_NAMES
        is_first = is_grouped and name.endswith("1")
        is_second = is_grouped and name.endswith("2")
        if is_grouped and state.cutoff2 is not None:
            if not (is_first or is_second):
                return None
            result_name = possible_base if is_first else name
        elif is_grouped and (is_first or is_second):
            return None
        else:
            result_name = name
        if result_name not in state._spec.result_names:
            return None
        return _result_leaf(state, result_name)

    get.__name__ = name
    doc = {
        "metadata_valid": "Scalar Boolean recovery-metadata status, or ``None`` when not applicable.",
        "metadata_valid1": "Primary-cutoff recovery-metadata status, or ``None``.",
        "metadata_valid2": "Secondary-cutoff recovery-metadata status, or ``None``.",
        "neighbor_distances": "Latest per-pair distances, or ``None`` when not requested.",
        "neighbor_vectors": "Latest per-pair displacement vectors, or ``None`` when not requested.",
        "pair_energies": "Latest pair-callback energies, or ``None`` when not requested.",
        "pair_forces": "Latest pair-callback forces, or ``None`` when not requested.",
    }.get(name)
    if doc is None and name.startswith("neighbor_matrix_shifts"):
        doc = "Latest matrix-format periodic shifts, or ``None`` when not applicable."
    elif doc is None and name.startswith("neighbor_matrix"):
        doc = "Latest fixed-width neighbor matrix, or ``None`` when not applicable."
    elif doc is None and name.startswith("num_neighbors"):
        doc = "Latest required neighbor count per source row, or ``None`` when not applicable."
    elif doc is None and name.startswith("neighbor_list_shifts"):
        doc = "Latest COO-format periodic shifts, or ``None`` when not applicable."
    elif doc is None and name.startswith("neighbor_list"):
        doc = "Latest COO neighbor indices, or ``None`` when not applicable."
    elif doc is None and name.startswith("neighbor_ptr"):
        doc = "Latest clipped CSR pointer for fixed COO output, or ``None``."
    elif doc is None and name == "pair_offsets":
        doc = "Segment offsets for the latest segmented COO output, or ``None``."
    elif doc is None and name == "pair_counts":
        doc = "Active pair count per segmented COO system, or ``None``."
    elif doc is None and name == "tile_offsets":
        doc = "Segment offsets for batched cluster tiles, or ``None``."
    elif doc is None and name == "tile_counts":
        doc = "Active cluster-tile count per system, or ``None``."
    elif doc is None and name == "num_tiles":
        doc = "Latest active cluster-tile count, or ``None``."
    elif doc is None and name.startswith("tile_"):
        doc = "Latest cluster-tile metadata, or ``None`` when not applicable."
    elif doc is None:
        doc = "Latest applicable route result, or ``None`` when not applicable."
    return property(get, doc=doc)


def _result_leaf(
    state: NeighborListState,
    name: str,
    leaves: tuple[Any, ...] | None = None,
) -> Any:
    """Return a named result leaf from the fixed prepared leaf layout."""
    values = state._leaves if leaves is None else leaves
    return values[_RESULT_LEAF_OFFSET + state._spec.result_names.index(name)]


@dataclass(frozen=True)
class _StateLeaves:
    """Named view over the fixed internal state-leaf layout."""

    values: tuple[Any, ...]

    @classmethod
    def build(
        cls,
        *,
        cell: Any,
        pbc: Any,
        batch_idx: Any,
        batch_ptr: Any,
        fixed_cell_geometry: Any,
        result: tuple[Any, ...] | list[Any],
        shift_range_per_dimension: Any,
        num_shifts_per_system: Any,
        target_indices: Any,
        span_capacity: Any,
        span_axis: Any,
        span_required: Any,
        initialized: Any,
        valid: Any,
        failure_code: Any,
        required: Any,
        capacity: Any,
        first_failure: Any,
        private_matrix1: Any,
        private_num1: Any,
        private_shift1: Any,
        private_matrix2: Any,
        private_num2: Any,
        private_shift2: Any,
    ) -> tuple[Any, ...]:
        """Build the fixed PyTree layout from named state components.

        The public state is intentionally a flat PyTree for JAX transform
        compatibility.  Keeping construction here makes the positional
        order auditable while preserving the established leaf sequence.
        """
        result = tuple(result)
        if len(result) > _RESULT_LEAF_COUNT:
            raise ValueError("prepared result contains too many state leaves")
        result = result + (None,) * (_RESULT_LEAF_COUNT - len(result))
        return (
            cell,
            pbc,
            batch_idx,
            batch_ptr,
            fixed_cell_geometry,
            *result,
            shift_range_per_dimension,
            num_shifts_per_system,
            target_indices,
            span_capacity,
            span_axis,
            span_required,
            initialized,
            valid,
            failure_code,
            required,
            capacity,
            first_failure,
            private_matrix1,
            private_num1,
            private_shift1,
            private_matrix2,
            private_num2,
            private_shift2,
        )

    def successor(
        self,
        result: tuple[Any, ...],
        *,
        span_axis: Any,
        span_required: Any,
        initialized: Any,
        valid: Any,
        failure_code: Any,
        required: Any,
        capacity: Any,
        first_failure: Any,
        private_leaves: tuple[Any, ...] | None = None,
    ) -> tuple[Any, ...]:
        """Build a successor using named fields and the same leaf order."""
        private = self.private if private_leaves is None else tuple(private_leaves)
        if len(private) != 6:
            raise ValueError("prepared private state must contain six leaves")
        return self.build(
            cell=self.cell,
            pbc=self.pbc,
            batch_idx=self.batch_idx,
            batch_ptr=self.batch_ptr,
            fixed_cell_geometry=self.fixed_cell_geometry,
            result=result,
            shift_range_per_dimension=self.values[_SHIFT_RANGE_LEAF],
            num_shifts_per_system=self.values[_NUM_SHIFTS_LEAF],
            target_indices=self.values[_TARGET_INDICES_LEAF],
            span_capacity=self.values[_SPAN_CAPACITY_LEAF],
            span_axis=span_axis,
            span_required=span_required,
            initialized=initialized,
            valid=valid,
            failure_code=failure_code,
            required=required,
            capacity=capacity,
            first_failure=first_failure,
            private_matrix1=private[0],
            private_num1=private[1],
            private_shift1=private[2],
            private_matrix2=private[3],
            private_num2=private[4],
            private_shift2=private[5],
        )

    @property
    def cell(self) -> Any:
        return self.values[_CELL_LEAF]

    @property
    def pbc(self) -> Any:
        return self.values[_PBC_LEAF]

    @property
    def batch_idx(self) -> Any:
        return self.values[_BATCH_IDX_LEAF]

    @property
    def batch_ptr(self) -> Any:
        return self.values[_BATCH_PTR_LEAF]

    @property
    def fixed_cell_geometry(self) -> Any:
        return self.values[_FIXED_CELL_GEOMETRY_LEAF]

    @property
    def initialized(self) -> Any:
        return self.values[_INITIALIZED_LEAF]

    @property
    def valid(self) -> Any:
        return self.values[_VALID_LEAF]

    @property
    def failure_code(self) -> Any:
        return self.values[_FAILURE_CODE_LEAF]

    @property
    def required(self) -> Any:
        return self.values[_REQUIRED_LEAF]

    @property
    def capacity(self) -> Any:
        return self.values[_CAPACITY_LEAF]

    @property
    def first_failure(self) -> Any:
        return self.values[_FIRST_FAILURE_LEAF]

    @property
    def span_axis(self) -> Any:
        return self.values[_SPAN_AXIS_LEAF]

    @property
    def span_required(self) -> Any:
        return self.values[_SPAN_REQUIRED_LEAF]

    @property
    def private_matrix1(self) -> Any:
        return self.values[_PRIVATE_MATRIX1_LEAF]

    @property
    def private_num1(self) -> Any:
        return self.values[_PRIVATE_NUM1_LEAF]

    @property
    def private_shift1(self) -> Any:
        return self.values[_PRIVATE_SHIFT1_LEAF]

    @property
    def private_matrix2(self) -> Any:
        return self.values[_PRIVATE_MATRIX2_LEAF]

    @property
    def private_num2(self) -> Any:
        return self.values[_PRIVATE_NUM2_LEAF]

    @property
    def private_shift2(self) -> Any:
        return self.values[_PRIVATE_SHIFT2_LEAF]

    @property
    def private(self) -> tuple[Any, ...]:
        return (
            self.private_matrix1,
            self.private_num1,
            self.private_shift1,
            self.private_matrix2,
            self.private_num2,
            self.private_shift2,
        )


def _private_state_leaves(leaves: tuple[Any, ...]) -> tuple[Any, ...]:
    """Return the fixed private matrix-storage suffix by field name."""
    return _StateLeaves(leaves).private


def _build_successor_leaves(
    leaves: tuple[Any, ...],
    result: tuple[Any, ...],
    *,
    span_axis: Any,
    span_required: Any,
    initialized: Any,
    valid: Any,
    failure_code: Any,
    required: Any,
    capacity: Any,
    first_failure_marker: Any,
    private_leaves: tuple[Any, ...] | None = None,
) -> tuple[Any, ...]:
    """Build a successor without changing the public PyTree leaf layout."""
    return _StateLeaves(leaves).successor(
        result,
        span_axis=span_axis,
        span_required=span_required,
        initialized=initialized,
        valid=valid,
        failure_code=failure_code,
        required=required,
        capacity=capacity,
        first_failure=first_failure_marker,
        private_leaves=private_leaves,
    )


@dataclass(frozen=True)
class _Spec:
    method: str
    strategy: str
    format: str
    coo_layout: str | None
    is_batched: bool
    num_atoms: int
    num_systems: int
    cutoff: float
    cutoff2: float | None
    half_fill: bool
    fill_value: int
    wrap_positions: bool
    selective: bool
    return_vectors: bool
    return_distances: bool
    span_margin: float
    synthesized_cell: bool
    fixed_cell: bool
    supports_compilation: bool
    compilation_blocker: str | None
    pair_fn: Any = None
    result_names: tuple[str, ...] = ()
    coo_capacity: int | tuple[int, int] | None = None
    max_neighbors: tuple[int, int | None] = (0, None)
    max_shifts_per_system: int | None = None
    max_atoms_per_system: int | None = None
    has_target_indices: bool = False
    shared_batch_cell: bool = False
    route_options: tuple[tuple[str, Any], ...] = ()


@jax.tree_util.register_pytree_node_class
class NeighborListState:
    """Prepared JAX neighbor-list configuration, results, and status.

    Instances are created by :func:`prepare_neighbor_list`; direct
    construction is unsupported. The state is an immutable JAX PyTree. Pass
    each successor returned by :func:`neighbor_list` to the next execution.

    Configuration properties describe the route resolved during eager
    preparation. Result properties mirror the selected route's existing tuple
    and return ``None`` when a component does not apply. ``initialized`` and
    ``valid`` are Boolean arrays of shape ``(num_systems,)``.

    Notes
    -----
    A failed system remains invalid after later successful executions. Call
    :func:`check_neighbor_list_state` when a host-side exception is needed;
    checking is optional and does not clear status. Preparing a new state is
    the only way to reset failure history.

    The read-only properties below expose the resolved method, strategy,
    output layout, fixed system metadata, compilation eligibility, and current
    lifecycle status. Result properties use the existing tuple component names
    and return ``None`` when a component is not applicable. Result arrays
    follow JAX functional semantics, and some are also successor-state leaves.
    Consume those results before donating the state to a later call, and do not
    subsequently use aliases retained from the donated state.
    """

    __slots__ = ("_spec", "_leaves")

    def __init__(
        self,
        spec: _Spec | None = None,
        leaves: tuple[Any, ...] | None = None,
        *,
        token: object | None = None,
    ) -> None:
        if token is not _TOKEN:
            raise TypeError(
                "state must be a NeighborListState returned by prepare_neighbor_list"
            )
        self._spec = spec
        self._leaves = tuple(leaves)

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, name):
            raise AttributeError(f"prepared state attribute {name!r} is read-only")
        super().__setattr__(name, value)

    def tree_flatten(self) -> tuple[tuple[Any, ...], _Spec]:
        """Return dynamic array leaves and the static prepared-route specification."""
        return self._leaves, self._spec

    @classmethod
    def tree_unflatten(cls, spec: _Spec, leaves: tuple[Any, ...]) -> NeighborListState:
        """Reconstruct a prepared state from its PyTree specification and leaves."""
        return cls(spec, tuple(leaves), token=_TOKEN)

    def _replace(self, leaves: tuple[Any, ...]) -> NeighborListState:
        return NeighborListState(self._spec, tuple(leaves), token=_TOKEN)

    @property
    def method(self) -> str:
        """Return the resolved algorithm family."""
        return self._spec.method

    @property
    def strategy(self) -> str:
        """Return the resolved implementation strategy."""
        return self._spec.strategy

    @property
    def format(self) -> str:
        """Return the prepared scientific output format."""
        return self._spec.format

    @property
    def coo_layout(self) -> str | None:
        """Return ``"compact"``, ``"segmented"``, or ``None``."""
        return self._spec.coo_layout

    @property
    def is_batched(self) -> bool:
        """Return whether the resolved route is batched."""
        return self._spec.is_batched

    @property
    def num_atoms(self) -> int:
        """Return the fixed atom count."""
        return self._spec.num_atoms

    @property
    def num_systems(self) -> int:
        """Return the fixed system count."""
        return self._spec.num_systems

    @property
    def cutoff(self) -> float:
        """Return the prepared primary cutoff."""
        return self._spec.cutoff

    @property
    def cutoff2(self) -> float | None:
        """Return the prepared secondary cutoff, if any."""
        return self._spec.cutoff2

    @property
    def half_fill(self) -> bool:
        """Return whether the route stores one directed pair half."""
        return self._spec.half_fill

    @property
    def fill_value(self) -> int:
        """Return the prepared matrix padding value."""
        return self._spec.fill_value

    @property
    def wrap_positions(self) -> bool:
        """Return whether applicable routes wrap positions before enumeration."""
        return self._spec.wrap_positions

    @property
    def fixed_cell(self) -> bool:
        """Return whether repeated execution reuses prepared cell geometry."""
        return self._spec.fixed_cell

    @property
    def selective(self) -> bool:
        """Return whether runtime per-system rebuild selection is enabled."""
        return self._spec.selective

    @property
    def return_vectors(self) -> bool:
        """Return whether displacement vectors are requested."""
        return self._spec.return_vectors

    @property
    def return_distances(self) -> bool:
        """Return whether pair distances are requested."""
        return self._spec.return_distances

    @property
    def span_margin(self) -> float:
        """Return the synthesized-cell per-axis span-growth allowance."""
        return self._spec.span_margin

    @property
    def supports_compilation(self) -> bool:
        """Return whether this prepared configuration is compilation eligible."""
        return self._spec.supports_compilation

    @property
    def compilation_blocker(self) -> str | None:
        """Return the known compilation blocker, if any."""
        return self._spec.compilation_blocker

    @property
    def initialized(self) -> jax.Array:
        """Return per-system result initialization status."""
        return self._leaves[_INITIALIZED_LEAF]

    @property
    def valid(self) -> jax.Array:
        """Return sticky per-system validity history."""
        return self._leaves[_VALID_LEAF]

    _failure_code = property(lambda state: state._leaves[_FAILURE_CODE_LEAF])
    _required = property(lambda state: state._leaves[_REQUIRED_LEAF])
    _capacity = property(lambda state: state._leaves[_CAPACITY_LEAF])
    _first_failure = property(lambda state: state._leaves[_FIRST_FAILURE_LEAF])
    # The optional fixed partial-row map is deliberately a private PyTree leaf:
    # it participates in JIT signatures but cannot be changed at execution.
    _target_indices = property(lambda state: state._leaves[_TARGET_INDICES_LEAF])
    _shift_range_per_dimension = property(
        lambda state: state._leaves[_SHIFT_RANGE_LEAF]
    )
    _num_shifts_per_system = property(lambda state: state._leaves[_NUM_SHIFTS_LEAF])
    _span_capacity = property(lambda state: state._leaves[_SPAN_CAPACITY_LEAF])
    _span_axis = property(lambda state: state._leaves[_SPAN_AXIS_LEAF])
    _span_required = property(lambda state: state._leaves[_SPAN_REQUIRED_LEAF])

    neighbor_matrix = _result_property("neighbor_matrix")
    num_neighbors = _result_property("num_neighbors")
    neighbor_matrix_shifts = _result_property("neighbor_matrix_shifts")
    neighbor_list = _result_property("neighbor_list")
    neighbor_ptr = _result_property("neighbor_ptr")
    neighbor_list_shifts = _result_property("neighbor_list_shifts")
    neighbor_matrix1 = _result_property("neighbor_matrix1")
    num_neighbors1 = _result_property("num_neighbors1")
    neighbor_matrix_shifts1 = _result_property("neighbor_matrix_shifts1")
    neighbor_list1 = _result_property("neighbor_list1")
    neighbor_ptr1 = _result_property("neighbor_ptr1")
    neighbor_list_shifts1 = _result_property("neighbor_list_shifts1")
    neighbor_matrix2 = _result_property("neighbor_matrix2")
    num_neighbors2 = _result_property("num_neighbors2")
    neighbor_matrix_shifts2 = _result_property("neighbor_matrix_shifts2")
    neighbor_list2 = _result_property("neighbor_list2")
    neighbor_ptr2 = _result_property("neighbor_ptr2")
    neighbor_list_shifts2 = _result_property("neighbor_list_shifts2")
    neighbor_vectors = _result_property("neighbor_vectors")
    neighbor_distances = _result_property("neighbor_distances")
    pair_energies = _result_property("pair_energies")
    pair_forces = _result_property("pair_forces")
    metadata_valid = _result_property("metadata_valid")
    metadata_valid1 = _result_property("metadata_valid1")
    metadata_valid2 = _result_property("metadata_valid2")
    pair_offsets = _result_property("pair_offsets")
    pair_counts = _result_property("pair_counts")
    tile_offsets = _result_property("tile_offsets")
    tile_counts = _result_property("tile_counts")
    num_tiles = _result_property("num_tiles")
    tile_row_group = _result_property("tile_row_group")
    tile_col_group = _result_property("tile_col_group")
    tile_system = _result_property("tile_system")
    sorted_atom_index = _result_property("sorted_atom_index")
    sorted_pos_x = _result_property("sorted_pos_x")
    sorted_pos_y = _result_property("sorted_pos_y")
    sorted_pos_z = _result_property("sorted_pos_z")
    batch_idx_sorted = _result_property("batch_idx_sorted")
    batch_ptr_padded = _result_property("batch_ptr_padded")
    group_ptr = _result_property("group_ptr")


@jax.jit
def _update_cluster_lifecycle(
    initialized,
    valid,
    failure_code,
    required,
    capacity,
    first_failure_marker,
    active,
    uninitialized_preserved,
    tile_overflow,
    row_overflow,
    coo_overflow,
    tile_required,
    tile_capacity,
    row_required,
    row_capacity,
    coo_required,
    coo_capacity,
) -> tuple[jax.Array, ...]:
    """Update the cluster-only sticky status in one compiled device call."""
    failed = (
        (tile_overflow | row_overflow | coo_overflow) & active
    ) | uninitialized_preserved
    current_code = jnp.where(
        uninitialized_preserved,
        6,
        jnp.where(
            tile_overflow,
            4,
            jnp.where(row_overflow, 1, jnp.where(coo_overflow, 2, 0)),
        ),
    ).astype(jnp.int32)
    status_updated = active | uninitialized_preserved
    first_failure = valid & failed & status_updated
    return (
        initialized | active,
        jnp.where(status_updated, valid & ~failed, valid),
        jnp.where(first_failure, current_code, failure_code),
        jnp.where(
            first_failure,
            jnp.where(
                current_code == 4,
                tile_required,
                jnp.where(current_code == 1, row_required, coo_required),
            ),
            required,
        ),
        jnp.where(
            first_failure,
            jnp.where(
                current_code == 4,
                tile_capacity,
                jnp.where(current_code == 1, row_capacity, coo_capacity),
            ),
            capacity,
        ),
        jnp.where(first_failure, 1, first_failure_marker).astype(jnp.int32),
    )


def _validate(
    positions: jax.Array,
    cell: jax.Array | None,
    pbc: jax.Array | None,
    batch_ptr: jax.Array | None,
) -> None:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if pbc is not None and cell is None:
        raise ValueError("cell is required when pbc is provided")
    if batch_ptr is not None and batch_ptr.shape[0] < 2:
        raise ValueError("batch_ptr must have length at least 2")
    if cell is not None and cell.ndim not in (2, 3):
        raise ValueError("cell must have shape (3, 3) or (num_systems, 3, 3)")


def _validate_span_margin(value: int | float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("span_margin must be a finite nonnegative scalar")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("span_margin must be a finite nonnegative scalar")
    return value


def _validate_atomic_density(value: Any) -> float | None:
    """Validate and normalize the optional Python scalar density hint."""
    if value is None:
        return None
    if type(value) not in (int, float):
        raise ValueError("atomic_density must be a finite positive Python int or float")
    try:
        density = float(value)
    except OverflowError as exc:
        raise ValueError(
            "atomic_density must be a finite positive Python int or float"
        ) from exc
    if not math.isfinite(density) or density <= 0:
        raise ValueError("atomic_density must be a finite positive Python int or float")
    return density


def _batch_indices(
    batch_idx: jax.Array | None,
    batch_ptr: jax.Array | None,
    num_systems: int,
    num_atoms: int,
) -> jax.Array:
    if batch_idx is not None:
        return batch_idx.astype(jnp.int32)
    if batch_ptr is not None:
        return jnp.repeat(
            jnp.arange(num_systems, dtype=jnp.int32), batch_ptr[1:] - batch_ptr[:-1]
        )
    return jnp.zeros((num_atoms,), dtype=jnp.int32)


def _fixed_cell_inverse(
    cell: jax.Array,
    positions: jax.Array,
    *,
    method: str,
    strategy: str,
    num_systems: int,
) -> jax.Array:
    """Prepare the route-precision inverse carried by a fixed-cell state.

    Cell-list count/binning and tile-cooperative naive launchers use Warp's
    ``wp.inverse`` implementation. Scalar JAX paths, including cluster
    geometry and pair-output wrappers, use ``jnp.linalg.inv``.
    """
    dtype = positions.dtype if positions.dtype == jnp.float64 else jnp.float32
    cell_batch = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]
    if method.startswith("batch_") and cell_batch.shape[0] == 1 and num_systems > 1:
        cell_batch = jnp.broadcast_to(cell_batch, (num_systems, 3, 3))
    cell_batch = cell_batch.astype(dtype)
    uses_warp_inverse = "cell_list" in method or (
        method in {"naive", "batch_naive"} and strategy == "tile"
    )
    if uses_warp_inverse:
        inverse = jnp.zeros_like(cell_batch)
        (inverse,) = _FIXED_CELL_INVERSE_REGISTRATION[dtype](
            cell_batch,
            inverse,
            launch_dims=(num_systems,),
        )
    else:
        inverse = jnp.linalg.inv(cell_batch)
    return jax.lax.stop_gradient(inverse)


def _synthetic_geometry(
    positions: jax.Array,
    batch_idx: jax.Array,
    batch_ptr: jax.Array | None,
    num_systems: int,
    span_capacity: jax.Array,
    cutoff: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Translate each nonperiodic system and return its fixed prepared cell."""
    if positions.shape[0] == 0:
        shifted = positions
        minimum = jnp.zeros((num_systems, 3), dtype=positions.dtype)
        maximum = jnp.zeros((num_systems, 3), dtype=positions.dtype)
        spans = jnp.zeros((num_systems, 3), dtype=positions.dtype)
    elif num_systems == 1:
        minimum = jnp.min(positions, axis=0)
        maximum = jnp.max(positions, axis=0)
        shifted = positions - minimum
        spans = (maximum - minimum).reshape(1, 3)
        minimum = minimum.reshape(1, 3)
        maximum = maximum.reshape(1, 3)
    else:
        minimum = jax.ops.segment_min(positions, batch_idx, num_segments=num_systems)
        maximum = jax.ops.segment_max(positions, batch_idx, num_segments=num_systems)
        counts = (
            batch_ptr[1:] - batch_ptr[:-1]
            if batch_ptr is not None
            else jnp.bincount(batch_idx, length=num_systems)
        )
        nonempty = counts[:, None] > 0
        minimum = jnp.where(nonempty, minimum, jnp.zeros_like(minimum))
        maximum = jnp.where(nonempty, maximum, jnp.zeros_like(maximum))
        spans = jnp.where(nonempty, maximum - minimum, jnp.zeros_like(minimum))
        shifted = positions - minimum[batch_idx]
    cell_lengths = span_capacity + jnp.asarray(0.1 * cutoff, dtype=positions.dtype)
    headroom = jnp.maximum(cell_lengths - span_capacity, 0)
    magnitude = jnp.maximum(
        jnp.maximum(jnp.abs(minimum), jnp.abs(maximum)), span_capacity
    )
    tolerance = jnp.minimum(
        jnp.asarray(4 * jnp.finfo(positions.dtype).eps, dtype=positions.dtype)
        * magnitude,
        jnp.asarray(0.5, dtype=positions.dtype) * headroom,
    )
    span_failure_axes = ~(
        (spans <= span_capacity)
        | ((spans - span_capacity <= tolerance) & (spans <= cell_lengths))
    )
    cell = cell_lengths[:, :, None] * jnp.eye(3, dtype=positions.dtype)
    return (
        shifted,
        cell if num_systems > 1 else cell.reshape(1, 3, 3),
        spans,
        span_failure_axes,
    )


def _prepare_cluster_state(
    positions: jax.Array,
    cell: jax.Array,
    pbc: jax.Array,
    batch_idx: jax.Array | None,
    batch_ptr: jax.Array | None,
    *,
    method: str,
    systems: int,
    num_atoms: int,
    cutoff: float,
    cutoff2: float | None,
    half_fill: bool,
    wrap_positions: bool,
    span_margin: float,
    fixed_cell: bool,
    format: str,
    selective: bool,
    fill_value: int,
    kwargs: dict[str, Any],
    capacity: int | tuple[int, int] | None,
) -> NeighborListState:
    """Allocate the fixed buffers and spec for a prepared cluster route."""
    max_tiles_per_group = int(
        kwargs.get("max_tiles_per_group", max(1, (num_atoms + 31) // 32))
    )
    if max_tiles_per_group <= 0:
        raise ValueError("max_tiles_per_group must be positive")
    max_neighbors = int(
        kwargs.get(
            "max_neighbors1" if cutoff2 is not None else "max_neighbors", num_atoms
        )
    )
    tile_offsets = None
    if format == "coo":
        capacity = int(capacity if capacity is not None else num_atoms * max_neighbors)
    else:
        capacity = None
    if method == "cluster_tile":
        groups = max(1, (num_atoms + 31) // 32)
        tile_capacity = groups * min(groups, max_tiles_per_group)
    else:
        counts = jnp.asarray(batch_ptr[1:] - batch_ptr[:-1], dtype=jnp.int32)
        groups = (counts + 31) // 32
        tile_capacities = groups * jnp.minimum(groups, max_tiles_per_group)
        tile_offsets = jnp.concatenate(
            (
                jnp.zeros((1,), jnp.int32),
                jnp.cumsum(tile_capacities, dtype=jnp.int32),
            )
        )
        tile_capacity = int(tile_offsets[-1])
        if (
            selective
            and format == "coo"
            and int(capacity) != int(jnp.sum(counts) * max_neighbors)
        ):
            raise ValueError(
                "batch selective cluster-tile COO capacity must equal total_atoms * max_neighbors"
            )
    if format == "tile":
        names = (
            (
                "num_tiles",
                "tile_row_group",
                "tile_col_group",
                "sorted_atom_index",
                "sorted_pos_x",
                "sorted_pos_y",
                "sorted_pos_z",
            )
            if method == "cluster_tile"
            else (
                "num_tiles",
                "tile_row_group",
                "tile_col_group",
                "tile_system",
                "sorted_atom_index",
                "sorted_pos_x",
                "sorted_pos_y",
                "sorted_pos_z",
                "batch_idx_sorted",
                "batch_ptr_padded",
                "group_ptr",
            )
        )
    elif format == "coo" and selective:
        names = (
            (
                "neighbor_list",
                "pair_offsets",
                "pair_counts",
                "neighbor_list_shifts",
                "num_tiles",
                "tile_row_group",
                "tile_col_group",
            )
            if method == "cluster_tile"
            else (
                "neighbor_list",
                "pair_offsets",
                "pair_counts",
                "neighbor_list_shifts",
                "tile_offsets",
                "tile_counts",
                "num_tiles",
                "tile_row_group",
                "tile_col_group",
                "tile_system",
            )
        )
    elif format == "coo":
        names = ("neighbor_list", "neighbor_ptr", "neighbor_list_shifts")
        if kwargs.get("return_distances"):
            names += ("neighbor_distances",)
        if kwargs.get("return_vectors"):
            names += ("neighbor_vectors",)
        if kwargs.get("pair_fn") is not None:
            names += ("pair_energies", "pair_forces")
    elif cutoff2 is not None:
        names = (
            "neighbor_matrix",
            "num_neighbors",
            "neighbor_matrix_shifts",
            "neighbor_matrix2",
            "num_neighbors2",
            "neighbor_matrix_shifts2",
        )
        if selective:
            names += (
                ("num_tiles", "tile_row_group", "tile_col_group")
                if method == "cluster_tile"
                else (
                    "tile_offsets",
                    "tile_counts",
                    "num_tiles",
                    "tile_row_group",
                    "tile_col_group",
                    "tile_system",
                )
            )
    else:
        names = ("neighbor_matrix", "num_neighbors", "neighbor_matrix_shifts")
        if kwargs.get("return_distances"):
            names += ("neighbor_distances",)
        if kwargs.get("return_vectors"):
            names += ("neighbor_vectors",)
        if kwargs.get("pair_fn") is not None:
            names += ("pair_energies", "pair_forces")
        if selective:
            names += (
                ("num_tiles", "tile_row_group", "tile_col_group")
                if method == "cluster_tile"
                else (
                    "tile_offsets",
                    "tile_counts",
                    "num_tiles",
                    "tile_row_group",
                    "tile_col_group",
                    "tile_system",
                )
            )
    outputs = [jnp.zeros((0,), dtype=jnp.int32) for _ in range(12)]
    output_index = {name: i for i, name in enumerate(names)}
    if selective:
        if "neighbor_matrix" in output_index:
            outputs[output_index["neighbor_matrix"]] = jnp.full(
                (num_atoms, max_neighbors), fill_value, jnp.int32
            )
        if "num_neighbors" in output_index:
            outputs[output_index["num_neighbors"]] = jnp.zeros((num_atoms,), jnp.int32)
        if "neighbor_matrix_shifts" in output_index:
            outputs[output_index["neighbor_matrix_shifts"]] = jnp.zeros(
                (num_atoms, max_neighbors, 3), jnp.int32
            )
        if cutoff2 is not None:
            width2 = int(kwargs.get("max_neighbors2", num_atoms))
            outputs[output_index["neighbor_matrix2"]] = jnp.full(
                (num_atoms, width2), fill_value, jnp.int32
            )
            outputs[output_index["num_neighbors2"]] = jnp.zeros((num_atoms,), jnp.int32)
            outputs[output_index["neighbor_matrix_shifts2"]] = jnp.zeros(
                (num_atoms, width2, 3), jnp.int32
            )
        elif format == "coo":
            outputs[output_index["neighbor_list"]] = jnp.zeros(
                (2, int(capacity)), jnp.int32
            )
            outputs[output_index["pair_offsets"]] = (
                jnp.asarray([0, int(capacity)], jnp.int32)
                if method == "cluster_tile"
                else jnp.concatenate(
                    (
                        jnp.zeros((1,), jnp.int32),
                        jnp.cumsum(
                            (batch_ptr[1:] - batch_ptr[:-1]) * max_neighbors,
                            dtype=jnp.int32,
                        ),
                    )
                )
            )
            outputs[output_index["pair_counts"]] = jnp.zeros((systems,), jnp.int32)
            outputs[output_index["neighbor_list_shifts"]] = jnp.zeros(
                (int(capacity), 3), jnp.int32
            )
        outputs[output_index["num_tiles"]] = jnp.zeros((1,), jnp.int32)
        outputs[output_index["tile_row_group"]] = jnp.zeros((tile_capacity,), jnp.int32)
        outputs[output_index["tile_col_group"]] = jnp.zeros((tile_capacity,), jnp.int32)
        if method == "batch_cluster_tile":
            outputs[output_index["tile_offsets"]] = tile_offsets
            outputs[output_index["tile_counts"]] = jnp.zeros((systems,), jnp.int32)
            outputs[output_index["tile_system"]] = jnp.zeros(
                (tile_capacity,), jnp.int32
            )
    elif format == "matrix":
        outputs[output_index["neighbor_matrix"]] = jnp.full(
            (num_atoms, max_neighbors), fill_value, jnp.int32
        )
        outputs[output_index["num_neighbors"]] = jnp.zeros((num_atoms,), jnp.int32)
        outputs[output_index["neighbor_matrix_shifts"]] = jnp.zeros(
            (num_atoms, max_neighbors, 3), jnp.int32
        )
        if cutoff2 is not None:
            width2 = int(kwargs.get("max_neighbors2", num_atoms))
            outputs[output_index["neighbor_matrix2"]] = jnp.full(
                (num_atoms, width2), fill_value, jnp.int32
            )
            outputs[output_index["num_neighbors2"]] = jnp.zeros((num_atoms,), jnp.int32)
            outputs[output_index["neighbor_matrix_shifts2"]] = jnp.zeros(
                (num_atoms, width2, 3), jnp.int32
            )
    route_options = (
        ("max_tiles_per_group", max_tiles_per_group),
        ("max_pairs", capacity),
        ("tile_capacity", tile_capacity),
    )
    shared_batch_cell = method == "batch_cluster_tile" and cell.ndim == 2
    prepared_cell = (
        jnp.broadcast_to(cell, (systems, 3, 3)) if shared_batch_cell else cell
    )
    fixed_cell_geometry = None
    if fixed_cell:
        from nvalchemiops.jax.neighbors._cluster_tile_preload import (
            _preload_cluster_tile_build_kernel,
        )

        cell_batch = prepared_cell if prepared_cell.ndim == 3 else prepared_cell[None]
        cell_batch = jax.lax.stop_gradient(cell_batch.astype(jnp.float32))
        inverse = jax.lax.stop_gradient(jnp.linalg.inv(cell_batch))
        _preload_cluster_tile_build_kernel(device_source=cell_batch, fixed_cell=True)
        qr_values = jnp.zeros((systems, 15), dtype=jnp.float32)
        axis_aligned = jnp.zeros((systems,), dtype=jnp.bool_)
        fractional_certified = jnp.zeros((systems,), dtype=jnp.bool_)
        height_certified = jnp.zeros((systems,), dtype=jnp.bool_)
        bbox_cutoff_bounds = jnp.zeros((systems, 3), dtype=jnp.float32)
        build_cutoff = cutoff if cutoff2 is None else max(cutoff, cutoff2)
        outer_cutoff_sq = float(
            jnp.asarray(build_cutoff * build_cutoff, dtype=jnp.float32)
        )
        (
            qr_values,
            axis_aligned,
            fractional_certified,
            height_certified,
            bbox_cutoff_bounds,
        ) = _FIXED_CLUSTER_GEOMETRY_REGISTRATION[jnp.float32](
            cell_batch,
            inverse,
            outer_cutoff_sq,
            qr_values,
            axis_aligned,
            fractional_certified,
            height_certified,
            bbox_cutoff_bounds,
            launch_dims=(systems,),
        )
        fixed_cell_geometry = (
            inverse,
            None,
            None,
            None,
            None,
            jax.lax.stop_gradient(qr_values),
            jax.lax.stop_gradient(axis_aligned),
            jax.lax.stop_gradient(fractional_certified),
            jax.lax.stop_gradient(height_certified),
            jax.lax.stop_gradient(bbox_cutoff_bounds),
        )
    spec = _Spec(
        method=method,
        strategy="auto",
        format=format,
        coo_layout=(
            "segmented"
            if format == "coo" and selective
            else ("compact" if format == "coo" else None)
        ),
        is_batched=method.startswith("batch_"),
        num_atoms=num_atoms,
        num_systems=systems,
        cutoff=float(cutoff),
        cutoff2=None if cutoff2 is None else float(cutoff2),
        half_fill=bool(half_fill),
        fill_value=fill_value,
        wrap_positions=bool(wrap_positions),
        selective=selective,
        return_vectors=bool(kwargs.get("return_vectors", False)),
        return_distances=bool(kwargs.get("return_distances", False)),
        span_margin=float(span_margin),
        synthesized_cell=False,
        fixed_cell=fixed_cell,
        supports_compilation=(
            method != "batch_cluster_tile" and not (format == "coo" and not selective)
        ),
        compilation_blocker=(
            "batch cluster-tile sizing requires concrete batch_ptr"
            if method == "batch_cluster_tile"
            else (
                "compact cluster-tile COO uses eager shape compaction"
                if format == "coo" and not selective
                else None
            )
        ),
        pair_fn=kwargs.get("pair_fn"),
        result_names=tuple(names),
        coo_capacity=capacity,
        max_neighbors=(
            max_neighbors,
            int(kwargs.get("max_neighbors2", num_atoms))
            if cutoff2 is not None
            else None,
        ),
        max_shifts_per_system=None,
        max_atoms_per_system=None,
        has_target_indices=False,
        shared_batch_cell=shared_batch_cell,
        route_options=route_options,
    )
    leaves = _StateLeaves.build(
        cell=prepared_cell,
        pbc=pbc,
        batch_idx=(
            _batch_indices(batch_idx, batch_ptr, systems, num_atoms)
            if method.startswith("batch_")
            else batch_idx
        ),
        batch_ptr=batch_ptr,
        fixed_cell_geometry=fixed_cell_geometry,
        result=outputs,
        shift_range_per_dimension=jnp.empty((0, 3), dtype=jnp.int32),
        num_shifts_per_system=jnp.empty((0,), dtype=jnp.int32),
        target_indices=jnp.empty((0,), dtype=jnp.int32),
        span_capacity=jnp.zeros((systems, 3), positions.dtype),
        span_axis=jnp.zeros((systems,), jnp.int32),
        span_required=jnp.zeros((systems,), positions.dtype),
        initialized=jnp.zeros((systems,), bool),
        valid=jnp.ones((systems,), bool),
        failure_code=jnp.zeros((systems,), jnp.int32),
        required=jnp.zeros((systems,), jnp.int32),
        capacity=jnp.zeros((systems,), jnp.int32),
        first_failure=jnp.zeros((systems,), jnp.int32),
        private_matrix1=jnp.empty((0,), dtype=jnp.int32),
        private_num1=jnp.empty((0,), dtype=jnp.int32),
        private_shift1=jnp.empty((0,), dtype=jnp.int32),
        private_matrix2=jnp.empty((0,), dtype=jnp.int32),
        private_num2=jnp.empty((0,), dtype=jnp.int32),
        private_shift2=jnp.empty((0,), dtype=jnp.int32),
    )
    return NeighborListState(spec, leaves, token=_TOKEN)


@dataclass(frozen=True)
class _PreparedResolution:
    """Resolved fixed route choices shared by prepared-state construction."""

    method: str
    kwargs: dict[str, Any]
    systems: int
    num_atoms: int
    dual: bool
    cluster: bool
    format: str
    capacity: int | tuple[int, int] | None
    fill_value: int
    target_indices: jax.Array | None
    num_rows: int


def _resolve_prepared_route(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array | None,
    pbc: jax.Array | None,
    batch_idx: jax.Array | None,
    batch_ptr: jax.Array | None,
    cutoff2: float | None,
    half_fill: bool,
    fill_value: int | None,
    return_neighbor_list: bool,
    method: str | None,
    wrap_positions: bool,
    selective: bool,
    span_margin: float,
    kwargs: dict[str, Any],
) -> _PreparedResolution:
    """Validate and resolve a prepared route without allocating its state."""
    automatic_method = method is None
    _validate(positions, cell, pbc, batch_ptr)
    if not isinstance(selective, bool):
        raise ValueError("selective must be a Boolean")
    if cutoff is None or cutoff <= 0:
        raise ValueError("cutoff must be positive")
    span_margin = _validate_span_margin(span_margin)
    if method is None:
        # Resolve the same eager strategy decomposition as ``neighbor_list``
        # now, while preparation is intentionally eager.  A prepared call must
        # never repeat this host-side selection under ``jit``.
        from nvalchemiops.jax.neighbors._dispatch import _auto_method_from_geometry
        from nvalchemiops.neighbors.base_dispatch import neighbor_list_strategy_run_args

        if batch_ptr is not None:
            inferred_systems = int(batch_ptr.shape[0] - 1)
        elif batch_idx is not None and batch_idx.size:
            inferred_systems = int(jnp.max(batch_idx)) + 1
        else:
            inferred_systems = 1
        strategy_name = _auto_method_from_geometry(
            positions,
            max(
                float(cutoff), float(cutoff2) if cutoff2 is not None else float(cutoff)
            ),
            cell,
            pbc,
            batch_idx,
            batch_ptr,
            inferred_systems,
            cutoff2=cutoff2,
            half_fill=half_fill,
            return_neighbor_list=return_neighbor_list,
            target_indices=kwargs.get("target_indices"),
            return_vectors=bool(kwargs.get("return_vectors", False)),
            return_distances=bool(kwargs.get("return_distances", False)),
            use_pair_fn=kwargs.get("pair_fn") is not None,
            rebuild_flags=(
                jnp.ones((inferred_systems,), dtype=jnp.bool_) if selective else None
            ),
            wrap_positions=wrap_positions,
        )
        base, native, cell_strategy, path = neighbor_list_strategy_run_args(
            strategy_name
        )
        method = (
            "batch_" if batch_idx is not None or batch_ptr is not None else ""
        ) + base
        kwargs = {
            **kwargs,
            "strategy": native if base == "naive" else cell_strategy,
            "atom_centric_path": path,
        }
    if batch_idx is not None or batch_ptr is not None:
        # Match the eager dispatcher for explicit methods: a batch-owned
        # geometry always selects the batched binding, including callers that
        # spell the method with its single-system name.  Do this before dual
        # cutoff normalization so ``naive_dual_cutoff`` becomes exactly
        # ``batch_naive_dual_cutoff`` rather than receiving a second prefix.
        method = {
            "naive": "batch_naive",
            "naive_dual_cutoff": "batch_naive_dual_cutoff",
            "cell_list": "batch_cell_list",
            "cluster_tile": "batch_cluster_tile",
        }.get(method, method)
    # Cluster-tile owns its dual-matrix implementation.  Only naive routes
    # are renamed to their distinct dual-cutoff entry points.
    if (
        cutoff2 is not None
        and "dual_cutoff" not in method
        and method not in {"cluster_tile", "batch_cluster_tile"}
    ):
        method = (
            "batch_naive_dual_cutoff"
            if method.startswith("batch_")
            else "naive_dual_cutoff"
        )
    if method not in {
        "naive",
        "batch_naive",
        "naive_dual_cutoff",
        "batch_naive_dual_cutoff",
        "cell_list",
        "batch_cell_list",
        "cluster_tile",
        "batch_cluster_tile",
    }:
        raise NotImplementedError(
            f"unsupported prepared JAX neighbor-list method {method!r}"
        )
    if (
        automatic_method
        and selective
        and method
        in {"naive", "batch_naive", "naive_dual_cutoff", "batch_naive_dual_cutoff"}
        and kwargs.get("strategy") == "tile"
    ):
        # The automatic selector may rank the direct tiled kernel first for a
        # small geometry. Selective execution has no tiled kernel, so resolve
        # that automatic choice to the supported scalar implementation while
        # retaining the explicit strategy='tile' rejection below.
        kwargs = {**kwargs, "strategy": "scalar"}
    if cutoff2 is None and "dual_cutoff" in method:
        raise ValueError("cutoff2 must be provided for a dual-cutoff prepared state")
    if cutoff2 is not None and float(cutoff) > float(cutoff2):
        raise ValueError(
            "cutoff must not exceed cutoff2 for a dual-cutoff prepared state"
        )
    unknown = set(kwargs) - _CONFIG_KEYS
    if unknown:
        raise TypeError(f"unexpected prepared neighbor-list options: {sorted(unknown)}")
    if kwargs.get("pair_params") is not None:
        raise ValueError(
            "pair_params is a runtime input; do not pass it to prepare_neighbor_list"
        )
    if method in {
        "naive",
        "batch_naive",
        "naive_dual_cutoff",
        "batch_naive_dual_cutoff",
    } and kwargs.get("strategy", "auto") not in {"auto", "scalar", "tile"}:
        raise ValueError("unsupported naive strategy")
    if (
        method
        in {
            "naive",
            "batch_naive",
            "naive_dual_cutoff",
            "batch_naive_dual_cutoff",
        }
        and kwargs.get("strategy", "auto") == "auto"
    ):
        # JAX's direct naive ``auto`` policy is scalar. Store the resolved
        # choice so prepared execution does not repeat strategy selection.
        kwargs = {**kwargs, "strategy": "scalar"}
    if method in {"cell_list", "batch_cell_list"}:
        from nvalchemiops.jax.neighbors._resolution import resolve_cell_strategy

        selected_cell_strategy = kwargs.get("strategy", "auto")
        automatic_cell_strategy = selected_cell_strategy == "auto"
        selected_cell_strategy = resolve_cell_strategy(
            selected_cell_strategy,
            total_atoms=int(positions.shape[0]),
            cutoff=float(cutoff),
            device_is_cpu=all(
                device.platform == "cpu" for device in positions.devices()
            ),
            half_fill=bool(half_fill),
        )
        if (
            automatic_cell_strategy
            and selected_cell_strategy == "pair_centric"
            and (
                kwargs.get("return_vectors")
                or kwargs.get("return_distances")
                or kwargs.get("pair_fn") is not None
            )
        ):
            # The atom-centric equivalent is already compilation eligible;
            # prefer it for automatic preparation with pair outputs.
            selected_cell_strategy = "atom_centric"
        kwargs = {**kwargs, "strategy": selected_cell_strategy}
    if batch_ptr is not None:
        systems = int(batch_ptr.shape[0] - 1)
    elif batch_idx is not None and batch_idx.size:
        systems = int(jnp.max(batch_idx)) + 1
    else:
        systems = 1
    n = int(positions.shape[0])
    dual = cutoff2 is not None
    if (
        method in {"cell_list", "batch_cell_list"}
        and return_neighbor_list
        and kwargs.get("coo_capacity") is None
    ):
        raise ValueError("coo_capacity is required for a prepared fixed-COO state")
    cluster = method in {"cluster_tile", "batch_cluster_tile"}
    derived_format = "coo" if return_neighbor_list else "matrix"
    explicit_format = kwargs.get("format")
    if (
        explicit_format is not None
        and explicit_format != derived_format
        and not (explicit_format == "tile" and not return_neighbor_list and cluster)
    ):
        raise ValueError(
            "format contradicts return_neighbor_list; omit format or use "
            f"format={derived_format!r}"
        )
    fmt = derived_format if explicit_format is None else explicit_format
    if fmt not in {"matrix", "coo", "tile"}:
        raise ValueError("format must be one of 'matrix', 'coo', or 'tile'")
    if fmt == "tile" and not cluster:
        raise ValueError("format='tile' requires a cluster_tile method")
    if (
        automatic_method
        and cluster
        and cutoff2 is None
        and fmt != "tile"
        and (fmt == "matrix" or kwargs.get("coo_capacity") is not None)
    ):
        # The selector's cluster result is performance-oriented.  Prepared
        # execution instead prefers the existing compiled atom-centric cell
        # route when its fixed output representation is equivalent.
        method = "batch_cell_list" if method.startswith("batch_") else "cell_list"
        kwargs = {**kwargs, "strategy": "atom_centric"}
        cluster = False
    capacity = kwargs.get("coo_capacity")
    if fmt == "coo" and capacity is None and not cluster:
        raise ValueError("coo_capacity is required for a prepared fixed-COO state")
    if not cluster and capacity is not None:
        from nvalchemiops.jax.neighbors.neighbor_utils import _validate_coo_capacities

        normalized_capacity = _validate_coo_capacities(
            capacity, fmt == "coo", num_cutoffs=2 if dual else 1
        )
        capacity = normalized_capacity if dual else normalized_capacity[0]
    naive_method = method in {
        "naive",
        "batch_naive",
        "naive_dual_cutoff",
        "batch_naive_dual_cutoff",
    }
    if (
        selective
        and fmt != "matrix"
        and not cluster
        and not (naive_method and fmt == "coo")
    ):
        raise NotImplementedError(
            "prepared selective execution supports matrix routes only"
        )
    target_indices = kwargs.get("target_indices")
    if target_indices is not None:
        target_indices = jnp.asarray(target_indices, dtype=jnp.int32)
        if target_indices.ndim != 1:
            raise ValueError("target_indices must be a one-dimensional array")
    if method in {"naive", "batch_naive"} and kwargs.get("strategy") == "tile":
        if all(device.platform == "cpu" for device in positions.devices()):
            raise ValueError(
                "strategy='tile' requires CUDA; the tile-cooperative naive kernel cannot run on a CPU device (Warp forces block_dim=1). Use strategy='scalar' or 'auto' on CPU."
            )
        if (
            kwargs.get("return_distances")
            or kwargs.get("return_vectors")
            or kwargs.get("pair_fn") is not None
        ):
            raise NotImplementedError(
                "strategy='tile' has no pair-output (return_distances / return_vectors / pair_fn) variant; use strategy='scalar'."
            )
        if target_indices is not None:
            raise NotImplementedError(
                "strategy='tile' has no target_indices (partial neighbor-list) variant; use strategy='scalar'."
            )
        if selective:
            raise NotImplementedError(
                "strategy='tile' has no selective (rebuild_flags) variant; use strategy='scalar'."
            )
        if method == "batch_naive" and pbc is not None and not wrap_positions:
            raise NotImplementedError(
                "strategy='tile' has no batched prewrapped-PBC tiled kernel (wrap_positions=False with PBC). Use strategy='scalar', or wrap_positions=True for the tile path."
            )
    if selective and (
        target_indices is not None
        or kwargs.get("return_vectors")
        or kwargs.get("return_distances")
        or kwargs.get("pair_fn") is not None
    ):
        raise NotImplementedError(
            "partial and pair-output prepared states do not support selective execution"
        )
    if cutoff2 is not None and (
        target_indices is not None
        or kwargs.get("pair_fn") is not None
        or kwargs.get("return_vectors")
        or kwargs.get("return_distances")
        or (cluster and fmt != "matrix")
    ):
        raise NotImplementedError(
            "dual-cutoff prepared states do not support target_indices or pair outputs"
        )
    if (
        method in {"cell_list", "batch_cell_list"}
        and kwargs.get("strategy") == "pair_centric"
        and target_indices is not None
    ):
        raise NotImplementedError(
            "strategy='pair_centric' with target_indices (partial neighbor lists) is not supported"
        )
    if cluster:
        if pbc is None or not bool(jnp.all(pbc)):
            raise NotImplementedError(
                "cluster_tile prepared states require fully periodic pbc"
            )
        if half_fill:
            raise NotImplementedError(
                "cluster_tile prepared states do not support half_fill"
            )
        if target_indices is not None:
            raise NotImplementedError(
                "cluster_tile prepared states do not support target_indices"
            )
        if selective and fmt == "tile":
            raise NotImplementedError(
                "cluster_tile tile output does not support selective execution"
            )
        if (fmt == "tile" or selective) and (
            kwargs.get("return_vectors")
            or kwargs.get("return_distances")
            or kwargs.get("pair_fn") is not None
        ):
            raise NotImplementedError(
                "cluster_tile geometry and pair outputs do not support tile or selective execution"
            )
    fv = n if fill_value is None else int(fill_value)
    num_rows = n if target_indices is None else int(target_indices.shape[0])
    return _PreparedResolution(
        method=method,
        kwargs=kwargs,
        systems=systems,
        num_atoms=n,
        dual=dual,
        cluster=cluster,
        format=fmt,
        capacity=capacity,
        fill_value=fv,
        target_indices=target_indices,
        num_rows=num_rows,
    )


def prepare_neighbor_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array | None = None,
    pbc: jax.Array | None = None,
    batch_idx: jax.Array | None = None,
    batch_ptr: jax.Array | None = None,
    cutoff2: float | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    method: str | None = None,
    format: str | None = None,
    wrap_positions: bool = True,
    *,
    max_neighbors: int | None = None,
    max_neighbors1: int | None = None,
    max_neighbors2: int | None = None,
    atomic_density: float | None = None,
    coo_capacity: int | tuple[int, int] | None = None,
    max_tiles_per_group: int | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    selective: bool = False,
    pair_fn: Any = None,
    span_margin: float = 0.0,
    target_indices: jax.Array | None = None,
    strategy: str = "auto",
    atom_centric_path: str = "auto",
    fixed_cell: bool = False,
    **kwargs: Any,
) -> NeighborListState:
    """Resolve and allocate a reusable JAX neighbor-list route.

    Preparation runs eagerly. It validates fixed metadata, resolves automatic
    method and strategy selection once, and allocates the arrays carried by
    the immutable state. Automatic selection may prefer an existing
    compilation-eligible equivalent; inspect ``state.method`` and
    ``state.strategy``, or pass explicit choices to pin the implementation.
    Execute the state with
    ``results, state = neighbor_list(..., state=state)``.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3)
        Exemplar positions. Atom count and order, shape, and dtype become part
        of the prepared contract; coordinate values may change at execution.
    cutoff : float
        Primary neighbor cutoff.
    cell : jax.Array, shape (3, 3) or (num_systems, 3, 3), optional
        Exemplar cell. If omitted for a resolved cell-list route, preparation
        creates a fixed nonperiodic bounding cell. Applicable cell values may
        change later without changing shape, dtype, device, or PBC pattern.
    pbc : jax.Array, shape (3,) or (num_systems, 3), dtype=bool, optional
        Fixed periodic-axis pattern associated with ``cell``.
    batch_idx, batch_ptr : jax.Array, dtype=int32, optional
        Fixed, system-contiguous batch ownership metadata.
    cutoff2 : float, optional
        Secondary cutoff for an existing dual-cutoff route. It must be greater
        than or equal to ``cutoff``.
    half_fill : bool, default=False
        Store one directed half of each pair where the route supports it.
    fill_value : int, optional
        Neighbor-matrix padding value. Defaults to ``total_atoms``.
    return_neighbor_list : bool, default=False
        Request the route's COO representation instead of matrix output.
    method : str, optional
        Method accepted by :func:`neighbor_list`, including batched and
        dual-cutoff names. ``None`` runs automatic selection once during
        preparation and may choose a compilation-eligible equivalent route.
    format : {"matrix", "coo", "tile"}, optional
        Fixed result representation. If omitted, matrix versus COO is derived
        from ``return_neighbor_list``. An explicit matrix or COO value must
        agree with that flag; tile output requires a nonselective cluster-tile
        route.
    wrap_positions : bool, default=True
        Whether applicable naive routes wrap positions before enumeration.
    max_neighbors : int, optional
        Single-cutoff row width, or the primary width where the direct route
        uses that spelling.
    max_neighbors1, max_neighbors2 : int, optional
        Independent primary and secondary row widths for dual-cutoff routes.
    atomic_density : float, optional
        Finite positive Python ``int`` or ``float`` in atoms per unit volume,
        using the same length units as positions and cutoffs. Used during
        preparation to estimate only omitted neighbor widths; explicit widths
        take precedence. ``None`` keeps the existing atom-count-based defaults.
        Ordinary dual-cutoff routes estimate each missing width at its cutoff;
        cluster routes use the outer cutoff with a 32-neighbor floor. These
        estimates are capacity heuristics.
    coo_capacity : int or tuple of int, optional
        Fixed JAX COO capacity, required for prepared non-cluster COO. A scalar
        applies to both dual-cutoff groups; a pair sets them independently.
    max_tiles_per_group : int, optional
        Static cluster-tile capacity bound.
    return_vectors, return_distances : bool, default=False
        Request the corresponding existing pair-geometry outputs.
    selective : bool, default=False
        Prepare a state for runtime per-system ``rebuild_flags``. The flags
        are execution inputs and are never stored in the prepared state.
    pair_fn : callable, optional
        Warp/JAX pair function using the direct neighbor-route callback
        contract. Runtime ``pair_params`` are supplied to
        :func:`neighbor_list`.
    span_margin : float, default=0.0
        Finite nonnegative per-axis span-growth allowance for a synthesized
        nonperiodic cell-list cell. Runtime span is bounded by exemplar span
        plus this value, with a small dtype-dependent rounding allowance
        capped by the synthesized cell's remaining headroom. Equally small
        real expansion can also pass. A nonzero value requires ``cell=None``
        and a resolved cell-list route. This is not a neighbor-list skin.
    fixed_cell : bool, default=False
        Reuse cell-dependent topology geometry prepared from ``cell``. When
        true, the caller promises that runtime cell values remain unchanged;
        this promise is not checked by comparing cell values.
    target_indices : jax.Array, shape (num_rows,), dtype=int32, optional
        Fixed compact source-row selection for routes that support partial
        rows.
    strategy, atom_centric_path : str, default="auto"
        Route-specific implementation choices resolved once during
        preparation. Explicit values are never silently substituted.
    **kwargs : Any
        Additional fixed options already accepted by the resolved direct
        route, such as cell-list sizing metadata. Runtime ``rebuild_flags``
        and ``pair_params`` are rejected during preparation; unknown names
        raise ``TypeError``.

    Returns
    -------
    NeighborListState
        Immutable state for repeated JAX execution.

    Notes
    -----
    ``supports_compilation`` reports route eligibility rather than compiler
    history or a guarantee for every environment. A non-``None``
    ``compilation_blocker`` explains known route ineligibility.

    Prepared state is a convenience interface. Method-specific functions with
    explicit buffers and launch metadata remain the lower-overhead JAX option
    when call latency is critical.

    Unsupported prepared combinations include selective pair outputs or
    partial rows on every route, selective cell-list COO, pair-centric cell
    lists with partial rows, and dual-cutoff pair outputs or partial rows.
    Explicit tiled-naive execution does not support selective execution, pair
    outputs, partial rows, or batched prewrapped PBC. Cluster-tile preparation
    follows the direct route's CUDA, float32, fully periodic, and
    ``half_fill=False`` prerequisites; it does not support partial rows,
    selective tile or pair output, or pair output with tile format.

    Prepared batched cluster-tile execution, nonselective compact cluster-tile
    COO, and pair-centric cell-list geometry or callback output are eager-only.

    See Also
    --------
    neighbor_list : Execute the prepared state and return its successor.
    NeighborListState : Inspect resolved configuration, results, and status.
    check_neighbor_list_state : Convert sticky device status to an exception.
    """
    atomic_density = _validate_atomic_density(atomic_density)
    if not isinstance(fixed_cell, bool):
        raise ValueError("fixed_cell must be a Python bool")
    if fixed_cell and cell is None:
        raise ValueError("fixed_cell=True requires an explicit cell")
    explicit_options = {
        "format": format,
        "max_neighbors": max_neighbors,
        "max_neighbors1": max_neighbors1,
        "max_neighbors2": max_neighbors2,
        "coo_capacity": coo_capacity,
        "max_tiles_per_group": max_tiles_per_group,
        "pair_fn": pair_fn,
        "target_indices": target_indices,
    }
    kwargs = {
        **kwargs,
        **{
            name: value for name, value in explicit_options.items() if value is not None
        },
        "return_vectors": return_vectors,
        "return_distances": return_distances,
        "span_margin": span_margin,
        "strategy": strategy,
        "atom_centric_path": atom_centric_path,
    }
    resolution = _resolve_prepared_route(
        positions,
        cutoff,
        cell,
        pbc,
        batch_idx,
        batch_ptr,
        cutoff2,
        half_fill,
        fill_value,
        return_neighbor_list,
        method,
        wrap_positions,
        selective,
        span_margin,
        kwargs,
    )
    method = resolution.method
    kwargs = dict(resolution.kwargs)
    systems = resolution.systems
    n = resolution.num_atoms
    cluster = resolution.cluster
    fmt = resolution.format
    capacity = resolution.capacity
    fv = resolution.fill_value
    if atomic_density is not None:
        from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors

        if cluster:
            if cutoff2 is None:
                if kwargs.get("max_neighbors") is None:
                    estimate = estimate_max_neighbors(
                        float(cutoff),
                        atomic_density=atomic_density,
                        max_neighbors_lower_bound=32,
                    )
                    kwargs["max_neighbors"] = estimate
            else:
                missing_primary = kwargs.get("max_neighbors1") is None
                missing_secondary = kwargs.get("max_neighbors2") is None
                if missing_primary or missing_secondary:
                    estimate_cutoff = max(float(cutoff), float(cutoff2))
                    estimate = estimate_max_neighbors(
                        estimate_cutoff,
                        atomic_density=atomic_density,
                        max_neighbors_lower_bound=32,
                    )
                    if missing_primary:
                        kwargs["max_neighbors1"] = estimate
                    if missing_secondary:
                        kwargs["max_neighbors2"] = estimate
        elif cutoff2 is None:
            if kwargs.get("max_neighbors") is None:
                kwargs["max_neighbors"] = estimate_max_neighbors(
                    float(cutoff), atomic_density=atomic_density
                )
        else:
            if kwargs.get("max_neighbors1") is None:
                kwargs["max_neighbors1"] = estimate_max_neighbors(
                    float(cutoff), atomic_density=atomic_density
                )
            if kwargs.get("max_neighbors2") is None:
                kwargs["max_neighbors2"] = estimate_max_neighbors(
                    float(cutoff2), atomic_density=atomic_density
                )
    if span_margin and not (
        cell is None and method in {"cell_list", "batch_cell_list"}
    ):
        raise ValueError(
            "nonzero span_margin requires cell=None and a resolved nonperiodic cell-list method"
        )
    if cluster:
        return _prepare_cluster_state(
            positions,
            cell,
            pbc,
            batch_idx,
            batch_ptr,
            method=method,
            systems=systems,
            num_atoms=n,
            cutoff=float(cutoff),
            cutoff2=None if cutoff2 is None else float(cutoff2),
            half_fill=bool(half_fill),
            wrap_positions=bool(wrap_positions),
            span_margin=span_margin,
            fixed_cell=fixed_cell,
            format=fmt,
            selective=selective,
            fill_value=fv,
            kwargs=kwargs,
            capacity=capacity,
        )
    return _build_noncluster_state(
        positions,
        cell,
        pbc,
        batch_idx,
        batch_ptr,
        cutoff,
        cutoff2,
        half_fill,
        wrap_positions,
        selective,
        span_margin,
        fixed_cell,
        kwargs,
        resolution,
    )


def _build_noncluster_state(
    positions: jax.Array,
    cell: jax.Array | None,
    pbc: jax.Array | None,
    batch_idx: jax.Array | None,
    batch_ptr: jax.Array | None,
    cutoff: float,
    cutoff2: float | None,
    half_fill: bool,
    wrap_positions: bool,
    selective: bool,
    span_margin: float,
    fixed_cell: bool,
    kwargs: dict[str, Any],
    resolution: _PreparedResolution,
) -> NeighborListState:
    """Allocate and assemble the resolved non-cluster prepared state."""
    method = resolution.method
    systems = resolution.systems
    n = resolution.num_atoms
    dual = resolution.dual
    fmt = resolution.format
    capacity = resolution.capacity
    fv = resolution.fill_value
    target_indices = resolution.target_indices
    num_rows = resolution.num_rows
    naive_method = method in {
        "naive",
        "batch_naive",
        "naive_dual_cutoff",
        "batch_naive_dual_cutoff",
    }
    returns_shifts = pbc is not None or method in {"cell_list", "batch_cell_list"}
    names = (
        ("neighbor_list", "neighbor_ptr", "num_neighbors", "metadata_valid")
        if fmt == "coo"
        else ("neighbor_matrix", "num_neighbors")
    )
    if returns_shifts:
        if fmt == "coo":
            names = (
                "neighbor_list",
                "neighbor_ptr",
                "neighbor_list_shifts",
                "num_neighbors",
                "metadata_valid",
            )
        else:
            names += ("neighbor_matrix_shifts",)
    if kwargs.get("return_distances"):
        names += ("neighbor_distances",)
    if kwargs.get("return_vectors"):
        names += ("neighbor_vectors",)
    if kwargs.get("pair_fn") is not None:
        names += ("pair_energies", "pair_forces")
    if dual:
        names += (
            ("neighbor_list2", "neighbor_ptr2", "num_neighbors2", "metadata_valid2")
            if fmt == "coo"
            else ("neighbor_matrix2", "num_neighbors2")
        )
        if returns_shifts:
            if fmt == "coo":
                # Fixed COO keeps shifts before the count/metadata pair.
                names = names[:-4] + (
                    "neighbor_list2",
                    "neighbor_ptr2",
                    "neighbor_list_shifts2",
                    "num_neighbors2",
                    "metadata_valid2",
                )
            else:
                names += ("neighbor_matrix_shifts2",)
    width = int(kwargs.get("max_neighbors1" if dual else "max_neighbors", n))
    width2 = int(kwargs.get("max_neighbors2", n))
    if fmt == "matrix":
        outputs = [
            jnp.full((num_rows, width), fv, jnp.int32),
            jnp.zeros((num_rows,), jnp.int32),
        ]
        if returns_shifts:
            outputs.append(jnp.zeros((num_rows, width, 3), jnp.int32))
        if kwargs.get("return_distances"):
            outputs.append(jnp.zeros((num_rows, width), positions.dtype))
        if kwargs.get("return_vectors"):
            outputs.append(jnp.zeros((num_rows, width, 3), positions.dtype))
        if kwargs.get("pair_fn") is not None:
            outputs += [
                jnp.zeros((num_rows, width), positions.dtype),
                jnp.zeros((num_rows, width, 3), positions.dtype),
            ]
        if dual:
            outputs += [
                jnp.full((n, width2), fv, jnp.int32),
                jnp.zeros((n,), jnp.int32),
            ]
            if returns_shifts:
                outputs.append(jnp.zeros((n, width2, 3), jnp.int32))
    else:
        c1, c2 = (
            (capacity if isinstance(capacity, tuple) else (capacity, capacity))
            if dual
            else (capacity, None)
        )
        outputs = [
            jnp.zeros((2, int(c1)), jnp.int32),
            jnp.zeros((num_rows + 1,), jnp.int32),
        ]
        if returns_shifts:
            outputs.append(jnp.zeros((int(c1), 3), jnp.int32))
        outputs.extend(
            (
                jnp.zeros((num_rows,), jnp.int32),
                jnp.asarray(True, dtype=jnp.bool_),
            )
        )
        if kwargs.get("return_distances"):
            outputs.append(jnp.zeros((int(c1),), positions.dtype))
        if kwargs.get("return_vectors"):
            outputs.append(jnp.zeros((int(c1), 3), positions.dtype))
        if kwargs.get("pair_fn") is not None:
            outputs += [
                jnp.zeros((int(c1),), positions.dtype),
                jnp.zeros((int(c1), 3), positions.dtype),
            ]
        if dual:
            outputs += [
                jnp.zeros((2, int(c2)), jnp.int32),
                jnp.zeros((n + 1,), jnp.int32),
            ]
            if returns_shifts:
                outputs.append(jnp.zeros((int(c2), 3), jnp.int32))
            outputs.extend(
                (
                    jnp.zeros((n,), jnp.int32),
                    jnp.asarray(True, dtype=jnp.bool_),
                )
            )
    synthesized = method in {"cell_list", "batch_cell_list"} and cell is None
    resolved_batch_idx = _batch_indices(batch_idx, batch_ptr, systems, n)
    span_capacity = jnp.zeros((systems, 3), dtype=positions.dtype)
    route_options = ()
    fixed_cell_geometry = None
    needs_fixed_inverse = fixed_cell and (
        method in {"cell_list", "batch_cell_list"}
        or (pbc is not None and wrap_positions)
    )
    fixed_inverse = (
        _fixed_cell_inverse(
            cell,
            positions,
            method=method,
            strategy=str(kwargs.get("strategy", "auto")),
            num_systems=systems,
        )
        if needs_fixed_inverse
        else None
    )
    if fixed_cell:
        fixed_cell_geometry = (
            fixed_inverse,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    if method in {"cell_list", "batch_cell_list"}:
        if synthesized:
            shifted_positions, cell, spans, _span_failure_axes = _synthetic_geometry(
                positions,
                resolved_batch_idx,
                batch_ptr,
                systems,
                jnp.zeros((systems, 3), dtype=positions.dtype),
                cutoff,
            )
            span_capacity = spans + jnp.asarray(span_margin, dtype=positions.dtype)
            # Keep cells strictly positive for empty and zero-span systems.
            cell = (span_capacity + jnp.asarray(0.1 * cutoff, dtype=positions.dtype))[
                :, :, None
            ] * jnp.eye(3, dtype=positions.dtype)
            pbc = jnp.zeros((systems, 3), dtype=jnp.bool_)
            if systems == 1:
                pbc = pbc.reshape(3)
        elif span_margin:
            raise ValueError(
                "nonzero span_margin requires cell=None and a resolved nonperiodic cell-list method"
            )
        from nvalchemiops.neighbors.cell_list import (
            compute_batch_pair_centric_n_outer,
        )

        prepared_positions = shifted_positions if synthesized else positions
        configured_max_cells = kwargs.get("max_total_cells")
        atom_centric_path = kwargs.get("atom_centric_path", "auto")
        if method == "batch_cell_list":
            from nvalchemiops.jax.neighbors.batch_cell_list import (
                batch_build_cell_list,
                estimate_batch_cell_list_sizes,
            )

            if configured_max_cells is None:
                configured_max_cells, _, _ = estimate_batch_cell_list_sizes(
                    prepared_positions,
                    batch_ptr=batch_ptr,
                    batch_idx=resolved_batch_idx,
                    cell=cell,
                    cutoff=cutoff,
                    pbc=pbc,
                )
            max_cells = int(configured_max_cells)
            build_result = batch_build_cell_list(
                prepared_positions,
                batch_idx=resolved_batch_idx,
                batch_ptr=batch_ptr,
                cell=cell,
                pbc=pbc,
                cutoff=cutoff,
                max_total_cells=max_cells,
            )
            cells_dim = build_result[0]
            search_radius = build_result[6]
            if fixed_cell:
                cells_per_system = jnp.prod(cells_dim, axis=1, dtype=jnp.int32)
                cell_offsets = jnp.concatenate(
                    (
                        jnp.zeros((1,), dtype=jnp.int32),
                        jnp.cumsum(cells_per_system[:-1], dtype=jnp.int32),
                    )
                )
                fixed_cell_geometry = (
                    fixed_inverse,
                    jax.lax.stop_gradient(cells_dim),
                    jax.lax.stop_gradient(search_radius),
                    jax.lax.stop_gradient(cells_per_system),
                    jax.lax.stop_gradient(cell_offsets),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            total_cells = int(jnp.sum(jnp.prod(cells_dim, axis=1)))
            radius_max = tuple(
                int(value) for value in jnp.max(search_radius, axis=0).tolist()
            )
            n_outer = compute_batch_pair_centric_n_outer(radius_max, False)
            route_options = (
                ("max_total_cells", max_cells),
                ("atom_centric_path", atom_centric_path),
                ("pair_centric_total_cells", total_cells),
                ("pair_centric_n_outer", n_outer),
                ("pair_centric_r_max", radius_max),
            )
        else:
            from nvalchemiops.jax.neighbors.cell_list import (
                build_cell_list,
                estimate_cell_list_sizes,
            )

            if configured_max_cells is None:
                configured_max_cells, _, _ = estimate_cell_list_sizes(
                    prepared_positions, cell, cutoff, pbc
                )
            max_cells = int(configured_max_cells)
            build_result = build_cell_list(
                prepared_positions,
                cutoff,
                cell,
                pbc,
                max_total_cells=max_cells,
            )
            cells_dim = build_result[0]
            search_radius = build_result[6]
            if fixed_cell:
                fixed_cell_geometry = (
                    fixed_inverse,
                    jax.lax.stop_gradient(cells_dim.reshape((1, 3))),
                    jax.lax.stop_gradient(search_radius.reshape((1, 3))),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            radius = tuple(int(value) for value in search_radius.tolist())
            n_outer = compute_batch_pair_centric_n_outer(radius, bool(half_fill))
            route_options = (
                ("max_total_cells", max_cells),
                ("atom_centric_path", atom_centric_path),
                ("pair_centric_n_outer", n_outer),
            )
    if method not in {"cell_list", "batch_cell_list"} and span_margin:
        raise ValueError(
            "nonzero span_margin requires cell=None and a resolved nonperiodic cell-list method"
        )
    eager_only = (
        method in {"cell_list", "batch_cell_list"}
        and kwargs.get("strategy") == "pair_centric"
        and (
            kwargs.get("return_vectors")
            or kwargs.get("return_distances")
            or kwargs.get("pair_fn") is not None
        )
    )
    shift_range_per_dimension = None
    num_shifts_per_system = None
    max_shifts_per_system = None
    max_atoms_per_system = None
    if (
        method
        in {"naive", "batch_naive", "naive_dual_cutoff", "batch_naive_dual_cutoff"}
        and pbc is not None
    ):
        from nvalchemiops.jax.neighbors.neighbor_utils import compute_naive_num_shifts

        shift_cell = cell if cell.ndim == 3 else cell[None]
        shift_pbc = pbc if pbc.ndim == 2 else pbc[None]
        shift_range_per_dimension, num_shifts_per_system, max_shifts_per_system = (
            compute_naive_num_shifts(
                shift_cell, float(cutoff2 if dual else cutoff), shift_pbc
            )
        )
        if method.startswith("batch_"):
            max_atoms_per_system = int(kwargs.get("max_atoms_per_system", n))
    spec = _Spec(
        method=method,
        strategy=kwargs.get("strategy", "auto"),
        format=fmt,
        coo_layout="compact" if fmt == "coo" else None,
        is_batched=method.startswith("batch_"),
        num_atoms=n,
        num_systems=systems,
        cutoff=float(cutoff),
        cutoff2=None if cutoff2 is None else float(cutoff2),
        half_fill=bool(half_fill),
        fill_value=fv,
        wrap_positions=bool(wrap_positions),
        selective=selective,
        return_vectors=bool(kwargs.get("return_vectors", False)),
        return_distances=bool(kwargs.get("return_distances", False)),
        span_margin=span_margin,
        synthesized_cell=synthesized,
        fixed_cell=fixed_cell,
        supports_compilation=not eager_only,
        compilation_blocker=(
            "pair-centric cell geometry and pair outputs are eager-only"
            if eager_only
            else None
        ),
        pair_fn=kwargs.get("pair_fn"),
        result_names=tuple(names),
        coo_capacity=capacity,
        max_neighbors=(width, width2 if dual else None),
        max_shifts_per_system=max_shifts_per_system,
        max_atoms_per_system=max_atoms_per_system,
        has_target_indices=target_indices is not None,
        shared_batch_cell=False,
        route_options=route_options,
    )
    outputs += [None for _ in range(12 - len(outputs))]
    selective_fixed_naive = selective and naive_method and fmt == "coo"
    if selective_fixed_naive:
        private_matrix1 = jnp.full((n, width), fv, dtype=jnp.int32)
        private_num1 = jnp.zeros((n,), dtype=jnp.int32)
        private_shift1 = (
            jnp.zeros((n, width, 3), dtype=jnp.int32) if pbc is not None else None
        )
        if dual:
            private_matrix2 = jnp.full((n, width2), fv, dtype=jnp.int32)
            private_num2 = jnp.zeros((n,), dtype=jnp.int32)
            private_shift2 = (
                jnp.zeros((n, width2, 3), dtype=jnp.int32) if pbc is not None else None
            )
        else:
            private_matrix2 = None
            private_num2 = None
            private_shift2 = None
    else:
        private_matrix1 = None
        private_num1 = None
        private_shift1 = None
        private_matrix2 = None
        private_num2 = None
        private_shift2 = None
    fixed_targets = target_indices
    leaves = _StateLeaves.build(
        cell=cell,
        pbc=pbc,
        batch_idx=resolved_batch_idx if method.startswith("batch_") else batch_idx,
        batch_ptr=batch_ptr,
        fixed_cell_geometry=fixed_cell_geometry,
        result=outputs,
        shift_range_per_dimension=shift_range_per_dimension,
        num_shifts_per_system=num_shifts_per_system,
        target_indices=fixed_targets,
        span_capacity=span_capacity,
        span_axis=jnp.zeros((systems,), jnp.int32),
        span_required=jnp.zeros((systems,), positions.dtype),
        initialized=jnp.zeros((systems,), bool),
        valid=jnp.ones((systems,), bool),
        failure_code=jnp.zeros((systems,), jnp.int32),
        required=jnp.zeros((systems,), jnp.int32),
        capacity=jnp.zeros((systems,), jnp.int32),
        first_failure=jnp.zeros((systems,), jnp.int32),
        private_matrix1=private_matrix1,
        private_num1=private_num1,
        private_shift1=private_shift1,
        private_matrix2=private_matrix2,
        private_num2=private_num2,
        private_shift2=private_shift2,
    )
    return NeighborListState(spec, leaves, token=_TOKEN)


def _execute_prepared_cluster_route(
    positions: jax.Array,
    cell: jax.Array | None,
    state: NeighborListState,
    kwargs: dict[str, Any],
    a: tuple[Any, ...],
    leaf_view: _StateLeaves,
    active: jax.Array,
    uninitialized_preserved: jax.Array,
) -> tuple[tuple[Any, ...], NeighborListState]:
    """Execute and publish a prepared cluster-tile route."""
    from nvalchemiops.jax.neighbors.batch_cluster_tile import (
        batch_cluster_tile_neighbor_list,
    )
    from nvalchemiops.jax.neighbors.cluster_tile import cluster_tile_neighbor_list

    options = dict(state._spec.route_options)
    call = {
        "max_neighbors": state._spec.max_neighbors[0],
        "fill_value": state.fill_value,
        "format": state.format,
        "max_tiles_per_group": options["max_tiles_per_group"],
        "cutoff2": state.cutoff2,
        "_return_status": True,
    }
    if state.fixed_cell:
        call["_fixed_cell_geometry"] = leaf_view.fixed_cell_geometry

    if state.selective:
        if "rebuild_flags" not in kwargs:
            raise ValueError("cannot preserve uninitialized system 0")
        active = kwargs["rebuild_flags"].astype(jnp.bool_).reshape(state.num_systems)
        uninitialized_preserved = ~leaf_view.initialized & ~active

    def old(name: str) -> Any:
        return (
            _result_leaf(state, name, a) if name in state._spec.result_names else None
        )

    if state.cutoff2 is not None and state.format == "matrix":
        call.update(
            previous_neighbor_matrix=old("neighbor_matrix"),
            previous_num_neighbors=old("num_neighbors"),
            previous_neighbor_matrix_shifts=old("neighbor_matrix_shifts"),
            previous_neighbor_matrix2=old("neighbor_matrix2"),
            previous_num_neighbors2=old("num_neighbors2"),
            previous_neighbor_matrix_shifts2=old("neighbor_matrix_shifts2"),
        )
    if state.cutoff2 is None:
        call.update(
            return_vectors=state.return_vectors,
            return_distances=state.return_distances,
            pair_fn=state._spec.pair_fn,
            pair_params=kwargs.get("pair_params"),
        )
    if state.format == "coo":
        call["max_pairs"] = options["max_pairs"]
    if state.selective:
        call["rebuild_flags"] = active if state.is_batched else active[:1]
        call.update(
            previous_num_tiles=old("num_tiles"),
            previous_tile_row_group=old("tile_row_group"),
            previous_tile_col_group=old("tile_col_group"),
        )
        if state.is_batched:
            call.update(
                tile_offsets=old("tile_offsets"),
                previous_tile_counts=old("tile_counts"),
                previous_tile_system=old("tile_system"),
            )
        if state.format == "coo":
            call.update(
                pair_offsets=old("pair_offsets"),
                previous_pair_counts=old("pair_counts"),
                previous_neighbor_list=old("neighbor_list"),
                previous_neighbor_list_shifts=old("neighbor_list_shifts"),
            )
        else:
            call.update(
                previous_neighbor_matrix=old("neighbor_matrix"),
                previous_num_neighbors=old("num_neighbors"),
                previous_neighbor_matrix_shifts=old("neighbor_matrix_shifts"),
            )
    if state.is_batched:
        call["batch_ptr"] = leaf_view.batch_ptr
        runtime_cell = cell if cell is not None else leaf_view.cell
        if state._spec.shared_batch_cell and runtime_cell.ndim == 2:
            runtime_cell = jnp.broadcast_to(runtime_cell, leaf_view.cell.shape)
        raw = batch_cluster_tile_neighbor_list(
            positions, state.cutoff, runtime_cell, **call
        )
    else:
        raw = cluster_tile_neighbor_list(
            positions,
            state.cutoff,
            cell if cell is not None else leaf_view.cell,
            **call,
        )
    result, status_tail = raw[:-6], raw[-6:]
    (
        tile_overflow,
        coo_overflow,
        tile_required,
        tile_capacity,
        coo_required,
        coo_capacity,
    ) = status_tail
    matrix_overflow = jnp.zeros_like(jnp.asarray(coo_overflow))
    matrix_selected = jnp.zeros_like(matrix_overflow)
    matrix_required = jnp.zeros_like(jnp.asarray(coo_required))
    matrix_capacity = jnp.zeros_like(jnp.asarray(coo_capacity))
    if state.format == "matrix":
        for suffix in ("", "2"):
            count_name = f"num_neighbors{suffix}"
            if count_name not in state._spec.result_names:
                continue
            width = state._spec.max_neighbors[1 if suffix else 0]
            counts = result[state._spec.result_names.index(count_name)]
            if state.is_batched:
                owners = leaf_view.batch_idx
                group_overflow = (
                    jax.ops.segment_sum(
                        (counts > width).astype(jnp.int32),
                        owners,
                        num_segments=state.num_systems,
                    )
                    > 0
                )
                group_required = jax.ops.segment_max(
                    counts,
                    owners,
                    num_segments=state.num_systems,
                )
            else:
                group_overflow = jnp.any(counts > width).reshape(1)
                group_required = jnp.max(counts, initial=jnp.int32(0)).reshape(1)
            select_group = group_overflow & ~matrix_selected
            matrix_overflow = matrix_overflow | group_overflow
            matrix_required = jnp.where(
                select_group,
                group_required,
                matrix_required,
            )
            matrix_capacity = jnp.where(
                select_group,
                jnp.full_like(group_required, width),
                matrix_capacity,
            )
            matrix_selected = matrix_selected | group_overflow
    (
        initialized,
        valid,
        failure_code,
        required,
        capacity,
        first_failure_marker,
    ) = _update_cluster_lifecycle(
        leaf_view.initialized,
        leaf_view.valid,
        leaf_view.failure_code,
        leaf_view.required,
        leaf_view.capacity,
        leaf_view.first_failure,
        active,
        uninitialized_preserved,
        jnp.asarray(tile_overflow),
        jnp.asarray(matrix_overflow),
        jnp.asarray(coo_overflow),
        jnp.asarray(tile_required),
        jnp.asarray(tile_capacity),
        jnp.asarray(matrix_required),
        jnp.asarray(matrix_capacity),
        jnp.asarray(coo_required),
        jnp.asarray(coo_capacity),
    )
    successor = _build_successor_leaves(
        a,
        result,
        span_axis=leaf_view.span_axis,
        span_required=leaf_view.span_required,
        initialized=initialized,
        valid=valid,
        failure_code=failure_code,
        required=required,
        capacity=capacity,
        first_failure_marker=first_failure_marker,
    )
    return result, state._replace(successor)


@dataclass(frozen=True)
class _PreparedRouteExecution:
    """Resolved non-cluster route call and its reusable execution context."""

    positions: jax.Array
    cell: jax.Array | None
    spans: jax.Array
    span_failure_axes: jax.Array
    dual: bool
    selective_fixed_naive: bool
    private_matrix1: Any
    private_num1: Any
    private_shift1: Any
    private_matrix2: Any
    private_num2: Any
    private_shift2: Any
    active: jax.Array
    uninitialized_preserved: jax.Array
    periodic_coverage_failure: jax.Array
    periodic_required: jax.Array
    periodic_capacity: jax.Array
    raw: Any


def _prepare_noncluster_route_execution(
    positions: jax.Array,
    cell: jax.Array | None,
    state: NeighborListState,
    kwargs: dict[str, Any],
    a: tuple[Any, ...],
    leaf_view: _StateLeaves,
    active: jax.Array,
    uninitialized_preserved: jax.Array,
) -> _PreparedRouteExecution:
    """Assemble and dispatch a prepared naive or cell-list route."""
    from nvalchemiops.jax.neighbors.batch_cell_list import batch_cell_list
    from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list
    from nvalchemiops.jax.neighbors.batch_naive_dual_cutoff import (
        batch_naive_neighbor_list_dual_cutoff,
    )
    from nvalchemiops.jax.neighbors.cell_list import cell_list
    from nvalchemiops.jax.neighbors.naive import naive_neighbor_list
    from nvalchemiops.jax.neighbors.naive_dual_cutoff import (
        naive_neighbor_list_dual_cutoff,
    )

    if state._spec.synthesized_cell:
        positions, resolved_cell, spans, span_failure_axes = _synthetic_geometry(
            positions,
            leaf_view.batch_idx,
            leaf_view.batch_ptr,
            state.num_systems,
            state._span_capacity,
            state.cutoff,
        )
        cell = resolved_cell
    else:
        spans = jnp.zeros((state.num_systems, 3), dtype=positions.dtype)
        span_failure_axes = jnp.zeros((state.num_systems, 3), dtype=jnp.bool_)
    dual = state.cutoff2 is not None
    selective_fixed_naive = (
        state.selective
        and state.format == "coo"
        and state.method
        in {
            "naive",
            "batch_naive",
            "naive_dual_cutoff",
            "batch_naive_dual_cutoff",
        }
    )
    private_matrix1, private_num1, private_shift1 = (
        leaf_view.private_matrix1,
        leaf_view.private_num1,
        leaf_view.private_shift1,
    )
    private_matrix2, private_num2, private_shift2 = (
        leaf_view.private_matrix2,
        leaf_view.private_num2,
        leaf_view.private_shift2,
    )
    call = {
        "cell": cell if cell is not None else leaf_view.cell,
        "pbc": leaf_view.pbc,
        "half_fill": state.half_fill,
        "fill_value": state.fill_value,
        "return_neighbor_list": state.format == "coo" and not selective_fixed_naive,
    }
    if (
        state.fixed_cell
        and state.method
        in {
            "naive",
            "batch_naive",
            "naive_dual_cutoff",
            "batch_naive_dual_cutoff",
        }
        and leaf_view.pbc is None
    ):
        # The explicit fixed cell remains stored and validated, but nonperiodic
        # naive kernels do not consume it and reject cell without PBC.
        call["cell"] = None
    if state.fixed_cell:
        call["_fixed_cell_geometry"] = leaf_view.fixed_cell_geometry
    if not dual:
        call.update(
            return_vectors=state.return_vectors,
            return_distances=state.return_distances,
            strategy=state.strategy,
        )
    if state._spec.has_target_indices:
        call["target_indices"] = state._target_indices
    if state.selective:
        if "rebuild_flags" not in kwargs:
            raise ValueError("cannot preserve uninitialized system 0")
        active = kwargs["rebuild_flags"].astype(jnp.bool_).reshape(state.num_systems)
        uninitialized_preserved = ~leaf_view.initialized & ~active
        if (
            state.method not in {"cell_list", "batch_cell_list"}
            and not selective_fixed_naive
        ):
            call["rebuild_flags"] = active
            for name in state._spec.result_names:
                if name in {
                    "neighbor_matrix",
                    "num_neighbors",
                    "neighbor_matrix_shifts",
                    "neighbor_matrix2",
                    "num_neighbors2",
                    "neighbor_matrix_shifts2",
                }:
                    call[name] = _result_leaf(state, name, a)
        if selective_fixed_naive:
            call["rebuild_flags"] = active
            if dual:
                call["neighbor_matrix1"] = leaf_view.private_matrix1
                call["num_neighbors1"] = leaf_view.private_num1
                if leaf_view.pbc is not None:
                    call["neighbor_matrix_shifts1"] = leaf_view.private_shift1
                call["neighbor_matrix2"] = leaf_view.private_matrix2
                call["num_neighbors2"] = leaf_view.private_num2
                if leaf_view.pbc is not None:
                    call["neighbor_matrix_shifts2"] = leaf_view.private_shift2
            else:
                call["neighbor_matrix"] = leaf_view.private_matrix1
                call["num_neighbors"] = leaf_view.private_num1
                if leaf_view.pbc is not None:
                    call["neighbor_matrix_shifts"] = leaf_view.private_shift1
    if state.method in {
        "naive",
        "batch_naive",
        "naive_dual_cutoff",
        "batch_naive_dual_cutoff",
    }:
        call["wrap_positions"] = state.wrap_positions
        if leaf_view.pbc is not None:
            call.update(
                shift_range_per_dimension=state._shift_range_per_dimension,
                num_shifts_per_system=state._num_shifts_per_system,
                max_shifts_per_system=state._spec.max_shifts_per_system,
            )
            if state.is_batched:
                call["max_atoms_per_system"] = state._spec.max_atoms_per_system
    periodic_coverage_failure = jnp.zeros((state.num_systems,), dtype=jnp.bool_)
    periodic_required = jnp.zeros((state.num_systems,), dtype=jnp.int32)
    periodic_capacity = jnp.zeros((state.num_systems,), dtype=jnp.int32)
    if (
        state.method
        in {
            "naive",
            "batch_naive",
            "naive_dual_cutoff",
            "batch_naive_dual_cutoff",
        }
        and leaf_view.pbc is not None
        and not state.fixed_cell
    ):
        runtime_cell = call["cell"]
        if runtime_cell.ndim == 2:
            runtime_cell = runtime_cell[None]
        runtime_pbc = leaf_view.pbc
        if runtime_pbc.ndim == 1:
            runtime_pbc = runtime_pbc[None]
        reciprocal = jnp.swapaxes(jnp.linalg.inv(runtime_cell), -1, -2)
        runtime_ranges = jnp.where(
            runtime_pbc,
            jnp.ceil(
                jnp.linalg.vector_norm(reciprocal, axis=-1)
                * float(state.cutoff2 if dual else state.cutoff)
            ).astype(jnp.int32),
            0,
        )
        runtime_num_shifts = (
            runtime_ranges[:, 0]
            * (2 * runtime_ranges[:, 1] + 1)
            * (2 * runtime_ranges[:, 2] + 1)
            + runtime_ranges[:, 1] * (2 * runtime_ranges[:, 2] + 1)
            + runtime_ranges[:, 2]
            + 1
        ).astype(jnp.int32)
        periodic_coverage_failure = jnp.any(
            runtime_ranges > state._shift_range_per_dimension, axis=1
        )
        periodic_required = runtime_num_shifts
        periodic_capacity = state._num_shifts_per_system
    if state.format == "coo":
        if not selective_fixed_naive:
            call["coo_capacity"] = state._spec.coo_capacity
        if state.cutoff2 is None:
            call["max_neighbors"] = state._spec.max_neighbors[0]
        else:
            call["max_neighbors1"] = state._spec.max_neighbors[0]
            call["max_neighbors2"] = state._spec.max_neighbors[1]
    elif state.cutoff2 is None:
        call["max_neighbors"] = _result_leaf(state, "neighbor_matrix", a).shape[1]
    else:
        call["max_neighbors1"] = _result_leaf(state, "neighbor_matrix", a).shape[1]
        call["max_neighbors2"] = _result_leaf(state, "neighbor_matrix2", a).shape[1]
    if "pair_params" in kwargs:
        call["pair_params"] = kwargs["pair_params"]
    if state._spec.pair_fn is not None:
        call["pair_fn"] = state._spec.pair_fn
    if state.is_batched:
        call.update(batch_idx=leaf_view.batch_idx, batch_ptr=leaf_view.batch_ptr)
    fn = {
        ("naive", False): naive_neighbor_list,
        ("batch_naive", False): batch_naive_neighbor_list,
        ("naive_dual_cutoff", True): naive_neighbor_list_dual_cutoff,
        ("batch_naive_dual_cutoff", True): batch_naive_neighbor_list_dual_cutoff,
        ("cell_list", False): cell_list,
        ("batch_cell_list", False): batch_cell_list,
    }.get((state.method, dual))
    if state.method in {"cell_list", "batch_cell_list"}:
        call.update(dict(state._spec.route_options))
        call["_return_status"] = True
    raw = (
        fn(positions, state.cutoff, state.cutoff2, **call)
        if dual
        else fn(positions, state.cutoff, **call)
    )
    return _PreparedRouteExecution(
        positions=positions,
        cell=cell,
        spans=spans,
        span_failure_axes=span_failure_axes,
        dual=dual,
        selective_fixed_naive=selective_fixed_naive,
        private_matrix1=private_matrix1,
        private_num1=private_num1,
        private_shift1=private_shift1,
        private_matrix2=private_matrix2,
        private_num2=private_num2,
        private_shift2=private_shift2,
        active=active,
        uninitialized_preserved=uninitialized_preserved,
        periodic_coverage_failure=periodic_coverage_failure,
        periodic_required=periodic_required,
        periodic_capacity=periodic_capacity,
        raw=raw,
    )


@dataclass(frozen=True)
class _PreparedExecutionInputs:
    """Validated runtime inputs shared by prepared route execution."""

    leaves: tuple[Any, ...]
    leaf_view: _StateLeaves
    active: jax.Array
    uninitialized_preserved: jax.Array


def _validate_prepared_execution(
    positions: jax.Array,
    cell: jax.Array | None,
    state: NeighborListState,
    kwargs: dict[str, Any],
) -> _PreparedExecutionInputs:
    """Validate runtime arrays before dispatching the prepared route."""
    a = state._leaves
    leaf_view = _StateLeaves(a)
    if positions.shape != (state.num_atoms, 3):
        raise ValueError("runtime positions shape does not match prepared state")
    if positions.dtype != state._span_capacity.dtype:
        raise ValueError("runtime positions dtype does not match prepared state")
    if (
        isinstance(positions, jax.Array)
        and not isinstance(positions, jax.core.Tracer)
        and isinstance(state._span_capacity, jax.Array)
        and not isinstance(state._span_capacity, jax.core.Tracer)
    ):
        if positions.devices() != state._span_capacity.devices():
            raise ValueError("runtime positions device does not match prepared state")
    if state.selective:
        rebuild_flags = kwargs.get("rebuild_flags")
        if rebuild_flags is None:
            raise ValueError("selective prepared state requires rebuild_flags")
        if (
            rebuild_flags.shape != (state.num_systems,)
            or rebuild_flags.dtype != jnp.bool_
        ):
            raise ValueError(
                "rebuild_flags must be a bool array with shape (num_systems,)"
            )
    elif kwargs.get("rebuild_flags") is not None:
        raise ValueError("rebuild_flags requires a selective prepared state")
    if state._spec.synthesized_cell and cell is not None:
        raise ValueError("prepared nonperiodic state requires cell=None")
    if state._spec.pair_fn is not None:
        if "pair_params" not in kwargs or kwargs["pair_params"] is None:
            raise ValueError(
                "pair_fn requires pair_params (a per-atom (n_atoms, K) parameter array)."
            )
        pair_params = kwargs["pair_params"]
        if not hasattr(pair_params, "ndim") or pair_params.ndim != 2:
            raise ValueError(
                "pair_params must be a JAX per-atom (n_atoms, K) parameter array"
            )
        if (
            pair_params.shape[0] != state.num_atoms
            or pair_params.dtype != positions.dtype
        ):
            raise ValueError(
                "pair_params must match prepared positions dtype and leading atom dimension"
            )
        if (
            isinstance(pair_params, jax.Array)
            and not isinstance(pair_params, jax.core.Tracer)
            and isinstance(positions, jax.Array)
            and not isinstance(positions, jax.core.Tracer)
        ):
            if pair_params.devices() != positions.devices():
                raise ValueError("pair_params must be on the same device as positions")
    elif "pair_params" in kwargs and kwargs["pair_params"] is not None:
        raise ValueError("pair_params requires pair_fn.")
    active = (
        kwargs["rebuild_flags"].astype(jnp.bool_)
        if state.selective
        else jnp.ones((state.num_systems,), dtype=jnp.bool_)
    )
    uninitialized_preserved = (
        ~leaf_view.initialized & ~active
        if state.selective
        else jnp.zeros((state.num_systems,), dtype=jnp.bool_)
    )
    if cell is not None and leaf_view.cell is not None:
        expected_shape = (
            (3, 3) if state._spec.shared_batch_cell else leaf_view.cell.shape
        )
        if cell.shape != expected_shape or cell.dtype != leaf_view.cell.dtype:
            raise ValueError(
                "runtime cell shape or dtype does not match prepared state"
            )
        if (
            isinstance(cell, jax.Array)
            and not isinstance(cell, jax.core.Tracer)
            and isinstance(leaf_view.cell, jax.Array)
            and not isinstance(leaf_view.cell, jax.core.Tracer)
        ):
            if cell.devices() != leaf_view.cell.devices():
                raise ValueError("runtime cell device does not match prepared state")
    return _PreparedExecutionInputs(
        leaves=a,
        leaf_view=leaf_view,
        active=active,
        uninitialized_preserved=uninitialized_preserved,
    )


@dataclass(frozen=True)
class _ProcessedNonclusterResult:
    """Published results, diagnostics, and private matrices for one execution."""

    result: tuple[Any, ...]
    diagnostic_result: tuple[Any, ...]
    status: tuple[tuple[Any, ...], ...]
    private_leaves: tuple[Any, ...]


def _process_noncluster_result(
    state: NeighborListState,
    execution: _PreparedRouteExecution,
    leaves: tuple[Any, ...],
    leaf_view: _StateLeaves,
) -> _ProcessedNonclusterResult:
    """Pack route results and apply selective publication atomically."""
    dual = execution.dual
    private_matrix1 = execution.private_matrix1
    private_num1 = execution.private_num1
    private_shift1 = execution.private_shift1
    private_matrix2 = execution.private_matrix2
    private_num2 = execution.private_num2
    private_shift2 = execution.private_shift2
    raw = execution.raw
    nstatus = (
        12 if dual else (11 if state.method in {"cell_list", "batch_cell_list"} else 6)
    )
    if execution.selective_fixed_naive:
        from nvalchemiops.jax.neighbors.neighbor_utils import (
            _pack_fixed_capacity_neighbor_list_from_neighbor_matrix,
        )

        if dual:
            if leaf_view.pbc is not None:
                (
                    private_matrix1,
                    private_num1,
                    private_shift1,
                    private_matrix2,
                    private_num2,
                    private_shift2,
                ) = raw
            else:
                private_matrix1, private_num1, private_matrix2, private_num2 = raw
                private_shift1 = leaf_view.private_shift1
                private_shift2 = leaf_view.private_shift2
        elif leaf_view.pbc is not None:
            private_matrix1, private_num1, private_shift1 = raw
        else:
            private_matrix1, private_num1 = raw
            private_shift1 = leaf_view.private_shift1
        capacity1 = (
            state._spec.coo_capacity[0]
            if isinstance(state._spec.coo_capacity, tuple)
            else state._spec.coo_capacity
        )
        packed1, _ = _pack_fixed_capacity_neighbor_list_from_neighbor_matrix(
            private_matrix1,
            private_num1,
            capacity1,
            neighbor_shift_matrix=(
                private_shift1 if leaf_view.pbc is not None else None
            ),
            fill_value=state.fill_value,
            metadata_valid=jnp.ones((), dtype=jnp.bool_),
        )
        if dual:
            capacity2 = (
                state._spec.coo_capacity[1]
                if isinstance(state._spec.coo_capacity, tuple)
                else state._spec.coo_capacity
            )
            packed2, _ = _pack_fixed_capacity_neighbor_list_from_neighbor_matrix(
                private_matrix2,
                private_num2,
                capacity2,
                neighbor_shift_matrix=(
                    private_shift2 if leaf_view.pbc is not None else None
                ),
                fill_value=state.fill_value,
                metadata_valid=jnp.ones((), dtype=jnp.bool_),
            )
            result = (*packed1, *packed2)
        else:
            result = packed1
        status = ()
    elif len(raw) >= nstatus + len(state._spec.result_names):
        result, tail = raw[:-nstatus], raw[-nstatus:]
        status = (tuple(tail[:6]), tuple(tail[6:])) if dual else (tuple(tail),)
    else:
        result = raw
        status = ()

    # Global fixed-COO packing is transactional: a selected update must not
    # evict logical records owned by an initialized, unselected system.
    rollback = jnp.asarray(False, dtype=jnp.bool_)
    diagnostic_result = result
    if execution.selective_fixed_naive and state.is_batched:
        clipped_systems = []
        for group_index in range(2 if dual else 1):
            count_name = "num_neighbors2" if group_index else "num_neighbors"
            ptr_name = "neighbor_ptr2" if group_index else "neighbor_ptr"
            counts = jnp.asarray(
                diagnostic_result[state._spec.result_names.index(count_name)]
            )
            stored_ptr = jnp.asarray(
                diagnostic_result[state._spec.result_names.index(ptr_name)]
            )
            clipped_systems.append(
                jax.ops.segment_sum(
                    (counts > jnp.diff(stored_ptr)).astype(jnp.int32),
                    leaf_view.batch_idx,
                    num_segments=state.num_systems,
                )
                > 0
            )
        inactive_clipped = jnp.logical_or.reduce(jnp.stack(clipped_systems))
        inactive_clipped &= ~execution.active & leaf_view.initialized
        rollback = jnp.any(inactive_clipped)
        result = tuple(
            jnp.where(
                rollback,
                _result_leaf(state, state._spec.result_names[index], leaves),
                value,
            )
            for index, value in enumerate(result)
        )
    if execution.selective_fixed_naive:
        private_matrix1 = jnp.where(
            rollback, leaf_view.private_matrix1, private_matrix1
        )
        private_num1 = jnp.where(rollback, leaf_view.private_num1, private_num1)
        if private_shift1 is not None:
            private_shift1 = jnp.where(
                rollback, leaf_view.private_shift1, private_shift1
            )
        if dual:
            private_matrix2 = jnp.where(
                rollback, leaf_view.private_matrix2, private_matrix2
            )
            private_num2 = jnp.where(rollback, leaf_view.private_num2, private_num2)
            if private_shift2 is not None:
                private_shift2 = jnp.where(
                    rollback, leaf_view.private_shift2, private_shift2
                )
    if state.selective and state.method in {"cell_list", "batch_cell_list"}:
        row_indices = (
            state._target_indices
            if state._spec.has_target_indices
            else jnp.arange(state.num_atoms)
        )
        row_active = (
            execution.active[leaf_view.batch_idx[row_indices]]
            if state.is_batched
            else jnp.broadcast_to(execution.active[0], (row_indices.shape[0],))
        )
        merged = list(result)
        matrix_names = {
            "neighbor_matrix",
            "num_neighbors",
            "neighbor_matrix_shifts",
            "neighbor_matrix2",
            "num_neighbors2",
            "neighbor_matrix_shifts2",
        }
        for index, name in enumerate(state._spec.result_names):
            if name in matrix_names:
                mask = row_active.reshape(
                    (row_indices.shape[0],) + (1,) * (result[index].ndim - 1)
                )
                merged[index] = jnp.where(
                    mask, result[index], _result_leaf(state, name, leaves)
                )
        result = tuple(merged)
    return _ProcessedNonclusterResult(
        result=tuple(result),
        diagnostic_result=tuple(diagnostic_result),
        status=tuple(status),
        private_leaves=(
            private_matrix1,
            private_num1,
            private_shift1,
            private_matrix2,
            private_num2,
            private_shift2,
        ),
    )


def _derive_noncluster_diagnostics(
    state: NeighborListState,
    leaf_view: _StateLeaves,
    processed: _ProcessedNonclusterResult,
    active: jax.Array,
    *,
    dual: bool,
    selective_fixed_naive: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Choose one atomic failure diagnostic per system and cutoff group."""
    status = list(processed.status)
    if (
        state.method
        in {
            "naive",
            "batch_naive",
            "naive_dual_cutoff",
            "batch_naive_dual_cutoff",
        }
        or state.selective
    ):
        status = []
        for group_index in range(2 if dual else 1):
            count_name = "num_neighbors2" if group_index else "num_neighbors"
            counts = jnp.asarray(
                processed.diagnostic_result[state._spec.result_names.index(count_name)]
            )
            if state.is_batched:
                row_indices = (
                    state._target_indices
                    if state._spec.has_target_indices
                    else jnp.arange(state.num_atoms)
                )
                row_owners = leaf_view.batch_idx[row_indices]
                row_required = jax.ops.segment_max(
                    counts, row_owners, num_segments=state.num_systems
                )
                coo_required = jax.ops.segment_sum(
                    counts, row_owners, num_segments=state.num_systems
                )
            else:
                row_required = jnp.max(counts, initial=jnp.int32(0)).reshape(1)
                coo_required = jnp.sum(counts, dtype=jnp.int32).reshape(1)
            row_cap = jnp.full_like(
                row_required, state._spec.max_neighbors[group_index]
            )
            coo_cap_value = (
                state._spec.coo_capacity[group_index]
                if isinstance(state._spec.coo_capacity, tuple)
                else state._spec.coo_capacity
            )
            coo_cap = jnp.full_like(
                row_required, 0 if coo_cap_value is None else coo_cap_value
            )
            if state.is_batched and coo_cap_value is not None:
                ptr_name = "neighbor_ptr2" if group_index else "neighbor_ptr"
                stored_ptr = jnp.asarray(
                    processed.diagnostic_result[
                        state._spec.result_names.index(ptr_name)
                    ]
                )
                incomplete = counts > jnp.diff(stored_ptr)
                clipped_system = (
                    jax.ops.segment_sum(
                        incomplete.astype(jnp.int32),
                        row_owners,
                        num_segments=state.num_systems,
                    )
                    > 0
                )
                global_coo_required = jnp.sum(counts, dtype=jnp.int32)
                if selective_fixed_naive:
                    inactive_clipped = clipped_system & ~active & leaf_view.initialized
                    coo_failure_owner = clipped_system | (
                        active & jnp.any(inactive_clipped)
                    )
                else:
                    coo_failure_owner = clipped_system
                coo_required = jnp.where(
                    coo_failure_owner, global_coo_required, coo_required
                )
                coo_overflow = coo_failure_owner
            else:
                coo_overflow = coo_required > coo_cap
            metadata_name = "metadata_valid2" if group_index else "metadata_valid"
            metadata = (
                jnp.asarray(
                    processed.diagnostic_result[
                        state._spec.result_names.index(metadata_name)
                    ]
                )
                if metadata_name in state._spec.result_names
                else jnp.asarray(True)
            )
            metadata_bad = jnp.broadcast_to(~metadata, row_required.shape)
            status.append(
                (
                    row_required > row_cap,
                    coo_overflow
                    if coo_cap_value is not None
                    else jnp.zeros_like(metadata_bad),
                    row_required,
                    row_cap,
                    coo_required,
                    coo_cap,
                    metadata_bad,
                )
            )

    category_failures = [
        jnp.zeros((state.num_systems,), dtype=jnp.bool_) for _ in range(4)
    ]
    category_required = [
        jnp.zeros((state.num_systems,), dtype=jnp.int32) for _ in range(4)
    ]
    category_capacity = [
        jnp.zeros((state.num_systems,), dtype=jnp.int32) for _ in range(4)
    ]
    for group in status:
        if state.method in {"cell_list", "batch_cell_list"} and len(group) >= 11:
            metadata = (jnp.asarray(group[0]) == 3) | jnp.asarray(group[3])
            diagnostics = (
                (metadata, jnp.zeros_like(group[5]), jnp.zeros_like(group[6])),
                (jnp.asarray(group[4]), jnp.asarray(group[9]), jnp.asarray(group[10])),
                (jnp.asarray(group[1]), jnp.asarray(group[5]), jnp.asarray(group[6])),
                (jnp.asarray(group[2]), jnp.asarray(group[7]), jnp.asarray(group[8])),
            )
        else:
            row = jnp.asarray(group[0])
            coo = jnp.asarray(group[1])
            metadata = jnp.asarray(group[6]) if len(group) > 6 else jnp.zeros_like(row)
            diagnostics = (
                (metadata, jnp.zeros_like(row), jnp.zeros_like(row)),
                (jnp.zeros_like(row), jnp.zeros_like(row), jnp.zeros_like(row)),
                (row, jnp.asarray(group[2]), jnp.asarray(group[3])),
                (coo, jnp.asarray(group[4]), jnp.asarray(group[5])),
            )
        for index, (failed, required, capacity) in enumerate(diagnostics):
            failed = jnp.asarray(failed)
            choose = failed & ~category_failures[index]
            category_failures[index] |= failed
            category_required[index] = jnp.where(
                choose, required, category_required[index]
            )
            category_capacity[index] = jnp.where(
                choose, capacity, category_capacity[index]
            )

    winner = jnp.zeros((state.num_systems,), dtype=jnp.bool_)
    winner_code = jnp.zeros((state.num_systems,), dtype=jnp.int32)
    winner_required = jnp.zeros((state.num_systems,), dtype=jnp.int32)
    winner_capacity = jnp.zeros((state.num_systems,), dtype=jnp.int32)
    for index, category in enumerate((3, 7, 1, 2)):
        choose = category_failures[index] & ~winner
        winner_code = jnp.where(choose, jnp.int32(category), winner_code)
        winner_required = jnp.where(choose, category_required[index], winner_required)
        winner_capacity = jnp.where(choose, category_capacity[index], winner_capacity)
        winner |= category_failures[index]
    return winner, winner_code, winner_required, winner_capacity


def _execute_prepared_neighbor_list_full(
    positions: jax.Array,
    cell: jax.Array | None,
    state: NeighborListState,
    *,
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], NeighborListState]:
    """Run the resolved route and return ``(results, successor_state)``."""
    if not isinstance(state, NeighborListState):
        raise TypeError(
            "state must be a NeighborListState returned by prepare_neighbor_list"
        )
    if any(
        name in _CALLER_OWNED and value is not None for name, value in kwargs.items()
    ):
        raise ValueError(
            "caller-owned output or scratch buffers cannot be used with state"
        )
    unknown = next((name for name in kwargs if name not in _EXECUTION_KEYS), None)
    if unknown is not None:
        raise TypeError(f"unexpected prepared neighbor-list option: {unknown!r}")
    inputs = _validate_prepared_execution(positions, cell, state, kwargs)
    a = inputs.leaves
    leaf_view = inputs.leaf_view
    active = inputs.active
    uninitialized_preserved = inputs.uninitialized_preserved
    if state.method in {"cluster_tile", "batch_cluster_tile"}:
        return _execute_prepared_cluster_route(
            positions,
            cell,
            state,
            kwargs,
            a,
            leaf_view,
            active,
            uninitialized_preserved,
        )
    execution = _prepare_noncluster_route_execution(
        positions,
        cell,
        state,
        kwargs,
        a,
        leaf_view,
        active,
        uninitialized_preserved,
    )
    positions = execution.positions
    cell = execution.cell
    spans = execution.spans
    span_failure_axes = execution.span_failure_axes
    dual = execution.dual
    selective_fixed_naive = execution.selective_fixed_naive
    active = execution.active
    uninitialized_preserved = execution.uninitialized_preserved
    processed = _process_noncluster_result(state, execution, a, leaf_view)
    result = processed.result
    winner, winner_code, winner_required, winner_capacity = (
        _derive_noncluster_diagnostics(
            state,
            leaf_view,
            processed,
            active,
            dual=dual,
            selective_fixed_naive=selective_fixed_naive,
        )
    )
    periodic_coverage_failure = execution.periodic_coverage_failure & active
    span_failure = jnp.any(span_failure_axes, axis=1) & active
    winner = winner & active
    failed = winner | periodic_coverage_failure | span_failure | uninitialized_preserved
    code = jnp.where(
        uninitialized_preserved,
        6,
        jnp.where(
            span_failure,
            5,
            jnp.where(
                periodic_coverage_failure,
                8,
                jnp.where(
                    winner,
                    winner_code,
                    0,
                ),
            ),
        ),
    ).astype(jnp.int32)
    required = jnp.where(
        periodic_coverage_failure,
        execution.periodic_required,
        winner_required,
    )
    capacity = jnp.where(
        periodic_coverage_failure,
        execution.periodic_capacity,
        winner_capacity,
    )
    old_initialized, old_valid = leaf_view.initialized, leaf_view.valid
    current_code = code
    status_updated = active | uninitialized_preserved
    first_failure = old_valid & failed & status_updated
    latched_code = jnp.where(first_failure, current_code, leaf_view.failure_code)
    latched_required = jnp.where(first_failure, required, leaf_view.required)
    latched_capacity = jnp.where(first_failure, capacity, leaf_view.capacity)
    span_axis = jnp.argmax(span_failure_axes, axis=1).astype(jnp.int32)
    span_required = jnp.take_along_axis(spans, span_axis[:, None], axis=1).reshape(-1)
    latched_span_axis = jnp.where(
        first_failure & (current_code == 5), span_axis, leaf_view.span_axis
    )
    latched_span_required = jnp.where(
        first_failure & (current_code == 5), span_required, leaf_view.span_required
    )
    successor = _build_successor_leaves(
        a,
        result,
        span_axis=latched_span_axis,
        span_required=latched_span_required,
        initialized=old_initialized | active,
        valid=jnp.where(status_updated, old_valid & ~failed, old_valid),
        failure_code=latched_code,
        required=latched_required,
        capacity=latched_capacity,
        first_failure_marker=jnp.where(
            first_failure, 1, leaf_view.first_failure
        ).astype(jnp.int32),
        private_leaves=processed.private_leaves,
    )
    return result, state._replace(successor)


def _execute_prepared_neighbor_list(
    positions: jax.Array,
    cell: jax.Array | None,
    state: NeighborListState,
    *,
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], NeighborListState]:
    """Execute a prepared route, preserving initialized inactive fixed COO state."""
    if not isinstance(state, NeighborListState):
        raise TypeError(
            "state must be a NeighborListState returned by prepare_neighbor_list"
        )
    if any(
        name in _CALLER_OWNED and value is not None for name, value in kwargs.items()
    ):
        raise ValueError(
            "caller-owned output or scratch buffers cannot be used with state"
        )
    unknown = next((name for name in kwargs if name not in _EXECUTION_KEYS), None)
    if unknown is not None:
        raise TypeError(f"unexpected prepared neighbor-list option: {unknown!r}")
    inputs = _validate_prepared_execution(positions, cell, state, kwargs)
    selective_fixed_naive = (
        state.selective
        and state.format == "coo"
        and state.method
        in {
            "naive",
            "batch_naive",
            "naive_dual_cutoff",
            "batch_naive_dual_cutoff",
        }
    )
    if not selective_fixed_naive:
        return _execute_prepared_neighbor_list_full(
            positions, cell, state, kwargs=kwargs
        )

    published = tuple(
        _result_leaf(state, name, inputs.leaves) for name in state._spec.result_names
    )
    preserve = ~jnp.any(inputs.active) & jnp.all(inputs.leaf_view.initialized)
    return jax.lax.cond(
        preserve,
        lambda _: (published, state),
        lambda _: _execute_prepared_neighbor_list_full(
            positions, cell, state, kwargs=kwargs
        ),
        operand=None,
    )


def check_neighbor_list_state(state: NeighborListState) -> None:
    """Raise a host-side exception for latched prepared-state failures.

    Parameters
    ----------
    state : NeighborListState
        Successor state returned by prepared JAX execution.

    Raises
    ------
    TypeError
        If ``state`` was not created by :func:`prepare_neighbor_list`.
    NeighborOverflowError
        If the lowest-index failed system exhausted row, COO, or route
        capacity and required/capacity details are available.
    TileBufferOverflow
        If the lowest-index failed system exhausted cluster-tile capacity.
    RuntimeError
        For span, stale-metadata, preservation, or other runtime failures.

    Notes
    -----
    Call this function outside ``jax.jit``. It reads device status and may
    synchronize with the device.

    Every failed system index is included in sorted order in the exception
    message. The lowest failed index determines the structured exception or
    runtime detail. Calling this function does not change the state or clear
    sticky failure history. Check before consuming results when immediate
    failure detection matters; a final check detects historical failures but
    cannot make results already consumed after a failed step safe. Inspect
    ``state.initialized`` separately when initialization matters.
    """
    if not isinstance(state, NeighborListState):
        raise TypeError(
            "state must be a NeighborListState returned by prepare_neighbor_list"
        )
    failed_systems = [int(i) for i in jnp.nonzero(~state.valid)[0].tolist()]
    if failed_systems:
        first_failed_system = failed_systems[0]
        first_failure_code = int(
            jnp.asarray(state._failure_code[first_failed_system]).item()
        )
        if first_failure_code == 4:
            from nvalchemiops.neighbors.neighbor_utils import TileBufferOverflow

            error = TileBufferOverflow(
                int(state._capacity[failed_systems[0]]),
                int(state._required[failed_systems[0]]),
                failed_systems[0] if state.is_batched else None,
            )
            error.args = (
                "prepared neighbor-list state failed for systems "
                f"{failed_systems}; first failure: {error}",
            )
            raise error
        if first_failure_code in (1, 2, 7):
            from nvalchemiops.neighbors.neighbor_utils import NeighborOverflowError

            error = NeighborOverflowError(
                int(state._capacity[failed_systems[0]]),
                int(state._required[failed_systems[0]]),
                failed_systems[0] if state.is_batched else None,
            )
            error.args = (
                "prepared neighbor-list state failed for systems "
                f"{failed_systems}; first failure: {error}",
            )
            raise error
        if first_failure_code == 8:
            detail = (
                "prepared periodic-image coverage is insufficient for system "
                f"{first_failed_system}: cached {int(state._capacity[first_failed_system])} "
                f"shifts, runtime requires {int(state._required[first_failed_system])}"
            )
            raise RuntimeError(
                "prepared neighbor-list state failed for systems "
                f"{failed_systems}; first failure: {detail}"
            )
        if first_failure_code == 3:
            detail = (
                "pair-centric launch metadata is stale; rebuild the launch metadata"
            )
            raise RuntimeError(
                "prepared neighbor-list state failed for systems "
                f"{failed_systems}; first failure: {detail}"
            )
        if first_failure_code == 5:
            axis = int(state._span_axis[failed_systems[0]])
            capacity = float(state._span_capacity[failed_systems[0], axis])
            required = float(state._span_required[failed_systems[0]])
            detail = f"prepared nonperiodic span exceeded for system {failed_systems[0]} on axis {axis}: capacity {capacity}, required {required}"
            raise RuntimeError(
                f"prepared neighbor-list state failed for systems {failed_systems}; first failure: {detail}"
            )
        if first_failure_code == 6:
            detail = f"cannot preserve uninitialized system {failed_systems[0]}"
            raise RuntimeError(
                f"prepared neighbor-list state failed for systems {failed_systems}; first failure: {detail}"
            )
        raise RuntimeError(
            f"prepared neighbor-list state failed for systems {failed_systems}; first failure: runtime validation failed"
        )

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

"""Generic prepared state for the Torch neighbor-list frontends."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from nvalchemiops.neighbors.base_dispatch import neighbor_list_strategy_run_args
from nvalchemiops.neighbors.cluster_tile import estimate_max_tiles_per_group
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.neighbors._cluster_tile_state import (
    _ClusterTileStorage,
    _execute_cluster_tile_storage,
    _prepare_cluster_tile_storage,
)
from nvalchemiops.torch.neighbors._dispatch import (
    _auto_method_from_geometry,
    _reject_unsupported_cluster_tile_combo,
)
from nvalchemiops.torch.neighbors._naive_prepared_guard import (
    check_prepared_naive_shift_coverage,
)
from nvalchemiops.torch.neighbors._resolution import cell_volume
from nvalchemiops.torch.neighbors.batch_cell_list import estimate_batch_cell_list_sizes
from nvalchemiops.torch.neighbors.batch_cluster_tile import (
    estimate_batch_max_tiles_per_group,
)
from nvalchemiops.torch.neighbors.cell_list import estimate_cell_list_sizes
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list,
    compute_naive_num_shifts,
    prepare_batch_idx_ptr,
    synthesize_cell_for_batch,
    synthesize_cell_for_ss,
)

__all__ = ["NeighborListState", "prepare_neighbor_list"]

_FACTORY_TOKEN = object()
_CONFIGURATION_FIELDS = frozenset(
    {
        "method",
        "strategy",
        "format",
        "coo_layout",
        "is_batched",
        "num_atoms",
        "num_systems",
        "cutoff",
        "cutoff2",
        "half_fill",
        "fill_value",
        "wrap_positions",
        "selective",
        "return_vectors",
        "return_distances",
        "span_margin",
        "supports_compilation",
        "compilation_blocker",
    }
)
_RESULT_NAMES = (
    "neighbor_matrix",
    "num_neighbors",
    "neighbor_matrix_shifts",
    "neighbor_matrix1",
    "num_neighbors1",
    "neighbor_matrix_shifts1",
    "neighbor_matrix2",
    "num_neighbors2",
    "neighbor_matrix_shifts2",
    "neighbor_list",
    "neighbor_ptr",
    "neighbor_list_shifts",
    "neighbor_list1",
    "neighbor_ptr1",
    "neighbor_list_shifts1",
    "neighbor_list2",
    "neighbor_ptr2",
    "neighbor_list_shifts2",
    "pair_offsets",
    "pair_counts",
    "metadata_valid",
    "neighbor_vectors",
    "neighbor_distances",
    "pair_energies",
    "pair_forces",
    "tile_offsets",
    "tile_counts",
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
        "inv_cell_batch",
        "inv_cell_buffer",
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
        "neighbor_search_radius",
        "neighbor_vectors",
        "num_neighbors",
        "num_neighbors1",
        "num_neighbors2",
        "num_tiles",
        "pair_counts",
        "pair_counter",
        "pair_energies",
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
_EXECUTION_CONFIG_KEYS = frozenset(
    {
        "atom_centric_path",
        "batch_idx",
        "batch_ptr",
        "cutoff2",
        "fill_value",
        "format",
        "max_atoms_per_system",
        "max_neighbors",
        "max_neighbors1",
        "max_neighbors2",
        "max_pairs",
        "max_shifts_per_system",
        "max_tiles_per_group",
        "max_total_cells",
        "method",
        "num_shifts_per_system",
        "pair_centric_n_outer",
        "pair_centric_r_max",
        "pair_centric_total_cells",
        "pair_fn",
        "pair_params",
        "rebuild_flags",
        "return_distances",
        "return_state",
        "return_vectors",
        "shift_range_per_dimension",
        "span_margin",
        "strategy",
        "target_indices",
        "use_pair_fn",
    }
)
_EXECUTION_KEYS = _CALLER_OWNED | _EXECUTION_CONFIG_KEYS


@dataclass(frozen=True)
class _PreparedRouteConfig:
    """Resolved public route options used to construct prepared state."""

    method: str
    strategy: str
    format: str
    coo_layout: str | None
    is_batched: bool
    num_atoms: int
    num_systems: int
    cutoff: float
    cutoff2: float | None
    max_neighbors: int
    half_fill: bool
    fill_value: int
    wrap_positions: bool
    return_neighbor_list: bool
    selective: bool
    return_vectors: bool
    return_distances: bool
    span_margin: float
    supports_compilation: bool
    compilation_blocker: str | None
    route: str


@dataclass(frozen=True)
class _PreparedMetadata:
    """Fixed device and route metadata for prepared state."""

    device: torch.device
    dtype: torch.dtype
    pbc: torch.Tensor | None
    cell: torch.Tensor | None
    cell_shape: tuple[int, ...] | None
    cell_is_shared: bool
    batch_idx: torch.Tensor | None
    batch_ptr: torch.Tensor | None
    batch_ranges: tuple[tuple[int, int], ...] | None
    target_indices: torch.Tensor | None
    max_neighbors2: int | None
    max_pairs: int | None
    max_tiles_per_group: int | None
    max_atoms_per_system: int | None
    pair_fn: object | None


@dataclass(frozen=True)
class _PreparedStorage:
    """Reusable buffers and latest-result storage for prepared state."""

    buffers: dict[str, torch.Tensor]
    output_names: tuple[str | None, ...]
    public_output_names: tuple[str | None, ...]
    geometry_buffers: dict[str, torch.Tensor]
    span_capacity: torch.Tensor | None
    synthesized_cell: bool
    exemplar_positions: torch.Tensor
    initialized: torch.Tensor


def _build_prepared_state(
    config: _PreparedRouteConfig,
    metadata: _PreparedMetadata,
    storage: _PreparedStorage,
    cluster_storage: _ClusterTileStorage | None = None,
) -> NeighborListState:
    """Construct a prepared state from resolved config, metadata, and storage."""
    return NeighborListState(
        method=config.method,
        strategy=config.strategy,
        format=config.format,
        coo_layout=config.coo_layout,
        is_batched=config.is_batched,
        num_atoms=config.num_atoms,
        num_systems=config.num_systems,
        cutoff=config.cutoff,
        cutoff2=config.cutoff2,
        max_neighbors=config.max_neighbors,
        half_fill=config.half_fill,
        fill_value=config.fill_value,
        wrap_positions=config.wrap_positions,
        return_neighbor_list=config.return_neighbor_list,
        selective=config.selective,
        return_vectors=config.return_vectors,
        return_distances=config.return_distances,
        span_margin=config.span_margin,
        supports_compilation=config.supports_compilation,
        compilation_blocker=config.compilation_blocker,
        device=metadata.device,
        dtype=metadata.dtype,
        route=config.route,
        pbc=metadata.pbc,
        cell=metadata.cell,
        cell_shape=metadata.cell_shape,
        cell_is_shared=metadata.cell_is_shared,
        batch_idx=metadata.batch_idx,
        batch_ptr=metadata.batch_ptr,
        batch_ranges=metadata.batch_ranges,
        target_indices=metadata.target_indices,
        max_neighbors2=metadata.max_neighbors2,
        max_pairs=metadata.max_pairs,
        max_tiles_per_group=metadata.max_tiles_per_group,
        max_atoms_per_system=metadata.max_atoms_per_system,
        pair_fn=metadata.pair_fn,
        buffers=storage.buffers,
        output_names=storage.output_names,
        public_output_names=storage.public_output_names,
        geometry_buffers=storage.geometry_buffers,
        span_capacity=storage.span_capacity,
        synthesized_cell=storage.synthesized_cell,
        exemplar_positions=storage.exemplar_positions,
        initialized=storage.initialized,
        cluster_storage=cluster_storage,
        _factory_token=_FACTORY_TOKEN,
    )


def _validate_positions(positions: torch.Tensor) -> None:
    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a torch.Tensor")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")


def _positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _validate_pbc(
    pbc: torch.Tensor | None,
    positions: torch.Tensor,
    num_systems: int,
) -> torch.Tensor | None:
    if pbc is None:
        return None
    if pbc.dtype != torch.bool or pbc.device != positions.device:
        raise ValueError("pbc must be a bool tensor on the prepared device")
    if pbc.shape not in ((3,), (num_systems, 3)):
        raise ValueError("pbc must have shape (3,) or (num_systems, 3)")
    return pbc.detach().clone().contiguous()


def _validate_batch_metadata(
    batch_idx: torch.Tensor | None,
    batch_ptr: torch.Tensor | None,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if batch_idx is None and batch_ptr is None:
        raise ValueError("Either batch_idx or batch_ptr must be provided.")
    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx,
        batch_ptr,
        positions.shape[0],
        positions.device,
    )
    if batch_idx.dtype != torch.int32 or batch_ptr.dtype != torch.int32:
        raise ValueError("batch metadata must be int32")
    if batch_idx.device != positions.device or batch_ptr.device != positions.device:
        raise ValueError("batch metadata must match positions.device")
    return (
        batch_idx.detach().clone().contiguous(),
        batch_ptr.detach().clone().contiguous(),
        int(batch_ptr.shape[0] - 1),
    )


def _validate_span_margin(value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("span_margin must be a finite nonnegative scalar")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("span_margin must be a finite nonnegative scalar")
    return value


def _is_cluster_route(route: str) -> bool:
    return route in ("cluster_tile", "batch_cluster_tile")


def _state_layout(
    route_names: tuple[str | None, ...],
    geometry_names: tuple[str, ...],
) -> tuple[str | None, ...]:
    """Return the complete state-visible result-name layout."""
    return tuple(dict.fromkeys((*route_names, *geometry_names)))


def _result_property(name: str) -> property:
    """Create a documented read-only view of the latest prepared result."""

    def getter(state: NeighborListState) -> torch.Tensor | None:
        return state._latest[name]

    getter.__name__ = name
    doc = {
        "metadata_valid": "Scalar Boolean recovery-metadata status, or ``None`` when not applicable.",
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
        doc = "Latest CSR pointer for compact COO output, or ``None`` when not applicable."
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
    return property(getter, doc=doc)


class NeighborListState:
    """Prepared Torch neighbor-list configuration and reusable route storage.

    Instances are created by :func:`prepare_neighbor_list`; direct
    construction is unsupported. Configuration properties are read-only.
    Result properties mirror the tuple returned by the latest successful
    prepared execution and return ``None`` when a component does not apply.

    Notes
    -----
    Fixed-size result buffers are borrowed from the state and may be
    overwritten by a later call. Exact-size COO topology tensors have an
    independent lifetime, while the state properties always point to the most
    recent completed result. Structural argument errors rejected before route
    execution leave the state reusable. Preservation, span, or route-execution
    failures permanently invalidate it; after an eager failure, prepare a new
    state. A compiled asynchronous device assertion instead invalidates the
    CUDA context and requires restarting the process.

    The read-only properties below expose the resolved method, strategy,
    output layout, fixed system metadata, compilation eligibility, and current
    initialization status. Result properties use the existing tuple component
    names. Fixed buffers are visible before the first call but contain
    unspecified data while the corresponding ``initialized`` entry is false.
    Exact compact COO properties are ``None`` until the first successful call.
    Non-applicable properties always return ``None``.
    """

    __slots__ = (
        *_CONFIGURATION_FIELDS,
        "_route",
        "_device",
        "_dtype",
        "_pbc",
        "_cell",
        "_cell_shape",
        "_cell_is_shared",
        "_batch_idx",
        "_batch_ptr",
        "_batch_ranges",
        "_target_indices",
        "_max_neighbors2",
        "_max_neighbors",
        "_max_pairs",
        "_max_tiles_per_group",
        "_max_atoms_per_system",
        "_pair_fn",
        "_return_neighbor_list",
        "_native_strategy",
        "_cell_strategy",
        "_atom_centric_path",
        "_buffers",
        "_output_names",
        "_public_output_names",
        "_geometry_buffers",
        "_span_capacity",
        "_synthesized_cell",
        "_exemplar_positions",
        "_initialized",
        "_latest",
        "_failed",
        "_all_initialized",
        "_cluster_storage",
    )

    def __init__(
        self,
        *,
        method: str,
        strategy: str,
        format: str,
        coo_layout: str | None,
        is_batched: bool,
        num_atoms: int,
        num_systems: int,
        cutoff: float,
        cutoff2: float | None,
        max_neighbors: int,
        half_fill: bool,
        fill_value: int,
        wrap_positions: bool,
        return_neighbor_list: bool,
        selective: bool,
        return_vectors: bool,
        return_distances: bool,
        span_margin: float,
        supports_compilation: bool,
        compilation_blocker: str | None,
        device: torch.device,
        dtype: torch.dtype,
        route: str,
        pbc: torch.Tensor | None,
        cell: torch.Tensor | None,
        cell_shape: tuple[int, ...] | None,
        cell_is_shared: bool,
        batch_idx: torch.Tensor | None,
        batch_ptr: torch.Tensor | None,
        batch_ranges: tuple[tuple[int, int], ...] | None,
        target_indices: torch.Tensor | None,
        max_neighbors2: int | None,
        max_pairs: int | None,
        max_tiles_per_group: int | None,
        max_atoms_per_system: int | None,
        pair_fn: object | None,
        buffers: dict[str, torch.Tensor],
        output_names: tuple[str | None, ...],
        public_output_names: tuple[str | None, ...],
        geometry_buffers: dict[str, torch.Tensor],
        span_capacity: torch.Tensor | None,
        synthesized_cell: bool,
        exemplar_positions: torch.Tensor,
        initialized: torch.Tensor,
        cluster_storage: _ClusterTileStorage | None,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _FACTORY_TOKEN:
            raise TypeError(
                "state must be a NeighborListState returned by prepare_neighbor_list"
            )
        self.method = method
        self.strategy = strategy
        self.format = format
        self.coo_layout = coo_layout
        self.is_batched = is_batched
        self.num_atoms = num_atoms
        self.num_systems = num_systems
        self.cutoff = cutoff
        self.cutoff2 = cutoff2
        self.half_fill = half_fill
        self.fill_value = fill_value
        self.wrap_positions = wrap_positions
        self.selective = selective
        self.return_vectors = return_vectors
        self.return_distances = return_distances
        self.span_margin = span_margin
        self.supports_compilation = supports_compilation
        self.compilation_blocker = compilation_blocker
        self._device = device
        self._dtype = dtype
        self._max_neighbors = max_neighbors
        self._route = route
        self._pbc = pbc
        self._cell = cell
        self._cell_shape = cell_shape
        self._cell_is_shared = cell_is_shared
        self._batch_idx = batch_idx
        self._batch_ptr = batch_ptr
        self._batch_ranges = batch_ranges
        self._target_indices = target_indices
        self._max_neighbors2 = max_neighbors2
        self._max_pairs = max_pairs
        self._max_tiles_per_group = max_tiles_per_group
        self._max_atoms_per_system = max_atoms_per_system
        self._pair_fn = pair_fn
        self._return_neighbor_list = return_neighbor_list
        self._native_strategy = "auto"
        self._cell_strategy = "auto"
        self._atom_centric_path = "auto"
        self._buffers = buffers
        self._output_names = output_names
        self._public_output_names = public_output_names
        self._geometry_buffers = geometry_buffers
        self._span_capacity = span_capacity
        self._synthesized_cell = synthesized_cell
        self._exemplar_positions = exemplar_positions
        self._initialized = initialized
        initial = {name: None for name in _RESULT_NAMES}
        # Matrix and tile routes own fixed-capacity output storage.  Make that
        # storage inspectable before first execution, while exact COO remains
        # absent until its first dynamically-sized result is returned.
        if format != "coo" or coo_layout == "segmented":
            visible = set(name for name in output_names if name is not None)
            if _is_cluster_route(route):
                visible.update(
                    {
                        "tile_offsets",
                        "tile_counts",
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
                    }
                )
            for name in visible:
                if name in buffers:
                    initial[name] = buffers[name]
                elif name in geometry_buffers:
                    initial[name] = geometry_buffers[name]
                elif name.endswith("1") and name[:-1] in buffers:
                    initial[name] = buffers[name[:-1]]
        self._latest = initial
        self._failed = False
        self._all_initialized = False
        self._cluster_storage = cluster_storage

    def __setattr__(self, name: str, value: object) -> None:
        if name in _CONFIGURATION_FIELDS and hasattr(self, name):
            raise AttributeError(f"prepared state attribute {name!r} is read-only")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in _CONFIGURATION_FIELDS:
            raise AttributeError(f"prepared state attribute {name!r} is read-only")
        super().__delattr__(name)

    @property
    def initialized(self) -> torch.Tensor:
        """Return a Boolean tensor of shape ``(num_systems,)`` indicating initialized systems."""
        return self._initialized

    neighbor_matrix = _result_property("neighbor_matrix")
    num_neighbors = _result_property("num_neighbors")
    neighbor_matrix_shifts = _result_property("neighbor_matrix_shifts")
    neighbor_matrix1 = _result_property("neighbor_matrix1")
    num_neighbors1 = _result_property("num_neighbors1")
    neighbor_matrix_shifts1 = _result_property("neighbor_matrix_shifts1")
    neighbor_matrix2 = _result_property("neighbor_matrix2")
    num_neighbors2 = _result_property("num_neighbors2")
    neighbor_matrix_shifts2 = _result_property("neighbor_matrix_shifts2")
    neighbor_list = _result_property("neighbor_list")
    neighbor_ptr = _result_property("neighbor_ptr")
    neighbor_list_shifts = _result_property("neighbor_list_shifts")
    neighbor_list1 = _result_property("neighbor_list1")
    neighbor_ptr1 = _result_property("neighbor_ptr1")
    neighbor_list_shifts1 = _result_property("neighbor_list_shifts1")
    neighbor_list2 = _result_property("neighbor_list2")
    neighbor_ptr2 = _result_property("neighbor_ptr2")
    neighbor_list_shifts2 = _result_property("neighbor_list_shifts2")
    pair_offsets = _result_property("pair_offsets")
    pair_counts = _result_property("pair_counts")
    metadata_valid = _result_property("metadata_valid")
    neighbor_vectors = _result_property("neighbor_vectors")
    neighbor_distances = _result_property("neighbor_distances")
    pair_energies = _result_property("pair_energies")
    pair_forces = _result_property("pair_forces")
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

    def _invalidate(self) -> None:
        self._latest = {name: None for name in _RESULT_NAMES}
        self._initialized.zero_()
        self._failed = True
        self._all_initialized = False

    def _publish(self, output: tuple[torch.Tensor, ...]) -> None:
        latest = {name: None for name in _RESULT_NAMES}
        for index, name in enumerate(self._public_output_names):
            if name is not None and index < len(output):
                latest[name] = output[index]
        if _is_cluster_route(self._route):
            for name in (
                "tile_offsets",
                "tile_counts",
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
            ):
                if name in self._buffers:
                    latest[name] = self._buffers[name]
        for name, buffer in self._geometry_buffers.items():
            if self.format == "coo" and latest["neighbor_list"] is not None:
                pair_count = latest["neighbor_list"].shape[-1]
                latest[name] = buffer[:pair_count]
            else:
                # Matrix routes keep fixed row-padded callback/geometry buffers
                # in the state even when the direct route tuple omits them.
                latest[name] = buffer
        self._latest = latest


def _matrix_topology(
    rows: int,
    max_neighbors: int,
    device: torch.device,
    *,
    pbc: bool,
    dual: bool = False,
    max_neighbors2: int | None = None,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}

    def add(suffix: str = "") -> None:
        result[f"neighbor_matrix{suffix}"] = torch.empty(
            (rows, max_neighbors), dtype=torch.int32, device=device
        )
        result[f"num_neighbors{suffix}"] = torch.zeros(
            rows, dtype=torch.int32, device=device
        )
        if pbc:
            result[f"neighbor_matrix_shifts{suffix}"] = torch.empty(
                (rows, max_neighbors, 3), dtype=torch.int32, device=device
            )

    primary_width = max_neighbors
    add()
    if dual:
        max_neighbors = max_neighbors2 or primary_width
        add("2")
    return result


def _geometry_buffers(
    rows: int,
    max_neighbors: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    return_vectors: bool,
    return_distances: bool,
    coo: bool = False,
    max_pairs: int | None = None,
    pair_fn: object | None = None,
) -> dict[str, torch.Tensor]:
    shape = (max_pairs,) if coo else (rows, max_neighbors)
    result: dict[str, torch.Tensor] = {}
    if return_vectors:
        result["neighbor_vectors"] = torch.empty(
            (*shape, 3), dtype=dtype, device=device
        )
    if return_distances:
        result["neighbor_distances"] = torch.empty(shape, dtype=dtype, device=device)
    if pair_fn is not None:
        result["pair_energies"] = torch.empty(shape, dtype=dtype, device=device)
        result["pair_forces"] = torch.empty((*shape, 3), dtype=dtype, device=device)
    return result


def _resolve_method(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor | None,
    pbc: torch.Tensor | None,
    batch_idx: torch.Tensor | None,
    batch_ptr: torch.Tensor | None,
    method: str | None,
    *,
    num_systems: int,
    cutoff2: float | None,
    half_fill: bool,
    return_neighbor_list: bool,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: object | None,
    selective: bool,
    wrap_positions: bool,
    requested_strategy: str,
) -> tuple[str, str]:
    if method is None:
        strategy = _auto_method_from_geometry(
            positions,
            max(cutoff, cutoff2 if cutoff2 is not None else cutoff),
            cell,
            pbc,
            batch_idx,
            batch_ptr,
            num_systems,
            cutoff2=cutoff2,
            half_fill=half_fill,
            return_neighbor_list=return_neighbor_list,
            return_vectors=return_vectors,
            return_distances=return_distances,
            use_pair_fn=pair_fn is not None,
            rebuild_flags=(
                torch.ones(num_systems, dtype=torch.bool, device=positions.device)
                if selective
                else None
            ),
            wrap_positions=wrap_positions,
        )
        if num_systems > 1:
            strategy = "batch_" + strategy
    else:
        strategy = method
        if batch_idx is not None or batch_ptr is not None:
            if not strategy.startswith("batch_"):
                strategy = "batch_" + strategy
        is_batched_method = strategy.startswith("batch_")
        base = strategy[len("batch_") :] if strategy.startswith("batch_") else strategy
        if base in ("naive", "cell_list", "cluster_tile"):
            selected_strategy = requested_strategy
            if selected_strategy == "auto":
                selected_strategy = base
            elif base == "naive" and selected_strategy in ("scalar", "tile"):
                selected_strategy = f"{base}_{selected_strategy}"
            elif base == "cell_list" and selected_strategy in (
                "atom_centric",
                "pair_centric",
            ):
                selected_strategy = f"{base}_{selected_strategy}"
            elif base != "cluster_tile":
                raise ValueError(
                    f"unsupported {base} strategy {requested_strategy!r}; expected "
                    + (
                        "'auto' | 'scalar' | 'tile'"
                        if base == "naive"
                        else "'auto' | 'atom_centric' | 'pair_centric'"
                    )
                )
            strategy = selected_strategy
            if is_batched_method or num_systems > 1:
                strategy = (
                    "batch_" + strategy
                    if not strategy.startswith("batch_")
                    else strategy
                )
        else:
            neighbor_list_strategy_run_args(strategy)
    if strategy in ("naive", "cell_list", "cluster_tile"):
        strategy_name = {
            "naive": "naive_scalar",
            "cell_list": "cell_list_atom_centric",
            "cluster_tile": "cluster_tile",
        }[strategy]
    elif strategy in ("batch_naive", "batch_cell_list", "batch_cluster_tile"):
        strategy_name = {
            "batch_naive": "batch_naive_scalar",
            "batch_cell_list": "batch_cell_list_atom_centric",
            "batch_cluster_tile": "batch_cluster_tile",
        }[strategy]
    else:
        strategy_name = strategy
    base, _native, _cell, _path = neighbor_list_strategy_run_args(strategy_name)
    return base, strategy_name


def _validate_cell_runtime(cell: torch.Tensor | None, state: NeighborListState) -> None:
    if state._synthesized_cell:
        if cell is not None:
            raise ValueError("prepared nonperiodic state requires cell=None")
        return
    if state._cell is None:
        if cell is not None:
            raise ValueError("prepared state does not accept a runtime cell")
        return
    if cell is None:
        return
    if tuple(cell.shape) != state._cell_shape:
        raise ValueError("cell shape does not match prepared state")
    if cell.dtype != state._dtype:
        raise ValueError("cell dtype does not match prepared state")
    if cell.device != state._device:
        raise ValueError("cell device does not match prepared state")


def _prepared_synthetic_geometry(
    positions: torch.Tensor,
    state: NeighborListState,
    rebuild_flags: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if state.is_batched:
        if state._batch_ptr is None:
            raise RuntimeError("prepared batched state is missing batch_ptr")
        shifted = positions.clone()
        required: list[torch.Tensor] = []
        if state._batch_ranges is None:
            raise RuntimeError("prepared batched state is missing batch ranges")
        for start, stop in state._batch_ranges:
            segment = positions[start:stop]
            if segment.shape[0]:
                minimum = segment.min(dim=0).values
                span = segment.max(dim=0).values - minimum
                shifted[start:stop] = segment - minimum
            else:
                span = positions.new_zeros(3)
            required.append(span)
        required_tensor = torch.stack(required)
    else:
        if positions.shape[0]:
            minimum = positions.min(dim=0).values
            required_tensor = (positions.max(dim=0).values - minimum).reshape(1, 3)
            shifted = positions - minimum
        else:
            required_tensor = positions.new_zeros((1, 3))
            shifted = positions
    active = (
        torch.ones(state.num_systems, dtype=torch.bool, device=positions.device)
        if rebuild_flags is None
        else rebuild_flags
    )
    exceeded = required_tensor > state._span_capacity
    if torch.compiler.is_compiling():
        torch._assert_async(
            torch.all(~(exceeded & active[:, None])),
            "prepared nonperiodic span exceeded",
        )
    elif bool((exceeded & active[:, None]).any().item()):
        system, axis = (exceeded & active[:, None]).nonzero(as_tuple=False)[0].tolist()
        capacity = float(state._span_capacity[system, axis].item())
        required_value = float(required_tensor[system, axis].item())
        raise ValueError(
            f"prepared nonperiodic span exceeded for system {system} on axis {axis}: "
            f"capacity {capacity}, required {required_value}"
        )
    cell = torch.diag_embed(state._span_capacity + 0.1 * state.cutoff)
    return shifted, cell if state.is_batched else cell.reshape(1, 3, 3)


def _prepare_buffers(
    *,
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    pbc: torch.Tensor | None,
    route: str,
    format: str,
    cutoff: float,
    max_neighbors: int,
    max_neighbors2: int | None,
    max_pairs: int | None,
    max_tiles_per_group: int | None,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: object | None,
    selective: bool,
    batch_ptr: torch.Tensor | None,
    target_indices: torch.Tensor | None,
    cell_strategy: str,
    num_systems: int,
    cluster_storage: _ClusterTileStorage | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    device = positions.device
    num_atoms = positions.shape[0]
    rows = int(target_indices.shape[0]) if target_indices is not None else num_atoms
    buffers: dict[str, torch.Tensor] = {}
    pbc_enabled = pbc is not None
    dual = route in ("naive_dual_cutoff", "batch_naive_dual_cutoff") or (
        _is_cluster_route(route) and format == "matrix" and max_neighbors2 is not None
    )
    geometry = _geometry_buffers(
        rows,
        max_neighbors,
        positions.dtype,
        device,
        return_vectors=return_vectors,
        return_distances=return_distances,
        coo=_is_cluster_route(route) and format == "coo",
        max_pairs=max_pairs,
        pair_fn=pair_fn,
    )
    if route in (
        "naive",
        "batch_naive",
        "naive_dual_cutoff",
        "batch_naive_dual_cutoff",
    ):
        buffers.update(
            _matrix_topology(
                rows,
                max_neighbors,
                device,
                pbc=pbc_enabled,
                dual=dual,
                max_neighbors2=max_neighbors2,
            )
        )
        if pbc_enabled:
            shift_cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
            shift_pbc = pbc if pbc.ndim == 2 else pbc.unsqueeze(0)
            shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
                shift_cell,
                cutoff,
                shift_pbc,
            )
            buffers["shift_range_per_dimension"] = shift_range
            buffers["num_shifts_per_system"] = num_shifts
            buffers["max_shifts_per_system"] = max_shifts
            buffers["positions_wrapped_buffer"] = torch.empty_like(positions)
            buffers["per_atom_cell_offsets_buffer"] = torch.empty(
                (num_atoms, 3), dtype=torch.int32, device=device
            )
            buffers["inv_cell_buffer"] = torch.empty(
                (num_systems if route.startswith("batch_") else 1, 3, 3),
                dtype=positions.dtype,
                device=device,
            )
            buffers["naive_guard_flags"] = torch.ones(
                num_systems, dtype=torch.bool, device=device
            )
            buffers["naive_guard_status"] = torch.empty(
                num_systems, dtype=torch.int32, device=device
            )
    elif route in ("cell_list", "batch_cell_list"):
        if cell is None or pbc is None:
            raise ValueError("prepared cell-list route requires resolved cell and pbc")
        minimum = 1 if cell_strategy == "atom_centric" else 4
        if route == "cell_list":
            max_cells, radius = estimate_cell_list_sizes(
                cell, pbc, cutoff, min_cells_per_dimension=minimum
            )
        else:
            max_cells, radius = estimate_batch_cell_list_sizes(
                cell, pbc, cutoff, min_cells_per_dimension=minimum
            )
        cache_names = (
            "cells_per_dimension",
            "neighbor_search_radius",
            "atom_periodic_shifts",
            "atom_to_cell_mapping",
            "atoms_per_cell_count",
            "cell_atom_start_indices",
            "cell_atom_list",
        )
        buffers.update(
            dict(
                zip(
                    cache_names,
                    allocate_cell_list(num_atoms, max_cells, radius, device),
                )
            )
        )
        buffers.update(
            neighbor_matrix=torch.empty(
                (rows, max_neighbors), dtype=torch.int32, device=device
            ),
            num_neighbors=torch.zeros(rows, dtype=torch.int32, device=device),
            neighbor_matrix_shifts=torch.empty(
                (rows, max_neighbors, 3), dtype=torch.int32, device=device
            ),
        )
        if route == "cell_list":
            buffers["sorted_positions"] = torch.empty_like(positions)
            buffers["sorted_shifts"] = torch.empty(
                (num_atoms, 3), dtype=torch.int32, device=device
            )
    elif _is_cluster_route(route):
        if cluster_storage is None:
            raise RuntimeError("prepared cluster-tile storage was not initialized")
        if batch_ptr is None:
            names = (
                "sorted_atom_index",
                "morton_codes",
                "sorted_pos_x",
                "sorted_pos_y",
                "sorted_pos_z",
                "group_ctr_x",
                "group_ctr_y",
                "group_ctr_z",
                "group_ext_x",
                "group_ext_y",
                "group_ext_z",
                "num_tiles",
                "tile_row_group",
                "tile_col_group",
            )
        else:
            names = (
                "sorted_atom_index",
                "sort_inv",
                "sorted_pos_x",
                "sorted_pos_y",
                "sorted_pos_z",
                "batch_idx_sorted",
                "batch_ptr_padded",
                "group_system",
                "group_ptr",
                "group_ctr_x",
                "group_ctr_y",
                "group_ctr_z",
                "group_ext_x",
                "group_ext_y",
                "group_ext_z",
                "num_tiles",
                "tile_row_group",
                "tile_col_group",
                "tile_system",
            )
        buffers.update(dict(zip(names, cluster_storage._scratch)))
        if format == "matrix":
            topology_names = (
                "neighbor_matrix",
                "num_neighbors",
                "neighbor_matrix_shifts",
            )
            if max_neighbors2 is not None:
                topology_names += (
                    "neighbor_matrix2",
                    "num_neighbors2",
                    "neighbor_matrix_shifts2",
                )
            buffers.update(dict(zip(topology_names, cluster_storage._topology)))
        elif format == "coo":
            topology_names = ["neighbor_list", "neighbor_list_shifts", "pair_counter"]
            if len(cluster_storage._topology) > 3:
                topology_names.extend(["pair_offsets", "pair_counts"])
            buffers.update(
                dict(
                    zip(
                        topology_names,
                        cluster_storage._topology,
                    )
                )
            )
            if batch_ptr is not None and cluster_storage._selective_state:
                offset_index = 1 if cluster_storage.selective else 0
                buffers["tile_offsets"] = cluster_storage._selective_state[offset_index]
                buffers["tile_counts"] = cluster_storage._selective_state[
                    offset_index + 1
                ]
        geometry = {
            name: value
            for name, value in (
                ("neighbor_vectors", cluster_storage.neighbor_vectors),
                ("neighbor_distances", cluster_storage.neighbor_distances),
                ("pair_energies", cluster_storage._pair_energies),
                ("pair_forces", cluster_storage._pair_forces),
            )
            if value is not None
        }
    return buffers, geometry


def _layout(
    route: str,
    format: str,
    return_neighbor_list: bool,
    pbc: bool,
    dual: bool,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: object | None,
    selective: bool,
    is_batched: bool,
) -> tuple[str | None, ...]:
    if route in (
        "naive",
        "batch_naive",
        "naive_dual_cutoff",
        "batch_naive_dual_cutoff",
    ):
        if return_neighbor_list:
            names: list[str | None] = ["neighbor_list", "neighbor_ptr"]
            if pbc:
                names.append("neighbor_list_shifts")
            if dual:
                names = ["neighbor_list1", "neighbor_ptr1"]
                if pbc:
                    names.append("neighbor_list_shifts1")
                names.extend(["neighbor_list2", "neighbor_ptr2"])
                if pbc:
                    names.append("neighbor_list_shifts2")
        else:
            names = ["neighbor_matrix", "num_neighbors"]
            if pbc:
                names.append("neighbor_matrix_shifts")
            if dual:
                names = ["neighbor_matrix1", "num_neighbors1"]
                if pbc:
                    names.append("neighbor_matrix_shifts1")
                names.extend(["neighbor_matrix2", "num_neighbors2"])
                if pbc:
                    names.append("neighbor_matrix_shifts2")
    elif route in ("cell_list", "batch_cell_list"):
        names = (
            ["neighbor_list", "neighbor_ptr", "neighbor_list_shifts"]
            if return_neighbor_list
            else ["neighbor_matrix", "num_neighbors", "neighbor_matrix_shifts"]
        )
    elif _is_cluster_route(route):
        if format == "tile":
            names = ["num_tiles", "tile_row_group", "tile_col_group"]
            if is_batched:
                names.append("tile_system")
            names.extend(
                [
                    "sorted_atom_index",
                    "sorted_pos_x",
                    "sorted_pos_y",
                    "sorted_pos_z",
                ]
            )
            if is_batched:
                names.extend(["batch_idx_sorted", "batch_ptr_padded", "group_ptr"])
        elif format == "matrix":
            names = ["neighbor_matrix", "num_neighbors", "neighbor_matrix_shifts"]
            if dual:
                names = [
                    "neighbor_matrix1",
                    "num_neighbors1",
                    "neighbor_matrix_shifts1",
                    "neighbor_matrix2",
                    "num_neighbors2",
                    "neighbor_matrix_shifts2",
                ]
        elif is_batched:
            names = [
                "neighbor_list",
                "pair_offsets",
                "pair_counts",
                "neighbor_list_shifts",
            ]
        elif selective:
            names = [
                "neighbor_list",
                "pair_offsets",
                "pair_counts",
                "neighbor_list_shifts",
            ]
        else:
            names = ["neighbor_list", "neighbor_ptr", "neighbor_list_shifts"]
    else:
        raise ValueError(f"unsupported prepared route {route!r}")
    cluster_pair_callback = _is_cluster_route(route) and pair_fn is not None
    if not cluster_pair_callback:
        if return_distances:
            names.append("neighbor_distances")
        if return_vectors:
            names.append("neighbor_vectors")
        if pair_fn is not None and not cluster_pair_callback:
            names.extend(["pair_energies", "pair_forces"])
    return tuple(names)


def _route_kwargs(
    state: NeighborListState, pair_params: torch.Tensor | None
) -> dict[str, object]:
    kwargs: dict[str, object] = dict(state._buffers)
    kwargs.pop("naive_guard_flags", None)
    kwargs.pop("naive_guard_status", None)
    dual = state._route in ("naive_dual_cutoff", "batch_naive_dual_cutoff")
    if dual:
        # The direct dual-cutoff kernels retain their historical ``*1``
        # parameter spelling; the state presents the ordinary primary-result
        # names used by the public dispatcher.
        for name in ("neighbor_matrix", "num_neighbors", "neighbor_matrix_shifts"):
            if name in kwargs:
                kwargs[f"{name}1"] = kwargs.pop(name)
    if not dual:
        kwargs.update(state._geometry_buffers)
        kwargs.update(
            return_vectors=state.return_vectors,
            return_distances=state.return_distances,
            pair_fn=state._pair_fn,
            pair_params=pair_params,
        )
    if state._route in ("naive", "batch_naive", "cell_list", "batch_cell_list"):
        kwargs["target_indices"] = state._target_indices
    return kwargs


def _execute_route(
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    pbc: torch.Tensor | None,
    state: NeighborListState,
    *,
    rebuild_flags: torch.Tensor | None,
    pair_params: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    kwargs = _route_kwargs(state, pair_params)
    if _is_cluster_route(state._route):
        if state._cluster_storage is None:
            raise RuntimeError("prepared cluster-tile state is missing private storage")
        if cell is None:
            raise RuntimeError("prepared cluster-tile execution requires cell metadata")
        return _execute_cluster_tile_storage(
            positions,
            cell,
            state._cluster_storage,
            rebuild_flags=rebuild_flags,
            pair_params=pair_params,
        )
    kwargs["rebuild_flags"] = rebuild_flags
    return_neighbor_list = state._return_neighbor_list
    if state.format == "coo":
        return_neighbor_list = True
    if state._route == "naive":
        from nvalchemiops.torch.neighbors.naive import naive_neighbor_list

        return naive_neighbor_list(
            positions,
            state.cutoff,
            cell=cell,
            pbc=pbc,
            max_neighbors=state._max_neighbors,
            half_fill=state.half_fill,
            fill_value=state.fill_value,
            return_neighbor_list=return_neighbor_list,
            wrap_positions=state.wrap_positions,
            strategy=state._native_strategy,
            **kwargs,
        )
    if state._route == "batch_naive":
        from nvalchemiops.torch.neighbors.batch_naive import batch_naive_neighbor_list

        return batch_naive_neighbor_list(
            positions,
            state.cutoff,
            batch_idx=state._batch_idx,
            batch_ptr=state._batch_ptr,
            pbc=pbc,
            cell=cell,
            max_neighbors=state._max_neighbors,
            max_atoms_per_system=state._max_atoms_per_system,
            half_fill=state.half_fill,
            fill_value=state.fill_value,
            return_neighbor_list=return_neighbor_list,
            wrap_positions=state.wrap_positions,
            strategy=state._native_strategy,
            **kwargs,
        )
    if state._route == "naive_dual_cutoff":
        from nvalchemiops.torch.neighbors.naive_dual_cutoff import (
            naive_neighbor_list_dual_cutoff,
        )

        return naive_neighbor_list_dual_cutoff(
            positions,
            state.cutoff,
            state.cutoff2,
            pbc=pbc,
            cell=cell,
            max_neighbors1=state._max_neighbors,
            max_neighbors2=state._max_neighbors2,
            half_fill=state.half_fill,
            fill_value=state.fill_value,
            return_neighbor_list=return_neighbor_list,
            wrap_positions=state.wrap_positions,
            **kwargs,
        )
    if state._route == "batch_naive_dual_cutoff":
        from nvalchemiops.torch.neighbors.batch_naive_dual_cutoff import (
            batch_naive_neighbor_list_dual_cutoff,
        )

        return batch_naive_neighbor_list_dual_cutoff(
            positions,
            state.cutoff,
            state.cutoff2,
            batch_idx=state._batch_idx,
            batch_ptr=state._batch_ptr,
            pbc=pbc,
            cell=cell,
            max_neighbors1=state._max_neighbors,
            max_neighbors2=state._max_neighbors2,
            half_fill=state.half_fill,
            fill_value=state.fill_value,
            return_neighbor_list=return_neighbor_list,
            wrap_positions=state.wrap_positions,
            **kwargs,
        )
    if state._route == "cell_list":
        from nvalchemiops.torch.neighbors.cell_list import cell_list

        return cell_list(
            positions,
            state.cutoff,
            cell,
            pbc,
            max_neighbors=state._max_neighbors,
            half_fill=state.half_fill,
            fill_value=state.fill_value,
            return_neighbor_list=return_neighbor_list,
            strategy=state._cell_strategy,
            atom_centric_path=state._atom_centric_path,
            **kwargs,
        )
    if state._route == "batch_cell_list":
        from nvalchemiops.torch.neighbors.batch_cell_list import batch_cell_list

        return batch_cell_list(
            positions,
            state.cutoff,
            cell,
            pbc,
            state._batch_idx,
            max_neighbors=state._max_neighbors,
            half_fill=state.half_fill,
            fill_value=state.fill_value,
            return_neighbor_list=return_neighbor_list,
            strategy=state._cell_strategy,
            atom_centric_path=state._atom_centric_path,
            **kwargs,
        )
    if not _is_cluster_route(state._route):
        raise ValueError(f"unsupported prepared route {state._route!r}")
    kwargs.update(
        format=state.format,
        max_neighbors=state._max_neighbors,
        fill_value=state.fill_value,
        max_pairs=state._max_pairs,
        cutoff2=state.cutoff2,
        max_tiles_per_group=state._max_tiles_per_group,
    )
    if state.is_batched:
        from nvalchemiops.torch.neighbors.batch_cluster_tile import (
            batch_cluster_tile_neighbor_list,
        )

        return batch_cluster_tile_neighbor_list(
            positions,
            state.cutoff,
            cell_batch=cell,
            batch_ptr=state._batch_ptr,
            **kwargs,
        )
    from nvalchemiops.torch.neighbors.cluster_tile import cluster_tile_neighbor_list

    return cluster_tile_neighbor_list(positions, state.cutoff, cell, **kwargs)


def _validate_runtime(
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    state: NeighborListState,
    rebuild_flags: torch.Tensor | None,
) -> None:
    if positions.shape != (state.num_atoms, 3):
        raise ValueError("positions shape does not match prepared state")
    if positions.dtype != state._dtype:
        raise ValueError("positions dtype does not match prepared state")
    if positions.device != state._device:
        raise ValueError("positions device does not match prepared state")
    _validate_cell_runtime(cell, state)
    if state.selective and rebuild_flags is None:
        raise ValueError("selective prepared state requires rebuild_flags")
    if rebuild_flags is not None:
        if not state.selective:
            raise ValueError("rebuild_flags requires a selective prepared state")
        if (
            rebuild_flags.dtype != torch.bool
            or rebuild_flags.device != state._device
            or rebuild_flags.shape != (state.num_systems,)
        ):
            raise ValueError(
                "rebuild_flags must be a bool tensor on the prepared device "
                "with shape (num_systems,)"
            )


def _validate_preservation(
    state: NeighborListState,
    rebuild_flags: torch.Tensor | None,
) -> None:
    """Reject preservation before initialization without repeated eager syncs."""
    if not state.selective or rebuild_flags is None:
        return
    if not torch.compiler.is_compiling() and state._all_initialized:
        return
    may_preserve = state.initialized | rebuild_flags
    if torch.compiler.is_compiling():
        torch._assert_async(
            torch.all(may_preserve), "cannot preserve uninitialized system"
        )
    elif bool((~may_preserve).any().item()):
        index = int((~may_preserve).nonzero(as_tuple=False)[0, 0].item())
        raise ValueError(f"cannot preserve uninitialized system {index}")


def _validate_execution_options(kwargs: dict[str, object]) -> None:
    if any(
        name in _CALLER_OWNED and value is not None for name, value in kwargs.items()
    ):
        raise ValueError(
            "caller-owned output or scratch buffers cannot be used with state"
        )
    unknown = next((name for name in kwargs if name not in _EXECUTION_KEYS), None)
    if unknown is not None:
        raise TypeError(
            f"neighbor_list() got an unexpected keyword argument {unknown!r}"
        )


def prepare_neighbor_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    batch_ptr: torch.Tensor | None = None,
    cutoff2: float | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    method: str | None = None,
    format: str | None = None,
    wrap_positions: bool = True,
    *,
    max_neighbors: int | None = None,
    max_neighbors2: int | None = None,
    max_pairs: int | None = None,
    max_tiles_per_group: int | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    selective: bool = False,
    pair_fn: object | None = None,
    span_margin: float = 0.0,
    target_indices: torch.Tensor | None = None,
    strategy: str = "auto",
    atom_centric_path: str = "auto",
    **kwargs: object,
) -> NeighborListState:
    """Resolve and allocate a reusable Torch neighbor-list route.

    Preparation is eager: it resolves automatic method selection, validates
    fixed metadata, estimates omitted capacities, and allocates reusable route
    storage. Execute the result with ``neighbor_list(..., state=state)``.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Exemplar positions. Atom count and order, dtype, and device become part
        of the prepared contract; coordinate values may change at execution.
    cutoff : float
        Primary neighbor cutoff.
    cell : torch.Tensor, shape (3, 3) or (num_systems, 3, 3), optional
        Exemplar cell. If omitted for a resolved cell-list route, preparation
        creates a fixed nonperiodic bounding cell. Applicable cell values may
        change later without changing shape, dtype, device, or PBC pattern.
    pbc : torch.Tensor, shape (3,) or (num_systems, 3), dtype=bool, optional
        Fixed periodic-axis pattern associated with ``cell``.
    batch_idx, batch_ptr : torch.Tensor, dtype=int32, optional
        Fixed, system-contiguous batch ownership metadata.
    cutoff2 : float, optional
        Secondary cutoff. Naive dual-cutoff routes require
        ``cutoff2 >= cutoff``. Cluster-tile routes retain their direct API
        contract and accept either order.
    half_fill : bool, default=False
        Store one directed half of each pair where the route supports it.
    fill_value : int, optional
        Neighbor-matrix padding value. Defaults to ``total_atoms``.
    return_neighbor_list : bool, default=False
        Request the route's COO representation instead of matrix output.
    method : str, optional
        Method accepted by :func:`neighbor_list`, including batched and
        dual-cutoff names. ``None`` runs automatic selection once during
        preparation.
    format : {"matrix", "coo", "tile"}, optional
        Fixed output representation. Matrix and COO values must agree with
        ``return_neighbor_list``; therefore ``format="coo"`` also requires
        ``return_neighbor_list=True``. Tile output requires a nonselective
        cluster-tile route.
    wrap_positions : bool, default=True
        Whether applicable naive routes wrap positions before enumeration.
    max_neighbors : int, optional
        Primary matrix row width. Omitted values are estimated eagerly where
        the selected route supports estimation.
    max_neighbors2 : int, optional
        Secondary row width for naive dual-cutoff routes. Cluster-tile routes
        use ``max_neighbors`` for both groups and reject this argument.
    max_pairs : int, optional
        COO pair capacity for cluster-tile routes.
    max_tiles_per_group : int, optional
        Cluster-tile intermediate capacity factor.
    return_vectors, return_distances : bool, default=False
        Request the corresponding existing pair-geometry outputs.
    selective : bool, default=False
        Prepare a route that updates only systems selected by runtime
        ``rebuild_flags``.
    pair_fn : warp.Function or CompiledPairFn, optional
        Warp pair function using the direct neighbor-route callback contract.
        Runtime ``pair_params`` are supplied to :func:`neighbor_list`.
    span_margin : float, default=0.0
        Finite nonnegative per-axis span-growth allowance for a synthesized
        nonperiodic cell-list cell. For each axis, runtime span may not exceed
        exemplar span plus this value; equality is allowed. A nonzero value
        requires ``cell=None`` and a resolved cell-list route. This is not a
        neighbor-list skin.
    target_indices : torch.Tensor, shape (num_rows,), dtype=int32, optional
        Fixed compact source-row selection on the prepared device for routes
        that support partial rows.
    strategy : str, default="auto"
        Route-specific implementation strategy, such as ``"scalar"`` or
        ``"tile"`` for naive and ``"atom_centric"`` or ``"pair_centric"``
        for cell lists. ``"auto"`` resolves once during preparation.
    atom_centric_path : str, default="auto"
        Existing atom-centric cell-list implementation path.
    **kwargs : object
        No additional names are accepted; every supplied name raises
        ``TypeError``.

    Returns
    -------
    NeighborListState
        Mutable state for repeated Torch execution.

    Notes
    -----
    ``supports_compilation`` reports route eligibility, not compiler history or
    a guarantee for every environment. A non-``None``
    ``compilation_blocker`` explains known route ineligibility.

    Unsupported prepared combinations include dual-cutoff pair geometry or
    callbacks, cluster-tile partial rows, selective cluster-tile tile output,
    and selective cluster-tile pair outputs. Cluster-tile preparation also
    follows the direct route's CUDA, float32, fully periodic, and
    ``half_fill=False`` prerequisites.

    Prepared pair callbacks are eager-only. Batched selective cluster-tile COO
    is not a ``torch.compile(fullgraph=True)`` boundary; single-system
    selective cluster-tile COO is supported. Naive pair outputs use the scalar
    pair-output path even when ``strategy="tile"`` was requested; tiled
    execution applies only to the topology path.

    See Also
    --------
    neighbor_list : Execute the prepared state.
    NeighborListState : Inspect resolved configuration and latest results.
    """
    _validate_positions(positions)
    cutoff = _positive_float("cutoff", cutoff)
    if cutoff is None:
        raise ValueError("cutoff must be positive")
    cutoff2 = _positive_float("cutoff2", cutoff2)
    span_margin = _validate_span_margin(span_margin)
    if kwargs:
        unknown = next(iter(kwargs))
        raise TypeError(
            f"prepare_neighbor_list() got an unexpected keyword argument {unknown!r}"
        )
    max_neighbors = _positive_int("max_neighbors", max_neighbors)
    max_neighbors2 = _positive_int("max_neighbors2", max_neighbors2)
    max_pairs = _positive_int("max_pairs", max_pairs)
    max_tiles_per_group = _positive_int("max_tiles_per_group", max_tiles_per_group)
    if not isinstance(half_fill, bool) or not isinstance(wrap_positions, bool):
        raise ValueError("half_fill and wrap_positions must be Boolean values")
    if not isinstance(return_vectors, bool) or not isinstance(return_distances, bool):
        raise ValueError("return_vectors and return_distances must be Boolean values")
    if not isinstance(selective, bool):
        raise ValueError("selective must be a Boolean")
    if target_indices is not None and (
        target_indices.dtype != torch.int32
        or target_indices.device != positions.device
        or target_indices.ndim != 1
    ):
        raise ValueError(
            "target_indices must be an int32 tensor on the prepared device"
        )
    if batch_idx is not None or batch_ptr is not None:
        protected_batch_idx, protected_batch_ptr, num_systems = (
            _validate_batch_metadata(batch_idx, batch_ptr, positions)
        )
    else:
        protected_batch_idx = protected_batch_ptr = None
        num_systems = int(cell.shape[0]) if cell is not None and cell.ndim == 3 else 1
    resolved, supplied_strategy = _resolve_method(
        positions,
        cutoff,
        cell,
        pbc,
        protected_batch_idx,
        protected_batch_ptr,
        method,
        num_systems=num_systems,
        cutoff2=cutoff2,
        half_fill=half_fill,
        return_neighbor_list=return_neighbor_list,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        selective=selective,
        wrap_positions=wrap_positions,
        requested_strategy=strategy,
    )
    is_batched = resolved.startswith("batch_")
    route = resolved
    if is_batched and protected_batch_ptr is None:
        raise ValueError("batch metadata is required for a prepared batched route")
    if (
        route in ("naive", "batch_naive", "cell_list", "batch_cell_list")
        and cutoff2 is not None
    ):
        route = "batch_naive_dual_cutoff" if is_batched else "naive_dual_cutoff"
    if cutoff2 is not None and (
        return_vectors or return_distances or pair_fn is not None
    ):
        raise ValueError("dual-cutoff prepared execution does not support pair outputs")
    if _is_cluster_route(route):
        if positions.dtype != torch.float32 or not positions.is_cuda:
            raise ValueError("cluster-tile preparation requires CUDA float32 positions")
        derived_format = "coo" if return_neighbor_list else "matrix"
        if format is None:
            format = derived_format
        elif format != derived_format and not (
            format == "tile" and not return_neighbor_list
        ):
            raise ValueError(
                "format contradicts return_neighbor_list; omit format or use "
                f"format={derived_format!r}"
            )
        format = str(format)
        if format not in ("matrix", "coo", "tile"):
            raise ValueError("format must be 'matrix' | 'coo' | 'tile'")
        if cutoff2 is not None and format != "matrix":
            raise ValueError(
                "cluster_tile cutoff2 is supported only with format='matrix'"
            )
        if max_neighbors2 is not None:
            raise ValueError(
                "max_neighbors2 is not supported by cluster-tile preparation"
            )
        if target_indices is not None:
            raise ValueError(
                "target_indices is not supported by cluster-tile preparation"
            )
        if selective and format == "tile":
            raise ValueError(
                "cluster-tile selective rebuild is not supported with tile output"
            )
        if selective and (return_vectors or return_distances or pair_fn is not None):
            raise ValueError(
                "selective prepared cluster-tile execution does not support pair outputs"
            )
    else:
        derived_format = "coo" if return_neighbor_list else "matrix"
        if format is not None and format != derived_format:
            raise ValueError(
                "format contradicts return_neighbor_list; omit format or use "
                f"format={derived_format!r}"
            )
        format = derived_format
    synthesized = False
    span_capacity: torch.Tensor | None = None
    if route in ("cell_list", "batch_cell_list") and cell is None:
        synthesized = True
        if is_batched:
            if protected_batch_idx is None or protected_batch_ptr is None:
                raise ValueError(
                    "batch metadata is required for synthesized batch cells"
                )
            exemplar_positions, resolved_cell, resolved_pbc = synthesize_cell_for_batch(
                positions, protected_batch_idx, protected_batch_ptr, cutoff
            )
        else:
            exemplar_positions, resolved_cell, resolved_pbc = synthesize_cell_for_ss(
                positions, cutoff
            )
        spans: list[torch.Tensor] = []
        if is_batched:
            if protected_batch_ptr is None:
                raise RuntimeError("prepared batched state is missing batch_ptr")
            for start, stop in zip(
                protected_batch_ptr[:-1].tolist(), protected_batch_ptr[1:].tolist()
            ):
                segment = positions[start:stop]
                spans.append(
                    segment.max(dim=0).values - segment.min(dim=0).values
                    if segment.shape[0]
                    else positions.new_zeros(3)
                )
        else:
            spans.append(
                positions.max(dim=0).values - positions.min(dim=0).values
                if positions.shape[0]
                else positions.new_zeros(3)
            )
        span_capacity = torch.stack(spans) + span_margin
    else:
        exemplar_positions = positions
        resolved_cell = cell
        resolved_pbc = pbc
        if span_margin:
            raise ValueError(
                "nonzero span_margin requires cell=None and a resolved nonperiodic cell-list method"
            )
    if route not in ("cell_list", "batch_cell_list") and span_margin:
        raise ValueError(
            "nonzero span_margin requires cell=None and a resolved nonperiodic cell-list method"
        )
    cell_shape = tuple(resolved_cell.shape) if resolved_cell is not None else None
    cell_is_shared = route == "batch_cluster_tile" and cell_shape == (3, 3)
    if resolved_cell is not None:
        if (
            resolved_cell.dtype != positions.dtype
            or resolved_cell.device != positions.device
        ):
            raise ValueError("cell must match positions dtype and device")
    resolved_pbc = _validate_pbc(resolved_pbc, positions, num_systems)
    if route in ("cell_list", "batch_cell_list") and resolved_pbc is None:
        raise ValueError("cell-list preparation requires pbc metadata")
    if _is_cluster_route(route):
        _reject_unsupported_cluster_tile_combo(resolved_pbc, half_fill)
        if (
            resolved_cell is None
            or resolved_pbc is None
            or not bool(resolved_pbc.all().item())
        ):
            raise ValueError("prepared cluster-tile state requires fully periodic pbc")
    if cell_is_shared:
        resolved_cell = resolved_cell.expand(num_systems, -1, -1).contiguous()
    if route in (
        "naive",
        "batch_naive",
        "naive_dual_cutoff",
        "batch_naive_dual_cutoff",
    ):
        if resolved_cell is not None and resolved_pbc is None:
            raise ValueError("pbc is required when cell is provided")
        if resolved_cell is None and resolved_pbc is not None:
            raise ValueError("cell is required when pbc is provided")
    if fill_value is None:
        fill_value = positions.shape[0]
    elif not isinstance(fill_value, int) or isinstance(fill_value, bool):
        raise ValueError("fill_value must be an integer")
    if max_neighbors is None:
        max_neighbors = max(
            estimate_max_neighbors(
                max(float(cutoff), float(cutoff2 or cutoff))
                if _is_cluster_route(route)
                else cutoff
            ),
            32 if _is_cluster_route(route) else 1,
        )
    if (
        route in ("naive_dual_cutoff", "batch_naive_dual_cutoff")
        and max_neighbors2 is None
    ):
        max_neighbors2 = max(estimate_max_neighbors(cutoff2 or cutoff), 1)
    if _is_cluster_route(route) and max_tiles_per_group is None:
        build_cutoff = max(float(cutoff), float(cutoff2 or cutoff))
        if protected_batch_ptr is None:
            max_tiles_per_group = estimate_max_tiles_per_group(
                positions.shape[0], build_cutoff, cell_volume(resolved_cell)
            )
        else:
            max_tiles_per_group = estimate_batch_max_tiles_per_group(
                protected_batch_ptr, build_cutoff, resolved_cell
            )
    if _is_cluster_route(route) and max_pairs is None and positions.shape[0]:
        max_pairs = positions.shape[0] * max_neighbors
    if _is_cluster_route(route) and cutoff2 is not None and max_neighbors2 is None:
        max_neighbors2 = max_neighbors
    native = "auto"
    cell_strategy = "auto"
    atom_path = atom_centric_path
    if "naive_scalar" in supplied_strategy:
        native = "scalar"
    elif "naive_tile" in supplied_strategy:
        native = "tile"
    if "pair_centric" in supplied_strategy:
        cell_strategy = "pair_centric"
    elif "atom_centric" in supplied_strategy:
        cell_strategy = "atom_centric"
    cluster_storage = None
    if _is_cluster_route(route):
        if resolved_cell is None:
            raise RuntimeError("prepared cluster-tile state is missing cell metadata")
        cluster_storage = _prepare_cluster_tile_storage(
            positions,
            cutoff,
            resolved_cell,
            format=format,
            batch_ptr=protected_batch_ptr,
            selective=selective,
            max_neighbors=max_neighbors,
            fill_value=fill_value,
            max_pairs=max_pairs,
            cutoff2=cutoff2,
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=pair_fn,
            max_tiles_per_group=max_tiles_per_group,
        )
    buffers, geometry = _prepare_buffers(
        positions=positions,
        cell=resolved_cell,
        pbc=resolved_pbc,
        route=route,
        format=format,
        cutoff=max(float(cutoff), float(cutoff2 or cutoff)),
        max_neighbors=max_neighbors,
        max_neighbors2=max_neighbors2,
        max_pairs=max_pairs,
        max_tiles_per_group=max_tiles_per_group,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        selective=selective,
        batch_ptr=protected_batch_ptr,
        target_indices=target_indices,
        cell_strategy=cell_strategy,
        num_systems=num_systems,
        cluster_storage=cluster_storage,
    )
    public_output_names = _layout(
        route,
        format,
        return_neighbor_list,
        resolved_pbc is not None,
        route in ("naive_dual_cutoff", "batch_naive_dual_cutoff")
        or (_is_cluster_route(route) and cutoff2 is not None),
        return_vectors,
        return_distances,
        pair_fn,
        selective,
        is_batched,
    )
    output_names = _state_layout(
        public_output_names,
        tuple(geometry),
    )
    initialized = torch.zeros(num_systems, dtype=torch.bool, device=positions.device)
    unsupported_compiled_coo = (
        _is_cluster_route(route) and format == "coo" and selective and is_batched
    )
    supports_compilation = pair_fn is None and not unsupported_compiled_coo
    blocker = (
        "pair callbacks are eager-only"
        if pair_fn is not None
        else "batched selective cluster-tile COO is not fullgraph-supported"
        if unsupported_compiled_coo
        else None
    )
    route_config = _PreparedRouteConfig(
        method=route,
        strategy=supplied_strategy,
        format=format,
        coo_layout=(
            "segmented"
            if format == "coo"
            and _is_cluster_route(route)
            and (selective or is_batched)
            else "compact"
            if format == "coo"
            else None
        ),
        is_batched=is_batched,
        num_atoms=positions.shape[0],
        num_systems=num_systems,
        cutoff=float(cutoff),
        cutoff2=float(cutoff2) if cutoff2 is not None else None,
        max_neighbors=max_neighbors,
        half_fill=half_fill,
        fill_value=fill_value,
        wrap_positions=wrap_positions,
        return_neighbor_list=return_neighbor_list,
        selective=selective,
        return_vectors=return_vectors,
        return_distances=return_distances,
        span_margin=span_margin,
        supports_compilation=supports_compilation,
        compilation_blocker=blocker,
        route=route,
    )
    metadata = _PreparedMetadata(
        device=positions.device,
        dtype=positions.dtype,
        pbc=resolved_pbc,
        cell=resolved_cell,
        cell_shape=cell_shape,
        cell_is_shared=cell_is_shared,
        batch_idx=protected_batch_idx,
        batch_ptr=protected_batch_ptr,
        batch_ranges=(
            tuple(
                zip(
                    protected_batch_ptr[:-1].tolist(),
                    protected_batch_ptr[1:].tolist(),
                )
            )
            if protected_batch_ptr is not None
            else None
        ),
        target_indices=target_indices.detach().clone()
        if target_indices is not None
        else None,
        max_neighbors2=max_neighbors2,
        max_pairs=max_pairs,
        max_tiles_per_group=max_tiles_per_group,
        max_atoms_per_system=(
            int((protected_batch_ptr[1:] - protected_batch_ptr[:-1]).max().item())
            if protected_batch_ptr is not None
            else None
        ),
        pair_fn=pair_fn,
    )
    storage = _PreparedStorage(
        buffers=buffers,
        output_names=output_names,
        public_output_names=public_output_names,
        geometry_buffers=geometry,
        span_capacity=span_capacity,
        synthesized_cell=synthesized,
        exemplar_positions=exemplar_positions.detach().clone(),
        initialized=initialized,
    )
    state = _build_prepared_state(
        route_config, metadata, storage, cluster_storage=cluster_storage
    )

    state._native_strategy = native
    state._cell_strategy = cell_strategy
    state._atom_centric_path = atom_path
    return state


def _execute_prepared_neighbor_list(
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    state: NeighborListState,
    *,
    kwargs: dict[str, object],
) -> tuple[torch.Tensor, ...]:
    """Execute a previously prepared generic state."""
    if not isinstance(state, NeighborListState):
        raise TypeError(
            "state must be a NeighborListState returned by prepare_neighbor_list"
        )
    if state._failed:
        raise RuntimeError(
            "prepared Torch neighbor-list state is invalid; call prepare_neighbor_list again"
        )
    _validate_execution_options(kwargs)
    rebuild_flags = kwargs.get("rebuild_flags")
    pair_params = kwargs.get("pair_params")
    _validate_runtime(positions, cell, state, rebuild_flags)
    if state._pair_fn is not None and pair_params is None:
        raise ValueError("pair_params is required when pair_fn is provided")
    if pair_params is not None:
        if not isinstance(pair_params, torch.Tensor):
            raise TypeError("pair_params must be a torch.Tensor")
        if pair_params.device != state._device:
            raise ValueError("pair_params must match prepared device")
    try:
        _validate_preservation(state, rebuild_flags)
        if state._synthesized_cell:
            call_positions, call_cell = _prepared_synthetic_geometry(
                positions, state, rebuild_flags
            )
            call_pbc = torch.zeros(
                (state.num_systems, 3) if state.is_batched else (3,),
                dtype=torch.bool,
                device=state._device,
            )
        else:
            call_positions = positions
            call_cell = state._cell if cell is None else cell
            call_pbc = state._pbc
            if state._cell_is_shared:
                call_cell = call_cell.expand(state.num_systems, -1, -1).contiguous()
        if (
            state._route
            in (
                "naive",
                "batch_naive",
                "naive_dual_cutoff",
                "batch_naive_dual_cutoff",
            )
            and call_pbc is not None
        ):
            guard_cell = call_cell if call_cell.ndim == 3 else call_cell.unsqueeze(0)
            guard_pbc = call_pbc if call_pbc.ndim == 2 else call_pbc.unsqueeze(0)
            guard_flags = (
                rebuild_flags
                if rebuild_flags is not None
                else state._buffers["naive_guard_flags"]
            )
            check_prepared_naive_shift_coverage(
                guard_cell,
                max(state.cutoff, state.cutoff2 or state.cutoff),
                guard_pbc,
                state._buffers["shift_range_per_dimension"],
                guard_flags,
                state._buffers["naive_guard_status"],
            )
        output = _execute_route(
            call_positions,
            call_cell,
            call_pbc,
            state,
            rebuild_flags=rebuild_flags,
            pair_params=pair_params,
        )
        state._publish(output)
        if state.selective:
            state.initialized.copy_(state.initialized | rebuild_flags)
        else:
            state.initialized.fill_(True)
        if not torch.compiler.is_compiling():
            state._all_initialized = True
        return output[: len(state._public_output_names)]
    except Exception:
        state._invalidate()
        raise

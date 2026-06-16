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

"""Torch fullgraph support for pre-specialized neighbor-list pair functions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from threading import Lock
from typing import TypeVar

import warp as wp

__all__ = ["CompiledPairFn", "compile_pair_fn", "is_compiled_pair_fn"]

_T = TypeVar("_T")
_COUNTER = count()
_LOCK = Lock()


def _sanitize_name(name: str) -> str:
    """Return a torch-library-safe name fragment."""
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    cleaned = cleaned.strip("_")
    return cleaned or "pair_fn"


@dataclass(frozen=True)
class CompiledPairFn:
    """Pre-specialized Warp pair function for Torch fullgraph custom ops."""

    pair_fn: wp.Function
    op_prefix: str
    _ops: dict[str, object] = field(default_factory=dict, compare=False, hash=False)

    def op_name(self, route: str) -> str:
        """Return the stable generated torch-library op name for a route."""
        return f"{self.op_prefix}_{route}"

    def get_or_register(
        self, route: str, factory: Callable[[CompiledPairFn], _T]
    ) -> _T:
        """Return the cached route op, registering it on first use."""
        op = self._ops.get(route)
        if op is not None:
            return op  # type: ignore[return-value]
        with _LOCK:
            op = self._ops.get(route)
            if op is None:
                op = factory(self)
                self._ops[route] = op
            return op  # type: ignore[return-value]


def compile_pair_fn(pair_fn: wp.Function, *, name: str | None = None) -> CompiledPairFn:
    """Pre-specialize a Warp pair function for Torch ``fullgraph`` compilation.

    Use this helper before entering ``torch.compile(fullgraph=True)`` when a
    neighbor-list call needs ``pair_fn`` outputs. Raw ``wp.Function`` values keep
    their normal eager behavior, but they cannot cross a Torch custom-op schema
    in fullgraph mode; a :class:`CompiledPairFn` registers route-specific custom
    ops whose Python closures have already captured the Warp function.

    Fullgraph support is limited to fixed-shape matrix-output calls on
    ``naive_neighbor_list``, ``batch_naive_neighbor_list``, ``cell_list``, and
    ``batch_cell_list``. The compiled call site must provide all fixed buffers
    and metadata needed by that route, including neighbor matrix/count/shift
    buffers, distance/vector buffers when requested, pair energy/force buffers,
    PBC shift metadata for naive PBC routes, and cell-list caches/scratch for
    cell-list routes. Compact ``target_indices`` rows are supported on those
    matrix routes when the provided buffers use ``len(target_indices)`` rows.

    COO output (``return_neighbor_list=True``), cluster-tile methods, and
    automatic method selection are outside this first fullgraph specialization.
    Use the direct method-specific matrix APIs in compiled regions.

    Parameters
    ----------
    pair_fn : wp.Function
        Module-scope ``@wp.func`` with the neighbor-list pair-function
        signature ``(r_ij, distance, pair_params, i, j) -> (energy, force)``.
    name : str, optional
        Human-readable name fragment used in generated Torch custom-op names.

    Returns
    -------
    CompiledPairFn
        Wrapper accepted by Torch neighbor-list ``pair_fn`` kwargs.
    """
    base = name or getattr(pair_fn, "__name__", pair_fn.__class__.__name__)
    suffix = next(_COUNTER)
    compiled = CompiledPairFn(
        pair_fn=pair_fn,
        op_prefix=f"_compiled_pair_fn_{_sanitize_name(base)}_{suffix}",
    )
    _register_default_neighbor_routes(compiled)
    return compiled


def is_compiled_pair_fn(pair_fn: object) -> bool:
    """Return whether ``pair_fn`` is a compiled pair-function wrapper."""
    return isinstance(pair_fn, CompiledPairFn)


def _register_default_neighbor_routes(compiled: CompiledPairFn) -> None:
    """Pre-register the fixed-shape Torch neighbor pair_fn routes."""
    from nvalchemiops.torch.neighbors.batch_cell_list import (
        _register_compiled_batch_query_cell_list_optional_pair_op,
    )
    from nvalchemiops.torch.neighbors.batch_naive import (
        _register_compiled_batch_naive_no_pbc_pair_op,
        _register_compiled_batch_naive_pbc_pair_op,
    )
    from nvalchemiops.torch.neighbors.cell_list import (
        _register_compiled_query_cell_list_optional_pair_op,
    )
    from nvalchemiops.torch.neighbors.naive import (
        _register_compiled_naive_no_pbc_pair_op,
        _register_compiled_naive_pbc_pair_op,
    )

    compiled.get_or_register(
        "naive_no_pbc_pair",
        _register_compiled_naive_no_pbc_pair_op,
    )
    compiled.get_or_register(
        "naive_pbc_pair",
        _register_compiled_naive_pbc_pair_op,
    )
    compiled.get_or_register(
        "batch_naive_no_pbc_pair",
        _register_compiled_batch_naive_no_pbc_pair_op,
    )
    compiled.get_or_register(
        "batch_naive_pbc_pair",
        _register_compiled_batch_naive_pbc_pair_op,
    )
    compiled.get_or_register(
        "query_cell_list_optional_pair",
        _register_compiled_query_cell_list_optional_pair_op,
    )
    compiled.get_or_register(
        "batch_query_cell_list_optional_pair",
        _register_compiled_batch_query_cell_list_optional_pair_op,
    )

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

"""Pair-output argument helpers for neighbor-list implementations."""

from __future__ import annotations

from typing import Any

import warp as wp

from nvalchemiops.neighbors.neighbor_utils import DTYPE_INFO_ALL, empty_sentinel

__all__ = [
    "_has_partial_or_pair_outputs",
    "_is_empty_pair_params",
    "_prepare_pair_output_args",
    "_prepare_coo_pair_output_args",
]


def _array_ndim(array: Any) -> int:
    """Return the array rank using ``ndim`` when available."""
    ndim = getattr(array, "ndim", None)
    if ndim is not None:
        return int(ndim)
    return len(array.shape)


def _is_empty_pair_params(pair_params: wp.array | None) -> bool:
    """Return whether ``pair_params`` is the inactive ``(0, 0)`` sentinel."""
    if pair_params is None or _array_ndim(pair_params) != 2:
        return False
    return int(pair_params.shape[0]) == 0 and int(pair_params.shape[1]) == 0


def _require_present(value: Any | None, name: str, reason: str) -> Any:
    """Return ``value`` or raise a consistent missing-buffer error."""
    if value is None:
        raise ValueError(f"{name} is required {reason}")
    return value


def _reject_present(value: Any | None, name: str, reason: str) -> None:
    """Raise when ``value`` is populated for an inactive pair output."""
    if value is not None:
        raise ValueError(f"{name} is only valid {reason}")


def _validate_pair_params_ndim(pair_params: wp.array) -> None:
    """Validate the pair-parameter array rank.

    Only the rank (``ndim == 2``) is checked; the second dimension ``K`` is not
    validated against ``pair_fn`` (there is no compile-time ``K``), so the caller
    must size ``pair_params`` to cover every column the functor reads.
    """
    if _array_ndim(pair_params) != 2:
        raise ValueError("pair_params must have ndim == 2")


def _validate_pair_output_kwargs(
    *,
    wp_dtype: type,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: wp.Function | None,
    pair_params: wp.array | None,
    neighbor_vectors: wp.array | None,
    neighbor_distances: wp.array | None,
    pair_energies: wp.array | None,
    pair_forces: wp.array | None,
    neighbor_vectors_name: str,
    neighbor_distances_name: str,
    pair_energies_name: str,
    pair_forces_name: str,
) -> None:
    """Run the shared pair-output kwarg consistency rules.

    Rejects orphan buffers (provided without their matching flag),
    enforces ``pair_fn`` ↔ ``pair_params`` coupling, and checks the
    ``pair_params`` ndim / dtype contract.  Buffer-extent checks are
    format-specific and are *not* performed here: the COO variant
    (:func:`_prepare_coo_pair_output_args`) validates per-pair buffer
    length against ``max_pairs``, while the matrix variant
    (:func:`_prepare_pair_output_args`) trusts the caller to size the
    per-pair buffers to match ``neighbor_matrix`` (the matrix launchers are
    a low-level, caller-owns-allocation API).
    """
    if pair_params is not None:
        _validate_pair_params_ndim(pair_params)

    if return_vectors:
        _require_present(
            neighbor_vectors,
            neighbor_vectors_name,
            "when return_vectors=True",
        )
    else:
        _reject_present(
            neighbor_vectors,
            neighbor_vectors_name,
            "when return_vectors=True",
        )
    if return_distances:
        _require_present(
            neighbor_distances,
            neighbor_distances_name,
            "when return_distances=True",
        )
    else:
        _reject_present(
            neighbor_distances,
            neighbor_distances_name,
            "when return_distances=True",
        )

    if pair_fn is None:
        _reject_present(
            pair_energies,
            pair_energies_name,
            "when pair_fn is provided",
        )
        _reject_present(
            pair_forces,
            pair_forces_name,
            "when pair_fn is provided",
        )
        if pair_params is not None and not _is_empty_pair_params(pair_params):
            raise ValueError("pair_params is only valid when pair_fn is provided")
        return

    if pair_params is None:
        raise ValueError("pair_params is required when pair_fn is provided")
    if pair_params.dtype != wp_dtype:
        raise ValueError("pair_params must use the same scalar dtype as positions")
    _require_present(pair_energies, pair_energies_name, "when pair_fn is provided")
    _require_present(pair_forces, pair_forces_name, "when pair_fn is provided")


def _has_partial_or_pair_outputs(
    *,
    target_indices: Any | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    neighbor_vectors: Any | None = None,
    neighbor_distances: Any | None = None,
    pair_energies: Any | None = None,
    pair_forces: Any | None = None,
) -> bool:
    """Return True when any optional pair-output argument is active."""
    return (
        target_indices is not None
        or bool(return_vectors)
        or bool(return_distances)
        or pair_fn is not None
        or (pair_params is not None and not _is_empty_pair_params(pair_params))
        or neighbor_vectors is not None
        or neighbor_distances is not None
        or pair_energies is not None
        or pair_forces is not None
    )


def _prepare_pair_output_args(
    wp_dtype: type,
    device,
    *,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: wp.Function | None,
    pair_params: wp.array | None,
    neighbor_vectors: wp.array | None = None,
    neighbor_distances: wp.array | None = None,
    pair_energies: wp.array | None = None,
    pair_forces: wp.array | None = None,
    neighbor_vectors_name: str = "neighbor_vectors",
    neighbor_distances_name: str = "neighbor_distances",
    pair_energies_name: str = "pair_energies",
    pair_forces_name: str = "pair_forces",
) -> tuple[wp.array, wp.array, wp.array, wp.array, wp.array]:
    """Validate pair-output arguments and return kernel-ready arrays.

    Per-algorithm support
    ---------------------
    All neighbor-list families accept ``return_vectors``,
    ``return_distances``, ``pair_fn``, and ``pair_params`` through this
    helper.  The matrix and COO variants share the rejection rules; the
    only difference is the shape of the returned per-pair buffers.

    ``target_indices`` is validated upstream by the naive and cell-list
    launchers.  Cluster-tile launchers do not accept a ``target_indices``
    kwarg because those kernels iterate emitted tile pairs rather than
    source atoms.
    """
    _validate_pair_output_kwargs(
        wp_dtype=wp_dtype,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
        neighbor_vectors_name=neighbor_vectors_name,
        neighbor_distances_name=neighbor_distances_name,
        pair_energies_name=pair_energies_name,
        pair_forces_name=pair_forces_name,
    )

    vec_dtype = DTYPE_INFO_ALL[wp_dtype][0]
    neighbor_vectors_arg = (
        neighbor_vectors if return_vectors else empty_sentinel(2, vec_dtype, device)
    )
    neighbor_distances_arg = (
        neighbor_distances if return_distances else empty_sentinel(2, wp_dtype, device)
    )
    if pair_fn is None:
        return (
            neighbor_vectors_arg,
            neighbor_distances_arg,
            empty_sentinel(2, wp_dtype, device),
            empty_sentinel(2, wp_dtype, device),
            empty_sentinel(2, vec_dtype, device),
        )
    return (
        neighbor_vectors_arg,
        neighbor_distances_arg,
        pair_params,
        pair_energies,
        pair_forces,
    )


def _validate_coo_output(array: Any, name: str, max_pairs: int) -> None:
    """Validate a flat COO pair output buffer."""
    if _array_ndim(array) != 1:
        raise ValueError(f"{name} must have ndim == 1 after vector conversion")
    if int(array.shape[0]) < int(max_pairs):
        raise ValueError(f"{name} must have length at least max_pairs")


def _prepare_coo_pair_output_args(
    wp_dtype: type,
    device,
    max_pairs: int,
    *,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: wp.Function | None,
    pair_params: wp.array | None,
    neighbor_vectors: wp.array | None = None,
    neighbor_distances: wp.array | None = None,
    pair_energies: wp.array | None = None,
    pair_forces: wp.array | None = None,
    neighbor_vectors_name: str = "neighbor_vectors",
    neighbor_distances_name: str = "neighbor_distances",
    pair_energies_name: str = "pair_energies",
    pair_forces_name: str = "pair_forces",
) -> tuple[wp.array, wp.array, wp.array, wp.array, wp.array]:
    """Validate flat COO pair-output arguments and return kernel arrays."""
    _validate_pair_output_kwargs(
        wp_dtype=wp_dtype,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
        neighbor_vectors_name=neighbor_vectors_name,
        neighbor_distances_name=neighbor_distances_name,
        pair_energies_name=pair_energies_name,
        pair_forces_name=pair_forces_name,
    )

    vec_dtype = DTYPE_INFO_ALL[wp_dtype][0]
    if return_vectors:
        _validate_coo_output(neighbor_vectors, neighbor_vectors_name, max_pairs)
        neighbor_vectors_arg = neighbor_vectors
    else:
        neighbor_vectors_arg = empty_sentinel(1, vec_dtype, device)

    if return_distances:
        _validate_coo_output(neighbor_distances, neighbor_distances_name, max_pairs)
        neighbor_distances_arg = neighbor_distances
    else:
        neighbor_distances_arg = empty_sentinel(1, wp_dtype, device)

    if pair_fn is None:
        return (
            neighbor_vectors_arg,
            neighbor_distances_arg,
            empty_sentinel(2, wp_dtype, device),
            empty_sentinel(1, wp_dtype, device),
            empty_sentinel(1, vec_dtype, device),
        )

    _validate_coo_output(pair_energies, pair_energies_name, max_pairs)
    _validate_coo_output(pair_forces, pair_forces_name, max_pairs)
    return (
        neighbor_vectors_arg,
        neighbor_distances_arg,
        pair_params,
        pair_energies,
        pair_forces,
    )

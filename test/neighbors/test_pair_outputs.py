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

"""Tests for shared pair-output argument handling."""

import pytest
import warp as wp

from nvalchemiops.neighbors.output_args import (
    _has_partial_or_pair_outputs,
    _prepare_pair_output_args,
)

_PAIR_FN_SENTINEL = object()


def test_empty_pair_params_are_inactive_without_pair_fn():
    """A ``(0, 0)`` pair-params array is treated as the inactive sentinel."""
    pair_params = wp.empty((0, 0), dtype=wp.float32, device="cpu")

    assert not _has_partial_or_pair_outputs(pair_params=pair_params)
    _, _, pair_params_arg, _, _ = _prepare_pair_output_args(
        wp.float32,
        "cpu",
        return_vectors=False,
        return_distances=False,
        pair_fn=None,
        pair_params=pair_params,
    )

    assert pair_params_arg.shape == (0, 0)


def test_nonempty_pair_params_without_pair_fn_raise():
    """Non-sentinel pair params are only valid with ``pair_fn``."""
    pair_params = wp.empty((1, 0), dtype=wp.float32, device="cpu")

    assert _has_partial_or_pair_outputs(pair_params=pair_params)
    with pytest.raises(ValueError, match="pair_params is only valid"):
        _prepare_pair_output_args(
            wp.float32,
            "cpu",
            return_vectors=False,
            return_distances=False,
            pair_fn=None,
            pair_params=pair_params,
        )


@pytest.mark.parametrize("shape", [(0,), (1, 1, 1)])
def test_pair_params_must_be_two_dimensional(shape):
    """Pair params validate rank but not parameter-width positivity."""
    pair_params = wp.empty(shape, dtype=wp.float32, device="cpu")

    with pytest.raises(ValueError, match="pair_params must have ndim == 2"):
        _prepare_pair_output_args(
            wp.float32,
            "cpu",
            return_vectors=False,
            return_distances=False,
            pair_fn=_PAIR_FN_SENTINEL,
            pair_params=pair_params,
        )


@pytest.mark.parametrize("shape", [(0, 0), (1, 0)])
def test_pair_fn_accepts_zero_width_pair_params(shape):
    """A pair function may use a zero-width parameter table."""
    pair_params = wp.empty(shape, dtype=wp.float32, device="cpu")
    pair_energies = wp.empty((1, 4), dtype=wp.float32, device="cpu")
    pair_forces = wp.empty((1, 4), dtype=wp.vec3f, device="cpu")

    _, _, pair_params_arg, pair_energies_arg, pair_forces_arg = (
        _prepare_pair_output_args(
            wp.float32,
            "cpu",
            return_vectors=False,
            return_distances=False,
            pair_fn=_PAIR_FN_SENTINEL,
            pair_params=pair_params,
            pair_energies=pair_energies,
            pair_forces=pair_forces,
        )
    )

    assert pair_params_arg is pair_params
    assert pair_energies_arg is pair_energies
    assert pair_forces_arg is pair_forces


def test_pair_fn_rejects_wrong_pair_params_dtype():
    """Pair params must use the same scalar dtype as the kernel specialization."""
    pair_params = wp.empty((1, 0), dtype=wp.float64, device="cpu")

    with pytest.raises(ValueError, match="same scalar dtype"):
        _prepare_pair_output_args(
            wp.float32,
            "cpu",
            return_vectors=False,
            return_distances=False,
            pair_fn=_PAIR_FN_SENTINEL,
            pair_params=pair_params,
            pair_energies=wp.empty((1, 4), dtype=wp.float32, device="cpu"),
            pair_forces=wp.empty((1, 4), dtype=wp.vec3f, device="cpu"),
        )


def test_pair_fn_requires_energy_and_force_buffers():
    """Pair functions require both per-slot output buffers."""
    pair_params = wp.empty((1, 0), dtype=wp.float32, device="cpu")

    with pytest.raises(ValueError, match="pair_energies is required"):
        _prepare_pair_output_args(
            wp.float32,
            "cpu",
            return_vectors=False,
            return_distances=False,
            pair_fn=_PAIR_FN_SENTINEL,
            pair_params=pair_params,
            pair_forces=wp.empty((1, 4), dtype=wp.vec3f, device="cpu"),
        )


def test_vector_and_distance_flags_require_buffers():
    """Vector and distance feature flags require caller-owned output buffers."""
    with pytest.raises(ValueError, match="neighbor_vectors is required"):
        _prepare_pair_output_args(
            wp.float32,
            "cpu",
            return_vectors=True,
            return_distances=False,
            pair_fn=None,
            pair_params=None,
        )

    with pytest.raises(ValueError, match="neighbor_distances is required"):
        _prepare_pair_output_args(
            wp.float32,
            "cpu",
            return_vectors=False,
            return_distances=True,
            pair_fn=None,
            pair_params=None,
        )


def test_orphan_output_buffers_raise():
    """Output buffers require their matching feature flag or callback."""
    with pytest.raises(ValueError, match="neighbor_vectors is only valid"):
        _prepare_pair_output_args(
            wp.float32,
            "cpu",
            return_vectors=False,
            return_distances=False,
            pair_fn=None,
            pair_params=None,
            neighbor_vectors=wp.empty((1, 1), dtype=wp.vec3f, device="cpu"),
        )

    with pytest.raises(ValueError, match="neighbor_distances is only valid"):
        _prepare_pair_output_args(
            wp.float32,
            "cpu",
            return_vectors=False,
            return_distances=False,
            pair_fn=None,
            pair_params=None,
            neighbor_distances=wp.empty((1, 1), dtype=wp.float32, device="cpu"),
        )

    with pytest.raises(ValueError, match="pair_energies is only valid"):
        _prepare_pair_output_args(
            wp.float32,
            "cpu",
            return_vectors=False,
            return_distances=False,
            pair_fn=None,
            pair_params=None,
            pair_energies=wp.empty((1, 1), dtype=wp.float32, device="cpu"),
        )

    with pytest.raises(ValueError, match="pair_forces is only valid"):
        _prepare_pair_output_args(
            wp.float32,
            "cpu",
            return_vectors=False,
            return_distances=False,
            pair_fn=None,
            pair_params=None,
            pair_forces=wp.empty((1, 1), dtype=wp.vec3f, device="cpu"),
        )

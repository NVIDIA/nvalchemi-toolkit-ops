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

"""JAX tests for guarded frontend neighbor-list strategy estimation."""

import math

import jax.numpy as jnp
import pytest

from nvalchemiops.jax.neighbors import (
    estimate_neighbor_list_costs as report_jax,
)
from nvalchemiops.jax.neighbors import (
    suggest_neighbor_list_method as suggest_jax,
)
from nvalchemiops.neighbors.base_dispatch import neighbor_list_strategy_run_args

_ENV_KNOBS = (
    "NVALCHEMI_NEIGHLIST_CELL_SHELL",
    "NVALCHEMI_NEIGHLIST_CELL_SETUP",
)


def _cell(volume: float, num_systems: int) -> jnp.ndarray:
    length = float(volume) ** (1.0 / 3.0)
    cell = jnp.eye(3, dtype=jnp.float32) * length
    return jnp.broadcast_to(cell, (int(num_systems), 3, 3))


def _report(counts, volumes, cutoff, **kwargs):
    batch_ptr = jnp.asarray([0, *counts], dtype=jnp.int32).cumsum(axis=0)
    cell = _cell(volumes[0] if volumes else 1.0, max(len(counts), 0))
    if volumes and len(volumes) > 1:
        cell = jnp.stack([_cell(v, 1)[0] for v in volumes])
    pbc = jnp.zeros((max(len(counts), 0), 3), dtype=bool)
    return report_jax(batch_ptr, cell, pbc, cutoff, **kwargs)


def _names(report) -> list[str]:
    return [name for name, _ in report]


def _base_method(report) -> str:
    """Base method (``naive`` / ``cell_list`` / ``cluster_tile``) of the top pick."""
    return neighbor_list_strategy_run_args(report[0][0])[0]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Run each test against the default constants regardless of the host env."""
    for name in _ENV_KNOBS:
        monkeypatch.delenv(name, raising=False)


class TestReportNeighborListCosts:
    """Exercise JAX guarded naive/cell-list auto-dispatch via report/suggest."""

    def test_one_entry_batch_ptr_is_rejected(self):
        """One-entry batch_ptr is invalid selector metadata."""
        with pytest.raises(ValueError, match="batch_ptr.*length at least 2"):
            _report([], [], cutoff=5.0)

    def test_empty_one_system_batch_reports_costs(self):
        """An empty single-system batch uses batch_ptr=[0, 0]."""
        rep = _report([0], [1.0], cutoff=5.0)
        costs = [cost for _, cost in rep]
        assert rep
        assert costs == sorted(costs)
        assert all(math.isfinite(cost) for cost in costs)

    def test_small_system_picks_naive(self):
        """Tiny N keeps the direct naive path."""
        assert _base_method(_report([20], [8000.0], cutoff=5.0)) == "naive"

    def test_large_sparse_picks_cell_list(self):
        """Huge sparse systems select cell_list."""
        assert _base_method(_report([200_000], [1.25e8], cutoff=5.0)) == "cell_list"

    def test_many_tiny_systems_use_best_available_suboption(self):
        """Many tiny systems pick a naive or cell-list strategy."""
        n_sys = 128
        top = _report([18] * n_sys, [400.0] * n_sys, cutoff=15.0)[0][0]
        base = top[len("batch_") :] if top.startswith("batch_") else top
        assert base in {"naive_tile", "naive_scalar", "cell_list_pair_centric"}

    def test_large_high_cutoff_dense_picks_cell_list_not_naive(self):
        """Large dense high-cutoff systems must not route to O(N^2) naive."""
        assert _base_method(_report([1_000_000], [1e7], cutoff=15.0)) == "cell_list"

    def test_naive_viability_bound_excludes_naive_for_huge_systems(self):
        """Beyond the candidate-pair bound, naive is dropped from the report."""
        top = _report([1_000_000], [1e7], cutoff=15.0)
        assert all(
            not name.endswith(("naive_tile", "naive_scalar")) for name in _names(top)
        )

    def test_report_is_sorted_and_finite(self):
        """Report is sorted cheapest-first and contains finite costs."""
        rep = _report([20], [8000.0], cutoff=5.0)
        costs = [cost for _, cost in rep]
        assert costs == sorted(costs)
        assert all(c < float("inf") for c in costs)

    def test_shell_env_override_shifts_naive_cell_boundary(self, monkeypatch):
        """Increasing shell cost can flip a sparse system back to naive."""
        args = ([5000], [5.0e6])
        assert _base_method(_report(*args, cutoff=5.0)) == "cell_list"
        monkeypatch.setenv("NVALCHEMI_NEIGHLIST_CELL_SHELL", "100000.0")
        assert _base_method(_report(*args, cutoff=5.0)) == "naive"

    def test_setup_env_override_shifts_small_system_boundary(self, monkeypatch):
        """Lowering setup makes cell_list win sooner for modest sparse systems."""
        args = ([50], [1.0e6])
        assert _base_method(_report(*args, cutoff=5.0)) == "naive"
        monkeypatch.setenv("NVALCHEMI_NEIGHLIST_CELL_SETUP", "1.0")
        assert _base_method(_report(*args, cutoff=5.0)) == "cell_list"

    def test_suggest_matches_report_top(self):
        """suggest_neighbor_list_method returns report's cheapest name."""
        batch_ptr = jnp.asarray([0, 20], dtype=jnp.int32)
        cell = jnp.eye(3, dtype=jnp.float32)[None] * 20.0
        pbc = jnp.zeros((1, 3), dtype=bool)
        assert (
            suggest_jax(batch_ptr, cell, pbc, 5.0)
            == _names(_report([20], [8000.0], 5.0))[0]
        )

    def test_target_indices_optional_output_requires_count(self):
        """Optional target feasibility checks require a concrete target count."""
        batch_ptr = jnp.asarray([0, 10], dtype=jnp.int32)
        cell = jnp.eye(3, dtype=jnp.float32)[None] * 20.0
        pbc = jnp.zeros((1, 3), dtype=bool)

        with pytest.raises(ValueError, match="target_count is required"):
            report_jax(
                batch_ptr,
                cell,
                pbc,
                2.0,
                optional_outputs=["target_indices"],
            )

    def test_cpu_fallback_returns_sorted_finite_costs(self):
        """JAX report on host arrays returns a sorted finite cost list."""
        rep = _report([20], [8000.0], cutoff=5.0)
        assert rep, "expected at least one feasible strategy"
        costs = [cost for _, cost in rep]
        assert costs == sorted(costs)
        assert all(math.isfinite(c) for c in costs)


def test_report_validates_shape():
    """JAX report rejects mismatched metadata shapes."""
    with pytest.raises(ValueError, match="one matrix per system"):
        report_jax(
            jnp.asarray([0, 1, 2], dtype=jnp.int32),
            jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (3, 3, 3)),
            jnp.zeros((2, 3), dtype=bool),
            5.0,
        )


def test_extreme_cell_to_cutoff_ratio_stays_finite():
    """Extreme cell/cutoff ratios keep finite, non-negative strategy costs."""
    rep = _report([512], [1.0e21], 1.0e-4)
    assert rep, "expected at least one feasible strategy"
    assert all(math.isfinite(c) and c >= 0.0 for _, c in rep)
    assert _base_method(rep) in ("naive", "cell_list", "cluster_tile")

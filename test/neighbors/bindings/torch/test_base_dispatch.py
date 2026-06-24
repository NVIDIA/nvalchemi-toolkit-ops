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

"""Torch tests for guarded frontend neighbor-list strategy estimation."""

import math

import pytest
import torch

from nvalchemiops.neighbors.base_dispatch import neighbor_list_strategy_run_args
from nvalchemiops.torch.neighbors import (
    estimate_neighbor_list_costs as report_torch,
)
from nvalchemiops.torch.neighbors import (
    suggest_neighbor_list_method as suggest_torch,
)
from nvalchemiops.torch.neighbors.cell_list import estimate_cell_list_sizes

_ENV_KNOBS = (
    "NVALCHEMI_NEIGHLIST_CELL_SHELL",
    "NVALCHEMI_NEIGHLIST_CELL_SETUP",
)


def _cell(volume: float, num_systems: int) -> torch.Tensor:
    length = float(volume) ** (1.0 / 3.0)
    cell = torch.eye(3, dtype=torch.float32).mul(length).reshape(1, 3, 3)
    return cell.expand(int(num_systems), -1, -1).contiguous()


def _report(counts, volumes, cutoff, **kwargs):
    batch_ptr = torch.tensor([0, *counts], dtype=torch.int32).cumsum(dim=0)
    cell = _cell(volumes[0] if volumes else 1.0, max(len(counts), 0))
    if volumes and len(volumes) > 1:
        cell = torch.stack([_cell(v, 1)[0] for v in volumes])
    pbc = torch.zeros((max(len(counts), 0), 3), dtype=torch.bool)
    return report_torch(batch_ptr, cell, pbc, cutoff, **kwargs)


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
    """Exercise Torch guarded naive/cell-list auto-dispatch via report/suggest."""

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
        n_sys = 5365
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
        batch_ptr = torch.tensor([0, 20], dtype=torch.int32)
        cell = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3) * 20.0
        pbc = torch.zeros((1, 3), dtype=torch.bool)
        rep = report_torch(batch_ptr, cell, pbc, 5.0)
        assert suggest_torch(batch_ptr, cell, pbc, 5.0) == rep[0][0]

    def test_target_indices_reduce_estimated_source_work(self):
        """Partial-row requests scale the naive estimate by target count."""
        batch_ptr = torch.tensor([0, 10_000], dtype=torch.int32)
        cell = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3) * 100.0
        pbc = torch.zeros((1, 3), dtype=torch.bool)

        full = dict(report_torch(batch_ptr, cell, pbc, 5.0))
        partial = dict(
            report_torch(
                batch_ptr,
                cell,
                pbc,
                5.0,
                target_indices=torch.arange(100, dtype=torch.int32),
            )
        )
        # Fewer source rows -> cheaper naive estimate.
        assert partial["naive_scalar"] < full["naive_scalar"]
        # target_indices is incompatible with cluster_tile auto.
        assert "cluster_tile" not in partial

    def test_target_indices_optional_output_requires_count(self):
        """Optional target feasibility checks require a concrete target count."""
        batch_ptr = torch.tensor([0, 10], dtype=torch.int32)
        cell = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3) * 20.0
        pbc = torch.zeros((1, 3), dtype=torch.bool)

        with pytest.raises(ValueError, match="target_count is required"):
            report_torch(
                batch_ptr,
                cell,
                pbc,
                2.0,
                optional_outputs=["target_indices"],
            )

    def test_optional_outputs_use_public_neighbor_list_names(self):
        """Public optional-output names participate in feasibility checks."""
        batch_ptr = torch.tensor([0, 4096], dtype=torch.int32)
        cell = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3) * 80.0
        pbc = torch.ones((1, 3), dtype=torch.bool)

        from_names = report_torch(
            batch_ptr,
            cell,
            pbc,
            5.0,
            optional_outputs=["neighbor_vectors", "neighbor_distances"],
        )
        from_kwargs = report_torch(
            batch_ptr,
            cell,
            pbc,
            5.0,
            return_vectors=True,
            return_distances=True,
        )
        assert from_names == from_kwargs

    def test_pair_function_option_excludes_cluster_tile(self):
        """Generic pair-callable requests are excluded from cluster-tile auto."""
        batch_ptr = torch.tensor([0, 4096], dtype=torch.int32)
        cell = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3) * 80.0
        pbc = torch.ones((1, 3), dtype=torch.bool)

        rep = report_torch(batch_ptr, cell, pbc, 5.0, optional_outputs=["pair_fn"])
        assert "cluster_tile" not in _names(rep)

    def test_interleaved_batch_idx_excludes_cluster_tile(self):
        """Cluster-tile is excluded for noncontiguous batch layouts."""
        batch_ptr = torch.tensor([0, 3, 6], dtype=torch.int32)
        batch_idx = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int32)
        cell = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3)
        cell = cell.expand(2, -1, -1).contiguous() * 20.0
        pbc = torch.ones((2, 3), dtype=torch.bool)

        rep = report_torch(
            batch_ptr,
            cell,
            pbc,
            2.0,
            batch_idx=batch_idx,
            positions_dtype=torch.float32,
        )
        assert "batch_cluster_tile" not in _names(rep)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is required for this test parameter"
    )
    def test_cuda_dense_periodic_float32_selects_cluster_tile(self):
        """Dense fully periodic float32 geometry can auto-select cluster-tile."""
        batch_ptr = torch.tensor([0, 4096], dtype=torch.int32, device="cuda")
        cell = torch.eye(3, dtype=torch.float32, device="cuda").reshape(1, 3, 3) * 20.0
        pbc = torch.ones((1, 3), dtype=torch.bool, device="cuda")

        top = suggest_torch(batch_ptr, cell, pbc, 10.0, positions_dtype=torch.float32)
        assert top == "cluster_tile"


def test_report_validates_shape():
    """Torch report rejects mismatched metadata shapes."""
    with pytest.raises(ValueError, match="one matrix per system"):
        report_torch(
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.eye(3).reshape(1, 3, 3).expand(3, -1, -1).contiguous(),
            torch.zeros((2, 3), dtype=torch.bool),
            5.0,
        )


def test_estimate_cell_list_radius_uses_halved_grid():
    """Native/Torch sizing computes search radius from max_nbins-halved cells."""
    cell = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3) * 12.42
    pbc = torch.tensor([True, True, True], dtype=torch.bool)

    max_cells, neighbor_search_radius = estimate_cell_list_sizes(
        cell, pbc, 21.2, max_nbins=8
    )

    assert max_cells == 8
    assert neighbor_search_radius.cpu().tolist() == [4, 4, 4]


def test_extreme_cell_to_cutoff_ratio_stays_finite():
    """Extreme cell/cutoff ratios keep finite, non-negative strategy costs."""
    rep = _report([512], [1.0e21], 1.0e-4)
    assert rep, "expected at least one feasible strategy"
    assert all(math.isfinite(c) and c >= 0.0 for _, c in rep)
    assert _base_method(rep) in ("naive", "cell_list", "cluster_tile")

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

"""Unit tests for guarded frontend neighbor-list strategy estimation."""

import math

import jax.numpy as jnp
import pytest
import torch

from nvalchemiops.jax.neighbors import (
    estimate_neighbor_list_costs as report_jax,
)
from nvalchemiops.jax.neighbors import (
    suggest_neighbor_list_method as suggest_jax,
)
from nvalchemiops.neighbors.base_dispatch import (
    finalize_neighbor_list_method,
    neighbor_list_strategy_run_args,
)
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


def _torch_cell(volume: float, num_systems: int) -> torch.Tensor:
    length = float(volume) ** (1.0 / 3.0)
    cell = torch.eye(3, dtype=torch.float32).mul(length).reshape(1, 3, 3)
    return cell.expand(int(num_systems), -1, -1).contiguous()


def _jax_cell(volume: float, num_systems: int) -> jnp.ndarray:
    length = float(volume) ** (1.0 / 3.0)
    cell = jnp.eye(3, dtype=jnp.float32) * length
    return jnp.broadcast_to(cell, (int(num_systems), 3, 3))


def _torch_report(counts, volumes, cutoff, **kwargs):
    batch_ptr = torch.tensor([0, *counts], dtype=torch.int32).cumsum(dim=0)
    cell = _torch_cell(volumes[0] if volumes else 1.0, max(len(counts), 0))
    if volumes and len(volumes) > 1:
        cell = torch.stack([_torch_cell(v, 1)[0] for v in volumes])
    pbc = torch.zeros((max(len(counts), 0), 3), dtype=torch.bool)
    return report_torch(batch_ptr, cell, pbc, cutoff, **kwargs)


def _jax_report(counts, volumes, cutoff, **kwargs):
    batch_ptr = jnp.asarray([0, *counts], dtype=jnp.int32).cumsum(axis=0)
    cell = _jax_cell(volumes[0] if volumes else 1.0, max(len(counts), 0))
    if volumes and len(volumes) > 1:
        cell = jnp.stack([_jax_cell(v, 1)[0] for v in volumes])
    pbc = jnp.zeros((max(len(counts), 0), 3), dtype=bool)
    return report_jax(batch_ptr, cell, pbc, cutoff, **kwargs)


_REPORTERS = (
    pytest.param(_torch_report, id="torch"),
    pytest.param(_jax_report, id="jax"),
)


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
    """Exercise Torch/JAX guarded naive/cell-list auto-dispatch via report/suggest."""

    @pytest.mark.parametrize("report", _REPORTERS)
    def test_empty_system_returns_cell_list(self, report):
        """Empty per-system metadata selects cell_list."""
        assert _base_method(report([], [], cutoff=5.0)) == "cell_list"

    @pytest.mark.parametrize("report", _REPORTERS)
    def test_small_system_picks_naive(self, report):
        """Tiny N keeps the direct naive path."""
        assert _base_method(report([20], [8000.0], cutoff=5.0)) == "naive"

    @pytest.mark.parametrize("report", _REPORTERS)
    def test_large_sparse_picks_cell_list(self, report):
        """Huge sparse systems select cell_list."""
        assert _base_method(report([200_000], [1.25e8], cutoff=5.0)) == "cell_list"

    @pytest.mark.parametrize("report", _REPORTERS)
    def test_many_tiny_systems_use_best_available_suboption(self, report):
        """Many tiny systems pick a naive or cell-list strategy."""
        n_sys = 5365
        top = report([18] * n_sys, [400.0] * n_sys, cutoff=15.0)[0][0]
        base = top[len("batch_") :] if top.startswith("batch_") else top
        assert base in {"naive_tile", "naive_scalar", "cell_list_pair_centric"}

    @pytest.mark.parametrize("report", _REPORTERS)
    def test_large_high_cutoff_dense_picks_cell_list_not_naive(self, report):
        """Large dense high-cutoff systems must not route to O(N^2) naive."""
        assert _base_method(report([1_000_000], [1e7], cutoff=15.0)) == "cell_list"

    @pytest.mark.parametrize("report", _REPORTERS)
    def test_naive_viability_bound_excludes_naive_for_huge_systems(self, report):
        """Beyond the candidate-pair bound, naive is dropped from the report."""
        top = report([1_000_000], [1e7], cutoff=15.0)
        assert all(not name.endswith(("naive_tile", "naive_scalar")) for name in _names(top))

    @pytest.mark.parametrize("report", _REPORTERS)
    def test_report_is_sorted_and_suggest_matches_top(self, report):
        """Report is sorted cheapest-first; suggest returns the top name."""
        rep = report([20], [8000.0], cutoff=5.0)
        costs = [cost for _, cost in rep]
        assert costs == sorted(costs)
        assert all(c < float("inf") for c in costs)

    @pytest.mark.parametrize("report", _REPORTERS)
    def test_shell_env_override_shifts_naive_cell_boundary(self, report, monkeypatch):
        """Increasing shell cost can flip a sparse system back to naive."""
        args = ([5000], [5.0e6])
        assert _base_method(report(*args, cutoff=5.0)) == "cell_list"
        monkeypatch.setenv("NVALCHEMI_NEIGHLIST_CELL_SHELL", "100000.0")
        assert _base_method(report(*args, cutoff=5.0)) == "naive"

    @pytest.mark.parametrize("report", _REPORTERS)
    def test_setup_env_override_shifts_small_system_boundary(self, report, monkeypatch):
        """Lowering setup makes cell_list win sooner for modest sparse systems."""
        args = ([50], [1.0e6])
        assert _base_method(report(*args, cutoff=5.0)) == "naive"
        monkeypatch.setenv("NVALCHEMI_NEIGHLIST_CELL_SETUP", "1.0")
        assert _base_method(report(*args, cutoff=5.0)) == "cell_list"

    def test_suggest_matches_report_top(self):
        """suggest_neighbor_list_method returns report's cheapest name."""
        batch_ptr = torch.tensor([0, 20], dtype=torch.int32)
        cell = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3) * 20.0
        pbc = torch.zeros((1, 3), dtype=torch.bool)
        rep = report_torch(batch_ptr, cell, pbc, 5.0)
        assert suggest_torch(batch_ptr, cell, pbc, 5.0) == rep[0][0]
        assert suggest_jax(
            jnp.asarray([0, 20], dtype=jnp.int32),
            jnp.eye(3, dtype=jnp.float32)[None] * 20.0,
            jnp.zeros((1, 3), dtype=bool),
            5.0,
        ) == _names(_jax_report([20], [8000.0], 5.0))[0]

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

    def test_jax_target_indices_optional_output_requires_count(self):
        """JAX parity: target feasibility checks require a concrete target count.

        Runs on CPU because the validation fires before the GPU/CPU branch.
        """
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

    def test_jax_cpu_fallback_returns_sorted_finite_costs(self):
        """JAX report on host arrays exercises ``_jax_selector_cpu_fallback``.

        ``_jax_report`` builds CPU-backed jnp arrays, so on a CPU host this
        hits the CPU-fallback path; on a GPU host it takes the kernel path.
        Either way the report must be a non-empty, ascending, finite list.
        """
        rep = _jax_report([20], [8000.0], cutoff=5.0)
        assert rep, "expected at least one feasible strategy"
        costs = [cost for _, cost in rep]
        assert costs == sorted(costs)
        assert all(math.isfinite(c) for c in costs)

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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_dense_periodic_float32_selects_cluster_tile(self):
        """Dense fully periodic float32 geometry can auto-select cluster-tile."""
        batch_ptr = torch.tensor([0, 4096], dtype=torch.int32, device="cuda")
        cell = torch.eye(3, dtype=torch.float32, device="cuda").reshape(1, 3, 3) * 20.0
        pbc = torch.ones((1, 3), dtype=torch.bool, device="cuda")

        top = suggest_torch(batch_ptr, cell, pbc, 10.0, positions_dtype=torch.float32)
        assert top == "cluster_tile"


class TestFinalizeNeighborListMethod:
    """Exercise host-side reduction of costs/flags to sorted feasible strategies."""

    def test_selects_cheapest_naive_suboption(self):
        """Naive tile wins when it is the cheapest feasible strategy."""
        report = finalize_neighbor_list_method(
            costs=[100.0, 10.0, 1_000.0, 1_000.0, 1_000.0],
            flags=[0] * 9,
        )
        assert report[0][0] == "naive_tile"

    def test_selects_cheapest_cell_list_suboption(self):
        """Pair-centric cell-list wins when it is the cheapest feasible strategy."""
        report = finalize_neighbor_list_method(
            costs=[1_000.0, 1_000.0, 100.0, 10.0, 1_000.0],
            flags=[0] * 9,
        )
        assert report[0][0] == "cell_list_pair_centric"

    def test_selects_cluster_tile_when_feasible_and_cheapest(self):
        """Cluster-tile is selected when its feasibility flags are clear."""
        report = finalize_neighbor_list_method(
            costs=[1_000.0, 1_000.0, 1_000.0, 1_000.0, 10.0],
            flags=[0] * 9,
        )
        assert report[0][0] == "cluster_tile"

    def test_suboption_flags_exclude_unsafe_strategies(self):
        """Unsafe sub-options are dropped before ranking the survivors."""
        flags = [0] * 9
        flags[7] = 1  # pair_centric_unsafe
        flags[8] = 1  # naive_tile_unsafe

        report = finalize_neighbor_list_method(
            costs=[100.0, 10.0, 120.0, 1.0, 1_000.0],
            flags=flags,
        )
        names = _names(report)
        assert "naive_tile" not in names
        assert "cell_list_pair_centric" not in names
        assert report[0][0] == "naive_scalar"

    def test_invalid_input_flag_raises(self):
        """The invalid-input flag raises rather than returning a strategy."""
        flags = [0] * 9
        flags[0] = 1
        with pytest.raises(ValueError, match="invalid"):
            finalize_neighbor_list_method(costs=[1.0] * 5, flags=flags)


def test_torch_report_validates_shape():
    """Torch report rejects mismatched metadata shapes."""
    with pytest.raises(ValueError, match="one matrix per system"):
        report_torch(
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.eye(3).reshape(1, 3, 3).expand(3, -1, -1).contiguous(),
            torch.zeros((2, 3), dtype=torch.bool),
            5.0,
        )


def test_jax_report_validates_shape():
    """JAX report rejects mismatched metadata shapes."""
    with pytest.raises(ValueError, match="one matrix per system"):
        report_jax(
            jnp.asarray([0, 1, 2], dtype=jnp.int32),
            jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (3, 3, 3)),
            jnp.zeros((2, 3), dtype=bool),
            5.0,
        )


def test_torch_estimate_cell_list_radius_uses_halved_grid():
    """Native/Torch sizing computes search radius from max_nbins-halved cells."""
    cell = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3) * 12.42
    pbc = torch.tensor([True, True, True], dtype=torch.bool)

    max_cells, neighbor_search_radius = estimate_cell_list_sizes(
        cell, pbc, 21.2, max_nbins=8
    )

    assert max_cells == 8
    assert neighbor_search_radius.cpu().tolist() == [4, 4, 4]


@pytest.mark.parametrize("report", _REPORTERS)
def test_extreme_cell_to_cutoff_ratio_stays_finite(report):
    """An enormous cell with a tiny cutoff must not overflow the int32 cell-count
    product into a bogus (small/negative) value: the per-axis cell count is
    clamped and the product accumulated in int64, so the report stays valid with
    finite, non-negative costs instead of a wrapped grid."""
    rep = report([512], [1.0e21], 1.0e-4)
    assert rep, "expected at least one feasible strategy"
    assert all(math.isfinite(c) and c >= 0.0 for _, c in rep)
    assert _base_method(rep) in ("naive", "cell_list", "cluster_tile")


def test_is_pair_centric_parallelism_sufficient_boundary():
    """Pair-centric parallelism sufficiency tracks the block-count boundary.

    The helper compares pair-centric logical blocks
    ``total_cells * (n_outer + 1)`` against atom-centric blocks
    ``ceil(total_atoms / block_dim)``.
    """
    from nvalchemiops.neighbors.cell_list import (
        is_pair_centric_parallelism_sufficient,
    )

    # Large grid * stencil (64 * 27 = 1728 blocks) vs 2 atom blocks -> sufficient.
    assert is_pair_centric_parallelism_sufficient(
        total_atoms=128, total_cells=64, n_outer=26
    )
    # Tiny grid (1 block) vs ceil(1e6 / 64) = 15625 atom blocks -> insufficient.
    assert not is_pair_centric_parallelism_sufficient(
        total_atoms=1_000_000, total_cells=1, n_outer=0
    )

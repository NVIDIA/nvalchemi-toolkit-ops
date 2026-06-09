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

"""Tests for electrostatics benchmark helper policy."""

from __future__ import annotations

import warnings

import benchmarks.interactions.electrostatics.benchmark_electrostatics as bench


class _Timer:
    """Minimal timer that executes the benchmark callable once."""

    def time_function(self, fn):
        """Run ``fn`` and return a successful timing payload."""
        fn()
        return {"success": True, "median": 1.0}


def test_derivative_contract_defaults_to_energy_autograd():
    """Config and CLI resolution defaults to the energy-autograd contract."""
    assert bench.resolve_derivative_contract(None, None) == "energy_autograd"
    assert (
        bench.resolve_derivative_contract("legacy_direct", "energy_autograd")
        == "legacy_direct"
    )


def test_energy_autograd_workloads_include_double_backward_for_torch():
    """Torch energy-autograd rows include the committed double-backward workload."""
    assert bench.benchmark_workloads(
        method="pme",
        backend="torch",
        derivative_contract="energy_autograd",
        compute_forces=True,
        compute_virial=True,
    ) == ["forward", "backward", "double_backward"]


def test_legacy_direct_workload_is_opt_in():
    """Legacy direct output emits one explicitly labeled workload row."""
    assert bench.benchmark_workloads(
        method="ewald",
        backend="jax",
        derivative_contract="legacy_direct",
        compute_forces=True,
        compute_virial=True,
    ) == ["legacy_direct"]


def test_benchmark_result_row_records_contract_and_workload():
    """CSV rows include the derivative contract and workload labels."""
    row = bench.benchmark_result_row(
        system_data={"total_atoms": 2},
        method="ewald",
        backend="torch",
        component="full",
        compute_forces=True,
        compute_virial=True,
        derivative_contract="energy_autograd",
        workload="double_backward",
        neighbor_format="n/a",
        success=True,
        median_time_ms=1.0,
    )
    assert "derivative_contract" in bench.BENCHMARK_CSV_FIELDNAMES
    assert "workload" in bench.BENCHMARK_CSV_FIELDNAMES
    assert row["derivative_contract"] == "energy_autograd"
    assert row["workload"] == "double_backward"


def test_legacy_direct_benchmark_suppresses_deprecation_warning(monkeypatch):
    """Explicit legacy benchmark rows suppress their own deprecation warning."""

    def _legacy_runner(*args, **kwargs):
        warnings.warn("legacy direct output", DeprecationWarning, stacklevel=2)
        return object()

    monkeypatch.setattr(bench, "run_nvalchemiops_ewald", _legacy_runner)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        row = bench.run_benchmark(
            "ewald",
            "torch",
            {"total_atoms": 2},
            "full",
            True,
            True,
            _Timer(),
            derivative_contract="legacy_direct",
            workload="legacy_direct",
        )

    assert row["success"] is True
    assert row["derivative_contract"] == "legacy_direct"
    assert row["workload"] == "legacy_direct"
    assert caught == []

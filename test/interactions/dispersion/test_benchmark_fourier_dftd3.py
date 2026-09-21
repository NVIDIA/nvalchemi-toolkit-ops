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

"""Failure handling in the FourierD3 benchmark sweep.

A sweep runs unattended over a range of system sizes, and the large end is where it is most
likely to fail. A handler that raises instead of recording loses every row that would have
come after it, so the failure path needs coverage of its own.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pytest

_BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"
if str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))

benchmark_module = pytest.importorskip(
    "interactions.dispersion.benchmark_fourier_dftd3",
    reason="benchmark dependencies are not installed",
)

# The same shape as the shipped config: systems x scaling modes, trimmed to one cell.
CONFIG = {
    "parameters": {
        "real_space_cutoffs": [6.0],
        "timing_runs": 1,
        "warmup_runs": 0,
        "max_total_atoms": 131072,
    },
    "runtime": {"backend": "torch"},
    "systems": {
        "cscl": {"enabled": True, "atom_counts": [250]},
        "nh3": {"enabled": False},
    },
    "scaling": {
        "system_size": {"enabled": True, "batch_size": 1},
        "constant_workload": {"enabled": False},
        "batch_scaling": {"enabled": False},
    },
    "output": {"base_dir": "benchmarks/results"},
}


def _raise(exc):
    def fail(*_args, **_kwargs):
        raise exc

    return fail


@pytest.mark.gpu
class TestSweepFailureHandling:
    """Both handlers must record a row and let the sweep continue."""

    @pytest.mark.parametrize(
        ("message", "expected_type"),
        [
            ("synthetic failure", "RuntimeError"),
            # Framework OOM arrives as a RuntimeError; the sweep records it under a stable
            # name so a memory ceiling is distinguishable from a genuine error in the CSV.
            ("CUDA error: out of memory", "OutOfMemoryError"),
        ],
    )
    def test_a_failure_is_recorded_rather_than_raised(
        self, monkeypatch, message, expected_type
    ):
        """Every method still produces a row, and the sweep returns normally."""
        error = RuntimeError(message)
        monkeypatch.setattr(
            benchmark_module, "benchmark_fourier_d3", _raise(error), raising=True
        )
        monkeypatch.setattr(
            benchmark_module, "benchmark_real_space_d3", _raise(error), raising=True
        )

        results = benchmark_module.run_from_config(CONFIG, None)

        assert [row["method"] for row in results] == [
            "fourier_dftd3",
            "fourier_dftd3_setup",
            "dftd3_cutoff_6",
        ]
        for row in results:
            assert row["success"] is False
            assert row["error_type"] == expected_type


class TestConfiguration:
    """The YAML config has to be read, not bypassed.

    Building the config inline in ``main`` left ``benchmark_fourier_config.yaml`` unused,
    including its ``output.base_dir``, so results went nowhere unless ``--output-dir`` was
    passed and the configured cutoffs were silently ignored.
    """

    def test_the_shipped_config_parses_and_carries_what_the_sweep_needs(self):
        """A missing key here would fall back to a default and hide the config."""
        config = benchmark_module.load_yaml_config(benchmark_module.DEFAULT_CONFIG)
        parameters = config["parameters"]
        for key in ("real_space_cutoffs", "timing_runs", "warmup_runs"):
            assert key in parameters, key
        assert config["output"]["base_dir"]
        # Same schema as the DFT-D3 config: systems and scaling modes, not a flat sweep.
        assert set(config["systems"]) == {"cscl", "nh3"}
        assert set(config["scaling"]) == {
            "system_size",
            "constant_workload",
            "batch_scaling",
        }
        # Density comes from the lattice now, so configuring it would be misleading.
        assert "density" not in parameters

    def test_cli_flags_override_the_config_only_when_given(self):
        """An unset flag must leave the configured value alone."""
        args = argparse.Namespace(atom_counts=[9], timing_runs=None, warmup_runs=None)
        config = benchmark_module.merge_cli_overrides(
            {"parameters": {"atom_counts": [7], "timing_runs": 4}}, args
        )
        assert config["parameters"]["atom_counts"] == [9]
        assert config["parameters"]["timing_runs"] == 4

    @pytest.mark.gpu
    def test_results_land_in_the_configured_base_dir(self, monkeypatch, tmp_path):
        """Without ``--output-dir`` the sweep uses ``output.base_dir``.

        Driven through the failure path so no kernel runs: what is under test is where the
        CSV is written, not what is in it.
        """
        error = RuntimeError("synthetic failure")
        monkeypatch.setattr(
            benchmark_module, "benchmark_fourier_d3", _raise(error), raising=True
        )
        monkeypatch.setattr(
            benchmark_module, "benchmark_real_space_d3", _raise(error), raising=True
        )
        config = dict(CONFIG)
        config["output"] = {"base_dir": str(tmp_path)}

        benchmark_module.run_from_config(config, None)

        assert list(tmp_path.glob("fd3_*/fd3-cscl-system-size-scaling.csv"))


class TestBackendGuard:
    """Only torch is measurable here, so anything else must fail rather than mislabel.

    ``backend`` reached the row labels only -- the measurement functions import torch
    unconditionally -- so an unsupported value would have produced a CSV claiming torch
    numbers were something else.
    """

    @pytest.mark.parametrize("backend", ["jax", "warp", "numpy"])
    def test_an_unsupported_backend_is_refused(self, backend):
        """Both the sweep and the plan expansion have to refuse it."""
        with pytest.raises(ValueError, match="only the torch backend"):
            benchmark_module.run_from_config(CONFIG, None, backend)
        with pytest.raises(ValueError, match="only the torch backend"):
            benchmark_module.dry_run_from_config(CONFIG, backend)

    def test_an_unsupported_backend_in_the_config_is_refused(self):
        """``runtime.backend`` is the other way in, and is resolved the same way."""
        config = dict(CONFIG)
        config["runtime"] = {"backend": "jax"}
        with pytest.raises(ValueError, match="only the torch backend"):
            benchmark_module.dry_run_from_config(config)

    def test_torch_is_the_default_and_is_labelled(self):
        """An unset backend resolves to torch rather than to None."""
        assert all(
            row["backend"] == "torch"
            for row in benchmark_module.dry_run_from_config(CONFIG)
        )


@pytest.mark.gpu
class TestSharedBenchmarkContract:
    """The runner follows the same matrix and result contract as the DFT-D3 benchmark."""

    def test_the_matrix_covers_systems_and_scaling_modes(self):
        """Every enabled system crossed with every enabled scaling mode."""
        config = copy.deepcopy(CONFIG)
        config["systems"]["nh3"]["enabled"] = True
        config["systems"]["nh3"]["pdb_dir"] = "../../nh3"
        config["systems"]["nh3"]["atom_counts"] = [256]
        for mode in ("constant_workload", "batch_scaling"):
            config["scaling"][mode] = {"enabled": True, "target_atoms": 8192}
        config["scaling"]["batch_scaling"]["max_total_atoms"] = 8192

        rows = benchmark_module.dry_run_from_config(config, backend="torch")
        assert {(r["system"], r["mode"]) for r in rows} == {
            (system, mode)
            for system in ("cscl", "nh3")
            for mode in ("system_size", "constant_workload", "batch_scaling")
        }

    def test_every_method_appears_in_the_plan(self):
        """Both FourierD3 variants and one row per real-space cutoff."""
        rows = benchmark_module.dry_run_from_config(CONFIG, backend="torch")
        assert {r["method"] for r in rows} == {
            "fourier_dftd3",
            "fourier_dftd3_setup",
            "dftd3_cutoff_6",
        }

    def test_the_dry_run_reports_what_would_be_built(self):
        """A dry run's atom counts must match the rows a real sweep would emit."""
        rows = benchmark_module.dry_run_from_config(CONFIG, backend="torch")
        assert {row["atoms_per_system"] for row in rows} == {250}

    @pytest.mark.gpu
    def test_a_sweep_writes_one_standardized_csv_per_pair(self, tmp_path):
        """File naming and the canonical metric follow the shared contract."""
        rows = benchmark_module.run_from_config(
            copy.deepcopy(CONFIG), tmp_path, backend="torch"
        )
        assert rows
        written = [path.name for path in tmp_path.glob("*.csv")]
        assert written == ["fd3-cscl-system-size-scaling.csv"]
        for row in rows:
            # Evaluation time is canonical; the other two are supplementary.
            assert row["time_eval_seconds"] <= row["time_total_seconds"]
            assert "time_neighbor_seconds" in row

    @pytest.mark.gpu
    def test_batched_runs_keep_systems_apart(self, tmp_path):
        """A batch is several independent cells, so no neighbour may cross between them.

        Pinning a single-system list builder here would either link replicas or, as the
        shared builders are shaped, fail outright; both make the batch rows meaningless.
        """
        config = copy.deepcopy(CONFIG)
        config["scaling"] = {
            "system_size": {"enabled": False},
            "constant_workload": {"enabled": False},
            "batch_scaling": {"enabled": True, "max_total_atoms": 2000},
        }
        config["systems"]["cscl"]["constant_atoms_sizes"] = [250]
        rows = benchmark_module.run_from_config(config, tmp_path, backend="torch")
        batched = [r for r in rows if r["batch_size"] > 1]
        assert batched, "batch scaling produced no multi-system rows"
        assert all(r["success"] for r in batched), [
            r["error"] for r in batched if not r["success"]
        ]

    @pytest.mark.gpu
    def test_the_mesh_follows_the_cell(self, tmp_path):
        """A fixed spacing means the mesh grows with the cell, not with a size threshold."""
        small = copy.deepcopy(CONFIG)
        large = copy.deepcopy(CONFIG)
        large["systems"]["cscl"]["atom_counts"] = [2000]
        meshes = []
        for config in (small, large):
            rows = benchmark_module.run_from_config(config, tmp_path, backend="torch")
            meshes.append(
                next(r["mesh"] for r in rows if r["method"] == "fourier_dftd3")
            )
        assert meshes[1] > meshes[0], meshes

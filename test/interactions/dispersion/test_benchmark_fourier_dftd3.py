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

CONFIG = {
    "parameters": {
        "atom_counts": [64],
        "real_space_cutoffs": [6.0],
        "timing_runs": 1,
        "warmup_runs": 0,
        "device": "cuda",
    }
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
        for key in (
            "atom_counts",
            "real_space_cutoffs",
            "timing_runs",
            "warmup_runs",
        ):
            assert key in parameters, key
        assert config["output"]["base_dir"]
        # Density comes from the CsCl lattice now, so configuring it would be misleading.
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
class TestSystemSizeIsTheRealisedOne:
    """The CsCl builder rounds up to whole unit cells, so the request is not the size."""

    def test_the_mesh_follows_the_realised_count(self):
        """A request on a schedule boundary must not be timed on the mesh below it.

        20000 realises as 21296, which crosses the 20000 threshold, so selecting from the
        request gives a 64-cubed mesh where the built system calls for 128-cubed. Checked
        through the returned row rather than the source, so it holds however it is wired.
        """
        measured = benchmark_module.benchmark_fourier_d3(
            20000, "cuda", num_runs=1, warmup_runs=0
        )
        assert measured["atoms"] > 20000
        assert measured["mesh"] == benchmark_module.mesh_for(measured["atoms"])

    def test_the_dry_run_reports_what_would_be_built(self):
        """A dry run's atom counts must match the rows a real sweep would emit."""
        from suite_systems import cscl_actual_atoms

        rows = benchmark_module.dry_run_from_config(CONFIG, backend="torch")
        expected = cscl_actual_atoms(CONFIG["parameters"]["atom_counts"][0])
        assert {row["atoms_per_system"] for row in rows} == {expected}

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

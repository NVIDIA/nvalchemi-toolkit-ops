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

"""Validation of independently refreshed complete benchmark methods."""

import csv
from pathlib import Path

import pytest

from benchmarks.suite_utils import validate_result_files


def _row(method: str = "cell_list_pair_centric", run_id: str = "run-a") -> dict:
    """Build a minimal measured row with all required collection fields."""
    return {
        "backend": "torch",
        "method": method,
        "provenance_version": "2",
        "run_id": run_id,
        "gpu_context": "gpu",
        "software_context": "software",
        "input_context": "inputs",
        "execution_context": "node",
        "runtime_context": "runtime",
        "timing_runs": "10",
        "warmup_runs": "3",
        "success": "True",
        "error": "",
        "error_type": "",
    }


def _write(path: Path, rows: list[dict]) -> Path:
    """Write test rows using the ordinary CSV schema."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


class TestMethodRefreshValidation:
    """Refreshes preserve collection consistency across the complete method."""

    def test_default_remains_suite_wide(self, tmp_path):
        """Collection validation continues to reject different runs by default."""
        path = _write(tmp_path / "nl-cscl.csv", [_row(), _row("naive_scalar", "run-b")])
        with pytest.raises(ValueError, match="Mixed benchmark run_id"):
            validate_result_files([path])
        assert validate_result_files([path], per_method=True)["successes"] == 2
        with pytest.raises(ValueError, match="Run ID mismatch"):
            validate_result_files([path], per_method=True, expected_run_id="run-a")

    @pytest.mark.parametrize("module,method", [("el", "pme"), ("d3", "dftd3")])
    @pytest.mark.parametrize("field", ["run_id", "software_context", "runtime_context"])
    def test_other_modules_remain_suite_wide(self, tmp_path, module, method, field):
        """NL refresh validation preserves other modules' consistency checks."""
        first = _write(tmp_path / f"{module}-cscl.csv", [_row(method)])
        second = _write(
            tmp_path / f"{module}-nh3.csv", [dict(_row(method), **{field: "changed"})]
        )
        with pytest.raises(ValueError, match=field):
            validate_result_files([first, second], per_method=True)

    @pytest.mark.parametrize(
        "field",
        [
            "run_id",
            "gpu_context",
            "software_context",
            "input_context",
            "runtime_context",
            "provenance_version",
            "timing_runs",
            "warmup_runs",
        ],
    )
    def test_partial_refresh_rejected(self, tmp_path, field):
        """Single and batched cases share one collection across CSV files."""
        first = _write(tmp_path / "nl-cscl-system-size.csv", [_row()])
        batch = _row("batch_cell_list_pair_centric")
        batch[field] = "changed"
        second = _write(tmp_path / "nl-nh3-batch.csv", [batch])
        with pytest.raises(ValueError, match=field):
            validate_result_files([first, second], per_method=True)

    def test_complete_method_refresh(self, tmp_path):
        """Separate methods can carry their own accurate run and input records."""
        pair = _row()
        naive = _row("naive_scalar", "run-b")
        for field in (
            "gpu_context",
            "software_context",
            "input_context",
            "runtime_context",
        ):
            naive[field] = "other"
        first = _write(tmp_path / "nl-cscl.csv", [pair, naive])
        second = _write(
            tmp_path / "nl-nh3.csv",
            [
                dict(pair, method="batch_cell_list_pair_centric"),
                dict(naive, method="batch_naive_scalar"),
            ],
        )
        assert validate_result_files([first, second], per_method=True) == {
            "rows": 4,
            "successes": 4,
            "failures": 0,
        }

    def test_failure_records_remain_required(self, tmp_path):
        """A refresh retains explicit failure reasons for unsuccessful cases."""
        row = dict(_row(), success="False")
        path = _write(tmp_path / "nl-cscl.csv", [row])
        with pytest.raises(ValueError, match="Missing failure metadata"):
            validate_result_files([path], per_method=True)

    @pytest.mark.parametrize("field", ["backend", "method"])
    def test_missing_collection_identity(self, tmp_path, field):
        """Method validation requires an explicit framework and method."""
        row = _row()
        row[field] = ""
        path = _write(tmp_path / "nl-cscl.csv", [row])
        with pytest.raises(ValueError, match="Missing benchmark backend or method"):
            validate_result_files([path], per_method=True)

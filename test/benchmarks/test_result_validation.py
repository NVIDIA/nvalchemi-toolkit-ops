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

from benchmarks.suite_utils import save_results, validate_result_files


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

    @pytest.mark.parametrize("field", ["timing_batches", "timing_batch_aggregation"])
    def test_el_timing_groups_are_consistent(self, tmp_path, field):
        """EL timing groups stay identical across systems and scaling modes."""
        row = dict(
            _row("pme"), timing_batches="5", timing_batch_aggregation="arithmetic_mean"
        )
        first = _write(tmp_path / "el-cscl.csv", [row])
        changed = dict(row)
        changed[field] = "changed"
        second = _write(tmp_path / "el-nh3.csv", [changed])
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


class TestElMethodReplacement:
    """EL method reruns preserve the other measured methods and backends."""

    @pytest.mark.parametrize("backend", ["torch", "jax"])
    def test_method_rerun_preserves_other_rows(self, tmp_path, backend):
        """The EL progress-writer call replaces only its selected method."""
        path = tmp_path / "el-cscl-system-size-scaling.csv"
        initial = [
            dict(_row(method), backend=framework, time_us_per_atom="old")
            for framework in ("torch", "jax")
            for method in ("pme", "ewald")
        ]
        save_results(initial, path)
        with path.open(newline="") as stream:
            before = list(csv.DictReader(stream))
        replacement = dict(_row("pme"), backend=backend, time_us_per_atom="new")

        save_results(
            [replacement], path, replace_backend=backend, replace_methods=["pme"]
        )

        with path.open(newline="") as stream:
            after = list(csv.DictReader(stream))
        preserved = [
            row for row in before if (row["backend"], row["method"]) != (backend, "pme")
        ]
        assert after[:-1] == preserved
        assert after[-1]["backend"] == backend
        assert after[-1]["method"] == "pme"
        assert after[-1]["time_us_per_atom"] == "new"
        assert validate_result_files([path])["rows"] == 4

    def test_default_backend_replacement(self, tmp_path):
        """Existing callers replace all rows for the requested backend."""
        path = tmp_path / "el-cscl-system-size-scaling.csv"
        save_results(
            [_row("pme"), _row("ewald"), dict(_row("pme"), backend="jax")], path
        )
        save_results([_row("pme")], path, replace_backend="torch")
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert [(row["backend"], row["method"]) for row in rows] == [
            ("jax", "pme"),
            ("torch", "pme"),
        ]

    def test_empty_method_rerun_clears_matching_rows(self, tmp_path):
        """An empty method rerun removes its stale rows and retains Ewald."""
        path = tmp_path / "el-cscl-system-size-scaling.csv"
        save_results([_row("pme"), _row("ewald")], path)
        save_results([], path, replace_backend="torch", replace_methods=["pme"])
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert [row["method"] for row in rows] == ["ewald"]

    @pytest.mark.parametrize(
        "options,message",
        [
            ({"replace_methods": ["pme"]}, "requires replace_backend"),
            (
                {"replace_backend": "torch", "replace_methods": []},
                "must not be empty",
            ),
            (
                {"replace_backend": "torch", "replace_methods": ["ewald"]},
                "include every incoming row method",
            ),
            (
                {"replace_backend": "jax", "replace_methods": ["pme"]},
                "match every incoming row",
            ),
        ],
    )
    def test_invalid_replacement_arguments(self, tmp_path, options, message):
        """Invalid replacement scopes fail before writing a CSV."""
        path = tmp_path / "el-cscl-system-size-scaling.csv"
        with pytest.raises(ValueError, match=message):
            save_results([_row("pme")], path, **options)
        assert not path.exists()

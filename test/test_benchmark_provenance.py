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

"""Regression tests for benchmark timing and run-provenance safeguards."""

from __future__ import annotations

import csv
import json
import threading
from types import SimpleNamespace

import pytest

from benchmarks import suite_utils


def _result(backend: str, system: str) -> dict:
    """Build a minimal successful benchmark result."""
    return suite_utils.build_result(
        benchmark="nl",
        backend=backend,
        system=system,
        scaling_mode="system_size",
        method="cell_list",
        atoms_per_system=128,
        batch_size=1,
        total_atoms=128,
        time_seconds=1.0,
        timing_runs=10,
        warmup_runs=3,
        mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
    )


def _read_rows(path) -> list[dict[str, str]]:
    """Read all CSV rows from ``path``."""
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


class TestFrameworkTiming:
    """Protect the framework-specific timer dispatch contract."""

    def test_cuda_timed_runs_keeps_jax_on_wall_clock(self, monkeypatch):
        """JAX dispatches to its wall-clock timer, never CUDA events."""
        calls = []

        def fake_jax(fn, num_runs, warmup_runs):
            calls.append(("jax", fn, num_runs, warmup_runs))
            return 1.25

        def fake_cuda(fn, num_runs, warmup_runs):
            calls.append(("cuda", fn, num_runs, warmup_runs))
            return 2.5

        fn = object()
        monkeypatch.setattr(suite_utils, "jax_timed_batch", fake_jax)
        monkeypatch.setattr(suite_utils, "cuda_timed_batch", fake_cuda)

        assert suite_utils.cuda_timed_runs(fn, 7, 2, backend="jax") == 1.25
        assert suite_utils.cuda_timed_runs(fn, 7, 2, backend="torch") == 2.5
        assert calls == [("jax", fn, 7, 2), ("cuda", fn, 7, 2)]


class TestJaxEnvironment:
    """Protect standalone JAX precision configuration."""

    def test_x64_is_enabled_only_when_requested(self, monkeypatch):
        """NL/D3 stay on JAX defaults while electrostatics opts into x64."""
        monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
        monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

        suite_utils.configure_jax_environment(need_x64=False)
        assert "JAX_ENABLE_X64" not in suite_utils.os.environ

        suite_utils.configure_jax_environment(need_x64=True)
        assert suite_utils.os.environ["JAX_ENABLE_X64"] == "1"

    def test_runtime_context_records_actual_jax_x64_state(self, monkeypatch):
        """Runtime provenance distinguishes the request from JAX's live state."""
        fake_jax = SimpleNamespace(
            config=SimpleNamespace(jax_disable_jit=False, jax_enable_x64=True),
        )
        monkeypatch.setitem(suite_utils.sys.modules, "jax", fake_jax)

        context = json.loads(suite_utils._runtime_context())

        assert context["JAX_DISABLE_JIT_ACTUAL"] == "False"
        assert context["JAX_ENABLE_X64_ACTUAL"] == "True"

    def test_execution_provenance_refreshes_runtime_context(self, monkeypatch):
        """Runtime changes after setup are not hidden by the stable cache."""
        contexts = iter(("runtime-a", "runtime-b"))
        monkeypatch.setattr(suite_utils, "_EXECUTION_PROVENANCE", None)
        monkeypatch.setattr(suite_utils, "_gpu_context", lambda: "gpu")
        monkeypatch.setattr(suite_utils, "_execution_context", lambda: "execution")
        monkeypatch.setattr(suite_utils, "_runtime_context", lambda: next(contexts))

        first = suite_utils._execution_provenance()
        second = suite_utils._execution_provenance()

        assert first["runtime_context"] == "runtime-a"
        assert second["runtime_context"] == "runtime-b"


class TestSourceRevision:
    """Protect content-addressed source provenance outside Git checkouts."""

    def test_source_revision_hashes_files_without_git(self, monkeypatch):
        """A source archive gets a content hash instead of an unknown identity."""
        monkeypatch.setattr(suite_utils.shutil, "which", lambda _name: None)

        revision = suite_utils._source_revision()

        assert revision["git_head"] == "unavailable"
        assert len(revision["source_sha256"]) == 64
        assert revision["source_sha256"] != "unknown"


class TestRunProvenance:
    """Protect CSV rows from cross-context backend merges."""

    @staticmethod
    def _provenance(**overrides: str) -> dict[str, str]:
        provenance = {
            "provenance_version": "2",
            "run_id": "run-a",
            "gpu_context": "gpu-a",
            "software_context": "software-a",
            "input_context": "input-a",
            "execution_context": "node-a/gpu-a",
            "runtime_context": "runtime-a",
        }
        provenance.update(overrides)
        return provenance

    def test_emitted_rows_include_provenance(self, tmp_path, monkeypatch):
        """Every newly emitted CSV row carries the merge context."""
        csv_path = tmp_path / "results.csv"
        provenance = self._provenance()
        monkeypatch.setattr(
            suite_utils,
            "_run_provenance",
            lambda output_path: provenance,
        )

        suite_utils.save_results([_result("torch", "cscl")], csv_path)

        row = _read_rows(csv_path)[0]
        for field, value in provenance.items():
            assert row[field] == value

    def test_emitted_csv_uses_lf_line_endings(self, tmp_path, monkeypatch):
        """Checked-in result snapshots do not fail Git whitespace checks."""
        csv_path = tmp_path / "results.csv"
        monkeypatch.setattr(
            suite_utils,
            "_run_provenance",
            lambda output_path: self._provenance(),
        )

        suite_utils.save_results([_result("torch", "cscl")], csv_path)

        assert b"\r\n" not in csv_path.read_bytes()

    def test_runtime_context_excludes_scheduler_device_identity(self, monkeypatch):
        """Node-local GPU selection does not make homogeneous shards incompatible."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-node-a")
        first = json.loads(suite_utils._runtime_context())
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-node-b")
        second = json.loads(suite_utils._runtime_context())

        assert first == second
        assert "CUDA_VISIBLE_DEVICES" not in first
        assert "XLA_PYTHON_CLIENT_PREALLOCATE" in first

    def test_run_id_is_shared_after_process_cache_is_cleared(
        self,
        tmp_path,
        monkeypatch,
    ):
        """The output-directory marker joins separate backend processes."""
        run_ids = {}
        monkeypatch.setattr(suite_utils, "_RUN_IDS_BY_DIRECTORY", run_ids)

        first = suite_utils._run_id_for_directory(tmp_path)
        run_ids.clear()
        second = suite_utils._run_id_for_directory(tmp_path)

        assert second == first

    def test_explicit_run_id_initializes_directory_marker(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Parallel scheduler shards can agree on one caller-supplied UUID."""
        requested = "12345678123456781234567812345678"
        monkeypatch.setenv("BENCHMARK_RUN_ID", requested)
        monkeypatch.setattr(suite_utils, "_RUN_IDS_BY_DIRECTORY", {})

        assert suite_utils._run_id_for_directory(tmp_path) == requested
        assert (tmp_path / ".benchmark-run-id").read_text().strip() == requested

    def test_explicit_run_id_rejects_existing_different_run(
        self,
        tmp_path,
        monkeypatch,
    ):
        """A shard cannot silently join an output directory from another run."""
        marker = tmp_path / ".benchmark-run-id"
        marker.write_text("12345678123456781234567812345678\n")
        monkeypatch.setenv("BENCHMARK_RUN_ID", "87654321876543218765432187654321")
        monkeypatch.setattr(suite_utils, "_RUN_IDS_BY_DIRECTORY", {})

        with pytest.raises(RuntimeError, match="does not match"):
            suite_utils._run_id_for_directory(tmp_path)

        assert marker.read_text().strip() == "12345678123456781234567812345678"

    def test_matching_torch_and_jax_rows_can_share_csv(self, tmp_path, monkeypatch):
        """Separate backend passes from one run retain the other backend."""
        csv_path = tmp_path / "results.csv"
        provenance = self._provenance()
        monkeypatch.setattr(
            suite_utils,
            "_run_provenance",
            lambda output_path: provenance,
        )

        suite_utils.save_results(
            [_result("torch", "torch")],
            csv_path,
            replace_backend="torch",
        )
        suite_utils.save_results(
            [_result("jax", "jax")],
            csv_path,
            replace_backend="jax",
        )

        assert [(row["backend"], row["system"]) for row in _read_rows(csv_path)] == [
            ("torch", "torch"),
            ("jax", "jax"),
        ]

    def test_executor_identity_can_differ_across_cluster_shards(self, tmp_path):
        """Compatible H100 shards may come from different nodes and GPU UUIDs."""
        output_path = tmp_path / "results.csv"
        preserved = {
            **self._provenance(),
            "execution_context": "node-a/gpu-a",
        }
        incoming = {
            **self._provenance(),
            "execution_context": "node-b/gpu-b",
        }

        suite_utils._validate_preserved_provenance([preserved], incoming, output_path)

    @pytest.mark.parametrize(
        ("field", "replacement"),
        [
            ("run_id", "run-b"),
            ("gpu_context", "gpu-b"),
            ("software_context", "software-b"),
            ("input_context", "input-b"),
        ],
    )
    def test_replace_backend_rejects_context_mismatch(
        self,
        tmp_path,
        monkeypatch,
        field,
        replacement,
    ):
        """Preserved rows must match the incoming run, hardware, code, and inputs."""
        csv_path = tmp_path / "results.csv"
        current = self._provenance()
        monkeypatch.setattr(
            suite_utils,
            "_run_provenance",
            lambda output_path: current,
        )
        suite_utils.save_results(
            [_result("torch", "torch")],
            csv_path,
            replace_backend="torch",
        )
        current = self._provenance(**{field: replacement})

        with pytest.raises(ValueError, match=field):
            suite_utils.save_results(
                [_result("jax", "jax")],
                csv_path,
                replace_backend="jax",
            )

        assert [(row["backend"], row["system"]) for row in _read_rows(csv_path)] == [
            ("torch", "torch")
        ]

    def test_replace_backend_rejects_legacy_preserved_rows(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Unknown legacy provenance cannot be silently labeled compatible."""
        csv_path = tmp_path / "results.csv"
        csv_path.write_text(
            "backend,system,success\ntorch,legacy,True\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            suite_utils,
            "_run_provenance",
            lambda output_path: self._provenance(),
        )

        with pytest.raises(ValueError, match="missing provenance"):
            suite_utils.save_results(
                [_result("jax", "jax")],
                csv_path,
                replace_backend="jax",
            )

        assert _read_rows(csv_path)[0]["system"] == "legacy"

    def test_append_rejects_context_mismatch(self, tmp_path, monkeypatch):
        """Append mode cannot bypass the same provenance checks as backend merge."""
        csv_path = tmp_path / "results.csv"
        current = self._provenance()
        monkeypatch.setattr(
            suite_utils,
            "_run_provenance",
            lambda output_path: current,
        )
        suite_utils.save_results([_result("torch", "first")], csv_path)
        current = self._provenance(run_id="run-b")

        with pytest.raises(ValueError, match="run_id"):
            suite_utils.save_results(
                [_result("torch", "second")],
                csv_path,
                append=True,
            )

        assert [row["system"] for row in _read_rows(csv_path)] == ["first"]

    def test_append_rejects_legacy_rows(self, tmp_path, monkeypatch):
        """Append mode refuses legacy rows whose source run cannot be established."""
        csv_path = tmp_path / "results.csv"
        csv_path.write_text(
            "backend,system,success\ntorch,legacy,True\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            suite_utils,
            "_run_provenance",
            lambda output_path: self._provenance(),
        )

        with pytest.raises(ValueError, match="missing provenance"):
            suite_utils.save_results(
                [_result("torch", "new")],
                csv_path,
                append=True,
            )

        assert _read_rows(csv_path)[0]["system"] == "legacy"

    def test_same_backend_runtime_context_must_match(self, tmp_path, monkeypatch):
        """Allocator settings cannot vary across shards of one backend."""
        csv_path = tmp_path / "results.csv"
        current = self._provenance(runtime_context="runtime-a")
        monkeypatch.setattr(
            suite_utils,
            "_run_provenance",
            lambda output_path: current,
        )
        suite_utils.save_results([_result("jax", "first")], csv_path)
        current = self._provenance(runtime_context="runtime-b")

        with pytest.raises(ValueError, match="runtime settings"):
            suite_utils.save_results(
                [_result("jax", "second")],
                csv_path,
                append=True,
            )

    def test_backend_merge_rejects_timing_protocol_mismatch(
        self, tmp_path, monkeypatch
    ):
        """Separate backend passes cannot use different run counts."""
        csv_path = tmp_path / "results.csv"
        provenance = self._provenance()
        monkeypatch.setattr(
            suite_utils,
            "_run_provenance",
            lambda output_path: provenance,
        )
        suite_utils.save_results([_result("torch", "cscl")], csv_path)
        jax_result = _result("jax", "cscl")
        jax_result["timing_runs"] = 1

        with pytest.raises(ValueError, match="measurement protocol"):
            suite_utils.save_results(
                [jax_result],
                csv_path,
                replace_backend="jax",
            )

    def test_directory_validation_rejects_component_protocol_mismatch(
        self, tmp_path, monkeypatch
    ):
        """A complete suite cannot mix EL component-profiling policies."""
        provenance = self._provenance()
        monkeypatch.setattr(
            suite_utils,
            "_run_provenance",
            lambda output_path: provenance,
        )
        paths = [tmp_path / "first.csv", tmp_path / "second.csv"]
        for path, backend, profiled in (
            (paths[0], "torch", False),
            (paths[1], "jax", True),
        ):
            result = _result(backend, "cscl")
            result.update(
                {
                    "derivative_contract": "energy_autograd",
                    "workload": "energy_forces_charge_gradients",
                    "compute_forces": True,
                    "compute_charge_gradients": True,
                    "component_profiled": profiled,
                }
            )
            suite_utils.save_results([result], path)

        with pytest.raises(ValueError, match="measurement protocol"):
            suite_utils.validate_result_files(paths)

    def test_input_artifacts_are_content_addressed(self, tmp_path, monkeypatch):
        """External artifacts record content, not scratch mount paths."""
        monkeypatch.setattr(suite_utils, "_INPUT_CONTEXT", suite_utils._INPUT_CONTEXT)
        first_dir = tmp_path / "first"
        second_dir = tmp_path / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        (first_dir / "system.pdb").write_text("same-data", encoding="utf-8")
        (second_dir / "system.pdb").write_text("same-data", encoding="utf-8")

        suite_utils.configure_input_provenance(
            {"nh3": first_dir}, metadata_values={"benchmark": "nl"}
        )
        first_context = suite_utils._INPUT_CONTEXT
        suite_utils.configure_input_provenance(
            {"nh3": second_dir}, metadata_values={"benchmark": "nl"}
        )

        assert suite_utils._INPUT_CONTEXT == first_context

    def test_directory_validation_rejects_mixed_runs(self, tmp_path):
        """Cross-file validation catches manually copied rows from another run."""
        first = tmp_path / "first.csv"
        second = tmp_path / "second.csv"
        fields = [
            "success",
            "error",
            "error_type",
            "provenance_version",
            "run_id",
            "gpu_context",
            "software_context",
            "input_context",
            "execution_context",
            "runtime_context",
        ]
        for path, run_id in ((first, "run-a"), (second, "run-b")):
            with path.open("w", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "success": True,
                        "error": "",
                        "error_type": "",
                        "provenance_version": "2",
                        "run_id": run_id,
                        "gpu_context": "gpu-a",
                        "software_context": "software-a",
                        "input_context": "input-a",
                        "execution_context": path.stem,
                        "runtime_context": "runtime-a",
                    }
                )

        with pytest.raises(ValueError, match="Run ID mismatch|Mixed benchmark run_id"):
            suite_utils.validate_result_files([first, second])

    def test_directory_validation_requires_failure_reason(self, tmp_path):
        """A failed row without a reason cannot pass reportable validation."""
        csv_path = tmp_path / "results.csv"
        csv_path.write_text(
            "success,error,error_type,provenance_version,run_id,gpu_context,"
            "software_context,input_context,execution_context,runtime_context\n"
            "False,,,2,run-a,gpu-a,software-a,input-a,node-a,runtime-a\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Missing failure metadata"):
            suite_utils.validate_result_files([csv_path])

    def test_parallel_backend_writes_are_serialized(self, tmp_path, monkeypatch):
        """Overlapping Torch/JAX writers preserve both backend shards."""
        csv_path = tmp_path / "results.csv"
        provenance = self._provenance()
        monkeypatch.setattr(
            suite_utils,
            "_run_provenance",
            lambda output_path: provenance,
        )

        first_write_entered = threading.Event()
        release_first_write = threading.Event()
        second_write_done = threading.Event()
        selection_lock = threading.Lock()
        original_write = suite_utils._write_csv_atomic
        delay_next_write = True

        def delayed_write(*args, **kwargs):
            nonlocal delay_next_write
            with selection_lock:
                delay_this_write = delay_next_write
                delay_next_write = False
            if delay_this_write:
                first_write_entered.set()
                if not release_first_write.wait(timeout=5):
                    raise TimeoutError("test did not release first CSV write")
            return original_write(*args, **kwargs)

        monkeypatch.setattr(suite_utils, "_write_csv_atomic", delayed_write)
        errors = []

        def write_backend(backend, system, done=None):
            try:
                suite_utils.save_results(
                    [_result(backend, system)],
                    csv_path,
                    replace_backend=backend,
                )
            except Exception as error:  # pragma: no cover - asserted in parent thread
                errors.append(error)
            finally:
                if done is not None:
                    done.set()

        torch_writer = threading.Thread(
            target=write_backend,
            args=("torch", "torch"),
        )
        jax_writer = threading.Thread(
            target=write_backend,
            args=("jax", "jax", second_write_done),
        )
        torch_writer.start()
        assert first_write_entered.wait(timeout=5)
        jax_writer.start()

        assert not second_write_done.wait(timeout=0.1)
        release_first_write.set()
        torch_writer.join(timeout=5)
        jax_writer.join(timeout=5)

        assert not torch_writer.is_alive()
        assert not jax_writer.is_alive()
        assert errors == []
        assert {(row["backend"], row["system"]) for row in _read_rows(csv_path)} == {
            ("torch", "torch"),
            ("jax", "jax"),
        }

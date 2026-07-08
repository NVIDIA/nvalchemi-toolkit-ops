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
"""Regression tests for reportable benchmark shell entry points."""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTABLE_SCRIPT = ROOT / "benchmarks" / "run_reportable_suite.sh"
NH3_GENERATOR = ROOT / "benchmarks" / "nh3" / "generate_pbc_pdbs.sh"
NH3_TEMPLATE = ROOT / "benchmarks" / "nh3" / "ammonia.pdb"
BASH = shutil.which("bash")
if BASH is None:
    raise RuntimeError("bash is required for shell entry-point tests")


def _write_executable(path: Path, source: str) -> None:
    """Write an executable shell fixture."""
    path.write_text(textwrap.dedent(source).lstrip())
    path.chmod(0o755)


def _write_benchmark_config(path: Path, nh3_atom_count: int) -> None:
    """Write the minimal config surface consumed by NH3 preflight."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        textwrap.dedent(
            f"""
            systems:
              cscl:
                enabled: true
                atom_counts: [128]
                constant_atoms_sizes: [128]
              nh3:
                enabled: true
                pdb_dir: ../nh3
                atom_counts: [{nh3_atom_count}]
                constant_atoms_sizes: [{nh3_atom_count}]
            scaling:
              system_size:
                enabled: true
              constant_workload:
                enabled: true
                target_atoms: 1024
              batch_scaling:
                enabled: true
                max_total_atoms: 1024
            """
        ).lstrip()
    )


def _reportable_fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Create a tiny reportable-suite checkout with fake external commands."""
    fixture_root = tmp_path / "reportable-repo"
    benchmark_dir = fixture_root / "benchmarks"
    benchmark_dir.mkdir(parents=True)
    shutil.copy2(REPORTABLE_SCRIPT, benchmark_dir / REPORTABLE_SCRIPT.name)
    (benchmark_dir / "__init__.py").write_text("")
    (benchmark_dir / "config.py").write_text(
        "import yaml\n\n"
        "def load_yaml_config(path):\n"
        "    with open(path) as config_file:\n"
        "        return yaml.safe_load(config_file)\n"
    )

    _write_benchmark_config(
        benchmark_dir / "neighborlist" / "benchmark_config.yaml", 128
    )
    _write_benchmark_config(
        benchmark_dir / "interactions" / "dispersion" / "benchmark_config.yaml",
        256,
    )
    _write_benchmark_config(
        benchmark_dir / "interactions" / "electrostatics" / "benchmark_config.yaml",
        512,
    )

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "git",
        """
        #!/usr/bin/env bash
        case "${1:-}" in
            rev-parse)
                if [[ "${2:-}" == "--abbrev-ref" ]]; then
                    echo test-branch
                else
                    printf '%040d\n' 0
                fi
                ;;
            status) echo "## test-branch" ;;
            diff) ;;
        esac
        """,
    )
    _write_executable(
        fake_bin / "nvidia-smi",
        """
        #!/usr/bin/env bash
        echo "nvidia-smi stub: no GPU required"
        """,
    )
    fake_uv = fake_bin / "uv"
    _write_executable(
        fake_uv,
        """
        #!/usr/bin/env bash
        set -euo pipefail

        if [[ "${1:-}" == "--version" ]]; then
            echo "uv 0.test"
            exit 0
        fi
        if [[ -n "${FAKE_UV_CALLS:-}" ]]; then
            printf '%s\n' "$*" >> "$FAKE_UV_CALLS"
        fi
        [[ "${1:-}" == "run" ]] || exit 90
        shift
        if [[ "${1:-}" == "--no-sync" ]]; then
            shift
        fi
        [[ "${1:-}" == "python" ]] || exit 91
        shift

        if [[ "${1:-}" == "-m" && "${2:-}" == "benchmarks.benchmark_suite" ]]; then
            simulate=1
        elif [[ "${1:-}" == "-" && "${2:-}" == "run-suite-with-nh3" ]]; then
            simulate=1
        else
            exec python3 "$@"
        fi

        run_dir=""
        previous=""
        for argument in "$@"; do
            if [[ "$previous" == "--run-dir" ]]; then
                run_dir="$argument"
                break
            fi
            previous="$argument"
        done
        [[ -n "$run_dir" ]] || exit 92
        mkdir -p "$run_dir"
        : > "$run_dir/RUN_LOG.md"
        """,
    )
    _write_executable(
        fake_bin / "python",
        """
        #!/usr/bin/env bash
        set -euo pipefail

        if [[ -n "${FAKE_UV_CALLS:-}" ]]; then
            printf '%s\n' "$*" >> "$FAKE_UV_CALLS"
        fi
        if [[ "${1:-}" == "-m" && "${2:-}" == "benchmarks.benchmark_suite" ]]; then
            simulate=1
        elif [[ "${1:-}" == "-" && "${2:-}" == "run-suite-with-nh3" ]]; then
            simulate=1
        else
            exec python3 "$@"
        fi

        run_dir=""
        previous=""
        for argument in "$@"; do
            if [[ "$previous" == "--run-dir" ]]; then
                run_dir="$argument"
                break
            fi
            previous="$argument"
        done
        [[ -n "$run_dir" ]] || exit 92
        mkdir -p "$run_dir"
        : > "$run_dir/RUN_LOG.md"
        """,
    )

    scratch = tmp_path / "scratch"
    result_dir = tmp_path / "results"
    environment = os.environ.copy()
    environment.update(
        {
            "BENCHMARK_PIP_PACKAGES": "",
            "BENCHMARK_SCRATCH": str(scratch),
            "FAKE_UV_CALLS": str(tmp_path / "uv-calls.txt"),
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "UV_BIN": str(fake_uv),
        }
    )
    environment.pop("UV_PROJECT_ENVIRONMENT", None)
    return benchmark_dir / REPORTABLE_SCRIPT.name, {
        **environment,
        "TEST_RESULT_DIR": str(result_dir),
    }


def _run_reportable(
    script: Path,
    environment: dict[str, str],
    *selection: str,
) -> subprocess.CompletedProcess[str]:
    """Run the reportable helper with its heavyweight operations stubbed."""
    return subprocess.run(  # noqa: S603 - test executes a controlled script fixture
        [
            BASH,
            str(script),
            "--backend",
            "torch",
            "--output-dir",
            environment["TEST_RESULT_DIR"],
            "--skip-sync",
            "--use-current-env",
            "--no-plot",
            *selection,
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def test_reportable_preflight_skips_nh3_for_cscl(tmp_path: Path) -> None:
    """A CsCl-only shard does not require generated NH3 structures."""
    script, environment = _reportable_fixture(tmp_path)

    result = _run_reportable(script, environment, "--system", "cscl")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Missing NH3 PBC benchmark inputs" not in result.stdout


def test_reportable_preflight_checks_only_selected_benchmark(tmp_path: Path) -> None:
    """An NL shard is not blocked by D3 or EL NH3 input requirements."""
    script, environment = _reportable_fixture(tmp_path)
    source_nh3_dir = script.parent / "nh3"
    scratch_nh3_dir = Path(environment["BENCHMARK_SCRATCH"]) / "inputs" / "nh3"
    source_nh3_dir.mkdir()
    scratch_nh3_dir.mkdir(parents=True)
    for nh3_dir in (source_nh3_dir, scratch_nh3_dir):
        (nh3_dir / "ammonia_pbc_128.pdb").write_text("")

    result = _run_reportable(
        script,
        environment,
        "--benchmark",
        "nl",
        "--system",
        "nh3",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Missing NH3 PBC benchmark inputs" not in result.stdout
    assert f"nh3_dir={scratch_nh3_dir}" in result.stdout


def test_backend_all_uses_only_supported_d3_backends(tmp_path: Path) -> None:
    """The all selector does not dispatch the NL-only Warp backend for D3."""
    script, environment = _reportable_fixture(tmp_path)

    result = subprocess.run(  # noqa: S603 - controlled shell fixture
        [
            BASH,
            str(script),
            "--backend",
            "all",
            "--benchmark",
            "d3",
            "--system",
            "cscl",
            "--output-dir",
            environment["TEST_RESULT_DIR"],
            "--skip-sync",
            "--use-current-env",
            "--no-plot",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = Path(environment["FAKE_UV_CALLS"]).read_text()
    suite_calls = [line for line in calls.splitlines() if "run-suite-with-nh3" in line]
    # Torch runs the selected system in one process; JAX isolates each scaling
    # mode so its process-global allocator cannot accumulate across CSVs.
    assert len(suite_calls) == 4
    assert any("--backend torch" in line for line in suite_calls)
    jax_calls = [line for line in suite_calls if "--backend jax" in line]
    assert len(jax_calls) == 3
    assert all("--system cscl" in line for line in jax_calls)
    assert {
        mode
        for line in jax_calls
        for mode in ("system_size", "constant_workload", "batch_scaling")
        if f"--mode {mode}" in line
    } == {"system_size", "constant_workload", "batch_scaling"}
    assert not any("--backend warp" in line for line in suite_calls)


def test_reportable_rejects_implicit_reuse_of_existing_run(tmp_path: Path) -> None:
    """An old result directory is not silently treated as a new run."""
    script, environment = _reportable_fixture(tmp_path)
    result_dir = Path(environment["TEST_RESULT_DIR"])
    result_dir.mkdir(parents=True)
    run_id = "12345678123456781234567812345678"
    (result_dir / ".benchmark-run-id").write_text(f"{run_id}\n")

    result = _run_reportable(script, environment, "--system", "cscl")

    assert result.returncode != 0
    assert "output directory already belongs to run" in result.stderr


def test_reportable_resume_reuses_only_the_recorded_run(tmp_path: Path) -> None:
    """Resume and scheduler sharding require an explicit matching identity."""
    script, environment = _reportable_fixture(tmp_path)
    result_dir = Path(environment["TEST_RESULT_DIR"])
    result_dir.mkdir(parents=True)
    run_id = "12345678123456781234567812345678"
    (result_dir / ".benchmark-run-id").write_text(f"{run_id}\n")

    resumed = _run_reportable(
        script,
        environment,
        "--system",
        "cscl",
        "--resume",
    )
    mismatched = _run_reportable(
        script,
        environment,
        "--system",
        "cscl",
        "--run-id",
        "87654321876543218765432187654321",
    )

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert f"run_id={run_id}" in resumed.stdout
    assert mismatched.returncode != 0
    assert "run ID mismatch" in mismatched.stderr


def _generator_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Copy the generator and template beside fake Packmol executables."""
    source_dir = tmp_path / "source" / "benchmarks" / "nh3"
    source_dir.mkdir(parents=True)
    generator = source_dir / NH3_GENERATOR.name
    shutil.copy2(NH3_GENERATOR, generator)
    shutil.copy2(NH3_TEMPLATE, source_dir / NH3_TEMPLATE.name)

    fake_bin = tmp_path / "generator-bin"
    fake_bin.mkdir()
    return generator, source_dir, fake_bin


def _write_fake_packmol(path: Path, tool_name: str) -> None:
    """Write a Packmol stand-in that records stdin and creates its output."""
    _write_executable(
        path,
        f"""
        #!/usr/bin/env bash
        set -euo pipefail
        printf '%s %s\n' {tool_name!r} "$*" > "$FAKE_PACKMOL_ARGS"
        printf '%s\n' "$PWD" > "$FAKE_PACKMOL_CWD"
        input="$(cat)"
        printf '%s\n' "$input" > "$FAKE_PACKMOL_INPUT"
        output_path="$(printf '%s\n' "$input" | awk '$1 == "output" {{print $2; exit}}')"
        [[ -n "$output_path" ]]
        expected_atoms="${{output_path##*_}}"
        expected_atoms="${{expected_atoms%.pdb}}"
        for ((atom = 1; atom <= expected_atoms; atom++)); do
          printf 'HETATM\n'
        done > "$output_path"
        """,
    )


def _generator_environment(fake_bin: Path, tmp_path: Path) -> dict[str, str]:
    """Return an isolated environment for generator subprocesses."""
    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_PACKMOL_ARGS": str(tmp_path / "packmol-args.txt"),
            "FAKE_PACKMOL_CWD": str(tmp_path / "packmol-cwd.txt"),
            "FAKE_PACKMOL_INPUT": str(tmp_path / "packmol-input.txt"),
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
        }
    )
    return environment


def test_generator_requires_explicit_output_directory(tmp_path: Path) -> None:
    """Generation cannot silently fall back to the source checkout."""
    generator, source_dir, fake_bin = _generator_fixture(tmp_path)
    _write_fake_packmol(fake_bin / "packmol", "packmol")
    environment = _generator_environment(fake_bin, tmp_path)

    result = subprocess.run(  # noqa: S603 - test executes a controlled script fixture
        [BASH, str(generator), "--selection", "1"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode != 0
    assert "output directory is required" in (result.stdout + result.stderr).lower()
    assert not list(source_dir.glob("ammonia_pbc_*"))
    assert not list(source_dir.glob("packmol_*.log"))


def test_generator_uses_installed_packmol_and_external_output(
    tmp_path: Path,
) -> None:
    """An installed Packmol is allowed and all generated files stay in output."""
    generator, source_dir, fake_bin = _generator_fixture(tmp_path)
    _write_fake_packmol(fake_bin / "packmol", "packmol")
    _write_executable(
        fake_bin / "uvx",
        """
        #!/usr/bin/env bash
        echo uvx-called > "$FAKE_PACKMOL_ARGS"
        exit 99
        """,
    )
    environment = _generator_environment(fake_bin, tmp_path)
    output_dir = tmp_path / "nh3-output"

    result = subprocess.run(  # noqa: S603 - test executes a controlled script fixture
        [BASH, str(generator), "--output-dir", str(output_dir)],
        input="1\n",
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_dir / "ammonia_pbc_128.pdb").exists()
    assert (output_dir / "ammonia_pbc_128.inp").exists()
    assert not (output_dir / "packmol_128.log").exists()
    assert Path(environment["FAKE_PACKMOL_ARGS"]).read_text().startswith("packmol ")
    assert Path(environment["FAKE_PACKMOL_CWD"]).read_text().strip() == str(output_dir)
    packmol_input = Path(environment["FAKE_PACKMOL_INPUT"]).read_text()
    assert f"structure {source_dir / 'ammonia.pdb'}" in packmol_input
    assert not list(source_dir.glob("ammonia_pbc_*"))
    assert not list(source_dir.glob("packmol_*.log"))


def test_generator_rejects_packmol_atom_count_mismatch(tmp_path: Path) -> None:
    """Malformed Packmol output is retained as a logged generation failure."""
    generator, _source_dir, fake_bin = _generator_fixture(tmp_path)
    _write_executable(
        fake_bin / "packmol",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        input="$(cat)"
        output_path="$(printf '%s\n' "$input" | awk '$1 == "output" {print $2; exit}')"
        : > "$output_path"
        """,
    )
    environment = _generator_environment(fake_bin, tmp_path)
    output_dir = tmp_path / "nh3-output"

    result = subprocess.run(  # noqa: S603 - controlled script fixture
        [
            BASH,
            str(generator),
            "--output-dir",
            str(output_dir),
            "--selection",
            "1",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode != 0
    assert "Atom count mismatch: expected 128, got 0" in result.stderr
    assert not (output_dir / "ammonia_pbc_128.pdb").exists()
    assert (output_dir / "packmol_128.log").exists()


def test_reportable_shards_use_unique_logs_and_defer_plots() -> None:
    """Parallel shards do not race on one run log or generate shared PNGs."""
    source = REPORTABLE_SCRIPT.read_text(encoding="utf-8")

    assert '--run-log-name "RUN_LOG-${log_suffix}.md"' in source
    assert 'cp "$RESULT_DIR/RUN_LOG.md"' not in source
    assert 'INVOCATION_TAG="${SLURM_JOB_ID:-${PBS_JOBID:-$$}}"' in source
    assert (
        'full_suite_selection && [[ "$BACKEND" == "both" || "$BACKEND" == "all" ]]'
        in source
    )


def test_generator_pins_uvx_packmol_version(tmp_path: Path) -> None:
    """The uvx fallback resolves the Packmol version stated by the generator."""
    generator, source_dir, fake_bin = _generator_fixture(tmp_path)
    _write_fake_packmol(fake_bin / "uvx", "uvx")
    environment = _generator_environment(fake_bin, tmp_path)
    output_dir = tmp_path / "nh3-output"

    result = subprocess.run(  # noqa: S603 - test executes a controlled script fixture
        [
            BASH,
            str(generator),
            "--output-dir",
            str(output_dir),
            "--selection",
            "1",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(environment["FAKE_PACKMOL_ARGS"]).read_text().strip() == (
        "uvx --from packmol==21.1.4 packmol"
    )
    assert (output_dir / "ammonia_pbc_128.pdb").exists()
    assert not list(source_dir.glob("ammonia_pbc_*"))


def test_generator_reports_packmol_failure_and_external_log(tmp_path: Path) -> None:
    """A Packmol failure is nonzero and leaves its log only in the output tree."""
    generator, source_dir, fake_bin = _generator_fixture(tmp_path)
    _write_executable(
        fake_bin / "packmol",
        """
        #!/usr/bin/env bash
        cat >/dev/null
        echo "synthetic Packmol failure" >&2
        exit 42
        """,
    )
    environment = _generator_environment(fake_bin, tmp_path)
    output_dir = tmp_path / "nh3-output"

    result = subprocess.run(  # noqa: S603 - test executes a controlled script fixture
        [
            BASH,
            str(generator),
            "--output-dir",
            str(output_dir),
            "--selection",
            "1",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode != 0
    assert "Generation failed for: 128" in result.stderr
    assert (output_dir / "packmol_128.log").read_text().strip() == (
        "synthetic Packmol failure"
    )
    assert not list(source_dir.glob("packmol_*.log"))


@pytest.mark.parametrize("option", ["-h", "--help"])
def test_generator_help_does_not_require_packmol(tmp_path: Path, option: str) -> None:
    """Help remains available before output and dependency validation."""
    generator, _, fake_bin = _generator_fixture(tmp_path)
    environment = _generator_environment(fake_bin, tmp_path)

    result = subprocess.run(  # noqa: S603 - test executes a controlled script fixture
        [BASH, str(generator), option],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--output-dir" in result.stdout

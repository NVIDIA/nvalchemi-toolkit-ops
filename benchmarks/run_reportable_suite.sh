#!/usr/bin/env bash
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

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ORIGINAL_HOME="${HOME:-}"

BACKEND="all"
BENCHMARK="all"
SYSTEM_FILTER=""
MODE_FILTER=""
DRY_RUN=0
RUN_PLOTS=-1
RESULT_DIR=""
RUN_ID="${BENCHMARK_RUN_ID:-}"
RESUME=0
SKIP_SYNC=0
USE_CURRENT_ENV=0
VALIDATE_ONLY=0
UV_BIN="${UV_BIN:-uv}"
PYTHON_BIN="${PYTHON_BIN:-python}"
UV_SYNC_ARGS="${UV_SYNC_ARGS:---extra torch --extra jax --group docs}"
BENCHMARK_PIP_PACKAGES="${BENCHMARK_PIP_PACKAGES:-pyyaml>=6.0.3 nvidia-ml-py==13.590.48}"
D3_PARAMS_PATH="${BENCHMARK_D3_PARAMS_PATH:-}"
NH3_DIR="${BENCHMARK_NH3_DIR:-}"

usage() {
    cat <<'USAGE'
Usage:
  benchmarks/run_reportable_suite.sh [options]

Options:
  --backend torch|jax|warp|both|all
                             Backend pass(es) to run. `warp` is NL-only;
                             `both` means Torch and JAX. `all` adds the Warp
                             neighbor-list pass. Default: all.
  --benchmark all|nl|d3|el   Benchmark module to run. Default: all.
  --system SYSTEM            Run one system shard (for example: cscl or nh3).
  --mode MODE                Run one scaling-mode shard (for example:
                             system_size, constant_workload, or batch_scaling).
  --dry-run                  Expand plans only; do not allocate GPU memory.
  --output-dir DIR           Exact result directory. Must be outside the user's
                             home directory.
  --run-id UUID              Shared run ID for parallel scheduler shards.
  --resume                   Continue the run already recorded in --output-dir.
  --validate-only            Validate an existing complete run without executing
                             benchmark kernels. Requires an unfiltered selection.
  --plot                     Generate plots after this invocation. Use only once
                             after all parallel shards have completed.
  --no-plot                  Skip plot generation after benchmark passes.
  --skip-sync                Do not run uv sync before benchmark passes.
  --use-current-env          Do not force UV_PROJECT_ENVIRONMENT into scratch.
  --nh3-dir DIR              Generated NH3 PDB input directory. Default:
                             $BENCHMARK_SCRATCH/inputs/nh3. Must be outside the
                             user's home directory.
  --d3-params-path PATH      Pre-seeded DFT-D3 parameter .pt file or cache path.
                             Must be outside the user's home directory.
  -h, --help                 Show this help.

Environment:
  BENCHMARK_SCRATCH          Required unless /scratch/$USER exists. All caches,
                             venvs, logs, and default results go under this tree.
  BENCHMARK_D3_PARAMS_PATH   Same as --d3-params-path.
  BENCHMARK_NH3_DIR          Same as --nh3-dir.
  BENCHMARK_RUN_ID           Same as --run-id.
  UV_BIN                     uv executable to use. Default: uv.
  PYTHON_BIN                 Python executable used with --use-current-env.
                             Default: python.
  UV_SYNC_ARGS               Arguments for uv sync. Default selects compatible
                             CUDA 13 Torch/JAX extras plus docs plotting deps.
  BENCHMARK_PIP_PACKAGES     Extra runtime packages installed into the uv env.
                             Default: pyyaml and nvidia-ml-py.

This helper runs the reportable NL/D3/EL grid with 3 warmups and 10 timed runs.
It does not pass --max-total-atoms or any hardware-specific skip limits; OOMs
are recorded by the benchmark suite as success=False CSV rows and omitted from
plots.

Use --benchmark/--system/--mode with a shared --output-dir to shard reportable
runs across multiple processes or scheduler jobs. Generate plots once after all
shards have written their CSV rows.
USAGE
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

generate_run_id() {
    if [[ -r /proc/sys/kernel/random/uuid ]]; then
        tr -d '[:space:]-' < /proc/sys/kernel/random/uuid
    elif command -v uuidgen >/dev/null 2>&1; then
        uuidgen | tr -d '[:space:]-'
    else
        python3 -c 'import uuid; print(uuid.uuid4().hex)'
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            [[ $# -ge 2 ]] || die "missing value for --backend"
            BACKEND="$2"
            shift 2
            ;;
        --benchmark)
            [[ $# -ge 2 ]] || die "missing value for --benchmark"
            BENCHMARK="$2"
            shift 2
            ;;
        --system)
            [[ $# -ge 2 ]] || die "missing value for --system"
            SYSTEM_FILTER="$2"
            shift 2
            ;;
        --mode)
            [[ $# -ge 2 ]] || die "missing value for --mode"
            MODE_FILTER="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || die "missing value for --output-dir"
            RESULT_DIR="$2"
            shift 2
            ;;
        --run-id)
            [[ $# -ge 2 ]] || die "missing value for --run-id"
            RUN_ID="$2"
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --no-plot)
            RUN_PLOTS=0
            shift
            ;;
        --plot)
            RUN_PLOTS=1
            shift
            ;;
        --skip-sync)
            SKIP_SYNC=1
            shift
            ;;
        --use-current-env)
            USE_CURRENT_ENV=1
            shift
            ;;
        --validate-only)
            VALIDATE_ONLY=1
            shift
            ;;
        --nh3-dir)
            [[ $# -ge 2 ]] || die "missing value for --nh3-dir"
            NH3_DIR="$2"
            shift 2
            ;;
        --d3-params-path)
            [[ $# -ge 2 ]] || die "missing value for --d3-params-path"
            D3_PARAMS_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unexpected argument: $1"
            ;;
    esac
done

case "$BACKEND" in
    torch) BACKENDS=(torch) ;;
    jax) BACKENDS=(jax) ;;
    warp) BACKENDS=(warp) ;;
    both) BACKENDS=(torch jax) ;;
    all) BACKENDS=(torch jax warp) ;;
    *) die "unsupported backend: $BACKEND" ;;
esac

case "$BENCHMARK" in
    all|nl|d3|el) ;;
    *) die "unsupported benchmark: $BENCHMARK" ;;
esac

if [[ "$BACKEND" == "all" && "$BENCHMARK" != "all" && "$BENCHMARK" != "nl" ]]; then
    BACKENDS=(torch jax)
fi

if [[ "$BACKEND" == "warp" && "$BENCHMARK" != "nl" ]]; then
    die "the warp backend is supported only for --benchmark nl"
fi
if [[ "$VALIDATE_ONLY" -eq 1 && ( "$DRY_RUN" -eq 1 || "$BENCHMARK" != "all" || -n "$SYSTEM_FILTER" || -n "$MODE_FILTER" ) ]]; then
    die "--validate-only requires --benchmark all with no system or mode filter"
fi

if [[ -n "${BENCHMARK_SCRATCH:-}" ]]; then
    SCRATCH="$BENCHMARK_SCRATCH"
elif [[ -d "/scratch/${USER:-}" ]]; then
    SCRATCH="/scratch/${USER}/nvalchemiops-benchmarks"
else
    die "set BENCHMARK_SCRATCH to a scratch filesystem path"
fi

mkdir -p "$SCRATCH"
SCRATCH="$(cd "$SCRATCH" && pwd -P)"

reject_home_path() {
    local label="$1"
    local path="$2"
    case "$path" in
        /home)
            die "$label must not be the /home directory: $path"
            ;;
    esac
    if [[ -n "$ORIGINAL_HOME" ]]; then
        case "$path" in
            "$ORIGINAL_HOME"|"$ORIGINAL_HOME"/*)
                die "$label must not be under HOME: $path"
                ;;
        esac
    fi
}

reject_home_path "BENCHMARK_SCRATCH" "$SCRATCH"

if [[ -z "$NH3_DIR" ]]; then
    NH3_DIR="$SCRATCH/inputs/nh3"
elif [[ "$NH3_DIR" != /* ]]; then
    NH3_DIR="$(pwd -P)/$NH3_DIR"
fi
reject_home_path "NH3 input directory" "$NH3_DIR"
mkdir -p "$NH3_DIR"
NH3_DIR="$(cd "$NH3_DIR" && pwd -P)"
reject_home_path "NH3 input directory" "$NH3_DIR"

if [[ -n "$D3_PARAMS_PATH" ]]; then
    D3_PARAMS_DIR="$(dirname "$D3_PARAMS_PATH")"
    mkdir -p "$D3_PARAMS_DIR"
    D3_PARAMS_DIR="$(cd "$D3_PARAMS_DIR" && pwd -P)"
    D3_PARAMS_PATH="${D3_PARAMS_DIR}/$(basename "$D3_PARAMS_PATH")"
    reject_home_path "D3 parameter path" "$D3_PARAMS_PATH"
fi

if [[ -z "$RESULT_DIR" ]]; then
    STAMP="$(date +%Y%m%d-%H%M%S)"
    RESULT_DIR="${SCRATCH}/results/reportable-suite-${STAMP}"
fi
mkdir -p "$RESULT_DIR"
RESULT_DIR="$(cd "$RESULT_DIR" && pwd -P)"
reject_home_path "output directory" "$RESULT_DIR"

RUN_ID_MARKER="$RESULT_DIR/.benchmark-run-id"
if [[ -f "$RUN_ID_MARKER" ]]; then
    EXISTING_RUN_ID="$(tr -d '[:space:]' < "$RUN_ID_MARKER")"
    [[ -n "$EXISTING_RUN_ID" ]] || die "empty benchmark run ID: $RUN_ID_MARKER"
    if [[ -n "$RUN_ID" && "$RUN_ID" != "$EXISTING_RUN_ID" ]]; then
        die "run ID mismatch for $RESULT_DIR: requested $RUN_ID, found $EXISTING_RUN_ID"
    fi
    if [[ -z "$RUN_ID" && "$RESUME" -eq 0 ]]; then
        die "output directory already belongs to run $EXISTING_RUN_ID; pass --resume, pass --run-id $EXISTING_RUN_ID for another shard, or choose a fresh directory"
    fi
    RUN_ID="$EXISTING_RUN_ID"
elif compgen -G "$RESULT_DIR/*.csv" >/dev/null; then
    die "output directory contains CSVs without run provenance; choose a fresh directory"
elif [[ "$RESUME" -eq 1 ]]; then
    die "cannot resume: no benchmark run ID exists in $RESULT_DIR"
fi
if [[ -z "$RUN_ID" ]]; then
    RUN_ID="$(generate_run_id)"
fi
if [[ ! -f "$RUN_ID_MARKER" ]]; then
    if ! (set -o noclobber; printf '%s\n' "$RUN_ID" > "$RUN_ID_MARKER") 2>/dev/null; then
        EXISTING_RUN_ID="$(tr -d '[:space:]' < "$RUN_ID_MARKER")"
        if [[ "$RUN_ID" != "$EXISTING_RUN_ID" ]]; then
            die "another run initialized $RESULT_DIR as $EXISTING_RUN_ID; choose a fresh directory or pass that explicit --run-id for a planned shard"
        fi
    fi
fi
export BENCHMARK_RUN_ID="$RUN_ID"

mkdir -p \
    "$RESULT_DIR/logs" \
    "$SCRATCH/cache/xdg" \
    "$SCRATCH/cache/uv" \
    "$SCRATCH/cache/pre-commit" \
    "$SCRATCH/cache/warp" \
    "$SCRATCH/cache/torch-extensions" \
    "$SCRATCH/cache/pytorch-kernels" \
    "$SCRATCH/cache/jax" \
    "$SCRATCH/cache/matplotlib" \
    "$SCRATCH/cache/cuda" \
    "$SCRATCH/home"

export HOME="$SCRATCH/home"
export XDG_CACHE_HOME="$SCRATCH/cache/xdg"
export UV_CACHE_DIR="$SCRATCH/cache/uv"
export PRE_COMMIT_HOME="$SCRATCH/cache/pre-commit"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
if [[ "$USE_CURRENT_ENV" -eq 0 ]]; then
    export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$SCRATCH/venv}"
fi
export WARP_CACHE_PATH="$SCRATCH/cache/warp"
export TORCH_EXTENSIONS_DIR="$SCRATCH/cache/torch-extensions"
export PYTORCH_KERNEL_CACHE_PATH="$SCRATCH/cache/pytorch-kernels"
export JAX_COMPILATION_CACHE_DIR="$SCRATCH/cache/jax"
export MPLCONFIGDIR="$SCRATCH/cache/matplotlib"
export CUDA_CACHE_PATH="$SCRATCH/cache/cuda"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

cd "$ROOT"

INVOCATION_TAG="${SLURM_JOB_ID:-${PBS_JOBID:-$$}}"
INVOCATION_SUFFIX="${BACKEND}-${BENCHMARK}"
[[ -z "$SYSTEM_FILTER" ]] || INVOCATION_SUFFIX="${INVOCATION_SUFFIX}-${SYSTEM_FILTER}"
[[ -z "$MODE_FILTER" ]] || INVOCATION_SUFFIX="${INVOCATION_SUFFIX}-${MODE_FILTER}"
INVOCATION_SUFFIX="${INVOCATION_SUFFIX}-${INVOCATION_TAG}"
LOG_PATH="$RESULT_DIR/logs/reportable-suite-${INVOCATION_SUFFIX}.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "reportable_suite_started_at=$(date -Is)"
echo "root=$ROOT"
echo "scratch=$SCRATCH"
echo "result_dir=$RESULT_DIR"
echo "run_id=$RUN_ID"
echo "resume=$RESUME"
echo "backend=$BACKEND"
echo "benchmark=$BENCHMARK"
echo "system_filter=${SYSTEM_FILTER:-<all>}"
echo "mode_filter=${MODE_FILTER:-<all>}"
echo "dry_run=$DRY_RUN"
echo "run_plots=$([[ "$RUN_PLOTS" -eq -1 ]] && echo auto || echo "$RUN_PLOTS")"
echo "skip_sync=$SKIP_SYNC"
echo "use_current_env=$USE_CURRENT_ENV"
echo "validate_only=$VALIDATE_ONLY"
echo "uv_bin=$UV_BIN"
echo "python_bin=$PYTHON_BIN"
echo "uv_sync_args=$UV_SYNC_ARGS"
echo "benchmark_pip_packages=${BENCHMARK_PIP_PACKAGES:-<none>}"
echo "uv_project_environment=${UV_PROJECT_ENVIRONMENT:-<current>}"
echo "xla_python_client_allocator=${XLA_PYTHON_CLIENT_ALLOCATOR:-<unset>}"
echo "tf_gpu_allocator=${TF_GPU_ALLOCATOR:-<unset>}"
echo "d3_params_path=${D3_PARAMS_PATH:-<xdg-cache-default>}"
echo "nh3_dir=$NH3_DIR"
echo

git rev-parse --abbrev-ref HEAD | tee "$RESULT_DIR/logs/git-branch.txt"
git rev-parse HEAD | tee "$RESULT_DIR/logs/git-head.txt"
git status --short --branch | tee "$RESULT_DIR/logs/git-status.txt"
git diff --stat | tee "$RESULT_DIR/logs/git-diff-stat.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | tee "$RESULT_DIR/logs/nvidia-smi.txt"
fi

"$UV_BIN" --version
if [[ "$SKIP_SYNC" -eq 0 ]]; then
    read -r -a uv_sync_args <<< "$UV_SYNC_ARGS"
    "$UV_BIN" sync "${uv_sync_args[@]}"
    if [[ -n "$BENCHMARK_PIP_PACKAGES" ]]; then
        read -r -a benchmark_pip_packages <<< "$BENCHMARK_PIP_PACKAGES"
        if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" && -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
            "$UV_BIN" pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" "${benchmark_pip_packages[@]}"
        else
            "$UV_BIN" pip install "${benchmark_pip_packages[@]}"
        fi
    fi
fi

uv_run_args=()
if [[ "$SKIP_SYNC" -eq 1 ]]; then
    uv_run_args+=(--no-sync)
fi

run_python() {
    if [[ "$USE_CURRENT_ENV" -eq 1 ]]; then
        "$PYTHON_BIN" "$@"
    else
        "$UV_BIN" run "${uv_run_args[@]}" python "$@"
    fi
}

selection_uses_nh3() {
    [[ -z "$SYSTEM_FILTER" || "$SYSTEM_FILTER" == "all" || "$SYSTEM_FILTER" == "nh3" ]]
}

if [[ "$DRY_RUN" -eq 0 ]] && selection_uses_nh3; then
    run_python - preflight-nh3-inputs \
        "$BENCHMARK" "$MODE_FILTER" "$NH3_DIR" "$ROOT" <<'PY'
from __future__ import annotations

import shlex
import sys
from pathlib import Path

from benchmarks.config import load_yaml_config

_, marker, benchmark, mode_filter, nh3_dir_arg, root_arg = sys.argv
if marker != "preflight-nh3-inputs":
    raise SystemExit(f"unexpected NH3 preflight marker: {marker}")

root = Path(root_arg)
nh3_dir = Path(nh3_dir_arg)
CONFIG_PATHS = {
    "nl": root / "benchmarks/neighborlist/benchmark_config.yaml",
    "d3": root / "benchmarks/interactions/dispersion/benchmark_config.yaml",
    "el": root / "benchmarks/interactions/electrostatics/benchmark_config.yaml",
}
selected_benchmarks = tuple(CONFIG_PATHS) if benchmark == "all" else (benchmark,)


def selected_modes(config: dict) -> tuple[str, ...]:
    """Return enabled scaling modes relevant to this reportable shard."""
    scaling = config.get("scaling", {})
    if mode_filter and mode_filter != "all":
        candidates = (mode_filter,)
    else:
        candidates = tuple(scaling)
    return tuple(
        name
        for name in candidates
        if isinstance(scaling.get(name), dict)
        and scaling[name].get("enabled", True)
    )


def required_atom_counts(config: dict, nh3_config: dict) -> set[int]:
    """Return PDB sizes needed by the selected scaling modes."""
    atom_counts = {int(value) for value in nh3_config.get("atom_counts", [])}
    required: set[int] = set()
    for mode_name in selected_modes(config):
        if mode_name == "batch_scaling":
            required.update(
                int(value)
                for value in nh3_config.get("constant_atoms_sizes", [1024, 8192])
            )
        elif mode_name == "constant_workload":
            target_atoms = int(config["scaling"][mode_name]["target_atoms"])
            required.update(value for value in atom_counts if value <= target_atoms)
        else:
            required.update(atom_counts)
    return required

missing: set[Path] = set()
for benchmark_name in selected_benchmarks:
    config_path = CONFIG_PATHS[benchmark_name]
    config = load_yaml_config(config_path)
    nh3_config = config.get("systems", {}).get("nh3", {})
    if not nh3_config.get("enabled", True):
        continue
    for atom_count in required_atom_counts(config, nh3_config):
        path = nh3_dir / f"ammonia_pbc_{atom_count}.pdb"
        if not path.exists():
            missing.add(path)

if missing:
    files = "\n".join(f"  - {path}" for path in sorted(missing))
    generator = root / "benchmarks/nh3/generate_pbc_pdbs.sh"
    command = (
        f"bash {shlex.quote(str(generator))} "
        f"--output-dir {shlex.quote(str(nh3_dir))} --selection 1-11"
    )
    raise SystemExit(
        "Missing NH3 PBC benchmark inputs for reportable run:\n"
        f"{files}\n"
        f"Generate the reportable NH3 inputs with:\n  {command}"
    )
PY
fi

run_benchmark_suite_with_nh3() {
    run_python - run-suite-with-nh3 "$NH3_DIR" "$@" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from benchmarks import benchmark_suite

_, marker, nh3_dir_arg, *suite_args = sys.argv
if marker != "run-suite-with-nh3":
    raise SystemExit(f"unexpected benchmark suite marker: {marker}")

nh3_dir = Path(nh3_dir_arg).resolve()
config_paths = {
    Path(info["config"]).resolve() for info in benchmark_suite.RUNNERS.values()
}
load_yaml_config = benchmark_suite.load_yaml_config


def load_reportable_config(config_path):
    """Load a suite config with its NH3 input directory redirected to scratch."""
    config = load_yaml_config(config_path)
    if Path(config_path).resolve() in config_paths:
        nh3_config = config.get("systems", {}).get("nh3")
        if isinstance(nh3_config, dict):
            nh3_config["pdb_dir"] = str(nh3_dir)
    return config


benchmark_suite.load_yaml_config = load_reportable_config
sys.argv = [str(Path(benchmark_suite.__file__)), *suite_args]
raise SystemExit(benchmark_suite.main())
PY
}

run_suite() {
    local backend="$1"
    local suite_benchmark="${2:-$BENCHMARK}"
    local suite_system="${3:-$SYSTEM_FILTER}"
    local suite_mode="${4:-$MODE_FILTER}"
    if [[ "$backend" == "warp" && "$suite_benchmark" == "all" ]]; then
        suite_benchmark="nl"
    fi
    local log_suffix="${backend}-${INVOCATION_TAG}"
    [[ "$suite_benchmark" == "all" ]] || log_suffix="${log_suffix}-${suite_benchmark}"
    [[ -z "$suite_system" ]] || log_suffix="${log_suffix}-${suite_system}"
    [[ -z "$suite_mode" ]] || log_suffix="${log_suffix}-${suite_mode}"
    local log_file="$RESULT_DIR/logs/${log_suffix}.log"
    local common_args=(
        --benchmark "$suite_benchmark"
        --backend "$backend"
        --timing-runs 10
        --warmup-runs 3
    )
    if [[ -n "$suite_system" ]]; then
        common_args+=(--system "$suite_system")
    fi
    if [[ -n "$suite_mode" ]]; then
        common_args+=(--mode "$suite_mode")
    fi
    if [[ -n "$D3_PARAMS_PATH" ]]; then
        common_args+=(--d3-params-path "$D3_PARAMS_PATH")
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        run_benchmark_suite_with_nh3 \
            "${common_args[@]}" \
            --dry-run | tee "$log_file"
        return "${PIPESTATUS[0]}"
    fi

    run_benchmark_suite_with_nh3 \
        "${common_args[@]}" \
        --run-dir "$RESULT_DIR" \
        --run-log-name "RUN_LOG-${log_suffix}.md" \
        --no-plot | tee "$log_file"
    local status="${PIPESTATUS[0]}"
    if [[ "$status" -ne 0 ]]; then
        return "$status"
    fi
}

run_jax_isolated_shards() {
    local benchmarks=(nl d3 el)
    local systems=(cscl nh3)
    local modes=(system_size constant_workload batch_scaling)
    if [[ "$BENCHMARK" != "all" ]]; then
        benchmarks=("$BENCHMARK")
    fi
    if [[ -n "$SYSTEM_FILTER" && "$SYSTEM_FILTER" != "all" ]]; then
        systems=("$SYSTEM_FILTER")
    fi
    if [[ -n "$MODE_FILTER" && "$MODE_FILTER" != "all" ]]; then
        modes=("$MODE_FILTER")
    fi

    # XLA's device allocator is process-global and keeps a high-water-mark
    # pool. Each reportable CSV covers unrelated shapes, so run every JAX CSV
    # shard in a fresh interpreter and let process exit release the pool.
    for suite_benchmark in "${benchmarks[@]}"; do
        for suite_system in "${systems[@]}"; do
            for suite_mode in "${modes[@]}"; do
                echo
                echo "--- jax shard=${suite_benchmark}/${suite_system}/${suite_mode} ---"
                run_suite \
                    jax \
                    "$suite_benchmark" \
                    "$suite_system" \
                    "$suite_mode"
            done
        done
    done
}

if [[ "$VALIDATE_ONLY" -eq 0 ]]; then
    for backend in "${BACKENDS[@]}"; do
        echo
        echo "=== backend=$backend ==="
        if [[ "$backend" == "jax" && "$DRY_RUN" -eq 0 ]]; then
            run_jax_isolated_shards
        else
            run_suite "$backend"
        fi
    done
else
    echo
    echo "Skipping benchmark execution; validating the existing result set."
fi

full_suite_selection() {
    [[ "$BENCHMARK" == "all" && -z "$SYSTEM_FILTER" && -z "$MODE_FILTER" ]]
}

if [[ "$DRY_RUN" -eq 0 ]]; then
    if ! full_suite_selection; then
        echo
        echo "Skipping full-suite CSV completeness check for selected shard."
    else
    run_python - "$RESULT_DIR" "$NH3_DIR" "${BACKENDS[@]}" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

from benchmarks.benchmark_suite import (
    REPORTABLE_CSV_NAMES,
    validate_reportable_case_matrix,
)
from benchmarks.suite_utils import validate_result_files

result_dir = Path(sys.argv[1])
nh3_dir = Path(sys.argv[2])
expected_backends = set(sys.argv[3:])
csv_paths = sorted(result_dir.glob("*.csv"))
found_names = {path.name for path in csv_paths}
if found_names != REPORTABLE_CSV_NAMES:
    missing = sorted(REPORTABLE_CSV_NAMES - found_names)
    unexpected = sorted(found_names - REPORTABLE_CSV_NAMES)
    details = []
    if missing:
        details.append("missing: " + ", ".join(missing))
    if unexpected:
        details.append("unexpected: " + ", ".join(unexpected))
    raise SystemExit("invalid reportable CSV set; " + "; ".join(details))
run_id = (result_dir / ".benchmark-run-id").read_text(encoding="ascii").strip()
try:
    validation = validate_result_files(csv_paths, expected_run_id=run_id)
except ValueError as error:
    raise SystemExit(str(error)) from error
print(
    "validated provenance: "
    f"rows={validation['rows']} successes={validation['successes']} "
    f"failures={validation['failures']}"
)
try:
    matrix = validate_reportable_case_matrix(
        csv_paths,
        expected_backends,
        nh3_dir=nh3_dir,
    )
except ValueError as error:
    raise SystemExit(str(error)) from error
print(
    "validated case matrix: "
    f"planned={matrix['planned']} emitted={matrix['emitted']}"
)

summary: dict[tuple[str, str], list[bool]] = {}
for path in csv_paths:
    prefix = path.name.split("-", 1)[0]
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"empty CSV: {path.name}")
    for row in rows:
        backend = row.get("backend", "")
        success = row.get("success", "True") == "True"
        summary.setdefault((prefix, backend), []).append(success)

supported_backends = {
    "nl": {"torch", "jax", "warp"},
    "d3": {"torch", "jax"},
    "el": {"torch", "jax"},
}
for prefix in ("nl", "d3", "el"):
    for backend in sorted(expected_backends & supported_backends[prefix]):
        rows = summary.get((prefix, backend), [])
        if not rows:
            raise SystemExit(f"missing rows for {prefix}/{backend}")
        if not any(rows):
            raise SystemExit(f"no successful rows for {prefix}/{backend}")
        print(
            f"{prefix}/{backend}: rows={len(rows)} "
            f"successes={sum(rows)} failures={len(rows) - sum(rows)}"
        )
PY
    fi

    if [[ "$RUN_PLOTS" -eq -1 ]]; then
        if full_suite_selection && [[ "$BACKEND" == "both" || "$BACKEND" == "all" ]]; then
            RUN_PLOTS=1
        else
            RUN_PLOTS=0
            echo
            echo "Skipping automatic plots for a shard or single-backend pass."
            echo "Run one final invocation with --plot after every shard completes."
        fi
    fi
    if [[ "$RUN_PLOTS" -eq 1 ]]; then
        run_python -m benchmarks.benchmark_suite \
            --plot-only "$RESULT_DIR" \
            --expected-backends "${BACKENDS[@]}" \
            --plots all
    fi
fi

echo
echo "reportable_suite_finished_at=$(date -Is)"
echo "result_dir=$RESULT_DIR"

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
# Measure per-test line coverage for every test module.
#
# Writes one coverage data file per module under .covctx/ with a dynamic
# context per test node id, so a later pass can solve for the smallest set of
# tests that reproduces the full suite's coverage.
#
# Modules run in separate pytest processes for the same reason CI splits them:
# JAX and Torch both grab GPU memory and do not release it cleanly to each other.
set -u -o pipefail

PY="${PY:-$PWD/.venv-cu12/bin/python}"
OUT="${OUT:-$PWD/.covctx}"
mkdir -p "$OUT"

MODULES=(
  "types:test/test_types.py"
  "math:test/math"
  "neighbors:test/neighbors"
  "dynamics:test/dynamics"
  "batch_utils:test/test_batch_utils.py"
  "warp_dispatch:test/test_warp_dispatch.py"
  "torch_boundary:test/torch"
  "segment_ops:test/test_segment_ops.py"
  "segment_ops_backward:test/test_segment_ops_backward.py"
  "segment_ops_torch:test/test_segment_ops_torch.py"
  "segment_ops_jax:test/test_segment_ops_jax.py"
  "interactions:test/interactions"
)

for entry in "${MODULES[@]}"; do
  name="${entry%%:*}"
  path="${entry#*:}"
  if [ -f "$OUT/$name.done" ]; then
    echo "[skip] $name already measured"
    continue
  fi
  echo "[run ] $name ($path) at $(date -Is)"
  rm -f "$OUT/.coverage.$name" "$OUT/.coverage.$name".*
  COVERAGE_FILE="$OUT/.coverage.$name" \
    "$PY" -m pytest "$path" \
      -p no:cacheprovider \
      --cov=nvalchemiops --cov-context=test --cov-report= --cov-fail-under=0 \
      -q --no-header \
      > "$OUT/$name.log" 2>&1
  rc=$?
  echo "[done] $name rc=$rc at $(date -Is)"
  # rc 5 == no tests collected; both 0 and 5 are acceptable outcomes here.
  if [ $rc -eq 0 ] || [ $rc -eq 5 ]; then touch "$OUT/$name.done"; fi
done

echo "[all ] finished at $(date -Is)"

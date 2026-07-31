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
# Run only the tests marked `slow`, which pytest-skip-slow otherwise skips.
#
# These have not executed in CI at all: nothing passed --slow, so the marker has
# meant "never runs" rather than "runs nightly". Before making the nightly tier
# pass --slow we need to know whether they still pass.
#
# Groups run in separate processes for the usual reason: JAX and Torch do not
# release GPU memory to each other cleanly.
set -u -o pipefail

PY="${PY:-$PWD/.venv-cu12/bin/python}"
OUT="${OUT:-$PWD/.slowcheck}"
mkdir -p "$OUT"

# Not named GROUPS: bash reserves that for the caller's group ids and silently
# ignores an assignment to it.
TEST_GROUPS=(
  "math:test/math"
  "neighbors:test/neighbors"
  "dynamics:test/dynamics"
  "segment_ops_torch:test/test_segment_ops_torch.py"
  "dispersion:test/interactions/dispersion"
  "electrostatics:test/interactions/electrostatics --ignore=test/interactions/electrostatics/bindings"
  "electrostatics_jax:test/interactions/electrostatics/bindings/jax"
  "electrostatics_torch:test/interactions/electrostatics/bindings/torch"
)

for entry in "${TEST_GROUPS[@]}"; do
  name="${entry%%:*}"
  args="${entry#*:}"
  if [ -f "$OUT/$name.done" ]; then
    echo "[skip] $name"
    continue
  fi
  echo "[run ] $name at $(date -Is)"
  # shellcheck disable=SC2086 -- args intentionally word-split into pytest flags
  "$PY" -m pytest $args --slow -m slow \
    -p no:cacheprovider --no-header \
    > "$OUT/$name.log" 2>&1
  rc=$?
  tail -1 "$OUT/$name.log"
  echo "[done] $name rc=$rc at $(date -Is)"
  if [ $rc -eq 0 ] || [ $rc -eq 5 ]; then touch "$OUT/$name.done"; fi
done

echo "[all ] finished at $(date -Is)"
grep -h -E "^(FAILED|ERROR)" "$OUT"/*.log 2>/dev/null | sort -u > "$OUT/failures.txt"
echo "failures recorded: $(wc -l < "$OUT/failures.txt")"

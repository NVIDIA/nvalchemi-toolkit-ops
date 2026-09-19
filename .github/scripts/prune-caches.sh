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
#
# Delete cache entries on the current ref whose key starts with $PREFIX.
#
# Cache entries are immutable, so every key embeds a run id and each run would
# otherwise leave a ~200 MB entry behind until the seven-day sweep. Rotating
# explicitly is what keeps the repository inside its 10 GB budget.
#
# Uses the REST API rather than the GitHub CLI on purpose: the GPU jobs run in a
# bare CUDA container where `gh` is not installed, so a `gh` call exits 127 and
# rotation silently never happens. That failure mode is invisible precisely
# because a stale cache still restores.
#
# Requires: PREFIX, GH_TOKEN, and `actions: write` on the calling job.
# Failures warn rather than fail the build: an unrotated cache is a budget
# problem, not a correctness one, and losing a test run over it is worse.
set -u -o pipefail

: "${PREFIX:?PREFIX must be set}"
: "${GH_TOKEN:?GH_TOKEN must be set}"

API="https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/caches"
AUTH=(-H "Authorization: Bearer ${GH_TOKEN}" -H "Accept: application/vnd.github+json")

listing=$(curl -sS --fail-with-body "${AUTH[@]}" \
  "${API}?per_page=100&ref=${GITHUB_REF}") || {
  echo "::warning title=Cache prune::Could not list caches; entries were not rotated."
  exit 0
}

ids=$(PREFIX="$PREFIX" python3 -c '
import json, os, sys

prefix = os.environ["PREFIX"]
entries = json.load(sys.stdin).get("actions_caches", [])
for entry in entries:
    if entry["key"].startswith(prefix):
        print(entry["id"])
' <<<"$listing") || {
  echo "::warning title=Cache prune::Could not parse cache listing."
  exit 0
}

if [[ -z "$ids" ]]; then
  echo "No superseded caches matching '${PREFIX}' on ${GITHUB_REF}"
  exit 0
fi

deleted=0
for id in $ids; do
  if curl -sS --fail-with-body -X DELETE "${AUTH[@]}" "${API}/${id}" >/dev/null; then
    deleted=$((deleted + 1))
  else
    # 403 here almost always means the job is missing `actions: write`.
    echo "::warning title=Cache prune::Failed to delete cache ${id} (missing actions: write?)"
  fi
done
echo "Pruned ${deleted} cache entries matching '${PREFIX}'"

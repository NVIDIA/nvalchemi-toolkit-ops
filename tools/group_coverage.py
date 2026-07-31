#!/usr/bin/env python3
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
"""Choose the cheapest set of tests that still clears the coverage gate.

Works in the gate's own units — percentage of `nvalchemiops` statements as
`coverage report` computes it — rather than in a line-set proxy, so the number
this prints is the number CI will check.

Three details make that non-trivial, and all three are handled explicitly.

*Context-free lines.* Some lines are recorded under an empty context: they ran at
import, in a session-scoped fixture, or on a worker thread, since coverage tracks
context per thread. They come with the process rather than with a particular test,
so any selection that runs a group at all is credited with that group's
context-free lines, and no individual test is credited with them.

*Exclusions.* The gate does not count every line. This project excludes
`@wp.kernel`, `@wp.func`, `@torch.jit.script` and `.register_fake` bodies, which
is a large share of the source. Rather than re-derive that, the denominator comes
from coverage's own analysis with the project config loaded.

*Process separation.* Selected files are emitted grouped by the Makefile's groups,
never merged across them, because JAX and Torch must not share a pytest process.

Usage
-----
    # what each group costs and covers
    python tools/group_coverage.py --durations ci_durations.json

    # cheapest set of whole test files reaching 72%
    python tools/group_coverage.py --durations ci_durations.json \\
        --select --granularity file --target 72
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import coverage

# Maps a test path onto the Makefile group that runs it. Ordered longest-prefix
# first so the jax and torch bindings win over the broader entries containing
# them. Anything unmatched falls back to its measurement group.
MAKEFILE_GROUPS: tuple[tuple[str, str], ...] = (
    ("test/interactions/electrostatics/bindings/jax", "electrostatics_jax"),
    ("test/interactions/electrostatics/bindings/torch", "electrostatics_torch"),
    ("test/interactions/electrostatics", "electrostatics"),
    ("test/interactions/dispersion", "dispersion"),
    ("test/interactions", "interactions"),
    ("test/neighbors", "neighbors"),
    ("test/dynamics", "dynamics"),
    ("test/math", "math"),
    ("test/torch", "torch_boundary"),
)

Line = tuple[str, int]


def makefile_group(nodeid: str, fallback: str) -> str:
    """Return the Makefile group that would run a given test node id."""
    for prefix, name in MAKEFILE_GROUPS:
        if nodeid.startswith(prefix):
            return name
    return fallback


def countable_statements(source_root: Path) -> dict[str, set[int]]:
    """Return the statements the coverage gate counts, per source file."""
    cov = coverage.Coverage(config_file="pyproject.toml")
    cov.load()
    countable: dict[str, set[int]] = {}
    for path in sorted(source_root.rglob("*.py")):
        try:
            statements = cov.analysis2(str(path))[1]
        except Exception as exc:  # noqa: S112 - reported, then skipped
            print(f"skipping {path}: {type(exc).__name__}: {exc}")
            continue
        countable[str(path.resolve())] = set(statements)
    return countable


def load(cov_dir: Path) -> tuple[dict[str, dict[str, set[Line]]], dict[str, set[Line]]]:
    """Read per-group databases into per-test line sets and context-free lines."""
    per_test: dict[str, dict[str, set[Line]]] = defaultdict(lambda: defaultdict(set))
    free: dict[str, set[Line]] = defaultdict(set)

    for data_file in sorted(cov_dir.iterdir()):
        if not data_file.name.startswith(".coverage.") or not data_file.is_file():
            continue
        measurement_group = data_file.name.split(".")[2]
        data = coverage.CoverageData(basename=str(data_file))
        data.read()
        for measured in data.measured_files():
            for lineno, contexts in data.contexts_by_lineno(measured).items():
                key = (measured, lineno)
                for context in contexts:
                    if context:
                        nodeid = context.split("|", 1)[0]
                        per_test[measurement_group][nodeid].add(key)
                    else:
                        free[measurement_group].add(key)

    return per_test, dict(free)


def build_candidates(
    per_test: dict[str, dict[str, set[Line]]],
    durations: dict[str, float],
    granularity: str,
) -> tuple[dict[str, set[Line]], dict[str, float], dict[str, str], dict[str, str]]:
    """Return candidate line sets, costs, owning Makefile group, and source group."""
    lines: dict[str, set[Line]] = defaultdict(set)
    cost: dict[str, float] = defaultdict(float)
    group_of: dict[str, str] = {}
    measurement_of: dict[str, str] = {}

    for measurement_group, tests in per_test.items():
        for nodeid, covered in tests.items():
            group = makefile_group(nodeid, measurement_group)
            key = group if granularity == "group" else nodeid.split("::", 1)[0]
            lines[key].update(covered)
            cost[key] += durations.get(nodeid, 0.0)
            group_of[key] = group
            measurement_of[key] = measurement_group

    return dict(lines), dict(cost), group_of, measurement_of


def main() -> None:
    """Report per-group coverage and optionally select a cheaper subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cov-dir", type=Path, default=Path(".covctx"))
    parser.add_argument("--durations", type=Path, required=True)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--granularity", choices=("group", "file"), default="group")
    parser.add_argument("--target", type=float, default=70.0)
    args = parser.parse_args()

    durations = json.loads(args.durations.read_text())
    per_test, free = load(args.cov_dir)
    statements = countable_statements(Path("nvalchemiops"))
    total = sum(len(s) for s in statements.values())

    def pct(lines: set[Line]) -> float:
        """Covered statements as a percentage, the way the gate counts."""
        hit = sum(1 for path, no in lines if no in statements.get(path, ()))
        return 100.0 * hit / total if total else 0.0

    candidates, cost, group_of, measurement_of = build_candidates(
        per_test, durations, args.granularity
    )

    everything: set[Line] = set()
    for lines in candidates.values():
        everything |= lines
    for lines in free.values():
        everything |= lines

    print(f"total statements: {total}")
    print(f"everything measured: {pct(everything):.1f}%")
    print(f"candidates: {len(candidates)} at {args.granularity} granularity")

    if not args.select:
        print(f"\n{'alone%':>7} {'min':>7}  candidate")
        for name in sorted(candidates, key=lambda n: -cost.get(n, 0.0)):
            covered = candidates[name] | free.get(measurement_of[name], set())
            print(f"{pct(covered):7.1f} {cost.get(name, 0.0) / 60:7.1f}  {name}")
        return

    chosen: list[str] = []
    accumulated: set[Line] = set()
    used_groups: set[str] = set()
    pool = dict(candidates)
    spent = 0.0

    print(f"\n=== greedy selection to {args.target:.0f}% ===")
    print(f"{'cum%':>7} {'min':>7} {'cumMin':>7}  candidate")

    while pool and pct(accumulated) < args.target:
        current = pct(accumulated)
        best_name, best_ratio = None, 0.0
        for name, lines in pool.items():
            # Running a candidate also runs its group's process, so a group not
            # yet in the selection brings its context-free lines along.
            extra = free.get(measurement_of[name], set())
            gain = pct(accumulated | lines | extra) - current
            if gain <= 0.0:
                continue
            ratio = gain / max(cost.get(name, 0.0), 1.0)
            if ratio > best_ratio:
                best_name, best_ratio = name, ratio

        if best_name is None:
            break

        accumulated |= pool.pop(best_name)
        accumulated |= free.get(measurement_of[best_name], set())
        used_groups.add(group_of[best_name])
        spent += cost.get(best_name, 0.0)
        chosen.append(best_name)
        print(
            f"{pct(accumulated):7.1f} {cost.get(best_name, 0.0) / 60:7.1f} "
            f"{spent / 60:7.1f}  {best_name}"
        )

    print(
        f"\nreaches {pct(accumulated):.1f}% in {spent / 60:.1f} min "
        f"({len(chosen)} of {len(candidates)} candidates)"
    )

    if args.granularity == "group":
        print(f"\nMINIMAL_GROUPS := {' '.join(sorted(chosen))}")
        return

    by_group: dict[str, list[str]] = defaultdict(list)
    for name in chosen:
        by_group[group_of[name]].append(name)
    print("\n# Paste into the Makefile. Groups stay separate so JAX and Torch")
    print("# never share a pytest process.")
    print(f"MINIMAL_GROUPS := {' '.join(sorted(f'{g}_min' for g in by_group))}")
    for group in sorted(by_group):
        paths = " \\\n\t".join(sorted(by_group[group]))
        print(f"ARGS_{group}_min := {paths}")


if __name__ == "__main__":
    main()

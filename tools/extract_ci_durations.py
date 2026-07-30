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
"""Recover a per-test duration table from a GitHub Actions job log.

The suite runs under ``-vv``, so each test writes one line, and Actions prefixes
every line with a timestamp. Successive differences therefore give a duration per
test without instrumenting the suite.

The attribution is one test off: a line is written when a test *finishes*, so a
gap covers the following test plus any fixture setup charged between them. That is
accurate enough to find the tests costing minutes rather than milliseconds, which
is all these numbers are used for.

Usage
-----
    python tools/extract_ci_durations.py <job-log> [-o durations.json]
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
from pathlib import Path

# "<iso-timestamp> test/path.py::Class::test_name PASSED [ 42%]"
_RESULT_LINE = re.compile(
    r"^(?P<ts>\S+Z) (?P<nodeid>test/\S+::\S+) "
    r"(?P<outcome>PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR)"
)

# A gap larger than this spans a pytest process restart between module groups,
# so it measures interpreter startup rather than the test it precedes.
_MAX_PLAUSIBLE_SECONDS = 3000.0


def parse_log(path: Path) -> dict[str, float]:
    """Return the longest observed duration for each test node id in a job log."""
    durations: dict[str, float] = {}
    previous: datetime.datetime | None = None

    with path.open(errors="replace") as handle:
        for line in handle:
            match = _RESULT_LINE.match(line)
            if match is None:
                continue

            stamp = datetime.datetime.fromisoformat(
                match.group("ts").replace("Z", "+00:00")
            )
            if previous is not None:
                elapsed = (stamp - previous).total_seconds()
                if 0.0 <= elapsed < _MAX_PLAUSIBLE_SECONDS:
                    nodeid = match.group("nodeid")
                    # Keep the worst observation; a test may appear in several
                    # module groups and we size the suite for the slow case.
                    durations[nodeid] = max(durations.get(nodeid, 0.0), elapsed)
            previous = stamp

    return durations


def main() -> None:
    """Parse a job log and write the duration table as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="GitHub Actions job log")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("ci_durations.json"),
        help="where to write the duration table",
    )
    args = parser.parse_args()

    durations = parse_log(args.log)
    args.output.write_text(json.dumps(durations, indent=0, sort_keys=True))

    total = sum(durations.values())
    print(f"{len(durations)} tests, {total / 3600:.2f} h total -> {args.output}")


if __name__ == "__main__":
    main()

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
"""Compile every Warp kernel in nvalchemiops without running the tests.

The expensive part of a cold CI run is not arithmetic, it is NVIDIA Warp
generating and compiling CUDA source. The PME kernels are specialised per
(spline order, l_max, dtype) with fully unrolled loops, so order 6 emits 216
unrolled blocks and NVRTC takes minutes on it. One eight-atom test measured at
eighteen minutes on a cold cache, essentially all of it compilation.

That cost lands directly on the critical path of the test suite, which runs
serially, so moving it into a job nobody waits on is worth a lot. Warming
compiles two to three kernels at a time; a test run compiles them one at a time,
in between doing everything else.

Two properties make this cheap to run often:

*Warp content-hashes each module*, so re-warming an already-warm cache costs
seconds — only kernels whose source actually changed are rebuilt. A cold build
takes tens of minutes; the steady-state build takes about thirty seconds.

*Compilation is parallelised.* Warp accepts ``max_workers`` for module loading
but defaults it to ``0``, meaning serial, which leaves one core saturated while
the rest idle. We default to 60% of visible cores. Speed-up is well below linear
regardless: the build ends on a handful of very large translation units that
cannot be split, so the tail is effectively serial either way.

Usage
-----
    WARP_CACHE_PATH=.warp-cache python tools/warm_warp_cache.py
    python tools/warm_warp_cache.py --jobs 0     # serial, for comparison
"""

from __future__ import annotations

import argparse
import importlib
import os
import pkgutil
import time

# Fraction of visible cores to compile with. Warp parallelises module loading
# with threads and releases the GIL for the compiler itself, so this tracks real
# cores rather than being capped by Python. Raising it past ~60% buys little,
# because the tail of the build is a few huge modules that cannot be split.
_CORE_FRACTION = 0.6


def default_workers() -> int:
    """Return the worker count to compile with: 60% of visible cores, at least 1."""
    return max(1, int((os.cpu_count() or 1) * _CORE_FRACTION))


def import_package(quiet: bool) -> tuple[int, list[str]]:
    """Import every nvalchemiops submodule so its kernels get registered.

    Registration happens at import: the per-order kernel factories run at module
    scope, so importing is what makes the specialisations visible to Warp.
    """
    import nvalchemiops

    imported = 0
    failed: list[str] = []
    for module in pkgutil.walk_packages(
        nvalchemiops.__path__, f"{nvalchemiops.__name__}."
    ):
        try:
            importlib.import_module(module.name)
            imported += 1
        except Exception as exc:
            failed.append(f"{module.name}: {type(exc).__name__}: {exc}")
            if not quiet:
                print(f"  skip {module.name}: {type(exc).__name__}: {exc}")
    return imported, failed


def main() -> None:
    """Import the package and compile its kernels into the Warp cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default=None,
        help="Warp device to compile for; defaults to cuda:0 when available",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Parallel compile workers; 0 is serial. Defaults to 60%% of cores.",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    workers = default_workers() if args.jobs is None else args.jobs
    print(f"warp cache: {os.environ.get('WARP_CACHE_PATH', '<warp default>')}")

    started = time.perf_counter()
    import warp as wp

    wp.init()
    after_init = time.perf_counter()

    imported, failed = import_package(args.quiet)
    after_import = time.perf_counter()

    device = args.device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    print(
        f"imported {imported} modules ({len(failed)} skipped), compiling for "
        f"{device} with {workers or 'serial'} "
        f"worker{'' if workers == 1 else 's'} of {os.cpu_count()} cores"
    )

    # A couple of registered specialisations do not type-check when the whole
    # package is imported at once; Warp reports the failure and moves on, and the
    # affected kernels simply compile during the test run as they do today.
    # Warming is best-effort by design, so this must not fail the job.
    wp.force_load(device=device, max_workers=workers)
    finished = time.perf_counter()

    print(
        f"\ninit {after_init - started:6.1f}s"
        f" | import {after_import - after_init:6.1f}s"
        f" | compile {finished - after_import:6.1f}s"
        f" | total {finished - started:6.1f}s"
    )


if __name__ == "__main__":
    main()

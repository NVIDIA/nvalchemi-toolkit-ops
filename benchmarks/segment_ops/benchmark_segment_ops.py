#!/usr/bin/env python
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

"""Benchmark all segmented operations vs PyTorch equivalents.

Uses CUDA events for accurate GPU-only timing (no Python dispatch overhead).
Sweeps N x L (average segment length) to show behaviour across regimes.
Covers float32, float64, vec3f, and vec3d where applicable.

Configuration is loaded from ``benchmark_config.yaml``.

Usage::

    python -m benchmarks.segment_ops.benchmark_segment_ops [--config benchmark_config.yaml]
                                                            [--output-dir ./benchmark_results]
                                                            [--device cuda:0]
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np
import torch
import warp as wp
import yaml

from nvalchemiops.segment_ops import (
    segmented_add,
    segmented_component_sum,
    segmented_dot,
    segmented_matvec,
    segmented_max_norm,
    segmented_mul,
    segmented_sum,
)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get_gpu_sku() -> str:
    """Return a short GPU identifier for filenames."""
    try:
        name = torch.cuda.get_device_name(0)
        return name.lower().replace(" ", "_").replace("-", "_")
    except Exception:
        try:
            out = (
                subprocess.check_output(  # noqa: S603
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],  # noqa: S607
                    text=True,
                )
                .strip()
                .split("\n")[0]
            )
            return out.lower().replace(" ", "_").replace("-", "_")
        except Exception:
            return "unknown_gpu"


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _bench_cuda(setup, fn, torch_device, warmup, runs) -> float:
    """Return median GPU time (ms) over *runs* using CUDA events."""
    for _ in range(warmup):
        setup()
        fn()
    torch.cuda.synchronize(torch_device)

    stream = torch.cuda.current_stream(torch_device)
    times = []
    for _ in range(runs):
        setup()
        torch.cuda.synchronize(torch_device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        fn()
        end.record(stream)
        torch.cuda.synchronize(torch_device)
        times.append(start.elapsed_time(end))
    return float(np.median(times))


def _wp_vec_array(arr_np, wp_dtype, device):
    return wp.array([tuple(r) for r in arr_np], dtype=wp_dtype, device=device)


def _make_segments(N, M, rng):
    return np.sort(rng.integers(0, M, size=N).astype(np.int32))


def _noop():
    pass


def _print_header(name):
    header = (
        f"{'N':>10}  {'M':>7}  {'L':>9}  "
        f"{'warp ms':>9}  {'torch ms':>9}  {'speedup':>8}"
    )
    sep = "-" * len(header)
    print(f"\n{name}")
    print(sep)
    print(header)
    print(sep)


def _print_row(N, M, ms_wp, ms_torch):
    L = N / max(M, 1)
    speedup = ms_torch / ms_wp if ms_wp > 0 else float("inf")
    print(
        f"{N:>10}  {M:>7}  {L:>9.1f}  {ms_wp:>9.4f}  {ms_torch:>9.4f}  {speedup:>7.2f}x"
    )


# ---------------------------------------------------------------------------
# Dtype configuration
# ---------------------------------------------------------------------------

# (wp_dtype, torch_dtype, np_scalar_dtype, is_vec, label)
_SUM_DOT_DTYPE_VARIANTS = {
    "float32": (wp.float32, torch.float32, np.float32, False, "f32"),
    "float64": (wp.float64, torch.float64, np.float64, False, "f64"),
    "vec3f": (wp.vec3f, torch.float32, np.float32, True, "vec3f"),
    "vec3d": (wp.vec3d, torch.float64, np.float64, True, "vec3d"),
}


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def bench_segmented_sum(N, M, device, torch_device, rng, warmup, runs, dtypes=None):
    """segmented_sum vs torch.index_add_ for requested dtype variants."""
    if dtypes is None:
        dtypes = ["float32", "vec3f"]

    results = []
    for dtype_key in dtypes:
        if dtype_key not in _SUM_DOT_DTYPE_VARIANTS:
            continue
        wp_dt, torch_dt, np_dt, is_vec, label = _SUM_DOT_DTYPE_VARIANTS[dtype_key]

        if is_vec:
            x_np = rng.standard_normal((N, 3)).astype(np_dt)
            x = _wp_vec_array(x_np, wp_dt, device)
            t_x = torch.from_numpy(x_np).to(torch_device)
            out = wp.zeros(M, dtype=wp_dt, device=device)
            t_out = torch.zeros(M, 3, dtype=torch_dt, device=torch_device)
        else:
            x_np = rng.standard_normal(N).astype(np_dt)
            x = wp.array(x_np, dtype=wp_dt, device=device)
            t_x = torch.from_numpy(x_np).to(torch_device)
            out = wp.zeros(M, dtype=wp_dt, device=device)
            t_out = torch.zeros(M, dtype=torch_dt, device=torch_device)

        idx_np = _make_segments(N, M, rng)
        idx = wp.array(idx_np, device=device)
        t_idx = torch.from_numpy(idx_np.astype(np.int64)).to(torch_device)

        ms_wp = _bench_cuda(
            lambda: out.zero_(),
            lambda: segmented_sum(x, idx, out),
            torch_device,
            warmup,
            runs,
        )
        ms_t = _bench_cuda(
            lambda: t_out.zero_(),
            lambda: t_out.index_add_(0, t_idx, t_x),
            torch_device,
            warmup,
            runs,
        )
        _print_row(N, M, ms_wp, ms_t)

        results.append(("segmented_sum", label, N, M, ms_wp, ms_t))

    return results


def bench_segmented_component_sum(
    N, M, device, torch_device, rng, warmup, runs, **_kwargs
):
    """segmented_component_sum vs torch (sum components + index_add)."""
    x_np = rng.standard_normal((N, 3)).astype(np.float32)
    idx_np = _make_segments(N, M, rng)

    x = _wp_vec_array(x_np, wp.vec3f, device)
    idx = wp.array(idx_np, device=device)
    out = wp.zeros(M, dtype=wp.float32, device=device)

    t_x = torch.from_numpy(x_np).to(torch_device)
    t_idx = torch.from_numpy(idx_np.astype(np.int64)).to(torch_device)
    t_out = torch.zeros(M, dtype=torch.float32, device=torch_device)

    def _torch_component_sum():
        t_out.zero_()
        t_out.index_add_(0, t_idx, t_x.sum(dim=1))

    ms_wp = _bench_cuda(
        lambda: out.zero_(),
        lambda: segmented_component_sum(x, idx, out),
        torch_device,
        warmup,
        runs,
    )
    ms_t = _bench_cuda(_noop, _torch_component_sum, torch_device, warmup, runs)
    _print_row(N, M, ms_wp, ms_t)

    return [("segmented_component_sum", "vec3f", N, M, ms_wp, ms_t)]


def bench_segmented_dot(N, M, device, torch_device, rng, warmup, runs, dtypes=None):
    """segmented_dot (scalar + vec3) vs torch (multiply + index_add)."""
    if dtypes is None:
        dtypes = ["float32", "vec3f"]

    results = []
    for dtype_key in dtypes:
        if dtype_key not in _SUM_DOT_DTYPE_VARIANTS:
            continue
        wp_dt, torch_dt, np_dt, is_vec, label = _SUM_DOT_DTYPE_VARIANTS[dtype_key]

        idx_np = _make_segments(N, M, rng)
        idx = wp.array(idx_np, device=device)
        t_idx = torch.from_numpy(idx_np.astype(np.int64)).to(torch_device)

        if is_vec:
            x_np = rng.standard_normal((N, 3)).astype(np_dt)
            y_np = rng.standard_normal((N, 3)).astype(np_dt)
            x = _wp_vec_array(x_np, wp_dt, device)
            y = _wp_vec_array(y_np, wp_dt, device)
            scalar_wp = wp.float32 if np_dt == np.float32 else wp.float64
            out = wp.zeros(M, dtype=scalar_wp, device=device)
            t_x = torch.from_numpy(x_np).to(torch_device)
            t_y = torch.from_numpy(y_np).to(torch_device)
            t_out = torch.zeros(M, dtype=torch_dt, device=torch_device)

            def _torch_dot():
                t_out.zero_()
                t_out.index_add_(0, t_idx, (t_x * t_y).sum(dim=1))
        else:
            x_np = rng.standard_normal(N).astype(np_dt)
            y_np = rng.standard_normal(N).astype(np_dt)
            x = wp.array(x_np, dtype=wp_dt, device=device)
            y = wp.array(y_np, dtype=wp_dt, device=device)
            out = wp.zeros(M, dtype=wp_dt, device=device)
            t_x = torch.from_numpy(x_np).to(torch_device)
            t_y = torch.from_numpy(y_np).to(torch_device)
            t_out = torch.zeros(M, dtype=torch_dt, device=torch_device)

            def _torch_dot():
                t_out.zero_()
                t_out.index_add_(0, t_idx, t_x * t_y)

        ms_wp = _bench_cuda(
            lambda: out.zero_(),
            lambda: segmented_dot(x, y, idx, out),
            torch_device,
            warmup,
            runs,
        )
        ms_t = _bench_cuda(_noop, _torch_dot, torch_device, warmup, runs)
        _print_row(N, M, ms_wp, ms_t)

        results.append(("segmented_dot", label, N, M, ms_wp, ms_t))

    return results


def bench_segmented_max_norm(N, M, device, torch_device, rng, warmup, runs, **_kwargs):
    """segmented_max_norm vs torch (norm + scatter_reduce amax)."""
    x_np = rng.standard_normal((N, 3)).astype(np.float32)
    idx_np = _make_segments(N, M, rng)

    x = _wp_vec_array(x_np, wp.vec3f, device)
    idx = wp.array(idx_np, device=device)
    out = wp.zeros(M, dtype=wp.float32, device=device)

    t_x = torch.from_numpy(x_np).to(torch_device)
    t_idx = torch.from_numpy(idx_np.astype(np.int64)).to(torch_device)
    t_out = torch.zeros(M, dtype=torch.float32, device=torch_device)

    def _torch_max_norm():
        t_out.zero_()
        t_out.scatter_reduce_(0, t_idx, t_x.norm(dim=1), reduce="amax")

    ms_wp = _bench_cuda(
        lambda: out.zero_(),
        lambda: segmented_max_norm(x, idx, out),
        torch_device,
        warmup,
        runs,
    )
    ms_t = _bench_cuda(_noop, _torch_max_norm, torch_device, warmup, runs)
    _print_row(N, M, ms_wp, ms_t)

    return [("segmented_max_norm", "vec3f", N, M, ms_wp, ms_t)]


def bench_segmented_mul(N, M, device, torch_device, rng, warmup, runs, **_kwargs):
    """segmented_mul (f32 + vec3f*scalar) vs torch gather+mul."""
    results = []
    for label, wp_xt, wp_yt, np_dt, is_vec in [
        ("f32", wp.float32, wp.float32, np.float32, False),
        ("vec3f_scalar", wp.vec3f, wp.float32, np.float32, True),
    ]:
        idx_np = _make_segments(N, M, rng)
        idx = wp.array(idx_np, device=device)
        t_idx = torch.from_numpy(idx_np.astype(np.int64)).to(torch_device)

        if is_vec:
            x_np = rng.standard_normal((N, 3)).astype(np_dt)
            y_np = rng.standard_normal(M).astype(np_dt)
            x = _wp_vec_array(x_np, wp_xt, device)
            y = wp.array(y_np, dtype=wp_yt, device=device)
            out = wp.zeros(N, dtype=wp_xt, device=device)
            t_x = torch.from_numpy(x_np).to(torch_device)
            t_y = torch.from_numpy(y_np).to(torch_device)
            t_out = torch.zeros(N, 3, dtype=torch.float32, device=torch_device)

            def _torch_mul():
                t_out.copy_(t_x * t_y[t_idx].unsqueeze(1))
        else:
            x_np = rng.standard_normal(N).astype(np_dt)
            y_np = rng.standard_normal(M).astype(np_dt)
            x = wp.array(x_np, dtype=wp_xt, device=device)
            y = wp.array(y_np, dtype=wp_yt, device=device)
            out = wp.zeros(N, dtype=wp_xt, device=device)
            t_x = torch.from_numpy(x_np).to(torch_device)
            t_y = torch.from_numpy(y_np).to(torch_device)
            t_out = torch.zeros(N, dtype=torch.float32, device=torch_device)

            def _torch_mul():
                t_out.copy_(t_x * t_y[t_idx])

        ms_wp = _bench_cuda(
            _noop, lambda: segmented_mul(x, y, idx, out), torch_device, warmup, runs
        )
        ms_t = _bench_cuda(_noop, _torch_mul, torch_device, warmup, runs)
        _print_row(N, M, ms_wp, ms_t)

        results.append(("segmented_mul", label, N, M, ms_wp, ms_t))

    return results


def bench_segmented_add(N, M, device, torch_device, rng, warmup, runs, **_kwargs):
    """segmented_add (f32 + vec3f+scalar) vs torch gather+add."""
    results = []
    for label, wp_xt, wp_yt, np_dt, is_vec in [
        ("f32", wp.float32, wp.float32, np.float32, False),
        ("vec3f_scalar", wp.vec3f, wp.float32, np.float32, True),
    ]:
        idx_np = _make_segments(N, M, rng)
        idx = wp.array(idx_np, device=device)
        t_idx = torch.from_numpy(idx_np.astype(np.int64)).to(torch_device)

        if is_vec:
            x_np = rng.standard_normal((N, 3)).astype(np_dt)
            y_np = rng.standard_normal(M).astype(np_dt)
            x = _wp_vec_array(x_np, wp_xt, device)
            y = wp.array(y_np, dtype=wp_yt, device=device)
            out = wp.zeros(N, dtype=wp_xt, device=device)
            t_x = torch.from_numpy(x_np).to(torch_device)
            t_y = torch.from_numpy(y_np).to(torch_device)
            t_out = torch.zeros(N, 3, dtype=torch.float32, device=torch_device)

            def _torch_add():
                t_out.copy_(t_x + t_y[t_idx].unsqueeze(1))
        else:
            x_np = rng.standard_normal(N).astype(np_dt)
            y_np = rng.standard_normal(M).astype(np_dt)
            x = wp.array(x_np, dtype=wp_xt, device=device)
            y = wp.array(y_np, dtype=wp_yt, device=device)
            out = wp.zeros(N, dtype=wp_xt, device=device)
            t_x = torch.from_numpy(x_np).to(torch_device)
            t_y = torch.from_numpy(y_np).to(torch_device)
            t_out = torch.zeros(N, dtype=torch.float32, device=torch_device)

            def _torch_add():
                t_out.copy_(t_x + t_y[t_idx])

        ms_wp = _bench_cuda(
            _noop, lambda: segmented_add(x, y, idx, out), torch_device, warmup, runs
        )
        ms_t = _bench_cuda(_noop, _torch_add, torch_device, warmup, runs)
        _print_row(N, M, ms_wp, ms_t)

        results.append(("segmented_add", label, N, M, ms_wp, ms_t))

    return results


def bench_segmented_matvec(N, M, device, torch_device, rng, warmup, runs, **_kwargs):
    """segmented_matvec vs torch gather+matmul."""
    idx_np = _make_segments(N, M, rng)
    v_np = rng.standard_normal((N, 3)).astype(np.float32)
    m_np = rng.standard_normal((M, 3, 3)).astype(np.float32)

    v = _wp_vec_array(v_np, wp.vec3f, device)
    m_tuples = [tuple(tuple(row) for row in mat) for mat in m_np]
    m = wp.array(m_tuples, dtype=wp.mat33f, device=device)
    idx = wp.array(idx_np, device=device)
    out = wp.zeros(N, dtype=wp.vec3f, device=device)

    t_v = torch.from_numpy(v_np).to(torch_device)
    t_m = torch.from_numpy(m_np).to(torch_device)
    t_idx = torch.from_numpy(idx_np.astype(np.int64)).to(torch_device)
    t_out = torch.zeros(N, 3, dtype=torch.float32, device=torch_device)

    def _torch_matvec():
        gathered = t_m[t_idx]
        t_out.copy_(torch.einsum("nji,nj->ni", gathered, t_v))

    ms_wp = _bench_cuda(
        _noop, lambda: segmented_matvec(v, m, idx, out), torch_device, warmup, runs
    )
    ms_t = _bench_cuda(_noop, _torch_matvec, torch_device, warmup, runs)
    _print_row(N, M, ms_wp, ms_t)

    return [("segmented_matvec", "mat33f", N, M, ms_wp, ms_t)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Registry: (config_key, display_name, bench_fn, supports_dtypes_kwarg)
_BENCHMARKS = [
    ("segmented_sum", "segmented_sum", bench_segmented_sum, True),
    (
        "segmented_component_sum",
        "segmented_component_sum",
        bench_segmented_component_sum,
        False,
    ),
    ("segmented_dot", "segmented_dot", bench_segmented_dot, True),
    ("segmented_max_norm", "segmented_max_norm", bench_segmented_max_norm, False),
    ("segmented_mul", "segmented_mul", bench_segmented_mul, False),
    ("segmented_add", "segmented_add", bench_segmented_add, False),
    ("segmented_matvec", "segmented_matvec", bench_segmented_matvec, False),
]


def run_benchmarks(config: dict, output_dir: Path, device_str: str) -> None:
    """Run segment operations benchmarks."""
    params = config.get("parameters", {})
    warmup = params.get("warmup", 5)
    runs = params.get("runs", 50)

    sweep = config.get("sweep", {})
    n_values = sweep.get("total_elements", [1_000, 10_000, 100_000, 1_000_000])
    l_values = sweep.get("avg_segment_lengths", [10, 100, 1_000, 10_000])

    ops_config = config.get("operations", {})

    wp_device = wp.get_device(device_str)
    torch_device = torch.device(device_str)
    rng = np.random.default_rng(42)
    gpu_sku = _get_gpu_sku()

    print("Segment Operations Benchmark")
    print(f"Device: {wp_device.name}  (SM count: {wp_device.sm_count})")
    print(f"Timing: CUDA events, {warmup} warmup, median of {runs} runs")

    all_results = []

    for config_key, display_name, bench_fn, supports_dtypes in _BENCHMARKS:
        op_cfg = ops_config.get(config_key, {})
        if isinstance(op_cfg, dict) and not op_cfg.get("enabled", True):
            continue

        # Determine dtype variants for this operation
        kwargs = {}
        if supports_dtypes:
            op_dtypes = op_cfg.get("dtypes", None) if isinstance(op_cfg, dict) else None
            if op_dtypes is not None:
                kwargs["dtypes"] = op_dtypes

        # Build display name with dtype info
        dtype_info = ""
        if supports_dtypes and "dtypes" in kwargs:
            dtype_info = f" ({', '.join(kwargs['dtypes'])})"
        elif supports_dtypes:
            dtype_info = " (f32, vec3f)"

        _print_header(f"{display_name}{dtype_info}")
        for N in n_values:
            for L in l_values:
                if L > N:
                    continue
                M = max(N // L, 1)
                results = bench_fn(
                    N, M, wp_device, torch_device, rng, warmup, runs, **kwargs
                )
                all_results.extend(results)
            print()

    # Write CSV
    if all_results:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"segment_ops_benchmark_{gpu_sku}.csv"
        fieldnames = [
            "operation",
            "dtype",
            "total_elements",
            "num_segments",
            "avg_segment_length",
            "warp_median_ms",
            "torch_median_ms",
            "speedup",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for op, dtype_label, N, M, ms_wp, ms_t in all_results:
                L = N / max(M, 1)
                speedup = ms_t / ms_wp if ms_wp > 0 else float("inf")
                writer.writerow(
                    {
                        "operation": op,
                        "dtype": dtype_label,
                        "total_elements": N,
                        "num_segments": M,
                        "avg_segment_length": f"{L:.1f}",
                        "warp_median_ms": f"{ms_wp:.4f}",
                        "torch_median_ms": f"{ms_t:.4f}",
                        "speedup": f"{speedup:.2f}",
                    }
                )
        print(f"Wrote results to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark segmented ops vs PyTorch")
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for CSV files",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = _load_config(args.config)
    output_dir = Path(args.output_dir)
    run_benchmarks(config, output_dir, args.device)


if __name__ == "__main__":
    main()

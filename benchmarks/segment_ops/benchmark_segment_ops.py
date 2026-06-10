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
import time
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
from nvalchemiops.segment_ops_backward import (
    segmented_dot_backward,
    segmented_dot_double_backward,
    segmented_matvec_backward,
    segmented_matvec_double_backward,
    segmented_max_norm_backward,
    segmented_max_norm_double_backward,
    segmented_max_norm_forward_precompute,
    segmented_mul_backward,
    segmented_mul_double_backward,
    segmented_rms_norm_backward,
    segmented_rms_norm_double_backward,
    segmented_rms_norm_forward_precompute,
    segmented_sum_backward,
    segmented_sum_double_backward,
)

# Public Torch bindings (``torch.library`` custom ops). Aliased so they don't
# shadow the raw Warp launchers of the same name imported above.
from nvalchemiops.torch import segment_ops as torch_seg

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
    """Return median GPU time (ms) over *runs* using CUDA events.

    Binds Warp launches to torch's current stream via ``wp.ScopedStream`` so
    that the CUDA events recorded on that stream actually capture Warp work.
    Without the binding Warp uses its own default stream and the events
    under-report (or miss entirely) the timed kernels.  The scoped stream is a
    no-op for pure-torch ``fn`` bodies, so the wrapper is applied uniformly to
    keep the timed region identical between the Warp and torch comparison
    paths.
    """
    stream = torch.cuda.current_stream(torch_device)
    wp_stream = wp.stream_from_torch(stream)

    for _ in range(warmup):
        setup()
        with wp.ScopedStream(wp_stream):
            fn()
    torch.cuda.synchronize(torch_device)

    times = []
    for _ in range(runs):
        setup()
        torch.cuda.synchronize(torch_device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        with wp.ScopedStream(wp_stream):
            fn()
        end.record(stream)
        torch.cuda.synchronize(torch_device)
        times.append(start.elapsed_time(end))
    return float(np.median(times))


def _bench_cuda_graph(setup, fn, torch_device, warmup, runs, wp_device) -> float:
    """Return median time (ms) of replaying ``(setup, fn)`` from a captured CUDA graph.

    Captures ``setup(); fn()`` once into a Warp CUDA graph and times
    ``wp.capture_launch`` of the replay over *runs* iterations.  Returns
    ``None`` if capture fails (some op chains aren't capturable).

    Timing methodology
    ------------------
    The graph runs on Warp's default device stream, which is not necessarily
    the same as torch's current stream.  Using ``torch.cuda.Event`` on
    torch's stream would therefore record only host-side issue time —
    correct only if the two libraries happen to share a stream handle, which
    is fragile to depend on.

    Instead we time with ``time.perf_counter`` bracketed by
    ``wp.synchronize_device``: the leading sync drains any prior work on
    Warp's stream, and the trailing sync blocks until the graph has fully
    executed.  The wall-clock delta between the two ``perf_counter`` calls
    is therefore the graph's GPU execution time, by construction, regardless
    of stream aliasing.  The trade-off is ~1-2 µs of host overhead per
    sample from the syncs themselves — invisible against ms-scale graph
    times and acceptable against µs-scale ones because we report the
    median over ``runs`` samples.
    """
    # Pre-capture warmup so all Warp modules are loaded and any first-call
    # work (overload compilation, mempool init) happens outside the capture.
    for _ in range(3):
        setup()
        fn()
    wp.synchronize_device(wp_device)

    try:
        with wp.ScopedCapture(device=wp_device) as cap:
            setup()
            fn()
    except Exception:
        return None

    for _ in range(warmup):
        wp.capture_launch(cap.graph)
    wp.synchronize_device(wp_device)

    times = []
    for _ in range(runs):
        wp.synchronize_device(wp_device)
        start = time.perf_counter()
        wp.capture_launch(cap.graph)
        wp.synchronize_device(wp_device)
        # perf_counter returns seconds; convert to ms to match the rest of
        # the benchmark output (CUDA-event elapsed_time is already ms).
        times.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(times))


def _bench_warp_both(setup, fn, torch_device, warmup, runs):
    """Time ``fn`` on the warp side both eagerly and via a captured CUDA graph.

    Returns ``(eager_ms, graph_ms)``.  ``graph_ms`` is ``None`` if capture failed
    (e.g. the op chain mutates host state or uses an unsupported API).  The
    Warp device is derived from ``torch_device`` so existing call sites don't
    need a signature change beyond renaming ``_bench_cuda`` to this helper.
    """
    wp_device = wp.get_device(str(torch_device))
    eager_ms = _bench_cuda(setup, fn, torch_device, warmup, runs)
    graph_ms = _bench_cuda_graph(setup, fn, torch_device, warmup, runs, wp_device)
    return eager_ms, graph_ms


def _wp_vec_array(arr_np, wp_dtype, device):
    return wp.array([tuple(r) for r in arr_np], dtype=wp_dtype, device=device)


def _make_segments(N, M, rng):
    return np.sort(rng.integers(0, M, size=N).astype(np.int32))


def _noop():
    pass


def _print_header(name, with_graph=False):
    if with_graph:
        header = (
            f"{'N':>10}  {'M':>7}  {'L':>9}  "
            f"{'warp ms':>9}  {'wp_graph ms':>11}  {'torch ms':>9}  "
            f"{'eager spd':>9}  {'graph spd':>9}"
        )
    else:
        header = (
            f"{'N':>10}  {'M':>7}  {'L':>9}  "
            f"{'warp ms':>9}  {'torch ms':>9}  {'speedup':>8}"
        )
    sep = "-" * len(header)
    print(f"\n{name}")
    print(sep)
    print(header)
    print(sep)


def _print_row(N, M, ms_wp, ms_torch, ms_wp_graph=None):
    L = N / max(M, 1)
    speedup = ms_torch / ms_wp if ms_wp > 0 else float("inf")
    if ms_wp_graph is None:
        print(
            f"{N:>10}  {M:>7}  {L:>9.1f}  {ms_wp:>9.4f}  {ms_torch:>9.4f}  "
            f"{speedup:>7.2f}x"
        )
    else:
        graph_speedup = ms_torch / ms_wp_graph if ms_wp_graph > 0 else float("inf")
        print(
            f"{N:>10}  {M:>7}  {L:>9.1f}  "
            f"{ms_wp:>9.4f}  {ms_wp_graph:>11.4f}  {ms_torch:>9.4f}  "
            f"{speedup:>8.2f}x  {graph_speedup:>8.2f}x"
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

        ms_wp, ms_wp_graph = _bench_warp_both(
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
        _print_row(N, M, ms_wp, ms_t, ms_wp_graph)

        results.append(("segmented_sum", label, N, M, ms_wp, ms_t, ms_wp_graph))

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

    ms_wp, ms_wp_graph = _bench_warp_both(
        lambda: out.zero_(),
        lambda: segmented_component_sum(x, idx, out),
        torch_device,
        warmup,
        runs,
    )
    ms_t = _bench_cuda(_noop, _torch_component_sum, torch_device, warmup, runs)
    _print_row(N, M, ms_wp, ms_t, ms_wp_graph)

    return [("segmented_component_sum", "vec3f", N, M, ms_wp, ms_t, ms_wp_graph)]


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

        ms_wp, ms_wp_graph = _bench_warp_both(
            lambda: out.zero_(),
            lambda: segmented_dot(x, y, idx, out),
            torch_device,
            warmup,
            runs,
        )
        ms_t = _bench_cuda(_noop, _torch_dot, torch_device, warmup, runs)
        _print_row(N, M, ms_wp, ms_t, ms_wp_graph)

        results.append(("segmented_dot", label, N, M, ms_wp, ms_t, ms_wp_graph))

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

    ms_wp, ms_wp_graph = _bench_warp_both(
        lambda: out.zero_(),
        lambda: segmented_max_norm(x, idx, out),
        torch_device,
        warmup,
        runs,
    )
    ms_t = _bench_cuda(_noop, _torch_max_norm, torch_device, warmup, runs)
    _print_row(N, M, ms_wp, ms_t, ms_wp_graph)

    return [("segmented_max_norm", "vec3f", N, M, ms_wp, ms_t, ms_wp_graph)]


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

        ms_wp, ms_wp_graph = _bench_warp_both(
            _noop, lambda: segmented_mul(x, y, idx, out), torch_device, warmup, runs
        )
        ms_t = _bench_cuda(_noop, _torch_mul, torch_device, warmup, runs)
        _print_row(N, M, ms_wp, ms_t, ms_wp_graph)

        results.append(("segmented_mul", label, N, M, ms_wp, ms_t, ms_wp_graph))

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

        ms_wp, ms_wp_graph = _bench_warp_both(
            _noop, lambda: segmented_add(x, y, idx, out), torch_device, warmup, runs
        )
        ms_t = _bench_cuda(_noop, _torch_add, torch_device, warmup, runs)
        _print_row(N, M, ms_wp, ms_t, ms_wp_graph)

        results.append(("segmented_add", label, N, M, ms_wp, ms_t, ms_wp_graph))

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

    ms_wp, ms_wp_graph = _bench_warp_both(
        _noop, lambda: segmented_matvec(v, m, idx, out), torch_device, warmup, runs
    )
    ms_t = _bench_cuda(_noop, _torch_matvec, torch_device, warmup, runs)
    _print_row(N, M, ms_wp, ms_t, ms_wp_graph)

    return [("segmented_matvec", "mat33f", N, M, ms_wp, ms_t, ms_wp_graph)]


# ---------------------------------------------------------------------------
# Backward benchmark runners (PR 1)
# ---------------------------------------------------------------------------


def _setup_sum_arrays(N, M, device, torch_device, rng, is_vec, wp_dt, torch_dt, np_dt):
    idx_np = _make_segments(N, M, rng)
    idx = wp.array(idx_np, device=device)
    t_idx_long = torch.from_numpy(idx_np.astype(np.int64)).to(torch_device)
    if is_vec:
        g_out_np = rng.standard_normal((M, 3)).astype(np_dt)
        gg_x_np = rng.standard_normal((N, 3)).astype(np_dt)
        g_out = _wp_vec_array(g_out_np, wp_dt, device)
        gg_x = _wp_vec_array(gg_x_np, wp_dt, device)
        grad_x = wp.zeros(N, dtype=wp_dt, device=device)
        grad_g_out = wp.zeros(M, dtype=wp_dt, device=device)
        t_g_out = torch.from_numpy(g_out_np).to(torch_device)
        t_gg_x = torch.from_numpy(gg_x_np).to(torch_device)
        t_grad_g_out = torch.zeros(M, 3, dtype=torch_dt, device=torch_device)
    else:
        g_out_np = rng.standard_normal(M).astype(np_dt)
        gg_x_np = rng.standard_normal(N).astype(np_dt)
        g_out = wp.array(g_out_np, dtype=wp_dt, device=device)
        gg_x = wp.array(gg_x_np, dtype=wp_dt, device=device)
        grad_x = wp.zeros(N, dtype=wp_dt, device=device)
        grad_g_out = wp.zeros(M, dtype=wp_dt, device=device)
        t_g_out = torch.from_numpy(g_out_np).to(torch_device)
        t_gg_x = torch.from_numpy(gg_x_np).to(torch_device)
        t_grad_g_out = torch.zeros(M, dtype=torch_dt, device=torch_device)
    return (
        idx,
        g_out,
        gg_x,
        grad_x,
        grad_g_out,
        t_idx_long,
        t_g_out,
        t_gg_x,
        t_grad_g_out,
    )


def bench_segmented_sum_bwd(N, M, device, torch_device, rng, warmup, runs, **_kwargs):
    """1st-order backward of segmented_sum: warp gather vs torch ``g_out[idx]``."""
    results = []
    for dtype_key in ["float32", "vec3f"]:
        wp_dt, torch_dt, np_dt, is_vec, label = _SUM_DOT_DTYPE_VARIANTS[dtype_key]
        (
            idx,
            g_out,
            _gg_x,
            grad_x,
            _grad_g_out,
            t_idx,
            t_g_out,
            _t_gg_x,
            _t_gd,
        ) = _setup_sum_arrays(
            N, M, device, torch_device, rng, is_vec, wp_dt, torch_dt, np_dt
        )

        ms_wp, ms_wp_graph = _bench_warp_both(
            _noop,
            lambda: segmented_sum_backward(g_out, idx, grad_x),
            torch_device,
            warmup,
            runs,
        )
        ms_t = _bench_cuda(
            _noop,
            lambda: t_g_out[t_idx],
            torch_device,
            warmup,
            runs,
        )
        _print_row(N, M, ms_wp, ms_t, ms_wp_graph)
        results.append(
            ("segmented_sum_backward", label, N, M, ms_wp, ms_t, ms_wp_graph)
        )
    return results


def bench_segmented_sum_dbl(N, M, device, torch_device, rng, warmup, runs, **_kwargs):
    """Double-backward of segmented_sum: warp scatter-sum vs torch ``index_add_``."""
    results = []
    for dtype_key in ["float32", "vec3f"]:
        wp_dt, torch_dt, np_dt, is_vec, label = _SUM_DOT_DTYPE_VARIANTS[dtype_key]
        (
            idx,
            _g_out,
            gg_x,
            _grad_x,
            grad_g_out,
            t_idx,
            _t_g_out,
            t_gg_x,
            t_grad_g_out,
        ) = _setup_sum_arrays(
            N, M, device, torch_device, rng, is_vec, wp_dt, torch_dt, np_dt
        )

        ms_wp, ms_wp_graph = _bench_warp_both(
            _noop,
            lambda: segmented_sum_double_backward(gg_x, idx, M, grad_g_out),
            torch_device,
            warmup,
            runs,
        )

        def _t_dbl():
            t_grad_g_out.zero_()
            t_grad_g_out.index_add_(0, t_idx, t_gg_x)

        ms_t = _bench_cuda(_noop, _t_dbl, torch_device, warmup, runs)
        _print_row(N, M, ms_wp, ms_t, ms_wp_graph)
        results.append(
            ("segmented_sum_double_backward", label, N, M, ms_wp, ms_t, ms_wp_graph)
        )
    return results


def _setup_dot_arrays(N, M, device, torch_device, rng, is_vec, wp_dt, torch_dt, np_dt):
    scalar_wp = wp.float32 if np_dt == np.float32 else wp.float64
    idx_np = _make_segments(N, M, rng)
    idx = wp.array(idx_np, device=device)
    t_idx = torch.from_numpy(idx_np.astype(np.int64)).to(torch_device)
    if is_vec:
        x_np = rng.standard_normal((N, 3)).astype(np_dt)
        y_np = rng.standard_normal((N, 3)).astype(np_dt)
        gg_gx_np = rng.standard_normal((N, 3)).astype(np_dt)
        gg_gy_np = rng.standard_normal((N, 3)).astype(np_dt)
        x = _wp_vec_array(x_np, wp_dt, device)
        y = _wp_vec_array(y_np, wp_dt, device)
        gg_gx = _wp_vec_array(gg_gx_np, wp_dt, device)
        gg_gy = _wp_vec_array(gg_gy_np, wp_dt, device)
        t_x = torch.from_numpy(x_np).to(torch_device)
        t_y = torch.from_numpy(y_np).to(torch_device)
    else:
        x_np = rng.standard_normal(N).astype(np_dt)
        y_np = rng.standard_normal(N).astype(np_dt)
        gg_gx_np = rng.standard_normal(N).astype(np_dt)
        gg_gy_np = rng.standard_normal(N).astype(np_dt)
        x = wp.array(x_np, dtype=wp_dt, device=device)
        y = wp.array(y_np, dtype=wp_dt, device=device)
        gg_gx = wp.array(gg_gx_np, dtype=wp_dt, device=device)
        gg_gy = wp.array(gg_gy_np, dtype=wp_dt, device=device)
        t_x = torch.from_numpy(x_np).to(torch_device)
        t_y = torch.from_numpy(y_np).to(torch_device)

    # The Warp and Torch cotangents must come from the same NumPy buffer so
    # the two paths benchmark equivalent math.  Materializing via
    # ``wp.zeros_like`` + ``torch.randn_like`` (the previous setup) had the
    # Warp side seeing all-zeros and the Torch side seeing random data,
    # making the reported speedup meaningless.
    t_gg_x = torch.from_numpy(gg_gx_np).to(torch_device)
    t_gg_y = torch.from_numpy(gg_gy_np).to(torch_device)

    g_out = wp.array(
        rng.standard_normal(M).astype(np_dt), dtype=scalar_wp, device=device
    )
    t_g_out = torch.from_numpy(np.asarray(g_out.numpy())).to(torch_device)
    grad_x = wp.zeros(N, dtype=wp_dt, device=device)
    grad_y = wp.zeros(N, dtype=wp_dt, device=device)
    grad_g_out = wp.zeros(M, dtype=scalar_wp, device=device)
    grad_x_ex = wp.zeros(N, dtype=wp_dt, device=device)
    grad_y_ex = wp.zeros(N, dtype=wp_dt, device=device)
    t_grad_g_out = torch.zeros(M, dtype=torch_dt, device=torch_device)
    return (
        idx,
        x,
        y,
        g_out,
        grad_x,
        grad_y,
        gg_gx,
        gg_gy,
        grad_g_out,
        grad_x_ex,
        grad_y_ex,
        t_idx,
        t_x,
        t_y,
        t_g_out,
        t_grad_g_out,
        t_gg_x,
        t_gg_y,
        is_vec,
    )


def bench_segmented_dot_bwd(N, M, device, torch_device, rng, warmup, runs, **_kwargs):
    """1st-order backward of segmented_dot vs torch broadcast multiply."""
    results = []
    for dtype_key in ["float32", "vec3f"]:
        wp_dt, torch_dt, np_dt, is_vec, label = _SUM_DOT_DTYPE_VARIANTS[dtype_key]
        (
            idx,
            x,
            y,
            g_out,
            grad_x,
            grad_y,
            *_rest,
            t_idx,
            t_x,
            t_y,
            t_g_out,
            _t_gd,
            _t_gg_x,
            _t_gg_y,
            _is_vec,
        ) = _setup_dot_arrays(
            N, M, device, torch_device, rng, is_vec, wp_dt, torch_dt, np_dt
        )

        ms_wp, ms_wp_graph = _bench_warp_both(
            _noop,
            lambda: segmented_dot_backward(g_out, x, y, idx, grad_x, grad_y),
            torch_device,
            warmup,
            runs,
        )

        if is_vec:

            def _t_bwd():
                gx = t_g_out[t_idx, None] * t_y
                gy = t_g_out[t_idx, None] * t_x
                return gx, gy
        else:

            def _t_bwd():
                gx = t_g_out[t_idx] * t_y
                gy = t_g_out[t_idx] * t_x
                return gx, gy

        ms_t = _bench_cuda(_noop, _t_bwd, torch_device, warmup, runs)
        _print_row(N, M, ms_wp, ms_t, ms_wp_graph)
        results.append(
            ("segmented_dot_backward", label, N, M, ms_wp, ms_t, ms_wp_graph)
        )
    return results


def bench_segmented_dot_dbl(N, M, device, torch_device, rng, warmup, runs, **_kwargs):
    """Double-backward of segmented_dot vs torch (broadcast + index_add)."""
    results = []
    for dtype_key in ["float32", "vec3f"]:
        wp_dt, torch_dt, np_dt, is_vec, label = _SUM_DOT_DTYPE_VARIANTS[dtype_key]
        (
            idx,
            x,
            y,
            g_out,
            _gx,
            _gy,
            gg_gx,
            gg_gy,
            grad_g_out,
            grad_x_ex,
            grad_y_ex,
            t_idx,
            t_x,
            t_y,
            t_g_out,
            t_grad_g_out,
            t_gg_x,
            t_gg_y,
            _is_vec,
        ) = _setup_dot_arrays(
            N, M, device, torch_device, rng, is_vec, wp_dt, torch_dt, np_dt
        )

        ms_wp, ms_wp_graph = _bench_warp_both(
            _noop,
            lambda: segmented_dot_double_backward(
                gg_gx, gg_gy, g_out, x, y, idx, M, grad_g_out, grad_x_ex, grad_y_ex
            ),
            torch_device,
            warmup,
            runs,
        )

        if is_vec:

            def _t_dbl():
                t_grad_g_out.zero_()
                t_grad_g_out.index_add_(0, t_idx, (t_gg_x * t_y).sum(-1))
                t_grad_g_out.index_add_(0, t_idx, (t_gg_y * t_x).sum(-1))
                _gx_ex = t_g_out[t_idx, None] * t_gg_y
                _gy_ex = t_g_out[t_idx, None] * t_gg_x
                return _gx_ex, _gy_ex
        else:

            def _t_dbl():
                t_grad_g_out.zero_()
                t_grad_g_out.index_add_(0, t_idx, t_gg_x * t_y)
                t_grad_g_out.index_add_(0, t_idx, t_gg_y * t_x)
                return t_g_out[t_idx] * t_gg_y, t_g_out[t_idx] * t_gg_x

        ms_t = _bench_cuda(_noop, _t_dbl, torch_device, warmup, runs)
        _print_row(N, M, ms_wp, ms_t, ms_wp_graph)
        results.append(
            ("segmented_dot_double_backward", label, N, M, ms_wp, ms_t, ms_wp_graph)
        )
    return results


def _setup_mul_arrays(N, M, device, torch_device, rng, is_vec, np_dt):
    torch_dt = torch.float32 if np_dt == np.float32 else torch.float64
    wp_xt = (
        wp.vec3f
        if (is_vec and np_dt == np.float32)
        else (
            wp.vec3d if is_vec else (wp.float32 if np_dt == np.float32 else wp.float64)
        )
    )
    wp_yt = wp.float32 if np_dt == np.float32 else wp.float64
    idx_np = _make_segments(N, M, rng)
    idx = wp.array(idx_np, device=device)
    t_idx = torch.from_numpy(idx_np.astype(np.int64)).to(torch_device)
    if is_vec:
        x_np = rng.standard_normal((N, 3)).astype(np_dt)
    else:
        x_np = rng.standard_normal(N).astype(np_dt)
    y_np = rng.standard_normal(M).astype(np_dt)
    g_out_np = rng.standard_normal(x_np.shape).astype(np_dt)

    # Generate ``gg_gx`` / ``gg_gy`` from shared NumPy buffers so the Warp
    # and Torch double-backward paths benchmark equivalent math.  The previous
    # setup created ``gg_gx = wp.zeros_like(x)`` (zeros) on the Warp side and
    # ``torch.randn_like(t_x)`` on the Torch side — different inputs, so the
    # reported speedup was meaningless.
    gg_gx_np = rng.standard_normal(x_np.shape).astype(np_dt)
    gg_gy_np = rng.standard_normal(M).astype(np_dt)

    if is_vec:
        x = _wp_vec_array(x_np, wp_xt, device)
        g_out = _wp_vec_array(g_out_np, wp_xt, device)
        gg_gx = _wp_vec_array(gg_gx_np, wp_xt, device)
    else:
        x = wp.array(x_np, dtype=wp_xt, device=device)
        g_out = wp.array(g_out_np, dtype=wp_xt, device=device)
        gg_gx = wp.array(gg_gx_np, dtype=wp_xt, device=device)
    y = wp.array(y_np, dtype=wp_yt, device=device)
    gg_gy = wp.array(gg_gy_np, dtype=wp_yt, device=device)
    grad_x = wp.zeros_like(x)
    grad_y = wp.zeros(M, dtype=wp_yt, device=device)
    grad_g_out = wp.zeros_like(x)
    grad_x_ex = wp.zeros_like(x)
    grad_y_ex = wp.zeros(M, dtype=wp_yt, device=device)
    t_x = torch.from_numpy(x_np).to(torch_device)
    t_y = torch.from_numpy(y_np).to(torch_device)
    t_g_out = torch.from_numpy(g_out_np).to(torch_device)
    t_gg_gx = torch.from_numpy(gg_gx_np).to(torch_device)
    t_gg_gy = torch.from_numpy(gg_gy_np).to(torch_device)
    t_grad_y = torch.zeros(M, dtype=torch_dt, device=torch_device)
    t_grad_g_out = torch.zeros_like(t_x)
    return (
        idx,
        x,
        y,
        g_out,
        grad_x,
        grad_y,
        gg_gx,
        gg_gy,
        grad_g_out,
        grad_x_ex,
        grad_y_ex,
        t_idx,
        t_x,
        t_y,
        t_g_out,
        t_grad_y,
        t_grad_g_out,
        t_gg_gx,
        t_gg_gy,
        is_vec,
    )


def bench_segmented_mul_bwd(N, M, device, torch_device, rng, warmup, runs, **_kwargs):
    """1st-order backward of segmented_mul vs torch broadcast/index_add."""
    results = []
    for label, np_dt, is_vec in [
        ("f32", np.float32, False),
        ("vec3f_scalar", np.float32, True),
    ]:
        (
            idx,
            x,
            y,
            g_out,
            grad_x,
            grad_y,
            *_rest,
            t_idx,
            t_x,
            t_y,
            t_g_out,
            t_grad_y,
            _t_gd,
            _t_gg_gx,
            _t_gg_gy,
            _is_vec,
        ) = _setup_mul_arrays(N, M, device, torch_device, rng, is_vec, np_dt)

        ms_wp, ms_wp_graph = _bench_warp_both(
            _noop,
            lambda: segmented_mul_backward(g_out, x, y, idx, M, grad_x, grad_y),
            torch_device,
            warmup,
            runs,
        )

        if is_vec:

            def _t_bwd():
                gx = t_g_out * t_y[t_idx, None]
                t_grad_y.zero_()
                t_grad_y.index_add_(0, t_idx, (t_g_out * t_x).sum(-1))
                return gx, t_grad_y
        else:

            def _t_bwd():
                gx = t_g_out * t_y[t_idx]
                t_grad_y.zero_()
                t_grad_y.index_add_(0, t_idx, t_g_out * t_x)
                return gx, t_grad_y

        ms_t = _bench_cuda(_noop, _t_bwd, torch_device, warmup, runs)
        _print_row(N, M, ms_wp, ms_t, ms_wp_graph)
        results.append(
            ("segmented_mul_backward", label, N, M, ms_wp, ms_t, ms_wp_graph)
        )
    return results


def bench_segmented_mul_dbl(N, M, device, torch_device, rng, warmup, runs, **_kwargs):
    """Double-backward of segmented_mul vs torch."""
    results = []
    for label, np_dt, is_vec in [
        ("f32", np.float32, False),
        ("vec3f_scalar", np.float32, True),
    ]:
        (
            idx,
            x,
            y,
            g_out,
            _gx,
            _gy,
            gg_gx,
            gg_gy,
            grad_g_out,
            grad_x_ex,
            grad_y_ex,
            t_idx,
            t_x,
            t_y,
            t_g_out,
            t_grad_y,
            t_grad_g_out,
            t_gg_gx,
            t_gg_gy,
            _is_vec,
        ) = _setup_mul_arrays(N, M, device, torch_device, rng, is_vec, np_dt)

        ms_wp, ms_wp_graph = _bench_warp_both(
            _noop,
            lambda: segmented_mul_double_backward(
                gg_gx, gg_gy, g_out, x, y, idx, grad_g_out, grad_x_ex, grad_y_ex
            ),
            torch_device,
            warmup,
            runs,
        )

        if is_vec:

            def _t_dbl():
                _g_o = t_gg_gx * t_y[t_idx, None] + t_gg_gy[t_idx, None] * t_x
                _g_x = t_gg_gy[t_idx, None] * t_g_out
                t_grad_y.zero_()
                t_grad_y.index_add_(0, t_idx, (t_gg_gx * t_g_out).sum(-1))
                return _g_o, _g_x, t_grad_y
        else:

            def _t_dbl():
                _g_o = t_gg_gx * t_y[t_idx] + t_gg_gy[t_idx] * t_x
                _g_x = t_gg_gy[t_idx] * t_g_out
                t_grad_y.zero_()
                t_grad_y.index_add_(0, t_idx, t_gg_gx * t_g_out)
                return _g_o, _g_x, t_grad_y

        ms_t = _bench_cuda(_noop, _t_dbl, torch_device, warmup, runs)
        _print_row(N, M, ms_wp, ms_t, ms_wp_graph)
        results.append(
            ("segmented_mul_double_backward", label, N, M, ms_wp, ms_t, ms_wp_graph)
        )
    return results


def _setup_matvec_arrays(N, M, device, torch_device, rng):
    idx_np = _make_segments(N, M, rng)
    v_np = rng.standard_normal((N, 3)).astype(np.float32)
    m_np = rng.standard_normal((M, 3, 3)).astype(np.float32)
    g_out_np = rng.standard_normal((N, 3)).astype(np.float32)
    gg_gv_np = rng.standard_normal((N, 3)).astype(np.float32)
    gg_gM_np = rng.standard_normal((M, 3, 3)).astype(np.float32)

    v = _wp_vec_array(v_np, wp.vec3f, device)
    m = wp.array(
        [tuple(tuple(r) for r in mat) for mat in m_np],
        dtype=wp.mat33f,
        device=device,
    )
    g_out = _wp_vec_array(g_out_np, wp.vec3f, device)
    gg_gv = _wp_vec_array(gg_gv_np, wp.vec3f, device)
    gg_gM = wp.array(
        [tuple(tuple(r) for r in mat) for mat in gg_gM_np],
        dtype=wp.mat33f,
        device=device,
    )
    idx = wp.array(idx_np, device=device)
    grad_v = wp.zeros(N, dtype=wp.vec3f, device=device)
    grad_M = wp.zeros(M, dtype=wp.mat33f, device=device)
    grad_g_out = wp.zeros(N, dtype=wp.vec3f, device=device)
    grad_v_ex = wp.zeros(N, dtype=wp.vec3f, device=device)
    grad_M_ex = wp.zeros(M, dtype=wp.mat33f, device=device)

    t_v = torch.from_numpy(v_np).to(torch_device)
    t_m = torch.from_numpy(m_np).to(torch_device)
    t_g_out = torch.from_numpy(g_out_np).to(torch_device)
    t_gg_gv = torch.from_numpy(gg_gv_np).to(torch_device)
    t_gg_gM = torch.from_numpy(gg_gM_np).to(torch_device)
    t_idx = torch.from_numpy(idx_np.astype(np.int64)).to(torch_device)
    t_grad_M = torch.zeros(M, 3, 3, dtype=torch.float32, device=torch_device)
    return (
        idx,
        v,
        m,
        g_out,
        gg_gv,
        gg_gM,
        grad_v,
        grad_M,
        grad_g_out,
        grad_v_ex,
        grad_M_ex,
        t_idx,
        t_v,
        t_m,
        t_g_out,
        t_gg_gv,
        t_gg_gM,
        t_grad_M,
    )


def bench_segmented_matvec_bwd(
    N, M, device, torch_device, rng, warmup, runs, **_kwargs
):
    """1st-order backward of segmented_matvec vs torch (gather + bmm + index_add)."""
    (
        idx,
        v,
        m,
        g_out,
        _gg_gv,
        _gg_gM,
        grad_v,
        grad_M,
        *_rest,
        t_idx,
        t_v,
        t_m,
        t_g_out,
        _t_gg_gv,
        _t_gg_gM,
        t_grad_M,
    ) = _setup_matvec_arrays(N, M, device, torch_device, rng)

    ms_wp, ms_wp_graph = _bench_warp_both(
        _noop,
        lambda: segmented_matvec_backward(g_out, v, m, idx, grad_v, grad_M),
        torch_device,
        warmup,
        runs,
    )

    def _t_bwd():
        # grad_v[i] = M[s] @ g_out[i]
        gv = (t_m[t_idx] @ t_g_out[..., None]).squeeze(-1)
        # grad_M[s] = sum_i outer(v[i], g_out[i])
        outer = t_v[..., None] * t_g_out[:, None, :]
        t_grad_M.zero_()
        t_grad_M.index_add_(0, t_idx, outer)
        return gv, t_grad_M

    ms_t = _bench_cuda(_noop, _t_bwd, torch_device, warmup, runs)
    _print_row(N, M, ms_wp, ms_t, ms_wp_graph)
    return [("segmented_matvec_backward", "mat33f", N, M, ms_wp, ms_t, ms_wp_graph)]


def bench_segmented_matvec_dbl(
    N, M, device, torch_device, rng, warmup, runs, **_kwargs
):
    """Double-backward of segmented_matvec vs torch (mirrors paired-bmm structure)."""
    (
        idx,
        v,
        m,
        g_out,
        gg_gv,
        gg_gM,
        _gv,
        _gM,
        grad_g_out,
        grad_v_ex,
        grad_M_ex,
        t_idx,
        t_v,
        t_m,
        t_g_out,
        t_gg_gv,
        t_gg_gM,
        t_grad_M,
    ) = _setup_matvec_arrays(N, M, device, torch_device, rng)

    ms_wp, ms_wp_graph = _bench_warp_both(
        _noop,
        lambda: segmented_matvec_double_backward(
            gg_gv, gg_gM, g_out, v, m, idx, grad_g_out, grad_v_ex, grad_M_ex
        ),
        torch_device,
        warmup,
        runs,
    )

    def _t_dbl():
        # grad_g_out[i] = M[s]^T @ gg_gv[i] + gg_gM[s]^T @ v[i]
        gg = (t_m[t_idx].mT @ t_gg_gv[..., None]).squeeze(-1)
        gg = gg + (t_gg_gM[t_idx].mT @ t_v[..., None]).squeeze(-1)
        # grad_v_extra[i] = gg_gM[s] @ g_out[i]
        gv_ex = (t_gg_gM[t_idx] @ t_g_out[..., None]).squeeze(-1)
        # grad_M_extra[s] = sum_i outer(gg_gv[i], g_out[i])
        outer = t_gg_gv[..., None] * t_g_out[:, None, :]
        t_grad_M.zero_()
        t_grad_M.index_add_(0, t_idx, outer)
        return gg, gv_ex, t_grad_M

    ms_t = _bench_cuda(_noop, _t_dbl, torch_device, warmup, runs)
    _print_row(N, M, ms_wp, ms_t, ms_wp_graph)
    return [
        ("segmented_matvec_double_backward", "mat33f", N, M, ms_wp, ms_t, ms_wp_graph)
    ]


def _setup_max_norm_arrays(N, M, device, torch_device, rng):
    x_np = rng.standard_normal((N, 3)).astype(np.float32)
    idx_np = _make_segments(N, M, rng)
    g_out_np = rng.standard_normal(M).astype(np.float32)
    gg_gx_np = rng.standard_normal((N, 3)).astype(np.float32)

    x = _wp_vec_array(x_np, wp.vec3f, device)
    idx = wp.array(idx_np, device=device)
    out = wp.zeros(M, dtype=wp.float32, device=device)
    argmax_idx = wp.zeros(M, dtype=wp.int32, device=device)
    g_out = wp.array(g_out_np, dtype=wp.float32, device=device)
    gg_gx = _wp_vec_array(gg_gx_np, wp.vec3f, device)
    grad_x = wp.zeros(N, dtype=wp.vec3f, device=device)
    grad_x_ex = wp.zeros(N, dtype=wp.vec3f, device=device)
    grad_g_out = wp.zeros(M, dtype=wp.float32, device=device)
    # Populate argmax via precompute fwd before any bwd benchmark.
    segmented_max_norm_forward_precompute(x, idx, out, argmax_idx)

    t_x = torch.from_numpy(x_np).to(torch_device)
    t_idx = torch.from_numpy(idx_np.astype(np.int64)).to(torch_device)
    t_g_out = torch.from_numpy(g_out_np).to(torch_device)
    t_gg_gx = torch.from_numpy(gg_gx_np).to(torch_device)
    t_out = torch.from_numpy(out.numpy()).to(torch_device)
    t_argmax = torch.from_numpy(argmax_idx.numpy().astype(np.int64)).to(torch_device)
    return (
        idx,
        x,
        out,
        argmax_idx,
        g_out,
        gg_gx,
        grad_x,
        grad_x_ex,
        grad_g_out,
        t_idx,
        t_x,
        t_out,
        t_argmax,
        t_g_out,
        t_gg_gx,
    )


def bench_segmented_max_norm_bwd(
    N, M, device, torch_device, rng, warmup, runs, **_kwargs
):
    """1st-order backward of segmented_max_norm vs torch (scatter from argmax)."""
    (
        idx,
        x,
        _out,
        argmax_idx,
        g_out,
        _gg,
        grad_x,
        _gex,
        _gout,
        t_idx,
        t_x,
        t_out,
        t_argmax,
        t_g_out,
        _t_gg,
    ) = _setup_max_norm_arrays(N, M, device, torch_device, rng)

    ms_wp, ms_wp_graph = _bench_warp_both(
        _noop,
        lambda: segmented_max_norm_backward(g_out, x, argmax_idx, idx, grad_x),
        torch_device,
        warmup,
        runs,
    )

    safe_norm = t_out.clamp(min=1e-30)
    # ``argmax`` is -1 for empty segments — clamp to a safe in-range value and
    # zero out the contribution via the validity mask so the scatter stays in-bounds.
    valid = (t_argmax >= 0).to(t_x.dtype).unsqueeze(-1)
    safe_argmax = t_argmax.clamp(min=0)

    def _t_bwd():
        scale = (t_g_out / safe_norm).unsqueeze(-1)  # (M, 1)
        contrib = scale * t_x[safe_argmax] * valid  # (M, 3)
        gx = torch.zeros_like(t_x)
        gx.index_add_(0, safe_argmax, contrib)
        return gx

    ms_t = _bench_cuda(_noop, _t_bwd, torch_device, warmup, runs)
    _print_row(N, M, ms_wp, ms_t, ms_wp_graph)
    return [("segmented_max_norm_backward", "vec3f", N, M, ms_wp, ms_t, ms_wp_graph)]


def bench_segmented_max_norm_dbl(
    N, M, device, torch_device, rng, warmup, runs, **_kwargs
):
    """Double-backward of segmented_max_norm vs torch (projection-onto-tangent)."""
    (
        idx,
        x,
        _out,
        argmax_idx,
        g_out,
        gg_gx,
        _gx,
        grad_x_ex,
        grad_g_out,
        t_idx,
        t_x,
        t_out,
        t_argmax,
        t_g_out,
        t_gg_gx,
    ) = _setup_max_norm_arrays(N, M, device, torch_device, rng)

    ms_wp, ms_wp_graph = _bench_warp_both(
        _noop,
        lambda: segmented_max_norm_double_backward(
            gg_gx, g_out, x, argmax_idx, idx, grad_x_ex, grad_g_out
        ),
        torch_device,
        warmup,
        runs,
    )

    safe_norm = t_out.clamp(min=1e-30)
    valid = (t_argmax >= 0).to(t_x.dtype).unsqueeze(-1)
    safe_argmax = t_argmax.clamp(min=0)

    def _t_dbl():
        x_hat = t_x[safe_argmax] / safe_norm.unsqueeze(-1)  # (M, 3)
        gg_at = t_gg_gx[safe_argmax]  # (M, 3)
        proj = (x_hat * gg_at).sum(-1)  # (M,)
        scale = (t_g_out / safe_norm).unsqueeze(-1)  # (M, 1)
        contrib = scale * (gg_at - x_hat * proj.unsqueeze(-1)) * valid  # (M, 3)
        gx_ex = torch.zeros_like(t_x)
        gx_ex.index_add_(0, safe_argmax, contrib)
        return gx_ex, proj * valid.squeeze(-1)

    ms_t = _bench_cuda(_noop, _t_dbl, torch_device, warmup, runs)
    _print_row(N, M, ms_wp, ms_t, ms_wp_graph)
    return [
        ("segmented_max_norm_double_backward", "vec3f", N, M, ms_wp, ms_t, ms_wp_graph)
    ]


def _setup_rms_norm_arrays(N, M, device, torch_device, rng):
    x_np = rng.standard_normal((N, 3)).astype(np.float32)
    idx_np = _make_segments(N, M, rng)
    g_out_np = rng.standard_normal(M).astype(np.float32)
    gg_x_np = rng.standard_normal((N, 3)).astype(np.float32)

    x = _wp_vec_array(x_np, wp.vec3f, device)
    idx = wp.array(idx_np, device=device)
    out = wp.zeros(M, dtype=wp.float32, device=device)
    sum_sq = wp.zeros(M, dtype=wp.float32, device=device)
    counts = wp.zeros(M, dtype=wp.int32, device=device)
    inv_norm = wp.zeros(M, dtype=wp.float32, device=device)
    g_out = wp.array(g_out_np, dtype=wp.float32, device=device)
    gg_x = _wp_vec_array(gg_x_np, wp.vec3f, device)
    grad_x = wp.zeros(N, dtype=wp.vec3f, device=device)
    grad_x_ex = wp.zeros(N, dtype=wp.vec3f, device=device)
    grad_g_out_ex = wp.zeros(M, dtype=wp.float32, device=device)
    segmented_rms_norm_forward_precompute(x, idx, sum_sq, counts, out, inv_norm)

    t_x = torch.from_numpy(x_np).to(torch_device)
    t_idx = torch.from_numpy(idx_np.astype(np.int64)).to(torch_device)
    t_g_out = torch.from_numpy(g_out_np).to(torch_device)
    t_gg_x = torch.from_numpy(gg_x_np).to(torch_device)
    t_inv_norm = torch.from_numpy(inv_norm.numpy()).to(torch_device)
    t_counts = torch.from_numpy(counts.numpy().astype(np.float32)).to(torch_device)
    t_inner = torch.zeros(M, dtype=torch.float32, device=torch_device)
    return (
        idx,
        x,
        inv_norm,
        counts,
        g_out,
        gg_x,
        grad_x,
        grad_x_ex,
        grad_g_out_ex,
        t_idx,
        t_x,
        t_inv_norm,
        t_counts,
        t_g_out,
        t_gg_x,
        t_inner,
    )


def bench_segmented_rms_norm_bwd(
    N, M, device, torch_device, rng, warmup, runs, **_kwargs
):
    """1st-order backward of segmented_rms_norm vs torch broadcast."""
    (
        idx,
        x,
        inv_norm,
        _counts,
        g_out,
        _gg,
        grad_x,
        _gex,
        _gout_ex,
        t_idx,
        t_x,
        t_inv_norm,
        _t_counts,
        t_g_out,
        _t_gg,
        _t_inner,
    ) = _setup_rms_norm_arrays(N, M, device, torch_device, rng)

    ms_wp, ms_wp_graph = _bench_warp_both(
        _noop,
        lambda: segmented_rms_norm_backward(g_out, x, inv_norm, idx, grad_x),
        torch_device,
        warmup,
        runs,
    )

    def _t_bwd():
        return t_g_out[t_idx, None] * t_x * t_inv_norm[t_idx, None]

    ms_t = _bench_cuda(_noop, _t_bwd, torch_device, warmup, runs)
    _print_row(N, M, ms_wp, ms_t, ms_wp_graph)
    return [("segmented_rms_norm_backward", "vec3f", N, M, ms_wp, ms_t, ms_wp_graph)]


def bench_segmented_rms_norm_dbl(
    N, M, device, torch_device, rng, warmup, runs, **_kwargs
):
    """Double-backward of segmented_rms_norm vs torch."""
    (
        idx,
        x,
        inv_norm,
        counts,
        g_out,
        gg_x,
        _gx,
        grad_x_ex,
        grad_g_out_ex,
        t_idx,
        t_x,
        t_inv_norm,
        t_counts,
        t_g_out,
        t_gg_x,
        t_inner,
    ) = _setup_rms_norm_arrays(N, M, device, torch_device, rng)

    ms_wp, ms_wp_graph = _bench_warp_both(
        _noop,
        lambda: segmented_rms_norm_double_backward(
            gg_x, x, g_out, inv_norm, counts, idx, M, grad_x_ex, grad_g_out_ex
        ),
        torch_device,
        warmup,
        runs,
    )

    def _t_dbl():
        # inner[s] = sum_i dot(gg_x[i], x[i])
        t_inner.zero_()
        t_inner.index_add_(0, t_idx, (t_gg_x * t_x).sum(-1))
        g_g_out_ex = t_inner * t_inv_norm
        # grad_x_extra[i] = g[s]*n*gg_x[i] - g[s]*n^3*c*inner[s]*x[i]
        n = t_inv_norm[t_idx, None]
        g = t_g_out[t_idx, None]
        c = t_counts[t_idx, None]
        inn = t_inner[t_idx, None]
        gx_ex = g * n * t_gg_x - g * n**3 * c * inn * t_x
        return gx_ex, g_g_out_ex

    ms_t = _bench_cuda(_noop, _t_dbl, torch_device, warmup, runs)
    _print_row(N, M, ms_wp, ms_t, ms_wp_graph)
    return [
        ("segmented_rms_norm_double_backward", "vec3f", N, M, ms_wp, ms_t, ms_wp_graph)
    ]


# ---------------------------------------------------------------------------
# Torch-binding benchmarks (public ``nvalchemiops.torch.segment_ops`` API)
# ---------------------------------------------------------------------------
#
# The benchmarks above time the raw Warp launchers against hand-written torch.
# This section times the *public Torch bindings* — the ``torch.library`` custom
# ops users actually call — in two modes: eager, and (optionally)
# ``torch.compile(fullgraph=True)``.  Each is compared against the equivalent
# hand-written torch the binding is meant to replace.  Forward only: the bindings
# are opaque custom ops, so the interesting question is dispatch/wrapper overhead
# vs. plain torch and what ``torch.compile`` recovers by fusing the surrounding
# graph.  (Eager binding calls also pay the public wrapper's input-validation
# host sync, which ``torch.compile`` elides via ``torch.compiler.is_compiling``.)

_TORCH_BINDING_OPS = [
    "segmented_sum",
    "segmented_dot",
    "segmented_mul",
    "segmented_mean",
    "segmented_rms_norm",
    "segmented_matvec",
]


def _torch_binding_case(op_name, N, M, torch_device, rng):
    """Build ``(binding_fn, ref_fn, label)`` for one public op.

    Both are zero-arg closures over freshly-built leaf tensors. ``binding_fn``
    calls the public Torch binding; ``ref_fn`` is the hand-written torch the
    binding replaces (forward only). ``idx`` is ``int32`` for the binding and
    ``int64`` for torch ``index_add_`` / advanced indexing.
    """
    idx_np = _make_segments(N, M, rng)
    idx32 = torch.from_numpy(idx_np).to(torch_device)
    idx64 = torch.from_numpy(idx_np.astype(np.int64)).to(torch_device)
    kw = {"dtype": torch.float32, "device": torch_device}

    def _randn(*shape):
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).to(
            torch_device
        )

    if op_name == "segmented_sum":
        x = _randn(N, 3)
        out = torch.zeros(M, 3, **kw)

        def ref():
            out.zero_()
            out.index_add_(0, idx64, x)
            return out

        return (lambda: torch_seg.segmented_sum(x, idx32, M)), ref, "vec3f"

    if op_name == "segmented_dot":
        x = _randn(N, 3)
        y = _randn(N, 3)
        out = torch.zeros(M, **kw)

        def ref():
            out.zero_()
            out.index_add_(0, idx64, (x * y).sum(-1))
            return out

        return (lambda: torch_seg.segmented_dot(x, y, idx32, M)), ref, "vec3f"

    if op_name == "segmented_mul":
        x = _randn(N, 3)
        y = _randn(M)

        def ref():
            return x * y[idx64].unsqueeze(1)

        return (lambda: torch_seg.segmented_mul(x, y, idx32, M)), ref, "vec3f_scalar"

    if op_name == "segmented_mean":
        x = _randn(N, 3)
        counts = torch.bincount(idx64, minlength=M).clamp(min=1).to(torch.float32)
        sums = torch.zeros(M, 3, **kw)

        def ref():
            sums.zero_()
            sums.index_add_(0, idx64, x)
            return sums / counts.unsqueeze(1)

        return (lambda: torch_seg.segmented_mean(x, idx32, M)), ref, "vec3f"

    if op_name == "segmented_rms_norm":
        x = _randn(N, 3)
        counts = torch.bincount(idx64, minlength=M).clamp(min=1).to(torch.float32)
        ssq = torch.zeros(M, **kw)

        def ref():
            ssq.zero_()
            ssq.index_add_(0, idx64, (x * x).sum(-1))
            return (ssq / counts).sqrt()

        return (lambda: torch_seg.segmented_rms_norm(x, idx32, M)), ref, "vec3f"

    if op_name == "segmented_matvec":
        v = _randn(N, 3)
        m = _randn(M, 3, 3)

        def ref():
            return torch.einsum("nji,nj->ni", m[idx64], v)

        return (lambda: torch_seg.segmented_matvec(v, m, idx32, M)), ref, "mat33f"

    raise ValueError(f"unknown torch-binding op: {op_name}")


def _print_binding_header(with_compile):
    cols = f"{'N':>10}  {'M':>7}  {'L':>9}  {'binding ms':>11}  "
    if with_compile:
        cols += f"{'compiled ms':>12}  "
    cols += f"{'torch ms':>9}  {'bind spd':>9}"
    if with_compile:
        cols += f"  {'comp spd':>9}"
    sep = "-" * len(cols)
    print(sep)
    print(cols)
    print(sep)


def _print_binding_row(N, M, ms_bind, ms_ref, ms_comp):
    L = N / max(M, 1)
    bind_spd = ms_ref / ms_bind if ms_bind > 0 else float("inf")
    line = f"{N:>10}  {M:>7}  {L:>9.1f}  {ms_bind:>11.4f}  "
    if ms_comp is not None:
        line += f"{ms_comp:>12.4f}  "
    line += f"{ms_ref:>9.4f}  {bind_spd:>8.2f}x"
    if ms_comp is not None:
        comp_spd = ms_ref / ms_comp if ms_comp > 0 else float("inf")
        line += f"  {comp_spd:>8.2f}x"
    print(line)


def bench_torch_binding(
    N, M, torch_device, rng, warmup, runs, op_name, compile_binding
):
    """Time one public Torch binding (eager + optional compiled) vs hand-torch."""
    binding, ref, label = _torch_binding_case(op_name, N, M, torch_device, rng)

    ms_bind = _bench_cuda(_noop, binding, torch_device, warmup, runs)
    ms_ref = _bench_cuda(_noop, ref, torch_device, warmup, runs)

    ms_comp = None
    if compile_binding:
        # Each sweep point closes over differently-shaped tensors (and a
        # different ``M``), so reusing one Dynamo cache would recompile the same
        # code object past the recompile limit. Reset first so every point gets
        # a clean, single compile — which is exactly the graph we want to time.
        torch._dynamo.reset()
        compiled = torch.compile(binding, fullgraph=True)
        # Trigger compilation + warm the cache outside the timed region.
        for _ in range(3):
            compiled()
        torch.cuda.synchronize(torch_device)
        ms_comp = _bench_cuda(_noop, compiled, torch_device, warmup, runs)

    _print_binding_row(N, M, ms_bind, ms_ref, ms_comp)
    return (op_name, label, N, M, ms_bind, ms_comp, ms_ref)


def run_torch_binding_benchmarks(config, output_dir, device_str, compile_binding):
    """Benchmark the public Torch bindings (forward) across the N x L sweep."""
    params = config.get("parameters", {})
    warmup = params.get("warmup", 5)
    runs = params.get("runs", 50)

    sweep = config.get("sweep", {})
    n_values = sweep.get("total_elements", [1_000, 10_000, 100_000, 1_000_000])
    l_values = sweep.get("avg_segment_lengths", [10, 100, 1_000, 10_000])

    binding_cfg = config.get("torch_bindings", {}) or {}
    ops = binding_cfg.get("ops", _TORCH_BINDING_OPS)

    torch_device = torch.device(device_str)
    rng = np.random.default_rng(42)
    gpu_sku = _get_gpu_sku()

    print("\n" + "=" * 70)
    print("Torch bindings (public custom ops) — forward")
    mode = "eager + torch.compile" if compile_binding else "eager"
    print(f"Mode: {mode}; {warmup} warmup, median of {runs} runs")
    print("=" * 70)

    all_results = []
    for op_name in ops:
        print(f"\n{op_name} (torch binding)")
        _print_binding_header(compile_binding)
        for N in n_values:
            for L in l_values:
                if L > N:
                    continue
                M = max(N // L, 1)
                all_results.append(
                    bench_torch_binding(
                        N, M, torch_device, rng, warmup, runs, op_name, compile_binding
                    )
                )
            print()

    if all_results:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"segment_ops_torch_binding_{gpu_sku}.csv"
        fieldnames = [
            "operation",
            "dtype",
            "total_elements",
            "num_segments",
            "avg_segment_length",
            "binding_median_ms",
            "compiled_median_ms",
            "torch_ref_median_ms",
            "binding_speedup",
            "compiled_speedup",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for op, label, N, M, ms_bind, ms_comp, ms_ref in all_results:
                L = N / max(M, 1)
                bind_spd = ms_ref / ms_bind if ms_bind > 0 else float("inf")
                comp_spd = ms_ref / ms_comp if ms_comp and ms_comp > 0 else None
                writer.writerow(
                    {
                        "operation": op,
                        "dtype": label,
                        "total_elements": N,
                        "num_segments": M,
                        "avg_segment_length": f"{L:.1f}",
                        "binding_median_ms": f"{ms_bind:.4f}",
                        "compiled_median_ms": (
                            f"{ms_comp:.4f}" if ms_comp is not None else ""
                        ),
                        "torch_ref_median_ms": f"{ms_ref:.4f}",
                        "binding_speedup": f"{bind_spd:.2f}",
                        "compiled_speedup": (
                            f"{comp_spd:.2f}" if comp_spd is not None else ""
                        ),
                    }
                )
        print(f"Wrote torch-binding results to {csv_path}")


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
    # PR 1 backward benchmarks (split into 1st-order and double-backward sections)
    (
        "segmented_sum_bwd",
        "segmented_sum backward (f32, vec3f)",
        bench_segmented_sum_bwd,
        False,
    ),
    (
        "segmented_sum_dbl",
        "segmented_sum double-backward (f32, vec3f)",
        bench_segmented_sum_dbl,
        False,
    ),
    (
        "segmented_dot_bwd",
        "segmented_dot backward (f32, vec3f)",
        bench_segmented_dot_bwd,
        False,
    ),
    (
        "segmented_dot_dbl",
        "segmented_dot double-backward (f32, vec3f)",
        bench_segmented_dot_dbl,
        False,
    ),
    (
        "segmented_mul_bwd",
        "segmented_mul backward (f32, vec3f*scalar)",
        bench_segmented_mul_bwd,
        False,
    ),
    (
        "segmented_mul_dbl",
        "segmented_mul double-backward (f32, vec3f*scalar)",
        bench_segmented_mul_dbl,
        False,
    ),
    (
        "segmented_matvec_bwd",
        "segmented_matvec backward (mat33f)",
        bench_segmented_matvec_bwd,
        False,
    ),
    (
        "segmented_matvec_dbl",
        "segmented_matvec double-backward (mat33f)",
        bench_segmented_matvec_dbl,
        False,
    ),
    (
        "segmented_max_norm_bwd",
        "segmented_max_norm backward (vec3f)",
        bench_segmented_max_norm_bwd,
        False,
    ),
    (
        "segmented_max_norm_dbl",
        "segmented_max_norm double-backward (vec3f)",
        bench_segmented_max_norm_dbl,
        False,
    ),
    (
        "segmented_rms_norm_bwd",
        "segmented_rms_norm backward (vec3f)",
        bench_segmented_rms_norm_bwd,
        False,
    ),
    (
        "segmented_rms_norm_dbl",
        "segmented_rms_norm double-backward (vec3f)",
        bench_segmented_rms_norm_dbl,
        False,
    ),
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
        # Default to disabled: new registry entries must opt in via the yaml.
        # This prevents adding a group to ``_BENCHMARKS`` from silently expanding
        # the default benchmark run.
        if not isinstance(op_cfg, dict) or not op_cfg.get("enabled", False):
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

        _print_header(f"{display_name}{dtype_info}", with_graph=True)
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
            "warp_graph_median_ms",
            "torch_median_ms",
            "speedup",
            "graph_speedup",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                # Backward-compatible: forward ops emit 6-tuples (no graph timing);
                # bwd/dbl ops emit 7-tuples with the graph median.
                if len(row) == 7:
                    op, dtype_label, N, M, ms_wp, ms_t, ms_wp_graph = row
                else:
                    op, dtype_label, N, M, ms_wp, ms_t = row
                    ms_wp_graph = None
                L = N / max(M, 1)
                speedup = ms_t / ms_wp if ms_wp > 0 else float("inf")
                graph_speedup = (
                    ms_t / ms_wp_graph if ms_wp_graph and ms_wp_graph > 0 else None
                )
                writer.writerow(
                    {
                        "operation": op,
                        "dtype": dtype_label,
                        "total_elements": N,
                        "num_segments": M,
                        "avg_segment_length": f"{L:.1f}",
                        "warp_median_ms": f"{ms_wp:.4f}",
                        "warp_graph_median_ms": (
                            f"{ms_wp_graph:.4f}" if ms_wp_graph is not None else ""
                        ),
                        "torch_median_ms": f"{ms_t:.4f}",
                        "speedup": f"{speedup:.2f}",
                        "graph_speedup": (
                            f"{graph_speedup:.2f}" if graph_speedup is not None else ""
                        ),
                    }
                )
        print(f"Wrote results to {csv_path}")


def main():
    """Execute the benchmark suite for segmented operations."""
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
    parser.add_argument(
        "--torch-bindings",
        action="store_true",
        help="Benchmark the public nvalchemiops.torch.segment_ops bindings "
        "(custom ops) vs hand-written torch.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="With --torch-bindings, also time the bindings under "
        "torch.compile(fullgraph=True).",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    output_dir = Path(args.output_dir)
    run_benchmarks(config, output_dir, args.device)

    binding_cfg = config.get("torch_bindings", {}) or {}
    run_bindings = args.torch_bindings or binding_cfg.get("enabled", False)
    compile_binding = args.torch_compile or binding_cfg.get("compile", False)
    if run_bindings:
        run_torch_binding_benchmarks(
            config, output_dir, args.device, compile_binding=compile_binding
        )


if __name__ == "__main__":
    main()

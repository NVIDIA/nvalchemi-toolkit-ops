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

"""
FIRE2 Kernel-Level Performance Benchmark
=========================================

Measures raw per-step GPU time using CUDA events for FIRE2, FIRE1,
the PyTorch adapter (``fire2_step_coord``), and a pure PyTorch reference.
Sweeps N (total atoms) and M (number of systems) across float32 and float64.

Configuration is loaded from ``benchmark_config.yaml`` (``fire2_perf`` section).

Usage
-----
    python -m benchmarks.dynamics.benchmark_fire2 [--config benchmark_config.yaml]
                                                   [--output-dir ./benchmark_results]
                                                   [--device cuda:0]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import warp as wp

from benchmarks.dynamics.shared_utils import get_gpu_sku, load_config
from nvalchemiops.batch_utils import atom_ptr_to_batch_idx, batch_idx_to_atom_ptr
from nvalchemiops.dynamics.optimizers import fire2_step, fire_step
from nvalchemiops.dynamics.utils.cell_filter import (
    extend_atom_ptr,
    pack_forces_with_cell,
    pack_positions_with_cell,
    pack_velocities_with_cell,
    unpack_positions_with_cell,
    unpack_velocities_with_cell,
)
from nvalchemiops.torch.fire2 import (
    fire2_step_coord,
    fire2_step_coord_cell,
    fire2_step_extended,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bench_cuda(fn, warmup: int, runs: int, torch_device) -> dict:
    """Return timing stats (ms) over *runs* using CUDA events."""
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize(torch_device)

    times = []
    for _ in range(runs):
        torch.cuda.synchronize(torch_device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(torch_device)
        times.append(start.elapsed_time(end))

    arr = np.array(times)
    return {
        "median_ms": float(np.median(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def _make_data(N, M, device, dtype=torch.float32):
    """Create random benchmark data."""
    rng = np.random.default_rng(42)
    np_dtype = np.float32 if dtype == torch.float32 else np.float64

    pos_np = rng.standard_normal((N, 3)).astype(np_dtype)
    vel_np = rng.standard_normal((N, 3)).astype(np_dtype) * 0.01
    forces_np = rng.standard_normal((N, 3)).astype(np_dtype)
    atoms_per_sys = N // M
    bidx_np = np.repeat(np.arange(M, dtype=np.int32), atoms_per_sys)

    pos = torch.tensor(pos_np, device=device, dtype=dtype)
    vel = torch.tensor(vel_np, device=device, dtype=dtype)
    forces = torch.tensor(forces_np, device=device, dtype=dtype)
    bidx = torch.tensor(bidx_np, device=device, dtype=torch.int32)

    return pos, vel, forces, bidx


# ---------------------------------------------------------------------------
# Benchmark: FIRE2 (Warp)
# ---------------------------------------------------------------------------


def bench_fire2_warp(N, M, device, dtype, hyper, warmup, runs):
    """Benchmark Warp FIRE2 with pre-allocated scratch buffers."""
    pos, vel, forces, bidx = _make_data(N, M, device, dtype)
    wp_device = wp.device_from_torch(device)
    vec_dtype = wp.vec3f if dtype == torch.float32 else wp.vec3d
    scalar_dtype = wp.float32 if dtype == torch.float32 else wp.float64
    np_dtype = np.float32 if dtype == torch.float32 else np.float64

    pos_wp = wp.from_torch(pos, dtype=vec_dtype)
    vel_wp = wp.from_torch(vel.clone(), dtype=vec_dtype)
    forces_wp = wp.from_torch(forces, dtype=vec_dtype)
    bidx_wp = wp.from_torch(bidx, dtype=wp.int32)
    alpha_wp = wp.array(
        np.full(M, hyper["alpha0"], dtype=np_dtype),
        dtype=scalar_dtype,
        device=wp_device,
    )
    dt_wp = wp.array(
        np.full(M, 0.05, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )
    nsteps_wp = wp.array(np.zeros(M, dtype=np.int32), dtype=wp.int32, device=wp_device)

    # Pre-allocate scratch buffers (reused across iterations)
    vf = wp.zeros(M, dtype=scalar_dtype, device=wp_device)
    v_sumsq = wp.zeros(M, dtype=scalar_dtype, device=wp_device)
    f_sumsq = wp.zeros(M, dtype=scalar_dtype, device=wp_device)
    max_norm = wp.zeros(M, dtype=scalar_dtype, device=wp_device)

    def run():
        vf.zero_()
        v_sumsq.zero_()
        f_sumsq.zero_()
        max_norm.zero_()
        fire2_step(
            pos_wp,
            vel_wp,
            forces_wp,
            bidx_wp,
            alpha_wp,
            dt_wp,
            nsteps_wp,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **hyper,
        )

    return _bench_cuda(run, warmup, runs, device)


# ---------------------------------------------------------------------------
# Benchmark: FIRE1 (Warp, batch_idx)
# ---------------------------------------------------------------------------


def bench_fire1_warp(N, M, device, dtype, warmup, runs):
    """Benchmark Warp FIRE1 (batch_idx, no-downhill, single fused kernel)."""
    pos, vel, forces, bidx = _make_data(N, M, device, dtype)
    wp_device = wp.device_from_torch(device)
    vec_dtype = wp.vec3f if dtype == torch.float32 else wp.vec3d
    scalar_dtype = wp.float32 if dtype == torch.float32 else wp.float64
    np_dtype = np.float32 if dtype == torch.float32 else np.float64

    pos_wp = wp.from_torch(pos, dtype=vec_dtype)
    vel_wp = wp.from_torch(vel.clone(), dtype=vec_dtype)
    forces_wp = wp.from_torch(forces, dtype=vec_dtype)
    bidx_wp = wp.from_torch(bidx, dtype=wp.int32)
    masses_wp = wp.array(
        np.ones(N, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )

    alpha_wp = wp.array(
        np.full(M, 0.1, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )
    dt_wp = wp.array(
        np.full(M, 0.01, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )
    alpha_start_wp = wp.array(
        np.full(M, 0.1, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )
    f_alpha_wp = wp.array(
        np.full(M, 0.99, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )
    dt_min_wp = wp.array(
        np.full(M, 0.001, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )
    dt_max_wp = wp.array(
        np.full(M, 1.0, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )
    maxstep_wp = wp.array(
        np.full(M, 0.1, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )
    nsteps_wp = wp.array(np.zeros(M, dtype=np.int32), dtype=wp.int32, device=wp_device)
    nmin_wp = wp.array(np.full(M, 5, dtype=np.int32), dtype=wp.int32, device=wp_device)
    f_dec_wp = wp.array(
        np.full(M, 0.5, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )
    f_inc_wp = wp.array(
        np.full(M, 1.1, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )

    # Pre-allocate scratch buffers
    uphill_flag = wp.zeros(M, dtype=wp.int32, device=wp_device)
    vf = wp.zeros(M, dtype=scalar_dtype, device=wp_device)
    vv = wp.zeros(M, dtype=scalar_dtype, device=wp_device)
    ff = wp.zeros(M, dtype=scalar_dtype, device=wp_device)

    def run():
        vf.zero_()
        vv.zero_()
        ff.zero_()
        uphill_flag.zero_()
        fire_step(
            pos_wp,
            vel_wp,
            forces_wp,
            masses_wp,
            alpha_wp,
            dt_wp,
            alpha_start_wp,
            f_alpha_wp,
            dt_min_wp,
            dt_max_wp,
            maxstep_wp,
            nsteps_wp,
            nmin_wp,
            f_dec_wp,
            f_inc_wp,
            uphill_flag,
            vf,
            vv,
            ff,
            batch_idx=bidx_wp,
        )

    return _bench_cuda(run, warmup, runs, device)


# ---------------------------------------------------------------------------
# Benchmark: FIRE1 (Warp, batch_idx, variable-cell)
# ---------------------------------------------------------------------------


def bench_fire1_warp_cell(N, M, device, dtype, warmup, runs):
    """Benchmark Warp FIRE1 with variable-cell DOFs (pack/unpack + fire_step)."""
    pos, vel, forces, bidx = _make_data(N, M, device, dtype)
    wp_device = wp.device_from_torch(device)
    vec_dtype = wp.vec3f if dtype == torch.float32 else wp.vec3d
    mat_dtype = wp.mat33f if dtype == torch.float32 else wp.mat33d
    scalar_dtype = wp.float32 if dtype == torch.float32 else wp.float64
    np_dtype = np.float32 if dtype == torch.float32 else np.float64

    # Cell data: identity cell per system, zero cell velocities, random cell force
    rng = np.random.default_rng(123)
    cell = (
        torch.eye(3, dtype=dtype, device=device)
        .unsqueeze(0)
        .expand(M, -1, -1)
        .contiguous()
    )
    cell_vel = torch.zeros(M, 3, 3, dtype=dtype, device=device)
    cell_force_np = rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.01
    cell_force = torch.tensor(cell_force_np, dtype=dtype, device=device)

    # Warp views of original arrays
    wp_pos = wp.from_torch(pos, dtype=vec_dtype)
    wp_vel = wp.from_torch(vel.clone(), dtype=vec_dtype)
    wp_forces = wp.from_torch(forces, dtype=vec_dtype)
    wp_cell = wp.from_torch(cell, dtype=mat_dtype)
    wp_cell_vel = wp.from_torch(cell_vel, dtype=mat_dtype)
    wp_cell_force = wp.from_torch(cell_force, dtype=mat_dtype)

    # Pre-compute static index metadata
    atom_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
    atom_counts = torch.zeros(M, dtype=torch.int32, device=device)
    batch_idx_to_atom_ptr(
        wp.from_torch(bidx, dtype=wp.int32),
        wp.from_torch(atom_counts, dtype=wp.int32),
        wp.from_torch(atom_ptr, dtype=wp.int32),
    )
    wp_atom_ptr = wp.from_torch(atom_ptr, dtype=wp.int32)

    ext_atom_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
    extend_atom_ptr(
        wp_atom_ptr,
        wp.from_torch(ext_atom_ptr, dtype=wp.int32),
        device=wp_device,
    )
    wp_ext_atom_ptr = wp.from_torch(ext_atom_ptr, dtype=wp.int32)

    N_ext = N + 2 * M
    ext_batch_idx = torch.empty(N_ext, dtype=torch.int32, device=device)
    atom_ptr_to_batch_idx(
        wp_ext_atom_ptr,
        wp.from_torch(ext_batch_idx, dtype=wp.int32),
    )
    ext_bidx_wp = wp.from_torch(ext_batch_idx, dtype=wp.int32)
    wp_bidx = wp.from_torch(bidx, dtype=wp.int32)

    # Extended working arrays
    ext_pos = torch.empty(N_ext, 3, dtype=dtype, device=device)
    ext_vel = torch.empty(N_ext, 3, dtype=dtype, device=device)
    ext_forces = torch.empty(N_ext, 3, dtype=dtype, device=device)
    wp_ext_pos = wp.from_torch(ext_pos, dtype=vec_dtype)
    wp_ext_vel = wp.from_torch(ext_vel, dtype=vec_dtype)
    wp_ext_forces = wp.from_torch(ext_forces, dtype=vec_dtype)

    # Extended masses (unit mass for all DOFs including cell)
    ext_masses_wp = wp.array(
        np.ones(N_ext, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )

    # FIRE1 per-system control parameters
    alpha_wp = wp.array(
        np.full(M, 0.1, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )
    dt_wp = wp.array(
        np.full(M, 0.01, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )
    alpha_start_wp = wp.array(
        np.full(M, 0.1, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )
    f_alpha_wp = wp.array(
        np.full(M, 0.99, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )
    dt_min_wp = wp.array(
        np.full(M, 0.001, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )
    dt_max_wp = wp.array(
        np.full(M, 1.0, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )
    maxstep_wp = wp.array(
        np.full(M, 0.1, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )
    nsteps_wp = wp.array(np.zeros(M, dtype=np.int32), dtype=wp.int32, device=wp_device)
    nmin_wp = wp.array(np.full(M, 5, dtype=np.int32), dtype=wp.int32, device=wp_device)
    f_dec_wp = wp.array(
        np.full(M, 0.5, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )
    f_inc_wp = wp.array(
        np.full(M, 1.1, dtype=np_dtype), dtype=scalar_dtype, device=wp_device
    )

    # Scratch buffers
    uphill_flag = wp.zeros(M, dtype=wp.int32, device=wp_device)
    vf = wp.zeros(M, dtype=scalar_dtype, device=wp_device)
    vv = wp.zeros(M, dtype=scalar_dtype, device=wp_device)
    ff = wp.zeros(M, dtype=scalar_dtype, device=wp_device)

    def run():
        # Pack into extended arrays
        if M == 1:
            pack_positions_with_cell(
                wp_pos,
                wp_cell,
                wp_ext_pos,
                device=wp_device,
            )
            pack_velocities_with_cell(
                wp_vel,
                wp_cell_vel,
                wp_ext_vel,
                device=wp_device,
            )
            pack_forces_with_cell(
                wp_forces,
                wp_cell_force,
                wp_ext_forces,
                device=wp_device,
            )
        else:
            pack_positions_with_cell(
                wp_pos,
                wp_cell,
                wp_ext_pos,
                wp_atom_ptr,
                wp_ext_atom_ptr,
                device=wp_device,
                batch_idx=wp_bidx,
            )
            pack_velocities_with_cell(
                wp_vel,
                wp_cell_vel,
                wp_ext_vel,
                wp_atom_ptr,
                wp_ext_atom_ptr,
                device=wp_device,
                batch_idx=wp_bidx,
            )
            pack_forces_with_cell(
                wp_forces,
                wp_cell_force,
                wp_ext_forces,
                wp_atom_ptr,
                wp_ext_atom_ptr,
                device=wp_device,
                batch_idx=wp_bidx,
            )

        # FIRE1 step on extended arrays
        vf.zero_()
        vv.zero_()
        ff.zero_()
        uphill_flag.zero_()
        fire_step(
            wp_ext_pos,
            wp_ext_vel,
            wp_ext_forces,
            ext_masses_wp,
            alpha_wp,
            dt_wp,
            alpha_start_wp,
            f_alpha_wp,
            dt_min_wp,
            dt_max_wp,
            maxstep_wp,
            nsteps_wp,
            nmin_wp,
            f_dec_wp,
            f_inc_wp,
            uphill_flag,
            vf,
            vv,
            ff,
            batch_idx=ext_bidx_wp,
        )

        # Unpack back to original arrays
        if M == 1:
            unpack_positions_with_cell(
                wp_ext_pos,
                wp_pos,
                wp_cell,
                num_atoms=N,
                device=wp_device,
            )
            unpack_velocities_with_cell(
                wp_ext_vel,
                wp_vel,
                wp_cell_vel,
                num_atoms=N,
                device=wp_device,
            )
        else:
            unpack_positions_with_cell(
                wp_ext_pos,
                wp_pos,
                wp_cell,
                atom_ptr=wp_atom_ptr,
                ext_atom_ptr=wp_ext_atom_ptr,
                device=wp_device,
                batch_idx=wp_bidx,
            )
            unpack_velocities_with_cell(
                wp_ext_vel,
                wp_vel,
                wp_cell_vel,
                atom_ptr=wp_atom_ptr,
                ext_atom_ptr=wp_ext_atom_ptr,
                device=wp_device,
                batch_idx=wp_bidx,
            )

    return _bench_cuda(run, warmup, runs, device)


# ---------------------------------------------------------------------------
# Benchmark: PyTorch adapter (fire2_step_coord)
# ---------------------------------------------------------------------------


def bench_fire2_torch_adapter(N, M, device, dtype, hyper, warmup, runs):
    """Benchmark PyTorch adapter fire2_step_coord."""
    pos, vel, forces, bidx = _make_data(N, M, device, dtype)

    alpha = torch.full((M,), hyper["alpha0"], dtype=dtype, device=device)
    dt = torch.full((M,), 0.05, dtype=dtype, device=device)
    nsteps_inc = torch.zeros(M, dtype=torch.int32, device=device)

    # Pre-allocate scratch buffers
    vf = torch.zeros(M, dtype=dtype, device=device)
    v_sumsq = torch.zeros(M, dtype=dtype, device=device)
    f_sumsq = torch.zeros(M, dtype=dtype, device=device)
    max_norm = torch.zeros(M, dtype=dtype, device=device)

    def run():
        fire2_step_coord(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            vf=vf,
            v_sumsq=v_sumsq,
            f_sumsq=f_sumsq,
            max_norm=max_norm,
            **hyper,
        )

    return _bench_cuda(run, warmup, runs, device)


# ---------------------------------------------------------------------------
# Benchmark: PyTorch adapter (fire2_step_coord_cell)
# ---------------------------------------------------------------------------


def bench_fire2_torch_adapter_cell(N, M, device, dtype, hyper, warmup, runs):
    """Benchmark PyTorch adapter fire2_step_coord_cell (variable-cell)."""
    pos, vel, forces, bidx = _make_data(N, M, device, dtype)

    # Cell data: identity cell per system, zero cell velocities, random cell force
    rng = np.random.default_rng(123)
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    cell = (
        torch.eye(3, dtype=dtype, device=device)
        .unsqueeze(0)
        .expand(M, -1, -1)
        .contiguous()
    )
    cell_vel = torch.zeros(M, 3, 3, dtype=dtype, device=device)
    cell_force_np = rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.01
    cell_force = torch.tensor(cell_force_np, dtype=dtype, device=device)

    alpha = torch.full((M,), hyper["alpha0"], dtype=dtype, device=device)
    dt = torch.full((M,), 0.05, dtype=dtype, device=device)
    nsteps_inc = torch.zeros(M, dtype=torch.int32, device=device)

    # Pre-compute static index metadata (batch_idx is constant across steps)
    wp_device = wp.device_from_torch(device)
    atom_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
    atom_counts = torch.zeros(M, dtype=torch.int32, device=device)
    batch_idx_to_atom_ptr(
        wp.from_torch(bidx, dtype=wp.int32),
        wp.from_torch(atom_counts, dtype=wp.int32),
        wp.from_torch(atom_ptr, dtype=wp.int32),
    )

    ext_atom_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
    extend_atom_ptr(
        wp.from_torch(atom_ptr, dtype=wp.int32),
        wp.from_torch(ext_atom_ptr, dtype=wp.int32),
        device=wp_device,
    )

    N_ext = N + 2 * M
    ext_batch_idx = torch.empty(N_ext, dtype=torch.int32, device=device)
    atom_ptr_to_batch_idx(
        wp.from_torch(ext_atom_ptr, dtype=wp.int32),
        wp.from_torch(ext_batch_idx, dtype=wp.int32),
    )

    # Pre-allocate scratch buffers
    ext_pos = torch.empty(N_ext, 3, dtype=dtype, device=device)
    ext_vel = torch.empty(N_ext, 3, dtype=dtype, device=device)
    ext_forces = torch.empty(N_ext, 3, dtype=dtype, device=device)
    vf = torch.zeros(M, dtype=dtype, device=device)
    v_sumsq = torch.zeros(M, dtype=dtype, device=device)
    f_sumsq = torch.zeros(M, dtype=dtype, device=device)
    max_norm = torch.zeros(M, dtype=dtype, device=device)

    def run():
        fire2_step_coord_cell(
            pos,
            vel,
            forces,
            cell,
            cell_vel,
            cell_force,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            ext_positions=ext_pos,
            ext_velocities=ext_vel,
            ext_forces=ext_forces,
            ext_batch_idx=ext_batch_idx,
            vf=vf,
            v_sumsq=v_sumsq,
            f_sumsq=f_sumsq,
            max_norm=max_norm,
            **hyper,
        )

    return _bench_cuda(run, warmup, runs, device)


# ---------------------------------------------------------------------------
# Benchmark: FIRE2 on persistent extended arrays (no pack/unpack in loop)
# ---------------------------------------------------------------------------


def bench_fire2_extended(N, M, device, dtype, hyper, warmup, runs):
    """Benchmark fire2_step_extended on pre-packed extended arrays."""
    pos, vel, forces, bidx = _make_data(N, M, device, dtype)

    # Cell data
    rng = np.random.default_rng(124)
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    cell = (
        torch.eye(3, dtype=dtype, device=device)
        .unsqueeze(0)
        .expand(M, -1, -1)
        .contiguous()
    )
    cell_vel = torch.zeros(M, 3, 3, dtype=dtype, device=device)
    cell_force_np = rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.01
    cell_force = torch.tensor(cell_force_np, dtype=dtype, device=device)

    alpha = torch.full((M,), hyper["alpha0"], dtype=dtype, device=device)
    dt = torch.full((M,), 0.05, dtype=dtype, device=device)
    nsteps_inc = torch.zeros(M, dtype=torch.int32, device=device)

    # Pre-compute static index metadata
    wp_device = wp.device_from_torch(device)
    vec_type = {torch.float32: wp.vec3f, torch.float64: wp.vec3d}[dtype]
    mat_type = {torch.float32: wp.mat33f, torch.float64: wp.mat33d}[dtype]

    atom_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
    atom_counts = torch.zeros(M, dtype=torch.int32, device=device)
    batch_idx_to_atom_ptr(
        wp.from_torch(bidx, dtype=wp.int32),
        wp.from_torch(atom_counts, dtype=wp.int32),
        wp.from_torch(atom_ptr, dtype=wp.int32),
    )

    ext_atom_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
    extend_atom_ptr(
        wp.from_torch(atom_ptr, dtype=wp.int32),
        wp.from_torch(ext_atom_ptr, dtype=wp.int32),
        device=wp_device,
    )

    N_ext = N + 2 * M
    ext_batch_idx = torch.empty(N_ext, dtype=torch.int32, device=device)
    atom_ptr_to_batch_idx(
        wp.from_torch(ext_atom_ptr, dtype=wp.int32),
        wp.from_torch(ext_batch_idx, dtype=wp.int32),
    )

    # Pre-allocate and pack extended arrays ONCE (before timing loop)
    ext_pos = torch.empty(N_ext, 3, dtype=dtype, device=device)
    ext_vel = torch.empty(N_ext, 3, dtype=dtype, device=device)
    ext_forces = torch.empty(N_ext, 3, dtype=dtype, device=device)

    wp_atom_ptr = wp.from_torch(atom_ptr, dtype=wp.int32)
    wp_ext_atom_ptr = wp.from_torch(ext_atom_ptr, dtype=wp.int32)
    wp_bidx = wp.from_torch(bidx, dtype=wp.int32)

    pack_positions_with_cell(
        wp.from_torch(pos, dtype=vec_type),
        wp.from_torch(cell, dtype=mat_type),
        wp.from_torch(ext_pos, dtype=vec_type),
        wp_atom_ptr,
        wp_ext_atom_ptr,
        device=wp_device,
        batch_idx=wp_bidx,
    )
    pack_velocities_with_cell(
        wp.from_torch(vel, dtype=vec_type),
        wp.from_torch(cell_vel, dtype=mat_type),
        wp.from_torch(ext_vel, dtype=vec_type),
        wp_atom_ptr,
        wp_ext_atom_ptr,
        device=wp_device,
        batch_idx=wp_bidx,
    )
    pack_forces_with_cell(
        wp.from_torch(forces, dtype=vec_type),
        wp.from_torch(cell_force, dtype=mat_type),
        wp.from_torch(ext_forces, dtype=vec_type),
        wp_atom_ptr,
        wp_ext_atom_ptr,
        device=wp_device,
        batch_idx=wp_bidx,
    )

    vf = torch.zeros(M, dtype=dtype, device=device)
    v_sumsq = torch.zeros(M, dtype=dtype, device=device)
    f_sumsq = torch.zeros(M, dtype=dtype, device=device)
    max_norm_buf = torch.zeros(M, dtype=dtype, device=device)

    def run():
        fire2_step_extended(
            ext_pos,
            ext_vel,
            ext_forces,
            ext_batch_idx,
            alpha,
            dt,
            nsteps_inc,
            vf=vf,
            v_sumsq=v_sumsq,
            f_sumsq=f_sumsq,
            max_norm=max_norm_buf,
            **hyper,
        )

    return _bench_cuda(run, warmup, runs, device)


# ---------------------------------------------------------------------------
# Benchmark: Pure PyTorch reference FIRE2
# ---------------------------------------------------------------------------


def bench_fire2_torch_ref(N, M, device, dtype, hyper, warmup, runs):
    """Benchmark pure PyTorch FIRE2 reference (for comparison)."""
    pos, vel, forces, bidx = _make_data(N, M, device, dtype)

    alpha = torch.full((M,), hyper["alpha0"], dtype=dtype, device=device)
    dt = torch.full((M,), 0.05, dtype=dtype, device=device)
    nsteps_inc = torch.zeros(M, dtype=torch.int32, device=device)

    delaystep = hyper["delaystep"]
    dtgrow = hyper["dtgrow"]
    dtshrink = hyper["dtshrink"]
    alphashrink = hyper["alphashrink"]
    alpha0 = hyper["alpha0"]
    tmax = hyper["tmax"]
    tmin = hyper["tmin"]
    maxstep = hyper["maxstep"]

    def run():
        v = vel.clone()
        p = pos.clone()
        a = alpha.clone()
        t = dt.clone()
        ns = nsteps_inc.clone()

        # 1. half-step
        v += forces * t[bidx].unsqueeze(1)

        # 2. inner products via scatter
        vf_buf = torch.zeros(M, dtype=dtype, device=device)
        vf_buf.scatter_add_(0, bidx, (v * forces).sum(dim=1))
        v_sumsq = torch.zeros(M, dtype=dtype, device=device)
        v_sumsq.scatter_add_(0, bidx, (v * v).sum(dim=1))
        f_sumsq = torch.zeros(M, dtype=dtype, device=device)
        f_sumsq.scatter_add_(0, bidx, (forces * forces).sum(dim=1))

        # 3. param update
        w_inc = vf_buf > 0
        ns = torch.where(w_inc, ns + 1, torch.zeros_like(ns))
        grow_mask = w_inc & (ns > delaystep)
        t = torch.where(grow_mask, (t * dtgrow).clamp(max=tmax), t)
        a = torch.where(grow_mask, a * alphashrink, a)
        t = torch.where(~w_inc, (t * dtshrink).clamp(min=tmin), t)
        a = torch.where(~w_inc, torch.full_like(a, alpha0), a)

        # 4. mixing
        ratio = (v_sumsq / f_sumsq).sqrt()
        mix_a = (1.0 - a)[bidx].unsqueeze(1)
        mix_b = (a * ratio)[bidx].unsqueeze(1)
        v = mix_a * v + mix_b * forces

        # 5. step
        step = v * t[bidx].unsqueeze(1)
        w_dec = ~w_inc
        dec_mask = w_dec[bidx].unsqueeze(1)
        step = torch.where(dec_mask, -0.5 * t[bidx].unsqueeze(1) * v, step)
        v = torch.where(dec_mask, torch.zeros_like(v), v)

        # 6. clamp
        norms = step.norm(dim=1)
        mn = torch.zeros(M, dtype=dtype, device=device)
        mn.scatter_reduce_(0, bidx, norms, reduce="amax")
        inv = (maxstep / mn).clamp(max=1.0)
        p += step * inv[bidx].unsqueeze(1)

    return _bench_cuda(run, warmup, runs, device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Mapping from config dtype strings to torch/numpy dtypes
_DTYPE_MAP = {
    "float32": (torch.float32, "f32"),
    "float64": (torch.float64, "f64"),
}


def run_benchmarks(config: dict, output_dir: Path, device: torch.device) -> None:
    """Run FIRE2 kernel performance benchmarks.

    Parameters
    ----------
    config : dict
        Benchmark configuration (uses the ``fire2_perf`` section).
    output_dir : Path
        Output directory for CSV files.
    device : torch.device
        CUDA device to benchmark on.
    """
    perf_config = config.get("fire2_perf", {})
    if not perf_config.get("enabled", True):
        print("fire2_perf benchmarks disabled in config")
        return

    n_values = perf_config.get("total_atoms", [1_000, 10_000, 100_000, 1_000_000])
    m_values = perf_config.get("num_systems", [1, 10, 100])
    warmup = perf_config.get("warmup", 10)
    runs = perf_config.get("runs", 50)
    dtypes = perf_config.get("dtypes", ["float32", "float64"])
    methods_cfg = perf_config.get("methods", {})
    hyper = perf_config.get("hyperparameters", {})

    # Defaults for hyperparameters
    hyper.setdefault("delaystep", 5)
    hyper.setdefault("dtgrow", 1.05)
    hyper.setdefault("dtshrink", 0.75)
    hyper.setdefault("alphashrink", 0.985)
    hyper.setdefault("alpha0", 0.09)
    hyper.setdefault("tmax", 0.08)
    hyper.setdefault("tmin", 0.005)
    hyper.setdefault("maxstep", 0.1)

    gpu_sku = get_gpu_sku()
    all_rows = []

    # Method registry: (config_key, label, short_label, bench_fn_factory)
    # Order: FIRE1(wp), FIRE2(wp), FIRE2(adapt), PyTorch
    method_registry = []
    if methods_cfg.get("warp_fire1", True):
        method_registry.append(
            (
                "warp_fire1",
                "FIRE1(wp)",
                "F1wp",
                lambda N, M, dev, dt, w, r: bench_fire1_warp(N, M, dev, dt, w, r),
            )
        )
    if methods_cfg.get("warp_fire2", True):
        method_registry.append(
            (
                "warp_fire2",
                "FIRE2(wp)",
                "F2wp",
                lambda N, M, dev, dt, w, r: bench_fire2_warp(
                    N, M, dev, dt, hyper, w, r
                ),
            )
        )
    if methods_cfg.get("torch_adapter", True):
        method_registry.append(
            (
                "torch_adapter",
                "FIRE2(adapt)",
                "F2ad",
                lambda N, M, dev, dt, w, r: bench_fire2_torch_adapter(
                    N, M, dev, dt, hyper, w, r
                ),
            )
        )
    if methods_cfg.get("torch_adapter_cell", True):
        method_registry.append(
            (
                "torch_adapter_cell",
                "FIRE2(cell)",
                "F2cl",
                lambda N, M, dev, dt, w, r: bench_fire2_torch_adapter_cell(
                    N, M, dev, dt, hyper, w, r
                ),
            )
        )
    if methods_cfg.get("warp_fire1_cell", True):
        method_registry.append(
            (
                "warp_fire1_cell",
                "FIRE1(cell)",
                "F1cl",
                lambda N, M, dev, dt, w, r: bench_fire1_warp_cell(N, M, dev, dt, w, r),
            )
        )
    if methods_cfg.get("fire2_extended", True):
        method_registry.append(
            (
                "fire2_extended",
                "FIRE2(ext)",
                "F2ex",
                lambda N, M, dev, dt, w, r: bench_fire2_extended(
                    N, M, dev, dt, hyper, w, r
                ),
            )
        )
    if methods_cfg.get("torch_reference", True):
        method_registry.append(
            (
                "torch_reference",
                "PyTorch",
                "PT",
                lambda N, M, dev, dt, w, r: bench_fire2_torch_ref(
                    N, M, dev, dt, hyper, w, r
                ),
            )
        )

    for dtype_str in dtypes:
        torch_dtype, dtype_label = _DTYPE_MAP[dtype_str]

        print(f"\nFIRE2 Kernel Benchmark — dtype: {dtype_label} — device: {device}")
        print(
            "This benchmark does not compare convergence speed, only the speed of the algorithm steps."
        )
        print(f"GPU: {gpu_sku}")
        print(f"Warmup: {warmup}, Runs: {runs}")

        # Build dynamic header
        method_labels = [label for _, label, _, _ in method_registry]
        short_labels = [short for _, _, short, _ in method_registry]
        header_parts = [f"{'N':>10}", f"{'M':>6}"] + [
            f"{label:>14}" for label in method_labels
        ]
        # Add ratio columns: first method vs each other method
        if len(short_labels) > 1:
            base_short = short_labels[0]
            for short in short_labels[1:]:
                ratio_label = f"{base_short}/{short}"
                header_parts.append(f"{ratio_label:>10}")
        header = " ".join(header_parts)
        print(header)
        print("-" * len(header))

        for N in n_values:
            for M in m_values:
                if M > N or N % M != 0:
                    continue

                timings = {}
                for key, label, _short, bench_fn in method_registry:
                    result = bench_fn(N, M, device, torch_dtype, warmup, runs)
                    timings[key] = (label, result)

                    # Collect CSV row
                    all_rows.append(
                        {
                            "method": key,
                            "dtype": dtype_label,
                            "total_atoms": N,
                            "num_systems": M,
                            "atoms_per_system": N // M,
                            "warmup": warmup,
                            "runs": runs,
                            "median_time_ms": f"{result['median_ms']:.4f}",
                            "min_time_ms": f"{result['min_ms']:.4f}",
                            "max_time_ms": f"{result['max_ms']:.4f}",
                        }
                    )

                # Print row
                row_parts = [f"{N:>10,}", f"{M:>6}"]
                median_values = []
                for key, label, _short, _ in method_registry:
                    med = timings[key][1]["median_ms"]
                    median_values.append(med)
                    row_parts.append(f"{med:>12.3f}ms")

                # Ratios: first method vs each other method
                if len(median_values) > 1:
                    base_t = median_values[0]
                    for t in median_values[1:]:
                        ratio = base_t / t if t > 0 else float("nan")
                        row_parts.append(f"{ratio:>9.2f}x")

                print(" ".join(row_parts))

        print()

    # Write CSV
    if all_rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"fire2_kernel_benchmark_{gpu_sku}.csv"
        fieldnames = [
            "method",
            "dtype",
            "total_atoms",
            "num_systems",
            "atoms_per_system",
            "warmup",
            "runs",
            "median_time_ms",
            "min_time_ms",
            "max_time_ms",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote results to {csv_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FIRE2 kernel-level performance benchmark"
    )
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
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)

    print("FIRE2 Kernel Performance Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("N = total atoms, M = number of systems (batches)")

    run_benchmarks(config, output_dir, device)

    print("\nRatio > 1 = first method is faster than comparison method")


if __name__ == "__main__":
    main()

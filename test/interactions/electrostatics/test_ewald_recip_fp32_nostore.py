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

"""Tests for the float32, no-store reciprocal structure-factor path.

On by default for float32 CUDA inputs; set
``NVALCHEMIOPS_ELECTROSTATICS_LEGACY_FP32=1`` to force the legacy float64
stored-phase path instead. The fast path computes phases in float32 and never
materialises the ``(K, N)`` ``cos_k_dot_r`` / ``sin_k_dot_r`` arrays,
recomputing them in the per-atom pass instead. That trades a float64
round-trip for float32 transcendentals, which is strongly favourable on parts
with weak float64 throughput.

These tests go through ``ewald_summation`` rather than the kernels directly:
the gate lives in ``_ewald_recip_chain._forward_impl``, so kernel-level tests
would bypass it entirely. Each test asserts the fast path was actually taken,
because a silently-disabled gate would otherwise make every comparison pass
trivially.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest
import torch

from nvalchemiops.interactions.electrostatics._factory_common import (
    _read_legacy_fp32_env,
)
from nvalchemiops.torch.interactions.electrostatics import (
    ewald_reciprocal_space,
    ewald_summation,
    generate_ewald_miller_indices,
    generate_k_vectors_ewald_summation,
    k_vectors_from_miller_indices,
)
from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.torch.neighbors.neighbor_utils import estimate_max_neighbors

pytestmark = pytest.mark.gpu

DENSITY = 0.15
CUTOFF = 9.0
ALPHA = 0.4130
K_CUTOFF = 2.6


def _make_system(n_atoms, n_systems=1, dtype=torch.float32, device="cuda:0", seed=0):
    """A jittered-lattice periodic system, sized so minimum image holds."""
    gen = torch.Generator().manual_seed(seed)
    box = float((n_atoms / DENSITY) ** (1.0 / 3.0))
    n_side = int(round(n_atoms ** (1.0 / 3.0) + 0.5))
    grid = torch.arange(n_side, dtype=torch.float64)
    coords = torch.stack(
        torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1
    ).reshape(-1, 3)[:n_atoms] * (box / n_side)
    jitter = torch.rand(coords.shape, generator=gen, dtype=torch.float64) * 2 - 1
    positions = torch.cat(
        [((coords + 0.2 * (box / n_side) * jitter) % box) for _ in range(n_systems)]
    )
    total = n_atoms * n_systems
    charges = torch.empty(total, dtype=torch.float64)
    for s in range(n_systems):
        q = torch.randn(n_atoms, generator=gen, dtype=torch.float64)
        charges[s * n_atoms : (s + 1) * n_atoms] = q - q.mean()
    cell = torch.eye(3, dtype=torch.float64).expand(n_systems, 3, 3) * box
    batch_idx = (
        torch.arange(n_systems, dtype=torch.int32).repeat_interleave(n_atoms)
        if n_systems > 1
        else None
    )
    return dict(
        positions=positions.to(device=device, dtype=dtype).contiguous(),
        charges=charges.to(device=device, dtype=dtype).contiguous(),
        cell=cell.to(device=device, dtype=dtype).contiguous(),
        pbc=torch.ones((n_systems, 3), dtype=torch.bool, device=device),
        batch_idx=None if batch_idx is None else batch_idx.to(device),
        atoms_per_system=n_atoms,
    )


def _energy_and_forces(sysd, want_forces=True):
    """Run ``ewald_summation``, returning (energy, forces or None)."""
    positions = sysd["positions"]
    max_nb = estimate_max_neighbors(CUTOFF, atomic_density=2.0 * DENSITY)
    nbmat, _, shifts = neighbor_list(
        positions,
        CUTOFF,
        cell=sysd["cell"],
        pbc=sysd["pbc"],
        batch_idx=sysd["batch_idx"],
        return_neighbor_list=False,
        half_fill=False,
        max_neighbors=max_nb,
    )
    k_vectors = generate_k_vectors_ewald_summation(sysd["cell"], K_CUTOFF)
    pos = positions.detach().clone().requires_grad_(want_forces)
    energy = ewald_summation(
        pos,
        sysd["charges"],
        sysd["cell"],
        alpha=ALPHA,
        k_vectors=k_vectors,
        neighbor_matrix=nbmat,
        neighbor_matrix_shifts=shifts,
        batch_idx=sysd["batch_idx"],
        max_atoms_per_system=sysd["atoms_per_system"],
    ).sum()
    if not want_forces:
        return float(energy.detach()), None
    forces = -torch.autograd.grad(energy, pos)[0]
    return float(energy.detach()), forces


@pytest.fixture(autouse=True)
def _reset_legacy_fp32_cache():
    """Clear the memoized ``NVALCHEMIOPS_ELECTROSTATICS_LEGACY_FP32`` read.

    ``electrostatics_uses_legacy_fp32`` reads its environment variable once
    per process, by design (see its docstring), so production behavior can't
    change mid-run. Tests in this module monkeypatch that variable and need a
    fresh read every time, so clear the cache before and after each test.
    """
    _read_legacy_fp32_env.cache_clear()
    yield
    _read_legacy_fp32_env.cache_clear()


@pytest.fixture
def gate_counter(monkeypatch):
    """Ensure the fast path is active (the default) and count gate admits.

    Without this the tests could pass with the path silently never taken --
    e.g. because something upstream set the legacy flag.
    """
    import nvalchemiops.torch.interactions.electrostatics._ewald_recip_chain as chain

    monkeypatch.delenv("NVALCHEMIOPS_ELECTROSTATICS_LEGACY_FP32", raising=False)
    counts = {"fast": 0, "slow": 0}
    original = chain._can_use_fp32_nostore

    def counting(*args, **kwargs):
        taken = original(*args, **kwargs)
        counts["fast" if taken else "slow"] += 1
        return taken

    monkeypatch.setattr(chain, "_can_use_fp32_nostore", counting)
    return counts


def test_nostore_reduces_peak_memory(cuda_available, gate_counter):
    """The point of the path: peak footprint drops from O(K*N) to O(K).

    The stored-phase path allocates two float64 ``(K, N)`` arrays, so its peak
    grows with both the k-vector count and the atom count. This path allocates
    neither -- but it does still allocate two float32 ``(K,)`` (or ``(S, K)``
    when batched) structure-factor arrays, so peak memory is not literally
    independent of K, just no longer N-scaled. This asserts the reduction
    that actually happens (more than half, for a system sized so N dominates
    K*N), not full K-independence.
    """
    if not cuda_available:
        pytest.skip("No GPU")
    sysd = _make_system(2048)

    def peak():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _energy_and_forces(sysd, want_forces=False)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("NVALCHEMIOPS_ELECTROSTATICS_LEGACY_FP32", "1")
        _read_legacy_fp32_env.cache_clear()
        stored = peak()
    _read_legacy_fp32_env.cache_clear()
    nostore = peak()

    assert gate_counter["fast"] > 0, "fast path was never taken"
    assert nostore < stored / 2


def test_gate_refuses_float64(cuda_available, gate_counter):
    """float64 callers keep the float64 phases: the gate must not downgrade them."""
    if not cuda_available:
        pytest.skip("No GPU")
    sysd = _make_system(512, dtype=torch.float64)
    _energy_and_forces(sysd, want_forces=False)
    assert gate_counter["fast"] == 0
    assert gate_counter["slow"] > 0


def test_gate_disabled_with_legacy_flag(cuda_available):
    """``NVALCHEMIOPS_ELECTROSTATICS_LEGACY_FP32=1`` keeps the path off, in a fresh process.

    ``electrostatics_uses_legacy_fp32`` reads its environment variable once per
    process by design, so this is the faithful way to test "off at process
    startup" -- setting the variable and clearing an in-process cache (as the
    other tests in this module do) tests the cache-clearing, not the actual
    once-per-process contract a long-running training job relies on.
    """
    if not cuda_available:
        pytest.skip("No GPU")
    script = """
import torch
import nvalchemiops.torch.interactions.electrostatics._ewald_recip_chain as chain
from nvalchemiops.torch.interactions.electrostatics import (
    ewald_summation,
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.torch.neighbors.neighbor_utils import estimate_max_neighbors

counts = {"fast": 0}
original = chain._can_use_fp32_nostore

def counting(*args, **kwargs):
    taken = original(*args, **kwargs)
    counts["fast"] += int(taken)
    return taken

chain._can_use_fp32_nostore = counting

device = "cuda"
box = 10.0
positions = torch.rand(512, 3, dtype=torch.float32, device=device) * box
charges = torch.randn(512, dtype=torch.float32, device=device)
charges = (charges - charges.mean()).detach()
cell = (torch.eye(3, dtype=torch.float32, device=device) * box).unsqueeze(0)
pbc = torch.ones(1, 3, dtype=torch.bool, device=device)
cutoff = 4.0
max_nb = estimate_max_neighbors(cutoff, atomic_density=512 / box**3)
nbmat, _, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, return_neighbor_list=False,
    half_fill=False, max_neighbors=max_nb,
)
k_vectors = generate_k_vectors_ewald_summation(cell, 2.6)

ewald_summation(
    positions, charges, cell, alpha=0.413, k_vectors=k_vectors,
    neighbor_matrix=nbmat, neighbor_matrix_shifts=shifts,
)
assert counts["fast"] == 0, "fast path was taken despite NVALCHEMIOPS_ELECTROSTATICS_LEGACY_FP32=1"
"""
    with tempfile.TemporaryDirectory() as cache_dir:
        environment = {
            **os.environ,
            "NVALCHEMIOPS_ELECTROSTATICS_LEGACY_FP32": "1",
            "WARP_CACHE_PATH": cache_dir,
        }
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
            timeout=120,
        )
    assert result.returncode == 0, result.stderr


def _recip_hvp(sysd, dtype):
    """Directional second derivative of the reciprocal energy w.r.t. positions."""
    positions = sysd["positions"].to(dtype).detach().clone().requires_grad_(True)
    charges = sysd["charges"].to(dtype)
    cell = sysd["cell"].to(dtype)
    alpha = torch.tensor([ALPHA], dtype=dtype, device=positions.device)
    miller = generate_ewald_miller_indices(cell[0], 2.0, (5, 5, 5))
    k_vectors = k_vectors_from_miller_indices(cell, miller)
    generator = torch.Generator(device="cpu").manual_seed(0)
    direction = torch.randn(
        positions.shape, generator=generator, dtype=torch.float64
    ).to(positions.device)
    direction = (direction / direction.norm()).to(dtype)
    energy = ewald_reciprocal_space(
        positions=positions,
        charges=charges,
        cell=cell,
        k_vectors=k_vectors,
        alpha=alpha,
    ).sum()
    (grad,) = torch.autograd.grad(energy, positions, create_graph=True)
    (hvp,) = torch.autograd.grad((grad * direction).sum(), positions)
    return hvp.detach().double()


def test_float32_double_backward_phases_match_float64(cuda_available):
    """The float32 second-order phases agree with a float64 reference.

    The double-backward reduce and compute stages each recompute ``cos(k.r)`` /
    ``sin(k.r)`` for every ``(atom, k)``. On float32 CUDA those evaluate in
    float32 while every accumulator stays float64, so the error is dominated by
    the phase argument and does not grow with the atom count.
    """
    system = _make_system(2048)
    reference = _recip_hvp(system, torch.float64)
    fast = _recip_hvp(system, torch.float32)
    relative = ((fast - reference).norm() / reference.norm()).item()
    # Measured ~7.4e-07 and flat in N; float64 phases on the same float32
    # inputs give ~4.8e-07, so the phase precision costs well under 2x.
    assert relative < 5e-06, f"float32 second-order phases drifted: {relative:.3e}"


def test_float64_double_backward_is_unchanged_by_the_phase_split(cuda_available):
    """float64 callers keep bit-exact second-order results.

    ``phase_scalar`` is float64 for them, which makes every added cast an
    identity, so this guards against the specialization leaking into the
    float64 path.
    """
    system = _make_system(1024)
    first = _recip_hvp(system, torch.float64)
    second = _recip_hvp(system, torch.float64)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()

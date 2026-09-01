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

Enabled by ``NVALCHEMIOPS_EWALD_RECIP_FP32_SF``. It computes the phases in
float32 and never materialises the ``(K, N)`` ``cos_k_dot_r`` / ``sin_k_dot_r``
arrays, recomputing them in the per-atom pass instead. That trades a float64
round-trip for float32 transcendentals, which is strongly favourable on parts
with weak float64 throughput.

These tests go through ``ewald_summation`` rather than the kernels directly:
the gate lives in ``_ewald_recip_chain._forward_impl``, so kernel-level tests
would bypass it entirely. Each test asserts the fast path was actually taken,
because a silently-disabled gate would otherwise make every comparison pass
trivially.
"""

from __future__ import annotations

import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    ewald_summation,
    generate_k_vectors_ewald_summation,
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


@pytest.fixture
def gate_counter(monkeypatch):
    """Enable the fast path and count how often the gate admits a call.

    Without this the tests could pass with the path silently never taken.
    """
    import nvalchemiops.torch.interactions.electrostatics._ewald_recip_chain as chain

    monkeypatch.setenv("NVALCHEMIOPS_EWALD_RECIP_FP32_SF", "1")
    counts = {"fast": 0, "slow": 0}
    original = chain._can_use_fp32_nostore

    def counting(*args, **kwargs):
        taken = original(*args, **kwargs)
        counts["fast" if taken else "slow"] += 1
        return taken

    monkeypatch.setattr(chain, "_can_use_fp32_nostore", counting)
    return counts


@pytest.mark.parametrize("n_systems", [1, 4], ids=["single", "batched"])
def test_energy_matches_default_path(cuda_available, gate_counter, n_systems):
    """float32 no-store energy agrees with the float64 stored-phase path."""
    if not cuda_available:
        pytest.skip("No GPU")
    sysd = _make_system(1024, n_systems=n_systems)

    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("NVALCHEMIOPS_EWALD_RECIP_FP32_SF", raising=False)
        reference, _ = _energy_and_forces(sysd, want_forces=False)

    fast, _ = _energy_and_forces(sysd, want_forces=False)

    assert gate_counter["fast"] > 0, "fast path was never taken"
    assert abs(fast - reference) / abs(reference) < 1e-5


@pytest.mark.parametrize("n_systems", [1, 4], ids=["single", "batched"])
def test_forces_match_default_path(cuda_available, gate_counter, n_systems):
    """Forces agree too -- the recompute path must serve derivatives, not just E."""
    if not cuda_available:
        pytest.skip("No GPU")
    sysd = _make_system(1024, n_systems=n_systems)

    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("NVALCHEMIOPS_EWALD_RECIP_FP32_SF", raising=False)
        _, reference = _energy_and_forces(sysd)

    _, fast = _energy_and_forces(sysd)

    assert gate_counter["fast"] > 0, "fast path was never taken"
    scale = reference.abs().max().clamp_min(1e-30)
    assert float((fast - reference).abs().max() / scale) < 1e-5


def test_peak_memory_is_independent_of_k(cuda_available, gate_counter):
    """The point of the path: footprint must stop scaling with K.

    The stored-phase path allocates two float64 ``(K, N)`` arrays, so its peak
    grows with the k-vector count; this path allocates neither.
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
        mp.delenv("NVALCHEMIOPS_EWALD_RECIP_FP32_SF", raising=False)
        stored = peak()
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


def test_gate_disabled_by_default(cuda_available, monkeypatch):
    """Absent the environment variable the path must stay off."""
    if not cuda_available:
        pytest.skip("No GPU")
    import nvalchemiops.torch.interactions.electrostatics._ewald_recip_chain as chain

    monkeypatch.delenv("NVALCHEMIOPS_EWALD_RECIP_FP32_SF", raising=False)
    counts = {"fast": 0}
    original = chain._can_use_fp32_nostore

    def counting(*args, **kwargs):
        taken = original(*args, **kwargs)
        counts["fast"] += int(taken)
        return taken

    monkeypatch.setattr(chain, "_can_use_fp32_nostore", counting)
    _energy_and_forces(_make_system(512), want_forces=False)
    assert counts["fast"] == 0

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

"""Integration tests for the unified multipole electrostatics API: public energy/
feature entry points, the capability/accuracy matrix, torch.compile, and batching."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    multipole_electrostatic_energy,
    multipole_electrostatic_features,
    multipole_scf_step_energy,
    multipole_scf_step_features,
    pack_multipole_moments,
    prepare_multipole_scf_cache,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    infer_l_max,
    pack_charges_dipoles,
    split_multipole_moments,
)
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_electrostatics import (
    multipole_reciprocal_space_energy,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
    multipole_ewald_summation,
    multipole_real_space_energy,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (
    multipole_real_space_quadrupole_energy,
)
from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
    multipole_particle_mesh_ewald,
)
from nvalchemiops.torch.math import FIELD_CONSTANT
from nvalchemiops.torch.math.gto import NormMode


def _es_torch_device(device: str) -> str:
    """Map conftest's ``"cuda:0"`` to PyTorch's ``"cuda"`` form."""
    return "cuda" if "cuda" in device else "cpu"


def _random_system(
    *,
    seed: int = 0,
    n_atoms: int = 8,
    box_len: float = 6.0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    with_dipoles: bool = True,
) -> dict:
    """Build a small neutral random periodic system for parity tests."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, box_len, size=(n_atoms, 3))
    charges = rng.uniform(-1.0, 1.0, n_atoms)
    charges -= charges.mean()
    dipoles = rng.standard_normal((n_atoms, 3)) * 0.3
    cell = np.eye(3) * box_len

    out = {
        "positions": torch.from_numpy(positions).to(device=device, dtype=dtype),
        "charges": torch.from_numpy(charges).to(device=device, dtype=dtype),
        "cell": torch.from_numpy(cell).to(device=device, dtype=dtype),
        "positions_np": positions,
        "charges_np": charges,
        "cell_np": cell,
    }
    if with_dipoles:
        out["dipoles"] = torch.from_numpy(dipoles).to(device=device, dtype=dtype)
        out["dipoles_np"] = dipoles
    else:
        out["dipoles"] = None
        out["dipoles_np"] = np.zeros_like(dipoles)
    return out


class TestBasics:
    """Shape / dtype / device invariants for the public entry point."""

    def test_returns_scalar_float64(self, device):
        td = _es_torch_device(device)
        sys = _random_system(seed=0, n_atoms=4, device=td, dtype=torch.float64)
        energy = multipole_electrostatic_energy(
            sys["positions"],
            pack_charges_dipoles(sys["charges"], sys["dipoles"]),
            sys["cell"],
            sigma=1.0,
            k_cutoff=4.0,
        )
        assert energy.shape == (4,)
        assert energy.dtype == torch.float64
        assert energy.device.type == td

    def test_float32_positions_give_float64_energy(self, device):
        td = _es_torch_device(device)
        sys = _random_system(seed=1, n_atoms=4, device=td, dtype=torch.float32)
        energy = multipole_electrostatic_energy(
            sys["positions"],
            pack_charges_dipoles(sys["charges"], sys["dipoles"]),
            sys["cell"],
            sigma=1.0,
            k_cutoff=4.0,
        )
        assert energy.sum().dtype == torch.float64

    def test_charges_only_matches_zero_dipole_branch(self, device):
        """Passing ``dipoles=None`` matches passing an explicit zero array."""
        td = _es_torch_device(device)
        sys = _random_system(seed=2, n_atoms=5, device=td, dtype=torch.float64)
        zero_dipoles = torch.zeros_like(sys["dipoles"])

        e_none = multipole_electrostatic_energy(
            sys["positions"],
            pack_charges_dipoles(sys["charges"], None),
            sys["cell"],
            sigma=1.0,
            k_cutoff=4.0,
        )
        e_zero = multipole_electrostatic_energy(
            sys["positions"],
            pack_charges_dipoles(sys["charges"], zero_dipoles),
            sys["cell"],
            sigma=1.0,
            k_cutoff=4.0,
        )
        assert torch.allclose(e_none.sum(), e_zero.sum(), rtol=1e-14, atol=1e-14)

    def test_include_self_interaction_adds_back_self_energy(self, device):
        """include_self_interaction=True equals the other path plus 0.5·E_self."""
        td = _es_torch_device(device)
        sys = _random_system(seed=3, n_atoms=4, device=td, dtype=torch.float64)
        source_feats = pack_charges_dipoles(sys["charges"], sys["dipoles"])
        e_with = multipole_electrostatic_energy(
            sys["positions"],
            source_feats,
            sys["cell"],
            sigma=1.0,
            k_cutoff=4.0,
            include_self_interaction=True,
        )
        e_without = multipole_electrostatic_energy(
            sys["positions"],
            source_feats,
            sys["cell"],
            sigma=1.0,
            k_cutoff=4.0,
            include_self_interaction=False,
        )
        # E_self is positive, so include_self=False gives the smaller energy.
        assert float(e_with.sum()) != float(e_without.sum())
        assert float(e_with.sum()) > float(e_without.sum())


class TestValidation:
    """User-facing input validation errors."""

    def test_bad_positions_shape(self):
        with pytest.raises(ValueError, match="positions"):
            multipole_electrostatic_energy(
                torch.zeros(5),
                torch.zeros((5, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64),
                sigma=1.0,
                k_cutoff=4.0,
            )

    def test_source_feats_wrong_length(self):
        with pytest.raises(ValueError, match="multipole_moments"):
            multipole_electrostatic_energy(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((3, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                k_cutoff=4.0,
            )

    def test_source_feats_wrong_last_dim(self):
        with pytest.raises(ValueError, match="source_feats|last-dim"):
            multipole_electrostatic_energy(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 5), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                k_cutoff=4.0,
            )

    def test_bad_cell_shape(self):
        with pytest.raises(ValueError, match="cell"):
            multipole_electrostatic_energy(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 1), dtype=torch.float64),
                torch.eye(4, dtype=torch.float64),
                sigma=1.0,
                k_cutoff=4.0,
            )

    def test_non_positive_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            multipole_electrostatic_energy(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=0.0,
                k_cutoff=4.0,
            )

    def test_non_positive_k_cutoff(self):
        with pytest.raises(ValueError, match="k_cutoff"):
            multipole_electrostatic_energy(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                k_cutoff=-1.0,
            )

    def test_missing_both_k_cutoff_and_k_vectors(self):
        """At least one of ``k_cutoff`` / ``k_vectors`` must be supplied."""
        with pytest.raises(ValueError, match="k_vectors"):
            multipole_electrostatic_energy(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
            )

    def test_k_vectors_bad_shape(self):
        with pytest.raises(ValueError, match="k_vectors"):
            multipole_electrostatic_energy(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                k_vectors=torch.zeros(5, dtype=torch.float64),
            )


class TestNormalizationAndKVectors:
    """Normalization handling and precomputed-k-vector equivalence."""

    def test_precomputed_k_vectors_match_internal_generation(self, device):
        """Pre-supplying the same k-grid the binding would build internally gives the same energy."""
        td = _es_torch_device(device)
        sys = _random_system(seed=13, n_atoms=6, box_len=5.0, device=td)
        k_cutoff = 3.5

        source_feats = pack_charges_dipoles(sys["charges"], sys["dipoles"])
        e_internal = multipole_electrostatic_energy(
            sys["positions"],
            source_feats,
            sys["cell"],
            sigma=1.0,
            k_cutoff=k_cutoff,
        )

        # Same k-grid, fed in explicitly.
        k_half = generate_k_vectors_ewald_summation(sys["cell"], k_cutoff)
        k_vecs = torch.cat([k_half.new_zeros((1, 3)), k_half], dim=0).to(
            dtype=torch.float64
        )
        e_external = multipole_electrostatic_energy(
            sys["positions"],
            source_feats,
            sys["cell"],
            sigma=1.0,
            k_vectors=k_vecs,
        )
        assert float(e_internal.sum()) == float(e_external.sum())

    def test_accepts_string_and_int_normalize(self, device):
        """``normalize=`` accepts ``NormMode``, int, and lowercase/uppercase strings."""
        td = _es_torch_device(device)
        sys = _random_system(seed=11, n_atoms=5, device=td, dtype=torch.float64)
        kwargs = dict(
            positions=sys["positions"],
            multipole_moments=pack_charges_dipoles(sys["charges"], sys["dipoles"]),
            cell=sys["cell"],
            sigma=1.0,
            k_cutoff=3.5,
        )
        e_enum = multipole_electrostatic_energy(**kwargs, normalize=NormMode.RECEIVER)
        e_int = multipole_electrostatic_energy(
            **kwargs, normalize=int(NormMode.RECEIVER)
        )
        e_str_lower = multipole_electrostatic_energy(**kwargs, normalize="receiver")
        e_str_upper = multipole_electrostatic_energy(**kwargs, normalize="RECEIVER")
        assert (
            float(e_enum.sum())
            == float(e_int.sum())
            == float(e_str_lower.sum())
            == float(e_str_upper.sum())
        )


class TestPhysicalInvariants:
    """Physical sanity checks that don't depend on an external reference."""

    def test_translation_invariance(self, device):
        """Rigidly translating all atoms leaves the energy unchanged (PBC)."""
        td = _es_torch_device(device)
        sys = _random_system(seed=5, n_atoms=6, box_len=5.0, device=td)
        shift = torch.tensor([1.3, -0.7, 2.1], dtype=torch.float64, device=td)
        source_feats = pack_charges_dipoles(sys["charges"], sys["dipoles"])
        e_before = multipole_electrostatic_energy(
            sys["positions"],
            source_feats,
            sys["cell"],
            sigma=1.0,
            k_cutoff=3.5,
        )
        e_after = multipole_electrostatic_energy(
            sys["positions"] + shift,
            source_feats,
            sys["cell"],
            sigma=1.0,
            k_cutoff=3.5,
        )
        # |ρ(k)|² is phase-invariant, so the energy agrees to float64 noise.
        np.testing.assert_allclose(
            float(e_before.sum()), float(e_after.sum()), rtol=1e-12, atol=1e-13
        )

    def test_zero_moments_give_zero_energy(self, device):
        """All-zero charges and dipoles → ``E = 0`` exactly."""
        td = _es_torch_device(device)
        sys = _random_system(seed=9, n_atoms=4, box_len=5.0, device=td)
        zero_q = torch.zeros_like(sys["charges"])
        zero_d = torch.zeros_like(sys["dipoles"])
        e = multipole_electrostatic_energy(
            sys["positions"],
            pack_charges_dipoles(zero_q, zero_d),
            sys["cell"],
            sigma=1.0,
            k_cutoff=3.5,
        )
        assert float(e.sum()) == 0.0


def _quadrupole_fixture(seed: int, n_atoms: int, L: float, device: str):
    """Neutral random system with detraced (traceless) symmetric quadrupoles."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, L, size=(n_atoms, 3))
    q = rng.normal(size=n_atoms)
    q -= q.mean()
    mu = rng.normal(size=(n_atoms, 3)) * 0.3
    Qr = rng.normal(size=(n_atoms, 3, 3)) * 0.1
    Q = 0.5 * (Qr + Qr.transpose(0, 2, 1))
    Q -= (np.trace(Q, axis1=1, axis2=2) / 3.0)[:, None, None] * np.eye(3)
    cell = np.eye(3) * L
    return pos, q, mu, Q, cell


def _pathb_reference_quadrupole(pos, q, mu, Q, cell, sigma, kcut):
    """Path-B l=2 oracle: numpy GTO direct-k reciprocal reduction (envelope
    ``exp(-σ²k²)``, no Ewald damping) minus the analytic GTO self overlap.
    """
    F = float(FIELD_CONSTANT)
    pi32 = math.pi**1.5
    V = float(abs(np.linalg.det(cell)))
    G = 2.0 * np.pi * np.linalg.inv(cell).T
    nmax = int(np.ceil(kcut / np.linalg.norm(G, axis=1).min())) + 1
    ks = [
        a * G[0] + b * G[1] + c * G[2]
        for a in range(-nmax, nmax + 1)
        for b in range(-nmax, nmax + 1)
        for c in range(-nmax, nmax + 1)
    ]
    ks = np.array([k for k in ks if 0.0 < np.linalg.norm(k) <= kcut])
    e_recip = 0.0
    for k in ks:
        k2 = float(k @ k)
        e = np.exp(-1j * (pos @ k))
        rho = (q * e).sum() - 1j * ((mu @ k) * e).sum()
        rho -= 0.5 * (np.einsum("a,nab,b->n", k, Q, k) * e).sum()
        e_recip += (
            (4.0 * math.pi / k2)
            * math.exp(-k2 * sigma**2)
            * (rho * rho.conjugate()).real
        )
    e_recip = (F / (4.0 * math.pi)) * 0.5 / V * e_recip
    e_self = (
        (F / (8.0 * pi32 * sigma)) * (q**2).sum()
        + (F / (48.0 * pi32 * sigma**3)) * (mu**2).sum()
        # Cartesian-Frobenius |Q|_F² self uses denom 320 (the angular
        # ⟨(k̂·Q·k̂)²⟩ = (2/15)|Q|_F² factor 3/2 turns 480 into 320).
        + (F / (320.0 * pi32 * sigma**5)) * (Q**2).sum()
    )
    return e_recip - e_self


class TestQuadrupole:
    """Path-B l_max=2 Cartesian-quadrupole energy."""

    def test_matches_reciprocal_reference(self, device):
        """Full Path-B l=2 energy matches recip-reference − analytic self."""
        td = _es_torch_device(device)
        sigma, kcut = 0.5, 16.0
        pos, q, mu, Q, cell = _quadrupole_fixture(7, 6, 6.0, td)
        mm = pack_multipole_moments(
            torch.tensor(q, device=td),
            torch.tensor(mu, device=td),
            torch.tensor(Q, device=td),
        )
        E = multipole_electrostatic_energy(
            torch.tensor(pos, device=td),
            mm,
            torch.tensor(cell, device=td),
            sigma=sigma,
            k_cutoff=kcut,
        )
        E_ref = _pathb_reference_quadrupole(pos, q, mu, Q, cell, sigma, kcut)
        assert abs(float(E.sum()) - E_ref) / abs(E_ref) < 1e-8

    def test_zero_quadrupole_matches_dipole(self, device):
        """Zero Q (l=2 packed) is bit-close to the l=1 packed energy."""
        td = _es_torch_device(device)
        pos, q, mu, _, cell = _quadrupole_fixture(3, 5, 6.0, td)
        pos_t = torch.tensor(pos, device=td)
        cell_t = torch.tensor(cell, device=td)
        Qz = torch.zeros((pos.shape[0], 3, 3), dtype=torch.float64, device=td)
        kw = dict(sigma=0.5, k_cutoff=12.0)
        e_l1 = multipole_electrostatic_energy(
            pos_t,
            pack_multipole_moments(
                torch.tensor(q, device=td), torch.tensor(mu, device=td)
            ),
            cell_t,
            **kw,
        )
        e_l2 = multipole_electrostatic_energy(
            pos_t,
            pack_multipole_moments(
                torch.tensor(q, device=td), torch.tensor(mu, device=td), Qz
            ),
            cell_t,
            **kw,
        )
        assert abs(float(e_l2.sum()) - float(e_l1.sum())) < 1e-9 * max(
            abs(float(e_l1.sum())), 1.0
        )

    def test_grads_match_fd(self, device):
        """∂E/∂{pos,q,μ,Q} via autograd match central finite differences."""
        td = _es_torch_device(device)
        sigma, kcut = 0.5, 12.0
        pos, q, mu, Q, cell = _quadrupole_fixture(11, 4, 6.0, td)
        cell_t = torch.tensor(cell, device=td)

        def energy(pos_, q_, mu_, Q_):
            mm = pack_multipole_moments(q_, mu_, Q_)
            return multipole_electrostatic_energy(
                pos_, mm, cell_t, sigma=sigma, k_cutoff=kcut
            ).sum()

        pos_t = torch.tensor(pos, device=td, requires_grad=True)
        q_t = torch.tensor(q, device=td, requires_grad=True)
        mu_t = torch.tensor(mu, device=td, requires_grad=True)
        Q_t = torch.tensor(Q, device=td, requires_grad=True)
        E = energy(pos_t, q_t, mu_t, Q_t)
        gpos, gq, gmu, gQ = torch.autograd.grad(E, [pos_t, q_t, mu_t, Q_t])
        # ∂E/∂Q must be symmetric (Q enters only via k·Q·k).
        assert (gQ - gQ.transpose(-1, -2)).abs().max() < 1e-12

        h = 1e-6
        base = (
            torch.tensor(pos, device=td),
            torch.tensor(q, device=td),
            torch.tensor(mu, device=td),
            torch.tensor(Q, device=td),
        )

        def fd(idx, shape):
            out = torch.zeros(shape, dtype=torch.float64, device=td)
            flat = out.view(-1)
            b0 = base[idx]
            for i in range(flat.numel()):
                args_p, args_m = list(base), list(base)
                bp = b0.clone()
                bp.view(-1)[i] += h
                bm = b0.clone()
                bm.view(-1)[i] -= h
                args_p[idx], args_m[idx] = bp, bm
                flat[i] = (float(energy(*args_p)) - float(energy(*args_m))) / (2 * h)
            return out

        N = pos.shape[0]
        assert (gpos - fd(0, (N, 3))).abs().max() / gpos.abs().max() < 1e-5
        assert (gq - fd(1, (N,))).abs().max() / gq.abs().max() < 1e-5
        assert (gmu - fd(2, (N, 3))).abs().max() / gmu.abs().max() < 1e-5
        # Symmetric-Q FD: symmetrize the per-component FD before comparing.
        fd_Q = fd(3, (N, 3, 3))
        fd_Q = 0.5 * (fd_Q + fd_Q.transpose(-1, -2))
        assert (gQ - fd_Q).abs().max() / fd_Q.abs().max() < 1e-5


_SIGMA = 0.5


_ALPHA = 0.6


_BOX = 6.0


_RSIG = (1.0, 1.5)


_SIGMA_C = math.sqrt(_SIGMA**2 + 1.0 / (4.0 * _ALPHA**2))


_RCUT = 12.0 * _SIGMA_C  # generous → fixed neighbor list valid under FD


_KCUT = 7.0 / _SIGMA_C


_MESH = (48, 48, 48)


_FD_EPS = 1e-6


_FD_RTOL = 5e-4


_FD_ATOL = 1e-6


ENTRIES = ("directk", "features", "ewald", "pme")


LMAXES = (0, 1, 2)


MODES = ("single", "batch")


def _torch_device(device: str) -> str:
    return "cuda" if "cuda" in device else "cpu"


def _pack(charges, dipoles, quad, l_max):
    """Packed e3nn ``multipole_moments`` of width ``(l_max+1)**2``."""
    if l_max == 0:
        return charges.unsqueeze(-1).contiguous()
    if l_max == 1:
        return pack_multipole_moments(charges, dipoles)
    return pack_multipole_moments(charges, dipoles, quad)


def _rand_moments(n, l_max, rng, td):
    ch = rng.standard_normal(n)
    ch -= ch.mean()  # neutral (reciprocal q-q needs it)
    charges = torch.tensor(ch, device=td)
    dipoles = torch.tensor(0.4 * rng.standard_normal((n, 3)), device=td)
    A = rng.standard_normal((n, 3, 3))
    quad = torch.tensor(0.5 * (A + A.transpose(0, 2, 1)), device=td)
    return _pack(charges, dipoles, quad, l_max)


def _neigh(pos_np, L, cutoff, atom_offset=0):
    """O(N²·shells) CSR neighbor list for one cubic cell (test-only)."""
    n = pos_np.shape[0]
    shell = int(math.ceil(cutoff / L)) + 1
    idx, counts, shifts = [], [], []
    for i in range(n):
        c = 0
        for a in range(-shell, shell + 1):
            for b in range(-shell, shell + 1):
                for cc in range(-shell, shell + 1):
                    for j in range(n):
                        if i == j and (a, b, cc) == (0, 0, 0):
                            continue
                        d = pos_np[j] - pos_np[i] + np.array([a, b, cc]) * L
                        if np.linalg.norm(d) < cutoff:
                            idx.append(j + atom_offset)
                            shifts.append([a, b, cc])
                            c += 1
        counts.append(c)
    return idx, counts, shifts


def _build_system(mode, l_max, td, seed=0):
    """Return base tensors + (for Ewald/PME) a fixed CSR neighbor list + batch_idx."""
    rng = np.random.default_rng(seed)
    if mode == "single":
        sizes = [3]
    else:
        sizes = [2, 3]
    pos_list, mm_list, cells, bidx = [], [], [], []
    idx, ptr, sh = [], [0], []
    off = 0
    for b, n in enumerate(sizes):
        p = rng.uniform(0.0, _BOX, (n, 3))
        pos_list.append(p)
        cells.append(np.eye(3) * _BOX)
        bidx += [b] * n
        mm_list.append(_rand_moments(n, l_max, rng, td))
        i_b, c_b, s_b = _neigh(p, _BOX, _RCUT, atom_offset=off)
        idx += i_b
        ptr += list(np.cumsum(c_b) + (ptr[-1]))
        sh += s_b
        off += n
    pos = torch.tensor(np.concatenate(pos_list), device=td)
    mm = torch.cat(mm_list)
    out = {
        "pos": pos,
        "mm": mm,
        "idx_j": torch.tensor(idx, dtype=torch.int32, device=td),
        "neighbor_ptr": torch.tensor(ptr, dtype=torch.int32, device=td),
        "unit_shifts": torch.tensor(sh, dtype=torch.int32, device=td).reshape(-1, 3),
    }
    if mode == "single":
        out["cell"] = torch.tensor(cells[0], device=td)
        out["batch_idx"] = None
    else:
        out["cell"] = torch.tensor(np.stack(cells), device=td)
        out["batch_idx"] = torch.tensor(bidx, dtype=torch.int32, device=td)
    return out


def _make_value_fn(entry, mode, l_max, sys):
    """Return ``value(pos, mm, cell) -> scalar`` for one matrix cell.

    Energy entries reduce to ``E.sum()``; the feature tensor is contracted with
    fixed sin-weights so the matrix harness is uniform.
    """
    bidx = sys["batch_idx"]
    idx_j, ptr, sh = sys["idx_j"], sys["neighbor_ptr"], sys["unit_shifts"]
    fml = l_max  # receiver cap matches source for the matrix

    def _feat_weights(f):
        w = torch.sin(
            torch.arange(f.numel(), device=f.device, dtype=torch.float64)
        ).reshape(f.shape)
        return (f * w).sum()

    if entry == "directk":
        if mode == "single":

            def value(pos, mm, cell):
                return multipole_electrostatic_energy(
                    pos, mm, cell, sigma=_SIGMA, k_cutoff=_KCUT
                ).sum()
        else:

            def value(pos, mm, cell):
                return multipole_electrostatic_energy(
                    pos, mm, cell, batch_idx=bidx, sigma=_SIGMA, k_cutoff=_KCUT
                ).sum()
    elif entry == "features":
        if mode == "single":

            def value(pos, mm, cell):
                f = multipole_electrostatic_features(
                    pos,
                    mm,
                    cell,
                    sigma=_SIGMA,
                    receiver_sigmas=list(_RSIG),
                    k_cutoff=_KCUT,
                    feature_max_l=fml,
                )
                return _feat_weights(f)
        else:

            def value(pos, mm, cell):
                f = multipole_electrostatic_features(
                    pos,
                    mm,
                    cell,
                    batch_idx=bidx,
                    sigma=_SIGMA,
                    receiver_sigmas=list(_RSIG),
                    k_cutoff=_KCUT,
                    feature_max_l=fml,
                )
                return _feat_weights(f)
    elif entry == "ewald":

        def value(pos, mm, cell):
            return multipole_ewald_summation(
                pos,
                mm,
                cell,
                idx_j,
                ptr,
                sh,
                sigma=_SIGMA,
                alpha=_ALPHA,
                k_cutoff=_KCUT,
                batch_idx=bidx,
            ).sum()
    elif entry == "pme":

        def value(pos, mm, cell):
            return multipole_particle_mesh_ewald(
                pos,
                mm,
                cell,
                idx_j,
                ptr,
                sh,
                sigma=_SIGMA,
                alpha=_ALPHA,
                mesh_dimensions=_MESH,
                batch_idx=bidx,
            ).sum()
    else:  # pragma: no cover
        raise ValueError(entry)
    return value


def _fd_grad(scalar_fn, x):
    """Central finite-difference gradient of scalar_fn w.r.t. tensor x (float64)."""
    base = x.detach().clone()
    fd = torch.zeros_like(base)
    flat = base.reshape(-1)
    for i in range(flat.numel()):
        xp = base.clone().reshape(-1)
        xp[i] += _FD_EPS
        xm = base.clone().reshape(-1)
        xm[i] -= _FD_EPS
        fp = float(scalar_fn(xp.reshape(base.shape)))
        fm = float(scalar_fn(xm.reshape(base.shape)))
        fd.reshape(-1)[i] = (fp - fm) / (2.0 * _FD_EPS)
    return fd


_FEATURE_STRESS_XFAIL = "features ∂/∂cell create-free 1st-order may be partial; tracked"


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("l_max", LMAXES)
class TestCapabilityMatrix:
    """One class instance per (entry, mode, l_max) cell."""

    def test_energy_finite(self, entry, mode, l_max, device):
        td = _torch_device(device)
        sys = _build_system(mode, l_max, td, seed=1)
        value = _make_value_fn(entry, mode, l_max, sys)
        v = value(sys["pos"], sys["mm"], sys["cell"])
        assert torch.isfinite(v).all()
        assert v.ndim == 0  # we reduced everything to a scalar

    def test_grad_positions_fd(self, entry, mode, l_max, device):
        """Forces: ∂value/∂positions vs central FD."""
        td = _torch_device(device)
        sys = _build_system(mode, l_max, td, seed=2)
        value = _make_value_fn(entry, mode, l_max, sys)
        mm, cell = sys["mm"], sys["cell"]
        p = sys["pos"].clone().requires_grad_(True)
        (g,) = torch.autograd.grad(value(p, mm, cell), p)
        fd = _fd_grad(lambda x: value(x, mm, cell), sys["pos"])
        torch.testing.assert_close(g, fd, rtol=_FD_RTOL, atol=_FD_ATOL)

    def test_grad_moments_fd(self, entry, mode, l_max, device):
        """Value-loss training: ∂value/∂multipole_moments vs central FD."""
        td = _torch_device(device)
        sys = _build_system(mode, l_max, td, seed=3)
        value = _make_value_fn(entry, mode, l_max, sys)
        pos, cell = sys["pos"], sys["cell"]
        m = sys["mm"].clone().requires_grad_(True)
        (g,) = torch.autograd.grad(value(pos, m, cell), m)
        fd = _fd_grad(lambda x: value(pos, x, cell), sys["mm"])
        torch.testing.assert_close(g, fd, rtol=_FD_RTOL, atol=_FD_ATOL)

    def test_force_loss_create_graph_fd(self, entry, mode, l_max, device, request):
        """Force-loss training: d(||d value/d pos||^2)/d moments vs FD (2nd-order)."""
        td = _torch_device(device)
        sys = _build_system(mode, l_max, td, seed=4)
        value = _make_value_fn(entry, mode, l_max, sys)
        pos, cell = sys["pos"], sys["cell"]

        def force_loss(m):
            p = pos.clone().requires_grad_(True)
            (forces,) = torch.autograd.grad(value(p, m, cell), p, create_graph=True)
            return (forces**2).sum()

        m = sys["mm"].clone().requires_grad_(True)
        (g,) = torch.autograd.grad(force_loss(m), m)
        fd = _fd_grad(lambda x: float(force_loss(x)), sys["mm"])
        torch.testing.assert_close(g, fd, rtol=_FD_RTOL, atol=1e-5)

    def test_grad_cell_fd(self, entry, mode, l_max, device, request):
        """Stress / virial: d value/d cell vs central FD (fixed topological nlist)."""
        td = _torch_device(device)
        sys = _build_system(mode, l_max, td, seed=5)
        value = _make_value_fn(entry, mode, l_max, sys)
        pos, mm = sys["pos"], sys["mm"]
        c = sys["cell"].clone().requires_grad_(True)
        (g,) = torch.autograd.grad(value(pos, mm, c), c)
        fd = _fd_grad(lambda x: value(pos, mm, x), sys["cell"])
        torch.testing.assert_close(g, fd, rtol=_FD_RTOL, atol=_FD_ATOL)


@pytest.mark.parametrize("l_max", LMAXES)
class TestCrossMethodPhysics:
    """Ewald/PME/direct-k must agree, and the Ewald total must be α-independent."""

    def _system(self, td, seed=11):
        rng = np.random.default_rng(seed)
        n = 4
        pos_np = rng.uniform(0.0, _BOX, (n, 3))
        return pos_np, rng

    def test_ewald_alpha_independent(self, l_max, device):
        td = _torch_device(device)
        rng = np.random.default_rng(20 + l_max)
        n = 4
        pos_np = rng.uniform(0.0, _BOX, (n, 3))
        pos = torch.tensor(pos_np, device=td)
        cell = torch.tensor(np.eye(3) * _BOX, device=td)
        mm = _rand_moments(n, l_max, rng, td)
        totals = []
        for alpha in (0.4, 0.6, 0.9):
            sc = math.sqrt(_SIGMA**2 + 1.0 / (4.0 * alpha**2))
            idx, cnt, sh = _neigh(pos_np, _BOX, 12.0 * sc)
            ptr = [0] + list(np.cumsum(cnt))
            totals.append(
                float(
                    multipole_ewald_summation(
                        pos,
                        mm,
                        cell,
                        torch.tensor(idx, dtype=torch.int32, device=td),
                        torch.tensor(ptr, dtype=torch.int32, device=td),
                        torch.tensor(sh, dtype=torch.int32, device=td).reshape(-1, 3),
                        sigma=_SIGMA,
                        alpha=alpha,
                        k_cutoff=7.0 / sc,
                    ).sum()
                )
            )
        assert max(totals) - min(totals) < 1e-4, (
            f"l={l_max} Ewald total α-dependent: {totals}"
        )

    def test_ewald_pme_directk_agree(self, l_max, device):
        td = _torch_device(device)
        rng = np.random.default_rng(30 + l_max)
        n = 4
        pos_np = rng.uniform(0.0, _BOX, (n, 3))
        pos = torch.tensor(pos_np, device=td)
        cell = torch.tensor(np.eye(3) * _BOX, device=td)
        mm = _rand_moments(n, l_max, rng, td)
        idx, cnt, sh = _neigh(pos_np, _BOX, _RCUT)
        ptr = [0] + list(np.cumsum(cnt))
        ij = torch.tensor(idx, dtype=torch.int32, device=td)
        pt = torch.tensor(ptr, dtype=torch.int32, device=td)
        st = torch.tensor(sh, dtype=torch.int32, device=td).reshape(-1, 3)

        e_b = float(
            multipole_electrostatic_energy(
                pos, mm, cell, sigma=_SIGMA, k_cutoff=_KCUT
            ).sum()
        )
        e_ewald = float(
            multipole_ewald_summation(
                pos,
                mm,
                cell,
                ij,
                pt,
                st,
                sigma=_SIGMA,
                alpha=_ALPHA,
                k_cutoff=_KCUT,
            ).sum()
        )
        e_pme = float(
            multipole_particle_mesh_ewald(
                pos,
                mm,
                cell,
                ij,
                pt,
                st,
                sigma=_SIGMA,
                alpha=_ALPHA,
                mesh_dimensions=_MESH,
            ).sum()
        )

        # Ewald == direct-k to convergence floor; PME == Ewald to MESH accuracy
        # (the l=2 mesh form factor needs a finer grid for tight agreement —
        # 2e-3 is the honest PME-vs-exact-reciprocal tolerance at mesh=48).
        assert abs(e_ewald - e_b) / abs(e_b) < 1e-4, (e_ewald, e_b)
        assert abs(e_pme - e_ewald) / abs(e_ewald) < 2e-3, (e_pme, e_ewald)


def _build_test_system(*, seed: int, n_atoms: int, box_len: float, device: str):
    rng = np.random.default_rng(seed)
    positions = torch.from_numpy(rng.uniform(0.0, box_len, size=(n_atoms, 3))).to(
        device=device, dtype=torch.float64
    )
    charges_np = rng.uniform(-1.0, 1.0, n_atoms)
    charges_np -= charges_np.mean()
    charges = torch.from_numpy(charges_np).to(device=device, dtype=torch.float64)
    dipoles = torch.from_numpy(rng.standard_normal((n_atoms, 3)) * 0.3).to(
        device=device, dtype=torch.float64
    )
    cell = torch.eye(3, dtype=torch.float64, device=device) * box_len
    source_feats = pack_charges_dipoles(charges, dipoles)
    return positions, charges, dipoles, cell, source_feats


@pytest.mark.slow
class TestTorchCompile:
    r"""``torch.compile`` smoke tests — the custom op should appear as a single opaque node and numerics must match eager.

    Marked ``slow``: these exercise the full Inductor codegen (expensive under
    coverage). The 0-graph-break contract stays in the default lane via
    ``TestStepGraphBreaks`` (``torch._dynamo.explain``, no codegen).
    """

    def test_compile_scf_step_energy(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=0, n_atoms=5, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], k_cutoff=3.5
        )

        def step(sf):
            return multipole_scf_step_energy(cache, positions, sf).sum()

        e_eager = step(source_feats)
        compiled = torch.compile(step, fullgraph=False)
        e_compiled = compiled(source_feats)
        # Tolerate <=1-ULP float64 summation-order drift from graph breaks at
        # the Warp-op boundary.
        np.testing.assert_allclose(
            float(e_eager), float(e_compiled), rtol=1e-14, atol=1e-14
        )

    def test_compile_scf_step_features(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=1, n_atoms=4, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[0.7, 1.3],
            k_cutoff=3.5,
        )

        def step(sf):
            return multipole_scf_step_features(cache, positions, sf)

        f_eager = step(source_feats)
        compiled = torch.compile(step, fullgraph=False)
        f_compiled = compiled(source_feats)
        # Tolerate <=1-ULP float64 summation-order drift.
        np.testing.assert_allclose(
            f_eager.detach().cpu().numpy(),
            f_compiled.detach().cpu().numpy(),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_compile_one_shot_energy(self, device):
        """One-shot binding survives ``torch.compile`` with <=1-ULP drift.

        The one-shot path rebuilds the SCF cache on every call (including a
        scipy ``compute_overlap_constants`` call that causes dynamo graph
        breaks), so a few trailing ULPs of summation-order noise slip in. The
        step-level tests above reuse a pre-built cache and stay bit-exact.
        """
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=2, n_atoms=4, box_len=5.0, device=td
        )

        def fn(sf):
            return multipole_electrostatic_energy(
                positions,
                sf,
                cell,
                sigma=1.0,
                k_cutoff=3.5,
            ).sum()

        e_eager = fn(source_feats)
        compiled = torch.compile(fn, fullgraph=False)
        e_compiled = compiled(source_feats)
        np.testing.assert_allclose(
            float(e_eager), float(e_compiled), rtol=1e-12, atol=1e-14
        )

    def test_compile_one_shot_features(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=3, n_atoms=4, box_len=5.0, device=td
        )

        def fn(sf):
            return multipole_electrostatic_features(
                positions,
                sf,
                cell,
                sigma=1.0,
                receiver_sigmas=[0.8, 1.2],
                k_cutoff=3.5,
            )

        f_eager = fn(source_feats)
        compiled = torch.compile(fn, fullgraph=False)
        f_compiled = compiled(source_feats)
        np.testing.assert_allclose(
            f_eager.detach().cpu().numpy(),
            f_compiled.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-14,
        )


class TestAutogradEnergy:
    """``multipole_scf_step_energy`` autograd tests.

    Analytical moment gradients flow through ``MultipoleRhoFunction``: the
    rho-pipeline contribution reaches ``source_feats`` alongside the
    self-interaction term; the position gradient is wired separately.
    """

    def test_forward_with_requires_grad_does_not_raise(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=0, n_atoms=5, box_len=5.0, device=td
        )
        sf = source_feats.clone().requires_grad_(True)
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], k_cutoff=3.5
        )
        e = multipole_scf_step_energy(cache, positions, sf)
        # requires_grad: the self-interaction torch subtract combines the
        # detached raw energy with grad-tracking charge/dipole terms.
        assert e.sum().requires_grad

    @pytest.mark.parametrize("seed", [11, 23, 31])
    def test_gradcheck_source_feats(self, device, seed):
        r"""``gradcheck`` on ``source_feats``.

        The full gradient is the sum of the self-interaction torch term and
        the rho-pipeline term; ``gradcheck`` verifies both together vs FD.
        """
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=seed, n_atoms=4, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], k_cutoff=3.5
        )

        def fn(sf):
            return multipole_scf_step_energy(cache, positions, sf).sum()

        sf = source_feats.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(fn, (sf,), eps=1e-6, atol=1e-4)

    @pytest.mark.parametrize("seed", [11, 23, 31])
    def test_gradcheck_positions(self, device, seed):
        r"""``gradcheck`` on positions.

        The position gradient flows via
        ``_position_gradient_from_rhok_kernel`` (analytical backward, one Warp
        launch). source_feats is held fixed; the next test does the joint check.
        """
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=seed, n_atoms=4, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], k_cutoff=3.5
        )

        def fn(p):
            return multipole_scf_step_energy(cache, p, source_feats).sum()

        p = positions.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(fn, (p,), eps=1e-6, atol=1e-4)

    @pytest.mark.parametrize("seed", [11, 23, 31])
    def test_gradcheck_joint(self, device, seed):
        r"""``gradcheck`` passes on all inputs simultaneously.

        Confirms that ``MultipoleRhoFunction``'s backward produces
        consistent gradients for both slots (positions, source_feats)
        under the same cotangent path, not just any one at a time.
        """
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=seed, n_atoms=4, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], k_cutoff=3.5
        )

        def fn(p, sf):
            return multipole_scf_step_energy(cache, p, sf).sum()

        p = positions.clone().requires_grad_(True)
        sf = source_feats.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(fn, (p, sf), eps=1e-6, atol=1e-4)

    def test_positions_grad_flows_through_one_shot_binding(self, device):
        """The ``multipole_electrostatic_energy`` one-shot binding also reports a position gradient."""
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=17, n_atoms=4, box_len=5.0, device=td
        )
        p = positions.clone().requires_grad_(True)
        e = multipole_electrostatic_energy(
            p, source_feats, cell, sigma=1.0, k_cutoff=3.5
        )
        e.sum().backward()
        assert p.grad is not None
        assert p.grad.shape == positions.shape
        assert float(p.grad.abs().max()) > 0.0

    def test_warp_pipeline_contributes_to_gradients(self, device):
        r"""The full gradient differs from the self-interaction term alone.

        Regression anchor: fails if a future refactor drops the analytical
        backward and leaves the Warp-pipeline branch detached.
        """
        td = _torch_device(device)
        positions, charges, dipoles, cell, source_feats = _build_test_system(
            seed=11, n_atoms=6, box_len=5.0, device=td
        )
        sf = source_feats.clone().requires_grad_(True)
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], k_cutoff=3.5
        )
        e = multipole_scf_step_energy(cache, positions, sf)
        e.sum().backward()
        # Extract charge / dipole(Cartesian) gradients from source_feats.grad.
        # sph layout: [:, 0] = charge, [:, 1:4] = (mu_y, mu_z, mu_x).
        # Cartesian (x, y, z) dipole grads live at columns [3, 1, 2].
        chg_grad = sf.grad.detach()[..., 0]
        dip_grad_cart = sf.grad.detach()[..., [3, 1, 2]]
        # Self-interaction contribution alone.
        self_int_c = -cache.source_overlap_constants[0].detach() * charges.detach()
        self_int_d = -cache.source_overlap_constants[1].detach() * dipoles.detach()
        # The full gradient must include a non-zero Warp-pipeline term
        # on top of the self-interaction.
        warp_contrib_c = float((chg_grad - self_int_c).abs().max())
        warp_contrib_d = float((dip_grad_cart - self_int_d).abs().max())
        assert warp_contrib_c > 1e-6, (
            f"Warp contribution to charges.grad unexpectedly zero: "
            f"max |Δ| = {warp_contrib_c:.2e}"
        )
        assert warp_contrib_d > 1e-6, (
            f"Warp contribution to dipoles.grad unexpectedly zero: "
            f"max |Δ| = {warp_contrib_d:.2e}"
        )

    def test_include_self_interaction_true_output_still_requires_grad(self, device):
        r"""With ``include_self_interaction=True``, the output still requires grad.

        The rho-pipeline flows through ``MultipoleRhoFunction``, so the output
        is autograd-connected regardless of the self-interaction subtract.
        """
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=23, n_atoms=4, box_len=5.0, device=td
        )
        sf = source_feats.clone().requires_grad_(True)
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], k_cutoff=3.5
        )
        e = multipole_scf_step_energy(
            cache, positions, sf, include_self_interaction=True
        )
        assert e.sum().requires_grad
        e.sum().backward()
        assert sf.grad is not None
        assert float(sf.grad.abs().max()) > 0.0


class TestAutogradFeatures:
    """``multipole_scf_step_features`` autograd tests.

    Analytical gradients flow through the feature projection:
    ``MultipoleProjectRawFeaturesFunction`` handles d/dV and d/dr;
    ``MultipoleRhoFunction`` handles d/d source_feats through the rho->V
    chain; the self-interaction subtract and output permutation are
    autograd-native torch ops.
    """

    def test_output_requires_grad(self, device):
        """The feature output is autograd-connected to its inputs."""
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=0, n_atoms=4, box_len=5.0, device=td
        )
        sf = source_feats.clone().requires_grad_(True)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[0.8, 1.2],
            k_cutoff=3.5,
        )
        f = multipole_scf_step_features(cache, positions, sf)
        assert f.requires_grad

    def test_one_shot_features_autograd_connected(self, device):
        """The one-shot binding also produces a grad-tracking feature tensor."""
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=4, n_atoms=4, box_len=5.0, device=td
        )
        sf = source_feats.clone().requires_grad_(True)
        f = multipole_electrostatic_features(
            positions,
            sf,
            cell,
            sigma=1.0,
            receiver_sigmas=[0.8, 1.2],
            k_cutoff=3.5,
        )
        assert f.requires_grad

    @pytest.mark.parametrize("seed", [11, 23, 31])
    def test_gradcheck_features(self, device, seed):
        r"""``gradcheck`` on (positions, source_feats) for features.

        Exercises the full autograd path: MultipoleRhoFunction -> torch
        per_k_factor multiply -> MultipoleProjectRawFeaturesFunction -> torch
        self-int subtract -> torch index_select permutation.
        """
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=seed, n_atoms=3, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[0.8, 1.2], k_cutoff=3.5
        )

        def fn(p, sf):
            return multipole_scf_step_features(cache, p, sf)

        p = positions.clone().requires_grad_(True)
        sf = source_feats.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(fn, (p, sf), eps=1e-6, atol=1e-4)

    def test_feature_backward_matches_one_shot(self, device):
        """scf_step_features and multipole_electrostatic_features give the same gradients."""
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=7, n_atoms=4, box_len=5.0, device=td
        )
        sigma = 1.0
        receiver_sigmas = [0.8, 1.2]
        k_cutoff = 3.5
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            k_cutoff=k_cutoff,
        )

        p_step = positions.clone().requires_grad_(True)
        sf_step = source_feats.clone().requires_grad_(True)
        f_step = multipole_scf_step_features(cache, p_step, sf_step)
        f_step.sum().backward()

        p_one = positions.clone().requires_grad_(True)
        sf_one = source_feats.clone().requires_grad_(True)
        f_one = multipole_electrostatic_features(
            p_one,
            sf_one,
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            k_cutoff=k_cutoff,
        )
        f_one.sum().backward()

        np.testing.assert_allclose(
            p_step.grad.detach().cpu().numpy(),
            p_one.grad.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            sf_step.grad.detach().cpu().numpy(),
            sf_one.grad.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-13,
        )


class TestAutogradOneShotEnergy:
    """One-shot energy binding inherits the step's autograd behavior."""

    def test_one_shot_energy_backward_matches_step(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=5, n_atoms=5, box_len=5.0, device=td
        )
        sf_one = source_feats.clone().requires_grad_(True)
        sf_step = source_feats.clone().requires_grad_(True)

        e_one = multipole_electrostatic_energy(
            positions,
            sf_one,
            cell,
            sigma=1.0,
            k_cutoff=3.5,
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], k_cutoff=3.5
        )
        e_step = multipole_scf_step_energy(cache, positions, sf_step)
        e_one.sum().backward()
        e_step.sum().backward()
        np.testing.assert_allclose(
            sf_one.grad.detach().cpu().numpy(),
            sf_step.grad.detach().cpu().numpy(),
            rtol=1e-14,
            atol=1e-14,
        )


@pytest.mark.slow
class TestCompileAutograd:
    """``torch.compile``-wrapped autograd still produces the same gradients."""

    def test_compiled_scf_step_energy_backward(self, device):
        """``torch.compile(step)`` + ``.backward()`` reproduces eager gradients on all inputs.

        Exercises the ``torch.autograd.Function`` boundary under
        ``torch.compile`` specifically: the compiled graph must
        register the Function as an opaque differentiable primitive
        and replay its backward when ``.backward()`` is called on the
        compiled output. Regression anchor: if Dynamo changes how
        autograd.Functions are handled, this test catches it.
        """
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=7, n_atoms=5, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], k_cutoff=3.5
        )

        def step(p, sf):
            return multipole_scf_step_energy(cache, p, sf).sum()

        p_eager = positions.clone().requires_grad_(True)
        sf_eager = source_feats.clone().requires_grad_(True)
        e_eager = step(p_eager, sf_eager)
        e_eager.backward()

        compiled = torch.compile(step, fullgraph=False)
        p_compiled = positions.clone().requires_grad_(True)
        sf_compiled = source_feats.clone().requires_grad_(True)
        e_compiled = compiled(p_compiled, sf_compiled)
        e_compiled.backward()

        # rtol=1e-12 tolerates <=1-ULP graph-break reordering in the cos/sin
        # fresh-compute path.
        np.testing.assert_allclose(
            p_eager.grad.detach().cpu().numpy(),
            p_compiled.grad.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            sf_eager.grad.detach().cpu().numpy(),
            sf_compiled.grad.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-14,
        )


class TestDoubleBackward:
    r"""Second-order autograd: ``create_graph=True`` on d E/d r so MLIP losses
    of the form ``l = w(E) + w(F) + w(S)`` flow gradients back to source_feats
    (and cell via cell autograd) through the full position gradient.

    The second-order derivative kernels are verified vs FD at the kernel level
    elsewhere; here we check that the autograd glue composes correctly.
    """

    @staticmethod
    def _system(device: str, n_atoms: int = 4, box_len: float = 6.0, seed: int = 13):
        return _build_test_system(
            seed=seed, n_atoms=n_atoms, box_len=box_len, device=device
        )

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
    )
    def test_gradgradcheck_energy_moments(self, device):
        """``gradgradcheck`` on d E/d source_feats — the moment-grad double-backward path."""
        positions, _, _, cell, source_feats = self._system(device)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.5,
            receiver_sigmas=[1.5],
            k_cutoff=1.5,
            l_max=1,
        )
        pos = positions.clone()  # detached — not the variable we differentiate here
        sf = source_feats.clone().requires_grad_(True)

        def f(sf_):
            return multipole_scf_step_energy(cache, pos, sf_).sum()

        assert torch.autograd.gradgradcheck(f, (sf,), eps=1e-6, atol=1e-4)

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
    )
    def test_gradgradcheck_energy_positions(self, device):
        """``gradgradcheck`` on d E/d r — the position-Hessian path."""
        positions, _, _, cell, source_feats = self._system(device)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.5,
            receiver_sigmas=[1.5],
            k_cutoff=1.5,
            l_max=1,
        )
        sf = source_feats.clone()
        p = positions.clone().requires_grad_(True)

        def f(p_):
            return multipole_scf_step_energy(cache, p_, sf).sum()

        assert torch.autograd.gradgradcheck(f, (p,), eps=1e-6, atol=1e-4)

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
    )
    def test_force_loss_backprop_to_moments(self, device):
        """End-to-end MLIP-style: ``forces = -d E/d r`` with
        ``create_graph=True``; then ``force_loss.backward()`` must flow
        gradients back to source_feats (the d F/d theta = -d^2 E/(d r d theta) path).
        """
        positions, _, _, cell, source_feats = self._system(device)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.5,
            receiver_sigmas=[1.5],
            k_cutoff=1.5,
            l_max=1,
        )
        sf = source_feats.clone().requires_grad_(True)
        p = positions.clone().requires_grad_(True)

        energy = multipole_scf_step_energy(cache, p, sf).sum()
        (forces_neg,) = torch.autograd.grad(energy, p, create_graph=True)
        forces = -forces_neg
        # Any scalar loss on forces — check non-zero gradient to source_feats.
        loss = (forces * forces).sum()
        (grad_sf,) = torch.autograd.grad(loss, (sf,), retain_graph=False)
        assert grad_sf.abs().max().item() > 0.0, (
            "force loss produced no source_feats gradient"
        )

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
    )
    def test_force_loss_backprop_to_positions(self, device):
        """Position Hessian diagonal: d F/d r. Cheap smoke check — require a
        non-zero gradient and that the pipeline doesn't raise."""
        positions, _, _, cell, source_feats = self._system(device)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.5,
            receiver_sigmas=[1.5],
            k_cutoff=1.5,
            l_max=1,
        )
        sf = source_feats.clone()
        p = positions.clone().requires_grad_(True)

        energy = multipole_scf_step_energy(cache, p, sf).sum()
        (forces_neg,) = torch.autograd.grad(energy, p, create_graph=True)
        loss = forces_neg.pow(2).sum()
        (gp,) = torch.autograd.grad(loss, (p,))
        assert gp.abs().max().item() > 0.0

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
    )
    def test_stress_via_cell_autograd(self, device):
        """First-order stress: ``-d E/d cell`` via cell autograd. The cache
        carries autograd through ``(source_phi_hat, receiver_phi_hat,
        per_k_factor, volume)``, so a gradient w.r.t. ``cell`` materializes
        without any manual virial computation."""
        positions, _, _, _cell, source_feats = self._system(device)
        box_len = 6.0
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .mul_(box_len)
            .requires_grad_(True)
        )
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.5,
            receiver_sigmas=[1.5],
            k_cutoff=1.5,
            l_max=1,
        )
        energy = multipole_scf_step_energy(cache, positions, source_feats).sum()
        (gc,) = torch.autograd.grad(energy, cell)
        assert gc.abs().max().item() > 0.0, "expected nonzero ∂E/∂cell"

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
    )
    def test_features_force_like_double_backward(self, device):
        """Feature-step variant of the force-loss path."""
        positions, _, _, cell, source_feats = self._system(device)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.5,
            receiver_sigmas=[1.5],
            k_cutoff=1.5,
            l_max=1,
        )
        sf = source_feats.clone().requires_grad_(True)
        p = positions.clone().requires_grad_(True)

        feats = multipole_scf_step_features(cache, p, sf)
        # A positions-depending scalar: Σ_{i,col} feats[i, col].
        s = feats.sum()
        (gp,) = torch.autograd.grad(s, p, create_graph=True)
        # Scalar loss that's a function of the position gradient of the
        # feature sum; backprop to source_feats.
        loss = gp.pow(2).sum()
        (grad_sf,) = torch.autograd.grad(loss, (sf,))
        assert grad_sf.abs().max().item() > 0.0


_TD = "cpu"


_KCUT = 7.0


def _pos(n=3):
    return torch.tensor(np.random.default_rng(0).uniform(0.0, _BOX, (n, 3)), device=_TD)


def _charges(n=3):
    q = np.random.default_rng(1).standard_normal(n)
    q -= q.mean()
    return torch.tensor(q, device=_TD).unsqueeze(-1).contiguous()


def _cell():
    return torch.tensor(np.eye(3) * _BOX, device=_TD)


class TestReciprocalValidation:
    def test_bad_positions(self):
        with pytest.raises(ValueError, match="positions must be"):
            multipole_reciprocal_space_energy(
                torch.zeros(3, 2, device=_TD),
                _charges(),
                _cell(),
                sigma=0.5,
                alpha=0.6,
                k_cutoff=_KCUT,
            )

    def test_bad_moments(self):
        with pytest.raises(ValueError, match="multipole_moments must be"):
            multipole_reciprocal_space_energy(
                _pos(),
                torch.zeros(5, 1, device=_TD),
                _cell(),
                sigma=0.5,
                alpha=0.6,
                k_cutoff=_KCUT,
            )

    def test_bad_cell(self):
        with pytest.raises(ValueError, match="cell must be"):
            multipole_reciprocal_space_energy(
                _pos(),
                _charges(),
                torch.zeros(2, 2, device=_TD),
                sigma=0.5,
                alpha=0.6,
                k_cutoff=_KCUT,
            )

    def test_nonpositive_sigma(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            multipole_reciprocal_space_energy(
                _pos(),
                _charges(),
                _cell(),
                sigma=0.0,
                alpha=0.6,
                k_cutoff=_KCUT,
            )

    def test_nonpositive_alpha(self):
        with pytest.raises(ValueError, match="alpha must be positive"):
            multipole_reciprocal_space_energy(
                _pos(),
                _charges(),
                _cell(),
                sigma=0.5,
                alpha=0.0,
                k_cutoff=_KCUT,
            )

    def test_missing_k_cutoff_and_kvectors(self):
        with pytest.raises(ValueError, match="k_vectors|k_cutoff"):
            multipole_reciprocal_space_energy(
                _pos(),
                _charges(),
                _cell(),
                sigma=0.5,
                alpha=0.6,
            )


class TestBatchedValidation:
    def _bidx(self, n=3):
        return torch.zeros(n, dtype=torch.int32, device=_TD)

    def test_energy_bad_positions(self):
        with pytest.raises(ValueError, match="positions must be"):
            multipole_electrostatic_energy(
                torch.zeros(3, 2, device=_TD),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=self._bidx(),
                sigma=0.5,
                k_cutoff=_KCUT,
            )

    def test_energy_bad_moments(self):
        with pytest.raises(ValueError, match="multipole_moments must be"):
            multipole_electrostatic_energy(
                _pos(),
                torch.zeros(5, 1, device=_TD),
                _cell().unsqueeze(0),
                batch_idx=self._bidx(),
                sigma=0.5,
                k_cutoff=_KCUT,
            )

    def test_energy_bad_cells(self):
        with pytest.raises(ValueError, match="batched cell must be"):
            multipole_electrostatic_energy(
                _pos(),
                _charges(),
                _cell(),
                batch_idx=self._bidx(),
                sigma=0.5,
                k_cutoff=_KCUT,
            )

    def test_energy_bad_batch_idx(self):
        with pytest.raises(ValueError, match="batch_idx must match"):
            multipole_electrostatic_energy(
                _pos(),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=self._bidx(5),
                sigma=0.5,
                k_cutoff=_KCUT,
            )

    def test_reciprocal_bad_cells(self):
        with pytest.raises(ValueError, match="batched cell must be"):
            multipole_reciprocal_space_energy(
                _pos(),
                _charges(),
                _cell(),
                batch_idx=self._bidx(),
                sigma=0.5,
                alpha=0.6,
                k_cutoff=_KCUT,
            )

    def test_reciprocal_bad_batch_idx(self):
        with pytest.raises(ValueError, match="batch_idx must match"):
            multipole_reciprocal_space_energy(
                _pos(),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=self._bidx(5),
                sigma=0.5,
                alpha=0.6,
                k_cutoff=_KCUT,
            )

    def test_reciprocal_nonpositive_alpha(self):
        with pytest.raises(ValueError, match="alpha must be positive"):
            multipole_reciprocal_space_energy(
                _pos(),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=self._bidx(),
                sigma=0.5,
                alpha=0.0,
                k_cutoff=_KCUT,
            )

    def test_reciprocal_bad_positions(self):
        with pytest.raises(ValueError, match="positions must be"):
            multipole_reciprocal_space_energy(
                torch.zeros(3, 2, device=_TD),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=self._bidx(),
                sigma=0.5,
                alpha=0.6,
                k_cutoff=_KCUT,
            )

    def test_reciprocal_bad_moments(self):
        with pytest.raises(ValueError, match="multipole_moments must be"):
            multipole_reciprocal_space_energy(
                _pos(),
                torch.zeros(5, 1, device=_TD),
                _cell().unsqueeze(0),
                batch_idx=self._bidx(),
                sigma=0.5,
                alpha=0.6,
                k_cutoff=_KCUT,
            )


class TestFeaturesValidation:
    def test_bad_positions(self):
        with pytest.raises(ValueError, match="positions must be"):
            multipole_electrostatic_features(
                torch.zeros(3, 2, device=_TD),
                _charges(),
                _cell(),
                sigma=0.5,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
            )

    def test_bad_cell(self):
        with pytest.raises(ValueError, match="cell must be"):
            multipole_electrostatic_features(
                _pos(),
                _charges(),
                torch.zeros(2, 2, device=_TD),
                sigma=0.5,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
            )

    def test_nonpositive_sigma(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            multipole_electrostatic_features(
                _pos(),
                _charges(),
                _cell(),
                sigma=-1.0,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
            )

    def test_bad_moments(self):
        with pytest.raises(ValueError, match="multipole_moments must be"):
            multipole_electrostatic_features(
                _pos(),
                torch.zeros(5, 1, device=_TD),
                _cell(),
                sigma=0.5,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
            )

    def test_bad_feature_max_l(self):
        with pytest.raises(ValueError, match="feature_max_l must be"):
            multipole_electrostatic_features(
                _pos(),
                _charges(),
                _cell(),
                sigma=0.5,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
                feature_max_l=3,
            )

    def test_receiver_sigmas_tensor_empty(self):
        # Exercises the ``isinstance(receiver_sigmas, torch.Tensor)`` branch.
        with pytest.raises(ValueError, match="receiver_sigmas must be non-empty"):
            multipole_electrostatic_features(
                _pos(),
                _charges(),
                _cell(),
                sigma=0.5,
                receiver_sigmas=torch.tensor([], dtype=torch.float64),
                k_cutoff=_KCUT,
            )

    def test_receiver_sigmas_tensor_valid(self):
        # Tensor receiver_sigmas on the happy path (covers the tolist branch end
        # to end). Tiny system → fast on CPU.
        f = multipole_electrostatic_features(
            _pos(),
            _charges(),
            _cell(),
            sigma=0.5,
            receiver_sigmas=torch.tensor([1.0, 1.5], dtype=torch.float64),
            k_cutoff=_KCUT,
        )
        assert torch.isfinite(f).all()

    def test_batch_bad_positions(self):
        with pytest.raises(ValueError, match="positions must be"):
            multipole_electrostatic_features(
                torch.zeros(3, 2, device=_TD),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=torch.zeros(3, dtype=torch.int32, device=_TD),
                sigma=0.5,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
            )

    def test_batch_bad_moments(self):
        with pytest.raises(ValueError, match="multipole_moments must be"):
            multipole_electrostatic_features(
                _pos(),
                torch.zeros(5, 1, device=_TD),
                _cell().unsqueeze(0),
                batch_idx=torch.zeros(3, dtype=torch.int32, device=_TD),
                sigma=0.5,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
            )

    def test_batch_bad_cells(self):
        with pytest.raises(ValueError, match="batched cell must be"):
            multipole_electrostatic_features(
                _pos(),
                _charges(),
                _cell(),
                batch_idx=torch.zeros(3, dtype=torch.int32, device=_TD),
                sigma=0.5,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
            )

    def test_batch_bad_batch_idx(self):
        with pytest.raises(ValueError, match="batch_idx must match"):
            multipole_electrostatic_features(
                _pos(),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=torch.zeros(5, dtype=torch.int32, device=_TD),
                sigma=0.5,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
            )

    def test_batch_receiver_sigmas_tensor_empty(self):
        with pytest.raises(ValueError, match="receiver_sigmas must be non-empty"):
            multipole_electrostatic_features(
                _pos(),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=torch.zeros(3, dtype=torch.int32, device=_TD),
                sigma=0.5,
                receiver_sigmas=torch.tensor([], dtype=torch.float64),
                k_cutoff=_KCUT,
            )

    def test_batch_bad_feature_max_l(self):
        with pytest.raises(ValueError, match="feature_max_l must be"):
            multipole_electrostatic_features(
                _pos(),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=torch.zeros(3, dtype=torch.int32, device=_TD),
                sigma=0.5,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
                feature_max_l=5,
            )


class TestScfCacheValidation:
    def test_nonpositive_sigma(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            prepare_multipole_scf_cache(
                _cell(),
                sigma=0.0,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
            )

    def test_bad_feature_max_l(self):
        with pytest.raises(ValueError, match="feature_max_l must be"):
            prepare_multipole_scf_cache(
                _cell(),
                sigma=0.5,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
                feature_max_l=7,
            )

    def test_nonpositive_alpha(self):
        with pytest.raises(ValueError, match="alpha, when given"):
            prepare_multipole_scf_cache(
                _cell(),
                sigma=0.5,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
                alpha=-1.0,
            )

    def test_receiver_sigmas_tensor_negative(self):
        # Tensor branch + "all positive" guard.
        with pytest.raises(ValueError, match="receiver_sigmas must all be positive"):
            prepare_multipole_scf_cache(
                _cell(),
                sigma=0.5,
                receiver_sigmas=torch.tensor([-1.0], dtype=torch.float64),
                k_cutoff=_KCUT,
            )

    def test_batch_empty_cells(self):
        with pytest.raises(ValueError, match="at least one system"):
            prepare_multipole_scf_cache(
                torch.zeros(0, 3, 3, device=_TD),
                sigma=0.5,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
            )

    def test_batch_receiver_sigmas_empty(self):
        with pytest.raises(ValueError, match="receiver_sigmas must be non-empty"):
            prepare_multipole_scf_cache(
                _cell().unsqueeze(0),
                sigma=0.5,
                receiver_sigmas=[],
                k_cutoff=_KCUT,
            )

    def test_batch_nonpositive_sigma(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            prepare_multipole_scf_cache(
                _cell().unsqueeze(0),
                sigma=0.0,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
            )

    def test_batch_bad_l_max(self):
        with pytest.raises(ValueError, match="l_max must be"):
            prepare_multipole_scf_cache(
                _cell().unsqueeze(0),
                sigma=0.5,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
                l_max=4,
            )

    def test_batch_bad_feature_max_l(self):
        with pytest.raises(ValueError, match="feature_max_l must be"):
            prepare_multipole_scf_cache(
                _cell().unsqueeze(0),
                sigma=0.5,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
                feature_max_l=9,
            )

    def test_batch_bad_k_cutoff(self):
        with pytest.raises(ValueError, match="k_cutoff must be"):
            prepare_multipole_scf_cache(
                _cell().unsqueeze(0),
                sigma=0.5,
                receiver_sigmas=[1.0],
                k_cutoff=-1.0,
            )

    def test_batch_receiver_sigmas_negative(self):
        with pytest.raises(ValueError, match="receiver_sigmas must all be positive"):
            prepare_multipole_scf_cache(
                _cell().unsqueeze(0),
                sigma=0.5,
                receiver_sigmas=torch.tensor([-2.0], dtype=torch.float64),
                k_cutoff=_KCUT,
            )

    def test_batch_alpha_nonpositive(self):
        with pytest.raises(ValueError, match="alpha, when given"):
            prepare_multipole_scf_cache(
                _cell().unsqueeze(0),
                sigma=0.5,
                receiver_sigmas=[1.0],
                k_cutoff=_KCUT,
                alpha=-0.5,
            )


def _halflist(n):
    """Tiny half neighbor list (i<j, zero shift) for an n-atom system."""
    idx, ptr, sh = [], [0], []
    for i in range(n):
        for j in range(i + 1, n):
            idx.append(j)
            sh.append([0, 0, 0])
        ptr.append(len(idx))
    return (
        torch.tensor(idx, dtype=torch.int32, device=_TD),
        torch.tensor(ptr, dtype=torch.int32, device=_TD),
        torch.tensor(sh, dtype=torch.int32, device=_TD).reshape(-1, 3),
    )


class TestEwaldAutoParameters:
    """``multipole_ewald_summation`` with ``alpha``/``k_cutoff`` left None
    triggers ``estimate_multipole_ewald_parameters`` (the auto-param path)."""

    def test_single_system_auto_alpha_and_cutoff(self):
        n = 3
        idx, ptr, sh = _halflist(n)
        e = multipole_ewald_summation(
            _pos(n),
            _charges(n),
            _cell(),
            idx,
            ptr,
            sh,
            sigma=0.5,
        )  # alpha=None, k_cutoff=None → auto-estimate
        assert torch.isfinite(e).all()

    def test_batched_auto_alpha_identical_cells(self):
        # Two identical-cell systems → auto-estimated alpha agrees across the
        # batch, exercising the (B,)-collapse branch.
        n = 2
        pos = torch.cat([_pos(n), _pos(n)])
        mm = torch.cat([_charges(n), _charges(n)])
        cells = torch.stack([_cell(), _cell()])
        bidx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=_TD)
        # neighbor list: within each 2-atom system, one pair.
        idx = torch.tensor([1, 3], dtype=torch.int32, device=_TD)
        ptr = torch.tensor([0, 1, 1, 2, 2], dtype=torch.int32, device=_TD)
        sh = torch.zeros(2, 3, dtype=torch.int32, device=_TD)
        e = multipole_ewald_summation(
            pos,
            mm,
            cells,
            idx,
            ptr,
            sh,
            sigma=0.5,
            batch_idx=bidx,
            k_cutoff=8.0,  # provide kcut; let alpha auto-estimate
        )
        assert torch.isfinite(e).all() and e.shape == (4,)


class TestBatchRealSpaceValidation:
    def _args(self, **over):
        n = 3
        idx, ptr, sh = _halflist(n)
        d = dict(
            positions=_pos(n),
            multipole_moments=_charges(n),
            cells=_cell().unsqueeze(0),
            idx_j=idx,
            neighbor_ptr=ptr,
            unit_shifts=sh,
            sigmas=torch.tensor([0.5], device=_TD),
            alphas=torch.tensor([0.6], device=_TD),
            batch_idx=torch.zeros(n, dtype=torch.int32, device=_TD),
        )
        d.update(over)
        return d

    def _call(self, **over):
        a = self._args(**over)
        return multipole_real_space_energy(
            a["positions"],
            a["multipole_moments"],
            a["cells"],
            a["idx_j"],
            a["neighbor_ptr"],
            a["unit_shifts"],
            a["sigmas"],
            a["alphas"],
            batch_idx=a["batch_idx"],
        )

    def test_bad_positions(self):
        with pytest.raises(ValueError, match="positions must be"):
            self._call(positions=torch.zeros(3, 2, device=_TD))

    def test_bad_moments(self):
        with pytest.raises(ValueError, match="multipole_moments must be"):
            self._call(multipole_moments=torch.zeros(5, 1, device=_TD))

    def test_bad_cells(self):
        with pytest.raises(ValueError, match="cells must be"):
            self._call(cells=_cell())

    def test_bad_alphas(self):
        with pytest.raises(ValueError, match="alphas must be"):
            self._call(alphas=torch.tensor([0.6, 0.7], device=_TD))

    def test_bad_sigmas(self):
        with pytest.raises(ValueError, match="sigmas must be"):
            self._call(sigmas=torch.tensor([0.5, 0.5], device=_TD))

    def test_bad_batch_idx(self):
        with pytest.raises(ValueError, match="batch_idx must match"):
            self._call(batch_idx=torch.zeros(5, dtype=torch.int32, device=_TD))


class TestQuadrupoleRealSpaceValidation:
    def _args(self, **over):
        n = 3
        idx, ptr, sh = _halflist(n)
        d = dict(
            positions=_pos(n),
            charges=_charges(n).squeeze(-1),
            dipoles=torch.zeros(n, 3, device=_TD),
            quadrupoles=torch.zeros(n, 3, 3, device=_TD),
            cell=_cell(),
            idx_j=idx,
            neighbor_ptr=ptr,
            unit_shifts=sh,
            sigma=torch.tensor([0.5], device=_TD),
            alpha=torch.tensor([0.6], device=_TD),
        )
        d.update(over)
        return d

    def _call(self, **over):
        a = self._args(**over)
        return multipole_real_space_quadrupole_energy(
            a["positions"],
            a["charges"],
            a["dipoles"],
            a["quadrupoles"],
            a["cell"],
            a["idx_j"],
            a["neighbor_ptr"],
            a["unit_shifts"],
            a["sigma"],
            a["alpha"],
        )

    def test_bad_positions(self):
        with pytest.raises(ValueError, match="positions must be"):
            self._call(positions=torch.zeros(3, 2, device=_TD))

    def test_bad_cell(self):
        with pytest.raises(ValueError, match="cell must be"):
            self._call(cell=torch.zeros(2, 2, device=_TD))

    def test_bad_dipoles(self):
        with pytest.raises(ValueError, match="dipoles must be"):
            self._call(dipoles=torch.zeros(3, 2, device=_TD))

    def test_bad_quadrupoles(self):
        with pytest.raises(ValueError, match="quadrupoles must be"):
            self._call(quadrupoles=torch.zeros(3, 2, 2, device=_TD))


class TestMomentHelpersValidation:
    def test_infer_l_max_bad_rank(self):
        with pytest.raises(ValueError, match="rank-2"):
            infer_l_max(torch.zeros(4, device=_TD))

    def test_infer_l_max_bad_last_dim(self):
        with pytest.raises(ValueError, match="last-dim must be"):
            infer_l_max(torch.zeros(3, 7, device=_TD))

    def test_split_bad_last_dim(self):
        with pytest.raises(ValueError, match="last-dim must be|rank-2"):
            split_multipole_moments(torch.zeros(3, 5, device=_TD))

    def test_pack_quadrupoles_without_dipoles(self):
        with pytest.raises(ValueError, match="quadrupoles given without dipoles"):
            pack_multipole_moments(
                torch.zeros(3, device=_TD),
                None,
                torch.zeros(3, 3, 3, device=_TD),
            )


def _batch_random_system(seed: int, n_atoms: int, box_len: float, device: str):
    """Build one random system: (positions, charges, dipoles, cell)."""
    rng = np.random.default_rng(seed)
    positions = torch.from_numpy(rng.uniform(0.0, box_len, size=(n_atoms, 3))).to(
        device=device, dtype=torch.float64
    )
    charges_np = rng.uniform(-1.0, 1.0, n_atoms)
    charges_np -= charges_np.mean()
    charges = torch.from_numpy(charges_np).to(device=device, dtype=torch.float64)
    dipoles_np = rng.standard_normal((n_atoms, 3)) * 0.3
    dipoles = torch.from_numpy(dipoles_np).to(device=device, dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64, device=device) * box_len
    return positions, charges, dipoles, cell


def _batch_fixture(device: str, *, seed_base: int = 0):
    """Build a 3-system batch with unequal (N_b, box) shapes.

    Returns a dict bundling both the per-system tensors and the flat
    batched tensors, so tests can exercise both paths.
    """
    td = _torch_device(device)
    sizes = [(6, 4.5), (10, 5.2), (4, 3.8)]
    per_system = [
        _batch_random_system(seed_base + b, n, L, td) for b, (n, L) in enumerate(sizes)
    ]

    positions_flat = torch.cat([s[0] for s in per_system], dim=0)
    charges_flat = torch.cat([s[1] for s in per_system], dim=0)
    dipoles_flat = torch.cat([s[2] for s in per_system], dim=0)
    cells = torch.stack([s[3] for s in per_system], dim=0)

    batch_idx = torch.cat(
        [
            torch.full((sizes[b][0],), b, dtype=torch.int32, device=td)
            for b in range(len(sizes))
        ]
    )
    return {
        "sizes": sizes,
        "per_system": per_system,
        "positions": positions_flat,
        "charges": charges_flat,
        "dipoles": dipoles_flat,
        "cells": cells,
        "batch_idx": batch_idx,
        "device": td,
    }


class TestBatchCache:
    def test_shapes_and_pad_invariants(self, device):
        td = _torch_device(device)
        cells = torch.stack(
            [torch.eye(3, dtype=torch.float64, device=td) * L for L in (4.0, 5.5, 3.5)],
            dim=0,
        )
        cache = prepare_multipole_scf_cache(
            cells, sigma=0.5, receiver_sigmas=[0.7], k_cutoff=3.5, l_max=1
        )
        assert cache.is_batched
        assert cache.batch_size == 3
        assert cache.k_vectors.shape == (3, cache.n_k_max, 3)
        assert cache.source_phi_hat.shape == (3, cache.n_k_max, 4, 2)
        assert cache.receiver_phi_hat.shape == (3, cache.n_k_max, 1, 4, 2)

        # Pad rows carry zero weights.
        k_indices = torch.arange(cache.n_k_max, device=td)
        for b in range(3):
            valid = cache.valid_k_counts[b].item()
            pad_mask = k_indices >= valid
            if pad_mask.any():
                assert torch.all(cache.per_k_factor[b][pad_mask] == 0)
                assert torch.all(cache.k_factor_proj[b][pad_mask] == 0)
                assert torch.all(cache.source_phi_hat[b][pad_mask] == 0)
                assert torch.all(cache.receiver_phi_hat[b][pad_mask] == 0)
                assert torch.all((cache.k_vectors[b][pad_mask] == 0).all(dim=-1))


def _per_system_energies_features(
    batch: dict, *, sigma: float, receiver_sigmas: list[float], k_cut: float
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    energies = []
    features = []
    for positions, charges, dipoles, cell in batch["per_system"]:
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            k_cutoff=k_cut,
            l_max=1,
        )
        source_feats = pack_charges_dipoles(charges, dipoles)
        e = multipole_scf_step_energy(cache, positions, source_feats)
        f = multipole_scf_step_features(cache, positions, source_feats)
        energies.append(e)
        features.append(f)
    return energies, features


class TestBatchStepParity:
    sigma = 0.5
    receiver_sigmas = [0.7]
    k_cut = 3.5

    def test_energy_bit_parity(self, device):
        batch = _batch_fixture(device)
        td = batch["device"]
        cache_b = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            k_cutoff=self.k_cut,
            l_max=1,
        )
        e_b = multipole_scf_step_energy(
            cache_b,
            batch["positions"],
            pack_charges_dipoles(batch["charges"], batch["dipoles"]),
            batch_idx=batch["batch_idx"],
        )
        per_e, _ = _per_system_energies_features(
            batch,
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            k_cut=self.k_cut,
        )
        B = batch["cells"].shape[0]
        assert e_b.shape == (sum(n for n, _ in batch["sizes"]),)
        # Reduce per-atom to per-system for comparison.
        e_b_per_sys = torch.zeros(B, dtype=torch.float64, device=td).scatter_add(
            0, batch["batch_idx"].long(), e_b
        )
        for b, ref in enumerate(per_e):
            torch.testing.assert_close(e_b_per_sys[b], ref.sum(), rtol=0, atol=1e-12)

    def test_features_parity(self, device):
        batch = _batch_fixture(device)
        cache_b = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            k_cutoff=self.k_cut,
            l_max=1,
        )
        f_b = multipole_scf_step_features(
            cache_b,
            batch["positions"],
            pack_charges_dipoles(batch["charges"], batch["dipoles"]),
            batch_idx=batch["batch_idx"],
        )
        _, per_f = _per_system_energies_features(
            batch,
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            k_cut=self.k_cut,
        )
        off = 0
        for b, (n, _) in enumerate(batch["sizes"]):
            torch.testing.assert_close(f_b[off : off + n], per_f[b], rtol=0, atol=5e-13)
            off += n

    def test_charges_only_matches_zero_dipoles(self, device):
        """l_max=0 source_feats must match explicit-zero-dipole l_max=1 source_feats.

        Both paths represent the same physical system (no dipole moment); the
        two code paths go through different caches (l_max=0 vs l_max=1), so
        matching energies verify the l=1 contribution vanishes cleanly when
        the dipole block is zero.
        """
        batch = _batch_fixture(device)
        zeros = torch.zeros_like(batch["dipoles"])

        cache_l0 = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            k_cutoff=self.k_cut,
            l_max=0,
        )
        cache_l1 = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            k_cutoff=self.k_cut,
            l_max=1,
        )
        e_l0 = multipole_scf_step_energy(
            cache_l0,
            batch["positions"],
            pack_charges_dipoles(batch["charges"], None),
            batch_idx=batch["batch_idx"],
        )
        e_l1_zero = multipole_scf_step_energy(
            cache_l1,
            batch["positions"],
            pack_charges_dipoles(batch["charges"], zeros),
            batch_idx=batch["batch_idx"],
        )
        torch.testing.assert_close(e_l0, e_l1_zero, rtol=0, atol=1e-12)


class TestBatchBackwardParity:
    sigma = 0.5
    receiver_sigmas = [0.7]
    k_cut = 3.5

    def test_energy_backward_matches_per_system(self, device):
        """Per-atom gradients of Σ E_b must match per-system ∂E/∂(pos/source_feats)."""
        batch = _batch_fixture(device)

        # Per-system reference grads.
        per_grads = []
        for positions, charges, dipoles, cell in batch["per_system"]:
            p = positions.detach().clone().requires_grad_(True)
            sf = (
                pack_charges_dipoles(charges.detach(), dipoles.detach())
                .clone()
                .requires_grad_(True)
            )
            cache = prepare_multipole_scf_cache(
                cell,
                sigma=self.sigma,
                receiver_sigmas=self.receiver_sigmas,
                k_cutoff=self.k_cut,
                l_max=1,
            )
            e = multipole_scf_step_energy(cache, p, sf)
            e.sum().backward()
            per_grads.append((p.grad, sf.grad))

        # Batched grads.
        p_b = batch["positions"].detach().clone().requires_grad_(True)
        sf_b = (
            pack_charges_dipoles(batch["charges"].detach(), batch["dipoles"].detach())
            .clone()
            .requires_grad_(True)
        )
        cache_b = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            k_cutoff=self.k_cut,
            l_max=1,
        )
        e_vec = multipole_scf_step_energy(
            cache_b, p_b, sf_b, batch_idx=batch["batch_idx"]
        )
        e_vec.sum().backward()

        off = 0
        for b, (n, _) in enumerate(batch["sizes"]):
            gp_ref, gsf_ref = per_grads[b]
            torch.testing.assert_close(
                p_b.grad[off : off + n], gp_ref, rtol=1e-10, atol=1e-10
            )
            torch.testing.assert_close(
                sf_b.grad[off : off + n], gsf_ref, rtol=1e-10, atol=1e-10
            )
            off += n

    def test_features_backward_finite(self, device):
        """Feature-loss backward produces finite grads for positions and source_feats."""
        batch = _batch_fixture(device)
        p = batch["positions"].detach().clone().requires_grad_(True)
        sf = (
            pack_charges_dipoles(batch["charges"].detach(), batch["dipoles"].detach())
            .clone()
            .requires_grad_(True)
        )
        cache_b = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            k_cutoff=self.k_cut,
            l_max=1,
        )
        f = multipole_scf_step_features(cache_b, p, sf, batch_idx=batch["batch_idx"])
        (f**2).sum().backward()
        for t in (p, sf):
            assert torch.isfinite(t.grad).all()
            assert t.grad.abs().sum() > 0


class TestBatchDoubleBackward:
    """Double-backward paths: ensure d(F^2)/dx stays on the autograd tape."""

    sigma = 0.5
    receiver_sigmas = [0.7]
    k_cut = 3.0  # smaller cutoff → cheaper test

    def test_force_loss_backward(self, device):
        batch = _batch_fixture(device)
        p = batch["positions"].detach().clone().requires_grad_(True)
        sf = (
            pack_charges_dipoles(batch["charges"].detach(), batch["dipoles"].detach())
            .clone()
            .requires_grad_(True)
        )
        cache_b = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            k_cutoff=self.k_cut,
            l_max=1,
        )
        e_vec = multipole_scf_step_energy(cache_b, p, sf, batch_idx=batch["batch_idx"])
        e_total = e_vec.sum()
        forces = -torch.autograd.grad(e_total, p, create_graph=True)[0]
        # Force-loss surrogate: simulate matching a zero-force target.
        loss = (forces**2).sum()
        loss.backward()
        for t in (p, sf):
            assert torch.isfinite(t.grad).all()
            assert t.grad.abs().sum() > 0

    def test_feature_force_loss_backward(self, device):
        """Same but via the feature pipeline."""
        batch = _batch_fixture(device)
        p = batch["positions"].detach().clone().requires_grad_(True)
        sf = (
            pack_charges_dipoles(batch["charges"].detach(), batch["dipoles"].detach())
            .clone()
            .requires_grad_(True)
        )
        cache_b = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            k_cutoff=self.k_cut,
            l_max=1,
        )
        f = multipole_scf_step_features(cache_b, p, sf, batch_idx=batch["batch_idx"])
        scalar = f.sum()
        (grad_p,) = torch.autograd.grad(scalar, p, create_graph=True)
        (grad_p**2).sum().backward()
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0


class TestBatchOneShot:
    sigma = 0.5
    receiver_sigmas = [0.7]
    k_cut = 3.5

    def test_energy_matches_scf_step(self, device):
        batch = _batch_fixture(device)
        source_feats = pack_charges_dipoles(batch["charges"], batch["dipoles"])
        e_oneshot = multipole_electrostatic_energy(
            batch["positions"],
            source_feats,
            batch["cells"],
            batch_idx=batch["batch_idx"],
            sigma=self.sigma,
            k_cutoff=self.k_cut,
        )
        cache_b = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=[self.sigma],  # one-shot uses sigma as the only receiver σ
            k_cutoff=self.k_cut,
            l_max=1,
        )
        e_step = multipole_scf_step_energy(
            cache_b,
            batch["positions"],
            source_feats,
            batch_idx=batch["batch_idx"],
        )
        torch.testing.assert_close(e_oneshot, e_step, rtol=0, atol=1e-12)

    def test_features_matches_scf_step(self, device):
        batch = _batch_fixture(device)
        source_feats = pack_charges_dipoles(batch["charges"], batch["dipoles"])
        f_oneshot = multipole_electrostatic_features(
            batch["positions"],
            source_feats,
            batch["cells"],
            batch_idx=batch["batch_idx"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            k_cutoff=self.k_cut,
        )
        cache_b = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            k_cutoff=self.k_cut,
            l_max=1,
        )
        f_step = multipole_scf_step_features(
            cache_b,
            batch["positions"],
            source_feats,
            batch_idx=batch["batch_idx"],
        )
        torch.testing.assert_close(f_oneshot, f_step, rtol=0, atol=1e-12)


class TestBatchValidation:
    def test_bad_cells_shape(self, device):
        td = _torch_device(device)
        cells = torch.zeros((4,), dtype=torch.float64, device=td)  # not (3,3)/(B,3,3)
        with pytest.raises(ValueError, match="cell must be"):
            prepare_multipole_scf_cache(
                cells, sigma=0.5, receiver_sigmas=[0.7], k_cutoff=3.0
            )

    def test_batch_idx_length_mismatch(self, device):
        td = _torch_device(device)
        cells = torch.stack(
            [torch.eye(3, dtype=torch.float64, device=td) * 4.0 for _ in range(2)],
            dim=0,
        )
        cache_b = prepare_multipole_scf_cache(
            cells, sigma=0.5, receiver_sigmas=[0.7], k_cutoff=3.0, l_max=1
        )
        pos = torch.zeros((5, 3), dtype=torch.float64, device=td)
        source_feats = torch.zeros((5, 4), dtype=torch.float64, device=td)
        batch_idx_bad = torch.zeros(4, dtype=torch.int32, device=td)  # wrong length
        with pytest.raises(ValueError, match="batch_idx must be"):
            multipole_scf_step_energy(
                cache_b, pos, source_feats, batch_idx=batch_idx_bad
            )


def _quadrupole_batch_fixture(device: str, *, seed_base: int = 100):
    """3-system batch with per-atom detraced (traceless) symmetric quadrupoles."""
    td = _torch_device(device)
    sizes = [(6, 4.5), (10, 5.2), (4, 3.8)]
    systems = []
    for b, (n, L) in enumerate(sizes):
        rng = np.random.default_rng(seed_base + b)
        pos = rng.uniform(0.0, L, size=(n, 3))
        q = rng.normal(size=n)
        q -= q.mean()
        mu = rng.normal(size=(n, 3)) * 0.3
        Qr = rng.normal(size=(n, 3, 3)) * 0.1
        Q = 0.5 * (Qr + Qr.transpose(0, 2, 1))
        Q -= (np.trace(Q, axis1=1, axis2=2) / 3.0)[:, None, None] * np.eye(3)
        systems.append(
            (
                torch.tensor(pos, device=td),
                torch.tensor(q, device=td),
                torch.tensor(mu, device=td),
                torch.tensor(Q, device=td),
                torch.eye(3, dtype=torch.float64, device=td) * L,
            )
        )
    return td, sizes, systems


class TestBatchQuadrupole:
    """Batched Path-B l=2 energy matches per-system (the validated single path)."""

    sigma = 0.5
    k_cut = 12.0

    def test_energy_matches_per_system(self, device):
        td, sizes, systems = _quadrupole_batch_fixture(device)
        pos = torch.cat([s[0] for s in systems], dim=0)
        mm = pack_multipole_moments(
            torch.cat([s[1] for s in systems], dim=0),
            torch.cat([s[2] for s in systems], dim=0),
            torch.cat([s[3] for s in systems], dim=0),
        )
        cells = torch.stack([s[4] for s in systems], dim=0)
        batch_idx = torch.cat(
            [
                torch.full((sizes[b][0],), b, dtype=torch.int32, device=td)
                for b in range(len(sizes))
            ]
        )
        e_b = multipole_electrostatic_energy(
            pos,
            mm,
            cells,
            batch_idx=batch_idx,
            sigma=self.sigma,
            k_cutoff=self.k_cut,
        )
        N_total = sum(n for n, _ in sizes)
        B = len(sizes)
        assert e_b.shape == (N_total,)
        e_b_per_sys = torch.zeros(B, dtype=torch.float64, device=td).scatter_add(
            0, batch_idx.long(), e_b
        )
        for b, (p, q, mu, Q, cell) in enumerate(systems):
            e_single = multipole_electrostatic_energy(
                p,
                pack_multipole_moments(q, mu, Q),
                cell,
                sigma=self.sigma,
                k_cutoff=self.k_cut,
            )
            torch.testing.assert_close(
                e_b_per_sys[b], e_single.sum(), rtol=1e-9, atol=1e-9
            )

    def test_forces_match_per_system(self, device):
        td, sizes, systems = _quadrupole_batch_fixture(device, seed_base=200)
        pos = torch.cat([s[0] for s in systems], dim=0).requires_grad_(True)
        mm = pack_multipole_moments(
            torch.cat([s[1] for s in systems], dim=0),
            torch.cat([s[2] for s in systems], dim=0),
            torch.cat([s[3] for s in systems], dim=0),
        )
        cells = torch.stack([s[4] for s in systems], dim=0)
        batch_idx = torch.cat(
            [
                torch.full((sizes[b][0],), b, dtype=torch.int32, device=td)
                for b in range(len(sizes))
            ]
        )
        e_b = multipole_electrostatic_energy(
            pos,
            mm,
            cells,
            batch_idx=batch_idx,
            sigma=self.sigma,
            k_cutoff=self.k_cut,
        )
        (g_b,) = torch.autograd.grad(e_b.sum(), pos)
        off = 0
        for b, (p, q, mu, Q, cell) in enumerate(systems):
            n = sizes[b][0]
            p_g = p.clone().requires_grad_(True)
            e_single = multipole_electrostatic_energy(
                p_g,
                pack_multipole_moments(q, mu, Q),
                cell,
                sigma=self.sigma,
                k_cutoff=self.k_cut,
            )
            (g_s,) = torch.autograd.grad(e_single.sum(), p_g)
            torch.testing.assert_close(g_b[off : off + n], g_s, rtol=1e-7, atol=1e-7)
            off += n


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestStepGraphBreaks:
    """Regression guard: the SCF-step hot path must stay torch.compile
    graph-break-free.

    The other compile tests use ``torch.compile(..., fullgraph=False)`` and only
    assert eager-vs-compiled equivalence, so a ``.item()``/device sync added to
    the hot path (a real graph break) would pass silently. These tests use
    ``torch._dynamo.explain`` to assert *zero* user-visible graph breaks across
    single + batched systems and l = 0/1/2.
    """

    @pytest.mark.parametrize("mode", ["single", "batch"])
    @pytest.mark.parametrize("l_max", [0, 1, 2])
    def test_scf_step_energy_zero_graph_breaks(self, mode, l_max):
        import torch._dynamo as dynamo

        td = torch.device("cuda:0")
        sys = _build_system(mode, l_max, td, seed=0)
        charges, dipoles, quads, _ = split_multipole_moments(sys["mm"])
        source_feats = pack_charges_dipoles(charges, dipoles)
        cache = prepare_multipole_scf_cache(
            sys["cell"],
            sigma=_SIGMA,
            receiver_sigmas=list(_RSIG),
            k_cutoff=_KCUT,
            l_max=l_max,
        )
        pos, bidx = sys["pos"], sys["batch_idx"]

        def step(sf, q):
            return multipole_scf_step_energy(
                cache, pos, sf, batch_idx=bidx, quadrupoles=q
            )

        dynamo.reset()
        explanation = dynamo.explain(step)(source_feats, quads)
        assert explanation.graph_break_count == 0, (
            f"{mode} l={l_max}: expected 0 graph breaks on the SCF-step hot path, "
            f"got {explanation.graph_break_count}; reasons: "
            f"{[r.reason for r in explanation.break_reasons]}"
        )

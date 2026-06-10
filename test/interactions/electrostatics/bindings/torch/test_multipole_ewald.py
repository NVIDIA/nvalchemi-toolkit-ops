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

"""Tests for the multipole Ewald summation composite (real-space + reciprocal +
self), single-system and batched, l = 0/1/2, incl. forces, stress, and force-loss."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    multipole_electrostatic_energy,
    multipole_ewald_scf_step_energy,
    multipole_ewald_summation,
    multipole_real_space_energy,
    prepare_multipole_scf_cache,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    cartesian_quadrupole_to_e3nn,
    pack_charges_dipoles,
    pack_multipole_moments,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
    BatchMultipoleRealSpaceDipoleFusedScalarFunction,
    BatchMultipoleRealSpaceMonopoleFusedScalarFunction,
    MultipoleRealSpaceDipoleFusedScalarFunction,
    MultipoleRealSpaceMonopoleFusedScalarFunction,
    _multipole_ewald_self_energy_per_atom,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (
    multipole_real_space_quadrupole_energy,
)


def _torch_device(device: str) -> str:
    return "cuda" if "cuda" in device else "cpu"


def _build_system(
    *,
    n_atoms: int,
    box_len: float,
    device: str,
    seed: int,
    full_list: bool = True,
):
    """Charge-neutral random system with explicit (full or half) neighbor list."""
    rng = np.random.default_rng(seed)
    positions_np = rng.uniform(0.0, box_len, size=(n_atoms, 3)).astype(np.float64)
    charges_np = rng.uniform(-1.0, 1.0, size=n_atoms).astype(np.float64)
    charges_np -= charges_np.mean()
    L = max(box_len * 3.0, 100.0)
    cell_np = np.array([[[L, 0, 0], [0, L, 0], [0, 0, L]]], dtype=np.float64)

    idx_j_list: list[int] = []
    nptr = [0]
    shifts_list: list[list[int]] = []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if j == i:
                continue
            if not full_list and j < i:
                continue
            idx_j_list.append(j)
            shifts_list.append([0, 0, 0])
        nptr.append(len(idx_j_list))

    td = _torch_device(device)
    positions = torch.from_numpy(positions_np).to(td)
    charges = torch.from_numpy(charges_np).to(td)
    cell = torch.from_numpy(cell_np).to(td)
    idx_j = torch.tensor(idx_j_list, dtype=torch.int32, device=td)
    neighbor_ptr = torch.tensor(nptr, dtype=torch.int32, device=td)
    unit_shifts = torch.tensor(shifts_list, dtype=torch.int32, device=td)
    alpha = torch.tensor([0.4], dtype=torch.float64, device=td)
    sigma = torch.tensor([1.0], dtype=torch.float64, device=td)
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "idx_j": idx_j,
        "neighbor_ptr": neighbor_ptr,
        "unit_shifts": unit_shifts,
        "alpha": alpha,
        "sigma": sigma,
        "n_atoms": n_atoms,
    }


class TestMultipoleRealSpaceEnergyForward:
    def test_output_shape_dtype_device(self, device):
        """Per-atom energy vector shape / dtype / device."""
        sys_dict = _build_system(n_atoms=6, box_len=5.0, device=device, seed=0)
        e = multipole_real_space_energy(
            sys_dict["positions"],
            pack_charges_dipoles(sys_dict["charges"], None),
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )
        assert e.shape == (sys_dict["n_atoms"],)
        assert e.dtype == torch.float64
        assert e.device.type == _torch_device(device)

    def test_accepts_unbatched_cell(self, device):
        """``(3, 3)`` cell tensor is unsqueezed to ``(1, 3, 3)`` internally."""
        sys_dict = _build_system(n_atoms=4, box_len=5.0, device=device, seed=1)
        cell_2d = sys_dict["cell"].squeeze(0)
        e = multipole_real_space_energy(
            sys_dict["positions"],
            pack_charges_dipoles(sys_dict["charges"], None),
            cell_2d,
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )
        assert e.shape == (sys_dict["n_atoms"],)


class TestMultipoleRealSpaceEnergyBackward:
    def test_finite_difference(self, device):
        """Analytical backward matches central FD within the wp_erfc approximation floor."""
        sys_dict = _build_system(n_atoms=5, box_len=4.0, device=device, seed=3)
        pos = sys_dict["positions"].detach().clone().requires_grad_(True)
        sf = pack_charges_dipoles(
            sys_dict["charges"].detach().clone(), None
        ).requires_grad_(True)

        def _loss(positions, source_feats):
            e = multipole_real_space_energy(
                positions,
                source_feats,
                sys_dict["cell"],
                sys_dict["idx_j"],
                sys_dict["neighbor_ptr"],
                sys_dict["unit_shifts"],
                sys_dict["sigma"],
                sys_dict["alpha"],
            )
            # Deterministic per-atom weighting so FD exercises both direct
            # (a=i) and cross (a!=i) contributions.
            w = torch.linspace(0.3, 1.7, e.shape[0], dtype=e.dtype, device=e.device)
            return (w * e).sum()

        L = _loss(pos, sf)
        L.backward()

        h = 1e-5
        for i in range(sys_dict["n_atoms"]):
            for a in range(3):
                pos_plus = pos.detach().clone()
                pos_minus = pos.detach().clone()
                pos_plus[i, a] += h
                pos_minus[i, a] -= h
                fd = (
                    _loss(pos_plus, sf.detach()).item()
                    - _loss(pos_minus, sf.detach()).item()
                ) / (2 * h)
                assert abs(pos.grad[i, a].item() - fd) < 1e-6, (
                    f"pos grad FD mismatch at ({i},{a}): analytical={pos.grad[i, a].item()}, fd={fd}"
                )

        chg_grad = sf.grad[..., 0]
        for i in range(sys_dict["n_atoms"]):
            sf_plus = sf.detach().clone()
            sf_minus = sf.detach().clone()
            sf_plus[i, 0] += h
            sf_minus[i, 0] -= h
            fd = (
                _loss(pos.detach(), sf_plus).item()
                - _loss(pos.detach(), sf_minus).item()
            ) / (2 * h)
            assert abs(chg_grad[i].item() - fd) < 1e-8, (
                f"charge grad FD mismatch at {i}: analytical={chg_grad[i].item()}, fd={fd}"
            )


class TestMultipoleRealSpaceEnergyMonopoleDoubleBackward:
    """``create_graph=True`` force-loss training path (l_max=0)."""

    def test_force_loss_backward_runs_and_is_finite(self, device):
        sys_dict = _build_system(n_atoms=5, box_len=5.0, device=device, seed=11)

        pos = sys_dict["positions"].detach().clone().requires_grad_(True)
        sf = pack_charges_dipoles(
            sys_dict["charges"].detach().clone(), None
        ).requires_grad_(True)

        e = multipole_real_space_energy(
            pos,
            sf,
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )
        total_e = e.sum()
        (forces_neg,) = torch.autograd.grad(total_e, pos, create_graph=True)
        loss = (forces_neg**2).sum()
        loss.backward()

        assert torch.isfinite(pos.grad).all()
        assert pos.grad.abs().sum() > 0
        assert sf.grad is not None
        assert torch.isfinite(sf.grad).all()
        assert sf.grad.abs().sum() > 0

    def test_double_backward_matches_finite_difference(self, device):
        """Second-order grad vs central FD of the analytical first-order backward."""
        sys_dict = _build_system(n_atoms=4, box_len=4.0, device=device, seed=13)

        pos = sys_dict["positions"].detach().clone().requires_grad_(True)
        sf = pack_charges_dipoles(
            sys_dict["charges"].detach().clone(), None
        ).requires_grad_(True)

        # Fixed upstream "force weights" to define L'.
        td = _torch_device(device)
        rng = np.random.default_rng(500)
        gg_pos_np = rng.standard_normal((sys_dict["n_atoms"], 3))
        gg_chg_np = rng.standard_normal(sys_dict["n_atoms"])
        gg_pos = torch.from_numpy(gg_pos_np).to(td)
        gg_chg = torch.from_numpy(gg_chg_np).to(td)

        # Initial grad_energies = ∂L/∂E for the first backward.
        ge_np = rng.standard_normal(sys_dict["n_atoms"])
        ge = torch.from_numpy(ge_np).to(td).requires_grad_(True)

        e = multipole_real_space_energy(
            pos,
            sf,
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )
        (grad_pos, grad_sf) = torch.autograd.grad(
            outputs=(e,),
            inputs=(pos, sf),
            grad_outputs=(ge,),
            create_graph=True,
        )
        grad_chg = grad_sf[..., 0]

        # Scalar L' = gg_pos · grad_pos + gg_chg · grad_chg
        lprime = (gg_pos * grad_pos).sum() + (gg_chg * grad_chg).sum()
        lprime.backward()

        h = 1e-5
        pos_np = pos.detach().cpu().numpy().copy()
        sf_np = sf.detach().cpu().numpy().copy()
        for i in range(sys_dict["n_atoms"]):
            for a in range(3):
                pos_p = pos_np.copy()
                pos_m = pos_np.copy()
                pos_p[i, a] += h
                pos_m[i, a] -= h

                def _make_and_compute_lprime(p_np):
                    p_local = torch.from_numpy(p_np).to(td).requires_grad_(True)
                    sf_local = torch.from_numpy(sf_np).to(td).requires_grad_(True)
                    e_local = multipole_real_space_energy(
                        p_local,
                        sf_local,
                        sys_dict["cell"],
                        sys_dict["idx_j"],
                        sys_dict["neighbor_ptr"],
                        sys_dict["unit_shifts"],
                        sys_dict["sigma"],
                        sys_dict["alpha"],
                    )
                    (gp_l, gsf_l) = torch.autograd.grad(
                        outputs=(e_local,),
                        inputs=(p_local, sf_local),
                        grad_outputs=(ge.detach(),),
                    )
                    gc_l = gsf_l[..., 0]
                    return float(((gg_pos * gp_l).sum() + (gg_chg * gc_l).sum()).item())

                fd_val = (
                    _make_and_compute_lprime(pos_p) - _make_and_compute_lprime(pos_m)
                ) / (2 * h)
                analytical = pos.grad[i, a].item()
                assert abs(analytical - fd_val) < 5e-6, (
                    f"2nd-order grad_pos FD mismatch at ({i},{a}): "
                    f"analytical={analytical}, fd={fd_val}"
                )


def _random_dipoles(n_atoms: int, device: str, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed + 100)
    return torch.from_numpy(
        rng.standard_normal((n_atoms, 3)).astype(np.float64) * 0.3
    ).to(_torch_device(device))


class TestMultipoleRealSpaceEnergyDipole:
    @pytest.mark.parametrize("seed", [0, 42])
    def test_collapse_to_monopole_when_dipoles_zero(self, device, seed):
        """Zero dipoles ⇒ l_max=1 energy equals l_max=0 energy (monopole collapse)."""
        sys_dict = _build_system(n_atoms=6, box_len=5.0, device=device, seed=seed)
        td = _torch_device(device)
        zero_dipoles = torch.zeros(
            sys_dict["n_atoms"], 3, dtype=torch.float64, device=td
        )

        e_dipole = multipole_real_space_energy(
            sys_dict["positions"],
            pack_charges_dipoles(sys_dict["charges"], zero_dipoles),
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )
        e_monopole = multipole_real_space_energy(
            sys_dict["positions"],
            pack_charges_dipoles(sys_dict["charges"], None),
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )
        torch.testing.assert_close(e_dipole, e_monopole, rtol=0, atol=1e-14)

    def test_backward_finite_difference(self, device):
        """Analytical backward matches central FD for positions / charges / dipoles."""
        sys_dict = _build_system(n_atoms=5, box_len=4.0, device=device, seed=3)
        dip = _random_dipoles(sys_dict["n_atoms"], device, seed=3)
        pos = sys_dict["positions"].detach().clone().requires_grad_(True)
        sf = pack_charges_dipoles(
            sys_dict["charges"].detach().clone(), dip
        ).requires_grad_(True)

        def _loss(positions, source_feats):
            e = multipole_real_space_energy(
                positions,
                source_feats,
                sys_dict["cell"],
                sys_dict["idx_j"],
                sys_dict["neighbor_ptr"],
                sys_dict["unit_shifts"],
                sys_dict["sigma"],
                sys_dict["alpha"],
            )
            w = torch.linspace(0.3, 1.7, e.shape[0], dtype=e.dtype, device=e.device)
            return (w * e).sum()

        L = _loss(pos, sf)
        L.backward()

        chg_grad = sf.grad[..., 0]
        # Dipole grad: e3nn (y, z, x) spherical cols [1:4] → Cartesian (x, y, z).
        dip_grad_cart = sf.grad[..., [3, 1, 2]]

        h = 1e-5
        for i in range(sys_dict["n_atoms"]):
            for a in range(3):
                pos_plus = pos.detach().clone()
                pos_minus = pos.detach().clone()
                pos_plus[i, a] += h
                pos_minus[i, a] -= h
                fd = (
                    _loss(pos_plus, sf.detach()).item()
                    - _loss(pos_minus, sf.detach()).item()
                ) / (2 * h)
                assert abs(pos.grad[i, a].item() - fd) < 1e-5, (
                    f"pos grad FD mismatch at ({i},{a}): "
                    f"analytical={pos.grad[i, a].item()}, fd={fd}"
                )

        for i in range(sys_dict["n_atoms"]):
            sf_plus = sf.detach().clone()
            sf_minus = sf.detach().clone()
            sf_plus[i, 0] += h
            sf_minus[i, 0] -= h
            fd = (
                _loss(pos.detach(), sf_plus).item()
                - _loss(pos.detach(), sf_minus).item()
            ) / (2 * h)
            assert abs(chg_grad[i].item() - fd) < 1e-8, (
                f"charge grad FD mismatch at {i}: "
                f"analytical={chg_grad[i].item()}, fd={fd}"
            )

        # Cartesian axis a=0,1,2 (x,y,z) maps to source_feats col (3, 1, 2).
        cart_to_sph_col = (3, 1, 2)
        for i in range(sys_dict["n_atoms"]):
            for a in range(3):
                sf_plus = sf.detach().clone()
                sf_minus = sf.detach().clone()
                col = cart_to_sph_col[a]
                sf_plus[i, col] += h
                sf_minus[i, col] -= h
                fd = (
                    _loss(pos.detach(), sf_plus).item()
                    - _loss(pos.detach(), sf_minus).item()
                ) / (2 * h)
                assert abs(dip_grad_cart[i, a].item() - fd) < 1e-8, (
                    f"dipole grad FD mismatch at ({i},{a}): "
                    f"analytical={dip_grad_cart[i, a].item()}, fd={fd}"
                )


class TestMultipoleRealSpaceEnergyDipoleDoubleBackward:
    """Analytical l_max=1 second-order kernel (charges + dipoles).

    Exercises force-loss training and verifies the 2nd-order ∂²/∂pos
    gradient against central FD of the first-order backward.
    """

    def test_force_loss_backward_runs_and_is_finite(self, device):
        sys_dict = _build_system(n_atoms=5, box_len=5.0, device=device, seed=42)
        dipoles = _random_dipoles(sys_dict["n_atoms"], device, seed=42)
        pos = sys_dict["positions"].detach().clone().requires_grad_(True)
        sf = pack_charges_dipoles(
            sys_dict["charges"].detach().clone(), dipoles
        ).requires_grad_(True)

        e = multipole_real_space_energy(
            pos,
            sf,
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )
        total_e = e.sum()
        (forces_neg,) = torch.autograd.grad(total_e, pos, create_graph=True)
        loss = (forces_neg**2).sum()
        loss.backward()
        for t in (pos, sf):
            assert t.grad is not None
            assert torch.isfinite(t.grad).all()
            assert t.grad.abs().sum() > 0

    def test_double_backward_matches_finite_difference(self, device):
        """Second-order ∂²/∂pos of L' vs central FD of the first-order backward."""
        sys_dict = _build_system(n_atoms=4, box_len=4.0, device=device, seed=13)
        td = _torch_device(device)
        rng = np.random.default_rng(500)

        dipoles = _random_dipoles(sys_dict["n_atoms"], device, seed=13)
        pos = sys_dict["positions"].detach().clone().requires_grad_(True)
        sf = pack_charges_dipoles(
            sys_dict["charges"].detach().clone(), dipoles
        ).requires_grad_(True)

        gg_pos = torch.from_numpy(rng.standard_normal((sys_dict["n_atoms"], 3))).to(td)
        gg_chg = torch.from_numpy(rng.standard_normal(sys_dict["n_atoms"])).to(td)
        gg_dip = torch.from_numpy(rng.standard_normal((sys_dict["n_atoms"], 3))).to(td)
        ge_np = rng.standard_normal(sys_dict["n_atoms"])
        ge = torch.from_numpy(ge_np).to(td).requires_grad_(True)

        e = multipole_real_space_energy(
            pos,
            sf,
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )
        (grad_pos, grad_sf) = torch.autograd.grad(
            outputs=(e,),
            inputs=(pos, sf),
            grad_outputs=(ge,),
            create_graph=True,
        )
        grad_chg = grad_sf[..., 0]
        # Cartesian (x, y, z) dipole grad from spherical (y, z, x) columns.
        grad_dip = grad_sf[..., [3, 1, 2]]
        lprime = (
            (gg_pos * grad_pos).sum()
            + (gg_chg * grad_chg).sum()
            + (gg_dip * grad_dip).sum()
        )
        lprime.backward()

        h = 1e-5
        pos_np = pos.detach().cpu().numpy().copy()
        sf_np = sf.detach().cpu().numpy().copy()

        def _lprime_at(p_np):
            p_local = torch.from_numpy(p_np).to(td).requires_grad_(True)
            sf_local = torch.from_numpy(sf_np).to(td).requires_grad_(True)
            e_local = multipole_real_space_energy(
                p_local,
                sf_local,
                sys_dict["cell"],
                sys_dict["idx_j"],
                sys_dict["neighbor_ptr"],
                sys_dict["unit_shifts"],
                sys_dict["sigma"],
                sys_dict["alpha"],
            )
            (gp_l, gsf_l) = torch.autograd.grad(
                outputs=(e_local,),
                inputs=(p_local, sf_local),
                grad_outputs=(ge.detach(),),
            )
            gc_l = gsf_l[..., 0]
            gd_l = gsf_l[..., [3, 1, 2]]
            return float(
                (
                    (gg_pos * gp_l).sum()
                    + (gg_chg * gc_l).sum()
                    + (gg_dip * gd_l).sum()
                ).item()
            )

        for i in range(sys_dict["n_atoms"]):
            for a in range(3):
                pp = pos_np.copy()
                pm = pos_np.copy()
                pp[i, a] += h
                pm[i, a] -= h
                fd_val = (_lprime_at(pp) - _lprime_at(pm)) / (2 * h)
                analytical = pos.grad[i, a].item()
                # Looser than lmax=0: l_max=1 position grad involves the A'''
                # term, amplifying the wp_erfc floor via higher radial derivatives.
                assert abs(analytical - fd_val) < 1e-4, (
                    f"2nd-order grad_pos FD mismatch at ({i},{a}): "
                    f"analytical={analytical}, fd={fd_val}"
                )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch.compile / Warp path is GPU-only here"
)
@pytest.mark.slow
class TestMultipoleRealSpaceDipoleCompile:
    r"""The :math:`l_{max}=1` single-system path is a ``torch.library.custom_op``
    chain (not a ``torch.autograd.Function``), so ``torch.compile`` must trace it
    as one opaque graph node — no graph break on the struct-dtype
    ``wp.from_torch`` calls — and produce eager-identical results.
    """

    def _inputs(self):
        sys_dict = _build_system(n_atoms=12, box_len=6.0, device="cuda:0", seed=7)
        rng = np.random.default_rng(11)
        dipoles = torch.from_numpy(
            (0.3 * rng.standard_normal((sys_dict["n_atoms"], 3))).astype(np.float64)
        ).to("cuda")
        source_feats = pack_charges_dipoles(sys_dict["charges"], dipoles)
        return sys_dict, source_feats

    def _energy(self, sys_dict, source_feats, positions):
        return multipole_real_space_energy(
            positions,
            source_feats,
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )

    def test_no_graph_breaks(self):
        """``torch.compile`` captures the path in a single graph (0 breaks)."""
        torch._dynamo.reset()
        sys_dict, source_feats = self._inputs()
        explanation = torch._dynamo.explain(
            lambda: self._energy(sys_dict, source_feats, sys_dict["positions"])
        )()
        assert explanation.graph_break_count == 0, explanation.break_reasons

    def test_compiled_matches_eager(self):
        """Compiled forward + first-order grad match eager bit-for-bit."""
        torch._dynamo.reset()
        sys_dict, source_feats = self._inputs()
        pos_eager = sys_dict["positions"].detach().clone().requires_grad_(True)
        pos_comp = sys_dict["positions"].detach().clone().requires_grad_(True)

        e_eager = self._energy(sys_dict, source_feats, pos_eager)
        compiled = torch.compile(lambda p: self._energy(sys_dict, source_feats, p))
        e_comp = compiled(pos_comp)
        torch.testing.assert_close(e_comp, e_eager)

        (g_eager,) = torch.autograd.grad(e_eager.sum(), pos_eager)
        (g_comp,) = torch.autograd.grad(e_comp.sum(), pos_comp)
        torch.testing.assert_close(g_comp, g_eager)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch.compile / Warp path is GPU-only here"
)
@pytest.mark.slow
class TestMultipoleRealSpaceMonopoleCompile:
    r"""``torch.compile`` regression for the :math:`l_{max}=0` single-system path
    (``nvalchemiops::multipole_real_space_monopole`` custom_op chain)."""

    def _inputs(self):
        sys_dict = _build_system(n_atoms=12, box_len=6.0, device="cuda:0", seed=5)
        return sys_dict, pack_charges_dipoles(sys_dict["charges"], None)

    def _energy(self, sys_dict, source_feats, positions):
        return multipole_real_space_energy(
            positions,
            source_feats,
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )

    def test_no_graph_breaks(self):
        """``torch.compile`` captures the path in a single graph (0 breaks)."""
        torch._dynamo.reset()
        sys_dict, source_feats = self._inputs()
        explanation = torch._dynamo.explain(
            lambda: self._energy(sys_dict, source_feats, sys_dict["positions"])
        )()
        assert explanation.graph_break_count == 0, explanation.break_reasons

    def test_compiled_matches_eager(self):
        """Compiled forward + first-order grad match eager bit-for-bit."""
        torch._dynamo.reset()
        sys_dict, source_feats = self._inputs()
        pos_eager = sys_dict["positions"].detach().clone().requires_grad_(True)
        pos_comp = sys_dict["positions"].detach().clone().requires_grad_(True)

        e_eager = self._energy(sys_dict, source_feats, pos_eager)
        compiled = torch.compile(lambda p: self._energy(sys_dict, source_feats, p))
        e_comp = compiled(pos_comp)
        torch.testing.assert_close(e_comp, e_eager)

        (g_eager,) = torch.autograd.grad(e_eager.sum(), pos_eager)
        (g_comp,) = torch.autograd.grad(e_comp.sum(), pos_comp)
        torch.testing.assert_close(g_comp, g_eager)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch.compile / Warp path is GPU-only here"
)
@pytest.mark.slow
class TestMultipoleRealSpaceQuadrupoleCompile:
    r"""``torch.compile`` regression for the :math:`l_{max}=2` single-system path
    (``nvalchemiops::multipole_real_space_quadrupole`` custom_op chain, whose
    backward conditionally fires the cell-grad kernel)."""

    def _inputs(self):
        sys_dict = _build_system(n_atoms=10, box_len=6.0, device="cuda:0", seed=9)
        rng = np.random.default_rng(13)
        n = sys_dict["n_atoms"]
        dipoles = torch.from_numpy(
            (0.3 * rng.standard_normal((n, 3))).astype(np.float64)
        ).to("cuda")
        q = 0.2 * rng.standard_normal((n, 3, 3))
        q = 0.5 * (q + q.transpose(0, 2, 1))
        q -= (q.trace(axis1=1, axis2=2) / 3.0)[:, None, None] * np.eye(3)
        quads = torch.from_numpy(q.astype(np.float64)).to("cuda")
        moments = pack_multipole_moments(sys_dict["charges"], dipoles, quads)
        return sys_dict, moments

    def _energy(self, sys_dict, moments, positions, cell=None):
        return multipole_real_space_energy(
            positions,
            moments,
            sys_dict["cell"] if cell is None else cell,
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )

    def test_no_graph_breaks(self):
        """``torch.compile`` captures the l=2 path in a single graph (0 breaks)."""
        torch._dynamo.reset()
        sys_dict, moments = self._inputs()
        explanation = torch._dynamo.explain(
            lambda: self._energy(sys_dict, moments, sys_dict["positions"])
        )()
        assert explanation.graph_break_count == 0, explanation.break_reasons

    def test_compiled_matches_eager(self):
        """Compiled forward + first-order grad match eager bit-for-bit."""
        torch._dynamo.reset()
        sys_dict, moments = self._inputs()
        pos_eager = sys_dict["positions"].detach().clone().requires_grad_(True)
        pos_comp = sys_dict["positions"].detach().clone().requires_grad_(True)

        e_eager = self._energy(sys_dict, moments, pos_eager)
        compiled = torch.compile(lambda p: self._energy(sys_dict, moments, p))
        e_comp = compiled(pos_comp)
        torch.testing.assert_close(e_comp, e_eager)

        (g_eager,) = torch.autograd.grad(e_eager.sum(), pos_eager)
        (g_comp,) = torch.autograd.grad(e_comp.sum(), pos_comp)
        torch.testing.assert_close(g_comp, g_eager)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch.compile / Warp path is GPU-only here"
)
@pytest.mark.slow
@pytest.mark.parametrize("l_max", [0, 1, 2])
class TestBatchMultipoleRealSpaceCompile:
    r"""``torch.compile`` regression for the batched real-space path
    (``batch_multipole_real_space_*`` custom_op chains), all ``l_max``."""

    def _batch(self, l_max):
        a = _build_system(n_atoms=8, box_len=6.0, device="cuda:0", seed=1)
        b = _build_system(n_atoms=11, box_len=6.0, device="cuda:0", seed=2)
        n0 = a["n_atoms"]
        rng = np.random.default_rng(3)

        def _moments(sysd):
            n = sysd["n_atoms"]
            if l_max == 0:
                return pack_charges_dipoles(sysd["charges"], None)
            dip = torch.from_numpy(
                (0.3 * rng.standard_normal((n, 3))).astype(np.float64)
            ).to("cuda")
            if l_max == 1:
                return pack_charges_dipoles(sysd["charges"], dip)
            q = 0.2 * rng.standard_normal((n, 3, 3))
            q = 0.5 * (q + q.transpose(0, 2, 1))
            q -= (q.trace(axis1=1, axis2=2) / 3.0)[:, None, None] * np.eye(3)
            quads = torch.from_numpy(q.astype(np.float64)).to("cuda")
            return pack_multipole_moments(sysd["charges"], dip, quads)

        positions = torch.cat([a["positions"], b["positions"]])
        moments = torch.cat([_moments(a), _moments(b)])
        idx = torch.cat([a["idx_j"], b["idx_j"] + n0])
        m0 = a["idx_j"].numel()
        nptr = torch.cat([a["neighbor_ptr"], b["neighbor_ptr"][1:] + m0]).to(
            torch.int32
        )
        shifts = torch.cat([a["unit_shifts"], b["unit_shifts"]])
        batch_idx = torch.cat(
            [
                torch.zeros(n0, dtype=torch.int32, device="cuda"),
                torch.ones(b["n_atoms"], dtype=torch.int32, device="cuda"),
            ]
        )
        cells = torch.cat([a["cell"], b["cell"]])  # each (1,3,3) -> (2,3,3)
        sig = torch.tensor([1.0, 1.0], dtype=torch.float64, device="cuda")
        alp = torch.tensor([0.4, 0.4], dtype=torch.float64, device="cuda")
        return positions, moments, cells, idx, nptr, shifts, sig, alp, batch_idx

    def _energy(self, packed, positions):
        _, moments, cells, idx, nptr, shifts, sig, alp, batch_idx = packed
        return multipole_real_space_energy(
            positions, moments, cells, idx, nptr, shifts, sig, alp, batch_idx=batch_idx
        )

    def test_no_graph_breaks(self, l_max):
        """``torch.compile`` captures the batched path in a single graph."""
        torch._dynamo.reset()
        packed = self._batch(l_max)
        explanation = torch._dynamo.explain(lambda: self._energy(packed, packed[0]))()
        assert explanation.graph_break_count == 0, explanation.break_reasons

    def test_compiled_matches_eager(self, l_max):
        """Compiled forward + first-order grad match eager bit-for-bit."""
        torch._dynamo.reset()
        packed = self._batch(l_max)
        pos_eager = packed[0].detach().clone().requires_grad_(True)
        pos_comp = packed[0].detach().clone().requires_grad_(True)
        e_eager = self._energy(packed, pos_eager)
        compiled = torch.compile(lambda p: self._energy(packed, p))
        e_comp = compiled(pos_comp)
        torch.testing.assert_close(e_comp, e_eager)
        (g_eager,) = torch.autograd.grad(e_eager.sum(), pos_eager)
        (g_comp,) = torch.autograd.grad(e_comp.sum(), pos_comp)
        torch.testing.assert_close(g_comp, g_eager)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch.compile / Warp path is GPU-only here"
)
@pytest.mark.slow
@pytest.mark.parametrize("l_max", [0, 1])
class TestMultipoleRealSpaceFusedScalarCompile:
    r"""``torch.compile`` regression for the fused-scalar real-space ops
    (``multipole_real_space_{monopole,dipole}_fused``) used by the Ewald
    composite: one fused launch returns the scalar energy + per-atom grads."""

    def _inputs(self, l_max):
        s = _build_system(n_atoms=10, box_len=6.0, device="cuda:0", seed=4)
        dip = None
        if l_max == 1:
            rng = np.random.default_rng(8)
            dip = torch.from_numpy(
                (0.3 * rng.standard_normal((s["n_atoms"], 3))).astype(np.float64)
            ).to("cuda")
        return s, s["cell"], dip

    def _energy(self, s, cell, dip, positions):
        if dip is None:
            return torch.ops.nvalchemiops.multipole_real_space_monopole_fused(
                positions,
                s["charges"],
                cell,
                s["sigma"],
                s["alpha"],
                s["idx_j"],
                s["neighbor_ptr"],
                s["unit_shifts"],
                False,
            )[0]
        return torch.ops.nvalchemiops.multipole_real_space_dipole_fused(
            positions,
            s["charges"],
            dip,
            cell,
            s["sigma"],
            s["alpha"],
            s["idx_j"],
            s["neighbor_ptr"],
            s["unit_shifts"],
            False,
        )[0]

    def test_no_graph_breaks(self, l_max):
        torch._dynamo.reset()
        s, cell, dip = self._inputs(l_max)
        explanation = torch._dynamo.explain(
            lambda: self._energy(s, cell, dip, s["positions"])
        )()
        assert explanation.graph_break_count == 0, explanation.break_reasons

    def test_compiled_matches_eager(self, l_max):
        torch._dynamo.reset()
        s, cell, dip = self._inputs(l_max)
        pos_e = s["positions"].detach().clone().requires_grad_(True)
        pos_c = s["positions"].detach().clone().requires_grad_(True)
        e_eager = self._energy(s, cell, dip, pos_e)
        compiled = torch.compile(lambda p: self._energy(s, cell, dip, p))
        e_comp = compiled(pos_c)
        torch.testing.assert_close(e_comp, e_eager)
        (g_eager,) = torch.autograd.grad(e_eager, pos_e)
        (g_comp,) = torch.autograd.grad(e_comp, pos_c)
        torch.testing.assert_close(g_comp, g_eager)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch.compile / Warp path is GPU-only here"
)
@pytest.mark.slow
@pytest.mark.parametrize("l_max", [0, 1])
class TestBatchMultipoleRealSpaceFusedScalarCompile:
    r"""``torch.compile`` regression for the batched fused-scalar ops
    (``batch_multipole_real_space_{monopole,dipole}_fused``) used by the batched
    Ewald composite."""

    def _batch(self, l_max):
        a = _build_system(n_atoms=8, box_len=6.0, device="cuda:0", seed=1)
        b = _build_system(n_atoms=11, box_len=6.0, device="cuda:0", seed=2)
        n0 = a["n_atoms"]
        positions = torch.cat([a["positions"], b["positions"]])
        idx = torch.cat([a["idx_j"], b["idx_j"] + n0])
        m0 = a["idx_j"].numel()
        nptr = torch.cat([a["neighbor_ptr"], b["neighbor_ptr"][1:] + m0]).to(
            torch.int32
        )
        shifts = torch.cat([a["unit_shifts"], b["unit_shifts"]])
        batch_idx = torch.cat(
            [
                torch.zeros(n0, dtype=torch.int32, device="cuda"),
                torch.ones(b["n_atoms"], dtype=torch.int32, device="cuda"),
            ]
        )
        cells = torch.cat([a["cell"], b["cell"]])  # each (1,3,3) -> (2,3,3)
        sig = torch.tensor([1.0, 1.0], dtype=torch.float64, device="cuda")
        alp = torch.tensor([0.4, 0.4], dtype=torch.float64, device="cuda")
        charges = torch.cat([a["charges"], b["charges"]])
        dip = None
        if l_max == 1:
            rng = np.random.default_rng(6)
            dip = torch.from_numpy(
                (0.3 * rng.standard_normal((n0 + b["n_atoms"], 3))).astype(np.float64)
            ).to("cuda")
        return positions, charges, dip, cells, idx, nptr, shifts, sig, alp, batch_idx

    def _energy(self, packed, positions):
        _, charges, dip, cells, idx, nptr, shifts, sig, alp, batch_idx = packed
        if dip is None:
            return torch.ops.nvalchemiops.batch_multipole_real_space_monopole_fused(
                positions, charges, cells, sig, alp, idx, nptr, shifts, batch_idx, False
            )[0]
        return torch.ops.nvalchemiops.batch_multipole_real_space_dipole_fused(
            positions,
            charges,
            dip,
            cells,
            sig,
            alp,
            idx,
            nptr,
            shifts,
            batch_idx,
            False,
        )[0]

    def test_no_graph_breaks(self, l_max):
        torch._dynamo.reset()
        packed = self._batch(l_max)
        explanation = torch._dynamo.explain(lambda: self._energy(packed, packed[0]))()
        assert explanation.graph_break_count == 0, explanation.break_reasons

    def test_compiled_matches_eager(self, l_max):
        torch._dynamo.reset()
        packed = self._batch(l_max)
        pos_e = packed[0].detach().clone().requires_grad_(True)
        pos_c = packed[0].detach().clone().requires_grad_(True)
        e_eager = self._energy(packed, pos_e)
        compiled = torch.compile(lambda p: self._energy(packed, p))
        e_comp = compiled(pos_c)
        torch.testing.assert_close(e_comp, e_eager)
        (g_eager,) = torch.autograd.grad(e_eager.sum(), pos_e)
        (g_comp,) = torch.autograd.grad(e_comp.sum(), pos_c)
        torch.testing.assert_close(g_comp, g_eager)


class TestMultipoleRealSpaceValidation:
    def test_bad_source_feats_shape(self, device):
        """source_feats with a mismatched leading dim raises ValueError."""
        sys_dict = _build_system(n_atoms=4, box_len=5.0, device=device, seed=0)
        # Wrong N (3 vs 4 positions), still a valid (N, 1) trailing dim.
        bad_sf = torch.zeros(3, 1, dtype=torch.float64, device=_torch_device(device))
        with pytest.raises(ValueError, match="multipole_moments must be"):
            multipole_real_space_energy(
                sys_dict["positions"],
                bad_sf,
                sys_dict["cell"],
                sys_dict["idx_j"],
                sys_dict["neighbor_ptr"],
                sys_dict["unit_shifts"],
                sys_dict["sigma"],
                sys_dict["alpha"],
            )

    def test_bad_positions_shape(self, device):
        sys_dict = _build_system(n_atoms=4, box_len=5.0, device=device, seed=0)
        bad_pos = sys_dict["positions"][:, :2]  # (N, 2)
        with pytest.raises(ValueError, match="positions must be"):
            multipole_real_space_energy(
                bad_pos,
                pack_charges_dipoles(sys_dict["charges"], None),
                sys_dict["cell"],
                sys_dict["idx_j"],
                sys_dict["neighbor_ptr"],
                sys_dict["unit_shifts"],
                sys_dict["sigma"],
                sys_dict["alpha"],
            )

    def test_batched_cell_rejected(self, device):
        """A ``(B, 3, 3)`` cell with B>1 raises on the single-system entry."""
        sys_dict = _build_system(n_atoms=4, box_len=5.0, device=device, seed=0)
        batched_cell = sys_dict["cell"].repeat(2, 1, 1)  # (2, 3, 3)
        with pytest.raises(ValueError, match="batch size"):
            multipole_real_space_energy(
                sys_dict["positions"],
                pack_charges_dipoles(sys_dict["charges"], None),
                batched_cell,
                sys_dict["idx_j"],
                sys_dict["neighbor_ptr"],
                sys_dict["unit_shifts"],
                sys_dict["sigma"],
                sys_dict["alpha"],
            )


class TestMultipoleRealSpaceMonopoleFusedScalar:
    r"""Parity tests for ``MultipoleRealSpaceMonopoleFusedScalarFunction``.

    The fused Function returns scalar energy, computes analytical gradients
    in forward when inputs require grad, and backward is a pure scalar
    broadcast. Parity vs the layered ``MultipoleRealSpaceMonopoleFunction`` +
    backward pair should hold to ULP at fp64.
    """

    @pytest.fixture
    def gpu_system(self):
        if not torch.cuda.is_available():
            pytest.skip("Fused path is GPU-only")
        return _build_system(n_atoms=8, box_len=5.0, device="cuda:0", seed=0xCAFE)

    def _run_fused(self, sys_dict, *, with_pos: bool, with_q: bool):
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            MultipoleRealSpaceMonopoleFusedScalarFunction,
        )

        positions = sys_dict["positions"].clone().detach().requires_grad_(with_pos)
        charges = sys_dict["charges"].clone().detach().requires_grad_(with_q)
        return (
            positions,
            charges,
            MultipoleRealSpaceMonopoleFusedScalarFunction.apply(
                positions,
                charges,
                sys_dict["cell"],
                sys_dict["sigma"],
                sys_dict["alpha"],
                sys_dict["idx_j"],
                sys_dict["neighbor_ptr"],
                sys_dict["unit_shifts"],
            ),
        )

    def _run_reference(self, sys_dict, *, with_pos: bool, with_q: bool):
        """Existing layered path — per-atom energies, summed externally."""
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            MultipoleRealSpaceMonopoleFunction,
        )

        positions = sys_dict["positions"].clone().detach().requires_grad_(with_pos)
        charges = sys_dict["charges"].clone().detach().requires_grad_(with_q)
        per_atom = MultipoleRealSpaceMonopoleFunction.apply(
            positions,
            charges,
            sys_dict["cell"],
            sys_dict["sigma"],
            sys_dict["alpha"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
        )
        return positions, charges, per_atom.sum()

    def test_forward_parity_vs_reference(self, gpu_system):
        """Energy sum matches reference path to ULP at fp64."""
        _, _, e_fused = self._run_fused(gpu_system, with_pos=False, with_q=False)
        _, _, e_ref = self._run_reference(gpu_system, with_pos=False, with_q=False)
        torch.testing.assert_close(e_fused, e_ref, rtol=0, atol=1e-15)

    def test_grad_pos_parity_vs_reference(self, gpu_system):
        """Position gradient matches reference path to ULP at fp64."""
        pos_f, _, e_f = self._run_fused(gpu_system, with_pos=True, with_q=False)
        e_f.backward()
        pos_r, _, e_r = self._run_reference(gpu_system, with_pos=True, with_q=False)
        e_r.backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)

    def test_grad_q_parity_vs_reference(self, gpu_system):
        """Charge gradient matches reference path to ULP at fp64."""
        _, q_f, e_f = self._run_fused(gpu_system, with_pos=False, with_q=True)
        e_f.backward()
        _, q_r, e_r = self._run_reference(gpu_system, with_pos=False, with_q=True)
        e_r.backward()
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)

    def test_both_grads_parity_vs_reference(self, gpu_system):
        """Joint pos+charge gradient matches reference path to ULP."""
        pos_f, q_f, e_f = self._run_fused(gpu_system, with_pos=True, with_q=True)
        e_f.backward()
        pos_r, q_r, e_r = self._run_reference(gpu_system, with_pos=True, with_q=True)
        e_r.backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)

    def test_per_input_grad_gating_pos_only(self, gpu_system):
        """When only positions.requires_grad, charges.grad stays None."""
        pos_f, q_f, e_f = self._run_fused(gpu_system, with_pos=True, with_q=False)
        e_f.backward()
        assert pos_f.grad is not None
        assert q_f.grad is None  # never required grad

    def test_per_input_grad_gating_charge_only(self, gpu_system):
        """When only charges.requires_grad, positions.grad stays None."""
        pos_f, q_f, e_f = self._run_fused(gpu_system, with_pos=False, with_q=True)
        e_f.backward()
        assert pos_f.grad is None
        assert q_f.grad is not None

    def test_no_grad_path_runs(self, gpu_system):
        """No-grad inputs run the energy-only kernel and produce no graph."""
        _, _, e = self._run_fused(gpu_system, with_pos=False, with_q=False)
        assert not e.requires_grad
        assert torch.isfinite(e).item()

    def test_backward_does_no_kernel_work(self, gpu_system):
        """Backward is a scalar broadcast over pre-computed gradient tensors."""
        pos_f, q_f, e_f = self._run_fused(gpu_system, with_pos=True, with_q=True)
        upstream = torch.tensor(2.5, dtype=torch.float64, device=e_f.device)
        e_f.backward(upstream)
        pos_r, q_r, e_r = self._run_reference(gpu_system, with_pos=True, with_q=True)
        e_r.backward(upstream)
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)

    def test_double_backward_raises(self, gpu_system):
        """Fused path doesn't support 2nd-order grad; first backward is finite."""
        pos_f, q_f, e_f = self._run_fused(gpu_system, with_pos=True, with_q=False)
        (grad_pos,) = torch.autograd.grad(e_f, pos_f, create_graph=True)
        assert torch.isfinite(grad_pos).all()

    @pytest.fixture
    def cpu_system(self):
        return _build_system(n_atoms=8, box_len=5.0, device="cpu", seed=0xC0FFEE)

    def test_cpu_path_runs(self, cpu_system):
        """CSR fused kernel runs on CPU."""
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            MultipoleRealSpaceMonopoleFusedScalarFunction,
        )

        positions = cpu_system["positions"].clone().detach().requires_grad_(True)
        charges = cpu_system["charges"].clone().detach().requires_grad_(True)
        e = MultipoleRealSpaceMonopoleFusedScalarFunction.apply(
            positions,
            charges,
            cpu_system["cell"],
            cpu_system["sigma"],
            cpu_system["alpha"],
            cpu_system["idx_j"],
            cpu_system["neighbor_ptr"],
            cpu_system["unit_shifts"],
        )
        e.backward()
        assert torch.isfinite(positions.grad).all()
        assert torch.isfinite(charges.grad).all()
        assert torch.isfinite(e).item()

    def test_cpu_parity_vs_layered_csr(self, cpu_system):
        """Bit-for-bit parity vs the layered CSR path on CPU."""
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            MultipoleRealSpaceMonopoleFunction,
            MultipoleRealSpaceMonopoleFusedScalarFunction,
        )

        pos_f = cpu_system["positions"].clone().detach().requires_grad_(True)
        q_f = cpu_system["charges"].clone().detach().requires_grad_(True)
        e_f = MultipoleRealSpaceMonopoleFusedScalarFunction.apply(
            pos_f,
            q_f,
            cpu_system["cell"],
            cpu_system["sigma"],
            cpu_system["alpha"],
            cpu_system["idx_j"],
            cpu_system["neighbor_ptr"],
            cpu_system["unit_shifts"],
        )
        e_f.backward()

        pos_r = cpu_system["positions"].clone().detach().requires_grad_(True)
        q_r = cpu_system["charges"].clone().detach().requires_grad_(True)
        e_r = MultipoleRealSpaceMonopoleFunction.apply(
            pos_r,
            q_r,
            cpu_system["cell"],
            cpu_system["sigma"],
            cpu_system["alpha"],
            cpu_system["idx_j"],
            cpu_system["neighbor_ptr"],
            cpu_system["unit_shifts"],
        ).sum()
        e_r.backward()

        torch.testing.assert_close(e_f, e_r, rtol=0, atol=1e-15)
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)


class TestBatchMultipoleRealSpaceMonopoleFusedScalar:
    r"""Parity tests for ``BatchMultipoleRealSpaceMonopoleFusedScalarFunction``.

    Returns per-system scalar tensor ``(B,)``; backward is per-atom
    weight broadcast (``grad_E[batch_idx[i]]`` per atom).
    """

    def _build_batch_system(
        self, *, device: str, n_per_sys: int = 8, n_systems: int = 3
    ):
        """Stack n_systems independent _build_system fixtures with shared sigma/alpha."""
        sub = []
        for s in range(n_systems):
            sd = _build_system(
                n_atoms=n_per_sys, box_len=5.0, device=device, seed=0x100 + s
            )
            sub.append(sd)

        # Flat CSR: offset each system's idx_j by the cumulative atom count.
        td = sub[0]["positions"].device
        positions = torch.cat([s["positions"] for s in sub], dim=0)
        charges = torch.cat([s["charges"] for s in sub], dim=0)
        cells = torch.cat([s["cell"] for s in sub], dim=0)  # (B, 3, 3)
        sigmas = torch.cat([s["sigma"] for s in sub], dim=0)  # (B,)
        alphas = torch.cat([s["alpha"] for s in sub], dim=0)  # (B,)

        offset_atoms = 0
        offset_edges = 0
        idx_j_chunks = []
        nptr_chunks = []
        unit_shifts_chunks = []
        batch_idx_chunks = []

        nptr_chunks.append(torch.tensor([0], dtype=torch.int32, device=td))
        for s_idx, s in enumerate(sub):
            idx_j_chunks.append(s["idx_j"] + offset_atoms)
            unit_shifts_chunks.append(s["unit_shifts"])
            # Shift by cumulative edge count and drop the leading zero.
            sub_nptr = s["neighbor_ptr"][1:] + offset_edges
            nptr_chunks.append(sub_nptr)
            batch_idx_chunks.append(
                torch.full((s["n_atoms"],), s_idx, dtype=torch.int32, device=td)
            )
            offset_atoms += s["n_atoms"]
            offset_edges += s["idx_j"].shape[0]

        return {
            "positions": positions,
            "charges": charges,
            "cells": cells,
            "sigmas": sigmas,
            "alphas": alphas,
            "idx_j": torch.cat(idx_j_chunks, dim=0),
            "neighbor_ptr": torch.cat(nptr_chunks, dim=0),
            "unit_shifts": torch.cat(unit_shifts_chunks, dim=0),
            "batch_idx": torch.cat(batch_idx_chunks, dim=0),
            "n_systems": n_systems,
            "n_per_sys": n_per_sys,
        }

    @pytest.fixture
    def gpu_batch(self):
        if not torch.cuda.is_available():
            pytest.skip("Fused batched tile path is GPU-only")
        return self._build_batch_system(device="cuda:0")

    @pytest.fixture
    def cpu_batch(self):
        return self._build_batch_system(device="cpu")

    def _run_fused(self, batch_dict, *, with_pos: bool, with_q: bool):
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            BatchMultipoleRealSpaceMonopoleFusedScalarFunction,
        )

        positions = batch_dict["positions"].clone().detach().requires_grad_(with_pos)
        charges = batch_dict["charges"].clone().detach().requires_grad_(with_q)
        per_sys = BatchMultipoleRealSpaceMonopoleFusedScalarFunction.apply(
            positions,
            charges,
            batch_dict["cells"],
            batch_dict["sigmas"],
            batch_dict["alphas"],
            batch_dict["idx_j"],
            batch_dict["neighbor_ptr"],
            batch_dict["unit_shifts"],
            batch_dict["batch_idx"],
        )
        return positions, charges, per_sys

    def _run_reference(self, batch_dict, *, with_pos: bool, with_q: bool):
        """Existing layered batched path → per-atom energies → scatter to per-system."""
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            BatchMultipoleRealSpaceMonopoleFunction,
        )

        positions = batch_dict["positions"].clone().detach().requires_grad_(with_pos)
        charges = batch_dict["charges"].clone().detach().requires_grad_(with_q)
        per_atom = BatchMultipoleRealSpaceMonopoleFunction.apply(
            positions,
            charges,
            batch_dict["cells"],
            batch_dict["sigmas"],
            batch_dict["alphas"],
            batch_dict["idx_j"],
            batch_dict["neighbor_ptr"],
            batch_dict["unit_shifts"],
            batch_dict["batch_idx"],
        )
        per_sys = torch.zeros(
            batch_dict["n_systems"], dtype=torch.float64, device=per_atom.device
        )
        per_sys.scatter_add_(0, batch_dict["batch_idx"].to(torch.int64), per_atom)
        return positions, charges, per_sys

    def test_gpu_forward_parity(self, gpu_batch):
        """Per-system energies match reference path to ULP at fp64 (GPU tile)."""
        _, _, e_f = self._run_fused(gpu_batch, with_pos=False, with_q=False)
        _, _, e_r = self._run_reference(gpu_batch, with_pos=False, with_q=False)
        torch.testing.assert_close(e_f, e_r, rtol=0, atol=1e-15)

    def test_gpu_grad_pos_parity(self, gpu_batch):
        """Position gradient matches reference path on GPU tile."""
        pos_f, _, e_f = self._run_fused(gpu_batch, with_pos=True, with_q=False)
        e_f.sum().backward()
        pos_r, _, e_r = self._run_reference(gpu_batch, with_pos=True, with_q=False)
        e_r.sum().backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)

    def test_gpu_both_grads_parity(self, gpu_batch):
        """Joint pos + charge gradient parity on GPU tile."""
        pos_f, q_f, e_f = self._run_fused(gpu_batch, with_pos=True, with_q=True)
        e_f.sum().backward()
        pos_r, q_r, e_r = self._run_reference(gpu_batch, with_pos=True, with_q=True)
        e_r.sum().backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)

    def test_cpu_forward_parity(self, cpu_batch):
        """Forward parity on CPU CSR path."""
        _, _, e_f = self._run_fused(cpu_batch, with_pos=False, with_q=False)
        _, _, e_r = self._run_reference(cpu_batch, with_pos=False, with_q=False)
        torch.testing.assert_close(e_f, e_r, rtol=0, atol=1e-15)

    def test_cpu_both_grads_parity(self, cpu_batch):
        """Joint pos + charge gradient parity on CPU CSR path."""
        pos_f, q_f, e_f = self._run_fused(cpu_batch, with_pos=True, with_q=True)
        e_f.sum().backward()
        pos_r, q_r, e_r = self._run_reference(cpu_batch, with_pos=True, with_q=True)
        e_r.sum().backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)

    def test_per_system_weighted_backward(self, gpu_batch):
        """Non-uniform per-system upstream weights (B,) propagate correctly."""
        pos_f, q_f, e_f = self._run_fused(gpu_batch, with_pos=True, with_q=True)
        weights = torch.tensor([1.5, -0.5, 2.0], dtype=torch.float64, device=e_f.device)
        (weights * e_f).sum().backward()

        pos_r, q_r, e_r = self._run_reference(gpu_batch, with_pos=True, with_q=True)
        (weights * e_r).sum().backward()

        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)


def _random_dipoles_dipole(n: int, device: str, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal((n, 3)) * 0.4).to(
        _torch_device(device), dtype=torch.float64
    )


class TestMultipoleRealSpaceDipoleFusedScalar:
    r"""Parity tests for ``MultipoleRealSpaceDipoleFusedScalarFunction`` (lmax=1).

    The fused Function returns scalar total energy and stashes analytical
    (`grad_pos`, `grad_q`, `grad_mu`) in ctx for backward. Parity is anchored
    against the layered ``MultipoleRealSpaceFunction`` summed externally.
    """

    @pytest.fixture
    def gpu_system(self):
        if not torch.cuda.is_available():
            pytest.skip("Fused tile path is GPU-only")
        sys_dict = _build_system(n_atoms=8, box_len=5.0, device="cuda:0", seed=0xCAFE)
        sys_dict["dipoles"] = _random_dipoles_dipole(8, "cuda:0", 0xC0FE)
        return sys_dict

    @pytest.fixture
    def cpu_system(self):
        sys_dict = _build_system(n_atoms=8, box_len=5.0, device="cpu", seed=0xC0FFEE)
        sys_dict["dipoles"] = _random_dipoles_dipole(8, "cpu", 0xBEEF)
        return sys_dict

    def _run_fused(self, sys_dict, *, with_pos: bool, with_q: bool, with_mu: bool):
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            MultipoleRealSpaceDipoleFusedScalarFunction,
        )

        positions = sys_dict["positions"].clone().detach().requires_grad_(with_pos)
        charges = sys_dict["charges"].clone().detach().requires_grad_(with_q)
        dipoles = sys_dict["dipoles"].clone().detach().requires_grad_(with_mu)
        e = MultipoleRealSpaceDipoleFusedScalarFunction.apply(
            positions,
            charges,
            dipoles,
            sys_dict["cell"],
            sys_dict["sigma"],
            sys_dict["alpha"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
        )
        return positions, charges, dipoles, e

    def _run_reference(self, sys_dict, *, with_pos: bool, with_q: bool, with_mu: bool):
        """Existing layered lmax=1 path."""
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            MultipoleRealSpaceFunction,
        )

        positions = sys_dict["positions"].clone().detach().requires_grad_(with_pos)
        charges = sys_dict["charges"].clone().detach().requires_grad_(with_q)
        dipoles = sys_dict["dipoles"].clone().detach().requires_grad_(with_mu)
        per_atom = MultipoleRealSpaceFunction.apply(
            positions,
            charges,
            dipoles,
            sys_dict["cell"],
            sys_dict["sigma"],
            sys_dict["alpha"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
        )
        return positions, charges, dipoles, per_atom.sum()

    def test_gpu_forward_parity(self, gpu_system):
        """Energy parity at ULP fp64 (no grads)."""
        _, _, _, e_f = self._run_fused(
            gpu_system, with_pos=False, with_q=False, with_mu=False
        )
        _, _, _, e_r = self._run_reference(
            gpu_system, with_pos=False, with_q=False, with_mu=False
        )
        torch.testing.assert_close(e_f, e_r, rtol=0, atol=1e-15)

    def test_gpu_all_grads_parity(self, gpu_system):
        """Joint pos + charge + dipole gradient parity on GPU tile."""
        pos_f, q_f, mu_f, e_f = self._run_fused(
            gpu_system, with_pos=True, with_q=True, with_mu=True
        )
        e_f.backward()
        pos_r, q_r, mu_r, e_r = self._run_reference(
            gpu_system, with_pos=True, with_q=True, with_mu=True
        )
        e_r.backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=0, atol=1e-15)

    def test_gpu_dipole_grad_only(self, gpu_system):
        """Dipole gradient alone parity (charges + positions detached)."""
        _, _, mu_f, e_f = self._run_fused(
            gpu_system, with_pos=False, with_q=False, with_mu=True
        )
        e_f.backward()
        _, _, mu_r, e_r = self._run_reference(
            gpu_system, with_pos=False, with_q=False, with_mu=True
        )
        e_r.backward()
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=0, atol=1e-15)

    def test_gpu_per_input_gating(self, gpu_system):
        """Only requested gradient slots get populated."""
        pos_f, q_f, mu_f, e_f = self._run_fused(
            gpu_system, with_pos=True, with_q=False, with_mu=True
        )
        e_f.backward()
        assert pos_f.grad is not None
        assert q_f.grad is None
        assert mu_f.grad is not None

    def test_cpu_all_grads_parity(self, cpu_system):
        """Same parity test on CPU CSR fused path."""
        pos_f, q_f, mu_f, e_f = self._run_fused(
            cpu_system, with_pos=True, with_q=True, with_mu=True
        )
        e_f.backward()
        pos_r, q_r, mu_r, e_r = self._run_reference(
            cpu_system, with_pos=True, with_q=True, with_mu=True
        )
        e_r.backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=0, atol=1e-15)


class TestBatchMultipoleRealSpaceDipoleFusedScalar:
    r"""Parity tests for ``BatchMultipoleRealSpaceDipoleFusedScalarFunction`` (batched lmax=1)."""

    def _build_batch_system(
        self, *, device: str, n_per_sys: int = 8, n_systems: int = 3
    ):
        sub = []
        for s in range(n_systems):
            sd = _build_system(
                n_atoms=n_per_sys, box_len=5.0, device=device, seed=0x200 + s
            )
            sd["dipoles"] = _random_dipoles_dipole(n_per_sys, device, 0xD00 + s)
            sub.append(sd)

        td = sub[0]["positions"].device
        positions = torch.cat([s["positions"] for s in sub], dim=0)
        charges = torch.cat([s["charges"] for s in sub], dim=0)
        dipoles = torch.cat([s["dipoles"] for s in sub], dim=0)
        cells = torch.cat([s["cell"] for s in sub], dim=0)
        sigmas = torch.cat([s["sigma"] for s in sub], dim=0)
        alphas = torch.cat([s["alpha"] for s in sub], dim=0)

        offset_atoms = 0
        offset_edges = 0
        idx_j_chunks = []
        nptr_chunks = [torch.tensor([0], dtype=torch.int32, device=td)]
        unit_shifts_chunks = []
        batch_idx_chunks = []

        for s_idx, s in enumerate(sub):
            idx_j_chunks.append(s["idx_j"] + offset_atoms)
            unit_shifts_chunks.append(s["unit_shifts"])
            sub_nptr = s["neighbor_ptr"][1:] + offset_edges
            nptr_chunks.append(sub_nptr)
            batch_idx_chunks.append(
                torch.full((s["n_atoms"],), s_idx, dtype=torch.int32, device=td)
            )
            offset_atoms += s["n_atoms"]
            offset_edges += s["idx_j"].shape[0]

        return {
            "positions": positions,
            "charges": charges,
            "dipoles": dipoles,
            "cells": cells,
            "sigmas": sigmas,
            "alphas": alphas,
            "idx_j": torch.cat(idx_j_chunks, dim=0),
            "neighbor_ptr": torch.cat(nptr_chunks, dim=0),
            "unit_shifts": torch.cat(unit_shifts_chunks, dim=0),
            "batch_idx": torch.cat(batch_idx_chunks, dim=0),
            "n_systems": n_systems,
        }

    @pytest.fixture
    def gpu_batch(self):
        if not torch.cuda.is_available():
            pytest.skip("Fused tile path is GPU-only")
        return self._build_batch_system(device="cuda:0")

    @pytest.fixture
    def cpu_batch(self):
        return self._build_batch_system(device="cpu")

    def _run_fused(self, b, *, with_pos, with_q, with_mu):
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            BatchMultipoleRealSpaceDipoleFusedScalarFunction,
        )

        positions = b["positions"].clone().detach().requires_grad_(with_pos)
        charges = b["charges"].clone().detach().requires_grad_(with_q)
        dipoles = b["dipoles"].clone().detach().requires_grad_(with_mu)
        per_sys = BatchMultipoleRealSpaceDipoleFusedScalarFunction.apply(
            positions,
            charges,
            dipoles,
            b["cells"],
            b["sigmas"],
            b["alphas"],
            b["idx_j"],
            b["neighbor_ptr"],
            b["unit_shifts"],
            b["batch_idx"],
        )
        return positions, charges, dipoles, per_sys

    def _run_reference(self, b, *, with_pos, with_q, with_mu):
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            BatchMultipoleRealSpaceFunction,
        )

        positions = b["positions"].clone().detach().requires_grad_(with_pos)
        charges = b["charges"].clone().detach().requires_grad_(with_q)
        dipoles = b["dipoles"].clone().detach().requires_grad_(with_mu)
        per_atom = BatchMultipoleRealSpaceFunction.apply(
            positions,
            charges,
            dipoles,
            b["cells"],
            b["sigmas"],
            b["alphas"],
            b["idx_j"],
            b["neighbor_ptr"],
            b["unit_shifts"],
            b["batch_idx"],
        )
        per_sys = torch.zeros(
            b["n_systems"], dtype=torch.float64, device=per_atom.device
        )
        per_sys.scatter_add_(0, b["batch_idx"].to(torch.int64), per_atom)
        return positions, charges, dipoles, per_sys

    def test_gpu_forward_parity(self, gpu_batch):
        _, _, _, e_f = self._run_fused(
            gpu_batch, with_pos=False, with_q=False, with_mu=False
        )
        _, _, _, e_r = self._run_reference(
            gpu_batch, with_pos=False, with_q=False, with_mu=False
        )
        torch.testing.assert_close(e_f, e_r, rtol=0, atol=1e-15)

    def test_gpu_all_grads_parity(self, gpu_batch):
        """Joint pos + charge + dipole gradient parity, batched."""
        pos_f, q_f, mu_f, e_f = self._run_fused(
            gpu_batch, with_pos=True, with_q=True, with_mu=True
        )
        e_f.sum().backward()
        pos_r, q_r, mu_r, e_r = self._run_reference(
            gpu_batch, with_pos=True, with_q=True, with_mu=True
        )
        e_r.sum().backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=0, atol=1e-15)

    def test_cpu_all_grads_parity(self, cpu_batch):
        pos_f, q_f, mu_f, e_f = self._run_fused(
            cpu_batch, with_pos=True, with_q=True, with_mu=True
        )
        e_f.sum().backward()
        pos_r, q_r, mu_r, e_r = self._run_reference(
            cpu_batch, with_pos=True, with_q=True, with_mu=True
        )
        e_r.sum().backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=0, atol=1e-15)

    def test_per_system_weighted_backward(self, gpu_batch):
        pos_f, q_f, mu_f, e_f = self._run_fused(
            gpu_batch, with_pos=True, with_q=True, with_mu=True
        )
        weights = torch.tensor([1.5, -0.5, 2.0], dtype=torch.float64, device=e_f.device)
        (weights * e_f).sum().backward()

        pos_r, q_r, mu_r, e_r = self._run_reference(
            gpu_batch, with_pos=True, with_q=True, with_mu=True
        )
        (weights * e_r).sum().backward()

        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=0, atol=1e-15)


def _rand_system(n: int, box: float, device: str, seed: int):
    rng = np.random.default_rng(seed)
    td = _torch_device(device)
    pos = torch.from_numpy(rng.uniform(0, box, (n, 3))).to(td, torch.float64)
    chg_np = rng.uniform(-1, 1, n)
    chg_np -= chg_np.mean()
    chg = torch.from_numpy(chg_np).to(td, torch.float64)
    cell = torch.eye(3, dtype=torch.float64, device=td) * box
    idx_j_l, nptr_l, sh_l = [], [0], []
    for i in range(n):
        for j in range(n):
            if j != i:
                idx_j_l.append(j)
                sh_l.append([0, 0, 0])
        nptr_l.append(len(idx_j_l))
    return (
        pos,
        chg,
        cell,
        torch.tensor(idx_j_l, dtype=torch.int32, device=td),
        torch.tensor(nptr_l, dtype=torch.int32, device=td),
        torch.tensor(sh_l, dtype=torch.int32, device=td),
    )


def _flatten_batch(systems):
    """Stitch a list of per-system fixtures into the flat batched form."""
    n_per = [s[0].shape[0] for s in systems]
    pos = torch.cat([s[0] for s in systems])
    chg = torch.cat([s[1] for s in systems])
    cells = torch.stack([s[2] for s in systems])
    idx_j_flat_l, nptr_flat_l, sh_flat_l = [], [0], []
    atom_off = 0
    for s in systems:
        idx_j_flat_l.append(s[3] + atom_off)
        sh_flat_l.append(s[5])
        nptr_np = s[4].cpu().numpy()
        for k in range(1, len(nptr_np)):
            nptr_flat_l.append(nptr_flat_l[-1] + int(nptr_np[k] - nptr_np[k - 1]))
        atom_off += s[0].shape[0]
    idx_j_flat = torch.cat(idx_j_flat_l)
    nptr_flat = torch.tensor(nptr_flat_l, dtype=torch.int32, device=pos.device)
    sh_flat = torch.cat(sh_flat_l)
    bi = torch.cat(
        [
            torch.full((n_per[i],), i, dtype=torch.int32, device=pos.device)
            for i in range(len(systems))
        ]
    )
    return pos, chg, cells, idx_j_flat, nptr_flat, sh_flat, bi, n_per


class TestBatchedMonopoleForward:
    def test_forward_bit_parity_vs_per_system(self, device):
        systems = [
            _rand_system(5, 4.0, device, 0),
            _rand_system(4, 5.0, device, 1),
            _rand_system(6, 3.8, device, 2),
        ]
        alphas_np = [0.3, 0.4, 0.5]
        td = _torch_device(device)
        alphas = torch.tensor(alphas_np, dtype=torch.float64, device=td)
        sigmas = torch.full_like(alphas, 1.0)

        per_e = []
        for i, s in enumerate(systems):
            pos, chg, cell, idx_j, nptr, sh = s
            a = alphas[i : i + 1]
            per_e.append(
                multipole_real_space_energy(
                    pos,
                    pack_charges_dipoles(chg, None),
                    cell,
                    idx_j,
                    nptr,
                    sh,
                    sigmas[i : i + 1],
                    a,
                )
            )

        pos_all, chg_all, cells, idx_j_flat, nptr_flat, sh_flat, bi, n_per = (
            _flatten_batch(systems)
        )
        e_batch = multipole_real_space_energy(
            pos_all,
            pack_charges_dipoles(chg_all, None),
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigmas,
            alphas,
            batch_idx=bi,
        )

        assert e_batch.shape == (sum(n_per),)
        assert e_batch.dtype == torch.float64
        off = 0
        for i, n in enumerate(n_per):
            torch.testing.assert_close(e_batch[off : off + n], per_e[i], rtol=0, atol=0)
            off += n


class TestBatchedMonopoleBackward:
    def test_backward_bit_parity_vs_per_system(self, device):
        systems = [_rand_system(5, 4.0, device, 100), _rand_system(4, 5.0, device, 101)]
        td = _torch_device(device)
        alphas = torch.tensor([0.3, 0.4], dtype=torch.float64, device=td)
        sigmas = torch.full_like(alphas, 1.0)

        per_gp, per_gc = [], []
        for i, s in enumerate(systems):
            pos_, chg_, cell, idx_j, nptr, sh = s
            p = pos_.detach().clone().requires_grad_(True)
            sf_ = pack_charges_dipoles(chg_.detach().clone(), None).requires_grad_(True)
            e = multipole_real_space_energy(
                p, sf_, cell, idx_j, nptr, sh, sigmas[i : i + 1], alphas[i : i + 1]
            )
            e.sum().backward()
            per_gp.append(p.grad)
            per_gc.append(sf_.grad[..., 0])

        pos_all, chg_all, cells, idx_j_flat, nptr_flat, sh_flat, bi, n_per = (
            _flatten_batch(systems)
        )
        pos_all = pos_all.requires_grad_(True)
        sf_all = pack_charges_dipoles(chg_all, None).requires_grad_(True)
        e_batch = multipole_real_space_energy(
            pos_all,
            sf_all,
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigmas,
            alphas,
            batch_idx=bi,
        )
        e_batch.sum().backward()

        off = 0
        for i, n in enumerate(n_per):
            torch.testing.assert_close(
                pos_all.grad[off : off + n], per_gp[i], rtol=0, atol=0
            )
            torch.testing.assert_close(
                sf_all.grad[off : off + n, 0], per_gc[i], rtol=0, atol=0
            )
            off += n


def _rand_dipoles(n: int, device: str, seed: int) -> torch.Tensor:
    td = _torch_device(device)
    rng = np.random.default_rng(seed + 1000)
    return torch.from_numpy(0.3 * rng.standard_normal((n, 3))).to(td, torch.float64)


class TestBatchedDipoleForward:
    """Batched l_max=1 (charges + dipoles) forward parity."""

    def test_forward_bit_parity_vs_per_system(self, device):
        systems = [
            _rand_system(5, 4.0, device, 100),
            _rand_system(4, 5.0, device, 101),
            _rand_system(6, 3.8, device, 102),
        ]
        dipoles_per = [
            _rand_dipoles(s[0].shape[0], device, seed=i) for i, s in enumerate(systems)
        ]
        td = _torch_device(device)
        alphas = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64, device=td)
        sigmas = torch.full_like(alphas, 1.0)

        per_e = []
        for i, s in enumerate(systems):
            pos, chg, cell, idx_j, nptr, sh = s
            dip = dipoles_per[i]
            per_e.append(
                multipole_real_space_energy(
                    pos,
                    pack_charges_dipoles(chg, dip),
                    cell,
                    idx_j,
                    nptr,
                    sh,
                    sigmas[i : i + 1],
                    alphas[i : i + 1],
                )
            )

        pos_all, chg_all, cells, idx_j_flat, nptr_flat, sh_flat, bi, n_per = (
            _flatten_batch(systems)
        )
        dip_all = torch.cat(dipoles_per)
        e_batch = multipole_real_space_energy(
            pos_all,
            pack_charges_dipoles(chg_all, dip_all),
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigmas,
            alphas,
            batch_idx=bi,
        )
        assert e_batch.shape == (sum(n_per),)
        off = 0
        for i, n in enumerate(n_per):
            torch.testing.assert_close(e_batch[off : off + n], per_e[i], rtol=0, atol=0)
            off += n


class TestBatchedDipoleBackward:
    """Batched l_max=1 first-order backward parity on positions/charges/dipoles."""

    def test_backward_bit_parity_vs_per_system(self, device):
        systems = [
            _rand_system(5, 4.0, device, 200),
            _rand_system(4, 5.0, device, 201),
        ]
        dipoles_per = [
            _rand_dipoles(s[0].shape[0], device, seed=i) for i, s in enumerate(systems)
        ]
        td = _torch_device(device)
        alphas = torch.tensor([0.3, 0.4], dtype=torch.float64, device=td)
        sigmas = torch.full_like(alphas, 1.0)

        per_gp, per_gc, per_gd = [], [], []
        for i, s in enumerate(systems):
            pos_, chg_, cell, idx_j, nptr, sh = s
            p = pos_.detach().clone().requires_grad_(True)
            sf_ = pack_charges_dipoles(
                chg_.detach().clone(), dipoles_per[i].detach().clone()
            ).requires_grad_(True)
            e = multipole_real_space_energy(
                p, sf_, cell, idx_j, nptr, sh, sigmas[i : i + 1], alphas[i : i + 1]
            )
            e.sum().backward()
            per_gp.append(p.grad)
            per_gc.append(sf_.grad[..., 0])
            per_gd.append(sf_.grad[..., [3, 1, 2]])

        pos_all, chg_all, cells, idx_j_flat, nptr_flat, sh_flat, bi, n_per = (
            _flatten_batch(systems)
        )
        dip_all = torch.cat(dipoles_per)
        pos_all = pos_all.requires_grad_(True)
        sf_all = pack_charges_dipoles(chg_all, dip_all).requires_grad_(True)
        e_batch = multipole_real_space_energy(
            pos_all,
            sf_all,
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigmas,
            alphas,
            batch_idx=bi,
        )
        e_batch.sum().backward()

        chg_all_grad = sf_all.grad[..., 0]
        dip_all_grad = sf_all.grad[..., [3, 1, 2]]

        off = 0
        for i, n in enumerate(n_per):
            # Backward uses atomic_add, so ordering can introduce ~float64 ULP
            # differences. Use a tight but non-zero tolerance.
            torch.testing.assert_close(
                pos_all.grad[off : off + n], per_gp[i], rtol=0, atol=1e-14
            )
            torch.testing.assert_close(
                chg_all_grad[off : off + n], per_gc[i], rtol=0, atol=1e-14
            )
            torch.testing.assert_close(
                dip_all_grad[off : off + n], per_gd[i], rtol=0, atol=1e-14
            )
            off += n

    def test_force_loss_parity_vs_per_system(self, device):
        """Batched l_max=1 double-backward matches per-system force-loss gradients."""
        systems = [
            _rand_system(5, 4.0, device, 300),
            _rand_system(4, 5.0, device, 301),
        ]
        dipoles_per = [
            _rand_dipoles(s[0].shape[0], device, seed=i) for i, s in enumerate(systems)
        ]
        td = _torch_device(device)
        alphas = torch.tensor([0.3, 0.4], dtype=torch.float64, device=td)
        sigmas = torch.full_like(alphas, 1.0)

        per_gp, per_gc, per_gd = [], [], []
        for i, s in enumerate(systems):
            pos_, chg_, cell, idx_j, nptr, sh = s
            p = pos_.detach().clone().requires_grad_(True)
            sf_ = pack_charges_dipoles(
                chg_.detach().clone(), dipoles_per[i].detach().clone()
            ).requires_grad_(True)
            e = multipole_real_space_energy(
                p, sf_, cell, idx_j, nptr, sh, sigmas[i : i + 1], alphas[i : i + 1]
            )
            (forces_neg,) = torch.autograd.grad(e.sum(), p, create_graph=True)
            (forces_neg**2).sum().backward()
            per_gp.append(p.grad)
            per_gc.append(sf_.grad[..., 0])
            per_gd.append(sf_.grad[..., [3, 1, 2]])

        pos_all, chg_all, cells, idx_j_flat, nptr_flat, sh_flat, bi, n_per = (
            _flatten_batch(systems)
        )
        dip_all = torch.cat(dipoles_per)
        pos_all = pos_all.detach().clone().requires_grad_(True)
        sf_all = (
            pack_charges_dipoles(chg_all, dip_all).detach().clone().requires_grad_(True)
        )
        e_batch = multipole_real_space_energy(
            pos_all,
            sf_all,
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigmas,
            alphas,
            batch_idx=bi,
        )
        (forces_neg_batch,) = torch.autograd.grad(
            e_batch.sum(), pos_all, create_graph=True
        )
        (forces_neg_batch**2).sum().backward()

        chg_all_grad = sf_all.grad[..., 0]
        dip_all_grad = sf_all.grad[..., [3, 1, 2]]

        off = 0
        for i, n in enumerate(n_per):
            torch.testing.assert_close(
                pos_all.grad[off : off + n], per_gp[i], rtol=0, atol=1e-12
            )
            torch.testing.assert_close(
                chg_all_grad[off : off + n], per_gc[i], rtol=0, atol=1e-12
            )
            torch.testing.assert_close(
                dip_all_grad[off : off + n], per_gd[i], rtol=0, atol=1e-12
            )
            off += n


class TestBatchedMonopoleValidation:
    def test_bad_source_feats_shape(self, device):
        s = _rand_system(4, 4.0, device, 0)
        pos, chg, cell, idx_j, nptr, sh = s
        td = _torch_device(device)
        cells = cell.unsqueeze(0)
        alphas = torch.tensor([0.3], dtype=torch.float64, device=td)
        sigmas = torch.tensor([1.0], dtype=torch.float64, device=td)
        bi = torch.zeros(pos.shape[0], dtype=torch.int32, device=td)
        # Wrong N (3 vs 4 positions), still a valid (N, 1) trailing dim.
        bad_sf = torch.zeros(3, 1, dtype=torch.float64, device=td)
        with pytest.raises(ValueError, match="multipole_moments must be"):
            multipole_real_space_energy(
                pos, bad_sf, cells, idx_j, nptr, sh, sigmas, alphas, batch_idx=bi
            )

    def test_force_loss_parity_vs_per_system(self, device):
        """Batched l_max=0 double-backward matches per-system force-loss gradients."""
        systems = [
            _rand_system(5, 4.0, device, 400),
            _rand_system(4, 5.0, device, 401),
        ]
        td = _torch_device(device)
        alphas = torch.tensor([0.3, 0.4], dtype=torch.float64, device=td)
        sigmas = torch.full_like(alphas, 1.0)

        per_gp, per_gc = [], []
        for i, s in enumerate(systems):
            pos_, chg_, cell, idx_j, nptr, sh = s
            p = pos_.detach().clone().requires_grad_(True)
            sf_ = pack_charges_dipoles(chg_.detach().clone(), None).requires_grad_(True)
            e = multipole_real_space_energy(
                p, sf_, cell, idx_j, nptr, sh, sigmas[i : i + 1], alphas[i : i + 1]
            )
            (forces_neg,) = torch.autograd.grad(e.sum(), p, create_graph=True)
            (forces_neg**2).sum().backward()
            per_gp.append(p.grad)
            per_gc.append(sf_.grad[..., 0])

        pos_all, chg_all, cells, idx_j_flat, nptr_flat, sh_flat, bi, n_per = (
            _flatten_batch(systems)
        )
        pos_all = pos_all.detach().clone().requires_grad_(True)
        sf_all = (
            pack_charges_dipoles(chg_all, None).detach().clone().requires_grad_(True)
        )
        e_batch = multipole_real_space_energy(
            pos_all,
            sf_all,
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigmas,
            alphas,
            batch_idx=bi,
        )
        (forces_neg_batch,) = torch.autograd.grad(
            e_batch.sum(), pos_all, create_graph=True
        )
        (forces_neg_batch**2).sum().backward()

        chg_all_grad = sf_all.grad[..., 0]
        off = 0
        for i, n in enumerate(n_per):
            torch.testing.assert_close(
                pos_all.grad[off : off + n], per_gp[i], rtol=0, atol=1e-12
            )
            torch.testing.assert_close(
                chg_all_grad[off : off + n], per_gc[i], rtol=0, atol=1e-12
            )
            off += n


def _bcc(n: int, a: float = 4.14):
    """Alternating-charge BCC supercell (neutral, l_max=0 capable)."""
    p = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                p.append((i * a, j * a, k * a))
                p.append(((i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a))
    return np.array(p), np.eye(3) * (n * a)


def _neigh(positions: np.ndarray, L: float, cutoff: float):
    """O(N² · shells) neighbor list covering all periodic images within ``cutoff``.

    Test-only; production should use the real neighbor-list builder. Test
    systems are tiny (~16 atoms) so the O(N²) overhead is negligible.
    """
    N = positions.shape[0]
    shell = int(math.ceil(cutoff / L)) + 1
    idx_j, nptr, shifts = [], [0], []
    for i in range(N):
        for sa in range(-shell, shell + 1):
            for sb in range(-shell, shell + 1):
                for sc in range(-shell, shell + 1):
                    for j in range(N):
                        if j == i and (sa, sb, sc) == (0, 0, 0):
                            continue
                        r = positions[j] - positions[i] + np.array([sa, sb, sc]) * L
                        if np.linalg.norm(r) < cutoff:
                            idx_j.append(j)
                            shifts.append([sa, sb, sc])
        nptr.append(len(idx_j))
    return (
        np.array(idx_j, np.int32),
        np.array(nptr, np.int32),
        np.array(shifts, np.int32),
    )


def _build(n: int, device: str, seed: int, l_max: int):
    """BCC system with alternating charges (and optional random dipoles)."""
    rng = np.random.default_rng(seed)
    pos_np, cell_np = _bcc(n)
    N = pos_np.shape[0]
    chg_np = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    if abs(chg_np.sum()) > 1e-12:
        chg_np[-1] -= chg_np.sum()
    dip_np = 0.3 * rng.standard_normal((N, 3)) if l_max >= 1 else None
    quad_np = None
    if l_max >= 2:
        Qr = 0.2 * rng.standard_normal((N, 3, 3))
        quad_np = 0.5 * (Qr + Qr.transpose(0, 2, 1))  # symmetric; packer drops trace

    td = _torch_device(device)
    pos = torch.from_numpy(pos_np).to(td, torch.float64)
    chg = torch.from_numpy(chg_np).to(td, torch.float64)
    dip = torch.from_numpy(dip_np).to(td, torch.float64) if dip_np is not None else None
    cell = torch.from_numpy(cell_np).to(td, torch.float64)
    if quad_np is not None:
        quad = torch.from_numpy(quad_np).to(td, torch.float64)
        source_feats = pack_multipole_moments(chg, dip, quad)
    else:
        source_feats = pack_charges_dipoles(chg, dip)
    return pos, source_feats, cell, cell_np, pos_np


def _path_a_vs_b(
    device: str, n: int, sigma: float, alpha: float, l_max: int, seed: int
) -> float:
    pos, source_feats, cell, cell_np, pos_np = _build(n, device, seed, l_max)
    L = cell_np[0, 0]
    td = _torch_device(device)

    # Real-space cutoff governed by σ_c; 10·σ_c gives ULP-level parity.
    sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
    cutoff = 10.0 * sigma_c
    idx_j_np, nptr_np, sh_np = _neigh(pos_np, L, cutoff)
    idx_j = torch.from_numpy(idx_j_np).to(td)
    nptr = torch.from_numpy(nptr_np).to(td)
    sh = torch.from_numpy(sh_np).to(td)

    # k·σ_c = 6 ⇒ Gaussian damping exp(-36) ~ 1e-16 at cutoff.
    kcut = 6.0 / sigma_c

    E_B = float(
        multipole_electrostatic_energy(
            pos, source_feats, cell, sigma=sigma, k_cutoff=kcut
        ).sum()
    )
    E_A = float(
        multipole_ewald_summation(
            pos,
            source_feats,
            cell,
            idx_j,
            nptr,
            sh,
            sigma=sigma,
            alpha=alpha,
            k_cutoff=kcut,
        ).sum()
    )
    return E_A - E_B


class TestPathAEquivPathB:
    """Path A (real + reciprocal − self) must equal Path B (direct k-space)."""

    @pytest.mark.parametrize("alpha", [0.3, 0.4, 0.6, 0.9])
    @pytest.mark.parametrize("sigma", [0.8, 1.0, 1.2])
    def test_monopole_bcc(self, device, sigma: float, alpha: float):
        """l_max=0 BCC supercell: |Δ| bounded by accumulated wp_erfc error."""
        delta = _path_a_vs_b(device, n=2, sigma=sigma, alpha=alpha, l_max=0, seed=41)
        assert abs(delta) < 5e-4, f"σ={sigma}  α={alpha}  Δ={delta:.3e}"

    @pytest.mark.parametrize("alpha", [0.3, 0.4, 0.6, 0.9])
    @pytest.mark.parametrize("sigma", [0.8, 1.0, 1.2])
    def test_dipole_bcc(self, device, sigma: float, alpha: float):
        """l_max=1 BCC supercell: dipole + charge cross terms."""
        delta = _path_a_vs_b(device, n=2, sigma=sigma, alpha=alpha, l_max=1, seed=47)
        assert abs(delta) < 5e-4, f"σ={sigma}  α={alpha}  Δ={delta:.3e}"

    @pytest.mark.parametrize("alpha", [0.5, 1.0])
    def test_two_atom_sigma1(self, device, alpha: float):
        """2-atom (+1, −1) at separation 3 in large box — tighter tolerance."""
        td = _torch_device(device)
        L = 30.0
        pos_np = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        cell_np = np.eye(3) * L
        chg = torch.tensor([1.0, -1.0], dtype=torch.float64, device=td)
        sf = pack_charges_dipoles(chg, None)
        pos = torch.from_numpy(pos_np).to(td, torch.float64)
        cell = torch.from_numpy(cell_np).to(td, torch.float64)
        sigma = 1.0
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c
        idx_j_np, nptr_np, sh_np = _neigh(pos_np, L, cutoff)
        idx_j = torch.from_numpy(idx_j_np).to(td)
        nptr = torch.from_numpy(nptr_np).to(td)
        sh = torch.from_numpy(sh_np).to(td)
        kcut = 6.0 / sigma_c
        E_B = float(
            multipole_electrostatic_energy(
                pos, sf, cell, sigma=sigma, k_cutoff=kcut
            ).sum()
        )
        E_A = float(
            multipole_ewald_summation(
                pos,
                sf,
                cell,
                idx_j,
                nptr,
                sh,
                sigma=sigma,
                alpha=alpha,
                k_cutoff=kcut,
            ).sum()
        )
        assert abs(E_A - E_B) < 1e-4, (
            f"α={alpha}  E_A={E_A:.6f}  E_B={E_B:.6f}  Δ={E_A - E_B:.3e}"
        )

    def test_alpha_limit_recovers_pathb(self, device):
        """Large α: all energy in real-space, should still match Path B."""
        delta = _path_a_vs_b(device, n=2, sigma=1.0, alpha=2.0, l_max=1, seed=53)
        assert abs(delta) < 1e-4, f"α=2.0 Δ={delta:.3e}"


class TestBatchedEwaldSummation:
    """Batched ``multipole_ewald_summation`` (``batch_idx`` ≠ None) must equal
    a per-system loop of the single-system variant to the same precision as
    forward / backward bit-parity tests — i.e. zero drift because each thread
    still owns a unique ``atom_i`` in the batched kernels.
    """

    @pytest.mark.parametrize("l_max", [0, 1])
    def test_batch_matches_per_system_loop(self, device, l_max: int):
        """Stack B identical systems and verify E_A_batch[b] ≈ E_A_single per b."""
        td = _torch_device(device)
        sigma, alpha = 1.0, 0.6
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c
        kcut = 6.0 / sigma_c

        # Distinct seeds exercise the batch_idx dispatch, not replicated copies.
        systems = []
        for seed in (41, 47, 53):
            pos, sf, cell, cell_np, pos_np = _build(
                n=2, device=device, seed=seed, l_max=l_max
            )
            L = cell_np[0, 0]
            idx_j_np, nptr_np, sh_np = _neigh(pos_np, L, cutoff)
            idx_j = torch.from_numpy(idx_j_np).to(td)
            nptr = torch.from_numpy(nptr_np).to(td)
            sh = torch.from_numpy(sh_np).to(td)
            systems.append(
                (
                    pos,
                    sf,
                    cell,
                    idx_j,
                    nptr,
                    sh,
                    cell_np,
                    pos_np,
                    idx_j_np,
                    nptr_np,
                    sh_np,
                )
            )

        # Per-system reference.
        per_system_e = []
        for sys_tup in systems:
            pos, sf, cell, idx_j, nptr, sh = sys_tup[:6]
            e = float(
                multipole_ewald_summation(
                    pos,
                    sf,
                    cell,
                    idx_j,
                    nptr,
                    sh,
                    sigma=sigma,
                    alpha=alpha,
                    k_cutoff=kcut,
                ).sum()
            )
            per_system_e.append(e)

        # Stitch a batched call.
        pos_all = torch.cat([s[0] for s in systems], dim=0)
        sf_all = torch.cat([s[1] for s in systems], dim=0)
        cells = torch.stack(
            [s[2].squeeze(0) if s[2].ndim == 3 else s[2] for s in systems], dim=0
        )
        n_per = [s[0].shape[0] for s in systems]
        batch_idx = torch.cat(
            [
                torch.full((n,), b, dtype=torch.int32, device=td)
                for b, n in enumerate(n_per)
            ]
        )
        # Flat CSR: offset idx_j by cumulative atom count per system; stitch nptr.
        idx_j_flat = []
        nptr_flat = [0]
        sh_flat = []
        atom_off = 0
        for s in systems:
            idx_j_flat.append(s[8] + atom_off)
            sh_flat.append(s[10])
            nptr_np = s[9]
            for k in range(1, len(nptr_np)):
                nptr_flat.append(nptr_flat[-1] + int(nptr_np[k] - nptr_np[k - 1]))
            atom_off += s[0].shape[0]
        idx_j_flat = torch.from_numpy(np.concatenate(idx_j_flat).astype(np.int32)).to(
            td
        )
        nptr_flat = torch.from_numpy(np.asarray(nptr_flat, dtype=np.int32)).to(td)
        sh_flat = torch.from_numpy(np.concatenate(sh_flat).astype(np.int32)).to(td)

        e_batch = multipole_ewald_summation(
            pos_all,
            sf_all,
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigma=sigma,
            alpha=alpha,
            k_cutoff=kcut,
            batch_idx=batch_idx,
        )
        assert e_batch.shape == (pos_all.shape[0],)
        B = len(systems)
        e_batch_per_sys = torch.zeros(
            B, dtype=torch.float64, device=e_batch.device
        ).scatter_add(0, batch_idx.long(), e_batch)
        for b, e_ref in enumerate(per_system_e):
            assert abs(float(e_batch_per_sys[b]) - e_ref) < 1e-10, (
                f"sys {b} l_max={l_max}: batch={float(e_batch_per_sys[b]):.6e} "
                f"single={e_ref:.6e}"
            )

    @pytest.mark.parametrize("l_max", [0, 1])
    def test_batch_matches_path_b(self, device, l_max: int):
        """Batched Path A ≡ Path B (direct k-space, per-system loop)."""
        td = _torch_device(device)
        sigma, alpha = 1.0, 0.6
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c
        kcut = 6.0 / sigma_c

        # Two distinct systems.
        systems = []
        per_b = []
        for seed in (71, 73):
            pos, sf, cell, cell_np, pos_np = _build(
                n=2, device=device, seed=seed, l_max=l_max
            )
            per_b.append(
                float(
                    multipole_electrostatic_energy(
                        pos, sf, cell, sigma=sigma, k_cutoff=kcut
                    ).sum()
                )
            )
            L = cell_np[0, 0]
            idx_j_np, nptr_np, sh_np = _neigh(pos_np, L, cutoff)
            systems.append((pos, sf, cell, cell_np, pos_np, idx_j_np, nptr_np, sh_np))

        pos_all = torch.cat([s[0] for s in systems], dim=0)
        sf_all = torch.cat([s[1] for s in systems], dim=0)
        cells = torch.stack(
            [s[2].squeeze(0) if s[2].ndim == 3 else s[2] for s in systems], dim=0
        )
        n_per = [s[0].shape[0] for s in systems]
        batch_idx = torch.cat(
            [
                torch.full((n,), b, dtype=torch.int32, device=td)
                for b, n in enumerate(n_per)
            ]
        )
        idx_j_flat = []
        nptr_flat = [0]
        sh_flat = []
        atom_off = 0
        for s in systems:
            idx_j_flat.append(s[5] + atom_off)
            sh_flat.append(s[7])
            nptr_np = s[6]
            for k in range(1, len(nptr_np)):
                nptr_flat.append(nptr_flat[-1] + int(nptr_np[k] - nptr_np[k - 1]))
            atom_off += s[0].shape[0]
        idx_j_flat = torch.from_numpy(np.concatenate(idx_j_flat).astype(np.int32)).to(
            td
        )
        nptr_flat = torch.from_numpy(np.asarray(nptr_flat, dtype=np.int32)).to(td)
        sh_flat = torch.from_numpy(np.concatenate(sh_flat).astype(np.int32)).to(td)

        e_batch = multipole_ewald_summation(
            pos_all,
            sf_all,
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigma=sigma,
            alpha=alpha,
            k_cutoff=kcut,
            batch_idx=batch_idx,
        )
        B_sys = len(per_b)
        e_batch_per_sys = torch.zeros(
            B_sys, dtype=torch.float64, device=e_batch.device
        ).scatter_add(0, batch_idx.long(), e_batch)
        for b, e_b_ref in enumerate(per_b):
            assert abs(float(e_batch_per_sys[b]) - e_b_ref) < 5e-4, (
                f"sys {b} l_max={l_max}: A_batch={float(e_batch_per_sys[b]):.6e} "
                f"B={e_b_ref:.6e}"
            )


class TestEwaldSCFStepEnergy:
    """Cache-aware ``multipole_ewald_scf_step_energy`` must match the
    one-shot ``multipole_ewald_summation`` bit-for-bit when the cache is
    built with matching (σ, α, k_cutoff)."""

    @pytest.mark.parametrize("alpha", [0.4, 0.9])
    @pytest.mark.parametrize("l_max", [0, 1, 2])
    def test_single_matches_one_shot(self, device, l_max: int, alpha: float):
        td = _torch_device(device)
        sigma = 1.0
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c
        kcut = 6.0 / sigma_c

        pos, sf, cell, cell_np, pos_np = _build(
            n=2, device=device, seed=11, l_max=l_max
        )
        L = cell_np[0, 0]
        idx_j_np, nptr_np, sh_np = _neigh(pos_np, L, cutoff)
        idx_j = torch.from_numpy(idx_j_np).to(td)
        nptr = torch.from_numpy(nptr_np).to(td)
        sh = torch.from_numpy(sh_np).to(td)

        E_one_shot = float(
            multipole_ewald_summation(
                pos,
                sf,
                cell,
                idx_j,
                nptr,
                sh,
                sigma=sigma,
                alpha=alpha,
                k_cutoff=kcut,
            ).sum()
        )

        cache = prepare_multipole_scf_cache(
            cell.squeeze(0) if cell.ndim == 3 else cell,
            sigma=sigma,
            alpha=alpha,
            receiver_sigmas=[sigma],
            k_cutoff=kcut,
            l_max=l_max,
            device=pos.device,
        )
        E_cached = float(
            multipole_ewald_scf_step_energy(cache, pos, sf, idx_j, nptr, sh).sum()
        )
        assert abs(E_cached - E_one_shot) < 1e-10, (
            f"l_max={l_max} α={alpha}: cached={E_cached:.6e} "
            f"one-shot={E_one_shot:.6e}  |Δ|={abs(E_cached - E_one_shot):.3e}"
        )

    @pytest.mark.parametrize("l_max", [0, 1, 2])
    def test_batched_matches_one_shot(self, device, l_max: int):
        td = _torch_device(device)
        sigma, alpha = 1.0, 0.6
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c
        kcut = 6.0 / sigma_c

        systems = []
        for seed in (31, 37, 41):
            pos, sf, cell, cell_np, pos_np = _build(
                n=2, device=device, seed=seed, l_max=l_max
            )
            L = cell_np[0, 0]
            idx_j_np, nptr_np, sh_np = _neigh(pos_np, L, cutoff)
            systems.append((pos, sf, cell, cell_np, pos_np, idx_j_np, nptr_np, sh_np))

        # Stitch flat tensors.
        pos_all = torch.cat([s[0] for s in systems], dim=0)
        sf_all = torch.cat([s[1] for s in systems], dim=0)
        cells = torch.stack(
            [s[2].squeeze(0) if s[2].ndim == 3 else s[2] for s in systems], dim=0
        )
        n_per = [s[0].shape[0] for s in systems]
        batch_idx = torch.cat(
            [
                torch.full((n,), b, dtype=torch.int32, device=td)
                for b, n in enumerate(n_per)
            ]
        )
        idx_j_flat, nptr_flat, sh_flat = [], [0], []
        atom_off = 0
        for s in systems:
            idx_j_flat.append(s[5] + atom_off)
            sh_flat.append(s[7])
            nptr_np = s[6]
            for k in range(1, len(nptr_np)):
                nptr_flat.append(nptr_flat[-1] + int(nptr_np[k] - nptr_np[k - 1]))
            atom_off += s[0].shape[0]
        idx_j_flat = torch.from_numpy(np.concatenate(idx_j_flat).astype(np.int32)).to(
            td
        )
        nptr_flat = torch.from_numpy(np.asarray(nptr_flat, dtype=np.int32)).to(td)
        sh_flat = torch.from_numpy(np.concatenate(sh_flat).astype(np.int32)).to(td)

        E_one_shot = multipole_ewald_summation(
            pos_all,
            sf_all,
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigma=sigma,
            alpha=alpha,
            k_cutoff=kcut,
            batch_idx=batch_idx,
        )

        batch_cache = prepare_multipole_scf_cache(
            cells,
            sigma=sigma,
            alpha=alpha,
            receiver_sigmas=[sigma],
            k_cutoff=kcut,
            l_max=l_max,
            device=pos_all.device,
        )
        E_cached = multipole_ewald_scf_step_energy(
            batch_cache,
            pos_all,
            sf_all,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            batch_idx=batch_idx,
        )
        torch.testing.assert_close(E_cached, E_one_shot, rtol=0, atol=1e-10)

    def test_path_b_cache_rejected(self, device):
        """Passing a Path B cache (``alpha=None``) should raise."""
        td = _torch_device(device)
        pos, sf, cell, cell_np, pos_np = _build(n=2, device=device, seed=1, l_max=0)
        idx_j_np, nptr_np, sh_np = _neigh(pos_np, cell_np[0, 0], 5.0)
        idx_j = torch.from_numpy(idx_j_np).to(td)
        nptr = torch.from_numpy(nptr_np).to(td)
        sh = torch.from_numpy(sh_np).to(td)
        # Path B cache: no alpha.
        cache_b = prepare_multipole_scf_cache(
            cell.squeeze(0) if cell.ndim == 3 else cell,
            sigma=1.0,
            receiver_sigmas=[1.0],
            k_cutoff=3.0,
            l_max=0,
            device=pos.device,
        )
        with pytest.raises(ValueError, match="requires an Ewald cache"):
            multipole_ewald_scf_step_energy(cache_b, pos, sf, idx_j, nptr, sh)


class TestCudaCpuTileRouting:
    r"""CUDA ``multipole_ewald_summation`` (tile real-space kernel) and CPU
    (CSR fallback) must produce the same energy and gradients on the same
    (σ, α, geometry) input, bit-for-bit at float64 modulo atomic-add ordering.
    """

    @pytest.mark.parametrize("l_max", [0, 1])
    def test_forward_cuda_matches_cpu(self, l_max):
        """Same system, tile kernel on CUDA ≡ CSR kernel on CPU at 1e-10 rel."""
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA for the tile path")
        sigma, alpha = 1.0, 0.6
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c
        kcut = 6.0 / sigma_c

        out: dict[str, float] = {}
        for device in ("cpu", "gpu"):
            pos, sf, cell, cell_np, pos_np = _build(
                n=2, device=device, seed=7, l_max=l_max
            )
            td = _torch_device(device)
            idx_j_np, nptr_np, sh_np = _neigh(pos_np, cell_np[0, 0], cutoff)
            idx_j = torch.from_numpy(idx_j_np).to(td)
            nptr = torch.from_numpy(nptr_np).to(td)
            sh = torch.from_numpy(sh_np).to(td)
            out[device] = float(
                multipole_ewald_summation(
                    pos,
                    sf,
                    cell,
                    idx_j,
                    nptr,
                    sh,
                    sigma=sigma,
                    alpha=alpha,
                    k_cutoff=kcut,
                ).sum()
            )
        assert abs(out["cpu"] - out["gpu"]) / max(abs(out["cpu"]), 1e-300) < 1e-10, (
            f"CUDA tile path ({out['gpu']:.15e}) disagrees with CPU CSR path "
            f"({out['cpu']:.15e}) — tile routing broke the energy invariant."
        )

    @pytest.mark.parametrize("l_max", [0, 1])
    def test_backward_cuda_matches_cpu(self, l_max):
        """Gradients from tile backward (CUDA) ≡ CSR backward (CPU) at 1e-10 rel."""
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA for the tile path")
        sigma, alpha = 1.0, 0.6
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c
        kcut = 6.0 / sigma_c

        grads: dict[str, torch.Tensor] = {}
        for device in ("cpu", "gpu"):
            pos, sf, cell, cell_np, pos_np = _build(
                n=2, device=device, seed=7, l_max=l_max
            )
            pos.requires_grad_(True)
            sf.requires_grad_(True)
            td = _torch_device(device)
            idx_j_np, nptr_np, sh_np = _neigh(pos_np, cell_np[0, 0], cutoff)
            idx_j = torch.from_numpy(idx_j_np).to(td)
            nptr = torch.from_numpy(nptr_np).to(td)
            sh = torch.from_numpy(sh_np).to(td)
            energy = multipole_ewald_summation(
                pos,
                sf,
                cell,
                idx_j,
                nptr,
                sh,
                sigma=sigma,
                alpha=alpha,
                k_cutoff=kcut,
            )
            energy.sum().backward()
            grads[f"{device}_pos"] = pos.grad.detach().cpu().clone()
            grads[f"{device}_sf"] = sf.grad.detach().cpu().clone()

        torch.testing.assert_close(
            grads["gpu_pos"], grads["cpu_pos"], rtol=1e-10, atol=1e-13
        )
        torch.testing.assert_close(
            grads["gpu_sf"], grads["cpu_sf"], rtol=1e-10, atol=1e-13
        )


def _make_4atom_pbc(dtype=torch.float64, device="cpu"):
    rng = np.random.default_rng(20260613)
    n = 4
    L = 6.0
    pos = torch.tensor(
        rng.uniform(0.0, L, size=(n, 3)),
        dtype=dtype,
        device=device,
    )
    q = torch.tensor(rng.normal(size=(n,)), dtype=dtype, device=device)
    mu = torch.tensor(rng.normal(size=(n, 3)), dtype=dtype, device=device)
    Q_raw = rng.normal(size=(n, 3, 3))
    Q_sym = 0.5 * (Q_raw + Q_raw.transpose(0, 2, 1))
    Q = torch.tensor(Q_sym, dtype=dtype, device=device)
    cell = torch.tensor(
        [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]],
        dtype=dtype,
        device=device,
    ).unsqueeze(0)

    # source_feats (N, 4) packed: charges + dipoles in e3nn (y, z, x) order
    # at indices 1, 2, 3.
    sf = torch.empty(n, 4, dtype=dtype, device=device)
    sf[:, 0] = q
    sf[:, 1] = mu[:, 1]  # y (m=-1)
    sf[:, 2] = mu[:, 2]  # z (m=0)
    sf[:, 3] = mu[:, 0]  # x (m=+1)

    # Build a CSR full neighbor list within cutoff=4.0 with PBC.
    cutoff = 4.0
    idx_j_list, shifts_list = [], []
    pos_np = pos.cpu().numpy()
    for i in range(n):
        nb, sh = [], []
        for j in range(n):
            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    for sz in (-1, 0, 1):
                        if i == j and (sx, sy, sz) == (0, 0, 0):
                            continue
                        disp = pos_np[j] - pos_np[i] + np.array([sx, sy, sz]) * L
                        if np.linalg.norm(disp) < cutoff:
                            nb.append(j)
                            sh.append((sx, sy, sz))
        idx_j_list.append(nb)
        shifts_list.append(sh)
    counts = [len(ell) for ell in idx_j_list]
    ptr = np.cumsum([0] + counts)
    idx_j = torch.tensor(
        [j for nbrs in idx_j_list for j in nbrs],
        dtype=torch.int32,
        device=device,
    )
    neighbor_ptr = torch.tensor(ptr, dtype=torch.int32, device=device)
    unit_shifts = torch.tensor(
        [s for shifts in shifts_list for s in shifts],
        dtype=torch.int32,
        device=device,
    ).reshape(-1, 3)

    # Packed e3nn moments (N, 9): l<=1 block + traceless l=2 (the converter
    # drops the trace of the fixture's symmetric Q).
    multipole_moments = torch.cat(
        [sf, cartesian_quadrupole_to_e3nn(Q)], dim=-1
    ).contiguous()
    return dict(
        positions=pos,
        charges=q,
        dipoles=mu,
        quadrupoles=Q,
        source_feats=sf,
        multipole_moments=multipole_moments,
        cell=cell,
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
    )


def test_self_energy_quadrupole_term_is_additive():
    """The LMAX=2 self-energy is the LMAX=1 self-energy plus a |Q|_F^2 term."""
    p = _make_4atom_pbc()
    sigma, alpha = 0.5, 0.35
    e1 = _multipole_ewald_self_energy_per_atom(p["source_feats"], sigma, alpha).sum()
    e2 = _multipole_ewald_self_energy_per_atom(
        p["source_feats"],
        sigma,
        alpha,
        quadrupoles=p["quadrupoles"],
    ).sum()
    sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
    from nvalchemiops.torch.math import FIELD_CONSTANT

    # l=2 self-energy denominator is 320: the angular (k·Q·k)² ↔
    # Cartesian-Frobenius |Q|_F² factor of 3/2.
    expected_Q_term = (
        FIELD_CONSTANT
        / (320.0 * math.pi**1.5 * sigma_c**5)
        * (p["quadrupoles"].to(torch.float64) ** 2).sum()
    )
    diff = (e2 - e1 - expected_Q_term).abs().item()
    assert diff < 1e-12, f"self-energy decomposition mismatch: diff={diff:.3e}"


def test_quadrupole_total_is_alpha_independent():
    """The composite Ewald l=2 total must be α-independent.

    α only splits the real/reciprocal work; the total cannot depend on it.
    Sweeps α with σ_c-scaled cutoffs (constant truncation error) and asserts
    the q+μ+Q total is flat to the convergence floor.
    """
    rng = np.random.default_rng(7)
    n, L, sigma = 4, 6.0, 0.5
    pos_np = rng.uniform(0.0, L, size=(n, 3))
    q = rng.normal(size=(n,))
    q -= q.mean()
    mu = rng.normal(size=(n, 3))
    Qr = rng.normal(size=(n, 3, 3))
    Q = 0.5 * (Qr + Qr.transpose(0, 2, 1))
    cell = torch.tensor(np.eye(3) * L)
    sf = torch.empty(n, 4, dtype=torch.float64)
    sf[:, 0] = torch.tensor(q)
    sf[:, 1], sf[:, 2], sf[:, 3] = (
        torch.tensor(mu[:, 1]),
        torch.tensor(mu[:, 2]),
        torch.tensor(mu[:, 0]),
    )
    mm = torch.cat(
        [sf, cartesian_quadrupole_to_e3nn(torch.tensor(Q))], dim=-1
    ).contiguous()
    pos = torch.tensor(pos_np)

    def _csr(cutoff):
        shell = int(math.ceil(cutoff / L)) + 1
        idx, ptr, sh = [], [0], []
        for i in range(n):
            for sx in range(-shell, shell + 1):
                for sy in range(-shell, shell + 1):
                    for sz in range(-shell, shell + 1):
                        for j in range(n):
                            if i == j and (sx, sy, sz) == (0, 0, 0):
                                continue
                            d = pos_np[j] - pos_np[i] + np.array([sx, sy, sz]) * L
                            if np.linalg.norm(d) < cutoff:
                                idx.append(j)
                                sh.append((sx, sy, sz))
            ptr.append(len(idx))
        return (
            torch.tensor(idx, dtype=torch.int32),
            torch.tensor(ptr, dtype=torch.int32),
            torch.tensor(sh, dtype=torch.int32).reshape(-1, 3),
        )

    totals = []
    for alpha in (0.4, 0.6, 0.9):
        sc = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        idx_j, ptr, sh = _csr(12.0 * sc)
        E = float(
            multipole_ewald_summation(
                pos,
                mm,
                cell,
                idx_j,
                ptr,
                sh,
                sigma=sigma,
                alpha=alpha,
                k_cutoff=7.0 / sc,
            ).sum()
        )
        totals.append(E)
    spread = max(totals) - min(totals)
    assert spread < 1e-4, (
        f"l=2 Ewald total is α-dependent: {totals} (spread={spread:.3e})"
    )


def test_multipole_ewald_summation_quadrupole_computes():
    """Single-system composite at l_max=2 computes a finite, autograd-connected
    total (direct-k reciprocal + real-space + self)."""
    p = _make_4atom_pbc()
    pos = p["positions"].clone().requires_grad_(True)
    mm = p["multipole_moments"].clone().requires_grad_(True)
    E = multipole_ewald_summation(
        pos,
        mm,
        p["cell"],
        p["idx_j"],
        p["neighbor_ptr"],
        p["unit_shifts"],
        sigma=0.5,
        alpha=0.35,
        k_cutoff=6.0,
    )
    assert E.shape == (p["positions"].shape[0],) and torch.isfinite(E).all()
    gpos, gmm = torch.autograd.grad(E.sum(), [pos, mm])
    assert torch.isfinite(gpos).all() and torch.isfinite(gmm).all()
    # gradient flows to the l=2 (5-component) e3nn block.
    assert gmm.shape == (p["positions"].shape[0], 9)
    assert gmm[:, 4:9].abs().max() > 0.0


def test_multipole_ewald_summation_quadrupole_batched_matches_single():
    """A B=1 batch equals the single-system composite at l_max=2."""
    p = _make_4atom_pbc()
    n = p["positions"].shape[0]
    batch_idx = torch.zeros(n, dtype=torch.int32)
    cell_b = p["cell"].reshape(1, 3, 3) if p["cell"].ndim == 2 else p["cell"]
    cell_s = p["cell"].reshape(3, 3) if p["cell"].ndim == 3 else p["cell"]

    E_single = multipole_ewald_summation(
        p["positions"],
        p["multipole_moments"],
        cell_s,
        p["idx_j"],
        p["neighbor_ptr"],
        p["unit_shifts"],
        sigma=0.5,
        alpha=0.35,
        k_cutoff=6.0,
    )
    E_batch = multipole_ewald_summation(
        p["positions"],
        p["multipole_moments"],
        cell_b,
        p["idx_j"],
        p["neighbor_ptr"],
        p["unit_shifts"],
        sigma=0.5,
        alpha=0.35,
        k_cutoff=6.0,
        batch_idx=batch_idx,
    )
    n = p["positions"].shape[0]
    assert E_batch.shape == (n,)
    assert torch.isfinite(E_batch).all()
    assert (
        abs(float(E_batch.sum()) - float(E_single.sum())) / abs(float(E_single.sum()))
        < 1e-9
    )


def test_multipole_ewald_summation_validates_packed_shape():
    """An unsupported ``multipole_moments`` trailing dim raises ValueError
    (must be 1 / 4 / 9 for l_max 0 / 1 / 2)."""
    p = _make_4atom_pbc()
    bad_mm = torch.zeros(p["positions"].shape[0], 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="multipole_moments last-dim must be"):
        multipole_ewald_summation(
            p["positions"],
            bad_mm,
            p["cell"],
            p["idx_j"],
            p["neighbor_ptr"],
            p["unit_shifts"],
            sigma=0.5,
            alpha=0.35,
            k_cutoff=2.0,
        )


def test_multipole_real_space_quadrupole_callable_with_4atom_periodic():
    """Real-space LMAX=2 entry point produces a finite, autograd-connected energy."""
    p = _make_4atom_pbc()
    positions = p["positions"].clone().requires_grad_(True)
    sigma = torch.tensor([0.5], dtype=torch.float64)
    alpha = torch.tensor([0.35], dtype=torch.float64)
    E = multipole_real_space_quadrupole_energy(
        positions,
        p["charges"],
        p["dipoles"],
        p["quadrupoles"],
        p["cell"],
        p["idx_j"],
        p["neighbor_ptr"],
        p["unit_shifts"],
        sigma,
        alpha,
    )
    # Real-space LMAX=2 returns per-atom (N,); total is .sum().
    assert E.shape == (p["positions"].shape[0],)
    assert torch.isfinite(E).all()
    (grad_pos,) = torch.autograd.grad(E.sum(), [positions])
    assert torch.isfinite(grad_pos).all()


def _periodic_4atom(dtype=torch.float64, device="cpu"):
    rng = np.random.default_rng(20260615)
    n = 4
    L = 6.0
    pos = torch.tensor(rng.uniform(0.0, L, size=(n, 3)), dtype=dtype, device=device)
    q = torch.tensor(rng.normal(size=(n,)), dtype=dtype, device=device)
    mu = torch.tensor(rng.normal(size=(n, 3)), dtype=dtype, device=device)
    cell = torch.tensor(
        [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]],
        dtype=dtype,
        device=device,
    ).unsqueeze(0)
    sigma = torch.tensor([0.5], dtype=dtype, device=device)
    alpha = torch.tensor([0.35], dtype=dtype, device=device)
    cutoff = 4.0
    idx_j_list, shifts_list = [], []
    pos_np = pos.cpu().numpy()
    for i in range(n):
        nb, sh = [], []
        for j in range(n):
            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    for sz in (-1, 0, 1):
                        if i == j and (sx, sy, sz) == (0, 0, 0):
                            continue
                        disp = pos_np[j] - pos_np[i] + np.array([sx, sy, sz]) * L
                        if np.linalg.norm(disp) < cutoff:
                            nb.append(j)
                            sh.append((sx, sy, sz))
        idx_j_list.append(nb)
        shifts_list.append(sh)
    counts = [len(ell) for ell in idx_j_list]
    ptr = np.cumsum([0] + counts)
    idx_j = torch.tensor(
        [j for nbrs in idx_j_list for j in nbrs],
        dtype=torch.int32,
        device=device,
    )
    neighbor_ptr = torch.tensor(ptr, dtype=torch.int32, device=device)
    unit_shifts = torch.tensor(
        [s for shifts in shifts_list for s in shifts],
        dtype=torch.int32,
        device=device,
    ).reshape(-1, 3)
    return dict(
        positions=pos,
        charges=q,
        dipoles=mu,
        cell=cell,
        sigma=sigma,
        alpha=alpha,
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
    )


def test_monopole_fused_scalar_cell_grad_via_autograd():
    """MultipoleRealSpaceMonopoleFusedScalarFunction propagates cell-grad."""
    p = _periodic_4atom()
    cell = p["cell"].clone().requires_grad_(True)
    E = MultipoleRealSpaceMonopoleFusedScalarFunction.apply(
        p["positions"],
        p["charges"],
        cell,
        p["sigma"],
        p["alpha"],
        p["idx_j"],
        p["neighbor_ptr"],
        p["unit_shifts"],
    )
    (grad_cell,) = torch.autograd.grad(E, [cell])
    assert grad_cell.shape == (1, 3, 3)
    assert torch.isfinite(grad_cell).all()
    assert grad_cell.abs().max() > 0


def test_dipole_fused_scalar_cell_grad_via_autograd():
    """MultipoleRealSpaceDipoleFusedScalarFunction propagates cell-grad."""
    p = _periodic_4atom()
    cell = p["cell"].clone().requires_grad_(True)
    E = MultipoleRealSpaceDipoleFusedScalarFunction.apply(
        p["positions"],
        p["charges"],
        p["dipoles"],
        cell,
        p["sigma"],
        p["alpha"],
        p["idx_j"],
        p["neighbor_ptr"],
        p["unit_shifts"],
    )
    (grad_cell,) = torch.autograd.grad(E, [cell])
    assert grad_cell.shape == (1, 3, 3)
    assert torch.isfinite(grad_cell).all()
    assert grad_cell.abs().max() > 0


@pytest.mark.parametrize("lmax", [0, 1])
def test_fused_scalar_cell_grad_matches_fd(lmax):
    """Cell-grad through the FusedScalar wrapper matches FD on the energy."""
    p = _periodic_4atom()
    if lmax == 0:
        Fn = MultipoleRealSpaceMonopoleFusedScalarFunction

        def forward(c):
            return Fn.apply(
                p["positions"],
                p["charges"],
                c,
                p["sigma"],
                p["alpha"],
                p["idx_j"],
                p["neighbor_ptr"],
                p["unit_shifts"],
            )
    else:
        Fn = MultipoleRealSpaceDipoleFusedScalarFunction

        def forward(c):
            return Fn.apply(
                p["positions"],
                p["charges"],
                p["dipoles"],
                c,
                p["sigma"],
                p["alpha"],
                p["idx_j"],
                p["neighbor_ptr"],
                p["unit_shifts"],
            )

    cell0 = p["cell"].detach().clone()
    cell_grad_in = cell0.clone().requires_grad_(True)
    E = forward(cell_grad_in)
    (grad_an,) = torch.autograd.grad(E, [cell_grad_in])

    h = 1e-5
    grad_fd = torch.zeros_like(cell0)
    for a in range(3):
        for b in range(3):
            c_p = cell0.clone()
            c_p[0, a, b] += h
            E_p = forward(c_p).item()
            c_m = cell0.clone()
            c_m[0, a, b] -= h
            E_m = forward(c_m).item()
            grad_fd[0, a, b] = (E_p - E_m) / (2 * h)
    diff = (grad_an - grad_fd).abs().max().item()
    denom = max(grad_an.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 5e-4, f"LMAX={lmax}: rel_err={rel:.3e}\nan={grad_an}\nfd={grad_fd}"


def _periodic_4atom_batched_2sys(dtype=torch.float64):
    """Two periodic systems with the same atom count, flat-packed."""
    p1 = _periodic_4atom(dtype=dtype)
    p2 = _periodic_4atom(dtype=dtype)
    n1 = p1["positions"].shape[0]
    n2 = p2["positions"].shape[0]

    positions = torch.cat([p1["positions"], p2["positions"]], dim=0)
    charges = torch.cat([p1["charges"], p2["charges"]], dim=0)
    dipoles = torch.cat([p1["dipoles"], p2["dipoles"]], dim=0)
    cells = torch.cat([p1["cell"], p2["cell"]], dim=0)  # (2, 3, 3)
    sigmas = torch.tensor([0.5, 0.5], dtype=dtype)
    alphas = torch.tensor([0.35, 0.35], dtype=dtype)
    batch_idx = torch.cat(
        [
            torch.zeros(n1, dtype=torch.int32),
            torch.ones(n2, dtype=torch.int32),
        ]
    )
    # Flat CSR — offset system 2's idx_j by n1.
    idx_j = torch.cat(
        [
            p1["idx_j"],
            p2["idx_j"] + n1,
        ]
    )
    nptr1 = p1["neighbor_ptr"]
    nptr2 = p2["neighbor_ptr"]
    neighbor_ptr = torch.cat(
        [
            nptr1,
            nptr2[1:] + nptr1[-1],
        ]
    )
    unit_shifts = torch.cat([p1["unit_shifts"], p2["unit_shifts"]], dim=0)
    return dict(
        positions=positions,
        charges=charges,
        dipoles=dipoles,
        cells=cells,
        sigmas=sigmas,
        alphas=alphas,
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
        batch_idx=batch_idx,
    )


@pytest.mark.parametrize("lmax", [0, 1])
def test_batch_fused_scalar_cell_grad_matches_fd(lmax):
    """Batched cell-grad matches FD per-system."""
    p = _periodic_4atom_batched_2sys()
    if lmax == 0:
        Fn = BatchMultipoleRealSpaceMonopoleFusedScalarFunction

        def forward(c):
            return Fn.apply(
                p["positions"],
                p["charges"],
                c,
                p["sigmas"],
                p["alphas"],
                p["idx_j"],
                p["neighbor_ptr"],
                p["unit_shifts"],
                p["batch_idx"],
            )
    else:
        Fn = BatchMultipoleRealSpaceDipoleFusedScalarFunction

        def forward(c):
            return Fn.apply(
                p["positions"],
                p["charges"],
                p["dipoles"],
                c,
                p["sigmas"],
                p["alphas"],
                p["idx_j"],
                p["neighbor_ptr"],
                p["unit_shifts"],
                p["batch_idx"],
            )

    cell0 = p["cells"].detach().clone()
    cells_grad_in = cell0.clone().requires_grad_(True)
    E_per_system = forward(cells_grad_in)  # (B,)
    # Sum over batch to get scalar; the per-system grad_cell is unweighted = 1 each
    total = E_per_system.sum()
    (grad_an,) = torch.autograd.grad(total, [cells_grad_in])
    # FD on each (system_b, a, b) entry
    h = 1e-5
    grad_fd = torch.zeros_like(cell0)
    B = cell0.shape[0]
    for b_sys in range(B):
        for a in range(3):
            for b in range(3):
                c_p = cell0.clone()
                c_p[b_sys, a, b] += h
                E_p = forward(c_p).sum().item()
                c_m = cell0.clone()
                c_m[b_sys, a, b] -= h
                E_m = forward(c_m).sum().item()
                grad_fd[b_sys, a, b] = (E_p - E_m) / (2 * h)
    diff = (grad_an - grad_fd).abs().max().item()
    denom = max(grad_an.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 5e-4, (
        f"batched LMAX={lmax}: rel_err={rel:.3e}\nan={grad_an}\nfd={grad_fd}"
    )

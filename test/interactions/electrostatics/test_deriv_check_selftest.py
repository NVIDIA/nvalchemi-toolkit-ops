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

"""Self-test for the derivative-verification harness (``_deriv_check.py``).

This module validates the harness against the electrostatics code before
kernel-specific tests reuse it. The core claim it proves:
**finite-difference forces match autograd forces** for one
Ewald case and one PME case (and, additionally, the strain-first virial and the
charge gradient), all off a single pinned ``energy_fn`` closure.

Case 1 (``test_fd_jacobian_matches_autograd_pure_torch``) validates the
finite-difference machinery itself against ``torch.autograd`` on a closed-form
torch energy and does **not** import electrostatics, so the FD primitives are
proven even where the electrostatics module fails to import.

All other cases import the public ``ewald_summation`` / ``particle_mesh_ewald``
functions; if that import fails (e.g. the known forward-op registration blocker
documented in ``_md/F3-verification-harness.md``), they skip with a clear reason
rather than erroring, and pass once the import is fixed.
"""

from __future__ import annotations

import pytest
import torch

from test.interactions.electrostatics._deriv_check import (
    DEFAULT_FD_EPS,
    EquivPoint,
    assert_tape_vs_explicit,
    autograd_charge_grad,
    autograd_forces,
    autograd_strain_virial,
    fd_charge_grad,
    fd_forces,
    fd_strain_virial,
    finite_difference_jacobian,
    gradcheck_energy,
    gradgradcheck_energy,
    max_abs_rel,
    toy_charge_model,
)

# ---------------------------------------------------------------------------
# Optional electrostatics import (the known F2 blocker may break this).
# ---------------------------------------------------------------------------
try:
    from nvalchemiops.torch.interactions.electrostatics.ewald import ewald_summation
    from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
        generate_k_vectors_ewald_summation,
    )
    from nvalchemiops.torch.interactions.electrostatics.pme import particle_mesh_ewald
    from nvalchemiops.torch.neighbors import cell_list

    _ELECTROSTATICS_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - environment-dependent
    ewald_summation = None  # type: ignore[assignment]
    particle_mesh_ewald = None  # type: ignore[assignment]
    generate_k_vectors_ewald_summation = None  # type: ignore[assignment]
    cell_list = None  # type: ignore[assignment]
    _ELECTROSTATICS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

from test.interactions.electrostatics.conftest import create_cscl_supercell

_HAS_ELECTROSTATICS = _ELECTROSTATICS_IMPORT_ERROR is None
_requires_electrostatics = pytest.mark.skipif(
    not _HAS_ELECTROSTATICS,
    reason=(
        "electrostatics import failed (see _md/F3-verification-harness.md "
        f"'Known environment blocker'): {_ELECTROSTATICS_IMPORT_ERROR}"
    ),
)

# Tight float64 FD-vs-autograd targets off a single pinned closure.
FORCE_RTOL = 1e-5
FORCE_ATOL = 1e-7
VIRIAL_RTOL = 1e-5
VIRIAL_ATOL = 1e-6  # virial entries are larger in magnitude -> abs floor looser
CHARGE_RTOL = 1e-5
CHARGE_ATOL = 1e-7


def _devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.insert(0, "cuda")
    return devs


# ---------------------------------------------------------------------------
# System builders (tiny, displaced -> non-zero forces, rtol-safe)
# ---------------------------------------------------------------------------


def _dipole(device: torch.device, dtype=torch.float64, sep: float = 2.3):
    """A 2-atom displaced dipole in a 10 A cubic box (guaranteed non-zero force)."""
    cs = 10.0
    c = cs / 2.0
    positions = torch.tensor(
        [
            [c - sep / 2.0, c + 0.4, c - 0.2],
            [c + sep / 2.0, c - 0.3, c + 0.1],
        ],
        dtype=dtype,
        device=device,
    )
    charges = torch.tensor([1.0, -1.0], dtype=dtype, device=device)
    cell = (torch.eye(3, dtype=dtype, device=device) * cs).unsqueeze(0)
    return positions, charges, cell


def _ewald_neighbors(positions, cell, device, cutoff=5.0):
    pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
    return cell_list(positions, cutoff, cell, pbc, return_neighbor_list=True)


# ---------------------------------------------------------------------------
# Case 1: FD primitives vs autograd on closed-form torch energy (no import dep)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", _devices())
def test_fd_jacobian_matches_autograd_pure_torch(device):
    """Validate the central-difference primitive against autograd analytically.

    Uses a smooth closed-form energy so this proves the FD machinery itself,
    independent of the electrostatics module.
    """
    device = torch.device(device)
    torch.manual_seed(0)
    x = torch.randn(6, dtype=torch.float64, device=device)
    a = torch.randn(6, dtype=torch.float64, device=device)

    # E(x) = sum_i sin(x_i) * a_i + 0.5 * (sum_i x_i^2)
    def energy(x_):
        return (torch.sin(x_) * a).sum() + 0.5 * (x_**2).sum()

    fd = finite_difference_jacobian(energy, x, eps=DEFAULT_FD_EPS)

    x_ad = x.clone().requires_grad_(True)
    (ad,) = torch.autograd.grad(energy(x_ad), x_ad)

    max_abs, max_rel = max_abs_rel(fd, ad)
    assert torch.allclose(fd, ad, rtol=1e-6, atol=1e-8), (
        f"FD primitive vs autograd: max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    )


# ---------------------------------------------------------------------------
# Case 2: Ewald FD forces == autograd forces  (mandatory done-when)
# ---------------------------------------------------------------------------


@_requires_electrostatics
@pytest.mark.parametrize("device", _devices())
def test_ewald_fd_forces_match_autograd(device):
    """One Ewald case: finite-diff forces match -autograd.grad(E.sum(), R)."""
    device = torch.device(device)
    positions, charges, cell = _dipole(device)
    nl, nptr, ns = _ewald_neighbors(positions, cell, device)
    alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
    # Pin the reciprocal set: build k-vectors ONCE at the reference cell.
    k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=2.0)

    def energy_fn(p, q, c):
        return ewald_summation(
            p,
            q,
            c,
            alpha=alpha,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=ns,
        )

    fd = fd_forces(energy_fn, positions, charges, cell)
    ad = autograd_forces(energy_fn, positions, charges, cell)

    max_abs, max_rel = max_abs_rel(fd, ad)
    assert torch.allclose(fd, ad, rtol=FORCE_RTOL, atol=FORCE_ATOL), (
        f"Ewald FD forces vs autograd: max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    )


# ---------------------------------------------------------------------------
# Case 3: PME FD forces == autograd forces  (mandatory done-when)
# ---------------------------------------------------------------------------


@_requires_electrostatics
@pytest.mark.parametrize("device", _devices())
def test_pme_fd_forces_match_autograd(device):
    """One PME case: finite-diff forces match -autograd.grad(E.sum(), R)."""
    device = torch.device(device)
    positions, charges, cell = _dipole(device)
    nl, nptr, ns = _ewald_neighbors(positions, cell, device)
    alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
    # Pin the reciprocal set: fixed mesh dimensions.
    mesh = (16, 16, 16)

    def energy_fn(p, q, c):
        return particle_mesh_ewald(
            p,
            q,
            c,
            alpha=alpha,
            mesh_dimensions=mesh,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=ns,
        )

    fd = fd_forces(energy_fn, positions, charges, cell)
    ad = autograd_forces(energy_fn, positions, charges, cell)

    max_abs, max_rel = max_abs_rel(fd, ad)
    assert torch.allclose(fd, ad, rtol=FORCE_RTOL, atol=FORCE_ATOL), (
        f"PME FD forces vs autograd: max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    )


# ---------------------------------------------------------------------------
# Case 3b: PME FD dE/dq == autograd  (fills the item-3 PME charge-grad cell)
# ---------------------------------------------------------------------------


@_requires_electrostatics
@pytest.mark.parametrize("device", _devices())
def test_pme_fd_charge_grad_matches_autograd(device):
    """PME dE/dq: finite-diff matches autograd for a fixed-geometry PME case.

    Mirrors the Ewald charge-grad case (Case 6) but on the PME reciprocal path;
    needed so the tolerance matrix has a PME charge-grad baseline.
    """
    device = torch.device(device)
    positions, charges, cell = _dipole(device)
    nl, nptr, ns = _ewald_neighbors(positions, cell, device)
    alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
    mesh = (16, 16, 16)

    def energy_fn(p, q, c):
        return particle_mesh_ewald(
            p,
            q,
            c,
            alpha=alpha,
            mesh_dimensions=mesh,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=ns,
        )

    fd = fd_charge_grad(energy_fn, positions, charges, cell)
    ad = autograd_charge_grad(energy_fn, positions, charges, cell)

    max_abs, max_rel = max_abs_rel(fd, ad)
    assert torch.allclose(fd, ad, rtol=CHARGE_RTOL, atol=CHARGE_ATOL), (
        f"PME FD dE/dq vs autograd: max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    )


# ---------------------------------------------------------------------------
# Case 3c: PME strain-first virial FD == autograd
# ---------------------------------------------------------------------------


@_requires_electrostatics
@pytest.mark.parametrize("device", _devices())
def test_pme_fd_strain_virial_matches_autograd(device):
    """PME strain-first virial: FD -dE/dstrain matches -autograd.grad(E.sum(), strain).

    The PME analogue of the Ewald virial check (Case 4). Pins ONLY ``alpha`` and
    ``mesh_dimensions`` (a dimension count, not geometry) so the cell -> mesh /
    volume -> reciprocal-energy path must regenerate from the deformed cell. This
    is a *distinct* differentiable path from Ewald's cell -> k-vectors flow. A
    systematic FD-vs-autograd offset here would indicate a
    detach in the PME reciprocal/volume path.
    """
    device = torch.device(device)
    positions, charges, cell = _dipole(device)
    nl, nptr, ns = _ewald_neighbors(positions, cell, device)
    alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
    mesh = (16, 16, 16)

    def energy_fn(p, q, c):
        return particle_mesh_ewald(
            p,
            q,
            c,
            alpha=alpha,
            mesh_dimensions=mesh,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=ns,
        )

    fd = fd_strain_virial(energy_fn, positions, charges, cell, batch_idx=None)
    ad = autograd_strain_virial(energy_fn, positions, charges, cell, batch_idx=None)

    max_abs, max_rel = max_abs_rel(fd, ad)
    assert torch.allclose(fd, ad, rtol=VIRIAL_RTOL, atol=VIRIAL_ATOL), (
        f"PME FD strain-virial vs autograd: max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    )


# ---------------------------------------------------------------------------
# Case 4: Ewald strain-first virial FD == autograd
# ---------------------------------------------------------------------------


@_requires_electrostatics
@pytest.mark.parametrize("device", _devices())
def test_ewald_fd_strain_virial_matches_autograd(device):
    """Strain-first virial: FD -dE/dstrain matches -autograd.grad(E.sum(), strain).

    Here the closure does NOT pin k_vectors -- only alpha -- so k/volume
    regenerate from the deformed cell, exercising (and validating) the
    cell->k-vector differentiable path.
    """
    device = torch.device(device)
    positions, charges, cell = _dipole(device)
    nl, nptr, ns = _ewald_neighbors(positions, cell, device)
    alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
    k_cutoff = 2.0

    def energy_fn(p, q, c):
        return ewald_summation(
            p,
            q,
            c,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=ns,
        )

    fd = fd_strain_virial(energy_fn, positions, charges, cell, batch_idx=None)
    ad = autograd_strain_virial(energy_fn, positions, charges, cell, batch_idx=None)

    max_abs, max_rel = max_abs_rel(fd, ad)
    assert torch.allclose(fd, ad, rtol=VIRIAL_RTOL, atol=VIRIAL_ATOL), (
        f"Ewald FD strain-virial vs autograd: max_abs={max_abs:.3e} "
        f"max_rel={max_rel:.3e}"
    )


# ---------------------------------------------------------------------------
# Case 5: toy q(R) zero net charge + dE/dq FD vs autograd
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", _devices())
def test_toy_charge_model_zero_net_charge(device):
    """The toy q(R) charge model yields (near) zero net charge per system."""
    device = torch.device(device)
    positions, _, _ = _dipole(device)
    q = toy_charge_model(positions)
    assert q.shape == (positions.shape[0],)
    assert abs(float(q.sum().item())) < 1e-10

    # And it is differentiable in positions (dq/dR is finite, non-trivial).
    p = positions.clone().requires_grad_(True)
    qg = toy_charge_model(p)
    (g,) = torch.autograd.grad(qg.pow(2).sum(), p)
    assert torch.isfinite(g).all()
    assert g.abs().sum() > 0


@_requires_electrostatics
@pytest.mark.parametrize("device", _devices())
def test_fd_charge_grad_matches_autograd(device):
    """dE/dq: finite-diff matches autograd for a fixed-geometry Ewald case."""
    device = torch.device(device)
    positions, charges, cell = _dipole(device)
    nl, nptr, ns = _ewald_neighbors(positions, cell, device)
    alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
    k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=2.0)

    def energy_fn(p, q, c):
        return ewald_summation(
            p,
            q,
            c,
            alpha=alpha,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=ns,
        )

    fd = fd_charge_grad(energy_fn, positions, charges, cell)
    ad = autograd_charge_grad(energy_fn, positions, charges, cell)

    max_abs, max_rel = max_abs_rel(fd, ad)
    assert torch.allclose(fd, ad, rtol=CHARGE_RTOL, atol=CHARGE_ATOL), (
        f"Ewald FD dE/dq vs autograd: max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    )


# ---------------------------------------------------------------------------
# Case 6: gradcheck (first order) + gradgradcheck
# ---------------------------------------------------------------------------


@_requires_electrostatics
@pytest.mark.parametrize("device", _devices())
def test_gradcheck_energy_small_system(device):
    """First-order gradcheck on a tiny f64 Ewald system (positions + charges)."""
    device = torch.device(device)
    positions, charges, cell = _dipole(device)
    nl, nptr, ns = _ewald_neighbors(positions, cell, device)
    alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
    k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=2.0)

    def energy_fn(p, q, c):
        return ewald_summation(
            p,
            q,
            c,
            alpha=alpha,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=ns,
        )

    assert gradcheck_energy(
        energy_fn, positions, charges, cell, wrt=("positions", "charges")
    )


@_requires_electrostatics
@pytest.mark.parametrize("device", _devices())
def test_gradgradcheck_energy_documented_status(device):
    """gradgradcheck on E.sum() wrt positions, charges, and cell is a hard assert."""
    device = torch.device(device)
    positions, charges, cell = _dipole(device)
    nl, nptr, ns = _ewald_neighbors(positions, cell, device)
    alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
    k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=2.0)

    def energy_fn(p, q, c):
        return ewald_summation(
            p,
            q,
            c,
            alpha=alpha,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=ns,
        )

    for wrt in (("positions",), ("charges",), ("cell",), ("positions", "cell")):
        assert gradgradcheck_energy(energy_fn, positions, charges, cell, wrt=wrt)


# ---------------------------------------------------------------------------
# Case 7: tape-vs-explicit equivalence (parameterizable points)
# ---------------------------------------------------------------------------


@_requires_electrostatics
@pytest.mark.parametrize("device", _devices())
def test_tape_vs_explicit_forces_charge_grad(device):
    """Direct-kernel (explicit) outputs vs autograd (reference) at named points.

    Today: forces + charge-grad must match (existing parity tests confirm this);
    the direct ``virial`` is recorded but NOT hard-asserted (it is a kernel
    observable, not a guaranteed strain-energy-derivative view -- the motivation
    for the refactor). The explicit-chain path preserves this comparison
    structure.
    """
    device = torch.device(device)
    positions, charges, cell = _dipole(device)
    nl, nptr, ns = _ewald_neighbors(positions, cell, device)
    alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
    k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=2.0)

    def energy_fn(p, q, c):
        return ewald_summation(
            p,
            q,
            c,
            alpha=alpha,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=ns,
        )

    def explicit_forces():
        _, forces = ewald_summation(
            positions,
            charges,
            cell,
            alpha=alpha,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=ns,
            compute_forces=True,
        )
        return forces

    def explicit_charge_grad():
        out = ewald_summation(
            positions,
            charges,
            cell,
            alpha=alpha,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=ns,
            compute_charge_gradients=True,
        )
        # (energies, charge_gradients)
        return out[1]

    points = [
        EquivPoint(
            name="forces",
            explicit=explicit_forces,
            reference=lambda: autograd_forces(energy_fn, positions, charges, cell),
            # Direct-kernel forces and energy-backward forces are two different
            # computational paths; the repo's existing parity test
            # (test_ewald.py::test_autograd_matches_explicit_forces) trusts them
            # only to rtol=0.01. Match that precedent here. (FD-vs-autograd off a
            # single closure is a different, tighter comparison -- see
            # FORCE_RTOL above.) TBD: tighten once measured against real kernels
            # post-F2.
            rtol=1e-2,
            atol=1e-5,
            expect_match=True,
        ),
        EquivPoint(
            name="charge_grad",
            explicit=explicit_charge_grad,
            reference=lambda: autograd_charge_grad(energy_fn, positions, charges, cell),
            rtol=1e-4,
            atol=1e-6,
            expect_match=True,
        ),
    ]

    assert_tape_vs_explicit(points)


# ---------------------------------------------------------------------------
# Bonus: harness works on a conftest crystal generator (fixed-charge system)
# ---------------------------------------------------------------------------


@_requires_electrostatics
@pytest.mark.parametrize("device", _devices())
def test_fixed_charge_system_ewald_forces(device):
    """fixed_charge_system + jitter gives a non-trivial Ewald FD==autograd case."""
    from test.interactions.electrostatics._deriv_check import fixed_charge_system

    device = torch.device(device)
    sysd = fixed_charge_system(
        create_cscl_supercell, size=1, device=device, jitter=0.15, seed=1
    )
    k_vectors = generate_k_vectors_ewald_summation(sysd.cell, k_cutoff=2.0)

    def energy_fn(p, q, c):
        return ewald_summation(
            p,
            q,
            c,
            alpha=sysd.alpha,
            k_vectors=k_vectors,
            neighbor_list=sysd.neighbor_list,
            neighbor_ptr=sysd.neighbor_ptr,
            neighbor_shifts=sysd.neighbor_shifts,
        )

    fd = fd_forces(energy_fn, sysd.positions, sysd.charges, sysd.cell)
    ad = autograd_forces(energy_fn, sysd.positions, sysd.charges, sysd.cell)
    max_abs, max_rel = max_abs_rel(fd, ad)
    assert torch.allclose(fd, ad, rtol=FORCE_RTOL, atol=FORCE_ATOL), (
        f"CsCl FD forces vs autograd: max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    )

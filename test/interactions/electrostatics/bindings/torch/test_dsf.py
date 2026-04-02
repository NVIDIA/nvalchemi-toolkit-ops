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
Unit tests for DSF (Damped Shifted Force) PyTorch bindings.

Tests compare the nvalchemiops Warp-based DSF implementation against
a pure PyTorch reference implementation (dsf_reference) that serves as
the single source of truth.

Tests cover:
- Energy, forces, virial computation via public API
- Charge gradient autograd support (dE/dq)
- Both neighbor list (CSR) and neighbor matrix formats
- Batched calculations
- Periodic boundary conditions
- Numerical stability and edge cases
- Input validation
- dtype and device handling
- Alpha=0 (shifted-force bare Coulomb)
- Regression tests
"""

from __future__ import annotations

import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics.dsf import dsf_coulomb

# ==============================================================================
# Pure PyTorch DSF Reference Implementation
# ==============================================================================


def dsf_reference(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cutoff: float,
    alpha: float,
    neighbor_list: torch.Tensor | None = None,
    neighbor_ptr: torch.Tensor | None = None,
    neighbor_matrix: torch.Tensor | None = None,
    fill_value: int | None = None,
    cell: torch.Tensor | None = None,
    unit_shifts: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    num_systems: int = 1,
    compute_forces: bool = True,
    compute_virial: bool = False,
) -> dict[str, torch.Tensor]:
    """Pure PyTorch DSF reference implementation.

    Runs in input precision. Uses autograd for forces, virial, and charge
    gradients. Virial follows the torch binding convention ``W = -dE/dε``.
    Returns a dict with keys: energy, forces, virial, charge_grad.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
    charges : torch.Tensor, shape (N,)
        Must match positions dtype.
    cutoff : float
    alpha : float
    neighbor_list : torch.Tensor, shape (2, E), optional
    neighbor_ptr : torch.Tensor, shape (N+1,), optional
    neighbor_matrix : torch.Tensor, shape (N, M), optional
    fill_value : int, optional
    cell : torch.Tensor, shape (B, 3, 3), optional
    unit_shifts : torch.Tensor, shape (E, 3), optional
    neighbor_matrix_shifts : torch.Tensor, shape (N, M, 3), optional
    batch_idx : torch.Tensor, shape (N,), optional
    num_systems : int
    compute_forces : bool
    compute_virial : bool

    Returns
    -------
    dict with keys:
        energy : torch.Tensor, shape (num_systems,)
        forces : torch.Tensor, shape (N, 3)
        virial : torch.Tensor, shape (num_systems, 3, 3)
        charge_grad : torch.Tensor, shape (N,)
    """
    assert charges.dtype == positions.dtype, (
        f"charges dtype ({charges.dtype}) must match positions dtype ({positions.dtype})"
    )
    device = positions.device
    dtype = positions.dtype
    N = positions.shape[0]

    if batch_idx is None:
        batch_idx = torch.zeros(N, dtype=torch.long, device=device)
    else:
        batch_idx = batch_idx.long()

    # Clone inputs and enable grad for autograd-based derivatives
    pos = positions.detach().clone().requires_grad_(True)
    q = charges.detach().clone().requires_grad_(True)

    # For virial, parameterize through a strain tensor
    grad_targets = [pos, q]
    if compute_virial and cell is not None:
        strain = torch.zeros(
            num_systems, 3, 3, dtype=dtype, device=device, requires_grad=True
        )
        deform = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) + strain
        deform_per_atom = deform[batch_idx]  # (N, 3, 3)
        pos_c = torch.bmm(pos.unsqueeze(1), deform_per_atom.transpose(1, 2)).squeeze(1)
        cell_c = torch.bmm(cell.to(dtype=dtype), deform.transpose(1, 2))
        grad_targets.append(strain)
    else:
        pos_c = pos
        cell_c = cell.to(dtype=dtype) if cell is not None else None
        strain = None

    # --- Build flat pair indices (idx_i, idx_j) and shifts from either format ---
    if neighbor_list is not None:
        idx_i = neighbor_list[0].long()
        idx_j = neighbor_list[1].long()
        if unit_shifts is not None:
            pair_shifts = unit_shifts.to(dtype=dtype)  # (E, 3)
        else:
            pair_shifts = None
    elif neighbor_matrix is not None:
        # Flatten matrix format to edge list, filtering fill_value
        if fill_value is None:
            fill_value = N
        M = neighbor_matrix.shape[1]
        atom_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, M)
        mask_valid = neighbor_matrix != fill_value
        idx_i = atom_idx[mask_valid].long()
        idx_j = neighbor_matrix[mask_valid].long()
        if neighbor_matrix_shifts is not None:
            pair_shifts = neighbor_matrix_shifts[mask_valid].to(dtype=dtype)
        else:
            pair_shifts = None
    else:
        # No neighbors: only self-energy
        idx_i = torch.zeros(0, dtype=torch.long, device=device)
        idx_j = torch.zeros(0, dtype=torch.long, device=device)
        pair_shifts = None

    num_pairs = idx_i.shape[0]

    # --- Compute displacement vectors and distances ---
    if num_pairs > 0:
        pos_i = torch.index_select(pos_c, 0, idx_i)
        pos_j = torch.index_select(pos_c, 0, idx_j)
        r_ij = pos_j - pos_i

        # Apply PBC shifts
        if cell_c is not None and pair_shifts is not None:
            batch_i = torch.index_select(batch_idx, 0, idx_i)
            cell_per_pair = torch.index_select(cell_c, 0, batch_i)
            shift_cart = torch.bmm(pair_shifts.unsqueeze(1), cell_per_pair).squeeze(1)
            r_ij = r_ij + shift_cart

        dist = torch.norm(r_ij, dim=1)

        # Filter to within-cutoff pairs ONCE
        mask = dist < cutoff
        r_ij = r_ij[mask]
        dist = dist[mask]
        idx_i_f = idx_i[mask]
        idx_j_f = idx_j[mask]
    else:
        r_ij = torch.zeros((0, 3), dtype=dtype, device=device)
        dist = torch.zeros(0, dtype=dtype, device=device)
        idx_i_f = torch.zeros(0, dtype=torch.long, device=device)
        idx_j_f = torch.zeros(0, dtype=torch.long, device=device)

    # --- Precompute cutoff constants ---
    alpha_t = torch.tensor(alpha, dtype=dtype, device=device)
    cutoff_t = torch.tensor(cutoff, dtype=dtype, device=device)
    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, dtype=dtype, device=device))

    if alpha > 0.0:
        erfc_Rc = torch.erfc(alpha_t * cutoff_t)
        exp_Rc = torch.exp(-(alpha_t**2) * cutoff_t**2)
    else:
        erfc_Rc = torch.ones(1, dtype=dtype, device=device)
        exp_Rc = torch.ones(1, dtype=dtype, device=device)

    V_shift = erfc_Rc / cutoff_t
    B = erfc_Rc / cutoff_t**2 + 2.0 * alpha_t / sqrt_pi * exp_Rc / cutoff_t
    self_coeff = -(erfc_Rc / (2.0 * cutoff_t) + alpha_t / sqrt_pi)

    # --- Gather filtered charges ---
    q_i = torch.index_select(q, 0, idx_i_f)
    q_j = torch.index_select(q, 0, idx_j_f)

    # --- DSF pair potential V(r) (excluding qi*qj) ---
    if alpha > 0.0:
        erfc_r = torch.erfc(alpha_t * dist)
    else:
        erfc_r = torch.ones_like(dist)

    V_pair = erfc_r / dist - V_shift + B * (dist - cutoff_t)

    # --- Energy: 0.5 * sum_pairs qi*qj*V + sum_atoms self_coeff*qi^2 ---
    pair_energy_contrib = 0.5 * q_i * q_j * V_pair
    batch_i_f = torch.index_select(batch_idx, 0, idx_i_f)

    energy = torch.zeros(num_systems, dtype=dtype, device=device)
    if pair_energy_contrib.numel() > 0:
        energy = energy.index_add(0, batch_i_f, pair_energy_contrib)

    # Self-energy
    self_energy_per_atom = self_coeff * q**2
    energy = energy.index_add(0, batch_idx, self_energy_per_atom)

    # --- Autograd for forces, charge_grad, and virial ---
    e_total = energy.sum()
    grads = torch.autograd.grad(e_total, grad_targets, allow_unused=True)

    forces = (
        -grads[0]
        if grads[0] is not None
        else torch.zeros(N, 3, dtype=dtype, device=device)
    )
    charge_grad = (
        grads[1] if grads[1] is not None else torch.zeros(N, dtype=dtype, device=device)
    )

    if strain is not None:
        virial = (
            -grads[2]
            if grads[2] is not None
            else torch.zeros(num_systems, 3, 3, dtype=dtype, device=device)
        )
    else:
        virial = torch.zeros(num_systems, 3, 3, dtype=dtype, device=device)

    return {
        "energy": energy.detach(),
        "forces": forces.detach(),
        "virial": virial.detach(),
        "charge_grad": charge_grad.detach(),
    }


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(params=["cpu", "cuda:0"], ids=["cpu", "gpu"])
def device(request):
    """Fixture providing both CPU and GPU devices.

    GPU tests are skipped if CUDA is not available.

    Returns
    -------
    str
        Device name ("cpu" or "cuda:0")
    """
    device_name = request.param
    if device_name == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(device_name)


@pytest.fixture
def two_charge_pair(device):
    """Two opposite charges along x-axis, full NL."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
    )
    charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
    # Full NL
    neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
    neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    return positions, charges, neighbor_list, neighbor_ptr


@pytest.fixture
def pbc_system(device):
    """PBC system with cell and shifts."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
    )
    charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
    cell = torch.tensor(
        [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
        dtype=torch.float64,
        device=device,
    )
    neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
    neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    unit_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)
    return positions, charges, cell, neighbor_list, neighbor_ptr, unit_shifts


# ==============================================================================
# Test Energy
# ==============================================================================


class TestDSFEnergy:
    """Test DSF energy computation via public API against reference."""

    def test_basic_energy(self, two_charge_pair):
        """Basic energy computation returns correct shape."""
        positions, charges, nl, ptr = two_charge_pair
        result = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        assert len(result) == 1
        energy = result[0]
        assert energy.shape == (1,)
        assert energy.dtype == torch.float64

    @pytest.mark.parametrize("alpha", [0.0, 0.2, 0.5])
    def test_energy_matches_reference(self, two_charge_pair, alpha):
        """Energy matches PyTorch reference implementation."""
        positions, charges, nl, ptr = two_charge_pair
        cutoff = 10.0

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )

        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        torch.testing.assert_close(energy, ref["energy"], atol=1e-6, rtol=1e-6)

    def test_energy_only_returns_tuple(self, two_charge_pair):
        """compute_forces=False returns 1-tuple."""
        positions, charges, nl, ptr = two_charge_pair
        result = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_opposite_charges_negative_energy(self, two_charge_pair):
        """Opposite charges should give negative total energy."""
        positions, charges, nl, ptr = two_charge_pair
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        assert energy.item() < 0.0

    def test_like_charges_energy_matches_reference(self, device):
        """Like charges energy matches reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        torch.testing.assert_close(energy, ref["energy"], atol=1e-6, rtol=1e-6)

    def test_three_atom_energy_matches_reference(self, device):
        """Three-atom system energy matches reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64, device=device)
        nl = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]],
            dtype=torch.int32,
            device=device,
        )
        ptr = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=device)
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        torch.testing.assert_close(energy, ref["energy"], atol=1e-5, rtol=1e-5)


# ==============================================================================
# Test Forces
# ==============================================================================


class TestDSFForces:
    """Test DSF force computation against reference."""

    def test_forces_shape(self, two_charge_pair):
        """Forces have correct shape and match input dtype."""
        positions, charges, nl, ptr = two_charge_pair
        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        assert forces.shape == (2, 3)
        assert forces.dtype == positions.dtype

    @pytest.mark.parametrize("alpha", [0.0, 0.2, 0.5])
    def test_forces_match_reference(self, two_charge_pair, alpha):
        """Forces match PyTorch reference."""
        positions, charges, nl, ptr = two_charge_pair
        cutoff = 10.0

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        torch.testing.assert_close(forces, ref["forces"], atol=1e-6, rtol=1e-6)

    def test_opposite_charges_attract(self, two_charge_pair):
        """Opposite charges attract along bond axis."""
        positions, charges, nl, ptr = two_charge_pair
        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        # Atom 0 at origin, atom 1 at (3,0,0)
        assert forces[0, 0].item() > 0  # pulled toward atom 1
        assert forces[1, 0].item() < 0  # pulled toward atom 0

    def test_forces_optional(self, two_charge_pair):
        """compute_forces=False skips forces."""
        positions, charges, nl, ptr = two_charge_pair
        result = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        assert len(result) == 1  # only energy

    def test_three_atom_forces_match_reference(self, device):
        """Three-atom forces match reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64, device=device)
        nl = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]],
            dtype=torch.int32,
            device=device,
        )
        ptr = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=device)
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        torch.testing.assert_close(forces, ref["forces"], atol=1e-5, rtol=1e-5)


# ==============================================================================
# Test Virial
# ==============================================================================


class TestDSFVirial:
    """Test DSF virial computation against reference."""

    def test_virial_shape(self, pbc_system):
        """Virial has correct shape and matches input dtype."""
        positions, charges, cell, nl, ptr, shifts = pbc_system
        energy, forces, virial = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            cell=cell,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            unit_shifts=shifts,
            compute_virial=True,
        )
        assert virial.shape == (1, 3, 3)
        assert virial.dtype == positions.dtype

    def test_virial_matches_reference(self, pbc_system):
        """Virial matches PyTorch reference."""
        positions, charges, cell, nl, ptr, shifts = pbc_system
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            cell=cell,
            unit_shifts=shifts,
            compute_virial=True,
        )
        energy, forces, virial = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            cell=cell,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            unit_shifts=shifts,
            compute_virial=True,
        )
        torch.testing.assert_close(virial, ref["virial"], atol=1e-6, rtol=1e-6)

    def test_virial_requires_pbc(self, two_charge_pair):
        """compute_virial=True without cell raises error."""
        positions, charges, nl, ptr = two_charge_pair
        with pytest.raises(ValueError, match="periodic boundary"):
            dsf_coulomb(
                positions,
                charges,
                cutoff=10.0,
                alpha=0.2,
                neighbor_list=nl,
                neighbor_ptr=ptr,
                compute_virial=True,
            )

    def test_virial_requires_forces(self, pbc_system):
        """compute_virial=True without compute_forces raises error."""
        positions, charges, cell, nl, ptr, shifts = pbc_system
        with pytest.raises(ValueError, match="compute_forces"):
            dsf_coulomb(
                positions,
                charges,
                cutoff=10.0,
                alpha=0.2,
                cell=cell,
                neighbor_list=nl,
                neighbor_ptr=ptr,
                unit_shifts=shifts,
                compute_forces=False,
                compute_virial=True,
            )


# ==============================================================================
# Test Autograd / Charge Gradients
# ==============================================================================


class TestAutograd:
    """Test autograd support for charge gradients."""

    def test_energy_differentiable_wrt_charges(self, two_charge_pair):
        """energy.backward() populates charges.grad."""
        positions, charges, nl, ptr = two_charge_pair
        charges = charges.clone().requires_grad_(True)

        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        energy.sum().backward()

        assert charges.grad is not None
        assert charges.grad.shape == charges.shape

    def test_charge_grad_matches_reference(self, two_charge_pair):
        """Autograd charge gradient matches reference dE/dq."""
        positions, charges, nl, ptr = two_charge_pair
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )

        charges_ag = charges.clone().requires_grad_(True)
        (energy,) = dsf_coulomb(
            positions,
            charges_ag,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        energy.sum().backward()

        torch.testing.assert_close(
            charges_ag.grad, ref["charge_grad"], atol=1e-6, rtol=1e-6
        )

    def test_charge_grad_matches_finite_difference(self, device):
        """Charge gradient matches finite difference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        cutoff = 10.0
        alpha = 0.2

        # Finite difference
        delta = 1e-5
        energies_fd = []
        for dq in [-delta, delta]:
            charges = torch.tensor([1.0 + dq, -1.0], dtype=torch.float64, device=device)
            (energy,) = dsf_coulomb(
                positions,
                charges,
                cutoff=cutoff,
                alpha=alpha,
                neighbor_list=nl,
                neighbor_ptr=ptr,
                compute_forces=False,
            )
            energies_fd.append(energy.item())
        fd_grad_q0 = (energies_fd[1] - energies_fd[0]) / (2 * delta)

        # Analytical via autograd
        charges = torch.tensor(
            [1.0, -1.0], dtype=torch.float64, device=device, requires_grad=True
        )
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        energy.sum().backward()
        analytical_grad_q0 = charges.grad[0].item()

        assert abs(analytical_grad_q0 - fd_grad_q0) < 1e-4, (
            f"analytical={analytical_grad_q0}, fd={fd_grad_q0}"
        )

    def test_charge_grad_no_grad_when_not_required(self, two_charge_pair):
        """If charges don't require grad, energy is not connected to graph."""
        positions, charges, nl, ptr = two_charge_pair
        assert not charges.requires_grad
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        # Energy should not require grad
        assert not energy.requires_grad


class TestChargeGradients:
    """Test explicit charge gradient values against reference."""

    def test_single_atom_charge_grad(self, device):
        """Single atom dE/dq matches reference."""
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64, device=device)
        charges = torch.tensor(
            [2.0], dtype=torch.float64, device=device, requires_grad=True
        )
        # Empty NL
        nl = torch.tensor([[], []], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 0], dtype=torch.int32, device=device)
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges.detach(),
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )

        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        energy.sum().backward()
        torch.testing.assert_close(
            charges.grad, ref["charge_grad"], atol=1e-6, rtol=1e-6
        )

    def test_two_atom_charge_grad(self, device):
        """Two-atom charge gradient matches reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges_val = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges_val,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )

        charges = charges_val.clone().requires_grad_(True)
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        energy.sum().backward()
        torch.testing.assert_close(
            charges.grad, ref["charge_grad"], atol=1e-6, rtol=1e-6
        )


# ==============================================================================
# Test Batched Calculations
# ==============================================================================


class TestBatchedCalculations:
    """Test batched DSF calculations against reference."""

    def test_batch_energy_matches_individual(self, device):
        """Batched energy per system matches individual computation."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [50.0, 0.0, 0.0], [53.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        # Full NL for 2 separate systems
        nl = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            batch_idx=batch_idx,
            num_systems=2,
        )

        assert energy.shape == (2,)
        # Both systems are identical, so energies should match
        torch.testing.assert_close(energy[0], energy[1], atol=1e-10, rtol=0.0)

    def test_batch_matches_reference(self, device):
        """Batched results match reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [50.0, 0.0, 0.0], [53.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        nl = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        ref = dsf_reference(
            positions,
            charges,
            10.0,
            0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            batch_idx=batch_idx,
            num_systems=2,
        )
        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            batch_idx=batch_idx,
            num_systems=2,
        )
        torch.testing.assert_close(energy, ref["energy"], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(forces, ref["forces"], atol=1e-6, rtol=1e-6)


# ==============================================================================
# Test Neighbor Formats
# ==============================================================================


class TestNeighborFormats:
    """Test CSR vs matrix format consistency against reference."""

    def test_csr_matrix_energy_match(self, device):
        """CSR and matrix formats give the same energy."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cutoff = 10.0
        alpha = 0.2

        # CSR format
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        (energy_csr,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )

        # Matrix format
        neighbor_matrix = torch.tensor(
            [[1, 999], [0, 999]], dtype=torch.int32, device=device
        )
        (energy_mat,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
            compute_forces=False,
        )

        torch.testing.assert_close(energy_csr, energy_mat, atol=1e-10, rtol=0.0)

    def test_matrix_format_matches_reference(self, device):
        """Matrix format energy matches reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cutoff = 10.0
        alpha = 0.2

        neighbor_matrix = torch.tensor(
            [[1, 999], [0, 999]], dtype=torch.int32, device=device
        )
        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
        )
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
            compute_forces=False,
        )
        torch.testing.assert_close(energy, ref["energy"], atol=1e-6, rtol=1e-6)


# ==============================================================================
# Test Periodic Boundaries
# ==============================================================================


class TestPeriodicBoundaries:
    """Test PBC calculations against reference."""

    def test_pbc_energy_matches_nonpbc(self, pbc_system):
        """PBC with zero shifts matches non-PBC."""
        positions, charges, cell, nl, ptr, shifts = pbc_system
        cutoff = 10.0
        alpha = 0.2

        # Non-PBC
        (energy_np,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )

        # PBC (zero shifts)
        (energy_pbc,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            cell=cell,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            unit_shifts=shifts,
            compute_forces=False,
        )

        torch.testing.assert_close(energy_np, energy_pbc, atol=1e-10, rtol=0.0)

    def test_pbc_matches_reference(self, pbc_system):
        """PBC energy matches reference."""
        positions, charges, cell, nl, ptr, shifts = pbc_system
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            cell=cell,
            unit_shifts=shifts,
        )
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            cell=cell,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            unit_shifts=shifts,
            compute_forces=False,
        )
        torch.testing.assert_close(energy, ref["energy"], atol=1e-6, rtol=1e-6)

    def test_pbc_matrix_format(self, device):
        """PBC with matrix format works correctly."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_matrix = torch.tensor(
            [[1, 999], [0, 999]], dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (2, 2, 3), dtype=torch.int32, device=device
        )

        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            cell=cell,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=999,
        )
        assert energy.shape == (1,)
        assert forces.shape == (2, 3)


# ==============================================================================
# Test Input Validation
# ==============================================================================


class TestInputValidation:
    """Test input validation."""

    def test_no_neighbors_raises(self, device):
        """No neighbor list or matrix raises ValueError."""
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64, device=device)
        charges = torch.tensor([1.0], dtype=torch.float64, device=device)
        with pytest.raises(ValueError, match="neighbor_list.*neighbor_matrix"):
            dsf_coulomb(positions, charges, cutoff=10.0, alpha=0.2)

    def test_neighbor_list_without_ptr_raises(self, device):
        """Neighbor list without ptr raises ValueError."""
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64, device=device)
        charges = torch.tensor([1.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0], [0]], dtype=torch.int32, device=device)
        with pytest.raises(ValueError, match="neighbor_ptr"):
            dsf_coulomb(positions, charges, cutoff=10.0, alpha=0.2, neighbor_list=nl)

    def test_both_neighbor_formats_raises(self, device):
        """Providing both neighbor list and matrix raises ValueError."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        nm = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
        with pytest.raises(ValueError, match="Cannot provide both"):
            dsf_coulomb(
                positions,
                charges,
                cutoff=10.0,
                alpha=0.2,
                neighbor_list=nl,
                neighbor_ptr=ptr,
                neighbor_matrix=nm,
            )

    def test_csr_pbc_missing_unit_shifts_raises(self, device):
        """CSR format with cell but missing unit_shifts raises ValueError."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 20.0
        with pytest.raises(ValueError, match="unit_shifts is required"):
            dsf_coulomb(
                positions,
                charges,
                cutoff=10.0,
                alpha=0.2,
                neighbor_list=nl,
                neighbor_ptr=ptr,
                cell=cell,
            )

    def test_matrix_pbc_missing_shifts_raises(self, device):
        """Matrix format with cell but missing neighbor_matrix_shifts raises ValueError."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        nm = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 20.0
        with pytest.raises(ValueError, match="neighbor_matrix_shifts is required"):
            dsf_coulomb(
                positions,
                charges,
                cutoff=10.0,
                alpha=0.2,
                neighbor_matrix=nm,
                cell=cell,
            )

    def test_virial_with_csr_missing_shifts_raises(self, device):
        """compute_virial=True with CSR and missing unit_shifts raises ValueError."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 20.0
        with pytest.raises(ValueError, match="unit_shifts is required"):
            dsf_coulomb(
                positions,
                charges,
                cutoff=10.0,
                alpha=0.2,
                neighbor_list=nl,
                neighbor_ptr=ptr,
                cell=cell,
                compute_virial=True,
            )


# ==============================================================================
# Test Empty Inputs
# ==============================================================================


class TestEmptyInputs:
    """Test handling of empty inputs."""

    def test_zero_atoms(self, device):
        """Zero atoms returns empty tensors."""
        positions = torch.zeros((0, 3), dtype=torch.float64, device=device)
        charges = torch.zeros(0, dtype=torch.float64, device=device)
        nl = torch.zeros((2, 0), dtype=torch.int32, device=device)
        ptr = torch.zeros(1, dtype=torch.int32, device=device)

        result = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        assert len(result) == 1
        energy = result[0]
        assert energy.shape == (1,)
        assert energy.item() == 0.0


# ==============================================================================
# Test Alpha=0
# ==============================================================================


class TestAlphaZero:
    """Test alpha=0 (shifted-force bare Coulomb) against reference."""

    def test_alpha_zero_energy_matches_reference(self, device):
        """Alpha=0 energy matches reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        cutoff = 10.0

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            0.0,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=0.0,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        torch.testing.assert_close(energy, ref["energy"], atol=1e-6, rtol=1e-6)

    def test_alpha_zero_forces_match_reference(self, device):
        """Alpha=0 forces match reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        cutoff = 10.0

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            0.0,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=0.0,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        torch.testing.assert_close(forces, ref["forces"], atol=1e-6, rtol=1e-6)

    def test_alpha_zero_opposite_charges_attract(self, device):
        """Alpha=0: opposite charges attract."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        # Opposite charges attract
        assert forces[0, 0].item() > 0
        assert forces[1, 0].item() < 0


# ==============================================================================
# Test Device Handling
# ==============================================================================


class TestDeviceHandling:
    """Test device handling."""

    def test_output_on_same_device(self, device):
        """Outputs are on the same device as inputs."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        assert energy.device == positions.device
        assert forces.device == positions.device


# ==============================================================================
# Test Dtype Support
# ==============================================================================


class TestDtypeSupport:
    """Test dtype handling."""

    def test_float32_inputs(self, device):
        """Float32 positions and charges produce float32 forces, float64 energy."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        assert energy.dtype == torch.float64
        assert forces.dtype == torch.float32

    def test_float64_inputs(self, device):
        """Float64 positions and charges produce float64 forces and energy."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        assert energy.dtype == torch.float64
        assert forces.dtype == torch.float64

    def test_mismatched_dtypes_raises(self, device):
        """Mismatched positions and charges dtypes raises ValueError."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

        with pytest.raises(ValueError, match="charges dtype"):
            dsf_coulomb(
                positions,
                charges,
                cutoff=10.0,
                alpha=0.2,
                neighbor_list=nl,
                neighbor_ptr=ptr,
            )

    def test_float32_energy_matches_reference(self, device):
        """Float32 energy matches float32 reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        # Energy: kernel accumulates in f64, ref computes in f32 then detaches
        torch.testing.assert_close(
            energy,
            ref["energy"].to(torch.float64),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_float32_forces_match_reference(self, device):
        """Float32 forces match float32 reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        torch.testing.assert_close(forces, ref["forces"], atol=1e-4, rtol=1e-4)

    def test_float32_virial_matches_reference(self, device):
        """Float32 virial matches float32 reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float32,
            device=device,
        )
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            cell=cell,
            unit_shifts=shifts,
            compute_virial=True,
        )
        energy, forces, virial = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            cell=cell,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            unit_shifts=shifts,
            compute_virial=True,
        )
        assert virial.dtype == torch.float32
        torch.testing.assert_close(virial, ref["virial"], atol=1e-4, rtol=1e-4)

    def test_float32_charge_grad_matches_reference(self, device):
        """Float32 charge gradient matches float32 reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        charges_val = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges_val,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        charges = charges_val.clone().requires_grad_(True)
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        energy.sum().backward()
        torch.testing.assert_close(
            charges.grad,
            ref["charge_grad"],
            atol=1e-4,
            rtol=1e-4,
        )


# ==============================================================================
# Regression Tests
# ==============================================================================


class TestRegression:
    """Regression tests comparing against reference."""

    def test_regression_opposite_charges(self, device):
        """Regression: two opposite charges, known configuration."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        torch.testing.assert_close(energy, ref["energy"], atol=1e-6, rtol=1e-6)

    def test_regression_three_atoms(self, device):
        """Regression: three atoms with known DSF energy."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64, device=device)
        # Full NL: 6 edges
        nl = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]],
            dtype=torch.int32,
            device=device,
        )
        ptr = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=device)
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        torch.testing.assert_close(energy, ref["energy"], atol=1e-5, rtol=1e-5)


# ==============================================================================
# Test Numerical Stability
# ==============================================================================


class TestNumericalStability:
    """Test numerical stability edge cases."""

    def test_large_charges(self, device):
        """Large charges don't cause overflow."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([100.0, -100.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()

    def test_near_cutoff_distance(self, device):
        """Distance very close to cutoff doesn't cause issues."""
        cutoff = 10.0
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [cutoff - 1e-6, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        # Energy and forces should be small near cutoff
        assert abs(forces[0, 0].item()) < 0.1

    def test_large_charges_match_reference(self, device):
        """Large charges energy matches reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([100.0, -100.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        torch.testing.assert_close(energy, ref["energy"], atol=1e-3, rtol=1e-6)
        torch.testing.assert_close(forces, ref["forces"], atol=1e-3, rtol=1e-6)


# ==============================================================================
# Test Matrix Format Forces
# ==============================================================================


class TestMatrixForces:
    """Test matrix format force computation against reference."""

    def test_matrix_forces_match_reference(self, device):
        """Matrix format forces should match reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cutoff = 10.0
        alpha = 0.2

        neighbor_matrix = torch.tensor(
            [[1, 999], [0, 999]], dtype=torch.int32, device=device
        )
        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
        )
        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
        )
        torch.testing.assert_close(forces, ref["forces"], atol=1e-6, rtol=1e-6)

    def test_matrix_forces_match_csr_forces(self, device):
        """Matrix and CSR format forces should match."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cutoff = 10.0
        alpha = 0.2

        # CSR
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        energy_csr, forces_csr = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )

        # Matrix
        neighbor_matrix = torch.tensor(
            [[1, 999], [0, 999]], dtype=torch.int32, device=device
        )
        energy_mat, forces_mat = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
        )

        torch.testing.assert_close(forces_csr, forces_mat, atol=1e-10, rtol=0.0)


# ==============================================================================
# Test Force Conservation
# ==============================================================================


class TestForceConservation:
    """Test that total force on isolated systems is zero."""

    def test_two_atom_force_sum_zero(self, device):
        """Sum of forces should be zero for two-atom system."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        torch.testing.assert_close(
            forces.sum(dim=0),
            torch.zeros(3, dtype=torch.float64, device=device),
            atol=1e-10,
            rtol=0.0,
        )

    def test_three_atom_force_sum_zero(self, device):
        """Sum of forces should be zero for three-atom system."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64, device=device)
        nl = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]],
            dtype=torch.int32,
            device=device,
        )
        ptr = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=device)

        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )
        torch.testing.assert_close(
            forces.sum(dim=0),
            torch.zeros(3, dtype=torch.float64, device=device),
            atol=1e-10,
            rtol=0.0,
        )


# ==============================================================================
# Test Energy Consistency Across compute_forces Flag
# ==============================================================================


class TestEnergyConsistency:
    """Test that energy is the same regardless of compute_forces flag."""

    def test_csr_energy_same_with_and_without_forces(self, device):
        """CSR: energy with compute_forces=True matches compute_forces=False."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

        (energy_no_forces,) = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=False,
        )
        energy_with_forces, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            compute_forces=True,
        )
        torch.testing.assert_close(
            energy_no_forces, energy_with_forces, atol=1e-10, rtol=0.0
        )

    def test_matrix_energy_same_with_and_without_forces(self, device):
        """Matrix: energy with compute_forces=True matches compute_forces=False."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        neighbor_matrix = torch.tensor(
            [[1, 999], [0, 999]], dtype=torch.int32, device=device
        )

        (energy_no_forces,) = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
            compute_forces=False,
        )
        energy_with_forces, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
            compute_forces=True,
        )
        torch.testing.assert_close(
            energy_no_forces, energy_with_forces, atol=1e-10, rtol=0.0
        )


# ==============================================================================
# Test PBC with Non-Zero Shifts
# ==============================================================================


class TestPBCNonZeroShifts:
    """Test PBC with non-trivial periodic image interactions."""

    def test_pbc_nonzero_shift_energy_matches_reference(self, device):
        """PBC with non-zero shifts gives correct energy."""
        # Two atoms near opposite cell edges: PBC distance = 1.0
        positions = torch.tensor(
            [[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=torch.float64,
            device=device,
        )
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        unit_shifts = torch.tensor(
            [[-1, 0, 0], [1, 0, 0]], dtype=torch.int32, device=device
        )
        cutoff = 5.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            cell=cell,
            unit_shifts=unit_shifts,
        )
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            cell=cell,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            unit_shifts=unit_shifts,
            compute_forces=False,
        )
        torch.testing.assert_close(energy, ref["energy"], atol=1e-6, rtol=1e-6)

    def test_pbc_nonzero_shift_forces_match_reference(self, device):
        """PBC with non-zero shifts gives correct forces."""
        positions = torch.tensor(
            [[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=torch.float64,
            device=device,
        )
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        unit_shifts = torch.tensor(
            [[-1, 0, 0], [1, 0, 0]], dtype=torch.int32, device=device
        )
        cutoff = 5.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            cell=cell,
            unit_shifts=unit_shifts,
        )
        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            cell=cell,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            unit_shifts=unit_shifts,
        )
        torch.testing.assert_close(forces, ref["forces"], atol=1e-6, rtol=1e-6)


# ==============================================================================
# Test Batched Matrix Format
# ==============================================================================


class TestBatchedMatrixFormat:
    """Test batched calculations with matrix neighbor format."""

    def test_batch_matrix_energy_matches_individual(self, device):
        """Batched matrix energy per system matches individual computation."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [50.0, 0.0, 0.0], [53.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        # Matrix: each atom has 1 neighbor within its system
        neighbor_matrix = torch.tensor(
            [[1, 999], [0, 999], [3, 999], [2, 999]],
            dtype=torch.int32,
            device=device,
        )
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        cutoff = 10.0
        alpha = 0.2

        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
            batch_idx=batch_idx,
            num_systems=2,
            compute_forces=False,
        )
        assert energy.shape == (2,)
        # Both systems are identical, so energies should match
        torch.testing.assert_close(energy[0], energy[1], atol=1e-10, rtol=0.0)


# ==============================================================================
# Test torch.compile Compatibility
# ==============================================================================


class TestTorchCompile:
    """Smoke tests for torch.compile compatibility."""

    def test_compile_csr_energy_forces(self, device):
        """torch.compile should produce same results as eager mode (CSR)."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        cutoff = 10.0
        alpha = 0.2

        # Eager mode
        energy_eager, forces_eager = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )

        # Compiled mode
        dsf_compiled = torch.compile(dsf_coulomb)
        energy_compiled, forces_compiled = dsf_compiled(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
        )

        torch.testing.assert_close(energy_compiled, energy_eager, atol=1e-10, rtol=0.0)
        torch.testing.assert_close(forces_compiled, forces_eager, atol=1e-10, rtol=0.0)


# ==============================================================================
# Test Matrix PBC Virial
# ==============================================================================


class TestMatrixPBCVirial:
    """Test virial computation with matrix neighbor format and PBC."""

    def test_matrix_pbc_virial_matches_csr(self, device):
        """Matrix PBC virial should match CSR PBC virial."""
        positions = torch.tensor(
            [[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=torch.float64,
            device=device,
        )
        cutoff = 5.0
        alpha = 0.2

        # CSR format
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        unit_shifts = torch.tensor(
            [[-1, 0, 0], [1, 0, 0]], dtype=torch.int32, device=device
        )

        energy_csr, forces_csr, virial_csr = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            cell=cell,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            unit_shifts=unit_shifts,
            compute_virial=True,
        )

        # Matrix format
        neighbor_matrix = torch.tensor(
            [[1, 2], [0, 2]], dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.tensor(
            [[[-1, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 0, 0]]],
            dtype=torch.int32,
            device=device,
        )

        energy_mat, forces_mat, virial_mat = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            cell=cell,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=2,
            compute_virial=True,
        )

        torch.testing.assert_close(energy_mat, energy_csr, atol=1e-10, rtol=0.0)
        torch.testing.assert_close(forces_mat, forces_csr, atol=1e-10, rtol=0.0)
        torch.testing.assert_close(virial_mat, virial_csr, atol=1e-10, rtol=0.0)


# ==============================================================================
# Test Batch + PBC
# ==============================================================================


class TestBatchPBC:
    """Test batched calculations with periodic boundary conditions."""

    def test_batch_pbc_energy_matches_individual(self, device):
        """Batched PBC energy per system matches individual computation."""
        pos_single = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges_single = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell_single = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        nl_single = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr_single = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        shifts_single = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energy_single, forces_single = dsf_coulomb(
            pos_single,
            charges_single,
            cutoff=10.0,
            alpha=0.2,
            cell=cell_single,
            neighbor_list=nl_single,
            neighbor_ptr=ptr_single,
            unit_shifts=shifts_single,
        )

        positions_batch = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [50.0, 0.0, 0.0], [53.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges_batch = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell_batch = torch.tensor(
            [
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            ],
            dtype=torch.float64,
            device=device,
        )
        nl_batch = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        ptr_batch = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        shifts_batch = torch.zeros((4, 3), dtype=torch.int32, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        energy_batch, forces_batch = dsf_coulomb(
            positions_batch,
            charges_batch,
            cutoff=10.0,
            alpha=0.2,
            cell=cell_batch,
            neighbor_list=nl_batch,
            neighbor_ptr=ptr_batch,
            unit_shifts=shifts_batch,
            batch_idx=batch_idx,
            num_systems=2,
        )

        assert energy_batch.shape == (2,)
        torch.testing.assert_close(
            energy_batch[0], energy_single[0], atol=1e-10, rtol=0.0
        )
        torch.testing.assert_close(
            energy_batch[1], energy_single[0], atol=1e-10, rtol=0.0
        )

    def test_batch_pbc_matches_reference(self, device):
        """Batched PBC results match reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [50.0, 0.0, 0.0], [53.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = torch.tensor(
            [
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            ],
            dtype=torch.float64,
            device=device,
        )
        nl = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        shifts = torch.zeros((4, 3), dtype=torch.int32, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        ref = dsf_reference(
            positions,
            charges,
            10.0,
            0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            cell=cell,
            unit_shifts=shifts,
            batch_idx=batch_idx,
            num_systems=2,
        )
        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            cell=cell,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            unit_shifts=shifts,
            batch_idx=batch_idx,
            num_systems=2,
        )
        torch.testing.assert_close(energy, ref["energy"], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(forces, ref["forces"], atol=1e-6, rtol=1e-6)


# ==============================================================================
# Test Matrix Autograd
# ==============================================================================


class TestMatrixAutograd:
    """Test charge gradient autograd support with matrix neighbor format."""

    def test_matrix_charge_grad_matches_reference(self, device):
        """Matrix format charge gradient matches reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges_val = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        neighbor_matrix = torch.tensor(
            [[1, 999], [0, 999]], dtype=torch.int32, device=device
        )
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges_val,
            cutoff,
            alpha,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
        )

        charges = charges_val.clone().requires_grad_(True)
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
            compute_forces=False,
        )
        energy.sum().backward()
        torch.testing.assert_close(
            charges.grad, ref["charge_grad"], atol=1e-6, rtol=1e-6
        )

    def test_matrix_charge_grad_finite_difference(self, device):
        """Matrix format charge gradient matches finite difference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_matrix = torch.tensor(
            [[1, 2, 999], [0, 2, 999], [0, 1, 999]], dtype=torch.int32, device=device
        )
        cutoff = 10.0
        alpha = 0.2
        delta = 1e-5

        charges_base = torch.tensor(
            [1.0, -0.5, 0.8], dtype=torch.float64, device=device
        )
        fd_grad = torch.zeros(3, dtype=torch.float64, device=device)
        for atom in range(3):
            for sign, coeff in [(-1.0, -1.0), (1.0, 1.0)]:
                q = charges_base.clone()
                q[atom] += sign * delta
                (e,) = dsf_coulomb(
                    positions,
                    q,
                    cutoff=cutoff,
                    alpha=alpha,
                    neighbor_matrix=neighbor_matrix,
                    fill_value=999,
                    compute_forces=False,
                )
                fd_grad[atom] += coeff * e.item()
            fd_grad[atom] /= 2 * delta

        charges = charges_base.clone().requires_grad_(True)
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
            compute_forces=False,
        )
        energy.sum().backward()

        torch.testing.assert_close(charges.grad, fd_grad, atol=1e-4, rtol=1e-4)


# ==============================================================================
# Test Batch Autograd
# ==============================================================================


class TestBatchAutograd:
    """Test charge gradient autograd support in batched mode."""

    def test_batch_charge_grad_matches_reference(self, device):
        """Batched charge gradient matches reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [50.0, 0.0, 0.0], [53.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges_val = torch.tensor(
            [1.0, -1.0, 0.5, -0.5], dtype=torch.float64, device=device
        )
        nl = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        ref = dsf_reference(
            positions,
            charges_val,
            10.0,
            0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            batch_idx=batch_idx,
            num_systems=2,
        )

        charges = charges_val.clone().requires_grad_(True)
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            batch_idx=batch_idx,
            num_systems=2,
            compute_forces=False,
        )
        energy.sum().backward()
        torch.testing.assert_close(
            charges.grad, ref["charge_grad"], atol=1e-6, rtol=1e-6
        )


# ==============================================================================
# Test Matrix Float32
# ==============================================================================


class TestMatrixFloat32:
    """Test matrix format with float32 precision."""

    def test_matrix_float32_energy_matches_reference(self, device):
        """Matrix float32 energy matches reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        neighbor_matrix = torch.tensor(
            [[1, 999], [0, 999]], dtype=torch.int32, device=device
        )
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
        )
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
            compute_forces=False,
        )
        torch.testing.assert_close(
            energy,
            ref["energy"].to(torch.float64),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_matrix_float32_forces_match_reference(self, device):
        """Matrix float32 forces match reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        neighbor_matrix = torch.tensor(
            [[1, 999], [0, 999]], dtype=torch.int32, device=device
        )
        cutoff = 10.0
        alpha = 0.2

        ref = dsf_reference(
            positions,
            charges,
            cutoff,
            alpha,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
        )
        energy, forces = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
        )
        assert forces.dtype == torch.float32
        torch.testing.assert_close(forces, ref["forces"], atol=1e-4, rtol=1e-4)


# ==============================================================================
# Test PBC Autograd
# ==============================================================================


class TestPBCAutograd:
    """Test charge gradient autograd support with PBC."""

    def test_pbc_charge_grad_matches_reference(self, device):
        """PBC charge gradient matches reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges_val = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        ref = dsf_reference(
            positions,
            charges_val,
            10.0,
            0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            cell=cell,
            unit_shifts=shifts,
        )

        charges = charges_val.clone().requires_grad_(True)
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            cell=cell,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            unit_shifts=shifts,
            compute_forces=False,
        )
        energy.sum().backward()
        torch.testing.assert_close(
            charges.grad, ref["charge_grad"], atol=1e-6, rtol=1e-6
        )

    def test_pbc_nonzero_shift_charge_grad_matches_reference(self, device):
        """PBC with non-zero shifts: charge gradient matches reference."""
        positions = torch.tensor(
            [[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges_val = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=torch.float64,
            device=device,
        )
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        unit_shifts = torch.tensor(
            [[-1, 0, 0], [1, 0, 0]], dtype=torch.int32, device=device
        )

        ref = dsf_reference(
            positions,
            charges_val,
            5.0,
            0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            cell=cell,
            unit_shifts=unit_shifts,
        )

        charges = charges_val.clone().requires_grad_(True)
        (energy,) = dsf_coulomb(
            positions,
            charges,
            cutoff=5.0,
            alpha=0.2,
            cell=cell,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            unit_shifts=unit_shifts,
            compute_forces=False,
        )
        energy.sum().backward()
        torch.testing.assert_close(
            charges.grad, ref["charge_grad"], atol=1e-6, rtol=1e-6
        )


# ==============================================================================
# Test Matrix torch.compile
# ==============================================================================


class TestMatrixTorchCompile:
    """Smoke tests for torch.compile compatibility with matrix format."""

    def test_compile_matrix_energy_forces(self, device):
        """torch.compile should produce same results as eager mode (matrix)."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        neighbor_matrix = torch.tensor(
            [[1, 999], [0, 999]], dtype=torch.int32, device=device
        )
        cutoff = 10.0
        alpha = 0.2

        energy_eager, forces_eager = dsf_coulomb(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
        )

        dsf_compiled = torch.compile(dsf_coulomb)
        energy_compiled, forces_compiled = dsf_compiled(
            positions,
            charges,
            cutoff=cutoff,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            fill_value=999,
        )

        torch.testing.assert_close(energy_compiled, energy_eager, atol=1e-10, rtol=0.0)
        torch.testing.assert_close(forces_compiled, forces_eager, atol=1e-10, rtol=0.0)


# ==============================================================================
# Test Batch Virial
# ==============================================================================


class TestBatchVirial:
    """Test virial computation in batched mode with PBC."""

    def test_batch_virial_matches_individual(self, device):
        """Batched virial per system matches individual computation."""
        pos_single = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        charges_single = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell_single = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        nl_single = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr_single = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        shifts_single = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energy_s, forces_s, virial_s = dsf_coulomb(
            pos_single,
            charges_single,
            cutoff=10.0,
            alpha=0.2,
            cell=cell_single,
            neighbor_list=nl_single,
            neighbor_ptr=ptr_single,
            unit_shifts=shifts_single,
            compute_virial=True,
        )

        positions_batch = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [50.0, 0.0, 0.0], [53.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges_batch = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell_batch = torch.tensor(
            [
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            ],
            dtype=torch.float64,
            device=device,
        )
        nl_batch = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        ptr_batch = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        shifts_batch = torch.zeros((4, 3), dtype=torch.int32, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        energy_b, forces_b, virial_b = dsf_coulomb(
            positions_batch,
            charges_batch,
            cutoff=10.0,
            alpha=0.2,
            cell=cell_batch,
            neighbor_list=nl_batch,
            neighbor_ptr=ptr_batch,
            unit_shifts=shifts_batch,
            batch_idx=batch_idx,
            num_systems=2,
            compute_virial=True,
        )

        assert virial_b.shape == (2, 3, 3)
        torch.testing.assert_close(virial_b[0], virial_s[0], atol=1e-10, rtol=0.0)
        torch.testing.assert_close(virial_b[1], virial_s[0], atol=1e-10, rtol=0.0)

    def test_batch_virial_matches_reference(self, device):
        """Batched virial matches reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [50.0, 0.0, 0.0], [53.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = torch.tensor(
            [
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            ],
            dtype=torch.float64,
            device=device,
        )
        nl = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        shifts = torch.zeros((4, 3), dtype=torch.int32, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        ref = dsf_reference(
            positions,
            charges,
            10.0,
            0.2,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            cell=cell,
            unit_shifts=shifts,
            batch_idx=batch_idx,
            num_systems=2,
            compute_virial=True,
        )
        energy, forces, virial = dsf_coulomb(
            positions,
            charges,
            cutoff=10.0,
            alpha=0.2,
            cell=cell,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            unit_shifts=shifts,
            batch_idx=batch_idx,
            num_systems=2,
            compute_virial=True,
        )
        torch.testing.assert_close(virial, ref["virial"], atol=1e-6, rtol=1e-6)

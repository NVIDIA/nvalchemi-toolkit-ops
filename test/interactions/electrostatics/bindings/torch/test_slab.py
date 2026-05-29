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
Integration tests for slab correction (Yeh-Berkowitz / Ballenegger Eq. 29).

Coverage:
- LAMMPS reference energy for CsCl slab
- Cross-validation against torch-pme (energies + forces)
- Standalone slab custom-op gradcheck
- Full PME slab energy gradcheck
- 3D periodic = zero correction
- Triclinic projected-normal geometry
- Standalone compute_slab_correction() API edge cases
- Hybrid-force slab semantics
"""

from __future__ import annotations

import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    compute_slab_correction,
    ewald_real_space,
    particle_mesh_ewald,
    pme_reciprocal_space,
)
from nvalchemiops.torch.interactions.electrostatics.ewald import ewald_summation
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.torch.neighbors import batch_cell_list, cell_list

try:
    from torchpme import EwaldCalculator, PMECalculator
    from torchpme.potentials import CoulombPotential

    HAS_TORCHPME = True
except ModuleNotFoundError:
    HAS_TORCHPME = False
    EwaldCalculator = None
    PMECalculator = None
    CoulombPotential = None

KCALMOL_PER_ANGSTROM = 332.0637132991921
EWALD_ALPHA = 0.3
EWALD_K_CUTOFF = 8.0
TRICLINIC_EWALD_K_CUTOFF = 7.0
REAL_SPACE_CUTOFF = 5.0
SLAB_STRICT_RTOL = 1e-12
SLAB_STRICT_ATOL = 1e-14
SLAB_CHARGE_GRAD_RTOL = 5e-8
SLAB_CHARGE_GRAD_ATOL = 1e-9
PME_SLAB_GRADCHECK_RTOL = 1e-6
PME_SLAB_GRADCHECK_ATOL = 1e-9


# ==============================================================================
# Helpers
# ==============================================================================


def _make_cscl_slab_system(dtype=torch.float64, device="cpu"):
    """Create CsCl slab system matching torch-pme's LAMMPS test.

    2 atoms, cell=[10, 10, 30], pbc=[True, True, False] (slab in z).
    """
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=dtype, device=device
    )
    charges = torch.tensor([1.0, -1.0], dtype=dtype, device=device)
    cell = torch.diag(
        torch.tensor([10.0, 10.0, 30.0], dtype=dtype, device=device)
    ).unsqueeze(0)  # (1, 3, 3)
    pbc = torch.tensor([True, True, False], device=device)  # (3,)
    return positions, charges, cell, pbc


def _make_triclinic_slab_system(dtype=torch.float64, device="cpu"):
    """Create a small non-neutral triclinic T/T/F slab system."""
    positions = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 1.5, 6.0], [2.0, 3.5, 7.5]],
        dtype=dtype,
        device=device,
    )
    charges = torch.tensor([1.0, -0.5, 0.3], dtype=dtype, device=device)
    cell = torch.tensor(
        [[9.0, 0.0, 0.0], [2.0, 8.0, 1.5], [0.5, 0.2, 25.0]],
        dtype=dtype,
        device=device,
    ).unsqueeze(0)
    pbc = torch.tensor([True, True, False], device=device)
    return positions, charges, cell, pbc


def _make_cscl_ewald_inputs(dtype=torch.float64, device="cpu"):
    """Create CsCl slab inputs plus the full-3D real-space neighbor list."""
    positions, charges, cell, pbc = _make_cscl_slab_system(dtype, device)
    neighbor_list, neighbor_ptr, neighbor_shifts = _build_neighbor_list(
        positions, cell, REAL_SPACE_CUTOFF, [True, True, True], device
    )
    return positions, charges, cell, pbc, neighbor_list, neighbor_ptr, neighbor_shifts


def _make_triclinic_ewald_inputs(dtype=torch.float64, device="cpu"):
    """Create triclinic slab inputs plus the full-3D real-space neighbor list."""
    positions, charges, cell, pbc = _make_triclinic_slab_system(dtype, device)
    neighbor_list, neighbor_ptr, neighbor_shifts = _build_neighbor_list(
        positions, cell, REAL_SPACE_CUTOFF, [True, True, True], device
    )
    return positions, charges, cell, pbc, neighbor_list, neighbor_ptr, neighbor_shifts


def _build_neighbor_list(positions, cell, cutoff, pbc_full3d, device="cpu"):
    """Build neighbor list using cell_list with full 3D pbc.

    The slab correction handles the 2D periodicity separately; for the
    real-space neighbor list we use full 3D periodicity (the vacuum gap
    in z guarantees no real-space neighbors leak across).
    """
    pbc_tensor = torch.tensor(pbc_full3d, dtype=torch.bool, device=device).unsqueeze(0)
    neighbor_list, neighbor_ptr, unit_shifts = cell_list(
        positions, cutoff, cell, pbc_tensor, return_neighbor_list=True
    )
    return neighbor_list, neighbor_ptr, unit_shifts


def _run_torchpme_ewald(positions, charges, cell_2d, pbc, alpha, k_cutoff):
    """Run torch-pme EwaldCalculator for cross-validation.

    Parameters
    ----------
    positions, charges : torch.Tensor
    cell_2d : torch.Tensor, shape (3, 3)
        Cell matrix without a batch dimension.
    pbc : torch.Tensor, shape (3,) bool
    alpha : float
        Ewald splitting parameter (toolkit-ops convention).
    k_cutoff : float
        K-vector cutoff.

    Returns
    -------
    potential : torch.Tensor, shape (N, 1)
        Per-atom potential from torch-pme.
    """
    import math

    dtype = positions.dtype
    device = positions.device

    # Convert alpha (toolkit-ops convention) to torch-pme smearing
    smearing = 1.0 / (math.sqrt(2.0) * alpha)
    lr_wavelength = 2.0 * math.pi / k_cutoff

    potential = CoulombPotential(smearing=smearing)
    calculator = EwaldCalculator(
        potential=potential,
        lr_wavelength=lr_wavelength,
        full_neighbor_list=True,
    ).to(device=device, dtype=dtype)

    charges_2d = charges.unsqueeze(-1)

    # Full pairwise neighbor list for this small system
    N = len(positions)
    i_indices = []
    j_indices = []
    for i in range(N):
        for j in range(N):
            if i != j:
                i_indices.append(i)
                j_indices.append(j)
    neighbor_indices = torch.tensor(
        [i_indices, j_indices], dtype=torch.int64, device=device
    ).T
    diff = positions[neighbor_indices[:, 1]] - positions[neighbor_indices[:, 0]]
    neighbor_distances = torch.norm(diff, dim=1)

    return calculator.forward(
        charges=charges_2d,
        cell=cell_2d,
        positions=positions,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        periodic=pbc,
    )


def _run_torchpme_pme(positions, charges, cell_2d, pbc, alpha, mesh_spacing):
    """Run torch-pme PMECalculator for full PME slab cross-validation."""
    import math

    dtype = positions.dtype
    device = positions.device

    smearing = 1.0 / (math.sqrt(2.0) * alpha)
    potential = CoulombPotential(smearing=smearing)
    calculator = PMECalculator(
        potential=potential,
        mesh_spacing=mesh_spacing,
        interpolation_nodes=4,
        full_neighbor_list=True,
    ).to(device=device, dtype=dtype)

    charges_2d = charges.unsqueeze(-1)
    num_atoms = len(positions)
    i_indices = []
    j_indices = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                i_indices.append(i)
                j_indices.append(j)
    neighbor_indices = torch.tensor(
        [i_indices, j_indices], dtype=torch.int64, device=device
    ).T
    diff = positions[neighbor_indices[:, 1]] - positions[neighbor_indices[:, 0]]
    neighbor_distances = torch.norm(diff, dim=1)

    return calculator.forward(
        charges=charges_2d,
        cell=cell_2d,
        positions=positions,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        periodic=pbc,
    )


def _reference_slab_correction(positions, charges, cell, pbc):
    """Independent Torch reference for projected-normal slab correction."""
    cell_2d = cell.squeeze(0) if cell.dim() == 3 else cell
    pbc_1d = pbc.squeeze(0) if pbc.dim() == 2 else pbc
    nonperiodic_axis = int(torch.nonzero(~pbc_1d, as_tuple=False).flatten()[0])

    periodic_a = cell_2d[(nonperiodic_axis + 1) % 3]
    periodic_b = cell_2d[(nonperiodic_axis + 2) % 3]
    normal = torch.cross(periodic_a, periodic_b, dim=0)
    normal = normal / torch.linalg.norm(normal)

    z = positions @ normal
    volume = torch.abs(torch.linalg.det(cell_2d))
    height_sq = torch.dot(cell_2d[nonperiodic_axis], normal) ** 2
    qtotal = charges.sum()
    moment = torch.sum(charges * z)
    moment2 = torch.sum(charges * z * z)

    bracket = z * moment - 0.5 * (moment2 + qtotal * z * z) - qtotal * height_sq / 12.0
    energies = (2.0 * torch.pi / volume) * charges * bracket
    forces = (-(4.0 * torch.pi / volume) * charges * (moment - qtotal * z)).unsqueeze(
        -1
    ) * normal
    charge_grads = (4.0 * torch.pi / volume) * bracket
    projector = torch.eye(
        3, dtype=positions.dtype, device=positions.device
    ) - 2.0 * torch.outer(normal, normal)
    virial = energies.sum() * projector
    return energies, forces, charge_grads, virial


# ==============================================================================
# LAMMPS reference energy
# ==============================================================================


class TestLAMMPSReference:
    """CsCl slab energy should match LAMMPS value of -383.44635 kcal/mol/A."""

    def test_lammps_cscl_slab_energy(self, device):
        dtype = torch.float64
        lammps_energy = -383.44635  # kcal/mol/A

        positions, charges, cell, pbc, nl, ptr, shifts = _make_cscl_ewald_inputs(
            dtype, device
        )

        energies = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_cutoff=EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            pbc=pbc,
            slab_correction=True,
        )

        total_energy_kcal = energies.sum() * KCALMOL_PER_ANGSTROM
        # Total slab-corrected Ewald energy matches the LAMMPS reference.
        torch.testing.assert_close(
            total_energy_kcal,
            torch.tensor(lammps_energy, dtype=dtype, device=device),
            rtol=1e-3,
            atol=0.0,
        )


# ==============================================================================
# Cross-validation against torch-pme
# ==============================================================================


class TestTorchPMECrossValidation:
    """Cross-validate slab correction against torch-pme EwaldCalculator."""

    @pytest.mark.skipif(not HAS_TORCHPME, reason="torch-pme not installed")
    def test_outputs_match_torchpme(self, device):
        """Energy and forces match torch-pme for the CsCl slab."""
        dtype = torch.float64

        positions, charges, cell, pbc, nl, ptr, shifts = _make_cscl_ewald_inputs(
            dtype, device
        )

        our_energies, our_forces = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_cutoff=EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            pbc=pbc,
            slab_correction=True,
        )

        positions_tp = positions.clone().detach().requires_grad_(True)
        torchpme_potential = _run_torchpme_ewald(
            positions_tp,
            charges,
            cell.squeeze(0),
            pbc,
            EWALD_ALPHA,
            EWALD_K_CUTOFF,
        )
        torchpme_total = (torchpme_potential.squeeze(-1) * charges).sum()
        torchpme_forces = -torch.autograd.grad(
            torchpme_total, positions_tp, create_graph=False
        )[0]

        # Total slab-corrected Ewald energy matches torch-pme.
        torch.testing.assert_close(
            our_energies.sum(), torchpme_total, rtol=1e-5, atol=0.0
        )
        # Total slab-corrected Ewald forces match torch-pme autograd forces.
        torch.testing.assert_close(our_forces, torchpme_forces, rtol=1e-4, atol=1e-8)


# ==============================================================================
# Analytical kernel outputs vs autograd
# ==============================================================================


class TestAnalyticalVsAutograd:
    """Analytical kernel outputs should match autograd derivatives."""

    def test_slab_gradcheck_all_outputs(self, device):
        """Slab energy, forces, charge gradients, and virial pass gradcheck."""
        dtype = torch.float64

        positions, charges, cell, pbc = _make_triclinic_slab_system(dtype, device)
        positions = positions.clone().detach().requires_grad_(True)
        charges = charges.clone().detach().requires_grad_(True)
        cell = cell.clone().detach().requires_grad_(True)

        def slab_outputs(positions_in, charges_in, cell_in):
            return compute_slab_correction(
                positions_in,
                charges_in,
                cell_in,
                pbc,
                compute_forces=True,
                compute_charge_gradients=True,
                compute_virial=True,
            )

        assert torch.autograd.gradcheck(
            slab_outputs,
            (positions, charges, cell),
            eps=1e-6,
            atol=1e-10,
            rtol=1e-8,
            nondet_tol=1e-12,
        )


# ==============================================================================
# 3D periodic = zero correction
# ==============================================================================


class TestZeroCorrection:
    """3D periodic systems should give identical results with/without slab_correction."""

    def test_pbc_3d_matches_no_compute_slab_correction(self, device):
        dtype = torch.float64

        positions, charges, cell, _, nl, ptr, shifts = _make_cscl_ewald_inputs(
            dtype, device
        )

        kwargs = dict(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=EWALD_ALPHA,
            k_cutoff=EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
        )

        # No slab correction (default)
        energies_off, forces_off = ewald_summation(**kwargs)

        # Slab correction enabled but pbc is 3D -> no contribution
        pbc_3d = torch.tensor([True, True, True], device=device)
        energies_3d, forces_3d = ewald_summation(
            **kwargs, pbc=pbc_3d, slab_correction=True
        )

        # 3D periodic slab correction leaves per-atom energies unchanged.
        torch.testing.assert_close(energies_3d, energies_off, rtol=0, atol=0)
        # 3D periodic slab correction leaves forces unchanged.
        torch.testing.assert_close(forces_3d, forces_off, rtol=0, atol=0)


# ==============================================================================
# Triclinic cells
# ==============================================================================


class TestTriclinicCells:
    """Triclinic slab cells use projected-normal geometry."""

    def test_triclinic_standalone_matches_reference(self, device):
        dtype = torch.float64

        positions, charges, cell, pbc = _make_triclinic_slab_system(dtype, device)

        energies, forces, charge_grads, virial = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        ref_e, ref_f, ref_cg, ref_v = _reference_slab_correction(
            positions, charges, cell, pbc
        )

        # Triclinic standalone slab energies match the independent reference.
        torch.testing.assert_close(energies, ref_e, rtol=1e-12, atol=1e-15)
        # Triclinic standalone slab forces match the independent reference.
        torch.testing.assert_close(forces, ref_f, rtol=1e-12, atol=1e-15)
        # Triclinic standalone slab charge gradients match the reference.
        torch.testing.assert_close(charge_grads, ref_cg, rtol=1e-12, atol=1e-15)
        # Triclinic standalone slab virial matches the independent reference.
        torch.testing.assert_close(virial[0], ref_v, rtol=1e-12, atol=1e-15)

    def test_triclinic_full_ewald_matches_decomposition(self, device):
        dtype = torch.float64

        positions, charges, cell, pbc, nl, ptr, shifts = _make_triclinic_ewald_inputs(
            dtype, device
        )

        e_full, f_full, cg_full, v_full = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_cutoff=TRICLINIC_EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
            pbc=pbc,
            slab_correction=True,
        )
        e_3d, f_3d, cg_3d, v_3d = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_cutoff=TRICLINIC_EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        e_slab, f_slab, cg_slab, v_slab = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )

        # Full triclinic Ewald energies equal 3D Ewald plus slab correction.
        torch.testing.assert_close(e_full, e_3d + e_slab, rtol=0, atol=0)
        # Full triclinic Ewald forces equal 3D Ewald plus slab correction.
        torch.testing.assert_close(f_full, f_3d + f_slab, rtol=0, atol=0)
        # Full triclinic Ewald charge gradients equal 3D Ewald plus slab correction.
        torch.testing.assert_close(cg_full, cg_3d + cg_slab, rtol=0, atol=0)
        # Full triclinic Ewald virial equals 3D Ewald plus slab correction.
        torch.testing.assert_close(v_full, v_3d + v_slab, rtol=1e-12, atol=1e-15)

    def test_triclinic_full_ewald_matches_autograd(self, device):
        dtype = torch.float64

        positions, charges, cell, pbc, nl, ptr, shifts = _make_triclinic_ewald_inputs(
            dtype, device
        )
        positions = positions.clone().detach().requires_grad_(True)
        charges = charges.clone().detach().requires_grad_(True)

        energies, forces, charge_grads = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_cutoff=TRICLINIC_EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_charge_gradients=True,
            pbc=pbc,
            slab_correction=True,
        )
        autograd_positions, autograd_charge_grads = torch.autograd.grad(
            energies.sum(), (positions, charges), create_graph=False
        )
        autograd_forces = -autograd_positions

        # Full triclinic Ewald forces match autograd forces.
        torch.testing.assert_close(forces, autograd_forces, rtol=1e-8, atol=2e-8)
        # Full triclinic Ewald charge gradients match autograd charge gradients.
        torch.testing.assert_close(
            charge_grads,
            autograd_charge_grads,
            rtol=SLAB_CHARGE_GRAD_RTOL,
            atol=SLAB_CHARGE_GRAD_ATOL,
        )

        _, _, virial = ewald_summation(
            positions.detach(),
            charges.detach(),
            cell,
            alpha=EWALD_ALPHA,
            k_cutoff=TRICLINIC_EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_virial=True,
            pbc=pbc,
            slab_correction=True,
        )
        eps = torch.zeros(3, 3, dtype=dtype, device=device, requires_grad=True)
        deformation = torch.eye(3, dtype=dtype, device=device) + eps
        e_strained = ewald_summation(
            positions.detach() @ deformation.T,
            charges.detach(),
            cell @ deformation.T,
            alpha=EWALD_ALPHA,
            k_cutoff=TRICLINIC_EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            pbc=pbc,
            slab_correction=True,
        ).sum()
        autograd_virial = -torch.autograd.grad(e_strained, eps)[0]

        # Full triclinic Ewald virial matches strain autograd.
        torch.testing.assert_close(virial[0], autograd_virial, rtol=1e-8, atol=1e-7)

    def test_triclinic_translation_invariance_non_neutral(self, device):
        dtype = torch.float64

        positions, charges, cell, pbc = _make_triclinic_slab_system(dtype, device)
        shift = torch.tensor([1.3, -0.7, 2.1], dtype=dtype, device=device)

        e0, f0, cg0 = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
        )
        e1, f1, cg1 = compute_slab_correction(
            positions + shift,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
        )

        # Non-neutral triclinic slab total energy is translation invariant.
        torch.testing.assert_close(e1.sum(), e0.sum(), rtol=1e-12, atol=1e-15)
        # Non-neutral triclinic slab forces are translation invariant.
        torch.testing.assert_close(f1, f0, rtol=1e-12, atol=1e-15)
        # Non-neutral triclinic slab charge gradients are translation invariant.
        torch.testing.assert_close(cg1, cg0, rtol=1e-12, atol=1e-15)


# ==============================================================================
# Ewald slab dtype behavior
# ==============================================================================


class TestEwaldSlabDtypes:
    """Ewald slab outputs should preserve established dtype conventions."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_ewald_slab_output_dtypes(self, device, dtype):
        """Ewald slab energies/charge-gradients are fp64; forces/virial match input."""
        positions, charges, cell, pbc, nl, ptr, shifts = _make_cscl_ewald_inputs(
            dtype, device
        )

        energies, forces, charge_grads, virial = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_cutoff=EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
            pbc=pbc,
            slab_correction=True,
        )

        assert positions.dtype == dtype
        assert charges.dtype == dtype
        assert cell.dtype == dtype
        assert energies.dtype == torch.float64
        assert forces.dtype == dtype
        assert charge_grads.dtype == torch.float64
        assert virial.dtype == dtype


# ==============================================================================
# pbc=None handling
# ==============================================================================


class TestPbcNoneHandling:
    """slab_correction=True without pbc must raise a clear error."""

    def test_missing_pbc_raises(self, device):
        dtype = torch.float64

        positions, charges, cell, _ = _make_cscl_slab_system(dtype, device)
        nl, ptr, shifts = _build_neighbor_list(
            positions, cell, 5.0, [True, True, True], device
        )

        with pytest.raises(ValueError, match="pbc"):
            ewald_summation(
                positions,
                charges,
                cell,
                alpha=EWALD_ALPHA,
                k_cutoff=EWALD_K_CUTOFF,
                neighbor_list=nl,
                neighbor_ptr=ptr,
                neighbor_shifts=shifts,
                slab_correction=True,  # no pbc provided
            )


# ==============================================================================
# Standalone compute_slab_correction() API tests
# ==============================================================================


class TestStandaloneSlabAPI:
    """Standalone compute_slab_correction() should validate output shapes and edges."""

    def test_standalone_outputs_subset(self, device):
        """Standalone API should return the right tuple based on flags."""
        dtype = torch.float64

        positions, charges, cell, pbc = _make_cscl_slab_system(dtype, device)

        # Energy only -> single tensor
        out = compute_slab_correction(positions, charges, cell, pbc)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (positions.shape[0],)

        # Energy + forces
        out = compute_slab_correction(
            positions, charges, cell, pbc, compute_forces=True
        )
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert out[1].shape == positions.shape

        # Energy + forces + charge grads + virial
        out = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        assert isinstance(out, tuple)
        assert len(out) == 4
        e, f, cg, v = out
        assert e.shape == (positions.shape[0],)
        assert f.shape == positions.shape
        assert cg.shape == (positions.shape[0],)
        assert v.shape == (1, 3, 3)

    def test_standalone_pbc_single_system_1d_matches_2d(self, device):
        """A (3,) pbc tensor is accepted for a single system."""
        dtype = torch.float64

        positions, charges, cell, _ = _make_cscl_slab_system(dtype, device)

        pbc_1d = torch.tensor([True, True, False], device=device)
        pbc_2d = torch.tensor([[True, True, False]], device=device)

        e_1d = compute_slab_correction(positions, charges, cell, pbc_1d)
        e_2d = compute_slab_correction(positions, charges, cell, pbc_2d)

        # Single-system 1D and explicit 2D pbc produce identical energies.
        torch.testing.assert_close(e_1d, e_2d, rtol=0, atol=0)

    def test_standalone_pbc_batched_requires_per_system_pbc(self, device):
        """Batched slab calls require explicit per-system pbc."""
        dtype = torch.float64

        positions_a, charges_a, cell_a, _ = _make_cscl_slab_system(dtype, device)
        positions_b = positions_a + torch.tensor(
            [0.2, -0.1, 0.3], dtype=dtype, device=device
        )
        charges_b = charges_a.clone()
        cell_b = cell_a.clone()

        positions = torch.cat([positions_a, positions_b], dim=0)
        charges = torch.cat([charges_a, charges_b], dim=0)
        cell = torch.cat([cell_a, cell_b], dim=0)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        pbc_1d = torch.tensor([True, True, False], device=device)

        with pytest.raises(ValueError, match="requires `pbc` shape"):
            compute_slab_correction(
                positions,
                charges,
                cell,
                pbc_1d,
                batch_idx=batch_idx,
            )

    def test_single_atom_non_neutral(self, device):
        """Single charged slabs should keep finite background terms."""
        dtype = torch.float64

        positions = torch.tensor([[1.2, -0.4, 3.5]], dtype=dtype, device=device)
        charges = torch.tensor([0.7], dtype=dtype, device=device)
        cell = torch.diag(
            torch.tensor([8.0, 9.0, 24.0], dtype=dtype, device=device)
        ).unsqueeze(0)
        pbc = torch.tensor([True, True, False], device=device)

        energies, forces, charge_grads, virial = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        ref_energies, ref_forces, ref_charge_grads, ref_virial = (
            _reference_slab_correction(positions, charges, cell, pbc)
        )

        # Single-atom non-neutral energies match the independent reference.
        torch.testing.assert_close(energies, ref_energies, rtol=1e-12, atol=1e-15)
        # Single-atom non-neutral forces match the independent reference.
        torch.testing.assert_close(forces, ref_forces, rtol=1e-12, atol=1e-15)
        # Single-atom non-neutral charge gradients match the independent reference.
        torch.testing.assert_close(
            charge_grads, ref_charge_grads, rtol=1e-12, atol=1e-15
        )
        # Single-atom non-neutral virial matches the independent reference.
        torch.testing.assert_close(virial[0], ref_virial, rtol=1e-12, atol=1e-15)

    def test_empty_standalone_system(self, device):
        """Empty standalone slab calls should return empty outputs and zero virial."""
        dtype = torch.float64

        positions = torch.empty((0, 3), dtype=dtype, device=device)
        charges = torch.empty((0,), dtype=dtype, device=device)
        cell = torch.diag(
            torch.tensor([8.0, 9.0, 24.0], dtype=dtype, device=device)
        ).unsqueeze(0)
        pbc = torch.tensor([True, True, False], device=device)

        energies, forces, charge_grads, virial = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )

        assert energies.shape == (0,)
        assert forces.shape == (0, 3)
        assert charge_grads.shape == (0,)
        assert virial.shape == (1, 3, 3)
        # Empty slab virial is exactly zero.
        torch.testing.assert_close(virial, torch.zeros_like(virial), rtol=0, atol=0)


# ==============================================================================
# Hybrid forces integration
# ==============================================================================


class TestSlabHybridForces:
    """Slab correction must respect ewald_summation hybrid_forces semantics."""

    def _ewald_inputs(self, positions, cell):
        """Build detached geometry inputs for focused hybrid tests."""
        device = positions.device
        nl, ptr, shifts = _build_neighbor_list(
            positions.detach(),
            cell.detach(),
            REAL_SPACE_CUTOFF,
            [True, True, True],
            device,
        )
        k_vectors = generate_k_vectors_ewald_summation(cell.detach(), EWALD_K_CUTOFF)
        return nl, ptr, shifts, k_vectors

    def test_hybrid_backward_charge_grad_matches_standard(self, device):
        """Hybrid backward injects charge gradients without position/cell paths."""
        dtype = torch.float64

        positions_ref, charges_ref, cell_ref, pbc = _make_cscl_slab_system(
            dtype, device
        )
        nl, ptr, shifts, k_vectors = self._ewald_inputs(positions_ref, cell_ref)

        charges_std = charges_ref.clone().requires_grad_(True)
        e_std = ewald_summation(
            positions_ref,
            charges_std,
            cell_ref,
            alpha=EWALD_ALPHA,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            pbc=pbc,
            slab_correction=True,
        )
        grad_std = torch.autograd.grad(e_std.sum(), charges_std)[0]

        positions_hyb = positions_ref.clone().requires_grad_(True)
        charges_hyb = charges_ref.clone().requires_grad_(True)
        cell_hyb = cell_ref.clone().requires_grad_(True)
        e_hyb = ewald_summation(
            positions_hyb,
            charges_hyb,
            cell_hyb,
            alpha=EWALD_ALPHA,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            pbc=pbc,
            slab_correction=True,
            hybrid_forces=True,
        )
        e_hyb.sum().backward()

        assert positions_hyb.grad is None or torch.all(positions_hyb.grad == 0)
        assert cell_hyb.grad is None or torch.all(cell_hyb.grad == 0)
        assert charges_hyb.grad is not None
        # Hybrid injected charge gradients match standard autograd gradients.
        torch.testing.assert_close(
            charges_hyb.grad,
            grad_std,
            rtol=SLAB_CHARGE_GRAD_RTOL,
            atol=SLAB_CHARGE_GRAD_ATOL,
        )

    def test_hybrid_forward_outputs_match_standard(self, device):
        """Hybrid forward outputs match standard energy, forces, charge grads, virial."""
        dtype = torch.float64

        positions, charges, cell, pbc = _make_cscl_slab_system(dtype, device)
        nl, ptr, shifts, k_vectors = self._ewald_inputs(positions, cell)

        e_std, f_std, cg_std, v_std = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
            pbc=pbc,
            slab_correction=True,
        )
        e_hyb, f_hyb, cg_hyb, v_hyb = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
            pbc=pbc,
            slab_correction=True,
            hybrid_forces=True,
        )

        # Hybrid forward energies match standard mode energies.
        torch.testing.assert_close(e_hyb, e_std, rtol=1e-12, atol=1e-15)
        # Hybrid forward forces match standard mode forces.
        torch.testing.assert_close(f_hyb, f_std, rtol=1e-12, atol=1e-15)
        # Hybrid returned charge gradients match standard returned charge gradients.
        torch.testing.assert_close(cg_hyb, cg_std, rtol=1e-12, atol=1e-15)
        # Hybrid forward virial matches standard mode virial.
        torch.testing.assert_close(v_hyb, v_std, rtol=1e-12, atol=1e-15)
        assert v_hyb.grad_fn is None


# ==============================================================================
# PME slab integration
# ==============================================================================


class TestPMESlabIntegration:
    """Slab correction integration should work through full PME."""

    def test_full_pme_slab_matches_component_sum(self, device):
        """PME slab equals real + reciprocal + standalone slab correction."""
        dtype = torch.float64
        mesh_dimensions = (16, 16, 16)

        positions, charges, cell, pbc, nl, ptr, shifts = _make_triclinic_ewald_inputs(
            dtype, device
        )
        alpha = torch.tensor([EWALD_ALPHA], dtype=dtype, device=device)

        energies, forces, charge_grads, virial = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
            pbc=pbc,
            slab_correction=True,
        )

        real = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        reciprocal = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=mesh_dimensions,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        slab = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )

        # Full PME slab energies equal real + reciprocal + slab correction.
        torch.testing.assert_close(
            energies,
            real[0] + reciprocal[0] + slab[0],
            rtol=SLAB_STRICT_RTOL,
            atol=SLAB_STRICT_ATOL,
        )
        # Full PME slab forces equal real + reciprocal + slab correction.
        torch.testing.assert_close(
            forces,
            real[1] + reciprocal[1] + slab[1],
            rtol=SLAB_STRICT_RTOL,
            atol=SLAB_STRICT_ATOL,
        )
        # Full PME slab charge gradients equal real + reciprocal + slab correction.
        torch.testing.assert_close(
            charge_grads,
            real[2] + reciprocal[2] + slab[2],
            rtol=SLAB_STRICT_RTOL,
            atol=SLAB_STRICT_ATOL,
        )
        # Full PME slab virial equals real + reciprocal + slab correction.
        torch.testing.assert_close(
            virial,
            real[3] + reciprocal[3] + slab[3],
            rtol=SLAB_STRICT_RTOL,
            atol=SLAB_STRICT_ATOL,
        )

    def test_full_pme_slab_energy_gradcheck(self, device):
        """PME slab total energy passes gradcheck for positions, charges, and cell."""
        dtype = torch.float64

        positions, charges, cell, pbc, nl, ptr, shifts = _make_cscl_ewald_inputs(
            dtype, device
        )
        positions = positions.clone().detach().requires_grad_(True)
        charges = charges.clone().detach().requires_grad_(True)
        cell = cell.clone().detach().requires_grad_(True)

        def pme_slab_energy(positions_in, charges_in, cell_in):
            return particle_mesh_ewald(
                positions_in,
                charges_in,
                cell_in,
                alpha=EWALD_ALPHA,
                mesh_dimensions=(8, 8, 8),
                neighbor_list=nl,
                neighbor_ptr=ptr,
                neighbor_shifts=shifts,
                pbc=pbc,
                slab_correction=True,
            ).sum()

        assert torch.autograd.gradcheck(
            pme_slab_energy,
            (positions, charges, cell),
            eps=1e-6,
            atol=PME_SLAB_GRADCHECK_ATOL,
            rtol=PME_SLAB_GRADCHECK_RTOL,
            nondet_tol=1e-12,
        )

    def test_full_pme_slab_matches_autograd(self, device):
        """PME slab analytical forces and charge gradients match autograd."""
        dtype = torch.float64

        positions, charges, cell, pbc, nl, ptr, shifts = _make_triclinic_ewald_inputs(
            dtype, device
        )
        positions = positions.clone().detach().requires_grad_(True)
        charges = charges.clone().detach().requires_grad_(True)

        energies, forces, charge_grads = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            mesh_dimensions=(24, 24, 24),
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_charge_gradients=True,
            pbc=pbc,
            slab_correction=True,
        )
        autograd_positions, autograd_charge_grads = torch.autograd.grad(
            energies.sum(), (positions, charges), create_graph=False
        )

        # Full PME slab forces match autograd forces.
        torch.testing.assert_close(forces, -autograd_positions, rtol=1e-4, atol=5e-6)
        # Full PME slab charge gradients match autograd charge gradients.
        torch.testing.assert_close(
            charge_grads, autograd_charge_grads, rtol=1e-8, atol=1e-10
        )

    def test_full_pme_slab_virial_matches_strain_autograd(self, device):
        """PME slab virial matches autograd under affine strain."""
        dtype = torch.float64

        positions, charges, cell, pbc, nl, ptr, shifts = _make_triclinic_ewald_inputs(
            dtype, device
        )

        _, _, virial = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            mesh_dimensions=(16, 16, 16),
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_virial=True,
            pbc=pbc,
            slab_correction=True,
        )
        eps = torch.zeros(3, 3, dtype=dtype, device=device, requires_grad=True)
        deformation = torch.eye(3, dtype=dtype, device=device) + eps
        e_strained = particle_mesh_ewald(
            positions @ deformation.T,
            charges,
            cell @ deformation.T,
            alpha=EWALD_ALPHA,
            mesh_dimensions=(16, 16, 16),
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            pbc=pbc,
            slab_correction=True,
        ).sum()
        autograd_virial = -torch.autograd.grad(e_strained, eps)[0]

        # Full PME slab virial matches strain autograd.
        torch.testing.assert_close(virial[0], autograd_virial, rtol=1e-8, atol=1e-7)

    def test_full_pme_slab_3d_pbc_noop(self, device):
        """3D periodic PME slab mode matches standard PME exactly."""
        dtype = torch.float64

        positions, charges, cell, _, nl, ptr, shifts = _make_triclinic_ewald_inputs(
            dtype, device
        )
        pbc_3d = torch.tensor([True, True, True], dtype=torch.bool, device=device)
        common_kwargs = {
            "alpha": EWALD_ALPHA,
            "mesh_dimensions": (16, 16, 16),
            "neighbor_list": nl,
            "neighbor_ptr": ptr,
            "neighbor_shifts": shifts,
            "compute_forces": True,
            "compute_charge_gradients": True,
            "compute_virial": True,
        }

        e_off, f_off, cg_off, v_off = particle_mesh_ewald(
            positions,
            charges,
            cell,
            **common_kwargs,
        )
        e_3d, f_3d, cg_3d, v_3d = particle_mesh_ewald(
            positions,
            charges,
            cell,
            pbc=pbc_3d,
            slab_correction=True,
            **common_kwargs,
        )

        # 3D pbc slab mode leaves PME energies unchanged.
        torch.testing.assert_close(
            e_3d, e_off, rtol=SLAB_STRICT_RTOL, atol=SLAB_STRICT_ATOL
        )
        # 3D pbc slab mode leaves PME forces unchanged.
        torch.testing.assert_close(
            f_3d, f_off, rtol=SLAB_STRICT_RTOL, atol=SLAB_STRICT_ATOL
        )
        # 3D pbc slab mode leaves PME charge gradients unchanged.
        torch.testing.assert_close(
            cg_3d, cg_off, rtol=SLAB_STRICT_RTOL, atol=SLAB_STRICT_ATOL
        )
        # 3D pbc slab mode leaves PME virial unchanged.
        torch.testing.assert_close(
            v_3d, v_off, rtol=SLAB_STRICT_RTOL, atol=SLAB_STRICT_ATOL
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_full_pme_slab_output_dtypes(self, device, dtype):
        """PME slab energies/charge-gradients are fp64; forces/virial match input."""
        positions, charges, cell, pbc, nl, ptr, shifts = _make_cscl_ewald_inputs(
            dtype, device
        )

        energies, forces, charge_grads, virial = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            mesh_dimensions=(16, 16, 16),
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
            pbc=pbc,
            slab_correction=True,
        )

        assert positions.dtype == dtype
        assert charges.dtype == dtype
        assert cell.dtype == dtype
        assert energies.dtype == torch.float64
        assert forces.dtype == dtype
        assert charge_grads.dtype == torch.float64
        assert virial.dtype == dtype

    @pytest.mark.parametrize(
        (
            "compute_forces",
            "compute_charge_gradients",
            "compute_virial",
            "output_names",
        ),
        [
            (False, False, False, ("energies",)),
            (True, False, False, ("energies", "forces")),
            (False, True, False, ("energies", "charge_grads")),
            (False, False, True, ("energies", "virial")),
            (True, True, False, ("energies", "forces", "charge_grads")),
            (True, False, True, ("energies", "forces", "virial")),
            (False, True, True, ("energies", "charge_grads", "virial")),
            (True, True, True, ("energies", "forces", "charge_grads", "virial")),
        ],
    )
    def test_full_pme_slab_output_subsets(
        self,
        device,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
        output_names,
    ):
        """PME slab output tuple ordering and values follow enabled output flags."""
        dtype = torch.float64
        mesh_dimensions = (16, 16, 16)

        positions, charges, cell, pbc, nl, ptr, shifts = _make_cscl_ewald_inputs(
            dtype, device
        )
        alpha = torch.tensor([EWALD_ALPHA], dtype=dtype, device=device)
        common_kwargs = {
            "positions": positions,
            "charges": charges,
            "cell": cell,
            "alpha": alpha,
            "mesh_dimensions": mesh_dimensions,
            "neighbor_list": nl,
            "neighbor_ptr": ptr,
            "neighbor_shifts": shifts,
            "pbc": pbc,
            "slab_correction": True,
        }

        result = particle_mesh_ewald(
            **common_kwargs,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
        )
        if len(output_names) == 1:
            assert isinstance(result, torch.Tensor)
        else:
            assert isinstance(result, tuple)
        result_tuple = result if isinstance(result, tuple) else (result,)

        real = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        reciprocal = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=mesh_dimensions,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        slab = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        expected_by_name = {
            "energies": real[0] + reciprocal[0] + slab[0],
            "forces": real[1] + reciprocal[1] + slab[1],
            "charge_grads": real[2] + reciprocal[2] + slab[2],
            "virial": real[3] + reciprocal[3] + slab[3],
        }

        assert len(result_tuple) == len(output_names)
        for output, name in zip(result_tuple, output_names, strict=True):
            torch.testing.assert_close(
                output,
                expected_by_name[name],
                rtol=SLAB_STRICT_RTOL,
                atol=SLAB_STRICT_ATOL,
            )

    @pytest.mark.skipif(not HAS_TORCHPME, reason="torch-pme not installed")
    def test_full_pme_slab_matches_torchpme(self, device):
        """Full PME slab energy and forces match torch-pme PMECalculator."""
        dtype = torch.float64
        mesh_spacing = 1.0

        positions, charges, cell, pbc, nl, ptr, shifts = _make_cscl_ewald_inputs(
            dtype, device
        )

        our_energies, our_forces = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            mesh_spacing=mesh_spacing,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            pbc=pbc,
            slab_correction=True,
        )

        positions_tp = positions.clone().detach().requires_grad_(True)
        torchpme_potential = _run_torchpme_pme(
            positions_tp,
            charges,
            cell.squeeze(0),
            pbc,
            EWALD_ALPHA,
            mesh_spacing,
        )
        torchpme_total = (torchpme_potential.squeeze(-1) * charges).sum()
        torchpme_forces = -torch.autograd.grad(
            torchpme_total, positions_tp, create_graph=False
        )[0]

        # Total PME slab energy matches torch-pme.
        torch.testing.assert_close(
            our_energies.sum(), torchpme_total, rtol=1e-5, atol=1e-8
        )
        # Total PME slab forces match torch-pme autograd forces.
        torch.testing.assert_close(our_forces, torchpme_forces, rtol=1e-4, atol=1e-6)

    def test_full_pme_slab_requires_pbc(self, device):
        """PME slab correction requires explicit slab periodicity."""
        dtype = torch.float64

        positions, charges, cell, _, nl, ptr, shifts = _make_triclinic_ewald_inputs(
            dtype, device
        )

        with pytest.raises(ValueError, match="requires an explicit `pbc`"):
            particle_mesh_ewald(
                positions,
                charges,
                cell,
                alpha=EWALD_ALPHA,
                mesh_dimensions=(16, 16, 16),
                neighbor_list=nl,
                neighbor_ptr=ptr,
                neighbor_shifts=shifts,
                slab_correction=True,
            )

    def test_full_pme_slab_pbc_single_system_1d_matches_2d(self, device):
        """PME slab wrapper accepts equivalent (3,) and (1, 3) pbc tensors."""
        dtype = torch.float64

        positions, charges, cell, pbc_1d, nl, ptr, shifts = _make_cscl_ewald_inputs(
            dtype, device
        )
        pbc_2d = pbc_1d.unsqueeze(0)
        common_kwargs = {
            "alpha": EWALD_ALPHA,
            "mesh_dimensions": (16, 16, 16),
            "neighbor_list": nl,
            "neighbor_ptr": ptr,
            "neighbor_shifts": shifts,
            "compute_forces": True,
            "compute_charge_gradients": True,
            "compute_virial": True,
            "slab_correction": True,
        }

        out_1d = particle_mesh_ewald(
            positions,
            charges,
            cell,
            pbc=pbc_1d,
            **common_kwargs,
        )
        out_2d = particle_mesh_ewald(
            positions,
            charges,
            cell,
            pbc=pbc_2d,
            **common_kwargs,
        )

        for actual, expected in zip(out_1d, out_2d, strict=True):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=SLAB_STRICT_RTOL,
                atol=SLAB_STRICT_ATOL,
            )

    def test_full_pme_slab_pbc_batched_requires_per_system_pbc(self, device):
        """PME slab wrapper rejects (3,) pbc for batched systems."""
        dtype = torch.float64

        positions_a, charges_a, cell_a, pbc_1d = _make_cscl_slab_system(dtype, device)
        positions_b = positions_a + torch.tensor(
            [0.2, -0.1, 0.3], dtype=dtype, device=device
        )
        positions = torch.cat([positions_a, positions_b], dim=0)
        charges = torch.cat([charges_a, charges_a], dim=0)
        cell = torch.cat([cell_a, cell_a], dim=0)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        with pytest.raises(ValueError, match="requires `pbc` shape"):
            particle_mesh_ewald(
                positions,
                charges,
                cell,
                alpha=EWALD_ALPHA,
                mesh_dimensions=(16, 16, 16),
                batch_idx=batch_idx,
                pbc=pbc_1d,
                slab_correction=True,
            )

    def test_mixed_pbc_batch_matches_component_sum(self, device):
        """Batched PME slab handles one slab system and one 3D-periodic system."""
        dtype = torch.float64
        mesh_dimensions = (16, 16, 16)

        pos_slab, q_slab, cell_slab, _ = _make_triclinic_slab_system(dtype, device)
        cell_slab = cell_slab.squeeze(0)
        pos_3d = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [2.0, 3.0, 8.0],
                [7.0, 4.0, 1.0],
            ],
            dtype=dtype,
            device=device,
        )
        q_3d = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=dtype, device=device)
        cell_3d = torch.eye(3, dtype=dtype, device=device) * 12.0

        positions = torch.cat([pos_slab, pos_3d], dim=0)
        charges = torch.cat([q_slab, q_3d], dim=0)
        cell = torch.stack([cell_slab, cell_3d], dim=0)
        batch_idx = torch.tensor(
            [0] * pos_slab.shape[0] + [1] * pos_3d.shape[0],
            dtype=torch.int32,
            device=device,
        )
        pbc_3d = torch.tensor(
            [[True, True, True], [True, True, True]],
            dtype=torch.bool,
            device=device,
        )
        pbc_slab_mixed = torch.tensor(
            [[True, True, False], [True, True, True]],
            dtype=torch.bool,
            device=device,
        )
        nl, ptr, shifts = batch_cell_list(
            positions,
            cutoff=REAL_SPACE_CUTOFF,
            cell=cell,
            pbc=pbc_3d,
            batch_idx=batch_idx,
            return_neighbor_list=True,
        )
        alpha = torch.tensor([EWALD_ALPHA, EWALD_ALPHA], dtype=dtype, device=device)

        energies, forces = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            batch_idx=batch_idx,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            pbc=pbc_slab_mixed,
            slab_correction=True,
        )
        energies_3d, forces_3d = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            batch_idx=batch_idx,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
        )
        slab_energies, slab_forces = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc_slab_mixed,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        # Mixed-batch PME energies equal 3D PME plus slab correction.
        torch.testing.assert_close(
            energies,
            energies_3d + slab_energies,
            rtol=SLAB_STRICT_RTOL,
            atol=SLAB_STRICT_ATOL,
        )
        # Mixed-batch PME forces equal 3D PME plus slab correction.
        torch.testing.assert_close(
            forces,
            forces_3d + slab_forces,
            rtol=SLAB_STRICT_RTOL,
            atol=SLAB_STRICT_ATOL,
        )
        # The 3D-periodic system receives zero slab energy contribution.
        torch.testing.assert_close(
            slab_energies[pos_slab.shape[0] :],
            torch.zeros_like(slab_energies[pos_slab.shape[0] :]),
            rtol=0.0,
            atol=1e-14,
        )
        # The 3D-periodic system receives zero slab force contribution.
        torch.testing.assert_close(
            slab_forces[pos_slab.shape[0] :],
            torch.zeros_like(slab_forces[pos_slab.shape[0] :]),
            rtol=0.0,
            atol=1e-14,
        )

    def test_hybrid_returned_charge_grads_match_standard(self, device):
        """Hybrid PME slab returns the same charge gradients as standard mode."""
        dtype = torch.float64

        positions, charges, cell, pbc, nl, ptr, shifts = _make_triclinic_ewald_inputs(
            dtype, device
        )
        charges = charges.clone().requires_grad_(True)
        common_kwargs = {
            "alpha": EWALD_ALPHA,
            "mesh_dimensions": (16, 16, 16),
            "neighbor_list": nl,
            "neighbor_ptr": ptr,
            "neighbor_shifts": shifts,
            "compute_forces": True,
            "compute_charge_gradients": True,
            "compute_virial": True,
            "pbc": pbc,
            "slab_correction": True,
        }

        e_std, f_std, cg_std, v_std = particle_mesh_ewald(
            positions,
            charges,
            cell,
            **common_kwargs,
        )
        e_hyb, f_hyb, cg_hyb, v_hyb = particle_mesh_ewald(
            positions,
            charges,
            cell,
            hybrid_forces=True,
            **common_kwargs,
        )

        # Hybrid PME slab energies match standard mode energies.
        torch.testing.assert_close(e_hyb, e_std, rtol=1e-12, atol=1e-15)
        # Hybrid PME slab forces match standard mode forces.
        torch.testing.assert_close(f_hyb, f_std, rtol=1e-12, atol=1e-15)
        # Hybrid PME slab returned charge gradients match standard mode.
        torch.testing.assert_close(cg_hyb, cg_std, rtol=1e-12, atol=1e-15)
        # Hybrid PME slab virial matches standard mode virial.
        torch.testing.assert_close(v_hyb, v_std, rtol=1e-12, atol=1e-15)

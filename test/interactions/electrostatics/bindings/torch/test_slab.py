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

import warnings

import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    compute_slab_correction,
)
from nvalchemiops.torch.interactions.electrostatics import (
    ewald_real_space as _ewald_real_space,
)
from nvalchemiops.torch.interactions.electrostatics import (
    particle_mesh_ewald as _particle_mesh_ewald,
)
from nvalchemiops.torch.interactions.electrostatics import (
    pme_reciprocal_space as _pme_reciprocal_space,
)
from nvalchemiops.torch.interactions.electrostatics.ewald import (
    ewald_summation as _ewald_summation,
)
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
SLAB_CHARGE_GRAD_RTOL = 2e-7
SLAB_CHARGE_GRAD_ATOL = 5e-8
PME_SLAB_GRADCHECK_RTOL = 1e-6
PME_SLAB_GRADCHECK_ATOL = 1e-9


def _call_without_direct_output_deprecation(api_name, api, *args, **kwargs):
    """Call a deprecated direct-output full API without polluting test warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=f"The direct-output flags .* on {api_name} are deprecated",
            category=DeprecationWarning,
        )
        return api(*args, **kwargs)


def _call_without_component_direct_output_deprecation(api_name, api, *args, **kwargs):
    """Call deprecated component direct outputs without polluting test warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=f"The component direct-output flag.* on {api_name} are deprecated",
            category=DeprecationWarning,
        )
        return api(*args, **kwargs)


def ewald_summation(*args, **kwargs):
    """Test-local wrapper suppressing intentional direct-output deprecations."""
    return _call_without_direct_output_deprecation(
        "ewald_summation",
        _ewald_summation,
        *args,
        **kwargs,
    )


def ewald_real_space(*args, **kwargs):
    """Test-local wrapper suppressing intentional component deprecations."""
    return _call_without_component_direct_output_deprecation(
        "ewald_real_space",
        _ewald_real_space,
        *args,
        **kwargs,
    )


def particle_mesh_ewald(*args, **kwargs):
    """Test-local wrapper suppressing intentional direct-output deprecations."""
    return _call_without_direct_output_deprecation(
        "particle_mesh_ewald",
        _particle_mesh_ewald,
        *args,
        **kwargs,
    )


def pme_reciprocal_space(*args, **kwargs):
    """Test-local wrapper suppressing intentional component deprecations."""
    return _call_without_component_direct_output_deprecation(
        "pme_reciprocal_space",
        _pme_reciprocal_space,
        *args,
        **kwargs,
    )


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
    """Create CsCl slab inputs plus a T/T/F real-space neighbor list."""
    positions, charges, cell, pbc = _make_cscl_slab_system(dtype, device)
    neighbor_list, neighbor_ptr, neighbor_shifts = _build_neighbor_list(
        positions, cell, REAL_SPACE_CUTOFF, [True, True, False], device
    )
    return positions, charges, cell, pbc, neighbor_list, neighbor_ptr, neighbor_shifts


def _make_triclinic_ewald_inputs(dtype=torch.float64, device="cpu"):
    """Create triclinic slab inputs plus a T/T/F real-space neighbor list."""
    positions, charges, cell, pbc = _make_triclinic_slab_system(dtype, device)
    neighbor_list, neighbor_ptr, neighbor_shifts = _build_neighbor_list(
        positions, cell, REAL_SPACE_CUTOFF, [True, True, False], device
    )
    return positions, charges, cell, pbc, neighbor_list, neighbor_ptr, neighbor_shifts


def _build_neighbor_list(positions, cell, cutoff, pbc, device="cpu"):
    """Build neighbor list using cell_list with explicit pbc."""
    pbc_tensor = torch.tensor(pbc, dtype=torch.bool, device=device).unsqueeze(0)
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
    """Slab direct-output calls should preserve energy autograd."""

    def test_slab_energy_gradcheck_with_direct_outputs(self, device):
        """Slab energy remains differentiable when direct outputs are requested."""
        dtype = torch.float64

        positions, charges, cell, pbc = _make_triclinic_slab_system(dtype, device)
        positions = positions.clone().detach().requires_grad_(True)
        charges = charges.clone().detach().requires_grad_(True)
        cell = cell.clone().detach().requires_grad_(True)

        def slab_energy(positions_in, charges_in, cell_in):
            return compute_slab_correction(
                positions_in,
                charges_in,
                cell_in,
                pbc,
                compute_forces=True,
                compute_charge_gradients=True,
                compute_virial=True,
            )[0]

        assert torch.autograd.gradcheck(
            slab_energy,
            (positions, charges, cell),
            eps=1e-6,
            atol=1e-10,
            rtol=1e-8,
            nondet_tol=1e-12,
        )


class TestSlabEnergyDerivativeContract:
    """Slab energy should support the PME/Ewald energy-derivative contract."""

    @pytest.mark.slow
    def test_energy_gradgradcheck_positions_charges_cell(self, device):
        """Energy-only slab correction supports second derivatives."""
        dtype = torch.float64

        positions, charges, cell, pbc = _make_triclinic_slab_system(dtype, device)
        positions = positions.clone().detach().requires_grad_(True)
        charges = charges.clone().detach().requires_grad_(True)
        cell = cell.clone().detach().requires_grad_(True)

        def slab_energy(positions_in, charges_in, cell_in):
            return compute_slab_correction(
                positions_in,
                charges_in,
                cell_in,
                pbc,
            ).sum()

        assert torch.autograd.gradgradcheck(
            slab_energy,
            (positions, charges, cell),
            eps=1e-6,
            atol=1e-9,
            rtol=1e-6,
            nondet_tol=1e-12,
        )

    @pytest.mark.parametrize("scale", [1.0e-3, 1.0e3])
    def test_scaled_hvp_matches_torch_reference(self, device, scale):
        """Scaled coordinates/cells use analytic HVPs, not an absolute FD step."""
        dtype = torch.float64

        positions, charges, cell, pbc = _make_triclinic_slab_system(dtype, device)
        positions = (positions * scale).clone().detach().requires_grad_(True)
        charges = charges.clone().detach().requires_grad_(True)
        cell = (cell * scale).clone().detach().requires_grad_(True)
        h_positions = torch.tensor(
            [[0.2, -0.1, 0.3], [-0.4, 0.5, -0.2], [0.1, 0.2, -0.3]],
            device=device,
            dtype=dtype,
        )
        h_charges = torch.tensor([0.3, -0.2, 0.5], device=device, dtype=dtype)
        h_cell = torch.tensor(
            [[[0.03, -0.02, 0.01], [-0.01, 0.04, -0.03], [0.02, 0.01, -0.02]]],
            device=device,
            dtype=dtype,
        )

        def hvp(energy_fn):
            grad_pos, grad_q, grad_cell = torch.autograd.grad(
                energy_fn(positions, charges, cell),
                (positions, charges, cell),
                create_graph=True,
            )
            directional = (
                (grad_pos * h_positions).sum()
                + (grad_q * h_charges).sum()
                + (grad_cell * h_cell).sum()
            )
            return torch.autograd.grad(directional, (positions, charges, cell))

        def slab_energy(positions_in, charges_in, cell_in):
            return compute_slab_correction(
                positions_in,
                charges_in,
                cell_in,
                pbc,
            ).sum()

        def ref_energy(positions_in, charges_in, cell_in):
            return _reference_slab_correction(positions_in, charges_in, cell_in, pbc)[
                0
            ].sum()

        actual = hvp(slab_energy)
        expected = hvp(ref_energy)
        for actual_part, expected_part in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                actual_part,
                expected_part,
                rtol=1e-8,
                atol=1e-8,
            )

    def test_weighted_energy_hvp_matches_torch_reference(self, device):
        """Non-uniform per-atom energy weights use the analytic Warp HVP path."""
        dtype = torch.float64

        positions, charges, cell, pbc = _make_triclinic_slab_system(dtype, device)
        positions = positions.clone().detach().requires_grad_(True)
        charges = charges.clone().detach().requires_grad_(True)
        cell = cell.clone().detach().requires_grad_(True)
        weights = torch.tensor([0.4, -1.2, 0.7], device=device, dtype=dtype)
        h_positions = torch.tensor(
            [[0.2, -0.1, 0.3], [-0.4, 0.5, -0.2], [0.1, 0.2, -0.3]],
            device=device,
            dtype=dtype,
        )
        h_charges = torch.tensor([0.3, -0.2, 0.5], device=device, dtype=dtype)
        h_cell = torch.tensor(
            [[[0.03, -0.02, 0.01], [-0.01, 0.04, -0.03], [0.02, 0.01, -0.02]]],
            device=device,
            dtype=dtype,
        )

        def hvp(energy_fn):
            grad_pos, grad_q, grad_cell = torch.autograd.grad(
                energy_fn(positions, charges, cell),
                (positions, charges, cell),
                create_graph=True,
            )
            directional = (
                (grad_pos * h_positions).sum()
                + (grad_q * h_charges).sum()
                + (grad_cell * h_cell).sum()
            )
            return torch.autograd.grad(directional, (positions, charges, cell))

        def slab_energy(positions_in, charges_in, cell_in):
            return (
                compute_slab_correction(positions_in, charges_in, cell_in, pbc)
                * weights
            ).sum()

        def ref_energy(positions_in, charges_in, cell_in):
            return (
                _reference_slab_correction(positions_in, charges_in, cell_in, pbc)[0]
                * weights
            ).sum()

        actual = hvp(slab_energy)
        expected = hvp(ref_energy)
        for actual_part, expected_part in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                actual_part,
                expected_part,
                rtol=1e-8,
                atol=1e-8,
            )

    def test_qr_force_loss_double_backward(self, device):
        """q(R) slab forces include the charge-model chain rule in second order."""
        dtype = torch.float64

        positions, charges_ref, cell, pbc = _make_triclinic_slab_system(dtype, device)
        positions = positions.clone().detach().requires_grad_(True)
        cell = cell.clone().detach().requires_grad_(True)

        charges = charges_ref + 0.02 * positions[:, 2]
        energy = compute_slab_correction(positions, charges, cell, pbc)
        forces = -torch.autograd.grad(
            energy.sum(),
            positions,
            create_graph=True,
        )[0]
        loss = forces.square().sum()
        grad_positions, grad_cell = torch.autograd.grad(loss, (positions, cell))

        assert torch.isfinite(grad_positions).all()
        assert torch.isfinite(grad_cell).all()


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

    def test_batched_mixed_axes_precomputed_geometry_matches_reference(self, device):
        """Batched slab geometry handles mixed slab axes and 3D no-op systems."""
        dtype = torch.float64

        pos_z, q_z, cell_z, pbc_z = _make_triclinic_slab_system(dtype, device)
        pos_y = torch.tensor(
            [[1.5, 2.0, 3.0], [4.0, 2.5, 5.5], [7.0, 3.5, 6.0]],
            dtype=dtype,
            device=device,
        )
        q_y = torch.tensor([0.7, -0.2, 0.1], dtype=dtype, device=device)
        cell_y = torch.tensor(
            [[9.5, 0.1, 0.2], [0.4, 24.0, 0.3], [1.1, 0.2, 8.5]],
            dtype=dtype,
            device=device,
        ).unsqueeze(0)
        pbc_y = torch.tensor([True, False, True], device=device)
        pos_3d = torch.tensor(
            [[2.0, 1.0, 3.0], [5.0, 2.0, 4.0]],
            dtype=dtype,
            device=device,
        )
        q_3d = torch.tensor([0.4, -0.4], dtype=dtype, device=device)
        cell_3d = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * 10.0
        pbc_3d = torch.tensor([True, True, True], device=device)

        positions = torch.cat([pos_z, pos_y, pos_3d], dim=0)
        charges = torch.cat([q_z, q_y, q_3d], dim=0)
        cell = torch.cat([cell_z, cell_y, cell_3d], dim=0)
        pbc = torch.stack([pbc_z, pbc_y, pbc_3d], dim=0)
        batch_idx = torch.tensor(
            [0] * pos_z.shape[0] + [1] * pos_y.shape[0] + [2] * pos_3d.shape[0],
            dtype=torch.int32,
            device=device,
        )

        energies, forces, charge_grads, virial = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            batch_idx=batch_idx,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        ref_z = _reference_slab_correction(pos_z, q_z, cell_z, pbc_z)
        ref_y = _reference_slab_correction(pos_y, q_y, cell_y, pbc_y)

        n_z = pos_z.shape[0]
        n_y = pos_y.shape[0]
        torch.testing.assert_close(energies[:n_z], ref_z[0], rtol=1e-12, atol=1e-15)
        torch.testing.assert_close(
            energies[n_z : n_z + n_y], ref_y[0], rtol=1e-12, atol=1e-15
        )
        torch.testing.assert_close(forces[:n_z], ref_z[1], rtol=1e-12, atol=1e-15)
        torch.testing.assert_close(
            forces[n_z : n_z + n_y], ref_y[1], rtol=1e-12, atol=1e-15
        )
        torch.testing.assert_close(charge_grads[:n_z], ref_z[2], rtol=1e-12, atol=1e-15)
        torch.testing.assert_close(
            charge_grads[n_z : n_z + n_y], ref_y[2], rtol=1e-12, atol=1e-15
        )
        torch.testing.assert_close(virial[0], ref_z[3], rtol=1e-12, atol=1e-15)
        torch.testing.assert_close(virial[1], ref_y[3], rtol=1e-12, atol=1e-15)
        torch.testing.assert_close(
            energies[n_z + n_y :], torch.zeros_like(energies[n_z + n_y :])
        )
        torch.testing.assert_close(
            forces[n_z + n_y :], torch.zeros_like(forces[n_z + n_y :])
        )
        torch.testing.assert_close(
            charge_grads[n_z + n_y :], torch.zeros_like(charge_grads[n_z + n_y :])
        )
        torch.testing.assert_close(virial[2], torch.zeros_like(virial[2]))

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
            positions.detach() @ deformation,
            charges.detach(),
            cell @ deformation,
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


class TestSlabNeighborPbc:
    """Vacuum slab neighbor lists should agree for T/T/F and T/T/T pbc."""

    def test_ewald_neighbor_pbc_ttf_matches_ttt_with_vacuum(self, device):
        """Ewald slab outputs are unchanged by T/T/F vs T/T/T vacuum neighbors."""
        dtype = torch.float64
        positions, charges, cell, pbc = _make_cscl_slab_system(dtype, device)
        pbc_slab = pbc.unsqueeze(0)
        pbc_3d = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)

        nl_slab, ptr_slab, shifts_slab = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc_slab,
            return_neighbor_list=True,
        )
        nl_3d, ptr_3d, shifts_3d = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc_3d,
            return_neighbor_list=True,
        )
        common_kwargs = {
            "alpha": EWALD_ALPHA,
            "k_cutoff": EWALD_K_CUTOFF,
            "compute_forces": True,
            "compute_charge_gradients": True,
            "compute_virial": True,
            "pbc": pbc,
            "slab_correction": True,
        }

        outputs_slab = ewald_summation(
            positions,
            charges,
            cell,
            neighbor_list=nl_slab,
            neighbor_ptr=ptr_slab,
            neighbor_shifts=shifts_slab,
            **common_kwargs,
        )
        outputs_3d = ewald_summation(
            positions,
            charges,
            cell,
            neighbor_list=nl_3d,
            neighbor_ptr=ptr_3d,
            neighbor_shifts=shifts_3d,
            **common_kwargs,
        )

        for actual, expected in zip(outputs_slab, outputs_3d, strict=True):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=SLAB_STRICT_RTOL,
                atol=SLAB_STRICT_ATOL,
            )

    def test_pme_neighbor_pbc_ttf_matches_ttt_with_vacuum(self, device):
        """PME slab outputs are unchanged by T/T/F vs T/T/T vacuum neighbors."""
        dtype = torch.float64
        positions, charges, cell, pbc = _make_cscl_slab_system(dtype, device)
        pbc_slab = pbc.unsqueeze(0)
        pbc_3d = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)

        nl_slab, ptr_slab, shifts_slab = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc_slab,
            return_neighbor_list=True,
        )
        nl_3d, ptr_3d, shifts_3d = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc_3d,
            return_neighbor_list=True,
        )
        common_kwargs = {
            "alpha": torch.tensor([EWALD_ALPHA], dtype=dtype, device=device),
            "mesh_dimensions": (16, 16, 16),
            "compute_forces": True,
            "compute_charge_gradients": True,
            "compute_virial": True,
            "pbc": pbc,
            "slab_correction": True,
        }

        outputs_slab = particle_mesh_ewald(
            positions,
            charges,
            cell,
            neighbor_list=nl_slab,
            neighbor_ptr=ptr_slab,
            neighbor_shifts=shifts_slab,
            **common_kwargs,
        )
        outputs_3d = particle_mesh_ewald(
            positions,
            charges,
            cell,
            neighbor_list=nl_3d,
            neighbor_ptr=ptr_3d,
            neighbor_shifts=shifts_3d,
            **common_kwargs,
        )

        for actual, expected in zip(outputs_slab, outputs_3d, strict=True):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=SLAB_STRICT_RTOL,
                atol=SLAB_STRICT_ATOL,
            )


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
# torch.compile behavior
# ==============================================================================


class TestSlabTorchCompile:
    """Standalone slab correction should compile without tracing raw Warp setup."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for torch.compile"
    )
    def test_standalone_slab_direct_outputs_compile(self):
        """Compiled direct outputs match eager and keep energy gradients."""
        device = torch.device("cuda")
        dtype = torch.float64
        positions, charges, cell, pbc = _make_triclinic_slab_system(dtype, device)

        def slab_direct(pos, chg, cell_in):
            return compute_slab_correction(
                pos,
                chg,
                cell_in,
                pbc,
                compute_forces=True,
                compute_charge_gradients=True,
                compute_virial=True,
            )

        eager_pos = positions.clone().requires_grad_(True)
        eager_chg = charges.clone().requires_grad_(True)
        eager_cell = cell.clone().requires_grad_(True)
        eager = slab_direct(eager_pos, eager_chg, eager_cell)
        eager_grads = torch.autograd.grad(
            eager[0].sum(), (eager_pos, eager_chg, eager_cell)
        )

        compiled_pos = positions.clone().requires_grad_(True)
        compiled_chg = charges.clone().requires_grad_(True)
        compiled_cell = cell.clone().requires_grad_(True)
        compiled = torch.compile(slab_direct)(
            compiled_pos,
            compiled_chg,
            compiled_cell,
        )
        compiled_grads = torch.autograd.grad(
            compiled[0].sum(), (compiled_pos, compiled_chg, compiled_cell)
        )

        for actual, expected in zip(compiled, eager, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
        for actual, expected in zip(compiled_grads, eager_grads, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


# ==============================================================================
# pbc=None handling
# ==============================================================================


class TestPbcNoneHandling:
    """slab_correction=True without pbc must raise a clear error."""

    def test_missing_pbc_raises(self, device):
        dtype = torch.float64

        positions, charges, cell, _ = _make_cscl_slab_system(dtype, device)
        nl, ptr, shifts = _build_neighbor_list(
            positions, cell, 5.0, [True, True, False], device
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
            [True, True, False],
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

    @pytest.mark.gpu
    @pytest.mark.parametrize("device", ["cuda"])
    @pytest.mark.parametrize("method", ["ewald", "pme"])
    def test_full_slab_wrapper_second_order_cuda_canary(self, device, method):
        """Non-slow CUDA canary for full Ewald/PME slab wrapper paths."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        dtype = torch.float64
        device = torch.device(device)
        positions, charges, cell, pbc, nl, ptr, shifts = _make_cscl_ewald_inputs(
            dtype, device
        )
        positions = positions.detach().clone().requires_grad_(True)
        charges = charges.detach().clone().requires_grad_(True)
        cell = cell.detach().clone().requires_grad_(True)

        def energy_fn(pos, chg, cell_in):
            if method == "ewald":
                return ewald_summation(
                    pos,
                    chg,
                    cell_in,
                    alpha=EWALD_ALPHA,
                    k_cutoff=EWALD_K_CUTOFF,
                    neighbor_list=nl,
                    neighbor_ptr=ptr,
                    neighbor_shifts=shifts,
                    pbc=pbc,
                    slab_correction=True,
                )
            return particle_mesh_ewald(
                pos,
                chg,
                cell_in,
                alpha=EWALD_ALPHA,
                mesh_dimensions=(8, 8, 8),
                neighbor_list=nl,
                neighbor_ptr=ptr,
                neighbor_shifts=shifts,
                pbc=pbc,
                slab_correction=True,
            )

        energy = energy_fn(positions, charges, cell).sum()
        grad_positions, grad_charges, grad_cell = torch.autograd.grad(
            energy,
            (positions, charges, cell),
            create_graph=True,
        )
        loss = (
            grad_positions.square().sum()
            + grad_charges.square().sum()
            + grad_cell.square().sum()
        )
        loss.backward()

        for tensor in (
            grad_positions,
            grad_charges,
            grad_cell,
            positions.grad,
            charges.grad,
            cell.grad,
        ):
            assert tensor is not None
            assert torch.isfinite(tensor).all()

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

    @pytest.mark.slow
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
            charge_grads,
            autograd_charge_grads,
            rtol=SLAB_CHARGE_GRAD_RTOL,
            atol=SLAB_CHARGE_GRAD_ATOL,
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
            positions @ deformation,
            charges,
            cell @ deformation,
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

        def outputs_by_name(outputs):
            outputs_tuple = outputs if isinstance(outputs, tuple) else (outputs,)
            return dict(zip(output_names, outputs_tuple, strict=True))

        real = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
        )
        reciprocal = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=mesh_dimensions,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
        )
        slab = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
        )
        real_by_name = outputs_by_name(real)
        reciprocal_by_name = outputs_by_name(reciprocal)
        slab_by_name = outputs_by_name(slab)
        expected_by_name = {
            name: real_by_name[name] + reciprocal_by_name[name] + slab_by_name[name]
            for name in output_names
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
        pbc_slab_mixed = torch.tensor(
            [[True, True, False], [True, True, True]],
            dtype=torch.bool,
            device=device,
        )
        nl, ptr, shifts = batch_cell_list(
            positions,
            cutoff=REAL_SPACE_CUTOFF,
            cell=cell,
            pbc=pbc_slab_mixed,
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

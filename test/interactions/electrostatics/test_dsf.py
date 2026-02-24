# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Unit tests for DSF (Damped Shifted Force) warp kernel launchers.

Tests cover:
- DSF pair potential correctness
- Self-energy correction
- Energy, forces, virial, charge gradients
- Both CSR and neighbor matrix formats
- PBC and non-PBC
- Batched calculations
- Alpha=0 (shifted-force bare Coulomb)
- Edge cases (empty neighbors, zero charges, cutoff enforcement)
- CPU/GPU consistency
- Regression tests with hardcoded values
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from nvalchemiops.interactions.electrostatics.dsf import (
    dsf_csr,
    dsf_matrix,
)

# Mathematical constants for reference computations
PI = math.pi
SQRT_PI = math.sqrt(PI)
TWO_OVER_SQRT_PI = 2.0 / SQRT_PI


def _dsf_pair_potential_ref(r, cutoff, alpha):
    """Reference DSF pair potential (excluding qi*qj) computed in Python."""
    if alpha > 0.0:
        erfc_r = math.erfc(alpha * r)
        erfc_rc = math.erfc(alpha * cutoff)
        exp_rc = math.exp(-(alpha**2) * cutoff**2)
    else:
        erfc_r = 1.0
        erfc_rc = 1.0
        exp_rc = 1.0

    v_shift = erfc_rc / cutoff
    force_shift = erfc_rc / cutoff**2 + TWO_OVER_SQRT_PI * alpha * exp_rc / cutoff
    return erfc_r / r - v_shift + force_shift * (r - cutoff)


def _dsf_self_energy_coeff_ref(cutoff, alpha):
    """Reference DSF self-energy coefficient computed in Python."""
    if alpha > 0.0:
        erfc_rc = math.erfc(alpha * cutoff)
    else:
        erfc_rc = 1.0
    return -(erfc_rc / (2.0 * cutoff) + alpha / SQRT_PI)


def _dsf_force_factor_ref(r, cutoff, alpha):
    """Reference DSF force factor (excluding qi*qj) computed in Python."""
    if alpha > 0.0:
        erfc_r = math.erfc(alpha * r)
        exp_r = math.exp(-(alpha**2) * r**2)
        erfc_rc = math.erfc(alpha * cutoff)
        exp_rc = math.exp(-(alpha**2) * cutoff**2)
    else:
        erfc_r = 1.0
        exp_r = 1.0
        erfc_rc = 1.0
        exp_rc = 1.0

    force_shift = erfc_rc / cutoff**2 + TWO_OVER_SQRT_PI * alpha * exp_rc / cutoff
    force_factor = erfc_r / r**2 + TWO_OVER_SQRT_PI * alpha * exp_r / r - force_shift
    return force_factor


# ==============================================================================
# Helpers
# ==============================================================================


def _make_warp_system(system, device):
    """Convert numpy system dict to warp arrays on given device."""
    wp_arrays = {}
    wp_arrays["positions"] = wp.array(
        system["positions"], dtype=wp.vec3d, device=device
    )
    wp_arrays["charges"] = wp.array(system["charges"], dtype=wp.float64, device=device)
    wp_arrays["idx_j"] = wp.array(system["idx_j"], dtype=wp.int32, device=device)
    wp_arrays["neighbor_ptr"] = wp.array(
        system["neighbor_ptr"], dtype=wp.int32, device=device
    )
    wp_arrays["batch_idx"] = wp.array(
        system.get("batch_idx", np.zeros(system["num_atoms"], dtype=np.int32)),
        dtype=wp.int32,
        device=device,
    )
    num_systems = system.get("num_systems", 1)
    wp_arrays["energy"] = wp.zeros(num_systems, dtype=wp.float64, device=device)
    wp_arrays["forces"] = wp.zeros(system["num_atoms"], dtype=wp.vec3d, device=device)
    wp_arrays["virial"] = wp.zeros(num_systems, dtype=wp.mat33d, device=device)
    wp_arrays["charge_grad"] = wp.zeros(
        system["num_atoms"], dtype=wp.float64, device=device
    )
    return wp_arrays


def _make_warp_matrix_system(system, device):
    """Convert numpy system dict (matrix format) to warp arrays on given device."""
    wp_arrays = {}
    wp_arrays["positions"] = wp.array(
        system["positions"], dtype=wp.vec3d, device=device
    )
    wp_arrays["charges"] = wp.array(system["charges"], dtype=wp.float64, device=device)
    wp_arrays["neighbor_matrix"] = wp.array(
        system["neighbor_matrix"], dtype=wp.int32, device=device
    )
    wp_arrays["batch_idx"] = wp.array(
        system.get("batch_idx", np.zeros(system["num_atoms"], dtype=np.int32)),
        dtype=wp.int32,
        device=device,
    )
    num_systems = system.get("num_systems", 1)
    wp_arrays["energy"] = wp.zeros(num_systems, dtype=wp.float64, device=device)
    wp_arrays["forces"] = wp.zeros(system["num_atoms"], dtype=wp.vec3d, device=device)
    wp_arrays["virial"] = wp.zeros(num_systems, dtype=wp.mat33d, device=device)
    wp_arrays["charge_grad"] = wp.zeros(
        system["num_atoms"], dtype=wp.float64, device=device
    )
    return wp_arrays


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def two_charge_system():
    """Two opposite charges along x-axis.

    - Atom 0 at origin, charge +1
    - Atom 1 at (3, 0, 0), charge -1
    Distance r = 3.0, full NL (both directions).
    """
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    charges = np.array([1.0, -1.0], dtype=np.float64)
    # Full NL: both directions
    idx_j = np.array([1, 0], dtype=np.int32)
    neighbor_ptr = np.array([0, 1, 2], dtype=np.int32)
    unit_shifts = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int32)
    # Matrix format
    neighbor_matrix = np.array([[1, 999], [0, 999]], dtype=np.int32)
    neighbor_shifts = np.array(
        [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]], dtype=np.int32
    )
    return {
        "positions": positions,
        "charges": charges,
        "idx_j": idx_j,
        "neighbor_ptr": neighbor_ptr,
        "unit_shifts": unit_shifts,
        "neighbor_matrix": neighbor_matrix,
        "neighbor_shifts": neighbor_shifts,
        "fill_value": 999,
        "num_atoms": 2,
        "distance": 3.0,
    }


@pytest.fixture(scope="session")
def like_charges_system():
    """Two positive charges along x-axis (repulsive)."""
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    charges = np.array([1.0, 1.0], dtype=np.float64)
    idx_j = np.array([1, 0], dtype=np.int32)
    neighbor_ptr = np.array([0, 1, 2], dtype=np.int32)
    unit_shifts = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int32)
    return {
        "positions": positions,
        "charges": charges,
        "idx_j": idx_j,
        "neighbor_ptr": neighbor_ptr,
        "unit_shifts": unit_shifts,
        "num_atoms": 2,
        "distance": 2.0,
    }


@pytest.fixture(scope="session")
def three_atom_system():
    """Three atoms along x-axis.

    - Atom 0 at (0,0,0), charge +1
    - Atom 1 at (2,0,0), charge -1
    - Atom 2 at (5,0,0), charge +0.5
    Full NL: all 6 directed edges.
    """
    positions = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64
    )
    charges = np.array([1.0, -1.0, 0.5], dtype=np.float64)
    # Full NL: 0->1, 0->2, 1->0, 1->2, 2->0, 2->1
    idx_j = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)
    neighbor_ptr = np.array([0, 2, 4, 6], dtype=np.int32)
    unit_shifts = np.zeros((6, 3), dtype=np.int32)
    return {
        "positions": positions,
        "charges": charges,
        "idx_j": idx_j,
        "neighbor_ptr": neighbor_ptr,
        "unit_shifts": unit_shifts,
        "num_atoms": 3,
    }


@pytest.fixture(scope="session")
def single_atom_system():
    """Single atom with no neighbors (tests self-energy only)."""
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    charges = np.array([2.0], dtype=np.float64)
    idx_j = np.array([], dtype=np.int32)
    neighbor_ptr = np.array([0, 0], dtype=np.int32)
    unit_shifts = np.zeros((0, 3), dtype=np.int32)
    return {
        "positions": positions,
        "charges": charges,
        "idx_j": idx_j,
        "neighbor_ptr": neighbor_ptr,
        "unit_shifts": unit_shifts,
        "num_atoms": 1,
    }


# ==============================================================================
# Test DSF Pair Potential
# ==============================================================================


class TestDSFPairPotential:
    """Test DSF pair potential correctness."""

    @pytest.mark.parametrize("alpha", [0.0, 0.1, 0.2, 0.5])
    def test_potential_zero_at_cutoff(self, alpha):
        """V(Rc) should be zero."""
        cutoff = 10.0
        V = _dsf_pair_potential_ref(cutoff, cutoff, alpha)
        assert V == pytest.approx(0.0, abs=1e-14), f"V(Rc)={V} for alpha={alpha}"

    @pytest.mark.parametrize("alpha", [0.0, 0.1, 0.2])
    def test_potential_continuous_near_cutoff(self, alpha):
        """V(r) should be continuous as r -> Rc."""
        cutoff = 10.0
        V_near = _dsf_pair_potential_ref(cutoff - 1e-8, cutoff, alpha)
        assert V_near == pytest.approx(0.0, abs=1e-6), f"V near cutoff = {V_near}"

    @pytest.mark.parametrize("alpha", [0.0, 0.1, 0.2, 0.5])
    def test_force_zero_at_cutoff(self, alpha):
        """Force factor A(Rc) should be zero."""
        cutoff = 10.0
        A = _dsf_force_factor_ref(cutoff, cutoff, alpha)
        assert A == pytest.approx(0.0, abs=1e-14), f"A(Rc)={A} for alpha={alpha}"

    def test_alpha_zero_simplification(self):
        """Alpha=0 should give shifted-force bare Coulomb."""
        cutoff = 10.0
        r = 3.0
        V = _dsf_pair_potential_ref(r, cutoff, 0.0)
        # For alpha=0: V(r) = 1/r - 2/Rc + r/Rc^2
        expected = 1.0 / r - 2.0 / cutoff + r / cutoff**2
        assert V == pytest.approx(expected, abs=1e-14)


# ==============================================================================
# Test DSF Energy (CSR Format)
# ==============================================================================


class TestDSFEnergyCsr:
    """Test DSF energy computation with CSR neighbor list."""

    def test_opposite_charges_negative_energy(self, two_charge_system, device):
        """Opposite charges should give negative pair energy."""
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2
        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
            compute_forces=False,
        )
        energy = w["energy"].numpy()[0]
        # Pair energy is negative (opposite charges), self-energy is always negative
        assert energy < 0.0

    def test_like_charges_positive_pair_energy(self, like_charges_system, device):
        """Like charges should give positive pair energy contribution."""
        sys = like_charges_system
        cutoff = 10.0
        alpha = 0.2
        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
            compute_forces=False,
        )
        energy = w["energy"].numpy()[0]
        # Pair energy is positive, self-energy is negative
        # For like charges close together, pair should dominate
        r = sys["distance"]
        v_pair = _dsf_pair_potential_ref(r, cutoff, alpha)
        pair_E = 1.0 * 1.0 * v_pair  # qi*qj*V
        self_E = 2.0 * _dsf_self_energy_coeff_ref(cutoff, alpha) * 1.0**2
        expected = pair_E + self_E  # 0.5*2*pair + 2*self
        assert energy == pytest.approx(expected, abs=1e-6)

    def test_energy_matches_reference(self, two_charge_system, device):
        """Energy should match Python reference computation."""
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2
        r = sys["distance"]

        v_pair = _dsf_pair_potential_ref(r, cutoff, alpha)
        self_coeff = _dsf_self_energy_coeff_ref(cutoff, alpha)
        # Total: 0.5 * (qi*qj + qj*qi) * V + self_i + self_j
        # = qi*qj*V + self_coeff*(qi^2 + qj^2)
        expected = 1.0 * (-1.0) * v_pair + self_coeff * (1.0**2 + (-1.0) ** 2)

        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
            compute_forces=False,
        )
        energy = w["energy"].numpy()[0]
        assert energy == pytest.approx(expected, abs=1e-6), (
            f"got {energy}, expected {expected}"
        )

    def test_cutoff_excludes_far_pairs(self, device):
        """Pairs beyond cutoff should not contribute."""
        positions = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float64)
        charges = np.array([1.0, -1.0], dtype=np.float64)
        idx_j = np.array([1, 0], dtype=np.int32)
        neighbor_ptr = np.array([0, 1, 2], dtype=np.int32)

        sys = {
            "positions": positions,
            "charges": charges,
            "idx_j": idx_j,
            "neighbor_ptr": neighbor_ptr,
            "num_atoms": 2,
        }
        cutoff = 10.0
        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=cutoff,
            alpha=0.2,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
            compute_forces=False,
        )
        energy = w["energy"].numpy()[0]
        # Only self-energy, no pair contribution
        self_coeff = _dsf_self_energy_coeff_ref(cutoff, 0.2)
        expected = self_coeff * (1.0**2 + (-1.0) ** 2)
        assert energy == pytest.approx(expected, abs=1e-6)


# ==============================================================================
# Test Self-Energy
# ==============================================================================


class TestDSFSelfEnergy:
    """Test DSF self-energy correction."""

    def test_single_atom_self_energy(self, single_atom_system, device):
        """Single atom with no neighbors should have only self-energy."""
        sys = single_atom_system
        cutoff = 10.0
        alpha = 0.2
        q = sys["charges"][0]
        self_coeff = _dsf_self_energy_coeff_ref(cutoff, alpha)
        expected = self_coeff * q**2

        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
            compute_forces=False,
        )
        energy = w["energy"].numpy()[0]
        # wp_erfc approximation has ~1.5e-7 error vs math.erfc
        assert energy == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("alpha", [0.0, 0.1, 0.2, 0.5])
    @pytest.mark.parametrize("cutoff", [5.0, 10.0, 20.0])
    def test_self_energy_always_negative(self, device, alpha, cutoff):
        """Self-energy coefficient is always negative."""
        coeff = _dsf_self_energy_coeff_ref(cutoff, alpha)
        assert coeff < 0.0, f"Self-energy coeff not negative: {coeff}"


# ==============================================================================
# Test DSF Forces
# ==============================================================================


class TestDSFForces:
    """Test DSF force computation."""

    def test_opposite_charges_attractive(self, two_charge_system, device):
        """Opposite charges should attract (forces point toward each other)."""
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2
        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
        )
        forces = w["forces"].numpy()
        # Atom 0 at origin, atom 1 at (3,0,0)
        # F on atom 0 should point toward atom 1 (positive x)
        # F on atom 1 should point toward atom 0 (negative x)
        assert forces[0, 0] > 0.0, "Atom 0 should be pulled toward atom 1"
        assert forces[1, 0] < 0.0, "Atom 1 should be pulled toward atom 0"

    def test_like_charges_repulsive(self, like_charges_system, device):
        """Like charges should repel (forces point away from each other)."""
        sys = like_charges_system
        cutoff = 10.0
        alpha = 0.2
        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
        )
        forces = w["forces"].numpy()
        # Atom 0 at origin, atom 1 at (2,0,0)
        # F on atom 0 should point away from atom 1 (negative x)
        assert forces[0, 0] < 0.0
        assert forces[1, 0] > 0.0

    def test_forces_match_reference(self, two_charge_system, device):
        """Forces should match Python reference computation."""
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2
        r = sys["distance"]

        A_force = _dsf_force_factor_ref(r, cutoff, alpha)
        qi, qj = sys["charges"]
        # Force on atom 0: F_0 = qi*qj * A(r)/r * r_01 where r_01 = pos[0] - pos[1]
        # r_01 = (-3, 0, 0), so Fx = qi*qj * A/r * (-r) = (-1)*A/3*(-3) = A > 0
        expected_Fx_0 = qi * qj * A_force / r * (-r)

        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
        )
        forces = w["forces"].numpy()
        assert forces[0, 0] == pytest.approx(expected_Fx_0, abs=1e-6)

    def test_force_zero_components(self, two_charge_system, device):
        """Forces should be zero in y and z for atoms along x-axis."""
        sys = two_charge_system
        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=10.0,
            alpha=0.2,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
        )
        forces = w["forces"].numpy()
        assert forces[0, 1] == pytest.approx(0.0, abs=1e-14)
        assert forces[0, 2] == pytest.approx(0.0, abs=1e-14)


# ==============================================================================
# Test Charge Gradients
# ==============================================================================


class TestDSFChargeGrad:
    """Test DSF charge gradient (dE/dq) computation."""

    def test_charge_grad_single_atom(self, single_atom_system, device):
        """Single atom charge grad = derivative of self-energy."""
        sys = single_atom_system
        cutoff = 10.0
        alpha = 0.2
        q = sys["charges"][0]
        self_coeff = _dsf_self_energy_coeff_ref(cutoff, alpha)
        expected = 2.0 * self_coeff * q

        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
            compute_forces=False,
            compute_charge_grad=True,
        )
        cg = w["charge_grad"].numpy()[0]
        # wp_erfc approximation has ~1.5e-7 error vs math.erfc
        assert cg == pytest.approx(expected, abs=1e-6)

    def test_charge_grad_finite_difference(self, two_charge_system, device):
        """Charge gradient should match finite difference of energy."""
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2
        delta = 1e-5

        # Compute dE/dq_0 via finite difference
        energies = []
        for dq in [-delta, delta]:
            charges_mod = sys["charges"].copy()
            charges_mod[0] += dq
            mod_sys = dict(sys)
            mod_sys["charges"] = charges_mod
            w = _make_warp_system(mod_sys, device)
            dsf_csr(
                positions=w["positions"],
                charges=w["charges"],
                idx_j=w["idx_j"],
                neighbor_ptr=w["neighbor_ptr"],
                cutoff=cutoff,
                alpha=alpha,
                energy=w["energy"],
                forces=w["forces"],
                virial=w["virial"],
                charge_grad=w["charge_grad"],
                device=device,
                batch_idx=w["batch_idx"],
                compute_forces=False,
            )
            energies.append(w["energy"].numpy()[0])
        fd_grad = (energies[1] - energies[0]) / (2 * delta)

        # Compute analytical charge grad
        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
            compute_forces=False,
            compute_charge_grad=True,
        )
        cg = w["charge_grad"].numpy()[0]

        assert cg == pytest.approx(fd_grad, abs=1e-4), f"analytical={cg}, fd={fd_grad}"


# ==============================================================================
# Test Alpha=0
# ==============================================================================


class TestDSFAlphaZero:
    """Test alpha=0 (shifted-force bare Coulomb)."""

    def test_energy_alpha_zero(self, two_charge_system, device):
        """Alpha=0 should give shifted-force Coulomb energy."""
        sys = two_charge_system
        cutoff = 10.0
        r = sys["distance"]

        v_pair = _dsf_pair_potential_ref(r, cutoff, 0.0)
        self_coeff = _dsf_self_energy_coeff_ref(cutoff, 0.0)
        qi, qj = sys["charges"]
        expected = qi * qj * v_pair + self_coeff * (qi**2 + qj**2)

        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=cutoff,
            alpha=0.0,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
            compute_forces=False,
        )
        energy = w["energy"].numpy()[0]
        assert energy == pytest.approx(expected, abs=1e-6)


# ==============================================================================
# Test Neighbor Matrix Format
# ==============================================================================


class TestDSFMatrix:
    """Test DSF with neighbor matrix format."""

    def test_matrix_energy_matches_csr(self, two_charge_system, device):
        """Matrix and CSR formats should give identical energy."""
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2

        # CSR format
        w_csr = _make_warp_system(sys, device)
        dsf_csr(
            positions=w_csr["positions"],
            charges=w_csr["charges"],
            idx_j=w_csr["idx_j"],
            neighbor_ptr=w_csr["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w_csr["energy"],
            forces=w_csr["forces"],
            virial=w_csr["virial"],
            charge_grad=w_csr["charge_grad"],
            device=device,
            batch_idx=w_csr["batch_idx"],
            compute_forces=False,
        )

        # Matrix format
        w_mat = _make_warp_matrix_system(sys, device)
        dsf_matrix(
            positions=w_mat["positions"],
            charges=w_mat["charges"],
            neighbor_matrix=w_mat["neighbor_matrix"],
            cutoff=cutoff,
            alpha=alpha,
            fill_value=sys["fill_value"],
            energy=w_mat["energy"],
            forces=w_mat["forces"],
            virial=w_mat["virial"],
            charge_grad=w_mat["charge_grad"],
            device=device,
            batch_idx=w_mat["batch_idx"],
            compute_forces=False,
        )

        e_csr = w_csr["energy"].numpy()[0]
        e_mat = w_mat["energy"].numpy()[0]
        assert e_csr == pytest.approx(e_mat, abs=1e-10), f"CSR={e_csr}, Matrix={e_mat}"


# ==============================================================================
# Test Batched Calculations
# ==============================================================================


class TestDSFBatch:
    """Test batched DSF calculations."""

    def test_two_identical_systems(self, two_charge_system, device):
        """Two identical systems should give 2x the single system energy."""
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2

        # Single system
        w_single = _make_warp_system(sys, device)
        dsf_csr(
            positions=w_single["positions"],
            charges=w_single["charges"],
            idx_j=w_single["idx_j"],
            neighbor_ptr=w_single["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w_single["energy"],
            forces=w_single["forces"],
            virial=w_single["virial"],
            charge_grad=w_single["charge_grad"],
            device=device,
            batch_idx=w_single["batch_idx"],
            compute_forces=False,
        )
        single_energy = w_single["energy"].numpy()[0]

        # Batched: two copies
        positions_batch = np.concatenate(
            [sys["positions"], sys["positions"] + [50.0, 0.0, 0.0]]
        )
        charges_batch = np.concatenate([sys["charges"], sys["charges"]])
        # Offset idx_j for second system
        idx_j_2 = sys["idx_j"] + sys["num_atoms"]
        idx_j_batch = np.concatenate([sys["idx_j"], idx_j_2])
        # Offset neighbor_ptr for second system
        offset = sys["neighbor_ptr"][-1]
        neighbor_ptr_2 = sys["neighbor_ptr"][1:] + offset
        neighbor_ptr_batch = np.concatenate([sys["neighbor_ptr"], neighbor_ptr_2])
        batch_idx_batch = np.array([0, 0, 1, 1], dtype=np.int32)

        batch_sys = {
            "positions": positions_batch,
            "charges": charges_batch,
            "idx_j": idx_j_batch,
            "neighbor_ptr": neighbor_ptr_batch,
            "batch_idx": batch_idx_batch,
            "num_atoms": 4,
            "num_systems": 2,
        }
        w_batch = _make_warp_system(batch_sys, device)
        dsf_csr(
            positions=w_batch["positions"],
            charges=w_batch["charges"],
            idx_j=w_batch["idx_j"],
            neighbor_ptr=w_batch["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w_batch["energy"],
            forces=w_batch["forces"],
            virial=w_batch["virial"],
            charge_grad=w_batch["charge_grad"],
            device=device,
            batch_idx=w_batch["batch_idx"],
            compute_forces=False,
        )
        batch_energies = w_batch["energy"].numpy()
        assert batch_energies[0] == pytest.approx(single_energy, abs=1e-10)
        assert batch_energies[1] == pytest.approx(single_energy, abs=1e-10)


# ==============================================================================
# Test PBC
# ==============================================================================


class TestDSFPBC:
    """Test DSF with periodic boundary conditions."""

    def test_pbc_matches_nonpbc_for_zero_shifts(self, two_charge_system, device):
        """PBC with zero shifts should match non-PBC result."""
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2
        cell = np.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        )

        # Non-PBC
        w_np = _make_warp_system(sys, device)
        dsf_csr(
            positions=w_np["positions"],
            charges=w_np["charges"],
            idx_j=w_np["idx_j"],
            neighbor_ptr=w_np["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w_np["energy"],
            forces=w_np["forces"],
            virial=w_np["virial"],
            charge_grad=w_np["charge_grad"],
            device=device,
            batch_idx=w_np["batch_idx"],
        )

        # PBC
        w_pbc = _make_warp_system(sys, device)
        cell_wp = wp.array(cell, dtype=wp.mat33d, device=device)
        unit_shifts_wp = wp.array(sys["unit_shifts"], dtype=wp.vec3i, device=device)
        dsf_csr(
            positions=w_pbc["positions"],
            charges=w_pbc["charges"],
            cell=cell_wp,
            idx_j=w_pbc["idx_j"],
            neighbor_ptr=w_pbc["neighbor_ptr"],
            unit_shifts=unit_shifts_wp,
            cutoff=cutoff,
            alpha=alpha,
            energy=w_pbc["energy"],
            forces=w_pbc["forces"],
            virial=w_pbc["virial"],
            charge_grad=w_pbc["charge_grad"],
            device=device,
            batch_idx=w_pbc["batch_idx"],
        )

        e_np = w_np["energy"].numpy()[0]
        e_pbc = w_pbc["energy"].numpy()[0]
        assert e_np == pytest.approx(e_pbc, abs=1e-10)

        f_np = w_np["forces"].numpy()
        f_pbc = w_pbc["forces"].numpy()
        np.testing.assert_allclose(f_np, f_pbc, atol=1e-10)


# ==============================================================================
# Test Edge Cases
# ==============================================================================


class TestDSFEdgeCases:
    """Test DSF edge cases."""

    def test_zero_charges(self, device):
        """All-zero charges should give zero energy."""
        positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
        charges = np.array([0.0, 0.0], dtype=np.float64)
        idx_j = np.array([1, 0], dtype=np.int32)
        neighbor_ptr = np.array([0, 1, 2], dtype=np.int32)

        sys = {
            "positions": positions,
            "charges": charges,
            "idx_j": idx_j,
            "neighbor_ptr": neighbor_ptr,
            "num_atoms": 2,
        }
        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=10.0,
            alpha=0.2,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
        )
        energy = w["energy"].numpy()[0]
        assert energy == pytest.approx(0.0, abs=1e-14)

    def test_empty_neighbor_list(self, device):
        """Empty neighbor list should still compute self-energy."""
        positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
        charges = np.array([1.0, -1.0], dtype=np.float64)
        idx_j = np.array([], dtype=np.int32)
        neighbor_ptr = np.array([0, 0, 0], dtype=np.int32)

        sys = {
            "positions": positions,
            "charges": charges,
            "idx_j": idx_j,
            "neighbor_ptr": neighbor_ptr,
            "num_atoms": 2,
        }
        cutoff = 10.0
        alpha = 0.2
        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
            compute_forces=False,
        )
        energy = w["energy"].numpy()[0]
        self_coeff = _dsf_self_energy_coeff_ref(cutoff, alpha)
        expected = self_coeff * (1.0**2 + (-1.0) ** 2)
        # wp_erfc approximation has ~1.5e-7 error vs math.erfc
        assert energy == pytest.approx(expected, abs=1e-6)


# ==============================================================================
# Test CPU/GPU Consistency
# ==============================================================================


class TestCPUGPUConsistency:
    """Test that CPU and GPU give consistent results."""

    def test_energy_consistency(self, two_charge_system):
        """CPU and GPU energy should match."""
        if not wp.is_cuda_available():
            pytest.skip("CUDA not available")

        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2

        results = {}
        for dev in ["cpu", "cuda:0"]:
            w = _make_warp_system(sys, dev)
            dsf_csr(
                positions=w["positions"],
                charges=w["charges"],
                idx_j=w["idx_j"],
                neighbor_ptr=w["neighbor_ptr"],
                cutoff=cutoff,
                alpha=alpha,
                energy=w["energy"],
                forces=w["forces"],
                virial=w["virial"],
                charge_grad=w["charge_grad"],
                device=dev,
                batch_idx=w["batch_idx"],
                compute_charge_grad=True,
            )
            results[dev] = {
                "energy": w["energy"].numpy()[0],
                "forces": w["forces"].numpy(),
                "charge_grad": w["charge_grad"].numpy(),
            }

        assert results["cpu"]["energy"] == pytest.approx(
            results["cuda:0"]["energy"], abs=1e-10
        )
        np.testing.assert_allclose(
            results["cpu"]["forces"], results["cuda:0"]["forces"], atol=1e-10
        )
        np.testing.assert_allclose(
            results["cpu"]["charge_grad"],
            results["cuda:0"]["charge_grad"],
            atol=1e-10,
        )


# ==============================================================================
# Regression Tests
# ==============================================================================


class TestDSFRegression:
    """Hardcoded regression tests for known configurations."""

    def test_regression_two_atoms_damped(self, device):
        """Regression: two opposite charges at r=3, alpha=0.2, Rc=10."""
        positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
        charges = np.array([1.0, -1.0], dtype=np.float64)
        idx_j = np.array([1, 0], dtype=np.int32)
        neighbor_ptr = np.array([0, 1, 2], dtype=np.int32)
        cutoff = 10.0
        alpha = 0.2

        # Compute expected values analytically
        r = 3.0
        v_pair = _dsf_pair_potential_ref(r, cutoff, alpha)
        self_coeff = _dsf_self_energy_coeff_ref(cutoff, alpha)
        expected_energy = 1.0 * (-1.0) * v_pair + self_coeff * 2.0

        sys = {
            "positions": positions,
            "charges": charges,
            "idx_j": idx_j,
            "neighbor_ptr": neighbor_ptr,
            "num_atoms": 2,
        }
        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
        )
        energy = w["energy"].numpy()[0]
        assert energy == pytest.approx(expected_energy, abs=1e-6), (
            f"got {energy}, expected {expected_energy}"
        )


# ==============================================================================
# Test Force Conservation (Newton's 3rd Law)
# ==============================================================================


class TestDSFForceConservation:
    """Test that total force on an isolated system is zero."""

    def test_two_atom_force_sum_zero(self, two_charge_system, device):
        """Sum of forces on two atoms should be zero."""
        sys = two_charge_system
        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=10.0,
            alpha=0.2,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
        )
        forces = w["forces"].numpy()
        total_force = forces.sum(axis=0)
        np.testing.assert_allclose(total_force, 0.0, atol=1e-10)

    def test_three_atom_force_sum_zero(self, three_atom_system, device):
        """Sum of forces on three atoms should be zero."""
        sys = three_atom_system
        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=10.0,
            alpha=0.2,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
        )
        forces = w["forces"].numpy()
        total_force = forces.sum(axis=0)
        np.testing.assert_allclose(total_force, 0.0, atol=1e-10)


# ==============================================================================
# Test Three-Atom Forces
# ==============================================================================


class TestDSFThreeAtomForces:
    """Test forces on three-atom system against reference."""

    def test_three_atom_forces_match_reference(self, three_atom_system, device):
        """Each atom's force should match pair-sum of DSF force contributions."""
        sys = three_atom_system
        cutoff = 10.0
        alpha = 0.2
        positions = sys["positions"]
        charges = sys["charges"]

        # Compute expected forces analytically from reference
        # Full NL: each atom i has forces from all other atoms j
        expected_forces = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                r_ij = positions[i] - positions[j]
                r = np.linalg.norm(r_ij)
                if r >= cutoff:
                    continue
                ff = _dsf_force_factor_ref(r, cutoff, alpha)
                expected_forces[i] += charges[i] * charges[j] * ff / r * r_ij

        w = _make_warp_system(sys, device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
        )
        forces = w["forces"].numpy()
        np.testing.assert_allclose(forces, expected_forces, atol=1e-6)


# ==============================================================================
# Test Virial
# ==============================================================================


class TestDSFVirial:
    """Test DSF virial tensor computation."""

    def test_virial_nonzero_for_pbc(self, two_charge_system, device):
        """Virial should be non-zero for interacting PBC system."""
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2
        cell = np.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        )
        w = _make_warp_system(sys, device)
        cell_wp = wp.array(cell, dtype=wp.mat33d, device=device)
        unit_shifts_wp = wp.array(sys["unit_shifts"], dtype=wp.vec3i, device=device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            cell=cell_wp,
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            unit_shifts=unit_shifts_wp,
            cutoff=cutoff,
            alpha=alpha,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
            compute_virial=True,
        )
        virial = w["virial"].numpy()[0]
        # For atoms along x-axis, virial should have nonzero xx component
        assert abs(virial[0, 0]) > 1e-10, "Virial xx should be nonzero"
        # Off-axis components should be zero (1D configuration)
        assert virial[0, 1] == pytest.approx(0.0, abs=1e-10)
        assert virial[0, 2] == pytest.approx(0.0, abs=1e-10)

    def test_virial_matches_force_times_displacement(self, two_charge_system, device):
        """Virial = -0.5 * sum_pairs F_ij outer r_ij for full NL."""
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2
        cell = np.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        )
        w = _make_warp_system(sys, device)
        cell_wp = wp.array(cell, dtype=wp.mat33d, device=device)
        unit_shifts_wp = wp.array(sys["unit_shifts"], dtype=wp.vec3i, device=device)
        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            cell=cell_wp,
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            unit_shifts=unit_shifts_wp,
            cutoff=cutoff,
            alpha=alpha,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
            compute_virial=True,
        )
        virial = w["virial"].numpy()[0]

        # Compute reference virial: -0.5 * sum over all directed pairs (F_ij outer r_ij)
        positions = sys["positions"]
        charges = sys["charges"]
        ref_virial = np.zeros((3, 3))
        for i in range(2):
            for j in range(2):
                if i == j:
                    continue
                r_ij_vec = positions[i] - positions[j]
                r = np.linalg.norm(r_ij_vec)
                if r >= cutoff:
                    continue
                ff = _dsf_force_factor_ref(r, cutoff, alpha)
                f_ij = charges[i] * charges[j] * ff / r * r_ij_vec
                ref_virial += np.outer(f_ij, r_ij_vec)
        ref_virial *= -0.5

        np.testing.assert_allclose(virial, ref_virial, atol=1e-6)


# ==============================================================================
# Test PBC with Non-Zero Shifts
# ==============================================================================


class TestDSFPBCNonZeroShifts:
    """Test DSF PBC with actual periodic image interactions."""

    def test_pbc_nonzero_shift_energy(self, device):
        """Atoms across periodic boundary should interact via shift."""
        # Two atoms: one at (0.5, 0, 0), one at (9.5, 0, 0) in a 10 A box
        # Direct distance = 9.0, but via PBC shift = 1.0
        positions = np.array([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]], dtype=np.float64)
        charges = np.array([1.0, -1.0], dtype=np.float64)
        cell = np.array(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=np.float64,
        )
        # Full NL with PBC shifts: atom 0's neighbor is atom 1 with shift [-1,0,0]
        # (image of atom 1 at 9.5 - 10 = -0.5, distance to atom 0 at 0.5 is 1.0)
        # atom 1's neighbor is atom 0 with shift [1,0,0]
        # (image of atom 0 at 0.5 + 10 = 10.5, distance to atom 1 at 9.5 is 1.0)
        idx_j = np.array([1, 0], dtype=np.int32)
        neighbor_ptr = np.array([0, 1, 2], dtype=np.int32)
        unit_shifts = np.array([[-1, 0, 0], [1, 0, 0]], dtype=np.int32)
        batch_idx = np.zeros(2, dtype=np.int32)
        cutoff = 5.0
        alpha = 0.2

        # The PBC-corrected distance is 1.0 (not 9.0)
        pbc_distance = 1.0

        w = {}
        w["positions"] = wp.array(positions, dtype=wp.vec3d, device=device)
        w["charges"] = wp.array(charges, dtype=wp.float64, device=device)
        w["idx_j"] = wp.array(idx_j, dtype=wp.int32, device=device)
        w["neighbor_ptr"] = wp.array(neighbor_ptr, dtype=wp.int32, device=device)
        w["batch_idx"] = wp.array(batch_idx, dtype=wp.int32, device=device)
        w["energy"] = wp.zeros(1, dtype=wp.float64, device=device)
        w["forces"] = wp.zeros(2, dtype=wp.vec3d, device=device)
        w["virial"] = wp.zeros(1, dtype=wp.mat33d, device=device)
        w["charge_grad"] = wp.zeros(2, dtype=wp.float64, device=device)

        cell_wp = wp.array(cell, dtype=wp.mat33d, device=device)
        unit_shifts_wp = wp.array(unit_shifts, dtype=wp.vec3i, device=device)

        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            cell=cell_wp,
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            unit_shifts=unit_shifts_wp,
            cutoff=cutoff,
            alpha=alpha,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
        )
        energy = w["energy"].numpy()[0]

        # Expected: pair at distance 1.0
        v_pair = _dsf_pair_potential_ref(pbc_distance, cutoff, alpha)
        self_coeff = _dsf_self_energy_coeff_ref(cutoff, alpha)
        expected = 1.0 * (-1.0) * v_pair + self_coeff * (1.0**2 + 1.0**2)
        assert energy == pytest.approx(expected, abs=1e-6), (
            f"PBC energy {energy} != expected {expected}"
        )

    def test_pbc_nonzero_shift_forces(self, device):
        """Forces should reflect PBC-corrected distances."""
        positions = np.array([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]], dtype=np.float64)
        charges = np.array([1.0, -1.0], dtype=np.float64)
        cell = np.array(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=np.float64,
        )
        idx_j = np.array([1, 0], dtype=np.int32)
        neighbor_ptr = np.array([0, 1, 2], dtype=np.int32)
        unit_shifts = np.array([[-1, 0, 0], [1, 0, 0]], dtype=np.int32)
        batch_idx = np.zeros(2, dtype=np.int32)
        cutoff = 5.0
        alpha = 0.2

        w = {}
        w["positions"] = wp.array(positions, dtype=wp.vec3d, device=device)
        w["charges"] = wp.array(charges, dtype=wp.float64, device=device)
        w["idx_j"] = wp.array(idx_j, dtype=wp.int32, device=device)
        w["neighbor_ptr"] = wp.array(neighbor_ptr, dtype=wp.int32, device=device)
        w["batch_idx"] = wp.array(batch_idx, dtype=wp.int32, device=device)
        w["energy"] = wp.zeros(1, dtype=wp.float64, device=device)
        w["forces"] = wp.zeros(2, dtype=wp.vec3d, device=device)
        w["virial"] = wp.zeros(1, dtype=wp.mat33d, device=device)
        w["charge_grad"] = wp.zeros(2, dtype=wp.float64, device=device)

        cell_wp = wp.array(cell, dtype=wp.mat33d, device=device)
        unit_shifts_wp = wp.array(unit_shifts, dtype=wp.vec3i, device=device)

        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            cell=cell_wp,
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            unit_shifts=unit_shifts_wp,
            cutoff=cutoff,
            alpha=alpha,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
        )
        forces = w["forces"].numpy()
        # Force sum should be zero (Newton's 3rd law)
        np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-10)
        # Opposite charges attract: atom 0 is attracted toward the PBC image of
        # atom 1 at 9.5 - 10 = -0.5 (negative x direction from atom 0 at 0.5).
        assert forces[0, 0] < 0.0, "Atom 0 attracted toward -x PBC image of atom 1"


# ==============================================================================
# Test Float32 Support
# ==============================================================================


class TestDSFFloat32:
    """Test DSF kernels with float32 precision."""

    def test_float32_energy(self, two_charge_system, device):
        """Float32 energy should be close to float64 reference."""
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2

        # Float64 reference
        r = sys["distance"]
        v_pair = _dsf_pair_potential_ref(r, cutoff, alpha)
        self_coeff = _dsf_self_energy_coeff_ref(cutoff, alpha)
        expected = 1.0 * (-1.0) * v_pair + self_coeff * (1.0**2 + (-1.0) ** 2)

        # Float32 computation
        positions = wp.array(
            sys["positions"].astype(np.float32), dtype=wp.vec3f, device=device
        )
        charges = wp.array(
            sys["charges"].astype(np.float32), dtype=wp.float32, device=device
        )
        idx_j = wp.array(sys["idx_j"], dtype=wp.int32, device=device)
        neighbor_ptr = wp.array(sys["neighbor_ptr"], dtype=wp.int32, device=device)
        batch_idx = wp.zeros(2, dtype=wp.int32, device=device)
        energy = wp.zeros(1, dtype=wp.float64, device=device)
        forces = wp.zeros(2, dtype=wp.vec3f, device=device)
        virial = wp.zeros(1, dtype=wp.mat33f, device=device)
        charge_grad = wp.zeros(2, dtype=wp.float32, device=device)

        dsf_csr(
            positions=positions,
            charges=charges,
            idx_j=idx_j,
            neighbor_ptr=neighbor_ptr,
            cutoff=cutoff,
            alpha=alpha,
            energy=energy,
            forces=forces,
            virial=virial,
            charge_grad=charge_grad,
            device=device,
            batch_idx=batch_idx,
            compute_forces=False,
        )
        energy_val = energy.numpy()[0]
        # Float32 has ~1e-7 relative error; allow wider tolerance
        assert energy_val == pytest.approx(expected, abs=1e-4)

    def test_float32_forces(self, two_charge_system, device):
        """Float32 forces should be finite and have correct sign."""
        sys = two_charge_system
        positions = wp.array(
            sys["positions"].astype(np.float32), dtype=wp.vec3f, device=device
        )
        charges = wp.array(
            sys["charges"].astype(np.float32), dtype=wp.float32, device=device
        )
        idx_j = wp.array(sys["idx_j"], dtype=wp.int32, device=device)
        neighbor_ptr = wp.array(sys["neighbor_ptr"], dtype=wp.int32, device=device)
        batch_idx = wp.zeros(2, dtype=wp.int32, device=device)
        energy = wp.zeros(1, dtype=wp.float64, device=device)
        forces = wp.zeros(2, dtype=wp.vec3f, device=device)
        virial = wp.zeros(1, dtype=wp.mat33f, device=device)
        charge_grad = wp.zeros(2, dtype=wp.float32, device=device)

        dsf_csr(
            positions=positions,
            charges=charges,
            idx_j=idx_j,
            neighbor_ptr=neighbor_ptr,
            cutoff=10.0,
            alpha=0.2,
            energy=energy,
            forces=forces,
            virial=virial,
            charge_grad=charge_grad,
            device=device,
            batch_idx=batch_idx,
        )
        f = forces.numpy()
        assert np.all(np.isfinite(f))
        # Opposite charges attract
        assert f[0, 0] > 0.0
        assert f[1, 0] < 0.0

    def test_float32_charge_grad(self, two_charge_system, device):
        """Float32 charge gradients should be finite."""
        sys = two_charge_system
        positions = wp.array(
            sys["positions"].astype(np.float32), dtype=wp.vec3f, device=device
        )
        charges = wp.array(
            sys["charges"].astype(np.float32), dtype=wp.float32, device=device
        )
        idx_j = wp.array(sys["idx_j"], dtype=wp.int32, device=device)
        neighbor_ptr = wp.array(sys["neighbor_ptr"], dtype=wp.int32, device=device)
        batch_idx = wp.zeros(2, dtype=wp.int32, device=device)
        energy = wp.zeros(1, dtype=wp.float64, device=device)
        forces = wp.zeros(2, dtype=wp.vec3f, device=device)
        virial = wp.zeros(1, dtype=wp.mat33f, device=device)
        charge_grad = wp.zeros(2, dtype=wp.float32, device=device)

        dsf_csr(
            positions=positions,
            charges=charges,
            idx_j=idx_j,
            neighbor_ptr=neighbor_ptr,
            cutoff=10.0,
            alpha=0.2,
            energy=energy,
            forces=forces,
            virial=virial,
            charge_grad=charge_grad,
            device=device,
            batch_idx=batch_idx,
            compute_forces=False,
            compute_charge_grad=True,
        )
        cg = charge_grad.numpy()
        assert np.all(np.isfinite(cg))
        assert not np.all(cg == 0.0), "Charge gradients should be nonzero"


# ==============================================================================
# Test Matrix Format Extensions
# ==============================================================================


class TestDSFMatrixExtended:
    """Extended matrix format tests: forces, PBC, multi-atom."""

    def test_matrix_forces_match_csr(self, two_charge_system, device):
        """Matrix format forces should match CSR format forces."""
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2

        # CSR
        w_csr = _make_warp_system(sys, device)
        dsf_csr(
            positions=w_csr["positions"],
            charges=w_csr["charges"],
            idx_j=w_csr["idx_j"],
            neighbor_ptr=w_csr["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w_csr["energy"],
            forces=w_csr["forces"],
            virial=w_csr["virial"],
            charge_grad=w_csr["charge_grad"],
            device=device,
            batch_idx=w_csr["batch_idx"],
        )

        # Matrix
        w_mat = _make_warp_matrix_system(sys, device)
        dsf_matrix(
            positions=w_mat["positions"],
            charges=w_mat["charges"],
            neighbor_matrix=w_mat["neighbor_matrix"],
            cutoff=cutoff,
            alpha=alpha,
            fill_value=sys["fill_value"],
            energy=w_mat["energy"],
            forces=w_mat["forces"],
            virial=w_mat["virial"],
            charge_grad=w_mat["charge_grad"],
            device=device,
            batch_idx=w_mat["batch_idx"],
        )

        f_csr = w_csr["forces"].numpy()
        f_mat = w_mat["forces"].numpy()
        np.testing.assert_allclose(f_csr, f_mat, atol=1e-10)

    def test_matrix_pbc_energy_matches_csr_pbc(self, two_charge_system, device):
        """Matrix PBC energy should match CSR PBC energy."""
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2
        cell = np.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        )

        # CSR PBC
        w_csr = _make_warp_system(sys, device)
        cell_wp = wp.array(cell, dtype=wp.mat33d, device=device)
        unit_shifts_wp = wp.array(sys["unit_shifts"], dtype=wp.vec3i, device=device)
        dsf_csr(
            positions=w_csr["positions"],
            charges=w_csr["charges"],
            cell=cell_wp,
            idx_j=w_csr["idx_j"],
            neighbor_ptr=w_csr["neighbor_ptr"],
            unit_shifts=unit_shifts_wp,
            cutoff=cutoff,
            alpha=alpha,
            energy=w_csr["energy"],
            forces=w_csr["forces"],
            virial=w_csr["virial"],
            charge_grad=w_csr["charge_grad"],
            device=device,
            batch_idx=w_csr["batch_idx"],
            compute_forces=False,
        )

        # Matrix PBC
        w_mat = _make_warp_matrix_system(sys, device)
        neighbor_shifts_wp = wp.array(
            sys["neighbor_shifts"], dtype=wp.vec3i, device=device
        )
        dsf_matrix(
            positions=w_mat["positions"],
            charges=w_mat["charges"],
            neighbor_matrix=w_mat["neighbor_matrix"],
            cutoff=cutoff,
            alpha=alpha,
            fill_value=sys["fill_value"],
            energy=w_mat["energy"],
            forces=w_mat["forces"],
            virial=w_mat["virial"],
            charge_grad=w_mat["charge_grad"],
            cell=cell_wp,
            neighbor_matrix_shifts=neighbor_shifts_wp,
            device=device,
            batch_idx=w_mat["batch_idx"],
            compute_forces=False,
        )

        e_csr = w_csr["energy"].numpy()[0]
        e_mat = w_mat["energy"].numpy()[0]
        assert e_csr == pytest.approx(e_mat, abs=1e-10)

    def test_three_atom_matrix_energy(self, three_atom_system, device):
        """Three-atom matrix format energy should match CSR format."""
        sys = three_atom_system
        cutoff = 10.0
        alpha = 0.2

        # CSR
        w_csr = _make_warp_system(sys, device)
        dsf_csr(
            positions=w_csr["positions"],
            charges=w_csr["charges"],
            idx_j=w_csr["idx_j"],
            neighbor_ptr=w_csr["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w_csr["energy"],
            forces=w_csr["forces"],
            virial=w_csr["virial"],
            charge_grad=w_csr["charge_grad"],
            device=device,
            batch_idx=w_csr["batch_idx"],
            compute_forces=False,
        )

        # Build matrix format for 3 atoms: each has 2 neighbors (full NL)
        neighbor_matrix = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32)
        mat_sys = {
            "positions": sys["positions"],
            "charges": sys["charges"],
            "neighbor_matrix": neighbor_matrix,
            "num_atoms": 3,
        }
        w_mat = _make_warp_matrix_system(mat_sys, device)
        dsf_matrix(
            positions=w_mat["positions"],
            charges=w_mat["charges"],
            neighbor_matrix=w_mat["neighbor_matrix"],
            cutoff=cutoff,
            alpha=alpha,
            fill_value=3,
            energy=w_mat["energy"],
            forces=w_mat["forces"],
            virial=w_mat["virial"],
            charge_grad=w_mat["charge_grad"],
            device=device,
            batch_idx=w_mat["batch_idx"],
            compute_forces=False,
        )

        e_csr = w_csr["energy"].numpy()[0]
        e_mat = w_mat["energy"].numpy()[0]
        assert e_csr == pytest.approx(e_mat, abs=1e-10)


# ==============================================================================
# Test Batched Forces and Charge Gradients
# ==============================================================================


class TestDSFBatchExtended:
    """Extended batch tests: forces and charge gradients."""

    def test_batched_forces_match_individual(self, two_charge_system, device):
        """Batched per-atom forces should match individual system forces."""
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2

        # Single system
        w_single = _make_warp_system(sys, device)
        dsf_csr(
            positions=w_single["positions"],
            charges=w_single["charges"],
            idx_j=w_single["idx_j"],
            neighbor_ptr=w_single["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w_single["energy"],
            forces=w_single["forces"],
            virial=w_single["virial"],
            charge_grad=w_single["charge_grad"],
            device=device,
            batch_idx=w_single["batch_idx"],
        )
        single_forces = w_single["forces"].numpy()

        # Batched: two copies
        positions_batch = np.concatenate(
            [sys["positions"], sys["positions"] + [50.0, 0.0, 0.0]]
        )
        charges_batch = np.concatenate([sys["charges"], sys["charges"]])
        idx_j_2 = sys["idx_j"] + sys["num_atoms"]
        idx_j_batch = np.concatenate([sys["idx_j"], idx_j_2])
        offset = sys["neighbor_ptr"][-1]
        neighbor_ptr_2 = sys["neighbor_ptr"][1:] + offset
        neighbor_ptr_batch = np.concatenate([sys["neighbor_ptr"], neighbor_ptr_2])
        batch_idx_batch = np.array([0, 0, 1, 1], dtype=np.int32)

        batch_sys = {
            "positions": positions_batch,
            "charges": charges_batch,
            "idx_j": idx_j_batch,
            "neighbor_ptr": neighbor_ptr_batch,
            "batch_idx": batch_idx_batch,
            "num_atoms": 4,
            "num_systems": 2,
        }
        w_batch = _make_warp_system(batch_sys, device)
        dsf_csr(
            positions=w_batch["positions"],
            charges=w_batch["charges"],
            idx_j=w_batch["idx_j"],
            neighbor_ptr=w_batch["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w_batch["energy"],
            forces=w_batch["forces"],
            virial=w_batch["virial"],
            charge_grad=w_batch["charge_grad"],
            device=device,
            batch_idx=w_batch["batch_idx"],
        )
        batch_forces = w_batch["forces"].numpy()
        np.testing.assert_allclose(batch_forces[:2], single_forces, atol=1e-10)
        np.testing.assert_allclose(batch_forces[2:], single_forces, atol=1e-10)

    def test_batched_charge_grad_match_individual(self, two_charge_system, device):
        """Batched per-atom charge grads should match individual system."""
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2

        # Single system
        w_single = _make_warp_system(sys, device)
        dsf_csr(
            positions=w_single["positions"],
            charges=w_single["charges"],
            idx_j=w_single["idx_j"],
            neighbor_ptr=w_single["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w_single["energy"],
            forces=w_single["forces"],
            virial=w_single["virial"],
            charge_grad=w_single["charge_grad"],
            device=device,
            batch_idx=w_single["batch_idx"],
            compute_forces=False,
            compute_charge_grad=True,
        )
        single_cg = w_single["charge_grad"].numpy()

        # Batched: two copies
        positions_batch = np.concatenate(
            [sys["positions"], sys["positions"] + [50.0, 0.0, 0.0]]
        )
        charges_batch = np.concatenate([sys["charges"], sys["charges"]])
        idx_j_2 = sys["idx_j"] + sys["num_atoms"]
        idx_j_batch = np.concatenate([sys["idx_j"], idx_j_2])
        offset = sys["neighbor_ptr"][-1]
        neighbor_ptr_2 = sys["neighbor_ptr"][1:] + offset
        neighbor_ptr_batch = np.concatenate([sys["neighbor_ptr"], neighbor_ptr_2])
        batch_idx_batch = np.array([0, 0, 1, 1], dtype=np.int32)

        batch_sys = {
            "positions": positions_batch,
            "charges": charges_batch,
            "idx_j": idx_j_batch,
            "neighbor_ptr": neighbor_ptr_batch,
            "batch_idx": batch_idx_batch,
            "num_atoms": 4,
            "num_systems": 2,
        }
        w_batch = _make_warp_system(batch_sys, device)
        dsf_csr(
            positions=w_batch["positions"],
            charges=w_batch["charges"],
            idx_j=w_batch["idx_j"],
            neighbor_ptr=w_batch["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w_batch["energy"],
            forces=w_batch["forces"],
            virial=w_batch["virial"],
            charge_grad=w_batch["charge_grad"],
            device=device,
            batch_idx=w_batch["batch_idx"],
            compute_forces=False,
            compute_charge_grad=True,
        )
        batch_cg = w_batch["charge_grad"].numpy()
        np.testing.assert_allclose(batch_cg[:2], single_cg, atol=1e-10)
        np.testing.assert_allclose(batch_cg[2:], single_cg, atol=1e-10)


# ==============================================================================
# Test 3D Geometry (atoms not along x-axis)
# ==============================================================================


class TestDSF3DGeometry:
    """Test DSF with 3D atom positions (not aligned to a single axis)."""

    def test_3d_triangle_forces_energy(self, device):
        """Three atoms in a triangle: forces should have nonzero y/z components."""
        # Equilateral triangle in the xy-plane with side length 3.0
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [1.5, 3.0 * np.sqrt(3.0) / 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        charges = np.array([1.0, -1.0, 0.5], dtype=np.float64)

        # Full neighbor list: all 6 directed edges
        idx_j = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)
        neighbor_ptr = np.array([0, 2, 4, 6], dtype=np.int32)
        num_atoms = 3
        cutoff = 10.0
        alpha = 0.2

        sys = {
            "positions": positions,
            "charges": charges,
            "idx_j": idx_j,
            "neighbor_ptr": neighbor_ptr,
            "num_atoms": num_atoms,
        }
        w = _make_warp_system(sys, device)

        dsf_csr(
            positions=w["positions"],
            charges=w["charges"],
            idx_j=w["idx_j"],
            neighbor_ptr=w["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w["energy"],
            forces=w["forces"],
            virial=w["virial"],
            charge_grad=w["charge_grad"],
            device=device,
            batch_idx=w["batch_idx"],
            compute_charge_grad=True,
        )

        forces = w["forces"].numpy()
        energy_val = w["energy"].numpy()[0]
        cg = w["charge_grad"].numpy()

        # Energy should be finite and nonzero
        assert np.isfinite(energy_val)
        assert abs(energy_val) > 1e-10

        # Forces should have nonzero y-components (triangle in xy-plane)
        assert np.any(np.abs(forces[:, 1]) > 1e-10), (
            "y-components of forces should be nonzero for triangle geometry"
        )

        # z-components should be zero (all atoms in xy-plane)
        np.testing.assert_allclose(forces[:, 2], 0.0, atol=1e-10)

        # Force conservation: sum of forces should be zero
        np.testing.assert_allclose(forces.sum(axis=0), [0.0, 0.0, 0.0], atol=1e-8)

        # Charge gradients should be finite and nonzero
        assert np.all(np.isfinite(cg))
        assert np.any(np.abs(cg) > 1e-10)

        # Verify against reference computation
        for i in range(num_atoms):
            expected_cg = 0.0
            for j in range(num_atoms):
                if i == j:
                    continue
                r_ij = positions[i] - positions[j]
                r = np.linalg.norm(r_ij)
                if r >= cutoff:
                    continue
                v_pair = _dsf_pair_potential_ref(r, cutoff, alpha)
                expected_cg += charges[j] * v_pair
            self_coeff = _dsf_self_energy_coeff_ref(cutoff, alpha)
            expected_cg += 2.0 * self_coeff * charges[i]
            assert cg[i] == pytest.approx(expected_cg, abs=1e-6)


def _make_two_system_batch(sys, device, offset=50.0):
    """Helper: create a 2-system batch from a single system definition."""
    positions_batch = np.concatenate(
        [sys["positions"], sys["positions"] + [offset, 0.0, 0.0]]
    )
    charges_batch = np.concatenate([sys["charges"], sys["charges"]])
    n = sys["num_atoms"]
    batch_idx_batch = np.concatenate(
        [np.zeros(n, dtype=np.int32), np.ones(n, dtype=np.int32)]
    )
    return positions_batch, charges_batch, batch_idx_batch, n


class TestDSFMatrixBatch:
    """Matrix format + batched mode."""

    def test_matrix_batched_energy_matches_individual(self, two_charge_system, device):
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2

        w_single = _make_warp_system(sys, device)
        dsf_csr(
            positions=w_single["positions"],
            charges=w_single["charges"],
            idx_j=w_single["idx_j"],
            neighbor_ptr=w_single["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w_single["energy"],
            forces=w_single["forces"],
            virial=w_single["virial"],
            charge_grad=w_single["charge_grad"],
            device=device,
            compute_forces=False,
        )
        single_energy = w_single["energy"].numpy()[0]

        positions_batch, charges_batch, batch_idx_batch, n = _make_two_system_batch(
            sys, device
        )
        neighbor_matrix = np.full((2 * n, 2 * n), 2 * n, dtype=np.int32)
        for i in range(n):
            col = 0
            for j in range(n):
                if i != j:
                    neighbor_matrix[i, col] = j
                    col += 1
        for i in range(n):
            col = 0
            for j in range(n):
                if i != j:
                    neighbor_matrix[n + i, col] = n + j
                    col += 1

        mat_sys = {
            "positions": positions_batch,
            "charges": charges_batch,
            "neighbor_matrix": neighbor_matrix,
            "batch_idx": batch_idx_batch,
            "num_atoms": 2 * n,
            "num_systems": 2,
        }
        w_mat = _make_warp_matrix_system(mat_sys, device)
        dsf_matrix(
            positions=w_mat["positions"],
            charges=w_mat["charges"],
            neighbor_matrix=w_mat["neighbor_matrix"],
            cutoff=cutoff,
            alpha=alpha,
            fill_value=2 * n,
            energy=w_mat["energy"],
            forces=w_mat["forces"],
            virial=w_mat["virial"],
            charge_grad=w_mat["charge_grad"],
            device=device,
            batch_idx=w_mat["batch_idx"],
            compute_forces=False,
        )
        batch_energies = w_mat["energy"].numpy()
        assert batch_energies[0] == pytest.approx(single_energy, abs=1e-10)
        assert batch_energies[1] == pytest.approx(single_energy, abs=1e-10)


class TestDSFBatchPBC:
    """Batched + PBC combinations."""

    def test_batched_pbc_energy_matches_individual(self, two_charge_system, device):
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2
        cell_np = np.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        )

        w_single = _make_warp_system(sys, device)
        unit_shifts_single = np.zeros((len(sys["idx_j"]), 3), dtype=np.int32)
        cell_wp = wp.array(cell_np, dtype=wp.mat33d, device=device)
        shifts_wp = wp.array(unit_shifts_single, dtype=wp.vec3i, device=device)
        dsf_csr(
            positions=w_single["positions"],
            charges=w_single["charges"],
            idx_j=w_single["idx_j"],
            neighbor_ptr=w_single["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w_single["energy"],
            forces=w_single["forces"],
            virial=w_single["virial"],
            charge_grad=w_single["charge_grad"],
            cell=cell_wp,
            unit_shifts=shifts_wp,
            device=device,
            compute_forces=False,
        )
        single_energy = w_single["energy"].numpy()[0]

        positions_batch, charges_batch, batch_idx_batch, n = _make_two_system_batch(
            sys, device
        )
        idx_j_2 = sys["idx_j"] + n
        idx_j_batch = np.concatenate([sys["idx_j"], idx_j_2])
        offset = sys["neighbor_ptr"][-1]
        neighbor_ptr_2 = sys["neighbor_ptr"][1:] + offset
        neighbor_ptr_batch = np.concatenate([sys["neighbor_ptr"], neighbor_ptr_2])
        unit_shifts_batch = np.zeros((len(idx_j_batch), 3), dtype=np.int32)
        cell_batch = np.tile(cell_np, (2, 1, 1))

        batch_sys = {
            "positions": positions_batch,
            "charges": charges_batch,
            "idx_j": idx_j_batch,
            "neighbor_ptr": neighbor_ptr_batch,
            "batch_idx": batch_idx_batch,
            "num_atoms": 2 * n,
            "num_systems": 2,
        }
        w_batch = _make_warp_system(batch_sys, device)
        cell_batch_wp = wp.array(cell_batch, dtype=wp.mat33d, device=device)
        shifts_batch_wp = wp.array(unit_shifts_batch, dtype=wp.vec3i, device=device)
        dsf_csr(
            positions=w_batch["positions"],
            charges=w_batch["charges"],
            idx_j=w_batch["idx_j"],
            neighbor_ptr=w_batch["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w_batch["energy"],
            forces=w_batch["forces"],
            virial=w_batch["virial"],
            charge_grad=w_batch["charge_grad"],
            cell=cell_batch_wp,
            unit_shifts=shifts_batch_wp,
            device=device,
            batch_idx=w_batch["batch_idx"],
            compute_forces=False,
        )
        batch_energies = w_batch["energy"].numpy()
        assert batch_energies[0] == pytest.approx(single_energy, abs=1e-10)
        assert batch_energies[1] == pytest.approx(single_energy, abs=1e-10)


class TestDSFMatrixFloat32:
    """Matrix format + float32 precision."""

    def test_matrix_float32_energy(self, two_charge_system, device):
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2

        positions_f32 = sys["positions"].astype(np.float32)
        charges_f32 = sys["charges"].astype(np.float32)
        n = sys["num_atoms"]
        fill_value = n
        neighbor_matrix = np.full((n, n), fill_value, dtype=np.int32)
        for i in range(n):
            col = 0
            for j in range(n):
                if i != j:
                    neighbor_matrix[i, col] = j
                    col += 1

        pos_wp = wp.array(positions_f32, dtype=wp.vec3f, device=device)
        ch_wp = wp.array(charges_f32, dtype=wp.float32, device=device)
        nm_wp = wp.array(neighbor_matrix, dtype=wp.int32, device=device)
        energy_wp = wp.zeros(1, dtype=wp.float64, device=device)
        forces_wp = wp.zeros(n, dtype=wp.vec3f, device=device)
        virial_wp = wp.zeros(1, dtype=wp.mat33f, device=device)
        cg_wp = wp.zeros(n, dtype=wp.float32, device=device)

        dsf_matrix(
            positions=pos_wp,
            charges=ch_wp,
            neighbor_matrix=nm_wp,
            cutoff=cutoff,
            alpha=alpha,
            fill_value=fill_value,
            energy=energy_wp,
            forces=forces_wp,
            virial=virial_wp,
            charge_grad=cg_wp,
            device=device,
            compute_forces=False,
        )
        energy_f32 = energy_wp.numpy()[0]

        r = sys["distance"]
        v_pair = _dsf_pair_potential_ref(r, cutoff, alpha)
        self_coeff = _dsf_self_energy_coeff_ref(cutoff, alpha)
        expected = sys["charges"][0] * sys["charges"][1] * v_pair + self_coeff * (
            sys["charges"][0] ** 2 + sys["charges"][1] ** 2
        )
        assert energy_f32 == pytest.approx(expected, abs=1e-3)


class TestDSFMatrixChargeGrad:
    """Matrix format + charge gradients."""

    def test_matrix_charge_grad_finite_difference(self, device):
        positions = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float64
        )
        charges = np.array([1.0, -0.5, 0.8], dtype=np.float64)
        num_atoms = 3
        cutoff = 10.0
        alpha = 0.2
        fill_value = num_atoms
        neighbor_matrix = np.full((num_atoms, num_atoms), fill_value, dtype=np.int32)
        for i in range(num_atoms):
            col = 0
            for j in range(num_atoms):
                if i != j:
                    neighbor_matrix[i, col] = j
                    col += 1

        def compute_energy(q):
            pos_wp = wp.array(positions, dtype=wp.vec3d, device=device)
            ch_wp = wp.array(q, dtype=wp.float64, device=device)
            nm_wp = wp.array(neighbor_matrix, dtype=wp.int32, device=device)
            e_wp = wp.zeros(1, dtype=wp.float64, device=device)
            f_wp = wp.zeros(num_atoms, dtype=wp.vec3d, device=device)
            v_wp = wp.zeros(1, dtype=wp.mat33d, device=device)
            cg_wp = wp.zeros(num_atoms, dtype=wp.float64, device=device)
            dsf_matrix(
                positions=pos_wp,
                charges=ch_wp,
                neighbor_matrix=nm_wp,
                cutoff=cutoff,
                alpha=alpha,
                fill_value=fill_value,
                energy=e_wp,
                forces=f_wp,
                virial=v_wp,
                charge_grad=cg_wp,
                device=device,
                compute_forces=False,
                compute_charge_grad=True,
            )
            return e_wp.numpy()[0], cg_wp.numpy()

        _, cg = compute_energy(charges)

        eps = 1e-6
        for i in range(num_atoms):
            q_plus = charges.copy()
            q_minus = charges.copy()
            q_plus[i] += eps
            q_minus[i] -= eps
            e_plus, _ = compute_energy(q_plus)
            e_minus, _ = compute_energy(q_minus)
            fd_cg = (e_plus - e_minus) / (2 * eps)
            assert cg[i] == pytest.approx(fd_cg, abs=1e-4)


class TestDSFBatchVirial:
    """Batched + virial."""

    def test_batched_virial_matches_individual(self, two_charge_system, device):
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.2
        cell_np = np.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        )

        w_single = _make_warp_system(sys, device)
        unit_shifts_single = np.zeros((len(sys["idx_j"]), 3), dtype=np.int32)
        cell_wp = wp.array(cell_np, dtype=wp.mat33d, device=device)
        shifts_wp = wp.array(unit_shifts_single, dtype=wp.vec3i, device=device)
        dsf_csr(
            positions=w_single["positions"],
            charges=w_single["charges"],
            idx_j=w_single["idx_j"],
            neighbor_ptr=w_single["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w_single["energy"],
            forces=w_single["forces"],
            virial=w_single["virial"],
            charge_grad=w_single["charge_grad"],
            cell=cell_wp,
            unit_shifts=shifts_wp,
            device=device,
            compute_virial=True,
        )
        single_virial = w_single["virial"].numpy()[0]

        positions_batch, charges_batch, batch_idx_batch, n = _make_two_system_batch(
            sys, device
        )
        idx_j_2 = sys["idx_j"] + n
        idx_j_batch = np.concatenate([sys["idx_j"], idx_j_2])
        offset = sys["neighbor_ptr"][-1]
        neighbor_ptr_2 = sys["neighbor_ptr"][1:] + offset
        neighbor_ptr_batch = np.concatenate([sys["neighbor_ptr"], neighbor_ptr_2])
        unit_shifts_batch = np.zeros((len(idx_j_batch), 3), dtype=np.int32)
        cell_batch = np.tile(cell_np, (2, 1, 1))

        batch_sys = {
            "positions": positions_batch,
            "charges": charges_batch,
            "idx_j": idx_j_batch,
            "neighbor_ptr": neighbor_ptr_batch,
            "batch_idx": batch_idx_batch,
            "num_atoms": 2 * n,
            "num_systems": 2,
        }
        w_batch = _make_warp_system(batch_sys, device)
        cell_batch_wp = wp.array(cell_batch, dtype=wp.mat33d, device=device)
        shifts_batch_wp = wp.array(unit_shifts_batch, dtype=wp.vec3i, device=device)
        dsf_csr(
            positions=w_batch["positions"],
            charges=w_batch["charges"],
            idx_j=w_batch["idx_j"],
            neighbor_ptr=w_batch["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w_batch["energy"],
            forces=w_batch["forces"],
            virial=w_batch["virial"],
            charge_grad=w_batch["charge_grad"],
            cell=cell_batch_wp,
            unit_shifts=shifts_batch_wp,
            device=device,
            batch_idx=w_batch["batch_idx"],
            compute_virial=True,
        )
        batch_virials = w_batch["virial"].numpy()
        np.testing.assert_allclose(batch_virials[0], single_virial, atol=1e-10)
        np.testing.assert_allclose(batch_virials[1], single_virial, atol=1e-10)


class TestDSFAlphaZeroMatrix:
    """Matrix + alpha=0."""

    def test_matrix_energy_alpha_zero(self, two_charge_system, device):
        sys = two_charge_system
        cutoff = 10.0
        alpha = 0.0
        n = sys["num_atoms"]
        fill_value = n
        neighbor_matrix = np.full((n, n), fill_value, dtype=np.int32)
        for i in range(n):
            col = 0
            for j in range(n):
                if i != j:
                    neighbor_matrix[i, col] = j
                    col += 1

        mat_sys = {
            "positions": sys["positions"],
            "charges": sys["charges"],
            "neighbor_matrix": neighbor_matrix,
            "num_atoms": n,
        }
        w_mat = _make_warp_matrix_system(mat_sys, device)
        dsf_matrix(
            positions=w_mat["positions"],
            charges=w_mat["charges"],
            neighbor_matrix=w_mat["neighbor_matrix"],
            cutoff=cutoff,
            alpha=alpha,
            fill_value=fill_value,
            energy=w_mat["energy"],
            forces=w_mat["forces"],
            virial=w_mat["virial"],
            charge_grad=w_mat["charge_grad"],
            device=device,
            compute_forces=False,
        )

        w_csr = _make_warp_system(sys, device)
        dsf_csr(
            positions=w_csr["positions"],
            charges=w_csr["charges"],
            idx_j=w_csr["idx_j"],
            neighbor_ptr=w_csr["neighbor_ptr"],
            cutoff=cutoff,
            alpha=alpha,
            energy=w_csr["energy"],
            forces=w_csr["forces"],
            virial=w_csr["virial"],
            charge_grad=w_csr["charge_grad"],
            device=device,
            compute_forces=False,
        )
        e_mat = w_mat["energy"].numpy()[0]
        e_csr = w_csr["energy"].numpy()[0]
        assert e_mat == pytest.approx(e_csr, abs=1e-10)


class TestDSFForceConservationMatrix:
    """Matrix force conservation (Newton's 3rd law)."""

    def test_matrix_force_sum_zero(self, device):
        positions = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float64
        )
        charges = np.array([1.0, -0.5, 0.8], dtype=np.float64)
        num_atoms = 3
        cutoff = 10.0
        alpha = 0.2
        fill_value = num_atoms
        neighbor_matrix = np.full((num_atoms, num_atoms), fill_value, dtype=np.int32)
        for i in range(num_atoms):
            col = 0
            for j in range(num_atoms):
                if i != j:
                    neighbor_matrix[i, col] = j
                    col += 1

        mat_sys = {
            "positions": positions,
            "charges": charges,
            "neighbor_matrix": neighbor_matrix,
            "num_atoms": num_atoms,
        }
        w_mat = _make_warp_matrix_system(mat_sys, device)
        dsf_matrix(
            positions=w_mat["positions"],
            charges=w_mat["charges"],
            neighbor_matrix=w_mat["neighbor_matrix"],
            cutoff=cutoff,
            alpha=alpha,
            fill_value=fill_value,
            energy=w_mat["energy"],
            forces=w_mat["forces"],
            virial=w_mat["virial"],
            charge_grad=w_mat["charge_grad"],
            device=device,
        )
        forces = w_mat["forces"].numpy()
        force_sum = np.sum(forces, axis=0)
        np.testing.assert_allclose(force_sum, [0.0, 0.0, 0.0], atol=1e-10)

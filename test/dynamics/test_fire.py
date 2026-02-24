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
Unit tests for FIRE optimizer and cell filter utilities.

Tests cover:
- fire_step: Full FIRE optimization step with MD integration
  - Single system mode
  - batch_idx mode
  - atom_ptr mode
  - No-downhill and downhill variants
- fire_update: FIRE velocity mixing without MD integration
- Cell filter utilities for variable-cell optimization
  - align_cell
  - extend_batch_idx / extend_atom_ptr
  - pack/unpack positions, velocities, forces, masses
  - stress_to_cell_force
- Physics correctness (harmonic potential convergence)
- Float32 and float64 support
"""

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.optimizers import fire_step, fire_update
from nvalchemiops.dynamics.utils import (
    align_cell,
    compute_cell_volume,
    extend_atom_ptr,
    extend_batch_idx,
    pack_forces_with_cell,
    pack_masses_with_cell,
    pack_positions_with_cell,
    pack_velocities_with_cell,
    stress_to_cell_force,
    unpack_positions_with_cell,
    unpack_velocities_with_cell,
)

# ==============================================================================
# Test Configuration
# ==============================================================================

DEVICES = ["cuda:0"]

DTYPE_CONFIGS = [
    pytest.param(wp.vec3f, wp.float32, wp.mat33f, np.float32, id="float32"),
    pytest.param(wp.vec3d, wp.float64, wp.mat33d, np.float64, id="float64"),
]


# ==============================================================================
# Helper Functions
# ==============================================================================


def make_fire_params(num_systems, dtype_scalar, device, np_dtype):
    """Create FIRE parameter arrays for testing."""
    alpha = wp.array(
        np.full(num_systems, 0.1, dtype=np_dtype), dtype=dtype_scalar, device=device
    )
    dt = wp.array(
        np.full(num_systems, 0.01, dtype=np_dtype), dtype=dtype_scalar, device=device
    )
    alpha_start = wp.array(
        np.full(num_systems, 0.1, dtype=np_dtype), dtype=dtype_scalar, device=device
    )
    f_alpha = wp.array(
        np.full(num_systems, 0.99, dtype=np_dtype), dtype=dtype_scalar, device=device
    )
    dt_min = wp.array(
        np.full(num_systems, 0.001, dtype=np_dtype), dtype=dtype_scalar, device=device
    )
    dt_max = wp.array(
        np.full(num_systems, 0.1, dtype=np_dtype), dtype=dtype_scalar, device=device
    )
    maxstep = wp.array(
        np.full(num_systems, 0.2, dtype=np_dtype), dtype=dtype_scalar, device=device
    )
    n_steps_positive = wp.zeros(num_systems, dtype=wp.int32, device=device)
    n_min = wp.array(
        np.full(num_systems, 5, dtype=np.int32), dtype=wp.int32, device=device
    )
    f_dec = wp.array(
        np.full(num_systems, 0.5, dtype=np_dtype), dtype=dtype_scalar, device=device
    )
    f_inc = wp.array(
        np.full(num_systems, 1.1, dtype=np_dtype), dtype=dtype_scalar, device=device
    )

    uphill_flag = wp.empty(num_systems, dtype=wp.int32, device=device)

    return {
        "alpha": alpha,
        "dt": dt,
        "alpha_start": alpha_start,
        "f_alpha": f_alpha,
        "dt_min": dt_min,
        "dt_max": dt_max,
        "maxstep": maxstep,
        "n_steps_positive": n_steps_positive,
        "n_min": n_min,
        "f_dec": f_dec,
        "f_inc": f_inc,
        "uphill_flag": uphill_flag,
    }


def make_accumulators(num_systems, dtype_scalar, device, np_dtype):
    """Create accumulator arrays for single/batch_idx modes."""
    return {
        "vf": wp.zeros(num_systems, dtype=dtype_scalar, device=device),
        "vv": wp.zeros(num_systems, dtype=dtype_scalar, device=device),
        "ff": wp.zeros(num_systems, dtype=dtype_scalar, device=device),
    }


def make_downhill_arrays(
    num_atoms, num_systems, dtype_vec, dtype_scalar, device, np_dtype
):
    """Create downhill check arrays."""
    energy = wp.array(
        np.full(num_systems, 100.0, dtype=np_dtype), dtype=dtype_scalar, device=device
    )
    energy_last = wp.array(
        np.full(num_systems, 100.0, dtype=np_dtype), dtype=dtype_scalar, device=device
    )
    positions_last = wp.empty(num_atoms, dtype=dtype_vec, device=device)
    velocities_last = wp.empty(num_atoms, dtype=dtype_vec, device=device)

    return {
        "energy": energy,
        "energy_last": energy_last,
        "positions_last": positions_last,
        "velocities_last": velocities_last,
    }


@wp.kernel
def _compute_harmonic_forces_kernel_f32(
    positions: wp.array(dtype=wp.vec3f),
    forces: wp.array(dtype=wp.vec3f),
    spring_k: wp.float32,
):
    """Compute harmonic forces F = -k * r (float32)."""
    i = wp.tid()
    pos = positions[i]
    forces[i] = wp.vec3f(-spring_k * pos[0], -spring_k * pos[1], -spring_k * pos[2])


@wp.kernel
def _compute_harmonic_forces_kernel_f64(
    positions: wp.array(dtype=wp.vec3d),
    forces: wp.array(dtype=wp.vec3d),
    spring_k: wp.float64,
):
    """Compute harmonic forces F = -k * r (float64)."""
    i = wp.tid()
    pos = positions[i]
    forces[i] = wp.vec3d(-spring_k * pos[0], -spring_k * pos[1], -spring_k * pos[2])


def compute_harmonic_forces(positions: wp.array, forces: wp.array, k: float):
    """Compute harmonic forces F = -k * r."""
    if positions.dtype == wp.vec3f:
        wp.launch(
            _compute_harmonic_forces_kernel_f32,
            dim=positions.shape[0],
            inputs=[positions, forces, wp.float32(k)],
            device=positions.device,
        )
    else:
        wp.launch(
            _compute_harmonic_forces_kernel_f64,
            dim=positions.shape[0],
            inputs=[positions, forces, wp.float64(k)],
            device=positions.device,
        )


def compute_harmonic_energy(positions: np.ndarray, k: float) -> float:
    """Compute harmonic potential energy E = 0.5 * k * |r|^2."""
    r_sq = np.sum(positions**2)
    return 0.5 * k * r_sq


def compute_max_force(forces: np.ndarray) -> float:
    """Compute maximum force magnitude."""
    force_norms = np.linalg.norm(forces, axis=-1)
    return np.max(force_norms)


# ==============================================================================
# Harmonic Cell Force Computation
# ==============================================================================
# For testing variable-cell optimization, we define:
#   E_cell = 0.5 * k_cell * ||H - H_target||^2_F  (Frobenius norm)
#   F_cell = -dE/dH = -k_cell * (H - H_target)
#
# This gives a simple harmonic potential on cell parameters.


@wp.kernel
def _compute_harmonic_cell_stress_kernel_f32(
    cells: wp.array(dtype=wp.mat33f),
    target_cells: wp.array(dtype=wp.mat33f),
    k_cell: wp.float32,
    stress: wp.array(dtype=wp.mat33f),
):
    """Compute stress from harmonic cell potential (float32).

    E = 0.5 * k * ||H - H_target||^2
    Stress = k * (H - H_target) / V  (simplified for testing)
    """
    sys = wp.tid()
    H = cells[sys]
    H0 = target_cells[sys]

    # dE/dH = k * (H - H_target)
    # For simplicity, we use this directly as stress-like quantity
    # In reality, stress = -dE/dεV where ε is strain
    dH = wp.mat33f(
        H[0, 0] - H0[0, 0],
        H[0, 1] - H0[0, 1],
        H[0, 2] - H0[0, 2],
        H[1, 0] - H0[1, 0],
        H[1, 1] - H0[1, 1],
        H[1, 2] - H0[1, 2],
        H[2, 0] - H0[2, 0],
        H[2, 1] - H0[2, 1],
        H[2, 2] - H0[2, 2],
    )
    stress[sys] = wp.mul(dH, k_cell)


@wp.kernel
def _compute_harmonic_cell_stress_kernel_f64(
    cells: wp.array(dtype=wp.mat33d),
    target_cells: wp.array(dtype=wp.mat33d),
    k_cell: wp.float64,
    stress: wp.array(dtype=wp.mat33d),
):
    """Compute stress from harmonic cell potential (float64)."""
    sys = wp.tid()
    H = cells[sys]
    H0 = target_cells[sys]

    dH = wp.mat33d(
        H[0, 0] - H0[0, 0],
        H[0, 1] - H0[0, 1],
        H[0, 2] - H0[0, 2],
        H[1, 0] - H0[1, 0],
        H[1, 1] - H0[1, 1],
        H[1, 2] - H0[1, 2],
        H[2, 0] - H0[2, 0],
        H[2, 1] - H0[2, 1],
        H[2, 2] - H0[2, 2],
    )
    stress[sys] = wp.mul(dH, k_cell)


def compute_harmonic_cell_stress(cells, target_cells, k_cell, stress=None, device=None):
    """Compute stress from harmonic cell potential.

    Returns a stress-like tensor that drives the cell toward target_cells.
    """
    if device is None:
        device = cells.device

    if stress is None:
        stress = wp.zeros(cells.shape[0], dtype=cells.dtype, device=device)

    if cells.dtype == wp.mat33f:
        wp.launch(
            _compute_harmonic_cell_stress_kernel_f32,
            dim=cells.shape[0],
            inputs=[cells, target_cells, wp.float32(k_cell), stress],
            device=device,
        )
    else:
        wp.launch(
            _compute_harmonic_cell_stress_kernel_f64,
            dim=cells.shape[0],
            inputs=[cells, target_cells, wp.float64(k_cell), stress],
            device=device,
        )

    return stress


def compute_harmonic_cell_energy_np(cell_np, target_cell_np, k_cell):
    """Compute harmonic cell energy E = 0.5 * k * ||H - H_target||^2."""
    diff = cell_np - target_cell_np
    return 0.5 * k_cell * np.sum(diff**2)


def make_cell(cell_np, mat_dtype, device):
    """Create a (1,) shaped cell array from a (3,3) numpy array."""
    mat = mat_dtype(
        cell_np[0, 0],
        cell_np[0, 1],
        cell_np[0, 2],
        cell_np[1, 0],
        cell_np[1, 1],
        cell_np[1, 2],
        cell_np[2, 0],
        cell_np[2, 1],
        cell_np[2, 2],
    )
    return wp.array([mat], dtype=mat_dtype, device=device)


def cell_to_numpy(cells_wp, sys_idx=0):
    """Extract a cell from warp array to numpy (3, 3)."""
    wp.synchronize_device(cells_wp.device)
    mat = cells_wp.numpy()[sys_idx]
    return mat.reshape(3, 3)


# ==============================================================================
# fire_step Tests - Ptr Mode (Most Reliable)
# ==============================================================================


class TestFireStepPtr:
    """Test fire_step with atom_ptr (CSR) batching - one thread per system."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_step_ptr_runs(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test that fire_step with atom_ptr executes without error."""
        num_systems = 2
        atoms_per_system = 10
        total_atoms = num_systems * atoms_per_system

        positions = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        forces = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        atom_ptr = wp.array(
            np.array([0, atoms_per_system, total_atoms], dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )

        params = make_fire_params(num_systems, dtype_scalar, device, np_dtype)

        fire_step(
            positions=positions,
            velocities=velocities,
            forces=forces,
            masses=masses,
            atom_ptr=atom_ptr,
            **params,
        )

        wp.synchronize_device(device)
        # Should complete without error

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_step_ptr_modifies_positions(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test that fire_step actually modifies positions."""
        num_systems = 1
        num_atoms = 10

        np.random.seed(42)
        initial_pos = np.random.randn(num_atoms, 3).astype(np_dtype)

        positions = wp.array(initial_pos.copy(), dtype=dtype_vec, device=device)
        velocities = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype), dtype=dtype_scalar, device=device
        )
        atom_ptr = wp.array(
            np.array([0, num_atoms], dtype=np.int32), dtype=wp.int32, device=device
        )

        params = make_fire_params(num_systems, dtype_scalar, device, np_dtype)

        fire_step(
            positions=positions,
            velocities=velocities,
            forces=forces,
            masses=masses,
            atom_ptr=atom_ptr,
            **params,
        )

        wp.synchronize_device(device)
        final_pos = positions.numpy()

        # Positions should have changed
        assert not np.allclose(final_pos, initial_pos), "Positions should be modified"

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_step_ptr_downhill(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test fire_step with downhill check enabled."""
        num_systems = 1
        num_atoms = 10

        np.random.seed(42)
        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype), dtype=dtype_scalar, device=device
        )
        atom_ptr = wp.array(
            np.array([0, num_atoms], dtype=np.int32), dtype=wp.int32, device=device
        )

        params = make_fire_params(num_systems, dtype_scalar, device, np_dtype)
        downhill = make_downhill_arrays(
            num_atoms, num_systems, dtype_vec, dtype_scalar, device, np_dtype
        )

        # Initialize positions_last
        wp.copy(downhill["positions_last"], positions)

        fire_step(
            positions=positions,
            velocities=velocities,
            forces=forces,
            masses=masses,
            atom_ptr=atom_ptr,
            **params,
            **downhill,
        )

        wp.synchronize_device(device)


class TestFireStepPtrPhysics:
    """Physics correctness tests using ptr mode."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_harmonic_convergence_ptr(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test FIRE converges to minimum of harmonic potential using ptr mode."""
        num_atoms = 20
        spring_k = 1.0
        force_tol = 1e-3
        max_steps = 500

        np.random.seed(42)
        initial_pos = np.random.randn(num_atoms, 3).astype(np_dtype) * 2.0

        positions = wp.array(initial_pos.copy(), dtype=dtype_vec, device=device)
        velocities = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        forces = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype), dtype=dtype_scalar, device=device
        )
        atom_ptr = wp.array(
            np.array([0, num_atoms], dtype=np.int32), dtype=wp.int32, device=device
        )

        params = make_fire_params(1, dtype_scalar, device, np_dtype)

        initial_energy = compute_harmonic_energy(initial_pos, spring_k)

        for step in range(max_steps):
            compute_harmonic_forces(positions, forces, spring_k)

            fire_step(
                positions=positions,
                velocities=velocities,
                forces=forces,
                masses=masses,
                atom_ptr=atom_ptr,
                **params,
            )

            wp.synchronize_device(device)
            forces_np = forces.numpy()
            max_force = np.max(np.linalg.norm(forces_np, axis=1))

            if max_force < force_tol:
                break

        wp.synchronize_device(device)
        final_pos = positions.numpy()
        final_energy = compute_harmonic_energy(final_pos, spring_k)

        # Energy should have decreased significantly
        assert final_energy < initial_energy * 0.1, (
            f"Energy should decrease: {initial_energy:.4f} -> {final_energy:.4f}"
        )

        # Positions should be near origin
        max_displacement = np.max(np.abs(final_pos))
        assert max_displacement < 0.5, (
            f"Should converge near origin: max_disp={max_displacement}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_batched_independent_convergence_ptr(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test that batched systems converge independently with ptr mode."""
        num_systems = 2
        atoms_per_system = 10
        total_atoms = num_systems * atoms_per_system
        spring_k = 1.0
        max_steps = 300

        np.random.seed(42)
        # System 0: small displacement, System 1: large displacement
        pos_sys0 = np.random.randn(atoms_per_system, 3).astype(np_dtype) * 0.5
        pos_sys1 = np.random.randn(atoms_per_system, 3).astype(np_dtype) * 3.0
        initial_pos = np.vstack([pos_sys0, pos_sys1])

        positions = wp.array(initial_pos.copy(), dtype=dtype_vec, device=device)
        velocities = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        forces = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype), dtype=dtype_scalar, device=device
        )
        atom_ptr = wp.array(
            np.array([0, atoms_per_system, total_atoms], dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )

        params = make_fire_params(num_systems, dtype_scalar, device, np_dtype)

        for step in range(max_steps):
            compute_harmonic_forces(positions, forces, spring_k)

            fire_step(
                positions=positions,
                velocities=velocities,
                forces=forces,
                masses=masses,
                atom_ptr=atom_ptr,
                **params,
            )

        wp.synchronize_device(device)
        final_pos = positions.numpy()

        # Both systems should converge to near origin
        for sys_id in range(num_systems):
            start = sys_id * atoms_per_system
            end = (sys_id + 1) * atoms_per_system
            max_disp = np.max(np.abs(final_pos[start:end]))
            assert max_disp < 1.0, (
                f"System {sys_id} should converge: max_disp={max_disp}"
            )


# ==============================================================================
# fire_update Tests
# ==============================================================================


class TestFireUpdate:
    """Test fire_update - velocity mixing without MD integration."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_update_ptr_runs(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test that fire_update with atom_ptr executes without error."""
        num_systems = 2
        atoms_per_system = 10
        total_atoms = num_systems * atoms_per_system

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        atom_ptr = wp.array(
            np.array([0, atoms_per_system, total_atoms], dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )

        params = make_fire_params(num_systems, dtype_scalar, device, np_dtype)
        # Remove params not needed for fire_update
        del params["maxstep"]
        del params["uphill_flag"]

        fire_update(
            velocities=velocities,
            forces=forces,
            atom_ptr=atom_ptr,
            **params,
        )

        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_update_modifies_velocities(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test that fire_update modifies velocities but not positions."""
        num_systems = 1
        num_atoms = 10

        np.random.seed(42)
        initial_vel = np.random.randn(num_atoms, 3).astype(np_dtype)

        velocities = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        atom_ptr = wp.array(
            np.array([0, num_atoms], dtype=np.int32), dtype=wp.int32, device=device
        )

        params = make_fire_params(num_systems, dtype_scalar, device, np_dtype)
        del params["maxstep"]
        del params["uphill_flag"]

        fire_update(
            velocities=velocities,
            forces=forces,
            atom_ptr=atom_ptr,
            **params,
        )

        wp.synchronize_device(device)
        final_vel = velocities.numpy()

        # Velocities should have changed (velocity mixing)
        assert not np.allclose(final_vel, initial_vel), "Velocities should be modified"

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_update_single_system(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test fire_update in single system mode."""
        num_atoms = 10
        num_systems = 1

        np.random.seed(42)
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )

        params = make_fire_params(num_systems, dtype_scalar, device, np_dtype)
        del params["maxstep"]
        del params["uphill_flag"]
        accum = make_accumulators(num_systems, dtype_scalar, device, np_dtype)

        fire_update(
            velocities=velocities,
            forces=forces,
            **params,
            **accum,
        )

        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_update_batch_idx(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test fire_update with batch_idx mode."""
        num_systems = 2
        atoms_per_system = 10
        total_atoms = num_systems * atoms_per_system

        np.random.seed(42)
        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        params = make_fire_params(num_systems, dtype_scalar, device, np_dtype)
        del params["maxstep"]
        del params["uphill_flag"]
        accum = make_accumulators(num_systems, dtype_scalar, device, np_dtype)

        fire_update(
            velocities=velocities,
            forces=forces,
            batch_idx=batch_idx,
            **params,
            **accum,
        )

        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_update_ptr_downhill(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test fire_update with downhill check enabled."""
        num_systems = 1
        num_atoms = 10

        np.random.seed(42)
        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        atom_ptr = wp.array(
            np.array([0, num_atoms], dtype=np.int32), dtype=wp.int32, device=device
        )

        params = make_fire_params(num_systems, dtype_scalar, device, np_dtype)
        del params["maxstep"]
        del params["uphill_flag"]

        # Create downhill arrays for fire_update
        energy = wp.array([1.0], dtype=dtype_scalar, device=device)
        energy_last = wp.array([2.0], dtype=dtype_scalar, device=device)
        positions_last = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities_last = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )

        fire_update(
            velocities=velocities,
            forces=forces,
            atom_ptr=atom_ptr,
            energy=energy,
            energy_last=energy_last,
            positions=positions,
            positions_last=positions_last,
            velocities_last=velocities_last,
            **params,
        )

        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_update_device_inference(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test fire_update device inference."""
        num_atoms = 10

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        atom_ptr = wp.array(
            np.array([0, num_atoms], dtype=np.int32), dtype=wp.int32, device=device
        )

        params = make_fire_params(1, dtype_scalar, device, np_dtype)
        del params["maxstep"]
        del params["uphill_flag"]

        # Don't pass device - should be inferred
        fire_update(
            velocities=velocities,
            forces=forces,
            atom_ptr=atom_ptr,
            **params,
        )

        wp.synchronize_device(device)


class TestFireUpdateErrors:
    """Test fire_update error cases."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_fire_update_both_batch_idx_and_atom_ptr_error(self, device):
        """Test fire_update raises error when both batch_idx and atom_ptr are provided."""
        num_atoms = 10
        np_dtype = np.float64

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=wp.vec3d,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=wp.vec3d,
            device=device,
        )
        atom_ptr = wp.array([0, num_atoms], dtype=wp.int32, device=device)
        batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=device)

        params = make_fire_params(1, wp.float64, device, np_dtype)
        del params["maxstep"]
        del params["uphill_flag"]

        with pytest.raises(ValueError, match="batch_idx OR atom_ptr, not both"):
            fire_update(
                velocities=velocities,
                forces=forces,
                atom_ptr=atom_ptr,
                batch_idx=batch_idx,
                **params,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_fire_update_partial_downhill_arrays_error(self, device):
        """Test fire_update raises error with partial downhill arrays."""
        num_atoms = 10
        np_dtype = np.float64

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=wp.vec3d,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=wp.vec3d,
            device=device,
        )
        atom_ptr = wp.array([0, num_atoms], dtype=wp.int32, device=device)

        params = make_fire_params(1, wp.float64, device, np_dtype)
        del params["maxstep"]
        del params["uphill_flag"]

        # Only provide energy, not the other required arrays
        energy = wp.array([1.0], dtype=wp.float64, device=device)

        with pytest.raises(ValueError, match="For downhill check, must provide ALL"):
            fire_update(
                velocities=velocities,
                forces=forces,
                atom_ptr=atom_ptr,
                energy=energy,
                **params,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_fire_update_batch_idx_missing_accumulators_error(self, device):
        """Test fire_update with batch_idx raises error without accumulators."""
        num_systems = 2
        atoms_per_system = 5
        total_atoms = num_systems * atoms_per_system
        np_dtype = np.float64

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=wp.vec3d,
            device=device,
        )
        forces = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=wp.vec3d,
            device=device,
        )
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        params = make_fire_params(num_systems, wp.float64, device, np_dtype)
        del params["maxstep"]
        del params["uphill_flag"]

        with pytest.raises(ValueError, match="vf, vv, ff accumulators required"):
            fire_update(
                velocities=velocities,
                forces=forces,
                batch_idx=batch_idx,
                **params,
            )


# ==============================================================================
# Cell Filter Utilities Tests
# ==============================================================================


class TestAlignCell:
    """Test align_cell utility."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_align_cell_cubic(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test that cubic cell remains unchanged after alignment."""
        cell_np = np.array(
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ],
            dtype=np_dtype,
        )

        cell = make_cell(cell_np, dtype_mat, device)
        positions = wp.array(
            np.random.randn(5, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        transform = wp.empty(1, dtype=dtype_mat, device=device)

        align_cell(positions, cell, transform, device=device)

        wp.synchronize_device(device)
        aligned_cell = cell_to_numpy(cell)

        # Cubic cell should already be upper-triangular
        # Check that lower-triangular elements (above diagonal) are zero
        np.testing.assert_allclose(aligned_cell[0, 1], 0.0, atol=1e-5)
        np.testing.assert_allclose(aligned_cell[0, 2], 0.0, atol=1e-5)
        np.testing.assert_allclose(aligned_cell[1, 2], 0.0, atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_align_cell_triclinic(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test alignment of triclinic cell to upper-triangular form."""
        # Create a rotated cell
        theta = np.pi / 6  # 30 degrees
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=np_dtype,
        )
        cell_np = rotation @ np.diag([10.0, 8.0, 6.0]).astype(np_dtype)

        original_vol = np.abs(np.linalg.det(cell_np))

        cell = make_cell(cell_np, dtype_mat, device)
        positions = wp.array(
            np.random.randn(5, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        transform = wp.empty(1, dtype=dtype_mat, device=device)

        align_cell(positions, cell, transform, device=device)

        wp.synchronize_device(device)
        aligned_cell = cell_to_numpy(cell)
        aligned_vol = np.abs(np.linalg.det(aligned_cell))

        # Volume should be preserved
        np.testing.assert_allclose(aligned_vol, original_vol, rtol=1e-4)

        # Should be upper-triangular (elements above diagonal should be zero)
        np.testing.assert_allclose(aligned_cell[0, 1], 0.0, atol=1e-4)
        np.testing.assert_allclose(aligned_cell[0, 2], 0.0, atol=1e-4)
        np.testing.assert_allclose(aligned_cell[1, 2], 0.0, atol=1e-4)


class TestExtendBatchIdx:
    """Test extend_batch_idx utility."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_extend_batch_idx_single_system(self, device):
        """Test batch_idx extension for single system."""
        num_atoms = 10
        num_systems = 1

        batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=device)
        extended_batch_idx = wp.empty(
            num_atoms + 2 * num_systems, dtype=wp.int32, device=device
        )
        extended = extend_batch_idx(
            batch_idx, num_atoms, num_systems, extended_batch_idx, device=device
        )

        wp.synchronize_device(device)
        result = extended.numpy()

        # Should have num_atoms + 2 entries
        assert result.shape[0] == num_atoms + 2
        # All should be system 0
        assert np.all(result == 0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_extend_batch_idx_multiple_systems(self, device):
        """Test batch_idx extension for multiple systems."""
        atoms_per_system = 5
        num_systems = 3
        num_atoms = atoms_per_system * num_systems

        batch_idx_np = np.repeat(np.arange(num_systems), atoms_per_system).astype(
            np.int32
        )
        batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)
        extended_batch_idx = wp.empty(
            num_atoms + 2 * num_systems, dtype=wp.int32, device=device
        )

        extended = extend_batch_idx(
            batch_idx, num_atoms, num_systems, extended_batch_idx, device=device
        )

        wp.synchronize_device(device)
        result = extended.numpy()

        # Should have num_atoms + 2*num_systems entries
        expected_size = num_atoms + 2 * num_systems
        assert result.shape[0] == expected_size

        # First num_atoms entries should match original
        np.testing.assert_array_equal(result[:num_atoms], batch_idx_np)

        # Cell DOF entries: [sys0, sys0, sys1, sys1, sys2, sys2]
        cell_dof_idx = result[num_atoms:]
        expected_cell_idx = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        np.testing.assert_array_equal(cell_dof_idx, expected_cell_idx)


class TestExtendAtomPtr:
    """Test extend_atom_ptr utility."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_extend_atom_ptr(self, device):
        """Test atom_ptr extension."""
        # Original: 2 systems, 10 and 15 atoms
        atom_ptr_np = np.array([0, 10, 25], dtype=np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)
        extended_atom_ptr = wp.empty(
            atom_ptr_np.shape[0], dtype=wp.int32, device=device
        )

        extended = extend_atom_ptr(atom_ptr, extended_atom_ptr, device=device)

        wp.synchronize_device(device)
        result = extended.numpy()

        # extended[sys] = atom_ptr[sys] + 2*sys
        expected = np.array([0, 12, 29], dtype=np.int32)  # [0+0, 10+2, 25+4]
        np.testing.assert_array_equal(result, expected)


class TestPackUnpack:
    """Test pack/unpack utilities for positions, velocities, forces, and masses.

    The pack/unpack utilities are designed for single-system use.
    For batched systems, pack each system separately, concatenate, and use
    extend_batch_idx() or extend_atom_ptr() to update batching arrays.
    """

    # =========================================================================
    # Positions Pack/Unpack Tests
    # =========================================================================

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_positions_pack_unpack_roundtrip(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test that pack then unpack recovers original position and cell data."""
        num_atoms = 10

        # Create test data
        np.random.seed(42)
        positions_np = np.random.randn(num_atoms, 3).astype(np_dtype)
        cell_np = np.array(
            [
                [10.0, 0.0, 0.0],
                [2.0, 9.0, 0.0],
                [1.0, 2.0, 8.0],
            ],
            dtype=np_dtype,
        )

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        cell = make_cell(cell_np, dtype_mat, device)

        # Pack
        extended = wp.empty(num_atoms + 2, dtype=dtype_vec, device=device)
        pack_positions_with_cell(positions, cell, extended, device=device)

        wp.synchronize_device(device)
        assert extended.shape[0] == num_atoms + 2

        # Unpack
        pos_out = wp.empty(num_atoms, dtype=dtype_vec, device=device)
        cell_out = wp.empty(1, dtype=dtype_mat, device=device)
        unpack_positions_with_cell(
            extended, pos_out, cell_out, num_atoms=num_atoms, device=device
        )

        wp.synchronize_device(device)
        pos_result = pos_out.numpy()
        cell_result = cell_to_numpy(cell_out)

        # Check positions recovered
        np.testing.assert_allclose(pos_result, positions_np, rtol=1e-5)

        # Check cell recovered
        np.testing.assert_allclose(cell_result, cell_np, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_positions_batched_with_batch_idx(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test positions pack/unpack for batched systems using batch_idx."""
        num_systems = 2
        atoms_per_system = [5, 8]  # Different sizes per system
        total_atoms = sum(atoms_per_system)

        np.random.seed(42)

        # Create per-system data
        positions_list = []
        cells_list = []
        extended_list = []

        for sys_id in range(num_systems):
            n_atoms = atoms_per_system[sys_id]
            pos_np = np.random.randn(n_atoms, 3).astype(np_dtype)
            cell_np = np.diag([10.0 + sys_id, 10.0 + sys_id, 10.0 + sys_id]).astype(
                np_dtype
            )

            positions_list.append(pos_np)
            cells_list.append(cell_np)

            # Pack each system
            pos_wp = wp.array(pos_np, dtype=dtype_vec, device=device)
            cell_wp = make_cell(cell_np, dtype_mat, device)
            ext = wp.empty(n_atoms + 2, dtype=dtype_vec, device=device)
            pack_positions_with_cell(pos_wp, cell_wp, ext, device=device)

            wp.synchronize_device(device)
            extended_list.append(ext.numpy())

        # Concatenate extended arrays
        extended_concat = np.vstack(extended_list)

        # Create extended batch_idx
        original_batch_idx = np.concatenate(
            [
                np.full(atoms_per_system[i], i, dtype=np.int32)
                for i in range(num_systems)
            ]
        )
        batch_idx = wp.array(original_batch_idx, dtype=wp.int32, device=device)
        ext_batch_idx_arr = wp.empty(
            total_atoms + 2 * num_systems, dtype=wp.int32, device=device
        )
        ext_batch_idx = extend_batch_idx(
            batch_idx, total_atoms, num_systems, ext_batch_idx_arr, device=device
        )

        wp.synchronize_device(device)
        ext_batch_idx_np = ext_batch_idx.numpy()

        # Verify extended batch_idx size
        expected_ext_size = total_atoms + 2 * num_systems
        assert len(ext_batch_idx_np) == expected_ext_size
        assert len(extended_concat) == expected_ext_size

        # Verify each system's data can be unpacked correctly
        for sys_id in range(num_systems):
            n_atoms = atoms_per_system[sys_id]
            n_extended = n_atoms + 2
            start = sum(atoms_per_system[:sys_id]) + 2 * sys_id
            end = start + n_extended

            sys_extended = wp.array(
                extended_concat[start:end], dtype=dtype_vec, device=device
            )
            pos_out = wp.empty(n_atoms, dtype=dtype_vec, device=device)
            cell_out = wp.empty(1, dtype=dtype_mat, device=device)
            unpack_positions_with_cell(
                sys_extended, pos_out, cell_out, num_atoms=n_atoms, device=device
            )

            wp.synchronize_device(device)
            np.testing.assert_allclose(
                pos_out.numpy(), positions_list[sys_id], rtol=1e-5
            )
            np.testing.assert_allclose(
                cell_to_numpy(cell_out), cells_list[sys_id], rtol=1e-5
            )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_positions_batched_with_atom_ptr(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test positions pack/unpack for batched systems using atom_ptr.

        This tests the proper batched usage pattern where:
        - Positions are concatenated across all systems
        - Cells are batched with shape (num_systems,)
        - pack_positions_with_cell is called once with atom_ptr
        """
        num_systems = 3
        atoms_per_system = [4, 6, 5]
        total_atoms = sum(atoms_per_system)

        np.random.seed(123)

        # Create concatenated positions and batched cells
        positions_list = []
        cells_list = []
        for sys_id in range(num_systems):
            n_atoms = atoms_per_system[sys_id]
            pos_np = np.random.randn(n_atoms, 3).astype(np_dtype)
            cell_np = np.eye(3, dtype=np_dtype) * (8.0 + sys_id)
            positions_list.append(pos_np)
            cells_list.append(cell_np)

        # Concatenate positions and stack cells
        positions_concat_np = np.vstack(positions_list)
        cells_np = np.stack(cells_list, axis=0)  # Shape: (num_systems, 3, 3)

        # Create warp arrays
        positions = wp.array(positions_concat_np, dtype=dtype_vec, device=device)
        cells = wp.array(cells_np.reshape(-1), dtype=dtype_mat, device=device)

        # Create atom_ptr
        atom_ptr_np = np.array([0] + list(np.cumsum(atoms_per_system)), dtype=np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        # Extended atom_ptr
        ext_atom_ptr_arr = wp.empty(atom_ptr_np.shape[0], dtype=wp.int32, device=device)
        ext_atom_ptr = extend_atom_ptr(atom_ptr, ext_atom_ptr_arr, device=device)
        wp.synchronize_device(device)
        ext_atom_ptr_np = ext_atom_ptr.numpy()

        # Verify extended atom_ptr: extended[sys] = atom_ptr[sys] + 2*sys
        for sys_id in range(num_systems + 1):
            expected = atom_ptr_np[sys_id] + 2 * sys_id
            assert ext_atom_ptr_np[sys_id] == expected, (
                f"System {sys_id}: expected {expected}, got {ext_atom_ptr_np[sys_id]}"
            )

        # Pack all systems at once using atom_ptr
        extended = wp.empty(
            total_atoms + 2 * num_systems, dtype=dtype_vec, device=device
        )
        pack_positions_with_cell(
            positions,
            cells,
            extended,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
        )

        wp.synchronize_device(device)
        extended_np = extended.numpy()

        # Verify size
        expected_size = total_atoms + 2 * num_systems
        assert len(extended_np) == expected_size
        assert len(extended_np) == ext_atom_ptr_np[-1]

        # Verify each system's data in the extended array
        for sys_id in range(num_systems):
            ext_start = ext_atom_ptr_np[sys_id]
            n_atoms = atoms_per_system[sys_id]

            # Check atomic positions
            sys_positions = extended_np[ext_start : ext_start + n_atoms]
            np.testing.assert_allclose(sys_positions, positions_list[sys_id], rtol=1e-5)

            # Check cell DOFs
            cell_v1 = extended_np[ext_start + n_atoms]  # [H[0,0], H[1,0], H[2,0]]
            cell_v2 = extended_np[ext_start + n_atoms + 1]  # [H[1,1], H[2,1], H[2,2]]
            expected_cell = cells_list[sys_id]

            # Upper-triangular cell: only diagonal for this test
            np.testing.assert_allclose(cell_v1[0], expected_cell[0, 0], rtol=1e-5)
            np.testing.assert_allclose(cell_v2[0], expected_cell[1, 1], rtol=1e-5)
            np.testing.assert_allclose(cell_v2[2], expected_cell[2, 2], rtol=1e-5)

        # Test unpack roundtrip
        pos_out = wp.empty(total_atoms, dtype=dtype_vec, device=device)
        cells_out = wp.empty(num_systems, dtype=dtype_mat, device=device)
        unpack_positions_with_cell(
            extended,
            pos_out,
            cells_out,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
        )

        wp.synchronize_device(device)

        # Verify positions recovered
        np.testing.assert_allclose(pos_out.numpy(), positions_concat_np, rtol=1e-5)

        # Verify cells recovered
        cells_out_np = cells_out.numpy().reshape(num_systems, 3, 3)
        for sys_id in range(num_systems):
            np.testing.assert_allclose(
                cells_out_np[sys_id], cells_list[sys_id], rtol=1e-5
            )

    # =========================================================================
    # Velocities Pack/Unpack Tests
    # =========================================================================

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_velocities_pack_unpack_roundtrip(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test that velocity pack then unpack recovers original data.

        Note: The pack/unpack utilities are designed for upper-triangular cells
        (the aligned form from align_cell()). Only 6 DOFs are preserved.
        """
        num_atoms = 10

        np.random.seed(42)
        velocities_np = np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1

        # Cell velocity must be upper-triangular (only 6 DOFs are preserved)
        # Upper-triangular form: non-zero elements at [0,0], [1,0], [1,1], [2,0], [2,1], [2,2]
        cell_vel_np = np.array(
            [
                [0.01, 0.0, 0.0],
                [0.005, 0.02, 0.0],
                [0.003, -0.01, 0.015],
            ],
            dtype=np_dtype,
        )

        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)
        cell_vel = make_cell(cell_vel_np, dtype_mat, device)

        # Pack
        extended = wp.empty(num_atoms + 2, dtype=dtype_vec, device=device)
        pack_velocities_with_cell(velocities, cell_vel, extended, device=device)

        wp.synchronize_device(device)
        assert extended.shape[0] == num_atoms + 2

        # Unpack
        vel_out = wp.empty(num_atoms, dtype=dtype_vec, device=device)
        cell_vel_out = wp.empty(1, dtype=dtype_mat, device=device)
        unpack_velocities_with_cell(
            extended, vel_out, cell_vel_out, num_atoms=num_atoms, device=device
        )

        wp.synchronize_device(device)
        vel_result = vel_out.numpy()
        cell_vel_result = cell_to_numpy(cell_vel_out)

        # Check velocities recovered
        np.testing.assert_allclose(vel_result, velocities_np, rtol=1e-5)

        # Check cell velocity recovered (upper-triangular form)
        np.testing.assert_allclose(cell_vel_result, cell_vel_np, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_velocities_batched_with_atom_ptr(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test velocities pack/unpack for batched systems using atom_ptr.

        This tests the proper batched usage pattern where:
        - Velocities are concatenated across all systems
        - Cell velocities are batched with shape (num_systems,)
        - pack_velocities_with_cell is called once with atom_ptr
        """
        num_systems = 3
        atoms_per_system = [8, 12, 6]
        total_atoms = sum(atoms_per_system)

        np.random.seed(42)

        # Create concatenated velocities and batched cell velocities
        velocities_list = []
        cell_vels_list = []
        for sys_id in range(num_systems):
            n_atoms = atoms_per_system[sys_id]
            vel_np = np.random.randn(n_atoms, 3).astype(np_dtype) * 0.1
            # Cell velocity must be upper-triangular (only 6 DOFs preserved)
            cell_vel_np = np.tril(np.random.randn(3, 3).astype(np_dtype) * 0.01)
            velocities_list.append(vel_np)
            cell_vels_list.append(cell_vel_np)

        # Concatenate velocities and stack cell velocities
        velocities_concat_np = np.vstack(velocities_list)
        cell_vels_np = np.stack(cell_vels_list, axis=0)  # Shape: (num_systems, 3, 3)

        # Create warp arrays
        velocities = wp.array(velocities_concat_np, dtype=dtype_vec, device=device)
        cell_vels = wp.array(cell_vels_np.reshape(-1), dtype=dtype_mat, device=device)

        # Create atom_ptr
        atom_ptr_np = np.array([0] + list(np.cumsum(atoms_per_system)), dtype=np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        # Extended atom_ptr
        ext_atom_ptr_arr = wp.empty(atom_ptr_np.shape[0], dtype=wp.int32, device=device)
        ext_atom_ptr = extend_atom_ptr(atom_ptr, ext_atom_ptr_arr, device=device)
        wp.synchronize_device(device)
        ext_atom_ptr_np = ext_atom_ptr.numpy()

        # Pack all systems at once using atom_ptr
        extended = wp.empty(
            total_atoms + 2 * num_systems, dtype=dtype_vec, device=device
        )
        pack_velocities_with_cell(
            velocities,
            cell_vels,
            extended,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
        )

        wp.synchronize_device(device)
        extended_np = extended.numpy()

        # Verify size
        expected_size = total_atoms + 2 * num_systems
        assert len(extended_np) == expected_size
        assert len(extended_np) == ext_atom_ptr_np[-1]

        # Test unpack roundtrip
        vel_out = wp.empty(total_atoms, dtype=dtype_vec, device=device)
        cell_vels_out = wp.empty(num_systems, dtype=dtype_mat, device=device)
        unpack_velocities_with_cell(
            extended,
            vel_out,
            cell_vels_out,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
        )

        wp.synchronize_device(device)

        # Verify velocities recovered
        np.testing.assert_allclose(vel_out.numpy(), velocities_concat_np, rtol=1e-5)

        # Verify cell velocities recovered
        cell_vels_out_np = cell_vels_out.numpy().reshape(num_systems, 3, 3)
        for sys_id in range(num_systems):
            np.testing.assert_allclose(
                cell_vels_out_np[sys_id], cell_vels_list[sys_id], rtol=1e-5
            )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_velocities_batched_with_batch_idx(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test velocities pack/unpack for batched systems using batch_idx.

        This tests using extend_batch_idx where each system is packed individually
        and then concatenated. The extended batch_idx maps elements to systems.
        """
        num_systems = 2
        atoms_per_system = [6, 10]
        total_atoms = sum(atoms_per_system)

        np.random.seed(42)

        # Create per-system data
        velocities_list = []
        cell_vels_list = []
        extended_list = []

        for sys_id in range(num_systems):
            n_atoms = atoms_per_system[sys_id]
            vel_np = np.random.randn(n_atoms, 3).astype(np_dtype) * 0.1
            # Cell velocity must be upper-triangular (only 6 DOFs preserved)
            cell_vel_np = np.tril(np.random.randn(3, 3).astype(np_dtype) * 0.01)

            velocities_list.append(vel_np)
            cell_vels_list.append(cell_vel_np)

            # Pack each system
            vel_wp = wp.array(vel_np, dtype=dtype_vec, device=device)
            cell_vel_wp = make_cell(cell_vel_np, dtype_mat, device)
            ext = wp.empty(n_atoms + 2, dtype=dtype_vec, device=device)
            pack_velocities_with_cell(vel_wp, cell_vel_wp, ext, device=device)

            wp.synchronize_device(device)
            extended_list.append(ext.numpy())

        # Concatenate extended arrays
        extended_concat = np.vstack(extended_list)

        # Create extended batch_idx
        original_batch_idx = np.concatenate(
            [
                np.full(atoms_per_system[i], i, dtype=np.int32)
                for i in range(num_systems)
            ]
        )
        batch_idx = wp.array(original_batch_idx, dtype=wp.int32, device=device)
        ext_batch_idx_arr = wp.empty(
            total_atoms + 2 * num_systems, dtype=wp.int32, device=device
        )
        ext_batch_idx = extend_batch_idx(
            batch_idx, total_atoms, num_systems, ext_batch_idx_arr, device=device
        )

        wp.synchronize_device(device)
        ext_batch_idx_np = ext_batch_idx.numpy()

        # Verify extended batch_idx size
        expected_ext_size = total_atoms + 2 * num_systems
        assert len(ext_batch_idx_np) == expected_ext_size
        assert len(extended_concat) == expected_ext_size

        # Verify extended batch_idx correctly identifies systems
        # Cell DOFs at end should have batch_idx = 0, 0, 1, 1
        cell_dofs_batch_idx = ext_batch_idx_np[total_atoms:]
        expected_cell_batch_idx = np.array([0, 0, 1, 1], dtype=np.int32)
        np.testing.assert_array_equal(cell_dofs_batch_idx, expected_cell_batch_idx)

        # Verify each system's data can be unpacked correctly
        for sys_id in range(num_systems):
            n_atoms = atoms_per_system[sys_id]
            n_extended = n_atoms + 2
            start = sum(atoms_per_system[:sys_id]) + 2 * sys_id
            end = start + n_extended

            sys_extended = wp.array(
                extended_concat[start:end], dtype=dtype_vec, device=device
            )
            vel_out = wp.empty(n_atoms, dtype=dtype_vec, device=device)
            cell_vel_out = wp.empty(1, dtype=dtype_mat, device=device)
            unpack_velocities_with_cell(
                sys_extended, vel_out, cell_vel_out, num_atoms=n_atoms, device=device
            )

            wp.synchronize_device(device)
            np.testing.assert_allclose(
                vel_out.numpy(), velocities_list[sys_id], rtol=1e-5
            )
            np.testing.assert_allclose(
                cell_to_numpy(cell_vel_out), cell_vels_list[sys_id], rtol=1e-5
            )

    # =========================================================================
    # Forces Pack Tests
    # =========================================================================

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_forces_pack_roundtrip(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test that force pack extracts correctly (uses position unpack for verification)."""
        num_atoms = 10

        np.random.seed(42)
        forces_np = np.random.randn(num_atoms, 3).astype(np_dtype)
        cell_force_np = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.5, 2.0, 0.0],
                [0.3, 0.4, 3.0],
            ],
            dtype=np_dtype,
        )

        forces = wp.array(forces_np, dtype=dtype_vec, device=device)
        cell_force = make_cell(cell_force_np, dtype_mat, device)

        # Pack
        extended = wp.empty(num_atoms + 2, dtype=dtype_vec, device=device)
        pack_forces_with_cell(forces, cell_force, extended, device=device)

        wp.synchronize_device(device)
        assert extended.shape[0] == num_atoms + 2

        # Use position unpack to verify (same format)
        forces_out = wp.empty(num_atoms, dtype=dtype_vec, device=device)
        cell_force_out = wp.empty(1, dtype=dtype_mat, device=device)
        unpack_positions_with_cell(
            extended, forces_out, cell_force_out, num_atoms=num_atoms, device=device
        )

        wp.synchronize_device(device)
        forces_result = forces_out.numpy()
        cell_force_result = cell_to_numpy(cell_force_out)

        # Check forces recovered
        np.testing.assert_allclose(forces_result, forces_np, rtol=1e-5)

        # Check cell force recovered
        np.testing.assert_allclose(cell_force_result, cell_force_np, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_forces_batched_with_atom_ptr(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test forces pack for batched systems using atom_ptr.

        This tests the proper batched usage pattern where:
        - Forces are concatenated across all systems
        - Cell forces are batched with shape (num_systems,)
        - pack_forces_with_cell is called once with atom_ptr
        """
        num_systems = 3
        atoms_per_system = [7, 9, 5]
        total_atoms = sum(atoms_per_system)

        np.random.seed(42)

        # Create concatenated forces and batched cell forces
        forces_list = []
        cell_forces_list = []
        for sys_id in range(num_systems):
            n_atoms = atoms_per_system[sys_id]
            forces_np = np.random.randn(n_atoms, 3).astype(np_dtype)
            # Upper-triangular cell force
            cell_force_np = np.tril(np.eye(3, dtype=np_dtype) * (sys_id + 1))
            forces_list.append(forces_np)
            cell_forces_list.append(cell_force_np)

        # Concatenate forces and stack cell forces
        forces_concat_np = np.vstack(forces_list)
        cell_forces_np = np.stack(
            cell_forces_list, axis=0
        )  # Shape: (num_systems, 3, 3)

        # Create warp arrays
        forces = wp.array(forces_concat_np, dtype=dtype_vec, device=device)
        cell_forces = wp.array(
            cell_forces_np.reshape(-1), dtype=dtype_mat, device=device
        )

        # Create atom_ptr
        atom_ptr_np = np.array([0] + list(np.cumsum(atoms_per_system)), dtype=np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        # Extended atom_ptr
        ext_atom_ptr_arr = wp.empty(atom_ptr_np.shape[0], dtype=wp.int32, device=device)
        ext_atom_ptr = extend_atom_ptr(atom_ptr, ext_atom_ptr_arr, device=device)
        wp.synchronize_device(device)
        ext_atom_ptr_np = ext_atom_ptr.numpy()

        # Pack all systems at once using atom_ptr
        extended = wp.empty(
            total_atoms + 2 * num_systems, dtype=dtype_vec, device=device
        )
        pack_forces_with_cell(
            forces,
            cell_forces,
            extended,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
        )

        wp.synchronize_device(device)
        extended_np = extended.numpy()

        # Verify size
        expected_size = total_atoms + 2 * num_systems
        assert len(extended_np) == expected_size
        assert len(extended_np) == ext_atom_ptr_np[-1]

        # Verify each system's data in the extended array
        for sys_id in range(num_systems):
            ext_start = ext_atom_ptr_np[sys_id]
            n_atoms = atoms_per_system[sys_id]

            # Check atomic forces
            sys_forces = extended_np[ext_start : ext_start + n_atoms]
            np.testing.assert_allclose(sys_forces, forces_list[sys_id], rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_forces_batched_with_batch_idx(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test forces pack for batched systems using batch_idx.

        This tests using extend_batch_idx where each system is packed individually
        and then concatenated. The extended batch_idx maps elements to systems.
        """
        num_systems = 3
        atoms_per_system = [5, 7, 4]
        total_atoms = sum(atoms_per_system)

        np.random.seed(42)

        # Create per-system data
        forces_list = []
        cell_forces_list = []
        extended_list = []

        for sys_id in range(num_systems):
            n_atoms = atoms_per_system[sys_id]
            forces_np = np.random.randn(n_atoms, 3).astype(np_dtype)
            # Upper-triangular cell force
            cell_force_np = np.tril(np.eye(3, dtype=np_dtype) * (sys_id + 1))

            forces_list.append(forces_np)
            cell_forces_list.append(cell_force_np)

            # Pack each system
            forces_wp = wp.array(forces_np, dtype=dtype_vec, device=device)
            cell_force_wp = make_cell(cell_force_np, dtype_mat, device)
            ext = wp.empty(n_atoms + 2, dtype=dtype_vec, device=device)
            pack_forces_with_cell(forces_wp, cell_force_wp, ext, device=device)

            wp.synchronize_device(device)
            extended_list.append(ext.numpy())

        # Concatenate extended arrays
        extended_concat = np.vstack(extended_list)

        # Create extended batch_idx
        original_batch_idx = np.concatenate(
            [
                np.full(atoms_per_system[i], i, dtype=np.int32)
                for i in range(num_systems)
            ]
        )
        batch_idx = wp.array(original_batch_idx, dtype=wp.int32, device=device)
        ext_batch_idx_arr = wp.empty(
            total_atoms + 2 * num_systems, dtype=wp.int32, device=device
        )
        ext_batch_idx = extend_batch_idx(
            batch_idx, total_atoms, num_systems, ext_batch_idx_arr, device=device
        )

        wp.synchronize_device(device)
        ext_batch_idx_np = ext_batch_idx.numpy()

        # Verify extended batch_idx size
        expected_ext_size = total_atoms + 2 * num_systems
        assert len(ext_batch_idx_np) == expected_ext_size
        assert len(extended_concat) == expected_ext_size

        # Verify extended batch_idx correctly identifies systems
        # Cell DOFs at end should have batch_idx = 0, 0, 1, 1, 2, 2
        cell_dofs_batch_idx = ext_batch_idx_np[total_atoms:]
        expected_cell_batch_idx = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        np.testing.assert_array_equal(cell_dofs_batch_idx, expected_cell_batch_idx)

        # Verify each system's data can be accessed correctly via extended array
        for sys_id in range(num_systems):
            n_atoms = atoms_per_system[sys_id]
            n_extended = n_atoms + 2
            start = sum(atoms_per_system[:sys_id]) + 2 * sys_id
            end = start + n_extended

            sys_extended = wp.array(
                extended_concat[start:end], dtype=dtype_vec, device=device
            )
            # Use position unpack to verify (forces use same format)
            forces_out = wp.empty(n_atoms, dtype=dtype_vec, device=device)
            cell_force_out = wp.empty(1, dtype=dtype_mat, device=device)
            unpack_positions_with_cell(
                sys_extended,
                forces_out,
                cell_force_out,
                num_atoms=n_atoms,
                device=device,
            )

            wp.synchronize_device(device)
            np.testing.assert_allclose(
                forces_out.numpy(), forces_list[sys_id], rtol=1e-5
            )
            np.testing.assert_allclose(
                cell_to_numpy(cell_force_out), cell_forces_list[sys_id], rtol=1e-5
            )

    # =========================================================================
    # Masses Pack Tests
    # =========================================================================

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_masses_pack_roundtrip(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test packing masses with cell mass and verify values."""
        num_atoms = 10
        cell_mass = 100.0
        atomic_mass = 12.0  # Carbon-like

        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype) * atomic_mass,
            dtype=dtype_scalar,
            device=device,
        )
        cell_mass_arr = wp.array([cell_mass], dtype=dtype_scalar, device=device)

        extended = wp.empty(num_atoms + 2, dtype=dtype_scalar, device=device)
        pack_masses_with_cell(masses, cell_mass_arr, extended, device=device)

        wp.synchronize_device(device)
        result = extended.numpy()

        assert result.shape[0] == num_atoms + 2

        # Atomic masses
        np.testing.assert_allclose(result[:num_atoms], atomic_mass, rtol=1e-5)

        # Cell DOF masses
        np.testing.assert_allclose(result[num_atoms:], cell_mass, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_masses_batched_with_atom_ptr(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test masses pack for batched systems using atom_ptr.

        This tests the proper batched usage pattern where:
        - Masses are concatenated across all systems
        - Cell mass is applied to all systems
        - pack_masses_with_cell is called once with atom_ptr
        """
        num_systems = 3
        atoms_per_system = [6, 8, 4]
        total_atoms = sum(atoms_per_system)
        cell_mass = 75.0
        atomic_mass = 14.0  # Nitrogen-like

        np.random.seed(42)

        # Create concatenated masses
        masses_list = []
        for sys_id in range(num_systems):
            n_atoms = atoms_per_system[sys_id]
            masses_np = np.ones(n_atoms, dtype=np_dtype) * atomic_mass
            masses_list.append(masses_np)

        masses_concat_np = np.concatenate(masses_list)

        # Create warp arrays
        masses = wp.array(masses_concat_np, dtype=dtype_scalar, device=device)

        # Create atom_ptr
        atom_ptr_np = np.array([0] + list(np.cumsum(atoms_per_system)), dtype=np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        # Extended atom_ptr
        ext_atom_ptr_arr = wp.empty(atom_ptr_np.shape[0], dtype=wp.int32, device=device)
        ext_atom_ptr = extend_atom_ptr(atom_ptr, ext_atom_ptr_arr, device=device)
        wp.synchronize_device(device)
        ext_atom_ptr_np = ext_atom_ptr.numpy()

        # Pack all systems at once using atom_ptr
        cell_mass_arr = wp.array(
            np.full(num_systems, cell_mass, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        extended = wp.empty(
            total_atoms + 2 * num_systems, dtype=dtype_scalar, device=device
        )
        pack_masses_with_cell(
            masses,
            cell_mass_arr,
            extended,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
        )

        wp.synchronize_device(device)
        extended_np = extended.numpy()

        # Verify size
        expected_size = total_atoms + 2 * num_systems
        assert len(extended_np) == expected_size
        assert len(extended_np) == ext_atom_ptr_np[-1]

        # Verify each system's masses in the extended array
        for sys_id in range(num_systems):
            ext_start = ext_atom_ptr_np[sys_id]
            n_atoms = atoms_per_system[sys_id]

            # Check atomic masses
            sys_masses = extended_np[ext_start : ext_start + n_atoms]
            np.testing.assert_allclose(sys_masses, atomic_mass, rtol=1e-5)

            # Check cell DOF masses
            cell_dof_masses = extended_np[ext_start + n_atoms : ext_start + n_atoms + 2]
            np.testing.assert_allclose(cell_dof_masses, cell_mass, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_masses_batched_with_batch_idx(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test masses pack for batched systems using batch_idx.

        This tests using extend_batch_idx where each system is packed individually
        and then concatenated. The extended batch_idx maps elements to systems.
        """
        num_systems = 2
        atoms_per_system = [8, 12]
        total_atoms = sum(atoms_per_system)
        cell_mass = 50.0

        np.random.seed(42)

        # Create per-system data with different masses per system
        masses_list = []
        extended_list = []

        for sys_id in range(num_systems):
            n_atoms = atoms_per_system[sys_id]
            # Different masses per system
            mass_val = 12.0 + sys_id * 4.0  # 12, 16 (C, O-like)
            masses_np = np.ones(n_atoms, dtype=np_dtype) * mass_val

            masses_list.append(masses_np)

            # Pack each system
            masses_wp = wp.array(masses_np, dtype=dtype_scalar, device=device)
            cell_mass_arr = wp.array([cell_mass], dtype=dtype_scalar, device=device)
            ext = wp.empty(n_atoms + 2, dtype=dtype_scalar, device=device)
            pack_masses_with_cell(masses_wp, cell_mass_arr, ext, device=device)

            wp.synchronize_device(device)
            ext_np = ext.numpy()
            extended_list.append(ext_np)

            # Verify this system's masses in isolation
            np.testing.assert_allclose(ext_np[:n_atoms], mass_val, rtol=1e-5)
            np.testing.assert_allclose(ext_np[n_atoms:], cell_mass, rtol=1e-5)

        # Concatenate extended arrays
        extended_concat = np.concatenate(extended_list)

        # Create extended batch_idx
        original_batch_idx = np.concatenate(
            [
                np.full(atoms_per_system[i], i, dtype=np.int32)
                for i in range(num_systems)
            ]
        )
        batch_idx = wp.array(original_batch_idx, dtype=wp.int32, device=device)
        ext_batch_idx_arr = wp.empty(
            total_atoms + 2 * num_systems, dtype=wp.int32, device=device
        )
        ext_batch_idx = extend_batch_idx(
            batch_idx, total_atoms, num_systems, ext_batch_idx_arr, device=device
        )

        wp.synchronize_device(device)
        ext_batch_idx_np = ext_batch_idx.numpy()

        # Verify extended batch_idx size
        expected_ext_size = total_atoms + 2 * num_systems
        assert len(ext_batch_idx_np) == expected_ext_size
        assert len(extended_concat) == expected_ext_size

        # Verify extended batch_idx correctly identifies systems
        # Cell DOFs at end should have batch_idx = 0, 0, 1, 1
        cell_dofs_batch_idx = ext_batch_idx_np[total_atoms:]
        expected_cell_batch_idx = np.array([0, 0, 1, 1], dtype=np.int32)
        np.testing.assert_array_equal(cell_dofs_batch_idx, expected_cell_batch_idx)

        # Verify each system's data can be accessed via extended indices
        for sys_id in range(num_systems):
            n_atoms = atoms_per_system[sys_id]
            n_extended = n_atoms + 2
            start = sum(atoms_per_system[:sys_id]) + 2 * sys_id
            end = start + n_extended

            sys_extended = extended_concat[start:end]
            expected_mass_val = 12.0 + sys_id * 4.0

            # Check atomic masses
            np.testing.assert_allclose(
                sys_extended[:n_atoms], expected_mass_val, rtol=1e-5
            )

            # Check cell DOF masses
            np.testing.assert_allclose(sys_extended[n_atoms:], cell_mass, rtol=1e-5)

    # =========================================================================
    # Combined Pack/Unpack Workflow Tests
    # =========================================================================

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_full_pack_workflow_single_system(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test complete pack workflow for single system: positions, velocities, forces, masses."""
        num_atoms = 15
        cell_mass = 100.0

        np.random.seed(42)

        # Create all data
        positions_np = np.random.randn(num_atoms, 3).astype(np_dtype)
        velocities_np = np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1
        forces_np = np.random.randn(num_atoms, 3).astype(np_dtype)
        masses_np = np.ones(num_atoms, dtype=np_dtype) * 12.0

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cell_vel_np = np.zeros((3, 3), dtype=np_dtype)
        cell_force_np = np.eye(3, dtype=np_dtype) * 0.1

        # Pack all
        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)
        forces = wp.array(forces_np, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        cell = make_cell(cell_np, dtype_mat, device)
        cell_vel = make_cell(cell_vel_np, dtype_mat, device)
        cell_force = make_cell(cell_force_np, dtype_mat, device)

        ext_pos = wp.empty(num_atoms + 2, dtype=dtype_vec, device=device)
        pack_positions_with_cell(positions, cell, ext_pos, device=device)
        ext_vel = wp.empty(num_atoms + 2, dtype=dtype_vec, device=device)
        pack_velocities_with_cell(velocities, cell_vel, ext_vel, device=device)
        ext_forces = wp.empty(num_atoms + 2, dtype=dtype_vec, device=device)
        pack_forces_with_cell(forces, cell_force, ext_forces, device=device)
        cell_mass_arr = wp.array([cell_mass], dtype=dtype_scalar, device=device)
        ext_masses = wp.empty(num_atoms + 2, dtype=dtype_scalar, device=device)
        pack_masses_with_cell(masses, cell_mass_arr, ext_masses, device=device)

        wp.synchronize_device(device)

        # All should have same extended size
        expected_size = num_atoms + 2
        assert ext_pos.shape[0] == expected_size
        assert ext_vel.shape[0] == expected_size
        assert ext_forces.shape[0] == expected_size
        assert ext_masses.shape[0] == expected_size

        # Unpack and verify positions and cell
        pos_out = wp.empty(num_atoms, dtype=dtype_vec, device=device)
        cell_out = wp.empty(1, dtype=dtype_mat, device=device)
        unpack_positions_with_cell(
            ext_pos, pos_out, cell_out, num_atoms=num_atoms, device=device
        )
        vel_out = wp.empty(num_atoms, dtype=dtype_vec, device=device)
        cell_vel_out = wp.empty(1, dtype=dtype_mat, device=device)
        unpack_velocities_with_cell(
            ext_vel, vel_out, cell_vel_out, num_atoms=num_atoms, device=device
        )

        wp.synchronize_device(device)

        np.testing.assert_allclose(pos_out.numpy(), positions_np, rtol=1e-5)
        np.testing.assert_allclose(vel_out.numpy(), velocities_np, rtol=1e-5)
        np.testing.assert_allclose(cell_to_numpy(cell_out), cell_np, rtol=1e-5)
        np.testing.assert_allclose(cell_to_numpy(cell_vel_out), cell_vel_np, rtol=1e-5)


class TestBatchedPackUnpackParity:
    """Test batched pack/unpack kernel parity and edge cases."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_positions_pack_batched_parity(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Atom-parallel pack produces identical results to legacy batched pack."""
        num_systems = 3
        atoms_per_system = [4, 6, 5]
        total_atoms = sum(atoms_per_system)

        np.random.seed(200)
        positions_np = np.random.randn(total_atoms, 3).astype(np_dtype)
        cells_np = np.zeros((num_systems, 3, 3), dtype=np_dtype)
        for i in range(num_systems):
            cells_np[i] = np.eye(3, dtype=np_dtype) * (8.0 + i)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        cells = wp.array(cells_np.reshape(-1), dtype=dtype_mat, device=device)

        # Create atom_ptr, ext_atom_ptr, batch_idx
        atom_ptr_np = np.array([0] + list(np.cumsum(atoms_per_system)), dtype=np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)
        ext_atom_ptr = wp.empty(num_systems + 1, dtype=wp.int32, device=device)
        extend_atom_ptr(atom_ptr, ext_atom_ptr, device=device)

        batch_idx_np = np.repeat(
            np.arange(num_systems, dtype=np.int32),
            atoms_per_system,
        )
        batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)

        ext_size = total_atoms + 2 * num_systems

        # Batched pack (auto-compute batch_idx)
        ext_legacy = wp.empty(ext_size, dtype=dtype_vec, device=device)
        pack_positions_with_cell(
            positions,
            cells,
            ext_legacy,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
        )

        # Batched pack (pre-computed batch_idx)
        ext_parallel = wp.empty(ext_size, dtype=dtype_vec, device=device)
        pack_positions_with_cell(
            positions,
            cells,
            ext_parallel,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
            batch_idx=batch_idx,
        )

        wp.synchronize_device(device)
        np.testing.assert_allclose(
            ext_parallel.numpy(),
            ext_legacy.numpy(),
            rtol=1e-6,
            err_msg="Atom-parallel pack positions does not match legacy batched",
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_positions_unpack_batched_parity(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Atom-parallel unpack produces identical results to legacy batched unpack."""
        num_systems = 3
        atoms_per_system = [4, 6, 5]
        total_atoms = sum(atoms_per_system)

        np.random.seed(201)
        positions_np = np.random.randn(total_atoms, 3).astype(np_dtype)
        cells_np = np.zeros((num_systems, 3, 3), dtype=np_dtype)
        for i in range(num_systems):
            cells_np[i] = np.eye(3, dtype=np_dtype) * (8.0 + i)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        cells = wp.array(cells_np.reshape(-1), dtype=dtype_mat, device=device)

        atom_ptr_np = np.array([0] + list(np.cumsum(atoms_per_system)), dtype=np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)
        ext_atom_ptr = wp.empty(num_systems + 1, dtype=wp.int32, device=device)
        extend_atom_ptr(atom_ptr, ext_atom_ptr, device=device)

        batch_idx_np = np.repeat(
            np.arange(num_systems, dtype=np.int32),
            atoms_per_system,
        )
        batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)

        ext_size = total_atoms + 2 * num_systems
        extended = wp.empty(ext_size, dtype=dtype_vec, device=device)
        pack_positions_with_cell(
            positions,
            cells,
            extended,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
        )

        # Batched unpack (auto-compute batch_idx)
        pos_legacy = wp.empty(total_atoms, dtype=dtype_vec, device=device)
        cell_legacy = wp.empty(num_systems, dtype=dtype_mat, device=device)
        unpack_positions_with_cell(
            extended,
            pos_legacy,
            cell_legacy,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
        )

        # Batched unpack (pre-computed batch_idx)
        pos_parallel = wp.empty(total_atoms, dtype=dtype_vec, device=device)
        cell_parallel = wp.empty(num_systems, dtype=dtype_mat, device=device)
        unpack_positions_with_cell(
            extended,
            pos_parallel,
            cell_parallel,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
            batch_idx=batch_idx,
        )

        wp.synchronize_device(device)
        np.testing.assert_allclose(
            pos_parallel.numpy(),
            pos_legacy.numpy(),
            rtol=1e-6,
            err_msg="Atom-parallel unpack positions mismatch",
        )
        np.testing.assert_allclose(
            cell_parallel.numpy().reshape(-1),
            cell_legacy.numpy().reshape(-1),
            rtol=1e-6,
            err_msg="Atom-parallel unpack cells mismatch",
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_forces_pack_batched_parity(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Atom-parallel force pack produces identical results to legacy batched pack."""
        num_systems = 3
        atoms_per_system = [4, 6, 5]
        total_atoms = sum(atoms_per_system)

        np.random.seed(202)
        forces_np = np.random.randn(total_atoms, 3).astype(np_dtype)
        cell_forces_np = np.zeros((num_systems, 3, 3), dtype=np_dtype)
        for i in range(num_systems):
            cell_forces_np[i] = np.random.randn(3, 3).astype(np_dtype) * 0.1

        forces = wp.array(forces_np, dtype=dtype_vec, device=device)
        cell_forces = wp.array(
            cell_forces_np.reshape(-1), dtype=dtype_mat, device=device
        )

        atom_ptr_np = np.array([0] + list(np.cumsum(atoms_per_system)), dtype=np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)
        ext_atom_ptr = wp.empty(num_systems + 1, dtype=wp.int32, device=device)
        extend_atom_ptr(atom_ptr, ext_atom_ptr, device=device)

        batch_idx_np = np.repeat(
            np.arange(num_systems, dtype=np.int32),
            atoms_per_system,
        )
        batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)

        ext_size = total_atoms + 2 * num_systems

        # Batched pack (auto-compute batch_idx)
        ext_legacy = wp.empty(ext_size, dtype=dtype_vec, device=device)
        pack_forces_with_cell(
            forces,
            cell_forces,
            ext_legacy,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
        )

        # Batched pack (pre-computed batch_idx)
        ext_parallel = wp.empty(ext_size, dtype=dtype_vec, device=device)
        pack_forces_with_cell(
            forces,
            cell_forces,
            ext_parallel,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
            batch_idx=batch_idx,
        )

        wp.synchronize_device(device)
        np.testing.assert_allclose(
            ext_parallel.numpy(),
            ext_legacy.numpy(),
            rtol=1e-6,
            err_msg="Pre-computed batch_idx pack forces does not match auto-computed",
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_pack_unpack_roundtrip_uneven_systems(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Pack/unpack roundtrip with highly uneven system sizes (10, 100, 1000)."""
        atoms_per_system = [10, 100, 1000]
        num_systems = len(atoms_per_system)
        total_atoms = sum(atoms_per_system)

        np.random.seed(203)
        positions_np = np.random.randn(total_atoms, 3).astype(np_dtype)
        cells_np = np.zeros((num_systems, 3, 3), dtype=np_dtype)
        for i in range(num_systems):
            cells_np[i] = np.diag(np.random.uniform(5, 15, 3).astype(np_dtype))

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        cells = wp.array(cells_np.reshape(-1), dtype=dtype_mat, device=device)

        atom_ptr_np = np.array([0] + list(np.cumsum(atoms_per_system)), dtype=np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)
        ext_atom_ptr = wp.empty(num_systems + 1, dtype=wp.int32, device=device)
        extend_atom_ptr(atom_ptr, ext_atom_ptr, device=device)

        batch_idx_np = np.repeat(
            np.arange(num_systems, dtype=np.int32),
            atoms_per_system,
        )
        batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)

        ext_size = total_atoms + 2 * num_systems

        # Pack with batched kernels
        ext_packed = wp.empty(ext_size, dtype=dtype_vec, device=device)
        pack_positions_with_cell(
            positions,
            cells,
            ext_packed,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
            batch_idx=batch_idx,
        )

        # Unpack with batched kernels
        pos_out = wp.empty(total_atoms, dtype=dtype_vec, device=device)
        cell_out = wp.empty(num_systems, dtype=dtype_mat, device=device)
        unpack_positions_with_cell(
            ext_packed,
            pos_out,
            cell_out,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
            batch_idx=batch_idx,
        )

        wp.synchronize_device(device)
        np.testing.assert_allclose(
            pos_out.numpy(),
            positions_np,
            rtol=1e-5,
            err_msg="Positions not recovered after pack/unpack roundtrip",
        )
        recovered_cells = cell_out.numpy().reshape(num_systems, 3, 3)
        np.testing.assert_allclose(
            recovered_cells,
            cells_np,
            rtol=1e-5,
            err_msg="Cells not recovered after pack/unpack roundtrip",
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_large_n_small_m_stress(self, device):
        """Stress test: N=100000, M=1 with batched kernels."""
        N = 100000
        M = 1
        np_dtype = np.float64
        dtype_vec = wp.vec3d
        dtype_mat = wp.mat33d

        np.random.seed(204)
        positions_np = np.random.randn(N, 3).astype(np_dtype)
        cells_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype).reshape(1, 3, 3)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        cells = wp.array(cells_np.reshape(-1), dtype=dtype_mat, device=device)

        atom_ptr_np = np.array([0, N], dtype=np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)
        ext_atom_ptr = wp.empty(M + 1, dtype=wp.int32, device=device)
        extend_atom_ptr(atom_ptr, ext_atom_ptr, device=device)

        batch_idx_np = np.zeros(N, dtype=np.int32)
        batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)

        ext_size = N + 2 * M

        # Pack
        ext_packed = wp.empty(ext_size, dtype=dtype_vec, device=device)
        pack_positions_with_cell(
            positions,
            cells,
            ext_packed,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
            batch_idx=batch_idx,
        )

        # Unpack
        pos_out = wp.empty(N, dtype=dtype_vec, device=device)
        cell_out = wp.empty(M, dtype=dtype_mat, device=device)
        unpack_positions_with_cell(
            ext_packed,
            pos_out,
            cell_out,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
            batch_idx=batch_idx,
        )

        wp.synchronize_device(device)
        np.testing.assert_allclose(
            pos_out.numpy(),
            positions_np,
            rtol=1e-10,
            err_msg="Large-N/small-M pack/unpack mismatch",
        )
        recovered_cells = cell_out.numpy().reshape(M, 3, 3)
        np.testing.assert_allclose(
            recovered_cells,
            cells_np,
            rtol=1e-10,
            err_msg="Large-N/small-M cell mismatch",
        )


class TestStressToCellForce:
    """Test stress to cell force conversion."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_stress_to_cell_force_runs(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test that stress_to_cell_force executes."""
        # Hydrostatic stress (pressure)
        pressure = 1.0  # GPa-like units
        stress_np = np.eye(3, dtype=np_dtype) * pressure
        stress = make_cell(stress_np, dtype_mat, device)

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cell = make_cell(cell_np, dtype_mat, device)

        volume = wp.empty(1, dtype=dtype_scalar, device=device)
        compute_cell_volume(cell, volume, device=device)
        cell_force = wp.empty(1, dtype=dtype_mat, device=device)
        stress_to_cell_force(stress, cell, volume, cell_force, device=device)

        wp.synchronize_device(device)
        assert cell_force.shape[0] == 1

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_stress_to_cell_force_keep_aligned(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test that keep_aligned=True zeros upper off-diagonal elements."""
        # Non-diagonal stress
        stress_np = np.array(
            [
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.4],
                [0.3, 0.4, 1.0],
            ],
            dtype=np_dtype,
        )
        stress = make_cell(stress_np, dtype_mat, device)

        cell_np = np.array(
            [
                [10.0, 0.0, 0.0],
                [2.0, 9.0, 0.0],
                [1.0, 2.0, 8.0],
            ],
            dtype=np_dtype,
        )
        cell = make_cell(cell_np, dtype_mat, device)

        volume = wp.empty(1, dtype=dtype_scalar, device=device)
        compute_cell_volume(cell, volume, device=device)
        cell_force = wp.empty(1, dtype=dtype_mat, device=device)
        stress_to_cell_force(
            stress, cell, volume, cell_force, keep_aligned=True, device=device
        )

        wp.synchronize_device(device)
        result = cell_to_numpy(cell_force)

        # Upper off-diagonal should be zero
        np.testing.assert_allclose(result[0, 1], 0.0, atol=1e-5)
        np.testing.assert_allclose(result[0, 2], 0.0, atol=1e-5)
        np.testing.assert_allclose(result[1, 2], 0.0, atol=1e-5)


# ==============================================================================
# Cell Filter Device Inference and Edge Cases
# ==============================================================================


class TestCellFilterDeviceInference:
    """Test device inference for cell filter utilities."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_pack_positions_device_inference(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test pack_positions_with_cell infers device from positions."""
        num_atoms = 5
        np.random.seed(42)

        positions_np = np.random.randn(num_atoms, 3).astype(np_dtype)
        cell_np = np.eye(3, dtype=np_dtype) * 10.0

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        cell = make_cell(cell_np, dtype_mat, device)

        # Don't pass device
        extended = wp.empty(num_atoms + 2, dtype=dtype_vec, device=device)
        pack_positions_with_cell(positions, cell, extended)

        wp.synchronize_device(device)
        assert extended.device == device
        assert extended.shape[0] == num_atoms + 2

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_unpack_positions_device_inference(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test unpack_positions_with_cell infers device from extended."""
        num_atoms = 5

        # Create extended array
        extended_np = np.random.randn(num_atoms + 2, 3).astype(np_dtype)
        extended = wp.array(extended_np, dtype=dtype_vec, device=device)

        # Don't pass device
        positions = wp.empty(num_atoms, dtype=dtype_vec, device=device)
        cell = wp.empty(1, dtype=dtype_mat, device=device)
        unpack_positions_with_cell(extended, positions, cell, num_atoms=num_atoms)

        wp.synchronize_device(device)
        assert positions.device == device
        assert cell.device == device

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_align_cell_device_inference(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test align_cell infers device from positions."""
        num_atoms = 10

        positions_np = np.random.randn(num_atoms, 3).astype(np_dtype)
        cell_np = np.array(
            [[10.0, 0.0, 0.0], [2.0, 9.0, 0.0], [1.0, 2.0, 8.0]], dtype=np_dtype
        )

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        cell = make_cell(cell_np, dtype_mat, device)
        transform = wp.empty(1, dtype=dtype_mat, device=device)

        # Don't pass device
        pos_out, cell_out = align_cell(positions, cell, transform)

        wp.synchronize_device(device)
        assert pos_out.device == device
        assert cell_out.device == device

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_extend_batch_idx_device_inference(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test extend_batch_idx infers device from batch_idx."""
        num_atoms = 15
        num_systems = 2

        batch_idx_np = np.array([0] * 8 + [1] * 7, dtype=np.int32)
        batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)
        extended_batch_idx = wp.empty(
            num_atoms + 2 * num_systems, dtype=wp.int32, device=device
        )

        # Don't pass device
        extended = extend_batch_idx(
            batch_idx, num_atoms, num_systems, extended_batch_idx
        )

        wp.synchronize_device(device)
        assert extended.device == device
        assert extended.shape[0] == num_atoms + 2 * num_systems

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_extend_atom_ptr_device_inference(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test extend_atom_ptr infers device from atom_ptr."""
        atom_ptr_np = np.array([0, 8, 15], dtype=np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)
        extended_atom_ptr = wp.empty(
            atom_ptr_np.shape[0], dtype=wp.int32, device=device
        )

        # Don't pass device
        extended = extend_atom_ptr(atom_ptr, extended_atom_ptr)

        wp.synchronize_device(device)
        assert extended.device == device


class TestCellFilterBatchedModeEdgeCases:
    """Test edge cases and additional batched mode scenarios."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_unpack_positions_batched_with_preallocated_outputs(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test batched unpack with pre-allocated outputs."""
        num_systems = 2
        atoms_per_system = [5, 8]
        total_atoms = sum(atoms_per_system)
        ext_size = total_atoms + 2 * num_systems

        # Create extended array
        np.random.seed(42)
        extended_np = np.random.randn(ext_size, 3).astype(np_dtype)
        extended = wp.array(extended_np, dtype=dtype_vec, device=device)

        # Create atom_ptr
        atom_ptr_np = np.array([0, atoms_per_system[0], total_atoms], dtype=np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)
        ext_atom_ptr_arr = wp.empty(atom_ptr_np.shape[0], dtype=wp.int32, device=device)
        ext_atom_ptr = extend_atom_ptr(atom_ptr, ext_atom_ptr_arr, device=device)

        # Pre-allocate outputs
        if dtype_mat == wp.mat33f:
            mat_dtype = wp.mat33f
        else:
            mat_dtype = wp.mat33d

        positions = wp.empty(total_atoms, dtype=dtype_vec, device=device)
        cell = wp.empty(num_systems, dtype=mat_dtype, device=device)

        # Unpack with pre-allocated outputs
        pos_out, cell_out = unpack_positions_with_cell(
            extended,
            positions,
            cell,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            device=device,
        )

        wp.synchronize_device(device)

        assert pos_out is positions
        assert cell_out is cell

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_align_cell_batched_with_batch_idx(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test align_cell with batched systems using batch_idx."""
        num_atoms_per_system = 5
        num_systems = 2
        total_atoms = num_atoms_per_system * num_systems

        np.random.seed(42)
        positions_np = np.random.randn(total_atoms, 3).astype(np_dtype)

        # Create cells that need alignment (off-diagonal elements)
        cells_np = np.array(
            [
                [[10.0, 0.0, 0.0], [2.0, 9.0, 0.0], [1.0, 2.0, 8.0]],
                [[11.0, 0.0, 0.0], [1.5, 10.0, 0.0], [0.5, 1.5, 9.0]],
            ],
            dtype=np_dtype,
        )

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        cell = wp.array(cells_np, dtype=dtype_mat, device=device)

        # Create batch_idx
        batch_idx_np = np.repeat(np.arange(num_systems), num_atoms_per_system).astype(
            np.int32
        )
        batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)
        transform = wp.empty(num_systems, dtype=dtype_mat, device=device)

        # Align
        pos_out, cell_out = align_cell(
            positions, cell, transform, batch_idx=batch_idx, device=device
        )

        wp.synchronize_device(device)

        # Cell should be upper triangular
        cell_result = cell_out.numpy()
        for s in range(num_systems):
            np.testing.assert_allclose(cell_result[s, 0, 1], 0.0, atol=1e-5)
            np.testing.assert_allclose(cell_result[s, 0, 2], 0.0, atol=1e-5)
            np.testing.assert_allclose(cell_result[s, 1, 2], 0.0, atol=1e-5)


# ==============================================================================
# Integration Tests - Variable Cell Optimization
# ==============================================================================


class TestVariableCellOptimization:
    """Integration tests for variable-cell optimization workflow."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_extended_fire_step(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test fire_step with extended arrays (atomic + cell DOFs)."""
        num_atoms = 10
        num_extended = num_atoms + 2  # + 2 for cell DOFs
        num_systems = 1

        # Create extended arrays
        positions = wp.array(
            np.random.randn(num_extended, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.zeros(num_extended, dtype=dtype_vec, device=device)
        forces = wp.array(
            np.random.randn(num_extended, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_extended, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        atom_ptr = wp.array(
            np.array([0, num_extended], dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )

        params = make_fire_params(num_systems, dtype_scalar, device, np_dtype)

        # Should work with extended arrays
        fire_step(
            positions=positions,
            velocities=velocities,
            forces=forces,
            masses=masses,
            atom_ptr=atom_ptr,
            **params,
        )

        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_update_with_extended_arrays(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test fire_update with extended arrays for variable-cell optimization."""
        num_atoms = 10
        num_extended = num_atoms + 2
        num_systems = 1

        velocities = wp.array(
            np.random.randn(num_extended, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_extended, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        atom_ptr = wp.array(
            np.array([0, num_extended], dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )

        params = make_fire_params(num_systems, dtype_scalar, device, np_dtype)
        del params["maxstep"]
        del params["uphill_flag"]

        fire_update(
            velocities=velocities,
            forces=forces,
            atom_ptr=atom_ptr,
            **params,
        )

        wp.synchronize_device(device)

    # =========================================================================
    # Physical Tests: Cell-Only Optimization
    # =========================================================================

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_cell_only_harmonic_converges(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test that cell-only optimization converges to target cell.

        This tests the stress_to_cell_force + FIRE optimization workflow
        on a harmonic cell potential where E = 0.5*k*||H - H_target||^2.
        The cell should converge to H_target.
        """
        num_extended = 2  # Just cell DOFs (no atoms)
        num_systems = 1
        k_cell = 1.0

        # Target cell: cubic 10 Angstrom
        target_cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        target_cell = make_cell(target_cell_np, dtype_mat, device)

        # Initial cell: distorted
        initial_cell_np = np.array(
            [
                [12.0, 0.0, 0.0],
                [1.0, 9.0, 0.0],
                [0.5, -0.5, 11.0],
            ],
            dtype=np_dtype,
        )
        _ = make_cell(initial_cell_np, dtype_mat, device)

        # Pack cell into extended position array (no atoms)
        # For cell-only, we manually create the 2-element extended array
        v1 = np.array(
            [initial_cell_np[0, 0], initial_cell_np[1, 0], initial_cell_np[2, 0]],
            dtype=np_dtype,
        )
        v2 = np.array(
            [initial_cell_np[1, 1], initial_cell_np[2, 1], initial_cell_np[2, 2]],
            dtype=np_dtype,
        )
        ext_pos_np = np.vstack([v1, v2])

        ext_positions = wp.array(ext_pos_np, dtype=dtype_vec, device=device)
        ext_velocities = wp.zeros(num_extended, dtype=dtype_vec, device=device)
        ext_forces = wp.empty(num_extended, dtype=dtype_vec, device=device)
        ext_masses = wp.array(
            np.ones(num_extended, dtype=np_dtype) * 100.0,
            dtype=dtype_scalar,
            device=device,
        )

        atom_ptr = wp.array(
            np.array([0, num_extended], dtype=np.int32), dtype=wp.int32, device=device
        )
        params = make_fire_params(num_systems, dtype_scalar, device, np_dtype)

        # Optimization loop
        max_steps = 500
        force_tol = 1e-4

        for step in range(max_steps):
            wp.synchronize_device(device)

            # Reconstruct cell from extended positions
            ext_pos_current = ext_positions.numpy()
            current_cell_np = np.array(
                [
                    [ext_pos_current[0, 0], 0.0, 0.0],
                    [ext_pos_current[0, 1], ext_pos_current[1, 0], 0.0],
                    [
                        ext_pos_current[0, 2],
                        ext_pos_current[1, 1],
                        ext_pos_current[1, 2],
                    ],
                ],
                dtype=np_dtype,
            )
            cell_current = make_cell(current_cell_np, dtype_mat, device)

            # Compute stress from harmonic cell potential
            stress = compute_harmonic_cell_stress(
                cell_current, target_cell, k_cell, device=device
            )

            # Convert stress to cell force
            volume = wp.empty(1, dtype=dtype_scalar, device=device)
            compute_cell_volume(cell_current, volume, device=device)
            cell_force = wp.empty(1, dtype=dtype_mat, device=device)
            stress_to_cell_force(
                stress,
                cell_current,
                volume,
                cell_force,
                keep_aligned=True,
                device=device,
            )

            # Pack cell force into extended force array
            wp.synchronize_device(device)
            cf = cell_to_numpy(cell_force)
            f1 = np.array([cf[0, 0], cf[1, 0], cf[2, 0]], dtype=np_dtype)
            f2 = np.array([cf[1, 1], cf[2, 1], cf[2, 2]], dtype=np_dtype)
            ext_forces_np = np.vstack([f1, f2])
            ext_forces = wp.array(ext_forces_np, dtype=dtype_vec, device=device)

            # Check convergence
            max_force = compute_max_force(ext_forces_np)
            if max_force < force_tol:
                break

            # FIRE step
            fire_step(
                positions=ext_positions,
                velocities=ext_velocities,
                forces=ext_forces,
                masses=ext_masses,
                atom_ptr=atom_ptr,
                **params,
            )

        wp.synchronize_device(device)

        # Verify convergence
        final_ext_pos = ext_positions.numpy()
        final_cell_np = np.array(
            [
                [final_ext_pos[0, 0], 0.0, 0.0],
                [final_ext_pos[0, 1], final_ext_pos[1, 0], 0.0],
                [final_ext_pos[0, 2], final_ext_pos[1, 1], final_ext_pos[1, 2]],
            ],
            dtype=np_dtype,
        )

        # Cell should be close to target
        # Use atol for off-diagonal elements that should be 0 (rtol fails when expected=0)
        np.testing.assert_allclose(
            final_cell_np,
            target_cell_np,
            rtol=1e-2,
            atol=1e-5,  # Absolute tolerance for elements near zero
            err_msg=f"Cell did not converge to target after {step + 1} steps",
        )

    # =========================================================================
    # Physical Tests: Joint Atom + Cell Optimization (Full Integration)
    # =========================================================================

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_joint_atom_cell_optimization(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test joint optimization of atomic positions and cell parameters.

        This is a full integration test:
        1. align_cell() to put cell in standard form
        2. pack_positions_with_cell() / pack_forces_with_cell()
        3. Run FIRE optimization
        4. unpack_positions_with_cell() and verify

        System: Atoms with harmonic force to origin + cell with harmonic force to target.
        """
        num_atoms = 5
        num_extended = num_atoms + 2
        num_systems = 1
        k_atom = 1.0
        k_cell = 0.5

        np.random.seed(42)

        # Initial atomic positions (displaced from origin)
        initial_pos_np = np.random.randn(num_atoms, 3).astype(np_dtype) * 2.0

        # Target cell: cubic 10 Angstrom
        target_cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        target_cell = make_cell(target_cell_np, dtype_mat, device)

        # Initial cell: slightly distorted (already upper-triangular for simplicity)
        initial_cell_np = np.array(
            [
                [12.0, 0.0, 0.0],
                [0.5, 9.0, 0.0],
                [0.3, -0.2, 11.0],
            ],
            dtype=np_dtype,
        )

        # Create warp arrays
        positions = wp.array(initial_pos_np, dtype=dtype_vec, device=device)
        cell = make_cell(initial_cell_np, dtype_mat, device)

        # Align cell (should be no-op for already upper-triangular, but test workflow)
        transform = wp.empty(1, dtype=dtype_mat, device=device)
        positions, cell = align_cell(positions, cell, transform, device=device)

        # Pack into extended arrays
        ext_positions = wp.empty(num_extended, dtype=dtype_vec, device=device)
        pack_positions_with_cell(positions, cell, ext_positions, device=device)
        ext_velocities = wp.zeros(num_extended, dtype=dtype_vec, device=device)
        ext_masses_np = np.concatenate(
            [
                np.ones(num_atoms, dtype=np_dtype),  # atom masses
                np.ones(2, dtype=np_dtype) * 100.0,  # cell DOF masses (heavier)
            ]
        )
        ext_masses = wp.array(ext_masses_np, dtype=dtype_scalar, device=device)

        atom_ptr = wp.array(
            np.array([0, num_extended], dtype=np.int32), dtype=wp.int32, device=device
        )
        params = make_fire_params(num_systems, dtype_scalar, device, np_dtype)

        # Allocate force arrays
        forces = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        cell_force = wp.empty(1, dtype=dtype_mat, device=device)

        # Optimization loop
        max_steps = 1000
        force_tol = 1e-3

        for step in range(max_steps):
            # Unpack current state
            pos_current = wp.empty(num_atoms, dtype=dtype_vec, device=device)
            cell_current = wp.empty(1, dtype=dtype_mat, device=device)
            unpack_positions_with_cell(
                ext_positions,
                pos_current,
                cell_current,
                num_atoms=num_atoms,
                device=device,
            )

            # Compute atomic forces (harmonic to origin)
            compute_harmonic_forces(pos_current, forces, k_atom)

            # Compute cell stress (harmonic to target)
            stress = compute_harmonic_cell_stress(
                cell_current, target_cell, k_cell, device=device
            )

            # Convert stress to cell force
            volume = wp.empty(1, dtype=dtype_scalar, device=device)
            compute_cell_volume(cell_current, volume, device=device)
            cell_force = wp.empty(1, dtype=dtype_mat, device=device)
            stress_to_cell_force(
                stress,
                cell_current,
                volume,
                cell_force,
                keep_aligned=True,
                device=device,
            )

            # Pack forces into extended array
            ext_forces = wp.empty(num_extended, dtype=dtype_vec, device=device)
            pack_forces_with_cell(forces, cell_force, ext_forces, device=device)

            # Check convergence
            wp.synchronize_device(device)
            ext_forces_np = ext_forces.numpy()
            max_force = compute_max_force(ext_forces_np)

            if max_force < force_tol:
                break

            # FIRE step
            fire_step(
                positions=ext_positions,
                velocities=ext_velocities,
                forces=ext_forces,
                masses=ext_masses,
                atom_ptr=atom_ptr,
                **params,
            )

        wp.synchronize_device(device)

        # Unpack final state
        final_pos = wp.empty(num_atoms, dtype=dtype_vec, device=device)
        final_cell = wp.empty(1, dtype=dtype_mat, device=device)
        unpack_positions_with_cell(
            ext_positions, final_pos, final_cell, num_atoms=num_atoms, device=device
        )
        wp.synchronize_device(device)

        final_pos_np = final_pos.numpy()
        final_cell_np = cell_to_numpy(final_cell)

        # Atoms should converge toward origin
        max_pos = np.max(np.abs(final_pos_np))
        assert max_pos < 0.5, (
            f"Atoms did not converge to origin (max displacement: {max_pos})"
        )

        # Cell should converge toward target
        # Use atol for off-diagonal elements that should be 0 (rtol fails when expected=0)
        np.testing.assert_allclose(
            final_cell_np,
            target_cell_np,
            rtol=0.1,  # Allow 10% error for diagonal
            atol=1e-4,  # Absolute tolerance for elements near zero
            err_msg=f"Cell did not converge to target after {step + 1} steps",
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_cell_alignment_roundtrip(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test that align_cell followed by pack/unpack preserves the system.

        Workflow:
        1. Start with arbitrary (non-upper-triangular) cell
        2. Apply align_cell() to transform to standard form
        3. Pack positions/cell
        4. Unpack positions/cell
        5. Verify cell is upper-triangular and positions are consistent
        """
        num_atoms = 10
        np.random.seed(123)

        # Create non-upper-triangular cell (general triclinic)
        # Use a rotation to create a non-standard cell
        theta = np.pi / 6  # 30 degree rotation around z
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=np_dtype,
        )

        cell_diag = np.diag([10.0, 8.0, 12.0]).astype(np_dtype)
        original_cell_np = R @ cell_diag  # Rotated cell

        # Random positions in original cell
        original_pos_np = np.random.rand(num_atoms, 3).astype(np_dtype) * 5.0

        # Compute original fractional coordinates
        original_frac = original_pos_np @ np.linalg.inv(original_cell_np)

        # Create warp arrays
        positions = wp.array(original_pos_np.copy(), dtype=dtype_vec, device=device)
        cell = make_cell(original_cell_np.copy(), dtype_mat, device)

        # Apply cell alignment
        transform = wp.empty(1, dtype=dtype_mat, device=device)
        positions, cell = align_cell(positions, cell, transform, device=device)

        wp.synchronize_device(device)
        aligned_pos_np = positions.numpy()
        aligned_cell_np = cell_to_numpy(cell)

        # Verify cell is upper-triangular
        assert np.abs(aligned_cell_np[0, 1]) < 1e-5, "Cell[0,1] should be zero"
        assert np.abs(aligned_cell_np[0, 2]) < 1e-5, "Cell[0,2] should be zero"
        assert np.abs(aligned_cell_np[1, 2]) < 1e-5, "Cell[1,2] should be zero"

        # Verify fractional coordinates are preserved
        aligned_frac = aligned_pos_np @ np.linalg.inv(aligned_cell_np)
        np.testing.assert_allclose(
            aligned_frac,
            original_frac,
            rtol=1e-4,
            err_msg="Fractional coordinates not preserved after align_cell",
        )

        # Verify volume is preserved
        original_vol = np.abs(np.linalg.det(original_cell_np))
        aligned_vol = np.abs(np.linalg.det(aligned_cell_np))
        np.testing.assert_allclose(
            aligned_vol,
            original_vol,
            rtol=1e-5,
            err_msg="Cell volume not preserved after align_cell",
        )

        # Now test pack/unpack roundtrip
        ext_positions = wp.empty(num_atoms + 2, dtype=dtype_vec, device=device)
        pack_positions_with_cell(positions, cell, ext_positions, device=device)
        unpacked_pos = wp.empty(num_atoms, dtype=dtype_vec, device=device)
        unpacked_cell = wp.empty(1, dtype=dtype_mat, device=device)
        unpack_positions_with_cell(
            ext_positions,
            unpacked_pos,
            unpacked_cell,
            num_atoms=num_atoms,
            device=device,
        )

        wp.synchronize_device(device)
        unpacked_pos_np = unpacked_pos.numpy()
        unpacked_cell_np = cell_to_numpy(unpacked_cell)

        # Verify pack/unpack preserves data
        np.testing.assert_allclose(
            unpacked_pos_np,
            aligned_pos_np,
            rtol=1e-5,
            err_msg="Positions not preserved after pack/unpack",
        )
        np.testing.assert_allclose(
            unpacked_cell_np,
            aligned_cell_np,
            rtol=1e-5,
            err_msg="Cell not preserved after pack/unpack",
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_anisotropic_cell_response(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test that cell responds correctly to anisotropic stress.

        Apply different target cells in each dimension and verify
        the optimization responds appropriately.
        """
        num_extended = 2  # Cell only
        num_systems = 1
        k_cell = 2.0

        # Anisotropic target: different dimensions
        target_cell_np = np.diag([8.0, 12.0, 10.0]).astype(np_dtype)
        target_cell = make_cell(target_cell_np, dtype_mat, device)

        # Start from cubic cell
        initial_cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)

        # Pack cell into extended positions
        v1 = np.array(
            [initial_cell_np[0, 0], initial_cell_np[1, 0], initial_cell_np[2, 0]],
            dtype=np_dtype,
        )
        v2 = np.array(
            [initial_cell_np[1, 1], initial_cell_np[2, 1], initial_cell_np[2, 2]],
            dtype=np_dtype,
        )
        ext_pos_np = np.vstack([v1, v2])

        ext_positions = wp.array(ext_pos_np, dtype=dtype_vec, device=device)
        ext_velocities = wp.zeros(num_extended, dtype=dtype_vec, device=device)
        ext_masses = wp.array(
            np.ones(num_extended, dtype=np_dtype) * 50.0,
            dtype=dtype_scalar,
            device=device,
        )

        atom_ptr = wp.array(
            np.array([0, num_extended], dtype=np.int32), dtype=wp.int32, device=device
        )
        params = make_fire_params(num_systems, dtype_scalar, device, np_dtype)

        # Run optimization
        max_steps = 300

        for step in range(max_steps):
            wp.synchronize_device(device)

            ext_pos_current = ext_positions.numpy()
            current_cell_np = np.array(
                [
                    [ext_pos_current[0, 0], 0.0, 0.0],
                    [ext_pos_current[0, 1], ext_pos_current[1, 0], 0.0],
                    [
                        ext_pos_current[0, 2],
                        ext_pos_current[1, 1],
                        ext_pos_current[1, 2],
                    ],
                ],
                dtype=np_dtype,
            )
            cell_current = make_cell(current_cell_np, dtype_mat, device)

            stress = compute_harmonic_cell_stress(
                cell_current, target_cell, k_cell, device=device
            )
            volume = wp.empty(1, dtype=dtype_scalar, device=device)
            compute_cell_volume(cell_current, volume, device=device)
            cell_force = wp.empty(1, dtype=dtype_mat, device=device)
            stress_to_cell_force(
                stress,
                cell_current,
                volume,
                cell_force,
                keep_aligned=True,
                device=device,
            )

            wp.synchronize_device(device)
            cf = cell_to_numpy(cell_force)
            f1 = np.array([cf[0, 0], cf[1, 0], cf[2, 0]], dtype=np_dtype)
            f2 = np.array([cf[1, 1], cf[2, 1], cf[2, 2]], dtype=np_dtype)
            ext_forces = wp.array(np.vstack([f1, f2]), dtype=dtype_vec, device=device)

            fire_step(
                positions=ext_positions,
                velocities=ext_velocities,
                forces=ext_forces,
                masses=ext_masses,
                atom_ptr=atom_ptr,
                **params,
            )

        wp.synchronize_device(device)

        final_ext_pos = ext_positions.numpy()
        final_cell_np = np.array(
            [
                [final_ext_pos[0, 0], 0.0, 0.0],
                [final_ext_pos[0, 1], final_ext_pos[1, 0], 0.0],
                [final_ext_pos[0, 2], final_ext_pos[1, 1], final_ext_pos[1, 2]],
            ],
            dtype=np_dtype,
        )

        # Verify each diagonal element converged to its target
        np.testing.assert_allclose(
            np.diag(final_cell_np),
            np.diag(target_cell_np),
            rtol=0.05,  # 5% tolerance
            err_msg="Cell diagonal did not converge to anisotropic target",
        )

        # Verify off-diagonal elements stayed near zero
        assert np.abs(final_cell_np[1, 0]) < 0.5, (
            "Off-diagonal [1,0] should remain small"
        )
        assert np.abs(final_cell_np[2, 0]) < 0.5, (
            "Off-diagonal [2,0] should remain small"
        )
        assert np.abs(final_cell_np[2, 1]) < 0.5, (
            "Off-diagonal [2,1] should remain small"
        )


# ==============================================================================
# Edge Cases and Error Handling
# ==============================================================================


class TestFireStepErrors:
    """Test error handling in fire_step."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_both_batch_idx_and_atom_ptr_raises(self, device):
        """Test that providing both batch_idx and atom_ptr raises error."""
        num_atoms = 10
        dtype_vec = wp.vec3f
        dtype_scalar = wp.float32
        np_dtype = np.float32

        positions = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        velocities = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        forces = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.ones(num_atoms, dtype=dtype_scalar, device=device)
        batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=device)
        atom_ptr = wp.array(
            np.array([0, num_atoms], dtype=np.int32), dtype=wp.int32, device=device
        )

        params = make_fire_params(1, dtype_scalar, device, np_dtype)

        with pytest.raises(ValueError, match="batch_idx OR atom_ptr, not both"):
            fire_step(
                positions=positions,
                velocities=velocities,
                forces=forces,
                masses=masses,
                batch_idx=batch_idx,
                atom_ptr=atom_ptr,
                **params,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_missing_accumulators_single_raises(self, device):
        """Test that missing accumulators in single system mode raises error."""
        num_atoms = 10
        dtype_vec = wp.vec3f
        dtype_scalar = wp.float32
        np_dtype = np.float32

        positions = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        velocities = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        forces = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.ones(num_atoms, dtype=dtype_scalar, device=device)

        params = make_fire_params(1, dtype_scalar, device, np_dtype)

        # Single system mode (no batch_idx, no atom_ptr) requires accumulators
        with pytest.raises(ValueError, match="accumulators required"):
            fire_step(
                positions=positions,
                velocities=velocities,
                forces=forces,
                masses=masses,
                # No vf, vv, ff provided
                **params,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_partial_downhill_arrays_raises(self, device):
        """Test that providing partial downhill arrays raises error."""
        num_atoms = 10
        dtype_vec = wp.vec3f
        dtype_scalar = wp.float32
        np_dtype = np.float32

        positions = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        velocities = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        forces = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.ones(num_atoms, dtype=dtype_scalar, device=device)
        atom_ptr = wp.array(
            np.array([0, num_atoms], dtype=np.int32), dtype=wp.int32, device=device
        )

        params = make_fire_params(1, dtype_scalar, device, np_dtype)

        # Only provide energy, not the others
        energy = wp.zeros(1, dtype=dtype_scalar, device=device)

        with pytest.raises(ValueError, match="must provide ALL"):
            fire_step(
                positions=positions,
                velocities=velocities,
                forces=forces,
                masses=masses,
                atom_ptr=atom_ptr,
                energy=energy,  # Partial downhill
                **params,
            )


# ==============================================================================
# Single System Mode Tests (with accumulators)
# ==============================================================================


class TestFireStepSingle:
    """Test fire_step in single system mode (requires accumulators)."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_step_single_runs(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test fire_step in single system mode."""
        num_atoms = 10

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype), dtype=dtype_scalar, device=device
        )

        params = make_fire_params(1, dtype_scalar, device, np_dtype)
        accum = make_accumulators(1, dtype_scalar, device, np_dtype)

        fire_step(
            positions=positions,
            velocities=velocities,
            forces=forces,
            masses=masses,
            **params,
            **accum,
        )

        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_step_single_downhill(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test fire_step in single system mode with downhill check."""
        num_atoms = 10
        num_systems = 1

        np.random.seed(42)
        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype), dtype=dtype_scalar, device=device
        )

        params = make_fire_params(num_systems, dtype_scalar, device, np_dtype)
        accum = make_accumulators(num_systems, dtype_scalar, device, np_dtype)
        downhill = make_downhill_arrays(
            num_atoms, num_systems, dtype_vec, dtype_scalar, device, np_dtype
        )

        # Initialize positions_last
        wp.copy(downhill["positions_last"], positions)

        fire_step(
            positions=positions,
            velocities=velocities,
            forces=forces,
            masses=masses,
            **params,
            **accum,
            **downhill,
        )

        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_step_single_device_inference(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test fire_step in single system mode with device inference."""
        num_atoms = 10

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype), dtype=dtype_scalar, device=device
        )

        params = make_fire_params(1, dtype_scalar, device, np_dtype)
        accum = make_accumulators(1, dtype_scalar, device, np_dtype)

        # Don't pass device - should be inferred from positions
        fire_step(
            positions=positions,
            velocities=velocities,
            forces=forces,
            masses=masses,
            **params,
            **accum,
        )

        wp.synchronize_device(device)


class TestFireStepBatchIdx:
    """Test fire_step with batch_idx mode."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_step_batch_idx_runs(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test fire_step with batch_idx."""
        num_systems = 2
        atoms_per_system = 10
        total_atoms = num_systems * atoms_per_system

        positions = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        forces = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype), dtype=dtype_scalar, device=device
        )
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        params = make_fire_params(num_systems, dtype_scalar, device, np_dtype)
        accum = make_accumulators(num_systems, dtype_scalar, device, np_dtype)

        fire_step(
            positions=positions,
            velocities=velocities,
            forces=forces,
            masses=masses,
            batch_idx=batch_idx,
            **params,
            **accum,
        )

        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_step_batch_idx_downhill(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """Test fire_step with batch_idx and downhill check."""
        num_systems = 2
        atoms_per_system = 10
        total_atoms = num_systems * atoms_per_system

        np.random.seed(42)
        positions = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        forces = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype), dtype=dtype_scalar, device=device
        )
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        params = make_fire_params(num_systems, dtype_scalar, device, np_dtype)
        accum = make_accumulators(num_systems, dtype_scalar, device, np_dtype)
        downhill = make_downhill_arrays(
            total_atoms, num_systems, dtype_vec, dtype_scalar, device, np_dtype
        )

        # Initialize positions_last
        wp.copy(downhill["positions_last"], positions)

        fire_step(
            positions=positions,
            velocities=velocities,
            forces=forces,
            masses=masses,
            batch_idx=batch_idx,
            **params,
            **accum,
            **downhill,
        )

        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    def test_fire_step_batch_idx_missing_accumulators(self, device):
        """Test fire_step batch_idx mode raises error without accumulators."""
        num_systems = 2
        atoms_per_system = 5
        total_atoms = num_systems * atoms_per_system
        np_dtype = np.float64

        positions = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=wp.vec3d,
            device=device,
        )
        velocities = wp.zeros(total_atoms, dtype=wp.vec3d, device=device)
        forces = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=wp.vec3d,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype), dtype=wp.float64, device=device
        )
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        params = make_fire_params(num_systems, wp.float64, device, np_dtype)

        # Don't pass accumulators - should raise error
        with pytest.raises(ValueError, match="vf, vv, ff accumulators required"):
            fire_step(
                positions=positions,
                velocities=velocities,
                forces=forces,
                masses=masses,
                batch_idx=batch_idx,
                **params,
            )


# ==============================================================================
# Determinism Tests
# ==============================================================================


class TestFireDeterminism:
    """Verify that fire_step and fire_update produce bitwise-identical results
    across repeated runs with identical inputs.

    The previous fused kernels had read-before-reduction-complete races that
    caused non-deterministic outputs. The RLE-based multi-kernel approach should
    produce identical results every time.
    """

    NUM_REPEATS = 20

    def _clone_wp_array(self, arr):
        """Create a deep copy of a warp array."""
        clone = wp.empty_like(arr)
        wp.copy(clone, arr)
        return clone

    def _snapshot(self, arrays_dict):
        """Deep-copy a dict of warp arrays."""
        return {k: self._clone_wp_array(v) for k, v in arrays_dict.items()}

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_step_single_deterministic(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """fire_step single-system must be deterministic across repeats."""
        np.random.seed(42)
        num_atoms = 64

        positions_ref = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities_ref = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.01,
            dtype=dtype_vec,
            device=device,
        )
        forces_ref = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype), dtype=dtype_scalar, device=device
        )

        params_ref = make_fire_params(1, dtype_scalar, device, np_dtype)
        accum_ref = make_accumulators(1, dtype_scalar, device, np_dtype)

        ref_pos = ref_vel = ref_dt = ref_alpha = None

        for i in range(self.NUM_REPEATS):
            pos = self._clone_wp_array(positions_ref)
            vel = self._clone_wp_array(velocities_ref)
            forces = self._clone_wp_array(forces_ref)
            params = self._snapshot(params_ref)
            accum = self._snapshot(accum_ref)

            fire_step(
                positions=pos,
                velocities=vel,
                forces=forces,
                masses=masses,
                **params,
                **accum,
            )
            wp.synchronize()

            if i == 0:
                ref_pos = pos
                ref_vel = vel
                ref_dt = params["dt"]
                ref_alpha = params["alpha"]
            else:
                assert np.array_equal(ref_pos.numpy(), pos.numpy()), (
                    f"Positions differ on repeat {i}"
                )
                assert np.array_equal(ref_vel.numpy(), vel.numpy()), (
                    f"Velocities differ on repeat {i}"
                )
                assert np.array_equal(ref_dt.numpy(), params["dt"].numpy()), (
                    f"dt differs on repeat {i}"
                )
                assert np.array_equal(ref_alpha.numpy(), params["alpha"].numpy()), (
                    f"alpha differs on repeat {i}"
                )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_step_batch_idx_deterministic(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """fire_step batch_idx mode must be deterministic across repeats."""
        np.random.seed(123)
        num_systems = 3
        atoms_per_system = 32
        total_atoms = num_systems * atoms_per_system

        positions_ref = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities_ref = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype) * 0.01,
            dtype=dtype_vec,
            device=device,
        )
        forces_ref = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype), dtype=dtype_scalar, device=device
        )
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        params_ref = make_fire_params(num_systems, dtype_scalar, device, np_dtype)
        accum_ref = make_accumulators(num_systems, dtype_scalar, device, np_dtype)

        ref_pos = ref_vel = None

        for i in range(self.NUM_REPEATS):
            pos = self._clone_wp_array(positions_ref)
            vel = self._clone_wp_array(velocities_ref)
            forces = self._clone_wp_array(forces_ref)
            params = self._snapshot(params_ref)
            accum = self._snapshot(accum_ref)

            fire_step(
                positions=pos,
                velocities=vel,
                forces=forces,
                masses=masses,
                batch_idx=batch_idx,
                **params,
                **accum,
            )
            wp.synchronize()

            if i == 0:
                ref_pos = pos
                ref_vel = vel
            else:
                assert np.array_equal(ref_pos.numpy(), pos.numpy()), (
                    f"Positions differ on repeat {i}"
                )
                assert np.array_equal(ref_vel.numpy(), vel.numpy()), (
                    f"Velocities differ on repeat {i}"
                )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_update_single_deterministic(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """fire_update single-system must be deterministic across repeats."""
        np.random.seed(200)
        num_atoms = 64

        velocities_ref = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.01,
            dtype=dtype_vec,
            device=device,
        )
        forces_ref = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )

        params_ref = make_fire_params(1, dtype_scalar, device, np_dtype)
        accum_ref = make_accumulators(1, dtype_scalar, device, np_dtype)

        ref_vel = ref_dt = ref_alpha = None

        for i in range(self.NUM_REPEATS):
            vel = self._clone_wp_array(velocities_ref)
            forces = self._clone_wp_array(forces_ref)
            params = self._snapshot(params_ref)
            accum = self._snapshot(accum_ref)

            fire_update(
                velocities=vel,
                forces=forces,
                alpha=params["alpha"],
                dt=params["dt"],
                alpha_start=params["alpha_start"],
                f_alpha=params["f_alpha"],
                dt_min=params["dt_min"],
                dt_max=params["dt_max"],
                n_steps_positive=params["n_steps_positive"],
                n_min=params["n_min"],
                f_dec=params["f_dec"],
                f_inc=params["f_inc"],
                **accum,
            )
            wp.synchronize()

            if i == 0:
                ref_vel = vel
                ref_dt = params["dt"]
                ref_alpha = params["alpha"]
            else:
                assert np.array_equal(ref_vel.numpy(), vel.numpy()), (
                    f"Velocities differ on repeat {i}"
                )
                assert np.array_equal(ref_dt.numpy(), params["dt"].numpy()), (
                    f"dt differs on repeat {i}"
                )
                assert np.array_equal(ref_alpha.numpy(), params["alpha"].numpy()), (
                    f"alpha differs on repeat {i}"
                )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,dtype_mat,np_dtype", DTYPE_CONFIGS)
    def test_fire_update_batch_idx_deterministic(
        self, device, dtype_vec, dtype_scalar, dtype_mat, np_dtype
    ):
        """fire_update batch_idx mode must be deterministic across repeats."""
        np.random.seed(300)
        num_systems = 3
        atoms_per_system = 32
        total_atoms = num_systems * atoms_per_system

        velocities_ref = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype) * 0.01,
            dtype=dtype_vec,
            device=device,
        )
        forces_ref = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        params_ref = make_fire_params(num_systems, dtype_scalar, device, np_dtype)
        accum_ref = make_accumulators(num_systems, dtype_scalar, device, np_dtype)

        ref_vel = None

        for i in range(self.NUM_REPEATS):
            vel = self._clone_wp_array(velocities_ref)
            forces = self._clone_wp_array(forces_ref)
            params = self._snapshot(params_ref)
            accum = self._snapshot(accum_ref)

            fire_update(
                velocities=vel,
                forces=forces,
                alpha=params["alpha"],
                dt=params["dt"],
                alpha_start=params["alpha_start"],
                f_alpha=params["f_alpha"],
                dt_min=params["dt_min"],
                dt_max=params["dt_max"],
                n_steps_positive=params["n_steps_positive"],
                n_min=params["n_min"],
                f_dec=params["f_dec"],
                f_inc=params["f_inc"],
                batch_idx=batch_idx,
                **accum,
            )
            wp.synchronize()

            if i == 0:
                ref_vel = vel
            else:
                assert np.array_equal(ref_vel.numpy(), vel.numpy()), (
                    f"Velocities differ on repeat {i}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

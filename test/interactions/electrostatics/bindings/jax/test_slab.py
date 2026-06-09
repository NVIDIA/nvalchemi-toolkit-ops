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

"""Tests for explicit-output JAX slab electrostatics bindings."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

from nvalchemiops.jax.interactions.electrostatics import compute_slab_correction
from nvalchemiops.jax.interactions.electrostatics.ewald import (
    ewald_real_space as _ewald_real_space,
)
from nvalchemiops.jax.interactions.electrostatics.ewald import (
    ewald_reciprocal_space as _ewald_reciprocal_space,
)
from nvalchemiops.jax.interactions.electrostatics.ewald import (
    ewald_summation as _ewald_summation,
)
from nvalchemiops.jax.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.jax.interactions.electrostatics.pme import (
    particle_mesh_ewald as _particle_mesh_ewald,
)
from nvalchemiops.jax.interactions.electrostatics.pme import (
    pme_reciprocal_space as _pme_reciprocal_space,
)
from nvalchemiops.jax.neighbors import batch_cell_list, cell_list

EWALD_ALPHA = 0.35
EWALD_K_CUTOFF = 8.0
PME_MESH = (8, 8, 8)
REAL_SPACE_CUTOFF = 6.0
SLAB_STRICT_RTOL = 1e-12
SLAB_STRICT_ATOL = 1e-14
OUTPUT_CASES = [
    (False, False, False, ("energies",)),
    (True, False, False, ("energies", "forces")),
    (False, True, False, ("energies", "charge_grads")),
    (False, False, True, ("energies", "virial")),
    (True, True, False, ("energies", "forces", "charge_grads")),
    (True, False, True, ("energies", "forces", "virial")),
    (False, True, True, ("energies", "charge_grads", "virial")),
    (True, True, True, ("energies", "forces", "charge_grads", "virial")),
]


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


def ewald_reciprocal_space(*args, **kwargs):
    """Test-local wrapper suppressing intentional component deprecations."""
    return _call_without_component_direct_output_deprecation(
        "ewald_reciprocal_space",
        _ewald_reciprocal_space,
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


def _make_slab_system(dtype=jnp.float64):
    """Return a small triclinic single-system slab fixture."""
    positions = jnp.array(
        [
            [2.0, 2.0, 4.0],
            [4.0, 5.0, 9.0],
            [7.0, 3.0, 15.0],
        ],
        dtype=dtype,
    )
    charges = jnp.array([1.0, -1.0, 0.5], dtype=dtype)
    cell = jnp.array(
        [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [1.0, 0.5, 30.0]]],
        dtype=dtype,
    )
    pbc_slab = jnp.array([[True, True, False]], dtype=jnp.bool_)
    return positions, charges, cell, pbc_slab


def _make_batched_system(dtype=jnp.float64):
    """Return one slab system batched with one 3D-periodic system."""
    slab_positions, slab_charges, slab_cell, _ = _make_slab_system(dtype)
    other_positions = jnp.array(
        [[1.0, 1.0, 2.0], [3.0, 4.0, 8.0]],
        dtype=dtype,
    )
    other_charges = jnp.array([1.0, -1.0], dtype=dtype)
    other_cell = jnp.eye(3, dtype=dtype) * 12.0
    positions = jnp.concatenate([slab_positions, other_positions], axis=0)
    charges = jnp.concatenate([slab_charges, other_charges], axis=0)
    cell = jnp.concatenate([slab_cell, other_cell[jnp.newaxis, :, :]], axis=0)
    batch_idx = jnp.array([0, 0, 0, 1, 1], dtype=jnp.int32)
    pbc = jnp.array(
        [[True, True, False], [True, True, True]],
        dtype=jnp.bool_,
    )
    return positions, charges, cell, batch_idx, pbc


def _reference_slab_correction(positions, charges, cell, pbc, batch_idx=None):
    """Compute slab corrections using an independent reference formula."""
    input_dtype = positions.dtype
    positions_f64 = positions.astype(jnp.float64)
    charges_f64 = charges.astype(jnp.float64)
    cell_f64 = cell.astype(jnp.float64)
    if cell_f64.ndim == 2:
        cell_f64 = cell_f64[jnp.newaxis, :, :]
    pbc_array = jnp.asarray(pbc)
    if pbc_array.ndim == 1:
        pbc_array = pbc_array[jnp.newaxis, :]
    if batch_idx is None:
        batch_idx_i32 = jnp.zeros(positions_f64.shape[0], dtype=jnp.int32)
    else:
        batch_idx_i32 = batch_idx.astype(jnp.int32)

    num_systems = cell_f64.shape[0]
    axis_order_a = jnp.array([1, 2, 0])
    axis_order_b = jnp.array([2, 0, 1])
    periodic_a = cell_f64[:, axis_order_a, :]
    periodic_b = cell_f64[:, axis_order_b, :]
    normals = jnp.cross(periodic_a, periodic_b)
    normals = normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)
    nonperiodic_vectors = cell_f64
    volume = jnp.abs(jnp.linalg.det(cell_f64))
    height_sq = jnp.sum(nonperiodic_vectors * normals, axis=-1) ** 2
    slab_axis_mask = jnp.logical_and(
        ~pbc_array,
        jnp.sum(~pbc_array, axis=1, keepdims=True) == 1,
    ).astype(jnp.float64)

    normal_atoms = normals[batch_idx_i32]
    z_values = jnp.einsum("nd,nad->na", positions_f64, normal_atoms)
    charge_column = charges_f64[:, jnp.newaxis]
    moments = jnp.zeros((num_systems, 3), dtype=jnp.float64)
    projected_moment = moments.at[batch_idx_i32].add(charge_column * z_values)
    projected_second_moment = moments.at[batch_idx_i32].add(
        charge_column * z_values * z_values
    )
    total_charge = (
        jnp.zeros((num_systems,), dtype=jnp.float64).at[batch_idx_i32].add(charges_f64)
    )

    projected_moment_atoms = projected_moment[batch_idx_i32]
    projected_second_moment_atoms = projected_second_moment[batch_idx_i32]
    total_charge_atoms = total_charge[batch_idx_i32, jnp.newaxis]
    volume_atoms = volume[batch_idx_i32, jnp.newaxis]
    height_sq_atoms = height_sq[batch_idx_i32]
    slab_axis_mask_atoms = slab_axis_mask[batch_idx_i32]
    bracket = (
        z_values * projected_moment_atoms
        - 0.5
        * (projected_second_moment_atoms + total_charge_atoms * z_values * z_values)
        - total_charge_atoms * height_sq_atoms / 12.0
    )
    axis_energies = (
        (2.0 * jnp.pi / volume_atoms) * charge_column * bracket * slab_axis_mask_atoms
    )
    energies = jnp.sum(axis_energies, axis=1)
    force_magnitudes = (
        -(4.0 * jnp.pi / volume_atoms)
        * charge_column
        * (projected_moment_atoms - total_charge_atoms * z_values)
        * slab_axis_mask_atoms
    )
    forces = jnp.sum(force_magnitudes[:, :, jnp.newaxis] * normal_atoms, axis=1)
    charge_grads = jnp.sum(
        (4.0 * jnp.pi / volume_atoms) * bracket * slab_axis_mask_atoms,
        axis=1,
    )
    system_axis_energies = jnp.zeros((num_systems, 3), dtype=jnp.float64)
    system_axis_energies = system_axis_energies.at[batch_idx_i32].add(axis_energies)
    identity = jnp.eye(3, dtype=jnp.float64)
    virial_axes = system_axis_energies[:, :, jnp.newaxis, jnp.newaxis] * (
        identity[jnp.newaxis, jnp.newaxis, :, :]
        - 2.0 * normals[:, :, :, jnp.newaxis] * normals[:, :, jnp.newaxis, :]
    )
    virial = jnp.sum(virial_axes, axis=1)

    return (
        energies,
        forces.astype(input_dtype),
        charge_grads,
        virial.astype(input_dtype),
    )


def _assert_close(actual, expected, rtol=1e-10, atol=1e-12):
    """Assert two JAX-compatible arrays are numerically close."""
    actual_array = jnp.asarray(actual)
    expected_array = jnp.asarray(expected)
    assert actual_array.shape == expected_array.shape
    assert bool(jnp.allclose(actual_array, expected_array, rtol=rtol, atol=atol))


def _as_named_outputs(output, compute_forces, compute_charge_gradients, compute_virial):
    """Convert an electrostatics output tuple into a name-to-array mapping."""
    output_tuple = output if isinstance(output, tuple) else (output,)
    cursor = 0
    named = {"energies": output_tuple[cursor]}
    cursor += 1
    if compute_forces:
        named["forces"] = output_tuple[cursor]
        cursor += 1
    if compute_charge_gradients:
        named["charge_grads"] = output_tuple[cursor]
        cursor += 1
    if compute_virial:
        named["virial"] = output_tuple[cursor]
    return named


def _component_sum(
    real_outputs,
    reciprocal_outputs,
    slab_outputs,
    compute_forces,
    compute_charge_gradients,
    compute_virial,
):
    """Return named real + reciprocal + slab component sums."""
    real_named = _as_named_outputs(
        real_outputs, compute_forces, compute_charge_gradients, compute_virial
    )
    reciprocal_named = _as_named_outputs(
        reciprocal_outputs, compute_forces, compute_charge_gradients, compute_virial
    )
    slab_named = _as_named_outputs(
        slab_outputs, compute_forces, compute_charge_gradients, compute_virial
    )
    return {
        name: real_named[name] + reciprocal_named[name] + slab_named[name]
        for name in real_named
    }


def _assert_full_slab_energy_autodiff_matches_outputs(
    slab_energy_from_full_api,
    positions,
    charges,
    cell,
    slab_outputs,
    *,
    rtol=1e-9,
    atol=1e-10,
):
    """Validate full energy-only slab contribution derivatives."""
    _energies, forces, charge_grads, virial = slab_outputs

    grad_positions = jax.grad(
        lambda pos: slab_energy_from_full_api(pos, charges, cell)
    )(positions)
    grad_charges = jax.grad(
        lambda chg: slab_energy_from_full_api(positions, chg, cell)
    )(charges)

    eps = jnp.zeros((3, 3), dtype=positions.dtype)

    def strained_energy(eps_in):
        deformation = jnp.eye(3, dtype=positions.dtype) + eps_in
        return slab_energy_from_full_api(
            positions @ deformation,
            charges,
            cell @ deformation,
        )

    grad_strain = jax.grad(strained_energy)(eps)

    _assert_close(-grad_positions, forces, rtol=rtol, atol=atol)
    _assert_close(grad_charges, charge_grads, rtol=rtol, atol=atol)
    _assert_close(-grad_strain, virial[0], rtol=rtol, atol=atol)


def _expected_by_name(outputs):
    """Return a name-to-array mapping for full slab reference outputs."""
    return {
        "energies": outputs[0],
        "forces": outputs[1],
        "charge_grads": outputs[2],
        "virial": outputs[3],
    }


class TestStandaloneSlabCorrection:
    """Standalone JAX slab correction API behavior."""

    @pytest.mark.parametrize(
        (
            "compute_forces",
            "compute_charge_gradients",
            "compute_virial",
            "output_names",
        ),
        OUTPUT_CASES,
    )
    def test_output_subsets_match_reference(
        self,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
        output_names,
        device,  # noqa: ARG002
    ):
        """Standalone output tuple ordering and values follow enabled flags."""
        positions, charges, cell, pbc = _make_slab_system()

        result = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
        )
        expected = _expected_by_name(
            _reference_slab_correction(positions, charges, cell, pbc)
        )

        if len(output_names) == 1:
            assert not isinstance(result, tuple)
        else:
            assert isinstance(result, tuple)
        result_tuple = result if isinstance(result, tuple) else (result,)

        assert len(result_tuple) == len(output_names)
        for output, name in zip(result_tuple, output_names, strict=True):
            _assert_close(output, expected[name])

    def test_outputs_match_reference(self, device):  # noqa: ARG002
        """Standalone slab outputs match the analytical reference formula."""
        positions, charges, cell, pbc = _make_slab_system()

        outputs = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        expected = _reference_slab_correction(positions, charges, cell, pbc)

        for actual, reference in zip(outputs, expected):
            _assert_close(actual, reference)

    def test_explicit_outputs_match_jax_autograd_reference(self, device):  # noqa: ARG002
        """Standalone explicit outputs match pure-JAX autodiff derivatives."""
        positions, charges, cell, pbc = _make_slab_system()

        def reference_energy(positions_in, charges_in, cell_in):
            return _reference_slab_correction(positions_in, charges_in, cell_in, pbc)[
                0
            ].sum()

        grad_positions, grad_charges = jax.grad(
            reference_energy,
            argnums=(0, 1),
        )(positions, charges, cell)
        eps = jnp.zeros((3, 3), dtype=positions.dtype)

        def strained_reference_energy(eps_in):
            deformation = jnp.eye(3, dtype=positions.dtype) + eps_in
            return reference_energy(
                positions @ deformation,
                charges,
                cell @ deformation,
            )

        autograd_virial = -jax.grad(strained_reference_energy)(eps)
        _, forces, charge_grads, virial = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )

        _assert_close(forces, -grad_positions)
        _assert_close(charge_grads, grad_charges)
        _assert_close(virial[0], autograd_virial, rtol=1e-9, atol=1e-10)

    def test_energy_autodiff_matches_explicit_derivative_outputs(self, device):  # noqa: ARG002
        """Energy-only slab path exposes explicit Warp first derivatives."""
        positions, charges, cell, pbc = _make_slab_system()

        def slab_energy(pos, chg, cell_in):
            return compute_slab_correction(pos, chg, cell_in, pbc).sum()

        grad_positions, grad_charges, grad_cell = jax.grad(
            slab_energy,
            argnums=(0, 1, 2),
        )(positions, charges, cell)
        _, forces, charge_grads, _virial = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        reference_grad_cell = jax.grad(
            lambda cell_in: _reference_slab_correction(
                positions,
                charges,
                cell_in,
                pbc,
            )[0].sum()
        )(cell)
        jit_grad_positions = jax.jit(
            jax.grad(lambda pos: slab_energy(pos, charges, cell))
        )(positions)

        _assert_close(-grad_positions, forces, rtol=1e-9, atol=1e-10)
        _assert_close(grad_charges, charge_grads, rtol=1e-9, atol=1e-10)
        _assert_close(grad_cell, reference_grad_cell, rtol=1e-9, atol=1e-10)
        _assert_close(jit_grad_positions, grad_positions, rtol=1e-9, atol=1e-10)

    def test_energy_second_derivative_losses_are_finite(self, device):  # noqa: ARG002
        """Standalone slab force, charge, and cell losses support nested gradients."""
        positions, charges, cell, pbc = _make_slab_system()

        def slab_energy(pos, chg, cell_in):
            return compute_slab_correction(pos, chg, cell_in, pbc).sum()

        def force_loss(pos):
            grad_positions = jax.grad(
                lambda pos_in: slab_energy(pos_in, charges, cell)
            )(pos)
            return jnp.sum(grad_positions * grad_positions)

        def charge_loss(chg):
            grad_charges = jax.grad(
                lambda charges_in: slab_energy(positions, charges_in, cell)
            )(chg)
            return jnp.sum(grad_charges * grad_charges)

        def cell_loss(cell_in):
            grad_cell = jax.grad(
                lambda cell_arg: slab_energy(positions, charges, cell_arg)
            )(cell_in)
            return jnp.sum(grad_cell * grad_cell)

        h_positions = jnp.array(
            [[0.3, -0.2, 0.1], [0.0, 0.4, -0.5], [0.2, 0.1, -0.3]],
            dtype=positions.dtype,
        )
        h_charges = jnp.array([0.2, -0.1, 0.3], dtype=charges.dtype)
        h_cell = jnp.array(
            [[[0.02, -0.01, 0.03], [0.01, 0.02, -0.02], [0.0, 0.01, 0.02]]],
            dtype=cell.dtype,
        )

        def first_derivatives(pos, chg, cell_in):
            return jax.grad(slab_energy, argnums=(0, 1, 2))(pos, chg, cell_in)

        _, hvp = jax.jvp(
            first_derivatives,
            (positions, charges, cell),
            (h_positions, h_charges, h_cell),
        )
        step = jnp.asarray(1e-4, dtype=positions.dtype)
        plus = first_derivatives(
            positions + step * h_positions,
            charges + step * h_charges,
            cell + step * h_cell,
        )
        minus = first_derivatives(
            positions - step * h_positions,
            charges - step * h_charges,
            cell - step * h_cell,
        )
        fd_hvp = tuple((p - m) / (2.0 * step) for p, m in zip(plus, minus))
        for actual, expected in zip(hvp, fd_hvp, strict=True):
            _assert_close(actual, expected, rtol=1e-5, atol=1e-7)

        jit_hvp = jax.jit(
            lambda pos, chg, cell_in, hpos, hchg, hcell: jax.jvp(
                first_derivatives,
                (pos, chg, cell_in),
                (hpos, hchg, hcell),
            )[1]
        )(positions, charges, cell, h_positions, h_charges, h_cell)
        for actual, expected in zip(jit_hvp, hvp, strict=True):
            _assert_close(actual, expected, rtol=1e-12, atol=1e-12)

        grad_force_loss = jax.grad(force_loss)(positions)
        grad_charge_loss = jax.grad(charge_loss)(charges)
        grad_cell_loss = jax.grad(cell_loss)(cell)
        direction = h_positions
        direction = direction / jnp.linalg.norm(direction)
        force_loss_fd = (
            force_loss(positions + step * direction)
            - force_loss(positions - step * direction)
        ) / (2.0 * step)
        force_loss_jvp = jnp.sum(grad_force_loss * direction)

        assert bool(jnp.isfinite(grad_force_loss).all())
        assert bool(jnp.isfinite(grad_charge_loss).all())
        assert bool(jnp.isfinite(grad_cell_loss).all())
        _assert_close(force_loss_jvp, force_loss_fd, rtol=1e-4, atol=1e-7)

    def test_translation_invariance_non_neutral(self, device):  # noqa: ARG002
        """Non-neutral triclinic slab outputs are translation invariant."""
        positions, charges, cell, pbc = _make_slab_system()
        shift = jnp.array([1.3, -0.7, 2.1], dtype=positions.dtype)

        energies_0, forces_0, charge_grads_0 = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
        )
        energies_1, forces_1, charge_grads_1 = compute_slab_correction(
            positions + shift,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
        )

        _assert_close(energies_1.sum(), energies_0.sum())
        _assert_close(forces_1, forces_0)
        _assert_close(charge_grads_1, charge_grads_0)

    def test_output_dtypes(self, device):  # noqa: ARG002
        """Energy and charge gradients use float64; vector outputs use input dtype."""
        positions, charges, cell, pbc = _make_slab_system(dtype=jnp.float32)

        energies, forces, charge_grads, virial = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )

        assert energies.dtype == jnp.float64
        assert forces.dtype == positions.dtype
        assert charge_grads.dtype == jnp.float64
        assert virial.dtype == positions.dtype

    def test_3d_periodic_system_is_noop(self, device):  # noqa: ARG002
        """A fully 3D periodic pbc row contributes zero slab correction."""
        positions, charges, cell, _ = _make_slab_system()
        pbc_3d = jnp.array([[True, True, True]], dtype=jnp.bool_)

        outputs = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc_3d,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )

        _assert_close(outputs[0], jnp.zeros_like(outputs[0]), rtol=0, atol=0)
        _assert_close(outputs[1], jnp.zeros_like(outputs[1]), rtol=0, atol=0)
        _assert_close(outputs[2], jnp.zeros_like(outputs[2]), rtol=0, atol=0)
        _assert_close(outputs[3], jnp.zeros_like(outputs[3]), rtol=0, atol=0)

    def test_mixed_pbc_batch_zeroes_3d_system(self, device):  # noqa: ARG002
        """Batched standalone slab applies only to slab-like pbc rows."""
        positions, charges, cell, batch_idx, pbc = _make_batched_system()

        outputs = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            batch_idx=batch_idx,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        expected = _reference_slab_correction(
            positions, charges, cell, pbc, batch_idx=batch_idx
        )

        for actual, reference in zip(outputs, expected):
            _assert_close(actual, reference)
        _assert_close(outputs[0][3:], jnp.zeros_like(outputs[0][3:]), rtol=0, atol=0)
        _assert_close(outputs[1][3:], jnp.zeros_like(outputs[1][3:]), rtol=0, atol=0)
        _assert_close(outputs[2][3:], jnp.zeros_like(outputs[2][3:]), rtol=0, atol=0)
        _assert_close(outputs[3][1], jnp.zeros_like(outputs[3][1]), rtol=0, atol=0)

    def test_single_atom_non_neutral(self, device):  # noqa: ARG002
        """Single charged slabs keep finite background terms."""
        dtype = jnp.float64
        positions = jnp.array([[1.2, -0.4, 3.5]], dtype=dtype)
        charges = jnp.array([0.7], dtype=dtype)
        cell = jnp.diag(jnp.array([8.0, 9.0, 24.0], dtype=dtype))[jnp.newaxis, :, :]
        pbc = jnp.array([True, True, False], dtype=jnp.bool_)

        outputs = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        expected = _reference_slab_correction(positions, charges, cell, pbc)

        for actual, reference in zip(outputs, expected):
            _assert_close(actual, reference)

    def test_empty_standalone_system(self, device):  # noqa: ARG002
        """Empty standalone slab calls return empty outputs and zero virial."""
        dtype = jnp.float64
        positions = jnp.empty((0, 3), dtype=dtype)
        charges = jnp.empty((0,), dtype=dtype)
        cell = jnp.diag(jnp.array([8.0, 9.0, 24.0], dtype=dtype))[jnp.newaxis, :, :]
        pbc = jnp.array([True, True, False], dtype=jnp.bool_)

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
        _assert_close(virial, jnp.zeros_like(virial), rtol=0, atol=0)


class TestJaxSlabPbcShapeContracts:
    """Public slab APIs validate pbc shape contracts."""

    def test_single_system_accepts_1d_pbc(self, device):  # noqa: ARG002
        """A single-system (3,) pbc is equivalent to explicit (1, 3) pbc."""
        positions, charges, cell, pbc = _make_slab_system()

        energy_1d = compute_slab_correction(positions, charges, cell, pbc[0])
        energy_2d = compute_slab_correction(positions, charges, cell, pbc)

        _assert_close(energy_1d, energy_2d, rtol=0, atol=0)

    def test_single_system_pme_accepts_1d_pbc(self, device):  # noqa: ARG002
        """A single-system PME slab call accepts either (3,) or (1, 3) pbc."""
        positions, charges, cell, pbc = _make_slab_system()
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc,
        )

        energies_1d = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            mesh_dimensions=PME_MESH,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            pbc=pbc[0],
            slab_correction=True,
        )
        energies_2d = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            mesh_dimensions=PME_MESH,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            pbc=pbc,
            slab_correction=True,
        )

        _assert_close(
            energies_1d,
            energies_2d,
            rtol=SLAB_STRICT_RTOL,
            atol=SLAB_STRICT_ATOL,
        )

    def test_batched_standalone_rejects_1d_pbc(self, device):  # noqa: ARG002
        """Batched standalone calls require explicit per-system pbc rows."""
        positions, charges, cell, batch_idx, pbc = _make_batched_system()

        with pytest.raises(ValueError, match="batched.*pbc"):
            compute_slab_correction(
                positions, charges, cell, pbc[0], batch_idx=batch_idx
            )

    def test_standalone_rejects_non_bool_pbc(self, device):  # noqa: ARG002
        """Standalone slab calls require boolean pbc arrays."""
        positions, charges, cell, _ = _make_slab_system()
        pbc = jnp.array([1, 1, 0], dtype=jnp.int32)

        with pytest.raises(ValueError, match="bool"):
            compute_slab_correction(positions, charges, cell, pbc)

    def test_standalone_rejects_bad_pbc_shape(self, device):  # noqa: ARG002
        """Standalone slab calls reject pbc arrays that are not (3,) or (B, 3)."""
        positions, charges, cell, _ = _make_slab_system()
        pbc = jnp.array([True, False], dtype=jnp.bool_)

        with pytest.raises(ValueError, match="shape"):
            compute_slab_correction(positions, charges, cell, pbc)

    def test_full_ewald_slab_requires_pbc(self, device):  # noqa: ARG002
        """Ewald slab correction requires explicit slab periodicity."""
        positions, charges, cell, _ = _make_slab_system()

        with pytest.raises(ValueError, match="pbc"):
            ewald_summation(
                positions,
                charges,
                cell,
                alpha=EWALD_ALPHA,
                k_cutoff=EWALD_K_CUTOFF,
                slab_correction=True,
            )

    def test_full_pme_slab_requires_pbc(self, device):  # noqa: ARG002
        """PME slab correction requires explicit slab periodicity."""
        positions, charges, cell, _ = _make_slab_system()

        with pytest.raises(ValueError, match="pbc"):
            particle_mesh_ewald(
                positions,
                charges,
                cell,
                alpha=EWALD_ALPHA,
                mesh_dimensions=PME_MESH,
                slab_correction=True,
            )

    def test_single_system_ewald_accepts_1d_pbc(self, device):  # noqa: ARG002
        """A single-system Ewald slab call accepts either (3,) or (1, 3) pbc."""
        positions, charges, cell, pbc = _make_slab_system()
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc,
        )
        common_kwargs = {
            "alpha": EWALD_ALPHA,
            "k_cutoff": EWALD_K_CUTOFF,
            "neighbor_matrix": neighbor_matrix,
            "neighbor_matrix_shifts": neighbor_matrix_shifts,
            "compute_forces": True,
            "compute_charge_gradients": True,
            "compute_virial": True,
            "slab_correction": True,
        }

        outputs_1d = ewald_summation(
            positions,
            charges,
            cell,
            pbc=pbc[0],
            **common_kwargs,
        )
        outputs_2d = ewald_summation(
            positions,
            charges,
            cell,
            pbc=pbc,
            **common_kwargs,
        )

        for actual, expected in zip(outputs_1d, outputs_2d, strict=True):
            _assert_close(
                actual,
                expected,
                rtol=SLAB_STRICT_RTOL,
                atol=SLAB_STRICT_ATOL,
            )

    def test_batched_ewald_rejects_1d_pbc(self, device):  # noqa: ARG002
        """Batched Ewald slab calls require explicit per-system pbc rows."""
        positions, charges, cell, batch_idx, pbc = _make_batched_system()

        with pytest.raises(ValueError, match="batched.*pbc"):
            ewald_summation(
                positions,
                charges,
                cell,
                alpha=jnp.array([EWALD_ALPHA, EWALD_ALPHA], dtype=positions.dtype),
                k_cutoff=EWALD_K_CUTOFF,
                batch_idx=batch_idx,
                pbc=pbc[0],
                slab_correction=True,
            )

    def test_batched_pme_rejects_1d_pbc(self, device):  # noqa: ARG002
        """Batched PME slab calls require explicit per-system pbc rows."""
        positions, charges, cell, batch_idx, pbc = _make_batched_system()

        with pytest.raises(ValueError, match="batched.*pbc"):
            particle_mesh_ewald(
                positions,
                charges,
                cell,
                alpha=jnp.array([EWALD_ALPHA, EWALD_ALPHA], dtype=positions.dtype),
                mesh_dimensions=PME_MESH,
                batch_idx=batch_idx,
                pbc=pbc[0],
                slab_correction=True,
            )


class TestJaxEwaldSlabIntegration:
    """Full JAX Ewald slab wrapper composition."""

    def test_neighbor_pbc_ttf_matches_ttt_with_vacuum(self, device):  # noqa: ARG002
        """Ewald slab outputs are unchanged by T/T/F vs T/T/T vacuum neighbors."""
        positions, charges, cell, pbc = _make_slab_system()
        pbc_3d = jnp.array([[True, True, True]], dtype=jnp.bool_)
        neighbor_matrix_slab, _, neighbor_matrix_shifts_slab = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc,
        )
        neighbor_matrix_3d, _, neighbor_matrix_shifts_3d = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc_3d,
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
            neighbor_matrix=neighbor_matrix_slab,
            neighbor_matrix_shifts=neighbor_matrix_shifts_slab,
            **common_kwargs,
        )
        outputs_3d = ewald_summation(
            positions,
            charges,
            cell,
            neighbor_matrix=neighbor_matrix_3d,
            neighbor_matrix_shifts=neighbor_matrix_shifts_3d,
            **common_kwargs,
        )

        for actual, expected in zip(outputs_slab, outputs_3d, strict=True):
            _assert_close(
                actual,
                expected,
                rtol=SLAB_STRICT_RTOL,
                atol=SLAB_STRICT_ATOL,
            )

    def test_full_ewald_3d_pbc_noop(self, device):  # noqa: ARG002
        """3D periodic Ewald slab mode matches standard Ewald."""
        positions, charges, cell, pbc = _make_slab_system()
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc,
        )
        common_kwargs = {
            "alpha": EWALD_ALPHA,
            "k_cutoff": EWALD_K_CUTOFF,
            "neighbor_matrix": neighbor_matrix,
            "neighbor_matrix_shifts": neighbor_matrix_shifts,
            "compute_forces": True,
            "compute_charge_gradients": True,
            "compute_virial": True,
        }

        outputs_off = ewald_summation(positions, charges, cell, **common_kwargs)
        outputs_3d = ewald_summation(
            positions,
            charges,
            cell,
            pbc=jnp.array([True, True, True], dtype=jnp.bool_),
            slab_correction=True,
            **common_kwargs,
        )

        for actual, expected in zip(outputs_3d, outputs_off, strict=True):
            _assert_close(
                actual,
                expected,
                rtol=SLAB_STRICT_RTOL,
                atol=SLAB_STRICT_ATOL,
            )

    @pytest.mark.parametrize(
        (
            "compute_forces",
            "compute_charge_gradients",
            "compute_virial",
            "output_names",
        ),
        OUTPUT_CASES,
    )
    def test_full_ewald_matches_component_sum(
        self,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
        output_names,
        device,  # noqa: ARG002
    ):
        """Every Ewald slab flag combination preserves tuple order and values."""
        positions, charges, cell, pbc = _make_slab_system()
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc,
        )
        k_vectors = generate_k_vectors_ewald_summation(cell, EWALD_K_CUTOFF)

        full_outputs = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_vectors=k_vectors,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            pbc=pbc,
            slab_correction=True,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
        )
        real_outputs = ewald_real_space(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
        )
        reciprocal_outputs = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            EWALD_ALPHA,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
        )
        slab_outputs = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
        )

        expected = _component_sum(
            real_outputs,
            reciprocal_outputs,
            slab_outputs,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )
        full_named = _as_named_outputs(
            full_outputs,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )

        assert tuple(full_named) == tuple(output_names)
        for name, expected_value in expected.items():
            _assert_close(full_named[name], expected_value)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_full_ewald_slab_output_dtypes(self, dtype, device):  # noqa: ARG002
        """Ewald slab energies/charge gradients are fp64; vectors match input dtype."""
        positions, charges, cell, pbc = _make_slab_system(dtype=dtype)
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc,
        )

        energies, forces, charge_grads, virial = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_cutoff=EWALD_K_CUTOFF,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            pbc=pbc,
            slab_correction=True,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )

        assert positions.dtype == dtype
        assert charges.dtype == dtype
        assert cell.dtype == dtype
        assert energies.dtype == jnp.float64
        assert forces.dtype == dtype
        assert charge_grads.dtype == jnp.float64
        assert virial.dtype == dtype

    def test_full_ewald_energy_slab_autodiff_matches_standalone_outputs(self, device):  # noqa: ARG002
        """Energy-only Ewald slab contribution differentiates through full API."""
        positions, charges, cell, pbc = _make_slab_system()
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc,
        )
        common_kwargs = {
            "alpha": EWALD_ALPHA,
            "k_cutoff": EWALD_K_CUTOFF,
            "neighbor_matrix": neighbor_matrix,
            "neighbor_matrix_shifts": neighbor_matrix_shifts,
        }

        def slab_energy_from_full_api(pos, chg, cell_in):
            slab_energy = ewald_summation(
                pos,
                chg,
                cell_in,
                pbc=pbc,
                slab_correction=True,
                **common_kwargs,
            ).sum()
            base_energy = ewald_summation(
                pos,
                chg,
                cell_in,
                **common_kwargs,
            ).sum()
            return slab_energy - base_energy

        slab_outputs = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )

        _assert_full_slab_energy_autodiff_matches_outputs(
            slab_energy_from_full_api,
            positions,
            charges,
            cell,
            slab_outputs,
        )


class TestJaxPMESlabIntegration:
    """Full JAX PME slab wrapper composition and output order."""

    def test_neighbor_pbc_ttf_matches_ttt_with_vacuum(self, device):  # noqa: ARG002
        """PME slab outputs are unchanged by T/T/F vs T/T/T vacuum neighbors."""
        positions, charges, cell, pbc = _make_slab_system()
        pbc_3d = jnp.array([[True, True, True]], dtype=jnp.bool_)
        neighbor_matrix_slab, _, neighbor_matrix_shifts_slab = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc,
        )
        neighbor_matrix_3d, _, neighbor_matrix_shifts_3d = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc_3d,
        )
        common_kwargs = {
            "alpha": EWALD_ALPHA,
            "mesh_dimensions": PME_MESH,
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
            neighbor_matrix=neighbor_matrix_slab,
            neighbor_matrix_shifts=neighbor_matrix_shifts_slab,
            **common_kwargs,
        )
        outputs_3d = particle_mesh_ewald(
            positions,
            charges,
            cell,
            neighbor_matrix=neighbor_matrix_3d,
            neighbor_matrix_shifts=neighbor_matrix_shifts_3d,
            **common_kwargs,
        )

        for actual, expected in zip(outputs_slab, outputs_3d, strict=True):
            _assert_close(actual, expected, rtol=1e-9, atol=1e-10)

    @pytest.mark.parametrize(
        (
            "compute_forces",
            "compute_charge_gradients",
            "compute_virial",
            "output_names",
        ),
        OUTPUT_CASES,
    )
    def test_full_pme_outputs_match_component_sum(
        self,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
        output_names,
        device,  # noqa: ARG002
    ):
        """Every PME slab flag combination preserves tuple order and values."""
        positions, charges, cell, pbc = _make_slab_system()
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc,
        )

        full_outputs = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            mesh_dimensions=PME_MESH,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            pbc=pbc,
            slab_correction=True,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
        )
        real_outputs = ewald_real_space(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
        )
        reciprocal_outputs = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha=jnp.array([EWALD_ALPHA], dtype=positions.dtype),
            mesh_dimensions=PME_MESH,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
        )
        slab_outputs = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
        )

        expected = _component_sum(
            real_outputs,
            reciprocal_outputs,
            slab_outputs,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )
        full_named = _as_named_outputs(
            full_outputs,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )

        assert tuple(full_named) == tuple(output_names)
        for name, expected_value in expected.items():
            _assert_close(full_named[name], expected_value, rtol=1e-9, atol=1e-10)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_full_pme_slab_output_dtypes(self, dtype, device):  # noqa: ARG002
        """PME slab energies/charge gradients are fp64; vectors match input dtype."""
        positions, charges, cell, pbc = _make_slab_system(dtype=dtype)
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc,
        )

        energies, forces, charge_grads, virial = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            mesh_dimensions=PME_MESH,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            pbc=pbc,
            slab_correction=True,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )

        assert positions.dtype == dtype
        assert charges.dtype == dtype
        assert cell.dtype == dtype
        assert energies.dtype == jnp.float64
        assert forces.dtype == dtype
        assert charge_grads.dtype == jnp.float64
        assert virial.dtype == dtype

    def test_full_pme_energy_slab_autodiff_matches_standalone_outputs(self, device):  # noqa: ARG002
        """Energy-only PME slab contribution differentiates through full API."""
        positions, charges, cell, pbc = _make_slab_system()
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc,
        )
        common_kwargs = {
            "alpha": EWALD_ALPHA,
            "mesh_dimensions": PME_MESH,
            "neighbor_matrix": neighbor_matrix,
            "neighbor_matrix_shifts": neighbor_matrix_shifts,
        }

        def slab_energy_from_full_api(pos, chg, cell_in):
            slab_energy = particle_mesh_ewald(
                pos,
                chg,
                cell_in,
                pbc=pbc,
                slab_correction=True,
                **common_kwargs,
            ).sum()
            base_energy = particle_mesh_ewald(
                pos,
                chg,
                cell_in,
                **common_kwargs,
            ).sum()
            return slab_energy - base_energy

        slab_outputs = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )

        _assert_full_slab_energy_autodiff_matches_outputs(
            slab_energy_from_full_api,
            positions,
            charges,
            cell,
            slab_outputs,
        )

    def test_full_pme_slab_jit_smoke(self, device):  # noqa: ARG002
        """Energy-only full PME slab composition works under jax.jit."""
        positions, charges, cell, pbc = _make_slab_system()
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc,
        )

        def full_pme_slab(pos, chg, cell_in, nm, nms):
            return particle_mesh_ewald(
                pos,
                chg,
                cell_in,
                alpha=EWALD_ALPHA,
                mesh_dimensions=PME_MESH,
                neighbor_matrix=nm,
                neighbor_matrix_shifts=nms,
                pbc=pbc,
                slab_correction=True,
            )

        eager = full_pme_slab(
            positions,
            charges,
            cell,
            neighbor_matrix,
            neighbor_matrix_shifts,
        )
        jitted = jax.jit(full_pme_slab)(
            positions,
            charges,
            cell,
            neighbor_matrix,
            neighbor_matrix_shifts,
        )

        _assert_close(jitted, eager, rtol=1e-9, atol=1e-10)

    def test_full_pme_slab_3d_pbc_noop(self, device):  # noqa: ARG002
        """3D periodic PME slab mode matches standard PME."""
        positions, charges, cell, pbc = _make_slab_system()
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc,
        )
        common_kwargs = {
            "alpha": EWALD_ALPHA,
            "mesh_dimensions": PME_MESH,
            "neighbor_matrix": neighbor_matrix,
            "neighbor_matrix_shifts": neighbor_matrix_shifts,
            "compute_forces": True,
            "compute_charge_gradients": True,
            "compute_virial": True,
        }

        outputs_off = particle_mesh_ewald(positions, charges, cell, **common_kwargs)
        outputs_3d = particle_mesh_ewald(
            positions,
            charges,
            cell,
            pbc=jnp.array([True, True, True], dtype=jnp.bool_),
            slab_correction=True,
            **common_kwargs,
        )

        for actual, expected in zip(outputs_3d, outputs_off, strict=True):
            _assert_close(actual, expected, rtol=1e-9, atol=1e-10)

    def test_mixed_pbc_batch_matches_component_sum(self, device):  # noqa: ARG002
        """Batched PME slab applies correction only to slab-like systems."""
        positions, charges, cell, batch_idx, pbc = _make_batched_system()
        neighbor_matrix, _, neighbor_matrix_shifts = batch_cell_list(
            positions,
            REAL_SPACE_CUTOFF,
            cell,
            pbc,
            batch_idx=batch_idx,
            max_neighbors=32,
        )
        alpha = jnp.array([EWALD_ALPHA, EWALD_ALPHA], dtype=positions.dtype)

        full_outputs = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=PME_MESH,
            batch_idx=batch_idx,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            pbc=pbc,
            slab_correction=True,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        pme_3d_outputs = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=PME_MESH,
            batch_idx=batch_idx,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        slab_outputs = compute_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            batch_idx=batch_idx,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )

        for actual, pme_3d, slab in zip(full_outputs, pme_3d_outputs, slab_outputs):
            _assert_close(actual, pme_3d + slab, rtol=1e-9, atol=1e-10)

        _assert_close(slab_outputs[0][3:], jnp.zeros_like(slab_outputs[0][3:]))
        _assert_close(slab_outputs[1][3:], jnp.zeros_like(slab_outputs[1][3:]))
        _assert_close(slab_outputs[2][3:], jnp.zeros_like(slab_outputs[2][3:]))
        _assert_close(slab_outputs[3][1], jnp.zeros_like(slab_outputs[3][1]))

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

from __future__ import annotations

import importlib
import warnings

import jax
import jax.numpy as jnp
import pytest

from nvalchemiops.jax.interactions.electrostatics import (
    compute_slab_correction,
    ewald_real_space,
    ewald_reciprocal_space,
    generate_k_vectors_ewald_summation,
    generate_k_vectors_pme,
    particle_mesh_ewald,
    pme_reciprocal_space,
)

pytestmark = pytest.mark.gpu


def _system(device):  # noqa: ARG001
    """Build a tiny neutral periodic system."""
    positions = jnp.array(
        [[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.6, 1.1, 0.5]],
        dtype=jnp.float64,
    )
    charges = jnp.array([0.7, -0.4, -0.3], dtype=jnp.float64)
    cell = jnp.eye(3, dtype=jnp.float64) * 5.0
    alpha = jnp.array([0.35], dtype=jnp.float64)
    return positions, charges, cell, alpha


def _dense_neighbors() -> tuple[jax.Array, jax.Array]:
    """Return all atom pairs as a tiny dense neighbor matrix."""
    neighbor_matrix = jnp.array(
        [[1, 2], [0, 2], [0, 1]],
        dtype=jnp.int32,
    )
    neighbor_shifts = jnp.zeros((3, 2, 3), dtype=jnp.int32)
    return neighbor_matrix, neighbor_shifts


def _assert_no_cache_warning(records: list[warnings.WarningMessage]) -> None:
    """Assert no static-cache contract warning was emitted."""
    messages = [str(record.message) for record in records]
    assert not any("Precomputed" in message for message in messages)
    assert not any("current cell" in message for message in messages)


def _finite_difference_positions(fn, positions):
    """Return central finite-difference position gradients for tiny systems."""
    eps = jnp.asarray(1e-4, dtype=positions.dtype)
    rows = []
    for atom in range(positions.shape[0]):
        comps = []
        for dim in range(3):
            delta = jnp.zeros_like(positions).at[atom, dim].set(eps)
            comps.append((fn(positions + delta) - fn(positions - delta)) / (2.0 * eps))
        rows.append(jnp.stack(comps))
    return jnp.stack(rows)


def _finite_difference_charges(fn, charges):
    """Return central finite-difference charge gradients for tiny systems."""
    eps = jnp.asarray(1e-4, dtype=charges.dtype)
    comps = []
    for atom in range(charges.shape[0]):
        delta = jnp.zeros_like(charges).at[atom].set(eps)
        comps.append((fn(charges + delta) - fn(charges - delta)) / (2.0 * eps))
    return jnp.stack(comps)


def test_jax_ewald_real_weighted_loss_matches_finite_difference(device) -> None:
    """Ewald real-space per-atom energy supports non-uniform weighted losses."""
    positions, charges, cell, alpha = _system(device)
    neighbor_matrix, neighbor_shifts = _dense_neighbors()
    weights = jnp.array([1.1, -0.5, 0.3], dtype=jnp.float64)

    def loss_pos(pos):
        energies = ewald_real_space(
            pos,
            charges,
            cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
        )
        return (weights * energies).sum()

    def loss_chg(chg):
        energies = ewald_real_space(
            positions,
            chg,
            cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
        )
        return (weights * energies).sum()

    grad_positions = jax.grad(loss_pos)(positions)
    grad_charges = jax.grad(loss_chg)(charges)
    fd_positions = _finite_difference_positions(loss_pos, positions)
    fd_charges = _finite_difference_charges(loss_chg, charges)

    assert jnp.allclose(grad_positions, fd_positions, rtol=1e-4, atol=1e-7)
    assert jnp.allclose(grad_charges, fd_charges, rtol=1e-4, atol=1e-7)
    assert jnp.all(jnp.isfinite(jax.jit(jax.grad(loss_chg))(charges)))


def test_jax_pme_ignores_alpha_tangent(device) -> None:
    """PME alpha is a setup constant in JAX custom-JVP rules."""
    positions, charges, cell, alpha = _system(device)

    def energy_alpha(alpha_arg):
        return pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha=alpha_arg,
            mesh_dimensions=(8, 8, 8),
        ).sum()

    _value, tangent = jax.jvp(
        energy_alpha,
        (alpha,),
        (jnp.ones_like(alpha),),
    )
    assert jnp.allclose(tangent, 0.0)


def test_jax_ewald_reciprocal_silently_accepts_cell_tangent_with_k_vectors(
    device,
) -> None:
    """Public Ewald reciprocal k-vectors are static cell-JVP metadata."""
    positions, charges, cell, alpha = _system(device)
    cell_static = jax.lax.stop_gradient(cell)
    k_vectors = generate_k_vectors_ewald_summation(cell_static, k_cutoff=5.0)

    def energy_cell(cell_arg):
        return ewald_reciprocal_space(
            positions,
            charges,
            cell_arg,
            k_vectors,
            alpha,
        ).sum()

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        value, tangent = jax.jvp(energy_cell, (cell,), (jnp.ones_like(cell),))
    _assert_no_cache_warning(records)
    assert jnp.isfinite(value)
    assert jnp.isfinite(tangent)


@pytest.mark.parametrize(
    "cache_name", ["k_vectors", "k_squared", "volume", "cell_inv_t"]
)
def test_jax_pme_silently_accepts_cell_tangent_with_precomputed_metadata(
    device,
    cache_name: str,
) -> None:
    """PME reciprocal metadata is static for JAX cell JVPs."""
    positions, charges, cell, alpha = _system(device)
    mesh_dimensions = (8, 8, 8)
    cell_static = jax.lax.stop_gradient(cell)
    k_vectors, k_squared = generate_k_vectors_pme(cell_static, mesh_dimensions)
    caches = {
        "k_vectors": k_vectors,
        "k_squared": k_squared,
        "volume": jnp.abs(jnp.linalg.det(cell_static)).reshape(1),
        "cell_inv_t": jnp.linalg.inv(cell_static).T,
    }

    def energy_cell(cell_arg):
        return pme_reciprocal_space(
            positions,
            charges,
            cell_arg,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            **{cache_name: caches[cache_name]},
        ).sum()

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        value, tangent = jax.jvp(energy_cell, (cell,), (jnp.ones_like(cell),))
    _assert_no_cache_warning(records)
    assert jnp.isfinite(value)
    assert jnp.isfinite(tangent)


@pytest.mark.parametrize(
    "cache_name", ["k_vectors", "k_squared", "volume", "cell_inv_t"]
)
def test_jax_pme_cache_source_cell_tangent_is_static(
    device,
    cache_name: str,
) -> None:
    """PME metadata-producing cell tangents do not contribute to public JVPs."""
    positions, charges, cell, alpha = _system(device)
    mesh_dimensions = (8, 8, 8)
    current_cell = cell
    cache_cell = cell * jnp.asarray(1.02, dtype=cell.dtype)

    def energy_cell(current_cell_arg, cache_cell_arg):
        k_vectors, k_squared = generate_k_vectors_pme(cache_cell_arg, mesh_dimensions)
        caches = {
            "k_vectors": k_vectors,
            "k_squared": k_squared,
            "volume": jnp.abs(jnp.linalg.det(cache_cell_arg)).reshape(1),
            "cell_inv_t": jnp.linalg.inv(cache_cell_arg).T,
        }
        return pme_reciprocal_space(
            positions,
            charges,
            current_cell_arg,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            **{cache_name: caches[cache_name]},
        ).sum()

    current_tangent = jnp.ones_like(current_cell)
    cache_tangent = jnp.ones_like(cache_cell)
    _value_static, tangent_static = jax.jvp(
        energy_cell,
        (current_cell, cache_cell),
        (current_tangent, jnp.zeros_like(cache_cell)),
    )
    value_dynamic, tangent_dynamic = jax.jvp(
        energy_cell,
        (current_cell, cache_cell),
        (current_tangent, cache_tangent),
    )

    assert jnp.isfinite(value_dynamic)
    assert jnp.isfinite(tangent_dynamic)
    assert jnp.allclose(tangent_dynamic, tangent_static, rtol=1e-9, atol=1e-10)


def test_jax_pme_threads_static_volume_to_corrections(device, monkeypatch) -> None:
    """Caller-supplied PME volume is used consistently as static metadata."""
    positions, _charges, cell, alpha = _system(device)
    charges = jnp.array([0.7, -0.4, 0.2], dtype=jnp.float64)
    mesh_dimensions = (8, 8, 8)
    volume = jax.lax.stop_gradient(jnp.abs(jnp.linalg.det(cell)).reshape(1))
    pme_module = importlib.import_module(
        "nvalchemiops.jax.interactions.electrostatics.pme"
    )
    seen: dict[str, jax.Array | None] = {}

    original_corrections = pme_module.pme_energy_corrections
    original_virial_bg = pme_module.pme_virial_bg_correction

    def corrections_wrapper(
        raw_energies,
        charges_arg,
        cell_arg,
        alpha_arg,
        batch_idx=None,
        volume=None,
    ):
        seen["corrections"] = volume
        return original_corrections(
            raw_energies,
            charges_arg,
            cell_arg,
            alpha_arg,
            batch_idx,
            volume=volume,
        )

    def virial_bg_wrapper(
        charges,
        cell,
        alpha,
        virial,
        batch_idx=None,
        volume=None,
    ):
        seen["virial_bg"] = volume
        return original_virial_bg(
            charges,
            cell,
            alpha,
            virial,
            batch_idx,
            volume=volume,
        )

    monkeypatch.setattr(pme_module, "pme_energy_corrections", corrections_wrapper)
    monkeypatch.setattr(pme_module, "pme_virial_bg_correction", virial_bg_wrapper)

    energy = pme_reciprocal_space(
        positions,
        charges,
        cell,
        alpha=alpha,
        mesh_dimensions=mesh_dimensions,
        volume=volume,
    )
    assert jnp.isfinite(energy).all()
    assert seen["corrections"] is not None
    assert jnp.allclose(seen["corrections"], volume)

    with pytest.warns(DeprecationWarning):
        outputs = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            compute_virial=True,
            volume=volume,
        )
    energies, _virial = outputs
    assert jnp.isfinite(energies).all()
    assert seen["virial_bg"] is not None
    assert jnp.allclose(seen["virial_bg"], volume)


def test_jax_pme_reciprocal_weighted_loss_matches_finite_difference(device) -> None:
    """PME reciprocal per-atom energy supports non-uniform weighted losses."""
    positions, charges, cell, alpha = _system(device)
    weights = jnp.array([0.2, -0.7, 1.4], dtype=jnp.float64)

    def loss_pos(pos):
        energies = pme_reciprocal_space(
            pos,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=(8, 8, 8),
        )
        return (weights * energies).sum()

    def loss_chg(chg):
        energies = pme_reciprocal_space(
            positions,
            chg,
            cell,
            alpha=alpha,
            mesh_dimensions=(8, 8, 8),
        )
        return (weights * energies).sum()

    grad_positions = jax.grad(loss_pos)(positions)
    grad_charges = jax.grad(loss_chg)(charges)
    fd_positions = _finite_difference_positions(loss_pos, positions)
    fd_charges = _finite_difference_charges(loss_chg, charges)

    assert jnp.allclose(grad_positions, fd_positions, rtol=2e-3, atol=2e-5)
    assert jnp.allclose(grad_charges, fd_charges, rtol=2e-3, atol=2e-5)
    assert jnp.all(jnp.isfinite(jax.jit(jax.grad(loss_pos))(positions)))


@pytest.mark.parametrize("slab_correction", [False, True])
def test_jax_full_pme_weighted_loss_matches_finite_difference(
    device,
    slab_correction: bool,
) -> None:
    """Full PME per-atom energy supports non-uniform weighted losses."""
    positions, charges, cell, alpha = _system(device)
    neighbor_matrix, neighbor_shifts = _dense_neighbors()
    pbc = jnp.array([[True, True, False]])
    weights = jnp.array([-0.3, 0.8, 1.5], dtype=jnp.float64)

    def loss_pos(pos):
        energies = particle_mesh_ewald(
            pos,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=(8, 8, 8),
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
            pbc=pbc,
            slab_correction=slab_correction,
        )
        return (weights * energies).sum()

    def loss_chg(chg):
        energies = particle_mesh_ewald(
            positions,
            chg,
            cell,
            alpha=alpha,
            mesh_dimensions=(8, 8, 8),
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
            pbc=pbc,
            slab_correction=slab_correction,
        )
        return (weights * energies).sum()

    grad_positions = jax.grad(loss_pos)(positions)
    grad_charges = jax.grad(loss_chg)(charges)
    fd_positions = _finite_difference_positions(loss_pos, positions)
    fd_charges = _finite_difference_charges(loss_chg, charges)

    assert jnp.allclose(grad_positions, fd_positions, rtol=3e-3, atol=3e-5)
    assert jnp.allclose(grad_charges, fd_charges, rtol=3e-3, atol=3e-5)


def test_jax_slab_weighted_loss_matches_finite_difference(device) -> None:
    """Slab per-atom energy supports non-uniform weighted losses."""
    positions, charges, cell, _alpha = _system(device)
    pbc = jnp.array([[True, True, False]])
    weights = jnp.array([0.4, 1.3, -0.6], dtype=jnp.float64)

    def loss_pos(pos):
        energies = compute_slab_correction(pos, charges, cell, pbc)
        return (weights * energies).sum()

    def loss_chg(chg):
        energies = compute_slab_correction(positions, chg, cell, pbc)
        return (weights * energies).sum()

    grad_positions = jax.grad(loss_pos)(positions)
    grad_charges = jax.grad(loss_chg)(charges)
    fd_positions = _finite_difference_positions(loss_pos, positions)
    fd_charges = _finite_difference_charges(loss_chg, charges)

    assert jnp.allclose(grad_positions, fd_positions, rtol=1e-4, atol=1e-7)
    assert jnp.allclose(grad_charges, fd_charges, rtol=1e-4, atol=1e-7)
    assert jnp.all(jnp.isfinite(jax.jit(jax.grad(loss_chg))(charges)))

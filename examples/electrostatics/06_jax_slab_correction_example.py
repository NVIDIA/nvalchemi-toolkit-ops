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
2D Slab Correction for Ewald and PME with JAX
=============================================

This example demonstrates how to apply the Yeh-Berkowitz / Ballenegger
two-dimensional slab correction to JAX Ewald and Particle Mesh Ewald (PME)
electrostatics. The correction is used for slab-like systems with two periodic
directions and one non-periodic direction, such as interfaces with vacuum
padding.

In this example you will learn:

- How to run ``ewald_summation(..., slab_correction=True)`` in JAX
- How to run ``particle_mesh_ewald(..., slab_correction=True)`` in JAX
- How to pass slab periodicity with a boolean ``pbc`` array
- How to compute the standalone slab correction with ``compute_slab_correction``
- How the standalone correction equals the integrated Ewald energy/force delta
- How to derive slab forces from energy autograd for training
- How to compose total Ewald and PME component workflows with slab
- How triclinic slab cells use the normal to the periodic plane
- How to use ``jax.jit`` with a full neighbor list + PME slab pipeline

.. important::
    This script is intended as an API demonstration. Do not use this script
    for performance benchmarking; refer to the `benchmarks` folder instead.
"""

# %%
# Setup and Imports
# -----------------
# The slab correction is available through the high-level JAX Ewald and PME
# APIs, and as a standalone helper. The standalone helper is useful for
# debugging, validation, and adding the correction when composing total
# component workflows.

from __future__ import annotations

import sys
import warnings

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print(
        "This example requires JAX. Install with: pip install 'nvalchemi-toolkit-ops[jax]'"
    )
    sys.exit(0)

try:
    from nvalchemiops.jax.interactions.electrostatics import (
        compute_slab_correction,
        ewald_real_space,
        ewald_reciprocal_space,
        ewald_summation,
        generate_k_vectors_ewald_summation,
        particle_mesh_ewald,
        pme_reciprocal_space,
    )
    from nvalchemiops.jax.neighbors import neighbor_list as neighbor_list_fn
    from nvalchemiops.jax.neighbors.naive import naive_neighbor_list
    from nvalchemiops.jax.neighbors.neighbor_utils import compute_naive_num_shifts
except Exception as exc:
    print(
        f"JAX/Warp backend unavailable ({exc}). This example requires a CUDA-backed runtime."
    )
    sys.exit(0)


def _legacy_direct_output_call(function, *args, **kwargs):
    """Call a deprecated direct-output path used for explicit migration checks."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="The direct-output flags.*",
        )
        return function(*args, **kwargs)


def _format_array(array: jax.Array) -> str:
    """Format small arrays consistently for gallery text output."""
    return np.array2string(
        np.asarray(jax.device_get(array)),
        precision=6,
        suppress_small=False,
    )


# %%
# Check Device
# ------------

print("=" * 70)
print("JAX 2D SLAB CORRECTION EXAMPLE")
print("=" * 70)

devices = jax.devices()
print(f"\nJAX devices: {devices}")
print(f"Default backend: {jax.default_backend()}")

# %%
# Create a Small Slab System
# --------------------------
# We use a two-ion CsCl-like system in a cell with a long z direction. The
# long cell vector represents vacuum padding normal to the slab.
#
# ``pbc_slab`` marks x and y as periodic and z as non-periodic. Batched slab
# simulations should pass an explicit ``(B, 3)`` array so each system carries
# its own slab geometry.


def create_cscl_slab_system() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Create a small T/T/F slab system with vacuum along z."""
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        dtype=jnp.float64,
    )
    charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
    cell = jnp.diag(jnp.array([10.0, 10.0, 30.0], dtype=jnp.float64))[None, :, :]
    pbc_slab = jnp.array([[True, True, False]], dtype=jnp.bool_)
    return positions, charges, cell, pbc_slab


positions, charges, cell, pbc_slab = create_cscl_slab_system()

print("Slab system:")
print(f"  Number of atoms: {positions.shape[0]}")
print(f"  Cell rows:\n{_format_array(cell[0])}")
print(f"  Slab pbc: {_format_array(pbc_slab[0])}")
print(f"  Total charge: {float(charges.sum()):.1f}")

# %%
# Build the Real-Space Neighbor List
# ----------------------------------
# The neighbor list controls real-space periodic images. For this slab setup,
# use the same T/T/F periodicity and a cell with enough vacuum along z.

alpha = 0.3
real_space_cutoff = 5.0
k_cutoff = 2.5
mesh_dimensions = (16, 16, 16)
alpha_array = jnp.array([alpha], dtype=jnp.float64)

neighbor_list, neighbor_ptr, neighbor_shifts = neighbor_list_fn(
    positions,
    real_space_cutoff,
    cell=cell,
    pbc=pbc_slab,
    return_neighbor_list=True,
)

print("\nNeighbor list:")
print(f"  Number of neighbor entries: {neighbor_list.shape[1]}")
print(f"  Real-space cutoff: {real_space_cutoff:.1f} Å")

# %%
# Standard 3D Ewald (Legacy Direct-Output Baseline)
# -------------------------------------------------
# First compute the uncorrected 3D-periodic Ewald result. This is the quantity
# that will receive the slab correction. This section intentionally uses
# deprecated full-API direct-output flags only to build a compatibility baseline
# for the checks below.

energies_3d, forces_3d, charge_grads_3d, virial_3d = _legacy_direct_output_call(
    ewald_summation,
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=alpha,
    k_cutoff=k_cutoff,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
    compute_charge_gradients=True,
    compute_virial=True,
)

print("\nStandard 3D Ewald:")
print(f"  Total energy: {float(energies_3d.sum()): .8f}")
print(f"  Max force magnitude: {float(jnp.linalg.norm(forces_3d, axis=1).max()): .8f}")
print(f"  Charge gradients: {_format_array(charge_grads_3d)}")
print(f"  Virial trace: {float(jnp.trace(virial_3d[0])): .8f}")

# %%
# Ewald with Slab Correction (Legacy Direct-Output Check)
# ------------------------------------------------------
# Set ``slab_correction=True`` and pass the slab periodicity. The output tuple
# follows the same ordering as ordinary Ewald: energies, forces, charge
# gradients, and virial when all optional quantities are requested. For training
# code, copy the energy-autograd section immediately below instead.

energies_slab, forces_slab, charge_grads_slab, virial_slab = _legacy_direct_output_call(
    ewald_summation,
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=alpha,
    k_cutoff=k_cutoff,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
    compute_charge_gradients=True,
    compute_virial=True,
    pbc=pbc_slab,
    slab_correction=True,
)

print("\nEwald with slab correction:")
print(f"  Total energy: {float(energies_slab.sum()): .8f}")
print(f"  Energy delta: {float((energies_slab - energies_3d).sum()): .8f}")
print(
    f"  Max force magnitude: {float(jnp.linalg.norm(forces_slab, axis=1).max()): .8f}"
)
print(f"  Charge gradients: {_format_array(charge_grads_slab)}")
print(f"  Virial trace: {float(jnp.trace(virial_slab[0])): .8f}")

# %%
# Energy-Autograd Slab Forces
# ---------------------------
# For differentiable training, call the full API without direct-output flags and
# derive forces from the returned per-atom energies. The direct-output calls
# above are legacy compatibility checks.


def ewald_slab_total_energy(pos: jax.Array) -> jax.Array:
    """Return total slab-corrected Ewald energy for force autograd."""
    return jnp.sum(
        ewald_summation(
            positions=pos,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            pbc=pbc_slab,
            slab_correction=True,
        )
    )


ewald_forces_ag = -jax.grad(ewald_slab_total_energy)(positions)
print("\nEwald slab energy-autograd force check:")
print(
    f"  Max force delta vs legacy direct: {float(jnp.max(jnp.abs(ewald_forces_ag - forces_slab))):.2e}"
)

# %%
# Standalone Slab Correction
# --------------------------
# The same correction can be computed directly. The standalone result equals
# the difference between the slab-corrected and uncorrected Ewald outputs.

correction_energy, correction_forces, correction_charge_grads, correction_virial = (
    compute_slab_correction(
        positions=positions,
        charges=charges,
        cell=cell,
        pbc=pbc_slab,
        compute_forces=True,
        compute_charge_gradients=True,
        compute_virial=True,
    )
)

energy_delta_error = jnp.max(jnp.abs((energies_slab - energies_3d) - correction_energy))
force_delta_error = jnp.max(jnp.abs((forces_slab - forces_3d) - correction_forces))
charge_grad_delta_error = jnp.max(
    jnp.abs((charge_grads_slab - charge_grads_3d) - correction_charge_grads)
)
virial_delta_error = jnp.max(jnp.abs((virial_slab - virial_3d) - correction_virial))

print("\nStandalone correction:")
print(f"  Correction energy: {float(correction_energy.sum()): .8f}")
print(f"  Max energy delta error: {float(energy_delta_error):.2e}")
print(f"  Max force delta error: {float(force_delta_error):.2e}")
print(f"  Max charge-gradient delta error: {float(charge_grad_delta_error):.2e}")
print(f"  Max virial delta error: {float(virial_delta_error):.2e}")

# %%
# Total Ewald Component Composition
# ---------------------------------
# If you need the Ewald real-space, Ewald reciprocal-space, and slab terms
# separately, compute the three pieces explicitly and add matching outputs.

ewald_component_real_energy, ewald_component_real_forces = ewald_real_space(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=alpha_array,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
)
ewald_k_vectors = generate_k_vectors_ewald_summation(
    jax.lax.stop_gradient(cell), k_cutoff
)
ewald_component_reciprocal_energy, ewald_component_reciprocal_forces = (
    ewald_reciprocal_space(
        positions=positions,
        charges=charges,
        cell=cell,
        k_vectors=ewald_k_vectors,
        alpha=alpha_array,
        compute_forces=True,
    )
)
ewald_component_slab_energy, ewald_component_slab_forces = compute_slab_correction(
    positions=positions,
    charges=charges,
    cell=cell,
    pbc=pbc_slab,
    compute_forces=True,
)

total_ewald_energy = (
    ewald_component_real_energy
    + ewald_component_reciprocal_energy
    + ewald_component_slab_energy
)
total_ewald_forces = (
    ewald_component_real_forces
    + ewald_component_reciprocal_forces
    + ewald_component_slab_forces
)

print("\nTotal Ewald composition from components:")
print(
    f"  Max energy error: {float(jnp.max(jnp.abs(total_ewald_energy - energies_slab))):.2e}"
)
print(
    f"  Max force error: {float(jnp.max(jnp.abs(total_ewald_forces - forces_slab))):.2e}"
)

# %%
# PME with Slab Correction (Legacy Direct-Output Check)
# ----------------------------------------------------
# Full PME accepts the same slab correction arguments as Ewald. The reciprocal
# PME component itself remains a 3D-periodic reciprocal-space calculation; the
# slab term is added by the high-level ``particle_mesh_ewald`` wrapper. This
# direct-output tuple is a legacy compatibility check; the following section is
# the training recipe.

pme_3d_energy, pme_3d_forces = _legacy_direct_output_call(
    particle_mesh_ewald,
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=alpha,
    mesh_dimensions=mesh_dimensions,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
)

pme_slab_energy, pme_slab_forces = _legacy_direct_output_call(
    particle_mesh_ewald,
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=alpha,
    mesh_dimensions=mesh_dimensions,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
    pbc=pbc_slab,
    slab_correction=True,
)

print("\nPME with slab correction:")
print(f"  Standard PME energy: {float(pme_3d_energy.sum()): .8f}")
print(f"  Slab PME energy: {float(pme_slab_energy.sum()): .8f}")
print(f"  Energy delta: {float((pme_slab_energy - pme_3d_energy).sum()): .8f}")
print(
    f"  Max force magnitude: {float(jnp.linalg.norm(pme_slab_forces, axis=1).max()): .8f}"
)

# %%
# PME Energy-Autograd Slab Forces
# -------------------------------
# PME uses the same training recipe: omit direct-output flags and differentiate
# the energy. This is the path to copy into force-loss training code.


def pme_slab_total_energy(pos: jax.Array) -> jax.Array:
    """Return total slab-corrected PME energy for force autograd."""
    return jnp.sum(
        particle_mesh_ewald(
            positions=pos,
            charges=charges,
            cell=cell,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            pbc=pbc_slab,
            slab_correction=True,
        )
    )


pme_forces_ag = -jax.grad(pme_slab_total_energy)(positions)
print("\nPME slab energy-autograd force check:")
print(
    f"  Max force delta vs legacy direct: {float(jnp.max(jnp.abs(pme_forces_ag - pme_slab_forces))):.2e}"
)

# %%
# Total PME Component Composition
# -------------------------------
# If you need real-space, PME reciprocal-space, and slab terms separately,
# compute the three pieces explicitly and add matching outputs.

real_energy, real_forces = ewald_real_space(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=alpha_array,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
)
reciprocal_energy, reciprocal_forces = pme_reciprocal_space(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=alpha_array,
    mesh_dimensions=mesh_dimensions,
    compute_forces=True,
)
slab_energy_correction, slab_forces_correction = compute_slab_correction(
    positions=positions,
    charges=charges,
    cell=cell,
    pbc=pbc_slab,
    compute_forces=True,
)

total_pme_energy = real_energy + reciprocal_energy + slab_energy_correction
total_pme_forces = real_forces + reciprocal_forces + slab_forces_correction

print("\nTotal PME composition from components:")
print(
    f"  Max energy error: {float(jnp.max(jnp.abs(total_pme_energy - pme_slab_energy))):.2e}"
)
print(
    f"  Max force error: {float(jnp.max(jnp.abs(total_pme_forces - pme_slab_forces))):.2e}"
)

# %%
# Triclinic Slab Cells
# --------------------
# Triclinic cells are also supported. The slab normal follows the plane spanned
# by the two periodic cell vectors; it is not locked to a Cartesian axis.
#
# Here we reuse the same positions and charges with a tilted cell and compute
# the standalone correction.

triclinic_cell = jnp.array(
    [[[10.0, 0.0, 0.0], [1.5, 9.0, 0.8], [0.2, 0.4, 30.0]]],
    dtype=jnp.float64,
)

triclinic_energy, triclinic_forces = compute_slab_correction(
    positions=positions,
    charges=charges,
    cell=triclinic_cell,
    pbc=pbc_slab,
    compute_forces=True,
)

print("\nTriclinic standalone correction:")
print(f"  Cell rows:\n{_format_array(triclinic_cell[0])}")
print(f"  Correction energy: {float(triclinic_energy.sum()): .8f}")
print(f"  Forces:\n{_format_array(triclinic_forces)}")

# %%
# JIT Compilation
# ---------------
# Demonstrate combining the neighbor list build and PME slab calculation into a
# single ``jax.jit``-compiled function. This compiles the array program into one
# callable while keeping launch-size metadata static.
#
# For JIT compatibility:
#
# - ``max_neighbors`` must be specified (static array shapes)
# - ``mesh_dimensions`` must be a concrete tuple (static FFT sizes)
# - ``alpha`` can be a traced JAX array
# - ``compute_forces`` and other boolean flags must be static
# - Periodic shift metadata (``shift_range``, ``num_shifts_per_system``,
#   ``max_shifts_per_system``) must be pre-computed outside jit using
#   ``compute_naive_num_shifts``, since the launch dimensions must be concrete

print("\n" + "=" * 70)
print("JIT COMPILATION")
print("=" * 70)

jit_positions, jit_charges, jit_cell, jit_pbc_slab = create_cscl_slab_system()
jit_alpha = jnp.array([alpha], dtype=jnp.float64)
jit_max_neighbors = 16
jit_mask_value = jit_positions.shape[0]

shift_range, num_shifts_per_system, max_shifts_per_system = compute_naive_num_shifts(
    jit_cell,
    real_space_cutoff,
    jit_pbc_slab,
)


def compute_pme_slab_energy_forces(
    positions_in: jax.Array,
    charges_in: jax.Array,
    cell_in: jax.Array,
    pbc_slab_in: jax.Array,
    alpha_in: jax.Array,
    shift_range_per_dimension: jax.Array = shift_range,
    num_shifts: jax.Array = num_shifts_per_system,
    cutoff: float = real_space_cutoff,
    max_neighbors: int = jit_max_neighbors,
    max_shifts: int = max_shifts_per_system,
    mask_value: int = jit_mask_value,
    mesh_dims: tuple[int, int, int] = mesh_dimensions,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compatible neighbor matrix + PME slab pipeline."""

    def total_energy(pos):
        neighbor_matrix, _, neighbor_matrix_shifts = naive_neighbor_list(
            pos,
            cutoff,
            cell=cell_in,
            pbc=pbc_slab_in,
            max_neighbors=max_neighbors,
            fill_value=mask_value,
            shift_range_per_dimension=shift_range_per_dimension,
            num_shifts_per_system=num_shifts,
            max_shifts_per_system=max_shifts,
        )

        energies = particle_mesh_ewald(
            positions=pos,
            charges=charges_in,
            cell=cell_in,
            alpha=alpha_in,
            mesh_dimensions=mesh_dims,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            mask_value=mask_value,
            pbc=pbc_slab_in,
            slab_correction=True,
        )
        return energies.sum()

    energy, grad_positions = jax.value_and_grad(total_energy)(positions_in)
    return energy, -grad_positions


jit_compute_pme_slab_energy_forces = jax.jit(compute_pme_slab_energy_forces)

energies_nonjit, forces_nonjit = compute_pme_slab_energy_forces(
    jit_positions,
    jit_charges,
    jit_cell,
    jit_pbc_slab,
    jit_alpha,
)
energies_nonjit.block_until_ready()
forces_nonjit.block_until_ready()

print("  Non-jitted PME slab:")
print(f"    Total energy: {float(energies_nonjit.sum()): .8f}")
print(
    f"    Max force magnitude: {float(jnp.linalg.norm(forces_nonjit, axis=1).max()): .8f}"
)

print("\nCompiling and running jitted PME slab pipeline...")
energies_jit, forces_jit = jit_compute_pme_slab_energy_forces(
    jit_positions,
    jit_charges,
    jit_cell,
    jit_pbc_slab,
    jit_alpha,
)
energies_jit.block_until_ready()
forces_jit.block_until_ready()

print("  JIT PME slab:")
print(f"    Total energy: {float(energies_jit.sum()): .8f}")
print(
    f"    Max force magnitude: {float(jnp.linalg.norm(forces_jit, axis=1).max()): .8f}"
)
print(
    f"    Max energy error vs non-jit: {float(jnp.max(jnp.abs(energies_jit - energies_nonjit))):.2e}"
)
print(
    f"    Max force error vs non-jit: {float(jnp.max(jnp.abs(forces_jit - forces_nonjit))):.2e}"
)

# %%
# Summary
# -------
# Use ``ewald_summation(..., slab_correction=True, pbc=pbc_slab)`` or
# ``particle_mesh_ewald(..., slab_correction=True, pbc=pbc_slab)`` when you want
# the correction included in the total energy. For training derivatives, copy
# the energy-autograd force sections above. Use ``compute_slab_correction``
# directly when you need the correction term alone or when composing
# ``ewald_real_space`` with ``ewald_reciprocal_space`` or
# ``pme_reciprocal_space``.
#
# For ``jax.jit`` workflows, keep shape-determining values outside the jitted
# function: neighbor capacity, shift metadata, FFT mesh dimensions, and output
# flags should all be static.

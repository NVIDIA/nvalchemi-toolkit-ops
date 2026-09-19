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

"""Drives the FourierD3 Warp launchers across the transforms, for testing.

No single Warp call spans the pipeline, because Warp has no full-mesh FFT and the transforms
are the calling framework's job. Shipping bindings will interleave ``torch.fft`` or
``jax.numpy.fft``; this harness does the same with NumPy so the Warp layer can be exercised
end to end without pulling in a framework.

It is a test artifact rather than package API.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from nvalchemiops.interactions.dispersion import _fourier_dftd3 as fd3

__all__ = ["bspline_moduli", "fourier_d3_energy"]


def bspline_moduli(miller, mesh_size, spline_order):
    """B-spline attenuation for one mesh axis.

    Interpolating onto a mesh damps each frequency; dividing it out recovers the structure
    factor the mesh stands in for. Same ``sinc(m / N) ** p`` convention as the bindings and
    the in-repo PME.
    """
    return np.sinc(miller / mesh_size) ** spline_order


def _wp_types(np_dtype):
    """Warp scalar, vector, matrix and pair dtypes matching a NumPy floating dtype."""
    if np_dtype == np.float64:
        return wp.float64, wp.vec3d, wp.mat33d, wp.vec2d
    return wp.float32, wp.vec3f, wp.mat33f, wp.vec2f


def fourier_d3_energy(
    positions,
    numbers,
    channels,
    batch_idx,
    cells,
    rcov,
    decomposition,
    sqrt_q,
    targets,
    pointer,
    shifts_cartesian,
    r_cut,
    mesh_dimensions,
    damping,
    spline_order=4,
    device="cuda:0",
    np_dtype=np.float64,
    compute_forces=True,
    compute_virial=False,
    coordination_override=None,
    apply_cn_chain=True,
):
    """Run the whole FourierD3 evaluation and return its outputs.

    Parameters mirror the Warp launchers. ``damping`` is ``(s6, s8, a1, a2)``.
    ``coordination_override`` substitutes coordination numbers instead of computing them and
    ``apply_cn_chain`` skips the chain rule; together they let a test hold the coordination
    numbers fixed and see the mesh path on its own.

    Returns
    -------
    dict
        ``energy`` per system, plus ``forces``, ``virial``, ``coordination``, ``c6`` and
        ``d_energy_d_c6`` for inspection.
    """
    s6, s8, a1, a2 = damping
    wp_dtype, vec_dtype, mat_dtype, pair_dtype = _wp_types(np_dtype)
    n_atoms = len(positions)
    n_systems = len(cells)
    n_species = decomposition.n_species
    rank = decomposition.rank
    n_channels = n_species * rank
    nx, ny, nz = mesh_dimensions

    def to_wp(array, dtype):
        return wp.array(
            np.ascontiguousarray(array, dtype=np_dtype), dtype=dtype, device=device
        )

    def to_int(array):
        return wp.array(
            np.ascontiguousarray(array, dtype=np.int32), dtype=wp.int32, device=device
        )

    positions_wp = to_wp(positions, vec_dtype)
    numbers_wp = to_int(numbers)
    targets_wp = to_int(targets)
    pointer_wp = to_int(pointer)
    shifts_wp = to_wp(shifts_cartesian, vec_dtype)
    rcov_wp = to_wp(rcov, wp_dtype)
    batch_wp = to_int(batch_idx)
    channels_wp = to_int(channels)

    # Pass 1: coordination numbers.
    coordination = wp.zeros(n_atoms, dtype=wp_dtype, device=device)
    if coordination_override is None:
        fd3.fd3_coordination_numbers(
            positions_wp,
            numbers_wp,
            targets_wp,
            pointer_wp,
            shifts_wp,
            rcov_wp,
            r_cut,
            coordination,
            wp_dtype,
            device,
        )
    else:
        coordination = to_wp(coordination_override, wp_dtype)

    # Pass 2: separable coefficients.
    c6 = wp.zeros((n_atoms, rank), dtype=wp_dtype, device=device)
    dc6_dcn = wp.zeros((n_atoms, rank), dtype=wp_dtype, device=device)
    fd3.fd3_coefficients(
        coordination,
        channels_wp,
        to_wp(decomposition.cn_ref, wp_dtype),
        to_wp(decomposition.v_q, wp_dtype),
        c6,
        dc6_dcn,
        wp_dtype,
        device,
    )

    # Pass 3: spread onto the (system, species, rank) mesh.
    group = np.asarray(batch_idx) * n_species + np.asarray(channels)
    cell_inv_t = np.stack([np.linalg.inv(c).T for c in cells])
    cell_inv_grouped = np.repeat(cell_inv_t, n_species, axis=0)
    mesh = wp.zeros((n_systems * n_channels, nx, ny, nz), dtype=wp_dtype, device=device)
    fd3.fd3_spread(
        positions_wp,
        c6,
        to_int(group),
        to_wp(cell_inv_grouped, mat_dtype),
        spline_order,
        rank,
        mesh,
        wp_dtype,
        device,
    )

    # Pass 4: forward transform.
    mesh_np = mesh.numpy().reshape(n_systems * n_channels, nx, ny, nz)
    mesh_fft = np.fft.rfftn(mesh_np, axes=(-3, -2, -1))

    miller_x = np.fft.fftfreq(nx, d=1.0 / nx)
    miller_y = np.fft.fftfreq(ny, d=1.0 / ny)
    miller_z = np.fft.rfftfreq(nz, d=1.0 / nz)
    volumes = np.abs(np.linalg.det(cells))
    k_matrix = np.stack([2.0 * np.pi * np.linalg.inv(c) for c in cells])

    # Pass 5: reciprocal-space contraction.
    fft_pairs = np.stack([mesh_fft.real, mesh_fft.imag], axis=-1)
    energy = wp.zeros(n_systems, dtype=wp_dtype, device=device)
    virial = wp.zeros(n_systems, dtype=mat_dtype, device=device)
    cotangent = wp.zeros(fft_pairs.shape[:-1], dtype=pair_dtype, device=device)
    fd3.fd3_kspace(
        wp.array(
            np.ascontiguousarray(fft_pairs, dtype=np_dtype),
            dtype=pair_dtype,
            device=device,
        ),
        to_wp(k_matrix, mat_dtype),
        to_wp(bspline_moduli(miller_x, nx, spline_order), wp_dtype),
        to_wp(bspline_moduli(miller_y, ny, spline_order), wp_dtype),
        to_wp(bspline_moduli(miller_z, nz, spline_order), wp_dtype),
        to_wp(volumes, wp_dtype),
        to_wp(sqrt_q, wp_dtype),
        to_wp(decomposition.eigs, wp_dtype),
        s6,
        s8,
        a1,
        a2,
        mesh_dimensions,
        n_species,
        rank,
        energy,
        cotangent,
        virial,
        wp_dtype,
        device,
        compute_virial,
    )

    d_energy_d_c6 = wp.zeros((n_atoms, rank), dtype=wp_dtype, device=device)
    forces = wp.zeros(n_atoms, dtype=vec_dtype, device=device)

    if compute_forces:
        # Pass 6: inverse transform. The unnormalised convention is what makes this the
        # adjoint of the forward transform rather than its inverse.
        cotangent_np = cotangent.numpy()
        field = np.fft.irfftn(
            cotangent_np[..., 0] + 1j * cotangent_np[..., 1],
            s=(nx, ny, nz),
            axes=(-3, -2, -1),
            norm="forward",
        )
        # Pass 7: one mesh visit yields both dE/dc6 and the direct force.
        fd3.fd3_gather_and_force(
            to_wp(field, wp_dtype),
            positions_wp,
            c6,
            to_int(group),
            to_wp(cell_inv_grouped, mat_dtype),
            spline_order,
            rank,
            d_energy_d_c6,
            forces,
            wp_dtype,
            device,
        )

    # Pass 8: self-energy, before the chain rule.
    fd3.fd3_self_energy(
        c6,
        channels_wp,
        batch_wp,
        to_wp(sqrt_q, wp_dtype),
        to_wp(decomposition.eigs, wp_dtype),
        s6,
        s8,
        a1,
        a2,
        energy,
        d_energy_d_c6,
        wp_dtype,
        device,
    )

    # Pass 9: chain rule to forces.
    d_energy_d_cn = wp.zeros(n_atoms, dtype=wp_dtype, device=device)
    if compute_forces and apply_cn_chain:
        fd3.fd3_cn_chain(
            d_energy_d_c6,
            dc6_dcn,
            positions_wp,
            numbers_wp,
            targets_wp,
            pointer_wp,
            shifts_wp,
            rcov_wp,
            r_cut,
            batch_wp,
            d_energy_d_cn,
            forces,
            virial,
            wp_dtype,
            device,
            compute_virial,
        )

    return {
        "energy": energy.numpy(),
        "forces": forces.numpy(),
        "virial": virial.numpy(),
        "coordination": coordination.numpy(),
        "c6": c6.numpy(),
        "d_energy_d_c6": d_energy_d_c6.numpy(),
    }

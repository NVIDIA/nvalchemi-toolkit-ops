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

r"""
CSVR Thermostat
===============

Canonical sampling through stochastic velocity rescaling (Bussi-Donadio-Parrinello).
Each call advances the thermostat by ``dt`` and rescales the velocities of each
system by a single factor :math:`\alpha`, drawn so that the kinetic energy
:math:`K` relaxes to :math:`\bar K = N_f k_B T / 2` with time constant :math:`\tau`:

.. math::

    \alpha^2 = c + r\,(R_1^2 + S) + 2 R_1 \sqrt{c\,r},
    \qquad c = e^{-\Delta t/\tau},
    \qquad r = \frac{(1-c)\,\bar K}{N_f K}

where :math:`R_1 \sim \mathcal{N}(0, 1)` and :math:`S \sim \chi^2_{N_f-1}`.

Reference: G. Bussi, D. Donadio, M. Parrinello, J. Chem. Phys. 126, 014101 (2007).
"""

from __future__ import annotations

from typing import Any

import warp as wp

from nvalchemiops.dynamics.utils.launch_helpers import dispatch_family
from nvalchemiops.dynamics.utils.shared_kernels import velocity_rescale_families
from nvalchemiops.dynamics.utils.thermostat_utils import compute_kinetic_energy
from nvalchemiops.warp_dispatch import register_overloads, validate_out_array

__all__ = [
    # Mutating
    "csvr_thermostat",
    # Non-mutating
    "csvr_thermostat_out",
]


@wp.func
def _sample_gamma(state: wp.uint32, shape: wp.float64) -> wp.float64:
    """Draw from Gamma(shape, 1) with Marsaglia-Tsang; returns 0 for ``shape <= 0``.

    Kept in float64 regardless of the array dtype: the acceptance test
    ``0.5 x^2 + d - d v + d log v`` cancels terms of size ``d ~ ndof / 2``, so
    float32 shifts it by ~1e-2 at ``ndof ~ 1e6`` and biases the draw.
    """
    if shape <= wp.float64(0.0):
        return wp.float64(0.0)
    boost = wp.float64(1.0)
    a = shape
    if a < wp.float64(1.0):
        # Gamma(a) = Gamma(a + 1) * U^(1/a)
        boost = wp.pow(wp.float64(wp.randf(state)), wp.float64(1.0) / a)
        a = a + wp.float64(1.0)
    d = a - wp.float64(1.0) / wp.float64(3.0)
    c = wp.float64(1.0) / wp.sqrt(wp.float64(9.0) * d)
    result = wp.float64(-1.0)
    while result < wp.float64(0.0):
        x = wp.float64(wp.randn(state))
        v = wp.float64(1.0) + c * x
        if v > wp.float64(0.0):
            v = v * v * v
            u = wp.float64(wp.randf(state))
            if wp.log(u) < wp.float64(0.5) * x * x + d - d * v + d * wp.log(v):
                result = d * v
    return result * boost


@wp.kernel
def _csvr_scale_kernel(
    ke: wp.array(dtype=Any),
    target_temp: wp.array(dtype=Any),
    tau: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    ndof: wp.array(dtype=Any),
    random_seed: wp.uint64,
    scale: wp.array(dtype=Any),
):
    """Sample the CSVR velocity scale factor of each system.

    Arithmetic is done in float64. Systems with ``ke <= 0`` or ``ndof <= 0``
    get ``scale = 1``.

    Launch Grid: dim = [num_systems]

    Modifies: ``scale``.
    """
    s = wp.tid()
    kin = wp.float64(ke[s])
    n = wp.float64(ndof[s])
    alpha = wp.float64(1.0)
    if kin > wp.float64(0.0) and n > wp.float64(0.0):
        state = wp.rand_init(int(random_seed), s)
        r1 = wp.float64(wp.randn(state))
        chi2 = wp.float64(2.0) * _sample_gamma(
            state, wp.float64(0.5) * (n - wp.float64(1.0))
        )
        c = wp.exp(-wp.float64(dt[s]) / wp.float64(tau[s]))
        # r = (1 - c) * K_target / (ndof * K) with K_target = ndof * kT / 2
        r = (wp.float64(1.0) - c) * wp.float64(target_temp[s]) / (wp.float64(2.0) * kin)
        t = wp.sqrt(c) + r1 * wp.sqrt(r)
        alpha = wp.sign(t) * wp.sqrt(t * t + r * chi2)
    scale[s] = type(scale[s])(alpha)


_csvr_scale_overloads = register_overloads(
    _csvr_scale_kernel,
    lambda t: [
        wp.array(dtype=t),
        wp.array(dtype=t),
        wp.array(dtype=t),
        wp.array(dtype=t),
        wp.array(dtype=t),
        wp.uint64,
        wp.array(dtype=t),
    ],
    dtypes=[wp.float32, wp.float64],
)


def _csvr_update(
    velocities: wp.array,
    velocities_out: wp.array,
    masses: wp.array,
    target_temp: wp.array,
    tau: wp.array,
    dt: wp.array,
    ndof: wp.array,
    random_seed: int,
    ke: wp.array,
    scale: wp.array,
    batch_idx: wp.array | None,
    atom_ptr: wp.array | None,
    device: str | None,
    compute_ke: bool,
) -> None:
    """Compute ``ke`` and ``scale`` from ``velocities``; write the rescaled velocities to ``velocities_out``."""
    if device is None:
        device = velocities.device
    if compute_ke:
        compute_kinetic_energy(
            velocities,
            masses,
            ke,
            batch_idx=batch_idx,
            atom_ptr=atom_ptr,
            device=device,
        )
    wp.launch(
        _csvr_scale_overloads[ke.dtype],
        dim=ke.shape[0],
        inputs=[ke, target_temp, tau, dt, ndof, wp.uint64(random_seed), scale],
        device=device,
    )
    dispatch_family(
        velocity_rescale_families,
        velocities,
        batch_idx=batch_idx,
        atom_ptr=atom_ptr,
        device=device,
        inputs_single=[velocities, scale, velocities_out],
        inputs_batch=[velocities, batch_idx, scale, velocities_out],
        inputs_ptr=[velocities, atom_ptr, scale, velocities_out],
    )


def csvr_thermostat(
    velocities: wp.array,
    masses: wp.array,
    target_temp: wp.array,
    tau: wp.array,
    dt: wp.array,
    ndof: wp.array,
    random_seed: int,
    ke: wp.array,
    scale: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
    compute_ke: bool = True,
) -> None:
    r"""
    Advance the CSVR thermostat by ``dt`` and rescale velocities (in-place).

    .. math::

        \mathbf{v}_i \leftarrow \alpha\,\mathbf{v}_i

    Call once per step with a new ``random_seed``, e.g. before
    ``velocity_verlet_position_update``. Per-system arrays have shape (1,) for
    a single system or (B,) for batched, and share the dtype of ``masses``.

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    target_temp : wp.array
        Target temperature :math:`k_B T`.
    tau : wp.array
        Relaxation time :math:`\tau`. ``0`` resamples the kinetic energy from
        the canonical distribution; ``inf`` disables the thermostat.
    dt : wp.array
        Time step.
    ndof : wp.array
        Degrees of freedom :math:`N_f`.
    random_seed : int
        Seed for this step. The noise depends only on ``(random_seed, system)``.
    ke : wp.array
        Kinetic energy :math:`K` before rescaling. Pre-allocated OUTPUT when
        ``compute_ke`` is True; otherwise a caller-supplied input.
    scale : wp.array
        Pre-allocated OUTPUT scale factor :math:`\alpha`. The energy transferred
        to the bath is ``ke * (1 - scale**2)``.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom.
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (B + 1,).
    device : str, optional
        Warp device. If None, inferred from velocities.
    compute_ke : bool, optional
        If True (default), compute ``ke`` from velocities and masses. If False,
        use the values already in ``ke`` (e.g. a globally reduced kinetic energy).
    """
    _csvr_update(
        velocities,
        velocities,
        masses,
        target_temp,
        tau,
        dt,
        ndof,
        random_seed,
        ke,
        scale,
        batch_idx,
        atom_ptr,
        device,
        compute_ke,
    )


def csvr_thermostat_out(
    velocities: wp.array,
    masses: wp.array,
    target_temp: wp.array,
    tau: wp.array,
    dt: wp.array,
    ndof: wp.array,
    random_seed: int,
    ke: wp.array,
    scale: wp.array,
    velocities_out: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
    compute_ke: bool = True,
) -> wp.array:
    r"""
    Advance the CSVR thermostat by ``dt`` and rescale velocities (non-mutating).

    Same as :func:`csvr_thermostat`, but writes the rescaled velocities into
    ``velocities_out`` and leaves ``velocities`` unchanged. All other
    parameters are as in :func:`csvr_thermostat`.

    Parameters
    ----------
    velocities_out : wp.array
        Pre-allocated OUTPUT array. Must match ``velocities`` in shape,
        dtype, and device.

    Returns
    -------
    wp.array
        Rescaled velocities (same object as ``velocities_out``).
    """
    validate_out_array(velocities_out, velocities, "velocities_out")
    _csvr_update(
        velocities,
        velocities_out,
        masses,
        target_temp,
        tau,
        dt,
        ndof,
        random_seed,
        ke,
        scale,
        batch_idx,
        atom_ptr,
        device,
        compute_ke,
    )
    return velocities_out

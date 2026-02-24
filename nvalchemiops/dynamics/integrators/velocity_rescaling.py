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
Velocity Rescaling Thermostat
=============================

GPU-accelerated Warp kernels for simple velocity rescaling thermostat.

This module provides both mutating (in-place) and non-mutating versions
of each kernel for gradient tracking compatibility.

MATHEMATICAL FORMULATION
========================

Simple velocity rescaling:

.. math::

    \\mathbf{v}_i \\leftarrow \\mathbf{v}_i \\cdot \\sqrt{\\frac{T_{\\text{target}}}{T_{\\text{current}}}}

where the instantaneous temperature is:

.. math::

    T_{\\text{current}} = \\frac{2 \\cdot KE}{N_{\\text{DOF}} \\cdot k_B}
                        = \\frac{\\sum_i m_i |\\mathbf{v}_i|^2}{N_{\\text{DOF}} \\cdot k_B}

USAGE
=====

Velocity rescaling is useful for:
- Quick equilibration to target temperature
- Simple temperature control (non-canonical sampling)
- Initial velocity scaling before production runs

Note: Velocity rescaling does NOT produce canonical (NVT) sampling.
For proper NVT sampling, use Langevin or Nosé-Hoover thermostats.

BATCH MODE
==========

Supports three execution modes for scaling velocities:

**Single System Mode**::

    from nvalchemiops.dynamics.utils import compute_temperature, compute_kinetic_energy

    # Compute current temperature
    ke = wp.empty(1, dtype=wp.float64, device="cuda:0")
    compute_kinetic_energy(velocities, masses, ke)
    T_current = wp.empty(1, dtype=wp.float64, device="cuda:0")
    compute_temperature(ke, T_current, num_atoms=100)

    # Compute scaling factor
    factor = _compute_rescale_factor(T_current, T_target=1.0)
    scale_factor = wp.array([factor], dtype=wp.float64, device="cuda:0")

    # Apply rescaling
    velocity_rescale(velocities, scale_factor)

**Batch Mode with batch_idx**::

    # Different scale factors for each system
    batch_idx = wp.array([0]*N0 + [1]*N1 + [2]*N2, dtype=wp.int32, device="cuda:0")
    scale_factors = wp.array([s0, s1, s2], dtype=wp.float64, device="cuda:0")

    velocity_rescale(velocities, scale_factors, batch_idx=batch_idx)

**Batch Mode with atom_ptr**::

    atom_ptr = wp.array([0, N0, N0+N1, N0+N1+N2], dtype=wp.int32, device="cuda:0")
    scale_factors = wp.array([s0, s1, s2], dtype=wp.float64, device="cuda:0")

    velocity_rescale(velocities, scale_factors, atom_ptr=atom_ptr)

REFERENCES
==========

- Berendsen et al. (1984). J. Chem. Phys. 81, 3684 (weak coupling)
- Bussi et al. (2007). J. Chem. Phys. 126, 014101 (stochastic rescaling)
"""

from __future__ import annotations

from typing import Any

import warp as wp

from nvalchemiops.dynamics.utils.launch_helpers import dispatch_family
from nvalchemiops.dynamics.utils.shared_kernels import velocity_rescale_families
from nvalchemiops.warp_dispatch import validate_out_array

__all__ = [
    # Mutating APIs
    "velocity_rescale",
    # Non-mutating APIs
    "velocity_rescale_out",
    # Utility
    "_compute_rescale_factor",
]


# ==============================================================================
# Functional Interface
# ==============================================================================


@wp.func
def _compute_rescale_factor(
    current_temperature: Any,
    target_temperature: Any,
) -> Any:
    """
    Compute velocity rescaling factor.

    Parameters
    ----------
    current_temperature : float
        Current instantaneous temperature.
    target_temperature : float
        Target temperature.

    Returns
    -------
    float
        Scaling factor sqrt(T_target / T_current).

    """
    if current_temperature <= type(current_temperature)(0.0):
        return type(current_temperature)(1.0)
    return wp.sqrt(target_temperature / current_temperature)


def velocity_rescale(
    velocities: wp.array,
    scale_factor: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> None:
    """
    Rescale velocities to achieve target temperature (in-place).

    Applies ``v_i *= scale_factor`` to all velocities.

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    scale_factor : wp.array
        Scaling factor(s). Shape (1,) for single system, (B,) for batched.
        Typically sqrt(T_target / T_current).
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    device : str, optional
        Warp device. If None, inferred from velocities.

    Example
    -------
    >>> # Compute current temperature
    >>> ke = wp.empty(1, dtype=wp.float64, device=device)
    >>> compute_kinetic_energy(velocities, masses, ke)
    >>> T_current = wp.empty(1, dtype=wp.float64, device=device)
    >>> compute_temperature(ke, T_current, num_atoms=100)
    >>>
    >>> # Compute rescaling factor
    >>> factor = _compute_rescale_factor(T_current, T_target)
    >>> scale = wp.array([factor], dtype=wp.float32, device=device)
    >>>
    >>> # Apply rescaling
    >>> velocity_rescale(velocities, scale)
    """
    dispatch_family(
        velocity_rescale_families,
        velocities,
        batch_idx=batch_idx,
        atom_ptr=atom_ptr,
        device=device,
        inputs_single=[velocities, scale_factor, velocities],
        inputs_batch=[velocities, batch_idx, scale_factor, velocities],
        inputs_ptr=[velocities, atom_ptr, scale_factor, velocities],
    )


def velocity_rescale_out(
    velocities: wp.array,
    scale_factor: wp.array,
    velocities_out: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> wp.array:
    """
    Rescale velocities to achieve target temperature (non-mutating).

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    scale_factor : wp.array
        Scaling factor(s). Shape (1,) for single system, (B,) for batched.
    velocities_out : wp.array
        Pre-allocated output array.  Must match ``velocities`` in shape,
        dtype, and device.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Rescaled velocities (same object as ``velocities_out``).
    """
    validate_out_array(velocities_out, velocities, "velocities_out")
    dispatch_family(
        velocity_rescale_families,
        velocities,
        batch_idx=batch_idx,
        atom_ptr=atom_ptr,
        device=device,
        inputs_single=[velocities, scale_factor, velocities_out],
        inputs_batch=[velocities, batch_idx, scale_factor, velocities_out],
        inputs_ptr=[velocities, atom_ptr, scale_factor, velocities_out],
    )
    return velocities_out

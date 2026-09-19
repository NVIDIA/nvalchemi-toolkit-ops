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
FourierD3: Particle-Mesh Evaluation of the DFT-D3 Dispersion Correction
======================================================================

Periodic DFT-D3(BJ) dispersion in :math:`O(N \log N)` with **no real-space cutoff on the
dispersion sum**; only the short coordination-number list remains.

Mesh summation needs the pairwise coefficient to separate into atom-centred factors, and D3's
:math:`C_6` does not -- it couples both atoms' coordination numbers. A host-side low-rank
decomposition (:mod:`nvalchemiops.interactions.dispersion._c6_decomposition`) restores it:

.. math::

    C_6^{ij} = \sum_{\ell} \lambda_{\ell}\, c_6[i, \ell]\, c_6[j, \ell]

Becke-Johnson damping makes the pair potential absolutely integrable, so its transform is
closed-form with an exponentially decaying envelope. **No Ewald splitting parameter and no
real-space dispersion sum**: the mesh Nyquist frequency is the only truncation, unlike the
electrostatics PME path here.

Pass structure
--------------

Warp has no full-mesh FFT, so the transforms belong to the calling framework:

===== =========================================================== ==============
Pass  Operation                                                   Performed by
===== =========================================================== ==============
1     Coordination numbers from the neighbour list                this module
2     Reference weights, low-rank coefficients, ``dc6/dCN``       this module
3     B-spline spread onto the ``(system, species, rank)`` mesh   ``math.spline``
4     Forward real-to-complex FFT                                 framework
5     Reciprocal-space contraction, energy and virial             this module
6     Inverse FFT of the cotangent field                          framework
7     Gather ``dE/dc6`` and the direct mesh forces                ``math.spline``
8     Self-energy, and its contribution to ``dE/dc6``             this module
9     Contract to ``dE/dCN`` and apply the chain rule to forces   this module
===== =========================================================== ==============

Pass 8 must precede pass 9: the self-energy is quadratic in coefficients that depend on
coordination number, so applying it as a final scalar would drop its force and virial terms.

Units
-----
Unit-agnostic, but ``positions``, ``cell``, ``covalent_radii``, ``r_cut`` and the mesh spacing must
share one system (D3 parameters are conventionally atomic units). ``r_cut`` must equal the
neighbour-list cutoff, because the counting function reaches zero exactly there.

References
----------
Valeeva et al., *A fast summation method for the DFT-D3 dispersion correction*,
arXiv:2607.15103.
"""

from __future__ import annotations

import math
from typing import Any

import warp as wp

from nvalchemiops.math import wp_logistic
from nvalchemiops.math.spline import (
    bspline_grid_offset,
    bspline_weight_3d,
    bspline_weight_gradient_3d,
    compute_fractional_coords,
    wrap_grid_index,
)

__all__ = [
    "fd3_coordination_numbers",
    "fd3_coordination_numbers_matrix",
    "fd3_coefficients",
    "fd3_cn_chain_matrix",
    "fd3_gather_and_force",
    "fd3_spread",
    "fd3_kspace",
    "fd3_self_energy",
    "fd3_cn_chain",
]

_PI = math.pi

# Base steepness of the D3 counting function, retained out to the transition radius.
_CN_STEEPNESS = 16.0

# The shipped covalent_radii table already folds in Grimme's 4/3, which the counting ratio is defined
# on. The transition radius uses the bare covalent radius, so it divides that back out.
_CN_UNSCALE = 3.0 / 4.0

# Keeps the steepness finite exactly at the cutoff, where the gap term vanishes.
_CN_EPSILON = 1.0e-6

# Gaussian width of the reference weighting, L = exp(-_CN_GAUSSIAN * (cn - cn_ref)^2).
_CN_GAUSSIAN = 4.0

FD3_CN_BLOCK_SIZE = 32
"""Threads per block in the coordination-number force pass.

One warp per atom, with the warp's threads striding over that atom's neighbours. An atom
carries on the order of a hundred neighbours, so a thread per atom leaves the machine idle
and reads the neighbour list in a scattered order; a warp per atom fixes both. A warp is
also the largest block whose reduction is pure shuffles, needing no shared memory, and
measured faster here than 64 or 128.
"""

FD3_KSPACE_BLOCK_SIZE = 256
"""Threads per block in the reciprocal-space pass.

Every thread in a system contributes to the same energy and virial accumulators, so the
pass reduces within a block and issues one atomic per block instead of one per bin. The
launch pads each system's bin count up to a multiple of this so that no block straddles
two systems.
"""

# Below this dimensionless argument the closed-form transforms lose precision to
# cancellation and the series is used instead. The two agree to ~6e-14 at the crossover.
_SERIES_CUTOFF = 0.1

# Taylor coefficients of N(x)/x. N(x) is entire, so these series are exact rather than
# asymptotic. Seven terms hold the crossover error near 1e-14.
_R6_C0, _R6_C2, _R6_C3, _R6_C4, _R6_C6 = (
    1.0,
    -1.0 / 3.0,
    0.125,
    -1.0 / 60.0,
    1.0 / 5040.0,
)
_R8_C0 = -0.54119610014619668
_R8_C2 = 0.090199350024366076
_R8_C4 = -0.010888024707303151
_R8_C5 = 1.0 / 360.0
_R8_C6 = -0.00025923868350721731

# Coefficients of the term-by-term derivative of the series above, named rather than formed
# inline because Warp resolves a scalar cast only for a bare constant.
_R6_D2, _R6_D3, _R6_D4, _R6_D6 = 2.0 * _R6_C2, 3.0 * _R6_C3, 4.0 * _R6_C4, 6.0 * _R6_C6
_R8_D2, _R8_D4, _R8_D5, _R8_D6 = 2.0 * _R8_C2, 4.0 * _R8_C4, 5.0 * _R8_C5, 6.0 * _R8_C6

_SQRT3 = math.sqrt(3.0)
_SIN_PI_8 = math.sin(_PI / 8.0)
_COS_PI_8 = math.cos(_PI / 8.0)
_PI_OVER_3 = _PI / 3.0
_PI_OVER_4 = _PI / 4.0
_THREE_PI_OVER_4 = 3.0 * _PI / 4.0
_TWO_PI_SQ_OVER_3 = 2.0 * _PI * _PI / 3.0
_PI_SQ = _PI * _PI


@wp.func
def _cn_counting(
    distance: Any,
    covalent_distance: Any,
    r_cut: Any,
    compute_derivative: bool,
) -> tuple[Any, Any]:
    """Contribution of one neighbour to a coordination number, and its radial derivative.

    FourierD3 replaces the fixed-steepness D3 counting function with one whose steepness grows
    without bound as the separation approaches ``r_cut``. The standard form tends to a non-zero
    constant at large separation, so truncating it leaves a step and the energy never converges
    as the list grows. This form reaches exactly zero at the cutoff, which is what makes the
    coordination numbers independent of the list used to build them.

    Parameters
    ----------
    distance : Any
        Interatomic separation.
    covalent_distance : Any
        Sum of the two covalent radii, on the same scale ``dftd3`` uses: the shipped table
        already folds in Grimme's 4/3 factor, so the counting function crosses one half at
        ``distance == covalent_distance``. See ``_CN_UNSCALE`` for where that factor is
        divided back out.
    r_cut : Any
        Neighbour-list cutoff. The counting function reaches zero here, so this must be the
        radius the list was actually built with.
    compute_derivative : bool
        When false the derivative is returned as zero and its evaluation is skipped.

    Returns
    -------
    value : Any
        Counting-function value in ``[0, 1]``.
    derivative : Any
        ``d(value)/d(distance)``, or zero when not requested.
    """
    one = type(distance)(1.0)
    transition = type(distance)(0.5) * (
        type(distance)(_CN_UNSCALE) * covalent_distance + r_cut
    )
    stabilised = wp.max(distance, transition)
    gap = r_cut - stabilised
    denominator = gap * gap + type(distance)(_CN_EPSILON)
    offset = stabilised - transition
    steepness = type(distance)(_CN_STEEPNESS) + offset * offset / denominator

    ratio = covalent_distance / distance
    argument = steepness * (ratio - one)
    value = wp_logistic(argument)

    if not compute_derivative:
        return value, type(distance)(0.0)

    # d(value)/dr = value * (1 - value) * d(argument)/dr. The steepness is constant inside the
    # transition radius, so it only contributes beyond it.
    d_ratio = -covalent_distance / (distance * distance)
    d_steepness = type(distance)(0.0)
    if distance > transition:
        d_steepness = type(distance)(2.0) * offset / denominator + type(distance)(
            2.0
        ) * offset * offset * gap / (denominator * denominator)
    d_argument = d_steepness * (ratio - one) + steepness * d_ratio
    return value, value * (one - value) * d_argument


@wp.func
def _shape_r6(x: Any) -> tuple[Any, Any]:
    """``N6(x)/x`` and its derivative, on whichever branch is stable at ``x``.

    ``N6`` vanishes at the origin, so the closed form divides two cancelling quantities and
    loses all precision for small ``x``. The series is the same entire function written so
    that it does not cancel.
    """
    if x < type(x)(_SERIES_CUTOFF):
        value = type(x)(_R6_C0) + x * x * (
            type(x)(_R6_C2)
            + x * (type(x)(_R6_C3) + x * (type(x)(_R6_C4) + x * x * type(x)(_R6_C6)))
        )
        derivative = x * (
            type(x)(_R6_D2)
            + x * (type(x)(_R6_D3) + x * (type(x)(_R6_D4) + x * x * type(x)(_R6_D6)))
        )
        return value, derivative

    root3 = type(x)(_SQRT3)
    phase = type(x)(_PI_OVER_3) + x * root3 * type(x)(0.5)
    decay_full = wp.exp(-x)
    decay_half = wp.exp(-x * type(x)(0.5))
    numerator = decay_full - type(x)(2.0) * decay_half * wp.cos(phase)
    d_numerator = (
        -decay_full + decay_half * wp.cos(phase) + root3 * decay_half * wp.sin(phase)
    )
    return numerator / x, d_numerator / x - numerator / (x * x)


@wp.func
def _shape_r8(x: Any) -> tuple[Any, Any]:
    """``N8(x)/x`` and its derivative, on whichever branch is stable at ``x``."""
    if x < type(x)(_SERIES_CUTOFF):
        value = type(x)(_R8_C0) + x * x * (
            type(x)(_R8_C2)
            + x * x * (type(x)(_R8_C4) + x * (type(x)(_R8_C5) + x * type(x)(_R8_C6)))
        )
        derivative = x * (
            type(x)(_R8_D2)
            + x * x * (type(x)(_R8_D4) + x * (type(x)(_R8_D5) + x * type(x)(_R8_D6)))
        )
        return value, derivative

    sin8 = type(x)(_SIN_PI_8)
    cos8 = type(x)(_COS_PI_8)
    phase_a = type(x)(_PI_OVER_4) + x * cos8
    phase_b = type(x)(_THREE_PI_OVER_4) + x * sin8
    decay_a = wp.exp(-x * sin8)
    decay_b = wp.exp(-x * cos8)
    numerator = decay_a * wp.cos(phase_a) + decay_b * wp.cos(phase_b)
    d_numerator = (
        -sin8 * decay_a * wp.cos(phase_a)
        - cos8 * decay_a * wp.sin(phase_a)
        - cos8 * decay_b * wp.cos(phase_b)
        - sin8 * decay_b * wp.sin(phase_b)
    )
    return numerator / x, d_numerator / x - numerator / (x * x)


@wp.func
def _reciprocal_kernel(
    k_norm: Any,
    r0: Any,
    sqrt_q_product: Any,
    s6: Any,
    s8: Any,
) -> tuple[Any, Any]:
    r"""Fourier transform of the damped dispersion potential, and its radial derivative.

    Returns :math:`K_{AB}(|k|)` and :math:`dK_{AB}/d|k|`. The derivative supplies the
    reciprocal-space contribution to the virial, which cannot be recovered once ``k`` has left
    scope.

    The transform is an exponentially damped *oscillation*: it changes sign, with the first
    zero near :math:`|k| R_0 \approx 4.3`. Only the envelope decays monotonically.
    """
    x = k_norm * r0
    g6, dg6 = _shape_r6(x)
    g8, dg8 = _shape_r8(x)

    scale6 = type(k_norm)(_TWO_PI_SQ_OVER_3) / (r0 * r0 * r0)
    scale8 = -type(k_norm)(_PI_SQ) / (r0 * r0 * r0 * r0 * r0)
    weight8 = type(k_norm)(3.0) * s8 * sqrt_q_product

    value = s6 * scale6 * g6 + weight8 * scale8 * g8
    # d/dk = R0 * d/dx, since x = k * R0.
    derivative = r0 * (s6 * scale6 * dg6 + weight8 * scale8 * dg8)
    return value, derivative


@wp.kernel(enable_backward=False)
def _fd3_cn_kernel(
    positions: wp.array(dtype=Any),
    numbers: wp.array(dtype=wp.int32),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    cartesian_shifts: wp.array(dtype=Any),
    covalent_radii: wp.array(dtype=Any),
    r_cut: Any,
    block_stride: wp.int32,
    coord_num: wp.array(dtype=Any),
):
    """Accumulate coordination numbers from a directed neighbour list.

    Thread launch
    -------------
    ``dim = (num_atoms, FD3_CN_BLOCK_SIZE)`` with ``block_dim = FD3_CN_BLOCK_SIZE`` -- one
    block per atom, its threads striding ``block_stride`` apart through that atom's slice of
    the neighbour list and reducing across the block at the end.

    Modifies
    --------
    ``coord_num[i]`` is overwritten with the coordination number of atom ``i``. The neighbour
    list must contain both orientations of every pair for the result to be symmetric.
    """
    atom_i, thread_in_block = wp.tid()
    if numbers[atom_i] == 0:
        return

    zero = type(r_cut)(0.0)
    total = zero
    rcov_i = covalent_radii[numbers[atom_i]]
    position_i = positions[atom_i]

    edge = neighbor_ptr[atom_i] + thread_in_block
    last_edge = neighbor_ptr[atom_i + 1]
    while edge < last_edge:
        atom_j = idx_j[edge]
        if numbers[atom_j] != 0:
            delta = positions[atom_j] + cartesian_shifts[edge] - position_i
            distance = wp.length(delta)
            if distance < r_cut and distance != zero:
                value, _unused = _cn_counting(
                    distance, rcov_i + covalent_radii[numbers[atom_j]], r_cut, False
                )
                total += value
        edge += block_stride

    block_total = wp.tile_sum(wp.tile(total))[0]
    if thread_in_block == 0:
        coord_num[atom_i] = block_total


@wp.kernel(enable_backward=False)
def _fd3_coefficients_kernel(
    coord_num: wp.array(dtype=Any),
    species_index: wp.array(dtype=wp.int32),
    cnref: wp.array2d(dtype=Any),
    v_q: wp.array3d(dtype=Any),
    c6: wp.array2d(dtype=Any),
    dc6_dcn: wp.array2d(dtype=Any),
):
    """Turn coordination numbers into separable per-atom coefficients.

    The D3 reference weighting is a Gaussian in the distance from each reference coordination
    number, normalised across references. Because the weighting factorises over the two atoms
    of a pair, applying it per atom is exact: the only approximation in the separable form is
    the rank truncation already made when the reference tensor was decomposed.

    Thread launch
    -------------
    One thread per atom.

    Modifies
    --------
    ``c6[i, :]`` and ``dc6_dcn[i, :]``. Reference slots the species does not use are marked
    by a negative entry in ``cnref`` and skipped.
    """
    atom_i = wp.tid()
    channel = species_index[atom_i]
    n_ref = cnref.shape[1]
    rank = v_q.shape[2]

    if channel < 0:
        for slot in range(rank):
            c6[atom_i, slot] = type(coord_num[0])(0.0)
            dc6_dcn[atom_i, slot] = type(coord_num[0])(0.0)
        return

    cn = coord_num[atom_i]
    gaussian = type(cn)(_CN_GAUSSIAN)

    # Largest exponent first, so the weights below cannot all underflow to zero when the
    # coordination number sits far from every reference.
    peak = type(cn)(-1.0e30)
    for p in range(n_ref):
        if cnref[channel, p] >= type(cn)(0.0):
            delta = cn - cnref[channel, p]
            exponent = -gaussian * delta * delta
            if exponent > peak:
                peak = exponent

    weight_sum = type(cn)(0.0)
    weight_derivative_sum = type(cn)(0.0)
    for p in range(n_ref):
        if cnref[channel, p] >= type(cn)(0.0):
            delta = cn - cnref[channel, p]
            weight = wp.exp(-gaussian * delta * delta - peak)
            weight_sum += weight
            weight_derivative_sum += -type(cn)(2.0) * gaussian * delta * weight

    inverse = type(cn)(1.0) / weight_sum
    for slot in range(rank):
        numerator = type(cn)(0.0)
        numerator_derivative = type(cn)(0.0)
        for p in range(n_ref):
            if cnref[channel, p] >= type(cn)(0.0):
                delta = cn - cnref[channel, p]
                weight = wp.exp(-gaussian * delta * delta - peak)
                factor = v_q[channel, p, slot]
                numerator += weight * factor
                numerator_derivative += (
                    -type(cn)(2.0) * gaussian * delta * weight * factor
                )
        value = numerator * inverse
        c6[atom_i, slot] = value
        dc6_dcn[atom_i, slot] = (
            numerator_derivative - value * weight_derivative_sum
        ) * inverse


_SCALARS = [wp.float32, wp.float64]
_VECTORS = [wp.vec3f, wp.vec3d]

_fd3_cn_kernel_overload = {}
_fd3_coefficients_kernel_overload = {}

for _scalar, _vector in zip(_SCALARS, _VECTORS):
    _fd3_cn_kernel_overload[_scalar] = wp.overload(
        _fd3_cn_kernel,
        [
            wp.array(dtype=_vector),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=_vector),
            wp.array(dtype=_scalar),
            _scalar,
            wp.int32,
            wp.array(dtype=_scalar),
        ],
    )
    _fd3_coefficients_kernel_overload[_scalar] = wp.overload(
        _fd3_coefficients_kernel,
        [
            wp.array(dtype=_scalar),
            wp.array(dtype=wp.int32),
            wp.array2d(dtype=_scalar),
            wp.array3d(dtype=_scalar),
            wp.array2d(dtype=_scalar),
            wp.array2d(dtype=_scalar),
        ],
    )


def fd3_coordination_numbers(
    positions: wp.array,
    numbers: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    cartesian_shifts: wp.array,
    covalent_radii: wp.array,
    r_cut: float,
    coord_num: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Compute coordination numbers with the FourierD3 counting function.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    numbers : wp.array, shape (N,), dtype=wp.int32
        Atomic numbers. Zero marks a padding atom, which is skipped.
    idx_j : wp.array, shape (E,), dtype=wp.int32
        Neighbour index for each directed edge.
    neighbor_ptr : wp.array, shape (N + 1,), dtype=wp.int32
        Start of each atom's edge slice.
    cartesian_shifts : wp.array, shape (E,), dtype=wp.vec3f or wp.vec3d
        Periodic image offset for each edge, already in Cartesian units.
    covalent_radii : wp.array, shape (max_z + 1,)
        Covalent radii indexed by atomic number, in the same length unit as ``positions``.
    r_cut : float
        Neighbour-list cutoff. Must equal the radius the list was built with: the counting
        function is constructed to reach zero exactly here, and a mismatch reintroduces the
        truncation discontinuity the modified form exists to remove.
    coord_num : wp.array, shape (N,)
        OUTPUT: coordination number per atom.
    wp_dtype : type
        Warp scalar dtype (``wp.float32`` or ``wp.float64``).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    if num_atoms == 0:
        return
    # Tile reductions need a real block launch, which only CUDA provides; on CPU a block is
    # one thread, which walks the whole neighbour slice itself.
    block = FD3_CN_BLOCK_SIZE if "cuda" in str(device) else 1
    wp.launch(
        _fd3_cn_kernel_overload[wp_dtype],
        dim=(num_atoms, block),
        inputs=[
            positions,
            numbers,
            idx_j,
            neighbor_ptr,
            cartesian_shifts,
            covalent_radii,
            wp_dtype(r_cut),
            wp.int32(block),
        ],
        outputs=[coord_num],
        block_dim=block,
        device=device,
    )


def fd3_coefficients(
    coord_num: wp.array,
    species_index: wp.array,
    cnref: wp.array,
    v_q: wp.array,
    c6: wp.array,
    dc6_dcn: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Evaluate the separable per-atom coefficients and their coordination derivative.

    Parameters
    ----------
    coord_num : wp.array, shape (N,)
        Coordination number per atom.
    species_index : wp.array, shape (N,), dtype=wp.int32
        Channel index per atom, from ``C6Decomposition.species_map``. Negative marks an atom
        whose species is not covered; its coefficients are set to zero.
    cnref : wp.array2d, shape (n_species, n_ref)
        Reference coordination numbers. Negative entries mark unused reference slots.
    v_q : wp.array3d, shape (n_species, n_ref, rank)
        Eigenvectors of the decomposed reference tensor.
    c6 : wp.array2d, shape (N, rank)
        OUTPUT: separable coefficients.
    dc6_dcn : wp.array2d, shape (N, rank)
        OUTPUT: their derivative with respect to the coordination number.
    wp_dtype : type
        Warp scalar dtype (``wp.float32`` or ``wp.float64``).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = coord_num.shape[0]
    if num_atoms == 0:
        return
    wp.launch(
        _fd3_coefficients_kernel_overload[wp_dtype],
        dim=num_atoms,
        inputs=[coord_num, species_index, cnref, v_q],
        outputs=[c6, dc6_dcn],
        device=device,
    )


@wp.func
def _hermitian_weight(index: int, size: int, unit: Any) -> Any:
    """Multiplicity of a real-to-complex transform bin in a full-spectrum sum.

    Only half the spectrum is stored, so every bin except those that are their own conjugate
    stands in for a second one. Applying this to a field that is about to be
    inverse-transformed would double-count instead: the inverse transform reconstructs the
    missing half itself.
    """
    if index == 0:
        return unit
    if size % 2 == 0 and index * 2 == size:
        return unit
    return unit + unit


@wp.kernel(enable_backward=False)
def _fd3_kspace_kernel(
    mesh_fft: wp.array4d(dtype=Any),
    k_matrix: wp.array(dtype=Any),
    moduli_x: wp.array(dtype=Any),
    moduli_y: wp.array(dtype=Any),
    moduli_z: wp.array(dtype=Any),
    volumes: wp.array(dtype=Any),
    sqrt_q: wp.array(dtype=Any),
    eigs: wp.array(dtype=Any),
    s6: Any,
    s8: Any,
    a1: Any,
    a2: Any,
    mesh_nx: wp.int32,
    mesh_ny: wp.int32,
    mesh_nz: wp.int32,
    num_bins: wp.int32,
    block_size: wp.int32,
    n_species: wp.int32,
    rank: wp.int32,
    compute_virial: bool,
    energy: wp.array(dtype=Any),
    cotangent: wp.array4d(dtype=Any),
    virial: wp.array(dtype=Any),
):
    """Contract the transformed mesh against the dispersion kernel.

    Computes the reciprocal-space energy and the cotangent field whose inverse transform is
    the derivative of that energy with respect to the mesh.

    Thread launch
    -------------
    ``dim = (num_systems, padded_bins)`` with ``block_dim = FD3_KSPACE_BLOCK_SIZE`` -- one
    thread per stored transform bin, flattened, and padded so that a block never spans two
    systems. Threads past ``num_bins`` contribute nothing.

    Modifies
    --------
    ``energy[b]`` and, when requested, ``virial[b]`` are accumulated once per block.
    ``cotangent`` is overwritten at this bin for every channel. The multiplicity weight
    applies to the reductions only; the cotangent is left unweighted because the inverse
    transform restores the unstored half itself.

    Notes
    -----
    The reciprocal part of the virial exists only here. Under a strain the fractional
    coordinates, and so the mesh, do not move; what does move is the reciprocal lattice and
    the cell volume. Once the wave vector has gone out of scope there is nothing left to
    differentiate, so a later pass cannot recover this term.
    """
    system, flat = wp.tid()

    unit = type(s6)(1.0)
    zero = type(s6)(0.0)

    # Padding threads still run the block reduction below, so they read a valid bin and
    # discard the result rather than returning early.
    active = flat < num_bins
    sample = wp.where(active, flat, 0)
    nz_half = mesh_nz // 2 + 1
    ix = sample / (mesh_ny * nz_half)
    remainder = sample % (mesh_ny * nz_half)
    iy = remainder / nz_half
    iz = remainder % nz_half

    # Bins past the midpoint carry negative frequencies. The last axis is halved by the
    # real transform, so it has no negative half to fold.
    miller = wp.vector(
        type(s6)(wp.where(ix * 2 > mesh_nx, ix - mesh_nx, ix)),
        type(s6)(wp.where(iy * 2 > mesh_ny, iy - mesh_ny, iy)),
        type(s6)(iz),
    )
    k_vector = k_matrix[system] * miller
    k_norm = wp.length(k_vector)

    # B-spline interpolation attenuates each bin; dividing it out recovers the structure
    # factor the mesh is standing in for. The floor guards bins the spline cannot represent.
    modulus = moduli_x[ix] * moduli_y[iy] * moduli_z[iz]
    if modulus < type(s6)(1.0e-10):
        modulus = type(s6)(1.0e-10)

    weight = _hermitian_weight(iz, mesh_nz, unit)
    prefactor = -unit / (type(s6)(2.0) * volumes[system])
    accumulated = zero
    accumulated_slope = zero

    inverse_modulus = unit / modulus
    # dE/d(mesh) follows from the inverse transform of this field, so it carries the
    # prefactor and the remaining spline factor but not the multiplicity.
    cotangent_scale = type(s6)(2.0) * prefactor * inverse_modulus
    first_channel = (system * n_species) * rank

    # Each pair feeds two channels, so this accumulates and must start at zero. One thread
    # per bin makes these private locations -- no atomics needed.
    if active:
        for channel in range(n_species * rank):
            cotangent[first_channel + channel, ix, iy, iz] = wp.vector(zero, zero)

    # The kernel depends on (k, pair) but not slot, and is pair-symmetric. Hoisting both
    # loops and going upper-triangular cuts evaluations from rank*n^2 to n*(n+1)/2 per bin.
    for species_a in range(n_species):
        offset_a = first_channel + species_a * rank
        for species_b in range(species_a, n_species):
            offset_b = first_channel + species_b * rank
            q_product = sqrt_q[species_a] * sqrt_q[species_b]
            r0 = a1 * wp.sqrt(type(s6)(3.0) * q_product) + a2
            value, slope = _reciprocal_kernel(k_norm, r0, q_product, s6, s8)
            # An off-diagonal pair stands in for both (a, b) and (b, a) in the reductions.
            multiplicity = wp.where(species_a == species_b, unit, type(s6)(2.0))

            for slot in range(rank):
                amplitude_a = mesh_fft[offset_a + slot, ix, iy, iz] * inverse_modulus
                amplitude_b = mesh_fft[offset_b + slot, ix, iy, iz] * inverse_modulus
                overlap = (
                    amplitude_a[0] * amplitude_b[0] + amplitude_a[1] * amplitude_b[1]
                )
                gain = eigs[slot] * value
                accumulated += weight * multiplicity * gain * overlap
                if compute_virial:
                    accumulated_slope += (
                        weight * multiplicity * eigs[slot] * slope * overlap
                    )
                if active:
                    cotangent[offset_a + slot, ix, iy, iz] = (
                        cotangent[offset_a + slot, ix, iy, iz]
                        + (cotangent_scale * gain) * amplitude_b
                    )
                    if species_a != species_b:
                        cotangent[offset_b + slot, ix, iy, iz] = (
                            cotangent[offset_b + slot, ix, iy, iz]
                            + (cotangent_scale * gain) * amplitude_a
                        )

    contribution = wp.where(active, prefactor * accumulated, zero)
    # The launch block size, not the constant: on CPU a block is one thread, and
    # reducing against the wrong width would silence every bin but the first of each.
    thread_in_block = flat % block_size

    block_energy = wp.tile_sum(wp.tile(contribution))[0]
    if thread_in_block == 0:
        wp.atomic_add(energy, system, block_energy)

    if compute_virial:
        # Strain scales the volume and shrinks the reciprocal lattice: d(1/V) gives the
        # isotropic term, d|k|/d(strain) = -k_a k_b / |k| the anisotropic one.
        radial = zero
        if k_norm > type(s6)(1.0e-12):
            radial = wp.where(active, prefactor * accumulated_slope / k_norm, zero)
        # Only the six independent components are reduced; both terms are symmetric.
        sum_xx = wp.tile_sum(
            wp.tile(-contribution - radial * k_vector[0] * k_vector[0])
        )[0]
        sum_yy = wp.tile_sum(
            wp.tile(-contribution - radial * k_vector[1] * k_vector[1])
        )[0]
        sum_zz = wp.tile_sum(
            wp.tile(-contribution - radial * k_vector[2] * k_vector[2])
        )[0]
        sum_xy = wp.tile_sum(wp.tile(-radial * k_vector[0] * k_vector[1]))[0]
        sum_xz = wp.tile_sum(wp.tile(-radial * k_vector[0] * k_vector[2]))[0]
        sum_yz = wp.tile_sum(wp.tile(-radial * k_vector[1] * k_vector[2]))[0]
        if thread_in_block == 0:
            # Negate: conventions.md (and dftd3) define the virial as -dE/du.
            wp.atomic_add(
                virial,
                system,
                type(s6)(-1.0)
                * wp.matrix_from_rows(
                    wp.vector(sum_xx, sum_xy, sum_xz),
                    wp.vector(sum_xy, sum_yy, sum_yz),
                    wp.vector(sum_xz, sum_yz, sum_zz),
                ),
            )


@wp.kernel(enable_backward=False)
def _fd3_self_energy_kernel(
    c6: wp.array2d(dtype=Any),
    species_index: wp.array(dtype=wp.int32),
    batch_idx: wp.array(dtype=wp.int32),
    sqrt_q: wp.array(dtype=Any),
    eigs: wp.array(dtype=Any),
    s6: Any,
    s8: Any,
    a1: Any,
    a2: Any,
    energy: wp.array(dtype=Any),
    d_energy_d_c6: wp.array2d(dtype=Any),
):
    """Remove the self pair the reciprocal sum unavoidably includes.

    The reciprocal-space sum runs over every ordered pair, the ``i == j`` term at zero
    separation among them, which no physical pair sum contains. Becke-Johnson damping leaves
    that term finite, so it is cancelled here.

    Because the term is quadratic in the coefficients, and the coefficients are functions of
    coordination number, it also contributes to ``dE/dc6`` and so to the forces. Applying it
    as a scalar correction after the chain rule would silently drop that contribution.

    Thread launch
    -------------
    One thread per atom.

    Modifies
    --------
    ``energy[batch_idx[i]]`` is accumulated atomically, and the atom's row of
    ``d_energy_d_c6`` is accumulated in place.
    """
    atom_i = wp.tid()
    channel = species_index[atom_i]
    if channel < 0:
        return

    rank = c6.shape[1]
    q_self = sqrt_q[channel] * sqrt_q[channel]
    r0 = a1 * wp.sqrt(type(s6)(3.0) * q_self) + a2

    # Zero-separation limit of the damped potential.
    r0_sq = r0 * r0
    r0_6 = r0_sq * r0_sq * r0_sq
    r0_8 = r0_6 * r0_sq
    potential = s6 / r0_6 + type(s6)(3.0) * s8 * q_self / r0_8

    total = type(s6)(0.0)
    for slot in range(rank):
        coefficient = c6[atom_i, slot]
        total += eigs[slot] * coefficient * coefficient * potential
        d_energy_d_c6[atom_i, slot] += eigs[slot] * coefficient * potential

    wp.atomic_add(energy, batch_idx[atom_i], type(s6)(0.5) * total)


@wp.kernel(enable_backward=False)
def _fd3_cn_sensitivity_kernel(
    d_energy_d_c6: wp.array2d(dtype=Any),
    dc6_dcn: wp.array2d(dtype=Any),
    d_energy_d_cn: wp.array(dtype=Any),
):
    """Contract the coefficient derivative down to a per-atom coordination sensitivity.

    Every term that depends on the coefficients must already have been folded into
    ``d_energy_d_c6`` before this runs, the self-energy included.

    Thread launch
    -------------
    One thread per atom.

    Modifies
    --------
    ``d_energy_d_cn[i]`` is overwritten.
    """
    atom_i = wp.tid()
    rank = d_energy_d_c6.shape[1]
    total = d_energy_d_c6[atom_i, 0] * dc6_dcn[atom_i, 0]
    for slot in range(1, rank):
        total += d_energy_d_c6[atom_i, slot] * dc6_dcn[atom_i, slot]
    d_energy_d_cn[atom_i] = total


@wp.kernel(enable_backward=False)
def _fd3_cn_forces_kernel(
    d_energy_d_cn: wp.array(dtype=Any),
    positions: wp.array(dtype=Any),
    numbers: wp.array(dtype=wp.int32),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    cartesian_shifts: wp.array(dtype=Any),
    covalent_radii: wp.array(dtype=Any),
    r_cut: Any,
    batch_idx: wp.array(dtype=wp.int32),
    block_stride: wp.int32,
    compute_virial: bool,
    forces: wp.array(dtype=Any),
    virial: wp.array(dtype=Any),
):
    """Apply the coordination-number chain rule to forces and the virial.

    A separation changes the coordination number of both atoms it connects, so each edge
    carries the sum of the two sensitivities. The reverse edge is walked by the thread that
    owns the other atom, which is what makes the resulting forces sum to zero.

    Thread launch
    -------------
    ``dim = (num_atoms, FD3_CN_BLOCK_SIZE)`` with ``block_dim = FD3_CN_BLOCK_SIZE`` -- one
    block per atom, its threads striding ``block_stride`` apart through that atom's slice of
    the neighbour list and reducing across the block at the end.

    Modifies
    --------
    ``forces[i]`` for the atom the block owns, and ``virial`` per system when requested.
    """
    atom_i, thread_in_block = wp.tid()
    if numbers[atom_i] == 0:
        return

    rcov_i = covalent_radii[numbers[atom_i]]
    position_i = positions[atom_i]
    sensitivity_i = d_energy_d_cn[atom_i]

    zero = type(r_cut)(0.0)
    half = type(r_cut)(0.5)
    force_x = zero
    force_y = zero
    force_z = zero
    virial_xx = zero
    virial_yy = zero
    virial_zz = zero
    virial_xy = zero
    virial_xz = zero
    virial_yz = zero

    edge = neighbor_ptr[atom_i] + thread_in_block
    last_edge = neighbor_ptr[atom_i + 1]
    while edge < last_edge:
        atom_j = idx_j[edge]
        if numbers[atom_j] != 0:
            delta = positions[atom_j] + cartesian_shifts[edge] - position_i
            distance = wp.length(delta)
            if distance < r_cut and distance != zero:
                _value, slope = _cn_counting(
                    distance, rcov_i + covalent_radii[numbers[atom_j]], r_cut, True
                )
                magnitude = (sensitivity_i + d_energy_d_cn[atom_j]) * slope
                pair_force = (magnitude / distance) * delta
                force_x += pair_force[0]
                force_y += pair_force[1]
                force_z += pair_force[2]

                if compute_virial:
                    # F outer delta, symmetric because the force lies along the separation.
                    # Halved because the reverse edge contributes the same again.
                    virial_xx += half * pair_force[0] * delta[0]
                    virial_yy += half * pair_force[1] * delta[1]
                    virial_zz += half * pair_force[2] * delta[2]
                    virial_xy += half * pair_force[0] * delta[1]
                    virial_xz += half * pair_force[0] * delta[2]
                    virial_yz += half * pair_force[1] * delta[2]
        edge += block_stride

    sum_fx = wp.tile_sum(wp.tile(force_x))[0]
    sum_fy = wp.tile_sum(wp.tile(force_y))[0]
    sum_fz = wp.tile_sum(wp.tile(force_z))[0]
    if thread_in_block == 0:
        # The gather pass has already written this atom's mesh force, so this accumulates.
        wp.atomic_add(forces, atom_i, wp.vector(sum_fx, sum_fy, sum_fz))

    if compute_virial:
        sum_xx = wp.tile_sum(wp.tile(virial_xx))[0]
        sum_yy = wp.tile_sum(wp.tile(virial_yy))[0]
        sum_zz = wp.tile_sum(wp.tile(virial_zz))[0]
        sum_xy = wp.tile_sum(wp.tile(virial_xy))[0]
        sum_xz = wp.tile_sum(wp.tile(virial_xz))[0]
        sum_yz = wp.tile_sum(wp.tile(virial_yz))[0]
        if thread_in_block == 0:
            # Negate: conventions.md (and dftd3) define the virial as -dE/du.
            wp.atomic_add(
                virial,
                batch_idx[atom_i],
                type(r_cut)(-1.0)
                * wp.matrix_from_rows(
                    wp.vector(sum_xx, sum_xy, sum_xz),
                    wp.vector(sum_xy, sum_yy, sum_yz),
                    wp.vector(sum_xz, sum_yz, sum_zz),
                ),
            )


_MATRICES = [wp.mat33f, wp.mat33d]
_PAIRS = [wp.vec2f, wp.vec2d]

_fd3_kspace_kernel_overload = {}
_fd3_self_energy_kernel_overload = {}
_fd3_cn_sensitivity_kernel_overload = {}
_fd3_cn_forces_kernel_overload = {}

for _scalar, _vector, _matrix, _pair in zip(_SCALARS, _VECTORS, _MATRICES, _PAIRS):
    _fd3_kspace_kernel_overload[_scalar] = wp.overload(
        _fd3_kspace_kernel,
        [
            wp.array4d(dtype=_pair),
            wp.array(dtype=_matrix),
            wp.array(dtype=_scalar),
            wp.array(dtype=_scalar),
            wp.array(dtype=_scalar),
            wp.array(dtype=_scalar),
            wp.array(dtype=_scalar),
            wp.array(dtype=_scalar),
            _scalar,
            _scalar,
            _scalar,
            _scalar,
            wp.int32,
            wp.int32,
            wp.int32,
            wp.int32,
            wp.int32,
            wp.int32,
            wp.int32,
            wp.bool,
            wp.array(dtype=_scalar),
            wp.array4d(dtype=_pair),
            wp.array(dtype=_matrix),
        ],
    )
    _fd3_self_energy_kernel_overload[_scalar] = wp.overload(
        _fd3_self_energy_kernel,
        [
            wp.array2d(dtype=_scalar),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=_scalar),
            wp.array(dtype=_scalar),
            _scalar,
            _scalar,
            _scalar,
            _scalar,
            wp.array(dtype=_scalar),
            wp.array2d(dtype=_scalar),
        ],
    )
    _fd3_cn_sensitivity_kernel_overload[_scalar] = wp.overload(
        _fd3_cn_sensitivity_kernel,
        [
            wp.array2d(dtype=_scalar),
            wp.array2d(dtype=_scalar),
            wp.array(dtype=_scalar),
        ],
    )
    _fd3_cn_forces_kernel_overload[_scalar] = wp.overload(
        _fd3_cn_forces_kernel,
        [
            wp.array(dtype=_scalar),
            wp.array(dtype=_vector),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=_vector),
            wp.array(dtype=_scalar),
            _scalar,
            wp.array(dtype=wp.int32),
            wp.int32,
            wp.bool,
            wp.array(dtype=_vector),
            wp.array(dtype=_matrix),
        ],
    )


def fd3_kspace(
    mesh_fft: wp.array,
    k_matrix: wp.array,
    moduli_x: wp.array,
    moduli_y: wp.array,
    moduli_z: wp.array,
    volumes: wp.array,
    sqrt_q: wp.array,
    eigs: wp.array,
    s6: float,
    s8: float,
    a1: float,
    a2: float,
    mesh_dimensions: tuple[int, int, int],
    n_species: int,
    rank: int,
    energy: wp.array,
    cotangent: wp.array,
    virial: wp.array,
    wp_dtype: type,
    device: str | None = None,
    compute_virial: bool = False,
) -> None:
    r"""Contract the transformed mesh against the dispersion kernel.

    Consumes the forward real-to-complex transform of the spread mesh and produces the
    reciprocal-space energy together with the cotangent field.

    Parameters
    ----------
    mesh_fft : wp.array4d, shape (B * n_species * rank, nx, ny, nz // 2 + 1), dtype=wp.vec2f
        or wp.vec2d
        Transformed mesh, real and imaginary parts adjacent. Channel
        ``(b * n_species + species) * rank + slot``.
    k_matrix : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        ``2 * pi * inverse(cell)`` per system, so that ``k_matrix @ miller`` is the Cartesian
        wave vector.
    moduli_x, moduli_y, moduli_z : wp.array
        B-spline attenuation per axis, lengths ``nx``, ``ny`` and ``nz // 2 + 1``.
    volumes : wp.array, shape (B,)
        Cell volume per system.
    sqrt_q : wp.array, shape (n_species,)
        Square root of the quadrupole-to-dipole ratio per species channel.
    eigs : wp.array, shape (rank,)
        Eigenvalues of the decomposed reference tensor. May be negative.
    s6, s8, a1, a2 : float
        Becke-Johnson damping parameters. These are the only source of the damping used
        anywhere in the evaluation.
    mesh_dimensions : tuple[int, int, int]
        Full mesh size, before the real-to-complex transform halves the last axis.
    n_species, rank : int
        Channel layout of ``mesh_fft``.
    energy : wp.array, shape (B,)
        OUTPUT: accumulated reciprocal-space energy. Must be zero-initialised.
    cotangent : wp.array4d
        OUTPUT: field whose inverse transform, taken with an unnormalised convention, is
        ``dE/d(mesh)``. Same shape and dtype as ``mesh_fft``; fully overwritten.
    virial : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        IN-OUT: the reciprocal-space strain derivative is accumulated when
        ``compute_virial`` is set. Must be zero-initialised.
    wp_dtype : type
        Warp scalar dtype (``wp.float32`` or ``wp.float64``).
    device : str | None
        Warp device string. If None, inferred from arrays.
    compute_virial : bool, default=False
        Whether to accumulate the reciprocal-space virial. This is the only pass that can
        produce it.

    Notes
    -----
    The energy sum carries the real-to-complex multiplicity weight, because it reduces over a
    half spectrum. The cotangent does not: the inverse transform reconstructs the unstored
    half itself, so weighting it as well would double-count every interior bin.
    """
    num_systems = volumes.shape[0]
    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions
    num_bins = mesh_nx * mesh_ny * (mesh_nz // 2 + 1)
    # Tile reductions need a real block launch, which only CUDA provides; on CPU a block is
    # one thread, which reduces and accumulates its own bin.
    block = FD3_KSPACE_BLOCK_SIZE if "cuda" in str(device) else 1
    padded_bins = -(-num_bins // block) * block
    wp.launch(
        _fd3_kspace_kernel_overload[wp_dtype],
        dim=(num_systems, padded_bins),
        inputs=[
            mesh_fft,
            k_matrix,
            moduli_x,
            moduli_y,
            moduli_z,
            volumes,
            sqrt_q,
            eigs,
            wp_dtype(s6),
            wp_dtype(s8),
            wp_dtype(a1),
            wp_dtype(a2),
            wp.int32(mesh_nx),
            wp.int32(mesh_ny),
            wp.int32(mesh_nz),
            wp.int32(num_bins),
            wp.int32(block),
            wp.int32(n_species),
            wp.int32(rank),
            compute_virial,
        ],
        outputs=[energy, cotangent, virial],
        block_dim=block,
        device=device,
    )


def fd3_self_energy(
    c6: wp.array,
    species_index: wp.array,
    batch_idx: wp.array,
    sqrt_q: wp.array,
    eigs: wp.array,
    s6: float,
    s8: float,
    a1: float,
    a2: float,
    energy: wp.array,
    d_energy_d_c6: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Cancel the self pair and record its coefficient derivative.

    Must run before :func:`fd3_cn_chain`, because the term it adds depends on the
    coefficients and therefore reaches the forces through the coordination-number chain rule.

    Parameters
    ----------
    c6 : wp.array2d, shape (N, rank)
        Separable per-atom coefficients.
    species_index : wp.array, shape (N,), dtype=wp.int32
        Channel index per atom; negative marks an uncovered species, which is skipped.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index per atom.
    sqrt_q : wp.array, shape (n_species,)
        Square root of the quadrupole-to-dipole ratio per species channel.
    eigs : wp.array, shape (rank,)
        Eigenvalues of the decomposed reference tensor.
    s6, s8, a1, a2 : float
        Becke-Johnson damping parameters, the same values passed to :func:`fd3_kspace`.
    energy : wp.array, shape (B,)
        IN-OUT: the self-energy is added to the existing contents.
    d_energy_d_c6 : wp.array2d, shape (N, rank)
        IN-OUT: the coefficient derivative is added to the existing contents.
    wp_dtype : type
        Warp scalar dtype (``wp.float32`` or ``wp.float64``).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = c6.shape[0]
    if num_atoms == 0:
        return
    wp.launch(
        _fd3_self_energy_kernel_overload[wp_dtype],
        dim=num_atoms,
        inputs=[
            c6,
            species_index,
            batch_idx,
            sqrt_q,
            eigs,
            wp_dtype(s6),
            wp_dtype(s8),
            wp_dtype(a1),
            wp_dtype(a2),
        ],
        outputs=[energy, d_energy_d_c6],
        device=device,
    )


def fd3_cn_chain(
    d_energy_d_c6: wp.array,
    dc6_dcn: wp.array,
    positions: wp.array,
    numbers: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    cartesian_shifts: wp.array,
    covalent_radii: wp.array,
    r_cut: float,
    batch_idx: wp.array,
    d_energy_d_cn: wp.array,
    forces: wp.array,
    virial: wp.array,
    wp_dtype: type,
    device: str | None = None,
    compute_virial: bool = False,
) -> None:
    """Carry the coefficient derivative back to positions through the coordination numbers.

    Parameters
    ----------
    d_energy_d_c6 : wp.array2d, shape (N, rank)
        Derivative of the energy with respect to the separable coefficients. Every
        contribution must already be folded in, the self-energy included.
    dc6_dcn : wp.array2d, shape (N, rank)
        Derivative of the coefficients with respect to coordination number, from
        :func:`fd3_coefficients`.
    positions, numbers, idx_j, neighbor_ptr, cartesian_shifts, covalent_radii, r_cut
        The same neighbour-list description passed to :func:`fd3_coordination_numbers`.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index per atom.
    d_energy_d_cn : wp.array, shape (N,)
        OUTPUT: per-atom coordination sensitivity. Overwritten.
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        IN-OUT: the chain-rule contribution is accumulated into the existing contents.
    virial : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        IN-OUT: accumulated when ``compute_virial`` is set.
    wp_dtype : type
        Warp scalar dtype (``wp.float32`` or ``wp.float64``).
    device : str | None
        Warp device string. If None, inferred from arrays.
    compute_virial : bool, default=False
        Whether to accumulate the virial.
    """
    num_atoms = positions.shape[0]
    if num_atoms == 0:
        return
    wp.launch(
        _fd3_cn_sensitivity_kernel_overload[wp_dtype],
        dim=num_atoms,
        inputs=[d_energy_d_c6, dc6_dcn],
        outputs=[d_energy_d_cn],
        device=device,
    )
    # Tile reductions need a real block launch, which only CUDA provides; on CPU a block is
    # one thread, which walks the whole neighbour slice itself.
    block = FD3_CN_BLOCK_SIZE if "cuda" in str(device) else 1
    wp.launch(
        _fd3_cn_forces_kernel_overload[wp_dtype],
        dim=(num_atoms, block),
        inputs=[
            d_energy_d_cn,
            positions,
            numbers,
            idx_j,
            neighbor_ptr,
            cartesian_shifts,
            covalent_radii,
            wp_dtype(r_cut),
            batch_idx,
            wp.int32(block),
            compute_virial,
        ],
        outputs=[forces, virial],
        block_dim=block,
        device=device,
    )


@wp.kernel(enable_backward=False)
def _fd3_gather_and_force_kernel(
    potential: wp.array4d(dtype=Any),
    positions: wp.array(dtype=Any),
    c6: wp.array2d(dtype=Any),
    group_idx: wp.array(dtype=wp.int32),
    cell_inv_t: wp.array(dtype=Any),
    order: wp.int32,
    rank: wp.int32,
    d_energy_d_c6: wp.array2d(dtype=Any),
    forces: wp.array(dtype=Any),
):
    """Read the potential mesh once for both the coefficient derivative and the force.

    Two quantities come off the same mesh visit: the derivative with respect to the separable
    coefficients, which continues into the coordination-number chain rule, and the force from
    the gradient of the interpolation weights.

    No stencil-weight threshold is applied, matching :func:`fd3_spread`. The force has to be
    the gradient of the energy that was actually computed, so the two must agree on which
    stencil points exist; and since both keep every point, the interpolation is the exact
    B-spline rather than a thresholded approximation to it.

    Thread launch
    -------------
    One thread per atom, accumulating over that atom's stencil in registers.

    Modifies
    --------
    ``d_energy_d_c6[i, :]`` and ``forces[i]`` are accumulated for the atom owning the thread,
    so both must be zero-initialised. One thread owns each atom, so no atomics are needed.
    """
    atom_i = wp.tid()
    group = group_idx[atom_i]
    if group < 0:
        return

    mesh_dims = wp.vec3i(potential.shape[1], potential.shape[2], potential.shape[3])
    base_grid, theta = compute_fractional_coords(
        positions[atom_i], cell_inv_t[group], mesh_dims
    )

    force = positions[atom_i] - positions[atom_i]
    for point in range(order * order * order):
        offset = bspline_grid_offset(point, order, theta)
        weight = bspline_weight_3d(theta, offset, order)
        gradient = bspline_weight_gradient_3d(theta, offset, order, mesh_dims)

        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        # Fractional gradients become Cartesian through the inverse cell. The forward map
        # applies the transpose, so the adjoint takes it back off -- identical only for a
        # diagonal cell, which is why a cubic box cannot detect the difference.
        cartesian = wp.transpose(cell_inv_t[group]) * gradient

        for slot in range(rank):
            value = potential[group * rank + slot, gx, gy, gz]
            d_energy_d_c6[atom_i, slot] += value * weight
            force -= (c6[atom_i, slot] * value) * cartesian

    forces[atom_i] = forces[atom_i] + force


_fd3_gather_and_force_kernel_overload = {}

for _scalar, _vector, _matrix in zip(_SCALARS, _VECTORS, _MATRICES):
    _fd3_gather_and_force_kernel_overload[_scalar] = wp.overload(
        _fd3_gather_and_force_kernel,
        [
            wp.array4d(dtype=_scalar),
            wp.array(dtype=_vector),
            wp.array2d(dtype=_scalar),
            wp.array(dtype=wp.int32),
            wp.array(dtype=_matrix),
            wp.int32,
            wp.int32,
            wp.array2d(dtype=_scalar),
            wp.array(dtype=_vector),
        ],
    )


def fd3_gather_and_force(
    potential: wp.array,
    positions: wp.array,
    c6: wp.array,
    group_idx: wp.array,
    cell_inv_t: wp.array,
    spline_order: int,
    rank: int,
    d_energy_d_c6: wp.array,
    forces: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Gather the coefficient derivative and the direct mesh force in one pass.

    Parameters
    ----------
    potential : wp.array4d, shape (B * n_species * rank, nx, ny, nz)
        Inverse transform of the cotangent field from :func:`fd3_kspace`, taken with an
        unnormalised convention so that it is the derivative of the energy with respect to
        the mesh.
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    c6 : wp.array2d, shape (N, rank)
        Separable per-atom coefficients.
    group_idx : wp.array, shape (N,), dtype=wp.int32
        ``system * n_species + species`` per atom, the same index used when spreading.
        Negative marks an atom to skip.
    cell_inv_t : wp.array, shape (B * n_species,), dtype=wp.mat33f or wp.mat33d
        Transpose of the inverse cell per group, repeat-interleaved across species.
    spline_order : int
        B-spline order used for the spread.
    rank : int
        Channels per group.
    d_energy_d_c6 : wp.array2d, shape (N, rank)
        OUTPUT: derivative with respect to the coefficients. Must be zero-initialised.
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        IN-OUT: the direct mesh force is added to the existing contents.
    wp_dtype : type
        Warp scalar dtype (``wp.float32`` or ``wp.float64``).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    if num_atoms == 0:
        return
    wp.launch(
        _fd3_gather_and_force_kernel_overload[wp_dtype],
        dim=num_atoms,
        inputs=[
            potential,
            positions,
            c6,
            group_idx,
            cell_inv_t,
            wp.int32(spline_order),
            wp.int32(rank),
        ],
        outputs=[d_energy_d_c6, forces],
        device=device,
    )


@wp.kernel(enable_backward=False)
def _fd3_cn_matrix_kernel(
    positions: wp.array(dtype=Any),
    numbers: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    cartesian_shifts: wp.array2d(dtype=Any),
    covalent_radii: wp.array(dtype=Any),
    r_cut: Any,
    fill_value: wp.int32,
    block_stride: wp.int32,
    coord_num: wp.array(dtype=Any),
):
    """Coordination numbers from a dense padded neighbour matrix.

    Thread launch
    -------------
    ``dim = (num_atoms, FD3_CN_BLOCK_SIZE)`` with ``block_dim = FD3_CN_BLOCK_SIZE`` -- one
    block per atom, its threads striding ``block_stride`` apart along that atom's row.

    Modifies
    --------
    ``coord_num[i]`` is overwritten. Padding slots, marked by an index at or above
    ``fill_value``, are skipped.
    """
    atom_i, thread_in_block = wp.tid()
    if numbers[atom_i] == 0:
        return

    zero = type(r_cut)(0.0)
    total = zero
    rcov_i = covalent_radii[numbers[atom_i]]
    position_i = positions[atom_i]

    for slot in range(thread_in_block, neighbor_matrix.shape[1], block_stride):
        atom_j = neighbor_matrix[atom_i, slot]
        if atom_j < fill_value and numbers[atom_j] != 0:
            delta = positions[atom_j] + cartesian_shifts[atom_i, slot] - position_i
            distance = wp.length(delta)
            if distance < r_cut and distance != zero:
                value, _unused = _cn_counting(
                    distance, rcov_i + covalent_radii[numbers[atom_j]], r_cut, False
                )
                total += value

    block_total = wp.tile_sum(wp.tile(total))[0]
    if thread_in_block == 0:
        coord_num[atom_i] = block_total


@wp.kernel(enable_backward=False)
def _fd3_cn_forces_matrix_kernel(
    d_energy_d_cn: wp.array(dtype=Any),
    positions: wp.array(dtype=Any),
    numbers: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    cartesian_shifts: wp.array2d(dtype=Any),
    covalent_radii: wp.array(dtype=Any),
    r_cut: Any,
    fill_value: wp.int32,
    batch_idx: wp.array(dtype=wp.int32),
    block_stride: wp.int32,
    compute_virial: bool,
    forces: wp.array(dtype=Any),
    virial: wp.array(dtype=Any),
):
    """Coordination chain rule over a dense padded neighbour matrix.

    Thread launch
    -------------
    ``dim = (num_atoms, FD3_CN_BLOCK_SIZE)`` with ``block_dim = FD3_CN_BLOCK_SIZE`` -- one
    block per atom, its threads striding ``block_stride`` apart along that atom's row and
    reducing across the block at the end.

    Modifies
    --------
    ``forces[i]`` for the atom the block owns, and ``virial`` per system when requested.
    """
    atom_i, thread_in_block = wp.tid()
    if numbers[atom_i] == 0:
        return

    rcov_i = covalent_radii[numbers[atom_i]]
    position_i = positions[atom_i]
    sensitivity_i = d_energy_d_cn[atom_i]

    zero = type(r_cut)(0.0)
    half = type(r_cut)(0.5)
    force_x = zero
    force_y = zero
    force_z = zero
    virial_xx = zero
    virial_yy = zero
    virial_zz = zero
    virial_xy = zero
    virial_xz = zero
    virial_yz = zero

    for slot in range(thread_in_block, neighbor_matrix.shape[1], block_stride):
        atom_j = neighbor_matrix[atom_i, slot]
        if atom_j < fill_value and numbers[atom_j] != 0:
            delta = positions[atom_j] + cartesian_shifts[atom_i, slot] - position_i
            distance = wp.length(delta)
            if distance < r_cut and distance != zero:
                _value, slope = _cn_counting(
                    distance, rcov_i + covalent_radii[numbers[atom_j]], r_cut, True
                )
                magnitude = (sensitivity_i + d_energy_d_cn[atom_j]) * slope
                pair_force = (magnitude / distance) * delta
                force_x += pair_force[0]
                force_y += pair_force[1]
                force_z += pair_force[2]

                if compute_virial:
                    # F outer delta, symmetric because the force lies along the separation.
                    # Halved because the reverse edge contributes the same again.
                    virial_xx += half * pair_force[0] * delta[0]
                    virial_yy += half * pair_force[1] * delta[1]
                    virial_zz += half * pair_force[2] * delta[2]
                    virial_xy += half * pair_force[0] * delta[1]
                    virial_xz += half * pair_force[0] * delta[2]
                    virial_yz += half * pair_force[1] * delta[2]

    sum_fx = wp.tile_sum(wp.tile(force_x))[0]
    sum_fy = wp.tile_sum(wp.tile(force_y))[0]
    sum_fz = wp.tile_sum(wp.tile(force_z))[0]
    if thread_in_block == 0:
        # The gather pass has already written this atom's mesh force, so this accumulates.
        wp.atomic_add(forces, atom_i, wp.vector(sum_fx, sum_fy, sum_fz))

    if compute_virial:
        sum_xx = wp.tile_sum(wp.tile(virial_xx))[0]
        sum_yy = wp.tile_sum(wp.tile(virial_yy))[0]
        sum_zz = wp.tile_sum(wp.tile(virial_zz))[0]
        sum_xy = wp.tile_sum(wp.tile(virial_xy))[0]
        sum_xz = wp.tile_sum(wp.tile(virial_xz))[0]
        sum_yz = wp.tile_sum(wp.tile(virial_yz))[0]
        if thread_in_block == 0:
            # One atomic per atom, not per neighbour: a system shares one accumulator.
            # Negate: conventions.md (and dftd3) define the virial as -dE/du.
            wp.atomic_add(
                virial,
                batch_idx[atom_i],
                type(r_cut)(-1.0)
                * wp.matrix_from_rows(
                    wp.vector(sum_xx, sum_xy, sum_xz),
                    wp.vector(sum_xy, sum_yy, sum_yz),
                    wp.vector(sum_xz, sum_yz, sum_zz),
                ),
            )


_fd3_cn_matrix_kernel_overload = {}
_fd3_cn_forces_matrix_kernel_overload = {}

for _scalar, _vector, _matrix in zip(_SCALARS, _VECTORS, _MATRICES):
    _fd3_cn_matrix_kernel_overload[_scalar] = wp.overload(
        _fd3_cn_matrix_kernel,
        [
            wp.array(dtype=_vector),
            wp.array(dtype=wp.int32),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=_vector),
            wp.array(dtype=_scalar),
            _scalar,
            wp.int32,
            wp.int32,
            wp.array(dtype=_scalar),
        ],
    )
    _fd3_cn_forces_matrix_kernel_overload[_scalar] = wp.overload(
        _fd3_cn_forces_matrix_kernel,
        [
            wp.array(dtype=_scalar),
            wp.array(dtype=_vector),
            wp.array(dtype=wp.int32),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=_vector),
            wp.array(dtype=_scalar),
            _scalar,
            wp.int32,
            wp.array(dtype=wp.int32),
            wp.int32,
            wp.bool,
            wp.array(dtype=_vector),
            wp.array(dtype=_matrix),
        ],
    )


def fd3_coordination_numbers_matrix(
    positions: wp.array,
    numbers: wp.array,
    neighbor_matrix: wp.array,
    cartesian_shifts: wp.array,
    covalent_radii: wp.array,
    r_cut: float,
    coord_num: wp.array,
    wp_dtype: type,
    device: str | None = None,
    fill_value: int | None = None,
) -> None:
    """Coordination numbers from a dense neighbour matrix.

    The dense counterpart of :func:`fd3_coordination_numbers`; see that function for the
    shared parameters and for the requirement that ``r_cut`` match the radius the neighbour
    list was built with.

    Parameters
    ----------
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Neighbour indices, padded with values at or above ``fill_value``.
    cartesian_shifts : wp.array2d, shape (N, max_neighbors)
        Periodic image offset for each slot, in Cartesian units.
    fill_value : int, optional
        Padding sentinel. Defaults to the atom count, matching the neighbour-list builders.
    """
    num_atoms = positions.shape[0]
    if num_atoms == 0:
        return
    if fill_value is None:
        fill_value = num_atoms
    # Tile reductions need a real block launch, which only CUDA provides; on CPU a block is
    # one thread, which walks the whole neighbour row itself.
    block = FD3_CN_BLOCK_SIZE if "cuda" in str(device) else 1
    wp.launch(
        _fd3_cn_matrix_kernel_overload[wp_dtype],
        dim=(num_atoms, block),
        inputs=[
            positions,
            numbers,
            neighbor_matrix,
            cartesian_shifts,
            covalent_radii,
            wp_dtype(r_cut),
            wp.int32(fill_value),
            wp.int32(block),
        ],
        outputs=[coord_num],
        block_dim=block,
        device=device,
    )


def fd3_cn_chain_matrix(
    d_energy_d_c6: wp.array,
    dc6_dcn: wp.array,
    positions: wp.array,
    numbers: wp.array,
    neighbor_matrix: wp.array,
    cartesian_shifts: wp.array,
    covalent_radii: wp.array,
    r_cut: float,
    batch_idx: wp.array,
    d_energy_d_cn: wp.array,
    forces: wp.array,
    virial: wp.array,
    wp_dtype: type,
    device: str | None = None,
    compute_virial: bool = False,
    fill_value: int | None = None,
) -> None:
    """Coordination chain rule over a dense neighbour matrix.

    The dense counterpart of :func:`fd3_cn_chain`; see that function for the shared
    parameters.

    Parameters
    ----------
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Neighbour indices, padded with values at or above ``fill_value``.
    cartesian_shifts : wp.array2d, shape (N, max_neighbors)
        Periodic image offset for each slot, in Cartesian units.
    fill_value : int, optional
        Padding sentinel. Defaults to the atom count.
    """
    num_atoms = positions.shape[0]
    if num_atoms == 0:
        return
    if fill_value is None:
        fill_value = num_atoms
    wp.launch(
        _fd3_cn_sensitivity_kernel_overload[wp_dtype],
        dim=num_atoms,
        inputs=[d_energy_d_c6, dc6_dcn],
        outputs=[d_energy_d_cn],
        device=device,
    )
    # Tile reductions need a real block launch, which only CUDA provides; on CPU a block is
    # one thread, which walks the whole neighbour row itself.
    block = FD3_CN_BLOCK_SIZE if "cuda" in str(device) else 1
    wp.launch(
        _fd3_cn_forces_matrix_kernel_overload[wp_dtype],
        dim=(num_atoms, block),
        inputs=[
            d_energy_d_cn,
            positions,
            numbers,
            neighbor_matrix,
            cartesian_shifts,
            covalent_radii,
            wp_dtype(r_cut),
            wp.int32(fill_value),
            batch_idx,
            wp.int32(block),
            compute_virial,
        ],
        outputs=[forces, virial],
        block_dim=block,
        device=device,
    )


@wp.kernel(enable_backward=False)
def _fd3_spread_kernel(
    positions: wp.array(dtype=Any),
    values: wp.array2d(dtype=Any),
    group_idx: wp.array(dtype=wp.int32),
    cell_inv_t: wp.array(dtype=Any),
    order: wp.int32,
    num_channels: wp.int32,
    mesh: wp.array4d(dtype=Any),
):
    """Spread the separable coefficients onto the mesh.

    Equivalent to the general-purpose channel spread in :mod:`nvalchemiops.math.spline`, but
    with **no stencil-weight threshold**. That threshold is an efficiency measure: it skips
    atomic adds whose contribution is negligible. It is not free, though, because the
    interpolation is what the forces are differentiated through. Dropping a stencil point
    removes its gradient as well as its value, and while the energy shifts by a negligible
    amount the force does not: for an atom sitting a thousandth of a cell from a boundary a
    quarter of the stencil falls below the threshold, and the resulting force error survives
    mesh refinement.

    Keeping every stencil point costs at most a few extra atomic adds per atom and makes the
    spread the exact B-spline, which is what the gather in this module differentiates.

    Thread launch
    -------------
    ``dim = (num_atoms, order**3)`` -- one thread per atom and stencil point.

    Modifies
    --------
    ``mesh`` is accumulated into at ``group_idx * num_channels + channel``, so it must be
    zero-initialised.
    """
    atom_idx, point_idx = wp.tid()
    group = group_idx[atom_idx]
    if group < 0:
        return

    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    base_grid, theta = compute_fractional_coords(
        positions[atom_idx], cell_inv_t[group], mesh_dims
    )
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
    gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
    gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

    for channel in range(num_channels):
        wp.atomic_add(
            mesh,
            group * num_channels + channel,
            gx,
            gy,
            gz,
            values[atom_idx, channel] * weight,
        )


_fd3_spread_kernel_overload = {}

for _scalar, _vector, _matrix in zip(_SCALARS, _VECTORS, _MATRICES):
    _fd3_spread_kernel_overload[_scalar] = wp.overload(
        _fd3_spread_kernel,
        [
            wp.array(dtype=_vector),
            wp.array2d(dtype=_scalar),
            wp.array(dtype=wp.int32),
            wp.array(dtype=_matrix),
            wp.int32,
            wp.int32,
            wp.array4d(dtype=_scalar),
        ],
    )


def fd3_spread(
    positions: wp.array,
    c6: wp.array,
    group_idx: wp.array,
    cell_inv_t: wp.array,
    spline_order: int,
    rank: int,
    mesh: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Spread the separable coefficients onto the mesh.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    c6 : wp.array2d, shape (N, rank)
        Separable per-atom coefficients.
    group_idx : wp.array, shape (N,), dtype=wp.int32
        ``system * n_species + species`` per atom. Negative marks an atom to skip.
    cell_inv_t : wp.array, shape (B * n_species,), dtype=wp.mat33f or wp.mat33d
        Transpose of the inverse cell per group, repeat-interleaved across species.
    spline_order : int
        B-spline order.
    rank : int
        Channels per group.
    mesh : wp.array4d, shape (B * n_species * rank, nx, ny, nz)
        OUTPUT: accumulated into, so it must be zero-initialised.
    wp_dtype : type
        Warp scalar dtype (``wp.float32`` or ``wp.float64``).
    device : str | None
        Warp device string. If None, inferred from arrays.

    See Also
    --------
    fd3_gather_and_force : The adjoint, which applies the same interpolation.
    """
    num_atoms = positions.shape[0]
    if num_atoms == 0:
        return
    wp.launch(
        _fd3_spread_kernel_overload[wp_dtype],
        dim=(num_atoms, spline_order**3),
        inputs=[
            positions,
            c6,
            group_idx,
            cell_inv_t,
            wp.int32(spline_order),
            wp.int32(rank),
        ],
        outputs=[mesh],
        device=device,
    )

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

r"""``ewald_real`` per-component kernel factory.

One source per ``(wp_dtype, batched, neighbor_input)`` triple is specialized into
the derivative matrix by capturing the ``DERIV`` / ``CELL_GRAD`` axes as Python
compile-time constants; Warp's codegen dead-eliminates unused branches. Three
``order`` families share the same per-pair scalar ``@wp.func`` cores so derivatives
are consistent and forward energies/forces stay bit-exact vs the hand-written
kernels:

* ``order="forward"`` -- energy (``E``), energy+forces (``E_F``),
  energy+forces+charge-grad (``E_F_dQ``), plus the optional virial gated by
  ``cell_grad``. ``E_F`` / ``E_F_dQ`` / virial reproduce the hand-written
  ``ewald_real_space_energy_forces[_charge_grad][_matrix]`` kernels exactly and are
  the parity oracle.
* ``order="backward"`` -- the first-derivative autograd node: the same per-pair
  force / charge-grad / virial core as ``forward`` ``E_F``, scaled by the upstream
  per-system energy cotangent ``grad_energy``. Emits ``dL/dR`` into
  ``atomic_forces``, ``dL/dq`` into ``charge_gradients`` and (``cell_grad``)
  ``dL/dcell``-as-virial into ``virial``.
* ``order="double_backward"`` -- the second-derivative (pair-Hessian / HVP) node,
  recompute mode: recomputes per-pair erfc/force terms from the neighbor data
  and contracts the pair Hessian with the cotangents on the backward outputs.

The forward 15-parameter list is identical across
``(dtype, batched, neighbor_input)``. The backward / double-backward signatures are
defined here (the contract froze only ``forward``) and are the sibling-factory pattern:
.. math::

    \mathrm{backward}(\bar{E}, x) \rightarrow \bar{x}

.. math::

    \mathrm{double\_backward}(\bar{\bar{x}}, x, \bar{E})
    \rightarrow \bar{\bar{x}}_\mathrm{inputs}

These signatures match the codebase ``register_warp_op_chain`` convention and the
PME ``*_double_backward`` precedent.

The double-backward kernel implements the position, charge **and cell** second-order
terms, so a stress-loss double-backward (nonzero ``v_cell``) is fully supported: it
emits ``grad_cell`` (the cell-self term) plus the cross-terms from ``v_cell`` into
``grad_positions`` (cell<->position) and ``grad_charges`` (cell<->charge). These are
    the directional derivative of the backward **cell output** -- the strain-virial state
``W = -dE/dstrain``, not the literal ``dE/dcell`` -- which is the validation contract.
Cell enters only through the periodic separation for integer lattice shift ``n``:

.. math::

    r_{ij} = R_j - R_i + h^\mathsf{T} n,
    \qquad
    \frac{\partial r_{ij,a}}{\partial h_{p,q}} = \delta_{a,q} n_p.

Equivalently, ``d Phi / d cell`` is the outer product of ``n`` and
``d Phi / d sep``.
"""

from functools import lru_cache
from typing import Any

import warp as wp

from nvalchemiops.interactions.electrostatics._factory_common import (
    _DISTANCE_EPSILON,
    _DTYPE_INFO,
    _alloc_sentinels,
    _deriv_token,
    _DerivState,
    _ewald_charge_potential_deriv,
    _ewald_half_force_scale,
    _ewald_half_force_scale_deriv,
    _make_specialization_module_name,
    _name_and_document_kernel,
    _pair_virial_outer,
    _require_component,
    _require_supported_dtype,
    _validate_common_axes,
)
from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    _ewald_real_space_charge_grad_potential,
    _ewald_real_space_energy_kernel_compute_energy,
    _ewald_real_space_force_magnitude,
)

__all__ = [
    "alloc_ewald_real_sentinels",
    "get_ewald_real_kernel",
    "make_ewald_real_kernel",
]


@wp.func
def _periodic_separation(
    pos_i: Any, pos_j: Any, cell_t: Any, shift_vec: wp.vec3i
) -> Any:
    """Return ``pos_j - pos_i + cell_t @ shift_vec`` in the input vector dtype."""
    shift = type(pos_i)(
        type(pos_i[0])(shift_vec[0]),
        type(pos_i[0])(shift_vec[1]),
        type(pos_i[0])(shift_vec[2]),
    )
    return pos_j - pos_i + cell_t * shift


# === Sentinel allocators ===
#
# slot_spec: name -> (shape, dtype_token).  ``vec`` / ``mat`` resolve against
# ``wp_dtype``; ``f64`` / ``i32`` / ``vec3i`` are fixed.  These must match each
# kernel slot's dtype exactly (Warp type-checks even zero-size arrays at launch).
_SENTINEL_SLOT_SPEC: dict[str, tuple[tuple[int, ...], str]] = {
    # forward inputs
    "batch_id": ((0,), "i32"),
    "idx_j": ((0,), "i32"),
    "neighbor_ptr": ((0,), "i32"),
    "unit_shifts": ((0,), "vec3i"),
    "neighbor_matrix": ((0, 0), "i32"),
    "unit_shifts_matrix": ((0, 0), "vec3i"),
    # forward outputs
    "atomic_forces": ((0,), "vec"),
    "charge_gradients": ((0,), "f64"),
    "virial": ((0,), "mat"),
    # backward / double-backward cotangents + outputs
    "grad_energy": ((0,), "f64"),
    "v_pos": ((0,), "vec"),
    "v_charge": ((0,), "f64"),
    "v_cell": ((0,), "mat"),
    "grad_grad_energy": ((0,), "f64"),
    "grad_positions": ((0,), "vec"),
    "grad_charges": ((0,), "f64"),
    "grad_cell": ((0,), "mat"),
}


_SENTINEL_CACHE: dict[tuple[type, str], dict[str, wp.array]] = {}


def alloc_ewald_real_sentinels(wp_dtype: type, device: str) -> dict[str, wp.array]:
    """Return cached zero-size sentinel arrays for inactive ``ewald_real`` slots.

    Callers pass these for any input/output slot not active in the current
    specialization; the kernel's compile-time branches never index them. Covers
    the forward 15-param list plus the extra slots the backward / double-backward
    kernels add (``grad_energy``, ``v_pos``, ``v_charge``, ``v_cell``,
    ``grad_grad_energy``, ``grad_positions``, ``grad_charges``, ``grad_cell``).

    Memoized per ``(wp_dtype, device)``: the sentinels are zero-size and never
    written (only type/shape-checked at launch), so the same arrays are reused
    across calls instead of re-allocating ~17 arrays every forward/backward (a
    measurable per-call cost at small system sizes).

    Returns
    -------
    dict[str, wp.array]
        Sentinels keyed by parameter name (shared, read-only -- do not mutate).
    """
    key = (wp_dtype, str(device))
    cached = _SENTINEL_CACHE.get(key)
    if cached is None:
        cached = _alloc_sentinels(wp_dtype, device, _SENTINEL_SLOT_SPEC)
        _SENTINEL_CACHE[key] = cached
    return cached


def _ewald_real_module_name(
    wp_dtype: type,
    batched: bool,
    neighbor_input: str,
    order: str = "forward",
    tiled: bool = False,
    cell_literal: bool = False,
) -> str:
    """Deterministic Warp ``module=`` name for one ``ewald_real`` specialization.

    A stable per-spec module name (rather than mtime-newest) lets the
    dead-branch-elimination test locate generated source deterministically. The
    ``tiled`` cooperative-block variant gets a distinct ``_tiled`` module so it
    never collides with the serial (bit-exact parity) kernel of the same axes. The
    ``cell_literal`` forward variant (extra ``dedcell_atom`` output) appends a
    further ``_cellliteral`` segment so it never collides with the plain ``_tiled``
    forward kernel of the same axes.
    """
    parts = []
    if tiled:
        parts.append("tiled" if order == "forward" else f"{order}_tiled")
    if cell_literal:
        parts.append("cellliteral")
    suffix = "_".join(parts) if parts else None
    return _make_specialization_module_name(
        "ewald_real",
        wp_dtype=wp_dtype,
        batched=batched,
        neighbor_input=neighbor_input,
        order=order,
        suffix=suffix,
    )


def _name_and_document(
    kernel: wp.Kernel,
    *,
    base: str,
    wp_dtype: type,
    batched: bool,
    neighbor_input: str,
    deriv_state: _DerivState,
    cell_grad: bool,
    order: str,
    tiled: bool = False,
    cell_literal: bool = False,
) -> None:
    """Give a generated ``ewald_real`` kernel a descriptive name + spec docstring."""
    features = [
        _deriv_token(deriv_state),
        "cellgrad" if cell_grad else "",
        "cellliteral" if cell_literal else "",
        "batch" if batched else "single",
        neighbor_input,
        "" if order == "forward" else order,
        "tiled" if tiled else "",
    ]
    _name_and_document_kernel(
        kernel,
        base=base,
        wp_dtype=wp_dtype,
        features=features,
        entries=[
            ("batched", bool(batched)),
            ("neighbor_input", neighbor_input),
            ("deriv_state", deriv_state.name),
            ("cell_grad", bool(cell_grad)),
            ("order", order),
        ],
    )


# === Per-pair helper factories ===


@lru_cache(maxsize=None)
def _make_forward_pair_fn(
    wp_dtype: type,
    *,
    deriv_state: _DerivState,
    cell_grad: bool,
    cell_literal: bool = False,
) -> wp.Function:
    """Build the specialized forward per-pair accumulator.

    ``cell_literal=True`` builds a variant with an extra per-atom ``dedcell_acc``
    (``wp.mat33d``) accumulator + return value that sums the literal cell-gradient
    block ``n (x) (-force)`` (with ``n`` the integer lattice shift). This is the
    forward-fused literal ``dE/dcell`` used by the torch autograd chain so its first
    backward is a pure scatter (no separate cell kernel launch). It is distinct from
    the strain-virial ``W`` term (``CELL_GRAD``), which the legacy direct
    ``compute_virial`` output still needs. The ``cell_literal=False`` function is the
    byte-identical accumulator every other launch (JAX, direct, tests) uses.
    """
    info = _DTYPE_INFO[wp_dtype]
    vec_dtype = info.vec
    mat_dtype = info.mat
    has_force = deriv_state in {_DerivState.E_F, _DerivState.E_F_dQ}
    has_charge = deriv_state in {_DerivState.E_dQ, _DerivState.E_F_dQ}
    HAS_FORCE = wp.constant(has_force)
    HAS_CHARGE = wp.constant(has_charge)
    CELL_GRAD = wp.constant(bool(cell_grad))

    if cell_literal:

        @wp.func
        def _forward_pair_cell_literal(
            j: wp.int32,
            qi: wp.float64,
            pos_i: vec_dtype,
            pos_j: vec_dtype,
            shift_vec: wp.vec3i,
            cell_t: mat_dtype,
            alpha_: wp.float64,
            energy_acc: wp.float64,
            force_i_acc: vec_dtype,
            cg_i_acc: wp.float64,
            virial_acc: wp.mat33d,
            dedcell_acc: wp.mat33d,
            charges: wp.array(dtype=wp_dtype),
            atomic_forces: wp.array(dtype=vec_dtype),
            charge_gradients: wp.array(dtype=wp.float64),
        ):
            """Forward per-pair accumulator + literal ``dE/dcell`` block (atom ``i``).

            Identical pair math to :func:`_forward_pair`, plus the literal cell
            gradient contribution ``dedcell_acc += n (x) (-force)`` (``-force`` ==
            ``dE/dsep``; ``n`` == integer shift). The block is f64 (``wp.mat33d``)
            regardless of ``wp_dtype`` to match the f64 cache the chain allocates.
            """
            qj = wp.float64(charges[j])
            separation_vector = _periodic_separation(pos_i, pos_j, cell_t, shift_vec)
            distance = wp.float64(wp.length(separation_vector))

            if distance > wp.float64(_DISTANCE_EPSILON):
                energy_acc += _ewald_real_space_energy_kernel_compute_energy(
                    qi, qj, distance, alpha_
                )
                if HAS_FORCE:
                    force_mag = _ewald_real_space_force_magnitude(
                        qi, qj, distance, alpha_
                    )
                    force = type(pos_i)(
                        type(pos_i[0])(force_mag) * separation_vector[0],
                        type(pos_i[0])(force_mag) * separation_vector[1],
                        type(pos_i[0])(force_mag) * separation_vector[2],
                    )
                    force_i_acc -= force
                    wp.atomic_add(atomic_forces, j, force)
                    if HAS_CHARGE:
                        potential = _ewald_real_space_charge_grad_potential(
                            distance, alpha_
                        )
                        cg_i_acc += qj * potential
                        wp.atomic_add(charge_gradients, j, qi * potential)
                    if CELL_GRAD:
                        virial_acc += _pair_virial_outer(separation_vector, force)
                    # Literal dE/dcell block for atom i: n (x) dE/dsep, dE/dsep = -force.
                    n_vec = wp.vec3d(
                        wp.float64(shift_vec[0]),
                        wp.float64(shift_vec[1]),
                        wp.float64(shift_vec[2]),
                    )
                    neg_force = wp.vec3d(
                        -wp.float64(force[0]),
                        -wp.float64(force[1]),
                        -wp.float64(force[2]),
                    )
                    dedcell_acc += wp.outer(n_vec, neg_force)

            return energy_acc, force_i_acc, cg_i_acc, virial_acc, dedcell_acc

        return _forward_pair_cell_literal

    @wp.func
    def _forward_pair(
        j: wp.int32,
        qi: wp.float64,
        pos_i: vec_dtype,
        pos_j: vec_dtype,
        shift_vec: wp.vec3i,
        cell_t: mat_dtype,
        alpha_: wp.float64,
        energy_acc: wp.float64,
        force_i_acc: vec_dtype,
        cg_i_acc: wp.float64,
        virial_acc: wp.mat33d,
        charges: wp.array(dtype=wp_dtype),
        atomic_forces: wp.array(dtype=vec_dtype),
        charge_gradients: wp.array(dtype=wp.float64),
    ):
        """Accumulate one real-space pair for the forward kernel.

        Parameters are the thread-local atom ``i`` state plus the neighbor ``j`` and
        its periodic shift. Enabled derivative buffers are modified in place via
        atomics for neighbor-``j`` outputs; returned accumulators belong to atom
        ``i``. ``HAS_FORCE``, ``HAS_CHARGE`` and ``CELL_GRAD`` are static
        specializations, so inactive sentinel buffers are not read.
        """
        qj = wp.float64(charges[j])
        separation_vector = _periodic_separation(pos_i, pos_j, cell_t, shift_vec)
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(_DISTANCE_EPSILON):
            energy_acc += _ewald_real_space_energy_kernel_compute_energy(
                qi, qj, distance, alpha_
            )
            if HAS_FORCE:
                force_mag = _ewald_real_space_force_magnitude(qi, qj, distance, alpha_)
                force = type(pos_i)(
                    type(pos_i[0])(force_mag) * separation_vector[0],
                    type(pos_i[0])(force_mag) * separation_vector[1],
                    type(pos_i[0])(force_mag) * separation_vector[2],
                )
                force_i_acc -= force
                wp.atomic_add(atomic_forces, j, force)
                if CELL_GRAD:
                    virial_acc += _pair_virial_outer(separation_vector, force)
            if HAS_CHARGE:
                potential = _ewald_real_space_charge_grad_potential(distance, alpha_)
                cg_i_acc += qj * potential
                wp.atomic_add(charge_gradients, j, qi * potential)

        return energy_acc, force_i_acc, cg_i_acc, virial_acc

    return _forward_pair


@lru_cache(maxsize=None)
def _make_backward_pair_fn(
    wp_dtype: type,
    *,
    deriv_state: _DerivState,
    cell_grad: bool,
) -> wp.Function:
    """Build the specialized backward per-pair accumulator."""
    info = _DTYPE_INFO[wp_dtype]
    vec_dtype = info.vec
    mat_dtype = info.mat
    has_force = deriv_state in {_DerivState.E_F, _DerivState.E_F_dQ}
    has_charge = deriv_state in {_DerivState.E_dQ, _DerivState.E_F_dQ}
    HAS_FORCE = wp.constant(has_force)
    HAS_CHARGE = wp.constant(has_charge)
    CELL_GRAD = wp.constant(bool(cell_grad))

    @wp.func
    def _backward_pair(
        j: wp.int32,
        qi: wp.float64,
        pos_i: vec_dtype,
        pos_j: vec_dtype,
        shift_vec: wp.vec3i,
        cell_t: mat_dtype,
        alpha_: wp.float64,
        ge: wp.float64,
        gpos_i: vec_dtype,
        cg_i_acc: wp.float64,
        virial_acc: wp.mat33d,
        charges: wp.array(dtype=wp_dtype),
        atomic_forces: wp.array(dtype=vec_dtype),
        charge_gradients: wp.array(dtype=wp.float64),
    ):
        """Accumulate one real-space pair for the backward kernel.

        The helper emits neighbor-``j`` gradients atomically and returns the
        thread-local atom-``i`` accumulators. ``HAS_CHARGE`` and ``CELL_GRAD`` are
        static specializations, so inactive sentinel buffers are not read.
        """
        qj = wp.float64(charges[j])
        separation_vector = _periodic_separation(pos_i, pos_j, cell_t, shift_vec)
        distance = wp.float64(wp.length(separation_vector))
        if distance > wp.float64(_DISTANCE_EPSILON):
            ge_fm = wp.float64(0.0)
            if HAS_FORCE or CELL_GRAD:
                force_mag = _ewald_real_space_force_magnitude(qi, qj, distance, alpha_)
                ge_fm = ge * force_mag
                if HAS_FORCE:
                    # dL/dR_i += ge*(+F); dL/dR_j += ge*(-F).
                    gpos_i += type(pos_i)(
                        type(pos_i[0])(ge_fm) * separation_vector[0],
                        type(pos_i[0])(ge_fm) * separation_vector[1],
                        type(pos_i[0])(ge_fm) * separation_vector[2],
                    )
                    wp.atomic_add(
                        atomic_forces,
                        j,
                        type(pos_i)(
                            -type(pos_i[0])(ge_fm) * separation_vector[0],
                            -type(pos_i[0])(ge_fm) * separation_vector[1],
                            -type(pos_i[0])(ge_fm) * separation_vector[2],
                        ),
                    )
            if HAS_CHARGE:
                potential = _ewald_real_space_charge_grad_potential(distance, alpha_)
                cg_i_acc += ge * qj * potential
                wp.atomic_add(charge_gradients, j, ge * qi * potential)
            if CELL_GRAD:
                force_d = wp.vec3d(
                    ge_fm * wp.float64(separation_vector[0]),
                    ge_fm * wp.float64(separation_vector[1]),
                    ge_fm * wp.float64(separation_vector[2]),
                )
                sep_d = wp.vec3d(
                    wp.float64(separation_vector[0]),
                    wp.float64(separation_vector[1]),
                    wp.float64(separation_vector[2]),
                )
                virial_acc += wp.mat33d(wp.outer(sep_d, force_d))

        return gpos_i, cg_i_acc, virial_acc

    return _backward_pair


@lru_cache(maxsize=None)
def _make_double_backward_pair_fn(
    wp_dtype: type,
    *,
    deriv_state: _DerivState,
    cell_grad: bool,
) -> wp.Function:
    """Build the specialized double-backward per-pair accumulator."""
    info = _DTYPE_INFO[wp_dtype]
    vec_dtype = info.vec
    mat_dtype = info.mat
    HAS_CHARGE = wp.constant(deriv_state in {_DerivState.E_dQ, _DerivState.E_F_dQ})
    CELL_GRAD = wp.constant(bool(cell_grad))

    @wp.func
    def _double_backward_pair(
        j: wp.int32,
        qi: wp.float64,
        pos_i: vec_dtype,
        pos_j: vec_dtype,
        shift_vec: wp.vec3i,
        cell_t: mat_dtype,
        alpha_: wp.float64,
        ge: wp.float64,
        vi: wp.vec3d,
        vqi: wp.float64,
        m_cell: wp.mat33d,
        m_sym: wp.mat33d,
        ddE_acc: wp.float64,
        gpos_i: wp.vec3d,
        gq_i: wp.float64,
        gcell_acc: wp.mat33d,
        charges: wp.array(dtype=wp_dtype),
        v_pos: wp.array(dtype=vec_dtype),
        v_charge: wp.array(dtype=wp.float64),
        grad_positions: wp.array(dtype=vec_dtype),
        grad_charges: wp.array(dtype=wp.float64),
    ):
        """Accumulate one real-space pair for the double-backward kernel.

        This is the shared pair-Hessian / HVP body for CSR and matrix neighbor
        inputs. It atomically emits neighbor-``j`` position and charge terms and
        returns atom-``i`` / per-system accumulators. ``HAS_CHARGE`` and
        ``CELL_GRAD`` are static specializations, so inactive sentinel buffers are
        not read.
        """
        qj = wp.float64(charges[j])
        separation_vector = _periodic_separation(pos_i, pos_j, cell_t, shift_vec)
        distance = wp.float64(wp.length(separation_vector))
        if distance > wp.float64(_DISTANCE_EPSILON):
            sep = wp.vec3d(
                wp.float64(separation_vector[0]),
                wp.float64(separation_vector[1]),
                wp.float64(separation_vector[2]),
            )
            vj = wp.vec3d(
                wp.float64(v_pos[j][0]),
                wp.float64(v_pos[j][1]),
                wp.float64(v_pos[j][2]),
            )

            half_s = _ewald_half_force_scale(distance, alpha_)
            half_ds = _ewald_half_force_scale_deriv(distance, alpha_)

            fm = qi * qj * half_s
            coeff = qi * qj * half_ds / distance
            vij = vi - vj

            # Position Hessian: J (vj - vi) on i; J (vi - vj) on j.
            jw = -fm * vij - coeff * wp.dot(sep, vij) * sep
            gpos_i += ge * jw
            gpos_j = -ge * jw

            force_vec = fm * sep
            ddE_acc += wp.dot(force_vec, vij)

            vquad = wp.float64(0.0)
            if CELL_GRAD:
                n_vec = wp.vec3d(
                    wp.float64(shift_vec[0]),
                    wp.float64(shift_vec[1]),
                    wp.float64(shift_vec[2]),
                )
                vquad = wp.dot(sep, m_cell * sep)
                ddE_acc += fm * vquad
                gR_cell = -ge * (coeff * vquad * sep + fm * (m_sym * sep))
                gpos_i += gR_cell
                gpos_j += -gR_cell
                sv = wp.dot(sep, vij)
                grad_sep = (
                    coeff * vquad * sep
                    + fm * (m_sym * sep)
                    + coeff * sv * sep
                    + fm * vij
                )
                gcell_acc += ge * wp.outer(n_vec, grad_sep)

            if HAS_CHARGE:
                vqj = v_charge[j]
                g_pot = _ewald_real_space_charge_grad_potential(distance, alpha_)
                dg = _ewald_charge_potential_deriv(distance, alpha_)
                ddE_acc += g_pot * (qj * vqi + qi * vqj)

                dgr = dg / distance
                cross_i = -ge * dgr * (qj * vqi + qi * vqj)
                gpos_i += cross_i * sep
                gpos_j += -cross_i * sep

                dv = wp.dot(sep, vij)
                gq_i += ge * qj * half_s * dv
                gq_j = ge * qi * half_s * dv
                gq_i += ge * g_pot * vqj
                gq_j += ge * g_pot * vqi
                if CELL_GRAD:
                    gq_i += ge * qj * half_s * vquad
                    gq_j += ge * qi * half_s * vquad
                    gcell_acc += ge * wp.outer(n_vec, dgr * (qj * vqi + qi * vqj) * sep)
                wp.atomic_add(grad_charges, j, gq_j)

            wp.atomic_add(
                grad_positions,
                j,
                type(pos_i)(
                    type(pos_i[0])(gpos_j[0]),
                    type(pos_i[0])(gpos_j[1]),
                    type(pos_i[0])(gpos_j[2]),
                ),
            )

        return ddE_acc, gpos_i, gq_i, gcell_acc

    return _double_backward_pair


# === Axis validation shared by every order ===


def _validate_axes(
    wp_dtype: type,
    batched: bool,
    neighbor_input: str,
    deriv_state: _DerivState,
    cell_grad: bool,
    order: str,
    tiled: bool = False,
    cell_literal: bool = False,
) -> None:
    """Raise for unsupported / invalid component-axis combinations.

    ``NotImplementedError`` for not-yet / out-of-scope axes; ``ValueError`` for the
    permanently invalid ``cell_grad=True`` + ``deriv_state=E`` combination (no force
    terms to sum) and for derivative orders requesting ``deriv_state=E`` (nothing to
    differentiate).

    The ``tiled`` cooperative-block variant is matrix-only (the CSR loop has
    variable length per atom, so a fixed block stride does not apply) and is
    implemented for ``order in {"forward", "double_backward"}`` -- the launches
    that dominate the real-space cost. ``tiled`` with ``order="backward"`` raises
    (the chain's first backward is a Torch-side scale, no kernel).
    """
    _validate_common_axes(
        wp_dtype,
        deriv_state=deriv_state,
        cell_grad=cell_grad,
        order=order,
        component="ewald_real",
    )
    if neighbor_input not in ("list", "matrix"):
        raise NotImplementedError(
            f"ewald_real factory supports neighbor_input in ('list', 'matrix'); "
            f"got {neighbor_input!r}"
        )
    if tiled:
        if neighbor_input != "matrix":
            raise NotImplementedError(
                "ewald_real tiled kernels are matrix-only; "
                f"got neighbor_input={neighbor_input!r}"
            )
        if order not in ("forward", "double_backward"):
            raise NotImplementedError(
                "ewald_real tiled kernels support order in "
                f"('forward', 'double_backward'); got {order!r}"
            )
    if cell_literal:
        # The forward-fused literal dE/dcell output needs the per-pair force, so it
        # is restricted to order="forward" and a force-bearing deriv_state. Matrix
        # uses the cooperative tiled variant; CSR/list uses the serial row variant.
        if order != "forward":
            raise NotImplementedError(
                "ewald_real cell_literal is only implemented for order='forward'; "
                f"got order={order!r}"
            )
        if tiled and neighbor_input != "matrix":
            raise NotImplementedError(
                "ewald_real tiled cell_literal is matrix-only; got "
                f"neighbor_input={neighbor_input!r}"
            )
        if not tiled and neighbor_input != "list":
            raise NotImplementedError(
                "ewald_real non-tiled cell_literal is CSR/list-only; got "
                f"neighbor_input={neighbor_input!r}"
            )
        if deriv_state not in {_DerivState.E_F, _DerivState.E_F_dQ}:
            raise ValueError(
                "ewald_real cell_literal requires a force-bearing deriv_state "
                f"(E_F or E_F_dQ); got {deriv_state.name}"
            )


# === ewald_real kernel factory ===


@lru_cache(maxsize=None)
def make_ewald_real_kernel(
    wp_dtype: type,
    *,
    batched: bool = False,
    neighbor_input: str = "list",
    deriv_state: _DerivState = _DerivState.E,
    cell_grad: bool = False,
    order: str = "forward",
    tiled: bool = False,
    cell_literal: bool = False,
) -> wp.Kernel:
    """Return a cached, specialized ``ewald_real`` Warp kernel.

    The body branches on the Python compile-time constants ``BATCHED``,
    ``NEIGHBOR_MATRIX``, ``DERIV`` and ``CELL_GRAD``; Warp dead-eliminates the
    unused branches. The forward parameter list is shared across specializations; the
    backward / double-backward parameter lists are defined here (see module
    docstring) and are the sibling-factory pattern.

    Parameters
    ----------
    wp_dtype : type
        ``wp.float32`` or ``wp.float64``.
    batched : bool
        Single-system (``False``) vs batched (``True``). Batched reads the
        per-atom system index from ``batch_id``; single-system uses system 0.
    neighbor_input : {"list", "matrix"}
        CSR neighbor list (``"list"``) or neighbor matrix (``"matrix"``).
    deriv_state : _DerivState
        ``E`` (forward only), ``E_F`` (+forces), ``E_F_dQ`` (+charge gradient).
    cell_grad : bool
        Single compile-time switch for the optional virial / cell-gradient output
        (virial == ``-dE/dstrain``, accumulated from the same per-pair force terms
        as ``E_F``). Valid only with ``deriv_state`` in ``{E_F, E_F_dQ}``;
        ``cell_grad=True`` + ``deriv_state=E`` raises ``ValueError``.
    order : {"forward", "backward", "double_backward"}
        Forward output, first-derivative autograd node, or second-derivative node.
    tiled : bool
        Matrix-only cooperative-block variant. ``block_dim`` threads share each
        atom's neighbor-matrix row (strided loop) and reduce the per-atom
        accumulators with ``wp.tile_sum`` / ``wp.tile_atomic_add``. Must be
        launched with :func:`warp.launch_tiled` (``block_dim=REAL_SPACE_TILED_BLOCK_DIM``).
        The serial (``tiled=False``) kernel stays the bit-exact parity oracle and the
        per-pair ``@wp.func`` math is shared, so the tiled output agrees to round-off.
    """
    _validate_axes(
        wp_dtype,
        batched,
        neighbor_input,
        deriv_state,
        cell_grad,
        order,
        tiled,
        cell_literal,
    )

    if order == "forward":
        if tiled:
            if cell_literal:
                return _make_forward_kernel_tiled_cell_literal(
                    wp_dtype, batched, neighbor_input, deriv_state, cell_grad
                )
            return _make_forward_kernel_tiled(
                wp_dtype, batched, neighbor_input, deriv_state, cell_grad
            )
        if cell_literal:
            return _make_forward_kernel_cell_literal(
                wp_dtype, batched, neighbor_input, deriv_state, cell_grad
            )
        return _make_forward_kernel(
            wp_dtype, batched, neighbor_input, deriv_state, cell_grad
        )
    if order == "backward":
        return _make_backward_kernel(
            wp_dtype, batched, neighbor_input, deriv_state, cell_grad
        )
    if tiled:
        return _make_double_backward_kernel_tiled(
            wp_dtype, batched, neighbor_input, deriv_state, cell_grad
        )
    return _make_double_backward_kernel(
        wp_dtype, batched, neighbor_input, deriv_state, cell_grad
    )


def get_ewald_real_kernel(
    wp_dtype: type,
    *,
    batched: bool = False,
    neighbor_input: str = "list",
    deriv_state: _DerivState = _DerivState.E,
    cell_grad: bool = False,
    order: str = "forward",
    tiled: bool = False,
    cell_literal: bool = False,
    component: str = "ewald_real",
) -> wp.Kernel:
    """Return a cached ``ewald_real`` kernel, validating dtype + component.

    Validates the dtype and ``component`` (the only specialization axis not also a
    :func:`make_ewald_real_kernel` argument), then delegates to that factory.
    Memoization is the ``@lru_cache`` on ``make_*``; there is no separate cache
    dict. Every argument is forwarded **by keyword** (including ``wp_dtype``) so a
    positional vs keyword call can never produce a duplicate ``@lru_cache`` entry.
    """
    _require_supported_dtype(wp_dtype)
    _require_component(component, "ewald_real")
    return make_ewald_real_kernel(
        wp_dtype=wp_dtype,
        batched=batched,
        neighbor_input=neighbor_input,
        deriv_state=deriv_state,
        cell_grad=cell_grad,
        order=order,
        tiled=tiled,
        cell_literal=cell_literal,
    )


# === order="forward" builder ===


def _make_forward_kernel(
    wp_dtype: type,
    batched: bool,
    neighbor_input: str,
    deriv_state: _DerivState,
    cell_grad: bool,
) -> wp.Kernel:
    """Build the forward kernel: energy (+forces +charge-grad +virial)."""
    info = _DTYPE_INFO[wp_dtype]
    vec_dtype = info.vec
    mat_dtype = info.mat

    BATCHED = bool(batched)
    NEIGHBOR_MATRIX = neighbor_input == "matrix"
    HAS_FORCE = deriv_state in {_DerivState.E_F, _DerivState.E_F_dQ}
    HAS_CHARGE = deriv_state in {_DerivState.E_dQ, _DerivState.E_F_dQ}
    CELL_GRAD = bool(cell_grad)

    module_name = _ewald_real_module_name(wp_dtype, BATCHED, neighbor_input, "forward")
    accumulate_pair = _make_forward_pair_fn(
        wp_dtype, deriv_state=deriv_state, cell_grad=cell_grad
    )

    @wp.kernel(module=module_name)
    def _ewald_real(
        positions: wp.array(dtype=vec_dtype),
        charges: wp.array(dtype=wp_dtype),
        cell: wp.array(dtype=mat_dtype),
        batch_id: wp.array(dtype=wp.int32),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        neighbor_matrix: wp.array2d(dtype=wp.int32),
        unit_shifts_matrix: wp.array2d(dtype=wp.vec3i),
        mask_value: wp.int32,
        alpha: wp.array(dtype=wp_dtype),
        pair_energies: wp.array(dtype=wp.float64),
        atomic_forces: wp.array(dtype=vec_dtype),
        charge_gradients: wp.array(dtype=wp.float64),
        virial: wp.array(dtype=mat_dtype),
    ) -> None:
        """Real-space Ewald forward kernel (energy + optional derivatives).

        One thread per atom; loops over its neighbors (CSR or matrix). Energy is
        always accumulated in float64; forces / charge gradients / virial are
        accumulated only for the active ``DERIV`` / ``CELL_GRAD`` branches. The
        per-pair math is verbatim from the hand-written kernels, so the outputs are
        bit-exact parity oracles.
        """
        atom_i = wp.tid()

        qi = wp.float64(charges[atom_i])
        pos_i = positions[atom_i]
        if BATCHED:
            isys = batch_id[atom_i]
        else:
            isys = wp.int32(0)
        alpha_ = wp.float64(alpha[isys])
        cell_t = wp.transpose(cell[isys])

        energy_acc = wp.float64(0.0)
        force_i_acc = type(pos_i)(
            type(pos_i[0])(0.0), type(pos_i[0])(0.0), type(pos_i[0])(0.0)
        )
        cg_i_acc = wp.float64(0.0)
        virial_acc = wp.mat33d()

        if NEIGHBOR_MATRIX:
            max_neighbors = neighbor_matrix.shape[1]
            for neighbor_idx in range(max_neighbors):
                j = neighbor_matrix[atom_i, neighbor_idx]
                if j == mask_value:
                    continue

                pos_j = positions[j]
                shift_vec = unit_shifts_matrix[atom_i, neighbor_idx]
                energy_acc, force_i_acc, cg_i_acc, virial_acc = accumulate_pair(
                    j,
                    qi,
                    pos_i,
                    pos_j,
                    shift_vec,
                    cell_t,
                    alpha_,
                    energy_acc,
                    force_i_acc,
                    cg_i_acc,
                    virial_acc,
                    charges,
                    atomic_forces,
                    charge_gradients,
                )
        else:
            j_range_start = neighbor_ptr[atom_i]
            j_range_end = neighbor_ptr[atom_i + 1]
            for edge_idx in range(j_range_start, j_range_end):
                j = idx_j[edge_idx]

                pos_j = positions[j]
                shift_vec = unit_shifts[edge_idx]
                energy_acc, force_i_acc, cg_i_acc, virial_acc = accumulate_pair(
                    j,
                    qi,
                    pos_i,
                    pos_j,
                    shift_vec,
                    cell_t,
                    alpha_,
                    energy_acc,
                    force_i_acc,
                    cg_i_acc,
                    virial_acc,
                    charges,
                    atomic_forces,
                    charge_gradients,
                )

        wp.atomic_add(pair_energies, atom_i, energy_acc)
        if HAS_FORCE:
            wp.atomic_add(atomic_forces, atom_i, force_i_acc)
            if CELL_GRAD:
                wp.atomic_add(virial, isys, type(cell_t)(virial_acc))
        if HAS_CHARGE:
            wp.atomic_add(charge_gradients, atom_i, cg_i_acc)

    _name_and_document(
        _ewald_real,
        base="ewald_real_forward",
        wp_dtype=wp_dtype,
        batched=batched,
        neighbor_input=neighbor_input,
        deriv_state=deriv_state,
        cell_grad=cell_grad,
        order="forward",
    )
    return _ewald_real


# === order="forward" CSR/list cell_literal builder (extra dE/dcell out) ===


def _make_forward_kernel_cell_literal(
    wp_dtype: type,
    batched: bool,
    neighbor_input: str,
    deriv_state: _DerivState,
    cell_grad: bool,
) -> wp.Kernel:
    """Build the CSR/list forward kernel that also emits literal ``dE/dcell``.

    One thread owns one atom's CSR row, accumulates the per-atom literal cell
    gradient block in registers, and writes one ``dedcell_atom[i]`` output. No
    edge-sized intermediate is materialized.
    """
    info = _DTYPE_INFO[wp_dtype]
    vec_dtype = info.vec
    mat_dtype = info.mat

    BATCHED = bool(batched)
    HAS_FORCE = deriv_state in {_DerivState.E_F, _DerivState.E_F_dQ}
    HAS_CHARGE = deriv_state in {_DerivState.E_dQ, _DerivState.E_F_dQ}
    CELL_GRAD = bool(cell_grad)

    module_name = _ewald_real_module_name(
        wp_dtype, BATCHED, neighbor_input, "forward", cell_literal=True
    )
    accumulate_pair = _make_forward_pair_fn(
        wp_dtype, deriv_state=deriv_state, cell_grad=cell_grad, cell_literal=True
    )

    @wp.kernel(module=module_name)
    def _ewald_real_cell_literal(
        positions: wp.array(dtype=vec_dtype),
        charges: wp.array(dtype=wp_dtype),
        cell: wp.array(dtype=mat_dtype),
        batch_id: wp.array(dtype=wp.int32),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        neighbor_matrix: wp.array2d(dtype=wp.int32),
        unit_shifts_matrix: wp.array2d(dtype=wp.vec3i),
        mask_value: wp.int32,
        alpha: wp.array(dtype=wp_dtype),
        pair_energies: wp.array(dtype=wp.float64),
        atomic_forces: wp.array(dtype=vec_dtype),
        charge_gradients: wp.array(dtype=wp.float64),
        virial: wp.array(dtype=mat_dtype),
        dedcell_atom: wp.array(dtype=wp.mat33d),
    ) -> None:
        """Real-space Ewald CSR/list forward kernel + literal ``dE/dcell``."""
        atom_i = wp.tid()

        qi = wp.float64(charges[atom_i])
        pos_i = positions[atom_i]
        if BATCHED:
            isys = batch_id[atom_i]
        else:
            isys = wp.int32(0)
        alpha_ = wp.float64(alpha[isys])
        cell_t = wp.transpose(cell[isys])

        energy_acc = wp.float64(0.0)
        force_i_acc = type(pos_i)(
            type(pos_i[0])(0.0), type(pos_i[0])(0.0), type(pos_i[0])(0.0)
        )
        cg_i_acc = wp.float64(0.0)
        virial_acc = wp.mat33d()
        dedcell_acc = wp.mat33d()

        j_range_start = neighbor_ptr[atom_i]
        j_range_end = neighbor_ptr[atom_i + 1]
        for edge_idx in range(j_range_start, j_range_end):
            j = idx_j[edge_idx]
            pos_j = positions[j]
            shift_vec = unit_shifts[edge_idx]
            (
                energy_acc,
                force_i_acc,
                cg_i_acc,
                virial_acc,
                dedcell_acc,
            ) = accumulate_pair(
                j,
                qi,
                pos_i,
                pos_j,
                shift_vec,
                cell_t,
                alpha_,
                energy_acc,
                force_i_acc,
                cg_i_acc,
                virial_acc,
                dedcell_acc,
                charges,
                atomic_forces,
                charge_gradients,
            )

        wp.atomic_add(pair_energies, atom_i, energy_acc)
        if HAS_FORCE:
            wp.atomic_add(atomic_forces, atom_i, force_i_acc)
            wp.atomic_add(dedcell_atom, atom_i, dedcell_acc)
            if CELL_GRAD:
                wp.atomic_add(virial, isys, type(cell_t)(virial_acc))
        if HAS_CHARGE:
            wp.atomic_add(charge_gradients, atom_i, cg_i_acc)

    _name_and_document(
        _ewald_real_cell_literal,
        base="ewald_real_forward",
        wp_dtype=wp_dtype,
        batched=batched,
        neighbor_input=neighbor_input,
        deriv_state=deriv_state,
        cell_grad=cell_grad,
        order="forward",
        cell_literal=True,
    )
    return _ewald_real_cell_literal


# === order="forward" tiled builder (matrix-only, cooperative block) ===


def _make_forward_kernel_tiled(
    wp_dtype: type,
    batched: bool,
    neighbor_input: str,
    deriv_state: _DerivState,
    cell_grad: bool,
) -> wp.Kernel:
    """Build the cooperative-block forward kernel for the neighbor-matrix layout.

    Identical per-pair math to :func:`_make_forward_kernel` (the same
    ``_make_forward_pair_fn`` ``@wp.func`` core, so neighbor-``j`` reaction terms
    are emitted per pair exactly as the serial kernel), but ``block_dim`` threads
    cooperate on each atom: ``(atom_i, lane) = wp.tid()`` and lane ``l`` walks the
    neighbor-matrix row at stride ``block_dim``. The per-atom ``i`` accumulators are
    block-reduced via ``wp.tile_sum`` and written once by lane 0 with
    ``wp.tile_atomic_add``; the per-system virial is reduced the same way and added
    at ``isys``. Launch with :func:`warp.launch_tiled`
    (``block_dim=REAL_SPACE_TILED_BLOCK_DIM``). On CPU ``block_dim`` clamps to 1, so
    the strided loop visits every neighbor and the tile reductions degrade to scalar
    passthrough -- numerically identical to the serial kernel.

    Matrix-only: the CSR row length varies per atom, so a fixed block stride does
    not apply (``_validate_axes`` rejects ``tiled`` + ``neighbor_input="list"``).
    """
    info = _DTYPE_INFO[wp_dtype]
    vec_dtype = info.vec
    mat_dtype = info.mat

    BATCHED = bool(batched)
    HAS_FORCE = deriv_state in {_DerivState.E_F, _DerivState.E_F_dQ}
    HAS_CHARGE = deriv_state in {_DerivState.E_dQ, _DerivState.E_F_dQ}
    CELL_GRAD = bool(cell_grad)

    module_name = _ewald_real_module_name(
        wp_dtype, BATCHED, neighbor_input, "forward", tiled=True
    )
    accumulate_pair = _make_forward_pair_fn(
        wp_dtype, deriv_state=deriv_state, cell_grad=cell_grad
    )

    @wp.kernel(module=module_name)
    def _ewald_real_tiled(
        positions: wp.array(dtype=vec_dtype),
        charges: wp.array(dtype=wp_dtype),
        cell: wp.array(dtype=mat_dtype),
        batch_id: wp.array(dtype=wp.int32),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        neighbor_matrix: wp.array2d(dtype=wp.int32),
        unit_shifts_matrix: wp.array2d(dtype=wp.vec3i),
        mask_value: wp.int32,
        alpha: wp.array(dtype=wp_dtype),
        pair_energies: wp.array(dtype=wp.float64),
        atomic_forces: wp.array(dtype=vec_dtype),
        charge_gradients: wp.array(dtype=wp.float64),
        virial: wp.array(dtype=mat_dtype),
    ) -> None:
        """Real-space Ewald forward kernel, cooperative-block (neighbor matrix)."""
        atom_i, lane = wp.tid()
        block_size = wp.block_dim()

        qi = wp.float64(charges[atom_i])
        pos_i = positions[atom_i]
        if BATCHED:
            isys = batch_id[atom_i]
        else:
            isys = wp.int32(0)
        alpha_ = wp.float64(alpha[isys])
        cell_t = wp.transpose(cell[isys])

        energy_acc = wp.float64(0.0)
        force_i_acc = type(pos_i)(
            type(pos_i[0])(0.0), type(pos_i[0])(0.0), type(pos_i[0])(0.0)
        )
        cg_i_acc = wp.float64(0.0)
        virial_acc = wp.mat33d()

        max_neighbors = neighbor_matrix.shape[1]
        k = lane
        while k < max_neighbors:
            j = neighbor_matrix[atom_i, k]
            if j != mask_value:
                pos_j = positions[j]
                shift_vec = unit_shifts_matrix[atom_i, k]
                energy_acc, force_i_acc, cg_i_acc, virial_acc = accumulate_pair(
                    j,
                    qi,
                    pos_i,
                    pos_j,
                    shift_vec,
                    cell_t,
                    alpha_,
                    energy_acc,
                    force_i_acc,
                    cg_i_acc,
                    virial_acc,
                    charges,
                    atomic_forces,
                    charge_gradients,
                )
            k += block_size

        # Cooperative block reductions. The guards are Python compile-time
        # constants, so every lane takes the same branch (required for the tile
        # collectives) and unused reductions are dead-eliminated.
        energy_sum = wp.tile_sum(wp.tile(energy_acc))
        if HAS_FORCE:
            force_sum = wp.tile_sum(wp.tile(force_i_acc, preserve_type=True))
            if CELL_GRAD:
                virial_sum = wp.tile_sum(wp.tile(virial_acc, preserve_type=True))
        if HAS_CHARGE:
            cg_sum = wp.tile_sum(wp.tile(cg_i_acc))

        if lane == 0:
            wp.tile_atomic_add(pair_energies, energy_sum, offset=(atom_i,))
            if HAS_FORCE:
                wp.tile_atomic_add(atomic_forces, force_sum, offset=(atom_i,))
                if CELL_GRAD:
                    wp.atomic_add(
                        virial, isys, type(cell_t)(wp.tile_extract(virial_sum, 0))
                    )
            if HAS_CHARGE:
                wp.tile_atomic_add(charge_gradients, cg_sum, offset=(atom_i,))

    _name_and_document(
        _ewald_real_tiled,
        base="ewald_real_forward",
        wp_dtype=wp_dtype,
        batched=batched,
        neighbor_input=neighbor_input,
        deriv_state=deriv_state,
        cell_grad=cell_grad,
        order="forward",
        tiled=True,
    )
    return _ewald_real_tiled


# === order="forward" tiled cell_literal builder (matrix-only, extra dE/dcell out) ===


def _make_forward_kernel_tiled_cell_literal(
    wp_dtype: type,
    batched: bool,
    neighbor_input: str,
    deriv_state: _DerivState,
    cell_grad: bool,
) -> wp.Kernel:
    """Cooperative-block forward kernel that also emits the literal ``dE/dcell``.

    Identical to :func:`_make_forward_kernel_tiled` plus one extra output:
    ``dedcell_atom`` (``wp.array(dtype=wp.mat33d)``, shape ``(N,)``), the per-atom
    literal cell-gradient block ``sum_neighbors n (x) (-force)``. The block is
    block-reduced (``wp.tile_sum``, ``preserve_type=True``) and written once per
    atom with ``wp.atomic_add`` (per-atom index -> contention-free, the same write
    pattern the mat33 virial uses). The torch autograd chain caches this so its
    first cell backward is a pure scatter with no separate kernel launch. This is a
    distinct compile-time variant (``cell_literal=True``); every other launch keeps
    the byte-identical 15-arg forward kernel.
    """
    info = _DTYPE_INFO[wp_dtype]
    vec_dtype = info.vec
    mat_dtype = info.mat

    BATCHED = bool(batched)
    HAS_FORCE = deriv_state in {_DerivState.E_F, _DerivState.E_F_dQ}
    HAS_CHARGE = deriv_state in {_DerivState.E_dQ, _DerivState.E_F_dQ}
    CELL_GRAD = bool(cell_grad)

    module_name = _ewald_real_module_name(
        wp_dtype, BATCHED, neighbor_input, "forward", tiled=True, cell_literal=True
    )
    accumulate_pair = _make_forward_pair_fn(
        wp_dtype, deriv_state=deriv_state, cell_grad=cell_grad, cell_literal=True
    )

    @wp.kernel(module=module_name)
    def _ewald_real_tiled_cell_literal(
        positions: wp.array(dtype=vec_dtype),
        charges: wp.array(dtype=wp_dtype),
        cell: wp.array(dtype=mat_dtype),
        batch_id: wp.array(dtype=wp.int32),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        neighbor_matrix: wp.array2d(dtype=wp.int32),
        unit_shifts_matrix: wp.array2d(dtype=wp.vec3i),
        mask_value: wp.int32,
        alpha: wp.array(dtype=wp_dtype),
        pair_energies: wp.array(dtype=wp.float64),
        atomic_forces: wp.array(dtype=vec_dtype),
        charge_gradients: wp.array(dtype=wp.float64),
        virial: wp.array(dtype=mat_dtype),
        dedcell_atom: wp.array(dtype=wp.mat33d),
    ) -> None:
        """Real-space Ewald forward kernel, cooperative-block + literal dE/dcell."""
        atom_i, lane = wp.tid()
        block_size = wp.block_dim()

        qi = wp.float64(charges[atom_i])
        pos_i = positions[atom_i]
        if BATCHED:
            isys = batch_id[atom_i]
        else:
            isys = wp.int32(0)
        alpha_ = wp.float64(alpha[isys])
        cell_t = wp.transpose(cell[isys])

        energy_acc = wp.float64(0.0)
        force_i_acc = type(pos_i)(
            type(pos_i[0])(0.0), type(pos_i[0])(0.0), type(pos_i[0])(0.0)
        )
        cg_i_acc = wp.float64(0.0)
        virial_acc = wp.mat33d()
        dedcell_acc = wp.mat33d()

        max_neighbors = neighbor_matrix.shape[1]
        k = lane
        while k < max_neighbors:
            j = neighbor_matrix[atom_i, k]
            if j != mask_value:
                pos_j = positions[j]
                shift_vec = unit_shifts_matrix[atom_i, k]
                (
                    energy_acc,
                    force_i_acc,
                    cg_i_acc,
                    virial_acc,
                    dedcell_acc,
                ) = accumulate_pair(
                    j,
                    qi,
                    pos_i,
                    pos_j,
                    shift_vec,
                    cell_t,
                    alpha_,
                    energy_acc,
                    force_i_acc,
                    cg_i_acc,
                    virial_acc,
                    dedcell_acc,
                    charges,
                    atomic_forces,
                    charge_gradients,
                )
            k += block_size

        energy_sum = wp.tile_sum(wp.tile(energy_acc))
        if HAS_FORCE:
            force_sum = wp.tile_sum(wp.tile(force_i_acc, preserve_type=True))
            dedcell_sum = wp.tile_sum(wp.tile(dedcell_acc, preserve_type=True))
            if CELL_GRAD:
                virial_sum = wp.tile_sum(wp.tile(virial_acc, preserve_type=True))
        if HAS_CHARGE:
            cg_sum = wp.tile_sum(wp.tile(cg_i_acc))

        if lane == 0:
            wp.tile_atomic_add(pair_energies, energy_sum, offset=(atom_i,))
            if HAS_FORCE:
                wp.tile_atomic_add(atomic_forces, force_sum, offset=(atom_i,))
                # Per-atom mat33d write: one owner per atom_i, so plain atomic_add
                # is contention-free (mirrors the per-system virial write pattern).
                wp.atomic_add(dedcell_atom, atom_i, wp.tile_extract(dedcell_sum, 0))
                if CELL_GRAD:
                    wp.atomic_add(
                        virial, isys, type(cell_t)(wp.tile_extract(virial_sum, 0))
                    )
            if HAS_CHARGE:
                wp.tile_atomic_add(charge_gradients, cg_sum, offset=(atom_i,))

    _name_and_document(
        _ewald_real_tiled_cell_literal,
        base="ewald_real_forward",
        wp_dtype=wp_dtype,
        batched=batched,
        neighbor_input=neighbor_input,
        deriv_state=deriv_state,
        cell_grad=cell_grad,
        order="forward",
        tiled=True,
        cell_literal=True,
    )
    return _ewald_real_tiled_cell_literal


# === order="backward" builder ===


def _make_backward_kernel(
    wp_dtype: type,
    batched: bool,
    neighbor_input: str,
    deriv_state: _DerivState,
    cell_grad: bool,
) -> wp.Kernel:
    """Build the first-derivative autograd node.

    Same per-pair core as forward ``E_F`` (so it is bit-consistent), but every
    accumulated quantity is scaled by the per-system upstream energy cotangent
    ``grad_energy[isys]`` and written as a *gradient* (``dL/dR`` etc.):

    * ``atomic_forces`` receives ``dL/dR = grad_energy * dE/dR`` (note ``dE/dR``,
      i.e. the negative of the physical force ``-dE/dR``);
    * ``charge_gradients`` receives ``dL/dq = grad_energy * dE/dq``;
    * ``virial`` (cell_grad) receives ``grad_energy * (sum r (x) F)`` -- the same
      ``W`` the forward virial accumulates, scaled by ``grad_energy``; the autograd
      connector routes this as the cell-gradient state.

    Signature = ``grad_energy`` prepended to the frozen forward 15-param list;
    ``pair_energies`` is an unused (sentinel) slot here.

    Sign convention: physical pair force is ``F = fm*sep`` (on j: ``+F``, on i:
    ``-F``); ``dE/dR_i = +F``, ``dE/dR_j = -F``. The signs are pinned by the
    finite-difference test, not asserted by fiat.
    """
    info = _DTYPE_INFO[wp_dtype]
    vec_dtype = info.vec
    mat_dtype = info.mat

    BATCHED = bool(batched)
    NEIGHBOR_MATRIX = neighbor_input == "matrix"
    HAS_FORCE = deriv_state in {_DerivState.E_F, _DerivState.E_F_dQ}
    HAS_CHARGE = deriv_state in {_DerivState.E_dQ, _DerivState.E_F_dQ}
    CELL_GRAD = bool(cell_grad)

    module_name = _ewald_real_module_name(wp_dtype, BATCHED, neighbor_input, "backward")
    accumulate_pair = _make_backward_pair_fn(
        wp_dtype, deriv_state=deriv_state, cell_grad=cell_grad
    )

    @wp.kernel(module=module_name)
    def _ewald_real_backward(
        grad_energy: wp.array(dtype=wp.float64),
        positions: wp.array(dtype=vec_dtype),
        charges: wp.array(dtype=wp_dtype),
        cell: wp.array(dtype=mat_dtype),
        batch_id: wp.array(dtype=wp.int32),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        neighbor_matrix: wp.array2d(dtype=wp.int32),
        unit_shifts_matrix: wp.array2d(dtype=wp.vec3i),
        mask_value: wp.int32,
        alpha: wp.array(dtype=wp_dtype),
        pair_energies: wp.array(dtype=wp.float64),
        atomic_forces: wp.array(dtype=vec_dtype),
        charge_gradients: wp.array(dtype=wp.float64),
        virial: wp.array(dtype=mat_dtype),
    ) -> None:
        """Real-space Ewald first-derivative node (forward core scaled by grad_E)."""
        atom_i = wp.tid()

        qi = wp.float64(charges[atom_i])
        pos_i = positions[atom_i]
        if BATCHED:
            isys = batch_id[atom_i]
        else:
            isys = wp.int32(0)
        alpha_ = wp.float64(alpha[isys])
        cell_t = wp.transpose(cell[isys])
        ge = grad_energy[isys]

        # dL/dR_i (= grad_energy * dE/dR_i). dE/dR_i = +F summed over neighbors.
        gpos_i = type(pos_i)(
            type(pos_i[0])(0.0), type(pos_i[0])(0.0), type(pos_i[0])(0.0)
        )
        cg_i_acc = wp.float64(0.0)
        virial_acc = wp.mat33d()

        if NEIGHBOR_MATRIX:
            max_neighbors = neighbor_matrix.shape[1]
            for neighbor_idx in range(max_neighbors):
                j = neighbor_matrix[atom_i, neighbor_idx]
                if j == mask_value:
                    continue
                pos_j = positions[j]
                shift_vec = unit_shifts_matrix[atom_i, neighbor_idx]
                gpos_i, cg_i_acc, virial_acc = accumulate_pair(
                    j,
                    qi,
                    pos_i,
                    pos_j,
                    shift_vec,
                    cell_t,
                    alpha_,
                    ge,
                    gpos_i,
                    cg_i_acc,
                    virial_acc,
                    charges,
                    atomic_forces,
                    charge_gradients,
                )
        else:
            j_range_start = neighbor_ptr[atom_i]
            j_range_end = neighbor_ptr[atom_i + 1]
            for edge_idx in range(j_range_start, j_range_end):
                j = idx_j[edge_idx]
                pos_j = positions[j]
                shift_vec = unit_shifts[edge_idx]
                gpos_i, cg_i_acc, virial_acc = accumulate_pair(
                    j,
                    qi,
                    pos_i,
                    pos_j,
                    shift_vec,
                    cell_t,
                    alpha_,
                    ge,
                    gpos_i,
                    cg_i_acc,
                    virial_acc,
                    charges,
                    atomic_forces,
                    charge_gradients,
                )

        if HAS_FORCE:
            wp.atomic_add(atomic_forces, atom_i, gpos_i)
        if HAS_CHARGE:
            wp.atomic_add(charge_gradients, atom_i, cg_i_acc)
        if CELL_GRAD:
            wp.atomic_add(virial, isys, type(cell_t)(virial_acc))

    _name_and_document(
        _ewald_real_backward,
        base="ewald_real_backward",
        wp_dtype=wp_dtype,
        batched=batched,
        neighbor_input=neighbor_input,
        deriv_state=deriv_state,
        cell_grad=cell_grad,
        order="backward",
    )
    return _ewald_real_backward


# === order="double_backward" builder ===


def _make_double_backward_kernel(
    wp_dtype: type,
    batched: bool,
    neighbor_input: str,
    deriv_state: _DerivState,
    cell_grad: bool,
) -> wp.Kernel:
    """Build the second-derivative (pair-Hessian / HVP) node, recompute mode.

    Backward emitted ``grad_R_i = ge * dE/dR_i`` and ``grad_q_i = ge * dE/dq_i``.
    The loss seen by the chain is ``L = sum_i v_pos_i . grad_R_i + v_charge_i *
    grad_q_i``. This kernel forms ``dL/d(grad_energy)``, ``dL/dR`` and ``dL/dq`` by
    contracting the pair Hessian with the cotangents ``(v_pos, v_charge)``,
    recomputing per-pair erfc/force terms from the neighbor data.

    Per pair ``(i, j)`` with ``d = sep``, ``r = |d|`` (let ``vij = v_pos_i -
    v_pos_j``):

    * ``fm = qi qj * half_S``, ``J = fm I + (qi qj half_dS / r) (d (x) d)`` is the
      position pair-Hessian block; ``dL/dR_i += ge J (v_pos_j - v_pos_i)``,
      ``dL/dR_j += ge J (v_pos_i - v_pos_j)``.
    * charge<->position cross term uses ``half_S`` (force) and ``g'`` (potential):
      ``dL/dR_i += ge g'(r) (d/r) (qj v_charge_i + qi v_charge_j)`` (with the
      i/j sign from ``d = pos_j - pos_i``); the symmetric force-cross feeds
      ``dL/dq``.
    * charge self second-derivative: ``dL/dq_i += ge g(r) v_charge_j``,
      ``dL/dq_j += ge g(r) v_charge_i``.

    Cell second-order: cell enters only through the periodic separation
    ``sep = pos_j - pos_i + cell_t @ n`` (integer shift ``n``), so
    ``d sep_a / d cell[p, q] = delta_{a, q} n_p`` and ``d Phi / d cell = n (x) d Phi / d sep``.
    With ``M = v_cell[s]`` and ``vquad = sep^T M sep`` the chain's ``v_cell`` term is
    ``L_cell = sum ge fm vquad`` (``fm = qi qj halfS``). It contributes
    ``fm vquad`` to ``grad_grad_energy``; the force<->cell cross
    ``-ge [coeff vquad sep + fm (M+M^T) sep]`` (``coeff = qi qj halfdS / r``) into
    ``grad_positions`` (negated on ``j``); and ``ge qj halfS vquad`` /
    ``ge qi halfS vquad`` into ``grad_charges``. ``grad_cell`` accumulates
    ``ge (n (x) grad_sep)`` where ``grad_sep`` collects the cell-self piece (from
    ``L_cell``), the cell<->position piece (from the ``v_pos`` loss term) and the
    cell<->charge piece (from the ``v_charge`` loss term).

    NB: this is the directional derivative of the backward **cell output**, which is
    the strain-virial state ``W = sum sep (x) F = -dE/dstrain`` (see ``test_virial_fd``),
    NOT the literal ``dE/dcell = -sum n (x) F``. So ``grad_cell`` is the second
    derivative of that cell-grad (strain-virial) state w.r.t. the cell input, which is
    exactly the contract "double_backward == directional derivative of backward".
    The public autograd layer must route ``v_cell`` as the cotangent on that same
    W-shaped state.

    Outputs (zero-initialized by caller): ``grad_grad_energy`` (per-system),
    ``grad_positions``, ``grad_charges``, ``grad_cell`` (per-system).
    """
    info = _DTYPE_INFO[wp_dtype]
    vec_dtype = info.vec
    mat_dtype = info.mat

    BATCHED = bool(batched)
    NEIGHBOR_MATRIX = neighbor_input == "matrix"
    HAS_CHARGE = deriv_state in {_DerivState.E_dQ, _DerivState.E_F_dQ}
    CELL_GRAD = bool(cell_grad)

    module_name = _ewald_real_module_name(
        wp_dtype, BATCHED, neighbor_input, "double_backward"
    )
    accumulate_pair = _make_double_backward_pair_fn(
        wp_dtype, deriv_state=deriv_state, cell_grad=cell_grad
    )

    @wp.kernel(module=module_name)
    def _ewald_real_double_backward(
        v_pos: wp.array(dtype=vec_dtype),
        v_charge: wp.array(dtype=wp.float64),
        v_cell: wp.array(dtype=mat_dtype),
        grad_energy: wp.array(dtype=wp.float64),
        positions: wp.array(dtype=vec_dtype),
        charges: wp.array(dtype=wp_dtype),
        cell: wp.array(dtype=mat_dtype),
        batch_id: wp.array(dtype=wp.int32),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        neighbor_matrix: wp.array2d(dtype=wp.int32),
        unit_shifts_matrix: wp.array2d(dtype=wp.vec3i),
        mask_value: wp.int32,
        alpha: wp.array(dtype=wp_dtype),
        grad_grad_energy: wp.array(dtype=wp.float64),
        grad_positions: wp.array(dtype=vec_dtype),
        grad_charges: wp.array(dtype=wp.float64),
        grad_cell: wp.array(dtype=mat_dtype),
    ) -> None:
        """Second-derivative node: pair Hessian contracted with backward cotangents."""
        atom_i = wp.tid()

        qi = wp.float64(charges[atom_i])
        pos_i = positions[atom_i]
        if BATCHED:
            isys = batch_id[atom_i]
        else:
            isys = wp.int32(0)
        alpha_ = wp.float64(alpha[isys])
        cell_t = wp.transpose(cell[isys])
        ge = grad_energy[isys]
        vi = wp.vec3d(
            wp.float64(v_pos[atom_i][0]),
            wp.float64(v_pos[atom_i][1]),
            wp.float64(v_pos[atom_i][2]),
        )
        vqi = wp.float64(0.0)
        if HAS_CHARGE:
            vqi = v_charge[atom_i]

        # v_cell cotangent of this atom's system (symmetrized once: M + M^T).
        m_cell = wp.mat33d()
        if CELL_GRAD:
            m = v_cell[isys]
            m_cell = wp.mat33d(
                wp.float64(m[0, 0]),
                wp.float64(m[0, 1]),
                wp.float64(m[0, 2]),
                wp.float64(m[1, 0]),
                wp.float64(m[1, 1]),
                wp.float64(m[1, 2]),
                wp.float64(m[2, 0]),
                wp.float64(m[2, 1]),
                wp.float64(m[2, 2]),
            )
        m_sym = m_cell + wp.transpose(m_cell)

        ddE_acc = wp.float64(0.0)  # contribution to grad_grad_energy[isys]
        gpos_i = wp.vec3d(0.0, 0.0, 0.0)  # dL/dR_i
        gq_i = wp.float64(0.0)  # dL/dq_i
        gcell_acc = wp.mat33d()  # dL/dcell[isys] contribution

        if NEIGHBOR_MATRIX:
            max_neighbors = neighbor_matrix.shape[1]
            for neighbor_idx in range(max_neighbors):
                j = neighbor_matrix[atom_i, neighbor_idx]
                if j == mask_value:
                    continue
                pos_j = positions[j]
                shift_vec = unit_shifts_matrix[atom_i, neighbor_idx]
                ddE_acc, gpos_i, gq_i, gcell_acc = accumulate_pair(
                    j,
                    qi,
                    pos_i,
                    pos_j,
                    shift_vec,
                    cell_t,
                    alpha_,
                    ge,
                    vi,
                    vqi,
                    m_cell,
                    m_sym,
                    ddE_acc,
                    gpos_i,
                    gq_i,
                    gcell_acc,
                    charges,
                    v_pos,
                    v_charge,
                    grad_positions,
                    grad_charges,
                )
        else:
            j_range_start = neighbor_ptr[atom_i]
            j_range_end = neighbor_ptr[atom_i + 1]
            for edge_idx in range(j_range_start, j_range_end):
                j = idx_j[edge_idx]
                pos_j = positions[j]
                shift_vec = unit_shifts[edge_idx]
                ddE_acc, gpos_i, gq_i, gcell_acc = accumulate_pair(
                    j,
                    qi,
                    pos_i,
                    pos_j,
                    shift_vec,
                    cell_t,
                    alpha_,
                    ge,
                    vi,
                    vqi,
                    m_cell,
                    m_sym,
                    ddE_acc,
                    gpos_i,
                    gq_i,
                    gcell_acc,
                    charges,
                    v_pos,
                    v_charge,
                    grad_positions,
                    grad_charges,
                )

        wp.atomic_add(grad_grad_energy, isys, ddE_acc)
        wp.atomic_add(
            grad_positions,
            atom_i,
            type(pos_i)(
                type(pos_i[0])(gpos_i[0]),
                type(pos_i[0])(gpos_i[1]),
                type(pos_i[0])(gpos_i[2]),
            ),
        )
        if HAS_CHARGE:
            wp.atomic_add(grad_charges, atom_i, gq_i)
        if CELL_GRAD:
            wp.atomic_add(grad_cell, isys, type(cell_t)(gcell_acc))

    _name_and_document(
        _ewald_real_double_backward,
        base="ewald_real_double_backward",
        wp_dtype=wp_dtype,
        batched=batched,
        neighbor_input=neighbor_input,
        deriv_state=deriv_state,
        cell_grad=cell_grad,
        order="double_backward",
    )
    return _ewald_real_double_backward


# === order="double_backward" tiled builder (matrix-only, cooperative block) ===


def _make_double_backward_kernel_tiled(
    wp_dtype: type,
    batched: bool,
    neighbor_input: str,
    deriv_state: _DerivState,
    cell_grad: bool,
) -> wp.Kernel:
    """Cooperative-block second-derivative kernel for the neighbor-matrix layout.

    Same pair-Hessian / HVP math as :func:`_make_double_backward_kernel` (shared
    ``_make_double_backward_pair_fn`` core, so neighbor-``j`` terms are emitted per
    pair exactly as the serial kernel), but ``block_dim`` threads share each atom's
    neighbor-matrix row (strided loop). The per-atom (``gpos_i``, ``gq_i``) and
    per-system (``ddE_acc``, ``gcell_acc``) accumulators are block-reduced with
    ``wp.tile_sum`` and written once by lane 0. Launch with
    :func:`warp.launch_tiled` (``block_dim=REAL_SPACE_TILED_BLOCK_DIM``); CPU clamps
    ``block_dim`` to 1 and degrades to the serial result.
    """
    info = _DTYPE_INFO[wp_dtype]
    vec_dtype = info.vec
    mat_dtype = info.mat

    BATCHED = bool(batched)
    HAS_CHARGE = deriv_state in {_DerivState.E_dQ, _DerivState.E_F_dQ}
    CELL_GRAD = bool(cell_grad)

    module_name = _ewald_real_module_name(
        wp_dtype, BATCHED, neighbor_input, "double_backward", tiled=True
    )
    accumulate_pair = _make_double_backward_pair_fn(
        wp_dtype, deriv_state=deriv_state, cell_grad=cell_grad
    )

    @wp.kernel(module=module_name)
    def _ewald_real_double_backward_tiled(
        v_pos: wp.array(dtype=vec_dtype),
        v_charge: wp.array(dtype=wp.float64),
        v_cell: wp.array(dtype=mat_dtype),
        grad_energy: wp.array(dtype=wp.float64),
        positions: wp.array(dtype=vec_dtype),
        charges: wp.array(dtype=wp_dtype),
        cell: wp.array(dtype=mat_dtype),
        batch_id: wp.array(dtype=wp.int32),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        neighbor_matrix: wp.array2d(dtype=wp.int32),
        unit_shifts_matrix: wp.array2d(dtype=wp.vec3i),
        mask_value: wp.int32,
        alpha: wp.array(dtype=wp_dtype),
        grad_grad_energy: wp.array(dtype=wp.float64),
        grad_positions: wp.array(dtype=vec_dtype),
        grad_charges: wp.array(dtype=wp.float64),
        grad_cell: wp.array(dtype=mat_dtype),
    ) -> None:
        """Second-derivative node, cooperative-block (neighbor matrix)."""
        atom_i, lane = wp.tid()
        block_size = wp.block_dim()

        qi = wp.float64(charges[atom_i])
        pos_i = positions[atom_i]
        if BATCHED:
            isys = batch_id[atom_i]
        else:
            isys = wp.int32(0)
        alpha_ = wp.float64(alpha[isys])
        cell_t = wp.transpose(cell[isys])
        ge = grad_energy[isys]
        vi = wp.vec3d(
            wp.float64(v_pos[atom_i][0]),
            wp.float64(v_pos[atom_i][1]),
            wp.float64(v_pos[atom_i][2]),
        )
        vqi = wp.float64(0.0)
        if HAS_CHARGE:
            vqi = v_charge[atom_i]

        m_cell = wp.mat33d()
        if CELL_GRAD:
            m = v_cell[isys]
            m_cell = wp.mat33d(
                wp.float64(m[0, 0]),
                wp.float64(m[0, 1]),
                wp.float64(m[0, 2]),
                wp.float64(m[1, 0]),
                wp.float64(m[1, 1]),
                wp.float64(m[1, 2]),
                wp.float64(m[2, 0]),
                wp.float64(m[2, 1]),
                wp.float64(m[2, 2]),
            )
        m_sym = m_cell + wp.transpose(m_cell)

        ddE_acc = wp.float64(0.0)
        gpos_i = wp.vec3d(0.0, 0.0, 0.0)
        gq_i = wp.float64(0.0)
        gcell_acc = wp.mat33d()

        max_neighbors = neighbor_matrix.shape[1]
        k = lane
        while k < max_neighbors:
            j = neighbor_matrix[atom_i, k]
            if j != mask_value:
                pos_j = positions[j]
                shift_vec = unit_shifts_matrix[atom_i, k]
                ddE_acc, gpos_i, gq_i, gcell_acc = accumulate_pair(
                    j,
                    qi,
                    pos_i,
                    pos_j,
                    shift_vec,
                    cell_t,
                    alpha_,
                    ge,
                    vi,
                    vqi,
                    m_cell,
                    m_sym,
                    ddE_acc,
                    gpos_i,
                    gq_i,
                    gcell_acc,
                    charges,
                    v_pos,
                    v_charge,
                    grad_positions,
                    grad_charges,
                )
            k += block_size

        # Cooperative block reductions (compile-time-constant guards -> uniform).
        ddE_sum = wp.tile_sum(wp.tile(ddE_acc))
        gpos_sum = wp.tile_sum(wp.tile(gpos_i, preserve_type=True))
        if HAS_CHARGE:
            gq_sum = wp.tile_sum(wp.tile(gq_i))
        if CELL_GRAD:
            gcell_sum = wp.tile_sum(wp.tile(gcell_acc, preserve_type=True))

        if lane == 0:
            wp.atomic_add(grad_grad_energy, isys, wp.tile_extract(ddE_sum, 0))
            gp = wp.tile_extract(gpos_sum, 0)
            wp.atomic_add(
                grad_positions,
                atom_i,
                type(pos_i)(
                    type(pos_i[0])(gp[0]),
                    type(pos_i[0])(gp[1]),
                    type(pos_i[0])(gp[2]),
                ),
            )
            if HAS_CHARGE:
                wp.atomic_add(grad_charges, atom_i, wp.tile_extract(gq_sum, 0))
            if CELL_GRAD:
                wp.atomic_add(
                    grad_cell, isys, type(cell_t)(wp.tile_extract(gcell_sum, 0))
                )

    _name_and_document(
        _ewald_real_double_backward_tiled,
        base="ewald_real_double_backward",
        wp_dtype=wp_dtype,
        batched=batched,
        neighbor_input=neighbor_input,
        deriv_state=deriv_state,
        cell_grad=cell_grad,
        order="double_backward",
        tiled=True,
    )
    return _ewald_real_double_backward_tiled

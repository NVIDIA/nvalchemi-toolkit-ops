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

"""Shared factory infrastructure for electrostatics Warp kernels.

This module owns the **component-agnostic** pieces of the kernel factory contract:
the :class:`_DerivState` axis enum, the per-dtype Warp vec/mat bundle
(:class:`_WarpDtypes` / :data:`_DTYPE_INFO`), the dtype guard, and the shared
axis literal types. Every electrostatics kernel specialization is identified
conceptually by the axes owned by its component factory, for example::

    (wp_dtype, component, batched, neighbor_input, deriv_state, cell_grad, order)

This tuple is the *conceptual* key; the concrete cache is the ``@lru_cache`` on
each per-component factory (e.g. :func:`make_ewald_real_kernel`), keyed on the
axes that factory accepts.

Per-component kernel code lives in sibling factory modules
(``ewald_real_factory.py`` for ``ewald_real``; sibling factories add
``ewald_recip`` / ``pme``). Those modules import the shared infra below and
define their own ``@lru_cache``d ``make_*`` builders. Keep this module's shared
private infra stable for sibling factories -- adding a new component must not
require editing it.

Factory boundary
----------------
The factory owns **Warp-owned work only**. Torch-native spline / FFT / grid
orchestration stays outside, on Torch autograd.
"""

import enum
import math
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any, Literal, NamedTuple

import warp as wp

from nvalchemiops.math import wp_erfc

__all__ = ["get_backward_scale_kernel"]


# === Factory axis value types ===


class _DerivState(enum.Enum):
    """Derivative-state axis for electrostatics kernel specializations.

    * :attr:`E` -- energy only.
    * :attr:`E_F` -- energy + forces (``-dE/dR``).
    * :attr:`E_F_dQ` -- energy + forces + charge gradient (``dE/dq``).
    * :attr:`E_dQ` -- energy + charge gradient only. This is a forward-only
      direct-output specialization; backward and double-backward construction
      must use :attr:`E_F_dQ`.
    """

    E = 0
    E_F = 1
    E_F_dQ = 2
    E_dQ = 3


# Neighbor-input / order axis literal types (documentation + validation).
_NeighborInput = Literal["list", "matrix", "none"]
_Order = Literal["forward", "backward", "double_backward"]
_CellCacheMode = Literal["none", "system", "atom"]


class _WarpDtypes(NamedTuple):
    """Per-Python-float-dtype Warp vec/mat dtype bundle."""

    vec: type
    mat: type


# === Dtype maps ===


# Electrostatics inputs are float32 or float64 only (accumulators are always
# float64 internally). Unlike the NL factory there is no float16 path.
_DTYPE_INFO: dict[type, _WarpDtypes] = {
    wp.float32: _WarpDtypes(wp.vec3f, wp.mat33f),
    wp.float64: _WarpDtypes(wp.vec3d, wp.mat33d),
}


def _require_supported_dtype(wp_dtype: type) -> None:
    """Validate that ``wp_dtype`` is a supported electrostatics scalar type."""
    if wp_dtype not in _DTYPE_INFO:
        raise ValueError(f"wp_dtype must be wp.float32 or wp.float64; got {wp_dtype!r}")


# Complex-as-vec2 Warp type per scalar dtype (rfftn output is complex64 for f32,
# complex128 for f64; passed to Warp via torch.view_as_real). Used by the PME
# convolve sentinels / slot specs.
_VEC2_INFO: dict[type, type] = {
    wp.float32: wp.vec2f,
    wp.float64: wp.vec2d,
}


# === Shared numeric constants ===

# 2/sqrt(pi); matches ewald_kernels.TWO_OVER_SQRT_PI.
_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)

# Per-pair distance guard for the real-space sum: pairs closer than this (i.e. an
# atom with its own periodic image at the origin) are skipped.
_DISTANCE_EPSILON = 1e-8

# Reciprocal-space ``k_sq`` guard: k-points with ``|k|^2`` below this (the k=0 term)
# are skipped in the recip / PME convolve kernels.
_K_SQUARED_EPSILON = 1e-10


# === Shared per-pair scalar cores (float64) ===
#
# Component-agnostic erfc calculus shared by the ewald_real forward / backward /
# double-backward kernels. Charge-unbundled factors (no division by a charge, so
# they stay finite when a charge is zero):
#   half_S(r)    = (1/2) S(r),  S(r) = erfc(a r)/r^3 + (2a/sqrt(pi)) e^{-a^2 r^2}/r^2
#   half_dS(r)   = (1/2) S'(r)
#   g(r)         = (1/2) erfc(a r)/r           (== _ewald_real_space_charge_grad_potential)
#   dg(r)        = (1/2) d/dr[erfc(a r)/r]
# The bundled force magnitude is fm = qi*qj*half_S = _ewald_real_space_force_magnitude.


@wp.func
def _ewald_half_force_scale(distance: wp.float64, alpha: wp.float64) -> wp.float64:
    """``(1/2) S(r)`` -- charge-unbundled force-magnitude scale."""
    two_over_sqrt_pi = wp.float64(_TWO_OVER_SQRT_PI)
    alpha_r = alpha * distance
    erfc_alpha_r = wp_erfc(alpha_r)
    exp_term = wp.exp(-alpha_r * alpha_r)
    r2 = distance * distance
    s = erfc_alpha_r / (r2 * distance) + two_over_sqrt_pi * alpha * exp_term / r2
    return wp.float64(0.5) * s


@wp.func
def _ewald_half_force_scale_deriv(
    distance: wp.float64, alpha: wp.float64
) -> wp.float64:
    """``(1/2) S'(r)`` -- radial derivative of the force-magnitude scale.

    .. math::

        S'(r) = -3\\,\\mathrm{erfc}(a r)/r^4
                - (2a/\\sqrt{\\pi}) e^{-a^2 r^2} (3/r^3 + 2 a^2 / r).
    """
    two_over_sqrt_pi = wp.float64(_TWO_OVER_SQRT_PI)
    alpha_r = alpha * distance
    erfc_alpha_r = wp_erfc(alpha_r)
    exp_term = wp.exp(-alpha_r * alpha_r)
    r2 = distance * distance
    r3 = r2 * distance
    r4 = r3 * distance
    s_prime = -wp.float64(3.0) * erfc_alpha_r / r4 - two_over_sqrt_pi * alpha * (
        exp_term * (wp.float64(3.0) / r3 + wp.float64(2.0) * alpha * alpha / distance)
    )
    return wp.float64(0.5) * s_prime


@wp.func
def _pair_virial_outer(sep: Any, force: Any) -> wp.mat33d:
    """``sep (x) force`` as a float64 3x3 -- the per-pair strain-virial contribution.

    Both arguments are vec3 in input precision; the outer product is accumulated in
    float64 (``W = sum_pairs sep (x) F``). Shared by the ``ewald_real`` forward
    matrix / CSR loops (the accumulation ``virial_acc += ...`` stays in the loop, so
    the float reduction order is unchanged).
    """
    return wp.mat33d(
        wp.outer(
            wp.vec3d(wp.float64(sep[0]), wp.float64(sep[1]), wp.float64(sep[2])),
            wp.vec3d(wp.float64(force[0]), wp.float64(force[1]), wp.float64(force[2])),
        )
    )


@wp.func
def _ewald_charge_potential_deriv(
    distance: wp.float64, alpha: wp.float64
) -> wp.float64:
    """``g'(r) = (1/2) d/dr[erfc(a r)/r]``.

    .. math::

        g'(r) = -\\tfrac{1}{2}\\left[(2a/\\sqrt{\\pi}) e^{-a^2 r^2}/r
                + \\mathrm{erfc}(a r)/r^2\\right].
    """
    two_over_sqrt_pi = wp.float64(_TWO_OVER_SQRT_PI)
    alpha_r = alpha * distance
    erfc_alpha_r = wp_erfc(alpha_r)
    exp_term = wp.exp(-alpha_r * alpha_r)
    r2 = distance * distance
    return -wp.float64(0.5) * (
        two_over_sqrt_pi * alpha * exp_term / distance + erfc_alpha_r / r2
    )


# === Shared factory boilerplate (component-parameterized) ===


def _make_specialization_module_name(
    component: str,
    *,
    wp_dtype: type,
    batched: bool,
    neighbor_input: str | None = None,
    order: str = "forward",
    suffix: str | None = None,
) -> str:
    """Deterministic Warp ``module=`` name for one kernel specialization.

    A stable per-spec module name (rather than the mtime-newest cache dir) lets the
    dead-branch-elimination test locate generated source deterministically.

    The name is ``"{component}_{dt}_{batch}[_{neighbor_input}][_{tag}]"`` where
    ``dt`` is ``f64``/``f32``, ``batch`` is ``batch``/``single``, the optional
    ``neighbor_input`` segment is appended only when given (ewald_real), and the
    trailing ``tag`` is empty for ``order="forward"`` else ``suffix`` (or ``order``).
    """
    dt = "f64" if wp_dtype is wp.float64 else "f32"
    batch = "batch" if batched else "single"
    name = f"{component}_{dt}_{batch}"
    if neighbor_input is not None:
        name += f"_{neighbor_input}"
    tag = suffix if suffix is not None else ("" if order == "forward" else order)
    if tag:
        name += f"_{tag}"
    return name


def _require_component(component: str, expected: str) -> None:
    """Raise ``NotImplementedError`` unless ``component == expected``."""
    if component != expected:
        raise NotImplementedError(
            f"only component={expected!r} is implemented; got {component!r}"
        )


def _validate_common_axes(
    wp_dtype: type,
    *,
    deriv_state: "_DerivState",
    cell_grad: bool,
    order: str,
    component: str,
    supported_orders: Sequence[str] = ("forward", "backward", "double_backward"),
) -> None:
    """Validate the axes shared by the ``ewald_real`` / ``ewald_recip`` factories.

    Checks the dtype, ``order`` membership, that ``deriv_state`` is a
    :class:`_DerivState`, the permanently-invalid ``cell_grad=True`` + ``deriv_state=E``
    combination, and derivative orders requesting a non-force-bearing
    ``deriv_state``. The
    ``neighbor_input`` axis is component-specific and validated by the caller. (PME has
    a different axis set and does not use this helper.)
    """
    _require_supported_dtype(wp_dtype)
    if order not in supported_orders:
        raise NotImplementedError(
            f"{component} factory supports order in "
            f"{tuple(supported_orders)}; got {order!r}"
        )
    if not isinstance(deriv_state, _DerivState):
        raise ValueError(f"deriv_state must be a _DerivState; got {deriv_state!r}")
    if cell_grad and deriv_state in {_DerivState.E, _DerivState.E_dQ}:
        raise ValueError(
            f"cell_grad=True is invalid with deriv_state={deriv_state!r}: there are no "
            "force terms to sum into the virial. Use E_F or E_F_dQ."
        )
    if order in ("backward", "double_backward") and deriv_state in {
        _DerivState.E,
        _DerivState.E_dQ,
    }:
        raise ValueError(
            f"order={order!r} requires deriv_state in (E_F, E_F_dQ); "
            f"got {deriv_state.name}. E_dQ is a forward-only direct-output "
            "specialization."
        )


# Sentinel slot dtype keys resolved per ``wp_dtype`` by :func:`_alloc_sentinels`.
def _resolve_slot_dtype(slot_kind: str, wp_dtype: type) -> type:
    """Resolve a ``slot_spec`` dtype key to a concrete Warp dtype."""
    info = _DTYPE_INFO[wp_dtype]
    if slot_kind == "vec":
        return info.vec
    if slot_kind == "mat":
        return info.mat
    if slot_kind == "vec2":
        return _VEC2_INFO[wp_dtype]
    if slot_kind == "scalar":
        return wp_dtype
    return {
        "f64": wp.float64,
        "i32": wp.int32,
        "vec3i": wp.vec3i,
    }[slot_kind]


def _alloc_sentinels(
    wp_dtype: type,
    device: str,
    slot_spec: Mapping[str, tuple[tuple[int, ...], str]],
) -> dict[str, wp.array]:
    """Allocate zero-size sentinel arrays from a ``slot_spec``.

    ``slot_spec`` maps a parameter name to ``(shape, dtype_token)`` where
    ``dtype_token`` is one of ``"vec"`` / ``"mat"`` / ``"vec2"`` / ``"scalar"``
    (resolved against ``wp_dtype``) or a fixed ``"f64"`` / ``"i32"`` / ``"vec3i"``.
    The dtype/shape of each sentinel must match its kernel slot exactly -- Warp
    type-checks even zero-size arrays at launch.

    Returns
    -------
    dict[str, wp.array]
        Sentinels keyed by parameter name.
    """
    _require_supported_dtype(wp_dtype)
    return {
        name: wp.empty(shape, dtype=_resolve_slot_dtype(token, wp_dtype), device=device)
        for name, (shape, token) in slot_spec.items()
    }


# === Named + documented generated kernels (mirror of the NL convention) ===
#
# The neighbor-list factory gives each specialization a descriptive ``__name__`` /
# Warp ``key`` and a contract + "Specialization" docstring. These helpers reproduce
# that convention locally (no import from ``neighbors`` -- avoids a cross-package
# import edge). They mutate only kernels the factory itself builds; the reused
# hand-written ``wp.Kernel`` objects (recip ``fill`` / forward ``virial``) are never
# passed here.


def _kernel_specialization_name(
    base: str, *, wp_dtype: type, features: Iterable[str]
) -> str:
    """Build a descriptive specialization name from a base + axis feature tokens.

    Format: ``"{base}__{feature_a}_{feature_b}_...__{dt}"`` where ``dt`` is
    ``f64``/``f32`` and ``features`` are already-lowercased axis tokens (empty
    tokens are dropped).
    """
    dt = "f64" if wp_dtype is wp.float64 else "f32"
    feats = "_".join(f for f in features if f)
    if feats:
        return f"{base}__{feats}__{dt}"
    return f"{base}__{dt}"


def _dtype_token(wp_dtype: type) -> str:
    """Short dtype token used in generated specialization names and docs."""
    return "f64" if wp_dtype is wp.float64 else "f32"


def _deriv_token(deriv_state: _DerivState | None) -> str:
    """Lowercase feature token for a derivative-state axis."""
    if deriv_state is None:
        return ""
    return {
        _DerivState.E: "e",
        _DerivState.E_F: "e_f",
        _DerivState.E_F_dQ: "e_f_dq",
        _DerivState.E_dQ: "e_dq",
    }[deriv_state]


def _set_fn_name(kernel: wp.Kernel, name: str) -> None:
    """Set a generated kernel's descriptive name.

    Updates the Warp identity ``key`` (used in generated symbol names / profiles) and
    the Python ``__name__`` / ``__qualname__`` on both the kernel object and its
    underlying function. Does **not** touch ``kernel.module.name`` (the per-spec
    ``module=`` the dead-branch test keys on).
    """
    kernel.key = name
    kernel.__name__ = name
    kernel.func.__name__ = name
    kernel.func.__qualname__ = name


def _set_fn_doc(kernel: wp.Kernel, doc: str) -> None:
    """Set a generated kernel's docstring (on both the kernel and its function)."""
    kernel.__doc__ = doc
    kernel.func.__doc__ = doc


def _append_specialization_doc(
    base_doc: str | None, *, dtype: str, entries: Sequence[tuple[str, object]]
) -> str:
    """Append a numpydoc-style "Specialization" section to a kernel's contract doc.

    Parameters
    ----------
    base_doc : str or None
        The kernel's static contract docstring (the inner ``def`` docstring).
    dtype : str
        ``"f64"`` / ``"f32"`` -- listed first in the section.
    entries : sequence of (str, object)
        ``(axis_name, value)`` pairs (e.g. ``("batched", True)``).
    """
    base = (base_doc or "").rstrip()
    lines = ["Specialization", "--------------", f"* dtype = {dtype}"]
    lines += [f"* {name} = {value}" for name, value in entries]
    section = "\n".join(lines)
    return f"{base}\n\n{section}\n" if base else f"{section}\n"


def _name_and_document_kernel(
    kernel: wp.Kernel,
    *,
    base: str,
    wp_dtype: type,
    features: Iterable[str],
    entries: Sequence[tuple[str, object]],
) -> None:
    """Apply the standard specialization name and docstring to a generated kernel."""
    _set_fn_name(
        kernel,
        _kernel_specialization_name(base, wp_dtype=wp_dtype, features=features),
    )
    _set_fn_doc(
        kernel,
        _append_specialization_doc(
            kernel.func.__doc__,
            dtype=_dtype_token(wp_dtype),
            entries=entries,
        ),
    )


# === Shared first-backward cache-scaling kernel ===


@lru_cache(maxsize=None)
def get_backward_scale_kernel(
    wp_dtype: type,
    *,
    batched: bool,
    scale_positions: bool,
    scale_charges: bool,
) -> wp.Kernel:
    """Return a kernel that scales detached first-derivative caches.

    The registered Ewald chains precompute detached ``dE/dR`` / ``dE/dq`` caches in
    the forward pass. First backward only needs to multiply those caches by the
    per-system energy cotangent. This shared kernel fuses that multiplication for
    atom-major position and charge outputs.
    """
    _require_supported_dtype(wp_dtype)

    info = _DTYPE_INFO[wp_dtype]
    vec_dtype = info.vec

    BATCHED = wp.constant(bool(batched))
    SCALE_POSITIONS = wp.constant(bool(scale_positions))
    SCALE_CHARGES = wp.constant(bool(scale_charges))

    suffix_parts = ["scale"]
    if scale_positions:
        suffix_parts.append("pos")
    if scale_charges:
        suffix_parts.append("charge")
    module_name = _make_specialization_module_name(
        "backward_scale",
        wp_dtype=wp_dtype,
        batched=batched,
        suffix="_".join(suffix_parts),
    )

    @wp.kernel(module=module_name)
    def _backward_scale(
        grad_energy: wp.array(dtype=wp.float64),
        batch_id: wp.array(dtype=wp.int32),
        dEdR_cache: wp.array(dtype=vec_dtype),
        dEdq_cache: wp.array(dtype=wp.float64),
        grad_positions: wp.array(dtype=vec_dtype),
        grad_charges: wp.array(dtype=wp.float64),
        num_atoms: wp.int32,
    ) -> None:
        """Scale cached first derivatives by per-system energy cotangents."""
        tid = wp.tid()

        if tid < num_atoms:
            if BATCHED:
                isys = batch_id[tid]
            else:
                isys = wp.int32(0)
            ge = grad_energy[isys]

            if SCALE_POSITIONS:
                dpos = dEdR_cache[tid]
                grad_positions[tid] = type(dpos)(
                    type(dpos[0])(ge) * dpos[0],
                    type(dpos[0])(ge) * dpos[1],
                    type(dpos[0])(ge) * dpos[2],
                )

            if SCALE_CHARGES:
                grad_charges[tid] = ge * dEdq_cache[tid]

    features = [
        "batch" if batched else "single",
        "pos" if scale_positions else "",
        "charge" if scale_charges else "",
    ]
    _name_and_document_kernel(
        _backward_scale,
        base="backward_scale",
        wp_dtype=wp_dtype,
        features=features,
        entries=[
            ("batched", batched),
            ("scale_positions", scale_positions),
            ("scale_charges", scale_charges),
        ],
    )
    return _backward_scale

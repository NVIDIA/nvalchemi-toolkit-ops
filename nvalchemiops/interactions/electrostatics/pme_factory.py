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

"""``pme`` per-component kernel factory.

The Warp-owned PME kernels -- the reciprocal-space convolve (fused Green's
factor + B-spline deconvolution + complex multiply) and the per-atom energy
corrections -- are specialized across the derivative matrix. One source per
``(wp_dtype, batched, order)`` triple is built by capturing the ``BATCHED`` axis
as a Python compile-time constant; Warp's codegen dead-eliminates the unused
single-vs-batch branch.

Two components are covered:

* ``component="pme_convolve"`` -- the k-space convolve. ``order="forward"`` writes
  ``convolved = G(k)/C^2(k) * mesh_fft`` (complex x real). ``order="backward"``
  emits ``grad_mesh_fft``, the scalar ``grad_alpha`` / ``grad_volume`` (atomic
  reduction over k-points), and ``grad_k_squared``. ``order="double_backward"`` is
  the second-derivative node: it emits the position-relevant ``dL/dmesh_fft`` and
  ``dL/dgrad_convolved`` plus the cell/stress second-order terms
  ``dL/dk_squared`` (per-k) and ``dL/dalpha`` / ``dL/dvolume`` (atomic-summed).
* ``component="pme_corrections"`` -- the per-atom self + background energy
  corrections. ``order="forward"`` writes the corrected energies (and, with
  ``charge_grad=True``, the analytical ``dE/dq``). ``order="backward"`` /
  ``"double_backward"`` emit the first / second derivatives w.r.t.
  ``(raw, charges, volume, alpha, total_charge)``.

Each ``order`` family is the current Warp kernel body used by the low-level PME
wrappers. Parity coverage keeps direct factory launches and wrapper launchers
aligned, while finite-difference tests provide the independent derivative
oracle.

Factory boundary
----------------
The factory owns Warp-owned work ONLY. The FFT (``torch.fft.rfftn/irfftn``), the
B-spline spline spread / gather, and the k-vector / grid orchestration stay
OUTSIDE the factory, on Torch autograd -- exactly as in the public PME pipeline.
The convolve / corrections kernels here are the only PME pieces that
run on Warp, and they are the only pieces this factory emits.

Scope note: the convolve ``double_backward`` kernel implements both the position
second-order terms (the force-loss double-backward) AND the alpha/volume/cell
(k_squared) second-order terms (the stress-loss double-backward). k_squared and
volume are functions of the cell, so ``grad_alpha`` / ``grad_volume`` /
``grad_k_squared`` carry the cell/stress second order; PyTorch maps k²/V → cell
outside the kernel.
"""

from functools import lru_cache

import warp as wp

from nvalchemiops.interactions.electrostatics._factory_common import (
    _K_SQUARED_EPSILON,
    _VEC2_INFO,
    _alloc_sentinels,
    _make_specialization_module_name,
    _name_and_document_kernel,
    _require_supported_dtype,
)
from nvalchemiops.interactions.electrostatics.pme_kernels import (
    PI,
    TWOPI,
    wp_exp_kernel,
)

__all__ = [
    "alloc_pme_sentinels",
    "get_pme_kernel",
    "make_pme_kernel",
]

_COMPONENTS = ("pme_convolve", "pme_corrections")
_ORDERS = ("forward", "backward", "double_backward")

# Short per-component tag used in the deterministic module name (conv / corr).
_COMPONENT_SHORT = {"pme_convolve": "conv", "pme_corrections": "corr"}


# === Sentinel allocator ===


# slot_spec: name -> (shape, dtype_token). ``vec2`` resolves to the complex-as-vec2
# type; ``scalar`` to the kernel's float dtype. Must match each slot's dtype exactly.
_SENTINEL_SLOT_SPEC: dict[str, tuple[tuple[int, ...], str]] = {
    # convolve mesh slots (3d single-system shape; batch callers use 4d but the
    # kernel never indexes a sentinel so the rank is irrelevant)
    "mesh3d": ((0, 0, 0), "vec2"),
    "real3d": ((0, 0, 0), "scalar"),
    # scalar / per-system slots
    "scalar": ((0,), "scalar"),
    # corrections per-atom slots
    "atoms": ((0,), "scalar"),
    "batch_idx": ((0,), "i32"),
}


_SENTINEL_CACHE: dict[tuple[type, str], dict[str, wp.array]] = {}


def alloc_pme_sentinels(wp_dtype: type, device: str) -> dict[str, wp.array]:
    """Return cached zero-size sentinel arrays for inactive PME slots.

    Callers pass these for any input/output slot not active in the current
    specialization. Covers both the convolve (mesh / k-space) slots and the
    corrections (per-atom) slots. Memoized per ``(wp_dtype, device)`` (zero-size,
    read-only) so repeated launches do not re-allocate them per call.

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


def _pme_module_name(wp_dtype: type, component: str, batched: bool, order: str) -> str:
    """Deterministic Warp ``module=`` name for one PME specialization."""
    return _make_specialization_module_name(
        f"pme_{_COMPONENT_SHORT[component]}",
        wp_dtype=wp_dtype,
        batched=batched,
        order=order,
    )


def _name_and_document(
    kernel: wp.Kernel,
    *,
    component: str,
    wp_dtype: type,
    batched: bool,
    order: str,
    charge_grad: bool = False,
) -> None:
    """Give a generated PME kernel a descriptive name + spec docstring."""
    features = [
        "chargegrad" if charge_grad else "",
        "batch" if batched else "single",
        "" if order == "forward" else order,
    ]
    entries: list[tuple[str, object]] = [
        ("component", component),
        ("batched", bool(batched)),
        ("order", order),
    ]
    if component == "pme_corrections":
        entries.append(("charge_grad", bool(charge_grad)))
    _name_and_document_kernel(
        kernel,
        base=component,
        wp_dtype=wp_dtype,
        features=features,
        entries=entries,
    )


# === Shared convolve scalar helper ===


@lru_cache(maxsize=None)
def _make_convolve_factor_fn(wp_dtype: type) -> wp.Function:
    """Build the specialized PME Green/deconvolution factor helper."""
    _require_supported_dtype(wp_dtype)

    @wp.func
    def _convolve_factor(
        i: wp.int32,
        j: wp.int32,
        k: wp.int32,
        k_sq: wp_dtype,
        moduli_x: wp.array(dtype=wp_dtype),
        moduli_y: wp.array(dtype=wp_dtype),
        moduli_z: wp.array(dtype=wp_dtype),
        alpha_: wp_dtype,
        volume_: wp_dtype,
    ):
        """Return ``G(k) / C^2(k)`` for one PME mesh frequency.

        The zero-frequency and small-``k_sq`` guards mirror the hand-written
        kernels. The helper is shared by forward, backward, and double-backward
        convolve kernels; callers keep their order-specific derivative algebra.
        """
        zero = wp_dtype(0.0)
        one = wp_dtype(1.0)
        four = wp_dtype(4.0)
        threshold = wp_dtype(_K_SQUARED_EPSILON)
        clamp_threshold = wp_dtype(1e-10)
        twopi = wp_dtype(TWOPI)

        sf = moduli_x[i] * moduli_y[j] * moduli_z[k]
        if sf < clamp_threshold:
            sf = clamp_threshold
        sf_sq = sf * sf

        if k_sq < threshold:
            factor = zero
        else:
            exp_factor = wp_exp_kernel(k_sq, one / (four * alpha_ * alpha_))
            factor = twopi * exp_factor / (volume_ * sf_sq)
        if i == 0 and j == 0 and k == 0:
            factor = zero
        return factor

    return _convolve_factor


# === Axis validation ===


def _validate_axes(
    wp_dtype: type,
    component: str,
    batched: bool,
    order: str,
    charge_grad: bool,
) -> None:
    """Raise for unsupported / invalid component-axis combinations."""
    _require_supported_dtype(wp_dtype)
    if component not in _COMPONENTS:
        raise NotImplementedError(
            f"pme factory supports component in {_COMPONENTS}; got {component!r}"
        )
    if order not in _ORDERS:
        raise NotImplementedError(
            f"pme factory supports order in {_ORDERS}; got {order!r}"
        )
    if charge_grad and component != "pme_corrections":
        raise ValueError(
            "charge_grad=True is only meaningful for component='pme_corrections'"
        )
    if charge_grad and order != "forward":
        raise NotImplementedError(
            "charge_grad=True is only supported for order='forward' "
            "(the analytical dE/dq is a forward-only output; its own autograd "
            "routes through the regular corrections backward)"
        )


# === pme kernel factory ===


@lru_cache(maxsize=None)
def make_pme_kernel(
    wp_dtype: type,
    *,
    component: str,
    batched: bool = False,
    order: str = "forward",
    charge_grad: bool = False,
) -> wp.Kernel:
    """Return a cached, specialized PME Warp kernel.

    The body branches on the Python compile-time constant ``BATCHED`` (single vs
    batched indexing); Warp dead-eliminates the unused branch. Every ``order``
    family reproduces the corresponding hand-written kernel body verbatim, so the
    forward energies / first derivatives stay bit-exact vs the hand-written
    launchers.

    Parameters
    ----------
    wp_dtype : type
        ``wp.float32`` or ``wp.float64``.
    component : {"pme_convolve", "pme_corrections"}
        Which Warp-owned PME kernel family to build.
    batched : bool
        Single-system (``False``) vs batched (``True``).
    order : {"forward", "backward", "double_backward"}
        Forward output, first-derivative autograd node, or second-derivative node.
    charge_grad : bool
        For ``component="pme_corrections"``, ``order="forward"`` only: also write
        the analytical ``dE/dq`` output.
    """
    _validate_axes(wp_dtype, component, batched, order, charge_grad)
    if component == "pme_convolve":
        return _make_convolve_kernel(wp_dtype, batched, order)
    return _make_corrections_kernel(wp_dtype, batched, order, charge_grad)


def get_pme_kernel(
    wp_dtype: type,
    *,
    component: str,
    batched: bool = False,
    order: str = "forward",
    charge_grad: bool = False,
) -> wp.Kernel:
    """Return a cached PME kernel, validating dtype + component.

    Memoization is the ``@lru_cache`` on :func:`make_pme_kernel`; there is no
    separate cache dict. Every argument is forwarded **by keyword** (including
    ``wp_dtype``) so a positional vs keyword call can never produce a duplicate
    ``@lru_cache`` entry.
    """
    _require_supported_dtype(wp_dtype)
    if component not in _COMPONENTS:
        raise NotImplementedError(
            f"pme factory supports component in {_COMPONENTS}; got {component!r}"
        )
    return make_pme_kernel(
        wp_dtype=wp_dtype,
        component=component,
        batched=batched,
        order=order,
        charge_grad=charge_grad,
    )


# === pme_convolve builders ===


def _make_convolve_single_kernel(wp_dtype: type, order: str) -> wp.Kernel:
    """Build a single-system 3D convolve kernel."""
    vec2 = _VEC2_INFO[wp_dtype]
    module_name = _pme_module_name(wp_dtype, "pme_convolve", False, order)
    convolve_factor = _make_convolve_factor_fn(wp_dtype)

    if order == "forward":

        @wp.kernel(module=module_name)
        def _convolve_single(
            mesh_fft: wp.array3d(dtype=vec2),
            k_squared: wp.array3d(dtype=wp_dtype),
            moduli_x: wp.array(dtype=wp_dtype),
            moduli_y: wp.array(dtype=wp_dtype),
            moduli_z: wp.array(dtype=wp_dtype),
            alpha: wp.array(dtype=wp_dtype),
            volume: wp.array(dtype=wp_dtype),
            convolved_mesh: wp.array3d(dtype=vec2),
        ) -> None:
            """Single-system fused convolve forward. Meshes are 3d
            ``(nx, ny, nz_r)``."""
            i, j, k = wp.tid()
            k_sq = k_squared[i, j, k]
            alpha_ = alpha[0]
            volume_ = volume[0]

            factor = convolve_factor(
                i, j, k, k_sq, moduli_x, moduli_y, moduli_z, alpha_, volume_
            )

            c = mesh_fft[i, j, k]
            convolved_mesh[i, j, k] = vec2(c[0] * factor, c[1] * factor)

        _name_and_document(
            _convolve_single,
            component="pme_convolve",
            wp_dtype=wp_dtype,
            batched=False,
            order="forward",
        )
        return _convolve_single

    if order == "backward":

        @wp.kernel(module=module_name)
        def _convolve_single_backward(
            mesh_fft: wp.array3d(dtype=vec2),
            grad_convolved: wp.array3d(dtype=vec2),
            k_squared: wp.array3d(dtype=wp_dtype),
            moduli_x: wp.array(dtype=wp_dtype),
            moduli_y: wp.array(dtype=wp_dtype),
            moduli_z: wp.array(dtype=wp_dtype),
            alpha: wp.array(dtype=wp_dtype),
            volume: wp.array(dtype=wp_dtype),
            grad_mesh_fft: wp.array3d(dtype=vec2),
            grad_alpha: wp.array(dtype=wp_dtype),
            grad_volume: wp.array(dtype=wp_dtype),
            grad_k_squared: wp.array3d(dtype=wp_dtype),
        ) -> None:
            """Single-system convolve backward."""
            i, j, k = wp.tid()
            k_sq = k_squared[i, j, k]
            alpha_ = alpha[0]
            volume_ = volume[0]

            factor = convolve_factor(
                i, j, k, k_sq, moduli_x, moduli_y, moduli_z, alpha_, volume_
            )
            zero = wp_dtype(0.0)
            one = wp_dtype(1.0)
            two = wp_dtype(2.0)
            four = wp_dtype(4.0)

            g = grad_convolved[i, j, k]
            m = mesh_fft[i, j, k]
            grad_mesh_fft[i, j, k] = vec2(g[0] * factor, g[1] * factor)

            re_inner = g[0] * m[0] + g[1] * m[1]
            contrib = re_inner * factor
            if factor > zero:
                d_alpha = contrib * k_sq / (two * alpha_ * alpha_ * alpha_)
                wp.atomic_add(grad_alpha, 0, d_alpha)
                d_vol = -contrib / volume_
                wp.atomic_add(grad_volume, 0, d_vol)
                grad_k_squared[i, j, k] = -contrib * (
                    one / (four * alpha_ * alpha_) + one / k_sq
                )
            else:
                grad_k_squared[i, j, k] = zero

        _name_and_document(
            _convolve_single_backward,
            component="pme_convolve",
            wp_dtype=wp_dtype,
            batched=False,
            order="backward",
        )
        return _convolve_single_backward

    @wp.kernel(module=module_name)
    def _convolve_single_double_backward(
        h_grad_mesh: wp.array3d(dtype=vec2),
        h_alpha: wp.array(dtype=wp_dtype),
        h_volume: wp.array(dtype=wp_dtype),
        h_grad_ksq: wp.array3d(dtype=wp_dtype),
        mesh_fft: wp.array3d(dtype=vec2),
        grad_convolved: wp.array3d(dtype=vec2),
        k_squared: wp.array3d(dtype=wp_dtype),
        moduli_x: wp.array(dtype=wp_dtype),
        moduli_y: wp.array(dtype=wp_dtype),
        moduli_z: wp.array(dtype=wp_dtype),
        alpha: wp.array(dtype=wp_dtype),
        volume: wp.array(dtype=wp_dtype),
        grad_mesh_out: wp.array3d(dtype=vec2),
        grad_grad_convolved: wp.array3d(dtype=vec2),
        grad_k_squared_out: wp.array3d(dtype=wp_dtype),
        grad_alpha_out: wp.array(dtype=wp_dtype),
        grad_volume_out: wp.array(dtype=wp_dtype),
    ) -> None:
        """Single-system convolve double-backward."""
        i, j, k = wp.tid()
        k_sq = k_squared[i, j, k]
        alpha_ = alpha[0]
        volume_ = volume[0]

        factor = convolve_factor(
            i, j, k, k_sq, moduli_x, moduli_y, moduli_z, alpha_, volume_
        )
        zero = wp_dtype(0.0)
        one = wp_dtype(1.0)
        two = wp_dtype(2.0)
        three = wp_dtype(3.0)
        four = wp_dtype(4.0)

        m = mesh_fft[i, j, k]
        g = grad_convolved[i, j, k]
        hm = h_grad_mesh[i, j, k]

        if factor > zero:
            ha = h_alpha[0]
            hv = h_volume[0]
            hk = h_grad_ksq[i, j, k]
            c_alpha = k_sq / (two * alpha_ * alpha_ * alpha_)
            p_term = one / (four * alpha_ * alpha_) + one / k_sq
            inv_2a3 = one / (two * alpha_ * alpha_ * alpha_)

            w = factor * (ha * c_alpha - hv / volume_ - hk * p_term)

            grad_grad_convolved[i, j, k] = vec2(
                factor * hm[0] + w * m[0], factor * hm[1] + w * m[1]
            )
            grad_mesh_out[i, j, k] = vec2(w * g[0], w * g[1])

            re = g[0] * m[0] + g[1] * m[1]
            m_term = hm[0] * g[0] + hm[1] * g[1]
            ref = re * factor

            d_alpha = factor * c_alpha * m_term + ref * (
                ha * (c_alpha * c_alpha - three * c_alpha / alpha_)
                - hv * c_alpha / volume_
                - hk * (c_alpha * p_term - inv_2a3)
            )
            wp.atomic_add(grad_alpha_out, 0, d_alpha)

            d_vol = -factor * m_term / volume_ + ref * (
                -ha * c_alpha / volume_
                + two * hv / (volume_ * volume_)
                + hk * p_term / volume_
            )
            wp.atomic_add(grad_volume_out, 0, d_vol)

            grad_k_squared_out[i, j, k] = -factor * p_term * m_term + ref * (
                ha * (inv_2a3 - c_alpha * p_term)
                + hv * p_term / volume_
                + hk * (p_term * p_term + one / (k_sq * k_sq))
            )
        else:
            grad_grad_convolved[i, j, k] = vec2(factor * hm[0], factor * hm[1])
            grad_mesh_out[i, j, k] = vec2(zero, zero)
            grad_k_squared_out[i, j, k] = zero

    _name_and_document(
        _convolve_single_double_backward,
        component="pme_convolve",
        wp_dtype=wp_dtype,
        batched=False,
        order="double_backward",
    )
    return _convolve_single_double_backward


def _make_convolve_kernel(wp_dtype: type, batched: bool, order: str) -> wp.Kernel:
    """Build a convolve kernel (forward / backward / double_backward)."""
    vec2 = _VEC2_INFO[wp_dtype]
    BATCHED = bool(batched)
    module_name = _pme_module_name(wp_dtype, "pme_convolve", BATCHED, order)
    convolve_factor = _make_convolve_factor_fn(wp_dtype)

    if not BATCHED:
        return _make_convolve_single_kernel(wp_dtype, order)

    if order == "forward":

        @wp.kernel(module=module_name)
        def _convolve(
            mesh_fft: wp.array4d(dtype=vec2),
            k_squared: wp.array4d(dtype=wp_dtype),
            moduli_x: wp.array(dtype=wp_dtype),
            moduli_y: wp.array(dtype=wp_dtype),
            moduli_z: wp.array(dtype=wp_dtype),
            alpha: wp.array(dtype=wp_dtype),
            volume: wp.array(dtype=wp_dtype),
            convolved_mesh: wp.array4d(dtype=vec2),
        ) -> None:
            """Fused convolve forward. Meshes are 4d ``(B, nx, ny, nz_r)``;
            single-system callers pass ``B=1``."""
            b, i, j, k = wp.tid()
            if BATCHED:
                sys = b
            else:
                sys = wp.int32(0)

            k_sq = k_squared[b, i, j, k]
            alpha_ = alpha[sys]
            volume_ = volume[sys]

            factor = convolve_factor(
                i, j, k, k_sq, moduli_x, moduli_y, moduli_z, alpha_, volume_
            )

            c = mesh_fft[b, i, j, k]
            convolved_mesh[b, i, j, k] = vec2(c[0] * factor, c[1] * factor)

        _name_and_document(
            _convolve,
            component="pme_convolve",
            wp_dtype=wp_dtype,
            batched=batched,
            order="forward",
        )
        return _convolve

    if order == "backward":

        @wp.kernel(module=module_name)
        def _convolve_backward(
            mesh_fft: wp.array4d(dtype=vec2),
            grad_convolved: wp.array4d(dtype=vec2),
            k_squared: wp.array4d(dtype=wp_dtype),
            moduli_x: wp.array(dtype=wp_dtype),
            moduli_y: wp.array(dtype=wp_dtype),
            moduli_z: wp.array(dtype=wp_dtype),
            alpha: wp.array(dtype=wp_dtype),
            volume: wp.array(dtype=wp_dtype),
            grad_mesh_fft: wp.array4d(dtype=vec2),
            grad_alpha: wp.array(dtype=wp_dtype),
            grad_volume: wp.array(dtype=wp_dtype),
            grad_k_squared: wp.array4d(dtype=wp_dtype),
        ) -> None:
            """Fused convolve backward. ``grad_alpha`` / ``grad_volume`` are
            atomically accumulated per-system; both are zero-initialized by the
            caller."""
            b, i, j, k = wp.tid()
            if BATCHED:
                sys = b
            else:
                sys = wp.int32(0)

            k_sq = k_squared[b, i, j, k]
            alpha_ = alpha[sys]
            volume_ = volume[sys]

            factor = convolve_factor(
                i, j, k, k_sq, moduli_x, moduli_y, moduli_z, alpha_, volume_
            )
            zero = wp_dtype(0.0)
            one = wp_dtype(1.0)
            two = wp_dtype(2.0)
            four = wp_dtype(4.0)

            g = grad_convolved[b, i, j, k]
            m = mesh_fft[b, i, j, k]
            grad_mesh_fft[b, i, j, k] = vec2(g[0] * factor, g[1] * factor)

            re_inner = g[0] * m[0] + g[1] * m[1]
            contrib = re_inner * factor
            if factor > zero:
                d_alpha = contrib * k_sq / (two * alpha_ * alpha_ * alpha_)
                wp.atomic_add(grad_alpha, sys, d_alpha)
                d_vol = -contrib / volume_
                wp.atomic_add(grad_volume, sys, d_vol)
                grad_k_squared[b, i, j, k] = -contrib * (
                    one / (four * alpha_ * alpha_) + one / k_sq
                )
            else:
                grad_k_squared[b, i, j, k] = zero

        _name_and_document(
            _convolve_backward,
            component="pme_convolve",
            wp_dtype=wp_dtype,
            batched=batched,
            order="backward",
        )
        return _convolve_backward

    @wp.kernel(module=module_name)
    def _convolve_double_backward(
        h_grad_mesh: wp.array4d(dtype=vec2),
        h_alpha: wp.array(dtype=wp_dtype),
        h_volume: wp.array(dtype=wp_dtype),
        h_grad_ksq: wp.array4d(dtype=wp_dtype),
        mesh_fft: wp.array4d(dtype=vec2),
        grad_convolved: wp.array4d(dtype=vec2),
        k_squared: wp.array4d(dtype=wp_dtype),
        moduli_x: wp.array(dtype=wp_dtype),
        moduli_y: wp.array(dtype=wp_dtype),
        moduli_z: wp.array(dtype=wp_dtype),
        alpha: wp.array(dtype=wp_dtype),
        volume: wp.array(dtype=wp_dtype),
        grad_mesh_out: wp.array4d(dtype=vec2),
        grad_grad_convolved: wp.array4d(dtype=vec2),
        grad_k_squared_out: wp.array4d(dtype=wp_dtype),
        grad_alpha_out: wp.array(dtype=wp_dtype),
        grad_volume_out: wp.array(dtype=wp_dtype),
    ) -> None:
        """Fused convolve double-backward. Emits the position-relevant
        ``dL/dmesh_fft`` and ``dL/dgrad_convolved`` plus the cell/stress
        second-order terms ``grad_k_squared_out`` (per-k) and ``grad_alpha_out``
        / ``grad_volume_out`` (atomic-summed over k). Scalar grad outputs must be
        zero-initialized by the caller. See the module docstring for the
        derivative contract and saved-state layout."""
        b, i, j, k = wp.tid()
        if BATCHED:
            sys = b
        else:
            sys = wp.int32(0)

        k_sq = k_squared[b, i, j, k]
        alpha_ = alpha[sys]
        volume_ = volume[sys]

        factor = convolve_factor(
            i, j, k, k_sq, moduli_x, moduli_y, moduli_z, alpha_, volume_
        )
        zero = wp_dtype(0.0)
        one = wp_dtype(1.0)
        two = wp_dtype(2.0)
        three = wp_dtype(3.0)
        four = wp_dtype(4.0)

        m = mesh_fft[b, i, j, k]
        g = grad_convolved[b, i, j, k]
        hm = h_grad_mesh[b, i, j, k]

        if factor > zero:
            ha = h_alpha[sys]
            hv = h_volume[sys]
            hk = h_grad_ksq[b, i, j, k]
            c_alpha = k_sq / (two * alpha_ * alpha_ * alpha_)
            p_term = one / (four * alpha_ * alpha_) + one / k_sq
            inv_2a3 = one / (two * alpha_ * alpha_ * alpha_)

            w = factor * (ha * c_alpha - hv / volume_ - hk * p_term)

            grad_grad_convolved[b, i, j, k] = vec2(
                factor * hm[0] + w * m[0], factor * hm[1] + w * m[1]
            )
            grad_mesh_out[b, i, j, k] = vec2(w * g[0], w * g[1])

            re = g[0] * m[0] + g[1] * m[1]
            m_term = hm[0] * g[0] + hm[1] * g[1]
            ref = re * factor

            d_alpha = factor * c_alpha * m_term + ref * (
                ha * (c_alpha * c_alpha - three * c_alpha / alpha_)
                - hv * c_alpha / volume_
                - hk * (c_alpha * p_term - inv_2a3)
            )
            wp.atomic_add(grad_alpha_out, sys, d_alpha)

            d_vol = -factor * m_term / volume_ + ref * (
                -ha * c_alpha / volume_
                + two * hv / (volume_ * volume_)
                + hk * p_term / volume_
            )
            wp.atomic_add(grad_volume_out, sys, d_vol)

            grad_k_squared_out[b, i, j, k] = -factor * p_term * m_term + ref * (
                ha * (inv_2a3 - c_alpha * p_term)
                + hv * p_term / volume_
                + hk * (p_term * p_term + one / (k_sq * k_sq))
            )
        else:
            grad_grad_convolved[b, i, j, k] = vec2(factor * hm[0], factor * hm[1])
            grad_mesh_out[b, i, j, k] = vec2(zero, zero)
            grad_k_squared_out[b, i, j, k] = zero

    _name_and_document(
        _convolve_double_backward,
        component="pme_convolve",
        wp_dtype=wp_dtype,
        batched=batched,
        order="double_backward",
    )
    return _convolve_double_backward


# === pme_corrections builders ===


def _make_corrections_kernel(
    wp_dtype: type, batched: bool, order: str, charge_grad: bool
) -> wp.Kernel:
    """Build a corrections kernel (forward / backward / double_backward)."""
    BATCHED = bool(batched)
    CHARGE_GRAD = bool(charge_grad)
    module_name = _pme_module_name(wp_dtype, "pme_corrections", BATCHED, order)

    if order == "forward":

        @wp.kernel(module=module_name)
        def _corrections(
            raw_energies: wp.array(dtype=wp_dtype),
            charges: wp.array(dtype=wp_dtype),
            batch_idx: wp.array(dtype=wp.int32),
            volume: wp.array(dtype=wp_dtype),
            alpha: wp.array(dtype=wp_dtype),
            total_charge: wp.array(dtype=wp_dtype),
            corrected_energies: wp.array(dtype=wp_dtype),
            charge_gradients: wp.array(dtype=wp_dtype),
        ) -> None:
            """Per-atom self + background corrections (+ optional dE/dq)."""
            atom_idx = wp.tid()
            if BATCHED:
                sys = batch_idx[atom_idx]
            else:
                sys = wp.int32(0)

            charge = charges[atom_idx]
            raw_energy = raw_energies[atom_idx]
            alpha_ = alpha[sys]
            total_charge_ = total_charge[sys]
            volume_ = volume[sys]

            pi = wp_dtype(PI)
            two = wp_dtype(2.0)

            potential_energy = charge * raw_energy
            self_contrib = charge * charge * alpha_ / wp.sqrt(pi)
            background_contrib = (
                charge * pi * total_charge_ / (two * alpha_ * alpha_ * volume_)
            )
            corrected_energies[atom_idx] = (
                potential_energy - self_contrib - background_contrib
            )

            if CHARGE_GRAD:
                self_energy_grad = two * alpha_ * charge / wp.sqrt(pi)
                background_grad = pi * total_charge_ / (alpha_ * alpha_ * volume_)
                charge_gradients[atom_idx] = (
                    two * raw_energy - self_energy_grad - background_grad
                )

        _name_and_document(
            _corrections,
            component="pme_corrections",
            wp_dtype=wp_dtype,
            batched=batched,
            order="forward",
            charge_grad=charge_grad,
        )
        return _corrections

    if order == "backward":

        @wp.kernel(module=module_name)
        def _corrections_backward(
            grad_E: wp.array(dtype=wp_dtype),
            raw_energies: wp.array(dtype=wp_dtype),
            charges: wp.array(dtype=wp_dtype),
            batch_idx: wp.array(dtype=wp.int32),
            volume: wp.array(dtype=wp_dtype),
            alpha: wp.array(dtype=wp_dtype),
            total_charge: wp.array(dtype=wp_dtype),
            grad_raw: wp.array(dtype=wp_dtype),
            grad_charges: wp.array(dtype=wp_dtype),
            grad_volume: wp.array(dtype=wp_dtype),
            grad_alpha: wp.array(dtype=wp_dtype),
            grad_total_charge: wp.array(dtype=wp_dtype),
        ) -> None:
            """First-derivative node. Scalar grads are zero-initialized by the
            caller and accumulated atomically per-system."""
            i = wp.tid()
            if BATCHED:
                s = batch_idx[i]
            else:
                s = wp.int32(0)
            g = grad_E[i]
            q = charges[i]
            r = raw_energies[i]
            a = alpha[s]
            v = volume[s]
            qtot = total_charge[s]

            pi = wp_dtype(PI)
            two = wp_dtype(2.0)
            sqrt_pi = wp.sqrt(pi)
            c1 = a / sqrt_pi
            c2 = pi / (two * a * a * v)

            grad_raw[i] = g * q
            grad_charges[i] = g * (r - two * c1 * q - c2 * qtot)

            d_alpha = g * (-(q * q) / sqrt_pi + pi * q * qtot / (a * a * a * v))
            wp.atomic_add(grad_alpha, s, d_alpha)
            d_volume = g * pi * q * qtot / (two * a * a * v * v)
            wp.atomic_add(grad_volume, s, d_volume)
            d_qtot = -g * c2 * q
            wp.atomic_add(grad_total_charge, s, d_qtot)

        _name_and_document(
            _corrections_backward,
            component="pme_corrections",
            wp_dtype=wp_dtype,
            batched=batched,
            order="backward",
        )
        return _corrections_backward

    @wp.kernel(module=module_name)
    def _corrections_double_backward(
        h_raw: wp.array(dtype=wp_dtype),
        h_chg: wp.array(dtype=wp_dtype),
        h_vol: wp.array(dtype=wp_dtype),
        h_alpha: wp.array(dtype=wp_dtype),
        h_qtot: wp.array(dtype=wp_dtype),
        grad_E: wp.array(dtype=wp_dtype),
        raw_energies: wp.array(dtype=wp_dtype),
        charges: wp.array(dtype=wp_dtype),
        batch_idx: wp.array(dtype=wp.int32),
        volume: wp.array(dtype=wp_dtype),
        alpha: wp.array(dtype=wp_dtype),
        total_charge: wp.array(dtype=wp_dtype),
        grad_grad_E: wp.array(dtype=wp_dtype),
        grad_raw: wp.array(dtype=wp_dtype),
        grad_charges: wp.array(dtype=wp_dtype),
        grad_volume: wp.array(dtype=wp_dtype),
        grad_alpha: wp.array(dtype=wp_dtype),
        grad_total_charge: wp.array(dtype=wp_dtype),
    ) -> None:
        """Second-derivative node for the per-atom corrections. Scalar grads are
        zero-initialized by the caller (atomic per-system accumulation)."""
        i = wp.tid()
        if BATCHED:
            s = batch_idx[i]
        else:
            s = wp.int32(0)
        g_i = grad_E[i]
        q = charges[i]
        r = raw_energies[i]
        a = alpha[s]
        v = volume[s]
        qtot = total_charge[s]
        hr = h_raw[i]
        hc = h_chg[i]
        hv = h_vol[s]
        ha = h_alpha[s]
        hq = h_qtot[s]

        pi = wp_dtype(PI)
        two = wp_dtype(2.0)
        three = wp_dtype(3.0)
        sqrt_pi = wp.sqrt(pi)
        c1 = a / sqrt_pi
        c2 = pi / (two * a * a * v)
        A_i = -(q * q) / sqrt_pi + pi * q * qtot / (a * a * a * v)
        B_i = pi * q * qtot / (two * a * a * v * v)
        D_i = -pi * q / (two * a * a * v)

        grad_grad_E[i] = (
            hr * q
            + hc * (r - two * c1 * q - c2 * qtot)
            + ha * A_i
            + hv * B_i
            + hq * D_i
        )
        grad_raw[i] = hc * g_i

        dq = g_i * (
            hr
            + hc * (-two * c1)
            + ha * (-two * q / sqrt_pi + pi * qtot / (a * a * a * v))
            + hv * (pi * qtot / (two * a * a * v * v))
            + hq * (-pi / (two * a * a * v))
        )
        grad_charges[i] = dq

        g_q = g_i * q
        dV_atom = (
            hc * g_i * qtot * pi / (two * a * a * v * v)
            + ha * (-pi * qtot / (a * a * a * v * v)) * g_q
            + hv * (-pi * qtot / (a * a * v * v * v)) * g_q
            + hq * (pi / (two * a * a * v * v)) * g_q
        )
        wp.atomic_add(grad_volume, s, dV_atom)

        dA_atom = (
            hc * g_i * (-two * q / sqrt_pi + pi * qtot / (a * a * a * v))
            + ha * (-three * pi * qtot / (a * a * a * a * v)) * g_q
            + hv * (-pi * qtot / (a * a * a * v * v)) * g_q
            + hq * (pi / (a * a * a * v)) * g_q
        )
        wp.atomic_add(grad_alpha, s, dA_atom)

        dQ_atom = (
            hc * g_i * (-pi / (two * a * a * v))
            + ha * (pi / (a * a * a * v)) * g_q
            + hv * (pi / (two * a * a * v * v)) * g_q
        )
        wp.atomic_add(grad_total_charge, s, dQ_atom)

    _name_and_document(
        _corrections_double_backward,
        component="pme_corrections",
        wp_dtype=wp_dtype,
        batched=batched,
        order="double_backward",
    )
    return _corrections_double_backward

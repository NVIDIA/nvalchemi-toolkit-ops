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

r"""``ewald_recip`` per-component kernel factory.

The reciprocal analogue of ``ewald_real_factory``. Unlike the real-space side --
where every pair is independent and one atom-major kernel covers the whole
derivative matrix -- the reciprocal sum has a **global reduction barrier**: per-atom
energy / forces / charge-grad need the structure factors
which sum over *all* atoms:

.. math::

    S(k) = G(k) \sum_j q_j e^{i k \cdot R_j}.

A specialization is
therefore irreducibly **multi-stage** and ``make_ewald_recip_kernel`` returns a small
private bundle rather than a single ``wp.Kernel``:

* **fill** -- ``S(k)`` + ``cos(k.r)`` + ``sin(k.r)`` (k-major). Reuses the
  hand-written ``_ewald_reciprocal_space_energy_kernel_fill_structure_factors``
  (single) / ``_batch_...`` (batch) wholesale, so ``S(k)`` is bit-exact for free.
* **compute** -- E / F / dE/dq (atom-major). Factory-owned, specialized by
  ``DERIV`` / ``order`` via compile-time constants (dead-branch elimination), built
  from the same per-``k`` arithmetic as the hand-written compute kernels so forward
  outputs are bit-exact parity oracles.
* **virial** -- the ``cell_grad`` output (k-major, ``dim=[K]``, from ``|S(k)|^2``).
  Reuses the hand-written ``_ewald_reciprocal_space_virial_kernel`` (single) /
  ``_batch_...`` (batch); ``None`` unless ``cell_grad`` (it is *not* an atom-major
  branch, so ``cell_grad`` selects a kernel, not a branch).

The reciprocal kernels here compute the **k-space sum only**; self-energy and
background corrections are a separate higher-level kernel (as in the hand-written
path) and are excluded from both the factory output and the parity / finite-diff
oracles. Charge-grad has no ``1/2``:

.. math::

    \frac{\partial E}{\partial q_i} = \phi_i

while the energy carries the ``1/2``.

``order`` families
------------------
* ``order="forward"`` -- ``E`` / ``E_F`` / ``E_F_dQ`` from precomputed ``S(k)``,
  plus the optional ``cell_grad`` virial kernel.
* ``order="backward"`` -- the same compute core scaled by the upstream per-system
  energy cotangent ``grad_energy[isys]``: ``grad_R = ge dE/dR``, ``grad_q = ge phi``.
  The ``cell_grad`` virial likewise has ``grad_energy`` baked in (output ``ge * W``),
  matching the real-space backward kernel which scales its per-pair virial by
  ``grad_energy``. The reused forward virial kernel cannot take ``grad_energy``, so
  the backward path uses a factory-owned k-major variant
  (:func:`_make_backward_virial_kernel`).
* ``order="double_backward"`` -- the second-derivative node, recompute mode: a
  k-major reduce kernel recomputes the per-``k`` cotangent sums from the k-vectors +
  cotangents, and an atom-major kernel contracts the stored per-``k`` sums into
  ``grad_positions`` / ``grad_charges`` / ``grad_grad_energy``. With ``cell_grad`` the
  reduce stage also emits the cell-input second-order grads
  ``grad_kvectors`` / ``grad_volume``.

Cell second-order: the reciprocal kernel never receives the integer Miller
indices, so the following derivatives are structurally Torch-owned:

.. math::

    h \mapsto k(h), \qquad h \mapsto V(h).

The kernel therefore owns second derivatives w.r.t.
its differentiable inputs ``k_vectors`` (vec3 per k) and ``volume`` (scalar per
system), emitting ``grad_kvectors`` / ``grad_volume`` (mirroring PME's
``grad_k_squared`` / ``grad_volume``); Torch maps those back to ``cell``. The
``order="backward"`` first-derivative cell-input kernel is the bundle's ``kspace`` slot;
the ``double_backward`` node emits the same cell-input grads from its reduce stage
plus the k/V<->position/charge cross terms. ``v_cell`` / ``grad_cell`` stay
sentinel-only on the recip path (real-space holds ``cell`` directly; recip does not).
"""

from functools import lru_cache
from typing import NamedTuple

import warp as wp

from nvalchemiops.interactions.electrostatics._factory_common import (
    _DTYPE_INFO,
    _K_SQUARED_EPSILON,
    _alloc_sentinels,
    _deriv_token,
    _DerivState,
    _make_specialization_module_name,
    _name_and_document_kernel,
    _require_component,
    _require_supported_dtype,
    _validate_common_axes,
)
from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    EIGHTPI,
    RECIP_TILED_BLOCK_DIM,
    _batch_ewald_energy_corrections_backward_kernel,
    _batch_ewald_energy_corrections_double_backward_kernel,
    _batch_ewald_energy_corrections_kernel,
    _batch_ewald_reciprocal_space_energy_forces_charge_grad_kernel,
    _batch_ewald_reciprocal_space_energy_forces_kernel,
    _batch_ewald_reciprocal_space_energy_kernel_compute_energy,
    _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors,
    _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_tiled,
    _batch_ewald_reciprocal_space_virial_kernel,
    _batch_ewald_subtract_self_energy_kernel,
    _ewald_energy_corrections_backward_kernel,
    _ewald_energy_corrections_double_backward_kernel,
    _ewald_energy_corrections_kernel,
    _ewald_reciprocal_space_energy_forces_charge_grad_kernel,
    _ewald_reciprocal_space_energy_forces_kernel,
    _ewald_reciprocal_space_energy_kernel_compute_energy,
    _ewald_reciprocal_space_energy_kernel_fill_structure_factors,
    _ewald_reciprocal_space_energy_kernel_fill_structure_factors_tiled,
    _ewald_reciprocal_space_virial_kernel,
    _ewald_subtract_self_energy_kernel,
)
from nvalchemiops.math import wp_exp_kernel

__all__ = [
    "alloc_ewald_recip_sentinels",
    "get_ewald_recip_kernel",
    "make_ewald_recip_kernel",
]


class _RecipKernels(NamedTuple):
    """Bundle of Warp kernels for one ``ewald_recip`` specialization.

    Attributes
    ----------
    fill : wp.Kernel
        Structure-factor fill (k-major); reused hand-written kernel. For
        ``order="double_backward"`` this is the k-major *reduce* kernel that
        recomputes the per-``k`` cotangent sums instead.
    compute : wp.Kernel
        Atom-major compute kernel (E / F / dq, or the double-backward contraction).
    virial : wp.Kernel or None
        k-major *strain-space* virial kernel (the legacy direct ``compute_virial``
        output ``W = -dE/dstrain``); ``None`` unless ``cell_grad`` is set (and only
        for the ``forward`` / ``backward`` orders).
    kspace : wp.Kernel or None
        k-major *cell-input* first-derivative kernel emitting ``grad_kvectors``
        (per-``k`` vec3) + ``grad_volume`` (per-system scalar) -- the cell-side grads
        the kernel actually owns, which Torch maps back to ``cell`` via the
        ``k = 2 pi (cell^{-1})^T m`` / ``V = det(cell)`` chain (cf. PME's
        ``grad_volume`` / ``grad_k_squared``). ``None`` unless ``cell_grad`` and
        ``order="backward"``. The double-backward node emits the same
        cell-input grads directly from its reduce stage.
    """

    fill: wp.Kernel
    compute: wp.Kernel
    virial: wp.Kernel | None
    kspace: wp.Kernel | None = None


# === Sentinel allocators ===


# slot_spec: name -> (shape, dtype_token); ``vec`` / ``mat`` resolve against
# ``wp_dtype``. Must match each kernel slot's dtype exactly.
_SENTINEL_SLOT_SPEC: dict[str, tuple[tuple[int, ...], str]] = {
    # compute inputs / outputs
    "reciprocal_energies": ((0,), "f64"),
    "atomic_forces": ((0,), "vec"),
    "charge_gradients": ((0,), "f64"),
    # backward / double-backward cotangents + outputs
    "grad_energy": ((0,), "f64"),
    "grad_grad_energy": ((0,), "f64"),
    "v_pos": ((0,), "vec"),
    "v_charge": ((0,), "f64"),
    "v_cell": ((0,), "mat"),
    "grad_positions": ((0,), "vec"),
    "grad_charges": ((0,), "f64"),
    "grad_cell": ((0,), "mat"),
    # cell-input (k-vector / volume) cotangents + outputs -- the cell second-order
    # slots the recip kernel owns. ``v_kvectors`` is per-(system, k),
    # ``v_volume`` per system; outputs mirror them.
    "v_kvectors": ((0, 0), "vec"),
    "v_volume": ((0,), "f64"),
    "grad_kvectors": ((0, 0), "vec"),
    "grad_volume": ((0,), "f64"),
}


_SENTINEL_CACHE: dict[tuple[type, str], dict[str, wp.array]] = {}


def alloc_ewald_recip_sentinels(wp_dtype: type, device: str) -> dict[str, wp.array]:
    """Return cached zero-size sentinel arrays for inactive ``ewald_recip`` slots.

    Callers pass these for any input/output slot not active in the current
    specialization; the kernel's compile-time branches never index them. Memoized
    per ``(wp_dtype, device)`` (the sentinels are zero-size and read-only) so the
    backward, launched every training step, does not re-allocate them per call.

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


def _ewald_recip_module_name(
    wp_dtype: type, batched: bool, order: str = "forward"
) -> str:
    """Deterministic Warp ``module=`` name for one ``ewald_recip`` specialization."""
    return _make_specialization_module_name(
        "ewald_recip", wp_dtype=wp_dtype, batched=batched, order=order
    )


def _name_and_document(
    kernel: wp.Kernel,
    *,
    base: str,
    wp_dtype: type,
    batched: bool,
    deriv_state: _DerivState | None,
    cell_grad: bool,
    order: str,
) -> None:
    """Give a generated ``ewald_recip`` kernel a descriptive name + spec docstring."""
    features = [
        _deriv_token(deriv_state),
        "cellgrad" if cell_grad else "",
        "batch" if batched else "single",
        "" if order == "forward" else order,
    ]
    entries: list[tuple[str, object]] = [("batched", bool(batched))]
    if deriv_state is not None:
        entries.append(("deriv_state", deriv_state.name))
    entries += [("cell_grad", bool(cell_grad)), ("order", order)]
    _name_and_document_kernel(
        kernel,
        base=base,
        wp_dtype=wp_dtype,
        features=features,
        entries=entries,
    )


# === Axis validation shared by every order ===


def _validate_axes(
    wp_dtype: type,
    batched: bool,
    neighbor_input: str,
    deriv_state: _DerivState,
    cell_grad: bool,
    order: str,
) -> None:
    """Raise for unsupported / invalid axis combinations.

    Mirrors the real-space validation: ``NotImplementedError`` for out-of-scope axes;
    ``ValueError`` for the permanently invalid ``cell_grad=True`` +
    ``deriv_state=E`` combination and for derivative orders requesting
    ``deriv_state=E``.
    """
    _validate_common_axes(
        wp_dtype,
        deriv_state=deriv_state,
        cell_grad=cell_grad,
        order=order,
        component="ewald_recip",
    )
    if deriv_state is _DerivState.E_dQ:
        raise NotImplementedError(
            "ewald_recip does not implement the charge-only E_dQ specialization; "
            "use E_F_dQ when reciprocal charge gradients are needed."
        )
    if neighbor_input != "none":
        raise NotImplementedError(
            f"ewald_recip factory is k-vector based; neighbor_input must be 'none', "
            f"got {neighbor_input!r}"
        )


# === ewald_recip kernel factory ===


@lru_cache(maxsize=None)
def make_ewald_recip_kernel(
    wp_dtype: type,
    *,
    batched: bool = False,
    neighbor_input: str = "none",
    deriv_state: _DerivState = _DerivState.E,
    cell_grad: bool = False,
    order: str = "forward",
    tiled: bool = False,
) -> _RecipKernels:
    """Return a cached, specialized ``ewald_recip`` kernel bundle.

    Unlike :func:`make_ewald_real_kernel`, the reciprocal sum's global reduction
    barrier forces a multi-stage specialization, so this returns a
    private bundle (``fill``, ``compute``, ``virial``) rather than a
    single ``wp.Kernel``. The compute body branches on the Python compile-time
    constants ``BATCHED`` / ``DERIV``; Warp dead-eliminates the unused branches. The
    ``fill`` and ``virial`` members reuse the hand-written kernels for bit-exact
    ``S(k)`` / virial parity.

    Parameters
    ----------
    wp_dtype : type
        ``wp.float32`` or ``wp.float64``.
    batched : bool
        Single-system (``False``) vs batched (``True``).
    neighbor_input : {"none"}
        Reciprocal kernels are k-vector based; must be ``"none"``.
    deriv_state : _DerivState
        ``E`` (forward only), ``E_F`` (+forces), ``E_F_dQ`` (+charge gradient).
    cell_grad : bool
        Single compile-time switch for the optional virial output. Valid only with
        ``deriv_state`` in ``{E_F, E_F_dQ}``.
    order : {"forward", "backward", "double_backward"}
        Forward output, first-derivative autograd node, or second-derivative node.
    """
    _validate_axes(wp_dtype, batched, neighbor_input, deriv_state, cell_grad, order)

    if order == "double_backward":
        return _make_double_backward_kernels(
            wp_dtype, batched, deriv_state, cell_grad, tiled=tiled
        )

    fill = get_ewald_recip_component_kernel(wp_dtype, component="fill", batched=batched)
    compute = _make_compute_kernel(wp_dtype, batched, deriv_state, order)
    virial = None
    kspace = None
    if cell_grad:
        if order == "backward":
            # The backward virial emits ge * W (grad_energy baked in, matching the
            # real-space backward path that scales its per-pair virial by grad_energy).
            # The reused forward kernel cannot take grad_energy, so the backward path
            # uses a factory-owned variant.
            virial = _make_backward_virial_kernel(wp_dtype, batched)
            # The cell-input first derivatives the kernel actually owns: grad_kvectors
            # (vec3 per k) + grad_volume (scalar per system). Torch maps these to
            # grad_cell via the cell->k / cell->V chain.
            kspace = _make_backward_kspace_kernel(wp_dtype, batched)
        else:
            virial = get_ewald_recip_component_kernel(
                wp_dtype, component="virial", batched=batched
            )
    return _RecipKernels(fill=fill, compute=compute, virial=virial, kspace=kspace)


def get_ewald_recip_kernel(
    wp_dtype: type,
    *,
    batched: bool = False,
    neighbor_input: str = "none",
    deriv_state: _DerivState = _DerivState.E,
    cell_grad: bool = False,
    order: str = "forward",
    component: str = "ewald_recip",
    tiled: bool = False,
) -> _RecipKernels:
    """Return a cached ``ewald_recip`` kernel bundle, validating dtype + component.

    Validates the dtype and ``component`` (the only specialization axis not also a
    :func:`make_ewald_recip_kernel` argument), then delegates to that factory.
    Memoization is the ``@lru_cache`` on ``make_*``; every argument is forwarded **by
    keyword** so a positional vs keyword call can never produce a duplicate cache
    entry.
    """
    _require_supported_dtype(wp_dtype)
    _require_component(component, "ewald_recip")
    return make_ewald_recip_kernel(
        wp_dtype=wp_dtype,
        batched=batched,
        neighbor_input=neighbor_input,
        deriv_state=deriv_state,
        cell_grad=cell_grad,
        order=order,
        tiled=tiled,
    )


@lru_cache(maxsize=None)
def get_ewald_recip_component_kernel(
    wp_dtype: type,
    *,
    component: str,
    batched: bool = False,
    tiled: bool = False,
) -> wp.Kernel:
    """Return a typed low-level Ewald reciprocal kernel.

    This accessor covers legacy public-launcher signatures that are not the
    bundled atom-major factory shape, notably single-system 1D ``k_vectors`` /
    structure-factor arrays and the self/background correction kernels.
    """
    _require_supported_dtype(wp_dtype)
    info = _DTYPE_INFO[wp_dtype]
    arrays = {
        "f": wp.array(dtype=wp_dtype),
        "f64": wp.array(dtype=wp.float64),
        "i32": wp.array(dtype=wp.int32),
        "v": wp.array(dtype=info.vec),
        "m": wp.array(dtype=info.mat),
        "f64_2d": wp.array2d(dtype=wp.float64),
        "v_2d": wp.array2d(dtype=info.vec),
    }
    fill_kernel = (
        _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_tiled
        if batched and tiled
        else _ewald_reciprocal_space_energy_kernel_fill_structure_factors_tiled
        if tiled
        else _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors
        if batched
        else _ewald_reciprocal_space_energy_kernel_fill_structure_factors
    )
    specs = {
        ("fill", False): (
            fill_kernel,
            ("v", "f", "v", "m", "f", "f64", "f64_2d", "f64_2d", "f64", "f64"),
        ),
        ("fill", True): (
            fill_kernel,
            (
                "v",
                "f",
                "v_2d",
                "m",
                "f",
                "i32",
                "i32",
                "f64",
                "f64_2d",
                "f64_2d",
                "f64_2d",
                "f64_2d",
            ),
        ),
        ("compute_energy", False): (
            _ewald_reciprocal_space_energy_kernel_compute_energy,
            ("f", "f64_2d", "f64_2d", "f64", "f64", "f64"),
        ),
        ("compute_energy", True): (
            _batch_ewald_reciprocal_space_energy_kernel_compute_energy,
            ("f", "i32", "f64_2d", "f64_2d", "f64_2d", "f64_2d", "f64"),
        ),
        ("compute_energy_forces", False): (
            _ewald_reciprocal_space_energy_forces_kernel,
            ("f", "v", "f64_2d", "f64_2d", "f64", "f64", "f64", "v"),
        ),
        ("compute_energy_forces", True): (
            _batch_ewald_reciprocal_space_energy_forces_kernel,
            ("f", "i32", "v_2d", "f64_2d", "f64_2d", "f64_2d", "f64_2d", "f64", "v"),
        ),
        ("compute_energy_forces_charge_grad", False): (
            _ewald_reciprocal_space_energy_forces_charge_grad_kernel,
            ("f", "v", "f64_2d", "f64_2d", "f64", "f64", "f64", "v", "f64"),
        ),
        ("compute_energy_forces_charge_grad", True): (
            _batch_ewald_reciprocal_space_energy_forces_charge_grad_kernel,
            (
                "f",
                "i32",
                "v_2d",
                "f64_2d",
                "f64_2d",
                "f64_2d",
                "f64_2d",
                "f64",
                "v",
                "f64",
            ),
        ),
        ("subtract_self", False): (
            _ewald_subtract_self_energy_kernel,
            ("f", "f", "f64", "f64", "f64"),
        ),
        ("subtract_self", True): (
            _batch_ewald_subtract_self_energy_kernel,
            ("f", "i32", "f", "f64", "f64", "f64"),
        ),
        ("corrections", False): (
            _ewald_energy_corrections_kernel,
            ("f", "f", "f", "f", "f", "f"),
        ),
        ("corrections", True): (
            _batch_ewald_energy_corrections_kernel,
            ("f", "f", "i32", "f", "f", "f", "f"),
        ),
        ("corrections_backward", False): (
            _ewald_energy_corrections_backward_kernel,
            ("f", "f", "f", "f", "f", "f", "f", "f", "f", "f", "f"),
        ),
        ("corrections_backward", True): (
            _batch_ewald_energy_corrections_backward_kernel,
            ("f", "f", "f", "i32", "f", "f", "f", "f", "f", "f", "f", "f"),
        ),
        ("corrections_double_backward", False): (
            _ewald_energy_corrections_double_backward_kernel,
            (
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
            ),
        ),
        ("corrections_double_backward", True): (
            _batch_ewald_energy_corrections_double_backward_kernel,
            (
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "i32",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
            ),
        ),
        ("virial", False): (
            _ewald_reciprocal_space_virial_kernel,
            ("v", "f", "f64", "f64", "f64", "m"),
        ),
        ("virial", True): (
            _batch_ewald_reciprocal_space_virial_kernel,
            ("v_2d", "f", "f64", "f64_2d", "f64_2d", "m"),
        ),
    }
    try:
        kernel, signature = specs[(component, batched)]
    except KeyError as exc:
        raise NotImplementedError(
            f"Unsupported ewald_recip component {component!r}"
        ) from exc
    return wp.overload(kernel, [arrays[token] for token in signature])


# === order="forward" / "backward" compute builder (atom-major) ===


def _make_compute_kernel(
    wp_dtype: type,
    batched: bool,
    deriv_state: _DerivState,
    order: str,
) -> wp.Kernel:
    """Build the atom-major reciprocal compute kernel.

    Consumes the precomputed structure factors ``S_real = G(k) A`` /
    ``S_imag = G(k) B`` and unweighted phases ``cos(k.r)`` / ``sin(k.r)`` from the
    fill stage. Per atom ``i``, summed over ``k`` (k-sum only -- self/background are a
    separate higher-level kernel):

    * ``E_i = (1/2) q_i sum_k (S_real cos_i + S_imag sin_i)``
    * ``F_i = q_i sum_k k (S_real sin_i - S_imag cos_i)`` (``= -dE/dR_i``)
    * ``dE/dq_i = sum_k (S_real cos_i + S_imag sin_i)`` (potential ``phi_i``, no 1/2)

    For ``order="backward"`` every emitted quantity is scaled by the per-system
    upstream energy cotangent ``grad_energy[isys]`` and written as a gradient:
    ``grad_R_i = ge dE/dR_i`` (note ``dE/dR``, the negative of the physical force),
    ``grad_q_i = ge dE/dq_i``. ``reciprocal_energies`` is written only by the forward
    order (it is a sentinel slot in backward).
    """
    info = _DTYPE_INFO[wp_dtype]
    vec_dtype = info.vec

    BATCHED = bool(batched)
    DERIV = deriv_state.value
    BACKWARD = order == "backward"
    E_F = _DerivState.E_F.value
    E_F_dQ = _DerivState.E_F_dQ.value

    module_name = _ewald_recip_module_name(wp_dtype, BATCHED, order)

    @wp.kernel(module=module_name)
    def _ewald_recip_compute(
        charges: wp.array(dtype=wp_dtype),
        batch_id: wp.array(dtype=wp.int32),
        k_vectors: wp.array2d(dtype=vec_dtype),
        cos_k_dot_r: wp.array2d(dtype=wp.float64),
        sin_k_dot_r: wp.array2d(dtype=wp.float64),
        real_structure_factors: wp.array2d(dtype=wp.float64),
        imag_structure_factors: wp.array2d(dtype=wp.float64),
        grad_energy: wp.array(dtype=wp.float64),
        reciprocal_energies: wp.array(dtype=wp.float64),
        atomic_forces: wp.array(dtype=vec_dtype),
        charge_gradients: wp.array(dtype=wp.float64),
    ) -> None:
        """Atom-major reciprocal compute (energy + optional forces / charge-grad).

        ``k_vectors`` / ``real_structure_factors`` / ``imag_structure_factors`` are
        2D ``(S, K)`` arrays indexed by ``isys`` (``0`` when single) so one body
        serves both single and batched via the ``BATCHED`` constant.
        """
        atom_idx = wp.tid()
        charge = wp.float64(charges[atom_idx])
        if BATCHED:
            isys = batch_id[atom_idx]
        else:
            isys = wp.int32(0)

        num_k = real_structure_factors.shape[1]
        if num_k == 0:
            if not BACKWARD:
                reciprocal_energies[atom_idx] = wp.float64(0.0)
            if DERIV >= E_F:
                atomic_forces[atom_idx] = vec_dtype(
                    wp_dtype(0.0),
                    wp_dtype(0.0),
                    wp_dtype(0.0),
                )
                if DERIV >= E_F_dQ:
                    charge_gradients[atom_idx] = wp.float64(0.0)
            return

        local_potential = wp.float64(0.0)  # weighted (q_i phi_i)
        local_potential_uncharged = wp.float64(0.0)  # phi_i
        local_force_x = wp.float64(0.0)
        local_force_y = wp.float64(0.0)
        local_force_z = wp.float64(0.0)

        # The energy accumulation intentionally follows the energy-only ordering
        # for all derivative states. This lets force-only precompute use E_F
        # without silently changing the energy value or doing charge-gradient work.
        # The E_F force arithmetic still mirrors the hand-written force kernel's
        # charge-folded order for force parity.
        for k_idx in range(num_k):
            s_real = real_structure_factors[isys, k_idx]
            s_imag = imag_structure_factors[isys, k_idx]

            if DERIV == E_F:
                cos_kr_raw = cos_k_dot_r[k_idx, atom_idx]
                sin_kr_raw = sin_k_dot_r[k_idx, atom_idx]
                local_potential += charge * (s_real * cos_kr_raw + s_imag * sin_kr_raw)
                cos_kr = charge * cos_kr_raw
                sin_kr = charge * sin_kr_raw
                force_scalar = s_real * sin_kr - s_imag * cos_kr
                k_vec = k_vectors[isys, k_idx]
                local_force_x += force_scalar * wp.float64(k_vec[0])
                local_force_y += force_scalar * wp.float64(k_vec[1])
                local_force_z += force_scalar * wp.float64(k_vec[2])
            else:
                cos_kr = cos_k_dot_r[k_idx, atom_idx]
                sin_kr = sin_k_dot_r[k_idx, atom_idx]
                phase_sum = s_real * cos_kr + s_imag * sin_kr
                local_potential += charge * phase_sum
                if DERIV >= E_F_dQ:
                    local_potential_uncharged += phase_sum
                    force_scalar = charge * (s_real * sin_kr - s_imag * cos_kr)
                    k_vec = k_vectors[isys, k_idx]
                    local_force_x += force_scalar * wp.float64(k_vec[0])
                    local_force_y += force_scalar * wp.float64(k_vec[1])
                    local_force_z += force_scalar * wp.float64(k_vec[2])

        if BACKWARD:
            ge = grad_energy[isys]
        else:
            ge = wp.float64(1.0)

        if not BACKWARD:
            reciprocal_energies[atom_idx] = wp.float64(0.5) * local_potential

        if DERIV >= E_F:
            if BACKWARD:
                # grad_R = ge * dE/dR = ge * (-F). Physical force F has components
                # (local_force_*); dE/dR = -F.
                atomic_forces[atom_idx] = vec_dtype(
                    wp_dtype(-ge * local_force_x),
                    wp_dtype(-ge * local_force_y),
                    wp_dtype(-ge * local_force_z),
                )
            else:
                atomic_forces[atom_idx] = vec_dtype(
                    wp_dtype(local_force_x),
                    wp_dtype(local_force_y),
                    wp_dtype(local_force_z),
                )
            if DERIV >= E_F_dQ:
                charge_gradients[atom_idx] = ge * local_potential_uncharged

    _name_and_document(
        _ewald_recip_compute,
        base="ewald_recip_compute",
        wp_dtype=wp_dtype,
        batched=batched,
        deriv_state=deriv_state,
        cell_grad=False,
        order=order,
    )
    return _ewald_recip_compute


# === order="backward" virial builder (k-major, grad_energy baked in) ===


def _make_backward_virial_kernel(wp_dtype: type, batched: bool) -> wp.Kernel:
    """Build the backward virial kernel (k-major): emits ``ge * W``.

    Mirrors the hand-written ``_ewald_reciprocal_space_virial_kernel`` math
    (``W_ab(k) = E(k) (delta_ab - kfac k_a k_b)`` with ``E(k) = |S|^2 / (2 G)``) but
    multiplies the per-``k`` contribution by the per-system upstream energy cotangent
    ``grad_energy[isys]``, so the output is the same ``dL/dcell``-as-virial state
    shape the real-space backward kernel emits (its per-pair virial is likewise
    scaled by ``grad_energy``).
    The forward order keeps the reused unscaled kernel; only ``order="backward"`` uses
    this variant. Single vs batched differ only in the ``(S, K)`` indexing of the
    k-major arrays, selected by the ``BATCHED`` compile-time constant.
    """
    info = _DTYPE_INFO[wp_dtype]
    vec_dtype = info.vec
    mat_dtype = info.mat

    BATCHED = bool(batched)
    module_name = _ewald_recip_module_name(wp_dtype, BATCHED, "backward_virial")

    @wp.kernel(module=module_name)
    def _ewald_recip_backward_virial(
        k_vectors: wp.array2d(dtype=vec_dtype),
        alpha: wp.array(dtype=wp_dtype),
        volume: wp.array(dtype=wp.float64),
        real_structure_factors: wp.array2d(dtype=wp.float64),
        imag_structure_factors: wp.array2d(dtype=wp.float64),
        grad_energy: wp.array(dtype=wp.float64),
        virial: wp.array(dtype=mat_dtype),
    ) -> None:
        """k-major: accumulate ``ge[isys] * W(k)`` into ``virial[isys]``."""
        if BATCHED:
            k_idx, isys = wp.tid()
        else:
            k_idx = wp.tid()
            isys = wp.int32(0)

        k_vec = k_vectors[isys, k_idx]
        kx = wp.float64(k_vec[0])
        ky = wp.float64(k_vec[1])
        kz = wp.float64(k_vec[2])
        k_sq = kx * kx + ky * ky + kz * kz
        if k_sq < wp.float64(_K_SQUARED_EPSILON):
            return

        alpha_ = wp.float64(alpha[isys])
        vol = volume[isys]
        s_real = real_structure_factors[isys, k_idx]
        s_imag = imag_structure_factors[isys, k_idx]
        ge = grad_energy[isys]

        s_sq = s_real * s_real + s_imag * s_imag
        exp_factor = wp.float64(0.25) / (alpha_ * alpha_)
        green = wp.float64(EIGHTPI) / vol * wp.exp(-k_sq * exp_factor) / k_sq
        energy_k = ge * wp.float64(0.5) * s_sq / green

        k_factor = wp.float64(2.0) * (wp.float64(1.0) + k_sq * exp_factor) / k_sq

        w00 = energy_k * (wp.float64(1.0) - k_factor * kx * kx)
        w01 = energy_k * (-k_factor * kx * ky)
        w02 = energy_k * (-k_factor * kx * kz)
        w10 = energy_k * (-k_factor * ky * kx)
        w11 = energy_k * (wp.float64(1.0) - k_factor * ky * ky)
        w12 = energy_k * (-k_factor * ky * kz)
        w20 = energy_k * (-k_factor * kz * kx)
        w21 = energy_k * (-k_factor * kz * ky)
        w22 = energy_k * (wp.float64(1.0) - k_factor * kz * kz)

        virial_k = mat_dtype(
            type(k_vec[0])(w00),
            type(k_vec[0])(w01),
            type(k_vec[0])(w02),
            type(k_vec[0])(w10),
            type(k_vec[0])(w11),
            type(k_vec[0])(w12),
            type(k_vec[0])(w20),
            type(k_vec[0])(w21),
            type(k_vec[0])(w22),
        )
        wp.atomic_add(virial, isys, virial_k)

    _name_and_document(
        _ewald_recip_backward_virial,
        base="ewald_recip_backward_virial",
        wp_dtype=wp_dtype,
        batched=batched,
        deriv_state=None,
        cell_grad=True,
        order="backward",
    )
    return _ewald_recip_backward_virial


# === order="backward" cell-input k-space builder (k-major) ===


def _make_backward_kspace_kernel(wp_dtype: type, batched: bool) -> wp.Kernel:
    """Build the backward cell-input kernel (k-major): ``grad_kvectors`` + ``grad_volume``.

    The reciprocal kernel never sees the integer Miller indices ``m``; it receives
    only the already-Cartesian ``k_vectors`` and the volume. It therefore cannot form
    ``dk/dcell`` / ``dV/dcell`` -- those live in Torch (``k_vectors.py`` /
    ``det(cell)``). The cell-side derivatives the kernel *does* own are w.r.t. its
    differentiable inputs ``k_vectors`` (vec3 per k) and ``volume`` (scalar per
    system), mirroring PME's ``grad_k_squared`` / ``grad_volume``. Torch then maps
    these to ``grad_cell``.

    With ``E = (1/2) sum_k g_k (A^2 + B^2)``, ``g_k = (8 pi / V) e^{-k^2/4a^2}/k^2``,
    ``A = sum_i q_i cos(k.r_i)``, ``B = sum_i q_i sin(k.r_i)`` and the per-``k`` vector
    sums ``Ra = sum_i q_i cos_i r_i``, ``Rb = sum_i q_i sin_i r_i``:

      dE/dk = g_k [ B Ra - A Rb - (1/2) mu (A^2 + B^2) k ]
      dE/dV = -(1/2) g_k (A^2 + B^2) / V             (= -E_k / V)

    with ``mu = 1/(2 a^2) + 2/k^2`` (the log-derivative magnitude of ``g_k`` w.r.t.
    ``k``: ``dg/dk = -g mu k``). Both outputs are scaled by the per-system upstream
    energy cotangent ``grad_energy[isys]`` (``ge * dE/dk``, ``ge * dE/dV``), matching
    the backward force / charge-grad scaling. Single vs batched differ only in the
    ``(S, K)`` indexing, selected by the ``BATCHED`` compile-time constant.
    """
    info = _DTYPE_INFO[wp_dtype]
    vec_dtype = info.vec

    BATCHED = bool(batched)
    module_name = _ewald_recip_module_name(wp_dtype, BATCHED, "backward_kspace")

    @wp.kernel(module=module_name)
    def _ewald_recip_backward_kspace(
        positions: wp.array(dtype=vec_dtype),
        charges: wp.array(dtype=wp_dtype),
        k_vectors: wp.array2d(dtype=vec_dtype),
        alpha: wp.array(dtype=wp_dtype),
        volume: wp.array(dtype=wp.float64),
        batch_id: wp.array(dtype=wp.int32),
        atom_start: wp.array(dtype=wp.int32),
        atom_end: wp.array(dtype=wp.int32),
        grad_energy: wp.array(dtype=wp.float64),
        grad_kvectors: wp.array2d(dtype=vec_dtype),
        grad_volume: wp.array(dtype=wp.float64),
    ) -> None:
        """k-major: emit ``ge * dE/dk`` (per k) and accumulate ``ge * dE/dV``."""
        if BATCHED:
            k_idx, isys = wp.tid()
        else:
            k_idx = wp.tid()
            isys = wp.int32(0)

        k_vec = k_vectors[isys, k_idx]
        kx = wp.float64(k_vec[0])
        ky = wp.float64(k_vec[1])
        kz = wp.float64(k_vec[2])
        k_squared = kx * kx + ky * ky + kz * kz
        if k_squared < wp.float64(_K_SQUARED_EPSILON):
            return

        alpha_ = wp.float64(alpha[isys])
        exp_factor = wp.float64(0.25) / (alpha_ * alpha_)
        vol = volume[isys]
        g_k = wp_exp_kernel(k_squared, exp_factor) * wp.float64(EIGHTPI) / vol
        ge = grad_energy[isys]

        a_start = wp.int32(0)
        a_end = positions.shape[0]
        if BATCHED:
            a_start = atom_start[isys]
            a_end = atom_end[isys]

        a_sum = wp.float64(0.0)
        b_sum = wp.float64(0.0)
        ra_x = wp.float64(0.0)
        ra_y = wp.float64(0.0)
        ra_z = wp.float64(0.0)
        rb_x = wp.float64(0.0)
        rb_y = wp.float64(0.0)
        rb_z = wp.float64(0.0)
        for atom_idx in range(a_start, a_end):
            position = positions[atom_idx]
            rx = wp.float64(position[0])
            ry = wp.float64(position[1])
            rz = wp.float64(position[2])
            qi = wp.float64(charges[atom_idx])
            k_dot_r = kx * rx + ky * ry + kz * rz
            cos_kr = wp.cos(k_dot_r)
            sin_kr = wp.sin(k_dot_r)
            qc = qi * cos_kr
            qs = qi * sin_kr
            a_sum += qc
            b_sum += qs
            ra_x += qc * rx
            ra_y += qc * ry
            ra_z += qc * rz
            rb_x += qs * rx
            rb_y += qs * ry
            rb_z += qs * rz

        s_sq = a_sum * a_sum + b_sum * b_sum
        mu = wp.float64(2.0) * exp_factor + wp.float64(2.0) / k_squared

        # dE/dk = g_k [ B Ra - A Rb - 1/2 mu S k ]
        half_mu_s = wp.float64(0.5) * mu * s_sq
        dk_x = g_k * (b_sum * ra_x - a_sum * rb_x - half_mu_s * kx)
        dk_y = g_k * (b_sum * ra_y - a_sum * rb_y - half_mu_s * ky)
        dk_z = g_k * (b_sum * ra_z - a_sum * rb_z - half_mu_s * kz)
        grad_kvectors[isys, k_idx] = type(k_vec)(
            type(k_vec[0])(ge * dk_x),
            type(k_vec[0])(ge * dk_y),
            type(k_vec[0])(ge * dk_z),
        )

        # dE/dV = -E_k / V = -1/2 g_k S / V
        dv = -wp.float64(0.5) * g_k * s_sq / vol
        wp.atomic_add(grad_volume, isys, ge * dv)

    _name_and_document(
        _ewald_recip_backward_kspace,
        base="ewald_recip_backward_kspace",
        wp_dtype=wp_dtype,
        batched=batched,
        deriv_state=None,
        cell_grad=True,
        order="backward",
    )
    return _ewald_recip_backward_kspace


@lru_cache(maxsize=None)
def _make_backward_kspace_from_cache_kernel(wp_dtype: type, batched: bool) -> wp.Kernel:
    """O(K) cell-input backward: ``grad_kvectors`` + ``grad_volume`` from a cached reduction.

    Identical math to :func:`_make_backward_kspace_kernel` (the
    ``dE/dk = g_k[B Ra - A Rb - 1/2 mu S k]``, ``dE/dV = -1/2 g_k S / V`` formula),
    but the per-``k`` un-weighted sums ``A, B, Ra, Rb`` are READ from
    ``cellgrad_cache`` (produced in the forward fill's atom loop --
    ``_ewald_reciprocal_space_energy_kernel_fill_structure_factors_cellgrad``)
    instead of recomputed via an O(K*N) atom loop. Cache layout is row
    ``isys*num_k + k_idx`` with columns ``[A, B, Ra_x, Ra_y, Ra_z, Rb_x, Rb_y, Rb_z]``
    (single-system uses ``isys=0``).
    """
    info = _DTYPE_INFO[wp_dtype]
    vec_dtype = info.vec
    BATCHED = bool(batched)
    module_name = _ewald_recip_module_name(wp_dtype, BATCHED, "backward_kspace_cache")

    @wp.kernel(module=module_name)
    def _ewald_recip_backward_kspace_from_cache(
        k_vectors: wp.array2d(dtype=vec_dtype),
        alpha: wp.array(dtype=wp_dtype),
        volume: wp.array(dtype=wp.float64),
        cellgrad_cache: wp.array2d(dtype=wp.float64),
        grad_energy: wp.array(dtype=wp.float64),
        num_k: wp.int32,
        grad_kvectors: wp.array2d(dtype=vec_dtype),
        grad_volume: wp.array(dtype=wp.float64),
    ) -> None:
        """k-major O(K): emit ``ge * dE/dk`` (per k) and accumulate ``ge * dE/dV``."""
        if BATCHED:
            k_idx, isys = wp.tid()
        else:
            k_idx = wp.tid()
            isys = wp.int32(0)

        k_vec = k_vectors[isys, k_idx]
        kx = wp.float64(k_vec[0])
        ky = wp.float64(k_vec[1])
        kz = wp.float64(k_vec[2])
        k_squared = kx * kx + ky * ky + kz * kz
        if k_squared < wp.float64(_K_SQUARED_EPSILON):
            return

        alpha_ = wp.float64(alpha[isys])
        exp_factor = wp.float64(0.25) / (alpha_ * alpha_)
        vol = volume[isys]
        g_k = wp_exp_kernel(k_squared, exp_factor) * wp.float64(EIGHTPI) / vol
        ge = grad_energy[isys]

        row = isys * num_k + k_idx
        a_sum = cellgrad_cache[row, 0]
        b_sum = cellgrad_cache[row, 1]
        ra_x = cellgrad_cache[row, 2]
        ra_y = cellgrad_cache[row, 3]
        ra_z = cellgrad_cache[row, 4]
        rb_x = cellgrad_cache[row, 5]
        rb_y = cellgrad_cache[row, 6]
        rb_z = cellgrad_cache[row, 7]

        s_sq = a_sum * a_sum + b_sum * b_sum
        mu = wp.float64(2.0) * exp_factor + wp.float64(2.0) / k_squared
        half_mu_s = wp.float64(0.5) * mu * s_sq
        dk_x = g_k * (b_sum * ra_x - a_sum * rb_x - half_mu_s * kx)
        dk_y = g_k * (b_sum * ra_y - a_sum * rb_y - half_mu_s * ky)
        dk_z = g_k * (b_sum * ra_z - a_sum * rb_z - half_mu_s * kz)
        grad_kvectors[isys, k_idx] = type(k_vec)(
            type(k_vec[0])(ge * dk_x),
            type(k_vec[0])(ge * dk_y),
            type(k_vec[0])(ge * dk_z),
        )

        dv = -wp.float64(0.5) * g_k * s_sq / vol
        wp.atomic_add(grad_volume, isys, ge * dv)

    _name_and_document(
        _ewald_recip_backward_kspace_from_cache,
        base="ewald_recip_backward_kspace_cache",
        wp_dtype=wp_dtype,
        batched=batched,
        deriv_state=None,
        cell_grad=True,
        order="backward",
    )
    return _ewald_recip_backward_kspace_from_cache


# === order="double_backward" builders (recompute mode) ===


def _make_double_backward_kernels(
    wp_dtype: type,
    batched: bool,
    deriv_state: _DerivState,
    cell_grad: bool = False,
    tiled: bool = False,
) -> _RecipKernels:
    """Build the second-derivative node (recompute mode).

    Backward emitted ``grad_R_i = ge dE/dR_i``, ``grad_q_i = ge phi_i`` and (when
    ``cell_grad``) the cell-input grads ``grad_kvectors = ge dE/dk`` /
    ``grad_volume = ge dE/dV``. The chain scalar is ``L = ge * Phi`` with

      Phi = sum_i v_pos_i . dE/dR_i + sum_i v_charge_i dE/dq_i
            + sum_k v_kvectors_k . dE/dk_k + v_volume dE/dV

    Writing ``E = (1/2) sum_k g_k (A^2 + B^2)``, ``g_k = (8 pi / V) e^{-k^2/4a^2}/k^2``,
    ``A = sum_i q_i cos_i``, ``B = sum_i q_i sin_i``, ``w_i = v_pos_i . k``,
    ``u_i = v_kvectors_k . r_i``, ``wk = v_kvectors_k . k`` and
    ``mu = 1/(2a^2) + 2/k^2`` (the magnitude of ``dg/dk = -g mu k``), the per-``k`` Phi
    in closed form (verified symbolically + by finite-diff -- see the test module):

      Phi_k = g_k [ (B P - A Q) + (A C + B D) + (B Pu - A Qu)
                    - (1/2) wk mu S - (v_volume/(2V)) S ]

    with ``S = A^2 + B^2`` and the raw per-``k`` sums ``P = sum q_i w_i cos_i``,
    ``Q = sum q_i w_i sin_i``, ``C = sum v_charge_i cos_i``, ``D = sum v_charge_i sin_i``,
    ``Pu = sum q_i u_i cos_i``, ``Qu = sum q_i u_i sin_i`` (``C`` / ``D`` only when
    ``DERIV >= E_F_dQ``; the ``Pu`` / ``Qu`` / ``wk`` / ``v_volume`` terms only when
    ``cell_grad``).

    Every output is a first derivative of the single scalar ``Phi`` (so k<->V symmetry
    holds by construction): ``grad_positions[m] = ge dPhi/dR_m``,
    ``grad_charges[m] = ge dPhi/dq_m``, ``grad_kvectors[k] = ge dPhi/dk_k``,
    ``grad_volume[isys] = ge dPhi/dV``, ``grad_grad_energy[isys] = Phi`` (NOT scaled by
    ``ge``; it is ``dL/d(grad_energy)``).

    The cell-input derivative boundary: the kernel owns derivatives w.r.t. ``k_vectors``
    and ``volume`` only -- it never sees the Miller indices, so ``dk/dcell`` /
    ``dV/dcell`` are Torch's (cf. PME). ``grad_cell`` / ``v_cell`` remain sentinel-only
    for the recip path.

    Two stages avoid the global reduction barrier. The k-major ``reduce`` recomputes the
    raw sums, stores per-(system, k) ``g_k``-scaled ``A,B,C,D,P,Q,Pu,Qu``, and emits the
    k-major outputs ``grad_kvectors`` / ``grad_volume`` and the k/V part of
    ``grad_grad_energy``. The atom-major ``compute`` contracts the stored sums into
    ``grad_positions`` / ``grad_charges`` (pos/charge terms + k/V cross terms).
    """
    info = _DTYPE_INFO[wp_dtype]
    vec_dtype = info.vec

    BATCHED = bool(batched)
    DERIV_DQ = wp.constant(deriv_state is _DerivState.E_F_dQ)
    CELL_GRAD = wp.constant(bool(cell_grad))

    deriv_suffix = "dq" if deriv_state is _DerivState.E_F_dQ else "force"
    cell_suffix = "_cell" if cell_grad else ""
    reduce_suffix = f"double_backward_reduce_{deriv_suffix}{cell_suffix}"
    compute_suffix = f"double_backward_{deriv_suffix}{cell_suffix}"
    reduce_module = _ewald_recip_module_name(wp_dtype, BATCHED, reduce_suffix)
    compute_module = _ewald_recip_module_name(wp_dtype, BATCHED, compute_suffix)

    @wp.kernel(module=reduce_module)
    def _ewald_recip_dbwd_reduce(
        positions: wp.array(dtype=vec_dtype),
        charges: wp.array(dtype=wp_dtype),
        k_vectors: wp.array2d(dtype=vec_dtype),
        cell: wp.array(dtype=info.mat),
        alpha: wp.array(dtype=wp_dtype),
        batch_id: wp.array(dtype=wp.int32),
        atom_start: wp.array(dtype=wp.int32),
        atom_end: wp.array(dtype=wp.int32),
        v_pos: wp.array(dtype=vec_dtype),
        v_charge: wp.array(dtype=wp.float64),
        grad_energy: wp.array(dtype=wp.float64),
        deriv_dq: wp.int32,
        gA: wp.array2d(dtype=wp.float64),
        gB: wp.array2d(dtype=wp.float64),
        gC: wp.array2d(dtype=wp.float64),
        gD: wp.array2d(dtype=wp.float64),
        gP: wp.array2d(dtype=wp.float64),
        gQ: wp.array2d(dtype=wp.float64),
        grad_grad_energy: wp.array(dtype=wp.float64),
        # --- cell second-order: inputs + outputs (sentinels unless cell_grad)
        cell_grad: wp.int32,
        volume: wp.array(dtype=wp.float64),
        v_kvectors: wp.array2d(dtype=vec_dtype),
        v_volume: wp.array(dtype=wp.float64),
        gPu: wp.array2d(dtype=wp.float64),
        gQu: wp.array2d(dtype=wp.float64),
        grad_kvectors: wp.array2d(dtype=vec_dtype),
        grad_volume: wp.array(dtype=wp.float64),
    ) -> None:
        """k-major: recompute the per-(system, k) cotangent sums + Phi (+ k/V grads)."""
        if BATCHED:
            k_idx, isys = wp.tid()
        else:
            k_idx = wp.tid()
            isys = wp.int32(0)

        alpha_ = wp.float64(alpha[isys])
        exp_factor = wp.float64(0.25) / (alpha_ * alpha_)
        if CELL_GRAD:
            vol = volume[isys]
        else:
            vol = wp.float64(wp.abs(wp.determinant(cell[isys])))

        k_vector = k_vectors[isys, k_idx]
        kx = wp.float64(k_vector[0])
        ky = wp.float64(k_vector[1])
        kz = wp.float64(k_vector[2])
        k_squared = kx * kx + ky * ky + kz * kz
        if k_squared < wp.float64(_K_SQUARED_EPSILON):
            return

        g_k = wp_exp_kernel(k_squared, exp_factor) * wp.float64(EIGHTPI) / vol

        a_start = wp.int32(0)
        a_end = positions.shape[0]
        if BATCHED:
            a_start = atom_start[isys]
            a_end = atom_end[isys]

        # k-vector cotangent (for u_i / wk) -- only read under cell_grad.
        vkx = wp.float64(0.0)
        vky = wp.float64(0.0)
        vkz = wp.float64(0.0)
        if CELL_GRAD:
            vk_vec = v_kvectors[isys, k_idx]
            vkx = wp.float64(vk_vec[0])
            vky = wp.float64(vk_vec[1])
            vkz = wp.float64(vk_vec[2])

        a_sum = wp.float64(0.0)
        b_sum = wp.float64(0.0)
        c_sum = wp.float64(0.0)
        d_sum = wp.float64(0.0)
        p_sum = wp.float64(0.0)
        q_sum = wp.float64(0.0)
        pu_sum = wp.float64(0.0)
        qu_sum = wp.float64(0.0)
        # vector sums for dPhi/dk (only used under cell_grad).
        rac_x = wp.float64(0.0)
        rac_y = wp.float64(0.0)
        rac_z = wp.float64(0.0)
        ras_x = wp.float64(0.0)
        ras_y = wp.float64(0.0)
        ras_z = wp.float64(0.0)
        rpc_x = wp.float64(0.0)
        rpc_y = wp.float64(0.0)
        rpc_z = wp.float64(0.0)
        rps_x = wp.float64(0.0)
        rps_y = wp.float64(0.0)
        rps_z = wp.float64(0.0)
        rcc_x = wp.float64(0.0)
        rcc_y = wp.float64(0.0)
        rcc_z = wp.float64(0.0)
        rcs_x = wp.float64(0.0)
        rcs_y = wp.float64(0.0)
        rcs_z = wp.float64(0.0)
        rpuc_x = wp.float64(0.0)
        rpuc_y = wp.float64(0.0)
        rpuc_z = wp.float64(0.0)
        rpus_x = wp.float64(0.0)
        rpus_y = wp.float64(0.0)
        rpus_z = wp.float64(0.0)
        vpc_x = wp.float64(0.0)
        vpc_y = wp.float64(0.0)
        vpc_z = wp.float64(0.0)
        vps_x = wp.float64(0.0)
        vps_y = wp.float64(0.0)
        vps_z = wp.float64(0.0)

        for atom_idx in range(a_start, a_end):
            position = positions[atom_idx]
            rx = wp.float64(position[0])
            ry = wp.float64(position[1])
            rz = wp.float64(position[2])
            qi = wp.float64(charges[atom_idx])
            k_dot_r = kx * rx + ky * ry + kz * rz
            cos_kr = wp.cos(k_dot_r)
            sin_kr = wp.sin(k_dot_r)
            qc = qi * cos_kr
            qs = qi * sin_kr

            a_sum += qc
            b_sum += qs

            vp = v_pos[atom_idx]
            w_i = (
                wp.float64(vp[0]) * kx + wp.float64(vp[1]) * ky + wp.float64(vp[2]) * kz
            )
            p_sum += w_i * qc
            q_sum += w_i * qs

            vq = wp.float64(0.0)
            if DERIV_DQ:
                vq = v_charge[atom_idx]
                c_sum += vq * cos_kr
                d_sum += vq * sin_kr

            if CELL_GRAD:
                u_i = vkx * rx + vky * ry + vkz * rz
                pu_sum += u_i * qc
                qu_sum += u_i * qs
                # vector sums (index over r components).
                rac_x += qc * rx
                rac_y += qc * ry
                rac_z += qc * rz
                ras_x += qs * rx
                ras_y += qs * ry
                ras_z += qs * rz
                rpc_x += w_i * qc * rx
                rpc_y += w_i * qc * ry
                rpc_z += w_i * qc * rz
                rps_x += w_i * qs * rx
                rps_y += w_i * qs * ry
                rps_z += w_i * qs * rz
                rpuc_x += u_i * qc * rx
                rpuc_y += u_i * qc * ry
                rpuc_z += u_i * qc * rz
                rpus_x += u_i * qs * rx
                rpus_y += u_i * qs * ry
                rpus_z += u_i * qs * rz
                vpc_x += qc * wp.float64(vp[0])
                vpc_y += qc * wp.float64(vp[1])
                vpc_z += qc * wp.float64(vp[2])
                vps_x += qs * wp.float64(vp[0])
                vps_y += qs * wp.float64(vp[1])
                vps_z += qs * wp.float64(vp[2])
                if DERIV_DQ:
                    rcc_x += vq * cos_kr * rx
                    rcc_y += vq * cos_kr * ry
                    rcc_z += vq * cos_kr * rz
                    rcs_x += vq * sin_kr * rx
                    rcs_y += vq * sin_kr * ry
                    rcs_z += vq * sin_kr * rz

        gA[isys, k_idx] = g_k * a_sum
        gB[isys, k_idx] = g_k * b_sum
        gP[isys, k_idx] = g_k * p_sum
        gQ[isys, k_idx] = g_k * q_sum
        if DERIV_DQ:
            gC[isys, k_idx] = g_k * c_sum
            gD[isys, k_idx] = g_k * d_sum
        if CELL_GRAD:
            gPu[isys, k_idx] = g_k * pu_sum
            gQu[isys, k_idx] = g_k * qu_sum

        # Phi contribution for this k. ge is applied by the chain at the public layer;
        # grad_grad_energy == dL/d(grad_energy).
        s_sq = a_sum * a_sum + b_sum * b_sum
        # Phi_k base (pos/charge terms); the k/V terms are added under cell_grad below.
        phi_k = b_sum * p_sum - a_sum * q_sum
        if DERIV_DQ:
            phi_k += a_sum * c_sum + b_sum * d_sum

        if CELL_GRAD:
            # mu = magnitude factor in dg/dk = -g mu k;  wk = vk . k;  vV = v_volume.
            mu = wp.float64(2.0) * exp_factor + wp.float64(2.0) / k_squared
            wk = vkx * kx + vky * ky + vkz * kz
            vV = v_volume[isys]
            ge = grad_energy[isys]
            inv_ksq2 = wp.float64(1.0) / (k_squared * k_squared)

            # Yc is the V-independent part of Phi_k/g_k; Phi_k = g_k (Yc - vV/(2V) S).
            y_const = (
                b_sum * p_sum
                - a_sum * q_sum
                + a_sum * c_sum
                + b_sum * d_sum
                + b_sum * pu_sum
                - a_sum * qu_sum
                - wp.float64(0.5) * wk * mu * s_sq
            )
            y_val = y_const - (vV / (wp.float64(2.0) * vol)) * s_sq
            # k/V contribution to Phi_k = g_k Y (g_k folded in at the atomic_add).
            phi_k += (
                b_sum * pu_sum
                - a_sum * qu_sum
                - wp.float64(0.5) * wk * mu * s_sq
                - (vV / (wp.float64(2.0) * vol)) * s_sq
            )

            # grad_kvectors[k] = g_k ( -k mu Y + dY/dk ),  Y = Yc - vV/(2V) S.
            # Building-block derivatives of the per-k sums w.r.t. k (vec3):
            #   dA=-Ras, dB=Rac, dP=VPc-Rps, dQ=VPs+Rpc, dC=-Rcs, dD=Rcc,
            #   dPu=-Rpus, dQu=Rpuc, dwk=vk, dmu=-4 k/k^4, dS=2(B Rac - A Ras).
            ds_x = wp.float64(2.0) * (b_sum * rac_x - a_sum * ras_x)
            ds_y = wp.float64(2.0) * (b_sum * rac_y - a_sum * ras_y)
            ds_z = wp.float64(2.0) * (b_sum * rac_z - a_sum * ras_z)

            dmu_x = -wp.float64(4.0) * kx * inv_ksq2
            dmu_y = -wp.float64(4.0) * ky * inv_ksq2
            dmu_z = -wp.float64(4.0) * kz * inv_ksq2

            # dYc/dk_d, inlined per component d in {x, y, z}. d_a = dA/dk_d, etc.
            # x-component:
            d_a = -ras_x
            d_b = rac_x
            d_p = vpc_x - rps_x
            d_q = vps_x + rpc_x
            d_c = -rcs_x
            d_d = rcc_x
            d_pu = -rpus_x
            d_qu = rpuc_x
            dyc_x = (
                d_b * p_sum
                + b_sum * d_p
                - (d_a * q_sum + a_sum * d_q)
                + d_a * c_sum
                + a_sum * d_c
                + d_b * d_sum
                + b_sum * d_d
                + d_b * pu_sum
                + b_sum * d_pu
                - (d_a * qu_sum + a_sum * d_qu)
                - wp.float64(0.5)
                * (vkx * mu * s_sq + wk * dmu_x * s_sq + wk * mu * ds_x)
            )
            # y-component:
            d_a = -ras_y
            d_b = rac_y
            d_p = vpc_y - rps_y
            d_q = vps_y + rpc_y
            d_c = -rcs_y
            d_d = rcc_y
            d_pu = -rpus_y
            d_qu = rpuc_y
            dyc_y = (
                d_b * p_sum
                + b_sum * d_p
                - (d_a * q_sum + a_sum * d_q)
                + d_a * c_sum
                + a_sum * d_c
                + d_b * d_sum
                + b_sum * d_d
                + d_b * pu_sum
                + b_sum * d_pu
                - (d_a * qu_sum + a_sum * d_qu)
                - wp.float64(0.5)
                * (vky * mu * s_sq + wk * dmu_y * s_sq + wk * mu * ds_y)
            )
            # z-component:
            d_a = -ras_z
            d_b = rac_z
            d_p = vpc_z - rps_z
            d_q = vps_z + rpc_z
            d_c = -rcs_z
            d_d = rcc_z
            d_pu = -rpus_z
            d_qu = rpuc_z
            dyc_z = (
                d_b * p_sum
                + b_sum * d_p
                - (d_a * q_sum + a_sum * d_q)
                + d_a * c_sum
                + a_sum * d_c
                + d_b * d_sum
                + b_sum * d_d
                + d_b * pu_sum
                + b_sum * d_pu
                - (d_a * qu_sum + a_sum * d_qu)
                - wp.float64(0.5)
                * (vkz * mu * s_sq + wk * dmu_z * s_sq + wk * mu * ds_z)
            )

            inv_2vol = vV / (wp.float64(2.0) * vol)
            dy_x = dyc_x - inv_2vol * ds_x
            dy_y = dyc_y - inv_2vol * ds_y
            dy_z = dyc_z - inv_2vol * ds_z

            gk_x = g_k * (-kx * mu * y_val + dy_x)
            gk_y = g_k * (-ky * mu * y_val + dy_y)
            gk_z = g_k * (-kz * mu * y_val + dy_z)
            grad_kvectors[isys, k_idx] = type(k_vector)(
                type(k_vector[0])(ge * gk_x),
                type(k_vector[0])(ge * gk_y),
                type(k_vector[0])(ge * gk_z),
            )

            # grad_volume += ge dPhi_k/dV.  Phi_k = g_k Y, g_k ~ 1/V, only the
            # -vV/(2V) S term carries an explicit extra 1/V:
            #   dPhi_k/dV = -(1/V) g_k y_const + g_k vV S / V^2
            dv = -(wp.float64(1.0) / vol) * g_k * y_const + g_k * vV * s_sq / (
                vol * vol
            )
            wp.atomic_add(grad_volume, isys, ge * dv)

        # grad_grad_energy[isys] += Phi_k (= dL/d(grad_energy), NOT scaled by ge).
        wp.atomic_add(grad_grad_energy, isys, g_k * phi_k)

    @wp.kernel(module=reduce_module)
    def _ewald_recip_dbwd_reduce_tiled(
        positions: wp.array(dtype=vec_dtype),
        charges: wp.array(dtype=wp_dtype),
        k_vectors: wp.array2d(dtype=vec_dtype),
        cell: wp.array(dtype=info.mat),
        alpha: wp.array(dtype=wp_dtype),
        batch_id: wp.array(dtype=wp.int32),
        atom_start: wp.array(dtype=wp.int32),
        atom_end: wp.array(dtype=wp.int32),
        v_pos: wp.array(dtype=vec_dtype),
        v_charge: wp.array(dtype=wp.float64),
        grad_energy: wp.array(dtype=wp.float64),
        deriv_dq: wp.int32,
        gA: wp.array2d(dtype=wp.float64),
        gB: wp.array2d(dtype=wp.float64),
        gC: wp.array2d(dtype=wp.float64),
        gD: wp.array2d(dtype=wp.float64),
        gP: wp.array2d(dtype=wp.float64),
        gQ: wp.array2d(dtype=wp.float64),
        grad_grad_energy: wp.array(dtype=wp.float64),
        cell_grad: wp.int32,
        volume: wp.array(dtype=wp.float64),
        v_kvectors: wp.array2d(dtype=vec_dtype),
        v_volume: wp.array(dtype=wp.float64),
        gPu: wp.array2d(dtype=wp.float64),
        gQu: wp.array2d(dtype=wp.float64),
        grad_kvectors: wp.array2d(dtype=vec_dtype),
        grad_volume: wp.array(dtype=wp.float64),
    ) -> None:
        """Cooperative tiled k-major reduce for non-cell reciprocal HVP."""
        if BATCHED:
            k_idx, isys, lane = wp.tid()
        else:
            k_idx, lane = wp.tid()
            isys = wp.int32(0)

        alpha_ = wp.float64(alpha[isys])
        exp_factor = wp.float64(0.25) / (alpha_ * alpha_)
        vol = wp.float64(wp.abs(wp.determinant(cell[isys])))

        k_vector = k_vectors[isys, k_idx]
        kx = wp.float64(k_vector[0])
        ky = wp.float64(k_vector[1])
        kz = wp.float64(k_vector[2])
        k_squared = kx * kx + ky * ky + kz * kz
        if k_squared < wp.float64(_K_SQUARED_EPSILON):
            return

        g_k = wp_exp_kernel(k_squared, exp_factor) * wp.float64(EIGHTPI) / vol

        a_start = wp.int32(0)
        a_end = positions.shape[0]
        if BATCHED:
            a_start = atom_start[isys]
            a_end = atom_end[isys]

        a_sum = wp.float64(0.0)
        b_sum = wp.float64(0.0)
        c_sum = wp.float64(0.0)
        d_sum = wp.float64(0.0)
        p_sum = wp.float64(0.0)
        q_sum = wp.float64(0.0)

        for atom_tile_start in range(a_start, a_end, RECIP_TILED_BLOCK_DIM):
            atom_idx = atom_tile_start + lane
            if atom_idx < a_end:
                position = positions[atom_idx]
                rx = wp.float64(position[0])
                ry = wp.float64(position[1])
                rz = wp.float64(position[2])
                qi = wp.float64(charges[atom_idx])
                k_dot_r = kx * rx + ky * ry + kz * rz
                cos_kr = wp.cos(k_dot_r)
                sin_kr = wp.sin(k_dot_r)
                qc = qi * cos_kr
                qs = qi * sin_kr

                a_sum += qc
                b_sum += qs

                vp = v_pos[atom_idx]
                w_i = (
                    wp.float64(vp[0]) * kx
                    + wp.float64(vp[1]) * ky
                    + wp.float64(vp[2]) * kz
                )
                p_sum += w_i * qc
                q_sum += w_i * qs

                if DERIV_DQ:
                    vq = v_charge[atom_idx]
                    c_sum += vq * cos_kr
                    d_sum += vq * sin_kr

        a_red = wp.tile_reduce(wp.add, wp.tile(a_sum))
        b_red = wp.tile_reduce(wp.add, wp.tile(b_sum))
        c_red = wp.tile_reduce(wp.add, wp.tile(c_sum))
        d_red = wp.tile_reduce(wp.add, wp.tile(d_sum))
        p_red = wp.tile_reduce(wp.add, wp.tile(p_sum))
        q_red = wp.tile_reduce(wp.add, wp.tile(q_sum))

        if lane == 0:
            a_val = wp.tile_extract(a_red, 0)
            b_val = wp.tile_extract(b_red, 0)
            c_val = wp.tile_extract(c_red, 0)
            d_val = wp.tile_extract(d_red, 0)
            p_val = wp.tile_extract(p_red, 0)
            q_val = wp.tile_extract(q_red, 0)

            gA[isys, k_idx] = g_k * a_val
            gB[isys, k_idx] = g_k * b_val
            gP[isys, k_idx] = g_k * p_val
            gQ[isys, k_idx] = g_k * q_val
            if DERIV_DQ:
                gC[isys, k_idx] = g_k * c_val
                gD[isys, k_idx] = g_k * d_val

            phi_k = b_val * p_val - a_val * q_val
            if DERIV_DQ:
                phi_k += a_val * c_val + b_val * d_val
            wp.atomic_add(grad_grad_energy, isys, g_k * phi_k)

    @wp.kernel(module=compute_module)
    def _ewald_recip_dbwd_compute(
        positions: wp.array(dtype=vec_dtype),
        charges: wp.array(dtype=wp_dtype),
        k_vectors: wp.array2d(dtype=vec_dtype),
        batch_id: wp.array(dtype=wp.int32),
        v_pos: wp.array(dtype=vec_dtype),
        v_charge: wp.array(dtype=wp.float64),
        grad_energy: wp.array(dtype=wp.float64),
        deriv_dq: wp.int32,
        gA: wp.array2d(dtype=wp.float64),
        gB: wp.array2d(dtype=wp.float64),
        gC: wp.array2d(dtype=wp.float64),
        gD: wp.array2d(dtype=wp.float64),
        gP: wp.array2d(dtype=wp.float64),
        gQ: wp.array2d(dtype=wp.float64),
        grad_positions: wp.array(dtype=vec_dtype),
        grad_charges: wp.array(dtype=wp.float64),
        # --- cell second-order: cross-term inputs (sentinels unless cell_grad)
        cell_grad: wp.int32,
        alpha: wp.array(dtype=wp_dtype),
        volume: wp.array(dtype=wp.float64),
        v_kvectors: wp.array2d(dtype=vec_dtype),
        v_volume: wp.array(dtype=wp.float64),
        gPu: wp.array2d(dtype=wp.float64),
        gQu: wp.array2d(dtype=wp.float64),
    ) -> None:
        """atom-major: contract the stored per-k sums into dPhi/dR_m, dPhi/dq_m."""
        atom_idx = wp.tid()
        if BATCHED:
            isys = batch_id[atom_idx]
        else:
            isys = wp.int32(0)
        ge = grad_energy[isys]
        qi = wp.float64(charges[atom_idx])
        position = positions[atom_idx]
        rx = wp.float64(position[0])
        ry = wp.float64(position[1])
        rz = wp.float64(position[2])
        vp = v_pos[atom_idx]
        if DERIV_DQ:
            vqi = v_charge[atom_idx]
        else:
            vqi = wp.float64(0.0)

        exp_factor = wp.float64(0.0)
        inv_vol = wp.float64(0.0)
        vV = wp.float64(0.0)
        if CELL_GRAD:
            alpha_ = wp.float64(alpha[isys])
            exp_factor = wp.float64(0.25) / (alpha_ * alpha_)
            inv_vol = wp.float64(1.0) / volume[isys]
            vV = v_volume[isys]

        num_k = gA.shape[1]
        if num_k == 0:
            grad_positions[atom_idx] = vec_dtype(
                wp_dtype(0.0),
                wp_dtype(0.0),
                wp_dtype(0.0),
            )
            if DERIV_DQ:
                grad_charges[atom_idx] = wp.float64(0.0)
            return

        gr_x = wp.float64(0.0)
        gr_y = wp.float64(0.0)
        gr_z = wp.float64(0.0)
        gq = wp.float64(0.0)

        for k_idx in range(num_k):
            k_vec = k_vectors[isys, k_idx]
            kx = wp.float64(k_vec[0])
            ky = wp.float64(k_vec[1])
            kz = wp.float64(k_vec[2])
            k_squared = kx * kx + ky * ky + kz * kz
            if k_squared < wp.float64(_K_SQUARED_EPSILON):
                continue

            k_dot_r = kx * rx + ky * ry + kz * rz
            cos_m = wp.cos(k_dot_r)
            sin_m = wp.sin(k_dot_r)
            w_m = (
                wp.float64(vp[0]) * kx + wp.float64(vp[1]) * ky + wp.float64(vp[2]) * kz
            )

            a_g = gA[isys, k_idx]
            b_g = gB[isys, k_idx]
            p_g = gP[isys, k_idx]
            q_g = gQ[isys, k_idx]
            if DERIV_DQ:
                c_g = gC[isys, k_idx]
                d_g = gD[isys, k_idx]
            else:
                c_g = wp.float64(0.0)
                d_g = wp.float64(0.0)

            # Pos/charge terms (g_k folded into a_g..d_g already).
            # dPhi/dR_m = sum_k k g_k [ q_m cos_m (P+D) + q_m sin_m (Q-C)
            #                           - q_m w_m (A cos_m + B sin_m)
            #                           + vq_m (B cos_m - A sin_m) ]
            radial = (
                qi * cos_m * (p_g + d_g)
                + qi * sin_m * (q_g - c_g)
                - qi * w_m * (a_g * cos_m + b_g * sin_m)
                + vqi * (b_g * cos_m - a_g * sin_m)
            )

            if DERIV_DQ:
                # dPhi/dq_m = sum_k g_k [ sin_m (P+D) + cos_m (C-Q)
                #                         + w_m (B cos_m - A sin_m) ]
                gq += (
                    sin_m * (p_g + d_g)
                    + cos_m * (c_g - q_g)
                    + w_m * (b_g * cos_m - a_g * sin_m)
                )

            if CELL_GRAD:
                pu_g = gPu[isys, k_idx]
                qu_g = gQu[isys, k_idx]
                vk_vec = v_kvectors[isys, k_idx]
                vkx = wp.float64(vk_vec[0])
                vky = wp.float64(vk_vec[1])
                vkz = wp.float64(vk_vec[2])
                u_m = vkx * rx + vky * ry + vkz * rz
                wk = vkx * kx + vky * ky + vkz * kz
                mu = wp.float64(2.0) * exp_factor + wp.float64(2.0) / k_squared
                bcas = b_g * cos_m - a_g * sin_m  # (B cos - A sin), g-folded
                acbs = a_g * cos_m + b_g * sin_m  # (A cos + B sin), g-folded
                wmu_vv = wk * mu + vV * inv_vol

                # k-direction part of dPhi/dR_m (x k):
                radial += (
                    qi * (cos_m * pu_g + sin_m * qu_g - u_m * acbs) - wmu_vv * qi * bcas
                )
                # vk-direction part of dPhi/dR_m (x vk):
                vkdir = qi * bcas
                gr_x += vkdir * vkx
                gr_y += vkdir * vky
                gr_z += vkdir * vkz

                if DERIV_DQ:
                    # k/V additions to dPhi/dq_m:
                    gq += sin_m * pu_g - cos_m * qu_g + u_m * bcas - wmu_vv * acbs

            gr_x += radial * kx
            gr_y += radial * ky
            gr_z += radial * kz

        grad_positions[atom_idx] = vec_dtype(
            wp_dtype(ge * gr_x),
            wp_dtype(ge * gr_y),
            wp_dtype(ge * gr_z),
        )
        if DERIV_DQ:
            grad_charges[atom_idx] = ge * gq

    for kernel, base in (
        (_ewald_recip_dbwd_reduce, "ewald_recip_double_backward_reduce"),
        (
            _ewald_recip_dbwd_reduce_tiled,
            "ewald_recip_double_backward_reduce_tiled",
        ),
        (
            _ewald_recip_dbwd_compute,
            "ewald_recip_double_backward_compute",
        ),
    ):
        _name_and_document(
            kernel,
            base=base,
            wp_dtype=wp_dtype,
            batched=batched,
            deriv_state=deriv_state,
            cell_grad=cell_grad,
            order="double_backward",
        )
    return _RecipKernels(
        fill=_ewald_recip_dbwd_reduce_tiled if tiled else _ewald_recip_dbwd_reduce,
        compute=_ewald_recip_dbwd_compute,
        virial=None,
    )

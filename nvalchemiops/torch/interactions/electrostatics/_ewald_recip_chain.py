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

"""Explicit ``ewald_recip`` autograd chain.

Replaces the reciprocal-space Warp-tape ops with an explicit
forward -> backward -> double_backward chain over the factory kernel bundle
registered through :func:`register_warp_op_chain`.

The chain computes the **k-sum** reciprocal energy only (``E_k = 1/2 sum_k g_k
(A^2 + B^2)``); the self-energy and background corrections are smooth, closed-form
functions of charges / alpha / volume and are added in Torch-native code so they
differentiate natively (bit-identical to the ``_ewald_subtract_self_energy_kernel``
math). The chain differentiates:

* ``positions`` / ``charges`` -- via the atom-major ``compute`` kernel
  (``order="backward"`` / ``order="double_backward"``).
* ``k_vectors`` (vec3 per k) / ``volume`` (scalar per system) -- the cell-input
  grads the recip kernel *owns* (``kspace`` first-order, reduce stage for the
  second order). The reciprocal kernel never sees the integer Miller indices, so
  ``dk/dcell`` / ``dV/dcell`` are Torch's: the public layer builds
  ``k_vectors`` / ``volume`` from ``cell`` (``k_vectors.py`` / ``det(cell)``) and
  Torch composes ``grad_kvectors`` / ``grad_volume`` back to ``grad_cell``
  (mirroring PME's ``grad_volume`` / ``grad_k_squared``).

As with the real chain, the public energy is per-atom ``(N,)`` while the kernels
consume a per-system ``grad_energy`` ``(S,)``; the chain reduces the per-atom
cotangent to per-system by a per-system **mean** (uniform for the ``E.sum()``
contract).

**Forward precompute.** The fused ``order="forward"`` E_F /
E_F_dQ ``compute`` kernels produce the per-atom energy AND the first-order
``dE/dR`` / ``dE/dq`` caches in one atom-major (O(N*K)) pass; the first backward
scales those caches instead of re-running ``compute``. The factory keeps the
energy accumulation order identical across derivative states, so force-only
training can use E_F and skip charge-gradient work without changing the energy
value. When cell gradients are requested, the fill loop also emits unweighted
k-space cell-gradient sums so the first backward consumes O(S*K) cached rows
instead of recomputing O(N*K). ``double_backward`` is unchanged (recompute from
forward inputs); the detached caches are first-order value only.
"""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.interactions.electrostatics._factory_common import (
    _DerivState,
    get_backward_scale_kernel,
)
from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    BATCH_BLOCK_SIZE,
    EIGHTPI,
    RECIP_TILED_BLOCK_DIM,
    _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_cellgrad,
    _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_cellgrad_tiled,
    _ewald_reciprocal_space_energy_kernel_fill_structure_factors_cellgrad,
    _ewald_reciprocal_space_energy_kernel_fill_structure_factors_cellgrad_tiled,
    can_tile_ewald_recip_on_device,
    should_tile_ewald_recip_fill,
)
from nvalchemiops.interactions.electrostatics.ewald_recip_factory import (
    _make_backward_kspace_from_cache_kernel,
    alloc_ewald_recip_sentinels,
    get_ewald_recip_component_kernel,
    get_ewald_recip_kernel,
)
from nvalchemiops.torch._warp_op_helpers import (
    register_warp_op_chain,
)
from nvalchemiops.torch.interactions.electrostatics._util import (
    _is_per_system_uniform_cotangent,
)
from nvalchemiops.torch.types import (
    get_wp_dtype,
    get_wp_mat_dtype,
    get_wp_vec_dtype,
)

__all__ = [
    "ewald_recip_energy_single",
    "ewald_recip_energy_batch",
    "register_ewald_recip_ops",
]

_RECIP_SINGLE: dict[str, object] | None = None
_RECIP_BATCH: dict[str, object] | None = None
_EWALD_RECIP_OPS_REGISTERED = False


def _wp(tensor: torch.Tensor, dtype):
    """``wp.from_torch`` with shadow-gradient allocation disabled (chain owns bwd)."""
    return wp.from_torch(tensor.detach().contiguous(), dtype=dtype, requires_grad=False)


def _wp_empty_f64(shape, device: torch.device):
    """Allocate a Torch CUDA buffer and expose it to Warp without zero-fill."""
    return _wp(torch.empty(shape, dtype=torch.float64, device=device), wp.float64)


def _wp_zeros_f64(shape, device: torch.device):
    """Allocate a zeroed Torch CUDA buffer and expose it to Warp."""
    return _wp(torch.zeros(shape, dtype=torch.float64, device=device), wp.float64)


def _scoped_stream(device: torch.device):
    """Bind Warp's stream to PyTorch's current CUDA stream (graph-capture safe)."""
    if device.type != "cuda":
        from contextlib import nullcontext

        return nullcontext()
    return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device)))


def _atom_counts(batch_idx: torch.Tensor, num_systems: int) -> torch.Tensor:
    counts = torch.zeros(num_systems, dtype=torch.float64, device=batch_idx.device)
    return counts.index_add(
        0,
        batch_idx,
        torch.ones(batch_idx.shape[0], dtype=torch.float64, device=batch_idx.device),
    )


def _per_system_cotangent(grad_energy_atom, batch_idx, num_systems, num_atoms):
    """Reduce a per-atom energy cotangent ``(N,)`` to per-system ``(S,)`` (mean)."""
    g = grad_energy_atom.reshape(-1).to(torch.float64)
    if batch_idx is None:
        if num_atoms == 0:
            return torch.zeros(1, dtype=torch.float64, device=g.device)
        return g.mean().reshape(1)
    sums = torch.zeros(num_systems, dtype=torch.float64, device=g.device)
    sums = sums.index_add(0, batch_idx, g)
    return sums / _atom_counts(batch_idx, num_systems).clamp_min(1.0)


def _distribute_to_atoms(per_system, batch_idx, num_systems, num_atoms):
    if num_atoms == 0:
        return torch.zeros(0, dtype=torch.float64, device=per_system.device)
    if batch_idx is None:
        return (per_system / float(num_atoms)).expand(num_atoms).clone()
    counts = _atom_counts(batch_idx, num_systems)
    return (per_system / counts.clamp_min(1.0)).index_select(0, batch_idx)


def _cotangent_per_system_uniform(grad_energy_atom, batch_idx, num_systems):
    """Whether the per-atom energy cotangent is constant WITHIN each system.

    The cached-scale first backward serves ``grad_input = mean(cotangent) *
    dE_total/dinput`` per system, which equals the exact VJP only when the cotangent is
    uniform within a system (e.g. ``energy.sum()``). A non-uniform per-atom cotangent
    (a per-atom-energy-weighted loss) needs the weighted recompute below.
    """
    return _is_per_system_uniform_cotangent(
        grad_energy_atom,
        batch_idx,
        num_systems,
    )


def _recip_ksum_energy_torch(
    positions, charges, k_vectors_2d, volume, alpha, batch_idx, num_systems
):
    """Per-atom reciprocal k-sum energy (float64), matching the Warp fill+combine kernels.

    ``E_i = 0.5 * q_i * sum_k [cos(k.r_i) Re_SF[k] + sin(k.r_i) Im_SF[k]]`` with
    ``Re_SF[k] = G(k) sum_j q_j cos(k.r_j)``, ``Im_SF[k] = G(k) sum_j q_j sin(k.r_j)``,
    ``G(k) = (8*pi/V) exp(-k^2/(4 alpha^2)) / k^2`` on the half-space k-vectors
    (``k^2 < 1e-10`` -> 0). Pure Torch and autograd-correct for an arbitrary cotangent;
    used only on the rare non-uniform-cotangent backward path.
    """
    pos = positions.to(torch.float64)
    q = charges.to(torch.float64)
    alpha_flat = alpha.reshape(-1).to(torch.float64)
    energy = pos.new_zeros(pos.shape[0])
    for s in range(num_systems):
        a = alpha_flat[0] if alpha_flat.numel() == 1 else alpha_flat[s]
        exp_factor = 0.25 / (a * a)
        k = k_vectors_2d[s].to(torch.float64)
        ksq = (k * k).sum(-1)
        green = (
            (EIGHTPI / volume[s].to(torch.float64)) * torch.exp(-ksq * exp_factor) / ksq
        )
        green = torch.where(ksq < 1e-10, torch.zeros_like(green), green)
        if batch_idx is None:
            p_s, q_s = pos, q
        else:
            sel = batch_idx == s
            p_s, q_s = pos[sel], q[sel]
        kr = p_s @ k.transpose(0, 1)
        cos_kr = torch.cos(kr)
        sin_kr = torch.sin(kr)
        re_sf = (q_s.unsqueeze(1) * cos_kr).sum(0) * green
        im_sf = (q_s.unsqueeze(1) * sin_kr).sum(0) * green
        e_s = 0.5 * q_s * (cos_kr @ re_sf + sin_kr @ im_sf)
        if batch_idx is None:
            energy = e_s
        else:
            idx = sel.nonzero(as_tuple=True)[0]
            energy = energy.index_copy(0, idx, e_s)
    return energy


def _run_fill(
    bundle,
    positions,
    charges,
    cell,
    k_vectors_2d,
    alpha,
    batch_idx,
    atom_start,
    atom_end,
    num_k,
    num_systems,
    num_atoms,
    wp_scalar,
    wp_vec,
    device,
    want_cellgrad=False,
):
    """Launch the (reused hand-written) fill kernel; return cos/sin/S_real/S_imag.

    ``cell`` enters the fill only for the Green's-function ``1/V`` factor (detached
    -- the differentiable volume path lives in the ``kspace`` / reduce kernels and
    the Torch background correction). Returns 2D ``(S, K)`` structure factors plus an
    optional ``cellgrad_cache`` torch tensor (``None`` unless ``want_cellgrad``): the
    un-weighted per-k sums ``[A, B, Ra(3), Rb(3)]`` accumulated in the fill's atom
    loop, consumed by the O(S*K) ``kspace`` backward. Single uses ``(K, 8)``;
    batched uses ``(S*K, 8)`` with row ``system_id * K + k_idx``.
    """
    batched = batch_idx is not None
    cellgrad_cache = None
    cos_kr = _wp_empty_f64((num_k, num_atoms), positions.device)
    sin_kr = _wp_empty_f64((num_k, num_atoms), positions.device)
    wp_cell = _wp(cell, get_wp_mat_dtype(cell.dtype))
    wp_alpha = _wp(alpha, wp_scalar)
    wp_pos = _wp(positions, wp_vec)
    wp_chg = _wp(charges, wp_scalar)
    with _scoped_stream(positions.device):
        if batched:
            real_sf = _wp_zeros_f64((num_systems, num_k), positions.device)
            imag_sf = _wp_zeros_f64((num_systems, num_k), positions.device)
            total_charges = _wp_zeros_f64(num_systems, positions.device)
            wp_as = _wp(atom_start, wp.int32)
            wp_ae = _wp(atom_end, wp.int32)
            max_atoms = int((atom_end - atom_start).max().item()) if num_atoms else 0
            max_blocks = (max_atoms + BATCH_BLOCK_SIZE - 1) // BATCH_BLOCK_SIZE
            max_blocks = max(max_blocks, 1)
            use_tiled_fill = can_tile_ewald_recip_on_device(
                device
            ) and should_tile_ewald_recip_fill(max_atoms)
            if want_cellgrad:
                cache_t = torch.zeros(
                    (num_systems * num_k, 8),
                    dtype=torch.float64,
                    device=positions.device,
                )
                cache_wp = _wp(cache_t, wp.float64)
                inputs = [
                    wp_pos,
                    wp_chg,
                    _wp(k_vectors_2d, wp_vec),
                    wp_cell,
                    wp_alpha,
                    wp_as,
                    wp_ae,
                    total_charges,
                    cos_kr,
                    sin_kr,
                    real_sf,
                    imag_sf,
                    cache_wp,
                ]
                if use_tiled_fill:
                    wp.launch_tiled(
                        _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_cellgrad_tiled,
                        dim=(num_k, num_systems),
                        inputs=inputs,
                        device=device,
                        block_dim=RECIP_TILED_BLOCK_DIM,
                    )
                else:
                    wp.launch(
                        _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_cellgrad,
                        dim=(num_k, num_systems, max_blocks),
                        inputs=inputs,
                        device=device,
                    )
                cellgrad_cache = cache_t
            else:
                inputs = [
                    wp_pos,
                    wp_chg,
                    _wp(k_vectors_2d, wp_vec),
                    wp_cell,
                    wp_alpha,
                    wp_as,
                    wp_ae,
                    total_charges,
                    cos_kr,
                    sin_kr,
                    real_sf,
                    imag_sf,
                ]
                if use_tiled_fill:
                    wp.launch_tiled(
                        get_ewald_recip_component_kernel(
                            wp_scalar,
                            component="fill",
                            batched=True,
                            tiled=True,
                        ),
                        dim=(num_k, num_systems),
                        inputs=inputs,
                        device=device,
                        block_dim=RECIP_TILED_BLOCK_DIM,
                    )
                else:
                    wp.launch(
                        bundle.fill,
                        dim=(num_k, num_systems, max_blocks),
                        inputs=inputs,
                        device=device,
                    )
        else:
            kv_1d = _wp(k_vectors_2d.reshape(num_k, 3), wp_vec)
            real_1d = _wp_empty_f64(num_k, positions.device)
            imag_1d = _wp_empty_f64(num_k, positions.device)
            total_charge = _wp_zeros_f64(1, positions.device)
            use_tiled_fill = can_tile_ewald_recip_on_device(
                device
            ) and should_tile_ewald_recip_fill(num_atoms)
            if want_cellgrad:
                # Fused variant: same outputs + the un-weighted (K, 8) cell-grad
                # reduction, accumulated in the SAME atom loop (marginal cost), so the
                # first backward consumes O(K) cached sums instead of the O(K*N) kspace
                # recompute.
                cache_t = torch.empty(
                    (num_k, 8), dtype=torch.float64, device=positions.device
                )
                cache_wp = _wp(cache_t, wp.float64)
                inputs = [
                    wp_pos,
                    wp_chg,
                    kv_1d,
                    wp_cell,
                    wp_alpha,
                    total_charge,
                    cos_kr,
                    sin_kr,
                    real_1d,
                    imag_1d,
                    cache_wp,
                ]
                if use_tiled_fill:
                    wp.launch_tiled(
                        _ewald_reciprocal_space_energy_kernel_fill_structure_factors_cellgrad_tiled,
                        dim=num_k,
                        inputs=inputs,
                        device=device,
                        block_dim=RECIP_TILED_BLOCK_DIM,
                    )
                else:
                    wp.launch(
                        _ewald_reciprocal_space_energy_kernel_fill_structure_factors_cellgrad,
                        dim=num_k,
                        inputs=inputs,
                        device=device,
                    )
                cellgrad_cache = cache_t
            else:
                inputs = [
                    wp_pos,
                    wp_chg,
                    kv_1d,
                    wp_cell,
                    wp_alpha,
                    total_charge,
                    cos_kr,
                    sin_kr,
                    real_1d,
                    imag_1d,
                ]
                if use_tiled_fill:
                    wp.launch_tiled(
                        get_ewald_recip_component_kernel(
                            wp_scalar,
                            component="fill",
                            tiled=True,
                        ),
                        dim=num_k,
                        inputs=inputs,
                        device=device,
                        block_dim=RECIP_TILED_BLOCK_DIM,
                    )
                else:
                    wp.launch(
                        bundle.fill,
                        dim=num_k,
                        inputs=inputs,
                        device=device,
                    )
            real_sf = real_1d.reshape((1, num_k))
            imag_sf = imag_1d.reshape((1, num_k))
    return cos_kr, sin_kr, real_sf, imag_sf, cellgrad_cache


# ===========================================================================
# Forward / backward / double-backward implementations (shared single + batch)
# ===========================================================================


def _atom_cotangent(grad_energy_atom, batch_idx, num_systems, num_atoms):
    """Per-system cotangent (mean) broadcast back to per-atom ``(N,)`` f64.

    Exactly the per-system ``grad_energy`` the atom-major ``compute`` kernel scaled by,
    mapped to atoms so the cached-derivative scale reproduces its first-backward value.
    """
    g_sys = _per_system_cotangent(grad_energy_atom, batch_idx, num_systems, num_atoms)
    if batch_idx is None:
        return g_sys.expand(num_atoms)
    return g_sys.index_select(0, batch_idx)


def _forward_impl(
    positions,
    charges,
    cell,
    k_vectors_2d,
    volume,
    alpha,
    batch_idx,
    atom_start,
    atom_end,
    need_pos,
    need_charge,
    need_cell,
):
    """Fused recip forward: energy (always) + detached ``dE/dR`` / ``dE/dq`` caches.

    When ``need_pos`` / ``need_charge`` is set, the atom-major ``compute`` writes
    forces / charge-grad in the SAME pass as the energy, and the caches are
    returned detached. The expensive O(N*K) atom-major traversal is thereby done
    once; the first backward scales the caches instead of re-running ``compute``.
    The k-major ``kspace`` (k/V cell grads, O(S*K)) stays on recompute in the
    backward.
    """
    num_atoms = positions.shape[0]
    num_k = k_vectors_2d.shape[-2]
    num_systems = volume.shape[0]
    input_dtype = positions.dtype
    device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    batched = batch_idx is not None

    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    dEdR = torch.zeros(
        num_atoms if need_pos else 0, 3, device=positions.device, dtype=input_dtype
    )
    dEdq = torch.zeros(
        num_atoms if need_charge else 0, device=positions.device, dtype=torch.float64
    )
    # Un-weighted cell-grad reduction cache: full-size only for first-order cell-grad
    # paths (single ``K`` rows, batched ``S*K`` rows); zero-size otherwise.
    cellgrad_cache = torch.zeros(0, 8, device=positions.device, dtype=torch.float64)
    if num_atoms == 0 or num_k == 0:
        return energies, dEdR, dEdq, cellgrad_cache

    need_deriv = need_pos or need_charge
    if need_charge:
        deriv_state = _DerivState.E_F_dQ
    elif need_pos:
        deriv_state = _DerivState.E_F
    else:
        deriv_state = _DerivState.E
    bundle = get_ewald_recip_kernel(
        wp_scalar, batched=batched, deriv_state=deriv_state, order="forward"
    )
    s = alloc_ewald_recip_sentinels(wp_scalar, device)
    want_cellgrad = bool(need_cell)
    cos_kr, sin_kr, real_sf, imag_sf, cellgrad_fill = _run_fill(
        bundle,
        positions,
        charges,
        cell,
        k_vectors_2d,
        alpha,
        batch_idx,
        atom_start,
        atom_end,
        num_k,
        num_systems,
        num_atoms,
        wp_scalar,
        wp_vec,
        device,
        want_cellgrad=want_cellgrad,
    )
    if cellgrad_fill is not None:
        cellgrad_cache = cellgrad_fill.detach()
    batch_id = _wp(batch_idx, wp.int32) if batched else _s_int_empty(device)
    # The forward kernel writes F into atomic_forces whenever DERIV >= E_F and
    # dE/dq into charge_gradients only when DERIV >= E_F_dQ. dE/dR = -F; the
    # dE/dq cache is allocated only when charges actually need grad.
    forces = (
        torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
        if need_deriv
        else None
    )
    out_forces = _wp(forces, wp_vec) if need_deriv else s["atomic_forces"]
    out_cg = _wp(dEdq, wp.float64) if need_charge else s["charge_gradients"]
    with _scoped_stream(positions.device):
        wp.launch(
            bundle.compute,
            dim=num_atoms,
            inputs=[
                _wp(charges, wp_scalar),
                batch_id,
                _wp(k_vectors_2d, wp_vec),
                cos_kr,
                sin_kr,
                real_sf,
                imag_sf,
                s["grad_energy"],
                _wp(energies, wp.float64),
                out_forces,
                out_cg,
            ],
            device=device,
        )
    if need_pos:
        dEdR = (-forces).detach()
    return energies, dEdR.detach(), dEdq.detach(), cellgrad_cache


def _backward_impl(
    dEdR_cache,
    dEdq_cache,
    cellgrad_cache,
    grad_energy_atom,
    positions,
    charges,
    cell,
    k_vectors_2d,
    volume,
    alpha,
    batch_idx,
    atom_start,
    atom_end,
    need_pos,
    need_charge,
    need_cell,
):
    """First backward: scale the cached atom-major dE/dR / dE/dq; recompute k/V on demand.

    ``grad_positions`` / ``grad_charges`` come from scaling the detached forward caches
    by the per-system ``grad_energy`` (no atom-major ``compute`` recompute -- identical
    value to the old ``compute(order="backward")``). The ``grad_kvectors`` / ``grad_volume``
    cell-input grads are produced by the k-major ``kspace`` kernel ONLY when ``need_cell``
    (``cell.requires_grad``); otherwise they stay zero (matched to ``None`` by the chain
    wiring). ``kspace`` consumes no structure factors, so no ``fill`` is launched here --
    the force+charge step is a true scale with no per-k loop: the
    cheap k-major piece stays on recompute, and only when cell actually needs grad).
    """
    num_atoms = positions.shape[0]
    num_k = k_vectors_2d.shape[-2]
    num_systems = volume.shape[0]
    input_dtype = positions.dtype
    device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    batched = batch_idx is not None

    grad_positions = torch.zeros(
        num_atoms, 3, device=positions.device, dtype=input_dtype
    )
    grad_charges = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    grad_kvectors = torch.zeros(
        num_systems, num_k, 3, device=positions.device, dtype=input_dtype
    )
    grad_volume = torch.zeros(num_systems, device=positions.device, dtype=torch.float64)
    if num_atoms == 0 or num_k == 0:
        return grad_positions, grad_charges, grad_kvectors, grad_volume

    # Non-uniform per-atom cotangent: the cached dE_total/dinput (summed over atoms)
    # cannot be re-weighted post-hoc, so the per-system-mean scale path below is wrong.
    # Recompute the exact weighted VJP from the differentiable Torch k-sum energy. The
    # uniform path (the common training case, e.g. energy.sum()) keeps the fast scale.
    any_need = need_pos or need_charge or need_cell
    if any_need and not _cotangent_per_system_uniform(
        grad_energy_atom, batch_idx, num_systems
    ):
        # The custom-op backward runs in inference mode; build a fresh autograd graph
        # (inference_mode(False) + materialized leaves) for the weighted recompute.
        with torch.inference_mode(False), torch.enable_grad():

            def _leaf(t, requires_grad):
                out = torch.empty_like(t, dtype=torch.float64).copy_(t).detach()
                return out.requires_grad_(requires_grad)

            p_leaf = _leaf(positions, True)
            q_leaf = _leaf(charges, True)
            kv_leaf = _leaf(k_vectors_2d, True)
            vol_leaf = _leaf(volume, True)
            alpha_f = _leaf(alpha, False)
            w_f = _leaf(grad_energy_atom.reshape(-1), False)
            e_i = _recip_ksum_energy_torch(
                p_leaf, q_leaf, kv_leaf, vol_leaf, alpha_f, batch_idx, num_systems
            )
            loss = (w_f * e_i).sum()
            gp, gq, gkv, gvol = torch.autograd.grad(
                loss, [p_leaf, q_leaf, kv_leaf, vol_leaf], allow_unused=True
            )
        if need_pos and gp is not None:
            grad_positions = gp.to(input_dtype)
        if need_charge and gq is not None:
            grad_charges = gq.to(torch.float64)
        if need_cell:
            if gkv is not None:
                grad_kvectors = gkv.to(input_dtype)
            if gvol is not None:
                grad_volume = gvol.to(torch.float64)
        return grad_positions, grad_charges, grad_kvectors, grad_volume

    grad_energy = None

    # Atom-major first-order grads: cheap scale of the detached forward caches.
    scale_positions = need_pos and dEdR_cache.shape[0] == num_atoms
    scale_charges = need_charge and dEdq_cache.shape[0] == num_atoms
    if scale_positions or scale_charges:
        grad_energy = _per_system_cotangent(
            grad_energy_atom, batch_idx, num_systems, num_atoms
        )
        batch_id = (
            _wp(batch_idx, wp.int32)
            if batched
            else wp.empty((0,), dtype=wp.int32, device=device)
        )
        kernel = get_backward_scale_kernel(
            wp_scalar,
            batched=batched,
            scale_positions=scale_positions,
            scale_charges=scale_charges,
        )
        with _scoped_stream(positions.device):
            wp.launch(
                kernel,
                dim=num_atoms,
                inputs=[
                    _wp(grad_energy, wp.float64),
                    batch_id,
                    _wp(dEdR_cache, wp_vec),
                    _wp(dEdq_cache, wp.float64),
                    _wp(grad_positions, wp_vec),
                    _wp(grad_charges, wp.float64),
                    wp.int32(num_atoms),
                ],
                device=device,
            )

    # k-major cell-input grads (grad_kvectors / grad_volume): consume the forward
    # cellgrad cache when available, otherwise recompute. This runs only when cell
    # needs grad.
    if need_cell:
        if grad_energy is None:
            grad_energy = _per_system_cotangent(
                grad_energy_atom, batch_idx, num_systems, num_atoms
            )
        wp_ge = _wp(grad_energy, wp.float64)
        expected_cache_rows = num_systems * num_k
        use_cache = (
            cellgrad_cache is not None
            and cellgrad_cache.shape[0] == expected_cache_rows
        )
        with _scoped_stream(positions.device):
            if use_cache:
                # O(S*K) consume of the forward-fused un-weighted sums -- no atom loop.
                consume = _make_backward_kspace_from_cache_kernel(wp_scalar, batched)
                wp.launch(
                    consume,
                    dim=(num_k, num_systems) if batched else num_k,
                    inputs=[
                        _wp(k_vectors_2d, wp_vec),
                        _wp(alpha, wp_scalar),
                        _wp(volume, wp.float64),
                        _wp(cellgrad_cache, wp.float64),
                        wp_ge,
                        wp.int32(num_k),
                        _wp(grad_kvectors, wp_vec),
                        _wp(grad_volume, wp.float64),
                    ],
                    device=device,
                )
            else:
                # Recompute path (batched, or no cache): O(K*N) kspace kernel.
                bundle = get_ewald_recip_kernel(
                    wp_scalar,
                    batched=batched,
                    deriv_state=_DerivState.E_F_dQ,
                    cell_grad=True,
                    order="backward",
                )
                batch_id = _wp(batch_idx, wp.int32) if batched else _s_int_empty(device)
                if batched:
                    wp_as = _wp(atom_start, wp.int32)
                    wp_ae = _wp(atom_end, wp.int32)
                else:
                    wp_as = _s_int_empty(device)
                    wp_ae = _s_int_empty(device)
                wp.launch(
                    bundle.kspace,
                    dim=(num_k, num_systems) if batched else num_k,
                    inputs=[
                        _wp(positions, wp_vec),
                        _wp(charges, wp_scalar),
                        _wp(k_vectors_2d, wp_vec),
                        _wp(alpha, wp_scalar),
                        _wp(volume, wp.float64),
                        batch_id,
                        wp_as,
                        wp_ae,
                        wp_ge,
                        _wp(grad_kvectors, wp_vec),
                        _wp(grad_volume, wp.float64),
                    ],
                    device=device,
                )
    return grad_positions, grad_charges, grad_kvectors, grad_volume


def _double_backward_impl(
    v_pos,
    v_charge,
    v_kvectors,
    v_volume,
    dEdR_cache,
    dEdq_cache,
    cellgrad_cache,
    grad_energy_atom,
    positions,
    charges,
    cell,
    k_vectors_2d,
    volume,
    alpha,
    batch_idx,
    atom_start,
    atom_end,
    need_pos,
    need_charge,
    need_cell,
):
    # ``dEdR_cache`` / ``dEdq_cache`` (the backward op's leading first-order caches) and
    # the trailing ``need_*`` flags are accepted for positional alignment but unused: the
    # second order recomputes the per-(system,k) sums from the forward inputs.
    #
    num_atoms = positions.shape[0]
    num_k = k_vectors_2d.shape[-2]
    num_systems = volume.shape[0]
    input_dtype = positions.dtype
    device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    batched = batch_idx is not None

    grad_grad_energy = torch.zeros(
        num_systems, device=positions.device, dtype=torch.float64
    )
    grad_positions = torch.zeros(
        num_atoms, 3, device=positions.device, dtype=input_dtype
    )
    grad_charges = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    grad_kvectors = torch.zeros(
        num_systems, num_k, 3, device=positions.device, dtype=input_dtype
    )
    grad_volume = torch.zeros(num_systems, device=positions.device, dtype=torch.float64)

    if num_atoms == 0 or num_k == 0:
        gge = _distribute_to_atoms(grad_grad_energy, batch_idx, num_systems, num_atoms)
        return gge, grad_positions, grad_charges, grad_kvectors, grad_volume

    grad_energy = _per_system_cotangent(
        grad_energy_atom, batch_idx, num_systems, num_atoms
    )

    use_charge_db = bool(need_charge)
    use_cell_db = bool(need_cell)
    if batched and num_atoms:
        max_atoms = int((atom_end - atom_start).max().item())
    else:
        max_atoms = num_atoms
    use_tiled_reduce = (
        (not use_cell_db)
        and can_tile_ewald_recip_on_device(device)
        and should_tile_ewald_recip_fill(max_atoms)
    )
    bundle = get_ewald_recip_kernel(
        wp_scalar,
        batched=batched,
        deriv_state=_DerivState.E_F_dQ,
        cell_grad=use_cell_db,
        order="double_backward",
        tiled=use_tiled_reduce,
    )
    # Per-(system,k) reduction scratch buffers (g_k-scaled sums).
    gA = wp.zeros((num_systems, num_k), dtype=wp.float64, device=device)
    gB = wp.zeros((num_systems, num_k), dtype=wp.float64, device=device)
    gC = wp.zeros((num_systems, num_k), dtype=wp.float64, device=device)
    gD = wp.zeros((num_systems, num_k), dtype=wp.float64, device=device)
    gP = wp.zeros((num_systems, num_k), dtype=wp.float64, device=device)
    gQ = wp.zeros((num_systems, num_k), dtype=wp.float64, device=device)
    gPu = wp.zeros((num_systems, num_k), dtype=wp.float64, device=device)
    gQu = wp.zeros((num_systems, num_k), dtype=wp.float64, device=device)

    cell_mat = _wp(cell, get_wp_mat_dtype(input_dtype))
    batch_id = _wp(batch_idx, wp.int32) if batched else _s_int_empty(device)
    wp_ge = _wp(grad_energy, wp.float64)
    wp_kv = _wp(k_vectors_2d, wp_vec)
    wp_pos = _wp(positions, wp_vec)
    wp_chg = _wp(charges, wp_scalar)
    wp_vpos = _wp(v_pos, wp_vec)
    wp_vq = _wp(v_charge, wp.float64)
    wp_vkv = _wp(v_kvectors, wp_vec)
    wp_vvol = _wp(v_volume, wp.float64)
    wp_vol = _wp(volume, wp.float64)
    wp_alpha = _wp(alpha, wp_scalar)
    wp_ggE = _wp(grad_grad_energy, wp.float64)
    wp_gkv = _wp(grad_kvectors, wp_vec)
    wp_gvol = _wp(grad_volume, wp.float64)
    if batched:
        wp_as = _wp(atom_start, wp.int32)
        wp_ae = _wp(atom_end, wp.int32)
    else:
        wp_as = _s_int_empty(device)
        wp_ae = _s_int_empty(device)
    deriv_dq = wp.int32(1 if use_charge_db else 0)
    cell_grad_flag = wp.int32(1 if use_cell_db else 0)
    with _scoped_stream(positions.device):
        reduce_inputs = [
            wp_pos,
            wp_chg,
            wp_kv,
            cell_mat,
            wp_alpha,
            batch_id,
            wp_as,
            wp_ae,
            wp_vpos,
            wp_vq,
            wp_ge,
            deriv_dq,
            gA,
            gB,
            gC,
            gD,
            gP,
            gQ,
            wp_ggE,
            cell_grad_flag,
            wp_vol,
            wp_vkv,
            wp_vvol,
            gPu,
            gQu,
            wp_gkv,
            wp_gvol,
        ]
        if use_tiled_reduce:
            wp.launch_tiled(
                bundle.fill,  # = tiled reduce kernel for double_backward bundle
                dim=(num_k, num_systems) if batched else num_k,
                inputs=reduce_inputs,
                device=device,
                block_dim=RECIP_TILED_BLOCK_DIM,
            )
        else:
            wp.launch(
                bundle.fill,  # = reduce kernel for double_backward bundle
                dim=(num_k, num_systems) if batched else num_k,
                inputs=reduce_inputs,
                device=device,
            )
        wp.launch(
            bundle.compute,
            dim=num_atoms,
            inputs=[
                wp_pos,
                wp_chg,
                wp_kv,
                batch_id,
                wp_vpos,
                wp_vq,
                wp_ge,
                deriv_dq,
                gA,
                gB,
                gC,
                gD,
                gP,
                gQ,
                _wp(grad_positions, wp_vec),
                _wp(grad_charges, wp.float64),
                cell_grad_flag,
                wp_alpha,
                wp_vol,
                wp_vkv,
                wp_vvol,
                gPu,
                gQu,
            ],
            device=device,
        )
    gge = _distribute_to_atoms(grad_grad_energy, batch_idx, num_systems, num_atoms)
    return gge, grad_positions, grad_charges, grad_kvectors, grad_volume


def _s_int_empty(device):
    """Return an empty int32 Warp array for single-system launch placeholders."""
    return wp.empty((0,), dtype=wp.int32, device=device)


# ===========================================================================
# Single-system chain launchers (batch_idx is None)
# ===========================================================================


def _recip_forward_single(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    need_pos: bool,
    need_charge: bool,
    need_cell: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _forward_impl(
        positions,
        charges,
        cell,
        k_vectors,
        volume,
        alpha,
        None,
        None,
        None,
        need_pos,
        need_charge,
        need_cell,
    )


def _recip_backward_single(
    dEdR_cache: torch.Tensor,
    dEdq_cache: torch.Tensor,
    cellgrad_cache: torch.Tensor,
    grad_energy: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    need_pos: bool,
    need_charge: bool,
    need_cell: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _backward_impl(
        dEdR_cache,
        dEdq_cache,
        cellgrad_cache,
        grad_energy,
        positions,
        charges,
        cell,
        k_vectors,
        volume,
        alpha,
        None,
        None,
        None,
        need_pos,
        need_charge,
        need_cell,
    )


def _recip_double_backward_single(
    v_pos: torch.Tensor,
    v_charge: torch.Tensor,
    v_kvectors: torch.Tensor,
    v_volume: torch.Tensor,
    dEdR_cache: torch.Tensor,
    dEdq_cache: torch.Tensor,
    cellgrad_cache: torch.Tensor,
    grad_energy: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    need_pos: bool,
    need_charge: bool,
    need_cell: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _double_backward_impl(
        v_pos,
        v_charge,
        v_kvectors,
        v_volume,
        dEdR_cache,
        dEdq_cache,
        cellgrad_cache,
        grad_energy,
        positions,
        charges,
        cell,
        k_vectors,
        volume,
        alpha,
        None,
        None,
        None,
        need_pos,
        need_charge,
        need_cell,
    )


# ===========================================================================
# Batched chain launchers (batch_idx provided)
# ===========================================================================


def _recip_forward_batch(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    atom_start: torch.Tensor,
    atom_end: torch.Tensor,
    need_pos: bool,
    need_charge: bool,
    need_cell: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _forward_impl(
        positions,
        charges,
        cell,
        k_vectors,
        volume,
        alpha,
        batch_idx,
        atom_start,
        atom_end,
        need_pos,
        need_charge,
        need_cell,
    )


def _recip_backward_batch(
    dEdR_cache: torch.Tensor,
    dEdq_cache: torch.Tensor,
    cellgrad_cache: torch.Tensor,
    grad_energy: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    atom_start: torch.Tensor,
    atom_end: torch.Tensor,
    need_pos: bool,
    need_charge: bool,
    need_cell: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _backward_impl(
        dEdR_cache,
        dEdq_cache,
        cellgrad_cache,
        grad_energy,
        positions,
        charges,
        cell,
        k_vectors,
        volume,
        alpha,
        batch_idx,
        atom_start,
        atom_end,
        need_pos,
        need_charge,
        need_cell,
    )


def _recip_double_backward_batch(
    v_pos: torch.Tensor,
    v_charge: torch.Tensor,
    v_kvectors: torch.Tensor,
    v_volume: torch.Tensor,
    dEdR_cache: torch.Tensor,
    dEdq_cache: torch.Tensor,
    cellgrad_cache: torch.Tensor,
    grad_energy: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    atom_start: torch.Tensor,
    atom_end: torch.Tensor,
    need_pos: bool,
    need_charge: bool,
    need_cell: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _double_backward_impl(
        v_pos,
        v_charge,
        v_kvectors,
        v_volume,
        dEdR_cache,
        dEdq_cache,
        cellgrad_cache,
        grad_energy,
        positions,
        charges,
        cell,
        k_vectors,
        volume,
        alpha,
        batch_idx,
        atom_start,
        atom_end,
        need_pos,
        need_charge,
        need_cell,
    )


# ===========================================================================
# Register the chains
# ===========================================================================
#
# Single-system forward inputs (9): 0 positions, 1 charges, 2 cell, 3 k_vectors,
#   4 volume, 5 alpha, 6 need_pos, 7 need_charge, 8 need_cell. Forward outputs are
#   energy plus detached position/charge/cellgrad caches. Differentiable forward
#   inputs are positions(0), charges(1), k_vectors(3), and volume(4). Only the energy
#   cotangent drives the backward; the three caches are threaded as saved outputs.


def _recip_forward_fake(positions, *args):
    """Forward fake: ``(energy, dE/dR cache, dE/dq cache, cellgrad cache)``.

    Cache shapes gated by the ``need_pos`` / ``need_charge`` booleans (the last three
    trailing args are ``need_pos, need_charge, need_cell``).
    """
    need_pos, need_charge = bool(args[-3]), bool(args[-2])
    n = positions.shape[0]
    energy = positions.new_empty(n, dtype=torch.float64)
    dEdR = positions.new_empty(n if need_pos else 0, 3, dtype=positions.dtype)
    dEdq = positions.new_empty(n if need_charge else 0, dtype=torch.float64)
    cellgrad_cache = positions.new_empty(0, 8, dtype=torch.float64)
    return energy, dEdR, dEdq, cellgrad_cache


def _recip_backward_fake(
    dEdR_cache,
    dEdq_cache,
    cellgrad_cache,
    grad_energy,
    positions,
    charges,
    cell,
    k_vectors,
    volume,
    *args,
):
    """Backward fake: ``(grad_positions, grad_charges, grad_kvectors, grad_volume)``."""
    n = positions.shape[0]
    num_k = k_vectors.shape[-2]
    num_systems = volume.shape[0]
    return (
        positions.new_empty(n, 3, dtype=positions.dtype),
        positions.new_empty(n, dtype=torch.float64),
        positions.new_empty(num_systems, num_k, 3, dtype=positions.dtype),
        positions.new_empty(num_systems, dtype=torch.float64),
    )


def _recip_double_backward_fake(
    v_pos,
    v_charge,
    v_kvectors,
    v_volume,
    dEdR_cache,
    dEdq_cache,
    cellgrad_cache,
    grad_energy,
    positions,
    charges,
    cell,
    k_vectors,
    volume,
    *args,
):
    """Double-backward fake: ``(grad_grad_energy, grad_pos, grad_chg, grad_kv, grad_vol)``."""
    n = positions.shape[0]
    num_k = k_vectors.shape[-2]
    num_systems = volume.shape[0]
    return (
        positions.new_empty(n, dtype=torch.float64),
        positions.new_empty(n, 3, dtype=positions.dtype),
        positions.new_empty(n, dtype=torch.float64),
        positions.new_empty(num_systems, num_k, 3, dtype=positions.dtype),
        positions.new_empty(num_systems, dtype=torch.float64),
    )


def register_ewald_recip_ops() -> None:
    """Register the Ewald reciprocal-space Torch custom-op chain once."""
    global _EWALD_RECIP_OPS_REGISTERED, _RECIP_BATCH, _RECIP_SINGLE
    if _EWALD_RECIP_OPS_REGISTERED:
        return

    _RECIP_SINGLE = register_warp_op_chain(
        name="nvalchemiops::ewald_recip_energy_single",
        forward=_recip_forward_single,
        backward=_recip_backward_single,
        double_backward=_recip_double_backward_single,
        forward_fake=_recip_forward_fake,
        backward_fake=_recip_backward_fake,
        double_backward_fake=_recip_double_backward_fake,
        forward_return_arity=4,
        propagate_outputs=(0,),
        save_forward_outputs=(1, 2, 3),
        diff_input_positions=(0, 1, 3, 4),
        n_forward_inputs=9,
        backward_return_arity=4,
        second_order_diff_positions=(3, 4, 5, 7, 8),
        n_backward_inputs=13,
        double_backward_return_arity=5,
    )

    # Batched forward inputs (12): 0 positions, 1 charges, 2 cell, 3 k_vectors,
    #   4 volume, 5 alpha, 6 batch_idx, 7 atom_start, 8 atom_end, 9 need_pos,
    #   10 need_charge, 11 need_cell.
    _RECIP_BATCH = register_warp_op_chain(
        name="nvalchemiops::ewald_recip_energy_batch",
        forward=_recip_forward_batch,
        backward=_recip_backward_batch,
        double_backward=_recip_double_backward_batch,
        forward_fake=_recip_forward_fake,
        backward_fake=_recip_backward_fake,
        double_backward_fake=_recip_double_backward_fake,
        forward_return_arity=4,
        propagate_outputs=(0,),
        save_forward_outputs=(1, 2, 3),
        diff_input_positions=(0, 1, 3, 4),
        n_forward_inputs=12,
        backward_return_arity=4,
        second_order_diff_positions=(3, 4, 5, 7, 8),
        n_backward_inputs=16,
        double_backward_return_arity=5,
        batch_match=True,
    )

    _EWALD_RECIP_OPS_REGISTERED = True


def ewald_recip_energy_single(*args, **kwargs):
    """Call the registered single-system Ewald reciprocal-space energy op."""
    register_ewald_recip_ops()
    if _RECIP_SINGLE is None:
        raise RuntimeError("Ewald reciprocal single-system op registration failed")
    return _RECIP_SINGLE["forward"](*args, **kwargs)


def ewald_recip_energy_batch(*args, **kwargs):
    """Call the registered batched Ewald reciprocal-space energy op."""
    register_ewald_recip_ops()
    if _RECIP_BATCH is None:
        raise RuntimeError("Ewald reciprocal batched op registration failed")
    return _RECIP_BATCH["forward"](*args, **kwargs)

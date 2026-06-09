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

"""Shared derivative-verification harness for PME / Ewald electrostatics.

This is an **importable helper module, not a test module** (no ``test_`` prefix,
no ``Test*``/``test_*`` callables), so pytest does not collect it directly. It is
consumed by ``test_deriv_check_selftest.py`` and the parity/autograd tests for
the energy-derivative PME/Ewald refactor.

It provides:

1. **Finite-difference** helpers (central difference, float64):

   - forces ``-dE/dR`` (:func:`fd_forces`),
   - charge gradients ``dE/dq`` (:func:`fd_charge_grad`),
   - strain-first virial ``-dE/dstrain`` (:func:`fd_strain_virial`) built from a
     differentiable strain tensor.

2. **gradcheck / gradgradcheck** wrappers configured for float64 + small systems
   (:func:`gradcheck_energy`, :func:`gradgradcheck_energy`).

3. **Tape-vs-explicit equivalence** helper (:func:`assert_tape_vs_explicit`,
   :class:`EquivPoint`) with parameterizable comparison points. Today the
   "explicit" callable returns direct-kernel outputs and the "reference" returns
   autograd-derived quantities.

4. Small reusable **fixed-charge** systems (:func:`fixed_charge_system`) and a
   tiny differentiable **toy q(R)** charge model (:func:`toy_charge_model`).

Design principle
----------------
Each test builds **one** ``energy_fn(positions, charges, cell) -> Tensor``
closure (returning the public per-atom energy vector) and feeds the *same*
closure to both the finite-difference and the autograd helpers. Finite-diff
perturbs the closure inputs; autograd differentiates it. The closure must
capture every non-differentiable / auto-estimated quantity as a fixed constant
(``alpha``, neighbor list + integer shifts, and -- for the force path only --
the reciprocal set ``k_vectors``/``mesh_dimensions``), otherwise finite-diff and
autograd legitimately disagree. See ``_md/F3-verification-harness.md`` for the
full pinned-parameter rationale.

Tolerances (float64)
--------------------
- Central-difference step ``h = 1e-6`` (~ ``eps_f64**(1/3)``, the central-diff
  optimum) for positions, cell, strain and charges.
- Force/charge-grad/virial comparisons target ``rtol=1e-5, atol=1e-7`` on a
  non-symmetric (displaced) configuration so components are non-zero. The
  achieved deviation is returned by the comparison helpers for reporting.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch

__all__ = [
    "DerivCheckSystem",
    "EquivPoint",
    "EnergyFn",
    "DEFAULT_FD_EPS",
    "finite_difference_jacobian",
    "fd_forces",
    "fd_charge_grad",
    "fd_strain_virial",
    "autograd_forces",
    "autograd_charge_grad",
    "autograd_strain_virial",
    "max_abs_rel",
    "gradcheck_energy",
    "gradgradcheck_energy",
    "assert_tape_vs_explicit",
    "fixed_charge_system",
    "toy_charge_model",
]

# Central-difference step. ~ eps_f64**(1/3) balances O(h^2) truncation error
# against O(eps/h) round-off for a central difference in float64.
DEFAULT_FD_EPS = 1e-6

# An energy_fn maps (positions, charges, cell) -> per-atom (or per-system)
# energy Tensor. Callers reduce with ``.sum()``. The closure captures all
# pinned / non-differentiable parameters.
EnergyFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# Reusable system container
# ---------------------------------------------------------------------------


class DerivCheckSystem(NamedTuple):
    """A small periodic system ready for derivative checks.

    All tensors are on the same device/dtype. The neighbor list (indices +
    integer shifts) is built once at the reference geometry and must be held
    fixed across finite-difference perturbations.

    Attributes
    ----------
    positions : torch.Tensor, shape (N, 3)
    charges : torch.Tensor, shape (N,)
    cell : torch.Tensor, shape (S, 3, 3)
    batch_idx : torch.Tensor or None, shape (N,)
        ``None`` for single-system.
    neighbor_list : torch.Tensor, shape (2, M)
        COO neighbor pairs (int32).
    neighbor_ptr : torch.Tensor, shape (N + 1,)
        CSR row pointers (int32).
    neighbor_shifts : torch.Tensor, shape (M, 3)
        Integer periodic-image shifts (int32).
    alpha : torch.Tensor, shape (S,)
        Fixed Ewald splitting parameter (never auto-estimated).
    """

    positions: torch.Tensor
    charges: torch.Tensor
    cell: torch.Tensor
    batch_idx: torch.Tensor | None
    neighbor_list: torch.Tensor
    neighbor_ptr: torch.Tensor
    neighbor_shifts: torch.Tensor
    alpha: torch.Tensor


# ---------------------------------------------------------------------------
# Central-difference primitives
# ---------------------------------------------------------------------------


def finite_difference_jacobian(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = DEFAULT_FD_EPS,
) -> torch.Tensor:
    """Central-difference gradient of a scalar function ``fn`` at ``x``.

    Computes ``d fn / d x`` elementwise with a central difference,
    ``(fn(x + h e_k) - fn(x - h e_k)) / (2h)``, perturbing each entry of ``x``
    independently. ``fn`` must return a scalar (0-dim or single-element) Tensor.

    Parameters
    ----------
    fn : callable
        ``x (any shape) -> scalar Tensor``.
    x : torch.Tensor
        Point at which to differentiate. Should be float64 for accuracy.
    eps : float, default=1e-6
        Central-difference step.

    Returns
    -------
    torch.Tensor
        Gradient with the same shape as ``x``.

    Notes
    -----
    This evaluates ``fn`` ``2 * x.numel()`` times. Keep ``x`` small.
    """
    x = x.detach()
    flat = x.reshape(-1)
    grad = torch.zeros_like(flat)
    for k in range(flat.numel()):
        orig = flat[k].item()
        flat[k] = orig + eps
        f_plus = fn(x).reshape(()).item()
        flat[k] = orig - eps
        f_minus = fn(x).reshape(()).item()
        flat[k] = orig
        grad[k] = (f_plus - f_minus) / (2.0 * eps)
    return grad.reshape(x.shape)


def _energy_scalar(
    energy_fn: EnergyFn,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    """Reduce the per-atom energy_fn output to a scalar total energy."""
    return energy_fn(positions, charges, cell).sum()


# ---------------------------------------------------------------------------
# Finite-difference physical-quantity wrappers
# ---------------------------------------------------------------------------


def fd_forces(
    energy_fn: EnergyFn,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    eps: float = DEFAULT_FD_EPS,
) -> torch.Tensor:
    """Finite-difference forces ``-dE/dR``.

    Parameters
    ----------
    energy_fn : EnergyFn
        Closure ``(positions, charges, cell) -> (N,) energy``. Must pin alpha,
        neighbor list, and the reciprocal set so E(R) is smooth.
    positions, charges, cell : torch.Tensor
        Reference inputs (positions float64 recommended).
    eps : float, default=1e-6

    Returns
    -------
    torch.Tensor, shape (N, 3)
        ``-dE/dR``.
    """
    charges = charges.detach()
    cell = cell.detach()

    def scalar(p: torch.Tensor) -> torch.Tensor:
        return _energy_scalar(energy_fn, p, charges, cell)

    dE_dR = finite_difference_jacobian(scalar, positions.detach(), eps)
    return -dE_dR


def fd_charge_grad(
    energy_fn: EnergyFn,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    eps: float = DEFAULT_FD_EPS,
) -> torch.Tensor:
    """Finite-difference charge gradient ``dE/dq``.

    Returns
    -------
    torch.Tensor, shape (N,)
        ``dE/dq``.
    """
    positions = positions.detach()
    cell = cell.detach()

    def scalar(q: torch.Tensor) -> torch.Tensor:
        return _energy_scalar(energy_fn, positions, q, cell)

    return finite_difference_jacobian(scalar, charges.detach(), eps)


def _deform_inputs(
    positions: torch.Tensor,
    cell: torch.Tensor,
    strain: torch.Tensor,
    batch_idx: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``deform = eye + strain`` to positions and cell (scoping recipe).

    ``positions_s = positions @ deform[batch_idx]`` and
    ``cell_s = cell @ deform``. Single-system uses system 0 for all atoms.
    """
    num_systems = cell.shape[0]
    eye = torch.eye(3, device=positions.device, dtype=positions.dtype).unsqueeze(0)
    deform = eye + strain  # (S, 3, 3)

    if batch_idx is None:
        atom_sys = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=positions.device
        )
    else:
        atom_sys = batch_idx

    positions_s = torch.einsum("ni,nij->nj", positions, deform[atom_sys])
    cell_s = torch.einsum("bij,bjk->bik", cell, deform)
    _ = num_systems  # documented shape; silence linters
    return positions_s, cell_s


def fd_strain_virial(
    energy_fn: EnergyFn,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    eps: float = DEFAULT_FD_EPS,
) -> torch.Tensor:
    """Strain-first finite-difference virial ``-dE/dstrain``.

    Builds deformed positions/cell from a per-system strain tensor (the scoping
    doc §"Virial and Stress" recipe: ``deform = eye + strain``,
    ``positions_s = positions @ deform[batch_idx]``,
    ``cell_s = cell @ deform``)
    and finite-differences the total energy w.r.t. each of the nine strain
    entries per system.

    Unlike :func:`fd_forces`, the closure here must let the reciprocal set
    (``k_vectors`` / ``volume``) regenerate from the deformed cell -- only
    ``alpha`` (and PME ``mesh_dimensions``) is pinned -- so this measures the
    same quantity autograd produces via ``-grad(E.sum(), strain)``.

    Parameters
    ----------
    batch_idx : torch.Tensor or None
        Per-atom system index; ``None`` for single-system.

    Returns
    -------
    torch.Tensor, shape (S, 3, 3)
        ``-dE/dstrain``.
    """
    positions = positions.detach()
    charges = charges.detach()
    cell = cell.detach()
    num_systems = cell.shape[0]

    def scalar(strain: torch.Tensor) -> torch.Tensor:
        positions_s, cell_s = _deform_inputs(positions, cell, strain, batch_idx)
        return _energy_scalar(energy_fn, positions_s, charges, cell_s)

    strain0 = torch.zeros(
        num_systems, 3, 3, device=positions.device, dtype=positions.dtype
    )
    dE_dstrain = finite_difference_jacobian(scalar, strain0, eps)
    return -dE_dstrain


# ---------------------------------------------------------------------------
# Autograd counterparts (same closure)
# ---------------------------------------------------------------------------


def autograd_forces(
    energy_fn: EnergyFn,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    """Autograd forces ``-dE/dR`` from the same ``energy_fn`` closure."""
    p = positions.detach().clone().requires_grad_(True)
    energy = _energy_scalar(energy_fn, p, charges.detach(), cell.detach())
    (grad,) = torch.autograd.grad(energy, p, create_graph=create_graph)
    return -grad


def autograd_charge_grad(
    energy_fn: EnergyFn,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    """Autograd charge gradient ``dE/dq`` from the same ``energy_fn`` closure."""
    q = charges.detach().clone().requires_grad_(True)
    energy = _energy_scalar(energy_fn, positions.detach(), q, cell.detach())
    (grad,) = torch.autograd.grad(energy, q, create_graph=create_graph)
    return grad


def autograd_strain_virial(
    energy_fn: EnergyFn,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    create_graph: bool = False,
) -> torch.Tensor:
    """Strain-first autograd virial ``-dE/dstrain``."""
    positions = positions.detach()
    charges = charges.detach()
    cell = cell.detach()
    num_systems = cell.shape[0]

    strain = torch.zeros(
        num_systems,
        3,
        3,
        device=positions.device,
        dtype=positions.dtype,
        requires_grad=True,
    )
    positions_s, cell_s = _deform_inputs(positions, cell, strain, batch_idx)
    energy = _energy_scalar(energy_fn, positions_s, charges, cell_s)
    (grad,) = torch.autograd.grad(energy, strain, create_graph=create_graph)
    return -grad


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


def max_abs_rel(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Return ``(max_abs_dev, max_rel_dev)`` between two tensors.

    ``max_rel_dev`` uses ``|a - b| / max(|b|, tiny)`` to avoid blow-up where the
    reference is near zero.
    """
    a = a.detach().to(torch.float64)
    b = b.detach().to(torch.float64)
    diff = (a - b).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = b.abs().clamp_min(1e-300)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    return max_abs, max_rel


# ---------------------------------------------------------------------------
# gradcheck / gradgradcheck wrappers
# ---------------------------------------------------------------------------


def _select_grad_inputs(
    energy_fn: EnergyFn,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    wrt: Sequence[str],
) -> tuple[list[torch.Tensor], Callable[..., torch.Tensor]]:
    """Build the leaf list + scalar-returning wrapper for (gradgrad)check.

    Returns the differentiable leaves (in ``wrt`` order) and a function that maps
    those leaves to ``energy_fn(...).sum()``, substituting each leaf into its
    named slot while the unchecked inputs stay fixed.
    """
    base = {
        "positions": positions.detach().clone(),
        "charges": charges.detach().clone(),
        "cell": cell.detach().clone(),
    }
    names = list(wrt)
    leaves = [base[n].requires_grad_(True) for n in names]

    def func(*leaf_args: torch.Tensor) -> torch.Tensor:
        kw = {**base, **dict(zip(names, leaf_args))}
        return energy_fn(kw["positions"], kw["charges"], kw["cell"]).sum()

    return leaves, func


def gradcheck_energy(
    energy_fn: EnergyFn,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    *,
    wrt: Sequence[str] = ("positions", "charges", "cell"),
    eps: float = DEFAULT_FD_EPS,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    nondet_tol: float = 1e-6,
) -> bool:
    """First-order ``torch.autograd.gradcheck`` on ``E.sum()`` (float64).

    Parameters
    ----------
    wrt : sequence of {"positions", "charges", "cell"}
        Which inputs to check gradients for.
    eps, atol, rtol : float
        gradcheck tolerances. ``rtol`` is loose (1e-3) because the energy is the
        sum of erfc/FFT terms; ``atol`` carries the precision.
    nondet_tol : float, default=1e-6
        Non-zero because Warp reductions are not bit-deterministic between the
        two forward passes gradcheck runs.

    Returns
    -------
    bool
        ``True`` on success (gradcheck raises on failure).
    """
    leaves, func = _select_grad_inputs(energy_fn, positions, charges, cell, wrt)
    return torch.autograd.gradcheck(
        func,
        tuple(leaves),
        eps=eps,
        atol=atol,
        rtol=rtol,
        nondet_tol=nondet_tol,
    )


def gradgradcheck_energy(
    energy_fn: EnergyFn,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    *,
    wrt: Sequence[str] = ("positions", "charges", "cell"),
    eps: float = DEFAULT_FD_EPS,
    atol: float = 1e-4,
    rtol: float = 1e-2,
    nondet_tol: float = 1e-6,
) -> bool:
    """Second-order ``torch.autograd.gradgradcheck`` on ``E.sum()`` (float64).

    Provided for explicit double-backward kernels.

    Returns
    -------
    bool
        ``True`` on success (raises on failure).
    """
    leaves, func = _select_grad_inputs(energy_fn, positions, charges, cell, wrt)
    return torch.autograd.gradgradcheck(
        func,
        tuple(leaves),
        eps=eps,
        atol=atol,
        rtol=rtol,
        nondet_tol=nondet_tol,
    )


# ---------------------------------------------------------------------------
# Tape-vs-explicit equivalence
# ---------------------------------------------------------------------------


@dataclass
class EquivPoint:
    """One comparison point for :func:`assert_tape_vs_explicit`.

    Parameters
    ----------
    name : str
        Label (e.g. ``"forces"``, ``"charge_grad"``, ``"virial"``).
    explicit : callable
        ``() -> Tensor``. Usually a direct-kernel output (``compute_forces`` etc.).
        Swapping this callable is the only change needed to repoint the
        comparison at another explicit path.
    reference : callable
        ``() -> Tensor``. Today: an autograd-derived quantity. The trusted side.
    rtol, atol : float
        Comparison tolerances.
    expect_match : bool, default=True
        When ``False``, the point is *measured and reported* but a mismatch does
        not raise. Used today for the direct ``virial`` (a kernel observable, not
        a guaranteed strain-energy-derivative view -- the motivation for the
        refactor).
    """

    name: str
    explicit: Callable[[], torch.Tensor]
    reference: Callable[[], torch.Tensor]
    rtol: float = 1e-5
    atol: float = 1e-7
    expect_match: bool = True


def assert_tape_vs_explicit(points: Sequence[EquivPoint]) -> dict[str, dict]:
    """Compare explicit vs reference outputs at each parameterized point.

    Parameters
    ----------
    points : sequence of EquivPoint

    Returns
    -------
    dict
        ``{name: {"max_abs": float, "max_rel": float, "matched": bool}}``.

    Raises
    ------
    AssertionError
        Only for points with ``expect_match=True`` whose values differ by more
        than ``(rtol, atol)``.
    """
    report: dict[str, dict] = {}
    failures: list[str] = []
    for pt in points:
        a = pt.explicit()
        b = pt.reference()
        max_abs, max_rel = max_abs_rel(a, b)
        matched = bool(torch.allclose(a.to(b.dtype), b, rtol=pt.rtol, atol=pt.atol))
        report[pt.name] = {
            "max_abs": max_abs,
            "max_rel": max_rel,
            "matched": matched,
        }
        if pt.expect_match and not matched:
            failures.append(
                f"{pt.name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
                f"(rtol={pt.rtol}, atol={pt.atol})"
            )
    if failures:
        raise AssertionError("tape-vs-explicit mismatch:\n  " + "\n  ".join(failures))
    return report


# ---------------------------------------------------------------------------
# Reusable small systems + toy q(R)
# ---------------------------------------------------------------------------


def fixed_charge_system(
    generator: Callable[[int], object],
    size: int = 1,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    jitter: float = 0.0,
    cutoff: float = 5.0,
    seed: int = 0,
) -> DerivCheckSystem:
    """Build a small fixed-charge periodic system from a conftest generator.

    Parameters
    ----------
    generator : callable
        A conftest crystal generator (e.g. ``create_cscl_supercell``) returning a
        ``CrystalSystem`` NamedTuple with ``positions``, ``cell``, ``charges``.
    size : int, default=1
        Linear supercell size. ``1`` keeps systems tiny (2-8 atoms).
    jitter : float, default=0.0
        Std-dev (Angstrom) of a deterministic random displacement applied to
        positions so forces are non-zero (rtol-safe). ``0.0`` leaves the crystal
        at its symmetric equilibrium.
    cutoff : float, default=5.0
        Real-space cutoff for the neighbor list.
    seed : int, default=0
        Seed for the jitter RNG (reproducible).

    Returns
    -------
    DerivCheckSystem
        Single-system (``batch_idx=None``) with a fixed neighbor list and a
        fixed ``alpha=0.3``.
    """
    from nvalchemiops.torch.neighbors import cell_list

    system = generator(size)
    device = torch.device(device)

    positions = torch.as_tensor(system.positions, dtype=dtype, device=device)
    charges = torch.as_tensor(system.charges, dtype=dtype, device=device)
    cell = torch.as_tensor(system.cell, dtype=dtype, device=device).unsqueeze(0)

    if jitter > 0.0:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        disp = torch.randn(positions.shape, generator=gen, dtype=torch.float64).to(
            device=device, dtype=dtype
        )
        positions = positions + jitter * disp

    pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
    neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
        positions, cutoff, cell, pbc, return_neighbor_list=True
    )

    alpha = torch.tensor([0.3], dtype=dtype, device=device)

    return DerivCheckSystem(
        positions=positions,
        charges=charges,
        cell=cell,
        batch_idx=None,
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
        alpha=alpha,
    )


def toy_charge_model(
    positions: torch.Tensor,
    *,
    scale: float = 0.3,
    length: float = 5.0,
    batch_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tiny differentiable charge model ``q(R)`` with exact zero net charge.

    ``q_raw_i = scale * sin(sum_d positions[i, d] / length)`` is smooth and
    differentiable in positions. Per-system mean subtraction enforces
    ``sum_i q_i = 0`` exactly (required for a well-defined periodic Ewald sum).

    The mean subtraction couples every charge to every position in its system,
    which deliberately exercises the ``dE/dq * dq/dR`` chain-rule term when the
    charges are produced from ``positions`` with ``requires_grad=True``.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Differentiable atomic positions.
    scale : float, default=0.3
        Charge amplitude.
    length : float, default=5.0
        Length scale (Angstrom) inside the sine.
    batch_idx : torch.Tensor or None
        Per-atom system index for per-system mean subtraction. ``None`` =
        single system.

    Returns
    -------
    torch.Tensor, shape (N,)
        Charges with the same dtype/device as ``positions``, summing to ~0 per
        system.
    """
    q_raw = scale * torch.sin(positions.sum(dim=1) / length)

    if batch_idx is None:
        return q_raw - q_raw.mean()

    bidx = batch_idx
    num_systems = int(bidx.max().item()) + 1
    sums = torch.zeros(num_systems, dtype=q_raw.dtype, device=q_raw.device)
    counts = torch.zeros(num_systems, dtype=q_raw.dtype, device=q_raw.device)
    sums = sums.index_add(0, bidx, q_raw)
    counts = counts.index_add(0, bidx, torch.ones_like(q_raw))
    means = sums / counts
    return q_raw - means.index_select(0, bidx)

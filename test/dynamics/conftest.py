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

"""Shared fixtures and reference implementations for the dynamics tests.

Framework-neutral on purpose: this module must not import torch or jax, so that
it can be collected alongside both binding suites in a single process.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.optimizers.lbfgs import (
    _CELL_BUFFERS,
    _OPTIMIZER_BUFFERS,
    LBFGS_NEED_EVAL,
)

DEVICES = ["cuda:0"]

# L-BFGS pins every per-system scalar to float64 regardless of the coordinate
# precision, so the scalar column is float64 in both configurations. Copying
# the matched-precision pairs used by the FIRE2 tests would exercise a
# combination that is deliberately not registered.
DTYPE_CONFIGS = [
    pytest.param(wp.vec3f, wp.float64, np.float32, id="dof_f32"),
    pytest.param(wp.vec3d, wp.float64, np.float64, id="dof_f64"),
]


def make_lbfgs_state(num_dofs, num_systems, history_size, vec_dtype, device):
    """Allocate the caller-owned optimizer buffers, ready for the first step.

    The package allocates and initializes nothing, so this is a test-only
    factory. It returns the buffers keyed by name, in the canonical order, so
    tests can splat it with ``**state`` or pass ``*state.values()``
    positionally.
    """

    def f64(n):
        return wp.zeros(n, dtype=wp.float64, device=device)

    def f64_2d(a, b):
        return wp.zeros((a, b), dtype=wp.float64, device=device)

    def vec(n):
        return wp.zeros(n, dtype=vec_dtype, device=device)

    def i32(n):
        return wp.zeros(n, dtype=wp.int32, device=device)

    buffers = {
        "x_base": vec(num_dofs),
        "force_base": vec(num_dofs),
        "direction": vec(num_dofs),
        "s_history": wp.zeros((history_size, num_dofs), dtype=vec_dtype, device=device),
        "y_history": wp.zeros((history_size, num_dofs), dtype=vec_dtype, device=device),
        "ys": f64_2d(history_size, num_systems),
        "yy": f64_2d(history_size, num_systems),
        "alpha_hist": f64_2d(history_size, num_systems),
        "beta_hist": f64_2d(history_size, num_systems),
        "ss": f64(num_systems),
        "f_base": f64(num_systems),
        "gg": f64(num_systems),
        "gd": f64(num_systems),
        "fmax": f64(num_systems),
        "frms_sq": f64(num_systems),
        "smax": f64(num_systems),
        "d0": f64(num_systems),
        "dmax": f64(num_systems),
        "dquad": f64(num_systems),
        "alpha_step": f64(num_systems),
        "status": i32(num_systems),
        "iteration": i32(num_systems),
        "end": i32(num_systems),
        "n_loop": i32(num_systems),
        "ls_trials": i32(num_systems),
        "history_count": i32(num_systems),
    }
    _check_order(buffers, _OPTIMIZER_BUFFERS, "make_lbfgs_state")
    # Exactly three buffers do not start at zero.
    buffers["alpha_step"].fill_(1.0)
    buffers["iteration"].fill_(-1)
    buffers["status"].fill_(LBFGS_NEED_EVAL)
    return buffers


def _check_order(buffers, expected, where):
    """Fail loudly if a factory drifts from the canonical buffer order.

    Buffers are passed positionally, so a reordering here would silently swap
    two arrays and nothing downstream would notice.
    """
    if tuple(buffers) != tuple(expected):
        raise AssertionError(
            f"{where} returns buffers in the wrong order:\n"
            f"  expected: {tuple(expected)}\n"
            f"  got:      {tuple(buffers)}"
        )


def numpy_two_loop(s_vecs, y_vecs, ys, yy, q, gamma=None):
    """Reference two-loop recursion.

    Parameters
    ----------
    s_vecs, y_vecs : list of ndarray
        History pairs ordered newest first.
    ys, yy : sequence of float
        ``s . y`` and ``y . y`` for each pair, in the same order.
    q : ndarray
        Vector to apply the inverse-Hessian approximation to.
    gamma : float, optional
        Initial scaling. Defaults to ``ys[0] / yy[0]``; pass ``1.0`` to match an
        implementation that uses an identity initial inverse Hessian.
    """
    b = len(s_vecs)
    alpha = np.zeros(b)
    q = q.copy()
    for i in range(b):
        alpha[i] = (s_vecs[i] * q).sum() / ys[i]
        q = q - alpha[i] * y_vecs[i]
    r = (ys[0] / yy[0] if gamma is None else gamma) * q
    for i in range(b - 1, -1, -1):
        beta = (y_vecs[i] * r).sum() / ys[i]
        r = r + s_vecs[i] * (alpha[i] - beta)
    return r


def history_slots(end, history_count, history_size):
    """Ring-buffer slot indices for one system, ordered newest first."""
    return [
        ((end - 1 - t) % history_size + history_size) % history_size
        for t in range(history_count)
    ]


class Quadratic:
    """Separable quadratic ``E = 0.5 * sum_a k_a x_a^2`` with its minimum at 0.

    Anisotropic by default so that a first-order method needs many more steps
    than a quasi-Newton one, which is what makes it useful for comparing
    evaluation counts.
    """

    def __init__(self, k=(1.0, 4.0, 9.0)):
        self.k = np.asarray(k, dtype=np.float64)

    def energy_forces(self, positions):
        per_atom_energy = 0.5 * (self.k * positions**2).sum(axis=1)
        forces = -(self.k * positions)
        return np.ascontiguousarray(per_atom_energy), np.ascontiguousarray(forces)


def lower_triangular_cell(seed, diagonal_offset=2.0):
    """A random lower-triangular cell with lattice vectors as columns.

    Deliberately triclinic: cubic or isotropically scaled cells commute with
    one another, so they would pass a chart test under either multiplication
    order and could not detect a reversed convention.
    """
    rng = np.random.default_rng(seed)
    cell = np.tril(rng.normal(size=(3, 3)))
    cell[np.diag_indices(3)] = np.abs(cell[np.diag_indices(3)]) + diagonal_offset
    return cell


class CellPotential:
    """Harmonic in fractional coordinates, with a volume term.

    ``E = 0.5 k sum |s - s0|^2 + p V + c / V`` where ``s = H^-1 r``. The
    fractional term does not depend on the cell, which keeps the stress simple
    while still giving a well-posed variable-cell minimum: the volume relaxes
    to ``sqrt(c / p)`` and the fractional coordinates to ``s0``.
    """

    def __init__(self, s0, k=0.7, p=0.35, c=None):
        self.s0 = np.asarray(s0, dtype=np.float64)
        self.k = k
        self.p = p
        self.c = p * 400.0 if c is None else c

    @property
    def target_volume(self):
        return float(np.sqrt(self.c / self.p))

    def energy_forces_stress(self, positions, cell):
        cell_inv = np.linalg.inv(cell)
        frac = (cell_inv @ positions.T).T
        volume = abs(np.linalg.det(cell))
        delta = frac - self.s0
        energy = 0.5 * self.k * (delta**2).sum() + self.p * volume + self.c / volume
        forces = -(cell_inv.T @ (self.k * delta).T).T
        # dE/dH at fixed fractional coordinates
        d_energy_d_cell = (self.p - self.c / volume**2) * volume * cell_inv.T
        stress = d_energy_d_cell @ cell.T / volume
        # Transposes leave these non-contiguous, and `torch.tensor` preserves
        # numpy strides, so hand back contiguous arrays.
        return energy, np.ascontiguousarray(forces), np.ascontiguousarray(stress)


def make_lbfgs_cell_state(num_atoms, num_systems, vec_dtype, device, counts=None):
    """Allocate the caller-owned variable-cell buffers, keyed by name.

    ``kappa``, ``ext_batch_idx`` and ``ext_atom_ptr`` depend only on topology
    and are filled here; the reference cell still has to be captured with
    ``lbfgs_set_reference_cell``.

    Parameters
    ----------
    counts : sequence of int, optional
        Atom count per system, for a ragged batch. Defaults to an even split,
        which requires ``num_atoms`` to divide evenly.
    """
    from nvalchemiops.batch_utils import atom_ptr_to_batch_idx
    from nvalchemiops.dynamics.optimizers.lbfgs import lbfgs_cell_kappa
    from nvalchemiops.dynamics.utils.cell_filter import extend_atom_ptr

    mat_dtype = wp.mat33f if vec_dtype == wp.vec3f else wp.mat33d
    scalar_dtype = wp.float32 if vec_dtype == wp.vec3f else wp.float64
    num_ext = num_atoms + 2 * num_systems

    if counts is None:
        if num_atoms % num_systems:
            raise ValueError(
                f"num_atoms {num_atoms} does not divide evenly into {num_systems} "
                "systems; pass explicit counts for a ragged batch"
            )
        counts = [num_atoms // num_systems] * num_systems
    counts = np.asarray(counts, np.int32)
    if int(counts.sum()) != num_atoms:
        raise ValueError(f"counts sum to {counts.sum()}, not num_atoms {num_atoms}")

    # Build the extended topology with the generic batch utilities rather than
    # by hand, which is what makes ragged batches expressible at all.
    atom_ptr = wp.array(
        np.concatenate([[0], np.cumsum(counts)]).astype(np.int32),
        dtype=wp.int32,
        device=device,
    )
    ext_atom_ptr = wp.zeros(num_systems + 1, dtype=wp.int32, device=device)
    extend_atom_ptr(atom_ptr, ext_atom_ptr, device=device)
    ext_batch_idx = wp.zeros(num_ext, dtype=wp.int32, device=device)
    atom_ptr_to_batch_idx(ext_atom_ptr, ext_batch_idx)

    kappa = wp.zeros(num_systems, dtype=scalar_dtype, device=device)
    n_atoms_per_system = wp.array(counts, dtype=wp.int32, device=device)
    lbfgs_cell_kappa(
        n_atoms_per_system, kappa, cell_force_scale=1.0 / float(counts.max())
    )

    def mat(n):
        return wp.zeros(n, dtype=mat_dtype, device=device)

    def vec(n):
        return wp.zeros(n, dtype=vec_dtype, device=device)

    buffers = {
        "ref_cell": mat(num_systems),
        "ref_cell_inv": mat(num_systems),
        "kappa": kappa,
        "ext_batch_idx": ext_batch_idx,
        "ext_atom_ptr": ext_atom_ptr,
        "phi": mat(num_systems),
        "phi_inv": mat(num_systems),
        "d_phi": mat(num_systems),
        "cell_dof_a": vec(num_systems),
        "cell_dof_b": vec(num_systems),
        "cell_force_a": vec(num_systems),
        "cell_force_b": vec(num_systems),
        "ext_positions": vec(num_ext),
        "ext_forces": vec(num_ext),
    }
    _check_order(buffers, _CELL_BUFFERS, "make_lbfgs_cell_state")
    return buffers

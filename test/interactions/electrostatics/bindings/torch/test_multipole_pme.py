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

"""Tests for multipole Particle-Mesh Ewald (l = 0/1/2), single-system and batched."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import warp as wp

from nvalchemiops.math.spline import (
    bspline_grid_offset,
    bspline_weight_3d,
    bspline_weight_gradient_3d,
    compute_fractional_coords,
    wrap_grid_index,
)
from nvalchemiops.torch.interactions.electrostatics import (  # noqa: E402
    multipole_ewald_summation,
    pack_multipole_moments,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    cartesian_quadrupole_to_e3nn,
    pack_charges_dipoles,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_electrostatics import (
    multipole_reciprocal_space_energy,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
    _multipole_ewald_self_energy_per_atom,
)
from nvalchemiops.torch.interactions.electrostatics.pme import (
    pme_green_structure_factor,
)
from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (  # noqa: E402
    multipole_particle_mesh_ewald,
    multipole_pme_energy_corrections,
    multipole_pme_energy_corrections_per_atom,
    multipole_pme_gather_field,
    multipole_pme_gather_hessian,
    multipole_pme_gather_potential,
    multipole_pme_green_structure_factor,
    multipole_pme_reciprocal_space,
)


def _spread_reference(
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    cell: torch.Tensor,
    mesh_dimensions: tuple[int, int, int],
    spline_order: int = 4,
) -> torch.Tensor:
    """Reference-side spread for adjoint identity tests.

    Wraps ``multipole_pme_spread_unified`` with ``quadrupoles = 0`` and
    ``lmax = 1`` (dipole-supporting).
    """
    cell_3x3 = cell if cell.dim() == 2 else cell[0]
    cell_inv_t = torch.linalg.inv(cell_3x3).transpose(-1, -2).contiguous().unsqueeze(0)
    nx, ny, nz = mesh_dimensions
    N = positions.shape[0]
    zero_Q = torch.zeros((N, 3, 3), dtype=positions.dtype, device=positions.device)
    return torch.ops.nvalchemiops.multipole_pme_spread_unified(
        positions,
        charges,
        dipoles,
        zero_Q,
        cell_inv_t,
        nx,
        ny,
        nz,
        spline_order,
        1,
    )


@wp.kernel
def _eval_spline_at_grid_kernel(
    positions: wp.array(dtype=wp.vec3d),
    cell_inv_t: wp.array(dtype=wp.mat33d),
    order: wp.int32,
    mesh_dims: wp.vec3i,
    n_points: wp.int32,
    weights_out: wp.array(dtype=wp.float64),
    grad_frac_out: wp.array(dtype=wp.vec3d),
    grid_idx_out: wp.array(dtype=wp.vec3i),
):
    """For each (atom, point_idx) write ``(weight, grad_frac, grid_idx)``.

    ``n_points`` is a kernel arg because Python-int divisions on ``order``
    aren't well-supported in ``@wp.kernel`` bodies.
    """
    atom_idx, point_idx = wp.tid()
    pos = positions[atom_idx]

    base_grid, theta = compute_fractional_coords(pos, cell_inv_t[0], mesh_dims)
    offset = bspline_grid_offset(point_idx, order, theta)
    w = bspline_weight_3d(theta, offset, order)
    g_frac = bspline_weight_gradient_3d(theta, offset, order, mesh_dims)

    flat = atom_idx * n_points + point_idx
    weights_out[flat] = w
    grad_frac_out[flat] = g_frac
    grid_idx_out[flat] = wp.vec3i(
        wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0]),
        wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1]),
        wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2]),
    )


def _spline_oracle(positions: np.ndarray, cell: np.ndarray, mesh_dims, order: int):
    """Per-(atom, stencil) ``(weight, grad_frac, (gx, gy, gz))`` using the
    same Warp primitives the spread kernel uses.
    """
    n_atoms = positions.shape[0]
    n_points = order**3
    cell_inv_t_np = np.linalg.inv(cell).T.reshape(1, 3, 3).copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_wp = wp.from_numpy(positions, dtype=wp.vec3d, device=device)
    cell_inv_t_wp = wp.from_numpy(cell_inv_t_np, dtype=wp.mat33d, device=device)
    weights_wp = wp.zeros(n_atoms * n_points, dtype=wp.float64, device=device)
    grad_frac_wp = wp.zeros(n_atoms * n_points, dtype=wp.vec3d, device=device)
    grid_idx_wp = wp.zeros(n_atoms * n_points, dtype=wp.vec3i, device=device)

    wp.launch(
        _eval_spline_at_grid_kernel,
        dim=(n_atoms, n_points),
        inputs=[
            pos_wp,
            cell_inv_t_wp,
            order,
            wp.vec3i(int(mesh_dims[0]), int(mesh_dims[1]), int(mesh_dims[2])),
            n_points,
            weights_wp,
            grad_frac_wp,
            grid_idx_wp,
        ],
        device=device,
    )
    wp.synchronize()

    weights = weights_wp.numpy().reshape(n_atoms, n_points)
    grad_frac = np.array(grad_frac_wp.numpy().tolist()).reshape(n_atoms, n_points, 3)
    grid_idx = np.array(grid_idx_wp.numpy().tolist()).reshape(n_atoms, n_points, 3)
    return weights, grad_frac, grid_idx, cell_inv_t_np[0]


def _torch_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_k_squared_and_miller(mesh_dims, cell: np.ndarray, device, dtype):
    """Build ``(k_squared, miller_x, miller_y, miller_z)`` on the rfft grid.

    Replicates the monopole PME wrapper's internal geometry setup so the
    test can pass identical k-grid inputs to both wrappers.
    """
    nx, ny, nz = mesh_dims
    miller_x = torch.fft.fftfreq(nx, d=1.0 / nx, device=device, dtype=dtype)
    miller_y = torch.fft.fftfreq(ny, d=1.0 / ny, device=device, dtype=dtype)
    miller_z = torch.fft.rfftfreq(nz, d=1.0 / nz, device=device, dtype=dtype)

    # k-vectors in Cartesian: k = 2π · cell^{-T} · miller_vec.
    cell_t = torch.from_numpy(cell.T).to(device=device, dtype=dtype)
    inv_cell_t = torch.linalg.inv(cell_t)
    # k_squared[i, j, k] = |2π · inv(cell^T) · (mx, my, mz)|².
    mx, my, mz = torch.meshgrid(miller_x, miller_y, miller_z, indexing="ij")
    mvec = torch.stack([mx, my, mz], dim=-1)  # (nx, ny, nz_rfft, 3)
    kvec = 2.0 * math.pi * torch.einsum("ij,...j->...i", inv_cell_t, mvec)
    k_squared = (kvec * kvec).sum(dim=-1)
    return k_squared, miller_x, miller_y, miller_z


class TestGreenStructureFactor:
    """Multipole Green's function parity tests."""

    def test_monopole_recovery_sigma_zero(self):
        """σ = 0 recovers the existing monopole Green's function bit-for-bit."""
        if not torch.cuda.is_available():
            pytest.skip("kernel is GPU-only at this stage")
        L = 10.0
        cell_np = np.eye(3) * L
        mesh_dims = (16, 16, 16)
        alpha_val = 0.4
        spline_order = 4

        td = torch.device("cuda:0")
        dtype = torch.float64
        cell = torch.from_numpy(cell_np).to(td, dtype)

        k_squared, mx, my, mz = _make_k_squared_and_miller(
            mesh_dims, cell_np, td, dtype
        )
        alpha = torch.tensor([alpha_val], device=td, dtype=dtype)
        sigma = torch.zeros(1, device=td, dtype=dtype)
        volume = torch.tensor([L**3], device=td, dtype=dtype)

        green_multi, struct_multi = multipole_pme_green_structure_factor(
            k_squared,
            mx,
            my,
            mz,
            alpha,
            sigma,
            volume,
            mesh_dimensions=mesh_dims,
            spline_order=spline_order,
        )
        green_mono, struct_mono = pme_green_structure_factor(
            k_squared,
            mesh_dims,
            alpha,
            cell,
            spline_order=spline_order,
        )

        # σ = 0 collapses the extra exp factor to 1, recovering monopole. The
        # multipole and monopole Green/structure-factor kernels are separate
        # code paths (they agree to fp64 precision, not bit-for-bit).
        torch.testing.assert_close(green_multi, green_mono, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(struct_multi, struct_mono, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("sigma_val", [0.5, 1.0, 1.5])
    def test_gto_factor_analytical(self, sigma_val):
        """σ > 0 output equals the monopole Green's function scaled by ``exp(-σ²k²)``."""
        if not torch.cuda.is_available():
            pytest.skip("kernel is GPU-only at this stage")
        L = 10.0
        cell_np = np.eye(3) * L
        mesh_dims = (16, 16, 16)
        alpha_val = 0.4
        spline_order = 4

        td = torch.device("cuda:0")
        dtype = torch.float64
        cell = torch.from_numpy(cell_np).to(td, dtype)

        k_squared, mx, my, mz = _make_k_squared_and_miller(
            mesh_dims, cell_np, td, dtype
        )
        alpha = torch.tensor([alpha_val], device=td, dtype=dtype)
        sigma = torch.tensor([sigma_val], device=td, dtype=dtype)
        volume = torch.tensor([L**3], device=td, dtype=dtype)

        green_multi, struct_multi = multipole_pme_green_structure_factor(
            k_squared,
            mx,
            my,
            mz,
            alpha,
            sigma,
            volume,
            mesh_dimensions=mesh_dims,
            spline_order=spline_order,
        )
        green_mono, struct_mono = pme_green_structure_factor(
            k_squared,
            mesh_dims,
            alpha,
            cell,
            spline_order=spline_order,
        )

        # Pair-sum convolution of the per-side ``exp(-σ² k²/2)`` GTO factor.
        gto = torch.exp(-(sigma_val**2) * k_squared)
        expected_green = green_mono * gto

        torch.testing.assert_close(green_multi, expected_green, rtol=1e-12, atol=1e-14)
        # Structure factor is source-distribution independent (multipole and
        # monopole kernels agree to fp64 precision, not bit-for-bit).
        torch.testing.assert_close(struct_multi, struct_mono, rtol=1e-12, atol=1e-14)

    def test_k_zero_is_zero(self):
        """Tin-foil boundary: G̃(k=0) is zero regardless of σ."""
        if not torch.cuda.is_available():
            pytest.skip("kernel is GPU-only at this stage")
        L = 10.0
        cell_np = np.eye(3) * L
        mesh_dims = (16, 16, 16)

        td = torch.device("cuda:0")
        dtype = torch.float64
        k_squared, mx, my, mz = _make_k_squared_and_miller(
            mesh_dims, cell_np, td, dtype
        )
        alpha = torch.tensor([0.4], device=td, dtype=dtype)
        sigma = torch.tensor([1.0], device=td, dtype=dtype)
        volume = torch.tensor([L**3], device=td, dtype=dtype)

        green, _ = multipole_pme_green_structure_factor(
            k_squared,
            mx,
            my,
            mz,
            alpha,
            sigma,
            volume,
            mesh_dimensions=mesh_dims,
            spline_order=4,
        )
        assert float(green[0, 0, 0].item()) == 0.0


class TestGatherPotential:
    """Gather φ(r_i) from a potential grid."""

    def test_gather_constant_grid_partition_of_unity(self):
        """Gather of a constant grid equals that constant per atom.

        The gather kernel discards contributions with ``weight < 1e-8``,
        which introduces up to ``~64 × 1e-8`` absolute error; tolerance
        accommodates this.
        """
        if not torch.cuda.is_available():
            pytest.skip("kernel is GPU-only at this stage")
        td = torch.device("cuda:0")
        dtype = torch.float64
        L = 10.0
        N = 6
        rng = np.random.default_rng(0xC057)

        positions = torch.from_numpy(rng.uniform(1.0, L - 1.0, size=(N, 3))).to(
            td, dtype
        )
        cell = torch.eye(3, device=td, dtype=dtype) * L
        const = 2.5
        mesh = torch.full((16, 16, 16), const, dtype=dtype, device=td)

        phi = multipole_pme_gather_potential(mesh, positions, cell, spline_order=4)
        expected = torch.full((N,), const, dtype=dtype, device=td)
        torch.testing.assert_close(phi, expected, rtol=0, atol=1e-6)

    def test_adjoint_of_spread(self):
        """Adjoint identity ``⟨gather(mesh), q⟩ = ⟨mesh, spread(q)⟩``.

        Cross-checks that both wrappers use a consistent B-spline
        convention.
        """
        if not torch.cuda.is_available():
            pytest.skip("kernel is GPU-only at this stage")
        td = torch.device("cuda:0")
        dtype = torch.float64
        L = 10.0
        N = 4
        rng = np.random.default_rng(0xAD70)

        positions = torch.from_numpy(rng.uniform(1.0, L - 1.0, size=(N, 3))).to(
            td, dtype
        )
        cell = torch.eye(3, device=td, dtype=dtype) * L
        mesh_dims = (12, 12, 12)

        q_src = torch.from_numpy(rng.standard_normal(N)).to(td, dtype)
        mesh_arbitrary = torch.from_numpy(rng.standard_normal(mesh_dims)).to(td, dtype)

        # LHS: ⟨gather(mesh), q⟩
        phi = multipole_pme_gather_potential(
            mesh_arbitrary, positions, cell, spline_order=4
        )
        lhs = (phi * q_src).sum()

        # RHS: ⟨mesh, spread(q)⟩
        zero_dip = torch.zeros((N, 3), device=td, dtype=dtype)
        rho = _spread_reference(
            positions,
            q_src,
            zero_dip,
            cell,
            mesh_dimensions=mesh_dims,
            spline_order=4,
        )
        rhs = (rho * mesh_arbitrary).sum()

        # Gather's 1e-8 weight threshold puts ``|lhs - rhs|`` in the 1e-7
        # range (the spread side is exact).
        torch.testing.assert_close(lhs, rhs, rtol=1e-6, atol=1e-7)

    def test_gradcheck_mesh(self):
        """Backward w.r.t. mesh via ``torch.autograd.gradcheck``."""
        if not torch.cuda.is_available():
            pytest.skip("kernel is GPU-only at this stage")
        td = torch.device("cuda:0")
        rng = np.random.default_rng(0xED1)
        L = 8.0
        N = 3

        positions = torch.from_numpy(rng.uniform(1.0, L - 1.0, size=(N, 3))).to(
            td, torch.float64
        )
        cell = torch.eye(3, device=td, dtype=torch.float64) * L
        mesh = (
            torch.from_numpy(rng.standard_normal((8, 8, 8)))
            .to(td, torch.float64)
            .requires_grad_(True)
        )

        def fn(m: torch.Tensor) -> torch.Tensor:
            phi = multipole_pme_gather_potential(m, positions, cell, spline_order=4)
            return phi.sum()

        torch.autograd.gradcheck(
            fn,
            mesh,
            eps=1e-6,
            atol=1e-6,
            rtol=1e-6,
            fast_mode=True,
            nondet_tol=1e-12,
        )

    def test_gradcheck_positions(self):
        """Backward w.r.t. positions via ``torch.autograd.gradcheck``."""
        if not torch.cuda.is_available():
            pytest.skip("kernel is GPU-only at this stage")
        td = torch.device("cuda:0")
        rng = np.random.default_rng(0xED2)
        L = 8.0
        N = 3

        positions = (
            torch.from_numpy(rng.uniform(1.5, L - 1.5, size=(N, 3)))
            .to(td, torch.float64)
            .requires_grad_(True)
        )
        cell = torch.eye(3, device=td, dtype=torch.float64) * L
        mesh = torch.from_numpy(rng.standard_normal((8, 8, 8))).to(td, torch.float64)

        def fn(p: torch.Tensor) -> torch.Tensor:
            phi = multipole_pme_gather_potential(mesh, p, cell, spline_order=4)
            # Random per-atom weights so the gradient signal doesn't cancel.
            w = torch.from_numpy(np.random.default_rng(0xED2A).standard_normal(N)).to(
                td, torch.float64
            )
            return (phi * w).sum()

        torch.autograd.gradcheck(
            fn,
            positions,
            eps=1e-6,
            atol=1e-6,
            rtol=1e-6,
            fast_mode=True,
            nondet_tol=1e-12,
        )


class TestGatherField:
    """Gather ∇φ(r_i) from a potential grid."""

    def test_field_of_constant_grid_is_zero(self):
        """``∇(const) = 0``; also checks the gather gives ``∇φ`` not ``-∇φ``."""
        if not torch.cuda.is_available():
            pytest.skip("kernel is GPU-only at this stage")
        td = torch.device("cuda:0")
        dtype = torch.float64
        L = 10.0
        N = 5
        rng = np.random.default_rng(0xF1)
        positions = torch.from_numpy(rng.uniform(1.0, L - 1.0, size=(N, 3))).to(
            td, dtype
        )
        cell = torch.eye(3, device=td, dtype=dtype) * L
        mesh = torch.full((16, 16, 16), 7.5, dtype=dtype, device=td)

        field = multipole_pme_gather_field(mesh, positions, cell, spline_order=4)
        torch.testing.assert_close(field, torch.zeros_like(field), rtol=0, atol=1e-6)

    def test_field_adjoint_of_dipole_spread(self):
        r"""``⟨gather_field(mesh), μ⟩ = ⟨mesh, spread(μ as dipoles)⟩``.

        Cross-checks the gather_field convention against the spread's
        dipole branch.
        """
        if not torch.cuda.is_available():
            pytest.skip("kernel is GPU-only at this stage")
        td = torch.device("cuda:0")
        dtype = torch.float64
        L = 10.0
        N = 4
        rng = np.random.default_rng(0xF2)

        positions = torch.from_numpy(rng.uniform(1.0, L - 1.0, size=(N, 3))).to(
            td, dtype
        )
        cell = torch.eye(3, device=td, dtype=dtype) * L
        mesh_dims = (12, 12, 12)
        mu = torch.from_numpy(rng.standard_normal((N, 3))).to(td, dtype) * 0.4
        mesh = torch.from_numpy(rng.standard_normal(mesh_dims)).to(td, dtype)

        # LHS: ⟨gather_field(mesh), μ⟩
        field = multipole_pme_gather_field(mesh, positions, cell, spline_order=4)
        lhs = (field * mu).sum()

        # RHS: ⟨mesh, spread(μ as dipoles)⟩
        zero_q = torch.zeros(N, device=td, dtype=dtype)
        rho = _spread_reference(
            positions,
            zero_q,
            mu,
            cell,
            mesh_dimensions=mesh_dims,
            spline_order=4,
        )
        rhs = (rho * mesh).sum()

        # Same gather threshold story as the potential adjoint (1e-7 abs).
        torch.testing.assert_close(lhs, rhs, rtol=1e-6, atol=1e-7)

    def test_field_finite_difference_vs_potential(self):
        """``∇φ`` vs central FD on the potential gather."""
        if not torch.cuda.is_available():
            pytest.skip("kernel is GPU-only at this stage")
        td = torch.device("cuda:0")
        dtype = torch.float64
        L = 10.0
        N = 3
        rng = np.random.default_rng(0xF3)

        positions = torch.from_numpy(rng.uniform(1.5, L - 1.5, size=(N, 3))).to(
            td, dtype
        )
        cell = torch.eye(3, device=td, dtype=dtype) * L
        mesh = torch.from_numpy(rng.standard_normal((16, 16, 16))).to(td, dtype)

        field = multipole_pme_gather_field(mesh, positions, cell, spline_order=4)

        h = 1e-3
        fd = torch.zeros_like(field)
        for axis in range(3):
            shift = torch.zeros(3, device=td, dtype=dtype)
            shift[axis] = h
            phi_plus = multipole_pme_gather_potential(
                mesh, positions + shift, cell, spline_order=4
            )
            phi_minus = multipole_pme_gather_potential(
                mesh, positions - shift, cell, spline_order=4
            )
            fd[:, axis] = (phi_plus - phi_minus) / (2 * h)

        torch.testing.assert_close(field, fd, rtol=1e-4, atol=1e-5)

    def test_gradcheck_mesh(self):
        """Backward w.r.t. ``mesh`` via gradcheck."""
        if not torch.cuda.is_available():
            pytest.skip("kernel is GPU-only at this stage")
        td = torch.device("cuda:0")
        rng = np.random.default_rng(0xF4)
        L = 8.0
        N = 3

        positions = torch.from_numpy(rng.uniform(1.0, L - 1.0, size=(N, 3))).to(
            td, torch.float64
        )
        cell = torch.eye(3, device=td, dtype=torch.float64) * L
        mesh = (
            torch.from_numpy(rng.standard_normal((8, 8, 8)))
            .to(td, torch.float64)
            .requires_grad_(True)
        )

        def fn(m: torch.Tensor) -> torch.Tensor:
            field = multipole_pme_gather_field(m, positions, cell, spline_order=4)
            return field.sum()

        torch.autograd.gradcheck(
            fn,
            mesh,
            eps=1e-6,
            atol=1e-6,
            rtol=1e-6,
            fast_mode=True,
            nondet_tol=1e-12,
        )

    def test_gradcheck_positions(self):
        """Backward w.r.t. ``positions`` via gradcheck."""
        if not torch.cuda.is_available():
            pytest.skip("kernel is GPU-only at this stage")
        td = torch.device("cuda:0")
        rng = np.random.default_rng(0xF4_42)
        L = 8.0
        N = 3
        # Push positions away from integer grid points so FD doesn't
        # straddle spline breakpoints.
        positions = (
            torch.from_numpy(rng.uniform(1.5, L - 1.5, size=(N, 3)))
            .to(td, torch.float64)
            .requires_grad_(True)
        )
        cell = torch.eye(3, device=td, dtype=torch.float64) * L
        mesh = torch.from_numpy(rng.standard_normal((8, 8, 8))).to(td, torch.float64)

        def fn(pos: torch.Tensor) -> torch.Tensor:
            field = multipole_pme_gather_field(mesh, pos, cell, spline_order=4)
            return field.sum()

        torch.autograd.gradcheck(
            fn,
            positions,
            eps=1e-6,
            atol=1e-6,
            rtol=1e-6,
            fast_mode=True,
            nondet_tol=1e-12,
        )


class TestGatherHessian:
    """Gather ∇²φ(r_i) from a potential grid."""

    def test_hessian_of_constant_grid_is_zero(self):
        """``∇²(constant) = 0`` per atom."""
        if not torch.cuda.is_available():
            pytest.skip("kernel is GPU-only at this stage")
        td = torch.device("cuda:0")
        dtype = torch.float64
        L = 10.0
        N = 5
        rng = np.random.default_rng(0xF5)
        positions = torch.from_numpy(rng.uniform(1.0, L - 1.0, size=(N, 3))).to(
            td, dtype
        )
        cell = torch.eye(3, device=td, dtype=dtype) * L
        mesh = torch.full((16, 16, 16), 7.5, dtype=dtype, device=td)

        H = multipole_pme_gather_hessian(mesh, positions, cell, spline_order=4)
        assert H.shape == (N, 3, 3)
        torch.testing.assert_close(H, torch.zeros_like(H), rtol=0, atol=1e-5)

    def test_hessian_is_symmetric(self):
        """Returned ``(3, 3)`` matrices are symmetric (kernel only writes
        the upper-triangle entries to ``off``; wrapper mirrors them)."""
        if not torch.cuda.is_available():
            pytest.skip("kernel is GPU-only at this stage")
        td = torch.device("cuda:0")
        dtype = torch.float64
        L = 10.0
        N = 4
        rng = np.random.default_rng(0xF6)
        positions = torch.from_numpy(rng.uniform(1.0, L - 1.0, size=(N, 3))).to(
            td, dtype
        )
        cell = torch.eye(3, device=td, dtype=dtype) * L
        mesh = torch.from_numpy(rng.standard_normal((16, 16, 16))).to(td, dtype)

        H = multipole_pme_gather_hessian(mesh, positions, cell, spline_order=4)
        torch.testing.assert_close(H, H.transpose(-1, -2), rtol=0, atol=1e-14)

    def test_hessian_finite_difference_vs_field(self):
        """Gathered Hessian vs central FD on the field gather."""
        if not torch.cuda.is_available():
            pytest.skip("kernel is GPU-only at this stage")
        td = torch.device("cuda:0")
        dtype = torch.float64
        L = 10.0
        N = 3
        rng = np.random.default_rng(0xF7)

        positions = torch.from_numpy(rng.uniform(1.5, L - 1.5, size=(N, 3))).to(
            td, dtype
        )
        cell = torch.eye(3, device=td, dtype=dtype) * L
        mesh = torch.from_numpy(rng.standard_normal((16, 16, 16))).to(td, dtype)

        H = multipole_pme_gather_hessian(mesh, positions, cell, spline_order=4)

        # FD via the field gather. ``H[α, β] = ∂(∇φ_β)/∂r_α``.
        h = 1e-3
        H_fd = torch.zeros_like(H)
        for axis in range(3):
            shift = torch.zeros(3, device=td, dtype=dtype)
            shift[axis] = h
            field_plus = multipole_pme_gather_field(
                mesh, positions + shift, cell, spline_order=4
            )
            field_minus = multipole_pme_gather_field(
                mesh, positions - shift, cell, spline_order=4
            )
            # H[i, axis, β] = (field_plus[i, β] - field_minus[i, β]) / (2h)
            H_fd[:, axis, :] = (field_plus - field_minus) / (2 * h)

        # Symmetrize the FD oracle to match the kernel's symmetric
        # output — FD can produce asymmetric (Hxy ≠ Hyx) values at
        # ULP scale due to roundoff in the field gather.
        H_fd_sym = 0.5 * (H_fd + H_fd.transpose(-1, -2))
        torch.testing.assert_close(H, H_fd_sym, rtol=1e-3, atol=1e-5)

    def test_gradcheck_mesh(self):
        """Backward w.r.t. ``mesh`` via gradcheck."""
        if not torch.cuda.is_available():
            pytest.skip("kernel is GPU-only at this stage")
        td = torch.device("cuda:0")
        rng = np.random.default_rng(0xF7)
        L = 8.0
        N = 3
        positions = torch.from_numpy(rng.uniform(1.5, L - 1.5, size=(N, 3))).to(
            td, torch.float64
        )
        cell = torch.eye(3, device=td, dtype=torch.float64) * L
        mesh = (
            torch.from_numpy(rng.standard_normal((8, 8, 8)))
            .to(td, torch.float64)
            .requires_grad_(True)
        )

        def fn(m: torch.Tensor) -> torch.Tensor:
            H = multipole_pme_gather_hessian(m, positions, cell, spline_order=4)
            return H.sum()

        torch.autograd.gradcheck(
            fn,
            mesh,
            eps=1e-6,
            atol=1e-6,
            rtol=1e-6,
            fast_mode=True,
            nondet_tol=1e-12,
        )

    def test_gradcheck_positions(self):
        """Backward w.r.t. ``positions`` via gradcheck."""
        if not torch.cuda.is_available():
            pytest.skip("kernel is GPU-only at this stage")
        td = torch.device("cuda:0")
        rng = np.random.default_rng(0xF7_42)
        L = 8.0
        N = 3
        positions = (
            torch.from_numpy(rng.uniform(1.5, L - 1.5, size=(N, 3)))
            .to(td, torch.float64)
            .requires_grad_(True)
        )
        cell = torch.eye(3, device=td, dtype=torch.float64) * L
        mesh = torch.from_numpy(rng.standard_normal((8, 8, 8))).to(td, torch.float64)

        def fn(pos: torch.Tensor) -> torch.Tensor:
            H = multipole_pme_gather_hessian(mesh, pos, cell, spline_order=4)
            return H.sum()

        torch.autograd.gradcheck(
            fn,
            positions,
            eps=1e-6,
            atol=1e-6,
            rtol=1e-6,
            fast_mode=True,
            nondet_tol=1e-12,
        )


def _path_a_pack_source_feats(
    charges: torch.Tensor, dipoles: torch.Tensor
) -> torch.Tensor:
    """Pack ``(charges, dipoles_xyz)`` into Path A's e3nn ``[q, μ_y, μ_z, μ_x]`` layout.

    The ``|μ|²`` in the self-energy is rotation-invariant, so the e3nn
    permutation doesn't affect the result.
    """
    n = charges.shape[0]
    sf = torch.zeros((n, 4), dtype=torch.float64, device=charges.device)
    sf[:, 0] = charges
    sf[:, 1] = dipoles[:, 1]  # μ_y
    sf[:, 2] = dipoles[:, 2]  # μ_z
    sf[:, 3] = dipoles[:, 0]  # μ_x
    return sf


class TestEnergyCorrections:
    """Self + background corrections vs Path A reference."""

    @pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5])
    @pytest.mark.parametrize("alpha", [0.3, 0.6, 0.9])
    def test_self_energy_matches_path_a(self, sigma, alpha):
        """Per-atom self-energy matches Path A's ``_multipole_ewald_self_energy_per_atom``.

        Both use the GTO-Ewald self-overlap (``σ_c = √(σ² + 1/(4α²))``)
        and the same ``FIELD_CONSTANT`` units.
        """
        if not torch.cuda.is_available():
            pytest.skip("Path A reference uses Warp launchers (GPU-only)")
        td = torch.device("cuda:0")
        rng = np.random.default_rng(0xE6)
        N = 12
        L = 10.0

        charges = torch.from_numpy(rng.uniform(-1.0, 1.0, N)).to(td, torch.float64)
        charges = charges - charges.mean()  # neutralize for clean comparison
        dipoles = torch.from_numpy(rng.standard_normal((N, 3)) * 0.4).to(
            td, torch.float64
        )
        volume = torch.tensor(L**3, device=td, dtype=torch.float64)

        # Correction = E_self + E_bg; for a neutral system E_bg = 0.
        ours = multipole_pme_energy_corrections(
            charges, dipoles, sigma=sigma, alpha=alpha, volume=volume
        )

        # Path A oracle.
        sf = _path_a_pack_source_feats(charges, dipoles)
        path_a = _multipole_ewald_self_energy_per_atom(sf, sigma, alpha).sum()

        torch.testing.assert_close(ours, path_a, rtol=1e-15, atol=1e-15)

    def test_background_for_non_neutral(self):
        """Background term ``F π Q² / (2 α² V)`` for non-neutral systems."""
        td = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        N = 4
        L = 8.0
        # All-positive charges → Q_total = N · q.
        charges = torch.full((N,), 1.0, dtype=torch.float64, device=td)
        dipoles = torch.zeros((N, 3), dtype=torch.float64, device=td)
        volume = torch.tensor(L**3, device=td, dtype=torch.float64)
        sigma = 1.0
        alpha = 0.5

        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        pi32 = math.pi**1.5

        # Expected: E_self + E_bg.
        F = 1.0 / 5.526349406e-3  # FIELD_CONSTANT.
        e_self_expected = (F / (8.0 * pi32 * sigma_c)) * sum(c**2 for c in [1.0] * N)
        e_bg_expected = (F * math.pi / (2.0 * alpha**2 * L**3)) * (N * 1.0) ** 2
        expected = e_self_expected + e_bg_expected

        got = multipole_pme_energy_corrections(
            charges, dipoles, sigma=sigma, alpha=alpha, volume=volume
        )
        torch.testing.assert_close(
            got,
            torch.tensor(expected, dtype=torch.float64, device=td),
            rtol=1e-12,
            atol=1e-10,
        )

    def test_batched_per_system_corrections(self):
        """Batched call returns per-system corrections of shape ``(B,)``."""
        if not torch.cuda.is_available():
            pytest.skip("Path A reference uses Warp launchers (GPU-only)")
        td = torch.device("cuda:0")
        rng = np.random.default_rng(0xBA7)
        B = 3
        N_per = 4
        L = 8.0
        sigma = 1.0
        alpha = 0.6

        charges_list = []
        dipoles_list = []
        for _ in range(B):
            q = rng.uniform(-1.0, 1.0, N_per)
            q = q - q.mean()
            charges_list.append(q)
            dipoles_list.append(rng.standard_normal((N_per, 3)) * 0.4)
        charges = torch.from_numpy(np.concatenate(charges_list)).to(td, torch.float64)
        dipoles = torch.from_numpy(np.concatenate(dipoles_list, axis=0)).to(
            td, torch.float64
        )
        batch_idx = torch.repeat_interleave(
            torch.arange(B, dtype=torch.int32, device=td), N_per
        )
        volume = torch.tensor([L**3] * B, device=td, dtype=torch.float64)

        per_sys = multipole_pme_energy_corrections(
            charges,
            dipoles,
            sigma=sigma,
            alpha=alpha,
            volume=volume,
            batch_idx=batch_idx,
        )
        assert per_sys.shape == (B,)

        # Each system independently: per-system Path A oracle.
        for b in range(B):
            mask = batch_idx == b
            sf_b = _path_a_pack_source_feats(charges[mask], dipoles[mask])
            expected_b = _multipole_ewald_self_energy_per_atom(sf_b, sigma, alpha).sum()
            torch.testing.assert_close(per_sys[b], expected_b, rtol=1e-15, atol=1e-15)


class TestEnergyCorrectionsPerAtom:
    """Per-atom corrections (Phase 4 component): ``Σ_i`` + ``grad`` match the
    reduced :func:`multipole_pme_energy_corrections` op bit-for-bit."""

    @pytest.mark.parametrize("l_max", [0, 1, 2])
    @pytest.mark.parametrize("batched", [False, True])
    def test_sum_and_grad_match_reduced(self, l_max, batched):
        td = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        rng = np.random.default_rng(0x9E + l_max + (10 if batched else 0))
        sigma, alpha = 0.9, 0.37

        if batched:
            batch_idx = torch.tensor(
                [0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.int32, device=td
            )
            N = batch_idx.numel()
            volume = torch.tensor([700.0, 800.0, 900.0], dtype=torch.float64, device=td)
        else:
            batch_idx = None
            N = 8
            volume = torch.tensor(729.0, dtype=torch.float64, device=td)

        # Non-neutral so the background term is exercised.
        charges = torch.from_numpy(rng.uniform(-1.0, 1.0, N)).to(td, torch.float64)
        dipoles = (
            torch.from_numpy(rng.standard_normal((N, 3)) * 0.4).to(td, torch.float64)
            if l_max >= 1
            else None
        )
        if l_max >= 2:
            q_full = torch.from_numpy(rng.standard_normal((N, 3, 3)) * 0.3).to(
                td, torch.float64
            )
            quadrupoles = 0.5 * (q_full + q_full.mT)
        else:
            quadrupoles = None

        def _reduce(per_atom):
            if batch_idx is None:
                return per_atom.sum()
            out = torch.zeros(3, dtype=torch.float64, device=td)
            return out.scatter_add(0, batch_idx, per_atom)

        # Value: Σ per-atom == reduced.
        per_atom = multipole_pme_energy_corrections_per_atom(
            charges,
            dipoles,
            sigma,
            alpha,
            volume,
            batch_idx=batch_idx,
            quadrupoles=quadrupoles,
        )
        assert per_atom.shape == (N,)
        reduced = multipole_pme_energy_corrections(
            charges,
            dipoles,
            sigma=sigma,
            alpha=alpha,
            volume=volume,
            batch_idx=batch_idx,
            quadrupoles=quadrupoles,
        )
        torch.testing.assert_close(_reduce(per_atom), reduced, rtol=1e-13, atol=1e-13)

        # Gradient: grad(Σ per-atom, charges) == grad(reduced, charges).
        q1 = charges.clone().requires_grad_(True)
        g_pa = torch.autograd.grad(
            _reduce(
                multipole_pme_energy_corrections_per_atom(
                    q1,
                    dipoles,
                    sigma,
                    alpha,
                    volume,
                    batch_idx=batch_idx,
                    quadrupoles=quadrupoles,
                )
            ).sum(),
            q1,
        )[0]
        q2 = charges.clone().requires_grad_(True)
        g_red = torch.autograd.grad(
            multipole_pme_energy_corrections(
                q2,
                dipoles,
                sigma=sigma,
                alpha=alpha,
                volume=volume,
                batch_idx=batch_idx,
                quadrupoles=quadrupoles,
            ).sum(),
            q2,
        )[0]
        torch.testing.assert_close(g_pa, g_red, rtol=1e-12, atol=1e-12)


class TestReciprocalSpace:
    """Full PME reciprocal pipeline vs Path A reciprocal-self.

    Each call returns ``E_recip_pme - E_self - E_bg`` in Path A's
    ``FIELD_CONSTANT``-scaled units. The oracle is the raw reciprocal sum
    minus the self correction. Convergence floor is set by mesh density
    (``mesh=(60, 60, 60)`` on ``L = 10``) and ``k_cutoff = 12``,
    pushing the residual below ``rtol = 1e-4``.
    """

    def _setup_neutral_system(self, N: int, L: float, seed: int):
        """Random neutral, well-separated atoms in a cubic box."""
        td = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        rng = np.random.default_rng(seed)
        positions = torch.from_numpy(rng.uniform(0.5, L - 0.5, size=(N, 3))).to(
            td, torch.float64
        )
        charges = torch.from_numpy(rng.uniform(-1.0, 1.0, N)).to(td, torch.float64)
        charges = charges - charges.mean()  # neutralize
        dipoles = torch.from_numpy(rng.standard_normal((N, 3)) * 0.4).to(
            td, torch.float64
        )
        cell = torch.eye(3, dtype=torch.float64, device=td) * L
        return td, positions, charges, dipoles, cell

    def _path_a_oracle(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        dipoles: torch.Tensor,
        cell: torch.Tensor,
        sigma: float,
        alpha: float,
        k_cutoff: float,
    ) -> torch.Tensor:
        """Path A reciprocal piece minus the self correction.

        Extracts only the reciprocal side of the ``multipole_ewald_summation``
        decomposition, matching the reciprocal-only wrapper under test.
        """
        N = positions.shape[0]
        sf = torch.zeros((N, 4), dtype=torch.float64, device=positions.device)
        sf[:, 0] = charges
        sf[:, 1] = dipoles[:, 1]  # μ_y
        sf[:, 2] = dipoles[:, 2]  # μ_z
        sf[:, 3] = dipoles[:, 0]  # μ_x
        recip = multipole_reciprocal_space_energy(
            positions, sf, cell, sigma=sigma, alpha=alpha, k_cutoff=k_cutoff
        )
        e_self = _multipole_ewald_self_energy_per_atom(sf, sigma, alpha).sum()
        return recip.sum() - e_self

    @pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5])
    @pytest.mark.parametrize("alpha", [0.4, 0.6])
    @pytest.mark.parametrize("spline_order", [4, 5, 6])
    def test_charges_only_matches_path_a(self, sigma, alpha, spline_order):
        """Pure-charge system parity vs Path A reciprocal-self.

        ``rtol = 1e-4`` accounts for spline-truncation residuals at mesh
        ``(60, 60, 60)``; higher orders only tighten the residual.
        """
        if not torch.cuda.is_available():
            pytest.skip("Path A reference uses Warp launchers (GPU-only)")
        td, positions, charges, _, cell = self._setup_neutral_system(
            N=8, L=10.0, seed=0xABCD
        )
        dipoles = torch.zeros((8, 3), dtype=torch.float64, device=td)

        ours = multipole_pme_reciprocal_space(
            positions,
            pack_multipole_moments(charges, dipoles),
            cell,
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=(60, 60, 60),
            spline_order=spline_order,
        )
        target = self._path_a_oracle(
            positions, charges, dipoles, cell, sigma, alpha, k_cutoff=12.0
        )
        torch.testing.assert_close(ours.sum(), target, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5])
    @pytest.mark.parametrize("alpha", [0.4, 0.6])
    @pytest.mark.parametrize("spline_order", [4, 5, 6])
    def test_charges_plus_dipoles_matches_path_a(self, sigma, alpha, spline_order):
        """Charges + dipoles parity vs Path A reciprocal-self."""
        if not torch.cuda.is_available():
            pytest.skip("Path A reference uses Warp launchers (GPU-only)")
        _, positions, charges, dipoles, cell = self._setup_neutral_system(
            N=8, L=10.0, seed=0xBEEF
        )

        ours = multipole_pme_reciprocal_space(
            positions,
            pack_multipole_moments(charges, dipoles),
            cell,
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=(60, 60, 60),
            spline_order=spline_order,
        )
        target = self._path_a_oracle(
            positions, charges, dipoles, cell, sigma, alpha, k_cutoff=12.0
        )
        torch.testing.assert_close(ours.sum(), target, rtol=1e-4, atol=1e-4)

    def test_translation_invariance(self):
        """Shifting all atoms by a lattice vector leaves the energy invariant."""
        if not torch.cuda.is_available():
            pytest.skip("Composite is GPU-only at this stage")
        td, positions, charges, dipoles, cell = self._setup_neutral_system(
            N=8, L=10.0, seed=0xCAFE
        )
        sigma, alpha = 1.0, 0.5
        e1 = multipole_pme_reciprocal_space(
            positions,
            pack_multipole_moments(charges, dipoles),
            cell,
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=(40, 40, 40),
            spline_order=4,
        )
        shift = torch.tensor([2.5, -1.7, 3.3], dtype=torch.float64, device=td)
        e2 = multipole_pme_reciprocal_space(
            positions + shift,
            pack_multipole_moments(charges, dipoles),
            cell,
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=(40, 40, 40),
            spline_order=4,
        )
        # Translation invariance up to PME aliasing (~1e-5 at this mesh).
        torch.testing.assert_close(e1.sum(), e2.sum(), rtol=1e-4, atol=1e-4)

    def test_position_backward_runs(self):
        """Autograd through the full composite returns finite gradients on positions."""
        if not torch.cuda.is_available():
            pytest.skip("Composite is GPU-only at this stage")
        _, positions, charges, dipoles, cell = self._setup_neutral_system(
            N=4, L=8.0, seed=0xD00D
        )
        positions = positions.clone().requires_grad_(True)
        sigma, alpha = 1.0, 0.5
        e = multipole_pme_reciprocal_space(
            positions,
            pack_multipole_moments(charges, dipoles),
            cell,
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=(32, 32, 32),
            spline_order=4,
        )
        e.sum().backward()
        assert positions.grad is not None
        assert positions.grad.shape == positions.shape
        assert torch.isfinite(positions.grad).all()
        assert positions.grad.abs().max() > 0

    def test_dipoles_none_matches_dipoles_zeros(self):
        """``dipoles=None`` (l_max=0 fast path) is bit-exact vs explicit zeros.

        The ``None`` path skips the field gather and dipole self-energy
        term, which all vanish when dipoles are zero.
        """
        if not torch.cuda.is_available():
            pytest.skip("Composite is GPU-only at this stage")
        td, positions, charges, _, cell = self._setup_neutral_system(
            N=6, L=10.0, seed=0xFADE
        )
        dipoles_zero = torch.zeros((6, 3), dtype=torch.float64, device=td)
        sigma, alpha = 1.0, 0.5
        mesh = (40, 40, 40)
        e_zeros = multipole_pme_reciprocal_space(
            positions,
            pack_multipole_moments(charges, dipoles_zero),
            cell,
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=mesh,
        )
        e_none = multipole_pme_reciprocal_space(
            positions,
            pack_multipole_moments(charges, None),
            cell,
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=mesh,
        )
        torch.testing.assert_close(e_none.sum(), e_zeros.sum(), rtol=0, atol=0)


def _o_n2_csr_neighbors(
    positions: np.ndarray, L: float, cutoff: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Brute-force ``O(N²·shells)`` CSR neighbor list within ``cutoff``.

    Test-only; tiny systems (N=8) make the O(N²) cost negligible.
    """
    n = positions.shape[0]
    shell = int(math.ceil(cutoff / L)) + 1
    idx_j: list[int] = []
    nptr: list[int] = [0]
    shifts: list[list[int]] = []
    for i in range(n):
        for sa in range(-shell, shell + 1):
            for sb in range(-shell, shell + 1):
                for sc in range(-shell, shell + 1):
                    for j in range(n):
                        if j == i and (sa, sb, sc) == (0, 0, 0):
                            continue
                        r = positions[j] - positions[i] + np.array([sa, sb, sc]) * L
                        if np.linalg.norm(r) < cutoff:
                            idx_j.append(j)
                            shifts.append([sa, sb, sc])
        nptr.append(len(idx_j))
    return (
        np.array(idx_j, np.int32),
        np.array(nptr, np.int32),
        np.array(shifts, np.int32),
    )


class TestParticleMeshEwald:
    """Top-level composite parity vs ``multipole_ewald_summation``.

    The composite returns ``E_real + E_recip - E_self - E_bg``; Path A
    omits the background (its reciprocal sum zeroes ``k=0``). For neutral
    systems the background vanishes and the two agree to the
    spline-truncation floor.
    """

    def _setup(self, N: int, L: float, seed: int, sigma: float, alpha: float):
        """Build a neutral random system + matching CSR neighbor list."""
        td = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        rng = np.random.default_rng(seed)
        positions_np = rng.uniform(0.5, L - 0.5, size=(N, 3))
        charges_np = rng.uniform(-1.0, 1.0, N)
        charges_np -= charges_np.mean()
        dipoles_np = rng.standard_normal((N, 3)) * 0.4
        cell_np = np.eye(3) * L

        # 10·σ_c gives a ULP-level erfc tail.
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c
        idx_j_np, nptr_np, sh_np = _o_n2_csr_neighbors(positions_np, L, cutoff)

        positions = torch.from_numpy(positions_np).to(td, torch.float64)
        charges = torch.from_numpy(charges_np).to(td, torch.float64)
        dipoles = torch.from_numpy(dipoles_np).to(td, torch.float64)
        cell = torch.from_numpy(cell_np).to(td, torch.float64)
        idx_j = torch.from_numpy(idx_j_np).to(td)
        nptr = torch.from_numpy(nptr_np).to(td)
        sh = torch.from_numpy(sh_np).to(td)
        k_cutoff = 6.0 / sigma_c
        return (
            td,
            positions,
            charges,
            dipoles,
            cell,
            idx_j,
            nptr,
            sh,
            k_cutoff,
        )

    @pytest.mark.parametrize("alpha", [0.4, 0.6])
    @pytest.mark.parametrize("sigma", [0.8, 1.0, 1.2])
    @pytest.mark.parametrize("spline_order", [4, 5, 6])
    def test_charges_only_matches_path_a(self, sigma, alpha, spline_order):
        """l_max=0 parity: Path A vs PME on a neutral random system."""
        if not torch.cuda.is_available():
            pytest.skip("Path A reference uses Warp launchers (GPU-only)")
        N = 8
        L = 10.0
        _, positions, charges, _, cell, idx_j, nptr, sh, kcut = self._setup(
            N=N, L=L, seed=0xABCD, sigma=sigma, alpha=alpha
        )
        sf = pack_charges_dipoles(charges, None)

        e_a = multipole_ewald_summation(
            positions,
            sf,
            cell,
            idx_j,
            nptr,
            sh,
            sigma=sigma,
            alpha=alpha,
            k_cutoff=kcut,
        )
        e_pme = multipole_particle_mesh_ewald(
            positions,
            sf,
            cell,
            idx_j,
            nptr,
            sh,
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=(60, 60, 60),
            spline_order=spline_order,
        )
        torch.testing.assert_close(e_pme.sum(), e_a.sum(), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("alpha", [0.4, 0.6])
    @pytest.mark.parametrize("sigma", [0.8, 1.0, 1.2])
    @pytest.mark.parametrize("spline_order", [4, 5, 6])
    def test_charges_plus_dipoles_matches_path_a(self, sigma, alpha, spline_order):
        """l_max=1 parity: Path A vs PME with dipoles enabled."""
        if not torch.cuda.is_available():
            pytest.skip("Path A reference uses Warp launchers (GPU-only)")
        N = 8
        L = 10.0
        _, positions, charges, dipoles, cell, idx_j, nptr, sh, kcut = self._setup(
            N=N, L=L, seed=0xBEEF, sigma=sigma, alpha=alpha
        )
        sf = pack_charges_dipoles(charges, dipoles)

        e_a = multipole_ewald_summation(
            positions,
            sf,
            cell,
            idx_j,
            nptr,
            sh,
            sigma=sigma,
            alpha=alpha,
            k_cutoff=kcut,
        )
        e_pme = multipole_particle_mesh_ewald(
            positions,
            sf,
            cell,
            idx_j,
            nptr,
            sh,
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=(60, 60, 60),
            spline_order=spline_order,
        )
        torch.testing.assert_close(e_pme.sum(), e_a.sum(), rtol=1e-4, atol=1e-4)

    def test_position_backward_runs(self):
        """Autograd flows through real + reciprocal halves to position gradients."""
        if not torch.cuda.is_available():
            pytest.skip("Composite is GPU-only at this stage")
        sigma, alpha = 1.0, 0.5
        _, positions, charges, dipoles, cell, idx_j, nptr, sh, _ = self._setup(
            N=6, L=8.0, seed=0xD00D, sigma=sigma, alpha=alpha
        )
        sf = pack_charges_dipoles(charges, dipoles)
        positions = positions.clone().requires_grad_(True)
        sf = sf.clone().requires_grad_(True)
        e = multipole_particle_mesh_ewald(
            positions,
            sf,
            cell,
            idx_j,
            nptr,
            sh,
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=(40, 40, 40),
            spline_order=4,
        )
        e.sum().backward()
        assert positions.grad is not None
        assert positions.grad.shape == positions.shape
        assert torch.isfinite(positions.grad).all()
        assert positions.grad.abs().max() > 0
        assert sf.grad is not None
        assert torch.isfinite(sf.grad).all()


class TestBatchedGather:
    """Batched gather_{potential, field} parity vs per-system loop.

    Each atom's gathered ``φ(r_i)`` / ``∇φ(r_i)`` must equal a per-system
    call on the corresponding mesh slice. Field-wrapper backward only
    implements the mesh side; positions are deferred.
    """

    def _build_two_systems(self, seed: int = 0xCAFE):
        td = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        rng = np.random.default_rng(seed)
        N1, N2 = 3, 5
        L = 8.0
        nx, ny, nz = 16, 16, 16
        pos1 = torch.from_numpy(rng.uniform(0.5, L - 0.5, size=(N1, 3))).to(
            td, torch.float64
        )
        pos2 = torch.from_numpy(rng.uniform(0.5, L - 0.5, size=(N2, 3))).to(
            td, torch.float64
        )
        mesh1 = torch.from_numpy(rng.standard_normal((nx, ny, nz))).to(
            td, torch.float64
        )
        mesh2 = torch.from_numpy(rng.standard_normal((nx, ny, nz))).to(
            td, torch.float64
        )
        cell = torch.eye(3, dtype=torch.float64, device=td) * L
        return td, (pos1, mesh1), (pos2, mesh2), cell

    def test_forward_potential_matches_per_system_loop(self):
        """Batched ``φ(r_i)`` matches B per-system gathers bit-for-bit."""
        if not torch.cuda.is_available():
            pytest.skip("multipole_pme_gather_potential is GPU-only")
        td, sys1, sys2, cell = self._build_two_systems()
        pos1, mesh1 = sys1
        pos2, mesh2 = sys2
        N1, N2 = pos1.shape[0], pos2.shape[0]

        phi1 = multipole_pme_gather_potential(mesh1, pos1, cell)
        phi2 = multipole_pme_gather_potential(mesh2, pos2, cell)

        positions = torch.cat([pos1, pos2], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(N1, dtype=torch.int32, device=td),
                torch.ones(N2, dtype=torch.int32, device=td),
            ]
        )
        mesh_batch = torch.stack([mesh1, mesh2], dim=0)
        cells_batch = torch.stack([cell, cell], dim=0)

        phi_batch = multipole_pme_gather_potential(
            mesh_batch, positions, cells_batch, batch_idx=batch_idx
        )
        assert phi_batch.shape == (N1 + N2,)
        # ULP-level sum-order difference between batched and per-system
        # accumulation; 1e-14 is well below any meaningful floor.
        torch.testing.assert_close(phi_batch[:N1], phi1, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(phi_batch[N1:], phi2, rtol=1e-14, atol=1e-14)

    def test_forward_field_matches_per_system_loop(self):
        """Batched ``∇φ(r_i)`` matches B per-system gathers bit-for-bit."""
        if not torch.cuda.is_available():
            pytest.skip("multipole_pme_gather_field is GPU-only")
        td, sys1, sys2, cell = self._build_two_systems()
        pos1, mesh1 = sys1
        pos2, mesh2 = sys2
        N1, N2 = pos1.shape[0], pos2.shape[0]

        field1 = multipole_pme_gather_field(mesh1, pos1, cell)
        field2 = multipole_pme_gather_field(mesh2, pos2, cell)

        positions = torch.cat([pos1, pos2], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(N1, dtype=torch.int32, device=td),
                torch.ones(N2, dtype=torch.int32, device=td),
            ]
        )
        mesh_batch = torch.stack([mesh1, mesh2], dim=0)
        cells_batch = torch.stack([cell, cell], dim=0)

        field_batch = multipole_pme_gather_field(
            mesh_batch, positions, cells_batch, batch_idx=batch_idx
        )
        assert field_batch.shape == (N1 + N2, 3)
        # Same ULP-level sum-order difference as the potential test.
        torch.testing.assert_close(field_batch[:N1], field1, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(field_batch[N1:], field2, rtol=1e-14, atol=1e-14)

    def test_gather_potential_gradcheck_mesh(self):
        """Mesh-slot gradcheck on the batched potential gather."""
        if not torch.cuda.is_available():
            pytest.skip("multipole_pme_gather_potential is GPU-only")
        td = torch.device("cuda:0")
        rng = np.random.default_rng(0xBA)
        N1, N2 = 2, 3
        L = 6.0
        positions = torch.from_numpy(rng.uniform(1.5, L - 1.5, size=(N1 + N2, 3))).to(
            td, torch.float64
        )
        mesh = (
            torch.from_numpy(rng.standard_normal((2, 8, 8, 8)))
            .to(td, torch.float64)
            .requires_grad_(True)
        )
        batch_idx = torch.cat(
            [
                torch.zeros(N1, dtype=torch.int32, device=td),
                torch.ones(N2, dtype=torch.int32, device=td),
            ]
        )
        cells = torch.stack(
            [torch.eye(3, dtype=torch.float64, device=td) * L for _ in range(2)]
        )

        def fn(m):
            phi = multipole_pme_gather_potential(
                m, positions, cells, batch_idx=batch_idx
            )
            return phi.sum()

        torch.autograd.gradcheck(
            fn, mesh, eps=1e-6, atol=1e-6, rtol=1e-6, fast_mode=True, nondet_tol=1e-12
        )

    def test_gather_potential_gradcheck_positions(self):
        """Position-slot gradcheck on the batched potential gather."""
        if not torch.cuda.is_available():
            pytest.skip("multipole_pme_gather_potential is GPU-only")
        td = torch.device("cuda:0")
        rng = np.random.default_rng(0xBE)
        N1, N2 = 2, 3
        L = 6.0
        positions = (
            torch.from_numpy(rng.uniform(1.5, L - 1.5, size=(N1 + N2, 3)))
            .to(td, torch.float64)
            .requires_grad_(True)
        )
        mesh = torch.from_numpy(rng.standard_normal((2, 8, 8, 8))).to(td, torch.float64)
        batch_idx = torch.cat(
            [
                torch.zeros(N1, dtype=torch.int32, device=td),
                torch.ones(N2, dtype=torch.int32, device=td),
            ]
        )
        cells = torch.stack(
            [torch.eye(3, dtype=torch.float64, device=td) * L for _ in range(2)]
        )

        def fn(p):
            phi = multipole_pme_gather_potential(mesh, p, cells, batch_idx=batch_idx)
            return phi.sum()

        torch.autograd.gradcheck(
            fn,
            positions,
            eps=1e-6,
            atol=1e-5,
            rtol=1e-5,
            fast_mode=True,
            nondet_tol=1e-12,
        )

    def test_gather_field_gradcheck_mesh(self):
        """Mesh-slot gradcheck on the batched field gather."""
        if not torch.cuda.is_available():
            pytest.skip("multipole_pme_gather_field is GPU-only")
        td = torch.device("cuda:0")
        rng = np.random.default_rng(0xFE)
        N1, N2 = 2, 3
        L = 6.0
        positions = torch.from_numpy(rng.uniform(1.5, L - 1.5, size=(N1 + N2, 3))).to(
            td, torch.float64
        )
        mesh = (
            torch.from_numpy(rng.standard_normal((2, 8, 8, 8)))
            .to(td, torch.float64)
            .requires_grad_(True)
        )
        batch_idx = torch.cat(
            [
                torch.zeros(N1, dtype=torch.int32, device=td),
                torch.ones(N2, dtype=torch.int32, device=td),
            ]
        )
        cells = torch.stack(
            [torch.eye(3, dtype=torch.float64, device=td) * L for _ in range(2)]
        )

        def fn(m):
            field = multipole_pme_gather_field(m, positions, cells, batch_idx=batch_idx)
            return field.sum()

        torch.autograd.gradcheck(
            fn, mesh, eps=1e-6, atol=1e-6, rtol=1e-6, fast_mode=True, nondet_tol=1e-12
        )


class TestBatchedGreenStructureFactor:
    """Batched Green's function parity vs B per-system kernels."""

    def _build(self, B: int, mesh_dim: int):
        td = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        mx = torch.fft.fftfreq(
            mesh_dim, d=1.0 / mesh_dim, device=td, dtype=torch.float64
        )
        my = torch.fft.fftfreq(
            mesh_dim, d=1.0 / mesh_dim, device=td, dtype=torch.float64
        )
        mz = torch.fft.rfftfreq(
            mesh_dim, d=1.0 / mesh_dim, device=td, dtype=torch.float64
        )

        # Per-system cells with different volumes — exercises per-system
        # alpha/sigma/volume independently of the shared k-grid.
        cells = torch.stack(
            [torch.eye(3, dtype=torch.float64, device=td) * (8.0 + b) for b in range(B)]
        )
        volumes = torch.tensor(
            [(8.0 + b) ** 3 for b in range(B)], dtype=torch.float64, device=td
        )
        alphas = torch.tensor(
            [0.4 + 0.1 * b for b in range(B)], dtype=torch.float64, device=td
        )
        sigmas = torch.tensor([1.0] * B, dtype=torch.float64, device=td)

        def build_k_sq(cell):
            inv_t = torch.linalg.inv(cell.T)
            mxg, myg, mzg = torch.meshgrid(mx, my, mz, indexing="ij")
            mvec = torch.stack([mxg, myg, mzg], dim=-1)
            kvec = 2.0 * math.pi * torch.einsum("ij,...j->...i", inv_t, mvec)
            return (kvec * kvec).sum(-1)

        k_sq_per = [build_k_sq(c) for c in cells]
        k_sq_batch = torch.stack(k_sq_per, dim=0)
        return td, k_sq_per, k_sq_batch, mx, my, mz, alphas, sigmas, volumes

    def test_forward_matches_per_system_loop(self):
        """Batched ``(G̃_b, |C|²)`` matches B per-system kernel calls."""
        if not torch.cuda.is_available():
            pytest.skip("multipole_pme_green_structure_factor is GPU-only")
        B = 3
        mesh_dim = 16
        _, k_sq_per, k_sq_batch, mx, my, mz, alphas, sigmas, volumes = self._build(
            B, mesh_dim
        )

        green_per = []
        struct_per = None
        for b in range(B):
            g, s = multipole_pme_green_structure_factor(
                k_sq_per[b],
                mx,
                my,
                mz,
                alphas[b : b + 1],
                sigmas[b : b + 1],
                volumes[b : b + 1],
                mesh_dimensions=(mesh_dim,) * 3,
                spline_order=4,
            )
            green_per.append(g)
            if struct_per is None:
                struct_per = s

        green_batch, struct_batch = multipole_pme_green_structure_factor(
            k_sq_batch,
            mx,
            my,
            mz,
            alphas,
            sigmas,
            volumes,
            mesh_dimensions=(mesh_dim,) * 3,
            spline_order=4,
        )

        assert green_batch.shape == (B, mesh_dim, mesh_dim, mesh_dim // 2 + 1)
        assert struct_batch.shape == (mesh_dim, mesh_dim, mesh_dim // 2 + 1)
        for b in range(B):
            torch.testing.assert_close(green_batch[b], green_per[b], rtol=0, atol=0)
        # The per-system structure_factor_sq is identical across systems
        # (mesh geometry is the same), so any one of them should match.
        torch.testing.assert_close(struct_batch, struct_per, rtol=0, atol=0)

    def test_k_zero_zeroed_per_system(self):
        """Tin-foil boundary holds independently for each batch slice."""
        if not torch.cuda.is_available():
            pytest.skip("multipole_pme_green_structure_factor is GPU-only")
        B = 3
        mesh_dim = 16
        _, _, k_sq_batch, mx, my, mz, alphas, sigmas, volumes = self._build(B, mesh_dim)
        green_batch, _ = multipole_pme_green_structure_factor(
            k_sq_batch,
            mx,
            my,
            mz,
            alphas,
            sigmas,
            volumes,
            mesh_dimensions=(mesh_dim,) * 3,
            spline_order=4,
        )
        for b in range(B):
            assert float(green_batch[b, 0, 0, 0].item()) == 0.0


def _build_batched_csr(
    systems: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    L: float,
    cutoff: float,
):
    """Build a flat batched CSR neighbor list from per-system positions.

    Each system's `idx_j` indices are offset by the cumulative atom count
    so the flat list points into the concatenated atom array; `nptr` is
    concatenated with running edge counts.
    """
    n_offsets = [0]
    for p, _, _ in systems:
        n_offsets.append(n_offsets[-1] + p.shape[0])
    idx_j_b: list[int] = []
    nptr_b: list[int] = [0]
    sh_b: list[list[int]] = []
    for b, (p, _, _) in enumerate(systems):
        idx_j, nptr, sh = _o_n2_csr_neighbors(p, L, cutoff)
        for atom_i in range(p.shape[0]):
            edges = nptr[atom_i + 1] - nptr[atom_i]
            for e in range(nptr[atom_i], nptr[atom_i + 1]):
                idx_j_b.append(int(idx_j[e]) + n_offsets[b])
                sh_b.append(list(sh[e]))
            nptr_b.append(nptr_b[-1] + edges)
    return (
        np.array(idx_j_b, np.int32),
        np.array(nptr_b, np.int32),
        np.array(sh_b, np.int32),
    )


class TestBatchedReciprocalSpace:
    """Batched reciprocal-space composite parity vs per-system loop.

    Output ``(B,)`` per-system reciprocal energies must match B
    independent calls within fp64 precision; sum-order differences
    account for the ``rtol = 1e-12`` budget.
    """

    def _build(self, B: int, L: float, seed_base: int):
        td = torch.device("cuda:0")
        rng = np.random.default_rng(seed_base)
        sizes = [3 + (b % 3) for b in range(B)]  # vary atom count per system
        systems = []
        for size in sizes:
            p = rng.uniform(0.5, L - 0.5, (size, 3))
            q = rng.uniform(-1, 1, size)
            q -= q.mean()
            d = rng.standard_normal((size, 3)) * 0.4
            systems.append((p, q, d))
        return td, systems

    def test_recip_matches_per_system_loop(self):
        """Batched reciprocal energies match B per-system computations."""
        if not torch.cuda.is_available():
            pytest.skip("multipole PME composites are GPU-only at this stage")
        td, systems = self._build(B=3, L=10.0, seed_base=0xCAFE)
        sigma, alpha, mesh_dim = 1.0, 0.5, 40
        L = 10.0

        # Per-system loop.
        per_system = []
        for p, q, d in systems:
            P = torch.from_numpy(p).to(td, torch.float64)
            Q = torch.from_numpy(q).to(td, torch.float64)
            D = torch.from_numpy(d).to(td, torch.float64)
            cell = torch.eye(3, dtype=torch.float64, device=td) * L
            e = multipole_pme_reciprocal_space(
                P,
                pack_multipole_moments(Q, D),
                cell,
                sigma=sigma,
                alpha=alpha,
                mesh_dimensions=(mesh_dim,) * 3,
            )
            per_system.append(float(e.sum()))

        positions = torch.from_numpy(np.concatenate([s[0] for s in systems])).to(
            td, torch.float64
        )
        charges = torch.from_numpy(np.concatenate([s[1] for s in systems])).to(
            td, torch.float64
        )
        dipoles = torch.from_numpy(np.concatenate([s[2] for s in systems])).to(
            td, torch.float64
        )
        batch_idx = torch.cat(
            [
                torch.full((s[0].shape[0],), b, dtype=torch.int32, device=td)
                for b, s in enumerate(systems)
            ]
        )
        cells = torch.stack(
            [torch.eye(3, dtype=torch.float64, device=td) * L for _ in systems]
        )

        B = 3
        e_batch = multipole_pme_reciprocal_space(
            positions,
            pack_multipole_moments(charges, dipoles),
            cells,
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=(mesh_dim,) * 3,
            batch_idx=batch_idx,
        )
        assert e_batch.shape == (positions.shape[0],)
        e_batch_per_sys = torch.zeros(B, dtype=torch.float64, device=td).scatter_add(
            0, batch_idx.long(), e_batch
        )
        for b in range(B):
            torch.testing.assert_close(
                e_batch_per_sys[b],
                torch.tensor(per_system[b], dtype=torch.float64, device=td),
                rtol=1e-12,
                atol=1e-12,
            )

    def test_recip_position_backward_runs(self):
        """Autograd flows through the batched reciprocal composite to positions."""
        if not torch.cuda.is_available():
            pytest.skip("multipole PME composites are GPU-only at this stage")
        td, systems = self._build(B=2, L=8.0, seed_base=0xD00D)
        positions = (
            torch.from_numpy(np.concatenate([s[0] for s in systems]))
            .to(td, torch.float64)
            .requires_grad_(True)
        )
        charges = torch.from_numpy(np.concatenate([s[1] for s in systems])).to(
            td, torch.float64
        )
        dipoles = torch.from_numpy(np.concatenate([s[2] for s in systems])).to(
            td, torch.float64
        )
        batch_idx = torch.cat(
            [
                torch.full((s[0].shape[0],), b, dtype=torch.int32, device=td)
                for b, s in enumerate(systems)
            ]
        )
        cells = torch.stack(
            [torch.eye(3, dtype=torch.float64, device=td) * 8.0 for _ in systems]
        )
        e = multipole_pme_reciprocal_space(
            positions,
            pack_multipole_moments(charges, dipoles),
            cells,
            sigma=1.0,
            alpha=0.5,
            mesh_dimensions=(32, 32, 32),
            batch_idx=batch_idx,
        )
        e.sum().backward()
        assert positions.grad is not None
        assert positions.grad.shape == positions.shape
        assert torch.isfinite(positions.grad).all()
        assert positions.grad.abs().max() > 0


class TestBatchedParticleMeshEwald:
    """Batched top-level composite parity vs per-system loop.

    ``multipole_particle_mesh_ewald`` returns ``(B,)`` per-system totals
    matching independent calls; fp64 sum-order drift sets the 1e-10 budget.
    """

    def test_total_matches_per_system_loop(self):
        """Batched total energies match B per-system computations."""
        if not torch.cuda.is_available():
            pytest.skip("multipole PME composites are GPU-only at this stage")
        td = torch.device("cuda:0")
        L = 10.0
        sigma, alpha, mesh_dim = 1.0, 0.5, 40
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c

        rng = np.random.default_rng(0xBABE)
        systems = []
        for _ in range(3):
            size = rng.integers(3, 6)
            p = rng.uniform(0.5, L - 0.5, (size, 3))
            q = rng.uniform(-1, 1, size)
            q -= q.mean()
            d = rng.standard_normal((size, 3)) * 0.4
            systems.append((p, q, d))

        per_system = []
        for p, q, d in systems:
            P = torch.from_numpy(p).to(td, torch.float64)
            Q = torch.from_numpy(q).to(td, torch.float64)
            D = torch.from_numpy(d).to(td, torch.float64)
            cell = torch.eye(3, dtype=torch.float64, device=td) * L
            sf = pack_charges_dipoles(Q, D)
            idx_j, nptr, sh = _o_n2_csr_neighbors(p, L, cutoff)
            e = multipole_particle_mesh_ewald(
                P,
                sf,
                cell,
                torch.from_numpy(idx_j).to(td),
                torch.from_numpy(nptr).to(td),
                torch.from_numpy(sh).to(td),
                sigma=sigma,
                alpha=alpha,
                mesh_dimensions=(mesh_dim,) * 3,
            )
            per_system.append(float(e.sum()))

        positions = torch.from_numpy(np.concatenate([s[0] for s in systems])).to(
            td, torch.float64
        )
        charges = torch.from_numpy(np.concatenate([s[1] for s in systems])).to(
            td, torch.float64
        )
        dipoles = torch.from_numpy(np.concatenate([s[2] for s in systems])).to(
            td, torch.float64
        )
        sf = pack_charges_dipoles(charges, dipoles)
        batch_idx = torch.cat(
            [
                torch.full((s[0].shape[0],), b, dtype=torch.int32, device=td)
                for b, s in enumerate(systems)
            ]
        )
        cells = torch.stack(
            [torch.eye(3, dtype=torch.float64, device=td) * L for _ in systems]
        )

        B = 3
        idx_j_b, nptr_b, sh_b = _build_batched_csr(systems, L, cutoff)
        e_batch = multipole_particle_mesh_ewald(
            positions,
            sf,
            cells,
            torch.from_numpy(idx_j_b).to(td),
            torch.from_numpy(nptr_b).to(td),
            torch.from_numpy(sh_b).to(td),
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=(mesh_dim,) * 3,
            batch_idx=batch_idx,
        )
        assert e_batch.shape == (positions.shape[0],)
        e_batch_per_sys = torch.zeros(B, dtype=torch.float64, device=td).scatter_add(
            0, batch_idx.long(), e_batch
        )
        for b in range(B):
            torch.testing.assert_close(
                e_batch_per_sys[b],
                torch.tensor(per_system[b], dtype=torch.float64, device=td),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_total_position_backward_runs(self):
        """Autograd flows through the batched top-level composite."""
        if not torch.cuda.is_available():
            pytest.skip("multipole PME composites are GPU-only at this stage")
        td = torch.device("cuda:0")
        L = 8.0
        sigma, alpha = 1.0, 0.5
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c

        rng = np.random.default_rng(0xC0DE)
        systems = []
        for _ in range(2):
            size = 4
            p = rng.uniform(0.5, L - 0.5, (size, 3))
            q = rng.uniform(-1, 1, size)
            q -= q.mean()
            d = rng.standard_normal((size, 3)) * 0.3
            systems.append((p, q, d))

        positions = (
            torch.from_numpy(np.concatenate([s[0] for s in systems]))
            .to(td, torch.float64)
            .requires_grad_(True)
        )
        charges = torch.from_numpy(np.concatenate([s[1] for s in systems])).to(
            td, torch.float64
        )
        dipoles = torch.from_numpy(np.concatenate([s[2] for s in systems])).to(
            td, torch.float64
        )
        sf = pack_charges_dipoles(charges, dipoles).requires_grad_(True)
        batch_idx = torch.cat(
            [
                torch.full((s[0].shape[0],), b, dtype=torch.int32, device=td)
                for b, s in enumerate(systems)
            ]
        )
        cells = torch.stack(
            [torch.eye(3, dtype=torch.float64, device=td) * L for _ in systems]
        )

        idx_j_b, nptr_b, sh_b = _build_batched_csr(systems, L, cutoff)
        e = multipole_particle_mesh_ewald(
            positions,
            sf,
            cells,
            torch.from_numpy(idx_j_b).to(td),
            torch.from_numpy(nptr_b).to(td),
            torch.from_numpy(sh_b).to(td),
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=(32, 32, 32),
            batch_idx=batch_idx,
        )
        e.sum().backward()
        assert positions.grad is not None
        assert positions.grad.shape == positions.shape
        assert torch.isfinite(positions.grad).all()
        assert positions.grad.abs().max() > 0
        assert sf.grad is not None
        assert torch.isfinite(sf.grad).all()

    def test_quadrupole_matches_per_system_loop(self):
        """Batched l_max=2 PME (energy + forces + stress + ∂E/∂Q) matches
        B independent single-system computations bit-for-bit."""
        if not torch.cuda.is_available():
            pytest.skip("multipole PME composites are GPU-only at this stage")
        td = torch.device("cuda:0")
        L = 7.0
        sigma, alpha, mesh_dim = 0.6, 0.45, 20
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c

        rng = np.random.default_rng(0xF00D)
        systems = []  # (p, q, d) for the shared CSR builder
        quads = []
        for _ in range(3):
            size = int(rng.integers(8, 12))
            p = rng.uniform(0.5, L - 0.5, (size, 3))
            q = rng.uniform(-1, 1, size)
            q -= q.mean()
            d = rng.standard_normal((size, 3)) * 0.3
            Qr = rng.standard_normal((size, 3, 3)) * 0.1
            systems.append((p, q, d))
            # Detrace so pack_multipole_moments round-trips without warning.
            Qs = 0.5 * (Qr + Qr.transpose(0, 2, 1))
            Qs -= (np.trace(Qs, axis1=1, axis2=2) / 3.0)[:, None, None] * np.eye(3)
            quads.append(Qs)

        def run_single(p, q, d, Qsym):
            P = torch.from_numpy(p).to(td, torch.float64).requires_grad_(True)
            cell = (torch.eye(3, dtype=torch.float64, device=td) * L).requires_grad_(
                True
            )
            Qt = torch.from_numpy(Qsym).to(td, torch.float64).requires_grad_(True)
            # Pack charges + dipoles + (traceless) Q into one ``multipole_moments``
            # leaf; grad flows back through the differentiable converter to Qt.
            mm = pack_multipole_moments(
                torch.from_numpy(q).to(td, torch.float64),
                torch.from_numpy(d).to(td, torch.float64),
                Qt,
            )
            idx_j, nptr, sh = _o_n2_csr_neighbors(p, L, cutoff)
            e = multipole_particle_mesh_ewald(
                P,
                mm,
                cell,
                torch.from_numpy(idx_j).to(td),
                torch.from_numpy(nptr).to(td),
                torch.from_numpy(sh).to(td),
                sigma=sigma,
                alpha=alpha,
                mesh_dimensions=(mesh_dim,) * 3,
            )
            gp, gc, gQ = torch.autograd.grad(e.sum(), [P, cell, Qt])
            return float(e.sum().detach()), gp, gc, gQ

        singles = [run_single(p, q, d, Q) for (p, q, d), Q in zip(systems, quads)]

        positions = (
            torch.from_numpy(np.concatenate([s[0] for s in systems]))
            .to(td, torch.float64)
            .requires_grad_(True)
        )
        Qt = (
            torch.from_numpy(np.concatenate(quads))
            .to(td, torch.float64)
            .requires_grad_(True)
        )
        mm = pack_multipole_moments(
            torch.from_numpy(np.concatenate([s[1] for s in systems])).to(
                td, torch.float64
            ),
            torch.from_numpy(np.concatenate([s[2] for s in systems])).to(
                td, torch.float64
            ),
            Qt,
        )
        batch_idx = torch.cat(
            [
                torch.full((s[0].shape[0],), b, dtype=torch.int32, device=td)
                for b, s in enumerate(systems)
            ]
        )
        cells = torch.stack(
            [torch.eye(3, dtype=torch.float64, device=td) * L for _ in systems]
        ).requires_grad_(True)
        idx_j_b, nptr_b, sh_b = _build_batched_csr(systems, L, cutoff)

        e_b = multipole_particle_mesh_ewald(
            positions,
            mm,
            cells,
            torch.from_numpy(idx_j_b).to(td),
            torch.from_numpy(nptr_b).to(td),
            torch.from_numpy(sh_b).to(td),
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=(mesh_dim,) * 3,
            batch_idx=batch_idx,
        )
        B = len(systems)
        assert e_b.shape == (positions.shape[0],)
        e_b_per_sys = torch.zeros(B, dtype=torch.float64, device=td).scatter_add(
            0, batch_idx.long(), e_b
        )
        gp_b, gc_b, gQ_b = torch.autograd.grad(e_b.sum(), [positions, cells, Qt])

        off = 0
        for b, (e_s, gp_s, gc_s, gQ_s) in enumerate(singles):
            n = systems[b][0].shape[0]
            torch.testing.assert_close(
                e_b_per_sys[b],
                torch.tensor(e_s, dtype=torch.float64, device=td),
                rtol=1e-9,
                atol=1e-9,
            )
            torch.testing.assert_close(gp_b[off : off + n], gp_s, rtol=1e-8, atol=1e-8)
            torch.testing.assert_close(gc_b[b], gc_s, rtol=1e-7, atol=1e-7)
            torch.testing.assert_close(gQ_b[off : off + n], gQ_s, rtol=1e-8, atol=1e-8)
            off += n

    @pytest.mark.parametrize("use_q", [False, True])
    def test_force_loss_pos_hvp_matches_per_system(self, use_q):
        """Batched PME create_graph (force-loss position-HVP) matches B
        independent single-system HVPs, l_max=1 (``use_q=False``) and l_max=2."""
        if not torch.cuda.is_available():
            pytest.skip("multipole PME composites are GPU-only at this stage")
        td = torch.device("cuda:0")
        L = 7.0
        sigma, alpha, mesh_dim = 0.6, 0.45, 20
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c
        rng = np.random.default_rng(0xBEEF)
        systems, quads = [], []
        for _ in range(3):
            size = int(rng.integers(8, 12))
            p = rng.uniform(0.5, L - 0.5, (size, 3))
            q = rng.uniform(-1, 1, size)
            q -= q.mean()
            d = rng.standard_normal((size, 3)) * 0.3
            Qr = rng.standard_normal((size, 3, 3)) * 0.1
            systems.append((p, q, d))
            # Detrace for exact pack round-trip.
            Qs = 0.5 * (Qr + Qr.transpose(0, 2, 1))
            Qs -= (np.trace(Qs, axis1=1, axis2=2) / 3.0)[:, None, None] * np.eye(3)
            quads.append(Qs)
        v_all = rng.standard_normal((sum(s[0].shape[0] for s in systems), 3))

        def _pack(q, d, Qn):
            q_t = torch.from_numpy(q).to(td, torch.float64)
            d_t = torch.from_numpy(d).to(td, torch.float64)
            Q_t = torch.from_numpy(Qn).to(td, torch.float64) if use_q else None
            return pack_multipole_moments(q_t, d_t, Q_t)

        def hvp_single(p, q, d, Qn, v):
            P = torch.from_numpy(p).to(td, torch.float64).requires_grad_(True)
            cell = torch.eye(3, dtype=torch.float64, device=td) * L
            mm = _pack(q, d, Qn)
            idx_j, nptr, sh = _o_n2_csr_neighbors(p, L, cutoff)
            E = multipole_particle_mesh_ewald(
                P,
                mm,
                cell,
                torch.from_numpy(idx_j).to(td),
                torch.from_numpy(nptr).to(td),
                torch.from_numpy(sh).to(td),
                sigma=sigma,
                alpha=alpha,
                mesh_dimensions=(mesh_dim,) * 3,
            )
            g = torch.autograd.grad(E.sum(), P, create_graph=True)[0]
            vt = torch.from_numpy(v).to(td, torch.float64)
            return torch.autograd.grad((g * vt).sum(), [P])[0].detach()

        off = 0
        hvp_singles = []
        for (p, q, d), Qn in zip(systems, quads):
            n = p.shape[0]
            hvp_singles.append(hvp_single(p, q, d, Qn, v_all[off : off + n]))
            off += n

        positions = (
            torch.from_numpy(np.concatenate([s[0] for s in systems]))
            .to(td, torch.float64)
            .requires_grad_(True)
        )
        mm = _pack(
            np.concatenate([s[1] for s in systems]),
            np.concatenate([s[2] for s in systems]),
            np.concatenate(quads),
        )
        batch_idx = torch.cat(
            [
                torch.full((s[0].shape[0],), b, dtype=torch.int32, device=td)
                for b, s in enumerate(systems)
            ]
        )
        cells = torch.stack(
            [torch.eye(3, dtype=torch.float64, device=td) * L for _ in systems]
        )
        idx_j_b, nptr_b, sh_b = _build_batched_csr(systems, L, cutoff)
        E_b = multipole_particle_mesh_ewald(
            positions,
            mm,
            cells,
            torch.from_numpy(idx_j_b).to(td),
            torch.from_numpy(nptr_b).to(td),
            torch.from_numpy(sh_b).to(td),
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=(mesh_dim,) * 3,
            batch_idx=batch_idx,
        )
        g_b = torch.autograd.grad(E_b.sum(), positions, create_graph=True)[0]
        v_t = torch.from_numpy(v_all).to(td, torch.float64)
        hvp_b = torch.autograd.grad((g_b * v_t).sum(), [positions])[0].detach()

        off = 0
        for b, hvp_s in enumerate(hvp_singles):
            n = systems[b][0].shape[0]
            torch.testing.assert_close(
                hvp_b[off : off + n], hvp_s, rtol=1e-7, atol=1e-7
            )
            off += n


class TestSpreadDoubleBackward:
    """The unified spread is multilinear, so its double-backward reuses the
    forward + backward ops with effective moments (l_max<=1). Validates
    create_graph=True force-loss HVPs vs finite differences."""

    @staticmethod
    def _energy(pos, q, mu, Qz, cit, dims, order):
        nx, ny, nz = dims
        rho = torch.ops.nvalchemiops.multipole_pme_spread_unified(
            pos, q, mu, Qz, cit, nx, ny, nz, order, 1
        )
        # Arbitrary fixed quadratic mesh functional (stands in for ρ·φ(ρ)).
        w = torch.cos(
            torch.arange(rho.numel(), dtype=torch.float64, device=rho.device)
        ).reshape(rho.shape)
        return 0.5 * (w * rho * rho).sum()

    def test_pos_and_moment_hvp_match_fd(self):
        N, L, order = 5, 6.0, 4
        dims = (16, 16, 16)
        rng = np.random.default_rng(0)
        cell = np.eye(3) * L
        cit = torch.tensor(np.linalg.inv(cell).T.reshape(1, 3, 3))
        pos0 = rng.uniform(0, L, (N, 3))
        q0 = rng.normal(size=N)
        q0 -= q0.mean()
        mu0 = rng.normal(size=(N, 3))
        Qz = torch.zeros((N, 3, 3), dtype=torch.float64)
        v = torch.tensor(rng.normal(size=(N, 3)))

        def forces(pos, q, mu, create):
            p = pos.clone().requires_grad_(True)
            E = self._energy(p, q, mu, Qz, cit, dims, order)
            return torch.autograd.grad(E, [p], create_graph=create)[0]

        h = 1e-6
        q = torch.tensor(q0)
        mu = torch.tensor(mu0)

        # pos-HVP
        p = torch.tensor(pos0, requires_grad=True)
        E = self._energy(p, q, mu, Qz, cit, dims, order)
        g = torch.autograd.grad(E, [p], create_graph=True)[0]
        hvp = torch.autograd.grad((g * v).sum(), [p])[0]
        fd = np.zeros((N, 3))
        for i in range(N):
            for d in range(3):
                pp = torch.tensor(pos0.copy())
                pp[i, d] += h
                pm = torch.tensor(pos0.copy())
                pm[i, d] -= h
                fd[i, d] = float(
                    ((forces(pp, q, mu, False) - forces(pm, q, mu, False)) * v).sum()
                ) / (2 * h)
        rel = np.abs(hvp.detach().numpy() - fd).max() / (np.abs(fd).max() + 1e-30)
        assert rel < 1e-4, f"pos-HVP rel={rel:.3e}"

        # mixed d(F·v)/dμ
        p = torch.tensor(pos0, requires_grad=True)
        mureq = torch.tensor(mu0, requires_grad=True)
        E = self._energy(p, q, mureq, Qz, cit, dims, order)
        g = torch.autograd.grad(E, [p], create_graph=True)[0]
        mxm = torch.autograd.grad((g * v).sum(), [mureq])[0]
        fdm = np.zeros((N, 3))
        for i in range(N):
            for d in range(3):
                mp = torch.tensor(mu0.copy())
                mp[i, d] += h
                mm = torch.tensor(mu0.copy())
                mm[i, d] -= h
                fdm[i, d] = float(
                    (
                        (
                            forces(torch.tensor(pos0), q, mp, False)
                            - forces(torch.tensor(pos0), q, mm, False)
                        )
                        * v
                    ).sum()
                ) / (2 * h)
        relm = np.abs(mxm.detach().numpy() - fdm).max() / (np.abs(fdm).max() + 1e-30)
        assert relm < 1e-4, f"mixed d(F·v)/dμ rel={relm:.3e}"

    def test_quadrupole_pos_q_hvp_match_fd(self):
        """l_max=2 spread double-back: pos-HVP + d(F·v)/dQ via the ∇³/∇⁴
        octupole kernels match finite differences. Q uses SYMMETRIC FD
        perturbation (kernel emits the symmetric free-index ∂/∂Q)."""
        N, L, order = 4, 6.0, 5
        nx = ny = nz = 20
        rng = np.random.default_rng(2)
        cell = np.eye(3) * L
        cit = torch.tensor(np.linalg.inv(cell).T.reshape(1, 3, 3))
        pos0 = rng.uniform(0, L, (N, 3))
        q = torch.tensor(rng.normal(size=N))
        mu = torch.tensor(rng.normal(size=(N, 3)))
        Qr = rng.normal(size=(N, 3, 3))
        Q0 = 0.5 * (Qr + Qr.transpose(0, 2, 1))
        v = torch.tensor(rng.normal(size=(N, 3)))
        w = torch.cos(torch.arange(nx * ny * nz, dtype=torch.float64)).reshape(
            nx, ny, nz
        )

        def energy(pos, Q):
            rho = torch.ops.nvalchemiops.multipole_pme_spread_unified(
                pos, q, mu, Q, cit, nx, ny, nz, order, 2
            )
            return 0.5 * (w * rho * rho).sum()

        def forces(pos, Q):
            p = pos.clone().requires_grad_(True)
            return torch.autograd.grad(energy(p, Q), [p], create_graph=False)[0]

        h = 1e-6
        Qt = torch.tensor(Q0)

        # pos-HVP
        p = torch.tensor(pos0, requires_grad=True)
        g = torch.autograd.grad(energy(p, Qt), [p], create_graph=True)[0]
        hvp = torch.autograd.grad((g * v).sum(), [p])[0]
        fd = np.zeros((N, 3))
        for i in range(N):
            for d in range(3):
                pp = torch.tensor(pos0.copy())
                pp[i, d] += h
                pm = torch.tensor(pos0.copy())
                pm[i, d] -= h
                fd[i, d] = float(((forces(pp, Qt) - forces(pm, Qt)) * v).sum()) / (
                    2 * h
                )
        rel = np.abs(hvp.detach().numpy() - fd).max() / (np.abs(fd).max() + 1e-30)
        assert rel < 1e-4, f"l2 pos-HVP rel={rel:.3e}"

        # d(F·v)/dQ — symmetric perturbation oracle.
        p = torch.tensor(pos0, requires_grad=True)
        Qreq = torch.tensor(Q0, requires_grad=True)
        g = torch.autograd.grad(energy(p, Qreq), [p], create_graph=True)[0]
        mxQ = torch.autograd.grad((g * v).sum(), [Qreq])[0]
        assert (mxQ - mxQ.transpose(-1, -2)).abs().max() < 1e-10  # symmetric
        fdQ = np.zeros((N, 3, 3))
        for i in range(N):
            for a in range(3):
                for b in range(a, 3):
                    Qp = Q0.copy()
                    Qm = Q0.copy()
                    Qp[i, a, b] += h
                    Qm[i, a, b] -= h
                    if a != b:
                        Qp[i, b, a] += h
                        Qm[i, b, a] -= h
                    val = float(
                        (
                            (
                                forces(torch.tensor(pos0), torch.tensor(Qp))
                                - forces(torch.tensor(pos0), torch.tensor(Qm))
                            )
                            * v
                        ).sum()
                    ) / (2 * h)
                    if a == b:
                        fdQ[i, a, b] = val
                    else:
                        fdQ[i, a, b] = val / 2
                        fdQ[i, b, a] = val / 2
        relQ = np.abs(mxQ.detach().numpy() - fdQ).max() / (np.abs(fdQ).max() + 1e-30)
        assert relQ < 1e-4, f"l2 d(F·v)/dQ rel={relQ:.3e}"


class TestCompositeForceLoss:
    """End-to-end ``create_graph=True`` force-loss through the full l_max=2
    ``multipole_particle_mesh_ewald`` composite — spread + convolve
    double-back plus the real-space l=2 second-backward.

    Positions are random (NOT grid-aligned): an atom on a B-spline knot
    makes the per-position FD of the gradient unreliable (discontinuous
    3rd derivative), yielding a spurious large rel even though the
    analytic HVP is exact.
    """

    def _build(self, seed: int):
        td = _torch_device()
        N, L = 12, 8.0
        sigma, alpha = 0.8, 0.5
        mesh = (20, 20, 20)
        rng = np.random.default_rng(seed)
        pos0 = rng.uniform(0.5, L - 0.5, size=(N, 3))
        charges = rng.uniform(-1.0, 1.0, N)
        charges -= charges.mean()
        dipoles = rng.standard_normal((N, 3)) * 0.3
        Qr = rng.standard_normal((N, 3, 3)) * 0.1
        Q0 = 0.5 * (Qr + Qr.transpose(0, 2, 1))
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c
        idx_j_np, nptr_np, sh_np = _o_n2_csr_neighbors(pos0, L, cutoff)
        cell = torch.tensor(np.eye(3) * L, dtype=torch.float64, device=td)
        sf = torch.cat(
            [
                torch.tensor(charges, dtype=torch.float64, device=td).unsqueeze(-1),
                torch.tensor(dipoles[:, [1, 2, 0]], dtype=torch.float64, device=td),
            ],
            dim=1,
        )
        Q = torch.tensor(Q0, dtype=torch.float64, device=td)
        idx_j = torch.from_numpy(idx_j_np).to(td)
        nptr = torch.from_numpy(nptr_np).to(td)
        sh = torch.from_numpy(sh_np).to(td)
        return td, pos0, sf, Q, cell, idx_j, nptr, sh, sigma, alpha, mesh

    def test_quadrupole_pos_hvp_matches_fd(self):
        """``d(F·v)/dx`` (position Hessian-vector product) through the full
        l_max=2 PME composite matches central finite differences."""
        td, pos0, sf, Q, cell, idx_j, nptr, sh, sigma, alpha, mesh = self._build(0)
        N = pos0.shape[0]
        rng = np.random.default_rng(1)
        v = torch.tensor(rng.normal(size=(N, 3)), device=td)

        mm = torch.cat([sf, cartesian_quadrupole_to_e3nn(Q)], dim=-1)

        def energy(pt):
            return multipole_particle_mesh_ewald(
                pt,
                mm,
                cell,
                idx_j,
                nptr,
                sh,
                sigma=sigma,
                alpha=alpha,
                mesh_dimensions=mesh,
            )

        def grad_at(p):
            pt = torch.tensor(p, dtype=torch.float64, device=td, requires_grad=True)
            return torch.autograd.grad(energy(pt).sum(), pt)[0]

        pt = torch.tensor(pos0, dtype=torch.float64, device=td, requires_grad=True)
        g = torch.autograd.grad(energy(pt).sum(), pt, create_graph=True)[0]
        hvp = torch.autograd.grad((g * v).sum(), [pt])[0]
        h = 1e-6
        fd = np.zeros((N, 3))
        for i in range(N):
            for d in range(3):
                pp = pos0.copy()
                pp[i, d] += h
                pm = pos0.copy()
                pm[i, d] -= h
                fd[i, d] = float(((grad_at(pp) - grad_at(pm)) * v).sum()) / (2 * h)
        rel = np.abs(hvp.detach().cpu().numpy() - fd).max() / (np.abs(fd).max() + 1e-30)
        assert rel < 1e-3, f"composite l2 pos-HVP rel={rel:.3e}"

    def test_quadrupole_quadrupole_hvp_matches_fd(self):
        """``vᵀ Hᵩᵩ v`` (quadrupole Hessian directional second derivative)
        through the full l_max=2 PME composite. Uses a SYMMETRIC direction +
        a directional second difference to respect the symmetric-Q free-index
        convention (per-component FD would report a spurious rel≈0.5)."""
        td, pos0, sf, Q, cell, idx_j, nptr, sh, sigma, alpha, mesh = self._build(0)
        N = pos0.shape[0]
        rng = np.random.default_rng(3)
        vqn = rng.normal(size=(N, 3, 3))
        vq = torch.tensor(0.5 * (vqn + vqn.transpose(0, 2, 1)), device=td)
        ppos = torch.tensor(pos0, dtype=torch.float64, device=td)

        def energy(Qt):
            mm = torch.cat([sf, cartesian_quadrupole_to_e3nn(Qt)], dim=-1)
            return multipole_particle_mesh_ewald(
                ppos,
                mm,
                cell,
                idx_j,
                nptr,
                sh,
                sigma=sigma,
                alpha=alpha,
                mesh_dimensions=mesh,
            )

        Ql = Q.clone().detach().requires_grad_(True)
        gQ = torch.autograd.grad(energy(Ql).sum(), Ql, create_graph=True)[0]
        hvpQ = torch.autograd.grad((gQ * vq).sum(), [Ql])[0]
        analytic = float((hvpQ * vq).sum())
        h = 1e-5
        ep = float(energy(Q + h * vq).sum().detach())
        e0 = float(energy(Q).sum().detach())
        em = float(energy(Q - h * vq).sum().detach())
        fd = (ep - 2.0 * e0 + em) / h**2
        rel = abs(analytic - fd) / (abs(fd) + 1e-30)
        assert rel < 1e-3, (
            f"composite l2 Q-HVP rel={rel:.3e} (analytic={analytic:.6e} fd={fd:.6e})"
        )


def _bcc_quadrupole_fixture(size: int = 2, device: str = "cuda:0"):
    """BCC NaCl-like fixture with random symmetric traceless quadrupoles."""
    a = 4.14
    ijk = np.indices((size, size, size)).reshape(3, -1).T
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    sites = (ijk[:, None, :] + basis[None, :, :]) * a
    pos = sites.reshape(-1, 3).astype(np.float64)
    parity = (ijk.sum(-1)[:, None] + np.array([[0, 1]])) % 2
    q = np.where(parity == 0, 1.0, -1.0).reshape(-1).astype(np.float64)
    if abs(float(q.sum())) > 1e-12:
        q[-1] -= float(q.sum())
    rng = np.random.default_rng(31415)
    mu = rng.standard_normal((pos.shape[0], 3)).astype(np.float64) * 0.3
    Q_raw = rng.standard_normal((pos.shape[0], 3, 3)).astype(np.float64) * 0.2
    Q = 0.5 * (Q_raw + Q_raw.transpose(0, 2, 1))
    trace = Q[:, 0, 0] + Q[:, 1, 1] + Q[:, 2, 2]
    Q[:, 0, 0] -= trace / 3.0
    Q[:, 1, 1] -= trace / 3.0
    Q[:, 2, 2] -= trace / 3.0
    cell = np.eye(3, dtype=np.float64) * (size * a)
    return {
        "positions": torch.from_numpy(pos).to(device, torch.float64),
        "cell": torch.from_numpy(cell).to(device, torch.float64),
        "charges": torch.from_numpy(q).to(device, torch.float64),
        "dipoles": torch.from_numpy(mu).to(device, torch.float64),
        "quadrupoles": torch.from_numpy(Q).to(device, torch.float64),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQuadrupolePMEForwardEnergy:
    """l_max = 2 forward energy tests."""

    @pytest.fixture(scope="class")
    def fix(self):
        return _bcc_quadrupole_fixture(size=2)

    def _e_recip(self, fix, *, quadrupoles, dipoles=None):
        # An l_max=2 packed tensor must carry the (zero) dipole block.
        if quadrupoles is not None and dipoles is None:
            dipoles = torch.zeros_like(fix["dipoles"])
        mm = pack_multipole_moments(fix["charges"], dipoles, quadrupoles)
        return float(
            multipole_pme_reciprocal_space(
                fix["positions"],
                mm,
                fix["cell"],
                sigma=1.0,
                alpha=0.4632,
                mesh_dimensions=(32, 32, 32),
                spline_order=4,
            )
            .sum()
            .item()
        )

    def test_zero_quadrupole_matches_dipole_no_dipoles(self, fix):
        """quadrupoles=zeros gives bit-identical result to quadrupoles=None,
        with no dipoles."""
        N = fix["positions"].shape[0]
        Q_zero = torch.zeros((N, 3, 3), dtype=torch.float64, device="cuda:0")
        e_with_Q_zero = self._e_recip(fix, quadrupoles=Q_zero, dipoles=None)
        e_no_Q = self._e_recip(fix, quadrupoles=None, dipoles=None)
        # ``Q=0`` adds nothing (each per-atom Q_eff is zero).
        assert abs(e_with_Q_zero - e_no_Q) < 1e-9 * max(abs(e_no_Q), 1.0), (
            f"Q=0 should match Q=None: e_Q={e_with_Q_zero}, e_noQ={e_no_Q}, "
            f"diff={e_with_Q_zero - e_no_Q}"
        )

    def test_zero_quadrupole_matches_dipole_with_dipoles(self, fix):
        """quadrupoles=zeros gives bit-identical result with dipoles present."""
        N = fix["positions"].shape[0]
        Q_zero = torch.zeros((N, 3, 3), dtype=torch.float64, device="cuda:0")
        e_with_Q_zero = self._e_recip(fix, quadrupoles=Q_zero, dipoles=fix["dipoles"])
        e_no_Q = self._e_recip(fix, quadrupoles=None, dipoles=fix["dipoles"])
        assert abs(e_with_Q_zero - e_no_Q) < 1e-9 * max(abs(e_no_Q), 1.0), (
            f"Q=0 should match Q=None with dipoles: "
            f"e_Q={e_with_Q_zero}, e_noQ={e_no_Q}, "
            f"diff={e_with_Q_zero - e_no_Q}"
        )

    def test_nonzero_quadrupole_changes_energy(self, fix):
        """Non-zero Q must produce a measurable energy delta from Q=0.

        Multipole interactions partially cancel, so the delta is
        conservatively bounded from below at ``10·ULP`` of the baseline.
        """
        e_baseline = self._e_recip(fix, quadrupoles=None, dipoles=fix["dipoles"])
        e_with_Q = self._e_recip(
            fix,
            quadrupoles=fix["quadrupoles"],
            dipoles=fix["dipoles"],
        )
        delta = e_with_Q - e_baseline
        print(
            f"\n  e_baseline (lmax=1)  = {e_baseline:.4f}"
            f"\n  e_with_Q   (lmax=2)  = {e_with_Q:.4f}"
            f"\n  Q-channel delta      = {delta:.4f}"
        )
        threshold = max(10.0 * 2.22e-16 * abs(e_baseline), 1e-4)
        assert abs(delta) > threshold, (
            f"Expected non-trivial Q contribution; got delta={delta} "
            f"(threshold={threshold})"
        )


def _small_quadrupole_fixture(device: str = "cuda:0"):
    """4-atom diagonal fixture with symmetric traceless Q.

    The small box + tighter mesh keep all atoms clear of B-spline cell
    breakpoints, so FD perturbations sit in the smooth spline interior
    (needed for sub-1e-4 FD agreement on the Q channel's 3rd derivative).
    """
    dtype = torch.float64
    L = 3.0
    cell = torch.eye(3, dtype=dtype, device=device) * L
    positions = torch.tensor(
        [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [1.5, 1.5, 1.5], [2.0, 2.0, 2.0]],
        dtype=dtype,
        device=device,
    )
    charges = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=dtype, device=device)
    dipoles = torch.tensor(
        [[0.1, 0.2, 0.3], [-0.2, 0.1, -0.1], [0.05, 0.05, 0.05], [0.3, -0.2, 0.1]],
        dtype=dtype,
        device=device,
    )
    rng = np.random.default_rng(42)
    Q_raw = rng.standard_normal((4, 3, 3)).astype(np.float64) * 0.2
    Q_sym = 0.5 * (Q_raw + Q_raw.transpose(0, 2, 1))
    trace = Q_sym[:, 0, 0] + Q_sym[:, 1, 1] + Q_sym[:, 2, 2]
    Q_sym[:, 0, 0] -= trace / 3.0
    Q_sym[:, 1, 1] -= trace / 3.0
    Q_sym[:, 2, 2] -= trace / 3.0
    Q = torch.from_numpy(Q_sym).to(device, dtype)
    return {
        "positions": positions,
        "cell": cell,
        "charges": charges,
        "dipoles": dipoles,
        "quadrupoles": Q,
        "sigma": 1.0,
        "alpha": 0.4632,
        "mesh": (16, 16, 16),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQuadrupoleAutogradBackward:
    """FD-vs-autograd tests for the l_max=2 PME backward.

    Validates all four diff-input gradients (positions, charges, dipoles,
    quadrupoles) against central-difference FD. The position gradient
    exercises the full ``∂L/∂r = q·∂B + μ·∂²B + (1/2)Q:∂³B`` chain rule.
    Uses the small diagonal fixture so all spline weights sit in the
    smooth interior of their pieces.
    """

    @staticmethod
    def _energy(
        positions,
        charges,
        dipoles,
        quadrupoles,
        cell,
        *,
        sigma,
        alpha,
        mesh,
        spline_order=4,
    ):
        mm = pack_multipole_moments(charges, dipoles, quadrupoles)
        return multipole_pme_reciprocal_space(
            positions,
            mm,
            cell,
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=mesh,
            spline_order=spline_order,
        )

    def test_position_gradient_fd(self):
        """``∂E/∂r`` via autograd matches central-difference FD."""
        fix = _small_quadrupole_fixture()
        pos_leaf = fix["positions"].detach().clone().requires_grad_(True)
        e = self._energy(
            pos_leaf,
            fix["charges"],
            fix["dipoles"],
            fix["quadrupoles"],
            fix["cell"],
            sigma=fix["sigma"],
            alpha=fix["alpha"],
            mesh=fix["mesh"],
        )
        e.sum().backward()
        grad_analytical = pos_leaf.grad.detach().clone()

        atom_idx, axis_idx = 3, 1
        eps = 1e-4
        with torch.no_grad():
            pos_plus = fix["positions"].clone()
            pos_plus[atom_idx, axis_idx] += eps
            e_plus = (
                self._energy(
                    pos_plus,
                    fix["charges"],
                    fix["dipoles"],
                    fix["quadrupoles"],
                    fix["cell"],
                    sigma=fix["sigma"],
                    alpha=fix["alpha"],
                    mesh=fix["mesh"],
                )
                .sum()
                .item()
            )
            pos_minus = fix["positions"].clone()
            pos_minus[atom_idx, axis_idx] -= eps
            e_minus = (
                self._energy(
                    pos_minus,
                    fix["charges"],
                    fix["dipoles"],
                    fix["quadrupoles"],
                    fix["cell"],
                    sigma=fix["sigma"],
                    alpha=fix["alpha"],
                    mesh=fix["mesh"],
                )
                .sum()
                .item()
            )
        grad_fd = (e_plus - e_minus) / (2 * eps)
        grad_an = float(grad_analytical[atom_idx, axis_idx].item())
        rel_err = abs(grad_an - grad_fd) / max(abs(grad_fd), 1e-12)
        print(
            f"\n  ∂E/∂r[{atom_idx},{axis_idx}] (analytical) = {grad_an:.6e}"
            f"\n  ∂E/∂r[{atom_idx},{axis_idx}] (FD)         = {grad_fd:.6e}"
            f"\n  rel_err                                  = {rel_err:.3e}"
        )
        assert rel_err < 1e-4, (
            f"FD vs analytical position gradient: "
            f"analytical={grad_an}, FD={grad_fd}, rel_err={rel_err}"
        )

    def test_cell_gradient_spread_fd(self):
        """``∂(K·ρ)/∂cell_inv_t`` via autograd matches central-difference FD.

        Isolates the unified-spread backward cell-gradient path. Loss is
        ``Σ_g K(g) · ρ(g)``; gradient flows to ``cell_inv_t`` through all
        three M-paths (theta = Mr, μ_frac = Mμ, Qe = MQM^T). Positions are
        shifted off integer mesh cells so FD doesn't straddle a breakpoint.
        """
        fix = _small_quadrupole_fixture()
        nx, ny, nz = fix["mesh"]
        spline_order = 4
        L = 3.0
        M_base = (
            torch.diag(torch.full((3,), 1.0 / L, dtype=torch.float64, device="cuda:0"))
            .unsqueeze(0)
            .contiguous()
        )
        positions = fix["positions"] + torch.tensor(
            [0.13, 0.07, 0.19], dtype=torch.float64, device="cuda:0"
        )
        torch.manual_seed(0)
        K = torch.randn(*fix["mesh"], dtype=torch.float64, device="cuda:0") * 0.3

        def call(M):
            rho = torch.ops.nvalchemiops.multipole_pme_spread_unified(
                positions,
                fix["charges"],
                fix["dipoles"],
                fix["quadrupoles"],
                M,
                nx,
                ny,
                nz,
                spline_order,
                2,
            )
            return (K * rho).sum()

        M_leaf = M_base.detach().clone().requires_grad_(True)
        loss = call(M_leaf)
        loss.backward()
        grad_M_an = M_leaf.grad.detach().clone()

        eps = 1e-5
        grad_M_fd = torch.zeros_like(M_base)
        for c in range(3):
            for d in range(3):
                with torch.no_grad():
                    Mp = M_base.clone()
                    Mp[0, c, d] += eps
                    lp = call(Mp).item()
                    Mm = M_base.clone()
                    Mm[0, c, d] -= eps
                    lm = call(Mm).item()
                grad_M_fd[0, c, d] = (lp - lm) / (2 * eps)
        max_abs_err = (grad_M_an - grad_M_fd).abs().max().item()
        rel_err = max_abs_err / max(grad_M_fd.abs().max().item(), 1e-12)
        print(f"\n  max abs error = {max_abs_err:.3e}")
        print(f"  rel_err       = {rel_err:.3e}")
        assert rel_err < 1e-4, (
            f"FD vs analytical cell gradient: max_abs_err={max_abs_err}, "
            f"rel_err={rel_err}"
        )

    def test_cell_gradient_pme_chain_fd(self):
        """End-to-end ``∂E_recip/∂cell`` via autograd matches FD.

        Goes through the full PME chain: ``cell -> cell_inv_t -> spread +
        k_squared -> FFT -> convolve -> IFFT -> energy``. Cell gradient
        flows through ``cell_inv_t`` (spread), ``volume = det(cell)`` and
        ``k_squared`` (convolve backward).
        """
        fix = _small_quadrupole_fixture()
        # Shift positions off mesh-cell boundaries.
        positions = fix["positions"] + torch.tensor(
            [0.13, 0.07, 0.19], dtype=torch.float64, device="cuda:0"
        )
        L = 3.0
        cell_diag = torch.tensor([L, L, L], dtype=torch.float64, device="cuda:0")

        mm = pack_multipole_moments(fix["charges"], fix["dipoles"], fix["quadrupoles"])

        def energy(cell_d):
            cell = torch.diag(cell_d)
            return multipole_pme_reciprocal_space(
                positions,
                mm,
                cell,
                sigma=fix["sigma"],
                alpha=fix["alpha"],
                mesh_dimensions=fix["mesh"],
                spline_order=4,
            )

        cell_leaf = cell_diag.detach().clone().requires_grad_(True)
        e = energy(cell_leaf)
        e.sum().backward()
        grad_an = cell_leaf.grad.detach().clone()

        # FD on each diagonal component of the cell.
        eps = 1e-5
        grad_fd = torch.zeros_like(cell_diag)
        for i in range(3):
            with torch.no_grad():
                cp = cell_diag.clone()
                cp[i] += eps
                ep = energy(cp).sum().item()
                cm = cell_diag.clone()
                cm[i] -= eps
                em = energy(cm).sum().item()
            grad_fd[i] = (ep - em) / (2 * eps)
        max_abs_err = (grad_an - grad_fd).abs().max().item()
        rel_err = max_abs_err / max(grad_fd.abs().max().item(), 1e-12)
        print(
            f"\n  analytical ∂E/∂cell_diag = {grad_an.cpu().tolist()}"
            f"\n  FD         ∂E/∂cell_diag = {grad_fd.cpu().tolist()}"
            f"\n  rel_err                  = {rel_err:.3e}"
        )
        assert rel_err < 1e-4, (
            f"FD vs analytical cell gradient (PME chain): "
            f"analytical={grad_an}, FD={grad_fd}, rel_err={rel_err}"
        )

    def test_quadrupole_gradient_fd(self):
        """``∂E/∂Q`` via autograd matches central-difference FD.

        Perturbs Q SYMMETRICALLY (``Q[i, α, β]`` and ``Q[i, β, α]``
        together), matching the kernel's symmetric ``(1/2) Q : H``
        contraction convention.
        """
        fix = _small_quadrupole_fixture()
        atom_idx = 2
        ai, bi = 0, 1  # off-diagonal pair

        Q_leaf = fix["quadrupoles"].detach().clone().requires_grad_(True)
        e = self._energy(
            fix["positions"],
            fix["charges"],
            fix["dipoles"],
            Q_leaf,
            fix["cell"],
            sigma=fix["sigma"],
            alpha=fix["alpha"],
            mesh=fix["mesh"],
        )
        e.sum().backward()
        # Symmetric DOF gradient = grad[ai,bi] + grad[bi,ai].
        grad_an = float(
            Q_leaf.grad[atom_idx, ai, bi].item() + Q_leaf.grad[atom_idx, bi, ai].item()
        )

        eps = 1e-4
        with torch.no_grad():
            Q_plus = fix["quadrupoles"].clone()
            Q_plus[atom_idx, ai, bi] += eps
            Q_plus[atom_idx, bi, ai] += eps  # symmetric
            e_plus = (
                self._energy(
                    fix["positions"],
                    fix["charges"],
                    fix["dipoles"],
                    Q_plus,
                    fix["cell"],
                    sigma=fix["sigma"],
                    alpha=fix["alpha"],
                    mesh=fix["mesh"],
                )
                .sum()
                .item()
            )
            Q_minus = fix["quadrupoles"].clone()
            Q_minus[atom_idx, ai, bi] -= eps
            Q_minus[atom_idx, bi, ai] -= eps  # symmetric
            e_minus = (
                self._energy(
                    fix["positions"],
                    fix["charges"],
                    fix["dipoles"],
                    Q_minus,
                    fix["cell"],
                    sigma=fix["sigma"],
                    alpha=fix["alpha"],
                    mesh=fix["mesh"],
                )
                .sum()
                .item()
            )
        grad_fd = (e_plus - e_minus) / (2 * eps)
        rel_err = abs(grad_an - grad_fd) / max(abs(grad_fd), 1e-12)
        print(
            f"\n  ∂E/∂Q_sym[{atom_idx},{ai},{bi}] (analytical) = {grad_an:.6e}"
            f"\n  ∂E/∂Q_sym[{atom_idx},{ai},{bi}] (FD)         = {grad_fd:.6e}"
            f"\n  rel_err                                    = {rel_err:.3e}"
        )
        assert rel_err < 1e-4, (
            f"FD vs analytical Q gradient (symmetric DOF): "
            f"analytical={grad_an}, FD={grad_fd}, rel_err={rel_err}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQuadrupoleSelfEnergyFormula:
    """Validate the sympy-derived quadrupole self-energy formula."""

    def test_quadrupole_self_energy_coefficient(self):
        """Q self-energy = F |Q|²_F / (320 π^{3/2} σ_c^5)."""
        from nvalchemiops.torch.math import FIELD_CONSTANT

        device = "cuda:0"
        N = 4
        rng = np.random.default_rng(7)
        Q_np = rng.standard_normal((N, 3, 3)).astype(np.float64) * 0.5
        Q_np = 0.5 * (Q_np + Q_np.transpose(0, 2, 1))
        trace = Q_np[:, 0, 0] + Q_np[:, 1, 1] + Q_np[:, 2, 2]
        for k in range(3):
            Q_np[:, k, k] -= trace / 3
        Q = torch.from_numpy(Q_np).to(device, torch.float64)
        charges = torch.zeros(N, dtype=torch.float64, device=device)
        volume = torch.tensor(1000.0, dtype=torch.float64, device=device)

        alpha = 0.5
        for sigma in [0.5, 1.0, 1.5]:
            sigma_c = math.sqrt(sigma**2 + 0.25 / alpha**2)
            corr = float(
                multipole_pme_energy_corrections(
                    charges,
                    dipoles=None,
                    quadrupoles=Q,
                    sigma=sigma,
                    alpha=alpha,
                    volume=volume,
                ).item()
            )
            # l=2 self denom is 320 (angular ⟨(k̂·Q·k̂)²⟩ = (2/15)|Q|_F²).
            expected = float(
                FIELD_CONSTANT
                / (320.0 * math.pi**1.5 * sigma_c**5)
                * float((Q_np * Q_np).sum())
            )
            assert abs(corr - expected) / max(abs(expected), 1e-12) < 1e-10, (
                f"σ={sigma}: corr={corr}, expected={expected}, "
                f"rel_err={(corr - expected) / expected}"
            )


class TestFractionalizeOp:
    """``multipole_pme_fractionalize`` torch op (Tier-1 stress-loss, B-warp).

    The op maps Cartesian (positions, moments) to the cell-fractional frame
    ``p = M·r``, ``d_frac = M·μ``, ``Q_frac = M·Q·Mᵀ`` so all ``cell_inv_t``
    coupling is factored out of spread (fed with an identity cell). Full
    2nd-order autograd (forward/backward/double-backward chain) — the second
    order is the genuine cell × {position, moment} cross term stress-loss needs.
    """

    def test_gradgradcheck(self):
        device = _torch_device()
        torch.manual_seed(0)
        dt = torch.float64
        n, b = 5, 2
        pos = torch.randn(n, 3, dtype=dt, device=device, requires_grad=True)
        cells = torch.randn(b, 3, 3, dtype=dt, device=device) + 3 * torch.eye(
            3, dtype=dt, device=device
        )
        cell_inv_t = (
            torch.linalg.inv(cells.transpose(1, 2)).clone().requires_grad_(True)
        )
        dip = torch.randn(n, 3, dtype=dt, device=device, requires_grad=True)
        quad = torch.randn(n, 3, 3, dtype=dt, device=device, requires_grad=True)
        bidx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32, device=device)
        op = torch.ops.nvalchemiops.multipole_pme_fractionalize

        def fn(p, m, d, q):
            return op(p, m, d, q, bidx)

        assert torch.autograd.gradcheck(
            fn, (pos, cell_inv_t, dip, quad), atol=1e-6, rtol=1e-4
        )
        assert torch.autograd.gradgradcheck(
            fn, (pos, cell_inv_t, dip, quad), atol=1e-6, rtol=1e-4
        )


class TestPMEStressLoss:
    """Single-system PME reciprocal stress-LOSS (∂²E/∂cell∂pos) — Tier-1.

    ``create_graph=True`` through ``dE/dcell`` then backprop to positions must
    not raise, must be nonzero (silent-zero guard), and must FD-match. Covers
    the B-warp ``fractionalize`` cell path + the convolve k²/volume double-back.
    """

    @pytest.mark.parametrize("lmax", [0, 1, 2])
    def test_stress_loss_fd(self, lmax):
        device = _torch_device()
        torch.manual_seed(3 + lmax)
        dt = torch.float64
        n = 6
        mesh = (20, 20, 20)
        sigma, alpha = 1.0, 0.4

        q = torch.randn(n, dtype=dt, device=device)
        q = q - q.mean()
        mu = torch.randn(n, 3, dtype=dt, device=device) if lmax >= 1 else None
        if lmax >= 2:
            qd = torch.randn(n, 3, 3, dtype=dt, device=device)
            qd = 0.5 * (qd + qd.transpose(1, 2))
            tr = torch.diagonal(qd, dim1=1, dim2=2).mean(-1)
            qd = qd - tr[:, None, None] * torch.eye(3, dtype=dt, device=device)
        else:
            qd = None
        mm = pack_multipole_moments(q, dipoles=mu, quadrupoles=qd)
        cell0 = torch.randn(3, 3, dtype=dt, device=device) * 0.3 + 6 * torch.eye(
            3, dtype=dt, device=device
        )
        # Off-grid jitter to avoid the B-spline-knot FD artifact.
        pos0 = (torch.rand(n, 3, dtype=dt, device=device) - 0.5) * 5.0 + 0.137
        w = torch.randn(3, 3, dtype=dt, device=device)

        def stress_loss(pos, cell):
            E = multipole_pme_reciprocal_space(
                pos,
                mm,
                cell,
                sigma=sigma,
                alpha=alpha,
                mesh_dimensions=mesh,
                spline_order=4,
            )
            (stress,) = torch.autograd.grad(E.sum(), cell, create_graph=True)
            return (stress * w).sum()

        pos = pos0.clone().requires_grad_(True)
        cell = cell0.clone().requires_grad_(True)
        (gp,) = torch.autograd.grad(stress_loss(pos, cell), pos)
        assert gp.abs().max() > 1e-8, "silent zero: d(stress-loss)/dpos == 0"

        v = torch.randn(n, 3, dtype=dt, device=device)
        eps = 1e-5

        def sval(p):
            return stress_loss(
                p.clone().requires_grad_(True), cell0.clone().requires_grad_(True)
            ).item()

        fd = (sval(pos0 + eps * v) - sval(pos0 - eps * v)) / (2 * eps)
        ana = (gp * v).sum().item()
        rel = abs(ana - fd) / (abs(fd) + 1e-12)
        assert rel < 1e-5, f"lmax={lmax}: stress-loss FD rel={rel:.2e}"

    @pytest.mark.parametrize("lmax", [0, 1, 2])
    def test_stress_loss_fd_batched(self, lmax):
        device = _torch_device()
        torch.manual_seed(11 + lmax)
        dt = torch.float64
        counts = [5, 4]
        B = len(counts)
        ntot = sum(counts)
        mesh = (20, 20, 20)
        sigma, alpha = 1.0, 0.4
        batch_idx = torch.cat(
            [torch.full((c,), b, dtype=torch.int32) for b, c in enumerate(counts)]
        ).to(device)

        q = torch.randn(ntot, dtype=dt, device=device)
        for b in range(B):
            m = batch_idx == b
            q[m] = q[m] - q[m].mean()
        mu = torch.randn(ntot, 3, dtype=dt, device=device) if lmax >= 1 else None
        if lmax >= 2:
            qd = torch.randn(ntot, 3, 3, dtype=dt, device=device)
            qd = 0.5 * (qd + qd.transpose(1, 2))
            tr = torch.diagonal(qd, dim1=1, dim2=2).mean(-1)
            qd = qd - tr[:, None, None] * torch.eye(3, dtype=dt, device=device)
        else:
            qd = None
        mm = pack_multipole_moments(q, dipoles=mu, quadrupoles=qd)
        cell0 = torch.randn(B, 3, 3, dtype=dt, device=device) * 0.3 + 6 * torch.eye(
            3, dtype=dt, device=device
        )
        pos0 = (torch.rand(ntot, 3, dtype=dt, device=device) - 0.5) * 5.0 + 0.137
        w = torch.randn(B, 3, 3, dtype=dt, device=device)

        def stress_loss(pos, cell):
            E = multipole_pme_reciprocal_space(
                pos,
                mm,
                cell,
                sigma=sigma,
                alpha=alpha,
                mesh_dimensions=mesh,
                spline_order=4,
                batch_idx=batch_idx,
            )
            (stress,) = torch.autograd.grad(E.sum(), cell, create_graph=True)
            return (stress * w).sum()

        pos = pos0.clone().requires_grad_(True)
        cell = cell0.clone().requires_grad_(True)
        (gp,) = torch.autograd.grad(stress_loss(pos, cell), pos)
        assert gp.abs().max() > 1e-8, "batched silent zero"

        v = torch.randn(ntot, 3, dtype=dt, device=device)
        eps = 1e-5

        def sval(p):
            return stress_loss(
                p.clone().requires_grad_(True), cell0.clone().requires_grad_(True)
            ).item()

        fd = (sval(pos0 + eps * v) - sval(pos0 - eps * v)) / (2 * eps)
        ana = (gp * v).sum().item()
        rel = abs(ana - fd) / (abs(fd) + 1e-12)
        assert rel < 1e-5, f"lmax={lmax}: batched stress-loss FD rel={rel:.2e}"

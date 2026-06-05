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

"""End-to-end ``pair_fn`` tests for the JAX ``naive_neighbor_list`` binding.

Mirrors ``test/neighbors/bindings/torch/test_pair_fn.py`` for the (single-system)
JAX naive path: an inline Warp ``pair_fn`` with required ``pair_params`` and
optional, auto-allocated ``pair_energies`` / ``pair_forces`` returned matrix- or
COO-shaped.

JAX↔Torch parity is *guaranteed by construction* — both backends launch the
identical specialized Warp kernel — so these tests verify the plumbing (arg /
return order, COO packing, dtype, gating, forward-only autograd), not kernel math.

Notes
-----
- These exercise the ``jax_kernel`` → Warp path, which requires a GPU device; the
  suite gate runs on CUDA (matching ``TestJaxNaiveAutograd`` in ``test_naive.py``).
- Like the geometry-only pair-output path, a *traced* (jit'd) ``cutoff`` is not
  supported yet (that is the #92 traceable-cutoff work); these run eagerly.
- Under JAX (functional arrays) caller-supplied energy/force buffers cannot be
  written in place, so they are always auto-allocated and returned. The *return*
  contract matches Torch; the in-place-buffer aspect does not.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

from nvalchemiops.jax.neighbors.naive import naive_neighbor_list

from .conftest import create_simple_cubic_system_jax


# Analytic pair functions: ``energy = p_i + p_j + distance``, ``force = -r_ij``.
# Chosen so the result cross-checks exactly against the kernel's own
# ``neighbor_vectors`` / ``neighbor_distances`` outputs.  Module-scope singletons:
# the kernel cache keys on the ``@wp.func`` object identity.
@wp.func
def _sum_pair_fn_f32(
    r_ij: wp.vec3f,
    distance: wp.float32,
    pair_params: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
):
    energy = pair_params[i, 0] + pair_params[j, 0] + distance
    force = -r_ij
    return energy, force


@wp.func
def _sum_pair_fn_f64(
    r_ij: wp.vec3d,
    distance: wp.float64,
    pair_params: wp.array2d(dtype=wp.float64),
    i: int,
    j: int,
):
    energy = pair_params[i, 0] + pair_params[j, 0] + distance
    force = -r_ij
    return energy, force


_PAIR_FN = {jnp.float32: _sum_pair_fn_f32, jnp.float64: _sum_pair_fn_f64}
_DTYPES = [jnp.float32, jnp.float64]


def _pair_params(n_atoms: int, dtype):
    """Per-atom parameter table (num_params == 1); distinct per atom."""
    return ((jnp.arange(n_atoms, dtype=dtype) + 1.0) * 0.5).reshape(n_atoms, 1)


def _check_pair_matrix(nm, nn, nv, nd, pe, pf, pp):
    """Verify matrix-layout pe/pf match ``_sum_pair_fn`` on filled slots."""
    nm, nn, nv, nd, pe, pf, pp = (np.asarray(x) for x in (nm, nn, nv, nd, pe, pf, pp))
    checked = 0
    n_atoms = nm.shape[0]
    for i in range(n_atoms):
        for slot in range(int(nn[i])):
            j = int(nm[i, slot])
            assert 0 <= j < n_atoms
            expected_energy = float(pp[i, 0]) + float(pp[j, 0]) + float(nd[i, slot])
            assert pe[i, slot] == pytest.approx(expected_energy, rel=1e-5, abs=1e-5)
            assert np.allclose(pf[i, slot], -nv[i, slot], rtol=1e-5, atol=1e-5)
            checked += 1
    assert checked > 0, "no neighbor pairs were found; test exercised nothing"
    return checked


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_naive_pair_fn_matrix_optional_buffers_and_returned(dtype):
    """JAX naive with ``pair_fn`` (PBC, matrix): energy/force buffers are optional
    (auto-allocated) and returned, matching ``_sum_pair_fn``."""
    positions, cell, pbc = create_simple_cubic_system_jax(8, 2.0, dtype=dtype)
    pp = _pair_params(8, dtype)
    max_neighbors = 16
    # Pass neither pair_energies nor pair_forces: they are auto-allocated.
    nm, nn, _shifts, nd, nv, pe, pf = naive_neighbor_list(
        positions,
        1.1,
        cell=cell,
        pbc=pbc,
        max_neighbors=max_neighbors,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    assert pe.shape == (8, max_neighbors)
    assert pf.shape == (8, max_neighbors, 3)
    assert pe.dtype == dtype and pf.dtype == dtype
    _check_pair_matrix(nm, nn, nv, nd, pe, pf, pp)


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_naive_pair_fn_no_pbc_matrix(dtype):
    """JAX naive ``pair_fn`` on the no-PBC kernel (matrix)."""
    positions, _cell, _pbc = create_simple_cubic_system_jax(8, 2.0, dtype=dtype)
    pp = _pair_params(8, dtype)
    nm, nn, nd, nv, pe, pf = naive_neighbor_list(
        positions,
        1.1,
        max_neighbors=16,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    _check_pair_matrix(nm, nn, nv, nd, pe, pf, pp)


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_naive_pair_fn_coo_outputs_aligned(dtype):
    """JAX naive ``pair_fn`` energies/forces are COO-packed and aligned with the
    neighbor list in COO mode."""
    positions, cell, pbc = create_simple_cubic_system_jax(8, 2.0, dtype=dtype)
    pp = _pair_params(8, dtype)
    nl, _nptr, _nl_shifts, d_coo, v_coo, pe_coo, pf_coo = naive_neighbor_list(
        positions,
        1.1,
        cell=cell,
        pbc=pbc,
        max_neighbors=16,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    num_pairs = nl.shape[1]
    assert pe_coo.shape == (num_pairs,)
    assert pf_coo.shape == (num_pairs, 3)
    i_idx, j_idx = np.asarray(nl[0]), np.asarray(nl[1])
    pp_np, d_np = np.asarray(pp), np.asarray(d_coo)
    expected_e = pp_np[i_idx, 0] + pp_np[j_idx, 0] + d_np
    assert np.allclose(np.asarray(pe_coo), expected_e, rtol=1e-5, atol=1e-5)
    assert np.allclose(np.asarray(pf_coo), -np.asarray(v_coo), rtol=1e-5, atol=1e-5)


def test_naive_pair_fn_tail_without_geometry():
    """``pair_fn`` set but no ``return_distances`` / ``return_vectors``: the return
    tail is exactly ``(..., pe, pf)`` with no distances/vectors."""
    positions, cell, pbc = create_simple_cubic_system_jax(8, 2.0, dtype=jnp.float32)
    pp = _pair_params(8, jnp.float32)
    out = naive_neighbor_list(
        positions,
        1.1,
        cell=cell,
        pbc=pbc,
        max_neighbors=16,
        pair_fn=_sum_pair_fn_f32,
        pair_params=pp,
    )
    # base = (nm, nn, shifts); tail = (pe, pf)  ->  5 elements.
    assert len(out) == 5
    nm, nn, _shifts, pe, pf = out
    assert pe.shape == (8, 16)
    assert pf.shape == (8, 16, 3)


def test_naive_pair_fn_forward_only_grad():
    """``distances`` stay differentiable w.r.t. positions; ``pe`` / ``pf`` are
    forward-only (zero gradient)."""
    positions, cell, pbc = create_simple_cubic_system_jax(8, 2.0, dtype=jnp.float64)
    # Perturb so the system isn't a degenerate lattice for the gradient.
    positions = positions + 0.05 * jax.random.normal(
        jax.random.key(0), positions.shape, dtype=jnp.float64
    )
    pp = _pair_params(8, jnp.float64)

    def call(p):
        return naive_neighbor_list(
            p,
            1.3,
            cell=cell,
            pbc=pbc,
            max_neighbors=16,
            return_distances=True,
            return_vectors=True,
            pair_fn=_sum_pair_fn_f64,
            pair_params=pp,
        )

    # nm, nn, shifts, distances, vectors, pe, pf
    g_dist = jax.grad(lambda p: call(p)[3].sum())(positions)
    assert jnp.isfinite(g_dist).all().item()
    assert float(jnp.abs(g_dist).max()) > 0.0

    g_pe = jax.grad(lambda p: call(p)[5].sum())(positions)
    assert np.allclose(np.asarray(g_pe), 0.0)
    g_pf = jax.grad(lambda p: call(p)[6].sum())(positions)
    assert np.allclose(np.asarray(g_pf), 0.0)


def test_naive_pair_fn_requires_pair_params():
    """``pair_fn`` set without ``pair_params`` raises a clear ``ValueError``."""
    positions, cell, pbc = create_simple_cubic_system_jax(8, 2.0, dtype=jnp.float32)
    with pytest.raises(ValueError, match="pair_fn requires pair_params"):
        naive_neighbor_list(
            positions,
            1.1,
            cell=cell,
            pbc=pbc,
            max_neighbors=16,
            pair_fn=_sum_pair_fn_f32,
        )


def test_naive_pair_fn_multiple_images_r_gt_1():
    """Adversarial: cutoff larger than the cell triggers R>1 periodic images; the
    ``pair_fn`` evaluation must still match the stored geometry, and every image
    must be enumerated (regression guard for the shift-axis launch-dim bug — a
    pinned shift axis would silently drop all non-zero images)."""
    positions, cell, pbc = create_simple_cubic_system_jax(8, 2.0, dtype=jnp.float64)
    pp = _pair_params(8, jnp.float64)
    # cutoff 3.0 > cell_size 2.0  ->  shift range R == 2.
    nm, nn, _shifts, nd, nv, pe, pf = naive_neighbor_list(
        positions,
        3.0,
        cell=cell,
        pbc=pbc,
        max_neighbors=128,
        return_distances=True,
        return_vectors=True,
        pair_fn=_sum_pair_fn_f64,
        pair_params=pp,
    )
    _check_pair_matrix(nm, nn, nv, nd, pe, pf, pp)
    # Multi-image: each atom sees far more than the R==1 lattice's 6 neighbors.
    # If the shift axis were pinned to 1, only the zero-shift (<=6) would survive.
    assert int(np.asarray(nn).min()) > 6


def test_naive_pair_fn_mixed_pbc():
    """Adversarial: periodic in x/z, open in y."""
    positions, cell, _pbc = create_simple_cubic_system_jax(8, 2.0, dtype=jnp.float64)
    pbc = jnp.array([[True, False, True]])
    pp = _pair_params(8, jnp.float64)
    nm, nn, _shifts, nd, nv, pe, pf = naive_neighbor_list(
        positions,
        1.1,
        cell=cell,
        pbc=pbc,
        max_neighbors=16,
        return_distances=True,
        return_vectors=True,
        pair_fn=_sum_pair_fn_f64,
        pair_params=pp,
    )
    _check_pair_matrix(nm, nn, nv, nd, pe, pf, pp)


def test_naive_pair_fn_empty_single_atom():
    """Adversarial: a single atom has no neighbors; pe/pf are returned as zeros."""
    positions = jnp.zeros((1, 3), dtype=jnp.float32)
    pp = jnp.ones((1, 1), dtype=jnp.float32)
    nm, nn, nd, nv, pe, pf = naive_neighbor_list(
        positions,
        1.1,
        max_neighbors=4,
        return_distances=True,
        return_vectors=True,
        pair_fn=_sum_pair_fn_f32,
        pair_params=pp,
    )
    assert int(np.asarray(nn)[0]) == 0
    assert np.allclose(np.asarray(pe), 0.0)
    assert np.allclose(np.asarray(pf), 0.0)


def _torch_naive_or_skip():
    import torch

    from nvalchemiops.torch.neighbors.naive import naive_neighbor_list as nl_torch

    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is required for this cross-backend check")
    return torch, nl_torch


# Cross-backend tests use a small cluster with cutoff well below half the cell
# width, so there are no multi-image neighbors: the neighbor *set* is identical
# across backends.  (In the multi-image regime — cutoff > half-cell — the two
# bindings legitimately differ in which/how many periodic images they enumerate;
# that is a pre-existing characteristic unrelated to ``pair_fn``, so the
# self-consistency tests above never assert cross-backend equality there.)
def _cross_backend_system(dtype):
    rng = np.random.default_rng(1)
    pos_np = rng.normal(0.0, 0.6, size=(8, 3))
    pp_np = ((np.arange(8) + 1.0) * 0.5).reshape(8, 1)
    return pos_np, pp_np


@pytest.mark.slow
@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_naive_pair_fn_matches_torch_no_pbc(dtype):
    """Cross-backend (no PBC): JAX and Torch naive launch the identical specialized
    Warp kernel, so the neighbor matrix and per-pair energies/forces match exactly."""
    torch, nl_torch = _torch_naive_or_skip()
    torch_dtype = torch.float32 if dtype == jnp.float32 else torch.float64
    pos_np, pp_np = _cross_backend_system(dtype)
    pair_fn = _PAIR_FN[dtype]

    nm_j, nn_j, _nd_j, _nv_j, pe_j, pf_j = naive_neighbor_list(
        jnp.asarray(pos_np, dtype=dtype),
        1.5,
        max_neighbors=16,
        return_distances=True,
        return_vectors=True,
        pair_fn=pair_fn,
        pair_params=jnp.asarray(pp_np, dtype=dtype),
    )
    nm_t, _nn_t, _nd_t, _nv_t, pe_t, pf_t = nl_torch(
        torch.tensor(pos_np, dtype=torch_dtype, device="cuda"),
        1.5,
        max_neighbors=16,
        return_distances=True,
        return_vectors=True,
        pair_fn=pair_fn,
        pair_params=torch.tensor(pp_np, dtype=torch_dtype, device="cuda"),
    )
    assert int(np.asarray(nn_j).sum()) > 0
    assert np.array_equal(np.asarray(nm_j), nm_t.cpu().numpy())
    assert np.allclose(
        np.asarray(pe_j), pe_t.detach().cpu().numpy(), rtol=1e-6, atol=1e-6
    )
    assert np.allclose(
        np.asarray(pf_j), pf_t.detach().cpu().numpy(), rtol=1e-6, atol=1e-6
    )


@pytest.mark.slow
@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_naive_pair_fn_matches_torch_pbc(dtype):
    """Cross-backend (PBC, big box → no multi-image): the COO pair outputs agree
    once aligned by ``(i, j)``.  The two bindings enumerate PBC neighbors in a
    different per-row order (pre-existing), so compare order-independently."""
    torch, nl_torch = _torch_naive_or_skip()
    torch_dtype = torch.float32 if dtype == jnp.float32 else torch.float64
    pos_np, pp_np = _cross_backend_system(dtype)
    cell_np = np.eye(3) * 10.0
    pair_fn = _PAIR_FN[dtype]
    n = 8

    nl_j, _nptr_j, _sh_j, _d_j, _v_j, pe_j, pf_j = naive_neighbor_list(
        jnp.asarray(pos_np, dtype=dtype),
        1.5,
        cell=jnp.asarray(cell_np, dtype=dtype).reshape(1, 3, 3),
        pbc=jnp.array([[True, True, True]]),
        max_neighbors=16,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=pair_fn,
        pair_params=jnp.asarray(pp_np, dtype=dtype),
    )
    nl_t, _nptr_t, _sh_t, _d_t, _v_t, pe_t, pf_t = nl_torch(
        torch.tensor(pos_np, dtype=torch_dtype, device="cuda"),
        1.5,
        torch.tensor(cell_np, dtype=torch_dtype, device="cuda"),
        torch.tensor([True, True, True], device="cuda"),
        max_neighbors=16,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=pair_fn,
        pair_params=torch.tensor(pp_np, dtype=torch_dtype, device="cuda"),
    )

    nl_j, pe_j, pf_j = np.asarray(nl_j), np.asarray(pe_j), np.asarray(pf_j)
    nl_t = nl_t.cpu().numpy()
    pe_t, pf_t = pe_t.detach().cpu().numpy(), pf_t.detach().cpu().numpy()
    # No multi-image -> each (i, j) pair is unique, so it is a canonical sort key.
    key_j = nl_j[0] * n + nl_j[1]
    key_t = nl_t[0] * n + nl_t[1]
    oj, ot = np.argsort(key_j, kind="stable"), np.argsort(key_t, kind="stable")
    assert len(key_j) > 0
    assert np.array_equal(key_j[oj], key_t[ot])
    assert np.allclose(pe_j[oj], pe_t[ot], rtol=1e-6, atol=1e-6)
    assert np.allclose(pf_j[oj], pf_t[ot], rtol=1e-6, atol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_naive_pair_fn_matches_torch_pbc_multi_image(dtype):
    """Cross-backend (PBC, multi-image: cutoff > half-cell → R>1). Regression guard
    for the shift-axis launch-dim bug: JAX must enumerate the *same* periodic images
    as Torch (the independent oracle). A pair ``(i, j)`` recurs across images, so the
    canonical sort key is ``(i, j, distance)``."""
    torch, nl_torch = _torch_naive_or_skip()
    torch_dtype = torch.float32 if dtype == jnp.float32 else torch.float64
    positions, cell, pbc = create_simple_cubic_system_jax(8, 2.0, dtype=dtype)
    pp = _pair_params(8, dtype)
    pair_fn = _PAIR_FN[dtype]
    pos_np = np.asarray(positions)
    cell_np = np.asarray(cell).reshape(3, 3)
    pp_np = np.asarray(pp)

    nl_j, _nptr_j, _sh_j, d_j, v_j, pe_j, pf_j = naive_neighbor_list(
        positions,
        1.1,
        cell=cell,
        pbc=pbc,
        max_neighbors=64,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=pair_fn,
        pair_params=pp,
    )
    nl_t, _nptr_t, _sh_t, d_t, v_t, pe_t, pf_t = nl_torch(
        torch.tensor(pos_np, dtype=torch_dtype, device="cuda"),
        1.1,
        torch.tensor(cell_np, dtype=torch_dtype, device="cuda"),
        torch.tensor([True, True, True], device="cuda"),
        max_neighbors=64,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=pair_fn,
        pair_params=torch.tensor(pp_np, dtype=torch_dtype, device="cuda"),
    )

    i_j, j_j, [dj, vj, pej, pfj] = _canon_coo(nl_j, [d_j, v_j, pe_j, pf_j])
    i_t, j_t, [dt, vt, pet, pft] = _canon_coo(
        nl_t.cpu().numpy(),
        [
            d_t.detach().cpu().numpy(),
            v_t.detach().cpu().numpy(),
            pe_t.detach().cpu().numpy(),
            pf_t.detach().cpu().numpy(),
        ],
    )
    # Same number of images, same (i, j) multiset, same per-pair geometry + outputs.
    assert nl_j.shape[1] == nl_t.shape[1]
    assert int(np.asarray(nl_j).shape[1]) > 8  # genuinely multi-image
    assert np.array_equal(i_j, i_t) and np.array_equal(j_j, j_t)
    assert np.allclose(dj, dt, rtol=1e-5, atol=1e-5)
    assert np.allclose(vj, vt, rtol=1e-5, atol=1e-5)
    assert np.allclose(pej, pet, rtol=1e-5, atol=1e-5)
    assert np.allclose(pfj, pft, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
def test_naive_multi_image_grad_and_hvp_matches_torch():
    """The launch-dim fix routes multi-image (R>1) neighbor slots through the autograd
    backward for the first time.  Verify the gradient AND a Hessian-vector product of a
    distances loss match Torch at fp64 on a ``cutoff > half-cell`` PBC system
    (``pair_fn`` present; geometry stays differentiable).

    ``loss = distances.sum()`` — the cotangent on ``distances`` is constant, the regime
    the reconstruction backward is exact for.  (A loss *nonlinear* in distance, e.g.
    ``d**2``, hits a separate pre-existing JAX ``custom_vjp`` higher-order limitation
    that also affects R==1 and is unrelated to this fix; see ``_md/task-6-review.md``.)
    """
    torch, nl_torch = _torch_naive_or_skip()
    rng = np.random.default_rng(0)
    base = np.array(
        [[i, j, k] for i in range(2) for j in range(2) for k in range(2)], dtype=float
    )
    pos_np = base + rng.normal(0.0, 0.12, size=(8, 3))
    cell_np = np.eye(3) * 2.0
    pp_np = ((np.arange(8) + 1.0) * 0.5).reshape(8, 1)
    v_np = rng.normal(0.0, 1.0, size=(8, 3))
    cutoff = 1.6  # > half-cell (1.0) -> R>1 multi-image

    pos_j = jnp.asarray(pos_np, dtype=jnp.float64)
    cell_j = jnp.asarray(cell_np, dtype=jnp.float64).reshape(1, 3, 3)
    pbc_j = jnp.array([[True, True, True]])
    pp_j = jnp.asarray(pp_np, dtype=jnp.float64)
    v_j = jnp.asarray(v_np, dtype=jnp.float64)

    def loss_j(p):
        out = naive_neighbor_list(
            p,
            cutoff,
            cell=cell_j,
            pbc=pbc_j,
            max_neighbors=128,
            return_distances=True,
            return_vectors=True,
            pair_fn=_sum_pair_fn_f64,
            pair_params=pp_j,
        )
        return out[3].sum()  # distances

    g_j = jax.grad(loss_j)(pos_j)
    hvp_j = jax.grad(lambda p: jnp.vdot(jax.grad(loss_j)(p), v_j))(pos_j)

    pos_t = torch.tensor(pos_np, dtype=torch.float64, device="cuda", requires_grad=True)
    cell_t = torch.tensor(cell_np, dtype=torch.float64, device="cuda")
    pbc_t = torch.tensor([True, True, True], device="cuda")
    pp_t = torch.tensor(pp_np, dtype=torch.float64, device="cuda")
    v_t = torch.tensor(v_np, dtype=torch.float64, device="cuda")

    def loss_t(p):
        out = nl_torch(
            p,
            cutoff,
            cell_t,
            pbc_t,
            max_neighbors=128,
            return_distances=True,
            return_vectors=True,
            pair_fn=_sum_pair_fn_f64,
            pair_params=pp_t,
        )
        return out[3].sum()

    g_t = torch.autograd.grad(loss_t(pos_t), pos_t, create_graph=True)[0]
    hvp_t = torch.autograd.grad((g_t * v_t).sum(), pos_t)[0]

    # Genuinely multi-image: an R==1 lattice tops out at 6 neighbors/atom.
    nl_check = naive_neighbor_list(
        pos_j,
        cutoff,
        cell=cell_j,
        pbc=pbc_j,
        max_neighbors=128,
        return_neighbor_list=True,
    )
    assert int(nl_check[0].shape[1]) > 24
    assert float(jnp.abs(g_j).max()) > 0.0
    assert np.allclose(
        np.asarray(g_j), g_t.detach().cpu().numpy(), atol=1e-9, rtol=1e-9
    )
    assert np.allclose(
        np.asarray(hvp_j), hvp_t.detach().cpu().numpy(), atol=1e-8, rtol=1e-8
    )


# ===========================================================================
# Fan-out: batch_naive / cell_list / batch_cell_list / cluster_tile /
# batch_cluster_tile (task 4).  Reuse ``_PAIR_FN`` / ``_check_pair_matrix`` /
# ``_pair_params`` and the conftest fixtures.
# ===========================================================================


def _canon_coo(nl, arrays):
    """Sort COO pairs and aligned arrays for order-independent comparison.

    ``arrays[0]`` must be the distances. Additional scalar or vector arrays are
    included as tie-breakers, which matters for duplicate periodic images that
    share ``(i, j, distance)`` but have different image vectors.
    """
    nl = np.asarray(nl)
    arrays = [np.asarray(a) for a in arrays]
    keys = [nl[0], nl[1], np.round(arrays[0], 5)]
    for array in arrays[1:]:
        rounded = np.round(array, 5)
        if rounded.ndim == 1:
            keys.append(rounded)
        else:
            keys.extend(rounded[:, dim] for dim in range(rounded.shape[1]))
    order = np.lexsort(tuple(reversed(keys)))
    return nl[0][order], nl[1][order], [a[order] for a in arrays]


def _make_batch_jax(dtype, n_per=6, box=4.0, scale=0.5):
    """Two-system batched setup, well inside the box (no multi-image)."""
    key = jax.random.key(0)
    pos = jax.random.normal(key, (2 * n_per, 3), dtype=dtype) * scale
    batch_idx = jnp.concatenate(
        [jnp.zeros(n_per, jnp.int32), jnp.ones(n_per, jnp.int32)]
    )
    batch_ptr = jnp.array([0, n_per, 2 * n_per], dtype=jnp.int32)
    cell = jnp.tile(jnp.eye(3, dtype=dtype)[None] * box, (2, 1, 1))
    pbc = jnp.ones((2, 3), dtype=jnp.bool_)
    pp = ((jnp.arange(2 * n_per, dtype=dtype) + 1.0) * 0.5).reshape(2 * n_per, 1)
    return pos, batch_idx, batch_ptr, cell, pbc, pp, n_per


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_batch_naive_pair_fn_matrix(dtype):
    """JAX batch_naive ``pair_fn`` (PBC, matrix): auto-allocated pe/pf match
    ``_sum_pair_fn`` on filled slots."""
    from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list

    pos, bidx, bptr, cell, pbc, pp, n_per = _make_batch_jax(dtype)
    nm, nn, _sh, nd, nv, pe, pf = batch_naive_neighbor_list(
        pos,
        1.5,
        batch_idx=bidx,
        batch_ptr=bptr,
        cell=cell,
        pbc=pbc,
        max_neighbors=16,
        max_atoms_per_system=n_per,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    assert pe.shape == (2 * n_per, 16)
    assert pf.shape == (2 * n_per, 16, 3)
    _check_pair_matrix(nm, nn, nv, nd, pe, pf, pp)


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_batch_naive_pair_fn_coo(dtype):
    """JAX batch_naive ``pair_fn`` COO outputs are aligned with the neighbor list."""
    from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list

    pos, bidx, bptr, cell, pbc, pp, n_per = _make_batch_jax(dtype)
    nl, _nptr, _sh, d_coo, v_coo, pe_coo, pf_coo = batch_naive_neighbor_list(
        pos,
        1.5,
        batch_idx=bidx,
        batch_ptr=bptr,
        cell=cell,
        pbc=pbc,
        max_neighbors=16,
        max_atoms_per_system=n_per,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    p = nl.shape[1]
    assert pe_coo.shape == (p,)
    assert pf_coo.shape == (p, 3)
    i, j = np.asarray(nl[0]), np.asarray(nl[1])
    pp_np = np.asarray(pp)
    assert np.allclose(
        np.asarray(pe_coo), pp_np[i, 0] + pp_np[j, 0] + np.asarray(d_coo), atol=1e-5
    )
    assert np.allclose(np.asarray(pf_coo), -np.asarray(v_coo), atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_batch_naive_pair_fn_matches_torch(dtype):
    """Cross-backend: JAX vs Torch batch_naive ``pair_fn`` (order-independent COO)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is required for this cross-backend check")
    from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list as bn_j
    from nvalchemiops.torch.neighbors.batch_naive import (
        batch_naive_neighbor_list as bn_t,
    )

    td = torch.float32 if dtype == jnp.float32 else torch.float64
    pos, bidx, bptr, cell, pbc, pp, n_per = _make_batch_jax(dtype)
    nl_j, _p, _s, d_j, _v, pe_j, pf_j = bn_j(
        pos,
        1.5,
        batch_idx=bidx,
        batch_ptr=bptr,
        cell=cell,
        pbc=pbc,
        max_neighbors=16,
        max_atoms_per_system=n_per,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    nl_t, _pt, _st, d_t, _vt, pe_t, pf_t = bn_t(
        torch.tensor(np.asarray(pos), dtype=td, device="cuda"),
        1.5,
        batch_idx=torch.tensor(np.asarray(bidx), dtype=torch.int32, device="cuda"),
        batch_ptr=torch.tensor(np.asarray(bptr), dtype=torch.int32, device="cuda"),
        cell=torch.tensor(np.asarray(cell), dtype=td, device="cuda"),
        pbc=torch.tensor(np.asarray(pbc), device="cuda"),
        max_neighbors=16,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=torch.tensor(np.asarray(pp), dtype=td, device="cuda"),
    )
    assert nl_j.shape[1] == nl_t.shape[1] > 0
    i_j, j_j, [_dj, pej, pfj] = _canon_coo(nl_j, [d_j, pe_j, pf_j])
    i_t, j_t, [_dt, pet, pft] = _canon_coo(
        nl_t.cpu().numpy(),
        [d_t.cpu().numpy(), pe_t.detach().cpu().numpy(), pf_t.detach().cpu().numpy()],
    )
    assert np.array_equal(i_j, i_t) and np.array_equal(j_j, j_t)
    assert np.allclose(pej, pet, atol=1e-5, rtol=1e-5)
    assert np.allclose(pfj, pft, atol=1e-5, rtol=1e-5)


def test_batch_naive_pair_fn_requires_pair_params():
    """``pair_fn`` set without ``pair_params`` raises a clear ``ValueError``."""
    from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list

    pos, bidx, bptr, cell, pbc, _pp, n_per = _make_batch_jax(jnp.float32)
    with pytest.raises(ValueError, match="pair_fn requires pair_params"):
        batch_naive_neighbor_list(
            pos,
            1.5,
            batch_idx=bidx,
            batch_ptr=bptr,
            cell=cell,
            pbc=pbc,
            max_neighbors=16,
            max_atoms_per_system=n_per,
            pair_fn=_sum_pair_fn_f32,
        )


# ---- cell_list -----------------------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_cell_list_pair_fn_matrix(dtype):
    """JAX cell_list ``pair_fn`` (PBC, matrix): auto-allocated pe/pf match
    ``_sum_pair_fn`` on filled slots."""
    from nvalchemiops.jax.neighbors.cell_list import cell_list

    positions, cell, pbc = create_simple_cubic_system_jax(8, 2.0, dtype=dtype)
    pp = _pair_params(8, dtype)
    nm, nn, _sh, nd, nv, pe, pf = cell_list(
        positions,
        1.1,
        cell,
        pbc,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    assert pe.shape == (8, nm.shape[1])
    assert pf.shape == (8, nm.shape[1], 3)
    _check_pair_matrix(nm, nn, nv, nd, pe, pf, pp)


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_cell_list_pair_fn_coo(dtype):
    """JAX cell_list ``pair_fn`` COO outputs are aligned with the neighbor list."""
    from nvalchemiops.jax.neighbors.cell_list import cell_list

    positions, cell, pbc = create_simple_cubic_system_jax(8, 2.0, dtype=dtype)
    pp = _pair_params(8, dtype)
    nl, _nptr, _sh, d_coo, v_coo, pe_coo, pf_coo = cell_list(
        positions,
        1.1,
        cell,
        pbc,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    p = nl.shape[1]
    assert pe_coo.shape == (p,)
    assert pf_coo.shape == (p, 3)
    i, j = np.asarray(nl[0]), np.asarray(nl[1])
    pp_np = np.asarray(pp)
    assert np.allclose(
        np.asarray(pe_coo), pp_np[i, 0] + pp_np[j, 0] + np.asarray(d_coo), atol=1e-5
    )
    assert np.allclose(np.asarray(pf_coo), -np.asarray(v_coo), atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_cell_list_pair_fn_matches_torch(dtype):
    """Cross-backend: JAX vs Torch cell_list ``pair_fn`` (order-independent COO,
    multi-image lattice)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is required for this cross-backend check")
    from nvalchemiops.jax.neighbors.cell_list import cell_list as cl_j
    from nvalchemiops.torch.neighbors.cell_list import cell_list as cl_t

    td = torch.float32 if dtype == jnp.float32 else torch.float64
    positions, cell, pbc = create_simple_cubic_system_jax(8, 2.0, dtype=dtype)
    pp = _pair_params(8, dtype)
    pos_np, cell_np, pp_np = (
        np.asarray(positions),
        np.asarray(cell).reshape(3, 3),
        np.asarray(pp),
    )
    nl_j, _p, _s, d_j, v_j, pe_j, pf_j = cl_j(
        positions,
        1.1,
        cell,
        pbc,
        max_neighbors=64,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    nl_t, _pt, _st, d_t, v_t, pe_t, pf_t = cl_t(
        torch.tensor(pos_np, dtype=td, device="cuda"),
        1.1,
        torch.tensor(cell_np, dtype=td, device="cuda"),
        torch.tensor([True, True, True], device="cuda"),
        max_neighbors=64,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=torch.tensor(pp_np, dtype=td, device="cuda"),
    )
    assert nl_j.shape[1] == nl_t.shape[1] > 0
    i_j, j_j, [_dj, vj, pej, pfj] = _canon_coo(nl_j, [d_j, v_j, pe_j, pf_j])
    i_t, j_t, [_dt, vt, pet, pft] = _canon_coo(
        nl_t.cpu().numpy(),
        [
            d_t.cpu().numpy(),
            v_t.detach().cpu().numpy(),
            pe_t.detach().cpu().numpy(),
            pf_t.detach().cpu().numpy(),
        ],
    )
    assert np.array_equal(i_j, i_t) and np.array_equal(j_j, j_t)
    assert np.allclose(vj, vt, atol=1e-5, rtol=1e-5)
    assert np.allclose(pej, pet, atol=1e-5, rtol=1e-5)
    assert np.allclose(pfj, pft, atol=1e-5, rtol=1e-5)


def _check_partial_pair_matrix(nm, nn, nv, nd, pe, pf, pp, targets):
    """Partial-aware pe/pf self-consistency: compact row ``r`` -> atom
    ``targets[r]`` (so ``pe[r,s] == pp[targets[r]] + pp[nm[r,s]] + nd[r,s]``)."""
    nm, nn, nv, nd, pe, pf, pp, tg = (
        np.asarray(x) for x in (nm, nn, nv, nd, pe, pf, pp, targets)
    )
    checked = 0
    n_atoms = pp.shape[0]
    for r in range(nm.shape[0]):
        ai = int(tg[r])
        for s in range(int(nn[r])):
            j = int(nm[r, s])
            assert 0 <= j < n_atoms
            exp = float(pp[ai, 0]) + float(pp[j, 0]) + float(nd[r, s])
            assert pe[r, s] == pytest.approx(exp, rel=1e-5, abs=1e-5)
            assert np.allclose(pf[r, s], -nv[r, s], rtol=1e-5, atol=1e-5)
            checked += 1
    assert checked > 0, "partial pair-fn test exercised no neighbor pairs"
    return checked


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_cell_list_target_indices_pair_fn_matrix(dtype):
    """JAX cell_list ``target_indices`` (partial) + ``pair_fn`` (matrix): compact
    ``num_targets`` rows; pe/pf self-consistent (row ``r`` -> atom
    ``target_indices[r]``)."""
    from nvalchemiops.jax.neighbors.cell_list import cell_list

    positions, cell, pbc = create_simple_cubic_system_jax(8, 2.0, dtype=dtype)
    n = positions.shape[0]
    pp = _pair_params(n, dtype)
    targets = jnp.arange(0, n, 2, dtype=jnp.int32)
    nt = int(targets.shape[0])
    w = 32
    nm, nn, _sh, nd, nv, pe, pf = cell_list(
        positions,
        1.1,
        cell,
        pbc,
        max_neighbors=w,
        target_indices=targets,
        fill_value=n,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    assert nm.shape == (nt, w) and pe.shape == (nt, w) and pf.shape == (nt, w, 3)
    _check_partial_pair_matrix(nm, nn, nv, nd, pe, pf, pp, targets)


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_cell_list_target_indices_pair_fn_coo(dtype):
    """JAX cell_list ``target_indices`` + ``pair_fn`` COO: source index ``nl[0]``
    is the compact row (matches torch); pe/pf align with the neighbor list."""
    from nvalchemiops.jax.neighbors.cell_list import cell_list

    positions, cell, pbc = create_simple_cubic_system_jax(8, 2.0, dtype=dtype)
    n = positions.shape[0]
    pp = _pair_params(n, dtype)
    targets = jnp.arange(0, n, 2, dtype=jnp.int32)
    nt = int(targets.shape[0])
    nl, _nptr, _sh, d_coo, v_coo, pe_coo, pf_coo = cell_list(
        positions,
        1.1,
        cell,
        pbc,
        max_neighbors=32,
        target_indices=targets,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    nl = np.asarray(nl)
    p = nl.shape[1]
    assert p > 0 and pe_coo.shape == (p,) and pf_coo.shape == (p, 3)
    assert int(nl[0].max()) < nt  # compact-row source index, not atom index
    i, j = nl[0], nl[1]
    pp_np, tg = np.asarray(pp), np.asarray(targets)
    src_atom = tg[i]  # compact row -> real atom
    assert np.allclose(
        np.asarray(pe_coo),
        pp_np[src_atom, 0] + pp_np[j, 0] + np.asarray(d_coo),
        atol=1e-5,
    )
    assert np.allclose(np.asarray(pf_coo), -np.asarray(v_coo), atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_cell_list_target_indices_pair_fn_matches_torch(dtype):
    """Cross-backend: JAX vs Torch cell_list ``target_indices`` + ``pair_fn``.

    Pins the compact-row COO contract -- both backends emit ``nl[0]`` as the
    compact row in ``[0, num_targets)`` (same target order), so an
    order-independent COO compare must agree on (i, j, distance, pe, pf)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is required for this cross-backend check")
    from nvalchemiops.jax.neighbors.cell_list import cell_list as cl_j
    from nvalchemiops.torch.neighbors.cell_list import cell_list as cl_t

    td = torch.float32 if dtype == jnp.float32 else torch.float64
    positions, cell, pbc = create_simple_cubic_system_jax(8, 2.0, dtype=dtype)
    n = positions.shape[0]
    pp = _pair_params(n, dtype)
    targets = jnp.arange(0, n, 2, dtype=jnp.int32)
    nt = int(targets.shape[0])
    w = 64
    pos_np, cell_np, pp_np = (
        np.asarray(positions),
        np.asarray(cell).reshape(3, 3),
        np.asarray(pp),
    )
    nl_j, _p, _s, d_j, _v, pe_j, pf_j = cl_j(
        positions,
        1.1,
        cell,
        pbc,
        max_neighbors=w,
        target_indices=targets,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    tgt_t = torch.tensor(np.asarray(targets), dtype=torch.int32, device="cuda")
    nt_t = int(tgt_t.numel())
    nm_t = torch.full((nt_t, w), n, dtype=torch.int32, device="cuda")
    nms_t = torch.zeros((nt_t, w, 3), dtype=torch.int32, device="cuda")
    nn_t = torch.zeros((nt_t,), dtype=torch.int32, device="cuda")
    nl_t, _pt, _st, d_t, _vt, pe_t, pf_t = cl_t(
        torch.tensor(pos_np, dtype=td, device="cuda"),
        1.1,
        torch.tensor(cell_np, dtype=td, device="cuda"),
        torch.tensor([True, True, True], device="cuda"),
        max_neighbors=w,
        fill_value=n,
        neighbor_matrix=nm_t,
        neighbor_matrix_shifts=nms_t,
        num_neighbors=nn_t,
        target_indices=tgt_t,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=torch.tensor(pp_np, dtype=td, device="cuda"),
    )
    assert nl_j.shape[1] == nl_t.shape[1] > 0
    assert int(np.asarray(nl_j)[0].max()) < nt  # JAX compact-row
    assert int(nl_t[0].max().item()) < nt_t  # Torch compact-row (same contract)
    i_j, j_j, [_dj, pej, pfj] = _canon_coo(nl_j, [d_j, pe_j, pf_j])
    i_t, j_t, [_dt, pet, pft] = _canon_coo(
        nl_t.cpu().numpy(),
        [d_t.cpu().numpy(), pe_t.detach().cpu().numpy(), pf_t.detach().cpu().numpy()],
    )
    assert np.array_equal(i_j, i_t) and np.array_equal(j_j, j_t)
    assert np.allclose(pej, pet, atol=1e-5, rtol=1e-5)
    assert np.allclose(pfj, pft, atol=1e-5, rtol=1e-5)


def _pc_safe_system(dtype, n_side=3, spacing=1.0):
    """Launch-safe pair-centric geometry: ``n_side**3`` cells in a cubic cell."""
    pos = _cubic_lattice(n_side, spacing, dtype)
    cell = (jnp.eye(3, dtype=dtype) * (n_side * spacing)).reshape(1, 3, 3)
    pbc = jnp.array([[True, True, True]])
    return pos, cell, pbc


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("use_pair_fn", [False, True], ids=["geom", "pair_fn"])
def test_cell_list_pair_centric_matches_atom_centric(dtype, use_pair_fn):
    """JAX cell_list ``strategy='pair_centric'`` pair outputs == ``atom_centric``.

    Pair-centric is a different launch strategy yielding an identical pair set;
    matrix + COO geometry (and pe/pf when ``pair_fn``) must match atom-centric
    exactly on a launch-safe geometry (so pair-centric genuinely fires, not a
    silent fallback)."""
    from nvalchemiops.jax.neighbors.cell_list import cell_list

    pos, cell, pbc = _pc_safe_system(dtype)
    n = pos.shape[0]
    pp = _pair_params(n, dtype) if use_pair_fn else None
    pf_fn = _PAIR_FN[dtype] if use_pair_fn else None
    w = 64
    kw = dict(
        max_neighbors=w,
        return_distances=True,
        return_vectors=True,
        pair_fn=pf_fn,
        pair_params=pp,
    )

    a = cell_list(pos, 1.1, cell, pbc, strategy="atom_centric", **kw)
    p = cell_list(pos, 1.1, cell, pbc, strategy="pair_centric", **kw)
    anm, ann, _as, and_, anv = a[:5]
    pnm, pnn, _ps, pnd, pnv = p[:5]

    def matrix_set(nm, nn, nd):
        nm, nn, nd = (np.asarray(x) for x in (nm, nn, nd))
        return {
            (i, int(nm[i, s]), round(float(nd[i, s]), 4))
            for i in range(nm.shape[0])
            for s in range(int(nn[i]))
        }

    aset = matrix_set(anm, ann, and_)
    assert aset == matrix_set(pnm, pnn, pnd) and len(aset) > 0
    assert np.array_equal(np.asarray(ann), np.asarray(pnn))
    if use_pair_fn:
        # pe self-consistency on the pair-centric result.
        _check_pair_matrix(pnm, pnn, pnv, pnd, p[5], p[6], pp)

    # COO parity (order-independent).
    ac = cell_list(
        pos, 1.1, cell, pbc, strategy="atom_centric", return_neighbor_list=True, **kw
    )
    pc = cell_list(
        pos, 1.1, cell, pbc, strategy="pair_centric", return_neighbor_list=True, **kw
    )
    ai, aj, _ = _canon_coo(ac[0], [ac[3]])
    pi, pj, _ = _canon_coo(pc[0], [pc[3]])
    assert np.array_equal(ai, pi) and np.array_equal(aj, pj)


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_cell_list_pair_centric_grad_matches_atom_centric(dtype):
    """Forward-only gradient through pair-centric pair outputs == atom-centric."""
    from nvalchemiops.jax.neighbors.cell_list import cell_list

    pos, cell, pbc = _pc_safe_system(dtype)
    # Break perfect-lattice symmetry so the distance-loss gradient is nonzero.
    rng = np.random.default_rng(0)
    pos = pos + jnp.asarray(rng.normal(scale=0.03, size=pos.shape), dtype=dtype)
    n = pos.shape[0]
    pp = _pair_params(n, dtype)

    def mkloss(strat):
        def loss(x):
            *_, nd, _nv, pe, _pf = cell_list(
                x,
                1.1,
                cell,
                pbc,
                max_neighbors=64,
                strategy=strat,
                return_distances=True,
                return_vectors=True,
                pair_fn=_PAIR_FN[dtype],
                pair_params=pp,
            )
            return jnp.sum(nd**2) + jnp.sum(pe)

        return loss

    ga = np.asarray(jax.grad(mkloss("atom_centric"))(pos))
    gp = np.asarray(jax.grad(mkloss("pair_centric"))(pos))
    assert np.isfinite(gp).all()
    atol = 1e-5 if dtype == jnp.float32 else 1e-9
    assert np.allclose(ga, gp, atol=atol)


def test_cell_list_pair_centric_target_indices_rejected():
    """pair_centric + target_indices raises a clear error (identical results via
    atom_centric)."""
    from nvalchemiops.jax.neighbors.cell_list import cell_list

    pos, cell, pbc = _pc_safe_system(jnp.float32)
    with pytest.raises(NotImplementedError, match="target_indices"):
        cell_list(
            pos,
            1.1,
            cell,
            pbc,
            strategy="pair_centric",
            return_distances=True,
            target_indices=jnp.array([0, 1], dtype=jnp.int32),
        )


# ---- cluster_tile (fp32-only; matrix and COO pair outputs) ---------------


def _make_cluster_tile_system(n=32, box=5.0, scale=1.0):
    key = jax.random.key(0)
    pos = jax.random.normal(key, (n, 3), dtype=jnp.float32) * scale
    cell = jnp.eye(3, dtype=jnp.float32) * box
    pp = ((jnp.arange(n, dtype=jnp.float32) + 1.0) * 0.5).reshape(n, 1)
    return pos, cell, pp


def test_cluster_tile_pair_fn_matrix():
    """JAX cluster_tile ``pair_fn`` (fp32, matrix): pe/pf match ``_sum_pair_fn`` on
    filled slots.  Analytic self-consistency is the primary oracle (parity with the
    Warp/Torch launcher is by construction — same ``_warp_query_cluster_tile``)."""
    from nvalchemiops.jax.neighbors.cluster_tile import cluster_tile_neighbor_list

    pos, cell, pp = _make_cluster_tile_system()
    nm, nn, _sh, nd, nv, pe, pf = cluster_tile_neighbor_list(
        pos,
        1.5,
        cell,
        max_neighbors=64,
        return_distances=True,
        return_vectors=True,
        pair_fn=_sum_pair_fn_f32,
        pair_params=pp,
    )
    assert pe.shape == (pos.shape[0], nm.shape[1])
    assert pf.shape == (pos.shape[0], nm.shape[1], 3)
    _check_pair_matrix(nm, nn, nv, nd, pe, pf, pp)


def test_cluster_tile_pair_fn_tail_without_geometry():
    """``pair_fn`` set with neither return_distances nor return_vectors: tail is
    exactly ``(..., pe, pf)``."""
    from nvalchemiops.jax.neighbors.cluster_tile import cluster_tile_neighbor_list

    pos, cell, pp = _make_cluster_tile_system()
    out = cluster_tile_neighbor_list(
        pos, 1.5, cell, max_neighbors=64, pair_fn=_sum_pair_fn_f32, pair_params=pp
    )
    # base = (nm, nn, shifts); tail = (pe, pf) -> 5 elements.
    assert len(out) == 5
    assert out[3].shape == (pos.shape[0], out[0].shape[1])  # pe
    assert out[4].shape == (pos.shape[0], out[0].shape[1], 3)  # pf


def test_cluster_tile_pair_fn_requires_pair_params():
    """``pair_fn`` set without ``pair_params`` raises ``ValueError``."""
    from nvalchemiops.jax.neighbors.cluster_tile import cluster_tile_neighbor_list

    pos, cell, _pp = _make_cluster_tile_system()
    with pytest.raises(ValueError, match="pair_fn requires pair_params"):
        cluster_tile_neighbor_list(
            pos,
            1.5,
            cell,
            max_neighbors=64,
            return_distances=True,
            pair_fn=_sum_pair_fn_f32,
        )


def test_cluster_tile_pair_fn_coo_matches_matrix():
    """cluster_tile COO pair outputs (eager) equal the matrix outputs packed into
    COO order: same topology, per-pair geometry, and pair_fn energies/forces.  COO
    distances/vectors are ``jnp.take`` of the (already tested) differentiable matrix
    values, so this parity also covers the eager differentiable-geometry path."""
    from nvalchemiops.jax.neighbors.cluster_tile import cluster_tile_neighbor_list

    pos, cell, pp = _make_cluster_tile_system()
    n = int(pos.shape[0])
    kwargs = dict(
        max_neighbors=64,
        return_distances=True,
        return_vectors=True,
        pair_fn=_sum_pair_fn_f32,
        pair_params=pp,
    )
    nm, nn, _sh, nd, nv, pe, pf = cluster_tile_neighbor_list(pos, 1.5, cell, **kwargs)
    nl, nptr, _shc, nd_c, nv_c, pe_c, pf_c = cluster_tile_neighbor_list(
        pos, 1.5, cell, format="coo", **kwargs
    )

    nm_np = np.asarray(nm)
    mask = nm_np != n  # active slots; default fill_value == n_atoms
    npairs = int(mask.sum())
    assert npairs > 0
    # COO arrays are the matrix arrays flattened in row-major active-slot order.
    np.testing.assert_allclose(
        np.asarray(nd_c), np.asarray(nd)[mask], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(nv_c), np.asarray(nv)[mask], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(pe_c), np.asarray(pe)[mask], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(pf_c), np.asarray(pf)[mask], rtol=1e-6, atol=1e-6
    )
    # Topology: pair count and target index agree with the matrix active order.
    assert int(np.asarray(nl).shape[1]) == npairs
    np.testing.assert_array_equal(np.asarray(nl)[1], nm_np[mask])
    assert int(np.asarray(nptr)[-1]) == npairs


def test_cluster_tile_pair_fn_coo_eager_only_under_jit():
    """cluster_tile COO pair outputs are eager-only (data-dependent pair count); the
    matrix->COO conversion raises under ``jax.jit``."""
    from nvalchemiops.jax.neighbors.cluster_tile import cluster_tile_neighbor_list

    pos, cell, pp = _make_cluster_tile_system()

    def _run(p):
        return cluster_tile_neighbor_list(
            p,
            1.5,
            cell,
            max_neighbors=64,
            format="coo",
            return_distances=True,
            pair_fn=_sum_pair_fn_f32,
            pair_params=pp,
        )

    with pytest.raises(Exception):  # noqa: B017 - concretization error under trace
        jax.jit(_run)(pos)


# ---- batch_cell_list -----------------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_batch_cell_list_pair_fn_matrix(dtype):
    """JAX batch_cell_list ``pair_fn`` (matrix): auto-allocated pe/pf match
    ``_sum_pair_fn`` on filled slots."""
    from nvalchemiops.jax.neighbors.batch_cell_list import batch_cell_list

    pos, bidx, _bptr, cell, pbc, pp, _n_per = _make_batch_jax(dtype)
    nm, nn, _sh, nd, nv, pe, pf = batch_cell_list(
        pos,
        1.5,
        cell=cell,
        pbc=pbc,
        batch_idx=bidx,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    assert pe.shape == (pos.shape[0], nm.shape[1])
    _check_pair_matrix(nm, nn, nv, nd, pe, pf, pp)


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_batch_cell_list_pair_fn_coo(dtype):
    """JAX batch_cell_list ``pair_fn`` COO outputs aligned with the neighbor list."""
    from nvalchemiops.jax.neighbors.batch_cell_list import batch_cell_list

    pos, bidx, _bptr, cell, pbc, pp, _n_per = _make_batch_jax(dtype)
    nl, _nptr, _sh, d_coo, v_coo, pe_coo, pf_coo = batch_cell_list(
        pos,
        1.5,
        cell=cell,
        pbc=pbc,
        batch_idx=bidx,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    p = nl.shape[1]
    assert pe_coo.shape == (p,)
    assert pf_coo.shape == (p, 3)
    i, j = np.asarray(nl[0]), np.asarray(nl[1])
    pp_np = np.asarray(pp)
    assert np.allclose(
        np.asarray(pe_coo), pp_np[i, 0] + pp_np[j, 0] + np.asarray(d_coo), atol=1e-5
    )
    assert np.allclose(np.asarray(pf_coo), -np.asarray(v_coo), atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_batch_cell_list_pair_fn_matches_torch(dtype):
    """Cross-backend: JAX vs Torch batch_cell_list ``pair_fn`` (order-independent COO)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is required for this cross-backend check")
    from nvalchemiops.jax.neighbors.batch_cell_list import batch_cell_list as bcl_j
    from nvalchemiops.torch.neighbors.batch_cell_list import batch_cell_list as bcl_t

    td = torch.float32 if dtype == jnp.float32 else torch.float64
    pos, bidx, _bptr, cell, pbc, pp, _n_per = _make_batch_jax(dtype)
    nl_j, _p, _s, d_j, _v, pe_j, pf_j = bcl_j(
        pos,
        1.5,
        cell=cell,
        pbc=pbc,
        batch_idx=bidx,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    nl_t, _pt, _st, d_t, _vt, pe_t, pf_t = bcl_t(
        torch.tensor(np.asarray(pos), dtype=td, device="cuda"),
        1.5,
        torch.tensor(np.asarray(cell), dtype=td, device="cuda"),
        torch.tensor(np.asarray(pbc), device="cuda"),
        torch.tensor(np.asarray(bidx), dtype=torch.int32, device="cuda"),
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=torch.tensor(np.asarray(pp), dtype=td, device="cuda"),
    )
    assert nl_j.shape[1] == nl_t.shape[1] > 0
    i_j, j_j, [_dj2, pej, pfj] = _canon_coo(nl_j, [d_j, pe_j, pf_j])
    i_t, j_t, [_dt2, pet, pft] = _canon_coo(
        nl_t.cpu().numpy(),
        [d_t.cpu().numpy(), pe_t.detach().cpu().numpy(), pf_t.detach().cpu().numpy()],
    )
    assert np.array_equal(i_j, i_t) and np.array_equal(j_j, j_t)
    assert np.allclose(pej, pet, atol=1e-5, rtol=1e-5)
    assert np.allclose(pfj, pft, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_batch_cell_list_target_indices_pair_fn_matrix(dtype):
    """JAX batch_cell_list ``target_indices`` + ``pair_fn`` (matrix): compact
    rows, pe/pf self-consistent; targets span both systems (per-target
    ``batch_idx`` lookup)."""
    from nvalchemiops.jax.neighbors.batch_cell_list import batch_cell_list

    pos, bidx, bptr, cell, pbc, pp, n_per = _make_batch_jax(dtype)
    n = pos.shape[0]
    targets = jnp.array([0, 3, n_per, n_per + 2], dtype=jnp.int32)
    nt = int(targets.shape[0])
    w = 24
    nm, nn, _sh, nd, nv, pe, pf = batch_cell_list(
        pos,
        1.5,
        cell,
        pbc,
        batch_idx=bidx,
        batch_ptr=bptr,
        max_neighbors=w,
        target_indices=targets,
        fill_value=n,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    assert nm.shape == (nt, w) and pe.shape == (nt, w) and pf.shape == (nt, w, 3)
    _check_partial_pair_matrix(nm, nn, nv, nd, pe, pf, pp, targets)


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_batch_cell_list_target_indices_pair_fn_coo(dtype):
    """JAX batch_cell_list ``target_indices`` + ``pair_fn`` COO: compact-row
    source index; pe/pf aligned with the neighbor list."""
    from nvalchemiops.jax.neighbors.batch_cell_list import batch_cell_list

    pos, bidx, bptr, cell, pbc, pp, n_per = _make_batch_jax(dtype)
    targets = jnp.array([0, 3, n_per, n_per + 2], dtype=jnp.int32)
    nt = int(targets.shape[0])
    nl, _nptr, _sh, d_coo, v_coo, pe_coo, pf_coo = batch_cell_list(
        pos,
        1.5,
        cell,
        pbc,
        batch_idx=bidx,
        batch_ptr=bptr,
        max_neighbors=24,
        target_indices=targets,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    nl = np.asarray(nl)
    p = nl.shape[1]
    assert p > 0 and pe_coo.shape == (p,) and pf_coo.shape == (p, 3)
    assert int(nl[0].max()) < nt  # compact-row source index
    i, j = nl[0], nl[1]
    pp_np, tg = np.asarray(pp), np.asarray(targets)
    src_atom = tg[i]
    assert np.allclose(
        np.asarray(pe_coo),
        pp_np[src_atom, 0] + pp_np[j, 0] + np.asarray(d_coo),
        atol=1e-5,
    )
    assert np.allclose(np.asarray(pf_coo), -np.asarray(v_coo), atol=1e-5)


def _batch_pc_safe_system(dtype, n_side=3, spacing=1.0):
    """Two identical cubic lattices -> launch-safe batched pair-centric geometry."""
    one = _cubic_lattice(n_side, spacing, dtype)
    n1 = one.shape[0]
    pos = jnp.concatenate([one, one], axis=0)
    bidx = jnp.concatenate([jnp.zeros(n1, jnp.int32), jnp.ones(n1, jnp.int32)])
    bptr = jnp.array([0, n1, 2 * n1], dtype=jnp.int32)
    box = n_side * spacing
    cell = jnp.tile((jnp.eye(3, dtype=dtype) * box)[None], (2, 1, 1))
    pbc = jnp.ones((2, 3), dtype=jnp.bool_)
    return pos, bidx, bptr, cell, pbc


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("use_pair_fn", [False, True], ids=["geom", "pair_fn"])
def test_batch_cell_list_pair_centric_matches_atom_centric(dtype, use_pair_fn):
    """JAX batch_cell_list ``strategy='pair_centric'`` pair outputs ==
    ``atom_centric`` (matrix + COO, with/without ``pair_fn``) on a launch-safe
    two-system geometry."""
    from nvalchemiops.jax.neighbors.batch_cell_list import batch_cell_list

    pos, bidx, bptr, cell, pbc = _batch_pc_safe_system(dtype)
    n = pos.shape[0]
    pp = _pair_params(n, dtype) if use_pair_fn else None
    pf_fn = _PAIR_FN[dtype] if use_pair_fn else None
    w = 64
    kw = dict(
        batch_idx=bidx,
        batch_ptr=bptr,
        max_neighbors=w,
        return_distances=True,
        return_vectors=True,
        pair_fn=pf_fn,
        pair_params=pp,
    )

    a = batch_cell_list(pos, 1.1, cell, pbc, strategy="atom_centric", **kw)
    p = batch_cell_list(pos, 1.1, cell, pbc, strategy="pair_centric", **kw)
    anm, ann, _as, and_, anv = a[:5]
    pnm, pnn, _ps, pnd, pnv = p[:5]

    def matrix_set(nm, nn, nd):
        nm, nn, nd = (np.asarray(x) for x in (nm, nn, nd))
        return {
            (i, int(nm[i, s]), round(float(nd[i, s]), 4))
            for i in range(nm.shape[0])
            for s in range(int(nn[i]))
        }

    aset = matrix_set(anm, ann, and_)
    assert aset == matrix_set(pnm, pnn, pnd) and len(aset) > 0
    assert np.array_equal(np.asarray(ann), np.asarray(pnn))
    if use_pair_fn:
        _check_pair_matrix(pnm, pnn, pnv, pnd, p[5], p[6], pp)

    ac = batch_cell_list(
        pos, 1.1, cell, pbc, strategy="atom_centric", return_neighbor_list=True, **kw
    )
    pc = batch_cell_list(
        pos, 1.1, cell, pbc, strategy="pair_centric", return_neighbor_list=True, **kw
    )
    ai, aj, _ = _canon_coo(ac[0], [ac[3]])
    pi, pj, _ = _canon_coo(pc[0], [pc[3]])
    assert np.array_equal(ai, pi) and np.array_equal(aj, pj)


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_batch_cell_list_pair_centric_grad_matches_atom_centric(dtype):
    """Forward-only gradient through batched pair-centric pair outputs ==
    atom-centric."""
    from nvalchemiops.jax.neighbors.batch_cell_list import batch_cell_list

    pos, bidx, bptr, cell, pbc = _batch_pc_safe_system(dtype)
    rng = np.random.default_rng(0)
    pos = pos + jnp.asarray(rng.normal(scale=0.03, size=pos.shape), dtype=dtype)
    n = pos.shape[0]
    pp = _pair_params(n, dtype)

    def mkloss(strat):
        def loss(x):
            *_, nd, _nv, pe, _pf = batch_cell_list(
                x,
                1.1,
                cell,
                pbc,
                batch_idx=bidx,
                batch_ptr=bptr,
                max_neighbors=64,
                strategy=strat,
                return_distances=True,
                return_vectors=True,
                pair_fn=_PAIR_FN[dtype],
                pair_params=pp,
            )
            return jnp.sum(nd**2) + jnp.sum(pe)

        return loss

    ga = np.asarray(jax.grad(mkloss("atom_centric"))(pos))
    gp = np.asarray(jax.grad(mkloss("pair_centric"))(pos))
    assert np.isfinite(gp).all()
    atol = 1e-5 if dtype == jnp.float32 else 1e-9
    assert np.allclose(ga, gp, atol=atol)


def test_batch_cell_list_pair_centric_target_indices_rejected():
    """batch pair_centric + target_indices raises a clear error."""
    from nvalchemiops.jax.neighbors.batch_cell_list import batch_cell_list

    pos, bidx, bptr, cell, pbc = _batch_pc_safe_system(jnp.float32)
    with pytest.raises(NotImplementedError, match="target_indices"):
        batch_cell_list(
            pos,
            1.1,
            cell,
            pbc,
            batch_idx=bidx,
            batch_ptr=bptr,
            strategy="pair_centric",
            return_distances=True,
            target_indices=jnp.array([0, 1], dtype=jnp.int32),
        )


def _pc_matrix_set(out):
    nm, nn, nd = (np.asarray(out[k]) for k in (0, 1, 3))
    return {
        (i, int(nm[i, s]), round(float(nd[i, s]), 4))
        for i in range(nm.shape[0])
        for s in range(int(nn[i]))
    }


@pytest.mark.parametrize("pbc_vec", [[True, False, True], [True, True, False]])
def test_cell_list_pair_centric_mixed_pbc_matches_atom_centric(pbc_vec):
    """pair_centric == atom_centric under MIXED PBC (the pair-centric kernel has
    its own non-periodic-axis offset handling -- a distinct code path)."""
    from nvalchemiops.jax.neighbors.cell_list import cell_list

    pos = _cubic_lattice(4, 1.0, jnp.float64)
    cell = (jnp.eye(3, dtype=jnp.float64) * 4.0).reshape(1, 3, 3)
    pbc = jnp.array(pbc_vec)
    n = pos.shape[0]
    pp = _pair_params(n, jnp.float64)
    kw = dict(
        max_neighbors=64,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[jnp.float64],
        pair_params=pp,
    )
    a = cell_list(pos, 1.1, cell, pbc, strategy="atom_centric", **kw)
    p = cell_list(pos, 1.1, cell, pbc, strategy="pair_centric", **kw)
    aset = _pc_matrix_set(a)
    assert aset == _pc_matrix_set(p) and len(aset) > 0
    _check_pair_matrix(p[0], p[1], p[4], p[3], p[5], p[6], pp)


def test_batch_cell_list_pair_centric_uneven_systems_mixed_pbc():
    """pair_centric == atom_centric on UNEVEN batched systems (different sizes +
    cutoff-per-cell) with mixed per-system PBC -- stresses R_max / total_cells /
    cell_to_system, which identical-system tests can't."""
    from nvalchemiops.jax.neighbors.batch_cell_list import batch_cell_list

    dt = jnp.float64
    g0 = _cubic_lattice(3, 1.0, dt)  # 27 atoms, cell 3.0
    g1 = _cubic_lattice(4, 1.2, dt)  # 64 atoms, cell 4.8
    n0, n1 = g0.shape[0], g1.shape[0]
    pos = jnp.concatenate([g0, g1], 0)
    bidx = jnp.concatenate([jnp.zeros(n0, jnp.int32), jnp.ones(n1, jnp.int32)])
    bptr = jnp.array([0, n0, n0 + n1], dtype=jnp.int32)
    cell = jnp.stack([jnp.eye(3, dtype=dt) * 3.0, jnp.eye(3, dtype=dt) * 4.8], 0)
    pbc = jnp.array([[True, True, True], [True, False, True]])
    nb = pos.shape[0]
    pp = _pair_params(nb, dt)
    kw = dict(
        batch_idx=bidx,
        batch_ptr=bptr,
        max_neighbors=128,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dt],
        pair_params=pp,
    )
    a = batch_cell_list(pos, 1.3, cell, pbc, strategy="atom_centric", **kw)
    p = batch_cell_list(pos, 1.3, cell, pbc, strategy="pair_centric", **kw)
    aset = _pc_matrix_set(a)
    assert aset == _pc_matrix_set(p) and len(aset) > 0
    assert np.array_equal(np.asarray(a[1]), np.asarray(p[1]))
    _check_pair_matrix(p[0], p[1], p[4], p[3], p[5], p[6], pp)


# ---- batch_cluster_tile (fp32-only; matrix and COO pair outputs) ----------


def _make_batch_cluster_tile_system(sys_sizes=(16, 20), cell_sizes=(5.0, 6.0)):
    rng = np.random.RandomState(0)
    chunks, cells = [], []
    for sz, length in zip(sys_sizes, cell_sizes):
        chunks.append(rng.uniform(0, length, size=(sz, 3)).astype(np.float32))
        cells.append(np.eye(3, dtype=np.float32) * length)
    positions = jnp.asarray(np.concatenate(chunks, axis=0))
    cell_batch = jnp.asarray(np.stack(cells, axis=0))
    bp = [0]
    for sz in sys_sizes:
        bp.append(bp[-1] + sz)
    batch_ptr = jnp.asarray(bp, dtype=jnp.int32)
    n = int(sum(sys_sizes))
    pp = ((jnp.arange(n, dtype=jnp.float32) + 1.0) * 0.5).reshape(n, 1)
    return positions, cell_batch, batch_ptr, pp


def test_batch_cluster_tile_pair_fn_matrix():
    """JAX batch_cluster_tile ``pair_fn`` (fp32, matrix): pe/pf match
    ``_sum_pair_fn`` on filled slots (analytic self-consistency oracle)."""
    from nvalchemiops.jax.neighbors.batch_cluster_tile import (
        batch_cluster_tile_neighbor_list,
    )

    pos, cell_batch, batch_ptr, pp = _make_batch_cluster_tile_system()
    nm, nn, _sh, nd, nv, pe, pf = batch_cluster_tile_neighbor_list(
        pos,
        1.5,
        cell_batch,
        batch_ptr,
        max_neighbors=64,
        return_distances=True,
        return_vectors=True,
        pair_fn=_sum_pair_fn_f32,
        pair_params=pp,
    )
    assert pe.shape == (pos.shape[0], nm.shape[1])
    _check_pair_matrix(nm, nn, nv, nd, pe, pf, pp)


def test_batch_cluster_tile_pair_fn_requires_pair_params():
    """``pair_fn`` set without ``pair_params`` raises ``ValueError``."""
    from nvalchemiops.jax.neighbors.batch_cluster_tile import (
        batch_cluster_tile_neighbor_list,
    )

    pos, cell_batch, batch_ptr, _pp = _make_batch_cluster_tile_system()
    with pytest.raises(ValueError, match="pair_fn requires pair_params"):
        batch_cluster_tile_neighbor_list(
            pos,
            1.5,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
            return_distances=True,
            pair_fn=_sum_pair_fn_f32,
        )


def test_batch_cluster_tile_pair_fn_coo_matches_matrix():
    """batch_cluster_tile COO pair outputs (eager) equal the matrix outputs packed
    into COO order: same topology, per-pair geometry, and pair_fn energies/forces."""
    from nvalchemiops.jax.neighbors.batch_cluster_tile import (
        batch_cluster_tile_neighbor_list,
    )

    pos, cell_batch, batch_ptr, pp = _make_batch_cluster_tile_system()
    n = int(pos.shape[0])
    kwargs = dict(
        max_neighbors=64,
        return_distances=True,
        return_vectors=True,
        pair_fn=_sum_pair_fn_f32,
        pair_params=pp,
    )
    nm, nn, _sh, nd, nv, pe, pf = batch_cluster_tile_neighbor_list(
        pos, 1.5, cell_batch, batch_ptr, **kwargs
    )
    nl, nptr, _shc, nd_c, nv_c, pe_c, pf_c = batch_cluster_tile_neighbor_list(
        pos, 1.5, cell_batch, batch_ptr, format="coo", **kwargs
    )

    nm_np = np.asarray(nm)
    mask = nm_np != n  # active slots; default fill_value == total atoms
    npairs = int(mask.sum())
    assert npairs > 0
    np.testing.assert_allclose(
        np.asarray(nd_c), np.asarray(nd)[mask], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(nv_c), np.asarray(nv)[mask], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(pe_c), np.asarray(pe)[mask], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(pf_c), np.asarray(pf)[mask], rtol=1e-6, atol=1e-6
    )
    assert int(np.asarray(nl).shape[1]) == npairs
    np.testing.assert_array_equal(np.asarray(nl)[1], nm_np[mask])
    assert int(np.asarray(nptr)[-1]) == npairs


# ---- multi-image (R>1) regression guards -------------------------------------
# A prior review found JAX `naive` PBC dropped non-zero periodic images via a
# pinned launch shift-axis.  These lock in that the fan-out bindings enumerate the
# *full* multi-image neighbor set (cutoff > half-cell) — the case
# ``_check_pair_matrix`` self-consistency cannot catch (it only validates pairs that
# *are* present).


def _cubic_lattice(n_side, spacing, dtype):
    coords = [
        [i * spacing, j * spacing, k * spacing]
        for i in range(n_side)
        for j in range(n_side)
        for k in range(n_side)
    ]
    return jnp.array(coords, dtype=dtype)


def test_batch_naive_pair_fn_multiple_images_r_gt_1():
    """Batched naive enumerates non-zero periodic images in each system.

    This is the JAX-local CI guard for the launch shift-axis regression covered
    more deeply by the slow JAX-vs-Torch parity test below.
    """
    from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list

    dtype = jnp.float64
    base = _cubic_lattice(2, 1.0, dtype)
    n = int(base.shape[0])
    pos = jnp.concatenate([base, base], axis=0)
    batch_idx = jnp.concatenate([jnp.zeros(n, jnp.int32), jnp.ones(n, jnp.int32)])
    batch_ptr = jnp.array([0, n, 2 * n], dtype=jnp.int32)
    cell = jnp.tile((jnp.eye(3, dtype=dtype) * 2.0)[None], (2, 1, 1))
    pbc = jnp.ones((2, 3), dtype=jnp.bool_)
    pair_params = _pair_params(2 * n, dtype)

    nm, nn, _shifts, nd, nv, pe, pf = batch_naive_neighbor_list(
        pos,
        1.1,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        cell=cell,
        pbc=pbc,
        max_neighbors=64,
        max_atoms_per_system=n,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pair_params,
    )

    _check_pair_matrix(nm, nn, nv, nd, pe, pf, pair_params)
    assert int(np.asarray(nn).min()) > 3
    assert int(np.asarray(nn).sum()) > 2 * n * 3


@pytest.mark.slow
@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_batch_naive_pair_fn_matches_torch_multi_image(dtype):
    """Cross-backend batch_naive in the multi-image regime (cutoff 1.1 / cell 2.0).

    batch_naive derives each periodic image from a *launch* shift-axis (like naive),
    so a future pin of that axis would silently drop images — this guards it against
    Torch (the independent oracle)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is required for this cross-backend check")
    from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list as bn_j
    from nvalchemiops.torch.neighbors.batch_naive import (
        batch_naive_neighbor_list as bn_t,
    )

    td = torch.float32 if dtype == jnp.float32 else torch.float64
    base = _cubic_lattice(2, 1.0, dtype)  # 8 atoms, spacing 1.0
    n = base.shape[0]
    pos = jnp.concatenate([base, base], 0)
    bidx = jnp.concatenate([jnp.zeros(n, jnp.int32), jnp.ones(n, jnp.int32)])
    bptr = jnp.array([0, n, 2 * n], jnp.int32)
    cell = jnp.tile((jnp.eye(3, dtype=dtype) * 2.0)[None], (2, 1, 1))
    pbc = jnp.ones((2, 3), jnp.bool_)
    pp = ((jnp.arange(2 * n, dtype=dtype) + 1.0) * 0.5).reshape(2 * n, 1)

    nl_j, _p, _s, d_j, v_j, pe_j, pf_j = bn_j(
        pos,
        1.1,
        batch_idx=bidx,
        batch_ptr=bptr,
        cell=cell,
        pbc=pbc,
        max_neighbors=64,
        max_atoms_per_system=n,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=pp,
    )
    nl_t, _pt, _st, d_t, v_t, pe_t, pf_t = bn_t(
        torch.tensor(np.asarray(pos), dtype=td, device="cuda"),
        1.1,
        batch_idx=torch.tensor(np.asarray(bidx), dtype=torch.int32, device="cuda"),
        batch_ptr=torch.tensor(np.asarray(bptr), dtype=torch.int32, device="cuda"),
        cell=torch.tensor(np.asarray(cell), dtype=td, device="cuda"),
        pbc=torch.tensor(np.asarray(pbc), device="cuda"),
        max_neighbors=64,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=_PAIR_FN[dtype],
        pair_params=torch.tensor(np.asarray(pp), dtype=td, device="cuda"),
    )
    assert nl_j.shape[1] == nl_t.shape[1]
    # Genuinely multi-image: each atom sees its 3 axis-neighbours via two images
    # (6/atom) vs 3/atom without images, so total > 2 * n * 3.
    assert int(np.asarray(nl_j).shape[1]) > 2 * n * 3
    i_j, j_j, [_dj, vj, pej, pfj] = _canon_coo(nl_j, [d_j, v_j, pe_j, pf_j])
    i_t, j_t, [_dt, vt, pet, pft] = _canon_coo(
        nl_t.cpu().numpy(),
        [
            d_t.cpu().numpy(),
            v_t.detach().cpu().numpy(),
            pe_t.detach().cpu().numpy(),
            pf_t.detach().cpu().numpy(),
        ],
    )
    assert np.array_equal(i_j, i_t) and np.array_equal(j_j, j_t)
    np.testing.assert_allclose(pej, pet, atol=1e-5, rtol=1e-5)
    # Duplicate periodic images can tie on (i, j, distance) while having opposite
    # image vectors, so force vectors are checked against each backend's vectors.
    np.testing.assert_allclose(pfj, -vj, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pft, -vt, atol=1e-5, rtol=1e-5)


def test_cluster_tile_pair_fn_multi_image_matches_naive():
    """cluster_tile (fp32) in the multi-image regime (cutoff 2.0 / cell 3.0) must
    enumerate the same neighbor set as the (verified) naive reference — guards
    against silent image-dropping on the tile path."""
    from nvalchemiops.jax.neighbors.cluster_tile import cluster_tile_neighbor_list
    from nvalchemiops.jax.neighbors.naive import naive_neighbor_list

    pos = _cubic_lattice(3, 1.0, jnp.float32)  # 27 atoms
    n = pos.shape[0]
    cell = jnp.eye(3, dtype=jnp.float32) * 3.0
    pp = ((jnp.arange(n, dtype=jnp.float32) + 1.0) * 0.5).reshape(n, 1)

    nm_ct, nn_ct, _sh, nd_ct, nv_ct, pe_ct, pf_ct = cluster_tile_neighbor_list(
        pos,
        2.0,
        cell,
        max_neighbors=256,
        return_distances=True,
        return_vectors=True,
        pair_fn=_sum_pair_fn_f32,
        pair_params=pp,
    )
    _check_pair_matrix(nm_ct, nn_ct, nv_ct, nd_ct, pe_ct, pf_ct, pp)
    # Cross-method count parity: naive is the verified multi-image reference.
    _nm, nn_naive, *_ = naive_neighbor_list(
        pos,
        2.0,
        cell=cell[jnp.newaxis],
        pbc=jnp.array([[True, True, True]]),
        max_neighbors=256,
        return_distances=True,
        return_vectors=True,
        pair_fn=_sum_pair_fn_f32,
        pair_params=pp,
    )
    assert int(np.asarray(nn_ct).sum()) == int(np.asarray(nn_naive).sum())
    assert int(np.asarray(nn_ct).min()) > 6  # genuinely multi-image

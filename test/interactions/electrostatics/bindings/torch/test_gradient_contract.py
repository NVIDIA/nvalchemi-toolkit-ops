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

from __future__ import annotations

import importlib
import warnings

import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    ewald_real_space,
    ewald_reciprocal_space,
    ewald_summation,
    generate_k_vectors_ewald_summation,
    generate_k_vectors_pme,
    particle_mesh_ewald,
    pme_reciprocal_space,
)
from nvalchemiops.torch.interactions.electrostatics._util import (
    _combine_electrostatic_outputs,
)

pytestmark = pytest.mark.gpu


def _device() -> torch.device:
    """Return CUDA device or skip the contract tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _system() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a tiny neutral periodic system."""
    device = _device()
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.6, 1.1, 0.5]],
        dtype=torch.float64,
        device=device,
    )
    charges = torch.tensor([0.7, -0.4, -0.3], dtype=torch.float64, device=device)
    cell = torch.eye(3, dtype=torch.float64, device=device) * 5.0
    return positions, charges, cell


def _neighbor_list() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a fixed complete directed neighbor list for the tiny system."""
    device = _device()
    neighbor_list = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]],
        dtype=torch.int32,
        device=device,
    )
    neighbor_ptr = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=device)
    shifts = torch.zeros(6, 3, dtype=torch.int32, device=device)
    return neighbor_list, neighbor_ptr, shifts


def _assert_alpha_has_no_grad(energy: torch.Tensor, alpha: torch.Tensor) -> None:
    """Assert alpha is outside the public autograd contract."""
    (grad_alpha,) = torch.autograd.grad(
        energy.sum(),
        (alpha,),
        allow_unused=True,
    )
    assert grad_alpha is None


def _assert_no_cache_warning(records: list[warnings.WarningMessage]) -> None:
    """Assert no static-cache contract warning was emitted."""
    messages = [str(record.message) for record in records]
    assert not any("Precomputed" in message for message in messages)
    assert not any("current cell" in message for message in messages)


def test_combined_charge_gradients_compile_without_disabled_helper() -> None:
    """Charge-gradient output composition stays traceable under torch.compile."""
    device = _device()
    energies_real = torch.tensor([1.0, 2.0], dtype=torch.float64, device=device)
    energies_recip = torch.tensor([0.5, 0.25], dtype=torch.float64, device=device)
    forces_real = torch.arange(6, dtype=torch.float64, device=device).reshape(2, 3)
    forces_recip = torch.ones_like(forces_real)
    charge_real = torch.tensor([0.1, 0.2], dtype=torch.float64, device=device)
    charge_recip = torch.tensor([0.3, 0.4], dtype=torch.float64, device=device)
    virial_real = torch.eye(3, dtype=torch.float64, device=device).reshape(1, 3, 3)
    virial_recip = torch.ones((1, 3, 3), dtype=torch.float64, device=device)

    def combine(
        real_charge_grads: torch.Tensor,
        reciprocal_charge_grads: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return _combine_electrostatic_outputs(
            (energies_real, forces_real, real_charge_grads, virial_real),
            (energies_recip, forces_recip, reciprocal_charge_grads, virial_recip),
            None,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )

    eager = combine(charge_real, charge_recip)
    compiled = torch.compile(combine)(charge_real, charge_recip)

    assert isinstance(compiled, tuple)
    assert len(compiled) == len(eager)
    for eager_tensor, compiled_tensor in zip(eager, compiled, strict=True):
        torch.testing.assert_close(compiled_tensor, eager_tensor)


def test_compiled_ewald_first_order_gradients_match_eager() -> None:
    """Compiled Ewald energy preserves supported first-order gradients."""
    positions, charges, cell = _system()
    alpha = torch.tensor([0.35], dtype=torch.float64, device=positions.device)
    neighbor_list, neighbor_ptr, shifts = _neighbor_list()

    def loss_fn(
        pos: torch.Tensor,
        q: torch.Tensor,
        lattice: torch.Tensor,
    ) -> torch.Tensor:
        return ewald_summation(
            pos,
            q,
            lattice,
            alpha=alpha,
            k_cutoff=5.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=shifts,
        ).sum()

    # Register lazy custom-op chains before Dynamo traces the runtime path.
    loss_fn(positions, charges, cell).detach()

    eager_inputs = (
        positions.detach().requires_grad_(True),
        charges.detach().requires_grad_(True),
        cell.detach().requires_grad_(True),
    )
    compiled_inputs = tuple(t.detach().requires_grad_(True) for t in eager_inputs)

    eager_loss = loss_fn(*eager_inputs)
    eager_grads = torch.autograd.grad(eager_loss, eager_inputs)

    compiled_loss = torch.compile(loss_fn)(*compiled_inputs)
    compiled_grads = torch.autograd.grad(compiled_loss, compiled_inputs)

    torch.testing.assert_close(compiled_loss, eager_loss)
    for compiled_grad, eager_grad in zip(compiled_grads, eager_grads, strict=True):
        torch.testing.assert_close(compiled_grad, eager_grad, rtol=5e-7, atol=5e-8)


def test_ewald_alpha_is_setup_constant() -> None:
    """Ewald real, reciprocal, and full APIs do not differentiate alpha."""
    positions, charges, cell = _system()
    charges = charges.detach().requires_grad_(True)
    alpha = torch.tensor([0.35], dtype=torch.float64, device=positions.device)
    alpha.requires_grad_(True)
    neighbor_list, neighbor_ptr, shifts = _neighbor_list()

    real = ewald_real_space(
        positions,
        charges,
        cell,
        alpha,
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=shifts,
    )
    _assert_alpha_has_no_grad(real, alpha)

    k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=5.0)
    reciprocal = ewald_reciprocal_space(
        positions,
        charges,
        cell,
        k_vectors,
        alpha,
    )
    _assert_alpha_has_no_grad(reciprocal, alpha)

    full = ewald_summation(
        positions,
        charges,
        cell,
        alpha=alpha,
        k_cutoff=5.0,
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=shifts,
    )
    _assert_alpha_has_no_grad(full, alpha)


def test_pme_alpha_is_setup_constant() -> None:
    """PME reciprocal and full APIs do not differentiate alpha."""
    positions, charges, cell = _system()
    charges = charges.detach().requires_grad_(True)
    alpha = torch.tensor([0.35], dtype=torch.float64, device=positions.device)
    alpha.requires_grad_(True)
    mesh_dimensions = (8, 8, 8)

    reciprocal = pme_reciprocal_space(
        positions,
        charges,
        cell,
        alpha=alpha,
        mesh_dimensions=mesh_dimensions,
    )
    _assert_alpha_has_no_grad(reciprocal, alpha)

    neighbor_list, neighbor_ptr, shifts = _neighbor_list()
    full = particle_mesh_ewald(
        positions,
        charges,
        cell,
        alpha=alpha,
        mesh_dimensions=mesh_dimensions,
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=shifts,
    )
    _assert_alpha_has_no_grad(full, alpha)


def test_ewald_silently_accepts_k_vectors_when_cell_requires_grad() -> None:
    """Caller-supplied Ewald k-vectors are static cell-gradient caches."""
    positions, charges, cell = _system()
    cell = cell.detach().requires_grad_(True)
    alpha = torch.tensor([0.35], dtype=torch.float64, device=positions.device)
    k_vectors = (
        generate_k_vectors_ewald_summation(cell.detach(), k_cutoff=5.0)
        .detach()
        .requires_grad_(True)
    )

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        reciprocal = ewald_reciprocal_space(positions, charges, cell, k_vectors, alpha)
    _assert_no_cache_warning(records)
    grad_cell, grad_k = torch.autograd.grad(
        reciprocal.sum(),
        (cell, k_vectors),
        allow_unused=True,
    )
    assert torch.isfinite(grad_cell).all()
    assert grad_k is None

    neighbor_list, neighbor_ptr, shifts = _neighbor_list()
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        full = ewald_summation(
            positions,
            charges,
            cell,
            alpha=alpha,
            k_vectors=k_vectors,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=shifts,
        )
    _assert_no_cache_warning(records)
    grad_cell, grad_k = torch.autograd.grad(
        full.sum(),
        (cell, k_vectors),
        allow_unused=True,
    )
    assert torch.isfinite(grad_cell).all()
    assert grad_k is None


def test_ewald_k_vector_source_cell_is_static_metadata() -> None:
    """Gradients do not flow through the cell that produced supplied k-vectors."""
    positions, charges, cell = _system()
    current_cell = cell.detach().requires_grad_(True)
    cache_cell = (cell.detach() * 1.02).requires_grad_(True)
    alpha = torch.tensor([0.35], dtype=torch.float64, device=positions.device)
    k_vectors = generate_k_vectors_ewald_summation(cache_cell, k_cutoff=5.0)

    reciprocal = ewald_reciprocal_space(
        positions,
        charges,
        current_cell,
        k_vectors,
        alpha,
    )
    grad_current, grad_cache_cell = torch.autograd.grad(
        reciprocal.sum(),
        (current_cell, cache_cell),
        allow_unused=True,
    )
    assert torch.isfinite(grad_current).all()
    assert grad_cache_cell is None


def test_ewald_public_reciprocal_exposes_no_cache_bypass_keyword() -> None:
    """The public reciprocal component has no internal cell-gradient bypass."""
    positions, charges, cell = _system()
    cell = cell.detach().requires_grad_(True)
    alpha = torch.tensor([0.35], dtype=torch.float64, device=positions.device)
    k_vectors = generate_k_vectors_ewald_summation(cell.detach(), k_cutoff=5.0)

    with pytest.raises(TypeError, match="unexpected keyword"):
        ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            allow_cell_grad_with_k_vectors=True,
        )


@pytest.mark.parametrize(
    "cache_name", ["k_vectors", "k_squared", "volume", "cell_inv_t"]
)
def test_pme_silently_accepts_cell_derived_caches_when_cell_requires_grad(
    cache_name: str,
) -> None:
    """Caller-supplied PME reciprocal metadata is static for cell gradients."""
    positions, charges, cell = _system()
    cell = cell.detach().requires_grad_(True)
    alpha = torch.tensor([0.35], dtype=torch.float64, device=positions.device)
    mesh_dimensions = (8, 8, 8)
    k_vectors, k_squared = generate_k_vectors_pme(cell.detach(), mesh_dimensions)
    kwargs = {
        "k_vectors": k_vectors.detach().requires_grad_(True),
        "k_squared": k_squared.detach().requires_grad_(True),
        "volume": torch.abs(torch.det(cell.detach()))
        .reshape(1)
        .detach()
        .requires_grad_(True),
        "cell_inv_t": torch.linalg.inv(cell.detach())
        .transpose(-1, -2)
        .detach()
        .requires_grad_(True),
    }

    call_kwargs = {cache_name: kwargs[cache_name]}
    cache_tensor = kwargs[cache_name]
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        reciprocal = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            **call_kwargs,
        )
    _assert_no_cache_warning(records)
    grad_cell, grad_cache = torch.autograd.grad(
        reciprocal.sum(),
        (cell, cache_tensor),
        allow_unused=True,
    )
    assert torch.isfinite(grad_cell).all()
    assert grad_cache is None

    neighbor_list, neighbor_ptr, shifts = _neighbor_list()
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        full = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=shifts,
            **call_kwargs,
        )
    _assert_no_cache_warning(records)
    grad_cell, grad_cache = torch.autograd.grad(
        full.sum(),
        (cell, cache_tensor),
        allow_unused=True,
    )
    assert torch.isfinite(grad_cell).all()
    assert grad_cache is None


@pytest.mark.parametrize(
    "cache_name", ["k_vectors", "k_squared", "volume", "cell_inv_t"]
)
def test_pme_cache_source_cell_is_static_metadata(cache_name: str) -> None:
    """Gradients do not flow through the cell that produced supplied PME metadata."""
    positions, charges, cell = _system()
    current_cell = cell.detach().requires_grad_(True)
    cache_cell = (cell.detach() * 1.02).requires_grad_(True)
    alpha = torch.tensor([0.35], dtype=torch.float64, device=positions.device)
    mesh_dimensions = (8, 8, 8)
    k_vectors, k_squared = generate_k_vectors_pme(cache_cell, mesh_dimensions)
    caches = {
        "k_vectors": k_vectors,
        "k_squared": k_squared,
        "volume": torch.abs(torch.det(cache_cell)).reshape(1),
        "cell_inv_t": torch.linalg.inv(cache_cell).transpose(-1, -2),
    }

    reciprocal = pme_reciprocal_space(
        positions,
        charges,
        current_cell,
        alpha=alpha,
        mesh_dimensions=mesh_dimensions,
        **{cache_name: caches[cache_name]},
    )
    grad_current, grad_cache_cell = torch.autograd.grad(
        reciprocal.sum(),
        (current_cell, cache_cell),
        allow_unused=True,
    )
    assert torch.isfinite(grad_current).all()
    assert grad_cache_cell is None


def test_full_apis_keep_cell_gradients_without_dynamic_caches() -> None:
    """Full Ewald and PME still differentiate cell when metadata is internal."""
    positions, charges, cell = _system()
    cell = cell.detach().requires_grad_(True)
    alpha = torch.tensor([0.35], dtype=torch.float64, device=positions.device)
    neighbor_list, neighbor_ptr, shifts = _neighbor_list()

    ewald_energy = ewald_summation(
        positions,
        charges,
        cell,
        alpha=alpha,
        k_cutoff=5.0,
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=shifts,
    )
    (ewald_cell_grad,) = torch.autograd.grad(ewald_energy.sum(), cell)
    assert torch.isfinite(ewald_cell_grad).all()

    pme_energy = particle_mesh_ewald(
        positions,
        charges,
        cell,
        alpha=alpha,
        mesh_dimensions=(8, 8, 8),
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=shifts,
    )
    (pme_cell_grad,) = torch.autograd.grad(pme_energy.sum(), cell)
    assert torch.isfinite(pme_cell_grad).all()


def test_torch_pme_virial_background_uses_supplied_volume(monkeypatch) -> None:
    """Torch PME virial background uses caller-supplied static volume."""
    positions, _charges, cell = _system()
    charges = torch.tensor(
        [0.7, -0.4, 0.2], dtype=torch.float64, device=positions.device
    )
    alpha = torch.tensor([0.35], dtype=torch.float64, device=positions.device)
    supplied_volume = (0.5 * torch.abs(torch.linalg.det(cell)).reshape(1)).contiguous()
    pme_module = importlib.import_module(
        "nvalchemiops.torch.interactions.electrostatics.pme"
    )

    def _zero_reciprocal_virial(*args, **kwargs):
        del args
        return torch.zeros(
            (1, 3, 3),
            dtype=kwargs["dtype"],
            device=kwargs["device"],
        )

    monkeypatch.setattr(
        pme_module,
        "_compute_pme_reciprocal_virial",
        _zero_reciprocal_virial,
    )

    with pytest.warns(DeprecationWarning):
        _energies, virial = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=(8, 8, 8),
            compute_virial=True,
            volume=supplied_volume,
        )

    q_total = charges.sum()
    expected_bg = (
        torch.pi * q_total * q_total / (2.0 * alpha[0] * alpha[0] * supplied_volume[0])
    )
    expected = -expected_bg * torch.eye(
        3, dtype=positions.dtype, device=positions.device
    )
    torch.testing.assert_close(virial[0], expected, rtol=1e-9, atol=1e-10)

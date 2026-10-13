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

"""PME mesh selection through public setup, full, and component APIs."""

import inspect
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from nvalchemiops.jax.interactions.electrostatics import parameters as jax_parameters
from nvalchemiops.jax.interactions.electrostatics import (
    particle_mesh_ewald as jax_particle_mesh_ewald,
)
from nvalchemiops.jax.interactions.electrostatics import (
    pme_reciprocal_space as jax_pme_reciprocal_space,
)
from nvalchemiops.torch.interactions.electrostatics import (
    parameters as torch_parameters,
)
from nvalchemiops.torch.interactions.electrostatics import (
    particle_mesh_ewald,
    pme_reciprocal_space,
)


@pytest.mark.parametrize(
    "padding, expected", [(None, (128, 140, 140)), (0.0, (128, 135, 140))]
)
def test_default_mesh_uses_smooth_rounding_at_dyadic_and_prime_boundaries(
    padding: float | None, expected: tuple[int, int, int]
) -> None:
    """Default snapping and the minimum-mesh option agree across backends."""
    accuracy = 1e-6
    targets = np.array([128.0 - 1e-6, 128.0 + 1e-6, 137.0])
    cells = np.diag(targets)[None]
    alpha = np.array([1.5 * accuracy**0.2])
    kwargs = {} if padding is None else {"fft_padding_fraction": padding}
    assert (
        torch_parameters.estimate_pme_mesh_dimensions(
            torch.from_numpy(cells), torch.from_numpy(alpha), accuracy, **kwargs
        )
        == expected
    )
    assert (
        jax_parameters.estimate_pme_mesh_dimensions(
            jnp.asarray(cells), jnp.asarray(alpha), accuracy, **kwargs
        )
        == expected
    )


@pytest.mark.parametrize("spline_order", [4, 5, 6])
@pytest.mark.parametrize("accuracy", [1e-4, 1e-6])
@pytest.mark.parametrize("padding", [0.0, 0.25])
def test_backend_mesh_selection_matches_for_heterogeneous_batch(
    spline_order: int, accuracy: float, padding: float
) -> None:
    """Both setup APIs size the same batch for its requested spline order."""
    cells = np.stack([np.diag([8.0, 12.0, 16.0]), np.diag([16.0, 8.0, 12.0])])
    alpha = np.array([0.25, 0.45])
    torch_dimensions = torch_parameters.estimate_pme_mesh_dimensions(
        torch.from_numpy(cells),
        torch.from_numpy(alpha),
        accuracy,
        spline_order=spline_order,
        fft_padding_fraction=padding,
    )
    jax_dimensions = jax_parameters.estimate_pme_mesh_dimensions(
        jnp.asarray(cells),
        jnp.asarray(alpha),
        accuracy,
        spline_order=spline_order,
        fft_padding_fraction=padding,
    )
    assert torch_dimensions == jax_dimensions


@pytest.mark.parametrize("parameters", [torch_parameters, jax_parameters])
@pytest.mark.parametrize(
    "name",
    [
        "estimate_pme_mesh_dimensions",
        "estimate_pme_parameters",
        "mesh_spacing_to_dimensions",
    ],
)
def test_public_padding_budget_is_keyword_only_with_shared_default(
    parameters, name: str
) -> None:
    """Sizing APIs expose the same optional total-point budget."""
    parameter = inspect.signature(getattr(parameters, name)).parameters[
        "fft_padding_fraction"
    ]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == 0.25


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_public_parameter_estimator_can_disable_mesh_snapping(backend: str) -> None:
    """The opt-out changes only mesh sizing and leaves alpha and cutoff intact."""
    convert = torch.from_numpy if backend == "torch" else jnp.asarray
    parameters = torch_parameters if backend == "torch" else jax_parameters
    positions = convert(np.zeros((128, 3), dtype=np.float64))
    cell = convert(np.eye(3, dtype=np.float64) * 20.0)
    smallest = parameters.estimate_pme_parameters(
        positions, cell, real_space_cutoff=9.0, fft_padding_fraction=0
    )
    snapped = parameters.estimate_pme_parameters(positions, cell, real_space_cutoff=9.0)
    assert smallest.mesh_dimensions == (90, 90, 90)
    assert snapped.mesh_dimensions == (96, 96, 96)
    assert math.prod(snapped.mesh_dimensions) <= 1.25 * math.prod(
        smallest.mesh_dimensions
    )
    np.testing.assert_array_equal(np.asarray(snapped.alpha), np.asarray(smallest.alpha))
    np.testing.assert_array_equal(
        np.asarray(snapped.real_space_cutoff), np.asarray(smallest.real_space_cutoff)
    )


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("spline_order", [4, 5, 6])
def test_public_spacing_and_accuracy_sizing_share_the_snapping_budget(
    backend: str, spline_order: int
) -> None:
    """Spacing always supports snapping while legacy accuracy orders stay dyadic."""
    convert = torch.from_numpy if backend == "torch" else jnp.asarray
    parameters = torch_parameters if backend == "torch" else jax_parameters
    cell = convert(np.eye(3, dtype=np.float64) * 500.0)
    alpha = convert(np.array([1.5 * 1e-6**0.2], dtype=np.float64))
    assert parameters.mesh_spacing_to_dimensions(
        cell, 1.0, spline_order=spline_order, fft_padding_fraction=0
    ) == (500, 500, 500)
    assert parameters.mesh_spacing_to_dimensions(
        cell, 1.0, spline_order=spline_order
    ) == (512, 512, 512)
    expected = (500, 500, 500) if spline_order == 5 else (512, 512, 512)
    assert (
        parameters.estimate_pme_mesh_dimensions(
            cell, alpha, spline_order=spline_order, fft_padding_fraction=0
        )
        == expected
    )
    assert parameters.estimate_pme_mesh_dimensions(
        cell, alpha, spline_order=spline_order
    ) == (512, 512, 512)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("padding", [-1, math.nan, math.inf, True])
def test_public_sizing_apis_reject_invalid_padding_budgets(
    backend: str, padding: float
) -> None:
    """Invalid padding fails in accuracy, full parameter, and spacing setup."""
    convert = torch.from_numpy if backend == "torch" else jnp.asarray
    parameters = torch_parameters if backend == "torch" else jax_parameters
    cell = convert(np.eye(3, dtype=np.float64) * 20.0)
    alpha = convert(np.array([0.3], dtype=np.float64))
    positions = convert(np.zeros((2, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="fft_padding_fraction"):
        parameters.estimate_pme_mesh_dimensions(
            cell, alpha, fft_padding_fraction=padding
        )
    with pytest.raises(ValueError, match="fft_padding_fraction"):
        parameters.estimate_pme_parameters(
            positions, cell, fft_padding_fraction=padding
        )
    with pytest.raises(ValueError, match="fft_padding_fraction"):
        parameters.mesh_spacing_to_dimensions(cell, 1.0, fft_padding_fraction=padding)


@pytest.mark.parametrize("padding", [0.0, 0.25])
def test_spacing_snapping_sizes_the_largest_axis_across_the_batch(
    padding: float,
) -> None:
    """Anisotropic batch sizing uses every system before applying its budget."""
    cells = np.stack([np.diag([500.0, 72.0, 125.0]), np.diag([480.0, 75.0, 120.0])])
    expected = (500, 75, 125) if padding == 0 else (500, 80, 128)
    for parameters, convert in (
        (torch_parameters, torch.from_numpy),
        (jax_parameters, jnp.asarray),
    ):
        assert (
            parameters.mesh_spacing_to_dimensions(
                convert(cells), 1.0, fft_padding_fraction=padding
            )
            == expected
        )


@pytest.mark.parametrize("spline_order", [4, 6])
@pytest.mark.parametrize("accuracy", [1e-4, 1e-6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_implicit_mesh_preserves_upstream_resolution(
    spline_order: int, accuracy: float, dtype: torch.dtype
) -> None:
    """Automatic dimensions retain upstream resolution and assignment support."""
    cells = torch.stack(
        [
            torch.diag(torch.tensor([8.0, 12.0, 16.0])),
            torch.diag(torch.tensor([16.0, 8.0, 12.0])),
        ]
    ).to(dtype)
    alpha = torch.tensor([0.25, 0.45], dtype=dtype)
    continuous_target = (
        2.0
        * alpha[:, None]
        * torch.linalg.vector_norm(cells, dim=2)
        / (3.0 * accuracy**0.2)
    )
    minimum = torch.ceil(continuous_target.max(dim=0).values).to(torch.int32)
    actual = torch_parameters.estimate_pme_mesh_dimensions(
        cells, alpha, accuracy, spline_order=spline_order
    )
    for value, expected in zip(actual, minimum.tolist(), strict=True):
        lower_bound = max(2, spline_order, expected)
        assert value >= lower_bound
        assert value == 1 << (lower_bound - 1).bit_length()


@pytest.mark.parametrize("spline_order", [4, 5, 6])
@pytest.mark.parametrize("parameter_source", ["automatic", "alpha", "spacing"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_full_torch_implicit_mesh_matches_explicit_energy_and_derivatives(
    spline_order: int, parameter_source: str, device: str
) -> None:
    """Full PME uses the setup mesh for its order and preserves derivatives."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    positions = torch.tensor(
        [[1.0, 1.0, 1.0], [2.0, 1.5, 1.0], [1.5, 2.5, 2.0], [3.0, 2.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
        device=device,
    )
    charges = torch.tensor(
        [1.0, -1.0, 0.5, -0.5], requires_grad=True, dtype=torch.float64, device=device
    )
    cell = 8.0 * torch.eye(3, dtype=torch.float64, device=device)
    neighbors = torch.arange(4, dtype=torch.int32, device=device).expand(4, 4).clone()
    neighbors.fill_diagonal_(4)
    common = {
        "neighbor_matrix": neighbors,
        "neighbor_matrix_shifts": torch.zeros(
            4, 4, 3, dtype=torch.int32, device=device
        ),
        "spline_order": spline_order,
        "accuracy": 1e-4,
    }
    if parameter_source == "spacing":
        alpha = torch.tensor([0.35], dtype=torch.float64, device=device)
        dimensions = torch_parameters.mesh_spacing_to_dimensions(
            cell, 3.0, spline_order=spline_order
        )
        assert dimensions == (spline_order,) * 3
        implicit = particle_mesh_ewald(
            positions, charges, cell, alpha=alpha, mesh_spacing=3.0, **common
        )
    elif parameter_source == "alpha":
        alpha = torch.tensor([0.35], dtype=torch.float64, device=device)
        dimensions = torch_parameters.estimate_pme_mesh_dimensions(
            cell, alpha, accuracy=1e-4, spline_order=spline_order
        )
        implicit = particle_mesh_ewald(positions, charges, cell, alpha=alpha, **common)
    else:
        parameters = torch_parameters.estimate_pme_parameters(
            positions, cell, accuracy=1e-4, spline_order=spline_order
        )
        alpha, dimensions = parameters.alpha, parameters.mesh_dimensions
        implicit = particle_mesh_ewald(positions, charges, cell, **common)
    explicit = particle_mesh_ewald(
        positions, charges, cell, alpha=alpha, mesh_dimensions=dimensions, **common
    )
    torch.testing.assert_close(implicit, explicit)
    implicit_gradients = torch.autograd.grad(implicit.sum(), (positions, charges))
    explicit_gradients = torch.autograd.grad(explicit.sum(), (positions, charges))
    for actual, expected in zip(implicit_gradients, explicit_gradients):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("spacing_kind", ["integer", "scalar_tensor"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_component_scalar_mesh_spacing_matches_float(
    spacing_kind: str, device: str
) -> None:
    """Scalar spacing forms preserve public reciprocal PME energies and gradients."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    positions = torch.tensor(
        [[1.1, 1.2, 1.3], [2.2, 1.7, 1.3]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    charges = torch.tensor(
        [1.0, -1.0], dtype=torch.float64, device=device, requires_grad=True
    )
    cell = 12.0 * torch.eye(3, dtype=torch.float64, device=device)
    spacing = (
        2
        if spacing_kind == "integer"
        else torch.tensor(2.0, dtype=cell.dtype, device=device)
    )
    common = {"alpha": 0.3, "spline_order": 5}

    expected = pme_reciprocal_space(
        positions, charges, cell, mesh_spacing=2.0, **common
    )
    actual = pme_reciprocal_space(
        positions, charges, cell, mesh_spacing=spacing, **common
    )
    torch.testing.assert_close(actual, expected)
    expected_gradients = torch.autograd.grad(expected.sum(), (positions, charges))
    actual_gradients = torch.autograd.grad(actual.sum(), (positions, charges))
    for result, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, reference)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("spline_order", [4, 5, 6])
def test_torch_component_mesh_spacing_uses_every_cell_in_batch(
    device: str, spline_order: int
) -> None:
    """Component spacing resolution matches an explicit batch-wide mesh."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    positions = torch.tensor(
        [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=torch.float64, device=device
    )
    charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
    cells = torch.stack([8.0 * torch.eye(3), 12.0 * torch.eye(3)]).to(
        dtype=torch.float64, device=device
    )
    batch_idx = torch.tensor([0, 1], dtype=torch.int32, device=device)
    common = {"alpha": 0.3, "batch_idx": batch_idx, "spline_order": spline_order}
    dimensions = torch_parameters.mesh_spacing_to_dimensions(
        cells, 3.0, spline_order=spline_order
    )
    assert dimensions == (spline_order,) * 3
    implicit = pme_reciprocal_space(
        positions, charges, cells, mesh_spacing=3.0, **common
    )
    explicit = pme_reciprocal_space(
        positions, charges, cells, mesh_dimensions=dimensions, **common
    )
    torch.testing.assert_close(implicit, explicit)


@pytest.mark.parametrize("spline_order", [4, 5, 6])
@pytest.mark.parametrize("parameter_source", ["automatic", "alpha", "spacing"])
def test_full_jax_implicit_mesh_matches_explicit_jit_and_derivatives(
    spline_order: int, parameter_source: str
) -> None:
    """Eager mesh selection and an explicit compiled call give equal derivatives."""
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not gpu_devices:
        pytest.skip("Warp JAX FFI requires a GPU")
    with jax.default_device(gpu_devices[0]):
        positions = jnp.array(
            [[1.0, 1.0, 1.0], [2.0, 1.5, 1.0], [1.5, 2.5, 2.0], [3.0, 2.0, 1.0]],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.0, -1.0, 0.5, -0.5], dtype=jnp.float64)
        cell = 8.0 * jnp.eye(3, dtype=jnp.float64)
        neighbors = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (4, 4))
        neighbors = neighbors.at[jnp.arange(4), jnp.arange(4)].set(4)
        common = {
            "neighbor_matrix": neighbors,
            "neighbor_matrix_shifts": jnp.zeros((4, 4, 3), dtype=jnp.int32),
            "spline_order": spline_order,
            "accuracy": 1e-4,
        }
        if parameter_source == "spacing":
            alpha = jnp.array([0.35], dtype=jnp.float64)
            dimensions = jax_parameters.mesh_spacing_to_dimensions(
                cell, 3.0, spline_order=spline_order
            )
            assert dimensions == (spline_order,) * 3
            implicit_kwargs = {"alpha": alpha, "mesh_spacing": 3.0}
            implicit = jax_particle_mesh_ewald(
                positions, charges, cell, **implicit_kwargs, **common
            )
        elif parameter_source == "alpha":
            alpha = jnp.array([0.35], dtype=jnp.float64)
            dimensions = jax_parameters.estimate_pme_mesh_dimensions(
                cell, alpha, accuracy=1e-4, spline_order=spline_order
            )
            implicit = jax_particle_mesh_ewald(
                positions, charges, cell, alpha=alpha, **common
            )
            implicit_kwargs = {"alpha": alpha}
        else:
            parameters = jax_parameters.estimate_pme_parameters(
                positions, cell, accuracy=1e-4, spline_order=spline_order
            )
            alpha, dimensions = parameters.alpha, parameters.mesh_dimensions
            implicit = jax_particle_mesh_ewald(positions, charges, cell, **common)
            implicit_kwargs = {}

        def explicit_energy(pos, charge):
            return jax_particle_mesh_ewald(
                pos, charge, cell, alpha=alpha, mesh_dimensions=dimensions, **common
            )

        expected, expected_gradients = jax.value_and_grad(
            lambda pos, charge: jnp.sum(explicit_energy(pos, charge)), argnums=(0, 1)
        )(positions, charges)
        implicit_gradients = jax.grad(
            lambda pos, charge: jnp.sum(
                jax_particle_mesh_ewald(pos, charge, cell, **implicit_kwargs, **common)
            ),
            argnums=(0, 1),
        )(positions, charges)
        compiled_energy = jax.jit(explicit_energy)(positions, charges)
        compiled_gradients = jax.jit(
            jax.grad(
                lambda pos, charge: jnp.sum(explicit_energy(pos, charge)),
                argnums=(0, 1),
            )
        )(positions, charges)
        np.testing.assert_allclose(np.asarray(implicit).sum(), np.asarray(expected))
        np.testing.assert_allclose(np.asarray(compiled_energy), np.asarray(implicit))
        for actual_gradients in (implicit_gradients, compiled_gradients):
            for actual, expected_gradient in zip(actual_gradients, expected_gradients):
                np.testing.assert_allclose(
                    np.asarray(actual), np.asarray(expected_gradient)
                )


@pytest.mark.parametrize("spline_order", [4, 5, 6])
def test_jax_component_mesh_spacing_matches_explicit_jit(spline_order: int) -> None:
    """Eager component spacing uses the same batch mesh as a prepared JIT call."""
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not gpu_devices:
        pytest.skip("Warp JAX FFI requires a GPU")
    with jax.default_device(gpu_devices[0]):
        positions = jnp.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cells = jnp.stack([8.0 * jnp.eye(3), 12.0 * jnp.eye(3)])
        batch_idx = jnp.array([0, 1], dtype=jnp.int32)
        dimensions = jax_parameters.mesh_spacing_to_dimensions(
            cells, 3.0, spline_order=spline_order
        )
        assert dimensions == (spline_order,) * 3
        common = {
            "alpha": jnp.full(cells.shape[0], 0.3, dtype=positions.dtype),
            "batch_idx": batch_idx,
            "spline_order": spline_order,
        }
        implicit = jax_pme_reciprocal_space(
            positions, charges, cells, mesh_spacing=3.0, **common
        )

        def explicit_energy(pos, charge):
            return jax_pme_reciprocal_space(
                pos, charge, cells, mesh_dimensions=dimensions, **common
            )

        explicit = jax.jit(explicit_energy)(positions, charges)
        np.testing.assert_allclose(np.asarray(implicit), np.asarray(explicit))


@pytest.mark.parametrize("spline_order", [4, 5, 6])
def test_torch_prepared_spacing_mesh_matches_compiled_full_pme(
    spline_order: int,
) -> None:
    """The public spacing setup result works in a compiled full-PME calculation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    positions = torch.tensor(
        [[1.0, 1.0, 1.0], [2.0, 1.5, 1.0]], dtype=torch.float64, device="cuda"
    )
    charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device="cuda")
    cell = 8.0 * torch.eye(3, dtype=torch.float64, device="cuda")
    neighbors = torch.tensor([[2, 1], [0, 2]], dtype=torch.int32, device="cuda")
    dimensions = torch_parameters.mesh_spacing_to_dimensions(
        cell, 3.0, spline_order=spline_order
    )
    common = {
        "alpha": 0.35,
        "neighbor_matrix": neighbors,
        "neighbor_matrix_shifts": torch.zeros(
            2, 2, 3, dtype=torch.int32, device="cuda"
        ),
        "spline_order": spline_order,
    }
    implicit = particle_mesh_ewald(positions, charges, cell, mesh_spacing=3.0, **common)

    def explicit_energy(pos, charge):
        return particle_mesh_ewald(
            pos, charge, cell, mesh_dimensions=dimensions, **common
        )

    explicit = torch.compile(explicit_energy)(positions, charges)
    torch.testing.assert_close(implicit, explicit)

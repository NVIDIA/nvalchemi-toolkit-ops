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

"""Tests for private JAX electrostatics autograd helpers."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")
SymbolicZero = pytest.importorskip("jax.custom_derivatives").SymbolicZero

_autograd = pytest.importorskip(
    "nvalchemiops.jax.interactions.electrostatics._autograd"
)
_inject_charge_grad = _autograd._inject_charge_grad
_ewald = pytest.importorskip("nvalchemiops.jax.interactions.electrostatics.ewald")
_pme = pytest.importorskip("nvalchemiops.jax.interactions.electrostatics.pme")
_slab = pytest.importorskip("nvalchemiops.jax.interactions.electrostatics.slab")


def test_inject_charge_grad_single_system_uses_per_atom_cotangent():
    """Charge-gradient injection preserves per-atom energy cotangents."""
    energy = jnp.arange(3, dtype=jnp.float64)
    charge_grad = jnp.array([0.2, -0.3, 0.1], dtype=jnp.float64)
    grad_energy = jnp.array([2.0, 4.0, 9.0], dtype=jnp.float64)
    batch_idx = jnp.zeros(3, dtype=jnp.int32)

    def loss(charges):
        out = _inject_charge_grad(
            energy,
            charges,
            charge_grad,
            False,
            batch_idx,
            1,
        )
        return (out * grad_energy).sum()

    charges = jnp.ones(3, dtype=jnp.float64)
    expected = charge_grad * grad_energy
    assert jnp.array_equal(jax.grad(loss)(charges), expected)


def test_inject_charge_grad_batched_uses_per_atom_cotangent():
    """Batched charge-gradient injection preserves per-atom cotangents."""
    energy = jnp.arange(4, dtype=jnp.float64)
    charge_grad = jnp.array([0.2, -0.3, 0.1, 0.4], dtype=jnp.float64)
    grad_energy = jnp.array([2.0, 4.0, 5.0, 7.0], dtype=jnp.float64)
    batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

    def loss(charges):
        out = _inject_charge_grad(
            energy,
            charges,
            charge_grad,
            True,
            batch_idx,
            2,
        )
        return (out * grad_energy).sum()

    charges = jnp.ones(4, dtype=jnp.float64)
    expected = charge_grad * grad_energy
    assert jnp.array_equal(jax.grad(loss)(charges), expected)


def test_derivative_seams_recognize_public_symbolic_zero():
    """Electrostatics custom-JVP seams accept JAX's public zero sentinel."""
    tangents = []

    @jax.custom_jvp
    def product(x, y):
        return x * y

    def product_jvp(primals, input_tangents):
        tangents.extend(input_tangents)
        x, y = primals
        x_tangent, _y_tangent = input_tangents
        return product(x, y), x_tangent * y

    product.defjvp(product_jvp, symbolic_zeros=True)

    assert jax.grad(lambda x: product(x, jnp.float64(2.0)))(jnp.float64(1.0)) == 2.0
    symbolic_zero = next(
        tangent for tangent in tangents if isinstance(tangent, SymbolicZero)
    )

    assert _ewald._is_symbolic_zero(symbolic_zero)
    assert _pme._is_symbolic_zero(symbolic_zero)
    assert _slab._is_symbolic_zero(symbolic_zero)

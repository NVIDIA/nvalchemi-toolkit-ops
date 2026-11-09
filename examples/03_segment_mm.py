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

"""
Segmented Matrix Multiplication
===============================

``segment_mm(features, weights, lengths)`` applies one matrix to each system's
contiguous feature rows. This example uses the same small batch in PyTorch and
JAX, checks outputs and gradients, and changes runtime lengths without changing
the allocated shapes.

Three systems share eight allocated rows. Lengths ``[3, 0, 2]`` assign three
rows to the first system, none to the second, and two to the third. The remaining
three rows are padding and produce zero output.

.. important::
    This comparison requires both the ``torch`` and ``jax`` package extras,
    plus Matplotlib. Each frontend selects CUDA when available, otherwise CPU.
    Lengths are nonnegative ``int32`` values whose sum fits the row capacity;
    all inputs for a call must share a device.
"""

# %%
# One batch and a reference calculation
# -------------------------------------
# The feature array has eight rows and four input channels. Each of the three
# weight matrices maps four input channels to five output channels. Nonzero
# values in the padded feature rows let us check that padding is ignored.

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

row_capacity, input_channels, output_channels, systems = 8, 4, 5, 3
initial_lengths = (3, 0, 2)
updated_lengths = (2, 1, 3)
feature_values = (
    np.arange(row_capacity * input_channels, dtype=np.float32).reshape(
        row_capacity, input_channels
    )
    - 8
) / 16
weight_values = (
    np.arange(systems * input_channels * output_channels, dtype=np.float32).reshape(
        systems, input_channels, output_channels
    )
    - 20
) / 32


def reference_products(lengths: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    """Compute fixture outputs and gradients of their squared sum with NumPy.

    Parameters
    ----------
    lengths : tuple of int
        Known fixture lengths used to slice the shared feature array.

    Returns
    -------
    tuple of numpy.ndarray
        Output, feature gradient, and weight gradient, including zero padding.
    """
    features = feature_values.astype(np.float64)
    weights = weight_values.astype(np.float64)
    output = np.zeros((row_capacity, output_channels), dtype=np.float64)
    grad_features = np.zeros_like(features)
    grad_weights = np.zeros_like(weights)
    start = 0
    for system, length in enumerate(lengths):
        stop = start + length
        output[start:stop] = features[start:stop] @ weights[system]
        output_gradient = 2 * output[start:stop]
        grad_features[start:stop] = output_gradient @ weights[system].T
        grad_weights[system] = features[start:stop].T @ output_gradient
        start = stop
    return tuple(
        value.astype(np.float32) for value in (output, grad_features, grad_weights)
    )


expected_output = reference_products(initial_lengths)[0]

# %%
# See the packed rows
# -------------------
# The narrow panel identifies each row's system. The heat map shows the
# reference output. System 1 is empty; the three gray rows are padding.

row_owner = np.array([0, 0, 0, 2, 2, 3, 3, 3])
row_labels = ["0", "0", "0", "2", "2", "pad", "pad", "pad"]
colors = ListedColormap(["#76b900", "#f0a340", "#3298cb", "#dddddd"])
fig, (map_ax, out_ax) = plt.subplots(
    1, 2, figsize=(8, 4), sharey=True, gridspec_kw={"width_ratios": [1, 5]}
)
map_ax.imshow(row_owner[:, None], cmap=colors, vmin=0, vmax=3, aspect="auto")
for row, label in enumerate(row_labels):
    map_ax.text(0, row, label, ha="center", va="center")
map_ax.set_xticks([0], ["system"])
map_ax.set_yticks(range(row_capacity))
map_ax.set_ylabel("packed row")
heatmap = out_ax.imshow(expected_output, cmap="coolwarm", aspect="auto")
out_ax.set_xticks(range(output_channels))
out_ax.set_xlabel("output channel")
out_ax.axhline(4.5, color="black", ls="--", lw=1)
fig.colorbar(heatmap, ax=out_ax, label="output value")
fig.suptitle("Lengths [3, 0, 2]: system 1 is empty; the last three rows are padding")
plt.tight_layout()
plt.show()

# %%
# PyTorch: outputs, gradients, and compiled replay
# ------------------------------------------------
# Both floating inputs participate in autograd. The reference uses known
# Python fixture lengths; the operator always receives lengths on the device.
# Reusing the compiled callable with ``[2, 1, 3]`` activates the empty system
# and leaves two padding rows. Finish each backward before changing lengths.

import torch

from nvalchemiops.torch.segment_mm import segment_mm as torch_segment_mm

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_features = torch.tensor(feature_values, device=torch_device, requires_grad=True)
torch_weights = torch.tensor(weight_values, device=torch_device, requires_grad=True)
torch_lengths = torch.tensor(initial_lengths, dtype=torch.int32, device=torch_device)
eager_output = torch_segment_mm(torch_features, torch_weights, torch_lengths)
eager_gradients = torch.autograd.grad(
    eager_output.square().sum(), (torch_features, torch_weights)
)
for actual, expected in zip(
    (eager_output, *eager_gradients), reference_products(initial_lengths), strict=True
):
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), expected, rtol=2e-5, atol=2e-5
    )

torch_compiled = torch.compile(torch_segment_mm, fullgraph=True)
for runtime_lengths in (initial_lengths, updated_lengths):
    torch_lengths.copy_(
        torch.tensor(runtime_lengths, dtype=torch.int32, device=torch_device)
    )
    output = torch_compiled(torch_features, torch_weights, torch_lengths)
    gradients = torch.autograd.grad(
        output.square().sum(), (torch_features, torch_weights)
    )
    for actual, expected in zip(
        (output, *gradients), reference_products(runtime_lengths), strict=True
    ):
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(), expected, rtol=2e-5, atol=2e-5
        )
    np.testing.assert_array_equal(
        output.detach().cpu().numpy()[sum(runtime_lengths) :], 0
    )
    np.testing.assert_array_equal(
        gradients[0].detach().cpu().numpy()[sum(runtime_lengths) :], 0
    )
    np.testing.assert_array_equal(
        gradients[1].detach().cpu().numpy()[np.asarray(runtime_lengths) == 0], 0
    )
    print(
        f"PyTorch on {torch_device}: output and both gradients match for {runtime_lengths}."
    )

# %%
# PyTorch: BF16 computation with FP32 parameters
# ----------------------------------------------
# CUDA autocast converts both operands to BF16. CPU uses explicit casts.
# Products accumulate in FP32 and return BF16; differentiating through the
# casts returns FP32 parameter gradients with BF16 numerical precision.

torch_lengths.copy_(
    torch.tensor(initial_lengths, dtype=torch.int32, device=torch_device)
)
if torch_device.type == "cuda":
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_bf16 = torch_segment_mm(torch_features, torch_weights, torch_lengths)
else:
    output_bf16 = torch_segment_mm(
        torch_features.to(torch.bfloat16),
        torch_weights.to(torch.bfloat16),
        torch_lengths,
    )
gradients_bf16 = torch.autograd.grad(
    output_bf16.float().square().sum(), (torch_features, torch_weights)
)
np.testing.assert_equal(str(output_bf16.dtype), "torch.bfloat16")
np.testing.assert_equal(
    [str(value.dtype) for value in gradients_bf16], ["torch.float32"] * 2
)
for actual, expected in zip(
    (output_bf16, *gradients_bf16), reference_products(initial_lengths), strict=True
):
    np.testing.assert_allclose(
        actual.detach().float().cpu().numpy(), expected, rtol=3e-2, atol=3e-2
    )
print("PyTorch BF16 output and FP32 parameter gradients match within BF16 tolerance.")

# %%
# JAX: the same batch and checks
# ------------------------------
# ``value_and_grad(..., argnums=(0, 1))`` differentiates both floating inputs.
# Compile the forward and loss-gradient functions once, then reuse them with
# both sets of runtime lengths. Integer lengths are not differentiated.

import jax
import jax.numpy as jnp

from nvalchemiops.jax.segment_mm import segment_mm as jax_segment_mm

jax_device = next(
    (device for device in jax.devices() if device.platform == "gpu"), None
)
if jax_device is None:
    jax_device = jax.devices("cpu")[0]
with jax.default_device(jax_device):
    jax_features = jnp.asarray(feature_values)
    jax_weights = jnp.asarray(weight_values)
    jax_lengths = jnp.asarray(initial_lengths, dtype=jnp.int32)


def jax_loss(features, weights, lengths):
    """Return the sum of squared segmented outputs."""
    return jnp.sum(jax_segment_mm(features, weights, lengths) ** 2)


jax_compiled = (
    jax.jit(jax_segment_mm).lower(jax_features, jax_weights, jax_lengths).compile()
)
jax_gradients = (
    jax.jit(jax.value_and_grad(jax_loss, argnums=(0, 1)))
    .lower(jax_features, jax_weights, jax_lengths)
    .compile()
)
for runtime_lengths in (initial_lengths, updated_lengths):
    with jax.default_device(jax_device):
        jax_lengths = jnp.asarray(runtime_lengths, dtype=jnp.int32)
    output = jax_compiled(jax_features, jax_weights, jax_lengths)
    loss, gradients = jax_gradients(jax_features, jax_weights, jax_lengths)
    for actual, expected in zip(
        (output, *gradients), reference_products(runtime_lengths), strict=True
    ):
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(np.asarray(output)[sum(runtime_lengths) :], 0)
    np.testing.assert_array_equal(np.asarray(gradients[0])[sum(runtime_lengths) :], 0)
    np.testing.assert_array_equal(
        np.asarray(gradients[1])[np.asarray(runtime_lengths) == 0], 0
    )
    print(
        f"JAX on {jax_device}: output and both gradients match for {runtime_lengths}."
    )

# %%
# JAX: explicit BF16 casts inside the loss
# ----------------------------------------
# JAX uses explicit casts instead of autocast. Keep the casts inside the
# differentiated function so the original FP32 parameters receive gradients.
# The loss is evaluated in FP32, while the operation's output remains BF16.


def jax_bf16_loss(features, weights, lengths):
    """Return an FP32 loss and the BF16 output used to compute it."""
    output = jax_segment_mm(
        features.astype(jnp.bfloat16), weights.astype(jnp.bfloat16), lengths
    )
    return jnp.sum(output.astype(jnp.float32) ** 2), output


with jax.default_device(jax_device):
    jax_lengths = jnp.asarray(initial_lengths, dtype=jnp.int32)
(loss, output_bf16), gradients_bf16 = jax.jit(
    jax.value_and_grad(jax_bf16_loss, argnums=(0, 1), has_aux=True)
)(jax_features, jax_weights, jax_lengths)
np.testing.assert_equal(str(output_bf16.dtype), "bfloat16")
np.testing.assert_equal([str(value.dtype) for value in gradients_bf16], ["float32"] * 2)
for actual, expected in zip(
    (output_bf16, *gradients_bf16), reference_products(initial_lengths), strict=True
):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32), expected, rtol=3e-2, atol=3e-2
    )
print("JAX BF16 output and FP32 parameter gradients match within BF16 tolerance.")

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

"""JAX segmented matrix multiplication with composable derivative rules."""

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import warp as wp
from jax.extend import core
from jax.interpreters import ad, batching, mlir
from warp import JaxCallableGraphMode

from nvalchemiops.segment_mm import segment_mm as _warp_segment_mm
from nvalchemiops.segment_mm import segment_mm_backward_b as _warp_segment_mm_backward_b

__all__ = ["segment_mm"]


@lru_cache(maxsize=2)
def _callables(dtype: str):
    """Build typed FFI callables lazily, with Warp owning graph capture."""
    wp_dtype = {"float32": wp.float32, "bfloat16": wp.bfloat16}[dtype]

    def forward(
        a: wp.array2d(dtype=wp_dtype),
        b: wp.array3d(dtype=wp_dtype),
        seglen: wp.array(dtype=wp.int32),
        out: wp.array2d(dtype=wp_dtype),
    ):
        _warp_segment_mm(a, b, seglen, out)

    def weight_product(
        a: wp.array2d(dtype=wp_dtype),
        grad_out: wp.array2d(dtype=wp_dtype),
        seglen: wp.array(dtype=wp.int32),
        grad_b: wp.array3d(dtype=wp_dtype),
    ):
        _warp_segment_mm_backward_b(a, seglen, grad_out, grad_b)

    return (
        wp.jax_callable(forward, num_outputs=1, graph_mode=JaxCallableGraphMode.WARP),
        wp.jax_callable(
            weight_product, num_outputs=1, graph_mode=JaxCallableGraphMode.WARP
        ),
    )


def _validate_inputs(a, b, seglen):
    """Validate static metadata without reading runtime segment lengths."""
    for name, array, ndim in (("a", a, 2), ("b", b, 3), ("seglen", seglen, 1)):
        if not isinstance(array, (jax.Array, jax.core.Tracer)) or array.ndim != ndim:
            raise ValueError(f"{name} must be a {ndim}D JAX array")
    if a.dtype not in (jnp.float32, jnp.bfloat16) or b.dtype != a.dtype:
        raise ValueError("a and b must have the same float32 or bfloat16 dtype")
    if seglen.dtype != jnp.int32:
        raise ValueError("seglen must have dtype int32")
    if a.shape[1] <= 0 or b.shape[2] <= 0:
        raise ValueError("Channel dimensions K and N must be positive")
    if b.shape[1] != a.shape[1]:
        raise ValueError("Input channel dimensions of a and b must match")
    if seglen.shape[0] != b.shape[0]:
        raise ValueError("seglen must contain one length per segment in b")

    # Tracers carry abstract metadata; XLA owns their placement. Concrete inputs
    # can be checked here without a device-to-host transfer.
    concrete = [x for x in (a, b, seglen) if not isinstance(x, jax.core.Tracer)]
    devices = [x.devices() for x in concrete]
    if any(len(device_set) != 1 for device_set in devices):
        raise ValueError("All arrays must be on the same single CUDA device")
    if any(device.platform != "gpu" for device_set in devices for device in device_set):
        raise RuntimeError("segment_mm requires a CUDA device")
    if devices and any(device_set != devices[0] for device_set in devices[1:]):
        raise ValueError("All arrays must be on the same CUDA device")


def _forward(a, b, seglen):
    """Allocate the statically shaped output through Warp's JAX FFI."""
    forward, _ = _callables(str(a.dtype))
    (out,) = forward(a, b, seglen, output_dims={"out": (a.shape[0], b.shape[2])})
    return out


def _weight_product(a, grad_out, seglen):
    """Compute segmented outer-product sums through the shared weight kernel."""
    _, weight_product = _callables(str(a.dtype))
    (grad_b,) = weight_product(
        a,
        grad_out,
        seglen,
        output_dims={"grad_b": (seglen.shape[0], a.shape[1], grad_out.shape[1])},
    )
    return grad_b


_forward_p = core.Primitive("nvalchemiops_segment_mm")
_forward_p.def_impl(_forward)
_weight_p = core.Primitive("nvalchemiops_segment_mm_weight_product")
_weight_p.def_impl(_weight_product)


def _bind_product(primitive, a, b, seglen):
    """Promote replicated operands so their transposes reduce across the mesh."""
    operands = (a, b, seglen)
    avals = tuple(jax.typeof(x) for x in operands)
    if any(value.mat.unreduced or value.mat.reduced for value in avals):
        raise ValueError("segment_mm requires ordinary local operands inside shard_map")
    varying = frozenset().union(*(value.mat.varying for value in avals))
    operands = tuple(
        jax.lax.pcast(x, tuple(varying - value.mat.varying), to="varying")
        for x, value in zip(operands, avals, strict=True)
    )
    return primitive.bind(*operands)


def _abstract_product(a, b, seglen, *, weight_product):
    """Preserve per-device variation for both local segmented products."""
    operands = (a, b, seglen)
    if any(value.mat.unreduced or value.mat.reduced for value in operands):
        raise ValueError("segment_mm requires ordinary local operands inside shard_map")
    if any(any(axis is not None for axis in value.sharding.spec) for value in operands):
        raise ValueError("Use shard_map to pass local segments to segment_mm")
    shape = (
        (seglen.shape[0], a.shape[1], b.shape[1])
        if weight_product
        else (a.shape[0], b.shape[2])
    )
    varying = frozenset().union(*(value.mat.varying for value in operands))
    return a.update(
        shape=shape,
        sharding=a.sharding.update(spec=jax.sharding.PartitionSpec()),
        manual_axis_type=a.mat.update(varying=varying),
    )


_forward_p.def_abstract_eval(partial(_abstract_product, weight_product=False))
_weight_p.def_abstract_eval(partial(_abstract_product, weight_product=True))
mlir.register_lowering(
    _forward_p, mlir.lower_fun(_forward, multiple_results=False), platform="cuda"
)
mlir.register_lowering(
    _weight_p, mlir.lower_fun(_weight_product, multiple_results=False), platform="cuda"
)


def _bilinear_jvp(primals, tangents, *, primitive):
    """Differentiate either bilinear product while skipping symbolic zero terms."""
    a, b, seglen = primals
    tangent_a, tangent_b, _ = tangents
    out = _bind_product(primitive, a, b, seglen)
    tangent_out = ad.Zero(jax.typeof(out))
    if not isinstance(tangent_a, ad.Zero):
        tangent_out = _bind_product(primitive, tangent_a, b, seglen)
    if not isinstance(tangent_b, ad.Zero):
        tangent_out = ad.add_tangents(
            tangent_out, _bind_product(primitive, a, tangent_b, seglen)
        )
    return out, tangent_out


def _transpose(cotangent, a, b, seglen, *, weight_product):
    """Transpose a linearized product using the two closed segmented operations."""
    a_unknown = ad.is_undefined_primal(a)
    b_unknown = ad.is_undefined_primal(b)
    if a_unknown and b_unknown:
        raise ValueError("A bilinear transpose requires one known operand")
    if isinstance(cotangent, ad.Zero):
        return (
            ad.Zero(a.aval) if a_unknown else None,
            ad.Zero(b.aval) if b_unknown else None,
            None,
        )
    if weight_product:
        if a_unknown:
            return (
                _bind_product(_forward_p, b, jnp.swapaxes(cotangent, 1, 2), seglen),
                None,
                None,
            )
        if b_unknown:
            return None, _bind_product(_forward_p, a, cotangent, seglen), None
    else:
        if a_unknown:
            return (
                _bind_product(_forward_p, cotangent, jnp.swapaxes(b, 1, 2), seglen),
                None,
                None,
            )
        if b_unknown:
            return None, _bind_product(_weight_p, a, cotangent, seglen), None
    return None, None, None


def _batch(args, axes, *, primitive):
    """Map independent products sequentially for vmap and Jacobian transforms."""
    mapped = tuple(
        jnp.moveaxis(value, axis, 0)
        for value, axis in zip(args, axes, strict=True)
        if axis is not batching.not_mapped
    )
    if not mapped:
        return _bind_product(primitive, *args), batching.not_mapped

    def apply(values):
        iterator = iter(values)
        operands = tuple(
            next(iterator) if axis is not batching.not_mapped else value
            for value, axis in zip(args, axes, strict=True)
        )
        return _bind_product(primitive, *operands)

    return jax.lax.map(apply, mapped), 0


for _primitive, _is_weight in ((_forward_p, False), (_weight_p, True)):
    ad.primitive_jvps[_primitive] = partial(_bilinear_jvp, primitive=_primitive)
    ad.primitive_transposes[_primitive] = partial(_transpose, weight_product=_is_weight)
    batching.primitive_batchers[_primitive] = partial(_batch, primitive=_primitive)


def segment_mm(a: jax.Array, b: jax.Array, seglen: jax.Array) -> jax.Array:
    """Multiply each contiguous segment of rows by its own matrix.

    For ``offsets = [0, cumsum(seglen)]``, segment ``g`` computes
    ``out[offsets[g]:offsets[g+1]] = a[offsets[g]:offsets[g+1]] @ b[g]``.
    Output rows after ``sum(seglen)`` are zero, including when input padding
    contains NaNs.

    Parameters
    ----------
    a : jax.Array, shape (capacity, K), dtype float32 or bfloat16
        Packed input features on a CUDA device.
    b : jax.Array, shape (R, K, N), dtype matches a
        Weight matrices for ``R`` segments, on the same device as ``a``.
    seglen : jax.Array, shape (R,), dtype int32
        Runtime CUDA segment lengths. Values must be nonnegative and sum
        to at most ``capacity``. Zero-length segments are supported.

    Returns
    -------
    jax.Array, shape (capacity, N), dtype matches a
        Newly allocated contiguous output on the same device as ``a``.

    Raises
    ------
    ValueError
        If input ranks, shapes, dtypes, or concrete device placement disagree.
    RuntimeError
        If concrete inputs are not on CUDA.

    Notes
    -----
    K and N must be positive. Products accumulate in float32 and round to
    the native output dtype. Inputs are read-only. Lengths and prefix sums
    stay on the device; length value constraints are caller preconditions.

    Supports ``jax.jit``, ``jax.lax.scan``, JVP, VJP, and higher-order
    differentiation with respect to either or both floating operands.
    Integer segment lengths are nondifferentiable. Shapes are static per
    compiled signature; lengths may change at runtime. Batched transforms
    map independent products sequentially. Derivative products use the same
    native dtype and FP32 accumulation as the primal operation.
    Execution uses Warp-owned CUDA graph capture through ``jax_callable``.

    For whole-system sharding, call this operation inside ``jax.shard_map``.
    Each device receives its own packed rows, per-system weights, and runtime
    lengths; their sum must fit that device's local row capacity. The caller
    assigns complete systems to devices and owns the mesh and data placement.
    Replicated floating operands receive summed gradients across the relevant
    mesh axes. Operands with unreduced or reduced manual-axis types must be
    converted to ordinary local operands before this call. Automatically
    partitioning globally packed rows or splitting one system across devices
    is not supported.

    For mixed precision with float32 parameters, explicitly cast both operands
    to bfloat16 inside the differentiated function. The operation and its
    derivative products then accumulate in float32 and return bfloat16 values.
    Differentiating through the casts returns gradients in each original
    operand's dtype; float32 parameter gradients retain bfloat16 numerical
    precision. Casting weights before passing them to the differentiated
    function instead makes those bfloat16 arrays the differentiation inputs.

    See Also
    --------
    nvalchemiops.segment_mm.segment_mm : Shared Warp forward operation.
    nvalchemiops.segment_mm.segment_mm_backward : Explicit operand gradients.
    nvalchemiops.torch.segment_mm.segment_mm : Differentiable PyTorch binding.

    Examples
    --------
    On a CUDA device, keep parameters in float32 and compile explicit casts
    with the loss. This batch has one empty segment and one padding row:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from nvalchemiops.jax.segment_mm import segment_mm
    >>> @jax.jit
    ... def loss(a_fp32, weights_fp32, lengths):
    ...     out = segment_mm(
    ...         a_fp32.astype(jnp.bfloat16),
    ...         weights_fp32.astype(jnp.bfloat16),
    ...         lengths,
    ...     )
    ...     return jnp.sum(out.astype(jnp.float32) ** 2)
    >>> loss_and_grad = jax.value_and_grad(loss, argnums=(0, 1))
    >>> with jax.default_device(jax.devices("gpu")[0]):
    ...     a = jnp.ones((6, 4), dtype=jnp.float32)
    ...     b = jnp.ones((3, 4, 2), dtype=jnp.float32)
    ...     lengths = jnp.array([2, 0, 3], dtype=jnp.int32)
    >>> value, (grad_a, grad_b) = loss_and_grad(a, b, lengths)
    >>> grad_a.shape, grad_b.shape
    ((6, 4), (3, 4, 2))

    With a one-dimensional mesh named ``"systems"``, use global
    inputs of shapes ``(devices, capacity, K)``, ``(devices, R, K, N)``, and
    ``(devices, R)``. All devices have equal static capacities, while active
    lengths may differ and change at runtime. ``shard_map`` preserves rank,
    so the mapped body removes and restores the singleton device dimension:

    >>> import numpy as np
    >>> from jax.sharding import Mesh, PartitionSpec as P
    >>> mesh = Mesh(np.asarray(jax.devices("gpu")), ("systems",))
    >>> distributed_segment_mm = jax.jit(
    ...     jax.shard_map(
    ...         lambda a, b, lengths: segment_mm(a[0], b[0], lengths[0])[None],
    ...         mesh=mesh,
    ...         in_specs=(P("systems"), P("systems"), P("systems")),
    ...         out_specs=P("systems"),
    ...         check_vma=True,
    ...     )
    ... )

    """
    _validate_inputs(a, b, seglen)
    return _bind_product(_forward_p, a, b, seglen)

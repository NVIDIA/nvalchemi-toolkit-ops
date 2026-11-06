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

from nvalchemiops.segment_mm import (
    _INDEX_LIMIT,
    _TILE_M,
    _grad_b_launch_config,
    _launch_forward,
    _launch_grad_b,
    _validate_array,
)

__all__ = ["segment_mm"]


@lru_cache(maxsize=4)
def _callables(dtype: str, graph_mode: JaxCallableGraphMode):
    """Build typed FFI callbacks whose buffers are all supplied by XLA."""
    wp_dtype = {"float32": wp.float32, "bfloat16": wp.bfloat16}[dtype]

    def forward(
        a: wp.array2d(dtype=wp_dtype),
        b: wp.array3d(dtype=wp_dtype),
        row_offsets: wp.array(dtype=wp.int32),
        tile_offsets: wp.array(dtype=wp.int32),
        out: wp.array2d(dtype=wp_dtype),
    ):
        for name, array, ndim in (("a", a, 2), ("b", b, 3), ("out", out, 2)):
            _validate_array(array, name, ndim)
        _launch_forward(a, b, row_offsets, tile_offsets, out)

    def weight_product(
        a: wp.array2d(dtype=wp_dtype),
        grad_out: wp.array2d(dtype=wp_dtype),
        row_offsets: wp.array(dtype=wp.int32),
        splits: wp.int32,
        grad_b: wp.array3d(dtype=wp_dtype),
        partials: wp.array3d(dtype=wp.float32),
    ):
        for name, array, ndim in (
            ("a", a, 2),
            ("grad_out", grad_out, 2),
            ("grad_b", grad_b, 3),
        ):
            _validate_array(array, name, ndim)
        _launch_grad_b(
            a,
            row_offsets,
            grad_out,
            grad_b,
            grad_b if splits == 1 else partials,
            int(splits),
        )

    return (
        wp.jax_callable(forward, num_outputs=1, graph_mode=graph_mode),
        wp.jax_callable(weight_product, num_outputs=2, graph_mode=graph_mode),
    )


def _validate_inputs(a, b, seglen):
    """Validate static metadata without reading runtime segment lengths."""
    for name, array, ndim in (("a", a, 2), ("b", b, 3), ("seglen", seglen, 1)):
        if not isinstance(array, (jax.Array, jax.core.Tracer)) or array.ndim != ndim:
            raise ValueError(f"{name} must be a {ndim}D JAX array")
        if array.size > _INDEX_LIMIT:
            raise ValueError(f"{name} array size exceeds the int32 index range")
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
        raise ValueError("All arrays must be on the same single CPU or CUDA device")
    if any(
        device.platform not in ("cpu", "gpu")
        for device_set in devices
        for device in device_set
    ):
        raise RuntimeError("segment_mm requires a CPU or CUDA device")
    if devices and any(device_set != devices[0] for device_set in devices[1:]):
        raise ValueError("All arrays must be on the same device")


def _row_offsets(seglen, capacity):
    """Build XLA-owned bounded prefixes without requiring 64-bit JAX arithmetic."""
    lengths = jnp.clip(seglen, 0, capacity)
    if seglen.shape[0] * capacity <= 2**31 - 1:
        # Clipping each length proves every partial sum fits int32, including
        # malformed inputs. Use XLA's optimized cumulative-sum lowering.
        offsets = jnp.minimum(jnp.cumsum(lengths, dtype=jnp.int32), capacity)
    else:
        # Saturating addition is associative on [0, capacity]. Subtract before
        # adding so even invalid int32 lengths cannot overflow an intermediate.
        offsets = jax.lax.associative_scan(
            lambda left, right: left + jnp.minimum(right, capacity - left), lengths
        )
    return jnp.concatenate((jnp.zeros((1,), dtype=jnp.int32), offsets))


def _forward(a, b, seglen, *, graph_mode):
    """Pass XLA-owned prefixes and output storage to the shared launch leaf."""
    row_offsets = _row_offsets(seglen, a.shape[0])
    lengths = jnp.diff(row_offsets)
    tile_counts = lengths // _TILE_M + (lengths % _TILE_M != 0).astype(jnp.int32)
    tile_offsets = jnp.concatenate(
        (jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(tile_counts, dtype=jnp.int32))
    )
    forward, _ = _callables(str(a.dtype), graph_mode)
    (out,) = forward(
        a,
        b,
        row_offsets,
        tile_offsets,
        output_dims={"out": (a.shape[0], b.shape[2])},
    )
    return out


def _weight_product(a, grad_out, seglen, *, graph_mode):
    """Allocate split-gradient scratch as an internal XLA-owned FFI result."""
    capacity, k = a.shape
    groups, n = seglen.shape[0], grad_out.shape[1]
    splits = 1
    if capacity and groups:
        splits, _ = _grad_b_launch_config(capacity, groups, k, n, a.dtype.itemsize)
    row_offsets = _row_offsets(seglen, capacity)
    _, weight_product = _callables(str(a.dtype), graph_mode)
    grad_b, _ = weight_product(
        a,
        grad_out,
        row_offsets,
        splits,
        output_dims={
            "grad_b": (groups, k, n),
            "partials": (groups * splits if splits > 1 else 0, k, n),
        },
    )
    return grad_b


def _eager_product(implementation, a, b, seglen):
    """Place eager temporaries and select graph mode from the actual operands."""
    device = next(iter(a.devices()))
    graph_mode = (
        JaxCallableGraphMode.NONE
        if device.platform == "cpu"
        else JaxCallableGraphMode.WARP
    )
    with jax.default_device(device):
        return implementation(a, b, seglen, graph_mode=graph_mode)


_forward_p = core.Primitive("nvalchemiops_segment_mm")
_forward_p.def_impl(partial(_eager_product, _forward))
_weight_p = core.Primitive("nvalchemiops_segment_mm_weight_product")
_weight_p.def_impl(partial(_eager_product, _weight_product))


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
for _platform, _graph_mode in (
    ("cpu", JaxCallableGraphMode.NONE),
    ("cuda", JaxCallableGraphMode.WARP),
):
    for _primitive, _implementation in (
        (_forward_p, _forward),
        (_weight_p, _weight_product),
    ):
        mlir.register_lowering(
            _primitive,
            mlir.lower_fun(
                partial(_implementation, graph_mode=_graph_mode), multiple_results=False
            ),
            platform=_platform,
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
        Packed input features on a CPU or CUDA device.
    b : jax.Array, shape (R, K, N), dtype matches a
        Weight matrices for ``R`` segments, on the same device as ``a``.
    seglen : jax.Array, shape (R,), dtype int32
        Runtime device segment lengths. Values must be nonnegative and sum
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
        If concrete inputs are not on CPU or CUDA.

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
    Inputs, outputs, and scratch remain owned by JAX/XLA. CUDA execution uses
    Warp-owned graph capture through ``jax_callable``; CPU callbacks execute
    without graph capture using the same tile kernels.

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

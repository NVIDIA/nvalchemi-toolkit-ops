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

from functools import cache, partial

import jax
import jax.numpy as jnp
import warp as wp
from jax._src import dispatch
from jax.experimental import checkify
from jax.extend import core
from jax.interpreters import ad, batching, mlir
from warp import JaxCallableGraphMode

from nvalchemiops.batch_utils import _segment_chunk_offsets
from nvalchemiops.segment_mm import (
    _INDEX_LIMIT,
    _TILE_M,
    _grad_b_launch_config,
    _launch_forward,
    _launch_grad_a,
    _launch_grad_b,
    _validate_array,
)

__all__ = ["segment_mm"]


# Warp retains registered FFI targets; retain one wrapper per policy so revisiting
# a limit reuses its existing graph cache instead of registering another target.
@cache
def _callables(
    dtype: str, graph_mode: JaxCallableGraphMode, graph_cache_max: int | None
):
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

    def input_product(
        grad_out: wp.array2d(dtype=wp_dtype),
        weights: wp.array3d(dtype=wp_dtype),
        row_offsets: wp.array(dtype=wp.int32),
        tile_offsets: wp.array(dtype=wp.int32),
        grad_a: wp.array2d(dtype=wp_dtype),
    ):
        for name, array, ndim in (
            ("grad_out", grad_out, 2),
            ("weights", weights, 3),
            ("grad_a", grad_a, 2),
        ):
            _validate_array(array, name, ndim)
        b = weights.transpose((0, 2, 1)) if wp_dtype == wp.float32 else weights
        _launch_grad_a(grad_out, b, row_offsets, tile_offsets, grad_a, weights)

    def weight_product(
        a: wp.array2d(dtype=wp_dtype),
        grad_out: wp.array2d(dtype=wp_dtype),
        row_offsets: wp.array(dtype=wp.int32),
        chunk_offsets: wp.array(dtype=wp.int32),
        chunk_rows: wp.int32,
        grad_b: wp.array3d(dtype=wp_dtype),
        partials: wp.array3d(dtype=wp.float32),
        chunk_scratch: wp.array(dtype=wp.int32),
    ):
        for name, array, ndim in (
            ("a", a, 2),
            ("grad_out", grad_out, 2),
            ("grad_b", grad_b, 3),
        ):
            _validate_array(array, name, ndim)
        if chunk_scratch.shape[0]:
            groups = grad_b.shape[0]
            bucket = max(32, 1 << (groups - 1).bit_length())
            wp.launch_tiled(
                _segment_chunk_offsets[bucket],
                dim=(1,),
                inputs=[
                    row_offsets,
                    int(chunk_rows),
                    max(partials.shape[0] // 2, partials.shape[0] - groups),
                    chunk_scratch,
                ],
                block_dim=min(bucket, 128),
                device=a.device,
            )
            chunk_offsets = chunk_scratch
        _launch_grad_b(
            a,
            row_offsets,
            chunk_offsets,
            grad_out,
            grad_b,
            grad_b if chunk_rows >= a.shape[0] else partials,
            int(chunk_rows),
        )

    return (
        wp.jax_callable(
            forward,
            num_outputs=1,
            graph_mode=graph_mode,
            graph_cache_max=graph_cache_max,
        ),
        wp.jax_callable(
            input_product,
            num_outputs=1,
            graph_mode=graph_mode,
            graph_cache_max=graph_cache_max,
        ),
        wp.jax_callable(
            weight_product,
            num_outputs=3,
            graph_mode=graph_mode,
            graph_cache_max=graph_cache_max,
        ),
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


@jax.jit
@checkify.checkify
def _length_error(valid):
    """Return device error state without inspecting the validity predicate."""
    checkify.check(valid, "seglen must be nonnegative and sum to at most row_capacity")


def _row_schedule(seglen, capacity, tile_rows=_TILE_M, skip_first_tile=False):
    """Build XLA-owned row and tile prefixes for forward and input products."""
    row_offsets = _row_offsets(seglen, capacity)
    lengths = jnp.diff(row_offsets)
    minimum_rows = 512 if seglen.shape[0] >= 32 else 256
    if skip_first_tile and minimum_rows < tile_rows < capacity:
        extra_slots = (capacity - 1) // tile_rows
        required = jnp.maximum(row_offsets[-1] - 1, 0) // (extra_slots + 1) + 1
        tile_rows = jnp.maximum(minimum_rows, (required + 31) // 32 * 32)
    tile_counts = lengths // tile_rows + (lengths % tile_rows != 0).astype(jnp.int32)
    if skip_first_tile:
        tile_counts = jnp.maximum(tile_counts - 1, 0)
    tile_offsets = jnp.concatenate(
        (jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(tile_counts, dtype=jnp.int32))
    )
    return row_offsets, tile_offsets


def _forward(a, b, seglen, *, graph_mode, graph_cache_max):
    """Pass XLA-owned prefixes and output storage to the shared launch leaf."""
    row_offsets, tile_offsets = _row_schedule(seglen, a.shape[0])
    forward, _, _ = _callables(str(a.dtype), graph_mode, graph_cache_max)
    (out,) = forward(
        a,
        b,
        row_offsets,
        tile_offsets,
        output_dims={"out": (a.shape[0], b.shape[2])},
    )
    return out


def _input_product(grad_out, b, seglen, *, graph_mode, graph_cache_max):
    """Compute dA with XLA-owned prefixes, output, and weight layout."""
    row_offsets, tile_offsets = _row_schedule(seglen, grad_out.shape[0])
    _, input_product, _ = _callables(str(grad_out.dtype), graph_mode, graph_cache_max)
    # XLA's FP32 transpose fuses layout conversion efficiently; BF16 keeps B.
    weights = jnp.swapaxes(b, 1, 2) if b.dtype == jnp.float32 else b
    (grad_a,) = input_product(
        grad_out,
        weights,
        row_offsets,
        tile_offsets,
        output_dims={"grad_a": (grad_out.shape[0], b.shape[1])},
    )
    return grad_a


def _weight_product(a, grad_out, seglen, *, graph_mode, graph_cache_max):
    """Allocate chunk-gradient scratch as an internal XLA-owned FFI result."""
    capacity, k = a.shape
    groups, n = seglen.shape[0], grad_out.shape[1]
    chunk_rows, scratch_slots = max(256, capacity), 0
    if capacity and groups:
        chunk_rows, scratch_slots = _grad_b_launch_config(
            capacity, groups, k, n, a.dtype.itemsize
        )
    direct = chunk_rows >= capacity
    callback_schedule = not direct and 1 <= groups <= 1024
    if direct or callback_schedule:
        row_offsets = _row_offsets(seglen, capacity)
        chunk_offsets = row_offsets
    else:
        row_offsets, chunk_offsets = _row_schedule(
            seglen, capacity, chunk_rows, skip_first_tile=True
        )
    _, _, weight_product = _callables(str(a.dtype), graph_mode, graph_cache_max)
    grad_b, _, _ = weight_product(
        a,
        grad_out,
        row_offsets,
        chunk_offsets,
        chunk_rows,
        output_dims={
            "grad_b": (groups, k, n),
            "partials": (0 if direct else scratch_slots, k, n),
            "chunk_scratch": (groups + 1 if callback_schedule else 0,),
        },
    )
    return grad_b


_forward_p = core.Primitive("nvalchemiops_segment_mm")
_input_p = core.Primitive("nvalchemiops_segment_mm_input_product")
_weight_p = core.Primitive("nvalchemiops_segment_mm_weight_product")


def _bind_product(primitive, a, b, seglen, **params):
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
    return primitive.bind(*operands, **params)


def _abstract_product(a, b, seglen, *, primitive, graph_cache_max):
    """Preserve per-device variation for all three local segmented products."""
    operands = (a, b, seglen)
    if any(value.mat.unreduced or value.mat.reduced for value in operands):
        raise ValueError("segment_mm requires ordinary local operands inside shard_map")
    if any(any(axis is not None for axis in value.sharding.spec) for value in operands):
        raise ValueError("Use shard_map to pass local segments to segment_mm")
    shape = (
        (seglen.shape[0], a.shape[1], b.shape[1])
        if primitive is _weight_p
        else (a.shape[0], b.shape[1 if primitive is _input_p else 2])
    )
    varying = frozenset().union(*(value.mat.varying for value in operands))
    return a.update(
        shape=shape,
        sharding=a.sharding.update(spec=jax.sharding.PartitionSpec()),
        manual_axis_type=a.mat.update(varying=varying),
    )


for _primitive in (_forward_p, _input_p, _weight_p):
    _primitive.def_impl(partial(dispatch.apply_primitive, _primitive))
    _primitive.def_abstract_eval(partial(_abstract_product, primitive=_primitive))
for _platform, _graph_mode in (
    ("cpu", JaxCallableGraphMode.NONE),
    ("cuda", JaxCallableGraphMode.WARP),
):
    for _primitive, _implementation in (
        (_forward_p, _forward),
        (_input_p, _input_product),
        (_weight_p, _weight_product),
    ):
        mlir.register_lowering(
            _primitive,
            mlir.lower_fun(
                partial(_implementation, graph_mode=_graph_mode), multiple_results=False
            ),
            platform=_platform,
        )


def _bilinear_jvp(primals, tangents, *, primitive, **params):
    """Differentiate any bilinear product while skipping symbolic zero terms."""
    a, b, seglen = primals
    tangent_a, tangent_b, _ = tangents
    out = _bind_product(primitive, a, b, seglen, **params)
    tangent_out = ad.Zero(jax.typeof(out))
    if not isinstance(tangent_a, ad.Zero):
        tangent_out = _bind_product(primitive, tangent_a, b, seglen, **params)
    if not isinstance(tangent_b, ad.Zero):
        tangent_out = ad.add_tangents(
            tangent_out, _bind_product(primitive, a, tangent_b, seglen, **params)
        )
    return out, tangent_out


def _transpose(cotangent, a, b, seglen, *, primitive, **params):
    """Close transposes over F(A, B) = AB, I(G, B) = GB^T, and W(A, G) = A^TG."""
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
    if primitive is _weight_p:
        if a_unknown:
            return (
                _bind_product(_input_p, b, cotangent, seglen, **params),
                None,
                None,
            )
        if b_unknown:
            return None, _bind_product(_forward_p, a, cotangent, seglen, **params), None
    elif primitive is _input_p:
        if a_unknown:
            return _bind_product(_forward_p, cotangent, b, seglen, **params), None, None
        if b_unknown:
            return None, _bind_product(_weight_p, cotangent, a, seglen, **params), None
    else:
        if a_unknown:
            return (
                _bind_product(_input_p, cotangent, b, seglen, **params),
                None,
                None,
            )
        if b_unknown:
            return None, _bind_product(_weight_p, a, cotangent, seglen, **params), None
    return None, None, None


def _batch(args, axes, *, primitive, **params):
    """Map independent products sequentially for vmap and Jacobian transforms."""
    mapped = tuple(
        jnp.moveaxis(value, axis, 0)
        for value, axis in zip(args, axes, strict=True)
        if axis is not None
    )
    if not mapped:
        return _bind_product(primitive, *args, **params), None

    def apply(values):
        iterator = iter(values)
        operands = tuple(
            next(iterator) if axis is not None else value
            for value, axis in zip(args, axes, strict=True)
        )
        return _bind_product(primitive, *operands, **params)

    return jax.lax.map(apply, mapped), 0


for _primitive in (_forward_p, _input_p, _weight_p):
    ad.primitive_jvps[_primitive] = partial(_bilinear_jvp, primitive=_primitive)
    ad.primitive_transposes[_primitive] = partial(_transpose, primitive=_primitive)
    batching.primitive_batchers[_primitive] = partial(_batch, primitive=_primitive)


def segment_mm(
    a: jax.Array,
    b: jax.Array,
    seglen: jax.Array,
    *,
    graph_cache_max: int | None = 256,
    validate: bool = False,
) -> jax.Array | tuple[checkify.Error, jax.Array]:
    """Multiply each contiguous segment of rows by its own matrix.

    For ``offsets = [0, cumsum(seglen)]``, segment ``g`` computes
    ``out[offsets[g]:offsets[g+1]] = a[offsets[g]:offsets[g+1]] @ b[g]``.
    Output rows after ``sum(seglen)`` are zero, including when input padding
    contains NaNs.

    ``row_capacity`` is the allocated row count, including padding, and
    ``num_segments`` is the number of segments. ``in_channels`` and
    ``out_channels`` are the input and output feature counts.

    Parameters
    ----------
    a : jax.Array, shape (row_capacity, in_channels), dtype float32 or bfloat16
        Packed input features on a CPU or CUDA device.
    b : jax.Array, shape (num_segments, in_channels, out_channels), dtype matches a
        One weight matrix per segment, on the same device as ``a``.
    seglen : jax.Array, shape (num_segments,), dtype int32
        Runtime device segment lengths. Values must be nonnegative and sum
        to at most ``row_capacity``. Zero-length segments are supported.
    graph_cache_max : int or None, optional
        Maximum CUDA graphs per product and dtype for this cache policy,
        shared across callsites and devices. Defaults to 256; must be a
        positive integer or ``None`` for an unlimited cache. Close over this
        value or mark it static under ``jax.jit``. Derivative products inherit
        the limit. Ignored on CPU.
    validate : bool, optional
        Check runtime length values and return ``(error, out)`` instead of
        ``out``. Defaults to ``False``. The error is a
        ``jax.experimental.checkify.Error`` with device-resident state;
        this operation never inspects it or synchronizes to report errors.
        Close over this flag or mark it static under ``jax.jit``.

    Returns
    -------
    jax.Array or tuple of (checkify.Error, jax.Array)
        Newly allocated contiguous output on the same device as ``a``.
        The output has shape ``(row_capacity, out_channels)`` and matches
        ``a.dtype``. With ``validate=True``, an error value precedes the
        output. No external ``checkify`` transformation is required.

    Raises
    ------
    ValueError
        If input metadata disagree, ``graph_cache_max`` is invalid, or
        ``validate`` is not a Python bool.
    RuntimeError
        If concrete inputs are not on CPU or CUDA.

    Notes
    -----
    ``in_channels`` and ``out_channels`` must be positive. Products accumulate
    in float32 and round to the native output dtype. Inputs are read-only.
    Lengths and prefix sums stay on the device. By default, length value
    constraints are caller preconditions. With ``validate=True``, checks
    report negative lengths or a total above the row capacity, including
    inputs whose int32 sum would overflow. Errors do not stop the product
    from executing; outputs for invalid lengths must not be used.
    Inspecting ``error.get()`` or calling ``error.throw()`` explicitly reads
    error state on the host and can synchronize. Preserve errors as auxiliary
    outputs with ``has_aux=True`` when differentiating a checked loss.

    Supports ``jax.jit``, ``jax.lax.scan``, JVP, VJP, and higher-order
    differentiation with respect to either or both floating operands.
    Integer segment lengths are nondifferentiable. Shapes are static per
    compiled signature; lengths may change at runtime. Batched transforms
    map independent products sequentially. Derivative products use the same
    native dtype and FP32 accumulation as the primal operation.
    Inputs, outputs, and scratch remain owned by JAX/XLA. CUDA execution uses
    Warp-owned graph capture through ``jax_callable``; CPU callbacks execute
    without graph capture using the same tile kernels.
    Graph reuse requires matching callsites and input/output buffer addresses.
    A larger cache can avoid recapture when many such combinations recur;
    fresh addresses still require capture. ``None`` permits graph memory to
    grow without a bound. Different cache limits use independent callables.

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

    With ``num_devices`` devices in a one-dimensional mesh named ``"systems"``,
    use global inputs of shapes ``(num_devices, row_capacity, in_channels)``,
    ``(num_devices, num_segments, in_channels, out_channels)``, and
    ``(num_devices, num_segments)``. All devices have equal static
    ``row_capacity``, while active lengths may differ and change at runtime.
    ``shard_map`` preserves rank, so the mapped body removes and restores the
    singleton device dimension:

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
    if not isinstance(validate, bool):
        raise ValueError("validate must be a Python bool; keep it static under jax.jit")
    if graph_cache_max is not None and (
        isinstance(graph_cache_max, bool)
        or not isinstance(graph_cache_max, int)
        or graph_cache_max <= 0
    ):
        raise ValueError("graph_cache_max must be a positive integer or None")
    _validate_inputs(a, b, seglen)
    if validate:
        offsets = _row_offsets(seglen, a.shape[0])
        error, _ = _length_error(jnp.all(jnp.diff(offsets) == seglen))
    out = _bind_product(_forward_p, a, b, seglen, graph_cache_max=graph_cache_max)
    return (error, out) if validate else out

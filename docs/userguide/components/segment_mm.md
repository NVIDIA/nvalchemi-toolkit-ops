<!-- markdownlint-disable MD013 -->

(segment_mm_userguide)=

# Segmented Matrix Multiplication

`segment_mm(a, b, seglen)` multiplies each group of rows in `a` by its own
matrix in `b`. A group is called a **segment**. For example, each segment can
hold the atom features for one system, with one weight matrix per system.

Segments can have different numbers of rows, including zero. All segments
share the same input and output channel counts. FP32 and BF16 are supported
on CPU and CUDA.

## How rows are grouped

Rows from the same segment must be next to each other in `a`, with segments
stored in order. `seglen` gives the number of rows in each segment.

For `a.shape = (8, 4)`, `b.shape = (3, 4, 5)`, and `seglen = [3, 0, 2]`:

| Segment | Input rows | Product |
| --- | --- | --- |
| 0 | First 3 rows | `out[0:3] = a[0:3] @ b[0]` |
| 1 | No rows | Empty segment; no product needed |
| 2 | Next 2 rows | `out[3:5] = a[3:5] @ b[2]` |

The output has shape `(8, 5)`. Only five rows are active. The last three rows
are unused capacity, so `out[5:8]` is zero.

## Inputs and output

| Array | Shape | Meaning |
| --- | --- | --- |
| `a` | `(row_capacity, in_channels)` | Feature rows for all segments |
| `b` | `(num_segments, in_channels, out_channels)` | One weight matrix per segment |
| `seglen` | `(num_segments,)` | Number of rows in each segment; `int32`, or `int64` in PyTorch |
| Output | `(row_capacity, out_channels)` | Products followed by zero rows |

`row_capacity` is the total number of allocated rows, including padding.
`sum(seglen)` is the number of active rows. Lengths must be nonnegative and
their sum must not exceed `row_capacity`. Input and output channel counts
must be positive.

All inputs belong on the same device. Both floating inputs must
have the same dtype, except when PyTorch CUDA BF16 autocast performs the casts.

Unused feature rows are ignored, even if they contain NaNs. Their outputs and
gradients are zero. An empty segment's weight gradient is also zero. An empty
segment list or zero row capacity is supported.

## Quick start

Both examples compute the same products and gradients for the batch above.

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
import torch
from nvalchemiops.torch.segment_mm import segment_mm

a = torch.ones(8, 4, device="cuda", requires_grad=True)
b = torch.ones(3, 4, 5, device="cuda", requires_grad=True)
lengths = torch.tensor([3, 0, 2], device="cuda", dtype=torch.int32)

compiled_mm = torch.compile(segment_mm, fullgraph=True)
out = compiled_mm(a, b, lengths)  # Shape (8, 5); rows 5:8 are zero.
out.square().sum().backward()    # Gradients for both a and b.
```

Use `device="cpu"` for CPU execution. Calling `segment_mm` directly supports
eager autograd as well.

:::

:::{tab-item} JAX
:sync: jax

```python
import jax
import jax.numpy as jnp
from nvalchemiops.jax.segment_mm import segment_mm

a = jnp.ones((8, 4), dtype=jnp.float32)
b = jnp.ones((3, 4, 5), dtype=jnp.float32)
lengths = jnp.array([3, 0, 2], dtype=jnp.int32)

compiled_mm = jax.jit(segment_mm)
out = compiled_mm(a, b, lengths)  # Shape (8, 5); rows 5:8 are zero.

def loss(a, b, lengths):
    return jnp.sum(segment_mm(a, b, lengths) ** 2)

grad_a, grad_b = jax.jit(jax.grad(loss, argnums=(0, 1)))(a, b, lengths)
```

JAX uses its default device. Use `jax.default_device` or `jax.device_put` to
choose a CPU or CUDA device explicitly.

:::

::::

### Changing segment sizes

The same compiled function can accept new lengths, such as `[2, 1, 3]`,
without recompiling when array shapes and dtypes stay fixed. That batch has
six active rows and two padding rows. Keep the allocated row capacity,
segment count, and channel counts fixed; change the values in `seglen`.

Lengths describe contiguous groups, rather than the per-row indices used by
[segment reductions](segment_ops.md). They must be nonnegative and sum to at
most the row capacity.

Both frontends check input shapes, dtypes, and devices. By default, they
rely on the caller to supply valid length values. Set `validate=True` to
check length values as well.

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

PyTorch accepts int32 or int64 lengths; all length values must fit int32.

```python
out = segment_mm(a, b, lengths, validate=True)
```

Invalid lengths raise `ValueError` on CPU or trigger a device assertion on
CUDA. Validation also works with `torch.compile` and CUDA graph capture.

:::

:::{tab-item} JAX
:sync: jax

JAX requires int32 lengths. With `validate=True`, it returns `(error, out)`
instead of just `out`.

```python
@jax.jit
def checked_mm(a, b, lengths):
    return segment_mm(a, b, lengths, validate=True)

error, out = checked_mm(a, b, lengths)
```

Keep `validate` static when compiling. The returned `error` is a JAX
`checkify.Error`; only use `out` when validation succeeds. For a differentiated
loss, return `(loss, error)` and use `has_aux=True` to keep the error alongside
the gradients.

:::

::::

For direct Warp calls, pass a caller-owned status buffer to enable checking.
With Warp arrays `a`, `b`, `lengths`, and a preallocated `out`:

```python
import warp as wp
from nvalchemiops.segment_mm import segment_mm as warp_segment_mm

status = wp.empty(1, dtype=wp.int32, device=a.device)
warp_segment_mm(a, b, lengths, out, validation=status)
```

Each call sets `status[0]` to `1` for valid lengths or `0` for invalid lengths.
Only use the output when the status is `1`. The same option is available on
`segment_mm_backward` and `segment_mm_backward_b`. Warp lengths must be int32;
omitting `validation` leaves value checking disabled.

## Gradients and mixed precision

You can differentiate with respect to both the features `a` and weights `b`.
Both frontends support forward derivatives (JVPs) and higher derivatives.
Integer lengths are not differentiable. JAX also supports `jax.lax.scan`.

Products and gradient products accumulate in FP32 and return the operand
dtype. BF16 inputs therefore produce BF16 outputs and derivative products.
To keep model parameters in FP32 while computing in BF16:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = segment_mm(a, b, lengths)
    loss = out.float().square().sum()
loss.backward()
```

CUDA BF16 autocast casts both operands. CUDA FP16 autocast is unsupported.
CPU execution uses the explicit input dtype, even inside CPU autocast.

:::

:::{tab-item} JAX
:sync: jax

```python
def loss(a, b, lengths):
    out = segment_mm(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16), lengths)
    return jnp.sum(out.astype(jnp.float32) ** 2)

grad_a, grad_b = jax.jit(jax.grad(loss, argnums=(0, 1)))(a, b, lengths)
```

Cast inside the function being differentiated so gradients return to the
original FP32 inputs.

:::

::::

In both cases, FP32 parameter gradients retain the numerical precision of
the BF16 products. Casting back to FP32 does not recover lost precision.

For compiled higher derivatives in PyTorch, compile the complete
`torch.func.grad` or `torch.func.jvp` computation. A second backward through
a separately compiled forward is not supported by PyTorch 2.10 AOTAutograd.

## Multiple GPUs

Assign complete systems to each GPU, then call `segment_mm` on each device's
local features, weights, and lengths. Each local length sum must fit that
device's row capacity. The caller handles system assignment and data placement.

- **JAX:** call inside `jax.shard_map`. Replicated operands receive gradients
  summed across the relevant mesh axes. See the {doc}`JAX API example </modules/jax/segment_mm>`
  for the mesh and array layout.
- **PyTorch:** call inside a `DistributedDataParallel` model. DDP reduces model
  parameter gradients. For force-loss training, compute forces inside the
  model's `forward` with `create_graph=True`, then call `loss.backward()`
  outside the DDP wrapper.

Splitting a single system across GPUs and automatically redistributing packed
rows are outside this interface.

## CUDA graph reuse

JAX reuses CUDA graphs when callsites and input/output buffer addresses repeat.
Each dtype has separate caches for the forward operation and both gradient
operations, holding up to 256 graphs each. If many buffer combinations recur,
set a larger static limit,
for example
`segment_mm(a, b, lengths, graph_cache_max=512)`. `None` removes the limit and
allows graph memory to grow. Fresh buffer addresses still require capture.
The limit also applies to derivative products and is ignored on CPU.

For explicit PyTorch or Warp graph capture, warm up the kernels first and
disable Warp's `verify_cuda` debugging option. Keep the Warp module loaded
and unchanged until captured graphs are discarded. Recapture after changing
module options or reloading the module.

## Example and API reference

The {doc}`worked example </examples/03_segment_mm>` checks forward results,
gradients, changing lengths, and BF16 using both frontends.

- {doc}`PyTorch API </modules/torch/segment_mm>`
- {doc}`JAX API </modules/jax/segment_mm>`
- {doc}`Warp API </modules/warp/segment_mm>`: caller-provided output buffers and
  explicit gradient functions.

<!-- markdownlint-disable MD013 -->

(segment_mm_userguide)=

# Segmented Matrix Multiplication

`segment_mm(a, b, seglen)` multiplies each contiguous segment of packed rows
by its own matrix. Use it when systems have different row counts but share
input and output channel sizes.

Segmented matrix multiplication requires CUDA.

## Packed-Row Contract

| Input | Shape | Meaning |
| --- | --- | --- |
| `a` | `(capacity, K)` | Feature rows packed in segment order |
| `b` | `(R, K, N)` | One matrix per segment |
| `seglen` | `(R,)` | Runtime CUDA `int32` row counts |
| Output | `(capacity, N)` | Packed products followed by zero padding |

For `offsets = [0, cumsum(seglen)]`, segment `g` computes
`a[offsets[g]:offsets[g + 1]] @ b[g]`. These are **lengths**, unlike the
per-row `idx` arrays used by [segment reductions](segment_ops.md).
The caller must provide nonnegative lengths whose sum is at most `capacity`;
values stay on the GPU and are not validated on the host. As a bounds-safety
fallback, negative lengths count as zero and segment ranges are truncated at
`capacity`. This fallback does not report an input error. All operands of a
local call must be on the same CUDA device, and `K` and `N` must be positive.

Zero-length segments, an empty segment list, and zero row capacity are
supported. Unused output rows are zero even when input padding contains NaNs.
The derivative with respect to padded feature rows and weights of empty
segments is zero. Shapes are static for a compiled signature; lengths may
change between executions while those shapes stay fixed.

Both floating inputs use the same dtype: FP32 or BF16. Products and derivative
products accumulate in FP32 and return the native operand dtype. BF16 outputs
and gradients therefore retain BF16 precision. The Warp API takes an explicit
contiguous output buffer; the framework wrappers allocate the output.

## PyTorch

```python
import torch

from nvalchemiops.torch.segment_mm import segment_mm

a = torch.randn(8, 4, device="cuda", requires_grad=True)
b = torch.randn(3, 4, 5, device="cuda", requires_grad=True)
lengths = torch.tensor([3, 0, 2], device="cuda", dtype=torch.int32)

with torch.autocast("cuda", dtype=torch.bfloat16):
    out = segment_mm(a, b, lengths)  # BF16, shape (8, 5); rows 5: are zero
    loss = out.float().square().sum()
loss.backward()  # a.grad and b.grad are FP32, through the BF16 casts
```

CUDA BF16 autocast converts both operands before the operation. Outside
autocast they must already have the same supported dtype; CUDA FP16 autocast
is unsupported. The FP32 parameter gradients in this example retain the
numerical precision of the BF16 derivative products.

Eager autograd, `torch.compile(fullgraph=True)`, forward AD, `torch.func.jvp`,
and higher derivatives are supported. For compiled higher derivatives,
compile the complete `torch.func.grad` or `torch.func.jvp` computation:
PyTorch 2.10 AOTAutograd does not support a second backward through a
separately compiled forward. CUDA graph capture requires kernel warmup and
Warp's `verify_cuda` debugging option to be disabled. Keep the SegMM Warp
module loaded and unchanged until all graph replays finish and the graphs are
discarded. Warm up and recapture after unloading, reloading, or changing module
options.

### DDP and Force-Loss Training

With `DistributedDataParallel`, each rank owns complete systems and passes
ordinary local CUDA tensors to `segment_mm`. DDP synchronizes replicated model
parameter gradients. The caller owns process-group initialization, device
selection, system assignment, packing, and loss weighting across ranks.

For a force loss, compute the position derivative **inside the module's
`forward`**, with `create_graph=True`, and use ordinary `loss.backward()`
outside the DDP wrapper. The following pattern assumes an initialized process
group, a current CUDA device selected for each process, and these local inputs:

| Variable | Shape or type | Meaning |
| --- | --- | --- |
| `local_rank` | `int` | CUDA device index within the process's node |
| `weights_fp32` | `(R, K, N)` | FP32 weights, initialized identically on each rank |
| `positions` | `(capacity, K)` | FP32 local input features used to form the energy |
| `target_forces` | `(capacity, K)` | FP32 derivative targets, zero on padded rows |
| `lengths` | `(R,)` | Local segment lengths as CUDA `int32` |

All tensors must be on the process's CUDA device. This example uses a toy
energy to illustrate differentiating through the input gradient.

```python
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from nvalchemiops.torch.segment_mm import segment_mm


class ForceModel(torch.nn.Module):
    def __init__(self, weights_fp32):
        super().__init__()
        self.weights = torch.nn.Parameter(weights_fp32)

    def forward(self, positions, lengths):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = segment_mm(torch.sin(positions), self.weights, lengths)
            energy = out.float().square().sum()
        return -torch.autograd.grad(energy, positions, create_graph=True)[0]


device = torch.device("cuda", local_rank)
model = DDP(ForceModel(weights_fp32).to(device), device_ids=[local_rank])
forces = model(positions.requires_grad_(), lengths)
loss = (forces - target_forces).square().sum()
loss.backward()
```

This illustrates eager DDP force training. Keep the outer parameter-gradient
accumulation in `backward()` so DDP's reduction hooks run.

## JAX

`jax.jit`, `jax.lax.scan`, JVP, VJP, and higher-order differentiation are
supported for either or both floating operands. Integer lengths are
nondifferentiable. Batched transforms map independent products sequentially.
For mixed precision, cast both operands inside the differentiated function:

```python
import jax
import jax.numpy as jnp

from nvalchemiops.jax.segment_mm import segment_mm


@jax.jit
def loss(a_fp32, weights_fp32, lengths):
    out = segment_mm(
        a_fp32.astype(jnp.bfloat16),
        weights_fp32.astype(jnp.bfloat16),
        lengths,
    )
    return jnp.sum(out.astype(jnp.float32) ** 2)


gpu = jax.devices("gpu")[0]
with jax.default_device(gpu):
    a = jnp.ones((8, 4), dtype=jnp.float32)
    b = jnp.ones((3, 4, 5), dtype=jnp.float32)
    lengths = jnp.array([3, 0, 2], dtype=jnp.int32)

loss_and_grad = jax.value_and_grad(loss, argnums=(0, 1))
value, (grad_a, grad_b) = loss_and_grad(a, b, lengths)
# grad_a.shape == (8, 4); grad_b.shape == (3, 4, 5).
# Both gradients are FP32; padded rows and empty segments have zero gradients.
```

The casts return gradients to FP32 parameters in their original dtype, with
BF16 numerical precision. Casting weights before passing them to
`loss_and_grad` instead makes those BF16 arrays the differentiation inputs.

### Whole-System Sharding

Call `segment_mm` inside `jax.shard_map` with complete local systems. For a
one-dimensional mesh named `"systems"`, use global shapes
`(devices, capacity, K)`, `(devices, R, K, N)`, and `(devices, R)`:

```python
import jax
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from nvalchemiops.jax.segment_mm import segment_mm

mesh = Mesh(np.array(jax.devices("gpu")), ("systems",))

distributed_segment_mm = jax.jit(
    jax.shard_map(
        lambda a, b, lengths: segment_mm(a[0], b[0], lengths[0])[None],
        mesh=mesh,
        in_specs=(P("systems"), P("systems"), P("systems")),
        out_specs=P("systems"),
        check_vma=True,
    )
)
```

`shard_map` preserves rank, so the mapped body removes and restores the
singleton device dimension. Devices have equal static capacities; local
lengths may differ and must fit each device's capacity. Replicated floating
operands receive summed gradients across the relevant mesh axes. Reduced or
unreduced manual-axis operands must first be converted to ordinary local
operands. The caller owns mesh construction and data placement. Automatic
redistribution of globally packed rows, DTensor integration, and splitting a
system across devices are outside this interface.

## API Reference

- {doc}`PyTorch </modules/torch/segment_mm>`
- {doc}`JAX </modules/jax/segment_mm>`
- {doc}`Warp forward and explicit gradients </modules/warp/segment_mm>`

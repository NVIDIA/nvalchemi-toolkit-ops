<!-- markdownlint-disable MD013 -->

(segment_ops_userguide)=

# Segment Operations

Segment operations reduce or transform per-element data grouped by a per-element
segment index. They are the workhorses for "reduce by category" patterns — sum
of edge messages per node in a GNN, per-body force accumulation in a particle
simulation, or per-cluster normalization. ALCHEMI Toolkit-Ops provides
GPU-accelerated forward and first/second-order backward kernels via
[NVIDIA Warp](https://nvidia.github.io/warp/), with bindings for both PyTorch
and JAX.

## Quick Start

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
import torch
from nvalchemiops.torch.segment_ops import segmented_sum

x   = torch.tensor([1., 2., 3., 4., 5., 6.], device="cuda", requires_grad=True)
idx = torch.tensor([0, 0, 1, 1, 1, 2], device="cuda", dtype=torch.int32)

out = segmented_sum(x, idx, num_segments=3)
# out = tensor([3., 12., 6.], device='cuda:0', grad_fn=<SegmentedSumBackward>)

out.sum().backward()
# x.grad = tensor([1., 1., 1., 1., 1., 1.], device='cuda:0')
```

:::

:::{tab-item} JAX
:sync: jax

```python
import jax.numpy as jnp
from nvalchemiops.jax.segment_ops import segmented_sum

x   = jnp.array([1., 2., 3., 4., 5., 6.])
idx = jnp.array([0, 0, 1, 1, 1, 2], dtype=jnp.int32)

out = segmented_sum(x, idx, num_segments=3)
# DeviceArray([3., 12., 6.], dtype=float32)
```

:::

::::

The full set of operations covers sums, dot products, scaled-broadcasts, mean,
RMS norm, max norm, and matrix-vector products — all first- and second-order
differentiable. See the {doc}`Torch </modules/torch/segment_ops>` and
{doc}`JAX </modules/jax/segment_ops>` segment-op API reference pages for the
complete public signatures.

(segment_ops_cuda_graphs)=

## Accelerating Hot Loops with CUDA Graphs

The segment-op kernels themselves are small and bandwidth-bound. For low-N
problem sizes (≲100k elements), the wall time is dominated by **host-side
launch overhead**: Python dispatch, dtype-keyed overload lookup, parameter
packing, and the CUDA driver's `cuLaunchKernel` call. This shows up as a
2-3× gap between our launchers and PyTorch's fused C++ kernels at small N,
even though our kernels themselves are equal or faster.

For workloads that call segment ops repeatedly with the same shapes — training
steps, MD timesteps, Monte Carlo iterations — the right tool is a
[CUDA graph](https://developer.nvidia.com/blog/cuda-graphs/). Warp exposes
this via [`wp.ScopedCapture`](https://nvidia.github.io/warp/_generated/warp.ScopedCapture.html):
record the op chain once into a graph node, then replay it on every iteration
with a single submission. All the per-call host work happens at capture time
instead of at every replay.

### Speedup

On an RTX PRO 6000 (Blackwell) at N=10k, M=1000, the difference is:

| Op                      | Eager                      | **Graph replay**           |
|-------------------------|----------------------------|----------------------------|
| `segmented_sum` bwd     | 0.84× vs torch (we lose)   | **1.50× vs torch**         |
| `segmented_dot` bwd     | 1.13×                      | **3.53×**                  |
| `segmented_rms_norm` dbl| 1.45×                      | **8.58×**                  |
| `segmented_matvec` dbl  | 1.68×                      | **5.81×**                  |

The pattern: at small N the host overhead is most of the eager time, so the
graph replay claws back the biggest fraction. At large N the kernel work
dominates and the relative win is smaller (but the absolute work is also where
our launchers already beat torch by 10-25×).

### What About `torch.compile`?

A natural first question is whether ``torch.compile(segmented_sum,
fullgraph=True)`` captures the public wrappers. In this release, it does:
the Torch segment ops are registered as custom op chains, so TorchDynamo sees
each public wrapper as an opaque graph node. Eager calls still validate
``idx`` on the host, including range checks, while compiled calls skip the
range check under ``torch.compiler.is_compiling()`` to avoid a data-dependent
host sync. Pass pre-validated segment indices when compiling with
``fullgraph=True``.

``mode="reduce-overhead"`` can reduce Torch's own launch overhead, but the
speedups reported in the table above come from explicit ``wp.ScopedCapture``
around the raw launchers. Use ``torch.compile`` when the segment op is part
of a larger compiled PyTorch model; use ``wp.ScopedCapture`` when repeatedly
replaying the same fixed-shape Warp launcher sequence.

### Minimal Pattern

```python
import warp as wp
from nvalchemiops.segment_ops_backward import segmented_sum_backward

wp_device = wp.get_device("cuda:0")

# 1. Pre-warm the eager path so JIT compilation and module loading happen
#    BEFORE capture.  If you skip this, the graph records the first-call
#    compile work and replay is much slower.
for _ in range(3):
    segmented_sum_backward(g_out, idx, grad_x)
wp.synchronize_device(wp_device)

# 2. Capture the op chain into a graph.  Every kernel launch and memset
#    inside the with-block is recorded, not executed.
with wp.ScopedCapture(device=wp_device) as cap:
    segmented_sum_backward(g_out, idx, grad_x)

# 3. Replay the graph from the hot loop.  One submission, no Python dispatch.
for step in range(num_steps):
    wp.capture_launch(cap.graph)
    # ... other work that doesn't need to be captured ...
```

The same pattern works for any combination of segment ops — capture a whole
fused chain (forward pass, backward, gradient update) into one graph if the
shapes are stable.

### When to Use It

CUDA graphs win when **the same shape/dtype op chain is replayed many times**.
The breakeven is roughly 100 replays: capture itself takes a few hundred
microseconds, so a single-shot use isn't worth it.

Best fits:

- **Training loops** — every minibatch runs the same forward + backward shape.
- **MD/simulation steps** — atom count and segment partitioning are constant
  across steps.
- **Inner loops of iterative solvers** — fixed-point or Krylov iterations on
  stable inputs.

```{warning}
**All shapes, dtypes, and tensor identities must be stable across replays.**
A captured graph hard-codes the pointer addresses of every input and output,
plus the kernel launch dimensions.  Re-allocating an input tensor or changing
the segment count invalidates the graph — you'll need to re-capture.

If your shapes change frequently, either bucket inputs to fixed sizes
(zero-padding) or accept the eager-path cost.
```

```{warning}
**The op chain must not branch on device state.**  CUDA graphs record one
deterministic sequence of kernel launches.  If the captured Python code has
`if x.sum() > 0:`-style branches, the graph only records the path taken at
capture time.  Subsequent replays will execute that same path regardless of
what the data says.
```

### Capturing Across Backward Passes (Torch Autograd)

Wrapping {class}`torch.autograd.Function.apply` calls in a graph requires
care: autograd hooks and saved-tensor management are host-side state that
doesn't capture cleanly.  Two patterns work:

1. **Capture only the forward**, run the backward eagerly.  Works well when
   you want graph speedup for inference loops.
2. **Capture the launcher calls directly** (e.g. ``segmented_sum_backward``)
   bypassing the autograd wrappers entirely.  This is what the benchmark
   harness does — it's the shortest path to the speedup numbers above.

For end-to-end graph capture of an entire training step, PyTorch's
{class}`torch.cuda.CUDAGraph` and the higher-level ``torch.cuda.graph()``
context manager interoperate with Warp launches as long as both sides use the
same CUDA stream.  Hand-off mechanics are documented in the
[PyTorch CUDA graphs guide](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).

### Caveats and Gotchas

- **Internal allocations**: Some second-order backward launchers (e.g.
  ``segmented_rms_norm_double_backward``) allocate small scratch
  buffers via {func}`warp.zeros`.  These are capturable because Warp routes
  through CUDA's stream-ordered memory pool — but the allocator itself is the
  only guaranteed-graph-safe path.  Don't add per-call ``torch.zeros``
  allocations inside the capture; route through ``warp.zeros`` or pre-allocate
  outside.
- **Stream binding**: ``ScopedCapture`` captures on Warp's own device stream.
  If your surrounding code uses a torch stream and times with
  ``torch.cuda.Event``, the final ``torch.cuda.synchronize()`` is what makes
  the timing reflect actual graph completion.
- **First-call overhead**: Module loading and JIT compilation happen lazily on
  the first kernel launch.  Always run the eager path 2-3 times before
  capture, then ``wp.synchronize_device(device)``, then capture.  Otherwise
  the graph bakes in compile-time work and replay is slow.

### Where to Look Next

- ``benchmarks/segment_ops/benchmark_segment_ops.py`` ships a working
  ``_bench_cuda_graph`` helper that captures any segment op and times the
  replay against the eager path — a useful template for adapting the pattern
  to your own workload.
- The benchmark CSV's ``warp_graph_median_ms`` and ``graph_speedup`` columns
  show the per-shape gain across the full op set.
- [Warp's ScopedCapture docs](https://nvidia.github.io/warp/modules/runtime.html#warp.ScopedCapture)
  for the underlying API, including ``capture_save``/``capture_load`` for
  graphs that survive across processes.

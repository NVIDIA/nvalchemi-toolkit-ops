(style-guide)=

# Kernel Style Guide

## Introduction

This style guide establishes conventions for developing NVIDIA Warp GPU/CPU kernels in `nvalchemiops`.
The intention is to ensure consistency in naming, design patterns, unit testing, and performance
evaluation for both human and agentic developers.

## Naming Conventions

Consult this section before naming variables to ensure consistency across modules. Try
and reuse one of the names below if they are semantically identical to what you are developing
for; in the event that they are different enough, consider using these as inspiration and
make a best effort to align to these conventions.

### Batching

| Variable | Shape | Dtype | Description |
|----------|-------|-------|-------------|
| `batch_idx` | `[total_atoms]` | `int32` | System index for each atom |
| `batch_ptr` | `[num_systems+1]` | `int32` | Cumulative atom counts |

### Neighbor Lists

| Variable | Shape | Dtype | Description |
|----------|-------|-------|-------------|
| `neighbor_matrix` | `[total_atoms, max_neighbors]` | `int32` | Neighbor indices |
| `neighbor_shift_matrix` | `[total_atoms, max_neighbors, 3]` | `int32` | Integer shifts |
| `cartesian_shifts` | `[total_atoms, max_neighbors, 3]` | `vec3*` | Cartesian shifts |
| `num_neighbors` | `[total_atoms]` | `int32` | Neighbor count |
| `neighbor_list` | `[2, num_pairs]` | `int32` | COO format |
| `max_neighbors` | scalar | `int` | Max neighbors |
| `fill_value` | scalar | `int` | Padding value |
| `cutoff`, `cutoff_sq` | scalar | `float*` | Distance cutoff |

### Geometry

| Variable | Shape | Dtype | Description |
|----------|-------|-------|-------------|
| `positions` | `[total_atoms, 3]` | `vec3*` | Atomic Cartesian coordinates |
| `cell` | `[num_systems, 3, 3]` or `[3, 3]` | `mat33*` | Lattice vectors (row format) |
| `pbc` | `[num_systems, 3]` or `[3]` | `bool` | Periodic boundary flags |
| `numbers` | `[total_atoms]` | `int32` | Atomic numbers (Z) |

**Units**: Document in docstrings; if relevant to correctness then Bohr or
Angstrom must be specified. In instances where units just have to be
consistent (e.g. neighbor lists) then that should be stated.

### Thread Indices

| Variable | Type | Description |
|----------|------|-------------|
| `tid` | `int` | Single thread ID from `wp.tid()` (1D launch) |
| `tid_i`, `tid_j` | `int` | Thread IDs from `wp.tid()` (N-D launch) |
| `atom_i`, `atom_j` | `int` | Semantic atom indices (use for clarity) |
| `i`, `j`, `k` | `int` | Generic loop indices |

**Convention**: Use `tid` for thread IDs; use `atom_i`/`atom_j` for
semantic clarity; i.e. if the threads map onto atom indices, then use
`atom_i`/`atom_j` over `tid_i`. See
`nvalchemiops/interactions/dispersion/dftd3.py` for examples.

### Warp Array Objects

The preferred way of instantiating `warp` arrays is to convert from PyTorch
tensors, rather than using Warp constructor methods like `wp.zeros`:

```python
output = torch.zeros(..., device=..., dtype=torch.float32)
# optionally, `return_ctype=True` when it makes sense for performance
output_wp = wp.from_torch(output)
```

Updates to `output_wp` will reflect in the PyTorch tensor, `output` and
eliminates the need for re-converting with overhead, e.g.:

```python
# anti-pattern
output_wp = wp.zeros(..., device=...)
# run kernel...
output = wp.to_torch(output_wp)
```

Some additional tips for performance:

- Use appropriate layouts for PyTorch tensors before constructing Warp
  references, including `torch.Tensor.contiguous()` and
  `torch.Tensor.to(memory_format=torch.channels_last)`. See
  [this blog post](https://pytorch.org/blog/tensor-memory-format-matters/)
  for details.
- Use Warp vector and matrix datatypes when possible, e.g.
  `wp.array(..., dtype=wp.vec3f)` for atomic coordinates, **instead of**
  `wp.array(..., dtype=wp.float32)`. This encourages optimal memory access
  patterns.
- When `dtype` s are known ahead of time, or preferably if wrappers can be
  written in such a way that they are known ahead of time, reduce Python
  overhead with `wp.from_torch(..., return_ctype=True)`, which avoids the
  need for a `wp.array` Python object altogether.
- Consider handling gradients manually, i.e. kernels for backward passes,
  when possible to do so. Ensure tensors are detached from the PyTorch
  computational graph to avoid duplication.

### Warp Array Suffix

**Suggested** (not mandatory): Use `_wp` suffix when converting PyTorch tensors
to Warp arrays in wrapper functions.

```python
positions_wp = wp.from_torch(positions.contiguous(), dtype=wp.vec3f)
numbers_wp = wp.from_torch(numbers.contiguous(), dtype=wp.int32)
```

**When to use**: In PyTorch wrappers with both tensor types in scope.
**Skip**: Inside pure Warp code or when no ambiguity exists.

---

## 3. Kernel Design Patterns

### Naming Conventions

Kernel wrappers are two-tiered:

- Low-level functions are decorated with `@torch.library.custom_op` with
  `mutates_args`, that take pre-allocated tensors and return `None`. These
  private functions dispatch the kernels and are registered so that
  `torch.compile`/backwards passes will recognize them.
- Higher-level wrapper functions handle tensor allocations and call the
  low-level wrapper.

```python
@torch.library.custom_op(
    "nvalchemiops::kernel_name",
    mutates_args=(...),
)
def _low_level_wrapper(...):
    """Infer devices, dtypes, dispatch correct warp kernel"""

def high_level_wrapper(...):
    """Handles tensor allocations; main entry point for users"""
```

| Type | Convention | Example |
|------|------------|---------|
| Private kernel | Leading underscore | `_geom_cn_kernel` |
| Helper (`wp.func`) | Leading underscore | `_valid_neighbor`, `_s5_switch` |
| Public API | No underscore | `neighbor_list`, `dftd3` |

### Linters

In many cases, naming scientific variables compactly can go against
PEP-style formatting/style guides. You can decorate lines where variables
are declared to disable false positives (particularly from Sonar):

```python
dE_dCN = ...  # NOSONAR (S125) "math formula"
```

Similarly, there can be branching based off numerics that are picked up
by Sonar, which is not necessarily bad advice, albeit somewhat irrelevant:

```python
if value == 0.0: ...  # NOSONAR (S1244) "warp kernel"
```

### Precision Support

- When appropriate to do so, overload kernels programmatically and store
  the results in a dictionary with `dtype`s as keys:

```python
@wp.kernel
def _my_kernel(positions: wp.array(dtype=Any), values: wp.array(dtype=Any)):
    # ... generic kernel code ...

# Register overloads
kernel_overloads = {}
for scalar_type, vec_type in zip([wp.float16, wp.float32, wp.float64],
                                  [wp.vec3h, wp.vec3f, wp.vec3d]):
    kernel_overloads[(scalar_type, vec_type)] = wp.overload(_my_kernel, {
        "positions": wp.array(dtype=vec_type),
        "values": wp.array(dtype=scalar_type),
    })

# runtime retrieval; determines and launches appropriate overload
def kernel_wrapper(...):
    scalar_type = scalar_data.dtype
    vec_type = vector_data.dtype
    kernel_func = kernel_overloads[(scalar_type, vec_type)]
    wp.launch(kernel_func, ...)
```

- Not all precisions should be supported: if results are known to
  underflow particularly at lower precisions, do not add them to the
  overloads.

### Documentation

Use **NumPy-style docstrings** with the following specific items within
each category:

- Summary line
- Parameters (with shape, dtype, description)
  - Array outputs should be denoted with `OUTPUT` to indicate arrays
    that are expected to be pre-allocated, and are mutated in place by
    a kernel.
- Returns
  - Not generally applicable to Warp kernels, and more for PyTorch
    wrappers
- Notes (launch patterns, caveats)
  - Document the thread abstraction (e.g. per-atom, per-system)
  - Known performance characteristics
- See Also
  - Reference related kernels, particularly those that are run
    immediately before or after the current kernel.

## Unit Testing

### Test Organization

- Try and mirror test modules with the Python package tree: for example,
  a `test_dftd3.py` mirrors `dftd3.py`.
- Use `conftest.py` for shared fixtures, such as systems to test against,
  devices, etc.
- Group categories of tests in classes:

```python
class TestCategory:
    @pytest.fixture
    def category_specific_fixture(): ...

    def test_kernel_basic(): ...

    def test_kernel_with_fixture(category_specific_fixture): ...
```

### Test Patterns

1. Parametrized tests for variations, particularly with device and
   datatypes (precision):

   ```python
   @pytest.mark.parametrize("dtype", ["float32", "float64"])
   @pytest.mark.parametrize("device", ["cuda:0", "cpu"])
   def test_kernel(dtype, device):
       # ... test implementation ...
   ```

1. Full pipeline tests (primary coverage): test the end-to-end workflow,
   making sure that the values are not just finite, but once correctness
   has been established, check numerical tests to make sure kernels are
   numerically stable between changes:

   ```python
   def test_dftd3_full_pipeline():
       """Test complete pipeline exercising all kernels."""
       energy, forces, cn = dftd3(...)
       assert torch.isfinite(energy).all()


   def test_kernel_values():
       """Check numerical results against hardcoded reference values"""
       values = kernel(...)
       hardcoded_values = torch.tensor(...)
       assert torch.allclose(values, hardcoded_values, rtol=..., atol=...)
   ```

1. Known fail states: If a kernel or function is known to fail
   predictably with certain inputs or conditions, there must be tests
   that capture this using patterns such as `pytest.raises(Exception)`.
   Good things to capture here are incorrect shapes and `dtype`s for
   `warp` kernels.

1. Edge cases: Empty systems, potentially odd input/outputs such as
   negative/positive values.

1. Helper function tests: Wrap `wp.func` in test kernel for isolated
   testing

## Performance Benchmarking

### GPU Profiling Requirements

**Critical for accurate timing**:

1. **Warmup runs**: GPU kernels compile on first launch (JIT)
2. **Synchronization**: GPU operations are asynchronous - always call
   `torch.cuda.synchronize()`/`wp.synchronize()`
3. **CUDA Events**: More relevant, and therefore preferable over
   `time.perf_counter()` calls
4. **Multiple runs**: following warm up runs to average statistics
5. **Units**: Report units at relevant timescales: micro/milliseconds is
   typical but may depend on the kernel

**General pattern for collecting performance data**:

```python
# Warmup
for _ in range(warmup_runs):
    func()
    torch.cuda.synchronize()

# Timing with CUDA events
times = []
torch.cuda.memory.reset_peak_memory_stats()
for _ in range(timing_runs):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    func()
    end_event.record()

    torch.cuda.synchronize()
    times.append(start_event.elapsed_time(end_event))  # milliseconds
peak_memory = torch.cuda.memory.max_memory_allocated()
```

**NVTX annotations** for Nsight Systems profiling:

```python
import nvtx

@nvtx.annotate("compute_dftd3", color="red")
def my_function(...):
    return dftd3(...)
```

### What to Exclude from Timing

**Pre-compute separately** (not part of kernel timing):

- Neighbor list construction (unless benchmarking neighbor lists)
- Parameter loading
- Data transfers (unless benchmarking transfers)
- One-time allocations

See `benchmarks/interactions/dispersion/benchmark_d3_synthetic.py` and
`benchmarks/benchmark_neighborlist.py` for complete examples.

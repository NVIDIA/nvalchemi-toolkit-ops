# Neighbor List Benchmarks

This page presents benchmark results for various neighbor list algorithms
across different GPU hardware. Results are automatically generated from
CSV files in the `benchmark_results/` directory.

```{warning}
These results are intended to be indicative _only_: your actual performance may
vary depending on the atomic system topology, software and hardware configuration
and we encourage users to benchmark on their own systems of interest.
```

## How to Read These Charts

Time Scaling
: Median execution time (ms) vs. system size. Lower is better. Cell list
  algorithms show $O(N)$ scaling while naive algorithms show $O(N^2)$.

Throughput
: Atoms processed per millisecond. Higher is better. This metric helps compare
  efficiency across different system sizes.

Memory
: Peak GPU memory usage (MB) vs. system size. Useful for estimating memory
  requirements for your target system.

## Performance Results

Select a method to view detailed benchmark data and scaling plots:

### Naive

Brute-force $O(N^2)$ algorithm. Best for very small systems where the overhead of
cell list construction exceeds the computational savings.

::::{tab-set}

:::{tab-item} Backend Comparison

Simple comparison of single (non-batched) system computations between backends,
where we scale up the size of the system.

#### Time Scaling

```{figure} _static/neighborlist_scaling_naive_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Naive algorithm backend time comparison

Median execution time comparison between backends.
The $O(N^2)$ scaling becomes apparent for larger systems.
```

#### Throughput

```{figure} _static/neighborlist_throughput_naive_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Naive algorithm backend throughput comparison

Throughput (atoms/ms) comparison between backends.
```

#### Memory Usage

```{figure} _static/neighborlist_memory_naive_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Naive algorithm backend memory comparison

Peak GPU memory consumption comparison between backends.
```

:::

:::{tab-item} nvalchemiops (Torch)

Scaling of the naive algorithm with the `nvalchemiops` Torch backend.
Shows how performance scales with different batch sizes.

#### Time Scaling

```{figure} _static/neighborlist_scaling_naive_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Naive algorithm Torch time scaling

Execution time scaling for different batch sizes.
```

#### Throughput

```{figure} _static/neighborlist_throughput_naive_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Naive algorithm Torch throughput

Throughput (atoms/ms) for different batch sizes.
```

#### Memory Usage

```{figure} _static/neighborlist_memory_naive_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Naive algorithm Torch memory usage

Peak GPU memory consumption for different batch sizes.
```

:::

:::{tab-item} nvalchemiops (JAX)

Scaling of the naive algorithm with the `nvalchemiops` JAX backend.
Shows how performance scales with different batch sizes.

#### Time Scaling

```{figure} _static/neighborlist_scaling_naive_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Naive algorithm JAX time scaling

Execution time scaling for different batch sizes.
```

#### Throughput

```{figure} _static/neighborlist_throughput_naive_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Naive algorithm JAX throughput

Throughput (atoms/ms) for different batch sizes.
```

#### Memory Usage

```{figure} _static/neighborlist_memory_naive_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Naive algorithm JAX memory usage

Peak GPU memory consumption for different batch sizes.
```

:::

::::

### Cell List

Spatial hashing $O(N)$ algorithm. Recommended for medium to large systems where
computational efficiency is critical.

::::{tab-set}

:::{tab-item} Backend Comparison

Simple comparison of single (non-batched) system computations between backends,
where we scale up the size of the system.

#### Time Scaling

```{figure} _static/neighborlist_scaling_cell-list_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Cell list algorithm backend time comparison

Median execution time comparison between backends.
Shows near-linear $O(N)$ scaling for large systems.
```

#### Throughput

```{figure} _static/neighborlist_throughput_cell-list_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Cell list algorithm backend throughput comparison

Throughput (atoms/ms) comparison between backends.
```

#### Memory Usage

```{figure} _static/neighborlist_memory_cell-list_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Cell list algorithm backend memory comparison

Peak GPU memory consumption comparison between backends.
```

:::

:::{tab-item} nvalchemiops (Torch)

Scaling of the cell list algorithm with the `nvalchemiops` Torch backend.
Shows how performance scales with different batch sizes.

#### Time Scaling

```{figure} _static/neighborlist_scaling_cell-list_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Cell list algorithm Torch time scaling

Execution time scaling for different batch sizes.
```

#### Throughput

```{figure} _static/neighborlist_throughput_cell-list_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Cell list algorithm Torch throughput

Throughput (atoms/ms) for different batch sizes.
```

#### Memory Usage

```{figure} _static/neighborlist_memory_cell-list_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Cell list algorithm Torch memory usage

Peak GPU memory consumption for different batch sizes.
```

:::

:::{tab-item} nvalchemiops (JAX)

Scaling of the cell list algorithm with the `nvalchemiops` JAX backend.
Shows how performance scales with different batch sizes.

#### Time Scaling

```{figure} _static/neighborlist_scaling_cell-list_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Cell list algorithm JAX time scaling

Execution time scaling for different batch sizes.
```

#### Throughput

```{figure} _static/neighborlist_throughput_cell-list_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Cell list algorithm JAX throughput

Throughput (atoms/ms) for different batch sizes.
```

#### Memory Usage

```{figure} _static/neighborlist_memory_cell-list_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Cell list algorithm JAX memory usage

Peak GPU memory consumption for different batch sizes.
```

:::

::::

### Batch Naive

Batched brute-force algorithm for processing multiple small systems
simultaneously. Useful for ML workflows with many small molecules.

::::{tab-set}

:::{tab-item} Backend Comparison

Simple comparison of single (non-batched) system computations between backends,
where we scale up the size of the system.

#### Time Scaling

```{figure} _static/neighborlist_scaling_batch-naive_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch naive algorithm backend time comparison

Median execution time comparison between backends.
```

#### Throughput

```{figure} _static/neighborlist_throughput_batch-naive_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch naive algorithm backend throughput comparison

Throughput (atoms/ms) comparison between backends.
```

#### Memory Usage

```{figure} _static/neighborlist_memory_batch-naive_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch naive algorithm backend memory comparison

Peak GPU memory consumption comparison between backends.
```

:::

:::{tab-item} nvalchemiops (Torch)

Scaling of the batched naive algorithm with the `nvalchemiops` Torch backend.
Shows how performance scales with different batch sizes.

#### Time Scaling

```{figure} _static/neighborlist_scaling_batch-naive_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch naive algorithm Torch time scaling

Execution time scaling for different batch sizes.
```

#### Throughput

```{figure} _static/neighborlist_throughput_batch-naive_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch naive algorithm Torch throughput

Throughput (atoms/ms) for different batch sizes.
```

#### Memory Usage

```{figure} _static/neighborlist_memory_batch-naive_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch naive algorithm Torch memory usage

Peak GPU memory consumption for different batch sizes.
```

:::

:::{tab-item} nvalchemiops (JAX)

Scaling of the batched naive algorithm with the `nvalchemiops` JAX backend.
Shows how performance scales with different batch sizes.

#### Time Scaling

```{figure} _static/neighborlist_scaling_batch-naive_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch naive algorithm JAX time scaling

Execution time scaling for different batch sizes.
```

#### Throughput

```{figure} _static/neighborlist_throughput_batch-naive_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch naive algorithm JAX throughput

Throughput (atoms/ms) for different batch sizes.
```

#### Memory Usage

```{figure} _static/neighborlist_memory_batch-naive_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch naive algorithm JAX memory usage

Peak GPU memory consumption for different batch sizes.
```

:::

::::

### Batch Cell List

Batched spatial hashing algorithm for processing multiple systems
simultaneously with O(N) scaling per system.

::::{tab-set}

:::{tab-item} Backend Comparison

Simple comparison of single (non-batched) system computations between backends,
where we scale up the size of the system.

#### Time Scaling

```{figure} _static/neighborlist_scaling_batch-cell-list_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch cell list algorithm backend time comparison

Median execution time comparison between backends.
```

#### Throughput

```{figure} _static/neighborlist_throughput_batch-cell-list_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch cell list algorithm backend throughput comparison

Throughput (atoms/ms) comparison between backends.
```

#### Memory Usage

```{figure} _static/neighborlist_memory_batch-cell-list_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch cell list algorithm backend memory comparison

Peak GPU memory consumption comparison between backends.
```

:::

:::{tab-item} nvalchemiops (Torch)

Scaling of the batched cell list algorithm with the `nvalchemiops` Torch backend.
Shows how performance scales with different batch sizes.

#### Time Scaling

```{figure} _static/neighborlist_scaling_batch-cell-list_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch cell list algorithm Torch time scaling

Execution time scaling for different batch sizes.
```

#### Throughput

```{figure} _static/neighborlist_throughput_batch-cell-list_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch cell list algorithm Torch throughput

Throughput (atoms/ms) for different batch sizes.
```

#### Memory Usage

```{figure} _static/neighborlist_memory_batch-cell-list_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch cell list algorithm Torch memory usage

Peak GPU memory consumption for different batch sizes.
```

:::

:::{tab-item} nvalchemiops (JAX)

Scaling of the batched cell list algorithm with the `nvalchemiops` JAX backend.
Shows how performance scales with different batch sizes.

#### Time Scaling

```{figure} _static/neighborlist_scaling_batch-cell-list_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch cell list algorithm JAX time scaling

Execution time scaling for different batch sizes.
```

#### Throughput

```{figure} _static/neighborlist_throughput_batch-cell-list_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch cell list algorithm JAX throughput

Throughput (atoms/ms) for different batch sizes.
```

#### Memory Usage

```{figure} _static/neighborlist_memory_batch-cell-list_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Batch cell list algorithm JAX memory usage

Peak GPU memory consumption for different batch sizes.
```

:::

::::

## Hardware Information

**GPU**: NVIDIA H100 80GB HBM3

## Benchmark Configuration

| Parameter | Value |
| --------- | ----- |
| Cutoff | 5.0 Å |
| System Type | FCC crystal lattice |
| Warmup Iterations | 3 |
| Timing Iterations | 10 |
| Dtype | `float32` |

## Interpreting Results

`method`
: Algorithm name.

`total_atoms`
: Total number of atoms in the system.

`atoms_per_system`
: Atoms per system (relevant for batch methods).

`total_neighbors`
: Total number of neighbor pairs found.

`batch_size`
: Number of systems processed simultaneously (1 for non-batch methods).

`median_time_ms`
: Median execution time in milliseconds (lower is better).

`peak_memory_mb`
: Peak GPU memory usage in megabytes.

## Running Your Own Benchmarks

To generate benchmark results for your hardware:

```bash
cd benchmarks/neighborlist
python benchmark_neighborlist.py \
    --config benchmark_config.yaml \
    --output-dir ../../docs/benchmarks/benchmark_results
```

Results will be saved as CSV files and plots will be automatically generated
during the next documentation build.

# Neighbor List Benchmarks

Performance benchmarks for neighbor list algorithms in ALCHEMI Toolkit-Ops.
Results show scaling behaviour across system sizes for multiple cutoff radii
and algorithms.

```{warning}
These results are intended to be indicative _only_: your actual performance may
vary depending on the atomic system topology, software and hardware configuration
and we encourage users to benchmark on their own systems of interest.
```

## How to Read These Charts

Time Scaling
: Mean execution time (ms) vs. system size. Lower is better. Cell list
  algorithms show $O(N)$ scaling while naive algorithms show $O(N^2)$.

Throughput
: Atoms processed per millisecond. Higher is better. This metric helps compare
  efficiency across different system sizes.

Memory
: Peak GPU memory usage (MB) vs. system size. Useful for estimating memory
  requirements for your target system.

## Performance Results

::::{tab-set}

:::{tab-item} Torch
:selected:

`````{tab-set}

````{tab-item} CsCl
:selected:

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/nl-cscl-system-size-scaling-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size for naive and cell list algorithms.

        .. figure:: _static/nl-cscl-system-size-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput (atoms/ms) vs. system size.

        .. figure:: _static/nl-cscl-system-size-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory (MB) vs. system size.

    .. tab-item:: Constant Workload

        .. figure:: _static/nl-cscl-constant-workload-scaling-time.png
           :width: 90%
           :align: center

           Execution time at constant total atom count, varying batch size.

        .. figure:: _static/nl-cscl-constant-workload-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput at constant total atom count.

        .. figure:: _static/nl-cscl-constant-workload-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory at constant total atom count.

    .. tab-item:: Batch Scaling

        .. figure:: _static/nl-cscl-batch-scaling-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (fixed atoms per system).

        .. figure:: _static/nl-cscl-batch-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size.

        .. figure:: _static/nl-cscl-batch-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory vs. batch size.
```

````

````{tab-item} NH₃

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/nl-nh3-system-size-scaling-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size for naive and cell list algorithms (NH₃).

        .. figure:: _static/nl-nh3-system-size-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput (atoms/ms) vs. system size (NH₃).

        .. figure:: _static/nl-nh3-system-size-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory (MB) vs. system size (NH₃).

    .. tab-item:: Constant Workload

        .. figure:: _static/nl-nh3-constant-workload-scaling-time.png
           :width: 90%
           :align: center

           Execution time at constant total atom count (NH₃).

        .. figure:: _static/nl-nh3-constant-workload-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput at constant total atom count (NH₃).

        .. figure:: _static/nl-nh3-constant-workload-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory at constant total atom count (NH₃).

    .. tab-item:: Batch Scaling

        .. figure:: _static/nl-nh3-batch-scaling-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (NH₃).

        .. figure:: _static/nl-nh3-batch-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size (NH₃).

        .. figure:: _static/nl-nh3-batch-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory vs. batch size (NH₃).
```

````

`````

:::

:::{tab-item} JAX

`````{tab-set}

````{tab-item} CsCl
:selected:

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/nl-cscl-system-size-scaling-jax-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size (JAX).

        .. figure:: _static/nl-cscl-system-size-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput (atoms/ms) vs. system size (JAX).

    .. tab-item:: Constant Workload

        .. figure:: _static/nl-cscl-constant-workload-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time at constant total atom count (JAX).

        .. figure:: _static/nl-cscl-constant-workload-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput at constant total atom count (JAX).

    .. tab-item:: Batch Scaling

        .. figure:: _static/nl-cscl-batch-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (JAX).

        .. figure:: _static/nl-cscl-batch-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size (JAX).

```

```{note}
JAX memory plots are omitted. XLA's pool allocator pre-allocates 75% of GPU
VRAM and reuses it across calls, preventing reliable per-config measurement.
Torch memory plots are representative — both backends use identical Warp GPU
kernels with the same memory footprint.
```

````

````{tab-item} NH₃

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/nl-nh3-system-size-scaling-jax-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size (JAX, NH₃).

        .. figure:: _static/nl-nh3-system-size-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput (atoms/ms) vs. system size (JAX, NH₃).

    .. tab-item:: Constant Workload

        .. figure:: _static/nl-nh3-constant-workload-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time at constant total atom count (JAX, NH₃).

        .. figure:: _static/nl-nh3-constant-workload-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput at constant total atom count (JAX, NH₃).

    .. tab-item:: Batch Scaling

        .. figure:: _static/nl-nh3-batch-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (JAX, NH₃).

        .. figure:: _static/nl-nh3-batch-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size (JAX, NH₃).

```

```{note}
JAX memory plots are omitted. XLA's pool allocator pre-allocates 75% of GPU
VRAM and reuses it across calls, preventing reliable per-config measurement.
Torch memory plots are representative — both backends use identical Warp GPU
kernels with the same memory footprint.
```

````

`````

:::

:::{tab-item} Backend Comparison

These plots compare the Torch and JAX backends at a **single cutoff of
15 Å** (the representative mid-range value from the cutoff sweep) to
keep the plot readable.

`````{tab-set}

````{tab-item} CsCl
:selected:

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/nl-cscl-system-size-comparison-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time comparison.

        .. figure:: _static/nl-cscl-system-size-comparison-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput comparison.

        .. figure:: _static/nl-cscl-system-size-comparison-memory.png
           :width: 90%
           :align: center

           Torch vs. JAX memory comparison.

    .. tab-item:: Constant Workload

        .. figure:: _static/nl-cscl-constant-workload-comparison-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time at constant workload.

        .. figure:: _static/nl-cscl-constant-workload-comparison-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput at constant workload.

        .. figure:: _static/nl-cscl-constant-workload-comparison-memory.png
           :width: 90%
           :align: center

           Torch vs. JAX memory at constant workload.

    .. tab-item:: Batch Scaling

        .. figure:: _static/nl-cscl-batch-comparison-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time vs. batch size.

        .. figure:: _static/nl-cscl-batch-comparison-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput vs. batch size.

        .. figure:: _static/nl-cscl-batch-comparison-memory.png
           :width: 90%
           :align: center

           Torch vs. JAX memory vs. batch size.
```

````

````{tab-item} NH₃

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/nl-nh3-system-size-comparison-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time comparison (NH₃).

        .. figure:: _static/nl-nh3-system-size-comparison-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput comparison (NH₃).

        .. figure:: _static/nl-nh3-system-size-comparison-memory.png
           :width: 90%
           :align: center

           Torch vs. JAX memory comparison (NH₃).

    .. tab-item:: Batch Scaling

        .. figure:: _static/nl-nh3-batch-comparison-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time vs. batch size (NH₃).

        .. figure:: _static/nl-nh3-batch-comparison-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput vs. batch size (NH₃).

        .. figure:: _static/nl-nh3-batch-comparison-memory.png
           :width: 90%
           :align: center

           Torch vs. JAX memory vs. batch size (NH₃).
```

````

`````

:::

::::

## Benchmark Configuration

| Parameter | Value |
| --------- | ----- |
| Cutoffs | 6.0, 15.0, 25.0 Å |
| Methods | naive, cell list |
| System Type | CsCl (pymatgen), NH₃ (PDB) |
| Warmup Iterations | 3 |
| Timing Iterations | 20 |
| Dtype | `float32` |

## Running Your Own Benchmarks

Run from the repository root:

```bash
python -m benchmarks.neighborlist.benchmark_neighborlist \
    --config benchmarks/neighborlist/benchmark_config.yaml \
    --output-dir docs/benchmarks/benchmark_results
```

For the JAX backend:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m benchmarks.neighborlist.benchmark_neighborlist \
    --config benchmarks/neighborlist/benchmark_config.yaml \
    --backend jax \
    --output-dir docs/benchmarks/benchmark_results
```

# DFT-D3 Dispersion Benchmarks

Performance benchmarks for DFT-D3 dispersion corrections in ALCHEMI Toolkit-Ops.
Results show scaling behaviour with increasing system size for periodic systems,
including both single-system and batched computations.

```{warning}
These results are intended to be indicative _only_: your actual performance may
vary depending on the atomic system topology, software and hardware configuration
and we encourage users to benchmark on their own systems of interest.
```

## How to Read These Charts

Time Scaling
: Mean execution time (µs/atom) vs. system size. Lower is better. Timings
  exclude neighbor list construction, and only comprise the DFT-D3 computation.

Throughput
: Atoms processed per second (plotted as 10⁶ atoms/s). Higher is better.
  This indicates the scaling point where the GPU saturates.

Memory
: Peak memory reported by the Torch CUDA allocator vs. system size. Units
  switch between MB and GB automatically on the y-axis. JAX memory is not
  measured by this suite.

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

        .. figure:: _static/d3-cscl-system-size-scaling-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size (Torch, CsCl).

        .. figure:: _static/d3-cscl-system-size-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput (10⁶ atoms/s) vs. system size.

        .. figure:: _static/d3-cscl-system-size-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory vs. system size.

    .. tab-item:: Constant Workload

        .. figure:: _static/d3-cscl-constant-workload-scaling-time.png
           :width: 90%
           :align: center

           Execution time near the configured total-atom target, varying batch size.

        .. figure:: _static/d3-cscl-constant-workload-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput near the configured total-atom target.

        .. figure:: _static/d3-cscl-constant-workload-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory near the configured total-atom target.

    .. tab-item:: Batch Scaling

        .. figure:: _static/d3-cscl-batch-scaling-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (fixed atoms per system).

        .. figure:: _static/d3-cscl-batch-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size.

        .. figure:: _static/d3-cscl-batch-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory vs. batch size.
```

````

````{tab-item} NH₃

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/d3-nh3-system-size-scaling-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size (Torch, NH₃).

        .. figure:: _static/d3-nh3-system-size-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput (10⁶ atoms/s) vs. system size (NH₃).

        .. figure:: _static/d3-nh3-system-size-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory vs. system size (NH₃).

    .. tab-item:: Constant Workload

        .. figure:: _static/d3-nh3-constant-workload-scaling-time.png
           :width: 90%
           :align: center

           Execution time near the configured total-atom target (NH₃).

        .. figure:: _static/d3-nh3-constant-workload-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput near the configured total-atom target (NH₃).

        .. figure:: _static/d3-nh3-constant-workload-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory near the configured total-atom target (NH₃).

    .. tab-item:: Batch Scaling

        .. figure:: _static/d3-nh3-batch-scaling-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (NH₃).

        .. figure:: _static/d3-nh3-batch-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size (NH₃).

        .. figure:: _static/d3-nh3-batch-scaling-memory.png
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

        .. figure:: _static/d3-cscl-system-size-scaling-jax-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size (JAX, CsCl).

        .. figure:: _static/d3-cscl-system-size-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput (10⁶ atoms/s) vs. system size (JAX).

    .. tab-item:: Constant Workload

        .. figure:: _static/d3-cscl-constant-workload-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time near the configured total-atom target (JAX).

        .. figure:: _static/d3-cscl-constant-workload-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput near the configured total-atom target (JAX).

    .. tab-item:: Batch Scaling

        .. figure:: _static/d3-cscl-batch-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (JAX).

        .. figure:: _static/d3-cscl-batch-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size (JAX).

```

```{note}
JAX memory plots are omitted. The suite does not measure JAX memory:
XLA's allocator pool and buffer reuse make per-call allocation deltas
unreliable. The Torch panels report the Torch allocator only; they are not a
proxy for JAX memory because the framework wrappers, buffer lifetimes, and
allocators differ.
```

````

````{tab-item} NH₃

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/d3-nh3-system-size-scaling-jax-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size (JAX, NH₃).

        .. figure:: _static/d3-nh3-system-size-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput (10⁶ atoms/s) vs. system size (JAX, NH₃).

    .. tab-item:: Constant Workload

        .. figure:: _static/d3-nh3-constant-workload-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time near the configured total-atom target (JAX, NH₃).

        .. figure:: _static/d3-nh3-constant-workload-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput near the configured total-atom target (JAX, NH₃).

    .. tab-item:: Batch Scaling

        .. figure:: _static/d3-nh3-batch-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (JAX, NH₃).

        .. figure:: _static/d3-nh3-batch-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size (JAX, NH₃).

```

```{note}
JAX memory plots are omitted. The suite does not measure JAX memory:
XLA's allocator pool and buffer reuse make per-call allocation deltas
unreliable. The Torch panels report the Torch allocator only; they are not a
proxy for JAX memory because the framework wrappers, buffer lifetimes, and
allocators differ.
```

````

`````

:::

:::{tab-item} Backend Comparison

These panels use matched successful Torch and JAX points for the same system,
cutoff, and scaling coordinate. Both backends time the DFT-D3 call with the
neighbor list built beforehand, and both exclude warmup/compile work. Torch
uses CUDA events while JAX uses wall-clock timing with a final synchronized
result, so the workload is aligned but the timing harness and framework
overheads are not identical. Memory panels contain Torch data only.

`````{tab-set}

````{tab-item} CsCl
:selected:

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/d3-cscl-system-size-comparison-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time comparison (CsCl).

        .. figure:: _static/d3-cscl-system-size-comparison-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput comparison.

        .. figure:: _static/d3-cscl-system-size-comparison-memory.png
           :width: 90%
           :align: center

           Memory vs. system size (Torch only — JAX memory not measured).

    .. tab-item:: Constant Workload

        .. figure:: _static/d3-cscl-constant-workload-comparison-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time at constant workload.

        .. figure:: _static/d3-cscl-constant-workload-comparison-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput at constant workload.

        .. figure:: _static/d3-cscl-constant-workload-comparison-memory.png
           :width: 90%
           :align: center

           Memory near the configured total-atom target (Torch only — JAX memory not measured).

    .. tab-item:: Batch Scaling

        .. figure:: _static/d3-cscl-batch-comparison-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time vs. batch size.

        .. figure:: _static/d3-cscl-batch-comparison-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput vs. batch size.

        .. figure:: _static/d3-cscl-batch-comparison-memory.png
           :width: 90%
           :align: center

           Memory vs. batch size (Torch only — JAX memory not measured).
```

````

````{tab-item} NH₃

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/d3-nh3-system-size-comparison-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time comparison (NH₃).

        .. figure:: _static/d3-nh3-system-size-comparison-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput comparison (NH₃).

        .. figure:: _static/d3-nh3-system-size-comparison-memory.png
           :width: 90%
           :align: center

           Memory vs. system size (NH₃; Torch only — JAX memory not measured).

    .. tab-item:: Constant Workload

        .. figure:: _static/d3-nh3-constant-workload-comparison-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time at constant workload (NH₃).

        .. figure:: _static/d3-nh3-constant-workload-comparison-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput at constant workload (NH₃).

        .. figure:: _static/d3-nh3-constant-workload-comparison-memory.png
           :width: 90%
           :align: center

           Memory near the configured total-atom target (NH₃; Torch only —
           JAX memory not measured).

    .. tab-item:: Batch Scaling

        .. figure:: _static/d3-nh3-batch-comparison-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time vs. batch size (NH₃).

        .. figure:: _static/d3-nh3-batch-comparison-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput vs. batch size (NH₃).

        .. figure:: _static/d3-nh3-batch-comparison-memory.png
           :width: 90%
           :align: center

           Memory vs. batch size (NH₃; Torch only — JAX memory not measured).
```

````

`````

:::

::::

## Benchmark Configuration

| Parameter | Value |
| --------- | ----- |
| Cutoffs | 15.0, 25.0 Å |
| System Type | CsCl supercells (programmatic), NH₃ (PDB) |
| Method | `dftd3` |
| Timed outputs | Energy, analytical forces, and coordination numbers |
| Neighbor List | Built outside the timed D3 region; CSV rows record `neighbor_setup_method` |
| Warmup Iterations | 3 |
| Timing Iterations | 10 |
| Precision | `float32` |

CSV rows include `time_d3_us_per_atom` for the D3-only timed region; neighbor
list setup is tracked separately through `neighbor_setup_method`.

### DFT-D3 Parameters (BJ-damping, PBE)

The suite overrides these functional-specific values from
``benchmarks/interactions/dispersion/benchmark_config.yaml``:

| Parameter | Value |
| --------- | ----- |
| `a1` | 0.4289 |
| `a2` | 4.4407 |
| `s8` | 0.7875 |

The remaining D3 controls use the public wrapper defaults: ``s6=1.0``,
``k1=16.0``, and ``k3=-4.0``. They are not benchmark-YAML controls in this
reportable suite.

## Running Your Own Benchmarks

Run from the repository root.

### Torch Backend (default)

```bash
RESULT_DIR="$BENCHMARK_SCRATCH/results/manual-d3-run"
python -m benchmarks.interactions.dispersion.benchmark_dftd3 \
    --config benchmarks/interactions/dispersion/benchmark_config.yaml \
    --backend torch \
    --output-dir "$RESULT_DIR"
```

### JAX Backend

```bash
RESULT_DIR="$BENCHMARK_SCRATCH/results/manual-d3-run"
python -m benchmarks.interactions.dispersion.benchmark_dftd3 \
    --config benchmarks/interactions/dispersion/benchmark_config.yaml \
    --backend jax \
    --output-dir "$RESULT_DIR"
```

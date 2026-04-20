# Electrostatics Benchmarks

Performance benchmarks for electrostatic interaction methods in ALCHEMI
Toolkit-Ops — Ewald summation and Particle Mesh Ewald (PME). Results
show scaling behaviour across system sizes for both single-system and
batched computations.

```{warning}
These results are intended to be indicative _only_: your actual performance may
vary depending on the atomic system topology, software and hardware configuration
and we encourage users to benchmark on their own systems of interest.
```

```{note}
Plots show base PME and Ewald — the ``_cg`` (charge-gradient)
variants stay in the CSVs but are filtered from the panels.
Panels time the full ``particle_mesh_ewald`` / ``ewald_summation``
calls. Missing points = OOM on the reference H100 80 GB.
```

## How to Read These Charts

Time Scaling
: Mean execution time (ms) vs. system size. Lower is better. Timings include
  both real-space and reciprocal-space contributions when running "full" mode.

Throughput
: Atoms processed per millisecond. Higher is better. This indicates the scaling
  point where the GPU saturates.

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

        .. figure:: _static/el-cscl-system-size-scaling-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size (Torch: PME + Ewald, CsCl).

        .. figure:: _static/el-cscl-system-size-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput (atoms/ms) vs. system size.

        .. figure:: _static/el-cscl-system-size-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory (MB) vs. system size.

    .. tab-item:: Constant Workload

        .. figure:: _static/el-cscl-constant-workload-scaling-time.png
           :width: 90%
           :align: center

           Execution time at constant total atom count, varying batch size.

        .. figure:: _static/el-cscl-constant-workload-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput at constant total atom count.

        .. figure:: _static/el-cscl-constant-workload-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory at constant total atom count.

    .. tab-item:: Batch Scaling

        .. figure:: _static/el-cscl-batch-scaling-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (fixed atoms per system).

        .. figure:: _static/el-cscl-batch-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size.

        .. figure:: _static/el-cscl-batch-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory vs. batch size.
```

````

````{tab-item} NH₃

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/el-nh3-system-size-scaling-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size (Torch, NH₃).

        .. figure:: _static/el-nh3-system-size-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput (atoms/ms) vs. system size (NH₃).

        .. figure:: _static/el-nh3-system-size-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory (MB) vs. system size (NH₃).

    .. tab-item:: Constant Workload

        .. figure:: _static/el-nh3-constant-workload-scaling-time.png
           :width: 90%
           :align: center

           Execution time at constant total atom count (NH₃).

        .. figure:: _static/el-nh3-constant-workload-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput at constant total atom count (NH₃).

        .. figure:: _static/el-nh3-constant-workload-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory at constant total atom count (NH₃).

    .. tab-item:: Batch Scaling

        .. figure:: _static/el-nh3-batch-scaling-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (NH₃).

        .. figure:: _static/el-nh3-batch-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size (NH₃).

        .. figure:: _static/el-nh3-batch-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory vs. batch size (NH₃).
```

````

`````

:::

:::{tab-item} JAX

```{note}
The JAX backend does not currently support ``compute_charge_gradients`` for
Ewald summation. Those configurations are skipped; PME results are unaffected.
```

`````{tab-set}

````{tab-item} CsCl
:selected:

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/el-cscl-system-size-scaling-jax-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size (JAX, CsCl).

        .. figure:: _static/el-cscl-system-size-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput (atoms/ms) vs. system size (JAX).

    .. tab-item:: Constant Workload

        .. figure:: _static/el-cscl-constant-workload-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time at constant total atom count (JAX).

        .. figure:: _static/el-cscl-constant-workload-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput at constant total atom count (JAX).

    .. tab-item:: Batch Scaling

        .. figure:: _static/el-cscl-batch-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (JAX).

        .. figure:: _static/el-cscl-batch-scaling-jax-throughput.png
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

        .. figure:: _static/el-nh3-system-size-scaling-jax-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size (JAX, NH₃).

        .. figure:: _static/el-nh3-system-size-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput (atoms/ms) vs. system size (JAX, NH₃).

    .. tab-item:: Constant Workload

        .. figure:: _static/el-nh3-constant-workload-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time at constant total atom count (JAX, NH₃).

        .. figure:: _static/el-nh3-constant-workload-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput at constant total atom count (JAX, NH₃).

    .. tab-item:: Batch Scaling

        .. figure:: _static/el-nh3-batch-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (JAX, NH₃).

        .. figure:: _static/el-nh3-batch-scaling-jax-throughput.png
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

These comparison plots show PME and Ewald **without charge gradients**
(i.e. ``compute_charge_gradients=False``), because the public JAX
``ewald_summation`` API does not yet expose that flag. Enabling charge
gradients on the Torch side (``pme_cg`` / ``ewald_cg``) adds roughly
**5–15 %** to the per-call time at the median across scaling modes, so
the torch-vs-jax comparison shown here is representative of per-step
cost for force-only usage.

`````{tab-set}

````{tab-item} CsCl
:selected:

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/el-cscl-system-size-comparison-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time comparison (CsCl).

        .. figure:: _static/el-cscl-system-size-comparison-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput comparison.

        .. figure:: _static/el-cscl-system-size-comparison-memory.png
           :width: 90%
           :align: center

           Torch vs. JAX memory comparison.

    .. tab-item:: Constant Workload

        .. figure:: _static/el-cscl-constant-workload-comparison-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time at constant workload.

        .. figure:: _static/el-cscl-constant-workload-comparison-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput at constant workload.

        .. figure:: _static/el-cscl-constant-workload-comparison-memory.png
           :width: 90%
           :align: center

           Torch vs. JAX memory at constant workload.

    .. tab-item:: Batch Scaling

        .. figure:: _static/el-cscl-batch-comparison-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time vs. batch size.

        .. figure:: _static/el-cscl-batch-comparison-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput vs. batch size.

        .. figure:: _static/el-cscl-batch-comparison-memory.png
           :width: 90%
           :align: center

           Torch vs. JAX memory vs. batch size.
```

````

````{tab-item} NH₃

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/el-nh3-system-size-comparison-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time comparison (NH₃).

        .. figure:: _static/el-nh3-system-size-comparison-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput comparison (NH₃).

        .. figure:: _static/el-nh3-system-size-comparison-memory.png
           :width: 90%
           :align: center

           Torch vs. JAX memory comparison (NH₃).

    .. tab-item:: Constant Workload

        .. figure:: _static/el-nh3-constant-workload-comparison-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time at constant workload (NH₃).

        .. figure:: _static/el-nh3-constant-workload-comparison-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput at constant workload (NH₃).

        .. figure:: _static/el-nh3-constant-workload-comparison-memory.png
           :width: 90%
           :align: center

           Torch vs. JAX memory at constant workload (NH₃).

    .. tab-item:: Batch Scaling

        .. figure:: _static/el-nh3-batch-comparison-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time vs. batch size (NH₃).

        .. figure:: _static/el-nh3-batch-comparison-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput vs. batch size (NH₃).

        .. figure:: _static/el-nh3-batch-comparison-memory.png
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
| Accuracies | $10^{-4}$ / $10^{-6}$ |
| Methods | PME, Ewald summation |
| System Type | CsCl supercells (pymatgen), NH₃ (PDB) |
| Neighbor List | Cell list algorithm ($O(N)$ scaling) |
| Warmup Iterations | 3 |
| Timing Iterations | 20 |
| Precision | `float64` |

### Ewald/PME Parameters

Parameters are automatically estimated using accuracy-based parameter estimation:

| Parameter | Description |
| --------- | ----------- |
| `alpha` | Ewald splitting parameter (auto-estimated) |
| `k_cutoff` | Reciprocal-space cutoff for Ewald (auto-estimated) |
| `real_space_cutoff` | Real-space cutoff distance (auto-estimated) |
| `mesh_dimensions` | PME mesh grid size (auto-estimated) |
| `spline_order` | B-spline interpolation order (4) |

## Running Your Own Benchmarks

Run from the repository root. The YAML config already enables both PME
and Ewald; pass ``--methods pme`` or ``--methods ewald`` to benchmark
only one.

### Torch Backend (default)

```bash
python -m benchmarks.interactions.electrostatics.benchmark_electrostatics \
    --config benchmarks/interactions/electrostatics/benchmark_config.yaml \
    --backend torch \
    --output-dir docs/benchmarks/benchmark_results
```

### JAX Backend

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m benchmarks.interactions.electrostatics.benchmark_electrostatics \
    --config benchmarks/interactions/electrostatics/benchmark_config.yaml \
    --backend jax \
    --output-dir docs/benchmarks/benchmark_results
```

### Options

See ``--help`` for the full list; the flags relevant to electrostatics
runs are:

`--backend {torch,jax}`
: Computational backend (default: whatever ``config['runtime']['backend']``
  is set to, else ``torch``).

`--methods {pme,ewald} [{pme,ewald} ...]`
: Restrict to a subset of methods. Default: run every method marked
  ``enabled: true`` in the YAML (PME + Ewald ship enabled).

`--accuracies ACC [ACC ...]`
: Override the list of target accuracies (Hartree/atom) to sweep. Default:
  values from ``config['accuracies']`` (ships as ``[1e-4, 1e-6]``).

`--system`, `--mode`, `--timing-runs`, `--warmup-runs`, `--output-dir`
: Shared flags defined in ``benchmarks/config.py``; see ``--help`` for details.

# Electrostatics Benchmarks

Performance benchmarks for long-range electrostatic methods -- Ewald summation
and Particle Mesh Ewald (PME) -- across two chemical systems (CsCl crystal and
NH₃ molecular clusters) at two accuracy levels ($10^{-4}$ and $10^{-6}$).
All timings include `compute_forces=True` and `compute_charge_gradients=True`.

The Ewald summation method splits the Coulomb interaction into real-space and
reciprocal-space components. This is the traditional $O(N^{3/2})$ to $O(N^2)$
method depending on parameter choices. PME achieves $O(N \log N)$ scaling by
using FFTs (cuFFT) for the reciprocal-space contribution and is the recommended
method for large systems.

```{warning}
These results are intended to be indicative _only_: your actual performance may
vary depending on the atomic system topology, software and hardware configuration
and we encourage users to benchmark on their own systems of interest.
```

## How to Read These Charts

Time per Atom
: Microseconds per atom (μs/atom). Lower is better. Timings include both
  real-space and reciprocal-space contributions with forces and charge
  gradients enabled.

Throughput
: Millions of atoms processed per second (10⁶ atoms/s). Higher is better.
  This indicates the scaling point where the GPU saturates.

Peak GPU VRAM
: GPU video memory usage (log scale). The dotted line marks H100 80 GB
  capacity. Useful for estimating memory requirements for your system.

```{note}
All axes use log₂ scaling. The x-axis displays linearized atom counts
(128, 256, 512, ...) for readability, but spacing is logarithmic.
Some data points at the largest sizes may be absent due to GPU out-of-memory
(OOM) or because high-accuracy ($10^{-6}$) configurations are skipped for
systems ≥ 128k atoms to prevent OOM.
```

## Performance Results

`````{tab-set}

````{tab-item} CsCl

CsCl is an ionic crystal (Cs⁺ at corners, Cl⁻ at body center) --
the canonical stress test for long-range electrostatics due to its strong
Madelung potential.

::::{tab-set}

:::{tab-item} System Size Scaling

Single system growing from 128 to 128k atoms. Compares Ewald vs. PME at two
accuracy levels. PME's $O(N \log N)$ advantage becomes apparent for larger systems.

```{figure} _static/el-cscl-system-size-scaling-time.png
:width: 90%
:alt: EL CsCl system size time scaling

Execution time scaling for Ewald and PME at $10^{-4}$ and $10^{-6}$ accuracy.
```

---

```{figure} _static/el-cscl-system-size-scaling-throughput.png
:width: 90%
:alt: EL CsCl system size throughput

Throughput (10⁶ atoms/s) for single systems.
```

---

```{figure} _static/el-cscl-system-size-scaling-memory.png
:width: 90%
:alt: EL CsCl system size memory

Peak GPU VRAM consumption for single systems.
```

:::

:::{tab-item} Constant Workload Scaling

Fixed total atom count (~128k), varying system size × batch size.
Reveals the overhead of batching many small charged systems vs. fewer
large ones.

```{figure} _static/el-cscl-constant-workload-scaling-time.png
:width: 90%
:alt: EL CsCl constant workload time

Time per atom at constant ~128k total atoms. X-axis shows system size × batch size.
```

---

```{figure} _static/el-cscl-constant-workload-scaling-throughput.png
:width: 90%
:alt: EL CsCl constant workload throughput

Throughput (10⁶ atoms/s) at constant total workload.
```

---

```{figure} _static/el-cscl-constant-workload-scaling-memory.png
:width: 90%
:alt: EL CsCl constant workload memory

Peak GPU VRAM consumption at constant total workload.
```

:::

:::{tab-item} Batch Scaling

Fixed atoms per system, growing batch size. Shows how batched Ewald/PME
performance scales with increasing number of concurrent systems.

```{figure} _static/el-cscl-batch-scaling-time.png
:width: 90%
:alt: EL CsCl batch scaling time

Time per atom for batched systems at two accuracy levels.
```

---

```{figure} _static/el-cscl-batch-scaling-throughput.png
:width: 90%
:alt: EL CsCl batch scaling throughput

Throughput (10⁶ atoms/s) for batched processing.
```

---

```{figure} _static/el-cscl-batch-scaling-memory.png
:width: 90%
:alt: EL CsCl batch scaling memory

Peak GPU VRAM consumption for batched systems.
```

:::

::::

````

````{tab-item} NH₃

NH₃ molecular clusters packed with Packmol -- the same benchmark system
used in the [NVIDIA Developer Blog](https://developer.nvidia.com/blog/accelerating-ai-powered-chemistry-and-materials-science-simulations-with-nvidia-alchemi-toolkit-ops/).
Ammonia features partial charges on N and H atoms, testing electrostatics
performance on a representative molecular (non-ionic) system.

::::{tab-set}

:::{tab-item} System Size Scaling

Single system growing from 128 to 128k atoms.

```{figure} _static/el-nh3-system-size-scaling-time.png
:width: 90%
:alt: EL NH3 system size time scaling

Execution time scaling for Ewald and PME on NH₃ clusters.
```

---

```{figure} _static/el-nh3-system-size-scaling-throughput.png
:width: 90%
:alt: EL NH3 system size throughput

Throughput (10⁶ atoms/s) vs. system size.
```

---

```{figure} _static/el-nh3-system-size-scaling-memory.png
:width: 90%
:alt: EL NH3 system size memory

Peak GPU VRAM consumption vs. system size.
```

:::

:::{tab-item} Constant Workload Scaling

Fixed total atom count (~128k), varying system size × batch size.

```{figure} _static/el-nh3-constant-workload-scaling-time.png
:width: 90%
:alt: EL NH3 constant workload time

Time per atom at constant ~128k total atoms. X-axis shows system size × batch size.
```

---

```{figure} _static/el-nh3-constant-workload-scaling-throughput.png
:width: 90%
:alt: EL NH3 constant workload throughput

Throughput (10⁶ atoms/s) at constant total workload.
```

---

```{figure} _static/el-nh3-constant-workload-scaling-memory.png
:width: 90%
:alt: EL NH3 constant workload memory

Peak GPU VRAM consumption at constant total workload.
```

:::

:::{tab-item} Batch Scaling

Fixed atoms per system, growing batch size.

```{figure} _static/el-nh3-batch-scaling-time.png
:width: 90%
:alt: EL NH3 batch scaling time
```

---

```{figure} _static/el-nh3-batch-scaling-throughput.png
:width: 90%
:alt: EL NH3 batch scaling throughput
```

---

```{figure} _static/el-nh3-batch-scaling-memory.png
:width: 90%
:alt: EL NH3 batch scaling memory
```

:::

::::

````

`````

## Benchmark Configuration

| Parameter              | Value                          |
| ---------------------- | ------------------------------ |
| Methods                | Ewald summation, PME (cuFFT)   |
| Accuracy levels        | $10^{-4}$, $10^{-6}$          |
| PME spline order       | 4                              |
| Forces                 | Always enabled                 |
| Charge gradients       | Always enabled                 |
| Warmup iterations      | 3                              |
| Timing iterations      | 20                             |

## Running Your Own Benchmarks

Install benchmark dependencies:

```bash
uv sync --all-extras --group benchmark
```

Run electrostatics benchmarks:

```bash
python benchmarks/benchmark_suite.py --benchmark el
python benchmarks/benchmark_suite.py --benchmark el --system cscl --mode system_size
```

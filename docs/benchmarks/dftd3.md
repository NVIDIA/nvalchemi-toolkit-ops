# DFT-D3 Dispersion Benchmarks

Performance benchmarks for DFT-D3(BJ) dispersion corrections across two
chemical systems (CsCl crystal and NH₃ molecular clusters) and three scaling
modes. Two cutoff radii (15 Å and 25 Å) are tested. All timings include
joint energy, forces, and coordination number computation. Timings exclude
neighbor list construction.

```{warning}
These results are intended to be indicative _only_: your actual performance may
vary depending on the atomic system topology, software and hardware configuration
and we encourage users to benchmark on their own systems of interest.
```

## How to Read These Charts

Time per Atom
: Microseconds per atom (μs/atom). Lower is better. Timings exclude neighbor
  list construction and only comprise the DFT-D3 computation.

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
(OOM), particularly for the 25 Å cutoff at 128k atoms.
For small systems with large cutoffs (e.g., 128 atoms at 25 Å), the
simulation cell may be smaller than 2×cutoff, violating the minimum image
convention. These points are included for completeness but should be
interpreted with care.
```

## Performance Results

`````{tab-set}

````{tab-item} CsCl

CsCl crystal supercells with periodic boundary conditions. DFT-D3 does not
include the 6 Å cutoff due to the long-range nature of dispersion interactions.

::::{tab-set}

:::{tab-item} System Size Scaling

Single system growing from 128 to 128k atoms. Shows how DFT-D3 computation
scales with the number of atoms for both cutoff radii.

```{figure} _static/d3-cscl-system-size-scaling-time.png
:width: 90%
:alt: D3 CsCl system size time scaling

Time per atom vs. system size for DFT-D3(BJ) with 15 Å and 25 Å cutoffs.
```

---

```{figure} _static/d3-cscl-system-size-scaling-throughput.png
:width: 90%
:alt: D3 CsCl system size throughput

Throughput (10⁶ atoms/s) vs. system size.
```

---

```{figure} _static/d3-cscl-system-size-scaling-memory.png
:width: 90%
:alt: D3 CsCl system size memory

Peak GPU VRAM consumption vs. system size.
```

:::

:::{tab-item} Constant Workload Scaling

Fixed total atom count (~128k), varying system size × batch size.
Reveals the overhead of batching many small systems vs. fewer large ones
at both cutoff radii.

```{figure} _static/d3-cscl-constant-workload-scaling-time.png
:width: 90%
:alt: D3 CsCl constant workload time

Time per atom at constant ~128k total atoms. X-axis shows system size × batch size.
```

---

```{figure} _static/d3-cscl-constant-workload-scaling-throughput.png
:width: 90%
:alt: D3 CsCl constant workload throughput

Throughput (10⁶ atoms/s) at constant total workload.
```

---

```{figure} _static/d3-cscl-constant-workload-scaling-memory.png
:width: 90%
:alt: D3 CsCl constant workload memory

Peak GPU VRAM consumption at constant total workload.
```

:::

:::{tab-item} Batch Scaling

Fixed atoms per system, growing batch size. Shows how batched DFT-D3
performance scales with increasing number of concurrent systems.

```{figure} _static/d3-cscl-batch-scaling-time.png
:width: 90%
:alt: D3 CsCl batch scaling time

Time per atom for batched systems at different cutoff radii and system sizes.
```

---

```{figure} _static/d3-cscl-batch-scaling-throughput.png
:width: 90%
:alt: D3 CsCl batch scaling throughput

Throughput (10⁶ atoms/s) for batched processing.
```

---

```{figure} _static/d3-cscl-batch-scaling-memory.png
:width: 90%
:alt: D3 CsCl batch scaling memory

Peak GPU VRAM consumption for batched systems.
```

:::

::::

````

````{tab-item} NH₃

NH₃ molecular clusters packed with Packmol -- the same benchmark system
used in the [NVIDIA Developer Blog](https://developer.nvidia.com/blog/accelerating-ai-powered-chemistry-and-materials-science-simulations-with-nvidia-alchemi-toolkit-ops/).
A representative molecular system for evaluating dispersion correction
performance on non-crystalline, non-uniform density systems.

::::{tab-set}

:::{tab-item} System Size Scaling

Single system growing from 128 to 128k atoms.

```{figure} _static/d3-nh3-system-size-scaling-time.png
:width: 90%
:alt: D3 NH3 system size time scaling

Time per atom vs. system size for NH₃ clusters.
```

---

```{figure} _static/d3-nh3-system-size-scaling-throughput.png
:width: 90%
:alt: D3 NH3 system size throughput

Throughput (10⁶ atoms/s) vs. system size.
```

---

```{figure} _static/d3-nh3-system-size-scaling-memory.png
:width: 90%
:alt: D3 NH3 system size memory

Peak GPU VRAM consumption vs. system size.
```

:::

:::{tab-item} Constant Workload Scaling

Fixed total atom count (~128k), varying system size × batch size.

```{figure} _static/d3-nh3-constant-workload-scaling-time.png
:width: 90%
:alt: D3 NH3 constant workload time

Time per atom at constant ~128k total atoms. X-axis shows system size × batch size.
```

---

```{figure} _static/d3-nh3-constant-workload-scaling-throughput.png
:width: 90%
:alt: D3 NH3 constant workload throughput

Throughput (10⁶ atoms/s) at constant total workload.
```

---

```{figure} _static/d3-nh3-constant-workload-scaling-memory.png
:width: 90%
:alt: D3 NH3 constant workload memory

Peak GPU VRAM consumption at constant total workload.
```

:::

:::{tab-item} Batch Scaling

Fixed atoms per system, growing batch size.

```{figure} _static/d3-nh3-batch-scaling-time.png
:width: 90%
:alt: D3 NH3 batch scaling time
```

---

```{figure} _static/d3-nh3-batch-scaling-throughput.png
:width: 90%
:alt: D3 NH3 batch scaling throughput
```

---

```{figure} _static/d3-nh3-batch-scaling-memory.png
:width: 90%
:alt: D3 NH3 batch scaling memory
```

:::

::::

````

`````

## Benchmark Configuration

| Parameter         | Value                     |
| ----------------- | ------------------------- |
| Damping           | Becke-Johnson (BJ)        |
| Cutoff radii      | 15 Å, 25 Å               |
| Terms             | Two-body (C6, C8)         |
| Computed          | Energy + forces + CN      |
| Warmup iterations | 3                         |
| Timing iterations | 20                        |

```{note}
The current implementation computes two-body terms only (C6 and C8).
Three-body Axilrod-Teller-Muto (ATM/C9) contributions are not included.
```

## Running Your Own Benchmarks

Install benchmark dependencies:

```bash
uv sync --all-extras --group benchmark
```

Run D3 benchmarks:

```bash
python benchmarks/benchmark_suite.py --benchmark d3
python benchmarks/benchmark_suite.py --benchmark d3 --system nh3 --mode system_size
```

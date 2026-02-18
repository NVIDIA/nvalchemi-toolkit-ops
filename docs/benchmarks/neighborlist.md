# Neighbor List Benchmarks

Performance benchmarks for neighbor list algorithms across two chemical systems
(CsCl crystal and NH₃ molecular clusters) and three scaling modes.
Both $O(N)$ (cell list) and $O(N^2)$ (naive) algorithms are benchmarked with
multiple cutoff radii (6 Å, 15 Å, 25 Å).

```{warning}
These results are intended to be indicative _only_: your actual performance may
vary depending on the atomic system topology, software and hardware configuration
and we encourage users to benchmark on their own systems of interest.
```

## How to Read These Charts

Time per Atom
: Microseconds per atom (μs/atom). Lower is better. Cell list algorithms
  show $O(N)$ scaling while naive algorithms show $O(N^2)$.

Throughput
: Millions of atoms processed per second (10⁶ atoms/s). Higher is better.
  This metric helps compare efficiency across different system sizes and
  indicates the scaling point where the GPU saturates.

Peak GPU VRAM
: GPU video memory usage (log scale). The dotted line marks H100 80 GB
  capacity. Useful for estimating memory requirements for your target system.

```{note}
All axes use log₂ scaling. The x-axis displays linearized atom counts
(128, 256, 512, ...) for readability, but spacing is logarithmic.
Some data points at the largest sizes may be absent due to GPU out-of-memory
(OOM), particularly for the 25 Å cutoff at 128k+ atoms.
```

## Performance Results

`````{tab-set}

````{tab-item} CsCl

CsCl is an ionic crystal (Cs⁺ at corners, Cl⁻ at body center; 2 atoms
per cubic unit cell) that produces uniform, high-density neighbor
environments -- a stress test for throughput at large system sizes.

::::{tab-set}

:::{tab-item} System Size Scaling

Single system growing from 128 to 128k atoms. Demonstrates how algorithm
complexity affects wall-clock time as the system grows.

```{figure} _static/nl-cscl-system-size-scaling-time.png
:width: 90%
:alt: NL CsCl system size time scaling

Time per atom vs. system size. The $O(N^2)$ naive scaling becomes apparent for
larger systems, while cell list maintains near-constant time per atom.
```

---

```{figure} _static/nl-cscl-system-size-scaling-throughput.png
:width: 90%
:alt: NL CsCl system size throughput

Throughput (10⁶ atoms/s) vs. system size. Cell list maintains high throughput
even for very large systems.
```

---

```{figure} _static/nl-cscl-system-size-scaling-memory.png
:width: 90%
:alt: NL CsCl system size memory

Peak GPU VRAM consumption vs. system size.
```

:::

:::{tab-item} Constant Workload Scaling

Fixed total atom count (~128k), varying system size × batch size.
Reveals the overhead of batching many small systems vs. fewer large ones.

```{figure} _static/nl-cscl-constant-workload-scaling-time.png
:width: 90%
:alt: NL CsCl constant workload time
```

---

```{figure} _static/nl-cscl-constant-workload-scaling-throughput.png
:width: 90%
:alt: NL CsCl constant workload throughput
```

---

```{figure} _static/nl-cscl-constant-workload-scaling-memory.png
:width: 90%
:alt: NL CsCl constant workload memory
```

:::

:::{tab-item} Batch Scaling

Fixed atoms per system, growing batch size. Useful for ML workflows that
process many small molecules simultaneously on a single GPU.

```{figure} _static/nl-cscl-batch-scaling-time.png
:width: 90%
:alt: NL CsCl batch scaling time

Time per atom decreases as batch size grows, showing GPU utilization improvement.
```

---

```{figure} _static/nl-cscl-batch-scaling-throughput.png
:width: 90%
:alt: NL CsCl batch scaling throughput

Throughput (10⁶ atoms/s) for batched processing. Different lines show
different cutoff radii and system sizes.
```

---

```{figure} _static/nl-cscl-batch-scaling-memory.png
:width: 90%
:alt: NL CsCl batch scaling memory

Peak GPU VRAM consumption for batched systems.
```

:::

::::

````

````{tab-item} NH₃

NH₃ molecular clusters packed with Packmol -- the same benchmark system
used in the [NVIDIA Developer Blog](https://developer.nvidia.com/blog/accelerating-ai-powered-chemistry-and-materials-science-simulations-with-nvidia-alchemi-toolkit-ops/).
A representative molecular system with non-uniform density and varying box sizes.

::::{tab-set}

:::{tab-item} System Size Scaling

Single system growing from 128 to 128k atoms.

```{figure} _static/nl-nh3-system-size-scaling-time.png
:width: 90%
:alt: NL NH3 system size time scaling

Time per atom vs. system size for NH₃ clusters.
```

---

```{figure} _static/nl-nh3-system-size-scaling-throughput.png
:width: 90%
:alt: NL NH3 system size throughput

Throughput (10⁶ atoms/s) vs. system size.
```

---

```{figure} _static/nl-nh3-system-size-scaling-memory.png
:width: 90%
:alt: NL NH3 system size memory

Peak GPU VRAM consumption vs. system size.
```

:::

:::{tab-item} Constant Workload Scaling

Fixed total atom count (~128k), varying system size × batch size.

```{figure} _static/nl-nh3-constant-workload-scaling-time.png
:width: 90%
:alt: NL NH3 constant workload time
```

---

```{figure} _static/nl-nh3-constant-workload-scaling-throughput.png
:width: 90%
:alt: NL NH3 constant workload throughput
```

---

```{figure} _static/nl-nh3-constant-workload-scaling-memory.png
:width: 90%
:alt: NL NH3 constant workload memory
```

:::

:::{tab-item} Batch Scaling

Fixed atoms per system, growing batch size.

```{figure} _static/nl-nh3-batch-scaling-time.png
:width: 90%
:alt: NL NH3 batch scaling time
```

---

```{figure} _static/nl-nh3-batch-scaling-throughput.png
:width: 90%
:alt: NL NH3 batch scaling throughput
```

---

```{figure} _static/nl-nh3-batch-scaling-memory.png
:width: 90%
:alt: NL NH3 batch scaling memory
```

:::

::::

````

`````

## Benchmark Configuration

| Parameter         | Value                          |
| ----------------- | ------------------------------ |
| Algorithms        | Naive $O(N^2)$, Cell list $O(N)$ |
| Cutoff radii      | 6 Å, 15 Å, 25 Å               |
| Warmup iterations | 3                              |
| Timing iterations | 20                             |
| Dtype             | float32                        |

## Running Your Own Benchmarks

Install benchmark dependencies:

```bash
uv sync --all-extras --group benchmark
```

Run the full neighbor list benchmark suite:

```bash
python benchmarks/benchmark_suite.py --benchmark nl
```

Or run individual scaling modes:

```bash
python benchmarks/benchmark_suite.py --benchmark nl --system nh3 --mode system_size
```

Results are saved as CSV files and plots are generated automatically.

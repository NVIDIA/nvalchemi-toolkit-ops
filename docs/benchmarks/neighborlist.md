# Neighbor List Benchmarks

Performance benchmarks for neighbor list algorithms across two chemical systems
(CsCl crystal and NH₃ molecular clusters) and three scaling modes.

```{warning}
These results are indicative only: actual performance varies depending on
system topology, software version, and hardware. We encourage users to
benchmark on their own systems of interest.
```

## How to Read These Charts

Time per Atom
: Microseconds per atom (μs/atom). Lower is better. Cell list shows O(N)
  scaling while naive shows O(N²).

Throughput
: Millions of atoms processed per second (10⁶ atoms/s). Higher is better.

Peak Memory
: GPU VRAM usage. The dotted line marks H100 80 GB capacity.

## Performance Results

`````{tab-set}

````{tab-item} CsCl

::::{tab-set}

:::{tab-item} System Size Scaling

```{figure} _static/nl-cscl-system-size-scaling-time.png
:width: 90%
```

```{figure} _static/nl-cscl-system-size-scaling-throughput.png
:width: 90%
```

```{figure} _static/nl-cscl-system-size-scaling-memory.png
:width: 90%
```

:::

:::{tab-item} Constant Workload Scaling

```{figure} _static/nl-cscl-constant-workload-scaling-time.png
:width: 90%
```

```{figure} _static/nl-cscl-constant-workload-scaling-throughput.png
:width: 90%
```

```{figure} _static/nl-cscl-constant-workload-scaling-memory.png
:width: 90%
```

:::

:::{tab-item} Batch Scaling

```{figure} _static/nl-cscl-batch-scaling-time.png
:width: 90%
```

```{figure} _static/nl-cscl-batch-scaling-throughput.png
:width: 90%
```

```{figure} _static/nl-cscl-batch-scaling-memory.png
:width: 90%
```

:::

::::

````

````{tab-item} NH₃

::::{tab-set}

:::{tab-item} System Size Scaling

```{figure} _static/nl-nh3-system-size-scaling-time.png
:width: 90%
```

```{figure} _static/nl-nh3-system-size-scaling-throughput.png
:width: 90%
```

```{figure} _static/nl-nh3-system-size-scaling-memory.png
:width: 90%
```

:::

:::{tab-item} Constant Workload Scaling

```{figure} _static/nl-nh3-constant-workload-scaling-time.png
:width: 90%
```

```{figure} _static/nl-nh3-constant-workload-scaling-throughput.png
:width: 90%
```

```{figure} _static/nl-nh3-constant-workload-scaling-memory.png
:width: 90%
```

:::

:::{tab-item} Batch Scaling

```{figure} _static/nl-nh3-batch-scaling-time.png
:width: 90%
```

```{figure} _static/nl-nh3-batch-scaling-throughput.png
:width: 90%
```

```{figure} _static/nl-nh3-batch-scaling-memory.png
:width: 90%
```

:::

::::

````

`````

## Running Your Own Benchmarks

Install benchmark dependencies:

```bash
# Using uv (recommended for development)
uv sync --all-extras --group benchmark

# Or using pip
pip install nvalchemi-toolkit-ops
pip install -r benchmarks/benchmark-requires.txt
```

Run the full benchmark suite (includes plotting):

```bash
python benchmarks/benchmark_suite.py --benchmark nl
```

Or run individual scaling modes:

```bash
python benchmarks/benchmark_suite.py --benchmark nl --system nh3 --mode system_size
```

To regenerate plots from existing CSV results:

```bash
python benchmarks/benchmark_suite.py --plot-only benchmarks/benchmark-results/run_YYYY-MM-DD/
```

Results are saved as CSV files and plots are generated automatically.
To update the documentation plots, copy your CSVs to
`docs/benchmarks/benchmark_results/` and rebuild docs with `make docs`.

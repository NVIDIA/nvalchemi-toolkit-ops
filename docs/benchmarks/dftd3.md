# DFT-D3 Dispersion Benchmarks

Performance benchmarks for DFT-D3(BJ) dispersion corrections across two
chemical systems (CsCl crystal and NH₃ molecular clusters) with two cutoff
radii (15 Å and 25 Å).

```{warning}
These results are indicative only: actual performance varies depending on
system topology, software version, and hardware.
```

## Performance Results

`````{tab-set}

````{tab-item} CsCl

::::{tab-set}

:::{tab-item} System Size Scaling

```{figure} _static/d3-cscl-system-size-scaling-time.png
:width: 90%
```

```{figure} _static/d3-cscl-system-size-scaling-throughput.png
:width: 90%
```

```{figure} _static/d3-cscl-system-size-scaling-memory.png
:width: 90%
```

:::

:::{tab-item} Constant Workload Scaling

```{figure} _static/d3-cscl-constant-workload-scaling-time.png
:width: 90%
```

```{figure} _static/d3-cscl-constant-workload-scaling-throughput.png
:width: 90%
```

```{figure} _static/d3-cscl-constant-workload-scaling-memory.png
:width: 90%
```

:::

:::{tab-item} Batch Scaling

```{figure} _static/d3-cscl-batch-scaling-time.png
:width: 90%
```

```{figure} _static/d3-cscl-batch-scaling-throughput.png
:width: 90%
```

```{figure} _static/d3-cscl-batch-scaling-memory.png
:width: 90%
```

:::

::::

````

````{tab-item} NH₃

::::{tab-set}

:::{tab-item} System Size Scaling

```{figure} _static/d3-nh3-system-size-scaling-time.png
:width: 90%
```

```{figure} _static/d3-nh3-system-size-scaling-throughput.png
:width: 90%
```

```{figure} _static/d3-nh3-system-size-scaling-memory.png
:width: 90%
```

:::

:::{tab-item} Constant Workload Scaling

```{figure} _static/d3-nh3-constant-workload-scaling-time.png
:width: 90%
```

```{figure} _static/d3-nh3-constant-workload-scaling-throughput.png
:width: 90%
```

```{figure} _static/d3-nh3-constant-workload-scaling-memory.png
:width: 90%
```

:::

:::{tab-item} Batch Scaling

```{figure} _static/d3-nh3-batch-scaling-time.png
:width: 90%
```

```{figure} _static/d3-nh3-batch-scaling-throughput.png
:width: 90%
```

```{figure} _static/d3-nh3-batch-scaling-memory.png
:width: 90%
```

:::

::::

````

`````

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

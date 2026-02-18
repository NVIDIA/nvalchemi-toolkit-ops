# Electrostatics Benchmarks

Performance benchmarks for Ewald summation and Particle Mesh Ewald (PME)
across two chemical systems (CsCl crystal and NH₃ molecular clusters) at
two accuracy levels (10⁻⁴ and 10⁻⁶). All results include forces and charge
gradient computation.

```{warning}
These results are indicative only: actual performance varies depending on
system topology, software version, and hardware.
```

## Performance Results

`````{tab-set}

````{tab-item} CsCl

::::{tab-set}

:::{tab-item} System Size Scaling

```{figure} _static/el-cscl-system-size-scaling-time.png
:width: 90%
```

```{figure} _static/el-cscl-system-size-scaling-throughput.png
:width: 90%
```

```{figure} _static/el-cscl-system-size-scaling-memory.png
:width: 90%
```

:::

:::{tab-item} Constant Workload Scaling

```{figure} _static/el-cscl-constant-workload-scaling-time.png
:width: 90%
```

```{figure} _static/el-cscl-constant-workload-scaling-throughput.png
:width: 90%
```

```{figure} _static/el-cscl-constant-workload-scaling-memory.png
:width: 90%
```

:::

:::{tab-item} Batch Scaling

```{figure} _static/el-cscl-batch-scaling-time.png
:width: 90%
```

```{figure} _static/el-cscl-batch-scaling-throughput.png
:width: 90%
```

```{figure} _static/el-cscl-batch-scaling-memory.png
:width: 90%
```

:::

::::

````

````{tab-item} NH₃

::::{tab-set}

:::{tab-item} System Size Scaling

```{figure} _static/el-nh3-system-size-scaling-time.png
:width: 90%
```

```{figure} _static/el-nh3-system-size-scaling-throughput.png
:width: 90%
```

```{figure} _static/el-nh3-system-size-scaling-memory.png
:width: 90%
```

:::

:::{tab-item} Constant Workload Scaling

```{figure} _static/el-nh3-constant-workload-scaling-time.png
:width: 90%
```

```{figure} _static/el-nh3-constant-workload-scaling-throughput.png
:width: 90%
```

```{figure} _static/el-nh3-constant-workload-scaling-memory.png
:width: 90%
```

:::

:::{tab-item} Batch Scaling

```{figure} _static/el-nh3-batch-scaling-time.png
:width: 90%
```

```{figure} _static/el-nh3-batch-scaling-throughput.png
:width: 90%
```

```{figure} _static/el-nh3-batch-scaling-memory.png
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

Run electrostatics benchmarks:

```bash
python benchmarks/benchmark_suite.py --benchmark el
python benchmarks/benchmark_suite.py --benchmark el --system cscl --mode system_size
```

# Benchmarks

Performance benchmarks for ALCHEMI Toolkit-Ops kernels. Currently, results
are static and cached but we intend to evolve to CI-generated benchmark
results gradually to cover different NVIDIA architectures, benchmark
systems, and so on.

## Available Benchmarks

```{toctree}
:maxdepth: 1

neighborlist
electrostatics
dftd3
```

## About These Benchmarks

Benchmarks are intended to be indicative of `nvalchemiops` performance under
a specific set of criteria; actual performance may differ depending
on a number of factors including but not limited to structure/system
topology, GPU architecture, driver and firmware versions.

## Benchmark Methodology

All benchmarks follow these principles:

- **Tensor allocation excluded**: Only _relevant_ kernel execution time
is measured, i.e. excluding neighbor lists and preprocessing if they
are not part of the benchmark.
- **Warm-up runs**: Multiple warm-up iterations to ensure kernels compile
overhead is removed, and that noise from cache effects are minimized.
- **Statistical sampling**: Multiple timing runs with median time,
maximum memory utilization, and throughput aggregated for reporting.
- **Error handling**: OOM results are included.
- **Consistent inputs**: Same cutoff, lattice type, and parameters across runs

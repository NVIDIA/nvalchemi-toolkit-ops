# Benchmarks

Performance benchmarks for ALCHEMI Toolkit-Ops kernels across two chemical
systems (CsCl crystal and NH₃ molecular clusters) and three scaling modes.
For a high-level overview of these benchmarks in context, see the
[NVIDIA Developer Blog post](https://developer.nvidia.com/blog/accelerating-ai-powered-chemistry-and-materials-science-simulations-with-nvidia-alchemi-toolkit-ops/).
The figures below expand on the blog with additional systems, cutoff radii,
and per-metric breakdowns.

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

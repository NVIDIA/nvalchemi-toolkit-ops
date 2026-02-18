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

## Hardware and Software

Results on this page were collected on the following system:

| Component | Version |
|-----------|---------|
| GPU | NVIDIA H100 80 GB HBM3 |
| CUDA | 12.8 |
| PyTorch | 2.10.0+cu128 |
| nvalchemi-toolkit-ops | 0.2.0 |
| Warp | 1.11.0 |
| Python | 3.12.3 |
| OS | Linux 5.15.0-1063-nvidia |

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

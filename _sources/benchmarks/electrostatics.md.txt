# Electrostatics Benchmarks

This page presents benchmark results for electrostatic interaction methods including
Ewald summation and Particle Mesh Ewald (PME) across different GPU hardware. Results
show the scaling behavior with increasing system size for periodic systems, including
both single-system and batched computations.

```{warning}
These results are intended to be indicative _only_: your actual performance may
vary depending on the atomic system topology, software and hardware configuration
and we encourage users to benchmark on their own systems of interest.
```

## How to Read These Charts

Time Scaling
: Median execution time (ms) vs. system size. Lower is better. Timings include
  both real-space and reciprocal-space contributions when running "full" mode.

Throughput
: Atoms processed per millisecond. Higher is better. This indicates the scaling
  point where the GPU saturates.

Memory
: Peak GPU memory usage (MB) vs. system size. This is particularly useful
  for estimating/gauging memory requirements for your system.

## Ewald Summation

The Ewald summation method splits the Coulomb interaction into real-space and
reciprocal-space components. This is the traditional $O(N^{3/2})$ to $O(N^2)$ method
depending on parameter choices.

Scaling of single and batched Ewald computation with the `nvalchemiops` backend.
Shows how performance scales with different batch sizes.

### Time Scaling

```{figure} _static/electrostatics_scaling_ewald_nvalchemiops_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Ewald nvalchemiops time scaling

Execution time scaling for single and batched systems.
```

### Throughput

```{figure} _static/electrostatics_throughput_ewald_nvalchemiops_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Ewald nvalchemiops throughput

Throughput (atoms/ms) for single and batched systems.
```

### Memory Usage

```{figure} _static/electrostatics_memory_ewald_nvalchemiops_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Ewald nvalchemiops memory usage

Peak GPU memory consumption for single and batched systems.
```

## Particle Mesh Ewald (PME)

PME achieves $O(N \log N)$ scaling by using FFTs for the reciprocal-space contribution.
This is the recommended method for large systems.

Scaling of single and batched PME computation with the `nvalchemiops` backend.
Shows how performance scales with different batch sizes.

### Time Scaling

```{figure} _static/electrostatics_scaling_pme_nvalchemiops_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: PME nvalchemiops time scaling

Execution time scaling for single and batched systems.
```

### Throughput

```{figure} _static/electrostatics_throughput_pme_nvalchemiops_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: PME nvalchemiops throughput

Throughput (atoms/ms) for single and batched systems.
```

### Memory Usage

```{figure} _static/electrostatics_memory_pme_nvalchemiops_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: PME nvalchemiops memory usage

Peak GPU memory consumption for single and batched systems.
```

## Hardware Information

**GPU**: NVIDIA H100 80GB HBM3

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| System Type | FCC crystal lattice with periodic boundaries |
| Neighbor List | Cell list algorithm ($O(N)$ scaling) |
| Warmup Iterations | 3 |
| Timing Iterations | 10 |
| Precision | `float32` |

### Ewald/PME Parameters

Parameters are automatically estimated using accuracy-based parameter estimation
targeting $10^{-6}$ relative accuracy:

| Parameter | Description |
|-----------|-------------|
| `alpha` | Ewald splitting parameter (auto-estimated) |
| `k_cutoff` | Reciprocal-space cutoff for Ewald (auto-estimated) |
| `real_space_cutoff` | Real-space cutoff distance (auto-estimated) |
| `mesh_dimensions` | PME mesh grid size (auto-estimated) |
| `spline_order` | B-spline interpolation order (4) |

## Interpreting Results

`total_atoms`
: Total number of atoms in the supercell (or across all batched systems).

`batch_size`
: Number of systems processed simultaneously (1 for single-system mode).

`method`
: The electrostatics method used (`ewald` or `pme`).

`backend`
: The computational backend (`nvalchemiops` or `torchpme`).

`component`
: Which part of the calculation was benchmarked (`real`, `reciprocal`, or `full`).

`compute_forces`
: Whether forces were computed in addition to energies.

`median_time_ms`
: Median execution time in milliseconds (lower is better).

`peak_memory_mb`
: Peak GPU memory usage in megabytes.

```{note}
Timings include the full electrostatics calculation (real-space + reciprocal-space
for "full" mode). Neighbor list construction is excluded from timings.
```

## Running Your Own Benchmarks

To generate benchmark results for your hardware:

```bash
cd benchmarks/interactions/electrostatics
python benchmark_electrostatics.py \
    --config benchmark_config.yaml \
    --backend nvalchemiops \
    --method both \
    --output-dir ../../../docs/benchmarks/benchmark_results
```

### Options

`--backend nvalchemiops`
: Use the `nvalchemiops` backend (default).

`--method {ewald,pme,both}`
: Select electrostatics method (default: `both`).

`--gpu-sku <name>`
: Override GPU SKU name for output files (default: auto-detect).

`--config <path>`
: Path to YAML configuration file.

Results will be saved as CSV files and plots will be automatically generated
during the next documentation build.

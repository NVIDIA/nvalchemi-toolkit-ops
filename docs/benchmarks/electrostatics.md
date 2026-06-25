# Electrostatics Benchmarks

This page presents benchmark results for electrostatic interaction methods including
Ewald summation, Particle Mesh Ewald (PME), and Damped Shifted Force (DSF) across
different GPU hardware. Results show the scaling behavior with increasing system size
for both single-system and batched computations.

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

::::{tab-set}

:::{tab-item} Backend Comparison

Simple comparison of single (non-batched) system computations between backends,
where we scale up the size of the supercell.

### Time Scaling

```{figure} _static/electrostatics_scaling_ewald_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Ewald backend time comparison

Median execution time comparison between backends for single systems.
```

### Throughput

```{figure} _static/electrostatics_throughput_ewald_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Ewald backend throughput comparison

Throughput (atoms/ms) comparison between backends. Higher values indicate better performance.
```

### Memory Usage

```{figure} _static/electrostatics_memory_ewald_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Ewald backend memory comparison

Peak GPU memory consumption comparison between backends. Lower is better,
indicating that the backend has lower memory requirements.
```

:::

:::{tab-item} nvalchemiops (Torch)

Scaling of single and batched Ewald computation with the `nvalchemiops` Torch backend.
Shows how performance scales with different batch sizes.

### Time Scaling

```{figure} _static/electrostatics_scaling_ewald_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Ewald nvalchemiops (Torch) time scaling

Execution time scaling for single and batched systems.
```

### Throughput

```{figure} _static/electrostatics_throughput_ewald_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Ewald nvalchemiops (Torch) throughput

Throughput (atoms/ms) for single and batched systems.
```

### Memory Usage

```{figure} _static/electrostatics_memory_ewald_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Ewald nvalchemiops (Torch) memory usage

Peak GPU memory consumption for single and batched systems.
```

:::

:::{tab-item} nvalchemiops (JAX)

Scaling of single and batched Ewald computation with the `nvalchemiops` JAX backend.
Shows how performance scales with different batch sizes.

### Time Scaling

```{figure} _static/electrostatics_scaling_ewald_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Ewald nvalchemiops (JAX) time scaling

Execution time scaling for single and batched systems.
```

### Throughput

```{figure} _static/electrostatics_throughput_ewald_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Ewald nvalchemiops (JAX) throughput

Throughput (atoms/ms) for single and batched systems.
```

### Memory Usage

```{figure} _static/electrostatics_memory_ewald_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: Ewald nvalchemiops (JAX) memory usage

Peak GPU memory consumption for single and batched systems.
```

:::

::::

## Particle Mesh Ewald (PME)

PME achieves $O(N \log N)$ scaling by using FFTs for the reciprocal-space contribution.
This is the recommended method for large systems.

::::{tab-set}

:::{tab-item} Backend Comparison

Simple comparison of single (non-batched) system computations between backends,
where we scale up the size of the supercell.

### Time Scaling

```{figure} _static/electrostatics_scaling_pme_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: PME backend time comparison

Median execution time comparison between backends for single systems.
```

### Throughput

```{figure} _static/electrostatics_throughput_pme_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: PME backend throughput comparison

Throughput (atoms/ms) comparison between backends. Higher values indicate better performance.
```

### Memory Usage

```{figure} _static/electrostatics_memory_pme_comparison_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: PME backend memory comparison

Peak GPU memory consumption comparison between backends. Lower is better,
indicating that the backend has lower memory requirements.
```

:::

:::{tab-item} nvalchemiops (Torch)

Scaling of single and batched PME computation with the `nvalchemiops` Torch backend.
Shows how performance scales with different batch sizes.

### Time Scaling

```{figure} _static/electrostatics_scaling_pme_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: PME nvalchemiops (Torch) time scaling

Execution time scaling for single and batched systems.
```

### Throughput

```{figure} _static/electrostatics_throughput_pme_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: PME nvalchemiops (Torch) throughput

Throughput (atoms/ms) for single and batched systems.
```

### Memory Usage

```{figure} _static/electrostatics_memory_pme_torch_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: PME nvalchemiops (Torch) memory usage

Peak GPU memory consumption for single and batched systems.
```

:::

:::{tab-item} nvalchemiops (JAX)

Scaling of single and batched PME computation with the `nvalchemiops` JAX backend.
Shows how performance scales with different batch sizes.

### Time Scaling

```{figure} _static/electrostatics_scaling_pme_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: PME nvalchemiops (JAX) time scaling

Execution time scaling for single and batched systems.
```

### Throughput

```{figure} _static/electrostatics_throughput_pme_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: PME nvalchemiops (JAX) throughput

Throughput (atoms/ms) for single and batched systems.
```

### Memory Usage

```{figure} _static/electrostatics_memory_pme_jax_h100-80gb-hbm3.png
:width: 90%
:align: center
:alt: PME nvalchemiops (JAX) memory usage

Peak GPU memory consumption for single and batched systems.
```

:::

::::

## Damped Shifted Force (DSF)

The DSF method is a purely real-space, pairwise $O(N)$ electrostatic summation.
Unlike Ewald and PME, DSF has no reciprocal-space component -- benchmarks measure
the full calculation directly. DSF supports both CSR neighbor list and dense
neighbor matrix formats, and can be compared against a pure PyTorch reference
implementation (`torch_dsf` backend).

```{note}
DSF benchmark plots will be added in a future update. Performance and accuracy
benchmark scripts are available in
`benchmarks/interactions/electrostatics/` and can be run on your hardware
using the instructions below.
```

### DSF Parameters

| Parameter | Value |
| --------- | ----- |
| `cutoff` | 12.0 |
| `alpha` | 0.2 |
| `component` | `full` (real-space only, no reciprocal-space split) |

## Hardware Information

**GPU**: NVIDIA H100 80GB HBM3

## Benchmark Configuration

| Parameter | Value |
| --------- | ----- |
| System Type | BCC/CsCl crystal lattices with periodic boundaries |
| Neighbor List | Cell list algorithm ($O(N)$ scaling) |
| Warmup Iterations | 5 |
| Timing Iterations | 10 |
| Precision | `float32` |

### Ewald/PME Parameters

Parameters are automatically estimated using accuracy-based parameter estimation
targeting $10^{-4}$ relative accuracy by default:

| Parameter | Description |
| --------- | ----------- |
| `alpha` | Ewald splitting parameter (auto-estimated) |
| `k_cutoff` | Reciprocal-space cutoff for Ewald (auto-estimated) |
| `real_space_cutoff` | Real-space cutoff distance (pinned to 12.0 Å by the default config; auto-estimated when unset) |
| `mesh_dimensions` | PME mesh grid size (auto-estimated) |
| `spline_order` | B-spline interpolation order (orders 1-6 supported; 4 by default) |

## Interpreting Results

`total_atoms`
: Total number of atoms in the supercell (or across all batched systems).

`batch_size`
: Number of systems processed simultaneously (1 for single-system mode).

`method`
: The electrostatics method used (`ewald`, `ewald_slab`, `pme`, `pme_slab`, or `dsf`).

`backend`
: The computational backend (`torch`, `jax`, `torchpme`, or `torch_dsf`).

`component`
: Which part of the calculation was benchmarked (`real`, `reciprocal`, or `full`).

`derivative_contract`
: Whether nvalchemiops Ewald/PME rows use the default `energy_autograd`
  contract or the opt-in deprecated `legacy_direct` output flags.

`workload`
: Runtime row type: `forward`, `backward`, `double_backward`,
  `legacy_direct`, or `autograd_reference`.

`compute_forces`
: Whether forces were computed in addition to energies.

`compute_virial`
: Whether virials were computed in addition to energies.

`neighbor_format`
: Neighbor-list representation used by the benchmark row.

`dtype`
: Floating-point dtype used for the row.

`median_time_ms`
: Median execution time in milliseconds (lower is better).

`peak_memory_mb`
: Peak GPU memory usage in megabytes.

`compile_ms`, `warp_compile_ms`, `framework_compile_ms`
: Timing columns for benchmark warmup, Warp NVRTC, and framework compile/JIT
  costs when available.

`framework`
: Framework execution mode, such as eager, `torch.compile`, or `jax.jit`.

`success`, `error`, `error_type`
: Row status and failure details for skipped or failed benchmark cases.

Requested compile modes are strict. A failed `torch.compile` or JAX JIT setup is
recorded as a failed CSV row with `success=False`, `framework`, `error`, and
`error_type` populated; the benchmark does not silently fall back to eager timing
for that row.

```{note}
Timings include the full electrostatics calculation (real-space + reciprocal-space
for "full" mode). Neighbor list construction is excluded from timings.
```

## Running Your Own Benchmarks

To generate benchmark results for your hardware:

### `torch` Backend (default)

```bash
cd benchmarks/interactions/electrostatics
python benchmark_electrostatics.py \
    --config benchmark_config.yaml \
    --backend torch \
    --method all \
    --output-dir ../../../docs/benchmarks/benchmark_results
```

### `jax` Backend

```bash
cd benchmarks/interactions/electrostatics
python benchmark_electrostatics.py \
    --config benchmark_config.yaml \
    --backend jax \
    --method both \
    --output-dir ../../../docs/benchmarks/benchmark_results
```

### `torchpme` Backend

```bash
cd benchmarks/interactions/electrostatics
python benchmark_electrostatics.py \
    --config benchmark_config.yaml \
    --backend torchpme \
    --method both \
    --output-dir ../../../docs/benchmarks/benchmark_results
```

### Options

`--backend {torch,jax,torchpme,torch_dsf,both}`
: Computational backend (default: `torch`). `both` dispatches per-method:
  `torch` + `torchpme` for Ewald/PME, `torch` for Ewald/PME slab
  methods, and `torch` + `torch_dsf` for DSF.

`--method {ewald,ewald_slab,pme,pme_slab,dsf,multipole_ewald,multipole_pme,both,all}`
: Electrostatics method. When omitted, the YAML `methods:` list controls; if
  that is empty, the script defaults to Ewald + PME. `both` = Ewald + PME.
  `all` = Ewald + Ewald slab + PME + PME slab + DSF + multipole Ewald +
  multipole PME. The `multipole_ewald` / `multipole_pme` methods are torch-only
  and take an extra `--l-max` flag.

`--l-max {0,1,2}`
: Maximum multipole order for the `multipole_*` methods (default: `1`):
  `0` = charges, `1` = + dipoles, `2` = + quadrupoles. Ignored by the
  point-charge methods.

`--neighbor-format {list,matrix,both}`
: Neighbor format for DSF benchmarks (default: `list`). Ewald/PME always use matrix.

`--dtype {float32,float64}`
: Override dtype from config file.

`--real-space-cutoff REAL_SPACE_CUTOFF`
: Override the PME real-space cutoff in Angstrom.

`--accuracy ACCURACY`
: Override the target relative force accuracy used by parameter estimation.

`--torch-compile`
: Time torch backends through `torch.compile(fullgraph=True)` and record
  framework compile costs. When the flag is omitted, the YAML `compile:` field
  controls the default.

`--no-torch-compile`
: Disable `torch.compile` for torch backends, overriding YAML `compile: true`.

`--derivative-contract {energy_autograd,legacy_direct}`
: Select the nvalchemiops Ewald/PME derivative contract. The default
  `energy_autograd` path benchmarks energy-only calls plus framework autograd
  force/stress workloads, including Torch double-backward rows when derivatives
  are requested. `legacy_direct` benchmarks deprecated direct-output flags and
  suppresses their migration warnings inside the explicit legacy rows.

`--gpu-sku <name>`
: Override GPU SKU name for output files (default: auto-detect).

`--config <path>`
: Path to YAML configuration file.

#### DSF-Only Benchmark

```bash
python benchmark_electrostatics.py \
    --config benchmark_config.yaml \
    --backend both \
    --method dsf \
    --neighbor-format list
```

#### Slab Benchmarks

```bash
# Ewald slab (PyTorch)
python benchmark_electrostatics.py \
    --config benchmark_config.yaml \
    --backend torch \
    --method ewald_slab

# Ewald slab (JAX)
python benchmark_electrostatics.py \
    --config benchmark_config.yaml \
    --backend jax \
    --method ewald_slab

# PME slab (PyTorch)
python benchmark_electrostatics.py \
    --config benchmark_config.yaml \
    --backend torch \
    --method pme_slab

# PME slab (JAX)
python benchmark_electrostatics.py \
    --config benchmark_config.yaml \
    --backend jax \
    --method pme_slab
```

Slab benchmarks run the full high-level Ewald or PME path with
`slab_correction=True` and two periodic directions.

Results are saved as CSV files. Documented benchmark plots are generated from
supported benchmark CSVs during the next documentation build.

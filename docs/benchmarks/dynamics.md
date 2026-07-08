# Dynamics Benchmarks

This page presents benchmark results for molecular dynamics (MD) integrators and
geometry optimization methods using the nvalchemiops GPU-accelerated implementations.
Results show scaling behavior for both single-system and batched simulations across
different system sizes using Lennard-Jones argon systems.

```{warning}
These results are intended to be indicative _only_: your actual performance may
vary depending on the atomic system topology, software and hardware configuration
and we encourage users to benchmark on their own systems of interest.
```

## How to Read These Charts

Time Scaling
: Average time per MD/optimization step (ms) vs. system size. Lower is better.
  For batched runs, this is the time to process all systems in the batch.

Throughput
: Atom-steps processed per second. Higher is better. For batched systems, this
  represents the total number of atoms across all systems in the batch multiplied
  by the number of steps per second.

Ensemble
: MD ensemble type - NVE (constant energy), NVT (constant temperature), NPT
  (constant pressure-temperature), or NPH (constant pressure-enthalpy).

Batch Size
: Number of independent systems processed simultaneously. Batch size of 1 represents
  single-system mode.

## Molecular Dynamics (MD)

GPU-accelerated MD integrators using NVIDIA Warp kernels with optimized neighbor lists.
Supports various ensembles including microcanonical (NVE), canonical (NVT), and
isobaric-isothermal (NPT).

### Single-System MD

Performance for single molecular dynamics systems showing how throughput scales
with system size.

#### Time Scaling

```{figure} _static/dynamics_md_single_nvalchemiops_scaling_h100.png
:width: 90%
:align: center
:alt: MD single-system time scaling

Average step time vs. system size for single-system MD integrators.
```

#### Throughput

```{figure} _static/dynamics_md_single_nvalchemiops_throughput_h100.png
:width: 90%
:align: center
:alt: MD single-system throughput

Throughput (atom-steps/s) for single-system MD integrators.
```

### Available Integrators

Velocity Verlet (NVE)
: Symplectic integrator that conserves total energy. Excellent stability for
  constant energy simulations. Standard choice for microcanonical ensemble.

Langevin (NVT)
: Stochastic dynamics using the BAOAB splitting scheme for accurate temperature
  control. Maintains canonical ensemble through friction and random forces.

Nose-Hoover Chain (NVT)
: Deterministic thermostat using extended system variables. Provides rigorous
  canonical sampling without stochastic forces.

NPT Integrator
: Isobaric-isothermal ensemble allowing cell fluctuations to maintain constant
  pressure and temperature. Uses Nose-Hoover chains for temperature control and
  barostat for pressure control.

NPH Integrator
: Isobaric-enthalpic ensemble with constant pressure. Similar to NPT but without
  temperature control.

## Geometry Optimization

GPU-accelerated FIRE and FIRE2 (Fast Inertial Relaxation Engine) optimizers for
efficient energy minimization. Both adapt timestep and velocity-force mixing for
robust convergence on diverse energy landscapes. FIRE2 (Guénolé et al., 2020)
introduces a deferred half-step and modified velocity mixing for improved
convergence behavior.

### Single-System Optimization

Performance for single-system geometry optimization showing convergence speed
and computational efficiency.

#### Time Scaling

```{figure} _static/dynamics_opt_single_nvalchemiops_scaling_h100.png
:width: 90%
:align: center
:alt: Optimization single-system scaling

Average step time vs. system size for FIRE optimizer.
```

#### Throughput

```{figure} _static/dynamics_opt_single_nvalchemiops_throughput_h100.png
:width: 90%
:align: center
:alt: Optimization single-system throughput

Throughput (atom-steps/s) during geometry optimization.
```

### Batched Optimization

Performance for batched optimization showing how multiple structures can be
relaxed simultaneously for efficient saddle point searches, transition state
finding, or structural screening.

#### Time Scaling

```{figure} _static/dynamics_opt_batch_nvalchemiops_scaling_h100.png
:width: 90%
:align: center
:alt: Optimization batched scaling

Average step time for batched FIRE optimization.
```

#### Throughput

```{figure} _static/dynamics_opt_batch_nvalchemiops_throughput_h100.png
:width: 90%
:align: center
:alt: Optimization batched throughput

Total throughput (atom-steps/s) for batched optimization.
```

### FIRE Algorithm Features

**Adaptive Timestep:**

- Increases timestep when optimization is progressing smoothly (power P = F · v > 0)
- Decreases timestep and resets velocities when moving uphill (P < 0)
- Parameters: `dt_max` (10.0 fs), `f_inc` (1.1), `f_dec` (0.5)

**Velocity Mixing:**

- Mixes velocity with force direction: v → (1-α)v + α|v|F̂
- Decreases mixing parameter α over time for faster convergence
- Parameter: `f_alpha` (0.99)

**Maximum Displacement:**

- Limits atomic displacement per step to prevent instability: `maxstep` (0.2 Å)

**Convergence:**

- Checks maximum force component: max(|F|) < fmax (default 0.01 eV/Å)

## Hardware Information

**GPU**: NVIDIA H100 80GB HBM3

## Benchmark Configuration

The checked-in dynamics CSVs carry the measured atom counts, step counts,
timestep, and warmup count for each row. The single-system MD snapshot uses
1,000 steps, a 0.001 fs timestep, and 100 untimed warmup steps. Current runner
defaults, including the 94.4 K single-system thermostat settings and all size
grids, live in `benchmarks/dynamics/benchmark_config.yaml`.

Dynamics results use their own historical schema and are not part of the
reportable NL/D3/electrostatics 18-file snapshot. When reproducing a plotted
dynamics point, treat the committed CSV row as the record of that measurement
and the YAML as the configuration for a new run.

## Running Your Own Benchmarks

To reproduce these benchmarks or test on your own hardware:

### Single-System MD

```bash
cd benchmarks/dynamics
python benchmark_md_single.py --config benchmark_config.yaml
```

### Batched MD

```bash
python benchmark_md_batch.py --config benchmark_config.yaml
```

### Single-System Optimization

```bash
python benchmark_opt_single.py --config benchmark_config.yaml
```

### Batched Optimization

```bash
python benchmark_opt_batch.py --config benchmark_config.yaml
```

### FIRE1 vs FIRE2 Comparison

Full optimization runs comparing FIRE1 and FIRE2 convergence and wall-clock time
on fixed-cell and variable-cell LJ systems:

```bash
python benchmark_fire_compare.py --config benchmark_config.yaml --output-dir ./benchmark_results
```

### FIRE2 Kernel Performance

Raw per-step GPU kernel timing using CUDA events, sweeping total atoms and batch
sizes across float32 and float64:

```bash
python benchmark_fire2.py --config benchmark_config.yaml --output-dir ./benchmark_results
```

### Configuration File

Edit `benchmarks/dynamics/benchmark_config.yaml` to select current size grids,
integrators, optimizers, and timing parameters. Keeping the executable YAML as
the single configuration source avoids stale copies in the documentation.

### Output

Results are saved as CSV files in `docs/benchmarks/benchmark_results/`:

- `dynamics_md_single_nvalchemiops_<gpu_sku>.csv`
- `dynamics_opt_single_nvalchemiops_<gpu_sku>.csv`
- `dynamics_opt_batch_nvalchemiops_<gpu_sku>.csv`
- `fire_compare_<gpu_sku>.csv`
- `fire2_kernel_benchmark_<gpu_sku>.csv`

Generate plots with:

```bash
cd docs/benchmarks
python generate_plots.py
```

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
convergence behavior. A quasi-Newton L-BFGS optimizer is also available;
the sections below compare it against FIRE2 on evaluation count and
per-step cost.

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

### L-BFGS vs FIRE2: Evaluations and Step Cost

A quasi-Newton method: it approximates the inverse Hessian from recent position
and force differences to choose a direction, then steps along it bounded by a
`maxstep` trust region. No line search and no energy input, so a step costs
exactly one force evaluation. As with FIRE2, convergence is the caller's, so
the figures below are optimizer time only.

**Evaluations to convergence.** These measurements use Argon clusters of 13,
32, and 55 atoms with the package's LJ kernels and ``fmax <= 1e-4 eV/Å``.
Five starting geometries were tested per size. For each case, FIRE2's
timestep was swept over eight settings; its best converged result is the
baseline. Both optimizers used the stated precision:

| Metric | float32 | float64 |
| --- | --- | --- |
| Geometric mean of L-BFGS/FIRE2 evaluation counts | 0.59 | 0.57 |
| Worst ratio in this sample | 0.94 | 1.08 |
| Cases where both converged | 15 / 15 | 15 / 15 |

For this sample, the geometric-mean evaluation-count ratio is below one in
both precisions; the worst float64 case is above one. Counts vary by about
20% between runs, partly because neighbor-list rebuild ordering changes forces
in their last bits. This small Lennard-Jones sample is evidence to help choose
what to test next, not a prediction of which optimizer will need fewer
evaluations on another system.

This reproducible benchmark runs in minutes on one GPU, but covers only
small Lennard-Jones clusters. It does not represent crystals, strongly
anisotropic curvature, or force fields whose forces are not energy gradients.
Measure both optimizers on the structures and force fields relevant to your
workload before choosing one.

These measurements replace the earlier ``0.129`` result, which used a NumPy
all-pairs potential in reduced units and a FIRE2 grid tuned to those units.
The current measurements use the package LJ kernels and the timestep sweep
described above.

**Per-step optimizer cost.** These measurements compare optimizer time for a
single system using a harmonic potential and the ``--gates`` benchmark. The
table transcribes ``lbfgs_gate_timings.csv``. The five-repeat 10,000-atom
float32 measurement ranged from 7.9x to 8.6x; the other rows report individual
measurements. These ratios describe the listed setup only.

| Atoms | Precision | Eager (ms) | CUDA graph (ms) | FIRE2 (ms) | vs FIRE2 |
| --- | --- | --- | --- | --- | --- |
| 10,000 | float32 | 0.46 | 0.15 | 0.052 | 8.8x |
| 10,000 | float64 | 0.44 | 0.18 | 0.056 | 7.9x |
| 100,000 | float32 | 1.06 | 1.06 | 0.130 | 8.2x |
| 100,000 | float64 | 1.10 | 1.11 | 0.119 | 9.3x |
| 1,000,000 | float32 | 3.52 | 3.53 | 0.289 | 12.2x |
| 1,000,000 | float64 | 5.00 | 5.00 | 0.353 | 14.2x |

For these sizes and settings, the measured L-BFGS step time is higher than
FIRE2's, with ratios from 7.9x to 14.2x. The implementation performs
``2 * history_size + O(1)`` passes over the degrees of freedom; FIRE2 performs
fewer passes. In these runs, CUDA graph replay reduced the 10,000-atom
measurement but made little difference at larger sizes.

In these measurements, float32 and float64 times were within the observed
run-to-run spread at 10,000 and 100,000 atoms. At 1,000,000 atoms, they were
3.52 ms and 5.00 ms respectively. With ``history_size = 6``, the two history
buffers occupy 144 MB in float32 and 288 MB in float64; the H100 used here has
50 MB of L2 cache.

The entity-first state layout gives strided access when a kernel reads one
history slot across degrees of freedom. In these measurements, the difference
from the history-major estimate appeared at 1,000,000 atoms, not at smaller
sizes. This is a result for the tested layout and hardware.

**Break-even estimate.** Combining the evaluation-count ratios above with the
measured optimizer times gives estimated force-evaluation costs of 0.5, 1.2,
and 4.4 ms in float32 (0.5, 1.3, and 6.3 ms in float64) at the three tested
sizes. These are estimates for the benchmark cases, not general thresholds.
Run both optimizers with your force model and structures before choosing.

**Memory.** With `P` degrees of freedom, `M` systems and history size `m`:

```text
bytes = (2m + 3) * 3 * sizeof(dof) * P + (4m + 6) * sizeof(dof) * M
        + 4 * 4 * M
```

At `m = 6` that is 180 bytes per degree of freedom with float32 coordinates,
360 with float64; the histories dominate, and 3 to 7 is the usual range for
`m`. Every scalar follows the coordinate dtype, so `sizeof(dof)` appears in
both terms. That changes nothing for one large system, but for $10^6$
two-atom systems an fp32 state is 496 MB against 616 MB when the scalars were
pinned to float64 — a 20% saving, with per-step time unchanged within ±5%.

### FIRE Algorithm Features

**Adaptive Timestep:**

- Increases timestep when optimization is progressing smoothly
  (power $P = \mathbf{F} \cdot \mathbf{v} > 0$)
- Decreases timestep and resets velocities when moving uphill ($P < 0$)
- Parameters: `dt_max` (10.0 fs), `f_inc` (1.1), `f_dec` (0.5)

**Velocity Mixing:**

- Mixes velocity with force direction:
  $\mathbf{v} \rightarrow (1-\alpha)\mathbf{v} + \alpha |\mathbf{v}| \hat{\mathbf{F}}$
- Decreases mixing parameter $\alpha$ over time for faster convergence
- Parameter: `f_alpha` (0.99)

**Maximum Displacement:**

- Limits atomic displacement per step to prevent instability: `maxstep` (0.2 Å)

**Convergence:**

- Checks maximum force component: $\max(|\mathbf{F}|) < f_{\max}$ (default 0.01 eV/Å)

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

### L-BFGS vs FIRE2

Energy/force evaluations to convergence, which is the cost that dominates
relaxation driven by a machine-learned potential, plus per-step cost gates:

```bash
python benchmark_lbfgs.py --config benchmark_config.yaml --output-dir ./benchmark_results
python benchmark_lbfgs.py --gates
```

FIRE2 is swept over the grid in the config's `lbfgs.fire2_sweep` block and its
best converged result is what L-BFGS is compared against, so the baseline is not
handicapped by an untuned timestep.

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

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

### L-BFGS

L-BFGS is a quasi-Newton method: it approximates the inverse Hessian from recent
position and force differences to choose a search direction, then takes a step
along it bounded by a `maxstep` trust region. There is no line search and no
energy input, so the step costs exactly one force evaluation. As with FIRE2,
convergence is the caller's, so the figures below are optimizer time only.

**Evaluations to convergence.** The metric that matters when a machine-learned
potential dominates the optimizer's own kernel time. Argon clusters of 13, 32
and 55 atoms, relaxed through the package's own LJ kernels with the shared
`potential` parameters, `fmax <= 1e-4 eV/Å`, five starting geometries per size,
with FIRE2's timestep swept over 8 settings per case and its *best* converged
result used as the baseline. Run in both coordinate precisions, and reported
separately — every optimizer array follows the coordinate dtype, and both arms
run at whichever is selected, so an fp32 row compares fp32 against fp32:

| Metric | float32 | float64 |
| --- | --- | --- |
| Geometric mean | **0.59** | **0.57** |
| Worst individual case | **1.01** | **1.12** |
| Cases where both converged | 15 / 15 | 15 / 15 |

**L-BFGS needs about 1.7x fewer force evaluations on average, and it loses
outright in the worst case.** The worst ratio exceeded 1.0 in every run
measured, so on this workload L-BFGS is not uniformly better — it is better on
average.

**fp32 costs nothing in evaluation count here.** Both precisions reach the same
`1e-4` tolerance in statistically indistinguishable counts, which is the point
of letting every scalar follow the coordinate dtype: the fp32 path is a
complete path, not a degraded one.

Evaluation counts vary by roughly 20% run to run. Neighbor-list rebuild
ordering perturbs the forces in their last bits, and both optimizers amplify
that into a different trajectory, so read the aggregate rather than a single
cell.

**What this workload is, and what it is not.** Argon clusters under a
Lennard-Jones potential are a *reproducible* benchmark: everything it needs is
in this repository, it runs in minutes on one GPU, and anyone can regenerate
the table above. It is not a stand-in for a periodic inorganic solid under a
machine-learned potential, which is the setting L-BFGS is actually meant for
here, and the two can disagree — a smooth pair potential on a 55-atom cluster
exercises neither the stiff, strongly anisotropic curvature of a relaxing
crystal nor a force field that is not the gradient of its own energy.

The comparison that would settle that is a relaxation over OMat24 structures
with an OMat24-trained potential. It is not reproduced here: both the
structures and the trained models are distributed under a gated third-party
licence and would add a large optional dependency to this benchmark suite, so
running them is a deliberate choice for whoever needs that evidence rather
than something this suite does by default. The OMat24 figures quoted in review
remain that reviewer's own measurement, cited rather than reproduced. Read the
table above as what it is: the in-repo, regenerable result.

These numbers are much less favourable than the `0.129` this table carried
previously. That figure came from a NumPy all-pairs potential in reduced units
rather than the configured workload, evaluated with a FIRE2 timestep grid tuned
for those units — which handicapped the baseline once the units changed. Both
are fixed: the forces now come from the package kernels, and the grid runs to
3.0 fs because FIRE2 keeps improving well past the value the MD blocks use.

**Per-step optimizer cost.** Optimizer time only, single system, harmonic
potential, measured with `--gates`. Both arms run at the stated precision:

The ratio varies by roughly +/- 0.3 between runs. Each run writes
`lbfgs_gate_timings.csv` alongside the other benchmark results, so this table
has a regenerable record behind it.

| Atoms | Precision | Eager (ms) | CUDA graph (ms) | FIRE2 (ms) | vs FIRE2 |
| --- | --- | --- | --- | --- | --- |
| 10,000 | float32 | 0.47 | 0.14 | 0.059 | 7.9x |
| 10,000 | float64 | 0.45 | 0.16 | 0.058 | 7.7x |
| 100,000 | float32 | 0.98 | 0.98 | 0.116 | 8.5x |
| 100,000 | float64 | 1.02 | 1.02 | 0.112 | 9.1x |
| 1,000,000 | float64 | 3.12 | 3.11 | 0.319 | 9.8x |

The million-atom row predates the precision split and has not been regenerated;
it is float64 only. Re-run `--gates` without `--gate-sizes` to refresh it.

**Precision moves this less than the halved byte count suggests**, because
FIRE2 halves too: at a hundred thousand atoms the step goes 1.02 ms to 0.98 ms
and the ratio moves from 9.1x to 8.5x. It is reported per precision because it
is a measurement, not something to infer from one run and a factor.

**A single L-BFGS step is roughly ten times more expensive than a FIRE2 step.**
It runs `2m + O(1)` passes over the degrees of freedom against FIRE2's handful.
At ten thousand atoms the step is launch-bound and CUDA-graph replay recovers
about 2.9x; from one hundred thousand upwards the device work dominates and
replay recovers nothing.

**Break-even is no longer comfortable.** The model must cost more than roughly
0.5, 1.1 and 3.7 milliseconds per evaluation at these three sizes for L-BFGS to
win end to end (fp32: 0.53 and 1.13 ms at the two regenerated sizes). A machine-learned potential is milliseconds per evaluation, so
at ten thousand atoms L-BFGS wins clearly, at a hundred thousand it is close,
and at a million it needs a genuinely expensive model.

That follows arithmetically from the evaluation ratio: at `0.129` L-BFGS saved
almost 8x the evaluations and could absorb a 10x step cost easily; at `0.59` it
saves 1.7x, so the model has to be far more expensive to pay for the same step.
Prefer FIRE2 when the force evaluation is cheap, when the system is large, or
when worst-case behaviour matters more than the average.

**Memory.** With `P` degrees of freedom, `M` systems and history size `m`:

```text
bytes = (2m + 3) * 3 * sizeof(dof) * P + (4m + 6) * sizeof(dof) * M
        + 4 * 4 * M
```

At `m = 6` that is 180 bytes per degree of freedom with float32 coordinates and
360 with float64. The two history buffers dominate; reduce `m` if memory is
tight, with 3 to 7 the usual range.

Every scalar follows the coordinate dtype, so `sizeof(dof)` appears in both
array terms. For one large system that changes nothing — the `O(M)` scalars are
lost next to the `O(P)` history — but for a batch of many small systems it is
worth having: at 10<sup>6</sup> two-atom systems an fp32 state is 496 MB against
616 MB when the scalars were pinned to float64, a 20% saving. Per-step time is
unchanged either way (measured within ±5%).

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

<!-- markdownlint-disable MD029 -->
# Dynamics Benchmarks

This directory contains benchmark scripts for molecular dynamics integrators
and geometry optimizers.

## Benchmark Scripts

The benchmarks are organized into 6 modular scripts:

### Single-System Benchmarks

1. **`benchmark_md_single.py`** - Single-system MD benchmarks

   - Methods: VelocityVerlet, Langevin, NoseHoover, NPT, NPH
   - Output: CSV files with 11 columns (single-system schema)

   ```bash
   python benchmark_md_single.py --backend both --config benchmark_config.yaml
   ```

2. **`benchmark_opt_single.py`** - Single-system optimization benchmarks

   - Methods: FIRE
   - Output: CSV files with convergence metrics

   ```bash
   python benchmark_opt_single.py --backend both --config benchmark_config.yaml
   ```

### Batched Benchmarks

3. **`benchmark_md_batch.py`** - Batched MD benchmarks

   - Methods: VelocityVerlet, Langevin
   - Output: CSV files with 14 columns (batched schema)

   ```bash
   python benchmark_md_batch.py --backend nvalchemiops --config benchmark_config.yaml
   ```

4. **`benchmark_opt_batch.py`** - Batched optimization benchmarks

   - Methods: FIRE optimizer
   - Output: CSV files with batch throughput metrics

   ```bash
   python benchmark_opt_batch.py --backend nvalchemiops --config benchmark_config.yaml
   ```

### FIRE2 Benchmarks

5. **`benchmark_fire2.py`** - FIRE2 kernel-level performance benchmark

   - Measures raw per-step GPU time using CUDA events
   - Compares: FIRE2 (Warp), FIRE1 (Warp), PyTorch adapter, pure PyTorch reference
   - Sweeps total atoms x number of systems in float32 and float64
   - Config section: `fire2_perf`
   - Output: `fire2_kernel_benchmark_{gpu_sku}.csv`

   ```bash
   python benchmark_fire2.py --config benchmark_config.yaml --output-dir ./benchmark_results
   ```

6. **`benchmark_fire_compare.py`** - FIRE1 vs FIRE2 accuracy comparison

   - Full optimization runs on LJ systems measuring convergence and wall-clock time
   - Fixed cell (coordinate-only): `run_fire()` vs `run_fire2()`
   - Variable cell: `run_fire_cell()` vs coupled coordinate/cell `run_fire2_cell()`
   - Config section: `fire_compare`
   - Output: `fire_compare_{gpu_sku}.csv`

   ```bash
   python benchmark_fire_compare.py --config benchmark_config.yaml --output-dir ./benchmark_results
   ```

## Shared Utilities

- **`shared_utils.py`** - Common utilities used by all benchmark scripts
  - `BenchmarkResult` dataclass with CSV export
  - `create_lj_system()` - FCC argon system generator
  - `write_results_csv()` - CSV writer with schema detection
  - `get_gpu_sku()` - GPU identification
  - Unit conversion constants

## Configuration

All benchmarks are configured via **`benchmark_config.yaml`**:

```yaml
# Single-system MD
md_single:
  enabled: true
  system_sizes: [256, 512, 1024, 2048, 4096]
  integrators:
    velocity_verlet:
      steps: 10000
      dt: 0.001  # fs
      warmup_steps: 100
    langevin:
      steps: 10000
      temperature: 300.0  # K
      friction: 0.01  # 1/fs

# Single-system optimization
opt_single:
  enabled: true
  system_sizes: [256, 512, 1024, 2048]
  optimizers:
    fire:
      max_steps: 1000
      force_tolerance: 0.01  # eV/Å

# Batched MD
md_batch:
  enabled: true
  system_sizes: [256, 512, 1024]
  batch_sizes: [1, 2, 4, 8, 16, 32]

# Batched optimization
opt_batch:
  enabled: true
  system_sizes: [256, 512]
  batch_sizes: [1, 2, 4, 8, 16]

# FIRE2 kernel performance
fire2_perf:
  enabled: true
  total_atoms: [1000, 10000, 100000, 1000000]
  num_systems: [1, 10, 100]
  dtypes: [float32, float64]
  methods:
    warp_fire2: true
    warp_fire1: true
    torch_adapter: true
    torch_reference: true

# FIRE1 vs FIRE2 accuracy comparison
fire_compare:
  enabled: true
  system_sizes: [256, 512, 1024, 2048]
  force_tolerance: 0.005
  fixed_cell:
    enabled: true
    position_jitter: 0.05
  variable_cell:
    enabled: true
    position_jitter: 0.05
```

`position_jitter` is uniform Cartesian jitter in `[-value, value]` Å applied
after P1 expansion.

## Output Format

### CSV Schemas

**Single-system schema (11 columns):**

```bash
backend,method,num_atoms,ensemble,steps,dt,warmup_steps,avg_step_time_ms,
total_time_s,throughput_steps_per_s,throughput_atom_steps_per_s
```

**Batched schema (14 columns):**

```bash
backend,method,num_atoms,ensemble,batch_size,total_atoms,steps,dt,warmup_steps,
avg_step_time_ms,total_time_s,throughput_steps_per_s,throughput_atom_steps_per_s,
batch_throughput_system_steps_per_s
```

**FIRE2 kernel performance schema:**

```bash
method,dtype,total_atoms,num_systems,atoms_per_system,warmup,runs,
median_time_ms,min_time_ms,max_time_ms
```

**FIRE accuracy comparison schema:**

```bash
num_atoms,opt_type,method,steps,wall_time_s,converged
```

### Output Location

CSV files are saved to `../../docs/benchmarks/benchmark_results/` with naming convention:

```bash
dynamics_{md|opt}_{single|batch}_{backend}_{gpu_sku}.csv
fire2_kernel_benchmark_{gpu_sku}.csv
fire_compare_{gpu_sku}.csv
```

Examples:

- `dynamics_md_single_nvalchemiops_rtx4090.csv`
- `dynamics_md_batch_nvalchemiops_rtx4090.csv`
- `fire2_kernel_benchmark_rtx4090.csv`
- `fire_compare_rtx4090.csv`

## Documentation

Benchmark results are documented in `../../docs/benchmarks/`:

- `dynamics_md_single.md` - Single-system MD documentation
- `dynamics_opt_single.md` - Single-system optimization documentation
- `dynamics_md_batch.md` - Batched MD documentation
- `dynamics_opt_batch.md` - Batched optimization documentation

### Generating Plots

Plots are generated from CSV files using:

```bash
cd ../../docs/benchmarks
python generate_plots.py
```

This creates plots in `docs/benchmarks/_static/` which are embedded in the documentation.

## Running All Benchmarks

To run the complete benchmark suite:

```bash
# Single-system benchmarks
python benchmark_md_single.py --backend both
python benchmark_opt_single.py --backend both

# Batched benchmarks (when enabled in config)
python benchmark_md_batch.py --backend nvalchemiops
python benchmark_opt_batch.py --backend nvalchemiops

# FIRE2 benchmarks
python benchmark_fire2.py --config benchmark_config.yaml --output-dir ./benchmark_results
python benchmark_fire_compare.py --config benchmark_config.yaml --output-dir ./benchmark_results

# Generate plots
cd ../../docs/benchmarks
python generate_plots.py

# Build documentation
cd ../..
make html  # or your documentation build command
```

## System Requirements

- **GPU:** NVIDIA GPU with CUDA support (tested on RTX 4090, A100, H100)
- **Python:** 3.8+
- **Dependencies:**
  - nvalchemiops (with warp backend)
  - PyTorch
  - NumPy, pandas, matplotlib (for plot generation)

## Performance Tips

1. **GPU Selection:** Use `CUDA_VISIBLE_DEVICES` to select specific GPU
2. **System Size:** Larger systems better utilize GPU parallelism
3. **Batch Size:** Increase until GPU saturates (monitor memory usage)
4. **Neighbor List:** Adjust rebuild interval in config for optimal performance
5. **Warmup:** Always include warmup steps to exclude kernel compilation overhead

## Contributing

To add new benchmarks:

1. Follow existing script patterns (shared_utils, CSV output, CLI args)
2. Update `benchmark_config.yaml` with new sections
3. Add plot generation to `docs/benchmarks/generate_plots.py`
4. Create documentation page in `docs/benchmarks/`
5. Update this README

## Questions?

- See documentation: `../../docs/benchmarks/`
- Check configuration: `benchmark_config.yaml`
- Review shared utilities: `shared_utils.py`

## `benchmark_lbfgs.py`

Compares L-BFGS with FIRE2 by **force evaluations to convergence**, the cost
that dominates relaxation driven by a machine-learned potential. The system is
an argon cluster relaxed through the package's own LJ kernels and neighbor
list, with parameters from the shared `potential` config block. FIRE2's
timestep is swept per case and its best converged configuration reported, so
the comparison is not skewed by an untuned baseline.

```bash
python -m benchmarks.dynamics.benchmark_lbfgs --output-dir ./benchmark_results
```

Defaults come from the `lbfgs` section of `benchmark_config.yaml`; any flag
overrides the file:

```bash
python -m benchmarks.dynamics.benchmark_lbfgs \
    --sizes 13 32 55 --seeds 5 --force-tol 1e-4 --dtype float32 \
    --output-dir ./benchmark_results
```

`--dtype` picks the coordinate precisions, and both arms run at whichever is
selected — comparing an fp32 L-BFGS against an fp64 FIRE2 would measure
precision rather than optimizer. Every optimizer array follows it, as do the
neighbor list and the LJ evaluation, so an fp32 run is fp32 end to end. It
defaults to `lbfgs.dtypes`, which is `float32` then `float64`. Results are
aggregated per precision, never pooled.

`--device` selects the GPU and defaults to `cuda:0`, matching
`benchmark_fire2.py` — point both at the same device when comparing numbers.
It is made the current CUDA device at entry, not just passed to allocations,
since CUDA events, streams and graphs take the current device.

Writes `lbfgs_vs_fire2_evaluations.csv`. Runs that hit the evaluation cap are
flagged; their ratios are upper bounds on L-BFGS's advantage. Evaluation
counts vary by roughly 20% run to run — neighbor-list rebuild ordering
perturbs the forces in their last bits — so read the aggregate, not a row.

With no `--output-dir`, results go to `output.results_dir` from the config,
resolved relative to the config file. Set `output.save_timing: false` to write
nothing; an explicit `--output-dir` still writes.

### Per-step cost gates

```bash
python -m benchmarks.dynamics.benchmark_lbfgs --gates
```

Reports optimizer-only step time against FIRE2 at scale, what CUDA-graph
replay recovers, and the break-even model cost. The step issues far more
kernels than FIRE2, so at small sizes it is Python-launch-bound; capture it in
a CUDA graph if that matters.

The break-even figure needs the measured evaluation ratio, read from
`lbfgs.gates.eval_ratio`. Re-run the benchmark above and update it if you
change the sizes or tolerance, or pass `--eval-ratio`.

Writes `lbfgs_gate_timings.csv` through the same output path — one row per
size *and precision*. That file is the record behind the per-step table in
`docs/benchmarks/dynamics.md`, so regenerate it when you update those numbers.

## `benchmark_lbfgs_omat24.py`

The same evaluation-count comparison on **periodic inorganic structures under
a machine-learned potential** — the setting L-BFGS is actually meant for. The
LJ runner above is reproducible from this repository alone but exercises
neither stiff crystalline curvature nor a force field that is not the gradient
of its own energy.

Not part of the default suite: it needs `mace-torch` and a model checkpoint,
neither of which this package depends on, plus a structure collection you
supply.

```bash
python -m benchmarks.dynamics.benchmark_lbfgs_omat24 \
    --structures path/to/omat-collection/ \
    --output-dir ./benchmark_results
```

Each structure is relaxed independently at fixed cell to `fmax <= 0.05 eV/Å`,
with FIRE2's timestep swept per structure and its best converged run used as
the baseline. Both optimizers share the same `maxstep`, and the evaluation
count is taken at the potential itself, so neither loop can undercount.
Writes `lbfgs_vs_fire2_omat24.csv`, one row per structure.

`--dtype` defaults to `float64` here rather than `float32`, because MACE
recommends it for geometry optimization; `--model` selects the checkpoint and
defaults to `medium-mpa-0`.

`mace-torch` 0.3.x needs `e3nn<0.5` while this package needs a newer one, so
run it from a virtual environment that pins the older `e3nn` instead of
downgrading in place:

```bash
python -m venv --system-site-packages /tmp/macevenv
/tmp/macevenv/bin/pip install "e3nn==0.4.4"
/tmp/macevenv/bin/python -m benchmarks.dynamics.benchmark_lbfgs_omat24 ...
```

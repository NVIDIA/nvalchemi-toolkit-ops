# Benchmark Results

Pre-computed CSVs consumed by the Sphinx docs build. The shipped numbers
under this directory were produced on an **NVIDIA H100 80 GB HBM3
(Hopper)** and cover three modules (neighbor list, DFT-D3 dispersion,
electrostatics) across two chemical systems (CsCl, NH₃) and three
scaling modes.

See the per-module doc pages for how to read the plots and how to
reproduce:
- `../neighborlist.md`
- `../dftd3.md`
- `../electrostatics.md`

## File naming

Names follow the scheme emitted by
`benchmarks.utils.make_csv_name(module, system, mode)`:

```
{module}-{system}-{mode-slug}.csv
```

Where `module` ∈ `{nl, d3, el}`, `system` ∈ `{cscl, nh3}`, and
`mode-slug` ∈ `{system-size-scaling, constant-workload-scaling,
batch-scaling}`. Example: `nl-cscl-system-size-scaling.csv`.

Legacy names like `neighbor_list_benchmark_<method>_<gpu_sku>.csv` are
not produced by the current runners.

## CSV schema

Emitted by `benchmarks.utils.build_result`:

| Column | Type | Description |
|---|---|---|
| `system` | str | `cscl` or `nh3` |
| `scaling_mode` | str | `system_size`, `constant_workload`, or `batch_scaling` |
| `method` | str | `naive` / `cell` (NL), `dftd3` (D3), `pme` / `pme_cg` / `ewald` / `ewald_cg` (EL) |
| `backend` | str | `torch` or `jax` |
| `atoms_per_system` | int | Atoms in one system |
| `batch_size` | int | Number of systems in the batch |
| `total_atoms` | int | `atoms_per_system` × `batch_size` |
| `time_us_per_atom` | float | Mean μs per atom across the batch timing |
| `throughput_atoms_per_sec` | float | Derived throughput |
| `mem_delta_mb` | float | Memory delta from the pre-timing measurement call (MB); always 0 for JAX |
| `mem_peak_gb` | float | Peak torch allocator memory (GB); always 0 for JAX |
| `success` | bool | `False` rows are filtered by the plotter |
| `cutoff` | float | Added by NL and D3 |
| `accuracy` | float | Added by EL |
| `time_d3_us_per_atom` | float | Added by D3 (excludes NL build time) |

Multiple runs that write to the same directory are appended rather
than overwritten when their headers match — this is how torch and jax
runs coexist in one file.

### Failures sidecar (`*-failures.csv`)

Each main CSV may have a matching `*-failures.csv` recording configs that
were skipped at runtime (OOM in NL build, OOM in the kernel itself, or
YAML-driven preemptive skips). Schema is a union of the NL/D3/EL columns
so that one file format works for all three modules:

| Column | Type | Description |
|---|---|---|
| `system` | str | `cscl` or `nh3` |
| `scaling_mode` | str | `system_size`, `constant_workload`, or `batch_scaling` |
| `method` | str | Same values as the main CSV (`naive`, `cell`, `pme`, …) |
| `backend` | str | `torch` or `jax` |
| `atoms_per_system` | int | Atoms in one system |
| `batch_size` | int | Number of systems in the batch |
| `cutoff` | float | Populated for NL and D3 rows; blank for EL |
| `accuracy` | float | Populated for EL rows; blank for NL and D3 |
| `failure_reason` | str | `OOM_kernel`, `OOM_NL_build`, or `OOM_preempt` |

**Contract:** a sidecar file is shipped only when the corresponding main
CSV has at least one skipped config. An absent sidecar means "no
failures on the reference H100"; it does not mean the run was
incomplete. Sidecars are consumed by the plotter to annotate missing
points; see `benchmarks/plotting/plot_benchmarks.py::load_failures`.

## Reproducing

Run from the repository root. Module-specific flags are documented on
each module's doc page; the flags below are common to all three.

```bash
python -m benchmarks.neighborlist.benchmark_neighborlist \
    --config benchmarks/neighborlist/benchmark_config.yaml \
    --output-dir docs/benchmarks/benchmark_results
```

Swap in `benchmarks.interactions.dispersion.benchmark_dftd3` or
`benchmarks.interactions.electrostatics.benchmark_electrostatics` for
the other modules, or invoke all three via the unified suite:

```bash
python -m benchmarks.benchmark_suite --benchmark all \
    --output-dir docs/benchmarks/benchmark_results
```

For the JAX backend, prepend `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
pass `--backend jax`.

## Visualization

Sphinx's generate_plots hook reads every CSV in this directory and
writes PNGs to `../_static/`. The benchmark pages embed those images.

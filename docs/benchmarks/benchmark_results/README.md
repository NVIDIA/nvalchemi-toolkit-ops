---
orphan: true
---

# Benchmark Results

Pre-computed CSVs consumed by the Sphinx docs build. The shipped numbers
under this directory were produced on an **NVIDIA H100 80 GB HBM3
(Hopper)** and cover three modules (neighbor list, DFT-D3 dispersion,
electrostatics) across two chemical systems (CsCl, NH₃) and three
scaling modes.

The current snapshot was collected on 3 July 2026 under run ID
`75522394-a5f1-4f9c-9f28-3b3d535dbff4` and source fingerprint
`08776c4583b04b9205674f13a7ec78b08ce940faf1fb56b9c2d9f4ba4237c1f8`.
Its 18 reportable CSVs contain all 3,504 planned rows: 3,368 successful
measurements, 124 explicit `OutOfMemoryError` rows, and 12
`UnsupportedConfiguration` rows.

Every CSV embeds the same fingerprint in ``software_context``. The recorded Git
head, `90af513a66537cbab8d836ea0c064d137a7b4a05`, identifies the 0.4
release-candidate base; the fingerprint identifies the complete measured source
tree, including the benchmark suite and approved JAX Ewald memory fix applied
above that base. After collection, the branch corrects benchmark-only help and
system-description wording, makes the suite writer use LF line endings,
normalizes the checked-in CSV line endings, and updates documentation and
generated figures. Those edits do not change parsed values, a timed kernel or
callable, the benchmark grid, or any CSV value. A future kernel, public API,
timing-boundary, or grid change requires a complete replacement run.

See the per-module doc pages for how to read the plots and how to
reproduce:

- `../neighborlist.md`
- `../dftd3.md`
- `../electrostatics.md`

## File naming

Names follow the scheme emitted by
`benchmarks.suite_utils.make_csv_name(module, system, mode)`:

```text
{module}-{system}-{mode-slug}.csv
```

Where `module` ∈ `{nl, d3, el}`, `system` ∈ `{cscl, nh3}`, and
`mode-slug` ∈ `{system-size-scaling, constant-workload-scaling,
batch-scaling}`. Example: `nl-cscl-system-size-scaling.csv`.

The current reportable NL/D3/EL suite uses only root-level `nl-*.csv`,
`d3-*.csv`, and `el-*.csv` files. Separate dynamics and segment-operation CSVs
may also remain at the root for their own docs pages. Outputs from the optional
extended electrostatics runner use `electrostatics_benchmark_*.csv` names and a
different schema; they are not read by the reportable plot generation path.

## CSV schema

Emitted by `benchmarks.suite_utils.build_result`:

| Column | Type | Description |
|---|---|---|
| `system` | str | `cscl` or `nh3` |
| `scaling_mode` | str | `system_size`, `constant_workload`, or `batch_scaling` |
| `method` | str | NL strategy (`naive_scalar`, `naive_tile`, `cell_list_atom_centric`, `cell_list_pair_centric`, `cluster_tile`, plus batch-prefixed concrete APIs where applicable), `dftd3` (D3), or `pme` / `ewald` (EL) |
| `backend` | str | `torch`, `jax`, or `warp` where supported |
| `atoms_per_system` | int | Atoms in one system |
| `batch_size` | int | Number of systems in the batch |
| `total_atoms` | int | `atoms_per_system` × `batch_size` |
| `time_us_per_atom` | float | Mean μs per atom across the batch timing |
| `throughput_atoms_per_sec` | float | Derived throughput |
| `mem_delta_mb` | float | Torch CUDA allocator delta from the pre-timing measurement call (MB); NaN for JAX |
| `mem_peak_gb` | float | Torch CUDA allocator peak (GB); NaN for JAX |
| `timing_runs` | int | Number of timed calls represented by the row |
| `warmup_runs` | int | Number of untimed warmup calls before measurement |
| `timing_method` | str | Timing path used for the row, such as `torch_cuda_events`, `jax_wall_block_until_ready`, or the serial fallback `jax_wall_block_each` |
| `timing_method_real` | str | Added by EL; `not_measured` unless component profiling is enabled |
| `timing_method_reciprocal` | str | Added by EL; `not_measured` unless component profiling is enabled |
| `compile_policy` | str | Compile/warmup policy; shipped rows use `warmup_excluded` |
| `success` | bool | `False` rows are filtered by the plotter |
| `error` | str | Concise failure or skip message for `success=False` rows |
| `error_type` | str | Stable failure class, such as `OutOfMemoryError`, `UnsupportedConfiguration`, `SkippedByPolicy`, or `SkippedAfterOOM` |
| `failure_stage` | str | Optional; populated by failure paths that can identify the setup or timing stage that raised the error |
| `cutoff` | float | Added by NL and D3 |
| `configured_max_neighbors` | int | Added by NL; neighbor-matrix capacity selected before the measured call |
| `max_neighbors` | int | Added by NL; width of the allocated or returned dense neighbor matrix |
| `max_total_cells` | int | Added by cell-list NL rows; allocated cell-grid capacity |
| `cell_list_min_cells` | int | Added by NL backends that expose this cell-grid metadata; minimum per-axis cell count used by the selected strategy |
| `total_neighbor_pairs` | int | Added by NL; backend-specific pair/storage count (dense paths report allocated slots, while direct Warp rows report populated pairs) |
| `accuracy` | float | Added by EL |
| `time_d3_us_per_atom` | float | Added by D3 (excludes NL build time) |
| `neighbor_setup_method` | str | Added by D3; neighbor-list setup API used before timing (`cell_list` or `batch_cell_list`) |
| `time_real_us_per_atom` | float | Added by EL; NaN unless component profiling is enabled |
| `time_reciprocal_us_per_atom` | float | Added by EL; NaN unless component profiling is enabled |
| `backend_comparable` | bool | Added by NL to mark rows included in backend-comparison plots |
| `timing_scope` | str | Added by NL to separate scalar backend-comparison rows from coverage-only eager, backend-specific, pair-centric, and cluster-tile rows |
| `allocation_boundary` | str | Added by NL to identify caller-preallocated, API-managed, or JIT-managed buffers |
| `derivative_contract` | str | Added by EL; reportable rows use `energy_autograd` |
| `workload` | str | Added by EL; reportable rows use `energy_forces_charge_gradients` |
| `compute_forces` | bool | Added by EL; always `True` for the reportable workload |
| `compute_charge_gradients` | bool | Added by EL; always `True` for the reportable workload |
| `component_profiled` | bool | Added by EL; `False` for the full-only reportable timing contract |
| `pme_cache_mode` | str | Added by EL; PME rows use `full_static` when fixed-cell volume, inverse-cell, and spline-modulus metadata are precomputed outside timing |
| `provenance_version` | str | Version of the benchmark provenance schema |
| `run_id` | UUID | Shared identity for all shards in one reportable run |
| `gpu_context` | JSON str | Comparable GPU model, compute capability, memory, and driver context |
| `software_context` | JSON str | Python/framework versions, Git revision, and benchmark-source fingerprint |
| `execution_context` | JSON str | Per-shard host and physical GPU UUID; may differ across compatible cluster nodes |
| `runtime_context` | JSON str | Backend runtime and allocator settings, including requested and actual JAX x64/JIT state; must match within each backend |
| `input_context` | JSON str | Content fingerprints for external NH3 and DFT-D3 inputs; absolute scratch paths are omitted |

When Torch and JAX runs share an output directory, each backend rerun replaces
only its own rows and preserves the other backend's rows after validating the
run, GPU, software, source, input, and per-backend runtime provenance.
Compatible scheduler shards may use different hosts and physical GPU UUIDs.
Backends contributing to the same CSV must use matching system filters; an
asymmetric filter can change the external-input fingerprint and is rejected
instead of being silently merged.
Failed, skipped, and OOM cases are
written directly into the main CSV with `success=False`; the plotter filters
those rows out. The suite no longer writes separate failure files.

The committed H100 CSVs use provenance schema version 2 and the current EL
energy-autograd contract. All 18 files share the run ID shown above; start a
fresh run directory and publish a complete replacement set for future reruns.

## Reproducing

Run from the repository root. Module-specific flags are documented on
each module's doc page; the flags below are common to all three reportable
runners. This standardized suite is the benchmark merge gate.

```bash
python -m benchmarks.neighborlist.benchmark_neighborlist \
    --config benchmarks/neighborlist/benchmark_config.yaml \
    --output-dir "$BENCHMARK_SCRATCH/results/manual-nl-run"
```

Swap in `benchmarks.interactions.dispersion.benchmark_dftd3` or
`benchmarks.interactions.electrostatics.benchmark_electrostatics_suite` for
the other modules, or invoke all three via the unified suite:

```bash
bash benchmarks/run_reportable_suite.sh \
    --output-dir "$BENCHMARK_SCRATCH/results/reportable-run"
```

The wrapper defaults to ``--backend all``: Torch and JAX run for NL, D3, and
electrostatics, and the direct Warp pass runs for NL. Use ``--backend both``
only when you intentionally want the two framework backends without Warp NL
rows.

Use a fresh directory for a new run. Pass `--resume` only to continue the run
already recorded there, or pass the same `--run-id` to parallel scheduler
shards; a mismatched run ID is rejected before execution.

Electrostatics has a separate optional surface:
`benchmark_electrostatics.py` uses `benchmark_config_extended.yaml` for
point-charge/slab/DSF studies and `benchmark_config_multipole.yaml` for
multipoles. It depends on additional benchmark packages not declared in the
project extras and does not write the reportable `el-*.csv` schema. Do not
substitute it for `benchmark_electrostatics_suite.py` in merge-gate commands.

The docs CSVs are reportable benchmark outputs: they use the full configured
grid, 3 warmups, and 10 timed runs unless an explicit command-line filter is
shown. Reduced smoke runs should write to a separate output directory.

The shipped H100 CSVs were collected as scheduler shards with the same full
grids and timing protocol. Only tasks whose rows remain in the published CSVs
are counted below; discarded diagnostic and replacement attempts are excluded.

| Module | Accepted H100 task time | Contributing hosts |
|---|---:|---|
| Neighbor list | 57 min 18 s | `s4124-0096`, `s4124-0105`, `s4124-0106`, `s4124-0129` |
| DFT-D3 | 17 min 59 s | `s4124-0096`, `s4124-0105`, `s4124-0106`, `s4124-0129` |
| Electrostatics | 1 h 47 min 57 s | `ipp2-0709`, `ipp2-0711`, `s4124-0096`, `s4124-0105`, `s4124-0106`, `s4124-0129` |
| **Total** | **3 h 3 min 14 s** | |

These task times add the elapsed time of every accepted one-GPU shard. They are
not collection wall time. The parallel collection ran from 12:33:24 to 13:17:25
PDT, or **44 min 1 s**. The main array used up to four H100s. Two JAX
electrostatics processes exceeded the same 128 GB host-memory allocation on
their first nodes; identical retries completed on other qualified H100 nodes
without raising the allocation or limiting the atom grid. One retry briefly
overlapped the four-slot array, so peak concurrency was five H100s. The two
discarded attempts consumed another 10 min 10 s of GPU task time and are not
included in the accepted-time table.

CSV rows record steady-state per-benchmark timings. Scheduler time also includes
compilation, warmup, input loading, process startup, and cleanup.

For the JAX backend, pass `--backend jax`. Reportable runners preserve explicit
JAX/XLA settings, otherwise default the memory fraction to `0.95` while leaving
XLA's default preallocation policy in effect. The unified suite enables x64
before import because its electrostatics pass requires float64.
For D3 on offline clusters, pass `--d3-params-path` to a scratch-local
`dftd3_parameters.pt` file or pre-populate
`$XDG_CACHE_HOME/nvalchemiops/dftd3_parameters.pt`.

## Visualization

Sphinx's generate_plots hook reads the standardized NL/D3/EL suite CSVs and
the separate dynamics CSV schema it knows how to parse, then writes PNGs to
`../_static/`. The benchmark pages embed those images.

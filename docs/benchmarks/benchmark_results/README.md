# Benchmark Reference CSVs

Pre-computed H100 benchmark results used by `docs/benchmarks/generate_plots.py`
during `make docs`. These CSVs are the source of truth for the Sphinx
documentation plots.

## File Naming

```text
{module}-{system}-{scaling-mode}.csv
```

- **module**: `nl` (Neighbor List), `d3` (DFT-D3), `el` (Electrostatics)
- **system**: `cscl` (CsCl crystal), `nh3` (NH₃ molecular clusters)
- **scaling-mode**: `system-size-scaling`, `constant-workload-scaling`, `batch-scaling`

## CSV Columns

All CSVs share these core columns:

| Column | Description |
|--------|-------------|
| `system` | System name (cscl or nh3) |
| `scaling_mode` | Scaling mode |
| `method` | Algorithm or method name |
| `atoms_per_system` | Atoms per individual system |
| `batch_size` | Number of batched replicas |
| `total_atoms` | Total atoms (atoms_per_system × batch_size) |
| `time_us_per_atom` | Time per atom (μs) |
| `throughput_matoms_per_sec` | Throughput (10⁶ atoms/s) |
| `mem_peak_gb` | Peak GPU VRAM (GB) |

Module-specific columns: `cutoff` (NL, D3), `accuracy` (EL),
`time_d3_us_per_atom` (D3-only time excluding NL).

## Updating Results

Run the benchmark suite on your hardware and copy CSVs here:

```bash
uv sync --all-extras --group benchmark
python benchmarks/benchmark_suite.py --benchmark all
cp benchmarks/benchmark-results/run_*/*.csv docs/benchmarks/benchmark_results/
```

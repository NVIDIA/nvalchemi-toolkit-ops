# Neighbor List Benchmark Results

This directory contains pre-computed benchmark results for different neighbor list algorithms
on various GPU hardware.

## File Naming Convention

Results are stored in CSV files with the following naming pattern:

```bash
neighbor_list_benchmark_<method>_<gpu_sku>.csv
```

Where:

- `<method>`: The neighbor list algorithm (`naive`, `cell_list`, `batch_naive`, `batch_cell_list`)
- `<gpu_sku>`: GPU identifier (e.g., `rtx_a3000_laptop_gpu`, `a100_sxm4_80gb`)

## Running Benchmarks

To generate new benchmark results:

```bash
cd benchmarks/neighborlist
python benchmark_neighborlist.py --config benchmark_config.yaml --output-dir ../../docs/benchmarks/benchmark_results
```

### Command Line Options

- `--config`: Path to YAML configuration file (required)
- `--output-dir`: Output directory for CSV files (default: `../../docs/benchmarks/benchmark_results`)
- `--methods`: Specific methods to benchmark (e.g., `--methods naive cell_list`)
- `--gpu-sku`: Override GPU SKU name for output files

### Examples

Run all benchmarks:

```bash
python benchmark_neighborlist.py --config benchmark_config.yaml
```

Run only specific methods:

```bash
python benchmark_neighborlist.py --config benchmark_config.yaml --methods naive cell_list
```

Override GPU SKU name:

```bash
python benchmark_neighborlist.py --config benchmark_config.yaml --gpu-sku custom_gpu_name
```

## CSV Format

Each CSV file contains the following columns:

- `method`: Algorithm name
- `total_atoms`: Total number of atoms in the system
- `atoms_per_system`: Atoms per system (for batch methods)
- `total_neighbors`: Total number of neighbor pairs found
- `batch_size`: Batch size (1 for single-system methods)
- `median_time_us`: Median execution time in microseconds
- `success`: Whether the benchmark completed successfully (optional)
- `error`: Error message if benchmark failed (optional)
- `error_type`: Error type (e.g., "OOM", "Timeout") if failed (optional)

## Visualization

The Sphinx documentation automatically discovers and visualizes all CSV files in this directory.
See the benchmarks section of the documentation for interactive plots and comparisons.

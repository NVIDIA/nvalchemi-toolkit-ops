# NVIDIA ALCHEMI Toolkit-Ops Benchmark Suite

Comprehensive GPU benchmark suite for [NVIDIA ALCHEMI Toolkit-Ops](https://github.com/NVIDIA/nvalchemi-toolkit-ops) kernels, measuring performance of neighbor lists, DFT-D3 dispersion, and long-range electrostatics on NVIDIA GPUs.

> **Reference**: Benchmark systems are consistent with the [NVIDIA ALCHEMI Toolkit-Ops blog post](https://developer.nvidia.com/blog/accelerating-ai-powered-chemistry-and-materials-science-simulations-with-nvidia-alchemi-toolkit-ops/):
> *"Test systems consisted of ammonia clusters of increasing size packed into various cells using Packmol."*

## Quick Start

```bash
# 1. Install dependencies
uv sync --group benchmark

# 2. Generate benchmark structures (if not present)
cd nh3 && bash generate_pbc_pdbs.sh

# 3. Run benchmarks
uv run python benchmark_suite.py --benchmark all

# 4. Plot results
uv run python plot_combined_benchmarks.py ./benchmark-results/benchmark_YYYY-MM-DD_HH-MM-SS/
```

## Directory Structure

```
benchmarks-temp/
├── README.md                    # This file
├── benchmark_suite.py           # Main benchmark runner
├── plot_combined_benchmarks.py  # Plotting script for results
└── nh3/                         # NH3 test systems
    ├── generate_pbc_pdbs.sh     # Interactive PDB generator
    ├── ammonia.pdb              # Single NH3 molecule template
    ├── ammonia_pbc_128.pdb      # 128 atoms
    ├── ammonia_pbc_256.pdb      # 256 atoms
    ├── ...
    └── ammonia_pbc_131072.pdb   # 128k atoms
```

## Benchmark Suite

### Available Benchmarks

| Alias | Description | Mode |
|-------|-------------|------|
| `nl` | Neighbor List | Batched (constant 128k total atoms) |
| `d3` | DFT-D3 Dispersion | Batched (constant 128k total atoms) |
| `d3s` | DFT-D3 Scaling | Single system (batch=1, varying size) |
| `el` | Electrostatics Scaling | Single system (batch=1, varying size) |
| `elb` | Electrostatics Batched | Batched (constant 128k total atoms) |
| `all` | Run all benchmarks | — |

### Usage Examples

```bash
# Run specific benchmarks
python benchmark_suite.py -b nl --cutoffs 6 15 25
python benchmark_suite.py -b d3 d3s --cutoffs 15 25
python benchmark_suite.py -b el elb --accuracy 1e-4 1e-6

# Run all with custom timing runs
python benchmark_suite.py -b all --timing-runs 20

# Custom target atoms and max system size
python benchmark_suite.py -b elb --target-atoms 65536 --max-atoms 65536
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `-b, --benchmark` | `all` | Benchmarks to run (can specify multiple) |
| `-n, --timing-runs` | `20` | Number of timing iterations per measurement |
| `-c, --cutoffs` | `6 15 25` | Cutoff radii in Ångström (NL/D3) |
| `-a, --accuracy` | `1e-4 1e-6` | Accuracy levels (Electrostatics) |
| `-t, --target-atoms` | `131072` | Target total atoms for batched benchmarks |
| `--max-atoms` | `131072` | Maximum atoms for scaling benchmarks |
| `--nh3-dir` | `./nh3` | Directory with NH3 PDB files |
| `--d3-params` | `~/.cache/nvalchemiops/dftd3_parameters.pt` | D3 parameters file |

### Output

Results are saved to timestamped directories:

```
benchmark-results/
└── benchmark_2026-01-27_14-30-00/
    ├── benchmark_nl_results.csv
    ├── benchmark_d3_results.csv
    ├── benchmark_d3_scaling_results.csv
    ├── benchmark_electrostatics_results.csv
    └── benchmark_electrostatics_128k_results.csv
```

## Timing Methodology

The benchmark uses the **correct CUDA batch timing pattern** (verified by senior engineer):

```python
# Pattern: sync → start.record() → N × fn() → end.record() → sync()
torch.cuda.synchronize()
start.record()
for _ in range(N):
    fn()  # NO sync inside loop!
end.record()
torch.cuda.synchronize()
mean_time = start.elapsed_time(end) / N
```

This measures **sustained throughput** without sync overhead pollution.

## Test Systems

NH3 (ammonia) packed into cubic periodic boxes using Packmol:

| Atoms | Cell Size (Å) | Use Case |
|-------|---------------|----------|
| 128 | 10.99 | Batching (×1024) |
| 256 | 13.85 | Batching (×512) |
| 512 | 17.44 | Batching (×256) |
| 1k | 21.98 | Batching (×128) |
| 2k | 27.69 | Batching (×64) |
| 4k | 34.89 | Batching (×32) |
| 8k | 43.96 | Batching (×16) |
| 16k | 55.38 | Batching (×8) |
| 32k | 69.78 | Batching (×4) |
| 64k | 87.91 | Batching (×2) |
| 128k | 110.76 | Single system |

Cell length formula: `L = (41.47 × N_atoms / 4)^(1/3)` Å

### Generating Test Structures

```bash
cd nh3
bash generate_pbc_pdbs.sh

# Interactive prompts:
#   - Enter numbers: 1 2 3
#   - Enter range: 1-5
#   - Presets: 'all', 'small' (1-6), 'medium' (1-10)
```

## Dependencies

### Required
- Python 3.11+
- PyTorch 2.8+
- NVIDIA GPU - tested on H100 and A6000 Pro
- CUDA 12+
- nvalchemi-toolkit-ops-0.2
- `packmol` - Generating test structures (`uvx packmol`)

### Optional
- `pynvml` - Accurate GPU memory tracking (`pip install nvidia-ml-py`)
- `natsort` - Natural sorting of PDB files
- `matplotlib` - Plotting results

Install benchmark dependencies:
```bash
uv sync --group benchmark
```

## Plotting Results

```bash
python plot_combined_benchmarks.py
```

Generates a 4-panel figure:
- **A**: Neighbor List scaling (constant 128k atoms)
- **B**: DFT-D3 system size scaling
- **C**: Electrostatics batched (constant 128k atoms)
- **D**: Electrostatics system size scaling

Output: `combined_benchmarks.png` and `combined_benchmarks_notitle.png` (for slides)

## Unit System

| Quantity | Input Unit | Output Unit |
|----------|------------|-------------|
| Positions | Å | — |
| Cell | Å | — |
| Charges | e (elementary charge) | — |
| D3 cutoff | Bohr (converted internally) | — |
| Energy | — | e²/Å (multiply by 14.3996 for eV) |
| Forces | — | e²/Å² |

## Known Limitations

1. **OOM Prevention**: High accuracy (1e-6) automatically skipped for systems ≥128k atoms
2. **D3 6Å cutoff**: Skipped in plots (too short for meaningful dispersion)
3. **Large systems**: 256k+ atoms require significant time with Packmol

## Troubleshooting

### Missing D3 Parameters
```
ERROR: D3 parameters required for D3 benchmarks
```

D3 parameters are **automatically downloaded** from the [Grimme group website](https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/) on first use. To generate them manually:

```python
# Run from repo root
from examples.dispersion.utils import extract_dftd3_parameters, save_dftd3_parameters

params = extract_dftd3_parameters()  # Downloads ~500KB from Grimme group
save_dftd3_parameters(params)        # Saves to ~/.cache/nvalchemiops/dftd3_parameters.pt
```

Or simply run one of the D3 examples first:
```bash
cd examples/dispersion && uv run python 01_dftd3_molecule.py
```

### CUDA Out of Memory
- Reduce `--target-atoms` or `--max-atoms`
- Skip high accuracy levels: `--accuracy 1e-4`

### Packmol Not Found
```bash
pip install packmol
# or
uvx packmol
```


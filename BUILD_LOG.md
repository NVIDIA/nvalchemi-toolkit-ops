# Build Log — `tme` branch

Ongoing log of changes on the `tme` branch for merge request.
Base: `f308f00` (`include pynvml; update readme; cplotting with titles`)

---

## 2026-02-12 — Ewald API: add `compute_charge_gradients` to `ewald_summation()`

### Problem

`ewald_summation()` was the only high-level electrostatics API missing
`compute_charge_gradients` support. Both low-level components
(`ewald_real_space`, `ewald_reciprocal_space`) already had full support,
and `particle_mesh_ewald()` already exposed the parameter — Ewald was
simply never wired up.

This blocked benchmarking `ewald_cg` through the public API (had to call
real-space + reciprocal-space manually and combine results).

### Changes

**`nvalchemiops/interactions/electrostatics/ewald.py`** (+49 lines)

- Added `compute_charge_gradients: bool = False` parameter to
  `ewald_summation()` signature
- Updated return type annotation:
  `torch.Tensor | tuple[..., 2] | tuple[..., 3]`
- Added docstring: parameter description, return value, return patterns,
  usage example
- Passes `compute_charge_gradients` through to both `ewald_real_space()`
  and `ewald_reciprocal_space()`
- Updated result combination logic to handle all 4 return patterns,
  mirroring `particle_mesh_ewald()`:
  - `forces=F, cg=F` → `energies`
  - `forces=T, cg=F` → `(energies, forces)`
  - `forces=F, cg=T` → `(energies, charge_gradients)`
  - `forces=T, cg=T` → `(energies, forces, charge_gradients)`

**`test/interactions/electrostatics/test_ewald.py`** (+212 lines)

Added 5 new test methods to `TestEwaldSummationAPI` (10 tests total with
CUDA/CPU parametrization):

| Test | What it validates |
|------|-------------------|
| `test_single_system_charge_gradients_only` | `cg=True, forces=False` returns `(energies, charge_grads)` with correct shapes |
| `test_single_system_forces_and_charge_gradients` | `cg=True, forces=True` returns all 3 tensors, energy is negative (attraction) |
| `test_batch_system_charge_gradients` | Batch mode (2 systems) with both flags returns correct shapes |
| `test_charge_gradients_match_autograd` | Explicit charge grads match `torch.autograd` ∂E/∂q (rtol=1e-4) |
| `test_charge_gradients_consistent_with_components` | `ewald_summation` charge grads == `ewald_real_space` + `ewald_reciprocal_space` charge grads (rtol=1e-10) |

### Test results

```
18 passed in 35.75s  (all TestEwaldSummationAPI — 8 existing + 10 new)
```

All pass on both CUDA and CPU. No regressions.

---

## Benchmark overhaul (in progress)

### Scope

Full rewrite of `benchmarks/` to support:
- **2 systems**: CsCl (crystal, programmatic) + NH3 (molecular, Packmol PDBs)
- **3 scaling modes**: system_size, constant_workload, batch_scaling
- **3 modules**: Neighbor List, DFT-D3 Dispersion, Electrostatics (PME + Ewald)
- **3 metrics per run**: time per atom (μs), throughput (10⁶ atoms/s), peak memory (GB)

### Files changed/added

**Core infrastructure:**

| File | Status | Description |
|------|--------|-------------|
| `benchmarks/utils.py` | Modified | Streamlined `clean_gpu()`, optimized `measure_memory()`, added `create_run_directory()`, `make_csv_name()`, `make_plot_name()`, `write_run_readme()` |
| `benchmarks/systems.py` | Modified | Natural-sort NH3 PDBs (removed `natsort` dep), `max_total_atoms` cap for constant workload |
| `benchmarks/benchmark_suite.py` | **New** | Unified entry point: loads per-module YAMLs, applies CLI overrides, dispatches in-process, generates README |
| `benchmarks/__init__.py` | Modified | Package init updates |
| `benchmarks/interactions/__init__.py` | **New** | Package init for interactions subpackage |

**Per-module benchmarks:**

| File | Status | Key changes |
|------|--------|-------------|
| `benchmarks/neighborlist/benchmark_config.yaml` | Modified | Renamed scaling mode keys, `max_total_atoms` |
| `benchmarks/neighborlist/benchmark_neighborlist.py` | Modified | Integrated memory into warmup, single `clean_gpu()` per size group, removed cell-size skip |
| `benchmarks/interactions/dispersion/benchmark_config.yaml` | Modified | Same scaling key renames, `max_total_atoms` |
| `benchmarks/interactions/dispersion/benchmark_dftd3.py` | Modified | Same optimizations as NL, removed cell-size skip |
| `benchmarks/interactions/electrostatics/benchmark_config.yaml` | Modified | Same scaling key renames, `max_total_atoms` |
| `benchmarks/interactions/electrostatics/benchmark_electrostatics.py` | Modified | Same optimizations, PME + Ewald with charge gradients |

**Plotting:**

| File | Status | Description |
|------|--------|-------------|
| `benchmarks/plotting/__init__.py` | **New** | Package init |
| `benchmarks/plotting/styles.py` | **New** | NVIDIA-branded styling: colors, log2 x-axis, table legends, VRAM refs, chemical formatting |
| `benchmarks/plotting/plot_benchmarks.py` | **New** | Generates 18 figures (3-panel: time, throughput, memory) for all system×mode combinations |

**NH3 system data:**

| File | Status | Description |
|------|--------|-------------|
| `benchmarks/nh3/` | **New** | Packmol-generated PDB files (128–131072 atoms), generation scripts |

**Other:**

| File | Status | Description |
|------|--------|-------------|
| `benchmarks/run_all_benchmarks.sh` | **New** | Master script for full suite execution |
| `benchmarks-original/` | **New** | Backup of original benchmarks before overhaul |
| `docs/benchmarks/neighborlist.md` | Modified | Updated for new output structure |
| `pyproject.toml` | Modified | Added `pynvml` dependency |

**Removed:**

| File | Reason |
|------|--------|
| `benchmarks/benchmark-requires.txt` | Replaced by `pyproject.toml` deps |
| `benchmarks/neighborlist/README.rst` | Replaced by generated `README.md` per run |
| `benchmarks/interactions/dispersion/README.rst` | Same |
| `benchmarks/interactions/dispersion/validate_d3_energies.py` | Separate validation script, not part of benchmark suite |

### Design decisions

- **Per-module YAML configs** (not a single monolithic YAML) — per engineering team guidance
- **Batch timing** (sync outside loop) instead of per-run timing (sync per run) — better for throughput measurement
- **In-process dispatch** from `benchmark_suite.py` — no subprocess overhead, shared GPU context
- **Peak VRAM** tracked (not delta) — more actionable for users planning GPU allocation
- **`natsort` dependency removed** — replaced with regex-based natural sort in `systems.py`

### Pending

- [ ] Sphinx docs update: tabbed interface (2 rows × 3 modes) per module
- [ ] Final review of all 18 plots
- [ ] Re-run EL system_size with `ewald_cg` (now unblocked by API fix above)
- [ ] `docs/benchmarks/generate_plots.py` update for new naming convention

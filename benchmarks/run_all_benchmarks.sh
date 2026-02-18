#!/bin/bash
# Batch benchmark run: NL + D3 + Electrostatics, both systems, all modes
# Then generate all plots and update sphinx docs
# Run from benchmarks/ directory
set -e

BENCHMARKS_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BENCHMARKS_DIR"

TIMING_RUNS=5
echo "========================================================"
echo "FULL BATCH BENCHMARK + PLOTTING + SPHINX"
echo "Timing runs: $TIMING_RUNS"
echo "Working dir: $BENCHMARKS_DIR"
echo "========================================================"

# =================================================================
# Step 1: Generate NH3 PDBs
# =================================================================
echo ""
echo ">>> Step 1/10: Generating NH3 PDB files..."
cd nh3
echo "medium" | bash generate_pbc_pdbs.sh || {
    echo "WARNING: Packmol generation had issues, continuing with available PDBs..."
}
cd "$BENCHMARKS_DIR"
echo "NH3 PDBs available: $(ls nh3/ammonia_pbc_*.pdb 2>/dev/null | wc -l)"

# =================================================================
# NL Benchmarks (Steps 2-5)
# =================================================================

echo ""
echo ">>> Step 2/10: NL CsCl constant_atoms_per_system..."
cd neighborlist
uv run python benchmark_neighborlist.py \
    --config benchmark_config.yaml \
    --system cscl --mode constant_atoms_per_system \
    --timing-runs $TIMING_RUNS
cd "$BENCHMARKS_DIR"

echo ""
echo ">>> Step 3/10: NL NH3 system_size..."
cd neighborlist
uv run python benchmark_neighborlist.py \
    --config benchmark_config.yaml \
    --system nh3 --mode system_size \
    --timing-runs $TIMING_RUNS
cd "$BENCHMARKS_DIR"

echo ""
echo ">>> Step 4/10: NL NH3 constant_total..."
cd neighborlist
uv run python benchmark_neighborlist.py \
    --config benchmark_config.yaml \
    --system nh3 --mode constant_total \
    --timing-runs $TIMING_RUNS
cd "$BENCHMARKS_DIR"

echo ""
echo ">>> Step 5/10: NL NH3 constant_atoms_per_system..."
cd neighborlist
uv run python benchmark_neighborlist.py \
    --config benchmark_config.yaml \
    --system nh3 --mode constant_atoms_per_system \
    --timing-runs $TIMING_RUNS
cd "$BENCHMARKS_DIR"

# =================================================================
# D3 Benchmarks (Steps 6-7)
# =================================================================

echo ""
echo ">>> Step 6/10: D3 CsCl all modes..."
cd interactions/dispersion
uv run python benchmark_dftd3.py \
    --config benchmark_config.yaml \
    --system cscl --mode all \
    --timing-runs $TIMING_RUNS
cd "$BENCHMARKS_DIR"

echo ""
echo ">>> Step 7/10: D3 NH3 all modes..."
cd interactions/dispersion
uv run python benchmark_dftd3.py \
    --config benchmark_config.yaml \
    --system nh3 --mode all \
    --timing-runs $TIMING_RUNS
cd "$BENCHMARKS_DIR"

# =================================================================
# Electrostatics Benchmarks (Steps 8-9)
# =================================================================

echo ""
echo ">>> Step 8/10: Electrostatics CsCl all modes..."
cd interactions/electrostatics
uv run python benchmark_electrostatics.py \
    --config benchmark_config.yaml \
    --system cscl --mode all \
    --timing-runs $TIMING_RUNS
cd "$BENCHMARKS_DIR"

echo ""
echo ">>> Step 9/10: Electrostatics NH3 all modes..."
cd interactions/electrostatics
uv run python benchmark_electrostatics.py \
    --config benchmark_config.yaml \
    --system nh3 --mode all \
    --timing-runs $TIMING_RUNS
cd "$BENCHMARKS_DIR"

# =================================================================
# Plotting (Step 10)
# =================================================================

echo ""
echo ">>> Step 10/10: Generating all plots..."
for dir in benchmark-results/nl_* benchmark-results/d3_* benchmark-results/el_*; do
    if [ -d "$dir" ]; then
        echo "  Plotting $dir..."
        uv run python plotting/plot_benchmarks.py "$dir" || echo "  (plot failed for $dir)"
    fi
done

# =================================================================
# Summary
# =================================================================

echo ""
echo "========================================================"
echo "ALL BENCHMARKS + PLOTS COMPLETE"
echo "========================================================"
echo "Results:"
for dir in benchmark-results/nl_* benchmark-results/d3_* benchmark-results/el_*; do
    if [ -d "$dir" ]; then
        csv_count=$(ls "$dir"/*.csv 2>/dev/null | wc -l)
        png_count=$(ls "$dir"/*.png 2>/dev/null | wc -l)
        echo "  $dir: ${csv_count} CSVs, ${png_count} PNGs"
    fi
done
echo "========================================================"

#!/usr/bin/env bash
# Run all electrostatics benchmarks: all methods, dtypes, formats.
# Usage: bash run_all_benchmarks.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/benchmark_config.yaml"
OUTPUT_DIR="${SCRIPT_DIR}/benchmark_results"

mkdir -p "${OUTPUT_DIR}"

for dtype in float32 float64; do
  for method in both dsf; do
    echo "========================================"
    echo "  method=${method}  dtype=${dtype}"
    echo "========================================"
    python "${SCRIPT_DIR}/benchmark_electrostatics.py" \
      --config "${CONFIG}" \
      --output-dir "${OUTPUT_DIR}" \
      --method "${method}" \
      --backend both \
      --neighbor-format both \
      --dtype "${dtype}"
  done
done

echo ""
echo "All benchmarks finished. Results in ${OUTPUT_DIR}/"

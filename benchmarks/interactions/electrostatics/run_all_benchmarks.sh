#!/usr/bin/env bash
# Run the extended point-charge, slab, and DSF matrix across dtypes/formats.
# Usage: bash run_all_benchmarks.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/benchmark_config_extended.yaml"
OUTPUT_DIR="${SCRIPT_DIR}/benchmark_results"

mkdir -p "${OUTPUT_DIR}"

if ! uv run python -c "import loguru, pymatgen, rdkit"; then
  echo "Install benchmarks/benchmark-requires.txt before running the extended suite." >&2
  exit 1
fi

for dtype in float32 float64; do
  for method in both ewald_slab pme_slab dsf; do
    echo "========================================"
    echo "  method=${method}  dtype=${dtype}"
    echo "========================================"
    uv run python "${SCRIPT_DIR}/benchmark_electrostatics.py" \
      --config "${CONFIG}" \
      --output-dir "${OUTPUT_DIR}" \
      --method "${method}" \
      --backend both \
      --neighbor-format both \
      --dtype "${dtype}"
  done
done

echo ""
echo "Extended point-charge/slab/DSF benchmarks finished. Results in ${OUTPUT_DIR}/"

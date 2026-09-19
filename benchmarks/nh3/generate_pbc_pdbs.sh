#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Generate PDB files with CRYST1 records for periodic boundary conditions
# Cell length formula: L = (41.47 * N_molecules)^(1/3)
# where N_molecules = N_atoms / 4 (NH3 has 4 atoms)
#
# These benchmark systems are consistent with the NVIDIA ALCHEMI Toolkit-Ops blog:
# https://developer.nvidia.com/blog/accelerating-ai-powered-chemistry-and-materials-science-simulations-with-nvidia-alchemi-toolkit-ops/
# "Test systems consisted of ammonia clusters of increasing size packed into various cells using Packmol."
#
# Packmol version: 21.1.4
# Random seed: 12345 (fixed for reproducibility)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SOURCE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
PACKMOL_VERSION="21.1.4"
OUTPUT_DIR="${NH3_OUTPUT_DIR:-}"
selection=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Error handling function
error_exit() {
    local msg=$1
    echo -e "${RED}ERROR: ${msg}${NC}" >&2
    exit 1
}

# Warning function
warn() {
    local msg=$1
    echo -e "${YELLOW}WARNING: ${msg}${NC}" >&2
    return 0
}

# Info function
info() {
    local msg=$1
    echo -e "${BLUE}${msg}${NC}"
    return 0
}

# Success function
success() {
    local msg=$1
    echo -e "${GREEN}${msg}${NC}"
    return 0
}

usage() {
    cat <<'USAGE'
Usage:
  benchmarks/nh3/generate_pbc_pdbs.sh --output-dir DIR [options]

Options:
  --output-dir DIR     Directory for generated PDBs, Packmol inputs, and logs.
                       Required unless NH3_OUTPUT_DIR is set. The directory
                       must be outside the source checkout.
  --selection VALUE   Sizes to generate, using the interactive prompt syntax
                       (for example: 1-11, "1 3 5", small, medium, or all).
                       If omitted, the selection is read interactively/stdin.
  -h, --help          Show this help.

Environment:
  NH3_OUTPUT_DIR      Same as --output-dir.
  PACKMOL_BIN         Installed Packmol executable to use. If unset, an
                       installed packmol is preferred; otherwise uvx runs the
                       pinned packmol==21.1.4 package.
USAGE
    return 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            [[ $# -ge 2 ]] || error_exit "Missing value for --output-dir."
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --selection)
            [[ $# -ge 2 ]] || error_exit "Missing value for --selection."
            selection="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error_exit "Unexpected argument: $1. Run with --help for usage."
            ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    error_exit "An output directory is required. Pass --output-dir DIR or set NH3_OUTPUT_DIR."
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$(pwd -P)/$OUTPUT_DIR"
fi
case "$OUTPUT_DIR" in
    "$SOURCE_ROOT"|"$SOURCE_ROOT"/*)
        error_exit "Output directory must be outside the source checkout: $OUTPUT_DIR"
        ;;
    *)
        ;;
esac
mkdir -p "$OUTPUT_DIR" || error_exit "Failed to create output directory: $OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd -P)"
case "$OUTPUT_DIR" in
    "$SOURCE_ROOT"|"$SOURCE_ROOT"/*)
        error_exit "Output directory must be outside the source checkout: $OUTPUT_DIR"
        ;;
    *)
        ;;
esac

# Progress bar function
# Usage: progress_bar current total prefix
progress_bar() {
    local current=$1
    local total=$2
    local prefix=${3:-"Progress"}
    local width=40
    local percent=$((current * 100 / total))
    local filled=$((current * width / total))

    # Build the bar from a blank canvas sliced at `filled`, rather than appending
    # one character at a time. The leading run becomes solid glyphs, whatever
    # spaces remain become the empty ones.
    local canvas
    printf -v canvas "%*s" "$width" ""
    local bar="${canvas:0:filled}"
    bar="${bar// /█}${canvas:filled}"
    bar="${bar// /░}"

    printf "\r${prefix}: [${bar}] %3d%% (%d/%d)" "$percent" "$current" "$total"
    return 0
}

# Function to calculate cell length
calc_cell_length() {
    local n_atoms=$1
    local n_mols=$((n_atoms / 4))
    # L = (41.47 * N)^(1/3)
    python3 -c "print(f'{(41.47 * $n_mols) ** (1/3):.3f}')" 2>/dev/null || \
        error_exit "Failed to calculate cell length. Is python3 installed?"
    return 0
}

# Function to add CRYST1 record to PDB
add_cryst_record() {
    local pdb_file=$1
    local cell_length=$2
    local tmp_file="${pdb_file}.tmp"

    # CRYST1 format: a, b, c, alpha, beta, gamma, space group
    # Cubic box: a=b=c, alpha=beta=gamma=90
    printf "CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1           1\n" \
        "$cell_length" "$cell_length" "$cell_length" > "$tmp_file" || \
        error_exit "Failed to create temp file for CRYST1 record"

    cat "$pdb_file" >> "$tmp_file" || error_exit "Failed to append PDB content"
    mv "$tmp_file" "$pdb_file" || error_exit "Failed to finalize PDB file"
    return 0
}

# Function to format atom count for display (powers of 2)
format_atoms() {
    local n=$1
    if [[ "$n" -ge 1048576 ]]; then
        # 1M = 1024*1024 = 1048576
        echo "$((n / 1048576))M"
    elif [[ "$n" -ge 1024 ]]; then
        # 1k = 1024
        echo "$((n / 1024))k"
    else
        echo "$n"
    fi
    return 0
}

# =============================================================================
# VALIDATION CHECKS
# =============================================================================

echo ""
info "=== NH3 Benchmark Suite | Structure Generator ==="
echo ""
echo "Benchmark systems consistent with NVIDIA ALCHEMI Toolkit-Ops blog:"
echo "  https://developer.nvidia.com/blog/accelerating-ai-powered-chemistry-and-materials-science-simulations-with-nvidia-alchemi-toolkit-ops/"
echo "  \"Test systems consisted of ammonia clusters of increasing size packed into"
echo "   various cells using Packmol.\""
echo ""

# Check for python3
if ! command -v python3 &> /dev/null; then
    error_exit "python3 not found. Please install Python 3."
fi

# Check for Packmol. Prefer an installed executable so offline environments can
# opt out of uvx; pin the uvx package for reproducible fallback behavior.
PACKMOL_CMD=()
if [[ -n "${PACKMOL_BIN:-}" ]]; then
    if [[ "$PACKMOL_BIN" == */* ]]; then
        [[ -x "$PACKMOL_BIN" ]] || error_exit "PACKMOL_BIN is not executable: $PACKMOL_BIN"
        PACKMOL_CMD=("$PACKMOL_BIN")
    elif packmol_path="$(command -v "$PACKMOL_BIN" 2>/dev/null)"; then
        PACKMOL_CMD=("$packmol_path")
    else
        error_exit "PACKMOL_BIN was not found on PATH: $PACKMOL_BIN"
    fi
elif packmol_path="$(command -v packmol 2>/dev/null)"; then
    PACKMOL_CMD=("$packmol_path")
elif uvx_path="$(command -v uvx 2>/dev/null)"; then
    PACKMOL_CMD=("$uvx_path" --from "packmol==${PACKMOL_VERSION}" packmol)
else
    error_exit "Packmol not found. Install a packmol executable or install uv so this script can run: uvx --from packmol==${PACKMOL_VERSION} packmol"
fi

info "Using Packmol: ${PACKMOL_CMD[*]}"
info "Output directory: $OUTPUT_DIR"

# Check for ammonia.pdb template
TEMPLATE_FILE="$SCRIPT_DIR/ammonia.pdb"
if [[ ! -f "$TEMPLATE_FILE" ]]; then
    error_exit "Template file '$TEMPLATE_FILE' not found in $SCRIPT_DIR

This file should contain a single NH3 molecule in PDB format.
Example content:
  COMPND    AMMONIA
  HETATM    1  N   NH3     1       0.000   0.000   0.000  1.00  0.00           N
  HETATM    2  H1  NH3     1       0.939   0.000   0.381  1.00  0.00           H
  HETATM    3  H2  NH3     1      -0.469   0.813   0.381  1.00  0.00           H
  HETATM    4  H3  NH3     1      -0.469  -0.813   0.381  1.00  0.00           H
  END"
fi

# Validate ammonia.pdb has expected atoms
atom_count=$(grep -c "^HETATM\|^ATOM" "$TEMPLATE_FILE" 2>/dev/null || echo "0")
if [[ "$atom_count" -ne 4 ]]; then
    error_exit "Template '$TEMPLATE_FILE' should have exactly 4 atoms (1 N + 3 H), found: $atom_count"
fi

success "✓ Template file validated: $TEMPLATE_FILE (4 atoms)"
echo ""

# =============================================================================
# SIZE SELECTION
# =============================================================================

# Define available sizes (total atoms)
ALL_SIZES=(128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288)

echo "Available system sizes:"
echo ""
printf "  %-4s  %-10s  %s\n" "No." "Atoms" "Est. Time"
echo "  ─────────────────────────────────"
for i in "${!ALL_SIZES[@]}"; do
    n_atoms=${ALL_SIZES[$i]}
    formatted=$(format_atoms $n_atoms)

    # Rough time estimates
    if [[ "$n_atoms" -le 4096 ]]; then
        est_time="< 1s"
    elif [[ "$n_atoms" -le 32768 ]]; then
        est_time="1-10s"
    elif [[ "$n_atoms" -le 131072 ]]; then
        est_time="10s-2min"
    else
        est_time="2-30min"
    fi

    printf "  %-4s  %-10s  %s\n" "$((i+1)))" "$formatted" "$est_time"
done
echo ""

# Prompt for selection unless it was supplied non-interactively.
if [[ -z "$selection" ]]; then
    echo "Select sizes to generate:"
    echo "  - Enter numbers separated by spaces (e.g., '1 2 3')"
    echo "  - Enter a range (e.g., '1-5')"
    echo "  - Enter 'all' for all sizes"
    echo "  - Enter 'small' for 128-4096 atoms (1-6)"
    echo "  - Enter 'medium' for 128-65536 atoms (1-10)"
    echo "  - Press Ctrl+C to cancel"
    echo ""
    read -r -p "Your selection: " selection || \
        error_exit "No selection received. Pass --selection VALUE or provide it on stdin."
fi

# Parse selection
SELECTED_SIZES=()

if [[ -z "$selection" ]]; then
    error_exit "No selection made. Exiting."
fi

# Convert selection to lowercase
selection=$(echo "$selection" | tr '[:upper:]' '[:lower:]')

case "$selection" in
    "all")
        SELECTED_SIZES=("${ALL_SIZES[@]}")
        ;;
    "small")
        SELECTED_SIZES=(128 256 512 1024 2048 4096)
        ;;
    "medium")
        SELECTED_SIZES=(128 256 512 1024 2048 4096 8192 16384 32768 65536)
        ;;
    *)
        # Parse numbers and ranges
        for item in $selection; do
            if [[ "$item" =~ ^([0-9]+)-([0-9]+)$ ]]; then
                # Range
                start=${BASH_REMATCH[1]}
                end=${BASH_REMATCH[2]}
                if [[ "$start" -gt "$end" ]]; then
                    error_exit "Invalid range: $item (start > end)"
                fi
                for ((j=start; j<=end; j++)); do
                    if [[ "$j" -ge 1 ]] && [[ "$j" -le ${#ALL_SIZES[@]} ]]; then
                        SELECTED_SIZES+=("${ALL_SIZES[$((j-1))]}")
                    else
                        warn "Ignoring out-of-range index: $j"
                    fi
                done
            elif [[ "$item" =~ ^[0-9]+$ ]]; then
                # Single number
                if [[ "$item" -ge 1 ]] && [[ "$item" -le ${#ALL_SIZES[@]} ]]; then
                    SELECTED_SIZES+=("${ALL_SIZES[$((item-1))]}")
                else
                    warn "Ignoring out-of-range index: $item"
                fi
            else
                warn "Ignoring invalid input: $item"
            fi
        done
        ;;
esac

# Remove duplicates and sort
SELECTED_SIZES=($(printf '%s\n' "${SELECTED_SIZES[@]}" | sort -n | uniq))

if [[ ${#SELECTED_SIZES[@]} -eq 0 ]]; then
    error_exit "No valid sizes selected. Exiting."
fi

echo ""
info "Selected ${#SELECTED_SIZES[@]} size(s): ${SELECTED_SIZES[*]}"
echo ""

# =============================================================================
# GENERATION
# =============================================================================

total=${#SELECTED_SIZES[@]}
current=0
success_count=0
fail_count=0
failed_sizes=()

echo "Starting generation..."
echo ""

for n_atoms in "${SELECTED_SIZES[@]}"; do
    current=$((current + 1))
    n_mols=$((n_atoms / 4))
    cell_length=$(calc_cell_length $n_atoms)
    inp_file="$OUTPUT_DIR/ammonia_pbc_${n_atoms}.inp"
    pdb_file="$OUTPUT_DIR/ammonia_pbc_${n_atoms}.pdb"
    log_file="$OUTPUT_DIR/packmol_${n_atoms}.log"
    formatted=$(format_atoms $n_atoms)

    # Show progress
    progress_bar $((current - 1)) $total "Overall"
    echo ""
    info "[$current/$total] Generating ${formatted} atoms, cell=${cell_length} Å"

    # Create Packmol input file
    cat > "$inp_file" << EOF
tolerance 2.0
seed 12345
filetype pdb
output ${pdb_file}

structure ${TEMPLATE_FILE}
  number ${n_mols}
  inside cube 0.0 0.0 0.0 ${cell_length}
end structure
EOF

    if [[ ! -f "$inp_file" ]]; then
        warn "Failed to create input file: $inp_file"
        fail_count=$((fail_count + 1))
        failed_sizes+=("$n_atoms")
        continue
    fi

    # Run Packmol with output capture
    echo "  Running Packmol..."

    if (
        cd "$OUTPUT_DIR"
        "${PACKMOL_CMD[@]}" < "$inp_file" > "$log_file" 2>&1
    ); then
        # Check if output file was created
        if [[ -f "$pdb_file" ]]; then
            # Validate PDB has expected atom count
            actual_atoms=$(grep -Ec "^(HETATM|ATOM)" "$pdb_file" 2>/dev/null || true)
            actual_atoms=${actual_atoms:-0}
            if [[ "$actual_atoms" -ne "$n_atoms" ]]; then
                warn "Atom count mismatch: expected $n_atoms, got $actual_atoms"
                echo "  See log: $log_file"
                rm -f "$pdb_file"
                fail_count=$((fail_count + 1))
                failed_sizes+=("$n_atoms")
                continue
            fi

            # Add CRYST1 record
            add_cryst_record "$pdb_file" "$cell_length"
            success "  ✓ Created: $pdb_file ($actual_atoms atoms)"
            success_count=$((success_count + 1))

            # Clean up log file on success
            rm -f "$log_file"
        else
            warn "Packmol completed but output file not found: $pdb_file"
            echo "  See log: $log_file"
            fail_count=$((fail_count + 1))
            failed_sizes+=("$n_atoms")
        fi
    else
        exit_code=$?
        warn "Packmol failed with exit code: $exit_code"
        echo "  See log: $log_file"
        fail_count=$((fail_count + 1))
        failed_sizes+=("$n_atoms")
    fi
done

# Final progress
progress_bar $total $total "Overall"
echo ""
echo ""

# =============================================================================
# SUMMARY
# =============================================================================

info "=== Generation Complete ==="
echo ""
success "  Successful: $success_count"
if [[ $fail_count -gt 0 ]]; then
    echo -e "  ${RED}Failed: $fail_count (${failed_sizes[*]})${NC}"
fi
echo ""

if [[ $success_count -gt 0 ]]; then
    echo "Generated files:"
    for n_atoms in "${SELECTED_SIZES[@]}"; do
        pdb_file="$OUTPUT_DIR/ammonia_pbc_${n_atoms}.pdb"
        if [[ -f "$pdb_file" ]]; then
            cell_length=$(calc_cell_length $n_atoms)
            size=$(du -h "$pdb_file" | cut -f1)
            formatted=$(format_atoms $n_atoms)
            printf "  %-25s  %6s atoms  %8s Å  %6s\n" \
                "$pdb_file" "$formatted" "$cell_length" "$size"
        fi
    done
fi

echo ""
if [[ "$fail_count" -gt 0 ]]; then
    error_exit "Generation failed for: ${failed_sizes[*]}. See Packmol logs in $OUTPUT_DIR."
fi
success "Done!"

#!/bin/bash
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

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Error handling function
error_exit() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

# Warning function
warn() {
    echo -e "${YELLOW}WARNING: $1${NC}" >&2
}

# Info function
info() {
    echo -e "${BLUE}$1${NC}"
}

# Success function
success() {
    echo -e "${GREEN}$1${NC}"
}

# Progress bar function
# Usage: progress_bar current total prefix
progress_bar() {
    local current=$1
    local total=$2
    local prefix=${3:-"Progress"}
    local width=40
    local percent=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))

    # Build the bar
    local bar=""
    for ((i=0; i<filled; i++)); do bar+="█"; done
    for ((i=0; i<empty; i++)); do bar+="░"; done

    printf "\r${prefix}: [${bar}] %3d%% (%d/%d)" "$percent" "$current" "$total"
}

# Function to calculate cell length
calc_cell_length() {
    local n_atoms=$1
    local n_mols=$((n_atoms / 4))
    # L = (41.47 * N)^(1/3)
    python3 -c "print(f'{(41.47 * $n_mols) ** (1/3):.3f}')" 2>/dev/null || \
        error_exit "Failed to calculate cell length. Is python3 installed?"
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
}

# Function to format atom count for display (powers of 2)
format_atoms() {
    local n=$1
    if [ "$n" -ge 1048576 ]; then
        # 1M = 1024*1024 = 1048576
        echo "$((n / 1048576))M"
    elif [ "$n" -ge 1024 ]; then
        # 1k = 1024
        echo "$((n / 1024))k"
    else
        echo "$n"
    fi
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

# Check for packmol
if command -v uvx &> /dev/null; then
    PACKMOL="uvx packmol"
elif command -v packmol &> /dev/null; then
    PACKMOL="packmol"
else
    error_exit "packmol not found. Install via: pip install packmol  OR  uvx packmol"
fi

info "Using: $PACKMOL"

# Check for ammonia.pdb template
TEMPLATE_FILE="ammonia.pdb"
if [ ! -f "$TEMPLATE_FILE" ]; then
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
if [ "$atom_count" -ne 4 ]; then
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
    if [ "$n_atoms" -le 4096 ]; then
        est_time="< 1s"
    elif [ "$n_atoms" -le 32768 ]; then
        est_time="1-10s"
    elif [ "$n_atoms" -le 131072 ]; then
        est_time="10s-2min"
    else
        est_time="2-30min"
    fi

    printf "  %-4s  %-10s  %s\n" "$((i+1)))" "$formatted" "$est_time"
done
echo ""

# Prompt for selection
echo "Select sizes to generate:"
echo "  - Enter numbers separated by spaces (e.g., '1 2 3')"
echo "  - Enter a range (e.g., '1-5')"
echo "  - Enter 'all' for all sizes"
echo "  - Enter 'small' for 128-4096 atoms (1-6)"
echo "  - Enter 'medium' for 128-65536 atoms (1-10)"
echo "  - Press Ctrl+C to cancel"
echo ""
read -p "Your selection: " selection

# Parse selection
SELECTED_SIZES=()

if [ -z "$selection" ]; then
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
                if [ "$start" -gt "$end" ]; then
                    error_exit "Invalid range: $item (start > end)"
                fi
                for ((j=start; j<=end; j++)); do
                    if [ "$j" -ge 1 ] && [ "$j" -le ${#ALL_SIZES[@]} ]; then
                        SELECTED_SIZES+=("${ALL_SIZES[$((j-1))]}")
                    else
                        warn "Ignoring out-of-range index: $j"
                    fi
                done
            elif [[ "$item" =~ ^[0-9]+$ ]]; then
                # Single number
                if [ "$item" -ge 1 ] && [ "$item" -le ${#ALL_SIZES[@]} ]; then
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

if [ ${#SELECTED_SIZES[@]} -eq 0 ]; then
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
    inp_file="ammonia_pbc_${n_atoms}.inp"
    pdb_file="ammonia_pbc_${n_atoms}.pdb"
    log_file="packmol_${n_atoms}.log"
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

structure ammonia.pdb
  number ${n_mols}
  inside cube 0.0 0.0 0.0 ${cell_length}
end structure
EOF

    if [ ! -f "$inp_file" ]; then
        warn "Failed to create input file: $inp_file"
        fail_count=$((fail_count + 1))
        failed_sizes+=("$n_atoms")
        continue
    fi

    # Run Packmol with output capture
    echo "  Running Packmol..."

    if $PACKMOL < "$inp_file" > "$log_file" 2>&1; then
        # Check if output file was created
        if [ -f "$pdb_file" ]; then
            # Validate PDB has expected atom count
            actual_atoms=$(grep -c "^HETATM\|^ATOM" "$pdb_file" 2>/dev/null || echo "0")
            if [ "$actual_atoms" -ne "$n_atoms" ]; then
                warn "Atom count mismatch: expected $n_atoms, got $actual_atoms"
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
if [ $fail_count -gt 0 ]; then
    echo -e "  ${RED}Failed: $fail_count (${failed_sizes[*]})${NC}"
fi
echo ""

if [ $success_count -gt 0 ]; then
    echo "Generated files:"
    for n_atoms in "${SELECTED_SIZES[@]}"; do
        pdb_file="ammonia_pbc_${n_atoms}.pdb"
        if [ -f "$pdb_file" ]; then
            cell_length=$(calc_cell_length $n_atoms)
            size=$(du -h "$pdb_file" | cut -f1)
            formatted=$(format_atoms $n_atoms)
            printf "  %-25s  %6s atoms  %8s Å  %6s\n" \
                "$pdb_file" "$formatted" "$cell_length" "$size"
        fi
    done
fi

echo ""
success "Done!"

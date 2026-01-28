#!/bin/bash
# =============================================================================
# Benchmark Suite Test Script
# =============================================================================
# Tests the benchmark suite with various configurations before sharing.
# Run this to validate everything works correctly.
#
# Usage:
#   bash test_benchmark_suite.sh           # Run all tests
#   bash test_benchmark_suite.sh --quick   # Quick smoke test only
#   bash test_benchmark_suite.sh --full    # Full test with all benchmarks
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
PASS=0
FAIL=0
SKIP=0

# Test output directory
TEST_OUTPUT_DIR="$SCRIPT_DIR/test-output-$(date +%Y%m%d_%H%M%S)"

# Parse arguments
TEST_MODE="standard"
if [[ "$1" == "--quick" ]]; then
    TEST_MODE="quick"
elif [[ "$1" == "--full" ]]; then
    TEST_MODE="full"
fi

# =============================================================================
# Helper Functions
# =============================================================================

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    PASS=$((PASS + 1))
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    FAIL=$((FAIL + 1))
}

skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    SKIP=$((SKIP + 1))
}

run_test() {
    local name="$1"
    local cmd="$2"
    local expected_files="${3:-}"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    info "Test: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Command: $cmd"
    echo ""
    
    # Run command and capture output
    local start_time=$(date +%s)
    if eval "$cmd" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Check for expected output files if specified
        if [[ -n "$expected_files" ]]; then
            local all_found=true
            for f in $expected_files; do
                if [[ ! -f "$f" ]]; then
                    all_found=false
                    echo -e "${RED}Missing expected file: $f${NC}"
                fi
            done
            if $all_found; then
                pass "$name (${duration}s)"
            else
                fail "$name - missing output files"
            fi
        else
            pass "$name (${duration}s)"
        fi
    else
        local exit_code=$?
        fail "$name (exit code: $exit_code)"
    fi
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║           BENCHMARK SUITE TEST SCRIPT                                 ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Test mode: $TEST_MODE"
echo "Output directory: $TEST_OUTPUT_DIR"
echo ""

# Check Python environment
info "Checking Python environment..."
if command -v uv &> /dev/null; then
    PYTHON="uv run python"
    echo "  Using: uv run python"
else
    PYTHON="python"
    echo "  Using: python"
fi

# Check CUDA availability
info "Checking CUDA..."
if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CUDA_DEVICE=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))")
    pass "CUDA available: $CUDA_DEVICE"
else
    fail "CUDA not available"
    echo "Benchmark suite requires GPU. Exiting."
    exit 1
fi

# Check nvalchemiops import
info "Checking nvalchemiops..."
if $PYTHON -c "import nvalchemiops" 2>/dev/null; then
    VERSION=$($PYTHON -c "import nvalchemiops; print(nvalchemiops.__version__)")
    pass "nvalchemiops version: $VERSION"
else
    fail "nvalchemiops not installed"
    echo "Install with: uv sync"
    exit 1
fi

# Check natsort import
info "Checking natsort..."
if $PYTHON -c "from natsort import natsorted" 2>/dev/null; then
    pass "natsort available"
else
    fail "natsort not installed"
    echo "Install with: uv sync --group benchmark"
    exit 1
fi

# Check PDB files
info "Checking NH3 PDB files..."
PDB_COUNT=$(ls -1 nh3/ammonia_pbc_*.pdb 2>/dev/null | wc -l)
if [[ $PDB_COUNT -gt 0 ]]; then
    pass "Found $PDB_COUNT PDB files in nh3/"
else
    fail "No PDB files found in nh3/"
    echo "Generate with: cd nh3 && bash generate_pbc_pdbs.sh"
    exit 1
fi

# Check D3 parameters
info "Checking D3 parameters..."
D3_PARAMS="$HOME/.cache/nvalchemiops/dftd3_parameters.pt"
if [[ -f "$D3_PARAMS" ]]; then
    pass "D3 parameters found: $D3_PARAMS"
    HAS_D3_PARAMS=true
else
    skip "D3 parameters not found (D3 benchmarks will be skipped)"
    HAS_D3_PARAMS=false
fi

# Create test output directory
mkdir -p "$TEST_OUTPUT_DIR"

# =============================================================================
# Test: Help and CLI
# =============================================================================

run_test "CLI --help" \
    "$PYTHON benchmark_suite.py --help"

# =============================================================================
# Test: Dry-run validation (parse args only)
# =============================================================================

run_test "Parse arguments: NL benchmark" \
    "$PYTHON -c \"
import sys
sys.argv = ['benchmark_suite.py', '-b', 'nl', '--cutoffs', '6', '--timing-runs', '1']
from benchmark_suite import parse_args
args = parse_args()
assert 'nl' in args.benchmark
assert args.cutoffs == [6.0]
print('Args parsed successfully')
\""

# =============================================================================
# Test: Actual Benchmarks
# =============================================================================

case $TEST_MODE in
    "quick")
        # Quick smoke test - smallest system, minimal runs
        run_test "NL benchmark (quick)" \
            "$PYTHON benchmark_suite.py -b nl --cutoffs 6 --timing-runs 2 --target-atoms 1024 --output-base $TEST_OUTPUT_DIR"
        ;;
        
    "standard")
        # Standard test - small systems, reasonable coverage
        run_test "NL benchmark (cell+naive, 6Å)" \
            "$PYTHON benchmark_suite.py -b nl --cutoffs 6 --timing-runs 5 --target-atoms 4096 --output-base $TEST_OUTPUT_DIR"
        
        if $HAS_D3_PARAMS; then
            run_test "D3 scaling benchmark (15Å)" \
                "$PYTHON benchmark_suite.py -b d3s --cutoffs 15 --timing-runs 5 --output-base $TEST_OUTPUT_DIR"
        else
            skip "D3 benchmark (no D3 params)"
        fi
        
        run_test "Electrostatics scaling (1e-4 accuracy)" \
            "$PYTHON benchmark_suite.py -b el --accuracy 1e-4 --timing-runs 5 --max-atoms 4096 --output-base $TEST_OUTPUT_DIR"
        ;;
        
    "full")
        # Full test - all benchmarks with multiple configurations
        run_test "NL benchmark (all cutoffs)" \
            "$PYTHON benchmark_suite.py -b nl --cutoffs 6 15 --timing-runs 10 --target-atoms 16384 --output-base $TEST_OUTPUT_DIR"
        
        if $HAS_D3_PARAMS; then
            run_test "D3 batched benchmark" \
                "$PYTHON benchmark_suite.py -b d3 --cutoffs 15 --timing-runs 10 --target-atoms 16384 --output-base $TEST_OUTPUT_DIR"
            
            run_test "D3 scaling benchmark" \
                "$PYTHON benchmark_suite.py -b d3s --cutoffs 15 25 --timing-runs 10 --output-base $TEST_OUTPUT_DIR"
        else
            skip "D3 benchmarks (no D3 params)"
        fi
        
        run_test "Electrostatics scaling" \
            "$PYTHON benchmark_suite.py -b el --accuracy 1e-4 --timing-runs 10 --max-atoms 16384 --output-base $TEST_OUTPUT_DIR"
        
        run_test "Electrostatics batched" \
            "$PYTHON benchmark_suite.py -b elb --accuracy 1e-4 --timing-runs 10 --target-atoms 16384 --output-base $TEST_OUTPUT_DIR"
        ;;
esac

# =============================================================================
# Test: Output Validation
# =============================================================================

info "Validating output files..."
echo ""

# Check for CSV files
CSV_COUNT=$(find "$TEST_OUTPUT_DIR" -name "*.csv" 2>/dev/null | wc -l)
if [[ $CSV_COUNT -gt 0 ]]; then
    pass "Generated $CSV_COUNT CSV file(s)"
    echo "  Output files:"
    find "$TEST_OUTPUT_DIR" -name "*.csv" -exec echo "    - {}" \;
else
    fail "No CSV files generated"
fi

# Validate CSV content (non-empty, has headers)
for csv in "$TEST_OUTPUT_DIR"/*/*.csv; do
    if [[ -f "$csv" ]]; then
        lines=$(wc -l < "$csv")
        if [[ $lines -gt 1 ]]; then
            pass "$(basename $csv): $((lines-1)) data rows"
        else
            fail "$(basename $csv): empty or header-only"
        fi
    fi
done

# =============================================================================
# Test: Plotting Script
# =============================================================================

if [[ $TEST_MODE != "quick" ]]; then
    info "Testing plot script (syntax check only)..."
    if $PYTHON -m py_compile plot_combined_benchmarks.py 2>/dev/null; then
        pass "plot_combined_benchmarks.py syntax OK"
    else
        fail "plot_combined_benchmarks.py syntax error"
    fi
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                         TEST SUMMARY                                  ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "  ${GREEN}PASSED:${NC}  $PASS"
echo -e "  ${RED}FAILED:${NC}  $FAIL"
echo -e "  ${YELLOW}SKIPPED:${NC} $SKIP"
echo ""
echo "  Output directory: $TEST_OUTPUT_DIR"
echo ""

if [[ $FAIL -eq 0 ]]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✓ ALL TESTS PASSED - Safe to share!                                  ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ✗ SOME TESTS FAILED - Review errors above                            ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi

#!/bin/bash
#
# Update WPT WebNN Test Data
#
# This script automates updating test data from the W3C Web Platform Tests repository.
# It handles cloning/updating the WPT repo and converting test files to JSON format.
#
# Usage:
#   ./scripts/update_wpt_tests.sh                    # Update all operations
#   ./scripts/update_wpt_tests.sh --operations relu,add  # Update specific operations
#   ./scripts/update_wpt_tests.sh --help             # Show help
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WPT_REPO_DIR="${HOME}/.cache/rustnn/wpt"
WPT_REPO_URL="https://github.com/web-platform-tests/wpt.git"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Default options
UPDATE_ALL=true
OPERATIONS=""
FORCE_CLONE=false
SKIP_PULL=false

# Print colored message
print_status() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC}  $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Show usage
show_help() {
    cat <<EOF
Update WPT WebNN Test Data

Usage:
  $0 [OPTIONS]

Options:
  --operations OPERATIONS   Comma-separated list of operations to update
                           (default: all available operations)
  --force-clone            Force fresh clone of WPT repository
  --skip-pull              Skip git pull (use existing WPT repo as-is)
  --help                   Show this help message

Examples:
  # Update all operations
  $0

  # Update specific operations only
  $0 --operations reduce_sum,reduce_mean,relu

  # Force fresh clone of WPT repo
  $0 --force-clone

  # Use existing WPT repo without pulling updates
  $0 --skip-pull

Environment:
  WPT_REPO_DIR    Directory for WPT repository (default: ~/.cache/rustnn/wpt)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --operations)
            OPERATIONS="$2"
            UPDATE_ALL=false
            shift 2
            ;;
        --force-clone)
            FORCE_CLONE=true
            shift
            ;;
        --skip-pull)
            SKIP_PULL=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

print_status "WPT Test Data Update Script"
echo

# Step 1: Clone or update WPT repository
print_status "Step 1: Preparing WPT repository"

if [ "$FORCE_CLONE" = true ] && [ -d "$WPT_REPO_DIR" ]; then
    print_warning "Removing existing WPT repository (--force-clone)"
    rm -rf "$WPT_REPO_DIR"
fi

if [ ! -d "$WPT_REPO_DIR" ]; then
    print_status "Cloning WPT repository..."
    mkdir -p "$(dirname "$WPT_REPO_DIR")"
    if git clone --depth 1 "$WPT_REPO_URL" "$WPT_REPO_DIR"; then
        print_success "WPT repository cloned to $WPT_REPO_DIR"
    else
        print_error "Failed to clone WPT repository"
        exit 1
    fi
else
    print_success "WPT repository exists at $WPT_REPO_DIR"

    if [ "$SKIP_PULL" = false ]; then
        print_status "Updating WPT repository..."
        cd "$WPT_REPO_DIR"
        if git pull --rebase; then
            print_success "WPT repository updated"
        else
            print_warning "Failed to update WPT repository (continuing with existing version)"
        fi
        cd "$PROJECT_ROOT"
    else
        print_warning "Skipping WPT repository update (--skip-pull)"
    fi
fi

# Verify WebNN tests exist
if [ ! -d "$WPT_REPO_DIR/webnn" ]; then
    print_error "WebNN tests not found in WPT repository"
    print_error "Expected directory: $WPT_REPO_DIR/webnn"
    exit 1
fi

print_success "WebNN tests found"
echo

# Step 2: Convert test data
print_status "Step 2: Converting test data to JSON"

CONVERTER_SCRIPT="$SCRIPT_DIR/convert_wpt_tests.py"

if [ ! -f "$CONVERTER_SCRIPT" ]; then
    print_error "Converter script not found: $CONVERTER_SCRIPT"
    exit 1
fi

# Build converter command
CONVERTER_CMD="python3 $CONVERTER_SCRIPT --wpt-repo $WPT_REPO_DIR --output $PROJECT_ROOT/tests/wpt_data"

if [ "$UPDATE_ALL" = true ]; then
    CONVERTER_CMD="$CONVERTER_CMD --all-operations"
else
    CONVERTER_CMD="$CONVERTER_CMD --operations $OPERATIONS"
fi

print_status "Running converter..."
echo "Command: $CONVERTER_CMD"
echo

if $CONVERTER_CMD; then
    print_success "Test data conversion completed"
else
    print_error "Test data conversion failed"
    exit 1
fi

echo

# Step 3: Show summary
print_status "Step 3: Summary"
echo

# Count test data files
CONFORMANCE_COUNT=$(find "$PROJECT_ROOT/tests/wpt_data/conformance" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
VALIDATION_COUNT=$(find "$PROJECT_ROOT/tests/wpt_data/validation" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')

echo "Test data files:"
echo "  Conformance: $CONFORMANCE_COUNT files"
echo "  Validation:  $VALIDATION_COUNT files"
echo

# Get WPT commit info
cd "$WPT_REPO_DIR"
WPT_COMMIT=$(git rev-parse --short HEAD)
WPT_DATE=$(git log -1 --format=%cd --date=short)
cd "$PROJECT_ROOT"

echo "WPT repository info:"
echo "  Location: $WPT_REPO_DIR"
echo "  Commit:   $WPT_COMMIT"
echo "  Date:     $WPT_DATE"
echo

print_success "Update complete!"
echo

# Next steps
print_status "Next steps:"
echo "  1. Review the generated/updated test data files in tests/wpt_data/"
echo "  2. Run WPT conformance tests: pytest tests/test_wpt_conformance.py -v"
echo "  3. Check for any failing tests and investigate"
echo

# Note about manual review
if [ "$CONFORMANCE_COUNT" -eq 0 ] && [ "$VALIDATION_COUNT" -eq 0 ]; then
    print_warning "No test data files were created."
    print_warning "The converter script may require manual test case population."
    print_warning "See: docs/wpt-integration-plan.md for guidance"
fi

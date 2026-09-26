#!/bin/bash
# Sync expected-failures file with current WPT results.
# Usage: ./scripts/update_expected_failures.sh <backend>
#
# 1. Empty the expected-failures file (every failure becomes visible).
# 2. Run WPT to collect actual failures.
# 3. Extract failures from the log, rebuild expected_failures.txt
#
# If no expected-failures file exists, one is created from the failures.

set -e

usage() {
    cat >&2 <<'EOF'
Usage: scripts/update_expected_failures.sh <backend>

  <backend>    onnx | trtx | litert | coreml | cann

EOF
    exit 2
}

BACKEND="${1:-}"
[ -z "$BACKEND" ] && usage

case "$BACKEND" in
    onnx)   MAKE_TARGET="test-wpt"       ;;
    trtx)   MAKE_TARGET="test-wpt-trtx"  ;;
    litert) MAKE_TARGET="test-wpt-litert" ;;
    coreml) MAKE_TARGET="test-wpt-coreml" ;;
    cann)   MAKE_TARGET="test-wpt-cann"   ;;
    *)      usage ;;
esac

PROJECT_DIR="$(dirname "$(cd "$(dirname "$0")" && pwd)")"
EXPECTED="$PROJECT_DIR/tests/wpt_conformance/${BACKEND}_expected_failures.txt"
BACKUP="${EXPECTED}.bak"
FAILURES="/tmp/wpt_${BACKEND}_failures.txt"
WPT_LOG="/tmp/wpt_${BACKEND}_sync.log"

_cleanup()   { rm -f "$FAILURES" "$WPT_LOG"; }
_restore()   { [ -f "$BACKUP" ] && [ ! -s "$EXPECTED" ] && cp "$BACKUP" "$EXPECTED"; }
_on_exit()   { _restore; _cleanup; }

trap _on_exit EXIT
trap '_on_exit; exit 130' INT TERM

cd "$PROJECT_DIR"

# ---- Run WPT ----

if [ -f "$EXPECTED" ]; then
    cp "$EXPECTED" "$BACKUP"
    echo "" > "$EXPECTED"
fi

set +e
make "$MAKE_TARGET" 2>&1 | tee "$WPT_LOG"
set -e

# A completed run always prints a "[WPT] result:" summary. If it is missing the
# run failed before finishing (build error, no device, ...): keep the baseline.
if ! grep -q -F '[WPT] result:' "$WPT_LOG"; then
    echo "error: WPT run did not complete (no '[WPT] result:' in ${WPT_LOG});" >&2
    echo "       keeping the existing baseline at ${EXPECTED}" >&2
    if [ -f "$BACKUP" ]; then
        cp "$BACKUP" "$EXPECTED"
    elif [ -f "$EXPECTED" ]; then
        rm -f "$EXPECTED"
    fi
    exit 1
fi

# ---- Record the totals for the sync commit message ----

SUMMARY="/tmp/wpt_${BACKEND}_summary.txt"
result_line=$(grep -m1 -F '[WPT] result:' "$WPT_LOG" | sed -E 's/^\[WPT\] result: //; s/;.*//')
printf '%s: %s\n' "$BACKEND" "$result_line" > "$SUMMARY"
echo "pass totals: $(cat "$SUMMARY")"

# ---- Extract failures from log ----

sed -nE "s/^[[:space:]]{4,}(${BACKEND}::[^[:space:]]+).*/\1/p" "$WPT_LOG" | LC_ALL=C sort -u > "$FAILURES"
num_failing=$(wc -l < "$FAILURES")

# ---- Rebuild (or create) expected-failures.txt ----
# $FAILURES is already sorted (LC_ALL=C) by the extraction above.

if [ -f "$EXPECTED" ]; then

    old_count=$(grep "^${BACKEND}::" "$BACKUP" 2>/dev/null | wc -l)

    grep '^#' "$BACKUP" > "$EXPECTED" 2>/dev/null || true
    cat "$FAILURES" >> "$EXPECTED"

    new_count=$(grep "^${BACKEND}::" "$EXPECTED" 2>/dev/null | wc -l)
    added_entries=$(comm -13 <(LC_ALL=C sort "$BACKUP" 2>/dev/null) "$EXPECTED" | grep "^${BACKEND}::" || true)
    removed_entries=$(comm -23 <(LC_ALL=C sort "$BACKUP" 2>/dev/null) "$EXPECTED" | grep "^${BACKEND}::" || true)
    added=$(printf '%s' "$added_entries" | grep -c '^'"${BACKEND}::" || true)
    removed=$(printf '%s' "$removed_entries" | grep -c '^'"${BACKEND}::" || true)

    echo ""
    echo "=== Rebuilding expected failures ==="
    echo "Current failures: $num_failing"
    echo "Synced: $old_count -> $new_count entries (+$added added, -$removed removed)"

    [ -n "$added_entries" ]   && { echo ""; echo "Added:";   echo "$added_entries"   | sed 's/^/  /'; }
    [ -n "$removed_entries" ] && { echo ""; echo "Removed:"; echo "$removed_entries" | sed 's/^/  /'; }
    echo ""
    echo "Backup: $BACKUP"

elif [ "$num_failing" -gt 0 ]; then

    cp "$FAILURES" "$EXPECTED"
    echo ""
    echo "=== Created expected-failures file ==="
    echo "Current failures: $num_failing"
    echo "New file: $EXPECTED"

else

    echo ""
    echo "No failures found and no expected-failures file exists."
    echo "See full output in: $WPT_LOG"

fi

rm -f "$BACKUP"
trap - EXIT
_cleanup

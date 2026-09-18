#!/bin/bash
# CANN OHOS device helpers — push and test the Rust device test binary.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -z "$CANN_DDK" ]; then
    echo "Error: CANN_DDK is not set."
    echo "  export CANN_DDK=/path/to/CANN-Kit-next/ddk/"
    exit 1
fi

DDK_LIB="${CANN_DDK}/ai_ddk_lib/lib64"

BINARY_DIR="${PROJECT_DIR}/target/aarch64-unknown-linux-ohos/release/deps"
# libtest gives the test binary a hashed name; pick the most recently built.
BINARY="$(ls -t "${BINARY_DIR}"/test_cann_execution-* 2>/dev/null | head -1)"
DEVICE_BIN="test_cann_execution"

# ── Helpers ────────────────────────────────────────────────────────────

ok()  { echo "  [OK] $*"; }
fail(){ echo "  [FAIL] $*"; exit 1; }

# ── Push to device ─────────────────────────────────────────────────────

push_to_device() {
    local target="${1:-/data/local/tmp/cann-test}"

    if ! command -v hdc &>/dev/null; then
        echo "Error: hdc not found. Install Huawei DevEco Device Tool."
        exit 1
    fi

    echo "=== Pushing to ${target} ==="
    hdc shell "mkdir -p ${target}"

    if [ -n "$BINARY" ] && [ -f "$BINARY" ]; then
        echo "  ${BINARY##*/} -> ${target}/${DEVICE_BIN}"
        hdc file send "$BINARY" "${target}/${DEVICE_BIN}"
    else
        echo "  WARN: test_cann_execution binary not found. Run 'make cann-device-test' first."
    fi

    for lib in libhiai.so libhiai_ir.so libhiai_ir_build.so libhiai_ir_build_aipp.so; do
        local src="${DDK_LIB}/${lib}"
        if [ -f "$src" ]; then
            echo "  ${lib} -> ${target}/"
            hdc file send "$src" "${target}/"
        fi
    done

    hdc shell "chmod +x ${target}/${DEVICE_BIN}"
    ok "pushed"
}

# ── Test on device ─────────────────────────────────────────────────────

test_on_device() {
    local target="${1:-/data/local/tmp/cann-test}"
    push_to_device "$target"
    echo ""
    echo "=== Running on device ==="
    hdc shell "cd ${target} && LD_LIBRARY_PATH=. ./${DEVICE_BIN} --nocapture --test-threads=1"
}

# ── WPT conformance on device ──────────────────────────────────────────

find_wpt_binary() {
    local bin
    bin=$(ls -t "${PROJECT_DIR}"/target/aarch64-unknown-linux-ohos/release/deps/run_wpt_conformance-* 2>/dev/null \
        | grep -v '\.d$' | head -n1)
    if [ -z "$bin" ] || [ ! -f "$bin" ]; then
        fail "run_wpt_conformance test binary not found; run 'make test-wpt-cann' first"
    fi
    echo "$bin"
}

wpt_on_device() {
    local target="${1:-/data/local/tmp/cann-wpt}"
    local wpt_bin
    wpt_bin=$(find_wpt_binary)

    if ! command -v hdc &>/dev/null; then
        echo "Error: hdc not found. Install Huawei DevEco Device Tool."
        exit 1
    fi
    if [ -z "$(hdc list targets 2>/dev/null | grep -v '^\[Empty\]' | grep -v '^\[Fail\]' | head -n1)" ]; then
        fail "no OHOS device connected (hdc list targets is empty)"
    fi

    echo "=== Pushing WPT to ${target} ==="
    hdc shell "mkdir -p '${target}'"

    local bin_name
    bin_name=$(basename "$wpt_bin")
    echo "  ${bin_name} -> ${target}/"
    hdc file send "$wpt_bin" "${target}/"
    hdc shell "chmod +x '${target}/${bin_name}'"

    for lib in libhiai.so libhiai_ir.so libhiai_ir_build.so libhiai_ir_build_aipp.so; do
        local src="${DDK_LIB}/${lib}"
        if [ -f "$src" ]; then
            echo "  ${lib} -> ${target}/"
            hdc file send "$src" "${target}/"
        fi
    done
    ok "pushed"

    echo ""
    echo "=== Running WPT on device (backend=cann) ==="
    # Stream the run so progress is visible; read the exit code from a marker
    local wpt_log="/tmp/wpt-cann-device.log" rc
    hdc shell "cd '${target}' && LD_LIBRARY_PATH=. WPT_BACKEND=cann WPT_REPORT_JSON=./wpt-conformance.json WPT_REPORT_HTML= ${WPT_AUDIT:+WPT_AUDIT=1 WPT_AUDIT_JSON=./wpt-audit.json} ./${bin_name} --test-threads=1; echo WPT_EXIT=\$?" | tee "${wpt_log}" || true
    rc=$(sed -n 's/^WPT_EXIT=\([0-9][0-9]*\)$/\1/p' "${wpt_log}" | tail -n1)
    rc=${rc:-1}

    echo ""
    echo "=== Retrieving report ==="
    local host_report="reports/wpt-cann-conformance.json"
    mkdir -p "$(dirname "${host_report}")"
    if hdc file recv "${target}/wpt-conformance.json" "${host_report}"; then
        ok "report -> ${host_report}"
    else
        echo "  [WARN] failed to retrieve ${target}/wpt-conformance.json"
    fi
    if [ -n "${WPT_AUDIT:-}" ]; then
        hdc file recv "${target}/wpt-audit.json" "$(dirname "${host_report}")/wpt-cann-audit.json" \
            && ok "audit -> $(dirname "${host_report}")/wpt-cann-audit.json" || true
    fi

    if [ "${rc}" -ne 0 ]; then
        echo "  [INFO] WPT trials reported failures (exit ${rc}); see ${host_report}"
    fi
    return "${rc}"
}

# ── Main ───────────────────────────────────────────────────────────────

TARGET="${1:-test}"

case "$TARGET" in
    push) push_to_device "${2:-}" ;;
    test) test_on_device "${2:-}" ;;
    wpt) wpt_on_device "${2:-}" ;;
    *)
        echo "Usage: $0 [push|test|wpt] [target-dir]"
        echo ""
        echo "  push    Transfer files to OHOS device via hdc"
        echo "  test    Push + execute test_cann_execution on device"
        echo "  wpt     Push + execute WPT conformance (backend=cann) on device"
        echo ""
        echo "  Default target-dir: /data/local/tmp/cann-test (wpt: /data/local/tmp/cann-wpt)"
        exit 1
        ;;
esac

# WPT WebNN Test Guide

This guide explains how to run the W3C Web Platform Tests (WPT) for WebNN against rustnn.

## Overview

The in-repo harness loads live WPT conformance tests from upstream `.https.any.js` files via a Node.js bridge, builds each graph through [`MLGraphBuilder`](../../src/mlgraphbuilder.rs), executes via [`MLContext`](../../src/mlcontext.rs), and validates outputs with ULP/ATOL tolerances.

See also: [wpt-harness-todos.md](wpt-harness-todos.md) for migration status.

## Prerequisites

- **Node.js** on `PATH` (for `scripts/wpt_bridge/dump_corpus.mjs`)
- **WPT corpus** at `.cache/wpt` (or set `WPT_DIR`):

```bash
node scripts/fetch_wpt.mjs
```

## Quick Start

```bash
# Full ONNX CPU conformance suite (~2482 cases)
make test-wpt

# Filter by operation or test name (libtest filter)
make test-wpt-op OP=relu
cargo test --test run_wpt_conformance --features onnx-runtime -- onnx::abs:: --test-threads 1

# TensorRT (mock) backend
make test-wpt-trtx

# Select backend at trial registration (optional)
WPT_BACKEND=onnx-gpu cargo test --test run_wpt_conformance --features onnx-runtime -- --test-threads 1
```

Trial names: `{backend}::{operation}::{test_name}` (e.g. `onnx::relu::relu_float32_2D_tensor`).

Backends: `onnx` (ORT CPU), `onnx-gpu` (ORT GPU when available, via `WPT_BACKEND`), `trtx` (TensorRT via TRTX).

## Architecture

```
rustnn/
  scripts/
    fetch_wpt.mjs                 # Download WPT checkout into .cache/wpt
    wpt_bridge/
      dump_corpus.mjs             # Evaluate WPT JS → JSON corpus (stdout)
      load-wpt-file.mjs           # VM harness for single WPT files
  tests/
    run_wpt_conformance.rs        # libtest_mimic entry point
    wpt_conformance/
      wpt_js_loader.rs            # Node bridge → WptCorpus
      wpt_types.rs                # JSON structs (WptGraph, WptOperator, …)
      wpt_tensor.rs               # Tensor packing / expected-value helpers
      wpt_execute_graph.rs        # WptGraph → MLGraphBuilder → dispatch
      wpt_backend.rs              # WPT_BACKEND / MLContext options
      mod.rs                      # Validation + run_one_test_case
      tolerance.rs                # ULP / ATOL checking
```

### Flow

1. `dump_corpus.mjs` evaluates each `webnn/conformance_tests/*.https.any.js` and captures `test.graph` from `webnn_conformance_test(...)`.
2. `run_wpt_conformance` registers one libtest trial per case (per selected backend).
3. `wpt_execute_graph::execute_wpt_graph` replays operators as `MLGraphBuilder` method calls.
4. `mod.rs` compares actual outputs to `expectedOutputs` using per-test or per-operation tolerance.

## Updating the WPT corpus

```bash
# Refresh .cache/wpt from upstream
node scripts/fetch_wpt.mjs

# Or point at an existing checkout
export WPT_DIR=/path/to/wpt
cargo test --test run_wpt_conformance --features onnx-runtime -- --test-threads 1
```

The Node bridge re-evaluates JS on each run; there is no checked-in `tests/wpt_data/*.json` snapshot anymore.

## Tolerance Checking

Tolerance logic lives in `tests/wpt_conformance/tolerance.rs`. Per-test overrides come from the WPT harness (`tolerance` field on each case). WPT metric types: **ULP** and **ATOL**.

### ULP (Units in Last Place)

ULP distance counts representable floating-point values between two numbers. Prefer ULP for float32/float16 conformance.

### Absolute Tolerance (ATOL)

Use ATOL when the WPT case specifies it, or for integer comparisons (exact match with optional integer tolerance).

## Python / pywebnn WPT

Python WPT conformance (`pytest tests/test_wpt_conformance.py`) and pywebnn live in the [pywebnn](https://github.com/rustnn/pywebnn) repository. This guide covers the **Rust** harness only.

For Python WPT conformance see [pywebnn](https://github.com/rustnn/pywebnn).

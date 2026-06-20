ll# WPT harness migration — remaining tasks

Tracks follow-up work for the in-repo WPT conformance harness (live WPT corpus → `MLGraphBuilder` → `MLContext`).

**Last updated:** 2026-06-19

## Baseline (ONNX CPU, 2026-06-19)

| Metric | Value |
|--------|-------|
| Command | `cargo test --test run_wpt_conformance --features onnx-runtime -- --test-threads 1` |
| Backend | `onnx` (ORT CPU, default `WPT_BACKEND`) |
| Corpus | Live WPT via `.cache/wpt` + `dump_corpus.mjs` |
| Trials | **2482** |
| Passed | **2482** |
| Failed | **0** |
| Wall time | **15.19 s** |
| Log | `wpt-baseline-run.log` (local, not committed) |

Prior in-repo snapshot (`tests/wpt_data/**`) was ~2480 pass / 2 fail (`abs_int64_4D_tensor`, `neg_int64_4D_tensor`). Live corpus at full pass suggests those int64 cases are fixed.

---

## Merge readiness

- [x] Run full WPT suite and record result — **2482 passed, 0 failed** (see Baseline above)
- [x] Confirm parity with deleted `tests/wpt_data/**` corpus — live corpus matches count; prior 2 int64 failures no longer reproduce
- [x] Wire CI: Node.js on PATH, `node scripts/fetch_wpt.mjs`, then `make test-wpt` (`.github/workflows/ci.yml`)
- [x] In-repo harness is canonical; CI runs `make test-wpt` (`.github/workflows/ci.yml`, `AGENTS.md`)

## Documentation

- [x] Rewrite `docs/testing/wpt-test-guide.md` for Rust harness + Node bridge
- [x] Document prerequisites: Node.js, `WPT_DIR` / `.cache/wpt`, `node scripts/fetch_wpt.mjs`
- [x] Document backend selection: `WPT_BACKEND=onnx|onnx-gpu|trtx` and libtest filters
- [x] Update `Makefile` help text for WPT targets
- [x] Note `make test-wpt` / `make test-wpt-op OP=<filter>` as canonical commands

## Harness correctness

- [ ] Run TRTX path via MLContext: `WPT_BACKEND=trtx` with `trtx-runtime-mock` (or real TRTX) and fix failures
- [ ] Add at least one pinned TRTX smoke trial or snapshot after removing `run_all_trtx.snap`

## Performance

- [ ] Reuse `MLContext` per backend per test thread (avoid `MLContext::create` on every case (~2482 trials)
- [ ] Measure before/after runtime for full suite with `--test-threads 1`

## Code quality / maintainability

- [x] Rename `wpt_to_graph.rs` → `wpt_tensor.rs`
- [x] Consolidate tensor packing into `wpt_tensor` (`tensor_*_values` + `expected_output_to_*`)
- [x] Reduce binary-op duplication via `invoke_binary_with_options`
- [x] Add `invoke_unary_simple`, `invoke_unary_reduce`, `invoke_pool2d` helpers in `wpt_execute_graph.rs`
- [x] Remove dead code: `discover_operations`, `run_all`, `WptActualOutput::is_integer` / `is_float`
- [x] Fix compiler warnings in harness (`wpt_js_loader` test import, unused imports)
- [x] Expand `WptBackend` with `OnnxGpu` (`WPT_BACKEND=onnx-gpu`); default trials still register CPU only

## Naming / UX

- [ ] Revisit trial prefix `onnx::` vs `webnn::` or `ort-cpu::` to avoid “harness = ONNX converter” confusion
- [x] Optional: `make test-wpt-trtx` target for TRTX feature set

## Library fixes surfaced by WPT (if still open)

- [x] Verify all int64 / uint64 unary and binary ops end-to-end — covered by full 2482-case pass (includes int64 abs/neg)
- [x] Confirm `split_equal_with_options` coverage for equal-split WPT cases — covered by full suite pass
- [x] Confirm `lstm` / `lstmCell` `OperationExtras` handling — covered by full suite pass

## Cleanup

- [x] Remove references to deleted scripts (`convert_wpt_tests.py`, `update_wpt_tests.sh`, `extract_wpt_tests.js`) from docs
- [x] Update `AGENTS.md`, `implementation-status.md`, `burn-plan.md`, `operator-enum-refactor-execution.md`

---

## Done (for context)

- [x] Wire `wpt_execute_graph` into `mod.rs` / `run_wpt_conformance.rs`
- [x] Remove legacy TRTX `GraphInfo` + `run_trtx_with_inputs` path from harness
- [x] Remove `wpt_graph_to_onnx_inputs` and ONNX `TensorData` from execution path
- [x] Add `WptBackend` + `WPT_BACKEND` env var
- [x] Remove ONNX panic-string skip logic; `MLContext::create` errors fail the trial
- [x] Remove unused `WptGraph.constants`
- [x] Node bridge: `scripts/wpt_bridge/dump_corpus.mjs` + `wpt_js_loader.rs`

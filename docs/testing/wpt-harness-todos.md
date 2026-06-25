# WPT harness — evaluation backlog

In-repo harness: live WPT corpus → Node bridge → `MLGraphBuilder` → `MLContext`.

**Baseline (ONNX CPU, 2026-06-24):** 2482 passed, 0 skipped, 0 failed (`make test-wpt`, `--test-threads 1`). Prior baseline: 2482 ok (skips were counted as passes before 2026-06-24).

Use the checkboxes below to track **evaluation** (measure, decide, fix, or accept). Mark done when resolved or explicitly deferred with a one-line note.

**Suggested order:** reporting (P1) → TRTX (P1) → WPT pin (P1) → context reuse (P2) → trial prefix (P2) → structural split (P3).

---

## P1 — High priority (evaluate first)

### Reporting & metrics

- [x] **Skips vs passes** — Dtype skips use `Trial::ignorable_test` + `Completion::ignored_with`; libtest reports `N ignored` separately from `M passed`.
- [ ] **Skip inventory** — Run suite; list which dtypes/ops are skipped (e.g. bool). *Current:* 0 skipped on full conformance corpus (2026-06-24).
- [x] **End-of-run summary** — `[WPT] registering …` (expected skip count) and `[WPT] result: passed / skipped / failed` after libtest summary.
- [ ] **Skip reason histogram** — Optional: aggregate skip reasons (e.g. `unsupported dataType: bool`) at end of run.

### Corpus & CI integrity

- [x] **Parse failures** — `corpus.file_errors` stays warning-only; CI does **not** fail on parse errors (accepted).
- [ ] **WPT pin strategy** — Compare `origin/master` vs pinned SHA for stability. *Evaluate:* record SHA in corpus JSON / CI env; update `fetch_wpt.mjs` or CI to use it.
- [ ] **Upstream drift risk** — Document what breaks if WPT updates tomorrow without a rustnn change; decide pin vs periodic bump workflow.

### Backend coverage

- [ ] **TRTX via MLContext** — Run `make test-wpt-trtx` or `WPT_BACKEND=trtx`; record pass/fail/skip baseline (mock and/or real TRTX).
- [ ] **TRTX in CI** — *Evaluate:* add optional job or nightly vs blocking PR gate; document TF32 vs ULP expectations (`trtx.rs`).
- [ ] **TRTX smoke artifact** — Replace removed `run_all_trtx.snap` with pinned smoke trial(s) or minimal snapshot.

---

## P2 — Medium priority

### Performance

- [x] **Per-trial `MLContext::create`** — `wpt_config::REUSE_ML_CONTEXT` defaults to **`false`**; optional thread-local reuse in `wpt_context_pool`.
- [x] **Measure reuse impact** — ONNX CPU, debug, `--test-threads 1`, 2482 trials (2026-06-24, Windows):

  | `REUSE_ML_CONTEXT` | libtest `finished in` |
  |--------------------|------------------------|
  | `true` (reuse)     | 16.63 s, 25.67 s       |
  | `false` (per trial)| 15.33 s, 18.23 s       |

  **No measurable win** — variance (~±5 s) dominates; reuse is not faster on ORT CPU today. Likely cause: `OrtContext` appends every trial’s tensors to `self.tensors` and never clears. **Default is `false`.** Revisit only with per-trial tensor cleanup and/or TRTX profiling.
- [ ] **Trial registration clones** — Profile memory / clone cost of `WptLoadedCase` in `Trial::test` closures. *Evaluate:* share corpus via `Arc` or register by index.

### Naming & backends

- [ ] **Trial prefix `onnx::`** — Survey filter strings in docs/CI/scripts. *Evaluate:* rename to `webnn::` or `ort-cpu::`; document breaking change for `make test-wpt-op OP=...`.
- [ ] **`OnnxGpu` default trials** — Confirm GPU path is never registered without `WPT_BACKEND=onnx-gpu`. *Evaluate:* document, add CI matrix entry, or register both CPU+GPU trials.

### Coverage gaps

- [ ] **Validation tests** — Compare WPT `validation_tests/` vs old `tests/wpt_data/validation/`. *Evaluate:* scope, load via bridge, error-expectation harness vs conformance-only.
- [ ] **Conformance file count** — Compare loaded case count vs WPT `conformance_tests/*.https.any.js` file count; account for `file_errors` and skipped dtypes.

### Maintainability

- [ ] **`wpt_execute_graph.rs` monolith** (~1777 lines) — *Evaluate:* split dispatch / `build_method_args` / I/O; estimate effort per new op today.
- [ ] **`invoke_builder_method` match** — List ops only reachable via manual arm; check parity with pywebnn `wpt_execute_graph.py`.

---

## P3 — Lower priority / polish

### Code structure

- [ ] **`mod.rs` format helpers** (~200 lines `format_*_nd`) — *Evaluate:* move to `wpt_failure_format.rs` or share with `wpt_tensor`.
- [ ] **Unit test coverage** — Inventory fast tests without Node (`sanitize_test_id` only today). *Evaluate:* add tests for `build_method_args`, packing, tolerance edge cases.
- [ ] **Bridge cache keys** — *Evaluate:* whether CI `.cache/wpt` key should include `scripts/wpt_bridge/*.mjs` hashes or only `fetch_wpt.mjs`.

### Runner behavior

- [ ] **Parallelism / thread safety** — Run without `--test-threads 1`. *Evaluate:* `MLContext` thread safety; enforce single-thread in binary or document risk.
- [ ] **Auto-fetch (`ensure_wpt_cache`)** — Offline / no-git scenario. *Evaluate:* fail fast with actionable message vs silent `fetch_wpt.mjs` spawn.
- [ ] **Large constant inline** (8 MiB scalar-fill threshold) — *Evaluate:* document in `wpt-test-guide.md`; confirm WPT cases hitting runtime-input path.

### Strengths (confirm still true after changes)

- [ ] WebNN API path (`MLGraphBuilder` + `MLContext::dispatch`) — not ONNX-converter shortcut
- [ ] Live corpus via Node bridge — no checked-in JSON snapshot
- [ ] Tolerance (ULP/ATOL) + failure output (inputs, ND dumps) — adequate for triage
- [ ] `WptBackend` + `WPT_BACKEND` + dispatch helpers — sufficient for multi-backend trials

---

## Reference — what works well (2026-06-19)

- Correct abstraction: WebNN builder/context, backend-agnostic graph replay
- Live WPT corpus; 2482 conformance cases at full libtest pass on ONNX CPU
- CI: `node scripts/fetch_wpt.mjs` + `make test-wpt`
- Tolerance aligned with pywebnn/WPT; binary/unary/pool helpers reduced match duplication

## Reference — concern summary

| ID | Concern | Section |
|----|---------|---------|
| R1 | Skips counted as passes | P1 Reporting — **fixed** (`ignorable_test`) |
| R2 | `file_errors` warning only | P1 Corpus — **accepted** (CI does not fail) |
| R6 | Per-trial context creation | P2 Performance — reuse opt-in (`false` default); no ONNX CPU win |
| R3 | Unpinned `origin/master` WPT | P1 Corpus |
| R4 | TRTX not in CI | P1 Backend |
| R5 | Misleading `onnx::` prefix | P2 Naming |
| R7 | `wpt_execute_graph.rs` size | P2 Maintainability |
| R8 | No validation test suite | P2 Coverage |
| R9 | `OnnxGpu` off by default | P2 Naming |
| R10 | Thin unit tests | P3 Structure |
| R11 | `mod.rs` format bloat | P3 Structure |
| R12 | Parallelism undocumented | P3 Runner |
| R13 | Auto-fetch surprises | P3 Runner |

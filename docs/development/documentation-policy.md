# Documentation Policy

This page defines what "documented" means for a change to rustnn and what a contributor or a
coding agent must update together with the code. The documentation checklist in the pull
request template points here.

## Where documentation lives

| Location | Content | Built by | Published at |
|---|---|---|---|
| `//!` and `///` comments in `src/` | Rust API reference: every public item (the crate has `#![warn(missing_docs)]`, and CI denies warnings), the crate overview with the feature and environment variable tables in `src/lib.rs` | `make docs-api` (rustdoc, warnings are errors) | https://rustnn.github.io/rustnn/api/rustnn/ |
| `docs/**/*.md`, `mkdocs.yml` | User guide, architecture, development, testing and integration pages | `make docs-build`; strict mode in CI with `make ci-docs` | https://rustnn.github.io/rustnn/ |
| `docs/development/backend-operator-support.md` | Operation-by-backend matrix generated from the converter sources | `make docs-backend-ops`; CI fails on drift | same site |
| WPT conformance dashboard | Per-operation pass and fail status per backend | nightly workflow | https://rustnn.github.io/rustnn/wpt-conformance/ |
| `README.md` | Front page on GitHub and crates.io | - | - |
| `AGENTS.md` (imported by `CLAUDE.md`) | Working guide for coding agents: architecture summary, conventions, workflows | - | - |
| `docs/archive/` | Superseded plans and investigation notes, excluded from the site, read-only | - | - |

## The rule

A change is complete when the documentation that describes the changed behaviour is updated in
the same pull request. Documentation follows the code; there is no later cleanup pass.

| If the change touches | Update |
|---|---|
| A public type, method, trait or module | The rustdoc comment of the item (mandatory: `cargo check` warns on undocumented public items and CI fails); the module docs (`//!`) when the module's responsibility changes. Builder operations pass their doc comment into the `impl_*_op!` macro invocation; option fields state meaning, unit and default |
| A Cargo feature or an environment variable | The tables in `src/lib.rs` and in `docs/user-guide/backends.md`; the backend's page under `docs/integration/` |
| Backend selection, a backend's requirements or its execution model | The module docs of `src/backend_selection.rs`, `docs/user-guide/backends.md` and the backend's page under `docs/integration/` |
| Operator support in a converter (`src/converters/*.rs`) | Run `make docs-backend-ops`, then refresh the WPT snapshots or expected-failure list of that backend (`make wpt-sync-<backend>`) |
| A lowering rule or a backend constraint in a converter | The backend section of `docs/development/converters.md` |
| An error variant or a new failure mode | The rustdoc of the variant and, when users will meet it, a row in `docs/user-guide/troubleshooting.md` |
| The `.webnn`, JSON or weight file handling (`src/loader.rs`, `src/webnn_json.rs`, `src/webnn_save.rs`) | `docs/reference/graph-files.md` |
| A new operation | The "Adding an operation" checklist in `docs/development/setup.md`, including the operation table in `docs/user-guide/api-reference.md` |
| The builder or context API (`src/mlgraphbuilder.rs`, `src/mlcontext.rs`) | `docs/user-guide/api-reference.md`; the code in `docs/user-guide/getting-started.md` and `docs/user-guide/examples.md` when it uses the changed call |
| Makefile targets, scripts or CI workflows | `docs/development/setup.md` and `.github/workflows/README.md` |
| The WPT harness (`tests/run_wpt_conformance.rs`, `tests/wpt_conformance/`) | `docs/testing/wpt-test-guide.md` |
| An example program | `docs/user-guide/examples.md` |
| Removing or renaming an API | Search for the old name in `docs/`, `README.md`, `AGENTS.md` and `src/` and update every hit |

## Rules for coding agents

1. The source is the reference. Read the current code before relying on a documentation page.
   When a page contradicts the code, fix the page in the same change.
2. Do not reintroduce removed APIs. Old pull requests, archived pages and documents written
   before 2026 describe code that no longer exists. The table below maps the names that keep
   resurfacing to the current API.
3. Do not edit `docs/archive/`. Pages there are frozen. If some content is still useful,
   rewrite it as a current page.
4. Do not write counts, percentages or dates that drift (operation totals, pass rates,
   coverage figures). Link to the generated report or the dashboard instead.
5. Plain text: no emojis, ASCII in code comments and error strings, absolute dates.
6. Before handing over a change that touched rustdoc, pages or converters, run
   `make docs-api`, `make ci-docs` and `make docs-backend-ops-check`.

### Removed or replaced names

| Seen in old documents or pull requests | Current state |
|---|---|
| `src/python/`, `PyMLContext`, `maturin develop --features python`, `tests/test_python_api.py` | Removed in January 2026 (PR #11). Python bindings live in the separate [pywebnn](https://github.com/rustnn/pywebnn) repository; rustnn contains no Python API. |
| `context.compute(graph, inputs)`, `compute_onnx()`, `compute_coreml()`, `compute_fallback()` | Replaced by `MLContext::dispatch` with `MLTensor` bindings. `compute()` is also gone from the WebNN specification. |
| `PyMLContext::select_backend()` | `select_backend` in `src/backend_selection.rs`, driven by `MLContextOptions` |
| `Operation { op_type: String, attributes: serde_json::Value, .. }` | `rustnn::operators::Operation` is an enum with one variant per operation and typed `ML*Options` structs from `src/operator_options.rs` |
| `HashMap<String, MLOperand>` and `HashMap<&str, &MLTensor>` arguments | `MLNamedOperands` and `MLNamedTensors` (`BTreeMap` aliases) |
| `run_onnx_with_inputs`, `run_coreml_zeroed`, `run_trtx_with_inputs` as the execution API | Legacy one-shot executors kept for the CLI and two older examples. New code uses `MLContext`. |
| `ensure_trtx_loaded` | Removed; TensorRT-RTX is loaded on demand. `dynamically_load_tensorrt` exists for custom library paths. |
| `MLGraphBuilder::conv2_with_options`, `MLTensor::destoy` | Renamed to `conv2d_with_options` (the old name is a deprecated alias) and `destroy` |
| Operations `round`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh` | Removed with the specification. JSON graphs with `round` are accepted as `roundEven`. |
| "88 of 105 operations", "84% coverage", "1350 ONNX tests passing" | Stale figures. Current coverage: `docs/development/backend-operator-support.md` and the WPT dashboard |
| `make python-dev`, `make python-test`, `make text-gen-demo`, `make mobilenet-demo`, `make minilm-demo-hub` | Targets no longer exist; `make help` lists the current ones |
| `docs/development/contributing.md`, `docs/api-reference.md`, `docs/development.md`, `docs/webnn-spec-reference.md` | Do not exist; the current pages are the ones listed in `mkdocs.yml` |
| `tarekziade/rustnn` | The repository is `rustnn/rustnn` |
| GGML backend | Never implemented; the plan is archived |

## Writing guidelines

- Lead with what the reader has to do. One idea per paragraph.
- Commands and code go in fenced blocks; prefer `make` targets where they exist.
- Link to rustdoc for signatures instead of copying them into pages.
- Name the exact file for anything the reader has to open.
- Archive a page (move it to `docs/archive/` and prepend the banner used there) when it
  describes a design that was not adopted or a state that no longer exists. Keep history out
  of the navigation, but do not delete it.

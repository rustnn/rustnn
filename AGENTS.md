# rustnn - Project Guide for Contributors and Coding Agents

rustnn is a Rust implementation of the W3C WebNN API: a WebNN-style graph builder, shape
inference and validation, and execution on pluggable backends (ONNX Runtime, NVIDIA
TensorRT-RTX, Apple CoreML, LiteRT, Huawei CANN). Python bindings live in the separate
[pywebnn](https://github.com/rustnn/pywebnn) repository; this repository contains no Python API.

Documentation map:

| Topic | Location |
|---|---|
| Site entry point | `docs/index.md` (built with MkDocs, `make docs-build`) |
| API overview, backends, examples, advanced topics, troubleshooting | `docs/user-guide/` |
| Architecture | `docs/architecture/overview.md` |
| Workflow, Make targets, adding operations and backends, CI | `docs/development/setup.md` |
| Converter contract, per-backend lowering rules, debugging emitted models | `docs/development/converters.md` |
| What to update together with a code change | `docs/development/documentation-policy.md` |
| Implementation status and known gaps | `docs/development/implementation-status.md` |
| Operator support per backend (generated) | `docs/development/backend-operator-support.md` |
| WPT conformance harness | `docs/testing/wpt-test-guide.md` |
| Backend pages: TensorRT-RTX, CoreML, LiteRT, CANN, browser WebNN | `docs/integration/` |
| Specification-to-API mapping; `.webnn`, JSON and weight file formats | `docs/reference/webnn-spec.md`, `docs/reference/graph-files.md` |
| Rust API reference | `make docs-api` (rustdoc), published under https://rustnn.github.io/rustnn/api/rustnn/ |

## Read this first

1. The source is the reference. Documentation pages describe the state at their last commit;
   when a page and the code disagree, the code wins and the page is fixed in the same change.
2. Removed APIs must not come back. `src/python/`, `PyMLContext`, `context.compute(...)`,
   `maturin`, the string-typed `Operation` with JSON attributes and `HashMap` argument maps are
   gone. `docs/development/documentation-policy.md` maps the old names to the current API.
3. Every change updates the documentation it affects, in the same pull request. The policy page
   says which page belongs to which part of the code.
4. `docs/archive/` is frozen history. Do not cite it as current and do not edit it.
5. Numbers drift. Do not write operation counts or pass rates into pages; link to the generated
   report or the WPT dashboard.

## Architecture in brief

```
MLContext::create(&MLContextOptions)     -> selects Backend / BackendDevice (src/backend_selection.rs)
MLGraphBuilder::new(&mut context)        -> records GraphInfo (operands, Operation enum, constants);
                                            shape inference runs on every builder call
builder.build(&MLNamedOperands)          -> converter (src/converters/) + backend compile -> MLGraph
context.create_tensor / write_tensor     -> MLTensor owned by the backend (readable / writable flags)
context.dispatch(&mut graph, &inputs, &outputs) -> binding validation, then the backend runs
context.read_tensor                      -> results
```

Key modules (`docs/architecture/overview.md` has the full table):

| Path | Content |
|---|---|
| `src/mlcontext.rs`, `src/mlcontextoptions.rs`, `src/backend_selection.rs` | WebNN context, graph and tensor types; options and hints; the selection order |
| `src/mlgraphbuilder.rs` | Builder: inputs, constants, all operation methods (macro-generated `op` and `op_with_options`), `build`, `rustnn_save_webnn` |
| `src/operators.rs`, `src/operator_options.rs`, `src/operator_enums.rs` | `Operation` enum (one variant per operation, `op_type()`), `ML*Options` structs, spec enums |
| `src/shape_inference.rs`, `src/validator.rs`, `src/runtime_checks.rs` | Shape rules, structural validation, dispatch-time binding checks |
| `src/graph.rs` | `GraphInfo`, `Operand`, `Dimension` (static or dynamic), `DataType`, 4-bit packing |
| `src/loader.rs`, `src/webnn_json.rs`, `src/webnn_save.rs` | `.webnn` text and JSON import and export through the `webnn-graph` crate, `.safetensors` weights |
| `src/converters/{onnx,coreml_mlprogram,trtx,litert,cann,webnn}.rs` | Lowering of `GraphInfo` to each backend format; `ConverterRegistry` |
| `src/backends/{ort,trtx,coreml,litert,cann}.rs`, `src/backends/caching.rs` | Backend contexts implementing the crate-private `MLBackendContext` and `MLBackendBuilder` traits; on-disk caches |
| `src/executors/` | Legacy one-shot executors used by the CLI (`--run-onnx`, `--run-trtx`, `--run-coreml`) |
| `src/main.rs` | CLI: validate, `--export-dot`, `--convert`, `--run-*` |
| `tests/run_wpt_conformance.rs`, `tests/wpt_conformance/` | WPT conformance harness (`make test-wpt*`) |
| `scripts/generate_backend_operator_report.py` | Generates `docs/development/backend-operator-support.md`; CI fails on drift |

Backend selection from the WebNN hints: accelerated with Default or HighPerformance ->
TensorRT-RTX, CoreML GPU, LiteRT GPU, ONNX Runtime GPU then NPU; accelerated with LowPower ->
CoreML NPU, LiteRT NPU, ONNX Runtime NPU; not accelerated -> CoreML CPU, LiteRT CPU, ONNX
Runtime CPU. ONNX Runtime CPU is the last resort. CANN is only selected through
`with_rustnn_backend_hint(Backend::Cann)`.

## Conventions

- Names: `snake_case` files and functions, `PascalCase` types. WebNN `camelCase` methods become
  `snake_case` (`reduce_sum`, `where_`). rustnn-specific API carries the `rustnn_` prefix.
- Errors: typed with `thiserror`, `Send + Sync`, with context. `rustnn::error::Error` for the
  WebNN API, `GraphBuilderError` and `ShapeInferenceError` for recording, `GraphError` for the
  legacy pipeline.
- Comments: one line for non-obvious decisions only, ASCII only (`->`, `<=`). No emojis anywhere
  in the repository (code, documentation, commit messages); use plain markers such as `[OK]` or
  `[WARNING]` when a marker is needed.
- Formatting and lints: `cargo fmt` and `cargo clippy --all-targets -- -D warnings`. CI denies
  warnings, including rustdoc warnings and `missing_docs`: every public item needs a `///`
  comment (builder operations pass it into the `impl_*_op!` invocation).
- Tests: unit tests in `#[cfg(test)]` modules at the end of the file; converter tests decode the
  emitted model and assert on it; the WPT corpus is the conformance oracle.
- Serde: option structs use `#[serde(rename_all = "camelCase")]` to match WebNN JSON.
- No code copied from other projects (ONNX Runtime, Chromium, TensorFlow). Re-derive from the
  specifications and cite them.

## Development commands

Use the Make targets; they set features and environment variables. `make help` lists them.

```bash
make build                      # cargo build
make test                       # fmt, clippy -D warnings, cargo test, operator report drift check
make test-wpt                   # WPT conformance on ONNX Runtime CPU (needs Node.js; fetches the corpus)
make test-wpt-op OP=relu        # one operation; WPT_BACKEND=onnx|trtx|litert|coreml selects a backend
make test-wpt-trtx              # also test-wpt-litert, test-wpt-coreml
make wpt-sync-onnx              # also wpt-sync-litert, wpt-sync-coreml, wpt-sync-trtx: regenerate baselines
make docs-backend-ops           # regenerate the operator support report
make docs-api                   # rustdoc with -D warnings
make ci-docs                    # MkDocs strict build
make onnxruntime-download       # ONNX Runtime for the onnx-runtime feature; then export ORT_DYLIB_PATH
```

Feature checks CI runs: `cargo check`, `cargo check --features onnx-runtime`,
`cargo check -F trtx-runtime --all-targets`, `cargo check --features litert-runtime`,
`cargo check --features cann-runtime`, `cargo check --features coreml-runtime` (macOS), wasm32
with `webnn-runtime`; `cargo test --lib` and `cargo test --lib --features cann-runtime-mock`.

Environment variables: `ORT_DYLIB_PATH` (ONNX Runtime library), `RUST_LOG`, `RUSTNN_DEBUG=1|2`
with `RUSTNN_DEBUG_ONNX_DIR`, `RUSTNN_TRTX_LOG_VERBOSITY`, `TRTX_JSON_DUMP_PATH`, `WPT_BACKEND`,
`WPT_DIR`. The tables in `src/lib.rs` are authoritative.

## Before proposing a commit

1. `make test` passes (formatting, clippy, tests, report drift).
2. WPT for the touched operations on every backend available locally; snapshots or expected
   failures regenerated and reviewed (`make wpt-sync-<backend>`).
3. `make docs-backend-ops` when a converter's operator support changed.
4. Documentation updated per `docs/development/documentation-policy.md`; `make docs-api` and
   `make ci-docs` pass when rustdoc or pages changed.
5. Line endings: files are LF in the index; do not introduce CRLF.
6. Commit messages: conventional prefix (`feat(trtx):`, `fix(coreml):`, `docs:`), imperative
   subject, body with the problem, the change and the validation performed.

## Adding a WebNN operation

Check the Chromium reference implementation first
(https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/, in particular
`ort/graph_builder_ort.cc`, `coreml/graph_builder_coreml.cc` and
`tflite/graph_builder_tflite.cc`) for the expected lowering and edge cases. Then follow the
checklist in `docs/development/setup.md`: options struct, `Operation` variant and `op_type`,
shape inference with tests, builder method, `webnn_json` mapping, each converter (or an explicit
unsupported entry), WPT per backend with snapshot sync, `make docs-backend-ops`, and the
operation table in `docs/user-guide/api-reference.md`.

## WebNN specification reference

`docs/reference/webnn-index.bs` is a cached copy of the specification source (date in
`docs/reference/README.md`). The `search-bikeshed` tool indexes the live specification:

**Note:** Python bindings are now in [pywebnn](https://github.com/rustnn/pywebnn), which uses rustnn as its core library.

### Key Architectural Principles

**1. Backend-Agnostic Graph Representation (WebNN Spec-Compliant)**
- `builder.build()` creates an immutable `GraphInfo` structure
- Graph representation is **platform-independent** and **backend-agnostic**
- No backend-specific artifacts at graph build time
- Same graph can be executed on multiple backends

**2. Runtime Backend Selection (WebNN Device Selection Explainer)**
- Follows [W3C WebNN Device Selection Explainer](https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md)
- Backend selection happens at **context creation** using hints, not compile-time
- `MLContext::new()` takes `accelerated` (bool) and `power_preference` (str) hints:
  - `accelerated=false` → `Backend::OnnxCpu` (CPU only)
  - `accelerated=true` + `power="low-power"` → NPU > GPU > CPU
  - `accelerated=true` + `power="high-performance"` → TensorRT > GPU > NPU > CPU
  - `accelerated=true` + `power="default"` → TensorRT > GPU > NPU > CPU
- Platform autonomously selects actual device based on availability
- Selection logic in `PyMLContext::select_backend()` (src/python/context.rs:473)
- Feature flags control availability, not selection
- Per explainer: "implementations have a better grasp of the system...control should be relinquished to them"

**3. Lazy Backend Conversion**
- Backend conversion happens during **`compute()`**, not `build()`
- `compute()` method routes to backend-specific execution:
  - `compute_trtx()` → Converts to ONNX protobuf, executes with TensorRT
  - `compute_onnx()` → Converts to ONNX protobuf, executes with ONNX Runtime
  - `compute_coreml()` → Converts to CoreML protobuf, executes with CoreML
  - `compute_fallback()` → Returns zeros when no backend available
- Conversion is transparent to the user
- **CANN exception:** CANN compiles and loads the model at `build()`

**4. Rust-First Architecture**
- All core logic implemented in pure Rust
- Python bindings are thin PyO3 wrappers
- Zero Python code in critical path (validation, conversion, execution)
- Rust library usable independently without Python

### Key Modules

#### **graph.rs** - Core Data Model
- `DataType`: Float32, Float16, Int32, Uint32, Int8, Uint8
- `OperandDescriptor`: Shape and type information
- `OperandKind`: Input, Constant, Output
- `Operand`: Graph nodes with descriptors and metadata
- `Operation`: Graph operations with inputs/outputs
- `ConstantData`: Weight/constant storage (base64 encoded)
- `GraphInfo`: Complete graph representation

**Key Convention:** Operands are referenced by their array index (u32) within the graph's operands list.

#### **validator.rs** - Validation Pipeline
- `ContextProperties`: Validation constraints and limits
- `GraphValidator`: Validates graph structure and dependencies
- `ValidationArtifacts`: Results including I/O descriptors and operation dependencies

**Validation Checks:**
1. Operand count limits
2. Tensor byte length limits
3. Valid input/output names
4. Constant data integrity
5. Operation dependency ordering
6. Operand usage consistency

#### **converters/** - Pluggable Format Conversion
- **Registry Pattern**: `ConverterRegistry` manages converters dynamically
- **Trait Interface**: `GraphConverter` defines conversion contract
- **Implementations**:
  - `OnnxConverter` → ONNX protobuf format
  - `CoremlMlProgramConverter` → CoreML MLProgram (MIL) protobuf format

#### **executors/** - Runtime Execution
- **Platform-specific**: Conditional compilation for macOS
- **TensorRT Runtime**: `run_trtx_with_inputs()` - NVIDIA GPU execution (Linux/Windows, with mock mode for development)
- **ONNX Runtime**: `run_onnx_with_inputs()` - executes with actual tensor I/O (cross-platform)
- **CoreML Runtime**: `run_coreml_zeroed_cached()` - macOS only via Objective-C FFI

#### **Backend Selection**
- Backend selection follows [W3C WebNN Device Selection Explainer](https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md)
- Selection based on `accelerated` (bool) and `power_preference` (str) hints
- Platform autonomously selects optimal backend based on availability
- Supports: TensorRT (NVIDIA GPU), ONNX Runtime (CPU/GPU), CoreML (macOS)

**For Python API documentation**, see [pywebnn](https://github.com/rustnn/pywebnn)

#### **graphviz.rs** - Visualization
- Generates DOT format for graph visualization
- Color-coded nodes: inputs (green), outputs (blue), constants (yellow)

## Development Conventions

### Design Principles

**Rust-First WebNN Implementation:**
- The Rust code is a fully valid, standalone WebNN implementation
- All core functionality, validation, and graph operations exist in pure Rust
- The Rust library is independently usable without any Python dependency
- Python bindings are a convenience layer to enable easy integration with Python projects
- Python code should be minimal wrappers that expose Rust functionality
- This ensures the library can be used in pure Rust projects, CLI tools, and Python projects alike

### Code Style

1. **Naming:**
   - Files: `snake_case.rs`
   - Types: `PascalCase`
   - Functions: `snake_case`
   - Enums: PascalCase variants, snake_case JSON serialization

2. **Error Handling:**
   - All fallible operations return `Result<T, GraphError>`
   - Use `?` operator for error propagation
   - `thiserror` for error type derivation
   - Include contextual information in errors

3. **Serde Integration:**
   - `#[derive(Serialize, Deserialize)]` on all data types
   - `#[serde(rename_all = "snake_case")]` for JSON compatibility
   - `serde_with` for base64 encoding of binary data
   - Optional fields use `Option<T>`

4. **Testing:**
   - Unit tests in `#[cfg(test)]` modules at end of files
   - Use realistic data structures matching actual usage
   - Test examples exist in `graphviz.rs` and `converters/mod.rs`

5. **Formatting:**
   - No emojis in code, documentation, commit messages, or any project files
   - Use plain text markers: [OK], [WARNING], [INFO], [TODO], etc.
   - Keep all text professional and readable in all terminals and editors
   - Prioritize clarity and accessibility over visual decoration

### Architecture Patterns

1. **Registry Pattern** (converters):
   - Trait objects: `Box<dyn GraphConverter + Send + Sync>`
   - Dynamic registration and lookup
   - Extensible without modifying core code

2. **Builder Pattern** (protobuf construction):
   - Incremental construction of complex structures
   - Used in ONNX and CoreML converters

3. **Validation Pipeline**:
   - Immutable graph input
   - Stateful validator with progressive checks
   - Comprehensive artifacts returned for downstream use

4. **Conditional Compilation**:
   - `#[cfg(target_os = "macos")]` for platform-specific code
   - `#[cfg(feature = "...")]` for optional features
   - Graceful degradation on unsupported platforms

5. **Explicit Dependencies**:
   - No singletons or global state
   - Pass dependencies via function parameters
   - Clear data flow through the system

### WebNN Specification Reference

The project uses the `search-bikeshed` tool for efficient browsing and searching of the W3C WebNN specification:

**Installation:**
```bash
pip install search-bikeshed
search-bs index https://github.com/webmachinelearning/webnn/blob/main/index.bs --name webnn
search-bs search --name webnn "MLTensor" --around 3
search-bs get --name webnn --line 1234 --count 40
```

Use it to verify operation signatures, option dictionaries and data type constraints.

## Claude Code - Approved Permissions

The following operations have been approved for Claude Code to execute without requiring additional user confirmation:

### Build & Development
- `cargo *` - All Cargo commands (check, build, fmt, test, clean, clippy, doc, etc.)
- `cd "c:\git\rustnn-workspace\rustnn" && cargo test --features trtx-runtime --test run_wpt_conformance -- --test-threads 1 --nocapture *` - Run WPT conformance tests with TensorRT backend; append test name filter (e.g. `trtx::not_equal`)
- `pip *` - All pip commands (install, uninstall, list, freeze, etc.)
- `maturin *` - All maturin commands (develop, build, publish, etc.)
- `make *` - All Makefile targets approved

### Python Execution & Testing
- `python` - Run Python scripts
- `python3.12` - Run Python 3.12 interpreter specifically
- `.venv/bin/python` - Run Python from virtual environment
- `.venv-test/bin/python` - Run Python from test virtual environment
- `python3.12 -m venv` - Create Python virtual environments
- `source` - Activate virtual environments (e.g., `source .venv-test/bin/activate`)
- `pytest` - Run Python test suite

### Documentation
- `mkdocs build` - Build documentation site

### Specification Tools
- `search-bs *` - All search-bikeshed commands (index, search, get)
  - Used for browsing and searching W3C WebNN specification
  - Provides fast local access to spec content without web browsing

### File Operations
- `find` - Search for files
- `cat` - Read file contents

### Git Operations
- `git *` - All git commands approved (status, add, commit, push, pull, checkout, branch, tag, log, diff, show, reset, rebase, merge, etc.)
  - Note: Destructive operations (force push to main, hard reset) should still be used cautiously
- `cd "c:\git\rustnn-workspace\rustnn" && git diff *` - View diffs for any file in the rustnn repo (read-only)

### GitHub Operations
- `gh run list` - List GitHub Actions workflow runs
- `gh run view` - View details of GitHub Actions runs

### Web Resources
- `WebFetch(domain:www.w3.org)` - Fetch W3C WebNN specifications

These permissions enable Claude Code to efficiently assist with development, testing, documentation, version control, and CI/CD monitoring tasks without interrupting the workflow.

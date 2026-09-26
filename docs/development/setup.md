# Setup and Workflow

## Toolchain

| Tool | Notes |
|---|---|
| Rust | `rust-toolchain.toml` pins the channel (1.97.0 at the time of writing) with `rustfmt` and `clippy`; rustup installs it on first use. The CI workflows pin the same version, so bump them together |
| `protoc` | Required: `build.rs` compiles the ONNX and CoreML protobuf schemas. Linux `apt-get install protobuf-compiler`, macOS `brew install protobuf`, Windows `winget install Google.Protobuf` |
| `flatc` | Required for `litert-runtime` (TFLite schema). CI downloads the flatbuffers release binary; macOS `brew install flatbuffers` |
| Node.js | WPT corpus fetch and evaluation (`scripts/fetch_wpt.mjs`, `scripts/wpt_bridge/`) |
| Python 3 | Only for MkDocs (`pip install -r docs/requirements.txt`) and `scripts/generate_backend_operator_report.py` |
| libclang | `trtx-runtime`: the `trtx-sys` crate generates bindings with autocxx; set `LIBCLANG_PATH` if it is not found |
| Graphviz | Optional, for `make viz` |

Windows: install the Visual Studio C++ build tools and run `git config --system core.longpaths true`
before cloning. Backend libraries (ONNX Runtime, TensorRT-RTX, LiteRT) are described in
[Backends](../user-guide/backends.md).

### Optional feature prerequisites

The default build needs no native runtime beyond the prerequisites above.

| Cargo feature | Additional requirements |
|---|---|
| `onnx-runtime` | The pinned ONNX Runtime; `make onnxruntime-download` installs it and the related Make targets configure `ORT_DYLIB_PATH` |
| `coreml-runtime` | macOS and Xcode Command Line Tools; CoreML is supplied by macOS. Some in-memory execution paths require macOS 15+ |
| `trtx-runtime`, `trtx-enterprise` | A compatible NVIDIA GPU and driver, CUDA 13.0, and a loadable TensorRT-RTX runtime |
| `litert-runtime` | `flatc` on `PATH` plus the native LiteRT library provisioned by `litert-sys`; WPT runs also configure its library search path |
| `cann-runtime` | An OHOS/Ascend HiAI environment and `libcann_shim.so` (or `CANN_SHIM_PATH`) |
| `webnn-runtime` with `webnn-wpt-tests` | The `wasm32-unknown-unknown` target; browser tests additionally need `wasm-pack`, Node.js, Chrome/Chromium, `curl`, and `unzip` |

The mock backend features and `dynamic-inputs`, `native-examples`, `pollster`, and
`zstd-cache-compression` do not add external runtime prerequisites.

## Build and test

Use the Makefile targets; they set feature flags and environment variables consistently.
`make help` prints the same list.

| Target | Effect |
|---|---|
| `build` | `cargo build`, no backend features |
| `test` | `cargo fmt`, `clippy -D warnings`, `cargo test`, operator report drift check |
| `fmt`, `fmt-check`, `lint` | rustfmt (apply / check), clippy only |
| `clean`, `clean-all` | `cargo clean`; also the docs site and coverage output |
| `onnxruntime-download` | Fetch the pinned ONNX Runtime into `target/onnxruntime`; export `ORT_DYLIB_PATH` afterwards |
| `trtxruntime-download` | Fetch the pinned TensorRT-RTX SDK into `target/tensorrt-rtx`; `test-wpt-trtx` configures its build and runtime paths |
| `run` | Validate `examples/sample_graph.json` with the CLI |
| `viz` | Export the sample graph as Graphviz DOT |
| `onnx`, `onnx-validate` | Convert the sample graph to ONNX (`GRAPH_FILE=...` selects another graph); also execute it with ONNX Runtime |
| `coreml`, `coreml-validate` | CoreML conversion and execution (macOS) |
| `build-coreml`, `test-coreml` | Build all targets; run library and ordinary integration tests. `COREML_FEATURES=coreml-runtime` also checks the build without dynamic inputs |
| `test-coreml-gather` | Focused active-dimension gather regressions, including scalar indices; `TEST_FILTER` selects a test |
| `litert`, `cann` | LiteRT and CANN conversion of the sample graph |
| `validate-cann-env`, `cann-build`, `cann-device-test` | OpenHarmony toolchain check, cross build, device test through `hdc`; see [CANN](../integration/cann.md) |
| `validate-all-env` | Build, unit tests, ONNX and CoreML validation in one run |
| `fetch-wpt` | Download the pinned WPT corpus into the cache (`WPT_DIR` overrides) |
| `test-wpt` | WPT conformance on ONNX Runtime CPU |
| `test-wpt-op OP=relu` | One operation; `WPT_BACKEND=onnx|trtx|litert|coreml` selects the backend |
| `test-wpt-trtx`, `test-wpt-litert`, `test-wpt-coreml`, `test-wpt-cann` | Per-backend WPT runs (CANN cross-compiles and runs on the device over `hdc`); `test-wpt-report` and `test-wpt-coreml-report` also write the JSON report |
| `wpt-sync-onnx`, `wpt-sync-trtx`, `wpt-sync-litert`, `wpt-sync-coreml`, `wpt-sync-cann` | Regenerate snapshots and expected-failure lists |
| `webnn-chromedriver`, `test-webnn-wpt-chrome`, `test-webnn-wpt-chrome-headless` | Browser WebNN graph-build tests in Chrome; see [Browser WebNN](../integration/webnn-browser.md) |
| `docs-api` | rustdoc with `-D warnings` |
| `docs-build`, `docs-serve`, `ci-docs`, `docs-clean` | MkDocs site into `site/`, live preview, strict mode as CI runs it, remove the site |
| `docs-backend-ops`, `docs-backend-ops-check` | Regenerate the operator support report; check it for drift |
| `coverage`, `coverage-html`, `coverage-lcov`, `coverage-open`, `coverage-clean` | cargo-llvm-cov reports; see [Code Coverage](code-coverage.md) |

CI type-checks every backend. Do the same before pushing when shared code changed:

```bash
cargo check
cargo check --features onnx-runtime
cargo check -F trtx-runtime --all-targets
cargo check --features litert-runtime
cargo check --features cann-runtime
cargo check --features coreml-runtime          # macOS
cargo test --lib
cargo test --lib --features cann-runtime-mock
```

## Workflow for a change

1. Branch from `main`.
2. Make the change with its tests. Unit tests live in `#[cfg(test)]` modules at the end of each
   file; converter tests decode the emitted model and assert on its structure.
3. Run `make test`.
4. If an operation or a converter changed: run the affected WPT cases on every backend you can
   (`make test-wpt-op OP=<name>`, `WPT_BACKEND=<backend>` to pick one), regenerate snapshots or
   expected-failure lists with `make wpt-sync-<backend>`, review the diff, and run
   `make docs-backend-ops`.
5. Update the documentation as described in the [Documentation Policy](documentation-policy.md).
   Run `make docs-api` and `make ci-docs` when rustdoc or pages changed.
6. Open a pull request with the template filled in. CI runs formatting, clippy, tests, rustdoc,
   the operator report drift check, the docs build and the WPT suites.

`scripts/install-git-hooks.sh` installs a pre-commit hook that runs `cargo fmt --check` and
clippy when Rust files are staged.

## Adding an operation

Check Chromium's implementation first. It is the WebNN reference and shows the lowering each
backend needs, for example casts for boolean types or decompositions:

- https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/ort/graph_builder_ort.cc (ONNX Runtime)
- https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/coreml/graph_builder_coreml.cc (CoreML)
- https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/tflite/graph_builder_tflite.cc (LiteRT)

Then:

1. `src/operator_options.rs`: add the `ML*Options` struct when the spec defines a new dictionary.
2. `src/operators.rs`: add the `Operation` variant with named operand fields, its `op_type()`
   name and the `from_json_attributes` parsing.
3. `src/shape_inference.rs`: output shape and data type rules, with unit tests.
4. `src/mlgraphbuilder.rs`: the builder method, through one of the `impl_*_op!` macros or
   explicitly when the signature does not fit.
5. `src/webnn_json.rs`: import and export mapping for the text and JSON formats.
6. Converters: `src/converters/onnx.rs`, `coreml_mlprogram.rs`, `trtx.rs`, `litert.rs`, `cann.rs`.
   A backend that cannot support the operation must reject it explicitly (LiteRT:
   `unsupported_ops`; CANN: `is_supported_op`).
7. WPT: `make test-wpt-op OP=<name>` on every backend you can run, then `make wpt-sync-<backend>`.
8. `make docs-backend-ops`, and add the method to the operation table in
   `docs/user-guide/api-reference.md`.

## Adding a backend

1. Converter: implement `GraphConverter` in `src/converters/<name>.rs` and register it in
   `ConverterRegistry::with_defaults`.
2. Backend: implement `MLBackendContext`, `MLBackendBuilder` and `ListDevices` in
   `src/backends/<name>.rs`; add the `MLBackendGraph` variant and a `DisabledContext` alias in
   `src/backends/mod.rs`.
3. Selection: add the `Backend` and `BackendDevice` variants and the position in the order in
   `src/backend_selection.rs`; extend the fields of the `NoBackendAvailable` errors.
4. Cargo: feature flag, optional dependencies, a mock feature when the hardware is not available
   in CI, and a `cargo check` step in `.github/workflows/ci.yml`.
5. Tests: a WPT backend entry in `tests/wpt_conformance/wpt_backend.rs`, an expected-failure
   file or snapshots, a `test-wpt-<name>` Make target, an integration test under `tests/`.
6. Report: a detection rule in `scripts/generate_backend_operator_report.py` and its tests.
7. Docs: `docs/user-guide/backends.md`, the feature table in `src/lib.rs`, an integration page
   when setup is involved.

## Continuous integration

| Workflow | Trigger | Content |
|---|---|---|
| `ci.yml` | push, pull request | Cargo.lock consistency, `cargo fmt --check`, `cargo check` per feature (including wasm32, and CoreML on macOS), `cargo test --lib` on Linux and macOS plus the CANN mock, rustdoc with `-D warnings`, operator report drift check and generator tests, MkDocs strict build, version check on tags |
| `wpt-conformance.yml` | push, pull request | WPT suites for ONNX Runtime (Linux), LiteRT (Linux, non-blocking) and CoreML (macOS) |
| `wpt-conformance-nightly.yml` | schedule | Full WPT run with JSON and HTML reports; publishes the dashboard together with the docs site |
| `snapshot-sync.yml` | weekly, manual | Regenerates snapshots and expected failures against the pinned WPT revision and opens a pull request |
| `rustnnpt-gate.yml` | pull request | Runs the external rustnnpt conformance runner against the PR's rustnn and enforces a minimum pass rate |
| `docs.yml`, `docs-pr.yml` | push to `main`, pull request | MkDocs strict build, rustdoc embedded under `/api/`, WPT report embedded, link check on PRs, deploy to GitHub Pages from `main` |
| `publish.yml` | GitHub release | fmt, clippy, tests, `cargo publish` |

TensorRT-RTX has no GPU runner. CI compiles it (`cargo check -F trtx-runtime --all-targets`);
contributors run `make test-wpt-trtx` locally before and after changing the converter.

## Releasing

Bump `version` in `Cargo.toml`, update `Cargo.lock`, tag `vX.Y.Z` and create a GitHub release.
`publish.yml` verifies that the version matches the tag and publishes to crates.io; docs.rs
builds the API docs with the features listed under `[package.metadata.docs.rs]`.

## Code style

- `cargo fmt` and `cargo clippy --all-targets -- -D warnings` must pass.
- Comments explain non-obvious decisions in one line and use ASCII only. No emojis anywhere in
  the repository.
- Errors are typed with `thiserror`, carry context and are `Send + Sync`.
- Every public item gets rustdoc: `src/lib.rs` enables `#![warn(missing_docs)]` and CI denies
  warnings. A new module gets a `//!` header saying what the module owns. Enum variants that
  are only operand indices (`Operation`) and error fields described by their message carry an
  explicit `#[allow(missing_docs)]`.

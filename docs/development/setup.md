# Setup and Workflow

## Toolchain

| Tool | Notes |
|---|---|
| Rust | `rust-toolchain.toml` pins the channel (1.97.0 at the time of writing) with `rustfmt` and `clippy`; rustup installs it on first use. The CI workflows pin the same version, so bump them together |
| `protoc` | Required: `build.rs` compiles the CoreML protobuf schemas, and `webnn-onnx-utils` the ONNX one (default `onnx` feature). Linux `apt-get install protobuf-compiler`, macOS `brew install protobuf`, Windows `winget install Google.Protobuf` |
| `flatc` | Required for `litert-runtime` (TFLite schema). CI downloads the flatbuffers release binary; macOS `brew install flatbuffers` |
| Node.js | WPT corpus fetch and evaluation (`scripts/fetch_wpt.mjs`, `scripts/wpt_bridge/`) |
| Python 3 | Only for MkDocs (`pip install -r docs/requirements.txt`) and `scripts/generate_backend_operator_report.py` |
| libclang | `trtx-runtime`: the `trtx-sys` crate generates bindings with autocxx; set `LIBCLANG_PATH` if it is not found |
| Doxygen and cbindgen | Required for the documentation website and `make docs-capi`; install Doxygen with your package manager and cbindgen with `cargo install cbindgen --locked` |

Windows: install the Visual Studio C++ build tools and run `git config --system core.longpaths true`
before cloning. Backend libraries (ONNX Runtime, TensorRT-RTX, LiteRT) are described in
[Backends](https://rustnn.github.io/rustnn/api/rustnn/backend_selection/).

### Optional feature prerequisites

The default build needs no native runtime beyond the prerequisites above.

| Cargo feature | Additional requirements |
|---|---|
| `onnx-runtime` | The pinned ONNX Runtime; `make onnxruntime-download` installs it and the related Make targets configure `ORT_DYLIB_PATH` |
| `coreml-runtime` | Xcode and system CoreML on macOS/iOS; see the [CoreML guide](../integration/coreml.md) for deployment requirements and the tvOS dependency caveat |
| `trtx-runtime`, `trtx-enterprise` | A compatible NVIDIA GPU and driver, CUDA 13.0, and a loadable TensorRT-RTX runtime |
| `litert-runtime` | `flatc` on `PATH` plus the native LiteRT library provisioned by `litert-sys`; WPT runs also configure its library search path |
| `cann-runtime` | An OHOS/Ascend HiAI environment and `libcann_shim.so` (or `CANN_SHIM_PATH`) |
| `webnn-runtime` with `webnn-wpt-tests` | The `wasm32-unknown-unknown` target; browser tests additionally need `wasm-pack`, Node.js, Chrome/Chromium, `curl`, and `unzip` |

The optional features `dynamic-inputs`, `native-examples`, `pollster`, and
`zstd-cache-compression` do not add external runtime prerequisites.

Defaults enable `onnx-converter` and `burn-cpu-runtime`. Each runtime includes its
converter. For serialization alone, disable defaults and enable `<backend>-converter`.
See the [crate feature table](https://rustnn.github.io/rustnn/api/rustnn/#cargo-features).

## Build and test

The Cargo workspace contains the `rustnn` library at the root and `rustnn-cli` in `cli/`.
Both are default members. Use `-p rustnn` for library-only commands and `-p rustnn-cli` for
the CLI. CLI backend features forward to the library with its defaults disabled.
The CLI is disabled on wasm32: its entry point is empty and its dependencies are omitted.

Use the Makefile targets; they set feature flags and environment variables consistently.
`make help` prints the same list.

| Target | Effect |
|---|---|
| `build` | `cargo build` with default features |
| `check-converters` | Check each converter independently with defaults disabled |
| `test-cli` | CLI default, no-default-feature and portable-converter integration tests |
| `test` | `cargo fmt`, `clippy -D warnings`, `cargo test`, operator report drift check |
| `fmt`, `fmt-check`, `lint` | rustfmt (apply / check), clippy only |
| `clean`, `clean-all` | `cargo clean`; also the docs site and coverage output |
| `onnxruntime-download` | Fetch the pinned ONNX Runtime into `target/onnxruntime`; export `ORT_DYLIB_PATH` afterwards |
| `trtxruntime-download` | Fetch the pinned TensorRT-RTX SDK into `target/tensorrt-rtx`; `test-wpt-trtx` configures its build and runtime paths |
| `benchmark-coreml-kv` | Self-contained attention/cache comparison; optionally set `COREML_KV_CONFIG` for an external model/reference fixture; see [Examples](../user-guide/examples.md#coreml-kv-cache-benchmark) |
| `test-coreml-kv-benchmark` | Oracle and cache-content regressions for all storage modes, without external fixtures; run in macOS CI |
| `build-coreml`, `test-coreml` | Build all targets; run library and ordinary integration tests, including execution. `COREML_FEATURES=coreml-runtime` also checks the build without dynamic inputs |
| `test-coreml-gather` | Focused active-dimension gather regressions, including scalar indices; `TEST_FILTER` selects a test |
| `test-coreml-dtypes` | Focused numeric boundary regressions for typed dispatch and the convenience API; CI runs with and without dynamic inputs |
| `test-coreml-gelu` | Exact Half GELU precision, shapes and compositions, including all Half encodings |
| `test-coreml-precision` | Native typed-boundary, Half layout, constant, fanout and bounded-dynamic regressions |
| `test-coreml-tensor-reuse` | Retained storage ownership, type conversion and shape regressions; CI runs with and without dynamic inputs |
| `validate-cann-env`, `cann-build`, `cann-device-test` | OpenHarmony toolchain check, cross build, device test through `hdc`; see [CANN](../integration/cann.md) |
| `validate-all-env` | Build and unit tests in one run |
| `fetch-wpt` | Download the pinned WPT corpus into the cache (`WPT_DIR` overrides) |
| `test-wpt` | WPT conformance on ONNX Runtime CPU |
| `test-wpt-tolerance`, `test-wpt-tolerance-parity` | Backend-independent comparison regressions and direct upstream JavaScript-helper parity; the latter requires Node.js and a fetched WPT corpus and also runs after successful full ONNX/CoreML WPT targets |
| `test-wpt-op OP=relu` | One operation; `WPT_BACKEND=onnx|trtx|litert|coreml` selects the backend |
| `test-wpt-trtx`, `test-wpt-litert`, `test-wpt-coreml`, `test-wpt-cann` | Per-backend WPT runs (CANN cross-compiles and runs on the device over `hdc`); `test-wpt-report` and `test-wpt-coreml-report` also write the JSON report |
| `wpt-sync-onnx`, `wpt-sync-trtx`, `wpt-sync-litert`, `wpt-sync-coreml`, `wpt-sync-cann` | Regenerate snapshots and expected-failure lists |
| `webnn-chromedriver`, `test-webnn-wpt-chrome`, `test-webnn-wpt-chrome-headless` | Browser WebNN graph-build tests in Chrome; see [Browser WebNN](../integration/webnn-browser.md) |
| `docs-api` | rustdoc with `-D warnings` |
| `docs-capi` | Regenerate `rustnn.h` from `src/capi.rs` and build the C/C++ Doxygen reference into `target/doxygen/html/`; see [C and C++ API](../reference/c-api.md) |
| `docs-build`, `docs-serve`, `ci-docs`, `docs-clean` | MkDocs site into `site/` with the Doxygen reference at `site/c-api/`, live preview, strict mode as CI runs it, remove the site |
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
8. `make docs-backend-ops`, and document the builder method and its option fields in source.

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
7. Docs: the feature table in `src/lib.rs`, selection module docs, and a shared integration
   page when setup is involved.

## Continuous integration

| Workflow | Trigger | Content |
|---|---|---|
| `ci.yml` | push, pull request | Cargo.lock consistency, `cargo fmt --check`, `cargo check` per feature (including wasm32, and CoreML on macOS), `cargo test --lib` on Linux and macOS plus the CANN mock, rustdoc with `-D warnings`, operator report drift check and generator tests, MkDocs strict build |
| `wpt-conformance.yml` | push, pull request | WPT suites for ONNX Runtime and Burn CPU (Linux), LiteRT (Linux, non-blocking) and CoreML (macOS) |
| `wpt-conformance-nightly.yml` | schedule | Full WPT run with JSON and HTML reports; publishes the dashboard together with the docs site |
| `wpt-sync.yml` | weekly, manual | Regenerates snapshots and expected failures against the pinned WPT revision and opens a pull request |
| `rustnnpt-gate.yml` | pull request | Runs the external rustnnpt conformance runner against the PR's rustnn and enforces a minimum pass rate |
| `docs.yml`, `docs-pr.yml` | push to `main`, pull request | MkDocs strict build, rustdoc embedded under `/api/`, WPT report embedded, link check on PRs, deploy to GitHub Pages from `main` |
| `publish.yml` | GitHub release, manual | release tag matches `Cargo.toml`, fmt, clippy, tests, `cargo publish` |

TensorRT-RTX has no GPU runner. CI compiles it (`cargo check -F trtx-runtime --all-targets`);
contributors run `make test-wpt-trtx` locally before and after changing the converter.

## Releasing

Bump `version` in `Cargo.toml`, update `Cargo.lock`, tag `vX.Y.Z` and create a GitHub release.
`publish.yml` fails when the release tag does not match the `Cargo.toml` version, then runs
formatting, Clippy and library tests before publishing to crates.io. The cargo-c metadata in
`Cargo.toml` (`[package.metadata.capi.*]`) carries the version twice more; keep the three in
sync. docs.rs builds the API docs with the features listed under `[package.metadata.docs.rs]`.

Before publishing, run `cargo package --list` and ensure every Markdown `include_str!` input is
present, then `cargo package` to verify the packaged crate builds. The active docs are compiler
inputs; excluding them would break both downstream compilation and docs.rs. Build Rustdoc
from the extracted package as well when changing shared documentation.

## Code style

- `cargo fmt` and `cargo clippy --all-targets -- -D warnings` must pass.
- Comments explain non-obvious decisions in one line and use ASCII only. No emojis anywhere in
  the repository.
- Errors are typed with `thiserror`, carry context and are `Send + Sync`.
- Every public item gets rustdoc: `src/lib.rs` enables `#![warn(missing_docs)]` and CI denies
  warnings. A new module gets a `//!` header saying what the module owns. Enum variants that
  are only operand indices (`Operation`) and error fields described by their message carry an
  explicit `#[allow(missing_docs)]`.

### Burn CPU development

Use `cargo check --features burn-cpu-runtime --all-targets` and
`cargo test --features burn-cpu-runtime --lib --test test_burn_precision` for the CPU
interpreter and its public-executor datatype regressions. No native runtime
installation is needed. Run `make test-wpt-burn-cpu` and regenerate its expected failures
with `make wpt-sync-burn-cpu`. See [Burn CPU](../integration/burn-cpu.md).

## ESRGAN shape benchmark

Run `make benchmark-esrgan` with `trtx-runtime`, CUDA and TensorRT-RTX available. The benchmark
reads `~/Downloads/esr_gan_fp16_fixed/real_esrgan_x2_fp16_resize_fixed.webnn` by default;
pass another topology with `ESRGAN_ARGS='--model /path/model.webnn'`. No safetensors file is
needed: finite, deterministic synthetic FP16 constants are generated before measurement.
These runs measure performance, not image quality.

All configurations infer at NCHW `[1, 3, 1080, 1920]` and produce `[1, 3, 2160, 3840]`:

| Configuration | Minimum spatial input | Preferred spatial input | Maximum spatial input |
|---|---|---|---|
| `static-1080p` | 1080 x 1920 | 1080 x 1920 | 1080 x 1920 |
| `dynamic-1080p` | 2 x 2 | 1080 x 1920 | 1080 x 1920 |
| `dynamic-2k` | 2 x 2 | 1080 x 1920 | 1440 x 2560 |

Here "2k" means QHD (2560 x 1440). ESRGAN's pixel-unshuffle requires even spatial dimensions.
The dynamic cases retain runtime dimensions throughout reshapes and resizes.

CSV output reports each engine compilation, first inference (including TensorRT JIT), and the
mean/minimum/maximum warmed inference times in milliseconds. Compilation includes graph lowering,
engine building, deserialization, refit and execution-context creation; it excludes parsing,
constant generation, context initialization and graph cloning. Dynamic activation memory is
initially allocated for the preferred 1080p shape. Inference includes dispatch and
GPU completion, excluding tensor allocation and host/device transfers. Engine caching, runtime
caching and CUDA graph capture are disabled for all cases. CUDA synchronization brackets each
measurement; warmup follows the separately timed first inference.

```bash
make benchmark-esrgan ESRGAN_ARGS='--device 0 --compile-samples 3 --infer-samples 10 --warmup 3'
make benchmark-esrgan ESRGAN_ARGS='--case dynamic-2k'
make benchmark-esrgan ESRGAN_ARGS='--min-height 1080 --min-width 1920'
make benchmark-esrgan ESRGAN_ARGS='--validate-only'
cargo test --no-default-features --features trtx-runtime,dynamic-inputs \
  --test test_trtx_dynamic_images -- --ignored
```

Use `--min-height` and `--min-width` to narrow dynamic profiles while keeping inference and
the preferred shape at 1080p. Both minima must be even and between 2 and the inference size.
The CSV includes each case's minimum height and width.

Use `--validate-only` to parse the real topology and check all three input/output shape
configurations without a GPU. Select `--case` to isolate a failure or benchmark one profile.
Full-resolution engine builds can require substantial device memory; compilation failures are
returned as errors, never recorded as successful timing samples.

To compare the original ONNX network with the TensorRT-RTX CLI, use the same profile bounds,
active shape, disabled CUDA graphs and runtime-sized activation memory. The ONNX CLI uses the
model's real weights; the rustnn benchmark uses synthetic weights, so this checks performance
rather than output equality. Its reported engine-build scope also differs from rustnn's full
compile scope.

```bash
~/Downloads/TensorRT-RTX-1.7.1.116/bin/tensorrt_rtx \
  --onnx="$HOME/Downloads/esr_gan_fp16_fixed/real_esrgan_x2_fp16_resize_fixed.onnx" \
  --minShapes=input:1x3x2x2 --optShapes=input:1x3x1080x1920 \
  --maxShapes=input:1x3x1440x2560 --shapes=input:1x3x1080x1920 \
  --device=0 --allocationStrategy=runtime --noCudaGraph \
  --iterations=5 --duration=0 --warmUp=4000 --avgRuns=1
```

Use `--maxShapes=input:1x3x1080x1920` for the smaller maximum, and
`--minShapes=input:1x3x1080x1920` for the narrower minimum. A fixed-shape ONNX comparison
requires a copy of the model whose input dimensions are made concrete before building.
For Nsight Systems captures, prefix the executable command with
`nsys-internal profile -tcuda,nvtx,osrt --gpu-metrics-devices=all --output=<report-prefix>`.
Run GPU benchmarks sequentially and keep timings gathered under profiling separate from
unprofiled timing samples.

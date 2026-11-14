# Changelog

All notable changes to this project are documented in this file.

This changelog consolidates the previous `RELEASE_NOTES_*.md` files into a single history.

## [Unreleased]

### Added
- CoreML: `CoremlOptions::compiled_bundle_cache` keeps the compiled `.mlmodelc` of every built graph under a caller-chosen directory and loads it on the next build of the same graph instead of compiling again (rustnn extension, off by default). Entries are published with a staging directory and one rename, unusable entries are rebuilt once, and a cache failure never fails a build; `CoremlLoadDiagnostics::cache` reports the `CacheOutcome` (`Disabled`, `Hit`, `Miss` or `Error`). With the cache enabled, graphs with inputs compile from a URL.
- CoreML: `CoremlOptions::float16_precision` selects the float16 lowering (rustnn extension). `Native`, the default, keeps native float16 MIL kernels and one MIL program; `Exact` evaluates float16 arithmetic, activations, normalization and data movement in float32 with float16 storage and materializes explicit float16 rounding boundaries as Pipeline children (#279). `CoremlMlProgramConverter::convert_with_float16_precision` exposes the same choice; `WPT_COREML_FLOAT16` selects it for WPT.

### Changed
- `clap` and `tracing-subscriber` are now optional, behind a new default `cli` feature that also gates the `rustnn` binary; build with `default-features = false` to use the library without them.
- The ONNX converter, `protos::onnx` and the `webnn-onnx-utils` dependency are behind a new default `onnx` feature (implied by `onnx-runtime`); `webnn-graph` is used without its `onnx` feature. Builds with `default-features = false` and a non-ONNX backend no longer compile the ONNX schema.

### Fixed
- CoreML: `reciprocal` keeps the sign of -0, `log` no longer adds an epsilon, `roundEven` rounds ties to even on GPU and Neural Engine requests, float32 `tanh`, `sign` and `neg` keep NaN and zero signs, and `layerNormalization` applies scale and bias in the declared order on nontrailing or permuted axes (#279).
- CoreML: `convTranspose2d` and zero-epsilon `instanceNormalization` compile from a URL, where Core ML matches WebNN for the reproduced cases (#279).
- CoreML: NHWC `instanceNormalization` with runtime or computed scale and bias normalizes over the spatial axes instead of the last two, and a following NHWC `conv2d` reads its transposed output (#279).

## [0.6.0] - 2026-10-14

First release of the WebNN `MLContext` API from `main`. The 0.5.x releases were cut from the
`v0.5` branch and are the earlier converter and loader crate; everything below is new relative to
`v0.5.12` (2026-05-04). Downstream bindings (pywebnn, webnnjs) need updates for the breaking
changes listed under Changed.

### Added
- Experimental C and C++ API: a cargo-c library with a generated `rustnn.h` and the header-only `rustnn.hpp` wrapper, documented in `docs/reference/c-api.md` and a Doxygen reference published under `/c-api/` (#233, #264, #271).
- `MLGraphBuilder::constant_from_value` for scalar constants (#232).
- `dynamically_load_tensorrt` for custom TensorRT-RTX library paths (#280).
- CoreML: opt-in tensor storage reuse for KV-cache chaining with `BackendStatistics` counters, retained load diagnostics after a compute-unit fallback, SmolLM shape tensors, exact boundary data type conversions and strided storage; the compiled model is `Send` (#223, #255, #256, #258, #285).
- LiteRT and CANN operator coverage extended (#210, #237, #242, #250).
- WPT harness: strict upstream tolerances behind `WPT_STRICT_TOLERANCE`, CoreML device and tensor-mode policies, numeric fidelity of the fixture bridge (signed zeros, exact float literals) and audit mode (#268, #274).
- WebNN API in Rust: `MLContext`, `MLGraphBuilder`, `MLGraph`, `MLTensor`, `dispatch`, tensor read and write; backend and device hints on `MLContextOptions`; `RustNNOptions` with `TrtxOptions` (#111, #155, #181).
- Backends behind the unified API: TensorRT-RTX with native network lowering, refittable weights, zstd-compressed engine and runtime caches and CUDA graphs (#128, #162, #182, #189, #192, #193); CoreML (#140); LiteRT (#149, #179, #187, #196); Huawei CANN on OpenHarmony (#198, #203); browser WebNN bindings for wasm32 (#206).
- In-repo WPT conformance harness with per-backend expected-failure lists, JSON and HTML reports, audit mode and a weekly `wpt-sync` workflow that bumps the pinned WPT revision (#151, #166, #176, #197, #226, #241).
- `MLGraphBuilder::rustnn_save_webnn`: save a graph as `.webnn` text plus `.safetensors` weights (#184).
- `int4` and `uint4` data types (#148); shape inference for the normalization operations (#125).
- Examples: builder-API fast style transfer, ResNet-50, SmolLM on `MLContext` (#124, #162).
- `TRTX_JSON_DUMP_PATH` engine dumps (#177).
- `MLOperand::rustnn_index()` and `From<MLOperand> for OperandIndex` to fill operand fields of `ML*Options` such as `MLConv2dOptions::bias`.
- Rust API docs (rustdoc) built with warnings denied in CI and published under `/api/`; every public item is documented and `#![warn(missing_docs)]` keeps it that way; `docs/development/documentation-policy.md`.
- Documentation pages: CoreML, LiteRT, CANN and browser WebNN backend pages under `docs/integration/`, `docs/development/converters.md` (converter contract and per-backend lowering rules), `docs/user-guide/troubleshooting.md`, `docs/reference/graph-files.md` (`.webnn`, JSON and weight formats, shared with the loader rustdoc); the Make target table in `docs/development/setup.md` is complete. Guides that duplicated rustdoc were consolidated into the API reference (#243, #287).
- Span instrumentation for a [`tracing`](https://docs.rs/tracing) subscriber, covering the crate's
  common paths; the CLI prints spans and `log` records for `RUST_LOG`.

### Changed
- **Breaking:** `MLConv2dOptions::input_layout`, `filter_layout` and the other layout fields use the `MLInputOperandLayout`, `MLConv2dFilterOperandLayout` and `MLConvTranspose2dFilterOperandLayout` enums instead of strings (#260).
- **Breaking:** the legacy constant operations are removed from the public graph API; constants are operands with data (#266).
- **Breaking:** the old TensorRT executor and `test_trtx_execution` are removed; TensorRT-RTX runs through `MLContext` (#185, #245).
- **Breaking:** the crate no longer declares `cdylib` and `staticlib` crate types; the C ABI is built with cargo-c (#246).
- Shape inference is unified so graphs reloaded from `.webnn` files are executable without re-inference (#238).
- WPT results are tracked through expected-failure lists only; the PASS snapshots and the LiteRT dtype and unsupported-operation gates are gone (#239, #241).
- TensorRT-RTX: the runtime cache is no longer invalidated by rustnn source changes; `triangular` uses `select` so NaN and infinity do not leak into masked positions (#248, #254).
- The selected backend is reported as a span event instead of a `log` line, and messages the spans
  already carry as fields moved from `debug` to `trace`.
- **Breaking:** `MLContext::dispatch`, `MLGraphBuilder::build` and `rustnn_save_webnn` take `MLNamedTensors` and `MLNamedOperands` (`BTreeMap`) instead of `HashMap` (#202).
- `webnn-graph` 0.4 and `webnn-onnx-utils` 0.1.1 are taken from crates.io instead of git branches, which is what allows this crate to be published.
- **Breaking:** `MLGraphBuilder::conv2_with_options` renamed to `conv2d_with_options` (the deprecated alias was removed in #264); `MLTensor::destoy` renamed to `destroy`.
- Errors are `Send + Sync` (#170); backend traits require `Send + Sync` (#145, #147).
- Dispatch validates tensor bindings: shapes and data types (#126, #132) and duplicate tensors (#220).
- Toolchain 1.97 (#211); `ort` 2.0.0-rc.13 (#208); `trtx` 0.8 for TensorRT-RTX 1.6 (#200); `thiserror` 2 (#190).
- CoreML: stable `reduceLogSumExp`, complete `resample2d`, gather normalization for dynamic and scalar indices, failing shims on Linux (#213, #215, #227, #209).
- Documentation rewritten for the Rust API; superseded plans moved to `docs/archive/`; Python-era files removed (`MANIFEST.in`, `pytest.ini`, the old `TODO.txt`); the built MkDocs site is no longer tracked.

### Fixed
- CoreML: gather bounds for active dimensions, triangular masking and diagonal handling (#228, #235).
- LiteRT: dispatch buffer order, prebuilt library path resolution from `litert-sys`, converter checked on macOS (#244, #282, #283).
- ONNX: exact output sizes for `resample2d` (#259).
- Documentation site: the WPT conformance dashboard no longer goes stale between nightly runs, the pass-rate badge is back in the README, and the Pages deploy is skipped on forks without Pages (#263, #265, #286).
- TensorRT: CUDA graph replay, write synchronization, negative scatter indices, cache write races and many operator lowerings (#167, #168, #169, #182, #194, #231).
- ONNX Runtime: bool and uint8 handling (#180); bidirectional LSTM and GRU, an fp16 cast that broke optimized ONNX Runtime builds, `Squeeze` axes (#219).
- `cargo test` passes again: prost-build's `cleanup-markdown` feature turns the code blocks in the CoreML `.proto` comments into text fences instead of failing doctests, and the WPT harness exits successfully instead of failing when no backend feature is compiled in.

## [0.5.12] - 2026-05-04

Published from a branch off `main` (2026-05-02) with the `web` feature dropped for crates.io. Contains everything merged since `v0.5.11`.

### Added
- Strongly typed graph model: the `Operation` enum with one variant per WebNN operation and `ML*Options` structs replace string operation types and JSON attributes (#57, #78, #80, #90, #98, #99).
- Dynamic dimensions (`{ name, maxSize }`) behind the `dynamic-inputs` feature (#16, #74).
- Operations: `cumulativeSum`, `roundEven`, `reverse`, `resample2d`, `notEqual`, `linear`, `isNaN`, `isInfinite`, `gruCell`; `gatherElements` on CoreML (#34, #36, #37, #39, #41, #46, #48, #49, #50, #69).
- TensorRT-RTX engine building through the builder interface, `--run-trtx`, public `TrtxConverter::build_network`, re-export of `dynamically_load_tensorrt` (#9, #13, #79, #105).
- ONNX external weights for models above 2 GB (#104); quantization support (#10).
- Generated backend operator support report with a CI drift check (#38).
- WPT-based test suites and the rustnnpt gate in CI (#21, #52, #53).

### Changed
- Python bindings split into the `pywebnn` repository; rustnn is a pure Rust crate (#11).
- **WebNN spec alignment:** `round` replaced by `roundEven`; legacy JSON with `op_type: "round"` is still accepted as `roundEven`. Non-spec operators `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh` removed.
- WebNN attributes use camelCase (#14).
- ONNX protos shared through `webnn-onnx-utils`.

### Fixed
- ONNX conformance sweep: conv2d and convTranspose2d layout and padding, reductions, quantize and dequantize, pad, layerNormalization, the gather family, softmax, clamp, argMin and argMax, pooling (#23 to #45).
- CoreML MLProgram: required parameters for conv, slice and reshape, explicit gelu mode, 0-D scalars, boolean operations, pooling (#63, #68, #72, #102, #103, #106 to #109).

## [0.5.5] to [0.5.11] - 2025-12-29

Patch releases for the Python wheel publishing pipeline of the time (bundled ONNX Runtime, manylinux builds, platform tarball names, `protoc` in CI) plus a backend info API. No graph or converter changes.

## [0.5.2] - 2025-12-28

### Overview
- Focused patch release with 3 commits since `v0.5.1`.
- Improves WebNN text/JSON import and adds a MiniLM embeddings demo.

### Added
- MiniLM embeddings demo from Hugging Face Hub.
- Make target: `make minilm-demo-hub`.
- `MINILM_MODEL_ID` override and companion usage/comparison docs.

### Changed
- WebNN text loader now sanitizes identifiers (`.` and `:` to `_`).
- Loader now inlines weights from adjacent `manifest.json` and `model.weights` to better support `onnx2webnn` exports.
- JSON import now runs shape inference.
- JSON import now deduplicates outputs.
- Python helpers added for unresolved shape debugging (`count_unknown_shapes`, structured debug output).

### Notes
- No breaking changes.
- Python wheel version follows Cargo version (`0.5.2`) via dynamic versioning.

## [0.3.0] - 2025-12-14

### Overview
- Major release with 130 commits since `v0.2.0`.
- Focus areas: WPT conformance, TensorRT support, CoreML Float16, large operation fixes, and documentation expansion.

### Highlights
- 91.3% WPT conformance.
- TensorRT backend integration for NVIDIA GPU acceleration.
- CoreML Float16 support with MLPackage weight files.
- 100+ operation/backend bug fixes.
- Refactoring and code quality improvements.

### Added
- TensorRT executor integration via `trtx-rs`.
- Windows TensorRT setup guide.
- TensorRT integration planning guide.
- CoreML Float16 support infrastructure across phases (weight builder, files, integration tests).
- WPT conformance datasets and mappings for Tier 1 and reduction operations.
- WPT converter tooling using Node.js extraction.
- Performance benchmark docs.
- IPC design document and Chromium comparison docs.
- GGML integration planning documentation.

### Changed
- ONNX runtime stack migrated from `onnxruntime-rs` to `ort v2.0.0-rc.10`.
- ONNX Runtime upgraded to `v1.23.2`.
- Runtime initialization hardened with `Once`.
- Backend-selection documentation/status tracking updated.
- Project-wide style cleanup to remove emojis and enforce no-emoji policy.

### Fixed
- Convolution family fixes (`conv2d`, `convTranspose2d`) including bias, layout, padding/output size mapping, and WPT parameter mapping.
- Normalization fixes (`batch_normalization`, `layer_normalization`, validation rules, shape handling).
- Element-wise fixes (`neg`, `hard_swish`, `clamp`, `logical_not`, `concat`, `cast`, `log`, etc.).
- Reduction fixes (`reduce_l1`, `reduceProduct`, axes handling for ONNX opset 13).
- Gather/expand and edge-case fixes (rank-increasing expand, out-of-bounds handling, 0D tensors).
- CoreML converter stability fixes (required params, dtype handling, scalar support, panic handling).
- Data-type support expansion (including bool to uint8 casting path for ONNX compatibility).
- Python test fixture and CI reliability issues.

### Testing and Quality
- Dual-backend testing support for ONNX and CoreML.
- Added separate Make targets for ONNX/CoreML WPT runs.
- Added multi-output test harness support.
- Statistics at release time:
  - 2262 Python tests passing.
  - 133 Rust tests passing.

### Compatibility
- No breaking changes.
- Drop-in replacement for `v0.2.0`.

## [0.2.0] - 2024-12-08

### Overview
- Major feature release transitioning rustnn from validation/conversion into a full WebNN implementation with execution.

### Highlights
- 85 WebNN operations implemented (89% spec coverage at release time).
- Real execution via ONNX Runtime and CoreML (with NumPy I/O).
- W3C WebNN explainer alignment (device selection and MLTensor concepts).
- Production-oriented examples (MobileNetV2, text generation, training flow).

### Added
- Full operation coverage across shape inference, Python API, ONNX backend, and CoreML MLProgram backend for the 85 implemented ops.
- Async execution support (`AsyncMLContext`, `dispatch()` semantics).
- Explicit MLTensor lifecycle APIs (`create_tensor`, `read_tensor`, `write_tensor`, `destroy`).
- Runtime backend selection with `accelerated` and `power_preference` hints.
- Extensive Makefile developer targets.
- WPT conformance integration in test workflow.
- Pre-commit hook installation flow.

### Changed
- CoreML backend fully migrated from NeuralNetwork format to MLProgram (MIL).
- Project renamed from `rust-webnn-graph` to `rustnn`.
- Python package published as `pywebnn`.
- Shape inference and NumPy-style broadcasting expanded and validated at build time.

### Fixed
- ONNX compatibility with older ONNX Runtime versions.
- Logic-op cast conversion issues for older runtime constraints.
- CoreML constant handling and GEMM support.
- Async dispatch/tensor workflow tests.
- Rust warnings cleanup (warning count reduced to zero).
- CI packaging path for ONNX-runtime-enabled Python builds.

### Breaking Changes
- Context creation API changed:
  - Old: `device_type="cpu"`
  - New: `accelerated` + optional `power_preference`
- Python requirement raised to 3.11+.
- CoreML MLProgram path requires newer Apple platform versions (iOS 18+/macOS 15+ noted at release time).

### Release Statistics
- 97 commits since `v0.1.0`.
- 320+ tests (Rust + Python + WPT).
- 3 execution backends at release time: ONNX CPU, ONNX GPU, CoreML.

## [0.1.0] - Initial Experimental Release

### Status
- Experimental proof-of-concept release.
- Not intended for production use.

### Included
- Python WebNN API foundation.
- ONNX conversion path with broad operation coverage.
- Basic CoreML conversion (`add`, `matmul`).
- Graph validation aligned with Chromium-style checks.
- Graphviz visualization support.
- NumPy tensor integration.
- Cross-platform wheel distribution:
  - Linux (`x86_64`, `aarch64`)
  - macOS (`x86_64`, `aarch64`)
  - Windows (`x64`, `x86`)

### Limitations at Release Time
- CoreML conversion support was minimal.
- `compute()` execution path did not perform real tensor computation yet.
- WebNN operation coverage was partial.
- Test coverage was limited.

---

## Legacy Release Notes

The following files were merged into this changelog:
- `RELEASE_NOTES_v0.1.0.md`
- `RELEASE_NOTES_v0.2.0.md`
- `RELEASE_NOTES_v0.3.0.md`
- `RELEASE_NOTES_v0.5.2.md`

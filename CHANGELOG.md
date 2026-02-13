# Changelog

All notable changes to this project are documented in this file.

This changelog consolidates the previous `RELEASE_NOTES_*.md` files into a single history.

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

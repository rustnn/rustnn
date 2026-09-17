# Backends

rustnn executes a compiled graph on one of several backends. The backend is chosen when the
`MLContext` is created and stays fixed for that context; graphs built from the context are
compiled for that backend only.

## Overview

| Backend | Cargo feature | Platforms | Devices | Runtime dependency | Converter |
|---|---|---|---|---|---|
| ONNX Runtime | `onnx-runtime` | Linux, macOS, Windows | CPU, GPU and NPU devices reported by ONNX Runtime execution providers | ONNX Runtime shared library, loaded from `ORT_DYLIB_PATH`; the release must provide the API level the `ort` crate is built against (`make onnxruntime-download` fetches a matching build) | `OnnxConverter`: ONNX protobuf, with an external weights file for large models |
| TensorRT-RTX | `trtx-runtime`, `trtx-runtime-mock`, `trtx-enterprise` | Linux and Windows with an NVIDIA RTX GPU | GPU, one device per CUDA device | NVIDIA driver and the TensorRT-RTX 1.6 library, found by name on `PATH` or `LD_LIBRARY_PATH`, or loaded explicitly with `dynamically_load_tensorrt` | `TrtxConverter`: builds the TensorRT network directly |
| CoreML | `coreml-runtime` | macOS (compiles to failing shims elsewhere) | CPU, GPU, NPU (Neural Engine) via compute units | none beyond macOS | `CoremlMlProgramConverter`: MLProgram (MIL) |
| LiteRT | `litert-runtime` | Linux, macOS | CPU, GPU and NPU accelerators | LiteRT libraries downloaded by `litert-sys` into `~/.cache/litert-sys/`; `flatc` at build time | `LiteRtConverter`: TFLite flatbuffer, NCHW operands transposed to NHWC |
| CANN | `cann-runtime`, `cann-runtime-mock` | OpenHarmony (`aarch64-unknown-linux-ohos`) with a Kirin NPU | NPU | HiAI through the `hiai-rs` crate | `CannConverter` |
| Browser WebNN | `webnn-runtime` (`wasm32-unknown-unknown`) | browsers with WebNN | as provided by the browser | none | generated bindings only; not selectable through `MLContext::create` yet |

Without a runtime feature the crate still validates graphs and converts them to ONNX and
CoreML, but `MLContext::create` fails with `Error::NoBackendAvailable`.

## Selection rules

`MLContext::create` resolves the WebNN hints in `MLContextOptions` (`accelerated` and the
power preference) in a fixed order, skipping backends that are not compiled in or report no
device. The order is implemented in `src/backend_selection.rs`:

1. A device hint (`with_rustnn_device_hint`) is used as given, without an availability check.
2. `accelerated` with `Default` or `HighPerformance`: TensorRT-RTX (first CUDA device), then
   CoreML (GPU), then LiteRT (GPU), then ONNX Runtime (GPU, then NPU).
3. `accelerated` with `LowPower`: CoreML (Neural Engine), then LiteRT (NPU), then ONNX Runtime
   (NPU).
4. Not accelerated: CoreML (CPU), then LiteRT (CPU), then ONNX Runtime (CPU).
5. ONNX Runtime CPU is the last resort for every accelerated request.
6. CANN is only selected when requested with a backend hint.

A backend hint restricts the search to one backend and fails with
`Error::NoBackendAvailableForBackendHint` when that backend cannot serve the hints:

```rust
use rustnn::mlcontext::{Backend, MLContext, MLContextOptions, MLPowerPreference};

let options = MLContextOptions::new(MLPowerPreference::HighPerformance, true)
    .with_rustnn_backend_hint(Backend::Trtx);
let context = MLContext::create(&options)?;
assert_eq!(context.rustnn_backend(), Backend::Trtx);
println!("{:?} {:?}", context.rustnn_device(), context.rustnn_device_type());
```

Both error variants list the backends that were wanted and the backends that are compiled in,
so a build without the expected feature is visible in the error message. `RUST_LOG=info` logs
the selected device.

## Execution model

- `MLGraphBuilder::build` converts the recorded graph with the backend's converter and compiles
  it once: an ONNX Runtime session, a TensorRT engine, a compiled CoreML model, a LiteRT
  interpreter or a HiAI model. The `MLGraph` keeps the compiled artifact together with the
  named input and output descriptors.
- `MLContext::create_tensor` allocates a tensor owned by the backend. Host access is controlled
  by the `readable` and `writable` flags of the `MLTensorDescriptor`; both are off by default.
- `MLContext::dispatch` rejects a tensor bound under two names, checks every binding's name,
  shape and data type against the compiled graph, then runs the backend. Results are written
  into the bound output tensors.
- `read_tensor` and `write_tensor` copy whole tensors. The host buffer must hold exactly
  `MLTensor::rustnn_required_bytes` bytes. The TensorRT backend synchronizes its stream in
  `write_tensor`.
- All calls are synchronous. `MLContext` is `Send + Sync`, so one context can be shared behind
  a mutex.

## Data types

| `MLOperandDataType` | Storage | Notes |
|---|---|---|
| `Float32`, `Float16` | 4 and 2 bytes | TensorRT keeps float32 math at full precision (TF32 disabled) |
| `Int32`, `Uint32`, `Int64`, `Uint64` | 4 and 8 bytes | CoreML computes integer operations in float32, which loses precision near the int32 and int64 limits |
| `Int8`, `Uint8` | 1 byte | Boolean results (comparisons, `logical*`, `isNaN`, `isInfinite`) are `Uint8` |
| `Int4`, `Uint4` | two values per byte | rustnn extension for quantized weights, used by the WPT corpus; cannot be saved to `.safetensors` |

Per-backend data type restrictions and the cases that still fail are tracked in
`tests/wpt_conformance/<backend>_expected_failures.txt`, in the PASS snapshots under
`tests/snapshots/` and on the [WPT conformance dashboard](https://rustnn.github.io/rustnn/wpt-conformance/).

## Testing a backend

| Backend | WPT conformance | Integration tests | In CI |
|---|---|---|---|
| ONNX Runtime | `make test-wpt` (PASS snapshots) | unit tests in `src/backends/ort.rs` and `src/converters/onnx.rs` | yes (Linux) |
| TensorRT-RTX | `make test-wpt-trtx` (PASS snapshots, GPU required) | `cargo test --test test_trtx_execution --features trtx-runtime` | compile check only, no GPU runner |
| CoreML | `make test-wpt-coreml` (expected-failure list) | unit tests with `--features coreml-runtime` | yes (macOS) |
| LiteRT | `make test-wpt-litert` (PASS snapshots and expected-failure list) | `cargo test --test test_litert_execution --features litert-runtime` | yes (Linux, non-blocking) |
| CANN | not run | `make cann-device-test` on an OpenHarmony device; `cargo test --lib --features cann-runtime-mock` | mock build and tests |

The [WPT Conformance Guide](../testing/wpt-test-guide.md) explains filtering by operation and
regenerating snapshots after a converter change.

## Caches

The TensorRT-RTX backend stores built engines (category `trtx`) and the TensorRT runtime cache
(category `trtx-jit`) under the platform cache directory: `~/.cache/rustnn/<category>` on Linux,
`~/Library/Caches/rustnn/<category>` on macOS and `%LOCALAPPDATA%\rustnn\<category>` on
Windows. Entries are zstd compressed. Engine keys include a hash of the converter sources, so
converter edits never reuse a stale engine, and deleting the directories is always safe.
[TensorRT-RTX](../integration/tensorrt.md) lists the options that control caching.

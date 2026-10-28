# Troubleshooting

Errors you are likely to meet, what causes them, and where to look. Error types are described
in the [error reference](https://rustnn.github.io/rustnn/api/rustnn/error/); `RUST_LOG=debug` adds the backend's own log
lines to most of these situations.

## Context creation

| Symptom | Cause and fix |
|---|---|
| `Error::NoBackendAvailable { want_*, have_* }` | No compiled backend can serve the hints. The `have_*` flags show which features the binary was built with; add `--features onnx-runtime` (or another backend) or relax the hints |
| `Error::NoBackendAvailableForBackendHint` | The hinted backend is compiled in but reports no device, or its library failed to load. Check the backend-specific rows below |
| ONNX Runtime is unavailable, or initialization reports a missing library or incompatible version | Set `ORT_DYLIB_PATH` to the downloaded ONNX Runtime 1.29 or newer shared library and restart the process. rustnn uses `ort::init_from` to return loading errors; backend discovery skips an unavailable runtime. An old `onnxruntime.dll` in Windows `System32` can be selected if the variable is unset. See [Getting Started](getting-started.md#provide-onnx-runtime) |
| ONNX Runtime repeats the same initialization error after fixing `ORT_DYLIB_PATH` | Initialization failures are cached for the process lifetime. Restart the process after fixing the library path |
| TensorRT-RTX not selected, log shows a library load error (`LoadLibraryExW` code 126 on Windows) | `tensorrt_rtx_1_6` is not on `PATH` / `LD_LIBRARY_PATH`; add the SDK library directory or call `dynamically_load_tensorrt` |
| CANN never selected | By design; pass `with_rustnn_backend_hint(Backend::Cann)` |

## Building graphs

| Symptom | Cause and fix |
|---|---|
| `GraphBuilderError::GraphAlreadyBuilt` | A builder compiles exactly one graph; create a new `MLGraphBuilder` for the next one |
| `GraphBuilderError::WrongConstantSize` | The byte size of the slice does not match the descriptor; check the element count and the data type (4-bit types pack two values per byte) |
| `GraphBuilderError::RequestedInputAsOutput` / `RequestedConstantAsOutput` | Outputs must be operation results; pass the result of `identity` if an input must be echoed |
| `ShapeInferenceError::InconsistentDataTypes` or `BroadcastError` | The inputs of a binary operation differ in data type or are not broadcast-compatible; insert `cast`, `reshape` or `expand` |
| `GraphError::DynamicInputsFeatureDisabled` | The graph file declares a dynamic dimension; build with `--features dynamic-inputs` |
| `Failed to convert graph: ... Unsupported operation: <name>` | The selected backend's converter has no lowering for that operation; see the [operator support report](../development/backend-operator-support.md) and pick another backend or decompose the operation |
| `Error::GraphBuildError` from CoreML, LiteRT or TensorRT with a backend message | The backend rejected the converted model. `RUSTNN_ONNX_DUMP_DIR=<dir>` dumps the ONNX model, `TRTX_JSON_DUMP_PATH` the TensorRT layers; run the WPT case for the operation (`make test-wpt-op OP=<name>`) to compare |

## Dispatch and tensors

| Symptom | Cause and fix |
|---|---|
| ``missing runtime input tensor `x` `` / ``unexpected runtime output tensor `y` `` | The binding map lacks a name declared with `MLGraphBuilder::input` or in the outputs passed to `build`, or contains an extra one; names are the binding keys |
| ``runtime input tensor `x` rank mismatch`` / ``dimension 1 mismatch (expected 2, got 3)`` | The tensor shape differs from the graph descriptor; for dynamic graphs call `rustnn_resize_tensor` before dispatch |
| ``dynamic dimension `seq` at axis 1 exceeds maxSize`` / ``runtime dynamic dimension `seq` mismatch`` | The bound tensor is larger than the declared `maxSize`, or two tensors sharing a dimension name have different sizes |
| `operand ... uses unsupported IO data type` / `Backend ... does not support data type` | The tensor or operand data type has no kernel on the selected backend; see the [operator report](../development/backend-operator-support.md) and the relevant integration guide |
| `Error::DuplicateTensorBinding` | One `MLTensor` appears under two names or as both input and output; use distinct tensors |
| `Error::WrongWriteSize` / `WrongReadSize` | The host buffer must hold exactly `tensor.rustnn_required_bytes()` bytes |
| `Error::WriteToNonWritableTensor` / `ReadToNonReadableTensor` | Create the tensor with `to_writable()` or `to_readable()` |
| `Error::TensorCapacityError` | `rustnn_set_tensor_capacity` requested less storage than the active shape. Resize it first; rejection leaves shape and values unchanged on every backend |
| Wrong results after editing a converter (TensorRT) | A cached engine from before the edit is unlikely (the cache key includes the converter sources) but possible for local uncommitted files; delete `<cache_dir>/rustnn/trtx` |

## Tests and tooling

| Symptom | Cause and fix |
|---|---|
| WPT: `No WPT backends available` (exit code 2) | A backend feature is compiled in but none of its runtimes could start; check the context-creation rows. Without any backend feature the harness exits successfully with a notice |
| WPT: `no WPT conformance cases loaded` or `Node.js is required` | Install Node.js and run `make fetch-wpt`, or set `WPT_DIR` |
| WPT: warning that the checkout does not match `WPT_REVISION` | Run `make fetch-wpt`; `WPT_STRICT_REVISION=1` turns the warning into an error |
| `insta` reports a missing or changed snapshot | A trial changed status; review the `.snap.new` files and run `make wpt-sync-<backend>` to regenerate baselines |
| `make docs-api` fails on `flatc` | The rustdoc feature set includes `litert-runtime`; install `flatc` (see [LiteRT](../integration/litert.md)) or build the docs without that feature |
| `make docs-backend-ops-check` fails | A converter's operator support changed; run `make docs-backend-ops` and commit the report |
| CI job fails with `detected conflict: 'bin/cargo-clippy'` from rustup | rustup was adding components from `rust-toolchain.toml` at the first cargo call; the workflows install them in the toolchain step, keep that when adding jobs |
| Windows: `Access is denied` from `cargo fmt` or the linker | A test binary is still running; wait for it |

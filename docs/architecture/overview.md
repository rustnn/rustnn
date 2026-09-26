# Architecture

rustnn has three layers on top of one graph model. The WebNN API records graphs and executes
them on a backend; converters lower the graph model to a backend format; the legacy pipeline
loads, validates and converts stored graphs without a context.

```
                 WebNN API (rustnn::mlcontext, rustnn::mlgraphbuilder)
   MLContext::create -> MLGraphBuilder -> build() -> MLGraph -> dispatch(MLTensor bindings)
          |                   |
          | selects           | records
          v                   v
   backend_selection      GraphInfo  <----  loader (.webnn / JSON)  <----  onnx2webnn, webnn-graph
   (hints -> device)      operands, Operation enum, constants
          |                   |
          v                   v
   backends::{ort, trtx, coreml, litert, cann}   uses   converters::{onnx, coreml_mlprogram, trtx, litert, cann}
   MLBackendContext / MLBackendBuilder          ----->  GraphConverter
          |
          v
   ONNX Runtime | TensorRT-RTX | CoreML | LiteRT | HiAI
```

## Data flow

1. **Context creation.** `MLContext::create` resolves the WebNN hints and the rustnn hints to a
   `BackendDevice` (`src/backend_selection.rs`) and instantiates the backend context, which owns
   the device handles and the tensors.
2. **Recording.** `MLGraphBuilder` appends operands and `Operation` variants to a `GraphInfo`.
   Every method runs shape inference (`src/shape_inference.rs`) for its outputs, so the graph is
   fully annotated at all times. Constants are stored as bytes in the graph.
3. **Build.** `build` marks the outputs and hands the `GraphInfo` to the backend builder, which
   calls the backend's converter and compiles the result once: an ONNX Runtime session, a
   TensorRT engine with refittable weights, a compiled CoreML model, a LiteRT interpreter or a
   HiAI model. The `MLGraph` keeps the compiled artifact and the named I/O descriptors.
4. **Dispatch.** `dispatch` validates the tensor bindings (unique tensors, names, shapes, data
   types) with `runtime_checks`, then the backend binds or copies the tensors and runs.

## Modules

| Path | Responsibility |
|---|---|
| `src/lib.rs` | Crate docs (features, environment variables), module list, re-exports of the legacy API |
| `src/mlcontext.rs` | `MLContext`, `MLGraph`, `MLTensor`, `MLOperand`, descriptors, the crate-private backend traits |
| `src/mlcontextoptions.rs` | `MLContextOptions`, `MLPowerPreference`, `RustNNOptions`, `TrtxOptions` |
| `src/backend_selection.rs` | `Backend`, `BackendDevice`, `DeviceType` and the selection order |
| `src/mlgraphbuilder.rs` | `MLGraphBuilder`: inputs, constants, all operation methods (mostly macro-generated), build, save |
| `src/operators.rs` | `Operation` enum with one variant per WebNN operation, `op_type()`, JSON attribute parsing |
| `src/operator_options.rs` | `ML*Options` structs mirroring the spec dictionaries, `MLDimension` |
| `src/operator_enums.rs` | `MLOperandDataType` and the other spec enums |
| `src/shape_inference.rs` | Output shape and data type rules per operation |
| `src/graph.rs` | `GraphInfo`, `Operand`, `OperandDescriptor`, `Dimension`, `DataType`, 4-bit packing, hashing for caches |
| `src/validator.rs` | `GraphValidator`: structural checks, I/O descriptor maps, dependency order |
| `src/runtime_checks.rs` | Shape checks of tensor bindings at dispatch, including dynamic dimensions |
| `src/loader.rs`, `src/webnn_json.rs`, `src/webnn_save.rs` | `.webnn` text and JSON import through the `webnn-graph` crate, export, `.safetensors` weights |
| `src/converters/` | `GraphConverter` trait, `ConverterRegistry`, one converter per format (`onnx.rs`, `coreml_mlprogram.rs`, `trtx.rs` with `trtx_gru.rs`, `trtx_lstm.rs` and `trtx_rnn.rs`, `litert.rs`, `cann.rs`, `webnn.rs` for the browser) and shared helpers (`pool2d_shared.rs`, `weight_file_builder.rs`) |
| `src/backends/` | One module per backend implementing the backend traits; `caching.rs` for on-disk caches; `webnn/` with generated browser bindings; `mod.rs` with `DisabledContext` aliases for backends that are compiled out |
| `src/executors/` | Legacy one-shot execution of converted bytes (ONNX Runtime, TensorRT, CoreML), used by the CLI |
| `src/protos.rs`, `build.rs`, `protos/` | Protobuf (ONNX, CoreML) and flatbuffer (TFLite) schemas compiled at build time |
| `src/graphviz.rs`, `src/debug.rs`, `src/tensor.rs` | DOT export, `RUSTNN_DEBUG` helpers, host tensor helpers |
| `src/main.rs` | The `rustnn` CLI |
| `tests/run_wpt_conformance.rs`, `tests/wpt_conformance/` | WPT conformance harness, see the [WPT Conformance Guide](../testing/wpt-test-guide.md) |
| `tests/test_*_execution.rs` | Backend integration tests (TensorRT, LiteRT, CANN on device) |
| `scripts/` | WPT corpus fetch and report tooling, the operator report generator, git hooks |

## Backend contract

A backend implements two crate-private traits from `src/mlcontext.rs`:

- `MLBackendContext`: `create_tensor`, `read_tensor`, `write_tensor`, `dispatch`, tensor
  capacity and resize, and `create_builder`.
- `MLBackendBuilder`: `build(GraphInfo) -> MLGraph`.

It also implements `ListDevices::list_devices()` for the selection code. Compiled artifacts are
stored in the `MLBackendGraph` enum. When a backend's feature is off, `backends/mod.rs` aliases
its context type to `DisabledContext`, so `MLContext` and the selection code compile under every
feature combination. The traits are crate-private on purpose: external backends are not
supported yet.

## Graph model

`GraphInfo` is the single backend-agnostic representation. Operands are addressed by index;
operations are variants of `Operation` with named fields for their operand indices and an
`Option<ML*Options>`. This replaced a string-typed `op_type` plus JSON attributes design:
attribute names are checked at compile time and every converter matches on the enum. The same
model round-trips through the `webnn-graph` crate's text and JSON formats, which is how
onnx2webnn hands models to rustnn.

Dynamic dimensions (`Dimension::Dynamic { name, max_size }`) are part of the model but only
accepted when the `dynamic-inputs` feature is enabled.

## Legacy pipeline

`load_graph_from_path` -> `GraphValidator` -> `ConverterRegistry::convert` -> `executors::*`
runs a stored graph without an `MLContext`. It predates the WebNN API and reloads the converted
model on every call. The CLI and two examples still use it; the converters are shared with the
backends.

## Design decisions

| Decision | Reason |
|---|---|
| Backend selection at context creation from hints | Follows the WebNN device selection explainer; the same graph code runs on every backend |
| Strongly typed `Operation` and `ML*Options` | Compile-time checking of operand wiring and attribute names across five converters |
| `BTreeMap` for `MLNamedOperands` and `MLNamedTensors` | Deterministic iteration order for the spec's record types |
| `Send + Sync` contexts and errors | Embeddings such as Servo dispatch from several threads; errors compose with `anyhow` |
| Native lowering for TensorRT, CoreML, LiteRT and CANN instead of going through ONNX | Avoids a second lowering and exposes backend features such as weight refit and caching |
| Refittable weights and a topology-keyed engine cache for TensorRT | Engine builds are expensive; weights change more often than topology |
| Feature flags per backend, mock features for TensorRT and CANN | Keeps the default build dependency-free and lets CI type-check every backend |
| Protobuf and flatbuffer codegen at build time | No checked-in generated code |
| Live WPT corpus as the conformance oracle, snapshots and expected-failure lists per backend | Upstream tests define the semantics; regressions show up as snapshot diffs |
| Source-generated operator support report | Documentation that cannot drift from the converters |

## Platform support

| Capability | Platforms |
|---|---|
| Validation, shape inference, conversion to ONNX and CoreML | Linux, macOS, Windows, wasm32 |
| ONNX Runtime execution | Linux, macOS, Windows |
| TensorRT-RTX execution | Linux and Windows with NVIDIA RTX GPUs |
| CoreML execution | macOS |
| LiteRT execution | Linux, macOS |
| CANN execution | OpenHarmony (aarch64) |
| Browser WebNN | wasm32 (bindings only) |

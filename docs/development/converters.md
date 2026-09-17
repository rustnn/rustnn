# Converter Internals

A converter lowers a validated `GraphInfo` to one backend format. This page describes the
contract, the structure the five converters share, the backend-specific rules that shaped them
and how to debug a wrong model. Read it before touching `src/converters/`.

## Contract

```rust
pub trait GraphConverter {
    fn format(&self) -> &'static str;                                   // registry key, e.g. "onnx"
    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError>;
}
```

`ConvertedGraph` carries the serialized model (`data`, `content_type`) and an optional
`weights_data` sidecar. `ConverterRegistry::with_defaults` registers every converter compiled
into the build; the CLI `--convert` option, the legacy executors and the `MLGraphBuilder::build`
path of each backend all go through the same `convert`.

Converters are pure: they read `GraphInfo` (operands with `OperandDescriptor`, the `Operation`
enum, constant bytes) and produce bytes. They must not touch a device. Anything that needs the
runtime (compiling, transposing runtime tensors, refitting weights) belongs to
`src/backends/<name>.rs`.

| Converter | File | Output | Registered when |
|---|---|---|---|
| `OnnxConverter` | `onnx.rs` | ONNX `ModelProto` (prost) | always |
| `CoremlMlProgramConverter` | `coreml_mlprogram.rs` | CoreML MLProgram (`protos::coreml`) | always |
| `TrtxConverter` | `trtx.rs`, `trtx_gru.rs`, `trtx_lstm.rs`, `trtx_rnn.rs` | TensorRT network built through the `trtx` crate, serialized engine | `trtx-runtime`, `trtx-runtime-mock` |
| `LiteRtConverter` | `litert.rs` | TFLite flatbuffer (`flatc`-generated schema) | `litert-runtime` |
| `CannConverter` | `cann.rs` | HiAI offline model | `cann-runtime`, `cann-runtime-mock` |
| browser | `webnn.rs` | replays the graph through the browser's `MLGraphBuilder` (`convert_async`, not a `GraphConverter`) | `webnn-runtime` on wasm32 |

Shared helpers: `pool2d_shared.rs` (window and padding arithmetic used by several backends) and
`weight_file_builder.rs` (the CoreML weight blob).

## Structure of a converter

Every converter follows the same passes:

1. **Scan.** Collect operand shapes and data types, decide which constants become initializers
   and which need widening or a different layout, and detect graph-level facts (unused
   constants, constants that only feed casts, operands that need NHWC).
2. **Boundary.** Declare inputs and outputs with the backend's types. This is where rank-0
   operands become `[1]` (CoreML), booleans become `uint8` (all backends), and `int64` or
   4-bit types are widened where the backend has no kernel.
3. **Lower.** One arm per `Operation` variant. Operands are addressed by index
   (`Operation::inputs()` for positional operands, `option_operands()` for `MLOperand`-valued
   options such as `gemm.c` or normalization `scale`/`bias`; `all_input_operands()` for both).
   Options come from the typed `ML*Options` structs through `OperatorOptions::as_*()`.
4. **Emit.** Serialize, and return the sidecar when weights are external.

Rules that hold for every converter:

- Do not change `Operation::inputs()` positions; hundreds of call sites in the ONNX and CoreML
  converters depend on them. Use `all_input_operands()` when the question is "what consumes this
  operand".
- Options are distinct types per operation (`MLSqueezeOptions` is not `MLUnsqueezeOptions`), so
  or-patterns across variants do not compile; write one arm each.
- Names in the emitted model come from `operand_name` (the WebNN name or `operand_<id>`);
  inserted helper nodes use a counter, and the node name and its output name must share the
  counter value.
- Never emit identity casts. ONNX Runtime's mandatory fp16 pass mis-types an `f16 -> f16` cast
  feeding `Where` and rejects the model.

## Backend rules that shaped the lowerings

### ONNX

- Two export generations coexist: attribute-based helpers and the opset-13+ path that passes
  axes and shapes as input tensors. A change to an operation's axes handling must cover both.
- `Squeeze` without `axes` removes every size-1 dimension; always emit the axes.
- Int4 exports as INT32 and Uint4 as UINT8 with unpacked elements, because ONNX Runtime does not
  accept packed 4-bit inputs on the path rustnn uses. Packed export at opset 21 is an open item.
- Recurrent operations: the initial state is duplicated from `[1,B,H]` to `[2,B,H]` only when
  direction is `both` and the first dimension is 1; outputs are wired by name with a positional
  fallback in spec order.
- Large initializers go to `rustnn_external_weights.data` (`ONNX_EXTERNAL_WEIGHTS_FILENAME`).

### TensorRT-RTX

- `uint8` is legal only at network I/O or through a cast to or from float; internal `uint8`
  results stay `int32` and become `uint8` at the outputs. Boolean masks are stored as `int8`.
- `IDequantizeLayer` cannot take `uint8`; unsigned `dequantizeLinear` uses the manual
  multiply path.
- Integer broadcast uses a stride-0 slice (as `expand` does); `IResizeLayer` rejects integers.
- Constants are refittable and set after the build; constants that TensorRT folds away (unused,
  or feeding only casts) are excluded from refit (`unused_constant_operand_ids`,
  `constant_only_feeds_casts`). Refit failures raise `TrtxError::ConstantRefitFailed` with the
  operand and its consumers.
- The engine cache key hashes the converter sources (`build.rs`, `SOURCE_HASH`), so converter
  edits invalidate cached engines; changes to builder flags do not. Delete the cache directories
  when in doubt.
- The nearest-neighbour rounding of `resample2d` is `round_prefer_floor` (`kHALF_DOWN`).

### CoreML

- MIL has no rank-0 tensors at the boundary, no dilation in pooling, no `edge`/`reflection`
  padding above two dimensions and no tensors of rank 6 and above; integer arithmetic runs in
  float32.
- Comparison results are `uint8`; `reduceLogSumExp` uses the max-shifted form; reductions with
  empty `axes` and `resample2d` on arbitrary axes are lowered explicitly.
- Float16 weights go to the weight blob written by `weight_file_builder.rs` and returned as
  `weights_data`.

### LiteRT

- Spatial operations are NHWC. `is_spatial_op` marks operands whose layout changes; filters are
  transposed to OHWI and boundary tensors are transposed by the backend at run time.
- Operations without a kernel are rejected in `backends::litert::unsupported_ops`, data types
  without a kernel in `dtype_unsupported_for_op`. The WPT harness uses the same functions to
  skip trials.

### CANN

- `is_supported_op` is the allow-list; everything else fails at build with an explicit error.

## Operator support report

`scripts/generate_backend_operator_report.py` derives the
[operator support matrix](backend-operator-support.md) from the sources: an operation counts as
supported when the ONNX or LiteRT converter references its `Operation::<Variant>`, when the
CoreML converter references the variant or compares against its lower-cased name, when the
TensorRT converter has the name as a `match op_type` key, or when CANN lists the variant in
`is_supported_op`. CI fails on drift; run `make docs-backend-ops` after every converter change.
If you add a new dispatch style, extend the detection rule and its tests in the script.

## Tests

Converter tests live in `mod tests` at the end of each file. They build a `GraphInfo` by hand,
call `convert`, decode the output (`ModelProto::decode`, the CoreML protobuf, the flatbuffer
root) and assert on structure: op types, input counts, initializer data types and dimensions,
absence of helper nodes. Name the test after the guarantee it protects
(`test_squeeze_with_axes_emits_axes_input`). The WPT corpus is the numerical oracle:
`make test-wpt-op OP=<name>` with `WPT_BACKEND` set, then `make wpt-sync-<backend>` and review
the snapshot diff.

## Debugging a wrong model

The signature of a converter bug: rustnn's shape inference is right, the emitted model is not,
and the error surfaces downstream, as a type error at load time or an out-of-bounds axis on the
next operation. Compare the emitted node with the backend's semantics before touching shape
inference.

| Backend | Dump | Inspect |
|---|---|---|
| ONNX Runtime | `RUSTNN_DEBUG=2 RUSTNN_DEBUG_ONNX_DIR=<dir>` writes every model the backend builds | `python -c "import onnx; print(onnx.load('<file>'))"` or Netron |
| TensorRT-RTX | `TRTX_JSON_DUMP_PATH=<dir>` writes the layer graph of each engine; `RUSTNN_TRTX_LOG_VERBOSITY=verbose` for the builder log | look for the layer that consumes the operand; Myelin-fused layers print as `__myl_...` |
| CoreML | `--coreml-compiled-output <dir>` in the CLI keeps the `.mlmodelc` | `coremltools` or Xcode |
| LiteRT | `--convert litert --convert-output model.tflite` | `flatc --json` with `protos/tflite/schema.fbs`, or the TFLite visualizer |
| any | `--export-dot graph.dot` shows the graph rustnn recorded | Graphviz |

Then write the regression test first, fix the lowering, and run the WPT cases for the operation
on every backend you have.

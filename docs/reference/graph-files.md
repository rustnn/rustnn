# Graph Files and Weights

rustnn reads and writes the graph interchange formats of the
[webnn-graph](https://github.com/rustnn/webnn-graph) crate. Both formats describe the same
model: named inputs with data type and shape, constants with an initializer, a list of
operation nodes and named outputs. `load_graph_from_path` accepts either by file extension.

## `.webnn` text

```
webnn_graph "sample_graph" v1 {
  inputs {
    lhs: f32[2, 2];
  }

  consts {
    rhs: f32[2, 2] @scalar(1.0);
  }

  nodes {
    sum = add(lhs, rhs);
  }

  outputs { sum; }
}
```

- Data types use the short spellings `f32`, `f16`, `i32`, `u32`, `i64`, `u64`, `i8`, `u8`,
  `i4`, `u4`.
- Dynamic dimensions are written as `dyn("name", maxSize)` inside the shape, for example
  `f32[dyn("batch", 8), 128]` (the JSON form is `{ "name": "batch", "maxSize": 8 }`).
  Only inputs may be dynamic; loading them requires the `dynamic-inputs` feature.
- Constant initializers: `@scalar(v)` fills every element, `@bytes(...)` lists the raw bytes of
  a small tensor, and `@weights("tensor-name")` refers to data stored next to the graph (see
  below).
- Node options use the spec's camelCase names, for example `conv2d(x, w, strides=[2, 2],
  padding=[1, 1, 1, 1])`.
- Identifiers exported by ONNX tools are sanitized on import (`sanitize_webnn_identifiers` in
  `src/loader.rs`): `.` becomes `_` in declarations and `%` references
  (`embeddings.LayerNorm.bias` to `embeddings_LayerNorm_bias`) and `::` becomes `__`
  (`onnx::MatMul_0` to `onnx__MatMul_0`). The weight resolver applies the same mapping when it
  looks up a `@weights` name in the sidecar, so unsanitized tensor names keep working.

## JSON

The same graph as `webnn-graph-json`:

```json
{
  "format": "webnn-graph-json",
  "version": 1,
  "inputs": { "lhs": { "dataType": "float32", "shape": [2, 2] } },
  "consts": {
    "rhs": { "dataType": "float32", "shape": [2, 2],
             "init": { "kind": "inlineBytes", "bytes": [0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64] } }
  },
  "nodes": [
    { "id": "add_lhs_rhs", "op": "add", "inputs": ["lhs", "rhs"], "options": {}, "outputs": ["sum"] }
  ],
  "outputs": { "sum": "sum" }
}
```

`src/webnn_json.rs` converts between this model and `GraphInfo` in both directions
(`from_graph_json`, `to_graph_json`); operation-level arguments such as `axis`, `newShape` or
`beginningPadding` are stored in `options` alongside the dictionary fields and separated on
import (`OperationExtras`). The loader runs shape inference after import, so a graph without
intermediate shapes is completed automatically.

## External weights

Two layouts are resolved by the loader relative to the graph file:

| Layout | Files | Written by |
|---|---|---|
| Manifest plus blob | `manifest.json` (`wg-weights-manifest`: per tensor `dataType`, `shape`, `byteOffset`, `byteLength`, `layout`) and `model.weights` | onnx2webnn and the webnn-graph tooling |
| safetensors | `<graph stem>.safetensors`, one tensor per `@weights` name | `MLGraphBuilder::rustnn_save_webnn` |

Bytes are little-endian in both. 4-bit types are nibble-packed and cannot be stored in
safetensors, so graphs with `int4` or `uint4` constants can only use the manifest layout or
inline bytes.

## Saving from the builder

```rust
builder.rustnn_save_webnn(&outputs, "model.webnn")?;   // writes model.webnn and model.safetensors
```

The exporter references every constant as `@weights(<name>)`, so the text file stays small and
the pair round-trips through `load_graph_from_path`. `rustnn_webnn_text_for_outputs` returns the
same text without writing files, which is useful for debugging a graph under construction.

## Exporting to backend formats

`ConverterRegistry::with_defaults().convert(format, &graph_info)` produces:

| Format | Content | Sidecar |
|---|---|---|
| `onnx` | ONNX `ModelProto` bytes | `rustnn_external_weights.data` next to the model when initializers are large (`ONNX_EXTERNAL_WEIGHTS_FILENAME`) |
| `coreml` | CoreML MLProgram `.mlmodel` | weights blob for float16 models |
| `trtx` | serialized TensorRT engine (`trtx-runtime`) | - |
| `litert` | TFLite flatbuffer (`litert-runtime`) | - |
| `cann` | HiAI offline model (`cann-runtime`) | - |

The CLI exposes the same through `--convert <format> --convert-output <path>` and writes the
sidecar automatically.

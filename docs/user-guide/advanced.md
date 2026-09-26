# Advanced Topics

## Backend hints and options

`MLContextOptions` carries the two WebNN hints and three rustnn extensions:

```rust
use rustnn::mlcontext::{
    Backend, BackendDevice, MLContext, MLContextOptions, MLPowerPreference, RustNNOptions,
};

// Restrict selection to one backend; fails if it cannot serve the hints.
let options = MLContextOptions::new(MLPowerPreference::HighPerformance, true)
    .with_rustnn_backend_hint(Backend::Trtx);

// Use exactly this device: no availability check, no fallback.
let options = MLContextOptions::new(MLPowerPreference::Default, true)
    .with_rustnn_device_hint(BackendDevice::Trtx { cuda_device_idx: 1 });

// Backend tuning. The option structs are `#[non_exhaustive]`: start from `Default` and set fields.
let mut tuning = RustNNOptions::default();
tuning.trtx.cuda_graphs = false;
tuning.trtx.fail_on_cache_miss = true;   // ahead-of-time flows: never build, only load cached engines
let options = MLContextOptions::new(MLPowerPreference::Default, true).with_rustnn_options(tuning);
let context = MLContext::create(&options)?;
```

`TrtxOptions` has `engine_caching`, `runtime_cache`, `fail_on_cache_miss` and `cuda_graphs`;
the ONNX Runtime, CoreML and LiteRT option structs exist but have no fields yet.

## Dynamic shapes

Dynamic dimensions are opt-in through the `dynamic-inputs` Cargo feature. Without it, loading
a graph with a dynamic dimension fails with `GraphError::DynamicInputsFeatureDisabled`.

- In graph files a dynamic dimension is `{ "name": "seq", "maxSize": 4096 }` (JSON) or the
  equivalent `.webnn` text; onnx2webnn exports produce them for symbolic ONNX dimensions. In the
  graph model this is `Dimension::Dynamic(DynamicDimension { name, max_size })`. Constants must
  stay static.
- Builder descriptors (`MLOperandDescriptor`) are static. Graphs with dynamic inputs come from
  files and are compiled with `MLGraphBuilder::build_graph_info`.
- Tensors bound to dynamic inputs or outputs are created with a starting shape, given a
  capacity for the largest shape they will take, and resized before each dispatch:

```rust
let mut mask = context.create_tensor(
    &MLTensorDescriptor::new(MLOperandDataType::Int64, vec![1, 1]).to_writable(),
)?;
context.rustnn_set_tensor_capacity(&mut mask, &[1, 4096])?;

for step in 1..=steps {
    context.rustnn_resize_tensor(&mut mask, &[1, step])?;   // active shape, within capacity
    context.write_tensor(&mask, &mask_values[..step as usize])?;
    context.dispatch(&mut graph, &inputs, &outputs)?;
}
```

`dispatch` checks the active shapes against the graph's dimension bounds and requires equal
values for dynamic dimensions that share a name. `examples/smollm_mlcontext.rs` runs a KV
cache this way. The checked legacy executors (`run_onnx_with_inputs_checked` and friends)
apply the same rules to one-shot runs; see [Flexible Input Shapes](../development/flexible-input-shapes.md).

## Saving and exporting graphs

| Goal | Call |
|---|---|
| Save the graph under construction as `.webnn` text plus `.safetensors` weights | `builder.rustnn_save_webnn(&outputs, "model.webnn")` |
| Inspect the graph so far as `.webnn` text | `builder.rustnn_webnn_text_for_outputs(&outputs)` |
| Get the finished `GraphInfo` without compiling | `builder.finish_graph_info(&outputs)` |
| Convert a `GraphInfo` to ONNX, CoreML, TensorRT engine, TFLite or CANN bytes | `ConverterRegistry::with_defaults().convert("onnx", &graph_info)` |
| Write the ONNX sidecar for large models | `converted.weights_data` into `ONNX_EXTERNAL_WEIGHTS_FILENAME` next to the model |
| Graphviz | `rustnn::graph_to_dot(&graph_info)` or the CLI `--export-dot` |

`MLGraphBuilder::new_uncompiled()` records without a backend, so conversion tools need no
runtime feature. `Int4` and `Uint4` constants cannot be written to `.safetensors`.

## Loading external weights

The loader resolves `@weights(...)` references through the `webnn-graph` crate. It looks next
to the graph file for a `manifest.json` plus `model.weights` pair (onnx2webnn layout) or for the
`.safetensors` file written by `rustnn_save_webnn`. Identifiers from ONNX exports are sanitized
on import: `.` becomes `_` and `::` becomes `__`, in declarations, references and weight lookups
alike (details in [Graph Files and Weights](../reference/graph-files.md)). Shape inference runs
on import, so a loaded graph carries complete descriptors.

## Caching

The TensorRT-RTX backend caches two things under the platform cache directory
(`~/.cache/rustnn/` on Linux, `~/Library/Caches/rustnn/` on macOS, `%LOCALAPPDATA%\rustnn\` on
Windows):

| Category | Content | Key |
|---|---|---|
| `trtx` | Serialized engines with stripped, refittable weights | Hash of the graph topology, non-refittable constants and the converter sources |
| `trtx-jit` | The TensorRT runtime (JIT kernel) cache shared by all engines | Global |

Because weights are refitted after loading, a cached engine serves every model with the same
topology. Entries are zstd compressed and written atomically. Disable caching with
`TrtxOptions::engine_caching` and `TrtxOptions::runtime_cache`; delete the directories to
start cold. Details are in [TensorRT-RTX](../integration/tensorrt.md).

## Debugging

| Setting | Effect |
|---|---|
| `RUST_LOG=info` (or `debug`, `trace`) | Library logging through the `log` crate: selected device, cache hits, per-dispatch shapes. Programs need a logger such as `pretty_env_logger` |
| `RUSTNN_DEBUG=1` | Enables the `debug_print!` output of converters |
| `RUSTNN_DEBUG=2` with `RUSTNN_DEBUG_ONNX_DIR=<dir>` | Also writes the converted ONNX model of every build for inspection in Netron or ONNX Runtime |
| `RUSTNN_TRTX_LOG_VERBOSITY=verbose` | TensorRT logger level (`internal_error`, `error`, `warning`, `info`, `verbose`) |
| `TRTX_JSON_DUMP_PATH=<dir>` | Writes TensorRT engine layer JSON per built engine |
| `builder.rustnn_operand_shape(op)` | Shape of an operand while recording |
| `cargo run --features onnx-runtime -- graph.webnn --export-dot graph.dot` | Graph structure as Graphviz |
| `make test-wpt-op OP=<operation>` | Runs the WPT conformance cases of one operation, printing expected and actual values on failure |

## Threads

`MLContext` is `Send + Sync`. `dispatch`, `write_tensor` and `read_tensor` take `&mut self`,
so concurrent use goes through a `Mutex<MLContext>`. Concurrent use of several contexts is not
validated; the WPT harness runs single-threaded (`--test-threads 1`). A builder borrows the
context mutably until `build`, so record graphs before sharing the context.

## Precision notes

- TensorRT-RTX keeps float32 math at full precision (TF32 disabled) and runs float16 graphs in
  float16.
- CoreML computes integer operations in float32; values near the int32 and int64 limits lose
  precision. Rank is limited to 5.
- LiteRT rejects some data type and operation combinations at build time; the
  [operator support report](../development/backend-operator-support.md) and the WPT dashboard
  show what runs, and the [LiteRT page](../integration/litert.md) describes the policy.
- The WPT tolerances applied per operation are in `tests/wpt_conformance/tolerance.rs`; the
  audit mode described in the [WPT Conformance Guide](../testing/wpt-test-guide.md) reports how
  much of the tolerance each passing case uses.

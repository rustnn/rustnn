# Flexible Input Shapes

This page describes how dynamic dimensions are encoded, validated and used at run time.
Converting dynamic graphs to ONNX or CoreML requires the `dynamic-inputs` Cargo feature;
without it these converters return `GraphError::DynamicInputsFeatureDisabled`. The graph
model and builder can still record dynamic dimensions.

## Encoding

A dimension is either a number (static) or an object with `name` and `maxSize` (dynamic):

```json
{
  "inputs": {
    "x": {
      "dataType": "float32",
      "shape": [ { "name": "batch", "maxSize": 16 }, 128 ]
    }
  }
}
```

In Rust this is `Dimension::Dynamic(DynamicDimension { name, max_size })` in `src/graph.rs`
(`MLDimension` in the options structs). Dynamic dimensions are allowed on inputs, outputs and
intermediate operands; constants must be static. The `.webnn` text format and onnx2webnn
exports use the same model. Shape inference carries dynamic dimensions through the operations
whose rules are defined for them and falls back to the maximum size elsewhere
(`Dimension::get_static_or_max_size`).

## Runtime validation

The [runtime-check module](https://rustnn.github.io/rustnn/api/rustnn/runtime_checks/)
owns the rank, static-dimension, dynamic-bound and shared-name contracts. See
[MLContext::dispatch](https://rustnn.github.io/rustnn/api/rustnn/mlcontext/struct.MLContext.html#method.dispatch)
for binding requirements.

## MLContext API

Tensors bound to dynamic operands are allocated once with a capacity and resized per dispatch:

- `MLContext::rustnn_set_tensor_capacity(&mut tensor, &max_shape)` reserves storage for the
  largest shape the tensor will take.
- `MLContext::rustnn_resize_tensor(&mut tensor, &shape)` sets the active shape. Staying within reserved
  capacity avoids reallocation; supported backends may grow storage beyond it. See the
  [method contract](https://rustnn.github.io/rustnn/api/rustnn/mlcontext/struct.MLContext.html#method.rustnn_resize_tensor).

Backends receive the active shape for each dispatch. `examples/smollm_mlcontext.rs` uses this
for a growing KV cache: the past key and value tensors start at length 0, get the maximum cache
length as capacity, and are resized every token.

## Legacy executors

`run_onnx_with_inputs_checked` and `run_coreml_with_inputs_checked` take the input and output
descriptor maps from `GraphInfo::io_binding_maps()` or `MLGraph` and apply the same checks to a one-shot run;
`run_onnx_with_inputs` skips them.

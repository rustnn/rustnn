# Flexible Input Shapes

This page describes how dynamic dimensions are encoded, validated and used at run time.
Dynamic dimensions require the `dynamic-inputs` Cargo feature; without it the validator rejects
such graphs with `GraphError::DynamicInputsFeatureDisabled`.

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

`MLContext::dispatch` and the checked legacy executors enforce, through `src/runtime_checks.rs`:

1. The bound tensor has the descriptor's rank and matches every static dimension.
2. The actual value of a dynamic dimension does not exceed `maxSize`.
3. Dynamic dimensions with the same `name` have the same value across all validated inputs and
   outputs.
4. The tensor byte length matches the active shape.

## MLContext API

Tensors bound to dynamic operands are allocated once with a capacity and resized per dispatch:

- `MLContext::rustnn_set_tensor_capacity(&mut tensor, &max_shape)` reserves storage for the
  largest shape the tensor will take.
- `MLContext::rustnn_resize_tensor(&mut tensor, &shape)` sets the active shape without
  reallocating; it must fit the capacity.

Backends receive the active shape for each dispatch. `examples/smollm_mlcontext.rs` uses this
for a growing KV cache: the past key and value tensors start at length 0, get the maximum cache
length as capacity, and are resized every token.

## Legacy executors

`run_onnx_with_inputs_checked` and `run_coreml_with_inputs_checked` take the input and output
descriptor maps from `ValidationArtifacts` and apply the same checks to a one-shot run;
`run_onnx_with_inputs` skips them.

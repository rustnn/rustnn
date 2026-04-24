# Flexible Input Shapes

This document describes how dynamic dimensions are encoded and validated at runtime in `rustnn`.

## JSON Encoding

Tensor dimensions use either:

- Static dimension: a number
- Dynamic dimension: an object with `name` and `maxSize`

Example:

```json
{
  "inputs": {
    "x": {
      "dataType": "float32",
      "shape": [
        { "name": "batch", "maxSize": 16 },
        128
      ]
    }
  }
}
```

Rules:

- Dynamic dimensions are allowed on input/output operands.
- Constant operands must remain concrete (static shape only).

## Runtime Validation

Phase 5 runtime checks enforce:

1. Actual tensor rank and static dimensions match descriptor.
2. Actual dynamic dimension values do not exceed `maxSize`.
3. Dynamic dimensions with the same `name` must match across validated inputs/outputs.
4. Tensor data length must match the runtime shape element count.

## Checked Execution APIs

Pass operand descriptor maps when runtime validation against the WebNN graph is required:

- ONNX: `run_onnx_with_inputs_checked(..., &input_descriptors, &output_descriptors)` passes the WebNN I/O descriptor maps from validation artifacts; ORT still validates feeds independently. Use `run_onnx_with_inputs` when you do not need descriptor checks.
- TensorRT: `run_trtx_with_inputs_checked(...)`
- CoreML: `run_coreml_with_inputs_checked(...)`

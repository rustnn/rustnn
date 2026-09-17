# API Overview

This page explains how the WebNN API maps to Rust in rustnn and where to find each part. Exact
signatures live in the generated [Rust API reference](https://rustnn.github.io/rustnn/api/rustnn/)
(`make docs-api` builds it locally into `target/doc/rustnn/`).

## Conventions

- JavaScript `camelCase` names become Rust `snake_case`: `reduceSum` is `reduce_sum`,
  `convTranspose2d` is `conv_transpose2d`, `where` is `where_`.
- Every operation has two methods: `op(...)` with the required operands and arguments, and
  `op_with_options(..., options)` taking the matching options struct from
  `rustnn::operator_options` (`MLConv2dOptions`, `MLReduceOptions`, ...). Operations without
  spec options take `MLOperatorOptions`, which only carries the `label`.
- Operand fields inside option structs (`MLConv2dOptions::bias`, `MLGemmOptions::c`, the
  quantization zero points) hold operand indices: pass `operand.rustnn_index()` or
  `operand.into()`.
- Methods that are not part of the WebNN specification carry the `rustnn_` prefix.
- The API is synchronous. `dispatch`, `read_tensor` and `write_tensor` return when the work is
  done.
- Results are `rustnn::error::Result<T>`, an alias for `Result<T, rustnn::error::Error>`.
- Lifetimes tie the objects together: `MLGraphBuilder<'context, 'builder>` borrows the
  `MLContext<'context>` mutably while recording, and `MLGraph<'context>` cannot outlive its
  context.

## Types

| Type | Module | Role |
|---|---|---|
| `MLContext` | `rustnn::mlcontext` | Owns a backend device, its tensors and compiled graphs |
| `MLContextOptions`, `MLPowerPreference`, `RustNNOptions`, `TrtxOptions` | `rustnn::mlcontext` (defined in `mlcontextoptions`) | WebNN hints plus rustnn backend, device and tuning hints |
| `Backend`, `BackendDevice`, `DeviceType` | `rustnn::mlcontext` (defined in `backend_selection`) | Selected backend and device |
| `MLGraphBuilder` | `rustnn::mlgraphbuilder`, re-exported from `mlcontext` | Records operations, compiles an `MLGraph` |
| `MLOperand` | `rustnn::mlcontext` | Copyable handle to an operand of one builder; `rustnn_index()` is the index that option structs take |
| `MLOperandDescriptor` | `rustnn::mlcontext` | Data type and shape (`Vec<u64>`) |
| `MLGraph` | `rustnn::mlcontext` | Compiled graph with `input_descriptors` and `output_descriptors` |
| `MLTensor`, `MLTensorDescriptor` | `rustnn::mlcontext` | Backend tensor and its shape, data type, `readable` and `writable` flags |
| `MLNamedOperands`, `MLNamedTensors` | `rustnn::mlcontext` | `BTreeMap<&str, MLOperand>` and `BTreeMap<&str, &MLTensor>` |
| `MLOperandDataType` and the other `ML*` enums | `rustnn::operator_enums` | Spec enums (`MLInputOperandLayout`, `MLPaddingMode`, ...) |
| `ML*Options` | `rustnn::operator_options` | One struct per spec options dictionary |
| `Operation` | `rustnn::operators` | The recorded operation enum, one variant per operation |
| `GraphInfo`, `Operand`, `OperandDescriptor`, `Dimension`, `DataType` | `rustnn::graph` | Backend-agnostic graph model |
| `Error`, `GraphBuilderError`, `ShapeInferenceError`, `GraphError` | `rustnn::error` | Error types, all `Send + Sync` |

## MLContext

| Method | Status | Notes |
|---|---|---|
| `MLContext::create(&options)` | implemented | Selects the backend; see [Backends](backends.md) |
| `accelerated()` | implemented | Whether the selected device is not a CPU |
| `create_tensor(&descriptor)` | implemented | Flags default to neither readable nor writable |
| `write_tensor(&tensor, &[T])`, `read_tensor(&tensor, &mut [T])` | implemented | `T: bytemuck::Pod`; the byte size must equal `tensor.rustnn_required_bytes()` |
| `dispatch(&mut graph, &inputs, &outputs)` | implemented | Validates bindings, then runs |
| `rustnn_backend()`, `rustnn_device()`, `rustnn_device_type()` | extension | Inspect the selection |
| `rustnn_set_tensor_capacity(&mut tensor, max_shape)`, `rustnn_resize_tensor(&mut tensor, shape)` | extension | Dynamic shapes; see [Advanced Topics](advanced.md) |
| `create_from_gpu_device`, `lost`, `create_constant_tensor`, `destroy`, `op_support_limits` | not implemented (`todo!()`) | Tensors and contexts are released by `Drop` |

## MLGraphBuilder

| Method | Status | Notes |
|---|---|---|
| `MLGraphBuilder::new(&mut context)` | implemented | One builder compiles one graph |
| `MLGraphBuilder::new_uncompiled()` | extension | Records a graph without a backend, for saving or converting |
| `input(name, &descriptor)` | implemented | The name is the dispatch key |
| `constant_from_slice(&descriptor, &[T])`, `constant_from_vec(&descriptor, Vec<T>)`, `constant_from_bytes(&descriptor, Vec<u8>)` | implemented | Byte size must match the descriptor; `constant_from_bytes` avoids a copy for large weights |
| `constant_from_tensor`, `constant_from_value` | not implemented | Use `constant_from_slice` with an empty shape for scalars |
| `build(&outputs)` | implemented | Names the outputs and compiles; fails on an empty map, an input or constant used as output, or two names for one operand |
| `finish_graph_info(&outputs)` | extension | Returns the finished `GraphInfo` without compiling |
| `build_graph_info(graph_info)` | extension | Compiles a complete `GraphInfo`, for example one loaded from a file |
| `rustnn_save_webnn(&outputs, path)` | extension | Writes `.webnn` text plus a `.safetensors` weights file; the builder stays usable |
| `rustnn_webnn_text_for_outputs(&outputs)` | extension | The `.webnn` text of the graph so far, for debugging |
| `rustnn_operand_shape(operand)`, `rustnn_operand_data_type(operand)` | extension | Inspect operands while recording |

Shape inference runs inside every operation method; a shape or data type conflict is returned
from that call as `Error::GraphBuilderError` wrapping a `ShapeInferenceError`.

### Operations

The Rust method for each WebNN operation, grouped as in the specification. Required arguments
follow the spec order; `a`, `b` stand for the two inputs of binary operations.

| Group | Methods |
|---|---|
| Element-wise binary | `add`, `sub`, `mul`, `div`, `pow`, `max`, `min` |
| Comparison and logical (`Uint8` results) | `equal`, `greater`, `greater_or_equal`, `lesser`, `lesser_or_equal`, `not_equal`, `logical_and`, `logical_or`, `logical_xor`, `logical_not` |
| Element-wise unary | `abs`, `ceil`, `floor`, `round_even`, `neg`, `exp`, `log`, `sqrt`, `reciprocal`, `sin`, `cos`, `tan`, `erf`, `sign`, `identity`, `is_nan`, `is_infinite` |
| Activations | `relu`, `sigmoid`, `tanh`, `softmax(input, axis)`, `softplus`, `softsign`, `elu`, `leaky_relu`, `prelu(input, slope)`, `gelu`, `hard_sigmoid`, `hard_swish`, `linear`, `clamp` |
| Convolution and pooling | `conv2d(input, filter)`, `conv_transpose2d`, `average_pool2d`, `max_pool2d`, `l2_pool2d`, `global_average_pool`, `global_max_pool`, `resample2d` |
| Normalization | `batch_normalization(input, mean, variance)`, `instance_normalization`, `layer_normalization` |
| Reduction and indices | `reduce_sum`, `reduce_mean`, `reduce_max`, `reduce_min`, `reduce_product`, `reduce_l1`, `reduce_l2`, `reduce_log_sum`, `reduce_log_sum_exp`, `reduce_sum_square`, `arg_max`, `arg_min`, `cumulative_sum` |
| Shape | `reshape(input, new_shape)`, `transpose`, `expand(input, new_shape)`, `squeeze`, `unsqueeze`, `concat(&[operands], axis)`, `split(input, &splits)`, `split_equal_with_options(input, count, options)`, `slice(input, &starts, &sizes)`, `pad`, `tile`, `reverse`, `triangular`, `cast(input, data_type)`, `shape` |
| Gather and scatter | `gather(input, indices)`, `gather_elements`, `gather_nd`, `scatter_elements(input, indices, updates)`, `scatter_nd`, `where_(condition, true_value, false_value)` |
| Matrix | `matmul`, `gemm` |
| Quantization | `quantize_linear(input, scale)`, `quantize_linear_with_zeropoint(input, scale, zero_point)`, `dequantize_linear`, `dequantize_linear_with_zeropoint`, plus the `_with_options` forms |
| Recurrent | `gru_with_options`, `gru_cell_with_options`, `lstm_with_options`, `lstm_cell_with_options` (only the options forms exist). `gru_with_options` returns the hidden state plus the sequence when `return_sequence` is set; `lstm_with_options` returns hidden state, cell state and optionally the sequence; `lstm_cell_with_options` returns hidden and cell state, all as `Vec<MLOperand>` |

`globalAveragePool`, `globalMaxPool`, `squeeze` and `unsqueeze` are kept from earlier
specification drafts; `shape` is a rustnn extension emitted by onnx2webnn. The per-backend
support matrix is generated into
[Backend Operator Support](../development/backend-operator-support.md).

## Tensors

`MLTensorDescriptor` wraps an `MLOperandDescriptor` and two flags. `to_readable()` and
`to_writable()` return flagged copies; `set_readable` and `set_writable` change a descriptor in
place. `MLTensor` exposes `shape()`, `data_type()`, `readable()`, `writable()` and
`rustnn_required_bytes()`. The spec's `destroy()` is not implemented: dropping the tensor
releases it.

## Data types

`MLOperandDataType` has `Float32`, `Float16`, `Int32`, `Uint32`, `Int64`, `Uint64`, `Int8`,
`Uint8` and the rustnn extensions `Int4` and `Uint4`. Four-bit values are packed two per byte;
`rustnn_storage_byte_length(elements)` returns the storage size for a count. The graph model
uses `rustnn::graph::DataType` for the same set; conversions in both directions exist.

## Errors

| Type | Returned by |
|---|---|
| `Error` | `MLContext` methods, `MLGraphBuilder::build`, `dispatch`, saving. Variants name the failing stage: `NoBackendAvailable`, `GraphBuildError`, `GraphDispatchError`, `WrongWriteSize`, `DuplicateTensorBinding`, ... |
| `GraphBuilderError` | Recording: `GraphAlreadyBuilt`, `WrongConstantSize`, `RequestedInputAsOutput`, `DuplicateOutput`, shape inference failures |
| `ShapeInferenceError` | Per-operation shape and data type checks |
| `GraphError` | Loading, validation, conversion and the legacy executors |

All error types are `Send + Sync` and compose with `anyhow` and similar crates.

## Graph model and legacy pipeline

`MLGraphBuilder` records into a `GraphInfo`: a `Vec<Operand>`, a `Vec<Operation>` and the
constant data. Operands are referenced by index (`u32`). `Operation` is an enum with named
operand fields and an `Option<ML*Options>` per variant; `Operation::op_type()` returns the
WebNN name.

The same `GraphInfo` drives the file formats and the legacy pipeline:

| Item | Module | Purpose |
|---|---|---|
| `load_graph_from_path(path)` | `rustnn::loader` | Read `.webnn` text or JSON, resolve external weights, run shape inference |
| `to_graph_json`, `from_graph_json` | `rustnn::webnn_json` | Convert between `GraphInfo` and the `webnn-graph` JSON model |
| `GraphValidator::new(&graph, ContextProperties)` | `rustnn::validator` | Structural validation; returns `ValidationArtifacts` with named I/O descriptors |
| `ConverterRegistry::with_defaults()` | `rustnn::converters` | `convert("onnx" / "coreml" / "trtx" / "litert" / "cann", &graph)` to a `ConvertedGraph` |
| `run_onnx_with_inputs`, `run_trtx_with_inputs`, `run_coreml_with_inputs` | `rustnn::executors` | One-shot execution of converted bytes; used by the CLI, superseded by `MLContext` |
| `graph_to_dot(&graph)` | `rustnn::graphviz` | Graphviz DOT export |

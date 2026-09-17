# WebNN Specification and rustnn

This page maps the concepts of the [W3C WebNN specification](https://www.w3.org/TR/webnn/) to
rustnn's Rust API. `docs/reference/webnn-index.bs` is a cached copy of the specification source;
its date is in `docs/reference/README.md` in the repository. To search the live text install
`search-bikeshed` and run `search-bs index https://github.com/webmachinelearning/webnn/blob/main/index.bs --name webnn`
once, then `search-bs search --name webnn "<term>"`.

## Interfaces

| WebNN | rustnn | Notes |
|---|---|---|
| `navigator.ml.createContext(options)` | `MLContext::create(&MLContextOptions)` | Selects a backend from the hints; see [Backends](../user-guide/backends.md) |
| `MLContextOptions` (`powerPreference`, `accelerated`) | `MLContextOptions::new(MLPowerPreference, accelerated)` plus `rustnn_` extensions (backend hint, TensorRT options) | |
| `MLContext.opSupportLimits()` | `MLContext::op_support_limits` | Returns the generic limits; per-backend limits are a known gap |
| `MLContext.lost`, `destroy()` | `lost()`, `destroy()` | Stubs; contexts are not lost outside the browser |
| `MLGraphBuilder(context)` | `MLGraphBuilder::new(&mut context)` | One graph per builder |
| `builder.input(name, descriptor)` | `input(&str, &MLOperandDescriptor)` | |
| `builder.constant(descriptor, buffer)` | `constant_from_slice`, `constant_from_vec`, `constant_from_bytes`, `constant_from_value` | Bytes are checked against the descriptor |
| `builder.<op>(...)` | `snake_case` method, `<op>_with_options` for the options dictionary | `where` is `where_`; see the operation table in the [API Overview](../user-guide/api-reference.md) |
| `MLOperand.dataType`, `shape` | `data_type()`, `shape()` | Shapes are `Vec<u32>`; dynamic dimensions need the `dynamic-inputs` feature |
| `builder.build(outputs)` | `build(&MLNamedOperands)` | Compiles on the selected backend |
| `MLTensorDescriptor` (`readable`, `writable`) | `MLTensorDescriptor::new(...).to_readable().to_writable()` | |
| `context.createTensor`, `writeTensor`, `readTensor` | `create_tensor`, `write_tensor`, `read_tensor` | Synchronous; there are no promises |
| `context.dispatch(graph, inputs, outputs)` | `dispatch(&mut graph, &MLNamedTensors, &MLNamedTensors)` | Binding checks in `src/runtime_checks.rs` |
| `MLGraph.destroy()`, `MLTensor.destroy()` | `Drop` | |
| `createConstantTensor`, `createContext(gpuDevice)` | `create_constant_tensor`, `create_from_gpu_device` | Not implemented; return an error |

## Enumerations and dictionaries

- `MLOperandDataType`, `MLPowerPreference`, `MLInputOperandLayout`, `MLConv2dFilterOperandLayout`,
  `MLConvTranspose2dFilterOperandLayout`, `MLRoundingType`, `MLPaddingMode`,
  `MLInterpolationMode`, `MLRecurrentNetworkDirection`, `MLRecurrentNetworkActivation`,
  `MLGruWeightLayout`, `MLLstmWeightLayout` live in `src/operator_enums.rs`. String values match
  the specification's enum strings and serialize with `serde` as camelCase.
- Every `ML*Options` dictionary is a struct in `src/operator_options.rs` with the same field
  names in `snake_case`, the specification's defaults documented per field, and `label`
  inherited from `MLOperatorOptions`. Fields of type `MLOperand` (for example `bias`, `scale`,
  `initialHiddenState`) hold operand indices and are resolved by the converters.
- `MLNumber` (a float or integer union in the spec, used by `clamp` and `pad`) is stored as a
  JSON number (`serde_json::Value`) in the option structs and interpreted according to the
  operand data type when the graph is lowered.

## Algorithms

| Specification section | rustnn |
|---|---|
| Validation of operand descriptors, broadcasting, `MLOperand` compatibility | `src/validator.rs` and the per-operation rules in `src/shape_inference.rs`; the builder runs both on every call and fails early with `GraphBuilderError` or `ShapeInferenceError` |
| Output shape computation per operation | `infer_<op>_shape` in `src/shape_inference.rs`, with unit tests |
| Constant validation (`byteLength` = product of shape times element size, 4-bit packing) | `MLGraphBuilder::constant_*`, `DataType::byte_length` in `src/graph.rs` |
| Graph build validation (outputs must be operation results, inputs and constants may not be outputs) | `MLGraphBuilder::build` |
| Dispatch validation (names, shapes, data types, no duplicate bindings) | `src/runtime_checks.rs` |
| Operation emulation appendix | The converters decompose operations the backends lack; the choices are documented in [Converter Internals](../development/converters.md) |

## Deviations and extensions

- Synchronous API: every promise-returning method is a blocking call returning `Result`.
- `rustnn_` methods extend the API: backend hints and options, `rustnn_save_webnn`,
  `rustnn_webnn_text_for_outputs`, `rustnn_set_tensor_capacity`, `rustnn_resize_tensor`,
  `rustnn_required_bytes`, `rustnn_index`.
- `squeeze` and `unsqueeze` are kept as convenience operations although the specification
  removed them; they are lowered to `reshape` and have no WPT coverage.
- `globalAveragePool` and `globalMaxPool` from earlier drafts and the `shape` extension emitted
  by onnx2webnn are kept as `Operation` variants; a converter without a lowering rejects them at
  build time.
- Data type support depends on the backend; the WPT conformance run per backend is the record
  of what passes. See [Implementation Status](../development/implementation-status.md) for the
  list of gaps.

## Conformance

The WebNN Web Platform Tests are the oracle: `tests/run_wpt_conformance.rs` runs the upstream
`conformance_tests` corpus against each backend, with tolerances taken from the tests
themselves. The [WPT Conformance Guide](../testing/wpt-test-guide.md) describes the harness and
the [dashboard](https://rustnn.github.io/rustnn/wpt-conformance/) the current per-operation
status.

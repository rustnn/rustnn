# Implementation Status

This page records which parts of the WebNN API rustnn implements and where the gaps are.
Counts and pass rates are not repeated here because they are generated:

- Operation-by-backend matrix: [Backend Operator Support](backend-operator-support.md), from `make docs-backend-ops`
- Conformance per operation and backend: [WPT conformance dashboard](https://rustnn.github.io/rustnn/wpt-conformance/), nightly
- Test baselines: PASS snapshots in `tests/snapshots/` and `tests/wpt_conformance/*_expected_failures.txt`

## WebNN API surface

| Interface | Implemented | Not implemented or different |
|---|---|---|
| `ML`, context creation | `MLContext::create(&MLContextOptions)` with the `powerPreference` and `accelerated` hints | `createContext(GPUDevice)`: `create_from_gpu_device` is a stub |
| `MLContext` | `createTensor`, `writeTensor`, `readTensor`, `dispatch`, `accelerated`, all synchronous | `lost`, `destroy`, `createConstantTensor` and `opSupportLimits` are stubs (`todo!()`); resources are released by `Drop` |
| `MLGraphBuilder` | constructor, `input`, `constant` from buffers, every operation of the specification (see below), `build` (synchronous) | `constant` from an `MLTensor` and the scalar `constant(type, value)` form |
| `MLOperand` | `shape` and `dataType` through a `&GraphInfo` (`MLOperand::shape(graph)`) or the builder (`rustnn_operand_shape`) | The spec exposes them directly on the operand |
| `MLGraph` | Opaque compiled graph with `input_descriptors` and `output_descriptors` | - |
| `MLTensor` | `shape`, `dataType`, `readable`, `writable`, `constant` | `destroy` and `isDestroyed` are stubs |
| Data types | All spec types plus `int4` and `uint4` | - |
| Asynchrony | None; the JavaScript promises map to blocking calls | An async API is under consideration; the affected methods are marked `//async` in the source |

rustnn extensions, all prefixed `rustnn_` or documented as such: `rustnn_backend`,
`rustnn_device`, `rustnn_device_type`, `rustnn_set_tensor_capacity`, `rustnn_resize_tensor`,
`rustnn_save_webnn`, `rustnn_webnn_text_for_outputs`, `rustnn_operand_shape`,
`rustnn_operand_data_type`, `MLGraphBuilder::new_uncompiled`, `build_graph_info`,
`finish_graph_info`, and the backend and device hints on `MLContextOptions`.

## Operations

The builder implements every operation method of the W3C WebNN specification as of the copy in
`docs/reference/webnn-index.bs` (its date is in `docs/reference/README.md`), including the
recurrent operations `gru`, `gruCell`, `lstm` and `lstmCell` and the 4-bit quantization paths.
Five additional operations are kept: `globalAveragePool` and `globalMaxPool` from earlier
drafts, `squeeze` and `unsqueeze` (moved to the specification's emulation appendix and still
emitted by onnx2webnn) and `shape`, a rustnn extension for onnx2webnn exports.

Backend gaps, from the generated report:

- ONNX Runtime lowers every operation.
- CoreML and TensorRT-RTX lower everything except the `shape` extension.
- LiteRT has no recurrent operations (`gru`, `gruCell`, `lstm`, `lstmCell`) and no
  `globalAveragePool`, `globalMaxPool`, `unsqueeze` or `shape`; several data type combinations
  are rejected up front (`dtype_unsupported_for_op` in `src/backends/litert.rs`).
- CANN supports a small subset; the CANN column of the generated report is the list
  (`is_supported_op` in `src/converters/cann.rs` is its source).

"Supported" means the converter emits a lowering. Numerical conformance per case is what the
WPT runs report; the remaining CoreML failures, for example, come from negative scatter indices
and integer precision limits of the hardware rather than from missing lowerings.

## Backends

| Backend | State |
|---|---|
| ONNX Runtime | Reference backend; runs the full WPT suite in CI with PASS snapshots |
| TensorRT-RTX | Native lowering with refittable weights, engine and runtime caches, CUDA graphs. WPT snapshots are maintained by contributors with GPUs; CI only compiles the backend |
| CoreML | MLProgram lowering; runs the full WPT suite in CI on macOS with an expected-failure list |
| LiteRT | TFLite lowering; runs in CI (non-blocking) with snapshots and an expected-failure list |
| CANN | Runs on OpenHarmony devices; mock mode in CI |
| Browser WebNN (wasm32) | Generated bindings and a graph-compilation test in Chrome; no `MLContext` backend yet |

## Known gaps

- Operand fields of option structs (`MLConv2dOptions::bias`, `MLGemmOptions::c`, quantization
  zero points) are operand indices rather than `MLOperand` values; `MLOperand::rustnn_index()`
  provides them.
- Dynamic dimensions are only available for graphs loaded from files and require the
  `dynamic-inputs` feature; builder descriptors are static.
- `Int4` and `Uint4` constants cannot be exported to `.safetensors`.
- The CLI needs a runtime feature at build time.
- The legacy executors deserialize the model on every call and have no device tensors.

## Conformance tracking

A WPT case is either passing, recorded as a PASS snapshot (`onnx`, `trtx`, `litert`), or listed
in the backend's expected-failure file (`coreml`, `litert`). The corpus is pinned by
`WPT_REVISION`; a weekly workflow refreshes snapshots and expectations and opens a pull
request. See the [WPT Conformance Guide](../testing/wpt-test-guide.md).

## Versioning

rustnn is published as a `0.5.x` development release. The API changes without notice;
`CHANGELOG.md` records notable changes per release.

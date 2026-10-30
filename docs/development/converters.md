# Converter Internals

A converter lowers a `GraphInfo` to one backend format. This page describes the
contract, the structure converters share, the backend-specific rules that shaped them
and how to debug a wrong model. Read it before touching `src/converters/`.

## Contract

```rust
use rustnn::GraphInfo;
use rustnn::converters::ConvertedGraph;
use rustnn::error::GraphError;

pub trait GraphConverter {
    fn format(&self) -> &'static str;                                   // registry key, e.g. "onnx"
    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError>;
}
```

`ConvertedGraph` carries the serialized model (`data`, `content_type`) and an optional
`weights_data` sidecar. `ConverterRegistry::with_defaults` registers every converter compiled
into the build. Each `<backend>-runtime` enables its matching `<backend>-converter`;
converter-only users can call the registry without enabling rustnn execution.

Graph building and graph import apply operation rules. Converters accept `GraphInfo` directly
and enforce their backend-specific constraints; there is no separate graph-wide validation
pass before conversion. Use `GraphInfo::io_binding_maps()` for named I/O descriptors.

Converters read `GraphInfo` and produce model bytes. ONNX, CoreML, LiteRT and Burn
serialization do not require execution libraries. TensorRT engine building and CANN offline
model compilation use their native SDKs during conversion. Tensor allocation, dispatch and
refitting runtime weights belong to `src/backends/<name>.rs`.

| Converter | File | Output | Registered when |
|---|---|---|---|
| `OnnxConverter` | `onnx.rs` | ONNX `ModelProto` (prost) | `onnx-converter` (default) |
| `CoremlMlProgramConverter` | `coreml_mlprogram.rs` | CoreML MLProgram (`protos::coreml`) | `coreml-converter` |
| `TrtxConverter` | `trtx.rs`, `trtx_gru.rs`, `trtx_lstm.rs`, `trtx_rnn.rs` | TensorRT network built through the `trtx` crate, serialized engine | `trtx-converter` |
| `LiteRtConverter` | `litert.rs` | TFLite flatbuffer (`flatc`-generated schema) | `litert-converter` |
| `CannConverter` | `cann.rs` | HiAI offline model | `cann-converter` |
| `BurnConverter` | `burn.rs` | Serialized Burn execution plan | `burn-cpu-converter` |
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
- `resample2d` emits exact integer sizes, computed as `floor(input_size * scale)` in
  double precision when explicit sizes are absent.

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
  edits invalidate cached engines. This source hash does not invalidate the separate TensorRT
  JIT runtime cache. See the [cache guide](https://rustnn.github.io/rustnn/integration/tensorrt/#caches).
- The nearest-neighbour rounding of `resample2d` is `round_prefer_floor` (`kHALF_DOWN`).
  Output dimensions use `floor(input_size * scale)` in double precision, matching shared
  shape inference; explicit sizes take precedence over scales.
- Float16 `reduceLogSum` promotes the input to float32 for the sum and logarithm,
  then casts the result back to float16. This avoids overflow when the sum exceeds
  float16's range but its logarithm remains finite.
- `triangular` builds a Boolean keep mask and selects the input or a typed zero. Multiplying by
  a zero mask would propagate NaN and infinity into positions that WebNN requires to be zero.

### CoreML

- MIL has no rank-0 tensors at the boundary, no dilation in pooling, no `edge`/`reflection`
  padding above two dimensions and no tensors of rank 6 and above; integer arithmetic runs in
  float32.
- Comparison results are `uint8`; `reduceLogSumExp` uses the max-shifted form; reductions with
  empty `axes` and `resample2d` on arbitrary axes are lowered explicitly.
- Normalization affine parameters use native constants or rank-aligned reshapes with
  elementwise broadcasting. Keep their representation independent of the activation tensor's
  element count: activation-sized integer gather maps become enormous textual MIL literals
  during CoreML compilation and model loading.
- Gather-family index normalization uses active indexed dimensions, not their declared maxima.
  Scalar `gather` indices stay as one-element vectors through normalization and gathering;
  an axis-specific squeeze removes only the indexed dimension afterward. This avoids native
  scalar-index gather failures while retaining dynamic extents and unrelated singleton axes.
  A scalar result keeps the `[1]` CoreML boundary representation without changing the WebNN rank.
- A power of two scalar constants promotes one argument to `[1]` so MIL infers the
  scalar result using the same boundary representation. Scalar constants used by other
  operations retain their native rank.
- Float16 weights go to the weight blob written by `weight_file_builder.rs` and returned as
  `weights_data`.
- `transpose`, `reshape` and `cast` of a constant are folded into a new constant before
  emission (`coreml_constant_fold.rs`, applied to a copy of the graph at the start of
  `convert_with_float16_precision`, for both float16 lowerings). The Neural Engine admits `matmul(x, const)` but refuses
  `matmul(x, transpose(const))`, and a literal lowering would also transpose the weight on
  every dispatch on the CPU and GPU paths; GPT-Neo style graphs carry one such transpose per
  projection. The folded operand keeps its index and name, graph inputs and outputs are never
  rewritten, a constant with other consumers keeps its payload, and chains fold in one pass.
  Narrow integer constants (8-bit and 4-bit, the quantized weights whose casts and reshapes
  are dequantization chains that would otherwise be stored widened), casts that would grow the
  stored payload (float16 weights read as float32 stay float16 in the weight blob), and casts
  that WebNN leaves implementation-defined (a float whose truncation does not fit the integer
  target, or NaN) stay in the graph.
- WebNN names need not be MIL identifiers, and its input/output namespaces are independent.
  Unsafe or colliding names receive unique physical identifiers. Creator-defined
  `rustnn.webnn.input_aliases` and `rustnn.webnn.output_aliases` JSON mappings preserve
  logical bindings in byte-buffer, retained-tensor and one-shot execution, and standalone
  exports. The JSON maps define the binding contract; models without them keep literal names.
  See the [tensor-name contract](https://rustnn.github.io/rustnn/integration/coreml/#tensor-names).
  Source-proven copies with identical type, shape and pending layout share one
  physical result: CoreML can omit duplicate scalar/dynamic copy features from prediction.
  Logical tensors still receive independent results. Unequal computations and real dtype
  conversions remain distinct. Same-float32 casts lower as identity rather than the native
  cast kernel, which can narrow represented values through float16.
  Produced copies rooted in a graph input additionally record that source and its descriptor
  in `rustnn.webnn.output_passthroughs`. Execution snapshots the original bound input before
  prediction, validates its dtype, actual shape and byte length, then supplies independent
  output bytes. This also preserves a public float32 copy when CoreML narrows another consumer.
  Copy proofs cover identity, same-type cast, same-shape reshape, identity transpose, and
  static full-span, unit-stride slice. An equal square shape does not prove a transpose is
  an identity; dynamic maximum extents do not prove a full-span slice.
  Copies rooted in original constants use version-2 `rustnn.webnn.output_constant_copies`
  metadata: output bindings reference a descriptor, original weight-record offset and
  consistency checksum, without embedding tensor payloads. Unchanged blob weights reuse
  their record; immediate or converted constants retain one original raw record per source.
  Execution validates referenced ranges and byte lengths, shares the immutable weight owner
  or compacts only the required ranges, then supplies independently owned results. Standalone
  compiled models require the original weights sidecar. The valid native graph still
  runs; arithmetic-derived outputs are not reconstructed from inputs or constants.
- Float16 lowering follows `CoremlOptions::float16_precision`
  (`CoremlMlProgramConverter::convert_with_float16_precision`; `GraphConverter::convert` uses
  `Native`). WebNN leaves float16 accumulation precision open (webmachinelearning/webnn#948),
  so both lowerings conform; they differ in which compute units reproduce the CPU's values.
  - `Native` (default): float16 operations lower to native float16 MIL kernels. One MIL
    program, no added operations, eligible for the Neural Engine.
  - `Exact`: compute is float32 and storage stays float16; each WebNN float16 result is
    rounded once. The rules below apply only to float16 operands.
    - `gelu` (MIL `mode="EXACT"`), `sigmoid`, `tanh` (NaN preserved), `relu`, `leakyRelu`,
      `elu`, `exp`, `neg` (division by -1), `reciprocal`, `hardSigmoid` and `hardSwish` (its
      sigmoid helper is not rounded separately) evaluate in float32. Parameters such as
      `alpha` are rounded to float16 first, as WebNN casts them at graph construction.
    - `matmul`, `layer_norm`, `sub` and `l2_pool`, and data movement (`identity`,
      `reshape`, `transpose`, `slice_by_size`, `slice_by_index`, `reverse`, `concat`,
      `split`, `gather`, `gather_along_axis`, `gather_nd`, `tile`, `pad`, `select`,
      `squeeze`, `expand_dims`, `scatter_along_axis`, `scatter_nd`) take float32 operands
      and are cast back after the operation, because accelerator kernels can flush
      represented subnormals or lose small values.
    - `layerNormalization` normalizes and applies scale and bias in float32. `sign`,
      `roundEven`, `isNaN`, `isInfinite` and identity copies of constants also evaluate in
      float32. Constant float32-to-float16 casts are folded on the host with
      round-to-nearest-even, and public float16 constant outputs are materialized.
    - Explicit float16 rounding boundaries become native Pipeline children when Core ML
      could otherwise eliminate a narrowing or fuse a widened kernel back into float16.
      Child interfaces carry live results only; constants are rematerialized from the
      shared weight blob. Public descriptors and the backend-independent graph are
      unchanged; private scalar and Boolean interfaces use explicit rank and type adapters.
    - Affected float16 widening layouts use compact private features. Original high-rank
      inputs declare `rustnn.webnn.compact_input_views`; contiguous storage is viewed with
      its owner retained, padded storage is copied as raw float16 bytes, and dynamic
      restoration uses the actual input shape.
- Rules for both lowerings and every float type:
  - `reciprocal` lowers to `real_div(1, x)`: MIL `inverse` loses the sign of -0 on older
    Core ML stacks. `log` passes `epsilon = 0`: WebNN specifies `log(x)`.
  - `roundEven` computes ties-to-even explicitly (MIL `round` ties away from zero on GPU
    and Neural Engine kernels), in the input type for `Native` and in float32 for `Exact`.
  - Float32 `tanh` keeps NaN, float32 `sign` returns 0 for NaN, float32 `neg` divides by
    -1 (both zero signs survive) and float32 identity copies of constants use
    `real_div(x, 1)`.
  - `layerNormalization` with constant scale and bias on trailing, ascending axes keeps the
    fused native `layer_norm(gamma, beta)`. Nontrailing or permuted axes and runtime scale or
    bias apply the affine work with explicit `mul`/`add` in the declared axis order; for
    nontrailing or permuted axes that work is placed in its own Pipeline child, because a
    native plan that fuses it back can omit or reorder the parameters. A missing epsilon
    defaults to 1e-5.
  - `instanceNormalization` with runtime scale or bias runs native `instance_norm` without
    them and applies `mul`/`add` over the channel axis. An NHWC input takes the same
    NHWC→NCHW and NCHW→NHWC transposes around the native op as the constant-affine path.
    Scale or bias computed from constants (a widening `cast` of float16 constants, which
    the constant-view fold keeps in the graph) takes this path too.
  - Pipeline boundaries take shape bounds through every shape-preserving MIL operation
    (casts and copies, unary arithmetic, activations, normalization, `softmax`), so a
    bounded dynamic dimension keeps its provenance across a cut.
- The internal `shape` extension lowers to MIL `shape`, retaining its native int32 result
  inside CoreML and widening the public int64 result at readback. Imported shape tensors
  retain their type and rank through the shared `unsqueeze` inference path.

### LiteRT

- Spatial operations are NHWC. `is_spatial_op` marks operands whose layout changes; filters are
  transposed to OHWI and boundary tensors are transposed by the backend at run time.
- The converter rejects unsupported operation/data-type combinations; the runtime can impose
  further kernel restrictions. See the shared
  [LiteRT guide](https://rustnn.github.io/rustnn/integration/litert/) and generated support report
  rather than assuming the WPT skip list is a complete runtime capability query.

### CANN

- `is_supported_op` is the allow-list; everything else fails at build with an explicit error.

## Operator support report

`scripts/generate_backend_operator_report.py` derives the
[operator support matrix](https://rustnn.github.io/rustnn/development/backend-operator-support/) from the sources: an operation counts as
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

Set `RUST_LOG=rustnn=debug` to include conversion diagnostics in the application's
configured logger. Conversion failures use error logs and recoverable fallbacks use warnings.

The signature of a converter bug: rustnn's shape inference is right, the emitted model is not,
and the error surfaces downstream, as a type error at load time or an out-of-bounds axis on the
next operation. Compare the emitted node with the backend's semantics before touching shape
inference.

| Backend | Dump | Inspect |
|---|---|---|
| ONNX Runtime | `RUSTNN_ONNX_DUMP_DIR=<dir>` writes each converted model to `debug_webnn_onnx.onnx` (overwriting the previous dump) | `python -c "import onnx; print(onnx.load('<file>'))"` or Netron |
| TensorRT-RTX | `TRTX_JSON_DUMP_PATH=<dir>` writes the layer graph of each engine; `RUSTNN_TRTX_LOG_VERBOSITY=verbose` for the builder log | look for the layer that consumes the operand; Myelin-fused layers print as `__myl_...` |
| CoreML | `ConvertedGraph::data` and `weights_data` | `coremltools` or Xcode |
| LiteRT | Write `ConvertedGraph::data` to `model.tflite` | `flatc --json` with `protos/tflite/schema.fbs`, or the TFLite visualizer |
| any | `--export-dot graph.dot` shows the graph rustnn recorded | Graphviz |

Then write the regression test first, fix the lowering, and run the WPT cases for the operation
on every backend you have.

## Burn CPU

With `burn-cpu-converter`, `BurnConverter` lowers typed operations and embedded constants
into an in-memory execution plan when the graph is built. The CPU runtime prepares and
replays it using Burn NdArray and host kernels. Conversion and execution are enabled
together by this single feature.
The MLContext builder passes its validated graph to Burn preparation, which checks only
backend tensor representation limits and initializes constants and lifetimes. Operation
arity, datatype and graph structure rules belong to the builder and are not revalidated
by Burn preparation. Dispatch does not serialize or deserialize the graph. Prepared
execution retains graph outputs and constants, releases intermediate tensors after their final consumer, and
clears temporary state after failed runs. Serialized standalone execution validates plan
structure and metadata at its external entry point and prepares a fresh runtime per call.
Both paths retain tensor payload validation and kernel-result publication checks.
Constant dimensions must be static; runtime inputs may use bounded dynamic dimensions.
Resampling shape inference floors scaled dimensions and dynamic bounds. Burn float
resampling uses the same flooring rule as its integer path; explicit sizes override
scales, and sampling coordinates use the actual output-to-input size ratio.
Packed int4/uint4 constants and tensors use the shared nibble codecs from `graph.rs`.
Decoding uses the logical shape's element count, so the unused high nibble of odd-length
buffers is ignored. Quantization computes logical values, clamps to the target 4-bit range,
and packs outputs with zero padding. The typed host interpreter handles integer arithmetic,
comparisons, casts, reductions, tensor movement and integer indices using exact integer
payloads. Broadcast and block quantization parameters preserve integer zero points;
dequantization subtracts them before converting to floating point. Integer arithmetic wraps
to the operand width, integer casts reinterpret the low bits, and float-to-integer casts
truncate and saturate (NaN becomes zero). Strided slices use WebNN input-window
extents, with output sizes computed as `ceil(size / stride)`, for floating-point and integer
tensors. Integer clamp and padding bounds are parsed without
passing through floating point. An operation involving integers must have an explicit typed
implementation; dispatch rejects it rather than falling back to a float kernel.
The runtime stores each operand in one opaque, validated value containing a Burn float
tensor, signed integers, or unsigned integers. Its private storage variants and restricted
integer kinds determine the datatype. Constructors check shape, length and integer range;
only the environment owns its value map. Integer kernels use borrowed exact views and
apply wrapping or cast conversion before constructing results. Import rejects out-of-range
exact integers and fractional or non-finite narrow integers supplied as floats. Integer operands
never acquire persistent float mirrors. The public serialized-plan executor generates its
legacy float output view only when collecting results. Device outputs declared as float16
are rounded at every operation boundary, as are host outputs. Accumulation inside
a float16 operation can still use float32. See
[Burn CPU](https://rustnn.github.io/rustnn/integration/burn-cpu/) for setup and precision limits.


Burn runtime maintenance keeps datatype-independent movement in `typed/movement.rs`,
integer arithmetic and conversion in `typed.rs`, recurrent kernels in `rnn.rs`, and floating
host kernels in `float_ops.rs` and `host_array.rs`. `value.rs` owns operand storage and output
conversion; `tensor_env.rs` owns the operand map and exposes read-only inputs to kernels.
`execute.rs` explicitly routes operations to shared movement, typed arithmetic or float
execution. Every kernel writes through `outputs.rs`, which requires datatype metadata,
checks output ownership and exactly-once completeness, and stages all results until the
operation succeeds. Checked value construction enforces storage shape, length and datatype;
graph output binding shapes are checked before publication, including dynamic dimensions.
Float16 rounding occurs at this shared output boundary.
`shape.rs` owns validated slice and reduction descriptors, broadcasting and axis helpers.
Slice windows and reduction output shapes reuse the main shape-inference functions with
resolved input dimensions; coordinate mapping is independent of datatype. Intermediate
shapes are computed from runtime inputs, without adding serialized plan metadata.
`runtime_tensor.rs` adapts runtime ranks to Burn const generics. Its local macros expand
mechanical read, map, matching-rank binary and construction dispatch; scalar rank remains
explicit, and numerical formulas stay in ordinary Rust functions. Adapter rank coverage is
separate from backend capability: NdArray's native reshape and kernels that depend on it
are limited to six ranks. Shared movement can use the plan's eight-rank limit for float
and exact integer values. Layout and activation choices
use the existing WebNN enums. Shared options that still contain strings are parsed at kernel
entry by `options.rs`; empty strings retain their documented defaults and other invalid
values return errors. Keep string conversion at serialization and API boundaries, not in
numeric loops. The operator report scans both numeric dispatchers and the shared movement
module, so moving an operation between them does not remove it from the report.

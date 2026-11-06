# CoreML Backend

The `coreml` backend runs WebNN graphs through Apple CoreML on macOS and iOS. The converter
(`src/converters/coreml_mlprogram.rs`) emits an MLProgram, the MIL-based model format, and the
backend (`src/backends/coreml.rs`) compiles it once and keeps the compiled model for repeated
dispatch. The Objective-C bridge lives in `src/executors/coreml.rs` and `src/executors/coreml_shim.mm`.

## Requirements

- A CoreML version that accepts MLProgram models (macOS 13 or newer). The native Rust
  backend also supports iOS; mobile validation uses iOS 18 as its deployment target.
- The source gates cover every Apple target (`target_vendor = "apple"`). CI builds and tests
  macOS only. tvOS, watchOS and visionOS builds additionally depend on the `objc` 0.2 message
  ABI, which the published crate selects correctly only for macOS and iOS; see
  [Compile routes](#compile-routes) for the watchOS requirements.
- The `coreml-runtime` Cargo feature. On Linux and Windows the feature compiles to shims whose
  calls always fail, so `cargo check --features coreml-runtime` works everywhere but the backend
  is never selected on those platforms.
- Xcode command line tools for the Objective-C++ shim (`build.rs` compiles it with `cc`).
- Optionally the `coreml-export` Cargo feature, which adds the pure-Rust `.mlmodelc` writer
  ([Compile routes](#compile-routes)). It is required for CoreML on watchOS.

## Selection and devices

`BackendDevice::Coreml { device_type }` maps the WebNN hints to CoreML compute units:

| Hint | `DeviceType` | `MLComputeUnits` |
|---|---|---|
| not accelerated | `Cpu` | `cpuOnly` |
| accelerated, `Default` or `HighPerformance` | `Gpu` | `cpuAndGPU` |
| accelerated, `LowPower` | `Npu` | `cpuAndNeuralEngine` |

CoreML decides at run time which units execute which layers; the hint is a ceiling, not a
guarantee. The WPT harness pins `accelerated = false` (CPU) so that results are deterministic.

For the unified `MLContext` path, inspect `graph.rustnn_load_diagnostics()` after
building and match `LoadDiagnostics::Coreml`. It retains the requested policy, successful policy/route, and earlier errors even
when CPU-only or URL fallback succeeds. Route preparation errors have no compute-unit policy;
a deliberately selected URL route is not reported as a failed in-memory attempt. The same
summary is logged once at debug level under `rustnn::executors::coreml::load`, not per dispatch.

This distinguishes load fallback from CPU scheduling within an accelerator-enabled model.
It helps investigate sustained CPU thermal throttling without attributing that throttling to
an unobserved GPU/Neural Engine workload. Neither a successful policy nor this diagnostic
establishes placement, energy savings or prediction-time fallback; use separate device traces.

## How a graph runs

1. The converter lowers the graph to MIL programs. Rank-0 operands are promoted to `[1]` at the
   model boundary, comparison and logical results are produced as `uint8`, and reductions with
   empty axes, `resample2d` on arbitrary axis pairs and the stable `reduceLogSumExp` form are
   lowered explicitly because MIL has no direct equivalent.
2. Non-scalar float16, float32, int8 and uint8 constants are written to a separate shared weights
   blob (`ConvertedGraph::weights_data`), which also keeps the original bytes of constants that
   outputs copy (see Tensor names below). The in-memory and URL compile routes do not produce
   identical results for every program, and neither is uniformly closer to WebNN, so the route
   follows the float16 lowering. With `Native`, the model is packaged as an in-memory asset
   together with that blob, and graphs with a reproduced difference compile from a temporary
   directory's URL instead (`supports_in_memory_asset`): `gather` with a rank-0 index, int32
   `neg`, int32 off-diagonal `triangular`, `convTranspose2d`, and `instanceNormalization` with
   zero epsilon. With `Exact`, graphs compile from a URL, the route through which its float16
   guarantees were qualified: in memory, widened float16 subtraction loses NaN classes under a
   CPU and Neural Engine request. Input-free programs made entirely of proven constant copies
   stay in memory with either lowering, because the URL compiler can crash in BNNS for such
   data-free programs under a CPU and Neural Engine request. An input-free program with no
   float16 operand follows the `Native` rule under `Exact` as well: the exact lowering changes
   nothing in it, and the URL compiler, which evaluates such a program itself, returns 0 on
   iOS and tvOS 26 and 27 for a constant `uint8` clamp that saturates to 255 (WPT mlNumber
   "cast float to integer overflows"). Neither route forces CPU-only execution.
   With [`CoremlFloat16Precision::Exact`](#float16-precision), explicit float16 rounding
   boundaries are materialized as native Pipeline children, and with either lowering a
   `layerNormalization` over nontrailing axes keeps its affine work in its own child. The
   children expose live values only and reuse the original weight storage; the public WebNN
   graph, types and shapes are unchanged.
3. `MLGraphBuilder::build` compiles the model with `MLModel` and keeps the compiled model;
   `dispatch` binds `MLMultiArray`s over the tensor storage and runs a prediction. With the
   `RustExport` [compile route](#compile-routes) the bundle is written in Rust instead and
   `MLModel` loads it from its URL.
4. The `_with_weights` one-shot executors accept that blob and try compute-unit
   configurations in turn. Converter-only users can generate protobufs and weights on
   any host with `coreml-converter`; execution requires `coreml-runtime` on Apple platforms.

## Float16 precision

WebNN leaves the accumulation precision of float16 operations to the implementation
([webmachinelearning/webnn#948](https://github.com/webmachinelearning/webnn/issues/948)), and
whether represented subnormals must survive is open
([webmachinelearning/webnn#967](https://github.com/webmachinelearning/webnn/issues/967)). Core ML
uses that freedom differently per compute unit and chip, so the converter offers two lowerings,
selected with `RustNNOptions::coreml.float16_precision`:

- `Native` (default) lowers float16 to native float16 MIL kernels. It adds no operations, keeps
  one MIL program and is the only form the Neural Engine can execute. In the CoreML WPT corpus
  on an M4 Max (macOS 27.2), CPU requests pass every float16 trial and GPU requests fail only
  the `resample2d` nearest-neighbour ties they share with float32. Neural Engine requests fail
  some float16 trials: arg reductions, widening of subnormals, `elu`, `l2Pool2d` and
  `roundEven` ties.
- `Exact` keeps float16 storage but evaluates float16 arithmetic, activations, normalization and
  data movement in float32, rounding once at each WebNN result, and materializes explicit
  float16 rounding boundaries as Pipeline children. In the same WPT corpus it passes every
  trial under CPU, GPU and Neural Engine requests, and on A12 and A10X it removed every native
  float16 GELU failure under all three. The cost is more operations, Pipelines, URL compilation
  and longer builds. Float32 work is not eligible for the Neural Engine, so under a Neural
  Engine request `Exact` gets its values by running that work elsewhere.

`docs/development/converters.md` lists the lowering rules for both. The WPT harness selects the
lowering with `WPT_COREML_FLOAT16=native|exact` and the device with `WPT_COREML_DEVICE`.

## Tensor names

Input and output names remain independent in the RustNN API, including names containing
spaces, Unicode, punctuation, leading digits or MIL keywords. The converter records JSON
logical-to-physical bindings in creator-defined `rustnn.webnn.input_aliases` and
`rustnn.webnn.output_aliases` metadata. RustNN applies them automatically; standalone
consumers should apply them when binding and retrieving CoreML features.
The JSON maps are the binding contract; models without them keep literal feature names.
Duplicate logical keys are rejected, even when their values match. Bindings and original
constant references are validated once when the model is loaded.
Proven equal copy outputs may share one physical result, avoiding CoreML's omission of
duplicate scalar/dynamic features. RustNN supplies each requested logical output tensor;
unequal computations and real dtype conversions remain separate. Returning an original
input or constant directly remains invalid under WebNN's build rules.
Same-dtype `cast` is a proven copy and may be fulfilled without reading CoreML's returned
feature; a cast that changes dtype still executes the conversion boundary.
For produced copy chains rooted in an input, the serialized
`rustnn.webnn.output_passthroughs` map identifies the logical input and original descriptor.
RustNN validates dtype, actual shape and byte length and snapshots that input before
prediction, supplying independent output copies even if CoreML omits or changes a copy
feature. Standalone consumers should honor these proven-copy bindings too; they must not
infer an input/output alias from matching names or values.
Copy proofs also cover same-shape reshape, identity transpose and static full-span
unit-stride slice. Original constant copies use version-2
`rustnn.webnn.output_constant_copies` metadata: each source has its original descriptor,
a `WeightMetadata` offset in `weights/weights.bin`, and a SeaHash checksum of its raw
bytes. Independent output bindings reference these sources; metadata contains no tensor
payloads. Unchanged blob-backed weights reuse the existing MIL record. Constants stored
as immediates or changed during lowering retain one additional raw UINT8 record per source.
Consumers validate the file records, referenced ranges, descriptor byte lengths and
checksums before prediction. The checksum detects a mismatched sidecar, not malicious
modification. In-memory compiled graphs share the existing immutable weight allocation;
one-shot diagnostics borrow it, without decoding a second source allocation. URL-compiled
graphs retain that owner only when its capacity is at most twice the uniquely referenced
bytes; otherwise they compact just those ranges into one shared allocation, so a tiny
copy output does not retain a large unrelated weight file.
Standalone consumers must retain the original weights sidecar alongside the compiled
model; a `.mlmodelc` alone cannot supply RustNN's exact original-copy values. RustNN does
not export an `.mlpackage`: converter users must write the `.mlmodel` and retain
`ConvertedGraph::weights_data` (the `weights.bin` bytes). A supplied sidecar whose records, byte lengths or checksums do
not match fails closed, regardless of tensor size. `MLContext` and the
`_with_weights` one-shot executors pass the sidecar. Without one, CoreML cannot compile a
program that references `weights.bin`; in a model that still loads, every copied constant
is a MIL immediate, and those outputs keep CoreML's computed values like other outputs.
The native graph still runs; these proofs do not replace arithmetic-derived results.
The same checked bindings apply with retained tensor storage enabled. Proven copies
are snapshotted before prediction and written into independently owned outputs;
they do not propose native output backings. Coalesced arithmetic outputs use the
returned physical result, with at most one backing proposed per physical feature.
`proven_copy_outputs` counts logical outputs supplied this way in successful dispatches,
separately from `output_copy_bytes` (which also includes those copies). WPT reports record
per-trial deltas, including trials whose later result comparison fails. Neither counter
measures work or copies inside CoreML or its selected hardware.

## Reusing tensor storage

With `RustNNOptions::coreml.reuse_tensor_storage` enabled, CoreML contexts keep float32,
float16 and int32 tensors in owned storage with
retained `MLMultiArray` views. Dispatch binds compatible input arrays directly. An output can
become the next graph's input without `read_tensor`/`write_tensor` or a temporary input array;
for KV caches, alternate two distinct tensor sets. A dispatch cannot bind one tensor as both
input and output.

Buffers of at least 16 KiB are page-aligned for CoreML's output-backing performance
recommendation; smaller scalar and index buffers use 16-byte alignment.

When supported, `outputBackings` proposes the destination array to CoreML. Only a returned
array with the same object identity counts as accepted. Backings require a fully static
graph (including intermediate operands) and a fixed output feature. A fixed-size output
alone is not sufficient when the graph has dynamic dimensions. Ineligible outputs,
declined backings, strided results and dtype conversions are copied
into the destination's owned storage. Returned arrays are never adopted as tensor storage,
because CoreML may alias them to an input or another output. This is not a guarantee of zero
copies inside CoreML, GPU or Neural Engine drivers.

Both host and retained-storage paths use the same checked numeric conversion and array
layout rules. Equal element widths do not imply equal types (for example int32 and
float32). Same-type copies preserve integer bits and float16 storage widths; overlapping
strided views are gathered before writing the destination.

Numeric conversions into float16 round directly from the source value using integer
round-to-nearest, ties-to-even, retaining all discarded bits. This preserves subnormals
and signed zero independently of hardware half-conversion support. Matching storage
types are copied bit-for-bit, including NaN payloads; CoreML's internal arithmetic policy
is unchanged.

`RustNNOptions::coreml` controls this experimental path. `reuse_tensor_storage` defaults to
`false`, preserving the byte-buffer reference implementation until an application measures
a benefit on its workload. `output_backings` defaults to `true` when reuse is enabled and
can be disabled independently to measure persistent input storage alone. The
WebNN API and compute-unit selection are unchanged. Other data types retain host storage and
the existing conversion path. Zero-extent prediction remains unsupported.

With `dynamic-inputs`, reserve the maximum capacity before a decode loop and resize the
active shape between dispatches. Shape changes rebuild the array view, but do not allocate
another data buffer while they fit the reserved capacity. Reserve replaces storage with
zeroed bytes; growth during resize preserves the existing prefix.

`MLContext::rustnn_backend_statistics()` returns `Some(BackendStatistics::Coreml(...))`
for this backend; backends that do not report statistics return `None`.
The CoreML variant reports cumulative host reads/writes, native allocations, direct input
bindings, proposed/accepted backings and logical copied payloads.
These count rustnn-side work, not total memory traffic or internal CoreML allocations. The
reported compute-unit policy includes load fallback but does not measure accelerator
placement. Take counter differences around the decode loop to exclude prefill and setup.

Precision Pipelines (the `Exact` float16 lowering, and the explicit LayerNorm affine child for
nontrailing axes) retain input tensor storage but copy returned outputs.
CoreML applies the outer output-backing names while evaluating earlier children,
which reject final features absent from their own descriptions. Backings are therefore
proposed only for a single MLProgram, even when a Pipeline's source graph is static.

## Converter-private input views

The `Exact` float16 lowering can request a compact native Half input through creator-defined
`rustnn.webnn.compact_input_views` metadata: a JSON array of `{source, view}` bindings
between declared physical input features. These are backend details, not additional
WebNN inputs; the original graph, logical dtype and shape remain unchanged.

RustNN binds each private rank-one feature using the source's actual element count,
including growing and shrinking bounded dimensions. Contiguous source arrays share
their pointer with a native view whose deallocator retains the source owner. Padded
or strided arrays instead receive an exact raw-Half copy, preserving subnormals,
signed zero and NaN bits without a Float32 round trip. The original feature remains
bound for actual-shape queries and other consumers. Metadata, native dtype, shape
constraints and storage layout are checked before prediction; standalone consumers
of these exports must supply the same private bindings.
Persistent `MLTensor` dispatch uses these bindings with output backings enabled or
disabled, retaining the tensor storage and private view owners through prediction.

## Compile routes

`CoremlOptions::compile_route` (`RustNNOptions.coreml.compile_route`) selects how the lowered
MLProgram becomes a loadable model. It changes the compile step only: the converter, the
numerical lowering, the compute-unit policy and the public graph are the same on every route,
and the successful route is reported as `CoremlLoadRoute` in `graph.rustnn_load_diagnostics()`.

| `CoremlCompileRoute` | What happens | Where it applies |
|---|---|---|
| `Native` | Apple's compiler: an in-memory `MLModelAsset`, or `MLModel.compileModelAtURL:` from a temporary `.mlmodel`/`.mlpackage` (`CoremlLoadRoute::InMemoryAsset` / `CompiledUrl`, see [How a graph runs](#how-a-graph-runs)). The only route without the `coreml-export` feature. | macOS, iOS, tvOS, visionOS |
| `RustExport` | rustnn writes the `.mlmodelc` bundle in its own process with the pure-Rust [`mlmodelc-export`](https://github.com/rebeckerspecialties/mlmodelc-export) crate and loads it with `MLModel(contentsOf:)` (`CoremlLoadRoute::RustExport`). Needs the `coreml-export` feature; otherwise `build` fails with a `CoremlRuntimeFailed` error that names the feature. | Required on watchOS, whose public Core ML API has no model compiler; the route for sandboxed hosts where `compileModel` is unavailable or undesirable (App Extensions and App Clips are the cases named by issue #110; they have not been validated) |
| `Auto` (default) | `Native` wherever Apple's compiler exists. With `coreml-export` compiled in, a failed native compile is retried through `RustExport` and the native failure stays in `failures`. On watchOS, `Auto` is `RustExport`. | Everywhere; keeps the default behavior byte-identical when the feature is off |

The exporter emits source MIL (`model.mil`) plus `coremldata.bin`, `metadata.json` and
`analytics/coremldata.bin`, and copies the converter's weight blob to `weights/weights.bin`
unchanged. Core ML compiles the MIL for the device when the bundle is loaded, so the first load
of an exported bundle costs what `compileModelAtURL:` costs elsewhere. The bundle is a plain
directory: `ConvertedGraph::write_coreml_bundle(dir)` writes one on any host, including Linux
CI, and `rustnn-cli --convert coreml --coreml-compiled-output model.mlmodelc` does the same
from a `.webnn` file (build the CLI with `--features coreml-export`). `rustnn-cli --run coreml
--coreml-compile-route rust-export` selects the route for execution.

Validation receipts (2026-10-30, `mlmodelc-export` at `f958affe`, graphs lowered by this
converter): bundles written by the exporter loaded and predicted on Apple Watch SE (2nd
generation, watchOS 11.6.2) and Apple Watch Series 10 (watchOS 27.2) in 1 to 19 s per model,
on Apple TV 4K and on iPhones; the exporter CLI produced 6 of 6 bundles
from the same packages with byte-identical `weights/weights.bin`. The route does not by itself
establish compute-unit placement or a memory bound; those remain separate measurements.

watchOS, tvOS and visionOS also need the `objc` 0.2 message-send ABI fixed for their targets
(the published crate uses the GNUstep lookup path there), for example with a `[patch.crates-io]`
entry in the application's workspace. These targets are tier 3 in Rust and are not built in CI.

## Testing

```bash
make test-wpt-coreml              # full WPT suite, expected failures are non-fatal
make test-wpt-coreml-report       # same, plus the JSON report
make wpt-sync-coreml              # regenerate tests/wpt_conformance/coreml_expected_failures.txt
make test-coreml                  # unit and integration tests
WPT_COREML_TENSOR_MODE=persistent make test-wpt-coreml
WPT_COREML_TENSOR_MODE=backings make test-wpt-coreml
WPT_COREML_FLOAT16=exact WPT_COREML_DEVICE=gpu make test-wpt-coreml   # Exact lowering, GPU request
```

CoreML has no PASS snapshots; failing trials are listed in
`tests/wpt_conformance/coreml_expected_failures.txt` and must be executed on every run. CI runs
the suite on macOS for every pull request.

## Known limits

The remaining expected failures come from the platform rather than from missing lowerings:

- Integer operations are computed in float32, so values near the int32 and all int64 extremes
  lose precision (`abs`, `clamp`, `relu`, `neg` on wide integer types).
- Tensors of rank 6 and above are rejected.
- Pooling has no dilation parameter; `maxPool2d` with `ceil` rounding and all-padding windows
  differs at the border.
- `pad` in `edge` and `reflection` mode is limited to two dimensions; `gatherND` above rank 5 and
  out-of-bounds gather and scatter indices follow CoreML's clamping rather than the WebNN text.
- The `shape` extension operation is not lowered.

See the [WPT conformance dashboard](https://rustnn.github.io/rustnn/wpt-conformance/) for the
current per-operation status.

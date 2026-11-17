# CoreML Backend

The `coreml` backend runs WebNN graphs through Apple CoreML on macOS and iOS. The converter
(`src/converters/coreml_mlprogram.rs`) emits an MLProgram, the MIL-based model format, and the
backend (`src/backends/coreml.rs`) compiles it once and keeps the compiled model for repeated
dispatch. The Objective-C bridge lives in `src/executors/coreml.rs` and `src/executors/coreml_shim.mm`.

## Requirements

- macOS 15, iOS 18, tvOS 18, watchOS 11 or visionOS 2, or newer. Every emitted model declares
  Core ML specification version 9 and the CoreML8 MIL opset (`ios18` in MIL text), the
  generation that has `MLState` and the blockwise constant operations; older systems refuse to
  load it.
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
   `MLModel` loads it from its URL. With
   [`CoremlOptions::compiled_bundle_cache`](#compiled-bundle-cache) set, the compiled
   `.mlmodelc` is kept on disk and the next build of the same graph loads it instead.
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

## Quantized constant weights

A graph can author a weight matrix as an explicit dequantization of constants: an `int8` or
`uint8` constant `[n, k]`, `cast` to `float16` or `float32`, an optional `sub` of a scalar zero
point, `reshape` to blocks `[n, g, k / g]`, `mul` by a constant scale `[n, g, 1]` and `reshape`
back to `[n, k]`. The converter keeps such a weight quantized. The weights blob holds the
original bytes, one scale per block and, for a nonzero zero point, one offset per block, and
the model declares a `constexpr_blockwise_shift_scale` constant that CoreML decompresses
itself, instead of evaluating the chain into a float copy of every weight while it compiles or
loads the model. Nothing has to be enabled, and a graph without such a chain converts as
before. [Converter internals](https://rustnn.github.io/rustnn/development/converters/#coreml)
lists what is recognized.

- The values are those of the chain: `(q - zero) * scale`, rounded once in its float type. On
  a CPU request, in memory and through the URL compiler, the outputs of a float16 language
  model were the same bits as with the literal chain (M4 Max, macOS 27.3).
- A table that a `gather` and an output projection `matmul(x, transpose(W))` share is stored
  once.
- A graph that [carries state](#carrying-recurrent-state-inside-the-model) reads the constant
  in `matmul` with `transpose_y`. A stateless graph keeps its `transpose` over the constant:
  on the in-memory route CoreML evaluates that `transpose` on every dispatch, as it did for
  the whole chain before, because a `matmul` reading the constant directly rounds float16
  results differently there.
- With `float16_precision = Exact`, a graph with float16 operands keeps the literal chain.

## Compiled-bundle cache

`MLModel.compileModel` is the slow step of a build on small devices: tens of seconds for a
language model on an A12 or A13, and the step whose memory peak kills a model before the
first prediction. `RustNNOptions::coreml.compiled_bundle_cache = Some(dir)` keeps the compiled
`.mlmodelc` of every graph that `MLGraphBuilder::build` compiles and loads it from `dir` on the
next build of the same graph, with `MLModel(contentsOf:configuration:)` and no compiler call.
The option defaults to `None` (no cache, today's behavior); rustnn chooses no location on its
own. On iOS and watchOS use a path under `Library/Caches`, which is what App Store review
expects for purgeable data.

What is cached is rustnn's compile step. Core ML's own per-device specialization of a loaded
model (the MLE5 program library) is cached by the OS separately and is not controlled here.

| Path | Content |
|---|---|
| `<dir>/<key>/bundle.mlmodelc` | The compiled bundle, copied from the compiler's output |
| `<dir>/<key>/manifest.json` | The key inputs by name, the crate version, the OS build, the write time and the bundle's byte count |
| `<dir>/<key>.partial-<pid>-<random>` | The staging directory of one writer while an entry is written; removed on publish or failure |

The key is a SeaHash over these inputs, each of which the manifest records:

- the graph: `GraphInfo::hash_identifier` over operands (names included), operations and every
  constant's bytes. The emitted MIL protobuf is not the input because its map fields encode
  from hash maps, so two conversions of one graph differ in byte order;
- the emitted weights blob and the emitted program's size. The converter emits constants in
  operand order, so the blob is the same bytes on every conversion of one graph;
- the compute-unit policy of the requested device and `float16_precision`;
- the compile route: `native` for a bundle from `MLModel.compileModel`, `rust-export` for a
  bundle the exporter wrote under the `RustExport` [compile route](#compile-routes), together
  with the version of the `mlmodelc-export` crate that wrote it;
- the carried-state pairs ([Carrying recurrent state](#carrying-recurrent-state-inside-the-model));
- the crate version, a hash of the CoreML converter sources and the compile shim computed by
  `build.rs`, and the OS version and build from `sysctl` (`kern.osproductversion` and
  `kern.osversion`, for example `27.2 (25C56)`), which do not change with the user's language.

Each writer fills a staging directory of its own and publishes it with one `rename`, so a
reader sees no entry, a partial entry (a writer that died before the manifest) or a complete
one, and concurrent writers of one key do not touch each other's files: the first rename wins
and the others discard their copy. A staging directory is removed by a later writer only when
no process has the id in its name. A
partial entry is a miss and is replaced; a complete entry whose bundle fails to load, or whose
manifest records other key inputs, is removed and rebuilt once. The cache never fails a build:
every failure is logged at warn level and the graph compiles as without a cache. rustnn never
evicts entries; each is about the size of the graph's weights, and deleting `dir` or any entry
is safe at any time.

Only the URL compile route yields a bundle. With the cache enabled, a graph that would load
as an in-memory asset therefore compiles from a URL instead (`CoremlLoadRoute::CompiledUrl`),
with two exceptions that keep the route they have without the cache and report the cache as
disabled when that route is the in-memory one:

- input-free programs. Constant copies can crash BNNS from a URL, and other input-free
  programs are evaluated by the URL compiler itself, which returns 0 for WPT's saturating
  uint8 `clamp` on iOS and tvOS 26 and 27;
- float16 graphs under the `Native` float16 lowering, where URL compilation is known to
  return other values than the in-memory asset (WPT's float16 `l2Pool2d` with `layout=nhwc`
  returns `inf` where 289 is expected). Use `float16_precision = Exact` to cache float16
  graphs: that lowering compiles from a URL with or without the cache.

With the `RustExport` compile route every load is already a URL load, so the cache keeps the
exported bundle of every graph except the constant-copy programs that route loads in memory.
That includes float16 graphs under the `Native` lowering, whose values on that route are the
URL route's with or without the cache. A hit reports `CoremlLoadRoute::RustExport`. Under
`Auto`, a bundle the exporter wrote after a failed native compile is not kept, so the
`native` key never holds an exported bundle: the build reports `Error` and the next build
compiles again.

For every other graph, enabling the cache changes the compile route from in-memory to URL.
The CoreML WPT corpus passes with the cache enabled under this rule, on a miss and on a hit;
`WPT_COREML_BUNDLE_CACHE=<dir>` runs the corpus with the option set.
`CoremlLoadDiagnostics::cache` reports one
`CacheOutcome` per build: `Disabled`, `Hit`, `Miss`, or `Error(message)` when the cache was
enabled but could not serve or store the entry.

```rust,ignore
let mut options = RustNNOptions::default();
options.coreml.compiled_bundle_cache = Some(caches_dir.join("rustnn-coreml"));
let mut context = MLContext::create(
    &MLContextOptions::new(MLPowerPreference::Default, false)
        .with_rustnn_backend_hint(Backend::Coreml)
        .with_rustnn_options(options),
)?;
// ... build the graph ...
if let Some(LoadDiagnostics::Coreml(diagnostics)) = graph.rustnn_load_diagnostics() {
    log::info!("compiled-bundle cache: {:?}", diagnostics.cache);
}
```

Measured on the fork's native source-session path with the same staging and rename design
(2026-10-20, SmolLM2 bundles, app v134; re-measure on this path): cold to warm load 16.4 s to
1.3 s on an A13 (iPhone SE 2nd generation), 22.4 s to 1.7 s on an A12 (iPhone XS Max), 39.5 s
to 2.3 s on an A10X (Apple TV 4K), 43.9 s to 1.1 s on an Apple Watch Series 10, with identical
transcripts. `tests/test_coreml_bundle_cache.rs` checks the miss-then-hit sequence, identical
results, the rebuild of partial, corrupt and foreign entries and an unwritable directory on
macOS.

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

The values are not the same on every route. An exported bundle is always loaded from disk, so
`RustExport` returns what the URL route (`CompiledUrl`) returns. For a graph that `Native`
loads as an in-memory asset, that can differ from the `Native` result. The known cases are
float16 graphs under the default `Native` float16 lowering: WPT's float16 `l2Pool2d` trial
with `layout=nhwc` passes in memory and returns `inf` where 289 is expected from a URL, and
on some hosts the float16 `gru` and `lstm` trials fail the same way. Use
`float16_precision = Exact` for float16 graphs on this route; that lowering is qualified
through URL compilation and passes the WPT corpus on it. The WPT harness selects the route
with `WPT_COREML_COMPILE_ROUTE=auto|native|rust-export`.

One class of program never goes through the exporter: an input-free program whose outputs
are all copies of constants (for example `transpose` of a constant with default options).
BNNS can crash on such a program when it is loaded from any URL, so `RustExport` loads it as
an in-memory asset and reports `CoremlLoadRoute::InMemoryAsset`, and the `Auto` fallback does
not retry it through the exporter. watchOS has no in-memory specification loader
(`MLModelAsset` creation from specification data is unavailable there), so `build` fails for
these programs on watchOS with a `CoremlRuntimeFailed` error instead of crashing the process.

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

## Carrying recurrent state inside the model

A decode step of a recurrent or attention model takes its state (recurrent state,
convolution history, KV cache) as graph inputs and returns the next state as graph outputs.
By default every step crosses the host boundary with those arrays. With
`RustNNOptions::coreml.carried_state_pairs` (explicit `(input, output)` name pairs, the
`kvCachePairs` of the WebNN stateful-model explainer) or
`RustNNOptions::coreml.carry_state_automatically` (pairs by naming convention: `X`/`next_X`,
`past_X`/`present_X`, `X_in`/`X_out`; see `rustnn::converters::coreml_state`), the converter
keeps each pair inside the compiled model as a CoreML `MLState`: the input is read with
`read_state`, the output is stored with `write_state`, and the pair leaves the model's feature
description. Both switches default to off, and a graph without pairs lowers as before.
`MLState` needs iOS 18, macOS 15 or watchOS 11, which is what every model rustnn emits
already requires ([Requirements](#requirements)).

Rules and consequences:

- A pair needs the same `float16` or `float32` data type and the same static shape on both
  operands. CoreML stores every state as `float16`; a `float32` pair is cast after the read and
  before the write inside the model, so the carried value is rounded to `float16` once per step.
  Integer or dynamically shaped pairs are rejected (explicit) or skipped (automatic).
- The paired names are not dispatch bindings any more: `MLGraph::input_descriptors` and
  `output_descriptors` omit them and `dispatch` rejects tensors bound under those names.
- Every state starts as zeros when the graph is built. `MLGraph::rustnn_reset_state` replaces
  the state with fresh zeros, which is how a new conversation starts; an initial state cannot
  be supplied through the pair. Only one state set exists per graph.
- `CoremlLoadDiagnostics::carried_state_pairs` lists the pairs the model carries, so an
  application can report how many of its inputs became states.
- A state belongs to one MLProgram. A graph whose lowering needs Pipeline children (the
  rounding boundaries of [Float16 precision](#float16-precision)) cannot carry state, and
  building it with pairs is an error.
- A stateful program always compiles from a URL (`CoremlLoadRoute::CompiledUrl`, or
  `RustExport`): Core ML's in-memory asset refuses to plan one (error -14 on macOS 27.3).
- Every compile route carries state. The exported bundle (`coreml-export`, `mlmodelc-export`
  0.2 or newer) declares `state<tensor<fp16, ...>>` function inputs, emits `read_state` /
  `write_state`, and lists the states under `stateSchema` in `metadata.json`.
- The pairs are part of the [compiled-bundle cache](#compiled-bundle-cache) key, so the
  stateful and the stateless build of one graph are separate entries.

## Testing

```bash
make test-wpt-coreml              # full WPT suite, expected failures are non-fatal
make test-wpt-coreml-report       # same, plus the JSON report
make wpt-sync-coreml              # regenerate tests/wpt_conformance/coreml_expected_failures.txt
make test-coreml                  # unit and integration tests
WPT_COREML_TENSOR_MODE=persistent make test-wpt-coreml
WPT_COREML_TENSOR_MODE=backings make test-wpt-coreml
WPT_COREML_FLOAT16=exact WPT_COREML_DEVICE=gpu make test-wpt-coreml   # Exact lowering, GPU request
WPT_COREML_COMPILE_ROUTE=rust-export WPT_COREML_FLOAT16=exact \
  make test-wpt-coreml WPT_COREML_FEATURES=coreml-runtime,coreml-export   # Rust export route
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

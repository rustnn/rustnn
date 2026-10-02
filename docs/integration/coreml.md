# CoreML Backend

The `coreml` backend runs WebNN graphs through Apple CoreML on macOS and iOS. The converter
(`src/converters/coreml_mlprogram.rs`) emits an MLProgram, the MIL-based model format, and the
backend (`src/backends/coreml.rs`) compiles it once and keeps the compiled model for repeated
dispatch. The Objective-C bridge lives in `src/executors/coreml.rs` and `src/executors/coreml_shim.mm`.

## Requirements

- A CoreML version that accepts MLProgram models (macOS 13 or newer). The native Rust
  backend also supports iOS; mobile validation uses iOS 18 as its deployment target.
- tvOS is not enabled: the published `objc` dependency selects the wrong message ABI
  there. watchOS execution is not enabled by these target gates either.
- The `coreml-runtime` Cargo feature. On Linux and Windows the feature compiles to shims whose
  calls always fail, so `cargo check --features coreml-runtime` works everywhere but the backend
  is never selected on those platforms.
- Xcode command line tools for the Objective-C++ shim (`build.rs` compiles it with `cc`).

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

1. The converter lowers the graph to a MIL program. Rank-0 operands are promoted to `[1]` at the
   model boundary, comparison and logical results are produced as `uint8`, and reductions with
   empty axes, `resample2d` on arbitrary axis pairs and the stable `reduceLogSumExp` form are
   lowered explicitly because MIL has no direct equivalent.
2. Float16 weights are written as a separate weights blob (`ConvertedGraph::weights_data`), and
   the model is packaged as an in-memory asset together with that blob. Graphs for which the
   in-memory compiler is known to misbehave (`gather` with a rank-0 index) are written to a
   temporary directory and compiled from its URL instead (`supports_in_memory_asset`).
3. `MLGraphBuilder::build` compiles the model with `MLModel` and keeps the compiled model;
   `dispatch` binds `MLMultiArray`s over the tensor storage and runs a prediction.
4. The legacy CLI path (`--convert coreml --run-coreml`) tries the compute-unit configurations in
   turn and reports each attempt; `--coreml-compiled-output <dir>` stores the compiled
   `.mlmodelc` for reuse.

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

## Testing

```bash
make test-wpt-coreml              # full WPT suite, expected failures are non-fatal
make test-wpt-coreml-report       # same, plus the JSON report
make wpt-sync-coreml              # regenerate tests/wpt_conformance/coreml_expected_failures.txt
make test-coreml                  # unit and integration tests
WPT_COREML_TENSOR_MODE=persistent make test-wpt-coreml
WPT_COREML_TENSOR_MODE=backings make test-wpt-coreml
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

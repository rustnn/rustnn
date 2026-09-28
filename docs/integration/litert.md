# LiteRT Backend

The `litert` backend runs WebNN graphs with LiteRT (the runtime formerly called TensorFlow
Lite). The converter (`src/converters/litert.rs`) writes a TFLite flatbuffer, and the backend
(`src/backends/litert.rs`) compiles it with the LiteRT compiled-model API from the `litert-sys`
crate and runs it with the CPU, GPU or NPU accelerator.

## Requirements

- The `litert-runtime` Cargo feature.
- `flatc` on `PATH` at build time: `build.rs` compiles `protos/tflite/schema.fbs` into the Rust
  flatbuffer bindings. CI downloads the flatbuffers release binary; locally use
  `brew install flatbuffers`, the Linux or Windows release zip, or your package manager.
- The LiteRT shared libraries. The `litert-sys` build script downloads a pinned release into
  `~/.cache/litert-sys/<version>/<target triple>/` on the first build. `LITERT_LIB_DIR` points at
  a local build instead, `LITERT_CACHE_DIR` moves the cache and `LITERT_NO_DOWNLOAD` fails
  instead of downloading (air-gapped CI).
- At run time the libraries must be on the loader path. `make test-wpt-litert` sets
  `LD_LIBRARY_PATH` to the cache directory; do the same for your own binaries.

## Selection and devices

`LiteRtContext::list_devices` reports one device per class. Not accelerated selects the CPU
device; accelerated with `Default` or `HighPerformance` selects `Gpu`, with `LowPower` `Npu`.
The GPU and NPU configurations always include the CPU accelerator as fallback for operations the
delegate does not support.

## How a graph runs

1. Operands that carry NCHW semantics (convolution, pooling, normalization inputs and filters)
   are transposed to NHWC in the flatbuffer; `is_spatial_op` and the `transpose_*_to_ohwi`
   helpers in `src/backends/litert.rs` define the mapping, and the backend transposes filter
   data and boundary tensors at run time.
2. Comparison and logical results are produced as TFLite `BOOL` tensors and converted to
   `uint8` at the graph boundary.
3. `MLGraphBuilder::build` creates the compiled model once; `dispatch` copies tensors into
   LiteRT tensor buffers and runs it.

## Operation and data type policy

An operation the converter cannot lower fails the conversion with
`unsupported operation: <operation>`, which is how the recurrent operations (`gru`,
`gruCell`, `lstm`, `lstmCell`) and `scatterElements` with non-constant indices are reported.
`datatype_to_tflite` rejects data types the TFLite schema lacks, which is `uint4` today; the
WPT harness skips trials whose dtypes its own list omits.

Two limits come from the runtime rather than from the schema. No operand can be `uint64`: the
LiteRT model loader has no tensor type for it, so such a graph fails to compile with
"Element type not currently supported". And `convTranspose2d` with dilations other than 1 is
rejected by the converter, because `TRANSPOSE_CONV` carries no dilation factors and the
option would otherwise be dropped silently.

## Testing

```bash
make test-wpt-litert              # WPT suite; PASS snapshots plus an expected-failure list
make wpt-sync-litert              # regenerate snapshots and litert_expected_failures.txt
cargo test --test test_litert_execution --features litert-runtime
```

CI runs the LiteRT WPT suite on Linux for every pull request as a non-blocking job. The
generated [operator support report](https://rustnn.github.io/rustnn/development/backend-operator-support/) lists the
operations the converter lowers; the dashboard shows the per-case results.

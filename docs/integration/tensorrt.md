# TensorRT-RTX Backend

The `trtx` backend runs WebNN graphs on NVIDIA RTX GPUs through
[TensorRT for RTX](https://developer.nvidia.com/tensorrt-rtx) using the
[trtx](https://github.com/rustnn/trtx-rs) crate. Graphs are lowered directly into a TensorRT
network by `src/converters/trtx.rs` (with `trtx_gru.rs`, `trtx_lstm.rs` and `trtx_rnn.rs` for the
recurrent operations); there is no ONNX intermediate. The backend itself is
`src/backends/trtx.rs`.

## Features

| Feature | Use |
|---|---|
| `trtx-runtime` | The backend and converter; pulls in `trtx`, `cudarc` and `zstd-cache-compression` |
| `trtx-converter` | Builds serialized engines through the NVIDIA SDK without rustnn dispatch |
| `trtx-enterprise` | `trtx-runtime` linked against full TensorRT 10 (`nvinfer`, `nvonnxparser`) instead of the TensorRT-RTX libraries; RTX-only features such as CUDA graphs and the JIT runtime cache are compiled out. For validation only |
| `zstd-cache-compression` | Compresses the on-disk caches; enabled by `trtx-runtime` |

## Requirements

- An NVIDIA RTX GPU with a current driver. The CUDA driver API is loaded at run time through
  `cudarc`; the CUDA toolkit is not required. Device enumeration checks for the driver library
  before initializing CUDA. If the library is missing or initialization fails, it reports no
  TensorRT-RTX devices so automatic selection can try another backend.
- The TensorRT-RTX 1.7 library. The `trtx` crate loads it by name (`tensorrt_rtx_1_7`), so the
  SDK's `lib` directory (Linux) or `bin` directory (Windows) must be on `LD_LIBRARY_PATH` or
  `PATH`, or the application calls `rustnn::backends::trtx::dynamically_load_tensorrt(Some(path))`
  before creating a context. When the library cannot be loaded, backend selection skips
  TensorRT (on Windows the underlying error is `LoadLibraryExW` code 126).
- Build time: `trtx-sys` generates bindings from the TensorRT headers with autocxx and needs
  libclang (`LIBCLANG_PATH` when it is not found automatically). `TENSORRT_INCLUDE_DIR`,
  `TENSORRT_LIB_DIR` and `TENSORRT_SDK_DIR` override the header and library locations.

The Windows steps are in [Windows TensorRT-RTX Setup](https://rustnn.github.io/rustnn/integration/windows-tensorrt-setup/).

## Selection

With `accelerated = true` and the `Default` or `HighPerformance` power preference, TensorRT-RTX
is the first choice when the feature is enabled and a CUDA device is present. Force it with
`MLContextOptions::with_rustnn_backend_hint(Backend::Trtx)` or pick a GPU with
`with_rustnn_device_hint(BackendDevice::Trtx { cuda_device_idx })`. `RUST_LOG=info` prints the
devices found and `Backend selected: Trtx { cuda_device_idx: 0 }`.

## How a graph runs

1. `TrtxConverter::build_network` translates the `GraphInfo` into a TensorRT network. Constants
   become weights and are marked refittable, except constants that TensorRT bakes into the
   engine (for example scalars cast to another type) and constants without a consumer.
2. The engine is built with `kREFIT_INDIVIDUAL` and `kSTRIP_PLAN`, and with TF32 disabled so
   that float32 graphs compute in full float32 precision (the WPT references are strict IEEE
   float32). Float16 operations can promote intermediate values to float32 for numerical
   stability; `reduceLogSum` keeps its sum and logarithm in float32 before casting the result
   back to float16.
3. The stripped engine is stored in the engine cache and then refitted with the actual weights.
   Loading a cached engine skips the build entirely. Engines are cached when every baked
   constant is small. The bytes of the baked constants are part of the cache key; refittable
   weights never end up in the cache.
4. Tensors are CUDA device buffers. `dispatch` binds them by name, captures a CUDA graph per
   distinct binding set and replays it on later calls. `write_tensor` synchronizes the stream
   before it returns.

## URL constants (prototype)

`MLGraphBuilder::rustnn_constant_from_url` records a constant by the URL and byte range of its
bytes instead of a host buffer (the pull model from the discussion in
[webnn#901](https://github.com/webmachinelearning/webnn/issues/901)); the
[`weights` module documentation](https://rustnn.github.io/rustnn/api/rustnn/weights/)
owns the rules. On this backend `build` runs these steps:

1. It skips the URL constants without a consumer, which are never fetched, and fetches the URL
   constants that the converter bakes into the engine. Their values are needed for the engine
   cache key and to build the network. This fetch runs one constant after another on the
   calling thread.
2. It starts worker threads (`UrlFetchOptions::threads`) that fetch the byte ranges of the
   remaining URL constants. `UrlFetchOptions::max_inflight_bytes`, the byte bound, limits the
   fetched bytes that are held in host memory and not yet refitted. It does not apply to the
   fetch of step 1.
3. It loads the engine from the engine cache, or on a miss builds it from a network in which
   the remaining constants are constant layers with null weights (placeholders). The fetch
   runs meanwhile. An engine built from placeholders has its own cache key, and that key
   contains no URL.
4. It refits the engine from the fetched buffers: once at the end, or in batches that are
   freed after each refit when `max_inflight_bytes` or `UrlFetchOptions::bind_batch_bytes`
   is set.

The fetch options are set for a context with `RustNNOptions::url_fetch`; every builder of the
context starts with them, and `rustnn_set_url_fetch_options` replaces them for one builder.
No environment variable is read.

`build` returns after all weights are bound. Without a byte bound the whole fetch can overlap
the engine build. With a bound the workers fetch up to the bound during the build and continue
when the refit starts to free buffers. A failed fetch of a constant from step 2 is reported
after the engine was built or loaded. In a graph with quantize or dequantize operations every
constant is baked, so all of its URL constants are fetched in step 1. The byte bound is best
effort while TensorRT reports missing weights for a batch: the backend then keeps the fetched
buffers and logs a warning.

Placeholders need TensorRT-RTX 1.7 or later.

`examples/url_weights_bench.rs` compares host memory and the time to the first inference with
constants passed as host buffers (see the
[examples page](https://rustnn.github.io/rustnn/user-guide/examples/#url-constants-benchmark));
`tests/url_weights.rs` holds the tests.

## Options

Set through `RustNNOptions::trtx` (`TrtxOptions`):

The [TrtxOptions field documentation](https://rustnn.github.io/rustnn/api/rustnn/mlcontextoptions/struct.TrtxOptions.html)
owns defaults and option semantics. For example:

```rust
use rustnn::mlcontext::{Backend, MLContextOptions, MLPowerPreference, RustNNOptions};

let mut options = RustNNOptions::default();
options.trtx.cuda_graphs = false;
let context_options = MLContextOptions::new(MLPowerPreference::HighPerformance, true)
    .with_rustnn_backend_hint(Backend::Trtx)
    .with_rustnn_options(options);
```

## Caches

| Category | Location | Content |
|---|---|---|
| `trtx` | `<cache_dir>/rustnn/trtx` | Stripped engines, keyed by the graph topology and a hash of the converter sources |
| `trtx-jit` | `<cache_dir>/rustnn/trtx-jit` | The global TensorRT runtime cache, serialized after builds |

`<cache_dir>` is `~/.cache` on Linux, `~/Library/Caches` on macOS and `%LOCALAPPDATA%` on
Windows. Files are zstd compressed and written through a temporary file plus rename. Deleting
either directory is safe; the next build recreates the entries. A stale engine cannot survive a
converter change because the converter source hash is part of the key.

## Environment variables

| Variable | Effect |
|---|---|
| `RUSTNN_TRTX_LOG_VERBOSITY` | TensorRT logger level: `internal_error`, `error`, `warning`, `info`, `verbose` |
| `TRTX_JSON_DUMP_PATH` | Directory that receives a layer JSON dump per built engine |
| `RUST_LOG` | rustnn logging; `debug` shows cache keys and refit details |

## Testing

```bash
make test-wpt-trtx                                        # WPT suite, requires a GPU
WPT_BACKEND=trtx make test-wpt-op OP=conv2d               # one operation
cargo test --test test_trtx_execution --features trtx-runtime
cargo test --test url_weights --features trtx-runtime,url-weights-http -- --test-threads 1  # URL constants
make wpt-sync-trtx                                        # regenerate PASS snapshots after converter changes
cargo check -F trtx-runtime --all-targets                 # what CI runs
```

There is no GPU runner in CI. The TensorRT PASS snapshots under `tests/snapshots/` are
regenerated by contributors with a GPU and reviewed in the pull request. Debug builds compile an
engine per WPT trial and take tens of minutes for the full suite; use `make test-wpt-op` while
iterating.

## Limitations

- The `shape` extension operation is not lowered.
- Data type restrictions of TensorRT (for example integer element-wise operations) are bridged
  with casts in the converter; cases that still fail show up as missing PASS snapshots and on the
  WPT dashboard.
- Engines are specific to the GPU architecture; the cache is per machine.

# CANN Backend (OpenHarmony)

The `cann` backend targets Huawei NPUs on OpenHarmony devices through HiAI, using the
[hiai-rs](https://github.com/rustnn/hiai-rs) crate. The converter
(`src/converters/cann.rs`) encodes the graph through the HiAI IR adapter into an offline model,
and the backend (`src/backends/cann.rs`) runs it in a HiAI session.

## Features

| Feature | Use |
|---|---|
| `cann-runtime` | The real backend; only builds for `aarch64-unknown-linux-ohos` with the HiAI DDK |
| `cann-runtime-mock` | Compiles the converter and backend without a device; graph builds succeed, dispatch fails. Used by `cargo test --lib --features cann-runtime-mock` in CI |

CANN is never selected automatically: request it with
`MLContextOptions::with_rustnn_backend_hint(Backend::Cann)`. The backend reports one `Npu` device.

## Cross-compiling

Tools:

- The Rust target: `rustup target add aarch64-unknown-linux-ohos`.
- The OpenHarmony native SDK; `OHOS_SDK_NATIVE` points at its `sdk/native` directory. The
  Makefile derives the clang, `llvm-ar`, sysroot and linker settings from it (`CANN_CROSS_ENV`),
  and `.cargo/config.toml` adds the `lld`, `libunwind` and `compiler-rt` link arguments for the
  target.
- The Huawei CANN Kit DDK; `CANN_DDK` points at its `ddk` directory. The device libraries
  (`libhiai*.so`) come from `ai_ddk_lib/lib64`.
- `hdc`, the OpenHarmony device connector from DevEco Device Tool, for pushing binaries.

```bash
export OHOS_SDK_NATIVE=/path/to/OpenHarmony/<version>/sdk/native
export CANN_DDK=/path/to/CANN-Kit-next/ddk
make validate-cann-env            # checks target, SDK and DDK variables
make cann-build                   # release build with --features cann-runtime for the OHOS target
```

## Testing on a device

`tests/test_cann_execution.rs` runs a set of small graphs through the complete pipeline on the
device. It is compiled only for the OpenHarmony target.

```bash
make cann-device-test             # cross-compiles the test binary, then runs the helper
```

The helper (`scripts/ohos-test-helper.sh`) pushes the newest test binary and the HiAI
libraries to `/data/local/tmp/cann-test` with `hdc file send` and executes it on the device;
it prints `[OK]` and `[FAIL]` lines per step.

## Supported operations

The converter accepts the operations listed by `is_supported_op` in `src/converters/cann.rs`
and rejects everything else at build time. The CANN column of the generated
[operator support report](../development/backend-operator-support.md) is the current list.
There is no WPT run for CANN; conformance is checked with the device test.

## Conversion without a device

`cargo run --features cann-runtime -- graph.webnn --convert cann --convert-output model.cann`
writes the encoded model (`make cann` wraps it for the sample graph). With `cann-runtime-mock`
the encoder returns a placeholder, which is enough to check operator coverage on a desktop.

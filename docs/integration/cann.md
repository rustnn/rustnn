# CANN Backend (OpenHarmony)

The `cann` backend targets Huawei NPUs on OpenHarmony devices through HiAI, using the
[hiai-rs](https://github.com/rustnn/hiai-rs) crate. The converter
(`src/converters/cann.rs`) encodes the graph through the HiAI IR adapter into an offline model,
and the backend (`src/backends/cann.rs`) runs it in a HiAI session.

## Features

| Feature | Use |
|---|---|
| `cann-runtime` | Real device execution requires `aarch64-unknown-linux-ohos` and the HiAI DDK; CI also checks compilation on the host |
| `cann-runtime-mock` | Device-free backend validation and placeholder conversion; no DDK dependency, dispatch fails |
| `cann-converter` | Builds offline models using the HiAI DDK, without rustnn execution |

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
[operator support report](https://rustnn.github.io/rustnn/development/backend-operator-support/) is the current list.
WPT conformance runs on the device as well: `make test-wpt-cann` cross-compiles the harness
with the corpus embedded (`wpt-embed-corpus` feature), pushes it with the HiAI libraries over
`hdc`, runs it single-threaded and retrieves the JSON report; `make wpt-sync-cann` regenerates
`tests/wpt_conformance/cann_expected_failures.txt`. CI checks the host build and mock library tests; device WPT requires the OpenHarmony hardware.

## Conversion without execution

Enable `cann-converter` with default features disabled and call `CannConverter` or
`ConverterRegistry`. Conversion compiles a GE graph through the HiAI DDK; the DDK is
required even when `cann-runtime` is disabled.

For device-free validation use `cann-runtime-mock` with defaults disabled. It does not
enable `cann-converter` or the DDK. If `cann-converter` is also enabled, conversion uses
the native adapter; execution still requires `cann-runtime`.

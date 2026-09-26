# Windows TensorRT-RTX Setup

Steps to build and run rustnn with the TensorRT-RTX backend on Windows 10 or 11 (x64).

## Install

1. An NVIDIA driver for an RTX GPU. `nvidia-smi` must list the device.
2. The TensorRT-RTX 1.6 SDK from https://developer.nvidia.com/tensorrt-rtx (NVIDIA developer
   account required). Extract it, for example to `C:\TensorRT-RTX-1.6`, and add its `bin`
   directory to `PATH`: the `trtx` crate loads `tensorrt_rtx_1_6.dll` by name. Alternatively
   call `rustnn::executors::trtx::dynamically_load_tensorrt(Some(path))` at startup.
3. Visual Studio Build Tools with the "Desktop development with C++" workload (linker and
   Windows SDK).
4. LLVM for libclang, which autocxx uses while building `trtx-sys`: `winget install LLVM.LLVM`,
   and `LIBCLANG_PATH=C:\Program Files\LLVM\bin` when cargo cannot find it.
5. `protoc`: `winget install Google.Protobuf`.
6. Rust from https://rustup.rs; `rust-toolchain.toml` selects the pinned toolchain.
7. `git config --system core.longpaths true` before cloning; paths exceed 260 characters.
8. Optional: Node.js for the WPT tests, and ONNX Runtime for the `onnx-runtime` feature
   (`make onnxruntime-download`, then `ORT_DYLIB_PATH` as in [Getting Started](../user-guide/getting-started.md)).
   Do not rely on the `onnxruntime.dll` in `System32`; it is too old for the `ort` crate.

The CUDA toolkit is not required for running: the CUDA driver API is loaded at run time.

## Build

```powershell
cargo check -F trtx-runtime --all-targets
cargo build --release --features trtx-runtime
```

## Verify

```powershell
$env:RUST_LOG = "info"
cargo run --release --features trtx-runtime,onnx-runtime -- examples\sample_graph.webnn --convert onnx --run-trtx
```

The log lists the CUDA devices found and `Backend selected: Trtx { cuda_device_idx: 0 }`. In an
application, `MLContextOptions::new(MLPowerPreference::HighPerformance, true)` selects
TensorRT-RTX when the feature is enabled; `context.rustnn_backend()` confirms it.

## WPT conformance

```bash
# Git Bash
export PATH="/c/TensorRT-RTX-1.6/bin:$PATH"
export ORT_DYLIB_PATH=/c/git/rustnn/target/onnxruntime/onnxruntime-win-x64-1.29.0/lib/onnxruntime.dll
make test-wpt-trtx
WPT_BACKEND=trtx make test-wpt-op OP=gemm
```

Set `ORT_DYLIB_PATH` even for TensorRT runs: the harness probes the ONNX Runtime backend too,
and a wrong DLL poisons the ONNX Runtime initialization for the rest of the process. Test output
is fully buffered when redirected to a file; watch the console or wait for the process to exit.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `LoadLibraryExW` error 126 when creating a context; TensorRT not selected | `tensorrt_rtx_1_6.dll` is not on `PATH`. Add the SDK `bin` directory or load the library explicitly |
| `Failed to load ONNX Runtime dylib: BadVersion { version_str: "1.17.1" }` | `ORT_DYLIB_PATH` is unset and the `System32` DLL was picked up. Point it at the downloaded release |
| libclang not found while compiling `trtx-sys` | Install LLVM and set `LIBCLANG_PATH` |
| Results look stale after a converter change | Delete the engine caches `%LOCALAPPDATA%\rustnn\trtx` and `%LOCALAPPDATA%\rustnn\trtx-jit` |
| `Access is denied` from `cargo fmt` or the linker | A test binary is still running; wait for it to finish |
| Path too long errors during clone or build | `git config --system core.longpaths true`, or clone to a short path such as `C:\git` |

## Without a GPU

`cargo check -F trtx-runtime --all-targets` compiles the backend and is what CI runs. The
`trtx-runtime-mock` feature builds the converter against the mock `trtx` API for API-level
checks; the execution tests are excluded in mock mode.

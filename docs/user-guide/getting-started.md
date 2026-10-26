# Getting Started

rustnn is a Rust crate. This page goes from an empty project to a graph that runs on a backend,
then shows how to load graphs from files and how to use the command line tool.

## Requirements

| Requirement | Needed for |
|---|---|
| Rust 1.97 or newer (edition 2024); `rust-toolchain.toml` pins it for the repository itself | the crate |
| `protoc`, the Protocol Buffers compiler, on `PATH` | `build.rs` compiles the CoreML schemas, and `webnn-onnx-utils` the ONNX one (default `onnx` feature) |
| `flatc` | the `litert-runtime` feature only (TFLite schema) |
| Node.js | the WPT conformance tests only |
| A backend library | execution; see [Backends](https://rustnn.github.io/rustnn/api/rustnn/backend_selection/). The ONNX Runtime shared library works on every platform and is the usual starting point |

On Windows run `git config --system core.longpaths true` before cloning; the repository
contains paths longer than 260 characters.

## Add the crate

The WebNN `MLContext` API is published on crates.io from version 0.6.0 (the 0.5.x releases were
the earlier converter and loader crate). Add the crate with a backend feature:

```toml
[dependencies]
rustnn = { version = "0.6", features = ["onnx-runtime"] }
```

To follow unreleased changes on `main`, use the git dependency instead:
`rustnn = { git = "https://github.com/rustnn/rustnn", features = ["onnx-runtime"] }`.

Features select backends and go on this dependency line. Without a backend feature the crate
validates and converts graphs but cannot execute them. The full feature list is in the crate
documentation (`make docs-api` writes it to `target/doc/rustnn/index.html`; the site publishes
it under `/api/`) and in [Backends](https://rustnn.github.io/rustnn/api/rustnn/backend_selection/).

## Provide ONNX Runtime

The `onnx-runtime` feature loads the ONNX Runtime shared library at run time from the path in
`ORT_DYLIB_PATH`. The `ort` crate rustnn is built against requires ONNX Runtime 1.29 or newer.
Set the variable to select the intended library; otherwise the platform loader searches for
`libonnxruntime.so`, `libonnxruntime.dylib`, or `onnxruntime.dll`. rustnn loads it explicitly
with `ort::init_from`, so missing or incompatible libraries return an initialization error
and backend discovery skips ONNX Runtime. The environment or its initialization failure is
cached for the process lifetime; restart after changing the library path.

Download the pinned release. From the repository root with `make` installed:

```bash
make onnxruntime-download          # into target/onnxruntime/
```

Without `make`, fetch the same archive from the ONNX Runtime GitHub release
`v1.29.0` and unpack it into `target/onnxruntime/`; the archive names are
`onnxruntime-linux-x64-1.29.0.tgz`, `onnxruntime-osx-arm64-1.29.0.tgz` and
`onnxruntime-win-x64-1.29.0.zip`:

```powershell
# Windows PowerShell
New-Item -ItemType Directory -Force target\onnxruntime | Out-Null
Invoke-WebRequest https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/onnxruntime-win-x64-1.29.0.zip -OutFile target\onnxruntime\ort.zip
Expand-Archive target\onnxruntime\ort.zip -DestinationPath target\onnxruntime
```

Then point the variable at the library:

```bash
# Linux
export ORT_DYLIB_PATH=$PWD/target/onnxruntime/onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so.1.29.0
# macOS (Apple Silicon)
export ORT_DYLIB_PATH=$PWD/target/onnxruntime/onnxruntime-osx-arm64-1.29.0/lib/libonnxruntime.1.29.0.dylib
# Windows (Git Bash)
export ORT_DYLIB_PATH=$PWD/target/onnxruntime/onnxruntime-win-x64-1.29.0/lib/onnxruntime.dll
```

```powershell
# Windows PowerShell
$env:ORT_DYLIB_PATH = "$PWD\target\onnxruntime\onnxruntime-win-x64-1.29.0\lib\onnxruntime.dll"
```

## First graph

Follow the [crate example](https://rustnn.github.io/rustnn/api/rustnn/#example), which creates
a context, records `relu(x + 1)`, compiles a graph and dispatches tensors. The example lives
in `src/lib.rs`; API examples are maintained with the code.

## Load a graph from a file

See [Graph files and weights](../reference/graph-files.md) for the shared save/load/compile
workflow and external-weight format links.

## Convert a stored graph

Enable the format's converter feature and disable default features if execution is not
needed, for example `default-features = false, features = ["litert-converter"]`.
The registry exposes only enabled converters:

```rust,ignore
let graph = rustnn::load_graph_from_path("examples/sample_graph.webnn")?;
let converted = rustnn::ConverterRegistry::with_defaults().convert("litert", &graph)?;
std::fs::write("model.tflite", &converted.data)?;
```

Formats with external weights also return `ConvertedGraph::weights_data`; retain this
sidecar as described in the [converter guide](../development/converters.md).
See the [crate feature table](https://rustnn.github.io/rustnn/api/rustnn/#cargo-features)
for conversion and execution requirements.

## Next steps

- [Rust API reference](https://rustnn.github.io/rustnn/api/rustnn/): types, builder conventions and error types.
- [Backends](https://rustnn.github.io/rustnn/api/rustnn/backend_selection/): selection rules and per-backend requirements.
- [Examples](examples.md): the example programs and short recipes.
- [Flexible Input Shapes](../development/flexible-input-shapes.md): dynamic tensor workflow.

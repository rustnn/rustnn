# Getting Started

rustnn is a Rust crate. This page goes from an empty project to a graph that runs on a backend,
then shows how to load graphs from files and how to use the command line tool.

## Requirements

| Requirement | Needed for |
|---|---|
| Rust 1.97, pinned in `rust-toolchain.toml` (rustup installs it on the first build) | the crate (edition 2024) |
| `protoc`, the Protocol Buffers compiler, on `PATH` | `build.rs` compiles the ONNX and CoreML schemas |
| `flatc` | the `litert-runtime` feature only (TFLite schema) |
| Node.js | the WPT conformance tests only |
| A backend library | execution; see [Backends](backends.md). The ONNX Runtime shared library works on every platform and is the usual starting point |

On Windows run `git config --system core.longpaths true` before cloning; the repository
contains paths longer than 260 characters.

## Add the crate

The WebNN API described on this site is on the `main` branch and not yet published. The
`rustnn` crate on crates.io (0.5.x) is the earlier converter and loader crate without
`MLContext`; the docs.rs pages describe that release. Use the git dependency until the next
publish:

```toml
[dependencies]
rustnn = { git = "https://github.com/rustnn/rustnn", features = ["onnx-runtime"] }
```

Features select backends. Without one the crate validates and converts graphs but cannot
execute them. The full feature list is in the crate documentation (`make docs-api` in a clone
writes it to `target/doc/rustnn/index.html`; the site publishes it under `/api/`) and in
[Backends](backends.md).

## Provide ONNX Runtime

The `onnx-runtime` feature loads the ONNX Runtime shared library at run time from the path in
`ORT_DYLIB_PATH`. The `ort` crate rustnn is built against requires ONNX Runtime 1.29; the
library in Windows `System32` is older (1.17) and the process aborts with a `BadVersion` panic
from `ort` when it is picked up, so set the variable before every run.

Download the pinned release. In a clone with `make` installed:

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

The program below computes `y = relu(x + 1)` for a 2x2 tensor.

```rust
use rustnn::mlcontext::{
    MLContext, MLContextOptions, MLGraphBuilder, MLNamedOperands, MLNamedTensors,
    MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;

fn main() -> rustnn::error::Result<()> {
    // 1. Context: backend selection happens here. `accelerated = false` asks for a CPU device.
    let options = MLContextOptions::new(MLPowerPreference::Default, false);
    let mut context = MLContext::create(&options)?;
    println!("backend: {:?}", context.rustnn_backend());

    // 2. Builder: records operations for this context's backend.
    let mut builder = MLGraphBuilder::new(&mut context)?;
    let descriptor = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
    let x = builder.input("x", &descriptor)?;
    let one = builder.constant_from_slice(&descriptor, &[1.0f32; 4])?;
    let sum = builder.add(x, one)?;
    let y = builder.relu(sum)?;

    // 3. Build: names the outputs and compiles the graph. Output names are the dispatch keys.
    let mut outputs = MLNamedOperands::new();
    outputs.insert("y", y);
    let mut graph = builder.build(&outputs)?;

    // 4. Tensors: allocated by the context; the flags decide what the host may do with them.
    let tensor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
    let x_tensor = context.create_tensor(&tensor.to_writable())?;
    let y_tensor = context.create_tensor(&tensor.to_readable())?;
    context.write_tensor(&x_tensor, &[-2.0f32, -1.0, 0.0, 1.0])?;

    // 5. Dispatch: bind tensors by name and run.
    let mut inputs = MLNamedTensors::new();
    inputs.insert("x", &x_tensor);
    let mut output_tensors = MLNamedTensors::new();
    output_tensors.insert("y", &y_tensor);
    context.dispatch(&mut graph, &inputs, &output_tensors)?;

    // 6. Read back: the buffer must hold exactly the tensor's bytes.
    let mut result = [0.0f32; 4];
    context.read_tensor(&y_tensor, &mut result)?;
    assert_eq!(result, [0.0, 0.0, 1.0, 2.0]);
    Ok(())
}
```

Run it with a backend feature enabled and `ORT_DYLIB_PATH` set in the same shell:

```bash
export ORT_DYLIB_PATH=...          # PowerShell: $env:ORT_DYLIB_PATH = "..."
cargo run --features onnx-runtime
```

Points worth knowing:

- Every builder call infers the output shape and data type immediately, so shape errors
  surface at the call, not at build time.
- `build` consumes the recorded graph. A builder compiles exactly one graph; create a new
  builder for the next one.
- The names passed to `input` and `build` are the keys that `dispatch` validates the tensor
  bindings against. A missing input, a shape mismatch or a wrong data type is reported as
  `Error::GraphDispatchError` before the backend runs.
- Graphs and tensors borrow the context; the borrow checker keeps them from outliving it.

## Load a graph from a file

rustnn reads the `.webnn` text format and the JSON format of the
[webnn-graph](https://github.com/rustnn/webnn-graph) crate, including the exports of
[onnx2webnn](https://github.com/rustnn/onnx2webnn). Weights referenced with `@weights(...)`
are resolved from files next to the graph: `manifest.json` plus `model.weights`, or the
`.safetensors` file written by `rustnn_save_webnn`.

```rust
use rustnn::load_graph_from_path;
use rustnn::mlcontext::{MLContext, MLContextOptions, MLGraphBuilder, MLPowerPreference};

let graph_info = load_graph_from_path("model.webnn")?;
let mut context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, false))?;
let mut builder = MLGraphBuilder::new(&mut context)?;
let graph = builder.build_graph_info(graph_info)?;
// graph.input_descriptors and graph.output_descriptors list the names and shapes to bind.
```

`build_graph_info` is a rustnn extension: it compiles an already complete graph, bypassing the
recording methods. The `examples/` directory contains `sample_graph.webnn`, `sample_graph.json`
and `toy_transformer.webnn` to try this with.

## Command line tool

The `rustnn` binary validates a graph file, prints its inputs, outputs and dependency fan-out,
and optionally exports or executes it. Validation and conversion need no runtime feature and
no `ORT_DYLIB_PATH`; each `--run-*` flag exists only when its feature is compiled in
(`--run-onnx` with `onnx-runtime`, `--run-trtx` with `trtx-runtime`, `--run-coreml` with
`coreml-runtime` on macOS) and `--help` lists the flags of the current build.

```bash
# Validate and describe
cargo run --features onnx-runtime -- examples/sample_graph.webnn

# Graphviz export
cargo run --features onnx-runtime -- examples/sample_graph.webnn --export-dot target/graph.dot
dot -Tpng target/graph.dot -o target/graph.png

# Convert; formats are onnx, coreml and, with their features, trtx, litert and cann
cargo run --features onnx-runtime -- examples/sample_graph.webnn --convert onnx --convert-output target/graph.onnx

# Convert and execute once with zeroed inputs (ORT_DYLIB_PATH must be set for --run-onnx)
cargo run --features onnx-runtime -- examples/sample_graph.webnn --convert onnx --run-onnx
cargo run --features onnx-runtime,trtx-runtime -- examples/sample_graph.webnn --convert onnx --run-trtx
cargo run --features coreml-runtime -- examples/sample_graph.webnn --convert coreml --run-coreml   # macOS
```

`--tensor-limit <bytes>` raises the validator's tensor byte limit for very large models. The
`make run`, `make viz`, `make onnx` and `make coreml` targets wrap these commands.

## Next steps

- [API Overview](api-reference.md): types, builder conventions and error types.
- [Backends](backends.md): selection rules and per-backend requirements.
- [Examples](examples.md): the example programs and short recipes.
- [Advanced Topics](advanced.md): backend options, dynamic shapes, saving graphs, caching and debugging.

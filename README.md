<div align="center">
  <img src="logo/rustnn.png" alt="rustnn logo" width="200"/>

  # rustnn

  A Rust implementation of the W3C WebNN API with pluggable execution backends.
</div>

---

## [WARNING] EXPERIMENTAL - DO NOT USE IN PRODUCTION

rustnn is a development release (`0.5.x`). APIs change without notice.

---

## What is rustnn?

- **The WebNN API in Rust.** `MLContext`, `MLGraphBuilder`, `MLGraph`, `MLTensor` and
  `dispatch` mirror the [W3C WebNN](https://www.w3.org/TR/webnn/) JavaScript API. Every
  operation of the specification is available on the builder; rustnn-specific additions carry
  a `rustnn_` prefix.
- **Backends selected at context creation.** ONNX Runtime, NVIDIA TensorRT-RTX, Apple CoreML,
  LiteRT and Huawei CANN, chosen from the WebNN `accelerated` and power-preference hints or
  forced with a backend hint.
- **Graph interchange.** Loads `.webnn` text and JSON graphs from
  [webnn-graph](https://github.com/rustnn/webnn-graph) and
  [onnx2webnn](https://github.com/rustnn/onnx2webnn), saves graphs with `.safetensors` weights,
  exports ONNX and CoreML models and, with their features, TensorRT engines, TFLite and CANN
  models.
- **Conformance.** The upstream WebNN Web Platform Tests run in-repo against the backends on
  every pull request; the nightly [dashboard](https://rustnn.github.io/rustnn/wpt-conformance/)
  shows per-operation results.

Python users: the [pywebnn](https://github.com/rustnn/pywebnn) package wraps rustnn. This
repository contains no Python API; the scripts under `examples/experimental/` only run exported
ONNX models for parity checks.

## Quick start

The WebNN API below lives on `main` and is not yet published; the `rustnn` crate on crates.io
(0.5.x) is the earlier converter and loader crate without `MLContext`. Use the git dependency:

```toml
[dependencies]
rustnn = { git = "https://github.com/rustnn/rustnn", features = ["onnx-runtime"] }
```

```rust
use rustnn::mlcontext::{
    MLContext, MLContextOptions, MLGraphBuilder, MLNamedOperands, MLNamedTensors,
    MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;

fn main() -> rustnn::error::Result<()> {
    let options = MLContextOptions::new(MLPowerPreference::Default, false);
    let mut context = MLContext::create(&options)?;

    // y = relu(x + 1)
    let mut builder = MLGraphBuilder::new(&mut context)?;
    let descriptor = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
    let x = builder.input("x", &descriptor)?;
    let one = builder.constant_from_slice(&descriptor, &[1.0f32; 4])?;
    let sum = builder.add(x, one)?;
    let y = builder.relu(sum)?;
    let mut outputs = MLNamedOperands::new();
    outputs.insert("y", y);
    let mut graph = builder.build(&outputs)?;

    let tensor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
    let x_tensor = context.create_tensor(&tensor.to_writable())?;
    let y_tensor = context.create_tensor(&tensor.to_readable())?;
    context.write_tensor(&x_tensor, &[-2.0f32, -1.0, 0.0, 1.0])?;

    let mut inputs = MLNamedTensors::new();
    inputs.insert("x", &x_tensor);
    let mut output_tensors = MLNamedTensors::new();
    output_tensors.insert("y", &y_tensor);
    context.dispatch(&mut graph, &inputs, &output_tensors)?;

    let mut result = [0.0f32; 4];
    context.read_tensor(&y_tensor, &mut result)?;
    assert_eq!(result, [0.0, 0.0, 1.0, 2.0]);
    Ok(())
}
```

The ONNX Runtime backend loads the ONNX Runtime 1.29 shared library from `ORT_DYLIB_PATH`;
from the repository root, `make onnxruntime-download` fetches a matching release into
`target/onnxruntime/`.
Set the variable before running (`export ORT_DYLIB_PATH=...` in bash,
`$env:ORT_DYLIB_PATH = "..."` in PowerShell); without it the `ort` crate picks up an older
system library and aborts. See [Getting Started](docs/user-guide/getting-started.md).

## Features

| Feature | Backend |
|---|---|
| `onnx-runtime` | ONNX Runtime (CPU, GPU, NPU execution providers), all platforms |
| `trtx-runtime` | NVIDIA TensorRT-RTX (Linux, Windows); `trtx-runtime-mock` builds without a GPU |
| `coreml-runtime` | Apple CoreML (macOS) |
| `litert-runtime` | LiteRT / TensorFlow Lite; needs `flatc` at build time |
| `cann-runtime` | Huawei CANN on OpenHarmony; `cann-runtime-mock` for validation |
| `dynamic-inputs` | Dynamic dimensions bounded by a maximum size |
| `webnn-runtime` | Browser WebNN bindings for `wasm32-unknown-unknown` (in progress) |
| `trtx-enterprise` | The TensorRT backend linked against full TensorRT 10 (`nvinfer`) instead of TensorRT-RTX; RTX-only features such as CUDA graphs are compiled out. Used for validation, not a supported deployment target |
| `native-examples` | Compiles the large example programs |

Full list and environment variables: crate docs (`make docs-api`) or
[Backends](docs/user-guide/backends.md).

## Command line

```bash
cargo run --features onnx-runtime -- examples/sample_graph.webnn                         # validate
cargo run --features onnx-runtime -- examples/sample_graph.webnn --export-dot graph.dot  # Graphviz
cargo run --features onnx-runtime -- examples/sample_graph.webnn --convert onnx --convert-output model.onnx
cargo run --features onnx-runtime -- examples/sample_graph.webnn --convert onnx --run-onnx
```

## Documentation

- [Documentation site](https://rustnn.github.io/rustnn/) with the user guide, architecture and development pages
- Rust API reference: `make docs-api` writes it to `target/doc/rustnn/index.html`; the site publishes it under [`/api/`](https://rustnn.github.io/rustnn/api/rustnn/) (docs.rs still shows the 0.5.x crate)
- [Backend Operator Support](docs/development/backend-operator-support.md), generated from the converters
- [WPT conformance dashboard](https://rustnn.github.io/rustnn/wpt-conformance/)
- [Changelog](CHANGELOG.md)

## Contributing

<<<<<<< HEAD
Contributions welcome! Please see:

- [AGENTS.md](AGENTS.md) - Project architecture and conventions
- [docs/development/contributing.md](docs/development/contributing.md) - How to add features
- [TODO.txt](TODO.txt) - Feature requests and known issues

**Quick Contribution Guide:**

1. Fork and create feature branch: `git checkout -b feature/my-feature`
2. Install hooks (optional): `./scripts/install-git-hooks.sh`
3. Make changes and test: `make test && make python-test`
4. If a WPT expected-failure list needs to be updated (indicated by test failures),
   use the per-backend sync target, which regenerates `{backend}_expected_failures.txt`
   from a fresh run:

   - `make wpt-sync-onnx`
   - `make wpt-sync-litert`
   - `make wpt-sync-coreml` (macOS only)
   - `make wpt-sync-trtx` (requires an NVIDIA GPU)

   Review the diff before committing!
5. Format code: `make fmt`
6. Commit and push

## License

Apache License, Version 2.0. See [LICENSE](LICENSE).

## Links

- GitHub: https://github.com/rustnn/rustnn
- crates.io: https://crates.io/crates/rustnn
- Python bindings: https://github.com/rustnn/pywebnn
- W3C WebNN specification: https://www.w3.org/TR/webnn/

## Acknowledgments

- The W3C WebML Working Group for the specification
- The Chromium WebNN implementation, used as the reference for operator lowering
- Created by [Tarek Ziade](https://github.com/tarekziade)

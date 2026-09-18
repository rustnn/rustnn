<div align="center">
  <img src="https://raw.githubusercontent.com/rustnn/rustnn/main/logo/rustnn.png" alt="rustnn logo" width="200"/>
</div>

# rustnn

rustnn is a Rust implementation of the [W3C WebNN API](https://www.w3.org/TR/webnn/). It records
neural network graphs with a WebNN-style builder, validates and shape-infers them, and executes
them on a pluggable backend.

**Experimental.** rustnn is a development release. APIs change without notice and the crate is
not meant for production use. The API on this site is on the `main` branch and is used as a git
dependency; the `rustnn` crate on crates.io (0.5.x) is the earlier converter and loader crate
without `MLContext`.

## What rustnn provides

- **The WebNN API in Rust.** `MLContext`, `MLGraphBuilder`, `MLGraph`, `MLTensor` and
  `dispatch` mirror the JavaScript API. rustnn-specific additions carry a `rustnn_` prefix.
- **Backends selected at context creation.** ONNX Runtime, NVIDIA TensorRT-RTX, Apple CoreML,
  LiteRT and Huawei CANN, chosen from the WebNN `accelerated` and power-preference hints or
  forced with a backend hint. A browser WebNN backend for `wasm32` is in progress.
- **Graph interchange.** Loads `.webnn` text and JSON graphs written by the
  [webnn-graph](https://github.com/rustnn/webnn-graph) crate and by
  [onnx2webnn](https://github.com/rustnn/onnx2webnn), saves graphs with `.safetensors`
  weights, and exports ONNX and CoreML models and, with their features, TensorRT engines,
  TFLite and CANN models.
- **Conformance.** The upstream WebNN Web Platform Tests run in-repo against every backend on
  each pull request; the nightly [WPT dashboard](https://rustnn.github.io/rustnn/wpt-conformance/)
  shows per-operation results.

## Quick example

```rust
use rustnn::mlcontext::{
    MLContext, MLContextOptions, MLGraphBuilder, MLNamedOperands, MLNamedTensors,
    MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;

fn main() -> rustnn::error::Result<()> {
    // Pick a backend from the WebNN hints: not accelerated selects a CPU device.
    let options = MLContextOptions::new(MLPowerPreference::Default, false);
    let mut context = MLContext::create(&options)?;

    // Record y = relu(x + 1) and compile it for the selected backend.
    let mut builder = MLGraphBuilder::new(&mut context)?;
    let descriptor = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
    let x = builder.input("x", &descriptor)?;
    let one = builder.constant_from_slice(&descriptor, &[1.0f32; 4])?;
    let sum = builder.add(x, one)?;
    let y = builder.relu(sum)?;
    let mut graph_outputs = MLNamedOperands::new();
    graph_outputs.insert("y", y);
    let mut graph = builder.build(&graph_outputs)?;

    // Tensors live on the backend device; flags control host access.
    let tensor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
    let x_tensor = context.create_tensor(&tensor.to_writable())?;
    let y_tensor = context.create_tensor(&tensor.to_readable())?;
    context.write_tensor(&x_tensor, &[-2.0f32, -1.0, 0.0, 1.0])?;

    let mut inputs = MLNamedTensors::new();
    inputs.insert("x", &x_tensor);
    let mut outputs = MLNamedTensors::new();
    outputs.insert("y", &y_tensor);
    context.dispatch(&mut graph, &inputs, &outputs)?;

    let mut result = [0.0f32; 4];
    context.read_tensor(&y_tensor, &mut result)?;
    assert_eq!(result, [0.0, 0.0, 1.0, 2.0]);
    Ok(())
}
```

Depend on rustnn with a backend feature on the dependency line
(`rustnn = { git = "https://github.com/rustnn/rustnn", features = ["onnx-runtime"] }`), point
`ORT_DYLIB_PATH` at the ONNX Runtime shared library and `cargo run` (see
[Getting Started](user-guide/getting-started.md)). This API is on the `main` branch; the
`rustnn` crate published on crates.io (0.5.x) predates it, so depend on the git repository until
the next release.

## Documentation map

| Section | Content |
|---|---|
| [Getting Started](user-guide/getting-started.md) | Requirements, adding the crate, first graph, loading `.webnn` files, the CLI |
| [API Overview](user-guide/api-reference.md) | The WebNN API types, builder conventions, options, data types, errors |
| [Backends](user-guide/backends.md) | Selection rules and per-backend requirements, execution model and test coverage |
| [Examples](user-guide/examples.md) | The example programs in `examples/` and short recipes |
| [Advanced Topics](user-guide/advanced.md) | Backend hints and options, dynamic shapes, saving and exporting graphs, caching, debugging |
| [Troubleshooting](user-guide/troubleshooting.md) | Error messages by phase, their causes and fixes |
| [Rust API Reference](https://rustnn.github.io/rustnn/api/rustnn/) | Generated rustdoc for every public item, deployed from `main`; locally `make docs-api` writes it to `target/doc/rustnn/` |
| [Architecture](architecture/overview.md) | Layers, data flow, module map and design decisions |
| [Development](development/setup.md) | Toolchain, build and test commands, adding operations and backends, CI |
| [Converter Internals](development/converters.md) | The converter contract, backend-specific lowering rules, debugging emitted models |
| [Documentation Policy](development/documentation-policy.md) | What to update when code changes; rules for contributors and coding agents |
| [Implementation Status](development/implementation-status.md) | API surface, backend status and known gaps |
| [Backend Operator Support](development/backend-operator-support.md) | Generated operation-by-backend matrix |
| [WPT Conformance Guide](testing/wpt-test-guide.md) | Running and triaging the Web Platform Tests |
| [TensorRT-RTX](integration/tensorrt.md) | The native TensorRT backend, caching and precision |
| [CoreML](integration/coreml.md), [LiteRT](integration/litert.md), [CANN](integration/cann.md), [Browser WebNN](integration/webnn-browser.md) | Requirements, device mapping, testing and limits per backend |
| [WebNN Specification and rustnn](reference/webnn-spec.md) | Where each specification concept lives in the Rust API |
| [Graph Files and Weights](reference/graph-files.md) | The `.webnn` text and JSON formats, external weights, backend export formats |

## Python

Python bindings live in the separate [pywebnn](https://github.com/rustnn/pywebnn) package, which
uses rustnn as its core library. This repository contains no Python API.

## Support

- Issues and discussions: [github.com/rustnn/rustnn](https://github.com/rustnn/rustnn/issues)
- Specification: [W3C WebNN](https://www.w3.org/TR/webnn/)
- License: Apache-2.0 ([LICENSE](https://github.com/rustnn/rustnn/blob/main/LICENSE))

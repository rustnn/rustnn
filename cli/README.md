# rustnn CLI

The `rustnn-cli` workspace member builds the `rustnn` executable. Its features forward
to the library. Only some runtimes are enabled by default on each platform.
The CLI is native-only; wasm32 builds use an empty entry point and omit CLI dependencies.

```bash
cargo run -p rustnn-cli -- --list
cargo run -p rustnn-cli -- examples/sample_graph.webnn
cargo run -p rustnn-cli -- examples/sample_graph.webnn --export-dot graph.dot
cargo run -p rustnn-cli -- examples/sample_graph.webnn --convert onnx --convert-output model.onnx
cargo run -p rustnn-cli -- examples/sample_graph.webnn --run burn-cpu
cargo run -p rustnn-cli --features rustnn/trtx-runtime -- examples/sample_graph.webnn --run trtx --accelerated
cargo run -p rustnn-cli --features coreml-export -- examples/sample_graph.webnn --convert coreml --coreml-compiled-output model.mlmodelc
```

`--coreml-compiled-output` writes a runnable `.mlmodelc` with the pure-Rust exporter on any
host (no Apple compiler); `--run coreml --coreml-compile-route rust-export` loads graphs
through the same route. See the CoreML guide's "Compile routes" section.

`--list` lists enabled converters and probes compiled runtimes through `MLContext::create`.
`--help` derives the choices for `--convert` and `--run` from the library. A compiled
runtime may still need native libraries or hardware. CANN mock contexts can be created,
but cannot execute graphs.

Disable defaults for converter-only builds:

```bash
cargo run -p rustnn-cli --no-default-features --features rustnn/litert-converter -- --list
cargo run -p rustnn-cli --no-default-features --features rustnn/litert-converter -- examples/sample_graph.webnn --convert litert --convert-output model.tflite
```

# rustnn

This standalone crate mirrors Chromium's WebNN graph handling while adding
pluggable format converters (ONNX/CoreML) and helper tooling to visualize,
execute, and validate exported graphs on macOS.

The Rust validator matches Chromium's C++ flow:

- JSON files model the `GraphInfo` mojo structure (operands, operations,
  constants, tensor handles).
- `GraphValidator` replicates operand bookkeeping, data type checks, constant
  byte-length verification, and operation dependency tracking.
- `ContextProperties` exposes knobs that mirror `WebNNContextImpl::properties()`
  so tensor limits or supported IO data types can be adjusted.
- A converter registry emits ONNX/CoreML variants of the graph for downstream
  consumption.

## Layout

```
rustnn/
├── Cargo.toml
├── Cargo.lock
├── README.md
├── Makefile                # helper targets (viz/onnx/coreml/validate)
├── build.rs                # prost build script for ONNX/CoreML protos
├── protos/                 # vendored ONNX/CoreML protobuf definitions
├── examples/
│   └── sample_graph.json   # tiny graph with a constant weight
├── scripts/
│   (none; ONNX validation handled by Rust executor)
└── src/
    ├── converters/         # ONNX/CoreML converters + registry
    ├── executors/          # CoreML runtime bridge (macOS)
    ├── error.rs            # GraphError mirrors Chromium paths
    ├── graph.rs            # DataType/Operand/Operation/GraphInfo structs
    ├── graphviz.rs         # DOT exporter
    ├── loader.rs           # JSON loader
    ├── main.rs             # CLI entrypoint
    └── validator.rs        # GraphValidator + ContextProperties
```

## Running the validator

```
cd rustnn
cargo run -- examples/sample_graph.json
```

The CLI prints the number of operands/operations, the input/output tensor
descriptors, and the dependency fan-out recorded while validating operations.
Use `--tensor-limit <bytes>` to experiment with the limit enforced in
`WebNNGraphBuilderImpl::ValidateGraphImpl`.

To export a Graphviz DOT view of the graph, pass `--export-dot <path>`:

```
cargo run -- examples/sample_graph.json --export-dot /tmp/graph.dot
dot -Tpng /tmp/graph.dot > /tmp/graph.png
```

Or with the bundled helper target (requires `dot` and macOS `open`):

```
make viz
```

## Converting graphs

A pluggable converter registry can emit other graph formats. ONNX is the
first built-in converter:

```
cargo run -- examples/sample_graph.json --convert onnx --convert-output /tmp/graph.onnx
```

CoreML export produces a `.mlmodel` blob:

```
cargo run -- examples/sample_graph.json --convert coreml --convert-output /tmp/graph.mlmodel
```

To execute the converted CoreML model directly from Rust (macOS only), enable
the `coreml-runtime` feature and pass `--run-coreml`:

```
cargo run --features coreml-runtime -- examples/sample_graph.json --convert coreml --run-coreml
```

To cache the compiled `.mlmodelc` bundle for reuse, also pass
`--coreml-compiled-output <path>`:

```
cargo run --features coreml-runtime -- examples/sample_graph.json --convert coreml --run-coreml --coreml-compiled-output target/graph.mlmodelc
```
If the compiled bundle already exists at that path, it will be loaded without
recompiling; otherwise it is compiled once and persisted for future runs.

Omit `--convert-output` to print the converted bytes to stdout. More converters
can be registered via `ConverterRegistry`.

For ONNX, you can build and validate (optionally running inference via onnxruntime) with:

```
make onnx-validate
```

The Makefile will download a macOS/arm64 ONNX Runtime into
`target/onnxruntime/onnxruntime-osx-arm64-<version>` and run with
`ORT_STRATEGY=system`, `ORT_LIB_LOCATION` pointing to that directory, and the
library path exported. To use your own runtime, set those environment variables
before invoking the Makefile targets.

To run the whole pipeline (build, tests, converters, ONNX + CoreML validation):

```
make validate-all-env
```

## License

Licensed under the Apache License, Version 2.0.

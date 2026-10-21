# API and support map

Use the references below for API contracts, implementation gaps, and backend support.
Rustdoc documents unimplemented public methods on the methods themselves; an API appearing
in Rustdoc does not by itself mean it is implemented. Converter coverage describes available
lowerings, while WPT results describe tested numerical behavior.

This navigation table replaces the former Implementation Status, API Overview, Backends,
Advanced Topics, and WebNN Specification guides. Graph-file guidance is consolidated below.

| Topic | Authoritative reference |
|---|---|
| API entry point and first graph | [Crate Rustdoc](https://rustnn.github.io/rustnn/api/rustnn/) |
| Context operations and unimplemented methods | [MLContext](https://rustnn.github.io/rustnn/api/rustnn/mlcontext/struct.MLContext.html), including [op_support_limits](https://rustnn.github.io/rustnn/api/rustnn/mlcontext/struct.MLContext.html#method.op_support_limits) |
| Builder operations, naming conventions, and constant-construction gaps | [Builder module](https://rustnn.github.io/rustnn/api/rustnn/mlgraphbuilder/), [MLGraphBuilder](https://rustnn.github.io/rustnn/api/rustnn/mlgraphbuilder/struct.MLGraphBuilder.html) |
| Tensor access, lifetime, and unimplemented methods | [MLTensor](https://rustnn.github.io/rustnn/api/rustnn/mlcontext/struct.MLTensor.html) |
| WebNN dictionaries, defaults, and enums | [Options](https://rustnn.github.io/rustnn/api/rustnn/operator_options/), [enums](https://rustnn.github.io/rustnn/api/rustnn/operator_enums/) |
| Backend selection and feature flags | [Selection rules](https://rustnn.github.io/rustnn/api/rustnn/backend_selection/), [crate features](https://rustnn.github.io/rustnn/api/rustnn/#cargo-features) |
| Backend setup, caching, precision, and limitations | [TensorRT-RTX](../integration/tensorrt.md), [CoreML](../integration/coreml.md), [LiteRT](../integration/litert.md), [Burn CPU](../integration/burn-cpu.md), [CANN](../integration/cann.md), [browser WebNN](../integration/webnn-browser.md); [ONNX Runtime setup](../user-guide/getting-started.md) |
| Operations lowered by each backend | [Generated operator support report](backend-operator-support.md) |
| Tested numerical conformance and remaining failures | [WPT dashboard](https://rustnn.github.io/rustnn/wpt-conformance/), [test and baseline workflow](../testing/wpt-test-guide.md) |
| Dynamic shapes and tensor capacity | [Dynamic-shape workflow](flexible-input-shapes.md), [runtime validation](https://rustnn.github.io/rustnn/api/rustnn/runtime_checks/) |
| Graph files, external weights, loading, and saving | [Shared graph-file guide](../reference/graph-files.md) |
| Backend conversion and graph architecture | [Converter guide](converters.md), [architecture](../architecture/overview.md) |
| Logging, tracing, and model dumps | [Diagnostics](https://rustnn.github.io/rustnn/api/rustnn/#diagnostics), [environment variables](https://rustnn.github.io/rustnn/api/rustnn/#environment-variables) |
| Errors and troubleshooting | [Error types](https://rustnn.github.io/rustnn/api/rustnn/error/), [troubleshooting](../user-guide/troubleshooting.md) |

API comments link to the relevant WebNN specification sections. Consult those comments for
Rust-specific differences rather than maintaining a separate specification-to-API inventory.

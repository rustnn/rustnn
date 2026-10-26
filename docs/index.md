<div align="center">
  <img src="https://raw.githubusercontent.com/rustnn/rustnn/main/logo/rustnn.png" alt="rustnn logo" width="200"/>
</div>

# rustnn

rustnn is a Rust implementation of the [W3C WebNN API](https://www.w3.org/TR/webnn/). It records
neural network graphs with a WebNN-style builder, validates and shape-infers them, and executes
them on a pluggable backend.

**Experimental.** rustnn is a development release. APIs change without notice and the crate is
not meant for production use. This site documents the `main` branch; the `rustnn` crate on
crates.io (0.6.0 and later) carries the same `MLContext` API, while the 0.5.x releases were the
earlier converter and loader crate.

Start with [Getting Started](user-guide/getting-started.md) for installation, then the
[crate's runnable example](https://rustnn.github.io/rustnn/api/rustnn/#example).
The [Rust API reference](https://rustnn.github.io/rustnn/api/rustnn/) is generated from source
comments. It owns signatures, option defaults, backend selection and environment variables.
These guide pages cover workflows and link to those contracts.

For C and C++, see the [usage guide](reference/c-api.md) and
[generated API reference](c-api/index.html).

Backend support is tracked in the generated [operator report](development/backend-operator-support.md)
and [WPT dashboard](https://rustnn.github.io/rustnn/wpt-conformance/). See the
[WPT guide](testing/wpt-test-guide.md) for which backends and environments the harness exercises.


## Documentation map

| Section | Content |
|---|---|
| [Getting Started](user-guide/getting-started.md) | Requirements, adding the crate, first graph, loading `.webnn` files, conversion |
| [Examples](user-guide/examples.md) | The example programs in `examples/` and short recipes |
| [Troubleshooting](user-guide/troubleshooting.md) | Error messages by phase, their causes and fixes |
| [API and Support Map](development/implementation-status.md) | Pointers to API contracts, unimplemented methods, backend support, and consolidated guides |
| [Rust API Reference](https://rustnn.github.io/rustnn/api/rustnn/) | Generated rustdoc for every public item, deployed from `main`; locally `make docs-api` writes it to `target/doc/rustnn/` |
| [Architecture](architecture/overview.md) | Layers, data flow, module map and design decisions |
| [Development](development/setup.md) | Toolchain, build and test commands, adding operations and backends, CI |
| [Converter Internals](development/converters.md) | The converter contract, backend-specific lowering rules, debugging emitted models |
| [Documentation Policy](development/documentation-policy.md) | What to update when code changes; rules for contributors and coding agents |
| [Backend Operator Support](development/backend-operator-support.md) | Generated operation-by-backend matrix |
| [WPT Conformance Guide](testing/wpt-test-guide.md) | Running and triaging the Web Platform Tests |
| [TensorRT-RTX](integration/tensorrt.md) | The native TensorRT backend, caching and precision |
| [CoreML](integration/coreml.md), [LiteRT](integration/litert.md), [CANN](integration/cann.md), [Browser WebNN](integration/webnn-browser.md) | Requirements, device mapping, testing and limits per backend |
| [Graph Files and Weights](reference/graph-files.md) | Shared loader guide: source graphs, external weights, and links to format contracts |

## Python

Python bindings live in the separate [pywebnn](https://github.com/rustnn/pywebnn) package, which
uses rustnn as its core library. This repository contains no Python API.

## Support

- Issues and discussions: [github.com/rustnn/rustnn](https://github.com/rustnn/rustnn/issues)
- Specification: [W3C WebNN](https://www.w3.org/TR/webnn/)
- License: Apache-2.0 ([LICENSE](https://github.com/rustnn/rustnn/blob/main/LICENSE))

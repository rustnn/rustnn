# WebNN graph files

RustNN can save and load the textual `.webnn` graph format implemented by
[`webnn-graph`](https://github.com/rustnn/webnn-graph). This is an interchange format used by these projects,
not a serialized format defined by the W3C WebNN specification.

The canonical contracts live in `webnn-graph`:

- [WebNN graph format](https://github.com/rustnn/webnn-graph/blob/main/docs/webnn-format.md)
- [External weight format](https://github.com/rustnn/webnn-graph/blob/main/docs/external-weights.md)

This page describes only RustNN's integration with those formats.

## Saving a graph

`MLGraphBuilder::rustnn_save_webnn` finalizes the named outputs and writes two sibling files:

```text
model.webnn
model.safetensors
```

The `.webnn` file contains graph structure and external constant references. The SafeTensors sidecar contains
constant bytes. Keep the pair together when copying or publishing a model.

```rust
builder.rustnn_save_webnn(&named_outputs, "model.webnn")?;
```

Saving does not consume the builder. The same builder can still be finalized with `build` afterwards.

## Loading a graph

`load_graph_from_path` accepts `.webnn` text or the equivalent `.json` `GraphJson` representation:

```rust
let graph_info = rustnn::load_graph_from_path("model.webnn")?;
```

The loader:

1. Reads the graph file and selects the parser from its extension.
2. Parses `.webnn` through `webnn-graph` or deserializes `GraphJson`.
3. Resolves every external constant through the shared `webnn-graph` resolver.
4. Converts the resolved AST into RustNN's backend-independent `GraphInfo`.

By default, sidecar discovery checks graph-stem and `model` names next to the graph, preferring SafeTensors over
manifest-backed raw weights. See the canonical external-weight documentation for the exact order and validation
rules.

## Compiling and executing

Loading returns `GraphInfo`; it does not compile or execute a backend graph. Compile it through the selected
`MLContext` backend:

```rust
let graph_info = rustnn::load_graph_from_path("model.webnn")?;
let mut graph = context.rustnn_build_graph(graph_info)?;
```

Tensor creation, input dispatch, output reads, and backend lifetime follow the same RustNN APIs used by graphs
created through `MLGraphBuilder`.

External sidecars are used only while loading. RustNN owns the resolved constant bytes in `GraphInfo`, and the
compiled backend graph is reused across dispatches. SafeTensors is not read again for each inference or decode
step.

## Compatibility

RustNN saves ordinary constants using their corresponding SafeTensors storage types. Logical Int4 and Uint4
constants use the versioned packed-U8 extension defined by `webnn-graph`. Regenerate both files together when the
experimental format contract changes.

RustNN also loads manifest-backed `.weights` artifacts accepted by `webnn-graph`, but
`rustnn_save_webnn` produces SafeTensors sidecars.

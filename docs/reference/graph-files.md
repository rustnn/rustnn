# Graph files and weights

RustNN imports `.webnn` text and the equivalent `webnn-graph-json` representation.
These are project interchange formats, not serialization formats defined by W3C WebNN.
The format contracts are maintained by `webnn-graph`:

- [Graph syntax and JSON representation](https://github.com/rustnn/webnn-graph/blob/main/docs/webnn-format.md)
- [External weights and packed 4-bit storage](https://github.com/rustnn/webnn-graph/blob/main/docs/external-weights.md)

This guide is included in the loader's Rustdoc as well as the website. The API contracts live
on [load_graph_from_path](https://rustnn.github.io/rustnn/api/rustnn/loader/fn.load_graph_from_path.html)
and [rustnn_save_webnn](https://rustnn.github.io/rustnn/api/rustnn/mlgraphbuilder/struct.MLGraphBuilder.html#method.rustnn_save_webnn).

## Save, load, and compile

Saving a builder writes `model.webnn` and `model.safetensors`. Keep the pair together when
moving the model. Saving borrows the builder without consuming it; it can still be compiled.

```rust,ignore
builder.rustnn_save_webnn(&outputs, "model.webnn")?;
let graph_info = rustnn::load_graph_from_path("model.webnn")?;
let mut graph = context.rustnn_build_graph(graph_info)?;
```

The loader selects the parser from the extension: `.webnn` text is parsed into `GraphJson`,
while `.json` is deserialized directly into that same representation. Both then use the shared
external-weight resolver and RustNN graph recorder, including shape inference and validation.
Loading returns a backend-independent `GraphInfo`; it does not compile, execute, or numerically
validate a model. Compile it with
[MLContext::rustnn_build_graph](https://rustnn.github.io/rustnn/api/rustnn/mlcontext/struct.MLContext.html#method.rustnn_build_graph),
then use the usual tensor write, dispatch, and read calls. Model-output comparisons belong in
execution tests.

## Weights and compatibility

External constants are resolved relative to the graph file. The shared resolver supports
SafeTensors and manifest-backed raw weights; SafeTensors takes precedence. Its canonical
external-weight documentation above specifies sidecar discovery, name matching, and validation.
Resolved bytes are owned by `GraphInfo`, so sidecars are not reread on every dispatch.

The exporter writes ordinary constants using their corresponding SafeTensors storage types.
For logical Int4 and Uint4 it uses the versioned packed-U8 extension: low-nibble-first bytes
in a one-dimensional U8 tensor, with the graph declaration retaining the logical dtype and
shape. Native SafeTensors 4-bit types are not required. Regenerate both graph and sidecar with
compatible tooling; the loader validates the shared format's metadata and packing rules.

Dynamic dimensions require `dynamic-inputs`; constants remain static. See the
[dynamic-shape workflow](https://rustnn.github.io/rustnn/development/flexible-input-shapes/)
and [Dimension](https://rustnn.github.io/rustnn/api/rustnn/graph/enum.Dimension.html).

`float8e4m3` and `bfloat16` are until an implementation of those data types in webnn-graph unsupported.

## Other exports

For text without file output, use
[rustnn_webnn_text_for_outputs](https://rustnn.github.io/rustnn/api/rustnn/mlgraphbuilder/struct.MLGraphBuilder.html#method.rustnn_webnn_text_for_outputs).
For backend formats, finish a `GraphInfo` and use
[ConverterRegistry](https://rustnn.github.io/rustnn/api/rustnn/converters/struct.ConverterRegistry.html).
Persist both the model and any weights returned in
[ConvertedGraph](https://rustnn.github.io/rustnn/api/rustnn/converters/struct.ConvertedGraph.html);
these backend artifacts are separate from the source `.webnn`/SafeTensors pair. See the
[conversion recipe](https://rustnn.github.io/rustnn/user-guide/examples/#convert-to-onnx-without-a-backend).

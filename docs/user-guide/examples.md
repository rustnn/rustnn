# Examples

## Example programs

The `examples/` directory contains complete programs. Build them with the features they need;
the two largest are gated behind the `native-examples` feature so that `cargo build` stays
fast.

| Program | What it shows | Run |
|---|---|---|
| `fast_style_transfer_builder_api.rs` | Builds the fast style transfer network with the builder API (`conv2d`, `conv_transpose2d`, instance normalization with a fallback composed from reductions, `pad`), downloads the weights from the WebNN test-data repository, and pipelines inferences over pre-allocated tensors | `cargo run --release --features native-examples,onnx-runtime --example fast_style_transfer_builder_api -- --input photo.jpg --output styled.png` |
| `smollm_mlcontext.rs` | Text generation with SmolLM-135M from an onnx2webnn export: loads `.webnn` plus weights, uses dynamic tensor shapes for the KV cache (`rustnn_set_tensor_capacity`, `rustnn_resize_tensor`) and a tokenizer | `cargo run --release --features native-examples,onnx-runtime,dynamic-inputs --example smollm_mlcontext -- --model model.webnn --tokenizer tokenizer.json --max-new-tokens 32` |
| `resnet50_webnn_rust.rs` | ResNet-50 classification and a latency benchmark from a `.webnn` export, with ImageNet preprocessing of a JPEG or PNG input | `cargo run --release --features onnx-runtime --example resnet50_webnn_rust -- --model resnet50_Opset16.webnn --input cat.jpg --labels examples/imagenet_classes.txt --bench` |
| `gpt2_webnn_rust.rs`, `smollm_webnn_rust.rs` | Older generation loops built on the legacy executor path (`ConverterRegistry` plus `run_onnx_with_inputs`) instead of `MLContext` | `cargo run --features onnx-runtime --example smollm_webnn_rust -- --help` |

Replace `onnx-runtime` with `trtx-runtime` to run the same programs on TensorRT-RTX. All of
them accept `--help`.

Other files in `examples/`:

- `sample_graph.webnn`, `sample_graph.json`, `toy_transformer.webnn`, `toy_transformer.json`:
  small graphs for the CLI and for `load_graph_from_path`.
- `mobilenetv2_manifest.json`, `mobilenetv2.weights`, `mobilenetv2_weights/`: a MobileNetV2
  weight set in the `manifest.json` plus `.weights` layout that the loader resolves.
- `imagenet_classes.txt`, `images/test.jpg`, `sample_text.txt`: inputs for the examples.
- `experimental/*.py`: Python scripts that run the exported ONNX models through ONNX Runtime
  for parity checks. They do not use rustnn.

## Recipes

The snippets assume the imports from [Getting Started](getting-started.md) and a `context`
created there. `F32` abbreviates `MLOperandDataType::Float32`.

### Linear layer

```rust
let mut builder = MLGraphBuilder::new(&mut context)?;
let x = builder.input("x", &MLOperandDescriptor::new(F32, vec![1, 4]))?;
let weight = builder.constant_from_slice(&MLOperandDescriptor::new(F32, vec![4, 3]), &weights)?;
let bias = builder.constant_from_slice(&MLOperandDescriptor::new(F32, vec![3]), &biases)?;
let product = builder.matmul(x, weight)?;
let shifted = builder.add(product, bias)?;   // bias broadcasts over the batch dimension
let y = builder.relu(shifted)?;
println!("y: {:?}", builder.rustnn_operand_shape(y)?); // [1, 3]
```

Each call borrows the builder mutably, so keep intermediate operands in variables instead of
nesting calls.

### Options

```rust
use rustnn::operator_options::{MLConv2dOptions, MLReduceOptions};

let conv = MLConv2dOptions {
    strides: vec![2, 2],
    padding: vec![1, 1, 1, 1],           // top, bottom, left, right
    input_layout: "nchw".to_string(),
    filter_layout: "oihw".to_string(),
    bias: Some(bias.rustnn_index()),      // operand fields hold operand indices
    ..Default::default()
};
let y = builder.conv2d_with_options(x, filter, conv)?;

let reduce = MLReduceOptions {
    axes: Some(vec![1]),
    keep_dimensions: true,
    ..Default::default()
};
let mean = builder.reduce_mean_with_options(y, reduce)?;
```

Option structs mirror the specification dictionaries; unset fields keep the spec defaults, and
layouts are the spec strings (`"nchw"`, `"oihw"`). The field lists are in the
[rustdoc of `operator_options`](https://rustnn.github.io/rustnn/api/rustnn/operator_options/).

Operand fields of option structs (`MLConv2dOptions::bias`, `MLGemmOptions::c`, the quantization
zero points) hold operand indices; `MLOperand::rustnn_index()` or `.into()` provides them.

### Several outputs

```rust
let mut outputs = MLNamedOperands::new();
outputs.insert("logits", logits);
outputs.insert("hidden", hidden);
let mut graph = builder.build(&outputs)?;
```

Bind one output tensor per name in `dispatch`. Each tensor may appear only once across the
input and output maps.

### Reuse a graph

Compile once, then loop over `write_tensor`, `dispatch` and `read_tensor`. Tensors are
allocated once as well; the backend keeps them on the device between dispatches. For pipelining
several inferences with separate tensor sets see `examples/fast_style_transfer_builder_api.rs`.

### Save the graph you built

```rust
builder.rustnn_save_webnn(&outputs, "model.webnn")?;   // also writes model.safetensors
let reloaded = rustnn::load_graph_from_path("model.webnn")?;
```

`rustnn_save_webnn` borrows the builder read-only, so it can be called before `build`. The
`.webnn` file references the weights with `@weights(...)`, and the loader resolves them from
the sibling file.

### Convert to ONNX without a backend

```rust
use rustnn::{ConverterRegistry, ONNX_EXTERNAL_WEIGHTS_FILENAME};

let mut builder = MLGraphBuilder::new_uncompiled();
// ... record the graph ...
let graph_info = builder.finish_graph_info(&outputs)?;
let converted = ConverterRegistry::with_defaults().convert("onnx", &graph_info)?;
std::fs::write("model.onnx", &converted.data)?;
if let Some(weights) = converted.weights_data {
    // Large models keep their initializers in a sidecar file next to the model.
    std::fs::write(ONNX_EXTERNAL_WEIGHTS_FILENAME, weights)?;
}
```

`new_uncompiled` needs no runtime feature. The registry also knows `coreml` and, when their
features are enabled, `trtx`, `litert` and `cann`.

### Pick the backend

```rust
let options = MLContextOptions::new(MLPowerPreference::Default, true)
    .with_rustnn_backend_hint(Backend::Onnx);
let context = MLContext::create(&options)?;
match context.rustnn_device() {
    BackendDevice::Onnx { device_type, .. } => println!("ONNX Runtime on {device_type:?}"),
    other => println!("{other:?}"),
}
```

The selection order and the per-backend requirements are described in [Backends](backends.md).

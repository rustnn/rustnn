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

Where the models come from:

- Fast style transfer downloads its weights from the WebNN test-data repository on first run.
- `smollm_mlcontext` clones https://huggingface.co/tarekziade/SmolLM-135M-webnn (the `.webnn`
  export, its weights and `tokenizer.json`) when `--model` is omitted.
- ResNet-50 expects a `.webnn` export you produce yourself: take `resnet50_Opset16.onnx` from
  the ONNX model zoo and convert it with [onnx2webnn](https://github.com/rustnn/onnx2webnn),
  which writes the `.webnn` file next to its `manifest.json` and `model.weights`.

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
created there.

### Linear layer

```rust
use rustnn::operator_enums::MLOperandDataType::Float32;

let weights = [0.5f32; 12];
let biases = [0.1f32; 3];
let mut builder = MLGraphBuilder::new(&mut context)?;
let x = builder.input("x", &MLOperandDescriptor::new(Float32, vec![1, 4]))?;
let weight = builder.constant_from_slice(&MLOperandDescriptor::new(Float32, vec![4, 3]), &weights)?;
let bias = builder.constant_from_slice(&MLOperandDescriptor::new(Float32, vec![3]), &biases)?;
let product = builder.matmul(x, weight)?;
let shifted = builder.add(product, bias)?;   // bias broadcasts over the batch dimension
let y = builder.relu(shifted)?;
println!("y: {:?}", builder.rustnn_operand_shape(y)?); // [1, 3]
```

Each call borrows the builder mutably, so keep intermediate operands in variables instead of
nesting calls.

### Options

`x`, `filter` and `bias` are operands recorded on `builder` as in the previous recipe.

```rust
use rustnn::operator_options::{MLConv2dOptions, MLReduceOptions};
use rustnn::operator_enums::{MLConv2dFilterOperandLayout, MLInputOperandLayout};

let conv = MLConv2dOptions {
    strides: vec![2, 2],
    padding: vec![1, 1, 1, 1],           // top, bottom, left, right
    input_layout: MLInputOperandLayout::Nchw,
    filter_layout: MLConv2dFilterOperandLayout::Oihw,
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
convolution layouts use variants from `rustnn::operator_enums`. The field lists are in the
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
// The file writes return std::io::Error, which rustnn::error::Error does not wrap.
std::fs::write("model.onnx", &converted.data).expect("write model.onnx");
if let Some(weights) = converted.weights_data {
    // Initializers live in a sidecar file that must stay next to the model.
    std::fs::write(ONNX_EXTERNAL_WEIGHTS_FILENAME, weights).expect("write weights");
}
```

`new_uncompiled` needs no runtime feature. The registry also knows `coreml` and, when their
features are enabled, `trtx`, `litert` and `cann`.

### Pick the backend

```rust
use rustnn::mlcontext::{Backend, BackendDevice};

let options = MLContextOptions::new(MLPowerPreference::Default, true)
    .with_rustnn_backend_hint(Backend::Onnx);
let context = MLContext::create(&options)?;
match context.rustnn_device() {
    BackendDevice::Onnx { device_type, .. } => println!("ONNX Runtime on {device_type:?}"),
    other => println!("{other:?}"),
}
```

The hint fixes the backend; the device is still whatever that backend reports for the hints,
so `accelerated = true` on an ONNX Runtime build without a GPU execution provider prints `Cpu`.

The selection order and the per-backend requirements are described in [Backends](backends.md).

### CoreML KV-cache benchmark

`examples/coreml_kv_benchmark.rs` compares the byte-buffer reference path (`baseline`),
persistent native arrays (`persistent`), and optional output backings (`backings`). It uses
real `MLContext` dispatch and distinct ping-pong cache tensors.

Run `make benchmark-coreml-kv` on macOS without downloading a model or applying
other patches. The default workload builds fixed-window FP32 attention using
slice/concat, two matmuls and softmax. Each mode checks all 256 steps against an
independent f64 attention oracle and exact cache contents, then measures a second
pass without oracle computation or cache inspection. Its JSON includes per-mode
timings and host/native copy counters. This is a storage benchmark, not language-model
token throughput, model-quality qualification or an accelerator-placement measurement.
`make test-coreml-kv-benchmark` runs the numerical gates as regressions in macOS CI.

The optional external TinyStories fixture mode below requires currently unsupported
dynamic reshape/expand lowerings; those graphs are not the default runnable benchmark.

Supply the frozen fixture externally: `graphs/fp32_prefill.json`, `graphs/fp32_decode.json`,
`Resources/streams.json`, `Resources/boundary.json`, per-step little-endian float32 logits in
`Resources/references/fp32_cache/<stream>/<step:03>.f32`, and cache checkpoints in
`Resources/cache-references/<stream>/<step:03>/<past_tensor_name>.f32`. Stream manifests carry
`vocab`, `atol`, `rtol` and `streams` (`name` and `steps`, each with a full `tokens` prefix).
The example expects eight layers, 16 attention heads and head size four. Weights and oracle
files are not downloaded automatically or bundled in the repository.

```json
{
  "fixture_root": "/path/to/frozen-fixture",
  "output": "/path/to/results/baseline.json",
  "policy": "cpuOnly",
  "mode": "baseline",
  "streams": ["greedy_0", "boundary_grow"],
  "steps": 128,
  "warmups": 2,
  "repeats": 5,
  "min_measured_seconds": 30
}
```

```bash
make benchmark-coreml-kv COREML_KV_CONFIG=/path/to/config.json
```

Repeat with each mode and reverse the order in a second round. Use identical model and
reference hashes, compiler options, compute policy and lowering prerequisites in all modes.
The target disables release LTO; mobile host applications must also disable cross-language
LTO when their Rust/Xcode LLVM versions are incompatible. Keep other workloads idle.

Each configuration first checks all logits, available cache checkpoints, output independence
and tensor I/O counters. Timed replays must reproduce the verification fingerprint. Decode
timing includes input updates, shape changes, dispatch, logits readback and greedy selection;
it excludes compilation, prefill, oracle comparison and disk I/O. Reports include sustained
goodput, 32-token windows, latency percentiles and actual loaded compute policy. Compute
policy is not proof of placement, and logical copy counters are not total driver traffic.

Flexible outputs cannot use CoreML backings; zero accepted backings means any improvement is
from persistent input storage and reduced host copying. A `smollm_model` configuration probes
import/build/nonempty-cache dispatch readiness only. Its output explicitly records that no
numerical reference was validated, so it must not be reported as a SmolLM quality or throughput
benchmark.

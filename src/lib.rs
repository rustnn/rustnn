//! rustnn: a Rust implementation of the W3C [WebNN API](https://www.w3.org/TR/webnn/).
//!
//! rustnn records neural network graphs with a WebNN-style builder, infers shapes and
//! validates the graph, and executes it on a pluggable backend: ONNX Runtime, NVIDIA
//! TensorRT-RTX, Apple CoreML, LiteRT, Huawei CANN, or the browser's WebNN implementation
//! on `wasm32`.
//!
//! # API layers
//!
//! * **WebNN API** (primary). [`mlcontext::MLContext`] owns a backend and its tensors,
//!   [`mlgraphbuilder::MLGraphBuilder`] records operations and compiles an
//!   [`mlcontext::MLGraph`], and [`mlcontext::MLContext::dispatch`] runs the graph on
//!   [`mlcontext::MLTensor`]s. Names and semantics follow the JavaScript API; rustnn-specific
//!   additions carry a `rustnn_` prefix (for example
//!   [`mlgraphbuilder::MLGraphBuilder::rustnn_save_webnn`]).
//! * **Graph model.** [`graph::GraphInfo`] holds operands and strongly typed
//!   [`operators::Operation`] variants. It is backend-agnostic and round-trips through the
//!   `.webnn` text and JSON formats of the `webnn-graph` crate via [`loader`] and
//!   [`webnn_json`].
//! * **Legacy pipeline.** [`load_graph_from_path`], [`ConverterRegistry`] and the
//!   functions in [`executors`] run a stored graph without an
//!   `MLContext`. Executors reload the converted model on every call and have no device
//!   tensors. The converters are shared with the backends; new code should go through the
//!   WebNN API.
//!
//! # Example
//!
//! ```no_run
//! use rustnn::mlcontext::{
//!     MLContext, MLContextOptions, MLGraphBuilder, MLNamedOperands, MLNamedTensors,
//!     MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor,
//! };
//! use rustnn::operator_enums::MLOperandDataType;
//!
//! # fn main() -> rustnn::error::Result<()> {
//! // Backend selection happens here from the WebNN hints (and optional rustnn hints).
//! let options = MLContextOptions::new(MLPowerPreference::Default, false);
//! let mut context = MLContext::create(&options)?;
//!
//! // Record y = relu(x + 1).
//! let mut builder = MLGraphBuilder::new(&mut context)?;
//! let descriptor = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
//! let x = builder.input("x", &descriptor)?;
//! let one = builder.constant_from_slice(&descriptor, &[1.0f32; 4])?;
//! let sum = builder.add(x, one)?;
//! let y = builder.relu(sum)?;
//! let mut graph_outputs = MLNamedOperands::new();
//! graph_outputs.insert("y", y);
//! let mut graph = builder.build(&graph_outputs)?;
//!
//! // Tensors are created by the context; flags control host access.
//! let tensor_descriptor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
//! let x_tensor = context.create_tensor(&tensor_descriptor.to_writable())?;
//! let y_tensor = context.create_tensor(&tensor_descriptor.to_readable())?;
//! context.write_tensor(&x_tensor, &[-2.0f32, -1.0, 0.0, 1.0])?;
//!
//! let mut inputs = MLNamedTensors::new();
//! inputs.insert("x", &x_tensor);
//! let mut outputs = MLNamedTensors::new();
//! outputs.insert("y", &y_tensor);
//! context.dispatch(&mut graph, &inputs, &outputs)?;
//!
//! let mut result = [0.0f32; 4];
//! context.read_tensor(&y_tensor, &mut result)?;
//! assert_eq!(result, [0.0, 0.0, 1.0, 2.0]);
//! # Ok(())
//! # }
//! ```
//!
//! # Diagnostics
//!
//! Library users install their own [`tracing`] subscriber, optionally filtered by
//! `RUST_LOG`. Spans identify contexts, graphs, files and calls. Targets are module paths, so
//! filters can select a backend or subsystem. Instrumented public calls record errors on their
//! spans before returning them. See the environment variables below for converter dumps.
//!
//! # Cargo features
//!
//! Default features are `onnx-converter` and `burn-cpu-runtime`. Burn CPU is selected
//! after every other eligible backend. Each `<backend>-runtime` enables its matching
//! `<backend>-converter`; converter features expose serialization without rustnn execution.
//! With `default-features = false`, enable only the converters or runtimes you need.
//! Without a runtime, [`mlcontext::MLContext::create`] returns
//! [`error::Error::NoBackendAvailable`]; graph loading, builder validation and enabled
//! converters remain available.
//!
//! | Feature | Effect |
//! |---|---|
//! | `onnx-converter` | ONNX protobuf generation and `protos::onnx` (default) |
//! | `coreml-converter` | CoreML MLProgram protobufs and weights; works on non-Apple hosts |
//! | `litert-converter` | TFLite FlatBuffer generation; needs `flatc`, no LiteRT library |
//! | `burn-cpu-converter` | Portable Burn execution-plan serialization; no Burn tensor dependencies |
//! | `trtx-converter` | Serialized TensorRT-RTX engines through `trtx`; conversion requires the NVIDIA SDK and GPU |
//! | `cann-converter` | HiAI offline models through `hiai-rs`; conversion requires the CANN DDK |
//! | `onnx-runtime` | ONNX Runtime execution through `ort` (dynamic loading, `ORT_DYLIB_PATH`) |
//! | `coreml-export` | Pure-Rust `.mlmodelc` bundles through `mlmodelc-export`: `ConvertedGraph::write_coreml_bundle` and the `RustExport` compile route of [`mlcontextoptions::CoremlCompileRoute`]; required for CoreML on watchOS; works on non-Apple hosts |
//! | `coreml-runtime` | Apple CoreML execution on Apple targets (macOS, iOS; tvOS, watchOS and visionOS build gates are present, see the CoreML guide); failing shims on other targets |
//! | `litert-runtime` | LiteRT execution through `litert` and `litert-sys` |
//! | `burn-cpu-runtime` | Burn NdArray CPU execution (default); no native runtime library |
//! | `trtx-runtime` | TensorRT-RTX execution, CUDA tensors and compressed engine caches |
//! | `cann-runtime` | Huawei CANN/HiAI execution on OpenHarmony |
//! | `cann-runtime-mock` | Device-free CANN validation and placeholder conversion; no native dependencies, dispatch fails |
//! | `webnn-runtime` | Browser WebNN graph replay and execution on `wasm32`; no serialized model converter |
//! | `trtx-enterprise` | `trtx-runtime` linked against full TensorRT 10 (`nvinfer`) for validation |
//! | `dynamic-inputs` | Enable ONNX/CoreML conversion of [`graph::Dimension::Dynamic`] shapes bounded by `max_size` |
//! | `zstd-cache-compression` | Compress on-disk engine caches (enabled by `trtx-runtime`) |
//! | `native-examples` | Build the larger examples (`fast_style_transfer_builder_api`, `smollm_mlcontext`) |
//! | `capi` | C API exports and logging support |
//! | `wpt-embed-corpus` | Embed the WPT corpus for targets without filesystem access |
//! | `webnn-wpt-tests` | Browser WPT tests, including the embedded corpus |
//!
//! # Environment variables
//!
//! | Variable | Effect |
//! |---|---|
//! | `RUST_LOG` | Log and span filter; off unless set (`info` prints the selected backend, `rustnn=debug` adds conversion diagnostics, tensor feed details and per-call spans) |
//! | `RUSTNN_ONNX_DUMP_DIR` | When set, writes the converted ONNX model to `debug_webnn_onnx.onnx` in this directory |
//! | `RUSTNN_TRTX_LOG_VERBOSITY` | TensorRT logger filter: `internal_error`, `error`, `warning`, `info` or `verbose` |
//! | `TRTX_JSON_DUMP_PATH` | Directory for per-engine TensorRT layer dumps |
//! | `ORT_DYLIB_PATH` | Path of the ONNX Runtime shared library loaded by `ort` |
//! | `WPT_COREML_TENSOR_MODE` | WPT harness only: `baseline` (default), `persistent`, or `backings`; does not change library defaults |
//! | `WPT_COREML_FLOAT16` | WPT harness only: `native` (default) or `exact` CoreML float16 lowering; does not change library defaults |
//!
//! Persistent caches (TensorRT engines and the TensorRT runtime cache) live under the
//! platform cache directory, `<cache_dir>/rustnn/<category>`; see [`backends::caching`].

// Every public item needs a doc comment; CI denies warnings.
#![warn(missing_docs)]

pub mod backend_selection;
pub mod backends;
#[cfg(feature = "burn-cpu-converter")]
pub mod burn;
#[cfg(feature = "capi")]
pub mod capi;
pub mod converters;
pub mod error;
pub mod executors;
pub mod graph;
mod graph_recorder;
pub mod graphviz;
mod instrumentation;
pub mod limits;
pub mod loader;
pub mod mlcontext;
pub mod mlcontextoptions;
pub mod mlgraphbuilder;
pub mod operator_enums;
pub mod operator_options;
pub mod operators;
pub mod protos;
pub mod runtime_checks;
pub mod shape_inference;
pub mod tensor;
pub mod webnn_json;
pub(crate) mod webnn_save;

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
pub use executors::coreml;

#[cfg(feature = "coreml-export")]
pub use converters::CoremlBundleStats;
#[cfg(feature = "litert-runtime")]
pub use converters::litert::LiteRtConverter;
pub use converters::{
    ConvertedGraph, ConverterRegistry, GraphConverter, ONNX_EXTERNAL_WEIGHTS_FILENAME,
};
#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
pub use coreml::{CoremlOutput, CoremlRunAttempt, run_coreml_zeroed, run_coreml_zeroed_cached};
pub use error::GraphError;
#[cfg(any(feature = "cann-runtime", feature = "cann-runtime-mock"))]
pub use executors::cann::{CannInput, CannOutput};
#[cfg(feature = "onnx-runtime")]
pub use executors::onnx::{
    OnnxInput, OnnxOutput, OnnxOutputWithData, TensorData, run_onnx_path_with_inputs,
    run_onnx_with_inputs, run_onnx_with_inputs_checked, run_onnx_zeroed,
};
pub use graph::{ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind};
pub use graphviz::graph_to_dot;
pub use loader::load_graph_from_path;
pub use operators::Operation;
extern crate alloc;

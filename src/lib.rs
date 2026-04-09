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
//! * **Legacy pipeline.** [`load_graph_from_path`], [`GraphValidator`], the
//!   [`ConverterRegistry`] and the functions in [`executors`] run a stored graph without an
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
//! # Cargo features
//!
//! Backends are opt-in. Without a runtime feature the crate still validates graphs and
//! converts them to ONNX and CoreML protobufs, but [`mlcontext::MLContext::create`] fails
//! with [`error::Error::NoBackendAvailable`].
//!
//! | Feature | Effect |
//! |---|---|
//! | `onnx-runtime` | ONNX Runtime backend through the `ort` crate (dynamic loading, `ORT_DYLIB_PATH`) |
//! | `trtx-runtime` | NVIDIA TensorRT-RTX backend through the `trtx` crate; `trtx-runtime-mock` builds without a GPU |
//! | `trtx-enterprise` | `trtx-runtime` linked against full TensorRT 10 (`nvinfer`) instead of TensorRT-RTX; RTX-only features are compiled out (validation only) |
//! | `coreml-runtime` | Apple CoreML backend; executes on macOS, compiles to failing shims elsewhere |
//! | `litert-runtime` | LiteRT (TensorFlow Lite) backend through `litert-sys`; needs `flatc` at build time |
//! | `cann-runtime` | Huawei CANN/HiAI backend on OpenHarmony; `cann-runtime-mock` validates without a device |
//! | `webnn-runtime` | Browser WebNN backend for `wasm32-unknown-unknown` (`webnn-wpt-tests` embeds the WPT corpus) |
//! | `dynamic-inputs` | Accept [`graph::Dimension::Dynamic`] shapes bounded by `max_size` |
//! | `zstd-cache-compression` | Compress the on-disk engine caches (enabled by `trtx-runtime`) |
//! | `native-examples` | Build the larger examples (`fast_style_transfer_builder_api`, `smollm_mlcontext`) |
//!
//! # Environment variables
//!
//! | Variable | Effect |
//! |---|---|
//! | `RUST_LOG` | Log filter for the `log` crate (`info` prints the selected backend) |
//! | `RUSTNN_DEBUG` | `1` enables [`debug_print!`] output; `2` also writes the converted ONNX model |
//! | `RUSTNN_DEBUG_ONNX_DIR` | Directory for the ONNX dump written when `RUSTNN_DEBUG=2` |
//! | `RUSTNN_TRTX_LOG_VERBOSITY` | TensorRT logger filter: `internal_error`, `error`, `warning`, `info` or `verbose` |
//! | `TRTX_JSON_DUMP_PATH` | Directory for per-engine TensorRT layer dumps |
//! | `ORT_DYLIB_PATH` | Path of the ONNX Runtime shared library loaded by `ort` |
//!
//! Persistent caches (TensorRT engines and the TensorRT runtime cache) live under the
//! platform cache directory, `<cache_dir>/rustnn/<category>`; see [`backends::caching`].

// Every public item needs a doc comment; CI denies warnings.
#![warn(missing_docs)]

pub mod backend_selection;
pub mod backends;
pub mod converters;
pub mod debug;
pub mod error;
pub mod executors;
pub mod graph;
mod graph_recorder;
pub mod graphviz;
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
pub mod validator;
pub mod webnn_json;
pub(crate) mod webnn_save;

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
pub use executors::coreml;

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
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
pub use executors::trtx::{
    TrtxInput, TrtxOutput, TrtxOutputWithData, run_trtx_with_inputs, run_trtx_zeroed,
};
pub use graph::{ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind};
pub use graphviz::graph_to_dot;
pub use loader::load_graph_from_path;
pub use operators::Operation;
pub use validator::{ContextProperties, GraphValidator, ValidationArtifacts};
extern crate alloc;

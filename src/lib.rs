pub mod converters;
pub mod debug;
pub mod error;
pub mod executors;
pub mod graph;
pub mod graphviz;
pub mod loader;
pub mod protos;
pub mod runtime_checks;
pub mod shape_inference;
pub mod tensor;
pub mod validator;
pub mod webnn_json;

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
pub use executors::coreml;

pub use converters::{ConvertedGraph, ConverterRegistry, GraphConverter};
#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
pub use coreml::{CoremlOutput, CoremlRunAttempt, run_coreml_zeroed, run_coreml_zeroed_cached};
pub use error::GraphError;
#[cfg(feature = "onnx-runtime")]
pub use executors::onnx::{
    OnnxInput, OnnxOutput, OnnxOutputWithData, TensorData, run_onnx_with_inputs, run_onnx_zeroed,
};
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
pub use executors::trtx::{
    TrtxInput, TrtxOutput, TrtxOutputWithData, run_trtx_with_inputs, run_trtx_zeroed,
};
pub use graph::{
    ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, Operation,
};
pub use graphviz::graph_to_dot;
pub use loader::load_graph_from_path;
pub use validator::{ContextProperties, GraphValidator, ValidationArtifacts};

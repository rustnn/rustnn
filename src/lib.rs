#![cfg_attr(feature = "burn-runtime-webgpu", recursion_limit = "256")]

pub mod backend_selection;
pub mod backends;
#[cfg(feature = "burn-plan")]
pub mod burn;
pub mod converters;
pub mod debug;
pub mod error;
pub mod executors;
pub mod graph;
pub mod graphviz;
pub mod loader;
pub mod mlcontext;
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

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
pub use executors::coreml;

#[cfg(feature = "burn-plan")]
pub use burn::{BURN_PLAN_VERSION, BurnGraphPlan};
pub use converters::{
    ConvertedGraph, ConverterRegistry, GraphConverter, ONNX_EXTERNAL_WEIGHTS_FILENAME,
};
#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
pub use coreml::{CoremlOutput, CoremlRunAttempt, run_coreml_zeroed, run_coreml_zeroed_cached};
pub use error::GraphError;
#[cfg(feature = "burn-runtime-cpu")]
pub use executors::burn::{
    BurnInput, BurnOutput, BurnOutputWithData, run_burn_cpu_with_inputs, run_burn_cpu_zeroed,
};
#[cfg(all(feature = "burn-runtime-webgpu", not(feature = "burn-runtime-cpu")))]
pub use executors::burn::{
    BurnInput, BurnOutput, BurnOutputWithData, run_burn_webgpu_with_inputs, run_burn_webgpu_zeroed,
};
#[cfg(all(feature = "burn-runtime-webgpu", feature = "burn-runtime-cpu"))]
pub use executors::burn::{run_burn_webgpu_with_inputs, run_burn_webgpu_zeroed};
#[cfg(feature = "onnx-runtime")]
pub use executors::onnx::{
    OnnxInput, OnnxOutput, OnnxOutputWithData, TensorData, run_onnx_with_inputs,
    run_onnx_with_inputs_checked, run_onnx_zeroed,
};
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
pub use executors::trtx::{
    TrtxInput, TrtxOutput, TrtxOutputWithData, run_trtx_with_inputs, run_trtx_zeroed,
};
pub use graph::{ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind};
pub use graphviz::graph_to_dot;
pub use loader::load_graph_from_path;
#[cfg(any(feature = "burn-runtime-cpu", feature = "burn-runtime-webgpu"))]
pub use mlcontext::BackendPreference;
pub use mlcontext::{MLContext, MLContextOptions, MLPowerPreference};
pub use operators::Operation;
pub use validator::{ContextProperties, GraphValidator, ValidationArtifacts};

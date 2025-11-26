pub mod converters;
pub mod error;
pub mod graph;
pub mod graphviz;
pub mod loader;
pub mod protos;
pub mod validator;
pub mod executors;
#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
pub use executors::coreml;

pub use converters::{ConvertedGraph, ConverterRegistry, GraphConverter};
pub use error::GraphError;
pub use graph::{
    ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, Operation,
};
pub use graphviz::graph_to_dot;
pub use loader::load_graph_from_path;
pub use validator::{ContextProperties, GraphValidator, ValidationArtifacts};
#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
pub use coreml:: {CoremlOutput, CoremlRunAttempt, run_coreml_zeroed, run_coreml_zeroed_cached};

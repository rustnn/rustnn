//! Error types.
//!
//! * [`enum@Error`] is returned by the WebNN API (`MLContext`, `MLGraphBuilder::build`, dispatch).
//! * [`GraphBuilderError`] covers graph recording, including [`ShapeInferenceError`].
//! * [`GraphError`] covers loading, validation, conversion and the legacy executors.
//!
//! All variants are `Send + Sync` so they compose with `anyhow` and similar crates.

use crate::backend_selection::Backend;
use std::path::PathBuf;

use crate::{
    Operand, Operation,
    backends::caching::CacheError,
    graph::DataType,
    mlcontext::{MLOperand, MLOperandDescriptor, MLTensor, MLTensorDescriptor},
};
#[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
use cudarc::driver::DriverError;
use serde_json::Error as JsonError;
use thiserror::Error;

/// Result of the WebNN API (`MLContext`, `MLGraphBuilder::build`, dispatch).
pub type Result<T> = std::result::Result<T, Error>;

/// Shape or data type conflicts found while recording an operation.
///
/// Variant fields carry the values named in the message; they are not documented separately.
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum ShapeInferenceError {
    /// Input operands have inconsistent data types:\noperation:\n`operation`\ninputs:\n`inputs`.
    #[error(
        "Input operands have inconsistent data types:\noperation:\n{operation:#?}\ninputs:\n{inputs:#?}"
    )]
    InconsistentDataTypes {
        operation: Operation,
        inputs: Vec<(MLOperand, Operand)>,
    },

    /// Where condition does not have uint8 data type:\noperation:\n`operation`\ninputs:\n`inputs`.
    #[error(
        "Where condition does not have uint8 data type:\noperation:\n{operation:#?}\ninputs:\n{inputs:#?}"
    )]
    NonUint8Condition {
        operation: Operation,
        inputs: Vec<(MLOperand, Operand)>,
    },

    /// Broadcast shape inference failed: `source`\noperation:\n`operation`\ninputs:\n`inputs`.
    #[error(
        "Broadcast shape inference failed: {source}\noperation:\n{operation:#?}\ninputs:\n{inputs:#?}"
    )]
    BroadcastError {
        operation: Operation,
        inputs: Vec<(MLOperand, Operand)>,
        #[source]
        source: GraphError,
    },

    /// Concat shape inference failed: `source`\noperation:\n`operation`\ninputs:\n`inputs`.
    #[error(
        "Concat shape inference failed: {source}\noperation:\n{operation:#?}\ninputs:\n{inputs:#?}"
    )]
    ConcatError {
        operation: Operation,
        inputs: Vec<(MLOperand, Operand)>,
        #[source]
        source: GraphError,
    },

    /// Expand shape inference failed: `source`\noperation:\n`operation`\ninput:\n`input`.
    #[error(
        "Expand shape inference failed: {source}\noperation:\n{operation:#?}\ninput:\n{input:#?}"
    )]
    ExpandError {
        operation: Operation,
        input: (MLOperand, Operand),
        #[source]
        source: GraphError,
    },

    /// Slice shape inference failed: start and sizes must be have same rank as input tensor:\noperation:\n`operation`\ninput:\n`input`.
    #[error(
        "Slice shape inference failed: start and sizes must be have same rank as input tensor:\noperation:\n{operation:#?}\ninput:\n{input:#?}"
    )]
    SliceErrorWrongStartSizes {
        operation: Operation,
        input: (MLOperand, Operand),
    },

    /// `op_name` shape inference failed: `source`\noperation:\n`operation`.
    #[error("{op_name} shape inference failed: {source}\noperation:\n{operation:#?}")]
    InferError {
        op_name: &'static str,
        operation: Operation,
        #[source]
        source: GraphError,
    },

    /// Operation had options object missing. Should be fixed by RustNN:\noperation: `operation`.
    #[error(
        "Operation had options object missing. Should be fixed by RustNN:\noperation: {operation:#?}"
    )]
    MissingOptions { operation: Operation },

    /// Invalid split sizes: the number of splits must devide the input rank evenly:\noperation:\n`operation`\ninput:\n`input`.
    #[error(
        "Invalid split sizes: the number of splits must devide the input rank evenly:\noperation:\n{operation:#?}\ninput:\n{input:#?}"
    )]
    InvalidSplitSizes {
        operation: Operation,
        input: (MLOperand, Operand),
    },
}

// TODO: use graph_operation_to_webnn_node for error reporting the problematic node?
/// Errors of the graph-recording API.
///
/// Variant fields carry the values named in the message; they are not documented separately.
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum GraphBuilderError {
    /// Failed to build: requested MLGraphBuilder.build with an empty output map.
    #[error("Failed to build: requested MLGraphBuilder.build with an empty output map")]
    EmptyOutputHashMap,

    /// Duplicates in outputs: used MLOperand `operand` for ouput name `first_name` and `second_name`.
    #[error(
        "Duplicates in outputs: used MLOperand {operand:?} for ouput name {first_name:?} and {second_name:?}"
    )]
    DuplicateOutput {
        operand: MLOperand,
        first_name: String,
        second_name: String,
    },

    /// Failed to build: did not provide an input name for tensor with descriptor `0`.
    #[error("Failed to build: did not provide an input name for tensor with descriptor {0:?}")]
    EmptyInputName(MLOperandDescriptor),

    /// Failed to build: did not provide name for MLOperand `0`.
    #[error("Failed to build: did not provide name for MLOperand {0:?}")]
    EmptyOutputName(MLOperand),

    /// Failed to build: requested an MLOperand with id `id` as an output that is already an input:\n`operand`.
    #[error(
        "Failed to build: requested an MLOperand with id {id} as an output that is already an input:\n{operand:#?}"
    )]
    RequestedInputAsOutput { operand: Operand, id: usize },

    /// Failed to build: requested an MLOperand with id `id` as an output that is already an constant:\n`operand`.
    #[error(
        "Failed to build: requested an MLOperand with id {id} as an output that is already an constant:\n{operand:#?}"
    )]
    RequestedConstantAsOutput { operand: Operand, id: usize },

    /// Graph already built: a MLGraphBuilder can only build one graph.
    #[error("Graph already built: a MLGraphBuilder can only build one graph")]
    GraphAlreadyBuilt,

    /// Invalid operand `0`. Was this operand produced by a different GraphBuilder?.
    #[error("Invalid operand {0:?}. Was this operand produced by a different GraphBuilder?")]
    InvalidOperand(MLOperand),

    /// Shape inference failed: `source`.
    #[error("Shape inference failed: {source}")]
    ShapeInferenceError {
        #[from]
        source: Box<ShapeInferenceError>,
    },

    /// Internal data type cannot be converted to MLOperandDataType: `data_type`.
    #[error("Internal data type cannot be converted to MLOperandDataType: {data_type:?}")]
    InternalDataType { data_type: DataType },

    /// Tried to create a constant from a slice/vec with wrong size: required `required_size` bytes, provided `provided_size` bytes for descriptor `descriptor`.
    #[error(
        "Tried to create a constant from a slice/vec with wrong size: required {required_size} bytes, provided {provided_size} bytes for descriptor {descriptor:?}"
    )]
    WrongConstantSize {
        descriptor: MLOperandDescriptor,
        required_size: usize,
        provided_size: usize,
    },

    /// Build with invalid operand: operand `operand` assigned to name `name` is not part of this graph.
    #[error(
        "Build with invalid operand: operand {operand:?} assigned to name {name:?} is not part of this graph"
    )]
    BuildWithInvalidOperand { operand: MLOperand, name: String },

    /// A cache error occurred: `source`.
    #[error("A cache error occurred: {source}")]
    CacheError {
        #[from]
        source: CacheError,
    },

    /// Failed to save: output name ``name`` conflicts with existing `operand_kind` operand (exported `.webnn` references operands by global name).
    #[error(
        "Failed to save: output name `{name}` conflicts with existing {operand_kind} operand (exported `.webnn` references operands by global name)"
    )]
    OutputNameConflictsWithOperand {
        name: String,
        operand_kind: &'static str,
    },

    // TODO: this should not be needed, instead GraphInfo should ensure via methods to be always consistent
    // and impossible to construct invalid variants
    /// Internal error: inconsistent GraphInfo: `message`.
    #[error("Internal error: inconsistent GraphInfo: {message}")]
    InconsistentGraphInfo { message: String },
}

/// Errors of the WebNN API: context creation, building, tensors and dispatch.
///
/// Variant fields carry the values named in the message; they are not documented separately.
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum Error {
    /// No backend is available for context creation.
    #[error("No backend is available for context creation")]
    NoBackendAvailable {
        want_trtx: bool,
        have_trtx: bool,
        want_onnx: bool,
        have_onnx: bool,
        want_coreml: bool,
        have_coreml: bool,
        want_litert: bool,
        have_litert: bool,
        want_cann: bool,
        have_cann: bool,
    },
    /// No backend is available for backend hint: `backend_hint`.
    #[error("No backend is available for backend hint: {backend_hint:?}.")]
    NoBackendAvailableForBackendHint {
        backend_hint: Backend,
        want_trtx: bool,
        have_trtx: bool,
        want_onnx: bool,
        have_onnx: bool,
        want_coreml: bool,
        have_coreml: bool,
        want_litert: bool,
        have_litert: bool,
        want_cann: bool,
        have_cann: bool,
    },

    /// No device of selected type available.
    #[error("No device of selected type available")]
    NoDeviceAvailable,

    /// Lost MLContext: `0`.
    #[error("Lost MLContext: {0}")]
    ContextLost(String),

    // TODO: this error can at moment also occur in other situations than conversion. We should make
    // GraphError more specific to graph conversion
    /// Failed to convert graph: `source`.
    #[error("Failed to convert graph: {source}")]
    GraphConversionError {
        #[from]
        source: GraphError,
    },

    /// An error occurred while using the graph builder API: `source`.
    #[error("An error occurred while using the graph builder API: {source}")]
    GraphBuilderError {
        #[from]
        source: GraphBuilderError,
    },

    /// Failed to build graph: `source`.
    #[error("Failed to build graph: {source}")]
    GraphBuildError {
        #[from]
        source: Box<dyn std::error::Error + Sync + Send>,
    },

    /// Failed to run inference: `source`.
    #[error("Failed to run inference: {source}")]
    InferenceError {
        source: Box<dyn std::error::Error + Sync + Send>,
    },

    /// Failed to create context: `source`.
    #[error("Failed to create context: {source}")]
    ContextCreationError {
        source: Box<dyn std::error::Error + Sync + Send>,
    },

    /// Failed to create builder: `source`.
    #[error("Failed to create builder: {source}")]
    BuilderCreationError {
        source: Box<dyn std::error::Error + Sync + Send>,
    },

    /// Failed to create tensor: `source`.
    #[error("Failed to create tensor: {source}")]
    TensorCreationError {
        source: Box<dyn std::error::Error + Sync + Send>,
        descriptor: MLTensorDescriptor,
    },

    /// Failed to write tensor: `source`.
    #[error("Failed to write tensor: {source}")]
    TensorWriteError {
        source: Box<dyn std::error::Error + Sync + Send>,
        tensor: MLTensor,
    },

    /// Failed to read tensor: `source`.
    #[error("Failed to read tensor: {source}")]
    TensorReadError {
        source: Box<dyn std::error::Error + Sync + Send>,
        tensor: MLTensor,
    },

    /// Set capacity error: requested to set capacity to max shape `requested_shape` with `required_bytes` bytes but current shape is `current_shape` which needs `required_bytes` bytes.
    #[error(
        "Set capacity error: requested to set capacity to max shape {requested_shape:?} with {required_bytes} bytes but current shape is {current_shape:?} which needs {required_bytes} bytes"
    )]
    TensorCapacityError {
        requested_shape: Vec<u64>,
        current_shape: Vec<u64>,
        requested_bytes: u64,
        required_bytes: u64,
    },

    /// Tensor aliasing is not allowed in a dispatch call! The following tensor is used for multiple bindings (`first_binding` and `other_binding`): `aliased_tensor`.
    #[error(
        "Tensor aliasing is not allowed in a dispatch call! The following tensor is used for multiple bindings ({first_binding:?} and {other_binding:?}): {aliased_tensor:#?}"
    )]
    DuplicateTensorBinding {
        aliased_tensor: MLTensor,
        first_binding: String,
        other_binding: String,
    },

    /// Cannot write to non-writable tensor: `tensor`.
    #[error("Cannot write to non-writable tensor: {tensor:#?}")]
    WriteToNonWritableTensor { tensor: MLTensor },

    /// Write size does not match: write size: `write_size`, required size: `required_size`, tensor: `tensor`.
    #[error(
        "Write size does not match: write size: {write_size}, required size: {required_size}, tensor: {tensor:#?}"
    )]
    WrongWriteSize {
        write_size: usize,
        required_size: usize,
        tensor: MLTensor,
    },

    /// Read size does not match: read size: `read_size`, required size: `required_size`, tensor: `tensor`.
    #[error(
        "Read size does not match: read size: {read_size}, required size: {required_size}, tensor: {tensor:#?}"
    )]
    WrongReadSize {
        read_size: usize,
        required_size: usize,
        tensor: MLTensor,
    },

    /// Cannot read from non-readable tensor: `tensor`.
    #[error("Cannot read from non-readable tensor: {tensor:#?}")]
    ReadToNonReadableTensor { tensor: MLTensor },

    /// Failed to dispatch graph: `source`.
    #[error("Failed to dispatch graph: {source}")]
    GraphDispatchError {
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Failed to save graph: `source`.
    #[error("Failed to save graph: {source}")]
    GraphSaveError {
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
    /// An error in the Trtx: `source`.
    #[error("An error in the Trtx: {source}")]
    TrtxError {
        #[from]
        source: trtx::Error,
    },

    #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
    /// An error in the Trtx backend: `source`.
    #[error("An error in the Trtx backend: {source}")]
    TrtxBackendError {
        #[from]
        source: crate::backends::trtx::TrtxError,
    },

    #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
    /// An CUDA error occurred: `source`.
    #[error("An CUDA error occurred: {source}")]
    CudaError {
        #[from]
        source: DriverError,
    },
}

/// Errors of loading, validation, conversion and the legacy executors.
///
/// Variant fields carry the values named in the message; they are not documented separately.
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum GraphError {
    /// Graph file `path` could not be read: `source`.
    #[error("graph file {path} could not be read: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Graph JSON could not be parsed: `source`.
    #[error("graph JSON could not be parsed: {source}")]
    Parse {
        #[from]
        source: JsonError,
    },
    /// Graph must declare operands, operations, and outputs.
    #[error("graph must declare operands, operations, and outputs")]
    EmptyGraph,
    /// Graph declares `count` operands which exceeds the u32 id space.
    #[error("graph declares {count} operands which exceeds the u32 id space")]
    TooManyOperands { count: usize },
    /// Operand `operand` has a shape that overflows element count.
    #[error("operand {operand} has a shape that overflows element count")]
    OperandElementCountOverflow { operand: u32 },
    /// Operand `operand` exceeds tensor byte limit (`byte_length` > `limit`).
    #[error("operand {operand} exceeds tensor byte limit ({byte_length} > {limit})")]
    TensorLimit {
        operand: u32,
        byte_length: usize,
        limit: usize,
    },
    /// Input operand `operand` is missing a name.
    #[error("input operand {operand} is missing a name")]
    MissingInputName { operand: u32 },
    /// Input operand name ``name`` is duplicated.
    #[error("input operand name `{name}` is duplicated")]
    DuplicateInputName { name: String },
    /// Output operand `operand` is missing a name.
    #[error("output operand {operand} is missing a name")]
    MissingOutputName { operand: u32 },
    /// Output operand name ``name`` is duplicated.
    #[error("output operand name `{name}` is duplicated")]
    DuplicateOutputName { name: String },
    /// Operand `operand` uses unsupported IO data type `data_type`.
    #[error("operand {operand} uses unsupported IO data type {data_type:?}")]
    UnsupportedIoDataType { operand: u32, data_type: DataType },
    /// Backend `backend` does not support data type `data_type`.
    #[error("Backend {backend:?} does not support data type {data_type:?}")]
    UnsupportedDataType {
        data_type: DataType,
        backend: Backend,
    },
    /// Constant operand `operand` does not have data associated with it.
    #[error("constant operand {operand} does not have data associated with it")]
    MissingConstantData { operand: u32 },
    /// Constant operand `operand` byte mismatch (expected `expected`, got `actual`).
    #[error("constant operand {operand} byte mismatch (expected {expected}, got {actual})")]
    ConstantLengthMismatch {
        operand: u32,
        expected: usize,
        actual: usize,
    },
    // TODO: this indicates an invalid state of GraphError. Prevent this from occurring in GraphError and then remove the
    // error
    /// Graph input operand ids doesn't match the ids of OperandKind::Input: `input_ids` vs `input_ids_in_operands`.
    #[error(
        "graph input operand ids doesn't match the ids of OperandKind::Input: {input_ids:?} vs {input_ids_in_operands:?}"
    )]
    InputIdListMismatch {
        input_ids: Vec<u32>,
        input_ids_in_operands: Vec<u32>,
    },
    // TODO: this indicates an invalid state of GraphError. Prevent this from occurring in GraphError and then remove the
    // error
    /// Graph output operand ids doesn't match the ids of OperandKind::Input: `output_ids` vs `output_ids_in_operands`.
    #[error(
        "graph output operand ids doesn't match the ids of OperandKind::Input: {output_ids:?} vs {output_ids_in_operands:?}"
    )]
    OutputIdListMismatch {
        output_ids: Vec<u32>,
        output_ids_in_operands: Vec<u32>,
    },
    /// Graph input operand list does not match operand table.
    #[error("graph input operand list does not match operand table")]
    InputOperandListMismatch,
    /// Graph output operand list does not match operand table.
    #[error("graph output operand list does not match operand table")]
    OutputOperandListMismatch,
    /// Operand id `operand` referenced by ``operation`` is invalid.
    #[error("operand id {operand} referenced by `{operation}` is invalid")]
    InvalidOperandReference { operation: String, operand: u32 },
    /// Operation ``operation`` consumes operand `operand` before it is produced.
    #[error("operation `{operation}` consumes operand {operand} before it is produced")]
    OperandNotReady { operation: String, operand: u32 },
    /// Operation ``operation`` attempts to reuse operand `operand` as output.
    #[error("operation `{operation}` attempts to reuse operand {operand} as output")]
    OperandProducedTwice { operation: String, operand: u32 },
    /// Graph output operand `operand` is never produced by any operation.
    #[error("graph output operand {operand} is never produced by any operation")]
    OutputNotProduced { operand: u32 },
    /// Operand `operand` never feeds any operation.
    #[error("operand {operand} never feeds any operation")]
    OperandNeverUsed { operand: u32 },
    /// Graph contains unused constant data entries.
    #[error("graph contains unused constant data entries")]
    UnusedConstantHandles,
    /// Quantization validation failed for ``operation``: `reason`.
    #[error("quantization validation failed for `{operation}`: {reason}")]
    QuantizationValidation { operation: String, reason: String },
    /// Graph converter ``requested`` is not available. Supported: `available`.
    #[error("graph converter `{requested}` is not available. Supported: {available:?}")]
    UnknownConverter {
        requested: String,
        available: Vec<&'static str>,
    },
    /// Graph conversion failed for `format`: `reason`.
    #[error("graph conversion failed for {format}: {reason}")]
    ConversionFailed { format: String, reason: String },
    /// Operand id `operand` is invalid for conversion.
    #[error("operand id {operand} is invalid for conversion")]
    InvalidConversionOperand { operand: u32 },
    /// Graph could not be exported to `path`: `source`.
    #[error("graph could not be exported to {path}: {source}")]
    ExportIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Coreml runtime is only available on macOS with the `coreml-runtime` feature enabled.
    #[error("coreml runtime is only available on macOS with the `coreml-runtime` feature enabled")]
    CoremlRuntimeUnavailable,
    /// Coreml runtime failed: `reason`.
    #[error("coreml runtime failed: {reason}")]
    CoremlRuntimeFailed { reason: String },
    /// Coreml runtime only supports the coreml converter (got `format`).
    #[error("coreml runtime only supports the coreml converter (got {format})")]
    UnsupportedRuntimeFormat { format: String },
    /// Onnx runtime is only available with the `onnx-runtime` feature enabled.
    #[error("onnx runtime is only available with the `onnx-runtime` feature enabled")]
    OnnxRuntimeUnavailable,
    /// Onnx runtime failed: `reason`.
    #[error("onnx runtime failed: {reason}")]
    OnnxRuntimeFailed { reason: String },
    /// Tensorrt runtime is only available with the `trtx-runtime` or `trtx-runtime-mock` feature enabled.
    #[error(
        "tensorrt runtime is only available with the `trtx-runtime` or `trtx-runtime-mock` feature enabled"
    )]
    TrtxRuntimeUnavailable,
    /// Tensorrt runtime failed: `reason`.
    #[error("tensorrt runtime failed: {reason}")]
    TrtxRuntimeFailed { reason: String },
    /// Shape inference failed: `reason`.
    #[error("shape inference failed: {reason}")]
    ShapeInferenceFailed { reason: String },
    /// Graph uses dynamic dimensions but the `dynamic-inputs` feature is not enabled.
    #[error("graph uses dynamic dimensions but the `dynamic-inputs` feature is not enabled")]
    DynamicInputsFeatureDisabled,
    /// Device tensor operation failed: `reason`.
    #[error("device tensor operation failed: {reason}")]
    DeviceTensorFailed { reason: String },
    /// Device tensor has been destroyed.
    #[error("device tensor has been destroyed")]
    DeviceTensorDestroyed,
    /// Device tensor is not readable.
    #[error("device tensor is not readable")]
    DeviceTensorNotReadable,
    /// Device tensor is not writable.
    #[error("device tensor is not writable")]
    DeviceTensorNotWritable,
    /// Missing runtime `kind` tensor ``name``.
    #[error("missing runtime {kind} tensor `{name}`")]
    RuntimeTensorMissing { kind: String, name: String },
    /// Unexpected runtime `kind` tensor ``name``.
    #[error("unexpected runtime {kind} tensor `{name}`")]
    RuntimeTensorUnexpected { kind: String, name: String },
    /// Runtime `kind` tensor ``name`` rank mismatch (expected `expected_rank`, got `actual_rank`).
    #[error(
        "runtime {kind} tensor `{name}` rank mismatch (expected {expected_rank}, got {actual_rank})"
    )]
    RuntimeTensorRankMismatch {
        kind: String,
        name: String,
        expected_rank: usize,
        actual_rank: usize,
    },
    /// Runtime `kind` tensor ``name`` dimension `axis` mismatch (expected `expected`, got `actual`).
    #[error(
        "runtime {kind} tensor `{name}` dimension {axis} mismatch (expected {expected}, got {actual})"
    )]
    RuntimeStaticDimensionMismatch {
        kind: String,
        name: String,
        axis: usize,
        expected: u32,
        actual: usize,
    },
    /// Runtime `kind` tensor ``name`` dynamic dimension ``dim_name`` at axis `axis` exceeds maxSize (`actual` > `max_size`).
    #[error(
        "runtime {kind} tensor `{name}` dynamic dimension `{dim_name}` at axis {axis} exceeds maxSize ({actual} > {max_size})"
    )]
    RuntimeDynamicDimensionExceeded {
        kind: String,
        name: String,
        axis: usize,
        dim_name: String,
        max_size: u32,
        actual: usize,
    },
    /// Runtime dynamic dimension ``dim_name`` mismatch (expected `expected`, got `actual`).
    #[error("runtime dynamic dimension `{dim_name}` mismatch (expected {expected}, got {actual})")]
    RuntimeDynamicDimensionNameMismatch {
        dim_name: String,
        expected: usize,
        actual: usize,
    },
    /// Runtime tensor ``name`` shape `shape` overflows element count.
    #[error("runtime tensor `{name}` shape {shape:?} overflows element count")]
    RuntimeTensorShapeOverflow { name: String, shape: Vec<usize> },
    /// Runtime tensor ``name`` data length mismatch (expected `expected` elements, got `actual`).
    #[error(
        "runtime tensor `{name}` data length mismatch (expected {expected} elements, got {actual})"
    )]
    RuntimeTensorDataLengthMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
}

impl GraphError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        GraphError::Io {
            path: path.into(),
            source,
        }
    }

    /// `ExportIo` error for `path`.
    pub fn export(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        GraphError::ExportIo {
            path: path.into(),
            source,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_io_error_helper() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let graph_err = GraphError::io("test.json", io_err);

        match graph_err {
            GraphError::Io { path, .. } => {
                assert_eq!(path, PathBuf::from("test.json"));
            }
            _ => panic!("Expected Io error variant"),
        }
    }

    #[test]
    fn test_export_error_helper() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "permission denied");
        let graph_err = GraphError::export("/tmp/output.onnx", io_err);

        match graph_err {
            GraphError::ExportIo { path, .. } => {
                assert_eq!(path, PathBuf::from("/tmp/output.onnx"));
            }
            _ => panic!("Expected ExportIo error variant"),
        }
    }

    #[test]
    fn test_empty_graph_error_message() {
        let err = GraphError::EmptyGraph;
        let msg = format!("{}", err);
        assert_eq!(msg, "graph must declare operands, operations, and outputs");
    }

    #[test]
    fn test_too_many_operands_error() {
        let err = GraphError::TooManyOperands { count: 5_000_000 };
        let msg = format!("{}", err);
        assert!(msg.contains("5000000"));
        assert!(msg.contains("exceeds the u32 id space"));
    }

    #[test]
    fn test_tensor_limit_error() {
        let err = GraphError::TensorLimit {
            operand: 42,
            byte_length: 2_000_000,
            limit: 1_000_000,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("42"));
        assert!(msg.contains("2000000"));
        assert!(msg.contains("1000000"));
    }

    #[test]
    fn test_duplicate_input_name_error() {
        let err = GraphError::DuplicateInputName {
            name: "input_tensor".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("input_tensor"));
        assert!(msg.contains("duplicated"));
    }

    #[test]
    fn test_constant_length_mismatch_error() {
        let err = GraphError::ConstantLengthMismatch {
            operand: 10,
            expected: 1024,
            actual: 512,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("10"));
        assert!(msg.contains("1024"));
        assert!(msg.contains("512"));
    }

    #[test]
    fn test_invalid_operand_reference_error() {
        let err = GraphError::InvalidOperandReference {
            operation: "relu_1".to_string(),
            operand: 99,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("relu_1"));
        assert!(msg.contains("99"));
        assert!(msg.contains("invalid"));
    }

    #[test]
    fn test_operand_not_ready_error() {
        let err = GraphError::OperandNotReady {
            operation: "conv2d".to_string(),
            operand: 5,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("conv2d"));
        assert!(msg.contains("before it is produced"));
    }

    #[test]
    fn test_conversion_failed_error() {
        let err = GraphError::ConversionFailed {
            format: "ONNX".to_string(),
            reason: "unsupported operation".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("ONNX"));
        assert!(msg.contains("unsupported operation"));
    }

    #[test]
    fn test_runtime_unavailable_errors() {
        let coreml_err = GraphError::CoremlRuntimeUnavailable;
        assert!(format!("{}", coreml_err).contains("coreml runtime"));
        assert!(format!("{}", coreml_err).contains("macOS"));

        let onnx_err = GraphError::OnnxRuntimeUnavailable;
        assert!(format!("{}", onnx_err).contains("onnx runtime"));

        let trtx_err = GraphError::TrtxRuntimeUnavailable;
        assert!(format!("{}", trtx_err).contains("tensorrt runtime"));
    }

    #[test]
    fn test_device_tensor_errors() {
        let destroyed_err = GraphError::DeviceTensorDestroyed;
        assert!(format!("{}", destroyed_err).contains("destroyed"));

        let not_readable_err = GraphError::DeviceTensorNotReadable;
        assert!(format!("{}", not_readable_err).contains("not readable"));

        let not_writable_err = GraphError::DeviceTensorNotWritable;
        assert!(format!("{}", not_writable_err).contains("not writable"));
    }

    #[test]
    fn test_shape_inference_failed_error() {
        let err = GraphError::ShapeInferenceFailed {
            reason: "incompatible dimensions".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("shape inference failed"));
        assert!(msg.contains("incompatible dimensions"));
    }

    #[test]
    fn test_quantization_validation_error() {
        let err = GraphError::QuantizationValidation {
            operation: "quantizeLinear".to_string(),
            reason: "scale must be float32".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("quantizeLinear"));
        assert!(msg.contains("scale must be float32"));
    }
}

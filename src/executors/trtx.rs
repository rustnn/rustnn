#![cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]

use std::collections::HashMap;
use std::sync::Once;

use crate::error::GraphError;
use crate::graph::{OperandDescriptor, get_static_or_max_size};

// Reexport to allow downstream users (e.g. pywebnn) to set path to TensorRT RTX lib
pub use trtx::dynamically_load_tensorrt;

/// Bytes per element for TensorRT tensor data types (used for buffer sizing).
fn trt_dtype_bytes_per_element(dtype: &trtx::DataType) -> usize {
    use trtx::DataType;
    match *dtype {
        DataType::kFLOAT => 4,
        DataType::kHALF => 2,
        DataType::kINT8 | DataType::kUINT8 | DataType::kBOOL => 1,
        DataType::kINT32 => 4,
        DataType::kINT64 => 8,
        _ => 4,
    }
}

fn trt_dtype_to_string(dtype: &trtx::DataType) -> &'static str {
    use trtx::DataType;
    match *dtype {
        DataType::kFLOAT => "float32",
        DataType::kHALF => "float16",
        DataType::kINT8 => "int8",
        DataType::kINT32 => "int32",
        DataType::kUINT8 => "uint8",
        DataType::kBOOL => "bool",
        DataType::kINT64 => "int64",
        _ => "float32",
    }
}

static TRTX_INIT: Once = Once::new();

/// Log handler that only emits messages at or above a minimum severity.
/// Used when RUSTNN_TRTX_LOG_VERBOSITY is set (e.g. by WPT test runner).
struct FilteredStderrLogger {
    min_severity: trtx::Severity,
}

impl trtx::LogHandler for FilteredStderrLogger {
    fn log(&self, severity: trtx::Severity, message: &str) {
        if severity <= self.min_severity {
            eprintln!("[TensorRT {:?}] {}", severity, message);
        }
    }
}

/// Create a TensorRT logger. If RUSTNN_TRTX_LOG_VERBOSITY is set to
/// internal_error, error, warning, info, or verbose, uses a filtered logger;
/// otherwise uses unfiltered stderr.
pub(crate) fn create_trtx_logger() -> Result<trtx::Logger, GraphError> {
    let Some(verbosity) = std::env::var_os("RUSTNN_TRTX_LOG_VERBOSITY") else {
        return trtx::Logger::stderr().map_err(|e| GraphError::TrtxRuntimeFailed {
            reason: format!("failed to create TensorRT logger: {e}"),
        });
    };
    let min_severity = match verbosity.to_str().unwrap_or("").to_lowercase().as_str() {
        "internal_error" => trtx::Severity::InternalError,
        "error" => trtx::Severity::Error,
        "warning" => trtx::Severity::Warning,
        "info" => trtx::Severity::Info,
        "verbose" => trtx::Severity::Verbose,
        _ => {
            return trtx::Logger::stderr().map_err(|e| GraphError::TrtxRuntimeFailed {
                reason: format!("failed to create TensorRT logger: {e}"),
            });
        }
    };
    let handler = FilteredStderrLogger { min_severity };
    trtx::Logger::new(handler).map_err(|e| GraphError::TrtxRuntimeFailed {
        reason: format!("failed to create TensorRT logger: {e}"),
    })
}

/// Load TensorRT and ONNX parser libraries once per process (no-op when using mock).
/// Called by the converter and executor so the library is loaded before any trtx API use.
pub(crate) fn ensure_trtx_loaded() -> Result<(), GraphError> {
    #[cfg(feature = "trtx-runtime")]
    {
        let mut result = Ok(());
        TRTX_INIT.call_once(|| {
            result = trtx::dynamically_load_tensorrt(None::<&str>).map_err(|e| {
                GraphError::TrtxRuntimeFailed {
                    reason: format!("failed to load TensorRT library: {e}"),
                }
            });
            if result.is_ok() {
                result = trtx::dynamically_load_tensorrt_onnxparser(None::<&str>).map_err(|e| {
                    GraphError::TrtxRuntimeFailed {
                        reason: format!("failed to load TensorRT ONNX parser library: {e}"),
                    }
                });
            }
        });
        result
    }
    #[cfg(not(feature = "trtx-runtime"))]
    {
        TRTX_INIT.call_once(|| {});
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TrtxOutput {
    pub name: String,
    pub shape: Vec<i64>,
    pub data_type: String,
}

/// Input tensor for TensorRT execution. Caller provides raw bytes in the format
/// expected by the engine (e.g. f32 or f16 little-endian per element).
pub struct TrtxInput {
    pub name: String,
    /// Raw tensor bytes (length must match engine's expected size for this tensor).
    pub data: Vec<u8>,
}

/// Output tensor with raw bytes and data type so the caller can interpret or convert.
pub struct TrtxOutputWithData {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
    pub data_type: String,
}

/// Detect if bytes are ONNX format (starts with ONNX magic header)
fn is_onnx_format(bytes: &[u8]) -> bool {
    // ONNX protobuf files typically start with 0x08 or contain "onnx" in the header
    // For simplicity, check if it looks like protobuf
    bytes.len() >= 4 && (bytes[0] == 0x08 || bytes.starts_with(b"onnx"))
}

/// Run ONNX model or TensorRT engine with zero-filled inputs
/// This is useful for validation and testing graph structure
///
/// If model_bytes appears to be ONNX format, it will be parsed as ONNX and built into an engine.
/// Otherwise, it will be treated as a pre-serialized TensorRT engine.
pub fn run_trtx_zeroed(
    model_bytes: &[u8],
    inputs: &HashMap<String, OperandDescriptor>,
) -> Result<Vec<TrtxOutput>, GraphError> {
    ensure_trtx_loaded()?;

    if is_onnx_format(model_bytes) {
        // ONNX path: trtx crate expects f32 inputs
        let mut input_tensors = Vec::new();
        for (name, desc) in inputs {
            let shape: Vec<usize> = desc
                .shape
                .iter()
                .map(|s| get_static_or_max_size(s) as usize)
                .collect();
            let total: usize = shape.iter().product();
            input_tensors.push(trtx::executor::TensorInput {
                name: name.clone(),
                shape,
                data: vec![0f32; total.max(1)],
            });
        }
        let outputs =
            trtx::executor::run_onnx_with_tensorrt(model_bytes, &input_tensors).map_err(|e| {
                GraphError::TrtxRuntimeFailed {
                    reason: format!("TensorRT execution from ONNX failed: {e}"),
                }
            })?;
        return Ok(outputs
            .into_iter()
            .map(|o| TrtxOutput {
                name: o.name,
                shape: o.shape.iter().map(|&s| s as i64).collect(),
                data_type: "float32".to_string(),
            })
            .collect());
    }

    // Engine path: byte inputs (zeros)
    let mut input_list = Vec::new();
    for (name, desc) in inputs {
        let byte_len = desc.byte_length().unwrap_or(0).max(1);
        input_list.push(TrtxInput {
            name: name.clone(),
            data: vec![0u8; byte_len],
        });
    }
    let outputs = execute_trtx_engine(model_bytes, &input_list).map_err(|e| {
        GraphError::TrtxRuntimeFailed {
            reason: format!("TensorRT engine execution failed: {e}"),
        }
    })?;
    Ok(outputs
        .into_iter()
        .map(|o| TrtxOutput {
            name: o.name,
            shape: o.shape.iter().map(|&s| s as i64).collect(),
            data_type: o.data_type,
        })
        .collect())
}

/// Execute a pre-built TensorRT engine
fn execute_trtx_engine(
    engine_bytes: &[u8],
    inputs: &[TrtxInput],
) -> Result<Vec<TrtxOutputWithData>, trtx::Error> {
    // Create logger and runtime
    let logger = create_trtx_logger().map_err(|e| trtx::Error::Runtime(e.to_string()))?;
    let mut runtime = trtx::Runtime::new(&logger)?;

    // Deserialize engine
    let mut engine = runtime.deserialize_cuda_engine(engine_bytes)?;
    let mut context = engine.create_execution_context()?;

    // Get tensor information
    let num_tensors = engine.get_nb_io_tensors()?;

    // Prepare CUDA buffers for inputs and outputs
    let mut device_buffers: Vec<(String, trtx::DeviceBuffer)> = Vec::new();
    let mut output_info: Vec<(String, Vec<usize>, trtx::DataType)> = Vec::new();

    // Process each tensor - allocate buffers for ALL tensors (inputs and outputs)
    // TensorRT requires ALL tensor addresses to be set, even for intermediate results
    for i in 0..num_tensors {
        let name = engine.get_tensor_name(i)?;
        let dtype = engine.get_tensor_dtype(&name)?;
        let bytes_per_elem = trt_dtype_bytes_per_element(&dtype);

        if let Some(input) = inputs.iter().find(|inp| inp.name == name) {
            let expected_shape_i64 = engine.get_tensor_shape(&name)?;
            let expected_shape: Vec<usize> =
                expected_shape_i64.iter().map(|&d| d as usize).collect();
            let expected_size = expected_shape.iter().product::<usize>() * bytes_per_elem;

            if input.data.len() != expected_size {
                return Err(trtx::Error::InvalidArgument(format!(
                    "Input tensor '{}' size mismatch: expected {} bytes, got {}",
                    name,
                    expected_size,
                    input.data.len()
                )));
            }

            let mut buffer = trtx::DeviceBuffer::new(expected_size)?;
            buffer.copy_from_host(&input.data)?;

            unsafe {
                context.set_tensor_address(&name, buffer.as_ptr())?;
            }

            device_buffers.push((name.clone(), buffer));
        } else {
            // Non-input tensor (output or intermediate) - allocate buffer
            let shape_i64 = engine.get_tensor_shape(&name)?;
            let shape: Vec<usize> = shape_i64.iter().map(|&d| d as usize).collect();

            let num_elements: usize = shape.iter().product();
            let size_bytes = num_elements * bytes_per_elem;
            let buffer = trtx::DeviceBuffer::new(size_bytes)?;

            unsafe {
                context.set_tensor_address(&name, buffer.as_ptr())?;
            }

            output_info.push((name.clone(), shape, dtype));
            device_buffers.push((name.clone(), buffer));
        }
    }

    // Execute inference
    unsafe {
        context.enqueue_v3(trtx::cuda::get_default_stream())?;
    }

    // Synchronize to ensure completion
    trtx::cuda::synchronize()?;

    let mut outputs = Vec::new();
    for (name, shape, dtype) in output_info {
        if let Some((_, buffer)) = device_buffers.iter().find(|(n, _)| n == &name) {
            let size_bytes = shape.iter().product::<usize>() * trt_dtype_bytes_per_element(&dtype);
            let mut host_data = vec![0u8; size_bytes];
            buffer.copy_to_host(&mut host_data)?;
            outputs.push(TrtxOutputWithData {
                name,
                shape,
                data: host_data,
                data_type: trt_dtype_to_string(&dtype).to_string(),
            });
        }
    }
    Ok(outputs)
}

/// Run a pre-built TensorRT engine with the given byte inputs.
/// Caller is responsible for encoding inputs (e.g. f32 or f16) and decoding outputs.
/// ONNX model bytes are not supported here; use the trtx crate or run_trtx_zeroed for that.
pub fn run_trtx_with_inputs(
    engine_bytes: &[u8],
    inputs: Vec<TrtxInput>,
) -> Result<Vec<TrtxOutputWithData>, GraphError> {
    ensure_trtx_loaded()?;
    if is_onnx_format(engine_bytes) {
        return Err(GraphError::TrtxRuntimeFailed {
            reason: "run_trtx_with_inputs expects a serialized TensorRT engine, not ONNX bytes"
                .to_string(),
        });
    }
    execute_trtx_engine(engine_bytes, &inputs).map_err(|e| GraphError::TrtxRuntimeFailed {
        reason: format!("TensorRT engine execution failed: {e}"),
    })
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(feature = "trtx-runtime-mock")]
    fn test_trtx_executor_availability() {
        // This test just verifies the module compiles in mock mode
        // Real execution tests would require actual ONNX models
        assert!(true, "TensorRT executor module compiled successfully");
    }
}

#![cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]

use std::collections::HashMap;

use crate::error::GraphError;
use crate::graph::OperandDescriptor;
use crate::runtime_checks::{RuntimeShapeState, TensorKind, validate_shape_data_length};

#[derive(Debug, Clone)]
pub struct TrtxOutput {
    pub name: String,
    pub shape: Vec<i64>,
    pub data_type: String,
}

/// Input tensor data for TensorRT execution
pub struct TrtxInput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Output tensor with actual data
pub struct TrtxOutputWithData {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
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
    // Build zero-filled inputs from descriptors
    let mut input_tensors = Vec::new();
    for (name, desc) in inputs {
        let shape: Vec<usize> = desc
            .static_or_max_shape()
            .into_iter()
            .map(|d| d as usize)
            .collect();
        let total: usize = shape.iter().product();
        let zeros = vec![0f32; total.max(1)];

        input_tensors.push(trtx::executor::TensorInput {
            name: name.clone(),
            shape,
            data: zeros,
        });
    }

    // Determine if we have ONNX or pre-built TensorRT engine
    let outputs = if is_onnx_format(model_bytes) {
        // Parse ONNX and build engine, then execute
        trtx::executor::run_onnx_with_tensorrt(model_bytes, &input_tensors).map_err(|e| {
            GraphError::TrtxRuntimeFailed {
                reason: format!("TensorRT execution from ONNX failed: {e}"),
            }
        })?
    } else {
        // Directly execute pre-built TensorRT engine
        execute_trtx_engine(model_bytes, &input_tensors).map_err(|e| {
            GraphError::TrtxRuntimeFailed {
                reason: format!("TensorRT engine execution failed: {e}"),
            }
        })?
    };

    // Convert outputs to our format
    let mut results = Vec::new();
    for output in outputs {
        results.push(TrtxOutput {
            name: output.name,
            shape: output.shape.iter().map(|&s| s as i64).collect(),
            data_type: "f32".to_string(),
        });
    }

    Ok(results)
}

/// Execute a pre-built TensorRT engine
fn execute_trtx_engine(
    engine_bytes: &[u8],
    inputs: &[trtx::executor::TensorInput],
) -> Result<Vec<trtx::executor::TensorOutput>, trtx::Error> {
    // Create logger and runtime
    let logger = trtx::Logger::stderr()?;
    let runtime = trtx::Runtime::new(&logger)?;

    // Deserialize engine
    let engine = runtime.deserialize_cuda_engine(engine_bytes)?;
    let mut context = engine.create_execution_context()?;

    // Get tensor information
    let num_tensors = engine.get_nb_io_tensors()?;

    // Prepare CUDA buffers for inputs and outputs
    let mut device_buffers: Vec<(String, trtx::DeviceBuffer)> = Vec::new();
    let mut output_info: Vec<(String, Vec<usize>)> = Vec::new();

    // Process each tensor - allocate buffers for ALL tensors (inputs and outputs)
    // TensorRT requires ALL tensor addresses to be set, even for intermediate results
    for i in 0..num_tensors {
        let name = engine.get_tensor_name(i)?;

        // Check if this is an input tensor
        if let Some(input) = inputs.iter().find(|inp| inp.name == name) {
            // Input tensor - validate and copy data
            let expected_shape_i64 = engine.get_tensor_shape(&name)?;
            let expected_shape: Vec<usize> =
                expected_shape_i64.iter().map(|&d| d as usize).collect();
            let expected_elements: usize = expected_shape.iter().product();
            let provided_elements: usize = input.shape.iter().product();

            if provided_elements != expected_elements {
                return Err(trtx::Error::InvalidArgument(format!(
                    "Input tensor '{}' shape mismatch: expected {:?} ({} elements), got {:?} ({} elements)",
                    name, expected_shape, expected_elements, input.shape, provided_elements
                )));
            }

            if input.data.len() != provided_elements {
                return Err(trtx::Error::InvalidArgument(format!(
                    "Input tensor '{}' data length ({}) doesn't match shape {:?} ({} elements)",
                    name,
                    input.data.len(),
                    input.shape,
                    provided_elements
                )));
            }

            let size_bytes = input.data.len() * std::mem::size_of::<f32>();
            let mut buffer = trtx::DeviceBuffer::new(size_bytes)?;

            let input_bytes =
                unsafe { std::slice::from_raw_parts(input.data.as_ptr() as *const u8, size_bytes) };
            buffer.copy_from_host(input_bytes)?;

            unsafe {
                context.set_tensor_address(&name, buffer.as_ptr())?;
            }

            device_buffers.push((name.clone(), buffer));
        } else {
            // Non-input tensor (output or intermediate) - allocate buffer
            let shape_i64 = engine.get_tensor_shape(&name)?;
            let shape: Vec<usize> = shape_i64.iter().map(|&d| d as usize).collect();

            let num_elements: usize = shape.iter().product();
            let size_bytes = num_elements * std::mem::size_of::<f32>();
            let buffer = trtx::DeviceBuffer::new(size_bytes)?;

            unsafe {
                context.set_tensor_address(&name, buffer.as_ptr())?;
            }

            // Only return tensors whose names start with "output"
            if name.starts_with("output") {
                output_info.push((name.clone(), shape));
            }

            device_buffers.push((name.clone(), buffer));
        }
    }

    // Execute inference
    unsafe {
        context.enqueue_v3(trtx::cuda::get_default_stream())?;
    }

    // Synchronize to ensure completion
    trtx::cuda::synchronize()?;

    // Copy outputs back to host
    let mut outputs = Vec::new();

    for (name, shape) in output_info {
        if let Some((_, buffer)) = device_buffers.iter().find(|(n, _)| n == &name) {
            let size_bytes = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
            let mut host_data = vec![0u8; size_bytes];

            buffer.copy_to_host(&mut host_data)?;

            // Convert bytes to f32
            let data: Vec<f32> = unsafe {
                std::slice::from_raw_parts(
                    host_data.as_ptr() as *const f32,
                    size_bytes / std::mem::size_of::<f32>(),
                )
            }
            .to_vec();

            outputs.push(trtx::executor::TensorOutput { name, shape, data });
        }
    }

    Ok(outputs)
}

/// Run ONNX model with TensorRT using actual input tensors
/// This performs real inference and returns output data
pub fn run_trtx_with_inputs(
    model_bytes: &[u8],
    inputs: Vec<TrtxInput>,
) -> Result<Vec<TrtxOutputWithData>, GraphError> {
    run_trtx_with_inputs_impl(model_bytes, inputs, None, None)
}

/// Run TensorRT inference with runtime descriptor checks for dynamic dimensions.
pub fn run_trtx_with_inputs_checked(
    model_bytes: &[u8],
    inputs: Vec<TrtxInput>,
    input_descriptors: &HashMap<String, OperandDescriptor>,
    output_descriptors: &HashMap<String, OperandDescriptor>,
) -> Result<Vec<TrtxOutputWithData>, GraphError> {
    run_trtx_with_inputs_impl(
        model_bytes,
        inputs,
        Some(input_descriptors),
        Some(output_descriptors),
    )
}

fn run_trtx_with_inputs_impl(
    model_bytes: &[u8],
    inputs: Vec<TrtxInput>,
    input_descriptors: Option<&HashMap<String, OperandDescriptor>>,
    output_descriptors: Option<&HashMap<String, OperandDescriptor>>,
) -> Result<Vec<TrtxOutputWithData>, GraphError> {
    let mut runtime_shape_state = RuntimeShapeState::new();
    let mut actual_input_shapes = HashMap::new();
    for input in &inputs {
        validate_shape_data_length(&input.name, &input.shape, input.data.len())?;
        actual_input_shapes.insert(input.name.clone(), input.shape.clone());
    }
    if let Some(descriptors) = input_descriptors {
        runtime_shape_state.validate_named_shapes(
            &actual_input_shapes,
            descriptors,
            TensorKind::Input,
        )?;
    }

    // Convert our inputs to trtx format
    let trtx_inputs: Vec<trtx::executor::TensorInput> = inputs
        .into_iter()
        .map(|input| trtx::executor::TensorInput {
            name: input.name,
            shape: input.shape,
            data: input.data,
        })
        .collect();

    // Determine if we have ONNX or pre-built TensorRT engine
    let outputs = if is_onnx_format(model_bytes) {
        // Parse ONNX and build engine, then execute
        trtx::executor::run_onnx_with_tensorrt(model_bytes, &trtx_inputs).map_err(|e| {
            GraphError::TrtxRuntimeFailed {
                reason: format!("TensorRT execution from ONNX failed: {e}"),
            }
        })?
    } else {
        // Directly execute pre-built TensorRT engine
        execute_trtx_engine(model_bytes, &trtx_inputs).map_err(|e| {
            GraphError::TrtxRuntimeFailed {
                reason: format!("TensorRT engine execution failed: {e}"),
            }
        })?
    };

    // Convert outputs to our format
    let mut results = Vec::new();
    for output in outputs {
        results.push(TrtxOutputWithData {
            name: output.name,
            shape: output.shape,
            data: output.data,
        });
    }

    if let Some(descriptors) = output_descriptors {
        let mut actual_output_shapes = HashMap::new();
        for output in &results {
            actual_output_shapes.insert(output.name.clone(), output.shape.clone());
        }
        runtime_shape_state.validate_named_shapes(
            &actual_output_shapes,
            descriptors,
            TensorKind::Output,
        )?;
    }

    Ok(results)
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

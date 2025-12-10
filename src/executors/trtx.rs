#![cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]

use std::collections::HashMap;

use crate::error::GraphError;
use crate::graph::OperandDescriptor;

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

/// Run ONNX model with TensorRT using zero-filled inputs
/// This is useful for validation and testing graph structure
pub fn run_trtx_zeroed(
    model_bytes: &[u8],
    inputs: &HashMap<String, OperandDescriptor>,
) -> Result<Vec<TrtxOutput>, GraphError> {
    // Build zero-filled inputs from descriptors
    let mut input_tensors = Vec::new();
    for (name, desc) in inputs {
        let shape: Vec<usize> = desc.shape.iter().map(|&s| s as usize).collect();
        let total: usize = shape.iter().product();
        let zeros = vec![0f32; total.max(1)];

        input_tensors.push(trtx::executor::TensorInput {
            name: name.clone(),
            shape,
            data: zeros,
        });
    }

    // Execute with TensorRT
    let outputs =
        trtx::executor::run_onnx_with_tensorrt(model_bytes, &input_tensors).map_err(|e| {
            GraphError::TrtxRuntimeFailed {
                reason: format!("TensorRT execution failed: {e}"),
            }
        })?;

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

/// Run ONNX model with TensorRT using actual input tensors
/// This performs real inference and returns output data
pub fn run_trtx_with_inputs(
    model_bytes: &[u8],
    inputs: Vec<TrtxInput>,
) -> Result<Vec<TrtxOutputWithData>, GraphError> {
    // Convert our inputs to trtx format
    let trtx_inputs: Vec<trtx::executor::TensorInput> = inputs
        .into_iter()
        .map(|input| trtx::executor::TensorInput {
            name: input.name,
            shape: input.shape,
            data: input.data,
        })
        .collect();

    // Execute with TensorRT
    let outputs =
        trtx::executor::run_onnx_with_tensorrt(model_bytes, &trtx_inputs).map_err(|e| {
            GraphError::TrtxRuntimeFailed {
                reason: format!("TensorRT execution failed: {e}"),
            }
        })?;

    // Convert outputs to our format
    let mut results = Vec::new();
    for output in outputs {
        results.push(TrtxOutputWithData {
            name: output.name,
            shape: output.shape,
            data: output.data,
        });
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(any(feature = "trtx-runtime-mock"))]
    fn test_trtx_executor_availability() {
        // This test just verifies the module compiles in mock mode
        // Real execution tests would require actual ONNX models
        assert!(true, "TensorRT executor module compiled successfully");
    }
}

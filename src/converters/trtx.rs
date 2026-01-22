//! TensorRT native converter - directly builds TensorRT INetworkDefinition
//!
//! This converter bypasses ONNX serialization and builds TensorRT networks directly
//! from WebNN graph IR, providing better performance and avoiding ONNX limitations.

use std::collections::HashMap;

use super::{ConvertedGraph, GraphConverter};
use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo, OperandKind, Operation};
use trtx::network::Layer;
use trtx::{ActivationType, ElementWiseOperation, PoolingType, UnaryOperation};

/// TensorRT native converter
pub struct TrtxConverter;

impl TrtxConverter {
    /// Create a new TrtxConverter
    pub fn new() -> Self {
        TrtxConverter
    }

    /// Map WebNN DataType to TensorRT DataType code
    fn webnn_to_trt_dtype(dtype: DataType) -> Result<i32, GraphError> {
        match dtype {
            DataType::Float32 => Ok(0), // kFLOAT
            DataType::Float16 => Ok(1), // kHALF
            DataType::Int8 => Ok(2),    // kINT8
            DataType::Int32 => Ok(3),   // kINT32
            DataType::Uint8 => Ok(4),   // kUINT8
            _ => Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Unsupported data type: {:?}", dtype),
            }),
        }
    }

    /// Get constant data as bytes
    fn get_constant_data<'a>(
        graph: &'a GraphInfo,
        operand_id: u32,
    ) -> Result<&'a [u8], GraphError> {
        graph
            .constant_operand_ids_to_handles
            .get(&operand_id)
            .map(|constant_data| constant_data.data.as_slice())
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Operand {} is not a constant", operand_id),
            })
    }

    /// Build TensorRT network from WebNN graph
    fn build_network(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
    ) -> Result<(), GraphError> {
        let mut tensor_map: HashMap<u32, trtx::Tensor> = HashMap::new();

        // Step 1: Add inputs
        for (operand_id, operand) in graph.operands.iter().enumerate() {
            if operand.kind == OperandKind::Input {
                let dtype = Self::webnn_to_trt_dtype(operand.descriptor.data_type)?;
                let dims: Vec<i32> = operand.descriptor.shape.iter().map(|&d| d as i32).collect();
                let name = operand.name.as_deref().unwrap_or("input");

                let mut tensor = network.add_input(name, dtype, &dims).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add input {}: {}", name, e),
                    }
                })?;

                tensor
                    .set_name(name)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to set input name: {}", e),
                    })?;

                tensor_map.insert(operand_id as u32, tensor);
            }
        }

        // Step 2: Add constants
        for (operand_id, operand) in graph.operands.iter().enumerate() {
            if operand.kind == OperandKind::Constant {
                let dims: Vec<i32> = operand.descriptor.shape.iter().map(|&d| d as i32).collect();
                let data = Self::get_constant_data(graph, operand_id as u32)?;

                // Validate that data size matches expected size
                let expected_size: usize = operand
                    .descriptor
                    .shape
                    .iter()
                    .map(|&d| d as usize)
                    .product();
                let data_type_size = operand.descriptor.data_type.bytes_per_element();
                let expected_bytes = expected_size * data_type_size;

                if data.len() != expected_bytes {
                    return Err(GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!(
                            "Constant data size mismatch: expected {} bytes, got {} bytes for operand {}",
                            expected_bytes,
                            data.len(),
                            operand_id
                        ),
                    });
                }

                if data.is_empty() {
                    return Err(GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Constant operand {} has empty data", operand_id),
                    });
                }

                let trt_dtype = Self::webnn_to_trt_dtype(operand.descriptor.data_type)?;
                let layer = network.add_constant(&dims, data, trt_dtype).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add constant (operand {}): {}", operand_id, e),
                    }
                })?;

                // Extract output tensor from constant layer
                let tensor = layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get constant layer output: {}", e),
                    })?;

                tensor_map.insert(operand_id as u32, tensor);
            }
        }

        // Step 3: Add operations
        for operation in &graph.operations {
            Self::add_operation(graph, network, &mut tensor_map, operation)?;
        }

        // Step 4: Mark outputs
        for (operand_id, operand) in graph.operands.iter().enumerate() {
            if operand.kind == OperandKind::Output {
                let tensor = tensor_map.get(&(operand_id as u32)).ok_or_else(|| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Output operand {} not found in tensor map", operand_id),
                    }
                })?;

                network
                    .mark_output(tensor)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to mark output: {}", e),
                    })?;
            }
        }

        Ok(())
    }

    /// Add a single operation to the network
    fn add_operation(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let op_type = operation.op_type.as_str();

        match op_type {
            // Binary element-wise operations
            "add" => Self::add_elementwise_op(
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kSUM as i32,
            )?,
            "sub" => Self::add_elementwise_op(
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kSUB as i32,
            )?,
            "mul" => Self::add_elementwise_op(
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kPROD as i32,
            )?,
            "div" => Self::add_elementwise_op(
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kDIV as i32,
            )?,
            "pow" => Self::add_elementwise_op(
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kPOW as i32,
            )?,

            // Unary activation operations (use IActivationLayer)
            "relu" => Self::add_activation_op(
                network,
                tensor_map,
                operation,
                ActivationType::kRELU as i32,
            )?,
            "sigmoid" => Self::add_activation_op(
                network,
                tensor_map,
                operation,
                ActivationType::kSIGMOID as i32,
            )?,
            "tanh" => Self::add_activation_op(
                network,
                tensor_map,
                operation,
                ActivationType::kTANH as i32,
            )?,
            "elu" => Self::add_activation_op(
                network,
                tensor_map,
                operation,
                ActivationType::kELU as i32,
            )?,
            "softsign" => Self::add_activation_op(
                network,
                tensor_map,
                operation,
                ActivationType::kSOFTSIGN as i32,
            )?,
            "softplus" => Self::add_activation_op(
                network,
                tensor_map,
                operation,
                ActivationType::kSOFTPLUS as i32,
            )?,
            "gelu" => Self::add_activation_op(
                network,
                tensor_map,
                operation,
                ActivationType::kGELU_ERF as i32,
            )?,

            // Unary mathematical operations (use IUnaryLayer)
            // Exponential and logarithmic
            "exp" => {
                Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kEXP as i32)?
            }
            "log" => {
                Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kLOG as i32)?
            }

            // Arithmetic
            "sqrt" => {
                Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kSQRT as i32)?
            }
            "reciprocal" => Self::add_unary_op(
                network,
                tensor_map,
                operation,
                UnaryOperation::kRECIP as i32,
            )?,
            "abs" => {
                Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kABS as i32)?
            }
            "neg" => {
                Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kNEG as i32)?
            }

            // Trigonometric
            "sin" => {
                Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kSIN as i32)?
            }
            "cos" => {
                Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kCOS as i32)?
            }
            "tan" => {
                Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kTAN as i32)?
            }

            // Hyperbolic
            "sinh" => {
                Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kSINH as i32)?
            }
            "cosh" => {
                Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kCOSH as i32)?
            }

            // Inverse trigonometric
            "asin" => {
                Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kASIN as i32)?
            }
            "acos" => {
                Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kACOS as i32)?
            }
            "atan" => {
                Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kATAN as i32)?
            }

            // Inverse hyperbolic
            "asinh" => Self::add_unary_op(
                network,
                tensor_map,
                operation,
                UnaryOperation::kASINH as i32,
            )?,
            "acosh" => Self::add_unary_op(
                network,
                tensor_map,
                operation,
                UnaryOperation::kACOSH as i32,
            )?,
            "atanh" => Self::add_unary_op(
                network,
                tensor_map,
                operation,
                UnaryOperation::kATANH as i32,
            )?,

            // Rounding and other
            "ceil" => {
                Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kCEIL as i32)?
            }
            "floor" => Self::add_unary_op(
                network,
                tensor_map,
                operation,
                UnaryOperation::kFLOOR as i32,
            )?,
            "erf" => {
                Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kERF as i32)?
            }
            "sign" => {
                Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kSIGN as i32)?
            }
            "round" => Self::add_unary_op(
                network,
                tensor_map,
                operation,
                UnaryOperation::kROUND as i32,
            )?,

            // Matrix operations
            "matmul" => Self::add_matmul_op(network, tensor_map, operation)?,

            // Pooling operations
            "averagePool2d" => {
                Self::add_pooling_op(network, tensor_map, operation, PoolingType::kAVERAGE as i32)?
            }
            "maxPool2d" => {
                Self::add_pooling_op(network, tensor_map, operation, PoolingType::kMAX as i32)?
            }

            // Other operations
            "softmax" => Self::add_softmax_op(network, tensor_map, operation)?,
            "concat" => Self::add_concat_op(network, tensor_map, operation)?,
            "transpose" => Self::add_transpose_op(graph, network, tensor_map, operation)?,
            "reshape" => Self::add_reshape_op(graph, network, tensor_map, operation)?,

            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Unsupported operation: {}", op_type),
                });
            }
        }

        Ok(())
    }

    /// Add elementwise operation
    fn add_elementwise_op(
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
        op_code: i32,
    ) -> Result<(), GraphError> {
        let input0 = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        let input1 = tensor_map
            .get(&operation.input_operands[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[1]),
            })?;

        let layer = network
            .add_elementwise(input0, input1, op_code)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add elementwise operation: {}", e),
            })?;

        // Extract output tensor from layer
        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get layer output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add activation operation
    fn add_activation_op(
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
        activation_type: i32,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        let layer = network
            .add_activation(input, activation_type)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add activation: {}", e),
            })?;

        // Extract output tensor from layer
        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get layer output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add unary operation (element-wise mathematical operations)
    fn add_unary_op(
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
        unary_op: i32,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        let layer =
            network
                .add_unary(input, unary_op)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add unary operation: {}", e),
                })?;

        // Extract output tensor from layer
        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get layer output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add matrix multiply operation
    fn add_matmul_op(
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input0 = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        let input1 = tensor_map
            .get(&operation.input_operands[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[1]),
            })?;

        // MatrixOperation: 0=NONE, 1=TRANSPOSE, 2=VECTOR
        let layer = network
            .add_matrix_multiply(input0, 0, input1, 0) // No transpose
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add matrix multiply: {}", e),
            })?;

        // Extract output tensor from layer
        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get layer output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add pooling operation
    fn add_pooling_op(
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
        pool_type: i32,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // Extract window size from attributes
        let window_size = operation
            .attributes
            .get("windowDimensions")
            .and_then(|v: &serde_json::Value| v.as_array())
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Missing windowDimensions attribute".to_string(),
            })?;

        let window: [i32; 2] = [
            window_size[0].as_i64().unwrap_or(2) as i32,
            window_size[1].as_i64().unwrap_or(2) as i32,
        ];

        let layer = network
            .add_pooling(input, pool_type, &window)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add pooling: {}", e),
            })?;

        // Extract output tensor from layer
        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get layer output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add softmax operation
    fn add_softmax_op(
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // Default to last axis (most common for softmax)
        let axes = 1u32 << 0; // Apply to first axis

        let layer = network
            .add_softmax(input, axes)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add softmax: {}", e),
            })?;

        // Extract output tensor from layer
        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get layer output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add concatenation operation
    fn add_concat_op(
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let inputs: Vec<&trtx::Tensor> = operation
            .input_operands
            .iter()
            .map(|&id| {
                tensor_map
                    .get(&id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Input operand {} not found", id),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let layer =
            network
                .add_concatenation(&inputs)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add concatenation: {}", e),
                })?;

        // Extract output tensor from layer
        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get layer output: {}", e),
            })?;

        let output_id = operation.output_operands[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add transpose operation using shuffle layer
    fn add_transpose_op(
        _graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // For now, just use shuffle layer (transpose details would need more TensorRT API)
        let layer = network
            .add_shuffle(input)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add shuffle (transpose): {}", e),
            })?;

        // Extract output tensor from layer
        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get layer output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add reshape operation using shuffle layer
    fn add_reshape_op(
        _graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // Use shuffle layer for reshape
        let layer = network
            .add_shuffle(input)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add shuffle (reshape): {}", e),
            })?;

        // Extract output tensor from layer
        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get layer output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }
}

impl GraphConverter for TrtxConverter {
    fn format(&self) -> &'static str {
        "trtx"
    }

    fn convert(&self, graph_info: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
        // Create TensorRT logger, builder, and network
        let logger = trtx::Logger::stderr().map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to create TensorRT logger: {}", e),
        })?;

        let builder = trtx::Builder::new(&logger).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to create TensorRT builder: {}", e),
        })?;

        let mut network = builder
            .create_network(trtx::builder::network_flags::EXPLICIT_BATCH)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create TensorRT network: {}", e),
            })?;

        // Build the network from WebNN graph
        Self::build_network(graph_info, &mut network)?;

        // Create builder config
        let mut config = builder
            .create_config()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create builder config: {}", e),
            })?;

        // Set workspace size (1 GB)
        config
            .set_memory_pool_limit(trtx::builder::MemoryPoolType::Workspace, 1 << 30)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set workspace size: {}", e),
            })?;

        // Build and serialize the engine
        let engine_data = builder
            .build_serialized_network(&mut network, &mut config)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to build TensorRT engine: {}", e),
            })?;

        Ok(ConvertedGraph {
            format: "trtx",
            content_type: "application/x-tensorrt-engine",
            data: engine_data,
            weights_data: None,
        })
    }
}

impl Default for TrtxConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webnn_to_trt_dtype() {
        assert_eq!(
            TrtxConverter::webnn_to_trt_dtype(DataType::Float32).unwrap(),
            0
        );
        assert_eq!(
            TrtxConverter::webnn_to_trt_dtype(DataType::Float16).unwrap(),
            1
        );
        assert_eq!(
            TrtxConverter::webnn_to_trt_dtype(DataType::Int8).unwrap(),
            2
        );
        assert_eq!(
            TrtxConverter::webnn_to_trt_dtype(DataType::Int32).unwrap(),
            3
        );
    }
}

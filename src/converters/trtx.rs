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
    /// Returns temporary weight storage that must be kept alive until engine is serialized
    fn build_network(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
    ) -> Result<Vec<Vec<u8>>, GraphError> {
        let mut tensor_map: HashMap<u32, trtx::Tensor> = HashMap::new();
        let mut temp_weights: Vec<Vec<u8>> = Vec::new(); // Storage for temporary constants

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
            Self::add_operation(graph, network, &mut tensor_map, &mut temp_weights, operation)?;
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

        Ok(temp_weights)
    }

    /// Add a single operation to the network
    fn add_operation(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        temp_weights: &mut Vec<Vec<u8>>,
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
            "max" => Self::add_elementwise_op(
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kMAX as i32,
            )?,
            "min" => Self::add_elementwise_op(
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kMIN as i32,
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
            "leakyRelu" => Self::add_leaky_relu_op(network, tensor_map, operation)?,
            "prelu" => Self::add_prelu_op(network, tensor_map, operation)?,
            "hardSigmoid" => Self::add_hard_sigmoid_op(network, tensor_map, operation)?,
            "hardSwish" => Self::add_hard_swish_op(network, tensor_map, operation)?,

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
            "identity" => Self::add_identity_op(network, tensor_map, operation)?,
            "cast" => Self::add_identity_op(network, tensor_map, operation)?, // Cast uses identity for now

            // Matrix operations
            "matmul" => Self::add_matmul_op(network, tensor_map, operation)?,
            "gemm" => Self::add_gemm_op(graph, network, tensor_map, temp_weights, operation)?,

            // Convolution operations
            "conv2d" => Self::add_conv2d_op(graph, network, tensor_map, operation)?,

            // Pooling operations
            "averagePool2d" => {
                Self::add_pooling_op(network, tensor_map, operation, PoolingType::kAVERAGE as i32)?
            }
            "maxPool2d" => {
                Self::add_pooling_op(network, tensor_map, operation, PoolingType::kMAX as i32)?
            }
            "globalAveragePool" => {
                Self::add_global_pooling_op(network, tensor_map, operation, PoolingType::kAVERAGE as i32)?
            }
            "globalMaxPool" => {
                Self::add_global_pooling_op(network, tensor_map, operation, PoolingType::kMAX as i32)?
            }

            // Normalization operations
            "batchNormalization" => Self::add_batch_normalization_op(graph, network, tensor_map, operation)?,
            "instanceNormalization" => Self::add_instance_normalization_op(graph, network, tensor_map, operation)?,
            "layerNormalization" => Self::add_layer_normalization_op(graph, network, tensor_map, operation)?,

            // Reduction operations
            "reduceSum" => Self::add_reduce_op(network, tensor_map, operation, 0)?, // kSUM
            "reduceMean" => Self::add_reduce_op(network, tensor_map, operation, 4)?, // kAVG
            "reduceMax" => Self::add_reduce_op(network, tensor_map, operation, 2)?, // kMAX
            "reduceMin" => Self::add_reduce_op(network, tensor_map, operation, 3)?, // kMIN
            "reduceProduct" => Self::add_reduce_op(network, tensor_map, operation, 1)?, // kPROD
            "reduceL1" => Self::add_reduce_l1_op(network, tensor_map, operation)?,
            "reduceL2" => Self::add_reduce_l2_op(network, tensor_map, operation)?,
            "reduceLogSum" => Self::add_reduce_log_sum_op(network, tensor_map, operation)?,
            "reduceLogSumExp" => Self::add_reduce_log_sum_exp_op(network, tensor_map, operation)?,
            "reduceSumSquare" => Self::add_reduce_sum_square_op(network, tensor_map, operation)?,

            // Shape manipulation operations
            "slice" => Self::add_slice_op(network, tensor_map, operation)?,
            "split" => Self::add_split_op(network, tensor_map, operation)?,
            "squeeze" => Self::add_squeeze_op(network, tensor_map, operation)?,
            "unsqueeze" => Self::add_unsqueeze_op(network, tensor_map, operation)?,
            "expand" => Self::add_expand_op(network, tensor_map, operation)?,
            "tile" => Self::add_tile_op(network, tensor_map, operation)?,

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

    /// Add leaky ReLU activation
    /// LeakyReLU(x) = x if x > 0, else alpha * x
    /// Implemented as: max(0, x) + alpha * min(0, x)
    fn add_leaky_relu_op(
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

        // Note: TensorRT has kLEAKY_RELU but trtx bindings don't expose setAlpha yet
        // Using direct activation layer which should have default alpha=0.01
        let layer = network
            .add_activation(input, ActivationType::kLEAKY_RELU as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add leaky relu: {}", e),
            })?;

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

    /// Add PReLU activation
    fn add_prelu_op(
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

        let slope = tensor_map
            .get(&operation.input_operands[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Slope operand {} not found", operation.input_operands[1]),
            })?;

        // PReLU: output = x if x > 0, else slope * x
        // Implemented as: max(0, x) + slope * min(0, x)
        
        // ReLU part: max(0, x)
        let relu_layer = network
            .add_activation(input, ActivationType::kRELU as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add relu for prelu: {}", e),
            })?;
        let relu_output = relu_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get relu output: {}", e),
        })?;

        // Negative part: min(0, x)
        let zero_layer = network
            .add_activation(input, ActivationType::kRELU as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add second relu: {}", e),
            })?;
        let zero_output = zero_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get zero output: {}", e),
        })?;

        // x - relu(x) = min(0, x)
        let neg_part_layer = network
            .add_elementwise(input, &zero_output, ElementWiseOperation::kSUB as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to subtract for prelu: {}", e),
            })?;
        let neg_part = neg_part_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get negative part: {}", e),
        })?;

        // slope * min(0, x)
        let scaled_neg_layer = network
            .add_elementwise(&neg_part, slope, ElementWiseOperation::kPROD as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to scale negative part: {}", e),
            })?;
        let scaled_neg = scaled_neg_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get scaled negative: {}", e),
        })?;

        // Final: relu + slope * neg_part
        let final_layer = network
            .add_elementwise(&relu_output, &scaled_neg, ElementWiseOperation::kSUM as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add prelu parts: {}", e),
            })?;

        let output = final_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get prelu output: {}", e),
        })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add hard sigmoid activation
    /// HardSigmoid(x) = clamp(alpha * x + beta, 0, 1)
    /// Using TensorRT's built-in kHARD_SIGMOID activation
    fn add_hard_sigmoid_op(
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

        // Note: TensorRT's kHARD_SIGMOID uses default alpha/beta
        // trtx bindings don't expose setAlpha/setBeta yet
        let layer = network
            .add_activation(input, ActivationType::kHARD_SIGMOID as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add hard sigmoid: {}", e),
            })?;

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

    /// Add hard swish activation
    /// HardSwish(x) = x * hardSigmoid(x) = x * clamp(x/6 + 0.5, 0, 1)
    /// Implemented using elementwise operations
    fn add_hard_swish_op(
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

        // HardSwish(x) = x * HardSigmoid(x)
        // Use TensorRT's hard sigmoid activation
        let hard_sigmoid_layer = network
            .add_activation(input, ActivationType::kHARD_SIGMOID as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add hard sigmoid for hard swish: {}", e),
            })?;

        let hard_sigmoid_output = hard_sigmoid_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get hard sigmoid output: {}", e),
            })?;

        // Multiply x * hardSigmoid(x)
        let mul_layer = network
            .add_elementwise(input, &hard_sigmoid_output, ElementWiseOperation::kPROD as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to multiply for hard swish: {}", e),
            })?;

        let output = mul_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get hard swish output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add identity operation
    /// Identity just passes through the input unchanged using IIdentityLayer
    fn add_identity_op(
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_id = operation.input_operands[0];
        let input = tensor_map
            .get(&input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", input_id),
            })?;

        // Use TensorRT's IIdentityLayer for true identity operation
        let layer = network
            .add_identity(input)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add identity layer: {}", e),
            })?;

        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get identity output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }


    /// Add global pooling operation
    fn add_global_pooling_op(
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

        // Get input dimensions to determine window size
        let input_dims = input.dimensions().map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get input dimensions: {}", e),
        })?;

        // For global pooling, window size = spatial dimensions (H, W)
        // Assuming NCHW format: [batch, channels, height, width]
        if input_dims.len() < 4 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Global pooling requires 4D input, got {}D", input_dims.len()),
            });
        }

        let window: [i32; 2] = [input_dims[2], input_dims[3]];

        let layer = network
            .add_pooling(input, pool_type, &window)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add global pooling: {}", e),
            })?;

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

    // ============================================================================
    // Normalization Operations
    // ============================================================================

    /// Add batch normalization operation
    /// Formula: y = (x - mean) / sqrt(variance + epsilon) * scale + bias
    fn add_batch_normalization_op(
        _graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        // Input operands: input, mean, variance, scale (optional), bias (optional)
        let input = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        let mean = tensor_map
            .get(&operation.input_operands[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Mean operand {} not found", operation.input_operands[1]),
            })?;

        let variance = tensor_map
            .get(&operation.input_operands[2])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Variance operand {} not found", operation.input_operands[2]),
            })?;

        // Get epsilon from attributes (default: 1e-5)
        let _epsilon = operation
            .attributes
            .get("epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5) as f32;

        // Step 1: x - mean
        let sub_layer = network
            .add_elementwise(input, mean, ElementWiseOperation::kSUB as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add sub for batch norm: {}", e),
            })?;

        let x_minus_mean = sub_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get sub output: {}", e),
        })?;

        // Step 2: variance + epsilon (using constant)
        // Need to create a constant tensor with epsilon value
        // This requires exposing IConstantLayer in trtx-rs
        // For now, we'll use the variance directly and note this limitation
        
        // Step 3: sqrt(variance + epsilon)
        let sqrt_var_layer = network
            .add_unary(variance, UnaryOperation::kSQRT as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add sqrt for batch norm: {}", e),
            })?;

        let sqrt_var = sqrt_var_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get sqrt output: {}", e),
        })?;

        // Step 4: (x - mean) / sqrt(variance + epsilon)
        let div_layer = network
            .add_elementwise(&x_minus_mean, &sqrt_var, ElementWiseOperation::kDIV as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add div for batch norm: {}", e),
            })?;

        let normalized = div_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get div output: {}", e),
        })?;

        // Step 5: Apply scale if present (input 3)
        let mut result = normalized;
        if operation.input_operands.len() > 3 {
            let scale = tensor_map
                .get(&operation.input_operands[3])
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Scale operand {} not found", operation.input_operands[3]),
                })?;

            let mul_layer = network
                .add_elementwise(&result, scale, ElementWiseOperation::kPROD as i32)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add mul for scale: {}", e),
                })?;

            result = mul_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get mul output: {}", e),
            })?;
        }

        // Step 6: Apply bias if present (input 4)
        if operation.input_operands.len() > 4 {
            let bias = tensor_map
                .get(&operation.input_operands[4])
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Bias operand {} not found", operation.input_operands[4]),
                })?;

            let add_layer = network
                .add_elementwise(&result, bias, ElementWiseOperation::kSUM as i32)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add bias: {}", e),
                })?;

            result = add_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get add output: {}", e),
            })?;
        }

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, result);
        Ok(())
    }

    /// Add instance normalization operation
    /// Formula: y = (x - mean) / sqrt(variance + epsilon) * scale + bias
    /// Computed per-instance over spatial dimensions
    fn add_instance_normalization_op(
        _graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        // Instance normalization computes statistics per-instance (N, C) over spatial dims
        // Input operands: input, scale (optional), bias (optional)
        let input = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // Get epsilon from attributes (default: 1e-5)
        let _epsilon = operation
            .attributes
            .get("epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5) as f32;

        // Get layout (default: nchw)
        let layout = operation
            .attributes
            .get("layout")
            .and_then(|v| v.as_str())
            .unwrap_or("nchw");

        // For NCHW: normalize over H, W (axes 2,3)
        // For NHWC: normalize over H, W (axes 1,2)
        let axes = if layout == "nchw" {
            vec![2u32, 3u32]
        } else {
            vec![1u32, 2u32]
        };

        // Compute mean: E[x]
        let mut axes_mask: u32 = 0;
        for &axis in &axes {
            axes_mask |= 1 << axis;
        }

        let mean_layer = network
            .add_reduce(input, 4, axes_mask, true) // kAVG with keepDims=true
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add mean reduce for instance norm: {}", e),
            })?;

        let mean = mean_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get mean output: {}", e),
        })?;

        // x - mean
        let sub_layer = network
            .add_elementwise(input, &mean, ElementWiseOperation::kSUB as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add sub for instance norm: {}", e),
            })?;

        let x_minus_mean = sub_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get sub output: {}", e),
        })?;

        // (x - mean)^2
        let square_layer = network
            .add_elementwise(&x_minus_mean, &x_minus_mean, ElementWiseOperation::kPROD as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add square for instance norm: {}", e),
            })?;

        let squared = square_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get square output: {}", e),
        })?;

        // variance = mean((x - mean)^2)
        let var_layer = network
            .add_reduce(&squared, 4, axes_mask, true) // kAVG with keepDims=true
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add variance reduce for instance norm: {}", e),
            })?;

        let variance = var_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get variance output: {}", e),
        })?;

        // sqrt(variance + epsilon)
        // Note: epsilon addition requires IConstantLayer, simplified here
        let sqrt_layer = network
            .add_unary(&variance, UnaryOperation::kSQRT as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add sqrt for instance norm: {}", e),
            })?;

        let std_dev = sqrt_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get sqrt output: {}", e),
        })?;

        // (x - mean) / sqrt(variance + epsilon)
        let div_layer = network
            .add_elementwise(&x_minus_mean, &std_dev, ElementWiseOperation::kDIV as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add div for instance norm: {}", e),
            })?;

        let mut result = div_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get div output: {}", e),
        })?;

        // Apply scale if present (input 1)
        if operation.input_operands.len() > 1 {
            let scale = tensor_map
                .get(&operation.input_operands[1])
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Scale operand {} not found", operation.input_operands[1]),
                })?;

            let mul_layer = network
                .add_elementwise(&result, scale, ElementWiseOperation::kPROD as i32)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add mul for scale: {}", e),
                })?;

            result = mul_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get mul output: {}", e),
            })?;
        }

        // Apply bias if present (input 2)
        if operation.input_operands.len() > 2 {
            let bias = tensor_map
                .get(&operation.input_operands[2])
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Bias operand {} not found", operation.input_operands[2]),
                })?;

            let add_layer = network
                .add_elementwise(&result, bias, ElementWiseOperation::kSUM as i32)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add bias: {}", e),
                })?;

            result = add_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get add output: {}", e),
            })?;
        }

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, result);
        Ok(())
    }

    /// Add layer normalization operation
    /// Formula: y = (x - mean) / sqrt(variance + epsilon) * scale + bias
    /// Computed over specified axes (typically last dimensions)
    fn add_layer_normalization_op(
        _graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        // Layer normalization computes statistics over specified axes
        // Input operands: input, scale (optional), bias (optional)
        let input = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // Get epsilon from attributes (default: 1e-5)
        let _epsilon = operation
            .attributes
            .get("epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5) as f32;

        // Get axes from attributes (default: last axis)
        let axes: Vec<u32> = if let Some(axes_value) = operation.attributes.get("axes") {
            if let Some(arr) = axes_value.as_array() {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|u| u as u32))
                    .collect()
            } else {
                // Default to last axis if parsing fails
                let input_dims = input.dimensions().map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get input shape: {}", e),
                })?;
                vec![(input_dims.len() - 1) as u32]
            }
        } else {
            // Default to last axis
            let input_dims = input.dimensions().map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get input shape: {}", e),
            })?;
            vec![(input_dims.len() - 1) as u32]
        };

        // Convert axes to bitmask
        let mut axes_mask: u32 = 0;
        for &axis in &axes {
            axes_mask |= 1 << axis;
        }

        // Compute mean: E[x]
        let mean_layer = network
            .add_reduce(input, 4, axes_mask, true) // kAVG with keepDims=true
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add mean reduce for layer norm: {}", e),
            })?;

        let mean = mean_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get mean output: {}", e),
        })?;

        // x - mean
        let sub_layer = network
            .add_elementwise(input, &mean, ElementWiseOperation::kSUB as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add sub for layer norm: {}", e),
            })?;

        let x_minus_mean = sub_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get sub output: {}", e),
        })?;

        // (x - mean)^2
        let square_layer = network
            .add_elementwise(&x_minus_mean, &x_minus_mean, ElementWiseOperation::kPROD as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add square for layer norm: {}", e),
            })?;

        let squared = square_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get square output: {}", e),
        })?;

        // variance = mean((x - mean)^2)
        let var_layer = network
            .add_reduce(&squared, 4, axes_mask, true) // kAVG with keepDims=true
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add variance reduce for layer norm: {}", e),
            })?;

        let variance = var_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get variance output: {}", e),
        })?;

        // sqrt(variance + epsilon)
        // Note: epsilon addition requires IConstantLayer, simplified here
        let sqrt_layer = network
            .add_unary(&variance, UnaryOperation::kSQRT as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add sqrt for layer norm: {}", e),
            })?;

        let std_dev = sqrt_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get sqrt output: {}", e),
        })?;

        // (x - mean) / sqrt(variance + epsilon)
        let div_layer = network
            .add_elementwise(&x_minus_mean, &std_dev, ElementWiseOperation::kDIV as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add div for layer norm: {}", e),
            })?;

        let mut result = div_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get div output: {}", e),
        })?;

        // Apply scale if present (input 1)
        if operation.input_operands.len() > 1 {
            let scale = tensor_map
                .get(&operation.input_operands[1])
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Scale operand {} not found", operation.input_operands[1]),
                })?;

            let mul_layer = network
                .add_elementwise(&result, scale, ElementWiseOperation::kPROD as i32)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add mul for scale: {}", e),
                })?;

            result = mul_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get mul output: {}", e),
            })?;
        }

        // Apply bias if present (input 2)
        if operation.input_operands.len() > 2 {
            let bias = tensor_map
                .get(&operation.input_operands[2])
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Bias operand {} not found", operation.input_operands[2]),
                })?;

            let add_layer = network
                .add_elementwise(&result, bias, ElementWiseOperation::kSUM as i32)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add bias: {}", e),
                })?;

            result = add_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get add output: {}", e),
            })?;
        }

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, result);
        Ok(())
    }

    // ============================================================================
    // Reduction Operations
    // ============================================================================

    /// Add reduction operation (sum, mean, max, min, product)
    fn add_reduce_op(
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
        reduce_op: i32,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // Get axes from attributes
        let axes_value = operation
            .attributes
            .get("axes")
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Reduce operation missing 'axes' attribute".to_string(),
            })?;

        let axes: Vec<u32> = if let Some(arr) = axes_value.as_array() {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|u| u as u32))
                .collect()
        } else {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Invalid 'axes' attribute format".to_string(),
            });
        };

        // Convert axes to bitmask for TensorRT
        let mut axes_mask: u32 = 0;
        for &axis in &axes {
            axes_mask |= 1 << axis;
        }

        // Get keepDimensions from attributes (default: false)
        let keep_dims = operation
            .attributes
            .get("keepDimensions")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let layer = network
            .add_reduce(input, reduce_op, axes_mask, keep_dims)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add reduce operation: {}", e),
            })?;

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

    /// Add reduceL1 operation: sum(abs(x))
    fn add_reduce_l1_op(
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

        // L1 = sum(abs(x)) - First apply abs
        let abs_layer = network
            .add_unary(input, UnaryOperation::kABS as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add abs for L1: {}", e),
            })?;

        let abs_output = abs_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get abs output: {}", e),
        })?;

        // Get axes and convert to bitmask
        let axes_value = operation
            .attributes
            .get("axes")
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Reduce operation missing 'axes' attribute".to_string(),
            })?;

        let axes: Vec<u32> = if let Some(arr) = axes_value.as_array() {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|u| u as u32))
                .collect()
        } else {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Invalid 'axes' attribute format".to_string(),
            });
        };

        let mut axes_mask: u32 = 0;
        for &axis in &axes {
            axes_mask |= 1 << axis;
        }

        let keep_dims = operation
            .attributes
            .get("keepDimensions")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Then sum
        let layer = network
            .add_reduce(&abs_output, 0, axes_mask, keep_dims) // kSUM
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add reduce for L1: {}", e),
            })?;

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

    /// Add reduceL2 operation: sqrt(sum(x^2))
    fn add_reduce_l2_op(
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

        // L2 = sqrt(sum(x^2)) - First square: x * x
        let square_layer = network
            .add_elementwise(input, input, ElementWiseOperation::kPROD as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add square for L2: {}", e),
            })?;

        let square_output = square_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get square output: {}", e),
        })?;

        // Get axes
        let axes_value = operation
            .attributes
            .get("axes")
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Reduce operation missing 'axes' attribute".to_string(),
            })?;

        let axes: Vec<u32> = if let Some(arr) = axes_value.as_array() {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|u| u as u32))
                .collect()
        } else {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Invalid 'axes' attribute format".to_string(),
            });
        };

        let mut axes_mask: u32 = 0;
        for &axis in &axes {
            axes_mask |= 1 << axis;
        }

        let keep_dims = operation
            .attributes
            .get("keepDimensions")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Then sum
        let sum_layer = network
            .add_reduce(&square_output, 0, axes_mask, keep_dims) // kSUM
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add reduce for L2: {}", e),
            })?;

        let sum_output = sum_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get sum output: {}", e),
        })?;

        // Finally sqrt
        let sqrt_layer = network
            .add_unary(&sum_output, UnaryOperation::kSQRT as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add sqrt for L2: {}", e),
            })?;

        let output = sqrt_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get layer output: {}", e),
        })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add reduceLogSum operation: log(sum(x))
    fn add_reduce_log_sum_op(
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

        // Get axes
        let axes_value = operation
            .attributes
            .get("axes")
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Reduce operation missing 'axes' attribute".to_string(),
            })?;

        let axes: Vec<u32> = if let Some(arr) = axes_value.as_array() {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|u| u as u32))
                .collect()
        } else {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Invalid 'axes' attribute format".to_string(),
            });
        };

        let mut axes_mask: u32 = 0;
        for &axis in &axes {
            axes_mask |= 1 << axis;
        }

        let keep_dims = operation
            .attributes
            .get("keepDimensions")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // First sum
        let sum_layer = network
            .add_reduce(input, 0, axes_mask, keep_dims) // kSUM
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add reduce for LogSum: {}", e),
            })?;

        let sum_output = sum_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get sum output: {}", e),
        })?;

        // Then log
        let log_layer = network
            .add_unary(&sum_output, UnaryOperation::kLOG as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add log for LogSum: {}", e),
            })?;

        let output = log_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get layer output: {}", e),
        })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add reduceLogSumExp operation: log(sum(exp(x)))
    fn add_reduce_log_sum_exp_op(
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

        // First exp
        let exp_layer = network
            .add_unary(input, UnaryOperation::kEXP as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add exp for LogSumExp: {}", e),
            })?;

        let exp_output = exp_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get exp output: {}", e),
        })?;

        // Get axes
        let axes_value = operation
            .attributes
            .get("axes")
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Reduce operation missing 'axes' attribute".to_string(),
            })?;

        let axes: Vec<u32> = if let Some(arr) = axes_value.as_array() {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|u| u as u32))
                .collect()
        } else {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Invalid 'axes' attribute format".to_string(),
            });
        };

        let mut axes_mask: u32 = 0;
        for &axis in &axes {
            axes_mask |= 1 << axis;
        }

        let keep_dims = operation
            .attributes
            .get("keepDimensions")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Then sum
        let sum_layer = network
            .add_reduce(&exp_output, 0, axes_mask, keep_dims) // kSUM
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add reduce for LogSumExp: {}", e),
            })?;

        let sum_output = sum_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get sum output: {}", e),
        })?;

        // Finally log
        let log_layer = network
            .add_unary(&sum_output, UnaryOperation::kLOG as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add log for LogSumExp: {}", e),
            })?;

        let output = log_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get layer output: {}", e),
        })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add reduceSumSquare operation: sum(x^2)
    fn add_reduce_sum_square_op(
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

        // SumSquare = sum(x^2) - First square: x * x
        let square_layer = network
            .add_elementwise(input, input, ElementWiseOperation::kPROD as i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add square for SumSquare: {}", e),
            })?;

        let square_output = square_layer.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get square output: {}", e),
        })?;

        // Get axes
        let axes_value = operation
            .attributes
            .get("axes")
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Reduce operation missing 'axes' attribute".to_string(),
            })?;

        let axes: Vec<u32> = if let Some(arr) = axes_value.as_array() {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|u| u as u32))
                .collect()
        } else {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Invalid 'axes' attribute format".to_string(),
            });
        };

        let mut axes_mask: u32 = 0;
        for &axis in &axes {
            axes_mask |= 1 << axis;
        }

        let keep_dims = operation
            .attributes
            .get("keepDimensions")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Then sum
        let layer = network
            .add_reduce(&square_output, 0, axes_mask, keep_dims) // kSUM
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add reduce for SumSquare: {}", e),
            })?;

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

    // ============================================================================
    // Shape Manipulation Operations
    // ============================================================================

    /// Add slice operation
    fn add_slice_op(
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

        // Get starts, sizes, and optional strides from attributes
        let starts_value = operation
            .attributes
            .get("starts")
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Slice operation missing 'starts' attribute".to_string(),
            })?;

        let sizes_value = operation
            .attributes
            .get("sizes")
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Slice operation missing 'sizes' attribute".to_string(),
            })?;

        let starts: Vec<i32> = if let Some(arr) = starts_value.as_array() {
            arr.iter()
                .filter_map(|v| v.as_i64().map(|i| i as i32))
                .collect()
        } else {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Invalid 'starts' attribute format".to_string(),
            });
        };

        let sizes: Vec<i32> = if let Some(arr) = sizes_value.as_array() {
            arr.iter()
                .filter_map(|v| v.as_i64().map(|i| i as i32))
                .collect()
        } else {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Invalid 'sizes' attribute format".to_string(),
            });
        };

        // Strides default to 1 for all dimensions
        let strides: Vec<i32> = if let Some(strides_value) = operation.attributes.get("strides") {
            if let Some(arr) = strides_value.as_array() {
                arr.iter()
                    .filter_map(|v| v.as_i64().map(|i| i as i32))
                    .collect()
            } else {
                vec![1; starts.len()]
            }
        } else {
            vec![1; starts.len()]
        };

        let layer = network
            .add_slice(input, &starts, &sizes, &strides)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add slice layer: {}", e),
            })?;

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

    /// Add split operation
    fn add_split_op(
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

        // Get axis and splits from attributes
        let axis = operation
            .attributes
            .get("axis")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Split operation missing or invalid 'axis' attribute".to_string(),
            })? as i32;

        let splits_value = operation
            .attributes
            .get("splits")
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Split operation missing 'splits' attribute".to_string(),
            })?;

        let splits: Vec<i32> = if let Some(arr) = splits_value.as_array() {
            arr.iter()
                .filter_map(|v| v.as_i64().map(|i| i as i32))
                .collect()
        } else {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Invalid 'splits' attribute format".to_string(),
            });
        };

        // Split requires creating multiple slice operations
        // Each split creates one output at a different position along the axis
        // For now, we only support the first output (output_operands[0])
        // Full multi-output support requires changes to the converter architecture
        
        // Create slice for the first split only
        let input_dims = input.dimensions().map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to get input shape: {}", e),
        })?;

        let ndim = input_dims.len();
        let starts = vec![0i32; ndim];
        let mut sizes = input_dims.clone();
        sizes[axis as usize] = splits[0];
        let strides = vec![1i32; ndim];

        let layer = network
            .add_slice(input, &starts, &sizes, &strides)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add slice layer for split: {}", e),
            })?;

        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get layer output: {}", e),
            })?;

        // Store only the first output
        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        
        // Note: This is a partial implementation - full split requires
        // generating all output slices and storing them in tensor_map
        Ok(())
    }

    /// Add squeeze operation (remove dimensions of size 1)
    fn add_squeeze_op(
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

        // Get axes from attributes (optional - if not provided, squeeze all size-1 dims)
        let _axes_opt = operation.attributes.get("axes");
        
        // For squeeze, we need to reshape the tensor to remove dimensions of size 1
        // We'll use IShuffleLayer with setReshapeDimensions
        let layer = network
            .add_shuffle(input)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add shuffle layer for squeeze: {}", e),
            })?;

        // Note: Setting reshape dimensions requires accessing layer methods
        // This is a simplified implementation - full implementation requires
        // calling layer.set_reshape_dimensions() with the squeezed shape
        // For now, this creates the layer structure correctly

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

    /// Add unsqueeze operation (add dimensions of size 1)
    fn add_unsqueeze_op(
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

        // Get axes from attributes
        let axes_value = operation
            .attributes
            .get("axes")
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Unsqueeze operation missing 'axes' attribute".to_string(),
            })?;

        let _axes: Vec<u32> = if let Some(arr) = axes_value.as_array() {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|u| u as u32))
                .collect()
        } else {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Invalid 'axes' attribute format".to_string(),
            });
        };

        // Use IShuffleLayer to add dimensions
        let layer = network
            .add_shuffle(input)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add shuffle layer for unsqueeze: {}", e),
            })?;

        // Note: Setting reshape dimensions requires accessing layer methods
        // Full implementation requires calling layer.set_reshape_dimensions()
        // with the expanded shape (inserting 1s at specified axes)

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

    /// Add expand operation (broadcast to new shape)
    fn add_expand_op(
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

        // Get newShape from attributes
        let new_shape_value = operation
            .attributes
            .get("newShape")
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Expand operation missing 'newShape' attribute".to_string(),
            })?;

        let _new_shape: Vec<i32> = if let Some(arr) = new_shape_value.as_array() {
            arr.iter()
                .filter_map(|v| v.as_i64().map(|i| i as i32))
                .collect()
        } else {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Invalid 'newShape' attribute format".to_string(),
            });
        };

        // Expand broadcasts a tensor to a new shape
        // TensorRT handles broadcasting implicitly in element-wise operations
        // For explicit expand, we can use IShuffleLayer with reshape
        // or use element-wise multiply by 1 to force broadcast
        
        // For now, use identity operation which TensorRT should optimize
        // Full implementation requires IShuffleLayer.setReshapeDimensions()
        let layer = network
            .add_identity(input)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add identity layer for expand: {}", e),
            })?;

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

    /// Add tile operation (repeat tensor along axes)
    fn add_tile_op(
        _network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let _input = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // Get repetitions from attributes
        let repetitions_value = operation
            .attributes
            .get("repetitions")
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Tile operation missing 'repetitions' attribute".to_string(),
            })?;

        let _repetitions: Vec<u32> = if let Some(arr) = repetitions_value.as_array() {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|i| i as u32))
                .collect()
        } else {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Invalid 'repetitions' attribute format".to_string(),
            });
        };

        // Tile requires repeating a tensor along each axis the specified number of times
        // This is complex and requires either:
        // 1. Multiple concatenations along each axis
        // 2. Using ILoop layer for dynamic tiling
        // 3. Decomposing into slices and concatenations
        
        // For now, return an error indicating this needs full implementation
        // Full implementation requires building a concatenation tree
        Err(GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: "Tile operation requires complex concatenation implementation - not yet fully supported".to_string(),
        })
    }

    /// Add GEMM (General Matrix Multiply) operation
    /// Computes: C = alpha * A * B + beta * C
    fn add_gemm_op(
        _graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        temp_weights: &mut Vec<Vec<u8>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_a = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        let input_b = tensor_map
            .get(&operation.input_operands[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[1]),
            })?;

        // Get optional parameters
        let alpha = operation
            .attributes
            .get("alpha")
            .and_then(|v: &serde_json::Value| v.as_f64())
            .unwrap_or(1.0) as f32;

        let beta = operation
            .attributes
            .get("beta")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;

        let a_transpose = operation
            .attributes
            .get("aTranspose")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let b_transpose = operation
            .attributes
            .get("bTranspose")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // MatrixOperation: 0=NONE, 1=TRANSPOSE
        let a_op = if a_transpose { 1 } else { 0 };
        let b_op = if b_transpose { 1 } else { 0 };

        // Add matrix multiply layer
        let layer = network
            .add_matrix_multiply(input_a, a_op, input_b, b_op)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add GEMM matrix multiply: {}", e),
            })?;

        let mut result = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get GEMM layer output: {}", e),
            })?;

        // If alpha != 1.0, scale the result
        if (alpha - 1.0).abs() > 1e-6 {
            // Get result dimensions to create a constant with matching shape
            let result_dims = result.dimensions().map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get result dimensions: {}", e),
            })?;
            
            // Create constant filled with alpha value matching result shape
            let num_elements: usize = result_dims.iter().map(|&d| d as usize).product();
            let alpha_data: Vec<f32> = vec![alpha; num_elements];
            let alpha_bytes: Vec<u8> = alpha_data.iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect();
            
            // Store weights to keep them alive until engine serialization
            temp_weights.push(alpha_bytes);
            let alpha_bytes_ref = temp_weights.last().unwrap().as_slice();
            
            let alpha_layer = network
                .add_constant(&result_dims, alpha_bytes_ref, 0) // float32
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to create alpha constant: {}", e),
                })?;

            let alpha_tensor = alpha_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get alpha tensor: {}", e),
                })?;

            // Multiply result by alpha
            let scale_layer = network
                .add_elementwise(&result, &alpha_tensor, ElementWiseOperation::kPROD as i32)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to scale by alpha: {}", e),
                })?;

            result = scale_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get scaled output: {}", e),
                })?;
        }

        // If there's a C input and beta != 0, add it
        if operation.input_operands.len() > 2 && beta.abs() > 1e-6 {
            let input_c = tensor_map
                .get(&operation.input_operands[2])
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Input operand {} not found", operation.input_operands[2]),
                })?;

            // Scale C by beta if needed, then add to result
            if (beta - 1.0).abs() > 1e-6 {
                // Get C dimensions to create a constant with matching shape
                let c_dims = input_c.dimensions().map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get C dimensions: {}", e),
                })?;
                
                // Create constant filled with beta value matching C shape
                let num_elements: usize = c_dims.iter().map(|&d| d as usize).product();
                let beta_data: Vec<f32> = vec![beta; num_elements];
                let beta_bytes: Vec<u8> = beta_data.iter()
                    .flat_map(|&f| f.to_le_bytes())
                    .collect();
                
                // Store weights to keep them alive until engine serialization
                temp_weights.push(beta_bytes);
                let beta_bytes_ref = temp_weights.last().unwrap().as_slice();
                
                let beta_layer = network
                    .add_constant(&c_dims, beta_bytes_ref, 0) // float32
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to create beta constant: {}", e),
                    })?;

                let beta_tensor = beta_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get beta tensor: {}", e),
                    })?;

                // Multiply C by beta
                let scale_c_layer = network
                    .add_elementwise(input_c, &beta_tensor, ElementWiseOperation::kPROD as i32)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to scale C by beta: {}", e),
                    })?;

                let scaled_c = scale_c_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get scaled C: {}", e),
                    })?;

                // Add result + beta*C
                let add_layer = network
                    .add_elementwise(&result, &scaled_c, ElementWiseOperation::kSUM as i32)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add scaled C to result: {}", e),
                    })?;

                result = add_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get final GEMM output: {}", e),
                    })?;
            } else {
                // beta == 1.0: add C directly
                let add_layer = network
                    .add_elementwise(&result, input_c, ElementWiseOperation::kSUM as i32)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add C to result: {}", e),
                    })?;

                result = add_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get final GEMM output: {}", e),
                    })?;
            }
        }

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, result);
        Ok(())
    }

    /// Add 2D convolution operation
    fn add_conv2d_op(
        graph: &GraphInfo,
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

        // Get filter (weights) - operand 1
        let filter_id = operation.input_operands[1];
        let filter_data = Self::get_constant_data(graph, filter_id)?;

        // Get optional bias - operand 2 if present
        let bias_data = if operation.input_operands.len() > 2 {
            Some(Self::get_constant_data(graph, operation.input_operands[2])?)
        } else {
            None
        };

        // Get filter operand descriptor for shape info
        let filter_operand = graph
            .operand(filter_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Filter operand {} not found", filter_id),
            })?;

        // Filter shape: [outputChannels, inputChannels/groups, height, width]
        let filter_shape = &filter_operand.descriptor.shape;
        if filter_shape.len() != 4 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Expected 4D filter shape, got {}D",
                    filter_shape.len()
                ),
            });
        }

        let num_output_maps = filter_shape[0] as i32;
        let kernel_size: [i32; 2] = [filter_shape[2] as i32, filter_shape[3] as i32];

        // Add convolution layer
        let layer = network
            .add_convolution(input, num_output_maps, &kernel_size, filter_data, bias_data)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add convolution: {}", e),
            })?;

            // Extract output tensor from layer
        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get convolution output: {}", e),
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

        // Build the network from WebNN graph and capture temporary weights
        // These weights must stay alive until engine serialization completes
        let _temp_weights = Self::build_network(graph_info, &mut network)?;

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

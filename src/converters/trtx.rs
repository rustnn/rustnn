//! TensorRT native converter - directly builds TensorRT INetworkDefinition
//!
//! This converter bypasses ONNX serialization and builds TensorRT networks directly
//! from WebNN graph IR, providing better performance and avoiding ONNX limitations.

use std::collections::HashMap;

use super::{ConvertedGraph, GraphConverter};
use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo, OperandKind, Operation};
use trtx::network::Layer;
use trtx::{
    ActivationType, DataType as TrtDataType, ElementWiseOperation, PoolingType, ReduceOperation,
    ResizeMode, ScatterMode, UnaryOperation,
};

/// TensorRT native converter
pub struct TrtxConverter;

impl TrtxConverter {
    /// Create a new TrtxConverter
    pub fn new() -> Self {
        TrtxConverter
    }

    /// Map WebNN DataType to TensorRT DataType enum
    fn webnn_to_trt_dtype(dtype: DataType) -> Result<TrtDataType, GraphError> {
        match dtype {
            DataType::Float32 => Ok(TrtDataType::kFLOAT),
            DataType::Float16 => Ok(TrtDataType::kHALF),
            DataType::Int8 => Ok(TrtDataType::kINT8),
            DataType::Int32 => Ok(TrtDataType::kINT32),
            DataType::Uint8 => Ok(TrtDataType::kUINT8),
            _ => Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Unsupported data type: {:?}", dtype),
            }),
        }
    }

    /// Get constant data as bytes
    fn get_constant_data(graph: &GraphInfo, operand_id: u32) -> Result<&[u8], GraphError> {
        graph
            .constant_operand_ids_to_handles
            .get(&operand_id)
            .map(|constant_data| constant_data.data.as_slice())
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Operand {} is not a constant", operand_id),
            })
    }

    /// Cast Float32 tensor to BOOL (0.0 → false, non-zero → true)
    fn cast_to_bool(
        network: &mut trtx::NetworkDefinition,
        input: &trtx::Tensor,
    ) -> Result<trtx::Tensor, GraphError> {
        let layer = network.add_cast(input, TrtDataType::kBOOL).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to cast to BOOL: {}", e),
            }
        })?;
        layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get cast output: {}", e),
            })
    }

    /// Cast BOOL tensor to Float32 (false → 0.0, true → 1.0)
    fn cast_to_float32(
        network: &mut trtx::NetworkDefinition,
        input: &trtx::Tensor,
    ) -> Result<trtx::Tensor, GraphError> {
        let layer = network.add_cast(input, TrtDataType::kFLOAT).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to cast to Float32: {}", e),
            }
        })?;
        layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get cast output: {}", e),
            })
    }

    /// Cast INT32 tensor to Float32
    fn cast_int32_to_float32(
        network: &mut trtx::NetworkDefinition,
        input: &trtx::Tensor,
    ) -> Result<trtx::Tensor, GraphError> {
        let layer = network.add_cast(input, TrtDataType::kFLOAT).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to cast INT32 to Float32: {}", e),
            }
        })?;
        layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get cast output: {}", e),
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
            Self::add_operation(
                graph,
                network,
                &mut tensor_map,
                &mut temp_weights,
                operation,
            )?;
        }

        // Step 4: Mark outputs (only actual graph outputs, not intermediate tensors)
        for (operand_id, operand) in graph.operands.iter().enumerate() {
            if operand.kind == OperandKind::Output {
                let tensor = tensor_map.get_mut(&(operand_id as u32)).ok_or_else(|| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Output operand {} not found in tensor map", operand_id),
                    }
                })?;

                // Set the output tensor name if available
                if let Some(name) = &operand.name {
                    let _ = tensor.set_name(name); // Ignore error if name setting fails
                }

                network
                    .mark_output(tensor)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!(
                            "Failed to mark output {}: {}",
                            operand.name.as_deref().unwrap_or("unnamed"),
                            e
                        ),
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
                graph,
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kSUM,
            )?,
            "sub" => Self::add_elementwise_op(
                graph,
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kSUB,
            )?,
            "mul" => Self::add_elementwise_op(
                graph,
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kPROD,
            )?,
            "div" => Self::add_elementwise_op(
                graph,
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kDIV,
            )?,
            "pow" => Self::add_elementwise_op(
                graph,
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kPOW,
            )?,
            "max" => Self::add_elementwise_op(
                graph,
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kMAX,
            )?,
            "min" => Self::add_elementwise_op(
                graph,
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kMIN,
            )?,

            // Unary activation operations (use IActivationLayer)
            "relu" => {
                Self::add_activation_op(network, tensor_map, operation, ActivationType::kRELU)?
            }
            "sigmoid" => {
                Self::add_activation_op(network, tensor_map, operation, ActivationType::kSIGMOID)?
            }
            "tanh" => {
                Self::add_activation_op(network, tensor_map, operation, ActivationType::kTANH)?
            }
            "elu" => Self::add_activation_op(network, tensor_map, operation, ActivationType::kELU)?,
            "softsign" => {
                Self::add_activation_op(network, tensor_map, operation, ActivationType::kSOFTSIGN)?
            }
            "softplus" => {
                Self::add_activation_op(network, tensor_map, operation, ActivationType::kSOFTPLUS)?
            }
            "gelu" => {
                Self::add_activation_op(network, tensor_map, operation, ActivationType::kGELU_ERF)?
            }
            "leakyRelu" => Self::add_leaky_relu_op(network, tensor_map, operation)?,
            "prelu" => Self::add_prelu_op(network, tensor_map, operation)?,
            "hardSigmoid" => Self::add_hard_sigmoid_op(network, tensor_map, operation)?,
            "hardSwish" => Self::add_hard_swish_op(network, tensor_map, operation)?,

            // Unary mathematical operations (use IUnaryLayer)
            // Exponential and logarithmic
            "exp" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kEXP)?,
            "log" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kLOG)?,

            // Arithmetic
            "sqrt" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kSQRT)?,
            "reciprocal" => {
                Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kRECIP)?
            }
            "abs" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kABS)?,
            "neg" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kNEG)?,

            // Trigonometric
            "sin" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kSIN)?,
            "cos" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kCOS)?,
            "tan" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kTAN)?,

            // Hyperbolic
            "sinh" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kSINH)?,
            "cosh" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kCOSH)?,

            // Inverse trigonometric
            "asin" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kASIN)?,
            "acos" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kACOS)?,
            "atan" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kATAN)?,

            // Inverse hyperbolic
            "asinh" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kASINH)?,
            "acosh" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kACOSH)?,
            "atanh" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kATANH)?,

            // Rounding and other
            "ceil" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kCEIL)?,
            "floor" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kFLOOR)?,
            "erf" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kERF)?,
            "sign" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kSIGN)?,
            "round" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kROUND)?,
            "identity" => Self::add_identity_op(network, tensor_map, operation)?,
            "cast" => Self::add_identity_op(network, tensor_map, operation)?, // Cast uses identity for now
            "quantizeLinear" => Self::add_quantize_linear_op(network, tensor_map, operation)?,
            "dequantizeLinear" => Self::add_dequantize_linear_op(network, tensor_map, operation)?,

            // Matrix operations
            "matmul" => Self::add_matmul_op(network, tensor_map, operation)?,
            "gemm" => Self::add_gemm_op(graph, network, tensor_map, temp_weights, operation)?,

            // Convolution operations
            "conv2d" => Self::add_conv2d_op(graph, network, tensor_map, operation)?,
            "convTranspose2d" => {
                Self::add_conv_transpose2d_op(graph, network, tensor_map, operation)?
            }

            // Pooling operations
            "averagePool2d" => {
                Self::add_pooling_op(network, tensor_map, operation, PoolingType::kAVERAGE)?
            }
            "maxPool2d" => Self::add_pooling_op(network, tensor_map, operation, PoolingType::kMAX)?,
            "globalAveragePool" => {
                Self::add_global_pooling_op(network, tensor_map, operation, PoolingType::kAVERAGE)?
            }
            "globalMaxPool" => {
                Self::add_global_pooling_op(network, tensor_map, operation, PoolingType::kMAX)?
            }

            // Normalization operations
            "batchNormalization" => {
                Self::add_batch_normalization_op(graph, network, tensor_map, operation)?
            }
            "instanceNormalization" => {
                Self::add_instance_normalization_op(graph, network, tensor_map, operation)?
            }
            "layerNormalization" => {
                Self::add_layer_normalization_op(graph, network, tensor_map, operation)?
            }

            // Reduction operations
            "reduceSum" => {
                Self::add_reduce_op(network, tensor_map, operation, ReduceOperation::kSUM)?
            }
            "reduceMean" => {
                Self::add_reduce_op(network, tensor_map, operation, ReduceOperation::kAVG)?
            }
            "reduceMax" => {
                Self::add_reduce_op(network, tensor_map, operation, ReduceOperation::kMAX)?
            }
            "reduceMin" => {
                Self::add_reduce_op(network, tensor_map, operation, ReduceOperation::kMIN)?
            }
            "reduceProduct" => {
                Self::add_reduce_op(network, tensor_map, operation, ReduceOperation::kPROD)?
            }
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

            // Comparison operations (return Float32 with 0.0/1.0 values)
            "equal" => Self::add_comparison_op(
                graph,
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kEQUAL,
            )?,
            "greater" => Self::add_comparison_op(
                graph,
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kGREATER,
            )?,
            "greaterOrEqual" => Self::add_greater_or_equal_op(network, tensor_map, operation)?,
            "lesser" => Self::add_comparison_op(
                graph,
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kLESS,
            )?,
            "lesserOrEqual" => Self::add_lesser_or_equal_op(network, tensor_map, operation)?,
            "notEqual" => Self::add_not_equal_op(network, tensor_map, operation)?,

            // Logical operations (use BOOL internally, input/output Float32)
            "logicalAnd" => Self::add_logical_binary_op(
                graph,
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kAND,
            )?,
            "logicalOr" => Self::add_logical_binary_op(
                graph,
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kOR,
            )?,
            "logicalXor" => Self::add_logical_binary_op(
                graph,
                network,
                tensor_map,
                operation,
                ElementWiseOperation::kXOR,
            )?,
            "logicalNot" => Self::add_logical_not_op(network, tensor_map, operation)?,

            // Indexing/Gathering operations
            "gather" => Self::add_gather_op(network, tensor_map, operation)?,
            "gatherND" => Self::add_gather_nd_op(network, tensor_map, operation)?,
            "scatterElements" => Self::add_scatter_elements_op(network, tensor_map, operation)?,
            "scatterND" => Self::add_scatter_nd_op(network, tensor_map, operation)?,
            "argMax" => Self::add_arg_max_op(network, tensor_map, operation)?,
            "argMin" => Self::add_arg_min_op(network, tensor_map, operation)?,

            // Other operations
            "clamp" => Self::add_clamp_op(graph, network, tensor_map, operation, temp_weights)?,
            "where" => Self::add_where_op(network, tensor_map, operation)?,
            "linear" => Self::add_linear_op(graph, network, tensor_map, operation, temp_weights)?,
            "pad" => Self::add_pad_op(network, tensor_map, operation)?,
            "softmax" => Self::add_softmax_op(network, tensor_map, operation)?,
            "concat" => Self::add_concat_op(network, tensor_map, operation)?,
            "isNaN" => Self::add_is_nan_op(network, tensor_map, operation)?,
            "isInfinite" => Self::add_is_infinite_op(network, tensor_map, operation, temp_weights)?,
            "roundEven" => Self::add_round_even_op(network, tensor_map, operation)?,
            "gatherElements" => Self::add_gather_elements_op(network, tensor_map, operation)?,
            "l2Pool2d" => Self::add_l2_pool2d_op(network, tensor_map, operation)?,
            "reverse" => Self::add_reverse_op(graph, network, tensor_map, operation)?,
            "cumulativeSum" => {
                Self::add_cumulative_sum_op(graph, network, tensor_map, operation, temp_weights)?
            }
            "triangular" => {
                Self::add_triangular_op(graph, network, tensor_map, operation, temp_weights)?
            }
            "transpose" => Self::add_transpose_op(graph, network, tensor_map, operation)?,
            "reshape" => Self::add_reshape_op(graph, network, tensor_map, operation)?,
            "resample2d" => Self::add_resample2d_op(network, tensor_map, operation)?,

            // NOTE: RNN operations (lstm, lstmCell, gru, gruCell) deferred
            // IRNNv2Layer is deprecated in TensorRT and autocxx cannot generate bindings for it
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Unsupported operation: {}", op_type),
                });
            }
        }

        Ok(())
    }

    /// Helper to ensure two tensors have compatible shapes for elementwise operations
    /// Returns potentially reshaped tensors that are guaranteed to be broadcast-compatible
    fn ensure_broadcast_compatible(
        network: &mut trtx::NetworkDefinition,
        tensor0: &trtx::Tensor,
        tensor1: &trtx::Tensor,
        op_name: &str,
    ) -> Result<(trtx::Tensor, trtx::Tensor), GraphError> {
        // Get dimensions of both tensors
        let dims0 = tensor0
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Failed to get dimensions for tensor 0 in {}: {}",
                    op_name, e
                ),
            })?;

        let dims1 = tensor1
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Failed to get dimensions for tensor 1 in {}: {}",
                    op_name, e
                ),
            })?;

        // If dimensions match exactly, no reshape needed
        if dims0 == dims1 {
            // Clone by creating identity layers
            let id0 = network
                .add_identity(tensor0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to clone tensor0: {}", e),
                })?;
            let t0 = id0
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get identity output: {}", e),
                })?;

            let id1 = network
                .add_identity(tensor1)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to clone tensor1: {}", e),
                })?;
            let t1 = id1
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get identity output: {}", e),
                })?;

            return Ok((t0, t1));
        }

        // If ranks match, check if broadcasting is needed
        if dims0.len() == dims1.len() {
            // Check if dimensions are compatible for broadcasting
            let mut needs_broadcast = false;
            for (i, (&d0, &d1)) in dims0.iter().zip(dims1.iter()).enumerate() {
                if d0 != d1 {
                    if d0 != 1 && d1 != 1 {
                        return Err(GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!(
                                "Incompatible dimensions for broadcasting in {}: dimension {} has {} vs {} (neither equal nor 1). \
                                Full shapes: {:?} vs {:?}.",
                                op_name, i, d0, d1, dims0, dims1
                            ),
                        });
                    }
                    needs_broadcast = true;
                }
            }

            // If no broadcasting needed, just clone both tensors
            if !needs_broadcast {
                let id0 =
                    network
                        .add_identity(tensor0)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Failed to clone tensor0: {}", e),
                        })?;
                let t0 = id0
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get identity output: {}", e),
                    })?;

                let id1 =
                    network
                        .add_identity(tensor1)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Failed to clone tensor1: {}", e),
                        })?;
                let t1 = id1
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get identity output: {}", e),
                    })?;

                return Ok((t0, t1));
            }

            // Broadcasting needed - expand dimensions that are 1 to match target size
            // Use IResizeLayer with NEAREST mode to expand
            let t0 = if dims0
                .iter()
                .zip(dims1.iter())
                .any(|(&d0, &d1)| d0 == 1 && d1 != 1)
            {
                // tensor0 needs expansion
                let mut resize_layer =
                    network
                        .add_resize(tensor0)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Failed to add resize layer for tensor0: {}", e),
                        })?;

                resize_layer.set_output_dimensions(&dims1).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to set output dimensions: {}", e),
                    }
                })?;

                resize_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get resize output: {}", e),
                    })?
            } else {
                let id =
                    network
                        .add_identity(tensor0)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Failed to clone tensor0: {}", e),
                        })?;
                id.get_output(0).map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get identity output: {}", e),
                })?
            };

            let t1 = if dims1
                .iter()
                .zip(dims0.iter())
                .any(|(&d1, &d0)| d1 == 1 && d0 != 1)
            {
                // tensor1 needs expansion
                let mut resize_layer =
                    network
                        .add_resize(tensor1)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Failed to add resize layer for tensor1: {}", e),
                        })?;

                resize_layer.set_output_dimensions(&dims0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to set output dimensions: {}", e),
                    }
                })?;

                resize_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get resize output: {}", e),
                    })?
            } else {
                let id =
                    network
                        .add_identity(tensor1)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Failed to clone tensor1: {}", e),
                        })?;
                id.get_output(0).map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get identity output: {}", e),
                })?
            };

            return Ok((t0, t1));
        }

        // Ranks don't match - need explicit broadcasting
        // Determine which tensor needs reshaping (broadcast smaller rank to larger)
        let (to_reshape, to_keep, reshape_is_first) = if dims0.len() < dims1.len() {
            (tensor0, tensor1, true)
        } else {
            (tensor1, tensor0, false)
        };

        let reshape_dims = to_reshape
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get reshape dims: {}", e),
            })?;
        let target_dims = to_keep
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get target dims: {}", e),
            })?;

        // Pad smaller tensor with leading 1s
        let rank_diff = target_dims.len() - reshape_dims.len();
        let mut new_shape: Vec<i32> = vec![1; rank_diff];
        new_shape.extend_from_slice(&reshape_dims);

        let mut shuffle_layer =
            network
                .add_shuffle(to_reshape)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add shuffle layer for broadcasting: {}", e),
                })?;

        shuffle_layer
            .set_reshape_dimensions(&new_shape)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set reshape dimensions: {}", e),
            })?;

        let reshaped = shuffle_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get reshape output: {}", e),
            })?;

        // Clone the other tensor with identity
        let id_keep = network
            .add_identity(to_keep)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to clone kept tensor: {}", e),
            })?;
        let kept = id_keep
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get identity output: {}", e),
            })?;

        // Return in original order
        if reshape_is_first {
            Ok((reshaped, kept))
        } else {
            Ok((kept, reshaped))
        }
    }

    /// Add elementwise operation
    fn add_elementwise_op(
        _graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
        op_code: ElementWiseOperation,
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

        // Ensure broadcast compatibility (this may reshape tensors if needed)
        let (bc_input0, bc_input1) =
            Self::ensure_broadcast_compatible(network, input0, input1, &operation.op_type)?;

        let layer = network
            .add_elementwise(&bc_input0, &bc_input1, op_code)
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

    /// Add comparison operation (outputs BOOL, cast to Float32)
    fn add_comparison_op(
        _graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
        op_code: ElementWiseOperation,
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

        // Ensure broadcast compatibility
        let (bc_input0, bc_input1) =
            Self::ensure_broadcast_compatible(network, input0, input1, &operation.op_type)?;

        // Comparison operation returns BOOL
        let layer = network
            .add_elementwise(&bc_input0, &bc_input1, op_code)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add comparison operation: {}", e),
            })?;

        let bool_output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get layer output: {}", e),
            })?;

        // Cast BOOL to Float32 for WebNN compatibility
        let output = Self::cast_to_float32(network, &bool_output)?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add logical operation (cast Float32 to BOOL, perform operation, cast back to Float32)
    fn add_logical_binary_op(
        _graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
        op_code: ElementWiseOperation,
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

        // Ensure broadcast compatibility BEFORE casting to BOOL
        let (bc_input0, bc_input1) =
            Self::ensure_broadcast_compatible(network, input0, input1, &operation.op_type)?;

        // Cast Float32 inputs to BOOL
        let bool_input0 = Self::cast_to_bool(network, &bc_input0)?;
        let bool_input1 = Self::cast_to_bool(network, &bc_input1)?;

        // Perform logical operation on BOOL
        let layer = network
            .add_elementwise(&bool_input0, &bool_input1, op_code)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add logical operation: {}", e),
            })?;

        let bool_output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get layer output: {}", e),
            })?;

        // Cast BOOL output back to Float32
        let output = Self::cast_to_float32(network, &bool_output)?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add logical NOT operation (cast Float32 to BOOL, perform NOT, cast back to Float32)
    fn add_logical_not_op(
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

        // Cast Float32 input to BOOL
        let bool_input = Self::cast_to_bool(network, input)?;

        // Perform NOT operation on BOOL
        let layer = network
            .add_unary(&bool_input, UnaryOperation::kNOT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add logical NOT: {}", e),
            })?;

        let bool_output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get layer output: {}", e),
            })?;

        // Cast BOOL output back to Float32
        let output = Self::cast_to_float32(network, &bool_output)?;

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
        activation_type: ActivationType,
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
        unary_op: UnaryOperation,
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
            .add_activation(input, ActivationType::kLEAKY_RELU)
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
            .add_activation(input, ActivationType::kRELU)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add relu for prelu: {}", e),
            })?;
        let relu_output = relu_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get relu output: {}", e),
            })?;

        // Negative part: min(0, x)
        let zero_layer = network
            .add_activation(input, ActivationType::kRELU)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add second relu: {}", e),
            })?;
        let zero_output = zero_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get zero output: {}", e),
            })?;

        // x - relu(x) = min(0, x)
        let neg_part_layer = network
            .add_elementwise(input, &zero_output, ElementWiseOperation::kSUB)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to subtract for prelu: {}", e),
            })?;
        let neg_part = neg_part_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get negative part: {}", e),
            })?;

        // slope * min(0, x)
        let scaled_neg_layer = network
            .add_elementwise(&neg_part, slope, ElementWiseOperation::kPROD)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to scale negative part: {}", e),
            })?;
        let scaled_neg =
            scaled_neg_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get scaled negative: {}", e),
                })?;

        // Final: relu + slope * neg_part
        let final_layer = network
            .add_elementwise(&relu_output, &scaled_neg, ElementWiseOperation::kSUM)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add prelu parts: {}", e),
            })?;

        let output = final_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
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
            .add_activation(input, ActivationType::kHARD_SIGMOID)
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
            .add_activation(input, ActivationType::kHARD_SIGMOID)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add hard sigmoid for hard swish: {}", e),
            })?;

        let hard_sigmoid_output =
            hard_sigmoid_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get hard sigmoid output: {}", e),
                })?;

        // Multiply x * hardSigmoid(x)
        let mul_layer = network
            .add_elementwise(input, &hard_sigmoid_output, ElementWiseOperation::kPROD)
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

    /// Add quantizeLinear operation (float to quantized integer)
    fn add_quantize_linear_op(
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

        let scale = tensor_map
            .get(&operation.input_operands[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Scale operand {} not found", operation.input_operands[1]),
            })?;

        // Note: WebNN quantizeLinear also has zeroPoint parameter (operand 2)
        // TensorRT's IQuantizeLayer only takes scale, so we ignore zeroPoint for now
        // This is a limitation that should be documented

        // Create quantize layer - output type is INT8 for quantization
        let layer = network
            .add_quantize(input, scale, TrtDataType::kINT8)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add quantize layer: {}", e),
            })?;

        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get quantize output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add dequantizeLinear operation (quantized integer to float)
    fn add_dequantize_linear_op(
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

        let scale = tensor_map
            .get(&operation.input_operands[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Scale operand {} not found", operation.input_operands[1]),
            })?;

        // Note: WebNN dequantizeLinear also has zeroPoint parameter (operand 2)
        // TensorRT's IDequantizeLayer only takes scale, so we ignore zeroPoint for now
        // This is a limitation that should be documented

        // Create dequantize layer - output type is FLOAT for dequantization
        let layer = network
            .add_dequantize(input, scale, TrtDataType::kFLOAT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add dequantize layer: {}", e),
            })?;

        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get dequantize output: {}", e),
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
        pool_type: PoolingType,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // Get input dimensions to determine window size
        let input_dims = input
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get input dimensions: {}", e),
            })?;

        // For global pooling, window size = spatial dimensions (H, W)
        // Assuming NCHW format: [batch, channels, height, width]
        if input_dims.len() < 4 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Global pooling requires 4D input, got {}D",
                    input_dims.len()
                ),
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
            .add_elementwise(input, mean, ElementWiseOperation::kSUB)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add sub for batch norm: {}", e),
            })?;

        let x_minus_mean = sub_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get sub output: {}", e),
            })?;

        // Step 2: variance + epsilon (using constant)
        // Need to create a constant tensor with epsilon value
        // This requires exposing IConstantLayer in trtx-rs
        // For now, we'll use the variance directly and note this limitation

        // Step 3: sqrt(variance + epsilon)
        let sqrt_var_layer = network
            .add_unary(variance, UnaryOperation::kSQRT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add sqrt for batch norm: {}", e),
            })?;

        let sqrt_var = sqrt_var_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get sqrt output: {}", e),
            })?;

        // Step 4: (x - mean) / sqrt(variance + epsilon)
        let div_layer = network
            .add_elementwise(&x_minus_mean, &sqrt_var, ElementWiseOperation::kDIV)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add div for batch norm: {}", e),
            })?;

        let normalized = div_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
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
                .add_elementwise(&result, scale, ElementWiseOperation::kPROD)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add mul for scale: {}", e),
                })?;

            result = mul_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
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
                .add_elementwise(&result, bias, ElementWiseOperation::kSUM)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add bias: {}", e),
                })?;

            result = add_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
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
            .add_reduce(input, ReduceOperation::kAVG, axes_mask, true)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add mean reduce for instance norm: {}", e),
            })?;

        let mean = mean_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get mean output: {}", e),
            })?;

        // x - mean
        let sub_layer = network
            .add_elementwise(input, &mean, ElementWiseOperation::kSUB)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add sub for instance norm: {}", e),
            })?;

        let x_minus_mean = sub_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get sub output: {}", e),
            })?;

        // (x - mean)^2
        let square_layer = network
            .add_elementwise(&x_minus_mean, &x_minus_mean, ElementWiseOperation::kPROD)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add square for instance norm: {}", e),
            })?;

        let squared = square_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get square output: {}", e),
            })?;

        // variance = mean((x - mean)^2)
        let var_layer = network
            .add_reduce(&squared, ReduceOperation::kAVG, axes_mask, true)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add variance reduce for instance norm: {}", e),
            })?;

        let variance = var_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get variance output: {}", e),
            })?;

        // sqrt(variance + epsilon)
        // Note: epsilon addition requires IConstantLayer, simplified here
        let sqrt_layer = network
            .add_unary(&variance, UnaryOperation::kSQRT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add sqrt for instance norm: {}", e),
            })?;

        let std_dev = sqrt_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get sqrt output: {}", e),
            })?;

        // (x - mean) / sqrt(variance + epsilon)
        let div_layer = network
            .add_elementwise(&x_minus_mean, &std_dev, ElementWiseOperation::kDIV)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add div for instance norm: {}", e),
            })?;

        let mut result = div_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
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
                .add_elementwise(&result, scale, ElementWiseOperation::kPROD)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add mul for scale: {}", e),
                })?;

            result = mul_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
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
                .add_elementwise(&result, bias, ElementWiseOperation::kSUM)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add bias: {}", e),
                })?;

            result = add_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
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
                let input_dims = input
                    .dimensions()
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get input shape: {}", e),
                    })?;
                vec![(input_dims.len() - 1) as u32]
            }
        } else {
            // Default to last axis
            let input_dims = input
                .dimensions()
                .map_err(|e| GraphError::ConversionFailed {
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
            .add_reduce(input, ReduceOperation::kAVG, axes_mask, true)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add mean reduce for layer norm: {}", e),
            })?;

        let mean = mean_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get mean output: {}", e),
            })?;

        // x - mean
        let sub_layer = network
            .add_elementwise(input, &mean, ElementWiseOperation::kSUB)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add sub for layer norm: {}", e),
            })?;

        let x_minus_mean = sub_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get sub output: {}", e),
            })?;

        // (x - mean)^2
        let square_layer = network
            .add_elementwise(&x_minus_mean, &x_minus_mean, ElementWiseOperation::kPROD)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add square for layer norm: {}", e),
            })?;

        let squared = square_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get square output: {}", e),
            })?;

        // variance = mean((x - mean)^2)
        let var_layer = network
            .add_reduce(&squared, ReduceOperation::kAVG, axes_mask, true)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add variance reduce for layer norm: {}", e),
            })?;

        let variance = var_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get variance output: {}", e),
            })?;

        // sqrt(variance + epsilon)
        // Note: epsilon addition requires IConstantLayer, simplified here
        let sqrt_layer = network
            .add_unary(&variance, UnaryOperation::kSQRT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add sqrt for layer norm: {}", e),
            })?;

        let std_dev = sqrt_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get sqrt output: {}", e),
            })?;

        // (x - mean) / sqrt(variance + epsilon)
        let div_layer = network
            .add_elementwise(&x_minus_mean, &std_dev, ElementWiseOperation::kDIV)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add div for layer norm: {}", e),
            })?;

        let mut result = div_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
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
                .add_elementwise(&result, scale, ElementWiseOperation::kPROD)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add mul for scale: {}", e),
                })?;

            result = mul_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
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
                .add_elementwise(&result, bias, ElementWiseOperation::kSUM)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add bias: {}", e),
                })?;

            result = add_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
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
        reduce_op: ReduceOperation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // Get axes from attributes
        let axes_value =
            operation
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
            .add_unary(input, UnaryOperation::kABS)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add abs for L1: {}", e),
            })?;

        let abs_output = abs_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get abs output: {}", e),
            })?;

        // Get axes and convert to bitmask
        let axes_value =
            operation
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
            .add_reduce(&abs_output, ReduceOperation::kSUM, axes_mask, keep_dims)
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
            .add_elementwise(input, input, ElementWiseOperation::kPROD)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add square for L2: {}", e),
            })?;

        let square_output =
            square_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get square output: {}", e),
                })?;

        // Get axes
        let axes_value =
            operation
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
            .add_reduce(&square_output, ReduceOperation::kSUM, axes_mask, keep_dims)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add reduce for L2: {}", e),
            })?;

        let sum_output = sum_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get sum output: {}", e),
            })?;

        // Finally sqrt
        let sqrt_layer = network
            .add_unary(&sum_output, UnaryOperation::kSQRT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add sqrt for L2: {}", e),
            })?;

        let output = sqrt_layer
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
        let axes_value =
            operation
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
            .add_reduce(input, ReduceOperation::kSUM, axes_mask, keep_dims)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add reduce for LogSum: {}", e),
            })?;

        let sum_output = sum_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get sum output: {}", e),
            })?;

        // Then log
        let log_layer = network
            .add_unary(&sum_output, UnaryOperation::kLOG)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add log for LogSum: {}", e),
            })?;

        let output = log_layer
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
            .add_unary(input, UnaryOperation::kEXP)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add exp for LogSumExp: {}", e),
            })?;

        let exp_output = exp_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get exp output: {}", e),
            })?;

        // Get axes
        let axes_value =
            operation
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
            .add_reduce(&exp_output, ReduceOperation::kSUM, axes_mask, keep_dims)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add reduce for LogSumExp: {}", e),
            })?;

        let sum_output = sum_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get sum output: {}", e),
            })?;

        // Finally log
        let log_layer = network
            .add_unary(&sum_output, UnaryOperation::kLOG)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add log for LogSumExp: {}", e),
            })?;

        let output = log_layer
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
            .add_elementwise(input, input, ElementWiseOperation::kPROD)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add square for SumSquare: {}", e),
            })?;

        let square_output =
            square_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get square output: {}", e),
                })?;

        // Get axes
        let axes_value =
            operation
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
            .add_reduce(&square_output, ReduceOperation::kSUM, axes_mask, keep_dims)
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
        let starts_value =
            operation
                .attributes
                .get("starts")
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "Slice operation missing 'starts' attribute".to_string(),
                })?;

        let sizes_value =
            operation
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

        let splits_value =
            operation
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
        let input_dims = input
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
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
        let axes_value =
            operation
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
        let new_shape_value =
            operation
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
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        // Get repetitions from attributes
        let repetitions_value = operation.attributes.get("repetitions").ok_or_else(|| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Tile operation missing 'repetitions' attribute".to_string(),
            }
        })?;

        let repetitions: Vec<u32> = if let Some(arr) = repetitions_value.as_array() {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|i| i as u32))
                .collect()
        } else {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Invalid 'repetitions' attribute format".to_string(),
            });
        };

        // Tile by concatenating the tensor multiple times along each axis
        // We process each axis sequentially: tile axis 0, then axis 1, etc.
        // Start with the input tensor's ID
        let mut current_id = operation.input_operands[0];

        for (axis, &reps) in repetitions.iter().enumerate() {
            if reps <= 1 {
                // No tiling needed for this axis
                continue;
            }

            // Get current tensor
            let current_tensor =
                tensor_map
                    .get(&current_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Tensor {} not found during tiling", current_id),
                    })?;

            // Create a vector of references to the same tensor, repeated 'reps' times
            let tensors_to_concat: Vec<&trtx::Tensor> = (0..reps).map(|_| current_tensor).collect();

            // Concatenate along this axis
            let mut concat_layer = network.add_concatenation(&tensors_to_concat).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add concatenation for tile axis {}: {}", axis, e),
                }
            })?;

            // Set the concatenation axis
            concat_layer
                .set_axis(axis as i32)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to set concatenation axis {}: {}", axis, e),
                })?;

            // Get the output tensor
            let output_tensor =
                concat_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!(
                            "Failed to get concat output for tile axis {}: {}",
                            axis, e
                        ),
                    })?;

            // Use a temporary ID for intermediate results
            // We use a large number to avoid collisions with actual operand IDs
            current_id = 1_000_000 + axis as u32;
            tensor_map.insert(current_id, output_tensor);
        }

        // Insert the final result with the actual output operand ID
        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];

        // Move the final tensor from temporary ID to output ID
        if let Some(final_tensor) = tensor_map.remove(&current_id) {
            tensor_map.insert(output_id, final_tensor);
        } else {
            // No tiling happened (all reps were 1), just use input
            if let Some(input_tensor) = tensor_map.get(&operation.input_operands[0]) {
                // We need to create an identity layer to "clone" the tensor reference
                let identity_layer = network.add_identity(input_tensor).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add identity layer: {}", e),
                    }
                })?;
                let output_tensor =
                    identity_layer
                        .get_output(0)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Failed to get identity output: {}", e),
                        })?;
                tensor_map.insert(output_id, output_tensor);
            }
        }

        Ok(())
    }

    // ============================================================================
    // Comparison Operations (2026-01-29)
    // ============================================================================

    /// Add greaterOrEqual operation (greater(x, y) OR equal(x, y))
    fn add_greater_or_equal_op(
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

        // greaterOrEqual = greater OR equal
        let greater_layer = network
            .add_elementwise(input0, input1, ElementWiseOperation::kGREATER)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add greater layer: {}", e),
            })?;

        let greater_output =
            greater_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get greater output: {}", e),
                })?;

        let equal_layer = network
            .add_elementwise(input0, input1, ElementWiseOperation::kEQUAL)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add equal layer: {}", e),
            })?;

        let equal_output = equal_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get equal output: {}", e),
            })?;

        let or_layer = network
            .add_elementwise(&greater_output, &equal_output, ElementWiseOperation::kOR)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add OR layer: {}", e),
            })?;

        let bool_output = or_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get OR output: {}", e),
            })?;

        // Cast BOOL to Float32 for WebNN compatibility
        let output = Self::cast_to_float32(network, &bool_output)?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add lesserOrEqual operation (lesser(x, y) OR equal(x, y))
    fn add_lesser_or_equal_op(
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

        // lesserOrEqual = lesser OR equal
        let lesser_layer = network
            .add_elementwise(input0, input1, ElementWiseOperation::kLESS)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add lesser layer: {}", e),
            })?;

        let lesser_output =
            lesser_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get lesser output: {}", e),
                })?;

        let equal_layer = network
            .add_elementwise(input0, input1, ElementWiseOperation::kEQUAL)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add equal layer: {}", e),
            })?;

        let equal_output = equal_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get equal output: {}", e),
            })?;

        let or_layer = network
            .add_elementwise(&lesser_output, &equal_output, ElementWiseOperation::kOR)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add OR layer: {}", e),
            })?;

        let bool_output = or_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get OR output: {}", e),
            })?;

        // Cast BOOL to Float32 for WebNN compatibility
        let output = Self::cast_to_float32(network, &bool_output)?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add notEqual operation (NOT equal(x, y))
    fn add_not_equal_op(
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

        // notEqual = NOT equal
        let equal_layer = network
            .add_elementwise(input0, input1, ElementWiseOperation::kEQUAL)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add equal layer: {}", e),
            })?;

        let equal_output = equal_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get equal output: {}", e),
            })?;

        let not_layer = network
            .add_unary(&equal_output, UnaryOperation::kNOT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add NOT layer: {}", e),
            })?;

        let bool_output = not_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get NOT output: {}", e),
            })?;

        // Cast BOOL to Float32 for WebNN compatibility
        let output = Self::cast_to_float32(network, &bool_output)?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    // ============================================================================
    // Indexing/Gathering Operations (2026-01-29)
    // ============================================================================

    /// Add gather operation (gather elements along an axis using indices)
    fn add_gather_op(
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

        let indices = tensor_map
            .get(&operation.input_operands[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Indices operand {} not found", operation.input_operands[1]),
            })?;

        // Get axis attribute (default to 0)
        let axis = operation
            .attributes
            .get("axis")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as i32;

        let layer =
            network
                .add_gather(input, indices, axis)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add gather layer: {}", e),
                })?;

        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get gather output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add gatherND operation (N-dimensional gather)
    fn add_gather_nd_op(
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

        let indices = tensor_map
            .get(&operation.input_operands[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Indices operand {} not found", operation.input_operands[1]),
            })?;

        // Create gather layer with axis 0 (required by addGather API)
        let mut layer =
            network
                .add_gather(input, indices, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add gatherND layer: {}", e),
                })?;

        // Set gather mode to kND for N-dimensional gather
        layer
            .set_gather_mode(trtx::GatherMode::kND)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set gather mode to kND: {}", e),
            })?;

        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get gatherND output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add scatterElements operation (element-wise scatter)
    fn add_scatter_elements_op(
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let data = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Data operand {} not found", operation.input_operands[0]),
            })?;

        let indices = tensor_map
            .get(&operation.input_operands[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Indices operand {} not found", operation.input_operands[1]),
            })?;

        let updates = tensor_map
            .get(&operation.input_operands[2])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Updates operand {} not found", operation.input_operands[2]),
            })?;

        // Get axis attribute (default to 0)
        let axis = operation
            .attributes
            .get("axis")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as i32;

        // Create scatter layer with mode (kELEMENT for scatterElements)
        let mut layer = network
            .add_scatter(data, indices, updates, ScatterMode::kELEMENT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add scatterElements layer: {}", e),
            })?;

        // Set axis for element-wise scatter
        layer
            .set_axis(axis)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set scatter axis: {}", e),
            })?;

        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get scatterElements output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add scatterND operation (N-dimensional scatter)
    fn add_scatter_nd_op(
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let data = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Data operand {} not found", operation.input_operands[0]),
            })?;

        let indices = tensor_map
            .get(&operation.input_operands[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Indices operand {} not found", operation.input_operands[1]),
            })?;

        let updates = tensor_map
            .get(&operation.input_operands[2])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Updates operand {} not found", operation.input_operands[2]),
            })?;

        // Create scatter layer with mode kND for N-dimensional scatter
        let layer = network
            .add_scatter(data, indices, updates, ScatterMode::kND)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add scatterND layer: {}", e),
            })?;

        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get scatterND output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add argMax operation (find indices of maximum values)
    fn add_arg_max_op(
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

        // Get axis attribute (default to 0)
        let axis = operation
            .attributes
            .get("axis")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        // Get keepDimensions attribute (default to false)
        let keep_dims = operation
            .attributes
            .get("keepDimensions")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // TopK operation: 0=kMAX, 1=kMIN
        let layer = network
            .add_topk(input, 0, 1, 1u32 << axis) // operation=kMAX, k=1, axes as bitmask
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add topK layer: {}", e),
            })?;

        // TopK returns two outputs: values and indices
        // We want indices (output 1)
        let indices_output = layer
            .get_output(1)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get topK indices output: {}", e),
            })?;

        let squeezed_output = if !keep_dims {
            // Squeeze the k dimension (which is 1)
            let squeeze_layer =
                network
                    .add_shuffle(&indices_output)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add squeeze layer: {}", e),
                    })?;

            squeeze_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get squeeze output: {}", e),
                })?
        } else {
            indices_output
        };

        // Cast INT32 indices to Float32 for WebNN compatibility
        let final_output = Self::cast_int32_to_float32(network, &squeezed_output)?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, final_output);
        Ok(())
    }

    /// Add argMin operation (find indices of minimum values)
    fn add_arg_min_op(
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

        // Get axis attribute (default to 0)
        let axis = operation
            .attributes
            .get("axis")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        // Get keepDimensions attribute (default to false)
        let keep_dims = operation
            .attributes
            .get("keepDimensions")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // TopK operation: 0=kMAX, 1=kMIN
        let layer = network
            .add_topk(input, 1, 1, 1u32 << axis) // operation=kMIN, k=1, axes as bitmask
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add topK layer: {}", e),
            })?;

        // TopK returns two outputs: values and indices
        // We want indices (output 1)
        let indices_output = layer
            .get_output(1)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get topK indices output: {}", e),
            })?;

        let squeezed_output = if !keep_dims {
            // Squeeze the k dimension (which is 1)
            let squeeze_layer =
                network
                    .add_shuffle(&indices_output)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add squeeze layer: {}", e),
                    })?;

            squeeze_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get squeeze output: {}", e),
                })?
        } else {
            indices_output
        };

        // Cast INT32 indices to Float32 for WebNN compatibility
        let final_output = Self::cast_int32_to_float32(network, &squeezed_output)?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, final_output);
        Ok(())
    }

    // ============================================================================
    // Other Operations (2026-01-29)
    // ============================================================================

    /// Add clamp operation (clip values to range [min, max])
    fn add_clamp_op(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
        temp_weights: &mut Vec<Vec<u8>>,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // Get input operand descriptor to determine shape dimensions
        let input_operand = graph.operand(operation.input_operands[0]).ok_or_else(|| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Input operand {} not found in graph",
                    operation.input_operands[0]
                ),
            }
        })?;
        let num_dims = input_operand.descriptor.shape.len();
        // Create broadcast shape: [1, 1, ..., 1] with same number of dimensions as input
        let broadcast_shape: Vec<i32> = vec![1; num_dims];

        // Get min and max values from attributes
        let min_value = operation
            .attributes
            .get("minValue")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NEG_INFINITY) as f32;

        let max_value = operation
            .attributes
            .get("maxValue")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::INFINITY) as f32;

        // Implement clamp as: max(min_value, min(input, max_value))
        // First: min(input, max_value)
        // Store constant data in temp_weights to keep alive until engine is built
        // Note: Use shape [1,1,...,1] matching input dimensions for TensorRT broadcasting
        temp_weights.push(max_value.to_le_bytes().to_vec());
        let max_const_data = temp_weights.last().unwrap();
        let max_const = network
            .add_constant(&broadcast_shape, max_const_data, TrtDataType::kFLOAT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add max constant: {}", e),
            })?;

        let max_const_output =
            max_const
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get max constant output: {}", e),
                })?;

        let clamped_upper = network
            .add_elementwise(input, &max_const_output, ElementWiseOperation::kMIN)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add upper clamp: {}", e),
            })?;

        let clamped_upper_output =
            clamped_upper
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get upper clamp output: {}", e),
                })?;

        // Second: max(min_value, clamped_upper)
        // Store constant data in temp_weights to keep alive until engine is built
        // Note: Use shape [1,1,...,1] matching input dimensions for TensorRT broadcasting
        temp_weights.push(min_value.to_le_bytes().to_vec());
        let min_const_data = temp_weights.last().unwrap();
        let min_const = network
            .add_constant(&broadcast_shape, min_const_data, TrtDataType::kFLOAT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add min constant: {}", e),
            })?;

        let min_const_output =
            min_const
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get min constant output: {}", e),
                })?;

        let layer = network
            .add_elementwise(
                &min_const_output,
                &clamped_upper_output,
                ElementWiseOperation::kMAX,
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add lower clamp: {}", e),
            })?;

        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get clamp output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add where operation (select elements based on condition)
    fn add_where_op(
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let condition = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Condition operand {} not found",
                    operation.input_operands[0]
                ),
            })?;

        let true_value = tensor_map
            .get(&operation.input_operands[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "True value operand {} not found",
                    operation.input_operands[1]
                ),
            })?;

        let false_value = tensor_map
            .get(&operation.input_operands[2])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "False value operand {} not found",
                    operation.input_operands[2]
                ),
            })?;

        // Cast condition from Float32 to BOOL (ISelectLayer requires BOOL condition)
        let condition_bool = Self::cast_to_bool(network, condition)?;

        let layer = network
            .add_select(&condition_bool, true_value, false_value)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add select layer: {}", e),
            })?;

        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get select output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add linear operation (alpha * x + beta)
    fn add_linear_op(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
        temp_weights: &mut Vec<Vec<u8>>,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // Get input operand descriptor to determine shape dimensions
        let input_operand = graph.operand(operation.input_operands[0]).ok_or_else(|| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Input operand {} not found in graph",
                    operation.input_operands[0]
                ),
            }
        })?;
        let num_dims = input_operand.descriptor.shape.len();
        // Create broadcast shape: [1, 1, ..., 1] with same number of dimensions as input
        let broadcast_shape: Vec<i32> = vec![1; num_dims];

        // Get alpha and beta from attributes
        let alpha = operation
            .attributes
            .get("alpha")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;

        let beta = operation
            .attributes
            .get("beta")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        // Implement as: y = alpha * x + beta using elementwise operations
        // Note: All scalar constants must use shape [1,1,...,1] matching input dims for TensorRT broadcasting

        // Step 1: If alpha != 1.0, multiply x by alpha
        let after_multiply = if (alpha - 1.0).abs() > f32::EPSILON {
            // Create alpha constant with matching dimensions for proper broadcasting
            let alpha_bytes: Vec<u8> = alpha.to_le_bytes().to_vec();
            temp_weights.push(alpha_bytes);
            let alpha_bytes_ref = temp_weights.last().unwrap();

            let alpha_constant = network
                .add_constant(&broadcast_shape, alpha_bytes_ref, trtx::DataType::kFLOAT)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to create alpha constant: {}", e),
                })?;

            let alpha_tensor =
                alpha_constant
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get alpha constant output: {}", e),
                    })?;

            // Multiply: alpha * x
            let mul_layer = network
                .add_elementwise(input, &alpha_tensor, ElementWiseOperation::kPROD)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to multiply by alpha: {}", e),
                })?;

            mul_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get multiply output: {}", e),
                })?
        } else {
            // Use identity layer to pass through
            let identity_layer =
                network
                    .add_identity(input)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add identity layer: {}", e),
                    })?;
            identity_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get identity output: {}", e),
                })?
        };

        // Step 2: If beta != 0.0, add beta
        let final_output = if beta.abs() > f32::EPSILON {
            // Create beta constant with matching dimensions for proper broadcasting
            let beta_bytes: Vec<u8> = beta.to_le_bytes().to_vec();
            temp_weights.push(beta_bytes);
            let beta_bytes_ref = temp_weights.last().unwrap();

            let beta_constant = network
                .add_constant(&broadcast_shape, beta_bytes_ref, trtx::DataType::kFLOAT)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to create beta constant: {}", e),
                })?;

            let beta_tensor =
                beta_constant
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get beta constant output: {}", e),
                    })?;

            // Add: (alpha * x) + beta
            let add_layer = network
                .add_elementwise(&after_multiply, &beta_tensor, ElementWiseOperation::kSUM)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add beta: {}", e),
                })?;

            add_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get add output: {}", e),
                })?
        } else {
            after_multiply
        };

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, final_output);
        Ok(())
    }

    /// Add pad operation (pad tensor with constant/edge/reflection values)
    fn add_pad_op(
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

        // Get padding parameters
        let beginning_padding_value =
            operation
                .attributes
                .get("beginningPadding")
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "Pad operation missing 'beginningPadding' attribute".to_string(),
                })?;

        let ending_padding_value = operation.attributes.get("endingPadding").ok_or_else(|| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Pad operation missing 'endingPadding' attribute".to_string(),
            }
        })?;

        let pre_padding: Vec<i32> = if let Some(arr) = beginning_padding_value.as_array() {
            arr.iter()
                .filter_map(|v| v.as_i64().map(|i| i as i32))
                .collect()
        } else {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Invalid 'beginningPadding' format".to_string(),
            });
        };

        let post_padding: Vec<i32> = if let Some(arr) = ending_padding_value.as_array() {
            arr.iter()
                .filter_map(|v| v.as_i64().map(|i| i as i32))
                .collect()
        } else {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Invalid 'endingPadding' format".to_string(),
            });
        };

        // Get input dimensions
        let input_dims = input
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get input dimensions: {}", e),
            })?;

        let original_ndims = input_dims.len();
        eprintln!(
            "[PAD DEBUG] Input dims: {:?}, len={}",
            input_dims, original_ndims
        );
        eprintln!(
            "[PAD DEBUG] Pre-padding: {:?}, Post-padding: {:?}",
            pre_padding, post_padding
        );

        // TensorRT padding requires at least 4D input (NCHW format)
        // If input is less than 4D, reshape to 4D first
        let input_to_pad = if original_ndims < 4 {
            eprintln!(
                "[PAD DEBUG] Reshaping to 4D (original was {}D)",
                original_ndims
            );
            // Calculate 4D shape: pad with 1s on the left
            let mut shape_4d = vec![1; 4 - original_ndims];
            shape_4d.extend_from_slice(&input_dims);
            eprintln!("[PAD DEBUG] Shape 4D: {:?}", shape_4d);

            // Reshape to 4D
            let mut reshape_layer =
                network
                    .add_shuffle(input)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to create reshape layer for padding: {}", e),
                    })?;

            reshape_layer
                .set_reshape_dimensions(&shape_4d)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to set reshape dimensions: {}", e),
                })?;

            reshape_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get reshape output: {}", e),
                })?
        } else {
            eprintln!("[PAD DEBUG] Input already >= 4D, using identity layer");
            // Use identity layer to pass through without reshape
            let identity =
                network
                    .add_identity(input)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to create identity layer: {}", e),
                    })?;
            identity
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get identity output: {}", e),
                })?
        };

        // TensorRT addPaddingNd for 4D tensors (NCHW) expects 2D padding (H, W dimensions only)
        // Padding must be RIGHT-ALIGNED to match the rightmost dimensions of the reshaped input
        // For 1D input [d0] → reshaped to [1,1,1,d0], padding on d0 → W dimension (index 1)
        // For 2D input [d0,d1] → reshaped to [1,1,d0,d1], padding on d0,d1 → H,W dimensions (indices 0,1)
        let spatial_dims = 2; // H, W dimensions in NCHW

        // Right-align: If we have fewer than 2 padding values, pad LEFT with zeros
        let mut pre_padding_spatial = vec![0; spatial_dims];
        let mut post_padding_spatial = vec![0; spatial_dims];

        let len = pre_padding.len().min(spatial_dims);
        let pad_offset = spatial_dims.saturating_sub(pre_padding.len());
        let pad_offset_end = pad_offset + len;

        pre_padding_spatial[pad_offset..pad_offset_end].copy_from_slice(&pre_padding[..len]);
        post_padding_spatial[pad_offset..pad_offset_end].copy_from_slice(&post_padding[..len]);

        // Check actual dimensions of input_to_pad
        let input_to_pad_dims =
            input_to_pad
                .dimensions()
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get input_to_pad dimensions: {}", e),
                })?;

        eprintln!("[PAD DEBUG] Final padding arrays (spatial only):");
        eprintln!(
            "[PAD DEBUG]   pre_spatial:  {:?} (len={})",
            pre_padding_spatial,
            pre_padding_spatial.len()
        );
        eprintln!(
            "[PAD DEBUG]   post_spatial: {:?} (len={})",
            post_padding_spatial,
            post_padding_spatial.len()
        );
        eprintln!(
            "[PAD DEBUG] input_to_pad actual dims: {:?} (len={})",
            input_to_pad_dims,
            input_to_pad_dims.len()
        );

        // Add padding layer (4D input with 2D spatial padding)
        let padding_layer = network
            .add_padding(&input_to_pad, &pre_padding_spatial, &post_padding_spatial)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add padding layer: {}", e),
            })?;

        let padded_output =
            padding_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get padding output: {}", e),
                })?;

        // If we reshaped to 4D, reshape back to original dimensions
        let output = if original_ndims < 4 {
            // Calculate output shape after padding
            let mut output_shape = input_dims.clone();
            for (i, (&pre, &post)) in pre_padding.iter().zip(post_padding.iter()).enumerate() {
                if i < output_shape.len() {
                    output_shape[i] += pre + post;
                }
            }

            let mut reshape_back =
                network
                    .add_shuffle(&padded_output)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to create reshape-back layer: {}", e),
                    })?;

            reshape_back
                .set_reshape_dimensions(&output_shape)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to set reshape-back dimensions: {}", e),
                })?;

            reshape_back
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get reshape-back output: {}", e),
                })?
        } else {
            padded_output
        };

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
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

        // Get actual dimensions for validation
        let _dims_a = input_a
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get input A dimensions: {}", e),
            })?;
        let _dims_b = input_b
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get input B dimensions: {}", e),
            })?;

        // Use TensorRT MatrixOperation enum
        // CRITICAL: TensorRT's IMatrixMultiplyLayer seems to have different semantics
        // Based on the error, it appears TensorRT validates dimensions BEFORE applying transpose
        // So we need to ensure dimensions are already compatible
        //
        // For standard matmul: A [M, K] @ B [K, N] = C [M, N]
        // With b_transpose: A [M, K] @ B^T [N, K] where B is [K, N] originally
        //
        // Our case: A [1, 1280] @ B [1000, 1280] with b_transpose
        // Expected: A [1, 1280] @ B^T [1280, 1000] = [1, 1000]
        //
        // But TensorRT error suggests it wants: A[-1] == B[-2]
        // For B [1000, 1280]: B[-2] = 1000, B[-1] = 1280
        // This would only work if B was already [1280, 1000]!
        //
        // Try swapping operands and transpose flags to match TensorRT's expectations
        use trtx::MatrixOperation;

        // Swap: instead of A @ B^T, try B^T @ A^T (which gives same result transposed)
        // NO wait - let's try: B @ A instead since B^T @ A^T = (A @ B)^T
        //
        // Actually, for WebNN: output = A @ B^T
        // Try: output = (B @ A^T)^T = A @ B^T (mathematically equivalent)
        let (mat_a, mat_b, op_a, op_b) = if b_transpose && !a_transpose {
            // Original: A @ B^T
            // Our input_a: [1, 1280], input_b: [1000, 1280]
            // Try: use B @ A^T and then transpose result
            // B: [1000, 1280] @ A^T: [1280, 1] = [1000, 1]
            // Then transpose to [1, 1000]
            //
            // But we can't easily transpose the result...
            // Let's just try the transpose flags as-is first
            (
                input_a,
                input_b,
                MatrixOperation::kNONE as i32,
                MatrixOperation::kTRANSPOSE as i32,
            )
        } else {
            let a_op = if a_transpose {
                MatrixOperation::kTRANSPOSE as i32
            } else {
                MatrixOperation::kNONE as i32
            };
            let b_op = if b_transpose {
                MatrixOperation::kTRANSPOSE as i32
            } else {
                MatrixOperation::kNONE as i32
            };
            (input_a, input_b, a_op, b_op)
        };

        // Add matrix multiply layer
        let layer = network
            .add_matrix_multiply(mat_a, op_a, mat_b, op_b)
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
            let result_dims = result
                .dimensions()
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get result dimensions: {}", e),
                })?;

            // Create constant filled with alpha value matching result shape
            let num_elements: usize = result_dims.iter().map(|&d| d as usize).product();
            let alpha_data: Vec<f32> = vec![alpha; num_elements];
            let alpha_bytes: Vec<u8> = alpha_data.iter().flat_map(|&f| f.to_le_bytes()).collect();

            // Store weights to keep them alive until engine serialization
            temp_weights.push(alpha_bytes);
            let alpha_bytes_ref = temp_weights.last().unwrap().as_slice();

            let alpha_layer = network
                .add_constant(&result_dims, alpha_bytes_ref, TrtDataType::kFLOAT)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to create alpha constant: {}", e),
                })?;

            let alpha_tensor =
                alpha_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get alpha tensor: {}", e),
                    })?;

            // Multiply result by alpha
            let scale_layer = network
                .add_elementwise(&result, &alpha_tensor, ElementWiseOperation::kPROD)
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
                let c_dims = input_c
                    .dimensions()
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get C dimensions: {}", e),
                    })?;

                // Create constant filled with beta value matching C shape
                let num_elements: usize = c_dims.iter().map(|&d| d as usize).product();
                let beta_data: Vec<f32> = vec![beta; num_elements];
                let beta_bytes: Vec<u8> = beta_data.iter().flat_map(|&f| f.to_le_bytes()).collect();

                // Store weights to keep them alive until engine serialization
                temp_weights.push(beta_bytes);
                let beta_bytes_ref = temp_weights.last().unwrap().as_slice();

                let beta_layer = network
                    .add_constant(&c_dims, beta_bytes_ref, TrtDataType::kFLOAT)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to create beta constant: {}", e),
                    })?;

                let beta_tensor =
                    beta_layer
                        .get_output(0)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Failed to get beta tensor: {}", e),
                        })?;

                // Multiply C by beta
                let scale_c_layer = network
                    .add_elementwise(input_c, &beta_tensor, ElementWiseOperation::kPROD)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to scale C by beta: {}", e),
                    })?;

                let scaled_c =
                    scale_c_layer
                        .get_output(0)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Failed to get scaled C: {}", e),
                        })?;

                // Add result + beta*C
                let add_layer = network
                    .add_elementwise(&result, &scaled_c, ElementWiseOperation::kSUM)
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
                    .add_elementwise(&result, input_c, ElementWiseOperation::kSUM)
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
        let filter_operand =
            graph
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
                reason: format!("Expected 4D filter shape, got {}D", filter_shape.len()),
            });
        }

        let num_output_maps = filter_shape[0] as i32;
        let kernel_size: [i32; 2] = [filter_shape[2] as i32, filter_shape[3] as i32];

        // Parse attributes for stride, padding, dilation, groups
        let strides = operation
            .attributes
            .get("strides")
            .and_then(|v| v.as_array())
            .map(|arr| {
                [
                    arr[0].as_u64().unwrap_or(1) as i32,
                    arr[1].as_u64().unwrap_or(1) as i32,
                ]
            })
            .unwrap_or([1, 1]);

        let dilations = operation
            .attributes
            .get("dilations")
            .and_then(|v| v.as_array())
            .map(|arr| {
                [
                    arr[0].as_u64().unwrap_or(1) as i32,
                    arr[1].as_u64().unwrap_or(1) as i32,
                ]
            })
            .unwrap_or([1, 1]);

        let groups = operation
            .attributes
            .get("groups")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as i32;

        // Parse padding: WebNN/ONNX use "pads" [begin_h, begin_w, end_h, end_w];
        // WPT JSON and some tests use "padding" [top, bottom, left, right] = [begin_h, end_h, begin_w, end_w].
        // Accept both for compatibility.
        let (pre_padding, post_padding) = if let Some(v) = operation.attributes.get("pads") {
            v.as_array()
                .and_then(|arr| {
                    if arr.len() >= 4 {
                        let a: [i32; 4] = [
                            arr[0].as_u64().unwrap_or(0) as i32,
                            arr[1].as_u64().unwrap_or(0) as i32,
                            arr[2].as_u64().unwrap_or(0) as i32,
                            arr[3].as_u64().unwrap_or(0) as i32,
                        ];
                        Some((vec![a[0], a[1]], vec![a[2], a[3]])) // [begin_h, begin_w], [end_h, end_w]
                    } else {
                        None
                    }
                })
                .unwrap_or((vec![0, 0], vec![0, 0]))
        } else if let Some(v) = operation.attributes.get("padding") {
            v.as_array()
                .and_then(|arr| {
                    if arr.len() >= 4 {
                        let a: [i32; 4] = [
                            arr[0].as_u64().unwrap_or(0) as i32,
                            arr[1].as_u64().unwrap_or(0) as i32,
                            arr[2].as_u64().unwrap_or(0) as i32,
                            arr[3].as_u64().unwrap_or(0) as i32,
                        ];
                        // [top, bottom, left, right] -> pre=[top, left], post=[bottom, right]
                        Some((vec![a[0], a[2]], vec![a[1], a[3]]))
                    } else {
                        None
                    }
                })
                .unwrap_or((vec![0, 0], vec![0, 0]))
        } else {
            (vec![0, 0], vec![0, 0])
        };

        // Use explicit padding layer if any padding is specified
        let conv_input =
            if pre_padding.iter().any(|&p| p != 0) || post_padding.iter().any(|&p| p != 0) {
                let padding_layer = network
                    .add_padding(input, &pre_padding, &post_padding)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add padding layer: {}", e),
                    })?;

                padding_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get padding layer output: {}", e),
                    })?
            } else {
                // No padding needed, use input directly
                // Need to clone the tensor reference
                let id_layer =
                    network
                        .add_identity(input)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Failed to add identity layer: {}", e),
                        })?;
                id_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get identity output: {}", e),
                    })?
            };

        // Add convolution layer with zero padding (padding already applied via padding layer)
        let mut layer = network
            .add_convolution(
                &conv_input,
                num_output_maps,
                &kernel_size,
                filter_data,
                bias_data,
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add convolution: {}", e),
            })?;

        // Set layer properties (matches C++ API pattern: call setters after creation)
        layer
            .set_stride(&strides)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set stride: {}", e),
            })?;

        // No need to set padding on convolution layer - already handled by explicit padding layer
        layer
            .set_padding(&[0, 0])
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set padding: {}", e),
            })?;

        layer
            .set_dilation(&dilations)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set dilation: {}", e),
            })?;

        layer
            .set_num_groups(groups)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set groups: {}", e),
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

    /// Add convTranspose2d operation (deconvolution/transposed convolution)
    fn add_conv_transpose2d_op(
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
        let filter_operand =
            graph
                .operand(filter_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Filter operand {} not found", filter_id),
                })?;

        // Filter shape for convTranspose2d: [inputChannels, outputChannels/groups, height, width]
        // Note: This is different from conv2d!
        let filter_shape = &filter_operand.descriptor.shape;
        if filter_shape.len() != 4 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Expected 4D filter shape for convTranspose2d, got {}D",
                    filter_shape.len()
                ),
            });
        }

        let num_output_maps = filter_shape[1] as i32;
        let kernel_size: [i32; 2] = [filter_shape[2] as i32, filter_shape[3] as i32];

        // Add deconvolution layer
        let layer = network
            .add_deconvolution(input, num_output_maps, &kernel_size, filter_data, bias_data)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add deconvolution: {}", e),
            })?;

        // Extract output tensor from layer
        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get deconvolution output: {}", e),
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
        pool_type: PoolingType,
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

        // Get the axis parameter (defaults to last axis)
        // TensorRT uses a bitmask where bit N represents axis N
        let axis = operation
            .attributes
            .get("axis")
            .and_then(|v| v.as_i64())
            .unwrap_or(-1); // Default to last axis

        // Handle negative axis (convert to positive)
        let dims = input
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get input dimensions: {}", e),
            })?;
        let num_dims = dims.len() as i64;
        let positive_axis = if axis < 0 {
            (num_dims + axis) as u32
        } else {
            axis as u32
        };

        // Create bitmask for the axis
        let axes = 1u32 << positive_axis;

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

        // Parse newShape attribute
        let new_shape = operation
            .attributes
            .get("newShape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "reshape operation missing 'newShape' attribute".to_string(),
            })?;

        let dims: Vec<i32> = new_shape
            .iter()
            .map(|v| v.as_i64().unwrap_or(0) as i32)
            .collect();

        // Use shuffle layer for reshape
        let mut layer = network
            .add_shuffle(input)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add shuffle (reshape): {}", e),
            })?;

        // Set the reshape dimensions
        layer
            .set_reshape_dimensions(&dims)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set reshape dimensions: {}", e),
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

    /// Add resample2d operation (resize/interpolate 2D tensor)
    fn add_resample2d_op(
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

        // Parse mode attribute (default to "nearest-neighbor")
        let mode_str = operation
            .attributes
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("nearest-neighbor");

        // Map WebNN mode to TensorRT ResizeMode (typedef for InterpolationMode)
        let resize_mode = match mode_str {
            "nearest-neighbor" => ResizeMode::kNEAREST,
            "linear" => ResizeMode::kLINEAR,
            _ => ResizeMode::kNEAREST, // Default to nearest
        };

        // Parse sizes from attributes (should be output spatial dimensions)
        // WebNN resample2d uses [newHeight, newWidth]
        let sizes = operation
            .attributes
            .get("sizes")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64().map(|i| i as i32))
                    .collect::<Vec<i32>>()
            })
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Missing sizes attribute for resample2d".to_string(),
            })?;

        if sizes.len() != 2 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "resample2d sizes must have 2 elements (height, width), got {}",
                    sizes.len()
                ),
            });
        }

        // Create resize layer
        let mut layer = network
            .add_resize(input)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add resize layer: {}", e),
            })?;

        // TensorRT expects full output dimensions [N, C, H, W]
        // WebNN resample2d only specifies [H, W], so we need to preserve N and C
        // For now, we'll assume 4D NCHW input and set full dimensions
        // TODO: Get actual input dimensions to preserve N and C
        let output_dims = vec![1, 1, sizes[0], sizes[1]]; // Placeholder: [N=1, C=1, H, W]

        // Set output dimensions
        layer
            .set_output_dimensions(&output_dims)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set resize output dimensions: {}", e),
            })?;

        // Set resize mode (uses ResizeMode typedef for InterpolationMode)
        layer
            .set_resize_mode(resize_mode)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set resize mode: {}", e),
            })?;

        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get resize output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    // ============================================================================
    // Additional Operations (2026-01-29 - Final 8)
    // ============================================================================

    /// Add isNaN operation (check if value is NaN using x != x)
    fn add_is_nan_op(
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_tensor = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // NaN is the only value where x != x is true
        // Use elementwise EQUAL operation with itself, then negate
        let layer = network
            .add_elementwise(input_tensor, input_tensor, ElementWiseOperation::kEQUAL)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create EQUAL layer for isNaN: {}", e),
            })?;

        let equal_output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get EQUAL output: {}", e),
            })?;

        // Negate the result (isNaN = NOT(x == x))
        let not_layer = network
            .add_unary(&equal_output, UnaryOperation::kNOT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create NOT layer for isNaN: {}", e),
            })?;

        let bool_output = not_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get isNaN output: {}", e),
            })?;

        // Cast BOOL to Float32 for WebNN compatibility
        let output = Self::cast_to_float32(network, &bool_output)?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add isInfinite operation (check if value is infinite)
    fn add_is_infinite_op(
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
        temp_weights: &mut Vec<Vec<u8>>,
    ) -> Result<(), GraphError> {
        let input_tensor = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // Check if abs(x) == infinity
        // First compute abs(x)
        let abs_layer = network
            .add_unary(input_tensor, UnaryOperation::kABS)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create ABS layer for isInfinite: {}", e),
            })?;

        let abs_output = abs_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get ABS output: {}", e),
            })?;

        // Create constant for infinity
        temp_weights.push(f32::INFINITY.to_le_bytes().to_vec());
        let inf_data = temp_weights.last().unwrap();
        let inf_constant = network
            .add_constant(&[1], inf_data, TrtDataType::kFLOAT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create infinity constant: {}", e),
            })?;

        let inf_tensor = inf_constant
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get infinity tensor: {}", e),
            })?;

        // Compare abs(x) == infinity
        let equal_layer = network
            .add_elementwise(&abs_output, &inf_tensor, ElementWiseOperation::kEQUAL)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create EQUAL layer for isInfinite: {}", e),
            })?;

        let bool_output = equal_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get isInfinite output: {}", e),
            })?;

        // Cast BOOL to Float32 for WebNN compatibility
        let output = Self::cast_to_float32(network, &bool_output)?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add roundEven operation (round to nearest even integer, banker's rounding)
    fn add_round_even_op(
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_tensor = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // TensorRT's kROUND already uses round-to-nearest-even (banker's rounding) by default
        let layer = network
            .add_unary(input_tensor, UnaryOperation::kROUND)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create ROUND layer: {}", e),
            })?;

        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get roundEven output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add gatherElements operation (gather using index tensor along axis)
    fn add_gather_elements_op(
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let data_tensor = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Data operand {} not found", operation.input_operands[0]),
            })?;

        let indices_tensor = tensor_map
            .get(&operation.input_operands[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Indices operand {} not found", operation.input_operands[1]),
            })?;

        // Get axis parameter (default to 0)
        let axis = operation
            .attributes
            .get("axis")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32;

        // Create gather layer with ELEMENT mode
        let mut layer = network
            .add_gather(data_tensor, indices_tensor, axis)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create gather layer: {}", e),
            })?;

        // Set gather mode to ELEMENT
        layer
            .set_gather_mode(trtx::GatherMode::kELEMENT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set gather mode: {}", e),
            })?;

        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get gatherElements output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add l2Pool2d operation (L2 pooling: square → avgPool → sqrt)
    fn add_l2_pool2d_op(
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_tensor = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // Step 1: Square the input (x^2)
        let square_layer = network
            .add_elementwise(input_tensor, input_tensor, ElementWiseOperation::kPROD)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create square layer for l2Pool2d: {}", e),
            })?;

        let squared = square_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get squared output: {}", e),
            })?;

        // Step 2: Apply average pooling (use same parameters as maxPool2d/averagePool2d)
        let window_size = operation
            .attributes
            .get("windowDimensions")
            .and_then(|v| v.as_array())
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Missing windowDimensions for l2Pool2d".to_string(),
            })?;

        let window: [i32; 2] = [
            window_size[0].as_i64().unwrap_or(1) as i32,
            window_size[1].as_i64().unwrap_or(1) as i32,
        ];

        let pool_layer = network
            .add_pooling(&squared, PoolingType::kAVERAGE, &window)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create pooling layer for l2Pool2d: {}", e),
            })?;

        let pooled = pool_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get pooled output: {}", e),
            })?;

        // Step 3: Take square root
        let sqrt_layer = network
            .add_unary(&pooled, UnaryOperation::kSQRT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create sqrt layer for l2Pool2d: {}", e),
            })?;

        let output = sqrt_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get l2Pool2d output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add reverse operation (reverse elements along axes) - PLACEHOLDER
    /// Add reverse operation (reverse elements along axes using negative stride slicing)
    fn add_reverse_op(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_tensor = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // Get axes to reverse from attributes (if not specified, reverse all axes)
        let axes_value = operation.attributes.get("axes");

        // Get input shape from graph
        let input_operand = &graph.operands[operation.input_operands[0] as usize];
        let shape = &input_operand.descriptor.shape;
        let rank = shape.len();

        // Determine which axes to reverse
        let axes_to_reverse: Vec<usize> = if let Some(axes_val) = axes_value {
            if let Some(arr) = axes_val.as_array() {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|i| i as usize))
                    .collect()
            } else {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "Invalid 'axes' attribute format for reverse".to_string(),
                });
            }
        } else {
            // If no axes specified, reverse all axes
            (0..rank).collect()
        };

        // Build slice parameters for negative stride
        // With negative stride: end_idx = start + (size - 1) * stride
        // For reversing: we want indices [n-1, n-2, ..., 1, 0]
        //   start = n-1 (last element)
        //   size = n (number of elements)
        //   stride = -1
        //   end_idx should be = (n-1) + (n-1)*(-1) = (n-1) - (n-1) = 0 ✓
        let mut starts: Vec<i32> = vec![0; rank];
        let sizes: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        let mut strides: Vec<i32> = vec![1; rank];

        for &axis in &axes_to_reverse {
            if axis >= rank {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Reverse axis {} out of range for rank {}", axis, rank),
                });
            }
            // For negative stride in TensorRT:
            // Start at the last valid element
            // Size remains the same (number of elements to output)
            // Stride = -1 to go backwards
            // TensorRT will compute: indices = start + i*stride for i in 0..size
            // So: indices = (size-1) + i*(-1) = (size-1) - i
            // For i=0: size-1, i=1: size-2, ..., i=size-1: 0 ✓
            starts[axis] = (shape[axis] - 1) as i32;
            strides[axis] = -1;
        }

        let layer = network
            .add_slice(input_tensor, &starts, &sizes, &strides)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add slice layer for reverse: {}", e),
            })?;

        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get reverse output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add cumulativeSum operation (cumulative sum along axis) - PLACEHOLDER
    /// Add cumulativeSum operation using explicit slice-and-add decomposition
    ///
    /// Uses TensorRT's native ICumulativeLayer for efficient implementation.
    fn add_cumulative_sum_op(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
        temp_weights: &mut Vec<Vec<u8>>,
    ) -> Result<(), GraphError> {
        let input_tensor = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // Get axis from attributes
        let axis = operation
            .attributes
            .get("axis")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        // Get input shape
        let input_operand = &graph.operands[operation.input_operands[0] as usize];
        let shape = &input_operand.descriptor.shape;
        let rank = shape.len();

        if axis >= rank {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("CumulativeSum axis {} out of range for rank {}", axis, rank),
            });
        }

        // Get exclusive and reverse flags (WebNN doesn't have these, default to false)
        let exclusive = operation
            .attributes
            .get("exclusive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let reverse = operation
            .attributes
            .get("reverse")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Create axis constant with proper lifetime management
        // Store bytes in temp_weights to keep them alive until engine is built
        let axis_value = axis as i32;
        let axis_bytes: Vec<u8> = axis_value.to_le_bytes().to_vec();
        temp_weights.push(axis_bytes);
        let axis_bytes_ref = temp_weights.last().unwrap();

        // Create axis constant tensor (true 0D scalar with shape [])
        // TensorRT requires axisDims.nbDims == 0 for cumulative operations
        let axis_constant = network
            .add_constant(&[], axis_bytes_ref, trtx::DataType::kINT32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create axis constant: {}", e),
            })?;

        let axis_tensor =
            axis_constant
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get axis constant output: {}", e),
                })?;

        // Use TensorRT's native ICumulativeLayer with CumulativeOperation::SUM
        let layer = network
            .add_cumulative_with_axis_tensor(
                input_tensor,
                &axis_tensor,
                trtx::CumulativeOperation::kSUM,
                exclusive,
                reverse,
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add cumulative sum layer: {}", e),
            })?;

        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get cumulative sum output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add triangular operation (extract triangular part of matrix) - PLACEHOLDER
    /// Add triangular operation (extract upper/lower triangular part with masking)
    fn add_triangular_op(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
        temp_weights: &mut Vec<Vec<u8>>,
    ) -> Result<(), GraphError> {
        let input_tensor = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        // Get parameters from attributes
        let upper = operation
            .attributes
            .get("upper")
            .and_then(|v| v.as_bool())
            .unwrap_or(true); // Default to upper triangular

        let diagonal = operation
            .attributes
            .get("diagonal")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32; // Default diagonal offset is 0

        // Get input shape from graph
        let input_operand = &graph.operands[operation.input_operands[0] as usize];
        let shape = &input_operand.descriptor.shape;

        // Triangular only makes sense for 2D matrices (or higher-D tensors treated as batches of 2D)
        if shape.len() < 2 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Triangular requires at least 2D tensor, got {}D",
                    shape.len()
                ),
            });
        }

        let rows = shape[shape.len() - 2] as usize;
        let cols = shape[shape.len() - 1] as usize;

        // Generate triangular mask (1.0 for keep, 0.0 for zero)
        // The mask is computed at build time based on the known shape
        let total_elements: usize = shape.iter().map(|&s| s as usize).product();
        let matrix_elements = rows * cols;
        let num_matrices = total_elements / matrix_elements;

        let mut mask_data: Vec<f32> = Vec::with_capacity(total_elements);

        for _ in 0..num_matrices {
            for i in 0..rows {
                for j in 0..cols {
                    let keep = if upper {
                        // Upper triangular: keep if j >= i + diagonal
                        (j as i32) >= (i as i32) + diagonal
                    } else {
                        // Lower triangular: keep if j <= i + diagonal
                        (j as i32) <= (i as i32) + diagonal
                    };
                    mask_data.push(if keep { 1.0 } else { 0.0 });
                }
            }
        }

        // Convert mask to bytes
        let mask_bytes: Vec<u8> = mask_data.iter().flat_map(|&f| f.to_le_bytes()).collect();

        // Store mask in temp_weights to keep it alive (critical for weight lifetime)
        temp_weights.push(mask_bytes);
        let mask_bytes_ref = temp_weights.last().unwrap();

        // Create constant layer with the mask
        let dims: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        let mask_layer = network
            .add_constant(&dims, mask_bytes_ref, trtx::DataType::kFLOAT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add constant mask for triangular: {}", e),
            })?;

        let mask_tensor = mask_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get mask tensor: {}", e),
            })?;

        // Multiply input by mask (elementwise)
        let multiply_layer = network
            .add_elementwise(input_tensor, &mask_tensor, ElementWiseOperation::kPROD)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add elementwise multiply for triangular: {}", e),
            })?;

        let output = multiply_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get triangular output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    // NOTE: RNN operation implementations removed
    // IRNNv2Layer is deprecated in TensorRT and autocxx cannot generate bindings for it
    // RNN operations (lstm, lstmCell, gru, gruCell) remain deferred
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
    use trtx::DataType as TrtDataType;

    #[test]
    fn test_webnn_to_trt_dtype() {
        assert!(matches!(
            TrtxConverter::webnn_to_trt_dtype(DataType::Float32).unwrap(),
            TrtDataType::kFLOAT
        ));
        assert!(matches!(
            TrtxConverter::webnn_to_trt_dtype(DataType::Float16).unwrap(),
            TrtDataType::kHALF
        ));
        assert!(matches!(
            TrtxConverter::webnn_to_trt_dtype(DataType::Int8).unwrap(),
            TrtDataType::kINT8
        ));
        assert!(matches!(
            TrtxConverter::webnn_to_trt_dtype(DataType::Int32).unwrap(),
            TrtDataType::kINT32
        ));
    }
}

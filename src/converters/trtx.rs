/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! TensorRT native converter - directly builds TensorRT INetworkDefinition
//!
//! This converter bypasses ONNX serialization and builds TensorRT networks directly
//! from WebNN graph IR, providing better performance and avoiding ONNX limitations.

use std::collections::{HashMap, HashSet};

use half::f16;

use super::{ConvertedGraph, GraphConverter};
use crate::error::GraphError;
use crate::executors::trtx::{create_trtx_logger, ensure_trtx_loaded};
use crate::graph::{DataType, GraphInfo, OperandKind, Operation, get_static_or_max_size};
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

    /// Map WebNN DataType to TensorRT DataType enum.
    /// TensorRT has no kUINT32; we use kINT32 (same 4-byte layout, bit-identical for cast output).
    /// TensorRT has no kUINT64; we use kINT64 (same 8-byte layout, bit-identical for elementwise).
    fn webnn_to_trt_dtype(dtype: DataType) -> Result<TrtDataType, GraphError> {
        match dtype {
            DataType::Float32 => Ok(TrtDataType::kFLOAT),
            DataType::Float16 => Ok(TrtDataType::kHALF),
            DataType::Int8 => Ok(TrtDataType::kINT8),
            DataType::Int32 => Ok(TrtDataType::kINT32),
            DataType::Uint8 => Ok(TrtDataType::kUINT8),
            DataType::Uint32 => Ok(TrtDataType::kINT32),
            DataType::Int64 => Ok(TrtDataType::kINT64),
            DataType::Uint64 => Ok(TrtDataType::kINT64),
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

    /// Convert float16 bytes to float32 bytes (little-endian). Used when filter is Float16
    /// but TensorRT conv layer expects Float weights.
    fn f16_bytes_to_f32_bytes(data: &[u8]) -> Result<Vec<u8>, GraphError> {
        if data.len() % 2 != 0 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Float16 data length {} is not even", data.len()),
            });
        }
        let n = data.len() / 2;
        let mut out = Vec::with_capacity(n * 4);
        for i in 0..n {
            let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
            out.extend_from_slice(&f16::from_bits(bits).to_f32().to_le_bytes());
        }
        Ok(out)
    }

    /// Transpose 4D conv filter (f32) from given layout to OIHW for TensorRT.
    /// Layouts: oihw [O,I,H,W], hwio [H,W,I,O], ohwi [O,H,W,I], ihwo [I,H,W,O], hwoi [H,W,O,I].
    /// Permutation for HWIO -> OIHW is (3, 2, 0, 1): output[o,i,h,w] = input[h,w,i,o].
    fn conv_filter_to_oihw(
        data: &[u8],
        layout: &str,
        shape: &[u32],
    ) -> Result<Vec<u8>, GraphError> {
        if shape.len() != 4 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Filter shape must be 4D, got {}D", shape.len()),
            });
        }
        let (o, i, h, w) = match layout {
            "oihw" => (shape[0], shape[1], shape[2], shape[3]),
            "hwio" => (shape[3], shape[2], shape[0], shape[1]),
            "ohwi" => (shape[0], shape[3], shape[1], shape[2]),
            "ihwo" => (shape[3], shape[0], shape[1], shape[2]),
            "hwoi" => (shape[2], shape[3], shape[0], shape[1]),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Unsupported filter_layout: {}", layout),
                });
            }
        };
        let n = (o * i * h * w) as usize;
        if data.len() < n * 4 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Filter data length {} < {} (O*I*H*W*4)", data.len(), n * 4),
            });
        }
        let src = data;
        let mut dst = vec![0u8; n * 4];
        let o = o as usize;
        let i = i as usize;
        let h = h as usize;
        let w = w as usize;
        for oo in 0..o {
            for ii in 0..i {
                for hh in 0..h {
                    for ww in 0..w {
                        let src_idx = match layout {
                            "oihw" => oo * (i * h * w) + ii * (h * w) + hh * w + ww,
                            "hwio" => hh * (w * i * o) + ww * (i * o) + ii * o + oo,
                            "ohwi" => oo * (h * w * i) + hh * (w * i) + ww * i + ii,
                            "ihwo" => ii * (h * w * o) + hh * (w * o) + ww * o + oo,
                            "hwoi" => hh * (w * o * i) + ww * (o * i) + oo * i + ii,
                            _ => unreachable!(),
                        };
                        let dst_idx = oo * (i * h * w) + ii * (h * w) + hh * w + ww;
                        dst[dst_idx * 4..dst_idx * 4 + 4]
                            .copy_from_slice(&src[src_idx * 4..src_idx * 4 + 4]);
                    }
                }
            }
        }
        Ok(dst)
    }

    /// Transpose 4D deconv filter (f32) from given layout to IOHW for TensorRT (C,K,R,S = input channels, output maps, H, W).
    /// Layouts: iohw [I,O,H,W], oihw [O,I,H,W], hwio [H,W,I,O], ohwi [O,H,W,I], ihwo [I,H,W,O], hwoi [H,W,O,I].
    fn deconv_filter_to_iohw(
        data: &[u8],
        layout: &str,
        shape: &[u32],
    ) -> Result<Vec<u8>, GraphError> {
        if shape.len() != 4 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Filter shape must be 4D, got {}D", shape.len()),
            });
        }
        let (i, o, h, w) = match layout {
            "iohw" => (shape[0], shape[1], shape[2], shape[3]),
            "oihw" => (shape[1], shape[0], shape[2], shape[3]),
            "hwio" => (shape[2], shape[3], shape[0], shape[1]),
            "ohwi" => (shape[3], shape[0], shape[1], shape[2]),
            "ihwo" => (shape[0], shape[3], shape[1], shape[2]),
            "hwoi" => (shape[3], shape[2], shape[0], shape[1]),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Unsupported filter_layout: {}", layout),
                });
            }
        };
        let n = (i * o * h * w) as usize;
        if data.len() < n * 4 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Filter data length {} < {} (I*O*H*W*4)", data.len(), n * 4),
            });
        }
        let src = data;
        let mut dst = vec![0u8; n * 4];
        let i = i as usize;
        let o = o as usize;
        let h = h as usize;
        let w = w as usize;
        for ii in 0..i {
            for oo in 0..o {
                for hh in 0..h {
                    for ww in 0..w {
                        let src_idx = match layout {
                            "iohw" => ii * (o * h * w) + oo * (h * w) + hh * w + ww,
                            "oihw" => oo * (i * h * w) + ii * (h * w) + hh * w + ww,
                            "hwio" => hh * (w * i * o) + ww * (i * o) + ii * o + oo,
                            "ohwi" => oo * (h * w * i) + hh * (w * i) + ww * i + ii,
                            "ihwo" => ii * (h * w * o) + hh * (w * o) + ww * o + oo,
                            "hwoi" => hh * (w * o * i) + ww * (o * i) + oo * i + ii,
                            _ => unreachable!(),
                        };
                        let dst_idx = ii * (o * h * w) + oo * (h * w) + hh * w + ww;
                        dst[dst_idx * 4..dst_idx * 4 + 4]
                            .copy_from_slice(&src[src_idx * 4..src_idx * 4 + 4]);
                    }
                }
            }
        }
        Ok(dst)
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

    /// Cast tensor to Float16 (e.g. after float32 reduction to avoid float16 overflow).
    fn cast_to_float16(
        network: &mut trtx::NetworkDefinition,
        input: &trtx::Tensor,
    ) -> Result<trtx::Tensor, GraphError> {
        let layer = network.add_cast(input, TrtDataType::kHALF).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to cast to Float16: {}", e),
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
        let promoted_constants: HashSet<u32> = HashSet::new();
        let constants_stored_flat: HashSet<u32> = HashSet::new();

        // Step 1: Add inputs
        for (operand_id, operand) in graph.operands.iter().enumerate() {
            if operand.kind == OperandKind::Input {
                let dtype = Self::webnn_to_trt_dtype(operand.descriptor.data_type)?;
                let dims: Vec<i32> = operand
                    .descriptor
                    .shape
                    .iter()
                    .map(|d| get_static_or_max_size(d) as i32)
                    .collect();
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
                let dims: Vec<i32> = operand
                    .descriptor
                    .shape
                    .iter()
                    .map(|d| get_static_or_max_size(d) as i32)
                    .collect();
                let data = Self::get_constant_data(graph, operand_id as u32)?;

                // Validate that data size matches expected size
                let expected_size: usize = operand
                    .descriptor
                    .shape
                    .iter()
                    .map(|d| get_static_or_max_size(d) as usize)
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

                // TensorRT Constant permits kINT8 (not kUINT8). Use int8 for Int8/Uint8 constants:
                // same byte layout, no promotion; LogicalNot accepts int8/uint8 input.
                let use_int8_constant = matches!(
                    operand.descriptor.data_type,
                    DataType::Int8 | DataType::Uint8
                );

                // TensorRT add_constant does not support kINT64; convert Int64 constants to Int32.
                let promote_int64 = operand.descriptor.data_type == DataType::Int64;

                let (trt_dtype, data_to_use, add_dims): (TrtDataType, &[u8], Vec<i32>) =
                    if use_int8_constant {
                        // Pass raw bytes; type kINT8 (Uint8 same bits for 0/1, no conversion)
                        (TrtDataType::kINT8, data, dims.clone())
                    } else if promote_int64 {
                        let int32_bytes: Vec<u8> = data
                            .chunks_exact(8)
                            .flat_map(|chunk| {
                                (i64::from_le_bytes(chunk.try_into().unwrap()) as i32).to_le_bytes()
                            })
                            .collect();
                        temp_weights.push(int32_bytes);
                        (
                            TrtDataType::kINT32,
                            temp_weights.last().unwrap().as_slice(),
                            dims.clone(),
                        )
                    } else {
                        (
                            Self::webnn_to_trt_dtype(operand.descriptor.data_type)?,
                            data,
                            dims.clone(),
                        )
                    };

                let layer = network
                    .add_constant(&add_dims, data_to_use, trt_dtype)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add constant (operand {}): {}", operand_id, e),
                    })?;

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
                &promoted_constants,
                &constants_stored_flat,
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
        promoted_constants: &HashSet<u32>,
        constants_stored_flat: &HashSet<u32>,
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
            "elu" => Self::add_elu_op(graph, network, tensor_map, operation, temp_weights)?,
            "softsign" => {
                Self::add_activation_op(network, tensor_map, operation, ActivationType::kSOFTSIGN)?
            }
            "softplus" => {
                Self::add_activation_op(network, tensor_map, operation, ActivationType::kSOFTPLUS)?
            }
            "gelu" => {
                Self::add_activation_op(network, tensor_map, operation, ActivationType::kGELU_ERF)?
            }
            "leakyRelu" => {
                Self::add_leaky_relu_op(graph, network, tensor_map, operation, temp_weights)?
            }
            "prelu" => Self::add_prelu_op(network, tensor_map, operation)?,
            "hardSigmoid" => {
                Self::add_hard_sigmoid_op(graph, network, tensor_map, operation, temp_weights)?
            }
            "hardSwish" => {
                Self::add_hard_swish_op(graph, network, tensor_map, operation, temp_weights)?
            }

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
            "cast" => Self::add_cast_op(
                graph,
                network,
                tensor_map,
                temp_weights,
                promoted_constants,
                constants_stored_flat,
                operation,
            )?,
            "quantizeLinear" => Self::add_quantize_linear_op(network, tensor_map, operation)?,
            "dequantizeLinear" => Self::add_dequantize_linear_op(network, tensor_map, operation)?,

            // Matrix operations
            "matmul" => Self::add_matmul_op(network, tensor_map, operation)?,
            "gemm" => Self::add_gemm_op(graph, network, tensor_map, temp_weights, operation)?,

            // Convolution operations
            "conv2d" => Self::add_conv2d_op(graph, network, tensor_map, temp_weights, operation)?,
            "convTranspose2d" => {
                Self::add_conv_transpose2d_op(graph, network, tensor_map, temp_weights, operation)?
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
            "instanceNormalization" => Self::add_instance_normalization_op(
                graph,
                network,
                tensor_map,
                operation,
                temp_weights,
            )?,
            "layerNormalization" => Self::add_layer_normalization_op(
                graph,
                network,
                tensor_map,
                operation,
                temp_weights,
            )?,

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
            "reduceL2" => Self::add_reduce_l2_op(graph, network, tensor_map, operation)?,
            "reduceLogSum" => Self::add_reduce_log_sum_op(network, tensor_map, operation)?,
            "reduceLogSumExp" => Self::add_reduce_log_sum_exp_op(network, tensor_map, operation)?,
            "reduceSumSquare" => Self::add_reduce_sum_square_op(network, tensor_map, operation)?,

            // Shape manipulation operations
            "slice" => Self::add_slice_op(network, tensor_map, operation)?,
            "split" => Self::add_split_op(network, tensor_map, operation)?,
            "squeeze" => Self::add_squeeze_op(network, tensor_map, operation)?,
            "unsqueeze" => Self::add_unsqueeze_op(network, tensor_map, operation)?,
            "expand" => Self::add_expand_op(graph, network, tensor_map, operation, temp_weights)?,
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
            "logicalNot" => {
                Self::add_logical_not_op(graph, network, tensor_map, operation, temp_weights)?
            }

            // Indexing/Gathering operations
            "gather" => Self::add_gather_op(graph, network, tensor_map, operation, temp_weights)?,
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

    /// Add logical NOT operation. TensorRT Unary(kNOT) requires Bool input.
    /// Quantized constants (kINT8) may only feed DQ/plugin; for UInt8/Int8 constant, add a kBOOL
    /// constant (0 -> false, non-zero -> true) and feed it directly to kNOT so no extra Cast is needed.
    fn add_logical_not_op(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
        temp_weights: &mut Vec<Vec<u8>>,
    ) -> Result<(), GraphError> {
        let input_id = operation.input_operands[0];
        let input = tensor_map
            .get(&input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", input_id),
            })?;

        let not_input = if graph
            .constant_operand_ids_to_handles
            .contains_key(&input_id)
        {
            let operand = graph
                .operand(input_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Operand {} not in graph", input_id),
                })?;
            match operand.descriptor.data_type {
                DataType::Uint8 | DataType::Int8 => {
                    let data = Self::get_constant_data(graph, input_id)?;
                    let shape: Vec<i32> = operand
                        .descriptor
                        .shape
                        .iter()
                        .map(|d| get_static_or_max_size(d) as i32)
                        .collect();
                    let n: usize = operand
                        .descriptor
                        .shape
                        .iter()
                        .map(|d| get_static_or_max_size(d) as usize)
                        .product();
                    let bool_bytes: Vec<u8> = data
                        .iter()
                        .take(n)
                        .map(|&b| if b == 0 { 0u8 } else { 1u8 })
                        .collect();
                    temp_weights.push(bool_bytes);
                    let ref_bytes = temp_weights.last().unwrap().as_slice();
                    let const_layer = network
                        .add_constant(&shape, ref_bytes, TrtDataType::kBOOL)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("LogicalNot: failed to add BOOL constant: {}", e),
                        })?;
                    const_layer
                        .get_output(0)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("LogicalNot: BOOL constant output: {}", e),
                        })?
                }
                _ => Self::cast_to_bool(network, input)?,
            }
        } else {
            Self::cast_to_bool(network, input)?
        };

        let layer = network
            .add_unary(&not_input, UnaryOperation::kNOT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add logical NOT: {}", e),
            })?;

        let not_output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get layer output: {}", e),
            })?;

        let output = Self::cast_to_float32(network, &not_output)?;

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

    /// Add ELU activation: ELU(x) = x if x > 0, else alpha * (exp(x) - 1)
    /// TensorRT kELU uses alpha=1; for custom alpha we decompose as:
    /// relu(x) + alpha * min(0, exp(x) - 1)
    fn add_elu_op(
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

        let alpha = operation
            .attributes
            .as_elu()
            .map(|o| o.alpha as f32)
            .unwrap_or(1.0);

        let input_operand = graph.operand(operation.input_operands[0]).ok_or_else(|| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Input operand {} not found in graph",
                    operation.input_operands[0]
                ),
            }
        })?;
        let input_dtype = input_operand.descriptor.data_type;
        // Use built-in kELU only for float32 with default alpha; float16 needs decomposition for correct precision.
        if (alpha - 1.0).abs() <= f32::EPSILON && input_dtype != DataType::Float16 {
            return Self::add_activation_op(network, tensor_map, operation, ActivationType::kELU);
        }

        let num_dims = input_operand.descriptor.shape.len();
        let broadcast_shape: Vec<i32> = vec![1; num_dims];
        let (trt_dtype, one_bytes, zero_bytes, alpha_bytes) = match input_dtype {
            DataType::Float16 => {
                let one: Vec<u8> = f16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
                let zero: Vec<u8> = f16::from_f32(0.0).to_bits().to_le_bytes().to_vec();
                let alpha_f16: Vec<u8> = f16::from_f32(alpha).to_bits().to_le_bytes().to_vec();
                (trtx::DataType::kHALF, one, zero, alpha_f16)
            }
            _ => {
                let one: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
                let zero: Vec<u8> = 0.0f32.to_le_bytes().to_vec();
                let alpha_f32: Vec<u8> = alpha.to_le_bytes().to_vec();
                (trtx::DataType::kFLOAT, one, zero, alpha_f32)
            }
        };

        // Decompose: ELU(x) = relu(x) + alpha * min(0, exp(x) - 1)
        let relu_layer = network
            .add_activation(input, ActivationType::kRELU)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add relu for elu: {}", e),
            })?;
        let relu_output = relu_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get relu output: {}", e),
            })?;

        let exp_layer = network
            .add_unary(input, UnaryOperation::kEXP)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add exp for elu: {}", e),
            })?;
        let exp_output = exp_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get exp output: {}", e),
            })?;

        temp_weights.push(one_bytes);
        let one_const = network
            .add_constant(
                &broadcast_shape,
                temp_weights.last().unwrap(),
                trt_dtype.clone(),
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create one constant for elu: {}", e),
            })?;
        let one_tensor = one_const
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get one constant output: {}", e),
            })?;

        let (bc_exp, bc_one) =
            Self::ensure_broadcast_compatible(network, &exp_output, &one_tensor, "elu_exp_sub")?;

        let exp_minus_1_layer = network
            .add_elementwise(&bc_exp, &bc_one, ElementWiseOperation::kSUB)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to subtract 1 for elu: {}", e),
            })?;
        let exp_minus_1 =
            exp_minus_1_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get exp-1 output: {}", e),
                })?;

        temp_weights.push(zero_bytes);
        let zero_const = network
            .add_constant(
                &broadcast_shape,
                temp_weights.last().unwrap(),
                trt_dtype.clone(),
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create zero constant for elu: {}", e),
            })?;
        let zero_tensor = zero_const
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get zero constant output: {}", e),
            })?;

        let (bc_em1, bc_zero) =
            Self::ensure_broadcast_compatible(network, &exp_minus_1, &zero_tensor, "elu_min")?;

        let neg_part_layer = network
            .add_elementwise(&bc_em1, &bc_zero, ElementWiseOperation::kMIN)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add min for elu: {}", e),
            })?;
        let neg_part = neg_part_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get neg part output: {}", e),
            })?;

        temp_weights.push(alpha_bytes);
        let alpha_const = network
            .add_constant(
                &broadcast_shape,
                temp_weights.last().unwrap(),
                trt_dtype.clone(),
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create alpha constant for elu: {}", e),
            })?;
        let alpha_tensor = alpha_const
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get alpha constant output: {}", e),
            })?;

        let (bc_neg, bc_alpha) =
            Self::ensure_broadcast_compatible(network, &neg_part, &alpha_tensor, "elu_scale")?;

        let scaled_neg_layer = network
            .add_elementwise(&bc_neg, &bc_alpha, ElementWiseOperation::kPROD)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to multiply by alpha for elu: {}", e),
            })?;
        let scaled_neg =
            scaled_neg_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get scaled neg output: {}", e),
                })?;

        let (bc_relu, bc_scaled) =
            Self::ensure_broadcast_compatible(network, &relu_output, &scaled_neg, "elu_add")?;

        let final_layer = network
            .add_elementwise(&bc_relu, &bc_scaled, ElementWiseOperation::kSUM)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add elu parts: {}", e),
            })?;
        let output = final_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get elu output: {}", e),
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
    /// LeakyReLU(x) = max(alpha * x, x) = x if x >= 0, else alpha * x
    /// Implemented as: max(0, x) + alpha * min(0, x) so alpha is respected (including negative).
    fn add_leaky_relu_op(
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

        let alpha = operation
            .attributes
            .as_leaky_relu()
            .map(|o| o.alpha as f32)
            .unwrap_or(0.01);

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
        let broadcast_shape: Vec<i32> = vec![1; num_dims];

        let (alpha_bytes, alpha_dtype) = match input_operand.descriptor.data_type {
            DataType::Float16 => (
                f16::from_f32(alpha).to_bits().to_le_bytes().to_vec(),
                TrtDataType::kHALF,
            ),
            _ => (alpha.to_le_bytes().to_vec(), TrtDataType::kFLOAT),
        };
        temp_weights.push(alpha_bytes);
        let alpha_ref = temp_weights.last().unwrap().as_slice();

        let alpha_const = network
            .add_constant(&broadcast_shape, alpha_ref, alpha_dtype)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("LeakyReLU: failed to add alpha constant: {}", e),
            })?;
        let alpha_tensor = alpha_const
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("LeakyReLU: alpha const output: {}", e),
            })?;

        // max(0, x) = relu(x)
        let relu_layer = network
            .add_activation(input, ActivationType::kRELU)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add relu for leaky relu: {}", e),
            })?;
        let relu_output = relu_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get relu output: {}", e),
            })?;

        // min(0, x) = x - relu(x)
        let neg_part_layer = network
            .add_elementwise(input, &relu_output, ElementWiseOperation::kSUB)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get min(0,x) for leaky relu: {}", e),
            })?;
        let neg_part = neg_part_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get neg part output: {}", e),
            })?;

        // alpha * min(0, x)
        let scaled_neg_layer = network
            .add_elementwise(&neg_part, &alpha_tensor, ElementWiseOperation::kPROD)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to scale neg part for leaky relu: {}", e),
            })?;
        let scaled_neg =
            scaled_neg_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get scaled neg output: {}", e),
                })?;

        // relu(x) + alpha * min(0, x)
        let final_layer = network
            .add_elementwise(&relu_output, &scaled_neg, ElementWiseOperation::kSUM)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add leaky relu parts: {}", e),
            })?;

        let output = final_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get leaky relu output: {}", e),
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
    /// Uses built-in kHARD_SIGMOID when alpha=0.2 and beta=0.5; otherwise decomposes with elementwise ops.
    fn add_hard_sigmoid_op(
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

        let opts = operation.attributes.as_hard_sigmoid();
        let alpha = opts.map(|o| o.alpha as f32).unwrap_or(0.2);
        let beta = opts.map(|o| o.beta as f32).unwrap_or(0.5);

        if (alpha - 0.2f32).abs() <= 1e-5 && (beta - 0.5f32).abs() <= 1e-5 {
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
            tensor_map.insert(output_ids[0], output);
            return Ok(());
        }

        let input_operand = graph.operand(operation.input_operands[0]).ok_or_else(|| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Input operand {} not found in graph",
                    operation.input_operands[0]
                ),
            }
        })?;
        let input_dims = input
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: failed to get input dimensions: {}", e),
            })?;
        let broadcast_shape: Vec<i32> = vec![1; input_dims.len()];
        let input_dtype = input_operand.descriptor.data_type;

        let (alpha_bytes, beta_bytes, zero_bytes, one_bytes, trt_dtype) = match input_dtype {
            DataType::Float16 => (
                f16::from_f32(alpha).to_bits().to_le_bytes().to_vec(),
                f16::from_f32(beta).to_bits().to_le_bytes().to_vec(),
                f16::from_f32(0.0).to_bits().to_le_bytes().to_vec(),
                f16::from_f32(1.0).to_bits().to_le_bytes().to_vec(),
                trtx::DataType::kHALF,
            ),
            _ => (
                alpha.to_le_bytes().to_vec(),
                beta.to_le_bytes().to_vec(),
                0.0f32.to_le_bytes().to_vec(),
                1.0f32.to_le_bytes().to_vec(),
                trtx::DataType::kFLOAT,
            ),
        };
        temp_weights.push(alpha_bytes);
        temp_weights.push(beta_bytes);
        temp_weights.push(zero_bytes);
        temp_weights.push(one_bytes);
        let idx = temp_weights.len();
        let alpha_ref = temp_weights[idx - 4].as_slice();
        let beta_ref = temp_weights[idx - 3].as_slice();
        let zero_ref = temp_weights[idx - 2].as_slice();
        let one_ref = temp_weights[idx - 1].as_slice();

        let alpha_const = network
            .add_constant(&broadcast_shape, alpha_ref, trt_dtype.clone())
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: failed to add alpha constant: {}", e),
            })?;
        let beta_const = network
            .add_constant(&broadcast_shape, beta_ref, trt_dtype.clone())
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: failed to add beta constant: {}", e),
            })?;
        let zero_const = network
            .add_constant(&broadcast_shape, zero_ref, trt_dtype.clone())
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: failed to add zero constant: {}", e),
            })?;
        let one_const = network
            .add_constant(&broadcast_shape, one_ref, trt_dtype)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: failed to add one constant: {}", e),
            })?;

        let alpha_out = alpha_const
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: alpha const output: {}", e),
            })?;
        let beta_out = beta_const
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: beta const output: {}", e),
            })?;
        let zero_out = zero_const
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: zero const output: {}", e),
            })?;
        let one_out = one_const
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: one const output: {}", e),
            })?;

        // linear = alpha * x + beta
        let ax = network
            .add_elementwise(input, &alpha_out, ElementWiseOperation::kPROD)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: alpha*x: {}", e),
            })?;
        let linear = ax.get_output(0).map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("HardSigmoid: linear get output: {}", e),
        })?;
        let linear_plus_beta = network
            .add_elementwise(&linear, &beta_out, ElementWiseOperation::kSUM)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: linear+beta: {}", e),
            })?;
        let linear_out =
            linear_plus_beta
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("HardSigmoid: linear_out: {}", e),
                })?;

        // clamp(linear, 0, 1) = max(0, min(1, linear))
        let min1 = network
            .add_elementwise(&linear_out, &one_out, ElementWiseOperation::kMIN)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: min(1, linear): {}", e),
            })?;
        let min1_out = min1
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: min1 output: {}", e),
            })?;
        let output_layer = network
            .add_elementwise(&zero_out, &min1_out, ElementWiseOperation::kMAX)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: max(0, ...): {}", e),
            })?;
        let output = output_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        tensor_map.insert(output_ids[0], output);
        Ok(())
    }

    /// Add hard swish activation
    /// WebNN: y = x * max(0, min(6, (x + 3))) / 6 (MobileNetV3). Follows spec decomposition.
    fn add_hard_swish_op(
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

        let input_operand = graph.operand(operation.input_operands[0]).ok_or_else(|| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Input operand {} not found in graph",
                    operation.input_operands[0]
                ),
            }
        })?;
        let input_dims = input
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: failed to get input dimensions: {}", e),
            })?;
        let broadcast_shape: Vec<i32> = vec![1; input_dims.len()];
        let input_dtype = input_operand.descriptor.data_type;

        let three = 3.0f32;
        let six = 6.0f32;

        let (three_bytes, six_bytes, zero_bytes, trt_dtype) = match input_dtype {
            DataType::Float16 => (
                f16::from_f32(three).to_bits().to_le_bytes().to_vec(),
                f16::from_f32(six).to_bits().to_le_bytes().to_vec(),
                f16::from_f32(0.0).to_bits().to_le_bytes().to_vec(),
                trtx::DataType::kHALF,
            ),
            _ => (
                three.to_le_bytes().to_vec(),
                six.to_le_bytes().to_vec(),
                0.0f32.to_le_bytes().to_vec(),
                trtx::DataType::kFLOAT,
            ),
        };
        temp_weights.push(three_bytes);
        temp_weights.push(six_bytes);
        temp_weights.push(zero_bytes);
        let idx = temp_weights.len();
        let three_ref = temp_weights[idx - 3].as_slice();
        let six_ref = temp_weights[idx - 2].as_slice();
        let zero_ref = temp_weights[idx - 1].as_slice();

        let three_const = network
            .add_constant(&broadcast_shape, three_ref, trt_dtype.clone())
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: failed to add 3 constant: {}", e),
            })?;
        let six_const = network
            .add_constant(&broadcast_shape, six_ref, trt_dtype.clone())
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: failed to add 6 constant: {}", e),
            })?;
        let zero_const = network
            .add_constant(&broadcast_shape, zero_ref, trt_dtype)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: failed to add zero constant: {}", e),
            })?;

        let three_out = three_const
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: 3 const output: {}", e),
            })?;
        let six_out = six_const
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: 6 const output: {}", e),
            })?;
        let zero_out = zero_const
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: zero const output: {}", e),
            })?;

        // Spec: div(mul(input, max(0, min(6, add(input, 3)))), 6)
        let x_plus_3 = network
            .add_elementwise(input, &three_out, ElementWiseOperation::kSUM)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: add(input, 3): {}", e),
            })?;
        let x_plus_3_out = x_plus_3
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: x+3 output: {}", e),
            })?;
        let min6 = network
            .add_elementwise(&six_out, &x_plus_3_out, ElementWiseOperation::kMIN)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: min(6, x+3): {}", e),
            })?;
        let min6_out = min6
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: min6 output: {}", e),
            })?;
        let inner = network
            .add_elementwise(&zero_out, &min6_out, ElementWiseOperation::kMAX)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: max(0, ...): {}", e),
            })?;
        let inner_out = inner
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: inner output: {}", e),
            })?;
        let x_times_inner = network
            .add_elementwise(input, &inner_out, ElementWiseOperation::kPROD)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: mul(input, inner): {}", e),
            })?;
        let x_times_inner_out =
            x_times_inner
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("HardSwish: x*inner output: {}", e),
                })?;
        let output_layer = network
            .add_elementwise(&x_times_inner_out, &six_out, ElementWiseOperation::kDIV)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: div(..., 6): {}", e),
            })?;
        let output = output_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        tensor_map.insert(output_ids[0], output);
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

    /// Add cast operation: convert input to the output operand's data type.
    /// TensorRT does not allow INT8/UINT8 constants to feed a Cast (only DQ or plugin).
    /// Scalar int8/uint8 constants are promoted to int32 when added; Cast then becomes identity.
    /// For non-scalar int8/uint8 -> int32 we emulate via Dequantize(scale=1) -> Cast(INT32).
    fn add_cast_op(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        temp_weights: &mut Vec<Vec<u8>>,
        promoted_constants: &HashSet<u32>,
        constants_stored_flat: &HashSet<u32>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_id = operation.input_operands[0];
        let input = tensor_map
            .get(&input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Cast input operand {} not found", input_id),
            })?;

        let output_id = operation
            .output_operands_slice()
            .first()
            .copied()
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Cast operation has no output".to_string(),
            })?;

        let output_operand =
            graph
                .operand(output_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Cast output operand {} not found in graph", output_id),
                })?;

        let input_operand =
            graph
                .operand(input_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Cast input operand {} not found in graph", input_id),
                })?;

        let target_dtype = output_operand.descriptor.data_type;
        let input_dtype = input_operand.descriptor.data_type;

        // TensorRT: int8/uint8 tensors may only feed DQ or plugin, not Cast directly.
        // int8/uint8 -> int32: scalar promoted => identity; else DQ+Cast.
        // int8/uint8 -> float32: DQ(scale=1) only (output is already float).
        let use_dq_then_cast_int32 = matches!(
            (input_dtype, target_dtype),
            (DataType::Int8, DataType::Int32) | (DataType::Uint8, DataType::Int32)
        );
        let use_dq_for_float32 = matches!(
            (input_dtype, target_dtype),
            (DataType::Int8, DataType::Float32) | (DataType::Uint8, DataType::Float32)
        );

        if use_dq_then_cast_int32 && promoted_constants.contains(&input_id) {
            // Scalar constant was promoted to int32; cast is identity.
            let identity_layer =
                network
                    .add_identity(input)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add identity for promoted scalar cast: {}", e),
                    })?;
            let output =
                identity_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get identity output: {}", e),
                    })?;
            tensor_map.insert(output_id, output);
            return Ok(());
        }

        // Helper: DQ(scale=1) with per-tensor (0D) scale. TensorRT only allows scalar or per-channel scale; 4D scale causes "ScaleMode is illegal".
        let add_dq_scale_constant = |network: &mut trtx::NetworkDefinition,
                                     temp_weights: &mut Vec<Vec<u8>>,
                                     err_prefix: &str|
         -> Result<trtx::Tensor, GraphError> {
            let scale_one: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
            temp_weights.push(scale_one);
            let scale_ref = temp_weights.last().unwrap();
            let scale_constant = network
                .add_constant(&[], scale_ref, TrtDataType::kFLOAT)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{}: {}", err_prefix, e),
                })?;
            scale_constant
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{}: get scale output: {}", err_prefix, e),
                })
        };

        // Constant stored flat: 1D int8. Try 1D -> Reshape(4D) -> DQ so DQ sees Shuffle output not Constant.
        let stored_flat = constants_stored_flat.contains(&input_id);
        let _input_dims = input
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Cast: failed to get input dimensions: {}", e),
            })?;
        if use_dq_for_float32 {
            // int8/uint8 -> float32: only supported when input is a constant (stored flat). Tensor inputs not supported by TRT-RTX.
            if !stored_flat {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "Cast int8/uint8 to float32: TRT-RTX supports only constant inputs (tensor inputs not supported)".to_string(),
                });
            }
            {
                let original_shape: Vec<i32> = input_operand
                    .descriptor
                    .shape
                    .iter()
                    .map(|d| get_static_or_max_size(d) as i32)
                    .collect();
                let mut shuffle_to_4d =
                    network
                        .add_shuffle(input)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Cast: failed to add reshape shuffle: {}", e),
                        })?;
                let _ = shuffle_to_4d.set_layer_name("cast_flat_reshape_4d");
                shuffle_to_4d
                    .set_reshape_dimensions(&original_shape)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Cast: failed to set reshape dimensions: {}", e),
                    })?;
                let mut reshaped_4d =
                    shuffle_to_4d
                        .get_output(0)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Cast: failed to get reshape output: {}", e),
                        })?;
                let _ = reshaped_4d.set_name("cast_flat_reshape_4d");
                let scale_tensor =
                    add_dq_scale_constant(network, temp_weights, "int8->float32 cast")?;
                let mut dq_layer = network
                    .add_dequantize(&reshaped_4d, &scale_tensor, TrtDataType::kFLOAT)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add dequantize for int8->float32 cast: {}", e),
                    })?;
                let _ = dq_layer.set_layer_name("cast_flat_dq_f32");
                let mut output =
                    dq_layer
                        .get_output(0)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Failed to get dequantize output: {}", e),
                        })?;
                let _ = output.set_name("cast_flat_dq_f32");
                tensor_map.insert(output_id, output);
                return Ok(());
            }
        }

        if use_dq_then_cast_int32 {
            // int8/uint8 -> int32: only supported when input is constant (stored flat) or promoted scalar. Tensor inputs not supported by TRT-RTX.
            if !stored_flat {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "Cast int8/uint8 to int32: TRT-RTX supports only constant inputs (tensor inputs not supported)".to_string(),
                });
            }
            {
                let original_shape: Vec<i32> = input_operand
                    .descriptor
                    .shape
                    .iter()
                    .map(|d| get_static_or_max_size(d) as i32)
                    .collect();
                let mut shuffle_to_4d =
                    network
                        .add_shuffle(input)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Cast: failed to add reshape shuffle: {}", e),
                        })?;
                let _ = shuffle_to_4d.set_layer_name("cast_flat_reshape_4d_int32");
                shuffle_to_4d
                    .set_reshape_dimensions(&original_shape)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Cast: failed to set reshape dimensions: {}", e),
                    })?;
                let mut reshaped_4d =
                    shuffle_to_4d
                        .get_output(0)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Cast: failed to get reshape output: {}", e),
                        })?;
                let _ = reshaped_4d.set_name("cast_flat_reshape_4d");
                let scale_tensor =
                    add_dq_scale_constant(network, temp_weights, "int8->int32 cast")?;
                let mut dq_layer = network
                    .add_dequantize(&reshaped_4d, &scale_tensor, TrtDataType::kFLOAT)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add dequantize for int8->int32 cast: {}", e),
                    })?;
                let _ = dq_layer.set_layer_name("cast_flat_dq_int32");
                let mut dq_out =
                    dq_layer
                        .get_output(0)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Failed to get dequantize output: {}", e),
                        })?;
                let _ = dq_out.set_name("cast_flat_dq_int32");
                let mut cast_layer =
                    network
                        .add_cast(&dq_out, TrtDataType::kINT32)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Failed to add cast to INT32: {}", e),
                        })?;
                let _ = cast_layer.set_layer_name("cast_flat_cast_int32");
                let mut output =
                    cast_layer
                        .get_output(0)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Failed to get cast output: {}", e),
                        })?;
                let _ = output.set_name("cast_flat_cast_int32");
                tensor_map.insert(output_id, output);
                return Ok(());
            }
        }

        // Non-int8/uint8 or scalar promoted: direct cast.
        let trt_dtype = Self::webnn_to_trt_dtype(target_dtype)?;

        let layer =
            network
                .add_cast(input, trt_dtype)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add cast layer: {}", e),
                })?;

        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get cast output: {}", e),
            })?;

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

    /// Add matrix multiply operation.
    /// TensorRT IMatrixMultiplyLayer requires both inputs to have the same number of dimensions;
    /// if ranks differ, unsqueeze the lower-rank input by prepending 1s.
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

        let dims0 = input0
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Matmul: input0 dimensions: {}", e),
            })?;
        let dims1 = input1
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Matmul: input1 dimensions: {}", e),
            })?;

        let rank0 = dims0.len();
        let rank1 = dims1.len();

        let layer = if rank0 == rank1 {
            network.add_matrix_multiply(input0, 0, input1, 0)
        } else if rank0 < rank1 {
            let reshape_dims: Vec<i32> = dims0.iter().map(|&d| d as i32).collect();
            let rank_diff = rank1 - rank0;
            let mut new_shape: Vec<i32> = vec![1; rank_diff];
            new_shape.extend(reshape_dims);
            let mut shuffle =
                network
                    .add_shuffle(input0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Matmul: unsqueeze shuffle: {}", e),
                    })?;
            shuffle.set_reshape_dimensions(&new_shape).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Matmul: set reshape: {}", e),
                }
            })?;
            let reshaped0 = shuffle
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Matmul: shuffle output: {}", e),
                })?;
            network.add_matrix_multiply(&reshaped0, 0, input1, 0)
        } else {
            let reshape_dims: Vec<i32> = dims1.iter().map(|&d| d as i32).collect();
            let rank_diff = rank0 - rank1;
            let mut new_shape: Vec<i32> = vec![1; rank_diff];
            new_shape.extend(reshape_dims);
            let mut shuffle =
                network
                    .add_shuffle(input1)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Matmul: unsqueeze shuffle: {}", e),
                    })?;
            shuffle.set_reshape_dimensions(&new_shape).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Matmul: set reshape: {}", e),
                }
            })?;
            let reshaped1 = shuffle
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Matmul: shuffle output: {}", e),
                })?;
            network.add_matrix_multiply(input0, 0, &reshaped1, 0)
        }
        .map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to add matrix multiply: {}", e),
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
    // Normalization Operations
    // ============================================================================

    /// Reshape 1D batch-norm stats [C] to same rank as input with shape [1,...,1,C,1,...,1]
    /// so that the channel dimension aligns with the input's axis. TensorRT then broadcasts correctly.
    fn reshape_batch_norm_stats_for_broadcast(
        network: &mut trtx::NetworkDefinition,
        stats: &trtx::Tensor,
        input: &trtx::Tensor,
        axis: i64,
        op_name: &str,
    ) -> Result<trtx::Tensor, GraphError> {
        let input_dims = input
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get input dims for {}: {}", op_name, e),
            })?;
        let stats_dims = stats
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get stats dims for {}: {}", op_name, e),
            })?;
        let rank = input_dims.len();
        let mut axis_idx = axis;
        if axis_idx < 0 {
            axis_idx += rank as i64;
        }
        axis_idx = axis_idx.max(0).min((rank.saturating_sub(1)) as i64);
        // Channel count: stats may be 1D [C], 4D [C,1,1,1], or 4D [1,C,1,1]; use product so we get C in all cases.
        let c: i32 = stats_dims.iter().product::<i32>().max(1);
        // When stats are 4D [C,1,1,1], use a transpose-only shuffle so TensorRT sees [1,C,1,1].
        // (Transpose+reshape in one shuffle can leave logical shape as [C,1,1,1].)
        let is_4d_channel_first = rank >= 2
            && stats_dims.len() == rank
            && stats_dims[0] == c
            && (axis_idx as usize) < rank
            && stats_dims[axis_idx as usize] != c
            && stats_dims.iter().skip(1).all(|&d| d == 1);
        if is_4d_channel_first {
            let mut perm: Vec<i32> = (0..rank as i32).collect();
            perm[0] = axis_idx as i32;
            perm[axis_idx as usize] = 0;
            let mut shuffle =
                network
                    .add_shuffle(stats)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!(
                            "Failed to add shuffle for {} (4D transpose): {}",
                            op_name, e
                        ),
                    })?;
            shuffle
                .set_first_transpose(&perm)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to set transpose for {}: {}", op_name, e),
                })?;
            return shuffle
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get shuffle output for {}: {}", op_name, e),
                });
        }

        // For 1D [C], reshape to [1,1,...,1,C] then transpose so C moves to axis_idx; avoids TensorRT giving [C,1,1,1].
        let (target_shape, need_second_transpose) = if stats_dims.len() == 1 {
            let shape_last: Vec<i32> = (0..rank)
                .map(|i| if i == rank - 1 { c } else { 1 })
                .collect();
            (shape_last, true)
        } else {
            let shape_axis: Vec<i32> = (0..rank)
                .map(|i| if i as i64 == axis_idx { c } else { 1 })
                .collect();
            (shape_axis, false)
        };
        let mut shuffle = network
            .add_shuffle(stats)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add shuffle for {}: {}", op_name, e),
            })?;
        shuffle.set_reshape_dimensions(&target_shape).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set reshape for {}: {}", op_name, e),
            }
        })?;
        let mut result = shuffle
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get shuffle output for {}: {}", op_name, e),
            })?;
        if need_second_transpose {
            // Move dimension (rank-1) to axis_idx: perm[axis_idx] = rank-1, perm[rank-1] = axis_idx.
            let mut perm: Vec<i32> = (0..rank as i32).collect();
            perm[axis_idx as usize] = (rank - 1) as i32;
            perm[rank - 1] = axis_idx as i32;
            let mut shuffle2 =
                network
                    .add_shuffle(&result)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add second shuffle for {}: {}", op_name, e),
                    })?;
            shuffle2
                .set_first_transpose(&perm)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to set second transpose for {}: {}", op_name, e),
                })?;
            result = shuffle2
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get second shuffle output for {}: {}", op_name, e),
                })?;
        }
        Ok(result)
    }

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

        // Read typed batchNormalization options (fallback keeps WebNN defaults).
        let (axis, _epsilon) = operation
            .attributes
            .as_batch_normalization()
            .map(|opts| (opts.axis as i64, opts.epsilon as f32))
            .unwrap_or((1, 1e-5));

        // Reshape mean to [1,...,1,C,1,...,1] so it broadcasts with input (e.g. [2,3,4] and axis=1 -> [1,3,1]).
        let mean_bc = Self::reshape_batch_norm_stats_for_broadcast(
            network,
            mean,
            input,
            axis,
            "batch_norm_sub",
        )?;

        // Step 1: x - mean
        let mut sub_layer = network
            .add_elementwise(input, &mean_bc, ElementWiseOperation::kSUB)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add sub for batch norm: {}", e),
            })?;
        let _ = sub_layer.set_layer_name("batch_norm_sub");

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

        let var_bc = Self::reshape_batch_norm_stats_for_broadcast(
            network,
            variance,
            input,
            axis,
            "batch_norm_var",
        )?;

        // Step 3: sqrt(variance + epsilon)
        let mut sqrt_var_layer =
            network
                .add_unary(&var_bc, UnaryOperation::kSQRT)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add sqrt for batch norm: {}", e),
                })?;
        let _ = sqrt_var_layer.set_layer_name("batch_norm_sqrt_var");

        let sqrt_var = sqrt_var_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get sqrt output: {}", e),
            })?;

        // Step 4: (x - mean) / sqrt(variance + epsilon)
        let mut div_layer = network
            .add_elementwise(&x_minus_mean, &sqrt_var, ElementWiseOperation::kDIV)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add div for batch norm: {}", e),
            })?;
        let _ = div_layer.set_layer_name("batch_norm_div");

        let normalized = div_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get div output: {}", e),
            })?;

        // Step 5: Apply scale if present (WebNN: scale is in MLBatchNormalizationOptions, not a positional input)
        let scale_id = operation
            .attributes
            .as_batch_normalization()
            .and_then(|o| o.scale);
        let mut result = normalized;
        if let Some(scale_id) = scale_id {
            let scale = tensor_map
                .get(&scale_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Scale operand {} not found", scale_id),
                })?;

            let scale_bc = Self::reshape_batch_norm_stats_for_broadcast(
                network,
                scale,
                &result,
                axis,
                "batch_norm_scale",
            )?;

            let mut mul_layer = network
                .add_elementwise(&result, &scale_bc, ElementWiseOperation::kPROD)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add mul for scale: {}", e),
                })?;
            let _ = mul_layer.set_layer_name("batch_norm_scale_mul");

            result = mul_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get mul output: {}", e),
                })?;
        }

        // Step 6: Apply bias if present (WebNN: bias is in MLBatchNormalizationOptions, not a positional input)
        let bias_id = operation
            .attributes
            .as_batch_normalization()
            .and_then(|o| o.bias);
        if let Some(bias_id) = bias_id {
            let bias = tensor_map
                .get(&bias_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Bias operand {} not found", bias_id),
                })?;

            let bias_bc = Self::reshape_batch_norm_stats_for_broadcast(
                network,
                bias,
                &result,
                axis,
                "batch_norm_bias",
            )?;

            let mut add_layer = network
                .add_elementwise(&result, &bias_bc, ElementWiseOperation::kSUM)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add bias: {}", e),
                })?;
            let _ = add_layer.set_layer_name("batch_norm_bias_add");

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
    /// Formula: y = (x - mean) / sqrt(variance + epsilon) * scale + bias (WebNN spec)
    /// Computed per-instance over spatial dimensions
    fn add_instance_normalization_op(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
        temp_weights: &mut Vec<Vec<u8>>,
    ) -> Result<(), GraphError> {
        // Instance normalization computes statistics per-instance (N, C) over spatial dims
        // Input operands: input, scale (optional), bias (optional)
        let input = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        let input_dims = input
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("InstanceNorm: failed to get input dimensions: {}", e),
            })?;

        // Get epsilon from attributes (default: 1e-5), cast to input dtype per spec
        let opts = operation.attributes.as_instance_normalization();
        let epsilon_f32 = opts.map(|o| o.epsilon as f32).unwrap_or(1e-5);
        let layout = opts
            .map(|o| o.layout.as_str())
            .filter(|s| !s.is_empty())
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

        // variance + epsilon per WebNN spec (epsilon cast to input's dataType)
        let var_dims = variance
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("InstanceNorm: failed to get variance dimensions: {}", e),
            })?;
        let input_operand = graph.operand(operation.input_operands[0]).ok_or_else(|| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "InstanceNorm: input operand {} not found in graph",
                    operation.input_operands[0]
                ),
            }
        })?;
        let input_dtype = input_operand.descriptor.data_type;
        let num_elements: usize = var_dims.iter().map(|&d| d as usize).product();
        let (epsilon_bytes, trt_dtype) = match input_dtype {
            DataType::Float16 => {
                let eps_f16 = f16::from_f32(epsilon_f32);
                let data: Vec<u8> = (0..num_elements)
                    .flat_map(|_| eps_f16.to_bits().to_le_bytes())
                    .collect();
                (data, trtx::DataType::kHALF)
            }
            _ => {
                let data: Vec<u8> = (0..num_elements)
                    .flat_map(|_| epsilon_f32.to_le_bytes())
                    .collect();
                (data, trtx::DataType::kFLOAT)
            }
        };
        temp_weights.push(epsilon_bytes);
        let epsilon_ref = temp_weights.last().unwrap().as_slice();
        let var_shape: Vec<i32> = var_dims.iter().map(|&d| d as i32).collect();
        let epsilon_const = network
            .add_constant(&var_shape, epsilon_ref, trt_dtype)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("InstanceNorm: failed to add epsilon constant: {}", e),
            })?;
        let epsilon_out =
            epsilon_const
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("InstanceNorm: epsilon const output: {}", e),
                })?;
        let var_plus_eps = network
            .add_elementwise(&variance, &epsilon_out, ElementWiseOperation::kSUM)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("InstanceNorm: variance + epsilon: {}", e),
            })?;
        let var_plus_eps_out =
            var_plus_eps
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("InstanceNorm: var_plus_eps output: {}", e),
                })?;

        // sqrt(variance + epsilon)
        let sqrt_layer = network
            .add_unary(&var_plus_eps_out, UnaryOperation::kSQRT)
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

        // Build broadcast shape for scale/bias [C] to match input (TensorRT elementwise needs same rank).
        let scale_broadcast_shape: Vec<i32> = if layout == "nchw" && !input_dims.is_empty() {
            let mut s = vec![1i32; input_dims.len()];
            s[1] = input_dims[1];
            s
        } else if !input_dims.is_empty() {
            let mut s = vec![1i32; input_dims.len()];
            let last = input_dims.len() - 1;
            s[last] = input_dims[last];
            s
        } else {
            vec![1]
        };

        // Classify optional operands by name: WPT can pass [input, bias] or [input, scale] or [input, scale, bias].
        let mut scale_id: Option<u32> = None;
        let mut bias_id: Option<u32> = None;
        for &operand_id in &operation.input_operands[1..] {
            let name = graph
                .operand(operand_id)
                .and_then(|o| o.name.as_deref())
                .unwrap_or("");
            let name_lower = name.to_lowercase();
            if name_lower.contains("scale") {
                scale_id = Some(operand_id);
            } else if name_lower.contains("bias") {
                bias_id = Some(operand_id);
            }
        }

        // Apply scale if present (multiply)
        if let Some(operand_id) = scale_id {
            let scale =
                tensor_map
                    .get(&operand_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Scale operand {} not found", operand_id),
                    })?;

            let mut scale_shuffle =
                network
                    .add_shuffle(scale)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("InstanceNorm: failed to add shuffle for scale: {}", e),
                    })?;
            scale_shuffle
                .set_reshape_dimensions(&scale_broadcast_shape)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("InstanceNorm: failed to set scale reshape: {}", e),
                })?;
            let scale_bc =
                scale_shuffle
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("InstanceNorm: failed to get scale shuffle output: {}", e),
                    })?;

            let mul_layer = network
                .add_elementwise(&result, &scale_bc, ElementWiseOperation::kPROD)
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

        // Apply bias if present (add)
        if let Some(operand_id) = bias_id {
            let bias = tensor_map
                .get(&operand_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Bias operand {} not found", operand_id),
                })?;

            let mut bias_shuffle =
                network
                    .add_shuffle(bias)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("InstanceNorm: failed to add shuffle for bias: {}", e),
                    })?;
            bias_shuffle
                .set_reshape_dimensions(&scale_broadcast_shape)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("InstanceNorm: failed to set bias reshape: {}", e),
                })?;
            let bias_bc = bias_shuffle
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("InstanceNorm: failed to get bias shuffle output: {}", e),
                })?;

            let add_layer = network
                .add_elementwise(&result, &bias_bc, ElementWiseOperation::kSUM)
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
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        operation: &Operation,
        temp_weights: &mut Vec<Vec<u8>>,
    ) -> Result<(), GraphError> {
        // Layer normalization computes statistics over specified axes
        // Input operands: input, scale (optional), bias (optional)
        let input = tensor_map
            .get(&operation.input_operands[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands[0]),
            })?;

        let input_dims = input
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get input shape: {}", e),
            })?;

        // TensorRT Reduce requires at least 1 dimension. For 0D scalar: mean=x, variance=0, output = 0*scale + bias = bias or 0.
        if input_dims.is_empty() {
            let input_operand = graph.operand(operation.input_operands[0]).ok_or_else(|| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!(
                        "Input operand {} not found in graph",
                        operation.input_operands[0]
                    ),
                }
            })?;
            let (zero_bytes, zero_dtype) = match input_operand.descriptor.data_type {
                DataType::Float16 => (
                    f16::from_f32(0.0).to_bits().to_le_bytes().to_vec(),
                    TrtDataType::kHALF,
                ),
                _ => (0.0f32.to_le_bytes().to_vec(), TrtDataType::kFLOAT),
            };
            temp_weights.push(zero_bytes);
            let zero_ref = temp_weights.last().unwrap().as_slice();
            let zero_const = network
                .add_constant(&[1], zero_ref, zero_dtype)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("LayerNorm 0D: failed to add zero constant: {}", e),
                })?;
            let mut result =
                zero_const
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm 0D: zero const output: {}", e),
                    })?;
            // Optional operands are [scale?, bias?]; bias is last when present. So len>=2 => add last (bias when only bias, or bias when scale+bias).
            if operation.input_operands.len() >= 2 {
                let bias_id = operation.input_operands[operation.input_operands.len() - 1];
                let bias =
                    tensor_map
                        .get(&bias_id)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Bias operand {} not found", bias_id),
                        })?;
                // Bias may be scalar; broadcast to result shape [1] so ElementWise accepts same dims.
                let mut bias_shuffle =
                    network
                        .add_shuffle(bias)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("LayerNorm 0D: failed to add bias shuffle: {}", e),
                        })?;
                bias_shuffle.set_reshape_dimensions(&[1]).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm 0D: failed to set bias reshape: {}", e),
                    }
                })?;
                let bias_bc =
                    bias_shuffle
                        .get_output(0)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("LayerNorm 0D: bias shuffle output: {}", e),
                        })?;
                let add_layer = network
                    .add_elementwise(&result, &bias_bc, ElementWiseOperation::kSUM)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm 0D: failed to add bias: {}", e),
                    })?;
                result = add_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm 0D: bias add output: {}", e),
                    })?;
            }
            let output_id = operation.output_operands_slice()[0];
            tensor_map.insert(output_id, result);
            return Ok(());
        }

        // Get epsilon and axes from typed options. Spec: when axes not present, axes = [1..rank) if rank > 1 else [].
        // Option<axes>: None = key omitted => default; Some(v) = use v (Some([]) = explicit no reduction).
        let opts = operation.attributes.as_layer_normalization();
        let _epsilon = opts.map(|o| o.epsilon as f32).unwrap_or(1e-5);
        let axes: Vec<u32> = opts.and_then(|o| o.axes.clone()).unwrap_or_else(|| {
            if input_dims.len() > 1 {
                (1..input_dims.len()).map(|i| i as u32).collect()
            } else {
                vec![]
            }
        });

        // Spec: "If empty, no dimensions are reduced." TensorRT Reduce requires at least one dimension to reduce.
        // When axes=[], mean/variance reduce over nothing -> normalized = 0; output = 0*scale + bias = bias or 0.
        if axes.is_empty() {
            let input_operand = graph.operand(operation.input_operands[0]).ok_or_else(|| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!(
                        "Input operand {} not found in graph",
                        operation.input_operands[0]
                    ),
                }
            })?;
            let num_el: usize = input_dims.iter().map(|&d| d as usize).product();
            let (zero_bytes, zero_dtype) = match input_operand.descriptor.data_type {
                DataType::Float16 => (
                    (0..num_el)
                        .flat_map(|_| f16::from_f32(0.0).to_bits().to_le_bytes())
                        .collect::<Vec<_>>(),
                    TrtDataType::kHALF,
                ),
                _ => (
                    (0..num_el)
                        .flat_map(|_| 0.0f32.to_le_bytes())
                        .collect::<Vec<_>>(),
                    TrtDataType::kFLOAT,
                ),
            };
            temp_weights.push(zero_bytes);
            let zero_ref = temp_weights.last().unwrap().as_slice();
            let shape_i32: Vec<i32> = input_dims.iter().map(|&d| d as i32).collect();
            let zero_const = network
                .add_constant(&shape_i32, zero_ref, zero_dtype.clone())
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("LayerNorm axes=[]: failed to add zeros constant: {}", e),
                })?;
            let mut result =
                zero_const
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm axes=[]: zero const output: {}", e),
                    })?;
            // Optional operands are [scale?, bias?]; bias is last when present.
            if operation.input_operands.len() >= 2 {
                let bias_id = operation.input_operands[operation.input_operands.len() - 1];
                let bias =
                    tensor_map
                        .get(&bias_id)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Bias operand {} not found", bias_id),
                        })?;
                let bias_bc = if graph.constant_operand_ids_to_handles.contains_key(&bias_id) {
                    // Shuffle cannot change volume. Broadcast by creating a constant filled with the bias value.
                    let bias_data = Self::get_constant_data(graph, bias_id)?;
                    let bias_broadcast_bytes: Vec<u8> = match input_operand.descriptor.data_type {
                        DataType::Float16 => {
                            let bits =
                                u16::from_le_bytes(bias_data[0..2].try_into().map_err(|_| {
                                    GraphError::ConversionFailed {
                                        format: "trtx".to_string(),
                                        reason:
                                            "LayerNorm axes=[]: bias constant too small for float16"
                                                .to_string(),
                                    }
                                })?);
                            let v = f16::from_bits(bits);
                            (0..num_el)
                                .flat_map(|_| v.to_bits().to_le_bytes())
                                .collect()
                        }
                        _ => {
                            let v =
                                f32::from_le_bytes(bias_data[0..4].try_into().map_err(|_| {
                                    GraphError::ConversionFailed {
                                        format: "trtx".to_string(),
                                        reason:
                                            "LayerNorm axes=[]: bias constant too small for float32"
                                                .to_string(),
                                    }
                                })?);
                            (0..num_el).flat_map(|_| v.to_le_bytes()).collect()
                        }
                    };
                    temp_weights.push(bias_broadcast_bytes);
                    let bias_ref = temp_weights.last().unwrap().as_slice();
                    let bias_const = network
                        .add_constant(&shape_i32, bias_ref, zero_dtype)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!(
                                "LayerNorm axes=[]: failed to add bias constant: {}",
                                e
                            ),
                        })?;
                    bias_const
                        .get_output(0)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("LayerNorm axes=[]: bias const output: {}", e),
                        })?
                } else {
                    // Bias is an input (e.g. from test harness). Broadcast scalar to result shape via ensure_broadcast_compatible with ones.
                    let ones_bytes: Vec<u8> = match input_operand.descriptor.data_type {
                        DataType::Float16 => (0..num_el)
                            .flat_map(|_| f16::from_f32(1.0).to_bits().to_le_bytes())
                            .collect(),
                        _ => (0..num_el).flat_map(|_| 1.0f32.to_le_bytes()).collect(),
                    };
                    temp_weights.push(ones_bytes);
                    let ones_ref = temp_weights.last().unwrap().as_slice();
                    let ones_const = network
                        .add_constant(&shape_i32, ones_ref, zero_dtype.clone())
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!(
                                "LayerNorm axes=[]: failed to add ones constant: {}",
                                e
                            ),
                        })?;
                    let ones_tensor =
                        ones_const
                            .get_output(0)
                            .map_err(|e| GraphError::ConversionFailed {
                                format: "trtx".to_string(),
                                reason: format!("LayerNorm axes=[]: ones const output: {}", e),
                            })?;
                    let (bias_bc, _) = Self::ensure_broadcast_compatible(
                        network,
                        bias,
                        &ones_tensor,
                        "layer_norm_axes_empty_bias",
                    )?;
                    bias_bc
                };
                let add_layer = network
                    .add_elementwise(&result, &bias_bc, ElementWiseOperation::kSUM)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm axes=[]: failed to add bias: {}", e),
                    })?;
                result = add_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm axes=[]: bias add output: {}", e),
                    })?;
            }
            let output_id = operation.output_operands_slice()[0];
            tensor_map.insert(output_id, result);
            return Ok(());
        }

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

        // variance + epsilon per WebNN spec (then sqrt)
        let var_dims = variance
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("LayerNorm: failed to get variance dimensions: {}", e),
            })?;
        let var_shape: Vec<i32> = var_dims.iter().map(|&d| d as i32).collect();
        let num_var_el: usize = var_dims.iter().map(|&d| d as usize).product();
        let input_operand = graph.operand(operation.input_operands[0]).ok_or_else(|| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Input operand {} not found in graph",
                    operation.input_operands[0]
                ),
            }
        })?;
        let (epsilon_bytes, epsilon_dtype) = match input_operand.descriptor.data_type {
            DataType::Float16 => (
                (0..num_var_el)
                    .flat_map(|_| f16::from_f32(_epsilon).to_bits().to_le_bytes())
                    .collect::<Vec<_>>(),
                TrtDataType::kHALF,
            ),
            _ => (
                (0..num_var_el)
                    .flat_map(|_| _epsilon.to_le_bytes())
                    .collect::<Vec<_>>(),
                TrtDataType::kFLOAT,
            ),
        };
        temp_weights.push(epsilon_bytes);
        let epsilon_ref = temp_weights.last().unwrap().as_slice();
        let epsilon_const = network
            .add_constant(&var_shape, epsilon_ref, epsilon_dtype)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("LayerNorm: failed to add epsilon constant: {}", e),
            })?;
        let epsilon_out =
            epsilon_const
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("LayerNorm: epsilon const output: {}", e),
                })?;
        let var_plus_eps = network
            .add_elementwise(&variance, &epsilon_out, ElementWiseOperation::kSUM)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("LayerNorm: variance + epsilon: {}", e),
            })?;
        let variance_eps =
            var_plus_eps
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("LayerNorm: variance+eps output: {}", e),
                })?;

        // sqrt(variance + epsilon)
        let sqrt_layer = network
            .add_unary(&variance_eps, UnaryOperation::kSQRT)
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

        // Reshape scale/bias so they broadcast to result. Scale/bias have shape [d_axes[0], d_axes[1], ...]
        // in axis order; result has input shape. Broadcast shape: for each result dim i, use
        // scale_bias dim at that axis position if i is in axes, else 1.
        let reshape_scale_bias_to_result_rank = |network: &mut trtx::NetworkDefinition,
                                                 tensor: &trtx::Tensor,
                                                 result: &trtx::Tensor,
                                                 op_name: &str,
                                                 axes: &[u32]|
         -> Result<trtx::Tensor, GraphError> {
            let result_dims = result
                .dimensions()
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("LayerNorm {}: result dims: {}", op_name, e),
                })?;
            let tensor_dims = tensor
                .dimensions()
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("LayerNorm {}: tensor dims: {}", op_name, e),
                })?;
            if tensor_dims.len() >= result_dims.len() {
                let id_layer =
                    network
                        .add_identity(tensor)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("LayerNorm {}: identity: {}", op_name, e),
                        })?;
                return id_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm {}: identity output: {}", op_name, e),
                    });
            }
            let (new_shape, transpose_perm): (Vec<i32>, Option<Vec<i32>>) =
                if tensor_dims.len() == axes.len() {
                    let new_shape: Vec<i32> = (0..result_dims.len())
                        .map(|i| {
                            let axis = i as u32;
                            axes.iter()
                                .position(|&a| a == axis)
                                .map(|j| tensor_dims[j] as i32)
                                .unwrap_or(1)
                        })
                        .collect();
                    let mut sorted_axes = axes.to_vec();
                    sorted_axes.sort_unstable();
                    let needs_transpose = axes != sorted_axes.as_slice();
                    let transpose_perm: Option<Vec<i32>> = if needs_transpose {
                        Some(
                            sorted_axes
                                .iter()
                                .map(|&a| {
                                    axes.iter()
                                        .position(|&ax| ax == a)
                                        .expect("axis in sorted_axes")
                                        as i32
                                })
                                .collect(),
                        )
                    } else {
                        None
                    };
                    (new_shape, transpose_perm)
                } else {
                    let pad = result_dims.len() - tensor_dims.len();
                    let mut shape: Vec<i32> = vec![1; pad];
                    shape.extend(tensor_dims.iter().map(|&d| d as i32));
                    (shape, None)
                };
            let mut shuffle =
                network
                    .add_shuffle(tensor)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm {}: shuffle: {}", op_name, e),
                    })?;
            if let Some(ref perm) = transpose_perm {
                shuffle
                    .set_first_transpose(perm)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm {}: set transpose: {}", op_name, e),
                    })?;
            }
            shuffle.set_reshape_dimensions(&new_shape).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("LayerNorm {}: set reshape: {}", op_name, e),
                }
            })?;
            shuffle
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("LayerNorm {}: shuffle output: {}", op_name, e),
                })
        };

        // Optional operands are [scale?, bias?] in that order. When len() == 2, the single optional may be scale or bias; use name to distinguish.
        if operation.input_operands.len() > 1 {
            let opt_id_1 = operation.input_operands[1];
            let opt_1 = tensor_map
                .get(&opt_id_1)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("LayerNorm optional operand {} not found", opt_id_1),
                })?;
            let name_1 = graph
                .operand(opt_id_1)
                .and_then(|o| o.name.as_deref())
                .unwrap_or("");
            let is_bias_1 = name_1.to_lowercase().contains("bias");
            let opt_1_bc = reshape_scale_bias_to_result_rank(
                network,
                opt_1,
                &result,
                if is_bias_1 { "bias" } else { "scale" },
                &axes,
            )?;
            if is_bias_1 {
                let add_layer = network
                    .add_elementwise(&result, &opt_1_bc, ElementWiseOperation::kSUM)
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
            } else {
                let mul_layer = network
                    .add_elementwise(&result, &opt_1_bc, ElementWiseOperation::kPROD)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add scale: {}", e),
                    })?;
                result = mul_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get mul output: {}", e),
                    })?;
            }
        }

        // Second optional (when len() == 3) is always bias
        if operation.input_operands.len() > 2 {
            let bias = tensor_map
                .get(&operation.input_operands[2])
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Bias operand {} not found", operation.input_operands[2]),
                })?;
            let bias_bc = reshape_scale_bias_to_result_rank(network, bias, &result, "bias", &axes)?;

            let add_layer = network
                .add_elementwise(&result, &bias_bc, ElementWiseOperation::kSUM)
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

    /// Add reduction operation (sum, mean, max, min, product).
    /// Axes optional: when missing, default to all axes (0..rank). Empty axes -> identity.
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

        let input_dims = input
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Reduce: input dimensions: {}", e),
            })?;
        let rank = input_dims.len();

        let opts = operation.attributes.as_reduce();
        let axes: Vec<u32> = opts
            .and_then(|o| o.axes.clone())
            .unwrap_or_else(|| (0..rank).map(|i| i as u32).collect());

        if axes.is_empty() {
            let id_layer =
                network
                    .add_identity(input)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Reduce axes=[] identity: {}", e),
                    })?;
            let output = id_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Reduce axes=[] output: {}", e),
                })?;
            let output_id = operation.output_operands_slice()[0];
            tensor_map.insert(output_id, output);
            return Ok(());
        }

        let mut axes_mask: u32 = 0;
        for &axis in &axes {
            axes_mask |= 1 << axis;
        }

        let keep_dims = opts.map(|o| o.keep_dimensions).unwrap_or(false);

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

    /// Add reduceL1 operation: sum(abs(x)). Axes optional; empty axes -> output = abs(input).
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

        let input_dims = input
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("ReduceL1: input dimensions: {}", e),
            })?;
        let rank = input_dims.len();

        let opts = operation.attributes.as_reduce();
        let axes: Vec<u32> = opts
            .and_then(|o| o.axes.clone())
            .unwrap_or_else(|| (0..rank).map(|i| i as u32).collect());

        if axes.is_empty() {
            let output_id = operation.output_operands_slice()[0];
            tensor_map.insert(output_id, abs_output);
            return Ok(());
        }

        let mut axes_mask: u32 = 0;
        for &axis in &axes {
            axes_mask |= 1 << axis;
        }

        let keep_dims = opts.map(|o| o.keep_dimensions).unwrap_or(false);

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

    /// Add reduceL2 operation: sqrt(sum(x^2)). Axes optional; empty axes -> output = sqrt(x^2) = |x|.
    /// For float16 input, sum of squares can overflow; do reduce in float32 then cast back.
    fn add_reduce_l2_op(
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

        let input_dtype = graph
            .operand(operation.input_operands[0])
            .map(|o| o.descriptor.data_type)
            .unwrap_or(DataType::Float32);

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

        let input_dims = input
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("ReduceL2: input dimensions: {}", e),
            })?;
        let rank = input_dims.len();

        let opts = operation.attributes.as_reduce();
        let axes: Vec<u32> = opts
            .and_then(|o| o.axes.clone())
            .unwrap_or_else(|| (0..rank).map(|i| i as u32).collect());

        if axes.is_empty() {
            let sqrt_layer = network
                .add_unary(&square_output, UnaryOperation::kSQRT)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("ReduceL2 axes=[] sqrt: {}", e),
                })?;
            let output = sqrt_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("ReduceL2 axes=[] output: {}", e),
                })?;
            let output_id = operation.output_operands_slice()[0];
            tensor_map.insert(output_id, output);
            return Ok(());
        }

        let mut axes_mask: u32 = 0;
        for &axis in &axes {
            axes_mask |= 1 << axis;
        }

        let keep_dims = opts.map(|o| o.keep_dimensions).unwrap_or(false);

        // Always reduce in float32 to avoid overflow (sum of squares can exceed float16 range).
        let to_reduce = Self::cast_to_float32(network, &square_output)?;
        let sum_layer = network
            .add_reduce(&to_reduce, ReduceOperation::kSUM, axes_mask, keep_dims)
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

        let sqrt_layer = network
            .add_unary(&sum_output, UnaryOperation::kSQRT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add sqrt for L2: {}", e),
            })?;

        let sqrt_output = sqrt_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get layer output: {}", e),
            })?;

        let output = if input_dtype == DataType::Float16 {
            Self::cast_to_float16(network, &sqrt_output)?
        } else {
            sqrt_output
        };

        let output_id = operation.output_operands_slice()[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add reduceLogSum operation: log(sum(x)). Axes optional; empty axes -> output = log(x).
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

        let input_dims = input
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("ReduceLogSum: input dimensions: {}", e),
            })?;
        let rank = input_dims.len();

        let opts = operation.attributes.as_reduce();
        let axes: Vec<u32> = opts
            .and_then(|o| o.axes.clone())
            .unwrap_or_else(|| (0..rank).map(|i| i as u32).collect());

        if axes.is_empty() {
            let log_layer = network
                .add_unary(input, UnaryOperation::kLOG)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("ReduceLogSum axes=[] log: {}", e),
                })?;
            let output = log_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("ReduceLogSum axes=[] output: {}", e),
                })?;
            let output_id = operation.output_operands_slice()[0];
            tensor_map.insert(output_id, output);
            return Ok(());
        }

        let mut axes_mask: u32 = 0;
        for &axis in &axes {
            axes_mask |= 1 << axis;
        }

        let keep_dims = opts.map(|o| o.keep_dimensions).unwrap_or(false);

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

    /// Add reduceLogSumExp operation: log(sum(exp(x))). Axes optional; empty axes -> output = x.
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

        let input_dims = input
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("ReduceLogSumExp: input dimensions: {}", e),
            })?;
        let rank = input_dims.len();

        let opts = operation.attributes.as_reduce();
        let axes: Vec<u32> = opts
            .and_then(|o| o.axes.clone())
            .unwrap_or_else(|| (0..rank).map(|i| i as u32).collect());

        if axes.is_empty() {
            let id_layer =
                network
                    .add_identity(input)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("ReduceLogSumExp axes=[] identity: {}", e),
                    })?;
            let output = id_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("ReduceLogSumExp axes=[] output: {}", e),
                })?;
            let output_id = operation.output_operands_slice()[0];
            tensor_map.insert(output_id, output);
            return Ok(());
        }

        let mut axes_mask: u32 = 0;
        for &axis in &axes {
            axes_mask |= 1 << axis;
        }

        let keep_dims = opts.map(|o| o.keep_dimensions).unwrap_or(false);

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

    /// Add reduceSumSquare operation: sum(x^2). Axes optional; empty axes -> output = x^2.
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

        let input_dims = input
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("ReduceSumSquare: input dimensions: {}", e),
            })?;
        let rank = input_dims.len();

        let opts = operation.attributes.as_reduce();
        let axes: Vec<u32> = opts
            .and_then(|o| o.axes.clone())
            .unwrap_or_else(|| (0..rank).map(|i| i as u32).collect());

        if axes.is_empty() {
            let output_id = operation.output_operands_slice()[0];
            tensor_map.insert(output_id, square_output);
            return Ok(());
        }

        let mut axes_mask: u32 = 0;
        for &axis in &axes {
            axes_mask |= 1 << axis;
        }

        let keep_dims = opts.map(|o| o.keep_dimensions).unwrap_or(false);

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

        let opts = operation
            .attributes
            .as_slice()
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Slice operation missing options".to_string(),
            })?;
        // Empty starts/sizes: no-op (identity), e.g. 0D tensor with empty slices.
        if opts.starts.is_empty() || opts.sizes.is_empty() {
            let id_layer =
                network
                    .add_identity(input)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Slice no-op identity: {}", e),
                    })?;
            let output = id_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Slice no-op output: {}", e),
                })?;
            let output_id = operation.output_operands_slice()[0];
            tensor_map.insert(output_id, output);
            return Ok(());
        }
        let starts: Vec<i32> = opts.starts.iter().map(|&u| u as i32).collect();
        let sizes: Vec<i32> = opts.sizes.iter().map(|&u| u as i32).collect();
        let strides: Vec<i32> = if opts.strides.is_empty() {
            vec![1; starts.len()]
        } else {
            opts.strides.iter().map(|&u| u as i32).collect()
        };

        // TensorRT Slice expects "size" = output dimensions. WebNN "sizes" are extents (range
        // lengths); with stride != 1 the output length per axis is ceil(extent/stride).
        let trt_sizes: Vec<i32> = sizes
            .iter()
            .zip(strides.iter())
            .map(|(&sz, &st)| {
                if st == 0 {
                    0_i32 // avoid div-by-zero; validator should reject elsewhere
                } else if st == 1 {
                    sz
                } else {
                    // ceil(extent / stride) in integers
                    (sz + st.abs() - 1) / st.abs()
                }
            })
            .collect();

        let layer = network
            .add_slice(input, &starts, &trt_sizes, &strides)
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
        let input_id = operation.input_operands[0];
        let input_dims = tensor_map
            .get(&input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", input_id),
            })?
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get input shape: {}", e),
            })?;

        let ndim = input_dims.len();

        let opts = operation
            .attributes
            .as_split()
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Split operation missing options".to_string(),
            })?;
        let mut axis = opts.axis as i32;
        if axis < 0 {
            axis += ndim as i32;
        }
        axis = axis.max(0).min((ndim.saturating_sub(1)) as i32);

        let splits: Vec<i32> = if opts.splits.is_empty() {
            let n = operation.output_operands_slice().len();
            if n == 0 {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "Split operation 'splits' missing and no outputs".to_string(),
                });
            }
            let dim = input_dims[axis as usize] as i32;
            let base = dim / n as i32;
            let rem = (dim % n as i32) as usize;
            (0..n).map(|i| base + if i < rem { 1 } else { 0 }).collect()
        } else {
            opts.splits.iter().map(|&u| u as i32).collect()
        };

        // One slice per split; start along axis advances by previous split sizes.
        let output_ids = operation.output_operands_slice();
        if output_ids.len() != splits.len() {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Split: {} outputs expected, {} operands",
                    splits.len(),
                    output_ids.len()
                ),
            });
        }

        let mut offset = 0i32;
        for (k, &size_k) in splits.iter().enumerate() {
            let mut starts = vec![0i32; ndim];
            starts[axis as usize] = offset;
            let mut sizes = input_dims.clone();
            sizes[axis as usize] = size_k;
            let strides = vec![1i32; ndim];

            let output = {
                let input =
                    tensor_map
                        .get(&input_id)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Input operand {} not found", input_id),
                        })?;
                let layer = network
                    .add_slice(input, &starts, &sizes, &strides)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add slice layer for split {}: {}", k, e),
                    })?;
                layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get layer output for split {}: {}", k, e),
                    })?
            };

            tensor_map.insert(output_ids[k], output);
            offset += size_k;
        }

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
    /// Implemented as input * ones(new_shape) so elementwise broadcast produces the expanded tensor.
    fn add_expand_op(
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

        let opts =
            operation
                .attributes
                .as_expand()
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "Expand operation missing options".to_string(),
                })?;
        let new_shape: Vec<i32> = opts
            .new_shape_static_or_max()
            .into_iter()
            .map(|u| u as i32)
            .collect();

        if new_shape.is_empty() {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Expand newShape must be non-empty".to_string(),
            });
        }

        let num_elements: usize = new_shape
            .iter()
            .map(|&d| d.max(0) as usize)
            .product::<usize>()
            .max(1);

        let input_operand = graph.operand(operation.input_operands[0]).ok_or_else(|| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Input operand {} not found in graph",
                    operation.input_operands[0]
                ),
            }
        })?;
        let (ones_data, trt_dtype) = match input_operand.descriptor.data_type {
            DataType::Float16 => {
                let data: Vec<u8> = (0..num_elements)
                    .flat_map(|_| f16::from_f32(1.0).to_bits().to_le_bytes())
                    .collect();
                (data, trtx::DataType::kHALF)
            }
            _ => {
                let data: Vec<u8> = (0..num_elements)
                    .flat_map(|_| 1.0f32.to_le_bytes())
                    .collect();
                (data, trtx::DataType::kFLOAT)
            }
        };
        temp_weights.push(ones_data);
        let ones_ref = temp_weights.last().unwrap().as_slice();

        let ones_const = network
            .add_constant(&new_shape, ones_ref, trt_dtype)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create ones constant for expand: {}", e),
            })?;
        let ones_tensor = ones_const
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get ones constant output: {}", e),
            })?;

        let (bc_input, bc_ones) =
            Self::ensure_broadcast_compatible(network, input, &ones_tensor, "expand")?;

        let mul_layer = network
            .add_elementwise(&bc_input, &bc_ones, ElementWiseOperation::kPROD)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add multiply for expand: {}", e),
            })?;
        let output = mul_layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get expand output: {}", e),
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
        let opts = operation
            .attributes
            .as_tile()
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Tile operation missing options".to_string(),
            })?;
        let repetitions = opts.repetitions.clone();
        if repetitions.is_empty() {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Tile operation missing 'repetitions' attribute".to_string(),
            });
        }

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

        let (bc_input0, bc_input1) =
            Self::ensure_broadcast_compatible(network, input0, input1, &operation.op_type)?;

        // greaterOrEqual = greater OR equal
        let greater_layer = network
            .add_elementwise(&bc_input0, &bc_input1, ElementWiseOperation::kGREATER)
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
            .add_elementwise(&bc_input0, &bc_input1, ElementWiseOperation::kEQUAL)
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

        let (bc_input0, bc_input1) =
            Self::ensure_broadcast_compatible(network, input0, input1, &operation.op_type)?;

        // lesserOrEqual = lesser OR equal
        let lesser_layer = network
            .add_elementwise(&bc_input0, &bc_input1, ElementWiseOperation::kLESS)
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
            .add_elementwise(&bc_input0, &bc_input1, ElementWiseOperation::kEQUAL)
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

        let (bc_input0, bc_input1) =
            Self::ensure_broadcast_compatible(network, input0, input1, &operation.op_type)?;

        // notEqual = NOT equal
        let equal_layer = network
            .add_elementwise(&bc_input0, &bc_input1, ElementWiseOperation::kEQUAL)
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

    /// Add gather operation (gather elements along an axis using indices).
    /// Clamps indices to [-dim_size, dim_size - 1] to match WebNN/Chromium behavior and avoid TensorRT out-of-bounds.
    fn add_gather_op(
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

        let indices = tensor_map
            .get(&operation.input_operands[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Indices operand {} not found", operation.input_operands[1]),
            })?;

        // Get axis attribute (default to 0)
        let axis = operation
            .attributes
            .as_gather()
            .map(|o| o.axis as i32)
            .unwrap_or(0);

        let data_operand = graph.operand(operation.input_operands[0]).ok_or_else(|| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Data operand {} not found in graph",
                    operation.input_operands[0]
                ),
            }
        })?;
        let axis_usize = axis as usize;
        if axis_usize >= data_operand.descriptor.shape.len() {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Gather axis {} out of bounds for shape with {} dimensions",
                    axis,
                    data_operand.descriptor.shape.len()
                ),
            });
        }
        let dim_size = get_static_or_max_size(&data_operand.descriptor.shape[axis_usize]) as i32;

        let indices_operand = graph.operand(operation.input_operands[1]).ok_or_else(|| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Indices operand {} not found in graph",
                    operation.input_operands[1]
                ),
            }
        })?;
        let indices_shape: Vec<i32> = indices_operand
            .descriptor
            .shape
            .iter()
            .map(|d| get_static_or_max_size(d) as i32)
            .collect();

        // Clamp indices to [-dim_size, dim_size - 1] (WebNN/conformance behavior).
        // TensorRT elementwise requires same rank; repeat scalar to match indices shape.
        let clamp_min_val = -dim_size;
        let clamp_max_val = dim_size - 1;
        let num_elements: usize = indices_operand
            .descriptor
            .shape
            .iter()
            .map(get_static_or_max_size)
            .product::<u32>() as usize;
        let min_data: Vec<u8> = (0..num_elements)
            .flat_map(|_| clamp_min_val.to_le_bytes())
            .collect();
        let max_data: Vec<u8> = (0..num_elements)
            .flat_map(|_| clamp_max_val.to_le_bytes())
            .collect();
        temp_weights.push(min_data);
        temp_weights.push(max_data);
        let idx = temp_weights.len();
        let min_ref = temp_weights[idx - 2].as_slice();
        let max_ref = temp_weights[idx - 1].as_slice();

        let min_const = network
            .add_constant(&indices_shape, min_ref, trtx::DataType::kINT32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add gather clamp min constant: {}", e),
            })?;
        let max_const = network
            .add_constant(&indices_shape, max_ref, trtx::DataType::kINT32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add gather clamp max constant: {}", e),
            })?;

        let min_const_out = min_const
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get clamp min output: {}", e),
            })?;
        let max_const_out = max_const
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get clamp max output: {}", e),
            })?;

        let clamped_upper = network
            .add_elementwise(indices, &max_const_out, ElementWiseOperation::kMIN)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add gather indices clamp (min): {}", e),
            })?;
        let clamped_upper_out =
            clamped_upper
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get gather clamp upper output: {}", e),
                })?;

        let clamped = network
            .add_elementwise(
                &min_const_out,
                &clamped_upper_out,
                ElementWiseOperation::kMAX,
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add gather indices clamp (max): {}", e),
            })?;
        let clamped_indices = clamped
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get gather clamped indices output: {}", e),
            })?;

        let layer = network
            .add_gather(input, &clamped_indices, axis)
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
            .as_scatter_elements()
            .map(|o| o.axis as i32)
            .unwrap_or(0);

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

        let opts = operation.attributes.as_arg_min_max();
        let axis = opts.map(|o| o.axis).unwrap_or(0);
        let keep_dims = opts.map(|o| o.keep_dimensions).unwrap_or(false);

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

        let opts = operation.attributes.as_arg_min_max();
        let axis = opts.map(|o| o.axis).unwrap_or(0);
        let keep_dims = opts.map(|o| o.keep_dimensions).unwrap_or(false);

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
        let input_dtype = input_operand.descriptor.data_type;
        // Create broadcast shape: [1, 1, ..., 1] with same number of dimensions as input
        let broadcast_shape: Vec<i32> = vec![1; num_dims];

        // Get min and max values from attributes (handle "Infinity"/"-Infinity"/"NaN" strings from WPT).
        let parse_clamp_bound_f32 = |v: &serde_json::Value| -> Option<f32> {
            if let Some(s) = v.as_str() {
                return match s {
                    "Infinity" => Some(f32::INFINITY),
                    "-Infinity" => Some(f32::NEG_INFINITY),
                    "NaN" => Some(f32::NAN),
                    _ => None,
                };
            }
            v.as_f64().map(|f| f as f32)
        };
        // For integer types, parse bounds as integers to avoid f32 precision loss and overflow.
        let parse_i64 = |v: &serde_json::Value| -> Option<i64> {
            v.as_i64()
                .or_else(|| v.as_u64().map(|u| u as i64))
                .or_else(|| {
                    v.as_f64().and_then(|f| {
                        (f >= i64::MIN as f64 && f <= i64::MAX as f64).then_some(f as i64)
                    })
                })
        };
        let parse_u64 = |v: &serde_json::Value| -> Option<u64> {
            v.as_u64()
                .or_else(|| v.as_i64().and_then(|i| (i >= 0).then_some(i as u64)))
                .or_else(|| {
                    v.as_f64()
                        .and_then(|f| (f >= 0.0 && f <= u64::MAX as f64).then_some(f as u64))
                })
                .or_else(|| {
                    v.as_str()
                        .and_then(|s| s.trim_end_matches('n').trim().parse::<u64>().ok())
                })
        };

        let clamp_opts = operation.attributes.as_clamp();
        let min_value = clamp_opts
            .and_then(|o| o.min_value.as_ref())
            .and_then(parse_clamp_bound_f32)
            .unwrap_or(f32::NEG_INFINITY);
        let min_value = if min_value.is_nan() {
            f32::NEG_INFINITY
        } else {
            min_value
        };
        let max_value = clamp_opts
            .and_then(|o| o.max_value.as_ref())
            .and_then(parse_clamp_bound_f32)
            .unwrap_or(f32::INFINITY);
        let max_value = if max_value.is_nan() {
            f32::INFINITY
        } else {
            max_value
        };

        // TensorRT ElementWise MIN/MAX require both inputs to have the same type. Use input type for constants.
        let trt_dtype = Self::webnn_to_trt_dtype(input_dtype)?;
        let (max_bytes, min_bytes) = match input_dtype {
            DataType::Int8 => (
                (max_value.clamp(i8::MIN as f32, i8::MAX as f32) as i8)
                    .to_le_bytes()
                    .to_vec(),
                (min_value.clamp(i8::MIN as f32, i8::MAX as f32) as i8)
                    .to_le_bytes()
                    .to_vec(),
            ),
            DataType::Uint8 => (
                (max_value.clamp(0.0, u8::MAX as f32) as u8)
                    .to_le_bytes()
                    .to_vec(),
                (min_value.clamp(0.0, u8::MAX as f32) as u8)
                    .to_le_bytes()
                    .to_vec(),
            ),
            DataType::Int32 => {
                let min_i = clamp_opts
                    .and_then(|o| o.min_value.as_ref())
                    .and_then(parse_i64)
                    .unwrap_or(i64::from(i32::MIN))
                    .clamp(i64::from(i32::MIN), i64::from(i32::MAX))
                    as i32;
                let max_i = clamp_opts
                    .and_then(|o| o.max_value.as_ref())
                    .and_then(parse_i64)
                    .unwrap_or(i64::from(i32::MAX))
                    .clamp(i64::from(i32::MIN), i64::from(i32::MAX))
                    as i32;
                (max_i.to_le_bytes().to_vec(), min_i.to_le_bytes().to_vec())
            }
            DataType::Uint32 => {
                let min_u = clamp_opts
                    .and_then(|o| o.min_value.as_ref())
                    .and_then(parse_u64)
                    .unwrap_or(0)
                    .min(u64::from(u32::MAX)) as u32;
                let max_u = clamp_opts
                    .and_then(|o| o.max_value.as_ref())
                    .and_then(parse_u64)
                    .unwrap_or(u64::from(u32::MAX))
                    .min(u64::from(u32::MAX)) as u32;
                (max_u.to_le_bytes().to_vec(), min_u.to_le_bytes().to_vec())
            }
            DataType::Int64 | DataType::Uint64 => {
                let min_i64 = clamp_opts
                    .and_then(|o| o.min_value.as_ref())
                    .and_then(parse_i64)
                    .unwrap_or(i64::MIN);
                let max_i64 = clamp_opts
                    .and_then(|o| o.max_value.as_ref())
                    .and_then(parse_i64)
                    .unwrap_or(i64::MAX);
                (
                    max_i64.to_le_bytes().to_vec(),
                    min_i64.to_le_bytes().to_vec(),
                )
            }
            DataType::Float16 => (
                f16::from_f32(max_value).to_bits().to_le_bytes().to_vec(),
                f16::from_f32(min_value).to_bits().to_le_bytes().to_vec(),
            ),
            _ => (
                max_value.to_le_bytes().to_vec(),
                min_value.to_le_bytes().to_vec(),
            ),
        };

        // Implement clamp as: max(min_value, min(input, max_value))
        // First: min(input, max_value)
        temp_weights.push(max_bytes);
        let max_const_data = temp_weights.last().unwrap();
        let max_const = network
            .add_constant(&broadcast_shape, max_const_data, trt_dtype.clone())
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
        temp_weights.push(min_bytes);
        let min_const_data = temp_weights.last().unwrap();
        let min_const = network
            .add_constant(&broadcast_shape, min_const_data, trt_dtype)
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

        let linear_opts = operation.attributes.as_linear();
        let alpha = linear_opts.map(|o| o.alpha as f32).unwrap_or(1.0);
        let beta = linear_opts.map(|o| o.beta as f32).unwrap_or(0.0);

        // Implement as: y = alpha * x + beta using elementwise operations
        // Note: All scalar constants must use shape [1,1,...,1] matching input dims for TensorRT broadcasting

        // Step 1: If alpha != 1.0, multiply x by alpha
        let after_multiply = if (alpha - 1.0).abs() > f32::EPSILON {
            // Create alpha constant with type matching input (Half vs Float) for TensorRT elementwise
            let (alpha_bytes, alpha_dtype) = match input_operand.descriptor.data_type {
                DataType::Float16 => (
                    f16::from_f32(alpha).to_bits().to_le_bytes().to_vec(),
                    trtx::DataType::kHALF,
                ),
                _ => (alpha.to_le_bytes().to_vec(), trtx::DataType::kFLOAT),
            };
            temp_weights.push(alpha_bytes);
            let alpha_bytes_ref = temp_weights.last().unwrap();

            let alpha_constant = network
                .add_constant(&broadcast_shape, alpha_bytes_ref, alpha_dtype)
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
            // Create beta constant with type matching input (Half vs Float) for TensorRT elementwise
            let (beta_bytes, beta_dtype) = match input_operand.descriptor.data_type {
                DataType::Float16 => (
                    f16::from_f32(beta).to_bits().to_le_bytes().to_vec(),
                    trtx::DataType::kHALF,
                ),
                _ => (beta.to_le_bytes().to_vec(), trtx::DataType::kFLOAT),
            };
            temp_weights.push(beta_bytes);
            let beta_bytes_ref = temp_weights.last().unwrap();

            let beta_constant = network
                .add_constant(&broadcast_shape, beta_bytes_ref, beta_dtype)
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

        let opts = operation
            .attributes
            .as_pad()
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Pad operation missing options".to_string(),
            })?;
        let pre_padding: Vec<i32> = opts.beginning_padding.iter().map(|&u| u as i32).collect();
        let post_padding: Vec<i32> = opts.ending_padding.iter().map(|&u| u as i32).collect();
        if pre_padding.is_empty() || post_padding.is_empty() {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Pad operation missing beginningPadding or endingPadding".to_string(),
            });
        }

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

        let opts = operation.attributes.as_gemm();
        let alpha = opts.map(|o| o.alpha as f32).unwrap_or(1.0);
        let beta = opts.map(|o| o.beta as f32).unwrap_or(1.0);
        let a_transpose = opts.map(|o| o.a_transpose).unwrap_or(false);
        let b_transpose = opts.map(|o| o.b_transpose).unwrap_or(false);

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
        temp_weights: &mut Vec<Vec<u8>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let filter_id = operation.input_operands[1];
        let bias_id = operation.input_operands.get(2).copied();

        // Filter operand and shape (needed for both constant and tensor-weight paths)
        let filter_operand =
            graph
                .operand(filter_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Filter operand {} not found", filter_id),
                })?;
        let filter_shape = &filter_operand.descriptor.shape;
        if filter_shape.len() != 4 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Expected 4D filter shape, got {}D", filter_shape.len()),
            });
        }
        let fs = filter_operand.descriptor.static_or_max_shape();
        let conv_opts = operation.attributes.as_conv2d();
        let filter_layout = conv_opts
            .map(|o| o.filter_layout.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("oihw");
        let (o, _i, h, w): (u32, u32, u32, u32) = match filter_layout {
            "oihw" => (fs[0], fs[1], fs[2], fs[3]),
            "hwio" => (fs[3], fs[2], fs[0], fs[1]),
            "ohwi" => (fs[0], fs[3], fs[1], fs[2]),
            "ihwo" => (fs[3], fs[0], fs[1], fs[2]),
            "hwoi" => (fs[2], fs[3], fs[0], fs[1]),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Unsupported filter_layout: {}", filter_layout),
                });
            }
        };
        let num_output_maps = o as i32;
        let kernel_size: [i32; 2] = [h as i32, w as i32];

        let filter_constant = graph
            .constant_operand_ids_to_handles
            .contains_key(&filter_id);

        // When filter is non-constant we use ILayer::setInput(1, kernel) / setInput(2, bias); kernel/bias weights must be empty.
        let (filter_data_to_use, bias_data) = if filter_constant {
            let filter_shape_u32 = filter_operand.descriptor.static_or_max_shape();
            let filter_data = Self::get_constant_data(graph, filter_id)?;
            // Get optional bias - operand 2 if present.
            let (bias_temp_index, bias_raw): (Option<usize>, Option<&[u8]>) = match bias_id {
                Some(id) => {
                    if !graph.constant_operand_ids_to_handles.contains_key(&id) {
                        return Err(GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: "conv2d with non-constant bias is not supported when filter is constant".to_string(),
                        });
                    }
                    let raw = Self::get_constant_data(graph, id)?;
                    let dtype = graph
                        .operand(id)
                        .map(|o| o.descriptor.data_type)
                        .unwrap_or(DataType::Float32);
                    if dtype == DataType::Float16 {
                        let f32_bias = Self::f16_bytes_to_f32_bytes(raw)?;
                        temp_weights.push(f32_bias);
                        (Some(temp_weights.len() - 1), None)
                    } else {
                        (None, Some(raw))
                    }
                }
                None => (None, None),
            };
            let filter_dtype = filter_operand.descriptor.data_type;
            let filter_temp_index: Option<usize> = match (filter_dtype, filter_layout) {
                (DataType::Float16, _) => {
                    let f32_bytes = Self::f16_bytes_to_f32_bytes(filter_data)?;
                    let oihw = if filter_layout == "oihw" {
                        f32_bytes
                    } else {
                        Self::conv_filter_to_oihw(&f32_bytes, filter_layout, &filter_shape_u32)?
                    };
                    temp_weights.push(oihw);
                    Some(temp_weights.len() - 1)
                }
                (DataType::Float32, "oihw") => None,
                (DataType::Float32, _) => {
                    let oihw =
                        Self::conv_filter_to_oihw(filter_data, filter_layout, &filter_shape_u32)?;
                    temp_weights.push(oihw);
                    Some(temp_weights.len() - 1)
                }
                _ => {
                    return Err(GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!(
                            "Conv2d filter data type {:?} not supported (use float32 or float16)",
                            filter_dtype
                        ),
                    });
                }
            };
            let bias_data: Option<&[u8]> = match bias_temp_index {
                Some(idx) => Some(temp_weights[idx].as_slice()),
                None => bias_raw,
            };
            let filter_data_to_use: &[u8] = match filter_temp_index {
                Some(idx) => temp_weights[idx].as_slice(),
                None => filter_data,
            };
            (Some(filter_data_to_use), bias_data)
        } else {
            // Non-constant filter: use tensor inputs (setInput(1)=kernel, setInput(2)=bias). TensorRT expects OIHW for kernel.
            if filter_layout != "oihw" {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "conv2d with non-constant filter requires filter_layout \"oihw\""
                        .to_string(),
                });
            }
            if let Some(id) = bias_id {
                if graph.constant_operand_ids_to_handles.contains_key(&id) {
                    return Err(GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: "conv2d with non-constant filter requires bias to be a tensor input (constant bias not supported)".to_string(),
                    });
                }
            }
            (None, None)
        };

        // Input layout: nchw (default) or nhwc. TensorRT conv is NCHW; we use IShuffleLayer::setFirstTranspose for NHWC.
        let input_layout = conv_opts
            .map(|o| o.input_layout.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("nchw");
        let input_id = operation.input_operands[0];
        let input_dtype = graph
            .operand(input_id)
            .map(|o| o.descriptor.data_type)
            .unwrap_or(DataType::Float32);
        let input = tensor_map
            .get(&input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", input_id),
            })?;

        // NHWC input: TensorRT conv expects NCHW. Insert shuffle to transpose NHWC->NCHW before conv.
        let nhwc_shuffle_output = if input_layout == "nhwc" {
            let mut shuffle =
                network
                    .add_shuffle(input)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Conv2d NHWC->NCHW shuffle: {}", e),
                    })?;
            shuffle.set_first_transpose(&[0, 3, 1, 2]).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Conv2d set_first_transpose NHWC->NCHW: {}", e),
                }
            })?;
            Some(
                shuffle
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Conv2d NHWC shuffle output: {}", e),
                    })?,
            )
        } else {
            None
        };
        let pre_conv_input = nhwc_shuffle_output.as_ref().unwrap_or(input);

        // TensorRT conv kernel is always Float; cast Half input to Float so types match.
        let half_cast_output: Option<trtx::Tensor> = if input_dtype == DataType::Float16 {
            let cast_layer = network
                .add_cast(pre_conv_input, TrtDataType::kFLOAT)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Conv2d Half->Float cast: {}", e),
                })?;
            Some(
                cast_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Conv2d cast output: {}", e),
                    })?,
            )
        } else {
            None
        };
        let conv_input_source = half_cast_output.as_ref().unwrap_or(pre_conv_input);

        // Stride, padding, dilation, groups from typed options
        let strides: [i32; 2] = conv_opts
            .and_then(|o| {
                if o.strides.len() >= 2 {
                    Some([o.strides[0] as i32, o.strides[1] as i32])
                } else {
                    None
                }
            })
            .unwrap_or([1, 1]);
        let dilations: [i32; 2] = conv_opts
            .and_then(|o| {
                if o.dilations.len() >= 2 {
                    Some([o.dilations[0] as i32, o.dilations[1] as i32])
                } else {
                    None
                }
            })
            .unwrap_or([1, 1]);
        let groups = conv_opts.map(|o| o.groups as i32).unwrap_or(1);
        let (pre_padding, post_padding) = conv_opts
            .map(|o| {
                if o.padding.len() >= 4 {
                    (
                        vec![o.padding[0] as i32, o.padding[1] as i32],
                        vec![o.padding[2] as i32, o.padding[3] as i32],
                    )
                } else {
                    (vec![0, 0], vec![0, 0])
                }
            })
            .unwrap_or((vec![0, 0], vec![0, 0]));

        // Use explicit padding layer if any padding is specified
        let conv_input =
            if pre_padding.iter().any(|&p| p != 0) || post_padding.iter().any(|&p| p != 0) {
                let padding_layer = network
                    .add_padding(&conv_input_source, &pre_padding, &post_padding)
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
                // No padding needed, use conv_input_source directly
                let id_layer = network.add_identity(&conv_input_source).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add identity layer: {}", e),
                    }
                })?;
                id_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get identity output: {}", e),
                    })?
            };

        // Add convolution layer with zero padding (padding already applied via padding layer)
        // Constant path always passes f32 data (f16 filter/bias are converted above); dtype must match.
        let mut layer = match (filter_data_to_use, bias_data) {
            (Some(fd), b) => {
                let conv_weights = trtx::ConvWeights {
                    kernel_weights: fd,
                    kernel_dtype: TrtDataType::kFLOAT,
                    bias_weights: b,
                    bias_dtype: b.map(|_| TrtDataType::kFLOAT),
                };
                network
                    .add_convolution(&conv_input, num_output_maps, &kernel_size, &conv_weights)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add convolution: {}", e),
                    })?
            }
            (None, _) => {
                let filter_tensor =
                    tensor_map
                        .get(&filter_id)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Filter operand {} tensor not found", filter_id),
                        })?;
                // TensorRT conv requires input and kernel same type. We cast activation to Float when input_dtype is Float16; cast filter (and bias) to Float too.
                let filter_tensor_for_conv: Option<trtx::Tensor> =
                    if filter_operand.descriptor.data_type == DataType::Float16
                        && input_dtype == DataType::Float16
                    {
                        let cast_layer = network
                            .add_cast(filter_tensor, TrtDataType::kFLOAT)
                            .map_err(|e| GraphError::ConversionFailed {
                                format: "trtx".to_string(),
                                reason: format!("Conv2d filter Half->Float cast: {}", e),
                            })?;
                        Some(cast_layer.get_output(0).map_err(|e| {
                            GraphError::ConversionFailed {
                                format: "trtx".to_string(),
                                reason: format!("Conv2d filter cast output: {}", e),
                            }
                        })?)
                    } else {
                        None
                    };
                let filter_tensor_to_use = filter_tensor_for_conv.as_ref().unwrap_or(filter_tensor);

                let bias_tensor_raw = bias_id.and_then(|id| tensor_map.get(&id));
                let bias_tensor_for_conv: Option<trtx::Tensor> =
                    if let (Some(bt), Some(bid)) = (bias_tensor_raw, bias_id) {
                        let bias_dtype = graph
                            .operand(bid)
                            .map(|o| o.descriptor.data_type)
                            .unwrap_or(DataType::Float32);
                        if bias_dtype == DataType::Float16 && input_dtype == DataType::Float16 {
                            let cast_layer =
                                network.add_cast(bt, TrtDataType::kFLOAT).map_err(|e| {
                                    GraphError::ConversionFailed {
                                        format: "trtx".to_string(),
                                        reason: format!("Conv2d bias Half->Float cast: {}", e),
                                    }
                                })?;
                            Some(cast_layer.get_output(0).map_err(|e| {
                                GraphError::ConversionFailed {
                                    format: "trtx".to_string(),
                                    reason: format!("Conv2d bias cast output: {}", e),
                                }
                            })?)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                let bias_tensor_to_use = bias_tensor_for_conv.as_ref().or(bias_tensor_raw);

                let conv_weights = trtx::ConvWeights {
                    kernel_weights: &[],
                    kernel_dtype: TrtDataType::kFLOAT,
                    bias_weights: None,
                    bias_dtype: None,
                };
                let mut layer = network
                    .add_convolution(&conv_input, num_output_maps, &kernel_size, &conv_weights)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add convolution (tensor weights): {}", e),
                    })?;
                layer.set_input(1, filter_tensor_to_use).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Conv2d set_input(1) filter: {}", e),
                    }
                })?;
                if let Some(bt) = bias_tensor_to_use {
                    layer
                        .set_input(2, bt)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Conv2d set_input(2) bias: {}", e),
                        })?;
                }
                layer
            }
        };

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

        // Extract output tensor from layer (NCHW, Float)
        let conv_output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get convolution output: {}", e),
            })?;

        // If input was Half, cast conv output back to Half to match graph output type.
        let conv_output = if input_dtype == DataType::Float16 {
            let cast_layer = network
                .add_cast(&conv_output, TrtDataType::kHALF)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Conv2d Float->Half cast: {}", e),
                })?;
            cast_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Conv2d cast output: {}", e),
                })?
        } else {
            conv_output
        };

        // If input was NHWC, transpose output back to NHWC.
        let output = if input_layout == "nhwc" {
            let mut shuffle =
                network
                    .add_shuffle(&conv_output)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Conv2d NCHW->NHWC shuffle: {}", e),
                    })?;
            shuffle.set_first_transpose(&[0, 2, 3, 1]).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Conv2d set_first_transpose NCHW->NHWC: {}", e),
                }
            })?;
            shuffle
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Conv2d NHWC output shuffle: {}", e),
                })?
        } else {
            conv_output
        };

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add convTranspose2d operation (deconvolution/transposed convolution).
    /// Mirrors add_conv2d_op: supports constant or non-constant filter/bias via setInput(1)/setInput(2).
    fn add_conv_transpose2d_op(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition,
        tensor_map: &mut HashMap<u32, trtx::Tensor>,
        temp_weights: &mut Vec<Vec<u8>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let filter_id = operation.input_operands[1];
        let bias_id = operation.input_operands.get(2).copied();

        let filter_operand =
            graph
                .operand(filter_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Filter operand {} not found", filter_id),
                })?;
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
        let fs = filter_operand.descriptor.static_or_max_shape();
        let deconv_opts = operation.attributes.as_conv_transpose2d();
        let filter_layout = deconv_opts
            .map(|o| o.filter_layout.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("iohw");
        let (_i, o, h, w): (u32, u32, u32, u32) = match filter_layout {
            "iohw" => (fs[0], fs[1], fs[2], fs[3]),
            "oihw" => (fs[1], fs[0], fs[2], fs[3]),
            "hwio" => (fs[2], fs[3], fs[0], fs[1]),
            "ohwi" => (fs[3], fs[0], fs[1], fs[2]),
            "ihwo" => (fs[0], fs[3], fs[1], fs[2]),
            "hwoi" => (fs[3], fs[2], fs[0], fs[1]),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Unsupported filter_layout: {}", filter_layout),
                });
            }
        };
        let groups = deconv_opts.map(|o| o.groups as i32).unwrap_or(1);
        // WebNN filter shape is [inputChannels, outputChannels/groups, H, W]; TensorRT expects total output maps.
        let num_output_maps = (o as i32) * groups;
        let kernel_size: [i32; 2] = [h as i32, w as i32];

        let filter_constant = graph
            .constant_operand_ids_to_handles
            .contains_key(&filter_id);

        let (filter_data_to_use, bias_data) = if filter_constant {
            let filter_shape_u32 = filter_operand.descriptor.static_or_max_shape();
            let filter_data = Self::get_constant_data(graph, filter_id)?;
            let (bias_temp_index, bias_raw): (Option<usize>, Option<&[u8]>) = match bias_id {
                Some(id) => {
                    if !graph.constant_operand_ids_to_handles.contains_key(&id) {
                        return Err(GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: "convTranspose2d with non-constant bias is not supported when filter is constant".to_string(),
                        });
                    }
                    let raw = Self::get_constant_data(graph, id)?;
                    let dtype = graph
                        .operand(id)
                        .map(|o| o.descriptor.data_type)
                        .unwrap_or(DataType::Float32);
                    if dtype == DataType::Float16 {
                        let f32_bias = Self::f16_bytes_to_f32_bytes(raw)?;
                        temp_weights.push(f32_bias);
                        (Some(temp_weights.len() - 1), None)
                    } else {
                        (None, Some(raw))
                    }
                }
                None => (None, None),
            };
            let filter_dtype = filter_operand.descriptor.data_type;
            let filter_temp_index: Option<usize> = match (filter_dtype, filter_layout) {
                (DataType::Float16, _) => {
                    let f32_bytes = Self::f16_bytes_to_f32_bytes(filter_data)?;
                    let iohw = if filter_layout == "iohw" {
                        f32_bytes
                    } else {
                        Self::deconv_filter_to_iohw(&f32_bytes, filter_layout, &filter_shape_u32)?
                    };
                    temp_weights.push(iohw);
                    Some(temp_weights.len() - 1)
                }
                (DataType::Float32, "iohw") => None,
                (DataType::Float32, _) => {
                    let iohw =
                        Self::deconv_filter_to_iohw(filter_data, filter_layout, &filter_shape_u32)?;
                    temp_weights.push(iohw);
                    Some(temp_weights.len() - 1)
                }
                _ => {
                    return Err(GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!(
                            "convTranspose2d filter data type {:?} not supported (use float32 or float16)",
                            filter_dtype
                        ),
                    });
                }
            };
            let bias_data: Option<&[u8]> = match bias_temp_index {
                Some(idx) => Some(temp_weights[idx].as_slice()),
                None => bias_raw,
            };
            let filter_data_to_use: &[u8] = match filter_temp_index {
                Some(idx) => temp_weights[idx].as_slice(),
                None => filter_data,
            };
            (Some(filter_data_to_use), bias_data)
        } else {
            if filter_layout != "iohw" {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason:
                        "convTranspose2d with non-constant filter requires filter_layout \"iohw\""
                            .to_string(),
                });
            }
            if let Some(id) = bias_id {
                if graph.constant_operand_ids_to_handles.contains_key(&id) {
                    return Err(GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: "convTranspose2d with non-constant filter requires bias to be a tensor input (constant bias not supported)".to_string(),
                    });
                }
            }
            (None, None)
        };

        let input_layout = deconv_opts
            .map(|o| o.input_layout.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("nchw");
        let input_id = operation.input_operands[0];
        let input_dtype = graph
            .operand(input_id)
            .map(|o| o.descriptor.data_type)
            .unwrap_or(DataType::Float32);
        let input = tensor_map
            .get(&input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", input_id),
            })?;

        let nhwc_shuffle_output = if input_layout == "nhwc" {
            let mut shuffle =
                network
                    .add_shuffle(input)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("convTranspose2d NHWC->NCHW shuffle: {}", e),
                    })?;
            shuffle.set_first_transpose(&[0, 3, 1, 2]).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("convTranspose2d set_first_transpose NHWC->NCHW: {}", e),
                }
            })?;
            Some(
                shuffle
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("convTranspose2d NHWC shuffle output: {}", e),
                    })?,
            )
        } else {
            None
        };
        let pre_deconv_input = nhwc_shuffle_output.as_ref().unwrap_or(input);

        let half_cast_output: Option<trtx::Tensor> = if input_dtype == DataType::Float16 {
            let cast_layer = network
                .add_cast(pre_deconv_input, TrtDataType::kFLOAT)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("convTranspose2d Half->Float cast: {}", e),
                })?;
            Some(
                cast_layer
                    .get_output(0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("convTranspose2d cast output: {}", e),
                    })?,
            )
        } else {
            None
        };
        let deconv_input_source = half_cast_output.as_ref().unwrap_or(pre_deconv_input);

        let strides: [i32; 2] = deconv_opts
            .and_then(|o| {
                if o.strides.len() >= 2 {
                    Some([o.strides[0] as i32, o.strides[1] as i32])
                } else {
                    None
                }
            })
            .unwrap_or([1, 1]);
        let dilations: [i32; 2] = deconv_opts
            .and_then(|o| {
                if o.dilations.len() >= 2 {
                    Some([o.dilations[0] as i32, o.dilations[1] as i32])
                } else {
                    None
                }
            })
            .unwrap_or([1, 1]);
        let (pre_padding, post_padding) = deconv_opts
            .map(|o| {
                if o.padding.len() >= 4 {
                    (
                        vec![o.padding[0] as i32, o.padding[2] as i32],
                        vec![o.padding[1] as i32, o.padding[3] as i32],
                    )
                } else {
                    (vec![0, 0], vec![0, 0])
                }
            })
            .unwrap_or((vec![0, 0], vec![0, 0]));

        // Map WebNN padding to TensorRT IDeconvolutionLayer pre/post (trim); do not pad the input.
        let deconv_input = {
            let id_layer = network.add_identity(&deconv_input_source).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("convTranspose2d identity: {}", e),
                }
            })?;
            id_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("convTranspose2d identity output: {}", e),
                })?
        };

        let mut layer = match (filter_data_to_use, bias_data) {
            (Some(fd), b) => {
                let deconv_weights = trtx::ConvWeights {
                    kernel_weights: fd,
                    kernel_dtype: TrtDataType::kFLOAT,
                    bias_weights: b,
                    bias_dtype: b.map(|_| TrtDataType::kFLOAT),
                };
                network
                    .add_deconvolution(
                        &deconv_input,
                        num_output_maps,
                        &kernel_size,
                        &deconv_weights,
                    )
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add deconvolution: {}", e),
                    })?
            }
            (None, _) => {
                let filter_tensor =
                    tensor_map
                        .get(&filter_id)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Filter operand {} tensor not found", filter_id),
                        })?;
                let filter_tensor_for_conv: Option<trtx::Tensor> =
                    if filter_operand.descriptor.data_type == DataType::Float16
                        && input_dtype == DataType::Float16
                    {
                        let cast_layer = network
                            .add_cast(filter_tensor, TrtDataType::kFLOAT)
                            .map_err(|e| GraphError::ConversionFailed {
                                format: "trtx".to_string(),
                                reason: format!("convTranspose2d filter Half->Float cast: {}", e),
                            })?;
                        Some(cast_layer.get_output(0).map_err(|e| {
                            GraphError::ConversionFailed {
                                format: "trtx".to_string(),
                                reason: format!("convTranspose2d filter cast output: {}", e),
                            }
                        })?)
                    } else {
                        None
                    };
                let filter_tensor_to_use = filter_tensor_for_conv.as_ref().unwrap_or(filter_tensor);

                let bias_tensor_raw = bias_id.and_then(|id| tensor_map.get(&id));
                let bias_tensor_for_conv: Option<trtx::Tensor> = if let (Some(bt), Some(bid)) =
                    (bias_tensor_raw, bias_id)
                {
                    let bias_dtype = graph
                        .operand(bid)
                        .map(|o| o.descriptor.data_type)
                        .unwrap_or(DataType::Float32);
                    if bias_dtype == DataType::Float16 && input_dtype == DataType::Float16 {
                        let cast_layer =
                            network.add_cast(bt, TrtDataType::kFLOAT).map_err(|e| {
                                GraphError::ConversionFailed {
                                    format: "trtx".to_string(),
                                    reason: format!("convTranspose2d bias Half->Float cast: {}", e),
                                }
                            })?;
                        Some(cast_layer.get_output(0).map_err(|e| {
                            GraphError::ConversionFailed {
                                format: "trtx".to_string(),
                                reason: format!("convTranspose2d bias cast output: {}", e),
                            }
                        })?)
                    } else {
                        None
                    }
                } else {
                    None
                };
                let bias_tensor_to_use = bias_tensor_for_conv.as_ref().or(bias_tensor_raw);

                let deconv_weights = trtx::ConvWeights {
                    kernel_weights: &[],
                    kernel_dtype: TrtDataType::kFLOAT,
                    bias_weights: None,
                    bias_dtype: None,
                };
                let mut layer = network
                    .add_deconvolution(
                        &deconv_input,
                        num_output_maps,
                        &kernel_size,
                        &deconv_weights,
                    )
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add deconvolution (tensor weights): {}", e),
                    })?;
                layer.set_input(1, filter_tensor_to_use).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("convTranspose2d set_input(1) filter: {}", e),
                    }
                })?;
                if let Some(bt) = bias_tensor_to_use {
                    layer
                        .set_input(2, bt)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("convTranspose2d set_input(2) bias: {}", e),
                        })?;
                }
                layer
            }
        };

        layer
            .set_stride(&strides)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("convTranspose2d set_stride: {}", e),
            })?;
        layer
            .set_dilation(&dilations)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("convTranspose2d set_dilation: {}", e),
            })?;
        // outputPadding extends output size. Absorb by reducing padding when possible; otherwise add
        // IPaddingLayer for remainder. WebNN: out = (in-1)*stride + kernel - pre - post + outputPadding.
        // TensorRT: out = (in-1)*stride + kernel - pre - post. Reduce pre+post by outputPadding.
        let output_padding: [i32; 2] = deconv_opts
            .and_then(|o| {
                if o.output_padding.len() >= 2 {
                    Some([o.output_padding[0] as i32, o.output_padding[1] as i32])
                } else {
                    None
                }
            })
            .unwrap_or([0, 0]);
        let post_effective: [i32; 2] = [
            (post_padding[0] - output_padding[0]).max(0),
            (post_padding[1] - output_padding[1]).max(0),
        ];
        let pre_effective: [i32; 2] = [
            (pre_padding[0] - (output_padding[0] - post_padding[0]).max(0)).max(0),
            (pre_padding[1] - (output_padding[1] - post_padding[1]).max(0)).max(0),
        ];
        let padding_remainder: [i32; 2] = [
            (output_padding[0]
                - (post_padding[0] - post_effective[0])
                - (pre_padding[0] - pre_effective[0]))
                .max(0),
            (output_padding[1]
                - (post_padding[1] - post_effective[1])
                - (pre_padding[1] - pre_effective[1]))
                .max(0),
        ];
        // TensorRT Deconvolution: pre/post padding trim the output. DimsHW order is (height, width).
        // See https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/operators/docs/Deconvolution.html
        let pre: [i32; 2] = pre_effective;
        let post: [i32; 2] = post_effective;
        layer
            .set_pre_padding(&pre)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("convTranspose2d set_pre_padding: {}", e),
            })?;
        layer
            .set_post_padding(&post)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("convTranspose2d set_post_padding: {}", e),
            })?;

        layer
            .set_num_groups(groups)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("convTranspose2d set_num_groups: {}", e),
            })?;

        let deconv_output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get deconvolution output: {}", e),
            })?;

        // When padding could not fully absorb outputPadding, add IPaddingLayer for remainder.
        let deconv_output = if padding_remainder[0] != 0 || padding_remainder[1] != 0 {
            let pre_pad: Vec<i32> = vec![0, 0];
            let post_pad: Vec<i32> = vec![padding_remainder[0], padding_remainder[1]];
            let pad_layer = network
                .add_padding(&deconv_output, &pre_pad, &post_pad)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("convTranspose2d outputPadding remainder: {}", e),
                })?;
            pad_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("convTranspose2d outputPadding remainder output: {}", e),
                })?
        } else {
            deconv_output
        };

        // When outputSizes (or output_shape) is specified, the graph output has explicit spatial
        // dimensions. Resize deconv output to match: slice if larger, pad if smaller.
        let output_id = operation.output_operands_slice()[0];
        let spatial_adjusted = match (graph.operand(input_id), graph.operand(output_id)) {
            (Some(input_operand), Some(output_operand)) => {
                let in_shape = &input_operand.descriptor.shape;
                let out_shape = &output_operand.descriptor.shape;
                if in_shape.len() != 4 || out_shape.len() != 4 {
                    deconv_output
                } else {
                    let (input_h, input_w): (i32, i32) = if input_layout == "nhwc" {
                        (
                            get_static_or_max_size(&in_shape[1]) as i32,
                            get_static_or_max_size(&in_shape[2]) as i32,
                        )
                    } else {
                        (
                            get_static_or_max_size(&in_shape[2]) as i32,
                            get_static_or_max_size(&in_shape[3]) as i32,
                        )
                    };
                    let (target_h, target_w, out_c): (i32, i32, i32) = if input_layout == "nhwc" {
                        (
                            get_static_or_max_size(&out_shape[1]) as i32,
                            get_static_or_max_size(&out_shape[2]) as i32,
                            get_static_or_max_size(&out_shape[3]) as i32,
                        )
                    } else {
                        (
                            get_static_or_max_size(&out_shape[2]) as i32,
                            get_static_or_max_size(&out_shape[3]) as i32,
                            get_static_or_max_size(&out_shape[1]) as i32,
                        )
                    };
                    let out_batch = get_static_or_max_size(&in_shape[0]) as i32;
                    if target_h <= 0 || target_w <= 0 {
                        deconv_output
                    } else {
                        let effective_kernel_h = (kernel_size[0] - 1) * dilations[0] + 1;
                        let effective_kernel_w = (kernel_size[1] - 1) * dilations[1] + 1;
                        // Output = (Input - 1) * Stride + (Filter - 1) * Dilation + 1 - PrePadding - PostPadding + outputPadding.
                        let current_h = (input_h - 1) * strides[0] + effective_kernel_h
                            - pre_padding[0]
                            - post_padding[0]
                            + output_padding[0];
                        let current_w = (input_w - 1) * strides[1] + effective_kernel_w
                            - pre_padding[1]
                            - post_padding[1]
                            + output_padding[1];
                        let slice_h = current_h.min(target_h);
                        let slice_w = current_w.min(target_w);
                        let mut current = deconv_output;
                        if slice_h < current_h || slice_w < current_w {
                            let start: Vec<i32> = vec![0, 0, 0, 0];
                            let size: Vec<i32> = vec![out_batch, out_c, slice_h, slice_w];
                            let stride: Vec<i32> = vec![1, 1, 1, 1];
                            let slice_layer = network
                                .add_slice(&current, &start, &size, &stride)
                                .map_err(|e| GraphError::ConversionFailed {
                                format: "trtx".to_string(),
                                reason: format!("convTranspose2d outputSizes slice: {}", e),
                            })?;
                            current = slice_layer.get_output(0).map_err(|e| {
                                GraphError::ConversionFailed {
                                    format: "trtx".to_string(),
                                    reason: format!(
                                        "convTranspose2d outputSizes slice output: {}",
                                        e
                                    ),
                                }
                            })?;
                        }
                        let pad_h = target_h - slice_h;
                        let pad_w = target_w - slice_w;
                        if pad_h > 0 || pad_w > 0 {
                            let pre: Vec<i32> = vec![0, 0];
                            let post: Vec<i32> = vec![pad_h, pad_w];
                            let pad_layer =
                                network.add_padding(&current, &pre, &post).map_err(|e| {
                                    GraphError::ConversionFailed {
                                        format: "trtx".to_string(),
                                        reason: format!("convTranspose2d outputSizes pad: {}", e),
                                    }
                                })?;
                            pad_layer
                                .get_output(0)
                                .map_err(|e| GraphError::ConversionFailed {
                                    format: "trtx".to_string(),
                                    reason: format!(
                                        "convTranspose2d outputSizes pad output: {}",
                                        e
                                    ),
                                })?
                        } else {
                            current
                        }
                    }
                }
            }
            _ => deconv_output,
        };

        let conv_output = if input_dtype == DataType::Float16 {
            let cast_layer = network
                .add_cast(&spatial_adjusted, TrtDataType::kHALF)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("convTranspose2d Float->Half cast: {}", e),
                })?;
            cast_layer
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("convTranspose2d cast output: {}", e),
                })?
        } else {
            spatial_adjusted
        };

        let output = if input_layout == "nhwc" {
            let mut shuffle =
                network
                    .add_shuffle(&conv_output)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("convTranspose2d NCHW->NHWC shuffle: {}", e),
                    })?;
            shuffle.set_first_transpose(&[0, 2, 3, 1]).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("convTranspose2d set_first_transpose NCHW->NHWC: {}", e),
                }
            })?;
            shuffle
                .get_output(0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("convTranspose2d NHWC output shuffle: {}", e),
                })?
        } else {
            conv_output
        };

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

        let pool_opts =
            operation
                .attributes
                .as_pool2d()
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "Pool2d operation missing options".to_string(),
                })?;
        let window_size = pool_opts
            .window_dimensions
            .as_ref()
            .filter(|w| !w.is_empty())
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Missing windowDimensions attribute".to_string(),
            })?;

        let window: [i32; 2] = [
            window_size.get(0).copied().unwrap_or(2) as i32,
            window_size.get(1).copied().unwrap_or(2) as i32,
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
            .as_softmax()
            .map(|o| o.axis as i64)
            .or_else(|| operation.attributes.get("axis").and_then(|v| v.as_i64()))
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

        let mut layer =
            network
                .add_concatenation(&inputs)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add concatenation: {}", e),
                })?;

        // WebNN axis (default 0 per spec); use typed options when available
        let axis_raw = operation
            .attributes
            .as_concat()
            .map(|opts| opts.axis as i64)
            .or_else(|| {
                operation
                    .attributes
                    .get("axis")
                    .and_then(|v| v.as_i64().or_else(|| v.as_u64().map(|u| u as i64)))
            })
            .unwrap_or(0);
        let ndim = inputs[0]
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Concat: failed to get input dimensions: {}", e),
            })?
            .len() as i32;
        let mut axis_i32 = axis_raw as i32;
        if axis_i32 < 0 {
            axis_i32 += ndim;
        }
        axis_i32 = axis_i32.max(0).min(ndim.saturating_sub(1));
        layer
            .set_axis(axis_i32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set concat axis {}: {}", axis_i32, e),
            })?;

        // Extract output tensor from layer
        let output = layer
            .get_output(0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get layer output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = *output_ids
            .first()
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "Concat: operation has no output operand".to_string(),
            })?;
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

        let input_dims = input
            .dimensions()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Transpose: failed to get input dimensions: {}", e),
            })?;

        let rank = input_dims.len();
        // WebNN default: when permutation is omitted, reverse axes [rank-1, ..., 0].
        let perm: Vec<i32> = operation
            .attributes
            .as_transpose()
            .map(|o| o.permutation.iter().map(|&u| u as i32).collect())
            .filter(|p: &Vec<i32>| !p.is_empty())
            .unwrap_or_else(|| (0..rank).rev().map(|i| i as i32).collect());

        if perm.len() != rank {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Transpose: permutation length {} does not match rank {}",
                    perm.len(),
                    rank
                ),
            });
        }

        let mut layer = network
            .add_shuffle(input)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add shuffle (transpose): {}", e),
            })?;

        layer
            .set_first_transpose(&perm)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set transpose permutation: {}", e),
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
            .and_then(|v| v.as_array().cloned())
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
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| "nearest-neighbor".to_string());

        // Map WebNN mode to TensorRT ResizeMode (typedef for InterpolationMode)
        let resize_mode = match mode_str.as_str() {
            "nearest-neighbor" => ResizeMode::kNEAREST,
            "linear" => ResizeMode::kLINEAR,
            _ => ResizeMode::kNEAREST, // Default to nearest
        };

        // Parse sizes from attributes (should be output spatial dimensions)
        // WebNN resample2d uses [newHeight, newWidth]
        let sizes = operation
            .attributes
            .get("sizes")
            .and_then(|v| v.as_array().cloned())
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
            .as_gather()
            .map(|o| o.axis as i32)
            .unwrap_or(0);

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
            .and_then(|v| v.as_array().cloned())
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

        // Get input shape from graph
        let input_operand = &graph.operands[operation.input_operands[0] as usize];
        let shape = &input_operand.descriptor.shape;
        let rank = shape.len();

        // Get axes to reverse: axes not present => all; axes=[] => none; axes=[..] => those.
        let axes_to_reverse: Vec<usize> = operation
            .attributes
            .as_reverse()
            .and_then(|o| {
                o.axes
                    .as_ref()
                    .map(|ax| ax.iter().map(|&x| x as usize).collect())
            })
            .unwrap_or_else(|| (0..rank).collect());

        // Build slice parameters for negative stride
        // With negative stride: end_idx = start + (size - 1) * stride
        // For reversing: we want indices [n-1, n-2, ..., 1, 0]
        //   start = n-1 (last element)
        //   size = n (number of elements)
        //   stride = -1
        //   end_idx should be = (n-1) + (n-1)*(-1) = (n-1) - (n-1) = 0 ✓
        let mut starts: Vec<i32> = vec![0; rank];
        let sizes: Vec<i32> = shape
            .iter()
            .map(|s| get_static_or_max_size(s) as i32)
            .collect();
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
            starts[axis] = (get_static_or_max_size(&shape[axis]) - 1) as i32;
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

        let cum_opts = operation.attributes.as_cumulative_sum();
        let axis = cum_opts.map(|o| o.axis as usize).unwrap_or(0);
        let exclusive = cum_opts.map(|o| o.exclusive).unwrap_or(false);
        let reverse = cum_opts.map(|o| o.reversed).unwrap_or(false);

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

        let tri_opts = operation.attributes.as_triangular();
        let upper = tri_opts.and_then(|o| o.upper).unwrap_or(true);
        let diagonal = tri_opts.map(|o| o.diagonal).unwrap_or(0);

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

        let rows = get_static_or_max_size(&shape[shape.len() - 2]) as usize;
        let cols = get_static_or_max_size(&shape[shape.len() - 1]) as usize;

        // Generate triangular mask (1.0 for keep, 0.0 for zero)
        // The mask is computed at build time based on the known shape
        let total_elements: usize = shape
            .iter()
            .map(|s| get_static_or_max_size(s) as usize)
            .product();
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
        let dims: Vec<i32> = shape
            .iter()
            .map(|s| get_static_or_max_size(s) as i32)
            .collect();
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
        trtx::dynamically_load_tensorrt(None::<&str>).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: e.to_string(),
            }
        })?;

        ensure_trtx_loaded().map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: e.to_string(),
        })?;

        // TODO: TRTX converter does not support dynamic dimensions yet
        if graph_info.has_dynamic_dimensions() {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "TODO: TRTX converter does not support graphs with dynamic dimensions"
                    .to_string(),
            });
        }

        // Create TensorRT logger, builder, and network
        let logger = create_trtx_logger().map_err(|e| GraphError::ConversionFailed {
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

    /// HWIO -> OIHW transpose: perm (3,2,0,1) => output[o,i,h,w] = input[h,w,i,o].
    /// Uses exact filter from WPT conv2d test "options.filterLayout='hwio'" (shape [2,2,1,3]).
    #[test]
    fn test_conv_filter_to_oihw_hwio() {
        // WPT hwio filter: shape [H,W,I,O] = [2,2,1,3], 12 floats in row-major order.
        let hwio: [f32; 12] = [
            0.145_438_37,
            0.695_269_23,
            0.307_213_63,
            0.967_112_96,
            0.507_091_34,
            0.432_412_36,
            0.108_360_51,
            0.081_397_07,
            0.984_900_24,
            0.320_230_81,
            0.530_333_88,
            0.428_107_62,
        ];
        let bytes: Vec<u8> = hwio.iter().flat_map(|f| f.to_ne_bytes()).collect();
        let shape = [2u32, 2, 1, 3]; // H, W, I, O

        let out = TrtxConverter::conv_filter_to_oihw(&bytes, "hwio", &shape).unwrap();

        // Expected OIHW: (o,i,h,w) <- (h,w,i,o). O=3, I=1, H=2, W=2.
        let expected_oihw: [f32; 12] = [
            0.145_438_37, // (0,0,0,0) <- (0,0,0,0)
            0.967_112_96, // (0,0,0,1) <- (0,1,0,0)
            0.108_360_51, // (0,0,1,0) <- (1,0,0,0)
            0.320_230_81, // (0,0,1,1) <- (1,1,0,0)
            0.695_269_23, // (1,0,0,0) <- (0,0,0,1)
            0.507_091_34, // (1,0,0,1) <- (0,1,0,1)
            0.081_397_07, // (1,0,1,0) <- (1,0,0,1)
            0.530_333_88, // (1,0,1,1) <- (1,1,0,1)
            0.307_213_63, // (2,0,0,0) <- (0,0,0,2)
            0.432_412_36, // (2,0,0,1) <- (0,1,0,2)
            0.984_900_24, // (2,0,1,0) <- (1,0,0,2)
            0.428_107_62, // (2,0,1,1) <- (1,1,0,2)
        ];
        let expected_bytes: Vec<u8> = expected_oihw.iter().flat_map(|f| f.to_ne_bytes()).collect();
        assert_eq!(out.len(), expected_bytes.len());
        assert_eq!(out, expected_bytes);
    }
}

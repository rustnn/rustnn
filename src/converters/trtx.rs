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

use super::{
    ConvertedGraph, GraphConverter, pool2d_shared::infer_pool2d_ceil_mode_from_output_sizes,
};
use crate::error::GraphError;
use crate::executors::trtx::{create_trtx_logger, ensure_trtx_loaded};
use crate::graph::{DataType, GraphInfo, OperandKind, get_static_or_max_size};
use crate::operator_options::{MLDimension, MLPool2dOptions};
use crate::operators::Operation;
use crate::shape_inference::{
    Conv2dInputLayout, infer_arg_reduce_shape, infer_pool2d_shape, infer_where_shape,
};
use trtx::network::PoolingLayer;
use trtx::{
    ActivationType, Axes, DataType as TrtDataType, ElementWiseOperation, MatrixOperation,
    OwnedConvWeights, OwnedWeights, PaddingMode, PoolingType, ReduceOperation,
    ResizeCoordinateTransformation, ResizeMode, ResizeRoundMode, ScatterMode, TopKOperation,
    UnaryOperation,
};

/// TensorRT native converter
pub struct TrtxConverter;

impl TrtxConverter {
    /// Create a new TrtxConverter
    pub fn new() -> Self {
        TrtxConverter
    }

    /// TensorRT engine I/O tensor name for operand index `operand_id` (`GraphInfo::operands` index).
    ///
    /// WebNN/WPT names are intentionally not used as TRT binding names: TensorRT's QDQ rewrite
    /// matches ONNX-style tokens (e.g. `ZeroPoint`) and can fail engine build for unrelated native graphs.
    pub fn engine_binding_name(operand_id: u32) -> String {
        format!("webnn_operand_{operand_id}")
    }

    #[allow(dead_code)]
    #[inline]
    fn trtx_dims_i64(dims: &[i32]) -> Vec<i64> {
        dims.iter().map(|&d| d as i64).collect()
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
            // TensorRT has no distinct uint4; store 0..=15 in kUINT8 (one byte per element, same as rustnn).
            DataType::Uint4 => Ok(TrtDataType::kUINT8),
            DataType::Uint32 => Ok(TrtDataType::kINT32),
            DataType::Int64 => Ok(TrtDataType::kINT64),
            DataType::Uint64 => Ok(TrtDataType::kINT64),
            DataType::Int4 => Ok(TrtDataType::kINT4),
        }
    }

    /// `[1; rank]` for scalar weights in elementwise ops. Prefer TRT tensor rank when the operand
    /// descriptor has no shape (webnn-graph-json subgraphs often omit intermediate shapes).
    fn trtx_broadcast_ones_for_elementwise_scalar<'a>(
        input: &trtx::Tensor<'a>,
        network: &trtx::NetworkDefinition<'a>,
        descriptor_rank: usize,
        op_label: &'static str,
    ) -> Result<Vec<i64>, GraphError> {
        let rank_from_tensor = input
            .dimensions(network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{op_label}: input dimensions: {e}"),
            })?
            .len();
        let num_dims = if rank_from_tensor > 0 {
            rank_from_tensor
        } else if descriptor_rank > 0 {
            descriptor_rank
        } else {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "{op_label}: could not determine input tensor rank (empty dims and shape)"
                ),
            });
        };
        Ok(vec![1i64; num_dims])
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

    /// `IShuffleLayer::setFirstTranspose`: output axis `i` reads input axis `order[i]`.
    /// WebNN conv2d filter rank-4 layout to TensorRT kernel **OIHW**.
    fn conv_dynamic_filter_first_transpose(filter_layout: &str) -> Result<[i32; 4], GraphError> {
        let order = match filter_layout {
            "oihw" => [0, 1, 2, 3],
            "hwio" => [3, 2, 0, 1],
            "ohwi" => [0, 3, 1, 2],
            "ihwo" => [3, 0, 1, 2],
            "hwoi" => [2, 3, 0, 1],
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!(
                        "Unsupported filter_layout for dynamic conv2d kernel: {}",
                        filter_layout
                    ),
                });
            }
        };
        Ok(order)
    }

    /// WebNN convTranspose2d filter to TensorRT deconv kernel **IOHW** (matches `deconv_filter_to_iohw`).
    fn deconv_dynamic_filter_first_transpose(filter_layout: &str) -> Result<[i32; 4], GraphError> {
        let order = match filter_layout {
            "iohw" => [0, 1, 2, 3],
            "oihw" => [1, 0, 2, 3],
            "hwio" => [2, 3, 0, 1],
            "ohwi" => [3, 0, 1, 2],
            "ihwo" => [0, 3, 1, 2],
            "hwoi" => [3, 2, 0, 1],
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!(
                        "Unsupported filter_layout for dynamic convTranspose2d kernel: {}",
                        filter_layout
                    ),
                });
            }
        };
        Ok(order)
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
    fn cast_to_bool<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        input: &trtx::Tensor<'a>,
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        let layer = network.add_cast(input, TrtDataType::kBOOL).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to cast to BOOL: {}", e),
            }
        })?;
        layer
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get cast output: {}", e),
            })
    }

    /// Cast BOOL tensor to Float32 (false → 0.0, true → 1.0)
    fn cast_to_float32<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        input: &trtx::Tensor<'a>,
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        let layer = network.add_cast(input, TrtDataType::kFLOAT).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to cast to Float32: {}", e),
            }
        })?;
        layer
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get cast output: {}", e),
            })
    }

    /// Cast tensor to Float16 (e.g. after float32 reduction to avoid float16 overflow).
    fn cast_to_float16<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        input: &trtx::Tensor<'a>,
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        let layer = network.add_cast(input, TrtDataType::kHALF).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to cast to Float16: {}", e),
            }
        })?;
        layer
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get cast output: {}", e),
            })
    }

    /// Cast INT32 tensor to Float32
    fn cast_int32_to_float32<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        input: &trtx::Tensor<'a>,
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        let layer = network.add_cast(input, TrtDataType::kFLOAT).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to cast INT32 to Float32: {}", e),
            }
        })?;
        layer
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get cast output: {}", e),
            })
    }

    /// Build TensorRT network from WebNN graph.
    fn build_network<'a>(
        graph: &'a GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
    ) -> Result<(), GraphError> {
        let mut tensor_map: HashMap<u32, trtx::Tensor<'a>> = HashMap::new();
        let promoted_constants: HashSet<u32> = HashSet::new();
        let constants_stored_flat: HashSet<u32> = HashSet::new();

        // Step 1: Add inputs
        for (operand_id, operand) in graph.operands.iter().enumerate() {
            if operand.kind == OperandKind::Input {
                let dtype = Self::webnn_to_trt_dtype(operand.descriptor.data_type)?;
                let dims: Vec<i64> = operand
                    .descriptor
                    .shape
                    .iter()
                    .map(|d| get_static_or_max_size(d) as i64)
                    .collect();
                let trt_io_name = Self::engine_binding_name(operand_id as u32);

                let tensor = network.add_input(&trt_io_name, dtype, &dims).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add input {}: {}", trt_io_name, e),
                    }
                })?;

                tensor.set_name(network, &trt_io_name).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to set input name: {}", e),
                    }
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

                // TensorRT Constant permits kINT8 (not kUINT8). Use kINT8 for Int8/Uint8/Int4/Uint4
                // graph constants: rustnn stores one byte per logical int4/uint4 element (unpacked).
                //
                // Do not pass `kINT4` / sub-byte dtypes to `add_constant` / `add_small_constant_copied`:
                // trtx-rs validates weight size as `(element_count * size_bits) / 8` with truncating
                // division, so e.g. one INT4 scalar expects 0 bytes and panics. We do not patch trtx-rs.
                let use_int8_constant = matches!(
                    operand.descriptor.data_type,
                    DataType::Int8 | DataType::Uint8 | DataType::Int4 | DataType::Uint4
                );

                // TensorRT add_constant does not support kINT64; convert Int64 constants to Int32.
                let promote_int64 = operand.descriptor.data_type == DataType::Int64;

                let add_dims: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
                let layer = if use_int8_constant {
                    network.add_small_constant_copied(&add_dims, data, TrtDataType::kINT8)
                } else if promote_int64 {
                    let int32_bytes: Vec<u8> = data
                        .chunks_exact(8)
                        .flat_map(|chunk| {
                            (i64::from_le_bytes(chunk.try_into().unwrap()) as i32).to_le_bytes()
                        })
                        .collect();
                    network.add_small_constant_copied(&add_dims, &int32_bytes, TrtDataType::kINT32)
                } else {
                    network.add_constant(
                        &add_dims,
                        data,
                        Self::webnn_to_trt_dtype(operand.descriptor.data_type)?,
                    )
                }
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add constant (operand {}): {}", operand_id, e),
                })?;

                let tensor =
                    layer
                        .get_output(&*network, 0)
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

                let trt_io_name = Self::engine_binding_name(operand_id as u32);
                let _ = tensor.set_name(network, &trt_io_name);

                network.mark_output(tensor);
            }
        }

        Ok(())
    }

    /// Add a single operation to the network
    fn add_operation<'network_definition>(
        graph: &'network_definition GraphInfo,
        network: &mut trtx::NetworkDefinition<'network_definition>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'network_definition>>,
        promoted_constants: &HashSet<u32>,
        constants_stored_flat: &HashSet<u32>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let op_type = operation.op_type();

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
            "elu" => Self::add_elu_op(graph, network, tensor_map, operation)?,
            "softsign" => {
                Self::add_activation_op(network, tensor_map, operation, ActivationType::kSOFTSIGN)?
            }
            "softplus" => {
                Self::add_activation_op(network, tensor_map, operation, ActivationType::kSOFTPLUS)?
            }
            "gelu" => {
                Self::add_activation_op(network, tensor_map, operation, ActivationType::kGELU_ERF)?
            }
            "leakyRelu" => Self::add_leaky_relu_op(graph, network, tensor_map, operation)?,
            "prelu" => Self::add_prelu_op(network, tensor_map, operation)?,
            "hardSigmoid" => Self::add_hard_sigmoid_op(graph, network, tensor_map, operation)?,
            "hardSwish" => Self::add_hard_swish_op(graph, network, tensor_map, operation)?,

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

            // Rounding and other
            "ceil" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kCEIL)?,
            "floor" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kFLOOR)?,
            "erf" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kERF)?,
            "sign" => Self::add_unary_op(network, tensor_map, operation, UnaryOperation::kSIGN)?,
            "identity" => Self::add_identity_op(network, tensor_map, operation)?,
            "cast" => Self::add_cast_op(
                graph,
                network,
                tensor_map,
                promoted_constants,
                constants_stored_flat,
                operation,
            )?,
            "quantizeLinear" => {
                Self::add_quantize_linear_op(graph, network, tensor_map, operation)?
            }
            "dequantizeLinear" => {
                Self::add_dequantize_linear_op(graph, network, tensor_map, operation)?
            }

            // Matrix operations
            "matmul" => Self::add_matmul_op(network, tensor_map, operation)?,
            "gemm" => Self::add_gemm_op(graph, network, tensor_map, operation)?,

            // Convolution operations
            "conv2d" => Self::add_conv2d_op(graph, network, tensor_map, operation)?,
            "convTranspose2d" => {
                Self::add_conv_transpose2d_op(graph, network, tensor_map, operation)?
            }

            // Pooling operations
            "averagePool2d" => {
                Self::add_pooling_op(graph, network, tensor_map, operation, PoolingType::kAVERAGE)?
            }
            "maxPool2d" => {
                Self::add_pooling_op(graph, network, tensor_map, operation, PoolingType::kMAX)?
            }
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
            "reduceL2" => Self::add_reduce_l2_op(graph, network, tensor_map, operation)?,
            "reduceLogSum" => Self::add_reduce_log_sum_op(network, tensor_map, operation)?,
            "reduceLogSumExp" => Self::add_reduce_log_sum_exp_op(network, tensor_map, operation)?,
            "reduceSumSquare" => Self::add_reduce_sum_square_op(network, tensor_map, operation)?,

            // Shape manipulation operations
            "slice" => Self::add_slice_op(network, tensor_map, operation)?,
            "split" => Self::add_split_op(network, tensor_map, operation)?,
            "squeeze" => Self::add_squeeze_op(network, tensor_map, operation)?,
            "unsqueeze" => Self::add_unsqueeze_op(network, tensor_map, operation)?,
            "expand" => Self::add_expand_op(graph, network, tensor_map, operation)?,
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
            "logicalNot" => Self::add_logical_not_op(graph, network, tensor_map, operation)?,

            // Indexing/Gathering operations
            "gather" => Self::add_gather_op(graph, network, tensor_map, operation)?,
            "gatherND" => Self::add_gather_nd_op(graph, network, tensor_map, operation)?,
            "scatterElements" => Self::add_scatter_elements_op(network, tensor_map, operation)?,
            "scatterND" => Self::add_scatter_nd_op(network, tensor_map, operation)?,
            "argMax" => Self::add_arg_max_op(graph, network, tensor_map, operation)?,
            "argMin" => Self::add_arg_min_op(graph, network, tensor_map, operation)?,

            // Other operations
            "clamp" => Self::add_clamp_op(graph, network, tensor_map, operation)?,
            "where" => Self::add_where_op(graph, network, tensor_map, operation)?,
            "linear" => Self::add_linear_op(graph, network, tensor_map, operation)?,
            "pad" => Self::add_pad_op(graph, network, tensor_map, operation)?,
            "softmax" => Self::add_softmax_op(network, tensor_map, operation)?,
            "concat" => Self::add_concat_op(network, tensor_map, operation)?,
            "isNaN" => Self::add_is_nan_op(network, tensor_map, operation)?,
            "isInfinite" => Self::add_is_infinite_op(network, tensor_map, operation)?,
            "roundEven" => Self::add_round_even_op(network, tensor_map, operation)?,
            "gatherElements" => {
                Self::add_gather_elements_op(graph, network, tensor_map, operation)?
            }
            "l2Pool2d" => Self::add_l2_pool2d_op(graph, network, tensor_map, operation)?,
            "reverse" => Self::add_reverse_op(graph, network, tensor_map, operation)?,
            "cumulativeSum" => Self::add_cumulative_sum_op(graph, network, tensor_map, operation)?,
            "triangular" => Self::add_triangular_op(graph, network, tensor_map, operation)?,
            "transpose" => Self::add_transpose_op(graph, network, tensor_map, operation)?,
            "reshape" => Self::add_reshape_op(graph, network, tensor_map, operation)?,
            "resample2d" => Self::add_resample2d_op(graph, network, tensor_map, operation)?,

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
    fn ensure_broadcast_compatible<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor0: &trtx::Tensor<'a>,
        tensor1: &trtx::Tensor<'a>,
        op_name: &str,
    ) -> Result<(trtx::Tensor<'a>, trtx::Tensor<'a>), GraphError> {
        // Get dimensions of both tensors
        let dims0 = tensor0
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Failed to get dimensions for tensor 0 in {}: {}",
                    op_name, e
                ),
            })?;

        let dims1 = tensor1
            .dimensions(&*network)
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
                .get_output(&*network, 0)
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
                .get_output(&*network, 0)
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
                let t0 =
                    id0.get_output(&*network, 0)
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
                let t1 =
                    id1.get_output(&*network, 0)
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

                resize_layer.set_output_dimensions(network, &dims1);

                resize_layer
                    .get_output(&*network, 0)
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
                id.get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
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

                resize_layer.set_output_dimensions(network, &dims0);

                resize_layer
                    .get_output(&*network, 0)
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
                id.get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
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

        let reshape_dims =
            to_reshape
                .dimensions(&*network)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get reshape dims: {}", e),
                })?;
        let target_dims =
            to_keep
                .dimensions(&*network)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get target dims: {}", e),
                })?;

        // Pad smaller tensor with leading 1s
        let rank_diff = target_dims.len() - reshape_dims.len();
        let mut new_shape: Vec<i64> = vec![1i64; rank_diff];
        new_shape.extend_from_slice(&reshape_dims);

        let mut shuffle_layer =
            network
                .add_shuffle(to_reshape)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add shuffle layer for broadcasting: {}", e),
                })?;

        shuffle_layer
            .set_reshape_dimensions(network, &new_shape)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set reshape dimensions: {}", e),
            })?;

        let reshaped =
            shuffle_layer
                .get_output(&*network, 0)
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
            .get_output(&*network, 0)
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

    /// Static shape dimensions for TRTX broadcast (all dims must be known).
    fn operand_shape_u32_static(graph: &GraphInfo, id: u32) -> Result<Vec<u32>, GraphError> {
        let operand = graph
            .operand(id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("operand {} not in graph", id),
            })?;
        let mut out = Vec::with_capacity(operand.descriptor.shape.len());
        for d in &operand.descriptor.shape {
            let s = get_static_or_max_size(d);
            if s == 0 {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!(
                        "operand {} needs static dimensions for TRTX broadcast (got 0)",
                        id
                    ),
                });
            }
            out.push(s);
        }
        Ok(out)
    }

    /// Expand `tensor` from `from_dims` to `target_dims` (NumPy-style; dimensions must be conformable).
    fn broadcast_trtx_tensor_to_dims<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor: &trtx::Tensor<'a>,
        from_dims: &[i64],
        target_dims: &[i64],
        op_label: &str,
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        if from_dims == target_dims {
            let layer = network
                .add_identity(tensor)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{op_label} broadcast identity: {e}"),
                })?;
            return layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{op_label} broadcast identity output: {e}"),
                });
        }

        let mut cur: &trtx::Tensor<'a> = tensor;
        let mut dims = from_dims.to_vec();
        let mut staged: Option<trtx::Tensor<'a>> = None;

        if dims.len() < target_dims.len() {
            let rank_diff = target_dims.len() - dims.len();
            let mut new_shape: Vec<i64> = vec![1i64; rank_diff];
            new_shape.extend_from_slice(&dims);
            let mut shuffle_layer =
                network
                    .add_shuffle(cur)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("{op_label} broadcast shuffle: {e}"),
                    })?;
            shuffle_layer
                .set_reshape_dimensions(network, &new_shape)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{op_label} broadcast reshape: {e}"),
                })?;
            let out = shuffle_layer.get_output(&*network, 0).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{op_label} broadcast shuffle output: {e}"),
                }
            })?;
            staged = Some(out);
            dims = new_shape;
            cur = staged
                .as_ref()
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{op_label} broadcast shuffle missing tensor"),
                })?;
        }

        if dims.len() != target_dims.len() {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "{op_label} broadcast: rank mismatch {:?} vs {:?}",
                    dims, target_dims
                ),
            });
        }

        for (&d, &t) in dims.iter().zip(target_dims.iter()) {
            if d != t && d != 1 {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{op_label} broadcast: cannot expand dim {d} to {t}"),
                });
            }
        }

        if dims == target_dims {
            return if let Some(t) = staged {
                Ok(t)
            } else {
                Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{op_label} broadcast: missing tensor after rank pad"),
                })
            };
        }

        let mut resize_layer =
            network
                .add_resize(cur)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{op_label} broadcast resize: {e}"),
                })?;
        resize_layer.set_output_dimensions(network, target_dims);
        resize_layer
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{op_label} broadcast resize output: {e}"),
            })
    }

    /// Add elementwise operation
    fn add_elementwise_op<'a>(
        _graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
        op_code: ElementWiseOperation,
    ) -> Result<(), GraphError> {
        let input0 = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let input1 = tensor_map
            .get(&operation.input_operands()[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[1]),
            })?;

        // Ensure broadcast compatibility (this may reshape tensors if needed)
        let (bc_input0, bc_input1) =
            Self::ensure_broadcast_compatible(network, input0, input1, operation.op_type())?;

        let layer = network
            .add_elementwise(&bc_input0, &bc_input1, op_code)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add elementwise operation: {}", e),
            })?;

        // Extract output tensor from layer
        let output = layer
            .get_output(&*network, 0)
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
    fn add_comparison_op<'a>(
        _graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
        op_code: ElementWiseOperation,
    ) -> Result<(), GraphError> {
        let input0 = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let input1 = tensor_map
            .get(&operation.input_operands()[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[1]),
            })?;

        // Ensure broadcast compatibility
        let (bc_input0, bc_input1) =
            Self::ensure_broadcast_compatible(network, input0, input1, operation.op_type())?;

        // Comparison operation returns BOOL
        let layer = network
            .add_elementwise(&bc_input0, &bc_input1, op_code)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add comparison operation: {}", e),
            })?;

        let bool_output =
            layer
                .get_output(&*network, 0)
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

    /// Add logical operation: broadcast on float/half tensors, cast to BOOL, elementwise kAND/kOR/kXOR, Float32 output.
    ///
    /// TensorRT elementwise `kAND` / `kOR` / `kXOR` require BOOL inputs, so we cast after broadcast.
    /// [`ensure_broadcast_compatible`] may use `IResizeLayer`, which does not accept BOOL—so broadcast must
    /// run on Float/Half (original path). UInt8/Int8 are **`Cast` to Float32 first** so they never pass
    /// through `Identity` as internal UINT8/INT8 (strongly-typed TRT rejects that).
    fn add_logical_binary_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
        op_code: ElementWiseOperation,
    ) -> Result<(), GraphError> {
        let id0 = operation.input_operands()[0];
        let id1 = operation.input_operands()[1];
        let input0 = tensor_map
            .get(&id0)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", id0),
            })?;

        let input1 = tensor_map
            .get(&id1)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", id1),
            })?;

        let promoted0 = match graph.operand(id0).map(|o| o.descriptor.data_type) {
            Some(DataType::Uint8) | Some(DataType::Int8) => {
                Some(Self::cast_to_float32(network, input0)?)
            }
            _ => None,
        };
        let promoted1 = match graph.operand(id1).map(|o| o.descriptor.data_type) {
            Some(DataType::Uint8) | Some(DataType::Int8) => {
                Some(Self::cast_to_float32(network, input1)?)
            }
            _ => None,
        };

        let t0: &trtx::Tensor<'a> = promoted0.as_ref().unwrap_or(input0);
        let t1: &trtx::Tensor<'a> = promoted1.as_ref().unwrap_or(input1);

        let (bc_input0, bc_input1) =
            Self::ensure_broadcast_compatible(network, t0, t1, operation.op_type())?;

        let bool_input0 = Self::cast_to_bool(network, &bc_input0)?;
        let bool_input1 = Self::cast_to_bool(network, &bc_input1)?;

        let layer = network
            .add_elementwise(&bool_input0, &bool_input1, op_code)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add logical operation: {}", e),
            })?;

        let bool_output =
            layer
                .get_output(&*network, 0)
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
    fn add_logical_not_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_id = operation.input_operands()[0];
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
                    let shape: Vec<i64> = operand
                        .descriptor
                        .shape
                        .iter()
                        .map(|d| get_static_or_max_size(d) as i64)
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
                    let const_layer = network
                        .add_small_constant_copied(&shape, &bool_bytes, TrtDataType::kBOOL)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("LogicalNot: failed to add BOOL constant: {}", e),
                        })?;
                    const_layer.get_output(&*network, 0).map_err(|e| {
                        GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("LogicalNot: BOOL constant output: {}", e),
                        }
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

        let not_output =
            layer
                .get_output(&*network, 0)
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
    fn add_activation_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
        activation_type: ActivationType,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let layer = network
            .add_activation(input, activation_type)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add activation: {}", e),
            })?;

        // Extract output tensor from layer
        let output = layer
            .get_output(&*network, 0)
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
    fn add_elu_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let alpha = operation
            .attributes()
            .as_elu()
            .map(|o| o.alpha as f32)
            .unwrap_or(1.0);

        let input_operand = graph
            .operand(operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Input operand {} not found in graph",
                    operation.input_operands()[0]
                ),
            })?;
        let input_dtype = input_operand.descriptor.data_type;
        // Use built-in kELU only for float32 with default alpha; float16 needs decomposition for correct precision.
        if (alpha - 1.0).abs() <= f32::EPSILON && input_dtype != DataType::Float16 {
            return Self::add_activation_op(network, tensor_map, operation, ActivationType::kELU);
        }

        let broadcast_shape = Self::trtx_broadcast_ones_for_elementwise_scalar(
            input,
            &*network,
            input_operand.descriptor.shape.len(),
            "elu",
        )?;
        let (trt_dtype, one_bytes, zero_bytes, alpha_bytes) = match input_dtype {
            DataType::Float16 => {
                let one: Vec<u8> = f16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
                let zero: Vec<u8> = f16::from_f32(0.0).to_bits().to_le_bytes().to_vec();
                let alpha_f16: Vec<u8> = f16::from_f32(alpha).to_bits().to_le_bytes().to_vec();
                (TrtDataType::kHALF, one, zero, alpha_f16)
            }
            _ => {
                let one: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
                let zero: Vec<u8> = 0.0f32.to_le_bytes().to_vec();
                let alpha_f32: Vec<u8> = alpha.to_le_bytes().to_vec();
                (TrtDataType::kFLOAT, one, zero, alpha_f32)
            }
        };

        // Decompose: ELU(x) = relu(x) + alpha * min(0, exp(x) - 1)
        let relu_layer = network
            .add_activation(input, ActivationType::kRELU)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add relu for elu: {}", e),
            })?;
        let relu_output =
            relu_layer
                .get_output(&*network, 0)
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
        let exp_output =
            exp_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get exp output: {}", e),
                })?;

        let one_const = network
            .add_small_constant_copied(&broadcast_shape, &one_bytes, trt_dtype.clone())
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create one constant for elu: {}", e),
            })?;
        let one_tensor =
            one_const
                .get_output(&*network, 0)
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
        let exp_minus_1 = exp_minus_1_layer.get_output(&*network, 0).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get exp-1 output: {}", e),
            }
        })?;

        let zero_const = network
            .add_small_constant_copied(&broadcast_shape, &zero_bytes, trt_dtype.clone())
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create zero constant for elu: {}", e),
            })?;
        let zero_tensor =
            zero_const
                .get_output(&*network, 0)
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
        let neg_part =
            neg_part_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get neg part output: {}", e),
                })?;

        let alpha_const = network
            .add_small_constant_copied(&broadcast_shape, &alpha_bytes, trt_dtype.clone())
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create alpha constant for elu: {}", e),
            })?;
        let alpha_tensor =
            alpha_const
                .get_output(&*network, 0)
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
        let scaled_neg = scaled_neg_layer.get_output(&*network, 0).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get scaled neg output: {}", e),
            }
        })?;

        let (bc_relu, bc_scaled) =
            Self::ensure_broadcast_compatible(network, &relu_output, &scaled_neg, "elu_add")?;

        let final_layer = network
            .add_elementwise(&bc_relu, &bc_scaled, ElementWiseOperation::kSUM)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add elu parts: {}", e),
            })?;
        let output =
            final_layer
                .get_output(&*network, 0)
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
    fn add_unary_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
        unary_op: UnaryOperation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
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
            .get_output(&*network, 0)
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
    fn add_leaky_relu_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let alpha = operation
            .attributes()
            .as_leaky_relu()
            .map(|o| o.alpha as f32)
            .unwrap_or(0.01);

        let input_operand = graph
            .operand(operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Input operand {} not found in graph",
                    operation.input_operands()[0]
                ),
            })?;
        let broadcast_shape = Self::trtx_broadcast_ones_for_elementwise_scalar(
            input,
            &*network,
            input_operand.descriptor.shape.len(),
            "leakyRelu",
        )?;

        let (alpha_bytes, alpha_dtype) = match input_operand.descriptor.data_type {
            DataType::Float16 => (
                f16::from_f32(alpha).to_bits().to_le_bytes().to_vec(),
                TrtDataType::kHALF,
            ),
            _ => (alpha.to_le_bytes().to_vec(), TrtDataType::kFLOAT),
        };
        let alpha_const = network
            .add_small_constant_copied(&broadcast_shape, &alpha_bytes, alpha_dtype)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("LeakyReLU: failed to add alpha constant: {}", e),
            })?;
        let alpha_tensor =
            alpha_const
                .get_output(&*network, 0)
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
        let relu_output =
            relu_layer
                .get_output(&*network, 0)
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
        let neg_part =
            neg_part_layer
                .get_output(&*network, 0)
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
        let scaled_neg = scaled_neg_layer.get_output(&*network, 0).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get scaled neg output: {}", e),
            }
        })?;

        // relu(x) + alpha * min(0, x)
        let final_layer = network
            .add_elementwise(&relu_output, &scaled_neg, ElementWiseOperation::kSUM)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add leaky relu parts: {}", e),
            })?;

        let output =
            final_layer
                .get_output(&*network, 0)
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
    fn add_prelu_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let slope = tensor_map
            .get(&operation.input_operands()[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Slope operand {} not found", operation.input_operands()[1]),
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
        let relu_output =
            relu_layer
                .get_output(&*network, 0)
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
        let zero_output =
            zero_layer
                .get_output(&*network, 0)
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
        let neg_part =
            neg_part_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get negative part: {}", e),
                })?;

        // slope * min(0, x) — TRT requires same rank for elementwise; reshape slope with leading 1s (WebNN broadcast).
        let (neg_bc, slope_bc) =
            Self::ensure_broadcast_compatible(network, &neg_part, slope, "prelu_slope")?;
        let scaled_neg_layer = network
            .add_elementwise(&neg_bc, &slope_bc, ElementWiseOperation::kPROD)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to scale negative part: {}", e),
            })?;
        let scaled_neg = scaled_neg_layer.get_output(&*network, 0).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get scaled negative: {}", e),
            }
        })?;

        // Final: relu + slope * neg_part
        let final_layer = network
            .add_elementwise(&relu_output, &scaled_neg, ElementWiseOperation::kSUM)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add prelu parts: {}", e),
            })?;

        let output =
            final_layer
                .get_output(&*network, 0)
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
    fn add_hard_sigmoid_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let attrs = operation.attributes();
        let opts = attrs.as_hard_sigmoid();
        let alpha = opts.map(|o| o.alpha as f32).unwrap_or(0.2);
        let beta = opts.map(|o| o.beta as f32).unwrap_or(0.5);

        if (alpha - 0.2f32).abs() <= 1e-5 && (beta - 0.5f32).abs() <= 1e-5 {
            let layer = network
                .add_activation(input, ActivationType::kHARD_SIGMOID)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add hard sigmoid: {}", e),
                })?;
            let output =
                layer
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get layer output: {}", e),
                    })?;
            let output_ids = operation.output_operands_slice();
            tensor_map.insert(output_ids[0], output);
            return Ok(());
        }

        let input_operand = graph
            .operand(operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Input operand {} not found in graph",
                    operation.input_operands()[0]
                ),
            })?;
        let input_dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: failed to get input dimensions: {}", e),
            })?;
        let broadcast_shape: Vec<i64> = vec![1i64; input_dims.len()];
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
        let alpha_const = network
            .add_small_constant_copied(&broadcast_shape, &alpha_bytes, trt_dtype.clone())
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: failed to add alpha constant: {}", e),
            })?;
        let beta_const = network
            .add_small_constant_copied(&broadcast_shape, &beta_bytes, trt_dtype.clone())
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: failed to add beta constant: {}", e),
            })?;
        let zero_const = network
            .add_small_constant_copied(&broadcast_shape, &zero_bytes, trt_dtype.clone())
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: failed to add zero constant: {}", e),
            })?;
        let one_const = network
            .add_small_constant_copied(&broadcast_shape, &one_bytes, trt_dtype)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: failed to add one constant: {}", e),
            })?;

        let alpha_out =
            alpha_const
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("HardSigmoid: alpha const output: {}", e),
                })?;
        let beta_out =
            beta_const
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("HardSigmoid: beta const output: {}", e),
                })?;
        let zero_out =
            zero_const
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("HardSigmoid: zero const output: {}", e),
                })?;
        let one_out =
            one_const
                .get_output(&*network, 0)
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
        let linear = ax
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: linear get output: {}", e),
            })?;
        let linear_plus_beta = network
            .add_elementwise(&linear, &beta_out, ElementWiseOperation::kSUM)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: linear+beta: {}", e),
            })?;
        let linear_out = linear_plus_beta.get_output(&*network, 0).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: linear_out: {}", e),
            }
        })?;

        // clamp(linear, 0, 1) = max(0, min(1, linear))
        let min1 = network
            .add_elementwise(&linear_out, &one_out, ElementWiseOperation::kMIN)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSigmoid: min(1, linear): {}", e),
            })?;
        let min1_out = min1
            .get_output(&*network, 0)
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
        let output =
            output_layer
                .get_output(&*network, 0)
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
    fn add_hard_swish_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let input_operand = graph
            .operand(operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Input operand {} not found in graph",
                    operation.input_operands()[0]
                ),
            })?;
        let input_dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: failed to get input dimensions: {}", e),
            })?;
        let broadcast_shape: Vec<i64> = vec![1i64; input_dims.len()];
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
        let three_const = network
            .add_small_constant_copied(&broadcast_shape, &three_bytes, trt_dtype.clone())
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: failed to add 3 constant: {}", e),
            })?;
        let six_const = network
            .add_small_constant_copied(&broadcast_shape, &six_bytes, trt_dtype.clone())
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: failed to add 6 constant: {}", e),
            })?;
        let zero_const = network
            .add_small_constant_copied(&broadcast_shape, &zero_bytes, trt_dtype)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("HardSwish: failed to add zero constant: {}", e),
            })?;

        let three_out =
            three_const
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("HardSwish: 3 const output: {}", e),
                })?;
        let six_out =
            six_const
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("HardSwish: 6 const output: {}", e),
                })?;
        let zero_out =
            zero_const
                .get_output(&*network, 0)
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
        let x_plus_3_out =
            x_plus_3
                .get_output(&*network, 0)
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
            .get_output(&*network, 0)
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
        let inner_out =
            inner
                .get_output(&*network, 0)
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
                .get_output(&*network, 0)
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
        let output =
            output_layer
                .get_output(&*network, 0)
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
    fn add_identity_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_id = operation.input_operands()[0];
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
            .get_output(&*network, 0)
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
    fn add_cast_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        promoted_constants: &HashSet<u32>,
        constants_stored_flat: &HashSet<u32>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_id = operation.input_operands()[0];
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
            let output = identity_layer.get_output(&*network, 0).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get identity output: {}", e),
                }
            })?;
            tensor_map.insert(output_id, output);
            return Ok(());
        }

        // Helper: DQ(scale=1) with per-tensor (0D) scale. TensorRT only allows scalar or per-channel scale; 4D scale causes "ScaleMode is illegal".
        let add_dq_scale_constant = |network: &mut trtx::NetworkDefinition<'a>,
                                     err_prefix: &str|
         -> Result<trtx::Tensor<'a>, GraphError> {
            let scale_one = 1.0f32.to_le_bytes();
            let scale_constant = network
                .add_small_constant_copied(&[], scale_one.as_slice(), TrtDataType::kFLOAT)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{}: {}", err_prefix, e),
                })?;
            scale_constant
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{}: get scale output: {}", err_prefix, e),
                })
        };

        // Constant stored flat: 1D int8. Try 1D -> Reshape(4D) -> DQ so DQ sees Shuffle output not Constant.
        let stored_flat = constants_stored_flat.contains(&input_id);
        let _input_dims =
            input
                .dimensions(&*network)
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
                let original_shape: Vec<i64> = input_operand
                    .descriptor
                    .shape
                    .iter()
                    .map(|d| get_static_or_max_size(d) as i64)
                    .collect();
                let mut shuffle_to_4d =
                    network
                        .add_shuffle(input)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Cast: failed to add reshape shuffle: {}", e),
                        })?;
                let _ = shuffle_to_4d.set_name(network, "cast_flat_reshape_4d");
                shuffle_to_4d
                    .set_reshape_dimensions(network, &original_shape)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Cast: failed to set reshape dimensions: {}", e),
                    })?;
                let reshaped_4d = shuffle_to_4d.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Cast: failed to get reshape output: {}", e),
                    }
                })?;
                let _ = reshaped_4d.set_name(network, "cast_flat_reshape_4d");
                let scale_tensor = add_dq_scale_constant(network, "int8->float32 cast")?;
                let mut dq_layer = network
                    .add_dequantize(&reshaped_4d, &scale_tensor, TrtDataType::kFLOAT)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add dequantize for int8->float32 cast: {}", e),
                    })?;
                let _ = dq_layer.set_name(network, "cast_flat_dq_f32");
                let output = dq_layer.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get dequantize output: {}", e),
                    }
                })?;
                let _ = output.set_name(network, "cast_flat_dq_f32");
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
                let original_shape: Vec<i64> = input_operand
                    .descriptor
                    .shape
                    .iter()
                    .map(|d| get_static_or_max_size(d) as i64)
                    .collect();
                let mut shuffle_to_4d =
                    network
                        .add_shuffle(input)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Cast: failed to add reshape shuffle: {}", e),
                        })?;
                let _ = shuffle_to_4d.set_name(network, "cast_flat_reshape_4d_int32");
                shuffle_to_4d
                    .set_reshape_dimensions(network, &original_shape)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Cast: failed to set reshape dimensions: {}", e),
                    })?;
                let reshaped_4d = shuffle_to_4d.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Cast: failed to get reshape output: {}", e),
                    }
                })?;
                let _ = reshaped_4d.set_name(network, "cast_flat_reshape_4d");
                let scale_tensor = add_dq_scale_constant(network, "int8->int32 cast")?;
                let mut dq_layer = network
                    .add_dequantize(&reshaped_4d, &scale_tensor, TrtDataType::kFLOAT)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add dequantize for int8->int32 cast: {}", e),
                    })?;
                let _ = dq_layer.set_name(network, "cast_flat_dq_int32");
                let dq_out = dq_layer.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get dequantize output: {}", e),
                    }
                })?;
                let _ = dq_out.set_name(network, "cast_flat_dq_int32");
                let mut cast_layer =
                    network
                        .add_cast(&dq_out, TrtDataType::kINT32)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Failed to add cast to INT32: {}", e),
                        })?;
                let _ = cast_layer.set_name(network, "cast_flat_cast_int32");
                let output = cast_layer.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get cast output: {}", e),
                    }
                })?;
                let _ = output.set_name(network, "cast_flat_cast_int32");
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
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get cast output: {}", e),
            })?;

        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// TensorRT `IQuantizeLayer` / `IDequantizeLayer` require `input.nbDims > 0`.
    /// Expand 0D (scalar) tensors to `[1]` before Q/DQ.
    ///
    /// **Do not** reshape Q/DQ output back to rank-0: Myelin can assert when mask rank and
    /// tensor rank disagree on empty `IShuffleLayer` reshapes.
    fn trtx_qdq_ensure_rank1<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor: &trtx::Tensor<'a>,
        label: &str,
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        let dims = tensor
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label}: dimensions: {e}"),
            })?;
        if dims.is_empty() {
            let mut sh = network
                .add_shuffle(tensor)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{label}: rank-1 shuffle: {e}"),
                })?;
            sh.set_reshape_dimensions(network, &[1i64]).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{label}: reshape [1]: {e}"),
                }
            })?;
            sh.get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{label}: shuffle output: {e}"),
                })
        } else {
            let id = network
                .add_identity(tensor)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{label}: identity: {e}"),
                })?;
            id.get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{label}: identity output: {e}"),
                })
        }
    }

    /// Prepend size-1 dimensions so `tensor` has rank `target_rank` (NumPy-style leading ones).
    fn trtx_prepend_ones_to_trt_rank<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor: &trtx::Tensor<'a>,
        target_rank: usize,
        label: &str,
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        let dims = tensor
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label}: prepend dimensions: {e}"),
            })?;
        if dims.len() == target_rank {
            let id = network
                .add_identity(tensor)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{label}: prepend identity: {e}"),
                })?;
            return id
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{label}: prepend identity output: {e}"),
                });
        }
        if dims.len() > target_rank {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "{label}: tensor rank {} exceeds target rank {}",
                    dims.len(),
                    target_rank
                ),
            });
        }
        let mut new_shape = vec![1_i64; target_rank - dims.len()];
        new_shape.extend_from_slice(&dims);
        let mut sh = network
            .add_shuffle(tensor)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label}: prepend shuffle: {e}"),
            })?;
        sh.set_reshape_dimensions(network, &new_shape)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label}: prepend reshape: {e}"),
            })?;
        sh.get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label}: prepend shuffle output: {e}"),
            })
    }

    /// After `IDequantizeLayer` (scale = 1) on **kINT8** weights that hold **WebNN Uint8** bytes, recover
    /// ONNX zero-point 0..255: the constant uses the same raw bytes as WebNN but TRT reads them as
    /// int8 (−128..127), so e.g. zp byte 128 becomes −128 after DQ. Use `dq + 256` where `dq < 0`.
    fn trtx_int8_dq_reinterpret_uint8_storage<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        dq_out: &trtx::Tensor<'a>,
        dq_out_ty: TrtDataType,
        label: &str,
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        let dims = dq_out
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label} uint8 storage fix dimensions: {e}"),
            })?;
        let bshape: Vec<i64> = if dims.is_empty() {
            vec![]
        } else {
            vec![1_i64; dims.len()]
        };
        let (zero_bytes, c256_bytes) = match dq_out_ty {
            TrtDataType::kFLOAT => (
                0.0f32.to_le_bytes().to_vec(),
                256.0f32.to_le_bytes().to_vec(),
            ),
            TrtDataType::kHALF => (
                f16::from_f32(0.0f32).to_le_bytes().to_vec(),
                f16::from_f32(256.0f32).to_le_bytes().to_vec(),
            ),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{label} uint8 storage fix: expected float DQ output"),
                });
            }
        };
        let zero_t = network
            .add_small_constant_copied(&bshape, &zero_bytes, dq_out_ty)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label} uint8 zp fix zero: {e}"),
            })?
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label} uint8 zp fix zero out: {e}"),
            })?;
        let c256_t = network
            .add_small_constant_copied(&bshape, &c256_bytes, dq_out_ty)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label} uint8 zp fix 256: {e}"),
            })?
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label} uint8 zp fix 256 out: {e}"),
            })?;
        let less = network
            .add_elementwise(dq_out, &zero_t, ElementWiseOperation::kLESS)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label} uint8 zp fix less0: {e}"),
            })?
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label} uint8 zp fix less0 out: {e}"),
            })?;
        let cond = Self::cast_to_bool(network, &less)?;
        let shifted = network
            .add_elementwise(dq_out, &c256_t, ElementWiseOperation::kSUM)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label} uint8 zp fix +256: {e}"),
            })?
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label} uint8 zp fix +256 out: {e}"),
            })?;
        let layer = network.add_select(&cond, &shifted, dq_out).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label} uint8 zp fix select: {e}"),
            }
        })?;
        layer
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label} uint8 zp fix select out: {e}"),
            })
    }

    /// `IDequantizeLayer` with scale = 1 so int8/uint8 constants can feed float math (QDQ validator).
    ///
    /// WebNN **Uint8** operands are stored as **kINT8** constants (TensorRT); `webnn_integer_dtype`
    /// selects correct float conversion. True **kUINT8** tensors use symmetric DQ (+ **128** after DQ).
    fn trtx_int8_uint8_identity_dequantize<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor: &trtx::Tensor<'a>,
        dq_out_ty: TrtDataType,
        label: &str,
        webnn_integer_dtype: DataType,
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        let input_ty = tensor.get_type(&*network);
        // Always use a **scalar** (0D) scale for identity DQ. Matching zp rank (e.g. `[1,1]`) hits
        // TensorRT `ScaleMode is illegal`; per-tensor scale broadcasts. Myelin: data rank >= scale rank.
        let scale_shape: Vec<i64> = vec![];
        let scale_bytes: Vec<u8> = match dq_out_ty {
            TrtDataType::kFLOAT => 1.0f32.to_le_bytes().to_vec(),
            TrtDataType::kHALF => f16::from_f32(1.0f32).to_le_bytes().to_vec(),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!(
                        "{label} identity DQ expects kFLOAT/kHALF output, got {:?}",
                        dq_out_ty
                    ),
                });
            }
        };
        let scale_t = network
            .add_small_constant_copied(&scale_shape, &scale_bytes, dq_out_ty)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label} identity DQ scale constant: {e}"),
            })?
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label} identity DQ scale tensor: {e}"),
            })?;
        let dq_layer = network
            .add_dequantize(tensor, &scale_t, dq_out_ty)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label} identity DQ: {e}"),
            })?;
        let dq_out =
            dq_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{label} identity DQ output: {e}"),
                })?;

        match (webnn_integer_dtype, input_ty) {
            (DataType::Uint8, TrtDataType::kINT8) => {
                Self::trtx_int8_dq_reinterpret_uint8_storage(network, &dq_out, dq_out_ty, label)
            }
            (DataType::Uint8, TrtDataType::kUINT8) => {
                let dq_dims =
                    dq_out
                        .dimensions(&*network)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("{label} kUINT8 DQ+128 dimensions: {e}"),
                        })?;
                let bias_shape: Vec<i64> = if dq_dims.is_empty() {
                    vec![]
                } else {
                    vec![1_i64; dq_dims.len()]
                };
                let bias_bytes: Vec<u8> = match dq_out_ty {
                    TrtDataType::kFLOAT => 128.0f32.to_le_bytes().to_vec(),
                    TrtDataType::kHALF => f16::from_f32(128.0f32).to_le_bytes().to_vec(),
                    _ => {
                        return Err(GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("{label} kUINT8 DQ+128: unexpected DQ dtype"),
                        });
                    }
                };
                let bias_t = network
                    .add_small_constant_copied(&bias_shape, &bias_bytes, dq_out_ty)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("{label} kUINT8 +128 bias: {e}"),
                    })?
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("{label} kUINT8 +128 tensor: {e}"),
                    })?;
                network
                    .add_elementwise(&dq_out, &bias_t, ElementWiseOperation::kSUM)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("{label} kUINT8 +128 sum: {e}"),
                    })?
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("{label} kUINT8 +128 output: {e}"),
                    })
            }
            _ => Ok(dq_out),
        }
    }

    /// Expand `scale` / `zero_point` from WebNN **block** shapes (e.g. `[1,2]` vs input `[3,4]`) so
    /// TensorRT elementwise `x / scale` sees ordinary broadcast rules. Matches ONNX converter
    /// `align_param_with_input` (reshape + tile) using concatenation along each axis.
    ///
    /// When `int8_uint8_identity_dq_to` is `Some(fp)`, **`kINT8` / `kUINT8` tensors are passed through
    /// `IDequantizeLayer` (scale = 1) before any Shuffle** so `QuantizedConstantValidator` sees
    /// Constant → DQ. Must be `None` for float `scale` tensors.
    ///
    /// `identity_dq_webnn_dtype`: WebNN dtype of the **integer** param (zero_point). Uint8 zp is stored
    /// as `kINT8` in TRT; pass `DataType::Uint8` so bytes 128..255 map correctly.
    fn trtx_align_quantize_param_for_elementwise<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor: &trtx::Tensor<'a>,
        _param_shape: &[u32],
        input_shape: &[u32],
        label: &str,
        int8_uint8_identity_dq_to: Option<TrtDataType>,
        identity_dq_webnn_dtype: Option<DataType>,
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        let dq_tensor: Option<trtx::Tensor<'a>> = if let Some(dq_ty) = int8_uint8_identity_dq_to {
            if matches!(
                tensor.get_type(&*network),
                TrtDataType::kINT8 | TrtDataType::kUINT8
            ) {
                let wd = identity_dq_webnn_dtype.unwrap_or(DataType::Int8);
                Some(Self::trtx_int8_uint8_identity_dequantize(
                    network, tensor, dq_ty, label, wd,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        let src_for_rank1 = dq_tensor.as_ref().unwrap_or(tensor);
        let mut cur = Self::trtx_qdq_ensure_rank1(network, src_for_rank1, label)?;

        if input_shape.is_empty() {
            return Ok(cur);
        }

        cur = Self::trtx_prepend_ones_to_trt_rank(network, &cur, input_shape.len(), label)?;

        let eff_shape_u32: Vec<u32> = cur
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{label}: aligned param dimensions: {e}"),
            })?
            .iter()
            .map(|&d| d as u32)
            .collect();

        if eff_shape_u32 == *input_shape {
            return Ok(cur);
        }

        let broadcastable = eff_shape_u32
            .iter()
            .zip(input_shape.iter())
            .all(|(&p, &i)| p == 1 || p == i);
        if broadcastable {
            return Ok(cur);
        }

        let tileable = eff_shape_u32
            .iter()
            .zip(input_shape.iter())
            .all(|(&p, &i)| p > 0 && i % p == 0);
        if !tileable {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "{label}: shape {:?} is not broadcastable or block-tileable to input {:?}",
                    eff_shape_u32, input_shape
                ),
            });
        }

        // ONNX `align_param_with_input` uses Reshape+Tile so each **block** repeats consecutively
        // (e.g. `[1,2,1]` → `[1,4,1]` gives `[a,a,b,b]` along the tiled axis). Concatenating **r**
        // full copies of the param yields `[a,b,a,b]`, which is wrong for blockwise quantize.
        for ax in 0..input_shape.len() {
            let target = input_shape[ax] as i64;
            let cur_dims = cur
                .dimensions(&*network)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{label}: block tile cur dimensions: {e}"),
                })?;
            let p = cur_dims[ax];
            if p == target {
                continue;
            }
            if p <= 0 || target % p != 0 {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!(
                        "{label}: block tile ax={ax}: dim {p} cannot expand to {target}",
                    ),
                });
            }
            let r_us = (target / p) as usize;
            if r_us <= 1 {
                continue;
            }
            let rank = cur_dims.len();
            let p_us = p as usize;
            let mut expanded_parts: Vec<trtx::Tensor<'a>> = Vec::with_capacity(p_us);
            for k in 0..p_us {
                let mut start = vec![0_i64; rank];
                start[ax] = k as i64;
                let mut size = cur_dims.clone();
                size[ax] = 1;
                let stride = vec![1_i64; rank];
                let slice_out = network
                    .add_slice(&cur, &start, &size, &stride)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("{label}: block tile slice ax={ax} k={k}: {e}"),
                    })?
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("{label}: block tile slice output ax={ax} k={k}: {e}"),
                    })?;
                let dup_inputs: Vec<&trtx::Tensor<'a>> = (0..r_us).map(|_| &slice_out).collect();
                let mut cat = network.add_concatenation(&dup_inputs).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("{label}: block tile dup concat ax={ax} k={k}: {e}"),
                    }
                })?;
                cat.set_axis(network, ax as i32);
                expanded_parts.push(cat.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("{label}: block tile dup out ax={ax} k={k}: {e}"),
                    }
                })?);
            }
            let part_refs: Vec<&trtx::Tensor<'a>> = expanded_parts.iter().collect();
            let mut cat2 = network.add_concatenation(&part_refs).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{label}: block tile merge concat ax={ax}: {e}"),
                }
            })?;
            cat2.set_axis(network, ax as i32);
            cur = cat2
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{label}: block tile merge out ax={ax}: {e}"),
                })?;
        }

        Ok(cur)
    }

    #[inline]
    fn trtx_tensor_dims_empty<'a>(
        network: &trtx::NetworkDefinition<'a>,
        tensor: &trtx::Tensor<'a>,
    ) -> Result<bool, GraphError> {
        let dims = tensor
            .dimensions(network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("tensor dimensions: {e}"),
            })?;
        Ok(dims.is_empty())
    }

    /// `quantizeLinear` via elementwise ops (ONNX: `clip(round(x / scale) + zero_point, qmin, qmax)`).
    /// Used when any tensor is rank-0: `IQuantizeLayer` plus shuffles can hit a Myelin assert, and this
    /// path wires `zero_point` so no 0D network input stays unused.
    fn add_quantize_linear_elementwise_manual<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        input: &trtx::Tensor<'a>,
        scale: &trtx::Tensor<'a>,
        zero_point: &trtx::Tensor<'a>,
        fp_trt: TrtDataType,
        out_trt: TrtDataType,
        out_dtype: DataType,
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        let (qmin_f32, qmax_f32) = match out_dtype {
            DataType::Int8 => (-128.0_f32, 127.0_f32),
            DataType::Uint8 => (0.0_f32, 255.0_f32),
            DataType::Int4 => (-8.0_f32, 7.0_f32),
            DataType::Uint4 => (0.0_f32, 15.0_f32),
            DataType::Int32 => (i32::MIN as f32, i32::MAX as f32),
            DataType::Uint32 => (0.0_f32, u32::MAX as f32),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!(
                        "quantizeLinear elementwise fallback: unsupported output dtype {:?}",
                        out_dtype
                    ),
                });
            }
        };

        // 32-bit quantize output: `round(x/scale)+zp` can be far outside float16 range; qmin/qmax as
        // f16 do not represent i32::MIN/MAX (clamp breaks → INT_MIN). Run div/round/sum/clamp in f32.
        let compute_fp = if matches!(out_dtype, DataType::Int32 | DataType::Uint32) {
            TrtDataType::kFLOAT
        } else {
            fp_trt
        };

        let input_q = Self::trtx_qdq_ensure_rank1(network, input, "quantizeLinear input")?;
        let scale_q = Self::trtx_qdq_ensure_rank1(network, scale, "quantizeLinear scale")?;
        // Zero_point is already block-aligned; int8/uint8 zp was dequantized in
        // `trtx_align_quantize_param_for_elementwise` before any shuffle.
        let zp_q = Self::trtx_qdq_ensure_rank1(network, zero_point, "quantizeLinear zero_point")?;

        let mut cast_if_needed =
            |t: trtx::Tensor<'a>, lab: &str| -> Result<trtx::Tensor<'a>, GraphError> {
                if t.get_type(&*network) == compute_fp {
                    Ok(t)
                } else {
                    network
                        .add_cast(&t, compute_fp)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("quantizeLinear manual {lab} to compute fp: {e}"),
                        })?
                        .get_output(&*network, 0)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("quantizeLinear manual {lab} cast out: {e}"),
                        })
                }
            };
        let input_e = cast_if_needed(input_q, "input")?;
        let scale_e = cast_if_needed(scale_q, "scale")?;
        let zp_fp = cast_if_needed(zp_q, "zero_point")?;

        let div_out = network
            .add_elementwise(&input_e, &scale_e, ElementWiseOperation::kDIV)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear manual div: {e}"),
            })?
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear manual div output: {e}"),
            })?;

        let rounded = network
            .add_unary(&div_out, UnaryOperation::kROUND)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear manual round: {e}"),
            })?
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear manual round output: {e}"),
            })?;

        let summed = network
            .add_elementwise(&rounded, &zp_fp, ElementWiseOperation::kSUM)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear manual add zero_point: {e}"),
            })?
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear manual add output: {e}"),
            })?;

        // qmin/qmax must broadcast with `summed`. A single `[1]` constant is rank-1 and TRT rejects
        // elementwise with e.g. `[3,2]` (`nbDims` mismatch). Use `[1; rank]` so broadcasting applies.
        let summed_dims =
            summed
                .dimensions(&*network)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("quantizeLinear manual summed dimensions: {e}"),
                })?;
        let broadcast_shape: Vec<i64> = vec![1_i64; summed_dims.len()];

        let (min_bytes, max_bytes) = match compute_fp {
            TrtDataType::kFLOAT => (
                qmin_f32.to_le_bytes().to_vec(),
                qmax_f32.to_le_bytes().to_vec(),
            ),
            TrtDataType::kHALF => (
                f16::from_f32(qmin_f32).to_le_bytes().to_vec(),
                f16::from_f32(qmax_f32).to_le_bytes().to_vec(),
            ),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!(
                        "quantizeLinear manual clamp: unsupported compute type {:?}",
                        compute_fp
                    ),
                });
            }
        };

        let min_t = network
            .add_small_constant_copied(&broadcast_shape, &min_bytes, compute_fp)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear manual qmin constant: {e}"),
            })?
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear manual qmin output: {e}"),
            })?;

        let max_t = network
            .add_small_constant_copied(&broadcast_shape, &max_bytes, compute_fp)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear manual qmax constant: {e}"),
            })?
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear manual qmax output: {e}"),
            })?;

        let clamped_lo = network
            .add_elementwise(&summed, &min_t, ElementWiseOperation::kMAX)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear manual clamp max(qmin): {e}"),
            })?
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear manual clamp lo output: {e}"),
            })?;

        let clamped = network
            .add_elementwise(&clamped_lo, &max_t, ElementWiseOperation::kMIN)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear manual clamp min(qmax): {e}"),
            })?
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear manual clamp hi output: {e}"),
            })?;

        let out = network
            .add_cast(&clamped, out_trt)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear manual output cast: {e}"),
            })?
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear manual output: {e}"),
            })?;

        Ok(out)
    }

    /// Add quantizeLinear operation (float to quantized integer)
    fn add_quantize_linear_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let ins = operation.input_operands();
        let in_id = ins[0];
        let sc_id = ins[1];

        let output_id = operation.output_operands_slice()[0];
        let out_operand = graph
            .operand(output_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear output operand {output_id} not found"),
            })?;
        let out_dtype = out_operand.descriptor.data_type;
        let out_trt = Self::webnn_to_trt_dtype(out_dtype)?;

        let in_operand = graph
            .operand(ins[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("quantizeLinear input operand {} not found", ins[0]),
            })?;
        let fp_trt = Self::webnn_to_trt_dtype(in_operand.descriptor.data_type)?;
        if fp_trt != TrtDataType::kFLOAT && fp_trt != TrtDataType::kHALF {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "quantizeLinear expects float16 or float32 input, got {:?}",
                    in_operand.descriptor.data_type
                ),
            });
        }

        let in_rank0 = {
            let t = tensor_map
                .get(&in_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Input operand {in_id} not found"),
                })?;
            Self::trtx_tensor_dims_empty(&*network, t)?
        };
        let sc_rank0 = {
            let t = tensor_map
                .get(&sc_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Scale operand {sc_id} not found"),
                })?;
            Self::trtx_tensor_dims_empty(&*network, t)?
        };
        let zp_rank0 = if ins.len() > 2 {
            let zid = ins[2];
            let zp = tensor_map
                .get(&zid)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Zero point operand {zid} not found"),
                })?;
            Self::trtx_tensor_dims_empty(&*network, zp)?
        } else {
            false
        };

        // IQuantizeLayer has no zero-point input and rejects UInt8 output; Int4/Uint4 and 32-bit
        // integer outputs use the elementwise path (explicit zero-point and clamp range).
        let use_manual = in_rank0
            || sc_rank0
            || zp_rank0
            || ins.len() > 2
            || matches!(
                out_dtype,
                DataType::Uint8
                    | DataType::Int4
                    | DataType::Uint4
                    | DataType::Int32
                    | DataType::Uint32
            );

        if use_manual {
            let in_shape = in_operand.descriptor.static_or_max_shape();
            let out = if ins.len() > 2 {
                let zid = ins[2];
                let input = tensor_map
                    .get(&in_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Input operand {in_id} not found"),
                    })?;
                let scale = tensor_map
                    .get(&sc_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Scale operand {sc_id} not found"),
                    })?;
                let zp = tensor_map
                    .get(&zid)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Zero point operand {zid} not found"),
                    })?;
                let sc_operand =
                    graph
                        .operand(sc_id)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!(
                                "quantizeLinear scale operand {sc_id} missing from graph"
                            ),
                        })?;
                let z_operand = graph
                    .operand(zid)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!(
                            "quantizeLinear zero_point operand {zid} missing from graph"
                        ),
                    })?;
                let sc_shape = sc_operand.descriptor.static_or_max_shape();
                let z_shape = z_operand.descriptor.static_or_max_shape();
                let scale_a = Self::trtx_align_quantize_param_for_elementwise(
                    network,
                    scale,
                    &sc_shape,
                    &in_shape,
                    "quantizeLinear scale",
                    None,
                    None,
                )?;
                let zp_a = Self::trtx_align_quantize_param_for_elementwise(
                    network,
                    zp,
                    &z_shape,
                    &in_shape,
                    "quantizeLinear zero_point",
                    Some(fp_trt),
                    Some(z_operand.descriptor.data_type),
                )?;
                Self::add_quantize_linear_elementwise_manual(
                    network, input, &scale_a, &zp_a, fp_trt, out_trt, out_dtype,
                )?
            } else {
                let (zdt, zp_bytes): (TrtDataType, &[u8]) = match out_dtype {
                    DataType::Int8 => (TrtDataType::kINT8, &[0u8][..]),
                    DataType::Uint8 => (TrtDataType::kUINT8, &[0u8][..]),
                    // Scalar default zp `[0]` must not use kINT4 with `add_small_constant_copied` (trtx-rs
                    // byte-size check). kINT8 zero is fine: manual path casts zp to float.
                    DataType::Int4 => (TrtDataType::kINT8, &[0u8][..]),
                    DataType::Uint4 => (TrtDataType::kUINT8, &[0u8][..]),
                    DataType::Int32 | DataType::Uint32 => {
                        (TrtDataType::kINT32, &[0u8, 0u8, 0u8, 0u8][..])
                    }
                    _ => {
                        return Err(GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!(
                                "quantizeLinear default zero point: unsupported output dtype {:?}",
                                out_dtype
                            ),
                        });
                    }
                };
                let input = tensor_map
                    .get(&in_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Input operand {in_id} not found"),
                    })?;
                let scale = tensor_map
                    .get(&sc_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Scale operand {sc_id} not found"),
                    })?;
                let zp_const = network
                    .add_small_constant_copied(&[1_i64], zp_bytes, zdt)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("quantizeLinear default zero_point constant: {e}"),
                    })?
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("quantizeLinear default zero_point output: {e}"),
                    })?;
                let sc_operand =
                    graph
                        .operand(sc_id)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!(
                                "quantizeLinear scale operand {sc_id} missing from graph"
                            ),
                        })?;
                let sc_shape = sc_operand.descriptor.static_or_max_shape();
                let scale_a = Self::trtx_align_quantize_param_for_elementwise(
                    network,
                    scale,
                    &sc_shape,
                    &in_shape,
                    "quantizeLinear scale",
                    None,
                    None,
                )?;
                let zp_a = Self::trtx_align_quantize_param_for_elementwise(
                    network,
                    &zp_const,
                    &[1],
                    &in_shape,
                    "quantizeLinear zero_point",
                    Some(fp_trt),
                    Some(out_dtype),
                )?;
                Self::add_quantize_linear_elementwise_manual(
                    network, input, &scale_a, &zp_a, fp_trt, out_trt, out_dtype,
                )?
            };
            // Elementwise path uses `trtx_qdq_ensure_rank1`, so scalar inputs become `[1]`. WebNN
            // output rank must match the float input (e.g. `[]` not `[1]`).
            let out = if in_shape.is_empty() {
                let mut sh =
                    network
                        .add_shuffle(&out)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("quantizeLinear scalar output shuffle: {e}"),
                        })?;
                sh.set_reshape_dimensions(network, &[]).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("quantizeLinear scalar output reshape: {e}"),
                    }
                })?;
                sh.get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("quantizeLinear scalar shuffle output: {e}"),
                    })?
            } else {
                out
            };
            tensor_map.insert(output_id, out);
            return Ok(());
        }

        let input = tensor_map
            .get(&in_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {in_id} not found"),
            })?;
        let scale = tensor_map
            .get(&sc_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Scale operand {sc_id} not found"),
            })?;

        let input_q = Self::trtx_qdq_ensure_rank1(network, input, "quantizeLinear input")?;
        let scale_q = Self::trtx_qdq_ensure_rank1(network, scale, "quantizeLinear scale")?;

        let layer = network
            .add_quantize(&input_q, &scale_q, out_trt)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add quantize layer: {}", e),
            })?;

        let output = layer
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get quantize output: {}", e),
            })?;

        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add dequantizeLinear operation (quantized integer to float).
    ///
    /// TensorRT `IDequantizeLayer` is `(input - zeroPt) * scale` when zero-point is wired, with tight
    /// constraints on `zeroPt`. For a general WebNN zero-point tensor, use implicit zero-point 0
    /// (`input * scale`) and subtract `cast(zero_point) * scale`.
    fn add_dequantize_linear_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let ins = operation.input_operands();
        let in_id = ins[0];
        let sc_id = ins[1];

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        let out_operand = graph
            .operand(output_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dequantizeLinear output operand {output_id} not found"),
            })?;
        let out_trt = match out_operand.descriptor.data_type {
            DataType::Float32 => TrtDataType::kFLOAT,
            DataType::Float16 => TrtDataType::kHALF,
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!(
                        "dequantizeLinear expects float16 or float32 output, got {:?}",
                        out_operand.descriptor.data_type
                    ),
                });
            }
        };

        let input_operand = graph
            .operand(in_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dequantizeLinear input operand {in_id} not found"),
            })?;
        let scale_operand = graph
            .operand(sc_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dequantizeLinear scale operand {sc_id} not found"),
            })?;
        let fp_trt = match scale_operand.descriptor.data_type {
            DataType::Float32 => TrtDataType::kFLOAT,
            DataType::Float16 => TrtDataType::kHALF,
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!(
                        "dequantizeLinear expects float16 or float32 scale, got {:?}",
                        scale_operand.descriptor.data_type
                    ),
                });
            }
        };

        let input = tensor_map
            .get(&in_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {in_id} not found"),
            })?;

        let scale = tensor_map
            .get(&sc_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Scale operand {sc_id} not found"),
            })?;

        let input_dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dequantizeLinear input dimensions: {e}"),
            })?;
        let scale_dims = scale
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dequantizeLinear scale dimensions: {e}"),
            })?;
        let in_shape_web = input_operand.descriptor.static_or_max_shape();
        let sc_shape_web = scale_operand.descriptor.static_or_max_shape();
        // Per-tensor 0D: DO NOT insert Shuffle before DQ (QDQ fusion needs int8 binding -> DQ).
        // TensorRT may report scalars as rank-1 while WebNN uses `[]`, so trust graph shapes too.
        let scalar_dq = (in_shape_web.is_empty() && sc_shape_web.is_empty())
            || (input_dims.is_empty() && scale_dims.is_empty());

        let input_q_holder;
        let scale_q_holder;
        // `scale_for_zp` must be assigned next to `scale_q_holder` so the compiler proves it is
        // only `&scale_q_holder` when that binding was initialized (nested `if ins.len() > 2` loses
        // the `scalar_dq` / init correlation for `let scale_q_holder;` otherwise -> E0381).
        let scale_for_zp: &trtx::Tensor<'a>;
        let (input_for_dq, scale_for_dq): (&trtx::Tensor<'a>, &trtx::Tensor<'a>) = if scalar_dq {
            scale_for_zp = scale;
            (input, scale)
        } else {
            input_q_holder = Self::trtx_qdq_ensure_rank1(network, input, "dequantizeLinear input")?;
            scale_q_holder = Self::trtx_qdq_ensure_rank1(network, scale, "dequantizeLinear scale")?;
            scale_for_zp = &scale_q_holder;
            (&input_q_holder, &scale_q_holder)
        };

        let dq_layer = network
            .add_dequantize(input_for_dq, scale_for_dq, out_trt)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add dequantize layer: {}", e),
            })?;

        let mut dq_out =
            dq_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get dequantize output: {}", e),
                })?;

        if ins.len() > 2 {
            let z_id = ins[2];
            let zp = tensor_map
                .get(&z_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Zero point operand {z_id} not found"),
                })?;
            let z_operand = graph
                .operand(z_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("dequantizeLinear zero_point operand {z_id} not in graph"),
                })?;
            // kINT8/kUINT8 constants must feed Constant -> DQ before other layers (QDQ validator).
            let zp_q = if matches!(
                zp.get_type(&*network),
                TrtDataType::kINT8 | TrtDataType::kUINT8
            ) {
                let zp_dq = Self::trtx_int8_uint8_identity_dequantize(
                    network,
                    zp,
                    fp_trt,
                    "dequantizeLinear zero_point",
                    z_operand.descriptor.data_type,
                )?;
                let zp_shape_web = z_operand.descriptor.static_or_max_shape();
                if scalar_dq && zp_shape_web.is_empty() {
                    zp_dq
                } else {
                    Self::trtx_qdq_ensure_rank1(network, &zp_dq, "dequantizeLinear zero_point")?
                }
            } else if scalar_dq
                && z_operand.descriptor.static_or_max_shape().is_empty()
                && Self::trtx_tensor_dims_empty(network, zp)?
            {
                let id = network
                    .add_identity(zp)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("dequantizeLinear zero_point identity: {e}"),
                    })?;
                id.get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("dequantizeLinear zero_point identity output: {e}"),
                    })?
            } else {
                Self::trtx_qdq_ensure_rank1(network, zp, "dequantizeLinear zero_point")?
            };
            let zp_fp_layer =
                network
                    .add_cast(&zp_q, fp_trt)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("dequantizeLinear zero_point cast: {}", e),
                    })?;
            let zp_fp =
                zp_fp_layer
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("dequantizeLinear zero_point cast output: {}", e),
                    })?;
            let zp_times_scale = network
                .add_elementwise(&zp_fp, scale_for_zp, ElementWiseOperation::kPROD)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("dequantizeLinear zero_point * scale: {}", e),
                })?
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("dequantizeLinear zero_point * scale output: {}", e),
                })?;
            dq_out = network
                .add_elementwise(&dq_out, &zp_times_scale, ElementWiseOperation::kSUB)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("dequantizeLinear subtract zp*scale: {}", e),
                })?
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("dequantizeLinear corrected output: {}", e),
                })?;
        }

        tensor_map.insert(output_id, dq_out);
        Ok(())
    }

    /// Add global pooling operation
    fn add_global_pooling_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
        pool_type: PoolingType,
    ) -> Result<(), GraphError> {
        let (input_id, opts_ref) = match operation {
            Operation::GlobalAveragePool { input, options, .. } => (*input, options.as_ref()),
            Operation::GlobalMaxPool { input, options, .. } => (*input, options.as_ref()),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "add_global_pooling_op: expected globalAveragePool or globalMaxPool"
                        .to_string(),
                });
            }
        };

        let default_pool = MLPool2dOptions::default();
        let opts = opts_ref.unwrap_or(&default_pool);
        let layout = match opts.layout.as_str() {
            "" => "nchw",
            s => s,
        };

        let input = tensor_map
            .get(&input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", input_id),
            })?;

        let nhwc_to_nchw: Option<trtx::Tensor<'a>> =
            if layout == "nhwc" {
                let mut shuffle =
                    network
                        .add_shuffle(input)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("globalPool NHWC->NCHW shuffle: {e}"),
                        })?;
                shuffle
                    .set_first_transpose(network, &[0, 3, 1, 2])
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("globalPool set_first_transpose NHWC: {e}"),
                    })?;
                Some(shuffle.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("globalPool NHWC shuffle output: {e}"),
                    }
                })?)
            } else {
                None
            };
        let pool_in = nhwc_to_nchw.as_ref().unwrap_or(input);

        let input_dims =
            pool_in
                .dimensions(&*network)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get input dimensions: {}", e),
                })?;

        if input_dims.len() < 4 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Global pooling requires 4D input, got {}D",
                    input_dims.len()
                ),
            });
        }

        let window: [i64; 2] = [input_dims[2], input_dims[3]];

        let layer = network
            .add_pooling(pool_in, pool_type, &window)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add global pooling: {}", e),
            })?;

        let pooled_nchw =
            layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get layer output: {}", e),
                })?;

        let output = if layout == "nhwc" {
            let mut shuffle =
                network
                    .add_shuffle(&pooled_nchw)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("globalPool NCHW->NHWC shuffle: {e}"),
                    })?;
            shuffle
                .set_first_transpose(network, &[0, 2, 3, 1])
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("globalPool set_first_transpose NCHW: {e}"),
                })?;
            shuffle
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("globalPool NCHW shuffle output: {e}"),
                })?
        } else {
            pooled_nchw
        };

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add matrix multiply operation.
    /// TensorRT IMatrixMultiplyLayer requires both inputs to have the same number of dimensions;
    /// if ranks differ, unsqueeze the lower-rank input by prepending 1s.
    fn add_matmul_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input0 = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let input1 = tensor_map
            .get(&operation.input_operands()[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[1]),
            })?;

        let dims0 = input0
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Matmul: input0 dimensions: {}", e),
            })?;
        let dims1 = input1
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Matmul: input1 dimensions: {}", e),
            })?;

        let rank0 = dims0.len();
        let rank1 = dims1.len();

        let layer = if rank0 == rank1 {
            network.add_matrix_multiply(
                input0,
                MatrixOperation::kNONE,
                input1,
                MatrixOperation::kNONE,
            )
        } else if rank0 < rank1 {
            let reshape_dims: Vec<i64> = dims0.iter().copied().collect();
            let rank_diff = rank1 - rank0;
            let mut new_shape: Vec<i64> = vec![1i64; rank_diff];
            new_shape.extend(reshape_dims);
            let mut shuffle =
                network
                    .add_shuffle(input0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Matmul: unsqueeze shuffle: {}", e),
                    })?;
            shuffle
                .set_reshape_dimensions(network, &new_shape)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Matmul: set reshape: {}", e),
                })?;
            let reshaped0 =
                shuffle
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Matmul: shuffle output: {}", e),
                    })?;
            network.add_matrix_multiply(
                &reshaped0,
                MatrixOperation::kNONE,
                input1,
                MatrixOperation::kNONE,
            )
        } else {
            let reshape_dims: Vec<i64> = dims1.iter().copied().collect();
            let rank_diff = rank0 - rank1;
            let mut new_shape: Vec<i64> = vec![1i64; rank_diff];
            new_shape.extend(reshape_dims);
            let mut shuffle =
                network
                    .add_shuffle(input1)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Matmul: unsqueeze shuffle: {}", e),
                    })?;
            shuffle
                .set_reshape_dimensions(network, &new_shape)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Matmul: set reshape: {}", e),
                })?;
            let reshaped1 =
                shuffle
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Matmul: shuffle output: {}", e),
                    })?;
            network.add_matrix_multiply(
                input0,
                MatrixOperation::kNONE,
                &reshaped1,
                MatrixOperation::kNONE,
            )
        }
        .map_err(|e| GraphError::ConversionFailed {
            format: "trtx".to_string(),
            reason: format!("Failed to add matrix multiply: {}", e),
        })?;

        let output = layer
            .get_output(&*network, 0)
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
    fn reshape_batch_norm_stats_for_broadcast<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        stats: &trtx::Tensor<'a>,
        input: &trtx::Tensor<'a>,
        axis: i64,
        op_name: &str,
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        let input_dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get input dims for {}: {}", op_name, e),
            })?;
        let stats_dims = stats
            .dimensions(&*network)
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
        let c: i64 = stats_dims.iter().product::<i64>().max(1);
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
            shuffle.set_first_transpose(network, &perm).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to set transpose for {}: {}", op_name, e),
                }
            })?;
            return shuffle
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get shuffle output for {}: {}", op_name, e),
                });
        }

        // For 1D [C], reshape to [1,1,...,1,C] then transpose so C moves to axis_idx; avoids TensorRT giving [C,1,1,1].
        let (target_shape, need_second_transpose) = if stats_dims.len() == 1 {
            let shape_last: Vec<i64> = (0..rank)
                .map(|i| if i == rank - 1 { c } else { 1i64 })
                .collect();
            (shape_last, true)
        } else {
            let shape_axis: Vec<i64> = (0..rank)
                .map(|i| if i as i64 == axis_idx { c } else { 1i64 })
                .collect();
            (shape_axis, false)
        };
        let mut shuffle = network
            .add_shuffle(stats)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add shuffle for {}: {}", op_name, e),
            })?;
        shuffle
            .set_reshape_dimensions(network, &target_shape)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set reshape for {}: {}", op_name, e),
            })?;
        let mut result =
            shuffle
                .get_output(&*network, 0)
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
            shuffle2.set_first_transpose(network, &perm).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to set second transpose for {}: {}", op_name, e),
                }
            })?;
            result =
                shuffle2
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!(
                            "Failed to get second shuffle output for {}: {}",
                            op_name, e
                        ),
                    })?;
        }
        Ok(result)
    }

    /// Add batch normalization operation
    /// Formula: y = (x - mean) / sqrt(variance + epsilon) * scale + bias
    fn add_batch_normalization_op<'a>(
        _graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        // Input operands: input, mean, variance, scale (optional), bias (optional)
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let mean = tensor_map
            .get(&operation.input_operands()[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Mean operand {} not found", operation.input_operands()[1]),
            })?;

        let variance = tensor_map
            .get(&operation.input_operands()[2])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Variance operand {} not found",
                    operation.input_operands()[2]
                ),
            })?;

        // Read typed batchNormalization options (fallback keeps WebNN defaults).
        let (axis, _epsilon) = operation
            .attributes()
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
        let _ = sub_layer.set_name(network, "batch_norm_sub");

        let x_minus_mean =
            sub_layer
                .get_output(&*network, 0)
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
        let _ = sqrt_var_layer.set_name(network, "batch_norm_sqrt_var");

        let sqrt_var =
            sqrt_var_layer
                .get_output(&*network, 0)
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
        let _ = div_layer.set_name(network, "batch_norm_div");

        let normalized =
            div_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get div output: {}", e),
                })?;

        // Step 5: Apply scale if present (WebNN: scale is in MLBatchNormalizationOptions, not a positional input)
        let scale_id = operation
            .attributes()
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
            let _ = mul_layer.set_name(network, "batch_norm_scale_mul");

            result =
                mul_layer
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get mul output: {}", e),
                    })?;
        }

        // Step 6: Apply bias if present (WebNN: bias is in MLBatchNormalizationOptions, not a positional input)
        let bias_id = operation
            .attributes()
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
            let _ = add_layer.set_name(network, "batch_norm_bias_add");

            result =
                add_layer
                    .get_output(&*network, 0)
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
    fn add_instance_normalization_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        // Instance normalization computes statistics per-instance (N, C) over spatial dims
        // Input operands: input, scale (optional), bias (optional)
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let input_dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("InstanceNorm: failed to get input dimensions: {}", e),
            })?;

        // Get epsilon from attributes (default: 1e-5), cast to input dtype per spec
        let attrs = operation.attributes();
        let opts = attrs.as_instance_normalization();
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
            .add_reduce(
                input,
                ReduceOperation::kAVG.into(),
                Axes::from_bits(axes_mask),
                true,
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add mean reduce for instance norm: {}", e),
            })?;

        let mean =
            mean_layer
                .get_output(&*network, 0)
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

        let x_minus_mean =
            sub_layer
                .get_output(&*network, 0)
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

        let squared =
            square_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get square output: {}", e),
                })?;

        // variance = mean((x - mean)^2)
        let var_layer = network
            .add_reduce(
                &squared,
                ReduceOperation::kAVG.into(),
                Axes::from_bits(axes_mask),
                true,
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add variance reduce for instance norm: {}", e),
            })?;

        let variance =
            var_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get variance output: {}", e),
                })?;

        // variance + epsilon per WebNN spec (epsilon cast to input's dataType)
        let var_dims =
            variance
                .dimensions(&*network)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("InstanceNorm: failed to get variance dimensions: {}", e),
                })?;
        let input_operand = graph
            .operand(operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "InstanceNorm: input operand {} not found in graph",
                    operation.input_operands()[0]
                ),
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
        let var_shape: Vec<i64> = var_dims.iter().map(|&d| d as i64).collect();
        let epsilon_const = network
            .add_small_constant_copied(&var_shape, &epsilon_bytes, trt_dtype)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("InstanceNorm: failed to add epsilon constant: {}", e),
            })?;
        let epsilon_out =
            epsilon_const
                .get_output(&*network, 0)
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
                .get_output(&*network, 0)
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

        let std_dev =
            sqrt_layer
                .get_output(&*network, 0)
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

        let mut result =
            div_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get div output: {}", e),
                })?;

        // Build broadcast shape for scale/bias [C] to match input (TensorRT elementwise needs same rank).
        let scale_broadcast_shape: Vec<i64> = if layout == "nchw" && !input_dims.is_empty() {
            let mut s = vec![1i64; input_dims.len()];
            s[1] = input_dims[1];
            s
        } else if !input_dims.is_empty() {
            let mut s = vec![1i64; input_dims.len()];
            let last = input_dims.len() - 1;
            s[last] = input_dims[last];
            s
        } else {
            vec![1]
        };

        // Scale/bias operand indices (WebNN MLInstanceNormalizationOptions); optional extras on legacy graphs.
        let mut scale_id = opts.and_then(|o| o.scale);
        let mut bias_id = opts.and_then(|o| o.bias);
        if scale_id.is_none() || bias_id.is_none() {
            for &operand_id in &operation.input_operands()[1..] {
                let name = graph
                    .operand(operand_id)
                    .and_then(|o| o.name.as_deref())
                    .unwrap_or("");
                let name_lower = name.to_lowercase();
                if scale_id.is_none() && name_lower.contains("scale") {
                    scale_id = Some(operand_id);
                } else if bias_id.is_none() && name_lower.contains("bias") {
                    bias_id = Some(operand_id);
                }
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
            let scale_broadcast_i64: Vec<i64> =
                scale_broadcast_shape.iter().map(|&d| d as i64).collect();
            scale_shuffle
                .set_reshape_dimensions(network, &scale_broadcast_i64)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("InstanceNorm: failed to set scale reshape: {}", e),
                })?;
            let scale_bc = scale_shuffle.get_output(&*network, 0).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("InstanceNorm: failed to get scale shuffle output: {}", e),
                }
            })?;

            let mul_layer = network
                .add_elementwise(&result, &scale_bc, ElementWiseOperation::kPROD)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add mul for scale: {}", e),
                })?;

            result =
                mul_layer
                    .get_output(&*network, 0)
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
            let bias_broadcast_i64: Vec<i64> =
                scale_broadcast_shape.iter().map(|&d| d as i64).collect();
            bias_shuffle
                .set_reshape_dimensions(network, &bias_broadcast_i64)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("InstanceNorm: failed to set bias reshape: {}", e),
                })?;
            let bias_bc = bias_shuffle.get_output(&*network, 0).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("InstanceNorm: failed to get bias shuffle output: {}", e),
                }
            })?;

            let add_layer = network
                .add_elementwise(&result, &bias_bc, ElementWiseOperation::kSUM)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add bias: {}", e),
                })?;

            result =
                add_layer
                    .get_output(&*network, 0)
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
    fn add_layer_normalization_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        // Layer normalization computes statistics over specified axes
        // Input operands: input, scale (optional), bias (optional)
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let input_dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get input shape: {}", e),
            })?;

        // WebNN: scale and bias live in MLLayerNormalizationOptions, not in input_operands (only the input tensor is listed).
        let (scale_operand_id, bias_operand_id, epsilon, axes_from_options) = match operation {
            Operation::LayerNormalization { options, .. } => match options.as_ref() {
                Some(o) => (o.scale, o.bias, o.epsilon as f32, o.axes.clone()),
                None => (None, None, 1e-5_f32, None),
            },
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "add_layer_normalization_op: expected LayerNormalization operation"
                        .to_string(),
                });
            }
        };

        // TensorRT Reduce requires at least 1 dimension. For 0D scalar: mean=x, variance=0, output = 0*scale + bias = bias or 0.
        if input_dims.is_empty() {
            let input_operand = graph
                .operand(operation.input_operands()[0])
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!(
                        "Input operand {} not found in graph",
                        operation.input_operands()[0]
                    ),
                })?;
            let (zero_bytes, zero_dtype) = match input_operand.descriptor.data_type {
                DataType::Float16 => (
                    f16::from_f32(0.0).to_bits().to_le_bytes().to_vec(),
                    TrtDataType::kHALF,
                ),
                _ => (0.0f32.to_le_bytes().to_vec(), TrtDataType::kFLOAT),
            };
            let zero_const = network
                .add_small_constant_copied(&[], &zero_bytes, zero_dtype)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("LayerNorm 0D: failed to add zero constant: {}", e),
                })?;
            let mut result =
                zero_const
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm 0D: zero const output: {}", e),
                    })?;
            if let Some(bias_id) = bias_operand_id {
                let bias =
                    tensor_map
                        .get(&bias_id)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Bias operand {} not found", bias_id),
                        })?;
                // Bias may be [1] or scalar; reshape to rank-0 so output matches WebNN 0D (see quantizeLinear scalar shuffle).
                let mut bias_shuffle =
                    network
                        .add_shuffle(bias)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("LayerNorm 0D: failed to add bias shuffle: {}", e),
                        })?;
                bias_shuffle
                    .set_reshape_dimensions(network, &[])
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm 0D: failed to set bias reshape: {}", e),
                    })?;
                let bias_bc = bias_shuffle.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm 0D: bias shuffle output: {}", e),
                    }
                })?;
                let add_layer = network
                    .add_elementwise(&result, &bias_bc, ElementWiseOperation::kSUM)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm 0D: failed to add bias: {}", e),
                    })?;
                result = add_layer.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm 0D: bias add output: {}", e),
                    }
                })?;
            }
            let output_id = operation.output_operands_slice()[0];
            tensor_map.insert(output_id, result);
            return Ok(());
        }

        // Get epsilon and axes from typed options. Spec: when axes not present, axes = [1..rank) if rank > 1 else [].
        // Option<axes>: None = key omitted => default; Some(v) = use v (Some([]) = explicit no reduction).
        let axes: Vec<u32> = axes_from_options.unwrap_or_else(|| {
            if input_dims.len() > 1 {
                (1..input_dims.len()).map(|i| i as u32).collect()
            } else {
                vec![]
            }
        });

        // Spec: "If empty, no dimensions are reduced." TensorRT Reduce requires at least one dimension to reduce.
        // When axes=[], mean/variance reduce over nothing -> normalized = 0; output = 0*scale + bias = bias or 0.
        if axes.is_empty() {
            let input_operand = graph
                .operand(operation.input_operands()[0])
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!(
                        "Input operand {} not found in graph",
                        operation.input_operands()[0]
                    ),
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
            let shape_i64: Vec<i64> = input_dims.iter().map(|&d| d as i64).collect();
            let zero_const = network
                .add_small_constant_copied(&shape_i64, &zero_bytes, zero_dtype.clone())
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("LayerNorm axes=[]: failed to add zeros constant: {}", e),
                })?;
            let mut result =
                zero_const
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm axes=[]: zero const output: {}", e),
                    })?;
            if let Some(bias_id) = bias_operand_id {
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
                    let bias_const = network
                        .add_small_constant_copied(&shape_i64, &bias_broadcast_bytes, zero_dtype)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!(
                                "LayerNorm axes=[]: failed to add bias constant: {}",
                                e
                            ),
                        })?;
                    bias_const.get_output(&*network, 0).map_err(|e| {
                        GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("LayerNorm axes=[]: bias const output: {}", e),
                        }
                    })?
                } else {
                    // Bias is an input (e.g. from test harness). Broadcast scalar to result shape via ensure_broadcast_compatible with ones.
                    let ones_bytes: Vec<u8> = match input_operand.descriptor.data_type {
                        DataType::Float16 => (0..num_el)
                            .flat_map(|_| f16::from_f32(1.0).to_bits().to_le_bytes())
                            .collect(),
                        _ => (0..num_el).flat_map(|_| 1.0f32.to_le_bytes()).collect(),
                    };
                    let ones_const = network
                        .add_small_constant_copied(&shape_i64, &ones_bytes, zero_dtype.clone())
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!(
                                "LayerNorm axes=[]: failed to add ones constant: {}",
                                e
                            ),
                        })?;
                    let ones_tensor = ones_const.get_output(&*network, 0).map_err(|e| {
                        GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("LayerNorm axes=[]: ones const output: {}", e),
                        }
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
                result = add_layer.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm axes=[]: bias add output: {}", e),
                    }
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
            .add_reduce(
                input,
                ReduceOperation::kAVG.into(),
                Axes::from_bits(axes_mask),
                true,
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add mean reduce for layer norm: {}", e),
            })?;

        let mean =
            mean_layer
                .get_output(&*network, 0)
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

        let x_minus_mean =
            sub_layer
                .get_output(&*network, 0)
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

        let squared =
            square_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get square output: {}", e),
                })?;

        // variance = mean((x - mean)^2)
        let var_layer = network
            .add_reduce(
                &squared,
                ReduceOperation::kAVG.into(),
                Axes::from_bits(axes_mask),
                true,
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add variance reduce for layer norm: {}", e),
            })?;

        let variance =
            var_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get variance output: {}", e),
                })?;

        // variance + epsilon per WebNN spec (then sqrt)
        let var_dims =
            variance
                .dimensions(&*network)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("LayerNorm: failed to get variance dimensions: {}", e),
                })?;
        let var_shape: Vec<i64> = var_dims.clone();
        let num_var_el: usize = var_dims.iter().map(|&d| d as usize).product();
        let input_operand = graph
            .operand(operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Input operand {} not found in graph",
                    operation.input_operands()[0]
                ),
            })?;
        let (epsilon_bytes, epsilon_dtype) = match input_operand.descriptor.data_type {
            DataType::Float16 => (
                (0..num_var_el)
                    .flat_map(|_| f16::from_f32(epsilon).to_bits().to_le_bytes())
                    .collect::<Vec<_>>(),
                TrtDataType::kHALF,
            ),
            _ => (
                (0..num_var_el)
                    .flat_map(|_| epsilon.to_le_bytes())
                    .collect::<Vec<_>>(),
                TrtDataType::kFLOAT,
            ),
        };
        let epsilon_const = network
            .add_small_constant_copied(&var_shape, &epsilon_bytes, epsilon_dtype)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("LayerNorm: failed to add epsilon constant: {}", e),
            })?;
        let epsilon_out =
            epsilon_const
                .get_output(&*network, 0)
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
                .get_output(&*network, 0)
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

        let std_dev =
            sqrt_layer
                .get_output(&*network, 0)
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

        let mut result =
            div_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get div output: {}", e),
                })?;

        // Reshape scale/bias so they broadcast to result. Scale/bias have shape [d_axes[0], d_axes[1], ...]
        // in axis order; result has input shape. Broadcast shape: for each result dim i, use
        // scale_bias dim at that axis position if i is in axes, else 1.
        let reshape_scale_bias_to_result_rank = |network: &mut trtx::NetworkDefinition<'a>,
                                                 tensor: &trtx::Tensor<'a>,
                                                 result: &trtx::Tensor<'a>,
                                                 op_name: &str,
                                                 axes: &[u32]|
         -> Result<trtx::Tensor<'a>, GraphError> {
            let result_dims =
                result
                    .dimensions(&*network)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm {}: result dims: {}", op_name, e),
                    })?;
            let tensor_dims =
                tensor
                    .dimensions(&*network)
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
                return id_layer.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm {}: identity output: {}", op_name, e),
                    }
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
                    .set_first_transpose(network, perm.as_slice())
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("LayerNorm {}: set transpose: {}", op_name, e),
                    })?;
            }
            let new_shape_i64: Vec<i64> = new_shape.iter().map(|&d| d as i64).collect();
            shuffle
                .set_reshape_dimensions(network, &new_shape_i64)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("LayerNorm {}: set reshape: {}", op_name, e),
                })?;
            shuffle
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("LayerNorm {}: shuffle output: {}", op_name, e),
                })
        };

        if let Some(scale_id) = scale_operand_id {
            let scale = tensor_map
                .get(&scale_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("LayerNorm scale operand {} not found", scale_id),
                })?;
            let scale_bc =
                reshape_scale_bias_to_result_rank(network, scale, &result, "scale", &axes)?;

            let mul_layer = network
                .add_elementwise(&result, &scale_bc, ElementWiseOperation::kPROD)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add scale: {}", e),
                })?;
            result =
                mul_layer
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get mul output: {}", e),
                    })?;
        }

        if let Some(bias_id) = bias_operand_id {
            let bias = tensor_map
                .get(&bias_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("LayerNorm bias operand {} not found", bias_id),
                })?;
            let bias_bc = reshape_scale_bias_to_result_rank(network, bias, &result, "bias", &axes)?;

            let add_layer = network
                .add_elementwise(&result, &bias_bc, ElementWiseOperation::kSUM)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add bias: {}", e),
                })?;

            result =
                add_layer
                    .get_output(&*network, 0)
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
    fn add_reduce_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
        reduce_op: ReduceOperation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let input_dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Reduce: input dimensions: {}", e),
            })?;
        let rank = input_dims.len();

        let attrs = operation.attributes();
        let opts = attrs.as_reduce();
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
            let output =
                id_layer
                    .get_output(&*network, 0)
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
            .add_reduce(
                input,
                reduce_op.into(),
                Axes::from_bits(axes_mask),
                keep_dims,
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add reduce operation: {}", e),
            })?;

        let output = layer
            .get_output(&*network, 0)
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
    fn add_reduce_l1_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let abs_layer = network
            .add_unary(input, UnaryOperation::kABS)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add abs for L1: {}", e),
            })?;

        let abs_output =
            abs_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get abs output: {}", e),
                })?;

        let input_dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("ReduceL1: input dimensions: {}", e),
            })?;
        let rank = input_dims.len();

        let attrs = operation.attributes();
        let opts = attrs.as_reduce();
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
            .add_reduce(
                &abs_output,
                ReduceOperation::kSUM.into(),
                Axes::from_bits(axes_mask),
                keep_dims,
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add reduce for L1: {}", e),
            })?;

        let output = layer
            .get_output(&*network, 0)
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
    fn add_reduce_l2_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let input_dtype = graph
            .operand(operation.input_operands()[0])
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
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get square output: {}", e),
                })?;

        let input_dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("ReduceL2: input dimensions: {}", e),
            })?;
        let rank = input_dims.len();

        let attrs = operation.attributes();
        let opts = attrs.as_reduce();
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
            let output =
                sqrt_layer
                    .get_output(&*network, 0)
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
            .add_reduce(
                &to_reduce,
                ReduceOperation::kSUM.into(),
                Axes::from_bits(axes_mask),
                keep_dims,
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add reduce for L2: {}", e),
            })?;

        let sum_output =
            sum_layer
                .get_output(&*network, 0)
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

        let sqrt_output =
            sqrt_layer
                .get_output(&*network, 0)
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
    fn add_reduce_log_sum_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let input_dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("ReduceLogSum: input dimensions: {}", e),
            })?;
        let rank = input_dims.len();

        let attrs = operation.attributes();
        let opts = attrs.as_reduce();
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
            let output =
                log_layer
                    .get_output(&*network, 0)
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
            .add_reduce(
                input,
                ReduceOperation::kSUM.into(),
                Axes::from_bits(axes_mask),
                keep_dims,
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add reduce for LogSum: {}", e),
            })?;

        let sum_output =
            sum_layer
                .get_output(&*network, 0)
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

        let output =
            log_layer
                .get_output(&*network, 0)
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
    fn add_reduce_log_sum_exp_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let exp_layer = network
            .add_unary(input, UnaryOperation::kEXP)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add exp for LogSumExp: {}", e),
            })?;

        let exp_output =
            exp_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get exp output: {}", e),
                })?;

        let input_dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("ReduceLogSumExp: input dimensions: {}", e),
            })?;
        let rank = input_dims.len();

        let attrs = operation.attributes();
        let opts = attrs.as_reduce();
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
            let output =
                id_layer
                    .get_output(&*network, 0)
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
            .add_reduce(
                &exp_output,
                ReduceOperation::kSUM.into(),
                Axes::from_bits(axes_mask),
                keep_dims,
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add reduce for LogSumExp: {}", e),
            })?;

        let sum_output =
            sum_layer
                .get_output(&*network, 0)
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

        let output =
            log_layer
                .get_output(&*network, 0)
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
    fn add_reduce_sum_square_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let square_layer = network
            .add_elementwise(input, input, ElementWiseOperation::kPROD)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add square for SumSquare: {}", e),
            })?;

        let square_output =
            square_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get square output: {}", e),
                })?;

        let input_dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("ReduceSumSquare: input dimensions: {}", e),
            })?;
        let rank = input_dims.len();

        let attrs = operation.attributes();
        let opts = attrs.as_reduce();
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
            .add_reduce(
                &square_output,
                ReduceOperation::kSUM.into(),
                Axes::from_bits(axes_mask),
                keep_dims,
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add reduce for SumSquare: {}", e),
            })?;

        let output = layer
            .get_output(&*network, 0)
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
    fn add_slice_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let (starts_u32, sizes_ml, opts) = match operation {
            Operation::Slice {
                starts,
                sizes,
                options,
                ..
            } => (
                starts,
                sizes,
                options
                    .as_ref()
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: "Slice operation missing options".to_string(),
                    })?,
            ),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "expected Slice operation".to_string(),
                });
            }
        };
        // Empty starts/sizes: no-op (identity), e.g. 0D tensor with empty slices.
        if starts_u32.is_empty() || sizes_ml.is_empty() {
            let id_layer =
                network
                    .add_identity(input)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Slice no-op identity: {}", e),
                    })?;
            let output =
                id_layer
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Slice no-op output: {}", e),
                    })?;
            let output_id = operation.output_operands_slice()[0];
            tensor_map.insert(output_id, output);
            return Ok(());
        }
        let starts: Vec<i32> = starts_u32.iter().map(|&u| u as i32).collect();
        let sizes: Vec<i32> = sizes_ml.iter().map(|d| d.static_or_max() as i32).collect();
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
            .map(|(sz, st): (&i32, &i32)| {
                if *st == 0 {
                    0_i32 // avoid div-by-zero; validator should reject elsewhere
                } else if *st == 1 {
                    *sz
                } else {
                    // ceil(extent / stride) in integers
                    (*sz + st.abs() - 1) / st.abs()
                }
            })
            .collect();

        let starts_i64: Vec<i64> = starts.iter().map(|&x| x as i64).collect();
        let trt_sizes_i64: Vec<i64> = trt_sizes.iter().map(|&x| x as i64).collect();
        let strides_i64: Vec<i64> = strides.iter().map(|&x| x as i64).collect();

        let layer = network
            .add_slice(input, &starts_i64, &trt_sizes_i64, &strides_i64)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add slice layer: {}", e),
            })?;

        let output = layer
            .get_output(&*network, 0)
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
    fn add_split_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_id = operation.input_operands()[0];
        let input_dims = tensor_map
            .get(&input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", input_id),
            })?
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get input shape: {}", e),
            })?;

        let ndim = input_dims.len();

        let attrs = operation.attributes();
        let opts = attrs
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

        let split_sizes = match operation {
            Operation::Split { splits, .. } => splits.as_slice(),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "expected Split operation".to_string(),
                });
            }
        };
        let splits: Vec<i32> = if split_sizes.is_empty() {
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
            split_sizes.iter().map(|&u| u as i32).collect()
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

        let mut offset = 0i64;
        for (k, &size_k) in splits.iter().enumerate() {
            let mut starts = vec![0i64; ndim];
            starts[axis as usize] = offset;
            let mut sizes = input_dims.clone();
            sizes[axis as usize] = size_k as i64;
            let strides = vec![1i64; ndim];

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
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get layer output for split {}: {}", k, e),
                    })?
            };

            tensor_map.insert(output_ids[k], output);
            offset += size_k as i64;
        }

        Ok(())
    }

    /// Add squeeze operation (remove dimensions of size 1)
    fn add_squeeze_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        // Get axes from attributes (optional - if not provided, squeeze all size-1 dims)
        let _axes_opt = operation.attributes().get("axes");

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
            .get_output(&*network, 0)
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
    fn add_unsqueeze_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        // Get axes from attributes
        let axes_value =
            operation
                .attributes()
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
            .get_output(&*network, 0)
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
    fn add_expand_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let new_shape: Vec<i32> = match operation {
            Operation::Expand { new_shape, .. } => new_shape
                .iter()
                .map(|d| MLDimension::static_or_max(d) as i32)
                .collect(),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "Internal error: add_expand_op called for non-expand".to_string(),
                });
            }
        };

        let num_elements: usize = new_shape
            .iter()
            .map(|d: &i32| (*d).max(0) as usize)
            .product::<usize>()
            .max(1);

        let input_operand = graph
            .operand(operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Input operand {} not found in graph",
                    operation.input_operands()[0]
                ),
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
        let new_shape_i64: Vec<i64> = new_shape.iter().map(|&d| d as i64).collect();
        let ones_const = network
            .add_small_constant_copied(&new_shape_i64, &ones_data, trt_dtype)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create ones constant for expand: {}", e),
            })?;
        let ones_tensor =
            ones_const
                .get_output(&*network, 0)
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
        let output =
            mul_layer
                .get_output(&*network, 0)
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
    fn add_tile_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let repetitions = match operation {
            Operation::Tile { repetitions, .. } => repetitions.clone(),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "Tile operation expected".to_string(),
                });
            }
        };

        let input_id = operation.input_operands()[0];
        let input_rank = {
            let input = tensor_map
                .get(&input_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Tile: input operand {} not found", input_id),
                })?;
            input
                .dimensions(&*network)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Tile: input dimensions: {}", e),
                })?
                .len()
        };
        if repetitions.len() != input_rank {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Tile: repetitions length {} must equal input rank {} (WebNN tile)",
                    repetitions.len(),
                    input_rank
                ),
            });
        }

        // Tile by concatenating the tensor multiple times along each axis
        // We process each axis sequentially: tile axis 0, then axis 1, etc.
        let mut current_id = input_id;
        let mut produced_temp = false;

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
            let tensors_to_concat: Vec<&trtx::Tensor<'a>> =
                (0..reps).map(|_| current_tensor).collect();

            // Concatenate along this axis
            let mut concat_layer = network.add_concatenation(&tensors_to_concat).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add concatenation for tile axis {}: {}", axis, e),
                }
            })?;

            // Set the concatenation axis
            concat_layer.set_axis(network, axis as i32);

            // Get the output tensor
            let output_tensor = concat_layer.get_output(&*network, 0).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get concat output for tile axis {}: {}", axis, e),
                }
            })?;

            // Use a temporary ID for intermediate results
            // We use a large number to avoid collisions with actual operand IDs
            current_id = 1_000_000 + axis as u32;
            tensor_map.insert(current_id, output_tensor);
            produced_temp = true;
        }

        // Insert the final result with the actual output operand ID
        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];

        if produced_temp {
            let final_tensor =
                tensor_map
                    .remove(&current_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Tile: missing intermediate tensor {}", current_id),
                    })?;
            tensor_map.insert(output_id, final_tensor);
        } else {
            // No concat tiling (all reps <= 1, or rank-0 with repetitions []): keep input in map, identity to output.
            let input_tensor =
                tensor_map
                    .get(&input_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Tile: input {} missing for identity", input_id),
                    })?;
            let identity_layer =
                network
                    .add_identity(input_tensor)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Tile: identity: {}", e),
                    })?;
            let output_tensor = identity_layer.get_output(&*network, 0).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Tile: identity output: {}", e),
                }
            })?;
            tensor_map.insert(output_id, output_tensor);
        }

        Ok(())
    }

    // ============================================================================
    // Comparison Operations (2026-01-29)
    // ============================================================================

    /// Add greaterOrEqual operation (greater(x, y) OR equal(x, y))
    fn add_greater_or_equal_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input0 = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let input1 = tensor_map
            .get(&operation.input_operands()[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[1]),
            })?;

        let (bc_input0, bc_input1) =
            Self::ensure_broadcast_compatible(network, input0, input1, operation.op_type())?;

        // greaterOrEqual = greater OR equal
        let greater_layer = network
            .add_elementwise(&bc_input0, &bc_input1, ElementWiseOperation::kGREATER)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add greater layer: {}", e),
            })?;

        let greater_output =
            greater_layer
                .get_output(&*network, 0)
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

        let equal_output =
            equal_layer
                .get_output(&*network, 0)
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

        let bool_output =
            or_layer
                .get_output(&*network, 0)
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
    fn add_lesser_or_equal_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input0 = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let input1 = tensor_map
            .get(&operation.input_operands()[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[1]),
            })?;

        let (bc_input0, bc_input1) =
            Self::ensure_broadcast_compatible(network, input0, input1, operation.op_type())?;

        // lesserOrEqual = lesser OR equal
        let lesser_layer = network
            .add_elementwise(&bc_input0, &bc_input1, ElementWiseOperation::kLESS)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add lesser layer: {}", e),
            })?;

        let lesser_output =
            lesser_layer
                .get_output(&*network, 0)
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

        let equal_output =
            equal_layer
                .get_output(&*network, 0)
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

        let bool_output =
            or_layer
                .get_output(&*network, 0)
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
    fn add_not_equal_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input0 = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let input1 = tensor_map
            .get(&operation.input_operands()[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[1]),
            })?;

        let (bc_input0, bc_input1) =
            Self::ensure_broadcast_compatible(network, input0, input1, operation.op_type())?;

        // notEqual = NOT equal
        let equal_layer = network
            .add_elementwise(&bc_input0, &bc_input1, ElementWiseOperation::kEQUAL)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add equal layer: {}", e),
            })?;

        let equal_output =
            equal_layer
                .get_output(&*network, 0)
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

        let bool_output =
            not_layer
                .get_output(&*network, 0)
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
    fn add_gather_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let indices = tensor_map
            .get(&operation.input_operands()[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Indices operand {} not found",
                    operation.input_operands()[1]
                ),
            })?;

        // Get axis attribute (default to 0)
        let axis = operation
            .attributes()
            .as_gather()
            .map(|o| o.axis as i32)
            .unwrap_or(0);

        let data_operand = graph
            .operand(operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Data operand {} not found in graph",
                    operation.input_operands()[0]
                ),
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

        let indices_operand = graph
            .operand(operation.input_operands()[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Indices operand {} not found in graph",
                    operation.input_operands()[1]
                ),
            })?;
        let indices_shape: Vec<i32> = indices_operand
            .descriptor
            .shape
            .iter()
            .map(|d| get_static_or_max_size(d) as i32)
            .collect();
        let indices_shape_i64: Vec<i64> = indices_shape.iter().map(|&d| d as i64).collect();

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
        let min_const = network
            .add_small_constant_copied(&indices_shape_i64, &min_data, trtx::DataType::kINT32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add gather clamp min constant: {}", e),
            })?;
        let max_const = network
            .add_small_constant_copied(&indices_shape_i64, &max_data, trtx::DataType::kINT32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add gather clamp max constant: {}", e),
            })?;

        let min_const_out =
            min_const
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get clamp min output: {}", e),
                })?;
        let max_const_out =
            max_const
                .get_output(&*network, 0)
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
                .get_output(&*network, 0)
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
        let clamped_indices =
            clamped
                .get_output(&*network, 0)
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
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get gather output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add gatherND operation (N-dimensional gather).
    /// Clamps each index component to `[-dim_size, dim_size - 1]` on the corresponding data axis
    /// (WebNN / ONNX / WPT); TensorRT Gather-ND otherwise returns 0 for out-of-range indices.
    fn add_gather_nd_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let indices = tensor_map
            .get(&operation.input_operands()[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Indices operand {} not found",
                    operation.input_operands()[1]
                ),
            })?;

        let data_operand = graph
            .operand(operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "gatherND: data operand {} not in graph",
                    operation.input_operands()[0]
                ),
            })?;
        let indices_operand = graph
            .operand(operation.input_operands()[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "gatherND: indices operand {} not in graph",
                    operation.input_operands()[1]
                ),
            })?;

        let data_rank = data_operand.descriptor.shape.len();
        let k = indices_operand
            .descriptor
            .shape
            .last()
            .map(get_static_or_max_size)
            .unwrap_or(1) as usize;
        if k == 0 || k > data_rank {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "gatherND: invalid indices last dim {} for data rank {}",
                    k, data_rank
                ),
            });
        }

        let mins_k: Vec<i32> = data_operand.descriptor.shape[..k]
            .iter()
            .map(|d| -(get_static_or_max_size(d) as i32))
            .collect();
        let maxs_k: Vec<i32> = data_operand.descriptor.shape[..k]
            .iter()
            .map(|d| get_static_or_max_size(d) as i32 - 1)
            .collect();

        let idx_dims: Vec<u32> = indices_operand
            .descriptor
            .shape
            .iter()
            .map(get_static_or_max_size)
            .collect();
        let r = idx_dims.len();
        if r < 1 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "gatherND: indices tensor must have rank >= 1".to_string(),
            });
        }
        let prefix_count: usize = if r <= 1 {
            1usize
        } else {
            idx_dims[..r - 1].iter().map(|&x| x as usize).product()
        };
        let mut min_data: Vec<u8> =
            Vec::with_capacity(prefix_count.saturating_mul(k).saturating_mul(4));
        let mut max_data: Vec<u8> =
            Vec::with_capacity(prefix_count.saturating_mul(k).saturating_mul(4));
        for _ in 0..prefix_count {
            for j in 0..k {
                min_data.extend_from_slice(&mins_k[j].to_le_bytes());
                max_data.extend_from_slice(&maxs_k[j].to_le_bytes());
            }
        }

        let indices_shape_i64: Vec<i64> = idx_dims.iter().map(|&d| d as i64).collect();
        let min_const = network
            .add_small_constant_copied(&indices_shape_i64, &min_data, trtx::DataType::kINT32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("gatherND: clamp min constant: {}", e),
            })?;
        let max_const = network
            .add_small_constant_copied(&indices_shape_i64, &max_data, trtx::DataType::kINT32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("gatherND: clamp max constant: {}", e),
            })?;

        let min_const_out =
            min_const
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("gatherND: clamp min output: {}", e),
                })?;
        let max_const_out =
            max_const
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("gatherND: clamp max output: {}", e),
                })?;

        let clamped_upper = network
            .add_elementwise(indices, &max_const_out, ElementWiseOperation::kMIN)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("gatherND: indices clamp upper: {}", e),
            })?;
        let clamped_upper_out =
            clamped_upper
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("gatherND: clamp upper tensor: {}", e),
                })?;

        let clamped = network
            .add_elementwise(
                &min_const_out,
                &clamped_upper_out,
                ElementWiseOperation::kMAX,
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("gatherND: indices clamp lower: {}", e),
            })?;
        let clamped_indices =
            clamped
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("gatherND: clamped indices: {}", e),
                })?;

        let mut layer = network
            .add_gather(input, &clamped_indices, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add gatherND layer: {}", e),
            })?;

        layer.set_gather_mode(network, trtx::GatherMode::kND);

        let output = layer
            .get_output(&*network, 0)
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
    fn add_scatter_elements_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let data = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Data operand {} not found", operation.input_operands()[0]),
            })?;

        let indices = tensor_map
            .get(&operation.input_operands()[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Indices operand {} not found",
                    operation.input_operands()[1]
                ),
            })?;

        let updates = tensor_map
            .get(&operation.input_operands()[2])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Updates operand {} not found",
                    operation.input_operands()[2]
                ),
            })?;

        // Get axis attribute (default to 0)
        let axis = operation
            .attributes()
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
        layer.set_axis(network, axis);

        let output = layer
            .get_output(&*network, 0)
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
    fn add_scatter_nd_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let data = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Data operand {} not found", operation.input_operands()[0]),
            })?;

        let indices = tensor_map
            .get(&operation.input_operands()[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Indices operand {} not found",
                    operation.input_operands()[1]
                ),
            })?;

        let updates = tensor_map
            .get(&operation.input_operands()[2])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Updates operand {} not found",
                    operation.input_operands()[2]
                ),
            })?;

        // Create scatter layer with mode kND for N-dimensional scatter
        let layer = network
            .add_scatter(data, indices, updates, ScatterMode::kND)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add scatterND layer: {}", e),
            })?;

        let output = layer
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get scatterND output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Shared argMin/argMax via TopK: indices tensor reshaped to WebNN output, still INT32 (caller casts).
    ///
    /// TensorRT `ITopKLayer` for rank &gt;= 5 only allows reduction on one of the **last four**
    /// dimensions. When WebNN `axis` lies outside that set, swap that axis with the last axis,
    /// run TopK on the last axis, then apply the same transpose to the index tensor (swap is self-inverse).
    fn add_arg_reduce_common<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        input_id: u32,
        input: &trtx::Tensor<'a>,
        axis: u32,
        keep_dims: bool,
        topk_op: TopKOperation,
        label: &'static str,
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        let input_shape = graph
            .operand(input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{} input operand {} not in graph", label, input_id),
            })?
            .descriptor
            .static_or_max_shape();
        let rank = input_shape.len();
        if rank > 8 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "{}: TensorRT shuffle transpose uses at most 8 axes, got rank {}",
                    label, rank
                ),
            });
        }
        let axis_u = axis as usize;
        if axis_u >= rank {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{}: axis {} out of range for rank {}", label, axis, rank),
            });
        }

        let target_shape_u32 = infer_arg_reduce_shape(&input_shape, axis, keep_dims)?;
        let target_shape_i64: Vec<i64> = target_shape_u32.iter().map(|&d| d as i64).collect();

        let indices_pre_reshape = if rank == 1 {
            let n = input_shape[0] as i64;
            let mut shuffle_in =
                network
                    .add_shuffle(input)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("{}: unsqueeze 1D input for TopK: {}", label, e),
                    })?;
            shuffle_in
                .set_reshape_dimensions(network, &[n, 1])
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{}: set [N,1] reshape: {}", label, e),
                })?;
            let rank2 =
                shuffle_in
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("{}: rank-2 TopK input: {}", label, e),
                    })?;
            let layer = network
                .add_topk(&rank2, topk_op, 1, Axes::from_bits(1u32))
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{}: topK layer: {}", label, e),
                })?;
            layer
                .get_output(&*network, 1)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{}: topK indices output: {}", label, e),
                })?
        } else if rank >= 5 && axis_u < rank - 4 {
            let last = rank - 1;
            let mut perm: Vec<i32> = (0..rank as i32).collect();
            perm[axis_u] = last as i32;
            perm[last] = axis as i32;

            let mut shuffle_pre =
                network
                    .add_shuffle(input)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("{}: pre-transpose shuffle for TopK: {}", label, e),
                    })?;
            shuffle_pre
                .set_first_transpose(network, &perm)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{}: set pre-transpose: {}", label, e),
                })?;
            let shuffled =
                shuffle_pre
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("{}: pre-shuffle output: {}", label, e),
                    })?;

            let layer = network
                .add_topk(
                    &shuffled,
                    topk_op,
                    1,
                    Axes::from_bits(1u32 << (last as u32)),
                )
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{}: topK layer: {}", label, e),
                })?;
            let idx = layer
                .get_output(&*network, 1)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{}: topK indices output: {}", label, e),
                })?;

            let mut shuffle_post =
                network
                    .add_shuffle(&idx)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("{}: post-transpose shuffle: {}", label, e),
                    })?;
            shuffle_post
                .set_first_transpose(network, &perm)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{}: set post-transpose: {}", label, e),
                })?;
            shuffle_post
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{}: post-shuffle output: {}", label, e),
                })?
        } else {
            let layer = network
                .add_topk(input, topk_op, 1, Axes::from_bits(1u32 << axis))
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{}: topK layer: {}", label, e),
                })?;
            layer
                .get_output(&*network, 1)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("{}: topK indices output: {}", label, e),
                })?
        };

        let mut shuffle_layer = network.add_shuffle(&indices_pre_reshape).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{}: shuffle for output shape: {}", label, e),
            }
        })?;
        shuffle_layer
            .set_reshape_dimensions(network, &target_shape_i64)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{}: set reshape dimensions: {}", label, e),
            })?;
        shuffle_layer
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("{}: shuffle output: {}", label, e),
            })
    }

    /// Add argMax operation (find indices of maximum values)
    fn add_arg_max_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_id = operation.input_operands()[0];
        let input = tensor_map
            .get(&input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", input_id),
            })?;

        let (axis, keep_dims) = match operation {
            Operation::ArgMax { axis, options, .. } => (
                *axis,
                options.as_ref().map(|o| o.keep_dimensions).unwrap_or(false),
            ),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "expected ArgMax operation".to_string(),
                });
            }
        };

        let shaped_output = Self::add_arg_reduce_common(
            graph,
            network,
            input_id,
            input,
            axis,
            keep_dims,
            TopKOperation::kMAX,
            "ArgMax",
        )?;

        let final_output = Self::cast_int32_to_float32(network, &shaped_output)?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, final_output);
        Ok(())
    }

    /// Add argMin operation (find indices of minimum values)
    fn add_arg_min_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_id = operation.input_operands()[0];
        let input = tensor_map
            .get(&input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", input_id),
            })?;

        let (axis, keep_dims) = match operation {
            Operation::ArgMin { axis, options, .. } => (
                *axis,
                options.as_ref().map(|o| o.keep_dimensions).unwrap_or(false),
            ),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "expected ArgMin operation".to_string(),
                });
            }
        };

        let shaped_output = Self::add_arg_reduce_common(
            graph,
            network,
            input_id,
            input,
            axis,
            keep_dims,
            TopKOperation::kMIN,
            "ArgMin",
        )?;

        let final_output = Self::cast_int32_to_float32(network, &shaped_output)?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, final_output);
        Ok(())
    }

    // ============================================================================
    // Other Operations (2026-01-29)
    // ============================================================================

    /// Add clamp operation (clip values to range [min, max])
    fn add_clamp_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        // Get input operand descriptor to determine shape dimensions
        let input_operand = graph
            .operand(operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Input operand {} not found in graph",
                    operation.input_operands()[0]
                ),
            })?;
        let input_dtype = input_operand.descriptor.data_type;
        let broadcast_shape = Self::trtx_broadcast_ones_for_elementwise_scalar(
            input,
            &*network,
            input_operand.descriptor.shape.len(),
            "clamp",
        )?;

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

        let attrs = operation.attributes();
        let clamp_opts = attrs.as_clamp();
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
        let max_const = network
            .add_small_constant_copied(&broadcast_shape, &max_bytes, trt_dtype.clone())
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add max constant: {}", e),
            })?;

        let max_const_output =
            max_const
                .get_output(&*network, 0)
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
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get upper clamp output: {}", e),
                })?;

        // Second: max(min_value, clamped_upper)
        let min_const = network
            .add_small_constant_copied(&broadcast_shape, &min_bytes, trt_dtype)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add min constant: {}", e),
            })?;

        let min_const_output =
            min_const
                .get_output(&*network, 0)
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
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get clamp output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add where operation (select elements based on condition).
    /// WebNN broadcasts **condition**, **true**, and **false** to one common shape ([`infer_where_shape`]);
    /// pairwise [`ensure_broadcast_compatible`] is not enough. `ISelectLayer` needs BOOL condition:
    /// broadcast condition as Float32 (promote UInt8/Int8), then `cast_to_bool` after resize.
    fn add_where_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let cond_id = operation.input_operands()[0];
        let true_id = operation.input_operands()[1];
        let false_id = operation.input_operands()[2];

        let condition = tensor_map
            .get(&cond_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Condition operand {} not found", cond_id),
            })?;

        let true_value = tensor_map
            .get(&true_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("True value operand {} not found", true_id),
            })?;

        let false_value =
            tensor_map
                .get(&false_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("False value operand {} not found", false_id),
                })?;

        let shape_c = Self::operand_shape_u32_static(graph, cond_id)?;
        let shape_t = Self::operand_shape_u32_static(graph, true_id)?;
        let shape_f = Self::operand_shape_u32_static(graph, false_id)?;
        let out_shape_u32 = infer_where_shape(&shape_c, &shape_t, &shape_f)?;
        let target_i64: Vec<i64> = out_shape_u32.iter().map(|&x| x as i64).collect();

        let cond_in: Vec<i64> = shape_c.iter().map(|&x| x as i64).collect();
        let true_in: Vec<i64> = shape_t.iter().map(|&x| x as i64).collect();
        let false_in: Vec<i64> = shape_f.iter().map(|&x| x as i64).collect();

        let cond_float_promoted = match graph.operand(cond_id).map(|o| o.descriptor.data_type) {
            Some(DataType::Uint8) | Some(DataType::Int8) => {
                Some(Self::cast_to_float32(network, condition)?)
            }
            _ => None,
        };
        let cond_f: &trtx::Tensor<'a> = cond_float_promoted.as_ref().unwrap_or(condition);

        let cond_bc = Self::broadcast_trtx_tensor_to_dims(
            network,
            cond_f,
            &cond_in,
            &target_i64,
            "where_cond",
        )?;
        let true_bc = Self::broadcast_trtx_tensor_to_dims(
            network,
            true_value,
            &true_in,
            &target_i64,
            "where_true",
        )?;
        let false_bc = Self::broadcast_trtx_tensor_to_dims(
            network,
            false_value,
            &false_in,
            &target_i64,
            "where_false",
        )?;

        let condition_bool = Self::cast_to_bool(network, &cond_bc)?;

        let layer = network
            .add_select(&condition_bool, &true_bc, &false_bc)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add select layer: {}", e),
            })?;

        let output = layer
            .get_output(&*network, 0)
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
    fn add_linear_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        // Get input operand descriptor to determine shape dimensions
        let input_operand = graph
            .operand(operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Input operand {} not found in graph",
                    operation.input_operands()[0]
                ),
            })?;
        let broadcast_shape = Self::trtx_broadcast_ones_for_elementwise_scalar(
            input,
            &*network,
            input_operand.descriptor.shape.len(),
            "linear",
        )?;

        let attrs = operation.attributes();
        let linear_opts = attrs.as_linear();
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
            let alpha_constant = network
                .add_small_constant_copied(&broadcast_shape, &alpha_bytes, alpha_dtype)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to create alpha constant: {}", e),
                })?;

            let alpha_tensor = alpha_constant.get_output(&*network, 0).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get alpha constant output: {}", e),
                }
            })?;

            // Multiply: alpha * x
            let mul_layer = network
                .add_elementwise(input, &alpha_tensor, ElementWiseOperation::kPROD)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to multiply by alpha: {}", e),
                })?;

            mul_layer
                .get_output(&*network, 0)
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
                .get_output(&*network, 0)
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
            let beta_constant = network
                .add_small_constant_copied(&broadcast_shape, &beta_bytes, beta_dtype)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to create beta constant: {}", e),
                })?;

            let beta_tensor = beta_constant.get_output(&*network, 0).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get beta constant output: {}", e),
                }
            })?;

            // Add: (alpha * x) + beta
            let add_layer = network
                .add_elementwise(&after_multiply, &beta_tensor, ElementWiseOperation::kSUM)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add beta: {}", e),
                })?;

            add_layer
                .get_output(&*network, 0)
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

    /// Parse WebNN `MLPadOptions.value` (`MLNumber`: number or special float strings like `"NaN"`).
    fn pad_mlnumber_to_f32(v: &serde_json::Value) -> Option<f32> {
        if let Some(f) = v.as_f64() {
            return Some(f as f32);
        }
        if let Some(i) = v.as_i64() {
            return Some(i as f32);
        }
        if let Some(u) = v.as_u64() {
            return Some(u as f32);
        }
        let s = v.as_str()?.trim();
        let lower = s.to_lowercase();
        Some(match lower.as_str() {
            "nan" => f32::NAN,
            "infinity" | "+infinity" | "inf" | "+inf" => f32::INFINITY,
            "-infinity" | "-inf" => f32::NEG_INFINITY,
            _ => s.parse::<f32>().ok()?,
        })
    }

    /// Pad one axis by concatenating constant-filled tensors (constant-pad only).
    fn trtx_pad_axis_constant_concat<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        input: &trtx::Tensor<'a>,
        axis: usize,
        pre: i32,
        post: i32,
        fill_f32: f32,
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        if pre <= 0 && post <= 0 {
            let id = network
                .add_identity(input)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Pad concat: identity: {}", e),
                })?;
            return id
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Pad concat: identity output: {}", e),
                });
        }

        let dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Pad concat: dimensions: {}", e),
            })?;
        let rank = dims.len();
        if axis >= rank {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Pad concat: axis {axis} >= rank {rank}"),
            });
        }

        let trt_dtype = input.get_type(&*network);
        let elem_size = match trt_dtype {
            TrtDataType::kFLOAT => 4usize,
            TrtDataType::kHALF => 2,
            TrtDataType::kINT32 => 4,
            TrtDataType::kINT64 => 8,
            TrtDataType::kINT8 | TrtDataType::kUINT8 | TrtDataType::kBOOL => 1,
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!(
                        "Pad concat: unsupported TensorRT dtype for constant fill: {:?}",
                        trt_dtype
                    ),
                });
            }
        };

        let one_elem_bytes: Vec<u8> = match trt_dtype {
            TrtDataType::kFLOAT => fill_f32.to_le_bytes().to_vec(),
            TrtDataType::kHALF => f16::from_f32(fill_f32).to_bits().to_le_bytes().to_vec(),
            TrtDataType::kINT32 => (fill_f32 as i32).to_le_bytes().to_vec(),
            TrtDataType::kINT64 => (fill_f32 as i64).to_le_bytes().to_vec(),
            TrtDataType::kINT8 => [(fill_f32 as i8) as u8].to_vec(),
            TrtDataType::kUINT8 | TrtDataType::kBOOL => {
                vec![if fill_f32 != 0.0 { 1u8 } else { 0u8 }]
            }
            _ => unreachable!(),
        };

        let mut make_const = |axis_len: i64| -> Result<trtx::Tensor<'a>, GraphError> {
            let mut shape = dims.clone();
            shape[axis] = axis_len;
            let n: usize = shape.iter().map(|&d| d.max(0) as usize).product();
            let mut data = Vec::with_capacity(n * elem_size);
            for _ in 0..n {
                data.extend_from_slice(&one_elem_bytes);
            }
            let layer = network
                .add_small_constant_copied(&shape, &data, trt_dtype)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Pad concat: constant: {}", e),
                })?;
            layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Pad concat: constant output: {}", e),
                })
        };

        let mut pieces: Vec<&trtx::Tensor<'a>> = Vec::new();
        let pre_t = if pre > 0 {
            Some(make_const(pre as i64)?)
        } else {
            None
        };
        if let Some(ref t) = pre_t {
            pieces.push(t);
        }
        pieces.push(input);
        let post_t = if post > 0 {
            Some(make_const(post as i64)?)
        } else {
            None
        };
        if let Some(ref t) = post_t {
            pieces.push(t);
        }

        let mut cat =
            network
                .add_concatenation(&pieces)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Pad concat: concat: {}", e),
                })?;
        cat.set_axis(network, axis as i32);
        cat.get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Pad concat: concat out: {}", e),
            })
    }

    /// Edge pad on one axis: replicate first / last slice along `axis` (WebNN `mode == "edge"`).
    fn trtx_pad_axis_edge_concat<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        input: &trtx::Tensor<'a>,
        axis: usize,
        pre: i32,
        post: i32,
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        if pre <= 0 && post <= 0 {
            let id = network
                .add_identity(input)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Pad edge: identity: {}", e),
                })?;
            return id
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Pad edge: identity output: {}", e),
                });
        }

        let dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Pad edge: dimensions: {}", e),
            })?;
        let rank = dims.len();
        if axis >= rank {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Pad edge: axis {axis} >= rank {rank}"),
            });
        }
        let d_ax = dims[axis];
        if d_ax <= 0 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Pad edge: non-positive dim {d_ax} on axis {axis}"),
            });
        }

        let stride: Vec<i64> = vec![1i64; rank];

        let start_left = vec![0i64; rank];
        let mut size_left = dims.clone();
        size_left[axis] = 1;
        let left_layer = network
            .add_slice(input, &start_left, &size_left, &stride)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Pad edge: left slice: {}", e),
            })?;
        let left_t =
            left_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Pad edge: left slice output: {}", e),
                })?;

        let mut start_right = vec![0i64; rank];
        start_right[axis] = d_ax - 1;
        let mut size_right = dims.clone();
        size_right[axis] = 1;
        let right_layer = network
            .add_slice(input, &start_right, &size_right, &stride)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Pad edge: right slice: {}", e),
            })?;
        let right_t =
            right_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Pad edge: right slice output: {}", e),
                })?;

        let mut cat_inputs: Vec<&trtx::Tensor<'a>> = Vec::new();
        for _ in 0..pre {
            cat_inputs.push(&left_t);
        }
        cat_inputs.push(input);
        for _ in 0..post {
            cat_inputs.push(&right_t);
        }

        let mut cat =
            network
                .add_concatenation(&cat_inputs)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Pad edge: concat: {}", e),
                })?;
        cat.set_axis(network, axis as i32);
        cat.get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Pad edge: concat out: {}", e),
            })
    }

    /// Reflection pad on one axis (NumPy / ONNX `reflect`: mirror interior, excluding the edge).
    /// Requires `pre < dim` and `post < dim` on this axis.
    fn trtx_pad_axis_reflect_concat<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        input: &trtx::Tensor<'a>,
        axis: usize,
        pre: i32,
        post: i32,
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        if pre <= 0 && post <= 0 {
            let id = network
                .add_identity(input)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Pad reflect: identity: {}", e),
                })?;
            return id
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Pad reflect: identity output: {}", e),
                });
        }

        let dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Pad reflect: dimensions: {}", e),
            })?;
        let rank = dims.len();
        if axis >= rank {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Pad reflect: axis {axis} >= rank {rank}"),
            });
        }
        let d_ax = dims[axis];
        if d_ax <= 0 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Pad reflect: non-positive dim {d_ax} on axis {axis}"),
            });
        }
        if pre as i64 >= d_ax || post as i64 >= d_ax {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Pad reflect: axis {axis} dim {d_ax} requires pre/post padding each < dim (got pre={pre} post={post})"
                ),
            });
        }

        let stride: Vec<i64> = vec![1i64; rank];
        let mut pre_slabs: Vec<trtx::Tensor<'a>> = Vec::new();
        for off in (1_i32..=pre).rev() {
            let idx = i64::from(off);
            let mut start = vec![0i64; rank];
            start[axis] = idx;
            let mut size = dims.clone();
            size[axis] = 1;
            let layer = network
                .add_slice(input, &start, &size, &stride)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Pad reflect: pre slice: {}", e),
                })?;
            let slab =
                layer
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Pad reflect: pre slice out: {}", e),
                    })?;
            pre_slabs.push(slab);
        }

        let mut post_slabs: Vec<trtx::Tensor<'a>> = Vec::new();
        for t in 0..post {
            let idx = d_ax - 2 - i64::from(t);
            let mut start = vec![0i64; rank];
            start[axis] = idx;
            let mut size = dims.clone();
            size[axis] = 1;
            let layer = network
                .add_slice(input, &start, &size, &stride)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Pad reflect: post slice: {}", e),
                })?;
            let slab =
                layer
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Pad reflect: post slice out: {}", e),
                    })?;
            post_slabs.push(slab);
        }

        let mut cat_inputs: Vec<&trtx::Tensor<'a>> = pre_slabs.iter().collect::<Vec<_>>();
        cat_inputs.push(input);
        cat_inputs.extend(post_slabs.iter());

        let mut cat =
            network
                .add_concatenation(&cat_inputs)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Pad reflect: concat: {}", e),
                })?;
        cat.set_axis(network, axis as i32);
        cat.get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Pad reflect: concat out: {}", e),
            })
    }

    /// Add pad operation (pad tensor with constant/edge/reflection values)
    fn add_pad_op<'a>(
        _graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let (beginning_padding, ending_padding, pad_opts) = match operation {
            Operation::Pad {
                beginning_padding,
                ending_padding,
                options,
                ..
            } => (
                beginning_padding,
                ending_padding,
                options
                    .as_ref()
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: "Pad operation missing options".to_string(),
                    })?,
            ),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "expected Pad operation".to_string(),
                });
            }
        };
        let pre_padding: Vec<i32> = beginning_padding.iter().map(|&u| u as i32).collect();
        let post_padding: Vec<i32> = ending_padding.iter().map(|&u| u as i32).collect();

        // Get input dimensions
        let input_dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get input dimensions: {}", e),
            })?;

        if pre_padding.len() != post_padding.len() {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Pad: beginningPadding len {} != endingPadding len {}",
                    pre_padding.len(),
                    post_padding.len()
                ),
            });
        }
        if pre_padding.len() != input_dims.len() {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Pad: padding length {} does not match input rank {}",
                    pre_padding.len(),
                    input_dims.len()
                ),
            });
        }

        // WebNN: empty paddings on rank-0 input (or all zeros) is a no-op; TRT IPaddingLayer still needs work.
        let no_pad_amounts = pre_padding
            .iter()
            .chain(post_padding.iter())
            .all(|&p| p == 0);
        if no_pad_amounts {
            let identity =
                network
                    .add_identity(input)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Pad no-op: failed to add identity: {}", e),
                    })?;
            let out =
                identity
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Pad no-op: identity output: {}", e),
                    })?;
            tensor_map.insert(operation.output_operands_slice()[0], out);
            return Ok(());
        }

        let original_ndims = input_dims.len();

        let pad_mode_norm = pad_opts.mode.to_lowercase();
        let is_constant_pad_mode = pad_mode_norm.is_empty() || pad_mode_norm == "constant";
        let is_edge_pad_mode = pad_mode_norm == "edge";
        let is_reflect_pad_mode = pad_mode_norm == "reflection" || pad_mode_norm == "reflect";
        if !pad_mode_norm.is_empty()
            && !is_constant_pad_mode
            && !is_edge_pad_mode
            && !is_reflect_pad_mode
        {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Pad: TensorRT pad supports constant, edge, and reflection mode (got {:?})",
                    pad_opts.mode
                ),
            });
        }
        let fill_f32 = pad_opts
            .value
            .as_ref()
            .and_then(Self::pad_mlnumber_to_f32)
            .unwrap_or(0.0);

        // TensorRT `addPaddingNd` only accepts **two** padding values (last two tensor dims). When
        // WebNN padding affects other axes (e.g. 3D with pad on axis 0), use concat + constants.
        let lead_if_reshaped = if original_ndims < 4 {
            4 - original_ndims
        } else {
            0
        };
        let trt_ipadding_covers = if original_ndims < 4 {
            (0..original_ndims).all(|wi| {
                let trt_d = lead_if_reshaped + wi;
                trt_d >= 2 || (pre_padding[wi] == 0 && post_padding[wi] == 0)
            })
        } else {
            (0..original_ndims.saturating_sub(2))
                .all(|wi| pre_padding[wi] == 0 && post_padding[wi] == 0)
        };

        // `IPaddingLayer` is zero-only and 2D; edge / non-zero constant use slice+concat or constant+concat.
        let use_concat_pad = !trt_ipadding_covers
            || !is_constant_pad_mode
            || fill_f32 != 0.0
            || fill_f32.is_nan()
            || fill_f32.is_infinite();
        if use_concat_pad {
            let in_id = operation.input_operands()[0];
            let out_id = operation.output_operands_slice()[0];
            let mut cur_key = in_id;
            let mut step: u32 = 0;
            let mut any = false;
            for ax in 0..original_ndims {
                let pr = pre_padding[ax];
                let po = post_padding[ax];
                if pr == 0 && po == 0 {
                    continue;
                }
                any = true;
                let t = tensor_map
                    .get(&cur_key)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Pad concat: tensor {cur_key} not found"),
                    })?;
                let new_t = if is_edge_pad_mode {
                    Self::trtx_pad_axis_edge_concat(network, t, ax, pr, po)?
                } else if is_reflect_pad_mode {
                    Self::trtx_pad_axis_reflect_concat(network, t, ax, pr, po)?
                } else {
                    Self::trtx_pad_axis_constant_concat(network, t, ax, pr, po, fill_f32)?
                };
                let next_key = 0xFA_B0_0000u32.wrapping_add(step);
                step = step.saturating_add(1);
                if cur_key != in_id {
                    let _ = tensor_map.remove(&cur_key);
                }
                tensor_map.insert(next_key, new_t);
                cur_key = next_key;
            }
            if !any {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "Pad: internal error (concat path but no non-zero axis pad)"
                        .to_string(),
                });
            }
            let final_t =
                tensor_map
                    .remove(&cur_key)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: "Pad concat: missing final tensor".to_string(),
                    })?;
            tensor_map.insert(out_id, final_t);
            return Ok(());
        }

        // Rank < 4: left-pad shape with 1s to 4D; WebNN axis i maps to TRT axis (4 - original_ndims + i).
        let input_to_pad = if original_ndims < 4 {
            let mut shape_4d: Vec<i64> = vec![1i64; 4 - original_ndims];
            shape_4d.extend_from_slice(&input_dims);

            // Reshape to 4D
            let mut reshape_layer =
                network
                    .add_shuffle(input)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to create reshape layer for padding: {}", e),
                    })?;

            reshape_layer
                .set_reshape_dimensions(network, &shape_4d)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to set reshape dimensions: {}", e),
                })?;

            reshape_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get reshape output: {}", e),
                })?
        } else {
            let identity =
                network
                    .add_identity(input)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to create identity layer: {}", e),
                    })?;
            identity
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get identity output: {}", e),
                })?
        };

        let (pre_spatial, post_spatial): (Vec<i64>, Vec<i64>) = if original_ndims < 4 {
            let lead = 4 - original_ndims;
            let wn_pre = |trt_ax: usize| -> i64 {
                if trt_ax < lead {
                    return 0;
                }
                let wi = trt_ax - lead;
                if wi < original_ndims {
                    pre_padding[wi] as i64
                } else {
                    0
                }
            };
            let wn_post = |trt_ax: usize| -> i64 {
                if trt_ax < lead {
                    return 0;
                }
                let wi = trt_ax - lead;
                if wi < original_ndims {
                    post_padding[wi] as i64
                } else {
                    0
                }
            };
            (vec![wn_pre(2), wn_pre(3)], vec![wn_post(2), wn_post(3)])
        } else {
            if original_ndims < 2 {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Pad: rank {original_ndims} < 2 for IPaddingLayer"),
                });
            }
            (
                vec![
                    pre_padding[original_ndims - 2] as i64,
                    pre_padding[original_ndims - 1] as i64,
                ],
                vec![
                    post_padding[original_ndims - 2] as i64,
                    post_padding[original_ndims - 1] as i64,
                ],
            )
        };

        let padding_layer = network
            .add_padding(&input_to_pad, &pre_spatial, &post_spatial)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add padding layer: {}", e),
            })?;

        let padded_output =
            padding_layer
                .get_output(&*network, 0)
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
                    output_shape[i] += (pre + post) as i64;
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
                .set_reshape_dimensions(network, &output_shape)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to set reshape-back dimensions: {}", e),
                })?;

            reshape_back
                .get_output(&*network, 0)
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
    fn add_gemm_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let a_id = operation.input_operands()[0];
        let input_a = tensor_map
            .get(&a_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", a_id),
            })?;

        let input_b = tensor_map
            .get(&operation.input_operands()[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[1]),
            })?;

        let gemm_dtype = graph
            .operand(a_id)
            .map(|o| o.descriptor.data_type)
            .unwrap_or(DataType::Float32);

        let attrs = operation.attributes();
        let opts = attrs.as_gemm();
        let alpha = opts.map(|o| o.alpha as f32).unwrap_or(1.0);
        let beta = opts.map(|o| o.beta as f32).unwrap_or(1.0);
        let a_transpose = opts.map(|o| o.a_transpose).unwrap_or(false);
        let b_transpose = opts.map(|o| o.b_transpose).unwrap_or(false);

        // Get actual dimensions for validation
        let _dims_a = input_a
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get input A dimensions: {}", e),
            })?;
        let _dims_b = input_b
            .dimensions(&*network)
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
        // Swap: instead of A @ B^T, try B^T @ A^T (which gives same result transposed)
        // NO wait - let's try: B @ A instead since B^T @ A^T = (A @ B)^T
        //
        // Actually, for WebNN: output = A @ B^T
        // Try: output = (B @ A^T)^T = A @ B^T (mathematically equivalent)
        let (mat_a, mat_b, op_a, op_b) = if b_transpose && !a_transpose {
            (
                input_a,
                input_b,
                MatrixOperation::kNONE,
                MatrixOperation::kTRANSPOSE,
            )
        } else {
            let a_op = if a_transpose {
                MatrixOperation::kTRANSPOSE
            } else {
                MatrixOperation::kNONE
            };
            let b_op = if b_transpose {
                MatrixOperation::kTRANSPOSE
            } else {
                MatrixOperation::kNONE
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

        let mut result =
            layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get GEMM layer output: {}", e),
                })?;

        // If alpha != 1.0, scale the result
        if (alpha - 1.0).abs() > 1e-6 {
            // Get result dimensions to create a constant with matching shape
            let result_dims =
                result
                    .dimensions(&*network)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get result dimensions: {}", e),
                    })?;

            // Create constant filled with alpha value matching result shape and element type
            // (TensorRT elementwise PROD requires matching types; matrix multiply output follows A/B).
            let num_elements: usize = result_dims.iter().map(|&d| d as usize).product();
            let (alpha_bytes, alpha_trt_ty) = match gemm_dtype {
                DataType::Float16 => {
                    let bits = f16::from_f32(alpha).to_bits().to_le_bytes();
                    let mut v = Vec::with_capacity(num_elements * 2);
                    for _ in 0..num_elements {
                        v.extend_from_slice(&bits);
                    }
                    (v, TrtDataType::kHALF)
                }
                _ => {
                    let alpha_data: Vec<f32> = vec![alpha; num_elements];
                    let alpha_bytes: Vec<u8> =
                        alpha_data.iter().flat_map(|&f| f.to_le_bytes()).collect();
                    (alpha_bytes, TrtDataType::kFLOAT)
                }
            };

            let alpha_layer = network
                .add_small_constant_copied(&result_dims, &alpha_bytes, alpha_trt_ty)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to create alpha constant: {}", e),
                })?;

            let alpha_tensor =
                alpha_layer
                    .get_output(&*network, 0)
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

            result =
                scale_layer
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get scaled output: {}", e),
                    })?;
        }

        // Optional bias matrix C is WebNN `MLGemmOptions.c`; Gemm only lists [A, B] in
        // `input_operands` (same as ONNX node inputs in converters/onnx.rs).
        if let Some(c_id) = opts.and_then(|o| o.c).filter(|_| beta.abs() > 1e-6) {
            let input_c = tensor_map
                .get(&c_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("GEMM options.c operand {} not found", c_id),
                })?;

            // Scale C by beta if needed, then add to result
            if (beta - 1.0).abs() > 1e-6 {
                // Get C dimensions to create a constant with matching shape
                let c_dims =
                    input_c
                        .dimensions(&*network)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Failed to get C dimensions: {}", e),
                        })?;

                // Create constant filled with beta value matching C shape and GEMM element type
                let num_elements: usize = c_dims.iter().map(|&d| d as usize).product();
                let (beta_bytes, beta_trt_ty) = match gemm_dtype {
                    DataType::Float16 => {
                        let bits = f16::from_f32(beta).to_bits().to_le_bytes();
                        let mut v = Vec::with_capacity(num_elements * 2);
                        for _ in 0..num_elements {
                            v.extend_from_slice(&bits);
                        }
                        (v, TrtDataType::kHALF)
                    }
                    _ => {
                        let beta_data: Vec<f32> = vec![beta; num_elements];
                        let beta_bytes: Vec<u8> =
                            beta_data.iter().flat_map(|&f| f.to_le_bytes()).collect();
                        (beta_bytes, TrtDataType::kFLOAT)
                    }
                };

                let beta_layer = network
                    .add_small_constant_copied(&c_dims, &beta_bytes, beta_trt_ty)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to create beta constant: {}", e),
                    })?;

                let beta_tensor = beta_layer.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get beta tensor: {}", e),
                    }
                })?;

                // Multiply C by beta
                let scale_c_layer = network
                    .add_elementwise(input_c, &beta_tensor, ElementWiseOperation::kPROD)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to scale C by beta: {}", e),
                    })?;

                let scaled_c = scale_c_layer.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get scaled C: {}", e),
                    }
                })?;

                // WebNN/ONNX broadcast C to alpha*A*B (e.g. C [5] with output [3,5]). One call
                // pads rank ([5]->[1,5]); a second call expands 1s to match rows (TRT elementwise).
                let (r_bc, c_bc) = Self::ensure_broadcast_compatible(
                    network,
                    &result,
                    &scaled_c,
                    "gemm_options_c",
                )?;
                let (r_bc2, c_bc2) =
                    Self::ensure_broadcast_compatible(network, &r_bc, &c_bc, "gemm_options_c")?;

                let add_layer = network
                    .add_elementwise(&r_bc2, &c_bc2, ElementWiseOperation::kSUM)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add scaled C to result: {}", e),
                    })?;

                result = add_layer.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get final GEMM output: {}", e),
                    }
                })?;
            } else {
                let (r_bc, c_bc) =
                    Self::ensure_broadcast_compatible(network, &result, input_c, "gemm_options_c")?;
                let (r_bc2, c_bc2) =
                    Self::ensure_broadcast_compatible(network, &r_bc, &c_bc, "gemm_options_c")?;

                let add_layer = network
                    .add_elementwise(&r_bc2, &c_bc2, ElementWiseOperation::kSUM)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add C to result: {}", e),
                    })?;

                result = add_layer.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get final GEMM output: {}", e),
                    }
                })?;
            }
        }

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, result);
        Ok(())
    }

    /// Add 2D convolution operation
    fn add_conv2d_op<'a>(
        graph: &'a GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let (input_id, filter_id, conv_opts) = match operation {
            Operation::Conv2d {
                input,
                filter,
                options,
                ..
            } => (*input, *filter, options.as_ref()),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "add_conv2d_op: expected Conv2d operation".to_string(),
                });
            }
        };
        // WebNN: bias is MLConv2dOptions.bias. [`Operation::input_operands`] is only [input, filter];
        // a legacy third JSON input is merged into options.bias when the graph is parsed.
        let bias_id = conv_opts.and_then(|o| o.bias);

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
        let filter_layout = conv_opts
            .map(|o| o.filter_layout.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("oihw");
        let (o, in_ch, h, w): (u32, u32, u32, u32) = match filter_layout {
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
        // Promoted layouts / f16 use `add_convolution_owned_weights` so slices are not tied to `&'a GraphInfo` for the whole op loop.
        let (filter_data_to_use, bias_data, conv_weights_owned) = if filter_constant {
            let filter_shape_u32 = filter_operand.descriptor.static_or_max_shape();
            let filter_data = Self::get_constant_data(graph, filter_id)?;
            let mut bias_promoted: Option<Vec<u8>> = None;
            let bias_raw: Option<&[u8]> = match bias_id {
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
                        bias_promoted = Some(Self::f16_bytes_to_f32_bytes(raw)?);
                        None
                    } else {
                        Some(raw)
                    }
                }
                None => None,
            };
            let filter_dtype = filter_operand.descriptor.data_type;
            let mut kernel_promoted: Option<Vec<u8>> = None;
            match (filter_dtype, filter_layout) {
                (DataType::Float16, _) => {
                    let f32_bytes = Self::f16_bytes_to_f32_bytes(filter_data)?;
                    let oihw = if filter_layout == "oihw" {
                        f32_bytes
                    } else {
                        Self::conv_filter_to_oihw(&f32_bytes, filter_layout, &filter_shape_u32)?
                    };
                    kernel_promoted = Some(oihw);
                }
                (DataType::Float32, "oihw") => {}
                (DataType::Float32, _) => {
                    kernel_promoted = Some(Self::conv_filter_to_oihw(
                        filter_data,
                        filter_layout,
                        &filter_shape_u32,
                    )?);
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
            let conv_weights_owned = if kernel_promoted.is_some() || bias_promoted.is_some() {
                let conv_kernel_const_shape: Vec<i64> =
                    vec![o as i64, in_ch as i64, h as i64, w as i64];
                let kernel_values = kernel_promoted.unwrap_or_else(|| filter_data.to_vec());
                let bias_values = match (bias_promoted, bias_raw) {
                    (Some(b), _) => Some(b),
                    (None, Some(br)) => Some(br.to_vec()),
                    (None, None) => None,
                };
                Some(OwnedConvWeights {
                    kernel: OwnedWeights {
                        shape: conv_kernel_const_shape,
                        data_type: TrtDataType::kFLOAT,
                        values: kernel_values,
                    },
                    bias: bias_values.map(|values| OwnedWeights {
                        shape: vec![num_output_maps as i64],
                        data_type: TrtDataType::kFLOAT,
                        values,
                    }),
                })
            } else {
                None
            };
            if conv_weights_owned.is_some() {
                (None, None, conv_weights_owned)
            } else {
                (Some(filter_data), bias_raw, None)
            }
        } else {
            // Non-constant filter: TensorRT kernel tensor is OIHW; shuffle from WebNN layout when needed.
            if let Some(id) = bias_id {
                if graph.constant_operand_ids_to_handles.contains_key(&id) {
                    return Err(GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: "conv2d with non-constant filter requires bias to be a tensor input (constant bias not supported)".to_string(),
                    });
                }
            }
            (None, None, None)
        };

        // Input layout: nchw (default) or nhwc. TensorRT conv is NCHW; we use IShuffleLayer::setFirstTranspose for NHWC.
        let input_layout = conv_opts
            .map(|o| o.input_layout.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("nchw");
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
        let nhwc_shuffle_output =
            if input_layout == "nhwc" {
                let mut shuffle =
                    network
                        .add_shuffle(input)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Conv2d NHWC->NCHW shuffle: {}", e),
                        })?;
                shuffle
                    .set_first_transpose(network, &[0, 3, 1, 2])
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Conv2d set_first_transpose NHWC->NCHW: {}", e),
                    })?;
                Some(shuffle.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Conv2d NHWC shuffle output: {}", e),
                    }
                })?)
            } else {
                None
            };
        let pre_conv_input = nhwc_shuffle_output.as_ref().unwrap_or(input);

        // TensorRT conv kernel is always Float; cast Half input to Float so types match.
        let half_cast_output: Option<trtx::Tensor<'a>> =
            if input_dtype == DataType::Float16 {
                let cast_layer = network
                    .add_cast(pre_conv_input, TrtDataType::kFLOAT)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Conv2d Half->Float cast: {}", e),
                    })?;
                Some(cast_layer.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Conv2d cast output: {}", e),
                    }
                })?)
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
                let pre_pad_i64: Vec<i64> = pre_padding.iter().map(|&p| p as i64).collect();
                let post_pad_i64: Vec<i64> = post_padding.iter().map(|&p| p as i64).collect();
                let padding_layer = network
                    .add_padding(&conv_input_source, &pre_pad_i64, &post_pad_i64)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add padding layer: {}", e),
                    })?;

                padding_layer.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get padding layer output: {}", e),
                    }
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
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get identity output: {}", e),
                    })?
            };

        // Add convolution layer with zero padding (padding already applied via padding layer)
        // Constant path always passes f32 data (f16 filter/bias are converted above); dtype must match.
        let mut layer = match (conv_weights_owned, filter_data_to_use) {
            (Some(owned), _) => network
                .add_convolution_owned_weights(&conv_input, num_output_maps, &kernel_size, owned)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add convolution (owned weights): {}", e),
                })?,
            (None, Some(fd)) => {
                let conv_weights = trtx::ConvWeights {
                    kernel_weights: fd,
                    kernel_dtype: TrtDataType::kFLOAT,
                    bias_weights: bias_data,
                    bias_dtype: bias_data.map(|_| TrtDataType::kFLOAT),
                };
                network
                    .add_convolution(&conv_input, num_output_maps, &kernel_size, &conv_weights)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add convolution: {}", e),
                    })?
            }
            (None, None) => {
                let filter_tensor =
                    tensor_map
                        .get(&filter_id)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Filter operand {} tensor not found", filter_id),
                        })?;
                // TensorRT conv requires input and kernel same type. We cast activation to Float when input_dtype is Float16; cast filter (and bias) to Float too.
                let filter_tensor_for_conv: Option<trtx::Tensor<'a>> =
                    if filter_operand.descriptor.data_type == DataType::Float16
                        && input_dtype == DataType::Float16
                    {
                        let cast_layer = network
                            .add_cast(filter_tensor, TrtDataType::kFLOAT)
                            .map_err(|e| GraphError::ConversionFailed {
                                format: "trtx".to_string(),
                                reason: format!("Conv2d filter Half->Float cast: {}", e),
                            })?;
                        Some(cast_layer.get_output(&*network, 0).map_err(|e| {
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
                let bias_tensor_for_conv: Option<trtx::Tensor<'a>> =
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
                            Some(cast_layer.get_output(&*network, 0).map_err(|e| {
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

                let filter_layout_shuffle_out: Option<trtx::Tensor<'a>> = if filter_layout != "oihw"
                {
                    let perm = Self::conv_dynamic_filter_first_transpose(filter_layout)?;
                    let mut shuffle = network.add_shuffle(filter_tensor_to_use).map_err(|e| {
                        GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Conv2d dynamic filter layout shuffle: {}", e),
                        }
                    })?;
                    shuffle.set_first_transpose(network, &perm).map_err(|e| {
                        GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Conv2d dynamic filter set_first_transpose: {}", e),
                        }
                    })?;
                    Some(shuffle.get_output(&*network, 0).map_err(|e| {
                        GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Conv2d dynamic filter shuffle output: {}", e),
                        }
                    })?)
                } else {
                    None
                };
                let filter_for_set_input = filter_layout_shuffle_out
                    .as_ref()
                    .unwrap_or(filter_tensor_to_use);

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
                layer
                    .set_input(network, 1, filter_for_set_input)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Conv2d set_input(1) filter: {}", e),
                    })?;
                if let Some(bt) = bias_tensor_to_use {
                    layer
                        .set_input(network, 2, bt)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Conv2d set_input(2) bias: {}", e),
                        })?;
                }
                layer
            }
        };

        // Set layer properties (matches C++ API pattern: call setters after creation)
        layer.set_stride(network, &[strides[0] as i64, strides[1] as i64]);

        // No need to set padding on convolution layer - already handled by explicit padding layer
        layer.set_padding(network, &[0i64, 0]);

        layer.set_dilation(network, &[dilations[0] as i64, dilations[1] as i64]);

        layer.set_num_groups(network, groups as i64);

        // Extract output tensor from layer (NCHW, Float)
        let conv_output =
            layer
                .get_output(&*network, 0)
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
                .get_output(&*network, 0)
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
            shuffle
                .set_first_transpose(network, &[0, 2, 3, 1])
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Conv2d set_first_transpose NCHW->NHWC: {}", e),
                })?;
            shuffle
                .get_output(&*network, 0)
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
    fn add_conv_transpose2d_op<'a>(
        graph: &'a GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let (input_id, filter_id, deconv_opts) = match operation {
            Operation::ConvTranspose2d {
                input,
                filter,
                options,
                ..
            } => (*input, *filter, options.as_ref()),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "add_conv_transpose2d_op: expected ConvTranspose2d operation"
                        .to_string(),
                });
            }
        };
        // Same as conv2d: bias from MLConvTranspose2dOptions.bias (optional third JSON input merged at parse time).
        let bias_id = deconv_opts.and_then(|o| o.bias);

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
        let filter_layout = deconv_opts
            .map(|o| o.filter_layout.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("iohw");
        let (in_ch, out_ch, h, w): (u32, u32, u32, u32) = match filter_layout {
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
        let num_output_maps = (out_ch as i32) * groups;
        let kernel_size: [i32; 2] = [h as i32, w as i32];

        let filter_constant = graph
            .constant_operand_ids_to_handles
            .contains_key(&filter_id);

        let (filter_data_to_use, bias_data, deconv_weights_owned) = if filter_constant {
            let filter_shape_u32 = filter_operand.descriptor.static_or_max_shape();
            let filter_data = Self::get_constant_data(graph, filter_id)?;
            let mut bias_promoted: Option<Vec<u8>> = None;
            let bias_raw: Option<&[u8]> = match bias_id {
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
                        bias_promoted = Some(Self::f16_bytes_to_f32_bytes(raw)?);
                        None
                    } else {
                        Some(raw)
                    }
                }
                None => None,
            };
            let filter_dtype = filter_operand.descriptor.data_type;
            let mut kernel_promoted: Option<Vec<u8>> = None;
            match (filter_dtype, filter_layout) {
                (DataType::Float16, _) => {
                    let f32_bytes = Self::f16_bytes_to_f32_bytes(filter_data)?;
                    let iohw = if filter_layout == "iohw" {
                        f32_bytes
                    } else {
                        Self::deconv_filter_to_iohw(&f32_bytes, filter_layout, &filter_shape_u32)?
                    };
                    kernel_promoted = Some(iohw);
                }
                (DataType::Float32, "iohw") => {}
                (DataType::Float32, _) => {
                    kernel_promoted = Some(Self::deconv_filter_to_iohw(
                        filter_data,
                        filter_layout,
                        &filter_shape_u32,
                    )?);
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
            let deconv_weights_owned = if kernel_promoted.is_some() || bias_promoted.is_some() {
                let deconv_kernel_const_shape: Vec<i64> =
                    vec![in_ch as i64, out_ch as i64, h as i64, w as i64];
                let kernel_values = kernel_promoted.unwrap_or_else(|| filter_data.to_vec());
                let bias_values = match (bias_promoted, bias_raw) {
                    (Some(b), _) => Some(b),
                    (None, Some(br)) => Some(br.to_vec()),
                    (None, None) => None,
                };
                Some(OwnedConvWeights {
                    kernel: OwnedWeights {
                        shape: deconv_kernel_const_shape,
                        data_type: TrtDataType::kFLOAT,
                        values: kernel_values,
                    },
                    bias: bias_values.map(|values| OwnedWeights {
                        shape: vec![num_output_maps as i64],
                        data_type: TrtDataType::kFLOAT,
                        values,
                    }),
                })
            } else {
                None
            };
            if deconv_weights_owned.is_some() {
                (None, None, deconv_weights_owned)
            } else {
                (Some(filter_data), bias_raw, None)
            }
        } else {
            if let Some(id) = bias_id {
                if graph.constant_operand_ids_to_handles.contains_key(&id) {
                    return Err(GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: "convTranspose2d with non-constant filter requires bias to be a tensor input (constant bias not supported)".to_string(),
                    });
                }
            }
            (None, None, None)
        };

        let input_layout = deconv_opts
            .map(|o| o.input_layout.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("nchw");
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

        let nhwc_shuffle_output =
            if input_layout == "nhwc" {
                let mut shuffle =
                    network
                        .add_shuffle(input)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("convTranspose2d NHWC->NCHW shuffle: {}", e),
                        })?;
                shuffle
                    .set_first_transpose(network, &[0, 3, 1, 2])
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("convTranspose2d set_first_transpose NHWC->NCHW: {}", e),
                    })?;
                Some(shuffle.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("convTranspose2d NHWC shuffle output: {}", e),
                    }
                })?)
            } else {
                None
            };
        let pre_deconv_input = nhwc_shuffle_output.as_ref().unwrap_or(input);

        let half_cast_output: Option<trtx::Tensor<'a>> =
            if input_dtype == DataType::Float16 {
                let cast_layer = network
                    .add_cast(pre_deconv_input, TrtDataType::kFLOAT)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("convTranspose2d Half->Float cast: {}", e),
                    })?;
                Some(cast_layer.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("convTranspose2d cast output: {}", e),
                    }
                })?)
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
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("convTranspose2d identity output: {}", e),
                })?
        };

        let kernel_size_i64: [i64; 2] = [kernel_size[0] as i64, kernel_size[1] as i64];
        let mut layer = match (deconv_weights_owned, filter_data_to_use) {
            (Some(owned), _) => network
                .add_deconvolution_owned_weights(
                    &deconv_input,
                    num_output_maps as i64,
                    &kernel_size_i64,
                    owned,
                )
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to add deconvolution (owned weights): {}", e),
                })?,
            (None, Some(fd)) => {
                let deconv_weights = trtx::ConvWeights {
                    kernel_weights: fd,
                    kernel_dtype: TrtDataType::kFLOAT,
                    bias_weights: bias_data,
                    bias_dtype: bias_data.map(|_| TrtDataType::kFLOAT),
                };
                network
                    .add_deconvolution(
                        &deconv_input,
                        num_output_maps as i64,
                        &kernel_size_i64,
                        &deconv_weights,
                    )
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add deconvolution: {}", e),
                    })?
            }
            (None, None) => {
                let filter_tensor =
                    tensor_map
                        .get(&filter_id)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("Filter operand {} tensor not found", filter_id),
                        })?;
                let filter_tensor_for_conv: Option<trtx::Tensor<'a>> =
                    if filter_operand.descriptor.data_type == DataType::Float16
                        && input_dtype == DataType::Float16
                    {
                        let cast_layer = network
                            .add_cast(filter_tensor, TrtDataType::kFLOAT)
                            .map_err(|e| GraphError::ConversionFailed {
                                format: "trtx".to_string(),
                                reason: format!("convTranspose2d filter Half->Float cast: {}", e),
                            })?;
                        Some(cast_layer.get_output(&*network, 0).map_err(|e| {
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
                let bias_tensor_for_conv: Option<trtx::Tensor<'a>> = if let (Some(bt), Some(bid)) =
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
                        Some(cast_layer.get_output(&*network, 0).map_err(|e| {
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

                let filter_layout_shuffle_out: Option<trtx::Tensor<'a>> = if filter_layout != "iohw"
                {
                    let perm = Self::deconv_dynamic_filter_first_transpose(filter_layout)?;
                    let mut shuffle = network.add_shuffle(filter_tensor_to_use).map_err(|e| {
                        GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("convTranspose2d dynamic filter layout shuffle: {}", e),
                        }
                    })?;
                    shuffle.set_first_transpose(network, &perm).map_err(|e| {
                        GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!(
                                "convTranspose2d dynamic filter set_first_transpose: {}",
                                e
                            ),
                        }
                    })?;
                    Some(shuffle.get_output(&*network, 0).map_err(|e| {
                        GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("convTranspose2d dynamic filter shuffle output: {}", e),
                        }
                    })?)
                } else {
                    None
                };
                let filter_for_set_input = filter_layout_shuffle_out
                    .as_ref()
                    .unwrap_or(filter_tensor_to_use);

                let deconv_weights = trtx::ConvWeights {
                    kernel_weights: &[],
                    kernel_dtype: TrtDataType::kFLOAT,
                    bias_weights: None,
                    bias_dtype: None,
                };
                let mut layer = network
                    .add_deconvolution(
                        &deconv_input,
                        num_output_maps as i64,
                        &[kernel_size[0] as i64, kernel_size[1] as i64],
                        &deconv_weights,
                    )
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to add deconvolution (tensor weights): {}", e),
                    })?;
                layer
                    .set_input(network, 1, filter_for_set_input)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("convTranspose2d set_input(1) filter: {}", e),
                    })?;
                if let Some(bt) = bias_tensor_to_use {
                    layer
                        .set_input(network, 2, bt)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("convTranspose2d set_input(2) bias: {}", e),
                        })?;
                }
                layer
            }
        };

        layer
            .set_stride(network, &[strides[0] as i64, strides[1] as i64])
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("convTranspose2d set_stride: {}", e),
            })?;
        layer
            .set_dilation(network, &[dilations[0] as i64, dilations[1] as i64])
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
        let pre: [i64; 2] = [pre_effective[0] as i64, pre_effective[1] as i64];
        let post: [i64; 2] = [post_effective[0] as i64, post_effective[1] as i64];
        layer
            .set_pre_padding(network, &pre)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("convTranspose2d set_pre_padding: {}", e),
            })?;
        layer
            .set_post_padding(network, &post)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("convTranspose2d set_post_padding: {}", e),
            })?;

        layer
            .set_num_groups(network, groups as i64)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("convTranspose2d set_num_groups: {}", e),
            })?;

        let deconv_output =
            layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get deconvolution output: {}", e),
                })?;

        // When padding could not fully absorb outputPadding, add IPaddingLayer for remainder.
        let deconv_output = if padding_remainder[0] != 0 || padding_remainder[1] != 0 {
            let pre_pad: Vec<i64> = vec![0, 0];
            let post_pad: Vec<i64> = vec![padding_remainder[0] as i64, padding_remainder[1] as i64];
            let pad_layer = network
                .add_padding(&deconv_output, &pre_pad, &post_pad)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("convTranspose2d outputPadding remainder: {}", e),
                })?;
            pad_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("convTranspose2d outputPadding remainder output: {}", e),
                })?
        } else {
            deconv_output
        };

        // When outputSizes (or output_shape) is specified, the graph output has explicit spatial
        // dimensions. Resize deconv output to match: slice if larger, pad if smaller.
        // Prefer options.output_sizes when present so spatial targets are correct even if the
        // output operand descriptor lacks static dims (get_static_or_max_size would yield 0).
        let output_id = operation.output_operands_slice()[0];
        let spatial_adjusted = match graph.operand(input_id) {
            Some(input_operand) => {
                let in_shape = &input_operand.descriptor.shape;
                if in_shape.len() != 4 {
                    deconv_output
                } else {
                    let targets: Option<(i32, i32, i32)> = if let Some(sizes) =
                        deconv_opts.and_then(|o| o.output_sizes.as_ref())
                        && sizes.len() >= 2
                    {
                        Some((sizes[0] as i32, sizes[1] as i32, num_output_maps))
                    } else if let Some(output_operand) = graph.operand(output_id) {
                        let out_shape = &output_operand.descriptor.shape;
                        if out_shape.len() != 4 {
                            None
                        } else if input_layout == "nhwc" {
                            Some((
                                get_static_or_max_size(&out_shape[1]) as i32,
                                get_static_or_max_size(&out_shape[2]) as i32,
                                get_static_or_max_size(&out_shape[3]) as i32,
                            ))
                        } else {
                            Some((
                                get_static_or_max_size(&out_shape[2]) as i32,
                                get_static_or_max_size(&out_shape[3]) as i32,
                                get_static_or_max_size(&out_shape[1]) as i32,
                            ))
                        }
                    } else {
                        None
                    };

                    match targets {
                        Some((target_h, target_w, out_c)) if target_h > 0 && target_w > 0 => {
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
                            let out_batch = get_static_or_max_size(&in_shape[0]) as i32;
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
                                let start: Vec<i64> = vec![0, 0, 0, 0];
                                let size: Vec<i64> = vec![
                                    out_batch as i64,
                                    out_c as i64,
                                    slice_h as i64,
                                    slice_w as i64,
                                ];
                                let stride: Vec<i64> = vec![1, 1, 1, 1];
                                let slice_layer = network
                                    .add_slice(&current, &start, &size, &stride)
                                    .map_err(|e| GraphError::ConversionFailed {
                                        format: "trtx".to_string(),
                                        reason: format!("convTranspose2d outputSizes slice: {}", e),
                                    })?;
                                current = slice_layer.get_output(&*network, 0).map_err(|e| {
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
                                let pre: Vec<i64> = vec![0, 0];
                                let post: Vec<i64> = vec![pad_h as i64, pad_w as i64];
                                let pad_layer = network
                                    .add_padding(&current, &pre, &post)
                                    .map_err(|e| GraphError::ConversionFailed {
                                        format: "trtx".to_string(),
                                        reason: format!("convTranspose2d outputSizes pad: {}", e),
                                    })?;
                                pad_layer.get_output(&*network, 0).map_err(|e| {
                                    GraphError::ConversionFailed {
                                        format: "trtx".to_string(),
                                        reason: format!(
                                            "convTranspose2d outputSizes pad output: {}",
                                            e
                                        ),
                                    }
                                })?
                            } else {
                                current
                            }
                        }
                        _ => deconv_output,
                    }
                }
            }
            None => deconv_output,
        };

        let conv_output = if input_dtype == DataType::Float16 {
            let cast_layer = network
                .add_cast(&spatial_adjusted, TrtDataType::kHALF)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("convTranspose2d Float->Half cast: {}", e),
                })?;
            cast_layer
                .get_output(&*network, 0)
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
            shuffle
                .set_first_transpose(network, &[0, 2, 3, 1])
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("convTranspose2d set_first_transpose NCHW->NHWC: {}", e),
                })?;
            shuffle
                .get_output(&*network, 0)
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

    /// Pool kernel `[H, W]` from [`MLPool2dOptions`] and input tensor descriptor.
    /// When `window_dimensions` is omitted, WebNN uses the full spatial extent (matches ONNX
    /// `create_pool2d_attributes_with_graph`).
    fn pool2d_window_from_graph_input(
        graph: &GraphInfo,
        input_id: u32,
        op: &Operation,
        opts: &MLPool2dOptions,
    ) -> Result<[i64; 2], GraphError> {
        let has_explicit_window = opts
            .window_dimensions
            .as_ref()
            .is_some_and(|w| w.len() >= 2);
        if let Some(wd) = opts.window_dimensions.as_ref()
            && wd.len() >= 2
        {
            return Ok([wd[0] as i64, wd[1] as i64]);
        }

        let in_operand = graph
            .operand(input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Pool2d input operand {} not found", input_id),
            })?;

        let layout = opts.layout.to_ascii_lowercase();
        let layout = if layout.is_empty() {
            "nchw"
        } else {
            layout.as_str()
        };

        let shape = &in_operand.descriptor.shape;
        if shape.len() != 4 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Pool2d default window requires 4D input, got {}D",
                    shape.len()
                ),
            });
        }

        let mut h = if layout == "nhwc" {
            get_static_or_max_size(&shape[1]) as i64
        } else {
            get_static_or_max_size(&shape[2]) as i64
        };
        let mut w = if layout == "nhwc" {
            get_static_or_max_size(&shape[2]) as i64
        } else {
            get_static_or_max_size(&shape[3]) as i64
        };

        // Align with ONNX converter: averagePool2d with dilations and no explicit window uses
        // full input spatial extent as the kernel.
        if matches!(op, Operation::AveragePool2d { .. })
            && !opts.dilations.is_empty()
            && !has_explicit_window
            && let Some(out_id) = op.output_operand()
            && let Some(out_operand) = graph.operand(out_id)
            && out_operand.descriptor.shape.len() == 4
        {
            let (in_h, in_w) = if layout == "nhwc" {
                (
                    get_static_or_max_size(&shape[1]),
                    get_static_or_max_size(&shape[2]),
                )
            } else {
                (
                    get_static_or_max_size(&shape[2]),
                    get_static_or_max_size(&shape[3]),
                )
            };
            h = in_h as i64;
            w = in_w as i64;
        }

        Ok([h, w])
    }

    /// Concatenate same-rank tensors along `axis` (see `IConcatenationLayer::setAxis`).
    fn trtx_concat_tensors_along_axis<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensors: &[trtx::Tensor<'a>],
        axis: i32,
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        if tensors.is_empty() {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "trtx_concat_tensors_along_axis: empty input list".to_string(),
            });
        }
        if tensors.len() == 1 {
            let id_layer =
                network
                    .add_identity(&tensors[0])
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("pool concat identity: {e}"),
                    })?;
            return id_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("pool concat identity output: {e}"),
                });
        }
        let inputs: Vec<&trtx::Tensor<'a>> = tensors.iter().collect();
        let mut layer =
            network
                .add_concatenation(&inputs)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("pool concat: {e}"),
                })?;
        layer.set_axis(network, axis);
        layer
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("pool concat output: {e}"),
            })
    }

    /// Dilated max-pool: TensorRT `IPoolingLayer` has no dilation; decompose into slices + `kMAX`.
    ///
    /// `input` must be NCHW. WebNN padding order:
    /// `[beginning_height, ending_height, beginning_width, ending_width]`.
    fn add_max_pool2d_dilated_via_slices<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        input: &trtx::Tensor<'a>,
        operation: &Operation,
        opts: &MLPool2dOptions,
        window_hw: [i64; 2],
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        const MAX_DILATED_POOL_TAPS: usize = 4096;

        let dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dilated max pool input dimensions: {e}"),
            })?;
        if dims.len() != 4 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "dilated max pool expects 4D NCHW input, got {}D",
                    dims.len()
                ),
            });
        }
        let n = dims[0];
        let c = dims[1];
        let h_in = dims[2];
        let w_in = dims[3];
        if n <= 0 || c <= 0 || h_in <= 0 || w_in <= 0 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dilated max pool invalid dims {dims:?}"),
            });
        }

        let kh = window_hw[0] as u32;
        let kw = window_hw[1] as u32;
        if kh == 0 || kw == 0 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "dilated max pool zero window".to_string(),
            });
        }

        let sh = opts.strides.get(0).copied().unwrap_or(1);
        let sw = opts.strides.get(1).copied().unwrap_or(1);
        if sh == 0 || sw == 0 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "dilated max pool zero stride".to_string(),
            });
        }

        let dh = opts.dilations.get(0).copied().unwrap_or(1);
        let dw = opts.dilations.get(1).copied().unwrap_or(1);
        if dh == 0 || dw == 0 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "dilated max pool zero dilation".to_string(),
            });
        }

        let (pb_h, pe_h, pb_w, pe_w) = if opts.padding.len() >= 4 {
            (
                opts.padding[0],
                opts.padding[1],
                opts.padding[2],
                opts.padding[3],
            )
        } else {
            (0, 0, 0, 0)
        };

        let shape_u32: Vec<u32> = dims
            .iter()
            .map(|&d| u32::try_from(d.max(0)).unwrap_or(0))
            .collect();
        let pool_opts = MLPool2dOptions {
            window_dimensions: Some(vec![kh, kw]),
            strides: vec![sh, sw],
            dilations: vec![dh, dw],
            padding: vec![pb_h, pe_h, pb_w, pe_w],
            layout: "nchw".to_string(),
            output_shape_rounding: if Self::trtx_pool_padding_round_up(graph, operation, opts)? {
                "ceil".to_string()
            } else {
                "floor".to_string()
            },
            ..Default::default()
        };
        let out_shape = infer_pool2d_shape(&shape_u32, &pool_opts).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dilated max pool shape inference: {e}"),
            }
        })?;
        let oh = out_shape[2];
        let ow = out_shape[3];

        let num_taps = (kh as usize)
            .saturating_mul(kw as usize)
            .saturating_mul(oh as usize)
            .saturating_mul(ow as usize);
        if num_taps > MAX_DILATED_POOL_TAPS {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "dilated max pool too many taps ({num_taps} > {MAX_DILATED_POOL_TAPS}); reduce size or dilation"
                ),
            });
        }

        let neg_inf = f32::NEG_INFINITY.to_ne_bytes();
        let nc_count = (n * c) as usize;
        let mut neg_inf_weights = Vec::with_capacity(nc_count * 4);
        for _ in 0..nc_count {
            neg_inf_weights.extend_from_slice(&neg_inf);
        }
        let neg_inf_layer = network
            .add_constant_owned(&[n, c, 1, 1], neg_inf_weights, TrtDataType::kFLOAT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dilated max pool neg-inf constant: {e}"),
            })?;

        let kh_i = kh as i32;
        let kw_i = kw as i32;
        let dil_hi = dh as i32;
        let dil_wi = dw as i32;
        let sh_i = sh as i32;
        let sw_i = sw as i32;
        let pb_hi = pb_h as i32;
        let pb_wi = pb_w as i32;
        let h_in_i = h_in as i32;
        let w_in_i = w_in as i32;

        let mut rows: Vec<trtx::Tensor<'a>> = Vec::with_capacity(oh as usize);

        for oh_idx in 0..oh {
            let mut cols: Vec<trtx::Tensor<'a>> = Vec::with_capacity(ow as usize);
            for ow_idx in 0..ow {
                let h_start = (oh_idx as i32) * sh_i - pb_hi;
                let w_start = (ow_idx as i32) * sw_i - pb_wi;

                let mut taps: Vec<trtx::Tensor<'a>> = Vec::with_capacity((kh * kw) as usize);
                for ih in 0..kh_i {
                    for iw in 0..kw_i {
                        let h_idx = h_start + ih * dil_hi;
                        let w_idx = w_start + iw * dil_wi;
                        let tap_tensor =
                            if h_idx >= 0 && w_idx >= 0 && h_idx < h_in_i && w_idx < w_in_i {
                                let slice_layer = network
                                    .add_slice(
                                        input,
                                        &[0, 0, h_idx as i64, w_idx as i64],
                                        &[n, c, 1, 1],
                                        &[1, 1, 1, 1],
                                    )
                                    .map_err(|e| GraphError::ConversionFailed {
                                        format: "trtx".to_string(),
                                        reason: format!("dilated max pool slice: {e}"),
                                    })?;
                                slice_layer.get_output(&*network, 0).map_err(|e| {
                                    GraphError::ConversionFailed {
                                        format: "trtx".to_string(),
                                        reason: format!("dilated max pool slice output: {e}"),
                                    }
                                })?
                            } else {
                                neg_inf_layer.get_output(&*network, 0).map_err(|e| {
                                    GraphError::ConversionFailed {
                                        format: "trtx".to_string(),
                                        reason: format!("dilated max pool neg-inf tap: {e}"),
                                    }
                                })?
                            };
                        taps.push(tap_tensor);
                    }
                }

                if taps.is_empty() {
                    return Err(GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: "dilated max pool empty tap list".to_string(),
                    });
                }
                let mut acc = taps.remove(0);
                for t in taps {
                    let ew = network
                        .add_elementwise(&acc, &t, ElementWiseOperation::kMAX)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("dilated max pool elementwise max: {e}"),
                        })?;
                    acc =
                        ew.get_output(&*network, 0)
                            .map_err(|e| GraphError::ConversionFailed {
                                format: "trtx".to_string(),
                                reason: format!("dilated max pool max output: {e}"),
                            })?;
                }
                cols.push(acc);
            }
            let row = Self::trtx_concat_tensors_along_axis(network, &cols, 3)?;
            rows.push(row);
        }

        let out_nchw = Self::trtx_concat_tensors_along_axis(network, &rows, 2)?;

        // Match graph output dtype (pool path otherwise keeps input type).
        let out_id = operation.output_operands_slice()[0];
        let out_operand = graph
            .operand(out_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dilated max pool output operand {out_id} not found"),
            })?;
        let need_half = out_operand.descriptor.data_type == DataType::Float16;
        if need_half {
            let cast_layer = network
                .add_cast(&out_nchw, TrtDataType::kHALF)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("dilated max pool Float->Half: {e}"),
                })?;
            cast_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("dilated max pool cast output: {e}"),
                })
        } else {
            Ok(out_nchw)
        }
    }

    /// Dilated average-pool: same slice grid as max-pool; OOB taps are 0; sum then scale by `1/(kh*kw)`.
    ///
    /// `input` must be NCHW. Denominator is the full window size (WebNN / ONNX dilated average pool).
    fn add_average_pool2d_dilated_via_slices<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        input: &trtx::Tensor<'a>,
        operation: &Operation,
        opts: &MLPool2dOptions,
        window_hw: [i64; 2],
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        const MAX_DILATED_POOL_TAPS: usize = 4096;

        let dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dilated average pool input dimensions: {e}"),
            })?;
        if dims.len() != 4 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "dilated average pool expects 4D NCHW input, got {}D",
                    dims.len()
                ),
            });
        }
        let n = dims[0];
        let c = dims[1];
        let h_in = dims[2];
        let w_in = dims[3];
        if n <= 0 || c <= 0 || h_in <= 0 || w_in <= 0 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dilated average pool invalid dims {dims:?}"),
            });
        }

        let kh = window_hw[0] as u32;
        let kw = window_hw[1] as u32;
        if kh == 0 || kw == 0 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "dilated average pool zero window".to_string(),
            });
        }

        let sh = opts.strides.get(0).copied().unwrap_or(1);
        let sw = opts.strides.get(1).copied().unwrap_or(1);
        if sh == 0 || sw == 0 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "dilated average pool zero stride".to_string(),
            });
        }

        let dh = opts.dilations.get(0).copied().unwrap_or(1);
        let dw = opts.dilations.get(1).copied().unwrap_or(1);
        if dh == 0 || dw == 0 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "dilated average pool zero dilation".to_string(),
            });
        }

        let (pb_h, pe_h, pb_w, pe_w) = if opts.padding.len() >= 4 {
            (
                opts.padding[0],
                opts.padding[1],
                opts.padding[2],
                opts.padding[3],
            )
        } else {
            (0, 0, 0, 0)
        };

        let shape_u32: Vec<u32> = dims
            .iter()
            .map(|&d| u32::try_from(d.max(0)).unwrap_or(0))
            .collect();
        let pool_opts = MLPool2dOptions {
            window_dimensions: Some(vec![kh, kw]),
            strides: vec![sh, sw],
            dilations: vec![dh, dw],
            padding: vec![pb_h, pe_h, pb_w, pe_w],
            layout: "nchw".to_string(),
            output_shape_rounding: if Self::trtx_pool_padding_round_up(graph, operation, opts)? {
                "ceil".to_string()
            } else {
                "floor".to_string()
            },
            ..Default::default()
        };
        let out_shape = infer_pool2d_shape(&shape_u32, &pool_opts).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dilated average pool shape inference: {e}"),
            }
        })?;
        let oh = out_shape[2];
        let ow = out_shape[3];

        let num_taps = (kh as usize)
            .saturating_mul(kw as usize)
            .saturating_mul(oh as usize)
            .saturating_mul(ow as usize);
        if num_taps > MAX_DILATED_POOL_TAPS {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "dilated average pool too many taps ({num_taps} > {MAX_DILATED_POOL_TAPS}); reduce size or dilation"
                ),
            });
        }

        let z = 0.0f32.to_ne_bytes();
        let nc_count = (n * c) as usize;
        let mut zero_weights = Vec::with_capacity(nc_count * 4);
        for _ in 0..nc_count {
            zero_weights.extend_from_slice(&z);
        }
        let zero_layer = network
            .add_constant_owned(&[n, c, 1, 1], zero_weights, TrtDataType::kFLOAT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dilated average pool zero constant: {e}"),
            })?;

        let kh_i = kh as i32;
        let kw_i = kw as i32;
        let dil_hi = dh as i32;
        let dil_wi = dw as i32;
        let sh_i = sh as i32;
        let sw_i = sw as i32;
        let pb_hi = pb_h as i32;
        let pb_wi = pb_w as i32;
        let h_in_i = h_in as i32;
        let w_in_i = w_in as i32;

        let mut rows: Vec<trtx::Tensor<'a>> = Vec::with_capacity(oh as usize);

        for oh_idx in 0..oh {
            let mut cols: Vec<trtx::Tensor<'a>> = Vec::with_capacity(ow as usize);
            for ow_idx in 0..ow {
                let h_start = (oh_idx as i32) * sh_i - pb_hi;
                let w_start = (ow_idx as i32) * sw_i - pb_wi;

                let mut taps: Vec<trtx::Tensor<'a>> = Vec::with_capacity((kh * kw) as usize);
                for ih in 0..kh_i {
                    for iw in 0..kw_i {
                        let h_idx = h_start + ih * dil_hi;
                        let w_idx = w_start + iw * dil_wi;
                        let tap_tensor =
                            if h_idx >= 0 && w_idx >= 0 && h_idx < h_in_i && w_idx < w_in_i {
                                let slice_layer = network
                                    .add_slice(
                                        input,
                                        &[0, 0, h_idx as i64, w_idx as i64],
                                        &[n, c, 1, 1],
                                        &[1, 1, 1, 1],
                                    )
                                    .map_err(|e| GraphError::ConversionFailed {
                                        format: "trtx".to_string(),
                                        reason: format!("dilated average pool slice: {e}"),
                                    })?;
                                slice_layer.get_output(&*network, 0).map_err(|e| {
                                    GraphError::ConversionFailed {
                                        format: "trtx".to_string(),
                                        reason: format!("dilated average pool slice output: {e}"),
                                    }
                                })?
                            } else {
                                zero_layer.get_output(&*network, 0).map_err(|e| {
                                    GraphError::ConversionFailed {
                                        format: "trtx".to_string(),
                                        reason: format!("dilated average pool zero tap: {e}"),
                                    }
                                })?
                            };
                        taps.push(tap_tensor);
                    }
                }

                if taps.is_empty() {
                    return Err(GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: "dilated average pool empty tap list".to_string(),
                    });
                }
                let mut acc = taps.remove(0);
                for t in taps {
                    let ew = network
                        .add_elementwise(&acc, &t, ElementWiseOperation::kSUM)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("dilated average pool elementwise sum: {e}"),
                        })?;
                    acc =
                        ew.get_output(&*network, 0)
                            .map_err(|e| GraphError::ConversionFailed {
                                format: "trtx".to_string(),
                                reason: format!("dilated average pool sum output: {e}"),
                            })?;
                }
                cols.push(acc);
            }
            let row = Self::trtx_concat_tensors_along_axis(network, &cols, 3)?;
            rows.push(row);
        }

        let summed_nchw = Self::trtx_concat_tensors_along_axis(network, &rows, 2)?;

        // `add_scale` Weights take raw pointers; passing `&f32::to_ne_bytes()` temporaries is unsafe
        // because TRT may retain pointers until engine build. Use owned constant + elementwise prod.
        let inv = 1.0f32 / ((kh as f32) * (kw as f32));
        let inv_const_layer = network
            .add_constant_owned(
                &[1, 1, 1, 1],
                inv.to_ne_bytes().to_vec(),
                TrtDataType::kFLOAT,
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dilated average pool inv scale constant: {e}"),
            })?;
        let inv_tensor =
            inv_const_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("dilated average pool inv scale output: {e}"),
                })?;
        let prod_layer = network
            .add_elementwise(&summed_nchw, &inv_tensor, ElementWiseOperation::kPROD)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dilated average pool multiply by 1/n: {e}"),
            })?;
        let out_nchw =
            prod_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("dilated average pool prod output: {e}"),
                })?;

        let out_id = operation.output_operands_slice()[0];
        let out_operand = graph
            .operand(out_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dilated average pool output operand {out_id} not found"),
            })?;
        let need_half = out_operand.descriptor.data_type == DataType::Float16;
        if need_half {
            let cast_layer = network
                .add_cast(&out_nchw, TrtDataType::kHALF)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("dilated average pool Float->Half: {e}"),
                })?;
            cast_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("dilated average pool cast output: {e}"),
                })
        } else {
            Ok(out_nchw)
        }
    }

    /// Dilated L2 pool: `sqrt(sum(x^2))` over dilated taps — same slice grid as dilated average/max.
    ///
    /// `input` must be **x^2** in NCHW (typically `float32`; callers may upcast from half).
    fn add_l2_pool2d_dilated_via_slices<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        input: &trtx::Tensor<'a>,
        operation: &Operation,
        opts: &MLPool2dOptions,
        window_hw: [i64; 2],
    ) -> Result<trtx::Tensor<'a>, GraphError> {
        const MAX_DILATED_POOL_TAPS: usize = 4096;

        let dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dilated l2 pool input dimensions: {e}"),
            })?;
        if dims.len() != 4 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dilated l2 pool expects 4D NCHW input, got {}D", dims.len()),
            });
        }
        let n = dims[0];
        let c = dims[1];
        let h_in = dims[2];
        let w_in = dims[3];
        if n <= 0 || c <= 0 || h_in <= 0 || w_in <= 0 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dilated l2 pool invalid dims {dims:?}"),
            });
        }

        let kh = window_hw[0] as u32;
        let kw = window_hw[1] as u32;
        if kh == 0 || kw == 0 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "dilated l2 pool zero window".to_string(),
            });
        }

        let sh = opts.strides.get(0).copied().unwrap_or(1);
        let sw = opts.strides.get(1).copied().unwrap_or(1);
        if sh == 0 || sw == 0 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "dilated l2 pool zero stride".to_string(),
            });
        }

        let dh = opts.dilations.get(0).copied().unwrap_or(1);
        let dw = opts.dilations.get(1).copied().unwrap_or(1);
        if dh == 0 || dw == 0 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "dilated l2 pool zero dilation".to_string(),
            });
        }

        let (pb_h, pe_h, pb_w, pe_w) = if opts.padding.len() >= 4 {
            (
                opts.padding[0],
                opts.padding[1],
                opts.padding[2],
                opts.padding[3],
            )
        } else {
            (0, 0, 0, 0)
        };

        let shape_u32: Vec<u32> = dims
            .iter()
            .map(|&d| u32::try_from(d.max(0)).unwrap_or(0))
            .collect();
        let pool_opts = MLPool2dOptions {
            window_dimensions: Some(vec![kh, kw]),
            strides: vec![sh, sw],
            dilations: vec![dh, dw],
            padding: vec![pb_h, pe_h, pb_w, pe_w],
            layout: "nchw".to_string(),
            output_shape_rounding: if Self::trtx_pool_padding_round_up(graph, operation, opts)? {
                "ceil".to_string()
            } else {
                "floor".to_string()
            },
            ..Default::default()
        };
        let out_shape = infer_pool2d_shape(&shape_u32, &pool_opts).map_err(|e| {
            GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dilated l2 pool shape inference: {e}"),
            }
        })?;
        let oh = out_shape[2];
        let ow = out_shape[3];

        let num_taps = (kh as usize)
            .saturating_mul(kw as usize)
            .saturating_mul(oh as usize)
            .saturating_mul(ow as usize);
        if num_taps > MAX_DILATED_POOL_TAPS {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "dilated l2 pool too many taps ({num_taps} > {MAX_DILATED_POOL_TAPS}); reduce size or dilation"
                ),
            });
        }

        let z = 0.0f32.to_ne_bytes();
        let nc_count = (n * c) as usize;
        let mut zero_weights = Vec::with_capacity(nc_count * 4);
        for _ in 0..nc_count {
            zero_weights.extend_from_slice(&z);
        }
        let zero_layer = network
            .add_constant_owned(&[n, c, 1, 1], zero_weights, TrtDataType::kFLOAT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("dilated l2 pool zero constant: {e}"),
            })?;

        let kh_i = kh as i32;
        let kw_i = kw as i32;
        let dil_hi = dh as i32;
        let dil_wi = dw as i32;
        let sh_i = sh as i32;
        let sw_i = sw as i32;
        let pb_hi = pb_h as i32;
        let pb_wi = pb_w as i32;
        let h_in_i = h_in as i32;
        let w_in_i = w_in as i32;

        let mut rows: Vec<trtx::Tensor<'a>> = Vec::with_capacity(oh as usize);

        for oh_idx in 0..oh {
            let mut cols: Vec<trtx::Tensor<'a>> = Vec::with_capacity(ow as usize);
            for ow_idx in 0..ow {
                let h_start = (oh_idx as i32) * sh_i - pb_hi;
                let w_start = (ow_idx as i32) * sw_i - pb_wi;

                let mut taps: Vec<trtx::Tensor<'a>> = Vec::with_capacity((kh * kw) as usize);
                for ih in 0..kh_i {
                    for iw in 0..kw_i {
                        let h_idx = h_start + ih * dil_hi;
                        let w_idx = w_start + iw * dil_wi;
                        let tap_tensor =
                            if h_idx >= 0 && w_idx >= 0 && h_idx < h_in_i && w_idx < w_in_i {
                                let slice_layer = network
                                    .add_slice(
                                        input,
                                        &[0, 0, h_idx as i64, w_idx as i64],
                                        &[n, c, 1, 1],
                                        &[1, 1, 1, 1],
                                    )
                                    .map_err(|e| GraphError::ConversionFailed {
                                        format: "trtx".to_string(),
                                        reason: format!("dilated l2 pool slice: {e}"),
                                    })?;
                                slice_layer.get_output(&*network, 0).map_err(|e| {
                                    GraphError::ConversionFailed {
                                        format: "trtx".to_string(),
                                        reason: format!("dilated l2 pool slice output: {e}"),
                                    }
                                })?
                            } else {
                                zero_layer.get_output(&*network, 0).map_err(|e| {
                                    GraphError::ConversionFailed {
                                        format: "trtx".to_string(),
                                        reason: format!("dilated l2 pool zero tap: {e}"),
                                    }
                                })?
                            };
                        taps.push(tap_tensor);
                    }
                }

                if taps.is_empty() {
                    return Err(GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: "dilated l2 pool empty tap list".to_string(),
                    });
                }
                let mut acc = taps.remove(0);
                for t in taps {
                    let ew = network
                        .add_elementwise(&acc, &t, ElementWiseOperation::kSUM)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("dilated l2 pool elementwise sum: {e}"),
                        })?;
                    acc =
                        ew.get_output(&*network, 0)
                            .map_err(|e| GraphError::ConversionFailed {
                                format: "trtx".to_string(),
                                reason: format!("dilated l2 pool sum output: {e}"),
                            })?;
                }
                cols.push(acc);
            }
            let row = Self::trtx_concat_tensors_along_axis(network, &cols, 3)?;
            rows.push(row);
        }

        let sum_sq_nchw = Self::trtx_concat_tensors_along_axis(network, &rows, 2)?;

        // Keep sum(x^2) in Float32: values often exceed f16 range before sqrt; caller casts after sqrt.
        Ok(sum_sq_nchw)
    }

    /// TensorRT `kEXPLICIT_ROUND_UP` when WebNN needs ceil spatial size: `outputShapeRounding` or
    /// [`output_sizes`](crate::operator_options::MLPool2dOptions::output_sizes) matching the ceil implicit shape.
    fn trtx_pool_padding_round_up(
        graph: &GraphInfo,
        operation: &Operation,
        opts: &MLPool2dOptions,
    ) -> Result<bool, GraphError> {
        match infer_pool2d_ceil_mode_from_output_sizes(operation, graph) {
            Some(1) => Ok(true),
            Some(0) => Ok(false),
            None => {
                if opts.output_sizes.as_ref().is_some_and(|v| v.len() >= 2) {
                    return Err(GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason:
                            "pool2d output_sizes do not match floor or ceil implicit output shapes"
                                .to_string(),
                    });
                }
                Ok(opts.output_shape_rounding.eq_ignore_ascii_case("ceil"))
            }
            Some(_) => Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "pool2d: unexpected ceil_mode value from output_sizes inference"
                    .to_string(),
            }),
        }
    }

    /// Apply WebNN `MLPool2dOptions` stride and asymmetric padding to a TensorRT pooling layer.
    /// Padding order: `[beginning_height, ending_height, beginning_width, ending_width]`.
    fn trtx_apply_pool2d_options<'a>(
        layer: &mut PoolingLayer<'a>,
        network: &mut trtx::NetworkDefinition<'a>,
        opts: &MLPool2dOptions,
        round_up: bool,
    ) {
        let stride_h = opts.strides.get(0).copied().unwrap_or(1) as i64;
        let stride_w = opts.strides.get(1).copied().unwrap_or(1) as i64;
        layer.set_stride_nd(network, &[stride_h, stride_w]);

        let (pre_h, post_h, pre_w, post_w) = if opts.padding.len() >= 4 {
            (
                opts.padding[0] as i64,
                opts.padding[1] as i64,
                opts.padding[2] as i64,
                opts.padding[3] as i64,
            )
        } else {
            (0, 0, 0, 0)
        };
        // NCHW spatial order: pre/post per dimension [H, W].
        layer.set_pre_padding(network, &[pre_h, pre_w]);
        layer.set_post_padding(network, &[post_h, post_w]);

        if round_up {
            layer.set_padding_mode(network, PaddingMode::kEXPLICIT_ROUND_UP);
        }
    }

    /// Add pooling operation
    fn add_pooling_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
        pool_type: PoolingType,
    ) -> Result<(), GraphError> {
        let (input_id, opts_ref) = match operation {
            Operation::AveragePool2d { input, options, .. }
            | Operation::MaxPool2d { input, options, .. } => (*input, options.as_ref()),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "add_pooling_op: expected AveragePool2d or MaxPool2d".to_string(),
                });
            }
        };

        let default_pool = MLPool2dOptions::default();
        let opts = opts_ref.unwrap_or(&default_pool);

        let input = tensor_map
            .get(&input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", input_id),
            })?;

        let window = Self::pool2d_window_from_graph_input(graph, input_id, operation, opts)?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];

        let layout = match opts.layout.as_str() {
            "" => "nchw",
            s => s,
        };

        // TensorRT pooling is NCHW; WebNN NHWC must be transposed first (same as conv2d).
        let nhwc_to_nchw: Option<trtx::Tensor<'a>> =
            if layout == "nhwc" {
                let mut shuffle =
                    network
                        .add_shuffle(input)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("pool2d NHWC->NCHW shuffle: {e}"),
                        })?;
                shuffle
                    .set_first_transpose(network, &[0, 3, 1, 2])
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("pool2d set_first_transpose NHWC: {e}"),
                    })?;
                Some(shuffle.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("pool2d NHWC shuffle output: {e}"),
                    }
                })?)
            } else {
                None
            };
        let pool_nchw_in = nhwc_to_nchw.as_ref().unwrap_or(input);

        let dh = opts.dilations.get(0).copied().unwrap_or(1);
        let dw = opts.dilations.get(1).copied().unwrap_or(1);
        if dh != 1 || dw != 1 {
            if pool_type != PoolingType::kMAX && pool_type != PoolingType::kAVERAGE {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason:
                        "TRTX: pooling with dilations other than 1 is only supported for maxPool2d and averagePool2d"
                            .to_string(),
                });
            }

            let input_dtype = graph
                .operand(input_id)
                .map(|o| o.descriptor.data_type)
                .unwrap_or(DataType::Float32);

            let half_to_float: Option<trtx::Tensor<'a>> = if input_dtype == DataType::Float16 {
                let cast_layer = network
                    .add_cast(pool_nchw_in, TrtDataType::kFLOAT)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("pool2d dilated Half->Float: {e}"),
                    })?;
                Some(cast_layer.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("pool2d dilated cast output: {e}"),
                    }
                })?)
            } else {
                None
            };
            let dilated_input = half_to_float.as_ref().unwrap_or(pool_nchw_in);

            let out_nchw = if pool_type == PoolingType::kMAX {
                Self::add_max_pool2d_dilated_via_slices(
                    graph,
                    network,
                    dilated_input,
                    operation,
                    opts,
                    window,
                )?
            } else {
                Self::add_average_pool2d_dilated_via_slices(
                    graph,
                    network,
                    dilated_input,
                    operation,
                    opts,
                    window,
                )?
            };

            let output = if layout == "nhwc" {
                let mut shuffle =
                    network
                        .add_shuffle(&out_nchw)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("pool2d dilated NCHW->NHWC shuffle: {e}"),
                        })?;
                shuffle
                    .set_first_transpose(network, &[0, 2, 3, 1])
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("pool2d dilated NCHW->NHWC transpose: {e}"),
                    })?;
                shuffle
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("pool2d dilated NCHW shuffle output: {e}"),
                    })?
            } else {
                out_nchw
            };

            tensor_map.insert(output_id, output);
            return Ok(());
        }

        let round_up = Self::trtx_pool_padding_round_up(graph, operation, opts)?;

        let mut layer = network
            .add_pooling(pool_nchw_in, pool_type, &window)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add pooling: {}", e),
            })?;

        Self::trtx_apply_pool2d_options(&mut layer, network, opts, round_up);

        let pooled_nchw =
            layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get layer output: {}", e),
                })?;

        let output = if layout == "nhwc" {
            let mut shuffle =
                network
                    .add_shuffle(&pooled_nchw)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("pool2d NCHW->NHWC shuffle: {e}"),
                    })?;
            shuffle
                .set_first_transpose(network, &[0, 2, 3, 1])
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("pool2d set_first_transpose NCHW: {e}"),
                })?;
            shuffle
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("pool2d NCHW shuffle output: {e}"),
                })?
        } else {
            pooled_nchw
        };

        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add softmax operation
    fn add_softmax_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        // Axis is required by WebNN spec (unsigned long)
        let positive_axis = match operation {
            Operation::Softmax { axis, .. } => *axis,
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "expected Softmax operation".to_string(),
                });
            }
        };

        // TensorRT uses a bitmask where bit N represents axis N

        // Create bitmask for the axis
        let axes = 1u32 << positive_axis;

        let layer = network
            .add_softmax(input, Axes::from_bits(axes))
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add softmax: {}", e),
            })?;

        // Extract output tensor from layer
        let output = layer
            .get_output(&*network, 0)
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
    fn add_concat_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let inputs: Vec<&trtx::Tensor<'a>> = operation
            .input_operands()
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

        // WebNN `axis` is a concat() method parameter (see spec).
        let axis_raw = match operation {
            Operation::Concat { axis, .. } => *axis as i64,
            _ => operation
                .attributes()
                .get("axis")
                .and_then(|v| v.as_i64().or_else(|| v.as_u64().map(|u| u as i64)))
                .unwrap_or(0),
        };
        let ndim = inputs[0]
            .dimensions(&*network)
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
        layer.set_axis(network, axis_i32);

        // Extract output tensor from layer
        let output = layer
            .get_output(&*network, 0)
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
    fn add_transpose_op<'a>(
        _graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let input_dims = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Transpose: failed to get input dimensions: {}", e),
            })?;

        let rank = input_dims.len();
        // WebNN default: when permutation is omitted, reverse axes [rank-1, ..., 0].
        let perm: Vec<i32> = operation
            .attributes()
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
            .set_first_transpose(network, perm.as_slice())
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set transpose permutation: {}", e),
            })?;

        let output = layer
            .get_output(&*network, 0)
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
    fn add_reshape_op<'a>(
        _graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let new_shape = match operation {
            Operation::Reshape { new_shape, .. } => new_shape,
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "reshape operation expected".to_string(),
                });
            }
        };
        if new_shape.is_empty() {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "reshape operation missing 'newShape'".to_string(),
            });
        }

        let dims: Vec<i64> = crate::operator_options::mldimensions_static_or_max(new_shape)
            .into_iter()
            .map(|u| u as i64)
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
            .set_reshape_dimensions(network, &dims)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to set reshape dimensions: {}", e),
            })?;

        // Extract output tensor from layer
        let output = layer
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get layer output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add resample2d operation (resize/interpolate 2D tensor).
    ///
    /// WebNN: <https://www.w3.org/TR/webnn/#api-mlgraphbuilder-resample2d-method>
    /// Defaults per `MLResample2dOptions`: `axes` = [2, 3], `scales` = [1.0, 1.0] when omitted;
    /// `sizes` (length 2) overrides `scales`.
    fn add_resample2d_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let (input_id, opts) = match operation {
            Operation::Resample2d { input, options, .. } => {
                (*input, options.clone().unwrap_or_default())
            }
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "resample2d: expected Operation::Resample2d".to_string(),
                });
            }
        };

        let input = tensor_map
            .get(&input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("resample2d: input operand {input_id} not found"),
            })?;

        let input_operand =
            graph
                .operand(input_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("resample2d: operand {input_id} missing from graph"),
                })?;

        let dims_trt = input
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("resample2d: input dimensions: {e}"),
            })?;
        let desc_shape: Vec<i64> = input_operand
            .descriptor
            .shape
            .iter()
            .map(|d| get_static_or_max_size(d) as i64)
            .collect();

        let rank = if !dims_trt.is_empty() {
            dims_trt.len()
        } else if !desc_shape.is_empty() {
            desc_shape.len()
        } else {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: "resample2d: cannot determine input rank".to_string(),
            });
        };

        if rank != 4 {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("resample2d: WebNN requires a 4-D input tensor, got rank {rank}"),
            });
        }

        let mut input_shape = vec![1_i64; rank];
        for i in 0..rank {
            let from_trt = dims_trt.get(i).copied().unwrap_or(0);
            let from_desc = desc_shape.get(i).copied().unwrap_or(1);
            input_shape[i] = if from_trt > 0 {
                from_trt
            } else {
                from_desc.max(1)
            };
        }

        let axes: Vec<usize> = if opts.axes.len() == 2 {
            opts.axes.iter().map(|&a| a as usize).collect()
        } else {
            vec![2, 3]
        };
        if axes.len() != 2 || axes[0] >= rank || axes[1] >= rank || axes[0] == axes[1] {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "resample2d: axes must be two distinct valid indices for rank {rank}, got {:?}",
                    opts.axes
                ),
            });
        }

        let spatial_sizes: Vec<i64> = if let Some(ref sizes) = opts.sizes {
            if sizes.len() != 2 {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("resample2d: sizes must have length 2, got {:?}", sizes),
                });
            }
            sizes.iter().map(|&u| u as i64).collect()
        } else {
            let (s0, s1) = if opts.scales.is_empty() {
                (1.0_f32, 1.0_f32)
            } else if opts.scales.len() == 2 {
                (opts.scales[0], opts.scales[1])
            } else {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!(
                        "resample2d: scales must have length 2 or be omitted (default [1, 1]), got {:?}",
                        opts.scales
                    ),
                });
            };
            vec![
                (input_shape[axes[0]].max(1) as f64 * f64::from(s0))
                    .round()
                    .max(1.0) as i64,
                (input_shape[axes[1]].max(1) as f64 * f64::from(s1))
                    .round()
                    .max(1.0) as i64,
            ]
        };

        let mut output_dims = input_shape.clone();
        output_dims[axes[0]] = spatial_sizes[0];
        output_dims[axes[1]] = spatial_sizes[1];

        let mode_norm = if opts.mode.is_empty() {
            "nearest-neighbor".to_string()
        } else {
            opts.mode.to_ascii_lowercase()
        };
        let resize_mode = match mode_norm.as_str() {
            "linear" => ResizeMode::kLINEAR,
            "nearest-neighbor" | "nearest" => ResizeMode::kNEAREST,
            _ => ResizeMode::kNEAREST,
        };

        let mut layer = network
            .add_resize(input)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add resize layer: {}", e),
            })?;

        layer.set_output_dimensions(network, &output_dims);

        // Set resize mode (uses ResizeMode typedef for InterpolationMode)
        layer.set_resize_mode(network, resize_mode);
        // WebNN `resample2d` uses the half-pixel grid; TensorRT default is kASYMMETRIC.
        layer.set_coordinate_transformation(network, ResizeCoordinateTransformation::kHALF_PIXEL);
        // WebNN nearest: `ceil(coord - 0.5)` == `floor(coord + 0.5)` on the sampling grid; TRT default
        // `kFLOOR` uses `floor(coord)` and mis-aligns upsampling (e.g. 2x nearest on float grid).
        if resize_mode == ResizeMode::kNEAREST {
            layer.set_nearest_rounding(network, ResizeRoundMode::kHALF_UP);
        }

        let output = layer
            .get_output(&*network, 0)
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
    fn add_is_nan_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_tensor = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        // NaN is the only value where x != x is true
        // Use elementwise EQUAL operation with itself, then negate
        let layer = network
            .add_elementwise(input_tensor, input_tensor, ElementWiseOperation::kEQUAL)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create EQUAL layer for isNaN: {}", e),
            })?;

        let equal_output =
            layer
                .get_output(&*network, 0)
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

        let bool_output =
            not_layer
                .get_output(&*network, 0)
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
    fn add_is_infinite_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_tensor = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        // Check if abs(x) == infinity
        // First compute abs(x)
        let abs_layer = network
            .add_unary(input_tensor, UnaryOperation::kABS)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create ABS layer for isInfinite: {}", e),
            })?;

        let abs_output =
            abs_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get ABS output: {}", e),
                })?;

        // Infinity constant must match `abs` tensor rank. `[1]` is rank-1; 0D scalars use `abs` rank 0
        // and TRT rejects elementwise `[]` vs `[1]` (no broadcast between those ranks).
        let abs_rank = abs_output
            .dimensions(&*network)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("isInfinite: abs output dimensions: {e}"),
            })?
            .len();
        let inf_dims: Vec<i64> = vec![1i64; abs_rank];
        let abs_trt_type = abs_output.get_type(&*network);
        let (inf_bytes, inf_trt_dtype) = match abs_trt_type {
            TrtDataType::kHALF => (
                f16::from_f32(f32::INFINITY)
                    .to_bits()
                    .to_le_bytes()
                    .to_vec(),
                TrtDataType::kHALF,
            ),
            TrtDataType::kFLOAT => (f32::INFINITY.to_le_bytes().to_vec(), TrtDataType::kFLOAT),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!(
                        "isInfinite: expected float16 or float32 abs tensor, got TRT dtype id {:?}",
                        abs_trt_type
                    ),
                });
            }
        };
        let inf_constant = network
            .add_small_constant_copied(&inf_dims, inf_bytes.as_slice(), inf_trt_dtype)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create infinity constant: {}", e),
            })?;

        let inf_tensor =
            inf_constant
                .get_output(&*network, 0)
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

        let bool_output =
            equal_layer
                .get_output(&*network, 0)
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
    fn add_round_even_op<'a>(
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_tensor = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        // TensorRT's kROUND already uses round-to-nearest-even (banker's rounding) by default
        let layer = network
            .add_unary(input_tensor, UnaryOperation::kROUND)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create ROUND layer: {}", e),
            })?;

        let output = layer
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get roundEven output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add gatherElements operation (gather using index tensor along axis).
    /// Clamps indices to `[-dim_size, dim_size - 1]` on the gather axis (WebNN / WPT), matching
    /// [`add_gather_op`]: TensorRT Gather otherwise returns 0 for out-of-range indices.
    fn add_gather_elements_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let data_tensor = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Data operand {} not found", operation.input_operands()[0]),
            })?;

        let indices_tensor = tensor_map
            .get(&operation.input_operands()[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "Indices operand {} not found",
                    operation.input_operands()[1]
                ),
            })?;

        let axis = operation
            .attributes()
            .as_gather()
            .map(|o| o.axis as i32)
            .unwrap_or(0);

        let data_operand = graph
            .operand(operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "gatherElements: data operand {} not in graph",
                    operation.input_operands()[0]
                ),
            })?;
        let axis_usize = axis as usize;
        if axis_usize >= data_operand.descriptor.shape.len() {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "gatherElements axis {} out of bounds for rank {}",
                    axis,
                    data_operand.descriptor.shape.len()
                ),
            });
        }
        let dim_size = get_static_or_max_size(&data_operand.descriptor.shape[axis_usize]) as i32;

        let indices_operand = graph
            .operand(operation.input_operands()[1])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!(
                    "gatherElements: indices operand {} not in graph",
                    operation.input_operands()[1]
                ),
            })?;
        let indices_shape: Vec<i32> = indices_operand
            .descriptor
            .shape
            .iter()
            .map(|d| get_static_or_max_size(d) as i32)
            .collect();
        let indices_shape_i64: Vec<i64> = indices_shape.iter().map(|&d| d as i64).collect();

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
        let min_const = network
            .add_small_constant_copied(&indices_shape_i64, &min_data, trtx::DataType::kINT32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("gatherElements: clamp min constant: {}", e),
            })?;
        let max_const = network
            .add_small_constant_copied(&indices_shape_i64, &max_data, trtx::DataType::kINT32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("gatherElements: clamp max constant: {}", e),
            })?;

        let min_const_out =
            min_const
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("gatherElements: clamp min output: {}", e),
                })?;
        let max_const_out =
            max_const
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("gatherElements: clamp max output: {}", e),
                })?;

        let clamped_upper = network
            .add_elementwise(indices_tensor, &max_const_out, ElementWiseOperation::kMIN)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("gatherElements: indices clamp (min upper): {}", e),
            })?;
        let clamped_upper_out =
            clamped_upper
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("gatherElements: clamp upper output: {}", e),
                })?;

        let clamped = network
            .add_elementwise(
                &min_const_out,
                &clamped_upper_out,
                ElementWiseOperation::kMAX,
            )
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("gatherElements: indices clamp (max lower): {}", e),
            })?;
        let clamped_indices =
            clamped
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("gatherElements: clamped indices output: {}", e),
                })?;

        let mut layer = network
            .add_gather(data_tensor, &clamped_indices, axis)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create gather layer: {}", e),
            })?;

        layer.set_gather_mode(network, trtx::GatherMode::kELEMENT);

        let output = layer
            .get_output(&*network, 0)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to get gatherElements output: {}", e),
            })?;

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add l2Pool2d operation (L2 pooling: WebNN/ONNX LpPool p=2 is `sqrt(sum(x^2))` over the window,
    /// not RMS: square → mean pool → multiply by kernel area → sqrt).
    fn add_l2_pool2d_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let (input_id, opts_ref) = match operation {
            Operation::L2Pool2d { input, options, .. } => (*input, options.as_ref()),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: "add_l2_pool2d_op: expected L2Pool2d".to_string(),
                });
            }
        };

        let input_tensor =
            tensor_map
                .get(&input_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Input operand {} not found", input_id),
                })?;

        // Step 1: Square the input (x^2)
        let square_layer = network
            .add_elementwise(input_tensor, input_tensor, ElementWiseOperation::kPROD)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create square layer for l2Pool2d: {}", e),
            })?;

        let squared =
            square_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get squared output: {}", e),
                })?;

        // Step 2: Apply average pooling (same window resolution as averagePool2d / ONNX)
        let default_pool = MLPool2dOptions::default();
        let opts = opts_ref.unwrap_or(&default_pool);

        let dh = opts.dilations.get(0).copied().unwrap_or(1);
        let dw = opts.dilations.get(1).copied().unwrap_or(1);

        let window = Self::pool2d_window_from_graph_input(graph, input_id, operation, opts)?;

        let layout = match opts.layout.as_str() {
            "" => "nchw",
            s => s,
        };

        let nhwc_to_nchw: Option<trtx::Tensor<'a>> =
            if layout == "nhwc" {
                let mut shuffle =
                    network
                        .add_shuffle(&squared)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("l2Pool2d NHWC->NCHW shuffle: {e}"),
                        })?;
                shuffle
                    .set_first_transpose(network, &[0, 3, 1, 2])
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("l2Pool2d set_first_transpose NHWC: {e}"),
                    })?;
                Some(shuffle.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("l2Pool2d NHWC shuffle output: {e}"),
                    }
                })?)
            } else {
                None
            };
        let pool_in = nhwc_to_nchw.as_ref().unwrap_or(&squared);

        let input_dtype = graph
            .operand(input_id)
            .map(|o| o.descriptor.data_type)
            .unwrap_or(DataType::Float32);

        let half_to_float: Option<trtx::Tensor<'a>> =
            if (dh != 1 || dw != 1) && input_dtype == DataType::Float16 {
                let cast_layer = network
                    .add_cast(pool_in, TrtDataType::kFLOAT)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("l2Pool2d dilated Half->Float: {e}"),
                    })?;
                Some(cast_layer.get_output(&*network, 0).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("l2Pool2d dilated cast output: {e}"),
                    }
                })?)
            } else {
                None
            };
        let dilated_pool_in = half_to_float.as_ref().unwrap_or(pool_in);

        let pooled_nchw = if dh != 1 || dw != 1 {
            Self::add_l2_pool2d_dilated_via_slices(
                graph,
                network,
                dilated_pool_in,
                operation,
                opts,
                window,
            )?
        } else {
            let round_up = Self::trtx_pool_padding_round_up(graph, operation, opts)?;

            let mut pool_layer = network
                .add_pooling(pool_in, PoolingType::kAVERAGE, &window)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to create pooling layer for l2Pool2d: {}", e),
                })?;

            Self::trtx_apply_pool2d_options(&mut pool_layer, network, opts, round_up);

            // Default TRT average pooling excludes padding from the divisor (overlap with unpadded input),
            // so the effective count varies per output cell. LpPool is sum(x²) over the full window on the
            // padded tensor, then sqrt — match that by dividing every position by kh*kw.
            pool_layer.set_average_count_excludes_padding(network, false);

            let mean_sq_nchw =
                pool_layer
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("Failed to get pooled output: {}", e),
                    })?;

            // Mean of squares × (kh * kw) = sum of squares (Lp norm^2 before the final sqrt).
            // Sum of squares can exceed f16 range (~65504); do this multiply in Float32 when input is half.
            let k_vol = (window[0].max(0) as f32) * (window[1].max(0) as f32);
            let mean_for_prod = if input_dtype == DataType::Float16 {
                let cast_layer = network
                    .add_cast(&mean_sq_nchw, TrtDataType::kFLOAT)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("l2Pool2d mean_sq Half->Float: {e}"),
                    })?;
                cast_layer
                    .get_output(&*network, 0)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("l2Pool2d mean_sq cast output: {e}"),
                    })?
            } else {
                mean_sq_nchw
            };
            let k_const_layer = network
                .add_constant_owned(
                    &[1, 1, 1, 1],
                    k_vol.to_ne_bytes().to_vec(),
                    TrtDataType::kFLOAT,
                )
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("l2Pool2d kernel volume constant: {e}"),
                })?;
            let k_tensor = k_const_layer.get_output(&*network, 0).map_err(|e| {
                GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("l2Pool2d kernel volume output: {e}"),
                }
            })?;
            let sum_sq_layer = network
                .add_elementwise(&mean_for_prod, &k_tensor, ElementWiseOperation::kPROD)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("l2Pool2d mean_sq to sum_sq: {e}"),
                })?;
            sum_sq_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("l2Pool2d sum_sq output: {e}"),
                })?
        };

        let pooled = if layout == "nhwc" {
            let mut shuffle =
                network
                    .add_shuffle(&pooled_nchw)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "trtx".to_string(),
                        reason: format!("l2Pool2d NCHW->NHWC shuffle: {e}"),
                    })?;
            shuffle
                .set_first_transpose(network, &[0, 2, 3, 1])
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("l2Pool2d set_first_transpose NCHW: {e}"),
                })?;
            shuffle
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("l2Pool2d NCHW shuffle output: {e}"),
                })?
        } else {
            pooled_nchw
        };

        // Step 3: Take square root
        let sqrt_layer = network
            .add_unary(&pooled, UnaryOperation::kSQRT)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create sqrt layer for l2Pool2d: {}", e),
            })?;

        let sqrt_out =
            sqrt_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("Failed to get l2Pool2d output: {}", e),
                })?;

        let output = if input_dtype == DataType::Float16 {
            let cast_layer = network
                .add_cast(&sqrt_out, TrtDataType::kHALF)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("l2Pool2d sqrt Float->Half: {e}"),
                })?;
            cast_layer
                .get_output(&*network, 0)
                .map_err(|e| GraphError::ConversionFailed {
                    format: "trtx".to_string(),
                    reason: format!("l2Pool2d sqrt half cast output: {e}"),
                })?
        } else {
            sqrt_out
        };

        let output_ids = operation.output_operands_slice();
        let output_id = output_ids[0];
        tensor_map.insert(output_id, output);
        Ok(())
    }

    /// Add reverse operation (reverse elements along axes) - PLACEHOLDER
    /// Add reverse operation (reverse elements along axes using negative stride slicing)
    fn add_reverse_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_tensor = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        // Get input shape from graph
        let input_operand = &graph.operands[operation.input_operands()[0] as usize];
        let shape = &input_operand.descriptor.shape;
        let rank = shape.len();

        // Get axes to reverse: axes not present => all; axes=[] => none; axes=[..] => those.
        let axes_to_reverse: Vec<usize> = operation
            .attributes()
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
        let mut starts: Vec<i64> = vec![0; rank];
        let sizes: Vec<i64> = shape
            .iter()
            .map(|s| get_static_or_max_size(s) as i64)
            .collect();
        let mut strides: Vec<i64> = vec![1; rank];

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
            starts[axis] = (get_static_or_max_size(&shape[axis]) - 1) as i64;
            strides[axis] = -1;
        }

        let layer = network
            .add_slice(input_tensor, &starts, &sizes, &strides)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add slice layer for reverse: {}", e),
            })?;

        let output = layer
            .get_output(&*network, 0)
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
    fn add_cumulative_sum_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_tensor = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let attrs = operation.attributes();
        let cum_opts = attrs.as_cumulative_sum();
        let axis = match operation {
            Operation::CumulativeSum { axis, .. } => *axis as usize,
            _ => 0,
        };
        let exclusive = cum_opts.map(|o| o.exclusive).unwrap_or(false);
        let reverse = cum_opts.map(|o| o.reversed).unwrap_or(false);

        // Get input shape
        let input_operand = &graph.operands[operation.input_operands()[0] as usize];
        let shape = &input_operand.descriptor.shape;
        let rank = shape.len();

        if axis >= rank {
            return Err(GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("CumulativeSum axis {} out of range for rank {}", axis, rank),
            });
        }

        let axis_value = axis as i32;
        let axis_bytes = axis_value.to_le_bytes();

        // Create axis constant tensor (true 0D scalar with shape [])
        // TensorRT requires axisDims.nbDims == 0 for cumulative operations
        let axis_constant = network
            .add_small_constant_copied(&[], axis_bytes.as_slice(), trtx::DataType::kINT32)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create axis constant: {}", e),
            })?;

        let axis_tensor =
            axis_constant
                .get_output(&*network, 0)
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
            .get_output(&*network, 0)
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
    fn add_triangular_op<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let input_tensor = tensor_map
            .get(&operation.input_operands()[0])
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Input operand {} not found", operation.input_operands()[0]),
            })?;

        let attrs = operation.attributes();
        let tri_opts = attrs.as_triangular();
        let upper = tri_opts.and_then(|o| o.upper).unwrap_or(true);
        let diagonal = tri_opts.map(|o| o.diagonal).unwrap_or(0);

        // Get input shape from graph
        let input_operand = &graph.operands[operation.input_operands()[0] as usize];
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

        let input_dtype = input_operand.descriptor.data_type;
        // Elementwise PROD requires matching types; mask must match input (e.g. Half for float16).
        let (mask_bytes, mask_trt_ty) = match input_dtype {
            DataType::Float16 => {
                let mut bytes = Vec::with_capacity(total_elements * 2);
                for &f in &mask_data {
                    let v = if f == 1.0 { 1.0f32 } else { 0.0f32 };
                    bytes.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
                }
                (bytes, TrtDataType::kHALF)
            }
            _ => {
                let mask_bytes: Vec<u8> = mask_data.iter().flat_map(|&f| f.to_le_bytes()).collect();
                (mask_bytes, TrtDataType::kFLOAT)
            }
        };

        // Create constant layer with the mask
        let dims: Vec<i64> = shape
            .iter()
            .map(|s| get_static_or_max_size(s) as i64)
            .collect();
        let mask_layer = network
            .add_constant_owned(&dims, mask_bytes, mask_trt_ty)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to add constant mask for triangular: {}", e),
            })?;

        let mask_tensor =
            mask_layer
                .get_output(&*network, 0)
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

        let output =
            multiply_layer
                .get_output(&*network, 0)
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

        let mut builder =
            trtx::Builder::new(&logger).map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create TensorRT builder: {}", e),
            })?;

        let mut network = builder
            .create_network(trtx::builder::network_flags::EXPLICIT_BATCH)
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create TensorRT network: {}", e),
            })?;

        Self::build_network(graph_info, &mut network)?;

        // Create builder config
        let mut config = builder
            .create_config()
            .map_err(|e| GraphError::ConversionFailed {
                format: "trtx".to_string(),
                reason: format!("Failed to create builder config: {}", e),
            })?;

        // Set workspace size (1 GB)
        config.set_memory_pool_limit(trtx::builder::MemoryPoolType::kWORKSPACE, 1 << 30);

        // WPT / rustnnpt often compare TRT to strict IEEE fp32 (ulpTol=0). TF32 matmul/conv rounds to
        // 10-bit mantissa on supported GPUs and can differ by many ULP from CPU reference.
        config.clear_flag(trtx::builder::BuilderFlag::kTF32);

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
            data: engine_data.to_vec(),
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

    #[test]
    fn test_engine_binding_name_stable() {
        assert_eq!(
            TrtxConverter::engine_binding_name(0).as_str(),
            "webnn_operand_0"
        );
        assert_eq!(
            TrtxConverter::engine_binding_name(42).as_str(),
            "webnn_operand_42"
        );
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

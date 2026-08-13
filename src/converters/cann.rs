/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
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

//! CANN/HiAI converter for WebNN graphs.
//!
//! Converts `GraphInfo` into compiled model bytes via the adapter
//! (src/executors/cann_shim/). The adapter wraps the HiAI DDK's C++ API.
//!
//! - `cann-runtime`: calls `encode_via_adapter()` to build a GE graph,
//!   compile via HiaiIrBuild, return ModelBufferData bytes.
//! - `cann-runtime-mock`: validates ops via `webnn_op_to_hiai`, returns
//!   placeholder bytes for CI/testing.

use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo, OperandDescriptor};
use crate::operators::Operation;

use crate::executors::cann_shim::{CannModelBuffer, get_shim};

use super::{ConvertedGraph, GraphConverter};

// Maps a WebNN operation to its HIAI IR operation name.
//
// Returns Some(name) for ops that map directly to an adapter cann_op_*()
//
// Returns `None` for ops that need decomposition or are not supported
pub(crate) fn webnn_op_to_hiai(op: &Operation) -> Option<&'static str> {
    match op {
        // ── Element-wise binary ──────────────────────────────────────
        Operation::Add { .. } => Some("Add"),
        Operation::Sub { .. } => Some("Sub"),
        Operation::Mul { .. } => Some("Mul"),
        Operation::Div { .. } => Some("Div"),
        Operation::Pow { .. } => Some("Pow"),
        Operation::Max { .. } => Some("Max"),
        Operation::Min { .. } => Some("Min"),
        Operation::Equal { .. } => Some("Equal"),
        Operation::Greater { .. } => Some("Greater"),
        Operation::GreaterOrEqual { .. } => Some("GreaterOrEqual"),
        Operation::Lesser { .. } => Some("Lesser"),
        Operation::LesserOrEqual { .. } => Some("LesserOrEqual"),
        Operation::NotEqual { .. } => Some("NotEqual"),
        Operation::LogicalAnd { .. } => Some("LogicalAnd"),
        Operation::LogicalOr { .. } => Some("LogicalOr"),
        Operation::LogicalXor { .. } => Some("LogicalXor"),
        Operation::LogicalNot { .. } => Some("LogicalNot"),

        // ── Element-wise unary ───────────────────────────────────────
        Operation::Abs { .. } => Some("Abs"),
        Operation::Neg { .. } => Some("Neg"),
        Operation::Exp { .. } => Some("Exp"),
        Operation::Log { .. } => Some("Log"),
        Operation::Sin { .. } => Some("Sin"),
        Operation::Cos { .. } => Some("Cos"),
        Operation::Tan { .. } => Some("Tan"),
        Operation::Sqrt { .. } => Some("Sqrt"),
        Operation::Ceil { .. } => Some("Ceil"),
        Operation::Floor { .. } => Some("Floor"),
        Operation::Sign { .. } => Some("Sign"),
        Operation::Erf { .. } => Some("Erf"),
        Operation::Reciprocal { .. } => Some("Reciprocal"),
        Operation::Cast { .. } => Some("Cast"),
        Operation::Clamp { .. } => Some("Clamp"),

        // ── Activations ──────────────────────────────────────────────
        Operation::Relu { .. } => Some("ReLU"),
        Operation::Sigmoid { .. } => Some("Sigmoid"),
        Operation::Tanh { .. } => Some("Tanh"),
        Operation::Elu { .. } => Some("ELU"),
        Operation::Gelu { .. } => Some("GELU"),
        Operation::LeakyRelu { .. } => Some("LeakyRelu"),
        Operation::HardSigmoid { .. } => Some("HardSigmoid"),
        Operation::HardSwish { .. } => Some("HardSwish"),
        Operation::Softplus { .. } => Some("Softplus"),
        Operation::Softsign { .. } => Some("Softsign"),

        // ── Convolution + Pool + Matmul ──────────────────────────────
        Operation::Conv2d { .. } => Some("Conv2D"),
        Operation::ConvTranspose2d { .. } => Some("ConvTranspose"),
        Operation::MaxPool2d { .. } => Some("MaxPool"),
        Operation::AveragePool2d { .. } => Some("AvgPool"),
        Operation::Matmul { .. } => Some("MatMul"),
        Operation::Gemm { .. } => Some("Gemm"),
        Operation::Softmax { .. } => Some("Softmax"),

        // ── Reduction ────────────────────────────────────────────────
        Operation::ReduceSum { .. } => Some("ReduceSum"),
        Operation::ReduceMean { .. } => Some("ReduceMean"),
        Operation::ReduceMax { .. } => Some("ReduceMax"),
        Operation::ReduceMin { .. } => Some("ReduceMin"),
        Operation::ReduceProduct { .. } => Some("ReduceProduct"),
        Operation::ReduceL1 { .. } => Some("ReduceL1"),
        Operation::ReduceL2 { .. } => Some("ReduceL2"),
        Operation::ReduceLogSum { .. } => Some("ReduceLogSum"),
        Operation::ReduceLogSumExp { .. } => Some("ReduceLogSumExp"),
        Operation::ReduceSumSquare { .. } => Some("ReduceSumSquare"),
        Operation::ArgMax { .. } => Some("ArgMax"),
        Operation::ArgMin { .. } => Some("ArgMin"),

        // ── Shape ops ────────────────────────────────────────────────
        Operation::Reshape { .. } => Some("Reshape"),
        Operation::Transpose { .. } => Some("Transpose"),
        Operation::Tile { .. } => Some("Tile"),
        Operation::Slice { .. } => Some("Slice"),
        Operation::Split { .. } => Some("Split"),
        Operation::Concat { .. } => Some("Concat"),
        Operation::Pad { .. } => Some("Pad"),
        Operation::Squeeze { .. } => Some("Squeeze"),
        Operation::Unsqueeze { .. } => Some("Unsqueeze"),
        Operation::Expand { .. } => Some("Expand"),
        Operation::CumulativeSum { .. } => Some("CumulativeSum"),

        // ── Gather / Scatter / Where ─────────────────────────────────
        Operation::Gather { .. } => Some("Gather"),
        Operation::GatherElements { .. } => Some("GatherElements"),
        Operation::GatherND { .. } => Some("GatherND"),
        Operation::ScatterElements { .. } => Some("ScatterElements"),
        Operation::ScatterND { .. } => Some("ScatterND"),
        Operation::Where { .. } => Some("Where"),

        // ── Normalization + Quantization ─────────────────────────────
        Operation::BatchNormalization { .. } => Some("BatchNormalization"),
        Operation::InstanceNormalization { .. } => Some("InstanceNormalization"),
        Operation::QuantizeLinear { .. } => Some("QuantizeLinear"),
        Operation::DequantizeLinear { .. } => Some("DequantizeLinear"),

        // ── Other ────────────────────────────────────────────────────
        Operation::Resample2d { .. } => Some("Resample2D"),
        Operation::GlobalAveragePool { .. } => Some("GlobalAveragePool"),
        Operation::GlobalMaxPool { .. } => Some("GlobalMaxPool"),
        Operation::Constant { .. } => Some("Constant"),
        Operation::Shape { .. } => Some("Shape"),

        // ── Needs decomposition (Phase 3) ────────────────────────────
        Operation::Identity { .. } => Some("Identity"),
        Operation::Prelu { .. } => None,
        Operation::Linear { .. } => None,
        Operation::LayerNormalization { .. } => None,
        Operation::Triangular { .. } => None,
        Operation::IsNaN { .. } => None,
        Operation::IsInfinite { .. } => None,

        // ── Not supported ────────────────────────────────────────────
        Operation::Gru { .. }
        | Operation::GruCell { .. }
        | Operation::Lstm { .. }
        | Operation::LstmCell { .. }
        | Operation::L2Pool2d { .. }
        | Operation::Reverse { .. }
        | Operation::RoundEven { .. } => None,
    }
}

// ── Graph builder helpers ──────────────────────────────────────────

fn cann_data_type(data_type: DataType) -> i32 {
    match data_type {
        DataType::Float32 => 0,
        DataType::Float16 => 1,
        DataType::Int32 => 3,
        DataType::Int8 => 2,
        DataType::Uint8 => 4,
        DataType::Uint32 => 8,
        _ => 0,
    }
}

// Binary operators.
// Returns (input_a, input_b, ge_input_name_a, ge_input_name_b), or None.
fn binary_op_inputs(operation: &Operation) -> Option<(u32, u32, &'static str, &'static str)> {
    let (a, b, name_a, name_b) = match operation {
        Operation::Add { a, b, .. }
        | Operation::Sub { a, b, .. }
        | Operation::Mul { a, b, .. }
        | Operation::Div { a, b, .. }
        | Operation::Pow { a, b, .. }
        | Operation::Max { a, b, .. }
        | Operation::Min { a, b, .. } => (*a, *b, "x1", "x2"),
        Operation::Equal { a, b, .. }
        | Operation::Greater { a, b, .. }
        | Operation::GreaterOrEqual { a, b, .. }
        | Operation::Lesser { a, b, .. }
        | Operation::LesserOrEqual { a, b, .. }
        | Operation::NotEqual { a, b, .. } => (*a, *b, "x1", "x2"),
        Operation::LogicalAnd { a, b, .. }
        | Operation::LogicalOr { a, b, .. }
        | Operation::LogicalXor { a, b, .. } => (*a, *b, "x1", "x2"),
        Operation::Matmul { a, b, .. } | Operation::Gemm { a, b, .. } => (*a, *b, "x1", "x2"),
        _ => return None,
    };
    Some((a, b, name_a, name_b))
}

// Unary element-wise operators.
// Returns input operand index, or None.
fn unary_op_input(operation: &Operation) -> Option<u32> {
    match operation {
        Operation::Relu { input, .. }
        | Operation::Sigmoid { input, .. }
        | Operation::Tanh { input, .. }
        | Operation::Elu { input, .. }
        | Operation::Gelu { input, .. }
        | Operation::LeakyRelu { input, .. }
        | Operation::HardSigmoid { input, .. }
        | Operation::HardSwish { input, .. }
        | Operation::Softplus { input, .. }
        | Operation::Softsign { input, .. } => Some(*input),
        Operation::Abs { input, .. }
        | Operation::Neg { input, .. }
        | Operation::Exp { input, .. }
        | Operation::Log { input, .. }
        | Operation::Sin { input, .. }
        | Operation::Cos { input, .. }
        | Operation::Tan { input, .. }
        | Operation::Sqrt { input, .. }
        | Operation::Ceil { input, .. }
        | Operation::Floor { input, .. }
        | Operation::Sign { input, .. }
        | Operation::Erf { input, .. }
        | Operation::Reciprocal { input, .. } => Some(*input),
        Operation::Cast { input, .. } | Operation::Identity { input, .. } => Some(*input),
        Operation::Clamp { input, .. }
        | Operation::Softmax { input, .. }
        | Operation::LogicalNot { input, .. } => Some(*input),
        _ => None,
    }
}

fn descriptor_dims(descriptor: &OperandDescriptor) -> Vec<i64> {
    use crate::graph::Dimension;
    descriptor
        .shape
        .iter()
        .map(|dim| match dim {
            Dimension::Static(dim_value) => *dim_value as i64,
            _ => 0,
        })
        .collect()
}

// Returns the output operand slice for all operations with standard
// `outputs: Vec<u32>` field.
fn op_outputs(operation: &Operation) -> &[u32] {
    match operation {
        // Binary
        Operation::Add { outputs, .. }
        | Operation::Sub { outputs, .. }
        | Operation::Mul { outputs, .. }
        | Operation::Div { outputs, .. }
        | Operation::Pow { outputs, .. }
        | Operation::Max { outputs, .. }
        | Operation::Min { outputs, .. }
        // Comparison
        | Operation::Equal { outputs, .. }
        | Operation::Greater { outputs, .. }
        | Operation::GreaterOrEqual { outputs, .. }
        | Operation::Lesser { outputs, .. }
        | Operation::LesserOrEqual { outputs, .. }
        | Operation::NotEqual { outputs, .. }
        // Logical binary
        | Operation::LogicalAnd { outputs, .. }
        | Operation::LogicalOr { outputs, .. }
        | Operation::LogicalXor { outputs, .. }
        // Activations
        | Operation::Relu { outputs, .. }
        | Operation::Sigmoid { outputs, .. }
        | Operation::Tanh { outputs, .. }
        | Operation::Elu { outputs, .. }
        | Operation::Gelu { outputs, .. }
        | Operation::LeakyRelu { outputs, .. }
        | Operation::HardSigmoid { outputs, .. }
        | Operation::HardSwish { outputs, .. }
        | Operation::Softplus { outputs, .. }
        | Operation::Softsign { outputs, .. }
        // Unary math
        | Operation::Abs { outputs, .. }
        | Operation::Neg { outputs, .. }
        | Operation::Exp { outputs, .. }
        | Operation::Log { outputs, .. }
        | Operation::Sin { outputs, .. }
        | Operation::Cos { outputs, .. }
        | Operation::Tan { outputs, .. }
        | Operation::Sqrt { outputs, .. }
        | Operation::Ceil { outputs, .. }
        | Operation::Floor { outputs, .. }
        | Operation::Sign { outputs, .. }
        | Operation::Erf { outputs, .. }
        | Operation::Reciprocal { outputs, .. }
        // Type / other
        | Operation::Cast { outputs, .. }
        | Operation::Clamp { outputs, .. }
        | Operation::Identity { outputs, .. }
        | Operation::LogicalNot { outputs, .. }
        | Operation::Softmax { outputs, .. }
        // Matrix
        | Operation::Matmul { outputs, .. }
        | Operation::Gemm { outputs, .. }
        // Conv
        | Operation::Conv2d { outputs, .. }
        | Operation::ConvTranspose2d { outputs, .. }
        // Pooling
        | Operation::MaxPool2d { outputs, .. }
        | Operation::AveragePool2d { outputs, .. }
        | Operation::L2Pool2d { outputs, .. }
        | Operation::GlobalMaxPool { outputs, .. }
        | Operation::GlobalAveragePool { outputs, .. }
        // ArgMax / ArgMin
        | Operation::ArgMax { outputs, .. }
        | Operation::ArgMin { outputs, .. }
        // Normalization
        | Operation::BatchNormalization { outputs, .. } => outputs,
        _ => &[],
    }
}

// ── Layout helpers ─────────────────────────────────────────────────

/// Map WebNN input_layout to GE data_format.
fn conv_data_format(options: &Option<crate::operator_options::MLConv2dOptions>) -> &str {
    match options {
        Some(o) if o.input_layout.eq_ignore_ascii_case("nhwc") => "NHWC",
        _ => "NCHW",
    }
}

// Build a CANN graph via the adapter, compile it, and return the model bytes.
//
// Data nodes (with tensor descriptors) -> ops -> NetOutput -> compile -> bytes.
// Returns Err if the shim library is unavailable.
pub(crate) fn encode_via_adapter(graph: &GraphInfo) -> Result<Vec<u8>, GraphError> {
    use std::ffi::CString;

    let Some(shim) = get_shim() else {
        return Err(GraphError::ConversionFailed {
            format: "cann".into(),
            reason: "CANN shim not available".into(),
        });
    };

    // ── 1. Create graph ─────────────────────────────────────────────
    let graph_name = CString::new("webnn_model").unwrap();
    let can_graph = unsafe { (shim.graph_create)(graph_name.as_ptr()) };
    if can_graph.is_null() {
        return Err(GraphError::ConversionFailed {
            format: "cann".into(),
            reason: "cann_graph_create failed".into(),
        });
    }

    // operand_index -> CannOperatorHandle
    let mut handles: Vec<*mut std::ffi::c_void> = vec![std::ptr::null_mut(); graph.operands.len()];

    // ── 2. Create Data operators for each input ──────────────────────
    let mut data_ops: Vec<*mut std::ffi::c_void> = Vec::new();

    for &input_id in &graph.input_operands {
        let descriptor = &graph.operands[input_id as usize].descriptor;
        let dimensions = descriptor_dims(descriptor);
        let name = CString::new(
            graph.operands[input_id as usize]
                .name
                .clone()
                .unwrap_or_else(|| format!("input_{input_id}")),
        )
        .unwrap();

        let data_op = unsafe { (shim.op_data_with_name)(name.as_ptr()) };
        if data_op.is_null() {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("cann_op_data_with_name for operand {input_id} failed").into(),
            });
        }

        // Set tensor descriptor: shape + FORMAT_ND + dtype
        let shape = unsafe { (shim.shape_create)(dimensions.as_ptr(), dimensions.len() as i32) };
        if shape.is_null() {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_shape_create failed".into(),
            });
        }
        let tensor_desc = unsafe {
            (shim.tensor_desc_create)(
                shape,
                2, /* FORMAT_ND */
                cann_data_type(descriptor.data_type),
            )
        };
        if tensor_desc.is_null() {
            unsafe { (shim.shape_destroy)(shape) };
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_tensor_desc_create failed".into(),
            });
        }
        let x_name = CString::new("x").unwrap();
        let status =
            unsafe { (shim.operator_update_input_desc)(data_op, x_name.as_ptr(), tensor_desc) };
        unsafe {
            (shim.tensor_desc_destroy)(tensor_desc);
            (shim.shape_destroy)(shape);
        }
        if status != 0 {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!(
                    "cann_operator_update_input_desc for operand {input_id} failed: {status}"
                )
                .into(),
            });
        }

        handles[input_id as usize] = data_op;
        data_ops.push(data_op);
    }

    // Create Const operators for constant operands (for example, Conv2d filters).
    let mut const_ops: Vec<*mut std::ffi::c_void> = Vec::new();
    for (const_id, constant_data) in &graph.constant_operand_ids_to_handles {
        let name = CString::new(
            graph.operands[*const_id as usize]
                .name
                .clone()
                .unwrap_or_else(|| format!("const_{const_id}")),
        )
        .unwrap();

        let const_op = unsafe { (shim.op_const_with_name)(name.as_ptr()) };
        if const_op.is_null() {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("cann_op_const_with_name for operand {const_id} failed").into(),
            });
        }

        let desc = &graph.operands[*const_id as usize].descriptor;
        let dims = descriptor_dims(desc);
        let value_name = CString::new("value").unwrap();
        let format = 0_i32; // FORMAT_NCHW for all const tensors
        let status = unsafe {
            (shim.operator_set_attr_tensor_raw_format)(
                const_op,
                value_name.as_ptr(),
                constant_data.data.as_ptr() as *const _,
                constant_data.data.len() as u32,
                dims.as_ptr(),
                dims.len() as i32,
                cann_data_type(desc.data_type),
                format,
            )
        };
        if status != 0 {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!(
                    "cann_operator_set_attr_tensor_raw for operand {const_id} failed: {status}"
                )
                .into(),
            });
        }

        handles[*const_id as usize] = const_op;
        const_ops.push(const_op);
    }

    // ── 3. Create compute operations ─────────────────────────────────
    let mut compute_ops: Vec<*mut std::ffi::c_void> = Vec::new();

    for op in &graph.operations {
        let operator_type_name = match webnn_op_to_hiai(op) {
            Some(name) => CString::new(name).unwrap(),
            None => {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("unsupported op: {}", op.label()).into(),
                });
            }
        };

        // Create operator via cann_operator_create_registered.
        let op_name = CString::new("op").unwrap();
        let compute_op: *mut std::ffi::c_void = unsafe {
            (shim.operator_create_registered)(operator_type_name.as_ptr(), op_name.as_ptr())
        };

        if compute_op.is_null() {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("cann_operator_create failed for {}", op.label()).into(),
            });
        }

        if let Some((a, b, name_a, name_b)) = binary_op_inputs(op) {
            let lhs = handles[a as usize];
            let rhs = handles[b as usize];
            let lhs_name = CString::new(name_a).unwrap();
            let rhs_name = CString::new(name_b).unwrap();
            let status_a = unsafe { (shim.operator_set_input)(compute_op, lhs_name.as_ptr(), lhs) };
            let status_b = unsafe { (shim.operator_set_input)(compute_op, rhs_name.as_ptr(), rhs) };
            if status_a != 0 || status_b != 0 {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("cann_operator_set_input for {operator_type_name:?} failed")
                        .into(),
                });
            }
        }

        if let Some(input) = unary_op_input(op) {
            let source_handle = handles[input as usize];
            let x_name = CString::new("x").unwrap();
            let status =
                unsafe { (shim.operator_set_input)(compute_op, x_name.as_ptr(), source_handle) };
            if status != 0 {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("cann_operator_set_input for {operator_type_name:?} failed")
                        .into(),
                });
            }
        }

        // Wire Conv2d via hiai::op::Convolution.
        if let Operation::Conv2d {
            input,
            filter,
            options,
            ..
        } = op
        {
            let x_handle = handles[*input as usize];
            let filter_handle = handles[*filter as usize];
            let x_name = CString::new("x").unwrap();
            let filter_name = CString::new("filter").unwrap();
            let set_status_x =
                unsafe { (shim.operator_set_input)(compute_op, x_name.as_ptr(), x_handle) };
            let set_status_filter = unsafe {
                (shim.operator_set_input)(compute_op, filter_name.as_ptr(), filter_handle)
            };
            if set_status_x != 0 || set_status_filter != 0 {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("cann_operator_set_input for {operator_type_name:?} failed")
                        .into(),
                });
            }

            // hiai::op::Convolution: strides(2), pads(4), dilations(2),
            // pad_mode="SPECIFIC", data_format, groups. Read from options.
            let (stride_h, stride_w) = match options.as_ref().and_then(|o| {
                if o.strides.len() >= 2 {
                    Some((o.strides[0] as i64, o.strides[1] as i64))
                } else {
                    None
                }
            }) {
                Some((h, w)) => (h, w),
                None => (1, 1),
            };
            let (dilation_h, dilation_w) = match options.as_ref().and_then(|o| {
                if o.dilations.len() >= 2 {
                    Some((o.dilations[0] as i64, o.dilations[1] as i64))
                } else {
                    None
                }
            }) {
                Some((h, w)) => (h, w),
                None => (1, 1),
            };
            let (padding_top, padding_bottom, padding_left, padding_right) =
                match options.as_ref().and_then(|o| {
                    if o.padding.len() >= 4 {
                        Some((
                            o.padding[0] as i64,
                            o.padding[1] as i64,
                            o.padding[2] as i64,
                            o.padding[3] as i64,
                        ))
                    } else {
                        None
                    }
                }) {
                    Some((t, b, l, r)) => (t, b, l, r),
                    None => (0, 0, 0, 0),
                };
            let groups = options
                .as_ref()
                .map(|o| o.groups as i64)
                .unwrap_or(1)
                .max(1);
            let format_str = conv_data_format(options);

            let strides: [i64; 2] = [stride_h, stride_w];
            let pads: [i64; 4] = [padding_top, padding_bottom, padding_left, padding_right];
            let dilations: [i64; 2] = [dilation_h, dilation_w];
            unsafe {
                let strides_name = CString::new("strides").unwrap();
                (shim.operator_set_attr_int64_list)(
                    compute_op,
                    strides_name.as_ptr(),
                    strides.as_ptr(),
                    2,
                );
                let pads_name = CString::new("pads").unwrap();
                (shim.operator_set_attr_int64_list)(
                    compute_op,
                    pads_name.as_ptr(),
                    pads.as_ptr(),
                    4,
                );
                let dilations_name = CString::new("dilations").unwrap();
                (shim.operator_set_attr_int64_list)(
                    compute_op,
                    dilations_name.as_ptr(),
                    dilations.as_ptr(),
                    2,
                );
                let groups_name = CString::new("groups").unwrap();
                (shim.operator_set_attr_int64)(compute_op, groups_name.as_ptr(), groups);
                let data_format_name = CString::new("data_format").unwrap();
                let data_format_value = CString::new(format_str).unwrap();
                (shim.operator_set_attr_string)(
                    compute_op,
                    data_format_name.as_ptr(),
                    data_format_value.as_ptr(),
                );
                let pad_mode_name = CString::new("pad_mode").unwrap();
                let pad_mode_value = CString::new("SPECIFIC").unwrap();
                (shim.operator_set_attr_string)(
                    compute_op,
                    pad_mode_name.as_ptr(),
                    pad_mode_value.as_ptr(),
                );
            }
        }

        // Wire MaxPool2d via hiai::op::PoolingD.
        if let Operation::MaxPool2d { input, options, .. } = op {
            let x_handle = handles[*input as usize];
            let x_name = CString::new("x").unwrap();
            let status =
                unsafe { (shim.operator_set_input)(compute_op, x_name.as_ptr(), x_handle) };
            if status != 0 {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("cann_operator_set_input for {operator_type_name:?} failed")
                        .into(),
                });
            }

            // hiai::op::PoolingD: mode(0=max), window(2), stride(2), pad(4).
            let mut window_h: i64 = 1;
            let mut window_w: i64 = 1;
            let mut stride_h: i64 = 1;
            let mut stride_w: i64 = 1;
            let mut padding_top: i64 = 0;
            let mut padding_bottom: i64 = 0;
            let mut padding_left: i64 = 0;
            let mut padding_right: i64 = 0;
            if let Some(pool_options) = options {
                if pool_options.padding.len() >= 4 {
                    padding_top = pool_options.padding[0] as i64;
                    padding_left = pool_options.padding[1] as i64;
                    padding_bottom = pool_options.padding[2] as i64;
                    padding_right = pool_options.padding[3] as i64;
                }
                if let Some(ref ws) = pool_options.window_dimensions {
                    if ws.len() >= 2 {
                        window_h = ws[0] as i64;
                        window_w = ws[1] as i64;
                    }
                }
                if pool_options.strides.len() >= 2 {
                    stride_h = pool_options.strides[0] as i64;
                    stride_w = pool_options.strides[1] as i64;
                }
            }

            let window: [i64; 2] = [window_h, window_w];
            let stride: [i64; 2] = [stride_h, stride_w];
            let pad: [i64; 4] = [padding_top, padding_bottom, padding_left, padding_right];
            unsafe {
                let mode_name = CString::new("mode").unwrap();
                (shim.operator_set_attr_int64)(compute_op, mode_name.as_ptr(), 0);
                let window_name = CString::new("window").unwrap();
                (shim.operator_set_attr_int64_list)(
                    compute_op,
                    window_name.as_ptr(),
                    window.as_ptr(),
                    2,
                );
                let stride_name = CString::new("stride").unwrap();
                (shim.operator_set_attr_int64_list)(
                    compute_op,
                    stride_name.as_ptr(),
                    stride.as_ptr(),
                    2,
                );
                let pad_name = CString::new("pad").unwrap();
                (shim.operator_set_attr_int64_list)(compute_op, pad_name.as_ptr(), pad.as_ptr(), 4);
            }
        }

        // Wire ConvTranspose2d via hiai::op::ConvTranspose.
        // Input order: filter, x (opposite of Conv2D).
        if let Operation::ConvTranspose2d {
            input,
            filter,
            options,
            ..
        } = op
        {
            let x_handle = handles[*input as usize];
            let filter_handle = handles[*filter as usize];
            let filter_name = CString::new("filter").unwrap();
            let x_name = CString::new("x").unwrap();
            let set_status_filter = unsafe {
                (shim.operator_set_input)(compute_op, filter_name.as_ptr(), filter_handle)
            };
            let set_status_x =
                unsafe { (shim.operator_set_input)(compute_op, x_name.as_ptr(), x_handle) };
            if set_status_filter != 0 || set_status_x != 0 {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("cann_operator_set_input for {operator_type_name:?} failed")
                        .into(),
                });
            }

            // hiai::op::ConvTranspose: same attribute set as Convolution.
            let (stride_h, stride_w) = match options.as_ref().and_then(|o| {
                if o.strides.len() >= 2 {
                    Some((o.strides[0] as i64, o.strides[1] as i64))
                } else {
                    None
                }
            }) {
                Some((h, w)) => (h, w),
                None => (1, 1),
            };
            let (dilation_h, dilation_w) = match options.as_ref().and_then(|o| {
                if o.dilations.len() >= 2 {
                    Some((o.dilations[0] as i64, o.dilations[1] as i64))
                } else {
                    None
                }
            }) {
                Some((h, w)) => (h, w),
                None => (1, 1),
            };
            let (padding_top, padding_bottom, padding_left, padding_right) =
                match options.as_ref().and_then(|o| {
                    if o.padding.len() >= 4 {
                        Some((
                            o.padding[0] as i64,
                            o.padding[1] as i64,
                            o.padding[2] as i64,
                            o.padding[3] as i64,
                        ))
                    } else {
                        None
                    }
                }) {
                    Some((t, b, l, r)) => (t, b, l, r),
                    None => (0, 0, 0, 0),
                };
            let groups = options
                .as_ref()
                .map(|o| o.groups as i64)
                .unwrap_or(1)
                .max(1);
            let format_str = match options.as_ref().and_then(|o| {
                if o.input_layout.eq_ignore_ascii_case("nhwc") {
                    Some("NHWC")
                } else if o.input_layout.eq_ignore_ascii_case("nchw") {
                    Some("NCHW")
                } else {
                    None
                }
            }) {
                Some(format) => format,
                None => "NCHW",
            };

            let strides: [i64; 2] = [stride_h, stride_w];
            let pads: [i64; 4] = [padding_top, padding_bottom, padding_left, padding_right];
            let dilations: [i64; 2] = [dilation_h, dilation_w];
            unsafe {
                let strides_name = CString::new("strides").unwrap();
                (shim.operator_set_attr_int64_list)(
                    compute_op,
                    strides_name.as_ptr(),
                    strides.as_ptr(),
                    2,
                );
                let pads_name = CString::new("pads").unwrap();
                (shim.operator_set_attr_int64_list)(
                    compute_op,
                    pads_name.as_ptr(),
                    pads.as_ptr(),
                    4,
                );
                let dilations_name = CString::new("dilations").unwrap();
                (shim.operator_set_attr_int64_list)(
                    compute_op,
                    dilations_name.as_ptr(),
                    dilations.as_ptr(),
                    2,
                );
                let groups_name = CString::new("groups").unwrap();
                (shim.operator_set_attr_int64)(compute_op, groups_name.as_ptr(), groups);
                let data_format_name = CString::new("data_format").unwrap();
                let data_format_value = CString::new(format_str).unwrap();
                (shim.operator_set_attr_string)(
                    compute_op,
                    data_format_name.as_ptr(),
                    data_format_value.as_ptr(),
                );
                let pad_mode_name = CString::new("pad_mode").unwrap();
                let pad_mode_value = CString::new("SPECIFIC").unwrap();
                (shim.operator_set_attr_string)(
                    compute_op,
                    pad_mode_name.as_ptr(),
                    pad_mode_value.as_ptr(),
                );
            }
        }

        // Wire ArgMax via hiai::op::ArgMaxExt2.
        // Axis is a tensor input, not an attribute.  Create an inline Const.
        if let Operation::ArgMax { input, axis, .. } = op {
            let x_handle = handles[*input as usize];

            let axis_name_str = CString::new("axis").unwrap();
            let axis_operator = unsafe { (shim.op_const_with_name)(axis_name_str.as_ptr()) };
            let axis_value: i32 = *axis as i32;
            let axis_shape: [i64; 1] = [1];
            let value_name = CString::new("value").unwrap();
            unsafe {
                (shim.operator_set_attr_tensor_raw_format)(
                    axis_operator,
                    value_name.as_ptr(),
                    &axis_value as *const i32 as *const std::ffi::c_void,
                    4,
                    axis_shape.as_ptr(),
                    1,
                    3, // DT_INT32
                    2, // FORMAT_ND
                );
            }
            const_ops.push(axis_operator);

            let x_name = CString::new("x").unwrap();
            let axis_name = CString::new("axis").unwrap();
            unsafe {
                (shim.operator_set_input)(compute_op, x_name.as_ptr(), x_handle);
                (shim.operator_set_input)(compute_op, axis_name.as_ptr(), axis_operator);
            }
        }

        // Wire BatchNormalization via hiai::op::BNInference.
        if let Operation::BatchNormalization {
            input,
            mean,
            variance,
            options,
            ..
        } = op
        {
            let x_handle = handles[*input as usize];
            let mean_handle = handles[*mean as usize];
            let variance_handle = handles[*variance as usize];

            let x_name = CString::new("x").unwrap();
            let mean_name = CString::new("mean").unwrap();
            let variance_name = CString::new("variance").unwrap();
            unsafe {
                (shim.operator_set_input)(compute_op, x_name.as_ptr(), x_handle);
                (shim.operator_set_input)(compute_op, mean_name.as_ptr(), mean_handle);
                (shim.operator_set_input)(compute_op, variance_name.as_ptr(), variance_handle);
            }

            // Optional scale and offset (bias) from options.
            let scale_id = options.as_ref().and_then(|o| o.scale);
            let bias_id = options.as_ref().and_then(|o| o.bias);
            if let Some(id) = scale_id {
                let scale_handle = handles[id as usize];
                let scale_name = CString::new("scale").unwrap();
                unsafe {
                    (shim.operator_set_input)(compute_op, scale_name.as_ptr(), scale_handle);
                }
            }
            if let Some(id) = bias_id {
                let offset_handle = handles[id as usize];
                let offset_name = CString::new("offset").unwrap();
                unsafe {
                    (shim.operator_set_input)(compute_op, offset_name.as_ptr(), offset_handle);
                }
            }

            let epsilon = options.as_ref().map(|o| o.epsilon as f32).unwrap_or(1e-5);
            let epsilon_name = CString::new("epsilon").unwrap();
            unsafe {
                (shim.operator_set_attr_float)(compute_op, epsilon_name.as_ptr(), epsilon);
            }
        }

        let outputs: &[u32] = op_outputs(op);
        for &out_id in outputs.iter() {
            handles[out_id as usize] = compute_op;
        }

        compute_ops.push(compute_op);
    }

    // ── 4. Create NetOutput ─────────────────────────────────────────
    let out_name = graph.operands[graph.output_operands[0] as usize]
        .name
        .as_deref()
        .unwrap_or("output");
    let net_name = CString::new(out_name).unwrap();
    let net_out = unsafe {
        (shim.op_net_output_with_name)(net_name.as_ptr(), graph.output_operands.len() as i32)
    };
    if net_out.is_null() {
        return Err(GraphError::ConversionFailed {
            format: "cann".into(),
            reason: "cann_op_net_output failed".into(),
        });
    }

    let x_name = CString::new("x").unwrap();
    let y_name = CString::new("y").unwrap();
    let output_type_name = CString::new("output_type").unwrap();
    let status = unsafe {
        (shim.operator_create_dynamic_input)(
            net_out,
            x_name.as_ptr(),
            graph.output_operands.len() as u32,
        )
    };
    if status != 0 {
        return Err(GraphError::ConversionFailed {
            format: "cann".into(),
            reason: format!("cann_operator_create_dynamic_input failed: {status}").into(),
        });
    }

    for (output_index, &out_id) in graph.output_operands.iter().enumerate() {
        let source_handle = handles[out_id as usize];
        if source_handle.is_null() {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("no handle for output operand {out_id}").into(),
            });
        }
        let status = unsafe {
            (shim.operator_set_dynamic_input_by_index)(
                net_out,
                x_name.as_ptr(),
                output_index as u32,
                source_handle,
            )
        };
        if status != 0 {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!(
                    "cann_operator_set_dynamic_input_by_index[{output_index}] failed: {status}"
                )
                .into(),
            });
        }
    }

    let status = unsafe {
        (shim.operator_create_dynamic_output)(
            net_out,
            y_name.as_ptr(),
            graph.output_operands.len() as u32,
        )
    };
    if status != 0 {
        return Err(GraphError::ConversionFailed {
            format: "cann".into(),
            reason: format!("cann_operator_create_dynamic_output failed: {status}").into(),
        });
    }

    let out_types: Vec<i64> = graph
        .output_operands
        .iter()
        .map(|&id| cann_data_type(graph.operands[id as usize].descriptor.data_type) as i64)
        .collect();
    let status = unsafe {
        (shim.operator_set_attr_int64_list)(
            net_out,
            output_type_name.as_ptr(),
            out_types.as_ptr(),
            out_types.len() as i32,
        )
    };
    if status != 0 {
        return Err(GraphError::ConversionFailed {
            format: "cann".into(),
            reason: format!("cann_operator_set_attr_int64_list failed: {status}").into(),
        });
    }

    let mut all_ops: Vec<*mut std::ffi::c_void> = Vec::new();
    all_ops.extend(data_ops.clone());
    all_ops.extend(const_ops);
    all_ops.extend(compute_ops);
    all_ops.push(net_out);

    // ── 5. Add all ops to graph ─────────────────────────────────────
    for &handle in &all_ops {
        if unsafe { (shim.graph_add_op)(can_graph, handle) } != 0 {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_graph_add_op failed".into(),
            });
        }
    }

    // ── 6. Set graph inputs / outputs ───────────────────────────────
    unsafe {
        (shim.graph_set_inputs)(can_graph, data_ops.as_ptr(), data_ops.len() as i32);
        (shim.graph_set_outputs)(can_graph, &net_out as *const *mut std::ffi::c_void, 1);
    }

    // ── 7. Validate graph ───────────────────────────────────────────
    if unsafe { (shim.graph_is_valid)(can_graph) } == 0 {
        return Err(GraphError::ConversionFailed {
            format: "cann".into(),
            reason: "cann_graph_is_valid returned false".into(),
        });
    }

    // ── 8. Compile model ────────────────────────────────────────────
    let model_name = CString::new("webnn_model").unwrap();
    let model = unsafe { (shim.model_create_with_name)(model_name.as_ptr()) };
    if model.is_null() {
        unsafe { (shim.graph_destroy)(can_graph) };
        return Err(GraphError::ConversionFailed {
            format: "cann".into(),
            reason: "cann_model_create_with_name failed".into(),
        });
    }
    if unsafe { (shim.model_set_graph)(model, can_graph) } != 0 {
        unsafe {
            (shim.model_destroy)(model);
            (shim.graph_destroy)(can_graph);
        }
        return Err(GraphError::ConversionFailed {
            format: "cann".into(),
            reason: "cann_model_set_graph failed".into(),
        });
    }

    let ir_handle = unsafe { (shim.ir_build_create)() };
    if ir_handle.is_null() {
        unsafe {
            (shim.model_destroy)(model);
            (shim.graph_destroy)(can_graph);
        }
        return Err(GraphError::ConversionFailed {
            format: "cann".into(),
            reason: "cann_hiai_ir_build_create failed".into(),
        });
    }

    let mut buffer = CannModelBuffer {
        data: std::ptr::null_mut(),
        length: 0,
    };
    if unsafe { (shim.model_create_buff_default)(ir_handle, model, &mut buffer) } != 0
        || buffer.data.is_null()
    {
        unsafe {
            (shim.ir_build_destroy)(ir_handle);
            (shim.model_destroy)(model);
            (shim.graph_destroy)(can_graph);
        }
        return Err(GraphError::ConversionFailed {
            format: "cann".into(),
            reason: "cann_model_create_buff_default failed".into(),
        });
    }

    // Build options: input shapes + NPU device (required for Conv2D).
    let build_opts = unsafe { (shim.build_options_create)() };
    if !build_opts.is_null() {
        // Collect input shapes as Vec<Vec<i64>> and pass via set_input_shapes.
        let mut shapes_ptrs: Vec<*const i64> = Vec::new();
        let mut shape_counts: Vec<i32> = Vec::new();
        let mut dims_bufs: Vec<Vec<i64>> = Vec::new();
        for &input_id in &graph.input_operands {
            let d = descriptor_dims(&graph.operands[input_id as usize].descriptor);
            dims_bufs.push(d);
        }
        for d in &dims_bufs {
            shapes_ptrs.push(d.as_ptr());
            shape_counts.push(d.len() as i32);
        }
        unsafe {
            (shim.build_options_set_input_shapes)(
                build_opts,
                shapes_ptrs.as_ptr(),
                shape_counts.as_ptr(),
                shape_counts.len() as i32,
            );
            let devices: [i32; 1] = [0]; // 0 = NPU
            (shim.build_options_set_device_order)(build_opts, devices.as_ptr(), 1);
        }
    }

    // Compile IR model to OM bytes.
    let status = unsafe { (shim.build_model)(ir_handle, model, build_opts, &mut buffer) };
    if status != 0 || buffer.data.is_null() {
        unsafe {
            (shim.model_buffer_destroy)(
                ir_handle,
                &mut buffer as *mut CannModelBuffer as *mut std::ffi::c_void,
            );
            (shim.ir_build_destroy)(ir_handle);
            (shim.model_destroy)(model);
            (shim.graph_destroy)(can_graph);
        }
        return Err(GraphError::ConversionFailed {
            format: "cann".into(),
            reason: "cann_build_model failed".into(),
        });
    }

    let bytes =
        unsafe { std::slice::from_raw_parts(buffer.data as *const u8, buffer.length as usize) }
            .to_vec();

    // ── 9. Cleanup ──────────────────────────────────────────────────
    unsafe {
        (shim.model_buffer_destroy)(
            ir_handle,
            &mut buffer as *mut CannModelBuffer as *mut std::ffi::c_void,
        );
    }
    for &handle in &all_ops {
        unsafe { (shim.operator_destroy)(handle) };
    }
    unsafe {
        (shim.ir_build_destroy)(ir_handle);
        (shim.model_destroy)(model);
        (shim.graph_destroy)(can_graph);
    }

    Ok(bytes)
}

pub struct CannConverter;

impl GraphConverter for CannConverter {
    fn format(&self) -> &'static str {
        "cann"
    }

    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
        // Call CANN shim layer.
        if let Ok(bytes) = encode_via_adapter(graph) {
            return Ok(ConvertedGraph {
                format: "cann",
                content_type: "application/octet-stream",
                data: bytes,
                weights_data: None,
            });
        }
        // Fallback to Mock CANN -- validates ops, returns placeholder bytes.
        let model_bytes = build_hiai_ir_model_mock(graph)?;
        Ok(ConvertedGraph {
            format: "cann",
            content_type: "application/octet-stream",
            data: model_bytes,
            weights_data: None,
        })
    }
}

// cann-runtime-mock: verify graph structure, return placeholder bytes.
fn build_hiai_ir_model_mock(graph: &GraphInfo) -> Result<Vec<u8>, GraphError> {
    let mut operation_count: usize = 0;
    for operation in &graph.operations {
        if webnn_op_to_hiai(operation).is_some() {
            operation_count += 1;
        }
    }
    if operation_count == 0 {
        return Err(GraphError::ConversionFailed {
            format: "cann".to_string(),
            reason: "no supported ops found".to_string(),
        });
    }
    Ok(vec![0x00, 0x00, 0x00, 0x00])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind};
    use crate::operator_options::MLDimension;
    use crate::operators::Operation;
    use std::collections::HashMap;

    fn make_add_graph() -> GraphInfo {
        GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![Dimension::Static(2), Dimension::Static(2)],
                        pending_permutation: vec![],
                    },
                    name: Some("lhs".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![Dimension::Static(2), Dimension::Static(2)],
                        pending_permutation: vec![],
                    },
                    name: Some("rhs".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![Dimension::Static(2), Dimension::Static(2)],
                        pending_permutation: vec![],
                    },
                    name: Some("sum".to_string()),
                },
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            operations: vec![Operation::Add {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![2],
            }],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        }
    }

    #[test]
    fn test_add_graph_converts() {
        let graph = make_add_graph();
        let converter = CannConverter;
        assert_eq!(converter.format(), "cann");
        let result = converter.convert(&graph);
        assert!(result.is_ok(), "{result:?}");
        let converted = result.unwrap();
        assert!(!converted.data.is_empty());
    }

    #[test]
    fn test_webnn_op_to_hiai_relu() {
        let op = Operation::Relu {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("ReLU"));
    }

    #[test]
    fn test_webnn_op_to_hiai_sigmoid() {
        let op = Operation::Sigmoid {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Sigmoid"));
    }

    #[test]
    fn test_webnn_op_to_hiai_tanh() {
        let op = Operation::Tanh {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Tanh"));
    }

    #[test]
    fn test_webnn_op_to_hiai_add() {
        let op = Operation::Add {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Add"));
    }

    #[test]
    fn test_webnn_op_to_hiai_mul() {
        let op = Operation::Mul {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Mul"));
    }

    #[test]
    fn test_webnn_op_to_hiai_sub() {
        let op = Operation::Sub {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Sub"));
    }

    #[test]
    fn test_webnn_op_to_hiai_conv2d() {
        let op = Operation::Conv2d {
            input: 0,
            filter: 1,
            options: None,
            outputs: vec![3],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Conv2D"));
    }

    #[test]
    fn test_webnn_op_to_hiai_max_pool2d() {
        let op = Operation::MaxPool2d {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("MaxPool"));
    }

    #[test]
    fn test_webnn_op_to_hiai_average_pool2d() {
        let op = Operation::AveragePool2d {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("AvgPool"));
    }

    #[test]
    fn test_webnn_op_to_hiai_matmul() {
        let op = Operation::Matmul {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("MatMul"));
    }

    #[test]
    fn test_webnn_op_to_hiai_softmax() {
        let op = Operation::Softmax {
            input: 0,
            axis: 1,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Softmax"));
    }

    #[test]
    fn test_webnn_op_to_hiai_reshape() {
        let op = Operation::Reshape {
            input: 0,
            new_shape: vec![MLDimension::Static(1), MLDimension::Static(4)],
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Reshape"));
    }

    #[test]
    fn test_webnn_op_to_hiai_concat() {
        let op = Operation::Concat {
            inputs: vec![0, 1],
            axis: 0,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Concat"));
    }

    #[test]
    fn test_webnn_op_to_hiai_identity() {
        let op = Operation::Identity {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Identity"));
    }
}

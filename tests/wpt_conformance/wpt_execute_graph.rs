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
//! Build and execute WPT graphs via rustnn [`MLGraphBuilder`] / [`MLContext`].
//!
//! Port of pywebnn `tests/wpt_execute_graph.py`.

use half::f16;
use rustnn::error::GraphBuilderError;
use rustnn::graph::{unpack_int4, unpack_uint4};
use rustnn::mlcontext::{MLContext, MLOperand, MLOperandDescriptor, MLTensor, MLTensorDescriptor};
use rustnn::mlgraphbuilder::MLGraphBuilder;
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::{
    MLOperatorOptions, MLPool2dOptions, MLReduceOptions, OperationExtras, OperatorOptions,
};
use std::collections::{HashMap, HashSet};

use super::wpt_tensor::{
    is_operand_option, normalize_option_key, normalize_wpt_op_name, output_names, tensor_f16_bits,
    tensor_f32_values, tensor_i8_values, tensor_i32_values, tensor_i64_values,
    tensor_spec_to_bytes, tensor_u8_values, tensor_u32_values, tensor_u64_values,
};
use super::wpt_types::{WptGraph, WptOperator, WptTensorSpec};

const LARGE_SCALAR_INLINE_BYTES_THRESHOLD: usize = 8 * 1024 * 1024;

const MULTI_OUTPUT_OPS: &[&str] = &["split", "gru", "lstm", "lstmCell"];

const POOL2D_LIKE_OPS: &[&str] = &[
    "averagePool2d",
    "maxPool2d",
    "l2Pool2d",
    "globalAveragePool",
    "globalMaxPool",
];

/// Actual tensor output from graph execution, for tolerance validation in `mod.rs`.
#[derive(Debug, Clone)]
pub struct WptActualOutput {
    data_type: String,
    f32_data: Option<Vec<f32>>,
    i32_data: Option<Vec<i32>>,
    i64_data: Option<Vec<i64>>,
}

impl WptActualOutput {
    pub fn data_type(&self) -> &str {
        &self.data_type
    }

    pub fn f32_data(&self) -> Option<&[f32]> {
        self.f32_data.as_deref()
    }

    pub fn i32_data(&self) -> Option<&[i32]> {
        self.i32_data.as_deref()
    }

    pub fn i64_data(&self) -> Option<&[i64]> {
        self.i64_data.as_deref()
    }
}

pub(crate) enum PositionalArg {
    Operand(MLOperand),
    Operands(Vec<MLOperand>),
}

pub(crate) struct MethodCallArgs {
    pub positional: Vec<PositionalArg>,
    pub options: serde_json::Map<String, serde_json::Value>,
}

enum InvokeResult {
    Single(MLOperand),
    Multi(Vec<MLOperand>),
}

fn shape_element_count(shape: &[u32]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().map(|&d| d as usize).product()
    }
}

fn packed_storage_byte_length(elements: usize) -> usize {
    let n = elements.max(1);
    ((4 * n) + 7) / 8
}

fn storage_byte_count(data_type: &str, element_count: usize) -> usize {
    let n = element_count.max(1);
    match data_type {
        "float32" | "int32" | "uint32" => n * 4,
        "float16" => n * 2,
        "int64" | "uint64" => n * 8,
        "int4" | "uint4" => packed_storage_byte_length(n),
        _ => n,
    }
}

fn constant_raw_length(data: &serde_json::Value) -> usize {
    if let Some(arr) = data.as_array() {
        arr.len()
    } else if data.is_null() {
        0
    } else {
        1
    }
}

/// Whether a marked-constant WPT input should be inlined via `constant_from_slice`.
///
/// Mirrors pywebnn `wpt_pack_constants.should_inline_constant`: scalar-fill tensors at or above
/// 8 MiB are supplied as runtime inputs instead of builder constants.
pub fn should_inline_constant(spec: &WptTensorSpec) -> bool {
    if !spec.constant {
        return false;
    }
    let data_type = spec.data_type();
    let shape = spec.shape();
    let raw_len = constant_raw_length(&spec.data);
    let scalar_fill_like = raw_len <= 1;
    let est_bytes = storage_byte_count(data_type, shape_element_count(shape));
    !(scalar_fill_like && est_bytes >= LARGE_SCALAR_INLINE_BYTES_THRESHOLD)
}

fn is_pool2d_op(op_name: &str) -> bool {
    POOL2D_LIKE_OPS.contains(&op_name)
}

fn wpt_ml_data_type(s: &str) -> MLOperandDataType {
    match s.trim().to_lowercase().as_str() {
        "float32" => MLOperandDataType::Float32,
        "float16" => MLOperandDataType::Float16,
        "int32" => MLOperandDataType::Int32,
        "uint32" => MLOperandDataType::Uint32,
        "int64" => MLOperandDataType::Int64,
        "uint64" => MLOperandDataType::Uint64,
        "int8" => MLOperandDataType::Int8,
        "uint8" => MLOperandDataType::Uint8,
        "int4" => MLOperandDataType::Int4,
        "uint4" => MLOperandDataType::Uint4,
        _ => MLOperandDataType::Float32,
    }
}

fn wpt_operand_descriptor(spec: &WptTensorSpec) -> MLOperandDescriptor {
    let shape: Vec<u64> = spec.shape().iter().map(|&d| d as u64).collect();
    MLOperandDescriptor::new(wpt_ml_data_type(spec.data_type()), shape)
}

fn rw_tensor_descriptor(spec: &WptTensorSpec) -> MLTensorDescriptor {
    let mut desc = MLTensorDescriptor::from_operand_descriptor(&wpt_operand_descriptor(spec));
    desc.set_readable(true);
    desc.set_writable(true);
    desc
}

fn is_operand_ref(value: &serde_json::Value, operand_names: &HashSet<String>) -> bool {
    value.as_str().is_some_and(|s| operand_names.contains(s))
}

fn normalize_option_value(op_name: &str, key: &str, value: serde_json::Value) -> serde_json::Value {
    if op_name == "pad" && key == "mode" {
        if let Some(arr) = value.as_array() {
            if arr.len() == 1 {
                return arr[0].clone();
            }
            if !arr.is_empty() {
                return arr[0].clone();
            }
        }
    }
    if (op_name == "pad" && key == "value")
        || (op_name == "clamp" && matches!(key, "minValue" | "maxValue"))
    {
        if let Some(s) = value.as_str() {
            return match s {
                "NaN" => serde_json::json!("NaN"),
                "Infinity" | "+Infinity" => serde_json::json!("Infinity"),
                "-Infinity" => serde_json::json!("-Infinity"),
                _ => value,
            };
        }
    }
    value
}

fn option_json_key(op_name: &str, key: &str) -> String {
    let normalized = normalize_option_key(op_name, op_name, key);
    if is_pool2d_op(op_name) && (key == "padding" || normalized == "padding") {
        return "padding".to_string();
    }
    if (op_name == "conv2d" || op_name == "convTranspose2d")
        && (key == "padding" || normalized == "padding")
    {
        return "padding".to_string();
    }
    normalized
}

fn resolve_options_object(
    raw_options: &serde_json::Map<String, serde_json::Value>,
    name_to_id: &HashMap<String, u32>,
    operand_names: &HashSet<String>,
    op_name: &str,
) -> serde_json::Map<String, serde_json::Value> {
    let mut out = serde_json::Map::new();
    for (key, value) in raw_options {
        let opt_key = normalize_option_key(op_name, op_name, key);
        if is_operand_ref(value, operand_names) {
            let name = value.as_str().unwrap();
            let id = name_to_id.get(name).copied().unwrap_or(0);
            out.insert(
                option_json_key(op_name, &opt_key),
                serde_json::Value::Number(id.into()),
            );
            continue;
        }
        if let Some(arr) = value.as_array() {
            if arr.iter().all(|x| is_operand_ref(x, operand_names)) {
                let ids: Vec<serde_json::Value> = arr
                    .iter()
                    .map(|x| {
                        let name = x.as_str().unwrap();
                        serde_json::Value::Number(name_to_id.get(name).copied().unwrap_or(0).into())
                    })
                    .collect();
                out.insert(
                    option_json_key(op_name, &opt_key),
                    serde_json::Value::Array(ids),
                );
                continue;
            }
        }
        out.insert(
            option_json_key(op_name, &opt_key),
            normalize_option_value(op_name, &opt_key, value.clone()),
        );
    }
    out
}

/// Build positional operands and an options JSON map for one WPT operator.
///
/// Operand-option fields (e.g. `gemm.c`, `conv2d.bias`) are recorded as u32 operand ids in
/// `options`, matching [`OperatorOptions`] deserialization.
pub fn build_method_args(
    op_name: &str,
    op: &WptOperator,
    name_to_id: &HashMap<String, u32>,
    operand_map: &HashMap<String, MLOperand>,
    operand_names: &HashSet<String>,
) -> Result<MethodCallArgs, String> {
    let mut positional = Vec::new();
    let mut options = serde_json::Map::new();
    let mut pad_beginning: Option<Vec<u32>> = None;
    let mut pad_ending: Option<Vec<u32>> = None;

    let entries: Vec<(String, serde_json::Value)> = if let Some(arr) = op.arguments.as_array() {
        arr.iter()
            .filter_map(|item| item.as_object())
            .flat_map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())))
            .collect()
    } else if let Some(obj) = op.arguments.as_object() {
        obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    } else {
        Vec::new()
    };

    for (key, value) in entries {
        if key == "options" {
            if let Some(obj) = value.as_object() {
                options.extend(resolve_options_object(
                    obj,
                    name_to_id,
                    operand_names,
                    op_name,
                ));
            }
            continue;
        }

        if op_name == "pad" && key == "beginningPadding" {
            pad_beginning =
                serde_json::from_value(normalize_option_value(op_name, &key, value)).ok();
            continue;
        }
        if op_name == "pad" && key == "endingPadding" {
            pad_ending = serde_json::from_value(normalize_option_value(op_name, &key, value)).ok();
            continue;
        }

        if is_operand_ref(&value, operand_names) {
            let name = value.as_str().unwrap();
            if is_operand_option(op_name, &key) {
                let opt_key =
                    option_json_key(op_name, &normalize_option_key(op_name, op_name, &key));
                let id = *name_to_id
                    .get(name)
                    .ok_or_else(|| format!("unknown operand '{name}'"))?;
                options.insert(opt_key, serde_json::Value::Number(id.into()));
            } else {
                let operand = *operand_map
                    .get(name)
                    .ok_or_else(|| format!("unknown operand '{name}'"))?;
                positional.push(PositionalArg::Operand(operand));
            }
            continue;
        }

        if let Some(arr) = value.as_array() {
            if arr.iter().all(|x| is_operand_ref(x, operand_names)) {
                let ops: Vec<MLOperand> = arr
                    .iter()
                    .map(|x| {
                        let name = x.as_str().unwrap();
                        operand_map
                            .get(name)
                            .copied()
                            .ok_or_else(|| format!("unknown operand '{name}'"))
                    })
                    .collect::<Result<_, _>>()?;
                positional.push(PositionalArg::Operands(ops));
                continue;
            }
        }

        if op_name == "cast" && key == "type" {
            // Cast target type is handled via OperationExtras during invoke.
            options.insert("to".to_string(), value);
            continue;
        }

        if op_name == "pad" && (key == "mode" || key == "value") {
            options.insert(
                option_json_key(op_name, &key),
                normalize_option_value(op_name, &key, value),
            );
            continue;
        }

        if is_operand_option(op_name, &key) {
            return Err(format!(
                "operand option '{key}' must reference a graph operand for op {op_name}"
            ));
        }

        // Non-operand scalar method arguments (axis, steps, etc.) go into options for
        // OperationExtras extraction during invoke.
        options.insert(option_json_key(op_name, &key), value);
    }

    if op_name == "pad" {
        let beginning = pad_beginning.ok_or_else(|| "pad missing beginningPadding".to_string())?;
        let ending = pad_ending.ok_or_else(|| "pad missing endingPadding".to_string())?;
        options.insert(
            "beginningPadding".to_string(),
            serde_json::Value::Array(
                beginning
                    .iter()
                    .map(|&v| serde_json::Value::Number(v.into()))
                    .collect(),
            ),
        );
        options.insert(
            "endingPadding".to_string(),
            serde_json::Value::Array(
                ending
                    .iter()
                    .map(|&v| serde_json::Value::Number(v.into()))
                    .collect(),
            ),
        );
    }

    Ok(MethodCallArgs {
        positional,
        options,
    })
}

fn op_err(op_name: &str, msg: impl std::fmt::Display) -> String {
    format!("{op_name}: {msg}")
}

fn expect_operand(args: &MethodCallArgs, idx: usize, op_name: &str) -> Result<MLOperand, String> {
    match args.positional.get(idx) {
        Some(PositionalArg::Operand(o)) => Ok(*o),
        _ => Err(op_err(
            op_name,
            format!("missing positional operand at index {idx}"),
        )),
    }
}

fn expect_operands(
    args: &MethodCallArgs,
    idx: usize,
    op_name: &str,
) -> Result<Vec<MLOperand>, String> {
    match args.positional.get(idx) {
        Some(PositionalArg::Operands(v)) => Ok(v.clone()),
        Some(PositionalArg::Operand(o)) => Ok(vec![*o]),
        _ => Err(op_err(
            op_name,
            format!("missing positional operands at index {idx}"),
        )),
    }
}

fn base_operator_options(op: &OperatorOptions) -> MLOperatorOptions {
    op.as_operator()
        .cloned()
        .unwrap_or_else(MLOperatorOptions::default)
}

fn parse_options_and_extras(
    op_name: &str,
    options: &serde_json::Map<String, serde_json::Value>,
) -> (OperatorOptions, OperationExtras) {
    let opts_value = serde_json::Value::Object(options.clone());
    OperatorOptions::from_json_with_op_type_and_extras(op_name, &opts_value)
}

fn cast_data_type(
    extras: &OperationExtras,
    options: &serde_json::Map<String, serde_json::Value>,
) -> Result<MLOperandDataType, String> {
    if let Some(dt) = extras.to_data_type {
        return Ok(dt);
    }
    if let Some(v) = options.get("to").or_else(|| options.get("type")) {
        if let Ok(dt) = serde_json::from_value::<MLOperandDataType>(v.clone()) {
            return Ok(dt);
        }
        if let Some(s) = v.as_str() {
            return Ok(wpt_ml_data_type(s));
        }
    }
    Err("cast requires type".to_string())
}

fn invoke_binary_with_options(
    builder: &mut MLGraphBuilder,
    op_name: &str,
    args: &MethodCallArgs,
    operator_options: &OperatorOptions,
    build: impl FnOnce(
        &mut MLGraphBuilder,
        MLOperand,
        MLOperand,
        MLOperatorOptions,
    ) -> std::result::Result<MLOperand, GraphBuilderError>,
) -> Result<InvokeResult, String> {
    let a = expect_operand(args, 0, op_name)?;
    let b = expect_operand(args, 1, op_name)?;
    Ok(InvokeResult::Single(
        build(builder, a, b, base_operator_options(operator_options)).map_err(|e| e.to_string())?,
    ))
}

fn invoke_unary_simple(
    builder: &mut MLGraphBuilder,
    op_name: &str,
    args: &MethodCallArgs,
    f: impl FnOnce(&mut MLGraphBuilder, MLOperand) -> std::result::Result<MLOperand, GraphBuilderError>,
) -> Result<InvokeResult, String> {
    unary_simple(builder, expect_operand(args, 0, op_name)?, f)
}

fn invoke_unary_reduce(
    builder: &mut MLGraphBuilder,
    op_name: &str,
    args: &MethodCallArgs,
    operator_options: OperatorOptions,
    f: impl FnOnce(
        &mut MLGraphBuilder,
        MLOperand,
        MLReduceOptions,
    ) -> std::result::Result<MLOperand, GraphBuilderError>,
) -> Result<InvokeResult, String> {
    unary_reduce(
        builder,
        expect_operand(args, 0, op_name)?,
        operator_options,
        f,
    )
}

fn invoke_pool2d(
    builder: &mut MLGraphBuilder,
    op_name: &str,
    args: &MethodCallArgs,
    operator_options: OperatorOptions,
    f: impl FnOnce(
        &mut MLGraphBuilder,
        MLOperand,
        MLPool2dOptions,
    ) -> std::result::Result<MLOperand, GraphBuilderError>,
) -> Result<InvokeResult, String> {
    pool2d(
        builder,
        expect_operand(args, 0, op_name)?,
        operator_options,
        f,
    )
}

fn invoke_builder_method(
    builder: &mut MLGraphBuilder,
    op_name: &str,
    args: &MethodCallArgs,
) -> Result<InvokeResult, String> {
    let (operator_options, extras) = parse_options_and_extras(op_name, &args.options);

    match op_name {
        "add" => invoke_binary_with_options(
            builder,
            op_name,
            args,
            &operator_options,
            |b, a, o, opts| b.add_with_options(a, o, opts),
        ),
        "sub" => invoke_binary_with_options(
            builder,
            op_name,
            args,
            &operator_options,
            |b, a, o, opts| b.sub_with_options(a, o, opts),
        ),
        "mul" => invoke_binary_with_options(
            builder,
            op_name,
            args,
            &operator_options,
            |b, a, o, opts| b.mul_with_options(a, o, opts),
        ),
        "div" => invoke_binary_with_options(
            builder,
            op_name,
            args,
            &operator_options,
            |b, a, o, opts| b.div_with_options(a, o, opts),
        ),
        "pow" => invoke_binary_with_options(
            builder,
            op_name,
            args,
            &operator_options,
            |b, a, o, opts| b.pow_with_options(a, o, opts),
        ),
        "max" => invoke_binary_with_options(
            builder,
            op_name,
            args,
            &operator_options,
            |b, a, o, opts| b.max_with_options(a, o, opts),
        ),
        "min" => invoke_binary_with_options(
            builder,
            op_name,
            args,
            &operator_options,
            |b, a, o, opts| b.min_with_options(a, o, opts),
        ),
        "equal" => invoke_binary_with_options(
            builder,
            op_name,
            args,
            &operator_options,
            |b, a, o, opts| b.equal_with_options(a, o, opts),
        ),
        "greater" => invoke_binary_with_options(
            builder,
            op_name,
            args,
            &operator_options,
            |b, a, o, opts| b.greater_with_options(a, o, opts),
        ),
        "greaterOrEqual" => invoke_binary_with_options(
            builder,
            op_name,
            args,
            &operator_options,
            |b, a, o, opts| b.greater_or_equal_with_options(a, o, opts),
        ),
        "lesser" => invoke_binary_with_options(
            builder,
            op_name,
            args,
            &operator_options,
            |b, a, o, opts| b.lesser_with_options(a, o, opts),
        ),
        "lesserOrEqual" => invoke_binary_with_options(
            builder,
            op_name,
            args,
            &operator_options,
            |b, a, o, opts| b.lesser_or_equal_with_options(a, o, opts),
        ),
        "notEqual" => invoke_binary_with_options(
            builder,
            op_name,
            args,
            &operator_options,
            |b, a, o, opts| b.not_equal_with_options(a, o, opts),
        ),
        "logicalAnd" => invoke_binary_with_options(
            builder,
            op_name,
            args,
            &operator_options,
            |b, a, o, opts| b.logical_and_with_options(a, o, opts),
        ),
        "logicalOr" => invoke_binary_with_options(
            builder,
            op_name,
            args,
            &operator_options,
            |b, a, o, opts| b.logical_or_with_options(a, o, opts),
        ),
        "logicalXor" => invoke_binary_with_options(
            builder,
            op_name,
            args,
            &operator_options,
            |b, a, o, opts| b.logical_xor_with_options(a, o, opts),
        ),
        "matmul" => invoke_binary_with_options(
            builder,
            op_name,
            args,
            &operator_options,
            |b, a, o, opts| b.matmul_with_options(a, o, opts),
        ),
        "gemm" => {
            let a = expect_operand(args, 0, op_name)?;
            let b = expect_operand(args, 1, op_name)?;
            let opts = operator_options.as_gemm().cloned().unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .gemm_with_options(a, b, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "conv2d" => {
            let input = expect_operand(args, 0, op_name)?;
            let filter = expect_operand(args, 1, op_name)?;
            let opts = operator_options.as_conv2d().cloned().unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .conv2_with_options(input, filter, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "convTranspose2d" => {
            let input = expect_operand(args, 0, op_name)?;
            let filter = expect_operand(args, 1, op_name)?;
            let opts = operator_options
                .as_conv_transpose2d()
                .cloned()
                .unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .conv_transpose2d_with_options(input, filter, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "prelu" => {
            let input = expect_operand(args, 0, op_name)?;
            let slope = expect_operand(args, 1, op_name)?;
            Ok(InvokeResult::Single(
                builder
                    .prelu_with_options(input, slope, base_operator_options(&operator_options))
                    .map_err(|e| e.to_string())?,
            ))
        }
        "gatherND" => {
            let input = expect_operand(args, 0, op_name)?;
            let indices = expect_operand(args, 1, op_name)?;
            Ok(InvokeResult::Single(
                builder
                    .gather_nd_with_options(
                        input,
                        indices,
                        base_operator_options(&operator_options),
                    )
                    .map_err(|e| e.to_string())?,
            ))
        }
        "where" => {
            let condition = expect_operand(args, 0, op_name)?;
            let true_value = expect_operand(args, 1, op_name)?;
            let false_value = expect_operand(args, 2, op_name)?;
            Ok(InvokeResult::Single(
                builder
                    .where_with_options(
                        condition,
                        true_value,
                        false_value,
                        base_operator_options(&operator_options),
                    )
                    .map_err(|e| e.to_string())?,
            ))
        }
        "batchNormalization" => {
            let input = expect_operand(args, 0, op_name)?;
            let mean = expect_operand(args, 1, op_name)?;
            let variance = expect_operand(args, 2, op_name)?;
            let opts = operator_options
                .as_batch_normalization()
                .cloned()
                .unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .batch_normalization_with_options(input, mean, variance, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "scatterElements" => {
            let input = expect_operand(args, 0, op_name)?;
            let indices = expect_operand(args, 1, op_name)?;
            let updates = expect_operand(args, 2, op_name)?;
            let opts = operator_options
                .as_scatter_elements()
                .cloned()
                .unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .scatter_elements_with_options(input, indices, updates, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "scatterND" => {
            let input = expect_operand(args, 0, op_name)?;
            let indices = expect_operand(args, 1, op_name)?;
            let updates = expect_operand(args, 2, op_name)?;
            Ok(InvokeResult::Single(
                builder
                    .scatter_nd_with_options(
                        input,
                        indices,
                        updates,
                        base_operator_options(&operator_options),
                    )
                    .map_err(|e| e.to_string())?,
            ))
        }
        "quantizeLinear" => {
            let input = expect_operand(args, 0, op_name)?;
            let scale = expect_operand(args, 1, op_name)?;
            let zero_point = if args.positional.len() > 2 {
                Some(expect_operand(args, 2, op_name)?)
            } else {
                None
            };
            Ok(InvokeResult::Single(
                builder
                    .quantize_linear_with_options(
                        input,
                        scale,
                        zero_point,
                        base_operator_options(&operator_options),
                    )
                    .map_err(|e| e.to_string())?,
            ))
        }
        "dequantizeLinear" => {
            let input = expect_operand(args, 0, op_name)?;
            let scale = expect_operand(args, 1, op_name)?;
            let zero_point = if args.positional.len() > 2 {
                Some(expect_operand(args, 2, op_name)?)
            } else {
                None
            };
            Ok(InvokeResult::Single(
                builder
                    .dequantize_linear_with_options(
                        input,
                        scale,
                        zero_point,
                        base_operator_options(&operator_options),
                    )
                    .map_err(|e| e.to_string())?,
            ))
        }
        "concat" => {
            let inputs = expect_operands(args, 0, op_name)?;
            let axis = extras.axis.ok_or_else(|| op_err(op_name, "missing axis"))?;
            Ok(InvokeResult::Single(
                builder
                    .concat_with_options(&inputs, axis, base_operator_options(&operator_options))
                    .map_err(|e| e.to_string())?,
            ))
        }
        "slice" => {
            let input = expect_operand(args, 0, op_name)?;
            let opts = operator_options.as_slice().cloned().unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .slice_with_options(input, &extras.starts, &extras.sizes, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "split" => {
            let input = expect_operand(args, 0, op_name)?;
            let opts = operator_options.as_split().cloned().unwrap_or_default();
            let outs = if let Some(n) = extras.split_equal_parts {
                builder
                    .split_equal_with_options(input, n, opts)
                    .map_err(|e| e.to_string())?
            } else if !extras.splits.is_empty() {
                builder
                    .split_with_options(input, &extras.splits, opts)
                    .map_err(|e| e.to_string())?
            } else {
                return Err(op_err(op_name, "missing splits"));
            };
            Ok(InvokeResult::Multi(outs))
        }
        "gru" => {
            let input = expect_operand(args, 0, op_name)?;
            let weight = expect_operand(args, 1, op_name)?;
            let recurrent_weight = expect_operand(args, 2, op_name)?;
            let steps = extras
                .steps
                .ok_or_else(|| op_err(op_name, "missing steps"))?;
            let hidden_size = extras
                .hidden_size
                .ok_or_else(|| op_err(op_name, "missing hiddenSize"))?;
            let opts = operator_options.as_gru().cloned().unwrap_or_default();
            Ok(InvokeResult::Multi(
                builder
                    .gru_with_options(input, weight, recurrent_weight, steps, hidden_size, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "gruCell" => {
            let input = expect_operand(args, 0, op_name)?;
            let weight = expect_operand(args, 1, op_name)?;
            let recurrent_weight = expect_operand(args, 2, op_name)?;
            let hidden_state = expect_operand(args, 3, op_name)?;
            let hidden_size = extras
                .hidden_size
                .ok_or_else(|| op_err(op_name, "missing hiddenSize"))?;
            let opts = operator_options.as_gru_cell().cloned().unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .gru_cell_with_options(
                        input,
                        weight,
                        recurrent_weight,
                        hidden_state,
                        hidden_size,
                        opts,
                    )
                    .map_err(|e| e.to_string())?,
            ))
        }
        "lstm" => {
            let input = expect_operand(args, 0, op_name)?;
            let weight = expect_operand(args, 1, op_name)?;
            let recurrent_weight = expect_operand(args, 2, op_name)?;
            let steps = extras
                .steps
                .ok_or_else(|| op_err(op_name, "missing steps"))?;
            let hidden_size = extras
                .hidden_size
                .ok_or_else(|| op_err(op_name, "missing hiddenSize"))?;
            let opts = operator_options.as_lstm().cloned().unwrap_or_default();
            Ok(InvokeResult::Multi(
                builder
                    .lstm_with_options(input, weight, recurrent_weight, steps, hidden_size, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "lstmCell" => {
            let input = expect_operand(args, 0, op_name)?;
            let weight = expect_operand(args, 1, op_name)?;
            let recurrent_weight = expect_operand(args, 2, op_name)?;
            let hidden_state = expect_operand(args, 3, op_name)?;
            let cell_state = expect_operand(args, 4, op_name)?;
            let hidden_size = extras
                .hidden_size
                .ok_or_else(|| op_err(op_name, "missing hiddenSize"))?;
            let opts = operator_options.as_lstm_cell().cloned().unwrap_or_default();
            let results = builder
                .lstm_cell_with_options(
                    input,
                    weight,
                    recurrent_weight,
                    hidden_state,
                    cell_state,
                    hidden_size,
                    opts,
                )
                .map_err(|e| e.to_string())?;
            if results.len() != 2 {
                return Err(op_err(
                    op_name,
                    format!("expected 2 lstmCell outputs, got {}", results.len()),
                ));
            }
            Ok(InvokeResult::Multi(results))
        }
        "gather" => {
            let input = expect_operand(args, 0, op_name)?;
            let indices = expect_operand(args, 1, op_name)?;
            let opts = operator_options.as_gather().cloned().unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .gather_with_options(input, indices, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "gatherElements" => {
            let input = expect_operand(args, 0, op_name)?;
            let indices = expect_operand(args, 1, op_name)?;
            let opts = operator_options.as_gather().cloned().unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .gather_elements_with_options(input, indices, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "pad" => {
            let input = expect_operand(args, 0, op_name)?;
            let mut beginning = extras.beginning_padding;
            let mut ending = extras.ending_padding;
            if beginning.is_empty() && ending.is_empty() {
                let rank = builder
                    .rustnn_operand_shape(input)
                    .map_err(|e| e.to_string())?
                    .len();
                beginning = vec![0u32; rank];
                ending = vec![0u32; rank];
            } else if beginning.is_empty() || ending.is_empty() {
                return Err(op_err(
                    op_name,
                    "requires beginningPadding and endingPadding",
                ));
            }
            let opts = operator_options.as_pad().cloned().unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .pad_with_options(input, beginning, ending, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "cast" => {
            let input = expect_operand(args, 0, op_name)?;
            let data_type = cast_data_type(&extras, &args.options)?;
            Ok(InvokeResult::Single(
                builder
                    .cast_with_options(input, data_type, base_operator_options(&operator_options))
                    .map_err(|e| e.to_string())?,
            ))
        }
        "reshape" => {
            let input = expect_operand(args, 0, op_name)?;
            Ok(InvokeResult::Single(
                builder
                    .reshape_with_options(
                        input,
                        extras.reshape_new_shape.clone(),
                        base_operator_options(&operator_options),
                    )
                    .map_err(|e| e.to_string())?,
            ))
        }
        "expand" => {
            let input = expect_operand(args, 0, op_name)?;
            Ok(InvokeResult::Single(
                builder
                    .expand_with_options(
                        input,
                        extras.expand_new_shape.clone(),
                        base_operator_options(&operator_options),
                    )
                    .map_err(|e| e.to_string())?,
            ))
        }
        "tile" => {
            let input = expect_operand(args, 0, op_name)?;
            Ok(InvokeResult::Single(
                builder
                    .tile_with_options(
                        input,
                        extras.repetitions.clone(),
                        base_operator_options(&operator_options),
                    )
                    .map_err(|e| e.to_string())?,
            ))
        }
        "softmax" => {
            let input = expect_operand(args, 0, op_name)?;
            let axis = extras.axis.ok_or_else(|| op_err(op_name, "missing axis"))?;
            Ok(InvokeResult::Single(
                builder
                    .softmax_with_options(input, axis, base_operator_options(&operator_options))
                    .map_err(|e| e.to_string())?,
            ))
        }
        "argMax" => {
            let input = expect_operand(args, 0, op_name)?;
            let axis = extras.axis.ok_or_else(|| op_err(op_name, "missing axis"))?;
            let opts = operator_options
                .as_arg_min_max()
                .cloned()
                .unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .arg_max_with_options(input, axis, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "argMin" => {
            let input = expect_operand(args, 0, op_name)?;
            let axis = extras.axis.ok_or_else(|| op_err(op_name, "missing axis"))?;
            let opts = operator_options
                .as_arg_min_max()
                .cloned()
                .unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .arg_min_with_options(input, axis, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "cumulativeSum" => {
            let input = expect_operand(args, 0, op_name)?;
            let axis = extras.axis.ok_or_else(|| op_err(op_name, "missing axis"))?;
            let opts = operator_options
                .as_cumulative_sum()
                .cloned()
                .unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .cumulative_sum_with_options(input, axis, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "triangular" => {
            let input = expect_operand(args, 0, op_name)?;
            let mut opts = operator_options
                .as_triangular()
                .cloned()
                .unwrap_or_default();
            if opts.upper.is_none() {
                opts.upper = Some(true);
            }
            Ok(InvokeResult::Single(
                builder
                    .triangular_with_options(input, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "reduceSum" => invoke_unary_reduce(builder, op_name, args, operator_options, |b, i, o| {
            b.reduce_sum_with_options(i, o)
        }),
        "reduceMean" => invoke_unary_reduce(builder, op_name, args, operator_options, |b, i, o| {
            b.reduce_mean_with_options(i, o)
        }),
        "reduceMax" => invoke_unary_reduce(builder, op_name, args, operator_options, |b, i, o| {
            b.reduce_max_with_options(i, o)
        }),
        "reduceMin" => invoke_unary_reduce(builder, op_name, args, operator_options, |b, i, o| {
            b.reduce_min_with_options(i, o)
        }),
        "reduceProduct" => {
            invoke_unary_reduce(builder, op_name, args, operator_options, |b, i, o| {
                b.reduce_product_with_options(i, o)
            })
        }
        "reduceL1" => invoke_unary_reduce(builder, op_name, args, operator_options, |b, i, o| {
            b.reduce_l1_with_options(i, o)
        }),
        "reduceL2" => invoke_unary_reduce(builder, op_name, args, operator_options, |b, i, o| {
            b.reduce_l2_with_options(i, o)
        }),
        "reduceLogSum" => {
            invoke_unary_reduce(builder, op_name, args, operator_options, |b, i, o| {
                b.reduce_log_sum_with_options(i, o)
            })
        }
        "reduceLogSumExp" => {
            invoke_unary_reduce(builder, op_name, args, operator_options, |b, i, o| {
                b.reduce_log_sum_exp_with_options(i, o)
            })
        }
        "reduceSumSquare" => {
            invoke_unary_reduce(builder, op_name, args, operator_options, |b, i, o| {
                b.reduce_sum_square_with_options(i, o)
            })
        }
        "abs" => invoke_unary_simple(builder, op_name, args, |b, i| b.abs(i)),
        "ceil" => invoke_unary_simple(builder, op_name, args, |b, i| b.ceil(i)),
        "cos" => invoke_unary_simple(builder, op_name, args, |b, i| b.cos(i)),
        "exp" => invoke_unary_simple(builder, op_name, args, |b, i| b.exp(i)),
        "floor" => invoke_unary_simple(builder, op_name, args, |b, i| b.floor(i)),
        "log" => invoke_unary_simple(builder, op_name, args, |b, i| b.log(i)),
        "neg" => invoke_unary_simple(builder, op_name, args, |b, i| b.neg(i)),
        "reciprocal" => invoke_unary_simple(builder, op_name, args, |b, i| b.reciprocal(i)),
        "sign" => invoke_unary_simple(builder, op_name, args, |b, i| b.sign(i)),
        "sin" => invoke_unary_simple(builder, op_name, args, |b, i| b.sin(i)),
        "sqrt" => invoke_unary_simple(builder, op_name, args, |b, i| b.sqrt(i)),
        "tan" => invoke_unary_simple(builder, op_name, args, |b, i| b.tan(i)),
        "erf" => invoke_unary_simple(builder, op_name, args, |b, i| b.erf(i)),
        "roundEven" => invoke_unary_simple(builder, op_name, args, |b, i| b.round_even(i)),
        "relu" => invoke_unary_simple(builder, op_name, args, |b, i| b.relu(i)),
        "sigmoid" => invoke_unary_simple(builder, op_name, args, |b, i| b.sigmoid(i)),
        "tanh" => invoke_unary_simple(builder, op_name, args, |b, i| b.tanh(i)),
        "gelu" => invoke_unary_simple(builder, op_name, args, |b, i| b.gelu(i)),
        "hardSwish" => invoke_unary_simple(builder, op_name, args, |b, i| b.hard_swish(i)),
        "softplus" => invoke_unary_simple(builder, op_name, args, |b, i| b.softplus(i)),
        "softsign" => invoke_unary_simple(builder, op_name, args, |b, i| b.softsign(i)),
        "isNaN" => invoke_unary_simple(builder, op_name, args, |b, i| b.is_nan(i)),
        "isInfinite" => invoke_unary_simple(builder, op_name, args, |b, i| b.is_infinite(i)),
        "identity" => invoke_unary_simple(builder, op_name, args, |b, i| b.identity(i)),
        "logicalNot" => invoke_unary_simple(builder, op_name, args, |b, i| b.logical_not(i)),
        "shape" => invoke_unary_simple(builder, op_name, args, |b, i| b.shape(i)),
        "elu" => {
            let input = expect_operand(args, 0, op_name)?;
            let opts = operator_options.as_elu().cloned().unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .elu_with_options(input, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "leakyRelu" => {
            let input = expect_operand(args, 0, op_name)?;
            let opts = operator_options
                .as_leaky_relu()
                .cloned()
                .unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .leaky_relu_with_options(input, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "hardSigmoid" => {
            let input = expect_operand(args, 0, op_name)?;
            let opts = operator_options
                .as_hard_sigmoid()
                .cloned()
                .unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .hard_sigmoid_with_options(input, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "linear" => {
            let input = expect_operand(args, 0, op_name)?;
            let opts = operator_options.as_linear().cloned().unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .linear_with_options(input, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "clamp" => {
            let input = expect_operand(args, 0, op_name)?;
            let opts = operator_options.as_clamp().cloned().unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .clamp_with_options(input, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "transpose" => {
            let input = expect_operand(args, 0, op_name)?;
            let opts = operator_options.as_transpose().cloned().unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .transpose_with_options(input, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "squeeze" => {
            let input = expect_operand(args, 0, op_name)?;
            let opts = operator_options.as_squeeze().cloned().unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .squeeze_with_options(input, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "unsqueeze" => {
            let input = expect_operand(args, 0, op_name)?;
            let opts = operator_options.as_unsqueeze().cloned().unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .unsqueeze_with_options(input, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "reverse" => {
            let input = expect_operand(args, 0, op_name)?;
            let opts = operator_options.as_reverse().cloned().unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .reverse_with_options(input, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "resample2d" => {
            let input = expect_operand(args, 0, op_name)?;
            let opts = operator_options
                .as_resample2d()
                .cloned()
                .unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .resample2d_with_options(input, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "instanceNormalization" => {
            let input = expect_operand(args, 0, op_name)?;
            let opts = operator_options
                .as_instance_normalization()
                .cloned()
                .unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .instance_normalization_with_options(input, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "layerNormalization" => {
            let input = expect_operand(args, 0, op_name)?;
            let opts = operator_options
                .as_layer_normalization()
                .cloned()
                .unwrap_or_default();
            Ok(InvokeResult::Single(
                builder
                    .layer_normalization_with_options(input, opts)
                    .map_err(|e| e.to_string())?,
            ))
        }
        "averagePool2d" => invoke_pool2d(builder, op_name, args, operator_options, |b, i, o| {
            b.average_pool2d_with_options(i, o)
        }),
        "maxPool2d" => invoke_pool2d(builder, op_name, args, operator_options, |b, i, o| {
            b.max_pool2d_with_options(i, o)
        }),
        "l2Pool2d" => invoke_pool2d(builder, op_name, args, operator_options, |b, i, o| {
            b.l2_pool2d_with_options(i, o)
        }),
        "globalAveragePool" => {
            invoke_pool2d(builder, op_name, args, operator_options, |b, i, o| {
                b.global_average_pool_with_options(i, o)
            })
        }
        "globalMaxPool" => invoke_pool2d(builder, op_name, args, operator_options, |b, i, o| {
            b.global_max_pool_with_options(i, o)
        }),
        other => {
            let method = normalize_wpt_op_name(other);
            Err(format!(
                "Operation {other} (builder method {method}) not implemented"
            ))
        }
    }
}

fn unary_simple<F>(
    builder: &mut MLGraphBuilder,
    input: MLOperand,
    f: F,
) -> Result<InvokeResult, String>
where
    F: FnOnce(&mut MLGraphBuilder, MLOperand) -> std::result::Result<MLOperand, GraphBuilderError>,
{
    Ok(InvokeResult::Single(
        f(builder, input).map_err(|e| e.to_string())?,
    ))
}

fn unary_reduce<F>(
    builder: &mut MLGraphBuilder,
    input: MLOperand,
    operator_options: OperatorOptions,
    f: F,
) -> Result<InvokeResult, String>
where
    F: FnOnce(
        &mut MLGraphBuilder,
        MLOperand,
        MLReduceOptions,
    ) -> std::result::Result<MLOperand, GraphBuilderError>,
{
    let opts = operator_options.as_reduce().cloned().unwrap_or_default();
    Ok(InvokeResult::Single(
        f(builder, input, opts).map_err(|e| e.to_string())?,
    ))
}

fn pool2d<F>(
    builder: &mut MLGraphBuilder,
    input: MLOperand,
    operator_options: OperatorOptions,
    f: F,
) -> Result<InvokeResult, String>
where
    F: FnOnce(
        &mut MLGraphBuilder,
        MLOperand,
        MLPool2dOptions,
    ) -> std::result::Result<MLOperand, GraphBuilderError>,
{
    let opts = operator_options.as_pool2d().cloned().unwrap_or_default();
    Ok(InvokeResult::Single(
        f(builder, input, opts).map_err(|e| e.to_string())?,
    ))
}

fn register_operand(
    name: String,
    operand: MLOperand,
    operand_map: &mut HashMap<String, MLOperand>,
    name_to_id: &mut HashMap<String, u32>,
    next_operand_id: &mut u32,
    operand_names: &mut HashSet<String>,
) {
    name_to_id.insert(name.clone(), *next_operand_id);
    operand_map.insert(name.clone(), operand);
    operand_names.insert(name);
    *next_operand_id += 1;
}

fn add_constant_operand(
    builder: &mut MLGraphBuilder,
    spec: &WptTensorSpec,
) -> Result<MLOperand, String> {
    let desc = wpt_operand_descriptor(spec);
    let dtype = spec.data_type();
    match dtype {
        "float32" => builder
            .constant_from_slice(&desc, &tensor_f32_values(spec))
            .map_err(|e| e.to_string()),
        "float16" => builder
            .constant_from_slice(&desc, &tensor_f16_bits(spec))
            .map_err(|e| e.to_string()),
        "int32" => builder
            .constant_from_slice(&desc, &tensor_i32_values(spec))
            .map_err(|e| e.to_string()),
        "int8" => builder
            .constant_from_slice(&desc, &tensor_i8_values(spec))
            .map_err(|e| e.to_string()),
        "uint8" => builder
            .constant_from_slice(&desc, &tensor_u8_values(spec))
            .map_err(|e| e.to_string()),
        "uint32" => builder
            .constant_from_slice(&desc, &tensor_u32_values(spec))
            .map_err(|e| e.to_string()),
        "int64" => builder
            .constant_from_slice(&desc, &tensor_i64_values(spec))
            .map_err(|e| e.to_string()),
        "uint64" => builder
            .constant_from_slice(&desc, &tensor_u64_values(spec))
            .map_err(|e| e.to_string()),
        "int4" | "uint4" => {
            let bytes = tensor_spec_to_bytes(spec)?;
            builder
                .constant_from_slice(&desc, &bytes)
                .map_err(|e| e.to_string())
        }
        _ => {
            let bytes = tensor_spec_to_bytes(spec)?;
            builder
                .constant_from_slice(&desc, &bytes)
                .map_err(|e| e.to_string())
        }
    }
}

fn write_runtime_input(
    context: &mut MLContext,
    tensor: &MLTensor,
    spec: &WptTensorSpec,
) -> Result<(), String> {
    let dtype = spec.data_type();
    match dtype {
        "float32" => context
            .write_tensor(tensor, &tensor_f32_values(spec))
            .map_err(|e| e.to_string()),
        "float16" => context
            .write_tensor(tensor, &tensor_f16_bits(spec))
            .map_err(|e| e.to_string()),
        "int32" => context
            .write_tensor(tensor, &tensor_i32_values(spec))
            .map_err(|e| e.to_string()),
        "int8" => context
            .write_tensor(tensor, &tensor_i8_values(spec))
            .map_err(|e| e.to_string()),
        "uint8" => context
            .write_tensor(tensor, &tensor_u8_values(spec))
            .map_err(|e| e.to_string()),
        "int4" | "uint4" => {
            let bytes = tensor_spec_to_bytes(spec)?;
            context
                .write_tensor(tensor, &bytes)
                .map_err(|e| e.to_string())
        }
        "uint32" => context
            .write_tensor(tensor, &tensor_u32_values(spec))
            .map_err(|e| e.to_string()),
        "int64" => context
            .write_tensor(tensor, &tensor_i64_values(spec))
            .map_err(|e| e.to_string()),
        "uint64" => context
            .write_tensor(tensor, &tensor_u64_values(spec))
            .map_err(|e| e.to_string()),
        _ => {
            let bytes = tensor_spec_to_bytes(spec)?;
            context
                .write_tensor(tensor, &bytes)
                .map_err(|e| e.to_string())
        }
    }
}

fn read_output_tensor(
    context: &mut MLContext,
    tensor: &MLTensor,
    spec: &WptTensorSpec,
) -> Result<WptActualOutput, String> {
    let dtype = spec.data_type();
    let n = shape_element_count(spec.shape()).max(1);
    match dtype {
        "float32" => {
            let mut buf = vec![0.0f32; n];
            context
                .read_tensor(tensor, &mut buf)
                .map_err(|e| e.to_string())?;
            Ok(WptActualOutput {
                data_type: dtype.to_string(),
                f32_data: Some(buf),
                i32_data: None,
                i64_data: None,
            })
        }
        "float16" => {
            let mut buf = vec![0u16; n];
            context
                .read_tensor(tensor, &mut buf)
                .map_err(|e| e.to_string())?;
            let f32_data: Vec<f32> = buf
                .iter()
                .map(|&bits| f16::from_bits(bits).to_f32())
                .collect();
            Ok(WptActualOutput {
                data_type: dtype.to_string(),
                f32_data: Some(f32_data),
                i32_data: None,
                i64_data: None,
            })
        }
        "int32" => {
            let mut buf = vec![0i32; n];
            context
                .read_tensor(tensor, &mut buf)
                .map_err(|e| e.to_string())?;
            Ok(WptActualOutput {
                data_type: dtype.to_string(),
                f32_data: None,
                i32_data: Some(buf),
                i64_data: None,
            })
        }
        "int4" => {
            let packed_len = MLOperandDataType::Int4.rustnn_storage_byte_length(n).max(1);
            let mut bytes = vec![0u8; packed_len];
            context
                .read_tensor(tensor, &mut bytes)
                .map_err(|e| e.to_string())?;
            let logical = unpack_int4(&bytes, n);
            Ok(WptActualOutput {
                data_type: dtype.to_string(),
                f32_data: None,
                i32_data: Some(logical),
                i64_data: None,
            })
        }
        "int8" => {
            let mut buf = vec![0i8; n];
            context
                .read_tensor(tensor, &mut buf)
                .map_err(|e| e.to_string())?;
            Ok(WptActualOutput {
                data_type: dtype.to_string(),
                f32_data: None,
                i32_data: Some(buf.iter().map(|&v| v as i32).collect()),
                i64_data: None,
            })
        }
        "uint8" => {
            let mut buf = vec![0u8; n];
            context
                .read_tensor(tensor, &mut buf)
                .map_err(|e| e.to_string())?;
            Ok(WptActualOutput {
                data_type: dtype.to_string(),
                f32_data: None,
                i32_data: Some(buf.iter().map(|&v| v as i32).collect()),
                i64_data: None,
            })
        }
        "uint4" => {
            let packed_len = MLOperandDataType::Uint4
                .rustnn_storage_byte_length(n)
                .max(1);
            let mut bytes = vec![0u8; packed_len];
            context
                .read_tensor(tensor, &mut bytes)
                .map_err(|e| e.to_string())?;
            let logical = unpack_uint4(&bytes, n);
            Ok(WptActualOutput {
                data_type: dtype.to_string(),
                f32_data: None,
                i32_data: Some(logical.iter().map(|&v| v as i32).collect()),
                i64_data: None,
            })
        }
        "uint32" => {
            let mut buf = vec![0u32; n];
            context
                .read_tensor(tensor, &mut buf)
                .map_err(|e| e.to_string())?;
            Ok(WptActualOutput {
                data_type: dtype.to_string(),
                f32_data: None,
                i32_data: Some(buf.iter().map(|&v| v as i32).collect()),
                i64_data: None,
            })
        }
        "int64" | "uint64" => {
            let mut buf = vec![0i64; n];
            context
                .read_tensor(tensor, &mut buf)
                .map_err(|e| e.to_string())?;
            Ok(WptActualOutput {
                data_type: dtype.to_string(),
                f32_data: None,
                i32_data: None,
                i64_data: Some(buf),
            })
        }
        _ => {
            let mut buf = vec![0.0f32; n];
            context
                .read_tensor(tensor, &mut buf)
                .map_err(|e| e.to_string())?;
            Ok(WptActualOutput {
                data_type: dtype.to_string(),
                f32_data: Some(buf),
                i32_data: None,
                i64_data: None,
            })
        }
    }
}

pub fn append_webnn_graph_text(msg: String, webnn_text: Option<&str>) -> String {
    match webnn_text.filter(|text| !text.is_empty()) {
        Some(text) => format!("{msg}\n\n--- built graph ---\n{text}"),
        None => msg,
    }
}

/// Outputs from a successful WPT graph execution plus optional `.webnn` text captured at build time.
#[derive(Debug)]
pub struct WptExecuteArtifacts {
    pub outputs: HashMap<String, WptActualOutput>,
    pub webnn_text: Option<String>,
}

/// Execute a WPT graph through [`MLGraphBuilder`] and [`MLContext::dispatch`].
pub fn execute_wpt_graph(
    context: &mut MLContext,
    graph: &WptGraph,
) -> Result<WptExecuteArtifacts, String> {
    let mut builder = MLGraphBuilder::new(context).map_err(|e| e.to_string())?;
    let mut operand_map: HashMap<String, MLOperand> = HashMap::new();
    let mut name_to_id: HashMap<String, u32> = HashMap::new();
    let mut operand_names: HashSet<String> = graph.inputs.keys().cloned().collect();
    let mut runtime_input_names: Vec<String> = Vec::new();
    let mut next_operand_id = 0u32;

    for (name, spec) in &graph.inputs {
        if spec.constant && should_inline_constant(spec) {
            let operand = add_constant_operand(&mut builder, spec)?;
            register_operand(
                name.clone(),
                operand,
                &mut operand_map,
                &mut name_to_id,
                &mut next_operand_id,
                &mut operand_names,
            );
            continue;
        }
        let desc = wpt_operand_descriptor(spec);
        let operand = builder.input(name, &desc).map_err(|e| e.to_string())?;
        register_operand(
            name.clone(),
            operand,
            &mut operand_map,
            &mut name_to_id,
            &mut next_operand_id,
            &mut operand_names,
        );
        runtime_input_names.push(name.clone());
    }

    for op in &graph.operators {
        let op_name = op.name.as_str();
        let call_args = build_method_args(op_name, op, &name_to_id, &operand_map, &operand_names)?;
        let result = invoke_builder_method(&mut builder, op_name, &call_args)?;
        let out_names = output_names(op);

        match result {
            InvokeResult::Multi(results) if MULTI_OUTPUT_OPS.contains(&op_name) => {
                if results.len() != out_names.len() {
                    return Err(format!(
                        "{op_name}: expected {} outputs, got {}",
                        out_names.len(),
                        results.len()
                    ));
                }
                for (out_name, operand) in out_names.into_iter().zip(results) {
                    register_operand(
                        out_name,
                        operand,
                        &mut operand_map,
                        &mut name_to_id,
                        &mut next_operand_id,
                        &mut operand_names,
                    );
                }
            }
            InvokeResult::Single(result) => {
                if out_names.len() != 1 {
                    return Err(format!(
                        "{op_name}: expected {} outputs, got 1",
                        out_names.len()
                    ));
                }
                register_operand(
                    out_names[0].clone(),
                    result,
                    &mut operand_map,
                    &mut name_to_id,
                    &mut next_operand_id,
                    &mut operand_names,
                );
            }
            InvokeResult::Multi(_) => {
                return Err(format!("{op_name}: unexpected multi-output result"));
            }
        }
    }

    let mut build_outputs: HashMap<&str, MLOperand> = HashMap::new();
    for name in graph.expected_outputs.keys() {
        let operand = operand_map
            .get(name)
            .ok_or_else(|| format!("missing operand for expected output '{name}'"))?;
        build_outputs.insert(name.as_str(), *operand);
    }

    let webnn_text = builder.rustnn_webnn_text_for_outputs(&build_outputs);

    let mut ml_graph = builder
        .build(&build_outputs)
        .map_err(|e| append_webnn_graph_text(e.to_string(), webnn_text.as_deref()))?;

    let mut input_tensors: HashMap<&str, &MLTensor> = HashMap::new();
    let mut output_tensors_owned: HashMap<String, MLTensor> = HashMap::new();
    let mut input_owned: HashMap<String, MLTensor> = HashMap::new();

    for name in &runtime_input_names {
        let spec = graph
            .inputs
            .get(name)
            .ok_or_else(|| format!("runtime input '{name}' not found"))?;
        let tensor = context
            .create_tensor(&rw_tensor_descriptor(spec))
            .map_err(|e| append_webnn_graph_text(e.to_string(), webnn_text.as_deref()))?;
        write_runtime_input(context, &tensor, spec)?;
        input_owned.insert(name.clone(), tensor);
    }

    for (name, expected) in &graph.expected_outputs {
        let tensor = context
            .create_tensor(&rw_tensor_descriptor(expected))
            .map_err(|e| append_webnn_graph_text(e.to_string(), webnn_text.as_deref()))?;
        output_tensors_owned.insert(name.clone(), tensor);
    }

    for (name, tensor) in &input_owned {
        input_tensors.insert(name.as_str(), tensor);
    }
    let output_bindings: HashMap<&str, &MLTensor> = output_tensors_owned
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();

    context
        .dispatch(&mut ml_graph, &input_tensors, &output_bindings)
        .map_err(|e| append_webnn_graph_text(e.to_string(), webnn_text.as_deref()))?;

    let mut outputs = HashMap::new();
    for (name, expected) in &graph.expected_outputs {
        let tensor = output_tensors_owned
            .get(name)
            .ok_or_else(|| format!("output tensor '{name}' missing"))?;
        outputs.insert(
            name.clone(),
            read_output_tensor(context, tensor, expected)
                .map_err(|e| append_webnn_graph_text(e, webnn_text.as_deref()))?,
        );
    }
    Ok(WptExecuteArtifacts {
        outputs,
        webnn_text,
    })
}

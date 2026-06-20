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
//! WPT tensor spec helpers: parsing, packing, and expected-value conversion.

use rustnn::graph::{pack_int4, pack_uint4};

use super::wpt_types::{WptOperator, WptTensorSpec};

/// WPT camelCase operation name to rustnn op_type (matches Python method_name_map / graph builder).
pub(crate) fn normalize_wpt_op_name(name: &str) -> String {
    let s: &'static str = match name {
        "reduceSum" => "reduce_sum",
        "reduceMean" => "reduce_mean",
        "reduceMax" => "reduce_max",
        "reduceMin" => "reduce_min",
        "reduceProduct" => "reduce_product",
        "reduceL1" => "reduce_l1",
        "reduceL2" => "reduce_l2",
        "reduceLogSum" => "reduce_log_sum",
        "reduceLogSumExp" => "reduce_log_sum_exp",
        "reduceSumSquare" => "reduce_sum_square",
        "relu" => "relu",
        "sigmoid" => "sigmoid",
        "tanh" => "tanh",
        "softmax" => "softmax",
        "leakyRelu" => "leaky_relu",
        "hardSigmoid" => "hard_sigmoid",
        "hardSwish" => "hard_swish",
        "elu" => "elu",
        "gelu" => "gelu",
        "prelu" => "prelu",
        "softplus" => "softplus",
        "softsign" => "softsign",
        "batchNormalization" => "batch_normalization",
        "instanceNormalization" => "instance_normalization",
        "layerNormalization" => "layer_normalization",
        "conv2d" => "conv2d",
        "convTranspose2d" => "conv_transpose2d",
        "averagePool2d" => "average_pool2d",
        "maxPool2d" => "max_pool2d",
        "l2Pool2d" => "l2_pool2d",
        "globalAveragePool" => "global_average_pool",
        "globalMaxPool" => "global_max_pool",
        "add" => "add",
        "sub" => "sub",
        "mul" => "mul",
        "div" => "div",
        "matmul" => "matmul",
        "equal" => "equal",
        "greater" => "greater",
        "greaterOrEqual" => "greater_or_equal",
        "lesser" => "lesser",
        "lesserOrEqual" => "lesser_or_equal",
        "logicalAnd" => "logical_and",
        "logicalOr" => "logical_or",
        "logicalNot" => "logical_not",
        "logicalXor" => "logical_xor",
        "abs" => "abs",
        "ceil" => "ceil",
        "cos" => "cos",
        "exp" => "exp",
        "floor" => "floor",
        "log" => "log",
        "neg" => "neg",
        "reciprocal" => "reciprocal",
        "sign" => "sign",
        "sin" => "sin",
        "sqrt" => "sqrt",
        "tan" => "tan",
        "erf" => "erf",
        "roundEven" => "round_even",
        "reshape" => "reshape",
        "transpose" => "transpose",
        "concat" => "concat",
        "expand" => "expand",
        "gather" => "gather",
        "pad" => "pad",
        "slice" => "slice",
        "split" => "split",
        "squeeze" => "squeeze",
        "unsqueeze" => "unsqueeze",
        "tile" => "tile",
        "cast" => "cast",
        "clamp" => "clamp",
        "gemm" => "gemm",
        "where" => "where",
        "identity" => "identity",
        "quantizeLinear" => "quantize_linear",
        "dequantizeLinear" => "dequantize_linear",
        "scatterElements" => "scatter_elements",
        "scatterND" => "scatter_nd",
        "triangular" => "triangular",
        "argMax" => "arg_max",
        "argMin" => "arg_min",
        "pow" => "pow",
        "max" => "max",
        "min" => "min",
        _ => return name.to_string(),
    };
    s.to_string()
}

/// WebNN option keys that reference graph operands (mirrors pywebnn OPTION_OPERAND_KEYS).
pub(crate) fn is_operand_option(wpt_name: &str, key: &str) -> bool {
    match wpt_name {
        "batchNormalization" => matches!(key, "scale" | "bias"),
        "conv2d" | "convTranspose2d" => key == "bias",
        "gemm" => key == "c",
        "gru" => matches!(key, "bias" | "recurrentBias" | "initialHiddenState"),
        "gruCell" => matches!(key, "bias" | "recurrentBias"),
        "instanceNormalization" | "layerNormalization" => matches!(key, "scale" | "bias"),
        "lstm" => matches!(
            key,
            "bias" | "recurrentBias" | "peepholeWeight" | "initialHiddenState" | "initialCellState"
        ),
        "lstmCell" => matches!(key, "bias" | "recurrentBias" | "peepholeWeight"),
        _ => false,
    }
}

fn is_pool2d_op(op_type: &str) -> bool {
    matches!(
        op_type,
        "average_pool2d"
            | "max_pool2d"
            | "l2_pool2d"
            | "l2Pool2d"
            | "global_average_pool"
            | "global_max_pool"
    )
}

pub(crate) fn normalize_option_key(wpt_name: &str, op_type: &str, key: &str) -> String {
    if wpt_name == "cast" && key == "type" {
        return "to".to_string();
    }
    if is_pool2d_op(op_type) && key == "roundingType" {
        return "outputShapeRounding".to_string();
    }
    key.to_string()
}

/// Parse a numeric value from WPT JSON (handles "123n" bigint and numbers).
fn parse_number(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => {
            let t = s.trim_end_matches('n');
            t.parse::<i64>()
                .ok()
                .map(|i| i as f64)
                .or_else(|| t.parse::<f64>().ok())
        }
        _ => None,
    }
}

/// Parse a float from WPT JSON for tensor data (handles "Infinity", "-Infinity", "NaN" strings and null).
pub(crate) fn parse_float_for_tensor(v: &serde_json::Value) -> Option<f32> {
    if v.is_null() {
        return Some(f32::NAN);
    }
    if let Some(s) = v.as_str() {
        return match s {
            "Infinity" => Some(f32::INFINITY),
            "-Infinity" => Some(f32::NEG_INFINITY),
            "NaN" => Some(f32::NAN),
            _ => parse_number(v).map(|f| f as f32),
        };
    }
    parse_number(v).or_else(|| v.as_f64()).map(|f| f as f32)
}

fn parse_integer_for_tensor(v: &serde_json::Value) -> Option<i32> {
    if let Some(x) = v.as_i64() {
        return Some(x as i32);
    }
    if let Some(x) = v.as_u64() {
        return Some(x as i32);
    }
    if let Some(f) = parse_number(v) {
        return Some(f as i32);
    }
    v.as_str()
        .and_then(|s| s.trim_end_matches('n').parse::<i64>().ok())
        .map(|x| x as i32)
}

pub(crate) fn fill_i32_tensor_values(spec: &WptTensorSpec, n: usize) -> Vec<i32> {
    let mut buf = vec![0i32; n];
    if let Some(arr) = spec.data.as_array() {
        if arr.len() == 1 && n > 1 {
            let v = arr
                .first()
                .and_then(parse_integer_for_tensor)
                .unwrap_or(0)
                .clamp(-8, 7);
            buf.fill(v);
        } else {
            for (i, v) in arr.iter().enumerate().take(n) {
                if let Some(x) = parse_integer_for_tensor(v) {
                    buf[i] = x.clamp(-8, 7);
                }
            }
        }
    } else if let Some(x) = parse_integer_for_tensor(&spec.data) {
        buf.fill(x.clamp(-8, 7));
    }
    buf
}

pub(crate) fn fill_u8_tensor_values(spec: &WptTensorSpec, n: usize) -> Vec<u8> {
    let mut buf = vec![0u8; n];
    if let Some(arr) = spec.data.as_array() {
        if arr.len() == 1 && n > 1 {
            let v = arr
                .first()
                .and_then(parse_integer_for_tensor)
                .unwrap_or(0)
                .clamp(0, 15) as u8;
            buf.fill(v);
        } else {
            for (i, v) in arr.iter().enumerate().take(n) {
                if let Some(x) = parse_integer_for_tensor(v) {
                    buf[i] = (x as u8) & 0x0F;
                }
            }
        }
    } else if let Some(x) = parse_integer_for_tensor(&spec.data) {
        buf.fill((x as u8) & 0x0F);
    }
    buf
}

/// Get output name(s) from operator (string or array of strings).
pub(crate) fn output_names(op: &WptOperator) -> Vec<String> {
    if let Some(s) = op.outputs.as_str() {
        return vec![s.to_string()];
    }
    if let Some(arr) = op.outputs.as_array() {
        return arr
            .iter()
            .filter_map(|v: &serde_json::Value| v.as_str().map(String::from))
            .collect::<Vec<String>>();
    }
    Vec::new()
}

/// Serialize tensor spec data to bytes (for constants).
pub(crate) fn tensor_spec_to_bytes(spec: &WptTensorSpec) -> Result<Vec<u8>, String> {
    let shape = spec.shape();
    let dtype = spec.data_type();
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let n = n.max(1);

    let arr_opt: Option<&Vec<serde_json::Value>> = spec.data.as_array();
    let bytes = match dtype {
        "float32" => {
            let mut buf = vec![0.0f32; n];
            if let Some(arr) = arr_opt {
                for (i, v) in arr.iter().enumerate().take(n) {
                    buf[i] = parse_float_for_tensor(v).unwrap_or(buf[i]);
                }
            } else if let Some(f) = parse_float_for_tensor(&spec.data) {
                buf.fill(f);
            }
            buf.iter().flat_map(|f| f.to_ne_bytes()).collect()
        }
        "float16" => {
            let mut buf = vec![0u16; n];
            if let Some(arr) = arr_opt {
                for (i, v) in arr.iter().enumerate().take(n) {
                    let f = parse_float_for_tensor(v).unwrap_or(0.0);
                    buf[i] = half::f16::from_f32(f).to_bits();
                }
            } else if let Some(f) = parse_float_for_tensor(&spec.data) {
                let h = half::f16::from_f32(f).to_bits();
                buf.fill(h);
            }
            buf.iter().flat_map(|u| u.to_ne_bytes()).collect()
        }
        "int32" => {
            let mut buf = vec![0i32; n];
            if let Some(arr) = arr_opt {
                for (i, v) in arr.iter().enumerate().take(n) {
                    if let Some(x) = v.as_i64().or_else(|| parse_number(v).map(|f| f as i64)) {
                        buf[i] = x as i32;
                    }
                }
            } else if let Some(x) = spec
                .data
                .as_i64()
                .or_else(|| parse_number(&spec.data).map(|f| f as i64))
            {
                buf.fill(x as i32);
            }
            buf.iter().flat_map(|x| x.to_ne_bytes()).collect()
        }
        "uint32" => {
            let mut buf = vec![0u32; n];
            if let Some(arr) = arr_opt {
                for (i, v) in arr.iter().enumerate().take(n) {
                    if let Some(x) = v.as_u64().or_else(|| parse_number(v).map(|f| f as u64)) {
                        buf[i] = x as u32;
                    }
                }
            } else if let Some(x) = spec
                .data
                .as_u64()
                .or_else(|| parse_number(&spec.data).map(|f| f as u64))
            {
                buf.fill(x as u32);
            }
            buf.iter().flat_map(|x| x.to_ne_bytes()).collect()
        }
        "int8" => {
            let mut buf = vec![0i8; n];
            if let Some(arr) = arr_opt {
                for (i, v) in arr.iter().enumerate().take(n) {
                    if let Some(x) = v.as_i64().or_else(|| parse_number(v).map(|f| f as i64)) {
                        buf[i] = x as i8;
                    }
                }
            } else if let Some(x) = spec
                .data
                .as_i64()
                .or_else(|| parse_number(&spec.data).map(|f| f as i64))
            {
                buf.fill(x as i8);
            }
            buf.iter().map(|&x| x as u8).collect()
        }
        "uint8" => {
            let mut buf = vec![0u8; n];
            if let Some(arr) = arr_opt {
                for (i, v) in arr.iter().enumerate().take(n) {
                    if let Some(x) = v.as_u64().or_else(|| parse_number(v).map(|f| f as u64)) {
                        buf[i] = x as u8;
                    }
                }
            } else if let Some(x) = spec
                .data
                .as_u64()
                .or_else(|| parse_number(&spec.data).map(|f| f as u64))
            {
                buf.fill(x as u8);
            }
            buf
        }
        "int4" => pack_int4(&fill_i32_tensor_values(spec, n)),
        "uint4" => {
            let logical: Vec<u8> = fill_u8_tensor_values(spec, n);
            pack_uint4(&logical)
        }
        "int64" => {
            let mut buf = vec![0i64; n];
            if let Some(arr) = arr_opt {
                for (i, v) in arr.iter().enumerate().take(n) {
                    if let Some(x) = v
                        .as_i64()
                        .or_else(|| {
                            v.as_str()
                                .and_then(|s| s.trim_end_matches('n').parse::<i64>().ok())
                        })
                        .or_else(|| parse_number(v).map(|f| f as i64))
                    {
                        buf[i] = x;
                    }
                }
            } else if let Some(x) = spec
                .data
                .as_i64()
                .or_else(|| {
                    spec.data
                        .as_str()
                        .and_then(|s| s.trim_end_matches('n').parse::<i64>().ok())
                })
                .or_else(|| parse_number(&spec.data).map(|f| f as i64))
            {
                buf.fill(x);
            }
            buf.iter().flat_map(|x| x.to_ne_bytes()).collect()
        }
        "uint64" => {
            let mut buf = vec![0u64; n];
            if let Some(arr) = arr_opt {
                for (i, v) in arr.iter().enumerate().take(n) {
                    if let Some(x) = v
                        .as_u64()
                        .or_else(|| parse_number(v).map(|f| f as u64))
                        .or_else(|| {
                            v.as_str()
                                .and_then(|s| s.trim_end_matches('n').parse::<u64>().ok())
                        })
                    {
                        buf[i] = x;
                    }
                }
            } else if let Some(x) = spec
                .data
                .as_u64()
                .or_else(|| parse_number(&spec.data).map(|f| f as u64))
                .or_else(|| {
                    spec.data
                        .as_str()
                        .and_then(|s| s.trim_end_matches('n').parse::<u64>().ok())
                })
            {
                buf.fill(x);
            }
            buf.iter().flat_map(|x| x.to_ne_bytes()).collect()
        }
        _ => {
            let mut buf = vec![0.0f32; n];
            if let Some(arr) = arr_opt {
                for (i, v) in arr.iter().enumerate().take(n) {
                    if let Some(f) = parse_number(v).or_else(|| v.as_f64()) {
                        buf[i] = f as f32;
                    }
                }
            } else if let Some(f) = parse_number(&spec.data).or_else(|| spec.data.as_f64()) {
                buf.fill(f as f32);
            }
            buf.iter().flat_map(|f| f.to_ne_bytes()).collect()
        }
    };
    Ok(bytes)
}

/// Logical element count from a tensor spec shape.
pub fn element_count(spec: &WptTensorSpec) -> usize {
    let n: usize = spec.shape().iter().map(|&d| d as usize).product();
    n.max(1)
}

fn fit_len<T: Clone>(mut buf: Vec<T>, n: usize, fill: T) -> Vec<T> {
    if buf.len() < n {
        buf.resize(n, fill);
    }
    if buf.len() > n {
        buf.truncate(n);
    }
    buf
}

/// Float32 payload for runtime I/O and builder constants (scalar-fill when JSON has one value).
pub fn tensor_f32_values(spec: &WptTensorSpec) -> Vec<f32> {
    let n = element_count(spec);
    let mut buf = vec![0.0f32; n];
    if let Some(arr) = spec.data.as_array() {
        if arr.len() == 1 && n > 1 {
            if let Some(v) = parse_float_for_tensor(&arr[0]) {
                buf.fill(v);
            }
        } else {
            for (i, v) in arr.iter().enumerate().take(n) {
                if let Some(f) = parse_float_for_tensor(v) {
                    buf[i] = f;
                }
            }
        }
    } else if let Some(f) = parse_float_for_tensor(&spec.data) {
        buf.fill(f);
    }
    buf
}

/// Float16 payload as raw bits for runtime I/O and builder constants.
pub fn tensor_f16_bits(spec: &WptTensorSpec) -> Vec<u16> {
    tensor_f32_values(spec)
        .into_iter()
        .map(|f| half::f16::from_f32(f).to_bits())
        .collect()
}

pub fn tensor_i32_values(spec: &WptTensorSpec) -> Vec<i32> {
    fit_len(expected_output_to_i32(spec), element_count(spec), 0)
}

pub fn tensor_i8_values(spec: &WptTensorSpec) -> Vec<i8> {
    fit_len(expected_output_to_i8(spec), element_count(spec), 0)
}

pub fn tensor_u8_values(spec: &WptTensorSpec) -> Vec<u8> {
    fit_len(expected_output_to_u8(spec), element_count(spec), 0)
}

pub fn tensor_u32_values(spec: &WptTensorSpec) -> Vec<u32> {
    fit_len(expected_output_to_u32(spec), element_count(spec), 0)
}

pub fn tensor_i64_values(spec: &WptTensorSpec) -> Vec<i64> {
    fit_len(expected_output_to_i64(spec), element_count(spec), 0)
}

pub fn tensor_u64_values(spec: &WptTensorSpec) -> Vec<u64> {
    fit_len(expected_output_to_u64(spec), element_count(spec), 0)
}

/// Expected output as f32 slice (for validation). Converts from WPT expected_outputs.
/// Handles "Infinity", "-Infinity", "NaN" strings and null (→ NaN).
pub fn expected_output_to_f32(spec: &WptTensorSpec) -> Vec<f32> {
    let shape = spec.shape();
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let n = n.max(1);
    let mut buf = vec![0.0f32; n];
    let arr_opt: Option<&Vec<serde_json::Value>> = spec.data.as_array();
    if let Some(arr) = arr_opt {
        for (i, v) in arr.iter().enumerate().take(n) {
            buf[i] = parse_float_for_tensor(v).unwrap_or(buf[i]);
        }
    } else if let Some(f) = parse_float_for_tensor(&spec.data) {
        buf.fill(f);
    }
    // float16 expected tensors: round-trip through f16 (matches pywebnn wpt_assert).
    if spec.data_type().eq_ignore_ascii_case("float16") {
        for v in &mut buf {
            *v = half::f16::from_f32(*v).to_f32();
        }
    }
    buf
}

/// Expected output as i32 slice (for integer validation).
pub fn expected_output_to_i32(spec: &WptTensorSpec) -> Vec<i32> {
    let shape = spec.shape();
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let n = n.max(1);
    let mut buf = vec![0i32; n];
    let arr_opt: Option<&Vec<serde_json::Value>> = spec.data.as_array();
    if let Some(arr) = arr_opt {
        if arr.len() == 1 && n > 1 {
            if let Some(x) = arr.first().and_then(|v| {
                v.as_i64()
                    .or_else(|| parse_number(v).map(|f| f as i64))
                    .or_else(|| {
                        v.as_str()
                            .and_then(|s| s.trim_end_matches('n').parse::<i64>().ok())
                    })
            }) {
                buf.fill(x as i32);
            }
        } else {
            for (i, v) in arr.iter().enumerate().take(n) {
                if let Some(x) = v
                    .as_i64()
                    .or_else(|| parse_number(v).map(|f| f as i64))
                    .or_else(|| {
                        v.as_str()
                            .and_then(|s| s.trim_end_matches('n').parse::<i64>().ok())
                    })
                {
                    buf[i] = x as i32;
                }
            }
        }
    } else if let Some(x) = spec
        .data
        .as_i64()
        .or_else(|| parse_number(&spec.data).map(|f| f as i64))
        .or_else(|| {
            spec.data
                .as_str()
                .and_then(|s| s.trim_end_matches('n').parse::<i64>().ok())
        })
    {
        buf.fill(x as i32);
    }
    buf
}

/// Expected output as u32 slice (for uint32 validation).
pub fn expected_output_to_u32(spec: &WptTensorSpec) -> Vec<u32> {
    let shape = spec.shape();
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let n = n.max(1);
    let mut buf = vec![0u32; n];
    let arr_opt: Option<&Vec<serde_json::Value>> = spec.data.as_array();
    if let Some(arr) = arr_opt {
        for (i, v) in arr.iter().enumerate().take(n) {
            if let Some(x) = v
                .as_u64()
                .or_else(|| parse_number(v).map(|f| f as u64))
                .or_else(|| v.as_i64().map(|i| i as u64))
            {
                buf[i] = x as u32;
            }
        }
    } else if let Some(x) = spec
        .data
        .as_u64()
        .or_else(|| parse_number(&spec.data).map(|f| f as u64))
        .or_else(|| spec.data.as_i64().map(|i| i as u64))
    {
        buf.fill(x as u32);
    }
    buf
}

/// Expected output as u8 slice (for uint8 validation).
pub fn expected_output_to_u8(spec: &WptTensorSpec) -> Vec<u8> {
    let shape = spec.shape();
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let n = n.max(1);
    let mut buf = vec![0u8; n];
    let arr_opt: Option<&Vec<serde_json::Value>> = spec.data.as_array();
    if let Some(arr) = arr_opt {
        for (i, v) in arr.iter().enumerate().take(n) {
            if let Some(x) = v
                .as_u64()
                .or_else(|| parse_number(v).map(|f| f as u64))
                .or_else(|| v.as_i64().map(|i| i as u64))
            {
                buf[i] = x as u8;
            }
        }
    } else if let Some(x) = spec
        .data
        .as_u64()
        .or_else(|| parse_number(&spec.data).map(|f| f as u64))
        .or_else(|| spec.data.as_i64().map(|i| i as u64))
    {
        buf.fill(x as u8);
    }
    buf
}

/// Expected output as i8 slice (for int8 validation).
pub fn expected_output_to_i8(spec: &WptTensorSpec) -> Vec<i8> {
    let shape = spec.shape();
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let n = n.max(1);
    let mut buf = vec![0i8; n];
    let arr_opt: Option<&Vec<serde_json::Value>> = spec.data.as_array();
    if let Some(arr) = arr_opt {
        for (i, v) in arr.iter().enumerate().take(n) {
            if let Some(x) = v
                .as_i64()
                .or_else(|| parse_number(v).map(|f| f as i64))
                .or_else(|| {
                    v.as_str()
                        .and_then(|s| s.trim_end_matches('n').parse::<i64>().ok())
                })
            {
                buf[i] = x as i8;
            }
        }
    } else if let Some(x) = spec
        .data
        .as_i64()
        .or_else(|| parse_number(&spec.data).map(|f| f as i64))
        .or_else(|| {
            spec.data
                .as_str()
                .and_then(|s| s.trim_end_matches('n').parse::<i64>().ok())
        })
    {
        buf.fill(x as i8);
    }
    buf
}

/// Expected output as i64 slice (for int64 validation). Handles numbers and bigint strings.
/// Prefer string parse for i64 to avoid f64 precision loss for values outside ±2^53.
#[allow(dead_code)]
pub fn expected_output_to_i64(spec: &WptTensorSpec) -> Vec<i64> {
    let shape = spec.shape();
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let n = n.max(1);
    let mut buf = vec![0i64; n];
    let arr_opt: Option<&Vec<serde_json::Value>> = spec.data.as_array();
    if let Some(arr) = arr_opt {
        for (i, v) in arr.iter().enumerate().take(n) {
            if let Some(x) = v
                .as_i64()
                .or_else(|| {
                    v.as_str()
                        .and_then(|s| s.trim_end_matches('n').parse::<i64>().ok())
                })
                .or_else(|| parse_number(v).map(|f| f as i64))
            {
                buf[i] = x;
            }
        }
    } else if let Some(x) = spec
        .data
        .as_i64()
        .or_else(|| {
            spec.data
                .as_str()
                .and_then(|s| s.trim_end_matches('n').parse::<i64>().ok())
        })
        .or_else(|| parse_number(&spec.data).map(|f| f as i64))
    {
        buf.fill(x);
    }
    buf
}

/// Expected output as u64 slice (for uint64 validation). Handles numbers and bigint strings.
#[allow(dead_code)]
pub fn expected_output_to_u64(spec: &WptTensorSpec) -> Vec<u64> {
    let shape = spec.shape();
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let n = n.max(1);
    let mut buf = vec![0u64; n];
    let arr_opt: Option<&Vec<serde_json::Value>> = spec.data.as_array();
    if let Some(arr) = arr_opt {
        for (i, v) in arr.iter().enumerate().take(n) {
            if let Some(x) = v
                .as_u64()
                .or_else(|| parse_number(v).map(|f| f as u64))
                .or_else(|| {
                    v.as_str()
                        .and_then(|s| s.trim_end_matches('n').parse::<u64>().ok())
                })
            {
                buf[i] = x;
            }
        }
    } else if let Some(x) = spec
        .data
        .as_u64()
        .or_else(|| parse_number(&spec.data).map(|f| f as u64))
        .or_else(|| {
            spec.data
                .as_str()
                .and_then(|s| s.trim_end_matches('n').parse::<u64>().ok())
        })
    {
        buf.fill(x);
    }
    buf
}

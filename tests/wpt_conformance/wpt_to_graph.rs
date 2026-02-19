//! Convert WPT graph description to rustnn GraphInfo and prepare ONNX inputs.

use rustnn::graph::{
    ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, Operation,
};
#[cfg(feature = "onnx-runtime")]
use rustnn::{OnnxInput, TensorData};
use std::collections::HashMap;
use std::path::Path;

use super::wpt_types::{WptGraph, WptOperator, WptTensorSpec};

/// WPT camelCase operation name to rustnn snake_case op_type (matches Python method_name_map).
fn wpt_op_name_to_rustnn(name: &str) -> String {
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
        "acos" => "acos",
        "asin" => "asin",
        "atan" => "atan",
        "acosh" => "acosh",
        "asinh" => "asinh",
        "atanh" => "atanh",
        "cosh" => "cosh",
        "sinh" => "sinh",
        "erf" => "erf",
        "round" => "round",
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
        _ => return name.to_string(),
    };
    s.to_string()
}

fn wpt_data_type(s: &str) -> DataType {
    match s.to_lowercase().as_str() {
        "float32" => DataType::Float32,
        "float16" => DataType::Float16,
        "int32" => DataType::Int32,
        "uint32" => DataType::Uint32,
        "int8" => DataType::Int8,
        "uint8" => DataType::Uint8,
        "int64" => DataType::Int64,
        "uint64" => DataType::Uint64,
        _ => DataType::Float32,
    }
}

/// Parse a numeric value from WPT JSON (handles "123n" bigint and numbers).
fn parse_number(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => {
            let t = s.trim_end_matches('n');
            t.parse::<i64>().ok().map(|i| i as f64).or_else(|| t.parse::<f64>().ok())
        }
        _ => None,
    }
}

/// Parse a float from WPT JSON for tensor data (handles "Infinity", "-Infinity", "NaN" strings and null).
fn parse_float_for_tensor(v: &serde_json::Value) -> Option<f32> {
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

/// Flatten WPT arguments: list of single-key objects -> one map; and merge "options" into top level.
fn flatten_args(op: &WptOperator) -> HashMap<String, serde_json::Value> {
    let mut out: HashMap<String, serde_json::Value> = HashMap::new();
    if let Some(arr) = op.arguments.as_array() {
        for item in arr {
            if let Some(obj) = item.as_object() {
                for (k, v) in obj {
                    out.insert(k.clone(), v.clone());
                }
            }
        }
    } else if let Some(obj) = op.arguments.as_object() {
        for (k, v) in obj {
            out.insert(k.clone(), v.clone());
        }
    }
    if let Some(options) = out.get("options").and_then(|v: &serde_json::Value| v.as_object()) {
        let entries: Vec<(String, serde_json::Value)> = options
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        out.remove("options");
        for (k, v) in entries {
            out.insert(k, v);
        }
    }
    out
}

/// Get output name(s) from operator (string or array of strings).
fn output_names(op: &WptOperator) -> Vec<String> {
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
fn tensor_spec_to_bytes(spec: &WptTensorSpec) -> Result<Vec<u8>, String> {
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
            } else if let Some(x) = spec.data.as_i64().or_else(|| parse_number(&spec.data).map(|f| f as i64)) {
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
            } else if let Some(x) = spec.data.as_u64().or_else(|| parse_number(&spec.data).map(|f| f as u64)) {
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
            } else if let Some(x) = spec.data.as_i64().or_else(|| parse_number(&spec.data).map(|f| f as i64)) {
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
            } else if let Some(x) = spec.data.as_u64().or_else(|| parse_number(&spec.data).map(|f| f as u64)) {
                buf.fill(x as u8);
            }
            buf
        }
        "int64" => {
            let mut buf = vec![0i64; n];
            if let Some(arr) = arr_opt {
                for (i, v) in arr.iter().enumerate().take(n) {
                    if let Some(x) = v
                        .as_i64()
                        .or_else(|| v.as_str().and_then(|s| s.trim_end_matches('n').parse::<i64>().ok()))
                        .or_else(|| parse_number(v).map(|f| f as i64))
                    {
                        buf[i] = x;
                    }
                }
            } else if let Some(x) = spec
                .data
                .as_i64()
                .or_else(|| spec.data.as_str().and_then(|s| s.trim_end_matches('n').parse::<i64>().ok()))
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
                    if let Some(x) = v.as_u64()
                        .or_else(|| parse_number(v).map(|f| f as u64))
                        .or_else(|| v.as_str().and_then(|s| s.trim_end_matches('n').parse::<u64>().ok()))
                    {
                        buf[i] = x;
                    }
                }
            } else if let Some(x) = spec.data.as_u64()
                .or_else(|| parse_number(&spec.data).map(|f| f as u64))
                .or_else(|| spec.data.as_str().and_then(|s| s.trim_end_matches('n').parse::<u64>().ok()))
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

/// Build GraphInfo and list of (non-constant) input names in order for a WPT graph.
pub fn wpt_graph_to_graph_info(graph: &WptGraph) -> Result<(GraphInfo, Vec<String>), String> {
    let mut operands: Vec<Operand> = Vec::new();
    let mut name_to_id: HashMap<String, u32> = HashMap::new();
    let mut constant_data: HashMap<u32, ConstantData> = HashMap::new();
    let mut input_operand_ids: Vec<u32> = Vec::new();
    let mut operations: Vec<Operation> = Vec::new();

    let mut next_id = 0u32;

    for (input_name, spec) in &graph.inputs {
        let shape: Vec<u32> = spec.shape().to_vec();
        let data_type = wpt_data_type(spec.data_type());
        let kind = if spec.constant {
            let bytes = tensor_spec_to_bytes(spec)?;
            constant_data.insert(next_id, ConstantData { data: bytes, label: None });
            OperandKind::Constant
        } else {
            input_operand_ids.push(next_id);
            OperandKind::Input
        };
        operands.push(Operand {
            kind,
            descriptor: OperandDescriptor {
                data_type,
                shape: shape.clone(),
                pending_permutation: Vec::new(),
            },
            name: Some(input_name.clone()),
        });
        name_to_id.insert(input_name.clone(), next_id);
        next_id += 1;
    }

    for op in &graph.operators {
        let args = flatten_args(op);
        let op_type = wpt_op_name_to_rustnn(&op.name);
        let mut input_ids = Vec::new();
        let mut attributes = serde_json::Map::new();

        for (key, value) in &args {
            if value.is_string() {
                if let Some(name) = value.as_str() {
                    if let Some(&id) = name_to_id.get(name) {
                        if key == "input"
                            || key == "a"
                            || key == "b"
                            || key == "condition"
                            || key == "data"
                            || key == "x"
                            || key == "mean"
                            || key == "variance"
                            || key == "scale"
                            || key == "bias"
                        {
                            input_ids.push(id);
                        }
                        continue;
                    }
                }
            }
            if value.is_array() {
                let arr = value.as_array().unwrap();
                let mut all_refs = true;
                let mut ids = Vec::new();
                for item in arr {
                    if let Some(name) = item.as_str() {
                        if let Some(&id) = name_to_id.get(name) {
                            ids.push(id);
                            continue;
                        }
                    }
                    all_refs = false;
                    break;
                }
                if all_refs && ids.len() == arr.len() {
                    if key == "inputs" || key == "values" {
                        input_ids.extend(ids);
                        continue;
                    }
                }
            }
            // Omit null (no bound). Keep "Infinity"/"-Infinity"/"NaN" strings for clamp; normalize bigint to number.
            let value = if value.is_null() {
                continue;
            } else if (key == "minValue" || key == "maxValue") && value.is_string() {
                let s = value.as_str().unwrap_or("");
                if s == "Infinity" || s == "-Infinity" || s == "NaN" {
                    value.clone()
                } else {
                    parse_number(value)
                        .map(serde_json::Value::from)
                        .unwrap_or_else(|| value.clone())
                }
            } else {
                value.clone()
            };
            let attr_key = match key.as_str() {
                "newShape" => "newShape".to_string(),
                "keepDimensions" => "keepDimensions".to_string(),
                "axes" => "axes".to_string(),
                "axis" => "axis".to_string(),
                "padding" | "pads" => "pads".to_string(),
                "strides" => "strides".to_string(),
                "dilations" => "dilations".to_string(),
                "groups" => "groups".to_string(),
                "outputPadding" => "output_padding".to_string(),
                "outputSizes" => "output_shape".to_string(),
                "minValue" => "minValue".to_string(),
                "maxValue" => "maxValue".to_string(),
                "mode" => "mode".to_string(),
                "value" => "value".to_string(),
                "permutation" => "permutation".to_string(),
                "sizes" => "sizes".to_string(),
                "starts" => "starts".to_string(),
                "ends" => "ends".to_string(),
                "upper" => "upper".to_string(),
                "diagonal" => "diagonal".to_string(),
                "alpha" => "alpha".to_string(),
                "beta" => "beta".to_string(),
                "aTranspose" => "aTranspose".to_string(),
                "bTranspose" => "bTranspose".to_string(),
                "inputLayout" => "input_layout".to_string(),
                "filterLayout" => "filter_layout".to_string(),
                "type" => {
                    if op_type == "cast" {
                        "to".to_string()
                    } else {
                        "data_type".to_string()
                    }
                },
                "to" => "to".to_string(),
                _ => key.clone(),
            };
            attributes.insert(attr_key, value.clone());
        }

        if input_ids.is_empty() {
            for key in ["input", "a", "b", "condition", "data", "x"] {
                if let Some(v) = args.get(key) {
                    if let Some(name) = v.as_str() {
                        if let Some(&id) = name_to_id.get(name) {
                            input_ids.push(id);
                        }
                    }
                }
            }
            if input_ids.is_empty() {
                if let Some(arr) = args.get("inputs").or_else(|| args.get("values")).and_then(|v| v.as_array()) {
                    for item in arr {
                        if let Some(name) = item.as_str() {
                            if let Some(&id) = name_to_id.get(name) {
                                input_ids.push(id);
                            }
                        }
                    }
                }
            }
        }

        // batch_normalization: ONNX/TRTX expect fixed input_operands [input, mean, variance, scale?, bias?]
        // with scale at index 3 and bias at index 4. If only one of scale/bias is provided, insert
        // a default constant so converters always see 5 operands when scale or bias is used.
        if op_type == "batch_normalization" {
            let order = ["input", "mean", "variance", "scale", "bias"];
            let ordered: Vec<u32> = order
                .iter()
                .filter_map(|key| {
                    args.get(*key).and_then(|v| v.as_str()).and_then(|name| name_to_id.get(name).copied())
                })
                .collect();
            if !ordered.is_empty() {
                if ordered.len() == 4 {
                    let mean_id = ordered[1];
                    let mean_shape = operands
                        .get(mean_id as usize)
                        .map(|o| o.descriptor.shape.clone())
                        .unwrap_or_default();
                    let data_type = operands
                        .get(mean_id as usize)
                        .map(|o| o.descriptor.data_type)
                        .unwrap_or(DataType::Float32);
                    let n: usize = mean_shape.iter().product::<u32>().max(1) as usize;
                    let has_scale = args.get("scale").and_then(|v| v.as_str()).is_some();
                    let has_bias = args.get("bias").and_then(|v| v.as_str()).is_some();
                    if has_bias && !has_scale {
                        // Only bias provided: insert default scale (1.0) at index 3, then bias at 4
                        let scale_bytes: Vec<u8> = match data_type {
                            DataType::Float32 => (0..n).flat_map(|_| (1.0f32).to_ne_bytes()).collect(),
                            DataType::Float16 => (0..n).flat_map(|_| half::f16::from_f32(1.0).to_bits().to_ne_bytes()).collect(),
                            _ => (0..n).flat_map(|_| (1.0f32).to_ne_bytes()).collect(),
                        };
                        constant_data.insert(next_id, ConstantData { data: scale_bytes, label: None });
                        operands.push(Operand {
                            kind: OperandKind::Constant,
                            descriptor: OperandDescriptor {
                                data_type,
                                shape: mean_shape.clone(),
                                pending_permutation: Vec::new(),
                            },
                            name: Some("batch_norm_default_scale".to_string()),
                        });
                        input_ids = vec![ordered[0], ordered[1], ordered[2], next_id, ordered[3]];
                        next_id += 1;
                    } else if has_scale && !has_bias {
                        // Only scale provided: insert default bias (0.0) at index 4
                        let bias_bytes: Vec<u8> = match data_type {
                            DataType::Float32 => (0..n).flat_map(|_| (0.0f32).to_ne_bytes()).collect(),
                            DataType::Float16 => (0..n).flat_map(|_| half::f16::from_f32(0.0).to_bits().to_ne_bytes()).collect(),
                            _ => (0..n).flat_map(|_| (0.0f32).to_ne_bytes()).collect(),
                        };
                        constant_data.insert(next_id, ConstantData { data: bias_bytes, label: None });
                        operands.push(Operand {
                            kind: OperandKind::Constant,
                            descriptor: OperandDescriptor {
                                data_type,
                                shape: mean_shape,
                                pending_permutation: Vec::new(),
                            },
                            name: Some("batch_norm_default_bias".to_string()),
                        });
                        input_ids = vec![ordered[0], ordered[1], ordered[2], ordered[3], next_id];
                        next_id += 1;
                    } else {
                        input_ids = ordered;
                    }
                } else {
                    input_ids = ordered;
                }
            }
        }
        // instance_normalization: ONNX expects [input, scale?, bias?]
        if op_type == "instance_normalization" {
            let order = ["input", "scale", "bias"];
            let ordered: Vec<u32> = order
                .iter()
                .filter_map(|key| {
                    args.get(*key).and_then(|v| v.as_str()).and_then(|name| name_to_id.get(name).copied())
                })
                .collect();
            if !ordered.is_empty() {
                input_ids = ordered;
            }
        }
        // conv2d / conv_transpose2d: ONNX expects [input, filter, bias?]
        if op_type == "conv2d" || op_type == "conv_transpose2d" {
            let order = ["input", "filter", "bias"];
            let ordered: Vec<u32> = order
                .iter()
                .filter_map(|key| {
                    args.get(*key).and_then(|v| v.as_str()).and_then(|name| name_to_id.get(name).copied())
                })
                .collect();
            if !ordered.is_empty() {
                input_ids = ordered;
            }
        }

        let out_names = output_names(op);
        let first_input_id = input_ids.first().copied().unwrap_or(0);
        let first_input_dtype = operands.get(first_input_id as usize).map(|o| o.descriptor.data_type).unwrap_or(DataType::Float32);
        let mut output_ids = Vec::new();
        for out_name in &out_names {
            let expected: Option<&WptTensorSpec> = graph.expected_outputs.get(out_name);
            let out_shape: Vec<u32> = expected.map(|s: &WptTensorSpec| s.shape().to_vec()).unwrap_or_default();
            let out_dtype = expected.map(|s: &WptTensorSpec| wpt_data_type(s.data_type())).unwrap_or(first_input_dtype);
            let desc = OperandDescriptor {
                data_type: out_dtype,
                shape: out_shape,
                pending_permutation: Vec::new(),
            };
            operands.push(Operand {
                kind: OperandKind::Output,
                descriptor: desc,
                name: Some(out_name.clone()),
            });
            let id = next_id;
            next_id += 1;
            name_to_id.insert(out_name.clone(), id);
            output_ids.push(id);
        }

        let output_operand = if output_ids.len() == 1 {
            Some(output_ids[0])
        } else {
            None
        };
        let output_operands = if output_ids.len() > 1 {
            output_ids
        } else {
            Vec::new()
        };

        // ONNX converter expects camelCase for normalization ops
        let op_type_for_graph = match op_type.as_str() {
            "batch_normalization" => "batchNormalization",
            "instance_normalization" => "instanceNormalization",
            "layer_normalization" => "layerNormalization",
            _ => op_type.as_str(),
        };
        operations.push(Operation {
            op_type: op_type_for_graph.to_string(),
            input_operands: input_ids,
            output_operand,
            output_operands,
            attributes: serde_json::Value::Object(attributes),
            label: Some(format!("{}_op", op.name)),
        });
    }

    let output_operands: Vec<u32> = graph
        .expected_outputs
        .keys()
        .filter_map(|name| name_to_id.get(name).copied())
        .collect();

    let graph_info = GraphInfo {
        operands,
        input_operands: input_operand_ids.clone(),
        output_operands,
        operations,
        constant_operand_ids_to_handles: constant_data,
        id_to_constant_tensor_operand_map: HashMap::new(),
        quantized: false,
    };

    let input_names: Vec<String> = graph
        .inputs
        .iter()
        .filter(|(_, spec): &(&String, &WptTensorSpec)| !spec.constant)
        .map(|(name, _): (&String, &WptTensorSpec)| name.clone())
        .collect();

    Ok((graph_info, input_names))
}

/// Build ONNX input list from WPT graph (non-constant inputs only).
#[cfg(feature = "onnx-runtime")]
pub fn wpt_graph_to_onnx_inputs(
    graph: &WptGraph,
    input_names: &[String],
) -> Result<Vec<OnnxInput>, String> {
    let mut inputs = Vec::new();
    for name in input_names {
        let spec = graph.inputs.get(name).ok_or_else(|| format!("input {} not found", name))?;
        let shape: Vec<usize> = spec.shape().iter().map(|&d| d as usize).collect();
        let dtype = spec.data_type();
        let n: usize = shape.iter().product();
        let n = n.max(1);

        let arr_opt: Option<&Vec<serde_json::Value>> = spec.data.as_array();
        let data = match dtype {
            "float32" => {
                let mut buf = vec![0.0f32; n];
                if let Some(arr) = arr_opt {
                    for (i, v) in arr.iter().enumerate().take(n) {
                        buf[i] = parse_float_for_tensor(v).unwrap_or(buf[i]);
                    }
                } else if let Some(f) = parse_float_for_tensor(&spec.data) {
                    buf.fill(f);
                }
                TensorData::Float32(buf)
            }
            "float16" => {
                let mut buf = vec![0u16; n];
                if let Some(arr) = arr_opt {
                    for (i, v) in arr.iter().enumerate().take(n) {
                        let f = parse_float_for_tensor(v).unwrap_or(0.0);
                        buf[i] = half::f16::from_f32(f).to_bits();
                    }
                } else if let Some(f) = parse_float_for_tensor(&spec.data) {
                    buf.fill(half::f16::from_f32(f).to_bits());
                }
                TensorData::Float16(buf)
            }
            "int32" => {
                let mut buf = vec![0i32; n];
                if let Some(arr) = arr_opt {
                    for (i, v) in arr.iter().enumerate().take(n) {
                        if let Some(x) = v.as_i64().or_else(|| parse_number(v).map(|f| f as i64)) {
                            buf[i] = x as i32;
                        }
                    }
                } else if let Some(x) = spec.data.as_i64().or_else(|| parse_number(&spec.data).map(|f| f as i64)) {
                    buf.fill(x as i32);
                }
                TensorData::Int32(buf)
            }
            "uint32" => {
                let mut buf = vec![0u32; n];
                if let Some(arr) = arr_opt {
                    for (i, v) in arr.iter().enumerate().take(n) {
                        if let Some(x) = v.as_u64().or_else(|| parse_number(v).map(|f| f as u64)) {
                            buf[i] = x as u32;
                        }
                    }
                } else if let Some(x) = spec.data.as_u64().or_else(|| parse_number(&spec.data).map(|f| f as u64)) {
                    buf.fill(x as u32);
                }
                TensorData::Uint32(buf)
            }
            "int8" => {
                let mut buf = vec![0i8; n];
                if let Some(arr) = arr_opt {
                    for (i, v) in arr.iter().enumerate().take(n) {
                        if let Some(x) = v.as_i64().or_else(|| parse_number(v).map(|f| f as i64)) {
                            buf[i] = x as i8;
                        }
                    }
                } else if let Some(x) = spec.data.as_i64().or_else(|| parse_number(&spec.data).map(|f| f as i64)) {
                    buf.fill(x as i8);
                }
                TensorData::Int8(buf)
            }
            "uint8" => {
                let mut buf = vec![0u8; n];
                if let Some(arr) = arr_opt {
                    for (i, v) in arr.iter().enumerate().take(n) {
                        if let Some(x) = v.as_u64().or_else(|| parse_number(v).map(|f| f as u64)) {
                            buf[i] = x as u8;
                        }
                    }
                } else if let Some(x) = spec.data.as_u64().or_else(|| parse_number(&spec.data).map(|f| f as u64)) {
                    buf.fill(x as u8);
                }
                TensorData::Uint8(buf)
            }
            "int64" => {
                let mut buf = vec![0i64; n];
                if let Some(arr) = arr_opt {
                    for (i, v) in arr.iter().enumerate().take(n) {
                        if let Some(x) = v.as_i64()
                            .or_else(|| parse_number(v).map(|f| f as i64))
                            .or_else(|| v.as_str().and_then(|s| s.trim_end_matches('n').parse::<i64>().ok()))
                        {
                            buf[i] = x;
                        }
                    }
                } else if let Some(x) = spec.data.as_i64()
                    .or_else(|| parse_number(&spec.data).map(|f| f as i64))
                    .or_else(|| spec.data.as_str().and_then(|s| s.trim_end_matches('n').parse::<i64>().ok()))
                {
                    buf.fill(x);
                }
                TensorData::Int64(buf)
            }
            "uint64" => {
                let mut buf = vec![0u64; n];
                if let Some(arr) = arr_opt {
                    for (i, v) in arr.iter().enumerate().take(n) {
                        if let Some(x) = v.as_u64()
                            .or_else(|| parse_number(v).map(|f| f as u64))
                            .or_else(|| v.as_str().and_then(|s| s.trim_end_matches('n').parse::<u64>().ok()))
                        {
                            buf[i] = x;
                        }
                    }
                } else if let Some(x) = spec.data.as_u64()
                    .or_else(|| parse_number(&spec.data).map(|f| f as u64))
                    .or_else(|| spec.data.as_str().and_then(|s| s.trim_end_matches('n').parse::<u64>().ok()))
                {
                    buf.fill(x);
                }
                TensorData::Uint64(buf)
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
                TensorData::Float32(buf)
            }
        };
        inputs.push(OnnxInput {
            name: name.clone(),
            shape,
            data,
        });
    }
    Ok(inputs)
}

/// Encode f32 values as little-endian float16 bytes for TensorRT kHALF.
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
fn f32_to_f16_bytes(slice: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(slice.len() * 2);
    for &f in slice {
        out.extend_from_slice(&half::f16::from_f32(f).to_bits().to_le_bytes());
    }
    out
}

/// Parse one JSON value to i64 for integer tensor inputs.
/// For bigint strings (e.g. "-9223372036854775807n"), parse as i64 directly to avoid f64 precision loss.
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
fn parse_int_for_tensor(v: &serde_json::Value) -> Option<i64> {
    v.as_i64()
        .or_else(|| v.as_str().and_then(|s| s.trim_end_matches('n').parse::<i64>().ok()))
        .or_else(|| parse_number(v).map(|f| f as i64))
        .or_else(|| v.as_u64().map(|u| u as i64))
}

/// Build TensorRT input list from WPT graph. Caller encodes values as bytes per dtype (float32, float16, int8, int32, etc.).
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
pub fn wpt_graph_to_trtx_inputs(
    graph: &WptGraph,
    input_names: &[String],
) -> Result<Vec<rustnn::TrtxInput>, String> {
    let mut inputs = Vec::new();
    for name in input_names {
        let spec = graph.inputs.get(name).ok_or_else(|| format!("input {} not found", name))?;
        let dtype = spec.data_type();
        let shape: Vec<usize> = spec.shape().iter().map(|&d| d as usize).collect();
        let n: usize = shape.iter().product();
        let n = n.max(1);
        let arr_opt: Option<&Vec<serde_json::Value>> = spec.data.as_array();

        let data: Vec<u8> = match dtype {
            "float32" => {
                let mut buf = vec![0.0f32; n];
                if let Some(arr) = arr_opt {
                    for (i, v) in arr.iter().enumerate().take(n) {
                        buf[i] = parse_float_for_tensor(v).unwrap_or(buf[i]);
                    }
                } else if let Some(f) = parse_float_for_tensor(&spec.data) {
                    buf.fill(f);
                }
                buf.iter().flat_map(|f| f.to_le_bytes()).collect()
            }
            "float16" => {
                let mut buf = vec![0.0f32; n];
                if let Some(arr) = arr_opt {
                    for (i, v) in arr.iter().enumerate().take(n) {
                        buf[i] = parse_float_for_tensor(v).unwrap_or(buf[i]);
                    }
                } else if let Some(f) = parse_float_for_tensor(&spec.data) {
                    buf.fill(f);
                }
                f32_to_f16_bytes(&buf)
            }
            "int8" => {
                let mut buf = vec![0i8; n];
                if let Some(arr) = arr_opt {
                    for (i, v) in arr.iter().enumerate().take(n) {
                        if let Some(x) = parse_int_for_tensor(v) {
                            buf[i] = x as i8;
                        }
                    }
                } else if let Some(x) = parse_int_for_tensor(&spec.data) {
                    buf.fill(x as i8);
                }
                buf.iter().map(|&x| x as u8).collect()
            }
            "uint8" => {
                let mut buf = vec![0u8; n];
                if let Some(arr) = arr_opt {
                    for (i, v) in arr.iter().enumerate().take(n) {
                        if let Some(x) = v.as_u64().or_else(|| parse_int_for_tensor(v).map(|x| x as u64)) {
                            buf[i] = x as u8;
                        }
                    }
                } else if let Some(x) = spec.data.as_u64().or_else(|| parse_int_for_tensor(&spec.data).map(|x| x as u64)) {
                    buf.fill(x as u8);
                }
                buf
            }
            "int32" => {
                let mut buf = vec![0i32; n];
                if let Some(arr) = arr_opt {
                    for (i, v) in arr.iter().enumerate().take(n) {
                        if let Some(x) = parse_int_for_tensor(v) {
                            buf[i] = x as i32;
                        }
                    }
                } else if let Some(x) = parse_int_for_tensor(&spec.data) {
                    buf.fill(x as i32);
                }
                buf.iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            "uint32" => {
                let mut buf = vec![0u32; n];
                if let Some(arr) = arr_opt {
                    for (i, v) in arr.iter().enumerate().take(n) {
                        if let Some(x) = v.as_u64().or_else(|| parse_int_for_tensor(v).map(|x| x as u64)) {
                            buf[i] = x as u32;
                        }
                    }
                } else if let Some(x) = spec.data.as_u64().or_else(|| parse_int_for_tensor(&spec.data).map(|x| x as u64)) {
                    buf.fill(x as u32);
                }
                buf.iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            "int64" | "uint64" => {
                let mut buf = vec![0i64; n];
                if let Some(arr) = arr_opt {
                    for (i, v) in arr.iter().enumerate().take(n) {
                        if let Some(x) = parse_int_for_tensor(v) {
                            buf[i] = x;
                        }
                    }
                } else if let Some(x) = parse_int_for_tensor(&spec.data) {
                    buf.fill(x);
                }
                buf.iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            _ => {
                let mut buf = vec![0.0f32; n];
                if let Some(arr) = arr_opt {
                    for (i, v) in arr.iter().enumerate().take(n) {
                        if let Some(f) = parse_float_for_tensor(v).or_else(|| v.as_f64().map(|f| f as f32)) {
                            buf[i] = f;
                        }
                    }
                } else if let Some(f) = parse_float_for_tensor(&spec.data).or_else(|| spec.data.as_f64().map(|f| f as f32)) {
                    buf.fill(f);
                }
                buf.iter().flat_map(|f| f.to_le_bytes()).collect()
            }
        };

        inputs.push(rustnn::TrtxInput {
            name: name.clone(),
            data,
        });
    }
    Ok(inputs)
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
    buf
}

/// Expected output as i32 slice (for int32 validation).
pub fn expected_output_to_i32(spec: &WptTensorSpec) -> Vec<i32> {
    let shape = spec.shape();
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let n = n.max(1);
    let mut buf = vec![0i32; n];
    let arr_opt: Option<&Vec<serde_json::Value>> = spec.data.as_array();
    if let Some(arr) = arr_opt {
        for (i, v) in arr.iter().enumerate().take(n) {
            if let Some(x) = v.as_i64()
                .or_else(|| parse_number(v).map(|f| f as i64))
                .or_else(|| v.as_str().and_then(|s| s.trim_end_matches('n').parse::<i64>().ok()))
            {
                buf[i] = x as i32;
            }
        }
    } else if let Some(x) = spec.data.as_i64()
        .or_else(|| parse_number(&spec.data).map(|f| f as i64))
        .or_else(|| spec.data.as_str().and_then(|s| s.trim_end_matches('n').parse::<i64>().ok()))
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
            if let Some(x) = v.as_u64()
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
            if let Some(x) = v.as_u64()
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
            if let Some(x) = v.as_i64()
                .or_else(|| parse_number(v).map(|f| f as i64))
                .or_else(|| v.as_str().and_then(|s| s.trim_end_matches('n').parse::<i64>().ok()))
            {
                buf[i] = x as i8;
            }
        }
    } else if let Some(x) = spec.data.as_i64()
        .or_else(|| parse_number(&spec.data).map(|f| f as i64))
        .or_else(|| spec.data.as_str().and_then(|s| s.trim_end_matches('n').parse::<i64>().ok()))
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
                .or_else(|| v.as_str().and_then(|s| s.trim_end_matches('n').parse::<i64>().ok()))
                .or_else(|| parse_number(v).map(|f| f as i64))
            {
                buf[i] = x;
            }
        }
    } else if let Some(x) = spec
        .data
        .as_i64()
        .or_else(|| spec.data.as_str().and_then(|s| s.trim_end_matches('n').parse::<i64>().ok()))
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
            if let Some(x) = v.as_u64()
                .or_else(|| parse_number(v).map(|f| f as u64))
                .or_else(|| v.as_str().and_then(|s| s.trim_end_matches('n').parse::<u64>().ok()))
            {
                buf[i] = x;
            }
        }
    } else if let Some(x) = spec.data.as_u64()
        .or_else(|| parse_number(&spec.data).map(|f| f as u64))
        .or_else(|| spec.data.as_str().and_then(|s| s.trim_end_matches('n').parse::<u64>().ok()))
    {
        buf.fill(x);
    }
    buf
}

pub fn wpt_data_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("wpt_data").join("conformance")
}

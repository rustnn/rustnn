/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! WebNN operator options as a tagged union.
//!
//! All IDL dictionaries that extend MLOperatorOptions from the
//! [Web Neural Network API](https://www.w3.org/TR/webnn/) are represented
//! as an enum: each variant holds the corresponding options struct.

use serde::{Deserialize, Serialize};

/// Operand reference (graph operand index). Used in option structs for MLOperand fields.
pub type OperandIndex = u32;

// ---------------------------------------------------------------------------
// WebNN IDL: MLDimension (supports dynamic dimensions)
// ---------------------------------------------------------------------------

/// MLDynamicDimension. IDL: `dictionary MLDynamicDimension { required DOMString name; required [EnforceRange] unsigned long maxSize; };`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLDynamicDimension {
    pub name: String,
    pub max_size: u32,
}

/// MLDimension. IDL: `typedef ([EnforceRange] unsigned long or MLDynamicDimension) MLDimension;`
/// In JSON: either a number (static) or an object `{ "name": string, "maxSize": number }` (dynamic).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MLDimension {
    Static(u32),
    Dynamic(MLDynamicDimension),
}

impl MLDimension {
    /// Returns the static value or dynamic maxSize as u32.
    pub fn static_or_max(&self) -> u32 {
        match self {
            MLDimension::Static(n) => *n,
            MLDimension::Dynamic(d) => d.max_size,
        }
    }
}

/// Static size or dynamic `maxSize` for each `MLDimension` (shape hints, CoreML, TRT static paths).
#[inline]
pub fn mldimensions_static_or_max(dims: &[MLDimension]) -> Vec<u32> {
    dims.iter().map(MLDimension::static_or_max).collect()
}

/// Scalar and sequence parameters that belong on the WebNN graph builder operation
/// (method arguments) rather than in the options dictionary, as extracted from interchange JSON.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OperationExtras {
    pub axis: Option<u32>,
    pub to_data_type: Option<String>,
    pub batch_dimensions: Option<u32>,
    pub steps: Option<u32>,
    pub hidden_size: Option<u32>,
    pub beginning_padding: Vec<u32>,
    pub ending_padding: Vec<u32>,
    pub starts: Vec<u32>,
    pub sizes: Vec<MLDimension>,
    pub splits: Vec<u32>,
    pub split_equal_parts: Option<u32>,
    /// `expand()` method argument `newShape` (not part of MLOperatorOptions).
    pub expand_new_shape: Vec<MLDimension>,
    /// `tile()` method argument `repetitions` (not part of MLOperatorOptions).
    pub repetitions: Vec<u32>,
    /// `reshape()` method argument `newShape` (not part of MLOperatorOptions).
    pub reshape_new_shape: Vec<MLDimension>,
}

impl OperationExtras {
    /// Remove operation-level keys from `v` (must be a JSON object) and return their values.
    pub fn extract_and_strip(op_type: &str, v: &mut serde_json::Value) -> Self {
        let mut out = Self::default();
        let Some(obj) = v.as_object_mut() else {
            return out;
        };
        let op = op_type.trim();
        fn remove_u32(
            obj: &mut serde_json::Map<String, serde_json::Value>,
            key: &str,
        ) -> Option<u32> {
            obj.remove(key).and_then(|x| x.as_u64().map(|n| n as u32))
        }
        fn remove_u32_vec(
            obj: &mut serde_json::Map<String, serde_json::Value>,
            key: &str,
        ) -> Vec<u32> {
            obj.remove(key)
                .and_then(|x| serde_json::from_value::<Vec<u32>>(x).ok())
                .unwrap_or_default()
        }
        match op {
            "argMin" | "argMax" => {
                out.axis = remove_u32(obj, "axis");
            }
            "cast" => {
                if let Some(s) = obj
                    .remove("to")
                    .and_then(|x| x.as_str().map(|s| s.to_string()))
                {
                    out.to_data_type = Some(s);
                } else if let Some(s) = obj
                    .remove("dataType")
                    .and_then(|x| x.as_str().map(|s| s.to_string()))
                {
                    out.to_data_type = Some(s);
                }
            }
            "concat" => {
                out.axis = remove_u32(obj, "axis");
            }
            "expand" => {
                let _ = obj.remove("axes");
                if let Some(s) = obj.remove("newShape").or_else(|| obj.remove("new_shape"))
                    && let Ok(parsed) = serde_json::from_value::<Vec<MLDimension>>(s)
                {
                    out.expand_new_shape = parsed;
                }
            }
            "cumulativeSum" => {
                out.axis = remove_u32(obj, "axis");
            }
            "gather" | "gatherElements" => {
                out.batch_dimensions = remove_u32(obj, "batchDimensions")
                    .or_else(|| remove_u32(obj, "batch_dimensions"));
            }
            "gru" => {
                out.steps = remove_u32(obj, "steps");
                out.hidden_size =
                    remove_u32(obj, "hiddenSize").or_else(|| remove_u32(obj, "hidden_size"));
            }
            "gruCell" => {
                out.hidden_size =
                    remove_u32(obj, "hiddenSize").or_else(|| remove_u32(obj, "hidden_size"));
            }
            "instanceNormalization" => {
                // Legacy interchange; not part of MLInstanceNormalizationOptions.
                let _ = obj.remove("hasScale");
                let _ = obj.remove("hasBias");
                let _ = obj.remove("has_scale");
                let _ = obj.remove("has_bias");
            }
            "layerNormalization" => {
                let _ = obj.remove("hasScale");
                let _ = obj.remove("hasBias");
                let _ = obj.remove("has_scale");
                let _ = obj.remove("has_bias");
            }
            "pad" => {
                out.beginning_padding = remove_u32_vec(obj, "beginningPadding");
                if out.beginning_padding.is_empty() {
                    out.beginning_padding = remove_u32_vec(obj, "beginning_padding");
                }
                out.ending_padding = remove_u32_vec(obj, "endingPadding");
                if out.ending_padding.is_empty() {
                    out.ending_padding = remove_u32_vec(obj, "ending_padding");
                }
            }
            "softmax" => {
                out.axis = remove_u32(obj, "axis");
            }
            "slice" => {
                out.starts = remove_u32_vec(obj, "starts");
                if let Some(s) = obj.remove("sizes")
                    && let Ok(parsed) = serde_json::from_value::<Vec<MLDimension>>(s)
                {
                    out.sizes = parsed;
                }
            }
            "split" => {
                if let Some(sv) = obj.remove("splits") {
                    match sv {
                        serde_json::Value::Number(n) => {
                            out.split_equal_parts = n.as_u64().map(|u| u as u32);
                        }
                        serde_json::Value::Array(_) => {
                            if let Ok(parsed) = serde_json::from_value::<Vec<u32>>(sv) {
                                out.splits = parsed;
                            }
                        }
                        _ => {}
                    }
                }
            }
            "tile" => {
                out.repetitions = remove_u32_vec(obj, "repetitions");
            }
            "reshape" => {
                if let Some(s) = obj.remove("newShape").or_else(|| obj.remove("new_shape"))
                    && let Ok(parsed) = serde_json::from_value::<Vec<MLDimension>>(s)
                {
                    out.reshape_new_shape = parsed;
                }
            }
            _ => {}
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Base: MLOperatorOptions
// ---------------------------------------------------------------------------

/// MLOperatorOptions. Base type for all operator options (label only).
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mloperatoroptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLOperatorOptions {
    #[serde(default)]
    pub label: String,
}

// ---------------------------------------------------------------------------
// Dictionaries extending MLOperatorOptions (spec order)
// ---------------------------------------------------------------------------

/// MLArgMinMaxOptions. argMin / argMax (axis is a builder method parameter, not in this dictionary).
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlargminmaxoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLArgMinMaxOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub keep_dimensions: bool,
    #[serde(default)]
    pub output_data_type: String, // MLOperandDataType, e.g. "int32", "int64"
}

fn default_batch_norm_axis() -> u32 {
    1
}

fn default_batch_norm_epsilon() -> f64 {
    1e-5
}

/// MLBatchNormalizationOptions. batchNormalization.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlbatchnormalizationoptions>
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLBatchNormalizationOptions {
    #[serde(default)]
    pub label: String,
    pub scale: Option<OperandIndex>,
    pub bias: Option<OperandIndex>,
    #[serde(default = "default_batch_norm_axis")]
    pub axis: u32,
    #[serde(default = "default_batch_norm_epsilon")]
    pub epsilon: f64,
}

impl Default for MLBatchNormalizationOptions {
    fn default() -> Self {
        Self {
            label: String::new(),
            scale: None,
            bias: None,
            axis: default_batch_norm_axis(),
            epsilon: default_batch_norm_epsilon(),
        }
    }
}

/// MLClampOptions. clamp.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlclampoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLClampOptions {
    #[serde(default)]
    pub label: String,
    // TODO MTAX MLNumber is an union of any floating point or integral type
    pub min_value: Option<serde_json::Value>, // MLNumber
    pub max_value: Option<serde_json::Value>, // MLNumber
}

fn default_conv_groups() -> u32 {
    1
}

/// MLConv2dOptions. conv2d.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlconv2doptions>
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLConv2dOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub padding: Vec<u32>,
    #[serde(default)]
    pub strides: Vec<u32>,
    #[serde(default)]
    pub dilations: Vec<u32>,
    #[serde(default = "default_conv_groups")]
    pub groups: u32,
    #[serde(default)]
    pub input_layout: String, // "nchw" | "nhwc"
    #[serde(default)]
    pub filter_layout: String, // "oihw" | "hwio" | "ohwi" | "ihwo"
    pub bias: Option<OperandIndex>,
}

impl Default for MLConv2dOptions {
    fn default() -> Self {
        Self {
            label: String::new(),
            padding: Vec::new(),
            strides: Vec::new(),
            dilations: Vec::new(),
            groups: default_conv_groups(),
            input_layout: String::new(),
            filter_layout: String::new(),
            bias: None,
        }
    }
}

/// MLConvTranspose2dOptions. convTranspose2d.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLConvTranspose2dOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub padding: Vec<u32>,
    #[serde(default)]
    pub strides: Vec<u32>,
    #[serde(default)]
    pub dilations: Vec<u32>,
    #[serde(default)]
    pub output_padding: Vec<u32>,
    pub output_sizes: Option<Vec<u32>>,
    #[serde(default = "default_conv_groups")]
    pub groups: u32,
    #[serde(default)]
    pub input_layout: String,
    #[serde(default)]
    pub filter_layout: String, // "iohw" | "hwoi" | "ohwi"
    pub bias: Option<OperandIndex>,
}

impl Default for MLConvTranspose2dOptions {
    fn default() -> Self {
        Self {
            label: String::new(),
            padding: Vec::new(),
            strides: Vec::new(),
            dilations: Vec::new(),
            output_padding: Vec::new(),
            output_sizes: None,
            groups: default_conv_groups(),
            input_layout: String::new(),
            filter_layout: String::new(),
            bias: None,
        }
    }
}

/// MLConstantOptions. constant (interchange: init, data, dataType, shape).
///
/// Not an IDL dictionary; closest normative API is [`MLGraphBuilder`](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder) (`constant()` methods).
// TODO MTAX non-existing struct. defer removal for now since it's not like any other operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLConstantOptions {
    #[serde(default)]
    pub label: String,
    pub init: Option<String>,
    pub data: Option<String>, // base64
    #[serde(default, rename = "dataType")]
    pub data_type: String,
    #[serde(default)]
    pub shape: Vec<u32>,
}

/// MLCumulativeSumOptions. cumulativeSum (axis is a builder method parameter).
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlcumulativesumoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLCumulativeSumOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub exclusive: bool,
    #[serde(default)]
    pub reversed: bool,
}

fn default_elu_alpha() -> f64 {
    1.0
}

/// MLEluOptions. elu.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mleluoptions>
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLEluOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default = "default_elu_alpha")]
    pub alpha: f64,
}

impl Default for MLEluOptions {
    fn default() -> Self {
        Self {
            label: String::new(),
            alpha: default_elu_alpha(),
        }
    }
}

/// MLGatherOptions. gather / gatherElements (batchDimensions is a gatherElements parameter in WebNN).
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlgatheroptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLGatherOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub axis: u32,
}

fn default_gemm_alpha() -> f64 {
    1.0
}

fn default_gemm_beta() -> f64 {
    1.0
}

/// MLGemmOptions. gemm.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlgemmoptions>
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLGemmOptions {
    #[serde(default)]
    pub label: String,
    pub c: Option<OperandIndex>,
    #[serde(default = "default_gemm_alpha")]
    pub alpha: f64,
    #[serde(default = "default_gemm_beta")]
    pub beta: f64,
    #[serde(default)]
    pub a_transpose: bool,
    #[serde(default)]
    pub b_transpose: bool,
}

impl Default for MLGemmOptions {
    fn default() -> Self {
        Self {
            label: String::new(),
            c: None,
            alpha: default_gemm_alpha(),
            beta: default_gemm_beta(),
            a_transpose: false,
            b_transpose: false,
        }
    }
}

/// MLGruOptions. gru.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlgruoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLGruOptions {
    #[serde(default)]
    pub label: String,
    pub bias: Option<OperandIndex>,
    pub recurrent_bias: Option<OperandIndex>,
    pub initial_hidden_state: Option<OperandIndex>,
    #[serde(default)]
    pub reset_after: bool,
    #[serde(default)]
    pub return_sequence: bool,
    #[serde(default)]
    pub direction: String, // "forward" | "backward" | "both"
    #[serde(default)]
    pub layout: String, // "zrn" | "rzn"
    pub activations: Option<Vec<String>>, // MLRecurrentNetworkActivation
}

/// MLGruCellOptions. gruCell.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlgrucelloptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLGruCellOptions {
    #[serde(default)]
    pub label: String,
    pub bias: Option<OperandIndex>,
    pub recurrent_bias: Option<OperandIndex>,
    #[serde(default)]
    pub reset_after: bool,
    #[serde(default)]
    pub layout: String,
    pub activations: Option<Vec<String>>,
}

fn default_hard_sigmoid_alpha() -> f64 {
    0.2
}

fn default_hard_sigmoid_beta() -> f64 {
    0.5
}

/// MLHardSigmoidOptions. hardSigmoid.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlhardsigmoidoptions>
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLHardSigmoidOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default = "default_hard_sigmoid_alpha")]
    pub alpha: f64,
    #[serde(default = "default_hard_sigmoid_beta")]
    pub beta: f64,
}

impl Default for MLHardSigmoidOptions {
    fn default() -> Self {
        Self {
            label: String::new(),
            alpha: default_hard_sigmoid_alpha(),
            beta: default_hard_sigmoid_beta(),
        }
    }
}

fn default_instance_norm_epsilon() -> f64 {
    1e-5
}

/// MLInstanceNormalizationOptions. instanceNormalization.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlinstancenormalizationoptions>
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLInstanceNormalizationOptions {
    #[serde(default)]
    pub label: String,
    pub scale: Option<OperandIndex>,
    pub bias: Option<OperandIndex>,
    #[serde(default = "default_instance_norm_epsilon")]
    pub epsilon: f64,
    #[serde(default)]
    pub layout: String,
}

impl Default for MLInstanceNormalizationOptions {
    fn default() -> Self {
        Self {
            label: String::new(),
            scale: None,
            bias: None,
            epsilon: default_instance_norm_epsilon(),
            layout: String::new(), // TODO TMAX the default is "nchw"
        }
    }
}

fn default_layer_norm_epsilon() -> f64 {
    1e-5
}

/// MLLayerNormalizationOptions. layerNormalization.
/// `axes`: None = key omitted (spec default [1..rank)); Some(v) = use v (Some(vec![]) = reduce over no axes).
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mllayernormalizationoptions>
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLLayerNormalizationOptions {
    #[serde(default)]
    pub label: String,
    pub scale: Option<OperandIndex>,
    pub bias: Option<OperandIndex>,
    pub axes: Option<Vec<u32>>,
    #[serde(default = "default_layer_norm_epsilon")]
    pub epsilon: f64,
}

impl Default for MLLayerNormalizationOptions {
    fn default() -> Self {
        Self {
            label: String::new(),
            scale: None,
            bias: None,
            axes: None,
            epsilon: default_layer_norm_epsilon(),
        }
    }
}

fn default_leaky_relu_alpha() -> f64 {
    0.01
}

/// MLLeakyReluOptions. leakyRelu.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlleakyreluoptions>
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLLeakyReluOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default = "default_leaky_relu_alpha")]
    pub alpha: f64,
}

impl Default for MLLeakyReluOptions {
    fn default() -> Self {
        Self {
            label: String::new(),
            alpha: default_leaky_relu_alpha(),
        }
    }
}

fn default_linear_alpha() -> f64 {
    1.0
}

fn default_linear_beta() -> f64 {
    0.0
}

/// MLLinearOptions. linear.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mllinearoptions>
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLLinearOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default = "default_linear_alpha")]
    pub alpha: f64,
    #[serde(default = "default_linear_beta")]
    pub beta: f64,
}

impl Default for MLLinearOptions {
    fn default() -> Self {
        Self {
            label: String::new(),
            alpha: default_linear_alpha(),
            beta: default_linear_beta(),
        }
    }
}

/// MLLstmOptions. lstm.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mllstmoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLLstmOptions {
    #[serde(default)]
    pub label: String,
    pub bias: Option<OperandIndex>,
    pub recurrent_bias: Option<OperandIndex>,
    pub peephole_weight: Option<OperandIndex>,
    pub initial_hidden_state: Option<OperandIndex>,
    pub initial_cell_state: Option<OperandIndex>,
    #[serde(default)]
    pub return_sequence: bool,
    #[serde(default)]
    pub direction: String,
    #[serde(default)]
    pub layout: String, // "iofg" | "ifgo"
    pub activations: Option<Vec<String>>,
}

/// MLLstmCellOptions. lstmCell.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mllstmcelloptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLLstmCellOptions {
    #[serde(default)]
    pub label: String,
    pub bias: Option<OperandIndex>,
    pub recurrent_bias: Option<OperandIndex>,
    pub peephole_weight: Option<OperandIndex>,
    // TODO TMAX verify default
    #[serde(default)]
    pub layout: String,
    pub activations: Option<Vec<String>>,
}

/// MLPadOptions. pad (beginning/ending padding are builder method parameters).
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlpadoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLPadOptions {
    #[serde(default)]
    pub label: String,
    // TODO MTAX mode is an enum of type MLPaddingMode
    #[serde(default)]
    pub mode: String, // "constant" | "edge" | "reflection"
    pub value: Option<serde_json::Value>, // MLNumber
}

/// MLPool2dOptions. averagePool2d / l2Pool2d / maxPool2d.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlpool2doptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLPool2dOptions {
    #[serde(default)]
    pub label: String,
    pub window_dimensions: Option<Vec<u32>>,
    // TODO MTAX check default value
    #[serde(default)]
    pub padding: Vec<u32>,
    #[serde(default)]
    pub strides: Vec<u32>,
    #[serde(default)]
    pub dilations: Vec<u32>,
    // TODO MTAX layout is enum MLInputOperandLayout
    #[serde(default)]
    pub layout: String,
    // TODO MTAX enum MLRoundingType
    #[serde(default)]
    pub output_shape_rounding: String,
    pub output_sizes: Option<Vec<u32>>,
}

/// MLReduceOptions. reduceL1, reduceL2, reduceLogSum, etc.
/// `axes`: None = key omitted (spec default: all axes); Some(v) = use v (Some(vec![]) = reduce over no axes).
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlreduceoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLReduceOptions {
    #[serde(default)]
    pub label: String,
    pub axes: Option<Vec<u32>>,
    #[serde(default)]
    pub keep_dimensions: bool,
}

/// MLResample2dOptions. resample2d.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlresample2doptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLResample2dOptions {
    #[serde(default)]
    pub label: String,
    // TODO MTAX enum MLInterpolationMode
    #[serde(default)]
    pub mode: String, // "nearest-neighbor" | "linear"
    #[serde(default)]
    pub scales: Vec<f32>,
    #[serde(default)]
    pub sizes: Option<Vec<u32>>,

    #[serde(default)]
    pub axes: Vec<u32>,
}

/// MLReverseOptions. reverse.
/// axes: omitted => reverse all dimensions; present and [] => reverse none (identity); present and [..] => reverse those axes.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlreverseoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLReverseOptions {
    #[serde(default)]
    pub label: String,
    /// None = not present in JSON => reverse all. Some([]) => axes: [] => identity. Some([..]) => reverse those axes.
    pub axes: Option<Vec<u32>>,
}

/// MLScatterOptions. scatterElements / scatterND.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlscatteroptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLScatterOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub axis: u32,
}

/// MLSliceOptions. slice (starts and sizes are builder method parameters).
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlsliceoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLSliceOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub strides: Vec<u32>,
}

/// MLSplitOptions. split (splits is a builder method parameter).
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlsplitoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLSplitOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub axis: u32,
}

/// MLTransposeOptions. transpose.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mltransposeoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLTransposeOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub permutation: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Operation Emulation (squeeze, unsqueeze, flatten)
// These ops are not part of the official WebNN API; they are defined in
// § 11 Operation Emulation and can be implemented via reshape().
// ---------------------------------------------------------------------------

// TODO TMAX remove the unofficial ops!

/// MLSqueezeOptions. squeeze (emulation-only; not in WebNN IDL).
///
/// WebNN emulation: <https://www.w3.org/TR/webnn/#squeeze>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLSqueezeOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub axes: Vec<u32>,
}

/// MLUnsqueezeOptions. unsqueeze (emulation-only; not in WebNN IDL).
///
/// WebNN emulation: <https://www.w3.org/TR/webnn/#unsqueeze>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLUnsqueezeOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub axes: Vec<u32>,
}

/// MLTriangularOptions. triangular.
/// WebNN: when "upper" is not present, default is true (keep upper triangular).
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mltriangularoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLTriangularOptions {
    #[serde(default)]
    pub label: String,
    pub upper: Option<bool>,
    #[serde(default)]
    pub diagonal: i32,
}

// ---------------------------------------------------------------------------
// Tagged union: all option types
// ---------------------------------------------------------------------------

/// Tagged union of all ML*Options dictionaries that extend MLOperatorOptions.
///
/// Each variant holds the corresponding options struct from the
/// [WebNN specification](https://www.w3.org/TR/webnn/).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum OperatorOptions {
    /// MLOperatorOptions (base; label only).
    Operator(MLOperatorOptions),

    /// MLArgMinMaxOptions.
    ArgMinMax(MLArgMinMaxOptions),

    /// MLBatchNormalizationOptions.
    BatchNormalization(MLBatchNormalizationOptions),

    /// MLClampOptions.
    Clamp(MLClampOptions),

    /// MLConstantOptions.
    Constant(MLConstantOptions),

    /// MLConv2dOptions.
    Conv2d(MLConv2dOptions),

    /// MLConvTranspose2dOptions.
    ConvTranspose2d(MLConvTranspose2dOptions),

    /// MLCumulativeSumOptions.
    CumulativeSum(MLCumulativeSumOptions),

    /// MLEluOptions.
    Elu(MLEluOptions),

    /// MLGatherOptions.
    Gather(MLGatherOptions),

    /// MLGemmOptions.
    Gemm(MLGemmOptions),

    /// MLGruOptions.
    Gru(MLGruOptions),

    /// MLGruCellOptions.
    GruCell(MLGruCellOptions),

    /// MLHardSigmoidOptions.
    HardSigmoid(MLHardSigmoidOptions),

    /// MLInstanceNormalizationOptions.
    InstanceNormalization(MLInstanceNormalizationOptions),

    /// MLLayerNormalizationOptions.
    LayerNormalization(MLLayerNormalizationOptions),

    /// MLLeakyReluOptions.
    LeakyRelu(MLLeakyReluOptions),

    /// MLLinearOptions.
    Linear(MLLinearOptions),

    /// MLLstmOptions.
    Lstm(MLLstmOptions),

    /// MLLstmCellOptions.
    LstmCell(MLLstmCellOptions),

    /// MLPadOptions.
    Pad(MLPadOptions),

    /// MLPool2dOptions.
    Pool2d(MLPool2dOptions),

    /// MLReduceOptions.
    Reduce(MLReduceOptions),

    /// MLResample2dOptions.
    Resample2d(MLResample2dOptions),

    /// MLReverseOptions.
    Reverse(MLReverseOptions),

    /// MLScatterOptions.
    ScatterElements(MLScatterOptions),

    /// MLSliceOptions.
    Slice(MLSliceOptions),

    /// MLSplitOptions.
    Split(MLSplitOptions),

    /// MLTransposeOptions.
    Transpose(MLTransposeOptions),

    // Operation Emulation (not part of official WebNN API; § 11).
    /// MLSqueezeOptions. squeeze.
    Squeeze(MLSqueezeOptions),
    /// MLUnsqueezeOptions. unsqueeze.
    Unsqueeze(MLUnsqueezeOptions),

    /// MLTriangularOptions.
    Triangular(MLTriangularOptions),
}

impl Default for OperatorOptions {
    fn default() -> Self {
        OperatorOptions::Operator(MLOperatorOptions::default())
    }
}

impl OperatorOptions {
    pub fn label(&self) -> &str {
        match self {
            OperatorOptions::Operator(opt) => &opt.label,
            OperatorOptions::ArgMinMax(opt) => &opt.label,
            OperatorOptions::BatchNormalization(opt) => &opt.label,
            OperatorOptions::Clamp(opt) => &opt.label,
            OperatorOptions::Constant(opt) => &opt.label,
            OperatorOptions::Conv2d(opt) => &opt.label,
            OperatorOptions::ConvTranspose2d(opt) => &opt.label,
            OperatorOptions::CumulativeSum(opt) => &opt.label,
            OperatorOptions::Elu(opt) => &opt.label,
            OperatorOptions::Gather(opt) => &opt.label,
            OperatorOptions::Gemm(opt) => &opt.label,
            OperatorOptions::Gru(opt) => &opt.label,
            OperatorOptions::GruCell(opt) => &opt.label,
            OperatorOptions::HardSigmoid(opt) => &opt.label,
            OperatorOptions::InstanceNormalization(opt) => &opt.label,
            OperatorOptions::LayerNormalization(opt) => &opt.label,
            OperatorOptions::LeakyRelu(opt) => &opt.label,
            OperatorOptions::Linear(opt) => &opt.label,
            OperatorOptions::Lstm(opt) => &opt.label,
            OperatorOptions::LstmCell(opt) => &opt.label,
            OperatorOptions::Pad(opt) => &opt.label,
            OperatorOptions::Pool2d(opt) => &opt.label,
            OperatorOptions::Reduce(opt) => &opt.label,
            OperatorOptions::Resample2d(opt) => &opt.label,
            OperatorOptions::Reverse(opt) => &opt.label,
            OperatorOptions::ScatterElements(opt) => &opt.label,
            OperatorOptions::Slice(opt) => &opt.label,
            OperatorOptions::Split(opt) => &opt.label,
            OperatorOptions::Transpose(opt) => &opt.label,
            OperatorOptions::Squeeze(opt) => &opt.label,
            OperatorOptions::Unsqueeze(opt) => &opt.label,
            OperatorOptions::Triangular(opt) => &opt.label,
        }
    }
    /// Parse attributes from JSON using the operation type to select the options variant.
    /// Returns `None` if `value` is null or not an object; otherwise tries to deserialize
    /// into the variant for `op_type`, falling back to `Operator(MLOperatorOptions::default())`.
    pub fn from_json_with_op_type(op_type: &str, value: &serde_json::Value) -> Option<Self> {
        let _obj = value.as_object()?;
        let normalized = op_type.trim();
        // Try op-type-specific deserialization first (even for empty object, so slice/split with
        // default options get the right variant).
        let try_from = |v: &serde_json::Value| -> Option<OperatorOptions> {
            macro_rules! try_opt {
                ($t:ty, $variant:ident) => {
                    if let Ok(opts) = serde_json::from_value::<$t>(v.clone()) {
                        return Some(OperatorOptions::$variant(opts));
                    }
                };
            }
            match normalized {
                "argMin" | "argMax" => try_opt!(MLArgMinMaxOptions, ArgMinMax),
                "batchNormalization" => try_opt!(MLBatchNormalizationOptions, BatchNormalization),
                "cast" => try_opt!(MLOperatorOptions, Operator),
                "clamp" => try_opt!(MLClampOptions, Clamp),
                "conv2d" => try_opt!(MLConv2dOptions, Conv2d),
                "convTranspose2d" => try_opt!(MLConvTranspose2dOptions, ConvTranspose2d),
                "concat" => try_opt!(MLOperatorOptions, Operator),
                "constant" => try_opt!(MLConstantOptions, Constant),
                "cumulativeSum" => try_opt!(MLCumulativeSumOptions, CumulativeSum),
                "expand" => try_opt!(MLOperatorOptions, Operator),
                "elu" => try_opt!(MLEluOptions, Elu),
                "gather" | "gatherElements" => try_opt!(MLGatherOptions, Gather),
                "gemm" => try_opt!(MLGemmOptions, Gemm),
                "gru" => try_opt!(MLGruOptions, Gru),
                "gruCell" => try_opt!(MLGruCellOptions, GruCell),
                "hardSigmoid" => try_opt!(MLHardSigmoidOptions, HardSigmoid),
                "hardSwish" => try_opt!(MLOperatorOptions, Operator),
                "instanceNormalization" => {
                    try_opt!(MLInstanceNormalizationOptions, InstanceNormalization)
                }
                "layerNormalization" => try_opt!(MLLayerNormalizationOptions, LayerNormalization),
                "leakyRelu" => try_opt!(MLLeakyReluOptions, LeakyRelu),
                "linear" => try_opt!(MLLinearOptions, Linear),
                "lstm" => try_opt!(MLLstmOptions, Lstm),
                "lstmCell" => try_opt!(MLLstmCellOptions, LstmCell),
                "pad" => try_opt!(MLPadOptions, Pad),
                "averagePool2d" | "maxPool2d" | "l2Pool2d" | "globalAveragePool"
                | "globalMaxPool" => try_opt!(MLPool2dOptions, Pool2d),
                "reduceSum" | "reduceMean" | "reduceMax" | "reduceMin" | "reduceProduct"
                | "reduceL1" | "reduceL2" | "reduceLogSum" | "reduceLogSumExp"
                | "reduceSumSquare" => {
                    try_opt!(MLReduceOptions, Reduce)
                }
                "reshape" => try_opt!(MLOperatorOptions, Operator),
                "resample2d" => try_opt!(MLResample2dOptions, Resample2d),
                "reverse" => try_opt!(MLReverseOptions, Reverse),
                "scatterElements" => try_opt!(MLScatterOptions, ScatterElements),
                "softmax" => try_opt!(MLOperatorOptions, Operator),
                "slice" => try_opt!(MLSliceOptions, Slice),
                "split" => try_opt!(MLSplitOptions, Split),
                "transpose" => try_opt!(MLTransposeOptions, Transpose),
                "squeeze" => try_opt!(MLSqueezeOptions, Squeeze),
                "unsqueeze" => try_opt!(MLUnsqueezeOptions, Unsqueeze),
                "tile" => try_opt!(MLOperatorOptions, Operator),
                "triangular" => try_opt!(MLTriangularOptions, Triangular),
                _ => {}
            }
            if let Ok(opts) = serde_json::from_value::<MLOperatorOptions>(v.clone()) {
                return Some(OperatorOptions::Operator(opts));
            }
            None
        };
        try_from(value).or_else(|| Some(OperatorOptions::Operator(MLOperatorOptions::default())))
    }

    /// Like [`Self::from_json_with_op_type`], but strips operation-level fields into [`OperationExtras`]
    /// (axis, cast target type, padding lengths, etc.) before deserializing the options dictionary.
    ///
    /// For building an [`crate::operators::Operation`], prefer [`crate::operators::Operation::from_json_attributes`],
    /// which calls this and [`crate::operators::Operation::from_operator_options`] in one step.
    pub fn from_json_with_op_type_and_extras(
        op_type: &str,
        value: &serde_json::Value,
    ) -> (Self, OperationExtras) {
        let mut v = value.clone();
        let extras = OperationExtras::extract_and_strip(op_type, &mut v);
        let opts = Self::from_json_with_op_type(op_type, &v).unwrap_or_default();
        (opts, extras)
    }

    /// Return attributes as a JSON value (for code that expects a `serde_json::Value`).
    pub fn to_value(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    /// Fallback: get a named attribute as JSON. Prefer typed accessors (as_reduce, as_conv2d, etc.).
    #[doc(hidden)]
    pub fn get(&self, key: &str) -> Option<serde_json::Value> {
        let v = self.to_value();
        v.get(key).cloned()
    }

    // ---------------------------------------------------------------------------
    // Typed accessors: return the options struct when the variant matches.
    // ---------------------------------------------------------------------------

    pub fn as_operator(&self) -> Option<&MLOperatorOptions> {
        match self {
            OperatorOptions::Operator(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_arg_min_max(&self) -> Option<&MLArgMinMaxOptions> {
        match self {
            OperatorOptions::ArgMinMax(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_batch_normalization(&self) -> Option<&MLBatchNormalizationOptions> {
        match self {
            OperatorOptions::BatchNormalization(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_clamp(&self) -> Option<&MLClampOptions> {
        match self {
            OperatorOptions::Clamp(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_conv2d(&self) -> Option<&MLConv2dOptions> {
        match self {
            OperatorOptions::Conv2d(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_constant(&self) -> Option<&MLConstantOptions> {
        match self {
            OperatorOptions::Constant(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_conv_transpose2d(&self) -> Option<&MLConvTranspose2dOptions> {
        match self {
            OperatorOptions::ConvTranspose2d(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_cumulative_sum(&self) -> Option<&MLCumulativeSumOptions> {
        match self {
            OperatorOptions::CumulativeSum(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_elu(&self) -> Option<&MLEluOptions> {
        match self {
            OperatorOptions::Elu(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_gather(&self) -> Option<&MLGatherOptions> {
        match self {
            OperatorOptions::Gather(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_gemm(&self) -> Option<&MLGemmOptions> {
        match self {
            OperatorOptions::Gemm(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_gru(&self) -> Option<&MLGruOptions> {
        match self {
            OperatorOptions::Gru(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_gru_cell(&self) -> Option<&MLGruCellOptions> {
        match self {
            OperatorOptions::GruCell(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_hard_sigmoid(&self) -> Option<&MLHardSigmoidOptions> {
        match self {
            OperatorOptions::HardSigmoid(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_instance_normalization(&self) -> Option<&MLInstanceNormalizationOptions> {
        match self {
            OperatorOptions::InstanceNormalization(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_layer_normalization(&self) -> Option<&MLLayerNormalizationOptions> {
        match self {
            OperatorOptions::LayerNormalization(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_leaky_relu(&self) -> Option<&MLLeakyReluOptions> {
        match self {
            OperatorOptions::LeakyRelu(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_linear(&self) -> Option<&MLLinearOptions> {
        match self {
            OperatorOptions::Linear(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_lstm(&self) -> Option<&MLLstmOptions> {
        match self {
            OperatorOptions::Lstm(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_lstm_cell(&self) -> Option<&MLLstmCellOptions> {
        match self {
            OperatorOptions::LstmCell(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_pad(&self) -> Option<&MLPadOptions> {
        match self {
            OperatorOptions::Pad(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_pool2d(&self) -> Option<&MLPool2dOptions> {
        match self {
            OperatorOptions::Pool2d(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_reduce(&self) -> Option<&MLReduceOptions> {
        match self {
            OperatorOptions::Reduce(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_resample2d(&self) -> Option<&MLResample2dOptions> {
        match self {
            OperatorOptions::Resample2d(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_reverse(&self) -> Option<&MLReverseOptions> {
        match self {
            OperatorOptions::Reverse(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_scatter_elements(&self) -> Option<&MLScatterOptions> {
        match self {
            OperatorOptions::ScatterElements(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_slice(&self) -> Option<&MLSliceOptions> {
        match self {
            OperatorOptions::Slice(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_split(&self) -> Option<&MLSplitOptions> {
        match self {
            OperatorOptions::Split(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_transpose(&self) -> Option<&MLTransposeOptions> {
        match self {
            OperatorOptions::Transpose(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_squeeze(&self) -> Option<&MLSqueezeOptions> {
        match self {
            OperatorOptions::Squeeze(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_unsqueeze(&self) -> Option<&MLUnsqueezeOptions> {
        match self {
            OperatorOptions::Unsqueeze(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_triangular(&self) -> Option<&MLTriangularOptions> {
        match self {
            OperatorOptions::Triangular(o) => Some(o),
            _ => None,
        }
    }
}

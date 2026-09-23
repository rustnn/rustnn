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

use crate::operator_enums::MLOperandDataType;

/// Operand reference (graph operand index). Used in option structs for MLOperand fields.
pub type OperandIndex = u32;

// ---------------------------------------------------------------------------
// WebNN IDL: MLDimension (supports dynamic dimensions)
// ---------------------------------------------------------------------------

/// MLDynamicDimension. IDL: `dictionary MLDynamicDimension { required DOMString name; required [EnforceRange] unsigned long maxSize; };`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLDynamicDimension {
    /// Name shared by dimensions that must have the same size at run time.
    pub name: String,
    /// Upper bound of the dimension; tensors are allocated for this size.
    pub max_size: u32,
}

/// MLDimension. IDL: `typedef ([EnforceRange] unsigned long or MLDynamicDimension) MLDimension;`
/// In JSON: either a number (static) or an object `{ "name": string, "maxSize": number }` (dynamic).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum MLDimension {
    /// Fixed size.
    Static(u32),
    /// Named dimension bounded by `max_size`; requires the `dynamic-inputs` feature.
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
    /// `axis` argument of `argMin`, `argMax`, `concat`, `cumulativeSum` and `softmax`.
    pub axis: Option<u32>,
    /// Target data type of `cast`.
    pub to_data_type: Option<MLOperandDataType>,
    /// `batchDimensions` of `gather` and `gatherElements`.
    pub batch_dimensions: Option<u32>,
    /// `steps` of `gru` and `lstm`.
    pub steps: Option<u32>,
    /// `hiddenSize` of the recurrent operations.
    pub hidden_size: Option<u32>,
    /// `beginningPadding` of `pad`.
    pub beginning_padding: Vec<u32>,
    /// `endingPadding` of `pad`.
    pub ending_padding: Vec<u32>,
    /// `starts` of `slice`.
    pub starts: Vec<u32>,
    /// `sizes` of `slice`.
    pub sizes: Vec<MLDimension>,
    /// Explicit split sizes of `split`.
    pub splits: Vec<u32>,
    /// Number of equal parts of `split` when `splits` is a count.
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
                if let Some(v) = obj.remove("to").or_else(|| obj.remove("dataType"))
                    && let Ok(dt) = serde_json::from_value::<MLOperandDataType>(v)
                {
                    out.to_data_type = Some(dt);
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
            "lstm" => {
                out.steps = remove_u32(obj, "steps");
                out.hidden_size =
                    remove_u32(obj, "hiddenSize").or_else(|| remove_u32(obj, "hidden_size"));
            }
            "lstmCell" => {
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
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLOperatorOptions {
    /// Free-form name of the operation, used in error messages and exported graphs.
    #[serde(default)]
    pub label: String,
}

// ---------------------------------------------------------------------------
// Dictionaries extending MLOperatorOptions (spec order)
// ---------------------------------------------------------------------------

/// MLArgMinMaxOptions. argMin / argMax (axis is a builder method parameter, not in this dictionary).
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlargminmaxoptions>
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLArgMinMaxOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Keep the reduced axis as a size-1 dimension (default `false`).
    #[serde(default)]
    pub keep_dimensions: bool,
    /// Data type of the index output, `int32` (default) or `int64`.
    #[serde(default = "default_arg_min_max_output_data_type")]
    pub output_data_type: MLOperandDataType,
}

fn default_arg_min_max_output_data_type() -> MLOperandDataType {
    MLOperandDataType::Int32
}

impl Default for MLArgMinMaxOptions {
    fn default() -> Self {
        Self {
            label: String::new(),
            keep_dimensions: false,
            output_data_type: MLOperandDataType::Int32,
        }
    }
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
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// 1-D scale operand with one value per channel (see `MLOperand::rustnn_index`).
    pub scale: Option<OperandIndex>,
    /// 1-D bias operand with one value per channel.
    pub bias: Option<OperandIndex>,
    /// Index of the channel dimension (default `1`).
    #[serde(default = "default_batch_norm_axis")]
    pub axis: u32,
    /// Value added to the variance before the square root (default `1e-5`).
    #[serde(default = "default_batch_norm_epsilon")]
    pub epsilon: f64,
}

impl std::hash::Hash for MLBatchNormalizationOptions {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.label.hash(state);
        self.scale.hash(state);
        self.bias.hash(state);
        self.axis.hash(state);
        self.epsilon.to_le_bytes().hash(state);
    }
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
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLClampOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    // TODO MTAX MLNumber is an union of any floating point or integral type
    /// Lower bound as a JSON number (WebNN `MLNumber`); `None` means negative infinity.
    pub min_value: Option<serde_json::Value>, // MLNumber
    /// Upper bound as a JSON number (WebNN `MLNumber`); `None` means positive infinity.
    pub max_value: Option<serde_json::Value>, // MLNumber
}

fn default_conv_groups() -> u32 {
    1
}

/// MLConv2dOptions. conv2d.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlconv2doptions>
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLConv2dOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// `[beginning_height, ending_height, beginning_width, ending_width]`; empty means no padding.
    #[serde(default)]
    pub padding: Vec<u32>,
    /// `[height, width]` strides; empty means `[1, 1]`.
    #[serde(default)]
    pub strides: Vec<u32>,
    /// `[height, width]` dilations; empty means `[1, 1]`.
    #[serde(default)]
    pub dilations: Vec<u32>,
    /// Number of groups the input channels are split into (default `1`; equal to the channel count for depthwise).
    #[serde(default = "default_conv_groups")]
    pub groups: u32,
    /// Input layout, `"nchw"` (default when empty) or `"nhwc"`.
    #[serde(default)]
    pub input_layout: String, // "nchw" | "nhwc"
    /// Filter layout, `"oihw"` (default when empty), `"hwio"`, `"ohwi"` or `"ihwo"`.
    #[serde(default)]
    pub filter_layout: String, // "oihw" | "hwio" | "ohwi" | "ihwo"
    /// 1-D bias operand with one value per output channel (see `MLOperand::rustnn_index`).
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLConvTranspose2dOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// `[beginning_height, ending_height, beginning_width, ending_width]`; empty means no padding.
    #[serde(default)]
    pub padding: Vec<u32>,
    /// `[height, width]` strides; empty means `[1, 1]`.
    #[serde(default)]
    pub strides: Vec<u32>,
    /// `[height, width]` dilations; empty means `[1, 1]`.
    #[serde(default)]
    pub dilations: Vec<u32>,
    /// Extra `[height, width]` added to the output; empty means `[0, 0]`.
    #[serde(default)]
    pub output_padding: Vec<u32>,
    /// Explicit `[height, width]` of the output; overrides `output_padding` when set.
    pub output_sizes: Option<Vec<u32>>,
    /// Number of groups (default `1`).
    #[serde(default = "default_conv_groups")]
    pub groups: u32,
    /// Input layout, `"nchw"` (default when empty) or `"nhwc"`.
    #[serde(default)]
    pub input_layout: String,
    /// Filter layout, `"iohw"` (default when empty), `"hwoi"` or `"ohwi"`.
    #[serde(default)]
    pub filter_layout: String, // "iohw" | "hwoi" | "ohwi"
    /// 1-D bias operand with one value per output channel.
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
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLConstantOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Initializer expression of the interchange format (for example a scalar fill).
    pub init: Option<String>,
    /// Base64-encoded little-endian constant bytes.
    pub data: Option<String>, // base64
    /// WebNN data type name of the constant.
    pub data_type: String,
    /// Shape of the constant.
    #[serde(default)]
    pub shape: Vec<u32>,
}

/// MLCumulativeSumOptions. cumulativeSum (axis is a builder method parameter).
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlcumulativesumoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLCumulativeSumOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Exclude the current element from its own sum (default `false`).
    #[serde(default)]
    pub exclusive: bool,
    /// Accumulate from the end of the axis (default `false`).
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
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Scale of the negative branch `alpha * (e^x - 1)` (default `1`).
    #[serde(default = "default_elu_alpha")]
    pub alpha: f64,
}

impl std::hash::Hash for MLEluOptions {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.label.hash(state);
        self.alpha.to_le_bytes().hash(state);
    }
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
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLGatherOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Dimension of `input` that `indices` index into (default `0`).
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
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Optional third operand `C`, broadcast to the output shape (see `MLOperand::rustnn_index`).
    pub c: Option<OperandIndex>,
    /// Multiplier of `A * B` (default `1`).
    #[serde(default = "default_gemm_alpha")]
    pub alpha: f64,
    /// Multiplier of `C` (default `1`).
    #[serde(default = "default_gemm_beta")]
    pub beta: f64,
    /// Transpose `A` before the product (default `false`).
    #[serde(default)]
    pub a_transpose: bool,
    /// Transpose `B` before the product (default `false`).
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

impl std::hash::Hash for MLGemmOptions {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.label.hash(state);
        self.c.hash(state);
        self.alpha.to_le_bytes().hash(state);
        self.beta.to_le_bytes().hash(state);
        self.a_transpose.hash(state);
        self.b_transpose.hash(state);
    }
}

/// MLGruOptions. gru.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlgruoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLGruOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Input bias, shape `[num_directions, 3 * hidden_size]`.
    pub bias: Option<OperandIndex>,
    /// Recurrent bias, shape `[num_directions, 3 * hidden_size]`.
    pub recurrent_bias: Option<OperandIndex>,
    /// Initial hidden state, shape `[num_directions, batch_size, hidden_size]` (default zeros).
    pub initial_hidden_state: Option<OperandIndex>,
    /// Apply the reset gate after the recurrent matrix multiplication (default `false` here; the spec default is `true`).
    #[serde(default)]
    pub reset_after: bool,
    /// Also return the hidden state of every step (default `false`).
    #[serde(default)]
    pub return_sequence: bool,
    /// `"forward"` (default when empty), `"backward"` or `"both"`.
    #[serde(default)]
    pub direction: String, // "forward" | "backward" | "both"
    /// Gate order of the weights, `"zrn"` (default when empty) or `"rzn"`.
    #[serde(default)]
    pub layout: String, // "zrn" | "rzn"
    /// Gate activations, two names from `relu`, `sigmoid`, `tanh` (default `["sigmoid", "tanh"]`).
    pub activations: Option<Vec<String>>, // MLRecurrentNetworkActivation
}

/// MLGruCellOptions. gruCell.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlgrucelloptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLGruCellOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Input bias, shape `[3 * hidden_size]`.
    pub bias: Option<OperandIndex>,
    /// Recurrent bias, shape `[3 * hidden_size]`.
    pub recurrent_bias: Option<OperandIndex>,
    /// Apply the reset gate after the recurrent matrix multiplication (default `false` here; the spec default is `true`).
    #[serde(default)]
    pub reset_after: bool,
    /// Gate order of the weights, `"zrn"` (default when empty) or `"rzn"`.
    #[serde(default)]
    pub layout: String,
    /// Gate activations, two names from `relu`, `sigmoid`, `tanh` (default `["sigmoid", "tanh"]`).
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
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Slope of the linear segment (default `0.2`).
    #[serde(default = "default_hard_sigmoid_alpha")]
    pub alpha: f64,
    /// Offset of the linear segment (default `0.5`).
    #[serde(default = "default_hard_sigmoid_beta")]
    pub beta: f64,
}

impl std::hash::Hash for MLHardSigmoidOptions {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.label.hash(state);
        self.alpha.to_be_bytes().hash(state);
        self.beta.to_be_bytes().hash(state);
    }
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
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// 1-D scale operand with one value per channel.
    pub scale: Option<OperandIndex>,
    /// 1-D bias operand with one value per channel.
    pub bias: Option<OperandIndex>,
    /// Value added to the variance before the square root (default `1e-5`).
    #[serde(default = "default_instance_norm_epsilon")]
    pub epsilon: f64,
    /// Input layout, `"nchw"` (default when empty) or `"nhwc"`.
    #[serde(default)]
    pub layout: String,
}

impl std::hash::Hash for MLInstanceNormalizationOptions {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.label.hash(state);
        self.scale.hash(state);
        self.bias.hash(state);
        self.epsilon.to_le_bytes().hash(state);
        self.layout.hash(state);
    }
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
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Scale operand shaped like the normalized `axes`.
    pub scale: Option<OperandIndex>,
    /// Bias operand shaped like the normalized `axes`.
    pub bias: Option<OperandIndex>,
    /// Dimensions to normalize; `None` means all but the first, `Some(vec![])` normalizes nothing.
    pub axes: Option<Vec<u32>>,
    /// Value added to the variance before the square root (default `1e-5`).
    #[serde(default = "default_layer_norm_epsilon")]
    pub epsilon: f64,
}

impl std::hash::Hash for MLLayerNormalizationOptions {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.label.hash(state);
        self.scale.hash(state);
        self.bias.hash(state);
        self.axes.hash(state);
        self.epsilon.to_le_bytes().hash(state);
    }
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
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Slope for negative inputs (default `0.01`).
    #[serde(default = "default_leaky_relu_alpha")]
    pub alpha: f64,
}

impl std::hash::Hash for MLLeakyReluOptions {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.label.hash(state);
        self.alpha.to_le_bytes().hash(state);
    }
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
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Multiplier (default `1`).
    #[serde(default = "default_linear_alpha")]
    pub alpha: f64,
    /// Offset (default `0`).
    #[serde(default = "default_linear_beta")]
    pub beta: f64,
}

impl std::hash::Hash for MLLinearOptions {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.label.hash(state);
        self.alpha.to_le_bytes().hash(state);
        self.beta.to_le_bytes().hash(state);
    }
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
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLLstmOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Input bias, shape `[num_directions, 4 * hidden_size]`.
    pub bias: Option<OperandIndex>,
    /// Recurrent bias, shape `[num_directions, 4 * hidden_size]`.
    pub recurrent_bias: Option<OperandIndex>,
    /// Peephole weights, shape `[num_directions, 3 * hidden_size]`.
    pub peephole_weight: Option<OperandIndex>,
    /// Initial hidden state, shape `[num_directions, batch_size, hidden_size]` (default zeros).
    pub initial_hidden_state: Option<OperandIndex>,
    /// Initial cell state, shape `[num_directions, batch_size, hidden_size]` (default zeros).
    pub initial_cell_state: Option<OperandIndex>,
    /// Also return the hidden state of every step (default `false`).
    #[serde(default)]
    pub return_sequence: bool,
    /// `"forward"` (default when empty), `"backward"` or `"both"`.
    #[serde(default)]
    pub direction: String,
    /// Gate order of the weights, `"iofg"` (default when empty) or `"ifgo"`.
    #[serde(default)]
    pub layout: String, // "iofg" | "ifgo"
    /// Gate activations, three names from `relu`, `sigmoid`, `tanh` (default `["sigmoid", "tanh", "tanh"]`).
    pub activations: Option<Vec<String>>,
}

/// MLLstmCellOptions. lstmCell.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mllstmcelloptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLLstmCellOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Input bias, shape `[4 * hidden_size]`.
    pub bias: Option<OperandIndex>,
    /// Recurrent bias, shape `[4 * hidden_size]`.
    pub recurrent_bias: Option<OperandIndex>,
    /// Peephole weights, shape `[3 * hidden_size]`.
    pub peephole_weight: Option<OperandIndex>,
    // TODO TMAX verify default
    /// Gate order of the weights, `"iofg"` (default when empty) or `"ifgo"`.
    #[serde(default)]
    pub layout: String,
    /// Gate activations, three names from `relu`, `sigmoid`, `tanh` (default `["sigmoid", "tanh", "tanh"]`).
    pub activations: Option<Vec<String>>,
}

/// MLPadOptions. pad (beginning/ending padding are builder method parameters).
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlpadoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLPadOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    // TODO MTAX mode is an enum of type MLPaddingMode
    /// `"constant"` (default when empty), `"edge"` or `"reflection"`.
    #[serde(default)]
    pub mode: String, // "constant" | "edge" | "reflection"
    /// Fill value for `"constant"` mode as a JSON number (default `0`).
    pub value: Option<serde_json::Value>, // MLNumber
}

/// MLPool2dOptions. averagePool2d / l2Pool2d / maxPool2d.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlpool2doptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLPool2dOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// `[height, width]` of the pooling window; `None` pools over the whole spatial extent.
    pub window_dimensions: Option<Vec<u32>>,
    // TODO MTAX check default value
    /// `[beginning_height, ending_height, beginning_width, ending_width]`; empty means no padding.
    #[serde(default)]
    pub padding: Vec<u32>,
    /// `[height, width]` strides; empty means `[1, 1]`.
    #[serde(default)]
    pub strides: Vec<u32>,
    /// `[height, width]` dilations; empty means `[1, 1]`.
    #[serde(default)]
    pub dilations: Vec<u32>,
    // TODO MTAX layout is enum MLInputOperandLayout
    /// Input layout, `"nchw"` (default when empty) or `"nhwc"`.
    #[serde(default)]
    pub layout: String,
    // TODO MTAX enum MLRoundingType
    /// Rounding of the output size, `"floor"` (default when empty) or `"ceil"`.
    #[serde(default)]
    pub output_shape_rounding: String,
    /// Explicit `[height, width]` of the output; overrides the rounding when set.
    pub output_sizes: Option<Vec<u32>>,
}

/// MLReduceOptions. reduceL1, reduceL2, reduceLogSum, etc.
/// `axes`: None = key omitted (spec default: all axes); Some(v) = use v (Some(vec![]) = reduce over no axes).
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlreduceoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLReduceOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Dimensions to reduce; `None` means all, `Some(vec![])` reduces nothing.
    pub axes: Option<Vec<u32>>,
    /// Keep the reduced dimensions as size 1 (default `false`).
    #[serde(default)]
    pub keep_dimensions: bool,
}

/// MLResample2dOptions. resample2d.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlresample2doptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLResample2dOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    // TODO MTAX enum MLInterpolationMode
    /// `"nearest-neighbor"` (default when empty) or `"linear"`.
    #[serde(default)]
    pub mode: String, // "nearest-neighbor" | "linear"
    /// Scale factor per resampled axis; empty means `[1.0, 1.0]`. Ignored when `sizes` is set.
    #[serde(default)]
    pub scales: Vec<f32>,
    /// Explicit output size per resampled axis.
    #[serde(default)]
    pub sizes: Option<Vec<u32>>,

    /// The two dimensions to resample; empty means `[2, 3]`.
    #[serde(default)]
    pub axes: Vec<u32>,
}

impl std::hash::Hash for MLResample2dOptions {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.label.hash(state);
        self.mode.hash(state);
        bytemuck::cast_slice::<f32, u8>(&self.scales).hash(state);
        self.sizes.hash(state);
        self.axes.hash(state);
    }
}

/// MLReverseOptions. reverse.
/// axes: omitted => reverse all dimensions; present and [] => reverse none (identity); present and [..] => reverse those axes.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlreverseoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLReverseOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// None = not present in JSON => reverse all. Some([]) => axes: [] => identity. Some([..]) => reverse those axes.
    pub axes: Option<Vec<u32>>,
}

/// MLScatterOptions. scatterElements / scatterND.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlscatteroptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLScatterOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Dimension along which `indices` address elements (default `0`; `scatterND` ignores it).
    #[serde(default)]
    pub axis: u32,
}

/// MLSliceOptions. slice (starts and sizes are builder method parameters).
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlsliceoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLSliceOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Step per dimension; empty means `1` everywhere.
    #[serde(default)]
    pub strides: Vec<u32>,
}

/// MLSplitOptions. split (splits is a builder method parameter).
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mlsplitoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLSplitOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Dimension to split along (default `0`).
    #[serde(default)]
    pub axis: u32,
}

/// MLTransposeOptions. transpose.
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mltransposeoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLTransposeOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// New order of the dimensions; empty reverses them.
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
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLSqueezeOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Size-1 dimensions to remove; empty removes all of them.
    #[serde(default)]
    pub axes: Vec<u32>,
}

/// MLUnsqueezeOptions. unsqueeze (emulation-only; not in WebNN IDL).
///
/// WebNN emulation: <https://www.w3.org/TR/webnn/#unsqueeze>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLUnsqueezeOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Positions in the output at which size-1 dimensions are inserted.
    #[serde(default)]
    pub axes: Vec<u32>,
}

/// MLTriangularOptions. triangular.
/// WebNN: when "upper" is not present, default is true (keep upper triangular).
///
/// WebNN: <https://www.w3.org/TR/webnn/#dictdef-mltriangularoptions>
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct MLTriangularOptions {
    /// Operation label.
    #[serde(default)]
    pub label: String,
    /// Keep the upper (`true`, the default when `None`) or lower triangle.
    pub upper: Option<bool>,
    /// Diagonal offset: positive moves above the main diagonal, negative below (default `0`).
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
    /// The `label` shared by every options dictionary.
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

    /// The [`MLOperatorOptions`] when this is the `operator` variant.
    pub fn as_operator(&self) -> Option<&MLOperatorOptions> {
        match self {
            OperatorOptions::Operator(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLArgMinMaxOptions`] when this is the `arg_min_max` variant.
    pub fn as_arg_min_max(&self) -> Option<&MLArgMinMaxOptions> {
        match self {
            OperatorOptions::ArgMinMax(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLBatchNormalizationOptions`] when this is the `batch_normalization` variant.
    pub fn as_batch_normalization(&self) -> Option<&MLBatchNormalizationOptions> {
        match self {
            OperatorOptions::BatchNormalization(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLClampOptions`] when this is the `clamp` variant.
    pub fn as_clamp(&self) -> Option<&MLClampOptions> {
        match self {
            OperatorOptions::Clamp(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLConv2dOptions`] when this is the `conv2d` variant.
    pub fn as_conv2d(&self) -> Option<&MLConv2dOptions> {
        match self {
            OperatorOptions::Conv2d(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLConstantOptions`] when this is the `constant` variant.
    pub fn as_constant(&self) -> Option<&MLConstantOptions> {
        match self {
            OperatorOptions::Constant(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLConvTranspose2dOptions`] when this is the `conv_transpose2d` variant.
    pub fn as_conv_transpose2d(&self) -> Option<&MLConvTranspose2dOptions> {
        match self {
            OperatorOptions::ConvTranspose2d(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLCumulativeSumOptions`] when this is the `cumulative_sum` variant.
    pub fn as_cumulative_sum(&self) -> Option<&MLCumulativeSumOptions> {
        match self {
            OperatorOptions::CumulativeSum(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLEluOptions`] when this is the `elu` variant.
    pub fn as_elu(&self) -> Option<&MLEluOptions> {
        match self {
            OperatorOptions::Elu(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLGatherOptions`] when this is the `gather` variant.
    pub fn as_gather(&self) -> Option<&MLGatherOptions> {
        match self {
            OperatorOptions::Gather(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLGemmOptions`] when this is the `gemm` variant.
    pub fn as_gemm(&self) -> Option<&MLGemmOptions> {
        match self {
            OperatorOptions::Gemm(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLGruOptions`] when this is the `gru` variant.
    pub fn as_gru(&self) -> Option<&MLGruOptions> {
        match self {
            OperatorOptions::Gru(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLGruCellOptions`] when this is the `gru_cell` variant.
    pub fn as_gru_cell(&self) -> Option<&MLGruCellOptions> {
        match self {
            OperatorOptions::GruCell(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLHardSigmoidOptions`] when this is the `hard_sigmoid` variant.
    pub fn as_hard_sigmoid(&self) -> Option<&MLHardSigmoidOptions> {
        match self {
            OperatorOptions::HardSigmoid(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLInstanceNormalizationOptions`] when this is the `instance_normalization` variant.
    pub fn as_instance_normalization(&self) -> Option<&MLInstanceNormalizationOptions> {
        match self {
            OperatorOptions::InstanceNormalization(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLLayerNormalizationOptions`] when this is the `layer_normalization` variant.
    pub fn as_layer_normalization(&self) -> Option<&MLLayerNormalizationOptions> {
        match self {
            OperatorOptions::LayerNormalization(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLLeakyReluOptions`] when this is the `leaky_relu` variant.
    pub fn as_leaky_relu(&self) -> Option<&MLLeakyReluOptions> {
        match self {
            OperatorOptions::LeakyRelu(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLLinearOptions`] when this is the `linear` variant.
    pub fn as_linear(&self) -> Option<&MLLinearOptions> {
        match self {
            OperatorOptions::Linear(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLLstmOptions`] when this is the `lstm` variant.
    pub fn as_lstm(&self) -> Option<&MLLstmOptions> {
        match self {
            OperatorOptions::Lstm(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLLstmCellOptions`] when this is the `lstm_cell` variant.
    pub fn as_lstm_cell(&self) -> Option<&MLLstmCellOptions> {
        match self {
            OperatorOptions::LstmCell(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLPadOptions`] when this is the `pad` variant.
    pub fn as_pad(&self) -> Option<&MLPadOptions> {
        match self {
            OperatorOptions::Pad(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLPool2dOptions`] when this is the `pool2d` variant.
    pub fn as_pool2d(&self) -> Option<&MLPool2dOptions> {
        match self {
            OperatorOptions::Pool2d(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLReduceOptions`] when this is the `reduce` variant.
    pub fn as_reduce(&self) -> Option<&MLReduceOptions> {
        match self {
            OperatorOptions::Reduce(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLResample2dOptions`] when this is the `resample2d` variant.
    pub fn as_resample2d(&self) -> Option<&MLResample2dOptions> {
        match self {
            OperatorOptions::Resample2d(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLReverseOptions`] when this is the `reverse` variant.
    pub fn as_reverse(&self) -> Option<&MLReverseOptions> {
        match self {
            OperatorOptions::Reverse(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLScatterOptions`] when this is the `scatter_elements` variant.
    pub fn as_scatter_elements(&self) -> Option<&MLScatterOptions> {
        match self {
            OperatorOptions::ScatterElements(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLSliceOptions`] when this is the `slice` variant.
    pub fn as_slice(&self) -> Option<&MLSliceOptions> {
        match self {
            OperatorOptions::Slice(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLSplitOptions`] when this is the `split` variant.
    pub fn as_split(&self) -> Option<&MLSplitOptions> {
        match self {
            OperatorOptions::Split(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLTransposeOptions`] when this is the `transpose` variant.
    pub fn as_transpose(&self) -> Option<&MLTransposeOptions> {
        match self {
            OperatorOptions::Transpose(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLSqueezeOptions`] when this is the `squeeze` variant.
    pub fn as_squeeze(&self) -> Option<&MLSqueezeOptions> {
        match self {
            OperatorOptions::Squeeze(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLUnsqueezeOptions`] when this is the `unsqueeze` variant.
    pub fn as_unsqueeze(&self) -> Option<&MLUnsqueezeOptions> {
        match self {
            OperatorOptions::Unsqueeze(o) => Some(o),
            _ => None,
        }
    }
    /// The [`MLTriangularOptions`] when this is the `triangular` variant.
    pub fn as_triangular(&self) -> Option<&MLTriangularOptions> {
        match self {
            OperatorOptions::Triangular(o) => Some(o),
            _ => None,
        }
    }
}

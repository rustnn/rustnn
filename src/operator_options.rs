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

use serde::de::Error;
use serde::{Deserialize, Deserializer, Serialize};

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

// ---------------------------------------------------------------------------
// Base: MLOperatorOptions
// ---------------------------------------------------------------------------

/// MLOperatorOptions. Base type for all operator options (label only).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLOperatorOptions {
    #[serde(default)]
    pub label: String,
}

// ---------------------------------------------------------------------------
// Dictionaries extending MLOperatorOptions (spec order)
// ---------------------------------------------------------------------------

/// MLArgMinMaxOptions. argMin / argMax.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLArgMinMaxOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub axis: u32,
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

/// MLCastOptions. cast.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLCastOptions {
    #[serde(default)]
    pub label: String,
    /// Target data type (e.g. "float32", "int32").
    #[serde(default)]
    pub to: String,
}

/// MLClampOptions. clamp.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLClampOptions {
    #[serde(default)]
    pub label: String,
    pub min_value: Option<serde_json::Value>, // MLNumber
    pub max_value: Option<serde_json::Value>, // MLNumber
}

fn default_conv_groups() -> u32 {
    1
}

/// MLConv2dOptions. conv2d.
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
    /// Output spatial shape [H, W]. WebNN camelCase: outputSizes.
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

/// MLConcatOptions. concat.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLConcatOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub axis: u32,
}

/// MLCumulativeSumOptions. cumulativeSum.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLCumulativeSumOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub axis: u32,
    #[serde(default)]
    pub exclusive: bool,
    #[serde(default)]
    pub reversed: bool,
}

/// MLExpandOptions. expand (newShape or axes from attributes for interchange).
/// newShape uses MLDimension (static or dynamic) per WebNN IDL.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLExpandOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default, rename = "newShape")]
    pub new_shape: Vec<MLDimension>,
    #[serde(default)]
    pub axes: Vec<u32>,
}

impl MLExpandOptions {
    /// Returns each dimension as u32 (static value or dynamic maxSize).
    pub fn new_shape_static_or_max(&self) -> Vec<u32> {
        self.new_shape
            .iter()
            .map(MLDimension::static_or_max)
            .collect()
    }
}

fn default_elu_alpha() -> f64 {
    1.0
}

/// MLEluOptions. elu.
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

/// MLGatherOptions. gather / gatherElements.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLGatherOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub axis: u32,
    /// gatherElements: batchDimensions (optional).
    pub batch_dimensions: Option<u32>,
}

fn default_gemm_alpha() -> f64 {
    1.0
}

fn default_gemm_beta() -> f64 {
    1.0
}

/// MLGemmOptions. gemm.
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
    pub hidden_size: Option<u32>,
}

/// MLGruCellOptions. gruCell.
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
    pub hidden_size: Option<u32>,
}

fn default_hard_sigmoid_alpha() -> f64 {
    0.2
}

fn default_hard_sigmoid_beta() -> f64 {
    0.5
}

/// MLHardSigmoidOptions. hardSigmoid.
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

fn default_hard_swish_alpha() -> f64 {
    1.0 / 6.0
}

fn default_hard_swish_beta() -> f64 {
    0.5
}

/// MLHardSwishOptions. hardSwish (optional alpha/beta for interchange).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLHardSwishOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default = "default_hard_swish_alpha")]
    pub alpha: f64,
    #[serde(default = "default_hard_swish_beta")]
    pub beta: f64,
}

impl Default for MLHardSwishOptions {
    fn default() -> Self {
        Self {
            label: String::new(),
            alpha: default_hard_swish_alpha(),
            beta: default_hard_swish_beta(),
        }
    }
}

fn default_instance_norm_epsilon() -> f64 {
    1e-5
}

/// MLInstanceNormalizationOptions. instanceNormalization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLInstanceNormalizationOptions {
    #[serde(default)]
    pub label: String,
    pub scale: Option<OperandIndex>,
    pub bias: Option<OperandIndex>,
    /// When exactly one of scale/bias is provided (2 operands), disambiguates so converters
    /// know which optional is present. Omitted when 1 or 3 operands.
    #[serde(default)]
    pub has_scale: Option<bool>,
    #[serde(default)]
    pub has_bias: Option<bool>,
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
            has_scale: None,
            has_bias: None,
            epsilon: default_instance_norm_epsilon(),
            layout: String::new(),
        }
    }
}

fn default_layer_norm_epsilon() -> f64 {
    1e-5
}

/// MLLayerNormalizationOptions. layerNormalization.
/// `axes`: None = key omitted (spec default [1..rank)); Some(v) = use v (Some(vec![]) = reduce over no axes).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLLayerNormalizationOptions {
    #[serde(default)]
    pub label: String,
    pub scale: Option<OperandIndex>,
    pub bias: Option<OperandIndex>,
    /// When exactly one of scale/bias is provided (2 operands), disambiguates for converters.
    #[serde(default)]
    pub has_scale: Option<bool>,
    #[serde(default)]
    pub has_bias: Option<bool>,
    /// Omitted in JSON => None => use spec default [1..rank). Present (including []) => use as-is.
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
            has_scale: None,
            has_bias: None,
            axes: None,
            epsilon: default_layer_norm_epsilon(),
        }
    }
}

fn default_leaky_relu_alpha() -> f64 {
    0.01
}

/// MLLeakyReluOptions. leakyRelu.
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
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLLstmCellOptions {
    #[serde(default)]
    pub label: String,
    pub bias: Option<OperandIndex>,
    pub recurrent_bias: Option<OperandIndex>,
    pub peephole_weight: Option<OperandIndex>,
    #[serde(default)]
    pub layout: String,
    pub activations: Option<Vec<String>>,
}

/// MLPadOptions. pad.
/// Note: In WebNN, padding lengths are MLOperands; we also support serializing them as
/// beginning_padding/ending_padding arrays for graph interchange.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLPadOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub mode: String, // "constant" | "edge" | "reflection"
    pub value: Option<serde_json::Value>, // MLNumber
    #[serde(default, rename = "beginningPadding")]
    pub beginning_padding: Vec<u32>,
    #[serde(default, rename = "endingPadding")]
    pub ending_padding: Vec<u32>,
}

/// MLPool2dOptions. averagePool2d / l2Pool2d / maxPool2d.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLPool2dOptions {
    #[serde(default)]
    pub label: String,
    pub window_dimensions: Option<Vec<u32>>,
    #[serde(default)]
    pub padding: Vec<u32>,
    #[serde(default)]
    pub strides: Vec<u32>,
    #[serde(default)]
    pub dilations: Vec<u32>,
    #[serde(default)]
    pub layout: String,
    /// "floor" | "ceil". WebNN spec and WPT use "roundingType"; we accept both keys.
    #[serde(default, alias = "roundingType")]
    pub output_shape_rounding: String,
    pub output_sizes: Option<Vec<u32>>,
}

/// MLReduceOptions. reduceL1, reduceL2, reduceLogSum, etc.
/// `axes`: None = key omitted (spec default: all axes); Some(v) = use v (Some(vec![]) = reduce over no axes).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLReduceOptions {
    #[serde(default)]
    pub label: String,
    /// Omitted in JSON => None => reduce over all axes. Present (including []) => use as-is ([] = no reduction).
    pub axes: Option<Vec<u32>>,
    #[serde(default)]
    pub keep_dimensions: bool,
}

/// MLReshapeOptions. reshape (newShape from attributes for interchange).
/// newShape uses MLDimension (static or dynamic) per WebNN IDL.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLReshapeOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default, rename = "newShape")]
    pub new_shape: Vec<MLDimension>,
}

impl MLReshapeOptions {
    /// Returns each dimension as u32 (static value or dynamic maxSize).
    pub fn new_shape_static_or_max(&self) -> Vec<u32> {
        self.new_shape
            .iter()
            .map(MLDimension::static_or_max)
            .collect()
    }
}

/// MLResample2dOptions. resample2d.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLResample2dOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub mode: String, // "nearest-neighbor" | "linear"
    #[serde(default)]
    pub scales: Vec<f32>,
    pub sizes: Option<Vec<u32>>,
    #[serde(default)]
    pub axes: Vec<u32>,
}

/// MLReverseOptions. reverse.
/// axes: omitted => reverse all dimensions; present and [] => reverse none (identity); present and [..] => reverse those axes.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLReverseOptions {
    #[serde(default)]
    pub label: String,
    /// None = not present in JSON => reverse all. Some([]) => axes: [] => identity. Some([..]) => reverse those axes.
    pub axes: Option<Vec<u32>>,
}

/// MLSoftmaxOptions. softmax.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLSoftmaxOptions {
    #[serde(default)]
    pub label: String,
    /// Axis over which to compute softmax. Default -1 (last axis).
    #[serde(default = "default_softmax_axis")]
    pub axis: i32,
}

fn default_softmax_axis() -> i32 {
    -1
}

impl Default for MLSoftmaxOptions {
    fn default() -> Self {
        Self {
            label: String::new(),
            axis: default_softmax_axis(),
        }
    }
}

/// MLScatterOptions. scatterElements.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLScatterOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub axis: u32,
}

/// MLSliceOptions. slice.
/// In WebNN, starts/sizes are MLOperands; we also support them as arrays for interchange.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLSliceOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub starts: Vec<u32>,
    #[serde(default)]
    pub sizes: Vec<u32>,
    #[serde(default)]
    pub strides: Vec<u32>,
}

/// Deserialize splits as either a number (equal-split count; store empty vec, TRTX uses output count)
/// or an array of sizes.
fn deserialize_splits<'de, D>(d: D) -> Result<Vec<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    let v = serde_json::Value::deserialize(d)?;
    match v {
        serde_json::Value::Number(n) => {
            let _ = n
                .as_u64()
                .ok_or_else(|| D::Error::custom("splits number out of range"))?;
            Ok(Vec::new())
        }
        serde_json::Value::Array(arr) => arr
            .iter()
            .map(|e| {
                e.as_u64()
                    .ok_or_else(|| D::Error::custom("splits array element not u64"))
                    .map(|u| u as u32)
            })
            .collect::<Result<Vec<u32>, _>>(),
        serde_json::Value::Null => Ok(Vec::new()),
        _ => Err(D::Error::custom("splits must be number or array")),
    }
}

/// MLSplitOptions. split.
/// Splits array from attributes for interchange (WebNN also has splits as MLOperand or number).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLSplitOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub axis: u32,
    #[serde(default, deserialize_with = "deserialize_splits")]
    pub splits: Vec<u32>,
}

/// MLTransposeOptions. transpose.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLTransposeOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub permutation: Vec<u32>,
}

/// MLUnsqueezeOptions. unsqueeze.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLUnsqueezeOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub axes: Vec<u32>,
}

/// MLTileOptions. tile (repetitions from attributes for interchange).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MLTileOptions {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub repetitions: Vec<u32>,
}

/// MLTriangularOptions. triangular.
/// WebNN: when "upper" is not present, default is true (keep upper triangular).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MLTriangularOptions {
    #[serde(default)]
    pub label: String,
    /// None = not present => default true (upper). Some(b) => use b.
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
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OperatorOptions {
    /// MLOperatorOptions (base; label only).
    Operator(MLOperatorOptions),

    /// MLArgMinMaxOptions.
    ArgMinMax(MLArgMinMaxOptions),

    /// MLBatchNormalizationOptions.
    BatchNormalization(MLBatchNormalizationOptions),

    /// MLCastOptions.
    Cast(MLCastOptions),

    /// MLClampOptions.
    Clamp(MLClampOptions),

    /// MLConstantOptions.
    Constant(MLConstantOptions),

    /// MLConv2dOptions.
    Conv2d(MLConv2dOptions),

    /// MLConvTranspose2dOptions.
    ConvTranspose2d(MLConvTranspose2dOptions),

    /// MLConcatOptions.
    Concat(MLConcatOptions),

    /// MLCumulativeSumOptions.
    CumulativeSum(MLCumulativeSumOptions),

    /// MLExpandOptions.
    Expand(MLExpandOptions),

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

    /// MLHardSwishOptions.
    HardSwish(MLHardSwishOptions),

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

    /// MLReshapeOptions.
    Reshape(MLReshapeOptions),

    /// MLResample2dOptions.
    Resample2d(MLResample2dOptions),

    /// MLReverseOptions.
    Reverse(MLReverseOptions),

    /// MLScatterOptions.
    ScatterElements(MLScatterOptions),

    /// MLSoftmaxOptions.
    Softmax(MLSoftmaxOptions),

    /// MLSliceOptions.
    Slice(MLSliceOptions),

    /// MLSplitOptions.
    Split(MLSplitOptions),

    /// MLTransposeOptions.
    Transpose(MLTransposeOptions),

    /// MLUnsqueezeOptions.
    Unsqueeze(MLUnsqueezeOptions),

    /// MLTileOptions.
    Tile(MLTileOptions),

    /// MLTriangularOptions.
    Triangular(MLTriangularOptions),
}

impl Default for OperatorOptions {
    fn default() -> Self {
        OperatorOptions::Operator(MLOperatorOptions::default())
    }
}

impl OperatorOptions {
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
                "cast" => try_opt!(MLCastOptions, Cast),
                "clamp" => try_opt!(MLClampOptions, Clamp),
                "conv2d" => try_opt!(MLConv2dOptions, Conv2d),
                "convTranspose2d" => try_opt!(MLConvTranspose2dOptions, ConvTranspose2d),
                "concat" => try_opt!(MLConcatOptions, Concat),
                "constant" => try_opt!(MLConstantOptions, Constant),
                "cumulativeSum" => try_opt!(MLCumulativeSumOptions, CumulativeSum),
                "expand" => try_opt!(MLExpandOptions, Expand),
                "elu" => try_opt!(MLEluOptions, Elu),
                "gather" | "gatherElements" => try_opt!(MLGatherOptions, Gather),
                "gemm" => try_opt!(MLGemmOptions, Gemm),
                "gru" => try_opt!(MLGruOptions, Gru),
                "gruCell" => try_opt!(MLGruCellOptions, GruCell),
                "hardSigmoid" => try_opt!(MLHardSigmoidOptions, HardSigmoid),
                "hardSwish" => try_opt!(MLHardSwishOptions, HardSwish),
                "instanceNormalization" => {
                    try_opt!(MLInstanceNormalizationOptions, InstanceNormalization)
                }
                "layerNormalization" => try_opt!(MLLayerNormalizationOptions, LayerNormalization),
                "leakyRelu" => try_opt!(MLLeakyReluOptions, LeakyRelu),
                "linear" => try_opt!(MLLinearOptions, Linear),
                "lstm" => try_opt!(MLLstmOptions, Lstm),
                "lstmCell" => try_opt!(MLLstmCellOptions, LstmCell),
                "pad" => try_opt!(MLPadOptions, Pad),
                "averagePool2d" | "maxPool2d" | "l2Pool2d" => try_opt!(MLPool2dOptions, Pool2d),
                "reduceSum" | "reduceMean" | "reduceMax" | "reduceMin" | "reduceProduct"
                | "reduceL1" | "reduceL2" | "reduceLogSum" | "reduceLogSumExp"
                | "reduceSumSquare" => {
                    try_opt!(MLReduceOptions, Reduce)
                }
                "reshape" => try_opt!(MLReshapeOptions, Reshape),
                "resample2d" => try_opt!(MLResample2dOptions, Resample2d),
                "reverse" => try_opt!(MLReverseOptions, Reverse),
                "scatterElements" => try_opt!(MLScatterOptions, ScatterElements),
                "softmax" => try_opt!(MLSoftmaxOptions, Softmax),
                "slice" => try_opt!(MLSliceOptions, Slice),
                "split" => try_opt!(MLSplitOptions, Split),
                "transpose" => try_opt!(MLTransposeOptions, Transpose),
                "unsqueeze" => try_opt!(MLUnsqueezeOptions, Unsqueeze),
                "tile" => try_opt!(MLTileOptions, Tile),
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
    pub fn as_cast(&self) -> Option<&MLCastOptions> {
        match self {
            OperatorOptions::Cast(o) => Some(o),
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
    pub fn as_concat(&self) -> Option<&MLConcatOptions> {
        match self {
            OperatorOptions::Concat(o) => Some(o),
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
    pub fn as_hard_swish(&self) -> Option<&MLHardSwishOptions> {
        match self {
            OperatorOptions::HardSwish(o) => Some(o),
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
    pub fn as_reshape(&self) -> Option<&MLReshapeOptions> {
        match self {
            OperatorOptions::Reshape(o) => Some(o),
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
    pub fn as_softmax(&self) -> Option<&MLSoftmaxOptions> {
        match self {
            OperatorOptions::Softmax(o) => Some(o),
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
    pub fn as_unsqueeze(&self) -> Option<&MLUnsqueezeOptions> {
        match self {
            OperatorOptions::Unsqueeze(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_tile(&self) -> Option<&MLTileOptions> {
        match self {
            OperatorOptions::Tile(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_triangular(&self) -> Option<&MLTriangularOptions> {
        match self {
            OperatorOptions::Triangular(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_expand(&self) -> Option<&MLExpandOptions> {
        match self {
            OperatorOptions::Expand(o) => Some(o),
            _ => None,
        }
    }
}

//! Nested limit dictionaries from the WebNN specification (MLOpSupportLimits partials).

use serde::{Deserialize, Serialize};

use super::MLTensorLimits;

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLBatchNormalizationSupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mlbatchnormalizationsupportlimits).
pub struct MLBatchNormalizationSupportLimits {
    /// `MLTensorLimits` for input operand.
    pub input: MLTensorLimits,
    /// `MLTensorLimits` for mean operand.
    pub mean: MLTensorLimits,
    /// `MLTensorLimits` for variance operand.
    pub variance: MLTensorLimits,
    /// `MLTensorLimits` for scale operand.
    pub scale: MLTensorLimits,
    /// `MLTensorLimits` for bias operand.
    pub bias: MLTensorLimits,
    /// `MLTensorLimits` for output operand.
    pub output: MLTensorLimits,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLBinarySupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mlbinarysupportlimits).
pub struct MLBinarySupportLimits {
    /// `MLTensorLimits` for a operand.
    pub a: MLTensorLimits,
    /// `MLTensorLimits` for b operand.
    pub b: MLTensorLimits,
    /// `MLTensorLimits` for output operand.
    pub output: MLTensorLimits,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLConcatSupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mlconcatsupportlimits).
pub struct MLConcatSupportLimits {
    /// `MLTensorLimits` for all input operands.
    pub inputs: MLTensorLimits,
    /// `MLTensorLimits` for output operand.
    pub output: MLTensorLimits,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLConv2dSupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mlconv2dsupportlimits).
pub struct MLConv2dSupportLimits {
    /// `MLTensorLimits` for input operand.
    pub input: MLTensorLimits,
    /// `MLTensorLimits` for filter operand.
    pub filter: MLTensorLimits,
    /// `MLTensorLimits` for bias operand.
    pub bias: MLTensorLimits,
    /// `MLTensorLimits` for output operand.
    pub output: MLTensorLimits,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLGatherSupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mlgathersupportlimits).
pub struct MLGatherSupportLimits {
    /// `MLTensorLimits` for input operand.
    pub input: MLTensorLimits,
    /// `MLTensorLimits` for indices operand.
    pub indices: MLTensorLimits,
    /// `MLTensorLimits` for output operand.
    pub output: MLTensorLimits,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLGemmSupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mlgemmsupportlimits).
pub struct MLGemmSupportLimits {
    /// `MLTensorLimits` for a operand.
    pub a: MLTensorLimits,
    /// `MLTensorLimits` for b operand.
    pub b: MLTensorLimits,
    /// `MLTensorLimits` for c operand.
    pub c: MLTensorLimits,
    /// `MLTensorLimits` for output operand.
    pub output: MLTensorLimits,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLGruSupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mlgrusupportlimits).
pub struct MLGruSupportLimits {
    /// `MLTensorLimits` for input operand.
    pub input: MLTensorLimits,
    /// `MLTensorLimits` for weight operand.
    pub weight: MLTensorLimits,
    /// `MLTensorLimits` for recurrentWeight operand.
    pub recurrent_weight: MLTensorLimits,
    /// `MLTensorLimits` for bias operand.
    pub bias: MLTensorLimits,
    /// `MLTensorLimits` for recurrentBias operand.
    pub recurrent_bias: MLTensorLimits,
    /// `MLTensorLimits` for initialHiddenState operand.
    pub initial_hidden_state: MLTensorLimits,
    /// `MLTensorLimits` for the first output operand.
    pub output0: MLTensorLimits,
    /// `MLTensorLimits` for the second output operand.
    pub output1: MLTensorLimits,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLGruCellSupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mlgrucellsupportlimits).
pub struct MLGruCellSupportLimits {
    /// `MLTensorLimits` for input operand.
    pub input: MLTensorLimits,
    /// `MLTensorLimits` for weight operand.
    pub weight: MLTensorLimits,
    /// `MLTensorLimits` for recurrentWeight operand.
    pub recurrent_weight: MLTensorLimits,
    /// `MLTensorLimits` for hiddenState operand.
    pub hidden_state: MLTensorLimits,
    /// `MLTensorLimits` for bias operand.
    pub bias: MLTensorLimits,
    /// `MLTensorLimits` for recurrentBias operand.
    pub recurrent_bias: MLTensorLimits,
    /// `MLTensorLimits` for output operand.
    pub output: MLTensorLimits,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLLogicalNotSupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mllogicalnotsupportlimits).
pub struct MLLogicalNotSupportLimits {
    /// `MLTensorLimits` for a operand.
    pub a: MLTensorLimits,
    /// `MLTensorLimits` for output operand.
    pub output: MLTensorLimits,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLLstmSupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mllstmsupportlimits).
pub struct MLLstmSupportLimits {
    /// `MLTensorLimits` for input operand.
    pub input: MLTensorLimits,
    /// `MLTensorLimits` for weight operand.
    pub weight: MLTensorLimits,
    /// `MLTensorLimits` for recurrentWeight operand.
    pub recurrent_weight: MLTensorLimits,
    /// `MLTensorLimits` for bias operand.
    pub bias: MLTensorLimits,
    /// `MLTensorLimits` for recurrentBias operand.
    pub recurrent_bias: MLTensorLimits,
    /// `MLTensorLimits` for peepholeWeight operand.
    pub peephole_weight: MLTensorLimits,
    /// `MLTensorLimits` for initialHiddenState operand.
    pub initial_hidden_state: MLTensorLimits,
    /// `MLTensorLimits` for initialCellState operand.
    pub initial_cell_state: MLTensorLimits,
    /// `MLTensorLimits` for the first output operand.
    pub output0: MLTensorLimits,
    /// `MLTensorLimits` for the second output operand.
    pub output1: MLTensorLimits,
    /// `MLTensorLimits` for the third output operand.
    pub output2: MLTensorLimits,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLLstmCellSupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mllstmcellsupportlimits).
pub struct MLLstmCellSupportLimits {
    /// `MLTensorLimits` for input operand.
    pub input: MLTensorLimits,
    /// `MLTensorLimits` for weight operand.
    pub weight: MLTensorLimits,
    /// `MLTensorLimits` for recurrentWeight operand.
    pub recurrent_weight: MLTensorLimits,
    /// `MLTensorLimits` for hiddenState operand.
    pub hidden_state: MLTensorLimits,
    /// `MLTensorLimits` for cellState operand.
    pub cell_state: MLTensorLimits,
    /// `MLTensorLimits` for bias operand.
    pub bias: MLTensorLimits,
    /// `MLTensorLimits` for recurrentBias operand.
    pub recurrent_bias: MLTensorLimits,
    /// `MLTensorLimits` for peepholeWeight operand.
    pub peephole_weight: MLTensorLimits,
    /// `MLTensorLimits` for the first output operand.
    pub output0: MLTensorLimits,
    /// `MLTensorLimits` for the second output operand.
    pub output1: MLTensorLimits,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLNormalizationSupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mlnormalizationsupportlimits).
pub struct MLNormalizationSupportLimits {
    /// `MLTensorLimits` for input operand.
    pub input: MLTensorLimits,
    /// `MLTensorLimits` for scale operand.
    pub scale: MLTensorLimits,
    /// `MLTensorLimits` for bias operand.
    pub bias: MLTensorLimits,
    /// `MLTensorLimits` for output operand.
    pub output: MLTensorLimits,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLPreluSupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mlprelusupportlimits).
pub struct MLPreluSupportLimits {
    /// `MLTensorLimits` for input operand.
    pub input: MLTensorLimits,
    /// `MLTensorLimits` for slope operand.
    pub slope: MLTensorLimits,
    /// `MLTensorLimits` for output operand.
    pub output: MLTensorLimits,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLQuantizeDequantizeLinearSupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mlquantizedequantizelinearsupportlimits).
pub struct MLQuantizeDequantizeLinearSupportLimits {
    /// `MLTensorLimits` for input operand.
    pub input: MLTensorLimits,
    /// `MLTensorLimits` for scale operand.
    pub scale: MLTensorLimits,
    /// `MLTensorLimits` for zeroPoint operand.
    pub zero_point: MLTensorLimits,
    /// `MLTensorLimits` for output operand.
    pub output: MLTensorLimits,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLScatterSupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mlscattersupportlimits).
pub struct MLScatterSupportLimits {
    /// `MLTensorLimits` for input operand.
    pub input: MLTensorLimits,
    /// `MLTensorLimits` for indices operand.
    pub indices: MLTensorLimits,
    /// `MLTensorLimits` for updates operand.
    pub updates: MLTensorLimits,
    /// `MLTensorLimits` for output operand.
    pub output: MLTensorLimits,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLSingleInputSupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mlsingleinputsupportlimits).
pub struct MLSingleInputSupportLimits {
    /// `MLTensorLimits` for input operand.
    pub input: MLTensorLimits,
    /// `MLTensorLimits` for output operand.
    pub output: MLTensorLimits,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLSplitSupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mlsplitsupportlimits).
pub struct MLSplitSupportLimits {
    /// `MLTensorLimits` for input operand.
    pub input: MLTensorLimits,
    /// `MLTensorLimits` for output operands.
    pub outputs: MLTensorLimits,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// WebNN `MLWhereSupportLimits` dictionary.
///
/// See the [cached WebNN specification](https://www.w3.org/TR/webnn/#dictdef-mlwheresupportlimits).
pub struct MLWhereSupportLimits {
    /// `MLTensorLimits` for condition operand.
    pub condition: MLTensorLimits,
    /// `MLTensorLimits` for trueValue operand.
    pub true_value: MLTensorLimits,
    /// `MLTensorLimits` for falseValue operand.
    pub false_value: MLTensorLimits,
    /// `MLTensorLimits` for output operand.
    pub output: MLTensorLimits,
}

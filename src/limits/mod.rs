//! WebNN [`MLOpSupportLimits`](https://www.w3.org/TR/webnn/#dictdef-mlopsupportlimits) and related
//! limit dictionaries, for JSON interchange and alignment with `web-sys` / wasm-bindgen bindings.

// we don't want to invent docs to those types that don't have documentation in WebIDL
#![expect(missing_docs)]

mod types;

pub use types::{
    MLBatchNormalizationSupportLimits, MLBinarySupportLimits, MLConcatSupportLimits,
    MLConv2dSupportLimits, MLGatherSupportLimits, MLGemmSupportLimits, MLGruCellSupportLimits,
    MLGruSupportLimits, MLLogicalNotSupportLimits, MLLstmCellSupportLimits, MLLstmSupportLimits,
    MLNormalizationSupportLimits, MLPreluSupportLimits, MLQuantizeDequantizeLinearSupportLimits,
    MLScatterSupportLimits, MLSingleInputSupportLimits, MLSplitSupportLimits, MLWhereSupportLimits,
};

use serde::{Deserialize, Serialize};

use crate::operator_enums::MLOperandDataType;

/// `MLInputOperandLayout` — preferred layout for layout-dependent operators (e.g. conv2d).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MLInputOperandLayout {
    Nchw,
    #[default]
    Nhwc,
}

/// In RustNN, we decide here to support up to rank 8, individual backends might return a lower limit
pub const RUSTNN_MAX_RANK: u32 = 8;
/// In RustNN, we decide here to support all tensor sizes supported by the u32 value range, individual backends might return a lower limit
pub const RUSTNN_MAX_TENSOR: u32 = u32::MAX;

/// `MLRankRange` — inclusive min/max tensor rank supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLRankRange {
    pub min: u32,
    pub max: u32,
}

impl Default for MLRankRange {
    fn default() -> Self {
        Self {
            min: 0,
            max: RUSTNN_MAX_RANK,
        }
    }
}

/// `MLTensorLimits` — allowed operand data types and rank range.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLTensorLimits {
    pub data_types: Vec<MLOperandDataType>, // use &[DataType] ?
    pub rank_range: MLRankRange,
}

impl Default for MLTensorLimits {
    fn default() -> Self {
        Self {
            data_types: vec![
                MLOperandDataType::Float32,
                MLOperandDataType::Float16,
                MLOperandDataType::Int32,
                MLOperandDataType::Uint32,
                MLOperandDataType::Int64,
                MLOperandDataType::Uint64,
                MLOperandDataType::Int8,
                MLOperandDataType::Uint8,
                MLOperandDataType::Int4,
                MLOperandDataType::Uint4,
            ],
            rank_range: Default::default(),
        }
    }
}

/// `MLOpSupportLimits` — merged dictionary from the specification (base + all partials).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct MLOpSupportLimits {
    pub preferred_input_layout: MLInputOperandLayout,
    pub max_tensor_byte_length: u64,
    pub input: MLTensorLimits,
    pub constant: MLTensorLimits,
    pub output: MLTensorLimits,

    #[serde(default)]
    pub arg_min: MLSingleInputSupportLimits,
    #[serde(default)]
    pub arg_max: MLSingleInputSupportLimits,
    #[serde(default)]
    pub batch_normalization: MLBatchNormalizationSupportLimits,
    #[serde(default)]
    pub cast: MLSingleInputSupportLimits,
    #[serde(default)]
    pub clamp: MLSingleInputSupportLimits,
    #[serde(default)]
    pub concat: MLConcatSupportLimits,
    #[serde(default)]
    pub conv2d: MLConv2dSupportLimits,
    #[serde(default)]
    pub conv_transpose2d: MLConv2dSupportLimits,
    #[serde(default)]
    pub cumulative_sum: MLSingleInputSupportLimits,

    #[serde(default)]
    pub add: MLBinarySupportLimits,
    #[serde(default)]
    pub sub: MLBinarySupportLimits,
    #[serde(default)]
    pub mul: MLBinarySupportLimits,
    #[serde(default)]
    pub div: MLBinarySupportLimits,
    #[serde(default)]
    pub max: MLBinarySupportLimits,
    #[serde(default)]
    pub min: MLBinarySupportLimits,
    #[serde(default)]
    pub pow: MLBinarySupportLimits,

    #[serde(default)]
    pub equal: MLBinarySupportLimits,
    #[serde(default)]
    pub not_equal: MLBinarySupportLimits,
    #[serde(default)]
    pub greater: MLBinarySupportLimits,
    #[serde(default)]
    pub greater_or_equal: MLBinarySupportLimits,
    #[serde(default)]
    pub lesser: MLBinarySupportLimits,
    #[serde(default)]
    pub lesser_or_equal: MLBinarySupportLimits,
    #[serde(default)]
    pub logical_not: MLLogicalNotSupportLimits,
    #[serde(default)]
    pub logical_and: MLBinarySupportLimits,
    #[serde(default)]
    pub logical_or: MLBinarySupportLimits,
    #[serde(default)]
    pub logical_xor: MLBinarySupportLimits,
    #[serde(default)]
    pub is_na_n: MLLogicalNotSupportLimits,
    #[serde(default)]
    pub is_infinite: MLLogicalNotSupportLimits,

    #[serde(default)]
    pub abs: MLSingleInputSupportLimits,
    #[serde(default)]
    pub ceil: MLSingleInputSupportLimits,
    #[serde(default)]
    pub cos: MLSingleInputSupportLimits,
    #[serde(default)]
    pub erf: MLSingleInputSupportLimits,
    #[serde(default)]
    pub exp: MLSingleInputSupportLimits,
    #[serde(default)]
    pub floor: MLSingleInputSupportLimits,
    #[serde(default)]
    pub identity: MLSingleInputSupportLimits,
    #[serde(default)]
    pub log: MLSingleInputSupportLimits,
    #[serde(default)]
    pub neg: MLSingleInputSupportLimits,
    #[serde(default)]
    pub reciprocal: MLSingleInputSupportLimits,
    #[serde(default)]
    pub round_even: MLSingleInputSupportLimits,
    #[serde(default)]
    pub sin: MLSingleInputSupportLimits,
    #[serde(default)]
    pub sign: MLSingleInputSupportLimits,
    #[serde(default)]
    pub sqrt: MLSingleInputSupportLimits,
    #[serde(default)]
    pub tan: MLSingleInputSupportLimits,

    #[serde(default)]
    pub dequantize_linear: MLQuantizeDequantizeLinearSupportLimits,
    #[serde(default)]
    pub quantize_linear: MLQuantizeDequantizeLinearSupportLimits,

    #[serde(default)]
    pub elu: MLSingleInputSupportLimits,
    #[serde(default)]
    pub expand: MLSingleInputSupportLimits,

    #[serde(default)]
    pub gather: MLGatherSupportLimits,
    #[serde(default)]
    pub gather_elements: MLGatherSupportLimits,
    #[serde(default)]
    pub gather_nd: MLGatherSupportLimits,

    #[serde(default)]
    pub gelu: MLSingleInputSupportLimits,

    #[serde(default)]
    pub gemm: MLGemmSupportLimits,

    #[serde(default)]
    pub gru: MLGruSupportLimits,
    #[serde(default)]
    pub gru_cell: MLGruCellSupportLimits,

    #[serde(default)]
    pub hard_sigmoid: MLSingleInputSupportLimits,
    #[serde(default)]
    pub hard_swish: MLSingleInputSupportLimits,

    #[serde(default)]
    pub instance_normalization: MLNormalizationSupportLimits,
    #[serde(default)]
    pub layer_normalization: MLNormalizationSupportLimits,

    #[serde(default)]
    pub leaky_relu: MLSingleInputSupportLimits,
    #[serde(default)]
    pub linear: MLSingleInputSupportLimits,

    #[serde(default)]
    pub lstm: MLLstmSupportLimits,
    #[serde(default)]
    pub lstm_cell: MLLstmCellSupportLimits,

    #[serde(default)]
    pub matmul: MLBinarySupportLimits,

    #[serde(default)]
    pub pad: MLSingleInputSupportLimits,

    #[serde(default)]
    pub average_pool2d: MLSingleInputSupportLimits,
    #[serde(default)]
    pub l2_pool2d: MLSingleInputSupportLimits,
    #[serde(default)]
    pub max_pool2d: MLSingleInputSupportLimits,

    #[serde(default)]
    pub prelu: MLPreluSupportLimits,

    #[serde(default)]
    pub reduce_l1: MLSingleInputSupportLimits,
    #[serde(default)]
    pub reduce_l2: MLSingleInputSupportLimits,
    #[serde(default)]
    pub reduce_log_sum: MLSingleInputSupportLimits,
    #[serde(default)]
    pub reduce_log_sum_exp: MLSingleInputSupportLimits,
    #[serde(default)]
    pub reduce_max: MLSingleInputSupportLimits,
    #[serde(default)]
    pub reduce_mean: MLSingleInputSupportLimits,
    #[serde(default)]
    pub reduce_min: MLSingleInputSupportLimits,
    #[serde(default)]
    pub reduce_product: MLSingleInputSupportLimits,
    #[serde(default)]
    pub reduce_sum: MLSingleInputSupportLimits,
    #[serde(default)]
    pub reduce_sum_square: MLSingleInputSupportLimits,

    #[serde(default)]
    pub relu: MLSingleInputSupportLimits,

    #[serde(default)]
    pub resample2d: MLSingleInputSupportLimits,

    #[serde(default)]
    pub reshape: MLSingleInputSupportLimits,

    #[serde(default)]
    pub reverse: MLSingleInputSupportLimits,

    #[serde(default)]
    pub scatter_elements: MLScatterSupportLimits,
    #[serde(default)]
    pub scatter_nd: MLScatterSupportLimits,

    #[serde(default)]
    pub sigmoid: MLSingleInputSupportLimits,

    #[serde(default)]
    pub slice: MLSingleInputSupportLimits,

    #[serde(default)]
    pub softmax: MLSingleInputSupportLimits,

    #[serde(default)]
    pub softplus: MLSingleInputSupportLimits,

    #[serde(default)]
    pub softsign: MLSingleInputSupportLimits,

    #[serde(default)]
    pub split: MLSplitSupportLimits,

    #[serde(default)]
    pub tanh: MLSingleInputSupportLimits,

    #[serde(default)]
    pub tile: MLSingleInputSupportLimits,

    #[serde(default)]
    pub transpose: MLSingleInputSupportLimits,

    #[serde(default)]
    pub triangular: MLSingleInputSupportLimits,

    #[serde(default)]
    pub r#where: MLWhereSupportLimits,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tensor_limits() -> MLTensorLimits {
        MLTensorLimits::default()
    }

    #[test]
    fn serde_ml_op_support_limits_roundtrip() {
        let limits = MLOpSupportLimits {
            preferred_input_layout: MLInputOperandLayout::Nchw,
            max_tensor_byte_length: 268_435_456,
            input: sample_tensor_limits(),
            constant: sample_tensor_limits(),
            output: sample_tensor_limits(),
            conv2d: MLConv2dSupportLimits {
                input: sample_tensor_limits(),
                filter: sample_tensor_limits(),
                bias: sample_tensor_limits(),
                output: sample_tensor_limits(),
            },
            ..Default::default()
        };

        let json = serde_json::to_string(&limits).unwrap();
        let back: MLOpSupportLimits = serde_json::from_str(&json).unwrap();
        assert_eq!(back.preferred_input_layout, MLInputOperandLayout::Nchw);
        assert_eq!(back.max_tensor_byte_length, 268_435_456);
    }
}

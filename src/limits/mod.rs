//! WebNN [`MLOpSupportLimits`](https://www.w3.org/TR/webnn/#dictdef-mlopsupportlimits) and related
//! limit dictionaries, for JSON interchange and alignment with `web-sys` / wasm-bindgen bindings.

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
    /// Channels-first layout (`NCHW`).
    Nchw,
    /// Channels-last layout (`NHWC`).
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
    /// Minimum supported rank.
    pub min: u32,
    /// Maximum supported rank.
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
    /// Supported data types.
    pub data_types: Vec<MLOperandDataType>, // use &[DataType] ?
    /// Minimum and maximum supported ranks.
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
    /// Preferred input layout for layout dependent operators like conv2d().
    pub preferred_input_layout: MLInputOperandLayout,
    /// The maximum supported length of tensors, in bytes.
    pub max_tensor_byte_length: u64,
    /// Support limits for input MLOperands for an MLGraph.
    pub input: MLTensorLimits,
    /// Support limits for constant MLOperands for an MLGraph.
    pub constant: MLTensorLimits,
    /// Support limits for output MLOperands for an MLGraph.
    pub output: MLTensorLimits,

    #[serde(default)]
    /// Support limits for operator `argMin()`.
    pub arg_min: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `argMax()`.
    pub arg_max: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `batchNormalization()`.
    pub batch_normalization: MLBatchNormalizationSupportLimits,
    #[serde(default)]
    /// Support limits for operator `cast()`.
    pub cast: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `clamp()`.
    pub clamp: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `concat()`.
    pub concat: MLConcatSupportLimits,
    #[serde(default)]
    /// Support limits for operator `conv2d()`.
    pub conv2d: MLConv2dSupportLimits,
    #[serde(default)]
    /// Support limits for operator `convTranspose2d()`.
    pub conv_transpose2d: MLConv2dSupportLimits,
    #[serde(default)]
    /// Support limits for operator `cumulativeSum()`.
    pub cumulative_sum: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `add()`.
    pub add: MLBinarySupportLimits,
    #[serde(default)]
    /// Support limits for operator `sub()`.
    pub sub: MLBinarySupportLimits,
    #[serde(default)]
    /// Support limits for operator `mul()`.
    pub mul: MLBinarySupportLimits,
    #[serde(default)]
    /// Support limits for operator `div()`.
    pub div: MLBinarySupportLimits,
    #[serde(default)]
    /// Support limits for operator `max()`.
    pub max: MLBinarySupportLimits,
    #[serde(default)]
    /// Support limits for operator `min()`.
    pub min: MLBinarySupportLimits,
    #[serde(default)]
    /// Support limits for operator `pow()`.
    pub pow: MLBinarySupportLimits,

    #[serde(default)]
    /// Support limits for operator `equal()`.
    pub equal: MLBinarySupportLimits,
    #[serde(default)]
    /// Support limits for operator `notEqual()`.
    pub not_equal: MLBinarySupportLimits,
    #[serde(default)]
    /// Support limits for operator `greater()`.
    pub greater: MLBinarySupportLimits,
    #[serde(default)]
    /// Support limits for operator `greaterOrEqual()`.
    pub greater_or_equal: MLBinarySupportLimits,
    #[serde(default)]
    /// Support limits for operator `lesser()`.
    pub lesser: MLBinarySupportLimits,
    #[serde(default)]
    /// Support limits for operator `lesserOrEqual()`.
    pub lesser_or_equal: MLBinarySupportLimits,
    #[serde(default)]
    /// Support limits for operator `logicalNot()`.
    pub logical_not: MLLogicalNotSupportLimits,
    #[serde(default)]
    /// Support limits for operator `logicalAnd()`.
    pub logical_and: MLBinarySupportLimits,
    #[serde(default)]
    /// Support limits for operator `logicalOr()`.
    pub logical_or: MLBinarySupportLimits,
    #[serde(default)]
    /// Support limits for operator `logicalXor()`.
    pub logical_xor: MLBinarySupportLimits,
    #[serde(default)]
    /// Support limits for operator `isNaN()`.
    pub is_na_n: MLLogicalNotSupportLimits,
    #[serde(default)]
    /// Support limits for operator `isInfinite()`.
    pub is_infinite: MLLogicalNotSupportLimits,

    #[serde(default)]
    /// Support limits for operator `abs()`.
    pub abs: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `ceil()`.
    pub ceil: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `cos()`.
    pub cos: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `erf()`.
    pub erf: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `exp()`.
    pub exp: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `floor()`.
    pub floor: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `identity()`.
    pub identity: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `log()`.
    pub log: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `neg()`.
    pub neg: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `reciprocal()`.
    pub reciprocal: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `roundEven()`.
    pub round_even: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `sin()`.
    pub sin: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `sign()`.
    pub sign: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `sqrt()`.
    pub sqrt: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `tan()`.
    pub tan: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `dequantizeLinear()`.
    pub dequantize_linear: MLQuantizeDequantizeLinearSupportLimits,
    #[serde(default)]
    /// Support limits for operator `quantizeLinear()`.
    pub quantize_linear: MLQuantizeDequantizeLinearSupportLimits,

    #[serde(default)]
    /// Support limits for operator `elu()`.
    pub elu: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `expand()`.
    pub expand: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `gather()`.
    pub gather: MLGatherSupportLimits,
    #[serde(default)]
    /// Support limits for operator `gatherElements()`.
    pub gather_elements: MLGatherSupportLimits,
    #[serde(default)]
    /// Support limits for operator `gatherND()`.
    pub gather_nd: MLGatherSupportLimits,

    #[serde(default)]
    /// Support limits for operator `gelu()`.
    pub gelu: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `gemm()`.
    pub gemm: MLGemmSupportLimits,

    #[serde(default)]
    /// Support limits for operator `gru()`.
    pub gru: MLGruSupportLimits,
    #[serde(default)]
    /// Support limits for operator `gruCell()`.
    pub gru_cell: MLGruCellSupportLimits,

    #[serde(default)]
    /// Support limits for operator `hardSigmoid()`.
    pub hard_sigmoid: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `hardSwish()`.
    pub hard_swish: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `instanceNormalization()`.
    pub instance_normalization: MLNormalizationSupportLimits,
    #[serde(default)]
    /// Support limits for operator `layerNormalization()`.
    pub layer_normalization: MLNormalizationSupportLimits,

    #[serde(default)]
    /// Support limits for operator `leakyRelu()`.
    pub leaky_relu: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `linear()`.
    pub linear: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `lstm()`.
    pub lstm: MLLstmSupportLimits,
    #[serde(default)]
    /// Support limits for operator `lstmCell()`.
    pub lstm_cell: MLLstmCellSupportLimits,

    #[serde(default)]
    /// Support limits for operator `matmul()`.
    pub matmul: MLBinarySupportLimits,

    #[serde(default)]
    /// Support limits for operator `pad()`.
    pub pad: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `averagePool2d()`.
    pub average_pool2d: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `l2Pool2d()`.
    pub l2_pool2d: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `maxPool2d()`.
    pub max_pool2d: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `prelu()`.
    pub prelu: MLPreluSupportLimits,

    #[serde(default)]
    /// Support limits for operator `reduceL1()`.
    pub reduce_l1: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `reduceL2()`.
    pub reduce_l2: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `reduceLogSum()`.
    pub reduce_log_sum: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `reduceLogSumExp()`.
    pub reduce_log_sum_exp: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `reduceMax()`.
    pub reduce_max: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `reduceMean()`.
    pub reduce_mean: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `reduceMin()`.
    pub reduce_min: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `reduceProduct()`.
    pub reduce_product: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `reduceSum()`.
    pub reduce_sum: MLSingleInputSupportLimits,
    #[serde(default)]
    /// Support limits for operator `reduceSumSquare()`.
    pub reduce_sum_square: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `relu()`.
    pub relu: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `resample2d()`.
    pub resample2d: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `reshape()`.
    pub reshape: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `reverse()`.
    pub reverse: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `scatterElements()`.
    pub scatter_elements: MLScatterSupportLimits,
    #[serde(default)]
    /// Support limits for operator `scatterND()`.
    pub scatter_nd: MLScatterSupportLimits,

    #[serde(default)]
    /// Support limits for operator `sigmoid()`.
    pub sigmoid: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `slice()`.
    pub slice: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `softmax()`.
    pub softmax: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `softplus()`.
    pub softplus: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `softsign()`.
    pub softsign: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `split()`.
    pub split: MLSplitSupportLimits,

    #[serde(default)]
    /// Support limits for operator `tanh()`.
    pub tanh: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `tile()`.
    pub tile: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `transpose()`.
    pub transpose: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `triangular()`.
    pub triangular: MLSingleInputSupportLimits,

    #[serde(default)]
    /// Support limits for operator `where()`.
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

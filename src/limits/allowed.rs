//! Shared specification-derived limits used by backend contexts by default.

use std::sync::LazyLock;

use super::*;
use crate::operation_data_types::{OperandName, OperationName};
use crate::operation_type_support::allowed_data_types;

fn tensor_limits(operation: OperationName, operand: OperandName) -> MLTensorLimits {
    MLTensorLimits {
        data_types: allowed_data_types(operation, operand).to_vec(),
        rank_range: MLRankRange::default(),
    }
}

macro_rules! impl_allowed_limits {
    ($dictionary:ident { $($field:ident: $operand:ident),+ $(,)? }) => {
        impl $dictionary {
            fn for_operation(operation: OperationName) -> Self {
                Self { $($field: tensor_limits(operation, OperandName::$operand)),+ }
            }

            fn operand_limits(&self, operand: OperandName) -> Option<&MLTensorLimits> {
                match operand {
                    $(OperandName::$operand => Some(&self.$field)),+,
                    _ => None,
                }
            }
        }
    };
}

impl_allowed_limits!(MLBatchNormalizationSupportLimits {
    input: Input,
    mean: Mean,
    variance: Variance,
    scale: Scale,
    bias: Bias,
    output: Output,
});

impl_allowed_limits!(MLBinarySupportLimits {
    a: A,
    b: B,
    output: Output,
});

impl_allowed_limits!(MLConcatSupportLimits {
    inputs: Inputs,
    output: Output,
});

impl_allowed_limits!(MLConv2dSupportLimits {
    input: Input,
    filter: Filter,
    bias: Bias,
    output: Output,
});

impl_allowed_limits!(MLGatherSupportLimits {
    input: Input,
    indices: Indices,
    output: Output,
});

impl_allowed_limits!(MLGemmSupportLimits {
    a: A,
    b: B,
    c: C,
    output: Output,
});

impl_allowed_limits!(MLGruCellSupportLimits {
    input: Input,
    weight: Weight,
    recurrent_weight: RecurrentWeight,
    hidden_state: HiddenState,
    bias: Bias,
    recurrent_bias: RecurrentBias,
    output: Output,
});

impl_allowed_limits!(MLGruSupportLimits {
    input: Input,
    weight: Weight,
    recurrent_weight: RecurrentWeight,
    bias: Bias,
    recurrent_bias: RecurrentBias,
    initial_hidden_state: InitialHiddenState,
    output0: Outputs0,
    output1: Outputs1,
});

impl_allowed_limits!(MLLogicalNotSupportLimits {
    a: A,
    output: Output,
});

impl_allowed_limits!(MLLstmCellSupportLimits {
    input: Input,
    weight: Weight,
    recurrent_weight: RecurrentWeight,
    hidden_state: HiddenState,
    cell_state: CellState,
    bias: Bias,
    recurrent_bias: RecurrentBias,
    peephole_weight: PeepholeWeight,
    output0: Outputs0,
    output1: Outputs1,
});

impl_allowed_limits!(MLLstmSupportLimits {
    input: Input,
    weight: Weight,
    recurrent_weight: RecurrentWeight,
    bias: Bias,
    recurrent_bias: RecurrentBias,
    peephole_weight: PeepholeWeight,
    initial_hidden_state: InitialHiddenState,
    initial_cell_state: InitialCellState,
    output0: Outputs0,
    output1: Outputs1,
    output2: Outputs2,
});

impl_allowed_limits!(MLNormalizationSupportLimits {
    input: Input,
    scale: Scale,
    bias: Bias,
    output: Output,
});

impl_allowed_limits!(MLPreluSupportLimits {
    input: Input,
    slope: Slope,
    output: Output,
});

impl_allowed_limits!(MLQuantizeDequantizeLinearSupportLimits {
    input: Input,
    scale: Scale,
    zero_point: ZeroPoint,
    output: Output,
});

impl_allowed_limits!(MLScatterSupportLimits {
    input: Input,
    indices: Indices,
    updates: Updates,
    output: Output,
});

impl_allowed_limits!(MLSingleInputSupportLimits {
    input: Input,
    output: Output,
});

impl_allowed_limits!(MLSplitSupportLimits {
    input: Input,
    outputs: Outputs,
});

impl_allowed_limits!(MLWhereSupportLimits {
    condition: Condition,
    true_value: TrueValue,
    false_value: FalseValue,
    output: Output,
});

/// Shared default limits: every generated allowed data type and default rank range.
///
/// Operand types come from [`allowed_data_types`], including resolved `SameAs`
/// domains. Ranks use [`MLRankRange::default`] rather than per-operation spec rank
/// constraints. Graph I/O uses [`MLTensorLimits::default`], the preferred layout
/// is [`MLInputOperandLayout::default`], and the byte limit is [`RUSTNN_MAX_TENSOR`].
///
/// Initialized once because the dictionaries own `Vec` type lists. All current
/// backend contexts borrow this same value. It describes the shared validation
/// envelope; individual backends may reject graphs at compilation or execution.
/// Context type-check bypass and packed integer options do not modify this value.
pub static EVERYTHING_THAT_IS_ALLOWED: LazyLock<MLOpSupportLimits> =
    LazyLock::new(|| MLOpSupportLimits {
        preferred_input_layout: MLInputOperandLayout::default(),
        max_tensor_byte_length: u64::from(RUSTNN_MAX_TENSOR),
        input: MLTensorLimits::default(),
        constant: MLTensorLimits::default(),
        output: MLTensorLimits::default(),
        arg_min: MLSingleInputSupportLimits::for_operation(OperationName::ArgMin),
        arg_max: MLSingleInputSupportLimits::for_operation(OperationName::ArgMax),
        batch_normalization: MLBatchNormalizationSupportLimits::for_operation(
            OperationName::BatchNormalization,
        ),
        cast: MLSingleInputSupportLimits::for_operation(OperationName::Cast),
        clamp: MLSingleInputSupportLimits::for_operation(OperationName::Clamp),
        concat: MLConcatSupportLimits::for_operation(OperationName::Concat),
        conv2d: MLConv2dSupportLimits::for_operation(OperationName::Conv2d),
        conv_transpose2d: MLConv2dSupportLimits::for_operation(OperationName::ConvTranspose2d),
        cumulative_sum: MLSingleInputSupportLimits::for_operation(OperationName::CumulativeSum),
        add: MLBinarySupportLimits::for_operation(OperationName::Add),
        sub: MLBinarySupportLimits::for_operation(OperationName::Sub),
        mul: MLBinarySupportLimits::for_operation(OperationName::Mul),
        div: MLBinarySupportLimits::for_operation(OperationName::Div),
        max: MLBinarySupportLimits::for_operation(OperationName::Max),
        min: MLBinarySupportLimits::for_operation(OperationName::Min),
        pow: MLBinarySupportLimits::for_operation(OperationName::Pow),
        equal: MLBinarySupportLimits::for_operation(OperationName::Equal),
        not_equal: MLBinarySupportLimits::for_operation(OperationName::NotEqual),
        greater: MLBinarySupportLimits::for_operation(OperationName::Greater),
        greater_or_equal: MLBinarySupportLimits::for_operation(OperationName::GreaterOrEqual),
        lesser: MLBinarySupportLimits::for_operation(OperationName::Lesser),
        lesser_or_equal: MLBinarySupportLimits::for_operation(OperationName::LesserOrEqual),
        logical_not: MLLogicalNotSupportLimits::for_operation(OperationName::LogicalNot),
        logical_and: MLBinarySupportLimits::for_operation(OperationName::LogicalAnd),
        logical_or: MLBinarySupportLimits::for_operation(OperationName::LogicalOr),
        logical_xor: MLBinarySupportLimits::for_operation(OperationName::LogicalXor),
        is_na_n: MLLogicalNotSupportLimits::for_operation(OperationName::IsNaN),
        is_infinite: MLLogicalNotSupportLimits::for_operation(OperationName::IsInfinite),
        abs: MLSingleInputSupportLimits::for_operation(OperationName::Abs),
        ceil: MLSingleInputSupportLimits::for_operation(OperationName::Ceil),
        cos: MLSingleInputSupportLimits::for_operation(OperationName::Cos),
        erf: MLSingleInputSupportLimits::for_operation(OperationName::Erf),
        exp: MLSingleInputSupportLimits::for_operation(OperationName::Exp),
        floor: MLSingleInputSupportLimits::for_operation(OperationName::Floor),
        identity: MLSingleInputSupportLimits::for_operation(OperationName::Identity),
        log: MLSingleInputSupportLimits::for_operation(OperationName::Log),
        neg: MLSingleInputSupportLimits::for_operation(OperationName::Neg),
        reciprocal: MLSingleInputSupportLimits::for_operation(OperationName::Reciprocal),
        round_even: MLSingleInputSupportLimits::for_operation(OperationName::RoundEven),
        sin: MLSingleInputSupportLimits::for_operation(OperationName::Sin),
        sign: MLSingleInputSupportLimits::for_operation(OperationName::Sign),
        sqrt: MLSingleInputSupportLimits::for_operation(OperationName::Sqrt),
        tan: MLSingleInputSupportLimits::for_operation(OperationName::Tan),
        dequantize_linear: MLQuantizeDequantizeLinearSupportLimits::for_operation(
            OperationName::DequantizeLinear,
        ),
        quantize_linear: MLQuantizeDequantizeLinearSupportLimits::for_operation(
            OperationName::QuantizeLinear,
        ),
        elu: MLSingleInputSupportLimits::for_operation(OperationName::Elu),
        expand: MLSingleInputSupportLimits::for_operation(OperationName::Expand),
        gather: MLGatherSupportLimits::for_operation(OperationName::Gather),
        gather_elements: MLGatherSupportLimits::for_operation(OperationName::GatherElements),
        gather_nd: MLGatherSupportLimits::for_operation(OperationName::GatherND),
        gelu: MLSingleInputSupportLimits::for_operation(OperationName::Gelu),
        gemm: MLGemmSupportLimits::for_operation(OperationName::Gemm),
        gru: MLGruSupportLimits::for_operation(OperationName::Gru),
        gru_cell: MLGruCellSupportLimits::for_operation(OperationName::GruCell),
        hard_sigmoid: MLSingleInputSupportLimits::for_operation(OperationName::HardSigmoid),
        hard_swish: MLSingleInputSupportLimits::for_operation(OperationName::HardSwish),
        instance_normalization: MLNormalizationSupportLimits::for_operation(
            OperationName::InstanceNormalization,
        ),
        layer_normalization: MLNormalizationSupportLimits::for_operation(
            OperationName::LayerNormalization,
        ),
        leaky_relu: MLSingleInputSupportLimits::for_operation(OperationName::LeakyRelu),
        linear: MLSingleInputSupportLimits::for_operation(OperationName::Linear),
        lstm: MLLstmSupportLimits::for_operation(OperationName::Lstm),
        lstm_cell: MLLstmCellSupportLimits::for_operation(OperationName::LstmCell),
        matmul: MLBinarySupportLimits::for_operation(OperationName::Matmul),
        pad: MLSingleInputSupportLimits::for_operation(OperationName::Pad),
        average_pool2d: MLSingleInputSupportLimits::for_operation(OperationName::AveragePool2d),
        l2_pool2d: MLSingleInputSupportLimits::for_operation(OperationName::L2Pool2d),
        max_pool2d: MLSingleInputSupportLimits::for_operation(OperationName::MaxPool2d),
        prelu: MLPreluSupportLimits::for_operation(OperationName::Prelu),
        reduce_l1: MLSingleInputSupportLimits::for_operation(OperationName::ReduceL1),
        reduce_l2: MLSingleInputSupportLimits::for_operation(OperationName::ReduceL2),
        reduce_log_sum: MLSingleInputSupportLimits::for_operation(OperationName::ReduceLogSum),
        reduce_log_sum_exp: MLSingleInputSupportLimits::for_operation(
            OperationName::ReduceLogSumExp,
        ),
        reduce_max: MLSingleInputSupportLimits::for_operation(OperationName::ReduceMax),
        reduce_mean: MLSingleInputSupportLimits::for_operation(OperationName::ReduceMean),
        reduce_min: MLSingleInputSupportLimits::for_operation(OperationName::ReduceMin),
        reduce_product: MLSingleInputSupportLimits::for_operation(OperationName::ReduceProduct),
        reduce_sum: MLSingleInputSupportLimits::for_operation(OperationName::ReduceSum),
        reduce_sum_square: MLSingleInputSupportLimits::for_operation(
            OperationName::ReduceSumSquare,
        ),
        relu: MLSingleInputSupportLimits::for_operation(OperationName::Relu),
        resample2d: MLSingleInputSupportLimits::for_operation(OperationName::Resample2d),
        reshape: MLSingleInputSupportLimits::for_operation(OperationName::Reshape),
        reverse: MLSingleInputSupportLimits::for_operation(OperationName::Reverse),
        scatter_elements: MLScatterSupportLimits::for_operation(OperationName::ScatterElements),
        scatter_nd: MLScatterSupportLimits::for_operation(OperationName::ScatterND),
        sigmoid: MLSingleInputSupportLimits::for_operation(OperationName::Sigmoid),
        slice: MLSingleInputSupportLimits::for_operation(OperationName::Slice),
        softmax: MLSingleInputSupportLimits::for_operation(OperationName::Softmax),
        softplus: MLSingleInputSupportLimits::for_operation(OperationName::Softplus),
        softsign: MLSingleInputSupportLimits::for_operation(OperationName::Softsign),
        split: MLSplitSupportLimits::for_operation(OperationName::Split),
        tanh: MLSingleInputSupportLimits::for_operation(OperationName::Tanh),
        tile: MLSingleInputSupportLimits::for_operation(OperationName::Tile),
        transpose: MLSingleInputSupportLimits::for_operation(OperationName::Transpose),
        triangular: MLSingleInputSupportLimits::for_operation(OperationName::Triangular),
        r#where: MLWhereSupportLimits::for_operation(OperationName::Where),
    });

impl MLOpSupportLimits {
    pub(crate) fn operand_limits(
        &self,
        operation: OperationName,
        operand: OperandName,
    ) -> Option<&MLTensorLimits> {
        match operation {
            OperationName::ArgMin => self.arg_min.operand_limits(operand),
            OperationName::ArgMax => self.arg_max.operand_limits(operand),
            OperationName::BatchNormalization => self.batch_normalization.operand_limits(operand),
            OperationName::Cast => self.cast.operand_limits(operand),
            OperationName::Clamp => self.clamp.operand_limits(operand),
            OperationName::Concat => self.concat.operand_limits(operand),
            OperationName::Conv2d => self.conv2d.operand_limits(operand),
            OperationName::ConvTranspose2d => self.conv_transpose2d.operand_limits(operand),
            OperationName::CumulativeSum => self.cumulative_sum.operand_limits(operand),
            OperationName::Add => self.add.operand_limits(operand),
            OperationName::Sub => self.sub.operand_limits(operand),
            OperationName::Mul => self.mul.operand_limits(operand),
            OperationName::Div => self.div.operand_limits(operand),
            OperationName::Max => self.max.operand_limits(operand),
            OperationName::Min => self.min.operand_limits(operand),
            OperationName::Pow => self.pow.operand_limits(operand),
            OperationName::Equal => self.equal.operand_limits(operand),
            OperationName::NotEqual => self.not_equal.operand_limits(operand),
            OperationName::Greater => self.greater.operand_limits(operand),
            OperationName::GreaterOrEqual => self.greater_or_equal.operand_limits(operand),
            OperationName::Lesser => self.lesser.operand_limits(operand),
            OperationName::LesserOrEqual => self.lesser_or_equal.operand_limits(operand),
            OperationName::LogicalNot => self.logical_not.operand_limits(operand),
            OperationName::LogicalAnd => self.logical_and.operand_limits(operand),
            OperationName::LogicalOr => self.logical_or.operand_limits(operand),
            OperationName::LogicalXor => self.logical_xor.operand_limits(operand),
            OperationName::IsNaN => self.is_na_n.operand_limits(operand),
            OperationName::IsInfinite => self.is_infinite.operand_limits(operand),
            OperationName::Abs => self.abs.operand_limits(operand),
            OperationName::Ceil => self.ceil.operand_limits(operand),
            OperationName::Cos => self.cos.operand_limits(operand),
            OperationName::Erf => self.erf.operand_limits(operand),
            OperationName::Exp => self.exp.operand_limits(operand),
            OperationName::Floor => self.floor.operand_limits(operand),
            OperationName::Identity => self.identity.operand_limits(operand),
            OperationName::Log => self.log.operand_limits(operand),
            OperationName::Neg => self.neg.operand_limits(operand),
            OperationName::Reciprocal => self.reciprocal.operand_limits(operand),
            OperationName::RoundEven => self.round_even.operand_limits(operand),
            OperationName::Sin => self.sin.operand_limits(operand),
            OperationName::Sign => self.sign.operand_limits(operand),
            OperationName::Sqrt => self.sqrt.operand_limits(operand),
            OperationName::Tan => self.tan.operand_limits(operand),
            OperationName::DequantizeLinear => self.dequantize_linear.operand_limits(operand),
            OperationName::QuantizeLinear => self.quantize_linear.operand_limits(operand),
            OperationName::Elu => self.elu.operand_limits(operand),
            OperationName::Expand => self.expand.operand_limits(operand),
            OperationName::Gather => self.gather.operand_limits(operand),
            OperationName::GatherElements => self.gather_elements.operand_limits(operand),
            OperationName::GatherND => self.gather_nd.operand_limits(operand),
            OperationName::Gelu => self.gelu.operand_limits(operand),
            OperationName::Gemm => self.gemm.operand_limits(operand),
            OperationName::Gru => self.gru.operand_limits(operand),
            OperationName::GruCell => self.gru_cell.operand_limits(operand),
            OperationName::HardSigmoid => self.hard_sigmoid.operand_limits(operand),
            OperationName::HardSwish => self.hard_swish.operand_limits(operand),
            OperationName::InstanceNormalization => {
                self.instance_normalization.operand_limits(operand)
            }
            OperationName::LayerNormalization => self.layer_normalization.operand_limits(operand),
            OperationName::LeakyRelu => self.leaky_relu.operand_limits(operand),
            OperationName::Linear => self.linear.operand_limits(operand),
            OperationName::Lstm => self.lstm.operand_limits(operand),
            OperationName::LstmCell => self.lstm_cell.operand_limits(operand),
            OperationName::Matmul => self.matmul.operand_limits(operand),
            OperationName::Pad => self.pad.operand_limits(operand),
            OperationName::AveragePool2d => self.average_pool2d.operand_limits(operand),
            OperationName::L2Pool2d => self.l2_pool2d.operand_limits(operand),
            OperationName::MaxPool2d => self.max_pool2d.operand_limits(operand),
            OperationName::Prelu => self.prelu.operand_limits(operand),
            OperationName::ReduceL1 => self.reduce_l1.operand_limits(operand),
            OperationName::ReduceL2 => self.reduce_l2.operand_limits(operand),
            OperationName::ReduceLogSum => self.reduce_log_sum.operand_limits(operand),
            OperationName::ReduceLogSumExp => self.reduce_log_sum_exp.operand_limits(operand),
            OperationName::ReduceMax => self.reduce_max.operand_limits(operand),
            OperationName::ReduceMean => self.reduce_mean.operand_limits(operand),
            OperationName::ReduceMin => self.reduce_min.operand_limits(operand),
            OperationName::ReduceProduct => self.reduce_product.operand_limits(operand),
            OperationName::ReduceSum => self.reduce_sum.operand_limits(operand),
            OperationName::ReduceSumSquare => self.reduce_sum_square.operand_limits(operand),
            OperationName::Relu => self.relu.operand_limits(operand),
            OperationName::Resample2d => self.resample2d.operand_limits(operand),
            OperationName::Reshape => self.reshape.operand_limits(operand),
            OperationName::Reverse => self.reverse.operand_limits(operand),
            OperationName::ScatterElements => self.scatter_elements.operand_limits(operand),
            OperationName::ScatterND => self.scatter_nd.operand_limits(operand),
            OperationName::Sigmoid => self.sigmoid.operand_limits(operand),
            OperationName::Slice => self.slice.operand_limits(operand),
            OperationName::Softmax => self.softmax.operand_limits(operand),
            OperationName::Softplus => self.softplus.operand_limits(operand),
            OperationName::Softsign => self.softsign.operand_limits(operand),
            OperationName::Split => self.split.operand_limits(operand),
            OperationName::Tanh => self.tanh.operand_limits(operand),
            OperationName::Tile => self.tile.operand_limits(operand),
            OperationName::Transpose => self.transpose.operand_limits(operand),
            OperationName::Triangular => self.triangular.operand_limits(operand),
            OperationName::Where => self.r#where.operand_limits(operand),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operand_lookup_covers_every_generated_rule() {
        for rule in crate::operation_data_types::OPERATION_DATA_TYPES {
            let tensor = EVERYTHING_THAT_IS_ALLOWED
                .operand_limits(rule.operation, rule.operand)
                .unwrap_or_else(|| {
                    panic!(
                        "missing backend limits for {}.{}",
                        rule.operation, rule.operand
                    )
                });
            assert_eq!(
                tensor.data_types,
                allowed_data_types(rule.operation, rule.operand)
            );
        }
    }
}

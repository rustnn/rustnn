//! Specification operand type validation over generated WebNN rules.
//!
//! Checks explicit type domains and actual `SameAs` relationships, independently
//! of backend capabilities. Context options can bypass these checks with warnings.

use crate::error::AllowedDataTypesError;
use crate::graph::{GraphInfo, OperandDescriptor};
use crate::operation_data_types::{
    ALL_SPEC_DATA_TYPES, AllowedDataTypes, OPERATION_DATA_TYPES, OperandName, OperationName,
};
use crate::operator_enums::MLOperandDataType;
use crate::operators::Operation;

/// Allowed types for an operation and operand, evaluated in const contexts too.
///
/// Resolves `SameAs` chains within the operation and expands `Any` to
/// [`ALL_SPEC_DATA_TYPES`]. An absent operand rule, an empty explicit list, or a cycle
/// returns an empty slice. Optional operands and outputs apply when present.
/// This returns a type domain; callers must separately enforce equal actual
/// types for operands linked by `SameAs`.
///
/// ```
/// use rustnn::operation_data_types::{OperandName, OperationName};
/// use rustnn::operation_type_support::allowed_data_types;
/// use rustnn::operator_enums::MLOperandDataType;
///
/// const TYPES: &[MLOperandDataType] =
///     allowed_data_types(OperationName::ArgMax, OperandName::Output);
/// assert_eq!(TYPES, &[MLOperandDataType::Int32, MLOperandDataType::Int64]);
/// ```
pub const fn allowed_data_types(
    operation: OperationName,
    mut operand: OperandName,
) -> &'static [MLOperandDataType] {
    // Bounded traversal also rejects any cycles in future generated metadata.
    let mut remaining = OPERATION_DATA_TYPES.len();
    while remaining > 0 {
        let mut index = 0;
        let mut next_operand = None;
        while index < OPERATION_DATA_TYPES.len() {
            let rule = &OPERATION_DATA_TYPES[index];
            // Derived PartialEq is not const; both enums contain unit variants.
            if rule.operation as usize == operation as usize
                && rule.operand as usize == operand as usize
            {
                match rule.allowed {
                    AllowedDataTypes::Any => return ALL_SPEC_DATA_TYPES,
                    AllowedDataTypes::Types(types) => return types,
                    AllowedDataTypes::SameAs(reference) => next_operand = Some(reference),
                }
                break;
            }
            index += 1;
        }
        match next_operand {
            Some(reference) => operand = reference,
            None => return &[],
        }
        remaining -= 1;
    }
    &[]
}

/// Check every present input, option operand, and recorded output of an operation.
///
/// `GraphInfo` resolves the operand IDs stored by `Operation`. A violation reports
/// operation, operand, allowed types, and actual type through [`AllowedDataTypesError`].
/// `SameAs` rules compare actual operand types, including every sequence item.
/// Optional quantization zero points default to uint8 when absent.
///
/// `by_pass_check` skips type validation and logs a warning on every call. Shape
/// and reference validation elsewhere remain active. Emulated global pooling and
/// squeeze/unsqueeze use the corresponding pool/reshape rules. The rustnn `shape`
/// extension accepts every represented input type and produces int64.
/// Prospective outputs are checked by the shared inference path before recording.
pub fn check_allowed_data_types(
    operation: &Operation,
    graph: &GraphInfo,
    by_pass_check: bool,
) -> Result<(), AllowedDataTypesError> {
    check_allowed_data_types_with_policy(operation, graph, by_pass_check, false)
}

/// Check an operation using the context's bypass and packed 4-bit policies.
///
/// When enabled, int4/uint4 are accepted wherever the corresponding int8/uint8
/// type is allowed. Other domains and actual `SameAs` relationships still apply.
pub fn check_allowed_data_types_with_options(
    operation: &Operation,
    graph: &GraphInfo,
    options: &crate::mlcontextoptions::MLContextOptions,
) -> Result<(), AllowedDataTypesError> {
    check_allowed_data_types_with_policy(
        operation,
        graph,
        options.bypass_allowed_data_types_checks,
        options.allow_int4_uint4,
    )
}

pub(crate) fn check_allowed_data_types_with_policy(
    operation: &Operation,
    graph: &GraphInfo,
    by_pass_check: bool,
    allow_int4_uint4: bool,
) -> Result<(), AllowedDataTypesError> {
    if by_pass_check {
        log::warn!(
            "Bypassing allowed data type checks for operation {}",
            operation.op_type()
        );
        return Ok(());
    }
    let Some(operation_name) = spec_operation_name(operation) else {
        return Ok(());
    };
    let mut types = input_data_types(operation, operation_name, graph)?;
    for (index, id) in operation.output_operands().iter().enumerate() {
        if let Some(operand) = graph.operands.get(*id as usize) {
            types.push((
                output_name(operation, index),
                ml_data_type(operand.descriptor.data_type),
            ));
        }
    }
    check_types(operation_name, &types, allow_int4_uint4)
}

pub(crate) fn check_output_data_types(
    operation: &Operation,
    graph: &GraphInfo,
    outputs: &[OperandDescriptor],
    allow_int4_uint4: bool,
) -> Result<(), AllowedDataTypesError> {
    let Some(operation_name) = spec_operation_name(operation) else {
        return Ok(());
    };
    let mut types = input_data_types(operation, operation_name, graph)?;
    types.extend(outputs.iter().enumerate().map(|(index, output)| {
        (
            output_name(operation, index),
            ml_data_type(output.data_type),
        )
    }));
    check_types(operation_name, &types, allow_int4_uint4)
}

fn ml_data_type(data_type: crate::graph::DataType) -> MLOperandDataType {
    MLOperandDataType::try_from(data_type).expect("every graph type has an API data type")
}

fn spec_operation_name(operation: &Operation) -> Option<OperationName> {
    match operation {
        // TODO: get rid of those non-spec operations or find a proper way to handling those
        Operation::Squeeze { .. } | Operation::Unsqueeze { .. } => Some(OperationName::Reshape),
        Operation::Shape { .. } => None,
        _ => Some(
            operation
                .op_type()
                .parse()
                .expect("operation has generated spec type rules"),
        ),
    }
}

fn output_name(operation: &Operation, index: usize) -> OperandName {
    match operation {
        Operation::Split { .. } => OperandName::Outputs,
        Operation::Gru { .. } | Operation::Lstm { .. } | Operation::LstmCell { .. } => [
            OperandName::Outputs0,
            OperandName::Outputs1,
            OperandName::Outputs2,
        ]
        .get(index)
        .copied()
        .unwrap_or(OperandName::Outputs),
        _ => OperandName::Output,
    }
}

fn input_data_types(
    operation: &Operation,
    operation_name: OperationName,
    graph: &GraphInfo,
) -> Result<Vec<(OperandName, MLOperandDataType)>, AllowedDataTypesError> {
    let mut types = Vec::new();
    for (operand_name, operand_id) in named_input_operands(operation) {
        let operand = graph.operands.get(operand_id as usize).ok_or(
            AllowedDataTypesError::MissingOperand {
                operation_name,
                operand_name,
                operand_id,
            },
        )?;
        types.push((operand_name, ml_data_type(operand.descriptor.data_type)));
    }
    if matches!(
        operation,
        Operation::QuantizeLinear {
            zero_point: None,
            ..
        }
    ) {
        types.push((OperandName::ZeroPoint, MLOperandDataType::Uint8));
    }
    Ok(types)
}

fn check_types(
    operation_name: OperationName,
    types: &[(OperandName, MLOperandDataType)],
    allow_int4_uint4: bool,
) -> Result<(), AllowedDataTypesError> {
    for &(operand_name, used_dtype) in types {
        let allowed = allowed_data_types(operation_name, operand_name);
        // TODO: remove allow_int4_uint4 and int4->int8, uint4->uint8 promotion once
        // int4/uint4 are a regular part of the standard
        let packed_type_allowed = allow_int4_uint4
            && match used_dtype {
                MLOperandDataType::Int4 => allowed.contains(&MLOperandDataType::Int8),
                MLOperandDataType::Uint4 => allowed.contains(&MLOperandDataType::Uint8),
                _ => false,
            };
        if !allowed.contains(&used_dtype) && !packed_type_allowed {
            let mut allowed_dtype = allowed.to_vec();
            if allow_int4_uint4 {
                for (packed, integer) in [
                    (MLOperandDataType::Int4, MLOperandDataType::Int8),
                    (MLOperandDataType::Uint4, MLOperandDataType::Uint8),
                ] {
                    if allowed.contains(&integer) && !allowed_dtype.contains(&packed) {
                        allowed_dtype.push(packed);
                    }
                }
            }
            return Err(AllowedDataTypesError::NotAllowed {
                operation_name,
                operand_name,
                allowed_dtype,
                used_dtype,
            });
        }
        if let Some(rule) = OPERATION_DATA_TYPES
            .iter()
            .find(|rule| rule.operation == operation_name && rule.operand == operand_name)
            && let AllowedDataTypes::SameAs(reference) = rule.allowed
        {
            for &(_, expected) in types.iter().filter(|(name, _)| *name == reference) {
                if used_dtype != expected {
                    return Err(AllowedDataTypesError::NotAllowed {
                        operation_name,
                        operand_name,
                        allowed_dtype: vec![expected],
                        used_dtype,
                    });
                }
            }
        }
    }
    Ok(())
}

// Exhaustive role mapping keeps positional and option operands tied to typed fields.
pub(crate) fn named_input_operands(operation: &Operation) -> Vec<(OperandName, u32)> {
    let mut operands = match operation {
        Operation::Add { a, b, .. }
        | Operation::Sub { a, b, .. }
        | Operation::Mul { a, b, .. }
        | Operation::Div { a, b, .. }
        | Operation::Pow { a, b, .. }
        | Operation::Max { a, b, .. }
        | Operation::Min { a, b, .. }
        | Operation::Matmul { a, b, .. }
        | Operation::Equal { a, b, .. }
        | Operation::Greater { a, b, .. }
        | Operation::GreaterOrEqual { a, b, .. }
        | Operation::Lesser { a, b, .. }
        | Operation::LesserOrEqual { a, b, .. }
        | Operation::NotEqual { a, b, .. }
        | Operation::LogicalAnd { a, b, .. }
        | Operation::LogicalOr { a, b, .. }
        | Operation::LogicalXor { a, b, .. }
        | Operation::Gemm { a, b, .. } => vec![(OperandName::A, *a), (OperandName::B, *b)],

        Operation::Abs { input, .. }
        | Operation::Ceil { input, .. }
        | Operation::Cos { input, .. }
        | Operation::Exp { input, .. }
        | Operation::Floor { input, .. }
        | Operation::Log { input, .. }
        | Operation::Neg { input, .. }
        | Operation::Relu { input, .. }
        | Operation::Sigmoid { input, .. }
        | Operation::Sin { input, .. }
        | Operation::Sqrt { input, .. }
        | Operation::Tan { input, .. }
        | Operation::Tanh { input, .. }
        | Operation::Erf { input, .. }
        | Operation::Reciprocal { input, .. }
        | Operation::Sign { input, .. }
        | Operation::Identity { input, .. }
        | Operation::ArgMin { input, .. }
        | Operation::ArgMax { input, .. }
        | Operation::Cast { input, .. }
        | Operation::Clamp { input, .. }
        | Operation::CumulativeSum { input, .. }
        | Operation::Expand { input, .. }
        | Operation::Elu { input, .. }
        | Operation::HardSigmoid { input, .. }
        | Operation::HardSwish { input, .. }
        | Operation::InstanceNormalization { input, .. }
        | Operation::LayerNormalization { input, .. }
        | Operation::LeakyRelu { input, .. }
        | Operation::Linear { input, .. }
        | Operation::Pad { input, .. }
        | Operation::AveragePool2d { input, .. }
        | Operation::MaxPool2d { input, .. }
        | Operation::L2Pool2d { input, .. }
        | Operation::GlobalAveragePool { input, .. }
        | Operation::GlobalMaxPool { input, .. }
        | Operation::ReduceSum { input, .. }
        | Operation::ReduceMean { input, .. }
        | Operation::ReduceMax { input, .. }
        | Operation::ReduceMin { input, .. }
        | Operation::ReduceProduct { input, .. }
        | Operation::ReduceL1 { input, .. }
        | Operation::ReduceL2 { input, .. }
        | Operation::ReduceLogSum { input, .. }
        | Operation::ReduceLogSumExp { input, .. }
        | Operation::ReduceSumSquare { input, .. }
        | Operation::Reshape { input, .. }
        | Operation::Resample2d { input, .. }
        | Operation::Reverse { input, .. }
        | Operation::Softmax { input, .. }
        | Operation::Slice { input, .. }
        | Operation::Split { input, .. }
        | Operation::Transpose { input, .. }
        | Operation::Squeeze { input, .. }
        | Operation::Unsqueeze { input, .. }
        | Operation::Tile { input, .. }
        | Operation::Triangular { input, .. }
        | Operation::Softplus { input, .. }
        | Operation::Softsign { input, .. }
        | Operation::Gelu { input, .. }
        | Operation::Shape { input, .. }
        | Operation::RoundEven { input, .. } => vec![(OperandName::Input, *input)],

        Operation::LogicalNot { input, .. }
        | Operation::IsNaN { input, .. }
        | Operation::IsInfinite { input, .. } => vec![(OperandName::A, *input)],

        Operation::Where {
            condition,
            true_value,
            false_value,
            ..
        } => vec![
            (OperandName::Condition, *condition),
            (OperandName::TrueValue, *true_value),
            (OperandName::FalseValue, *false_value),
        ],

        Operation::BatchNormalization {
            input,
            mean,
            variance,
            ..
        } => vec![
            (OperandName::Input, *input),
            (OperandName::Mean, *mean),
            (OperandName::Variance, *variance),
        ],

        Operation::Conv2d { input, filter, .. }
        | Operation::ConvTranspose2d { input, filter, .. } => {
            vec![(OperandName::Input, *input), (OperandName::Filter, *filter)]
        }

        Operation::Concat { inputs, .. } => {
            inputs.iter().map(|id| (OperandName::Inputs, *id)).collect()
        }

        Operation::Gather { input, indices, .. }
        | Operation::GatherElements { input, indices, .. }
        | Operation::GatherND { input, indices, .. } => vec![
            (OperandName::Input, *input),
            (OperandName::Indices, *indices),
        ],

        Operation::Gru {
            input,
            weight,
            recurrence,
            ..
        }
        | Operation::Lstm {
            input,
            weight,
            recurrence,
            ..
        } => vec![
            (OperandName::Input, *input),
            (OperandName::Weight, *weight),
            (OperandName::RecurrentWeight, *recurrence),
        ],

        Operation::GruCell {
            input,
            weight,
            recurrence,
            hidden_state,
            ..
        } => vec![
            (OperandName::Input, *input),
            (OperandName::Weight, *weight),
            (OperandName::RecurrentWeight, *recurrence),
            (OperandName::HiddenState, *hidden_state),
        ],

        Operation::LstmCell {
            input,
            weight,
            recurrence,
            hidden_state,
            cell_state,
            ..
        } => vec![
            (OperandName::Input, *input),
            (OperandName::Weight, *weight),
            (OperandName::RecurrentWeight, *recurrence),
            (OperandName::HiddenState, *hidden_state),
            (OperandName::CellState, *cell_state),
        ],

        Operation::ScatterElements {
            input,
            indices,
            updates,
            ..
        }
        | Operation::ScatterND {
            input,
            indices,
            updates,
            ..
        } => vec![
            (OperandName::Input, *input),
            (OperandName::Indices, *indices),
            (OperandName::Updates, *updates),
        ],

        Operation::Prelu { input, slope, .. } => {
            vec![(OperandName::Input, *input), (OperandName::Slope, *slope)]
        }

        Operation::QuantizeLinear {
            input,
            scale,
            zero_point,
            ..
        }
        | Operation::DequantizeLinear {
            input,
            scale,
            zero_point,
            ..
        } => {
            let mut values = vec![(OperandName::Input, *input), (OperandName::Scale, *scale)];
            if let Some(id) = zero_point {
                values.push((OperandName::ZeroPoint, *id));
            }
            values
        }
    };
    match operation {
        Operation::BatchNormalization {
            options: Some(options),
            ..
        } => {
            if let Some(id) = options.scale {
                operands.push((OperandName::Scale, id));
            }
            if let Some(id) = options.bias {
                operands.push((OperandName::Bias, id));
            }
        }
        Operation::Conv2d {
            options: Some(options),
            ..
        } => {
            if let Some(id) = options.bias {
                operands.push((OperandName::Bias, id));
            }
        }
        Operation::ConvTranspose2d {
            options: Some(options),
            ..
        } => {
            if let Some(id) = options.bias {
                operands.push((OperandName::Bias, id));
            }
        }
        Operation::Gemm {
            options: Some(options),
            ..
        } => {
            if let Some(id) = options.c {
                operands.push((OperandName::C, id));
            }
        }
        Operation::Gru {
            options: Some(options),
            ..
        } => {
            if let Some(id) = options.bias {
                operands.push((OperandName::Bias, id));
            }
            if let Some(id) = options.recurrent_bias {
                operands.push((OperandName::RecurrentBias, id));
            }
            if let Some(id) = options.initial_hidden_state {
                operands.push((OperandName::InitialHiddenState, id));
            }
        }
        Operation::GruCell {
            options: Some(options),
            ..
        } => {
            if let Some(id) = options.bias {
                operands.push((OperandName::Bias, id));
            }
            if let Some(id) = options.recurrent_bias {
                operands.push((OperandName::RecurrentBias, id));
            }
        }
        Operation::InstanceNormalization {
            options: Some(options),
            ..
        } => {
            if let Some(id) = options.scale {
                operands.push((OperandName::Scale, id));
            }
            if let Some(id) = options.bias {
                operands.push((OperandName::Bias, id));
            }
        }
        Operation::LayerNormalization {
            options: Some(options),
            ..
        } => {
            if let Some(id) = options.scale {
                operands.push((OperandName::Scale, id));
            }
            if let Some(id) = options.bias {
                operands.push((OperandName::Bias, id));
            }
        }
        Operation::Lstm {
            options: Some(options),
            ..
        } => {
            if let Some(id) = options.bias {
                operands.push((OperandName::Bias, id));
            }
            if let Some(id) = options.recurrent_bias {
                operands.push((OperandName::RecurrentBias, id));
            }
            if let Some(id) = options.peephole_weight {
                operands.push((OperandName::PeepholeWeight, id));
            }
            if let Some(id) = options.initial_hidden_state {
                operands.push((OperandName::InitialHiddenState, id));
            }
            if let Some(id) = options.initial_cell_state {
                operands.push((OperandName::InitialCellState, id));
            }
        }
        Operation::LstmCell {
            options: Some(options),
            ..
        } => {
            if let Some(id) = options.bias {
                operands.push((OperandName::Bias, id));
            }
            if let Some(id) = options.recurrent_bias {
                operands.push((OperandName::RecurrentBias, id));
            }
            if let Some(id) = options.peephole_weight {
                operands.push((OperandName::PeepholeWeight, id));
            }
        }
        _ => {}
    }
    operands
}

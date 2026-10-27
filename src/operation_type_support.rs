//! Type-domain queries over the generated WebNN operation rules.
//!
//! These helpers check the specification's allowed types represented by rustnn,
//! independently of backend capabilities. They do not compare actual operand
//! types to enforce `SameAs` relationships or perform graph validation.

use crate::operation_data_types::{
    ALL_DATA_TYPES, AllowedDataTypes, OPERATION_DATA_TYPES, OperandName, OperationName,
};
use crate::operator_enums::MLOperandDataType;
use crate::operators::Operation;

/// Allowed types for an operation and operand, evaluated in const contexts too.
///
/// Resolves `SameAs` chains within the operation and expands `Any` to
/// [`ALL_DATA_TYPES`]. An absent operand rule, an empty explicit list, or a cycle
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
                    AllowedDataTypes::Any => return ALL_DATA_TYPES,
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

/// Whether `data_type` belongs to the allowed domain of `operation`'s `operand`.
///
/// Uses [`Operation::op_type`] to identify the recorded operation. Operations
/// absent from the specification tables (including rustnn's squeeze/unsqueeze
/// extensions) and absent operand rules return `false`.
///
/// Setting `bypass` returns `true` unconditionally and emits a [`log::warn!`]
/// record on every call, even for types that would otherwise be allowed.
/// This helper does not change builder or graph validation globally, check
/// backend support, or compare the actual types of `SameAs` operands.
pub fn is_data_type_allowed(
    operation: &Operation,
    operand: OperandName,
    data_type: MLOperandDataType,
    bypass: bool,
) -> bool {
    if bypass {
        log::warn!(
            "Bypassing allowed data type check for operation {}, operand {}, data type {}",
            operation.op_type(),
            operand,
            data_type.as_ref(),
        );
        return true;
    }
    let Ok(operation) = operation.op_type().parse::<OperationName>() else {
        return false;
    };
    allowed_data_types(operation, operand).contains(&data_type)
}

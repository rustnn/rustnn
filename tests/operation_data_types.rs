use rustnn::operation_data_types::{
    AllowedDataTypes, OPERATION_DATA_TYPES, OperandName, OperationName,
};
use strum::IntoEnumIterator;

#[test]
fn operation_names_round_trip() {
    for operation in OperationName::iter() {
        let name: &'static str = operation.into();
        assert_eq!(name.parse::<OperationName>(), Ok(operation));
        assert_eq!(operation.as_ref(), name);
        assert_eq!(operation.to_string(), name);
    }
    assert_eq!(OperationName::IsNaN.to_string(), "isNaN");
    assert_eq!(OperationName::GatherND.to_string(), "gatherND");
    assert_eq!(OperationName::Conv2d.to_string(), "conv2d");
    assert!("unknown".parse::<OperationName>().is_err());
    assert!("IsNaN".parse::<OperationName>().is_err());
}

#[test]
fn operand_names_round_trip() {
    for operand in OperandName::iter() {
        let name: &'static str = operand.into();
        assert_eq!(name.parse::<OperandName>(), Ok(operand));
        assert_eq!(operand.as_ref(), name);
        assert_eq!(operand.to_string(), name);
    }
    assert_eq!(OperandName::Outputs0.to_string(), "outputs[0]");
    assert_eq!(OperandName::ZeroPoint.to_string(), "zeroPoint");
    assert!("unknown".parse::<OperandName>().is_err());
    assert!("outputs0".parse::<OperandName>().is_err());
}

#[test]
fn same_type_rules_reference_an_operand_of_the_same_operation() {
    for rule in OPERATION_DATA_TYPES {
        if let AllowedDataTypes::SameAs(operand) = rule.allowed {
            assert!(OPERATION_DATA_TYPES.iter().any(|candidate| {
                candidate.operation == rule.operation && candidate.operand == operand
            }));
        }
    }
}

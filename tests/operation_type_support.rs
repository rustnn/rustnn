use rustnn::operation_data_types::{ALL_DATA_TYPES, OperandName, OperationName};
use rustnn::operation_type_support::{allowed_data_types, is_data_type_allowed};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operators::Operation;

const ARGMAX_OUTPUT_TYPES: &[MLOperandDataType] =
    allowed_data_types(OperationName::ArgMax, OperandName::Output);

#[test]
fn const_lookup_preserves_operand_specific_types() {
    assert_eq!(
        ARGMAX_OUTPUT_TYPES,
        &[MLOperandDataType::Int32, MLOperandDataType::Int64]
    );
    assert_eq!(
        allowed_data_types(OperationName::ArgMax, OperandName::Input),
        ALL_DATA_TYPES
    );
    assert!(allowed_data_types(OperationName::ArgMax, OperandName::Bias).is_empty());
}

#[test]
fn same_type_rules_resolve_to_the_referenced_domain() {
    assert_eq!(
        allowed_data_types(OperationName::Add, OperandName::B),
        ALL_DATA_TYPES
    );
    assert_eq!(
        allowed_data_types(OperationName::BatchNormalization, OperandName::Scale),
        &[MLOperandDataType::Float32, MLOperandDataType::Float16]
    );
    assert_eq!(
        allowed_data_types(OperationName::DequantizeLinear, OperandName::Output),
        allowed_data_types(OperationName::DequantizeLinear, OperandName::Scale)
    );
}

fn argmax() -> Operation {
    Operation::ArgMax {
        input: 0,
        axis: 0,
        options: None,
        outputs: vec![1],
    }
}

#[test]
fn runtime_check_accepts_allowed_types_and_rejects_invalid_pairs() {
    let operation = argmax();
    assert!(is_data_type_allowed(
        &operation,
        OperandName::Output,
        MLOperandDataType::Int64,
        false
    ));
    assert!(!is_data_type_allowed(
        &operation,
        OperandName::Output,
        MLOperandDataType::Float32,
        false
    ));
    assert!(!is_data_type_allowed(
        &operation,
        OperandName::Bias,
        MLOperandDataType::Float32,
        false
    ));
    let extension = Operation::Squeeze {
        input: 0,
        options: None,
        outputs: vec![1],
    };
    assert!(!is_data_type_allowed(
        &extension,
        OperandName::Input,
        MLOperandDataType::Float32,
        false
    ));
}

#[test]
fn bypass_accepts_types_and_warns_on_every_call() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct WarningCounter(AtomicUsize);
    impl log::Log for WarningCounter {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.level() == log::Level::Warn
                && metadata.target() == "rustnn::operation_type_support"
        }
        fn log(&self, record: &log::Record<'_>) {
            if self.enabled(record.metadata()) {
                let message = record.args().to_string();
                assert!(message.contains("Bypassing allowed data type check"));
                assert!(message.contains("argMax"));
                assert!(message.contains("output"));
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        fn flush(&self) {}
    }
    static LOGGER: WarningCounter = WarningCounter(AtomicUsize::new(0));
    log::set_logger(&LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Warn);

    // Both an invalid type and a valid type must warn when bypass is enabled.
    for data_type in [MLOperandDataType::Float32, MLOperandDataType::Int64] {
        assert!(is_data_type_allowed(
            &argmax(),
            OperandName::Output,
            data_type,
            true
        ));
    }
    assert_eq!(LOGGER.0.load(Ordering::SeqCst), 2);
}

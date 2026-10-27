use rustnn::operation_data_types::{ALL_DATA_TYPES, OperandName, OperationName};
use rustnn::operation_type_support::{allowed_data_types, check_allowed_data_types};
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

fn graph(types: &[MLOperandDataType]) -> rustnn::GraphInfo {
    let mut graph = rustnn::GraphInfo::default();
    for &data_type in types {
        graph.operands.push(rustnn::Operand {
            kind: rustnn::OperandKind::Input,
            descriptor: rustnn::OperandDescriptor {
                data_type: data_type.into(),
                shape: rustnn::graph::to_dimension_vector(&[2]),
                pending_permutation: vec![],
            },
            name: None,
        });
    }
    graph
}

#[test]
fn checks_all_operands_and_returns_structured_error() {
    use rustnn::error::AllowedDataTypesError;
    let valid = graph(&[MLOperandDataType::Float32, MLOperandDataType::Int64]);
    assert!(check_allowed_data_types(&argmax(), &valid, false).is_ok());
    let invalid = graph(&[MLOperandDataType::Float32, MLOperandDataType::Float32]);
    assert_eq!(
        check_allowed_data_types(&argmax(), &invalid, false),
        Err(AllowedDataTypesError::NotAllowed {
            operation_name: OperationName::ArgMax,
            operand_name: OperandName::Output,
            allowed_dtype: vec![MLOperandDataType::Int32, MLOperandDataType::Int64],
            used_dtype: MLOperandDataType::Float32,
        })
    );
    assert!(matches!(
        check_allowed_data_types(&argmax(), &graph(&[]), false),
        Err(AllowedDataTypesError::MissingOperand {
            operand_name: OperandName::Input,
            ..
        })
    ));
}

#[test]
fn same_as_checks_actual_types_including_option_operands() {
    let operation = Operation::Gemm {
        a: 0,
        b: 1,
        options: Some(rustnn::operator_options::MLGemmOptions {
            c: Some(2),
            ..Default::default()
        }),
        outputs: vec![3],
    };
    let graph = graph(&[
        MLOperandDataType::Float32,
        MLOperandDataType::Float32,
        MLOperandDataType::Float16,
    ]);
    assert!(matches!(
        check_allowed_data_types(&operation, &graph, false),
        Err(rustnn::error::AllowedDataTypesError::NotAllowed {
            operand_name: OperandName::C,
            used_dtype: MLOperandDataType::Float16,
            ..
        })
    ));
}

#[test]
fn bypass_warns_on_every_call() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct WarningCounter(AtomicUsize);
    impl log::Log for WarningCounter {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.level() == log::Level::Warn
                && metadata.target() == "rustnn::operation_type_support"
        }
        fn log(&self, record: &log::Record<'_>) {
            if self.enabled(record.metadata()) && record.args().to_string().contains("argMax") {
                assert!(record.args().to_string().contains("argMax"));
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        fn flush(&self) {}
    }
    static LOGGER: WarningCounter = WarningCounter(AtomicUsize::new(0));
    log::set_logger(&LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Warn);
    for data_type in [MLOperandDataType::Float32, MLOperandDataType::Int64] {
        assert!(
            check_allowed_data_types(
                &argmax(),
                &graph(&[MLOperandDataType::Float32, data_type]),
                true
            )
            .is_ok()
        );
    }
    assert_eq!(LOGGER.0.load(Ordering::SeqCst), 2);
}

#[cfg(feature = "burn-runtime-cpu")]
#[test]
fn context_option_controls_recording_without_bypassing_shape_validation() {
    use rustnn::mlcontext::{
        Backend, MLContext, MLContextOptions, MLGraphBuilder, MLOperandDescriptor,
        MLPowerPreference,
    };
    for bypass in [false, true] {
        let options = MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::BurnCpu)
            .with_rustnn_bypass_allowed_data_types_checks(bypass);
        let mut context = MLContext::create(&options).unwrap();
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let input = builder
            .input(
                "x",
                &MLOperandDescriptor::new(MLOperandDataType::Uint64, vec![2]),
            )
            .unwrap();
        let result = builder.relu(input);
        assert_eq!(result.is_ok(), bypass);
        if !bypass {
            assert!(matches!(
                result,
                Err(rustnn::error::GraphBuilderError::AllowedDataTypes(_))
            ));
        }
        // An invalid reshape still fails even when dtype checks are bypassed.
        assert!(
            builder
                .reshape(
                    input,
                    vec![rustnn::operator_options::MLDimension::Static(3)]
                )
                .is_err()
        );
    }
}

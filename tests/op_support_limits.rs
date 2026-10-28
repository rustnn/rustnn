use rustnn::limits::{
    EVERYTHING_THAT_IS_ALLOWED, MLOpSupportLimits, MLRankRange, MLTensorLimits, RUSTNN_MAX_TENSOR,
};
use rustnn::operation_data_types::{ALL_DATA_TYPES, OPERATION_DATA_TYPES};
use rustnn::operation_type_support::allowed_data_types;
use rustnn::operator_enums::MLInputOperandLayout;

#[test]
fn shared_limits_cover_every_generated_operand_rule() {
    let limits = serde_json::to_value(&*EVERYTHING_THAT_IS_ALLOWED).unwrap();
    for rule in OPERATION_DATA_TYPES {
        let operation = rule.operation.as_ref();
        let operand = match rule.operand.as_ref() {
            "outputs[0]" => "output0",
            "outputs[1]" => "output1",
            "outputs[2]" => "output2",
            name => name,
        };
        let entry = limits
            .get(operation)
            .and_then(|limits| limits.get(operand))
            .unwrap_or_else(|| panic!("missing limits for {operation}.{operand}"));
        let tensor: MLTensorLimits = serde_json::from_value(entry.clone()).unwrap();
        assert_eq!(
            tensor.data_types,
            allowed_data_types(rule.operation, rule.operand),
            "{operation}.{operand}"
        );
        assert_eq!(
            tensor.rank_range,
            MLRankRange::default(),
            "{operation}.{operand}"
        );
    }
}

#[test]
fn shared_limits_use_default_io_layout_ranks_and_tensor_byte_limit() {
    let limits = &*EVERYTHING_THAT_IS_ALLOWED;
    for tensor in [&limits.input, &limits.constant, &limits.output] {
        assert_eq!(tensor, &MLTensorLimits::default());
        assert_eq!(tensor.data_types, ALL_DATA_TYPES);
        assert_eq!(tensor.rank_range, MLRankRange::default());
    }
    assert_eq!(
        limits.preferred_input_layout,
        MLInputOperandLayout::default()
    );
    assert_eq!(limits.max_tensor_byte_length, u64::from(RUSTNN_MAX_TENSOR));
    let decoded: MLOpSupportLimits =
        serde_json::from_value(serde_json::to_value(limits).unwrap()).unwrap();
    assert_eq!(&decoded, limits);
}

#[cfg(feature = "burn-runtime-cpu")]
#[test]
fn contexts_borrow_the_same_static_even_with_type_policy_extensions() {
    use rustnn::mlcontext::{Backend, MLContext, MLContextOptions, MLPowerPreference};

    for extended in [false, true] {
        let options = MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::BurnCpu)
            .with_rustnn_allow_int4_uint4(extended)
            .with_rustnn_bypass_allowed_data_types_checks(extended);
        let context = MLContext::create(&options).unwrap();
        let limits: &rustnn::mlcontext::MLOpSupportLimits = context.op_support_limits();
        assert!(std::ptr::eq(limits, &*EVERYTHING_THAT_IS_ALLOWED));
        assert!(std::ptr::eq(limits, context.op_support_limits()));
    }
}

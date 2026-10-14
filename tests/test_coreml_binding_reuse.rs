//! Logical names and source-proven copies must survive every tensor-storage mode.

#![cfg(all(target_os = "macos", feature = "coreml-runtime"))]

#[cfg(feature = "dynamic-inputs")]
use rustnn::graph::{DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind};
use rustnn::mlcontext::{
    Backend, BackendStatistics, MLContext, MLContextOptions, MLGraphBuilder, MLNamedOperands,
    MLNamedTensors, MLOperandDescriptor, MLPowerPreference, MLTensor, MLTensorDescriptor,
    RustNNOptions,
};
use rustnn::operator_enums::MLOperandDataType;
#[cfg(feature = "dynamic-inputs")]
use rustnn::operators::Operation;

fn context(reuse: bool, backings: bool) -> MLContext<'static> {
    let mut tuning = RustNNOptions::default();
    tuning.coreml.reuse_tensor_storage = reuse;
    tuning.coreml.output_backings = backings;
    MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml)
            .with_rustnn_options(tuning),
    )
    .unwrap()
}

fn tensor(context: &mut MLContext<'_>, dtype: MLOperandDataType, shape: &[u64]) -> MLTensor {
    context
        .create_tensor(
            &MLTensorDescriptor::new(dtype, shape.to_vec())
                .to_readable()
                .to_writable(),
        )
        .unwrap()
}

fn read(context: &mut MLContext<'_>, tensor: &MLTensor) -> Vec<u8> {
    let mut bytes = vec![0; tensor.rustnn_required_bytes()];
    context.read_tensor(tensor, &mut bytes).unwrap();
    bytes
}

#[test]
fn exact_input_and_constant_copies_keep_names_payloads_and_independent_ownership() {
    for reuse in [false, true] {
        for backings in [false, true] {
            for (dtype, bytes) in [
                (
                    MLOperandDataType::Float32,
                    [1u32, 0x8000_0000, 0x3f80_0001, 0x7fc1_2345]
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                        .collect::<Vec<_>>(),
                ),
                (
                    MLOperandDataType::Float16,
                    [1u16, 0x8000, 0x3c01, 0x7e15]
                        .into_iter()
                        .flat_map(u16::to_le_bytes)
                        .collect(),
                ),
                (
                    MLOperandDataType::Int32,
                    [16_777_217i32, -16_777_217, i32::MIN, i32::MAX]
                        .into_iter()
                        .flat_map(i32::to_le_bytes)
                        .collect(),
                ),
            ] {
                for constant in [false, true] {
                    let mut context = context(reuse, backings);
                    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
                    let descriptor = MLOperandDescriptor::new(dtype, vec![4]);
                    let source = if constant {
                        builder
                            .constant_from_bytes(&descriptor, bytes.clone())
                            .unwrap()
                    } else {
                        builder.input("state", &descriptor).unwrap()
                    };
                    let first = builder.identity(source).unwrap();
                    let second = builder.cast(first, dtype).unwrap();
                    let mut graph = builder
                        .build(&MLNamedOperands::from([
                            ("state", first),
                            ("结果.2", second),
                        ]))
                        .unwrap();
                    let input = tensor(&mut context, dtype, &[4]);
                    let first = tensor(&mut context, dtype, &[4]);
                    let second = tensor(&mut context, dtype, &[4]);
                    for _ in 0..2 {
                        context.write_tensor(&input, &bytes).unwrap();
                        context
                            .dispatch(
                                &mut graph,
                                &if constant {
                                    MLNamedTensors::new()
                                } else {
                                    MLNamedTensors::from([("state", &input)])
                                },
                                &MLNamedTensors::from([("state", &first), ("结果.2", &second)]),
                            )
                            .unwrap();
                        assert_eq!(read(&mut context, &first), bytes);
                        assert_eq!(read(&mut context, &second), bytes);
                        context
                            .write_tensor(&input, &vec![0u8; bytes.len()])
                            .unwrap();
                        context
                            .write_tensor(&first, &vec![0u8; bytes.len()])
                            .unwrap();
                        assert_eq!(read(&mut context, &second), bytes);
                    }
                    let Some(BackendStatistics::Coreml(stats)) =
                        context.rustnn_backend_statistics()
                    else {
                        panic!("missing CoreML statistics");
                    };
                    assert_eq!(stats.output_backings_requested, 0);
                    assert_eq!(stats.output_backings_accepted, 0);
                    assert_eq!(stats.proven_copy_outputs, 4);
                }
            }
        }
    }
}

#[test]
fn arithmetic_alias_fanout_uses_native_results_not_original_input_bytes() {
    for reuse in [false, true] {
        for backings in [false, true] {
            let mut context = context(reuse, backings);
            let mut builder = MLGraphBuilder::new(&mut context).unwrap();
            let input = builder
                .input(
                    "fp32",
                    &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4]),
                )
                .unwrap();
            let negated = builder.neg(input).unwrap();
            let copied = builder.identity(negated).unwrap();
            let mut graph = builder
                .build(&MLNamedOperands::from([
                    ("program", negated),
                    ("0copy", copied),
                ]))
                .unwrap();
            let mut builder = MLGraphBuilder::new(&mut context).unwrap();
            let source = builder
                .input(
                    "fp32",
                    &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4]),
                )
                .unwrap();
            let negated = builder.neg(source).unwrap();
            let source_copy = builder.identity(source).unwrap();
            let mut mixed_graph = builder
                .build(&MLNamedOperands::from([
                    ("program", negated),
                    ("source_copy", source_copy),
                ]))
                .unwrap();
            let input = tensor(&mut context, MLOperandDataType::Float32, &[4]);
            let first = tensor(&mut context, MLOperandDataType::Float32, &[4]);
            let second = tensor(&mut context, MLOperandDataType::Float32, &[4]);
            context.write_tensor(&input, &[1f32, -2., 3., -4.]).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("fp32", &input)]),
                    &MLNamedTensors::from([("program", &first), ("0copy", &second)]),
                )
                .unwrap();
            let expected = bytemuck::cast_slice::<f32, u8>(&[-1., 2., -3., 4.]);
            assert_eq!(read(&mut context, &first), expected);
            assert_eq!(read(&mut context, &second), expected);
            let Some(BackendStatistics::Coreml(stats)) = context.rustnn_backend_statistics() else {
                panic!("missing CoreML statistics");
            };
            assert_eq!(stats.proven_copy_outputs, 0);
            context.write_tensor(&first, &[0f32; 4]).unwrap();
            assert_eq!(read(&mut context, &second), expected);

            context
                .dispatch(
                    &mut mixed_graph,
                    &MLNamedTensors::from([("fp32", &input)]),
                    &MLNamedTensors::from([("program", &first), ("source_copy", &second)]),
                )
                .unwrap();
            assert_eq!(read(&mut context, &first), expected);
            assert_eq!(
                read(&mut context, &second),
                bytemuck::cast_slice::<f32, u8>(&[1., -2., 3., -4.])
            );
            let Some(BackendStatistics::Coreml(before_failure)) =
                context.rustnn_backend_statistics()
            else {
                panic!("missing CoreML statistics");
            };
            assert_eq!(before_failure.proven_copy_outputs, 1);
            assert!(
                context
                    .dispatch(
                        &mut mixed_graph,
                        &MLNamedTensors::new(),
                        &MLNamedTensors::from([("program", &first), ("source_copy", &second)]),
                    )
                    .is_err()
            );
            let Some(BackendStatistics::Coreml(after_failure)) =
                context.rustnn_backend_statistics()
            else {
                panic!("missing CoreML statistics");
            };
            assert_eq!(
                after_failure.proven_copy_outputs - before_failure.proven_copy_outputs,
                0
            );
        }
    }
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn bounded_copy_proofs_validate_active_tensor_shapes_in_every_storage_mode() {
    for reuse in [false, true] {
        for backings in [false, true] {
            let mut context = context(reuse, backings);
            let shape = vec![Dimension::Dynamic(rustnn::graph::DynamicDimension {
                name: "length".into(),
                max_size: 6,
            })];
            let operand = |kind, name: &str| Operand {
                kind,
                name: Some(name.into()),
                descriptor: OperandDescriptor {
                    data_type: DataType::Int32,
                    shape: shape.clone(),
                    pending_permutation: vec![],
                },
            };
            let mut graph = context
                .rustnn_build_graph(GraphInfo {
                    operands: vec![
                        operand(OperandKind::Input, "state"),
                        operand(OperandKind::Output, "tensor"),
                    ],
                    input_operands: vec![0],
                    output_operands: vec![1],
                    operations: vec![Operation::Identity {
                        input: 0,
                        options: None,
                        outputs: vec![1],
                    }],
                    ..Default::default()
                })
                .unwrap();
            let mut input = tensor(&mut context, MLOperandDataType::Int32, &[1]);
            let mut output = tensor(&mut context, MLOperandDataType::Int32, &[1]);
            context
                .rustnn_set_tensor_capacity(&mut input, &[6])
                .unwrap();
            context
                .rustnn_set_tensor_capacity(&mut output, &[6])
                .unwrap();
            for length in [1, 6, 2, 1] {
                context.rustnn_resize_tensor(&mut input, &[length]).unwrap();
                context
                    .rustnn_resize_tensor(&mut output, &[length])
                    .unwrap();
                let values = vec![16_777_217i32; length as usize];
                context.write_tensor(&input, &values).unwrap();
                context
                    .dispatch(
                        &mut graph,
                        &MLNamedTensors::from([("state", &input)]),
                        &MLNamedTensors::from([("tensor", &output)]),
                    )
                    .unwrap();
                assert_eq!(
                    read(&mut context, &output),
                    bytemuck::cast_slice::<i32, u8>(&values)
                );
            }
        }
    }
}

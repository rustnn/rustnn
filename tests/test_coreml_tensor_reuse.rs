//! Storage lifetime, aliasing and shape regressions for the CoreML tensor paths.

#![cfg(all(target_os = "macos", feature = "coreml-runtime"))]

use rustnn::error::Error;
use rustnn::graph::{DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind};
use rustnn::mlcontext::{
    Backend, BackendStatistics, CoremlTensorStatistics, MLContext, MLContextOptions,
    MLGraphBuilder, MLNamedOperands, MLNamedTensors, MLOperandDescriptor, MLPowerPreference,
    MLTensor, MLTensorDescriptor, RustNNOptions,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::MLDimension;
use rustnn::operators::Operation;

#[derive(Clone, Copy, Debug)]
enum Mode {
    Baseline,
    Persistent,
    Backings,
}

const MODES: [Mode; 3] = [Mode::Persistent, Mode::Backings, Mode::Baseline];

fn statistics(context: &MLContext<'_>) -> CoremlTensorStatistics {
    match context.rustnn_backend_statistics() {
        Some(BackendStatistics::Coreml(statistics)) => statistics,
        _ => panic!("CoreML should report its typed statistics"),
    }
}

fn context(mode: Mode) -> MLContext<'static> {
    let mut options = RustNNOptions::default();
    options.coreml.reuse_tensor_storage = !matches!(mode, Mode::Baseline);
    options.coreml.output_backings = matches!(mode, Mode::Backings);
    MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml)
            .with_rustnn_options(options),
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
    let mut result = vec![0; tensor.rustnn_required_bytes()];
    context.read_tensor(tensor, &mut result).unwrap();
    result
}

#[test]
fn backend_statistics_reports_typed_coreml_snapshots() {
    for mode in MODES {
        let mut context = context(mode);
        let initial = context.rustnn_backend_statistics();
        assert_eq!(
            initial,
            Some(BackendStatistics::Coreml(CoremlTensorStatistics::default()))
        );

        let tensor = tensor(&mut context, MLOperandDataType::Float32, &[2]);
        context.write_tensor(&tensor, &[1.0f32, 2.0]).unwrap();
        let Some(BackendStatistics::Coreml(statistics)) = context.rustnn_backend_statistics()
        else {
            panic!("CoreML should report its typed statistics");
        };
        assert_eq!(statistics.host_write_bytes, 8);
        assert_ne!(initial, context.rustnn_backend_statistics());
        assert_eq!(
            initial,
            Some(BackendStatistics::Coreml(CoremlTensorStatistics::default()))
        );
    }
}

fn identity(
    dtype: DataType,
    input_shape: Vec<Dimension>,
    output_shape: Vec<Dimension>,
) -> GraphInfo {
    let operand = |name: &str, kind, shape| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: dtype,
            shape,
            pending_permutation: vec![],
        },
    };
    GraphInfo {
        operands: vec![
            operand("input", OperandKind::Input, input_shape),
            operand("result", OperandKind::Output, output_shape),
        ],
        input_operands: vec![0],
        output_operands: vec![1],
        operations: vec![Operation::Identity {
            input: 0,
            options: None,
            outputs: vec![1],
        }],
        ..Default::default()
    }
}

#[test]
fn reuse_identity_preserves_native_types_and_owned_results() {
    let cases: [(MLOperandDataType, Vec<u8>); 3] = [
        (
            MLOperandDataType::Float32,
            bytemuck::cast_slice(&[-2.5f32, 0., 1., 3.5, 65504., -0.125]).to_vec(),
        ),
        (
            MLOperandDataType::Float16,
            bytemuck::cast_slice(&[0xc100u16, 0x0000, 0x3c00, 0x4300, 0x7bff, 0xb000]).to_vec(),
        ),
        (
            MLOperandDataType::Int32,
            bytemuck::cast_slice(&[-4096i32, -1023, -1, 0, 1023, 4096]).to_vec(),
        ),
    ];
    for mode in MODES {
        for (dtype, values) in &cases {
            let mut context = context(mode);
            let shape = vec![Dimension::Static(6)];
            let mut graph = context
                .rustnn_build_graph(identity((*dtype).into(), shape.clone(), shape))
                .unwrap();
            let input = tensor(&mut context, *dtype, &[6]);
            let output = tensor(&mut context, *dtype, &[6]);
            let allocated = statistics(&context).native_allocations;
            for _ in 0..4 {
                context.write_tensor(&input, values).unwrap();
                context
                    .dispatch(
                        &mut graph,
                        &MLNamedTensors::from([("input", &input)]),
                        &MLNamedTensors::from([("result", &output)]),
                    )
                    .unwrap();
                assert_eq!(read(&mut context, &output), *values, "{mode:?} {dtype:?}");
                context
                    .write_tensor(&input, &vec![0u8; values.len()])
                    .unwrap();
                assert_eq!(
                    read(&mut context, &output),
                    *values,
                    "input alias {mode:?} {dtype:?}"
                );
            }
            let stats = statistics(&context);
            assert_eq!(stats.native_allocations, allocated);
            if matches!(mode, Mode::Baseline) {
                assert_eq!(stats.native_input_bindings, 0);
                assert_eq!(stats.input_copy_bytes, 4 * values.len() as u64);
            } else if *dtype == MLOperandDataType::Float32 {
                assert_eq!(stats.native_input_bindings, 4);
                assert_eq!(stats.input_copy_bytes, 0);
            }
            if !matches!(mode, Mode::Backings) {
                assert_eq!(stats.output_backings_requested, 0);
                assert_eq!(stats.output_copy_bytes, 4 * values.len() as u64);
            } else {
                // The native unit probe checks proposals against loaded metadata.
                // Here, both ineligible and declined backings must copy correctly.
                assert!(matches!(stats.output_backings_requested, 0 | 4));
                assert!(stats.output_backings_accepted <= stats.output_backings_requested);
                assert_eq!(
                    stats.output_copy_bytes,
                    (4 - stats.output_backings_accepted) * values.len() as u64
                );
            }
            assert!(stats.output_backings_accepted <= stats.output_backings_requested);
        }
    }
}

#[test]
fn reuse_host_roundtrip_keeps_all_int32_bits_without_model_execution() {
    let expected = [i32::MIN, -16_777_217, -1, 0, 16_777_217, i32::MAX];
    for mode in MODES {
        let mut context = context(mode);
        let tensor = tensor(&mut context, MLOperandDataType::Int32, &[6]);
        context.write_tensor(&tensor, &expected).unwrap();
        assert_eq!(
            read(&mut context, &tensor),
            bytemuck::cast_slice::<i32, u8>(&expected)
        );
    }
}

#[test]
fn reuse_cast_converts_same_width_types_and_preserves_owned_output_across_dispatch() {
    for mode in MODES {
        for (source, target) in [
            (MLOperandDataType::Int32, MLOperandDataType::Float32),
            (MLOperandDataType::Float32, MLOperandDataType::Int32),
            (MLOperandDataType::Float16, MLOperandDataType::Float32),
            (MLOperandDataType::Float32, MLOperandDataType::Float16),
            // Int64 uses host storage and a CoreML Int32 proxy.
            (MLOperandDataType::Int64, MLOperandDataType::Float32),
            (MLOperandDataType::Float32, MLOperandDataType::Int64),
        ] {
            let mut context = context(mode);
            let mut builder = MLGraphBuilder::new(&mut context).unwrap();
            let input = builder
                .input("input", &MLOperandDescriptor::new(source, vec![2, 3]))
                .unwrap();
            let converted = builder.cast(input, target).unwrap();
            let transposed = builder.transpose(converted).unwrap();
            let mut graph = builder
                .build(&MLNamedOperands::from([
                    ("converted", converted),
                    ("transposed", transposed),
                ]))
                .unwrap();
            let input = tensor(&mut context, source, &[2, 3]);
            let converted = tensor(&mut context, target, &[2, 3]);
            let transposed = tensor(&mut context, target, &[3, 2]);
            let encode = |dtype, values: &[i32]| -> Vec<u8> {
                match dtype {
                    MLOperandDataType::Int32 => bytemuck::cast_slice(values).to_vec(),
                    MLOperandDataType::Int64 => values
                        .iter()
                        .flat_map(|&value| i64::from(value).to_ne_bytes())
                        .collect(),
                    MLOperandDataType::Float32 => values
                        .iter()
                        .flat_map(|&value| (value as f32).to_ne_bytes())
                        .collect(),
                    MLOperandDataType::Float16 => values
                        .iter()
                        .flat_map(|&value| {
                            half::f16::from_f32(value as f32).to_bits().to_ne_bytes()
                        })
                        .collect(),
                    _ => unreachable!(),
                }
            };
            for offset in [0, 1, -1] {
                let values = [-31, -2, -1, 0, 123, 1024].map(|v| v + offset);
                let expected = encode(target, &values);
                let expected_transpose = encode(
                    target,
                    &[
                        values[0], values[3], values[1], values[4], values[2], values[5],
                    ],
                );
                context
                    .write_tensor(&input, &encode(source, &values))
                    .unwrap();
                context
                    .dispatch(
                        &mut graph,
                        &MLNamedTensors::from([("input", &input)]),
                        &MLNamedTensors::from([
                            ("converted", &converted),
                            ("transposed", &transposed),
                        ]),
                    )
                    .unwrap();
                assert_eq!(
                    read(&mut context, &converted),
                    expected,
                    "{mode:?} {source:?}->{target:?}"
                );
                assert_eq!(read(&mut context, &transposed), expected_transpose);
                context
                    .write_tensor(&input, &encode(source, &[0; 6]))
                    .unwrap();
                assert_eq!(read(&mut context, &converted), expected);
                context
                    .write_tensor(&converted, &encode(target, &[1; 6]))
                    .unwrap();
                assert_eq!(read(&mut context, &transposed), expected_transpose);
            }
        }
    }
}

#[test]
fn reuse_view_and_multiple_outputs_never_alias_input_or_each_other() {
    for mode in MODES {
        let mut context = context(mode);
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let input = builder
            .input(
                "input",
                &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 3]),
            )
            .unwrap();
        let same = builder.identity(input).unwrap();
        let flat = builder
            .reshape(input, vec![MLDimension::Static(6)])
            .unwrap();
        let transposed = builder.transpose(input).unwrap();
        let mut graph = builder
            .build(&MLNamedOperands::from([
                ("same", same),
                ("flat", flat),
                ("transposed", transposed),
            ]))
            .unwrap();
        let input = tensor(&mut context, MLOperandDataType::Float32, &[2, 3]);
        let same = tensor(&mut context, MLOperandDataType::Float32, &[2, 3]);
        let flat = tensor(&mut context, MLOperandDataType::Float32, &[6]);
        let transposed = tensor(&mut context, MLOperandDataType::Float32, &[3, 2]);
        for offset in [0., 10., -10.] {
            let values = [1., 2., 3., 4., 5., 6.].map(|v| v + offset);
            let transpose = [
                values[0], values[3], values[1], values[4], values[2], values[5],
            ];
            context.write_tensor(&input, &values).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("input", &input)]),
                    &MLNamedTensors::from([
                        ("same", &same),
                        ("flat", &flat),
                        ("transposed", &transposed),
                    ]),
                )
                .unwrap();
            let bytes = bytemuck::cast_slice::<f32, u8>(&values);
            let transpose_bytes = bytemuck::cast_slice::<f32, u8>(&transpose);
            assert_eq!(read(&mut context, &same), bytes, "{mode:?}");
            assert_eq!(read(&mut context, &flat), bytes, "{mode:?}");
            assert_eq!(read(&mut context, &transposed), transpose_bytes, "{mode:?}");
            context.write_tensor(&input, &[99f32; 6]).unwrap();
            assert_eq!(read(&mut context, &flat), bytes, "input mutation {mode:?}");
            context.write_tensor(&same, &[88f32; 6]).unwrap();
            assert_eq!(read(&mut context, &flat), bytes, "output alias {mode:?}");
            assert_eq!(
                read(&mut context, &transposed),
                transpose_bytes,
                "view alias {mode:?}"
            );
        }
    }
}

#[test]
fn reuse_rejects_duplicate_bindings_before_native_locks() {
    for mode in MODES {
        let mut context = context(mode);
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2]);
        let a = builder.input("a", &desc).unwrap();
        let b = builder.input("b", &desc).unwrap();
        let result = builder.add(a, b).unwrap();
        let mut graph = builder
            .build(&MLNamedOperands::from([("result", result)]))
            .unwrap();
        let a = tensor(&mut context, MLOperandDataType::Float32, &[2]);
        let b = tensor(&mut context, MLOperandDataType::Float32, &[2]);
        let result = tensor(&mut context, MLOperandDataType::Float32, &[2]);
        let alias = a.clone();
        let before = statistics(&context);
        let error = context
            .dispatch(
                &mut graph,
                &MLNamedTensors::from([("a", &a), ("b", &alias)]),
                &MLNamedTensors::from([("result", &result)]),
            )
            .unwrap_err();
        assert!(
            matches!(error, Error::DuplicateTensorBinding { .. }),
            "{error}"
        );
        let error = context
            .dispatch(
                &mut graph,
                &MLNamedTensors::from([("a", &a), ("b", &b)]),
                &MLNamedTensors::from([("result", &alias)]),
            )
            .unwrap_err();
        assert!(
            matches!(error, Error::DuplicateTensorBinding { .. }),
            "{error}"
        );
        assert_eq!(before, statistics(&context));
    }
}

#[test]
fn reuse_resize_growth_preserves_prefix_and_reserve_resets_storage() {
    for mode in MODES {
        let mut context = context(mode);
        let mut value = tensor(&mut context, MLOperandDataType::Float32, &[2]);
        context.write_tensor(&value, &[4f32, 9.]).unwrap();
        context.rustnn_resize_tensor(&mut value, &[4]).unwrap();
        assert_eq!(
            read(&mut context, &value),
            bytemuck::cast_slice::<f32, u8>(&[4f32, 9., 0., 0.])
        );
        context.rustnn_resize_tensor(&mut value, &[1]).unwrap();
        assert_eq!(
            read(&mut context, &value),
            bytemuck::cast_slice::<f32, u8>(&[4f32])
        );
        context.rustnn_resize_tensor(&mut value, &[4]).unwrap();
        assert_eq!(
            read(&mut context, &value),
            bytemuck::cast_slice::<f32, u8>(&[4f32, 9., 0., 0.])
        );
        let error = context
            .rustnn_set_tensor_capacity(&mut value, &[2])
            .unwrap_err();
        assert!(matches!(
            error,
            Error::TensorCapacityError {
                requested_bytes: 8,
                required_bytes: 16,
                ..
            }
        ));
        assert_eq!(
            read(&mut context, &value),
            bytemuck::cast_slice::<f32, u8>(&[4f32, 9., 0., 0.])
        );
        context
            .rustnn_set_tensor_capacity(&mut value, &[8])
            .unwrap();
        assert_eq!(value.shape(), [4]);
        assert_eq!(read(&mut context, &value), vec![0; 16]);
        assert!(context.write_tensor(&value, &[1f32; 8]).is_err());
        assert!(context.read_tensor(&value, &mut [0f32; 8]).is_err());
    }
}

#[cfg(feature = "dynamic-inputs")]
fn dynamic(name: &str) -> Dimension {
    Dimension::Dynamic(rustnn::graph::DynamicDimension {
        name: name.into(),
        max_size: 8,
    })
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn reuse_dynamic_shapes_grow_shrink_and_reset_without_new_allocations() {
    for mode in MODES {
        let mut context = context(mode);
        let shape = vec![dynamic("batch"), Dimension::Static(2)];
        let mut graph = context
            .rustnn_build_graph(identity(DataType::Float32, shape.clone(), shape))
            .unwrap();
        let mut input = tensor(&mut context, MLOperandDataType::Float32, &[1, 2]);
        let mut output = tensor(&mut context, MLOperandDataType::Float32, &[1, 2]);
        context
            .rustnn_set_tensor_capacity(&mut input, &[8, 2])
            .unwrap();
        context
            .rustnn_set_tensor_capacity(&mut output, &[8, 2])
            .unwrap();
        let allocations = statistics(&context).native_allocations;
        for n in [1, 4, 2, 8, 1] {
            context.rustnn_resize_tensor(&mut input, &[n, 2]).unwrap();
            context.rustnn_resize_tensor(&mut output, &[n, 2]).unwrap();
            let values = (0..n * 2).map(|i| i as f32 - 3.).collect::<Vec<_>>();
            context.write_tensor(&input, &values).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("input", &input)]),
                    &MLNamedTensors::from([("result", &output)]),
                )
                .unwrap();
            assert_eq!(
                read(&mut context, &output),
                bytemuck::cast_slice::<f32, u8>(&values),
                "{mode:?} shape {n}"
            );
            assert_eq!(output.shape(), [n, 2]);
        }
        assert_eq!(statistics(&context).native_allocations, allocations);
        if matches!(mode, Mode::Backings) {
            let stats = statistics(&context);
            assert_eq!(stats.output_backings_requested, 0);
            assert_eq!(stats.output_backings_accepted, 0);
            assert!(stats.output_copy_bytes > 0);
        }
        context.rustnn_resize_tensor(&mut output, &[2, 2]).unwrap();
        assert!(
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("input", &input)]),
                    &MLNamedTensors::from([("result", &output)])
                )
                .is_err()
        );
    }
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn reuse_equal_byte_counts_do_not_hide_wrong_output_shape() {
    // The legacy byte-buffer executor does not compare returned extents to the
    // active output binding when differently named dimensions have equal size.
    for mode in [Mode::Persistent, Mode::Backings] {
        let mut context = context(mode);
        let mut graph = context
            .rustnn_build_graph(identity(
                DataType::Float32,
                vec![dynamic("input_rows"), dynamic("input_columns")],
                vec![dynamic("output_rows"), dynamic("output_columns")],
            ))
            .unwrap();
        let input = tensor(&mut context, MLOperandDataType::Float32, &[2, 3]);
        let output = tensor(&mut context, MLOperandDataType::Float32, &[3, 2]);
        context
            .write_tensor(&input, &[1f32, 2., 3., 4., 5., 6.])
            .unwrap();
        let result = context.dispatch(
            &mut graph,
            &MLNamedTensors::from([("input", &input)]),
            &MLNamedTensors::from([("result", &output)]),
        );
        assert!(
            result.is_err(),
            "accepted [2,3] result in [3,2] binding: {mode:?}"
        );
    }
}

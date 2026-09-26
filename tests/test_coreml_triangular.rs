//! Exact CoreML triangular masking regressions, including non-finite values.
//!
//! Run with `make test-coreml TEST_FILTER=triangular` on macOS.

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
mod runtime {
    use half::f16;
    use rustnn::graph::{DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind};
    use rustnn::mlcontext::{
        Backend, MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::mlgraphbuilder::MLGraphBuilder;
    use rustnn::operator_enums::MLOperandDataType;
    use rustnn::operator_options::MLTriangularOptions;
    use rustnn::operators::Operation;

    fn graph(dtype: DataType, shape: Vec<Dimension>, upper: bool, diagonal: i32) -> GraphInfo {
        let descriptor = OperandDescriptor {
            data_type: dtype,
            shape,
            pending_permutation: vec![],
        };
        GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    name: Some("input".into()),
                    descriptor: descriptor.clone(),
                },
                Operand {
                    kind: OperandKind::Output,
                    name: Some("result".into()),
                    descriptor,
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![Operation::Triangular {
                input: 0,
                options: Some(MLTriangularOptions {
                    upper: Some(upper),
                    diagonal,
                    ..Default::default()
                }),
                outputs: vec![1],
            }],
            ..Default::default()
        }
    }

    fn check(
        dtype: DataType,
        graph_shape: Vec<Dimension>,
        upper: bool,
        diagonal: i32,
        runs: &[(Vec<u64>, Vec<f32>)],
    ) {
        let tensor_dtype = match dtype {
            DataType::Float16 => MLOperandDataType::Float16,
            DataType::Float32 => MLOperandDataType::Float32,
            DataType::Int32 => MLOperandDataType::Int32,
            _ => unreachable!(),
        };
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_rustnn_backend_hint(Backend::Coreml),
        )
        .unwrap();
        let mut graph = MLGraphBuilder::new(&mut context)
            .unwrap()
            .build_graph_info(graph(dtype, graph_shape, upper, diagonal))
            .unwrap();

        // A single graph/model is reused throughout changing active shapes.
        for (shape, values) in runs {
            assert_eq!(shape.iter().product::<u64>() as usize, values.len());
            let input = context
                .create_tensor(&MLTensorDescriptor::new(tensor_dtype, shape.clone()).to_writable())
                .unwrap();
            let output = context
                .create_tensor(&MLTensorDescriptor::new(tensor_dtype, shape.clone()).to_readable())
                .unwrap();
            let bytes: Vec<u8> = if dtype == DataType::Float16 {
                values
                    .iter()
                    .flat_map(|&value| f16::from_f32(value).to_le_bytes())
                    .collect()
            } else if dtype == DataType::Int32 {
                values
                    .iter()
                    .flat_map(|&value| (value as i32).to_le_bytes())
                    .collect()
            } else {
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect()
            };
            context.write_tensor(&input, &bytes).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("input", &input)]),
                    &MLNamedTensors::from([("result", &output)]),
                )
                .unwrap();
            let mut actual = vec![0u8; bytes.len()];
            context.read_tensor(&output, &mut actual).unwrap();
            let actual: Vec<f32> = if dtype == DataType::Float16 {
                actual
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&bytes| f16::from_le_bytes(bytes).to_f32())
                    .collect()
            } else if dtype == DataType::Int32 {
                actual
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&bytes| i32::from_le_bytes(bytes) as f32)
                    .collect()
            } else {
                actual
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&bytes| f32::from_le_bytes(bytes))
                    .collect()
            };
            let rows = shape[shape.len() - 2] as usize;
            let columns = shape[shape.len() - 1] as usize;
            for (index, (&input, &actual)) in values.iter().zip(&actual).enumerate() {
                let row = (index / columns) % rows;
                let column = index % columns;
                let offset = column as i64 - row as i64;
                let keep = if upper {
                    offset >= i64::from(diagonal)
                } else {
                    offset <= i64::from(diagonal)
                };
                let expected = if !keep {
                    0.0
                } else if dtype == DataType::Float16 {
                    f16::from_f32(input).to_f32()
                } else {
                    input
                };
                let detail = format!(
                    "{dtype:?}, shape={shape:?}, upper={upper}, diagonal={diagonal}, index={index}"
                );
                if expected.is_nan() {
                    assert!(actual.is_nan(), "{detail}: expected NaN, got {actual}");
                } else {
                    // Kept signed zeros/infinities must remain unchanged;
                    // masked values must become positive zero, not NaN or -0.
                    assert_eq!(actual.to_bits(), expected.to_bits(), "{detail}");
                }
            }
        }
    }

    fn finite(shape: &[u64]) -> Vec<f32> {
        (1..=shape.iter().product::<u64>())
            .map(|value| value as f32)
            .collect()
    }

    fn nonfinite(shape: &[u64]) -> Vec<f32> {
        let pattern = [
            f32::NAN,
            f32::INFINITY,
            -0.0,
            f32::NEG_INFINITY,
            0.0,
            7.0,
            -8.0,
        ];
        (0..shape.iter().product::<u64>() as usize)
            .map(|index| pattern[index % pattern.len()])
            .collect()
    }

    #[test]
    fn triangular_both_diagonal_signs_on_rectangular_matrices() {
        for dtype in [DataType::Float16, DataType::Float32, DataType::Int32] {
            for shape in [vec![3, 3], vec![4, 2], vec![2, 4]] {
                for upper in [false, true] {
                    for diagonal in [-2, -1, 0, 1, 2] {
                        check(
                            dtype,
                            shape.iter().map(|&x| Dimension::Static(x as u32)).collect(),
                            upper,
                            diagonal,
                            &[(shape.clone(), finite(&shape))],
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn triangular_masks_nonfinite_values_without_changing_retained_values() {
        for dtype in [DataType::Float16, DataType::Float32] {
            for upper in [false, true] {
                for diagonal in [-1, 0, 1] {
                    check(
                        dtype,
                        vec![Dimension::Static(3); 2],
                        upper,
                        diagonal,
                        &[(vec![3, 3], nonfinite(&[3, 3]))],
                    );
                }
            }
        }
    }

    #[test]
    fn triangular_extreme_diagonals_do_not_overflow_or_leak_nonfinite_values() {
        for dtype in [DataType::Float16, DataType::Float32] {
            for upper in [false, true] {
                for diagonal in [i32::MIN, -5, 5, i32::MAX] {
                    check(
                        dtype,
                        vec![Dimension::Static(3), Dimension::Static(4)],
                        upper,
                        diagonal,
                        &[(vec![3, 4], nonfinite(&[3, 4]))],
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "CoreML loses int32 precision on some configurations including CI: 16777217 - 16777216 returns 0; see rustnn/rustnn#235"]
    fn triangular_int32_keeps_values_beyond_float32_precision() {
        let values = [
            i32::MIN,
            i32::MAX,
            16_777_217,
            -16_777_217,
            0,
            7,
            i32::MAX,
            i32::MIN,
            16_777_219,
        ];
        // Keep the exact assertions for the excluded-main-diagonal lowering
        // while a portable fix for CoreML's int32 precision loss is pending.
        for (upper, diagonal) in [(true, 1), (false, -1)] {
            let mut context = MLContext::create(
                &MLContextOptions::new(MLPowerPreference::Default, false)
                    .with_rustnn_backend_hint(Backend::Coreml),
            )
            .unwrap();
            let mut graph = MLGraphBuilder::new(&mut context)
                .unwrap()
                .build_graph_info(graph(
                    DataType::Int32,
                    vec![Dimension::Static(3); 2],
                    upper,
                    diagonal,
                ))
                .unwrap();
            let descriptor = MLTensorDescriptor::new(MLOperandDataType::Int32, vec![3, 3]);
            let input = context
                .create_tensor(&descriptor.clone().to_writable())
                .unwrap();
            let output = context.create_tensor(&descriptor.to_readable()).unwrap();
            context.write_tensor(&input, &values).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("input", &input)]),
                    &MLNamedTensors::from([("result", &output)]),
                )
                .unwrap();
            let mut actual = [0i32; 9];
            context.read_tensor(&output, &mut actual).unwrap();
            for (index, (&value, &actual)) in values.iter().zip(&actual).enumerate() {
                let offset = (index % 3) as i32 - (index / 3) as i32;
                let keep = if upper {
                    offset >= diagonal
                } else {
                    offset <= diagonal
                };
                assert_eq!(
                    actual,
                    if keep { value } else { 0 },
                    "upper={upper}, diagonal={diagonal}, index={index}"
                );
            }
        }
    }

    #[cfg(feature = "dynamic-inputs")]
    #[test]
    fn triangular_reuses_one_model_for_growing_and_shrinking_active_matrices() {
        use rustnn::graph::DynamicDimension;
        let shape = vec![
            Dimension::Static(2),
            Dimension::Dynamic(DynamicDimension {
                name: "rows".into(),
                max_size: 8,
            }),
            Dimension::Dynamic(DynamicDimension {
                name: "columns".into(),
                max_size: 8,
            }),
        ];
        let runs: Vec<_> = [(1, 3), (4, 2), (2, 4), (1, 3)]
            .into_iter()
            .map(|(rows, columns)| {
                let shape = vec![2, rows, columns];
                let values = nonfinite(&shape);
                (shape, values)
            })
            .collect();
        for dtype in [DataType::Float16, DataType::Float32] {
            for upper in [false, true] {
                for diagonal in [-1, 0, 1] {
                    check(dtype, shape.clone(), upper, diagonal, &runs);
                }
            }
        }
    }
}

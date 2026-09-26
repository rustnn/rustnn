//! Numerical regressions for index normalization against active data dimensions.
//!
//! Dispatch tests compile each graph once, then use growing and shrinking data
//! tensors. Changing only the indices shape would not exercise these regressions.
//! A separate executor test checks actual CoreML output shapes.

#![cfg(all(
    target_os = "macos",
    feature = "coreml-runtime",
    feature = "dynamic-inputs"
))]

use std::collections::HashMap;

use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_checked};
use rustnn::graph::{
    ConstantData, DataType, Dimension, DynamicDimension, GraphInfo, Operand, OperandDescriptor,
    OperandKind, to_dimension_vector,
};
use rustnn::mlcontext::{
    Backend, MLContext, MLContextOptions, MLGraph, MLGraphBuilder, MLNamedTensors,
    MLPowerPreference, MLTensor, MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operators::Operation;

fn dynamic(name: &str, max_size: u32) -> Dimension {
    Dimension::Dynamic(DynamicDimension {
        name: name.into(),
        max_size,
    })
}

fn context() -> MLContext<'static> {
    MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml),
    )
    .expect("create CPU CoreML context")
}

fn gather_graph(
    operation: &str,
    axis: Option<u32>,
    data_shape: Vec<Dimension>,
    index_shape: Vec<Dimension>,
    output_shape: Vec<Dimension>,
    constant_indices: Option<&[i32]>,
) -> GraphInfo {
    let operand = |name: &str, kind, data_type, shape| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type,
            shape,
            pending_permutation: vec![],
        },
    };
    let attributes = axis.map_or(
        serde_json::Value::Null,
        |axis| serde_json::json!({ "axis": axis }),
    );
    let mut graph = GraphInfo {
        operands: vec![
            operand("data", OperandKind::Input, DataType::Float32, data_shape),
            operand(
                "indices",
                if constant_indices.is_some() {
                    OperandKind::Constant
                } else {
                    OperandKind::Input
                },
                DataType::Int32,
                index_shape,
            ),
            operand(
                "result",
                OperandKind::Output,
                DataType::Float32,
                output_shape,
            ),
        ],
        operations: vec![
            Operation::from_json_attributes(operation, &[0, 1], &[2], &attributes)
                .expect("gather operation"),
        ],
        input_operands: if constant_indices.is_some() {
            vec![0]
        } else {
            vec![0, 1]
        },
        output_operands: vec![2],
        ..Default::default()
    };
    if let Some(indices) = constant_indices {
        graph.constant_operand_ids_to_handles.insert(
            1,
            ConstantData {
                data: indices
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                label: None,
            },
        );
    }
    graph
}

fn data_tensor(context: &mut MLContext<'_>, shape: &[u64], values: &[f32]) -> MLTensor {
    let tensor = context
        .create_tensor(
            &MLTensorDescriptor::new(MLOperandDataType::Float32, shape.to_vec()).to_writable(),
        )
        .unwrap();
    context.write_tensor(&tensor, values).unwrap();
    tensor
}

fn index_tensor(context: &mut MLContext<'_>, shape: &[u64], values: &[i32]) -> MLTensor {
    let tensor = context
        .create_tensor(
            &MLTensorDescriptor::new(MLOperandDataType::Int32, shape.to_vec()).to_writable(),
        )
        .unwrap();
    context.write_tensor(&tensor, values).unwrap();
    tensor
}

fn check_dispatch(
    context: &mut MLContext<'_>,
    graph: &mut MLGraph<'_>,
    data: &MLTensor,
    indices: Option<&MLTensor>,
    shape: &[u64],
    expected: &[f32],
    case: &str,
) {
    let output = context
        .create_tensor(
            &MLTensorDescriptor::new(MLOperandDataType::Float32, shape.to_vec()).to_readable(),
        )
        .unwrap();
    let mut inputs = MLNamedTensors::from([("data", data)]);
    if let Some(indices) = indices {
        inputs.insert("indices", indices);
    }
    context
        .dispatch(graph, &inputs, &MLNamedTensors::from([("result", &output)]))
        .unwrap_or_else(|error| panic!("{case}: dispatch failed: {error:?}"));
    let mut actual = vec![f32::NAN; expected.len()];
    context.read_tensor(&output, &mut actual).unwrap();
    assert_eq!(output.shape(), shape, "{case}: output tensor descriptor");
    assert_eq!(actual, expected, "{case}: output values");
}

#[test]
fn gather_uses_active_indexed_axis_with_constant_and_runtime_indices() {
    let indices = [-1, 0, 100, -100];
    for constant in [false, true] {
        let mut context = context();
        let graph_info = gather_graph(
            "gather",
            Some(0),
            vec![dynamic("rows", 8), Dimension::Static(2)],
            to_dimension_vector(&[4]),
            to_dimension_vector(&[4, 2]),
            constant.then_some(indices.as_slice()),
        );
        let mut graph = MLGraphBuilder::new(&mut context)
            .unwrap()
            .build_graph_info(graph_info)
            .unwrap();
        let indices = (!constant).then(|| index_tensor(&mut context, &[4], &indices));
        for rows in [1u64, 4, 2, 1] {
            let values: Vec<f32> = (1..=rows)
                .flat_map(|row| [10.0 * row as f32, 10.0 * row as f32 + 1.0])
                .collect();
            let data = data_tensor(&mut context, &[rows, 2], &values);
            let last = rows as f32 * 10.0;
            check_dispatch(
                &mut context,
                &mut graph,
                &data,
                indices.as_ref(),
                &[4, 2],
                &[last, last + 1., 10., 11., last, last + 1., 10., 11.],
                &format!("gather rows={rows}, constant={constant}"),
            );
        }
    }
}

#[test]
fn gather_elements_uses_active_nonzero_axis() {
    let indices = [-1, 0, 100, -100, 100, -100, -1, 0];
    for constant in [false, true] {
        let mut context = context();
        let graph_info = gather_graph(
            "gatherElements",
            Some(1),
            vec![Dimension::Static(2), dynamic("columns", 8)],
            to_dimension_vector(&[2, 4]),
            to_dimension_vector(&[2, 4]),
            constant.then_some(indices.as_slice()),
        );
        let mut graph = MLGraphBuilder::new(&mut context)
            .unwrap()
            .build_graph_info(graph_info)
            .unwrap();
        let indices = (!constant).then(|| index_tensor(&mut context, &[2, 4], &indices));
        for columns in [2u64, 6, 3, 2] {
            let values: Vec<f32> = [10., 20.]
                .into_iter()
                .flat_map(|base| (0..columns).map(move |column| base + column as f32))
                .collect();
            let data = data_tensor(&mut context, &[2, columns], &values);
            let last = columns as f32 - 1.;
            check_dispatch(
                &mut context,
                &mut graph,
                &data,
                indices.as_ref(),
                &[2, 4],
                &[
                    10. + last,
                    10.,
                    10. + last,
                    10.,
                    20. + last,
                    20.,
                    20. + last,
                    20.,
                ],
                &format!("gatherElements columns={columns}, constant={constant}"),
            );
        }
    }
}

fn check_gather_nd(dynamic_columns: bool, constant: bool) {
    let indices = [-1, -1, 0, 100, 100, 0, -100, -100];
    let mut context = context();
    let graph_info = gather_graph(
        "gatherND",
        None,
        vec![
            dynamic("rows", 8),
            if dynamic_columns {
                dynamic("columns", 5)
            } else {
                Dimension::Static(3)
            },
            Dimension::Static(2),
        ],
        to_dimension_vector(&[4, 2]),
        to_dimension_vector(&[4, 2]),
        constant.then_some(indices.as_slice()),
    );
    let mut graph = MLGraphBuilder::new(&mut context)
        .unwrap()
        .build_graph_info(graph_info)
        .unwrap();
    let indices = (!constant).then(|| index_tensor(&mut context, &[4, 2], &indices));
    let dimensions = if dynamic_columns {
        [(2u64, 3u64), (4, 2), (1, 4), (2, 3)]
    } else {
        [(2, 3), (5, 3), (1, 3), (2, 3)]
    };
    for (rows, columns) in dimensions {
        let values: Vec<f32> = (0..rows)
            .flat_map(|row| {
                (0..columns).flat_map(move |column| {
                    let base = 100. * row as f32 + 10. * column as f32;
                    [base + 1., base + 2.]
                })
            })
            .collect();
        let data = data_tensor(&mut context, &[rows, columns, 2], &values);
        let last_row = 100. * (rows - 1) as f32;
        let last_column = 10. * (columns - 1) as f32;
        check_dispatch(
            &mut context,
            &mut graph,
            &data,
            indices.as_ref(),
            &[4, 2],
            &[
                last_row + last_column + 1.,
                last_row + last_column + 2.,
                last_column + 1.,
                last_column + 2.,
                last_row + 1.,
                last_row + 2.,
                1.,
                2.,
            ],
            &format!("gatherND rows={rows}, columns={columns}, constant={constant}"),
        );
    }
}

#[test]
fn gather_nd_normalizes_mixed_static_and_dynamic_components() {
    for constant in [false, true] {
        check_gather_nd(false, constant);
    }
}

#[test]
fn gather_nd_uses_each_active_indexed_dimension() {
    check_gather_nd(true, false);
}

#[test]
fn scalar_gather_uses_active_data_extent() {
    for constant in [false, true] {
        let mut context = context();
        let graph_info = gather_graph(
            "gather",
            Some(0),
            vec![dynamic("length", 8)],
            vec![],
            vec![],
            constant.then_some([-1i32].as_slice()),
        );
        let mut graph = MLGraphBuilder::new(&mut context)
            .unwrap()
            .build_graph_info(graph_info)
            .unwrap();
        let indices = (!constant).then(|| index_tensor(&mut context, &[], &[-1]));
        for length in [2u64, 4, 1, 2] {
            let values: Vec<f32> = (1..=length).map(|value| value as f32).collect();
            let data = data_tensor(&mut context, &[length], &values);
            let cases: &[(i32, f32)] = if constant {
                &[(-1, length as f32)]
            } else {
                &[
                    (-1, length as f32),
                    (100, length as f32),
                    (-100, 1.),
                    (0, 1.),
                ]
            };
            for &(value, expected) in cases {
                if let Some(indices) = indices.as_ref() {
                    context.write_tensor(indices, &[value]).unwrap();
                }
                check_dispatch(
                    &mut context,
                    &mut graph,
                    &data,
                    indices.as_ref(),
                    &[],
                    &[expected],
                    &format!("scalar gather length={length}, index={value}, constant={constant}"),
                );
            }
        }
    }
}

#[test]
fn gather_keeps_static_indexed_axis_when_other_axis_is_dynamic() {
    let mut context = context();
    let rows = dynamic("rows", 8);
    let graph_info = gather_graph(
        "gather",
        Some(1),
        vec![rows.clone(), Dimension::Static(3)],
        to_dimension_vector(&[2]),
        vec![rows, Dimension::Static(2)],
        Some(&[-1, 100]),
    );
    let mut graph = MLGraphBuilder::new(&mut context)
        .unwrap()
        .build_graph_info(graph_info)
        .unwrap();
    for rows in [1u64, 4, 2, 1] {
        let values: Vec<f32> = (0..rows)
            .flat_map(|row| {
                [
                    10. * row as f32 + 1.,
                    10. * row as f32 + 2.,
                    10. * row as f32 + 3.,
                ]
            })
            .collect();
        let expected: Vec<f32> = (0..rows)
            .flat_map(|row| [10. * row as f32 + 3.; 2])
            .collect();
        let data = data_tensor(&mut context, &[rows, 3], &values);
        check_dispatch(
            &mut context,
            &mut graph,
            &data,
            None,
            &[rows, 2],
            &expected,
            &format!("static indexed axis, rows={rows}"),
        );
    }
}

#[test]
fn scalar_index_gather_preserves_dynamic_nonindexed_output_on_dispatch() {
    let mut context = context();
    let rows = dynamic("rows", 8);
    let graph_info = gather_graph(
        "gather",
        Some(1),
        vec![rows.clone(), Dimension::Static(3)],
        vec![],
        vec![rows],
        Some(&[-1]),
    );
    let mut graph = MLGraphBuilder::new(&mut context)
        .unwrap()
        .build_graph_info(graph_info)
        .unwrap();
    for rows in [4u64, 2, 1, 8, 4] {
        let values: Vec<f32> = (0..rows)
            .flat_map(|row| {
                [
                    10. * row as f32 + 1.,
                    10. * row as f32 + 2.,
                    10. * row as f32 + 3.,
                ]
            })
            .collect();
        let expected: Vec<f32> = (0..rows).map(|row| 10. * row as f32 + 3.).collect();
        let data = data_tensor(&mut context, &[rows, 3], &values);
        check_dispatch(
            &mut context,
            &mut graph,
            &data,
            None,
            &[rows],
            &expected,
            &format!("scalar-index gather, nonindexed rows={rows}"),
        );
    }
}

#[test]
fn scalar_index_gather_preserves_singletons_on_first_middle_and_last_axes() {
    // Include rank five, where native scalar-index gather is also problematic.
    // Both the indexed and a retained axis vary on the same loaded model.
    for (rank, axis, row_axis) in [(3, 0, 2), (3, 1, 0), (3, 2, 1), (5, 2, 1), (5, 4, 1)] {
        for constant_index in [Some(-100), Some(-1), Some(100), None] {
            let mut context = context();
            let mut shape = vec![Dimension::Static(1); rank];
            shape[axis] = dynamic("columns", 5);
            shape[row_axis] = dynamic("rows", 8);
            let mut output_shape = shape.clone();
            output_shape.remove(axis);
            let graph_info = gather_graph(
                "gather",
                Some(axis as u32),
                shape,
                vec![],
                output_shape,
                constant_index.as_ref().map(std::slice::from_ref),
            );
            let mut graph = MLGraphBuilder::new(&mut context)
                .unwrap()
                .build_graph_info(graph_info)
                .unwrap();
            for (rows, columns) in [(1u64, 1u64), (4, 3), (8, 5), (2, 2), (1, 1)] {
                let mut shape = vec![1; rank];
                shape[axis] = columns;
                shape[row_axis] = rows;
                let values: Vec<f32> = (0..rows * columns).map(|i| i as f32 + 1.).collect();
                let data = data_tensor(&mut context, &shape, &values);
                let stride: u64 = shape[axis + 1..].iter().product();
                let mut output_shape = shape.clone();
                output_shape.remove(axis);
                let indices = constant_index.map_or_else(|| vec![-100, -1, 0, 100], |i| vec![i]);
                for index in indices {
                    let selected = if index < 0 {
                        (columns as i32 + index).max(0)
                    } else {
                        index.min(columns as i32 - 1)
                    } as usize;
                    let expected: Vec<f32> = values
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| (i / stride as usize) % columns as usize == selected)
                        .map(|(_, &value)| value)
                        .collect();
                    let indices = constant_index
                        .is_none()
                        .then(|| index_tensor(&mut context, &[], &[index]));
                    check_dispatch(
                        &mut context,
                        &mut graph,
                        &data,
                        indices.as_ref(),
                        &output_shape,
                        &expected,
                        &format!(
                            "rank={rank}, axis={axis}, rows={rows}, columns={columns}, index={index}, constant={constant_index:?}"
                        ),
                    );
                }
            }
        }
    }
}

#[test]
fn gather_reports_actual_scalar_and_dynamic_nonindexed_output_shapes() {
    // The direct executor exposes CoreML's output shape, unlike the public
    // tensor's preallocated descriptor. It compiles separately per invocation;
    // the tests above exercise reuse of one compiled and loaded model.
    for scalar in [true, false] {
        let rows = dynamic("rows", 8);
        let graph_info = if scalar {
            gather_graph("gather", Some(0), vec![rows], vec![], vec![], Some(&[-1]))
        } else {
            gather_graph(
                "gather",
                Some(1),
                vec![rows.clone(), Dimension::Static(3)],
                vec![],
                vec![rows],
                Some(&[-1]),
            )
        };
        let converted = CoremlMlProgramConverter.convert(&graph_info).unwrap();
        assert!(converted.weights_data.is_none());
        let mut descriptors: Vec<_> = graph_info
            .operands
            .iter()
            .map(|operand| operand.descriptor.clone())
            .collect();
        if scalar {
            // CoreML's model boundary represents a WebNN scalar as [1]. The
            // dispatch scalar test separately verifies the public rank-0 API.
            descriptors[2].shape = to_dimension_vector(&[1]);
        }
        let input_descriptors = HashMap::from([("data".into(), descriptors[0].clone())]);
        let output_descriptors = HashMap::from([("result".into(), descriptors[2].clone())]);
        for rows in [4usize, 2, 1, 8] {
            let (shape, values, expected_shape, expected) = if scalar {
                (
                    vec![rows],
                    (1..=rows).map(|value| value as f32).collect(),
                    vec![1],
                    vec![rows as f32],
                )
            } else {
                (
                    vec![rows, 3],
                    (0..rows)
                        .flat_map(|row| {
                            [
                                10. * row as f32 + 1.,
                                10. * row as f32 + 2.,
                                10. * row as f32 + 3.,
                            ]
                        })
                        .collect(),
                    vec![rows as i64],
                    (0..rows).map(|row| 10. * row as f32 + 3.).collect(),
                )
            };
            let attempts = run_coreml_with_inputs_checked(
                &converted.data,
                vec![CoremlInput {
                    name: "data".into(),
                    shape,
                    data: values,
                }],
                &input_descriptors,
                &output_descriptors,
            )
            .unwrap();
            let mut cpu_succeeded = false;
            for attempt in attempts {
                if let Err(error) = &attempt.result {
                    eprintln!(
                        "scalar={scalar}, rows={rows}, {}: {error}",
                        attempt.compute_unit
                    );
                }
                if attempt.compute_unit == "CPU_ONLY" {
                    assert!(
                        attempt.result.is_ok(),
                        "scalar={scalar}, rows={rows}, CPU_ONLY: {:?}",
                        attempt.result
                    );
                    cpu_succeeded = true;
                }
                if let Ok(outputs) = attempt.result {
                    assert_eq!(outputs.len(), 1);
                    assert_eq!(outputs[0].name, "result");
                    assert_eq!(outputs[0].shape, expected_shape);
                    assert_eq!(outputs[0].data, expected);
                }
            }
            assert!(cpu_succeeded, "CPU_ONLY must produce checked output");
        }
    }
}

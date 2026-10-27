//! Quantization parameters obey the same contracts through the builder and graph import.

use rustnn::graph::{DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind};
use rustnn::mlcontext::{
    MLContextOptions, MLNamedOperands, MLOperandDescriptor, MLPowerPreference,
};
use rustnn::mlgraphbuilder::MLGraphBuilder;
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operators::Operation;

fn graph(
    quantize: bool,
    input_dtype: DataType,
    scale_dtype: DataType,
    zero_dtype: DataType,
    input_shape: &[u32],
    scale_shape: &[u32],
    zero_shape: &[u32],
) -> GraphInfo {
    let operand = |name: &str, dtype, shape: &[u32], kind| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: dtype,
            shape: shape.iter().copied().map(Dimension::Static).collect(),
            pending_permutation: vec![],
        },
    };
    GraphInfo {
        operands: vec![
            operand("input", input_dtype, input_shape, OperandKind::Input),
            operand("scale", scale_dtype, scale_shape, OperandKind::Input),
            operand("zero", zero_dtype, zero_shape, OperandKind::Input),
            operand(
                "result",
                if quantize { zero_dtype } else { scale_dtype },
                input_shape,
                OperandKind::Output,
            ),
        ],
        input_operands: vec![0, 1, 2],
        output_operands: vec![3],
        operations: vec![if quantize {
            Operation::QuantizeLinear {
                input: 0,
                scale: 1,
                zero_point: Some(2),
                options: None,
                outputs: vec![3],
            }
        } else {
            Operation::DequantizeLinear {
                input: 0,
                scale: 1,
                zero_point: Some(2),
                options: None,
                outputs: vec![3],
            }
        }],
        ..Default::default()
    }
}

fn builder_result(graph: &GraphInfo) -> rustnn::error::Result<GraphInfo> {
    builder_result_with_options(
        graph,
        &MLContextOptions::new(MLPowerPreference::Default, false),
    )
}

fn builder_result_with_options(
    graph: &GraphInfo,
    options: &MLContextOptions,
) -> rustnn::error::Result<GraphInfo> {
    let mut builder = MLGraphBuilder::new_uncompiled_with_options(options);
    let mut inputs = vec![];
    for operand in &graph.operands[..3] {
        inputs.push(
            builder.input(
                operand.name.as_ref().unwrap(),
                &MLOperandDescriptor::new(
                    MLOperandDataType::try_from(operand.descriptor.data_type).unwrap(),
                    operand
                        .descriptor
                        .static_or_max_shape()
                        .iter()
                        .map(|&v| u64::from(v))
                        .collect(),
                ),
            )?,
        );
    }
    let result = if matches!(graph.operations[0], Operation::QuantizeLinear { .. }) {
        builder.quantize_linear_with_zeropoint(inputs[0], inputs[1], inputs[2])?
    } else {
        builder.dequantize_linear_with_zeropoint(inputs[0], inputs[1], inputs[2])?
    };
    let mut outputs = MLNamedOperands::new();
    outputs.insert("result", result);
    builder.finish_graph_info(&outputs)
}

fn assert_imports(graph: &GraphInfo) {
    let interchange = rustnn::webnn_json::to_graph_json(graph, false).unwrap();
    rustnn::webnn_json::from_graph_json(&interchange)
        .expect("valid quantization parameters must survive graph import");
}

fn rejected(graph: &GraphInfo, reason: &str) {
    let error = builder_result(graph).expect_err("invalid parameters must fail at builder call");
    assert!(error.to_string().contains(reason), "{error}");
    let interchange = rustnn::webnn_json::to_graph_json(graph, false).unwrap();
    let error = rustnn::webnn_json::from_graph_json(&interchange)
        .expect_err("invalid parameters must fail through the graph recorder import path");
    assert!(error.to_string().contains(reason), "{error}");
}

#[test]
fn quantization_rejects_scalar_parameters_for_non_scalar_inputs() {
    for quantize in [false, true] {
        rejected(
            &graph(
                quantize,
                if quantize {
                    DataType::Float32
                } else {
                    DataType::Int8
                },
                DataType::Float32,
                DataType::Int8,
                &[2, 3],
                &[],
                &[],
            ),
            "rank",
        );
    }
}

#[test]
fn quantization_rejects_different_parameter_shapes() {
    for quantize in [false, true] {
        rejected(
            &graph(
                quantize,
                if quantize {
                    DataType::Float16
                } else {
                    DataType::Uint8
                },
                DataType::Float16,
                DataType::Uint8,
                &[2, 3],
                &[1, 3],
                &[1, 1],
            ),
            "shape",
        );
    }
}

#[test]
fn quantization_rejects_non_dividing_blockwise_parameters() {
    for quantize in [false, true] {
        rejected(
            &graph(
                quantize,
                if quantize {
                    DataType::Float32
                } else {
                    DataType::Int8
                },
                DataType::Float32,
                DataType::Int8,
                &[3, 5],
                &[2, 1],
                &[2, 1],
            ),
            "divide",
        );
    }
}

#[test]
fn quantization_rejects_mismatched_parameter_dtypes() {
    rejected(
        &graph(
            true,
            DataType::Float16,
            DataType::Float32,
            DataType::Int8,
            &[2],
            &[1],
            &[1],
        ),
        "scale",
    );
    rejected(
        &graph(
            false,
            DataType::Int8,
            DataType::Float32,
            DataType::Uint8,
            &[2],
            &[1],
            &[1],
        ),
        "zeroPoint",
    );
}

#[test]
fn quantization_rejects_non_quantization_parameter_dtypes() {
    rejected(
        &graph(
            true,
            DataType::Uint8,
            DataType::Float32,
            DataType::Int8,
            &[2],
            &[1],
            &[1],
        ),
        "input",
    );
    rejected(
        &graph(
            false,
            DataType::Float16,
            DataType::Float32,
            DataType::Int8,
            &[2],
            &[1],
            &[1],
        ),
        "input",
    );
    rejected(
        &graph(
            false,
            DataType::Int8,
            DataType::Int8,
            DataType::Int8,
            &[2],
            &[1],
            &[1],
        ),
        "scale",
    );
    rejected(
        &graph(
            true,
            DataType::Float32,
            DataType::Float32,
            DataType::Float32,
            &[2],
            &[1],
            &[1],
        ),
        "zeroPoint",
    );
}

#[test]
fn quantization_preserves_scalar_per_tensor_per_axis_and_blockwise_parameters() {
    for quantize in [false, true] {
        for (input_shape, scale_shape) in [
            (&[][..], &[][..]),
            (&[2, 3][..], &[1, 1][..]),
            (&[2, 3][..], &[1, 3][..]),
            (&[4, 6][..], &[2, 3][..]),
        ] {
            let graph = graph(
                quantize,
                if quantize {
                    DataType::Float16
                } else {
                    DataType::Int8
                },
                DataType::Float16,
                DataType::Int8,
                input_shape,
                scale_shape,
                scale_shape,
            );
            assert_imports(&graph);
            let built = builder_result(&graph).unwrap();
            assert_imports(&built);
            assert_eq!(
                built.operands[3].descriptor.data_type,
                graph.operands[3].descriptor.data_type
            );
            assert_eq!(
                built.operands[3].descriptor.shape,
                graph.operands[3].descriptor.shape
            );
        }
    }
}

#[test]
fn omitted_zero_point_preserves_the_existing_builder_extension() {
    let mut builder = MLGraphBuilder::new_uncompiled();
    let input = builder
        .input(
            "input",
            &MLOperandDescriptor::new(MLOperandDataType::Int8, vec![2]),
        )
        .unwrap();
    let scale = builder
        .input(
            "scale",
            &MLOperandDescriptor::new(MLOperandDataType::Float16, vec![1]),
        )
        .unwrap();
    let result = builder.dequantize_linear(input, scale).unwrap();
    assert_eq!(
        builder.rustnn_operand_data_type(result).unwrap(),
        MLOperandDataType::Float16
    );
}

#[test]
fn invalid_quantization_does_not_partially_record_an_operation() {
    let mut builder = MLGraphBuilder::new_uncompiled();
    let input = builder
        .input(
            "input",
            &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 3]),
        )
        .unwrap();
    let scalar_scale = builder
        .input(
            "scalar_scale",
            &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![]),
        )
        .unwrap();
    let scalar_zero = builder
        .input(
            "scalar_zero",
            &MLOperandDescriptor::new(MLOperandDataType::Int8, vec![]),
        )
        .unwrap();
    assert!(
        builder
            .quantize_linear_with_zeropoint(input, scalar_scale, scalar_zero)
            .is_err()
    );
    let scale = builder
        .input(
            "scale",
            &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, 1]),
        )
        .unwrap();
    let zero = builder
        .input(
            "zero",
            &MLOperandDescriptor::new(MLOperandDataType::Int8, vec![1, 1]),
        )
        .unwrap();
    let result = builder
        .quantize_linear_with_zeropoint(input, scale, zero)
        .unwrap();
    let mut outputs = MLNamedOperands::new();
    outputs.insert("result", result);
    let graph = builder.finish_graph_info(&outputs).unwrap();
    assert_eq!(graph.operations.len(), 1);
    assert_eq!(graph.operands.len(), 6);
    assert_imports(&graph);
}

#[test]
fn quantization_integer_extensions_require_explicit_options() {
    let options =
        MLContextOptions::new(MLPowerPreference::Default, false).with_rustnn_allow_int4_uint4(true);
    for dtype in [DataType::Int4, DataType::Uint4, DataType::Int32] {
        for quantize in [false, true] {
            let graph = graph(
                quantize,
                if quantize { DataType::Float32 } else { dtype },
                DataType::Float32,
                dtype,
                &[2],
                &[1],
                &[1],
            );
            if dtype == DataType::Int32 {
                GraphValidator::new(&graph, ContextProperties::default())
                    .validate()
                    .unwrap();
                builder_result(&graph).unwrap();
            } else {
                assert!(
                    GraphValidator::new(&graph, ContextProperties::default())
                        .validate()
                        .is_err()
                );
                assert!(builder_result(&graph).is_err());
            }
            let built = builder_result_with_options(&graph, &options).unwrap();
            rustnn::operation_type_support::check_allowed_data_types_with_options(
                &built.operations[0],
                &built,
                &options,
            )
            .unwrap();
            let json = rustnn::webnn_json::to_graph_json(&built, false).unwrap();
            rustnn::webnn_json::from_graph_json_owned_with_options(json, &options).unwrap();
        }
    }
    // The packed-integer policy must not enable the separate int32-input extension.
    let graph = graph(
        true,
        DataType::Int32,
        DataType::Float32,
        DataType::Int8,
        &[2],
        &[1],
        &[1],
    );
    assert!(
        GraphValidator::new(&graph, ContextProperties::default())
            .validate()
            .is_err()
    );
    assert!(builder_result_with_options(&graph, &options).is_err());
    builder_result_with_options(
        &graph,
        &options.with_rustnn_bypass_allowed_data_types_checks(true),
    )
    .unwrap();
}

/// The specification's tensor-limits tables allow uint32 for the dequantize input and for both
/// zero points (https://www.w3.org/TR/webnn/#tensor-limits-dequantizelinear,
/// https://www.w3.org/TR/webnn/#tensor-limits-quantizelinear).
#[test]
fn quantization_accepts_the_specification_uint32_parameters() {
    for quantize in [false, true] {
        let graph = graph(
            quantize,
            if quantize {
                DataType::Float32
            } else {
                DataType::Uint32
            },
            DataType::Float32,
            DataType::Uint32,
            &[2],
            &[1],
            &[1],
        );
        assert_imports(&graph);
        assert_imports(&builder_result(&graph).unwrap());
    }
}

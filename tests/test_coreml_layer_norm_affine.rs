//! Layer normalization retains affine parameters on nontrailing axes.

use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::GraphInfo;
use rustnn::mlcontext::{MLNamedOperands, MLOperandDescriptor};
use rustnn::mlgraphbuilder::MLGraphBuilder;
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::MLLayerNormalizationOptions;
use rustnn::protos::coreml::specification::{Model, model};

fn graph(axes: &[u32], runtime: bool, bias: bool) -> GraphInfo {
    let mut builder = MLGraphBuilder::new_uncompiled();
    let input = builder
        .input(
            "input",
            &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 1, 4, 3]),
        )
        .unwrap();
    let shape = if axes == [0, 2] {
        vec![2, 4]
    } else {
        vec![4, 2]
    };
    let parameter = MLOperandDescriptor::new(MLOperandDataType::Float32, shape);
    let mut create = |name, values: &[f32]| {
        if runtime {
            builder.input(name, &parameter).unwrap()
        } else {
            builder
                .constant_from_bytes(
                    &parameter,
                    values.iter().flat_map(|x| x.to_le_bytes()).collect(),
                )
                .unwrap()
        }
    };
    let scale = create("scale", &scales());
    let bias = bias.then(|| create("bias", &biases()));
    let output = builder
        .layer_normalization_with_options(
            input,
            MLLayerNormalizationOptions {
                axes: Some(axes.to_vec()),
                scale: Some(scale.into()),
                bias: bias.map(Into::into),
                ..Default::default()
            },
        )
        .unwrap();
    let mut outputs = MLNamedOperands::new();
    outputs.insert("result", output);
    builder.finish_graph_info(&outputs).unwrap()
}

fn scales() -> [f32; 8] {
    [2.0, -3.0, 5.0, -7.0, 11.0, -13.0, 17.0, -19.0]
}

fn biases() -> [f32; 8] {
    [0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0]
}

#[test]
fn float32_layer_norm_applies_constant_and_runtime_affine_outside_native_kernel() {
    for axes in [[0, 2], [2, 0]] {
        for runtime in [false, true] {
            for bias in [false, true] {
                let graph = graph(&axes, runtime, bias);
                let before = serde_json::to_value(&graph).unwrap();
                let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
                assert_eq!(serde_json::to_value(&graph).unwrap(), before);
                let model = Model::decode(converted.data.as_slice()).unwrap();
                let Some(model::Type::Pipeline(pipeline)) = model.r#type else {
                    panic!("normalization must be materialized before affine work")
                };
                assert_eq!(pipeline.models.len(), 2);
                let Some(model::Type::MlProgram(program)) = &pipeline.models[0].r#type else {
                    panic!("normalization child is an MLProgram")
                };
                let function = &program.functions["main"];
                let operations = &function.block_specializations[&function.opset].operations;
                let norm = operations
                    .iter()
                    .find(|operation| operation.r#type == "layer_norm")
                    .unwrap();
                assert!(!norm.inputs.contains_key("gamma"));
                assert!(!norm.inputs.contains_key("beta"));
                assert!(
                    !operations
                        .iter()
                        .any(|operation| matches!(operation.r#type.as_str(), "mul" | "add"))
                );
                let Some(model::Type::MlProgram(program)) = &pipeline.models[1].r#type else {
                    panic!("affine child is an MLProgram")
                };
                let function = &program.functions["main"];
                let operations = &function.block_specializations[&function.opset].operations;
                assert!(operations.iter().any(|operation| operation.r#type == "mul"));
                assert_eq!(
                    operations.iter().any(|operation| operation.r#type == "add"),
                    bias
                );
                assert_eq!(
                    operations
                        .iter()
                        .any(|operation| operation.r#type == "transpose"),
                    axes == [2, 0]
                );
            }
        }
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn float32_layer_norm_nontrailing_axes_retains_affine_values() {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};

    let input: Vec<_> = (0..24).map(|x| x as f32 - 12.0).collect();
    for axes in [[0, 2], [2, 0]] {
        for runtime in [false, true] {
            for with_bias in [false, true] {
                let graph = graph(&axes, runtime, with_bias);
                let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
                let mut inputs = vec![CoremlInput {
                    name: "input".into(),
                    shape: vec![2, 1, 4, 3],
                    data: input.clone(),
                }];
                if runtime {
                    let shape = if axes == [0, 2] {
                        vec![2, 4]
                    } else {
                        vec![4, 2]
                    };
                    inputs.push(CoremlInput {
                        name: "scale".into(),
                        shape: shape.clone(),
                        data: scales().to_vec(),
                    });
                    if with_bias {
                        inputs.push(CoremlInput {
                            name: "bias".into(),
                            shape,
                            data: biases().to_vec(),
                        });
                    }
                }
                let mut expected = vec![0.0f32; 24];
                for column in 0..3 {
                    let values: Vec<_> = (0..8)
                        .map(|row| f64::from(input[row * 3 + column]))
                        .collect();
                    let mean = values.iter().sum::<f64>() / 8.0;
                    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 8.0;
                    for batch in 0..2 {
                        for row in 0..4 {
                            let index = (batch * 4 + row) * 3 + column;
                            let affine = if axes == [0, 2] {
                                batch * 4 + row
                            } else {
                                row * 2 + batch
                            };
                            let normalized =
                                (f64::from(input[index]) - mean) / (variance + 1e-5).sqrt();
                            expected[index] = (normalized * f64::from(scales()[affine])
                                + if with_bias {
                                    f64::from(biases()[affine])
                                } else {
                                    0.0
                                }) as f32;
                        }
                    }
                }
                for attempt in run_coreml_with_inputs_with_weights(
                    &converted.data,
                    converted.weights_data.as_deref(),
                    inputs,
                )
                .unwrap()
                {
                    let outputs = attempt.result.unwrap();
                    let result = outputs
                        .iter()
                        .find(|output| output.name == "result")
                        .unwrap();
                    assert_eq!(result.shape, [2, 1, 4, 3]);
                    for (index, (&actual, &expected)) in
                        result.data.iter().zip(&expected).enumerate()
                    {
                        // WebNN layerNormalization float32 accuracy is 16 ULP.
                        assert!(
                            actual.is_finite()
                                && actual.to_bits().abs_diff(expected.to_bits()) <= 16,
                            "{}, axes={axes:?}, runtime={runtime}, bias={with_bias}, index={index}: {actual} != {expected}",
                            attempt.compute_unit
                        );
                    }
                }
            }
        }
    }
}

fn operations(model: &Model) -> Vec<rustnn::protos::coreml::mil_spec::Operation> {
    let programs: Vec<&Model> = match &model.r#type {
        Some(model::Type::Pipeline(pipeline)) => pipeline.models.iter().collect(),
        _ => vec![model],
    };
    programs
        .into_iter()
        .flat_map(|model| {
            let Some(model::Type::MlProgram(program)) = &model.r#type else {
                panic!("child is an MLProgram")
            };
            let function = &program.functions["main"];
            function.block_specializations[&function.opset]
                .operations
                .clone()
        })
        .collect()
}

/// `[rows, 4, 3]` normalized over `axes`, with constant or runtime scale/bias in axis order.
fn layer_norm(
    dtype: MLOperandDataType,
    rows: u64,
    axes: &[u32],
    runtime: bool,
) -> rustnn::graph::GraphInfo {
    let input_shape = [rows, 4, 3];
    let mut builder = MLGraphBuilder::new_uncompiled();
    let input = builder
        .input(
            "input",
            &MLOperandDescriptor::new(dtype, input_shape.to_vec()),
        )
        .unwrap();
    let parameter_shape: Vec<u64> = axes
        .iter()
        .map(|&axis| input_shape[axis as usize])
        .collect();
    let count: u64 = parameter_shape.iter().product();
    let parameter = MLOperandDescriptor::new(dtype, parameter_shape);
    let mut create = |name: &str, offset: f32| {
        if runtime {
            builder.input(name, &parameter).unwrap()
        } else {
            let values = (0..count).map(|i| offset + i as f32 * 0.5);
            let bytes = match dtype {
                MLOperandDataType::Float16 => values
                    .flat_map(|x| half::f16::from_f32(x).to_le_bytes())
                    .collect(),
                _ => values.flat_map(|x| x.to_le_bytes()).collect(),
            };
            builder.constant_from_bytes(&parameter, bytes).unwrap()
        }
    };
    let scale = create("scale", 1.0);
    let bias = create("bias", -0.25);
    let output = builder
        .layer_normalization_with_options(
            input,
            MLLayerNormalizationOptions {
                axes: Some(axes.to_vec()),
                scale: Some(scale.into()),
                bias: Some(bias.into()),
                ..Default::default()
            },
        )
        .unwrap();
    let mut outputs = MLNamedOperands::new();
    outputs.insert("result", output);
    builder.finish_graph_info(&outputs).unwrap()
}

#[test]
fn trailing_axes_keep_constant_affine_in_one_native_layer_norm() {
    use rustnn::mlcontext::CoremlFloat16Precision;
    // Trailing ascending axes: native gamma/beta match WebNN's scale and bias exactly,
    // so neither float dtype needs explicit affine work or a Pipeline cut.
    for dtype in [MLOperandDataType::Float32, MLOperandDataType::Float16] {
        for axes in [&[2][..], &[1, 2][..]] {
            let graph = layer_norm(dtype, 2, axes, false);
            let converted = CoremlMlProgramConverter
                .convert_with_float16_precision(&graph, CoremlFloat16Precision::Native)
                .unwrap();
            let model = Model::decode(converted.data.as_slice()).unwrap();
            assert!(
                matches!(model.r#type, Some(model::Type::MlProgram(_))),
                "{dtype:?} {axes:?} stays one MLProgram"
            );
            let operations = operations(&model);
            let norms: Vec<_> = operations
                .iter()
                .filter(|operation| operation.r#type == "layer_norm")
                .collect();
            assert_eq!(norms.len(), 1);
            assert!(norms[0].inputs.contains_key("gamma") && norms[0].inputs.contains_key("beta"));
            assert!(
                !operations
                    .iter()
                    .any(|operation| matches!(operation.r#type.as_str(), "mul" | "add" | "cast")),
                "{dtype:?} {axes:?}: {:?}",
                operations.iter().map(|o| &o.r#type).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn nontrailing_axes_and_runtime_affine_stay_explicit_for_both_float_types() {
    for dtype in [MLOperandDataType::Float32, MLOperandDataType::Float16] {
        for (axes, runtime) in [(&[0, 1][..], false), (&[2, 1][..], false), (&[2][..], true)] {
            let graph = layer_norm(dtype, 2, axes, runtime);
            let model = Model::decode(
                CoremlMlProgramConverter
                    .convert(&graph)
                    .unwrap()
                    .data
                    .as_slice(),
            )
            .unwrap();
            let operations = operations(&model);
            let norm = operations
                .iter()
                .find(|operation| operation.r#type == "layer_norm")
                .unwrap();
            assert!(!norm.inputs.contains_key("gamma") && !norm.inputs.contains_key("beta"));
            assert!(operations.iter().any(|operation| operation.r#type == "mul"));
            assert!(operations.iter().any(|operation| operation.r#type == "add"));
            // Only nontrailing or permuted axes need the Pipeline cut that keeps
            // Core ML from fusing the explicit affine work back into layer_norm.
            assert_eq!(
                matches!(model.r#type, Some(model::Type::Pipeline(_))),
                !runtime,
                "{dtype:?} {axes:?} runtime={runtime}"
            );
        }
    }
}

#[test]
fn exact_float16_layer_norm_normalizes_and_applies_affine_in_float32() {
    use rustnn::mlcontext::CoremlFloat16Precision;
    use rustnn::protos::coreml::mil_spec::{DataType as MilDataType, value_type};
    let graph = layer_norm(MLOperandDataType::Float16, 2, &[2], false);
    let model = Model::decode(
        CoremlMlProgramConverter
            .convert_with_float16_precision(&graph, CoremlFloat16Precision::Exact)
            .unwrap()
            .data
            .as_slice(),
    )
    .unwrap();
    let operations = operations(&model);
    let norm = operations
        .iter()
        .find(|operation| operation.r#type == "layer_norm")
        .unwrap();
    let Some(value_type::Type::TensorType(tensor)) =
        norm.outputs[0].r#type.as_ref().unwrap().r#type.as_ref()
    else {
        panic!("tensor")
    };
    assert_eq!(tensor.data_type, MilDataType::Float32 as i32);
    assert!(!norm.inputs.contains_key("gamma"));
}

#[cfg(feature = "dynamic-inputs")]
fn with_dynamic_rows(mut graph: rustnn::graph::GraphInfo) -> rustnn::graph::GraphInfo {
    // The reviewer's case: a bounded leading dimension, [rows<=3, 4, 3].
    for &id in graph
        .input_operands
        .iter()
        .take(1)
        .chain(&graph.output_operands)
    {
        graph.operands[id as usize].descriptor.shape[0] =
            rustnn::graph::Dimension::Dynamic(rustnn::graph::DynamicDimension {
                name: "rows".into(),
                max_size: 3,
            });
    }
    graph
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn bounded_dynamic_layer_norm_with_affine_converts_for_both_float16_lowerings() {
    use rustnn::mlcontext::CoremlFloat16Precision;
    for dtype in [MLOperandDataType::Float32, MLOperandDataType::Float16] {
        // Trailing (fused), nontrailing (explicit affine and a Pipeline cut) and runtime.
        for (axes, runtime) in [(&[2][..], false), (&[1][..], false), (&[1, 2][..], true)] {
            for precision in [
                CoremlFloat16Precision::Native,
                CoremlFloat16Precision::Exact,
            ] {
                let graph = with_dynamic_rows(layer_norm(dtype, 3, axes, runtime));
                CoremlMlProgramConverter
                    .convert_with_float16_precision(&graph, precision)
                    .unwrap_or_else(|error| {
                        panic!("{dtype:?} {axes:?} runtime={runtime} {precision:?}: {error}")
                    });
            }
        }
    }
}

#[cfg(all(
    target_os = "macos",
    feature = "coreml-runtime",
    feature = "dynamic-inputs"
))]
#[test]
fn bounded_dynamic_nontrailing_layer_norm_runs_below_its_bound() {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
    let graph = with_dynamic_rows(layer_norm(MLOperandDataType::Float32, 3, &[1], false));
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let rows = 2;
    let input: Vec<f32> = (0..rows * 12).map(|x| x as f32 * 0.75 - 4.0).collect();
    let mut expected = vec![0.0f32; input.len()];
    for row in 0..rows {
        for column in 0..3 {
            let values: Vec<f64> = (0..4)
                .map(|i| f64::from(input[row * 12 + i * 3 + column]))
                .collect();
            let mean = values.iter().sum::<f64>() / 4.0;
            let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 4.0;
            for (i, value) in values.iter().enumerate() {
                let normalized = (value - mean) / (variance + 1e-5).sqrt();
                let scale = 1.0 + i as f64 * 0.5;
                let bias = -0.25 + i as f64 * 0.5;
                expected[row * 12 + i * 3 + column] = (normalized * scale + bias) as f32;
            }
        }
    }
    for attempt in run_coreml_with_inputs_with_weights(
        &converted.data,
        converted.weights_data.as_deref(),
        vec![CoremlInput {
            name: "input".into(),
            shape: vec![rows, 4, 3],
            data: input.clone(),
        }],
    )
    .unwrap()
    {
        let outputs = attempt.result.unwrap();
        let result = outputs
            .iter()
            .find(|output| output.name == "result")
            .unwrap();
        assert_eq!(result.shape, [rows as i64, 4, 3]);
        for (index, (&actual, &expected)) in result.data.iter().zip(&expected).enumerate() {
            assert!(
                actual.is_finite() && actual.to_bits().abs_diff(expected.to_bits()) <= 16,
                "{}, index={index}: {actual} != {expected}",
                attempt.compute_unit
            );
        }
    }
}

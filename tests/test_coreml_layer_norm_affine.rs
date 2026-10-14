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

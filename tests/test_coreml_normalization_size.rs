//! Normalization affine parameters must not expand into activation-sized constants.
#![cfg(feature = "coreml-converter")]

use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::mlcontext::{MLNamedOperands, MLOperandDescriptor};
use rustnn::mlgraphbuilder::MLGraphBuilder;
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::{MLInstanceNormalizationOptions, MLLayerNormalizationOptions};

fn convert(shape: Vec<u64>, layer: bool, runtime: bool, nhwc: bool) -> (usize, usize) {
    let mut builder = MLGraphBuilder::new_uncompiled();
    let input = builder
        .input(
            "input",
            &MLOperandDescriptor::new(MLOperandDataType::Float32, shape),
        )
        .unwrap();
    let descriptor = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1]);
    let mut parameter = |name, value: f32| {
        if runtime {
            builder.input(name, &descriptor).unwrap()
        } else {
            builder
                .constant_from_bytes(&descriptor, value.to_le_bytes().to_vec())
                .unwrap()
        }
    };
    let scale = parameter("scale", 1.0);
    let bias = parameter("bias", 0.0);
    let output = if layer {
        builder
            .layer_normalization_with_options(
                input,
                MLLayerNormalizationOptions {
                    axes: Some(vec![1]),
                    scale: Some(scale.into()),
                    bias: Some(bias.into()),
                    ..Default::default()
                },
            )
            .unwrap()
    } else {
        builder
            .instance_normalization_with_options(
                input,
                MLInstanceNormalizationOptions {
                    layout: if nhwc { "nhwc" } else { "nchw" }.into(),
                    scale: Some(scale.into()),
                    bias: Some(bias.into()),
                    ..Default::default()
                },
            )
            .unwrap()
    };
    let mut outputs = MLNamedOperands::new();
    outputs.insert("result", output);
    let graph = builder.finish_graph_info(&outputs).unwrap();
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    (
        converted.data.len(),
        converted.weights_data.as_ref().map_or(0, Vec::len),
    )
}

#[test]
fn affine_normalization_model_size_does_not_scale_with_activations() {
    for layer in [false, true] {
        for runtime in [false, true] {
            for nhwc in [false, true] {
                if layer && nhwc {
                    continue;
                }
                let (small, large) = if nhwc {
                    (vec![2, 3, 1, 1], vec![512, 32256, 1, 1])
                } else {
                    (vec![2, 1, 3, 1], vec![512, 1, 32256, 1])
                };
                let small = convert(small, layer, runtime, nhwc);
                let large = convert(large, layer, runtime, nhwc);
                assert!(
                    large.0 < 64 * 1024,
                    "model bytes: {large:?}; layer={layer} runtime={runtime} nhwc={nhwc}"
                );
                assert!(
                    large.0.abs_diff(small.0) < 1024,
                    "small={small:?} large={large:?}"
                );
                assert_eq!(
                    large.1, small.1,
                    "affine weights must not grow with activation shape"
                );
            }
        }
    }
}

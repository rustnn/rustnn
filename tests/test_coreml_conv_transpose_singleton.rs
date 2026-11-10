//! Height-only transposed convolutions use the correct native spatial axis.
#![cfg(feature = "coreml-runtime")]
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
use rustnn::mlcontext::{MLNamedOperands, MLOperandDescriptor};
use rustnn::mlgraphbuilder::MLGraphBuilder;
use rustnn::operator_enums::MLOperandDataType;

#[test]
fn height_only_transposed_convolution_matches_direct_sum() {
    use rustnn::operator_options::MLConvTranspose2dOptions;
    for (groups, padding, dilation) in [(1, 0, 1), (2, 1, 2)] {
        let mut b = MLGraphBuilder::new_uncompiled();
        let input = b
            .input(
                "x",
                &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, 2, 5, 1]),
            )
            .unwrap();
        let channels = if groups == 1 { 3 } else { 1 };
        let weights: Vec<f32> = (0..2 * channels * 8).map(|i| i as f32 / 48.).collect();
        let filter = b
            .constant_from_bytes(
                &MLOperandDescriptor::new(
                    MLOperandDataType::Float32,
                    vec![2, channels as u64, 8, 1],
                ),
                weights.iter().flat_map(|x| x.to_le_bytes()).collect(),
            )
            .unwrap();
        let output = b
            .conv_transpose2d_with_options(
                input,
                filter,
                MLConvTranspose2dOptions {
                    strides: vec![4, 1],
                    padding: vec![padding, padding, 0, 0],
                    dilations: vec![dilation, 1],
                    groups,
                    ..Default::default()
                },
            )
            .unwrap();
        let mut outputs = MLNamedOperands::new();
        outputs.insert("result", output);
        let graph = b.finish_graph_info(&outputs).unwrap();
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let x: Vec<f32> = (0..10).map(|i| (i as f32 - 8.) / 16.).collect();
        let length = 16 + 7 * dilation + 1 - 2 * padding;
        let mut expected = vec![0.; (channels * groups * length) as usize];
        for ic in 0..2 {
            for oc in 0..channels {
                for pos in 0..5 {
                    for k in 0..8 {
                        let target = (pos * 4 + k * dilation) as i32 - padding as i32;
                        if target >= 0 && target < length as i32 {
                            let out_channel = (ic / (2 / groups)) * channels + oc;
                            expected[(out_channel * length) as usize + target as usize] += x
                                [(ic * 5 + pos) as usize]
                                * weights[((ic * channels + oc) * 8 + k) as usize];
                        }
                    }
                }
            }
        }
        for attempt in run_coreml_with_inputs_with_weights(
            &converted.data,
            converted.weights_data.as_deref(),
            vec![CoremlInput {
                name: "x".into(),
                shape: vec![1, 2, 5, 1],
                data: x,
            }],
        )
        .unwrap()
        {
            let actual = attempt.result.unwrap();
            assert_eq!(actual[0].data.len(), expected.len());
            for (index, (actual, expected)) in actual[0].data.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "{} groups={groups} at {index}: {actual} != {expected}",
                    attempt.compute_unit
                );
            }
        }
    }
}

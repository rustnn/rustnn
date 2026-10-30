//! Resolve static positional embeddings before BNNS compiles their broadcast/trig/concat chain.
#![cfg(feature = "coreml-runtime")]
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
use rustnn::mlcontext::{MLNamedOperands, MLOperandDescriptor};
use rustnn::mlgraphbuilder::MLGraphBuilder;
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::MLTransposeOptions;

#[test]
fn constant_positional_embedding_compiles_and_matches_reference() {
    let mut builder = MLGraphBuilder::new_uncompiled();
    let positions: Vec<f32> = (0..1344).map(|i| i as f32).collect();
    let divisors: Vec<f32> = (0..256).map(|i| 10000f32.powf(i as f32 / 256.)).collect();
    let descriptor = |shape| MLOperandDescriptor::new(MLOperandDataType::Float32, shape);
    let position = builder
        .constant_from_bytes(
            &descriptor(vec![1344, 1, 1]),
            positions.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
    let exponent = builder
        .constant_from_bytes(
            &descriptor(vec![1, 1, 256]),
            (0..256)
                .map(|i| i as f32 / 256.)
                .flat_map(f32::to_le_bytes)
                .collect(),
        )
        .unwrap();
    let base = builder
        .constant_from_bytes(&descriptor(vec![]), 10000f32.to_le_bytes().to_vec())
        .unwrap();
    let divisor = builder.pow(base, exponent).unwrap();
    let angles = builder.div(position, divisor).unwrap();
    let cosine = builder.cos(angles).unwrap();
    let sine = builder.sin(angles).unwrap();
    let joined = builder.concat(&[cosine, sine], 2).unwrap();
    let embedding = builder
        .transpose_with_options(
            joined,
            MLTransposeOptions {
                permutation: vec![1, 0, 2],
                ..Default::default()
            },
        )
        .unwrap();
    let input = builder.input("x", &descriptor(vec![1])).unwrap();
    let result = builder.add(input, embedding).unwrap();
    let mut outputs = MLNamedOperands::new();
    outputs.insert("result", result);
    let graph = builder.finish_graph_info(&outputs).unwrap();
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    // This chain previously faults inside BNNS during CPU model loading on macOS 26.6.2.
    for attempt in run_coreml_with_inputs_with_weights(
        &converted.data,
        converted.weights_data.as_deref(),
        vec![CoremlInput {
            name: "x".into(),
            shape: vec![1],
            data: vec![2.],
        }],
    )
    .unwrap()
    {
        let actual = attempt.result.unwrap();
        assert_eq!(actual[0].data.len(), 1344 * 512);
        for (row, &position) in positions.iter().enumerate() {
            for (column, &divisor) in divisors.iter().enumerate() {
                let angle = position / divisor;
                for (offset, expected) in
                    [(column, 2. + angle.cos()), (256 + column, 2. + angle.sin())]
                {
                    assert!(
                        (actual[0].data[row * 512 + offset] - expected).abs() < 3e-5,
                        "{} [{row},{offset}]",
                        attempt.compute_unit
                    );
                }
            }
        }
    }
}

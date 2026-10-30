//! Constant scalar powers retain the boundary rank expected by CoreML.
#![cfg(feature = "coreml-runtime")]
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
use rustnn::mlcontext::{MLNamedOperands, MLOperandDescriptor};
use rustnn::mlgraphbuilder::MLGraphBuilder;
use rustnn::operator_enums::MLOperandDataType;
#[test]
fn scalar_constant_power_as_output_and_intermediate() {
    for intermediate in [false, true] {
        let mut b = MLGraphBuilder::new_uncompiled();
        let d = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![]);
        let base = b
            .constant_from_bytes(&d, 64f32.to_le_bytes().to_vec())
            .unwrap();
        let exponent = b
            .constant_from_bytes(&d, (-0.5f32).to_le_bytes().to_vec())
            .unwrap();
        let power = b.pow(base, exponent).unwrap();
        let (output, inputs, expected) = if intermediate {
            let x = b
                .input(
                    "x",
                    &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2]),
                )
                .unwrap();
            (
                b.mul(x, power).unwrap(),
                vec![CoremlInput {
                    name: "x".into(),
                    shape: vec![2],
                    data: vec![2., -4.],
                }],
                vec![0.25, -0.5],
            )
        } else {
            (power, vec![], vec![0.125])
        };
        let mut outputs = MLNamedOperands::new();
        outputs.insert("result", output);
        let graph = b.finish_graph_info(&outputs).unwrap();
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        for attempt in run_coreml_with_inputs_with_weights(
            &converted.data,
            converted.weights_data.as_deref(),
            inputs,
        )
        .unwrap()
        {
            let actual = attempt.result.unwrap();
            assert_eq!(actual[0].data, expected, "{}", attempt.compute_unit);
        }
    }
}

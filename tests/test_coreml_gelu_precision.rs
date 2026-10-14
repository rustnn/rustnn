//! Precision and shape boundaries for the CoreML exact-GELU lowering.

#[path = "common/half_reference.rs"]
mod half_reference;

use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind};
use rustnn::operators::Operation;
use rustnn::protos::coreml::{mil_spec, specification};

fn graph(data_type: DataType, shape: Vec<Dimension>) -> GraphInfo {
    let descriptor = OperandDescriptor {
        data_type,
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
        operations: vec![Operation::Gelu {
            input: 0,
            options: None,
            outputs: vec![1],
        }],
        ..Default::default()
    }
}

fn block(graph: &GraphInfo) -> mil_spec::Block {
    let converted = CoremlMlProgramConverter.convert(graph).unwrap();
    let model = specification::Model::decode(converted.data.as_slice()).unwrap();
    // Precision promotion must not raise the existing platform requirement.
    assert_eq!(model.specification_version, 9);
    let specification::model::Type::MlProgram(program) = model.r#type.unwrap() else {
        panic!("expected MLProgram");
    };
    program.functions["main"].block_specializations["CoreML7"].clone()
}

fn tensor_type(value: &mil_spec::NamedValueType) -> &mil_spec::TensorType {
    let Some(mil_spec::value_type::Type::TensorType(tensor)) =
        value.r#type.as_ref().unwrap().r#type.as_ref()
    else {
        panic!("expected tensor");
    };
    tensor
}

#[test]
fn gelu_float32_keeps_native_evaluation_without_boundary_casts() {
    let block = block(&graph(DataType::Float32, vec![Dimension::Static(11)]));
    assert_eq!(block.operations.len(), 1);
    assert_eq!(block.operations[0].r#type, "gelu");
    assert_eq!(
        tensor_type(&block.operations[0].outputs[0]).data_type,
        mil_spec::DataType::Float32 as i32
    );
    assert!(block.operations[0].inputs.contains_key("mode"));
    assert_eq!(block.outputs, ["result"]);
}

#[test]
fn gelu_float16_preserves_scalar_and_zero_dimension_conversion() {
    // A public scalar is represented as [1] at the CoreML boundary. A genuine
    // empty dimension must remain zero, not be mistaken for that scalar case.
    // This test checks conversion only, not runtime empty-tensor support.
    for (shape, expected_size) in [(vec![], 1), (vec![Dimension::Static(0)], 0)] {
        let block = block(&graph(DataType::Float16, shape));
        assert_eq!(block.operations.len(), 3);
        for operation in &block.operations {
            let tensor = tensor_type(&operation.outputs[0]);
            assert_eq!(tensor.rank, 1);
            assert_eq!(tensor.dimensions.len(), 1);
            let Some(mil_spec::dimension::Dimension::Constant(dimension)) =
                &tensor.dimensions[0].dimension
            else {
                panic!("expected constant dimension");
            };
            assert_eq!(dimension.size, expected_size);
        }
    }
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn gelu_float16_preserves_dynamic_dimensions_through_precision_casts() {
    let shape = vec![Dimension::Dynamic(rustnn::graph::DynamicDimension {
        name: "batch".into(),
        max_size: 11,
    })];
    let block = block(&graph(DataType::Float16, shape));
    assert_eq!(block.operations.len(), 3);
    for operation in &block.operations {
        let tensor = tensor_type(&operation.outputs[0]);
        assert_eq!(tensor.rank, 1);
        assert_eq!(tensor.dimensions.len(), 1);
        assert!(matches!(
            tensor.dimensions[0].dimension,
            Some(mil_spec::dimension::Dimension::Unknown(_))
        ));
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
mod runtime {
    use super::*;
    use half::f16;
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    use rustnn::mlcontext::{
        MLContext, MLContextOptions, MLGraph, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType;

    // libSystem's scalar double-precision erf is independent of CoreML's
    // implementation and needs no runtime dependency or network fixture.
    unsafe extern "C" {
        fn erf(value: f64) -> f64;
    }

    fn expected_bits(input: u16) -> u16 {
        let value = f16::from_bits(input).to_f64();
        // SAFETY: erf accepts every f64, including infinities and NaNs.
        let erf_value = unsafe { erf(value / std::f64::consts::SQRT_2) };
        half_reference::reference_half_bits(0.5 * value * (1.0 + erf_value))
    }

    fn check_output(input: u16, actual: u16, expected: u16, policy: DeviceType) {
        let actual_value = f16::from_bits(actual);
        let expected_value = f16::from_bits(expected);
        if expected_value.is_nan() {
            assert!(actual_value.is_nan(), "{policy:?}, input={input:#06x}");
            return;
        }
        if expected_value.is_infinite() {
            assert_eq!(actual, expected, "{policy:?}, input={input:#06x}");
            return;
        }
        assert!(
            actual_value.is_finite(),
            "{policy:?}, input={input:#06x}, nonfinite actual={actual:#06x}"
        );
        // Match WPT's finite FP16 bit-distance metric, treating signed zeros
        // as equal. The double-precision mathematical oracle is independent.
        let ulp = if actual_value == f16::ZERO && expected_value == f16::ZERO {
            0
        } else {
            actual.abs_diff(expected)
        };
        assert!(
            ulp <= 18,
            "{policy:?}, input={input:#06x}, actual={actual:#06x}, expected={expected:#06x}, ulp={ulp}"
        );
    }

    fn context(policy: DeviceType) -> MLContext<'static> {
        MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, policy != DeviceType::Cpu)
                .with_rustnn_device_hint(BackendDevice::Coreml {
                    device_type: policy,
                }),
        )
        .unwrap()
    }

    fn predict(
        context: &mut MLContext<'_>,
        graph: &mut MLGraph<'_>,
        shape: &[u64],
        values: &[u16],
    ) -> Vec<u16> {
        let descriptor = MLTensorDescriptor::new(MLOperandDataType::Float16, shape.to_vec());
        let input = context
            .create_tensor(&descriptor.clone().to_writable())
            .unwrap();
        let output = context.create_tensor(&descriptor.to_readable()).unwrap();
        context.write_tensor(&input, values).unwrap();
        context
            .dispatch(
                graph,
                &MLNamedTensors::from([("input", &input)]),
                &MLNamedTensors::from([("result", &output)]),
            )
            .unwrap();
        let mut result = vec![0u16; values.len()];
        context.read_tensor(&output, &mut result).unwrap();
        result
    }

    #[test]
    fn gelu_float16_all_bit_patterns_match_double_oracle_under_requested_policies() {
        let values: Vec<u16> = (0..=u16::MAX).collect();
        let expected: Vec<u16> = values.iter().copied().map(expected_bits).collect();
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            let mut context = context(policy);
            let mut graph = context
                .rustnn_build_graph(graph(DataType::Float16, vec![Dimension::Static(65536)]))
                .unwrap();
            let actual = predict(&mut context, &mut graph, &[65536], &values);
            for ((&input, actual), &expected) in values.iter().zip(actual).zip(&expected) {
                check_output(input, actual, expected, policy);
            }
        }
    }

    #[test]
    fn gelu_float16_scalar_matches_double_oracle() {
        let input = f16::from_f32(-2.707_031_2).to_bits();
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            let mut context = context(policy);
            let mut graph = context
                .rustnn_build_graph(graph(DataType::Float16, vec![]))
                .unwrap();
            let actual = predict(&mut context, &mut graph, &[], &[input]);
            check_output(input, actual[0], expected_bits(input), policy);
        }
    }

    #[cfg(feature = "dynamic-inputs")]
    #[test]
    fn gelu_float16_reuses_graph_across_growing_and_shrinking_shapes() {
        let values: Vec<_> = (-5..=5)
            .map(|value| f16::from_f32(value as f32).to_bits())
            .collect();
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            let mut context = context(policy);
            let shape = vec![Dimension::Dynamic(rustnn::graph::DynamicDimension {
                name: "batch".into(),
                max_size: 11,
            })];
            let mut graph = context
                .rustnn_build_graph(graph(DataType::Float16, shape))
                .unwrap();
            for length in [1usize, 11, 3] {
                let input = &values[..length];
                let actual = predict(&mut context, &mut graph, &[length as u64], input);
                for (&input, actual) in input.iter().zip(actual) {
                    check_output(input, actual, expected_bits(input), policy);
                }
            }
        }
    }
}

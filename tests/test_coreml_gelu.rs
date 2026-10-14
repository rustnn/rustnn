//! FP16 GELU must retain WebNN's exact formula under every requested policy.

use rustnn::graph::{DataType, GraphInfo, Operand, OperandDescriptor, OperandKind};
use rustnn::operators::Operation;

fn graph() -> GraphInfo {
    let descriptor = OperandDescriptor {
        data_type: DataType::Float16,
        shape: vec![rustnn::graph::Dimension::Static(11)],
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

fn convolution_graph(gelu_filter: bool) -> GraphInfo {
    use rustnn::graph::{ConstantData, to_dimension_vector};
    use rustnn::operator_enums::{MLConv2dFilterOperandLayout, MLInputOperandLayout};
    use rustnn::operator_options::MLConv2dOptions;

    let (input_shape, constant_shape, output_shape, values) = if gelu_filter {
        (
            vec![1, 1, 2, 1],
            vec![1, 2, 2, 3],
            vec![1, 1, 2, 3],
            (1..=12).map(|v| v as f32).collect::<Vec<_>>(),
        )
    } else {
        (
            vec![1, 2, 3, 2],
            vec![1, 2, 1, 1],
            vec![1, 2, 3, 1],
            vec![1., 2.],
        )
    };
    let operand = |name: &str, kind, shape: &[u32]| Operand {
        kind,
        name: Some(name.into()),
        descriptor: OperandDescriptor {
            data_type: DataType::Float16,
            shape: to_dimension_vector(shape),
            pending_permutation: vec![],
        },
    };
    GraphInfo {
        operands: vec![
            operand("input", OperandKind::Input, &input_shape),
            operand("activated", OperandKind::Intermediate, &input_shape),
            operand("weights", OperandKind::Constant, &constant_shape),
            operand("result", OperandKind::Output, &output_shape),
        ],
        input_operands: vec![0],
        output_operands: vec![3],
        constant_operand_ids_to_handles: [(
            2,
            ConstantData {
                data: values
                    .iter()
                    .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
                    .collect(),
                label: None,
            },
        )]
        .into_iter()
        .collect(),
        operations: vec![
            Operation::Gelu {
                input: 0,
                options: None,
                outputs: vec![1],
            },
            Operation::Conv2d {
                input: if gelu_filter { 2 } else { 1 },
                filter: if gelu_filter { 1 } else { 2 },
                options: Some(MLConv2dOptions {
                    input_layout: if gelu_filter {
                        MLInputOperandLayout::Nchw
                    } else {
                        MLInputOperandLayout::Nhwc
                    },
                    filter_layout: if gelu_filter {
                        MLConv2dFilterOperandLayout::Hwio
                    } else {
                        MLConv2dFilterOperandLayout::Oihw
                    },
                    ..Default::default()
                }),
                outputs: vec![3],
            },
        ],
        ..Default::default()
    }
}

#[test]
fn gelu_float16_emits_deferred_activation_and_filter_transposes() {
    use prost::Message;
    use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
    use rustnn::protos::coreml::{mil_spec::argument::binding::Binding, specification};
    for gelu_filter in [false, true] {
        let converted = CoremlMlProgramConverter
            .convert(&convolution_graph(gelu_filter))
            .unwrap();
        let model = specification::Model::decode(converted.data.as_slice()).unwrap();
        let programs = match model.r#type.unwrap() {
            specification::model::Type::MlProgram(program) => vec![program],
            specification::model::Type::Pipeline(pipeline) => pipeline
                .models
                .into_iter()
                .map(|model| {
                    let Some(specification::model::Type::MlProgram(program)) = model.r#type else {
                        panic!("MLProgram stage")
                    };
                    program
                })
                .collect(),
            _ => panic!("MLProgram or precision Pipeline"),
        };
        let operations: Vec<_> = programs
            .iter()
            .flat_map(|program| {
                &program.functions["main"].block_specializations["CoreML7"].operations
            })
            .collect();
        let (conv_index, conv) = operations
            .iter()
            .enumerate()
            .find(|(_, op)| op.r#type == "conv")
            .unwrap();
        let argument = if gelu_filter { "weight" } else { "x" };
        let Some(Binding::Name(input_name)) = &conv.inputs[argument].arguments[0].binding else {
            panic!("expected convolution value binding");
        };
        let source = |name: &str| {
            operations
                .iter()
                .find(|operation| operation.outputs.iter().any(|output| output.name == name))
                .copied()
        };
        let uncast = |mut name: String| {
            while name != "activated" {
                let Some(operation) = source(&name)
                    .filter(|operation| matches!(operation.r#type.as_str(), "cast" | "reshape"))
                else {
                    break;
                };
                let Some(Binding::Name(input)) = &operation.inputs["x"].arguments[0].binding else {
                    break;
                };
                name = input.clone();
            }
            name
        };
        let transpose_name = uncast(input_name.clone());
        let transpose = operations[..conv_index]
            .iter()
            .find(|op| op.r#type == "transpose" && op.outputs[0].name == transpose_name)
            .expect("GELU producer must emit the deferred transpose before convolution");
        let Some(Binding::Name(input)) = &transpose.inputs["x"].arguments[0].binding else {
            panic!("transpose input")
        };
        assert_eq!(uncast(input.clone()), "activated");
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn gelu_float16_convolution_compositions_predict_correct_values() {
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    use rustnn::mlcontext::{
        MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType;

    for gelu_filter in [false, true] {
        // GELU rounds these positive integers back to themselves in FP16.
        let values: Vec<u16> = (6..if gelu_filter { 8 } else { 18 })
            .map(|v| half::f16::from_f32(v as f32).to_bits())
            .collect();
        let expected = if gelu_filter {
            [55., 68., 81., 94., 107., 120.]
        } else {
            [20., 26., 32., 38., 44., 50.]
        };
        for device_type in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            let mut context = MLContext::create(
                &MLContextOptions::new(MLPowerPreference::Default, device_type != DeviceType::Cpu)
                    .with_rustnn_device_hint(BackendDevice::Coreml { device_type }),
            )
            .unwrap();
            let source = convolution_graph(gelu_filter);
            let shape = |id: usize| {
                source.operands[id]
                    .descriptor
                    .shape
                    .iter()
                    .map(|d| u64::from(d.get_static_or_max_size()))
                    .collect()
            };
            let input = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Float16, shape(0)).to_writable(),
                )
                .unwrap();
            let output = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Float16, shape(3)).to_readable(),
                )
                .unwrap();
            let mut graph = context.rustnn_build_graph(source).unwrap();
            context.write_tensor(&input, &values).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("input", &input)]),
                    &MLNamedTensors::from([("result", &output)]),
                )
                .unwrap();
            let mut actual = [0u16; 6];
            context.read_tensor(&output, &mut actual).unwrap();
            assert_eq!(
                actual.map(|v| half::f16::from_bits(v).to_f32()),
                expected,
                "filter={gelu_filter}, {device_type:?}"
            );
        }
    }
}

#[test]
fn gelu_float16_lowering_preserves_half_boundaries_and_float32_evaluation() {
    use prost::Message;
    use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
    use rustnn::protos::coreml::{mil_spec, specification};

    let converted = CoremlMlProgramConverter.convert(&graph()).unwrap();
    let model = specification::Model::decode(converted.data.as_slice()).unwrap();
    let specification::model::Type::MlProgram(program) = model.r#type.unwrap() else {
        panic!("expected MLProgram");
    };
    let block = &program.functions["main"].block_specializations["CoreML7"];
    assert_eq!(
        block
            .operations
            .iter()
            .map(|op| op.r#type.as_str())
            .collect::<Vec<_>>(),
        ["cast", "gelu", "cast"]
    );
    let types: Vec<i32> = block
        .operations
        .iter()
        .map(|op| {
            let Some(mil_spec::value_type::Type::TensorType(tensor)) =
                op.outputs[0].r#type.as_ref().unwrap().r#type.as_ref()
            else {
                panic!("expected tensor");
            };
            tensor.data_type
        })
        .collect();
    assert_eq!(
        types,
        [
            mil_spec::DataType::Float32 as i32,
            mil_spec::DataType::Float32 as i32,
            mil_spec::DataType::Float16 as i32
        ]
    );
    assert_eq!(block.outputs, ["result"]);
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn gelu_float16_matches_wpt_tolerance_under_all_requested_policies() {
    use half::f16;
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    use rustnn::mlcontext::{
        MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType;

    // Independent f64 erf oracle, rounded to binary16 after evaluation. Include
    // the negative tail and zero, not only values in [-1, 1].
    let inputs = [
        -5.0f32,
        -3.0,
        -2.707_031_2,
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        2.707_031_2,
        3.0,
        5.0,
    ];
    let expected = [
        32792u16, 39974, 41140, 43475, 45332, 0, 15035, 16337, 16741, 16894, 17664,
    ];
    let input_bits: Vec<u16> = inputs.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    for device_type in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
        let options =
            MLContextOptions::new(MLPowerPreference::Default, device_type != DeviceType::Cpu)
                .with_rustnn_device_hint(BackendDevice::Coreml { device_type });
        let mut context = MLContext::create(&options).unwrap();
        let mut graph = context.rustnn_build_graph(graph()).unwrap();
        let input = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Float16, vec![11]).to_writable(),
            )
            .unwrap();
        let output = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Float16, vec![11]).to_readable(),
            )
            .unwrap();
        context.write_tensor(&input, &input_bits).unwrap();
        context
            .dispatch(
                &mut graph,
                &MLNamedTensors::from([("input", &input)]),
                &MLNamedTensors::from([("result", &output)]),
            )
            .unwrap();
        let mut actual = [0u16; 11];
        context.read_tensor(&output, &mut actual).unwrap();
        for (index, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            let ulp = if f16::from_bits(a) == f16::ZERO && f16::from_bits(e) == f16::ZERO {
                0
            } else {
                a.abs_diff(e)
            };
            assert!(
                ulp <= 18,
                "{device_type:?}, input={}, actual={}, expected={}, ulp={ulp}",
                inputs[index],
                f16::from_bits(a).to_f32(),
                f16::from_bits(e).to_f32()
            );
        }
    }
}

#[test]
fn gelu_float16_temporaries_do_not_shadow_graph_names() {
    use prost::Message;
    use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
    use rustnn::protos::coreml::specification;

    let mut graph = graph();
    graph.operands[0].name = Some("result_gelu_input_fp32_1".into());
    for name in ["result_gelu_input_fp32_1_1", "result_gelu_result_fp32_1"] {
        let mut operand = graph.operands[0].clone();
        operand.name = Some(name.into());
        graph.operands.push(operand);
    }
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let model = specification::Model::decode(converted.data.as_slice()).unwrap();
    let specification::model::Type::MlProgram(program) = model.r#type.unwrap() else {
        panic!("expected MLProgram");
    };
    let block = &program.functions["main"].block_specializations["CoreML7"];
    assert_eq!(
        block.operations[0].outputs[0].name,
        "result_gelu_input_fp32_1_2"
    );
    assert_eq!(
        block.operations[1].outputs[0].name,
        "result_gelu_result_fp32_1_1"
    );
    assert_eq!(block.outputs, ["result"]);
}

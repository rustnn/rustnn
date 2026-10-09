//! Explicit narrowing casts remain observable after CoreML graph optimization.

use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{
    ConstantData, DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::{MLArgMinMaxOptions, MLGatherOptions, MLPadOptions};
use rustnn::operators::Operation;
use rustnn::protos::coreml::specification::{self, Model, model};

#[path = "common/half_reference.rs"]
mod half_reference;

fn reference_half(value: f32) -> half::f16 {
    half::f16::from_bits(half_reference::reference_half_bits(f64::from(value)))
}

fn operand(name: &str, kind: OperandKind, dtype: DataType, shape: &[Dimension]) -> Operand {
    Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: dtype,
            shape: shape.to_vec(),
            pending_permutation: vec![],
        },
    }
}

fn cast(input: u32, output: u32, dtype: MLOperandDataType) -> Operation {
    Operation::Cast {
        input,
        data_type: dtype,
        options: None,
        outputs: vec![output],
    }
}

fn pair(shape: &[Dimension]) -> GraphInfo {
    GraphInfo {
        operands: vec![
            operand("input", OperandKind::Input, DataType::Float32, shape),
            operand(
                "rounded",
                OperandKind::Intermediate,
                DataType::Float16,
                shape,
            ),
            operand("result", OperandKind::Output, DataType::Float32, shape),
        ],
        input_operands: vec![0],
        output_operands: vec![2],
        operations: vec![
            cast(0, 1, MLOperandDataType::Float16),
            cast(1, 2, MLOperandDataType::Float32),
        ],
        ..Default::default()
    }
}

fn direct_half_widening(shape: &[Dimension]) -> GraphInfo {
    GraphInfo {
        operands: vec![
            operand("input", OperandKind::Input, DataType::Float16, shape),
            operand("result", OperandKind::Output, DataType::Float32, shape),
        ],
        input_operands: vec![0],
        output_operands: vec![1],
        operations: vec![cast(0, 1, MLOperandDataType::Float32)],
        ..Default::default()
    }
}

fn escaped_pair() -> GraphInfo {
    let mut graph = pair(&[Dimension::Static(4)]);
    graph.operands[0].name = Some("input with spaces".into());
    graph.operands[1].name = Some("rounded value".into());
    graph.operands[2].name = Some("result with spaces".into());
    graph
}

#[test]
fn precision_pipeline_keeps_logical_binding_metadata_only_on_the_top_level() {
    let graph = escaped_pair();
    let original = serde_json::to_value(&graph).unwrap();
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    assert_eq!(serde_json::to_value(&graph).unwrap(), original);
    let model = Model::decode(converted.data.as_slice()).unwrap();
    let description = model.description.as_ref().unwrap();
    let metadata = &description.metadata.as_ref().unwrap().user_defined;
    let input: serde_json::Value =
        serde_json::from_str(&metadata["rustnn.webnn.input_aliases"]).unwrap();
    let output: serde_json::Value =
        serde_json::from_str(&metadata["rustnn.webnn.output_aliases"]).unwrap();
    assert_eq!(
        description.input[0].name,
        input["input with spaces"].as_str().unwrap()
    );
    assert_eq!(
        description.output[0].name,
        output["result with spaces"].as_str().unwrap()
    );
    let Some(model::Type::Pipeline(pipeline)) = &model.r#type else {
        panic!("pipeline");
    };
    for child in &pipeline.models {
        assert!(child.description.as_ref().unwrap().metadata.is_none());
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn precision_pipeline_binds_escaped_logical_names_without_losing_rounding() {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
    let input = vec![1.0003f32, 1.0007, -0.0, -2f32.powi(-25)];
    let converted = CoremlMlProgramConverter.convert(&escaped_pair()).unwrap();
    let attempts = run_coreml_with_inputs_with_weights(
        &converted.data,
        converted.weights_data.as_deref(),
        vec![CoremlInput {
            name: "input with spaces".into(),
            shape: vec![input.len()],
            data: input.clone(),
        }],
    )
    .unwrap();
    for attempt in attempts {
        let outputs = attempt.result.unwrap();
        let result = outputs
            .iter()
            .find(|output| output.name == "result with spaces")
            .unwrap();
        for (&actual, &source) in result.data.iter().zip(&input) {
            assert_eq!(
                actual.to_bits(),
                reference_half(source).to_f32_const().to_bits(),
                "{}",
                attempt.compute_unit
            );
        }
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn constant_transpose_uses_asset_without_policy_fallback() {
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    use rustnn::executors::coreml::CoremlLoadRoute;
    use rustnn::mlcontext::{
        LoadDiagnostics, MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference,
        MLTensorDescriptor, RustNNOptions,
    };

    // Upstream WPT transpose_float32_1D_constant_tensor_default_options.
    // Its data-free URL model crashes BNNS compilation under CPU+ANE on the
    // affected stack. Keep the exact bits and independently verify route,
    // requested policy and output bytes in all tensor-storage modes.
    let values = [
        0xc236b29e_u32,
        0x4255d645,
        0xc2707956,
        0x421853b6,
        0x429d48f2,
        0xc28a81a9,
        0x3febf673,
        0x42b99edd,
        0x4260667a,
        0x429a1de4,
        0x4265df50,
        0xc2a97c76,
        0x42398aa4,
        0xc2a9cb98,
        0x4262d14b,
        0xc1cd8fa8,
        0x40b3e8d9,
        0xc1cd4d72,
        0x42c6ecfa,
        0xc2af2dac,
        0xc282c17d,
        0xc2840512,
        0x4219de08,
        0x400ccbca,
    ];
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let shape = [Dimension::Static(values.len() as u32)];
    let source = GraphInfo {
        operands: vec![
            operand("constant", OperandKind::Constant, DataType::Float32, &shape),
            operand(
                "transposeOutput",
                OperandKind::Output,
                DataType::Float32,
                &shape,
            ),
        ],
        output_operands: vec![1],
        operations: vec![Operation::Transpose {
            input: 0,
            options: None,
            outputs: vec![1],
        }],
        constant_operand_ids_to_handles: [(
            0,
            ConstantData {
                data: bytes.clone(),
                label: None,
            },
        )]
        .into(),
        ..Default::default()
    };
    for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
        for (reuse, backings) in [(false, false), (true, false), (true, true)] {
            let mut options = RustNNOptions::default();
            options.coreml.reuse_tensor_storage = reuse;
            options.coreml.output_backings = backings;
            let mut context = MLContext::create(
                &MLContextOptions::new(MLPowerPreference::Default, policy != DeviceType::Cpu)
                    .with_rustnn_device_hint(BackendDevice::Coreml {
                        device_type: policy,
                    })
                    .with_rustnn_options(options),
            )
            .unwrap();
            let mut graph = context.rustnn_build_graph(source.clone()).unwrap();
            let Some(LoadDiagnostics::Coreml(load)) = graph.rustnn_load_diagnostics() else {
                panic!("missing CoreML load diagnostics");
            };
            assert_eq!(load.route, CoremlLoadRoute::InMemoryAsset);
            assert_eq!(load.requested_compute_units, load.loaded_compute_units);
            assert!(load.failures.is_empty(), "{load:?}");
            let output = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![24]).to_readable(),
                )
                .unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::new(),
                    &MLNamedTensors::from([("transposeOutput", &output)]),
                )
                .unwrap();
            let mut actual = vec![0u8; bytes.len()];
            context.read_tensor(&output, &mut actual).unwrap();
            assert_eq!(
                actual, bytes,
                "{policy:?}, reuse={reuse}, backings={backings}"
            );
        }
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn precision_pipeline_preserves_typed_boundaries_with_retained_tensor_storage() {
    // Bitwise preservation of signed zero and represented subnormals is a
    // recurrent-state fidelity check beyond WPT's numeric comparator. Keep it
    // local rather than inventing a composed-graph WebNN tolerance.
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    use rustnn::mlcontext::{
        BackendStatistics, MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference,
        MLTensorDescriptor, RustNNOptions,
    };

    for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
        for (reuse, backings) in [(false, false), (true, false), (true, true)] {
            for half_input in [false, true] {
                for shape in [
                    vec![Dimension::Static(4)],
                    #[cfg(feature = "dynamic-inputs")]
                    vec![Dimension::Dynamic(rustnn::graph::DynamicDimension {
                        name: "length".into(),
                        max_size: 4,
                    })],
                ] {
                    let dynamic = shape.iter().any(|d| !matches!(d, Dimension::Static(_)));
                    let mut options = RustNNOptions::default();
                    options.coreml.reuse_tensor_storage = reuse;
                    options.coreml.output_backings = backings;
                    let mut context = MLContext::create(
                        &MLContextOptions::new(
                            MLPowerPreference::Default,
                            policy != DeviceType::Cpu,
                        )
                        .with_rustnn_device_hint(BackendDevice::Coreml {
                            device_type: policy,
                        })
                        .with_rustnn_options(options),
                    )
                    .unwrap();
                    let mut source = if half_input {
                        direct_half_widening(&shape)
                    } else {
                        pair(&shape)
                    };
                    source.operands[0].name = Some("state input".into());
                    let last = *source.output_operands.first().unwrap() as usize;
                    source.operands[last].name = Some("state result".into());
                    let converted = CoremlMlProgramConverter.convert(&source).unwrap();
                    let is_pipeline = matches!(
                        Model::decode(converted.data.as_slice()).unwrap().r#type,
                        Some(model::Type::Pipeline(_))
                    );
                    if !half_input {
                        assert!(
                            is_pipeline,
                            "narrow/widen must exercise the native Pipeline guard"
                        );
                    }
                    let mut graph = context.rustnn_build_graph(source).unwrap();
                    let Some(rustnn::mlcontext::LoadDiagnostics::Coreml(load)) =
                        graph.rustnn_load_diagnostics()
                    else {
                        panic!("missing CoreML load diagnostics");
                    };
                    assert_eq!(load.requested_compute_units, load.loaded_compute_units);
                    assert_eq!(
                        load.route,
                        rustnn::executors::coreml::CoremlLoadRoute::CompiledUrl
                    );
                    assert!(load.failures.is_empty(), "{load:?}");
                    let input_type = if half_input {
                        MLOperandDataType::Float16
                    } else {
                        MLOperandDataType::Float32
                    };
                    let mut input = context
                        .create_tensor(
                            &MLTensorDescriptor::new(input_type, vec![4])
                                .to_readable()
                                .to_writable(),
                        )
                        .unwrap();
                    let mut output = context
                        .create_tensor(
                            &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4])
                                .to_readable(),
                        )
                        .unwrap();
                    // Reuse the same model and storage with changing contents, including
                    // values that vanish if the Half boundary or compact view is lost.
                    for (iteration, reverse) in [false, true, false].into_iter().enumerate() {
                        let count = if dynamic {
                            [1usize, 4, 2][iteration]
                        } else {
                            4
                        };
                        context
                            .rustnn_resize_tensor(&mut input, &[count as u64])
                            .unwrap();
                        context
                            .rustnn_resize_tensor(&mut output, &[count as u64])
                            .unwrap();
                        let mut half_bits = [0x8000u16, 1, 0x8001, 0x7bff];
                        let mut values = if half_input {
                            half_bits.map(|b| half::f16::from_bits(b).to_f32_const())
                        } else {
                            [1.0003, 1.0007, -0.0, -2f32.powi(-25)]
                        };
                        if reverse {
                            values.reverse();
                            half_bits.reverse();
                        }
                        let values = &values[..count];
                        let bytes: Vec<u8> = if half_input {
                            half_bits[..count]
                                .iter()
                                .flat_map(|b| b.to_le_bytes())
                                .collect()
                        } else {
                            values.iter().flat_map(|v| v.to_le_bytes()).collect()
                        };
                        context.write_tensor(&input, &bytes).unwrap();
                        context
                            .dispatch(
                                &mut graph,
                                &MLNamedTensors::from([("state input", &input)]),
                                &MLNamedTensors::from([("state result", &output)]),
                            )
                            .unwrap_or_else(|error| {
                                panic!("{policy:?}, reuse={reuse}, backings={backings}, half_input={half_input}, dynamic={dynamic}, iteration={iteration}: {error}")
                            });
                        let mut result = vec![0f32; count];
                        context.read_tensor(&output, &mut result).unwrap();
                        for (&actual, &source) in result.iter().zip(values) {
                            assert_eq!(
                                actual.to_bits(),
                                reference_half(source).to_f32_const().to_bits(),
                                "{policy:?}, reuse={reuse}, backings={backings}, half_input={half_input}"
                            );
                        }
                        let mut unchanged = vec![0u8; bytes.len()];
                        context.read_tensor(&input, &mut unchanged).unwrap();
                        assert_eq!(unchanged, bytes, "native output must not alias its input");
                    }
                    let Some(BackendStatistics::Coreml(statistics)) =
                        context.rustnn_backend_statistics()
                    else {
                        panic!("missing CoreML storage statistics");
                    };
                    if reuse {
                        assert_eq!(statistics.native_input_bindings, 3);
                        assert_eq!(statistics.input_copy_bytes, 0);
                    } else {
                        assert_eq!(statistics.native_input_bindings, 0);
                        assert_eq!(
                            statistics.input_copy_bytes,
                            (if half_input { 2 } else { 4 }) * (if dynamic { 7 } else { 12 })
                        );
                    }
                    if is_pipeline {
                        assert_eq!(statistics.output_backings_requested, 0);
                        assert_eq!(statistics.output_backings_accepted, 0);
                        assert_eq!(
                            statistics.output_copy_bytes,
                            4 * (if dynamic { 7 } else { 12 })
                        );
                    }
                }
            }
        }
    }
}

fn half_unary(operation: &str, length: u32) -> GraphInfo {
    let shape = [Dimension::Static(length)];
    GraphInfo {
        operands: vec![
            operand("input", OperandKind::Input, DataType::Float16, &shape),
            operand("result", OperandKind::Output, DataType::Float16, &shape),
        ],
        input_operands: vec![0],
        output_operands: vec![1],
        operations: vec![match operation {
            "reciprocal" => Operation::Reciprocal {
                input: 0,
                options: None,
                outputs: vec![1],
            },
            "roundEven" => Operation::RoundEven {
                input: 0,
                options: None,
                outputs: vec![1],
            },
            "neg" => Operation::Neg {
                input: 0,
                options: None,
                outputs: vec![1],
            },
            _ => unreachable!(),
        }],
        ..Default::default()
    }
}

fn half_subtraction(shape: &[Dimension], class: Option<&str>) -> GraphInfo {
    let mut graph = GraphInfo {
        operands: vec![
            operand("input", OperandKind::Input, DataType::Float16, shape),
            operand("difference", OperandKind::Output, DataType::Float16, shape),
        ],
        input_operands: vec![0],
        output_operands: vec![1],
        operations: vec![Operation::Sub {
            a: 0,
            b: 0,
            options: None,
            outputs: vec![1],
        }],
        ..Default::default()
    };
    if let Some(class) = class {
        graph.operands.push(operand(
            "classified",
            OperandKind::Output,
            DataType::Uint8,
            shape,
        ));
        graph.operations.push(match class {
            "isNaN" => Operation::IsNaN {
                input: 1,
                options: None,
                outputs: vec![2],
            },
            "isInfinite" => Operation::IsInfinite {
                input: 1,
                options: None,
                outputs: vec![2],
            },
            _ => unreachable!(),
        });
        graph.output_operands.push(2);
    }
    graph
}

#[test]
fn precision_pipeline_materializes_half_subtraction_and_classification() {
    use rustnn::protos::coreml::mil_spec::{self, argument, value_type};
    for shape in [
        vec![],
        vec![Dimension::Static(2), Dimension::Static(3)],
        #[cfg(feature = "dynamic-inputs")]
        vec![Dimension::Dynamic(rustnn::graph::DynamicDimension {
            name: "length".into(),
            max_size: 11,
        })],
    ] {
        for class in [None, Some("isNaN"), Some("isInfinite")] {
            let stages = pipeline(&half_subtraction(&shape, class));
            assert_eq!(stages.len(), if class.is_some() { 5 } else { 3 });
            let mut kernels = 0;
            for stage in &stages {
                let Some(model::Type::MlProgram(program)) = &stage.r#type else {
                    panic!("program")
                };
                let function = &program.functions["main"];
                let block = &function.block_specializations[&function.opset];
                let mut types: std::collections::HashMap<_, _> = function
                    .inputs
                    .iter()
                    .map(|input| (input.name.clone(), input.r#type.clone()))
                    .collect();
                for operation in &block.operations {
                    if matches!(operation.r#type.as_str(), "sub" | "equal" | "abs") {
                        kernels += 1;
                        let Some(argument::binding::Binding::Name(name)) =
                            &operation.inputs["x"].arguments[0].binding
                        else {
                            panic!("named input")
                        };
                        let Some(value_type::Type::TensorType(tensor)) =
                            &types[name].as_ref().unwrap().r#type
                        else {
                            panic!("tensor")
                        };
                        assert_eq!(tensor.data_type, mil_spec::DataType::Float32 as i32);
                    }
                    for output in &operation.outputs {
                        types.insert(output.name.clone(), output.r#type.clone());
                    }
                }
            }
            assert_eq!(kernels, if class.is_some() { 2 } else { 1 });
        }
    }
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn precision_pipeline_dynamic_prelu_retains_binary_broadcast_provenance() {
    let shape = vec![
        Dimension::Dynamic(rustnn::graph::DynamicDimension {
            name: "batch".into(),
            max_size: 3,
        }),
        Dimension::Static(4),
    ];
    let graph = GraphInfo {
        operands: vec![
            operand("input", OperandKind::Input, DataType::Float16, &shape),
            operand(
                "slope",
                OperandKind::Constant,
                DataType::Float16,
                &[Dimension::Static(4)],
            ),
            operand("result", OperandKind::Output, DataType::Float16, &shape),
        ],
        input_operands: vec![0],
        output_operands: vec![2],
        operations: vec![
            Operation::from_json_attributes("prelu", &[0, 1], &[2], &serde_json::json!({}))
                .unwrap(),
        ],
        constant_operand_ids_to_handles: [(
            1,
            ConstantData {
                data: [0.25f32, 0.5, 0.125, 0.75]
                    .iter()
                    .flat_map(|&value| reference_half(value).to_bits().to_le_bytes())
                    .collect(),
                label: None,
            },
        )]
        .into_iter()
        .collect(),
        ..Default::default()
    };
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let model = Model::decode(converted.data.as_slice()).unwrap();
    let Some(model::Type::Pipeline(pipeline)) = &model.r#type else {
        panic!("pipeline")
    };
    let mut found = false;
    for child in &pipeline.models {
        for output in &child.description.as_ref().unwrap().output {
            let Some(specification::feature_type::Type::MultiArrayType(array)) =
                &output.r#type.as_ref().unwrap().r#type
            else {
                panic!("array")
            };
            if array.shape.len() == 2 {
                assert_eq!(array.shape[1], 4);
                found = true;
            }
        }
    }
    assert!(found);
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn precision_pipeline_materializes_dynamic_float32_prelu_before_explicit_half_cast() {
    use rustnn::mlcontext::{MLNamedOperands, MLOperandDescriptor};
    use rustnn::mlgraphbuilder::MLGraphBuilder;
    let mut builder = MLGraphBuilder::new_uncompiled();
    let input = builder
        .input(
            "input",
            &MLOperandDescriptor::new(MLOperandDataType::Float16, vec![1, 4]),
        )
        .unwrap();
    let slope = builder
        .constant_from_bytes(
            &MLOperandDescriptor::new(MLOperandDataType::Float16, vec![4]),
            [0.25f32, 0.5, 0.125, 0.75]
                .iter()
                .flat_map(|&x| reference_half(x).to_bits().to_le_bytes())
                .collect(),
        )
        .unwrap();
    let input = builder.cast(input, MLOperandDataType::Float32).unwrap();
    let slope = builder.cast(slope, MLOperandDataType::Float32).unwrap();
    let result = builder.prelu(input, slope).unwrap();
    let result = builder.cast(result, MLOperandDataType::Float16).unwrap();
    let mut outputs = MLNamedOperands::new();
    outputs.insert("result", result);
    let mut graph = builder.finish_graph_info(&outputs).unwrap();
    for operand in &mut graph.operands {
        if operand.descriptor.shape.len() == 2 {
            operand.descriptor.shape[0] = Dimension::Dynamic(rustnn::graph::DynamicDimension {
                name: "batch".into(),
                max_size: 3,
            });
        }
    }
    let children = pipeline(&graph);
    assert_eq!(children.len(), 3);
    let Some(model::Type::MlProgram(program)) = &children[2].r#type else {
        panic!("program")
    };
    let function = &program.functions["main"];
    let block = &function.block_specializations[&function.opset];
    assert_eq!(block.operations.len(), 1);
    assert_eq!(block.operations[0].r#type, "cast");
    let Some(specification::feature_type::Type::MultiArrayType(array)) =
        &children[2].description.as_ref().unwrap().input[0]
            .r#type
            .as_ref()
            .unwrap()
            .r#type
    else {
        panic!("array")
    };
    assert_eq!(
        array.data_type,
        specification::array_feature_type::ArrayDataType::Float32 as i32
    );
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn precision_pipeline_half_subtraction_preserves_all_encoding_classes() {
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    use rustnn::mlcontext::{
        MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    let input_bytes: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
    for device_type in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, device_type != DeviceType::Cpu)
                .with_rustnn_device_hint(BackendDevice::Coreml { device_type }),
        )
        .unwrap();
        let mut graph = context
            .rustnn_build_graph(half_subtraction(&[Dimension::Static(65536)], Some("isNaN")))
            .unwrap();
        let input = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Float16, vec![65536]).to_writable(),
            )
            .unwrap();
        let difference = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Float16, vec![65536]).to_readable(),
            )
            .unwrap();
        let classified = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Uint8, vec![65536]).to_readable(),
            )
            .unwrap();
        context.write_tensor(&input, &input_bytes).unwrap();
        context
            .dispatch(
                &mut graph,
                &MLNamedTensors::from([("input", &input)]),
                &MLNamedTensors::from([("difference", &difference), ("classified", &classified)]),
            )
            .unwrap();
        let mut output = vec![0_u8; input_bytes.len()];
        let mut flags = vec![0_u8; 65536];
        context.read_tensor(&difference, &mut output).unwrap();
        context.read_tensor(&classified, &mut flags).unwrap();
        for (index, &bytes) in output.as_chunks::<2>().0.iter().enumerate() {
            let source = half::f16::from_bits(index as u16);
            let actual = half::f16::from_bits(u16::from_le_bytes(bytes));
            let expected_nan = source.is_nan() || source.is_infinite();
            assert_eq!(actual.is_nan(), expected_nan, "{device_type:?}/{index:04x}");
            assert_eq!(
                flags[index],
                u8::from(expected_nan),
                "{device_type:?}/{index:04x}"
            );
            if !expected_nan {
                assert_eq!(actual.to_bits(), 0, "{device_type:?}/{index:04x}");
            }
        }
    }
}

#[test]
fn precision_pipeline_materializes_widening_before_signed_zero_sensitive_math() {
    for kind in ["reciprocal", "roundEven", "neg"] {
        let children = pipeline(&half_unary(kind, 16));
        assert_eq!(
            children.len(),
            if kind == "reciprocal" { 2 } else { 3 },
            "{kind}"
        );
        let Some(model::Type::MlProgram(program)) = &children[0].r#type else {
            panic!("program")
        };
        let function = &program.functions["main"];
        let block = &function.block_specializations[&function.opset];
        assert_eq!(block.operations.len(), 1, "{kind}");
        assert_eq!(block.operations[0].r#type, "cast");
        let output = &children[0].description.as_ref().unwrap().output[0];
        let Some(specification::feature_type::Type::MultiArrayType(array)) =
            &output.r#type.as_ref().unwrap().r#type
        else {
            panic!("array")
        };
        assert_eq!(
            array.data_type,
            specification::array_feature_type::ArrayDataType::Float32 as i32
        );
        assert_eq!(
            children[1].description.as_ref().unwrap().input.as_slice(),
            std::slice::from_ref(output)
        );
        if kind == "neg" {
            let Some(model::Type::MlProgram(program)) = &children[1].r#type else {
                panic!("program")
            };
            let function = &program.functions["main"];
            let block = &function.block_specializations[&function.opset];
            assert_eq!(block.operations[0].r#type, "real_div");
        }
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn precision_pipeline_signed_zero_sensitive_half_math_preserves_all_encodings() {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
    let values: Vec<_> = (0..=u16::MAX)
        .map(|bits| half::f16::from_bits(bits).to_f32_const())
        .collect();
    for kind in ["reciprocal", "roundEven", "neg"] {
        let converted = CoremlMlProgramConverter
            .convert(&half_unary(kind, values.len() as u32))
            .unwrap();
        for attempt in run_coreml_with_inputs_with_weights(
            &converted.data,
            converted.weights_data.as_deref(),
            vec![CoremlInput {
                name: "input".into(),
                shape: vec![values.len()],
                data: values.clone(),
            }],
        )
        .unwrap()
        {
            let outputs = attempt.result.unwrap();
            let result = outputs
                .iter()
                .find(|output| output.name == "result")
                .unwrap();
            for (index, (&actual, &input)) in result.data.iter().zip(&values).enumerate() {
                let expected = reference_half(match kind {
                    "reciprocal" => 1.0 / input,
                    "neg" => -input,
                    _ => input.round_ties_even(),
                })
                .to_f32_const();
                if expected.is_nan() {
                    assert!(
                        actual.is_nan(),
                        "{kind} {} {index} NaN",
                        attempt.compute_unit
                    );
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{kind} {} {index}",
                        attempt.compute_unit
                    );
                }
            }
        }
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn precision_pipeline_nearest_even_float32_preserves_edges_without_native_round() {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
    let values = vec![
        f32::NEG_INFINITY,
        -f32::MAX,
        -8388608.0,
        -8388607.5,
        -2.5,
        -1.5,
        -0.5,
        -0.49999997,
        -f32::MIN_POSITIVE,
        -f32::from_bits(1),
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        0.49999997,
        0.5,
        1.5,
        2.5,
        8388607.5,
        8388608.0,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
    ];
    let mut graph = half_unary("roundEven", values.len() as u32);
    for operand in &mut graph.operands {
        operand.descriptor.data_type = DataType::Float32;
    }
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    for attempt in run_coreml_with_inputs_with_weights(
        &converted.data,
        converted.weights_data.as_deref(),
        vec![CoremlInput {
            name: "input".into(),
            shape: vec![values.len()],
            data: values.clone(),
        }],
    )
    .unwrap()
    {
        let outputs = attempt.result.unwrap();
        let actual = &outputs
            .iter()
            .find(|output| output.name == "result")
            .unwrap()
            .data;
        for (index, (&actual, &input)) in actual.iter().zip(&values).enumerate() {
            let expected = input.round_ties_even();
            if expected.is_nan() {
                assert!(actual.is_nan(), "{} {index}", attempt.compute_unit);
            } else {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{} {index}",
                    attempt.compute_unit
                );
            }
        }
    }
}

#[test]
#[cfg(feature = "dynamic-inputs")]
fn precision_pipeline_dynamic_restore_has_a_pure_fp32_input() {
    let graph = direct_half_widening(&[
        Dimension::Static(1),
        Dimension::Static(16),
        Dimension::Dynamic(rustnn::graph::DynamicDimension {
            name: "tokens".into(),
            max_size: 4,
        }),
        Dimension::Static(1),
    ]);
    let children = pipeline(&graph);
    assert_eq!(children.len(), 2);
    let Some(model::Type::MlProgram(program)) = &children[0].r#type else {
        panic!("program")
    };
    let function = &program.functions["main"];
    let block = &function.block_specializations[&function.opset];
    assert_eq!(block.operations.len(), 1);
    assert_eq!(block.operations[0].r#type, "cast");
    let output = &children[0].description.as_ref().unwrap().output[0];
    let Some(specification::feature_type::Type::MultiArrayType(array)) =
        &output.r#type.as_ref().unwrap().r#type
    else {
        panic!("array")
    };
    assert_eq!(
        array.data_type,
        specification::array_feature_type::ArrayDataType::Float32 as i32
    );
    assert!(
        children[1]
            .description
            .as_ref()
            .unwrap()
            .input
            .contains(output)
    );
    let Some(model::Type::MlProgram(program)) = &children[1].r#type else {
        panic!("program")
    };
    let function = &program.functions["main"];
    let block = &function.block_specializations[&function.opset];
    assert_eq!(
        block
            .operations
            .iter()
            .map(|operation| operation.r#type.as_str())
            .collect::<Vec<_>>(),
        ["shape", "reshape"]
    );
}

#[test]
fn precision_pipeline_packs_high_rank_half_widening_without_changing_public_shapes() {
    for shape in [vec![1, 16, 1, 1], vec![1, 4, 17]] {
        let dimensions: Vec<_> = shape.iter().copied().map(Dimension::Static).collect();
        let graph = direct_half_widening(&dimensions);
        let before = serde_json::to_value(&graph).unwrap();
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        assert_eq!(serde_json::to_value(&graph).unwrap(), before);
        let model = Model::decode(converted.data.as_slice()).unwrap();
        let description = model.description.as_ref().unwrap();
        assert_eq!(description.input.len(), 2);
        assert_eq!(description.input[0].name, "input");
        let views: serde_json::Value = serde_json::from_str(&description.metadata.as_ref().unwrap().user_defined["rustnn.webnn.compact_input_views"]).unwrap();
        assert_eq!(views[0]["source"], "input");
        assert_eq!(views[0]["view"], description.input[1].name);
        let Some(model::Type::Pipeline(pipeline)) = model.r#type else {
            panic!("pipeline")
        };
        let stages = pipeline.models;
        assert!(
            stages
                .iter()
                .all(|stage| stage.description.as_ref().unwrap().metadata.is_none())
        );
        assert_eq!(stages.len(), 2);
        let output = &stages[0].description.as_ref().unwrap().output[0];
        let Some(specification::feature_type::Type::MultiArrayType(array)) =
            &output.r#type.as_ref().unwrap().r#type
        else {
            panic!("array")
        };
        assert_eq!(
            array.shape,
            [shape.iter().copied().map(i64::from).product::<i64>()]
        );
        assert_eq!(
            array.data_type,
            specification::array_feature_type::ArrayDataType::Float32 as i32
        );
        let output = &stages[1].description.as_ref().unwrap().output[0];
        let Some(specification::feature_type::Type::MultiArrayType(array)) =
            &output.r#type.as_ref().unwrap().r#type
        else {
            panic!("array")
        };
        assert_eq!(
            array.shape,
            shape.iter().copied().map(i64::from).collect::<Vec<_>>()
        );
    }
}

#[test]
fn precision_pipeline_compact_input_views_are_deduplicated_and_collision_free() {
    let shape = [
        Dimension::Static(1),
        Dimension::Static(16),
        Dimension::Static(1),
    ];
    let mut graph = direct_half_widening(&shape);
    graph.operands[1].name = Some("input_compact_input".into());
    graph.operands.push(operand(
        "second",
        OperandKind::Output,
        DataType::Float32,
        &shape,
    ));
    graph
        .operations
        .push(cast(0, 2, MLOperandDataType::Float32));
    graph.output_operands.push(2);
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let model = Model::decode(converted.data.as_slice()).unwrap();
    let description = model.description.unwrap();
    assert_eq!(description.input.len(), 2);
    let views: serde_json::Value = serde_json::from_str(
        &description.metadata.unwrap().user_defined["rustnn.webnn.compact_input_views"],
    )
    .unwrap();
    assert_eq!(views.as_array().unwrap().len(), 1);
    assert_eq!(views[0]["view"], "input_compact_input_1");
    assert_eq!(graph.input_operands, [0]);
}

#[test]
fn precision_pipeline_rejects_overflowing_compact_bounds() {
    let graph = direct_half_widening(&[
        Dimension::Static(u32::MAX),
        Dimension::Static(2),
        Dimension::Static(1),
    ]);
    let error = CoremlMlProgramConverter.convert(&graph).unwrap_err();
    assert!(error.to_string().contains("compact feature size limit"));
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn precision_pipeline_packed_half_widening_preserves_special_value_bits() {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
    let bits = [
        0x8000u16, 0, 1, 0x8001, 0x3ff, 0x83ff, 0x400, 0x8400, 0x3c00, 0xbc00, 0x4000, 0xc000,
        0x7c00, 0xfc00, 0x7e00, 0x3555,
    ];
    let values: Vec<_> = bits
        .iter()
        .map(|&bits| half::f16::from_bits(bits).to_f32_const())
        .collect();
    let shape = [1, 16, 1, 1];
    let graph = direct_half_widening(&shape.map(Dimension::Static));
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let attempts = run_coreml_with_inputs_with_weights(
        &converted.data,
        converted.weights_data.as_deref(),
        vec![CoremlInput {
            name: "input".into(),
            shape: shape.map(|x| x as usize).to_vec(),
            data: values.clone(),
        }],
    )
    .unwrap();
    for attempt in attempts {
        let outputs = attempt.result.unwrap();
        let output = outputs
            .iter()
            .find(|output| output.name == "result")
            .unwrap();
        assert_eq!(output.shape, shape.map(i64::from));
        for (index, (&actual, &expected)) in output.data.iter().zip(&values).enumerate() {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{} index {index}",
                    attempt.compute_unit
                );
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
fn precision_pipeline_packed_half_widening_restores_actual_dynamic_shape() {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
    let graph = direct_half_widening(&[
        Dimension::Static(1),
        Dimension::Static(16),
        Dimension::Dynamic(rustnn::graph::DynamicDimension {
            name: "tokens".into(),
            max_size: 4,
        }),
        Dimension::Static(1),
    ]);
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    for length in [1usize, 3] {
        let expected: Vec<_> = (0..16 * length)
            .map(|index| {
                half::f16::from_bits(if index % 2 == 0 { 1 } else { 0x8000 }).to_f32_const()
            })
            .collect();
        let attempts = run_coreml_with_inputs_with_weights(
            &converted.data,
            converted.weights_data.as_deref(),
            vec![CoremlInput {
                name: "input".into(),
                shape: vec![1, 16, length, 1],
                data: expected.clone(),
            }],
        )
        .unwrap();
        for attempt in attempts {
            let outputs = attempt.result.unwrap();
            let output = outputs
                .iter()
                .find(|output| output.name == "result")
                .unwrap();
            assert_eq!(output.shape, [1, 16, length as i64, 1]);
            for (&actual, &expected) in output.data.iter().zip(&expected) {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{}",
                    attempt.compute_unit
                );
            }
        }
    }
}

fn same_type_float32_cast() -> GraphInfo {
    let shape = [Dimension::Static(14)];
    GraphInfo {
        operands: vec![
            operand("input", OperandKind::Input, DataType::Float32, &shape),
            operand("result", OperandKind::Output, DataType::Float32, &shape),
        ],
        input_operands: vec![0],
        output_operands: vec![1],
        operations: vec![cast(0, 1, MLOperandDataType::Float32)],
        ..Default::default()
    }
}

#[test]
fn precision_pipeline_same_type_float32_cast_is_exact_transport() {
    let converted = CoremlMlProgramConverter
        .convert(&same_type_float32_cast())
        .unwrap();
    let model = Model::decode(converted.data.as_slice()).unwrap();
    let Some(model::Type::MlProgram(program)) = model.r#type else {
        panic!("program")
    };
    let function = &program.functions["main"];
    let block = &function.block_specializations[&function.opset];
    assert!(block.operations.iter().any(|operation| {
        operation.r#type == "identity"
            && operation
                .outputs
                .iter()
                .any(|output| output.name == "result")
    }));
    assert!(
        block
            .operations
            .iter()
            .all(|operation| operation.r#type != "cast")
    );
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn precision_pipeline_same_type_float32_cast_preserves_bits() {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
    let values = vec![
        2.0f32,
        -4.0,
        7.0,
        1.0003,
        -1.0003,
        1.0007,
        -1.0007,
        2.0f32.powi(-24),
        -2.0f32.powi(-24),
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let converted = CoremlMlProgramConverter
        .convert(&same_type_float32_cast())
        .unwrap();
    for attempt in run_coreml_with_inputs_with_weights(
        &converted.data,
        converted.weights_data.as_deref(),
        vec![CoremlInput {
            name: "input".into(),
            shape: vec![values.len()],
            data: values.clone(),
        }],
    )
    .unwrap()
    {
        let outputs = attempt.result.unwrap();
        let output = outputs
            .iter()
            .find(|output| output.name == "result")
            .unwrap();
        for (index, (&actual, &expected)) in output.data.iter().zip(&values).enumerate() {
            if expected.is_nan() {
                assert!(actual.is_nan(), "{} {index} NaN", attempt.compute_unit);
            } else {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{} {index}",
                    attempt.compute_unit
                );
            }
        }
    }
}

fn successive_exposed_casts(shape: &[Dimension]) -> GraphInfo {
    let mut graph = pair(shape);
    graph.operands[1].kind = OperandKind::Output;
    graph.operands.extend([
        operand("second_half", OperandKind::Output, DataType::Float16, shape),
        operand("second_wide", OperandKind::Output, DataType::Float32, shape),
    ]);
    graph.operations.extend([
        cast(2, 3, MLOperandDataType::Float16),
        cast(3, 4, MLOperandDataType::Float32),
    ]);
    graph.output_operands = vec![1, 2, 3, 4];
    graph
}

#[test]
fn precision_pipeline_keeps_successive_exposed_casts_in_distinct_children() {
    let stages = pipeline(&successive_exposed_casts(&[Dimension::Static(20)]));
    assert_eq!(stages.len(), 4);
    for (stage, name) in stages
        .iter()
        .zip(["rounded", "result", "second_half", "second_wide"])
    {
        let description = stage.description.as_ref().unwrap();
        assert_eq!(description.input.len(), 1);
        assert_eq!(description.output.len(), 1);
        assert_eq!(description.output[0].name, name);
    }
}

fn pipeline(graph: &GraphInfo) -> Vec<Model> {
    let converted = CoremlMlProgramConverter.convert(graph).unwrap();
    let model = Model::decode(converted.data.as_slice()).unwrap();
    let Some(model::Type::Pipeline(pipeline)) = model.r#type else {
        panic!("expected precision Pipeline")
    };
    pipeline.models
}

#[test]
fn precision_pipeline_materializes_only_observable_explicit_half_casts() {
    let graph = pair(&[Dimension::Static(4)]);
    let stages = pipeline(&graph);
    assert_eq!(stages.len(), 2);
    let rounded = &stages[0].description.as_ref().unwrap().output[0];
    assert_eq!(rounded.name, "rounded");
    let Some(specification::feature_type::Type::MultiArrayType(array)) =
        rounded.r#type.as_ref().unwrap().r#type.as_ref()
    else {
        panic!("array output")
    };
    assert_eq!(
        array.data_type,
        specification::array_feature_type::ArrayDataType::Float16 as i32
    );
    assert_eq!(stages[1].description.as_ref().unwrap().input[0], *rounded);
    let mut terminal = graph;
    terminal.operations.pop();
    terminal.output_operands = vec![1];
    let converted = CoremlMlProgramConverter.convert(&terminal).unwrap();
    assert!(matches!(
        Model::decode(converted.data.as_slice()).unwrap().r#type,
        Some(model::Type::MlProgram(_))
    ));
}

#[test]
fn precision_pipeline_fanout_reads_an_earlier_stage_without_passthrough_buffers() {
    let mut graph = pair(&[Dimension::Static(4)]);
    graph.operands[2].kind = OperandKind::Intermediate;
    graph.operands[2].name = Some("first_widened".into());
    graph.operands.extend([
        operand(
            "second_half",
            OperandKind::Intermediate,
            DataType::Float16,
            &[Dimension::Static(4)],
        ),
        operand(
            "second_widened",
            OperandKind::Intermediate,
            DataType::Float32,
            &[Dimension::Static(4)],
        ),
        operand(
            "result",
            OperandKind::Output,
            DataType::Float32,
            &[Dimension::Static(4)],
        ),
    ]);
    graph.operations.extend([
        cast(2, 3, MLOperandDataType::Float16),
        cast(3, 4, MLOperandDataType::Float32),
        Operation::Add {
            a: 2,
            b: 4,
            options: None,
            outputs: vec![5],
        },
    ]);
    graph.output_operands = vec![1, 5];
    graph.operands[1].kind = OperandKind::Output;
    let stages = pipeline(&graph);
    assert_eq!(stages.len(), 5);
    assert_eq!(
        stages[1]
            .description
            .as_ref()
            .unwrap()
            .output
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
        ["first_widened"]
    );
    assert_eq!(
        stages[4]
            .description
            .as_ref()
            .unwrap()
            .input
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
        ["first_widened", "second_widened"]
    );
}

fn weighted() -> GraphInfo {
    let shape = [Dimension::Static(4)];
    GraphInfo {
        operands: vec![
            operand("input", OperandKind::Input, DataType::Float32, &shape),
            operand("weights", OperandKind::Constant, DataType::Float32, &shape),
            operand("sum", OperandKind::Intermediate, DataType::Float32, &shape),
            operand("rounded", OperandKind::Output, DataType::Float16, &shape),
            operand(
                "widened",
                OperandKind::Intermediate,
                DataType::Float32,
                &shape,
            ),
            operand("result", OperandKind::Output, DataType::Float32, &shape),
        ],
        input_operands: vec![0],
        output_operands: vec![3, 5],
        constant_operand_ids_to_handles: [(
            1,
            ConstantData {
                data: [0.25f32, -0.25, 0.5, -0.5]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect(),
                label: None,
            },
        )]
        .into_iter()
        .collect(),
        operations: vec![
            Operation::Add {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![2],
            },
            cast(2, 3, MLOperandDataType::Float16),
            cast(3, 4, MLOperandDataType::Float32),
            Operation::Add {
                a: 4,
                b: 1,
                options: None,
                outputs: vec![5],
            },
        ],
        ..Default::default()
    }
}

fn masked() -> GraphInfo {
    let shape = [Dimension::Static(4)];
    GraphInfo {
        operands: vec![
            operand("input", OperandKind::Input, DataType::Float32, &shape),
            operand("threshold", OperandKind::Constant, DataType::Float32, &[]),
            operand("mask", OperandKind::Intermediate, DataType::Uint8, &shape),
            operand(
                "rounded",
                OperandKind::Intermediate,
                DataType::Float16,
                &shape,
            ),
            operand(
                "widened",
                OperandKind::Intermediate,
                DataType::Float32,
                &shape,
            ),
            operand("result", OperandKind::Output, DataType::Float32, &shape),
        ],
        input_operands: vec![0],
        output_operands: vec![5],
        constant_operand_ids_to_handles: [(
            1,
            ConstantData {
                data: 0f32.to_le_bytes().to_vec(),
                label: None,
            },
        )]
        .into_iter()
        .collect(),
        operations: vec![
            Operation::Greater {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![2],
            },
            cast(0, 3, MLOperandDataType::Float16),
            cast(3, 4, MLOperandDataType::Float32),
            Operation::Where {
                condition: 2,
                true_value: 4,
                false_value: 0,
                options: None,
                outputs: vec![5],
            },
        ],
        ..Default::default()
    }
}

fn indexed() -> GraphInfo {
    let shape = [Dimension::Static(2), Dimension::Static(2)];
    GraphInfo {
        operands: vec![
            operand("input", OperandKind::Input, DataType::Float32, &shape),
            operand(
                "indices",
                OperandKind::Intermediate,
                DataType::Int64,
                &[Dimension::Static(2)],
            ),
            operand(
                "rounded",
                OperandKind::Intermediate,
                DataType::Float16,
                &shape,
            ),
            operand(
                "widened",
                OperandKind::Intermediate,
                DataType::Float32,
                &shape,
            ),
            operand("result", OperandKind::Output, DataType::Float32, &shape),
        ],
        input_operands: vec![0],
        output_operands: vec![4],
        operations: vec![
            Operation::ArgMax {
                input: 0,
                axis: 1,
                options: Some(MLArgMinMaxOptions {
                    output_data_type: MLOperandDataType::Int64,
                    ..Default::default()
                }),
                outputs: vec![1],
            },
            cast(0, 2, MLOperandDataType::Float16),
            cast(2, 3, MLOperandDataType::Float32),
            Operation::Gather {
                input: 3,
                indices: 1,
                batch_dimensions: None,
                options: Some(MLGatherOptions {
                    axis: 1,
                    ..Default::default()
                }),
                outputs: vec![4],
            },
        ],
        ..Default::default()
    }
}

fn half_where(shape: &[Dimension]) -> GraphInfo {
    GraphInfo {
        operands: vec![
            operand("condition", OperandKind::Input, DataType::Uint8, shape),
            operand("yes", OperandKind::Input, DataType::Float16, shape),
            operand("no", OperandKind::Input, DataType::Float16, shape),
            operand("result", OperandKind::Output, DataType::Float16, shape),
        ],
        input_operands: vec![0, 1, 2],
        output_operands: vec![3],
        operations: vec![Operation::Where {
            condition: 0,
            true_value: 1,
            false_value: 2,
            options: None,
            outputs: vec![3],
        }],
        ..Default::default()
    }
}

fn half_where_shared_condition(shape: &[Dimension]) -> GraphInfo {
    let mut graph = half_where(shape);
    graph.operands.push(operand(
        "inverse",
        OperandKind::Output,
        DataType::Float16,
        shape,
    ));
    graph.operations.push(Operation::Where {
        condition: 0,
        true_value: 2,
        false_value: 1,
        options: None,
        outputs: vec![4],
    });
    graph.output_operands.push(4);
    graph
}

fn half_where_constant_condition(shared: bool) -> GraphInfo {
    let shape = [Dimension::Static(4)];
    let mut graph = if shared {
        half_where_shared_condition(&shape)
    } else {
        half_where(&shape)
    };
    graph.operands[0].kind = OperandKind::Constant;
    graph.input_operands.remove(0);
    graph.constant_operand_ids_to_handles.insert(
        0,
        ConstantData {
            data: vec![1, 0, 1, 0],
            label: None,
        },
    );
    graph
}

fn half_where_computed_condition(shape: &[Dimension]) -> GraphInfo {
    let mut graph = half_where_shared_condition(shape);
    graph.operands[0].descriptor.data_type = DataType::Float32;
    graph.operands.push(operand(
        "mask",
        OperandKind::Intermediate,
        DataType::Uint8,
        shape,
    ));
    graph.operands.push(operand(
        "zero",
        OperandKind::Constant,
        DataType::Float32,
        &[],
    ));
    graph.constant_operand_ids_to_handles.insert(
        6,
        ConstantData {
            data: 0f32.to_le_bytes().to_vec(),
            label: None,
        },
    );
    for operation in &mut graph.operations {
        let Operation::Where { condition, .. } = operation else {
            unreachable!()
        };
        *condition = 5;
    }
    graph.operations.insert(
        0,
        Operation::Greater {
            a: 0,
            b: 6,
            options: None,
            outputs: vec![5],
        },
    );
    graph
}

fn assert_half_widening_has_no_integer_outputs(stages: &[Model]) {
    use rustnn::protos::coreml::mil_spec::{
        DataType as MilType, argument::binding::Binding, value_type,
    };
    for stage in stages {
        let Some(model::Type::MlProgram(program)) = &stage.r#type else {
            panic!("child program")
        };
        let function = &program.functions["main"];
        let block = &function.block_specializations[&function.opset];
        let types: std::collections::HashMap<_, _> = function
            .inputs
            .iter()
            .chain(
                block
                    .operations
                    .iter()
                    .flat_map(|operation| &operation.outputs),
            )
            .map(|value| {
                let Some(value_type::Type::TensorType(tensor)) =
                    &value.r#type.as_ref().unwrap().r#type
                else {
                    panic!("tensor")
                };
                (value.name.as_str(), tensor.data_type)
            })
            .collect();
        let half_widening = block.operations.iter().any(|operation| {
            operation.r#type == "cast" && operation.outputs.iter().any(|output|
                types[output.name.as_str()] == MilType::Float32 as i32
            ) && operation.inputs.get("x").unwrap().arguments.iter().any(|binding|
                matches!(&binding.binding, Some(Binding::Name(name)) if types[name.as_str()] == MilType::Float16 as i32)
            )
        });
        if half_widening {
            for output in &stage.description.as_ref().unwrap().output {
                let Some(specification::feature_type::Type::MultiArrayType(array)) =
                    &output.r#type.as_ref().unwrap().r#type
                else {
                    panic!("array output")
                };
                assert_ne!(
                    array.data_type,
                    specification::array_feature_type::ArrayDataType::Int32 as i32,
                    "Half widening must not expose an integer condition in the same child: {}",
                    output.name
                );
                assert_ne!(
                    array.data_type,
                    specification::array_feature_type::ArrayDataType::Float16 as i32,
                    "Half widening must not also expose a narrowed Half output: {}",
                    output.name
                );
            }
        }
    }
}

#[test]
fn precision_pipeline_keeps_condition_casts_out_of_half_widening_children() {
    for graph in [
        half_where(&[Dimension::Static(4)]),
        half_where_shared_condition(&[Dimension::Static(4)]),
    ] {
        let stages = pipeline(&graph);
        assert_half_widening_has_no_integer_outputs(&stages);
        // Shared narrow condition casts remain local to the select kernel.
        // No UInt8/Bool-to-Int32 feature adapter is needed for this fanout.
        assert!(stages.iter().all(|stage| {
            stage
                .description
                .as_ref()
                .unwrap()
                .output
                .iter()
                .all(|feature| {
                    let Some(specification::feature_type::Type::MultiArrayType(array)) =
                        &feature.r#type.as_ref().unwrap().r#type
                    else {
                        return false;
                    };
                    array.data_type
                        != specification::array_feature_type::ArrayDataType::Int32 as i32
                })
        }));
        assert_eq!(
            stages
                .iter()
                .flat_map(|stage| {
                    let Some(model::Type::MlProgram(program)) = &stage.r#type else {
                        panic!("program")
                    };
                    program.functions["main"].block_specializations["CoreML7"]
                        .operations
                        .iter()
                })
                .filter(|operation| operation.r#type == "select")
                .count(),
            graph.operations.len()
        );
    }
}

#[test]
fn precision_pipeline_constant_conditions_do_not_create_outputless_children() {
    for shared in [false, true] {
        let stages = pipeline(&half_where_constant_condition(shared));
        assert_half_widening_has_no_integer_outputs(&stages);
        assert!(
            stages
                .iter()
                .all(|stage| !stage.description.as_ref().unwrap().output.is_empty())
        );
    }
    let stages = pipeline(&half_where_computed_condition(&[Dimension::Static(65536)]));
    assert!(
        stages
            .iter()
            .all(|stage| !stage.description.as_ref().unwrap().output.is_empty())
    );
}

fn half_pad() -> GraphInfo {
    GraphInfo {
        operands: vec![
            operand(
                "input",
                OperandKind::Input,
                DataType::Float16,
                &[Dimension::Static(4)],
            ),
            operand(
                "result",
                OperandKind::Output,
                DataType::Float16,
                &[Dimension::Static(7)],
            ),
        ],
        input_operands: vec![0],
        output_operands: vec![1],
        operations: vec![Operation::Pad {
            input: 0,
            beginning_padding: vec![1],
            ending_padding: vec![2],
            options: Some(MLPadOptions {
                value: Some(serde_json::json!(2.0f32.powi(-24))),
                ..Default::default()
            }),
            outputs: vec![1],
        }],
        ..Default::default()
    }
}

fn half_add_then_widen() -> GraphInfo {
    let shape = [Dimension::Static(4)];
    GraphInfo {
        operands: vec![
            operand("input", OperandKind::Input, DataType::Float16, &shape),
            operand(
                "increment",
                OperandKind::Constant,
                DataType::Float16,
                &shape,
            ),
            operand(
                "rounded",
                OperandKind::Intermediate,
                DataType::Float16,
                &shape,
            ),
            operand("result", OperandKind::Output, DataType::Float32, &shape),
        ],
        input_operands: vec![0],
        output_operands: vec![3],
        constant_operand_ids_to_handles: [(
            1,
            ConstantData {
                data: [
                    2.0f32.powi(-11),
                    -2.0f32.powi(-11),
                    2.0f32.powi(-11),
                    -2.0f32.powi(-11),
                ]
                .iter()
                .flat_map(|&x| reference_half(x).to_bits().to_le_bytes())
                .collect(),
                label: None,
            },
        )]
        .into_iter()
        .collect(),
        operations: vec![
            Operation::Add {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![2],
            },
            cast(2, 3, MLOperandDataType::Float32),
        ],
        ..Default::default()
    }
}

#[test]
fn precision_pipeline_materializes_half_arithmetic_before_widening() {
    assert_eq!(pipeline(&half_add_then_widen()).len(), 2);
}

#[cfg(feature = "dynamic-inputs")]
fn half_dynamic_identity() -> GraphInfo {
    let shape = [Dimension::Dynamic(rustnn::graph::DynamicDimension {
        name: "tokens".into(),
        max_size: 8,
    })];
    GraphInfo {
        operands: vec![
            operand("input", OperandKind::Input, DataType::Float16, &shape),
            operand("result", OperandKind::Output, DataType::Float16, &shape),
        ],
        input_operands: vec![0],
        output_operands: vec![1],
        operations: vec![Operation::Identity {
            input: 0,
            options: None,
            outputs: vec![1],
        }],
        ..Default::default()
    }
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn precision_pipeline_materializes_dynamic_half_transport_with_bounds() {
    let stages = pipeline(&half_dynamic_identity());
    assert_eq!(stages.len(), 3);
    for stage in stages {
        for feature in stage.description.unwrap().input.into_iter() {
            let Some(specification::feature_type::Type::MultiArrayType(array)) =
                feature.r#type.unwrap().r#type
            else {
                panic!("array")
            };
            let Some(specification::array_feature_type::ShapeFlexibility::ShapeRange(range)) =
                array.shape_flexibility
            else {
                panic!("bounded range")
            };
            assert_eq!(range.size_ranges[0].upper_bound, 8);
        }
    }
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn precision_pipeline_retains_dynamic_triangular_mask_provenance() {
    let mut graph = half_dynamic_identity();
    let shape = vec![
        Dimension::Dynamic(rustnn::graph::DynamicDimension {
            name: "rows".into(),
            max_size: 8,
        }),
        Dimension::Static(4),
    ];
    graph
        .operands
        .iter_mut()
        .for_each(|operand| operand.descriptor.shape = shape.clone());
    graph.operations = vec![Operation::Triangular {
        input: 0,
        options: Some(rustnn::operator_options::MLTriangularOptions {
            upper: Some(true),
            diagonal: 1,
            ..Default::default()
        }),
        outputs: vec![1],
    }];
    let stages = pipeline(&graph);
    assert_half_widening_has_no_integer_outputs(&stages);
    let condition = &stages
        .iter()
        .flat_map(|stage| &stage.description.as_ref().unwrap().input)
        .find(|feature| feature.name.contains("triangular_keep"))
        .unwrap();
    let Some(specification::feature_type::Type::MultiArrayType(array)) =
        condition.r#type.as_ref().unwrap().r#type.as_ref()
    else {
        panic!("mask interface")
    };
    let Some(specification::array_feature_type::ShapeFlexibility::ShapeRange(range)) =
        &array.shape_flexibility
    else {
        panic!("mask bounds")
    };
    assert_eq!(range.size_ranges[0].upper_bound, 8);
    assert_eq!(
        array.data_type,
        specification::array_feature_type::ArrayDataType::Int32 as i32
    );
}

#[test]
fn precision_pipeline_materializes_promoted_select_and_pad() {
    for graph in [half_where(&[Dimension::Static(4)]), half_pad()] {
        let stages = pipeline(&graph);
        assert_eq!(stages.len(), 3);
        for feature in &stages[1].description.as_ref().unwrap().output {
            let Some(specification::feature_type::Type::MultiArrayType(array)) =
                feature.r#type.as_ref().unwrap().r#type.as_ref()
            else {
                panic!("protected wide result")
            };
            assert_eq!(
                array.data_type,
                specification::array_feature_type::ArrayDataType::Float32 as i32
            );
        }
    }
}

#[test]
fn precision_pipeline_retains_integer_index_proxies_between_programs() {
    let stages = pipeline(&indexed());
    let indices = &stages[0].description.as_ref().unwrap().output[0];
    assert_eq!(indices.name, "indices");
    let Some(specification::feature_type::Type::MultiArrayType(array)) =
        indices.r#type.as_ref().unwrap().r#type.as_ref()
    else {
        panic!("index array")
    };
    assert_eq!(
        array.data_type,
        specification::array_feature_type::ArrayDataType::Int32 as i32
    );
    assert!(
        stages
            .last()
            .unwrap()
            .description
            .as_ref()
            .unwrap()
            .input
            .contains(indices)
    );
}

#[test]
fn precision_pipeline_retains_direct_input_and_constant_outputs() {
    // Raw GraphInfo conversion accepts these passthroughs as an extension.
    // WebNN build() rejects original input/constant operands as outputs; its
    // conformance fixtures must instead expose a distinct produced operand.
    let mut graph = weighted();
    graph.output_operands = vec![0, 1, 3, 5];
    let stages = pipeline(&graph);
    assert_eq!(
        stages[0]
            .description
            .as_ref()
            .unwrap()
            .output
            .iter()
            .map(|feature| feature.name.as_str())
            .collect::<Vec<_>>(),
        ["sum", "weights"]
    );
    assert_eq!(
        stages[1]
            .description
            .as_ref()
            .unwrap()
            .output
            .iter()
            .map(|feature| feature.name.as_str())
            .collect::<Vec<_>>(),
        ["rounded"]
    );
}

#[test]
fn precision_pipeline_adapts_mask_values_and_retains_original_input_fanout() {
    let stages = pipeline(&masked());
    let stage0 = stages[0].description.as_ref().unwrap();
    assert_eq!(
        stage0
            .output
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
        ["mask_precision_io", "rounded"]
    );
    let Some(specification::feature_type::Type::MultiArrayType(mask)) =
        stage0.output[0].r#type.as_ref().unwrap().r#type.as_ref()
    else {
        panic!("mask array")
    };
    assert_eq!(
        mask.data_type,
        specification::array_feature_type::ArrayDataType::Int32 as i32
    );
    assert_eq!(
        stages
            .last()
            .unwrap()
            .description
            .as_ref()
            .unwrap()
            .input
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
        ["input", "mask_precision_io", "widened"]
    );
}

#[test]
fn precision_pipeline_shares_one_blob_for_constants_consumed_in_both_stages() {
    let graph = weighted();
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let weights = converted.weights_data.unwrap();
    assert_eq!(u32::from_le_bytes(weights[..4].try_into().unwrap()), 1);
    assert_eq!(pipeline(&graph).len(), 4);
    if let Some(directory) = std::env::var_os("RUSTNN_PRECISION_FIXTURE_DIR") {
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("weighted_pipeline.mlmodel"), converted.data).unwrap();
        std::fs::write(directory.join("weights.bin"), weights).unwrap();
        std::fs::write(
            directory.join("cast_pipeline.mlmodel"),
            CoremlMlProgramConverter
                .convert(&pair(&[Dimension::Static(4)]))
                .unwrap()
                .data,
        )
        .unwrap();
        for (name, graph) in [
            ("masked_pipeline", masked()),
            ("indexed_pipeline", indexed()),
            ("where_pipeline", half_where(&[Dimension::Static(4)])),
            ("pad_pipeline", half_pad()),
            ("scalar_pipeline", pair(&[])),
        ] {
            let fixture = CoremlMlProgramConverter.convert(&graph).unwrap();
            std::fs::write(directory.join(format!("{name}.mlmodel")), fixture.data).unwrap();
        }
        #[cfg(feature = "dynamic-inputs")]
        {
            let graph = pair(&[Dimension::Dynamic(rustnn::graph::DynamicDimension {
                name: "tokens".into(),
                max_size: 8,
            })]);
            let fixture = CoremlMlProgramConverter.convert(&graph).unwrap();
            std::fs::write(directory.join("dynamic_pipeline.mlmodel"), fixture.data).unwrap();
            let fixture = CoremlMlProgramConverter
                .convert(&half_dynamic_identity())
                .unwrap();
            std::fs::write(
                directory.join("dynamic_identity_pipeline.mlmodel"),
                fixture.data,
            )
            .unwrap();
        }
    }
}

#[test]
fn precision_pipeline_rematerializes_half_weight_cast_closures_without_features() {
    let shape = [Dimension::Static(4)];
    let mut graph = pair(&shape);
    graph.operands.extend([
        operand("weight", OperandKind::Constant, DataType::Float16, &shape),
        operand(
            "wide_weight",
            OperandKind::Intermediate,
            DataType::Float32,
            &shape,
        ),
        operand("sum", OperandKind::Intermediate, DataType::Float32, &shape),
    ]);
    graph.constant_operand_ids_to_handles.insert(
        3,
        ConstantData {
            data: [1u16, 0x8001, 0x3c00, 0x7c00]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect(),
            label: None,
        },
    );
    graph
        .operations
        .insert(0, cast(3, 4, MLOperandDataType::Float32));
    graph.operations.insert(
        1,
        Operation::Add {
            a: 0,
            b: 4,
            options: None,
            outputs: vec![5],
        },
    );
    graph.operations[2] = cast(5, 1, MLOperandDataType::Float16);
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let model = Model::decode(converted.data.as_slice()).unwrap();
    let Some(model::Type::Pipeline(pipeline)) = model.r#type else {
        panic!("pipeline")
    };
    assert_eq!(
        pipeline.models.len(),
        3,
        "only the runtime sum and explicit casts may create native children"
    );
    for stage in &pipeline.models {
        for feature in &stage.description.as_ref().unwrap().input {
            assert_ne!(feature.name, "wide_weight");
        }
        for feature in &stage.description.as_ref().unwrap().output {
            assert_ne!(feature.name, "wide_weight");
        }
    }
    let model::Type::MlProgram(program) = pipeline.models[0].r#type.as_ref().unwrap() else {
        panic!("program")
    };
    let function = &program.functions["main"];
    let block = &function.block_specializations[&function.opset];
    assert!(
        block
            .operations
            .iter()
            .any(|op| op.r#type == "cast" && op.outputs.iter().any(|x| x.name == "wide_weight"))
    );
    let original = &graph.constant_operand_ids_to_handles[&3].data;
    let weights = converted.weights_data.as_ref().unwrap();
    assert_eq!(u32::from_le_bytes(weights[..4].try_into().unwrap()), 1);
    assert_eq!(&weights[128..128 + original.len()], original);
}

#[test]
fn precision_pipeline_rematerializes_constant_transpose_transport_closures() {
    for permutation in [vec![0, 1], vec![1, 0]] {
        let shape = [Dimension::Static(2), Dimension::Static(2)];
        let mut graph = pair(&shape);
        graph.operands.extend([
            operand("weight", OperandKind::Constant, DataType::Float16, &shape),
            operand(
                "transposed",
                OperandKind::Intermediate,
                DataType::Float16,
                &shape,
            ),
            operand(
                "wide_weight",
                OperandKind::Intermediate,
                DataType::Float32,
                &shape,
            ),
            operand("sum", OperandKind::Intermediate, DataType::Float32, &shape),
        ]);
        graph.constant_operand_ids_to_handles.insert(
            3,
            ConstantData {
                data: [1u16, 0x8001, 0x3c00, 0x3555]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect(),
                label: None,
            },
        );
        graph.operations = vec![
            Operation::Transpose {
                input: 3,
                options: Some(rustnn::operator_options::MLTransposeOptions {
                    permutation,
                    ..Default::default()
                }),
                outputs: vec![4],
            },
            cast(4, 5, MLOperandDataType::Float32),
            Operation::Add {
                a: 0,
                b: 5,
                options: None,
                outputs: vec![6],
            },
            cast(6, 1, MLOperandDataType::Float16),
            cast(1, 2, MLOperandDataType::Float32),
        ];
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let model = Model::decode(converted.data.as_slice()).unwrap();
        let Some(model::Type::Pipeline(pipeline)) = model.r#type else {
            panic!("pipeline")
        };
        assert_eq!(
            pipeline.models.len(),
            3,
            "constant transpose must stay inside its consuming child; the runtime sum remains materialized"
        );
        for stage in &pipeline.models {
            for feature in stage
                .description
                .as_ref()
                .unwrap()
                .input
                .iter()
                .chain(&stage.description.as_ref().unwrap().output)
            {
                assert!(!feature.name.contains("weight"));
                assert!(!feature.name.contains("transposed"));
            }
        }
        let weights = converted.weights_data.as_ref().unwrap();
        assert_eq!(u32::from_le_bytes(weights[..4].try_into().unwrap()), 1);
    }
}

fn transposed_constant_narrowing() -> GraphInfo {
    let shape = [Dimension::Static(2), Dimension::Static(2)];
    let mut graph = pair(&shape);
    graph.operands[0].kind = OperandKind::Constant;
    graph.input_operands.clear();
    graph.operands.push(operand(
        "transposed",
        OperandKind::Intermediate,
        DataType::Float32,
        &shape,
    ));
    graph.constant_operand_ids_to_handles.insert(
        0,
        ConstantData {
            data: [1.0003f32, -1.0003, 1.0007, -1.0007]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect(),
            label: None,
        },
    );
    graph.operations = vec![
        Operation::Transpose {
            input: 0,
            options: Some(rustnn::operator_options::MLTransposeOptions {
                permutation: vec![1, 0],
                ..Default::default()
            }),
            outputs: vec![3],
        },
        cast(3, 1, MLOperandDataType::Float16),
        cast(1, 2, MLOperandDataType::Float32),
    ];
    graph
}

#[test]
fn precision_pipeline_retains_nonfolded_constant_narrowing_boundaries() {
    assert_eq!(pipeline(&transposed_constant_narrowing()).len(), 2);
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn precision_pipeline_rounds_nonfolded_constant_transpose_casts() {
    use rustnn::executors::coreml::run_coreml_with_inputs_with_weights;
    let converted = CoremlMlProgramConverter
        .convert(&transposed_constant_narrowing())
        .unwrap();
    let expected: Vec<_> = [1.0003f32, 1.0007, -1.0003, -1.0007]
        .iter()
        .map(|&x| reference_half(x).to_f32_const().to_bits())
        .collect();
    for attempt in run_coreml_with_inputs_with_weights(
        &converted.data,
        converted.weights_data.as_deref(),
        vec![],
    )
    .unwrap()
    {
        let outputs = attempt.result.unwrap();
        let output = outputs
            .iter()
            .find(|output| output.name == "result")
            .unwrap();
        assert_eq!(
            output.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected,
            "{}",
            attempt.compute_unit
        );
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn precision_pipeline_half_select_shared_condition_preserves_all_encodings() {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
    let values: Vec<f32> = (0..=u16::MAX)
        .map(|bits| half::f16::from_bits(bits).to_f32_const())
        .collect();
    for graph in [
        half_where(&[Dimension::Static(65536)]),
        half_where_shared_condition(&[Dimension::Static(65536)]),
        half_where_computed_condition(&[Dimension::Static(65536)]),
    ] {
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let inputs = vec![
            CoremlInput {
                name: "condition".into(),
                shape: vec![65536],
                data: (0..65536).map(|i| (i % 2) as f32).collect(),
            },
            CoremlInput {
                name: "yes".into(),
                shape: vec![65536],
                data: values.clone(),
            },
            CoremlInput {
                name: "no".into(),
                shape: vec![65536],
                data: values.iter().map(|x| -*x).collect(),
            },
        ];
        for attempt in run_coreml_with_inputs_with_weights(
            &converted.data,
            converted.weights_data.as_deref(),
            inputs,
        )
        .unwrap()
        {
            for output in attempt.result.unwrap() {
                for (i, actual) in output.data.iter().enumerate() {
                    let positive = (i % 2 != 0) == (output.name == "result");
                    let expected = if positive { values[i] } else { -values[i] };
                    if expected.is_nan() {
                        assert!(
                            actual.is_nan(),
                            "{} {} NaN[{i}]",
                            attempt.compute_unit,
                            output.name
                        );
                    } else {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "{} {}[{i}]",
                            attempt.compute_unit,
                            output.name
                        );
                    }
                }
            }
        }
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn precision_pipeline_half_select_and_pad_preserve_represented_values() {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
    let small = 2.0f32.powi(-24);
    let values = [small, -small, -0.0, 1.0];
    let where_graph = half_where(&[Dimension::Static(4)]);
    let cases = [
        (
            where_graph,
            vec![
                CoremlInput {
                    name: "condition".into(),
                    shape: vec![4],
                    data: vec![1., 0., 1., 0.],
                },
                CoremlInput {
                    name: "yes".into(),
                    shape: vec![4],
                    data: values.to_vec(),
                },
                CoremlInput {
                    name: "no".into(),
                    shape: vec![4],
                    data: values.map(|x| -x).to_vec(),
                },
            ],
            vec![small, small, -0.0, -1.0],
        ),
        (
            half_pad(),
            vec![CoremlInput {
                name: "input".into(),
                shape: vec![4],
                data: values.to_vec(),
            }],
            vec![small, small, -small, -0.0, 1.0, small, small],
        ),
        (
            half_add_then_widen(),
            vec![CoremlInput {
                name: "input".into(),
                shape: vec![4],
                data: vec![1.0, -1.0, 1.0 + 2.0f32.powi(-10), -1.0 - 2.0f32.powi(-10)],
            }],
            vec![1.0, -1.0, 1.0 + 2.0f32.powi(-9), -1.0 - 2.0f32.powi(-9)],
        ),
    ];
    for (graph, inputs, expected) in cases {
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        for attempt in run_coreml_with_inputs_with_weights(
            &converted.data,
            converted.weights_data.as_deref(),
            inputs,
        )
        .unwrap()
        {
            let outputs = attempt.result.unwrap();
            let actual = &outputs
                .iter()
                .find(|output| output.name == "result")
                .unwrap()
                .data;
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "{}",
                attempt.compute_unit
            );
        }
    }
    for shared in [false, true] {
        let converted = CoremlMlProgramConverter
            .convert(&half_where_constant_condition(shared))
            .unwrap();
        let inputs = vec![
            CoremlInput {
                name: "yes".into(),
                shape: vec![4],
                data: values.to_vec(),
            },
            CoremlInput {
                name: "no".into(),
                shape: vec![4],
                data: values.map(|x| -x).to_vec(),
            },
        ];
        for attempt in run_coreml_with_inputs_with_weights(
            &converted.data,
            converted.weights_data.as_deref(),
            inputs,
        )
        .unwrap()
        {
            for output in attempt.result.unwrap() {
                let expected = [small, small, -0.0, -1.0];
                let expected = if output.name == "inverse" {
                    expected.map(|x| -x)
                } else {
                    expected
                };
                assert_eq!(
                    output.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "{} {}",
                    attempt.compute_unit,
                    output.name
                );
            }
        }
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn precision_pipeline_rounds_constant_casts_without_runtime_inputs() {
    use rustnn::executors::coreml::run_coreml_with_inputs_with_weights;
    for values in [
        vec![1.0003f32],
        vec![
            1.0003f32,
            -1.0003,
            1.0007,
            -1.0007,
            1.0 + 2.0f32.powi(-11),
            1.0 + 3.0 * 2.0f32.powi(-11),
            2.0f32.powi(-24),
            -2.0f32.powi(-24),
            2.0f32.powi(-25),
            -2.0f32.powi(-25),
            3.0 * 2.0f32.powi(-25),
            -3.0 * 2.0f32.powi(-25),
            65504.0,
            65520.0,
            -65520.0,
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ],
    ] {
        let shape = if values.len() == 1 {
            vec![]
        } else {
            vec![Dimension::Static(values.len() as u32)]
        };
        let mut graph = successive_exposed_casts(&shape);
        graph.input_operands.clear();
        graph.operands[0].kind = OperandKind::Constant;
        graph.constant_operand_ids_to_handles.insert(
            0,
            ConstantData {
                data: values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                label: None,
            },
        );
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let model = Model::decode(converted.data.as_slice()).unwrap();
        let Some(model::Type::Pipeline(pipeline)) = model.r#type else {
            panic!("public Half constants must be materialized")
        };
        assert_eq!(
            pipeline.models.len(),
            4,
            "each public logical cast output is isolated"
        );
        for attempt in run_coreml_with_inputs_with_weights(
            &converted.data,
            converted.weights_data.as_deref(),
            vec![],
        )
        .unwrap()
        {
            let outputs = attempt.result.unwrap();
            for name in ["rounded", "result", "second_half", "second_wide"] {
                let output = outputs.iter().find(|output| output.name == name).unwrap();
                assert_eq!(output.data.len(), values.len());
                for (index, (&actual, &value)) in output.data.iter().zip(&values).enumerate() {
                    let expected = reference_half(value).to_f32_const();
                    if expected.is_nan() {
                        assert!(
                            actual.is_nan(),
                            "{} {name}[{index}] NaN classification",
                            attempt.compute_unit
                        );
                    } else {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "{} {name}[{index}]",
                            attempt.compute_unit
                        );
                    }
                }
            }
        }
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn precision_pipeline_constant_identity_preserves_every_half_encoding() {
    use rustnn::executors::coreml::run_coreml_with_inputs_with_weights;
    use rustnn::mlcontext::{MLNamedOperands, MLOperandDescriptor};
    use rustnn::mlgraphbuilder::MLGraphBuilder;
    let bits: Vec<_> = (0..=u16::MAX).collect();
    let mut builder = MLGraphBuilder::new_uncompiled();
    let constant = builder
        .constant_from_bytes(
            &MLOperandDescriptor::new(MLOperandDataType::Float16, vec![65536]),
            bits.iter().flat_map(|x| x.to_le_bytes()).collect(),
        )
        .unwrap();
    let result = builder.identity(constant).unwrap();
    let mut outputs = MLNamedOperands::new();
    outputs.insert("result", result);
    let graph = builder.finish_graph_info(&outputs).unwrap();
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    for attempt in run_coreml_with_inputs_with_weights(
        &converted.data,
        converted.weights_data.as_deref(),
        vec![],
    )
    .unwrap()
    {
        let outputs = attempt.result.unwrap();
        let output = outputs
            .iter()
            .find(|output| output.name == "result")
            .unwrap();
        assert_eq!(output.data.len(), bits.len());
        for (&actual, &bits) in output.data.iter().zip(&bits) {
            let expected = half::f16::from_bits(bits).to_f32_const();
            if expected.is_nan() {
                assert!(actual.is_nan(), "{} {bits:04x}", attempt.compute_unit);
            } else {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{} {bits:04x}",
                    attempt.compute_unit
                );
            }
        }
    }
}

#[test]
fn precision_pipeline_scalar_keeps_the_one_element_coreml_interface() {
    for stage in pipeline(&pair(&[])) {
        let description = stage.description.unwrap();
        for feature in description.input.into_iter().chain(description.output) {
            let Some(specification::feature_type::Type::MultiArrayType(array)) =
                feature.r#type.unwrap().r#type
            else {
                panic!("array")
            };
            assert_eq!(array.shape, [1]);
        }
    }
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn precision_pipeline_dynamic_bounds_are_retained_between_programs() {
    let graph = pair(&[Dimension::Dynamic(rustnn::graph::DynamicDimension {
        name: "tokens".into(),
        max_size: 8,
    })]);
    let stages = pipeline(&graph);
    let output = &stages[0].description.as_ref().unwrap().output[0];
    let Some(specification::feature_type::Type::MultiArrayType(array)) =
        output.r#type.as_ref().unwrap().r#type.as_ref()
    else {
        panic!("array")
    };
    let Some(specification::array_feature_type::ShapeFlexibility::ShapeRange(range)) =
        &array.shape_flexibility
    else {
        panic!("shape range")
    };
    assert_eq!(range.size_ranges[0].upper_bound, 8);
    assert_eq!(&stages[1].description.as_ref().unwrap().input[0], output);
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn precision_pipeline_dynamic_bounds_do_not_guess_graph_suffix_names() {
    let shape = |name: &str, max_size| {
        vec![Dimension::Dynamic(rustnn::graph::DynamicDimension {
            name: name.into(),
            max_size,
        })]
    };
    let short = shape("short", 3);
    let long = shape("long", 7);
    let graph = GraphInfo {
        operands: vec![
            operand(
                "foo_graph",
                OperandKind::Intermediate,
                DataType::Float16,
                &short,
            ),
            operand("foo", OperandKind::Input, DataType::Float32, &long),
            operand("input", OperandKind::Input, DataType::Float16, &short),
            operand("offset", OperandKind::Constant, DataType::Float16, &[]),
            operand("result", OperandKind::Output, DataType::Float32, &short),
            operand(
                "other_half",
                OperandKind::Intermediate,
                DataType::Float16,
                &long,
            ),
            operand(
                "other_result",
                OperandKind::Output,
                DataType::Float32,
                &long,
            ),
        ],
        input_operands: vec![1, 2],
        output_operands: vec![4, 6],
        constant_operand_ids_to_handles: [(
            3,
            ConstantData {
                data: reference_half(0.0007).to_bits().to_le_bytes().to_vec(),
                label: None,
            },
        )]
        .into_iter()
        .collect(),
        operations: vec![
            Operation::Add {
                a: 2,
                b: 3,
                options: None,
                outputs: vec![0],
            },
            cast(0, 4, MLOperandDataType::Float32),
            cast(1, 5, MLOperandDataType::Float16),
            cast(5, 6, MLOperandDataType::Float32),
        ],
        ..Default::default()
    };
    let stages = pipeline(&graph);
    let feature = stages
        .iter()
        .flat_map(|stage| &stage.description.as_ref().unwrap().output)
        .find(|feature| feature.name == "foo_graph")
        .unwrap();
    let Some(specification::feature_type::Type::MultiArrayType(array)) =
        &feature.r#type.as_ref().unwrap().r#type
    else {
        panic!("array")
    };
    let Some(specification::array_feature_type::ShapeFlexibility::ShapeRange(range)) =
        &array.shape_flexibility
    else {
        panic!("range")
    };
    assert_eq!(
        range.size_ranges[0].upper_bound, 3,
        "the unrelated foo input has max7, not foo_graph's max3"
    );
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn precision_pipeline_rounds_ties_subnormals_overflow_and_signed_zero() {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
    let values = vec![
        1.0003f32,
        -1.0003,
        1.0007,
        -1.0007,
        1.0 + 2.0f32.powi(-11),
        1.0 + 3.0 * 2.0f32.powi(-11),
        2.0f32.powi(-24),
        -2.0f32.powi(-24),
        2.0f32.powi(-25),
        -2.0f32.powi(-25),
        65504.0,
        65520.0,
        -65520.0,
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let converted = CoremlMlProgramConverter
        .convert(&pair(&[Dimension::Static(values.len() as u32)]))
        .unwrap();
    let input = CoremlInput {
        name: "input".into(),
        shape: vec![values.len()],
        data: values.clone(),
    };
    let attempts = run_coreml_with_inputs_with_weights(
        &converted.data,
        converted.weights_data.as_deref(),
        vec![input],
    )
    .unwrap();
    for attempt in attempts {
        let outputs = attempt.result.unwrap();
        let result = outputs
            .iter()
            .find(|output| output.name == "result")
            .unwrap();
        assert_eq!(result.data.len(), values.len());
        for (index, (&actual, &value)) in result.data.iter().zip(&values).enumerate() {
            let expected = reference_half(value).to_f32_const();
            if expected.is_nan() {
                assert!(
                    actual.is_nan(),
                    "{} {index}: NaN classification",
                    attempt.compute_unit
                );
            } else {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{} {index}: {value}",
                    attempt.compute_unit
                );
            }
        }
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn precision_pipeline_successive_exposed_casts_preserve_every_logical_output() {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
    let values = vec![
        1.0003f32,
        -1.0003,
        1.0007,
        -1.0007,
        1.0 + 2.0f32.powi(-11),
        1.0 + 3.0 * 2.0f32.powi(-11),
        2.0f32.powi(-24),
        -2.0f32.powi(-24),
        2.0f32.powi(-25),
        -2.0f32.powi(-25),
        3.0 * 2.0f32.powi(-25),
        -3.0 * 2.0f32.powi(-25),
        65504.0,
        65520.0,
        -65520.0,
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let graph = successive_exposed_casts(&[Dimension::Static(values.len() as u32)]);
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    for attempt in run_coreml_with_inputs_with_weights(
        &converted.data,
        converted.weights_data.as_deref(),
        vec![CoremlInput {
            name: "input".into(),
            shape: vec![values.len()],
            data: values.clone(),
        }],
    )
    .unwrap()
    {
        let outputs = attempt.result.unwrap();
        assert_eq!(outputs.len(), 4);
        for name in ["rounded", "result", "second_half", "second_wide"] {
            let output = outputs.iter().find(|output| output.name == name).unwrap();
            assert_eq!(output.shape, [values.len() as i64]);
            for (index, (&actual, &value)) in output.data.iter().zip(&values).enumerate() {
                let expected = reference_half(value).to_f32_const();
                if expected.is_nan() {
                    assert!(
                        actual.is_nan(),
                        "{} {name}[{index}] NaN classification",
                        attempt.compute_unit
                    );
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{} {name}[{index}]",
                        attempt.compute_unit
                    );
                }
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
fn precision_pipeline_predicts_actual_dynamic_lengths() {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
    let graph = pair(&[Dimension::Dynamic(rustnn::graph::DynamicDimension {
        name: "tokens".into(),
        max_size: 8,
    })]);
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    for length in [1, 3, 7] {
        let input = vec![1.0003f32; length];
        let attempts = run_coreml_with_inputs_with_weights(
            &converted.data,
            converted.weights_data.as_deref(),
            vec![CoremlInput {
                name: "input".into(),
                shape: vec![length],
                data: input,
            }],
        )
        .unwrap();
        for attempt in attempts {
            let outputs = attempt.result.unwrap();
            let output = outputs
                .iter()
                .find(|output| output.name == "result")
                .unwrap();
            assert_eq!(
                output.shape,
                [i64::try_from(length).unwrap()],
                "{}",
                attempt.compute_unit
            );
            assert_eq!(
                output.data,
                vec![reference_half(1.0003).to_f32_const(); length]
            );
        }
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn precision_pipeline_predicts_direct_input_and_constant_outputs() {
    // This qualifies the raw GraphInfo extension, not WebNN build() acceptance.
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
    let mut graph = weighted();
    graph.output_operands = vec![0, 1, 3, 5];
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let input = vec![1.0003f32; 4];
    for attempt in run_coreml_with_inputs_with_weights(
        &converted.data,
        converted.weights_data.as_deref(),
        vec![CoremlInput {
            name: "input".into(),
            shape: vec![4],
            data: input.clone(),
        }],
    )
    .unwrap()
    {
        let outputs = attempt.result.unwrap();
        assert_eq!(
            outputs
                .iter()
                .find(|output| output.name == "input")
                .unwrap()
                .data,
            input
        );
        assert_eq!(
            outputs
                .iter()
                .find(|output| output.name == "weights")
                .unwrap()
                .data,
            [0.25, -0.25, 0.5, -0.5]
        );
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn precision_pipeline_weighted_and_masked_compositions_predict_exact_values() {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
    let input = [1.0003f32, -1.0003, 1.0007, -1.0007];
    let weights = [0.25f32, -0.25, 0.5, -0.5];
    for (graph, expected) in [
        (
            weighted(),
            input
                .iter()
                .zip(weights)
                .map(|(&x, w)| reference_half(x + w).to_f32_const() + w)
                .collect::<Vec<_>>(),
        ),
        (
            masked(),
            input
                .iter()
                .map(|&x| {
                    if x > 0.0 {
                        reference_half(x).to_f32_const()
                    } else {
                        x
                    }
                })
                .collect::<Vec<_>>(),
        ),
        (
            indexed(),
            [input[0], input[0], input[2], input[2]]
                .map(|x| reference_half(x).to_f32_const())
                .to_vec(),
        ),
    ] {
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let attempts = run_coreml_with_inputs_with_weights(
            &converted.data,
            converted.weights_data.as_deref(),
            vec![CoremlInput {
                name: "input".into(),
                shape: graph.operands[0]
                    .descriptor
                    .static_or_max_shape()
                    .iter()
                    .map(|&dimension| dimension as usize)
                    .collect(),
                data: input.to_vec(),
            }],
        )
        .unwrap();
        for attempt in attempts {
            let outputs = attempt.result.unwrap();
            let output = outputs
                .iter()
                .find(|output| output.name == "result")
                .unwrap();
            assert_eq!(output.data, expected, "{}", attempt.compute_unit);
            if let Some(rounded) = outputs.iter().find(|output| output.name == "rounded") {
                assert_eq!(rounded.data_type_code, 65552);
                assert_eq!(
                    rounded.data,
                    input
                        .iter()
                        .zip(weights)
                        .map(|(&x, w)| reference_half(x + w).to_f32_const())
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

//! WebNN input and output namespaces stay distinct in the CoreML model.
#![cfg(all(target_os = "macos", feature = "coreml-runtime"))]

use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{
    DataType, Dimension, DynamicDimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::mlcontext::{
    Backend, MLContext, MLContextOptions, MLGraphBuilder, MLNamedOperands, MLNamedTensors,
    MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operators::Operation;
use rustnn::protos::coreml::specification;
use rustnn::validator::{ContextProperties, GraphValidator};
use serde_json::json;

fn api_dtype(dtype: DataType) -> MLOperandDataType {
    match dtype {
        DataType::Float32 => MLOperandDataType::Float32,
        DataType::Float16 => MLOperandDataType::Float16,
        DataType::Int32 => MLOperandDataType::Int32,
        DataType::Uint8 => MLOperandDataType::Uint8,
        _ => unreachable!(),
    }
}

#[test]
fn produced_output_may_have_the_same_logical_name_as_an_input() {
    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml),
    )
    .unwrap();
    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    let input = builder
        .input(
            "input",
            &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4]),
        )
        .unwrap();
    let copied = builder.identity(input).unwrap();
    let cast = builder.cast(input, MLOperandDataType::Float32).unwrap();
    let recorded = builder
        .finish_graph_info(&MLNamedOperands::from([("input", cast), ("copy", copied)]))
        .unwrap();
    GraphValidator::new(&recorded, ContextProperties::default())
        .validate()
        .unwrap();
    assert!(
        !recorded
            .input_operands
            .contains(&recorded.output_operands[0])
    );
    assert!(
        !recorded
            .input_operands
            .contains(&recorded.output_operands[1])
    );
    let mut graph = context.rustnn_build_graph(recorded).unwrap();
    let descriptor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4])
        .to_readable()
        .to_writable();
    let input = context.create_tensor(&descriptor).unwrap();
    let cast = context.create_tensor(&descriptor).unwrap();
    let copied = context.create_tensor(&descriptor).unwrap();
    let expected = [1.0003f32, -1.0003, -0.0, f32::from_bits(0x7fc12345)];
    context.write_tensor(&input, &expected).unwrap();
    context
        .dispatch(
            &mut graph,
            &MLNamedTensors::from([("input", &input)]),
            &MLNamedTensors::from([("input", &cast), ("copy", &copied)]),
        )
        .unwrap();
    for output in [&cast, &copied] {
        let mut actual = [0.0f32; 4];
        context.read_tensor(output, &mut actual).unwrap();
        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
    }
    // Dispatch fills each requested tensor, rather than exposing an input or
    // another output's mutable storage as a logical copy.
    context.write_tensor(&input, &[9.0f32; 4]).unwrap();
    context.write_tensor(&cast, &[8.0f32; 4]).unwrap();
    let mut unchanged = [0.0f32; 4];
    context.read_tensor(&copied, &mut unchanged).unwrap();
    assert_eq!(unchanged.map(f32::to_bits), expected.map(f32::to_bits));
}

fn named_graph(dtype: DataType, shape: Vec<Dimension>, half_output: bool) -> GraphInfo {
    let alias_dtype = if half_output {
        DataType::Float16
    } else {
        dtype
    };
    let operand = |name: &str, kind, dtype| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: dtype,
            shape: shape.clone(),
            pending_permutation: vec![],
        },
    };
    GraphInfo {
        operands: vec![
            operand("shared", OperandKind::Input, dtype),
            operand("shared", OperandKind::Output, alias_dtype),
            operand("copy", OperandKind::Output, dtype),
            operand("forwarded", OperandKind::Output, alias_dtype),
        ],
        input_operands: vec![0],
        output_operands: vec![1, 2, 3],
        operations: vec![
            Operation::Cast {
                input: 0,
                data_type: api_dtype(alias_dtype),
                options: None,
                outputs: vec![1],
            },
            Operation::Identity {
                input: 0,
                options: None,
                outputs: vec![2],
            },
            Operation::Identity {
                input: 1,
                options: None,
                outputs: vec![3],
            },
        ],
        ..Default::default()
    }
}

fn data(dtype: DataType, count: usize) -> Vec<u8> {
    match dtype {
        DataType::Float32 => [
            1.0003f32,
            -1.0003,
            2.0f32.powi(-24),
            -2.0f32.powi(-24),
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            2.0,
            -4.0,
        ]
        .into_iter()
        .cycle()
        .take(count)
        .flat_map(f32::to_le_bytes)
        .collect(),
        DataType::Float16 => [
            1u16, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0, 0x8000, 0x7c00, 0xfc00, 0x7e01,
        ]
        .into_iter()
        .cycle()
        .take(count)
        .flat_map(u16::to_le_bytes)
        .collect(),
        DataType::Int32 => [
            16777217i32,
            -16777217,
            i32::MAX,
            i32::MIN,
            0,
            1,
            -1,
            8388609,
            100000003,
            -100000003,
            i32::MAX - 1,
        ]
        .into_iter()
        .cycle()
        .take(count)
        .flat_map(i32::to_le_bytes)
        .collect(),
        DataType::Uint8 => [0u8, 1, 127, 128, 255, 2, 129, 254, 42, 17, 240]
            .into_iter()
            .cycle()
            .take(count)
            .collect(),
        _ => unreachable!(),
    }
}

fn half_bytes(input: &[u8]) -> Vec<u8> {
    input
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|bytes| {
            half::f16::from_f32(f32::from_le_bytes(*bytes))
                .to_bits()
                .to_le_bytes()
        })
        .collect()
}

#[test]
fn alias_metadata_preserves_names_dtypes_shapes_and_unique_ssa_values() {
    for (dtype, half_output) in [
        (DataType::Float32, false),
        (DataType::Float16, false),
        (DataType::Int32, false),
        (DataType::Uint8, false),
        (DataType::Float32, true),
    ] {
        for shape in [vec![], vec![Dimension::Static(11)]] {
            let graph = named_graph(dtype, shape, half_output);
            GraphValidator::new(&graph, ContextProperties::default())
                .validate()
                .unwrap();
            let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
            let model = specification::Model::decode(converted.data.as_slice()).unwrap();
            let description = model.description.as_ref().unwrap();
            assert_eq!(description.input[0].name, "shared");
            assert_ne!(description.output[0].name, "shared");
            assert_eq!(description.output.len(), if half_output { 2 } else { 1 });
            let aliases: std::collections::HashMap<String, String> = serde_json::from_str(
                &description.metadata.as_ref().unwrap().user_defined["rustnn.webnn.output_aliases"],
            )
            .unwrap();
            assert_eq!(aliases.len(), 3);
            assert_eq!(aliases["shared"], description.output[0].name);
            assert_eq!(aliases["forwarded"], aliases["shared"]);
            assert_eq!(aliases["copy"] == aliases["shared"], !half_output);
            let passthroughs: serde_json::Value = serde_json::from_str(
                &description.metadata.as_ref().unwrap().user_defined["rustnn.webnn.output_passthroughs"],
            ).unwrap();
            let passthroughs = passthroughs.as_object().unwrap();
            assert_eq!(passthroughs.len(), if half_output { 1 } else { 3 });
            assert_eq!(passthroughs["copy"]["input"], "shared");
            let children = match model.r#type.as_ref().unwrap() {
                specification::model::Type::MlProgram(_) => vec![&model],
                specification::model::Type::Pipeline(pipeline) => pipeline.models.iter().collect(),
                _ => panic!("expected MIL container"),
            };
            for child in children {
                let specification::model::Type::MlProgram(program) = child.r#type.as_ref().unwrap()
                else {
                    panic!("expected MLProgram child");
                };
                let function = &program.functions["main"];
                let block = &function.block_specializations["CoreML7"];
                assert_eq!(
                    block.outputs.len(),
                    child.description.as_ref().unwrap().output.len()
                );
                assert!(
                    block
                        .operations
                        .iter()
                        .all(|operation| operation.r#type != "mul")
                );
                let mut names: std::collections::HashSet<_> = function
                    .inputs
                    .iter()
                    .map(|input| input.name.as_str())
                    .collect();
                for operation in &block.operations {
                    for output in &operation.outputs {
                        assert!(names.insert(&output.name));
                    }
                }
            }
            assert_eq!(graph.operands[0].name.as_deref(), Some("shared"));
            assert_eq!(graph.operands[1].name.as_deref(), Some("shared"));
        }
    }
    let mut graph = named_graph(DataType::Float32, vec![Dimension::Static(1)], false);
    graph.operands[2].name = Some("__rustnn_operand_1".into());
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let description = specification::Model::decode(converted.data.as_slice())
        .unwrap()
        .description
        .unwrap();
    assert_ne!(description.output[0].name, "__rustnn_operand_1");
}

#[test]
fn ordinary_single_copy_output_keeps_its_name_and_serializes_the_source_proof() {
    let mut graph = named_graph(DataType::Float32, vec![Dimension::Static(11)], false);
    graph.operands[1].name = Some("different".into());
    graph.output_operands = vec![2];
    graph.operations = vec![Operation::Identity {
        input: 0,
        options: None,
        outputs: vec![2],
    }];
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let model = specification::Model::decode(converted.data.as_slice()).unwrap();
    let metadata = model.description.unwrap().metadata.unwrap();
    assert!(
        !metadata
            .user_defined
            .contains_key("rustnn.webnn.output_aliases")
    );
    let proof: serde_json::Value =
        serde_json::from_str(&metadata.user_defined["rustnn.webnn.output_passthroughs"]).unwrap();
    assert_eq!(proof["copy"]["input"], "shared");
    let specification::model::Type::MlProgram(program) = model.r#type.unwrap() else {
        panic!("expected program")
    };
    let block = &program.functions["main"].block_specializations["CoreML7"];
    assert!(block.operations.iter().all(|op| op.r#type != "mul"));

    graph.operations = vec![Operation::Neg {
        input: 0,
        options: None,
        outputs: vec![2],
    }];
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let model = specification::Model::decode(converted.data.as_slice()).unwrap();
    let metadata = model.description.unwrap().metadata.unwrap();
    assert_eq!(metadata.user_defined.len(), 1);
    assert_eq!(
        metadata.user_defined["rustnn.coreml.name_encoding"],
        "hex-v1"
    );
}

#[test]
fn checked_diagnostics_report_computed_scalars_without_relabeling_singletons() {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_checked};
    use std::collections::HashMap;
    for shape in [vec![], vec![Dimension::Static(1)]] {
        let mut graph = named_graph(DataType::Float32, shape, false);
        graph.operands[0].descriptor.shape = vec![Dimension::Static(1)];
        graph.operands[1].name = Some("unused".into());
        graph.output_operands = vec![2];
        graph.operations = vec![Operation::ReduceSum {
            input: 0,
            options: Some(rustnn::operator_options::MLReduceOptions {
                axes: Some(vec![0]),
                keep_dimensions: !graph.operands[2].descriptor.shape.is_empty(),
                ..Default::default()
            }),
            outputs: vec![2],
        }];
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let inputs = HashMap::from([("shared".into(), graph.operands[0].descriptor.clone())]);
        let outputs = HashMap::from([("copy".into(), graph.operands[2].descriptor.clone())]);
        let attempts = run_coreml_with_inputs_checked(
            &converted.data,
            vec![CoremlInput {
                name: "shared".into(),
                shape: vec![1],
                data: vec![2.0],
            }],
            &inputs,
            &outputs,
        )
        .unwrap();
        let result = attempts
            .iter()
            .find(|attempt| attempt.compute_unit == "CPU_ONLY")
            .unwrap()
            .result
            .as_ref()
            .unwrap();
        assert_eq!(
            result[0].shape,
            graph.operands[2]
                .descriptor
                .static_shape()
                .unwrap()
                .into_iter()
                .map(i64::from)
                .collect::<Vec<_>>()
        );
        assert_eq!(result[0].data, [2.0]);
    }
}

#[test]
fn input_as_output_still_requires_a_produced_operand() {
    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml),
    )
    .unwrap();
    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    let input = builder
        .input(
            "input",
            &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1]),
        )
        .unwrap();
    assert!(
        builder
            .finish_graph_info(&MLNamedOperands::from([("result", input)]))
            .is_err()
    );
}

#[test]
fn arbitrary_public_names_compile_and_bind_without_changing_the_graph() {
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    let names = [
        "input 1",
        "入口",
        "input-name",
        "input.name",
        "1input",
        "fp32",
        "return",
        "state",
        "any",
        "bf16",
        "__rustnn_operand_1",
        "output 1",
    ];
    for name in names {
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            let mut recorded = named_graph(DataType::Float32, vec![Dimension::Static(4)], false);
            recorded.operands[0].name = Some(name.into());
            recorded.operands[1].name = Some(name.into());
            recorded.operands[2].name = Some("result 2".into());
            recorded.operands[3].name = Some("结果".into());
            recorded.operations[2] = Operation::Neg {
                input: 0,
                options: None,
                outputs: vec![3],
            };
            GraphValidator::new(&recorded, ContextProperties::default())
                .validate()
                .unwrap();
            let converted = CoremlMlProgramConverter.convert(&recorded).unwrap();
            let model = specification::Model::decode(converted.data.as_slice()).unwrap();
            let description = model.description.unwrap();
            assert_eq!(
                description.output.len(),
                2,
                "unequal results must not coalesce"
            );
            for feature in description.input.iter().chain(&description.output) {
                assert!(
                    feature
                        .name
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
                );
                assert!(!feature.name.as_bytes()[0].is_ascii_digit());
            }
            let mut context = MLContext::create(
                &MLContextOptions::new(MLPowerPreference::Default, policy != DeviceType::Cpu)
                    .with_rustnn_device_hint(BackendDevice::Coreml {
                        device_type: policy,
                    }),
            )
            .unwrap();
            let mut graph = context.rustnn_build_graph(recorded.clone()).unwrap();
            let descriptor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4])
                .to_readable()
                .to_writable();
            let input = context.create_tensor(&descriptor).unwrap();
            let same = context.create_tensor(&descriptor).unwrap();
            let copy = context.create_tensor(&descriptor).unwrap();
            let negative = context.create_tensor(&descriptor).unwrap();
            let expected = [1.0003f32, -4.0, 0.0, -0.0];
            context.write_tensor(&input, &expected).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([(name, &input)]),
                    &MLNamedTensors::from([
                        (name, &same),
                        ("result 2", &copy),
                        ("结果", &negative),
                    ]),
                )
                .unwrap();
            for output in [same, copy] {
                let mut actual = [0f32; 4];
                context.read_tensor(&output, &mut actual).unwrap();
                assert_eq!(
                    actual.map(f32::to_bits),
                    expected.map(f32::to_bits),
                    "{name} {policy:?}"
                );
            }
            let mut actual = [0f32; 4];
            context.read_tensor(&negative, &mut actual).unwrap();
            assert_eq!(
                actual.map(f32::to_bits),
                expected.map(|value| (-value).to_bits()),
                "{name} {policy:?}"
            );
            assert_eq!(recorded.operands[0].name.as_deref(), Some(name));
            assert_eq!(recorded.operands[1].name.as_deref(), Some(name));
        }
    }
}

#[test]
fn scalar_static_and_resized_outputs_use_the_logical_namespace() {
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    let export =
        std::env::var_os("RUSTNN_COMPUTED_OUTPUT_NAMESPACE_DIR").map(std::path::PathBuf::from);
    let mut exported = Vec::new();
    for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
        for (dtype, half_output, computed_source) in [
            (DataType::Float32, false, false),
            (DataType::Float16, false, false),
            (DataType::Int32, false, false),
            (DataType::Float32, true, false),
            (DataType::Float32, false, true),
        ] {
            let shapes = vec![
                (vec![], vec![vec![]]),
                (vec![Dimension::Static(11)], vec![vec![11]]),
            ];
            #[cfg(feature = "dynamic-inputs")]
            let shapes = {
                let mut shapes = shapes;
                shapes.push((
                    vec![Dimension::Dynamic(DynamicDimension {
                        name: "length".into(),
                        max_size: 11,
                    })],
                    vec![vec![1], vec![11], vec![3]],
                ));
                shapes
            };
            for (shape, actual_shapes) in shapes {
                let mut context = MLContext::create(
                    &MLContextOptions::new(MLPowerPreference::Default, policy != DeviceType::Cpu)
                        .with_rustnn_device_hint(BackendDevice::Coreml {
                            device_type: policy,
                        }),
                )
                .unwrap();
                let label = if shape.is_empty() {
                    "scalar"
                } else if shape
                    .iter()
                    .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
                {
                    "dynamic"
                } else {
                    "static"
                };
                let mut graph = named_graph(dtype, shape, half_output);
                let mut export_bindings = None;
                if computed_source {
                    let mut intermediate = graph.operands[0].clone();
                    intermediate.name = Some("intermediate".into());
                    intermediate.kind = OperandKind::Intermediate;
                    graph.operands.push(intermediate);
                    graph.operations[0] = Operation::Cast {
                        input: 4,
                        data_type: MLOperandDataType::Float32,
                        options: None,
                        outputs: vec![1],
                    };
                    graph.operations[1] = Operation::Identity {
                        input: 4,
                        options: None,
                        outputs: vec![2],
                    };
                    graph.operations.insert(
                        0,
                        Operation::Neg {
                            input: 0,
                            options: None,
                            outputs: vec![4],
                        },
                    );
                    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
                    let model = specification::Model::decode(converted.data.as_slice()).unwrap();
                    let metadata = model.description.unwrap().metadata.unwrap();
                    assert!(
                        !metadata
                            .user_defined
                            .contains_key("rustnn.webnn.output_passthroughs")
                    );
                    if let Some(directory) = &export
                        && policy == DeviceType::Cpu
                    {
                        std::fs::create_dir_all(directory).unwrap();
                        std::fs::write(
                            directory.join(format!("computed_copy_{label}.mlmodel")),
                            converted.data,
                        )
                        .unwrap();
                        std::fs::write(
                            directory.join(format!("computed_copy_{label}.graph.json")),
                            serde_json::to_vec_pretty(&graph).unwrap(),
                        )
                        .unwrap();
                        export_bindings = Some(
                            serde_json::from_str::<std::collections::HashMap<String, String>>(
                                &metadata.user_defined["rustnn.webnn.output_aliases"],
                            )
                            .unwrap(),
                        );
                    }
                }
                GraphValidator::new(&graph, ContextProperties::default())
                    .validate()
                    .unwrap();
                let mut graph = MLGraphBuilder::new(&mut context)
                    .unwrap()
                    .build_graph_info(graph)
                    .unwrap();
                for actual_shape in actual_shapes {
                    let count = actual_shape.iter().product::<u64>().max(1) as usize;
                    let source = if computed_source {
                        [1.0003f32, -4., 0., -0.]
                            .into_iter()
                            .cycle()
                            .take(count)
                            .flat_map(f32::to_le_bytes)
                            .collect::<Vec<_>>()
                    } else {
                        data(dtype, count)
                    };
                    let expected = if computed_source {
                        source
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .flat_map(|bytes| (-f32::from_le_bytes(*bytes)).to_le_bytes())
                            .collect::<Vec<_>>()
                    } else {
                        source.clone()
                    };
                    if let (Some(directory), Some(bindings)) = (&export, &export_bindings) {
                        let name = format!("computed_copy_{label}_{count}");
                        let input_file = format!("{name}_input.bin");
                        let expected_file = format!("{name}_expected.bin");
                        std::fs::write(directory.join(&input_file), &source).unwrap();
                        std::fs::write(directory.join(&expected_file), &expected).unwrap();
                        let outputs: Vec<_> = ["shared", "copy", "forwarded"].iter().map(|&name| json!({"name":bindings[name],"logical_name":name,"dtype":"float32","shape":actual_shape,"expected":expected_file,"comparison":"bits"})).collect();
                        exported.push(json!({"name":name,"group":"computed-output-copy","model":format!("computed_copy_{label}.mlmodel"),"inputs":[{"name":"shared","dtype":"float32","shape":actual_shape,"data":input_file}],"outputs":outputs}));
                    }
                    let input = context
                        .create_tensor(
                            &MLTensorDescriptor::new(api_dtype(dtype), actual_shape.clone())
                                .to_writable(),
                        )
                        .unwrap();
                    context.write_tensor(&input, &source).unwrap();
                    let output_dtype = if half_output {
                        DataType::Float16
                    } else {
                        dtype
                    };
                    let shared = context
                        .create_tensor(
                            &MLTensorDescriptor::new(api_dtype(output_dtype), actual_shape.clone())
                                .to_readable(),
                        )
                        .unwrap();
                    let copy = context
                        .create_tensor(
                            &MLTensorDescriptor::new(api_dtype(dtype), actual_shape.clone())
                                .to_readable(),
                        )
                        .unwrap();
                    let forwarded = context
                        .create_tensor(
                            &MLTensorDescriptor::new(api_dtype(output_dtype), actual_shape)
                                .to_readable(),
                        )
                        .unwrap();
                    context
                        .dispatch(
                            &mut graph,
                            &MLNamedTensors::from([("shared", &input)]),
                            &MLNamedTensors::from([
                                ("shared", &shared),
                                ("copy", &copy),
                                ("forwarded", &forwarded),
                            ]),
                        )
                        .unwrap();
                    let alias_expected = if half_output {
                        half_bytes(&expected)
                    } else {
                        expected.clone()
                    };
                    for (output, expected) in [
                        (shared, &alias_expected),
                        (copy, &expected),
                        (forwarded, &alias_expected),
                    ] {
                        let mut actual = vec![0u8; expected.len()];
                        context.read_tensor(&output, &mut actual).unwrap();
                        assert_eq!(
                            &actual, expected,
                            "{policy:?} {dtype:?} half_output={half_output}"
                        );
                    }
                }
            }
        }
    }
    if let Some(directory) = &export {
        std::fs::write(
            directory.join("manifest.json"),
            serde_json::to_vec_pretty(&json!({"cases":exported})).unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn export_output_namespace_models_when_requested() {
    let Some(directory) = std::env::var_os("RUSTNN_OUTPUT_NAMESPACE_DIR") else {
        return;
    };
    let directory = std::path::Path::new(&directory);
    std::fs::create_dir_all(directory).unwrap();
    let mut cases = Vec::new();
    for (dtype, half_output, tag) in [
        (DataType::Float32, false, "float32"),
        (DataType::Float16, false, "float16"),
        (DataType::Int32, false, "int32"),
        (DataType::Float32, true, "float32_to_half"),
    ] {
        for (shape, actual_shapes, label) in [
            (vec![], vec![vec![]], "scalar"),
            (vec![Dimension::Static(11)], vec![vec![11]], "static"),
            (
                vec![Dimension::Dynamic(DynamicDimension {
                    name: "length".into(),
                    max_size: 11,
                })],
                vec![vec![1], vec![11], vec![3]],
                "dynamic",
            ),
        ] {
            let graph = named_graph(dtype, shape, half_output);
            GraphValidator::new(&graph, ContextProperties::default())
                .validate()
                .unwrap();
            let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
            let model = specification::Model::decode(converted.data.as_slice()).unwrap();
            let description = model.description.unwrap();
            let bindings: std::collections::HashMap<String, String> = serde_json::from_str(
                &description.metadata.as_ref().unwrap().user_defined["rustnn.webnn.output_aliases"],
            )
            .unwrap();
            let model_name = format!("alias_{tag}_{label}.mlmodel");
            std::fs::write(directory.join(&model_name), converted.data).unwrap();
            std::fs::write(
                directory.join(format!("alias_{tag}_{label}.graph.json")),
                serde_json::to_vec_pretty(&graph).unwrap(),
            )
            .unwrap();
            for actual_shape in actual_shapes {
                let count = actual_shape.iter().product::<u32>().max(1) as usize;
                let name = format!("alias_{tag}_{label}_{count}");
                let input = format!("{name}_input.bin");
                let expected = format!("{name}_expected.bin");
                let alias_expected = format!("{name}_alias_expected.bin");
                let bytes = data(dtype, count);
                std::fs::write(directory.join(&input), &bytes).unwrap();
                std::fs::write(directory.join(&expected), &bytes).unwrap();
                std::fs::write(
                    directory.join(&alias_expected),
                    if half_output {
                        half_bytes(&bytes)
                    } else {
                        bytes
                    },
                )
                .unwrap();
                let native_dtype = match dtype {
                    DataType::Float32 => "float32",
                    DataType::Float16 => "float16",
                    DataType::Int32 => "int32",
                    _ => unreachable!(),
                };
                let output_dtype = if half_output { "float16" } else { native_dtype };
                cases.push(json!({"name":name,"group":"output-namespace-coalesced","model":model_name,"inputs":[{"name":"shared","dtype":native_dtype,"shape":actual_shape,"data":input}],"outputs":[{"name":bindings["shared"],"logical_name":"shared","dtype":output_dtype,"shape":actual_shape,"expected":alias_expected,"comparison":"bits"},{"name":bindings["copy"],"logical_name":"copy","dtype":native_dtype,"shape":actual_shape,"expected":expected,"comparison":"bits"},{"name":bindings["forwarded"],"logical_name":"forwarded","dtype":output_dtype,"shape":actual_shape,"expected":alias_expected,"comparison":"bits"}]}));
            }
        }
    }
    for (index, logical) in [
        "input 1",
        "入口",
        "input-name",
        "input.name",
        "1input",
        "fp32",
        "return",
        "state",
        "any",
        "bf16",
        "__rustnn_operand_1",
        "output 1",
    ]
    .into_iter()
    .enumerate()
    {
        let mut graph = named_graph(DataType::Float32, vec![Dimension::Static(4)], false);
        graph.operands[0].name = Some(logical.into());
        graph.operands[1].name = Some(logical.into());
        graph.operands[2].name = Some("result 2".into());
        graph.operands[3].name = Some("结果".into());
        graph.operations[2] = Operation::Neg {
            input: 0,
            options: None,
            outputs: vec![3],
        };
        GraphValidator::new(&graph, ContextProperties::default())
            .validate()
            .unwrap();
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let model = specification::Model::decode(converted.data.as_slice()).unwrap();
        let description = model.description.unwrap();
        let outputs: std::collections::HashMap<String, String> = serde_json::from_str(
            &description.metadata.as_ref().unwrap().user_defined["rustnn.webnn.output_aliases"],
        )
        .unwrap();
        let name = format!("arbitrary_name_{index}");
        let model_name = format!("{name}.mlmodel");
        let input_file = format!("{name}_input.bin");
        let negative_file = format!("{name}_negative.bin");
        let values = [1.0003f32, -4.0, 0.0, -0.0];
        std::fs::write(directory.join(&model_name), converted.data).unwrap();
        std::fs::write(
            directory.join(&input_file),
            values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        std::fs::write(
            directory.join(&negative_file),
            values
                .into_iter()
                .flat_map(|value| (-value).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        std::fs::write(
            directory.join(format!("{name}.graph.json")),
            serde_json::to_vec_pretty(&graph).unwrap(),
        )
        .unwrap();
        cases.push(json!({"name":name,"group":"arbitrary-public-name-bindings","model":model_name,
            "inputs":[{"name":description.input[0].name,"logical_name":logical,"dtype":"float32","shape":[4],"data":input_file}],
            "outputs":[{"name":outputs[logical],"logical_name":logical,"dtype":"float32","shape":[4],"expected":input_file,"comparison":"bits"},
                {"name":outputs["result 2"],"logical_name":"result 2","dtype":"float32","shape":[4],"expected":input_file,"comparison":"bits"},
                {"name":outputs["结果"],"logical_name":"结果","dtype":"float32","shape":[4],"expected":negative_file,"comparison":"bits"}]}));
    }
    std::fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
    )
    .unwrap();
}

//! CoreML must not impose MIL identifier rules on WebNN tensor names.

use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{DataType, GraphInfo, Operand, OperandDescriptor, OperandKind};
use rustnn::operators::Operation;

const NAMES: &[(&str, &str)] = &[
    ("state", "tensor"),
    ("fp32", "program"),
    ("0cache", "dict"),
    ("cache.key", "next.key"),
    ("cache-key", "next-key"),
    ("缓存", "结果"),
    ("rustnn_escaped_7374617465", "rustnn_escaped_74656e736f72"),
    ("state_workaround", "tensor_workaround"),
];

fn named_graph() -> GraphInfo {
    let mut graph = GraphInfo::default();
    for &(input, output) in NAMES {
        let id = graph.operands.len() as u32;
        for (kind, name) in [(OperandKind::Input, input), (OperandKind::Output, output)] {
            graph.operands.push(Operand {
                kind,
                name: Some(name.into()),
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![rustnn::graph::Dimension::Static(2)],
                    pending_permutation: vec![],
                },
            });
        }
        graph.input_operands.push(id);
        graph.output_operands.push(id + 1);
        graph.operations.push(Operation::Neg {
            input: id,
            options: None,
            outputs: vec![id + 1],
        });
    }
    graph
}

#[test]
fn coreml_identifiers_are_valid_unique_and_preserve_graph_names() {
    let graph = named_graph();
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let model =
        rustnn::protos::coreml::specification::Model::decode(converted.data.as_slice()).unwrap();
    let description = model.description.unwrap();
    let mut names = std::collections::HashSet::new();
    for feature in description.input.iter().chain(&description.output) {
        let name = &feature.name;
        assert!(
            name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_'),
            "{name}"
        );
        assert!(
            name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "{name}"
        );
        assert!(
            !["state", "tensor", "fp32", "program", "dict"].contains(&name.as_str()),
            "reserved: {name}"
        );
        assert!(names.insert(name), "duplicate: {name}");
    }
    for (index, &(input, output)) in NAMES.iter().enumerate() {
        assert_eq!(graph.operands[index * 2].name.as_deref(), Some(input));
        assert_eq!(graph.operands[index * 2 + 1].name.as_deref(), Some(output));
    }
}

#[test]
fn coreml_identifiers_are_deterministic_for_large_duplicate_name_graphs() {
    const COUNT: usize = 2_048;
    let mut graph = GraphInfo::default();
    for id in 0..COUNT {
        graph.operands.push(Operand {
            kind: if id == 0 {
                OperandKind::Input
            } else {
                OperandKind::Output
            },
            name: Some(match id {
                2 => "__rustnn_operand_1".into(),
                3 => "__rustnn_operand_1_1".into(),
                id if id == COUNT - 1 => "result".into(),
                _ => "value".into(),
            }),
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: vec![rustnn::graph::Dimension::Static(2)],
                pending_permutation: vec![],
            },
        });
        if id != 0 {
            graph.operations.push(Operation::Neg {
                input: id as u32 - 1,
                options: None,
                outputs: vec![id as u32],
            });
        }
    }
    graph.input_operands = vec![0];
    graph.output_operands = vec![COUNT as u32 - 1];
    let names = || {
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let model = rustnn::protos::coreml::specification::Model::decode(converted.data.as_slice())
            .unwrap();
        let rustnn::protos::coreml::specification::model::Type::MlProgram(program) =
            model.r#type.unwrap()
        else {
            panic!("expected MLProgram")
        };
        let function = &program.functions["main"];
        function
            .inputs
            .iter()
            .chain(
                function.block_specializations["CoreML7"]
                    .operations
                    .iter()
                    .flat_map(|op| &op.outputs),
            )
            .map(|value| value.name.clone())
            .collect::<Vec<_>>()
    };
    let first = names();
    assert_eq!(first, names());
    assert_eq!(first.len(), COUNT);
    assert_eq!(
        first.iter().collect::<std::collections::HashSet<_>>().len(),
        COUNT
    );
    assert_eq!(
        &first[..4],
        &[
            "value",
            "__rustnn_operand_1_2",
            "__rustnn_operand_1",
            "__rustnn_operand_1_1"
        ]
    );
    assert_eq!(graph.operands[1].name.as_deref(), Some("value"));
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn coreml_identifiers_dispatch_using_original_names() {
    use rustnn::mlcontext::{
        Backend, MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::mlgraphbuilder::MLGraphBuilder;
    use rustnn::operator_enums::MLOperandDataType;

    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml),
    )
    .unwrap();
    let mut graph = MLGraphBuilder::new(&mut context)
        .unwrap()
        .build_graph_info(named_graph())
        .unwrap();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for index in 0..NAMES.len() {
        let input = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2]).to_writable(),
            )
            .unwrap();
        let output = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2]).to_readable(),
            )
            .unwrap();
        let values = [index as f32 + 1.0, -(index as f32 + 2.0)];
        context.write_tensor(&input, &values).unwrap();
        inputs.push(input);
        outputs.push(output);
    }
    let named_inputs: MLNamedTensors = NAMES
        .iter()
        .zip(&inputs)
        .map(|(&(name, _), tensor)| (name, tensor))
        .collect();
    let named_outputs: MLNamedTensors = NAMES
        .iter()
        .zip(&outputs)
        .map(|(&(_, name), tensor)| (name, tensor))
        .collect();
    context
        .dispatch(&mut graph, &named_inputs, &named_outputs)
        .unwrap();
    for (index, output) in outputs.iter().enumerate() {
        let mut bytes = [0u8; 8];
        context.read_tensor(output, &mut bytes).unwrap();
        let actual: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&v| f32::from_le_bytes(v))
            .collect();
        assert_eq!(
            actual,
            vec![-(index as f32 + 1.0), index as f32 + 2.0],
            "{}",
            NAMES[index].1
        );
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn coreml_identifiers_one_shot_and_checked_execution_use_original_names() {
    use rustnn::executors::coreml::{
        CoremlInput, run_coreml_with_inputs_checked, run_coreml_with_inputs_with_weights,
        run_coreml_zeroed_cached_with_weights,
    };
    let graph = named_graph();
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let input_descriptors = graph
        .input_operands
        .iter()
        .map(|&id| {
            let operand = &graph.operands[id as usize];
            (operand.name.clone().unwrap(), operand.descriptor.clone())
        })
        .collect();
    let output_descriptors = graph
        .output_operands
        .iter()
        .map(|&id| {
            let operand = &graph.operands[id as usize];
            (operand.name.clone().unwrap(), operand.descriptor.clone())
        })
        .collect();
    let inputs: Vec<_> = NAMES
        .iter()
        .enumerate()
        .map(|(index, &(name, _))| CoremlInput {
            name: name.into(),
            shape: vec![2],
            data: vec![index as f32 + 1.0, -(index as f32 + 2.0)],
        })
        .collect();
    let runs = [
        run_coreml_with_inputs_with_weights(
            &converted.data,
            converted.weights_data.as_deref(),
            inputs.clone(),
        )
        .unwrap(),
        run_coreml_with_inputs_checked(
            &converted.data,
            inputs,
            &input_descriptors,
            &output_descriptors,
        )
        .unwrap(),
    ];
    for attempts in runs {
        let outputs = attempts
            .iter()
            .find(|attempt| attempt.compute_unit == "CPU_ONLY")
            .unwrap()
            .result
            .as_ref()
            .unwrap();
        assert_eq!(outputs.len(), NAMES.len());
        for (index, &(_, name)) in NAMES.iter().enumerate() {
            let output = outputs.iter().find(|output| output.name == name).unwrap();
            assert_eq!(output.shape, vec![2]);
            assert_eq!(output.data, vec![-(index as f32 + 1.0), index as f32 + 2.0]);
        }
    }
    let attempts = run_coreml_zeroed_cached_with_weights(
        &converted.data,
        converted.weights_data.as_deref(),
        &input_descriptors,
        None,
    )
    .unwrap();
    let outputs = attempts
        .iter()
        .find(|attempt| attempt.compute_unit == "CPU_ONLY")
        .unwrap()
        .result
        .as_ref()
        .unwrap();
    assert_eq!(outputs.len(), NAMES.len());
    for &(_, name) in NAMES {
        let output = outputs.iter().find(|output| output.name == name).unwrap();
        assert_eq!(output.data, vec![0.0, 0.0]);
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn coreml_identifiers_leave_legacy_escape_prefix_names_literal() {
    use rustnn::executors::coreml::{
        CoremlInput, run_coreml_with_inputs_checked, run_coreml_with_inputs_with_weights,
        run_coreml_zeroed_cached_with_weights,
    };
    use rustnn::protos::coreml::{mil_spec::argument::binding::Binding, specification};
    use std::collections::HashMap;

    const INPUT: &str = "rustnn_escaped_7374617465";
    const OUTPUT: &str = "rustnn_escaped_74656e736f72";

    // Synthesize a legacy/third-party model: its literal names resemble our
    // encoding, but it does not opt into the reversible naming contract.
    let mut graph = named_graph();
    graph.operands.truncate(2);
    graph.operations.truncate(1);
    graph.input_operands.truncate(1);
    graph.output_operands.truncate(1);
    graph.operands[0].name = Some("legacy_input".into());
    graph.operands[1].name = Some("legacy_output".into());
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let mut model = specification::Model::decode(converted.data.as_slice()).unwrap();
    let rename = |name: &mut String| match name.as_str() {
        "legacy_input" => *name = INPUT.into(),
        "legacy_output" => *name = OUTPUT.into(),
        _ => {}
    };
    let description = model.description.as_mut().unwrap();
    description.metadata = None;
    for feature in description.input.iter_mut().chain(&mut description.output) {
        rename(&mut feature.name);
    }
    let Some(specification::model::Type::MlProgram(program)) = &mut model.r#type else {
        panic!("expected MLProgram");
    };
    for function in program.functions.values_mut() {
        for input in &mut function.inputs {
            rename(&mut input.name);
        }
        for block in function.block_specializations.values_mut() {
            for output in &mut block.outputs {
                rename(output);
            }
            for operation in &mut block.operations {
                assert!(operation.blocks.is_empty(), "fixture has no nested blocks");
                for argument in operation.inputs.values_mut() {
                    for binding in &mut argument.arguments {
                        if let Some(Binding::Name(name)) = &mut binding.binding {
                            rename(name);
                        }
                    }
                }
                for output in &mut operation.outputs {
                    rename(&mut output.name);
                }
            }
        }
    }
    let bytes = model.encode_to_vec();
    let inputs = vec![CoremlInput {
        name: INPUT.into(),
        shape: vec![2],
        data: vec![2., -3.],
    }];
    let input_descriptors = HashMap::from([(INPUT.into(), graph.operands[0].descriptor.clone())]);
    let output_descriptors = HashMap::from([(OUTPUT.into(), graph.operands[1].descriptor.clone())]);
    let runs = [
        (
            run_coreml_with_inputs_with_weights(&bytes, None, inputs.clone()).unwrap(),
            vec![-2., 3.],
        ),
        (
            run_coreml_with_inputs_checked(&bytes, inputs, &input_descriptors, &output_descriptors)
                .unwrap(),
            vec![-2., 3.],
        ),
        (
            run_coreml_zeroed_cached_with_weights(&bytes, None, &input_descriptors, None).unwrap(),
            vec![0., 0.],
        ),
    ];
    for (attempts, expected) in runs {
        let outputs = attempts
            .iter()
            .find(|attempt| attempt.compute_unit == "CPU_ONLY")
            .unwrap()
            .result
            .as_ref()
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].name, OUTPUT);
        assert_eq!(outputs[0].shape, [2]);
        assert_eq!(outputs[0].data, expected);
    }
}

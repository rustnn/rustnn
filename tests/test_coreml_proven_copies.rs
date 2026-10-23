//! Exact copies are selected by public graph provenance, not native output values.

use std::collections::HashMap;

use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{
    ConstantData, DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::{MLDimension, MLSliceOptions, MLTransposeOptions};
use rustnn::operators::Operation;
use rustnn::protos::coreml::specification;
use serde_json::Value;

const INPUT_COPIES: &str = "rustnn.webnn.output_passthroughs";
const CONSTANT_COPIES: &str = "rustnn.webnn.output_constant_copies";

#[derive(Clone, Copy)]
enum CopyOp {
    Identity,
    Cast,
    Transpose,
    Slice,
    Reshape,
}

fn descriptor() -> OperandDescriptor {
    OperandDescriptor {
        data_type: DataType::Int32,
        shape: vec![Dimension::Static(6)],
        pending_permutation: vec![],
    }
}

fn copy_operation(kind: CopyOp, input: u32, output: u32) -> Operation {
    match kind {
        CopyOp::Identity => Operation::Identity {
            input,
            options: None,
            outputs: vec![output],
        },
        CopyOp::Cast => Operation::Cast {
            input,
            data_type: MLOperandDataType::Int32,
            options: None,
            outputs: vec![output],
        },
        CopyOp::Transpose => Operation::Transpose {
            input,
            options: None,
            outputs: vec![output],
        },
        CopyOp::Slice => Operation::Slice {
            input,
            starts: vec![0],
            sizes: vec![MLDimension::Static(6)],
            options: None,
            outputs: vec![output],
        },
        CopyOp::Reshape => Operation::Reshape {
            input,
            new_shape: vec![MLDimension::Static(6)],
            options: None,
            outputs: vec![output],
        },
    }
}

fn exact_bytes() -> Vec<u8> {
    [0i32, -1, 16_777_217, -16_777_217, i32::MIN, i32::MAX]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect()
}

fn copy_graph(constant: bool, operations: &[CopyOp]) -> GraphInfo {
    let source = Operand {
        kind: if constant {
            OperandKind::Constant
        } else {
            OperandKind::Input
        },
        descriptor: descriptor(),
        name: Some("source".into()),
    };
    let mut graph = GraphInfo {
        operands: vec![source],
        input_operands: if constant { vec![] } else { vec![0] },
        ..Default::default()
    };
    if constant {
        graph.constant_operand_ids_to_handles.insert(
            0,
            ConstantData {
                data: exact_bytes(),
                label: None,
            },
        );
    }
    for (index, &kind) in operations.iter().enumerate() {
        let output = (index + 1) as u32;
        graph.operands.push(Operand {
            kind: OperandKind::Output,
            descriptor: descriptor(),
            name: Some(format!("copy{output}")),
        });
        graph
            .operations
            .push(copy_operation(kind, output - 1, output));
        graph.output_operands.push(output);
    }
    graph
}

fn metadata(graph: &GraphInfo) -> HashMap<String, String> {
    let converted = CoremlMlProgramConverter.convert(graph).unwrap();
    let model = specification::Model::decode(converted.data.as_slice()).unwrap();
    let specification::model::Type::MlProgram(program) = model.r#type.unwrap() else {
        panic!("expected MLProgram")
    };
    assert!(
        !program.functions["main"].block_specializations["CoreML7"]
            .operations
            .is_empty()
    );
    model.description.unwrap().metadata.unwrap().user_defined
}

#[test]
fn input_no_op_chains_keep_the_original_binding_proof() {
    let graph = copy_graph(
        false,
        &[
            CopyOp::Transpose,
            CopyOp::Slice,
            CopyOp::Reshape,
            CopyOp::Identity,
            CopyOp::Cast,
        ],
    );
    let metadata = metadata(&graph);
    let proofs: Value = serde_json::from_str(&metadata[INPUT_COPIES]).unwrap();
    assert_eq!(proofs.as_object().unwrap().len(), 5);
    for id in 1..=5 {
        assert_eq!(proofs[format!("copy{id}")]["input"], "source");
    }
    assert!(!metadata.contains_key(CONSTANT_COPIES));
}

#[test]
fn constant_no_op_fanout_serializes_one_original_exact_payload() {
    let graph = copy_graph(
        true,
        &[
            CopyOp::Identity,
            CopyOp::Transpose,
            CopyOp::Slice,
            CopyOp::Reshape,
            CopyOp::Cast,
        ],
    );
    let before = serde_json::to_vec(&graph).unwrap();
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let model = specification::Model::decode(converted.data.as_slice()).unwrap();
    let metadata = model.description.unwrap().metadata.unwrap().user_defined;
    let copies: Value = serde_json::from_str(&metadata[CONSTANT_COPIES]).unwrap();
    assert_eq!(copies["version"], 2);
    assert_eq!(copies["sources"].as_object().unwrap().len(), 1);
    assert_eq!(copies["outputs"].as_object().unwrap().len(), 5);
    assert_eq!(copies["sources"]["0"]["descriptor"]["data_type"], "int32");
    let offset = copies["sources"]["0"]["offset"].as_u64().unwrap() as usize;
    let weights = converted.weights_data.unwrap();
    assert_eq!(
        &weights[offset + 64..offset + 64 + exact_bytes().len()],
        exact_bytes()
    );
    assert!(!metadata.contains_key(INPUT_COPIES));
    assert_eq!(serde_json::to_vec(&graph).unwrap(), before);
}

#[test]
fn multi_megabyte_constant_fanout_reuses_the_existing_mil_weight_record() {
    for dtype in [DataType::Float32, DataType::Float16] {
        let mut graph = copy_graph(true, &[CopyOp::Identity, CopyOp::Identity]);
        let length = 5 * 1024 * 1024;
        let width = if dtype == DataType::Float32 { 4 } else { 2 };
        for operand in &mut graph.operands {
            operand.descriptor.data_type = dtype;
            operand.descriptor.shape = vec![Dimension::Static((length / width) as u32)];
        }
        let source = (0..length)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        graph
            .constant_operand_ids_to_handles
            .get_mut(&0)
            .unwrap()
            .data = source.clone();
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        drop(graph);
        let model = specification::Model::decode(converted.data.as_slice()).unwrap();
        let json = &model
            .description
            .as_ref()
            .unwrap()
            .metadata
            .as_ref()
            .unwrap()
            .user_defined[CONSTANT_COPIES];
        assert!(json.len() < 1024, "metadata must not contain tensor bytes");
        assert!(
            converted.data.len() < 8192,
            "the model must not duplicate the weight"
        );
        let proof: Value = serde_json::from_str(json).unwrap();
        let offset = proof["sources"]["0"]["offset"].as_u64().unwrap() as usize;
        let weights = converted.weights_data.unwrap();
        assert_eq!(weights.len(), source.len() + 128);
        assert_eq!(&weights[offset + 64..offset + 64 + source.len()], source);
        let specification::model::Type::MlProgram(program) = model.r#type.unwrap() else {
            panic!("expected MLProgram")
        };
        let mil_source = program.functions["main"].block_specializations["CoreML7"]
            .operations
            .iter()
            .find(|operation| operation.r#type == "const" && operation.outputs[0].name == "source")
            .unwrap();
        let Some(rustnn::protos::coreml::mil_spec::value::Value::BlobFileValue(blob)) =
            &mil_source.attributes["val"].value
        else {
            panic!("source must be blob-backed")
        };
        assert_eq!(blob.offset, offset as u64);
    }
}

#[test]
fn equal_shape_does_not_prove_a_transpose_or_strided_slice_is_a_copy() {
    let mut graph = copy_graph(false, &[CopyOp::Transpose]);
    for operand in &mut graph.operands {
        operand.descriptor.shape = vec![Dimension::Static(2), Dimension::Static(2)];
    }
    graph.operations[0] = Operation::Transpose {
        input: 0,
        options: Some(MLTransposeOptions {
            permutation: vec![1, 0],
            ..Default::default()
        }),
        outputs: vec![1],
    };
    assert!(!metadata(&graph).contains_key(INPUT_COPIES));

    let mut graph = copy_graph(false, &[CopyOp::Slice]);
    graph.operands[0].descriptor.shape = vec![Dimension::Static(12)];
    graph.operations[0] = Operation::Slice {
        input: 0,
        starts: vec![0],
        sizes: vec![MLDimension::Static(12)],
        options: Some(MLSliceOptions {
            strides: vec![2],
            ..Default::default()
        }),
        outputs: vec![1],
    };
    assert!(!metadata(&graph).contains_key(INPUT_COPIES));
}

#[test]
fn arithmetic_intermediates_are_not_original_source_copies() {
    let mut graph = copy_graph(false, &[CopyOp::Identity, CopyOp::Reshape]);
    graph.operations[0] = Operation::Neg {
        input: 0,
        options: None,
        outputs: vec![1],
    };
    assert!(!metadata(&graph).contains_key(INPUT_COPIES));
    assert!(!metadata(&graph).contains_key(CONSTANT_COPIES));
}

#[test]
fn copy_proofs_reject_operations_before_their_inputs() {
    let mut graph = copy_graph(false, &[CopyOp::Identity, CopyOp::Identity]);
    graph.operations.swap(0, 1);
    assert!(CoremlMlProgramConverter.convert(&graph).is_err());
}

#[test]
fn copy_proofs_reject_redefined_outputs() {
    let mut graph = copy_graph(false, &[CopyOp::Identity]);
    graph.operations.push(graph.operations[0].clone());
    assert!(CoremlMlProgramConverter.convert(&graph).is_err());
}

#[test]
fn copy_proofs_require_optional_inputs_to_be_ready() {
    let mut graph = copy_graph(false, &[CopyOp::Identity, CopyOp::Identity]);
    graph.operations[0] = Operation::LayerNormalization {
        input: 0,
        options: Some(rustnn::operator_options::MLLayerNormalizationOptions {
            scale: Some(2),
            ..Default::default()
        }),
        outputs: vec![1],
    };
    assert!(matches!(
        CoremlMlProgramConverter.convert(&graph),
        Err(rustnn::error::GraphError::OperandNotReady { operand: 2, .. })
    ));
}

#[test]
fn copy_proofs_reject_cycles_and_invalid_references() {
    let mut graph = copy_graph(false, &[CopyOp::Identity, CopyOp::Identity]);
    graph.operations[0] = copy_operation(CopyOp::Identity, 2, 1);
    assert!(matches!(
        CoremlMlProgramConverter.convert(&graph),
        Err(rustnn::error::GraphError::OperandNotReady { operand: 2, .. })
    ));
    graph.operations[0] = copy_operation(CopyOp::Identity, 99, 1);
    assert!(matches!(
        CoremlMlProgramConverter.convert(&graph),
        Err(rustnn::error::GraphError::InvalidOperandReference { operand: 99, .. })
    ));
    graph.operations[0] = copy_operation(CopyOp::Identity, 0, 99);
    assert!(matches!(
        CoremlMlProgramConverter.convert(&graph),
        Err(rustnn::error::GraphError::InvalidOperandReference { operand: 99, .. })
    ));
    graph.operations[0] = copy_operation(CopyOp::Identity, 0, 1);
    graph.output_operands.push(99);
    assert!(matches!(
        CoremlMlProgramConverter.convert(&graph),
        Err(rustnn::error::GraphError::InvalidConversionOperand { operand: 99 })
    ));
}

#[test]
fn different_dtype_constant_cast_keeps_native_conversion() {
    let mut graph = copy_graph(true, &[CopyOp::Cast]);
    graph.operands[0].descriptor.data_type = DataType::Float32;
    graph.operands[1].descriptor.data_type = DataType::Float16;
    graph.operations[0] = Operation::Cast {
        input: 0,
        data_type: MLOperandDataType::Float16,
        options: None,
        outputs: vec![1],
    };
    assert!(!metadata(&graph).contains_key(CONSTANT_COPIES));
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let model = specification::Model::decode(converted.data.as_slice()).unwrap();
    let specification::model::Type::MlProgram(program) = model.r#type.unwrap() else {
        panic!("expected MLProgram")
    };
    assert!(
        program.functions["main"].block_specializations["CoreML7"]
            .operations
            .iter()
            .any(|operation| operation.r#type == "cast")
    );
}

#[test]
fn constant_float32_no_op_cast_uses_the_constant_identity_lowering() {
    let mut graph = copy_graph(true, &[CopyOp::Cast]);
    for operand in &mut graph.operands {
        operand.descriptor.data_type = DataType::Float32;
    }
    graph.operations[0] = Operation::Cast {
        input: 0,
        data_type: MLOperandDataType::Float32,
        options: None,
        outputs: vec![1],
    };
    let lowering = |graph: &GraphInfo| {
        let converted = CoremlMlProgramConverter.convert(graph).unwrap();
        let model = specification::Model::decode(converted.data.as_slice()).unwrap();
        let specification::model::Type::MlProgram(program) = model.r#type.unwrap() else {
            panic!("expected MLProgram")
        };
        program.functions["main"].block_specializations["CoreML7"]
            .operations
            .clone()
    };
    let operations = lowering(&graph);
    graph.operations[0] = Operation::Identity {
        input: 0,
        options: None,
        outputs: vec![1],
    };
    // Precision lowering may use real_div instead of mul for floating-point
    // constant transport; a same-type Cast must inherit that exact lowering.
    assert_eq!(operations, lowering(&graph));
    assert!(
        !operations
            .iter()
            .any(|operation| operation.r#type == "identity")
    );
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
mod runtime {
    use super::*;
    use rustnn::mlcontext::{
        Backend, MLContext, MLContextOptions, MLGraphBuilder, MLNamedOperands, MLNamedTensors,
        MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor, RustNNOptions,
    };

    #[test]
    fn public_same_dtype_constant_cast_preserves_scalar_and_tensor_bytes() {
        for (reuse, backings) in [(false, false), (true, false), (true, true)] {
            for (dtype, values, width) in [
                (
                    MLOperandDataType::Float32,
                    [1u32, 0x80000000, 0x3f800001, 0x7fc12345]
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                        .collect::<Vec<_>>(),
                    4,
                ),
                (
                    MLOperandDataType::Float16,
                    [1u16, 0x8000, 0x3c01, 0x7e15]
                        .into_iter()
                        .flat_map(u16::to_le_bytes)
                        .collect(),
                    2,
                ),
                (
                    MLOperandDataType::Int32,
                    [16_777_217i32, -16_777_217, i32::MIN, i32::MAX]
                        .into_iter()
                        .flat_map(i32::to_le_bytes)
                        .collect(),
                    4,
                ),
            ] {
                for scalar in [false, true] {
                    let shape = if scalar { vec![] } else { vec![4] };
                    let expected = if scalar {
                        values[..width].to_vec()
                    } else {
                        values.clone()
                    };
                    let mut options = RustNNOptions::default();
                    options.coreml.reuse_tensor_storage = reuse;
                    options.coreml.output_backings = backings;
                    let mut context = MLContext::create(
                        &MLContextOptions::new(MLPowerPreference::Default, false)
                            .with_rustnn_backend_hint(Backend::Coreml)
                            .with_rustnn_options(options),
                    )
                    .unwrap();
                    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
                    let constant = builder
                        .constant_from_bytes(
                            &MLOperandDescriptor::new(dtype, shape.clone()),
                            expected.clone(),
                        )
                        .unwrap();
                    let cast = builder.cast(constant, dtype).unwrap();
                    let mut graph = builder
                        .build(&MLNamedOperands::from([("cast", cast)]))
                        .unwrap();
                    let output = context
                        .create_tensor(
                            &MLTensorDescriptor::new(dtype, shape)
                                .to_readable()
                                .to_writable(),
                        )
                        .unwrap();
                    for _ in 0..2 {
                        context
                            .dispatch(
                                &mut graph,
                                &MLNamedTensors::new(),
                                &MLNamedTensors::from([("cast", &output)]),
                            )
                            .unwrap();
                        let mut actual = vec![0; expected.len()];
                        context.read_tensor(&output, &mut actual).unwrap();
                        assert_eq!(
                            actual, expected,
                            "{dtype:?}, scalar={scalar}, reuse={reuse}, backings={backings}"
                        );
                        context
                            .write_tensor(&output, &vec![0u8; expected.len()])
                            .unwrap();
                    }
                }
            }
        }
    }

    #[test]
    fn public_constant_bytes_and_readable_outputs_preserve_float_payloads() {
        for (dtype, expected) in [
            (
                MLOperandDataType::Float32,
                [1u32, 0x80000000, 0x3f800001, 0x7fc12345]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
            (
                MLOperandDataType::Float16,
                [1u16, 0x8000, 0x3c01, 0x7e15]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
            ),
        ] {
            let mut context = MLContext::create(
                &MLContextOptions::new(MLPowerPreference::Default, false)
                    .with_rustnn_backend_hint(Backend::Coreml),
            )
            .unwrap();
            let mut builder = MLGraphBuilder::new(&mut context).unwrap();
            let constant = builder
                .constant_from_bytes(&MLOperandDescriptor::new(dtype, vec![4]), expected.clone())
                .unwrap();
            let identity = builder.identity(constant).unwrap();
            let sliced = builder
                .slice(constant, &[0], &[MLDimension::Static(4)])
                .unwrap();
            let reshaped = builder
                .reshape(constant, vec![MLDimension::Static(4)])
                .unwrap();
            let mut graph = builder
                .build(&MLNamedOperands::from([
                    ("identity", identity),
                    ("sliced", sliced),
                    ("reshaped", reshaped),
                ]))
                .unwrap();
            let descriptor = MLTensorDescriptor::new(dtype, vec![4])
                .to_readable()
                .to_writable();
            let identity = context.create_tensor(&descriptor).unwrap();
            let sliced = context.create_tensor(&descriptor).unwrap();
            let reshaped = context.create_tensor(&descriptor).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::new(),
                    &MLNamedTensors::from([
                        ("identity", &identity),
                        ("sliced", &sliced),
                        ("reshaped", &reshaped),
                    ]),
                )
                .unwrap();
            context
                .write_tensor(&identity, &vec![0u8; expected.len()])
                .unwrap();
            for output in [&sliced, &reshaped] {
                let mut actual = vec![0u8; expected.len()];
                context.read_tensor(output, &mut actual).unwrap();
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn typed_rank_one_copies_preserve_large_int32_values_and_independent_outputs() {
        for constant in [false, true] {
            let mut context = MLContext::create(
                &MLContextOptions::new(MLPowerPreference::Default, false)
                    .with_rustnn_backend_hint(Backend::Coreml),
            )
            .unwrap();
            let mut graph = context
                .rustnn_build_graph(copy_graph(
                    constant,
                    &[
                        CopyOp::Identity,
                        CopyOp::Transpose,
                        CopyOp::Slice,
                        CopyOp::Reshape,
                        CopyOp::Cast,
                    ],
                ))
                .unwrap();
            let descriptor = MLTensorDescriptor::new(MLOperandDataType::Int32, vec![6])
                .to_readable()
                .to_writable();
            let input = context.create_tensor(&descriptor).unwrap();
            let expected = exact_bytes();
            context.write_tensor(&input, &expected).unwrap();
            let outputs: Vec<_> = (0..5)
                .map(|_| context.create_tensor(&descriptor).unwrap())
                .collect();
            let names: Vec<_> = (1..=5).map(|index| format!("copy{index}")).collect();
            context
                .dispatch(
                    &mut graph,
                    &if constant {
                        MLNamedTensors::new()
                    } else {
                        MLNamedTensors::from([("source", &input)])
                    },
                    &names
                        .iter()
                        .zip(&outputs)
                        .map(|(name, tensor)| (name.as_str(), tensor))
                        .collect(),
                )
                .unwrap();
            context.write_tensor(&input, &[0u8; 24]).unwrap();
            for output in &outputs {
                let mut actual = vec![0u8; expected.len()];
                context.read_tensor(output, &mut actual).unwrap();
                assert_eq!(actual, expected);
            }
            context.write_tensor(&outputs[0], &[0u8; 24]).unwrap();
            let mut unaffected = vec![0u8; expected.len()];
            context.read_tensor(&outputs[1], &mut unaffected).unwrap();
            assert_eq!(unaffected, expected);
        }
    }

    /// The one-shot repro from review: `out` copies an `i32[4]` constant that the
    /// converter stores as a MIL immediate. Only the copy proof refers to its raw
    /// weight record, so the converted graph gains a small weights sidecar.
    const IMMEDIATE_CONSTANT_COPY: &str = r#"webnn_graph "immediate_constant_copy" v1 {
  inputs { x: f32[2]; }
  consts { c: i32[4] @bytes([1, 0, 0, 0, 254, 255, 255, 255, 3, 0, 0, 0, 0, 8, 0, 0]); }
  nodes {
    y = relu(x);
    out = identity(c);
  }
  outputs { y; out; }
}
"#;

    /// `w` is stored in `weights.bin`, which the MIL program itself references.
    const BLOB_CONSTANT_COPY: &str = r#"webnn_graph "blob_constant_copy" v1 {
  inputs { x: f32[2]; }
  consts { w: f32[2] @bytes([0, 0, 192, 63, 0, 0, 32, 192]); }
  nodes {
    y = add(x, w);
    out = identity(w);
  }
  outputs { y; out; }
}
"#;

    fn graph_file(text: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("graph.webnn");
        std::fs::write(&path, text).unwrap();
        (directory, path)
    }

    fn named_descriptors(graph: &GraphInfo, ids: &[u32]) -> HashMap<String, OperandDescriptor> {
        ids.iter()
            .map(|&id| {
                let operand = &graph.operands[id as usize];
                (operand.name.clone().unwrap(), operand.descriptor.clone())
            })
            .collect()
    }

    /// CPU is required; every policy that loaded must return the expected values.
    fn assert_outputs(
        attempts: &[rustnn::executors::coreml::CoremlRunAttempt],
        expected: &[(&str, &[f32])],
    ) {
        assert!(
            attempts
                .iter()
                .any(|attempt| attempt.compute_unit == "CPU_ONLY" && attempt.result.is_ok()),
            "{attempts:?}"
        );
        for attempt in attempts {
            let Ok(outputs) = &attempt.result else {
                continue;
            };
            for &(name, values) in expected {
                let output = outputs
                    .iter()
                    .find(|output| output.name == name)
                    .unwrap_or_else(|| {
                        panic!("{}: no `{name}` in {outputs:?}", attempt.compute_unit)
                    });
                assert_eq!(output.data, values, "{} `{name}`", attempt.compute_unit);
            }
        }
    }

    #[test]
    fn one_shot_executors_match_mlcontext_for_an_immediate_constant_copy() {
        use rustnn::executors::coreml::{
            CoremlInput, run_coreml_with_inputs, run_coreml_with_inputs_checked,
            run_coreml_with_inputs_with_weights, run_coreml_zeroed, run_coreml_zeroed_cached,
            run_coreml_zeroed_cached_with_weights,
        };
        let (_directory, path) = graph_file(IMMEDIATE_CONSTANT_COPY);
        let graph = rustnn::load_graph_from_path(&path).unwrap();
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let weights = converted.weights_data.as_deref();
        assert!(weights.is_some(), "the copy proof keeps a raw record");
        let input_descriptors = named_descriptors(&graph, &graph.input_operands);
        let output_descriptors = named_descriptors(&graph, &graph.output_operands);
        let inputs = vec![CoremlInput {
            name: "x".into(),
            shape: vec![2],
            data: vec![-1.5, 2.5],
        }];
        let copied: &[f32] = &[1.0, -2.0, 3.0, 2048.0];

        // With the sidecar the copy comes from the original record; without it,
        // CoreML computes the output, as it did before copy proofs existed.
        for attempts in [
            run_coreml_zeroed_cached_with_weights(
                &converted.data,
                weights,
                &input_descriptors,
                None,
            ),
            run_coreml_zeroed_cached(&converted.data, &input_descriptors, None),
            run_coreml_zeroed(&converted.data, &input_descriptors),
        ] {
            assert_outputs(&attempts.unwrap(), &[("y", &[0.0, 0.0]), ("out", copied)]);
        }
        for attempts in [
            run_coreml_with_inputs_with_weights(&converted.data, weights, inputs.clone()),
            run_coreml_with_inputs(&converted.data, inputs.clone()),
            run_coreml_with_inputs_checked(
                &converted.data,
                inputs,
                &input_descriptors,
                &output_descriptors,
            ),
        ] {
            assert_outputs(&attempts.unwrap(), &[("y", &[0.0, 2.5]), ("out", copied)]);
        }

        // A supplied sidecar that does not match the proof still fails closed.
        let copies: Value = serde_json::from_str(&metadata(&graph)[CONSTANT_COPIES]).unwrap();
        let (_, source) = copies["sources"]
            .as_object()
            .unwrap()
            .iter()
            .next()
            .unwrap();
        let offset = source["offset"].as_u64().unwrap() as usize;
        let mut mismatched = weights.unwrap().to_vec();
        mismatched[offset + 64] ^= 1;
        assert!(
            run_coreml_zeroed_cached_with_weights(
                &converted.data,
                Some(&mismatched),
                &input_descriptors,
                None,
            )
            .is_err()
        );

        // MLContext supplies the same values from the original bytes.
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_rustnn_backend_hint(Backend::Coreml),
        )
        .unwrap();
        let constant = graph
            .constant_operand_ids_to_handles
            .values()
            .next()
            .unwrap();
        let expected_out = constant.data.clone();
        let mut built = context.rustnn_build_graph(graph).unwrap();
        let descriptor = |data_type, length| {
            MLTensorDescriptor::new(data_type, vec![length])
                .to_readable()
                .to_writable()
        };
        let x = context
            .create_tensor(&descriptor(MLOperandDataType::Float32, 2))
            .unwrap();
        let y = context
            .create_tensor(&descriptor(MLOperandDataType::Float32, 2))
            .unwrap();
        let out = context
            .create_tensor(&descriptor(MLOperandDataType::Int32, 4))
            .unwrap();
        let f32_bytes = |values: [f32; 2]| -> Vec<u8> {
            values.into_iter().flat_map(f32::to_le_bytes).collect()
        };
        context.write_tensor(&x, &f32_bytes([-1.5, 2.5])).unwrap();
        context
            .dispatch(
                &mut built,
                &MLNamedTensors::from([("x", &x)]),
                &MLNamedTensors::from([("y", &y), ("out", &out)]),
            )
            .unwrap();
        let mut actual = vec![0u8; expected_out.len()];
        context.read_tensor(&out, &mut actual).unwrap();
        assert_eq!(actual, expected_out);
        let mut actual = vec![0u8; 8];
        context.read_tensor(&y, &mut actual).unwrap();
        assert_eq!(actual, f32_bytes([0.0, 2.5]));
    }

    #[test]
    fn one_shot_executors_need_the_sidecar_for_referenced_weights() {
        use rustnn::executors::coreml::{
            run_coreml_zeroed_cached, run_coreml_zeroed_cached_with_weights,
        };
        let (_directory, path) = graph_file(BLOB_CONSTANT_COPY);
        let graph = rustnn::load_graph_from_path(&path).unwrap();
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let input_descriptors = named_descriptors(&graph, &graph.input_operands);
        // CoreML cannot compile the program without the weights it references, so
        // a missing sidecar never leaves a blob-backed copy to CoreML.
        assert!(run_coreml_zeroed_cached(&converted.data, &input_descriptors, None).is_err());
        let attempts = run_coreml_zeroed_cached_with_weights(
            &converted.data,
            converted.weights_data.as_deref(),
            &input_descriptors,
            None,
        )
        .unwrap();
        assert_outputs(&attempts, &[("y", &[1.5, -2.5]), ("out", &[1.5, -2.5])]);
    }

    #[cfg(feature = "cli")]
    #[test]
    fn cli_run_coreml_executes_constant_copy_outputs() {
        for (text, shape) in [(IMMEDIATE_CONSTANT_COPY, 4), (BLOB_CONSTANT_COPY, 2)] {
            let (_directory, path) = graph_file(text);
            let output = std::process::Command::new(env!("CARGO_BIN_EXE_rustnn"))
                .arg(&path)
                .args(["--convert", "coreml", "--run-coreml"])
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success(),
                "{stdout}{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(stdout.contains("  - CPU_ONLY succeeded:"), "{stdout}");
            assert!(
                stdout.contains(&format!("      out: shape=[{shape}]")),
                "{stdout}"
            );
        }
    }
}

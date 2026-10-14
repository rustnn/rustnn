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
}

//! Converter-only coverage for gather bounds on dynamically sized data.
#![cfg(feature = "dynamic-inputs")]

use prost::Message;
use rustnn::GraphConverter;
use rustnn::converters::CoremlMlProgramConverter;
use rustnn::graph::{
    DataType, Dimension, DynamicDimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
    to_dimension_vector,
};
use rustnn::operators::Operation;
use rustnn::protos::coreml::mil_spec::{
    Argument, Block, DataType as MilDataType, NamedValueType, Operation as MilOperation,
    TensorType, Value, argument::binding::Binding, dimension, tensor_value, value, value_type,
};
use rustnn::protos::coreml::specification::{Model, feature_type, model};
use std::collections::HashMap;

fn dynamic(name: &str, max_size: u32) -> Dimension {
    Dimension::Dynamic(DynamicDimension {
        name: name.into(),
        max_size,
    })
}

fn graph(
    op: &str,
    data_type: DataType,
    data_shape: Vec<Dimension>,
    index_shape: Vec<Dimension>,
    output_shape: Vec<Dimension>,
    axis: u32,
) -> GraphInfo {
    let operand = |name: &str, kind, data_type, shape| Operand {
        kind,
        name: Some(name.into()),
        descriptor: OperandDescriptor {
            data_type,
            shape,
            pending_permutation: vec![],
        },
    };
    GraphInfo {
        operands: vec![
            operand("data", OperandKind::Input, data_type, data_shape),
            operand("indices", OperandKind::Input, DataType::Int32, index_shape),
            operand("result", OperandKind::Output, data_type, output_shape),
        ],
        input_operands: vec![0, 1],
        output_operands: vec![2],
        operations: vec![
            Operation::from_json_attributes(
                op,
                &[0, 1],
                &[2],
                &serde_json::json!({ "axis": axis }),
            )
            .expect("gather operation"),
        ],
        constant_operand_ids_to_handles: HashMap::new(),
        id_to_constant_tensor_operand_map: HashMap::new(),
        quantized: false,
    }
}

fn convert(graph: &GraphInfo) -> Model {
    let converted = CoremlMlProgramConverter.convert(graph).expect("conversion");
    let model = Model::decode(converted.data.as_slice()).expect("CoreML protobuf");
    assert_eq!(model.specification_version, 9);
    model
}

fn block(model: &Model) -> &Block {
    let Some(model::Type::MlProgram(program)) = &model.r#type else {
        panic!("expected MLProgram");
    };
    &program.functions["main"].block_specializations["CoreML7"]
}

fn producer<'a>(block: &'a Block, name: &str) -> &'a MilOperation {
    block
        .operations
        .iter()
        .find(|op| op.outputs.iter().any(|output| output.name == name))
        .unwrap_or_else(|| panic!("missing producer for {name}"))
}

fn named_input<'a>(op: &'a MilOperation, input: &str) -> &'a str {
    let Some(Binding::Name(name)) = &op.inputs[input].arguments[0].binding else {
        panic!("expected named {input} input on {}", op.r#type);
    };
    name
}

fn ints(value: &Value) -> &[i32] {
    let Some(value::Value::ImmediateValue(immediate)) = &value.value else {
        panic!("expected immediate value");
    };
    let Some(value::immediate_value::Value::Tensor(tensor)) = &immediate.value else {
        panic!("expected tensor value");
    };
    let Some(tensor_value::Value::Ints(ints)) = &tensor.value else {
        panic!("expected int tensor");
    };
    &ints.values
}

fn immediate_ints(argument: &Argument) -> &[i32] {
    let Some(Binding::Value(value)) = &argument.arguments[0].binding else {
        panic!("expected immediate argument");
    };
    ints(value)
}

fn tensor(output: &NamedValueType) -> &TensorType {
    let Some(value_type::Type::TensorType(tensor)) = output
        .r#type
        .as_ref()
        .and_then(|value| value.r#type.as_ref())
    else {
        panic!("expected tensor type for {}", output.name);
    };
    tensor
}

fn assert_type(output: &NamedValueType, dtype: MilDataType, shape: &[Option<u64>]) {
    let tensor = tensor(output);
    assert_eq!(tensor.data_type, dtype as i32, "{} dtype", output.name);
    assert_eq!(tensor.rank, shape.len() as i64, "{} rank", output.name);
    let actual: Vec<_> = tensor
        .dimensions
        .iter()
        .map(|dim| match &dim.dimension {
            Some(dimension::Dimension::Constant(size)) => Some(size.size),
            Some(dimension::Dimension::Unknown(_)) => None,
            None => panic!("missing dimension"),
        })
        .collect();
    assert_eq!(actual, shape, "{} shape", output.name);
}

fn assert_runtime_bounds(block: &Block, data_rank: u64, axis: i32, width: i32) {
    assert_eq!(
        block
            .operations
            .iter()
            .filter(|op| op.r#type == "shape")
            .count(),
        1
    );
    let shape = producer(block, "result_data_shape");
    assert_eq!(shape.r#type, "shape");
    assert_type(&shape.outputs[0], MilDataType::Int32, &[Some(data_rank)]);
    let bounds = producer(block, "result_gsz");
    assert_eq!(bounds.r#type, "slice_by_size");
    assert_eq!(named_input(bounds, "x"), shape.outputs[0].name);
    assert_eq!(immediate_ints(&bounds.inputs["begin"]), [axis]);
    assert_eq!(immediate_ints(&bounds.inputs["size"]), [width]);
    assert_type(
        &bounds.outputs[0],
        MilDataType::Int32,
        &[Some(width as u64)],
    );
    let last = producer(block, "result_gszm1");
    assert_eq!(last.r#type, "sub");
    assert_eq!(named_input(last, "x"), bounds.outputs[0].name);
    assert_eq!(
        ints(&producer(block, named_input(last, "y")).attributes["val"]),
        [1]
    );
    assert_eq!(
        named_input(producer(block, "result_goff"), "y"),
        bounds.outputs[0].name
    );
    assert_eq!(
        named_input(producer(block, "result_gcl"), "y"),
        last.outputs[0].name
    );
}

#[test]
fn static_indexed_axes_keep_constant_bounds() {
    for (op, data_shape, indices, output, axis, sizes) in [
        ("gather", vec![2, 3], vec![2], vec![2, 2], 1, vec![3]),
        (
            "gatherElements",
            vec![2, 3],
            vec![2, 2],
            vec![2, 2],
            1,
            vec![3],
        ),
        (
            "gatherND",
            vec![2, 3, 4],
            vec![2, 2],
            vec![2, 4],
            0,
            vec![2, 3],
        ),
    ] {
        let model = convert(&graph(
            op,
            DataType::Float32,
            to_dimension_vector(&data_shape),
            to_dimension_vector(&indices),
            to_dimension_vector(&output),
            axis,
        ));
        let block = block(&model);
        assert!(
            !block
                .operations
                .iter()
                .any(|op| matches!(op.r#type.as_str(), "shape" | "slice_by_size"))
        );
        for (name, expected) in [
            ("result_gsz", sizes.clone()),
            ("result_gszm1", sizes.iter().map(|size| size - 1).collect()),
        ] {
            let constant = producer(block, name);
            assert_eq!(constant.r#type, "const");
            assert_eq!(ints(&constant.attributes["val"]), expected);
        }
    }
}

#[test]
fn unrelated_dynamic_axes_do_not_add_runtime_bounds() {
    let batch = dynamic("batch", 4);
    let model = convert(&graph(
        "gather",
        DataType::Float32,
        vec![batch.clone(), Dimension::Static(3)],
        to_dimension_vector(&[2]),
        vec![batch, Dimension::Static(2)],
        1,
    ));
    assert!(
        !block(&model)
            .operations
            .iter()
            .any(|op| op.r#type == "shape")
    );
    assert_eq!(
        ints(&producer(block(&model), "result_gsz").attributes["val"]),
        [3]
    );
}

#[test]
fn gather_and_gather_elements_read_the_selected_runtime_axis() {
    for (op, indices, output) in [
        ("gather", vec![2], vec![2, 2, 3]),
        ("gatherElements", vec![2, 2, 3], vec![2, 2, 3]),
    ] {
        let model = convert(&graph(
            op,
            DataType::Float32,
            vec![
                Dimension::Static(2),
                dynamic("sequence", 4),
                Dimension::Static(3),
            ],
            to_dimension_vector(&indices),
            to_dimension_vector(&output),
            1,
        ));
        assert_runtime_bounds(block(&model), 3, 1, 1);
        assert_eq!(
            named_input(producer(block(&model), "result_data_shape"), "x"),
            "data"
        );
    }
}

#[test]
fn gather_nd_broadcasts_active_bounds_per_tuple_component() {
    let count = dynamic("count", 4);
    let model = convert(&graph(
        "gatherND",
        DataType::Float32,
        vec![
            Dimension::Static(2),
            dynamic("sequence", 4),
            Dimension::Static(3),
        ],
        vec![count.clone(), Dimension::Static(2)],
        vec![count, Dimension::Static(3)],
        0,
    ));
    let block = block(&model);
    assert_runtime_bounds(block, 3, 0, 2);
    for suffix in ["goff", "gwrap", "gmx", "gcl"] {
        assert_type(
            &producer(block, &format!("result_{suffix}")).outputs[0],
            MilDataType::Int32,
            &[None, Some(2)],
        );
    }
}

#[test]
fn scalar_gather_keeps_vector_indices_and_scalar_boundary_shape() {
    let graph = graph(
        "gather",
        DataType::Float32,
        vec![dynamic("sequence", 4)],
        vec![],
        vec![],
        0,
    );
    let model = convert(&graph);
    let block = block(&model);
    assert_runtime_bounds(block, 1, 0, 1);
    assert_type(
        &producer(block, "result_gcl").outputs[0],
        MilDataType::Int32,
        &[Some(1)],
    );
    let gather = block
        .operations
        .iter()
        .find(|op| op.r#type == "gather")
        .unwrap();
    let index = producer(block, named_input(gather, "indices"));
    assert_eq!(index.outputs[0].name, "result_gcl");
    assert_type(&index.outputs[0], MilDataType::Int32, &[Some(1)]);
    assert_type(&gather.outputs[0], MilDataType::Float32, &[Some(1)]);
    assert_eq!(gather.outputs[0].name, "result");
    assert!(graph.operands[1].descriptor.shape.is_empty());
    assert!(graph.operands[2].descriptor.shape.is_empty());
    let output = &model.description.as_ref().unwrap().output[0];
    let Some(feature_type::Type::MultiArrayType(array)) =
        output.r#type.as_ref().and_then(|ty| ty.r#type.as_ref())
    else {
        panic!("expected array output feature");
    };
    assert_eq!(array.shape, [1]);
}

#[test]
fn scalar_gather_squeezes_only_the_indexed_axis_and_keeps_dynamic_dimensions() {
    for axis in 0..5 {
        let mut data_shape = vec![
            dynamic("batch", 4),
            Dimension::Static(1),
            dynamic("sequence", 8),
            Dimension::Static(1),
            dynamic("channels", 3),
        ];
        data_shape[axis] = Dimension::Static(3);
        let mut output_shape = data_shape.clone();
        output_shape.remove(axis);
        let model = convert(&graph(
            "gather",
            DataType::Float32,
            data_shape.clone(),
            vec![],
            output_shape,
            axis as u32,
        ));
        let block = block(&model);
        let gather = producer(block, "result_gather_vector");
        assert_eq!(gather.r#type, "gather");
        assert_type(
            &producer(block, named_input(gather, "indices")).outputs[0],
            MilDataType::Int32,
            &[Some(1)],
        );
        let mut dimensions: Vec<_> = data_shape
            .iter()
            .map(|dim| match dim {
                Dimension::Static(size) => Some(u64::from(*size)),
                Dimension::Dynamic(_) => None,
            })
            .collect();
        dimensions[axis] = Some(1);
        assert_type(&gather.outputs[0], MilDataType::Float32, &dimensions);
        let squeeze = producer(block, "result");
        assert_eq!(squeeze.r#type, "squeeze");
        assert_eq!(named_input(squeeze, "x"), gather.outputs[0].name);
        assert_eq!(immediate_ints(&squeeze.inputs["axes"]), [axis as i32]);
        dimensions.remove(axis);
        assert_type(&squeeze.outputs[0], MilDataType::Float32, &dimensions);
    }
}

#[test]
fn narrow_integer_data_is_cast_only_on_the_shape_branch() {
    for data_type in [DataType::Int8, DataType::Uint8] {
        let model = convert(&graph(
            "gather",
            data_type,
            vec![dynamic("sequence", 4)],
            to_dimension_vector(&[2]),
            to_dimension_vector(&[2]),
            0,
        ));
        let block = block(&model);
        let bounds = producer(block, "result_graph_gsz");
        assert_eq!(bounds.r#type, "slice_by_size");
        assert_type(&bounds.outputs[0], MilDataType::Int32, &[Some(1)]);
        let shape = producer(block, named_input(bounds, "x"));
        assert_eq!(shape.r#type, "shape");
        let cast = producer(block, named_input(shape, "x"));
        assert_eq!(cast.r#type, "cast");
        assert_type(&cast.outputs[0], MilDataType::Int32, &[None]);
        let gather = block
            .operations
            .iter()
            .find(|op| op.r#type == "gather")
            .unwrap();
        assert_eq!(named_input(gather, "x"), named_input(cast, "x"));
        assert_ne!(named_input(gather, "x"), cast.outputs[0].name);
    }
}

#[test]
fn gather_nd_rejects_dynamic_tuple_width() {
    let graph = graph(
        "gatherND",
        DataType::Float32,
        to_dimension_vector(&[3, 4]),
        vec![Dimension::Static(2), dynamic("tuple_width", 2)],
        to_dimension_vector(&[2]),
        0,
    );
    let error = CoremlMlProgramConverter
        .convert(&graph)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("gatherND requires a static index tuple width"),
        "{error}"
    );
}

#[test]
fn gather_bounds_reject_zero_and_non_int32_extents() {
    for extent in [0, i32::MAX as u32 + 1, u32::MAX] {
        for dimension in [Dimension::Static(extent), dynamic("sequence", extent)] {
            for op in ["gather", "gatherElements", "gatherND"] {
                let index_shape = if op == "gatherND" {
                    vec![1, 1]
                } else {
                    vec![1]
                };
                let graph = graph(
                    op,
                    DataType::Float32,
                    vec![dimension.clone()],
                    to_dimension_vector(&index_shape),
                    to_dimension_vector(&[1]),
                    0,
                );
                let error = CoremlMlProgramConverter
                    .convert(&graph)
                    .unwrap_err()
                    .to_string();
                assert!(
                    error.contains("indexed extents must be in 1..=i32::MAX"),
                    "{op}: {error}"
                );
            }
        }
    }
}

#[test]
fn gather_bounds_reject_out_of_rank_axes_without_panicking() {
    for op in ["gather", "gatherElements"] {
        for axis in [1, u32::MAX] {
            let graph = graph(
                op,
                DataType::Float32,
                to_dimension_vector(&[3]),
                to_dimension_vector(&[1]),
                to_dimension_vector(&[1]),
                axis,
            );
            let error = CoremlMlProgramConverter
                .convert(&graph)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("indexed axes must be within the data rank"),
                "{op}: {error}"
            );
        }
    }
}

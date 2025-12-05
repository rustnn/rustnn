use crate::converters::ConvertedGraph;
use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo};
use crate::protos::onnx::{
    GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto,
    TypeProto, ValueInfoProto, tensor_proto::DataType as ProtoDataType,
    type_proto::Tensor as TensorTypeProto,
};
use prost::Message;

#[derive(Default)]
pub struct OnnxConverter;

impl OnnxConverter {
    fn operand_name(graph: &GraphInfo, id: u32) -> String {
        graph
            .operand(id)
            .and_then(|op| op.name.clone())
            .unwrap_or_else(|| format!("operand_{}", id))
    }

    fn data_type_code(data_type: DataType) -> ProtoDataType {
        match data_type {
            DataType::Float32 => ProtoDataType::Float,
            DataType::Uint8 => ProtoDataType::Uint8,
            DataType::Int8 => ProtoDataType::Int8,
            DataType::Int32 => ProtoDataType::Int32,
            DataType::Float16 => ProtoDataType::Float16,
            DataType::Uint32 => ProtoDataType::Uint32,
        }
    }

    fn onnx_op_type(op_type: &str) -> String {
        let mut chars = op_type.chars();
        if let Some(first) = chars.next() {
            let mut s = first.to_ascii_uppercase().to_string();
            s.push_str(&chars.collect::<String>());
            s
        } else {
            String::new()
        }
    }
}

impl crate::converters::GraphConverter for OnnxConverter {
    fn format(&self) -> &'static str {
        "onnx"
    }

    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
        let mut initializers = Vec::new();
        let mut inputs_val = Vec::new();
        let mut outputs_val = Vec::new();

        for &id in &graph.input_operands {
            let operand = graph
                .operand(id)
                .ok_or_else(|| GraphError::InvalidConversionOperand { operand: id })?;
            inputs_val.push(value_info(
                &Self::operand_name(graph, id),
                &operand.descriptor,
            ));
        }

        for &id in &graph.output_operands {
            let operand = graph
                .operand(id)
                .ok_or_else(|| GraphError::InvalidConversionOperand { operand: id })?;
            outputs_val.push(value_info(
                &Self::operand_name(graph, id),
                &operand.descriptor,
            ));
        }

        for (id, data) in &graph.constant_operand_ids_to_handles {
            let operand = graph
                .operand(*id)
                .ok_or_else(|| GraphError::InvalidConversionOperand { operand: *id })?;
            initializers.push(TensorProto {
                name: Some(Self::operand_name(graph, *id)),
                data_type: Some(Self::data_type_code(operand.descriptor.data_type) as i32),
                dims: operand.descriptor.shape.iter().map(|d| *d as i64).collect(),
                raw_data: Some(prost::bytes::Bytes::from(data.data.clone())),
                ..Default::default()
            });
        }

        let nodes = graph
            .operations
            .iter()
            .enumerate()
            .map(|(idx, op)| NodeProto {
                input: op
                    .input_operands
                    .iter()
                    .map(|id| Self::operand_name(graph, *id))
                    .collect(),
                output: vec![Self::operand_name(graph, op.output_operand)],
                name: Some(
                    op.label
                        .clone()
                        .unwrap_or_else(|| format!("{}_{}", op.op_type, idx)),
                ),
                op_type: Some(Self::onnx_op_type(&op.op_type)),
                attribute: Vec::new(),
                ..Default::default()
            })
            .collect();

        let graph_proto = GraphProto {
            name: Some("webnn_graph".to_string()),
            node: nodes,
            input: inputs_val,
            output: outputs_val,
            initializer: initializers,
            ..Default::default()
        };

        let model = ModelProto {
            ir_version: Some(8),
            producer_name: Some("rustnn".to_string()),
            graph: Some(graph_proto),
            opset_import: vec![OperatorSetIdProto {
                version: Some(13),
                ..Default::default()
            }],
            ..Default::default()
        };

        let data = model.encode_to_vec();

        Ok(ConvertedGraph {
            format: "onnx",
            content_type: "application/onnx",
            data,
        })
    }
}

fn value_info(name: &str, desc: &crate::graph::OperandDescriptor) -> ValueInfoProto {
    ValueInfoProto {
        name: Some(name.to_string()),
        r#type: Some(TypeProto {
            value: Some(crate::protos::onnx::type_proto::Value::TensorType(
                TensorTypeProto {
                    elem_type: Some(OnnxConverter::data_type_code(desc.data_type) as i32),
                    shape: Some(TensorShapeProto {
                        dim: desc
                            .shape
                            .iter()
                            .map(|d| crate::protos::onnx::tensor_shape_proto::Dimension {
                                value: Some(
                                    crate::protos::onnx::tensor_shape_proto::dimension::Value::DimValue(
                                        *d as i64,
                                    ),
                                ),
                                ..Default::default()
                            })
                            .collect(),
                    }),
                    ..Default::default()
                },
            )),
            ..Default::default()
        }),
        ..Default::default()
    }
}

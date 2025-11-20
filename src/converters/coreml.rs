use crate::converters::ConvertedGraph;
use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo};
use crate::protos::coreml::specification::{
    array_feature_type::ArrayDataType, feature_type, model, neural_network_layer::Layer,
    AddLayerParams, ArrayFeatureType, FeatureDescription, FeatureType, LoadConstantLayerParams,
    Model, ModelDescription, NeuralNetwork, NeuralNetworkLayer, WeightParams,
};
use prost::bytes::Bytes;
use prost::Message;

#[derive(Default)]
pub struct CoremlConverter;

impl CoremlConverter {
    fn operand_name(graph: &GraphInfo, id: u32) -> String {
        graph
            .operand(id)
            .and_then(|op| op.name.clone())
            .unwrap_or_else(|| format!("operand_{}", id))
    }

    fn coerce_shape(shape: &[u32]) -> Vec<i64> {
        let mut dims: Vec<i64> = shape.iter().map(|d| *d as i64).collect();
        match dims.len() {
            0 => vec![1],
            1 => dims,
            2 => {
                let mut with_batch = vec![1];
                with_batch.append(&mut dims);
                with_batch
            }
            3 => dims,
            _ => {
                let prod: i64 = dims.iter().product();
                vec![prod]
            }
        }
    }

    fn feature_type(desc: &crate::graph::OperandDescriptor) -> FeatureType {
        let shape = Self::coerce_shape(&desc.shape);
        FeatureType {
            r#type: Some(feature_type::Type::MultiArrayType(ArrayFeatureType {
                shape,
                data_type: match desc.data_type {
                    DataType::Float32 => ArrayDataType::Float32 as i32,
                    DataType::Float16 => ArrayDataType::Float16 as i32,
                    _ => ArrayDataType::Int32 as i32,
                },
                ..Default::default()
            })),
            ..Default::default()
        }
    }
}

impl crate::converters::GraphConverter for CoremlConverter {
    fn format(&self) -> &'static str {
        "coreml"
    }

    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
        let inputs = graph
            .input_operands
            .iter()
            .map(|id| {
                let operand = graph
                    .operand(*id)
                    .ok_or_else(|| GraphError::InvalidConversionOperand { operand: *id })?;
                Ok(FeatureDescription {
                    name: Self::operand_name(graph, *id),
                    r#type: Some(Self::feature_type(&operand.descriptor)),
                    ..Default::default()
                })
            })
            .collect::<Result<Vec<_>, GraphError>>()?;

        let outputs = graph
            .output_operands
            .iter()
            .map(|id| {
                let operand = graph
                    .operand(*id)
                    .ok_or_else(|| GraphError::InvalidConversionOperand { operand: *id })?;
                Ok(FeatureDescription {
                    name: Self::operand_name(graph, *id),
                    r#type: Some(Self::feature_type(&operand.descriptor)),
                    ..Default::default()
                })
            })
            .collect::<Result<Vec<_>, GraphError>>()?;

        let mut layers = Vec::new();

        // Emit constants as LoadConstant layers
        for (id, data) in &graph.constant_operand_ids_to_handles {
            let operand = graph
                .operand(*id)
                .ok_or_else(|| GraphError::InvalidConversionOperand { operand: *id })?;
            let name = Self::operand_name(graph, *id);
            let shape: Vec<u64> = Self::coerce_shape(&operand.descriptor.shape)
                .into_iter()
                .map(|d| d as u64)
                .collect();
            let float_value = if matches!(operand.descriptor.data_type, DataType::Float32) {
                bytemuck::cast_slice::<u8, f32>(&data.data).to_vec()
            } else {
                Vec::new()
            };

            let weight = WeightParams {
                float_value,
                raw_value: if matches!(operand.descriptor.data_type, DataType::Float32) {
                    Bytes::new()
                } else {
                    Bytes::from(data.data.clone())
                },
                ..Default::default()
            };
            layers.push(NeuralNetworkLayer {
                name: format!("const_{}", name),
                input: vec![],
                output: vec![name],
                layer: Some(Layer::LoadConstant(LoadConstantLayerParams {
                    shape: shape.clone(),
                    data: Some(weight.clone()),
                    ..Default::default()
                })),
                ..Default::default()
            });
        }

        for (idx, op) in graph.operations.iter().enumerate() {
            let layer_name = op
                .label
                .as_ref()
                .cloned()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| format!("{}_{}", op.op_type, idx));
            let input_names = op
                .input_operands
                .iter()
                .map(|id| Self::operand_name(graph, *id))
                .collect();
            let output_names = vec![Self::operand_name(graph, op.output_operand)];

            let layer = if op.op_type.eq_ignore_ascii_case("add") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Add(AddLayerParams {
                        alpha: 0.0,
                        ..Default::default()
                    })),
                    ..Default::default()
                }
            } else {
                return Err(GraphError::ConversionFailed {
                    format: "coreml".to_string(),
                    reason: format!("Unsupported op_type {}", op.op_type),
                });
            };
            layers.push(layer);
        }

        let nn = NeuralNetwork {
            layers,
            ..Default::default()
        };

        let description = ModelDescription {
            input: inputs,
            output: outputs,
            ..Default::default()
        };

        let model = Model {
            specification_version: 7,
            description: Some(description),
            r#type: Some(model::Type::NeuralNetwork(nn)),
            ..Default::default()
        };

        let data = model.encode_to_vec();

        Ok(ConvertedGraph {
            format: "coreml",
            content_type: "application/x-apple-mlmodel",
            data,
        })
    }
}

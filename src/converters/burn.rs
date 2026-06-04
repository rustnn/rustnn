//! WebNN graph to Burn execution-plan converter.

#![cfg(feature = "burn-plan")]

use std::collections::HashMap;

use crate::burn::{BurnGraphPlan, ConstantSlot, IOBinding};
use crate::converters::{ConvertedGraph, GraphConverter, operand_name};
use crate::error::GraphError;
use crate::graph::{DataType, Dimension, GraphInfo, OperandKind};

pub struct BurnConverter;

impl GraphConverter for BurnConverter {
    fn format(&self) -> &'static str {
        "burn"
    }

    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
        let plan = Self::build_plan(graph)?;
        let data = plan
            .serialize()
            .map_err(|err| GraphError::ConversionFailed {
                format: "burn".to_string(),
                reason: format!("failed to serialize burn plan: {err}"),
            })?;

        Ok(ConvertedGraph {
            format: "burn",
            content_type: "application/x-burn-plan",
            data,
            weights_data: None,
        })
    }
}

impl BurnConverter {
    fn build_plan(graph: &GraphInfo) -> Result<BurnGraphPlan, GraphError> {
        let inputs = collect_inputs(graph)?;
        let outputs = collect_outputs(graph)?;
        let constants = collect_constants(graph)?;
        let operand_types = collect_operand_types(graph);
        Ok(BurnGraphPlan::new(
            inputs,
            outputs,
            constants,
            graph.operations.clone(),
            operand_types,
        ))
    }
}

fn collect_operand_types(graph: &GraphInfo) -> HashMap<u32, DataType> {
    graph
        .operands
        .iter()
        .enumerate()
        .map(|(id, operand)| (id as u32, operand.descriptor.data_type))
        .collect()
}

fn collect_inputs(graph: &GraphInfo) -> Result<Vec<IOBinding>, GraphError> {
    graph
        .input_operands
        .iter()
        .map(|&operand_id| binding_for_operand(graph, operand_id))
        .collect()
}

fn collect_outputs(graph: &GraphInfo) -> Result<Vec<IOBinding>, GraphError> {
    graph
        .output_operands
        .iter()
        .map(|&operand_id| binding_for_operand(graph, operand_id))
        .collect()
}

fn binding_for_operand(graph: &GraphInfo, operand_id: u32) -> Result<IOBinding, GraphError> {
    let operand = graph
        .operand(operand_id)
        .ok_or(GraphError::InvalidConversionOperand {
            operand: operand_id,
        })?;
    let name = operand_name(graph, operand_id);
    Ok(IOBinding {
        name,
        operand_id,
        data_type: operand.descriptor.data_type,
        shape: descriptor_shape_to_i64(&operand.descriptor.shape),
    })
}

fn descriptor_shape_to_i64(shape: &[Dimension]) -> Vec<i64> {
    shape
        .iter()
        .map(|dim| match dim {
            Dimension::Static(v) => i64::from(*v),
            Dimension::Dynamic(_) => -1,
        })
        .collect()
}

fn collect_constants(graph: &GraphInfo) -> Result<Vec<ConstantSlot>, GraphError> {
    let mut constants = Vec::new();
    for (operand_id, constant) in &graph.constant_operand_ids_to_handles {
        let operand = graph
            .operand(*operand_id)
            .ok_or(GraphError::InvalidConversionOperand {
                operand: *operand_id,
            })?;
        if operand.kind != OperandKind::Constant {
            continue;
        }
        let shape = operand
            .descriptor
            .static_shape()
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "burn".to_string(),
                reason: format!(
                    "constant operand {operand_id} has dynamic shape; burn backend requires static shapes"
                ),
            })?;
        constants.push(ConstantSlot {
            operand_id: *operand_id,
            data_type: operand.descriptor.data_type,
            shape,
            data: constant.data.clone(),
        });
    }
    constants.sort_by_key(|slot| slot.operand_id);
    Ok(constants)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DataType, Operand, OperandDescriptor, to_dimension_vector};
    use crate::operators::Operation;
    use std::collections::HashMap;

    fn sample_add_graph() -> GraphInfo {
        GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[2, 2]),
                        pending_permutation: Vec::new(),
                    },
                    name: Some("lhs".to_string()),
                },
                Operand {
                    kind: OperandKind::Constant,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[2, 2]),
                        pending_permutation: Vec::new(),
                    },
                    name: Some("rhs".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[2, 2]),
                        pending_permutation: Vec::new(),
                    },
                    name: Some("sum".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![2],
            operations: vec![Operation::Add {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![2],
            }],
            constant_operand_ids_to_handles: HashMap::from([(
                1,
                crate::graph::ConstantData {
                    data: vec![0u8; 16],
                    label: None,
                },
            )]),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        }
    }

    #[test]
    fn converts_add_graph_to_plan_bytes() {
        let graph = sample_add_graph();
        let converted = BurnConverter.convert(&graph).unwrap();
        assert_eq!(converted.format, "burn");
        let plan = BurnGraphPlan::deserialize(&converted.data).unwrap();
        assert_eq!(plan.inputs.len(), 1);
        assert_eq!(plan.outputs.len(), 1);
        assert_eq!(plan.operations.len(), 1);
        assert!(matches!(plan.operations[0], Operation::Add { .. }));
    }

    #[cfg(feature = "burn-runtime-cpu")]
    mod sample_graph {
        use std::path::PathBuf;

        use crate::converters::{BurnConverter, GraphConverter};
        use crate::executors::burn::{BurnInput, run_burn_cpu_with_inputs};
        use crate::{ContextProperties, GraphValidator, load_graph_from_path};

        #[test]
        fn sample_graph_add_zeroed_lhs() {
            let path =
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/sample_graph.webnn");
            let graph = load_graph_from_path(&path).expect("sample graph should load");
            GraphValidator::new(&graph, ContextProperties::default())
                .validate()
                .expect("sample graph should validate");
            let converted = BurnConverter.convert(&graph).expect("convert to burn plan");
            let outputs = run_burn_cpu_with_inputs(
                &converted.data,
                vec![BurnInput {
                    name: "lhs".to_string(),
                    shape: vec![2, 2],
                    data: vec![0.0; 4],
                    int64_data: None,
                    uint64_data: None,
                }],
            )
            .expect("burn cpu execution");
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].name, "sum");
            assert_eq!(outputs[0].data, vec![1.0, 1.0, 1.0, 1.0]);
        }
    }
}

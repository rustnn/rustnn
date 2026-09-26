use std::ops::Deref;

use crate::Operation;
use crate::error::GraphBuilderError;
use crate::graph::{ConstantData, GraphInfo, Operand, OperandDescriptor, OperandKind};
use crate::mlcontext::MLOperand;
use crate::mlgraphbuilder::infer_operation_descriptors;

/// Owns an in-progress graph and records only fully inferred operations.
///
/// Operation insertion is atomic: inference and output validation happen before
/// the operation or any of its output operands are appended to `GraphInfo`.
#[derive(Clone, Debug, Default)]
pub(crate) struct GraphRecorder {
    graph: GraphInfo,
}

impl GraphRecorder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn graph(&self) -> &GraphInfo {
        &self.graph
    }

    pub(crate) fn set_quantized(&mut self, quantized: bool) {
        self.graph.quantized = quantized;
    }

    /// Returns the contiguous operand IDs expected for the next outputs.
    pub(crate) fn next_output_ids(&self, count: usize) -> Vec<u32> {
        let base = self.graph.operands.len() as u32;
        (0..count as u32).map(|offset| base + offset).collect()
    }

    /// Appends a named input and returns its operand ID.
    pub(crate) fn add_input(&mut self, name: String, descriptor: OperandDescriptor) -> u32 {
        let id = self.graph.operands.len() as u32;
        self.graph.operands.push(Operand {
            kind: OperandKind::Input,
            descriptor,
            name: Some(name),
        });
        self.graph.input_operands.push(id);
        id
    }

    /// Appends a constant and optionally associates its external tensor name.
    pub(crate) fn add_constant(
        &mut self,
        name: Option<String>,
        descriptor: OperandDescriptor,
        data: ConstantData,
        tensor_name: Option<String>,
    ) -> u32 {
        let id = self.graph.operands.len() as u32;
        self.graph.operands.push(Operand {
            kind: OperandKind::Constant,
            descriptor,
            name,
        });
        self.graph.constant_operand_ids_to_handles.insert(id, data);
        if let Some(tensor_name) = tensor_name {
            self.graph
                .id_to_constant_tensor_operand_map
                .insert(id, tensor_name);
        }
        id
    }

    /// Infers output descriptors and then appends the operation atomically.
    pub(crate) fn record_operation(
        &mut self,
        operation: Operation,
        output_names: Option<&[String]>,
    ) -> Result<Vec<MLOperand>, GraphBuilderError> {
        let output_ids = operation.output_operands();
        let expected_ids = self.next_output_ids(output_ids.len());
        if output_ids != expected_ids {
            return Err(GraphBuilderError::InconsistentGraphInfo {
                message: format!(
                    "operation {} declares output ids {output_ids:?}, expected {expected_ids:?}",
                    operation.op_type()
                ),
            });
        }
        if let Some(names) = output_names
            && names.len() != output_ids.len()
        {
            return Err(GraphBuilderError::InconsistentGraphInfo {
                message: format!(
                    "operation {} has {} output names for {} outputs",
                    operation.op_type(),
                    names.len(),
                    output_ids.len()
                ),
            });
        }

        let descriptors = if output_ids.is_empty() {
            Vec::new()
        } else {
            infer_operation_descriptors(&operation, &self.graph)?
        };
        let label = (!operation.label().is_empty()).then(|| operation.label().to_string());
        let operands = output_ids
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let name = output_names
                    .and_then(|names| names.get(index).cloned())
                    .or_else(|| match (&label, output_ids.len()) {
                        (Some(label), 1) => Some(label.clone()),
                        (Some(label), _) => Some(format!("{label}_{index}")),
                        (None, _) => None,
                    });
                Operand {
                    kind: OperandKind::Intermediate,
                    descriptor: descriptors[index].clone(),
                    name,
                }
            })
            .collect::<Vec<_>>();

        self.graph.operations.push(operation);
        self.graph.operands.extend(operands);
        Ok(expected_ids
            .into_iter()
            .map(|id| MLOperand { id: id as usize })
            .collect())
    }

    /// Marks an existing intermediate operand as a named graph output.
    pub(crate) fn mark_output(&mut self, id: u32, name: String) -> Result<(), GraphBuilderError> {
        let operand = self.graph.operands.get_mut(id as usize).ok_or(
            GraphBuilderError::BuildWithInvalidOperand {
                operand: MLOperand { id: id as usize },
                name: name.clone(),
            },
        )?;
        match operand.kind {
            OperandKind::Input => {
                return Err(GraphBuilderError::RequestedInputAsOutput {
                    operand: operand.clone(),
                    id: id as usize,
                });
            }
            OperandKind::Constant => {
                return Err(GraphBuilderError::RequestedConstantAsOutput {
                    operand: operand.clone(),
                    id: id as usize,
                });
            }
            OperandKind::Intermediate | OperandKind::Output => {}
        }
        operand.kind = OperandKind::Output;
        operand.name = Some(name);
        if !self.graph.output_operands.contains(&id) {
            self.graph.output_operands.push(id);
            self.graph.output_operands.sort_unstable();
        }
        Ok(())
    }

    /// Validates the I/O lists and returns the completed graph.
    pub(crate) fn into_graph(self) -> Result<GraphInfo, GraphBuilderError> {
        self.graph.validate_io_operand_lists().map_err(|error| {
            GraphBuilderError::InconsistentGraphInfo {
                message: error.to_string(),
            }
        })?;
        Ok(self.graph)
    }
}

impl Deref for GraphRecorder {
    type Target = GraphInfo;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DataType, Dimension, DynamicDimension, to_dimension_vector};
    use crate::operator_options::{
        MLConstantOptions, MLGruCellOptions, MLGruOptions, MLLstmOptions, MLSplitOptions,
    };

    fn descriptor(shape: &[u32]) -> OperandDescriptor {
        OperandDescriptor {
            data_type: DataType::Float32,
            shape: to_dimension_vector(shape),
            pending_permutation: Vec::new(),
        }
    }

    fn named_inputs(recorder: &mut GraphRecorder, shapes: &[&[u32]]) -> Vec<u32> {
        shapes
            .iter()
            .enumerate()
            .map(|(index, shape)| recorder.add_input(format!("input_{index}"), descriptor(shape)))
            .collect()
    }

    #[test]
    fn scalar_descriptors_are_known_for_every_operand_kind() {
        let mut recorder = GraphRecorder::new();
        let input = recorder.add_input("input".to_string(), descriptor(&[]));
        let constant = recorder.add_constant(
            Some("constant".to_string()),
            descriptor(&[]),
            ConstantData {
                data: 1.0f32.to_le_bytes().to_vec(),
                label: None,
            },
            None,
        );
        let output_id = recorder.next_output_ids(1)[0];
        let outputs = recorder
            .record_operation(
                Operation::Add {
                    a: input,
                    b: constant,
                    options: None,
                    outputs: vec![output_id],
                },
                None,
            )
            .unwrap();
        assert!(recorder.operands[outputs[0].id].descriptor.shape.is_empty());
        recorder
            .mark_output(output_id, "output".to_string())
            .unwrap();
        let graph = recorder.into_graph().unwrap();
        for operand in &graph.operands {
            assert!(operand.descriptor.shape.is_empty());
        }
        assert_eq!(graph.operands[0].kind, OperandKind::Input);
        assert_eq!(graph.operands[1].kind, OperandKind::Constant);
        assert_eq!(graph.operands[2].kind, OperandKind::Output);
    }

    #[test]
    fn known_dynamic_dimensions_survive_single_output_recording() {
        let dynamic = Dimension::Dynamic(DynamicDimension {
            name: "batch".to_string(),
            max_size: 8,
        });
        let mut recorder = GraphRecorder::new();
        let input = recorder.add_input(
            "input".to_string(),
            OperandDescriptor {
                data_type: DataType::Float32,
                shape: vec![dynamic.clone(), Dimension::Static(4)],
                pending_permutation: Vec::new(),
            },
        );
        let output = recorder.next_output_ids(1)[0];
        recorder
            .record_operation(
                Operation::Identity {
                    input,
                    options: None,
                    outputs: vec![output],
                },
                None,
            )
            .unwrap();
        assert_eq!(
            recorder.operands[output as usize].descriptor.shape,
            vec![dynamic, Dimension::Static(4)]
        );
    }

    #[test]
    fn failed_inference_is_atomic_and_never_adds_a_scalar_placeholder() {
        let mut recorder = GraphRecorder::new();
        let inputs = named_inputs(&mut recorder, &[&[2], &[3]]);
        let operand_count = recorder.operands.len();
        let operation_count = recorder.operations.len();
        let output = recorder.next_output_ids(1)[0];
        let error = recorder
            .record_operation(
                Operation::Add {
                    a: inputs[0],
                    b: inputs[1],
                    options: None,
                    outputs: vec![output],
                },
                None,
            )
            .unwrap_err();
        assert!(error.to_string().contains("Shape inference failed"));
        assert_eq!(recorder.operands.len(), operand_count);
        assert_eq!(recorder.operations.len(), operation_count);
    }

    #[test]
    fn constant_shape_distinguishes_scalar_from_missing() {
        let scalar_options = MLConstantOptions {
            data_type: "float32".to_string(),
            shape: Some(vec![]),
            ..Default::default()
        };
        let mut scalar = GraphRecorder::new();
        scalar
            .record_operation(
                Operation::Constant {
                    options: Some(scalar_options.clone()),
                    outputs: vec![0],
                },
                None,
            )
            .unwrap();
        assert!(scalar.operands[0].descriptor.shape.is_empty());

        let mut missing = GraphRecorder::new();
        let error = missing
            .record_operation(
                Operation::Constant {
                    options: Some(MLConstantOptions {
                        shape: None,
                        ..scalar_options
                    }),
                    outputs: vec![0],
                },
                None,
            )
            .unwrap_err();
        assert!(error.to_string().contains("missing its required shape"));
        assert!(missing.operands.is_empty());
        assert!(missing.operations.is_empty());
    }

    #[test]
    fn scalar_gru_cell_hidden_state_is_rejected_without_fallback() {
        let mut recorder = GraphRecorder::new();
        let ids = named_inputs(&mut recorder, &[&[2, 4], &[15, 4], &[15, 5], &[]]);
        let output = recorder.next_output_ids(1)[0];
        let error = recorder
            .record_operation(
                Operation::GruCell {
                    input: ids[0],
                    weight: ids[1],
                    recurrence: ids[2],
                    hidden_state: ids[3],
                    hidden_size: 5,
                    options: Some(MLGruCellOptions::default()),
                    outputs: vec![output],
                },
                None,
            )
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("hiddenState must be rank 2, got rank 0")
        );
        assert_eq!(recorder.operands.len(), 4);
        assert!(recorder.operations.is_empty());
    }

    #[test]
    fn missing_or_forward_input_is_rejected_atomically() {
        let mut recorder = GraphRecorder::new();
        let error = recorder
            .record_operation(
                Operation::Identity {
                    input: 1,
                    options: None,
                    outputs: vec![0],
                },
                None,
            )
            .unwrap_err();
        assert!(matches!(error, GraphBuilderError::InvalidOperand(_)));
        assert!(recorder.operands.is_empty());
        assert!(recorder.operations.is_empty());
    }

    #[test]
    fn inferred_output_count_mismatch_is_atomic() {
        let mut recorder = GraphRecorder::new();
        let input = recorder.add_input("input".to_string(), descriptor(&[4]));
        let error = recorder
            .record_operation(
                Operation::Split {
                    input,
                    splits: vec![2, 2],
                    split_equal_parts: None,
                    options: Some(MLSplitOptions::default()),
                    outputs: vec![1],
                },
                None,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            GraphBuilderError::InconsistentGraphInfo { .. }
        ));
        assert_eq!(recorder.operands.len(), 1);
        assert!(recorder.operations.is_empty());
    }

    #[test]
    fn every_multi_output_family_records_all_inferred_descriptors() {
        let mut split = GraphRecorder::new();
        let input = split.add_input("input".to_string(), descriptor(&[4]));
        let split_outputs = split
            .record_operation(
                Operation::Split {
                    input,
                    splits: vec![1, 3],
                    split_equal_parts: None,
                    options: Some(MLSplitOptions::default()),
                    outputs: vec![1, 2],
                },
                None,
            )
            .unwrap();
        assert_eq!(split_outputs.len(), 2);
        assert_eq!(
            split.operands[1].descriptor.shape,
            to_dimension_vector(&[1])
        );
        assert_eq!(
            split.operands[2].descriptor.shape,
            to_dimension_vector(&[3])
        );

        let mut gru = GraphRecorder::new();
        let ids = named_inputs(&mut gru, &[&[3, 2, 4], &[1], &[1]]);
        let gru_options = MLGruOptions {
            return_sequence: true,
            ..Default::default()
        };
        let gru_outputs = gru
            .record_operation(
                Operation::Gru {
                    input: ids[0],
                    weight: ids[1],
                    recurrence: ids[2],
                    steps: 3,
                    hidden_size: 5,
                    options: Some(gru_options),
                    outputs: vec![3, 4],
                },
                None,
            )
            .unwrap();
        assert_eq!(gru_outputs.len(), 2);
        assert_eq!(
            gru.operands[3].descriptor.shape,
            to_dimension_vector(&[1, 2, 5])
        );
        assert_eq!(
            gru.operands[4].descriptor.shape,
            to_dimension_vector(&[3, 1, 2, 5])
        );

        let mut lstm = GraphRecorder::new();
        let ids = named_inputs(&mut lstm, &[&[3, 2, 4], &[1], &[1]]);
        let lstm_options = MLLstmOptions {
            return_sequence: true,
            ..Default::default()
        };
        let lstm_outputs = lstm
            .record_operation(
                Operation::Lstm {
                    input: ids[0],
                    weight: ids[1],
                    recurrence: ids[2],
                    steps: 3,
                    hidden_size: 5,
                    options: Some(lstm_options),
                    outputs: vec![3, 4, 5],
                },
                None,
            )
            .unwrap();
        assert_eq!(lstm_outputs.len(), 3);
        assert_eq!(
            lstm.operands[3].descriptor.shape,
            to_dimension_vector(&[1, 2, 5])
        );
        assert_eq!(
            lstm.operands[4].descriptor.shape,
            to_dimension_vector(&[1, 2, 5])
        );
        assert_eq!(
            lstm.operands[5].descriptor.shape,
            to_dimension_vector(&[3, 1, 2, 5])
        );

        let mut lstm_cell = GraphRecorder::new();
        let ids = named_inputs(&mut lstm_cell, &[&[2, 4], &[1], &[1], &[2, 5], &[2, 5]]);
        let outputs = lstm_cell
            .record_operation(
                Operation::LstmCell {
                    input: ids[0],
                    weight: ids[1],
                    recurrence: ids[2],
                    hidden_state: ids[3],
                    cell_state: ids[4],
                    hidden_size: 5,
                    options: Default::default(),
                    outputs: vec![5, 6],
                },
                None,
            )
            .unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(
            lstm_cell.operands[5].descriptor.shape,
            to_dimension_vector(&[2, 5])
        );
        assert_eq!(
            lstm_cell.operands[6].descriptor.shape,
            to_dimension_vector(&[2, 5])
        );
        assert_eq!(
            lstm_cell.operands[5].descriptor.data_type,
            DataType::Float32
        );
        assert_eq!(
            lstm_cell.operands[6].descriptor.data_type,
            DataType::Float32
        );
    }
}

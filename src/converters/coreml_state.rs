/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Carried-state pairs for the CoreML stateful lowering.
//!
//! A recurrent model carries state from one step to the next: a graph input holds the state at
//! the start of a step and a graph output holds it afterwards. Without this module the caller
//! copies (or re-binds) that output into the next step's input. With a carried-state pair the
//! CoreML converter keeps the value inside the compiled model as a CoreML `MLState`
//! (`read_state` / `write_state`, MIL opset CoreML8, iOS 18 / macOS 15), and the pair leaves the
//! model's feature description.
//!
//! Pairs come from [`CoremlOptions::carried_state_pairs`] (explicit input/output names, mirroring
//! the `kvCachePairs` of the WebNN stateful-model explainer) and, when
//! [`CoremlOptions::carry_state_automatically`] is set, from the naming conventions accepted by
//! [`candidate_output_names`]. Both switches default to off, so existing graphs are lowered exactly
//! as before.
//!
//! A pair must connect a graph input and a graph output with the same `float16` or `float32`
//! data type and the same static shape. CoreML stores every state as `float16`: a `float32`
//! pair is read through a cast to `float32` and written through a cast to `float16`, which
//! rounds the carried value once per step. Every state starts as zeros when the graph is built
//! and after [`crate::mlcontext::MLGraph::rustnn_reset_state`]; an initial value cannot be
//! supplied through the pair.

use crate::converters::operand_name;
use crate::error::GraphError;
use crate::graph::{DataType, Dimension, GraphInfo};
use crate::mlcontextoptions::CoremlOptions;
use std::collections::{HashMap, HashSet};

/// One graph input whose value is carried from the matching graph output of the previous step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CarriedStatePair {
    /// Operand index of the graph input (the state at the start of a step).
    pub input: u32,
    /// Operand index of the graph output (the state after the step).
    pub output: u32,
}

impl CarriedStatePair {
    /// Name of the input operand; also the name of the CoreML state.
    pub fn input_name(&self, graph: &GraphInfo) -> String {
        operand_name(graph, self.input)
    }

    /// Name of the output operand.
    pub fn output_name(&self, graph: &GraphInfo) -> String {
        operand_name(graph, self.output)
    }
}

/// Output names that automatic pairing accepts for a graph input named `input`, in priority
/// order.
///
/// Conventions: `X` / `next_X` (which covers `state_N` / `next_state_N` and `conv_N` /
/// `next_conv_N`), `past_X` / `present_X`, and `X_in` / `X_out`.
pub fn candidate_output_names(input: &str) -> Vec<String> {
    let mut names = vec![format!("next_{input}")];
    if let Some(rest) = input.strip_prefix("past_") {
        names.push(format!("present_{rest}"));
    }
    if let Some(stem) = input.strip_suffix("_in") {
        names.push(format!("{stem}_out"));
    }
    names
}

fn conversion_error(reason: String) -> GraphError {
    GraphError::ConversionFailed {
        format: "coreml_mlprogram".to_string(),
        reason,
    }
}

fn static_shape(graph: &GraphInfo, id: u32) -> Option<Vec<u32>> {
    graph
        .operand(id)?
        .descriptor
        .shape
        .iter()
        .map(|dimension| match dimension {
            Dimension::Static(size) => Some(*size),
            _ => None,
        })
        .collect()
}

/// Why `input` and `output` cannot form a carried-state pair, or `None` when they can.
fn incompatibility(graph: &GraphInfo, input: u32, output: u32) -> Option<String> {
    let input_operand = graph.operand(input)?;
    let output_operand = graph.operand(output)?;
    let data_type = input_operand.descriptor.data_type;
    if data_type != output_operand.descriptor.data_type {
        return Some(format!(
            "data types differ ({:?} vs {:?})",
            data_type, output_operand.descriptor.data_type
        ));
    }
    if !matches!(data_type, DataType::Float16 | DataType::Float32) {
        return Some(format!(
            "data type {data_type:?} cannot be carried; CoreML states hold float16 or float32 values"
        ));
    }
    match (static_shape(graph, input), static_shape(graph, output)) {
        (Some(a), Some(b)) if a == b => None,
        (Some(a), Some(b)) => Some(format!("shapes differ ({a:?} vs {b:?})")),
        _ => Some("state shapes must be static".to_string()),
    }
}

/// Validate one pair for the converter: both operands must exist as a graph input and a graph
/// output and be compatible (see the module documentation).
pub(crate) fn check_pair(graph: &GraphInfo, pair: &CarriedStatePair) -> Result<(), GraphError> {
    let input_name = pair.input_name(graph);
    let output_name = pair.output_name(graph);
    if !graph.input_operands.contains(&pair.input) {
        return Err(conversion_error(format!(
            "carried state pair ({input_name}, {output_name}): operand {} is not a graph input",
            pair.input
        )));
    }
    if !graph.output_operands.contains(&pair.output) {
        return Err(conversion_error(format!(
            "carried state pair ({input_name}, {output_name}): operand {} is not a graph output",
            pair.output
        )));
    }
    match incompatibility(graph, pair.input, pair.output) {
        Some(reason) => Err(conversion_error(format!(
            "carried state pair ({input_name}, {output_name}): {reason}"
        ))),
        None => Ok(()),
    }
}

fn named_operands(graph: &GraphInfo, ids: &[u32]) -> HashMap<String, u32> {
    ids.iter()
        .map(|&id| (operand_name(graph, id), id))
        .collect()
}

/// Carried-state pairs of `graph` under `options`: the explicit pairs first, in option order,
/// then the automatically detected pairs in graph input order.
///
/// Explicit pairs that name unknown operands, duplicate an operand or combine incompatible
/// operands are errors. Automatic candidates that are incompatible are skipped with a
/// `debug` log record; accepted pairs are logged at `info`. Returns an empty vector when both
/// switches are off.
pub fn carried_state_pairs(
    graph: &GraphInfo,
    options: &CoremlOptions,
) -> Result<Vec<CarriedStatePair>, GraphError> {
    let inputs = named_operands(graph, &graph.input_operands);
    let outputs = named_operands(graph, &graph.output_operands);
    let mut pairs = Vec::new();
    let mut used_inputs = HashSet::new();
    let mut used_outputs = HashSet::new();

    for (input_name, output_name) in &options.carried_state_pairs {
        let &input = inputs.get(input_name).ok_or_else(|| {
            conversion_error(format!(
                "carried state pair ({input_name}, {output_name}): '{input_name}' is not a graph input"
            ))
        })?;
        let &output = outputs.get(output_name).ok_or_else(|| {
            conversion_error(format!(
                "carried state pair ({input_name}, {output_name}): '{output_name}' is not a graph output"
            ))
        })?;
        if !used_inputs.insert(input) || !used_outputs.insert(output) {
            return Err(conversion_error(format!(
                "carried state pair ({input_name}, {output_name}) reuses an operand of another pair"
            )));
        }
        if let Some(reason) = incompatibility(graph, input, output) {
            return Err(conversion_error(format!(
                "carried state pair ({input_name}, {output_name}): {reason}"
            )));
        }
        pairs.push(CarriedStatePair { input, output });
    }

    if options.carry_state_automatically {
        for &input in &graph.input_operands {
            if used_inputs.contains(&input) {
                continue;
            }
            let input_name = operand_name(graph, input);
            for output_name in candidate_output_names(&input_name) {
                let Some(&output) = outputs.get(&output_name) else {
                    continue;
                };
                if used_outputs.contains(&output) {
                    continue;
                }
                if let Some(reason) = incompatibility(graph, input, output) {
                    log::debug!(
                        "CoreML carried state: not pairing '{input_name}' with '{output_name}': {reason}"
                    );
                    continue;
                }
                used_inputs.insert(input);
                used_outputs.insert(output);
                pairs.push(CarriedStatePair { input, output });
                break;
            }
        }
    }

    for pair in &pairs {
        let operand = graph.operand(pair.input).expect("validated above");
        log::info!(
            "CoreML carried state: '{}' -> '{}' ({:?} {:?}) kept in the model as an MLState",
            pair.input_name(graph),
            pair.output_name(graph),
            operand.descriptor.data_type,
            static_shape(graph, pair.input).unwrap_or_default()
        );
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Operand, OperandDescriptor, OperandKind, to_dimension_vector};

    fn graph(
        inputs: &[(&str, DataType, &[u32])],
        outputs: &[(&str, DataType, &[u32])],
    ) -> GraphInfo {
        let mut graph = GraphInfo::default();
        for (name, data_type, shape) in inputs {
            graph.input_operands.push(graph.operands.len() as u32);
            graph.operands.push(Operand {
                kind: OperandKind::Input,
                name: Some(name.to_string()),
                descriptor: OperandDescriptor {
                    data_type: *data_type,
                    shape: to_dimension_vector(shape),
                    pending_permutation: vec![],
                },
            });
        }
        for (name, data_type, shape) in outputs {
            graph.output_operands.push(graph.operands.len() as u32);
            graph.operands.push(Operand {
                kind: OperandKind::Output,
                name: Some(name.to_string()),
                descriptor: OperandDescriptor {
                    data_type: *data_type,
                    shape: to_dimension_vector(shape),
                    pending_permutation: vec![],
                },
            });
        }
        graph
    }

    fn names(graph: &GraphInfo, pairs: &[CarriedStatePair]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|pair| (pair.input_name(graph), pair.output_name(graph)))
            .collect()
    }

    fn automatic() -> CoremlOptions {
        CoremlOptions {
            carry_state_automatically: true,
            ..Default::default()
        }
    }

    #[test]
    fn switches_default_to_off_and_pair_nothing() {
        let graph = graph(
            &[("state_0", DataType::Float32, &[1, 4])],
            &[("next_state_0", DataType::Float32, &[1, 4])],
        );
        assert!(
            carried_state_pairs(&graph, &CoremlOptions::default())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn every_naming_convention_is_detected() {
        let graph = graph(
            &[
                ("token", DataType::Int32, &[1, 1]),
                ("state_0", DataType::Float32, &[1, 1536, 16]),
                ("conv_0", DataType::Float32, &[1, 1536, 4]),
                ("past_key_values", DataType::Float16, &[1, 3, 512, 64]),
                ("hidden_in", DataType::Float16, &[2, 8]),
                ("key_7", DataType::Float16, &[1, 3, 512, 64]),
            ],
            &[
                ("logits", DataType::Float32, &[1, 50280]),
                ("next_state_0", DataType::Float32, &[1, 1536, 16]),
                ("next_conv_0", DataType::Float32, &[1, 1536, 4]),
                ("present_key_values", DataType::Float16, &[1, 3, 512, 64]),
                ("hidden_out", DataType::Float16, &[2, 8]),
                ("next_key_7", DataType::Float16, &[1, 3, 512, 64]),
            ],
        );
        let pairs = carried_state_pairs(&graph, &automatic()).unwrap();
        assert_eq!(
            names(&graph, &pairs),
            vec![
                ("state_0".into(), "next_state_0".into()),
                ("conv_0".into(), "next_conv_0".into()),
                ("past_key_values".into(), "present_key_values".into()),
                ("hidden_in".into(), "hidden_out".into()),
                ("key_7".into(), "next_key_7".into()),
            ]
        );
    }

    #[test]
    fn mismatched_candidates_are_not_paired() {
        // dtype mismatch, shape mismatch, integer dtype, and a convention miss.
        let graph = graph(
            &[
                ("a", DataType::Float32, &[1, 4]),
                ("b", DataType::Float32, &[1, 4]),
                ("c", DataType::Int32, &[1, 4]),
                ("d", DataType::Float32, &[1, 4]),
            ],
            &[
                ("next_a", DataType::Float16, &[1, 4]),
                ("next_b", DataType::Float32, &[1, 5]),
                ("next_c", DataType::Int32, &[1, 4]),
                ("d_next", DataType::Float32, &[1, 4]),
            ],
        );
        assert!(
            carried_state_pairs(&graph, &automatic())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn explicit_pairs_are_honoured_and_validated() {
        let graph = graph(
            &[
                ("h", DataType::Float16, &[1, 8]),
                ("state_0", DataType::Float32, &[1, 4]),
            ],
            &[
                ("h_after", DataType::Float16, &[1, 8]),
                ("next_state_0", DataType::Float32, &[1, 4]),
                ("y", DataType::Float32, &[1, 4]),
            ],
        );
        let explicit = CoremlOptions {
            carried_state_pairs: vec![("h".into(), "h_after".into())],
            ..Default::default()
        };
        let pairs = carried_state_pairs(&graph, &explicit).unwrap();
        assert_eq!(names(&graph, &pairs), vec![("h".into(), "h_after".into())]);

        // Explicit pairs combine with automatic detection; explicit ones come first.
        let both = CoremlOptions {
            carried_state_pairs: vec![("h".into(), "h_after".into())],
            carry_state_automatically: true,
            ..Default::default()
        };
        let pairs = carried_state_pairs(&graph, &both).unwrap();
        assert_eq!(
            names(&graph, &pairs),
            vec![
                ("h".into(), "h_after".into()),
                ("state_0".into(), "next_state_0".into()),
            ]
        );

        for (input, output) in [
            ("missing", "h_after"),
            ("h", "missing"),
            ("h", "y"),
            ("state_0", "h_after"),
        ] {
            let options = CoremlOptions {
                carried_state_pairs: vec![(input.into(), output.into())],
                ..Default::default()
            };
            assert!(
                carried_state_pairs(&graph, &options).is_err(),
                "{input} -> {output}"
            );
        }
        let duplicate = CoremlOptions {
            carried_state_pairs: vec![
                ("h".into(), "h_after".into()),
                ("h".into(), "h_after".into()),
            ],
            ..Default::default()
        };
        assert!(carried_state_pairs(&graph, &duplicate).is_err());
    }

    #[test]
    fn candidate_names_follow_the_documented_conventions() {
        assert_eq!(candidate_output_names("state_3"), vec!["next_state_3"]);
        assert_eq!(
            candidate_output_names("past_k"),
            vec!["next_past_k", "present_k"]
        );
        assert_eq!(candidate_output_names("h_in"), vec!["next_h_in", "h_out"]);
    }
}

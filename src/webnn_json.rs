/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! Conversion between [`GraphInfo`] and the `webnn-graph` JSON AST (`GraphJson`).
//!
//! [`from_graph_json`] imports a parsed `.webnn` or JSON graph, inlining constants and
//! inferring missing shapes; [`to_graph_json`] exports a graph for serialization. Both keep
//! the `quantized` marker of the interchange format.

use crate::error::GraphError;
use crate::graph::{
    ConstantData, DataType, Dimension, DynamicDimension, GraphInfo, Operand, OperandDescriptor,
    OperandKind, to_dimension_vector,
};
use crate::graph_recorder::GraphRecorder;
use crate::operators::Operation;
use std::collections::{BTreeMap, HashMap, HashSet};
use webnn_graph::ast::{ConstDecl, ConstInit, GraphJson, Node, OperandDesc};

/// The name used for an operand in the exported AST (and, for constants, as the safetensors tensor
/// key and `@weights(...)` reference). Falls back to `operand_<idx>` when the operand is unnamed.
///
/// Shared by the AST builder and the `.webnn`/`.safetensors` exporter so both agree on the scheme.
pub(crate) fn operand_export_name(operand: &Operand, idx: usize) -> String {
    operand
        .name
        .clone()
        .unwrap_or_else(|| format!("operand_{}", idx))
}

/// Maps operand id -> the name it should be exported under, overriding [`operand_export_name`].
///
/// Used by the builder's save path to name graph outputs without mutating the operands: the same
/// override must apply everywhere the operand is referenced (node inputs, node outputs, and the
/// graph `outputs` section) so all references stay consistent.
pub(crate) type OutputNameOverrides = HashMap<u32, String>;

/// Like [`operand_export_name`], but consults `overrides` first so callers can rename operands
/// (e.g. mark them as named graph outputs) without touching the `GraphInfo`.
fn operand_export_name_with_overrides(
    operand: &Operand,
    idx: usize,
    overrides: Option<&OutputNameOverrides>,
) -> String {
    if let Some(name) = overrides.and_then(|o| o.get(&(idx as u32))) {
        return name.clone();
    }
    operand_export_name(operand, idx)
}

/// Convert our DataType to webnn-graph DataType
fn to_webnn_datatype(dt: &DataType) -> webnn_graph::ast::DataType {
    match dt {
        DataType::Int4 => webnn_graph::ast::DataType::Int4,
        DataType::Uint4 => webnn_graph::ast::DataType::Uint4,
        DataType::Float32 => webnn_graph::ast::DataType::Float32,
        DataType::Float16 => webnn_graph::ast::DataType::Float16,
        DataType::Int32 => webnn_graph::ast::DataType::Int32,
        DataType::Uint32 => webnn_graph::ast::DataType::Uint32,
        DataType::Int64 => webnn_graph::ast::DataType::Int64,
        DataType::Uint64 => webnn_graph::ast::DataType::Uint64,
        DataType::Int8 => webnn_graph::ast::DataType::Int8,
        DataType::Uint8 => webnn_graph::ast::DataType::Uint8,
    }
}

/// Convert webnn-graph DataType to our DataType
fn from_webnn_datatype(dt: &webnn_graph::ast::DataType) -> DataType {
    match dt {
        webnn_graph::ast::DataType::Float32 => DataType::Float32,
        webnn_graph::ast::DataType::Float16 => DataType::Float16,
        webnn_graph::ast::DataType::Int4 => DataType::Int4,
        webnn_graph::ast::DataType::Uint4 => DataType::Uint4,
        webnn_graph::ast::DataType::Int32 => DataType::Int32,
        webnn_graph::ast::DataType::Uint32 => DataType::Uint32,
        webnn_graph::ast::DataType::Int64 => DataType::Int64,
        webnn_graph::ast::DataType::Uint64 => DataType::Uint64,
        webnn_graph::ast::DataType::Int8 => DataType::Int8,
        webnn_graph::ast::DataType::Uint8 => DataType::Uint8,
    }
}

fn to_webnn_dimension(dim: &Dimension) -> webnn_graph::ast::Dimension {
    match dim {
        Dimension::Static(v) => webnn_graph::ast::Dimension::Static(*v),
        Dimension::Dynamic(d) => {
            webnn_graph::ast::Dimension::Dynamic(webnn_graph::ast::DynamicDimension {
                name: d.name.clone(),
                max_size: d.max_size,
            })
        }
    }
}

fn from_webnn_dimension(dim: &webnn_graph::ast::Dimension) -> Dimension {
    match dim {
        webnn_graph::ast::Dimension::Static(v) => Dimension::Static(*v),
        webnn_graph::ast::Dimension::Dynamic(d) => Dimension::Dynamic(DynamicDimension {
            name: d.name.clone(),
            max_size: d.max_size,
        }),
    }
}

/// Build one [`webnn_graph::ast::Node`] for the operation at `op_index`, matching the shape produced by [`to_graph_json`].
///
/// Useful for diagnostics (for example ONNX conversion failures) without serializing the full graph.
pub fn graph_operation_to_webnn_node(
    graph: &GraphInfo,
    op_index: usize,
) -> Result<Node, GraphError> {
    graph_operation_to_webnn_node_with_overrides(graph, op_index, None)
}

/// Same as [`graph_operation_to_webnn_node`], but operand names can be overridden (see
/// [`OutputNameOverrides`]). Used by the save path to name graph outputs without mutating operands.
fn graph_operation_to_webnn_node_with_overrides(
    graph: &GraphInfo,
    op_index: usize,
    overrides: Option<&OutputNameOverrides>,
) -> Result<Node, GraphError> {
    let operation = graph
        .operations
        .get(op_index)
        .ok_or_else(|| GraphError::ConversionFailed {
            format: "webnn-graph-json".to_string(),
            reason: format!("operation index {op_index} out of range"),
        })?;
    let id = format!("op_{}", op_index);

    let input_names: Vec<String> = operation
        .input_operands()
        .iter()
        .map(|&idx| {
            operand_export_name_with_overrides(
                &graph.operands[idx as usize],
                idx as usize,
                overrides,
            )
        })
        .collect();

    let output_operands = operation.output_operands_slice();
    let output_names: Option<Vec<String>> = if output_operands.is_empty() {
        None
    } else {
        Some(
            output_operands
                .iter()
                .map(|&idx| {
                    operand_export_name_with_overrides(
                        &graph.operands[idx as usize],
                        idx as usize,
                        overrides,
                    )
                })
                .collect(),
        )
    };

    let mut options: serde_json::Map<String, serde_json::Value> = operation
        .attributes_json_value()
        .as_object()
        .cloned()
        .unwrap_or_else(serde_json::Map::new);
    options.remove("kind");
    // Operand-valued options cannot be serialized as raw graph-local indices: parsing
    // declarations and operation outputs can assign different indices on reload. Store
    // their stable exported names and resolve those names after parsing instead.
    let mut name_operand_option = |key: &str, operand_id: Option<u32>| {
        if let Some(operand_id) = operand_id {
            options.insert(
                key.to_string(),
                serde_json::Value::String(operand_export_name_with_overrides(
                    &graph.operands[operand_id as usize],
                    operand_id as usize,
                    overrides,
                )),
            );
        }
    };
    match operation {
        Operation::BatchNormalization {
            options: Some(o), ..
        } => {
            name_operand_option("scale", o.scale);
            name_operand_option("bias", o.bias);
        }
        Operation::Conv2d {
            options: Some(o), ..
        } => {
            name_operand_option("bias", o.bias);
        }
        Operation::ConvTranspose2d {
            options: Some(o), ..
        } => {
            name_operand_option("bias", o.bias);
        }
        Operation::Gemm {
            options: Some(o), ..
        } => name_operand_option("c", o.c),
        Operation::Gru {
            options: Some(o), ..
        } => {
            name_operand_option("bias", o.bias);
            name_operand_option("recurrentBias", o.recurrent_bias);
            name_operand_option("initialHiddenState", o.initial_hidden_state);
        }
        Operation::GruCell {
            options: Some(o), ..
        } => {
            name_operand_option("bias", o.bias);
            name_operand_option("recurrentBias", o.recurrent_bias);
        }
        Operation::InstanceNormalization {
            options: Some(o), ..
        } => {
            name_operand_option("scale", o.scale);
            name_operand_option("bias", o.bias);
        }
        Operation::LayerNormalization {
            options: Some(o), ..
        } => {
            name_operand_option("scale", o.scale);
            name_operand_option("bias", o.bias);
        }
        Operation::Lstm {
            options: Some(o), ..
        } => {
            name_operand_option("bias", o.bias);
            name_operand_option("recurrentBias", o.recurrent_bias);
            name_operand_option("peepholeWeight", o.peephole_weight);
            name_operand_option("initialHiddenState", o.initial_hidden_state);
            name_operand_option("initialCellState", o.initial_cell_state);
        }
        Operation::LstmCell {
            options: Some(o), ..
        } => {
            name_operand_option("bias", o.bias);
            name_operand_option("recurrentBias", o.recurrent_bias);
            name_operand_option("peepholeWeight", o.peephole_weight);
        }
        _ => {}
    }

    Ok(Node {
        id,
        op: operation.op_type().to_string(),
        inputs: input_names,
        options,
        outputs: output_names,
    })
}

/// Controls how constant operands are represented when building the AST.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConstExport {
    /// Embed constant bytes directly in the AST as `ConstInit::InlineBytes` (clones the data).
    Inline,
    /// Emit a `ConstInit::Weights { ref }` reference instead, leaving the bytes out of the AST.
    /// Callers are responsible for writing the weights out (e.g. to a sidecar safetensors file).
    ExternalWeights,
}

/// Convert GraphInfo to GraphJson, embedding constant data inline.
pub fn to_graph_json(graph: &GraphInfo, quantized: bool) -> Result<GraphJson, GraphError> {
    to_graph_json_with_consts(graph, quantized, ConstExport::Inline, None)
}

/// Convert GraphInfo to GraphJson, choosing how constants are represented.
///
/// With [`ConstExport::ExternalWeights`] the constant bytes are not copied into the AST; each
/// constant becomes a `@weights(...)` reference keyed by the same name used here (`operand.name`,
/// else `operand_<idx>`). This is the zero-copy path used by the `.webnn`/`.safetensors` exporter.
///
/// When `output_override` is `Some`, the graph `outputs` section (and the corresponding operand
/// names in nodes) is taken from the override map instead of `graph.output_operands`. This lets the
/// builder export a graph's outputs without mutating operand kinds/names first. When `None`, outputs
/// are read from `graph.output_operands` using each operand's own name.
pub(crate) fn to_graph_json_with_consts(
    graph: &GraphInfo,
    quantized: bool,
    const_export: ConstExport,
    output_override: Option<&OutputNameOverrides>,
) -> Result<GraphJson, GraphError> {
    let mut inputs = BTreeMap::new();
    let mut consts = BTreeMap::new();
    let mut nodes = Vec::new();
    let mut outputs = BTreeMap::new();

    // Process operands - separate inputs and constants
    for (idx, operand) in graph.operands.iter().enumerate() {
        let name = operand_export_name(operand, idx);

        match &operand.kind {
            OperandKind::Input => {
                inputs.insert(
                    name,
                    OperandDesc {
                        data_type: to_webnn_datatype(&operand.descriptor.data_type),
                        shape: operand
                            .descriptor
                            .shape
                            .iter()
                            .map(to_webnn_dimension)
                            .collect(),
                    },
                );
            }
            OperandKind::Constant => {
                // Get constant data from the map
                if let Some(constant) = graph.constant_operand_ids_to_handles.get(&(idx as u32)) {
                    let const_shape =
                        operand
                            .descriptor
                            .static_shape()
                            .ok_or(GraphError::ConversionFailed {
                                format: "webnn-graph-json".to_string(),
                                reason: format!(
                                    "constant operand {} has dynamic shape",
                                    operand.name.as_deref().unwrap_or("unknown")
                                ),
                            })?;
                    let init = match const_export {
                        ConstExport::Inline => ConstInit::InlineBytes {
                            bytes: constant.data.clone(),
                        },
                        ConstExport::ExternalWeights => ConstInit::Weights {
                            r#ref: name.clone(),
                        },
                    };

                    consts.insert(
                        name,
                        ConstDecl {
                            data_type: to_webnn_datatype(&operand.descriptor.data_type),
                            shape: const_shape,
                            init,
                        },
                    );
                }
            }
            OperandKind::Intermediate => {
                // Intermediates are not in graph json
            }
            OperandKind::Output => {
                // Outputs are handled separately below
            }
        }
    }

    // Process operations
    for op_idx in 0..graph.operations.len() {
        nodes.push(graph_operation_to_webnn_node_with_overrides(
            graph,
            op_idx,
            output_override,
        )?);
    }

    // Process outputs. Prefer the caller-supplied override (builder save path) so we don't need to
    // mutate operands; otherwise read the finalized `graph.output_operands`.
    match output_override {
        Some(override_names) => {
            for (&operand_idx, name) in override_names {
                if graph.operands.get(operand_idx as usize).is_some() {
                    outputs.insert(name.clone(), name.clone());
                }
            }
        }
        None => {
            for &operand_idx in &graph.output_operands {
                if let Some(operand) = graph.operands.get(operand_idx as usize) {
                    let name = operand_export_name(operand, operand_idx as usize);
                    outputs.insert(name.clone(), name);
                }
            }
        }
    }

    Ok(GraphJson {
        name: Some("graph".to_string()),
        format: "webnn-graph-json".to_string(),
        version: 2,
        quantized: graph.quantized || quantized,
        inputs,
        consts,
        nodes,
        outputs,
    })
}

/// Convert GraphJson to GraphInfo using the same recorder and shape inference as
/// the public `MLGraphBuilder` API.
pub fn from_graph_json(graph_json: &GraphJson) -> Result<GraphInfo, GraphError> {
    from_graph_json_owned(graph_json.clone())
}

/// Converts an owned GraphJson without cloning inline constant buffers.
pub fn from_graph_json_owned(graph_json: GraphJson) -> Result<GraphInfo, GraphError> {
    fn conversion_error(reason: impl Into<String>) -> GraphError {
        GraphError::ConversionFailed {
            format: "webnn-graph-json".to_string(),
            reason: reason.into(),
        }
    }

    fn ensure_new_name(
        operand_map: &BTreeMap<String, u32>,
        name: &str,
        declaration: &str,
    ) -> Result<(), GraphError> {
        if operand_map.contains_key(name) {
            return Err(conversion_error(format!(
                "{declaration} '{name}' conflicts with an existing operand"
            )));
        }
        Ok(())
    }

    let GraphJson {
        quantized,
        inputs,
        consts,
        nodes,
        outputs,
        ..
    } = graph_json;
    let mut recorder = GraphRecorder::new();
    recorder.set_quantized(quantized);
    let mut operand_map: BTreeMap<String, u32> = BTreeMap::new();

    for (name, desc) in &inputs {
        ensure_new_name(&operand_map, name, "input")?;
        let id = recorder.add_input(
            name.clone(),
            OperandDescriptor {
                data_type: from_webnn_datatype(&desc.data_type),
                shape: desc.shape.iter().map(from_webnn_dimension).collect(),
                pending_permutation: Vec::new(),
            },
        );
        operand_map.insert(name.clone(), id);
    }

    for (name, const_decl) in consts {
        ensure_new_name(&operand_map, &name, "constant")?;
        let data = match const_decl.init {
            ConstInit::InlineBytes { bytes } => bytes,
            // External tensor data is populated by the safetensors loader after parsing.
            ConstInit::Weights { r#ref: _ } => Vec::new(),
            ConstInit::Scalar { value } => {
                let element_count: usize =
                    const_decl.shape.iter().map(|&size| size as usize).product();
                let data_type = from_webnn_datatype(&const_decl.data_type);
                let scalar = value
                    .as_f64()
                    .map(|value| value as f32)
                    .or_else(|| value.as_i64().map(|value| value as f32))
                    .or_else(|| value.as_u64().map(|value| value as f32))
                    .ok_or_else(|| {
                        conversion_error(format!("Cannot parse scalar value: {value:?}"))
                    })?;

                let mut bytes = Vec::new();
                for _ in 0..element_count {
                    match data_type {
                        DataType::Int4 | DataType::Uint4 => {
                            return Err(conversion_error(
                                "int4/uint4 constants are not supported in scalar export",
                            ));
                        }
                        DataType::Float32 => bytes.extend_from_slice(&scalar.to_le_bytes()),
                        DataType::Float16 => bytes.extend_from_slice(
                            &half::f16::from_f32(scalar).to_bits().to_le_bytes(),
                        ),
                        DataType::Int32 => bytes.extend_from_slice(&(scalar as i32).to_le_bytes()),
                        DataType::Uint32 => bytes.extend_from_slice(&(scalar as u32).to_le_bytes()),
                        DataType::Int64 => bytes.extend_from_slice(&(scalar as i64).to_le_bytes()),
                        DataType::Uint64 => bytes.extend_from_slice(&(scalar as u64).to_le_bytes()),
                        DataType::Int8 => bytes.push(scalar as i8 as u8),
                        DataType::Uint8 => bytes.push(scalar as u8),
                    }
                }
                bytes
            }
        };
        let id = recorder.add_constant(
            Some(name.clone()),
            OperandDescriptor {
                data_type: from_webnn_datatype(&const_decl.data_type),
                shape: to_dimension_vector(&const_decl.shape),
                pending_permutation: Vec::new(),
            },
            ConstantData { data, label: None },
            None,
        );
        operand_map.insert(name.clone(), id);
    }

    for node in &nodes {
        let input_ids = node
            .inputs
            .iter()
            .map(|name| {
                operand_map.get(name).copied().ok_or_else(|| {
                    conversion_error(format!(
                        "node '{}' ({}) references input '{name}' that was not found (missing or forward reference)",
                        node.id, node.op
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_names = node
            .outputs
            .clone()
            .unwrap_or_else(|| vec![node.id.clone()]);
        let mut names_in_node = HashSet::new();
        for name in &output_names {
            ensure_new_name(&operand_map, name, "operation output")?;
            if !names_in_node.insert(name) {
                return Err(conversion_error(format!(
                    "node '{}' ({}) declares duplicate output '{name}'",
                    node.id, node.op
                )));
            }
        }

        let mut resolved_options = node.options.clone();
        for key in [
            "scale",
            "bias",
            "c",
            "recurrentBias",
            "peepholeWeight",
            "initialHiddenState",
            "initialCellState",
        ] {
            let Some(serde_json::Value::String(name)) = resolved_options.get(key) else {
                continue;
            };
            let operand_id = operand_map.get(name).copied().ok_or_else(|| {
                conversion_error(format!(
                    "node '{}' ({}) operand option '{key}' references operand '{name}' that was not found (missing or forward reference)",
                    node.id, node.op
                ))
            })?;
            resolved_options.insert(key.to_string(), serde_json::Value::from(operand_id));
        }

        let output_ids = recorder.next_output_ids(output_names.len());
        let operation = Operation::from_json_attributes(
            &node.op,
            &input_ids,
            &output_ids,
            &serde_json::Value::Object(resolved_options),
        )
        .ok_or_else(|| {
            conversion_error(format!(
                "node '{}' has unknown operation '{}', an invalid operand count, or missing required attributes",
                node.id, node.op
            ))
        })?;
        let outputs = recorder
            .record_operation(operation, Some(&output_names))
            .map_err(|error| {
                conversion_error(format!(
                    "node '{}' ({}) could not be recorded: {error}",
                    node.id, node.op
                ))
            })?;
        for (name, operand) in output_names.into_iter().zip(outputs) {
            operand_map.insert(name, operand.id as u32);
        }
    }

    let mut marked_outputs = HashSet::new();
    for (binding_name, operand_ref) in &outputs {
        let id = operand_map.get(operand_ref).copied().ok_or_else(|| {
            conversion_error(format!(
                "graph output '{binding_name}' references unknown operand '{operand_ref}'"
            ))
        })?;
        if !marked_outputs.insert(id) {
            return Err(conversion_error(format!(
                "graph output '{binding_name}' aliases an operand already exported under another name"
            )));
        }
        recorder
            .mark_output(id, binding_name.clone())
            .map_err(|error| conversion_error(error.to_string()))?;
    }

    recorder
        .into_graph()
        .map_err(|error| conversion_error(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use webnn_graph::serialize::{SerializeOptions, serialize_graph_to_wg_text};

    fn wshape(shape: &[u32]) -> Vec<webnn_graph::ast::Dimension> {
        webnn_graph::ast::to_dimension_vector(shape)
    }

    fn ushape(shape: &[u32]) -> Vec<u32> {
        shape.to_vec()
    }

    #[test]
    fn loader_layer_normalization_preserves_input_shape() {
        let text = r#"
        webnn_graph "ln_test" v1 {
            inputs { x: f32[1, 64, 3072]; }
            nodes {
                [y] = layerNormalization(x, epsilon=1e-06);
            }
            outputs { y; }
        }"#;
        let graph_json = webnn_graph::parser::parse_wg_text(text).expect("parse");
        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");
        let y_idx = graph_info.output_operands[0];
        assert_eq!(
            graph_info.operands[y_idx as usize].descriptor.shape,
            to_dimension_vector(&[1, 64, 3072]),
            "layerNormalization output shape should match input"
        );
    }

    #[test]
    fn infer_shape_tensor_type_through_unsqueeze() {
        let text = r#"
        webnn_graph "shape_type_test" v2 {
            inputs { x: f32[2, 3, 4]; }
            nodes {
                shape_out = shape(x);
                expanded_shape = unsqueeze(shape_out, axes=[0]);
            }
            outputs { expanded_shape; }
        }"#;
        let graph_json = webnn_graph::parser::parse_wg_text(text).expect("parse");
        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");

        let shape_output = graph_info.operations[0]
            .output_operand()
            .expect("shape output");
        let unsqueeze_output = graph_info.operations[1]
            .output_operand()
            .expect("unsqueeze output");
        assert_eq!(
            graph_info.operands[shape_output as usize]
                .descriptor
                .data_type,
            DataType::Int64
        );
        assert_eq!(
            graph_info.operands[unsqueeze_output as usize]
                .descriptor
                .data_type,
            DataType::Int64
        );
        assert_eq!(
            graph_info.operands[unsqueeze_output as usize]
                .descriptor
                .shape,
            to_dimension_vector(&[1, 3])
        );
    }

    #[test]
    fn test_datatype_conversion() {
        let types = vec![
            DataType::Float32,
            DataType::Float16,
            DataType::Int32,
            DataType::Uint32,
            DataType::Int64,
            DataType::Uint64,
            DataType::Int8,
            DataType::Uint8,
            DataType::Int4,
            DataType::Uint4,
        ];

        for dt in types {
            let webnn_dt = to_webnn_datatype(&dt);
            let back = from_webnn_datatype(&webnn_dt);
            assert_eq!(dt, back);
        }
    }

    fn build_quantized_graph_info(dtype: DataType) -> GraphInfo {
        let descriptor = OperandDescriptor {
            data_type: dtype,
            shape: to_dimension_vector(&[2, 3]),
            pending_permutation: vec![],
        };
        GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: descriptor.clone(),
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor,
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![Operation::Identity {
                input: 0,
                options: None,
                outputs: vec![1],
            }],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        }
    }

    #[test]
    fn scalar_shape_arguments_roundtrip_and_remain_explicit() {
        for op_type in ["reshape", "expand", "tile"] {
            let operation = match op_type {
                "reshape" => Operation::Reshape {
                    input: 0,
                    new_shape: vec![],
                    options: None,
                    outputs: vec![1],
                },
                "expand" => Operation::Expand {
                    input: 0,
                    new_shape: vec![],
                    options: None,
                    outputs: vec![1],
                },
                "tile" => Operation::Tile {
                    input: 0,
                    repetitions: vec![],
                    options: None,
                    outputs: vec![1],
                },
                _ => unreachable!(),
            };
            let descriptor = OperandDescriptor {
                data_type: DataType::Float32,
                shape: vec![],
                pending_permutation: vec![],
            };
            let graph = GraphInfo {
                operands: vec![
                    Operand {
                        kind: OperandKind::Input,
                        descriptor: descriptor.clone(),
                        name: Some("x".to_string()),
                    },
                    Operand {
                        kind: OperandKind::Output,
                        descriptor,
                        name: Some("y".to_string()),
                    },
                ],
                input_operands: vec![0],
                output_operands: vec![1],
                operations: vec![operation],
                constant_operand_ids_to_handles: HashMap::new(),
                id_to_constant_tensor_operand_map: HashMap::new(),
                quantized: false,
            };

            let json = to_graph_json(&graph, false).expect("to_graph_json");
            let argument = if op_type == "tile" {
                "repetitions"
            } else {
                "newShape"
            };
            assert_eq!(
                json.nodes[0].options.get(argument),
                Some(&serde_json::json!([])),
                "{op_type} must serialize a scalar-rank argument explicitly"
            );
            let loaded = from_graph_json(&json).expect("from_graph_json");
            assert!(
                loaded.operands[loaded.output_operands[0] as usize]
                    .descriptor
                    .shape
                    .is_empty()
            );

            let mut missing = json;
            missing.nodes[0].options.remove(argument);
            let error = from_graph_json(&missing).unwrap_err();
            assert!(
                error.to_string().contains("missing required attributes"),
                "unexpected {op_type} error: {error}"
            );
        }
    }

    #[test]
    fn shape_vector_arguments_distinguish_missing_from_explicit_empty() {
        for (op_type, attributes) in [
            ("reshape", serde_json::json!({ "newShape": [] })),
            ("expand", serde_json::json!({ "newShape": [] })),
            ("tile", serde_json::json!({ "repetitions": [] })),
            (
                "pad",
                serde_json::json!({ "beginningPadding": [], "endingPadding": [] }),
            ),
            ("slice", serde_json::json!({ "starts": [], "sizes": [] })),
        ] {
            assert!(
                Operation::from_json_attributes(op_type, &[0], &[1], &attributes).is_some(),
                "{op_type} must retain an explicitly empty vector"
            );
            assert!(
                Operation::from_json_attributes(op_type, &[0], &[1], &serde_json::json!({}))
                    .is_none(),
                "{op_type} must reject a missing required vector"
            );
        }
    }

    #[test]
    fn quantized_flag_roundtrips_json() {
        let graph = build_quantized_graph_info(DataType::Int8);
        let json = to_graph_json(&graph, false).expect("to_graph_json");
        assert!(json.quantized);

        let graph_from_json = from_graph_json(&json).expect("from_graph_json");
        assert!(graph_from_json.quantized);
        assert_eq!(
            graph_from_json.operands[0].descriptor.data_type,
            DataType::Int8
        );
        assert_eq!(graph_from_json.output_operands, vec![1]);

        // Passing quantized=true should also set the flag even if the graph info is false.
        let mut graph_not_marked = graph.clone();
        graph_not_marked.quantized = false;
        let json_explicit = to_graph_json(&graph_not_marked, true).expect("to_graph_json");
        assert!(json_explicit.quantized);
    }

    #[test]
    fn quantized_flag_roundtrips_text() {
        let graph = build_quantized_graph_info(DataType::Uint4);
        let graph_json = to_graph_json(&graph, true).expect("to_graph_json");
        let text = serialize_graph_to_wg_text(&graph_json, SerializeOptions { quantized: true })
            .expect("serialize to text");
        let parsed = webnn_graph::parser::parse_wg_text(&text).expect("parse text");
        assert!(
            parsed.quantized,
            "text serialization preserves quantized marker"
        );

        let graph_info = from_graph_json(&parsed).expect("graph from text");
        assert!(graph_info.quantized);
        assert_eq!(graph_info.operands[0].descriptor.data_type, DataType::Uint4);
    }

    #[test]
    fn test_to_graph_json_with_constants() {
        // Test conversion with constant operands
        let constant_data = vec![1u8, 2, 3, 4];
        let mut constant_map = HashMap::new();
        constant_map.insert(
            1u32,
            ConstantData {
                data: constant_data.clone(),
                label: None,
            },
        );

        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[1, 1]),
                        pending_permutation: vec![],
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Constant,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[1, 1]),
                        pending_permutation: vec![],
                    },
                    name: Some("weight".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[1, 1]),
                        pending_permutation: vec![],
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![2],
            operations: vec![],
            constant_operand_ids_to_handles: constant_map,
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let json = to_graph_json(&graph, false).expect("to_graph_json");

        assert_eq!(json.inputs.len(), 1);
        assert!(json.inputs.contains_key("input"));
        assert_eq!(json.consts.len(), 1);
        assert!(json.consts.contains_key("weight"));
        assert_eq!(json.outputs.len(), 1);
        assert!(json.outputs.contains_key("output"));
    }

    #[test]
    fn test_to_graph_json_with_operations() {
        // Test conversion with operations
        let mut attrs = serde_json::Map::new();
        attrs.insert("alpha".to_string(), serde_json::json!(0.01));

        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[1, 3]),
                        pending_permutation: vec![],
                    },
                    name: Some("x".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[1, 3]),
                        pending_permutation: vec![],
                    },
                    name: Some("y".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![{
                let attributes = crate::operator_options::OperatorOptions::from_json_with_op_type(
                    "leakyRelu",
                    &serde_json::Value::Object(attrs),
                )
                .expect("leakyRelu options");
                Operation::LeakyRelu {
                    input: 0,
                    options: attributes.as_leaky_relu().cloned(),
                    outputs: vec![1],
                }
            }],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let json = to_graph_json(&graph, false).expect("to_graph_json");

        assert_eq!(json.nodes.len(), 1);
        assert_eq!(json.nodes[0].op, "leakyRelu");
        assert_eq!(json.nodes[0].inputs, vec!["x"]);
        assert!(json.nodes[0].options.contains_key("alpha"));
    }

    #[test]
    fn test_from_graph_json_creates_operands() {
        use webnn_graph::ast::{ConstDecl, ConstInit, OperandDesc};

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: wshape(&[1, 2, 3]),
            },
        );

        let mut consts = BTreeMap::new();
        consts.insert(
            "weight".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: ushape(&[3, 3]),
                init: ConstInit::InlineBytes {
                    bytes: vec![0u8; 36],
                },
            },
        );

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts,
            nodes: vec![],
            outputs: BTreeMap::new(),
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");

        // Check we have operands: input and constant
        // Note: outputs in GraphJson don't create separate operands unless they're
        // referenced by nodes, they just mark existing operands as outputs
        assert!(graph_info.operands.len() >= 2);
        assert_eq!(graph_info.input_operands.len(), 1);

        // Check constant data was stored
        assert_eq!(graph_info.constant_operand_ids_to_handles.len(), 1);
    }

    #[test]
    fn test_from_graph_json_with_scalar_constant() {
        use webnn_graph::ast::{ConstDecl, ConstInit};

        let mut consts = BTreeMap::new();
        consts.insert(
            "scale".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: ushape(&[]),
                init: ConstInit::Scalar {
                    value: serde_json::json!(1.5),
                },
            },
        );

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs: BTreeMap::new(),
            consts,
            nodes: vec![],
            outputs: BTreeMap::new(),
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");

        // Scalar constant should be created
        assert_eq!(graph_info.operands.len(), 1);
        assert!(matches!(graph_info.operands[0].kind, OperandKind::Constant));
        let empty_shape: Vec<Dimension> = vec![];
        assert_eq!(graph_info.operands[0].descriptor.shape, empty_shape);
    }

    #[test]
    fn test_operand_name_generation() {
        // Test that unnamed operands get generated names
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[1]),
                        pending_permutation: vec![],
                    },
                    name: None, // Unnamed
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[1]),
                        pending_permutation: vec![],
                    },
                    name: None, // Unnamed
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let json = to_graph_json(&graph, false).expect("to_graph_json");

        // Generated names should be present
        assert!(json.inputs.contains_key("operand_0"));
        assert!(json.outputs.contains_key("operand_1"));
    }

    #[test]
    fn test_all_data_types_roundtrip() {
        let types = vec![
            DataType::Float32,
            DataType::Float16,
            DataType::Int32,
            DataType::Uint32,
            DataType::Int64,
            DataType::Uint64,
            DataType::Int8,
            DataType::Uint8,
            DataType::Int4,
            DataType::Uint4,
        ];

        for dtype in types {
            let graph = build_quantized_graph_info(dtype);
            let json = to_graph_json(&graph, false).expect("to_graph_json");
            let back = from_graph_json(&json).expect("from_graph_json");

            assert_eq!(
                back.operands[0].descriptor.data_type, dtype,
                "Data type {:?} should roundtrip correctly",
                dtype
            );
        }
    }

    #[test]
    fn test_scalar_constant_float16() {
        let mut consts = BTreeMap::new();
        consts.insert(
            "scale".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Float16,
                shape: ushape(&[1]),
                init: ConstInit::Scalar {
                    value: serde_json::json!(2.5),
                },
            },
        );

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs: BTreeMap::new(),
            consts,
            nodes: vec![],
            outputs: BTreeMap::new(),
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");
        assert_eq!(
            graph_info.operands[0].descriptor.data_type,
            DataType::Float16
        );
        assert!(graph_info.constant_operand_ids_to_handles.contains_key(&0));
    }

    #[test]
    fn test_scalar_constant_int_types() {
        // Test Int32
        let mut consts = BTreeMap::new();
        consts.insert(
            "int_val".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Int32,
                shape: ushape(&[1]),
                init: ConstInit::Scalar {
                    value: serde_json::json!(42),
                },
            },
        );

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs: BTreeMap::new(),
            consts: consts.clone(),
            nodes: vec![],
            outputs: BTreeMap::new(),
        };

        let result = from_graph_json(&graph_json);
        result.unwrap();

        // Test Uint32, Int64, Uint64, Int8, Uint8
        let types_to_test = vec![
            webnn_graph::ast::DataType::Uint32,
            webnn_graph::ast::DataType::Int64,
            webnn_graph::ast::DataType::Uint64,
            webnn_graph::ast::DataType::Int8,
            webnn_graph::ast::DataType::Uint8,
        ];

        for dtype in types_to_test {
            let mut consts = BTreeMap::new();
            consts.insert(
                "val".to_string(),
                ConstDecl {
                    data_type: dtype.clone(),
                    shape: ushape(&[1]),
                    init: ConstInit::Scalar {
                        value: serde_json::json!(10),
                    },
                },
            );

            let graph_json = GraphJson {
                name: Some("test".to_string()),
                format: "webnn-graph-json".to_string(),
                version: 2,
                quantized: false,
                inputs: BTreeMap::new(),
                consts,
                nodes: vec![],
                outputs: BTreeMap::new(),
            };

            from_graph_json(&graph_json)
                .unwrap_or_else(|e| panic!("Failed for type {:?}: {e}", dtype));
        }
    }

    #[test]
    fn test_scalar_constant_int4_error() {
        let mut consts = BTreeMap::new();
        consts.insert(
            "int4_val".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Int4,
                shape: ushape(&[1]),
                init: ConstInit::Scalar {
                    value: serde_json::json!(1),
                },
            },
        );

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs: BTreeMap::new(),
            consts,
            nodes: vec![],
            outputs: BTreeMap::new(),
        };

        let result = from_graph_json(&graph_json);
        assert!(result.is_err());
        match result.unwrap_err() {
            GraphError::ConversionFailed { reason, .. } => {
                assert!(reason.contains("int4/uint4"));
            }
            _ => panic!("Expected ConversionFailed error"),
        }
    }

    #[test]
    fn test_scalar_constant_invalid_value() {
        let mut consts = BTreeMap::new();
        consts.insert(
            "bad_val".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: ushape(&[1]),
                init: ConstInit::Scalar {
                    value: serde_json::json!({"not": "a number"}),
                },
            },
        );

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs: BTreeMap::new(),
            consts,
            nodes: vec![],
            outputs: BTreeMap::new(),
        };

        let result = from_graph_json(&graph_json);
        assert!(result.is_err());
        match result.unwrap_err() {
            GraphError::ConversionFailed { reason, .. } => {
                assert!(reason.contains("Cannot parse scalar value"));
            }
            _ => panic!("Expected ConversionFailed error"),
        }
    }

    #[test]
    fn test_weights_reference_constant() {
        let mut consts = BTreeMap::new();
        consts.insert(
            "weight_ref".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: ushape(&[2, 2]),
                init: ConstInit::Weights {
                    r#ref: "model_weight".to_string(),
                },
            },
        );

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs: BTreeMap::new(),
            consts,
            nodes: vec![],
            outputs: BTreeMap::new(),
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");
        assert_eq!(graph_info.operands.len(), 1);
        assert!(matches!(graph_info.operands[0].kind, OperandKind::Constant));
        // Weight references should create empty data (to be filled by loader)
        let const_data = graph_info.constant_operand_ids_to_handles.get(&0).unwrap();
        assert_eq!(const_data.data.len(), 0);
    }

    #[test]
    fn test_operation_missing_input() {
        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: wshape(&[2]),
            },
        );

        let nodes = vec![Node {
            id: "relu_0".to_string(),
            op: "relu".to_string(),
            inputs: vec!["missing_input".to_string()],
            options: serde_json::Map::new(),
            outputs: None,
        }];

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes,
            outputs: BTreeMap::new(),
        };

        let result = from_graph_json(&graph_json);
        assert!(result.is_err());
        match result.unwrap_err() {
            GraphError::ConversionFailed { reason, .. } => {
                assert!(reason.contains("not found"));
            }
            _ => panic!("Expected ConversionFailed error"),
        }
    }

    #[test]
    fn test_operation_with_none_outputs() {
        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: wshape(&[2]),
            },
        );

        let nodes = vec![Node {
            id: "relu_output".to_string(),
            op: "relu".to_string(),
            inputs: vec!["x".to_string()],
            options: serde_json::Map::new(),
            outputs: None, // Will default to node.id
        }];

        let mut outputs = BTreeMap::new();
        outputs.insert("result".to_string(), "relu_output".to_string());

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes,
            outputs,
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");
        // Should create an output operand named after the node.id
        assert!(
            graph_info
                .operands
                .iter()
                .any(|op| op.name.as_deref() == Some("result"))
        );
    }

    #[test]
    fn test_scalar_with_i64_value() {
        let mut consts = BTreeMap::new();
        consts.insert(
            "int_val".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Int32,
                shape: ushape(&[1]),
                init: ConstInit::Scalar {
                    value: serde_json::json!(42_i64),
                },
            },
        );

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs: BTreeMap::new(),
            consts,
            nodes: vec![],
            outputs: BTreeMap::new(),
        };

        let result = from_graph_json(&graph_json);
        result.unwrap();
    }

    #[test]
    fn test_scalar_with_u64_value() {
        let mut consts = BTreeMap::new();
        consts.insert(
            "uint_val".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Uint32,
                shape: ushape(&[1]),
                init: ConstInit::Scalar {
                    value: serde_json::json!(42_u64),
                },
            },
        );

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs: BTreeMap::new(),
            consts,
            nodes: vec![],
            outputs: BTreeMap::new(),
        };

        let result = from_graph_json(&graph_json);
        result.unwrap();
    }

    #[test]
    fn test_operation_empty_outputs() {
        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: wshape(&[2]),
            },
        );

        let nodes = vec![Node {
            id: "node_0".to_string(),
            op: "relu".to_string(),
            inputs: vec!["x".to_string()],
            options: serde_json::Map::new(),
            outputs: Some(vec![]), // Empty outputs vector
        }];

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes,
            outputs: BTreeMap::new(),
        };

        let result = from_graph_json(&graph_json);
        // Should succeed but create no output operands for the operation
        let graph_info = result.unwrap();
        assert_eq!(graph_info.operations.len(), 1);
        assert_eq!(graph_info.operations[0].output_operands().len(), 0);
    }

    #[test]
    fn test_quantize_linear_infers_output_shape_and_dtype() {
        use webnn_graph::ast::{ConstDecl, ConstInit, OperandDesc};

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: vec![
                    webnn_graph::ast::Dimension::Static(2),
                    webnn_graph::ast::Dimension::Static(3),
                ],
            },
        );

        let mut consts = BTreeMap::new();
        consts.insert(
            "scale".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: vec![],
                init: ConstInit::Scalar {
                    value: serde_json::json!(0.5),
                },
            },
        );
        consts.insert(
            "zero_point".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Uint8,
                shape: vec![],
                init: ConstInit::Scalar {
                    value: serde_json::json!(128),
                },
            },
        );

        let nodes = vec![Node {
            id: "q".to_string(),
            op: "quantizeLinear".to_string(),
            inputs: vec![
                "x".to_string(),
                "scale".to_string(),
                "zero_point".to_string(),
            ],
            options: serde_json::Map::new(),
            outputs: None,
        }];

        let mut outputs = BTreeMap::new();
        outputs.insert("result".to_string(), "q".to_string());

        let graph_json = GraphJson {
            name: Some("q_test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: true,
            inputs,
            consts,
            nodes,
            outputs,
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");

        let out_id = graph_info.output_operands[0] as usize;
        let out_desc = &graph_info.operands[out_id].descriptor;
        assert_eq!(
            out_desc.shape,
            vec![Dimension::Static(2), Dimension::Static(3)]
        );
        assert_eq!(out_desc.data_type, DataType::Uint8);
    }

    #[test]
    fn test_cumulative_sum_infers_output_shape_and_dtype() {
        use webnn_graph::ast::OperandDesc;

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: vec![
                    webnn_graph::ast::Dimension::Static(2),
                    webnn_graph::ast::Dimension::Static(3),
                ],
            },
        );

        let mut options = serde_json::Map::new();
        options.insert(
            "axis".to_string(),
            serde_json::Value::Number(serde_json::Number::from(1)),
        );
        options.insert("exclusive".to_string(), serde_json::Value::Bool(true));
        options.insert("reversed".to_string(), serde_json::Value::Bool(true));

        let nodes = vec![Node {
            id: "cumsum".to_string(),
            op: "cumulativeSum".to_string(),
            inputs: vec!["x".to_string()],
            options,
            outputs: None,
        }];

        let mut outputs = BTreeMap::new();
        outputs.insert("result".to_string(), "cumsum".to_string());

        let graph_json = GraphJson {
            name: Some("cumsum_test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes,
            outputs,
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");

        let out_id = graph_info.output_operands[0] as usize;
        let out_desc = &graph_info.operands[out_id].descriptor;
        assert_eq!(
            out_desc.shape,
            vec![Dimension::Static(2), Dimension::Static(3)]
        );
        assert_eq!(out_desc.data_type, DataType::Float32);
    }

    #[test]
    fn test_gru_cell_infers_output_shape_and_dtype() {
        use webnn_graph::ast::OperandDesc;

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: vec![
                    webnn_graph::ast::Dimension::Static(3),
                    webnn_graph::ast::Dimension::Static(2),
                ],
            },
        );
        inputs.insert(
            "w".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: vec![
                    webnn_graph::ast::Dimension::Static(12),
                    webnn_graph::ast::Dimension::Static(2),
                ],
            },
        );
        inputs.insert(
            "r".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: vec![
                    webnn_graph::ast::Dimension::Static(12),
                    webnn_graph::ast::Dimension::Static(4),
                ],
            },
        );
        inputs.insert(
            "h".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: vec![
                    webnn_graph::ast::Dimension::Static(3),
                    webnn_graph::ast::Dimension::Static(4),
                ],
            },
        );

        let mut options = serde_json::Map::new();
        options.insert(
            "hiddenSize".to_string(),
            serde_json::Value::Number(serde_json::Number::from(4)),
        );

        let nodes = vec![Node {
            id: "gru".to_string(),
            op: "gruCell".to_string(),
            inputs: vec![
                "x".to_string(),
                "w".to_string(),
                "r".to_string(),
                "h".to_string(),
            ],
            options,
            outputs: None,
        }];

        let mut outputs = BTreeMap::new();
        outputs.insert("result".to_string(), "gru".to_string());

        let graph_json = GraphJson {
            name: Some("gru_test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes,
            outputs,
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");

        let out_id = graph_info.output_operands[0] as usize;
        let out_desc = &graph_info.operands[out_id].descriptor;
        assert_eq!(
            out_desc.shape,
            vec![Dimension::Static(3), Dimension::Static(4)]
        );
        assert_eq!(out_desc.data_type, DataType::Float32);
    }

    #[test]
    fn test_from_graph_json_parses_dynamic_input_dimensions() {
        use webnn_graph::ast::{
            DataType as WDataType, Dimension as WDimension, DynamicDimension as WDynamicDimension,
            OperandDesc,
        };

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: WDataType::Float32,
                shape: vec![
                    WDimension::Dynamic(WDynamicDimension {
                        name: "batch".to_string(),
                        max_size: 16,
                    }),
                    WDimension::Static(64),
                ],
            },
        );

        let mut outputs = BTreeMap::new();
        outputs.insert("x_out".to_string(), "x_out".to_string());

        let graph_json = GraphJson {
            name: Some("dynamic_input".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes: vec![Node {
                id: "x_identity".to_string(),
                op: "identity".to_string(),
                inputs: vec!["x".to_string()],
                options: serde_json::Map::new(),
                outputs: Some(vec!["x_out".to_string()]),
            }],
            outputs,
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");
        assert_eq!(graph_info.input_operands.len(), 1);
        let input = &graph_info.operands[graph_info.input_operands[0] as usize];
        assert_eq!(input.name.as_deref(), Some("x"));
        assert_eq!(input.descriptor.shape.len(), 2);
        match &input.descriptor.shape[0] {
            Dimension::Dynamic(d) => {
                assert_eq!(d.name, "batch");
                assert_eq!(d.max_size, 16);
            }
            _ => panic!("expected dynamic dimension at axis 0"),
        }
        assert_eq!(input.descriptor.shape[1], Dimension::Static(64));
        let output = &graph_info.operands[graph_info.output_operands[0] as usize];
        assert_eq!(output.descriptor.shape, input.descriptor.shape);
    }

    #[test]
    fn test_to_graph_json_preserves_dynamic_dimensions() {
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![
                            Dimension::Dynamic(DynamicDimension {
                                name: "batch".to_string(),
                                max_size: 8,
                            }),
                            Dimension::Static(3),
                        ],
                        pending_permutation: vec![],
                    },
                    name: Some("x".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![
                            Dimension::Dynamic(DynamicDimension {
                                name: "batch".to_string(),
                                max_size: 8,
                            }),
                            Dimension::Static(3),
                        ],
                        pending_permutation: vec![],
                    },
                    name: Some("y".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let json = to_graph_json(&graph, false).expect("to_graph_json");
        let input_desc = json.inputs.get("x").expect("input x");
        assert_eq!(input_desc.shape.len(), 2);
        match &input_desc.shape[0] {
            webnn_graph::ast::Dimension::Dynamic(d) => {
                assert_eq!(d.name, "batch");
                assert_eq!(d.max_size, 8);
            }
            _ => panic!("expected dynamic input dimension"),
        }
        assert_eq!(input_desc.shape[1], webnn_graph::ast::Dimension::Static(3));
    }

    #[test]
    fn test_to_graph_json_rejects_dynamic_constant_shape() {
        let mut constants = HashMap::new();
        constants.insert(
            0u32,
            ConstantData {
                data: vec![0u8; 4],
                label: None,
            },
        );

        let graph = GraphInfo {
            operands: vec![Operand {
                kind: OperandKind::Constant,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![Dimension::Dynamic(DynamicDimension {
                        name: "n".to_string(),
                        max_size: 4,
                    })],
                    pending_permutation: vec![],
                },
                name: Some("const_dynamic".to_string()),
            }],
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: constants,
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let err = to_graph_json(&graph, false).unwrap_err();
        match err {
            GraphError::ConversionFailed { reason, .. } => {
                assert!(reason.contains("constant operand"));
                assert!(reason.contains("has dynamic shape"));
            }
            _ => panic!("expected ConversionFailed for dynamic constant shape"),
        }
    }

    #[test]
    fn test_from_graph_json_dynamic_expand_shape_inference_subset() {
        use webnn_graph::ast::{
            DataType as WDataType, Dimension as WDimension, DynamicDimension as WDynamicDimension,
            OperandDesc,
        };

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: WDataType::Float32,
                shape: vec![
                    WDimension::Dynamic(WDynamicDimension {
                        name: "batch".to_string(),
                        max_size: 8,
                    }),
                    WDimension::Static(1),
                ],
            },
        );

        let mut options = serde_json::Map::new();
        options.insert(
            "newShape".to_string(),
            serde_json::json!([
                { "name": "batch", "maxSize": 8 },
                4
            ]),
        );

        let nodes = vec![Node {
            id: "n0".to_string(),
            op: "expand".to_string(),
            inputs: vec!["x".to_string()],
            options,
            outputs: Some(vec!["y".to_string()]),
        }];

        let mut outputs = BTreeMap::new();
        outputs.insert("y".to_string(), "y".to_string());

        let graph_json = GraphJson {
            name: Some("dynamic_expand_subset".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes,
            outputs,
        };

        let graph = from_graph_json(&graph_json).expect("from_graph_json");
        let y_id = graph.output_operands[0] as usize;
        let y_shape = &graph.operands[y_id].descriptor.shape;
        assert_eq!(y_shape.len(), 2);
        match &y_shape[0] {
            Dimension::Dynamic(d) => {
                assert_eq!(d.name, "batch");
                assert_eq!(d.max_size, 8);
            }
            _ => panic!("expected dynamic batch dimension"),
        }
        assert_eq!(y_shape[1], Dimension::Static(4));
    }

    #[test]
    fn test_from_graph_json_dynamic_reshape_shape_inference_subset() {
        use webnn_graph::ast::{
            DataType as WDataType, Dimension as WDimension, DynamicDimension as WDynamicDimension,
            OperandDesc,
        };

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: WDataType::Float32,
                shape: vec![
                    WDimension::Dynamic(WDynamicDimension {
                        name: "batch".to_string(),
                        max_size: 8,
                    }),
                    WDimension::Static(2),
                    WDimension::Static(2),
                ],
            },
        );

        let mut options = serde_json::Map::new();
        options.insert(
            "newShape".to_string(),
            serde_json::json!([
                { "name": "batch", "maxSize": 8 },
                4
            ]),
        );

        let nodes = vec![Node {
            id: "n0".to_string(),
            op: "reshape".to_string(),
            inputs: vec!["x".to_string()],
            options,
            outputs: Some(vec!["y".to_string()]),
        }];

        let mut outputs = BTreeMap::new();
        outputs.insert("y".to_string(), "y".to_string());

        let graph_json = GraphJson {
            name: Some("dynamic_reshape_subset".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes,
            outputs,
        };

        let graph = from_graph_json(&graph_json).expect("from_graph_json");
        let y_id = graph.output_operands[0] as usize;
        let y_shape = &graph.operands[y_id].descriptor.shape;
        assert_eq!(y_shape.len(), 2);
        match &y_shape[0] {
            Dimension::Dynamic(d) => {
                assert_eq!(d.name, "batch");
                assert_eq!(d.max_size, 8);
            }
            _ => panic!("expected dynamic batch dimension"),
        }
        assert_eq!(y_shape[1], Dimension::Static(4));
    }

    #[test]
    fn test_from_graph_json_dynamic_where_broadcast_subset() {
        use webnn_graph::ast::{
            DataType as WDataType, Dimension as WDimension, DynamicDimension as WDynamicDimension,
            OperandDesc,
        };

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "cond".to_string(),
            OperandDesc {
                data_type: WDataType::Uint8,
                shape: vec![WDimension::Static(1), WDimension::Static(4)],
            },
        );
        inputs.insert(
            "a".to_string(),
            OperandDesc {
                data_type: WDataType::Float32,
                shape: vec![
                    WDimension::Dynamic(WDynamicDimension {
                        name: "batch".to_string(),
                        max_size: 8,
                    }),
                    WDimension::Static(4),
                ],
            },
        );
        inputs.insert(
            "b".to_string(),
            OperandDesc {
                data_type: WDataType::Float32,
                shape: vec![
                    WDimension::Dynamic(WDynamicDimension {
                        name: "batch".to_string(),
                        max_size: 8,
                    }),
                    WDimension::Static(1),
                ],
            },
        );

        let nodes = vec![Node {
            id: "n0".to_string(),
            op: "where".to_string(),
            inputs: vec!["cond".to_string(), "a".to_string(), "b".to_string()],
            options: serde_json::Map::new(),
            outputs: Some(vec!["y".to_string()]),
        }];

        let mut outputs = BTreeMap::new();
        outputs.insert("y".to_string(), "y".to_string());

        let graph_json = GraphJson {
            name: Some("dynamic_where_subset".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes,
            outputs,
        };

        let graph = from_graph_json(&graph_json).expect("from_graph_json");
        let y_id = graph.output_operands[0] as usize;
        let y_shape = &graph.operands[y_id].descriptor.shape;
        assert_eq!(y_shape.len(), 2);
        match &y_shape[0] {
            Dimension::Dynamic(d) => {
                assert_eq!(d.name, "batch");
                assert_eq!(d.max_size, 8);
            }
            _ => panic!("expected dynamic batch dimension"),
        }
        assert_eq!(y_shape[1], Dimension::Static(4));
    }

    #[test]
    fn loader_rejects_forward_operation_references() {
        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: wshape(&[2]),
            },
        );
        let graph_json = GraphJson {
            name: Some("forward_reference".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: "consumer".to_string(),
                    op: "relu".to_string(),
                    inputs: vec!["later".to_string()],
                    options: serde_json::Map::new(),
                    outputs: Some(vec!["result".to_string()]),
                },
                Node {
                    id: "producer".to_string(),
                    op: "identity".to_string(),
                    inputs: vec!["x".to_string()],
                    options: serde_json::Map::new(),
                    outputs: Some(vec!["later".to_string()]),
                },
            ],
            outputs: BTreeMap::new(),
        };
        let error = from_graph_json(&graph_json).unwrap_err();
        assert!(error.to_string().contains("missing or forward reference"));
    }

    #[test]
    fn owned_graph_json_moves_inline_constant_bytes() {
        let bytes = vec![0_u8, 0, 128, 63, 0, 0, 0, 64];
        let original_pointer = bytes.as_ptr();
        let graph_json = GraphJson {
            format: "webnn-graph-json".to_string(),
            version: 2,
            name: Some("owned".to_string()),
            quantized: false,
            inputs: BTreeMap::new(),
            consts: BTreeMap::from([(
                "weight".to_string(),
                webnn_graph::ast::ConstDecl {
                    data_type: webnn_graph::ast::DataType::Float32,
                    shape: vec![2],
                    init: ConstInit::InlineBytes { bytes },
                },
            )]),
            nodes: vec![webnn_graph::ast::Node {
                id: "identity".to_string(),
                op: "identity".to_string(),
                inputs: vec!["weight".to_string()],
                options: serde_json::Map::new(),
                outputs: Some(vec!["result".to_string()]),
            }],
            outputs: BTreeMap::from([("result".to_string(), "result".to_string())]),
        };
        let borrowed = from_graph_json(&graph_json).unwrap();
        let owned = from_graph_json_owned(graph_json).unwrap();

        assert_eq!(borrowed.operands.len(), owned.operands.len());
        assert_eq!(borrowed.operations.len(), owned.operations.len());
        assert_eq!(
            borrowed.constant_operand_ids_to_handles[&0].data,
            owned.constant_operand_ids_to_handles[&0].data
        );
        assert_eq!(
            owned.constant_operand_ids_to_handles[&0].data.as_ptr(),
            original_pointer
        );
    }
}

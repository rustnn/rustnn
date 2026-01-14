//! Compiled computational graph representation
//!
//! PyO3 macros generate unsafe code that triggers unsafe_op_in_unsafe_fn warnings.
//! This is expected behavior from the macro-generated code.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::graph::GraphInfo;
use crate::webnn_json;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use std::fs;
use std::path::Path;

/// Represents a compiled computational graph
#[pyclass(name = "MLGraph")]
pub struct PyMLGraph {
    pub(crate) graph_info: GraphInfo,
}

#[pymethods]
impl PyMLGraph {
    fn __repr__(&self) -> String {
        format!(
            "MLGraph(operands={}, operations={})",
            self.graph_info.operands.len(),
            self.graph_info.operations.len()
        )
    }

    /// Get the number of operands in the graph
    #[getter]
    fn operand_count(&self) -> usize {
        self.graph_info.operands.len()
    }

    /// Get the number of operations in the graph
    #[getter]
    fn operation_count(&self) -> usize {
        self.graph_info.operations.len()
    }

    /// Get input names
    fn get_input_names(&self) -> Vec<String> {
        self.graph_info
            .operands
            .iter()
            .filter_map(|op| {
                if matches!(op.kind, crate::graph::OperandKind::Input) {
                    op.name.clone()
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get output names
    fn get_output_names(&self) -> Vec<String> {
        // Use output_operands list instead of filtering by kind
        self.graph_info
            .output_operands
            .iter()
            .filter_map(|&idx| {
                self.graph_info
                    .operands
                    .get(idx as usize)
                    .and_then(|op| op.name.clone())
            })
            .collect()
    }

    /// Debug method to inspect operand at index (for debugging)
    fn debug_operand(&self, idx: usize) -> String {
        if let Some(op) = self.graph_info.operands.get(idx) {
            format!(
                "Operand[{}]: name={:?}, kind={:?}, type={:?}, shape={:?}",
                idx, op.name, op.kind, op.descriptor.data_type, op.descriptor.shape
            )
        } else {
            format!("Operand[{}]: not found", idx)
        }
    }

    /// Count operands with empty shapes
    fn count_empty_shapes(&self) -> usize {
        self.graph_info
            .operands
            .iter()
            .filter(|op| op.descriptor.shape.is_empty())
            .count()
    }

    /// Count operands with empty shapes that are not constants (likely unknown shapes).
    fn count_unknown_shapes(&self) -> usize {
        self.graph_info
            .operands
            .iter()
            .filter(|op| {
                op.descriptor.shape.is_empty()
                    && !matches!(op.kind, crate::graph::OperandKind::Constant)
            })
            .count()
    }

    /// Count constant operands with empty shapes (scalar constants).
    fn count_scalar_constants(&self) -> usize {
        self.graph_info
            .operands
            .iter()
            .filter(|op| {
                op.descriptor.shape.is_empty()
                    && matches!(op.kind, crate::graph::OperandKind::Constant)
            })
            .count()
    }

    /// Count unknown shapes, excluding outputs that are known to be scalar from reduction ops.
    fn count_unknown_shapes_excluding_scalar_ops(&self) -> usize {
        use std::collections::HashSet;

        fn parse_i64_array(value: &serde_json::Value) -> Option<Vec<i64>> {
            let arr = value.as_array()?;
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                if let Some(n) = v.as_i64() {
                    out.push(n);
                } else if let Some(n) = v.as_u64() {
                    out.push(n as i64);
                } else {
                    return None;
                }
            }
            Some(out)
        }

        let mut scalar_outputs = HashSet::new();
        for op in &self.graph_info.operations {
            let op_type = op.op_type.to_ascii_lowercase();
            if op_type == "constant" {
                for &output_id in op.output_operands_slice() {
                    scalar_outputs.insert(output_id);
                }
                continue;
            }
            if !matches!(
                op_type.as_str(),
                "reducemean"
                    | "reducesum"
                    | "reducemax"
                    | "reducemin"
                    | "reduceproduct"
                    | "reducel1"
                    | "reducel2"
                    | "reducelogsum"
                    | "reducelogsumexp"
                    | "reducesumsquare"
            ) {
                continue;
            }

            let keep_dimensions = op
                .attributes
                .get("keepDimensions")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if keep_dimensions {
                continue;
            }

            let Some(output_id) = op.output_operand else {
                continue;
            };
            let Some(input_id) = op.input_operands.first().copied() else {
                continue;
            };
            let input_shape = &self.graph_info.operands[input_id as usize].descriptor.shape;
            if input_shape.is_empty() {
                continue;
            }

            let axes = op.attributes.get("axes").and_then(parse_i64_array);
            let Some(axes) = axes else {
                continue;
            };
            let rank = input_shape.len() as i64;
            let mut normalized = HashSet::new();
            let mut valid = true;
            for axis in axes {
                let mut axis = axis;
                if axis < 0 {
                    axis += rank;
                }
                if axis < 0 || axis >= rank {
                    valid = false;
                    break;
                }
                normalized.insert(axis as usize);
            }
            if valid && normalized.len() == input_shape.len() {
                scalar_outputs.insert(output_id);
            }
        }

        self.graph_info
            .operands
            .iter()
            .enumerate()
            .filter(|(idx, op)| {
                op.descriptor.shape.is_empty()
                    && !matches!(op.kind, crate::graph::OperandKind::Constant)
                    && !scalar_outputs.contains(&(*idx as u32))
            })
            .count()
    }

    /// Debug unknown shapes with producer op and input shapes.
    fn debug_unknown_shapes(&self) -> Vec<String> {
        use std::collections::HashMap;

        let mut producer: HashMap<u32, (String, Vec<u32>)> = HashMap::new();
        for op in &self.graph_info.operations {
            let op_type = op.op_type.clone();
            let input_ids = op.input_operands.clone();
            for &output_id in op.output_operands_slice() {
                producer.insert(output_id, (op_type.clone(), input_ids.clone()));
            }
        }

        let mut out = Vec::new();
        for (idx, operand) in self.graph_info.operands.iter().enumerate() {
            let operand_id = idx as u32;
            if !operand.descriptor.shape.is_empty()
                || matches!(operand.kind, crate::graph::OperandKind::Constant)
            {
                continue;
            }

            let name = operand
                .name
                .clone()
                .unwrap_or_else(|| format!("operand_{}", operand_id));
            if let Some((op_type, inputs)) = producer.get(&operand_id) {
                let mut input_descs = Vec::with_capacity(inputs.len());
                for input_id in inputs {
                    let input_op = &self.graph_info.operands[*input_id as usize];
                    let input_name = input_op
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("operand_{}", input_id));
                    input_descs.push(format!("{}{:?}", input_name, input_op.descriptor.shape));
                }
                out.push(format!(
                    "{} (id={}, op={}, inputs=[{}])",
                    name,
                    operand_id,
                    op_type,
                    input_descs.join(", ")
                ));
            } else {
                out.push(format!("{} (id={}, op=<none>)", name, operand_id));
            }
        }

        out
    }

    /// Debug unknown shapes as structured data.
    fn debug_unknown_shapes_structured(&self, py: Python) -> PyResult<Vec<PyObject>> {
        use pyo3::types::PyDict;
        use std::collections::HashMap;

        let mut producer: HashMap<u32, (String, Vec<u32>)> = HashMap::new();
        for op in &self.graph_info.operations {
            let op_type = op.op_type.clone();
            let input_ids = op.input_operands.clone();
            for &output_id in op.output_operands_slice() {
                producer.insert(output_id, (op_type.clone(), input_ids.clone()));
            }
        }

        let mut out: Vec<PyObject> = Vec::new();
        for (idx, operand) in self.graph_info.operands.iter().enumerate() {
            let operand_id = idx as u32;
            if !operand.descriptor.shape.is_empty()
                || matches!(operand.kind, crate::graph::OperandKind::Constant)
            {
                continue;
            }

            let name = operand
                .name
                .clone()
                .unwrap_or_else(|| format!("operand_{}", operand_id));
            if let Some((op_type, inputs)) = producer.get(&operand_id) {
                let mut input_descs: Vec<PyObject> = Vec::with_capacity(inputs.len());
                for input_id in inputs {
                    let input_op = &self.graph_info.operands[*input_id as usize];
                    let input_name = input_op
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("operand_{}", input_id));
                    let entry = PyDict::new_bound(py);
                    entry.set_item("id", *input_id)?;
                    entry.set_item("name", input_name)?;
                    entry.set_item("shape", input_op.descriptor.shape.clone())?;
                    input_descs.push(entry.into_py(py));
                }
                let entry = PyDict::new_bound(py);
                entry.set_item("id", operand_id)?;
                entry.set_item("name", name)?;
                entry.set_item("op", op_type)?;
                entry.set_item("inputs", input_descs)?;
                out.push(entry.into_py(py));
            } else {
                let entry = PyDict::new_bound(py);
                entry.set_item("id", operand_id)?;
                entry.set_item("name", name)?;
                entry.set_item("op", py.None())?;
                entry.set_item("inputs", Vec::<PyObject>::new())?;
                out.push(entry.into_py(py));
            }
        }

        Ok(out)
    }

    /// Save the graph to a .webnn JSON file
    ///
    /// Args:
    ///     path: File path to save the graph (e.g., "model.webnn")
    ///     quantized: When True, mark the serialized graph as quantized in the header
    ///
    /// Example:
    ///     graph.save("my_model.webnn")
    #[pyo3(signature = (path, quantized=false))]
    fn save(&self, path: &str, quantized: bool) -> PyResult<()> {
        // Convert GraphInfo to GraphJson
        let graph_json = webnn_json::to_graph_json(&self.graph_info, quantized)
            .map_err(|e| PyIOError::new_err(format!("Failed to convert graph: {}", e)))?;

        // Serialize to JSON
        let json_string = serde_json::to_string_pretty(&graph_json)
            .map_err(|e| PyIOError::new_err(format!("Failed to serialize to JSON: {}", e)))?;

        // Write to file
        fs::write(path, json_string)
            .map_err(|e| PyIOError::new_err(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Load a graph from a .webnn file (JSON or text format)
    ///
    /// Args:
    ///     path: File path to load the graph from (e.g., "model.webnn")
    ///     manifest_path: Optional path to manifest.json file for external weights
    ///     weights_path: Optional path to weights file for external weights
    ///
    /// Returns:
    ///     MLGraph: The loaded graph
    ///
    /// The loader automatically detects the format:
    /// - JSON format: Legacy format with embedded base64 weights
    /// - Text format: WebNN DSL format (automatically detected)
    ///
    /// Example:
    ///     graph = MLGraph.load("my_model.webnn")
    ///     graph = MLGraph.load("model.webnn", manifest_path="manifest.json", weights_path="model.weights")
    #[staticmethod]
    #[pyo3(signature = (path, manifest_path=None, weights_path=None))]
    fn load(path: &str, manifest_path: Option<&str>, weights_path: Option<&str>) -> PyResult<Self> {
        // Check if file exists
        let path_obj = Path::new(path);
        if !path_obj.exists() {
            return Err(PyIOError::new_err(format!("File not found: {}", path)));
        }

        // Read file
        let content = fs::read_to_string(path)
            .map_err(|e| PyIOError::new_err(format!("Failed to read file: {}", e)))?;

        // Try to detect format and parse accordingly
        let mut graph_json: webnn_graph::ast::GraphJson = if content.trim().starts_with('{') {
            // JSON format
            serde_json::from_str(&content)
                .map_err(|e| PyIOError::new_err(format!("Failed to parse JSON: {}", e)))?
        } else {
            // WebNN text DSL format - sanitize identifiers first
            let sanitized = crate::loader::sanitize_webnn_identifiers(&content);
            webnn_graph::parser::parse_wg_text(&sanitized).map_err(|e| {
                PyIOError::new_err(format!("Failed to parse WebNN text format: {}", e))
            })?
        };

        // Resolve external weight references if present
        Self::resolve_external_weights(&mut graph_json, manifest_path, weights_path)?;

        // Convert GraphJson to GraphInfo
        let graph_info = webnn_json::from_graph_json(&graph_json)
            .map_err(|e| PyIOError::new_err(format!("Failed to convert graph: {}", e)))?;

        Ok(PyMLGraph { graph_info })
    }
}

impl PyMLGraph {
    pub fn new(graph_info: GraphInfo) -> Self {
        Self { graph_info }
    }

    /// Resolve external weight references in a GraphJson
    ///
    /// This function loads manifest and weights files and resolves all weight references to inline bytes.
    ///
    /// If manifest_path and weights_path are not provided, returns immediately (no external weights).
    fn resolve_external_weights(
        graph_json: &mut webnn_graph::ast::GraphJson,
        manifest_path: Option<&str>,
        weights_path: Option<&str>,
    ) -> PyResult<()> {
        use webnn_graph::ast::ConstInit;
        use webnn_graph::weights::WeightsManifest;

        // If no manifest path provided, assume no external weights
        let manifest_path = match manifest_path {
            Some(p) => p,
            None => return Ok(()),
        };

        // If no weights path provided, assume no external weights
        let weights_path = match weights_path {
            Some(p) => p,
            None => return Ok(()),
        };

        // Load manifest
        let manifest_content = fs::read_to_string(manifest_path)
            .map_err(|e| PyIOError::new_err(format!("Failed to read manifest: {}", e)))?;
        let manifest: WeightsManifest = serde_json::from_str(&manifest_content)
            .map_err(|e| PyIOError::new_err(format!("Failed to parse manifest: {}", e)))?;

        // Load weights file
        let weights_data = fs::read(weights_path)
            .map_err(|e| PyIOError::new_err(format!("Failed to read weights: {}", e)))?;

        // Create a sanitized lookup map: dots and colons in manifest keys -> underscores
        // This allows the sanitized graph references to match manifest entries
        use std::collections::HashMap;
        let sanitized_manifest: HashMap<String, _> = manifest
            .tensors
            .iter()
            .map(|(key, value)| (key.replace("::", "__").replace('.', "_"), value))
            .collect();

        // Resolve weight references in constants
        for (_name, const_decl) in graph_json.consts.iter_mut() {
            if let ConstInit::Weights { r#ref } = &const_decl.init {
                // Look up weight in sanitized manifest (all keys have underscores)
                let tensor_entry = sanitized_manifest.get(r#ref);

                if let Some(tensor_entry) = tensor_entry {
                    let offset = tensor_entry.byte_offset as usize;
                    let length = tensor_entry.byte_length as usize;

                    // Extract bytes from weights file
                    if offset + length > weights_data.len() {
                        return Err(PyIOError::new_err(format!(
                            "Weight '{}' offset/length exceeds weights file size",
                            r#ref
                        )));
                    }
                    let bytes = weights_data[offset..offset + length].to_vec();

                    // Replace weight reference with inline bytes
                    const_decl.init = ConstInit::InlineBytes { bytes };
                } else {
                    return Err(PyIOError::new_err(format!(
                        "Weight '{}' not found in manifest",
                        r#ref
                    )));
                }
            }
        }

        Ok(())
    }
}

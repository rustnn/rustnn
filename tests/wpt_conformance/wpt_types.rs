//! WPT (Web Platform Tests) JSON types for WebNN conformance test data.
//!
//! Matches the structure produced by the WPT-to-JSON conversion used in
//! tests/wpt_data/conformance/*.json.

use serde::Deserialize;
use std::collections::HashMap;

/// Top-level WPT test file (e.g. relu.json, reduce_sum.json).
#[derive(Debug, Clone, Deserialize)]
pub struct WptTestFile {
    pub operation: String,
    #[serde(default)]
    pub tests: Vec<WptTestCase>,
}

/// A single test case within a WPT file.
#[derive(Debug, Clone, Deserialize)]
pub struct WptTestCase {
    pub name: String,
    pub graph: WptGraph,
    /// Optional per-test tolerance override.
    #[serde(default)]
    pub tolerance: Option<WptTolerance>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WptTolerance {
    #[serde(default)]
    pub r#type: String,
    #[serde(default)]
    pub value: serde_json::Value,
}

/// Graph description: inputs, operators, expected outputs (and optional constants).
#[derive(Debug, Clone, Deserialize)]
pub struct WptGraph {
    pub inputs: HashMap<String, WptTensorSpec>,
    pub operators: Vec<WptOperator>,
    #[serde(rename = "expectedOutputs")]
    pub expected_outputs: HashMap<String, WptTensorSpec>,
    /// Optional named constants (same shape as inputs).
    #[allow(dead_code)]
    #[serde(default)]
    pub constants: HashMap<String, WptTensorSpec>,
}

/// Descriptor for shape and data type (used in inputs and expectedOutputs).
#[derive(Debug, Clone, Deserialize)]
pub struct WptDescriptor {
    #[serde(default)]
    pub shape: Vec<u32>,
    #[serde(default, rename = "dataType")]
    pub data_type: String,
}

/// Tensor spec: data (array or scalar) plus optional descriptor.
#[derive(Debug, Clone, Deserialize)]
pub struct WptTensorSpec {
    /// Can be array of numbers, single number, or (in JSON) string for bigint.
    pub data: serde_json::Value,
    /// Inline descriptor; may also appear nested under "descriptor" key in JSON.
    #[serde(default)]
    pub shape: Vec<u32>,
    #[serde(default, rename = "dataType")]
    pub data_type: String,
    #[serde(default)]
    pub constant: bool,
    /// Nested descriptor (WPT format often has descriptor: { shape, dataType }).
    #[serde(default)]
    pub descriptor: Option<WptDescriptor>,
}

impl WptTensorSpec {
    pub fn shape(&self) -> &[u32] {
        self.descriptor
            .as_ref()
            .map(|d| d.shape.as_slice())
            .unwrap_or_else(|| self.shape.as_slice())
    }

    pub fn data_type(&self) -> &str {
        self.descriptor
            .as_ref()
            .map(|d| d.data_type.as_str())
            .unwrap_or_else(|| self.data_type.as_str())
            .trim()
    }
}

/// Single operator in the graph: name, arguments (object or list of objects), outputs.
#[derive(Debug, Clone, Deserialize)]
pub struct WptOperator {
    pub name: String,
    /// Arguments: either a map or list of maps (WPT uses list of single-key dicts).
    #[serde(default)]
    pub arguments: serde_json::Value,
    /// Output name(s): string or array of strings.
    #[serde(default)]
    pub outputs: serde_json::Value,
}

/// Load a WPT test file from JSON.
pub fn load_wpt_file(json: &str) -> Result<WptTestFile, serde_json::Error> {
    serde_json::from_str(json)
}

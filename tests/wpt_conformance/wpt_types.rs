//! WPT (Web Platform Tests) JSON types for WebNN conformance test data.
//!
//! Matches JSON produced by `scripts/wpt_bridge/dump_corpus.mjs`.

use serde::Deserialize;
use std::collections::BTreeMap;

/// Full corpus dumped by the Node bridge in one invocation.
#[derive(Debug, Clone, Deserialize)]
pub struct WptCorpus {
    #[serde(default)]
    pub wpt_dir: String,
    #[allow(dead_code)]
    #[serde(default)]
    pub case_count: usize,
    #[serde(default)]
    pub cases: Vec<WptLoadedCase>,
    #[serde(default, rename = "fileErrors")]
    pub file_errors: Vec<WptFileError>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WptFileError {
    #[serde(rename = "fileName")]
    pub file_name: String,
    pub error: String,
}

/// One conformance case with file/operation metadata from live WPT JS.
#[derive(Debug, Clone, Deserialize)]
pub struct WptLoadedCase {
    #[allow(dead_code)]
    #[serde(rename = "fileName")]
    pub file_name: String,
    pub operation: String,
    pub name: String,
    pub graph: WptGraph,
    #[serde(default)]
    pub tolerance: Option<WptTolerance>,
}

impl WptLoadedCase {
    pub fn as_test_case(&self) -> WptTestCase {
        WptTestCase {
            name: self.name.clone(),
            graph: self.graph.clone(),
            tolerance: self.tolerance.clone(),
        }
    }
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
    #[serde(default, rename = "metricType", alias = "type")]
    pub metric_type: String,
    #[serde(default)]
    pub value: serde_json::Value,
}

/// Graph description: inputs, operators, and expected outputs.
#[derive(Debug, Clone, Deserialize)]
pub struct WptGraph {
    /// Canonically ordered so operand creation and diagnostics are reproducible. (not HashMap)
    pub inputs: BTreeMap<String, WptTensorSpec>,
    pub operators: Vec<WptOperator>,
    #[serde(rename = "expectedOutputs")]
    /// Canonically ordered so graph outputs and diagnostics are reproducible.
    pub expected_outputs: BTreeMap<String, WptTensorSpec>,
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

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_with::{base64::Base64, serde_as};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "camelCase")]
pub struct DynamicDimension {
    pub name: String,
    pub max_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(untagged)]
pub enum Dimension {
    Static(u32),
    Dynamic(DynamicDimension),
}

pub fn to_dimension_vector(shape: &[u32]) -> Vec<Dimension> {
    shape.iter().copied().map(Dimension::Static).collect()
}

pub fn get_static_or_max_size(dim: &Dimension) -> u32 {
    match dim {
        Dimension::Static(v) => *v,
        Dimension::Dynamic(d) => d.max_size,
    }
}

pub fn dynamic_inputs_enabled() -> bool {
    cfg!(feature = "dynamic-inputs")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    Int4,
    Uint4,
    Float16,
    Float32,
    Int32,
    Uint32,
    Int8,
    Uint8,
    Int64,
    Uint64,
}

impl DataType {
    pub fn bytes_per_element(self) -> usize {
        match self {
            // Int4/Uint4 are stored densely as one nibble; we currently treat them as one byte per element
            // to keep tensor sizing simple. If packed storage is introduced later, this should be revisited.
            DataType::Int4 | DataType::Uint4 => 1,
            DataType::Float16 => 2,
            DataType::Float32 => 4,
            DataType::Int32 => 4,
            DataType::Uint32 => 4,
            DataType::Int8 => 1,
            DataType::Uint8 => 1,
            DataType::Int64 => 8,
            DataType::Uint64 => 8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperandDescriptor {
    pub data_type: DataType,
    #[serde(default)]
    pub shape: Vec<Dimension>,
    #[serde(default)]
    pub pending_permutation: Vec<u32>,
}

impl OperandDescriptor {
    pub fn has_dynamic_dimensions(&self) -> bool {
        self.shape
            .iter()
            .any(|dim| matches!(dim, Dimension::Dynamic(_)))
    }

    pub fn static_shape(&self) -> Option<Vec<u32>> {
        let mut shape = Vec::with_capacity(self.shape.len());
        for dim in &self.shape {
            match dim {
                Dimension::Static(v) => shape.push(*v),
                Dimension::Dynamic(_) => return None,
            }
        }
        Some(shape)
    }

    pub fn static_or_max_shape(&self) -> Vec<u32> {
        self.shape.iter().map(get_static_or_max_size).collect()
    }

    pub fn element_count(&self) -> Option<usize> {
        if self.shape.is_empty() {
            return Some(1);
        }
        let mut count = 1usize;
        for dim in &self.shape {
            let size = get_static_or_max_size(dim) as usize;
            count = count.checked_mul(size)?;
        }
        Some(count)
    }

    pub fn byte_length(&self) -> Option<usize> {
        let elements = self.element_count()?;
        elements.checked_mul(self.data_type.bytes_per_element())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OperandKind {
    Input,
    Constant,
    Output,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operand {
    pub kind: OperandKind,
    pub descriptor: OperandDescriptor,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    #[serde(rename = "type")]
    pub op_type: String,
    #[serde(default)]
    pub input_operands: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_operand: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_operands: Vec<u32>,
    #[serde(default)]
    pub attributes: serde_json::Value,
    #[serde(default)]
    pub label: Option<String>,
}

impl Operation {
    /// Borrow all output operand IDs (handles single- and multi-output operations)
    pub fn output_operands_slice(&self) -> &[u32] {
        if !self.output_operands.is_empty() {
            &self.output_operands
        } else {
            self.output_operand.as_slice()
        }
    }

    /// Get all output operand IDs (handles both single and multi-output operations)
    pub fn get_output_operands(&self) -> Vec<u32> {
        self.output_operands_slice().to_vec()
    }
}

impl Operation {
    pub fn display_name(&self) -> String {
        self.label.clone().unwrap_or_else(|| self.op_type.clone())
    }
}

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantData {
    #[serde_as(as = "Base64")]
    pub data: Vec<u8>,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphInfo {
    pub operands: Vec<Operand>,
    #[serde(default)]
    pub input_operands: Vec<u32>,
    #[serde(default)]
    pub output_operands: Vec<u32>,
    #[serde(default)]
    pub operations: Vec<Operation>,
    #[serde(default)]
    pub constant_operand_ids_to_handles: HashMap<u32, ConstantData>,
    #[serde(default)]
    pub id_to_constant_tensor_operand_map: HashMap<u32, String>,
    #[serde(default)]
    pub quantized: bool,
}

impl GraphInfo {
    pub fn operand(&self, id: u32) -> Option<&Operand> {
        self.operands.get(id as usize)
    }

    pub fn has_dynamic_dimensions(&self) -> bool {
        self.operands
            .iter()
            .any(|operand| operand.descriptor.has_dynamic_dimensions())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_bytes_per_element() {
        assert_eq!(DataType::Int4.bytes_per_element(), 1);
        assert_eq!(DataType::Uint4.bytes_per_element(), 1);
        assert_eq!(DataType::Float16.bytes_per_element(), 2);
        assert_eq!(DataType::Float32.bytes_per_element(), 4);
        assert_eq!(DataType::Int32.bytes_per_element(), 4);
        assert_eq!(DataType::Uint32.bytes_per_element(), 4);
        assert_eq!(DataType::Int8.bytes_per_element(), 1);
        assert_eq!(DataType::Uint8.bytes_per_element(), 1);
        assert_eq!(DataType::Int64.bytes_per_element(), 8);
        assert_eq!(DataType::Uint64.bytes_per_element(), 8);
    }

    #[test]
    fn test_data_type_serialization() {
        assert_eq!(serde_json::to_string(&DataType::Int4).unwrap(), "\"int4\"");
        assert_eq!(
            serde_json::to_string(&DataType::Uint4).unwrap(),
            "\"uint4\""
        );
        assert_eq!(
            serde_json::to_string(&DataType::Float32).unwrap(),
            "\"float32\""
        );
    }

    #[test]
    fn test_data_type_deserialization() {
        assert_eq!(
            serde_json::from_str::<DataType>("\"int4\"").unwrap(),
            DataType::Int4
        );
        assert_eq!(
            serde_json::from_str::<DataType>("\"uint4\"").unwrap(),
            DataType::Uint4
        );
        assert_eq!(
            serde_json::from_str::<DataType>("\"float32\"").unwrap(),
            DataType::Float32
        );
    }

    #[test]
    fn test_operand_descriptor_element_count() {
        let desc = OperandDescriptor {
            data_type: DataType::Int4,
            shape: to_dimension_vector(&[2, 3, 4]),
            pending_permutation: vec![],
        };
        assert_eq!(desc.element_count(), Some(24));
    }

    #[test]
    fn test_operand_descriptor_byte_length_int4() {
        let desc = OperandDescriptor {
            data_type: DataType::Int4,
            shape: to_dimension_vector(&[10, 10]),
            pending_permutation: vec![],
        };
        assert_eq!(desc.byte_length(), Some(100));
    }

    #[test]
    fn test_operand_descriptor_byte_length_uint4() {
        let desc = OperandDescriptor {
            data_type: DataType::Uint4,
            shape: to_dimension_vector(&[8, 16]),
            pending_permutation: vec![],
        };
        assert_eq!(desc.byte_length(), Some(128));
    }

    #[test]
    fn test_operand_descriptor_byte_length_float32() {
        let desc = OperandDescriptor {
            data_type: DataType::Float32,
            shape: to_dimension_vector(&[4, 4]),
            pending_permutation: vec![],
        };
        assert_eq!(desc.byte_length(), Some(64));
    }

    #[test]
    fn test_graph_info_quantized_field_default() {
        let json =
            r#"{"operands": [], "input_operands": [], "output_operands": [], "operations": []}"#;
        let graph: GraphInfo = serde_json::from_str(json).unwrap();
        assert!(!graph.quantized);
    }

    #[test]
    fn test_graph_info_quantized_field_true() {
        let json = r#"{"operands": [], "input_operands": [], "output_operands": [], "operations": [], "quantized": true}"#;
        let graph: GraphInfo = serde_json::from_str(json).unwrap();
        assert!(graph.quantized);
    }

    #[test]
    fn test_graph_info_quantized_field_serialization() {
        let graph = GraphInfo {
            operands: vec![],
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };
        let json = serde_json::to_string(&graph).unwrap();
        assert!(json.contains("\"quantized\":true"));
    }

    #[test]
    fn test_graph_info_with_int4_operand() {
        let operand = Operand {
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Int4,
                shape: to_dimension_vector(&[1, 3, 224, 224]),
                pending_permutation: vec![],
            },
            name: Some("input".to_string()),
        };

        let graph = GraphInfo {
            operands: vec![operand],
            input_operands: vec![0],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };

        let json = serde_json::to_string(&graph).unwrap();
        assert!(json.contains("\"int4\""));
        assert!(json.contains("\"quantized\":true"));

        let deserialized: GraphInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.operands[0].descriptor.data_type,
            DataType::Int4
        );
        assert!(deserialized.quantized);
    }

    #[test]
    fn test_graph_info_with_uint4_operand() {
        let operand = Operand {
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Uint4,
                shape: to_dimension_vector(&[64, 64]),
                pending_permutation: vec![],
            },
            name: Some("weight".to_string()),
        };

        let graph = GraphInfo {
            operands: vec![operand],
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };

        let json = serde_json::to_string(&graph).unwrap();
        assert!(json.contains("\"uint4\""));

        let deserialized: GraphInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.operands[0].descriptor.data_type,
            DataType::Uint4
        );
    }
}

//! Serialized Burn execution plan produced by [`crate::converters::burn::BurnConverter`].

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::graph::DataType;
use crate::operators::Operation;

/// Current IR version; bump when breaking serialized layout.
pub const BURN_PLAN_VERSION: u32 = 3;

/// Graph input or output binding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IOBinding {
    pub name: String,
    pub operand_id: u32,
    pub data_type: DataType,
    /// Static size or `-1` for dynamic dimensions.
    pub shape: Vec<i64>,
}

/// Constant tensor embedded in the plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstantSlot {
    pub operand_id: u32,
    pub data_type: DataType,
    pub shape: Vec<u32>,
    pub data: Vec<u8>,
}

/// Portable runtime graph for Burn backends.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BurnGraphPlan {
    pub version: u32,
    pub inputs: Vec<IOBinding>,
    pub outputs: Vec<IOBinding>,
    pub constants: Vec<ConstantSlot>,
    #[serde(with = "json_operations")]
    pub operations: Vec<Operation>,
    /// Operand id to element type for output precision and f16 semantics.
    #[serde(default)]
    pub operand_types: HashMap<u32, DataType>,
}

mod json_operations {
    use super::Operation;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(operations: &[Operation], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = serde_json::to_vec(operations).map_err(serde::ser::Error::custom)?;
        serializer.serialize_bytes(&bytes)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Operation>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes = <Vec<u8>>::deserialize(deserializer)?;
        serde_json::from_slice(&bytes).map_err(serde::de::Error::custom)
    }
}

impl BurnGraphPlan {
    pub fn new(
        inputs: Vec<IOBinding>,
        outputs: Vec<IOBinding>,
        constants: Vec<ConstantSlot>,
        operations: Vec<Operation>,
        operand_types: HashMap<u32, DataType>,
    ) -> Self {
        Self {
            version: BURN_PLAN_VERSION,
            inputs,
            outputs,
            constants,
            operations,
            operand_types,
        }
    }

    pub fn serialize(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }

    pub fn deserialize(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_round_trip() {
        let plan = BurnGraphPlan::new(
            vec![IOBinding {
                name: "x".to_string(),
                operand_id: 0,
                data_type: DataType::Float32,
                shape: vec![2, 2],
            }],
            vec![IOBinding {
                name: "y".to_string(),
                operand_id: 2,
                data_type: DataType::Float32,
                shape: vec![2, 2],
            }],
            vec![],
            vec![Operation::Add {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![2],
            }],
            HashMap::from([(0, DataType::Float32), (2, DataType::Float32)]),
        );
        let bytes = plan.serialize().unwrap();
        let decoded = BurnGraphPlan::deserialize(&bytes).unwrap();
        assert_eq!(decoded.version, plan.version);
        assert_eq!(decoded.inputs, plan.inputs);
        assert_eq!(decoded.outputs, plan.outputs);
        assert_eq!(decoded.operations.len(), plan.operations.len());
        assert!(matches!(decoded.operations[0], Operation::Add { .. }));
    }
}

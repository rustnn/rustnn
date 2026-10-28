/*
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

//! Serialized Burn execution plan produced by [`crate::converters::BurnConverter`].

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::graph::DataType;
use crate::operators::Operation;

/// Current IR version; bump when breaking serialized layout.
pub const BURN_PLAN_VERSION: u32 = 3;

/// Graph input or output binding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IOBinding {
    /// Binding name.
    pub name: String,
    /// Operand index in the graph.
    pub operand_id: u32,
    /// Element data type.
    pub data_type: DataType,
    /// Static size or `-1` for dynamic dimensions.
    pub shape: Vec<i64>,
}

/// Constant tensor embedded in the plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstantSlot {
    /// Operand index in the graph.
    pub operand_id: u32,
    /// Element data type.
    pub data_type: DataType,
    /// Tensor dimensions.
    pub shape: Vec<u32>,
    /// Raw little-endian tensor bytes.
    pub data: Vec<u8>,
}

/// Portable runtime graph for Burn backends.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BurnGraphPlan {
    /// Serialized plan format version.
    pub version: u32,
    /// Graph input bindings.
    pub inputs: Vec<IOBinding>,
    /// Graph output bindings.
    pub outputs: Vec<IOBinding>,
    /// Embedded graph constants.
    pub constants: Vec<ConstantSlot>,
    #[serde(with = "json_operations")]
    /// Recorded typed operations.
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
    /// Construct a plan using the current format version.
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

    /// Encode the plan with postcard.
    pub fn serialize(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }

    /// Decode a plan; execution checks its format version.
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

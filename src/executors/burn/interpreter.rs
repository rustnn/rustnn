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

#![cfg(feature = "burn-runtime-cpu")]

//! Runtime replay of [`BurnGraphPlan`] using Burn tensor operations.

use std::collections::HashMap;

use burn_tensor::backend::Backend;

use crate::burn::{BurnGraphPlan, IOBinding};
use crate::error::GraphError;
use crate::graph::{DataType, OperandDescriptor};

use super::prepared::PreparedPlan;
use super::shape::checked_numel;

/// Input tensor for Burn execution.
#[derive(Debug, Clone)]
pub struct BurnInput {
    /// Binding name.
    pub name: String,
    /// Tensor dimensions.
    pub shape: Vec<usize>,
    /// Host values (float32 representation).
    /// Narrow integer inputs without an exact payload must be finite, integral
    /// and within their declared datatype's range.
    pub data: Vec<f32>,
    /// Exact signed integer values widened to i64. Required for int32/int64 inputs.
    /// Values must fit the declared signed datatype; importing does not wrap.
    pub int64_data: Option<Vec<i64>>,
    /// Exact unsigned integer values widened to u64. Required for uint32/uint64 inputs.
    /// Values must fit the declared unsigned datatype; importing does not clip.
    pub uint64_data: Option<Vec<u64>>,
}

/// Output metadata without host data.
#[derive(Debug, Clone)]
pub struct BurnOutput {
    /// Binding name.
    pub name: String,
    /// Tensor dimensions.
    pub shape: Vec<usize>,
    /// Element data type.
    pub data_type: DataType,
}

/// Output tensor with host-resident float32 data.
#[derive(Debug, Clone)]
pub struct BurnOutputWithData {
    /// Binding name.
    pub name: String,
    /// Tensor dimensions.
    pub shape: Vec<usize>,
    /// Host values (float32 representation).
    pub data: Vec<f32>,
    /// Exact signed integer values widened to i64.
    pub int64_data: Option<Vec<i64>>,
    /// Exact unsigned integer values widened to u64.
    pub uint64_data: Option<Vec<u64>>,
}

/// Replay a serialized plan using the selected Burn tensor backend.
///
/// Each call prepares the plan and its constants. MLContext graphs retain that
/// prepared state across dispatches. Invalid versions, structural references,
/// output arity, type metadata and input payloads return an error before use.
/// Kernel assertions that unwind are converted to runtime errors.
pub fn execute_plan<B: Backend>(
    plan_bytes: &[u8],
    inputs: Vec<BurnInput>,
) -> Result<Vec<BurnOutputWithData>, GraphError> {
    execute_plan_on_device::<B>(plan_bytes, inputs, B::Device::default())
}

/// Execute a plan on an initialized device, with the checks of [`execute_plan`].
pub fn execute_plan_on_device<B: Backend>(
    plan_bytes: &[u8],
    inputs: Vec<BurnInput>,
    device: B::Device,
) -> Result<Vec<BurnOutputWithData>, GraphError> {
    let plan =
        BurnGraphPlan::deserialize(plan_bytes).map_err(|err| GraphError::BurnRuntimeFailed {
            reason: format!("invalid burn plan bytes: {err}"),
        })?;
    PreparedPlan::<B>::new(plan, device)?.execute(inputs)
}

pub fn plan_output_metadata(plan_bytes: &[u8]) -> Result<Vec<BurnOutput>, GraphError> {
    let plan =
        BurnGraphPlan::deserialize(plan_bytes).map_err(|err| GraphError::BurnRuntimeFailed {
            reason: format!("invalid burn plan bytes: {err}"),
        })?;
    Ok(plan
        .outputs
        .iter()
        .map(|binding| BurnOutput {
            name: binding.name.clone(),
            shape: binding
                .shape
                .iter()
                .map(|dim| if *dim < 0 { 1 } else { *dim as usize })
                .collect(),
            data_type: binding.data_type,
        })
        .collect())
}

pub fn zeroed_inputs_from_descriptors(
    descriptors: &HashMap<String, OperandDescriptor>,
) -> Vec<BurnInput> {
    descriptors
        .iter()
        .map(|(name, desc)| {
            let shape: Vec<usize> = desc
                .shape
                .iter()
                .map(|dim| crate::graph::get_static_or_max_size(dim) as usize)
                .collect();
            let total = shape.iter().product::<usize>().max(1);
            BurnInput {
                name: name.clone(),
                shape,
                data: vec![0.0; total],
                int64_data: matches!(
                    desc.data_type,
                    DataType::Int4 | DataType::Int8 | DataType::Int32 | DataType::Int64
                )
                .then(|| vec![0; total]),
                uint64_data: matches!(
                    desc.data_type,
                    DataType::Uint4 | DataType::Uint8 | DataType::Uint32 | DataType::Uint64
                )
                .then(|| vec![0; total]),
            }
        })
        .collect()
}

pub(super) fn validate_input(binding: &IOBinding, input: &BurnInput) -> Result<(), GraphError> {
    if input.shape.len() != binding.shape.len() {
        return Err(GraphError::RuntimeTensorRankMismatch {
            kind: "input".to_string(),
            name: binding.name.clone(),
            expected_rank: binding.shape.len(),
            actual_rank: input.shape.len(),
        });
    }

    for (axis, (&actual, &expected_dim)) in input.shape.iter().zip(&binding.shape).enumerate() {
        if expected_dim >= 0 && actual != expected_dim as usize {
            return Err(GraphError::RuntimeStaticDimensionMismatch {
                kind: "input".to_string(),
                name: binding.name.clone(),
                axis,
                expected: expected_dim as u32,
                actual,
            });
        }
    }

    let count = checked_numel(&input.shape)?;
    if input.data.len() != count {
        return Err(GraphError::BurnRuntimeFailed {
            reason: format!(
                "input '{}': expected {count} elements, got {}",
                binding.name,
                input.data.len()
            ),
        });
    }
    let signed = matches!(
        binding.data_type,
        DataType::Int4 | DataType::Int8 | DataType::Int32 | DataType::Int64
    );
    let unsigned = matches!(
        binding.data_type,
        DataType::Uint4 | DataType::Uint8 | DataType::Uint32 | DataType::Uint64
    );
    if (matches!(binding.data_type, DataType::Int32 | DataType::Int64)
        && input.int64_data.is_none())
        || (matches!(binding.data_type, DataType::Uint32 | DataType::Uint64)
            && input.uint64_data.is_none())
        || input
            .int64_data
            .as_ref()
            .is_some_and(|v| !signed || v.len() != count)
        || input
            .uint64_data
            .as_ref()
            .is_some_and(|v| !unsigned || v.len() != count)
    {
        return Err(GraphError::BurnRuntimeFailed {
            reason: format!(
                "input '{}': missing, incompatible or wrong-length integer payload",
                binding.name
            ),
        });
    }
    Ok(())
}

type HostValues = (Vec<f32>, Option<Vec<i64>>, Option<Vec<u64>>);

pub(crate) fn bytes_to_host_values(
    bytes: &[u8],
    data_type: DataType,
    element_count: usize,
) -> Result<HostValues, GraphError> {
    let expected_bytes = data_type
        .storage_byte_length(element_count)
        .ok_or_else(|| GraphError::BurnRuntimeFailed {
            reason: "tensor byte length overflow".into(),
        })?;
    if bytes.len() != expected_bytes {
        return Err(GraphError::BurnRuntimeFailed {
            reason: format!(
                "{data_type:?} tensor with {element_count} elements requires {expected_bytes} bytes, got {}",
                bytes.len()
            ),
        });
    }
    match data_type {
        DataType::Int4 => Ok((
            crate::graph::unpack_int4(bytes, element_count)
                .into_iter()
                .map(|v| v as f32)
                .collect(),
            None,
            None,
        )),
        DataType::Uint4 => Ok((
            crate::graph::unpack_uint4(bytes, element_count)
                .into_iter()
                .map(f32::from)
                .collect(),
            None,
            None,
        )),
        DataType::Int32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err(GraphError::BurnRuntimeFailed {
                    reason: "int32 tensor byte length must be a multiple of 4".to_string(),
                });
            }
            let exact: Vec<i64> = bytes
                .chunks_exact(4)
                .map(|b| i64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                .collect();
            let values = exact.iter().map(|&v| v as f32).collect();
            Ok((values, Some(exact), None))
        }
        DataType::Uint32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err(GraphError::BurnRuntimeFailed {
                    reason: "uint32 tensor byte length must be a multiple of 4".to_string(),
                });
            }
            let exact: Vec<u64> = bytes
                .chunks_exact(4)
                .map(|b| u64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                .collect();
            let values = exact.iter().map(|&v| v as f32).collect();
            Ok((values, None, Some(exact)))
        }
        DataType::Int64 => {
            if !bytes.len().is_multiple_of(8) {
                return Err(GraphError::BurnRuntimeFailed {
                    reason: "int64 tensor byte length must be a multiple of 8".to_string(),
                });
            }
            let int64_data: Vec<i64> = bytes
                .chunks_exact(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect();
            let values = int64_data.iter().map(|&v| v as f32).collect();
            Ok((values, Some(int64_data), None))
        }
        DataType::Uint64 => {
            if !bytes.len().is_multiple_of(8) {
                return Err(GraphError::BurnRuntimeFailed {
                    reason: "uint64 tensor byte length must be a multiple of 8".to_string(),
                });
            }
            let uint64_data: Vec<u64> = bytes
                .chunks_exact(8)
                .map(|chunk| {
                    u64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect();
            let values = uint64_data.iter().map(|&v| v as f32).collect();
            Ok((values, None, Some(uint64_data)))
        }
        other => Ok((bytes_to_f32(bytes, other)?, None, None)),
    }
}

fn bytes_to_f32(bytes: &[u8], data_type: DataType) -> Result<Vec<f32>, GraphError> {
    match data_type {
        DataType::Float32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err(GraphError::BurnRuntimeFailed {
                    reason: "float32 tensor byte length must be a multiple of 4".to_string(),
                });
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
        DataType::Float16 => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect()),
        DataType::Int32 => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32)
            .collect()),
        DataType::Uint32 => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32)
            .collect()),
        DataType::Int8 => Ok(bytes.iter().map(|&b| b as i8 as f32).collect()),
        DataType::Uint8 => Ok(bytes.iter().map(|&b| b as f32).collect()),
        DataType::Int64 => {
            if !bytes.len().is_multiple_of(8) {
                return Err(GraphError::BurnRuntimeFailed {
                    reason: "int64 tensor byte length must be a multiple of 8".to_string(),
                });
            }
            Ok(bytes
                .chunks_exact(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]) as f32
                })
                .collect())
        }
        DataType::Uint64 => {
            if !bytes.len().is_multiple_of(8) {
                return Err(GraphError::BurnRuntimeFailed {
                    reason: "uint64 tensor byte length must be a multiple of 8".to_string(),
                });
            }
            Ok(bytes
                .chunks_exact(8)
                .map(|chunk| {
                    u64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]) as f32
                })
                .collect())
        }
        other => Err(GraphError::BurnRuntimeFailed {
            reason: format!("unsupported constant data type: {other:?}"),
        }),
    }
}

#[cfg(all(test, feature = "burn-runtime-cpu"))]
mod tests {
    use super::*;
    use crate::burn::{BurnGraphPlan, ConstantSlot, IOBinding};
    use crate::operators::Operation;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn executes_add_plan() {
        let ones: Vec<u8> = [1.0f32; 4]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let plan = BurnGraphPlan::new(
            vec![IOBinding {
                name: "a".to_string(),
                operand_id: 0,
                data_type: DataType::Float32,
                shape: vec![2, 2],
            }],
            vec![IOBinding {
                name: "out".to_string(),
                operand_id: 2,
                data_type: DataType::Float32,
                shape: vec![2, 2],
            }],
            vec![ConstantSlot {
                operand_id: 1,
                data_type: DataType::Float32,
                shape: vec![2, 2],
                data: ones,
            }],
            vec![Operation::Add {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![2],
            }],
            HashMap::from([
                (0, DataType::Float32),
                (1, DataType::Float32),
                (2, DataType::Float32),
            ]),
        );
        let bytes = plan.serialize().unwrap();
        let inputs = vec![BurnInput {
            name: "a".to_string(),
            shape: vec![2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
            int64_data: None,
            uint64_data: None,
        }];
        let outputs = execute_plan::<TestBackend>(&bytes, inputs).unwrap();
        assert_eq!(outputs[0].data, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn validate_input_allows_dynamic_dimensions() {
        let binding = IOBinding {
            name: "past_key".to_string(),
            operand_id: 0,
            data_type: DataType::Float32,
            shape: vec![-1, 12, -1, 64],
        };
        let input = BurnInput {
            name: "past_key".to_string(),
            shape: vec![1, 12, 0, 64],
            data: vec![],
            int64_data: None,
            uint64_data: None,
        };
        validate_input(&binding, &input).unwrap();
    }

    #[test]
    fn validate_input_rejects_static_dimension_mismatch() {
        let binding = IOBinding {
            name: "x".to_string(),
            operand_id: 0,
            data_type: DataType::Float32,
            shape: vec![1, 12, -1, 64],
        };
        let input = BurnInput {
            name: "x".to_string(),
            shape: vec![2, 12, 0, 64],
            data: vec![],
            int64_data: None,
            uint64_data: None,
        };
        let err = validate_input(&binding, &input).unwrap_err();
        assert!(matches!(
            err,
            GraphError::RuntimeStaticDimensionMismatch {
                axis: 0,
                expected: 1,
                actual: 2,
                ..
            }
        ));
    }
}

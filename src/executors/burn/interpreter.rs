#![cfg(any(feature = "burn-runtime-cpu", feature = "burn-runtime-webgpu"))]

//! Runtime replay of [`BurnGraphPlan`] using Burn tensor operations.

use std::collections::HashMap;

use burn::tensor::backend::Backend;

use crate::burn::{BurnGraphPlan, IOBinding};
use crate::error::GraphError;
use crate::graph::{DataType, OperandDescriptor};

use super::execute::execute_operations;
use super::tensor_env::{RuntimeTensor, TensorEnv};

/// Input tensor for Burn execution.
#[derive(Debug, Clone)]
pub struct BurnInput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub int64_data: Option<Vec<i64>>,
    pub uint64_data: Option<Vec<u64>>,
}

/// Output metadata without host data.
#[derive(Debug, Clone)]
pub struct BurnOutput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data_type: DataType,
}

/// Output tensor with host-resident float32 data.
#[derive(Debug, Clone)]
pub struct BurnOutputWithData {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub int64_data: Option<Vec<i64>>,
    pub uint64_data: Option<Vec<u64>>,
}

/// WebGPU [`maxStorageBufferBindingSize`](https://www.w3.org/TR/webgpu/#limits) minimum (128 MiB).
pub const WEBGPU_MIN_STORAGE_BUFFER_BINDING_BYTES: usize = 128 * 1024 * 1024;

/// Largest single-tensor byte size required by a plan and its runtime inputs.
pub fn plan_max_tensor_bytes(plan: &BurnGraphPlan, inputs: &[BurnInput]) -> usize {
    let mut max_bytes = 0usize;
    for slot in &plan.constants {
        max_bytes = max_bytes.max(slot.data.len());
    }
    for input in inputs {
        let byte_len = input.data.len().saturating_mul(std::mem::size_of::<f32>());
        if byte_len > 0 {
            max_bytes = max_bytes.max(byte_len);
        } else {
            let numel = input.shape.iter().product::<usize>().max(1);
            max_bytes = max_bytes.max(numel.saturating_mul(std::mem::size_of::<f32>()));
        }
    }
    for binding in &plan.outputs {
        let numel = binding
            .shape
            .iter()
            .map(|&d| if d < 0 { 1 } else { d as usize })
            .product::<usize>()
            .max(1);
        max_bytes = max_bytes.max(numel.saturating_mul(binding.data_type.bytes_per_element()));
    }
    max_bytes
}

pub fn exceeds_webgpu_min_storage_buffer_binding(bytes: usize) -> bool {
    bytes > WEBGPU_MIN_STORAGE_BUFFER_BINDING_BYTES
}

pub fn execute_plan<B: Backend>(
    plan_bytes: &[u8],
    inputs: Vec<BurnInput>,
) -> Result<Vec<BurnOutputWithData>, GraphError> {
    execute_plan_on_device::<B>(plan_bytes, inputs, B::Device::default())
}

/// Like [`execute_plan`] but uses an already-initialized device (required for Burn WGPU with custom limits).
pub fn execute_plan_on_device<B: Backend>(
    plan_bytes: &[u8],
    inputs: Vec<BurnInput>,
    device: B::Device,
) -> Result<Vec<BurnOutputWithData>, GraphError> {
    let plan =
        BurnGraphPlan::deserialize(plan_bytes).map_err(|err| GraphError::BurnRuntimeFailed {
            reason: format!("invalid burn plan bytes: {err}"),
        })?;
    if plan.version != crate::burn::BURN_PLAN_VERSION {
        return Err(GraphError::BurnRuntimeFailed {
            reason: format!(
                "unsupported burn plan version {} (expected {})",
                plan.version,
                crate::burn::BURN_PLAN_VERSION
            ),
        });
    }
    let mut input_map: HashMap<String, BurnInput> = inputs
        .into_iter()
        .map(|input| (input.name.clone(), input))
        .collect();

    let mut env = TensorEnv::new(device);

    for slot in &plan.constants {
        let shape: Vec<usize> = slot.shape.iter().map(|d| *d as usize).collect();
        let (values, int64_data, uint64_data) = bytes_to_host_values(&slot.data, slot.data_type)?;
        let tensor = RuntimeTensor::from_f32_data(shape, values, &env.device)?;
        env.insert_with_integer_sidecar(
            slot.operand_id,
            slot.data_type,
            tensor,
            int64_data,
            uint64_data,
        );
    }

    for binding in &plan.inputs {
        let input =
            input_map
                .remove(&binding.name)
                .ok_or_else(|| GraphError::RuntimeTensorMissing {
                    kind: "input".to_string(),
                    name: binding.name.clone(),
                })?;
        validate_input(binding, &input)?;
        let tensor = RuntimeTensor::from_f32_data(input.shape.clone(), input.data, &env.device)?;
        env.insert_with_integer_sidecar(
            binding.operand_id,
            binding.data_type,
            tensor,
            input.int64_data,
            input.uint64_data,
        );
    }

    execute_operations::<B>(&mut env, &plan.operations, &plan.operand_types)?;

    let mut outputs = Vec::with_capacity(plan.outputs.len());
    for binding in &plan.outputs {
        let tensor = env.get(binding.operand_id)?;
        let host = tensor.to_host_array()?;
        outputs.push(BurnOutputWithData {
            name: binding.name.clone(),
            shape: host.shape,
            data: host.data,
            int64_data: env.int64_data.get(&binding.operand_id).cloned(),
            uint64_data: env.uint64_data.get(&binding.operand_id).cloned(),
        });
    }

    Ok(outputs)
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
                int64_data: None,
                uint64_data: None,
            }
        })
        .collect()
}

fn validate_input(binding: &IOBinding, input: &BurnInput) -> Result<(), GraphError> {
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

    Ok(())
}

fn bytes_to_host_values(
    bytes: &[u8],
    data_type: DataType,
) -> Result<(Vec<f32>, Option<Vec<i64>>, Option<Vec<u64>>), GraphError> {
    match data_type {
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

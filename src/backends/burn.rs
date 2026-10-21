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

use std::fmt;

use crate::GraphInfo;
use crate::error::Error;
use crate::executors::burn::{BurnInput, PreparedPlan};
use crate::graph::DataType;
use crate::mlcontext::{MLBackendBuilder, MLBackendContext, MLGraph, MLTensor, MLTensorDescriptor};

#[derive(Debug)]
pub(crate) struct BurnCompiledGraph {
    pub(crate) runtime: Box<PreparedPlan<burn_ndarray::NdArray<f32>>>,
}

#[derive(Debug)]
struct BurnHostTensor {
    memory: Vec<u8>,
}

pub(crate) struct BurnContext {
    tensors: Vec<BurnHostTensor>,
}

impl fmt::Debug for BurnContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BurnContext")
            .field("tensor_count", &self.tensors.len())
            .finish()
    }
}

impl BurnContext {
    pub(crate) fn new() -> Self {
        Self {
            tensors: Vec::new(),
        }
    }
}

pub(crate) struct BurnBuilder;

impl fmt::Debug for BurnBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BurnBuilder").finish()
    }
}

impl<'context, 'builder> MLBackendBuilder<'context, 'builder> for BurnBuilder {
    fn build(&mut self, graph_info: GraphInfo) -> crate::error::Result<MLGraph<'context>> {
        // The builder owns WebNN operation validation. Preparation checks only
        // Burn representation limits and initializes constants and lifetimes.
        let runtime = PreparedPlan::from_builder_graph(&graph_info, Default::default())
            .map_err(|e| Error::GraphBuildError { source: e.into() })?;
        MLGraph::new(
            crate::mlcontext::MLBackendGraph::BurnPlan(BurnCompiledGraph {
                runtime: Box::new(runtime),
            }),
            &graph_info,
        )
    }
}

fn tensor_byte_len(descriptor: &MLTensorDescriptor) -> crate::error::Result<usize> {
    descriptor
        .shape()
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))
        .and_then(|n| n.checked_mul(descriptor.data_type().rustnn_element_size_bits()))
        .and_then(|bits| bits.checked_add(7))
        .map(|bits| bits / 8)
        .ok_or_else(|| Error::GraphDispatchError {
            source: "tensor byte length overflow".into(),
        })
}

fn write_output(
    host: &mut [u8],
    result: &crate::executors::burn::BurnOutputWithData,
    data_type: DataType,
    element_count: usize,
) -> crate::error::Result<()> {
    if result.data.len() != element_count {
        return Err(Error::GraphDispatchError {
            source: format!(
                "output '{}': element count mismatch (expected {element_count}, got {})",
                result.name,
                result.data.len()
            )
            .into(),
        });
    }
    let mut bytes = match data_type {
        DataType::Int4 => {
            crate::graph::pack_int4(&result.data.iter().map(|&v| v as i32).collect::<Vec<_>>())
        }
        DataType::Uint4 => crate::graph::pack_uint4_from_i32(
            &result.data.iter().map(|&v| v as i32).collect::<Vec<_>>(),
        ),
        _ => Vec::with_capacity(host.len()),
    };
    if !matches!(data_type, DataType::Int4 | DataType::Uint4) {
        for (i, &value) in result.data.iter().enumerate() {
            match data_type {
                DataType::Float32 => bytes.extend_from_slice(&value.to_le_bytes()),
                DataType::Float16 => {
                    bytes.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes())
                }
                DataType::Int64 => bytes.extend_from_slice(
                    &result
                        .int64_data
                        .as_ref()
                        .map_or(value as i64, |v| v[i])
                        .to_le_bytes(),
                ),
                DataType::Uint64 => bytes.extend_from_slice(
                    &result
                        .uint64_data
                        .as_ref()
                        .map_or(value as u64, |v| v[i])
                        .to_le_bytes(),
                ),
                DataType::Int32 => bytes.extend_from_slice(
                    &result
                        .int64_data
                        .as_ref()
                        .map_or(value as i32, |v| v[i] as i32)
                        .to_le_bytes(),
                ),
                DataType::Uint32 => bytes.extend_from_slice(
                    &result
                        .uint64_data
                        .as_ref()
                        .map_or(value as u32, |v| v[i] as u32)
                        .to_le_bytes(),
                ),
                DataType::Int8 => bytes.push(value as i8 as u8),
                DataType::Uint8 => bytes.push(value as u8),
                DataType::Int4 | DataType::Uint4 => {
                    unreachable!("packed output handled before the byte-aligned loop")
                }
            }
        }
    }
    if bytes.len() != host.len() {
        return Err(Error::GraphDispatchError {
            source: format!(
                "output '{}': byte length mismatch (expected {}, got {})",
                result.name,
                host.len(),
                bytes.len()
            )
            .into(),
        });
    }
    host.copy_from_slice(&bytes);
    Ok(())
}

impl<'context> MLBackendContext<'context> for BurnContext {
    fn accelerated(&self) -> bool {
        false
    }

    fn create_builder<'builder>(
        &mut self,
    ) -> crate::error::Result<Box<dyn MLBackendBuilder<'context, 'builder> + 'builder>>
    where
        'context: 'builder,
    {
        Ok(Box::new(BurnBuilder))
    }

    fn create_tensor(&mut self, descriptor: &MLTensorDescriptor) -> crate::error::Result<MLTensor> {
        let n = tensor_byte_len(descriptor)?;
        self.tensors.push(BurnHostTensor {
            memory: vec![0u8; n.max(1)],
        });
        Ok(MLTensor {
            id: self.tensors.len() - 1,
            constant: false,
            descriptor: descriptor.clone(),
        })
    }

    fn create_constant_tensor(
        &mut self,
        descriptor: &MLTensorDescriptor,
        input_data: &[u8],
    ) -> crate::error::Result<MLTensor> {
        let mut tensor = self.create_tensor(descriptor)?;
        tensor.constant = true;
        self.write_tensor(&tensor, input_data)
            .map_err(|e| Error::TensorCreationError {
                source: e.into(),
                descriptor: descriptor.clone(),
            })?;
        Ok(tensor)
    }

    fn read_tensor(&mut self, tensor: &MLTensor, array: &mut [u8]) -> crate::error::Result<()> {
        let host = &self.tensors[tensor.id].memory;
        let logical = tensor_byte_len(tensor.descriptor())?;
        if array.len() < logical {
            return Err(Error::TensorReadError {
                source: format!(
                    "buffer too small: need {logical} logical bytes, got {}",
                    array.len()
                )
                .into(),
                tensor: tensor.clone(),
            });
        }
        let slice = host.get(..logical).ok_or_else(|| Error::TensorReadError {
            source: format!("tensor storage shorter than logical size ({logical} bytes)").into(),
            tensor: tensor.clone(),
        })?;
        array[..logical].copy_from_slice(slice);
        Ok(())
    }

    fn write_tensor(&mut self, tensor: &MLTensor, array: &[u8]) -> crate::error::Result<()> {
        let host = &mut self.tensors[tensor.id].memory;
        if array.len() > host.len() {
            return Err(Error::TensorWriteError {
                source: format!(
                    "write exceeds tensor storage: {} bytes > {}",
                    array.len(),
                    host.len()
                )
                .into(),
                tensor: tensor.clone(),
            });
        }
        let n = array.len();
        host[..n].copy_from_slice(array);
        Ok(())
    }

    fn dispatch(
        &mut self,
        graph: &mut MLGraph,
        inputs: &crate::mlcontext::MLNamedTensors,
        outputs: &crate::mlcontext::MLNamedTensors,
    ) -> crate::error::Result<()> {
        let burn_graph =
            graph
                .backend
                .as_burn_plan_mut()
                .ok_or_else(|| Error::GraphDispatchError {
                    source: "MLGraph is not a Burn plan graph".into(),
                })?;

        let mut burn_inputs = Vec::with_capacity(burn_graph.runtime.plan.inputs.len());
        for binding in &burn_graph.runtime.plan.inputs {
            let tensor =
                inputs
                    .get(binding.name.as_str())
                    .ok_or_else(|| Error::GraphDispatchError {
                        source: format!("missing input '{}' for Burn dispatch", binding.name)
                            .into(),
                    })?;
            let shape: Vec<usize> = tensor.shape().iter().map(|&d| d as usize).collect();
            let full = &self.tensors[tensor.id].memory;
            let logical = tensor_byte_len(tensor.descriptor())?;
            let bytes = full
                .get(..logical)
                .ok_or_else(|| Error::GraphDispatchError {
                    source: format!(
                        "input '{}': tensor storage shorter than logical size ({logical} bytes)",
                        binding.name
                    )
                    .into(),
                })?;
            let (data, int64_data, uint64_data) = crate::executors::burn::bytes_to_host_values(
                bytes,
                binding.data_type,
                shape.iter().product(),
            )
            .map_err(|e| Error::GraphDispatchError { source: e.into() })?;
            burn_inputs.push(BurnInput {
                name: binding.name.clone(),
                shape,
                data,
                int64_data,
                uint64_data,
            });
        }

        let results = burn_graph
            .runtime
            .execute(burn_inputs)
            .map_err(|e| Error::GraphDispatchError { source: e.into() })?;

        for binding in &burn_graph.runtime.plan.outputs {
            let tensor =
                outputs
                    .get(binding.name.as_str())
                    .ok_or_else(|| Error::GraphDispatchError {
                        source: format!("missing output '{}' for Burn dispatch", binding.name)
                            .into(),
                    })?;
            let result = results
                .iter()
                .find(|out| out.name == binding.name)
                .ok_or_else(|| Error::GraphDispatchError {
                    source: format!("Burn plan produced no output '{}'", binding.name).into(),
                })?;
            let logical = tensor_byte_len(tensor.descriptor())?;
            let host = &mut self.tensors[tensor.id].memory;
            if host.len() < logical {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{}': tensor storage too small (have {}, need {})",
                        binding.name,
                        host.len(),
                        logical
                    )
                    .into(),
                });
            }
            write_output(
                &mut host[..logical],
                result,
                binding.data_type,
                tensor.shape().iter().map(|&d| d as usize).product(),
            )?;
        }

        Ok(())
    }

    fn rustnn_resize_tensor(
        &mut self,
        tensor: &mut MLTensor,
        new_shape: &[u64],
    ) -> crate::error::Result<()> {
        let mut new_desc = tensor.descriptor().clone();
        new_desc.set_shape(new_shape.to_vec());
        let new_bytes = tensor_byte_len(&new_desc)?;
        let host = &mut self.tensors[tensor.id].memory;
        if new_bytes > host.len() {
            host.resize(new_bytes, 0u8);
        }
        tensor.descriptor = new_desc;
        Ok(())
    }

    fn rustnn_set_tensor_capacity(
        &mut self,
        tensor: &mut MLTensor,
        max_shape: &[u64],
    ) -> crate::error::Result<()> {
        let mut descriptor = tensor.descriptor().clone();
        descriptor.set_shape(max_shape.to_vec());
        let new_bytes = tensor_byte_len(&descriptor)?;
        self.tensors[tensor.id].memory = vec![0u8; new_bytes.max(1)];
        Ok(())
    }
}

#[cfg(all(test, feature = "burn-runtime-cpu"))]
mod tests {
    use super::*;
    use crate::mlcontext::{Backend, MLContext, MLContextOptions, MLPowerPreference};
    use crate::webnn_json::from_graph_json;

    #[test]
    fn dispatch_preserves_tensor_types() {
        use crate::mlcontext::{MLNamedOperands, MLNamedTensors, MLOperandDescriptor};
        use crate::operator_enums::MLOperandDataType as T;
        let cases = [
            (
                T::Float16,
                bytemuck::cast_slice(&[0x3c00u16, 0xc000]).to_vec(),
            ),
            (
                T::Int64,
                bytemuck::cast_slice(&[i64::MIN + 1, i64::MAX]).to_vec(),
            ),
            (T::Uint64, bytemuck::cast_slice(&[0u64, u64::MAX]).to_vec()),
            (
                T::Int32,
                bytemuck::cast_slice(&[16_777_217i32, i32::MAX - 1]).to_vec(),
            ),
            (
                T::Uint32,
                bytemuck::cast_slice(&[16_777_217u32, u32::MAX - 1]).to_vec(),
            ),
            (T::Int8, vec![0x80, 0x7f]),
            (T::Uint8, vec![0, 255]),
        ];
        for (data_type, bytes) in cases {
            let mut context = MLContext::create(
                &MLContextOptions::new(MLPowerPreference::Default, false)
                    .with_rustnn_backend_hint(Backend::BurnCpu),
            )
            .unwrap();
            let mut builder = crate::mlcontext::MLGraphBuilder::new(&mut context).unwrap();
            let input = builder
                .input("x", &MLOperandDescriptor::new(data_type, vec![2]))
                .unwrap();
            let output = builder.identity(input).unwrap();
            let mut graph = builder
                .build(&MLNamedOperands::from([("y", output)]))
                .unwrap();
            let descriptor = MLTensorDescriptor::new(data_type, vec![2]);
            let input = context
                .create_tensor(&descriptor.clone().to_writable())
                .unwrap();
            let output = context.create_tensor(&descriptor.to_readable()).unwrap();
            context.write_tensor(&input, &bytes).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("x", &input)]),
                    &MLNamedTensors::from([("y", &output)]),
                )
                .unwrap();
            let mut result = vec![0u8; bytes.len()];
            context.read_tensor(&output, &mut result).unwrap();
            assert_eq!(result, bytes, "{data_type:?}");
        }
    }

    #[test]
    fn triangular_preserves_exact_integers_and_masks_both_batches() {
        use crate::mlcontext::{
            MLGraphBuilder, MLNamedOperands, MLNamedTensors, MLOperandDescriptor,
        };
        use crate::operator_enums::MLOperandDataType as T;
        use crate::operator_options::MLTriangularOptions;

        fn bytes<T: bytemuck::NoUninit>(values: &[T]) -> Vec<u8> {
            bytemuck::cast_slice(values).to_vec()
        }

        let cases = [
            (
                T::Int32,
                bytes(&[
                    i32::MIN,
                    16_777_217,
                    i32::MAX,
                    i32::MIN,
                    i32::MAX,
                    -16_777_217,
                    i32::MAX - 1,
                    -16_777_219,
                    i32::MIN,
                ]),
                bytes(&[0i32, 16_777_217, i32::MAX, 0, 0, -16_777_217, 0, 0, 0]),
                bytes(&[0i32, 0, 0, i32::MIN, 0, 0, i32::MAX - 1, -16_777_219, 0]),
            ),
            (
                T::Uint32,
                bytes(&[
                    0u32,
                    16_777_217,
                    u32::MAX,
                    u32::MAX - 1,
                    1,
                    u32::MAX - 2,
                    16_777_219,
                    u32::MAX - 3,
                    42,
                ]),
                bytes(&[0u32, 16_777_217, u32::MAX, 0, 0, u32::MAX - 2, 0, 0, 0]),
                bytes(&[0u32, 0, 0, u32::MAX - 1, 0, 0, 16_777_219, u32::MAX - 3, 0]),
            ),
            (
                T::Int64,
                bytes(&[
                    i64::MIN,
                    16_777_217,
                    i64::MAX,
                    i64::MIN,
                    i64::MAX,
                    -16_777_217,
                    i64::MAX - 1,
                    -16_777_219,
                    i64::MIN,
                ]),
                bytes(&[0i64, 16_777_217, i64::MAX, 0, 0, -16_777_217, 0, 0, 0]),
                bytes(&[0i64, 0, 0, i64::MIN, 0, 0, i64::MAX - 1, -16_777_219, 0]),
            ),
            (
                T::Uint64,
                bytes(&[
                    0u64,
                    16_777_217,
                    u64::MAX,
                    u64::MAX - 1,
                    1,
                    u64::MAX - 2,
                    16_777_219,
                    u64::MAX - 3,
                    42,
                ]),
                bytes(&[0u64, 16_777_217, u64::MAX, 0, 0, u64::MAX - 2, 0, 0, 0]),
                bytes(&[0u64, 0, 0, u64::MAX - 1, 0, 0, 16_777_219, u64::MAX - 3, 0]),
            ),
        ];
        for (data_type, input_bytes, upper_bytes, lower_bytes) in cases {
            for constant in [false, true] {
                let zeros = vec![0u8; input_bytes.len()];
                for (upper, diagonal, expected) in [
                    (true, 1, &upper_bytes),
                    (false, -1, &lower_bytes),
                    (true, i32::MAX, &zeros),
                    (false, i32::MIN, &zeros),
                    (true, i32::MIN, &input_bytes),
                    (false, i32::MAX, &input_bytes),
                ] {
                    let mut context = MLContext::create(
                        &MLContextOptions::new(MLPowerPreference::Default, false)
                            .with_rustnn_backend_hint(Backend::BurnCpu),
                    )
                    .unwrap();
                    let shape = vec![2, 3, 3];
                    let descriptor = MLOperandDescriptor::new(data_type, shape.clone());
                    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
                    let input = if constant {
                        builder
                            .constant_from_bytes(&descriptor, input_bytes.repeat(2))
                            .unwrap()
                    } else {
                        builder.input("x", &descriptor).unwrap()
                    };
                    let masked = builder
                        .triangular_with_options(
                            input,
                            MLTriangularOptions {
                                upper: Some(upper),
                                diagonal,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                    let output = builder.identity(masked).unwrap();
                    let mut graph = builder
                        .build(&MLNamedOperands::from([("y", output)]))
                        .unwrap();
                    let tensor_descriptor = MLTensorDescriptor::new(data_type, shape);
                    let input = context
                        .create_tensor(&tensor_descriptor.clone().to_writable())
                        .unwrap();
                    let output = context
                        .create_tensor(&tensor_descriptor.to_readable())
                        .unwrap();
                    let inputs = if constant {
                        MLNamedTensors::new()
                    } else {
                        context
                            .write_tensor(&input, &input_bytes.repeat(2))
                            .unwrap();
                        MLNamedTensors::from([("x", &input)])
                    };
                    context
                        .dispatch(&mut graph, &inputs, &MLNamedTensors::from([("y", &output)]))
                        .unwrap();
                    let mut result = vec![0u8; input_bytes.len() * 2];
                    context.read_tensor(&output, &mut result).unwrap();
                    assert_eq!(
                        result,
                        expected.repeat(2),
                        "{data_type:?} constant={constant} upper={upper} diagonal={diagonal}"
                    );
                }
            }
        }
    }

    #[test]
    fn integer_unary_chains_preserve_boundary_values() {
        use crate::mlcontext::{
            MLGraphBuilder, MLNamedOperands, MLNamedTensors, MLOperandDescriptor,
        };
        use crate::operator_enums::MLOperandDataType as T;
        use crate::operator_options::MLClampOptions;

        for (dtype, input_bytes, expected_bytes) in [
            (
                T::Int32,
                bytemuck::cast_slice::<_, u8>(&[
                    -2_147_483_647i32,
                    -16_777_217,
                    0,
                    16_777_217,
                    2_147_483_646,
                    2_147_483_647,
                ])
                .to_vec(),
                bytemuck::cast_slice::<_, u8>(&[
                    2_147_483_645i32,
                    16_777_217,
                    0,
                    16_777_217,
                    2_147_483_645,
                    2_147_483_645,
                ])
                .to_vec(),
            ),
            (
                T::Int64,
                bytemuck::cast_slice::<_, u8>(&[
                    i64::MIN + 1,
                    -16_777_217,
                    0,
                    16_777_217,
                    i64::MAX - 1,
                    i64::MAX,
                ])
                .to_vec(),
                bytemuck::cast_slice::<_, u8>(&[
                    i64::MAX - 2,
                    16_777_217,
                    0,
                    16_777_217,
                    i64::MAX - 2,
                    i64::MAX - 2,
                ])
                .to_vec(),
            ),
        ] {
            for constant in [false, true] {
                let mut context = MLContext::create(
                    &MLContextOptions::new(MLPowerPreference::Default, false)
                        .with_rustnn_backend_hint(Backend::BurnCpu),
                )
                .unwrap();
                let descriptor = MLOperandDescriptor::new(dtype, vec![2, 3]);
                let mut builder = MLGraphBuilder::new(&mut context).unwrap();
                let x = if constant {
                    builder
                        .constant_from_bytes(&descriptor, input_bytes.clone())
                        .unwrap()
                } else {
                    builder.input("x", &descriptor).unwrap()
                };
                let neg = builder.neg(x).unwrap();
                let abs = builder.abs(neg).unwrap();
                let relu = builder.relu(abs).unwrap();
                let max = if dtype == T::Int32 {
                    "2147483645"
                } else {
                    "9223372036854775805"
                };
                let clamped = builder
                    .clamp_with_options(
                        relu,
                        MLClampOptions {
                            max_value: Some(serde_json::from_str(max).unwrap()),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                let y = builder.identity(clamped).unwrap();
                let mut graph = builder.build(&MLNamedOperands::from([("y", y)])).unwrap();
                let desc = MLTensorDescriptor::new(dtype, vec![2, 3]);
                let input = context.create_tensor(&desc.clone().to_writable()).unwrap();
                let output = context.create_tensor(&desc.to_readable()).unwrap();
                context.write_tensor(&input, &input_bytes).unwrap();
                let inputs = if constant {
                    MLNamedTensors::new()
                } else {
                    MLNamedTensors::from([("x", &input)])
                };
                context
                    .dispatch(&mut graph, &inputs, &MLNamedTensors::from([("y", &output)]))
                    .unwrap();
                let mut actual = vec![0u8; expected_bytes.len()];
                context.read_tensor(&output, &mut actual).unwrap();
                assert_eq!(actual, expected_bytes, "{dtype:?} constant={constant}");
            }
        }
    }

    #[test]
    fn unsigned_clamp_preserves_exact_values_and_bounds() {
        use crate::mlcontext::{
            MLGraphBuilder, MLNamedOperands, MLNamedTensors, MLOperandDescriptor,
        };
        use crate::operator_enums::MLOperandDataType as T;
        use crate::operator_options::MLClampOptions;

        for (dtype, bytes, expected, max) in [
            (
                T::Uint32,
                bytemuck::cast_slice::<_, u8>(&[0u32, 16_777_217, u32::MAX - 1, u32::MAX]).to_vec(),
                bytemuck::cast_slice::<_, u8>(&[
                    16_777_215u32,
                    16_777_217,
                    u32::MAX - 2,
                    u32::MAX - 2,
                ])
                .to_vec(),
                "4294967293",
            ),
            (
                T::Uint64,
                bytemuck::cast_slice::<_, u8>(&[0u64, 16_777_217, u64::MAX - 1, u64::MAX]).to_vec(),
                bytemuck::cast_slice::<_, u8>(&[
                    16_777_215u64,
                    16_777_217,
                    u64::MAX - 2,
                    u64::MAX - 2,
                ])
                .to_vec(),
                "18446744073709551613",
            ),
        ] {
            let mut context = MLContext::create(
                &MLContextOptions::new(MLPowerPreference::Default, false)
                    .with_rustnn_backend_hint(Backend::BurnCpu),
            )
            .unwrap();
            let mut builder = MLGraphBuilder::new(&mut context).unwrap();
            let x = builder
                .input("x", &MLOperandDescriptor::new(dtype, vec![4]))
                .unwrap();
            let y = builder
                .clamp_with_options(
                    x,
                    MLClampOptions {
                        min_value: Some(serde_json::json!(16777215)),
                        max_value: Some(serde_json::from_str(max).unwrap()),
                        ..Default::default()
                    },
                )
                .unwrap();
            let y = builder.clamp(y).unwrap();
            let mut graph = builder.build(&MLNamedOperands::from([("y", y)])).unwrap();
            let desc = MLTensorDescriptor::new(dtype, vec![4]);
            let input = context.create_tensor(&desc.clone().to_writable()).unwrap();
            let output = context.create_tensor(&desc.to_readable()).unwrap();
            context.write_tensor(&input, &bytes).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("x", &input)]),
                    &MLNamedTensors::from([("y", &output)]),
                )
                .unwrap();
            let mut actual = vec![0u8; expected.len()];
            context.read_tensor(&output, &mut actual).unwrap();
            assert_eq!(actual, expected, "{dtype:?}");
        }
    }

    #[test]
    fn packed_four_bit_inputs_and_constants_ignore_unused_padding() {
        use crate::mlcontext::{
            MLGraphBuilder, MLNamedOperands, MLNamedTensors, MLOperandDescriptor,
        };
        use crate::operator_enums::MLOperandDataType as T;
        for (data_type, packed, zero_point, expected) in [
            (T::Int4, [0x78u8, 0xaf], 0xffu8, [-3.5f32, 4.0, 0.0, -2.5]),
            (T::Uint4, [0xf0u8, 0xa7], 0xf3u8, [-1.5f32, 6.0, 2.0, 3.5]),
        ] {
            for elements in [1usize, 3, 4] {
                for constant in [false, true] {
                    let mut context = MLContext::create(
                        &MLContextOptions::new(MLPowerPreference::Default, false)
                            .with_rustnn_backend_hint(Backend::BurnCpu),
                    )
                    .unwrap();
                    let shape = vec![elements as u64];
                    let descriptor = MLOperandDescriptor::new(data_type, shape.clone());
                    let input_bytes = packed[..elements.div_ceil(2)].to_vec();
                    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
                    let input = if constant {
                        builder
                            .constant_from_bytes(&descriptor, input_bytes.clone())
                            .unwrap()
                    } else {
                        builder.input("x", &descriptor).unwrap()
                    };
                    let scale = builder.constant_from_value(T::Float32, 0.5f32).unwrap();
                    let zero_point = builder
                        .constant_from_bytes(
                            &MLOperandDescriptor::new(data_type, vec![]),
                            vec![zero_point],
                        )
                        .unwrap();
                    let float = builder
                        .dequantize_linear_with_zeropoint(input, scale, zero_point)
                        .unwrap();
                    let packed_output = builder.identity(input).unwrap();
                    let mut graph = builder
                        .build(&MLNamedOperands::from([
                            ("packed", packed_output),
                            ("float", float),
                        ]))
                        .unwrap();
                    let tensor_descriptor = MLTensorDescriptor::new(data_type, shape.clone());
                    let input_tensor = context
                        .create_tensor(&tensor_descriptor.clone().to_writable())
                        .unwrap();
                    let packed_tensor = context
                        .create_tensor(&tensor_descriptor.to_readable())
                        .unwrap();
                    let float_tensor = context
                        .create_tensor(&MLTensorDescriptor::new(T::Float32, shape).to_readable())
                        .unwrap();
                    let inputs = if constant {
                        MLNamedTensors::new()
                    } else {
                        context.write_tensor(&input_tensor, &input_bytes).unwrap();
                        MLNamedTensors::from([("x", &input_tensor)])
                    };
                    context
                        .dispatch(
                            &mut graph,
                            &inputs,
                            &MLNamedTensors::from([
                                ("packed", &packed_tensor),
                                ("float", &float_tensor),
                            ]),
                        )
                        .unwrap();
                    let mut actual_packed = vec![0u8; input_bytes.len()];
                    context
                        .read_tensor(&packed_tensor, &mut actual_packed)
                        .unwrap();
                    let mut canonical = input_bytes.clone();
                    if elements % 2 == 1 {
                        *canonical.last_mut().unwrap() &= 0x0f;
                    }
                    assert_eq!(
                        actual_packed, canonical,
                        "{data_type:?} elements={elements} constant={constant}"
                    );
                    let mut actual_float = vec![0.0f32; elements];
                    context
                        .read_tensor(&float_tensor, &mut actual_float)
                        .unwrap();
                    assert_eq!(
                        actual_float,
                        expected[..elements],
                        "{data_type:?} elements={elements} constant={constant}"
                    );
                }
            }
        }
    }

    #[test]
    fn quantize_four_bit_rounds_clips_and_packs_outputs() {
        use crate::mlcontext::{
            MLGraphBuilder, MLNamedOperands, MLNamedTensors, MLOperandDescriptor,
        };
        use crate::operator_enums::MLOperandDataType as T;
        let values = [-100.0f32, -7.5, -6.5, -0.5, 0.5, 6.5, 7.5, 100.0, 1.5];
        for (data_type, packed, expected) in [
            (
                T::Int4,
                [0x88u8, 0x0a, 0x60, 0x77, 0x02],
                [-8.0f32, -8.0, -6.0, 0.0, 0.0, 6.0, 7.0, 7.0, 2.0],
            ),
            (
                T::Uint4,
                [0x00u8, 0x00, 0x60, 0xf8, 0x02],
                [0.0f32, 0.0, 0.0, 0.0, 0.0, 6.0, 8.0, 15.0, 2.0],
            ),
        ] {
            for elements in [8usize, 9] {
                let mut context = MLContext::create(
                    &MLContextOptions::new(MLPowerPreference::Default, false)
                        .with_rustnn_backend_hint(Backend::BurnCpu),
                )
                .unwrap();
                let shape = vec![elements as u64];
                let mut builder = MLGraphBuilder::new(&mut context).unwrap();
                let input = builder
                    .input("x", &MLOperandDescriptor::new(T::Float32, shape.clone()))
                    .unwrap();
                let scale = builder.constant_from_value(T::Float32, 1.0f32).unwrap();
                let zero_point = builder
                    .constant_from_bytes(&MLOperandDescriptor::new(data_type, vec![]), vec![0xf0])
                    .unwrap();
                let quantized = builder
                    .quantize_linear_with_zeropoint(input, scale, zero_point)
                    .unwrap();
                let dequantized = builder
                    .dequantize_linear_with_zeropoint(quantized, scale, zero_point)
                    .unwrap();
                let mut graph = builder
                    .build(&MLNamedOperands::from([
                        ("packed", quantized),
                        ("float", dequantized),
                    ]))
                    .unwrap();
                let descriptor = MLTensorDescriptor::new(T::Float32, shape.clone());
                let input = context
                    .create_tensor(&descriptor.clone().to_writable())
                    .unwrap();
                let float_output = context.create_tensor(&descriptor.to_readable()).unwrap();
                let packed_output = context
                    .create_tensor(&MLTensorDescriptor::new(data_type, shape).to_readable())
                    .unwrap();
                context.write_tensor(&input, &values[..elements]).unwrap();
                context
                    .dispatch(
                        &mut graph,
                        &MLNamedTensors::from([("x", &input)]),
                        &MLNamedTensors::from([
                            ("packed", &packed_output),
                            ("float", &float_output),
                        ]),
                    )
                    .unwrap();
                let mut actual_packed = vec![0u8; elements.div_ceil(2)];
                context
                    .read_tensor(&packed_output, &mut actual_packed)
                    .unwrap();
                assert_eq!(
                    actual_packed,
                    packed[..elements.div_ceil(2)],
                    "{data_type:?} elements={elements}"
                );
                let mut actual_float = vec![0.0f32; elements];
                context
                    .read_tensor(&float_output, &mut actual_float)
                    .unwrap();
                assert_eq!(
                    actual_float,
                    expected[..elements],
                    "{data_type:?} elements={elements}"
                );
            }
        }
    }

    #[test]
    fn create_context_with_burn_cpu_hint() {
        let context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_rustnn_backend_hint(Backend::BurnCpu),
        )
        .expect("burn cpu context");
        assert!(!context.accelerated());
    }

    #[test]
    fn dispatch_add_graph_via_burn_cpu_hint() {
        let contents = r#"
webnn_graph "sample_graph" v1 {
  inputs { lhs: f32[2, 2]; }
  consts { rhs: f32[2, 2] @scalar(1.0); }
  nodes { sum = add(lhs, rhs); }
  outputs { sum; }
}"#;
        let sanitized = crate::loader::sanitize_webnn_identifiers(contents);
        let graph_json = webnn_graph::parser::parse_wg_text(&sanitized).unwrap();
        let graph_info = from_graph_json(&graph_json).unwrap();

        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_rustnn_backend_hint(Backend::BurnCpu),
        )
        .expect("burn cpu context");

        let mut desc = MLTensorDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            vec![2, 2],
        );
        desc.set_readable(true);
        desc.set_writable(true);

        let mut builder = crate::mlcontext::MLGraphBuilder::new(&mut context).unwrap();
        let mut graph = builder.build_graph_info(graph_info).unwrap();
        drop(builder);

        let tensor = context.create_tensor(&desc).unwrap();
        let output = context.create_tensor(&desc).unwrap();
        let mut inputs = crate::mlcontext::MLNamedTensors::new();
        inputs.insert("lhs", &tensor);
        let mut outputs = crate::mlcontext::MLNamedTensors::new();
        outputs.insert("sum", &output);

        let upload = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut download = vec![0.0f32; 4];
        context.write_tensor(&tensor, &upload).unwrap();
        context.dispatch(&mut graph, &inputs, &outputs).unwrap();
        context.read_tensor(&output, &mut download).unwrap();
        assert_eq!(download, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn write_int64_tensor_with_capacity() {
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_rustnn_backend_hint(Backend::BurnCpu),
        )
        .expect("burn cpu context");

        let mut desc =
            MLTensorDescriptor::new(crate::operator_enums::MLOperandDataType::Int64, vec![1, 1]);
        desc.set_writable(true);
        desc.set_readable(true);
        let mut tensor = context.create_tensor(&desc).unwrap();
        context
            .rustnn_set_tensor_capacity(&mut tensor, &[1, 512])
            .unwrap();
        context.rustnn_resize_tensor(&mut tensor, &[1, 4]).unwrap();
        context.write_tensor(&tensor, &[1i64, 0, 1, 0]).unwrap();
        let mut out = [0i64; 4];
        context.read_tensor(&tensor, &mut out).unwrap();
        assert_eq!(out, [1, 0, 1, 0]);
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
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

//! Huawei CANN/HiAI backend for OpenHarmony NPUs (`cann-runtime` and `cann-runtime-mock`).
//!
//! The graph is compiled to an offline model through the `hiai-rs` adapter and executed
//! with a HiAI session; sessions are cached per compiled model. Only selected when requested
//! with a backend hint. The mock feature validates operator support without a device.
//!
#![doc = include_str!("../../docs/integration/cann.md")]

use std::fmt;

use crate::GraphInfo;
use crate::backend_selection::DeviceType;
use crate::converters::cann::encode_via_adapter;
use crate::error::{Error, Result};
use crate::executors::cann::{CannInput, CannOutput, CannSession};
use crate::mlcontext::MLBackendGraph::CannEngine;
use crate::mlcontext::{
    ListDevices, MLBackendBuilder, MLBackendContext, MLGraph, MLNamedTensors, MLTensor,
    MLTensorDescriptor, RustNNOptions,
};

#[derive(Debug)]
pub(crate) struct CannTensor {
    memory: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct CannGraph {
    // The compiled model, loaded onto the NPU at `build()` time; `dispatch()`
    // only runs it.
    pub(crate) session: CannSession,
    // Input/output names in the model's canonical order (dispatch feeds the NPU
    // positionally; MLNamedTensors sorts by name).
    pub(crate) input_names: Vec<String>,
    pub(crate) output_names: Vec<String>,
}

#[derive(Debug)]
pub(crate) struct CannContext {
    tensors: Vec<CannTensor>,
    _device_type: DeviceType,
}

impl CannContext {
    pub(crate) fn new_from_device_type(
        device_type: DeviceType,
        _options: Option<&RustNNOptions>,
    ) -> Result<Self> {
        Ok(Self {
            tensors: Vec::new(),
            _device_type: device_type,
        })
    }
}

impl ListDevices for CannContext {
    fn list_devices() -> Vec<crate::backend_selection::BackendDevice> {
        vec![crate::backend_selection::BackendDevice::Cann {
            device_type: crate::backend_selection::DeviceType::Npu,
        }]
    }
}

impl<'context> MLBackendContext<'context> for CannContext {
    fn accelerated(&self) -> bool {
        true
    }

    fn create_builder<'builder>(
        &mut self,
    ) -> Result<Box<dyn MLBackendBuilder<'context, 'builder> + 'builder>>
    where
        'context: 'builder,
    {
        Ok(Box::new(CannBuilder { graph: None }))
    }

    fn create_tensor(&mut self, descriptor: &MLTensorDescriptor) -> Result<MLTensor> {
        let byte_count = descriptor.rustnn_required_bytes();
        let memory = vec![0u8; byte_count];
        self.tensors.push(CannTensor { memory });
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
    ) -> Result<MLTensor> {
        let mut tensor = self.create_tensor(descriptor)?;
        tensor.constant = true;
        self.write_tensor(&tensor, input_data).map_err(|e| {
            crate::error::Error::TensorCreationError {
                source: e.into(),
                descriptor: descriptor.clone(),
            }
        })?;
        Ok(tensor)
    }

    fn read_tensor(&mut self, tensor: &MLTensor, array: &mut [u8]) -> Result<()> {
        let host = &self.tensors[tensor.id].memory;
        let logical = tensor.rustnn_required_bytes();
        if array.len() < logical {
            return Err(crate::error::Error::TensorReadError {
                source: format!(
                    "buffer too small: need {} logical bytes, got {}",
                    logical,
                    array.len()
                )
                .into(),
                tensor: tensor.clone(),
            });
        }
        let slice = host
            .get(..logical)
            .ok_or_else(|| crate::error::Error::TensorReadError {
                source: format!("tensor storage shorter than logical size ({logical} bytes)")
                    .into(),
                tensor: tensor.clone(),
            })?;
        array[..logical].copy_from_slice(slice);
        Ok(())
    }

    fn write_tensor(&mut self, tensor: &MLTensor, array: &[u8]) -> Result<()> {
        let host = &mut self.tensors[tensor.id].memory;
        if array.len() > host.len() {
            return Err(crate::error::Error::TensorWriteError {
                source: format!(
                    "write exceeds tensor storage: {} bytes > {}",
                    array.len(),
                    host.len()
                )
                .into(),
                tensor: tensor.clone(),
            });
        }
        let byte_len = array.len();
        host[..byte_len].copy_from_slice(array);
        Ok(())
    }

    fn dispatch(
        &mut self,
        graph: &mut MLGraph,
        inputs: &MLNamedTensors,
        outputs: &MLNamedTensors,
    ) -> Result<()> {
        let cann_graph = if let CannEngine(ref cann_graph) = graph.backend {
            cann_graph
        } else {
            return Err(Error::GraphDispatchError {
                source: "graph is not a CANN graph".into(),
            });
        };

        // Resolve the input/output `MLTensor`s in the model's canonical order
        // (BTreeMap sorts by name, so look each one up by name).
        let input_tensors: Vec<&MLTensor> = cann_graph
            .input_names
            .iter()
            .map(|name| {
                inputs
                    .get(name.as_str())
                    .copied()
                    .ok_or_else(|| Error::GraphDispatchError {
                        source: format!("missing input '{name}' for CANN dispatch").into(),
                    })
            })
            .collect::<Result<_>>()?;

        let output_tensors: Vec<&MLTensor> = cann_graph
            .output_names
            .iter()
            .map(|name| {
                outputs
                    .get(name.as_str())
                    .copied()
                    .ok_or_else(|| Error::GraphDispatchError {
                        source: format!("missing output '{name}' for CANN dispatch").into(),
                    })
            })
            .collect::<Result<_>>()?;

        // Input storage must cover the logical byte length; fail rather than
        // upload a short buffer to the NPU.
        for t in &input_tensors {
            let logical = t.rustnn_required_bytes();
            let actual = self.tensors[t.id].memory.len();
            if actual < logical {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "input tensor {} storage too short: {actual} bytes < logical {logical}",
                        t.id
                    )
                    .into(),
                });
            }
        }

        // Take output buffers out (zero-copy) so they can be borrowed mutably
        // while inputs borrow `self.tensors`. A tensor that is also an input is
        // cloned instead, so the take doesn't empty the input buffer.
        let mut output_buffers: Vec<Vec<u8>> = output_tensors
            .iter()
            .map(|t| {
                if input_tensors.iter().any(|i| i.id == t.id) {
                    self.tensors[t.id].memory.clone()
                } else {
                    std::mem::take(&mut self.tensors[t.id].memory)
                }
            })
            .collect();

        // Borrow the input buffers directly (no clone); the NPU reads them in
        // place.
        let input_descs: Vec<CannInput<'_>> = input_tensors
            .iter()
            .map(|t| CannInput {
                data: &self.tensors[t.id].memory,
                shape: t.shape().iter().map(|dim| *dim as u32).collect(),
                dtype: t.data_type(),
            })
            .collect();

        // Borrow the output buffers in place (no clone); the NPU result is
        // written straight into them.
        let mut output_descs: Vec<CannOutput<'_>> = output_tensors
            .iter()
            .zip(output_buffers.iter_mut())
            .map(|(t, buf)| CannOutput {
                data: buf.as_mut_slice(),
                shape: t.shape().iter().map(|dim| *dim as u32).collect(),
                dtype: t.data_type(),
                actual_len: 0,
            })
            .collect();

        // The model was compiled and loaded at `build()` time; just run it.
        let result = cann_graph.session.dispatch(&input_descs, &mut output_descs);

        // Restore output buffers at full logical length (never truncate to the
        // NPU-reported size). Warn on a size mismatch; on failure restore the
        // originals so storage is never left empty.
        let ok = result.is_ok();
        let actual_lens: Vec<usize> = output_descs.iter().map(|d| d.actual_len).collect();
        for ((t, buf), actual) in output_tensors.iter().zip(output_buffers).zip(actual_lens) {
            if ok {
                let logical = t.rustnn_required_bytes();
                if actual != logical {
                    log::warn!(
                        "CANN output tensor {} produced {actual} bytes, expected {logical}",
                        t.id
                    );
                }
            }
            self.tensors[t.id].memory = buf;
        }

        result.map_err(|e| Error::GraphDispatchError {
            source: Box::new(e),
        })?;

        Ok(())
    }

    fn rustnn_resize_tensor(&mut self, _tensor: &mut MLTensor, _new_shape: &[u64]) -> Result<()> {
        Ok(())
    }

    fn rustnn_set_tensor_capacity(
        &mut self,
        _tensor: &mut MLTensor,
        _max_shape: &[u64],
    ) -> Result<()> {
        Ok(())
    }
}

pub(crate) struct CannBuilder {
    graph: Option<GraphInfo>,
}

impl fmt::Debug for CannBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CannBuilder")
            .field("has_graph", &self.graph.is_some())
            .finish()
    }
}

impl<'context, 'builder> MLBackendBuilder<'context, 'builder> for CannBuilder {
    fn build(&mut self, graph_info: GraphInfo) -> Result<MLGraph<'context>> {
        // Record the input/output names in the model's canonical order. Names
        // are guaranteed present: MLGraph::new() below runs io_binding_maps(),
        // which errors on missing or duplicate names.
        let input_names = graph_info
            .input_operands
            .iter()
            .map(|&id| {
                graph_info.operands[id as usize]
                    .name
                    .clone()
                    .expect("input name validated by io_binding_maps")
            })
            .collect();
        let output_names = graph_info
            .output_operands
            .iter()
            .map(|&id| {
                graph_info.operands[id as usize]
                    .name
                    .clone()
                    .expect("output name validated by io_binding_maps")
            })
            .collect();

        // Compile and load the model onto the NPU now, at `build()` time, so
        // the compile cost is outside the measured "Inference"; `dispatch()`
        // then just runs the loaded session.
        let session = CannSession::compile(&graph_info, encode_via_adapter).map_err(|e| {
            Error::GraphBuildError {
                source: format!("CANN graph build failed: {e}").into(),
            }
        })?;

        let graph = CannGraph {
            session,
            input_names,
            output_names,
        };
        MLGraph::new(CannEngine(graph), &graph_info)
    }
}

#[cfg(test)]
mod tests {
    use super::CannContext;
    use crate::backend_selection::DeviceType;
    use crate::mlcontext::{MLBackendContext, MLTensorDescriptor};
    use crate::operator_enums::MLOperandDataType;

    #[test]
    fn test_context_new() {
        let context = CannContext::new_from_device_type(DeviceType::Npu, None).unwrap();
        assert!(context.accelerated());
    }

    #[test]
    fn test_create_tensor() {
        let mut context = CannContext::new_from_device_type(DeviceType::Npu, None).unwrap();
        let mut desc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
        desc.set_readable(true);
        desc.set_writable(true);
        let tensor = context.create_tensor(&desc).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
    }

    #[test]
    fn test_write_and_read_tensor() {
        let mut context = CannContext::new_from_device_type(DeviceType::Npu, None).unwrap();
        let mut desc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
        desc.set_readable(true);
        desc.set_writable(true);
        let tensor = context.create_tensor(&desc).unwrap();

        let upload = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut download = vec![0.0f32; 4];
        context
            .write_tensor(&tensor, bytemuck::cast_slice(&upload))
            .unwrap();
        context
            .read_tensor(&tensor, bytemuck::cast_slice_mut(&mut download))
            .unwrap();
        assert_eq!(&upload, &download);
    }
}

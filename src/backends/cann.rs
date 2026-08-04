// SPDX-FileCopyrightText: 2026 Shubham Gupta <shubhamg13.work@gmail.com>
//
// SPDX-License-Identifier: Apache-2

use std::fmt;

use crate::GraphInfo;
use crate::backend_selection::DeviceType;
use crate::converters::cann::encode_via_adapter;
use crate::error::{Error, Result};
use crate::executors::cann_shim::{CannTensorDesc, cann_dispatch};
use crate::mlcontext::MLBackendGraph::CannEngine;
use crate::mlcontext::{
    ListDevices, MLBackendBuilder, MLBackendContext, MLGraph, MLTensor, MLTensorDescriptor,
    RustNNOptions,
};
use crate::operator_enums::MLOperandDataType;

/// Map WebNN operand data type to CANN adapter enum
fn ml_operand_to_cann_dtype(data_type: MLOperandDataType) -> i32 {
    match data_type {
        MLOperandDataType::Float32 => 0, // CANN_DT_FLOAT
        MLOperandDataType::Float16 => 1, // CANN_DT_FLOAT16
        MLOperandDataType::Int32 => 3,   // CANN_DT_INT32
        MLOperandDataType::Int8 => 2,    // CANN_DT_INT8
        MLOperandDataType::Uint8 => 4,   // CANN_DT_UINT8
        MLOperandDataType::Int64 => 9,   // CANN_DT_INT64
        MLOperandDataType::Uint32 => 8,  // CANN_DT_UINT32
        _ => 0,                          // default CANN_DT_FLOAT
    }
}

#[derive(Debug)]
pub(crate) struct CannTensor {
    memory: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct CannGraph {
    pub(crate) model_bytes: Vec<u8>,
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
        inputs: &HashMap<&str, &MLTensor>,
        outputs: &HashMap<&str, &MLTensor>,
    ) -> Result<()> {
        let model_bytes = if let CannEngine(ref cann_graph) = graph.backend {
            &cann_graph.model_bytes
        } else {
            return Err(Error::GraphDispatchError {
                source: "graph is not a CANN graph".into(),
            });
        };

        let build_desc = |tensor: &&MLTensor| -> CannTensorDesc {
            CannTensorDesc {
                data: self.tensors[tensor.id].memory.clone(),
                shape: tensor.shape().iter().map(|dim| *dim as u32).collect(),
                dtype: ml_operand_to_cann_dtype(tensor.data_type()),
            }
        };

        let input_descs: Vec<CannTensorDesc> = inputs.values().map(build_desc).collect();
        let mut output_descs: Vec<CannTensorDesc> = outputs.values().map(build_desc).collect();

        cann_dispatch(model_bytes, &input_descs, &mut output_descs)?;

        for (tensor, output_desc) in outputs.values().zip(output_descs.iter()) {
            self.tensors[tensor.id]
                .memory
                .copy_from_slice(&output_desc.data);
        }

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
        // Build the CANN graph and compile to offline model bytes.
        let model_bytes =
            encode_via_adapter(&graph_info).map_err(|e| Error::GraphDispatchError {
                source: format!("CANN graph build failed: {e}").into(),
            })?;
        let graph = CannGraph { model_bytes };
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

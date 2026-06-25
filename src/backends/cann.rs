// SPDX-FileCopyrightText: 2026 Shubham Gupta <shubhamg13.work@gmail.com>
//
// SPDX-License-Identifier: Apache-2

use std::collections::HashMap;
use std::fmt;

use crate::GraphInfo;
use crate::backend_selection::DeviceType;
use crate::error::Result;
use crate::mlcontext::{
    ListDevices, MLBackendBuilder, MLBackendContext, MLGraph, MLTensor, MLTensorDescriptor,
    RustNNOptions,
};

#[derive(Debug)]
pub(crate) struct CannTensor {
    memory: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct CannGraph {
    pub(crate) _model_bytes: Vec<u8>,
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
        let n = descriptor.rustnn_required_bytes();
        let memory = vec![0u8; n];
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
        let n = array.len();
        host[..n].copy_from_slice(array);
        Ok(())
    }

    fn dispatch(
        &mut self,
        _graph: &mut MLGraph,
        _inputs: &HashMap<&str, &MLTensor>,
        _outputs: &HashMap<&str, &MLTensor>,
    ) -> Result<()> {
        Err(crate::error::Error::GraphDispatchError {
            source: "CANN dispatch not yet implemented".into(),
        })
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
        let graph = CannGraph {
            _model_bytes: Vec::new(),
        };
        MLGraph::new(
            crate::mlcontext::MLBackendGraph::CannGraph {
                graph,
                _phantom: std::marker::PhantomData,
            },
            &graph_info,
        )
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

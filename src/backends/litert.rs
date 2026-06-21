use std::collections::HashMap;
use std::fmt;

use crate::GraphInfo;
use crate::backend_selection::DeviceType;
use crate::error::{Error, Result};
use crate::mlcontext::{
    ListDevices, MLBackendBuilder, MLBackendContext, MLGraph, MLTensor, MLTensorDescriptor,
};

pub(crate) struct LiteRtTensor {
    memory: Vec<u8>,
}

pub(crate) struct LiteRtGraph;

impl fmt::Debug for LiteRtGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiteRtGraph").finish()
    }
}

pub(crate) struct LiteRtContext {
    tensors: Vec<LiteRtTensor>,
    device_type: DeviceType,
}

impl fmt::Debug for LiteRtContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiteRtContext")
            .field("num_tensors", &self.tensors.len())
            .field("device_type", &self.device_type)
            .finish()
    }
}

impl LiteRtContext {
    pub(crate) fn new_from_device_type(device_type: DeviceType) -> Result<Self> {
        Ok(Self {
            tensors: Vec::new(),
            device_type: device_type,
        })
    }
}

fn tensor_byte_len(descriptor: &MLTensorDescriptor) -> Result<usize> {
    let bits = descriptor.data_type().rustnn_element_size_bits();
    let elements: usize = descriptor
        .shape()
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| Error::GraphDispatchError {
            source: "tensor element count overflow".into(),
        })? as usize;
    Ok(bits * elements / 8)
}

impl ListDevices for LiteRtContext {
    fn list_devices() -> Vec<crate::backend_selection::BackendDevice> {
        Vec::new()
    }
}

impl<'context> MLBackendContext<'context> for LiteRtContext {
    fn accelerated(&self) -> bool {
        self.device_type != DeviceType::Cpu
    }

    fn create_builder<'builder>(
        &mut self,
    ) -> Result<Box<dyn MLBackendBuilder<'context, 'builder> + 'builder>>
    where
        'context: 'builder,
    {
        Ok(Box::new(LiteRtBuilder))
    }

    fn create_tensor(&mut self, descriptor: &MLTensorDescriptor) -> Result<MLTensor> {
        let n = tensor_byte_len(descriptor)?.max(1);
        let memory = vec![0u8; n];
        self.tensors.push(LiteRtTensor { memory });
        Ok(MLTensor {
            id: self.tensors.len() - 1,
            constant: false,
            descriptor: descriptor.clone(),
        })
    }

    fn rustnn_resize_tensor(&mut self, _tensor: &mut MLTensor, _new_shape: &[u64]) -> Result<()> {
        todo!("Not Implemented yet.")
    }

    fn rustnn_set_tensor_capacity(
        &mut self,
        _tensor: &mut MLTensor,
        _max_shape: &[u64],
    ) -> Result<()> {
        todo!("Not Implemented yet.")
    }

    fn create_constant_tensor(
        &mut self,
        descriptor: &MLTensorDescriptor,
        input_data: &[u8],
    ) -> Result<MLTensor> {
        let mut tensor = self.create_tensor(descriptor)?;
        tensor.constant = true;
        self.write_tensor(&tensor, input_data)
            .map_err(|e| Error::TensorCreationError {
                source: e.into(),
                descriptor: descriptor.clone(),
            })?;
        Ok(tensor)
    }

    fn read_tensor(&mut self, tensor: &MLTensor, array: &mut [u8]) -> Result<()> {
        let host = &self.tensors[tensor.id].memory;
        let logical = tensor_byte_len(tensor.descriptor())?;
        if array.len() < logical {
            return Err(Error::TensorReadError {
                source: format!(
                    "buffer too small: need {} logical bytes, got {}",
                    logical,
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

    fn write_tensor(&mut self, tensor: &MLTensor, array: &[u8]) -> Result<()> {
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
        host[..array.len()].copy_from_slice(array);
        Ok(())
    }

    fn dispatch(
        &mut self,
        _graph: &mut MLGraph,
        _inputs: &HashMap<&str, &MLTensor>,
        _outputs: &HashMap<&str, &MLTensor>,
    ) -> Result<()> {
        Err(Error::GraphDispatchError {
            source: "LiteRT dispatch not yet implemented".into(),
        })
    }
}

pub(crate) struct LiteRtBuilder;

impl fmt::Debug for LiteRtBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiteRtBuilder").finish()
    }
}

impl<'context, 'builder> MLBackendBuilder<'context, 'builder> for LiteRtBuilder {
    fn build(&mut self, _graph_info: GraphInfo) -> Result<MLGraph<'context>> {
        Err(Error::GraphBuildError {
            source: "LiteRT builder not yet implemented".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_enums::MLOperandDataType;

    fn make_desc(dt: MLOperandDataType, shape: Vec<u64>) -> MLTensorDescriptor {
        MLTensorDescriptor::new(dt, shape)
    }

    #[test]
    fn test_context_new() {
        let ctx = LiteRtContext::new_from_device_type(DeviceType::Cpu).unwrap();
        assert_eq!(ctx.tensors.len(), 0);
    }

    #[test]
    fn test_create_tensor() {
        let mut ctx = LiteRtContext::new_from_device_type(DeviceType::Cpu).unwrap();
        let desc = make_desc(MLOperandDataType::Float32, vec![1, 4]);
        let tensor = ctx.create_tensor(&desc).unwrap();
        assert_eq!(tensor.id, 0);
        assert!(!tensor.constant);
        assert_eq!(ctx.tensors.len(), 1);
        assert_eq!(ctx.tensors[0].memory.len(), 16);
    }

    #[test]
    fn test_write_and_read_tensor() {
        let mut ctx = LiteRtContext::new_from_device_type(DeviceType::Cpu).unwrap();
        let desc = make_desc(MLOperandDataType::Float32, vec![2]);
        let tensor = ctx.create_tensor(&desc).unwrap();

        let data: Vec<u8> = vec![0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x40];
        ctx.write_tensor(&tensor, &data).unwrap();

        let mut read_buf = vec![0u8; 8];
        ctx.read_tensor(&tensor, &mut read_buf).unwrap();
        assert_eq!(read_buf, data);
    }
}

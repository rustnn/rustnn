use crate::error::Result;
use std::{collections::HashMap, fmt::Display};

use crate::{
    backend_selection::{select_backend, select_backend_by_gpu},
    operator_enums::MLOperandDataType,
};

// Backend traits

// could make public later if interface stabilized
pub(crate) trait MLBackendContext: std::fmt::Debug {
    fn accelerated(&self) -> bool;
}

// types for MLContext

// aka WebGpuDevice
#[derive(Debug)]
pub struct GpuDevice {}

#[derive(Debug)]
pub struct MLContextLostInfo {
    message: String,
}

impl MLContextLostInfo {
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for MLContextLostInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

/// https://www.w3.org/TR/webnn/#api-mltensor
#[derive(Debug)]
pub struct MLTensor {
    id: u64,
    constant: bool,
    /// internal slots as per https://www.w3.org/TR/webnn/#api-mltensor
    descriptor: MLTensorDescriptor,
    //context: &'context MLContext, // todo, omit context?
    //// pending promises, need to be canceled when tensor is destroyed
}

impl MLTensor {
    pub fn data_type(&self) -> MLOperandDataType {
        self.descriptor.data_type
    }
    pub fn shape(&self) -> &[u64] {
        &self.descriptor.shape
    }
    pub fn readable(&self) -> bool {
        self.descriptor.readable
    }
    pub fn writable(&self) -> bool {
        self.descriptor.writable
    }
    pub fn constant(&self) -> bool {
        self.constant
    }
    // TODO: or replace by Rust's drop?
    pub fn destoy(&self) {
        todo!() // destroying needs to cancel pending promises
    }
    pub fn destroyed(&self) -> bool {
        todo!() // JS has a isDestroyed method
    }
}

#[derive(Debug)]
pub struct MLGraph {}
#[derive(Debug)]
pub struct MLOpSupportLimits {}

// https://www.w3.org/TR/webnn/#dictdef-mloperanddescriptor
#[derive(Debug, Eq, PartialEq, Default)]
pub struct MLOperandDescriptor {
    data_type: MLOperandDataType,
    shape: Vec<u64>,
}

impl MLOperandDescriptor {
    pub fn new(data_type: MLOperandDataType, shape: Vec<u64>) -> Self {
        Self { data_type, shape }
    }

    pub fn data_type(&self) -> MLOperandDataType {
        self.data_type
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn set_data_type(&mut self, data_type: MLOperandDataType) {
        self.data_type = data_type;
    }

    pub fn set_shape(&mut self, shape: Vec<u64>) {
        self.shape = shape;
    }
}

#[derive(Debug, Default, PartialEq, Eq, Copy, Clone)]
pub enum MLPowerPreference {
    #[default]
    Default,
    HighPerformance,
    LowPower,
}

/// https://www.w3.org/TR/webnn/#dictdef-mlcontextoptions
/// https://www.w3.org/TR/webnn/#api-ml
///
/// From specs: Note: MLContextOptions is under active development, and the design is expected to change,
#[derive(Debug, Eq, PartialEq)]
pub struct MLContextOptions {
    pub(crate) power_preference: MLPowerPreference,
    pub(crate) accelerated: bool,
    // could add our own experimental options
    // could add device_type (CPU, NPU, GPU) like pywebnn
    // could add backend preference
}

/// https://www.w3.org/TR/webnn/#dictdef-mltensordescriptor
#[derive(Debug, Eq, PartialEq, Default)]
pub struct MLTensorDescriptor {
    operand_descriptor: MLOperandDescriptor,
    readable: bool,
    writable: bool,
}

impl std::ops::Deref for MLTensorDescriptor {
    type Target = MLOperandDescriptor;

    fn deref(&self) -> &Self::Target {
        &self.operand_descriptor
    }
}

impl MLTensorDescriptor {
    pub fn new(data_type: MLOperandDataType, shape: Vec<u64>) -> Self {
        Self {
            operand_descriptor: MLOperandDescriptor { data_type, shape },
            writable: false,
            readable: false,
        }
    }
    pub fn readable(&self) -> bool {
        self.readable
    }

    pub fn writable(&self) -> bool {
        self.writable
    }

    pub fn set_writable(&mut self, writable: bool) {
        self.writable = writable;
    }

    pub fn set_readable(&mut self, readable: bool) {
        self.readable = readable;
    }

    pub fn operand_descriptor(&self) -> &MLOperandDescriptor {
        &self.operand_descriptor
    }

    pub fn set_operand_descriptor(&mut self, operand_descriptor: MLOperandDescriptor) {
        self.operand_descriptor = operand_descriptor;
    }
}

#[derive(Debug)]
pub struct MLContext {
    backend: Box<dyn MLBackendContext>,
}

impl MLContext {
    // those are methods on `create_context`
    pub async fn create(options: &MLContextOptions) -> Result<Self> {
        let desc = select_backend(options)?;
        match desc {
            crate::backend_selection::BackendDesc::OnnxRuntime { device_type } => todo!(),
            crate::backend_selection::BackendDesc::TrtxRuntime { cuda_device_idx } => todo!(),
            crate::backend_selection::BackendDesc::CoremlRuntime { device_type } => todo!(),
            crate::backend_selection::BackendDesc::WebNN => todo!(),
            crate::backend_selection::BackendDesc::ExternalBackend => todo!(),
        }
    }

    pub async fn create_from_gpu_device(gpu_device: &GpuDevice) -> Result<Self> {
        let desc = select_backend_by_gpu(gpu_device)?;
        match desc {
            crate::backend_selection::BackendDesc::OnnxRuntime { device_type } => todo!(),
            crate::backend_selection::BackendDesc::TrtxRuntime { cuda_device_idx } => todo!(),
            crate::backend_selection::BackendDesc::CoremlRuntime { device_type } => todo!(),
            crate::backend_selection::BackendDesc::WebNN => todo!(),
            crate::backend_selection::BackendDesc::ExternalBackend => todo!(),
        }
    }
    pub fn accelerated(&self) -> bool {
        self.backend.accelerated()
    }

    pub async fn lost(&self) -> MLContextLostInfo {
        todo!()
    }

    pub async fn create_constant_tensor(
        &mut self,
        descriptor: &MLOperandDescriptor,
        input_data: &[u8], // with owned variant?
    ) -> MLTensor {
        todo!()
    }

    pub async fn create_tensor(&mut self, descriptor: &MLTensorDescriptor) -> MLTensor {
        todo!()
    }

    // omit destroy()? We're not JS, objects can be destroyed via drop, we could do destroy stuff in Drop
    pub fn destroy(self) {
        todo!()
    }

    pub fn dispatch(
        &mut self,
        graph: &MLGraph,
        inputs: &HashMap<&str, MLTensor>,
        outputs: &HashMap<&str, MLTensor>,
    ) {
        todo!()
    }

    pub fn op_support_limits(&self) -> MLOpSupportLimits {
        todo!()
    }

    pub async fn read_tensor<T>(this: &MLContext, tensor: &MLTensor, array: &mut [T]) {
        todo!();
    }

    pub async fn write_tensor<T>(this: &MLContext, tensor: &MLTensor, array: &[T]) {
        todo!();
    }
}

#[cfg(test)]
mod test {
    use crate::mlcontext::*;

    #[test]
    fn test_tensor_desc() {
        let default_operand_desc = MLOperandDescriptor::default();
        let mut default_tensor_desc = MLTensorDescriptor::default();
        assert_eq!(default_tensor_desc.shape(), default_operand_desc.shape());
        assert_eq!(
            default_tensor_desc.data_type(),
            default_operand_desc.data_type()
        );
        assert_eq!(default_tensor_desc.data_type(), MLOperandDataType::Float32);
        assert_eq!(default_tensor_desc.writable(), false);
        assert_eq!(default_tensor_desc.readable(), false);
        default_tensor_desc.set_writable(true);
        assert_eq!(default_tensor_desc.writable(), true);
        default_tensor_desc.set_writable(true);
        assert_eq!(default_tensor_desc.writable(), true);

        let desc = MLTensorDescriptor::new(MLOperandDataType::Float16, vec![3, 4]);
        let op_desc = MLOperandDescriptor::new(MLOperandDataType::Float16, vec![3, 4]);
        assert_eq!(desc.operand_descriptor(), op_desc);
    }
}

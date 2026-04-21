use std::{collections::HashMap, fmt::Display};

use crate::operator_enums::MLOperandDataType;

// Backend traits

// could make public later if interface stabilized
pub(crate) trait MLBackendContext: std::fmt::Debug {
    fn accelerated(&self) -> bool;
}

pub(crate) trait MLBackendTensor: std::fmt::Debug {}

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

#[derive(Debug)]
pub struct MLTensor {
    backend: Box<dyn MLBackendTensor>,
}

#[derive(Debug)]
pub struct MLGraph {}
#[derive(Debug)]
pub struct MLOpSupportLimits {}

// https://www.w3.org/TR/webnn/#dictdef-mloperanddescriptor
#[derive(Debug, Eq, PartialEq)]
pub struct MLOperandDescriptor {
    data_type: MLOperandDataType,
    shape: Vec<u64>,
}

impl MLOperandDescriptor {
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

#[derive(Debug, Default, PartialEq, Eq)]
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
    power_preference: MLPowerPreference,
    accelerated: bool,
}

/// https://www.w3.org/TR/webnn/#dictdef-mltensordescriptor
#[derive(Debug, Eq, PartialEq)]
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
}

#[derive(Debug)]
pub struct MLContext {
    backend: Box<dyn MLBackendContext>,
}

impl MLContext {
    pub async fn create_context(options: &MLContextOptions) -> Self {
        todo!();
    }

    pub async fn create_context_with_gpu_device(gpu_device: &GpuDevice) -> Self {
        todo!()
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

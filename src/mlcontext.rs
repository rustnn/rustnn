#![allow(dead_code, unused_variables)]

use log::info;

#[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
use crate::executors::trtx::{TrtxContext, TrtxGraph};
use crate::{
    GraphInfo,
    backend_selection::BackendDevice,
    backends::ort::OrtContext,
    error::{Error, Result},
};
use std::{collections::HashMap, fmt::Display};

use crate::{
    backend_selection::{select_backend, select_backend_by_gpu},
    operator_enums::MLOperandDataType,
};

// Backend traits

pub(crate) trait ListDevices {
    // TODO: should probably be a Result or just be an empty Vec when something is not working the
    fn list_devices() -> Vec<BackendDevice>;
}

// could make public later if interface stabilized
pub(crate) trait MLBackendContext<'context>: std::fmt::Debug {
    fn accelerated(&self) -> bool;
    fn create_builder(&mut self) -> Result<Box<dyn MLBackendBuilder<'context> + 'context>>;
    fn create_tensor(&mut self, descriptor: &MLTensorDescriptor) -> Result<MLTensor>;
    fn create_constant_tensor(
        &mut self,
        descriptor: &MLTensorDescriptor,
        input_data: &[u8],
    ) -> Result<MLTensor>;
    /*async*/
    fn read_tensor(&mut self, tensor: &MLTensor, array: &mut [u8]) -> Result<()>;
    /*async*/
    fn write_tensor(&mut self, tensor: &MLTensor, array: &[u8]) -> Result<()>;
    fn dispatch(
        &mut self,
        graph: &mut MLGraph,
        inputs: &HashMap<&str, MLTensor>,
        outputs: &HashMap<&str, MLTensor>,
    ) -> Result<()>;
}

pub(crate) trait MLBackendBuilder<'context>: std::fmt::Debug {
    /*async*/
    fn build(&mut self, outputs: &HashMap<&str, MLOperand>) -> Result<MLGraph<'context>>;
    fn load_graph(&mut self, graph: &'context GraphInfo) -> Result<()>;
}

// can be made a Box<dyn better_any::Tid<'context> + 'context> for dynamic dispatch
// dyn Any does not work since Any requires 'static
#[derive(Debug)]
pub(crate) enum MLBackendGraph<'context> {
    #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
    TrtxEngine(TrtxGraph<'context>),
    OnnxSession(std::marker::PhantomData<&'context u8>),
}

impl<'context> MLBackendGraph<'context> {
    #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
    pub(crate) fn as_trtx_engine_mut(&mut self) -> Option<&mut TrtxGraph<'context>> {
        if let Self::TrtxEngine(v) = self {
            Some(v)
        } else {
            None
        }
    }
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
#[derive(Debug, Clone)]
pub struct MLTensor {
    pub(crate) id: usize,
    pub(crate) constant: bool,
    /// internal slots as per https://www.w3.org/TR/webnn/#api-mltensor
    pub(crate) descriptor: MLTensorDescriptor,
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
pub struct MLGraph<'context> {
    pub(crate) backend: MLBackendGraph<'context>,
}
#[derive(Debug)]
pub struct MLOpSupportLimits {}

// https://www.w3.org/TR/webnn/#dictdef-mloperanddescriptor
#[derive(Debug, Eq, PartialEq, Default, Clone)]
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
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub struct MLContextOptions {
    pub(crate) power_preference: MLPowerPreference,
    pub(crate) accelerated: bool,
    // could add our own experimental options
    // could add device_type (CPU, NPU, GPU) like pywebnn
    // could add backend preference
}

/// https://www.w3.org/TR/webnn/#dictdef-mltensordescriptor
#[derive(Debug, Eq, PartialEq, Default, Clone)]
pub struct MLTensorDescriptor {
    operand_descriptor: MLOperandDescriptor,
    readable: bool,
    writable: bool,
}

#[derive(Debug, Eq, PartialEq, Default, Clone)]
pub struct MLOperand {
    pub(crate) id: usize,
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
pub struct MLContext<'context> {
    backend: Box<dyn MLBackendContext<'context> + 'context>,
}

impl<'context> MLContext<'context> {
    // those are methods on `create_context`
    //pub async
    pub fn create(options: &MLContextOptions) -> Result<Self> {
        let desc = select_backend(options)?;
        info!("Backend selected: {desc:?}");
        let backend: Box<dyn MLBackendContext<'context> + 'context> = match desc {
            crate::backend_selection::BackendDevice::OnnxDevice { device_type } => {
                Box::new(OrtContext::new_from_ty(device_type)?)
            }
            #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
            crate::backend_selection::BackendDevice::TrtxDevice { cuda_device_idx } => Box::new(
                TrtxContext::new(cuda_device_idx)
                    .map_err(|e| Error::ContextCreationError { source: e.into() })?,
            ),
            crate::backend_selection::BackendDevice::CoremlDevice { device_type } => todo!(),
            crate::backend_selection::BackendDevice::WebNN => todo!(),
            crate::backend_selection::BackendDevice::ExternalBackend => todo!(),
            _ => todo!(),
        };
        Ok(Self { backend })
    }

    #[allow(unreachable_code)]
    pub async fn create_from_gpu_device(gpu_device: &GpuDevice) -> Result<Self> {
        let desc = select_backend_by_gpu(gpu_device)?;
        let backend = match desc {
            crate::backend_selection::BackendDevice::OnnxDevice { device_type } => todo!(),
            crate::backend_selection::BackendDevice::TrtxDevice { cuda_device_idx } => todo!(),
            crate::backend_selection::BackendDevice::CoremlDevice { device_type } => todo!(),
            crate::backend_selection::BackendDevice::WebNN => todo!(),
            crate::backend_selection::BackendDevice::ExternalBackend => todo!(),
        };
        Ok(Self { backend })
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

    // async
    pub fn create_tensor(&mut self, descriptor: &MLTensorDescriptor) -> Result<MLTensor> {
        self.backend.create_tensor(descriptor)
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

    //async
    pub fn read_tensor<T: bytemuck::Pod>(
        &mut self,
        tensor: &MLTensor,
        array: &mut [T],
    ) -> Result<()> {
        if !tensor.readable() {
            panic!("Attempt to write non-readable tensor: {tensor:?}");
        }
        self.backend
            .read_tensor(tensor, bytemuck::cast_slice_mut(array))
    }

    //async
    pub fn write_tensor<T: bytemuck::Pod>(&mut self, tensor: &MLTensor, array: &[T]) -> Result<()> {
        if !tensor.writable() {
            panic!("Attempt to write non-writeable tensor: {tensor:?}");
        }
        self.backend
            .write_tensor(tensor, bytemuck::cast_slice(array))
    }
}

#[derive(Debug)]
pub struct MLGraphBuilder<'context> {
    backend: Box<dyn MLBackendBuilder<'context> + 'context>,
}

impl<'context> MLGraphBuilder<'context> {
    fn new(context: &'context mut MLContext<'context>) -> Result<Self> {
        let backend = context.backend.create_builder()?;
        Ok(Self { backend })
    }

    fn load_graph(&mut self, graph: &'context GraphInfo) -> Result<()> {
        self.backend.load_graph(graph)
    }

    /*async*/
    fn build(&mut self, outputs: &HashMap<&str, MLOperand>) -> Result<MLGraph<'context>> {
        self.backend.build(outputs)
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
        assert!(!default_tensor_desc.writable());
        assert!(!default_tensor_desc.readable());
        default_tensor_desc.set_writable(true);
        assert!(default_tensor_desc.writable());
        default_tensor_desc.set_writable(true);
        assert!(default_tensor_desc.writable());

        let desc = MLTensorDescriptor::new(MLOperandDataType::Float16, vec![3, 4]);
        let op_desc = MLOperandDescriptor::new(MLOperandDataType::Float16, vec![3, 4]);
        assert_eq!(*desc.operand_descriptor(), op_desc);
    }

    #[test]
    fn test_create_context() {
        let _ = pretty_env_logger::try_init();
        let mut context = MLContext::create(&MLContextOptions {
            power_preference: MLPowerPreference::Default,
            accelerated: true,
        })
        .unwrap();
        dbg!(&context);
        let mut desc = MLTensorDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            [2, 2].to_vec(),
        );
        desc.set_readable(true);
        desc.set_writable(true);
        let tensor = context.create_tensor(&desc).unwrap();

        let upload = vec![1.0f32, 2., 3., 4.];
        let mut download = vec![0.0f32; 4];
        context.write_tensor(&tensor, &upload).unwrap();
        context.read_tensor(&tensor, &mut download).unwrap();
        assert_eq!(&upload, &download);
    }
}

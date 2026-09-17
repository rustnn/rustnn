//! WebNN `MLContext`, `MLGraph`, `MLTensor` and `MLOperand`.
//!
//! [`MLContext::create`] selects a backend from [`MLContextOptions`] (see
//! [`crate::backend_selection`]), [`MLGraphBuilder`] compiles a graph for that backend, and
//! [`MLContext::dispatch`] executes it on tensors created by [`MLContext::create_tensor`].
//! Methods that are not part of the WebNN specification carry a `rustnn_` prefix.
//!
//! Async parts of the JavaScript API (`compute`, `readTensor`, `writeTensor`) are
//! synchronous here; the corresponding methods are marked `//async` in the source until an
//! async API is settled. Methods that still `todo!()` are listed in `docs/development/implementation-status.md`.

#![allow(dead_code, unused_variables)]

use log::{debug, info};

use crate::GraphInfo;
use crate::OperandDescriptor;
pub use crate::backend_selection::{Backend, BackendDevice, DeviceType};
#[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
use crate::backends::trtx::TrtxGraph;
use crate::error::Error;
use crate::error::Result;
use crate::graph::{DataType, Dimension, Operand, get_static_or_max_size};
use crate::mlgraphbuilder::get_operand;
use crate::operator_options::OperandIndex;
use crate::runtime_checks::{RuntimeShapeState, TensorKind};

use crate::backends::cann::CannContext;
use crate::backends::coreml::CoremlContext;
use crate::backends::litert::LiteRtContext;
use crate::backends::ort::OrtContext;
use crate::backends::trtx::TrtxContext;
use std::collections::BTreeMap;
use std::{collections::HashMap, fmt::Display, marker::PhantomData};

pub use crate::mlcontextoptions::{
    CoremlOptions, LiteRtOptions, MLContextOptions, MLPowerPreference, OrtOptions, RustNNOptions,
    TrtxOptions,
};

/// <https://www.w3.org/TR/webnn/#typedefdef-mlnamedtensors>
pub type MLNamedTensors<'names> = BTreeMap<&'names str, &'names MLTensor>;
/// <https://www.w3.org/TR/webnn/#typedefdef-mlnamedoperands>
pub type MLNamedOperands<'names> = BTreeMap<&'names str, MLOperand>;

fn validate_unique_tensor_bindings(
    inputs: &MLNamedTensors,
    outputs: &MLNamedTensors,
) -> Result<()> {
    let mut all_tensor_ids = HashMap::new();
    for (&name, &tensor) in inputs.iter().chain(outputs.iter()) {
        if let Some(other_name) = all_tensor_ids.insert(tensor.id, name) {
            return Err(Error::DuplicateTensorBinding {
                aliased_tensor: tensor.clone(),
                first_binding: other_name.to_string(),
                other_binding: name.to_string(),
            });
        }
    }
    Ok(())
}

pub use crate::mlgraphbuilder::MLGraphBuilder;
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
pub(crate) trait MLBackendContext<'context>: std::fmt::Debug + Send + Sync {
    fn accelerated(&self) -> bool;
    fn create_builder<'builder>(
        &mut self,
    ) -> Result<Box<dyn MLBackendBuilder<'context, 'builder> + 'builder>>
    where
        'context: 'builder;
    fn create_tensor(&mut self, descriptor: &MLTensorDescriptor) -> Result<MLTensor>;
    fn rustnn_resize_tensor(&mut self, tensor: &mut MLTensor, new_shape: &[u64]) -> Result<()>;
    fn rustnn_set_tensor_capacity(
        &mut self,
        tensor: &mut MLTensor,
        max_shape: &[u64],
    ) -> Result<()>;
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
        inputs: &MLNamedTensors,
        outputs: &MLNamedTensors,
    ) -> Result<()>;
}

pub(crate) trait MLBackendBuilder<'context, 'builder>: std::fmt::Debug + Send {
    /*async*/
    fn build(&mut self, graph: GraphInfo) -> Result<MLGraph<'context>>;
}

// can be made a Box<dyn better_any::Tid<'context> + 'context> for dynamic dispatch
// dyn Any does not work since Any requires 'static
#[derive(Debug)]
pub(crate) enum MLBackendGraph<'context> {
    #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
    TrtxEngine(TrtxGraph<'context>),
    #[cfg(feature = "onnx-runtime")]
    OnnxSession(
        crate::backends::ort::OrtGraph,
        std::marker::PhantomData<&'context ()>,
    ),
    #[cfg(feature = "coreml-runtime")]
    CoremlModel(crate::backends::coreml::CoremlGraph),
    #[cfg(feature = "litert-runtime")]
    LiteRtGraph(crate::backends::litert::LiteRtGraph),
    #[cfg(any(feature = "cann-runtime", feature = "cann-runtime-mock"))]
    CannEngine(crate::backends::cann::CannGraph),
    PhantomData(PhantomData<&'context u8>),
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

    #[cfg(feature = "onnx-runtime")]
    pub(crate) fn as_onnx_session_mut(&mut self) -> Option<&mut crate::backends::ort::OrtGraph> {
        match self {
            Self::OnnxSession(g, _) => Some(g),
            _ => None,
        }
    }

    #[cfg(feature = "coreml-runtime")]
    pub(crate) fn as_coreml_model(&self) -> Option<&crate::backends::coreml::CoremlGraph> {
        if let Self::CoremlModel(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

// types for MLContext

/// Placeholder for the WebGPU device of `ML.createContext(GPUDevice)`; not implemented yet.
// aka WebGpuDevice
#[derive(Debug)]
pub struct GpuDevice {}

/// Reason a context was lost. <https://www.w3.org/TR/webnn/#api-mlcontext>
#[derive(Debug)]
pub struct MLContextLostInfo {
    message: String,
}

impl MLContextLostInfo {
    /// Human-readable description of why the context was lost.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for MLContextLostInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

/// <https://www.w3.org/TR/webnn/#api-mltensor>
#[derive(Debug, Clone)]
pub struct MLTensor {
    pub(crate) id: usize,
    pub(crate) constant: bool,
    /// internal slots as per <https://www.w3.org/TR/webnn/#api-mltensor>
    pub(crate) descriptor: MLTensorDescriptor,
    //context: &'context MLContext, // todo, omit context?
    //// pending promises, need to be canceled when tensor is destroyed
}

impl MLTensor {
    /// Element data type.
    pub fn data_type(&self) -> MLOperandDataType {
        self.descriptor.data_type
    }
    /// Active shape; changed by `MLContext::rustnn_resize_tensor` for dynamic graphs.
    pub fn shape(&self) -> &[u64] {
        &self.descriptor.shape
    }
    /// Whether `MLContext::read_tensor` may read this tensor.
    pub fn readable(&self) -> bool {
        self.descriptor.readable
    }
    /// Whether `MLContext::write_tensor` may write this tensor.
    pub fn writable(&self) -> bool {
        self.descriptor.writable
    }
    /// Whether the tensor was created as a constant.
    pub fn constant(&self) -> bool {
        self.constant
    }
    /// Not implemented: dropping the tensor releases it. <https://www.w3.org/TR/webnn/#api-mltensor-destroy>
    // TODO: or replace by Rust's drop?
    pub fn destroy(&self) {
        todo!() // destroying needs to cancel pending promises
    }
    /// Not implemented.
    pub fn destroyed(&self) -> bool {
        todo!() // JS has a isDestroyed method
    }

    /// Bytes a host buffer needs for `read_tensor` / `write_tensor` (4-bit types are packed).
    pub fn rustnn_required_bytes(&self) -> usize {
        self.descriptor.rustnn_required_bytes()
    }

    pub(crate) fn descriptor(&self) -> &MLTensorDescriptor {
        &self.descriptor
    }
}

/// A graph compiled for one backend by [`MLGraphBuilder::build`]. <https://www.w3.org/TR/webnn/#api-mlgraph>
///
/// `input_descriptors` and `output_descriptors` are the named graph inputs and outputs that
/// [`MLContext::dispatch`] validates tensor bindings against.
#[derive(Debug)]
pub struct MLGraph<'context> {
    pub(crate) backend: MLBackendGraph<'context>,

    /// Graph inputs by name, as declared with `MLGraphBuilder::input`.
    pub input_descriptors: HashMap<String, OperandDescriptor>,
    /// Graph outputs by name, as passed to `MLGraphBuilder::build`.
    pub output_descriptors: HashMap<String, OperandDescriptor>,
}

impl<'context> MLGraph<'context> {
    pub(crate) fn new(backend: MLBackendGraph<'context>, graph_info: &GraphInfo) -> Result<Self> {
        let (input_descriptors, output_descriptors) = graph_info
            .io_binding_maps()
            .map_err(|e| Error::GraphBuildError { source: e.into() })?;
        Ok(Self {
            backend,
            input_descriptors,
            output_descriptors,
        })
    }

    fn operand_descriptors(
        operands: &HashMap<String, Operand>,
    ) -> HashMap<String, OperandDescriptor> {
        operands
            .iter()
            .map(|(name, op)| (name.clone(), op.descriptor.clone()))
            .collect()
    }

    fn verify_dispatch_bindings(
        &mut self,
        inputs: &MLNamedTensors,
        outputs: &MLNamedTensors,
    ) -> Result<()> {
        let input_shapes: HashMap<String, Vec<usize>> = inputs
            .iter()
            .map(|(&name, tensor)| {
                (
                    name.to_string(),
                    tensor.shape().iter().map(|&d| d as usize).collect(),
                )
            })
            .collect();
        let output_shapes: HashMap<String, Vec<usize>> = outputs
            .iter()
            .map(|(&name, tensor)| {
                (
                    name.to_string(),
                    tensor.shape().iter().map(|&d| d as usize).collect(),
                )
            })
            .collect();

        let mut runtime_shape_state = RuntimeShapeState::new();
        runtime_shape_state
            .validate_named_shapes(&input_shapes, &self.input_descriptors, TensorKind::Input)
            .map_err(|e| Error::GraphDispatchError { source: e.into() })?;
        runtime_shape_state
            .validate_named_shapes(&output_shapes, &self.output_descriptors, TensorKind::Output)
            .map_err(|e| Error::GraphDispatchError { source: e.into() })?;

        for (&name, tensor) in inputs {
            let expected = self.input_descriptors.get(name).expect("validated above");
            if DataType::from(tensor.data_type()) != expected.data_type {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "input '{name}' data type mismatch (expected {:?}, got {:?})",
                        expected.data_type,
                        tensor.data_type()
                    )
                    .into(),
                });
            }
        }

        for (&name, tensor) in outputs {
            let expected = self.output_descriptors.get(name).expect("validated above");
            if DataType::from(tensor.data_type()) != expected.data_type {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{name}' data type mismatch (expected {:?}, got {:?})",
                        expected.data_type,
                        tensor.data_type()
                    )
                    .into(),
                });
            }
        }

        Ok(())
    }
}

/// Placeholder for `MLContext.opSupportLimits()`; not implemented yet.
#[derive(Debug)]
pub struct MLOpSupportLimits {}

/// Data type and shape of a graph operand. <https://www.w3.org/TR/webnn/#dictdef-mloperanddescriptor>
///
/// Shapes are `u64` here (the specification uses `unsigned long`).
#[derive(Debug, Eq, PartialEq, Default, Clone)]
pub struct MLOperandDescriptor {
    data_type: MLOperandDataType,
    shape: Vec<u64>, // TODO: this is u64 instead of WebNN's u32. u32 is screaming for problems on desktop
}

impl From<&MLOperandDescriptor> for OperandDescriptor {
    fn from(val: &MLOperandDescriptor) -> Self {
        OperandDescriptor {
            data_type: val.data_type.into(),
            shape: val
                .shape
                .iter()
                .map(|s| Dimension::Static(*s as u32))
                .collect(),
            pending_permutation: Default::default(),
        }
    }
}

impl MLOperandDescriptor {
    /// Descriptor for `data_type` and `shape` (an empty shape is a scalar).
    pub fn new(data_type: MLOperandDataType, shape: Vec<u64>) -> Self {
        Self { data_type, shape }
    }

    /// Element data type.
    pub fn data_type(&self) -> MLOperandDataType {
        self.data_type
    }

    /// Dimensions, outermost first.
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Replace the data type.
    pub fn set_data_type(&mut self, data_type: MLOperandDataType) {
        self.data_type = data_type;
    }

    /// Replace the shape.
    pub fn set_shape(&mut self, shape: Vec<u64>) {
        self.shape = shape;
    }

    pub(crate) fn rustnn_required_bytes(&self) -> usize {
        let elements = (self.shape().iter().copied().product::<u64>() as usize).max(1);
        self.data_type().rustnn_storage_byte_length(elements).max(1)
    }
}

/// Shape, data type and host-access flags of an [`MLTensor`].
///
/// A tensor is neither readable nor writable by default; use [`Self::to_readable`] /
/// [`Self::to_writable`] or the setters. <https://www.w3.org/TR/webnn/#dictdef-mltensordescriptor>
#[derive(Debug, Eq, PartialEq, Default, Clone)]
pub struct MLTensorDescriptor {
    operand_descriptor: MLOperandDescriptor,
    readable: bool,
    writable: bool,
}

/// Handle to an operand recorded by an [`MLGraphBuilder`]. <https://www.w3.org/TR/webnn/#api-mloperand>
///
/// The handle is an index into the builder's graph; shape and data type are looked up with
/// [`MLOperand::shape`] / [`MLOperand::data_type`] or, while building, with
/// [`MLGraphBuilder::rustnn_operand_shape`] / [`MLGraphBuilder::rustnn_operand_data_type`].
#[derive(Debug, Eq, PartialEq, Default, Copy, Clone, Hash)]
pub struct MLOperand {
    pub(crate) id: usize,
}

// TODO: actually, WebNN requires shape, data_type directly on MLOperand
// would require MLOperand==Operand and we give the user &MLOperand or Rc<MLOperand>
impl MLOperand {
    /// Shape of the operand in `graph` (dynamic dimensions report their maximum size).
    pub fn shape(self, graph: &GraphInfo) -> Result<Vec<u64>> {
        let operand = get_operand(self, graph)?;

        Ok(operand
            .descriptor
            .shape
            .iter()
            .map(|d| get_static_or_max_size(d) as u64)
            .collect())
    }

    /// Data type of the operand in `graph`.
    pub fn data_type(self, graph: &GraphInfo) -> Result<MLOperandDataType> {
        let operand = get_operand(self, graph)?;

        Ok(operand.descriptor.data_type.try_into()?)
    }

    /// Index of the operand in its builder's graph (rustnn extension). Operand fields of the
    /// `ML*Options` structs, such as `MLConv2dOptions::bias`, take this index.
    pub fn rustnn_index(self) -> OperandIndex {
        self.id as OperandIndex
    }
}

impl From<MLOperand> for OperandIndex {
    fn from(operand: MLOperand) -> Self {
        operand.rustnn_index()
    }
}

impl From<u32> for MLOperand {
    fn from(value: u32) -> Self {
        Self { id: value as usize }
    }
}

impl std::ops::Deref for MLTensorDescriptor {
    type Target = MLOperandDescriptor;

    fn deref(&self) -> &Self::Target {
        &self.operand_descriptor
    }
}

impl std::ops::DerefMut for MLTensorDescriptor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.operand_descriptor
    }
}

impl MLTensorDescriptor {
    /// Descriptor with both host-access flags off.
    pub fn new(data_type: MLOperandDataType, shape: Vec<u64>) -> Self {
        Self {
            operand_descriptor: MLOperandDescriptor { data_type, shape },
            writable: false,
            readable: false,
        }
    }
    /// Descriptor with the data type and shape of an operand descriptor and both flags off.
    pub fn from_operand_descriptor(operand_descriptor: &MLOperandDescriptor) -> Self {
        Self {
            operand_descriptor: operand_descriptor.clone(),
            writable: false,
            readable: false,
        }
    }
    /// Whether the tensor may be read with `MLContext::read_tensor`.
    pub fn readable(&self) -> bool {
        self.readable
    }

    /// Whether the tensor may be written with `MLContext::write_tensor`.
    pub fn writable(&self) -> bool {
        self.writable
    }

    /// Set the writable flag.
    pub fn set_writable(&mut self, writable: bool) {
        self.writable = writable;
    }

    /// Set the readable flag.
    pub fn set_readable(&mut self, readable: bool) {
        self.readable = readable;
    }

    /// Data type and shape part of the descriptor.
    pub fn operand_descriptor(&self) -> &MLOperandDescriptor {
        &self.operand_descriptor
    }

    /// Replace the data type and shape part of the descriptor.
    pub fn set_operand_descriptor(&mut self, operand_descriptor: MLOperandDescriptor) {
        self.operand_descriptor = operand_descriptor;
    }

    /// Copy with the writable flag set.
    pub fn to_writable(&self) -> Self {
        let mut copy = self.clone();
        copy.writable = true;
        copy
    }

    /// Copy with the readable flag set.
    pub fn to_readable(&self) -> Self {
        let mut copy = self.clone();
        copy.readable = true;
        copy
    }
}

/// Execution context bound to one backend device. <https://www.w3.org/TR/webnn/#api-mlcontext>
///
/// Create it with [`MLContext::create`], build graphs with [`MLGraphBuilder::new`], create
/// tensors with [`MLContext::create_tensor`], and run graphs with [`MLContext::dispatch`].
/// The context is `Send + Sync`, so it can be shared behind a `Mutex` across threads.
// TODO: this is wrong. must be 'context and `for <'builder>` to be valid for each builder lifetime (multiple children!)
#[derive(Debug)]
pub struct MLContext<'context> {
    pub(crate) backend: Box<dyn MLBackendContext<'context> + 'context>,
    pub(crate) device: BackendDevice,
}

impl<'context> MLContext<'context> {
    /// Select a backend device for `options` and create the context.
    ///
    /// The WebNN hints (`accelerated`, power preference) and the rustnn hints
    /// ([`MLContextOptions::with_rustnn_backend_hint`], [`MLContextOptions::with_rustnn_device_hint`])
    /// are resolved by [`crate::backend_selection`]. Fails with
    /// [`Error::NoBackendAvailable`] or [`Error::NoBackendAvailableForBackendHint`] when no
    /// compiled backend can serve the request.
    // those are methods on `create_context`
    //pub async
    pub fn create(options: &MLContextOptions) -> Result<Self> {
        let device = select_backend(options)
            .inspect_err(|e| log::warn!("Error selecting backend: {e:?}"))?;
        info!("Backend selected: {device:?}");
        let backend: Box<dyn MLBackendContext<'context> + 'context> = match device {
            crate::backend_selection::BackendDevice::Onnx { ep_device_idx, .. } => Box::new(
                OrtContext::new_from_ep_idx(ep_device_idx, Some(&options.rustnn_options))?,
            ),
            crate::backend_selection::BackendDevice::Trtx { cuda_device_idx } => Box::new(
                TrtxContext::new(cuda_device_idx, Some(&options.rustnn_options))
                    .map_err(|e| Error::ContextCreationError { source: e.into() })?,
            ),
            crate::backend_selection::BackendDevice::Coreml { device_type } => Box::new(
                CoremlContext::new_from_device_type(device_type, Some(&options.rustnn_options))?,
            ),
            crate::backend_selection::BackendDevice::LiteRt { device_type } => Box::new(
                LiteRtContext::new_from_device_type(device_type, Some(&options.rustnn_options))?,
            ),
            crate::backend_selection::BackendDevice::Cann { device_type } => Box::new(
                CannContext::new_from_device_type(device_type, Some(&options.rustnn_options))?,
            ),
        };
        Ok(Self { backend, device })
    }

    /// Not implemented: `ML.createContext(GPUDevice)`.
    #[expect(unreachable_code)]
    pub async fn create_from_gpu_device(gpu_device: &GpuDevice) -> Result<Self> {
        let device = select_backend_by_gpu(gpu_device)?;
        let backend = match device {
            crate::backend_selection::BackendDevice::Onnx { .. } => todo!(),
            crate::backend_selection::BackendDevice::Trtx { cuda_device_idx } => todo!(),
            crate::backend_selection::BackendDevice::Coreml { device_type } => todo!(),
            crate::backend_selection::BackendDevice::LiteRt { .. } => todo!(),
            crate::backend_selection::BackendDevice::Cann { .. } => todo!(),
        };
        Ok(Self { backend, device })
    }
    /// Whether the selected device is a GPU or NPU rather than a CPU.
    pub fn accelerated(&self) -> bool {
        self.backend.accelerated()
    }

    /// Not implemented: `MLContext.lost`.
    pub async fn lost(&self) -> MLContextLostInfo {
        todo!()
    }

    /// Not implemented: `MLContext.createConstantTensor`. Use `MLGraphBuilder::constant_from_slice`.
    pub async fn create_constant_tensor(
        &mut self,
        descriptor: &MLOperandDescriptor,
        input_data: &[u8], // with owned variant?
    ) -> MLTensor {
        todo!()
    }

    /// Allocate a tensor on the backend device. <https://www.w3.org/TR/webnn/#api-mlcontext-createtensor>
    // async
    pub fn create_tensor(&mut self, descriptor: &MLTensorDescriptor) -> Result<MLTensor> {
        self.backend.create_tensor(descriptor)
    }

    /// Not implemented: dropping the context releases it. <https://www.w3.org/TR/webnn/#api-mlcontext-destroy>
    // omit destroy()? We're not JS, objects can be destroyed via drop, we could do destroy stuff in Drop
    pub fn destroy(self) {
        todo!()
    }

    /// Backend selected for this context (rustnn extension).
    pub fn rustnn_backend(&self) -> Backend {
        self.device.backend()
    }

    /// Backend device selected for this context (rustnn extension).
    pub fn rustnn_device(&self) -> BackendDevice {
        self.device
    }

    /// Device class (CPU, GPU or NPU) of the selected device (rustnn extension).
    pub fn rustnn_device_type(&self) -> DeviceType {
        self.device.device_type()
    }

    /// Execute `graph` with the named input and output tensors.
    /// <https://www.w3.org/TR/webnn/#api-mlcontext-dispatch>
    ///
    /// Before the backend runs, rustnn rejects a tensor bound under two names
    /// ([`Error::DuplicateTensorBinding`]) and checks every binding's name, shape and data
    /// type against the compiled graph ([`Error::GraphDispatchError`]).
    pub fn dispatch(
        &mut self,
        graph: &mut MLGraph,
        inputs: &MLNamedTensors,
        outputs: &MLNamedTensors,
    ) -> crate::error::Result<()> {
        debug!("Dispatch {graph:?}, inputs={inputs:?}, outputs={outputs:?}");
        //https://www.w3.org/TR/webnn/#dom-mlcontext-dispatch
        // spec: 4. If allTensors contains any duplicate items, then throw a TypeError.
        validate_unique_tensor_bindings(inputs, outputs)?;

        graph.verify_dispatch_bindings(inputs, outputs)?;
        self.backend.dispatch(graph, inputs, outputs)
    }

    /// Not implemented: `MLContext.opSupportLimits()`. Backend coverage is in the generated
    /// operator support report.
    pub fn op_support_limits(&self) -> MLOpSupportLimits {
        todo!()
    }

    /// Copy a readable tensor into `array`; `array` must hold exactly
    /// [`MLTensor::rustnn_required_bytes`] bytes. <https://www.w3.org/TR/webnn/#api-mlcontext-readtensor>
    //async
    pub fn read_tensor<T: bytemuck::Pod>(
        &mut self,
        tensor: &MLTensor,
        array: &mut [T],
    ) -> Result<()> {
        debug!(
            "Read {} bytes from tensor {tensor:?}",
            std::mem::size_of_val(array)
        );
        if !tensor.readable() {
            return Err(Error::ReadToNonReadableTensor {
                tensor: tensor.clone(),
            });
        }
        if tensor.rustnn_required_bytes() != std::mem::size_of_val(array) {
            return Err(Error::WrongReadSize {
                read_size: std::mem::size_of_val(array),
                required_size: tensor.rustnn_required_bytes(),
                tensor: tensor.clone(),
            });
        }
        self.backend
            .read_tensor(tensor, bytemuck::cast_slice_mut(array))
    }

    /// Copy `array` into a writable tensor; `array` must hold exactly
    /// [`MLTensor::rustnn_required_bytes`] bytes. <https://www.w3.org/TR/webnn/#api-mlcontext-writetensor>
    //async
    pub fn write_tensor<T: bytemuck::Pod>(&mut self, tensor: &MLTensor, array: &[T]) -> Result<()> {
        debug!(
            "Write {} bytes to tensor {tensor:?}",
            std::mem::size_of_val(array)
        );
        if !tensor.writable() {
            return Err(Error::WriteToNonWritableTensor {
                tensor: tensor.clone(),
            });
        }
        if tensor.rustnn_required_bytes() != std::mem::size_of_val(array) {
            return Err(Error::WrongWriteSize {
                write_size: std::mem::size_of_val(array),
                required_size: tensor.rustnn_required_bytes(),
                tensor: tensor.clone(),
            });
        }
        self.backend
            .write_tensor(tensor, bytemuck::cast_slice(array))
    }

    /// Change the active shape of a tensor without reallocating (rustnn extension for
    /// graphs built with the `dynamic-inputs` feature). The new shape must fit the capacity
    /// reserved with [`Self::rustnn_set_tensor_capacity`].
    pub fn rustnn_resize_tensor(&mut self, tensor: &mut MLTensor, new_shape: &[u64]) -> Result<()> {
        self.backend.rustnn_resize_tensor(tensor, new_shape)
    }

    /// Reserve storage for the largest shape a tensor will take (rustnn extension); see
    /// [`Self::rustnn_resize_tensor`].
    pub fn rustnn_set_tensor_capacity(
        &mut self,
        tensor: &mut MLTensor,
        max_shape: &[u64],
    ) -> Result<()> {
        self.backend.rustnn_set_tensor_capacity(tensor, max_shape)
    }
}

#[cfg(test)]
mod test {
    use crate::{mlcontext::*, mlgraphbuilder::MLGraphBuilder, webnn_json::from_graph_json};

    fn create_add_graph_context_and_graph() -> Option<(MLContext<'static>, MLGraph<'static>)> {
        let contents = r#"
webnn_graph "sample_graph" v1 {
  inputs {
    lhs: f32[2, 2];
  }

  consts {
    rhs: f32[2, 2] @scalar(1.0);
  }

  nodes {
    sum = add(lhs, rhs);
  }

  outputs { sum; }
}"#;

        let _ = pretty_env_logger::try_init();
        let sanitized = crate::loader::sanitize_webnn_identifiers(contents);
        let graph_json = webnn_graph::parser::parse_wg_text(&sanitized).unwrap();
        let graph_info = from_graph_json(&graph_json).unwrap();

        let context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, true));
        if matches!(context, Err(crate::error::Error::NoBackendAvailable { .. })) {
            return None;
        };

        let mut context = context.unwrap();
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let graph = builder.build_graph_info(graph_info).unwrap();
        drop(builder);

        Some((context, graph))
    }

    fn rw_tensor_desc(shape: Vec<u64>) -> MLTensorDescriptor {
        MLTensorDescriptor::new(crate::operator_enums::MLOperandDataType::Float32, shape)
            .to_writable()
            .to_readable()
    }

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
    fn test_backend_hint() {
        let _ = pretty_env_logger::try_init();
        let context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, true)
                .with_rustnn_backend_hint(Backend::Onnx),
        );
        if matches!(
            context,
            Err(crate::error::Error::NoBackendAvailableForBackendHint { .. })
        ) {
            return;
        };
        assert_eq!(context.unwrap().rustnn_backend(), Backend::Onnx);
    }
    #[test]
    fn test_create_context() {
        let _ = pretty_env_logger::try_init();
        let context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, true));
        if matches!(context, Err(crate::error::Error::NoBackendAvailable { .. })) {
            return;
        };

        let mut context = context.unwrap();
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

    #[test]
    fn test_dispatch() {
        let Some((mut context, mut graph)) = create_add_graph_context_and_graph() else {
            return;
        };
        dbg!(&context);
        let desc = rw_tensor_desc([2, 2].to_vec());

        let input_tensor = context.create_tensor(&desc).unwrap();
        let output_tensor = context.create_tensor(&desc).unwrap();
        let mut inputs = MLNamedTensors::new();
        inputs.insert("lhs", &input_tensor);
        let mut outputs = MLNamedTensors::new();
        outputs.insert("sum", &output_tensor);

        let upload = vec![1.0f32, 2., 3., 4.];
        let upload_f64 = vec![1.0, 2., 3., 4.];
        let mut download = vec![0.0f32; 4];
        context
            .write_tensor(&input_tensor, &upload_f64)
            .unwrap_err();
        context.write_tensor(&input_tensor, &upload).unwrap();
        context.dispatch(&mut graph, &inputs, &outputs).unwrap();
        context.read_tensor(&output_tensor, &mut download).unwrap();
        assert_eq!(&vec![2.0f32, 3., 4., 5.], &download);
    }

    #[test]
    fn test_dispatch_rejects_tensor_bound_as_input_and_output() {
        let descriptor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
        let tensor = MLTensor {
            id: 7,
            constant: false,
            descriptor,
        };
        let inputs = MLNamedTensors::from([("lhs", &tensor)]);
        let outputs = MLNamedTensors::from([("sum", &tensor)]);

        let err = validate_unique_tensor_bindings(&inputs, &outputs).unwrap_err();
        std::assert_matches!(
            err,
            crate::error::Error::DuplicateTensorBinding {
                first_binding,
                other_binding,
                ..
            } if first_binding == "lhs" && other_binding == "sum"
        );
    }

    #[test]
    fn test_dispatch_allows_same_name_for_distinct_input_and_output_tensors() {
        let descriptor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
        let input_tensor = MLTensor {
            id: 7,
            constant: false,
            descriptor: descriptor.clone(),
        };
        let output_tensor = MLTensor {
            id: 8,
            constant: false,
            descriptor,
        };
        let inputs = MLNamedTensors::from([("value", &input_tensor)]);
        let outputs = MLNamedTensors::from([("value", &output_tensor)]);

        validate_unique_tensor_bindings(&inputs, &outputs).unwrap();
    }

    #[cfg(feature = "trtx-runtime")]
    #[test]
    fn test_trtx_cuda_graph_replay() {
        let Some((mut context, mut graph)) = create_add_graph_context_and_graph() else {
            return;
        };
        if context.rustnn_backend() != Backend::Trtx {
            return;
        }

        let desc = rw_tensor_desc([2, 2].to_vec());
        let input_tensor = context.create_tensor(&desc).unwrap();
        let output_tensor = context.create_tensor(&desc).unwrap();
        let inputs = MLNamedTensors::from([("lhs", &input_tensor)]);
        let outputs = MLNamedTensors::from([("sum", &output_tensor)]);

        for upload in [
            vec![1.0f32, 2., 3., 4.],
            vec![5.0f32, 6., 7., 8.],
            vec![9.0f32, 10., 11., 12.],
        ] {
            context.write_tensor(&input_tensor, &upload).unwrap();
            context.dispatch(&mut graph, &inputs, &outputs).unwrap();

            let mut download = vec![0.0f32; 4];
            context.read_tensor(&output_tensor, &mut download).unwrap();
            assert_eq!(
                upload.iter().map(|value| value + 1.0).collect::<Vec<_>>(),
                download
            );
        }
    }

    #[test]
    fn test_dispatch_invalid_input_name_error_message() {
        let Some((mut context, mut graph)) = create_add_graph_context_and_graph() else {
            return;
        };

        let desc = rw_tensor_desc([2, 2].to_vec());
        let input_tensor = context.create_tensor(&desc).unwrap();
        let output_tensor = context.create_tensor(&desc).unwrap();
        let mut inputs = MLNamedTensors::new();
        inputs.insert("invalid_input", &input_tensor);
        let mut outputs = MLNamedTensors::new();
        outputs.insert("sum", &output_tensor);

        let err = context.dispatch(&mut graph, &inputs, &outputs).unwrap_err();
        std::assert_matches!(
            err,
            crate::error::Error::GraphDispatchError { source }
                if source.to_string() == "missing runtime input tensor `lhs`"
        );
    }

    #[test]
    fn test_dispatch_invalid_input_shape_error_message() {
        let Some((mut context, mut graph)) = create_add_graph_context_and_graph() else {
            return;
        };

        let desc = rw_tensor_desc([2, 3].to_vec());
        let input_tensor = context.create_tensor(&desc).unwrap();
        let output_tensor = context.create_tensor(&desc).unwrap();
        let mut inputs = MLNamedTensors::new();
        inputs.insert("lhs", &input_tensor);
        let mut outputs = MLNamedTensors::new();
        outputs.insert("sum", &output_tensor);

        let err = context.dispatch(&mut graph, &inputs, &outputs).unwrap_err();
        std::assert_matches!(
            err,
            crate::error::Error::GraphDispatchError { source }
                if source.to_string()
                    == "runtime input tensor `lhs` dimension 1 mismatch (expected 2, got 3)"
        );
    }

    #[test]
    fn test_dispatch_invalid_output_shape_error_message() {
        let Some((mut context, mut graph)) = create_add_graph_context_and_graph() else {
            return;
        };

        let in_desc = rw_tensor_desc([2, 2].to_vec());
        let out_desc = rw_tensor_desc([2, 3].to_vec());
        let input_tensor = context.create_tensor(&in_desc).unwrap();
        let output_tensor = context.create_tensor(&out_desc).unwrap();
        let mut inputs = MLNamedTensors::new();
        inputs.insert("lhs", &input_tensor);
        let mut outputs = MLNamedTensors::new();
        outputs.insert("sum", &output_tensor);

        let err = context.dispatch(&mut graph, &inputs, &outputs).unwrap_err();
        std::assert_matches!(
            err,
            crate::error::Error::GraphDispatchError { source }
                if source.to_string()
                    == "runtime output tensor `sum` dimension 1 mismatch (expected 2, got 3)"
        );
    }
}

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use log::debug;
use ndarray::{ArrayD, IxDyn};
use ort::environment::Environment;
use ort::memory::DeviceType;
use ort::session::SessionInputValue;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionOutputs};
use ort::value::{DynValue, Outlet, TensorElementType, Value, ValueType};

use crate::backend_selection::BackendDevice;
use crate::converters::{GraphConverter, OnnxConverter};
use crate::error::Error;
use crate::executors::onnx::ensure_ort_initialized;
use crate::mlcontext::{
    ListDevices, MLBackendBuilder, MLBackendContext, MLGraph, MLOperand, MLTensor,
    MLTensorDescriptor,
};
use crate::{GraphError, GraphInfo, ONNX_EXTERNAL_WEIGHTS_FILENAME};

trait ToDispatchResult<T> {
    fn to_dispatch_result(self) -> crate::error::Result<T>;
}

impl<T> ToDispatchResult<T> for ort::Result<T> {
    fn to_dispatch_result(self) -> crate::error::Result<T> {
        self.map_err(|e| Error::GraphDispatchError { source: e.into() })
    }
}

fn tensor_byte_len(descriptor: &MLTensorDescriptor) -> crate::error::Result<usize> {
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

fn ort_tensor_element_size(ty: TensorElementType) -> Option<usize> {
    match ty {
        TensorElementType::Float32 => Some(4),
        TensorElementType::Float16 => Some(2),
        TensorElementType::Int8 | TensorElementType::Uint8 | TensorElementType::Bool => Some(1),
        TensorElementType::Int16 | TensorElementType::Uint16 => Some(2),
        TensorElementType::Int32 | TensorElementType::Uint32 => Some(4),
        TensorElementType::Int64 | TensorElementType::Uint64 => Some(8),
        _ => None,
    }
}

/// Host tensor storage for the ONNX Runtime backend (mirrors device buffers in TRTX).
#[derive(Debug)]
pub(crate) struct OrtTensor {
    memory: Vec<u8>,
}

/// Compiled ONNX model held by [`MLGraph`] (mirrors [`crate::executors::trtx::TrtxGraph`]).
pub(crate) struct OrtGraph {
    pub(crate) session: Session,
}

impl fmt::Debug for OrtGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OrtGraph")
            .field("num_inputs", &self.session.inputs().len())
            .field("num_outputs", &self.session.outputs().len())
            .finish()
    }
}

pub(crate) struct OrtBuilder<'a> {
    graph: Option<&'a GraphInfo>,
}

impl fmt::Debug for OrtBuilder<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OrtBuilder")
            .field("has_graph", &self.graph.is_some())
            .finish()
    }
}

impl<'context> MLBackendBuilder<'context> for OrtBuilder<'context> {
    fn build(
        &mut self,
        _outputs: &HashMap<&str, MLOperand>,
    ) -> crate::error::Result<MLGraph<'context>> {
        let graph_info = self.graph.ok_or_else(|| Error::GraphBuildError {
            source: "build called before load_graph".into(),
        })?;
        let converted = OnnxConverter.convert(graph_info)?;

        let mut builder = Session::builder()
            .map_err(|e| GraphError::OnnxRuntimeFailed {
                reason: format!("session builder failed: {e}"),
            })?
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(|e| GraphError::OnnxRuntimeFailed {
                reason: format!("set opt level failed: {e}"),
            })?;
        if let Some(weights) = converted.weights_data {
            builder = builder
                .with_external_initializer_file_in_memory(
                    ONNX_EXTERNAL_WEIGHTS_FILENAME,
                    Cow::Owned(weights.to_vec()),
                )
                .map_err(|e| GraphError::OnnxRuntimeFailed {
                    reason: format!("set external initializer failed: {e}"),
                })?;
        }

        ensure_ort_initialized().map_err(|e| Error::GraphBuildError { source: e.into() })?;
        let session = builder
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(|e| Error::GraphBuildError { source: e.into() })?
            .commit_from_memory(&converted.data)
            .map_err(|e| Error::GraphBuildError { source: e.into() })?;
        Ok(MLGraph {
            backend: crate::mlcontext::MLBackendGraph::OnnxSession(
                OrtGraph { session },
                std::marker::PhantomData,
            ),
        })
    }

    fn load_graph(&mut self, graph: &'context GraphInfo) -> crate::error::Result<()> {
        self.graph = Some(graph);
        Ok(())
    }
}

fn session_input_from_host(
    input_info: &Outlet,
    shape: &[u64],
    bytes: &[u8],
) -> crate::error::Result<SessionInputValue<'static>> {
    let ValueType::Tensor { ty, .. } = input_info.dtype() else {
        return Err(Error::GraphDispatchError {
            source: format!("input '{}' is not a tensor", input_info.name()).into(),
        });
    };
    let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let Some(elem_sz) = ort_tensor_element_size(*ty) else {
        return Err(Error::GraphDispatchError {
            source: format!(
                "input '{}': unsupported ONNX element type for WebNN I/O: {:?}",
                input_info.name(),
                ty
            )
            .into(),
        });
    };
    let elements: usize = shape_usize.iter().product();

    // 0-element tensors (e.g. empty KV cache at pos=0) must be handled before the byte-length
    // check: create_tensor allocates max(1) byte but the check below would require elem_sz bytes.
    // Use ndarray which handles zero-sized dimensions correctly, same as executors/onnx.rs.
    if elements == 0 {
        macro_rules! empty_ndarray {
            ($rust_ty:ty) => {{
                let array: ArrayD<$rust_ty> = ArrayD::from_shape_vec(IxDyn(&shape_usize), vec![])
                    .map_err(|e| Error::GraphDispatchError {
                    source: format!("0-element input '{}': {e}", input_info.name()).into(),
                })?;
                Value::from_array(array)
                    .map_err(|e| Error::GraphDispatchError { source: e.into() })?
                    .into_dyn()
            }};
        }
        let dyn_val: DynValue = match ty {
            TensorElementType::Float32 => empty_ndarray!(f32),
            TensorElementType::Float16 => empty_ndarray!(half::f16),
            TensorElementType::Int8 => empty_ndarray!(i8),
            TensorElementType::Uint8 => empty_ndarray!(u8),
            TensorElementType::Int32 => empty_ndarray!(i32),
            TensorElementType::Uint32 => empty_ndarray!(u32),
            TensorElementType::Int64 => empty_ndarray!(i64),
            TensorElementType::Uint64 => empty_ndarray!(u64),
            _ => {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "input '{}': unsupported element type {:?} for 0-element tensor",
                        input_info.name(),
                        ty
                    )
                    .into(),
                });
            }
        };
        return Ok(SessionInputValue::from(dyn_val));
    }

    let expected = elements.saturating_mul(elem_sz);
    if bytes.len() != expected {
        return Err(Error::GraphDispatchError {
            source: format!(
                "input '{}': byte length mismatch (expected {}, got {})",
                input_info.name(),
                expected,
                bytes.len()
            )
            .into(),
        });
    }

    let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let dyn_val: DynValue = match ty {
        TensorElementType::Float32 => Value::from_array((
            shape_i64.as_slice(),
            bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
        ))
        .map_err(|e| Error::GraphDispatchError { source: e.into() })?
        .into_dyn(),
        TensorElementType::Float16 => {
            let u16s: Vec<u16> = bytemuck::cast_slice(bytes).to_vec();
            let f16_data: Vec<half::f16> = u16s.iter().map(|&b| half::f16::from_bits(b)).collect();
            Value::from_array((shape_i64.as_slice(), f16_data))
                .map_err(|e| Error::GraphDispatchError { source: e.into() })?
                .into_dyn()
        }
        TensorElementType::Int8 => Value::from_array((
            shape_i64.clone(),
            bytemuck::cast_slice::<u8, i8>(bytes).to_vec(),
        ))
        .map_err(|e| Error::GraphDispatchError { source: e.into() })?
        .into_dyn(),
        TensorElementType::Uint8 => Value::from_array((shape_i64.clone(), bytes.to_vec()))
            .map_err(|e| Error::GraphDispatchError { source: e.into() })?
            .into_dyn(),
        TensorElementType::Int32 => Value::from_array((
            shape_i64.clone(),
            bytemuck::cast_slice::<u8, i32>(bytes).to_vec(),
        ))
        .map_err(|e| Error::GraphDispatchError { source: e.into() })?
        .into_dyn(),
        TensorElementType::Uint32 => Value::from_array((
            shape_i64.clone(),
            bytemuck::cast_slice::<u8, u32>(bytes).to_vec(),
        ))
        .map_err(|e| Error::GraphDispatchError { source: e.into() })?
        .into_dyn(),
        TensorElementType::Int64 => Value::from_array((
            shape_i64.clone(),
            bytemuck::cast_slice::<u8, i64>(bytes).to_vec(),
        ))
        .map_err(|e| Error::GraphDispatchError { source: e.into() })?
        .into_dyn(),
        TensorElementType::Uint64 => Value::from_array((
            shape_i64.clone(),
            bytemuck::cast_slice::<u8, u64>(bytes).to_vec(),
        ))
        .map_err(|e| Error::GraphDispatchError { source: e.into() })?
        .into_dyn(),
        _ => {
            return Err(Error::GraphDispatchError {
                source: format!(
                    "input '{}': unsupported element type {:?}",
                    input_info.name(),
                    ty
                )
                .into(),
            });
        }
    };
    Ok(SessionInputValue::from(dyn_val))
}

fn copy_dyn_value_to_buffer(
    name: &str,
    value: &ort::value::DynValue,
    descriptor: &MLTensorDescriptor,
    buf: &mut [u8],
) -> crate::error::Result<()> {
    let expected = tensor_byte_len(descriptor)?;
    if buf.len() != expected {
        return Err(Error::GraphDispatchError {
            source: format!(
                "output '{name}': buffer length {} does not match descriptor ({expected} bytes)",
                buf.len()
            )
            .into(),
        });
    }
    match descriptor.data_type() {
        crate::operator_enums::MLOperandDataType::Float32 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            buf.copy_from_slice(bytemuck::cast_slice(sl));
        }
        crate::operator_enums::MLOperandDataType::Float16 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<half::f16>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            let u16s: Vec<u16> = sl.iter().map(|h| h.to_bits()).collect();
            buf.copy_from_slice(bytemuck::cast_slice(&u16s));
        }
        crate::operator_enums::MLOperandDataType::Int32 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<i32>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            buf.copy_from_slice(bytemuck::cast_slice(sl));
        }
        crate::operator_enums::MLOperandDataType::Uint32 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<u32>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            buf.copy_from_slice(bytemuck::cast_slice(sl));
        }
        crate::operator_enums::MLOperandDataType::Int64 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<i64>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            buf.copy_from_slice(bytemuck::cast_slice(sl));
        }
        crate::operator_enums::MLOperandDataType::Uint64 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<u64>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            buf.copy_from_slice(bytemuck::cast_slice(sl));
        }
        crate::operator_enums::MLOperandDataType::Int8 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<i8>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            buf.copy_from_slice(bytemuck::cast_slice(sl));
        }
        crate::operator_enums::MLOperandDataType::Uint8 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<u8>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            buf.copy_from_slice(sl);
        }
    }
    Ok(())
}

fn write_outputs_from_session<'s>(
    session_outputs: &SessionOutputs<'s>,
    outputs: &HashMap<&str, &MLTensor>,
    tensors: &mut [OrtTensor],
) -> crate::error::Result<()> {
    for (&name, ml_tensor) in outputs.iter() {
        let Some(value) = session_outputs.get(name) else {
            return Err(Error::GraphDispatchError {
                source: format!("model did not produce output '{name}'").into(),
            });
        };
        let buf = &mut tensors[ml_tensor.id].memory;
        copy_dyn_value_to_buffer(name, value, &ml_tensor.descriptor, buf)?;
    }
    Ok(())
}

pub(crate) struct OrtContext {
    env: Arc<Environment>,
    device_idx: usize,
    tensors: Vec<OrtTensor>,
}

impl OrtContext {
    pub(crate) fn new_from_ep_idx(device_idx: usize) -> crate::error::Result<Self> {
        ensure_ort_initialized().map_err(|e| Error::ContextCreationError { source: e.into() })?;
        let env =
            Environment::current().map_err(|e| Error::ContextCreationError { source: e.into() })?;
        Ok(Self {
            env,
            device_idx,
            tensors: Vec::new(),
        })
    }

    #[allow(dead_code)]
    pub(crate) fn new_from_ty(
        device_type: crate::backend_selection::DeviceType,
    ) -> crate::error::Result<Self> {
        ensure_ort_initialized().map_err(|e| Error::ContextCreationError { source: e.into() })?;
        let env =
            Environment::current().map_err(|e| Error::ContextCreationError { source: e.into() })?;

        let selected = env
            .devices()
            .inspect(|d| {
                debug!(
                    "Saw ONNX device {:?}",
                    &(&d.vendor(), &d.ep_vendor(), &d.id(), &d.ty(),)
                )
            })
            .position(|d| device_type == d.ty().into())
            .ok_or(Error::NoDeviceAvailable)?;

        Ok(Self {
            env,
            device_idx: selected,
            tensors: Vec::new(),
        })
    }
}

impl fmt::Debug for OrtContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let device = self.env.devices().nth(self.device_idx).unwrap();
        f.debug_struct("OrtContext")
            .field(
                "device",
                &(
                    &device.vendor(),
                    &device.ep_vendor(),
                    &device.id(),
                    &device.ty(),
                ),
            )
            .field("tensor_count", &self.tensors.len())
            .finish()
    }
}

impl ListDevices for OrtContext {
    fn list_devices() -> Vec<crate::backend_selection::BackendDevice> {
        if ensure_ort_initialized().is_err() {
            return vec![];
        }

        let Ok(env) =
            Environment::current().map_err(|e| Error::ContextCreationError { source: e.into() })
        else {
            return vec![];
        };

        let mut rtn = vec![];
        for (idx, d) in env.devices().enumerate() {
            debug!(
                "Saw ONNX device {:?}",
                &(&d.vendor(), &d.ep_vendor(), &d.id(), &d.ty(),)
            );
            rtn.push(BackendDevice::Onnx {
                device_type: d.ty().into(),
                ep_device_idx: idx,
            })
        }

        rtn
    }
}

impl<'context> MLBackendContext<'context> for OrtContext {
    fn accelerated(&self) -> bool {
        let device = self.env.devices().nth(self.device_idx).unwrap();
        device.ty() != DeviceType::CPU
    }

    fn create_builder(
        &mut self,
    ) -> crate::error::Result<Box<dyn MLBackendBuilder<'context> + 'context>> {
        Ok(Box::new(OrtBuilder { graph: None }))
    }

    fn create_tensor(&mut self, descriptor: &MLTensorDescriptor) -> crate::error::Result<MLTensor> {
        let n = tensor_byte_len(descriptor)?;
        let memory = vec![0u8; n.max(1)];
        self.tensors.push(OrtTensor { memory });
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
        if array.len() < host.len() {
            return Err(Error::TensorReadError {
                source: format!(
                    "buffer too small: need {} bytes, got {}",
                    host.len(),
                    array.len()
                )
                .into(),
                tensor: tensor.clone(),
            });
        }
        array[..host.len()].copy_from_slice(host);
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
        inputs: &HashMap<&str, &MLTensor>,
        outputs: &HashMap<&str, &MLTensor>,
    ) -> crate::error::Result<()> {
        let ort_graph =
            graph
                .backend
                .as_onnx_session_mut()
                .ok_or_else(|| Error::GraphDispatchError {
                    source: "MLGraph is not an ONNX Runtime session graph".into(),
                })?;

        let mut input_session_values: Vec<SessionInputValue> = Vec::new();
        for input_info in ort_graph.session.inputs().iter() {
            let name = input_info.name();
            let tensor = inputs.get(name).ok_or_else(|| Error::GraphDispatchError {
                source: format!("missing input '{name}' for ONNX dispatch").into(),
            })?;
            let bytes = &self.tensors[tensor.id].memory;
            let v = session_input_from_host(input_info, tensor.shape(), bytes)?;
            input_session_values.push(v);
        }

        let session_outputs = ort_graph
            .session
            .run(input_session_values.as_slice())
            .to_dispatch_result()?;

        write_outputs_from_session(&session_outputs, outputs, &mut self.tensors)?;
        Ok(())
    }
}

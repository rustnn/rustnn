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
use crate::graph::{pack_int4, pack_uint4_from_i32, unpack_int4, unpack_uint4};
use crate::mlcontext::{
    ListDevices, MLBackendBuilder, MLBackendContext, MLGraph, MLTensor, MLTensorDescriptor,
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
    let elements: usize = descriptor
        .shape()
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| Error::GraphDispatchError {
            source: "tensor element count overflow".into(),
        })? as usize;
    Ok(descriptor.data_type().rustnn_storage_byte_length(elements))
}

fn ort_tensor_element_size(ty: TensorElementType) -> Option<usize> {
    match ty {
        TensorElementType::Int64 | TensorElementType::Uint64 => Some(8),
        TensorElementType::Float32 | TensorElementType::Int32 | TensorElementType::Uint32 => {
            Some(4)
        }
        TensorElementType::Float16 | TensorElementType::Int16 | TensorElementType::Uint16 => {
            Some(2)
        }
        TensorElementType::Int8 | TensorElementType::Uint8 | TensorElementType::Bool => Some(1),
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

impl<'context, 'builder> MLBackendBuilder<'context, 'builder> for OrtBuilder<'context> {
    fn build(&mut self, graph_info: GraphInfo) -> crate::error::Result<MLGraph<'context>> {
        let converted = OnnxConverter.convert(&graph_info)?;

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
            .map_err(|e| Error::GraphBuildError {
                source: format!("{e}").into(),
            })?
            .commit_from_memory(&converted.data)
            .map_err(|e| Error::GraphBuildError { source: e.into() })?;
        MLGraph::new(
            crate::mlcontext::MLBackendGraph::OnnxSession(
                OrtGraph { session },
                std::marker::PhantomData,
            ),
            &graph_info,
        )
    }
}

fn session_input_from_host(
    input_info: &Outlet,
    descriptor: &MLTensorDescriptor,
    bytes: &[u8],
) -> crate::error::Result<SessionInputValue<'static>> {
    let ValueType::Tensor { ty, .. } = input_info.dtype() else {
        return Err(Error::GraphDispatchError {
            source: format!("input '{}' is not a tensor", input_info.name()).into(),
        });
    };
    let shape = descriptor.shape();
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

    let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let expected = elements.saturating_mul(elem_sz);
    let host_packed = descriptor.data_type().rustnn_storage_byte_length(elements);
    if bytes.len() != expected {
        if *ty == TensorElementType::Int32
            && descriptor.data_type() == crate::operator_enums::MLOperandDataType::Int4
            && bytes.len() == host_packed
        {
            let int32_data = unpack_int4(bytes, elements);
            let dyn_val = Value::from_array((shape_i64.as_slice(), int32_data))
                .map_err(|e| Error::GraphDispatchError { source: e.into() })?
                .into_dyn();
            return Ok(SessionInputValue::from(dyn_val));
        }
        if *ty == TensorElementType::Uint8
            && descriptor.data_type() == crate::operator_enums::MLOperandDataType::Uint4
            && bytes.len() == host_packed
        {
            let uint8_data = unpack_uint4(bytes, elements);
            let dyn_val = Value::from_array((shape_i64.as_slice(), uint8_data))
                .map_err(|e| Error::GraphDispatchError { source: e.into() })?
                .into_dyn();
            return Ok(SessionInputValue::from(dyn_val));
        }
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
    if buf.len() < expected {
        return Err(Error::GraphDispatchError {
            source: format!(
                "output '{name}': buffer too small for descriptor shape {:?}: need {expected} bytes, storage {}",
                descriptor.shape(),
                buf.len()
            )
            .into(),
        });
    }
    if buf.len() != expected {
        debug!(
            target: "rustnn::backends::ort",
            "output '{name}': copy {expected} logical bytes into storage {} bytes (oversized buffer / rustnn_set_tensor_capacity)",
            buf.len()
        );
    }
    let dst = &mut buf[..expected];
    match descriptor.data_type() {
        crate::operator_enums::MLOperandDataType::Float32 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            let src = bytemuck::cast_slice(sl);
            if src.len() != expected {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{name}': ORT tensor {} bytes, descriptor expects {}",
                        src.len(),
                        expected
                    )
                    .into(),
                });
            }
            dst.copy_from_slice(src);
        }
        crate::operator_enums::MLOperandDataType::Float16 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<half::f16>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            let u16s: Vec<u16> = sl.iter().map(|h| h.to_bits()).collect();
            let src = bytemuck::cast_slice(&u16s);
            if src.len() != expected {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{name}': ORT tensor {} bytes, descriptor expects {}",
                        src.len(),
                        expected
                    )
                    .into(),
                });
            }
            dst.copy_from_slice(src);
        }
        crate::operator_enums::MLOperandDataType::Int32 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<i32>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            let src = bytemuck::cast_slice(sl);
            if src.len() != expected {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{name}': ORT tensor {} bytes, descriptor expects {}",
                        src.len(),
                        expected
                    )
                    .into(),
                });
            }
            dst.copy_from_slice(src);
        }
        crate::operator_enums::MLOperandDataType::Uint32 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<u32>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            let src = bytemuck::cast_slice(sl);
            if src.len() != expected {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{name}': ORT tensor {} bytes, descriptor expects {}",
                        src.len(),
                        expected
                    )
                    .into(),
                });
            }
            dst.copy_from_slice(src);
        }
        crate::operator_enums::MLOperandDataType::Int64 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<i64>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            let src = bytemuck::cast_slice(sl);
            if src.len() != expected {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{name}': ORT tensor {} bytes, descriptor expects {}",
                        src.len(),
                        expected
                    )
                    .into(),
                });
            }
            dst.copy_from_slice(src);
        }
        crate::operator_enums::MLOperandDataType::Uint64 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<u64>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            let src = bytemuck::cast_slice(sl);
            if src.len() != expected {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{name}': ORT tensor {} bytes, descriptor expects {}",
                        src.len(),
                        expected
                    )
                    .into(),
                });
            }
            dst.copy_from_slice(src);
        }
        crate::operator_enums::MLOperandDataType::Int8 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<i8>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            let src = bytemuck::cast_slice(sl);
            if src.len() != expected {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{name}': ORT tensor {} bytes, descriptor expects {}",
                        src.len(),
                        expected
                    )
                    .into(),
                });
            }
            dst.copy_from_slice(src);
        }
        crate::operator_enums::MLOperandDataType::Uint8 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<u8>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            if sl.len() != expected {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{name}': ORT tensor {} bytes, descriptor expects {}",
                        sl.len(),
                        expected
                    )
                    .into(),
                });
            }
            dst.copy_from_slice(sl);
        }
        crate::operator_enums::MLOperandDataType::Int4 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<i32>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            if sl.len() != descriptor.shape().iter().product::<u64>() as usize {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{name}': ORT tensor {} elements, descriptor expects {}",
                        sl.len(),
                        descriptor.shape().iter().product::<u64>()
                    )
                    .into(),
                });
            }
            let packed = pack_int4(sl);
            if packed.len() != expected {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{name}': packed int4 {} bytes, descriptor expects {}",
                        packed.len(),
                        expected
                    )
                    .into(),
                });
            }
            dst.copy_from_slice(&packed);
        }
        crate::operator_enums::MLOperandDataType::Uint4 => {
            let (_, sl) =
                value
                    .try_extract_tensor::<i32>()
                    .map_err(|e| Error::GraphDispatchError {
                        source: format!("output '{name}': {e}").into(),
                    })?;
            if sl.len() != descriptor.shape().iter().product::<u64>() as usize {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{name}': ORT tensor {} elements, descriptor expects {}",
                        sl.len(),
                        descriptor.shape().iter().product::<u64>()
                    )
                    .into(),
                });
            }
            let packed = pack_uint4_from_i32(sl);
            if packed.len() != expected {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{name}': packed uint4 {} bytes, descriptor expects {}",
                        packed.len(),
                        expected
                    )
                    .into(),
                });
            }
            dst.copy_from_slice(&packed);
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
        let logical = tensor_byte_len(&ml_tensor.descriptor)?;
        debug!(
            target: "rustnn::backends::ort",
            "write_output '{}' tensor_id={} shape={:?} logical_bytes={} storage_bytes={}",
            name,
            ml_tensor.id,
            ml_tensor.shape(),
            logical,
            buf.len()
        );
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
                    (&d.vendor(), &d.ep_vendor(), &d.id(), &d.ty(),)
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
                (&d.vendor(), &d.ep_vendor(), &d.id(), &d.ty(),)
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

    fn create_builder<'builder>(
        &mut self,
    ) -> crate::error::Result<Box<dyn MLBackendBuilder<'context, 'builder> + 'builder>>
    where
        'context: 'builder,
    {
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
        let logical = tensor_byte_len(tensor.descriptor())?;
        if host.len() > logical {
            debug!(
                target: "rustnn::backends::ort",
                "read_tensor tensor_id={} shape={:?} logical_bytes={} storage_bytes={}",
                tensor.id,
                tensor.shape(),
                logical,
                host.len()
            );
        }
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
            let full = &self.tensors[tensor.id].memory;
            let logical = tensor_byte_len(tensor.descriptor())?;
            let bytes = full
                .get(..logical)
                .ok_or_else(|| Error::GraphDispatchError {
                    source: format!(
                        "input '{name}': tensor buffer shorter than logical size ({logical} bytes)"
                    )
                    .into(),
                })?;
            debug!(
                target: "rustnn::backends::ort",
                "dispatch input '{}' tensor_id={} shape={:?} logical_bytes={} storage_bytes={}",
                name,
                tensor.id,
                tensor.shape(),
                logical,
                full.len()
            );
            let v = session_input_from_host(input_info, tensor.descriptor(), bytes)?;
            input_session_values.push(v);
        }

        debug!(
            target: "rustnn::backends::ort",
            "ORT session.run: {} inputs, {} output bindings",
            input_session_values.len(),
            outputs.len()
        );

        let session_outputs = ort_graph
            .session
            .run(input_session_values.as_slice())
            .to_dispatch_result()?;

        write_outputs_from_session(&session_outputs, outputs, &mut self.tensors)?;
        Ok(())
    }

    fn rustnn_resize_tensor(
        &mut self,
        tensor: &mut MLTensor,
        new_shape: &[u64],
    ) -> crate::error::Result<()> {
        let id = tensor.id;
        let old_shape = tensor.descriptor().shape().to_vec();
        let old_cap = self.tensors[tensor.id].memory.len();

        let mut new_desc = tensor.descriptor().clone();
        new_desc.set_shape(new_shape.to_vec());

        let new_bytes = tensor_byte_len(&new_desc)?;
        let host = &mut self.tensors[tensor.id].memory;
        let grew = new_bytes > host.len();
        if grew {
            host.resize(new_bytes, 0u8);
        }
        tensor.descriptor = new_desc;

        debug!(
            target: "rustnn::backends::ort",
            "rustnn_resize_tensor tensor_id={} old_shape={:?} new_shape={:?} old_cap={} new_cap={} logical_bytes={} grew={}",
            id,
            old_shape,
            new_shape,
            old_cap,
            host.len(),
            new_bytes,
            grew
        );
        Ok(())
    }

    fn rustnn_set_tensor_capacity(
        &mut self,
        tensor: &mut MLTensor,
        max_shape: &[u64],
    ) -> crate::error::Result<()> {
        let new_bytes = tensor.data_type().rustnn_storage_byte_length(
            max_shape
                .iter()
                .try_fold(1usize, |acc, &d| {
                    usize::try_from(d).ok().and_then(|x| acc.checked_mul(x))
                })
                .ok_or_else(|| Error::GraphDispatchError {
                    source: "rustnn_set_tensor_capacity: shape element count overflow".into(),
                })?,
        );
        let alloc = new_bytes.max(1);
        self.tensors[tensor.id].memory = vec![0u8; alloc];
        debug!(
            target: "rustnn::backends::ort",
            "rustnn_set_tensor_capacity tensor_id={} max_shape={:?} alloc_bytes={} (logical descriptor unchanged: {:?})",
            tensor.id,
            max_shape,
            alloc,
            tensor.descriptor().shape()
        );
        Ok(())
    }
}

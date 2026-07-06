use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;
use std::sync::OnceLock;

use litert_sys as sys;

use crate::backend_selection::DeviceType;
use crate::converters::{GraphConverter, LiteRtConverter};
use crate::error::{Error, Result};
use crate::mlcontext::{
    ListDevices, MLBackendBuilder, MLBackendContext, MLGraph, MLTensor, MLTensorDescriptor,
};

use crate::operator_enums::MLOperandDataType;
use crate::{GraphError, GraphInfo};

/// WebNN operations not (yet) supported by the LiteRT backend (crash, wrong results, or layout-dependent).
const LITERT_UNSUPPORTED_OPS: &[&str] = &[
    "averagePool2d",
    "conv2d",
    "equal",
    "gemm",
    "greater",
    "greater_or_equal",
    "identity",
    "lesser",
    "lesser_or_equal",
    "logical_and",
    "logical_or",
    "matmul",
    "maxPool2d",
    "not_equal",
    "pow",
    "slice",
    "split",
    "sub",
];

/// Returns the list of WebNN operations not supported by this backend.
pub fn unsupported_ops() -> &'static [&'static str] {
    LITERT_UNSUPPORTED_OPS
}

struct LiteRt;

impl LiteRt {
    fn env() -> sys::LiteRtEnvironment {
        static ENV: OnceLock<usize> = OnceLock::new();
        *ENV.get_or_init(|| {
            let mut env = std::ptr::null_mut();
            check(unsafe { sys::LiteRtCreateEnvironment(0, std::ptr::null(), &mut env) })
                .expect("LiteRtCreateEnvironment failed");
            env as usize
        }) as *mut _
    }
}

fn check(status: sys::LiteRtStatus) -> Result<()> {
    if status == sys::kLiteRtStatusOk {
        Ok(())
    } else {
        Err(Error::GraphDispatchError {
            source: format!("LiteRT status error: code={}", status).into(),
        })
    }
}

fn ml_operand_to_litert_element_type(dt: MLOperandDataType) -> Result<litert::ElementType> {
    use litert::ElementType;
    Ok(match dt {
        MLOperandDataType::Float32 => ElementType::Float32,
        MLOperandDataType::Float16 => ElementType::Float16,
        MLOperandDataType::Int32 => ElementType::Int32,
        MLOperandDataType::Uint32 => ElementType::UInt32,
        MLOperandDataType::Int64 => ElementType::Int64,
        MLOperandDataType::Uint64 => ElementType::UInt64,
        MLOperandDataType::Int8 => ElementType::Int8,
        MLOperandDataType::Uint8 => ElementType::UInt8,
        MLOperandDataType::Int4 => ElementType::Int4,
        _ => {
            return Err(Error::GraphBuildError {
                source: format!("unsupported ML data type for litert: {:?}", dt).into(),
            });
        }
    })
}

pub(crate) struct LiteRtGraph {
    compiled: NonNull<sys::LiteRtCompiledModelT>,
    model: NonNull<sys::LiteRtModelT>,
    _model_bytes: Box<[u8]>,
}

unsafe impl Send for LiteRtGraph {}
unsafe impl Sync for LiteRtGraph {}

impl LiteRtGraph {
    fn new(model_bytes: Vec<u8>, accelerator_bits: sys::LiteRtHwAcceleratorSet) -> Result<Self> {
        let owned = model_bytes.into_boxed_slice();
        unsafe {
            let mut model = std::ptr::null_mut();
            check(sys::LiteRtCreateModelFromBuffer(
                owned.as_ptr() as *const c_void,
                owned.len(),
                &mut model,
            ))?;
            let model = NonNull::new(model).ok_or_else(|| Error::GraphBuildError {
                source: "LiteRT: null model handle".into(),
            })?;

            let mut options = std::ptr::null_mut();
            check(sys::LiteRtCreateOptions(&mut options))?;
            check(sys::LiteRtSetOptionsHardwareAccelerators(
                options,
                accelerator_bits,
            ))?;

            let mut compiled = std::ptr::null_mut();
            let status = sys::LiteRtCreateCompiledModel(
                LiteRt::env(),
                model.as_ptr(),
                options,
                &mut compiled,
            );
            sys::LiteRtDestroyOptions(options);
            check(status)?;
            let compiled = NonNull::new(compiled).ok_or_else(|| Error::GraphBuildError {
                source: "LiteRT: null compiled model handle".into(),
            })?;

            Ok(Self {
                compiled,
                model,
                _model_bytes: owned,
            })
        }
    }

    fn run(
        &self,
        in_raw: &[sys::LiteRtTensorBuffer],
        out_raw: &mut [sys::LiteRtTensorBuffer],
    ) -> Result<()> {
        check(unsafe {
            sys::LiteRtRunCompiledModel(
                self.compiled.as_ptr(),
                0,
                in_raw.len(),
                in_raw.as_ptr() as *mut sys::LiteRtTensorBuffer,
                out_raw.len(),
                out_raw.as_mut_ptr(),
            )
        })
    }
}

impl fmt::Debug for LiteRtGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiteRtGraph").finish()
    }
}

impl Drop for LiteRtGraph {
    fn drop(&mut self) {
        unsafe {
            sys::LiteRtDestroyCompiledModel(self.compiled.as_ptr());
            sys::LiteRtDestroyModel(self.model.as_ptr());
        }
    }
}

// Host tensor storage for LiteRT backend.
pub(crate) struct LiteRtTensor {
    handle: sys::LiteRtTensorBuffer,
}

impl fmt::Debug for LiteRtTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiteRtTensor").finish()
    }
}

unsafe impl Send for LiteRtTensor {}
unsafe impl Sync for LiteRtTensor {}

impl LiteRtTensor {
    fn new(descriptor: &MLTensorDescriptor) -> Result<Self> {
        let element_type = ml_operand_to_litert_element_type(descriptor.data_type())?;
        let shape = litert::TensorShape {
            element_type,
            dims: descriptor.shape().iter().map(|&d| d as i32).collect(),
        };
        let element_size = match shape.element_type {
            litert::ElementType::Float32 => 4,
            litert::ElementType::Float16 => 2,
            litert::ElementType::Int32 => 4,
            litert::ElementType::UInt32 => 4,
            litert::ElementType::Int64 => 8,
            litert::ElementType::UInt64 => 8,
            litert::ElementType::Int16 => 2,
            litert::ElementType::UInt16 => 2,
            litert::ElementType::Int8 => 1,
            litert::ElementType::UInt8 => 1,
            litert::ElementType::Bool => 1,
            _ => {
                return Err(Error::GraphBuildError {
                    source: format!("unsupported element type: {:?}", shape.element_type).into(),
                });
            }
        };
        let size_bytes = shape.num_elements() * element_size;

        let mut layout = sys::LiteRtLayout::default();
        layout.set_rank(u32::try_from(shape.dims.len()).expect("rank fits in u32"));
        layout.set_has_strides(false);
        for (slot, &d) in layout.dimensions.iter_mut().zip(shape.dims.iter()) {
            *slot = d;
        }
        let tensor_type = sys::LiteRtRankedTensorType {
            element_type: shape.element_type as sys::LiteRtElementType,
            layout,
        };

        let mut handle = std::ptr::null_mut();
        check(unsafe {
            sys::LiteRtCreateManagedTensorBuffer(
                LiteRt::env(),
                sys::kLiteRtTensorBufferTypeHostMemory,
                &tensor_type,
                size_bytes,
                &mut handle,
            )
        })?;
        Ok(Self { handle })
    }

    fn lock(&self, mode: sys::LiteRtTensorBufferLockMode) -> Result<*mut u8> {
        let mut addr: *mut c_void = std::ptr::null_mut();
        check(unsafe { sys::LiteRtLockTensorBuffer(self.handle, &mut addr, mode) })?;
        Ok(addr as *mut u8)
    }

    fn unlock(&self) -> Result<()> {
        check(unsafe { sys::LiteRtUnlockTensorBuffer(self.handle) })
    }

    fn write(&self, data: &[u8]) -> Result<()> {
        let ptr = self.lock(sys::kLiteRtTensorBufferLockModeWrite)?;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        self.unlock()
    }

    fn read(&self, buf: &mut [u8]) -> Result<()> {
        let ptr = self.lock(sys::kLiteRtTensorBufferLockModeRead)?;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), buf.len());
        }
        self.unlock()
    }
}

impl Drop for LiteRtTensor {
    fn drop(&mut self) {
        unsafe { sys::LiteRtDestroyTensorBuffer(self.handle) };
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
        let _ = litert::set_global_log_severity(litert::LogSeverity::Warning);
        LiteRt::env();
        Ok(Self {
            tensors: Vec::new(),
            device_type,
        })
    }

    fn accelerator_bits(&self) -> sys::LiteRtHwAcceleratorSet {
        match self.device_type {
            DeviceType::Cpu => sys::kLiteRtHwAcceleratorCpu as _,
            DeviceType::Gpu => (sys::kLiteRtHwAcceleratorGpu | sys::kLiteRtHwAcceleratorCpu) as _,
            DeviceType::Npu => (sys::kLiteRtHwAcceleratorNpu | sys::kLiteRtHwAcceleratorCpu) as _,
        }
    }
}

// Impls for Backend Trait
impl ListDevices for LiteRtContext {
    fn list_devices() -> Vec<crate::backend_selection::BackendDevice> {
        if LiteRt::env().is_null() {
            return vec![];
        }
        vec![
            crate::backend_selection::BackendDevice::LiteRt {
                device_type: DeviceType::Cpu,
            },
            crate::backend_selection::BackendDevice::LiteRt {
                device_type: DeviceType::Gpu,
            },
            crate::backend_selection::BackendDevice::LiteRt {
                device_type: DeviceType::Npu,
            },
        ]
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
        Ok(Box::new(LiteRtBuilder {
            accelerator_bits: self.accelerator_bits(),
        }))
    }

    fn create_tensor(&mut self, descriptor: &MLTensorDescriptor) -> Result<MLTensor> {
        let tensor = LiteRtTensor::new(descriptor)?;
        self.tensors.push(tensor);
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
        let logical = tensor.descriptor().rustnn_required_bytes();
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
        self.tensors[tensor.id].read(&mut array[..logical])?;
        Ok(())
    }

    fn write_tensor(&mut self, tensor: &MLTensor, array: &[u8]) -> Result<()> {
        let logical = tensor.descriptor().rustnn_required_bytes();
        if array.len() < logical {
            return Err(Error::TensorWriteError {
                source: format!(
                    "write too small for tensor: {} bytes < {} logical bytes",
                    array.len(),
                    logical,
                )
                .into(),
                tensor: tensor.clone(),
            });
        }
        self.tensors[tensor.id].write(&array[..logical])?;
        Ok(())
    }

    fn dispatch(
        &mut self,
        graph: &mut MLGraph,
        inputs: &HashMap<&str, &MLTensor>,
        outputs: &HashMap<&str, &MLTensor>,
    ) -> Result<()> {
        let lite_graph = match &graph.backend {
            crate::mlcontext::MLBackendGraph::LiteRtGraph(graph) => graph,
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "litert".to_string(),
                    reason: "expected LiteRtGraph in dispatch".to_string(),
                }
                .into());
            }
        };

        let mut sorted_inputs: Vec<(&str, &MLTensor)> =
            inputs.iter().map(|(k, v)| (*k, *v)).collect();
        sorted_inputs.sort_by_key(|(name, _)| *name);

        let mut sorted_outputs: Vec<(&str, &MLTensor)> =
            outputs.iter().map(|(k, v)| (*k, *v)).collect();
        sorted_outputs.sort_by_key(|(name, _)| *name);

        let in_raw: Vec<sys::LiteRtTensorBuffer> = sorted_inputs
            .iter()
            .map(|(_, t)| self.tensors[t.id].handle)
            .collect();
        let mut out_raw: Vec<sys::LiteRtTensorBuffer> = sorted_outputs
            .iter()
            .map(|(_, t)| self.tensors[t.id].handle)
            .collect();

        lite_graph.run(&in_raw, &mut out_raw)?;

        Ok(())
    }
}

pub(crate) struct LiteRtBuilder {
    accelerator_bits: sys::LiteRtHwAcceleratorSet,
}

impl fmt::Debug for LiteRtBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiteRtBuilder").finish()
    }
}

impl<'context, 'builder> MLBackendBuilder<'context, 'builder> for LiteRtBuilder {
    fn build(&mut self, graph_info: GraphInfo) -> Result<MLGraph<'context>> {
        let converted = LiteRtConverter.convert(&graph_info)?;
        let tflite_bytes = converted.data;

        let graph = LiteRtGraph::new(tflite_bytes, self.accelerator_bits).map_err(|e| {
            Error::GraphBuildError {
                source: format!("failed to compile model: {e}").into(),
            }
        })?;

        MLGraph::new(
            crate::mlcontext::MLBackendGraph::LiteRtGraph(graph),
            &graph_info,
        )
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

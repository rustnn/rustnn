use std::collections::HashMap;
use std::ffi::c_void;

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::{CudaContext, DriverError, result, sys};
use cudarc::driver::{CudaEvent, DevicePtrMut};
use log::debug;
use log::info;
use log::trace;
use log::warn;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use trtx::CudaEngine;
use trtx::ExecutionContext;
use trtx::Tensor;

use crate::GraphInfo;
use crate::converters::TrtxConverter;
use crate::error::Error;

use crate::mlcontext::MLTensor;
use crate::mlcontext::{ListDevices, MLOperand};
use crate::mlcontext::{MLBackendBuilder, MLGraph};
use crate::mlcontext::{MLBackendContext, MLBackendGraph};

#[derive(Debug, thiserror::Error)]
pub enum TrtxError {
    #[error("Cuda driver error: {source}")]
    CudaError {
        #[from]
        source: DriverError,
    },
    #[error("TensorRT error: {source}")]
    TrtxError {
        #[from]
        source: trtx::Error,
    },
}
pub type TrtxResult<T> = std::result::Result<T, TrtxError>;

// TODO: the mapping to GraphDispatchError/TensorReadError/TensorWriteError
// should be done for all backends in mlcontext.rs
trait ToDispatchResult<T> {
    fn to_dispatch_result(self) -> crate::error::Result<T>;
}

impl<T> ToDispatchResult<T> for trtx::Result<T> {
    fn to_dispatch_result(self) -> crate::error::Result<T> {
        self.map_err(|e| crate::error::Error::GraphDispatchError { source: e.into() })
    }
}
impl<T> ToDispatchResult<T> for std::result::Result<T, DriverError> {
    fn to_dispatch_result(self) -> crate::error::Result<T> {
        self.map_err(|e| crate::error::Error::GraphDispatchError { source: e.into() })
    }
}

trait ToReadTensorResult<T> {
    fn to_read_tensor_result(
        self,
        get_tensor: impl FnOnce() -> MLTensor,
    ) -> crate::error::Result<T>;
}

impl<T> ToReadTensorResult<T> for std::result::Result<T, DriverError> {
    fn to_read_tensor_result(
        self,
        get_tensor: impl FnOnce() -> MLTensor,
    ) -> crate::error::Result<T> {
        self.map_err(|e| crate::error::Error::TensorReadError {
            source: Box::new(e),
            tensor: get_tensor(),
        })
    }
}

trait ToWriteTensorResult<T> {
    fn to_write_tensor_result(
        self,
        get_tensor: impl FnOnce() -> MLTensor,
    ) -> crate::error::Result<T>;
}

impl<T> ToWriteTensorResult<T> for std::result::Result<T, DriverError> {
    fn to_write_tensor_result(
        self,
        get_tensor: impl FnOnce() -> MLTensor,
    ) -> crate::error::Result<T> {
        self.map_err(|e| crate::error::Error::TensorWriteError {
            source: Box::new(e),
            tensor: get_tensor(),
        })
    }
}

pub(crate) struct TrtxGraph<'context> {
    exec: ExecutionContext<'context>,
    _engine: CudaEngine<'context>,
    cuda_stream: Arc<CudaStream>,
}

impl std::fmt::Debug for TrtxGraph<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrtxGraph")
            .field("engine", &"")
            .field("exec", &self.exec.name())
            .field("cuda_stream", &self.cuda_stream)
            .finish()
    }
}

#[derive(Debug)]
pub(crate) struct TrtxTensor {
    // todo: make all allocs owned by backend, only have view here
    memory: CudaSlice<u8>,
    stream: Arc<CudaStream>,
}

impl TrtxTensor {
    fn new(cuda_ctx: &Arc<CudaContext>, size: usize) -> TrtxResult<Self> {
        debug!("Allocating CUDA tensor of size {size}");
        let stream = cuda_ctx.new_stream()?;
        let memory = stream.alloc_zeros(size)?;
        Ok(Self { memory, stream })
    }
}

pub(crate) struct TrtxContext<'context> {
    cuda_ctx: Arc<CudaContext>,
    tensors: Vec<TrtxTensor>,
    events: Vec<CudaEvent>,
    runtime: Rc<Mutex<trtx::Runtime<'context>>>,
    config: Rc<Mutex<trtx::BuilderConfig<'context>>>, // needs to be destroyed before builder
    builder: Rc<Mutex<trtx::Builder<'context>>>,
}

impl std::fmt::Debug for TrtxContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrtxContext")
            .field("cuda_ctx", &self.cuda_ctx)
            .field("tensors", &self.tensors)
            //.field("logger", &self.logger)
            //.field("runtime", &self.runtime)
            .finish()
    }
}

// TODO: should make logger static or remove from API. It is anyway a global for TRT
static LOGGER: std::sync::LazyLock<trtx::Logger> =
    std::sync::LazyLock::new(|| trtx::Logger::log_crate().unwrap());

impl<'context> TrtxContext<'context> {
    pub(crate) fn new(cuda_device_idx: u32) -> TrtxResult<Self> {
        // this retains the primary context
        let cuda_ctx = CudaContext::new(cuda_device_idx as usize)?;
        let mut builder = trtx::Builder::new(&LOGGER)?;
        let config = Rc::new(builder.create_config()?.into());
        let runtime = Rc::new(trtx::Runtime::new(&LOGGER)?.into());
        debug!("Created new TrtxContext");
        Ok(Self {
            cuda_ctx,
            tensors: vec![],
            events: vec![],
            runtime,
            builder: Rc::new(builder.into()),
            config,
        })
    }
}

#[allow(dead_code)]
pub(crate) struct TrtxBuilder<'builder> {
    network: trtx::NetworkDefinition<'builder>,
    builder: Rc<Mutex<trtx::Builder<'builder>>>,
    config: Rc<Mutex<trtx::BuilderConfig<'builder>>>,
    cuda_context: Arc<CudaContext>,
    runtime: Rc<Mutex<trtx::Runtime<'builder>>>,
    operands: HashMap<String, MLOperand>,
    tensors: Vec<Tensor<'builder>>,
    strings: Vec<String>, //_parser: Option<OnnxParser<'builder>>,
}
impl std::fmt::Debug for TrtxBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrtxBuilder").finish()
    }
}

impl<'context> MLBackendBuilder<'context> for TrtxBuilder<'context> {
    /*async */
    fn build(
        &mut self,
        outputs: &HashMap<&str, MLOperand>,
    ) -> crate::error::Result<crate::mlcontext::MLGraph<'context>> {
        let num_outputs = self.network.nb_outputs();
        // Outputs already not set already via load_graph API
        if num_outputs == 0 {
            for (k, v) in outputs {
                let tensor = self.tensors[v.id];
                tensor.set_name(&mut self.network, k)?;
                self.network.mark_output(&tensor);
            }
        }

        let host_mem = self
            .builder
            .lock()
            .unwrap()
            .build_serialized_network(&mut self.network, &mut self.config.lock().unwrap())
            .map_err(|e| crate::error::Error::GraphBuildError { source: e.into() })?;

        self.cuda_context.bind_to_thread()?;
        let mut engine = self
            .runtime
            .lock()
            .unwrap()
            .deserialize_cuda_engine(&host_mem)
            .map_err(|e| crate::error::Error::GraphBuildError { source: e.into() })?;

        let exec = engine
            .create_execution_context()
            .map_err(|e| crate::error::Error::GraphBuildError { source: e.into() })?;

        Ok(MLGraph {
            backend: MLBackendGraph::TrtxEngine(TrtxGraph {
                _engine: engine,
                exec,
                cuda_stream: self
                    .cuda_context
                    .new_stream()
                    .map_err(|e| crate::error::Error::GraphBuildError { source: e.into() })?,
            }),
        })
    }

    fn load_graph(&mut self, graph: &'context GraphInfo) -> crate::error::Result<()> {
        Ok(TrtxConverter::build_network(graph, &mut self.network)?)
    }
}

#[allow(unused_variables)]
impl<'context> MLBackendContext<'context> for TrtxContext<'context> {
    fn accelerated(&self) -> bool {
        true
    }

    fn create_builder(
        &'_ mut self,
    ) -> crate::error::Result<Box<dyn crate::mlcontext::MLBackendBuilder<'context> + 'context>>
    {
        let network = self
            .builder
            .lock()
            .unwrap()
            .create_network(0)
            .map_err(|e| Error::BuilderCreationError {
                source: Box::new(e),
            })?;
        //self.networks.push(network);
        Ok(Box::new(TrtxBuilder {
            network,
            builder: Rc::clone(&self.builder),
            config: Rc::clone(&self.config),
            runtime: Rc::clone(&self.runtime),
            cuda_context: Arc::clone(&self.cuda_ctx),
            operands: HashMap::new(),
            tensors: vec![],
            strings: vec![],
        }))
    }

    fn create_tensor(
        &mut self,
        descriptor: &crate::mlcontext::MLTensorDescriptor,
    ) -> crate::error::Result<crate::mlcontext::MLTensor> {
        let size = descriptor.rustnn_required_bytes();

        let tensor = TrtxTensor::new(&self.cuda_ctx, size).map_err(|e| {
            crate::error::Error::TensorCreationError {
                source: e.into(),
                descriptor: descriptor.clone(),
            }
        })?;
        self.tensors.push(tensor);
        Ok(MLTensor {
            id: self.tensors.len() - 1,
            constant: false,
            descriptor: descriptor.clone(),
        })
    }

    fn create_constant_tensor(
        &mut self,
        descriptor: &crate::mlcontext::MLTensorDescriptor,
        input_data: &[u8],
    ) -> crate::error::Result<crate::mlcontext::MLTensor> {
        let mut tensor = self.create_tensor(descriptor)?;
        tensor.constant = true;
        self.write_tensor(&tensor, input_data).map_err(|e| {
            crate::error::Error::TensorCreationError {
                source: e.into(),
                descriptor: descriptor.clone(),
            }
        })?; // need to free tensor in case of error
        Ok(tensor)
    }

    fn read_tensor(
        &mut self,
        tensor: &crate::mlcontext::MLTensor,
        array: &mut [u8],
    ) -> crate::error::Result<()> {
        let cuda_tensor = &self.tensors[tensor.id];
        let stream = &cuda_tensor.stream;
        stream
            .memcpy_dtoh(&cuda_tensor.memory, array)
            .to_read_tensor_result(|| tensor.clone())?;
        stream
            .synchronize()
            .to_read_tensor_result(|| tensor.clone())?;
        Ok(())
    }

    fn write_tensor(
        &mut self,
        tensor: &crate::mlcontext::MLTensor,
        array: &[u8],
    ) -> crate::error::Result<()> {
        let cuda_tensor = &mut self.tensors[tensor.id];
        let stream = &cuda_tensor.stream;
        stream
            .memcpy_htod(array, &mut cuda_tensor.memory)
            .to_write_tensor_result(|| tensor.clone())?;
        Ok(())
    }

    fn dispatch(
        &mut self,
        graph: &mut crate::mlcontext::MLGraph,
        inputs: &HashMap<&str, &MLTensor>,
        outputs: &HashMap<&str, &MLTensor>,
    ) -> crate::error::Result<()> {
        let graph = graph
            .backend
            .as_trtx_engine_mut()
            .expect("Passed a graph that is not a Trtx engine to a TensorRT context");
        // TODO: just create a u64 device value for cudastreamwaitvalue64?
        let inference_stream = &graph.cuda_stream;

        for (input, tensor) in inputs.iter() {
            let cuda_tensor = &mut self.tensors[tensor.id];

            let (ptr, _) = cuda_tensor.memory.device_ptr_mut(&cuda_tensor.stream);
            unsafe {
                graph
                    .exec
                    .set_input_tensor_address(input, ptr as *mut c_void)
                    .to_dispatch_result()?
            };
            let event = cuda_tensor.stream.record_event(None).to_dispatch_result()?;
            inference_stream.wait(&event).to_dispatch_result()?;
            self.events.push(event);
        }
        for (output, tensor) in outputs.iter() {
            let cuda_tensor = &mut self.tensors[tensor.id];

            let (ptr, _) = cuda_tensor.memory.device_ptr_mut(&cuda_tensor.stream);
            unsafe {
                graph
                    .exec
                    .set_output_tensor_address(output, ptr as *mut c_void)
                    .to_dispatch_result()?
            };
        }

        unsafe {
            graph
                .exec
                .enqueue_v3(inference_stream.cu_stream() as *mut c_void)
                .to_dispatch_result()?
        };
        let inference_done = inference_stream.record_event(None).to_dispatch_result()?;
        for (_output, tensor) in outputs.iter() {
            let cuda_tensor = &mut self.tensors[tensor.id];
            cuda_tensor
                .stream
                .wait(&inference_done)
                .to_dispatch_result()?;
        }
        self.events.push(inference_done);

        Ok(())
    }

    fn rustnn_resize_tensor(
        &mut self,
        tensor: &mut MLTensor,
        new_shape: &[u64],
    ) -> crate::error::Result<()> {
        let mut new_desc = tensor.descriptor().clone();
        new_desc.set_shape(new_shape.to_vec());

        let new_bytes = new_desc.rustnn_required_bytes();
        let cuda_tensor = &mut self.tensors[tensor.id];

        if new_bytes > cuda_tensor.memory.num_bytes() {
            cuda_tensor.memory = unsafe { cuda_tensor.stream.alloc(new_bytes)? }
        }
        tensor.descriptor = new_desc;
        Ok(())
    }

    fn rustnn_set_tensor_capacity(
        &mut self,
        tensor: &mut MLTensor,
        max_shape: &[u64],
    ) -> crate::error::Result<()> {
        let new_bytes = ((max_shape.iter().product::<u64>() as usize
            * tensor.data_type().rustnn_element_size_bits())
            / 8) as usize;

        let cuda_tensor = &mut self.tensors[tensor.id];
        cuda_tensor.memory = unsafe { cuda_tensor.stream.alloc(new_bytes)? };
        Ok(())
    }
}

impl ListDevices for TrtxContext<'_> {
    fn list_devices() -> Vec<crate::backend_selection::BackendDevice> {
        trace!("Enumerating Trtx devices");
        let Ok(()) = result::init() else {
            warn!("Could not enumerate CUDA devices for TensorRT RTX");
            return Vec::new();
        };
        let Ok(device_count) = result::device::get_count() else {
            warn!("Could not get CUDA device count for TensorRT RTX");
            return Vec::new();
        };

        let mut devices = Vec::new();
        for cuda_device_idx in 0..device_count {
            let Ok(device) = result::device::get(cuda_device_idx) else {
                continue;
            };

            // SAFETY: `device` is returned by `result::device::get`, so it is a valid CUdevice.
            let Ok(major) = (unsafe {
                result::device::get_attribute(
                    device,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                )
            }) else {
                continue;
            };

            // SAFETY: `device` is returned by `result::device::get`, so it is a valid CUdevice.
            let Ok(minor) = (unsafe {
                result::device::get_attribute(
                    device,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                )
            }) else {
                continue;
            };

            // Accept Ampere+ devices only (compute capability >= 8.0).
            if major > 8 || (major == 8 && minor >= 0) {
                devices.push(crate::backend_selection::BackendDevice::Trtx {
                    cuda_device_idx: cuda_device_idx as u32,
                });
            }
        }

        info!("Found Trtx devices {devices:?} from {device_count} CUDA devices");
        devices
    }
}

#[cfg(test)]
#[cfg(feature = "trtx-runtime")]
mod tests {
    use crate::mlcontext::{MLBackendContext, MLTensorDescriptor};
    use crate::{backends::trtx::TrtxContext, mlcontext::ListDevices};

    #[test]
    fn test_context_creation() {
        let _ = pretty_env_logger::try_init();
        use crate::{backends::trtx::TrtxContext, mlcontext::ListDevices};
        let devices = TrtxContext::list_devices();
        let context = if let [first, ..] = devices.as_slice() {
            TrtxContext::new(*first.as_trtx_device().unwrap()).unwrap()
        } else {
            return;
        };

        assert!(context.accelerated());
    }

    #[test]
    fn test_builder() {
        let _ = pretty_env_logger::try_init();
        let devices = TrtxContext::list_devices();
        let mut context = if let [first, ..] = devices.as_slice() {
            TrtxContext::new(*first.as_trtx_device().unwrap()).unwrap()
        } else {
            return;
        };

        let _builder = context.create_builder().unwrap();
    }

    #[test]
    fn test_create_tensors() {
        let _ = pretty_env_logger::try_init();
        let devices = TrtxContext::list_devices();
        let mut context = if let [first, ..] = devices.as_slice() {
            TrtxContext::new(*first.as_trtx_device().unwrap()).unwrap()
        } else {
            return;
        };

        let mut desc = MLTensorDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            [2, 2].to_vec(),
        );
        desc.set_readable(true);
        desc.set_writable(true);
        let tensor = context.create_tensor(&desc).unwrap();

        let upload = vec![1.0f32, 2., 3., 4.];
        let mut download = vec![0.0f32; 4];
        context
            .write_tensor(&tensor, bytemuck::cast_slice(&upload))
            .unwrap();
        context
            .read_tensor(&tensor, bytemuck::cast_slice_mut(&mut download))
            .unwrap();
        assert_eq!(&upload, &download);
    }

    #[test]
    #[should_panic = "assertion failed: dst.len() >= src.len()"]
    fn test_write_too_large_tensor() {
        let _ = pretty_env_logger::try_init();
        let devices = TrtxContext::list_devices();
        let mut context = if let [first, ..] = devices.as_slice() {
            TrtxContext::new(*first.as_trtx_device().unwrap()).unwrap()
        } else {
            return;
        };
        let too_big = vec![0.0f32; 8];
        let mut desc = MLTensorDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            [2, 2].to_vec(),
        );
        desc.set_readable(true);
        desc.set_writable(true);
        let tensor = context.create_tensor(&desc).unwrap();
        context
            .write_tensor(&tensor, bytemuck::cast_slice(&too_big))
            .unwrap_err();
    }
}

use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::LazyLock;

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::{CudaContext, DriverError, result, sys};
use cudarc::driver::{CudaEvent, DevicePtrMut};
use log::debug;
use log::info;
use log::trace;
use log::warn;
use std::sync::{Arc, Mutex, OnceLock};
use trtx::CudaEngine;
use trtx::ExecutionContext;
use trtx::Refitter;
use trtx::host_memory::HostMemory;

use crate::GraphInfo;
use crate::backends::caching::CacheResult;
use crate::backends::caching::DefaultCache;
use crate::backends::caching::PersistentCache;
use crate::converters::TrtxConverter;
use crate::error::Error;

use crate::error::GraphBuilderError;
use crate::mlcontext::MLTensor;
use crate::mlcontext::{ListDevices, MLOperand};
use crate::mlcontext::{MLBackendBuilder, MLGraph};
use crate::mlcontext::{MLBackendContext, MLBackendGraph};

// TODO: also used in trtexec-rs. Should be part of trtx API?
enum HostMemoryOrVec<'memory> {
    HostMemory(HostMemory<'memory>),
    Cow(Cow<'memory, [u8]>),
}

impl<'memory> AsRef<[u8]> for HostMemoryOrVec<'memory> {
    fn as_ref(&self) -> &[u8] {
        match self {
            HostMemoryOrVec::HostMemory(host_memory) => host_memory.as_ref(),
            HostMemoryOrVec::Cow(items) => items.as_ref(),
        }
    }
}

impl<'memory> Deref for HostMemoryOrVec<'memory> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'buffer> From<HostMemory<'buffer>> for HostMemoryOrVec<'buffer> {
    fn from(value: HostMemory<'buffer>) -> Self {
        HostMemoryOrVec::HostMemory(value)
    }
}
impl<'memory> From<Cow<'memory, [u8]>> for HostMemoryOrVec<'memory> {
    fn from(value: Cow<'memory, [u8]>) -> Self {
        HostMemoryOrVec::Cow(value)
    }
}

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
    cuda_graphs: HashMap<Vec<(bool, String, usize)>, CapturedCudaGraph>,
}

struct CapturedCudaGraph {
    graph: sys::CUgraph,
    executable: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}

impl CapturedCudaGraph {
    fn finish_capture(stream: &Arc<CudaStream>) -> std::result::Result<Self, DriverError> {
        stream.context().bind_to_thread()?;
        let graph = unsafe { result::stream::end_capture(stream.cu_stream()) }?;
        if graph.is_null() {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE));
        }

        let mut executable = MaybeUninit::uninit();
        if let Err(error) =
            unsafe { sys::cuGraphInstantiateWithFlags(executable.as_mut_ptr(), graph, 0).result() }
        {
            let _ = unsafe { result::graph::destroy(graph) };
            return Err(error);
        }

        Ok(Self {
            graph,
            executable: unsafe { executable.assume_init() },
            stream: Arc::clone(stream),
        })
    }

    fn launch(&self) -> std::result::Result<(), DriverError> {
        self.stream.context().bind_to_thread()?;
        unsafe { result::graph::launch(self.executable, self.stream.cu_stream()) }
    }
}

impl Drop for CapturedCudaGraph {
    fn drop(&mut self) {
        if let Err(error) = self.stream.context().bind_to_thread() {
            warn!("Failed to destroy captured CUDA graph: {error}");
            return;
        }
        if let Err(error) = unsafe { result::graph::exec_destroy(self.executable) } {
            warn!("Failed to destroy captured CUDA graph executable: {error}");
        }
        if let Err(error) = unsafe { result::graph::destroy(self.graph) } {
            warn!("Failed to destroy captured CUDA graph: {error}");
        }
    }
}

/// CUDA graph capture/replay around `enqueue_v3`. Disabled by default.
/// Set `RUSTNN_TRTX_CUDA_GRAPHS=1` or `true` to enable.
static TRTX_CUDA_GRAPHS_ENABLED: OnceLock<bool> = OnceLock::new();

fn trtx_cuda_graphs_enabled() -> bool {
    *TRTX_CUDA_GRAPHS_ENABLED.get_or_init(|| {
        std::env::var("RUSTNN_TRTX_CUDA_GRAPHS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

fn enqueue_inference_v3(
    exec: &mut ExecutionContext<'_>,
    inference_stream: &CudaStream,
) -> crate::error::Result<()> {
    unsafe { exec.enqueue_v3(inference_stream.cu_stream() as *mut c_void) }.to_dispatch_result()
}

impl std::fmt::Debug for TrtxGraph<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrtxGraph")
            .field("engine", &"")
            .field("exec", &self.exec.name())
            .field("cuda_stream", &self.cuda_stream)
            .field("cuda_graph_count", &self.cuda_graphs.len())
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
    runtime: Arc<Mutex<trtx::Runtime<'context>>>,
    config: Arc<Mutex<trtx::BuilderConfig<'context>>>, // needs to be destroyed before builder
    builder: Arc<Mutex<trtx::Builder<'context>>>,
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
        let mut config = builder.create_config()?;
        // Strip marked weights weights from engine and makes them refittable, keeps other weights
        // (mostly scalars)
        config.set_flag(trtx::trtx_sys::BuilderFlag::kREFIT_INDIVIDUAL);
        config.set_flag(trtx::trtx_sys::BuilderFlag::kSTRIP_PLAN);
        let config = Arc::new(config.into());
        let runtime = Arc::new(trtx::Runtime::new(&LOGGER)?.into());
        debug!("Created new TrtxContext");
        Ok(Self {
            cuda_ctx,
            tensors: vec![],
            events: vec![],
            runtime,
            builder: Arc::new(builder.into()),
            config,
        })
    }
}

#[allow(dead_code)]
pub(crate) struct TrtxBuilder<'builder> {
    network: Mutex<Option<trtx::NetworkDefinition<'builder>>>,
    builder: Arc<Mutex<trtx::Builder<'builder>>>,
    config: Arc<Mutex<trtx::BuilderConfig<'builder>>>,
    cuda_context: Arc<CudaContext>,
    runtime: Arc<Mutex<trtx::Runtime<'builder>>>,
    operands: HashMap<String, MLOperand>,
    strings: Vec<String>, //_parser: Option<OnnxParser<'builder>>,
    caching_enabled: bool,
}

impl std::fmt::Debug for TrtxBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrtxBuilder").finish()
    }
}

static ENGINE_CACHE: LazyLock<CacheResult<DefaultCache>> =
    LazyLock::new(|| DefaultCache::new("trtx"));
static TRTX_SUFFIX: LazyLock<String> = LazyLock::new(|| {
    format!(
        "trtx_{}.{}.{}",
        unsafe { trtx::trtx_sys::get_tensorrt_major_version() },
        unsafe { trtx::trtx_sys::get_tensorrt_minor_version() },
        unsafe { trtx::trtx_sys::get_tensorrt_patch_version() }
    )
});

impl<'context, 'builder> MLBackendBuilder<'context, 'builder> for TrtxBuilder<'context> {
    /*async */
    fn build(
        &mut self,
        graph: GraphInfo,
    ) -> crate::error::Result<crate::mlcontext::MLGraph<'context>> {
        let mut engine_bytes: Option<HostMemoryOrVec> = None;
        let mut hash_key: Option<String> = None;

        if self.caching_enabled {
            let key = graph.hash_identifier_without_weights(&TRTX_SUFFIX);
            if let Ok(cache) = ENGINE_CACHE.as_ref()
                && let Ok(engine) = cache.get(&key)
            {
                debug!("Using cached engine of size {}", engine.len());
                engine_bytes = Some(engine.into());
            } else {
                debug!("Could not get cached engine. Rebuilding from network defintion...");
            }
            hash_key = Some(key);
        }

        if engine_bytes.is_none() {
            let mut network = self
                .network
                .lock()
                .unwrap()
                .take()
                .expect("Frontend API should prevent TrtxBuilder::build to be called twice");
            crate::converters::TrtxConverter::build_network(&graph, &mut network)?;
            for constant_id in graph.constant_operand_ids_to_handles.keys() {
                network.mark_weights_refittable(&format!("{constant_id}"))?;
            }

            let host_mem = self
                .builder
                .lock()
                .unwrap()
                .build_serialized_network(&mut network, &mut self.config.lock().unwrap())
                .map_err(|e| crate::error::Error::GraphBuildError { source: e.into() })?;
            if let Ok(cache) = ENGINE_CACHE.as_ref()
                && let Some(key) = hash_key
                && let Err(error) = cache.set(&key, &host_mem)
            {
                warn!("Failed to write engine to cache: {error}");
            }
            engine_bytes = Some(host_mem.into());
        }

        let engine_bytes =
            engine_bytes.expect("already got cached engine or tried building engine");

        self.cuda_context.bind_to_thread()?;
        let mut engine = self
            .runtime
            .lock()
            .unwrap()
            .deserialize_cuda_engine(&engine_bytes)
            .map_err(|e| crate::error::Error::GraphBuildError { source: e.into() })?;

        // TODO: wrap_err, when this fails usually the engine was built without kREFIT flag
        let mut refitter = Refitter::new(&engine, &LOGGER)?;

        for (id, constant) in graph.constant_operand_ids_to_handles.iter() {
            let operand = graph.operands.get(*id as usize);
            if let Some(operand) = operand.as_ref() {
                let trt_type = TrtxConverter::webnn_to_trt_dtype(operand.descriptor.data_type)?;
                let element_count = operand.descriptor.element_count().ok_or_else(|| {
                    std::convert::Into::<crate::error::Error>::into(
                        GraphBuilderError::InconsistentGraphInfo {
                            message: format!(
                                "Constant with dynamic size: id={id} operand={operand:#?}"
                            ),
                        },
                    )
                })?;
                let expected_bytes = element_count * trt_type.size_bits() / 8;
                if constant.data.len() != expected_bytes {
                    return Err(GraphBuilderError::InconsistentGraphInfo {
                        message: format!(
                            "Weight size mismatch: expected {expected_bytes} bytes, got {} bytes",
                            constant.data.len()
                        ),
                    }
                    .into());
                }
                let weight_name = format!("{id}");
                trace!("Trying to refit weight {weight_name}");
                unsafe {
                    refitter.set_named_weights_with_location(
                        &weight_name, // TODO: add API to name weights to trtx
                        trtx::trtx_sys::Weights {
                            type_: trt_type.into(),
                            values: constant.data.as_ptr() as *const std::ffi::c_void,
                            count: element_count as i64,
                        },
                        // TODO: register and upload during build, refit with device location
                        trtx::trtx_sys::nvinfer1::TensorLocation::kHOST,
                    )?
                };
            } else {
                return Err(GraphBuilderError::InconsistentGraphInfo {
                    message: format!(
                        "Inconsistent GraphInfo: Constant operation with {id} is missing operations"
                    ),
                }
                .into());
            }
        }
        refitter.refit_cuda_engine()?;

        let mut runtime_config = engine
            .create_runtime_config()
            .map_err(|e| crate::error::Error::GraphBuildError { source: e.into() })?;
        runtime_config
            .set_cuda_graph_strategy(trtx::CudaGraphStrategy::kDISABLED)
            .map_err(|e| crate::error::Error::GraphBuildError { source: e.into() })?;
        let exec = engine
            .create_execution_context_with_config(Rc::new(runtime_config))
            .map_err(|e| crate::error::Error::GraphBuildError { source: e.into() })?;

        MLGraph::new(
            MLBackendGraph::TrtxEngine(TrtxGraph {
                _engine: engine,
                exec,
                cuda_stream: self
                    .cuda_context
                    .new_stream()
                    .map_err(|e| crate::error::Error::GraphBuildError { source: e.into() })?,
                cuda_graphs: HashMap::new(),
            }),
            &graph,
        )
    }
}

#[allow(unused_variables)]
impl<'context> MLBackendContext<'context> for TrtxContext<'context> {
    fn accelerated(&self) -> bool {
        true
    }

    fn create_builder<'builder>(
        &'_ mut self,
    ) -> crate::error::Result<
        Box<dyn crate::mlcontext::MLBackendBuilder<'context, 'builder> + 'builder>,
    >
    where
        'context: 'builder,
    {
        let network = Some(
            self.builder
                .lock()
                .unwrap()
                .create_network(0)
                .map_err(|e| Error::BuilderCreationError {
                    source: Box::new(e),
                })?,
        )
        .into();
        //self.networks.push(network);
        Ok(Box::new(TrtxBuilder {
            network,
            builder: Arc::clone(&self.builder),
            config: Arc::clone(&self.config),
            runtime: Arc::clone(&self.runtime),
            cuda_context: Arc::clone(&self.cuda_ctx),
            operands: HashMap::new(),
            strings: vec![],
            // disabled for now, since feature experimental.
            // can be enabled with more test coverage, but will remain a double-sided sword
            // e.g. if you change trtx converter, changes might not be visible, since cache skips conversion
            // maybe the build hash of certain trtx related files could be included in hash
            caching_enabled: false,
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
        debug!(
            "Downloading tensor {cuda_tensor:?} to array (ptr={:?}, size={:?})",
            array.as_ptr(),
            array.len(),
        );
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
        debug!(
            "Uploading tensor {cuda_tensor:?} to array (ptr={:?}, size={:?})",
            array.as_ptr(),
            array.len(),
        );
        stream
            .memcpy_htod(array, &mut cuda_tensor.memory)
            .to_write_tensor_result(|| tensor.clone())?;
        stream
            .synchronize()
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

        // TODO: set shape for dynamic networks and validate shape of input/output
        // tensors with what the network expect (done automatically by setting io_shapes?)
        let cuda_graphs_enabled = trtx_cuda_graphs_enabled();
        let mut io_pointers = if cuda_graphs_enabled {
            Vec::with_capacity(inputs.len() + outputs.len())
        } else {
            Vec::new()
        };
        for (input, tensor) in inputs.iter() {
            let cuda_tensor = &mut self.tensors[tensor.id];

            let (ptr, _) = cuda_tensor.memory.device_ptr_mut(&cuda_tensor.stream);
            if cuda_graphs_enabled {
                io_pointers.push((false, (*input).to_owned(), ptr as usize));
            }
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
            if cuda_graphs_enabled {
                io_pointers.push((true, (*output).to_owned(), ptr as usize));
            }
            unsafe {
                graph
                    .exec
                    .set_output_tensor_address(output, ptr as *mut c_void)
                    .to_dispatch_result()?
            };
        }

        if cuda_graphs_enabled {
            io_pointers.sort_unstable();
            if let Some(cuda_graph) = graph.cuda_graphs.get(&io_pointers) {
                cuda_graph.launch().to_dispatch_result()?;
            } else {
                inference_stream
                    .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                    .to_dispatch_result()?;
                if let Err(source) = unsafe {
                    graph
                        .exec
                        .enqueue_v3(inference_stream.cu_stream() as *mut c_void)
                } {
                    if let Ok(captured_graph) =
                        unsafe { result::stream::end_capture(inference_stream.cu_stream()) }
                        && !captured_graph.is_null()
                    {
                        let _ = unsafe { result::graph::destroy(captured_graph) };
                    }
                    return Err(crate::error::Error::GraphDispatchError {
                        source: source.into(),
                    });
                }
                let cuda_graph =
                    CapturedCudaGraph::finish_capture(inference_stream).to_dispatch_result()?;
                // enqueue_v3 already ran during capture; only replay on later cache hits.
                graph.cuda_graphs.insert(io_pointers, cuda_graph);
            }
        } else {
            enqueue_inference_v3(&mut graph.exec, inference_stream)?;
        }
        let inference_done = inference_stream.record_event(None).to_dispatch_result()?;
        for tensor in outputs.values() {
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
        debug!(
            "Resizing tensor {cuda_tensor:?} old desc: {:?}, new shape {new_shape:?}, new bytes {new_bytes}",
            tensor.descriptor.shape()
        );

        if new_bytes > cuda_tensor.memory.num_bytes() {
            debug!("Need to reallocate for new size {new_bytes} bytes");
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
        let new_bytes = (max_shape.iter().product::<u64>() as usize
            * tensor.data_type().rustnn_element_size_bits())
            / 8;
        let required_bytes = tensor.descriptor().rustnn_required_bytes();
        if new_bytes < required_bytes {
            return Err(Error::TensorCapacityError {
                requested_shape: max_shape.to_vec(),
                current_shape: tensor.shape().to_vec(),
                requested_bytes: new_bytes as u64,
                required_bytes: required_bytes as u64,
            });
        }

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

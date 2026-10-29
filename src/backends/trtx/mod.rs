//! NVIDIA TensorRT-RTX backend (`trtx-runtime` feature).
//!
//! Graphs are lowered with [`TrtxConverter`] straight into a TensorRT network, built into an
//! engine, and executed with device tensors on a CUDA stream. Constants are marked
//! refittable and set after the build, so engines are cached by topology (not weights) in
//! the `trtx` cache category; the TensorRT runtime (JIT) cache lives in `trtx-jit`. Cache keys
//! include a hash of the converter sources so converter edits invalidate stale engines.
//! Behaviour is tuned with [`TrtxOptions`]; `TRTX_JSON_DUMP_PATH` dumps engine layer JSON.
//!
#![doc = include_str!("../../../docs/integration/tensorrt.md")]

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::LazyLock;
use std::sync::RwLock;

use cudarc::driver::CudaEvent;
use cudarc::driver::CudaStream;
use cudarc::driver::DevicePtr;
use cudarc::driver::{CudaContext, DriverError, result, sys};
use cudarc::driver::{CudaSlice, PinnedHostSlice};
use log::debug;
use log::info;
use log::trace;
use log::warn;
use std::sync::{Arc, Mutex};
use trtx::CudaEngine;
use trtx::ExecutionContext;
use trtx::Refitter;
use trtx::host_memory::HostMemory;

use crate::GraphInfo;
use crate::backends::caching::CacheResult;
#[cfg(not(feature = "zstd-cache-compression"))]
use crate::backends::caching::DefaultCache as TrtxCache;
use crate::backends::caching::PersistentCache;
#[cfg(feature = "zstd-cache-compression")]
use crate::backends::caching::ZstdCompressedFileCache as TrtxCache;
use crate::converters::TrtxConverter;
use crate::converters::trtx_logger as LOGGER;
use crate::error::Error;

use crate::error::GraphBuilderError;
use crate::graph::WeightsToHash;
use crate::mlcontext::ListDevices;
use crate::mlcontext::MLTensor;
use crate::mlcontext::RustNNOptions;
use crate::mlcontext::{MLBackendBuilder, MLGraph, MLNamedTensors};
use crate::mlcontext::{MLBackendContext, MLBackendGraph};
use crate::mlcontextoptions::TrtxOptions;

// Keep the I/O experiments separate while retaining access to backend-owned storage.
mod io_playground;
use io_playground::{IoConfig, upload_file_ref_constants};

/// Reexport [trtx::dynamically_load_tensorrt] so RustNN users (like pywebnn) can hint to location
/// of TensorRT library without adding trtx as dependency themselves
pub use trtx::dynamically_load_tensorrt;
/// Reexport TensorRT version information. Useful for users of [dynamically_load_tensorrt]
pub use trtx::trtx_sys::{
    get_tensorrt_major_version, get_tensorrt_minor_version, get_tensorrt_patch_version,
};

const TRTX_JSON_DUMP_PATH_ENV_VAR: &str = "TRTX_JSON_DUMP_PATH";

// TODO: also used in trtexec-rs. Should be part of trtx API?
enum HostMemoryOrVec<'memory> {
    HostMemory(HostMemory<'memory>),
    Cow(Cow<'memory, [u8]>),
}

// SAFETY: `HostMemory` uniquely owns the serialized TensorRT plan. After
// `build_serialized_network` returns, this code only moves that ownership to the calling thread,
// where the bytes are read and the allocation is eventually dropped. No thread accesses it
// concurrently, and the builder outlives the scoped worker and the returned host memory.
unsafe impl Send for HostMemoryOrVec<'_> {}

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

/// Errors of the TensorRT-RTX backend.
///
/// Variant fields carry the values named in the message; they are not documented separately.
#[derive(Debug, thiserror::Error)]
#[allow(missing_docs)]
pub enum TrtxError {
    /// A CUDA driver call failed.
    #[error("Cuda driver error: {source}")]
    CudaError {
        #[from]
        source: DriverError,
    },
    /// A TensorRT call failed.
    #[error("TensorRT error: {source}")]
    TrtxError {
        #[from]
        source: trtx::Error,
    },
    /// No cached engine and `TrtxOptions::fail_on_cache_miss` forbids building one.
    #[error(
        "Engine cache miss: cache_key={cache_key:?}. Failed engine build because TrtxOptions::fail_on_cache_miss options was enabled"
    )]
    TrtxEngineCacheMiss { cache_key: Option<String> },
    /// TensorRT rejected the refit of a constant operand. TensorRT itself only
    /// reports the weight name; "cannot be refitted" means the builder folded
    /// the constant into its consumers, which the converter must then mark as
    /// non-refittable.
    #[error(
        "Failed to refit constant operand {operand_id} ({:?} {:?}, consumed by {consumers:?}): {source}",
        .operand.descriptor.data_type,
        .operand.descriptor.static_or_max_shape()
    )]
    ConstantRefitFailed {
        operand_id: u32,
        operand: Box<crate::Operand>,
        consumers: Vec<String>,
        #[source]
        source: trtx::Error,
    },
}
/// Result of TensorRT-RTX backend operations.
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
    inference_count: u64,
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

    fn launch(&self, cuda_stream: &CudaStream) -> std::result::Result<(), DriverError> {
        cuda_stream.context().bind_to_thread()?;
        unsafe { result::graph::launch(self.executable, cuda_stream.cu_stream()) }
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
            .field("inference_count", &self.inference_count)
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

fn graph_build_error(source: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::GraphBuildError {
        source: Box::new(source),
    }
}

#[derive(Debug)]
pub(crate) struct HostTensor {
    memory: PinnedHostSlice<u8>,
    stream: Arc<CudaStream>,
    gpu_ptr: Option<*mut c_void>,
}

// Access to HostTensor is serialized by the context's RwLock, and the pointer
// is only an alias into `memory`, whose allocation is owned by this struct.
unsafe impl Send for HostTensor {}
unsafe impl Sync for HostTensor {}

impl HostTensor {
    fn new(cuda_ctx: &Arc<CudaContext>, size: usize) -> TrtxResult<Self> {
        let stream = cuda_ctx.new_stream()?;
        Self::new_with_stream(cuda_ctx, size, stream)
    }

    fn new_with_stream(
        cuda_ctx: &Arc<CudaContext>,
        size: usize,
        stream: Arc<CudaStream>,
    ) -> TrtxResult<Self> {
        debug!("Allocating pinned host tensor of size {size}");
        let memory = unsafe { cuda_ctx.alloc_pinned::<u8>(size.max(1))? };
        let host_ptr = memory.as_ptr()?;
        let mut device_ptr: sys::CUdeviceptr = 0;
        let gpu_ptr = unsafe {
            sys::cuPointerGetAttribute(
                (&mut device_ptr as *mut sys::CUdeviceptr).cast(),
                sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                host_ptr as sys::CUdeviceptr,
            )
        }
        .result()
        .ok()
        .and_then(|()| (device_ptr != 0).then_some(device_ptr as *mut c_void));

        Ok(Self {
            memory,
            stream,
            gpu_ptr,
        })
    }
}

#[cfg(feature = "cufile")]
#[derive(Debug)]
pub(crate) struct SharedCudaTensor {
    memory: Arc<CudaSlice<u8>>,
    offset: usize,
    len: usize,
    stream: Arc<CudaStream>,
}

#[derive(Debug)]
pub(crate) enum TrtxTensorStorage {
    TrtxTensor(TrtxTensor),
    HostTensor(HostTensor),
    #[cfg(feature = "cufile")]
    SharedCudaTensor(SharedCudaTensor), // multiple tensors share view to one big buffer
    Destroyed,
}

impl TrtxTensorStorage {
    fn stream(&self) -> Option<&Arc<CudaStream>> {
        match self {
            Self::TrtxTensor(tensor) => Some(&tensor.stream),
            Self::HostTensor(tensor) => Some(&tensor.stream),
            #[cfg(feature = "cufile")]
            Self::SharedCudaTensor(tensor) => Some(&tensor.stream),
            Self::Destroyed => None,
        }
    }

    fn gpu_ptr(&self) -> Option<*mut c_void> {
        match self {
            Self::TrtxTensor(tensor) => {
                Some(tensor.memory.device_ptr(&tensor.stream).0 as *mut c_void)
            }
            Self::HostTensor(tensor) => tensor.gpu_ptr,
            #[cfg(feature = "cufile")]
            Self::SharedCudaTensor(tensor) => Some(
                (tensor.memory.device_ptr(&tensor.stream).0 as usize + tensor.offset)
                    as *mut c_void,
            ),
            Self::Destroyed => None,
        }
    }

    fn num_bytes(&self) -> usize {
        match self {
            Self::TrtxTensor(tensor) => tensor.memory.num_bytes(),
            Self::HostTensor(tensor) => tensor.memory.num_bytes(),
            #[cfg(feature = "cufile")]
            Self::SharedCudaTensor(tensor) => tensor.len,
            Self::Destroyed => 0,
        }
    }
}

pub(crate) struct TrtxContext<'context> {
    cuda_ctx: Arc<CudaContext>,
    device_cache_id: String,
    tensors: Arc<RwLock<Vec<TrtxTensorStorage>>>,
    events: Vec<CudaEvent>,
    runtime: Arc<Mutex<trtx::Runtime<'context>>>,
    config: Arc<Mutex<trtx::BuilderConfig<'context>>>, // needs to be destroyed before builder
    builder: Arc<Mutex<trtx::Builder<'context>>>,
    options: TrtxOptions,
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

impl<'context> TrtxContext<'context> {
    pub(crate) fn new(cuda_device_idx: u32, options: Option<&RustNNOptions>) -> TrtxResult<Self> {
        // this retains the primary context
        let cuda_ctx = CudaContext::new(cuda_device_idx as usize)?;
        let device_cache_id = cuda_device_cache_id(&cuda_ctx)?;
        let mut builder = trtx::Builder::new(&LOGGER)?;
        let mut config = builder.create_config()?;
        // Strip marked weights weights from engine and makes them refittable, keeps other weights
        // (mostly scalars)
        config.set_flag(trtx::trtx_sys::BuilderFlag::kREFIT_INDIVIDUAL);
        config.set_flag(trtx::trtx_sys::BuilderFlag::kSTRIP_PLAN);
        // Keep float32 math at full precision, matching the GraphConverter path:
        // TF32 matmul/conv rounds mantissas to 10 bits and drifts many ULP from
        // CPU references (WebNN conformance compares against strict IEEE fp32).
        config.clear_flag(trtx::trtx_sys::BuilderFlag::kTF32);
        if std::env::var(TRTX_JSON_DUMP_PATH_ENV_VAR).is_ok() {
            config.set_profiling_verbosity(trtx::ProfilingVerbosity::kDETAILED);
        }
        let config = Arc::new(config.into());
        let runtime = Arc::new(trtx::Runtime::new(&LOGGER)?.into());
        debug!("Created new TrtxContext");
        Ok(Self {
            cuda_ctx,
            device_cache_id,
            tensors: Default::default(),
            events: vec![],
            runtime,
            builder: Arc::new(builder.into()),
            config,
            options: options.unwrap_or(&RustNNOptions::default()).trtx.clone(),
        })
    }
}

fn cuda_device_cache_id(cuda_ctx: &CudaContext) -> std::result::Result<String, DriverError> {
    let combined =
        cuda_ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID)? as u32;
    let compute_major = cuda_ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?
        as u32;
    let compute_minor = cuda_ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?
        as u32;
    Ok(format_cuda_device_cache_id(
        combined,
        compute_major,
        compute_minor,
    ))
}

fn format_cuda_device_cache_id(combined: u32, compute_major: u32, compute_minor: u32) -> String {
    // CUDA packs the PCI device ID into the high 16 bits and the vendor ID into the low 16 bits.
    let vendor_id = combined & 0xffff;
    let device_id = combined >> 16;
    format!("{vendor_id:04x}_{device_id:04x}_sm{compute_major}{compute_minor}")
}

#[cfg(not(feature = "trtx-enterprise"))]
impl Drop for TrtxContext<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.cuda_ctx.bind_to_thread() {
            warn!(
                "Failed to bind CUDA device {} before saving its TensorRT JIT cache: {error}",
                self.device_cache_id
            );
            return;
        }

        let runtime_cache = TRTX_RUNTIME_CACHE
            .lock()
            .unwrap()
            .get(&self.device_cache_id)
            .cloned();
        let Some(runtime_cache) = runtime_cache else {
            return;
        };

        let serialized = match runtime_cache.serialize() {
            Ok(serialized) => serialized,
            Err(error) => {
                warn!(
                    "Failed to serialize TensorRT JIT cache for CUDA device {}: {error}",
                    self.device_cache_id
                );
                return;
            }
        };
        if let Ok(cache) = JIT_CACHE.as_ref()
            && let Err(error) = cache.set(&jit_cache_disk_key(&self.device_cache_id), &serialized)
        {
            warn!(
                "Failed to save TensorRT JIT cache for CUDA device {}: {error}",
                self.device_cache_id
            );
        }
    }
}

pub(crate) struct TrtxBuilder<'builder> {
    network: Mutex<Option<trtx::NetworkDefinition<'builder>>>,
    builder: Arc<Mutex<trtx::Builder<'builder>>>,
    config: Arc<Mutex<trtx::BuilderConfig<'builder>>>,
    cuda_context: Arc<CudaContext>,
    runtime: Arc<Mutex<trtx::Runtime<'builder>>>,
    device_cache_id: String,
    tensors: Arc<RwLock<Vec<TrtxTensorStorage>>>,
    caching_enabled: bool,
    #[cfg(not(feature = "trtx-enterprise"))]
    runtime_cache_enabled: bool,
    fail_on_cache_miss: bool,
    io_config: IoConfig,
}

impl std::fmt::Debug for TrtxBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrtxBuilder").finish()
    }
}

const SOURCE_HASH: &str = include_str!(concat!(env!("OUT_DIR"), "/source_hash.txt"));

static ENGINE_CACHE: LazyLock<CacheResult<TrtxCache>> = LazyLock::new(|| TrtxCache::new("trtx"));
#[cfg(not(feature = "trtx-enterprise"))]
static JIT_CACHE: LazyLock<CacheResult<TrtxCache>> = LazyLock::new(|| TrtxCache::new("trtx-jit"));

#[cfg(not(feature = "trtx-enterprise"))]
static TRTX_RUNTIME_CACHE: LazyLock<Mutex<HashMap<String, Arc<trtx::RuntimeCache>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static TRTX_SUFFIX_ENGINES: LazyLock<String> = LazyLock::new(|| {
    format!(
        "trtx_{}.{}.{}_{}",
        unsafe { trtx::trtx_sys::get_tensorrt_major_version() },
        unsafe { trtx::trtx_sys::get_tensorrt_minor_version() },
        unsafe { trtx::trtx_sys::get_tensorrt_patch_version() },
        SOURCE_HASH
    )
});
static TRTX_SUFFIX_JIT_CACHE: LazyLock<String> = LazyLock::new(|| {
    format!(
        "trtx_{}.{}.{}",
        unsafe { trtx::trtx_sys::get_tensorrt_major_version() },
        unsafe { trtx::trtx_sys::get_tensorrt_minor_version() },
        unsafe { trtx::trtx_sys::get_tensorrt_patch_version() },
    )
});

#[cfg(not(feature = "trtx-enterprise"))]
fn jit_cache_disk_key(device_cache_id: &str) -> String {
    format!("{device_cache_id}_{}.cache", *TRTX_SUFFIX_JIT_CACHE)
}

impl<'context, 'builder> MLBackendBuilder<'context, 'builder> for TrtxBuilder<'context> {
    /*async */
    fn build(
        &mut self,
        mut graph: GraphInfo,
    ) -> crate::error::Result<crate::mlcontext::MLGraph<'context>> {
        let baked = TrtxConverter::gather_baked_constant_operand_ids(&graph);
        for &id in &baked {
            let Some(reference) = graph.constant_operand_ids_to_handles.get(&id) else {
                continue;
            };
            let data = match reference {
                crate::graph::ConstantReference::FileRef(file_ref) => {
                    use std::io::{Read, Seek, SeekFrom};
                    let mut file =
                        std::fs::File::open(&file_ref.path).map_err(graph_build_error)?;
                    file.seek(SeekFrom::Start(file_ref.offset as u64))
                        .map_err(graph_build_error)?;
                    let mut data = vec![0; file_ref.len];
                    file.read_exact(&mut data).map_err(graph_build_error)?;
                    data
                }
                crate::graph::ConstantReference::IdRef(id_ref) => {
                    let tensors = self.tensors.read().unwrap();
                    let tensor = tensors.get(id_ref.id as usize).ok_or_else(|| {
                        GraphBuilderError::InconsistentGraphInfo {
                            message: format!("Invalid baked tensor reference {id_ref:?}"),
                        }
                    })?;
                    // TODO: please note that were getting
                    match tensor {
                        TrtxTensorStorage::TrtxTensor(tensor) => {
                            tensor.stream.clone_dtoh(&tensor.memory)?
                        }
                        TrtxTensorStorage::HostTensor(tensor) => {
                            tensor.stream.synchronize()?;
                            tensor.memory.as_slice()?.to_vec()
                        }
                        #[cfg(feature = "cufile")]
                        TrtxTensorStorage::SharedCudaTensor(tensor) => tensor.stream.clone_dtoh(
                            &tensor
                                .memory
                                .slice(tensor.offset..tensor.offset + tensor.len),
                        )?,
                        TrtxTensorStorage::Destroyed => {
                            return Err(GraphBuilderError::InconsistentGraphInfo {
                                message: format!("Baked tensor {id_ref:?} is destroyed"),
                            }
                            .into());
                        }
                    }
                }
                _ => continue,
            };
            graph.constant_operand_ids_to_handles.insert(
                id,
                crate::graph::ConstantReference::OwnedData(crate::ConstantData {
                    data,
                    label: None,
                }),
            );
        }

        let unused_constants = TrtxConverter::unused_constant_operand_ids(&graph);
        let file_refs = graph
            .constant_operand_ids_to_handles
            .iter()
            .filter(|(id, _)| !unused_constants.contains(id))
            .filter_map(|(&constant_id, constant_ref)| match constant_ref {
                crate::graph::ConstantReference::FileRef(file_ref) => Some((
                    constant_id,
                    file_ref.clone(),
                    graph
                        .operands
                        .get(constant_id as usize)
                        .map(|operand| operand.descriptor.clone()),
                )),
                _ => None,
            })
            .collect::<Vec<_>>();

        let (engine_sender, engine_receiver) = std::sync::mpsc::sync_channel(1);
        let (upload_sender, upload_receiver) = std::sync::mpsc::sync_channel(1);

        // could consider rayon::scope for using rayon's thread pool if we were to use rayon elsewhere

        // create a thread scope to build and load weights in parallel
        // currently trtx-specific, but resolving unresolved constants could also move into generic
        // part when more backends support weightless building. This way .webnn-parsing and
        // .safetensor loading could happen in parallel
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let result = (|| -> crate::error::Result<HostMemoryOrVec<'_>> {
                    let mut engine_bytes: Option<HostMemoryOrVec<'_>> = None;
                    let mut hash_key = None;

                    if self.caching_enabled {
                        let key = graph.hash_identifier(&TRTX_SUFFIX_ENGINES, WeightsToHash::Some(&crate::converters::TrtxConverter::gather_baked_constant_operand_ids(&graph)));
                        if let Ok(cache) = ENGINE_CACHE.as_ref()
                            && let Ok(engine) = cache.get(&key)
                        {
                            debug!("Using cached engine of size {}", engine.len());
                            engine_bytes = Some(engine.into());
                        } else {
                            debug!(
                                "Could not get cached engine. Rebuilding from network definition..."
                            );
                        }
                        hash_key = Some(key);
                    }

                    if engine_bytes.is_none() {
                        if self.fail_on_cache_miss && self.caching_enabled {
                            return Err(TrtxError::TrtxEngineCacheMiss { cache_key: hash_key }.into());
                        }
                        let mut network = self.network.lock().unwrap().take().expect(
                            "Frontend API should prevent TrtxBuilder::build to be called twice",
                        );
                        crate::converters::TrtxConverter::build_network_with_placeholders(&graph, &mut network)?;
                        let non_refittable_constants =
                            crate::converters::TrtxConverter::gather_baked_constant_operand_ids(
                                &graph,
                            );
                        for constant_id in graph.constant_operand_ids_to_handles.keys() {
                            if !non_refittable_constants.contains(constant_id)
                                && !crate::converters::TrtxConverter::unused_constant_operand_ids(&graph).contains(constant_id) {
                                network.mark_weights_refittable(&format!("{constant_id}"))?;
                            }
                        }

                        let host_mem = self
                            .builder
                            .lock()
                            .unwrap()
                            .build_serialized_network(
                                &mut network,
                                &mut self.config.lock().unwrap(),
                            )
                            .map_err(graph_build_error)?;
                        if let Ok(cache) = ENGINE_CACHE.as_ref()
                            && let Some(key) = hash_key
                            && let Err(error) = cache.set(&key, &host_mem)
                        {
                            warn!("Failed to write engine to cache: {error}");
                        }
                        engine_bytes = Some(host_mem.into());
                    }

                    Ok(engine_bytes.expect("already got cached engine or tried building engine"))
                })();
                let _ = engine_sender.send(result);
            });

            //scope.spawn(|| { // do in main thread for now
            let result = upload_file_ref_constants(&self.cuda_context, &file_refs, &self.io_config);
            let _ = upload_sender.send(result);
            //});
        });

        let engine_bytes = engine_receiver.recv().map_err(graph_build_error)??;
        let uploaded_tensors = upload_receiver.recv().map_err(graph_build_error)??;
        let mut uploaded_tensor_ids = HashSet::new();
        if !uploaded_tensors.is_empty() {
            let mut tensors = self.tensors.write().unwrap();
            for (constant_id, tensor) in uploaded_tensors {
                let tensor_id = tensors.len() as u64;
                tensors.push(tensor);
                uploaded_tensor_ids.insert(tensor_id);
                graph.constant_operand_ids_to_handles.insert(
                    constant_id,
                    crate::graph::ConstantReference::IdRef(crate::graph::IdRef {
                        id: tensor_id,
                        label: None,
                    }),
                );
            }
        }

        self.cuda_context.bind_to_thread()?;
        let mut engine = self
            .runtime
            .lock()
            .unwrap()
            .deserialize_cuda_engine(&engine_bytes)
            .map_err(|e| crate::error::Error::GraphBuildError { source: e.into() })?;

        if let Ok(dump_dir) = std::env::var(TRTX_JSON_DUMP_PATH_ENV_VAR) {
            let mut key = graph.hash_identifier(
                &TRTX_SUFFIX_ENGINES,
                WeightsToHash::Some(
                    &crate::converters::TrtxConverter::gather_baked_constant_operand_ids(&graph),
                ),
            );
            key.push_str(".json");
            let inspector = engine.create_engine_inspector()?;
            let engine_layer_info_json =
                inspector.engine_information(trtx::LayerInformationFormat::kJSON)?;
            let path = std::path::PathBuf::from(dump_dir).join(key);
            if let Err(err) = std::fs::write(&path, engine_layer_info_json) {
                warn!("Failed to create file {path:?} for engine dump: {err}");
            } else {
                info!("Wrote trtx engine dump: {path:?}");
            }
        }

        let mut refitter = Refitter::new(&engine, &LOGGER)?;
        let max_weights = i32::try_from(graph.constant_operand_ids_to_handles.len())
            .map_err(graph_build_error)?;
        let mut remaining: std::collections::BTreeSet<String> =
            refitter.all_weights(max_weights)?.into_iter().collect();
        let stream = self.cuda_context.new_stream()?;
        while let Some(seed) = remaining.first().cloned() {
            let mut group = vec![seed];
            // Owned weights are uploaded for this group. Tensor references already reside on GPU.
            let mut source_buffers = Vec::new();
            let mut supplied = HashSet::new();
            loop {
                for name in &group {
                    if !supplied.insert(name.clone()) {
                        continue;
                    }
                    let id: u32 = name.parse().map_err(graph_build_error)?;
                    let reference = graph
                        .constant_operand_ids_to_handles
                        .get(&id)
                        .ok_or(GraphBuilderError::MissingConstantId { id })?;
                    let mut prototype = refitter.weights_prototype(name);
                    let tensors = self.tensors.read().unwrap();
                    prototype.values = if let Some(id_ref) = reference.as_id_ref() {
                        let tensor = tensors.get(id_ref.id as usize).ok_or_else(|| {
                            GraphBuilderError::InconsistentGraphInfo {
                                message: format!("Invalid tensor reference {id_ref:?}"),
                            }
                        })?;
                        let expected = graph.operands[id as usize]
                            .descriptor
                            .byte_length()
                            .ok_or_else(|| GraphBuilderError::InconsistentGraphInfo {
                                message: format!("Dynamic constant {id}"),
                            })?;
                        if tensor.num_bytes() < expected {
                            return Err(GraphBuilderError::InconsistentGraphInfo {
                                message: format!(
                                    "Weight {name} has {} bytes, expected at least {expected}",
                                    tensor.num_bytes()
                                ),
                            }
                            .into());
                        }
                        if let Some(source_stream) = tensor.stream() {
                            source_stream.synchronize()?;
                        }
                        tensor
                            .gpu_ptr()
                            .ok_or_else(|| GraphBuilderError::InconsistentGraphInfo {
                                message: format!(
                                    "Weight {name} is destroyed or not GPU-accessible"
                                ),
                            })?
                            .cast_const()
                    } else {
                        let data = graph.constant_data(id)?;
                        if graph.operands[id as usize].descriptor.byte_length() != Some(data.len())
                        {
                            return Err(GraphBuilderError::InconsistentGraphInfo {
                                message: format!(
                                    "Weight {name} has an invalid byte length: {}",
                                    data.len()
                                ),
                            }
                            .into());
                        }
                        let buffer = stream.clone_htod(data)?;
                        let ptr = buffer.device_ptr(&stream).0 as *const c_void;
                        source_buffers.push(buffer);
                        ptr
                    };
                    stream.synchronize()?;
                    // Sources remain alive until the synchronous refit completes and are then unset.
                    unsafe {
                        refitter
                            .set_named_weights_with_location(
                                name,
                                prototype,
                                trtx::TensorLocation::kDEVICE.into(),
                            )
                            .map_err(|source| TrtxError::ConstantRefitFailed {
                                operand_id: id,
                                operand: Box::new(graph.operands[id as usize].clone()),
                                consumers: graph
                                    .operations
                                    .iter()
                                    .filter(|op| op.all_input_operands().contains(&id))
                                    .map(|op| op.op_type().to_string())
                                    .collect(),
                                source,
                            })?;
                    }
                    remaining.remove(name);
                }
                let missing = refitter.missing_weights(max_weights)?;
                if missing.is_empty() {
                    break;
                }
                if missing.iter().all(|name| supplied.contains(name)) {
                    return Err(GraphBuilderError::InconsistentGraphInfo {
                        message: format!("TensorRT still requires supplied weights: {missing:?}"),
                    }
                    .into());
                }
                group.extend(missing.into_iter().filter(|name| !supplied.contains(name)));
            }
            refitter.refit_cuda_engine()?;
            for name in &group {
                if !refitter.unset_named_weights(name) {
                    return Err(GraphBuilderError::InconsistentGraphInfo {
                        message: format!("Could not unset refit weight {name}"),
                    }
                    .into());
                }
            }
            // Release only internal file-upload buffers; caller-owned tensors remain usable.
            let mut tensors = self.tensors.write().unwrap();
            for name in &group {
                let id: u32 = name.parse().map_err(graph_build_error)?;
                if let Some(id_ref) = graph.constant_operand_ids_to_handles[&id].as_id_ref()
                    && uploaded_tensor_ids.remove(&id_ref.id)
                {
                    tensors[id_ref.id as usize] = TrtxTensorStorage::Destroyed;
                }
            }
            drop(source_buffers);
        }

        let runtime_config = engine
            .create_runtime_config()
            .map_err(|e| crate::error::Error::GraphBuildError { source: e.into() })?;
        #[cfg(not(feature = "trtx-enterprise"))]
        let mut runtime_config = runtime_config;
        #[cfg(not(feature = "trtx-enterprise"))]
        if self.runtime_cache_enabled {
            let runtime_cache = {
                let mut caches = TRTX_RUNTIME_CACHE.lock().unwrap();
                if let Some(cache) = caches.get(&self.device_cache_id) {
                    Arc::clone(cache)
                } else {
                    let cache = runtime_config
                        .create_runtime_cache()
                        .map_err(|e| crate::error::Error::GraphBuildError { source: e.into() })?;
                    if let Ok(disk_cache) = JIT_CACHE.as_ref()
                        && let Ok(serialized) =
                            disk_cache.get(&jit_cache_disk_key(&self.device_cache_id))
                    {
                        match cache.deserialize(&serialized) {
                            Ok(()) => debug!(
                                "Loaded TensorRT JIT cache for CUDA device {} ({} bytes)",
                                self.device_cache_id,
                                serialized.len()
                            ),
                            Err(error) => warn!(
                                "Failed to deserialize TensorRT JIT cache for CUDA device {}: {error}",
                                self.device_cache_id
                            ),
                        }
                    }
                    let cache = Arc::new(cache);
                    caches.insert(self.device_cache_id.clone(), Arc::clone(&cache));
                    cache
                }
            };
            runtime_config
                .set_runtime_cache(runtime_cache)
                .map_err(|e| crate::error::Error::GraphBuildError { source: e.into() })?;
        }
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
                inference_count: 0,
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

        let tensors = Arc::clone(&self.tensors);

        Ok(Box::new(TrtxBuilder {
            network,
            builder: Arc::clone(&self.builder),
            config: Arc::clone(&self.config),
            runtime: Arc::clone(&self.runtime),
            device_cache_id: self.device_cache_id.clone(),
            cuda_context: Arc::clone(&self.cuda_ctx),
            tensors,
            caching_enabled: self.options.engine_caching,
            #[cfg(not(feature = "trtx-enterprise"))]
            runtime_cache_enabled: self.options.runtime_cache,
            fail_on_cache_miss: self.options.fail_on_cache_miss,
            io_config: Default::default(),
        }))
    }

    fn create_tensor(
        &mut self,
        descriptor: &crate::mlcontext::MLTensorDescriptor,
    ) -> crate::error::Result<crate::mlcontext::MLTensor> {
        let size = descriptor.rustnn_required_bytes();

        let tensor = HostTensor::new(&self.cuda_ctx, size).map_err(|e| {
            crate::error::Error::TensorCreationError {
                source: e.into(),
                descriptor: descriptor.clone(),
            }
        })?;
        let mut lock = self.tensors.write().unwrap();
        lock.push(TrtxTensorStorage::HostTensor(tensor));
        Ok(MLTensor::new(lock.len() - 1, descriptor.clone()))
    }

    fn create_constant_tensor(
        &mut self,
        descriptor: &crate::mlcontext::MLOperandDescriptor,
        input_data: &[u8],
    ) -> crate::error::Result<MLTensor> {
        let tensor_descriptor =
            crate::mlcontext::MLTensorDescriptor::from_operand_descriptor(descriptor);
        let backend_tensor = TrtxTensor::new(&self.cuda_ctx, input_data.len()).map_err(|e| {
            Error::TensorCreationError {
                source: e.into(),
                descriptor: tensor_descriptor.clone(),
            }
        })?;
        let id = {
            let mut lock = self.tensors.write().unwrap();
            lock.push(TrtxTensorStorage::TrtxTensor(backend_tensor));
            lock.len() - 1
        };
        let mut tensor = MLTensor::new(id, tensor_descriptor.clone());
        tensor.constant = true;
        self.write_tensor(&tensor, input_data)
            .map_err(|e| Error::TensorCreationError {
                source: e.into(),
                descriptor: tensor_descriptor,
            })?;
        Ok(tensor)
    }

    fn destroy_tensor(&mut self, tensor: &MLTensor) -> crate::error::Result<()> {
        if let Some(slot) = self.tensors.write().unwrap().get_mut(tensor.id) {
            *slot = TrtxTensorStorage::Destroyed;
        }
        Ok(())
    }

    fn read_tensor(
        &mut self,
        tensor: &crate::mlcontext::MLTensor,
        array: &mut [u8],
    ) -> crate::error::Result<()> {
        let lock = self.tensors.read().unwrap();
        let storage = &lock[tensor.id];
        debug!("Reading tensor {storage:?} into {} bytes", array.len());
        if matches!(storage, TrtxTensorStorage::Destroyed) {
            return Err(Error::TensorDestroyed {
                tensor: tensor.clone(),
            });
        }
        if array.len() > storage.num_bytes() {
            return Err(Error::TensorReadError {
                source: format!(
                    "read exceeds tensor storage: {} bytes > {}",
                    array.len(),
                    storage.num_bytes()
                )
                .into(),
                tensor: tensor.clone(),
            });
        }
        match storage {
            TrtxTensorStorage::TrtxTensor(cuda_tensor) => {
                cuda_tensor
                    .stream
                    .memcpy_dtoh(&cuda_tensor.memory, array)
                    .to_read_tensor_result(|| tensor.clone())?;
                cuda_tensor
                    .stream
                    .synchronize()
                    .to_read_tensor_result(|| tensor.clone())?;
            }
            TrtxTensorStorage::HostTensor(host_tensor) => {
                host_tensor
                    .stream
                    .synchronize()
                    .to_read_tensor_result(|| tensor.clone())?;
                array.copy_from_slice(
                    &host_tensor
                        .memory
                        .as_slice()
                        .to_read_tensor_result(|| tensor.clone())?[..array.len()],
                );
            }
            #[cfg(feature = "cufile")]
            TrtxTensorStorage::SharedCudaTensor(cuda_tensor) => {
                let source = cuda_tensor
                    .memory
                    .slice(cuda_tensor.offset..cuda_tensor.offset + array.len());
                cuda_tensor
                    .stream
                    .memcpy_dtoh(&source, array)
                    .to_read_tensor_result(|| tensor.clone())?;
                cuda_tensor
                    .stream
                    .synchronize()
                    .to_read_tensor_result(|| tensor.clone())?;
            }
            TrtxTensorStorage::Destroyed => {
                return Err(Error::TensorDestroyed {
                    tensor: tensor.clone(),
                });
            }
        }
        Ok(())
    }

    fn write_tensor(
        &mut self,
        tensor: &crate::mlcontext::MLTensor,
        array: &[u8],
    ) -> crate::error::Result<()> {
        let mut lock = self.tensors.write().unwrap();
        let storage = &mut lock[tensor.id];
        debug!("Writing {} bytes into tensor {storage:?}", array.len());
        if matches!(storage, TrtxTensorStorage::Destroyed) {
            return Err(Error::TensorDestroyed {
                tensor: tensor.clone(),
            });
        }
        if array.len() > storage.num_bytes() {
            return Err(Error::TensorWriteError {
                source: format!(
                    "write exceeds tensor storage: {} bytes > {}",
                    array.len(),
                    storage.num_bytes()
                )
                .into(),
                tensor: tensor.clone(),
            });
        }
        match storage {
            TrtxTensorStorage::TrtxTensor(cuda_tensor) => {
                cuda_tensor
                    .stream
                    .memcpy_htod(array, &mut cuda_tensor.memory)
                    .to_write_tensor_result(|| tensor.clone())?;
                cuda_tensor
                    .stream
                    .synchronize()
                    .to_write_tensor_result(|| tensor.clone())?;
            }
            TrtxTensorStorage::HostTensor(host_tensor) => {
                host_tensor
                    .stream
                    .synchronize()
                    .to_write_tensor_result(|| tensor.clone())?;
                host_tensor
                    .memory
                    .as_mut_slice()
                    .to_write_tensor_result(|| tensor.clone())?[..array.len()]
                    .copy_from_slice(array);
            }
            #[cfg(feature = "cufile")]
            TrtxTensorStorage::SharedCudaTensor(_) => {
                return Err(Error::TensorWriteError {
                    source: "cuFile-backed shared constant tensors are read-only".into(),
                    tensor: tensor.clone(),
                });
            }
            TrtxTensorStorage::Destroyed => {
                return Err(Error::TensorDestroyed {
                    tensor: tensor.clone(),
                });
            }
        }
        Ok(())
    }

    fn dispatch(
        &mut self,
        graph: &mut crate::mlcontext::MLGraph,
        inputs: &MLNamedTensors,
        outputs: &MLNamedTensors,
    ) -> crate::error::Result<()> {
        self.cuda_ctx.bind_to_thread()?;

        let graph = graph
            .backend
            .as_trtx_engine_mut()
            .expect("Passed a graph that is not a Trtx engine to a TensorRT context");
        // TODO: just create a u64 device value for cudastreamwaitvalue64?
        let inference_stream = &graph.cuda_stream;

        // TODO: set shape for dynamic networks and validate shape of input/output
        // tensors with what the network expect (done automatically by setting io_shapes?)
        let cuda_graphs_enabled = self.options.cuda_graphs;
        let mut io_pointers = if cuda_graphs_enabled {
            Vec::with_capacity(inputs.len() + outputs.len())
        } else {
            Vec::new()
        };
        for (input, tensor) in inputs.iter() {
            let lock = self.tensors.read().unwrap();
            let storage = &lock[tensor.id];
            let ptr = storage.gpu_ptr().ok_or_else(|| Error::GraphDispatchError {
                source: format!(
                    "input '{input}' tensor is destroyed or its host pointer is not GPU-accessible"
                )
                .into(),
            })?;
            let stream = storage.stream().ok_or_else(|| Error::TensorDestroyed {
                tensor: (*tensor).clone(),
            })?;
            if cuda_graphs_enabled {
                io_pointers.push((false, (*input).to_owned(), ptr as usize));
            }
            unsafe {
                graph
                    .exec
                    .set_input_tensor_address(input, ptr)
                    .to_dispatch_result()?
            };
            let event = stream.record_event(None).to_dispatch_result()?;
            inference_stream.wait(&event).to_dispatch_result()?;
            self.events.push(event);
        }
        for (output, tensor) in outputs.iter() {
            let lock = self.tensors.read().unwrap();
            let storage = &lock[tensor.id];
            let ptr = storage.gpu_ptr().ok_or_else(|| Error::GraphDispatchError {
                source: format!(
                    "output '{output}' tensor is destroyed or its host pointer is not GPU-accessible"
                )
                .into(),
            })?;
            if cuda_graphs_enabled {
                io_pointers.push((true, (*output).to_owned(), ptr as usize));
            }
            unsafe {
                graph
                    .exec
                    .set_output_tensor_address(output, ptr)
                    .to_dispatch_result()?
            };
            let event = storage
                .stream()
                .ok_or_else(|| Error::GraphDispatchError {
                    source: "Destroyed output tensor".into(),
                })?
                .record_event(None)
                .to_dispatch_result()?;
            inference_stream.wait(&event).to_dispatch_result()?;
            self.events.push(event);
        }

        if cuda_graphs_enabled {
            io_pointers.sort_unstable();
            if let Some(cuda_graph) = graph.cuda_graphs.get(&io_pointers) {
                cuda_graph.launch(inference_stream).to_dispatch_result()?;
            } else if graph.inference_count == 0 {
                // TensorRT-RTX performs setup and JIT work on the first enqueue. Run that
                // inference normally; the next dispatch can safely capture its enqueue.
                enqueue_inference_v3(&mut graph.exec, inference_stream)?;
            } else {
                inference_stream.synchronize().to_dispatch_result()?;
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
                cuda_graph.launch(inference_stream).to_dispatch_result()?;
                graph.cuda_graphs.insert(io_pointers, cuda_graph);
            }
        } else {
            enqueue_inference_v3(&mut graph.exec, inference_stream)?;
        }
        graph.inference_count = graph.inference_count.saturating_add(1);
        let inference_done = inference_stream.record_event(None).to_dispatch_result()?;
        for tensor in outputs.values() {
            let lock = self.tensors.read().unwrap();
            lock[tensor.id]
                .stream()
                .ok_or_else(|| Error::TensorDestroyed {
                    tensor: (*tensor).clone(),
                })?
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
        let mut lock = self.tensors.write().unwrap();
        let storage = &mut lock[tensor.id];
        debug!(
            "Resizing tensor {storage:?} old desc: {:?}, new shape {new_shape:?}, new bytes {new_bytes}",
            tensor.descriptor.shape()
        );

        if new_bytes > storage.num_bytes() {
            debug!("Need to reallocate for new size {new_bytes} bytes");
            match storage {
                TrtxTensorStorage::TrtxTensor(cuda_tensor) => {
                    cuda_tensor.memory = unsafe { cuda_tensor.stream.alloc(new_bytes)? };
                }
                TrtxTensorStorage::HostTensor(_) => {
                    *storage = TrtxTensorStorage::HostTensor(
                        HostTensor::new(&self.cuda_ctx, new_bytes).map_err(|source| {
                            Error::GraphDispatchError {
                                source: source.into(),
                            }
                        })?,
                    );
                }
                #[cfg(feature = "cufile")]
                TrtxTensorStorage::SharedCudaTensor(_) => {
                    *storage = TrtxTensorStorage::TrtxTensor(
                        TrtxTensor::new(&self.cuda_ctx, new_bytes).map_err(|source| {
                            Error::GraphDispatchError {
                                source: source.into(),
                            }
                        })?,
                    );
                }
                TrtxTensorStorage::Destroyed => {
                    return Err(Error::TensorDestroyed {
                        tensor: tensor.clone(),
                    });
                }
            }
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

        let mut lock = self.tensors.write().unwrap();
        let storage = &mut lock[tensor.id];
        match storage {
            TrtxTensorStorage::TrtxTensor(cuda_tensor) => {
                cuda_tensor.memory = unsafe { cuda_tensor.stream.alloc(new_bytes)? };
            }
            TrtxTensorStorage::HostTensor(_) => {
                *storage = TrtxTensorStorage::HostTensor(
                    HostTensor::new(&self.cuda_ctx, new_bytes).map_err(|source| {
                        Error::GraphDispatchError {
                            source: source.into(),
                        }
                    })?,
                );
            }
            #[cfg(feature = "cufile")]
            TrtxTensorStorage::SharedCudaTensor(_) => {
                *storage = TrtxTensorStorage::TrtxTensor(
                    TrtxTensor::new(&self.cuda_ctx, new_bytes).map_err(|source| {
                        Error::GraphDispatchError {
                            source: source.into(),
                        }
                    })?,
                );
            }
            TrtxTensorStorage::Destroyed => {
                return Err(Error::TensorDestroyed {
                    tensor: tensor.clone(),
                });
            }
        }
        Ok(())
    }
}

impl ListDevices for TrtxContext<'_> {
    fn list_devices() -> Vec<crate::backend_selection::BackendDevice> {
        trace!("Enumerating Trtx devices");
        // cudarc panics on a missing driver library before it can return a CUDA error.
        // SAFETY: Loads the system CUDA driver from trusted library search paths.
        if !unsafe { sys::is_culib_present() } {
            trace!("CUDA driver library not found; skipping TensorRT RTX");
            return Vec::new();
        }
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
            TrtxContext::new(*first.as_trtx_device().unwrap(), None).unwrap()
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
            TrtxContext::new(*first.as_trtx_device().unwrap(), None).unwrap()
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
            TrtxContext::new(*first.as_trtx_device().unwrap(), None).unwrap()
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
    fn test_write_too_large_tensor() {
        let _ = pretty_env_logger::try_init();
        let devices = TrtxContext::list_devices();
        let mut context = if let [first, ..] = devices.as_slice() {
            TrtxContext::new(*first.as_trtx_device().unwrap(), None).unwrap()
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
        let error = context
            .write_tensor(&tensor, bytemuck::cast_slice(&too_big))
            .unwrap_err();
        assert!(matches!(
            error,
            crate::error::Error::TensorWriteError { .. }
        ));
    }
}

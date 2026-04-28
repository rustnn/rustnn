#![cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DriverError, result, sys};
use cudarc::driver::{CudaEvent, DevicePtrMut};
use log::debug;
use log::info;
use log::trace;
use log::warn;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::Once;
use trtx::CudaEngine;
use trtx::ExecutionContext;

use crate::error::{Error, GraphError};
use crate::graph::{OperandDescriptor, get_static_or_max_size};
use crate::mlcontext::ListDevices;
use crate::mlcontext::MLBackendBuilder;
use crate::mlcontext::MLBackendContext;
use crate::mlcontext::MLTensor;

// Reexport to allow downstream users (e.g. pywebnn) to set path to TensorRT RTX lib
pub use trtx::dynamically_load_tensorrt;

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

/// Bytes per element for TensorRT tensor data types (used for buffer sizing).
fn trt_dtype_bytes_per_element(dtype: &trtx::DataType) -> usize {
    use trtx::DataType;
    match *dtype {
        DataType::kFLOAT => 4,
        DataType::kHALF => 2,
        DataType::kINT8 | DataType::kUINT8 | DataType::kBOOL => 1,
        DataType::kINT32 => 4,
        DataType::kINT64 => 8,
        _ => 4,
    }
}

fn trt_dtype_to_string(dtype: &trtx::DataType) -> &'static str {
    use trtx::DataType;
    match *dtype {
        DataType::kFLOAT => "float32",
        DataType::kHALF => "float16",
        DataType::kINT8 => "int8",
        DataType::kINT32 => "int32",
        DataType::kUINT8 => "uint8",
        DataType::kBOOL => "bool",
        DataType::kINT64 => "int64",
        _ => "float32",
    }
}

static TRTX_INIT: Once = Once::new();

/// Log handler that only emits messages at or above a minimum severity.
/// Used when RUSTNN_TRTX_LOG_VERBOSITY is set (e.g. by WPT test runner).
struct FilteredStderrLogger {
    min_severity: trtx::Severity,
}

impl trtx::LogHandler for FilteredStderrLogger {
    fn log(&self, severity: trtx::Severity, message: &str) {
        if severity <= self.min_severity {
            eprintln!("[TensorRT {:?}] {}", severity, message);
        }
    }
}

/// Create a TensorRT logger. If RUSTNN_TRTX_LOG_VERBOSITY is set to
/// internal_error, error, warning, info, or verbose, uses a filtered logger;
/// otherwise uses unfiltered stderr.
pub(crate) fn create_trtx_logger() -> Result<trtx::Logger, GraphError> {
    let Some(verbosity) = std::env::var_os("RUSTNN_TRTX_LOG_VERBOSITY") else {
        return trtx::Logger::stderr().map_err(|e| GraphError::TrtxRuntimeFailed {
            reason: format!("failed to create TensorRT logger: {e}"),
        });
    };
    let min_severity = match verbosity.to_str().unwrap_or("").to_lowercase().as_str() {
        "internal_error" => trtx::Severity::InternalError,
        "error" => trtx::Severity::Error,
        "warning" => trtx::Severity::Warning,
        "info" => trtx::Severity::Info,
        "verbose" => trtx::Severity::Verbose,
        _ => {
            return trtx::Logger::stderr().map_err(|e| GraphError::TrtxRuntimeFailed {
                reason: format!("failed to create TensorRT logger: {e}"),
            });
        }
    };
    let handler = FilteredStderrLogger { min_severity };
    trtx::Logger::new(handler).map_err(|e| GraphError::TrtxRuntimeFailed {
        reason: format!("failed to create TensorRT logger: {e}"),
    })
}

/// Load TensorRT and ONNX parser libraries once per process (no-op when using mock).
/// Called by the converter and executor so the library is loaded before any trtx API use.
pub(crate) fn ensure_trtx_loaded() -> Result<(), GraphError> {
    #[cfg(feature = "trtx-runtime")]
    {
        let mut result = Ok(());
        TRTX_INIT.call_once(|| {
            result = trtx::dynamically_load_tensorrt(None::<&str>).map_err(|e| {
                GraphError::TrtxRuntimeFailed {
                    reason: format!("failed to load TensorRT library: {e}"),
                }
            });
            if result.is_ok() {
                result = trtx::dynamically_load_tensorrt_onnxparser(None::<&str>).map_err(|e| {
                    GraphError::TrtxRuntimeFailed {
                        reason: format!("failed to load TensorRT ONNX parser library: {e}"),
                    }
                });
            }
        });
        result
    }
    #[cfg(not(feature = "trtx-runtime"))]
    {
        TRTX_INIT.call_once(|| {});
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TrtxOutput {
    pub name: String,
    pub shape: Vec<i64>,
    pub data_type: String,
}

/// Input tensor for TensorRT execution. Caller provides raw bytes in the format
/// expected by the engine (e.g. f32 or f16 little-endian per element).
pub struct TrtxInput {
    pub name: String,
    /// Raw tensor bytes (length must match engine's expected size for this tensor).
    pub data: Vec<u8>,
}

/// Output tensor with raw bytes and data type so the caller can interpret or convert.
pub struct TrtxOutputWithData {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
    pub data_type: String,
}

/// Detect if bytes are ONNX format (starts with ONNX magic header)
fn is_onnx_format(bytes: &[u8]) -> bool {
    // ONNX protobuf files typically start with 0x08 or contain "onnx" in the header
    // For simplicity, check if it looks like protobuf
    bytes.len() >= 4 && (bytes[0] == 0x08 || bytes.starts_with(b"onnx"))
}

/// Run ONNX model or TensorRT engine with zero-filled inputs
/// This is useful for validation and testing graph structure
///
/// For native WebNN [`crate::converters::TrtxConverter`] engines (not ONNX), input tensor names
/// must be [`TrtxConverter::engine_binding_name`] for each graph input operand id.
///
/// If model_bytes appears to be ONNX format, it will be parsed as ONNX and built into an engine.
/// Otherwise, it will be treated as a pre-serialized TensorRT engine.
pub fn run_trtx_zeroed(
    model_bytes: &[u8],
    inputs: &HashMap<String, OperandDescriptor>,
) -> Result<Vec<TrtxOutput>, GraphError> {
    ensure_trtx_loaded()?;

    if is_onnx_format(model_bytes) {
        // ONNX path: trtx crate expects f32 inputs
        let mut input_tensors = Vec::new();
        for (name, desc) in inputs {
            let shape: Vec<usize> = desc
                .shape
                .iter()
                .map(|s| get_static_or_max_size(s) as usize)
                .collect();
            let total: usize = shape.iter().product();
            input_tensors.push(trtx::executor::TensorInput {
                name: name.clone(),
                shape,
                data: vec![0f32; total.max(1)],
            });
        }
        let outputs =
            trtx::executor::run_onnx_with_tensorrt(model_bytes, &input_tensors).map_err(|e| {
                GraphError::TrtxRuntimeFailed {
                    reason: format!("TensorRT execution from ONNX failed: {e}"),
                }
            })?;
        return Ok(outputs
            .into_iter()
            .map(|o| TrtxOutput {
                name: o.name,
                shape: o.shape.iter().map(|&s| s as i64).collect(),
                data_type: "float32".to_string(),
            })
            .collect());
    }

    // Engine path: byte inputs (zeros)
    let mut input_list = Vec::new();
    for (name, desc) in inputs {
        let byte_len = desc.byte_length().unwrap_or(0).max(1);
        input_list.push(TrtxInput {
            name: name.clone(),
            data: vec![0u8; byte_len],
        });
    }
    let outputs = execute_trtx_engine(model_bytes, &input_list).map_err(|e| {
        GraphError::TrtxRuntimeFailed {
            reason: format!("TensorRT engine execution failed: {e}"),
        }
    })?;
    Ok(outputs
        .into_iter()
        .map(|o| TrtxOutput {
            name: o.name,
            shape: o.shape.iter().map(|&s| s as i64).collect(),
            data_type: o.data_type,
        })
        .collect())
}

/// Execute a pre-built TensorRT engine
fn execute_trtx_engine(
    engine_bytes: &[u8],
    inputs: &[TrtxInput],
) -> Result<Vec<TrtxOutputWithData>, trtx::Error> {
    // Create logger and runtime
    let logger = create_trtx_logger().map_err(|e| trtx::Error::Runtime(e.to_string()))?;
    let mut runtime = trtx::Runtime::new(&logger)?;

    // Deserialize engine
    let mut engine = runtime.deserialize_cuda_engine(engine_bytes)?;
    let mut context = engine.create_execution_context()?;

    // Get tensor information
    let num_tensors = engine.nb_io_tensors()?;

    // Prepare CUDA buffers for inputs and outputs
    let mut device_buffers: Vec<(String, trtx::DeviceBuffer)> = Vec::new();
    let mut output_info: Vec<(String, Vec<usize>, trtx::DataType)> = Vec::new();

    // Process each tensor - allocate buffers for ALL tensors (inputs and outputs)
    // TensorRT requires ALL tensor addresses to be set, even for intermediate results
    for i in 0..num_tensors {
        let name = engine.io_tensor_name(i)?;
        let dtype = engine.tensor_data_type(&name)?;
        let bytes_per_elem = trt_dtype_bytes_per_element(&dtype);

        if let Some(input) = inputs.iter().find(|inp| inp.name == name) {
            let expected_shape_i64 = engine.tensor_shape(&name)?;
            let expected_shape: Vec<usize> =
                expected_shape_i64.iter().map(|&d| d as usize).collect();
            let expected_size = expected_shape.iter().product::<usize>() * bytes_per_elem;

            if input.data.len() != expected_size {
                return Err(trtx::Error::InvalidArgument(format!(
                    "Input tensor '{}' size mismatch: expected {} bytes, got {}",
                    name,
                    expected_size,
                    input.data.len()
                )));
            }

            let mut buffer = trtx::DeviceBuffer::new(expected_size)?;
            buffer.copy_from_host(&input.data)?;

            unsafe {
                context.set_tensor_address(&name, buffer.as_ptr())?;
            }

            device_buffers.push((name.clone(), buffer));
        } else {
            // Non-input tensor (output or intermediate) - allocate buffer
            let shape_i64 = engine.tensor_shape(&name)?;
            let shape: Vec<usize> = shape_i64.iter().map(|&d| d as usize).collect();

            let num_elements: usize = shape.iter().product();
            let size_bytes = num_elements * bytes_per_elem;
            let buffer = trtx::DeviceBuffer::new(size_bytes)?;

            unsafe {
                context.set_tensor_address(&name, buffer.as_ptr())?;
            }

            output_info.push((name.clone(), shape, dtype));
            device_buffers.push((name.clone(), buffer));
        }
    }

    // Execute inference
    unsafe {
        context.enqueue_v3(trtx::cuda::default_stream())?;
    }

    // Synchronize to ensure completion
    trtx::cuda::synchronize()?;

    let mut outputs = Vec::new();
    for (name, shape, dtype) in output_info {
        if let Some((_, buffer)) = device_buffers.iter().find(|(n, _)| n == &name) {
            let size_bytes = shape.iter().product::<usize>() * trt_dtype_bytes_per_element(&dtype);
            let mut host_data = vec![0u8; size_bytes];
            buffer.copy_to_host(&mut host_data)?;
            outputs.push(TrtxOutputWithData {
                name,
                shape,
                data: host_data,
                data_type: trt_dtype_to_string(&dtype).to_string(),
            });
        }
    }
    Ok(outputs)
}

/// Run a pre-built TensorRT engine with the given byte inputs.
/// Caller is responsible for encoding inputs (e.g. f32 or f16) and decoding outputs.
/// ONNX model bytes are not supported here; use the trtx crate or run_trtx_zeroed for that.
pub fn run_trtx_with_inputs(
    engine_bytes: &[u8],
    inputs: Vec<TrtxInput>,
) -> Result<Vec<TrtxOutputWithData>, GraphError> {
    ensure_trtx_loaded()?;
    if is_onnx_format(engine_bytes) {
        return Err(GraphError::TrtxRuntimeFailed {
            reason: "run_trtx_with_inputs expects a serialized TensorRT engine, not ONNX bytes"
                .to_string(),
        });
    }
    execute_trtx_engine(engine_bytes, &inputs).map_err(|e| GraphError::TrtxRuntimeFailed {
        reason: format!("TensorRT engine execution failed: {e}"),
    })
}

pub(crate) struct TrtxGraph<'context> {
    engine: CudaEngine<'context>,
    exec: ExecutionContext<'context>,
    cuda_stream: CudaStream,
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
    runtime: trtx::Runtime<'context>,
    builder: trtx::Builder<'context>,
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
        let builder = trtx::Builder::new(&LOGGER)?;
        let runtime = trtx::Runtime::new(&LOGGER)?;
        debug!("Created new TrtxContext");
        Ok(Self {
            cuda_ctx,
            tensors: vec![],
            events: vec![],
            runtime,
            builder,
        })
    }
}

#[allow(dead_code)]
pub(crate) struct TrtxBuilder<'builder> {
    network: trtx::NetworkDefinition<'builder>,
}
impl std::fmt::Debug for TrtxBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrtxBuilder").finish()
    }
}

impl<'context> MLBackendBuilder<'context> for TrtxBuilder<'context> {
    /*async */
    fn build(&self) -> crate::error::Result<crate::mlcontext::MLGraph<'context>> {
        todo!()
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
            .create_network(0)
            .map_err(|e| Error::BuilderCreationError {
                source: Box::new(e),
            })?;
        //self.networks.push(network);
        Ok(Box::new(TrtxBuilder { network }))
    }

    fn create_tensor(
        &mut self,
        descriptor: &crate::mlcontext::MLTensorDescriptor,
    ) -> crate::error::Result<crate::mlcontext::MLTensor> {
        let bits = descriptor.data_type().element_size_bits();
        let elements = descriptor.shape().iter().copied().product::<u64>() as usize;
        let size = bits * elements / 8;

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
        inputs: &HashMap<&str, MLTensor>,
        outputs: &HashMap<&str, MLTensor>,
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
                devices.push(crate::backend_selection::BackendDevice::TrtxDevice {
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
    use crate::{executors::trtx::TrtxContext, mlcontext::ListDevices};

    #[test]
    fn test_context_creation() {
        let _ = pretty_env_logger::try_init();
        use crate::{executors::trtx::TrtxContext, mlcontext::ListDevices};
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

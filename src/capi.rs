//! C ABI for the public graph-construction API.
//!
//! The ABI intentionally wraps only types and methods exposed by `mlcontext` and
//! `mlgraphbuilder`. Rust values remain opaque and must be released with their
//! matching destroy function.
//!
//! # Safety
//!
//! For every exported function, non-null pointers must refer to live values of
//! the documented type for the duration of the call. Buffers must contain at
//! least the supplied number of elements or bytes. Handles must be destroyed
//! exactly once with their matching destroy function and must not be used after
//! destruction. The generated C header and C++ wrapper express these contracts
//! for non-Rust callers.

#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::slice;

use crate::mlcontext::{
    Backend, BackendDevice, DeviceType, MLContext, MLContextOptions, MLGraph, MLNamedOperands,
    MLNamedTensors, MLOperand, MLOperandDescriptor, MLPowerPreference, MLTensor,
    MLTensorDescriptor, RustNNOptions, TrtxOptions,
};
use crate::mlgraphbuilder::MLGraphBuilder;
use crate::operator_enums::MLOperandDataType;
use crate::operator_options::MLOperatorOptions;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RustnnStatus {
    Success = 0,
    NullPointer = 1,
    InvalidUtf8 = 2,
    InvalidArgument = 3,
    Error = 4,
    Panic = 5,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RustnnDataType {
    Float32 = 0,
    Float16 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Int8 = 6,
    Uint8 = 7,
    Int4 = 8,
    Uint4 = 9,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RustnnPowerPreference {
    Default = 0,
    HighPerformance = 1,
    LowPower = 2,
}

impl From<RustnnPowerPreference> for MLPowerPreference {
    fn from(value: RustnnPowerPreference) -> Self {
        match value {
            RustnnPowerPreference::Default => Self::Default,
            RustnnPowerPreference::HighPerformance => Self::HighPerformance,
            RustnnPowerPreference::LowPower => Self::LowPower,
        }
    }
}

/// Runtime implementation requested for an [`RustnnContextOptions`] context.
/// `Automatic` leaves backend selection to rustnn.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RustnnBackend {
    #[default]
    Automatic = 0,
    Onnx = 1,
    Trtx = 2,
    Coreml = 3,
    Litert = 4,
    Cann = 5,
}

impl RustnnBackend {
    fn hint(self) -> Option<Backend> {
        match self {
            Self::Automatic => None,
            Self::Onnx => Some(Backend::Onnx),
            Self::Trtx => Some(Backend::Trtx),
            Self::Coreml => Some(Backend::Coreml),
            Self::Litert => Some(Backend::Litert),
            Self::Cann => Some(Backend::Cann),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RustnnDeviceType {
    #[default]
    Cpu = 0,
    Gpu = 1,
    Npu = 2,
}

impl From<RustnnDeviceType> for DeviceType {
    fn from(value: RustnnDeviceType) -> Self {
        match value {
            RustnnDeviceType::Cpu => Self::Cpu,
            RustnnDeviceType::Gpu => Self::Gpu,
            RustnnDeviceType::Npu => Self::Npu,
        }
    }
}

impl From<RustnnDataType> for MLOperandDataType {
    fn from(value: RustnnDataType) -> Self {
        match value {
            RustnnDataType::Float32 => Self::Float32,
            RustnnDataType::Float16 => Self::Float16,
            RustnnDataType::Int32 => Self::Int32,
            RustnnDataType::Uint32 => Self::Uint32,
            RustnnDataType::Int64 => Self::Int64,
            RustnnDataType::Uint64 => Self::Uint64,
            RustnnDataType::Int8 => Self::Int8,
            RustnnDataType::Uint8 => Self::Uint8,
            RustnnDataType::Int4 => Self::Int4,
            RustnnDataType::Uint4 => Self::Uint4,
        }
    }
}

impl From<MLOperandDataType> for RustnnDataType {
    fn from(value: MLOperandDataType) -> Self {
        match value {
            MLOperandDataType::Float32 => Self::Float32,
            MLOperandDataType::Float16 => Self::Float16,
            MLOperandDataType::Int32 => Self::Int32,
            MLOperandDataType::Uint32 => Self::Uint32,
            MLOperandDataType::Int64 => Self::Int64,
            MLOperandDataType::Uint64 => Self::Uint64,
            MLOperandDataType::Int8 => Self::Int8,
            MLOperandDataType::Uint8 => Self::Uint8,
            MLOperandDataType::Int4 => Self::Int4,
            MLOperandDataType::Uint4 => Self::Uint4,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RustnnUnaryOperation {
    Abs,
    RoundEven,
    Ceil,
    Cos,
    Exp,
    Floor,
    Gelu,
    Log,
    Neg,
    Relu,
    Sigmoid,
    Sin,
    Sqrt,
    Tan,
    Tanh,
    Erf,
    Reciprocal,
    Sign,
    LogicalNot,
    Identity,
    Softplus,
    Softsign,
    IsNan,
    IsInfinite,
    Shape,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RustnnBinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
    Equal,
    Greater,
    GreaterOrEqual,
    Lesser,
    LesserOrEqual,
    NotEqual,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    Matmul,
}

pub struct RustnnOperandDescriptor(MLOperandDescriptor);
pub struct RustnnTensorDescriptor(MLTensorDescriptor);
pub struct RustnnOperand(MLOperand);
pub struct RustnnGraphBuilder(MLGraphBuilder<'static, 'static>);
pub struct RustnnContext(MLContext<'static>);
pub struct RustnnGraph(MLGraph<'static>);
pub struct RustnnTensor(MLTensor);

#[repr(C)]
pub struct RustnnNamedOperand {
    pub name: *const c_char,
    pub operand: *const RustnnOperand,
}

#[repr(C)]
pub struct RustnnNamedTensor {
    pub name: *const c_char,
    pub tensor: *const RustnnTensor,
}

/// A concrete backend device. `device_index` is the ONNX execution-provider
/// device index or TensorRT CUDA device index. It must be zero for the other
/// backends, whose Rust device variants do not carry an index.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RustnnBackendDevice {
    pub backend: RustnnBackend,
    pub device_type: RustnnDeviceType,
    pub device_index: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RustnnTrtxOptions {
    pub engine_caching: bool,
    pub runtime_cache: bool,
    pub fail_on_cache_miss: bool,
    pub cuda_graphs: bool,
}

impl Default for RustnnTrtxOptions {
    fn default() -> Self {
        let defaults = TrtxOptions::default();
        Self {
            engine_caching: defaults.engine_caching,
            runtime_cache: defaults.runtime_cache,
            fail_on_cache_miss: defaults.fail_on_cache_miss,
            cuda_graphs: defaults.cuda_graphs,
        }
    }
}

/// RustNN-specific backend options. The other backend option structures in the
/// Rust API are currently empty, so TensorRT is the only member with settings.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RustnnOptions {
    pub trtx: RustnnTrtxOptions,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RustnnContextOptions {
    pub power_preference: RustnnPowerPreference,
    pub accelerated: bool,
    /// `Automatic` corresponds to no Rust backend hint.
    pub backend_hint: RustnnBackend,
    /// Whether `device_hint` should override automatic device selection.
    pub has_device_hint: bool,
    pub device_hint: RustnnBackendDevice,
    /// Whether `rustnn_options` should replace the Rust defaults.
    pub has_rustnn_options: bool,
    pub rustnn_options: RustnnOptions,
}

impl Default for RustnnContextOptions {
    fn default() -> Self {
        Self {
            power_preference: RustnnPowerPreference::Default,
            accelerated: false,
            backend_hint: RustnnBackend::Automatic,
            has_device_hint: false,
            device_hint: RustnnBackendDevice::default(),
            has_rustnn_options: false,
            rustnn_options: RustnnOptions::default(),
        }
    }
}

/// Options shared by the unary and binary operations exposed by this ABI.
/// A null label is equivalent to an empty label.
#[repr(C)]
pub struct RustnnOperatorOptions {
    pub label: *const c_char,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(message: impl std::fmt::Display) {
    let message = message.to_string().replace('\0', "\\0");
    LAST_ERROR.with(|slot| *slot.borrow_mut() = CString::new(message).unwrap_or_default());
}

fn ffi_call(f: impl FnOnce() -> RustnnStatus) -> RustnnStatus {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(status) => status,
        Err(_) => {
            set_error("panic while executing rustnn C API call");
            RustnnStatus::Panic
        }
    }
}

unsafe fn required_ref<'a, T>(value: *const T, name: &str) -> Result<&'a T, RustnnStatus> {
    if value.is_null() {
        set_error(format!("{name} must not be null"));
        Err(RustnnStatus::NullPointer)
    } else {
        Ok(unsafe { &*value })
    }
}

unsafe fn required_mut<'a, T>(value: *mut T, name: &str) -> Result<&'a mut T, RustnnStatus> {
    if value.is_null() {
        set_error(format!("{name} must not be null"));
        Err(RustnnStatus::NullPointer)
    } else {
        Ok(unsafe { &mut *value })
    }
}

fn backend_device_from_c(device: RustnnBackendDevice) -> Result<BackendDevice, RustnnStatus> {
    let device_type = device.device_type.into();
    match device.backend {
        RustnnBackend::Automatic => {
            set_error("device_hint.backend must name a concrete backend");
            Err(RustnnStatus::InvalidArgument)
        }
        RustnnBackend::Onnx => Ok(BackendDevice::Onnx {
            ep_device_idx: device.device_index,
            device_type,
        }),
        RustnnBackend::Trtx => {
            if device.device_type != RustnnDeviceType::Gpu {
                set_error("a TensorRT device hint must use the GPU device type");
                return Err(RustnnStatus::InvalidArgument);
            }
            let cuda_device_idx = match u32::try_from(device.device_index) {
                Ok(index) => index,
                Err(_) => {
                    set_error("TensorRT device index does not fit in uint32_t");
                    return Err(RustnnStatus::InvalidArgument);
                }
            };
            Ok(BackendDevice::Trtx { cuda_device_idx })
        }
        RustnnBackend::Coreml => {
            if device.device_index != 0 {
                set_error("CoreML device hints do not support a non-zero device index");
                return Err(RustnnStatus::InvalidArgument);
            }
            Ok(BackendDevice::Coreml { device_type })
        }
        RustnnBackend::Litert => {
            if device.device_index != 0 {
                set_error("LiteRT device hints do not support a non-zero device index");
                return Err(RustnnStatus::InvalidArgument);
            }
            Ok(BackendDevice::LiteRt { device_type })
        }
        RustnnBackend::Cann => {
            if device.device_index != 0 {
                set_error("CANN device hints do not support a non-zero device index");
                return Err(RustnnStatus::InvalidArgument);
            }
            Ok(BackendDevice::Cann { device_type })
        }
    }
}

fn context_options_from_c(
    options: &RustnnContextOptions,
) -> Result<MLContextOptions, RustnnStatus> {
    let mut converted = MLContextOptions::new(options.power_preference.into(), options.accelerated);
    if let Some(backend) = options.backend_hint.hint() {
        converted = converted.with_rustnn_backend_hint(backend);
    }
    if options.has_device_hint {
        converted = converted.with_rustnn_device_hint(backend_device_from_c(options.device_hint)?);
    }
    if options.has_rustnn_options {
        converted = converted.with_rustnn_options(RustNNOptions {
            trtx: TrtxOptions {
                engine_caching: options.rustnn_options.trtx.engine_caching,
                runtime_cache: options.rustnn_options.trtx.runtime_cache,
                fail_on_cache_miss: options.rustnn_options.trtx.fail_on_cache_miss,
                cuda_graphs: options.rustnn_options.trtx.cuda_graphs,
            },
            ..RustNNOptions::default()
        });
    }
    Ok(converted)
}

unsafe fn input_slice<'a, T>(
    data: *const T,
    len: usize,
    name: &str,
) -> Result<&'a [T], RustnnStatus> {
    if len == 0 {
        Ok(&[])
    } else if data.is_null() {
        set_error(format!(
            "{name} must not be null when its length is non-zero"
        ));
        Err(RustnnStatus::NullPointer)
    } else {
        Ok(unsafe { slice::from_raw_parts(data, len) })
    }
}

unsafe fn input_str<'a>(value: *const c_char, name: &str) -> Result<&'a str, RustnnStatus> {
    let value = unsafe { required_ref(value, name)? };
    unsafe { CStr::from_ptr(value) }.to_str().map_err(|_| {
        set_error(format!("{name} is not valid UTF-8"));
        RustnnStatus::InvalidUtf8
    })
}

unsafe fn operator_options(
    options: *const RustnnOperatorOptions,
) -> Result<MLOperatorOptions, RustnnStatus> {
    if options.is_null() {
        return Ok(MLOperatorOptions::default());
    }
    let label = unsafe { (*options).label };
    let label = if label.is_null() {
        String::new()
    } else {
        unsafe { input_str(label, "options.label")? }.to_string()
    };
    Ok(MLOperatorOptions { label })
}

fn return_operand<E: std::fmt::Display>(
    result: Result<MLOperand, E>,
    output: &mut *mut RustnnOperand,
) -> RustnnStatus {
    match result {
        Ok(operand) => {
            *output = Box::into_raw(Box::new(RustnnOperand(operand)));
            RustnnStatus::Success
        }
        Err(error) => {
            set_error(error);
            RustnnStatus::Error
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustnn_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// Install rustnn's `pretty_env_logger` global logger if no logger is installed yet.
///
/// Set `RUST_LOG` before calling this function. Repeated calls are safe; calls made after another
/// global logger has already been installed have no effect.
#[unsafe(no_mangle)]
pub extern "C" fn rustnn_init_logger() {
    let _ = pretty_env_logger::try_init();
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_operand_descriptor_create(
    data_type: RustnnDataType,
    shape: *const u64,
    shape_len: usize,
    output: *mut *mut RustnnOperandDescriptor,
) -> RustnnStatus {
    ffi_call(|| {
        let output = match unsafe { required_mut(output, "output") } {
            Ok(output) => output,
            Err(status) => return status,
        };
        *output = ptr::null_mut();
        let shape = match unsafe { input_slice(shape, shape_len, "shape") } {
            Ok(shape) => shape,
            Err(status) => return status,
        };
        let descriptor = MLOperandDescriptor::new(data_type.into(), shape.to_vec());
        *output = Box::into_raw(Box::new(RustnnOperandDescriptor(descriptor)));
        RustnnStatus::Success
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_operand_descriptor_destroy(
    descriptor: *mut RustnnOperandDescriptor,
) {
    if !descriptor.is_null() {
        drop(unsafe { Box::from_raw(descriptor) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_tensor_descriptor_create(
    data_type: RustnnDataType,
    shape: *const u64,
    shape_len: usize,
    readable: bool,
    writable: bool,
    output: *mut *mut RustnnTensorDescriptor,
) -> RustnnStatus {
    ffi_call(|| {
        let output = match unsafe { required_mut(output, "output") } {
            Ok(output) => output,
            Err(status) => return status,
        };
        *output = ptr::null_mut();
        let shape = match unsafe { input_slice(shape, shape_len, "shape") } {
            Ok(shape) => shape,
            Err(status) => return status,
        };
        let mut descriptor = MLTensorDescriptor::new(data_type.into(), shape.to_vec());
        descriptor.set_readable(readable);
        descriptor.set_writable(writable);
        *output = Box::into_raw(Box::new(RustnnTensorDescriptor(descriptor)));
        RustnnStatus::Success
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_tensor_descriptor_destroy(descriptor: *mut RustnnTensorDescriptor) {
    if !descriptor.is_null() {
        drop(unsafe { Box::from_raw(descriptor) });
    }
}

/// Returns context options initialized to the same defaults as the Rust API.
#[unsafe(no_mangle)]
pub extern "C" fn rustnn_context_options_default() -> RustnnContextOptions {
    RustnnContextOptions::default()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_context_create(
    options: *const RustnnContextOptions,
    output: *mut *mut RustnnContext,
) -> RustnnStatus {
    ffi_call(|| {
        let output = match unsafe { required_mut(output, "output") } {
            Ok(output) => output,
            Err(status) => return status,
        };
        *output = ptr::null_mut();
        let options = if options.is_null() {
            MLContextOptions::new(MLPowerPreference::Default, false)
        } else {
            let options = unsafe { &*options };
            match context_options_from_c(options) {
                Ok(options) => options,
                Err(status) => return status,
            }
        };
        match MLContext::create(&options) {
            Ok(context) => {
                *output = Box::into_raw(Box::new(RustnnContext(context)));
                RustnnStatus::Success
            }
            Err(error) => {
                set_error(error);
                RustnnStatus::Error
            }
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_context_destroy(context: *mut RustnnContext) {
    if !context.is_null() {
        drop(unsafe { Box::from_raw(context) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_context_create_tensor(
    context: *mut RustnnContext,
    descriptor: *const RustnnTensorDescriptor,
    output: *mut *mut RustnnTensor,
) -> RustnnStatus {
    ffi_call(|| {
        let context = match unsafe { required_mut(context, "context") } {
            Ok(context) => context,
            Err(status) => return status,
        };
        let descriptor = match unsafe { required_ref(descriptor, "descriptor") } {
            Ok(descriptor) => descriptor,
            Err(status) => return status,
        };
        let output = match unsafe { required_mut(output, "output") } {
            Ok(output) => output,
            Err(status) => return status,
        };
        *output = ptr::null_mut();
        match context.0.create_tensor(&descriptor.0) {
            Ok(tensor) => {
                *output = Box::into_raw(Box::new(RustnnTensor(tensor)));
                RustnnStatus::Success
            }
            Err(error) => {
                set_error(error);
                RustnnStatus::Error
            }
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_tensor_destroy(tensor: *mut RustnnTensor) {
    if !tensor.is_null() {
        drop(unsafe { Box::from_raw(tensor) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_context_write_tensor(
    context: *mut RustnnContext,
    tensor: *const RustnnTensor,
    data: *const c_void,
    data_len: usize,
) -> RustnnStatus {
    ffi_call(|| {
        let context = match unsafe { required_mut(context, "context") } {
            Ok(context) => context,
            Err(status) => return status,
        };
        let tensor = match unsafe { required_ref(tensor, "tensor") } {
            Ok(tensor) => tensor,
            Err(status) => return status,
        };
        let data = match unsafe { input_slice(data.cast::<u8>(), data_len, "data") } {
            Ok(data) => data,
            Err(status) => return status,
        };
        match context.0.write_tensor(&tensor.0, data) {
            Ok(()) => RustnnStatus::Success,
            Err(error) => {
                set_error(error);
                RustnnStatus::Error
            }
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_context_read_tensor(
    context: *mut RustnnContext,
    tensor: *const RustnnTensor,
    data: *mut c_void,
    data_len: usize,
) -> RustnnStatus {
    ffi_call(|| {
        let context = match unsafe { required_mut(context, "context") } {
            Ok(context) => context,
            Err(status) => return status,
        };
        let tensor = match unsafe { required_ref(tensor, "tensor") } {
            Ok(tensor) => tensor,
            Err(status) => return status,
        };
        if data_len != 0 && data.is_null() {
            set_error("data must not be null when its length is non-zero");
            return RustnnStatus::NullPointer;
        }
        let data = if data_len == 0 {
            &mut []
        } else {
            unsafe { slice::from_raw_parts_mut(data.cast::<u8>(), data_len) }
        };
        match context.0.read_tensor(&tensor.0, data) {
            Ok(()) => RustnnStatus::Success,
            Err(error) => {
                set_error(error);
                RustnnStatus::Error
            }
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_create_uncompiled(
    output: *mut *mut RustnnGraphBuilder,
) -> RustnnStatus {
    ffi_call(|| {
        let output = match unsafe { required_mut(output, "output") } {
            Ok(output) => output,
            Err(status) => return status,
        };
        *output = Box::into_raw(Box::new(RustnnGraphBuilder(
            MLGraphBuilder::new_uncompiled(),
        )));
        RustnnStatus::Success
    })
}

#[unsafe(no_mangle)]
/// Create a runtime-backed builder that exclusively borrows `context`.
///
/// The context must remain alive and must not be used until this builder is
/// consumed by `rustnn_graph_builder_build` or destroyed.
pub unsafe extern "C" fn rustnn_graph_builder_create(
    context: *mut RustnnContext,
    output: *mut *mut RustnnGraphBuilder,
) -> RustnnStatus {
    ffi_call(|| {
        let context = match unsafe { required_mut(context, "context") } {
            Ok(context) => context,
            Err(status) => return status,
        };
        let output = match unsafe { required_mut(output, "output") } {
            Ok(output) => output,
            Err(status) => return status,
        };
        *output = ptr::null_mut();
        match MLGraphBuilder::new(&mut context.0) {
            Ok(builder) => {
                // The C API requires the context to outlive the builder and graph. The context is
                // heap allocated, so its address remains stable; the caller-visible lifetime is
                // enforced by the documented handle ownership rules.
                let builder = unsafe {
                    std::mem::transmute::<MLGraphBuilder<'_, '_>, MLGraphBuilder<'static, 'static>>(
                        builder,
                    )
                };
                *output = Box::into_raw(Box::new(RustnnGraphBuilder(builder)));
                RustnnStatus::Success
            }
            Err(error) => {
                set_error(error);
                RustnnStatus::Error
            }
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_destroy(builder: *mut RustnnGraphBuilder) {
    if !builder.is_null() {
        drop(unsafe { Box::from_raw(builder) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_operand_destroy(operand: *mut RustnnOperand) {
    if !operand.is_null() {
        drop(unsafe { Box::from_raw(operand) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_input(
    builder: *mut RustnnGraphBuilder,
    name: *const c_char,
    descriptor: *const RustnnOperandDescriptor,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    ffi_call(|| {
        let builder = match unsafe { required_mut(builder, "builder") } {
            Ok(builder) => builder,
            Err(status) => return status,
        };
        let name = match unsafe { input_str(name, "name") } {
            Ok(name) => name,
            Err(status) => return status,
        };
        let descriptor = match unsafe { required_ref(descriptor, "descriptor") } {
            Ok(descriptor) => descriptor,
            Err(status) => return status,
        };
        let output = match unsafe { required_mut(output, "output") } {
            Ok(output) => output,
            Err(status) => return status,
        };
        *output = ptr::null_mut();
        return_operand(builder.0.input(name, &descriptor.0), output)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_constant(
    builder: *mut RustnnGraphBuilder,
    descriptor: *const RustnnOperandDescriptor,
    data: *const c_void,
    data_len: usize,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    ffi_call(|| {
        let builder = match unsafe { required_mut(builder, "builder") } {
            Ok(builder) => builder,
            Err(status) => return status,
        };
        let descriptor = match unsafe { required_ref(descriptor, "descriptor") } {
            Ok(descriptor) => descriptor,
            Err(status) => return status,
        };
        let data = match unsafe { input_slice(data.cast::<u8>(), data_len, "data") } {
            Ok(data) => data,
            Err(status) => return status,
        };
        let output = match unsafe { required_mut(output, "output") } {
            Ok(output) => output,
            Err(status) => return status,
        };
        *output = ptr::null_mut();
        return_operand(
            builder.0.constant_from_bytes(&descriptor.0, data.to_vec()),
            output,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_unary(
    builder: *mut RustnnGraphBuilder,
    operation: RustnnUnaryOperation,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    ffi_call(|| {
        let builder = match unsafe { required_mut(builder, "builder") } {
            Ok(builder) => builder,
            Err(status) => return status,
        };
        let input = match unsafe { required_ref(input, "input") } {
            Ok(input) => input.0,
            Err(status) => return status,
        };
        let output = match unsafe { required_mut(output, "output") } {
            Ok(output) => output,
            Err(status) => return status,
        };
        *output = ptr::null_mut();
        let result = match operation {
            RustnnUnaryOperation::Abs => builder.0.abs(input),
            RustnnUnaryOperation::RoundEven => builder.0.round_even(input),
            RustnnUnaryOperation::Ceil => builder.0.ceil(input),
            RustnnUnaryOperation::Cos => builder.0.cos(input),
            RustnnUnaryOperation::Exp => builder.0.exp(input),
            RustnnUnaryOperation::Floor => builder.0.floor(input),
            RustnnUnaryOperation::Gelu => builder.0.gelu(input),
            RustnnUnaryOperation::Log => builder.0.log(input),
            RustnnUnaryOperation::Neg => builder.0.neg(input),
            RustnnUnaryOperation::Relu => builder.0.relu(input),
            RustnnUnaryOperation::Sigmoid => builder.0.sigmoid(input),
            RustnnUnaryOperation::Sin => builder.0.sin(input),
            RustnnUnaryOperation::Sqrt => builder.0.sqrt(input),
            RustnnUnaryOperation::Tan => builder.0.tan(input),
            RustnnUnaryOperation::Tanh => builder.0.tanh(input),
            RustnnUnaryOperation::Erf => builder.0.erf(input),
            RustnnUnaryOperation::Reciprocal => builder.0.reciprocal(input),
            RustnnUnaryOperation::Sign => builder.0.sign(input),
            RustnnUnaryOperation::LogicalNot => builder.0.logical_not(input),
            RustnnUnaryOperation::Identity => builder.0.identity(input),
            RustnnUnaryOperation::Softplus => builder.0.softplus(input),
            RustnnUnaryOperation::Softsign => builder.0.softsign(input),
            RustnnUnaryOperation::IsNan => builder.0.is_nan(input),
            RustnnUnaryOperation::IsInfinite => builder.0.is_infinite(input),
            RustnnUnaryOperation::Shape => builder.0.shape(input),
        };
        return_operand(result, output)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_binary(
    builder: *mut RustnnGraphBuilder,
    operation: RustnnBinaryOperation,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    ffi_call(|| {
        let builder = match unsafe { required_mut(builder, "builder") } {
            Ok(builder) => builder,
            Err(status) => return status,
        };
        let lhs = match unsafe { required_ref(lhs, "lhs") } {
            Ok(lhs) => lhs.0,
            Err(status) => return status,
        };
        let rhs = match unsafe { required_ref(rhs, "rhs") } {
            Ok(rhs) => rhs.0,
            Err(status) => return status,
        };
        let output = match unsafe { required_mut(output, "output") } {
            Ok(output) => output,
            Err(status) => return status,
        };
        *output = ptr::null_mut();
        let result = match operation {
            RustnnBinaryOperation::Add => builder.0.add(lhs, rhs),
            RustnnBinaryOperation::Sub => builder.0.sub(lhs, rhs),
            RustnnBinaryOperation::Mul => builder.0.mul(lhs, rhs),
            RustnnBinaryOperation::Div => builder.0.div(lhs, rhs),
            RustnnBinaryOperation::Pow => builder.0.pow(lhs, rhs),
            RustnnBinaryOperation::Max => builder.0.max(lhs, rhs),
            RustnnBinaryOperation::Min => builder.0.min(lhs, rhs),
            RustnnBinaryOperation::Equal => builder.0.equal(lhs, rhs),
            RustnnBinaryOperation::Greater => builder.0.greater(lhs, rhs),
            RustnnBinaryOperation::GreaterOrEqual => builder.0.greater_or_equal(lhs, rhs),
            RustnnBinaryOperation::Lesser => builder.0.lesser(lhs, rhs),
            RustnnBinaryOperation::LesserOrEqual => builder.0.lesser_or_equal(lhs, rhs),
            RustnnBinaryOperation::NotEqual => builder.0.not_equal(lhs, rhs),
            RustnnBinaryOperation::LogicalAnd => builder.0.logical_and(lhs, rhs),
            RustnnBinaryOperation::LogicalOr => builder.0.logical_or(lhs, rhs),
            RustnnBinaryOperation::LogicalXor => builder.0.logical_xor(lhs, rhs),
            RustnnBinaryOperation::Matmul => builder.0.matmul(lhs, rhs),
        };
        return_operand(result, output)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_unary_with_options(
    builder: *mut RustnnGraphBuilder,
    operation: RustnnUnaryOperation,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    ffi_call(|| {
        let builder = match unsafe { required_mut(builder, "builder") } {
            Ok(builder) => builder,
            Err(status) => return status,
        };
        let input = match unsafe { required_ref(input, "input") } {
            Ok(input) => input.0,
            Err(status) => return status,
        };
        let options = match unsafe { operator_options(options) } {
            Ok(options) => options,
            Err(status) => return status,
        };
        let output = match unsafe { required_mut(output, "output") } {
            Ok(output) => output,
            Err(status) => return status,
        };
        *output = ptr::null_mut();
        let result = match operation {
            RustnnUnaryOperation::Abs => builder.0.abs_with_options(input, options),
            RustnnUnaryOperation::RoundEven => builder.0.round_even_with_options(input, options),
            RustnnUnaryOperation::Ceil => builder.0.ceil_with_options(input, options),
            RustnnUnaryOperation::Cos => builder.0.cos_with_options(input, options),
            RustnnUnaryOperation::Exp => builder.0.exp_with_options(input, options),
            RustnnUnaryOperation::Floor => builder.0.floor_with_options(input, options),
            RustnnUnaryOperation::Gelu => builder.0.gelu_with_options(input, options),
            RustnnUnaryOperation::Log => builder.0.log_with_options(input, options),
            RustnnUnaryOperation::Neg => builder.0.neg_with_options(input, options),
            RustnnUnaryOperation::Relu => builder.0.relu_with_options(input, options),
            RustnnUnaryOperation::Sigmoid => builder.0.sigmoid_with_options(input, options),
            RustnnUnaryOperation::Sin => builder.0.sin_with_options(input, options),
            RustnnUnaryOperation::Sqrt => builder.0.sqrt_with_options(input, options),
            RustnnUnaryOperation::Tan => builder.0.tan_with_options(input, options),
            RustnnUnaryOperation::Tanh => builder.0.tanh_with_options(input, options),
            RustnnUnaryOperation::Erf => builder.0.erf_with_options(input, options),
            RustnnUnaryOperation::Reciprocal => builder.0.reciprocal_with_options(input, options),
            RustnnUnaryOperation::Sign => builder.0.sign_with_options(input, options),
            RustnnUnaryOperation::LogicalNot => builder.0.logical_not_with_options(input, options),
            RustnnUnaryOperation::Identity => builder.0.identity_with_options(input, options),
            RustnnUnaryOperation::Softplus => builder.0.softplus_with_options(input, options),
            RustnnUnaryOperation::Softsign => builder.0.softsign_with_options(input, options),
            RustnnUnaryOperation::IsNan => builder.0.is_nan_with_options(input, options),
            RustnnUnaryOperation::IsInfinite => builder.0.is_infinite_with_options(input, options),
            RustnnUnaryOperation::Shape => builder.0.shape_with_options(input, options),
        };
        return_operand(result, output)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_binary_with_options(
    builder: *mut RustnnGraphBuilder,
    operation: RustnnBinaryOperation,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    ffi_call(|| {
        let builder = match unsafe { required_mut(builder, "builder") } {
            Ok(builder) => builder,
            Err(status) => return status,
        };
        let lhs = match unsafe { required_ref(lhs, "lhs") } {
            Ok(lhs) => lhs.0,
            Err(status) => return status,
        };
        let rhs = match unsafe { required_ref(rhs, "rhs") } {
            Ok(rhs) => rhs.0,
            Err(status) => return status,
        };
        let options = match unsafe { operator_options(options) } {
            Ok(options) => options,
            Err(status) => return status,
        };
        let output = match unsafe { required_mut(output, "output") } {
            Ok(output) => output,
            Err(status) => return status,
        };
        *output = ptr::null_mut();
        let result = match operation {
            RustnnBinaryOperation::Add => builder.0.add_with_options(lhs, rhs, options),
            RustnnBinaryOperation::Sub => builder.0.sub_with_options(lhs, rhs, options),
            RustnnBinaryOperation::Mul => builder.0.mul_with_options(lhs, rhs, options),
            RustnnBinaryOperation::Div => builder.0.div_with_options(lhs, rhs, options),
            RustnnBinaryOperation::Pow => builder.0.pow_with_options(lhs, rhs, options),
            RustnnBinaryOperation::Max => builder.0.max_with_options(lhs, rhs, options),
            RustnnBinaryOperation::Min => builder.0.min_with_options(lhs, rhs, options),
            RustnnBinaryOperation::Equal => builder.0.equal_with_options(lhs, rhs, options),
            RustnnBinaryOperation::Greater => builder.0.greater_with_options(lhs, rhs, options),
            RustnnBinaryOperation::GreaterOrEqual => {
                builder.0.greater_or_equal_with_options(lhs, rhs, options)
            }
            RustnnBinaryOperation::Lesser => builder.0.lesser_with_options(lhs, rhs, options),
            RustnnBinaryOperation::LesserOrEqual => {
                builder.0.lesser_or_equal_with_options(lhs, rhs, options)
            }
            RustnnBinaryOperation::NotEqual => builder.0.not_equal_with_options(lhs, rhs, options),
            RustnnBinaryOperation::LogicalAnd => {
                builder.0.logical_and_with_options(lhs, rhs, options)
            }
            RustnnBinaryOperation::LogicalOr => {
                builder.0.logical_or_with_options(lhs, rhs, options)
            }
            RustnnBinaryOperation::LogicalXor => {
                builder.0.logical_xor_with_options(lhs, rhs, options)
            }
            RustnnBinaryOperation::Matmul => builder.0.matmul_with_options(lhs, rhs, options),
        };
        return_operand(result, output)
    })
}

// Named entry points mirror the corresponding public `MLGraphBuilder` methods.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_add(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_binary(builder, RustnnBinaryOperation::Add, lhs, rhs, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_subtract(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_binary(builder, RustnnBinaryOperation::Sub, lhs, rhs, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_multiply(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_binary(builder, RustnnBinaryOperation::Mul, lhs, rhs, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_divide(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_binary(builder, RustnnBinaryOperation::Div, lhs, rhs, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_power(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_binary(builder, RustnnBinaryOperation::Pow, lhs, rhs, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_maximum(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_binary(builder, RustnnBinaryOperation::Max, lhs, rhs, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_minimum(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_binary(builder, RustnnBinaryOperation::Min, lhs, rhs, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_equal(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_binary(builder, RustnnBinaryOperation::Equal, lhs, rhs, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_greater(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary(builder, RustnnBinaryOperation::Greater, lhs, rhs, output)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_greater_or_equal(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary(
            builder,
            RustnnBinaryOperation::GreaterOrEqual,
            lhs,
            rhs,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_lesser(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_binary(builder, RustnnBinaryOperation::Lesser, lhs, rhs, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_lesser_or_equal(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary(
            builder,
            RustnnBinaryOperation::LesserOrEqual,
            lhs,
            rhs,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_not_equal(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary(builder, RustnnBinaryOperation::NotEqual, lhs, rhs, output)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_logical_and(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary(builder, RustnnBinaryOperation::LogicalAnd, lhs, rhs, output)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_logical_or(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary(builder, RustnnBinaryOperation::LogicalOr, lhs, rhs, output)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_logical_xor(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary(builder, RustnnBinaryOperation::LogicalXor, lhs, rhs, output)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_matmul(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_binary(builder, RustnnBinaryOperation::Matmul, lhs, rhs, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_abs(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Abs, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_round_even(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::RoundEven, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_ceil(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Ceil, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_cos(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Cos, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_exp(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Exp, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_floor(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Floor, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_gelu(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Gelu, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_log(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Log, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_negate(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Neg, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_relu(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Relu, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_sigmoid(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Sigmoid, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_sin(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Sin, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_sqrt(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Sqrt, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_tan(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Tan, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_tanh(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Tanh, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_erf(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Erf, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_reciprocal(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Reciprocal, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_sign(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Sign, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_logical_not(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::LogicalNot, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_identity(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Identity, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_softplus(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Softplus, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_softsign(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Softsign, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_is_nan(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::IsNan, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_is_infinite(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::IsInfinite, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_shape(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe { rustnn_graph_builder_unary(builder, RustnnUnaryOperation::Shape, input, output) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_add_with_options(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary_with_options(
            builder,
            RustnnBinaryOperation::Add,
            lhs,
            rhs,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_subtract_with_options(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary_with_options(
            builder,
            RustnnBinaryOperation::Sub,
            lhs,
            rhs,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_multiply_with_options(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary_with_options(
            builder,
            RustnnBinaryOperation::Mul,
            lhs,
            rhs,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_divide_with_options(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary_with_options(
            builder,
            RustnnBinaryOperation::Div,
            lhs,
            rhs,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_power_with_options(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary_with_options(
            builder,
            RustnnBinaryOperation::Pow,
            lhs,
            rhs,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_maximum_with_options(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary_with_options(
            builder,
            RustnnBinaryOperation::Max,
            lhs,
            rhs,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_minimum_with_options(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary_with_options(
            builder,
            RustnnBinaryOperation::Min,
            lhs,
            rhs,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_equal_with_options(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary_with_options(
            builder,
            RustnnBinaryOperation::Equal,
            lhs,
            rhs,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_greater_with_options(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary_with_options(
            builder,
            RustnnBinaryOperation::Greater,
            lhs,
            rhs,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_greater_or_equal_with_options(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary_with_options(
            builder,
            RustnnBinaryOperation::GreaterOrEqual,
            lhs,
            rhs,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_lesser_with_options(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary_with_options(
            builder,
            RustnnBinaryOperation::Lesser,
            lhs,
            rhs,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_lesser_or_equal_with_options(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary_with_options(
            builder,
            RustnnBinaryOperation::LesserOrEqual,
            lhs,
            rhs,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_not_equal_with_options(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary_with_options(
            builder,
            RustnnBinaryOperation::NotEqual,
            lhs,
            rhs,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_logical_and_with_options(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary_with_options(
            builder,
            RustnnBinaryOperation::LogicalAnd,
            lhs,
            rhs,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_logical_or_with_options(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary_with_options(
            builder,
            RustnnBinaryOperation::LogicalOr,
            lhs,
            rhs,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_logical_xor_with_options(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary_with_options(
            builder,
            RustnnBinaryOperation::LogicalXor,
            lhs,
            rhs,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_matmul_with_options(
    builder: *mut RustnnGraphBuilder,
    lhs: *const RustnnOperand,
    rhs: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_binary_with_options(
            builder,
            RustnnBinaryOperation::Matmul,
            lhs,
            rhs,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_abs_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Abs,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_round_even_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::RoundEven,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_ceil_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Ceil,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_cos_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Cos,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_exp_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Exp,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_floor_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Floor,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_gelu_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Gelu,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_log_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Log,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_negate_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Neg,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_relu_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Relu,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_sigmoid_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Sigmoid,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_sin_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Sin,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_sqrt_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Sqrt,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_tan_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Tan,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_tanh_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Tanh,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_erf_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Erf,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_reciprocal_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Reciprocal,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_sign_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Sign,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_logical_not_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::LogicalNot,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_identity_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Identity,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_softplus_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Softplus,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_softsign_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Softsign,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_is_nan_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::IsNan,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_is_infinite_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::IsInfinite,
            input,
            options,
            output,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_shape_with_options(
    builder: *mut RustnnGraphBuilder,
    input: *const RustnnOperand,
    options: *const RustnnOperatorOptions,
    output: *mut *mut RustnnOperand,
) -> RustnnStatus {
    unsafe {
        rustnn_graph_builder_unary_with_options(
            builder,
            RustnnUnaryOperation::Shape,
            input,
            options,
            output,
        )
    }
}

/// Compile a runtime-backed builder and consume it, setting `builder` to null.
/// The originating context must outlive the returned graph.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_build(
    builder: *mut *mut RustnnGraphBuilder,
    outputs: *const RustnnNamedOperand,
    outputs_len: usize,
    graph: *mut *mut RustnnGraph,
) -> RustnnStatus {
    ffi_call(|| {
        let builder_slot = match unsafe { required_mut(builder, "builder") } {
            Ok(builder) => builder,
            Err(status) => return status,
        };
        if builder_slot.is_null() {
            set_error("builder must not be null");
            return RustnnStatus::NullPointer;
        }
        let graph = match unsafe { required_mut(graph, "graph") } {
            Ok(graph) => graph,
            Err(status) => return status,
        };
        *graph = ptr::null_mut();
        let mut owned_builder = unsafe { Box::from_raw(*builder_slot) };
        *builder_slot = ptr::null_mut();
        let outputs = match unsafe { input_slice(outputs, outputs_len, "outputs") } {
            Ok(outputs) => outputs,
            Err(status) => return status,
        };
        let mut named: MLNamedOperands<'_> = BTreeMap::new();
        for output in outputs {
            let name = match unsafe { input_str(output.name, "output name") } {
                Ok(name) => name,
                Err(status) => return status,
            };
            let operand = match unsafe { required_ref(output.operand, "output operand") } {
                Ok(operand) => operand.0,
                Err(status) => return status,
            };
            if named.insert(name, operand).is_some() {
                set_error(format!("duplicate output name: {name}"));
                return RustnnStatus::InvalidArgument;
            }
        }

        match owned_builder.0.build(&named) {
            Ok(built_graph) => {
                *graph = Box::into_raw(Box::new(RustnnGraph(built_graph)));
                RustnnStatus::Success
            }
            Err(error) => {
                set_error(error);
                RustnnStatus::Error
            }
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_destroy(graph: *mut RustnnGraph) {
    if !graph.is_null() {
        drop(unsafe { Box::from_raw(graph) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_context_dispatch(
    context: *mut RustnnContext,
    graph: *mut RustnnGraph,
    inputs: *const RustnnNamedTensor,
    inputs_len: usize,
    outputs: *const RustnnNamedTensor,
    outputs_len: usize,
) -> RustnnStatus {
    ffi_call(|| {
        let context = match unsafe { required_mut(context, "context") } {
            Ok(context) => context,
            Err(status) => return status,
        };
        let graph = match unsafe { required_mut(graph, "graph") } {
            Ok(graph) => graph,
            Err(status) => return status,
        };
        let inputs = match unsafe { input_slice(inputs, inputs_len, "inputs") } {
            Ok(inputs) => inputs,
            Err(status) => return status,
        };
        let outputs = match unsafe { input_slice(outputs, outputs_len, "outputs") } {
            Ok(outputs) => outputs,
            Err(status) => return status,
        };
        let mut named_inputs: MLNamedTensors<'_> = BTreeMap::new();
        for input in inputs {
            let name = match unsafe { input_str(input.name, "input name") } {
                Ok(name) => name,
                Err(status) => return status,
            };
            let tensor = match unsafe { required_ref(input.tensor, "input tensor") } {
                Ok(tensor) => &tensor.0,
                Err(status) => return status,
            };
            if named_inputs.insert(name, tensor).is_some() {
                set_error(format!("duplicate input name: {name}"));
                return RustnnStatus::InvalidArgument;
            }
        }
        let mut named_outputs: MLNamedTensors<'_> = BTreeMap::new();
        for output in outputs {
            let name = match unsafe { input_str(output.name, "output name") } {
                Ok(name) => name,
                Err(status) => return status,
            };
            let tensor = match unsafe { required_ref(output.tensor, "output tensor") } {
                Ok(tensor) => &tensor.0,
                Err(status) => return status,
            };
            if named_outputs.insert(name, tensor).is_some() {
                set_error(format!("duplicate output name: {name}"));
                return RustnnStatus::InvalidArgument;
            }
        }
        match context
            .0
            .dispatch(&mut graph.0, &named_inputs, &named_outputs)
        {
            Ok(()) => RustnnStatus::Success,
            Err(error) => {
                set_error(error);
                RustnnStatus::Error
            }
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_operand_shape(
    builder: *mut RustnnGraphBuilder,
    operand: *const RustnnOperand,
    dimensions: *mut u64,
    dimensions_capacity: usize,
    dimensions_len: *mut usize,
) -> RustnnStatus {
    ffi_call(|| {
        let builder = match unsafe { required_mut(builder, "builder") } {
            Ok(builder) => builder,
            Err(status) => return status,
        };
        let operand = match unsafe { required_ref(operand, "operand") } {
            Ok(operand) => operand.0,
            Err(status) => return status,
        };
        let dimensions_len = match unsafe { required_mut(dimensions_len, "dimensions_len") } {
            Ok(dimensions_len) => dimensions_len,
            Err(status) => return status,
        };
        let shape = match builder.0.rustnn_operand_shape(operand) {
            Ok(shape) => shape,
            Err(error) => {
                set_error(error);
                return RustnnStatus::Error;
            }
        };
        *dimensions_len = shape.len();
        if dimensions.is_null() && dimensions_capacity == 0 {
            return RustnnStatus::Success;
        }
        if dimensions_capacity < shape.len() {
            set_error(format!(
                "dimensions capacity {dimensions_capacity} is smaller than required {}",
                shape.len()
            ));
            return RustnnStatus::InvalidArgument;
        }
        if !shape.is_empty() {
            if dimensions.is_null() {
                set_error("dimensions must not be null when the operand rank is non-zero");
                return RustnnStatus::NullPointer;
            }
            unsafe { ptr::copy_nonoverlapping(shape.as_ptr(), dimensions, shape.len()) };
        }
        RustnnStatus::Success
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_operand_data_type(
    builder: *mut RustnnGraphBuilder,
    operand: *const RustnnOperand,
    data_type: *mut RustnnDataType,
) -> RustnnStatus {
    ffi_call(|| {
        let builder = match unsafe { required_mut(builder, "builder") } {
            Ok(builder) => builder,
            Err(status) => return status,
        };
        let operand = match unsafe { required_ref(operand, "operand") } {
            Ok(operand) => operand.0,
            Err(status) => return status,
        };
        let data_type = match unsafe { required_mut(data_type, "data_type") } {
            Ok(data_type) => data_type,
            Err(status) => return status,
        };
        match builder.0.rustnn_operand_data_type(operand) {
            Ok(value) => {
                *data_type = value.into();
                RustnnStatus::Success
            }
            Err(error) => {
                set_error(error);
                RustnnStatus::Error
            }
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_graph_builder_webnn_text(
    builder: *const RustnnGraphBuilder,
    outputs: *const RustnnNamedOperand,
    outputs_len: usize,
    text: *mut *mut c_char,
) -> RustnnStatus {
    ffi_call(|| {
        let builder = match unsafe { required_ref(builder, "builder") } {
            Ok(builder) => builder,
            Err(status) => return status,
        };
        let outputs = match unsafe { input_slice(outputs, outputs_len, "outputs") } {
            Ok(outputs) => outputs,
            Err(status) => return status,
        };
        let text = match unsafe { required_mut(text, "text") } {
            Ok(text) => text,
            Err(status) => return status,
        };
        *text = ptr::null_mut();
        let mut named: MLNamedOperands<'_> = BTreeMap::new();
        for output in outputs {
            let name = match unsafe { input_str(output.name, "output name") } {
                Ok(name) => name,
                Err(status) => return status,
            };
            let operand = match unsafe { required_ref(output.operand, "output operand") } {
                Ok(operand) => operand.0,
                Err(status) => return status,
            };
            if named.insert(name, operand).is_some() {
                set_error(format!("duplicate output name: {name}"));
                return RustnnStatus::InvalidArgument;
            }
        }
        let Some(serialized) = builder.0.rustnn_webnn_text_for_outputs(&named) else {
            set_error("the graph cannot be serialized with the supplied outputs");
            return RustnnStatus::Error;
        };
        match CString::new(serialized) {
            Ok(serialized) => {
                *text = serialized.into_raw();
                RustnnStatus::Success
            }
            Err(error) => {
                set_error(error);
                RustnnStatus::Error
            }
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustnn_string_destroy(value: *mut c_char) {
    if !value.is_null() {
        drop(unsafe { CString::from_raw(value) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_options_defaults_match_rust_defaults() {
        let options = rustnn_context_options_default();
        assert_eq!(options.backend_hint, RustnnBackend::Automatic);
        assert!(!options.has_device_hint);
        assert!(!options.has_rustnn_options);
        assert_eq!(options.rustnn_options.trtx, RustnnTrtxOptions::default());

        let converted = context_options_from_c(&options).unwrap();
        assert_eq!(converted.power_preference, MLPowerPreference::Default);
        assert!(!converted.accelerated);
        assert_eq!(converted.backend_hint, None);
        assert_eq!(converted.device_hint, None);
        assert_eq!(converted.rustnn_options, RustNNOptions::default());
    }

    #[test]
    fn context_options_convert_all_hints_and_backend_options() {
        let options = RustnnContextOptions {
            power_preference: RustnnPowerPreference::HighPerformance,
            accelerated: true,
            backend_hint: RustnnBackend::Trtx,
            has_device_hint: true,
            device_hint: RustnnBackendDevice {
                backend: RustnnBackend::Trtx,
                device_type: RustnnDeviceType::Gpu,
                device_index: 3,
            },
            has_rustnn_options: true,
            rustnn_options: RustnnOptions {
                trtx: RustnnTrtxOptions {
                    engine_caching: false,
                    runtime_cache: false,
                    fail_on_cache_miss: true,
                    cuda_graphs: false,
                },
            },
        };

        let converted = context_options_from_c(&options).unwrap();
        assert_eq!(
            converted.power_preference,
            MLPowerPreference::HighPerformance
        );
        assert!(converted.accelerated);
        assert_eq!(converted.backend_hint, Some(Backend::Trtx));
        assert_eq!(
            converted.device_hint,
            Some(BackendDevice::Trtx { cuda_device_idx: 3 })
        );
        assert_eq!(
            converted.rustnn_options.trtx,
            TrtxOptions {
                engine_caching: false,
                runtime_cache: false,
                fail_on_cache_miss: true,
                cuda_graphs: false,
            }
        );
    }

    #[test]
    fn context_options_reject_invalid_device_hint() {
        let options = RustnnContextOptions {
            has_device_hint: true,
            device_hint: RustnnBackendDevice {
                backend: RustnnBackend::Trtx,
                device_type: RustnnDeviceType::Cpu,
                device_index: 0,
            },
            ..RustnnContextOptions::default()
        };
        assert_eq!(
            context_options_from_c(&options),
            Err(RustnnStatus::InvalidArgument)
        );
    }

    #[cfg(feature = "trtx-runtime-mock")]
    #[test]
    fn context_create_honors_backend_hint() {
        unsafe {
            let mut options = rustnn_context_options_default();
            options.accelerated = true;
            options.backend_hint = RustnnBackend::Trtx;
            let mut context = ptr::null_mut();
            assert_eq!(
                rustnn_context_create(&options, &mut context),
                RustnnStatus::Success
            );
            assert_eq!((*context).0.rustnn_backend(), Backend::Trtx);
            rustnn_context_destroy(context);
        }
    }

    #[test]
    fn builds_and_serializes_an_arithmetic_graph() {
        unsafe {
            let mut builder = ptr::null_mut();
            assert_eq!(
                rustnn_graph_builder_create_uncompiled(&mut builder),
                RustnnStatus::Success
            );
            let shape = [2, 2];
            let mut descriptor = ptr::null_mut();
            assert_eq!(
                rustnn_operand_descriptor_create(
                    RustnnDataType::Float32,
                    shape.as_ptr(),
                    shape.len(),
                    &mut descriptor,
                ),
                RustnnStatus::Success
            );
            let mut lhs = ptr::null_mut();
            assert_eq!(
                rustnn_graph_builder_input(builder, c"lhs".as_ptr(), descriptor, &mut lhs),
                RustnnStatus::Success
            );
            let values = [1.0f32; 4];
            let mut rhs = ptr::null_mut();
            assert_eq!(
                rustnn_graph_builder_constant(
                    builder,
                    descriptor,
                    values.as_ptr().cast(),
                    std::mem::size_of_val(&values),
                    &mut rhs,
                ),
                RustnnStatus::Success
            );
            let mut sum = ptr::null_mut();
            let options = RustnnOperatorOptions {
                label: c"sum values".as_ptr(),
            };
            assert_eq!(
                rustnn_graph_builder_add_with_options(builder, lhs, rhs, &options, &mut sum),
                RustnnStatus::Success
            );
            let output = RustnnNamedOperand {
                name: c"sum".as_ptr(),
                operand: sum,
            };
            let mut text = ptr::null_mut();
            assert_eq!(
                rustnn_graph_builder_webnn_text(builder, &output, 1, &mut text),
                RustnnStatus::Success
            );
            let text_value = CStr::from_ptr(text).to_str().unwrap();
            assert!(text_value.contains("add(lhs"));
            assert!(text_value.contains("label=\"sum values\""));

            rustnn_string_destroy(text);
            rustnn_operand_destroy(sum);
            rustnn_operand_destroy(rhs);
            rustnn_operand_destroy(lhs);
            rustnn_operand_descriptor_destroy(descriptor);
            rustnn_graph_builder_destroy(builder);
        }
    }
}

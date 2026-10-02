//! Minimal CoreML execution bridge for macOS and iOS.
//! Loads a `.mlmodel`, compiles it if needed, and runs a zeroed inference
//! using CoreML's Objective-C API.

#![cfg(feature = "coreml-runtime")]

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::io::Write;
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;
use std::sync::mpsc;

use block::ConcreteBlock;
use objc::rc::autoreleasepool;
use objc::runtime::{Class, Object};
use objc::{class, msg_send, sel, sel_impl};

use super::coreml_dtype::{ArrayLayout, NativeType, boundary_error, from_native, to_native};
use crate::error::GraphError;
use crate::graph::{DataType, Dimension, OperandDescriptor, get_static_or_max_size};
use crate::runtime_checks::{RuntimeShapeState, TensorKind, validate_shape_data_length};

#[path = "coreml_tensor.rs"]
mod tensor_storage;
pub(crate) use tensor_storage::{CoremlTensorBinding, CoremlTensorStorage, run_coreml_tensors};

impl CompiledCoremlModel {
    pub(crate) fn compute_unit(&self) -> &'static str {
        self.compute_unit
    }
}
#[path = "coreml_load.rs"]
mod load;
use load::LoadTrace;
pub use load::{CoremlLoadDiagnostics, CoremlLoadFailure, CoremlLoadRoute};

// Link against the system frameworks we use.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[link(name = "CoreML", kind = "framework")]
unsafe extern "C" {}

// Objective-C++ exception firewall (src/executors/coreml_shim.mm).
// Return codes: 0 = success, 1 = NSError, 2 = NSException, 3 = C++ exception.
#[cfg(any(target_os = "macos", target_os = "ios"))]
unsafe extern "C" {
    fn rustnn_coreml_compile(
        model_url: *mut Object,
        out_url: *mut *mut Object,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn rustnn_coreml_load(
        compiled_url: *mut Object,
        configuration: *mut Object,
        out_model: *mut *mut Object,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn rustnn_coreml_predict(
        model: *mut Object,
        features: *mut Object,
        out_provider: *mut *mut Object,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
}

// Shims to check compilation on Linux
/// Always-failing stand-in for targets without the native CoreML shim.
///
/// # Safety
/// This stand-in does not dereference its pointer arguments. Its unsafe signature
/// mirrors the native shim; no output pointers are initialized on failure.
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub unsafe extern "C" fn rustnn_coreml_compile(
    _model_url: *mut Object,
    _out_url: *mut *mut Object,
    _error: *mut c_char,
    _error_length: usize,
) -> i32 {
    1
}
/// Always-failing stand-in for targets without the native CoreML shim.
///
/// # Safety
/// This stand-in does not dereference its pointer arguments. Its unsafe signature
/// mirrors the native shim; no output pointers are initialized on failure.
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub unsafe extern "C" fn rustnn_coreml_load(
    _compiled_url: *mut Object,
    _configuration: *mut Object,
    _out_model: *mut *mut Object,
    _error: *mut c_char,
    _error_length: usize,
) -> i32 {
    1
}
/// Always-failing stand-in for targets without the native CoreML shim.
///
/// # Safety
/// This stand-in does not dereference its pointer arguments. Its unsafe signature
/// mirrors the native shim; no output pointers are initialized on failure.
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub unsafe extern "C" fn rustnn_coreml_predict(
    _model: *mut Object,
    _features: *mut Object,
    _out_provider: *mut *mut Object,
    _error: *mut c_char,
    _error_length: usize,
) -> i32 {
    1
}

/// Releases an owned (+1) Objective-C object when dropped (including on early
/// `return`/`?`), balancing any +1 ownership we hold: the retain transferred by
/// the shim's `__bridge_retained` casts, or objects we created via
/// `new`/`alloc`+`init`.
struct ReleaseOnDrop(*mut Object);

impl ReleaseOnDrop {
    /// Take the pointer back out without releasing, handing ownership (+1)
    /// to the caller.
    fn into_inner(self) -> *mut Object {
        let ptr = self.0;
        std::mem::forget(self);
        ptr
    }
}

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

fn shim_error_to_string(buffer: &[u8]) -> String {
    let end = buffer
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(buffer.len());
    String::from_utf8_lossy(&buffer[..end]).into_owned()
}

/// Input tensor for the one-shot CoreML executors.
///
/// Values are numerically converted to the model's native input type. This f32
/// convenience API cannot represent every integer exactly; use typed
/// [`crate::mlcontext::MLContext`] tensors when exact integer I/O is required.
#[derive(Debug, Clone)]
pub struct CoremlInput {
    /// Model input name.
    pub name: String,
    /// Shape of the data.
    pub shape: Vec<usize>,
    /// Values as float32.
    pub data: Vec<f32>,
}

/// One output of a CoreML run.
#[derive(Debug, Clone)]
pub struct CoremlOutput {
    /// Model output name.
    pub name: String,
    /// Shape reported by CoreML.
    pub shape: Vec<i64>,
    /// CoreML `MLMultiArrayDataType` code of the original output.
    pub data_type_code: i64,
    /// Values numerically converted to float32. Large integers can lose
    /// precision here; typed MLTensor dispatch retains the declared host dtype.
    pub data: Vec<f32>, // Output data converted to f32 for consistency
}

/// Result of running a model on one compute-unit configuration.
#[derive(Debug, Clone)]
pub struct CoremlRunAttempt {
    /// Compute units tried, for example `"all"` or `"cpuOnly"`.
    pub compute_unit: &'static str,
    /// Outputs, or the CoreML error message.
    pub result: Result<Vec<CoremlOutput>, String>,
}

/// Compile and run a CoreML model once with zero-filled inputs on each compute-unit configuration.
pub fn run_coreml_zeroed(
    model_bytes: &[u8],
    inputs: &HashMap<String, OperandDescriptor>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    run_coreml_zeroed_cached(model_bytes, inputs, None)
}

/// [`run_coreml_zeroed`] that stores or reuses the compiled `.mlmodelc` at `compiled_path`.
pub fn run_coreml_zeroed_cached(
    model_bytes: &[u8],
    inputs: &HashMap<String, OperandDescriptor>,
    compiled_path: Option<&Path>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    run_coreml_zeroed_cached_with_weights(model_bytes, None, inputs, compiled_path)
}

/// Run CoreML inference with zeroed inputs and optional weight file
pub fn run_coreml_zeroed_cached_with_weights(
    model_bytes: &[u8],
    weights_data: Option<&[u8]>,
    inputs: &HashMap<String, OperandDescriptor>,
    compiled_path: Option<&Path>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    autoreleasepool(|| {
        run_impl_zeroed_with_weights(model_bytes, weights_data, inputs, compiled_path)
    })
}

/// Run CoreML inference with actual input data
pub fn run_coreml_with_inputs(
    model_bytes: &[u8],
    inputs: Vec<CoremlInput>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    run_coreml_with_inputs_with_weights(model_bytes, None, inputs)
}

/// Run CoreML inference with actual input data and optional weight file
pub fn run_coreml_with_inputs_with_weights(
    model_bytes: &[u8],
    weights_data: Option<&[u8]>,
    inputs: Vec<CoremlInput>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    autoreleasepool(|| {
        run_impl_with_inputs_with_weights(model_bytes, weights_data, inputs, None, None, None)
    })
}

/// Run CoreML inference with actual input data and model caching
pub fn run_coreml_with_inputs_cached(
    model_bytes: &[u8],
    inputs: Vec<CoremlInput>,
    cache_path: Option<&Path>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    autoreleasepool(|| {
        run_impl_with_inputs_with_weights(model_bytes, None, inputs, cache_path, None, None)
    })
}

/// Run CoreML inference with runtime descriptor checks for dynamic dimensions.
/// Invalid inputs or compilation failures fail the call; output validation
/// failures are reported in the corresponding compute-policy attempt.
pub fn run_coreml_with_inputs_checked(
    model_bytes: &[u8],
    inputs: Vec<CoremlInput>,
    input_descriptors: &HashMap<String, OperandDescriptor>,
    output_descriptors: &HashMap<String, OperandDescriptor>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    autoreleasepool(|| {
        run_impl_with_inputs_with_weights(
            model_bytes,
            None,
            inputs,
            None,
            Some(input_descriptors),
            Some(output_descriptors),
        )
    })
}

// ---------------------------------------------------------------------------
// Byte-oriented execution path for the unified WebNN IDL API (src/backends/coreml.rs).
//
// Unlike the f32-centric `CoremlInput`/`CoremlOutput` helpers above, this path
// works in raw bytes keyed by feature name plus an `OperandDescriptor`, mirroring
// the ONNX Runtime backend (`src/backends/ort.rs`). The model is compiled and
// loaded once (`compile_model`) and reused across dispatches (`run_coreml_bytes`).
// ---------------------------------------------------------------------------

/// A CoreML model that has been compiled and loaded once, ready for repeated dispatch.
///
/// Owns a retained `MLModel` and the in-memory CoreML asset backing it. All are
/// released when the value is dropped. This type is intentionally not `Send`/`Sync`:
/// `MLGraph`/`MLContext` are single-threaded, matching CoreML's usage model.
pub(crate) struct CompiledCoremlModel {
    /// Retained `MLModel` Objective-C object.
    model: *mut Object,
    /// Compute unit the model was successfully loaded with (diagnostic only).
    compute_unit: &'static str,
    diagnostics: CoremlLoadDiagnostics,
    backing: CoremlModelBacking,
}

enum CoremlModelBacking {
    InMemory {
        /// Retained `MLModelAsset`. CoreML may refer to it after loading.
        asset: *mut Object,
        /// Retained model specification data backing `asset`.
        specification_data: *mut Object,
        /// Retained external weights data backing `asset`, if any.
        weights_data: Option<*mut Object>,
    },
    OnDisk {
        compiled_dir: PathBuf,
        /// Owns the temporary `.mlmodel`/`.mlpackage` source; removed on drop. Held
        /// purely as a drop guard (never read directly).
        #[allow(dead_code)]
        temp_model: Option<TempModelSource>,
    },
}

impl std::fmt::Debug for CompiledCoremlModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledCoremlModel")
            .field("compute_unit", &self.compute_unit)
            .field("diagnostics", &self.diagnostics)
            .finish()
    }
}

impl Drop for CompiledCoremlModel {
    fn drop(&mut self) {
        if !self.model.is_null() {
            unsafe {
                let _: () = msg_send![self.model, release];
            }
        }
        match &self.backing {
            CoremlModelBacking::InMemory {
                asset,
                specification_data,
                weights_data,
            } => unsafe {
                let _: () = msg_send![*asset, release];
                let _: () = msg_send![*specification_data, release];
                if let Some(weights_data) = weights_data {
                    let _: () = msg_send![*weights_data, release];
                }
            },
            CoremlModelBacking::OnDisk { compiled_dir, .. } => {
                // The compiled `.mlmodelc` is produced by CoreML (not a tempfile handle),
                // so remove it explicitly. `temp_model`, if any, cleans itself up on drop.
                let _ = std::fs::remove_dir_all(compiled_dir);
            }
        }
    }
}

/// Raw-byte input for [`run_coreml_bytes`]: a tensor's bytes plus its descriptor.
pub(crate) struct CoremlByteInput<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) descriptor: &'a OperandDescriptor,
}

impl CompiledCoremlModel {
    pub(crate) fn load_diagnostics(&self) -> &CoremlLoadDiagnostics {
        &self.diagnostics
    }
}

/// Load a CoreML model directly from protobuf bytes and retain it for repeated
/// dispatch, falling back to CPU-only if the preferred compute units fail.
pub(crate) fn compile_model(
    model_bytes: Vec<u8>,
    weights_data: Option<Vec<u8>>,
    device_type: crate::backend_selection::DeviceType,
    use_in_memory_asset: bool,
) -> Result<CompiledCoremlModel, GraphError> {
    // Owned so the in-memory path can hand the buffers to NSData without
    // copying (weight blobs reach hundreds of MB); the Arcs keep them valid
    // for the URL fallback below even after the NSData objects are released.
    let model_bytes = Arc::new(model_bytes);
    let weights_data = weights_data.map(Arc::new);
    let mut trace = LoadTrace::new(device_type);
    trace.routes(use_in_memory_asset, |trace, route| match route {
        CoremlLoadRoute::InMemoryAsset => {
            compile_model_from_asset(&model_bytes, weights_data.as_ref(), trace)
        }
        CoremlLoadRoute::CompiledUrl => compile_model_from_url(
            &model_bytes,
            weights_data.as_deref().map(Vec::as_slice),
            trace,
        ),
    })
}

fn compile_model_from_asset(
    model_bytes: &Arc<Vec<u8>>,
    weights_data: Option<&Arc<Vec<u8>>>,
    trace: &mut LoadTrace,
) -> Result<CompiledCoremlModel, GraphError> {
    let route = CoremlLoadRoute::InMemoryAsset;
    autoreleasepool(|| unsafe {
        let (asset, specification_data, retained_weights_data) = trace.prepare(route, || {
            create_in_memory_model_asset(model_bytes, weights_data)
        })?;
        let loaded = trace.policies(route, |code| {
            let config: *mut Object = msg_send![class!(MLModelConfiguration), new];
            let _config_guard = ReleaseOnDrop(config);
            let () = msg_send![config, setComputeUnits: code];
            load_model_asset(asset, config)
        });
        if let Ok((model, name)) = loaded {
            return Ok(CompiledCoremlModel {
                model,
                compute_unit: name,
                diagnostics: trace.finish(route, name),
                backing: CoremlModelBacking::InMemory {
                    asset,
                    specification_data,
                    weights_data: retained_weights_data,
                },
            });
        }

        let _: () = msg_send![asset, release];
        let _: () = msg_send![specification_data, release];
        if let Some(weights_data) = retained_weights_data {
            let _: () = msg_send![weights_data, release];
        }
        loaded.map(|_| unreachable!("successful loads return above"))
    })
}

fn compile_model_from_url(
    model_bytes: &[u8],
    weights_data: Option<&[u8]>,
    trace: &mut LoadTrace,
) -> Result<CompiledCoremlModel, GraphError> {
    let route = CoremlLoadRoute::CompiledUrl;
    autoreleasepool(|| unsafe {
        let (compiled_url, compiled_dir, temp_model) = trace.prepare(route, || {
            prepare_compiled_model_with_weights(model_bytes, weights_data, None)
        })?;
        // Owned (+1) by us; released when this function returns on any path.
        let _compiled_url_guard = ReleaseOnDrop(compiled_url);
        let loaded = trace.policies(route, |code| {
            let config: *mut Object = msg_send![class!(MLModelConfiguration), new];
            let _config_guard = ReleaseOnDrop(config);
            let () = msg_send![config, setComputeUnits: code];
            let mut model: *mut Object = ptr::null_mut();
            let mut error = [0u8; 1024];
            let status = rustnn_coreml_load(
                compiled_url,
                config,
                &mut model,
                error.as_mut_ptr().cast(),
                error.len(),
            );
            if status != 0 || model.is_null() {
                return Err(format!(
                    "MLModel load failed: {}",
                    shim_error_to_string(&error)
                ));
            }
            Ok(model)
        });
        if let Ok((model, name)) = loaded {
            // The shim returns an owned (+1) model; `CompiledCoremlModel`'s
            // Drop releases it.
            return Ok(CompiledCoremlModel {
                model,
                compute_unit: name,
                diagnostics: trace.finish(route, name),
                backing: CoremlModelBacking::OnDisk {
                    compiled_dir,
                    temp_model,
                },
            });
        }

        let _ = std::fs::remove_dir_all(&compiled_dir);
        // `temp_model` removes its temp path when dropped at end of scope.
        drop(temp_model);
        loaded.map(|_| unreachable!("successful loads return above"))
    })
}

/// Wrap an `Arc`-owned buffer in an `NSData` WITHOUT copying. The NSData's
/// deallocator drops an `Arc` clone, so the bytes stay valid for as long as
/// EITHER the caller's `Arc` or the `NSData` lives — however long CoreML
/// internally retains the latter. Returns an owned (+1) object.
unsafe fn nsdata_from_arc(buffer: &Arc<Vec<u8>>) -> Result<*mut Object, GraphError> {
    let bytes = buffer.as_ptr() as *mut c_void;
    let length = buffer.len();
    let raw = Arc::into_raw(Arc::clone(buffer)) as usize;
    // NSData calls the deallocator exactly once, when its storage is freed.
    let deallocator = ConcreteBlock::new(move |_bytes: *mut c_void, _length: usize| {
        drop(unsafe { Arc::from_raw(raw as *const Vec<u8>) });
    })
    .copy();
    let data: *mut Object = msg_send![class!(NSData), alloc];
    let data: *mut Object = msg_send![data,
        initWithBytesNoCopy: bytes
        length: length
        deallocator: &*deallocator];
    if data.is_null() {
        // Never observed in practice; leak the Arc clone rather than risk a
        // double-free if the failed init already consumed the deallocator.
        return Err(GraphError::CoremlRuntimeFailed {
            reason: "failed to create NSData for CoreML model bytes".to_string(),
        });
    }
    Ok(data)
}

/// Create an `MLModelAsset` whose specification and optional external weight blob
/// are entirely memory-backed (zero-copy views over the given buffers). The
/// returned Objective-C objects are retained and must remain alive for at least
/// as long as the loaded `MLModel`.
unsafe fn create_in_memory_model_asset(
    model_bytes: &Arc<Vec<u8>>,
    weights: Option<&Arc<Vec<u8>>>,
) -> Result<(*mut Object, *mut Object, Option<*mut Object>), GraphError> {
    let Some(asset_class) = Class::get("MLModelAsset") else {
        return Err(GraphError::CoremlRuntimeFailed {
            reason: "in-memory CoreML model loading requires macOS 15 or newer".to_string(),
        });
    };

    let specification_data: *mut Object = unsafe { nsdata_from_arc(model_bytes)? };

    let mut error: *mut Object = ptr::null_mut();
    let (asset, weights_data): (*mut Object, Option<*mut Object>) = match weights {
        Some(weights) => {
            let weights_data: *mut Object = match unsafe { nsdata_from_arc(weights) } {
                Ok(data) => data,
                Err(err) => {
                    let _: () = msg_send![specification_data, release];
                    return Err(err);
                }
            };

            // BlobFileValue stores `@model_path/weights/weights.bin`. For an
            // in-memory asset CoreML expects the path relative to `@model_path`.
            let relative_path = unsafe { nsstring_from_str("weights/weights.bin")? };
            let blob_url: *mut Object = msg_send![class!(NSURL), fileURLWithPath: relative_path];
            let mapping: *mut Object = msg_send![class!(NSDictionary),
                dictionaryWithObject: weights_data forKey: blob_url];
            let asset: *mut Object = msg_send![asset_class,
                modelAssetWithSpecificationData: specification_data
                blobMapping: mapping
                error: &mut error];
            (asset, Some(weights_data))
        }
        None => {
            let asset: *mut Object = msg_send![asset_class,
                modelAssetWithSpecificationData: specification_data
                error: &mut error];
            (asset, None)
        }
    };

    if asset.is_null() {
        let reason = unsafe { ns_error_to_string(error, "MLModelAsset creation failed") };
        let _: () = msg_send![specification_data, release];
        if let Some(weights_data) = weights_data {
            let _: () = msg_send![weights_data, release];
        }
        return Err(GraphError::CoremlRuntimeFailed { reason });
    }
    let _: *mut Object = msg_send![asset, retain];
    Ok((asset, specification_data, weights_data))
}

/// Bridge CoreML's completion-handler API to this backend's synchronous graph
/// compilation contract. The callback retains the model before its autorelease
/// scope ends and transfers that ownership to the caller.
unsafe fn load_model_asset(
    asset: *mut Object,
    configuration: *mut Object,
) -> Result<*mut Object, String> {
    let (sender, receiver) = mpsc::sync_channel(1);
    let completion = ConcreteBlock::new(move |model: *mut Object, error: *mut Object| {
        let result = if model.is_null() {
            Err(unsafe { ns_error_to_string(error, "MLModelAsset load failed") })
        } else {
            unsafe {
                let _: *mut Object = msg_send![model, retain];
            }
            Ok(model as usize)
        };
        let _ = sender.send(result);
    })
    .copy();

    let (): () = msg_send![class!(MLModel),
        loadModelAsset: asset
        configuration: configuration
        completionHandler: &*completion];

    receiver
        .recv()
        .map_err(|_| "MLModelAsset load callback was dropped".to_string())?
        .map(|model| model as *mut Object)
}

/// Run a compiled CoreML model with raw-byte inputs, returning raw-byte outputs by name.
pub(crate) fn run_coreml_bytes(
    model: &CompiledCoremlModel,
    inputs: &HashMap<String, CoremlByteInput<'_>>,
    output_descriptors: &HashMap<String, OperandDescriptor>,
) -> Result<HashMap<String, Vec<u8>>, GraphError> {
    autoreleasepool(|| unsafe {
        let dict: *mut Object = msg_send![class!(NSMutableDictionary), dictionary];

        // Query the model's declared input types so the MLMultiArray we build matches
        // exactly what CoreML expects. Using our own dtype codes here is unsafe: an
        // array created with a code CoreML does not recognize (e.g. a bare `16` for
        // Float16) is treated as Float32, so CoreML reads past our 2-bytes-per-element
        // buffer -- garbage for small tensors, an out-of-bounds crash for large ones.
        let model_description: *mut Object = msg_send![model.model, modelDescription];
        let input_descs: *mut Object = msg_send![model_description, inputDescriptionsByName];

        for (name, input) in inputs {
            let key = nsstring_from_str(name)?;
            let mut shape_i64: Vec<i64> = input
                .descriptor
                .static_or_max_shape()
                .iter()
                .map(|&d| i64::from(d))
                .collect();
            if shape_i64.is_empty() {
                // Scalars are represented as a single-element 1-D array.
                shape_i64.push(1);
            }

            // Prefer the model's own data type code; fall back to our mapping only when
            // the model exposes no constraint for this input.
            let code = model_input_dtype_code(input_descs, key)
                .unwrap_or_else(|| map_dtype(input.descriptor.data_type));
            let array = create_multi_array(&shape_i64, code)?;
            fill_multiarray_from_bytes(array, input.data, input.descriptor.data_type, code)?;
            let feature_value: *mut Object =
                msg_send![class!(MLFeatureValue), featureValueWithMultiArray: array];
            let () = msg_send![dict, setObject: feature_value forKey: key];
        }

        let mut create_error: *mut Object = ptr::null_mut();
        let provider_alloc: *mut Object = msg_send![class!(MLDictionaryFeatureProvider), alloc];
        let provider: *mut Object =
            msg_send![provider_alloc, initWithDictionary: dict error: &mut create_error];
        if provider.is_null() {
            return Err(GraphError::CoremlRuntimeFailed {
                reason: ns_error_to_string(create_error, "MLDictionaryFeatureProvider init failed"),
            });
        }
        let _provider_guard = ReleaseOnDrop(provider);

        let mut output_provider: *mut Object = ptr::null_mut();
        let mut error = [0u8; 1024];
        let status = rustnn_coreml_predict(
            model.model,
            provider,
            &mut output_provider,
            error.as_mut_ptr().cast(),
            error.len(),
        );
        if status != 0 || output_provider.is_null() {
            return Err(GraphError::CoremlRuntimeFailed {
                reason: format!("prediction failed: {}", shim_error_to_string(&error)),
            });
        }
        // `rustnn_coreml_predict` (coreml_shim.mm) hands back `output_provider`
        // already retained (`__bridge_retained`) -- we own this reference and
        // must release it once we're done extracting outputs.
        let _output_provider_guard = ReleaseOnDrop(output_provider);

        let mut result = HashMap::with_capacity(output_descriptors.len());
        for (name, descriptor) in output_descriptors {
            let key = nsstring_from_str(name)?;
            let value: *mut Object = msg_send![output_provider, featureValueForName: key];
            if value.is_null() {
                return Err(GraphError::CoremlRuntimeFailed {
                    reason: format!("model did not produce output `{name}`"),
                });
            }
            let array: *mut Object = msg_send![value, multiArrayValue];
            if array.is_null() {
                return Err(GraphError::CoremlRuntimeFailed {
                    reason: format!("output `{name}` is not a MLMultiArray"),
                });
            }
            let bytes = extract_multiarray_bytes(array, descriptor)?;
            result.insert(name.clone(), bytes);
        }
        Ok(result)
    })
}

/// Read the actual allocation type and validate layout metadata before touching
/// its storage. CoreML strides are element offsets, not byte offsets.
unsafe fn multiarray_storage(
    array: *mut Object,
) -> Result<(NativeType, ArrayLayout, *mut u8), GraphError> {
    let code: i64 = msg_send![array, dataType];
    let kind = NativeType::from_code(code)?;
    let count: isize = msg_send![array, count];
    let count = usize::try_from(count)
        .map_err(|_| boundary_error("negative MLMultiArray element count"))?;
    let shape: *mut Object = msg_send![array, shape];
    let shape = unsafe { nsarray_to_i64_vec(shape)? };
    let strides: *mut Object = msg_send![array, strides];
    let strides = unsafe { nsarray_to_i64_vec(strides)? };
    let layout = ArrayLayout::new(&shape, &strides, count, kind.element_size())?;
    let data: *mut c_void = msg_send![array, dataPointer];
    if data.is_null() && count != 0 {
        return Err(boundary_error("MLMultiArray has no backing storage"));
    }
    Ok((kind, layout, data.cast()))
}

/// Copy a packed C-order host buffer into native element strides.
unsafe fn write_array_storage(
    data: *mut u8,
    layout: &ArrayLayout,
    element_size: usize,
    bytes: &[u8],
) -> Result<(), GraphError> {
    if bytes.len() != layout.byte_length {
        return Err(boundary_error(format!(
            "MLMultiArray input byte length mismatch: expected {}, got {}",
            layout.byte_length,
            bytes.len()
        )));
    }
    if layout.count == 0 {
        return Ok(());
    }
    if layout.contiguous {
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len()) };
    } else {
        for index in 0..layout.count {
            unsafe {
                ptr::copy_nonoverlapping(
                    bytes.as_ptr().add(index * element_size),
                    data.add(layout.byte_offset(index)),
                    element_size,
                );
            }
        }
    }
    Ok(())
}

/// Gather every native type with the same checked, element-stride-aware path.
unsafe fn read_array_storage(
    data: *const u8,
    layout: &ArrayLayout,
    element_size: usize,
) -> Vec<u8> {
    if layout.count == 0 {
        return vec![];
    }
    if layout.contiguous {
        return unsafe { std::slice::from_raw_parts(data, layout.byte_length) }.to_vec();
    }
    let mut bytes = Vec::with_capacity(layout.byte_length);
    for index in 0..layout.count {
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(data.add(layout.byte_offset(index)), element_size)
        });
    }
    bytes
}

/// Fill native storage by numeric type, never by element width alone.
unsafe fn fill_multiarray_from_bytes(
    array: *mut Object,
    src: &[u8],
    dtype: DataType,
    array_code: i32,
) -> Result<(), GraphError> {
    let (kind, layout, data) = unsafe { multiarray_storage(array)? };
    if kind != NativeType::from_code(i64::from(array_code))? {
        return Err(boundary_error(
            "MLMultiArray allocation data type differs from its requested type",
        ));
    }
    // Matching types retain all integer bits/NaN payloads without an extra
    // whole-tensor allocation. Promoted types require numeric conversion.
    if kind.matches(dtype) {
        unsafe { write_array_storage(data, &layout, kind.element_size(), src) }
    } else {
        let bytes = to_native(src, dtype, kind, layout.count)?;
        unsafe { write_array_storage(data, &layout, kind.element_size(), &bytes) }
    }
}

unsafe fn extract_multiarray_bytes(
    array: *mut Object,
    descriptor: &OperandDescriptor,
) -> Result<Vec<u8>, GraphError> {
    let (kind, layout, data) = unsafe { multiarray_storage(array)? };
    let bytes = unsafe { read_array_storage(data, &layout, kind.element_size()) };
    if kind.matches(descriptor.data_type) {
        return Ok(bytes);
    }
    from_native(&bytes, kind, descriptor.data_type, layout.count)
}

#[allow(dead_code)]
fn run_impl_zeroed(
    model_bytes: &[u8],
    inputs: &HashMap<String, OperandDescriptor>,
    compiled_path: Option<&Path>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    run_impl_zeroed_with_weights(model_bytes, None, inputs, compiled_path)
}

fn run_impl_zeroed_with_weights(
    model_bytes: &[u8],
    weights_data: Option<&[u8]>,
    inputs: &HashMap<String, OperandDescriptor>,
    compiled_path: Option<&Path>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    unsafe {
        let (compiled_url, compiled_path_buf, temp_mlmodel) =
            prepare_compiled_model_with_weights(model_bytes, weights_data, compiled_path)?;
        // Owned (+1) by us; released when this function returns on any path.
        let _compiled_url_guard = ReleaseOnDrop(compiled_url);

        // Try only Neural Engine + GPU (best performance on Apple Silicon)
        // Fallback to ALL if that fails
        let targets = [
            (3i64, "CPU_AND_NE"), // CPU + Neural Engine (best for Apple Silicon)
            (2i64, "ALL"),        // All available compute units
            (0i64, "CPU_ONLY"),   // Guaranteed-to-load last resort
        ];
        let mut attempts = Vec::new();

        for (code, name) in targets {
            let config: *mut Object = msg_send![class!(MLModelConfiguration), new];
            let _config_guard = ReleaseOnDrop(config);
            let () = msg_send![config, setComputeUnits: code];
            let mut model: *mut Object = ptr::null_mut();
            let mut error = [0u8; 1024];
            let status = rustnn_coreml_load(
                compiled_url,
                config,
                &mut model,
                error.as_mut_ptr().cast(),
                error.len(),
            );
            if status != 0 || model.is_null() {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(format!(
                        "MLModel load failed: {}",
                        shim_error_to_string(&error)
                    )),
                });
                continue;
            }
            // Owned (+1) by us for this attempt; released at end of iteration.
            let _model_guard = ReleaseOnDrop(model);
            let model_description: *mut Object = msg_send![model, modelDescription];
            let input_descs: *mut Object = msg_send![model_description, inputDescriptionsByName];

            let dict: *mut Object = msg_send![class!(NSMutableDictionary), dictionary];
            let mut feature_err: Option<String> = None;
            for (name, descriptor) in inputs {
                let key = nsstring_from_str(name)?;
                let desc_obj: *mut Object = msg_send![input_descs, objectForKey: key];
                let (shape, data_type_code) = if desc_obj.is_null() {
                    (
                        coerce_shape(&descriptor.shape),
                        map_dtype(descriptor.data_type),
                    )
                } else {
                    let constraint_obj: *mut Object = msg_send![desc_obj, multiArrayConstraint];
                    if constraint_obj.is_null() {
                        (
                            coerce_shape(&descriptor.shape),
                            map_dtype(descriptor.data_type),
                        )
                    } else {
                        let shape_obj: *mut Object = msg_send![constraint_obj, shape];
                        let ml_data_type: i64 = msg_send![constraint_obj, dataType];
                        (nsarray_to_i64_vec(shape_obj)?, ml_data_type as i32)
                    }
                };

                let array = match create_multi_array(&shape, data_type_code) {
                    Ok(arr) => arr,
                    Err(err) => {
                        feature_err = Some(err.to_string());
                        break;
                    }
                };
                if let Err(err) = fill_zero(array) {
                    feature_err = Some(err.to_string());
                    break;
                }
                let feature_value: *mut Object =
                    msg_send![class!(MLFeatureValue), featureValueWithMultiArray: array];
                let () = msg_send![dict, setObject: feature_value forKey: key];
            }

            if let Some(reason) = feature_err {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(reason),
                });
                continue;
            }

            let mut create_error: *mut Object = ptr::null_mut();
            let provider_alloc: *mut Object = msg_send![class!(MLDictionaryFeatureProvider), alloc];
            let provider: *mut Object =
                msg_send![provider_alloc, initWithDictionary: dict error: &mut create_error];
            if provider.is_null() {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(ns_error_to_string(
                        create_error,
                        "MLDictionaryFeatureProvider init failed",
                    )),
                });
                continue;
            }
            let _provider_guard = ReleaseOnDrop(provider);

            let mut output_provider: *mut Object = ptr::null_mut();
            let mut error = [0u8; 1024];
            let status = rustnn_coreml_predict(
                model,
                provider,
                &mut output_provider,
                error.as_mut_ptr().cast(),
                error.len(),
            );
            if status != 0 || output_provider.is_null() {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(format!(
                        "prediction failed: {}",
                        shim_error_to_string(&error)
                    )),
                });
                continue;
            }
            // The shim returns a retained provider; release it after collecting.
            let _output_provider_guard = ReleaseOnDrop(output_provider);

            match collect_outputs(output_provider, model, None) {
                Ok(outputs) => attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Ok(outputs),
                }),
                Err(err) => attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(err.to_string()),
                }),
            }
        }

        // The temporary source is removed when its handle drops.
        drop(temp_mlmodel);
        if compiled_path.is_none() {
            let _ = std::fs::remove_dir_all(&compiled_path_buf);
        }
        Ok(attempts)
    }
}

#[allow(dead_code)]
fn run_impl_with_inputs(
    model_bytes: &[u8],
    inputs: Vec<CoremlInput>,
    cache_path: Option<&Path>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    run_impl_with_inputs_with_weights(model_bytes, None, inputs, cache_path, None, None)
}

fn run_impl_with_inputs_with_weights(
    model_bytes: &[u8],
    weights_data: Option<&[u8]>,
    inputs: Vec<CoremlInput>,
    cache_path: Option<&Path>,
    input_descriptors: Option<&HashMap<String, OperandDescriptor>>,
    output_descriptors: Option<&HashMap<String, OperandDescriptor>>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    let mut runtime_shape_state = RuntimeShapeState::new();
    let mut actual_input_shapes = HashMap::new();
    for input in &inputs {
        validate_shape_data_length(&input.name, &input.shape, input.data.len())?;
        actual_input_shapes.insert(input.name.clone(), input.shape.clone());
    }
    if let Some(descriptors) = input_descriptors {
        runtime_shape_state.validate_named_shapes(
            &actual_input_shapes,
            descriptors,
            TensorKind::Input,
        )?;
    }

    unsafe {
        let (compiled_url, compiled_path_buf, temp_mlmodel) =
            prepare_compiled_model_with_weights(model_bytes, weights_data, cache_path)?;
        // Owned (+1) by us; released when this function returns on any path.
        let _compiled_url_guard = ReleaseOnDrop(compiled_url);

        // Try only Neural Engine + GPU (best performance on Apple Silicon)
        // Fallback to ALL if that fails
        let targets = [
            (3i64, "CPU_AND_NE"), // CPU + Neural Engine (best for Apple Silicon)
            (2i64, "ALL"),        // All available compute units
            (0i64, "CPU_ONLY"),   // Guaranteed-to-load last resort
        ];
        let mut attempts = Vec::new();

        for (code, name) in targets {
            let config: *mut Object = msg_send![class!(MLModelConfiguration), new];
            let _config_guard = ReleaseOnDrop(config);
            let () = msg_send![config, setComputeUnits: code];
            let mut model: *mut Object = ptr::null_mut();
            let mut error = [0u8; 1024];
            let status = rustnn_coreml_load(
                compiled_url,
                config,
                &mut model,
                error.as_mut_ptr().cast(),
                error.len(),
            );
            if status != 0 || model.is_null() {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(format!(
                        "MLModel load failed: {}",
                        shim_error_to_string(&error)
                    )),
                });
                continue;
            }
            // Owned (+1) by us for this attempt; released at end of iteration.
            let _model_guard = ReleaseOnDrop(model);

            // Get model input descriptions to query expected data types
            let model_description: *mut Object = msg_send![model, modelDescription];
            let input_descs: *mut Object = msg_send![model_description, inputDescriptionsByName];

            let dict: *mut Object = msg_send![class!(NSMutableDictionary), dictionary];
            let mut feature_err: Option<String> = None;

            // Create input features with actual data
            for input in &inputs {
                let key = nsstring_from_str(&input.name)?;
                let shape_i64: Vec<i64> = input.shape.iter().map(|&s| s as i64).collect();

                // Query model's expected data type for this input
                // Following Chromium's approach: match the model's expected type to avoid conversion errors
                let desc_obj: *mut Object = msg_send![input_descs, objectForKey: key];
                let data_type_code = if desc_obj.is_null() {
                    // No model info - default to canonical Float32.
                    NativeType::Float32.code()
                } else {
                    let constraint_obj: *mut Object = msg_send![desc_obj, multiArrayConstraint];
                    if constraint_obj.is_null() {
                        // No constraint - default to canonical Float32.
                        NativeType::Float32.code()
                    } else {
                        let ml_data_type: i64 = msg_send![constraint_obj, dataType];
                        ml_data_type as i32
                    }
                };

                // Create MLMultiArray with the model's expected data type
                let array = match create_multi_array(&shape_i64, data_type_code) {
                    Ok(arr) => arr,
                    Err(err) => {
                        feature_err = Some(err.to_string());
                        break;
                    }
                };

                // Fill with actual data, converting to the target type if needed
                if let Err(err) =
                    fill_data_with_type_conversion(array, &input.data, &shape_i64, data_type_code)
                {
                    feature_err = Some(err.to_string());
                    break;
                }

                let feature_value: *mut Object =
                    msg_send![class!(MLFeatureValue), featureValueWithMultiArray: array];
                let () = msg_send![dict, setObject: feature_value forKey: key];
            }

            if let Some(reason) = feature_err {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(reason),
                });
                continue;
            }

            let mut create_error: *mut Object = ptr::null_mut();
            let provider_alloc: *mut Object = msg_send![class!(MLDictionaryFeatureProvider), alloc];
            let provider: *mut Object =
                msg_send![provider_alloc, initWithDictionary: dict error: &mut create_error];
            if provider.is_null() {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(ns_error_to_string(
                        create_error,
                        "MLDictionaryFeatureProvider init failed",
                    )),
                });
                continue;
            }
            let _provider_guard = ReleaseOnDrop(provider);

            let mut output_provider: *mut Object = ptr::null_mut();
            let mut error = [0u8; 1024];
            let status = rustnn_coreml_predict(
                model,
                provider,
                &mut output_provider,
                error.as_mut_ptr().cast(),
                error.len(),
            );
            if status != 0 || output_provider.is_null() {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(format!(
                        "prediction failed: {}",
                        shim_error_to_string(&error)
                    )),
                });
                continue;
            }
            // The shim returns a retained provider; release it after collecting.
            let _output_provider_guard = ReleaseOnDrop(output_provider);

            match collect_outputs(output_provider, model, output_descriptors) {
                Ok(outputs) => attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Ok(outputs),
                }),
                Err(err) => attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(err.to_string()),
                }),
            }
        }

        // The temporary source is removed when its handle drops.
        drop(temp_mlmodel);
        // Only delete compiled model if not cached
        if cache_path.is_none() {
            let _ = std::fs::remove_dir_all(&compiled_path_buf);
        }

        if let Some(descriptors) = output_descriptors {
            validate_attempt_outputs(&mut attempts, &runtime_shape_state, descriptors);
        }

        Ok(attempts)
    }
}

/// Output failures belong to their compute-policy attempt, just like load and
/// prediction failures. Keep other attempts available for inspection or fallback.
fn validate_attempt_outputs(
    attempts: &mut [CoremlRunAttempt],
    input_shape_state: &RuntimeShapeState,
    descriptors: &HashMap<String, OperandDescriptor>,
) {
    for attempt in attempts {
        let Ok(outputs) = &attempt.result else {
            continue;
        };
        let validation = (|| {
            let mut actual_output_shapes = HashMap::new();
            for output in outputs {
                let mut shape = Vec::with_capacity(output.shape.len());
                for &dim in &output.shape {
                    let dim =
                        usize::try_from(dim).map_err(|_| GraphError::CoremlRuntimeFailed {
                            reason: format!(
                                "output `{}` has invalid negative dimension {}",
                                output.name, dim
                            ),
                        })?;
                    shape.push(dim);
                }
                actual_output_shapes.insert(output.name.clone(), shape);
            }
            // Output-only symbols from one attempt must not bind another, even
            // when validation partially succeeds before reporting an error.
            input_shape_state.clone().validate_named_shapes(
                &actual_output_shapes,
                descriptors,
                TensorKind::Output,
            )
        })();
        if let Err(error) = validation {
            attempt.result = Err(error.to_string());
        }
    }
}

fn collect_named_outputs(
    mut advertised_names: Vec<String>,
    expected: Option<&HashMap<String, OperandDescriptor>>,
    mut lookup: impl FnMut(&str) -> Result<CoremlOutput, GraphError>,
) -> Result<Vec<CoremlOutput>, GraphError> {
    if let Some(expected) = expected {
        advertised_names.extend(expected.keys().cloned());
    }
    // Query every expected name directly, but retain advertised extras so the
    // checked path still rejects unexpected outputs instead of hiding them.
    advertised_names.sort();
    advertised_names.dedup();
    advertised_names.iter().map(|name| lookup(name)).collect()
}

unsafe fn nsarray_to_strings(array: *mut Object) -> Vec<String> {
    let count: usize = msg_send![array, count];
    (0..count)
        .map(|index| {
            let name: *mut Object = msg_send![array, objectAtIndex: index];
            unsafe { nsstring_to_string(name) }
        })
        .collect()
}

unsafe fn collect_outputs(
    provider: *mut Object,
    model: *mut Object,
    expected: Option<&HashMap<String, OperandDescriptor>>,
) -> Result<Vec<CoremlOutput>, GraphError> {
    let feature_names: *mut Object = msg_send![provider, featureNames];
    let names_array: *mut Object = msg_send![feature_names, allObjects];
    let advertised_names = unsafe { nsarray_to_strings(names_array) };

    let lookup_error = |name: &str, detail: &str| {
        let model_description: *mut Object = msg_send![model, modelDescription];
        let descriptions: *mut Object = msg_send![model_description, outputDescriptionsByName];
        let keys: *mut Object = msg_send![descriptions, allKeys];
        let declared_names = unsafe { nsarray_to_strings(keys) };
        let class_name: *mut Object = msg_send![provider, className];
        let class_name = unsafe { nsstring_to_string(class_name) };
        GraphError::CoremlRuntimeFailed {
            reason: format!(
                "output `{name}` {detail}; provider class={class_name}, advertised outputs={advertised_names:?}, model-declared outputs={declared_names:?}"
            ),
        }
    };

    collect_named_outputs(advertised_names.clone(), expected, |name| {
        let name_obj = unsafe { nsstring_from_str(name)? };
        let value: *mut Object = msg_send![provider, featureValueForName: name_obj];
        if value.is_null() {
            return Err(lookup_error(name, "direct lookup returned nil"));
        }
        let array: *mut Object = msg_send![value, multiArrayValue];
        if array.is_null() {
            let feature_type: i64 = msg_send![value, type];
            return Err(lookup_error(
                name,
                &format!("direct lookup returned feature type {feature_type}, not a MLMultiArray"),
            ));
        }
        let data_type: i64 = msg_send![array, dataType];
        let shape_nsarray: *mut Object = msg_send![array, shape];
        let shape = unsafe { nsarray_to_i64_vec(shape_nsarray)? };

        // Extract actual data from MLMultiArray
        let data = unsafe { extract_mlmultiarray_data(array)? };

        Ok(CoremlOutput {
            name: name.to_string(),
            shape,
            data_type_code: data_type,
            data,
        })
    })
}

unsafe fn extract_mlmultiarray_data(array: *mut Object) -> Result<Vec<f32>, GraphError> {
    let (kind, layout, data) = unsafe { multiarray_storage(array)? };
    let bytes = unsafe { read_array_storage(data, &layout, kind.element_size()) };
    let floats = from_native(&bytes, kind, DataType::Float32, layout.count)?;
    Ok(floats
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&bytes| f32::from_ne_bytes(bytes))
        .collect())
}

#[allow(dead_code)]
unsafe fn prepare_compiled_model(
    model_bytes: &[u8],
    cached_compiled: Option<&Path>,
) -> Result<(*mut Object, PathBuf, Option<TempModelSource>), GraphError> {
    unsafe { prepare_compiled_model_with_weights(model_bytes, None, cached_compiled) }
}

/// Compile a model to a `.mlmodelc`, optionally persisting it to `cached_compiled`.
///
/// On success the returned NSURL is owned (+1); the caller must release it
/// (e.g. via [`ReleaseOnDrop`]) once the model has been loaded.
pub(crate) unsafe fn prepare_compiled_model_with_weights(
    model_bytes: &[u8],
    weights_data: Option<&[u8]>,
    cached_compiled: Option<&Path>,
) -> Result<(*mut Object, PathBuf, Option<TempModelSource>), GraphError> {
    let temp_mlmodel = write_temp_model_with_weights(model_bytes, weights_data)?;
    let url = unsafe { nsurl_from_path(temp_mlmodel.path())? };
    let mut compiled_url: *mut Object = ptr::null_mut();
    let mut error = [0u8; 1024];
    let status = unsafe {
        rustnn_coreml_compile(
            url,
            &mut compiled_url,
            error.as_mut_ptr().cast(),
            error.len(),
        )
    };
    if status != 0 || compiled_url.is_null() {
        return Err(GraphError::CoremlRuntimeFailed {
            reason: format!("MLModel compile failed: {}", shim_error_to_string(&error)),
        });
    }
    // The shim returns an owned (+1) URL; the guard releases it on every path
    // that doesn't hand it back to the caller.
    let compiled_url_guard = ReleaseOnDrop(compiled_url);

    let compiled_path_obj: *mut Object = msg_send![compiled_url, path];
    let compiled_src_path = PathBuf::from(unsafe { nsstring_to_string(compiled_path_obj) });

    if let Some(path) = cached_compiled {
        if path.exists() {
            let _ = std::fs::remove_dir_all(path);
        }
        let copy_result = copy_dir_recursively(&compiled_src_path, path);
        // Drop CoreML's temp .mlmodelc either way; the non-cached paths delete
        // it via their callers.
        let _ = std::fs::remove_dir_all(&compiled_src_path);
        if let Err(err) = copy_result {
            return Err(GraphError::CoremlRuntimeFailed {
                reason: format!("failed to persist compiled model: {}", err),
            });
        }
        let persisted_url = unsafe { nsurl_from_path(path)? };
        // `nsurl_from_path` returns an autoreleased URL; retain it so both
        // branches hand the caller an owned (+1) reference.
        let _: *mut Object = msg_send![persisted_url, retain];
        return Ok((persisted_url, path.to_path_buf(), Some(temp_mlmodel)));
    }

    Ok((
        compiled_url_guard.into_inner(),
        compiled_src_path,
        Some(temp_mlmodel),
    ))
}

#[allow(dead_code)]
fn write_temp_model(model_bytes: &[u8]) -> Result<TempModelSource, GraphError> {
    write_temp_model_with_weights(model_bytes, None)
}

/// An on-disk model source (`.mlmodel` file or `.mlpackage` directory) that CoreML
/// compiles from. The underlying temp path is created with a random, `O_EXCL`-guarded
/// name via `tempfile` and is deleted when this handle drops (RAII), so there is no
/// cross-process collision or symlink/TOCTOU exposure and cleanup survives panics.
pub(crate) enum TempModelSource {
    /// A `.mlpackage` directory (used when the model has external weights).
    Package(tempfile::TempDir),
    /// A single `.mlmodel` file.
    Model(tempfile::TempPath),
}

impl TempModelSource {
    fn path(&self) -> &Path {
        match self {
            TempModelSource::Package(dir) => dir.path(),
            TempModelSource::Model(path) => path,
        }
    }
}

/// Write a CoreML model to a temporary path, creating an `.mlpackage` when weights are
/// present. The returned [`TempModelSource`] owns the temp path and removes it on drop.
fn write_temp_model_with_weights(
    model_bytes: &[u8],
    weights_data: Option<&[u8]>,
) -> Result<TempModelSource, GraphError> {
    let map_io = |err: std::io::Error| GraphError::CoremlRuntimeFailed {
        reason: format!("failed to write temporary CoreML model: {err}"),
    };

    if let Some(weights) = weights_data {
        // Create .mlpackage directory structure with weights. The random directory
        // name keeps the `.mlpackage` suffix so CoreML recognizes it as a package.
        let package = tempfile::Builder::new()
            .prefix("rustnn_coreml_")
            .suffix(".mlpackage")
            .tempdir()
            .map_err(map_io)?;
        let package_path = package.path();
        let data_dir = package_path.join("Data").join("com.apple.CoreML");
        let weights_dir = data_dir.join("weights");

        // Create directories
        std::fs::create_dir_all(&weights_dir)
            .map_err(|err| GraphError::export(&weights_dir, err))?;

        // Write model.mlmodel (protobuf)
        let model_path = data_dir.join("model.mlmodel");
        std::fs::write(&model_path, model_bytes)
            .map_err(|err| GraphError::export(&model_path, err))?;

        // Write weights/weights.bin
        let weights_path = weights_dir.join("weights.bin");
        std::fs::write(&weights_path, weights)
            .map_err(|err| GraphError::export(&weights_path, err))?;

        // Write the package Manifest.json. CoreML refuses to load an .mlpackage
        // ("A valid manifest does not exist") without this root-level file
        // pointing at the model spec inside Data/.
        let manifest_path = package_path.join("Manifest.json");
        let model_id = "00000000-0000-0000-0000-0000000000AA";
        let weights_id = "00000000-0000-0000-0000-0000000000BB";
        let manifest = format!(
            r#"{{
  "fileFormatVersion": "1.0.0",
  "itemInfoEntries": {{
    "{model_id}": {{
      "author": "com.apple.CoreML",
      "description": "CoreML Model Specification",
      "name": "model.mlmodel",
      "path": "com.apple.CoreML/model.mlmodel"
    }},
    "{weights_id}": {{
      "author": "com.apple.CoreML",
      "description": "CoreML Model Weights",
      "name": "weights",
      "path": "com.apple.CoreML/weights"
    }}
  }},
  "rootModelIdentifier": "{model_id}"
}}
"#
        );
        std::fs::write(&manifest_path, manifest)
            .map_err(|err| GraphError::export(&manifest_path, err))?;

        Ok(TempModelSource::Package(package))
    } else {
        // No weights: write a single .mlmodel file with a random, exclusively-created
        // name. Convert to a closed `TempPath` so CoreML can open it by path.
        let mut file = tempfile::Builder::new()
            .prefix("rustnn_coreml_")
            .suffix(".mlmodel")
            .tempfile()
            .map_err(map_io)?;
        file.write_all(model_bytes).map_err(map_io)?;
        file.flush().map_err(map_io)?;
        Ok(TempModelSource::Model(file.into_temp_path()))
    }
}

fn coerce_shape(shape: &[Dimension]) -> Vec<i64> {
    let mut dims: Vec<i64> = shape
        .iter()
        .map(|d| i64::from(get_static_or_max_size(d)))
        .collect();
    match dims.len() {
        0 => vec![1],
        1 => dims,
        2 => {
            let mut with_batch = vec![1];
            with_batch.append(&mut dims);
            with_batch
        }
        3 => dims,
        _ => {
            let prod: i64 = dims.iter().product();
            vec![prod]
        }
    }
}

fn map_dtype(data_type: DataType) -> i32 {
    // Match the converter's feature-boundary promotion on every supported OS.
    // In particular, do not allocate OS 26-only Int8 arrays on older devices.
    match data_type {
        DataType::Float16 => NativeType::Float16,
        DataType::Int32 => NativeType::Int32,
        _ => NativeType::Float32,
    }
    .code()
}

/// Query the model's declared `MLMultiArrayDataType` code for the input named `key`.
/// Returns `None` when the model exposes no multi-array constraint for that input.
unsafe fn model_input_dtype_code(input_descs: *mut Object, key: *mut Object) -> Option<i32> {
    if input_descs.is_null() {
        return None;
    }
    let desc_obj: *mut Object = msg_send![input_descs, objectForKey: key];
    if desc_obj.is_null() {
        return None;
    }
    let constraint_obj: *mut Object = msg_send![desc_obj, multiArrayConstraint];
    if constraint_obj.is_null() {
        return None;
    }
    let ml_data_type: i64 = msg_send![constraint_obj, dataType];
    Some(ml_data_type as i32)
}

unsafe fn nsstring_from_str(value: &str) -> Result<*mut Object, GraphError> {
    let c_string = CString::new(value).map_err(|err| GraphError::CoremlRuntimeFailed {
        reason: format!("failed to build NSString: {err}"),
    })?;
    let obj: *mut Object = msg_send![class!(NSString), stringWithUTF8String: c_string.as_ptr()];
    Ok(obj)
}

unsafe fn nsurl_from_path(path: &Path) -> Result<*mut Object, GraphError> {
    let path_str = path
        .to_str()
        .ok_or_else(|| GraphError::CoremlRuntimeFailed {
            reason: format!("invalid path for CoreML model: {}", path.display()),
        })?;
    let ns_path = unsafe { nsstring_from_str(path_str)? };
    let url: *mut Object = msg_send![class!(NSURL), fileURLWithPath: ns_path];
    Ok(url)
}

unsafe fn create_multi_array(shape: &[i64], data_type: i32) -> Result<*mut Object, GraphError> {
    NativeType::from_code(i64::from(data_type))?;
    let numbers: Vec<*mut Object> = shape
        .iter()
        .map(|dim| {
            let number: *mut Object = msg_send![class!(NSNumber), numberWithLongLong: *dim];
            number
        })
        .collect();
    let nsarray: *mut Object =
        msg_send![class!(NSArray), arrayWithObjects: numbers.as_ptr() count: numbers.len()];
    let mut error: *mut Object = ptr::null_mut();
    let alloc: *mut Object = msg_send![class!(MLMultiArray), alloc];
    let array: *mut Object =
        msg_send![alloc, initWithShape: nsarray dataType: data_type error: &mut error];
    if array.is_null() {
        return Err(GraphError::CoremlRuntimeFailed {
            reason: unsafe { ns_error_to_string(error, "MLMultiArray init failed") },
        });
    }
    // Hand callers a pool-backed reference; MLFeatureValue retains the array
    // for as long as the feature dictionary needs it.
    let array: *mut Object = msg_send![array, autorelease];
    Ok(array)
}

unsafe fn fill_zero(array: *mut Object) -> Result<(), GraphError> {
    let (kind, layout, data) = unsafe { multiarray_storage(array)? };
    if layout.count == 0 {
        return Ok(());
    }
    if layout.contiguous {
        unsafe { ptr::write_bytes(data, 0, layout.byte_length) };
    } else {
        for index in 0..layout.count {
            unsafe {
                ptr::write_bytes(data.add(layout.byte_offset(index)), 0, kind.element_size())
            };
        }
    }
    Ok(())
}

/// The convenience API accepts/returns f32 values, so large integer values may
/// lose precision by design. Use typed MLTensor dispatch for exact integer I/O.
unsafe fn fill_data_with_type_conversion(
    array: *mut Object,
    data: &[f32],
    _shape: &[i64],
    data_type_code: i32,
) -> Result<(), GraphError> {
    unsafe {
        fill_multiarray_from_bytes(
            array,
            bytemuck::cast_slice(data),
            DataType::Float32,
            data_type_code,
        )
    }
}

unsafe fn nsarray_to_i64_vec(array: *mut Object) -> Result<Vec<i64>, GraphError> {
    let count: usize = msg_send![array, count];
    let mut result = Vec::with_capacity(count);
    for idx in 0..count {
        let obj: *mut Object = msg_send![array, objectAtIndex: idx];
        let value: i64 = msg_send![obj, longLongValue];
        result.push(value);
    }
    Ok(result)
}

unsafe fn nsstring_to_string(obj: *mut Object) -> String {
    let c_str: *const c_char = msg_send![obj, UTF8String];
    if c_str.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(c_str).to_string_lossy().into_owned() }
}

unsafe fn ns_error_to_string(error: *mut Object, default: &str) -> String {
    if error.is_null() {
        return default.to_string();
    }
    let desc: *mut Object = msg_send![error, localizedDescription];
    if desc.is_null() {
        return default.to_string();
    }
    unsafe { nsstring_to_string(desc) }
}

fn copy_dir_recursively(src: &Path, dst: &Path) -> std::io::Result<()> {
    if dst.exists() {
        std::fs::remove_dir_all(dst)?;
    }
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dst_path = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_recursively(&entry.path(), &dst_path)?;
        } else {
            std::fs::copy(entry.path(), dst_path)?;
        }
    }
    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
mod dtype_storage_tests {
    use super::*;

    unsafe fn strided_array(data: &mut [u8], dtype: NativeType) -> *mut Object {
        let shape: Vec<*mut Object> = [2i64, 3]
            .iter()
            .map(|&value| {
                let number: *mut Object = msg_send![class!(NSNumber), numberWithLongLong: value];
                number
            })
            .collect();
        let strides: Vec<*mut Object> = [8i64, 2]
            .iter()
            .map(|&value| {
                let number: *mut Object = msg_send![class!(NSNumber), numberWithLongLong: value];
                number
            })
            .collect();
        let shape: *mut Object =
            msg_send![class!(NSArray), arrayWithObjects: shape.as_ptr() count: shape.len()];
        let strides: *mut Object =
            msg_send![class!(NSArray), arrayWithObjects: strides.as_ptr() count: strides.len()];
        let alloc: *mut Object = msg_send![class!(MLMultiArray), alloc];
        let mut error: *mut Object = ptr::null_mut();
        let array: *mut Object = msg_send![alloc,
            initWithDataPointer: data.as_mut_ptr()
            shape: shape
            dataType: dtype.code()
            strides: strides
            deallocator: ptr::null_mut::<Object>()
            error: &mut error];
        assert!(!array.is_null(), "{}", unsafe {
            ns_error_to_string(error, "strided allocation failed")
        });
        array
    }

    #[test]
    fn coreml_dtypes_native_strides_conversion_and_zero_preserve_padding() {
        autoreleasepool(|| unsafe {
            let values = [-2f32, -1., 0., 1., 31., 123.];
            for kind in [
                NativeType::Float16,
                NativeType::Float32,
                NativeType::Double,
                NativeType::Int32,
            ] {
                let width = kind.element_size();
                // Six scalars in a padded [2, 3] layout, with an extra tail canary.
                let mut backing = vec![0xa5; 13 * width + 16];
                let array = strided_array(&mut backing, kind);
                let _guard = ReleaseOnDrop(array);
                fill_data_with_type_conversion(array, &values, &[2, 3], kind.code()).unwrap();
                assert_eq!(extract_mlmultiarray_data(array).unwrap(), values);
                let descriptor = OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: crate::graph::to_dimension_vector(&[2, 3]),
                    pending_permutation: vec![],
                };
                let bytes = extract_multiarray_bytes(array, &descriptor).unwrap();
                assert_eq!(bytes, bytemuck::cast_slice::<f32, u8>(&values));
                fill_zero(array).unwrap();
                assert_eq!(extract_mlmultiarray_data(array).unwrap(), [0.; 6]);
                for (index, &byte) in backing.iter().enumerate() {
                    let active = [0, 2, 4, 8, 10, 12].contains(&(index / width));
                    assert_eq!(
                        byte,
                        if active { 0 } else { 0xa5 },
                        "{kind:?}, byte {index}"
                    );
                }
            }
        });
    }

    #[test]
    fn coreml_dtypes_native_exact_int32_and_rejected_lengths() {
        autoreleasepool(|| unsafe {
            let array = create_multi_array(&[2], NativeType::Int32.code()).unwrap();
            let values = [16_777_217i32, i32::MIN + 1];
            fill_multiarray_from_bytes(
                array,
                bytemuck::cast_slice(&values),
                DataType::Int32,
                NativeType::Int32.code(),
            )
            .unwrap();
            let descriptor = OperandDescriptor {
                data_type: DataType::Int32,
                shape: crate::graph::to_dimension_vector(&[2]),
                pending_permutation: vec![],
            };
            assert_eq!(
                extract_multiarray_bytes(array, &descriptor).unwrap(),
                bytemuck::cast_slice::<i32, u8>(&values)
            );
            assert!(
                fill_multiarray_from_bytes(
                    array,
                    &[0; 4],
                    DataType::Int32,
                    NativeType::Int32.code()
                )
                .is_err()
            );
            assert!(
                fill_data_with_type_conversion(array, &[1.], &[2], NativeType::Int32.code())
                    .is_err()
            );
            assert!(create_multi_array(&[2], 16).is_err());
        });
    }

    #[test]
    fn coreml_dtypes_fallback_allocations_match_promoted_model_boundaries() {
        for dtype in [
            DataType::Int4,
            DataType::Uint4,
            DataType::Int8,
            DataType::Uint8,
            DataType::Uint32,
            DataType::Int64,
            DataType::Uint64,
        ] {
            assert_eq!(map_dtype(dtype), NativeType::Float32.code());
        }
        assert_eq!(map_dtype(DataType::Float16), NativeType::Float16.code());
        assert_eq!(map_dtype(DataType::Int32), NativeType::Int32.code());
    }
}

#[cfg(test)]
mod checked_attempt_tests {
    use super::*;
    use crate::graph::DynamicDimension;

    fn descriptor(shape: Vec<Dimension>) -> OperandDescriptor {
        OperandDescriptor {
            data_type: DataType::Float32,
            shape,
            pending_permutation: vec![],
        }
    }

    fn dynamic(name: &str) -> Dimension {
        Dimension::Dynamic(DynamicDimension {
            name: name.into(),
            max_size: 8,
        })
    }

    fn output(shape: &[i64]) -> CoremlOutput {
        let count: usize = shape.iter().map(|&dim| dim.max(0) as usize).product();
        CoremlOutput {
            name: "result".into(),
            shape: shape.to_vec(),
            data_type_code: 65568,
            data: vec![42.; count],
        }
    }

    fn attempt(compute_unit: &'static str, outputs: Vec<CoremlOutput>) -> CoremlRunAttempt {
        CoremlRunAttempt {
            compute_unit,
            result: Ok(outputs),
        }
    }

    #[test]
    fn collection_queries_expected_names_missing_from_advertisement() {
        let descriptors =
            HashMap::from([("result".into(), descriptor(vec![Dimension::Static(2)]))]);
        let mut queried = Vec::new();
        let outputs = collect_named_outputs(vec![], Some(&descriptors), |name| {
            queried.push(name.to_string());
            Ok(output(&[2]))
        })
        .unwrap();
        assert_eq!(queried, ["result"]);
        assert_eq!(outputs[0].name, "result");
        assert_eq!(outputs[0].data, [42., 42.]);
    }

    #[test]
    fn collection_retains_unexpected_advertised_outputs_for_validation() {
        let descriptors =
            HashMap::from([("result".into(), descriptor(vec![Dimension::Static(2)]))]);
        let mut queried = Vec::new();
        let outputs = collect_named_outputs(
            vec!["result".into(), "extra".into()],
            Some(&descriptors),
            |name| {
                queried.push(name.to_string());
                let mut value = output(&[2]);
                value.name = name.into();
                Ok(value)
            },
        )
        .unwrap();
        assert_eq!(queried, ["extra", "result"]);
        let mut attempts = [attempt("CPU_ONLY", outputs)];
        validate_attempt_outputs(&mut attempts, &RuntimeShapeState::new(), &descriptors);
        assert_eq!(
            attempts[0].result.as_ref().unwrap_err(),
            "unexpected runtime output tensor `extra`"
        );
    }

    #[test]
    fn collection_propagates_failed_direct_lookup() {
        let descriptors =
            HashMap::from([("result".into(), descriptor(vec![Dimension::Static(2)]))]);
        let error = collect_named_outputs(vec![], Some(&descriptors), |name| {
            Err(GraphError::CoremlRuntimeFailed {
                reason: format!("{name}: direct lookup returned nil"),
            })
        })
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("result: direct lookup returned nil")
        );
    }

    #[test]
    fn missing_output_does_not_discard_other_attempts_or_existing_errors() {
        let mut attempts = vec![
            attempt("CPU_AND_NE", vec![]),
            CoremlRunAttempt {
                compute_unit: "ALL",
                result: Err("prediction failed".into()),
            },
            attempt("CPU_ONLY", vec![output(&[2])]),
        ];
        let descriptors =
            HashMap::from([("result".into(), descriptor(vec![Dimension::Static(2)]))]);
        validate_attempt_outputs(&mut attempts, &RuntimeShapeState::new(), &descriptors);

        assert_eq!(
            attempts
                .iter()
                .map(|attempt| attempt.compute_unit)
                .collect::<Vec<_>>(),
            ["CPU_AND_NE", "ALL", "CPU_ONLY"]
        );
        assert_eq!(
            attempts[0].result.as_ref().unwrap_err(),
            "missing runtime output tensor `result`"
        );
        assert_eq!(
            attempts[1].result.as_ref().unwrap_err(),
            "prediction failed"
        );
        assert_eq!(attempts[2].result.as_ref().unwrap()[0].data, [42., 42.]);
    }

    #[test]
    fn malformed_output_shapes_fail_only_their_own_attempt() {
        let descriptors =
            HashMap::from([("result".into(), descriptor(vec![Dimension::Static(2)]))]);
        for (shape, expected_error) in [
            (vec![-1], "invalid negative dimension -1"),
            (vec![2, 1], "rank mismatch"),
            (vec![3], "dimension 0 mismatch"),
        ] {
            let mut attempts = vec![
                attempt("CPU_AND_NE", vec![output(&shape)]),
                attempt("CPU_ONLY", vec![output(&[2])]),
            ];
            validate_attempt_outputs(&mut attempts, &RuntimeShapeState::new(), &descriptors);
            assert!(
                attempts[0]
                    .result
                    .as_ref()
                    .unwrap_err()
                    .contains(expected_error)
            );
            assert_eq!(attempts[1].result.as_ref().unwrap()[0].shape, [2]);
        }
    }

    #[test]
    fn cpu_output_failure_remains_visible_after_accelerated_success() {
        let mut attempts = vec![
            attempt("ALL", vec![output(&[2])]),
            attempt("CPU_ONLY", vec![]),
        ];
        let descriptors =
            HashMap::from([("result".into(), descriptor(vec![Dimension::Static(2)]))]);
        validate_attempt_outputs(&mut attempts, &RuntimeShapeState::new(), &descriptors);
        assert!(attempts[0].result.is_ok());
        assert_eq!(attempts[1].compute_unit, "CPU_ONLY");
        assert!(attempts[1].result.is_err());
    }

    #[test]
    fn every_attempt_preserves_input_dynamic_dimension_bindings() {
        let descriptors = HashMap::from([("result".into(), descriptor(vec![dynamic("rows")]))]);
        let mut input_state = RuntimeShapeState::new();
        input_state
            .validate_shape("data", &[2], &descriptors["result"], TensorKind::Input)
            .unwrap();
        let mut attempts = vec![
            attempt("ALL", vec![output(&[3])]),
            attempt("CPU_ONLY", vec![output(&[2])]),
        ];
        validate_attempt_outputs(&mut attempts, &input_state, &descriptors);
        assert!(
            attempts[0]
                .result
                .as_ref()
                .unwrap_err()
                .contains("dynamic dimension `rows` mismatch")
        );
        assert!(attempts[1].result.is_ok());
    }

    #[test]
    fn output_bindings_from_failed_attempt_do_not_leak_to_next_attempt() {
        let descriptors = HashMap::from([(
            "result".into(),
            descriptor(vec![dynamic("output_rows"), Dimension::Static(1)]),
        )]);
        // The first output binds output_rows=2 before failing its second axis.
        let mut attempts = vec![
            attempt("ALL", vec![output(&[2, 2])]),
            attempt("CPU_ONLY", vec![output(&[3, 1])]),
        ];
        validate_attempt_outputs(&mut attempts, &RuntimeShapeState::new(), &descriptors);
        assert!(attempts[0].result.is_err());
        assert_eq!(attempts[1].result.as_ref().unwrap()[0].shape, [3, 1]);
    }
}

#[cfg(test)]
mod temp_model_tests {
    use super::{TempModelSource, write_temp_model_with_weights};

    #[test]
    fn bare_model_writes_mlmodel_file_and_cleans_up_on_drop() {
        let bytes = b"fake-mlmodel-protobuf";
        let source = write_temp_model_with_weights(bytes, None).expect("write temp model");
        let path = source.path().to_path_buf();

        assert!(matches!(&source, TempModelSource::Model(_)));
        assert!(path.is_file(), "expected a file at {path:?}");
        assert_eq!(path.extension().and_then(|e| e.to_str()), Some("mlmodel"));
        assert!(
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("rustnn_coreml_")),
            "unexpected temp name: {path:?}"
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);

        // Dropping the handle removes the temp file (RAII cleanup).
        drop(source);
        assert!(
            !path.exists(),
            "temp file should be removed on drop: {path:?}"
        );
    }

    #[test]
    fn model_with_weights_writes_mlpackage_and_cleans_up_on_drop() {
        let model = b"fake-mlmodel";
        let weights = b"\x00\x01\x02\x03weights-blob";
        let source =
            write_temp_model_with_weights(model, Some(weights)).expect("write temp package");
        let dir = source.path().to_path_buf();

        assert!(matches!(&source, TempModelSource::Package(_)));
        assert!(dir.is_dir(), "expected an .mlpackage dir at {dir:?}");
        assert_eq!(dir.extension().and_then(|e| e.to_str()), Some("mlpackage"));

        let model_path = dir.join("Data/com.apple.CoreML/model.mlmodel");
        let weights_path = dir.join("Data/com.apple.CoreML/weights/weights.bin");
        let manifest_path = dir.join("Manifest.json");
        assert_eq!(std::fs::read(&model_path).unwrap(), model);
        assert_eq!(std::fs::read(&weights_path).unwrap(), weights);
        let manifest = std::fs::read_to_string(&manifest_path).unwrap();
        assert!(manifest.contains("rootModelIdentifier"));
        assert!(manifest.contains("com.apple.CoreML/model.mlmodel"));

        // Dropping the handle removes the whole package directory (RAII cleanup).
        drop(source);
        assert!(
            !dir.exists(),
            "temp package should be removed on drop: {dir:?}"
        );
    }

    #[test]
    fn temp_paths_are_unique_across_calls() {
        // Randomized names must not collide even for identical content compiled
        // back-to-back (the old millisecond+counter scheme could).
        let a = write_temp_model_with_weights(b"same", None).unwrap();
        let b = write_temp_model_with_weights(b"same", None).unwrap();
        assert_ne!(a.path(), b.path(), "temp names must be unique");
    }
}

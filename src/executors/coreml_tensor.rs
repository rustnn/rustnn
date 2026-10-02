//! Owned, reusable tensor storage for synchronous CoreML dispatch.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::{HashMap, HashSet};
use std::ptr::NonNull;
use std::sync::Mutex;

use super::*;
use crate::mlcontextoptions::CoremlTensorStatistics;

fn failed(reason: impl Into<String>) -> GraphError {
    GraphError::CoremlRuntimeFailed {
        reason: reason.into(),
    }
}

fn native_code(dtype: DataType) -> Option<i32> {
    match dtype {
        DataType::Float32 => Some(NativeType::Float32.code()),
        DataType::Float16 => Some(NativeType::Float16.code()),
        DataType::Int32 => Some(NativeType::Int32.code()),
        _ => None,
    }
}

fn same_type(dtype: DataType, code: i32) -> bool {
    NativeType::from_code(i64::from(code)).is_ok_and(|kind| kind.matches(dtype))
}

fn is_fixed_output_shape(constraint_type: i64, enumerated_count: usize) -> bool {
    // MLMultiArrayShapeConstraintTypeEnumerated with exactly one allowed shape.
    constraint_type == 2 && enumerated_count == 1
}

/// Only this context-owned storage may mutate its array. Locks remain held
/// throughout synchronous prediction, including direct writes to output backings.
#[derive(Debug)]
pub(crate) enum CoremlTensorStorage {
    Host(Vec<u8>),
    Native(Mutex<NativeTensor>),
}

#[derive(Debug)]
pub(crate) struct NativeTensor {
    data: NonNull<u8>,
    layout: Layout,
    capacity: usize,
    dtype: DataType,
    view: *mut Object,
    shape: Vec<i64>,
}

// SAFETY: The allocation and retained view are exclusively owned. The view is
// only exposed inside synchronous dispatch while its owning mutex is locked;
// no Rust reference or Objective-C object escapes that operation. CoreML arrays
// have no thread affinity. Moving an idle owner between threads is safe; sharing
// mutable access without the mutex is deliberately not supported (no Sync impl).
unsafe impl Send for NativeTensor {}

impl Drop for NativeTensor {
    fn drop(&mut self) {
        unsafe {
            if !self.view.is_null() {
                let _: () = msg_send![self.view, release];
            }
            dealloc(self.data.as_ptr(), self.layout);
        }
    }
}

impl NativeTensor {
    fn new(dtype: DataType, capacity: usize) -> Result<Self, GraphError> {
        // Keep scalar/index allocations small. Larger cache buffers retain page
        // alignment for CoreML's outputBackings performance recommendation.
        // 16 KiB accommodates both supported Apple page sizes (4 and 16 KiB).
        let alignment = if capacity >= 16384 { 16384 } else { 16 };
        let layout = Layout::from_size_align(capacity.max(1), alignment)
            .map_err(|e| failed(format!("native tensor allocation: {e}")))?;
        let data = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or_else(|| failed("native tensor allocation failed"))?;
        Ok(Self {
            data,
            layout,
            capacity,
            dtype,
            view: ptr::null_mut(),
            shape: Vec::new(),
        })
    }

    fn bytes(&self, len: usize) -> Result<&[u8], GraphError> {
        if len > self.capacity {
            return Err(failed("tensor storage shorter than logical size"));
        }
        Ok(unsafe { std::slice::from_raw_parts(self.data.as_ptr(), len) })
    }

    fn write(&mut self, bytes: &[u8]) -> Result<(), GraphError> {
        if bytes.len() > self.capacity {
            return Err(failed("write exceeds tensor capacity"));
        }
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.as_ptr(), bytes.len()) };
        Ok(())
    }

    fn array(&mut self, descriptor: &OperandDescriptor) -> Result<*mut Object, GraphError> {
        let shape = physical_shape(descriptor)?;
        let length = descriptor
            .byte_length()
            .ok_or_else(|| failed("tensor byte length overflow"))?;
        if length > self.capacity || descriptor.data_type != self.dtype {
            return Err(failed("native tensor shape/type exceeds owned storage"));
        }
        if self.view.is_null() || self.shape != shape {
            let mut strides = vec![1i64; shape.len()];
            for i in (0..shape.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1]
                    .checked_mul(shape[i + 1])
                    .ok_or_else(|| failed("native tensor stride overflow"))?;
            }
            let mut view = ptr::null_mut();
            let mut error = [0u8; 1024];
            let status = unsafe {
                rustnn_coreml_array_view(
                    self.data.as_ptr().cast(),
                    shape.as_ptr(),
                    strides.as_ptr(),
                    shape.len(),
                    native_code(self.dtype).expect("native dtype"),
                    &mut view,
                    error.as_mut_ptr().cast(),
                    error.len(),
                )
            };
            if status != 0 || view.is_null() {
                return Err(failed(format!(
                    "native tensor view: {}",
                    shim_error_to_string(&error)
                )));
            }
            unsafe {
                if !self.view.is_null() {
                    let _: () = msg_send![self.view, release];
                }
            }
            self.view = view;
            self.shape = shape;
        }
        Ok(self.view)
    }
}

impl CoremlTensorStorage {
    pub(crate) fn new(dtype: DataType, capacity: usize, reuse: bool) -> Result<Self, GraphError> {
        if reuse && native_code(dtype).is_some() {
            Ok(Self::Native(Mutex::new(NativeTensor::new(
                dtype, capacity,
            )?)))
        } else {
            Ok(Self::Host(vec![0; capacity.max(1)]))
        }
    }

    pub(crate) fn is_native(&self) -> bool {
        matches!(self, Self::Native(_))
    }

    pub(crate) fn host(&self) -> Result<&[u8], GraphError> {
        match self {
            Self::Host(bytes) => Ok(bytes),
            Self::Native(_) => Err(failed("expected host tensor")),
        }
    }

    pub(crate) fn read(&self, destination: &mut [u8]) -> Result<(), GraphError> {
        match self {
            Self::Host(bytes) => destination.copy_from_slice(
                bytes
                    .get(..destination.len())
                    .ok_or_else(|| failed("tensor storage shorter than logical size"))?,
            ),
            Self::Native(native) => destination.copy_from_slice(
                native
                    .lock()
                    .map_err(|_| failed("poisoned native tensor"))?
                    .bytes(destination.len())?,
            ),
        }
        Ok(())
    }

    pub(crate) fn write(&mut self, source: &[u8]) -> Result<(), GraphError> {
        match self {
            Self::Host(bytes) => bytes
                .get_mut(..source.len())
                .ok_or_else(|| failed("write exceeds tensor capacity"))?
                .copy_from_slice(source),
            Self::Native(native) => native
                .get_mut()
                .map_err(|_| failed("poisoned native tensor"))?
                .write(source)?,
        }
        Ok(())
    }

    /// Grow while preserving bytes, or replace with zeroed capacity (reserve API).
    pub(crate) fn reserve(&mut self, capacity: usize, preserve: bool) -> Result<bool, GraphError> {
        match self {
            Self::Host(bytes) => {
                if !preserve {
                    *bytes = vec![0; capacity.max(1)];
                } else if capacity > bytes.len() {
                    bytes.resize(capacity, 0);
                }
                Ok(false)
            }
            Self::Native(native) => {
                let old = native
                    .get_mut()
                    .map_err(|_| failed("poisoned native tensor"))?;
                if preserve && capacity <= old.capacity {
                    return Ok(false);
                }
                let mut replacement = NativeTensor::new(old.dtype, capacity)?;
                if preserve {
                    replacement.write(old.bytes(old.capacity.min(capacity))?)?;
                }
                *old = replacement;
                Ok(true)
            }
        }
    }
}

fn physical_shape(descriptor: &OperandDescriptor) -> Result<Vec<i64>, GraphError> {
    let shape = descriptor.static_or_max_shape();
    if shape.contains(&0) {
        return Err(failed("CoreML cannot bind a zero-extent tensor"));
    }
    Ok(if shape.is_empty() {
        vec![1]
    } else {
        shape.iter().map(|&d| i64::from(d)).collect()
    })
}

pub(crate) struct CoremlTensorBinding<'a> {
    pub(crate) storage: &'a CoremlTensorStorage,
    pub(crate) descriptor: &'a OperandDescriptor,
}

/// Execute with retained input views and optional destination backings. Native
/// destinations keep exclusive ownership even when CoreML returns aliased views.
pub(crate) fn run_coreml_tensors(
    model: &CompiledCoremlModel,
    inputs: &HashMap<String, CoremlTensorBinding<'_>>,
    outputs: &HashMap<String, CoremlTensorBinding<'_>>,
    output_backings: bool,
    statistics: &mut CoremlTensorStatistics,
) -> Result<HashMap<String, Vec<u8>>, GraphError> {
    let mut unique = HashSet::new();
    for binding in inputs.values().chain(outputs.values()) {
        if !unique.insert(ptr::from_ref(binding.storage)) {
            return Err(failed("duplicate CoreML tensor storage binding"));
        }
    }
    autoreleasepool(|| unsafe {
        let model_description: *mut Object = msg_send![model.model, modelDescription];
        let input_descs: *mut Object = msg_send![model_description, inputDescriptionsByName];
        let output_descs: *mut Object = msg_send![model_description, outputDescriptionsByName];
        let dict: *mut Object = msg_send![class!(NSMutableDictionary), dictionary];
        let backings: *mut Object = msg_send![class!(NSMutableDictionary), dictionary];
        let mut guards = Vec::new();
        let mut destinations = HashMap::new();
        for (name, binding) in inputs {
            let key = nsstring_from_str(name)?;
            let code = model_input_dtype_code(input_descs, key)
                .unwrap_or_else(|| map_dtype(binding.descriptor.data_type));
            let logical = binding
                .descriptor
                .byte_length()
                .ok_or_else(|| failed("input size overflow"))?;
            let array = match binding.storage {
                CoremlTensorStorage::Native(native) => {
                    let mut guard = native
                        .lock()
                        .map_err(|_| failed("poisoned native tensor"))?;
                    let array = if same_type(binding.descriptor.data_type, code) {
                        statistics.native_input_bindings += 1;
                        guard.array(binding.descriptor)?
                    } else {
                        let array = create_multi_array(&physical_shape(binding.descriptor)?, code)?;
                        fill_multiarray_from_bytes(
                            array,
                            guard.bytes(logical)?,
                            binding.descriptor.data_type,
                            code,
                        )?;
                        statistics.input_copy_bytes += logical as u64;
                        array
                    };
                    guards.push(guard);
                    array
                }
                CoremlTensorStorage::Host(bytes) => {
                    let array = create_multi_array(&physical_shape(binding.descriptor)?, code)?;
                    fill_multiarray_from_bytes(
                        array,
                        bytes
                            .get(..logical)
                            .ok_or_else(|| failed("short input storage"))?,
                        binding.descriptor.data_type,
                        code,
                    )?;
                    statistics.input_copy_bytes += logical as u64;
                    array
                }
            };
            let feature: *mut Object =
                msg_send![class!(MLFeatureValue), featureValueWithMultiArray: array];
            let _: () = msg_send![dict, setObject: feature forKey: key];
        }
        for (name, binding) in outputs {
            if let CoremlTensorStorage::Native(native) = binding.storage {
                let mut guard = native
                    .lock()
                    .map_err(|_| failed("poisoned native tensor"))?;
                let array = guard.array(binding.descriptor)?;
                let key = nsstring_from_str(name)?;
                let code = model_input_dtype_code(output_descs, key);
                if output_backings
                    && code.is_some_and(|code| same_type(binding.descriptor.data_type, code))
                {
                    let feature: *mut Object =
                        msg_send![class!(MLFeatureValue), featureValueWithMultiArray: array];
                    let description: *mut Object = msg_send![output_descs, objectForKey: key];
                    let constraint: *mut Object = msg_send![description, multiArrayConstraint];
                    let shape_constraint: *mut Object = msg_send![constraint, shapeConstraint];
                    let constraint_type: i64 = msg_send![shape_constraint, type];
                    let allowed: bool = msg_send![description, isAllowedValue: feature];
                    // CoreML rejects outputBackings for flexible output features,
                    // even when isAllowedValue accepts this particular shape.
                    // Fixed outputs are represented as a singleton enumeration
                    // (MLMultiArrayShapeConstraintTypeEnumerated = 2). Ranged,
                    // multi-shape and unconstrained outputs use the copy fallback.
                    let enumerated_count = if constraint_type == 2 {
                        let shapes: *mut Object = msg_send![shape_constraint, enumeratedShapes];
                        msg_send![shapes, count]
                    } else {
                        0
                    };
                    if is_fixed_output_shape(constraint_type, enumerated_count) && allowed {
                        let _: () = msg_send![backings, setObject: array forKey: key];
                        statistics.output_backings_requested += 1;
                        destinations.insert(name, (guards.len(), true));
                    } else {
                        destinations.insert(name, (guards.len(), false));
                    }
                } else {
                    destinations.insert(name, (guards.len(), false));
                }
                guards.push(guard);
            }
        }
        let mut create_error: *mut Object = ptr::null_mut();
        let provider_alloc: *mut Object = msg_send![class!(MLDictionaryFeatureProvider), alloc];
        let provider: *mut Object =
            msg_send![provider_alloc, initWithDictionary: dict error: &mut create_error];
        if provider.is_null() {
            return Err(failed(ns_error_to_string(
                create_error,
                "feature provider init failed",
            )));
        }
        let _provider_guard = ReleaseOnDrop(provider);
        let mut output_provider: *mut Object = ptr::null_mut();
        let mut error = [0u8; 1024];
        let status = rustnn_coreml_predict_backed(
            model.model,
            provider,
            backings,
            &mut output_provider,
            error.as_mut_ptr().cast(),
            error.len(),
        );
        if status != 0 || output_provider.is_null() {
            return Err(failed(format!(
                "prediction failed: {}",
                shim_error_to_string(&error)
            )));
        }
        let _output_guard = ReleaseOnDrop(output_provider);
        let mut result = HashMap::new();
        for (name, binding) in outputs {
            let key = nsstring_from_str(name)?;
            let feature: *mut Object = msg_send![output_provider, featureValueForName: key];
            if feature.is_null() {
                return Err(failed(format!("missing output '{name}'")));
            }
            let array: *mut Object = msg_send![feature, multiArrayValue];
            if array.is_null() {
                return Err(failed(format!("output '{name}' is not a tensor")));
            }
            let actual_shape: *mut Object = msg_send![array, shape];
            let actual_shape = nsarray_to_i64_vec(actual_shape)?;
            let expected_shape = physical_shape(binding.descriptor)?;
            if actual_shape != expected_shape {
                return Err(failed(format!(
                    "output '{name}': actual shape {actual_shape:?}, expected {expected_shape:?}"
                )));
            }
            let logical = binding
                .descriptor
                .byte_length()
                .ok_or_else(|| failed("output size overflow"))?;
            if let Some(&(index, requested)) = destinations.get(name) {
                let destination = &mut guards[index];
                if requested && array == destination.view {
                    statistics.output_backings_accepted += 1;
                } else {
                    copy_output(array, destination, binding.descriptor)?;
                    statistics.output_copy_bytes += logical as u64;
                }
            } else {
                result.insert(
                    name.clone(),
                    extract_multiarray_bytes(array, binding.descriptor)?,
                );
                statistics.output_copy_bytes += logical as u64;
            }
        }
        Ok(result)
    })
}

unsafe fn copy_output(
    array: *mut Object,
    destination: &mut NativeTensor,
    descriptor: &OperandDescriptor,
) -> Result<(), GraphError> {
    let (kind, layout, source) = unsafe { multiarray_storage(array)? };
    let logical = descriptor
        .byte_length()
        .ok_or_else(|| failed("output size overflow"))?;
    if logical > destination.capacity
        || descriptor.data_type != destination.dtype
        || Some(logical)
            != layout
                .count
                .checked_mul(descriptor.data_type.bytes_per_element())
    {
        return Err(failed("invalid native output layout"));
    }
    if layout.count == 0 {
        return Ok(());
    }
    if !kind.matches(descriptor.data_type) {
        let bytes = unsafe { read_array_storage(source, &layout, kind.element_size()) };
        return destination.write(&from_native(
            &bytes,
            kind,
            descriptor.data_type,
            layout.count,
        )?);
    }
    if layout.contiguous {
        // Overlap is legal if CoreML returns another view of the proposed backing.
        unsafe { ptr::copy(source, destination.data.as_ptr(), logical) };
    } else {
        let element = kind.element_size();
        let source_end = source
            .addr()
            .checked_add(layout.storage_byte_length)
            .ok_or_else(|| failed("output address overflow"))?;
        let destination_end = destination
            .data
            .as_ptr()
            .addr()
            .checked_add(logical)
            .ok_or_else(|| failed("destination address overflow"))?;
        if source.addr() < destination_end && destination.data.as_ptr().addr() < source_end {
            // Per-element memmove is insufficient for an overlapping transpose:
            // an early destination write can destroy a later source element.
            let gathered = unsafe { read_array_storage(source, &layout, element) };
            return destination.write(&gathered);
        }
        // Distinct returned arrays may alias input buffers, so never retain them
        // as a logical output tensor. Gather into this tensor's private allocation.
        for index in 0..layout.count {
            unsafe {
                ptr::copy(
                    source.add(layout.byte_offset(index)),
                    destination.data.as_ptr().add(index * element),
                    element,
                )
            };
        }
    }
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
unsafe extern "C" {
    fn rustnn_coreml_array_view(
        data: *mut c_void,
        shape: *const i64,
        strides: *const i64,
        rank: usize,
        dtype: i32,
        out: *mut *mut Object,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn rustnn_coreml_predict_backed(
        model: *mut Object,
        features: *mut Object,
        backings: *mut Object,
        out: *mut *mut Object,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
#[allow(clippy::too_many_arguments)] // Mirrors the native C ABI above.
unsafe fn rustnn_coreml_array_view(
    _data: *mut c_void,
    _shape: *const i64,
    _strides: *const i64,
    _rank: usize,
    _dtype: i32,
    _out: *mut *mut Object,
    _error: *mut c_char,
    _length: usize,
) -> i32 {
    1
}
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
unsafe fn rustnn_coreml_predict_backed(
    _model: *mut Object,
    _features: *mut Object,
    _backings: *mut Object,
    _out: *mut *mut Object,
    _error: *mut c_char,
    _length: usize,
) -> i32 {
    1
}

#[cfg(test)]
mod shape_tests {
    use super::{DataType, is_fixed_output_shape, same_type};

    #[test]
    fn native_tensor_bindings_require_canonical_matching_types() {
        for (dtype, code) in [
            (DataType::Float16, 65552),
            (DataType::Float32, 65568),
            (DataType::Int32, 131104),
        ] {
            assert!(same_type(dtype, code));
            for other in [16, 32, 3, 65600, 131080, -1] {
                assert!(!same_type(dtype, other));
            }
            for other in [DataType::Float16, DataType::Float32, DataType::Int32] {
                assert_eq!(same_type(other, code), other == dtype);
            }
        }
    }

    #[test]
    fn output_backings_require_a_single_enumerated_shape() {
        assert!(is_fixed_output_shape(2, 1));
        for constraint_type in [0, 1, 3] {
            assert!(!is_fixed_output_shape(constraint_type, 1));
        }
        for count in [0, 2, 3] {
            assert!(!is_fixed_output_shape(2, count));
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn native_alignment_tracks_capacity_and_growth_preserves_contents() {
        for capacity in [1, 2, 4, 64, 16383, 16384, 65536] {
            let native = NativeTensor::new(DataType::Float32, capacity).unwrap();
            let expected = if capacity >= 16384 { 16384 } else { 16 };
            assert_eq!(native.layout.align(), expected);
            assert_eq!(native.layout.size(), capacity);
            assert_eq!(native.data.as_ptr() as usize % expected, 0);
        }
        let mut storage = CoremlTensorStorage::new(DataType::Float32, 4, true).unwrap();
        let contents = 42f32.to_ne_bytes();
        storage.write(&contents).unwrap();
        let CoremlTensorStorage::Native(native) = &mut storage else {
            panic!("native storage")
        };
        native.get_mut().unwrap().array(&descriptor(&[1])).unwrap();
        assert!(storage.reserve(32768, true).unwrap());
        let mut read = [0; 4];
        storage.read(&mut read).unwrap();
        assert_eq!(read, contents);
        let CoremlTensorStorage::Native(native) = &mut storage else {
            panic!("native storage")
        };
        let native = native.get_mut().unwrap();
        assert_eq!(native.layout.align(), 16384);
        native.array(&descriptor(&[8192])).unwrap();
        assert_eq!(native.shape, [8192]);
        assert!(storage.reserve(4, false).unwrap());
        storage.read(&mut read).unwrap();
        assert_eq!(read, [0; 4]);
        let CoremlTensorStorage::Native(native) = &mut storage else {
            panic!("native storage")
        };
        let native = native.get_mut().unwrap();
        assert_eq!(native.layout.align(), 16);
        native.array(&descriptor(&[1])).unwrap();
        assert_eq!(native.shape, [1]);
    }

    fn descriptor(shape: &[u32]) -> OperandDescriptor {
        OperandDescriptor {
            data_type: DataType::Float32,
            shape: crate::graph::to_dimension_vector(shape),
            pending_permutation: vec![],
        }
    }

    unsafe fn view(
        data: *mut u8,
        shape: &[i64],
        strides: &[i64],
        kind: NativeType,
    ) -> ReleaseOnDrop {
        let mut view = ptr::null_mut();
        let mut error = [0u8; 1024];
        let status = unsafe {
            rustnn_coreml_array_view(
                data.cast(),
                shape.as_ptr(),
                strides.as_ptr(),
                shape.len(),
                kind.code(),
                &mut view,
                error.as_mut_ptr().cast(),
                error.len(),
            )
        };
        assert_eq!(status, 0, "{}", shim_error_to_string(&error));
        assert!(!view.is_null());
        ReleaseOnDrop(view)
    }

    #[test]
    fn native_output_fallback_converts_types_and_gathers_checked_strides() {
        let cases: [(NativeType, DataType, Vec<u8>, Vec<u8>); 6] = [
            (
                NativeType::Int32,
                DataType::Int32,
                bytemuck::cast_slice(&[i32::MIN, -16_777_217, 16_777_217, i32::MAX]).to_vec(),
                bytemuck::cast_slice(&[i32::MIN, -16_777_217, 16_777_217, i32::MAX]).to_vec(),
            ),
            (
                NativeType::Int32,
                DataType::Float32,
                bytemuck::cast_slice(&[-2i32, 1, 123, 16_777_217]).to_vec(),
                bytemuck::cast_slice(&[-2f32, 1., 123., 16_777_216.]).to_vec(),
            ),
            (
                NativeType::Float32,
                DataType::Int32,
                bytemuck::cast_slice(&[-2.75f32, 1., 123., 65504.]).to_vec(),
                bytemuck::cast_slice(&[-2i32, 1, 123, 65504]).to_vec(),
            ),
            (
                NativeType::Float16,
                DataType::Float16,
                bytemuck::cast_slice(&[0xc180u16, 0x8000, 0x0001, 0x7bff]).to_vec(),
                bytemuck::cast_slice(&[0xc180u16, 0x8000, 0x0001, 0x7bff]).to_vec(),
            ),
            (
                NativeType::Float16,
                DataType::Float32,
                bytemuck::cast_slice(&[0xc180u16, 0x3c00, 0x57b0, 0x7bff]).to_vec(),
                bytemuck::cast_slice(&[-2.75f32, 1., 123., 65504.]).to_vec(),
            ),
            (
                NativeType::Double,
                DataType::Float16,
                bytemuck::cast_slice(&[-2.75f64, 1., 123., 65504.]).to_vec(),
                bytemuck::cast_slice(&[0xc180u16, 0x3c00, 0x57b0, 0x7bff]).to_vec(),
            ),
        ];
        autoreleasepool(|| {
            for (kind, dtype, packed, expected) in cases {
                for strides in [[2i64, 1], [3, 1], [1, 2]] {
                    let element = kind.element_size();
                    // Padded source and destination canaries catch width errors.
                    let mut source = vec![0xa5u8; 8 * element];
                    for index in 0..4 {
                        let offset = (index / 2 * strides[0] as usize
                            + index % 2 * strides[1] as usize)
                            * element;
                        source[offset..offset + element]
                            .copy_from_slice(&packed[index * element..(index + 1) * element]);
                    }
                    let before = source.clone();
                    let view = unsafe { view(source.as_mut_ptr(), &[2, 2], &strides, kind) };
                    let mut destination = NativeTensor::new(dtype, expected.len() + 16).unwrap();
                    destination.write(&vec![0xa5; expected.len() + 16]).unwrap();
                    let mut desc = descriptor(&[2, 2]);
                    desc.data_type = dtype;
                    unsafe { copy_output(view.0, &mut destination, &desc).unwrap() };
                    assert_eq!(destination.bytes(expected.len()).unwrap(), expected);
                    assert_eq!(
                        &destination.bytes(expected.len() + 16).unwrap()[expected.len()..],
                        &[0xa5; 16]
                    );
                    assert_eq!(source, before);
                    source.fill(0);
                    assert_eq!(destination.bytes(expected.len()).unwrap(), expected);
                }
            }
        });
    }

    #[test]
    fn native_output_fallback_rejects_mismatched_count_and_destination_type() {
        autoreleasepool(|| {
            let mut source = [1f32, 2., 3., 4.];
            let view = unsafe {
                view(
                    source.as_mut_ptr().cast(),
                    &[2, 2],
                    &[2, 1],
                    NativeType::Float32,
                )
            };
            let mut destination = NativeTensor::new(DataType::Float32, 16).unwrap();
            destination.write(&[0xa5; 16]).unwrap();
            assert!(unsafe { copy_output(view.0, &mut destination, &descriptor(&[3])) }.is_err());
            let mut wrong_type = descriptor(&[2, 2]);
            wrong_type.data_type = DataType::Int32;
            assert!(unsafe { copy_output(view.0, &mut destination, &wrong_type) }.is_err());
            assert_eq!(destination.bytes(16).unwrap(), &[0xa5; 16]);
        });
    }

    #[test]
    fn fixed_output_backing_proposals_follow_loaded_metadata() {
        use crate::backend_selection::DeviceType;
        use crate::converters::{CoremlMlProgramConverter, GraphConverter};
        use crate::graph::{GraphInfo, Operand, OperandKind};
        use crate::operators::Operation;

        let descriptor = descriptor(&[3]);
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    name: Some("input".into()),
                    kind: OperandKind::Input,
                    descriptor: descriptor.clone(),
                },
                Operand {
                    name: Some("result".into()),
                    kind: OperandKind::Output,
                    descriptor: descriptor.clone(),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![Operation::Relu {
                input: 0,
                options: None,
                outputs: vec![1],
            }],
            ..Default::default()
        };
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let model = compile_model(
            converted.data,
            converted.weights_data,
            DeviceType::Cpu,
            false,
        )
        .unwrap();
        let mut input = CoremlTensorStorage::new(DataType::Float32, 12, true).unwrap();
        let output = CoremlTensorStorage::new(DataType::Float32, 12, true).unwrap();
        let (eligible, dtype, constraint_type, shape_count, allowed) = autoreleasepool(|| unsafe {
            let model_description: *mut Object = msg_send![model.model, modelDescription];
            let descriptions: *mut Object = msg_send![model_description, outputDescriptionsByName];
            let key = nsstring_from_str("result").unwrap();
            let description: *mut Object = msg_send![descriptions, objectForKey: key];
            let constraint: *mut Object = msg_send![description, multiArrayConstraint];
            assert!(!constraint.is_null(), "result must have array metadata");
            let dtype: i64 = msg_send![constraint, dataType];
            let shape_constraint: *mut Object = msg_send![constraint, shapeConstraint];
            let constraint_type: i64 = msg_send![shape_constraint, type];
            let shape_count: usize = if constraint_type == 2 {
                let shapes: *mut Object = msg_send![shape_constraint, enumeratedShapes];
                msg_send![shapes, count]
            } else {
                0
            };
            let CoremlTensorStorage::Native(native) = &output else {
                panic!("native output storage");
            };
            let mut guard = native.lock().unwrap();
            let array = guard.array(&descriptor).unwrap();
            let feature: *mut Object =
                msg_send![class!(MLFeatureValue), featureValueWithMultiArray: array];
            let allowed: bool = msg_send![description, isAllowedValue: feature];
            // Inspect the loaded model independently of the production predicate.
            // Metadata can differ by runtime; an ineligible output must still copy.
            let eligible = dtype == 65568 && constraint_type == 2 && shape_count == 1 && allowed;
            (eligible, dtype, constraint_type, shape_count, allowed)
        });
        let mut statistics = CoremlTensorStatistics::default();
        for _ in 0..4 {
            input.write(bytemuck::cast_slice(&[-1f32, 2., 3.])).unwrap();
            let bind = |storage| CoremlTensorBinding {
                storage,
                descriptor: &descriptor,
            };
            let result = run_coreml_tensors(
                &model,
                &HashMap::from([("input".into(), bind(&input))]),
                &HashMap::from([("result".into(), bind(&output))]),
                true,
                &mut statistics,
            )
            .unwrap();
            assert!(result.is_empty());
            // Mutation after prediction must not change the logical output.
            input.write(bytemuck::cast_slice(&[99f32; 3])).unwrap();
            let mut actual = [0u8; 12];
            output.read(&mut actual).unwrap();
            assert_eq!(
                actual.as_slice(),
                bytemuck::cast_slice::<f32, u8>(&[0f32, 2., 3.])
            );
        }
        assert_eq!(
            statistics.output_backings_requested,
            if eligible { 4 } else { 0 },
            "dtype={dtype}, constraint_type={constraint_type}, shapes={shape_count}, allowed={allowed}"
        );
        assert!(statistics.output_backings_accepted <= statistics.output_backings_requested);
        assert_eq!(
            statistics.output_copy_bytes,
            (4 - statistics.output_backings_accepted) * 12
        );
    }

    #[test]
    fn native_capacity_preserves_bytes_and_rebuilds_active_views() {
        autoreleasepool(|| {
            let mut storage = CoremlTensorStorage::new(DataType::Float32, 8, true).unwrap();
            storage.write(bytemuck::cast_slice(&[1.0f32, 2.0])).unwrap();
            let CoremlTensorStorage::Native(native) = &storage else {
                panic!("native storage");
            };
            {
                let mut guard = native.lock().unwrap();
                assert_eq!(guard.layout.align(), 16);
                assert_eq!(guard.data.as_ptr().addr() % guard.layout.align(), 0);
                let first = guard.array(&descriptor(&[2])).unwrap();
                assert_eq!(first, guard.array(&descriptor(&[2])).unwrap());
            }
            assert!(storage.reserve(16, true).unwrap());
            let mut bytes = [0u8; 16];
            storage.read(&mut bytes).unwrap();
            assert_eq!(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .copied()
                    .map(f32::from_le_bytes)
                    .collect::<Vec<_>>(),
                [1.0, 2.0, 0.0, 0.0]
            );
            assert!(!storage.reserve(4, true).unwrap());
            let CoremlTensorStorage::Native(native) = &storage else {
                panic!("native storage");
            };
            let mut guard = native.lock().unwrap();
            for shape in [&[4][..], &[2, 2], &[1], &[]] {
                guard.array(&descriptor(shape)).unwrap();
                assert_eq!(guard.shape, physical_shape(&descriptor(shape)).unwrap());
            }
            assert!(guard.array(&descriptor(&[0])).is_err());
        });
    }

    #[test]
    fn strided_output_alias_is_gathered_before_destination_is_written() {
        autoreleasepool(|| {
            let mut destination = NativeTensor::new(DataType::Float32, 16).unwrap();
            destination
                .write(bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]))
                .unwrap();
            let mut view = ptr::null_mut();
            let mut error = [0u8; 1024];
            let status = unsafe {
                rustnn_coreml_array_view(
                    destination.data.as_ptr().cast(),
                    [2, 2].as_ptr(),
                    [1, 2].as_ptr(),
                    2,
                    65568,
                    &mut view,
                    error.as_mut_ptr().cast(),
                    error.len(),
                )
            };
            assert_eq!(status, 0, "{}", shim_error_to_string(&error));
            let _view_guard = ReleaseOnDrop(view);
            unsafe {
                copy_output(view, &mut destination, &descriptor(&[2, 2])).unwrap();
            }
            assert_eq!(
                bytemuck::cast_slice::<u8, f32>(destination.bytes(16).unwrap()),
                &[1.0, 3.0, 2.0, 4.0]
            );
        });
    }

    #[test]
    fn overlapping_native_outputs_preserve_int32_bits_and_fp16_widths() {
        autoreleasepool(|| {
            for (dtype, kind, values) in [
                (
                    DataType::Int32,
                    NativeType::Int32,
                    bytemuck::cast_slice(&[i32::MIN, -16_777_217, 16_777_217, i32::MAX]).to_vec(),
                ),
                (
                    DataType::Float16,
                    NativeType::Float16,
                    bytemuck::cast_slice(&[0xc180u16, 0x8000, 0x0001, 0x7bff]).to_vec(),
                ),
            ] {
                let element = kind.element_size();
                let mut destination = NativeTensor::new(dtype, values.len() + element).unwrap();
                destination.write(&values).unwrap();
                let transpose = unsafe { view(destination.data.as_ptr(), &[2, 2], &[1, 2], kind) };
                let mut desc = descriptor(&[2, 2]);
                desc.data_type = dtype;
                unsafe { copy_output(transpose.0, &mut destination, &desc).unwrap() };
                let expected: Vec<u8> = [0, 2, 1, 3]
                    .into_iter()
                    .flat_map(|index| {
                        values[index * element..(index + 1) * element]
                            .iter()
                            .copied()
                    })
                    .collect();
                assert_eq!(destination.bytes(values.len()).unwrap(), expected);

                // A shifted contiguous view overlaps too, but memmove is sufficient.
                let shifted_bytes: Vec<u8> = vec![0xa5; element]
                    .into_iter()
                    .chain(values.iter().copied())
                    .collect();
                destination.write(&shifted_bytes).unwrap();
                let shifted =
                    unsafe { view(destination.data.as_ptr().add(element), &[4], &[1], kind) };
                let mut desc = descriptor(&[4]);
                desc.data_type = dtype;
                unsafe { copy_output(shifted.0, &mut destination, &desc).unwrap() };
                assert_eq!(destination.bytes(values.len()).unwrap(), values);
            }
        });
    }

    #[test]
    fn native_tensor_storage_is_send_and_sync_through_its_mutex() {
        fn assert_traits<T: Send + Sync>() {}
        assert_traits::<CoremlTensorStorage>();
        let mut tensor = CoremlTensorStorage::new(DataType::Int32, 4, true).unwrap();
        tensor.write(&16777217i32.to_le_bytes()).unwrap();
        let bytes = std::thread::spawn(move || {
            let mut bytes = [0; 4];
            tensor.read(&mut bytes).unwrap();
            bytes
        })
        .join()
        .unwrap();
        assert_eq!(i32::from_le_bytes(bytes), 16777217);
    }
}

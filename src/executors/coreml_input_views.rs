//! Bind converter-private flat Half views without changing public WebNN inputs.

#[cfg(any(target_os = "macos", target_os = "ios", test))]
use std::collections::HashSet;
use std::ffi::c_void;
use std::ptr;

use block::ConcreteBlock;
use objc::runtime::{BOOL, NO, Object};
use objc::{class, msg_send, sel, sel_impl};
use serde::Deserialize;

use super::{
    NativeType, ReleaseOnDrop, boundary_error, create_multi_array, multiarray_storage,
    ns_error_to_string, nsarray_to_i64_vec, nsstring_from_str, read_array_storage,
    write_array_storage,
};
use crate::error::GraphError;

#[cfg(any(target_os = "macos", target_os = "ios"))]
use super::nsarray_to_strings;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(crate) const METADATA_KEY: &str = "rustnn.webnn.compact_input_views";

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Binding {
    pub(crate) source: String,
    pub(crate) view: String,
}

#[cfg(any(target_os = "macos", target_os = "ios", test))]
fn parse(json: &str, declared_inputs: &[String]) -> Result<Vec<Binding>, GraphError> {
    let bindings: Vec<Binding> = serde_json::from_str(json).map_err(|error| {
        boundary_error(format!("invalid CoreML compact-input metadata: {error}"))
    })?;
    let mut sources = HashSet::new();
    let mut views = HashSet::new();
    for binding in &bindings {
        if binding.source.is_empty()
            || binding.view.is_empty()
            || binding.source == binding.view
            || !declared_inputs.contains(&binding.source)
            || !declared_inputs.contains(&binding.view)
            || !sources.insert(&binding.source)
            || !views.insert(&binding.view)
        {
            return Err(boundary_error(format!(
                "invalid compact input `{}` from `{}`",
                binding.view, binding.source
            )));
        }
    }
    if sources.iter().any(|source| views.contains(source)) {
        return Err(boundary_error("compact-input bindings cannot chain"));
    }
    Ok(bindings)
}

pub(crate) unsafe fn from_model(model: *mut Object) -> Result<Vec<Binding>, GraphError> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let Some(json) = (unsafe { super::model_metadata_value(model, METADATA_KEY)? }) else {
            return Ok(Vec::new());
        };
        let description: *mut Object = msg_send![model, modelDescription];
        let inputs: *mut Object = msg_send![description, inputDescriptionsByName];
        let keys: *mut Object = msg_send![inputs, allKeys];
        let bindings = parse(&json, &unsafe { nsarray_to_strings(keys) })?;
        for binding in &bindings {
            let source = unsafe { declared_half_input(inputs, &binding.source, false)? };
            let view = unsafe { declared_half_input(inputs, &binding.view, true)? };
            let _ = (source, view);
        }
        Ok(bindings)
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        let _ = model;
        Ok(Vec::new())
    }
}

unsafe fn declared_half_input(
    input_descriptions: *mut Object,
    name: &str,
    flat: bool,
) -> Result<*mut Object, GraphError> {
    let key = unsafe { nsstring_from_str(name)? };
    let description: *mut Object = msg_send![input_descriptions, objectForKey: key];
    if description.is_null() {
        return Err(boundary_error(format!(
            "compact input `{name}` is not declared"
        )));
    }
    let constraint: *mut Object = msg_send![description, multiArrayConstraint];
    if constraint.is_null() {
        return Err(boundary_error(format!(
            "compact input `{name}` is not a multi-array"
        )));
    }
    let code: i64 = msg_send![constraint, dataType];
    if NativeType::from_code(code)? != NativeType::Float16 {
        return Err(boundary_error(format!(
            "compact input `{name}` must have Half storage"
        )));
    }
    if flat {
        let shape: *mut Object = msg_send![constraint, shape];
        if unsafe { nsarray_to_i64_vec(shape)? }.len() != 1 {
            return Err(boundary_error(format!(
                "compact view `{name}` must have rank one"
            )));
        }
    }
    Ok(description)
}

/// Owns the native view through prediction. A borrowed-pointer view's native
/// deallocator block additionally retains the source array until the view itself
/// dies, including if CoreML retains it beyond the caller's dictionary lifetime.
pub(crate) struct OwnedView {
    array: ReleaseOnDrop,
    #[cfg(all(test, target_vendor = "apple"))]
    copied: bool,
}

unsafe fn flat_view(source: *mut Object) -> Result<OwnedView, GraphError> {
    if source.is_null() {
        return Err(boundary_error("compact input source has no native array"));
    }
    let (kind, layout, data) = unsafe { multiarray_storage(source)? };
    if kind != NativeType::Float16 {
        return Err(boundary_error(
            "compact input source must retain Half storage",
        ));
    }
    let count = i64::try_from(layout.count)
        .map_err(|_| boundary_error("compact input element count exceeds native shape range"))?;
    if !layout.contiguous {
        let bytes = unsafe { read_array_storage(data, &layout, kind.element_size()) };
        let array = unsafe { create_multi_array(&[count], kind.code())? };
        let array: *mut Object = msg_send![array, retain];
        let array = ReleaseOnDrop(array);
        let (actual_kind, actual_layout, actual_data) = unsafe { multiarray_storage(array.0)? };
        if actual_kind != kind || actual_layout.count != layout.count {
            return Err(boundary_error(
                "compact input copy allocation has the wrong type or length",
            ));
        }
        unsafe { write_array_storage(actual_data, &actual_layout, kind.element_size(), &bytes)? };
        return Ok(OwnedView {
            array,
            #[cfg(all(test, target_vendor = "apple"))]
            copied: true,
        });
    }

    let retained: *mut Object = msg_send![source, retain];
    let source_owner = ReleaseOnDrop(retained);
    let deallocator = ConcreteBlock::new(move |_data: *mut c_void| {
        // Capture the entire owner, not its raw pointer. Dropping the copied
        // block releases the source after MLMultiArray has finished with it.
        let _ = &source_owner;
    })
    .copy();
    let number: *mut Object = msg_send![class!(NSNumber), numberWithLongLong: count];
    let shape: *mut Object = msg_send![class!(NSArray), arrayWithObject: number];
    let one: *mut Object = msg_send![class!(NSNumber), numberWithLongLong: 1i64];
    let strides: *mut Object = msg_send![class!(NSArray), arrayWithObject: one];
    let mut error: *mut Object = ptr::null_mut();
    let alloc: *mut Object = msg_send![class!(MLMultiArray), alloc];
    let array: *mut Object = msg_send![alloc,
        initWithDataPointer: data.cast::<c_void>()
        shape: shape
        dataType: kind.code()
        strides: strides
        deallocator: &*deallocator
        error: &mut error];
    if array.is_null() {
        return Err(boundary_error(unsafe {
            ns_error_to_string(error, "compact input view init failed")
        }));
    }
    let array = ReleaseOnDrop(array);
    let (actual_kind, actual_layout, actual_data) = unsafe { multiarray_storage(array.0)? };
    if actual_kind != kind
        || actual_layout.count != layout.count
        || !actual_layout.contiguous
        || actual_data != data
    {
        return Err(boundary_error(
            "compact input view did not preserve its storage pointer/type/count",
        ));
    }
    Ok(OwnedView {
        array,
        #[cfg(all(test, target_vendor = "apple"))]
        copied: false,
    })
}

/// Add only declared converter-private views to the input dictionary. Callers
/// keep these guards alive until synchronous prediction has finished.
pub(crate) unsafe fn bind(
    model: *mut Object,
    dictionary: *mut Object,
    bindings: &[Binding],
) -> Result<Vec<OwnedView>, GraphError> {
    if bindings.is_empty() {
        return Ok(Vec::new());
    }
    let description: *mut Object = msg_send![model, modelDescription];
    let descriptions: *mut Object = msg_send![description, inputDescriptionsByName];
    let mut result = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let source_description =
            unsafe { declared_half_input(descriptions, &binding.source, false)? };
        let view_description = unsafe { declared_half_input(descriptions, &binding.view, true)? };
        let source_key = unsafe { nsstring_from_str(&binding.source)? };
        let source_feature: *mut Object = msg_send![dictionary, objectForKey: source_key];
        if source_feature.is_null() {
            return Err(boundary_error(format!(
                "compact input source `{}` is not bound",
                binding.source
            )));
        }
        let allowed: BOOL = msg_send![source_description, isAllowedValue: source_feature];
        if allowed == NO {
            return Err(boundary_error(format!(
                "compact input source `{}` violates its declared type/shape",
                binding.source
            )));
        }
        let source: *mut Object = msg_send![source_feature, multiArrayValue];
        let view = unsafe { flat_view(source)? };
        let feature: *mut Object =
            msg_send![class!(MLFeatureValue), featureValueWithMultiArray: view.array.0];
        let allowed: BOOL = msg_send![view_description, isAllowedValue: feature];
        if allowed == NO {
            return Err(boundary_error(format!(
                "compact view `{}` violates its declared type/shape",
                binding.view
            )));
        }
        let key = unsafe { nsstring_from_str(&binding.view)? };
        let () = msg_send![dictionary, setObject: feature forKey: key];
        result.push(view);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bindings_need_no_native_model() {
        let views = unsafe { bind(ptr::null_mut(), ptr::null_mut(), &[]) }.unwrap();
        assert!(views.is_empty());
    }

    #[test]
    fn compact_metadata_requires_distinct_declared_one_level_bindings() {
        let declared = vec!["x".into(), "x_view".into(), "y".into(), "y_view".into()];
        assert_eq!(
            parse(r#"[{"source":"x","view":"x_view"}]"#, &declared)
                .unwrap()
                .len(),
            1
        );
        for invalid in [
            r#"{}"#,
            r#"[{"source":"x","view":"x_view","unknown":true}]"#,
            r#"[{"source":"missing","view":"x_view"}]"#,
            r#"[{"source":"x","view":"missing"}]"#,
            r#"[{"source":"x","view":"x"}]"#,
            r#"[{"source":"","view":"x_view"}]"#,
            r#"[{"source":"x","view":"x_view"},{"source":"x","view":"y_view"}]"#,
            r#"[{"source":"x","view":"x_view"},{"source":"y","view":"x_view"}]"#,
            r#"[{"source":"x","view":"x_view"},{"source":"x_view","view":"y_view"}]"#,
        ] {
            assert!(parse(invalid, &declared).is_err(), "accepted {invalid}");
        }
    }

    #[cfg(target_os = "macos")]
    fn mixed_views_and_copy_outputs(rows: &[u32], dynamic: bool, native: bool) {
        use crate::backend_selection::DeviceType;
        use crate::converters::{CoremlMlProgramConverter, GraphConverter, coreml_names};
        use crate::graph::{
            ConstantData, DataType, Dimension, DynamicDimension, GraphInfo, Operand,
            OperandDescriptor, OperandKind,
        };
        use crate::operators::Operation;
        use crate::protos::coreml::specification;
        use prost::Message;
        use std::collections::HashMap;

        let descriptor = |data_type, shape| OperandDescriptor {
            data_type,
            shape,
            pending_permutation: vec![],
        };
        let maximum = *rows.iter().max().unwrap();
        let dimension = |name: &str, size| {
            if dynamic {
                Dimension::Dynamic(DynamicDimension {
                    name: name.into(),
                    max_size: size,
                })
            } else {
                Dimension::Static(size)
            }
        };
        let source = descriptor(
            DataType::Float16,
            vec![dimension("rows", maximum), Dimension::Static(2)],
        );
        let flat = descriptor(
            DataType::Float16,
            vec![dimension("flat_values", maximum * 2)],
        );
        let constant = descriptor(DataType::Int32, vec![Dimension::Static(4)]);
        let operand = |name: &str, kind, descriptor| Operand {
            name: Some(name.into()),
            kind,
            descriptor,
        };
        let constant_bytes: Vec<_> = [16_777_217i32, -16_777_217, i32::MIN, i32::MAX]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect();
        let graph = GraphInfo {
            operands: vec![
                operand("tensor", OperandKind::Input, source.clone()),
                operand("flat.view", OperandKind::Input, flat.clone()),
                operand("constant", OperandKind::Constant, constant.clone()),
                operand("copy.source", OperandKind::Output, source.clone()),
                operand("copy.constant", OperandKind::Output, constant.clone()),
                operand("negated.view", OperandKind::Output, flat),
            ],
            input_operands: vec![0, 1],
            output_operands: vec![3, 4, 5],
            constant_operand_ids_to_handles: HashMap::from([(
                2,
                ConstantData {
                    data: constant_bytes.clone(),
                    label: None,
                },
            )]),
            operations: vec![
                Operation::Identity {
                    input: 0,
                    options: None,
                    outputs: vec![3],
                },
                Operation::Identity {
                    input: 2,
                    options: None,
                    outputs: vec![4],
                },
                Operation::Neg {
                    input: 1,
                    options: None,
                    outputs: vec![5],
                },
            ],
            ..Default::default()
        };
        let before = serde_json::to_vec(&graph).unwrap();
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        assert_eq!(serde_json::to_vec(&graph).unwrap(), before);
        let mut model = specification::Model::decode(converted.data.as_slice()).unwrap();
        let metadata = &mut model
            .description
            .as_mut()
            .unwrap()
            .metadata
            .as_mut()
            .unwrap()
            .user_defined;
        metadata.insert(
            METADATA_KEY.into(),
            serde_json::json!([{
                "source": coreml_names::encode("tensor"),
                "view": coreml_names::encode("flat.view"),
            }])
            .to_string(),
        );
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            let compiled = super::super::compile_model(
                model.encode_to_vec(),
                converted.weights_data.clone(),
                policy,
                false,
            )
            .unwrap();
            assert_eq!(compiled.aliases.compact_input_views.len(), 1);
            assert_eq!(compiled.aliases.passthroughs.len(), 1);
            assert_eq!(compiled.aliases.constant_copies.len(), 1);
            let mut input_storage = super::super::CoremlTensorStorage::new(
                DataType::Float16,
                maximum as usize * 4,
                true,
            )
            .unwrap();
            let mut source_output = super::super::CoremlTensorStorage::new(
                DataType::Float16,
                maximum as usize * 4,
                true,
            )
            .unwrap();
            let constant_output =
                super::super::CoremlTensorStorage::new(DataType::Int32, constant_bytes.len(), true)
                    .unwrap();
            let arithmetic_output = super::super::CoremlTensorStorage::new(
                DataType::Float16,
                maximum as usize * 4,
                true,
            )
            .unwrap();
            for (iteration, &rows) in rows.iter().enumerate() {
                let actual = descriptor(
                    DataType::Float16,
                    vec![Dimension::Static(rows), Dimension::Static(2)],
                );
                let original: Vec<_> = [0x3c00u16, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600]
                    .into_iter()
                    .take(rows as usize * 2)
                    .map(|bits| bits + iteration as u16 * 0x10)
                    .flat_map(u16::to_le_bytes)
                    .collect();
                let negated: Vec<_> = original
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .flat_map(|&bits| (u16::from_le_bytes(bits) ^ 0x8000).to_le_bytes())
                    .collect();
                let outputs = HashMap::from([
                    // The byte executor accepts graph bounds for proven copies;
                    // its returned bytes must still match the actual input.
                    ("copy.source".into(), source.clone()),
                    ("copy.constant".into(), constant.clone()),
                    (
                        "negated.view".into(),
                        descriptor(DataType::Float16, vec![Dimension::Static(rows * 2)]),
                    ),
                ]);
                if native {
                    let active_outputs: HashMap<String, OperandDescriptor> = HashMap::from([
                        ("copy.source".into(), actual.clone()),
                        ("copy.constant".into(), constant.clone()),
                        (
                            "negated.view".into(),
                            descriptor(DataType::Float16, vec![Dimension::Static(rows * 2)]),
                        ),
                    ]);
                    // Retain the same allocations across actual 1 -> 3 -> 1
                    // shapes. Both backing modes must bind the private view.
                    for output_backings in [false, true] {
                        input_storage.write(&original).unwrap();
                        let mut statistics =
                            crate::mlcontextoptions::CoremlTensorStatistics::default();
                        let result = super::super::run_coreml_tensors(
                            &compiled,
                            &HashMap::from([(
                                "tensor".into(),
                                super::super::CoremlTensorBinding {
                                    storage: &input_storage,
                                    descriptor: &actual,
                                },
                            )]),
                            &HashMap::from([
                                (
                                    "copy.source".into(),
                                    super::super::CoremlTensorBinding {
                                        storage: &source_output,
                                        descriptor: &active_outputs["copy.source"],
                                    },
                                ),
                                (
                                    "copy.constant".into(),
                                    super::super::CoremlTensorBinding {
                                        storage: &constant_output,
                                        descriptor: &active_outputs["copy.constant"],
                                    },
                                ),
                                (
                                    "negated.view".into(),
                                    super::super::CoremlTensorBinding {
                                        storage: &arithmetic_output,
                                        descriptor: &active_outputs["negated.view"],
                                    },
                                ),
                            ]),
                            output_backings,
                            &mut statistics,
                        )
                        .unwrap();
                        assert!(result.is_empty());
                        assert_eq!(statistics.native_input_bindings, 1);
                        let mut input_after = vec![0; original.len()];
                        input_storage.read(&mut input_after).unwrap();
                        assert_eq!(
                            input_after, original,
                            "native input changed during prediction"
                        );
                        input_storage.write(&vec![0; original.len()]).unwrap();
                        for (storage, expected) in [
                            (&source_output, &original),
                            (&constant_output, &constant_bytes),
                            (&arithmetic_output, &negated),
                        ] {
                            let mut bytes = vec![0; expected.len()];
                            storage.read(&mut bytes).unwrap();
                            assert_eq!(&bytes, expected, "{policy:?}, rows={rows}");
                        }
                        source_output.write(&vec![0; original.len()]).unwrap();
                        let mut constant_result = vec![0; constant_bytes.len()];
                        constant_output.read(&mut constant_result).unwrap();
                        assert_eq!(constant_result, constant_bytes);
                        let mut arithmetic_result = vec![0; negated.len()];
                        arithmetic_output.read(&mut arithmetic_result).unwrap();
                        assert_eq!(arithmetic_result, negated);
                    }
                    continue;
                }
                let mut bytes = original.clone();
                let mut result = super::super::run_coreml_bytes(
                    &compiled,
                    &HashMap::from([(
                        "tensor".into(),
                        super::super::CoremlByteInput {
                            data: &bytes,
                            descriptor: &actual,
                        },
                    )]),
                    &outputs,
                )
                .unwrap();
                bytes.fill(0);
                assert_eq!(result["copy.source"], original, "{policy:?}, rows={rows}");
                assert_eq!(result["copy.constant"], constant_bytes);
                assert_eq!(result["negated.view"], negated);
                result.get_mut("copy.source").unwrap().fill(0);
                assert_eq!(result["copy.constant"], constant_bytes);
                assert_eq!(result["negated.view"], negated);
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn compact_views_coexist_with_proven_input_and_constant_outputs() {
        mixed_views_and_copy_outputs(&[3, 3], false, false);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_reuse_binds_compact_views_with_proven_outputs_and_backing_modes() {
        mixed_views_and_copy_outputs(&[3, 3], false, true);
    }

    #[cfg(all(target_os = "macos", feature = "dynamic-inputs"))]
    #[test]
    fn compact_views_and_proven_outputs_keep_actual_grow_shrink_extents() {
        mixed_views_and_copy_outputs(&[1, 3, 1], true, false);
    }

    #[cfg(all(target_os = "macos", feature = "dynamic-inputs"))]
    #[test]
    fn native_reuse_compact_views_keep_actual_grow_shrink_extents() {
        mixed_views_and_copy_outputs(&[1, 3, 1], true, true);
    }

    #[cfg(target_vendor = "apple")]
    unsafe fn strided_source(data: &mut [u8], shape: &[i64], strides: &[i64]) -> *mut Object {
        let numbers = |values: &[i64]| {
            let values: Vec<*mut Object> = values
                .iter()
                .map(|&value| {
                    let number: *mut Object =
                        msg_send![class!(NSNumber), numberWithLongLong: value];
                    number
                })
                .collect();
            let array: *mut Object =
                msg_send![class!(NSArray), arrayWithObjects: values.as_ptr() count: values.len()];
            array
        };
        let alloc: *mut Object = msg_send![class!(MLMultiArray), alloc];
        let mut error: *mut Object = ptr::null_mut();
        let array: *mut Object = msg_send![alloc,
            initWithDataPointer: data.as_mut_ptr().cast::<c_void>()
            shape: numbers(shape)
            dataType: NativeType::Float16.code()
            strides: numbers(strides)
            deallocator: ptr::null_mut::<Object>()
            error: &mut error];
        assert!(!array.is_null(), "{}", unsafe {
            ns_error_to_string(error, "strided source init failed")
        });
        let array: *mut Object = msg_send![array, autorelease];
        array
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn contiguous_half_view_retains_source_after_its_pool_drains() {
        let expected: Vec<u8> = [0x8000_u16, 1, 0x8001, 0x7e01, 0x7c00, 0xfc00]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        let view = objc::rc::autoreleasepool(|| unsafe {
            let source = create_multi_array(&[1, 2, 1, 3], NativeType::Float16.code()).unwrap();
            let (kind, layout, pointer) = multiarray_storage(source).unwrap();
            write_array_storage(pointer, &layout, kind.element_size(), &expected).unwrap();
            let view = flat_view(source).unwrap();
            let (_, flat, flat_pointer) = multiarray_storage(view.array.0).unwrap();
            assert!(!view.copied);
            assert_eq!(pointer, flat_pointer);
            assert_eq!(flat.count, 6);
            view
        });
        // The source's +0 pool reference is gone. Only the native view's
        // copied deallocator block owns it while its bytes are read here.
        objc::rc::autoreleasepool(|| unsafe {
            let (kind, layout, pointer) = multiarray_storage(view.array.0).unwrap();
            assert_eq!(
                read_array_storage(pointer, &layout, kind.element_size()),
                expected
            );
        });
        drop(view);
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn padded_half_sources_copy_exact_bits_and_not_padding() {
        let expected: Vec<u8> = [0x8000_u16, 1, 0x8001, 0x7e01, 0x7c00, 0xfc00]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        for (shape, strides, length) in [
            (vec![2, 3], vec![8, 2], 28),
            (vec![1, 6, 1, 1], vec![512, 32, 32, 1], 324),
        ] {
            let mut source_bytes = vec![0x55; length];
            let view = objc::rc::autoreleasepool(|| unsafe {
                let source = strided_source(&mut source_bytes, &shape, &strides);
                let (kind, layout, pointer) = multiarray_storage(source).unwrap();
                assert!(!layout.contiguous);
                write_array_storage(pointer, &layout, kind.element_size(), &expected).unwrap();
                let before = source_bytes.clone();
                let view = flat_view(source).unwrap();
                let (_, flat, flat_pointer) = multiarray_storage(view.array.0).unwrap();
                assert!(view.copied);
                assert_ne!(pointer, flat_pointer);
                assert!(flat.contiguous);
                assert_eq!(source_bytes, before);
                view
            });
            source_bytes.fill(0);
            drop(source_bytes);
            objc::rc::autoreleasepool(|| unsafe {
                let (kind, layout, pointer) = multiarray_storage(view.array.0).unwrap();
                assert_eq!(
                    read_array_storage(pointer, &layout, kind.element_size()),
                    expected
                );
            });
        }
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn compact_views_reject_other_storage_and_null_sources() {
        objc::rc::autoreleasepool(|| unsafe {
            assert!(flat_view(ptr::null_mut()).is_err());
            let source = create_multi_array(&[2], NativeType::Float32.code()).unwrap();
            assert!(flat_view(source).is_err());
        });
    }
}

// SPDX-FileCopyrightText: 2026 Shubham Gupta <shubhamg13.work@gmail.com>
//
// SPDX-License-Identifier: Apache-2

use std::ffi::c_char;

// Raw Function pointer types

pub(crate) type FnVoidPtr = unsafe extern "C" fn() -> *mut std::ffi::c_void;
pub(crate) type FnPtrRetI32 = unsafe extern "C" fn(*mut std::ffi::c_void) -> i32;
pub(crate) type FnPtrPtrRetI32 =
    unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) -> i32;
pub(crate) type FnDestroy = unsafe extern "C" fn(*mut std::ffi::c_void);
pub(crate) type FnGraphIO =
    unsafe extern "C" fn(*mut std::ffi::c_void, *const *mut std::ffi::c_void, i32) -> i32;
pub(crate) type FnGraphCreate = unsafe extern "C" fn(*const c_char) -> *mut std::ffi::c_void;
pub(crate) type FnOpSetInput =
    unsafe extern "C" fn(*mut std::ffi::c_void, *const c_char, *mut std::ffi::c_void) -> i32;
pub(crate) type FnBuild = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    *mut std::ffi::c_void,
    *mut std::ffi::c_void,
    *mut CannModelBuffer,
) -> i32;
pub(crate) type FnBufFree = unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void);
pub(crate) type FnDescCreate =
    unsafe extern "C" fn(*const c_char, i32, i32, i32, i32) -> *mut std::ffi::c_void;
pub(crate) type FnDescSetBuf =
    unsafe extern "C" fn(*mut std::ffi::c_void, *const std::ffi::c_void, u32) -> i32;
pub(crate) type FnProcess = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    *mut std::ffi::c_void,
    *const *mut std::ffi::c_void,
    i32,
    *const *mut std::ffi::c_void,
    i32,
    u32,
    *mut i32,
) -> i32;
pub(crate) type FnGetBuf = unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut std::ffi::c_void;
pub(crate) type FnTensorInit =
    unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void, i32) -> i32;
pub(crate) type FnTensorInitData = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    *const std::ffi::c_void,
    *mut std::ffi::c_void,
    i32,
) -> i32;
pub(crate) type FnDimCreate = unsafe extern "C" fn(*const u32, i32) -> *mut std::ffi::c_void;
pub(crate) type FnCtxPara =
    unsafe extern "C" fn(*mut std::ffi::c_void, *const c_char, *const c_char) -> i32;
pub(crate) type FnSetData =
    unsafe extern "C" fn(*mut std::ffi::c_void, *const std::ffi::c_void, u32) -> i32;
pub(crate) type FnDynCreate =
    unsafe extern "C" fn(*mut std::ffi::c_void, *const c_char, u32) -> i32;
pub(crate) type FnSetDynInput =
    unsafe extern "C" fn(*mut std::ffi::c_void, *const c_char, u32, *mut std::ffi::c_void) -> i32;
pub(crate) type FnSetAttrInt64List =
    unsafe extern "C" fn(*mut std::ffi::c_void, *const c_char, *const i64, i32) -> i32;
pub(crate) type FnShapeCreate = unsafe extern "C" fn(*const i64, i32) -> *mut std::ffi::c_void;
pub(crate) type FnTensorDescCreate =
    unsafe extern "C" fn(*mut std::ffi::c_void, i32, i32) -> *mut std::ffi::c_void;
pub(crate) type FnCreateBuff =
    unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void, *mut CannModelBuffer) -> i32;
pub(crate) type FnNetOutputCreate =
    unsafe extern "C" fn(*const c_char, i32) -> *mut std::ffi::c_void;
pub(crate) type FnLoad =
    unsafe extern "C" fn(*mut std::ffi::c_void, *const *mut std::ffi::c_void, i32) -> i32;
pub(crate) type FnOperatorCreate =
    unsafe extern "C" fn(*const c_char, *const c_char) -> *mut std::ffi::c_void;
pub(crate) type FnSetAttrTensorRaw = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    *const c_char,
    *const std::ffi::c_void,
    u32,
    *const i64,
    i32,
    i32,
) -> i32;
pub(crate) type FnSetAttrTensorRawFormat = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    *const c_char,
    *const std::ffi::c_void,
    u32,
    *const i64,
    i32,
    i32,
    i32,
) -> i32;
pub(crate) type FnSetAttrInt64 =
    unsafe extern "C" fn(*mut std::ffi::c_void, *const c_char, i64) -> i32;
pub(crate) type FnSetAttrFloat =
    unsafe extern "C" fn(*mut std::ffi::c_void, *const c_char, f32) -> i32;
pub(crate) type FnSetAttrString =
    unsafe extern "C" fn(*mut std::ffi::c_void, *const c_char, *const c_char) -> i32;
pub(crate) type FnSetInputShapes =
    unsafe extern "C" fn(*mut std::ffi::c_void, *const *const i64, *const i32, i32) -> i32;
pub(crate) type FnSetDeviceOrder =
    unsafe extern "C" fn(*mut std::ffi::c_void, *const i32, i32) -> i32;

// Model buffer returned by `cann_build_model`
#[repr(C)]
pub(crate) struct CannModelBuffer {
    pub data: *mut std::ffi::c_void,
    pub length: u32,
}

/// Tensor descriptor passed between the backend and the shim.
#[allow(dead_code)]
pub(crate) struct CannTensorDesc {
    pub data: Vec<u8>,
    pub shape: Vec<u32>,
    pub dtype: i32,
}

// Raw function pointers to the CANN shim's exported symbols.
#[allow(dead_code)]
pub(crate) struct CannShim {
    // Dispatch: manager lifecycle
    pub manager_create: FnVoidPtr,
    pub manager_init: FnPtrRetI32,
    pub manager_load: FnLoad,
    pub manager_process: FnProcess,
    pub manager_destroy: FnDestroy,

    // Dispatch: model descriptor
    pub model_desc_create: FnDescCreate,
    pub model_desc_set_buffer: FnDescSetBuf,
    pub model_desc_destroy: FnDestroy,

    // Dispatch: tensor I/O
    pub tensor_create: FnVoidPtr,
    pub tensor_destroy: FnDestroy,
    pub tensor_init: FnTensorInit,
    pub tensor_init_with_data: FnTensorInitData,
    pub tensor_set_data: FnSetData,
    pub tensor_get_buffer: FnGetBuf,
    pub tensor_dim_create_nd: FnDimCreate,
    pub tensor_dim_destroy: FnDestroy,

    // Dispatch: context
    pub context_create: FnVoidPtr,
    pub context_destroy: FnDestroy,
    pub context_set_para: FnCtxPara,

    // Converter: graph building
    pub graph_create: FnGraphCreate,
    pub graph_add_op: FnPtrPtrRetI32,
    pub graph_set_inputs: FnGraphIO,
    pub graph_set_outputs: FnGraphIO,
    pub graph_destroy: FnDestroy,
    pub graph_is_valid: FnPtrRetI32,

    // Converter: operators
    pub op_data_with_name: FnGraphCreate,
    pub op_const_with_name: FnGraphCreate,
    pub op_net_output_with_name: FnNetOutputCreate,
    pub operator_set_input: FnOpSetInput,
    pub operator_update_input_desc: FnOpSetInput,
    pub operator_create_dynamic_input: FnDynCreate,
    pub operator_set_dynamic_input_by_index: FnSetDynInput,
    pub operator_create_dynamic_output: FnDynCreate,
    pub operator_set_attr_int64_list: FnSetAttrInt64List,
    pub operator_set_attr_tensor_raw: FnSetAttrTensorRaw,
    pub operator_set_attr_tensor_raw_format: FnSetAttrTensorRawFormat,
    pub operator_set_attr_int64: FnSetAttrInt64,
    pub operator_set_attr_float: FnSetAttrFloat,
    pub operator_set_attr_string: FnSetAttrString,
    pub operator_destroy: FnDestroy,
    pub operator_create: FnOperatorCreate,
    pub operator_create_registered: FnOperatorCreate,

    // Converter: tensor descriptors
    pub shape_create: FnShapeCreate,
    pub shape_destroy: FnDestroy,
    pub tensor_desc_create: FnTensorDescCreate,
    pub tensor_desc_destroy: FnDestroy,

    // Converter: model compile
    pub ir_build_create: FnVoidPtr,
    pub ir_build_destroy: FnDestroy,
    pub model_create: FnVoidPtr,
    pub model_create_with_name: FnGraphCreate,
    pub model_set_graph: FnPtrPtrRetI32,
    pub model_destroy: FnDestroy,
    pub model_create_buff_default: FnCreateBuff,
    pub build_model: FnBuild,
    pub model_buffer_destroy: FnBufFree,

    // Converter: build options
    pub build_options_create: FnVoidPtr,
    pub build_options_set_input_shapes: FnSetInputShapes,
    pub build_options_set_device_order: FnSetDeviceOrder,
}

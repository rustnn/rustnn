/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! CANN shim — direct FFI bridge to the C++ adapter library.
//!
//! With `cann-runtime`: the adapter is compiled and linked via `build.rs`.
//! All symbols are resolved at build time — no runtime `.so` dependency.
//!
//! Without `cann-runtime`: `get_shim()` returns `None`, all dispatch
//! returns `Err` (mock mode for CI).

use std::sync::LazyLock;

use crate::error::{Error, Result};
use crate::executors::cann_shim_types::CannShim;

pub(crate) use crate::executors::cann_shim_types::{CannModelBuffer, CannTensorDesc};

#[cfg(feature = "cann-runtime")]
unsafe extern "C" {
    // Dispatch: manager lifecycle
    fn cann_model_manager_create() -> *mut std::ffi::c_void;
    fn cann_model_manager_init(manager: *mut std::ffi::c_void) -> i32;
    fn cann_model_manager_load(
        manager: *mut std::ffi::c_void,
        descs: *const *mut std::ffi::c_void,
        count: i32,
    ) -> i32;
    fn cann_model_manager_process(
        manager: *mut std::ffi::c_void,
        context: *mut std::ffi::c_void,
        inputs: *const *mut std::ffi::c_void,
        input_count: i32,
        outputs: *const *mut std::ffi::c_void,
        output_count: i32,
        timeout_ms: u32,
        stamp: *mut i32,
    ) -> i32;
    fn cann_model_manager_destroy(manager: *mut std::ffi::c_void);

    // Dispatch: model descriptor
    fn cann_model_desc_create(
        name: *const std::ffi::c_char,
        frequency: i32,
        framework: i32,
        model_type: i32,
        device_type: i32,
    ) -> *mut std::ffi::c_void;
    fn cann_model_desc_set_model_buffer(
        desc: *mut std::ffi::c_void,
        data: *const std::ffi::c_void,
        length: u32,
    ) -> i32;
    fn cann_model_desc_destroy(desc: *mut std::ffi::c_void);

    // Dispatch: tensor I/O
    fn cann_io_tensor_create() -> *mut std::ffi::c_void;
    fn cann_io_tensor_destroy(tensor: *mut std::ffi::c_void);
    fn cann_io_tensor_init(
        tensor: *mut std::ffi::c_void,
        dim: *mut std::ffi::c_void,
        dtype: i32,
    ) -> i32;
    fn cann_io_tensor_init_with_data(
        tensor: *mut std::ffi::c_void,
        data: *const std::ffi::c_void,
        dim: *mut std::ffi::c_void,
        dtype: i32,
    ) -> i32;
    fn cann_io_tensor_set_data(
        tensor: *mut std::ffi::c_void,
        data: *const std::ffi::c_void,
        size: u32,
    ) -> i32;
    fn cann_io_tensor_get_buffer(tensor: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    fn cann_io_tensor_dim_create_nd(dims: *const u32, dim_count: i32) -> *mut std::ffi::c_void;
    fn cann_io_tensor_dim_destroy(dim: *mut std::ffi::c_void);

    // Dispatch: context
    fn cann_context_create() -> *mut std::ffi::c_void;
    fn cann_context_destroy(context: *mut std::ffi::c_void);
    fn cann_context_set_para(
        context: *mut std::ffi::c_void,
        name: *const std::ffi::c_char,
        value: *const std::ffi::c_char,
    ) -> i32;

    // Converter: graph building
    fn cann_graph_create(name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    fn cann_graph_add_op(graph: *mut std::ffi::c_void, op: *mut std::ffi::c_void) -> i32;
    fn cann_graph_set_inputs(
        graph: *mut std::ffi::c_void,
        inputs: *const *mut std::ffi::c_void,
        count: i32,
    ) -> i32;
    fn cann_graph_set_outputs(
        graph: *mut std::ffi::c_void,
        outputs: *const *mut std::ffi::c_void,
        count: i32,
    ) -> i32;
    fn cann_graph_destroy(graph: *mut std::ffi::c_void);
    fn cann_graph_is_valid(graph: *mut std::ffi::c_void) -> i32;

    // Converter: operators
    fn cann_op_data_with_name(name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    fn cann_op_const_with_name(name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    fn cann_op_net_output_with_name(
        name: *const std::ffi::c_char,
        input_count: i32,
    ) -> *mut std::ffi::c_void;
    fn cann_operator_set_input(
        op: *mut std::ffi::c_void,
        name: *const std::ffi::c_char,
        input_op: *mut std::ffi::c_void,
    ) -> i32;
    fn cann_operator_update_input_desc(
        op: *mut std::ffi::c_void,
        name: *const std::ffi::c_char,
        desc: *mut std::ffi::c_void,
    ) -> i32;
    fn cann_operator_create_dynamic_input(
        op: *mut std::ffi::c_void,
        name: *const std::ffi::c_char,
        count: u32,
    ) -> i32;
    fn cann_operator_set_dynamic_input_by_index(
        op: *mut std::ffi::c_void,
        name: *const std::ffi::c_char,
        index: u32,
        input_op: *mut std::ffi::c_void,
    ) -> i32;
    fn cann_operator_create_dynamic_output(
        op: *mut std::ffi::c_void,
        name: *const std::ffi::c_char,
        count: u32,
    ) -> i32;
    fn cann_operator_set_attr_int64_list(
        op: *mut std::ffi::c_void,
        name: *const std::ffi::c_char,
        values: *const i64,
        count: i32,
    ) -> i32;
    fn cann_operator_set_attr_tensor_raw(
        op: *mut std::ffi::c_void,
        name: *const std::ffi::c_char,
        data: *const std::ffi::c_void,
        size: u32,
        shape: *const i64,
        shape_count: i32,
        dtype: i32,
    ) -> i32;
    fn cann_operator_set_attr_tensor_raw_format(
        op: *mut std::ffi::c_void,
        name: *const std::ffi::c_char,
        data: *const std::ffi::c_void,
        size: u32,
        shape: *const i64,
        shape_count: i32,
        dtype: i32,
        format: i32,
    ) -> i32;
    fn cann_operator_set_attr_int64(
        op: *mut std::ffi::c_void,
        name: *const std::ffi::c_char,
        value: i64,
    ) -> i32;
    fn cann_operator_set_attr_float(
        op: *mut std::ffi::c_void,
        name: *const std::ffi::c_char,
        value: f32,
    ) -> i32;
    fn cann_operator_set_attr_string(
        op: *mut std::ffi::c_void,
        name: *const std::ffi::c_char,
        value: *const std::ffi::c_char,
    ) -> i32;
    fn cann_operator_destroy(op: *mut std::ffi::c_void);
    fn cann_operator_create(
        type_name: *const std::ffi::c_char,
        name: *const std::ffi::c_char,
    ) -> *mut std::ffi::c_void;
    fn cann_operator_create_registered(
        type_name: *const std::ffi::c_char,
        name: *const std::ffi::c_char,
    ) -> *mut std::ffi::c_void;

    // Converter: tensor descriptors
    fn cann_shape_create(dims: *const i64, dim_count: i32) -> *mut std::ffi::c_void;
    fn cann_shape_destroy(shape: *mut std::ffi::c_void);
    fn cann_tensor_desc_create(
        shape: *mut std::ffi::c_void,
        format: i32,
        dtype: i32,
    ) -> *mut std::ffi::c_void;
    fn cann_tensor_desc_destroy(desc: *mut std::ffi::c_void);

    // Converter: model compile
    fn cann_hiai_ir_build_create() -> *mut std::ffi::c_void;
    fn cann_hiai_ir_build_destroy(build: *mut std::ffi::c_void);
    fn cann_model_create() -> *mut std::ffi::c_void;
    fn cann_model_create_with_name(name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    fn cann_model_set_graph(model: *mut std::ffi::c_void, graph: *mut std::ffi::c_void) -> i32;
    fn cann_model_destroy(model: *mut std::ffi::c_void);
    fn cann_model_create_buff_default(
        build: *mut std::ffi::c_void,
        model: *mut std::ffi::c_void,
        buffer: *mut CannModelBuffer,
    ) -> i32;
    fn cann_build_model(
        build: *mut std::ffi::c_void,
        model: *mut std::ffi::c_void,
        options: *mut std::ffi::c_void,
        buffer: *mut CannModelBuffer,
    ) -> i32;
    fn cann_model_buffer_destroy(build: *mut std::ffi::c_void, buffer: *mut std::ffi::c_void);

    fn cann_build_options_create() -> *mut std::ffi::c_void;
    fn cann_build_options_set_input_shapes(
        options: *mut std::ffi::c_void,
        shapes: *const *const i64,
        shape_counts: *const i32,
        num_inputs: i32,
    ) -> i32;
    fn cann_build_options_set_device_order(
        options: *mut std::ffi::c_void,
        devices: *const i32,
        device_count: i32,
    ) -> i32;
}

static SHIM: LazyLock<Option<CannShim>> = LazyLock::new(|| CannShim::load().ok());

pub(crate) fn get_shim() -> Option<&'static CannShim> {
    SHIM.as_ref()
}

impl CannShim {
    #[cfg(feature = "cann-runtime")]
    fn load() -> Result<Self> {
        Ok(Self {
            manager_create: cann_model_manager_create,
            manager_init: cann_model_manager_init,
            manager_load: cann_model_manager_load,
            manager_process: cann_model_manager_process,
            manager_destroy: cann_model_manager_destroy,
            model_desc_create: cann_model_desc_create,
            model_desc_set_buffer: cann_model_desc_set_model_buffer,
            model_desc_destroy: cann_model_desc_destroy,
            tensor_create: cann_io_tensor_create,
            tensor_destroy: cann_io_tensor_destroy,
            tensor_init: cann_io_tensor_init,
            tensor_init_with_data: cann_io_tensor_init_with_data,
            tensor_set_data: cann_io_tensor_set_data,
            tensor_get_buffer: cann_io_tensor_get_buffer,
            tensor_dim_create_nd: cann_io_tensor_dim_create_nd,
            tensor_dim_destroy: cann_io_tensor_dim_destroy,
            context_create: cann_context_create,
            context_destroy: cann_context_destroy,
            context_set_para: cann_context_set_para,
            graph_create: cann_graph_create,
            graph_add_op: cann_graph_add_op,
            graph_set_inputs: cann_graph_set_inputs,
            graph_set_outputs: cann_graph_set_outputs,
            graph_destroy: cann_graph_destroy,
            graph_is_valid: cann_graph_is_valid,
            op_data_with_name: cann_op_data_with_name,
            op_const_with_name: cann_op_const_with_name,
            op_net_output_with_name: cann_op_net_output_with_name,
            operator_set_input: cann_operator_set_input,
            operator_update_input_desc: cann_operator_update_input_desc,
            operator_create_dynamic_input: cann_operator_create_dynamic_input,
            operator_set_dynamic_input_by_index: cann_operator_set_dynamic_input_by_index,
            operator_create_dynamic_output: cann_operator_create_dynamic_output,
            operator_set_attr_int64_list: cann_operator_set_attr_int64_list,
            operator_set_attr_tensor_raw: cann_operator_set_attr_tensor_raw,
            operator_set_attr_tensor_raw_format: cann_operator_set_attr_tensor_raw_format,
            operator_set_attr_int64: cann_operator_set_attr_int64,
            operator_set_attr_float: cann_operator_set_attr_float,
            operator_set_attr_string: cann_operator_set_attr_string,
            operator_destroy: cann_operator_destroy,
            operator_create: cann_operator_create,
            operator_create_registered: cann_operator_create_registered,
            shape_create: cann_shape_create,
            shape_destroy: cann_shape_destroy,
            tensor_desc_create: cann_tensor_desc_create,
            tensor_desc_destroy: cann_tensor_desc_destroy,
            ir_build_create: cann_hiai_ir_build_create,
            ir_build_destroy: cann_hiai_ir_build_destroy,
            model_create: cann_model_create,
            model_create_with_name: cann_model_create_with_name,
            model_set_graph: cann_model_set_graph,
            model_destroy: cann_model_destroy,
            model_create_buff_default: cann_model_create_buff_default,
            build_model: cann_build_model,
            model_buffer_destroy: cann_model_buffer_destroy,
            build_options_create: cann_build_options_create,
            build_options_set_input_shapes: cann_build_options_set_input_shapes,
            build_options_set_device_order: cann_build_options_set_device_order,
        })
    }

    #[cfg(not(feature = "cann-runtime"))]
    fn load() -> Result<Self> {
        Err(Error::GraphDispatchError {
            source: "CANN shim not available (mock mode)".into(),
        })
    }
}

/// Dispatches a graph using the CANN shim.
pub(crate) fn cann_dispatch(
    model_bytes: &[u8],
    inputs: &[CannTensorDesc],
    outputs: &mut [CannTensorDesc],
) -> Result<()> {
    use std::ffi::CString;

    let Some(shim) = get_shim() else {
        return Err(Error::GraphDispatchError {
            source: "CANN shim not available".into(),
        });
    };

    // 1. Init model manager
    let manager = unsafe { (shim.manager_create)() };
    if manager.is_null() {
        return Err(Error::GraphDispatchError {
            source: "cann_model_manager_create failed".into(),
        });
    }
    let status = unsafe { (shim.manager_init)(manager) };
    if status != 0 {
        return Err(Error::GraphDispatchError {
            source: format!("cann_model_manager_init failed: {status}").into(),
        });
    }

    // 2. Load model
    let name = CString::new("webnn_model").unwrap();
    let descriptor = unsafe {
        (shim.model_desc_create)(
            name.as_ptr(),
            3, // AiModelDescription_Frequency_HIGH
            0, // HIAI_FRAMEWORK_NONE
            0, // HIAI_MODELTYPE_OFFLINE
            0, // AiModelDescription_DeviceType_NPU
        )
    };
    if descriptor.is_null() {
        unsafe { (shim.manager_destroy)(manager) };
        return Err(Error::GraphDispatchError {
            source: "cann_model_desc_create failed".into(),
        });
    }
    if unsafe {
        (shim.model_desc_set_buffer)(
            descriptor,
            model_bytes.as_ptr() as *const _,
            model_bytes.len() as u32,
        )
    } != 0
    {
        unsafe { (shim.manager_destroy)(manager) };
        return Err(Error::GraphDispatchError {
            source: "cann_model_desc_set_buffer failed".into(),
        });
    }
    if unsafe { (shim.manager_load)(manager, &descriptor, 1) } != 0 {
        unsafe { (shim.manager_destroy)(manager) };
        return Err(Error::GraphDispatchError {
            source: "cann_model_manager_load failed".into(),
        });
    }

    // 3. Build input tensors
    let mut input_handles: Vec<*mut std::ffi::c_void> = Vec::new();
    let mut dimension_handles: Vec<*mut std::ffi::c_void> = Vec::new();

    for input in inputs.iter() {
        let dimension =
            unsafe { (shim.tensor_dim_create_nd)(input.shape.as_ptr(), input.shape.len() as i32) };
        if dimension.is_null() {
            return Err(Error::GraphDispatchError {
                source: "cann_io_tensor_dim_create_nd failed".into(),
            });
        }
        dimension_handles.push(dimension);

        let tensor = unsafe { (shim.tensor_create)() };
        if tensor.is_null() {
            return Err(Error::GraphDispatchError {
                source: "cann_io_tensor_create failed".into(),
            });
        }
        let status = unsafe { (shim.tensor_init)(tensor, dimension, input.dtype) };
        if status != 0 {
            return Err(Error::GraphDispatchError {
                source: format!("cann_io_tensor_init failed: {status}").into(),
            });
        }
        let status = unsafe {
            (shim.tensor_set_data)(
                tensor,
                input.data.as_ptr() as *const _,
                input.data.len() as u32,
            )
        };
        if status != 0 {
            return Err(Error::GraphDispatchError {
                source: format!("cann_io_tensor_set_data failed: {status}").into(),
            });
        }
        input_handles.push(tensor);
    }

    // 4. Build output tensors
    let mut output_handles: Vec<*mut std::ffi::c_void> = Vec::new();
    for output in outputs.iter() {
        let dimension = unsafe {
            (shim.tensor_dim_create_nd)(output.shape.as_ptr(), output.shape.len() as i32)
        };
        if dimension.is_null() {
            return Err(Error::GraphDispatchError {
                source: "cann_io_tensor_dim_create_nd failed".into(),
            });
        }
        dimension_handles.push(dimension);

        let tensor = unsafe { (shim.tensor_create)() };
        if tensor.is_null() {
            return Err(Error::GraphDispatchError {
                source: "cann_io_tensor_create failed".into(),
            });
        }
        let status = unsafe { (shim.tensor_init)(tensor, dimension, output.dtype) };
        if status != 0 {
            return Err(Error::GraphDispatchError {
                source: format!("cann_io_tensor_init failed: {status}").into(),
            });
        }
        output_handles.push(tensor);
    }

    // 5. Run inference
    let context = unsafe { (shim.context_create)() };
    let model_name = CString::new("model_name").unwrap();
    let model_val = CString::new("webnn_model").unwrap();
    unsafe { (shim.context_set_para)(context, model_name.as_ptr(), model_val.as_ptr()) };

    let mut stamp: i32 = 0;
    let status = unsafe {
        (shim.manager_process)(
            manager,
            context,
            input_handles.as_ptr(),
            input_handles.len() as i32,
            output_handles.as_ptr(),
            output_handles.len() as i32,
            1000,
            &mut stamp,
        )
    };
    if status != 0 {
        return Err(Error::GraphDispatchError {
            source: format!("cann_model_manager_process failed: {status}").into(),
        });
    }

    // 6. Read outputs back
    for (index, output) in outputs.iter_mut().enumerate() {
        let buffer = unsafe { (shim.tensor_get_buffer)(output_handles[index]) };
        let size = output.data.len();
        let source_slice = unsafe { std::slice::from_raw_parts(buffer as *const u8, size) };
        output.data.copy_from_slice(source_slice);
    }

    // 7. Cleanup
    for handle in input_handles {
        unsafe { (shim.tensor_destroy)(handle) };
    }
    for handle in output_handles {
        unsafe { (shim.tensor_destroy)(handle) };
    }
    for handle in dimension_handles {
        unsafe { (shim.tensor_dim_destroy)(handle) };
    }
    unsafe {
        (shim.context_destroy)(context);
        (shim.model_desc_destroy)(descriptor);
        (shim.manager_destroy)(manager);
    }

    Ok(())
}

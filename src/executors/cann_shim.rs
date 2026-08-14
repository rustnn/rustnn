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

//! CANN shim — thin wrapper over the auto-generated FFI bindings.
//!
//! Only compiled with `cann-runtime`: the adapter is compiled and linked via
//! `build.rs`, and the bindgen-generated `cann_sys` functions are called
//! directly. In mock mode (`cann-runtime-mock`) this module has no dispatch
//! implementation; the backend returns `Err` instead.

use crate::error::{Error, Result};

/// Rust-side tensor descriptor passed between the backend and the shim.
pub(crate) struct CannTensorDesc {
    pub data: Vec<u8>,
    pub shape: Vec<u32>,
    /// Raw CANN data type value (`CANN_DT_*`). Converted to the bindgen
    /// `ddk_CannDataType` enum at the FFI boundary in `cann_shim::cann_dispatch`.
    pub dtype: i32,
}

/// Reinterpret an i32 CANN data-type value as the bindgen enum.
///
/// `ddk_CannDataType` is `#[repr(u32)]` and all `CANN_DT_*` values are
/// non-negative, so this is sound for every value produced by
/// `ml_operand_to_cann_dtype`.
fn cann_dtype(value: i32) -> crate::executors::cann_sys::ddk_CannDataType {
    // SAFETY: `ddk_CannDataType` is `#[repr(u32)]` and all `CANN_DT_*` values
    // are non-negative, so this is sound for every value produced by
    // `ml_operand_to_cann_dtype`.
    unsafe {
        std::mem::transmute::<u32, crate::executors::cann_sys::ddk_CannDataType>(value as u32)
    }
}

/// Dispatches a graph using the CANN shim.
pub(crate) fn cann_dispatch(
    model_bytes: &[u8],
    inputs: &[CannTensorDesc],
    outputs: &mut [CannTensorDesc],
) -> Result<()> {
    use std::ffi::CString;

    use crate::executors::cann_sys::*;

    // 1. Init model manager
    let manager = unsafe { ddk_cann_model_manager_create() };
    if manager.is_null() {
        return Err(Error::GraphDispatchError {
            source: "cann_model_manager_create failed".into(),
        });
    }
    let status = unsafe { ddk_cann_model_manager_init(manager) };
    if status != 0 {
        return Err(Error::GraphDispatchError {
            source: format!("cann_model_manager_init failed: {status}").into(),
        });
    }

    // 2. Load model
    let name = CString::new("webnn_model").unwrap();
    let mut descriptor = unsafe {
        ddk_cann_model_desc_create(
            name.as_ptr(),
            3, // AiModelDescription_Frequency_HIGH
            0, // HIAI_FRAMEWORK_NONE
            0, // HIAI_MODELTYPE_OFFLINE
            0, // AiModelDescription_DeviceType_NPU
        )
    };
    if descriptor.is_null() {
        unsafe { ddk_cann_model_manager_destroy(manager) };
        return Err(Error::GraphDispatchError {
            source: "cann_model_desc_create failed".into(),
        });
    }
    if unsafe {
        ddk_cann_model_desc_set_model_buffer(
            descriptor,
            model_bytes.as_ptr() as *const _,
            model_bytes.len() as u32,
        )
    } != 0
    {
        unsafe { ddk_cann_model_manager_destroy(manager) };
        return Err(Error::GraphDispatchError {
            source: "cann_model_desc_set_model_buffer failed".into(),
        });
    }
    if unsafe { ddk_cann_model_manager_load(manager, &mut descriptor, 1) } != 0 {
        unsafe { ddk_cann_model_manager_destroy(manager) };
        return Err(Error::GraphDispatchError {
            source: "cann_model_manager_load failed".into(),
        });
    }

    // 3. Build input tensors
    let mut input_handles: Vec<ddk_CannIOTensorHandle> = Vec::new();
    let mut dimension_handles: Vec<ddk_CannIOTensorDimensionHandle> = Vec::new();

    for input in inputs.iter() {
        let dimension = unsafe {
            ddk_cann_io_tensor_dim_create_nd(input.shape.as_ptr(), input.shape.len() as i32)
        };
        if dimension.is_null() {
            return Err(Error::GraphDispatchError {
                source: "cann_io_tensor_dim_create_nd failed".into(),
            });
        }
        dimension_handles.push(dimension);

        let tensor = unsafe { ddk_cann_io_tensor_create() };
        if tensor.is_null() {
            return Err(Error::GraphDispatchError {
                source: "cann_io_tensor_create failed".into(),
            });
        }
        let dtype = cann_dtype(input.dtype);
        let status = unsafe { ddk_cann_io_tensor_init(tensor, dimension, dtype) };
        if status != 0 {
            return Err(Error::GraphDispatchError {
                source: format!("cann_io_tensor_init failed: {status}").into(),
            });
        }
        let status = unsafe {
            ddk_cann_io_tensor_set_data(
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
    let mut output_handles: Vec<ddk_CannIOTensorHandle> = Vec::new();
    for output in outputs.iter() {
        let dimension = unsafe {
            ddk_cann_io_tensor_dim_create_nd(output.shape.as_ptr(), output.shape.len() as i32)
        };
        if dimension.is_null() {
            return Err(Error::GraphDispatchError {
                source: "cann_io_tensor_dim_create_nd failed".into(),
            });
        }
        dimension_handles.push(dimension);

        let tensor = unsafe { ddk_cann_io_tensor_create() };
        if tensor.is_null() {
            return Err(Error::GraphDispatchError {
                source: "cann_io_tensor_create failed".into(),
            });
        }
        let dtype = cann_dtype(output.dtype);
        let status = unsafe { ddk_cann_io_tensor_init(tensor, dimension, dtype) };
        if status != 0 {
            return Err(Error::GraphDispatchError {
                source: format!("cann_io_tensor_init failed: {status}").into(),
            });
        }
        output_handles.push(tensor);
    }

    // 5. Run inference
    let context = unsafe { ddk_cann_context_create() };
    let model_name = CString::new("model_name").unwrap();
    let model_val = CString::new("webnn_model").unwrap();
    unsafe { ddk_cann_context_set_para(context, model_name.as_ptr(), model_val.as_ptr()) };

    let mut stamp: i32 = 0;
    let status = unsafe {
        ddk_cann_model_manager_process(
            manager,
            context,
            input_handles.as_mut_ptr(),
            input_handles.len() as i32,
            output_handles.as_mut_ptr(),
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
        let buffer = unsafe { ddk_cann_io_tensor_get_buffer(output_handles[index]) };
        let size = output.data.len();
        let source_slice = unsafe { std::slice::from_raw_parts(buffer as *const u8, size) };
        output.data.copy_from_slice(source_slice);
    }

    // 7. Cleanup
    for handle in input_handles {
        unsafe { ddk_cann_io_tensor_destroy(handle) };
    }
    for handle in output_handles {
        unsafe { ddk_cann_io_tensor_destroy(handle) };
    }
    for handle in dimension_handles {
        unsafe { ddk_cann_io_tensor_dim_destroy(handle) };
    }
    unsafe {
        ddk_cann_context_destroy(context);
        ddk_cann_model_desc_destroy(descriptor);
        ddk_cann_model_manager_destroy(manager);
    }

    Ok(())
}

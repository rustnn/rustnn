// SPDX-FileCopyrightText: 2026 Shubham Gupta <shubhamg13.work@gmail.com>
//
// SPDX-License-Identifier: Apache-2

//! LiteRT (TensorFlow Lite) backend (`litert-runtime` feature).
//!
//! Converts the graph to a TFLite flatbuffer with [`LiteRtConverter`] (NCHW operands are
//! transposed to NHWC first) and runs it with the LiteRT interpreter from `litert-sys`.
//!
#![doc = include_str!("../../docs/integration/litert.md")]

use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;
use std::sync::OnceLock;

use litert_sys::{self as sys};

use crate::backend_selection::DeviceType;
use crate::converters::{GraphConverter, LiteRtConverter};
use crate::error::{Error, Result};
use crate::mlcontext::{
    ListDevices, MLBackendBuilder, MLBackendContext, MLGraph, MLNamedTensors, MLTensor,
    MLTensorDescriptor, RustNNOptions,
};

use crate::operator_enums::MLOperandDataType;
use crate::operators::Operation;
use crate::{GraphError, GraphInfo};

struct LiteRt;

impl LiteRt {
    fn env() -> sys::LiteRtEnvironment {
        static ENV: OnceLock<usize> = OnceLock::new();
        *ENV.get_or_init(|| {
            let mut env = std::ptr::null_mut();
            check(unsafe { sys::LiteRtCreateEnvironment(0, std::ptr::null(), &mut env) })
                .expect("LiteRtCreateEnvironment failed");
            env as usize
        }) as *mut _
    }
}

fn check(status: sys::LiteRtStatus) -> Result<()> {
    if status == sys::kLiteRtStatusOk {
        Ok(())
    } else {
        Err(Error::GraphDispatchError {
            source: format!("LiteRT status error: code={}", status).into(),
        })
    }
}

fn ml_operand_to_litert_element_type(dt: MLOperandDataType) -> Result<litert::ElementType> {
    use litert::ElementType;
    Ok(match dt {
        MLOperandDataType::Float32 => ElementType::Float32,
        MLOperandDataType::Float16 => ElementType::Float16,
        MLOperandDataType::Int32 => ElementType::Int32,
        MLOperandDataType::Uint32 => ElementType::UInt32,
        MLOperandDataType::Int64 => ElementType::Int64,
        MLOperandDataType::Uint64 => ElementType::UInt64,
        MLOperandDataType::Int8 => ElementType::Int8,
        MLOperandDataType::Uint8 => ElementType::UInt8,
        MLOperandDataType::Int4 => ElementType::Int4,
        _ => {
            return Err(Error::GraphBuildError {
                source: format!("unsupported ML data type for litert: {:?}", dt).into(),
            });
        }
    })
}

pub(crate) struct LiteRtGraph {
    compiled: NonNull<sys::LiteRtCompiledModelT>,
    model: NonNull<sys::LiteRtModelT>,
    _model_bytes: Box<[u8]>,
    /// Operand ids that were spatially transposed NCHW→NHWC.
    spatial_operand_ids: std::collections::HashSet<u32>,
    /// Filter operand ids whose runtime data needs layout→OHWI transpose.
    /// Maps filter id → (WebNN filter_layout, target shape, is_depthwise [unused]).
    filter_transpose_info: std::collections::HashMap<u32, (String, Vec<i32>, bool)>,
    /// Operand ids needing BOOL type (WHERE condition, comparison ops).
    bool_operand_ids: std::collections::HashSet<u32>,
    /// Graph input names in signature order. See [`order_by_signature`].
    input_order: Vec<(String, u32)>,
    /// Graph output names in signature order, id included.
    output_order: Vec<(String, u32)>,
    /// Whether the graph was rewritten from float16 to float32. See [`emulate_float16`].
    float16_emulated: bool,
}

unsafe impl Send for LiteRtGraph {}
unsafe impl Sync for LiteRtGraph {}

impl LiteRtGraph {
    fn new(
        model_bytes: Vec<u8>,
        accelerator_bits: sys::LiteRtHwAcceleratorSet,
        spatial_operand_ids: std::collections::HashSet<u32>,
        filter_transpose_info: std::collections::HashMap<u32, (String, Vec<i32>, bool)>,
        bool_operand_ids: std::collections::HashSet<u32>,
        input_order: Vec<(String, u32)>,
        output_order: Vec<(String, u32)>,
        float16_emulated: bool,
    ) -> Result<Self> {
        let owned = model_bytes.into_boxed_slice();
        unsafe {
            let mut model = std::ptr::null_mut();
            check(sys::LiteRtCreateModelFromBuffer(
                owned.as_ptr() as *const c_void,
                owned.len(),
                &mut model,
            ))?;
            let model = NonNull::new(model).ok_or_else(|| Error::GraphBuildError {
                source: "LiteRT: null model handle".into(),
            })?;

            let mut options = std::ptr::null_mut();
            check(sys::LiteRtCreateOptions(&mut options))?;
            check(sys::LiteRtSetOptionsHardwareAccelerators(
                options,
                accelerator_bits,
            ))?;

            let mut compiled = std::ptr::null_mut();
            let status = sys::LiteRtCreateCompiledModel(
                LiteRt::env(),
                model.as_ptr(),
                options,
                &mut compiled,
            );
            sys::LiteRtDestroyOptions(options);
            check(status)?;
            let compiled = NonNull::new(compiled).ok_or_else(|| Error::GraphBuildError {
                source: "LiteRT: null compiled model handle".into(),
            })?;

            Ok(Self {
                compiled,
                model,
                _model_bytes: owned,
                spatial_operand_ids,
                filter_transpose_info,
                bool_operand_ids,
                input_order,
                output_order,
                float16_emulated,
            })
        }
    }

    fn run(
        &self,
        in_raw: &[sys::LiteRtTensorBuffer],
        out_raw: &mut [sys::LiteRtTensorBuffer],
    ) -> Result<()> {
        check(unsafe {
            sys::LiteRtRunCompiledModel(
                self.compiled.as_ptr(),
                0,
                in_raw.len(),
                in_raw.as_ptr() as *mut sys::LiteRtTensorBuffer,
                out_raw.len(),
                out_raw.as_mut_ptr(),
            )
        })
    }
}

impl fmt::Debug for LiteRtGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiteRtGraph").finish()
    }
}

impl Drop for LiteRtGraph {
    fn drop(&mut self) {
        unsafe {
            sys::LiteRtDestroyCompiledModel(self.compiled.as_ptr());
            sys::LiteRtDestroyModel(self.model.as_ptr());
        }
    }
}

// Host tensor storage for LiteRT backend.
pub(crate) struct LiteRtTensor {
    handle: sys::LiteRtTensorBuffer,
}

impl fmt::Debug for LiteRtTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiteRtTensor").finish()
    }
}

unsafe impl Send for LiteRtTensor {}
unsafe impl Sync for LiteRtTensor {}

impl LiteRtTensor {
    fn new_with_layout(descriptor: &MLTensorDescriptor, nhwc: bool) -> Result<Self> {
        let element_type = ml_operand_to_litert_element_type(descriptor.data_type())?;
        let orig = descriptor.shape();
        let dims: Vec<i32> = if nhwc && orig.len() == 4 {
            vec![
                orig[0] as i32,
                orig[2] as i32,
                orig[3] as i32,
                orig[1] as i32,
            ]
        } else {
            orig.iter().map(|&d| d as i32).collect()
        };
        Self::create_litert_tensor(&dims, element_type, true)
    }

    /// Create a LiteRT tensor with the given shape and element type.
    fn create_litert_tensor(
        dims: &[i32],
        element_type: litert::ElementType,
        has_strides: bool,
    ) -> Result<LiteRtTensor> {
        let shape = litert::TensorShape {
            element_type,
            dims: dims.to_vec(),
        };
        let element_size = match shape.element_type {
            litert::ElementType::Float32 => 4,
            litert::ElementType::Float16 => 2,
            litert::ElementType::Int32 => 4,
            litert::ElementType::UInt32 => 4,
            litert::ElementType::Int64 => 8,
            litert::ElementType::UInt64 => 8,
            litert::ElementType::Int16 => 2,
            litert::ElementType::UInt16 => 2,
            litert::ElementType::Int8 => 1,
            litert::ElementType::UInt8 => 1,
            litert::ElementType::Bool => 1,
            _ => {
                return Err(Error::GraphBuildError {
                    source: format!("unsupported element type: {:?}", shape.element_type).into(),
                });
            }
        };
        let size_bytes = shape.num_elements() * element_size;

        let mut layout = sys::LiteRtLayout::default();
        layout.set_rank(u32::try_from(shape.dims.len()).expect("rank fits in u32"));
        layout.set_has_strides(has_strides);
        for (slot, &d) in layout.dimensions.iter_mut().zip(shape.dims.iter()) {
            *slot = d;
        }
        if has_strides && shape.dims.len() >= 1 {
            let mut stride: u32 = 1;
            for i in (0..shape.dims.len()).rev() {
                layout.strides[i] = stride;
                stride *= shape.dims[i] as u32;
            }
        }
        let tensor_type = sys::LiteRtRankedTensorType {
            element_type: shape.element_type as sys::LiteRtElementType,
            layout,
        };

        let mut handle = std::ptr::null_mut();
        check(unsafe {
            sys::LiteRtCreateManagedTensorBuffer(
                LiteRt::env(),
                sys::kLiteRtTensorBufferTypeHostMemory,
                &tensor_type,
                size_bytes,
                &mut handle,
            )
        })?;
        Ok(Self { handle })
    }

    fn lock(&self, mode: sys::LiteRtTensorBufferLockMode) -> Result<*mut u8> {
        let mut addr: *mut c_void = std::ptr::null_mut();
        check(unsafe { sys::LiteRtLockTensorBuffer(self.handle, &mut addr, mode) })?;
        Ok(addr as *mut u8)
    }

    fn unlock(&self) -> Result<()> {
        check(unsafe { sys::LiteRtUnlockTensorBuffer(self.handle) })
    }

    fn write(&self, data: &[u8]) -> Result<()> {
        let ptr = self.lock(sys::kLiteRtTensorBufferLockModeWrite)?;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        self.unlock()
    }

    fn read(&self, buf: &mut [u8]) -> Result<()> {
        let ptr = self.lock(sys::kLiteRtTensorBufferLockModeRead)?;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), buf.len());
        }
        self.unlock()
    }
}

impl Drop for LiteRtTensor {
    fn drop(&mut self) {
        unsafe { sys::LiteRtDestroyTensorBuffer(self.handle) };
    }
}

pub(crate) struct LiteRtContext {
    tensors: Vec<LiteRtTensor>,
    device_type: DeviceType,
    pub(crate) needs_layout_fix: bool,
}

/// Whether `op` interprets its input as NCHW and needs the NHWC layout transposes.
pub fn is_spatial_op(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Conv2d { .. }
            | Operation::ConvTranspose2d { .. }
            | Operation::MaxPool2d { .. }
            | Operation::AveragePool2d { .. }
            | Operation::L2Pool2d { .. }
            | Operation::InstanceNormalization { .. }
    )
}

/// Operands whose 4-D layout must be rewritten NCHW -> NHWC.
///
/// Keyed on operand id: names exist only for graph inputs and outputs, so a name-keyed
/// set cannot contain an intermediate.
fn collect_spatial_operand_names(graph_info: &GraphInfo) -> std::collections::HashSet<u32> {
    let mut names = std::collections::HashSet::new();
    for op in &graph_info.operations {
        if !is_spatial_op(op) {
            continue;
        }
        let needs_nchw_swap = match op {
            Operation::Conv2d { options, .. } => {
                let layout = options
                    .as_ref()
                    .map(|o| o.input_layout.as_str())
                    .unwrap_or("");
                layout.is_empty() || layout.eq_ignore_ascii_case("nchw")
            }
            Operation::ConvTranspose2d { options, .. } => {
                let layout = options
                    .as_ref()
                    .map(|o| o.input_layout.as_str())
                    .unwrap_or("");
                layout.is_empty() || layout.eq_ignore_ascii_case("nchw")
            }
            Operation::MaxPool2d { options, .. }
            | Operation::AveragePool2d { options, .. }
            | Operation::L2Pool2d { options, .. } => {
                let layout = options.as_ref().map(|o| o.layout.as_str()).unwrap_or("");
                layout.is_empty() || layout.eq_ignore_ascii_case("nchw")
            }
            Operation::InstanceNormalization { options, .. } => {
                let layout = options.as_ref().map(|o| o.layout.as_str()).unwrap_or("");
                layout.is_empty() || layout.eq_ignore_ascii_case("nchw")
            }
            _ => false,
        };
        if !needs_nchw_swap {
            continue;
        }
        for id in op.inputs() {
            if let Some(op_info) = graph_info.operand(id) {
                if op_info.kind == crate::graph::OperandKind::Constant
                    || (matches!(
                        op,
                        Operation::Conv2d { .. } | Operation::ConvTranspose2d { .. }
                    ) && op.inputs().iter().position(|&x| x == id) == Some(1))
                {
                    continue;
                }
                if op_info.descriptor.shape.len() == 4 {
                    names.insert(id);
                }
            }
        }
        for &id in op.outputs() {
            if let Some(op_info) = graph_info.operand(id) {
                if op_info.descriptor.shape.len() == 4 {
                    names.insert(id);
                }
            }
        }
    }

    // Follow the dataflow through the ops below, so both sides of an op keep one layout.
    loop {
        let mut grew = false;
        for op in &graph_info.operations {
            // An op fed by a rewritten operand is rewritten too, otherwise it gets NHWC
            // inputs and NCHW outputs.
            if !op.inputs().iter().any(|id| names.contains(id)) {
                continue;
            }
            // Filters keep their own layout; see `collect_filter_transpose_info`.
            let filter_id = match op {
                Operation::Conv2d { .. } | Operation::ConvTranspose2d { .. } => {
                    op.inputs().get(1).copied()
                }
                _ => None,
            };
            for id in op.inputs().iter().chain(op.outputs()) {
                if Some(*id) == filter_id {
                    continue;
                }
                if let Some(op_info) = graph_info.operand(*id) {
                    if op_info.descriptor.shape.len() == 4 && names.insert(*id) {
                        grew = true;
                    }
                }
            }
        }
        if !grew {
            break;
        }
    }

    names
}

fn collect_filter_transpose_info(
    graph_info: &GraphInfo,
    spatial_operand_names: &mut std::collections::HashSet<u32>,
) -> std::collections::HashMap<u32, (String, Vec<i32>, bool)> {
    let mut filter_transpose_info: std::collections::HashMap<u32, (String, Vec<i32>, bool)> =
        std::collections::HashMap::new();
    for op in &graph_info.operations {
        let Operation::Conv2d { options, .. } = op else {
            continue;
        };
        let opts = options.as_ref().cloned().unwrap_or_default();
        let _input_layout = opts.input_layout.as_str();
        let mut filter_layout = opts.filter_layout.as_str();
        if filter_layout.is_empty() {
            filter_layout = "oihw";
        }
        if filter_layout != "ohwi" {
            if let Some(&fid) = op.inputs().get(1) {
                if let Some(fop) = graph_info.operand(fid) {
                    if fop.descriptor.shape.len() == 4 {
                        if fop.kind == crate::graph::OperandKind::Constant {
                            spatial_operand_names.remove(&fid);
                        } else {
                            let orig_shape = fop
                                .descriptor
                                .shape
                                .iter()
                                .map(|d| match d {
                                    crate::graph::Dimension::Static(v) => *v as i32,
                                    _ => 0,
                                })
                                .collect::<Vec<_>>();
                            let target_shape = ohwi_shape_from_layout(&orig_shape, filter_layout);
                            spatial_operand_names.insert(fid);
                            filter_transpose_info
                                .insert(fid, (filter_layout.to_string(), target_shape, false));
                        }
                    }
                }
            }
        } else if let Some(&fid) = op.inputs().get(1) {
            spatial_operand_names.remove(&fid);
        }
    }
    filter_transpose_info
}

fn collect_spatial_info(
    graph_info: &GraphInfo,
) -> (
    std::collections::HashSet<u32>,
    std::collections::HashMap<u32, (String, Vec<i32>, bool)>,
) {
    let mut spatial_operand_names = collect_spatial_operand_names(graph_info);
    let filter_transpose_info =
        collect_filter_transpose_info(graph_info, &mut spatial_operand_names);
    (spatial_operand_names, filter_transpose_info)
}

/// Operands read back as BOOL rather than their declared type, keyed on operand id.
fn collect_bool_operand_names(graph: &GraphInfo) -> std::collections::HashSet<u32> {
    let mut names = std::collections::HashSet::new();
    for op in &graph.operations {
        match op {
            Operation::Where { .. } => {
                if let Some(&cond_id) = op.inputs().get(0) {
                    if graph.operand(cond_id).is_some() {
                        names.insert(cond_id);
                    }
                }
            }
            Operation::IsNaN { .. }
            | Operation::IsInfinite { .. }
            | Operation::Equal { .. }
            | Operation::Greater { .. }
            | Operation::GreaterOrEqual { .. }
            | Operation::Lesser { .. }
            | Operation::LesserOrEqual { .. }
            | Operation::NotEqual { .. } => {
                for &out_id in op.outputs() {
                    if graph.operand(out_id).is_some() {
                        names.insert(out_id);
                    }
                }
            }
            _ => {}
        }
    }
    names
}

fn modify_graph_for_nhwc(
    graph: &mut GraphInfo,
    spatial_operand_names: &std::collections::HashSet<u32>,
) {
    let mut skip_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for op in &graph.operations {
        if let Operation::Conv2d { .. } | Operation::ConvTranspose2d { .. } = op {
            if let Some(&fid) = op.inputs().get(1) {
                skip_ids.insert(fid);
            }
        }
    }
    // A rank-4 constant the closure pulled in is re-laid out with its consumers: a PReLU
    // slope left in NCHW beside an NHWC input is not broadcastable. One that also feeds an
    // op outside the rewrite is left alone.
    let needs_swap = |op: &Operation| -> bool {
        match op {
            Operation::Conv2d { options, .. } => {
                let l = options
                    .as_ref()
                    .map(|o| o.input_layout.as_str())
                    .unwrap_or("");
                l.is_empty() || l.eq_ignore_ascii_case("nchw")
            }
            Operation::ConvTranspose2d { options, .. } => {
                let l = options
                    .as_ref()
                    .map(|o| o.input_layout.as_str())
                    .unwrap_or("");
                l.is_empty() || l.eq_ignore_ascii_case("nchw")
            }
            Operation::MaxPool2d { options, .. }
            | Operation::AveragePool2d { options, .. }
            | Operation::L2Pool2d { options, .. } => {
                let l = options.as_ref().map(|o| o.layout.as_str()).unwrap_or("");
                l.is_empty() || l.eq_ignore_ascii_case("nchw")
            }
            Operation::InstanceNormalization { options, .. } => {
                let l = options.as_ref().map(|o| o.layout.as_str()).unwrap_or("");
                l.is_empty() || l.eq_ignore_ascii_case("nchw")
            }
            _ => false,
        }
    };
    let rewriteable_constants: std::collections::HashSet<u32> =
        graph
            .operands
            .iter()
            .enumerate()
            .filter(|(_, operand)| operand.kind == crate::graph::OperandKind::Constant)
            .map(|(i, _)| i as u32)
            .filter(|&id| {
                // Rewritten when it is the rank-4 input of a spatial op, or when a consumer
                // already has an operand in the set (the seed skips constants).
                let feeds_spatial = graph
                    .operations
                    .iter()
                    .any(|op| is_spatial_op(op) && needs_swap(op) && op.inputs().contains(&id));
                if feeds_spatial {
                    return true;
                }
                spatial_operand_names.contains(&id)
                    && graph
                        .operations
                        .iter()
                        .filter(|op| op.inputs().contains(&id))
                        .all(|op| {
                            is_spatial_op(op)
                                || op.inputs().iter().chain(op.outputs()).any(|other| {
                                    other != &id && spatial_operand_names.contains(other)
                                })
                        })
            })
            .collect();

    for (i, operand) in graph.operands.iter_mut().enumerate() {
        let id = i as u32;
        if skip_ids.contains(&id) {
            continue;
        }
        if operand.descriptor.shape.len() != 4 {
            continue;
        }
        let in_set = spatial_operand_names.contains(&id);
        if operand.kind == crate::graph::OperandKind::Constant {
            if !rewriteable_constants.contains(&id) {
                continue;
            }
        } else if !in_set {
            continue;
        }
        let mut dims = [1u32; 4];
        let mut valid = true;
        for (j, d) in operand.descriptor.shape.iter().enumerate() {
            if j >= 4 {
                valid = false;
                break;
            }
            match d {
                crate::graph::Dimension::Static(v) => dims[j] = *v,
                _ => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid {
            continue;
        }
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        operand.descriptor.shape = vec![
            crate::graph::Dimension::Static(n),
            crate::graph::Dimension::Static(h),
            crate::graph::Dimension::Static(w),
            crate::graph::Dimension::Static(c),
        ];
        if operand.kind == crate::graph::OperandKind::Constant {
            if let Some(cd) = graph.constant_operand_ids_to_handles.get_mut(&id) {
                let esz = cd.data.len() / ((n * c * h * w) as usize);
                if esz > 0
                    && esz * (n as usize) * (c as usize) * (h as usize) * (w as usize)
                        == cd.data.len()
                {
                    cd.data =
                        transpose_nchw_to_nhwc(&cd.data, &[n as u64, c as u64, h as u64, w as u64]);
                }
            }
        }
    }

    // Operators naming an axis, or carrying per-axis arrays, still refer to NCHW
    // positions and are remapped alongside the operand.
    for op in graph.operations.iter_mut() {
        // The parameters handled below address the operand the op reads, so the op is
        // remapped exactly when one of its inputs was. Transpose is the exception: its
        // permutation spans both sides, and it can sit on the boundary itself.
        let remap = match op {
            Operation::Transpose { .. } => op
                .inputs()
                .iter()
                .chain(op.outputs())
                .any(|id| spatial_operand_names.contains(id)),
            _ => op
                .inputs()
                .first()
                .is_some_and(|id| spatial_operand_names.contains(id)),
        };
        if !remap {
            continue;
        }
        match op {
            Operation::Concat { axis, .. } => *axis = nchw_axis_to_nhwc(*axis),
            Operation::Softmax { axis, .. } => *axis = nchw_axis_to_nhwc(*axis),
            Operation::Split { options, .. } => {
                if let Some(options) = options {
                    options.axis = nchw_axis_to_nhwc(options.axis);
                }
            }
            Operation::Pad {
                beginning_padding,
                ending_padding,
                ..
            } => {
                *beginning_padding = permute_nchw_to_nhwc(beginning_padding);
                *ending_padding = permute_nchw_to_nhwc(ending_padding);
            }
            Operation::Slice {
                starts,
                sizes,
                options,
                ..
            } => {
                *starts = permute_nchw_to_nhwc(starts);
                *sizes = permute_nchw_to_nhwc(sizes);
                if let Some(options) = options {
                    options.strides = permute_nchw_to_nhwc(&options.strides);
                }
            }
            // Rank-4 only: rank-changing targets are barriers, not relabels.
            Operation::Reshape { new_shape, .. } => {
                *new_shape = permute_nchw_to_nhwc(new_shape);
            }
            Operation::ReduceSum { options, .. }
            | Operation::ReduceMean { options, .. }
            | Operation::ReduceMax { options, .. }
            | Operation::ReduceMin { options, .. }
            | Operation::ReduceProduct { options, .. }
            | Operation::ReduceL1 { options, .. }
            | Operation::ReduceL2 { options, .. }
            | Operation::ReduceLogSum { options, .. }
            | Operation::ReduceLogSumExp { options, .. }
            | Operation::ReduceSumSquare { options, .. } => {
                if let Some(axes) = options.as_mut().and_then(|o| o.axes.as_mut()) {
                    for axis in axes.iter_mut() {
                        *axis = nchw_axis_to_nhwc(*axis);
                    }
                }
            }
            Operation::Transpose { input, options, .. } => {
                let input_relaid = spatial_operand_names.contains(input);
                let options = options.get_or_insert_with(Default::default);
                if options.permutation.is_empty() {
                    // An empty permutation means a full reversal; materialise it first.
                    options.permutation = (0..4u32).rev().collect();
                }
                options.permutation = if input_relaid {
                    fold_relayout_into_permutation(&options.permutation)
                } else {
                    // Output-only relayout: the transpose must map NHWC to NHWC.
                    fold_output_relayout_into_permutation(&options.permutation)
                };
            }
            Operation::Resample2d { input, options, .. } => {
                let Some(options) = options.as_mut() else {
                    continue;
                };
                // The default axes are the last two dimensions; a rewritten rank-4
                // operand has them at 1 and 2, which is what the remap below produces.
                if options.axes.is_empty() && spatial_operand_names.contains(input) {
                    options.axes = vec![2, 3];
                }
                for axis in options.axes.iter_mut() {
                    *axis = nchw_axis_to_nhwc(*axis);
                }
            }
            _ => {}
        }
    }
}

/// Rewrites a permutation for the case where both operands are relaid out. The relayout
/// is `R = [0, 2, 3, 1]`, so it conjugates the original permutation with `R`.
fn fold_relayout_into_permutation(permutation: &[u32]) -> Vec<u32> {
    const R: [u32; 4] = [0, 2, 3, 1];
    const R_INV: [u32; 4] = [0, 3, 1, 2];
    if permutation.len() != 4 || permutation.iter().any(|&p| p >= 4) {
        return permutation.to_vec();
    }
    (0..4)
        .map(|i| R_INV[permutation[R[i] as usize] as usize])
        .collect()
}

/// Rewrites a permutation when only the transpose's output is relaid out, mapping NHWC to
/// NHWC.
fn fold_output_relayout_into_permutation(permutation: &[u32]) -> Vec<u32> {
    const R: [u32; 4] = [0, 2, 3, 1];
    if permutation.len() != 4 || permutation.iter().any(|&p| p >= 4) {
        return permutation.to_vec();
    }
    (0..4).map(|i| permutation[R[i] as usize]).collect()
}

/// Maps an axis index from NCHW order to its position in NHWC order.
fn nchw_axis_to_nhwc(axis: u32) -> u32 {
    debug_assert!(axis < 4, "axis {axis} is not a rank-4 NCHW axis");
    match axis {
        0 => 0,
        1 => 3,
        2 => 1,
        3 => 2,
        other => other,
    }
}

/// Reorders a full-rank per-axis array from NCHW order to NHWC order; shorter arrays are
/// returned unchanged.
fn permute_nchw_to_nhwc<T: Clone>(values: &[T]) -> Vec<T> {
    if values.len() != 4 {
        return values.to_vec();
    }
    vec![
        values[0].clone(),
        values[2].clone(),
        values[3].clone(),
        values[1].clone(),
    ]
}

impl fmt::Debug for LiteRtContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiteRtContext")
            .field("num_tensors", &self.tensors.len())
            .field("device_type", &self.device_type)
            .finish()
    }
}

/// Transpose float data from NCHW to NHWC layout.
fn transpose_nchw_to_nhwc(data: &[u8], shape: &[u64]) -> Vec<u8> {
    let n = shape[0] as usize;
    let c = shape[1] as usize;
    let h = shape[2] as usize;
    let w = shape[3] as usize;
    let esz = data.len() / (n * c * h * w);
    let mut out = vec![0u8; data.len()];
    for nn in 0..n {
        for cc in 0..c {
            for hh in 0..h {
                for ww in 0..w {
                    let src = ((nn * c + cc) * h + hh) * w + ww;
                    let dst = ((nn * h + hh) * w + ww) * c + cc;
                    out[dst * esz..(dst + 1) * esz]
                        .copy_from_slice(&data[src * esz..(src + 1) * esz]);
                }
            }
        }
    }
    out
}

/// Transpose float data from NHWC to NCHW layout.
fn transpose_nhwc_to_nchw(data: &[u8], shape: &[u64]) -> Vec<u8> {
    let n = shape[0] as usize;
    let h = shape[2] as usize;
    let w = shape[3] as usize;
    let c = shape[1] as usize;
    let esz = data.len() / (n * h * w * c);
    let mut out = vec![0u8; data.len()];
    for nn in 0..n {
        for cc in 0..c {
            for hh in 0..h {
                for ww in 0..w {
                    let src = ((nn * h + hh) * w + ww) * c + cc;
                    let dst = ((nn * c + cc) * h + hh) * w + ww;
                    out[dst * esz..(dst + 1) * esz]
                        .copy_from_slice(&data[src * esz..(src + 1) * esz]);
                }
            }
        }
    }
    out
}

/// Transpose weight data from OIHW (`[O, I, H, W]`) to OHWI (`[O, H, W, I]`) layout.
pub fn transpose_oihw_to_ohwi(data: &[u8], o: usize, i: usize, h: usize, w: usize) -> Vec<u8> {
    let esz = data.len() / (o * i * h * w);
    if esz == 0 || esz * o * i * h * w != data.len() {
        return data.to_vec();
    }
    let mut out = vec![0u8; data.len()];
    for oo in 0..o {
        for ii in 0..i {
            for hh in 0..h {
                for ww in 0..w {
                    let src = ((oo * i + ii) * h + hh) * w + ww;
                    let dst = ((oo * h + hh) * w + ww) * i + ii;
                    out[dst * esz..(dst + 1) * esz]
                        .copy_from_slice(&data[src * esz..(src + 1) * esz]);
                }
            }
        }
    }
    out
}

/// Transpose weight data from IOHW (`[I, O, H, W]`) to OHWI (`[O, H, W, I]`) layout.
pub fn transpose_iohw_to_ohwi(data: &[u8], i: usize, o: usize, h: usize, w: usize) -> Vec<u8> {
    let esz = data.len() / (i * o * h * w);
    if esz == 0 || esz * i * o * h * w != data.len() {
        return data.to_vec();
    }
    let mut out = vec![0u8; data.len()];
    for oo in 0..o {
        for hh in 0..h {
            for ww in 0..w {
                for ii in 0..i {
                    let src = ((ii * o + oo) * h + hh) * w + ww;
                    let dst = ((oo * h + hh) * w + ww) * i + ii;
                    out[dst * esz..(dst + 1) * esz]
                        .copy_from_slice(&data[src * esz..(src + 1) * esz]);
                }
            }
        }
    }
    out
}

/// Transpose weight data from HWOI (`[H, W, O, I]`) to OHWI (`[O, H, W, I]`) layout.
pub fn transpose_hwoi_to_ohwi(data: &[u8], h: usize, w: usize, o: usize, i: usize) -> Vec<u8> {
    let esz = data.len() / (h * w * o * i);
    if esz == 0 || esz * h * w * o * i != data.len() {
        return data.to_vec();
    }
    let mut out = vec![0u8; data.len()];
    for oo in 0..o {
        for hh in 0..h {
            for ww in 0..w {
                for ii in 0..i {
                    let src = ((hh * w + ww) * o + oo) * i + ii;
                    let dst = ((oo * h + hh) * w + ww) * i + ii;
                    out[dst * esz..(dst + 1) * esz]
                        .copy_from_slice(&data[src * esz..(src + 1) * esz]);
                }
            }
        }
    }
    out
}

/// Transpose weight data from HWIO (`[H, W, I, O]`) to OHWI (`[O, H, W, I]`) layout.
pub fn transpose_hwio_to_ohwi(data: &[u8], h: usize, w: usize, i: usize, o: usize) -> Vec<u8> {
    let esz = data.len() / (h * w * i * o);
    if esz == 0 || esz * h * w * i * o != data.len() {
        return data.to_vec();
    }
    let mut out = vec![0u8; data.len()];
    for oo in 0..o {
        for hh in 0..h {
            for ww in 0..w {
                for ii in 0..i {
                    let src = ((hh * w + ww) * i + ii) * o + oo;
                    let dst = ((oo * h + hh) * w + ww) * i + ii;
                    out[dst * esz..(dst + 1) * esz]
                        .copy_from_slice(&data[src * esz..(src + 1) * esz]);
                }
            }
        }
    }
    out
}

/// Transpose weight data from IHWO (`[I, H, W, O]`) to OHWI (`[O, H, W, I]`) layout.
pub fn transpose_ihwo_to_ohwi(data: &[u8], i: usize, h: usize, w: usize, o: usize) -> Vec<u8> {
    let esz = data.len() / (i * h * w * o);
    if esz == 0 || esz * i * h * w * o != data.len() {
        return data.to_vec();
    }
    let mut out = vec![0u8; data.len()];
    for oo in 0..o {
        for hh in 0..h {
            for ww in 0..w {
                for ii in 0..i {
                    let src = ((ii * h + hh) * w + ww) * o + oo;
                    let dst = ((oo * h + hh) * w + ww) * i + ii;
                    out[dst * esz..(dst + 1) * esz]
                        .copy_from_slice(&data[src * esz..(src + 1) * esz]);
                }
            }
        }
    }
    out
}

/// Transpose filter data from any WebNN layout (OIHW/HWIO/IHWO) to TFLite-native OHWI.
fn transpose_filter_to_ohwi(data: &[u8], shape: &[u64], layout: &str) -> Vec<u8> {
    let s = |i: usize| shape[i] as usize;
    match layout {
        "hwio" => transpose_hwio_to_ohwi(data, s(0), s(1), s(2), s(3)),
        "ihwo" => transpose_ihwo_to_ohwi(data, s(0), s(1), s(2), s(3)),
        "iohw" => transpose_iohw_to_ohwi(data, s(0), s(1), s(2), s(3)),
        "hwoi" => transpose_hwoi_to_ohwi(data, s(0), s(1), s(2), s(3)),
        _ => transpose_oihw_to_ohwi(data, s(0), s(1), s(2), s(3)), // "oihw" default
    }
}

/// Compute OHWI shape [O,H,W,I] from original shape [d0,d1,d2,d3] and filter layout.
fn ohwi_shape_from_layout(shape: &[i32], layout: &str) -> Vec<i32> {
    if shape.len() != 4 {
        return shape.to_vec();
    }
    match layout {
        "hwio" => vec![shape[3], shape[0], shape[1], shape[2]],
        "ihwo" => vec![shape[3], shape[1], shape[2], shape[0]],
        "iohw" => vec![shape[1], shape[2], shape[3], shape[0]],
        "hwoi" => vec![shape[2], shape[0], shape[1], shape[3]],
        "ohwi" => shape.to_vec(),
        _ => vec![shape[0], shape[2], shape[3], shape[1]], // "oihw" default
    }
}

/// Orders `names` as the model's signature declares them: LiteRT binds buffers to
/// signature slots positionally, while `MLNamedTensors` iterates alphabetically.
fn order_by_signature<'a>(
    order: &'a [(String, u32)],
    names: &'a MLNamedTensors<'a>,
) -> Vec<(&'a str, u32, &'a MLTensor)> {
    let mut ordered: Vec<(&'a str, u32, &'a MLTensor)> = order
        .iter()
        .filter_map(|(name, operand_id)| {
            names
                .get(name.as_str())
                .map(|t| (name.as_str(), *operand_id, *t))
        })
        .collect();
    // Keep anything bound but not declared, so the mismatch surfaces downstream.
    if ordered.len() != names.len() {
        for (name, tensor) in names {
            if !order.iter().any(|(declared, _)| declared == *name) {
                // No id is known for an undeclared binding; `u32::MAX` matches nothing.
                ordered.push((name, u32::MAX, *tensor));
            }
        }
    }
    ordered
}

/// Element type the model uses for a caller-side buffer.
fn model_element_type(t: &MLTensor) -> litert::ElementType {
    if t.descriptor().data_type() == MLOperandDataType::Float16 {
        litert::ElementType::Float32
    } else {
        ml_operand_to_litert_element_type(t.descriptor().data_type()).expect("element type")
    }
}

fn tensor_dims(t: &MLTensor) -> Vec<i32> {
    t.descriptor().shape().iter().map(|&d| d as i32).collect()
}

fn fp16_to_f32(data: &[u8]) -> Vec<u8> {
    data.chunks_exact(2)
        .flat_map(|p| {
            half::f16::from_le_bytes([p[0], p[1]])
                .to_f32()
                .to_le_bytes()
        })
        .collect()
}

fn f32_to_fp16(data: &[u8]) -> Vec<u8> {
    data.chunks_exact(4)
        .flat_map(|w| {
            half::f16::from_f32(f32::from_le_bytes([w[0], w[1], w[2], w[3]])).to_le_bytes()
        })
        .collect()
}

/// Rewrites a float16 graph to float32, its constants included.
///
/// TFLite has no float16 kernels, so a model containing FLOAT16 tensors fails to allocate.
/// Graphs that cast *to* float16 are left alone: the rounding is what they test.
fn emulate_float16(graph: &mut GraphInfo) -> bool {
    let is_fp16 = |dt: crate::graph::DataType| dt == crate::graph::DataType::Float16;
    if !graph
        .operands
        .iter()
        .any(|o| is_fp16(o.descriptor.data_type))
    {
        return false;
    }
    if graph.operations.iter().any(|op| {
        matches!(op, Operation::Cast { data_type, .. } if *data_type == MLOperandDataType::Float16)
    }) {
        return false;
    }
    for (id, operand) in graph.operands.iter_mut().enumerate() {
        if !is_fp16(operand.descriptor.data_type) {
            continue;
        }
        operand.descriptor.data_type = crate::graph::DataType::Float32;
        if let Some(constant) = graph.constant_operand_ids_to_handles.get_mut(&(id as u32)) {
            constant.data = fp16_to_f32(&constant.data);
        }
    }
    for op in graph.operations.iter_mut() {
        if let Operation::ArgMax {
            options: Some(options),
            ..
        }
        | Operation::ArgMin {
            options: Some(options),
            ..
        } = op
            && options.output_data_type == MLOperandDataType::Float16
        {
            options.output_data_type = MLOperandDataType::Float32;
        }
    }
    true
}

fn build_input_handles(
    sorted_inputs: &[(&str, u32, &MLTensor)],
    tensors: &mut [LiteRtTensor],
    spatial_ids: &std::collections::HashSet<u32>,
    filter_info: &std::collections::HashMap<u32, (String, Vec<i32>, bool)>,
    float16_emulated: bool,
) -> (Vec<sys::LiteRtTensorBuffer>, Vec<LiteRtTensor>) {
    let mut in_raw = Vec::with_capacity(sorted_inputs.len());
    let mut temp_in_tensors: Vec<LiteRtTensor> = Vec::new();
    for (_name, operand_id, t) in sorted_inputs {
        let fp16 = float16_emulated && t.descriptor().data_type() == MLOperandDataType::Float16;
        // Convert float16 first, so the transpose below works on float32.
        let model_data = |tensors: &[LiteRtTensor]| {
            let logical = t.descriptor().rustnn_required_bytes();
            let mut raw = vec![0u8; logical];
            tensors[t.id].read(&mut raw).ok();
            if fp16 { fp16_to_f32(&raw) } else { raw }
        };
        if spatial_ids.contains(operand_id) {
            let shape = t.descriptor().shape();
            if shape.len() == 4 {
                if let Some((filter_layout, target_shape, _is_depthwise)) =
                    filter_info.get(operand_id)
                {
                    let temp = LiteRtTensor::create_litert_tensor(
                        target_shape,
                        model_element_type(t),
                        true,
                    )
                    .expect("temp filter tensor");
                    let transposed =
                        transpose_filter_to_ohwi(&model_data(tensors), shape, filter_layout);
                    temp.write(&transposed).ok();
                    temp_in_tensors.push(temp);
                    in_raw.push(temp_in_tensors.last().unwrap().handle);
                    continue;
                }
                let nhwc_data = transpose_nchw_to_nhwc(&model_data(tensors), shape);
                let temp = if fp16 {
                    let dims = vec![
                        shape[0] as i32,
                        shape[2] as i32,
                        shape[3] as i32,
                        shape[1] as i32,
                    ];
                    LiteRtTensor::create_litert_tensor(&dims, litert::ElementType::Float32, true)
                        .expect("temp input tensor")
                } else {
                    LiteRtTensor::new_with_layout(t.descriptor(), true).expect("temp input tensor")
                };
                temp.write(&nhwc_data).ok();
                temp_in_tensors.push(temp);
                in_raw.push(temp_in_tensors.last().unwrap().handle);
                continue;
            }
        }
        if fp16 {
            let temp = LiteRtTensor::create_litert_tensor(
                &tensor_dims(t),
                litert::ElementType::Float32,
                false,
            )
            .expect("float32 input tensor");
            temp.write(&model_data(tensors)).ok();
            temp_in_tensors.push(temp);
            in_raw.push(temp_in_tensors.last().unwrap().handle);
            continue;
        }
        in_raw.push(tensors[t.id].handle);
    }
    (in_raw, temp_in_tensors)
}

fn build_output_handles(
    sorted_outputs: &[(&str, u32, &MLTensor)],
    tensors: &[LiteRtTensor],
    spatial_ids: &std::collections::HashSet<u32>,
    bool_operand_ids: &std::collections::HashSet<u32>,
    float16_emulated: bool,
) -> (Vec<sys::LiteRtTensorBuffer>, Vec<LiteRtTensor>) {
    let mut out_raw = Vec::with_capacity(sorted_outputs.len());
    let mut temp_out_tensors: Vec<LiteRtTensor> = Vec::new();
    for (_name, operand_id, t) in sorted_outputs {
        let spatial = spatial_ids.contains(operand_id) && t.descriptor().shape().len() == 4;
        if float16_emulated && t.descriptor().data_type() == MLOperandDataType::Float16 {
            let dims = tensor_dims(t);
            let dims = if spatial {
                vec![dims[0], dims[2], dims[3], dims[1]]
            } else {
                dims
            };
            let temp =
                LiteRtTensor::create_litert_tensor(&dims, litert::ElementType::Float32, spatial)
                    .expect("float32 output tensor");
            temp_out_tensors.push(temp);
            out_raw.push(temp_out_tensors.last().unwrap().handle);
            continue;
        }
        if spatial_ids.contains(operand_id) {
            let shape = t.descriptor().shape();
            if shape.len() == 4 {
                let temp = LiteRtTensor::new_with_layout(t.descriptor(), true)
                    .expect("temp output tensor");
                temp_out_tensors.push(temp);
                out_raw.push(temp_out_tensors.last().unwrap().handle);
                continue;
            }
        }
        if bool_operand_ids.contains(operand_id) {
            let dims: Vec<i32> = t.descriptor().shape().iter().map(|&d| d as i32).collect();
            let temp = LiteRtTensor::create_litert_tensor(&dims, litert::ElementType::Bool, false)
                .expect("bool output tensor");
            temp_out_tensors.push(temp);
            out_raw.push(temp_out_tensors.last().unwrap().handle);
            continue;
        }
        out_raw.push(tensors[t.id].handle);
    }
    (out_raw, temp_out_tensors)
}

fn readback_outputs(
    sorted_outputs: &[(&str, u32, &MLTensor)],
    out_raw: &[sys::LiteRtTensorBuffer],
    temp_out_tensors: &[LiteRtTensor],
    tensors: &mut [LiteRtTensor],
    bool_operand_ids: &std::collections::HashSet<u32>,
    spatial_ids: &std::collections::HashSet<u32>,
    float16_emulated: bool,
) {
    for ((_name, operand_id, t), out_handle) in sorted_outputs.iter().zip(out_raw.iter()) {
        if float16_emulated && t.descriptor().data_type() == MLOperandDataType::Float16 {
            let logical = t.descriptor().rustnn_required_bytes();
            let mut buf = vec![0u8; logical * 2];
            if let Some(temp) = temp_out_tensors.iter().find(|tt| tt.handle == *out_handle) {
                temp.read(&mut buf).ok();
                let shape = t.descriptor().shape();
                let buf = if spatial_ids.contains(operand_id) && shape.len() == 4 {
                    transpose_nhwc_to_nchw(&buf, shape)
                } else {
                    buf
                };
                tensors[t.id].write(&f32_to_fp16(&buf)).ok();
            }
        } else if bool_operand_ids.contains(operand_id) {
            let logical = t.descriptor().rustnn_required_bytes();
            let mut buf = vec![0u8; logical];
            if let Some(temp) = temp_out_tensors.iter().find(|tt| tt.handle == *out_handle) {
                temp.read(&mut buf).ok();
                tensors[t.id].write(&buf).ok();
            }
        } else if spatial_ids.contains(operand_id) {
            let shape = t.descriptor().shape();
            if shape.len() == 4 {
                let logical = t.descriptor().rustnn_required_bytes();
                let mut nhwc_buf = vec![0u8; logical];
                if let Some(temp) = temp_out_tensors.iter().find(|tt| tt.handle == *out_handle) {
                    temp.read(&mut nhwc_buf).ok();
                    let nchw_data = transpose_nhwc_to_nchw(&nhwc_buf, shape);
                    tensors[t.id].write(&nchw_data).ok();
                }
            }
        }
    }
}

impl LiteRtContext {
    pub(crate) fn new_from_device_type(
        device_type: DeviceType,
        _rustnn_options: Option<&RustNNOptions>,
    ) -> Result<Self> {
        let _ = litert::set_global_log_severity(litert::LogSeverity::Warning);
        LiteRt::env();
        Ok(Self {
            tensors: Vec::new(),
            device_type,
            needs_layout_fix: false,
        })
    }

    fn accelerator_bits(&self) -> sys::LiteRtHwAcceleratorSet {
        match self.device_type {
            DeviceType::Cpu => sys::kLiteRtHwAcceleratorCpu as _,
            DeviceType::Gpu => (sys::kLiteRtHwAcceleratorGpu | sys::kLiteRtHwAcceleratorCpu) as _,
            DeviceType::Npu => (sys::kLiteRtHwAcceleratorNpu | sys::kLiteRtHwAcceleratorCpu) as _,
        }
    }
}

// Impls for Backend Trait
impl ListDevices for LiteRtContext {
    fn list_devices() -> Vec<crate::backend_selection::BackendDevice> {
        if LiteRt::env().is_null() {
            return vec![];
        }
        vec![
            crate::backend_selection::BackendDevice::LiteRt {
                device_type: DeviceType::Cpu,
            },
            crate::backend_selection::BackendDevice::LiteRt {
                device_type: DeviceType::Gpu,
            },
            crate::backend_selection::BackendDevice::LiteRt {
                device_type: DeviceType::Npu,
            },
        ]
    }
}

impl<'context> MLBackendContext<'context> for LiteRtContext {
    fn accelerated(&self) -> bool {
        self.device_type != DeviceType::Cpu
    }

    fn create_builder<'builder>(
        &mut self,
    ) -> Result<Box<dyn MLBackendBuilder<'context, 'builder> + 'builder>>
    where
        'context: 'builder,
    {
        Ok(Box::new(LiteRtBuilder {
            accelerator_bits: self.accelerator_bits(),
        }))
    }

    fn create_tensor(&mut self, descriptor: &MLTensorDescriptor) -> Result<MLTensor> {
        let tensor = LiteRtTensor::new_with_layout(descriptor, self.needs_layout_fix)?;
        self.tensors.push(tensor);
        Ok(MLTensor {
            id: self.tensors.len() - 1,
            constant: false,
            descriptor: descriptor.clone(),
        })
    }

    fn rustnn_resize_tensor(&mut self, _tensor: &mut MLTensor, _new_shape: &[u64]) -> Result<()> {
        todo!("Not Implemented yet.")
    }

    fn rustnn_set_tensor_capacity(
        &mut self,
        _tensor: &mut MLTensor,
        _max_shape: &[u64],
    ) -> Result<()> {
        todo!("Not Implemented yet.")
    }

    fn create_constant_tensor(
        &mut self,
        descriptor: &MLTensorDescriptor,
        input_data: &[u8],
    ) -> Result<MLTensor> {
        let mut tensor = self.create_tensor(descriptor)?;
        tensor.constant = true;
        self.write_tensor(&tensor, input_data)
            .map_err(|e| Error::TensorCreationError {
                source: e.into(),
                descriptor: descriptor.clone(),
            })?;
        Ok(tensor)
    }

    fn read_tensor(&mut self, tensor: &MLTensor, array: &mut [u8]) -> Result<()> {
        let logical = tensor.descriptor().rustnn_required_bytes();
        if array.len() < logical {
            return Err(Error::TensorReadError {
                source: format!(
                    "buffer too small: need {} logical bytes, got {}",
                    logical,
                    array.len()
                )
                .into(),
                tensor: tensor.clone(),
            });
        }
        self.tensors[tensor.id].read(&mut array[..logical])?;
        Ok(())
    }

    fn write_tensor(&mut self, tensor: &MLTensor, array: &[u8]) -> Result<()> {
        let logical = tensor.descriptor().rustnn_required_bytes();
        if array.len() < logical {
            return Err(Error::TensorWriteError {
                source: format!(
                    "write too small for tensor: {} bytes < {} logical bytes",
                    array.len(),
                    logical,
                )
                .into(),
                tensor: tensor.clone(),
            });
        }
        self.tensors[tensor.id].write(&array[..logical])?;
        Ok(())
    }

    fn dispatch(
        &mut self,
        graph: &mut MLGraph,
        inputs: &MLNamedTensors,
        outputs: &MLNamedTensors,
    ) -> Result<()> {
        let lite_graph = match &graph.backend {
            crate::mlcontext::MLBackendGraph::LiteRtGraph(graph) => graph,
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "litert".to_string(),
                    reason: "expected LiteRtGraph in dispatch".to_string(),
                }
                .into());
            }
        };

        let sorted_inputs = order_by_signature(&lite_graph.input_order, inputs);
        let sorted_outputs = order_by_signature(&lite_graph.output_order, outputs);

        let (in_raw, _temp_in_tensors) = build_input_handles(
            &sorted_inputs,
            &mut self.tensors,
            &lite_graph.spatial_operand_ids,
            &lite_graph.filter_transpose_info,
            lite_graph.float16_emulated,
        );

        let (mut out_raw, temp_out_tensors) = build_output_handles(
            &sorted_outputs,
            &self.tensors,
            &lite_graph.spatial_operand_ids,
            &lite_graph.bool_operand_ids,
            lite_graph.float16_emulated,
        );

        lite_graph.run(&in_raw, &mut out_raw)?;

        readback_outputs(
            &sorted_outputs,
            &out_raw,
            &temp_out_tensors,
            &mut self.tensors,
            &lite_graph.bool_operand_ids,
            &lite_graph.spatial_operand_ids,
            lite_graph.float16_emulated,
        );

        Ok(())
    }
}

pub(crate) struct LiteRtBuilder {
    accelerator_bits: sys::LiteRtHwAcceleratorSet,
}

impl fmt::Debug for LiteRtBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiteRtBuilder").finish()
    }
}

impl<'context, 'builder> MLBackendBuilder<'context, 'builder> for LiteRtBuilder {
    fn build(&mut self, graph_info: GraphInfo) -> Result<MLGraph<'context>> {
        let (input_descriptors, output_descriptors) = graph_info
            .io_binding_maps()
            .map_err(|e| Error::GraphBuildError { source: e.into() })?;
        let operand_order = |ids: &[u32]| -> Vec<(String, u32)> {
            ids.iter()
                .filter_map(|&id| {
                    graph_info
                        .operand(id)
                        .and_then(|o| o.name.clone())
                        .map(|name| (name, id))
                })
                .collect()
        };
        let input_order = operand_order(&graph_info.input_operands);
        let output_order = operand_order(&graph_info.output_operands);

        let (spatial_operand_names, filter_transpose_info) = collect_spatial_info(&graph_info);
        let mut graph_info = graph_info;
        // A float16 graph runs as float32; the buffers convert at dispatch.
        let float16_emulated = emulate_float16(&mut graph_info);
        modify_graph_for_nhwc(&mut graph_info, &spatial_operand_names);
        let tflite_bytes = LiteRtConverter.convert(&graph_info)?.data;

        let bool_operand_ids = collect_bool_operand_names(&graph_info);

        let graph = LiteRtGraph::new(
            tflite_bytes,
            self.accelerator_bits,
            spatial_operand_names,
            filter_transpose_info,
            bool_operand_ids,
            input_order,
            output_order,
            float16_emulated,
        )
        .map_err(|e| Error::GraphBuildError {
            source: format!("failed to compile model: {e}").into(),
        })?;

        Ok(MLGraph {
            backend: crate::mlcontext::MLBackendGraph::LiteRtGraph(graph),
            input_descriptors,
            output_descriptors,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_enums::MLOperandDataType;

    fn make_desc(dt: MLOperandDataType, shape: Vec<u64>) -> MLTensorDescriptor {
        MLTensorDescriptor::new(dt, shape)
    }

    #[test]
    fn test_context_new() {
        let ctx = LiteRtContext::new_from_device_type(DeviceType::Cpu, None).unwrap();
        assert_eq!(ctx.tensors.len(), 0);
    }

    #[test]
    fn test_create_tensor() {
        let mut ctx = LiteRtContext::new_from_device_type(DeviceType::Cpu, None).unwrap();
        let desc = make_desc(MLOperandDataType::Float32, vec![1, 4]);
        let tensor = ctx.create_tensor(&desc).unwrap();
        assert_eq!(tensor.id, 0);
        assert!(!tensor.constant);
        assert_eq!(ctx.tensors.len(), 1);
    }

    #[test]
    fn test_write_and_read_tensor() {
        let mut ctx = LiteRtContext::new_from_device_type(DeviceType::Cpu, None).unwrap();
        let desc = make_desc(MLOperandDataType::Float32, vec![2]);
        let tensor = ctx.create_tensor(&desc).unwrap();

        let data: Vec<u8> = vec![0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x40];
        ctx.write_tensor(&tensor, &data).unwrap();

        let mut read_buf = vec![0u8; 8];
        ctx.read_tensor(&tensor, &mut read_buf).unwrap();
        assert_eq!(read_buf, data);
    }

    #[test]
    fn nchw_axes_map_to_their_nhwc_positions() {
        assert_eq!([0, 1, 2, 3].map(nchw_axis_to_nhwc), [0, 3, 1, 2]);
    }

    #[test]
    fn per_axis_arrays_are_reordered_or_left_alone() {
        assert_eq!(permute_nchw_to_nhwc(&[1, 2, 3, 4]), vec![1, 3, 4, 2]);
        assert_eq!(permute_nchw_to_nhwc(&[1, 2]), vec![1, 2]);
        assert_eq!(permute_nchw_to_nhwc(&[1, 2, 3, 4, 5]), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn relayout_conjugates_a_both_sides_permutation() {
        assert_eq!(
            fold_relayout_into_permutation(&[0, 1, 2, 3]),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            fold_relayout_into_permutation(&[0, 2, 3, 1]),
            vec![0, 2, 3, 1]
        );
        assert_eq!(
            fold_relayout_into_permutation(&[0, 3, 1, 2]),
            vec![0, 3, 1, 2]
        );
        assert_eq!(
            fold_relayout_into_permutation(&[1, 0, 2, 3]),
            vec![3, 1, 2, 0]
        );
        assert_eq!(
            fold_relayout_into_permutation(&[0, 1, 2, 4]),
            vec![0, 1, 2, 4]
        );
        assert_eq!(fold_relayout_into_permutation(&[1, 0]), vec![1, 0]);
    }

    #[test]
    fn relayout_prefixes_an_output_only_permutation() {
        assert_eq!(
            fold_output_relayout_into_permutation(&[0, 1, 2, 3]),
            vec![0, 2, 3, 1]
        );
        assert_eq!(
            fold_output_relayout_into_permutation(&[0, 3, 1, 2]),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            fold_output_relayout_into_permutation(&[0, 1, 2, 4]),
            vec![0, 1, 2, 4]
        );
    }
}

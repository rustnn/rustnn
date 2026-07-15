use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;
use std::sync::OnceLock;

use litert_sys::{self as sys};

use crate::backend_selection::DeviceType;
use crate::converters::{GraphConverter, LiteRtConverter};
use crate::error::{Error, Result};
use crate::mlcontext::{
    ListDevices, MLBackendBuilder, MLBackendContext, MLGraph, MLTensor, MLTensorDescriptor,
    RustNNOptions,
};

use crate::operator_enums::MLOperandDataType;
use crate::operators::Operation;
use crate::{GraphError, GraphInfo};

/// WebNN operations not (yet) supported by the LiteRT backend
const LITERT_UNSUPPORTED_OPS: &[&str] = &[
    "average_pool2d",
    "conv_transpose2d",
    "cumulativeSum",
    "dequantize_linear",
    "erf",
    "expand",
    "gatherElements",
    "gru",
    "gru_cell",
    "instance_normalization",
    "is_infinite",
    "is_nan",
    "layer_normalization",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "lstm",
    "lstm_cell",
    "max_pool2d",
    "quantize_linear",
    "reduce_l1",
    "reduce_l2",
    "reduce_log_sum",
    "reduce_log_sum_exp",
    "reduce_sum_square",
    "resample2d",
    "round_even",
    "scatter_elements",
    "split",
    "triangular",
    "where",
];

/// Returns the list of WebNN operations not supported by this backend.
pub fn unsupported_ops() -> &'static [&'static str] {
    LITERT_UNSUPPORTED_OPS
}

/// Returns true if this dtype is NOT supported for this operation.
pub fn dtype_unsupported_for_op(dtype: &str, op: &str) -> bool {
    match dtype.to_lowercase().as_str() {
        "float32" | "float16" => false,
        "int32" | "uint8" => !matches!(
            op,
            "equal"
                | "greater"
                | "greater_or_equal"
                | "lesser"
                | "lesser_or_equal"
                | "logical_and"
                | "logical_not"
                | "logical_or"
                | "not_equal"
                | "scatter_elements"
                | "where"
        ),
        _ => true,
    }
}

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
    /// Names of input/output operands that were spatially transposed NCHW→NHWC.
    spatial_operand_names: std::collections::HashSet<String>,
    /// Filter operand names whose runtime data needs layout→OHWI transpose.
    /// Maps filter name → (WebNN filter_layout, target shape, is_depthwise [unused]).
    filter_transpose_info: std::collections::HashMap<String, (String, Vec<i32>, bool)>,
    /// Output operand names needing BOOL type (WHERE condition, comparison ops).
    bool_operand_names: std::collections::HashSet<String>,
}

unsafe impl Send for LiteRtGraph {}
unsafe impl Sync for LiteRtGraph {}

impl LiteRtGraph {
    fn new(
        model_bytes: Vec<u8>,
        accelerator_bits: sys::LiteRtHwAcceleratorSet,
        spatial_operand_names: std::collections::HashSet<String>,
        filter_transpose_info: std::collections::HashMap<String, (String, Vec<i32>, bool)>,
        bool_operand_names: std::collections::HashSet<String>,
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
                spatial_operand_names,
                filter_transpose_info,
                bool_operand_names,
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

pub fn is_spatial_op(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Conv2d { .. } | Operation::MaxPool2d { .. } | Operation::AveragePool2d { .. }
    )
}

fn collect_spatial_operand_names(graph_info: &GraphInfo) -> std::collections::HashSet<String> {
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
            Operation::MaxPool2d { options, .. } | Operation::AveragePool2d { options, .. } => {
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
                    || (matches!(op, Operation::Conv2d { .. })
                        && op.inputs().iter().position(|&x| x == id) == Some(1))
                {
                    continue;
                }
                if op_info.descriptor.shape.len() == 4 {
                    if let Some(ref n) = op_info.name {
                        names.insert(n.clone());
                    }
                }
            }
        }
        for &id in op.outputs() {
            if let Some(op_info) = graph_info.operand(id) {
                if op_info.descriptor.shape.len() == 4 {
                    if let Some(ref n) = op_info.name {
                        names.insert(n.clone());
                    }
                }
            }
        }
    }
    names
}

fn collect_filter_transpose_info(
    graph_info: &GraphInfo,
    spatial_operand_names: &mut std::collections::HashSet<String>,
) -> std::collections::HashMap<String, (String, Vec<i32>, bool)> {
    let mut filter_transpose_info: std::collections::HashMap<String, (String, Vec<i32>, bool)> =
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
                            if let Some(ref fname) = fop.name {
                                spatial_operand_names.remove(fname);
                            }
                        } else if let Some(ref fname) = fop.name {
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
                            spatial_operand_names.insert(fname.clone());
                            filter_transpose_info.insert(
                                fname.clone(),
                                (filter_layout.to_string(), target_shape, false),
                            );
                        }
                    }
                }
            }
        } else if let Some(&fid) = op.inputs().get(1) {
            if let Some(fop) = graph_info.operand(fid) {
                if let Some(ref fname) = fop.name {
                    spatial_operand_names.remove(fname);
                }
            }
        }
    }
    filter_transpose_info
}

fn collect_spatial_info(
    graph_info: &GraphInfo,
) -> (
    std::collections::HashSet<String>,
    std::collections::HashMap<String, (String, Vec<i32>, bool)>,
) {
    let mut spatial_operand_names = collect_spatial_operand_names(graph_info);
    let filter_transpose_info =
        collect_filter_transpose_info(graph_info, &mut spatial_operand_names);
    (spatial_operand_names, filter_transpose_info)
}

fn collect_bool_operand_names(graph: &GraphInfo) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    for op in &graph.operations {
        match op {
            Operation::Where { .. } => {
                if let Some(&cond_id) = op.inputs().get(0) {
                    if let Some(op_info) = graph.operand(cond_id) {
                        if let Some(ref n) = op_info.name {
                            names.insert(n.clone());
                        }
                    }
                }
            }
            Operation::Equal { .. }
            | Operation::Greater { .. }
            | Operation::GreaterOrEqual { .. }
            | Operation::Lesser { .. }
            | Operation::LesserOrEqual { .. }
            | Operation::NotEqual { .. } => {
                for &out_id in op.outputs() {
                    if let Some(op_info) = graph.operand(out_id) {
                        if let Some(ref n) = op_info.name {
                            names.insert(n.clone());
                        }
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
    spatial_operand_names: &std::collections::HashSet<String>,
) {
    let mut skip_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for op in &graph.operations {
        if let Operation::Conv2d { .. } = op {
            if let Some(&fid) = op.inputs().get(1) {
                skip_ids.insert(fid);
            }
        }
    }
    for (i, operand) in graph.operands.iter_mut().enumerate() {
        let id = i as u32;
        if skip_ids.contains(&id) {
            continue;
        }
        if operand.descriptor.shape.len() != 4 {
            continue;
        }
        let name = operand.name.as_deref().unwrap_or("");
        let in_set = spatial_operand_names.contains(name);
        if operand.kind == crate::graph::OperandKind::Constant {
            let is_spatial_input = graph.operations.iter().any(|op| {
                is_spatial_op(op)
                    && op.inputs().contains(&id)
                    && match op {
                        Operation::Conv2d { options, .. } => {
                            let l = options
                                .as_ref()
                                .map(|o| o.input_layout.as_str())
                                .unwrap_or("");
                            l.is_empty() || l.eq_ignore_ascii_case("nchw")
                        }
                        Operation::MaxPool2d { options, .. }
                        | Operation::AveragePool2d { options, .. } => {
                            let l = options.as_ref().map(|o| o.layout.as_str()).unwrap_or("");
                            l.is_empty() || l.eq_ignore_ascii_case("nchw")
                        }
                        _ => false,
                    }
            });
            // Skip if not a spatial input, or if shared with non-spatial ops (would corrupt data).
            let has_non_spatial_consumer = graph
                .operations
                .iter()
                .any(|other_op| !is_spatial_op(other_op) && other_op.inputs().contains(&id));
            if !is_spatial_input || has_non_spatial_consumer {
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

/// Transpose weight data from OIHW [O,I,H,W] to OHWI [O,H,W,I] layout.
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

/// Transpose weight data from HWIO [H,W,I,O] to OHWI [O,H,W,I] layout.
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

/// Transpose weight data from IHWO [I,H,W,O] to OHWI [O,H,W,I] layout.
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
        "ohwi" => shape.to_vec(),
        _ => vec![shape[0], shape[2], shape[3], shape[1]], // "oihw" default
    }
}

fn build_input_handles(
    sorted_inputs: &[(&str, &MLTensor)],
    tensors: &mut [LiteRtTensor],
    spatial_names: &std::collections::HashSet<String>,
    filter_info: &std::collections::HashMap<String, (String, Vec<i32>, bool)>,
) -> (Vec<sys::LiteRtTensorBuffer>, Vec<LiteRtTensor>) {
    let mut in_raw = Vec::with_capacity(sorted_inputs.len());
    let mut temp_in_tensors: Vec<LiteRtTensor> = Vec::new();
    for (name, t) in sorted_inputs {
        if spatial_names.contains(*name) {
            let shape = t.descriptor().shape();
            if shape.len() == 4 {
                if let Some((filter_layout, target_shape, _is_depthwise)) = filter_info.get(*name) {
                    let element_type =
                        ml_operand_to_litert_element_type(t.descriptor().data_type())
                            .expect("filter element type");
                    let temp = LiteRtTensor::create_litert_tensor(target_shape, element_type, true)
                        .expect("temp filter tensor");
                    let logical = t.descriptor().rustnn_required_bytes();
                    let mut src_data = vec![0u8; logical];
                    tensors[t.id].read(&mut src_data).ok();
                    let transposed = transpose_filter_to_ohwi(&src_data, shape, filter_layout);
                    temp.write(&transposed).ok();
                    temp_in_tensors.push(temp);
                    in_raw.push(temp_in_tensors.last().unwrap().handle);
                    continue;
                }
                let logical = t.descriptor().rustnn_required_bytes();
                let mut nchw_data = vec![0u8; logical];
                tensors[t.id].read(&mut nchw_data).ok();
                let nhwc_data = transpose_nchw_to_nhwc(&nchw_data, shape);
                let temp =
                    LiteRtTensor::new_with_layout(t.descriptor(), true).expect("temp input tensor");
                temp.write(&nhwc_data).ok();
                temp_in_tensors.push(temp);
                in_raw.push(temp_in_tensors.last().unwrap().handle);
                continue;
            }
        }
        in_raw.push(tensors[t.id].handle);
    }
    (in_raw, temp_in_tensors)
}

fn build_output_handles(
    sorted_outputs: &[(&str, &MLTensor)],
    tensors: &[LiteRtTensor],
    spatial_names: &std::collections::HashSet<String>,
    bool_operand_names: &std::collections::HashSet<String>,
) -> (Vec<sys::LiteRtTensorBuffer>, Vec<LiteRtTensor>) {
    let mut out_raw = Vec::with_capacity(sorted_outputs.len());
    let mut temp_out_tensors: Vec<LiteRtTensor> = Vec::new();
    for (name, t) in sorted_outputs {
        if spatial_names.contains(*name) {
            let shape = t.descriptor().shape();
            if shape.len() == 4 {
                let temp = LiteRtTensor::new_with_layout(t.descriptor(), true)
                    .expect("temp output tensor");
                temp_out_tensors.push(temp);
                out_raw.push(temp_out_tensors.last().unwrap().handle);
                continue;
            }
        }
        if bool_operand_names.contains(*name) {
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
    sorted_outputs: &[(&str, &MLTensor)],
    out_raw: &[sys::LiteRtTensorBuffer],
    temp_out_tensors: &[LiteRtTensor],
    tensors: &mut [LiteRtTensor],
    bool_operand_names: &std::collections::HashSet<String>,
    spatial_names: &std::collections::HashSet<String>,
) {
    for ((name, t), out_handle) in sorted_outputs.iter().zip(out_raw.iter()) {
        if bool_operand_names.contains(*name) {
            let logical = t.descriptor().rustnn_required_bytes();
            let mut buf = vec![0u8; logical];
            if let Some(temp) = temp_out_tensors.iter().find(|tt| tt.handle == *out_handle) {
                temp.read(&mut buf).ok();
                tensors[t.id].write(&buf).ok();
            }
        } else if spatial_names.contains(*name) {
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
        inputs: &HashMap<&str, &MLTensor>,
        outputs: &HashMap<&str, &MLTensor>,
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

        let mut sorted_inputs: Vec<(&str, &MLTensor)> =
            inputs.iter().map(|(k, v)| (*k, *v)).collect();
        sorted_inputs.sort_by_key(|(name, _)| *name);

        let mut sorted_outputs: Vec<(&str, &MLTensor)> =
            outputs.iter().map(|(k, v)| (*k, *v)).collect();
        sorted_outputs.sort_by_key(|(name, _)| *name);

        let (in_raw, _temp_in_tensors) = build_input_handles(
            &sorted_inputs,
            &mut self.tensors,
            &lite_graph.spatial_operand_names,
            &lite_graph.filter_transpose_info,
        );

        let (mut out_raw, temp_out_tensors) = build_output_handles(
            &sorted_outputs,
            &self.tensors,
            &lite_graph.spatial_operand_names,
            &lite_graph.bool_operand_names,
        );

        lite_graph.run(&in_raw, &mut out_raw)?;

        readback_outputs(
            &sorted_outputs,
            &out_raw,
            &temp_out_tensors,
            &mut self.tensors,
            &lite_graph.bool_operand_names,
            &lite_graph.spatial_operand_names,
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
        let (spatial_operand_names, filter_transpose_info) = collect_spatial_info(&graph_info);
        let mut graph_info = graph_info;
        modify_graph_for_nhwc(&mut graph_info, &spatial_operand_names);
        let tflite_bytes = LiteRtConverter.convert(&graph_info)?.data;
        let bool_operand_names = collect_bool_operand_names(&graph_info);

        let graph = LiteRtGraph::new(
            tflite_bytes,
            self.accelerator_bits,
            spatial_operand_names,
            filter_transpose_info,
            bool_operand_names,
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
}

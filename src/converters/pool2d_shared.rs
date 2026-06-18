//! Shared pool2d helpers for ONNX, TensorRT, and shape inference.

use crate::graph::{GraphInfo, get_static_or_max_size};
use crate::operators::Operation;

/// Map WebNN [`output_sizes`](crate::operator_options::MLPool2dOptions::output_sizes) to ONNX `ceil_mode`
/// / TensorRT round-up: `0` = floor spatial shape, `1` = ceil spatial shape.
///
/// Returns [`None`] when `output_sizes` is absent, not length-2, or does not match either implicit shape.
pub(crate) fn infer_pool2d_ceil_mode_from_output_sizes(
    op: &Operation,
    graph: &GraphInfo,
) -> Option<i64> {
    let opts = match &op {
        Operation::AveragePool2d { options, .. }
        | Operation::MaxPool2d { options, .. }
        | Operation::L2Pool2d { options, .. }
        | Operation::GlobalAveragePool { options, .. }
        | Operation::GlobalMaxPool { options, .. } => options.as_ref()?,
        _ => return None,
    };

    let target: Vec<i64> = opts
        .output_sizes
        .as_ref()?
        .iter()
        .map(|&u| u as i64)
        .collect();
    if target.len() != 2 {
        return None;
    }

    let input_id = *op.input_operands().first()?;
    let input_shape = &graph.operand(input_id)?.descriptor.shape;
    if input_shape.len() != 4 {
        return None;
    }
    let layout = opts.layout.to_ascii_lowercase();
    let (input_h, input_w) = if layout == "nhwc" {
        (
            get_static_or_max_size(&input_shape[1]) as i64,
            get_static_or_max_size(&input_shape[2]) as i64,
        )
    } else {
        (
            get_static_or_max_size(&input_shape[2]) as i64,
            get_static_or_max_size(&input_shape[3]) as i64,
        )
    };

    let kernel: Vec<i64> = opts
        .window_dimensions
        .as_ref()
        .map(|v| v.iter().map(|&u| u as i64).collect())
        .or_else(|| {
            if layout == "nhwc" {
                Some(vec![
                    get_static_or_max_size(&input_shape[1]) as i64,
                    get_static_or_max_size(&input_shape[2]) as i64,
                ])
            } else {
                Some(vec![
                    get_static_or_max_size(&input_shape[2]) as i64,
                    get_static_or_max_size(&input_shape[3]) as i64,
                ])
            }
        })?;
    if kernel.len() != 2 {
        return None;
    }
    let strides: Vec<i64> = if opts.strides.is_empty() {
        vec![1, 1]
    } else {
        opts.strides.iter().map(|&u| u as i64).collect()
    };
    let dilations: Vec<i64> = if opts.dilations.is_empty() {
        vec![1, 1]
    } else {
        opts.dilations.iter().map(|&u| u as i64).collect()
    };
    let pads: Vec<i64> = if opts.padding.len() == 4 {
        vec![
            opts.padding[0] as i64,
            opts.padding[2] as i64,
            opts.padding[1] as i64,
            opts.padding[3] as i64,
        ]
    } else if opts.padding.is_empty() {
        vec![0, 0, 0, 0]
    } else {
        opts.padding.iter().map(|&u| u as i64).collect()
    };
    if strides.len() != 2 || dilations.len() != 2 || pads.len() != 4 {
        return None;
    }

    let eff_h = dilations[0] * (kernel[0] - 1) + 1;
    let eff_w = dilations[1] * (kernel[1] - 1) + 1;
    let numer_h = input_h + pads[0] + pads[2] - eff_h;
    let numer_w = input_w + pads[1] + pads[3] - eff_w;
    if numer_h < 0 || numer_w < 0 {
        return None;
    }
    let floor_h = (numer_h / strides[0]) + 1;
    let floor_w = (numer_w / strides[1]) + 1;
    let ceil_h = ((numer_h + strides[0] - 1) / strides[0]) + 1;
    let ceil_w = ((numer_w + strides[1] - 1) / strides[1]) + 1;

    if target[0] == floor_h && target[1] == floor_w {
        Some(0)
    } else if target[0] == ceil_h && target[1] == ceil_w {
        Some(1)
    } else {
        None
    }
}

/// Check whether ONNX `ceil_mode=1` would drop the last edge window for
/// a single spatial dimension.
///
/// ## Background
///
/// **WebNN spec** (§8.9.37) defines ceil rounding as pure math:
/// ```text
/// output size = ceil(1 + (input_size - filter_size + pad_start + pad_end) / stride)
/// ```
/// WebNN has no constraint about edge windows — it always produces the pure
/// mathematical ceiling result, even when the last window starts entirely in
/// the padded region. Those extra positions are filled with the padding value
/// (0 for maxPool2d, -INF for averagePool2d).
///
/// **ONNX** `ceil_mode=1` computes the same ceiling value but then applies an
/// additional boundary constraint: if the last window starts at or beyond
/// `input_size + pad_start`, it is dropped because it covers only padding.
/// This is the PyTorch/ONNX Runtime convention.
///
/// ### Example where they diverge
///
/// Input=5, kernel=3, stride=3, pad=[1,1] (symmetric):
/// - WebNN: ceil(1 + (5-3+1+1)/3) = ceil(2.333) = **3**
/// - ONNX ceil_mode=1: same math = 3, but window 3 starts at index 6 (= 5+1),
///   which is the right-padding-only region → **dropped** → result = **2**
///
/// Returns `true` when ONNX would drop the last window for this dimension,
/// meaning we must use floor pooling + post-pool Pad to reach the WebNN output.
///
/// ## ONNX boundary condition
///
/// ONNX drops when: `(ceil_out - 1) * stride >= input_size + pad_start`
///
/// ## Parameters (ONNX pad convention)
///
/// - `input_size`, `kernel_size`, `stride`, `pad_start`, `pad_end`, `dilation`
pub(crate) fn onnx_ceil_drops_edge_window(
    input_size: i64,
    kernel_size: i64,
    stride: i64,
    pad_start: i64,
    pad_end: i64,
    dilation: i64,
) -> bool {
    let eff_size = dilation * (kernel_size - 1) + 1;
    let numer = input_size + pad_start + pad_end - eff_size;
    if numer < 0 {
        return false;
    }
    let ceil_out = (numer + stride - 1) / stride + 1;
    let floor_out = numer / stride + 1;
    // If ceil == floor, no window is dropped
    if ceil_out == floor_out {
        return false;
    }
    // ONNX boundary check: last window starts at (ceil_out - 1) * stride
    // If that position >= input_size + pad_start, the window is in padding-only region
    ceil_out.saturating_sub(1) * stride >= input_size + pad_start
}

/// Compute the WebNN ceil output size for a single pooling dimension.
/// Pure math ceiling — no edge-window dropping.
pub(crate) fn webnn_ceil_output_size(
    input_size: i64,
    kernel_size: i64,
    stride: i64,
    pad_start: i64,
    pad_end: i64,
    dilation: i64,
) -> i64 {
    let eff_size = dilation * (kernel_size - 1) + 1;
    let numer = input_size + pad_start + pad_end - eff_size;
    if numer < 0 {
        return 1;
    }
    (numer + stride - 1) / stride + 1
}

/// Compute the floor output size for a single pooling dimension.
pub(crate) fn webnn_floor_output_size(
    input_size: i64,
    kernel_size: i64,
    stride: i64,
    pad_start: i64,
    pad_end: i64,
    dilation: i64,
) -> i64 {
    let eff_size = dilation * (kernel_size - 1) + 1;
    let numer = input_size + pad_start + pad_end - eff_size;
    if numer < 0 {
        return 1;
    }
    numer / stride + 1
}

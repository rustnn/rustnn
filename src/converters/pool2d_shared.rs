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

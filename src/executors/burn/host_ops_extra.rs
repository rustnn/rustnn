//! Additional host-side WebNN operators for the Burn executor.

use crate::error::GraphError;
use crate::graph::DataType;
use crate::operator_options::{
    MLArgMinMaxOptions, MLCumulativeSumOptions, MLGemmOptions, MLGruCellOptions, MLGruOptions,
    MLLstmCellOptions, MLLstmOptions, MLPadOptions, MLPool2dOptions, MLResample2dOptions,
    MLReverseOptions, MLScatterOptions, MLSqueezeOptions, MLTriangularOptions, MLUnsqueezeOptions,
};
use crate::shape_inference::infer_pad_shape;

use super::host_array::{
    HostArray, PoolKind, broadcast_shapes, broadcast_to, burn_err, linear_index, matmul, pool2d,
    transpose, unravel,
};

// ---------------------------------------------------------------------------
// Unary elementwise
// ---------------------------------------------------------------------------

pub fn cos(arr: &HostArray) -> HostArray {
    arr.map_unary(f32::cos)
}

pub fn sin(arr: &HostArray) -> HostArray {
    arr.map_unary(f32::sin)
}

pub fn tan(arr: &HostArray) -> HostArray {
    arr.map_unary(f32::tan)
}

pub fn erf(arr: &HostArray) -> HostArray {
    arr.map_unary(erf_f32)
}

pub fn reciprocal(arr: &HostArray) -> HostArray {
    arr.map_unary(|x| 1.0 / x)
}

pub fn sign(arr: &HostArray) -> HostArray {
    arr.map_unary(|x| {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    })
}

pub fn gelu(arr: &HostArray) -> HostArray {
    arr.map_unary(|x| 0.5 * x * (1.0 + erf_f32(x / std::f32::consts::SQRT_2)))
}

pub fn softplus(arr: &HostArray) -> HostArray {
    arr.map_unary(|x| {
        if x > 20.0 {
            x
        } else if x < -20.0 {
            0.0
        } else {
            (1.0 + x.exp()).ln()
        }
    })
}

pub fn softsign(arr: &HostArray) -> HostArray {
    arr.map_unary(|x| x / (1.0 + x.abs()))
}

pub fn is_nan(arr: &HostArray) -> HostArray {
    arr.map_unary(|x| if x.is_nan() { 1.0 } else { 0.0 })
}

pub fn is_infinite(arr: &HostArray) -> HostArray {
    arr.map_unary(|x| if x.is_infinite() { 1.0 } else { 0.0 })
}

pub fn round_even(arr: &HostArray) -> HostArray {
    arr.map_unary(round_to_nearest_even)
}

pub fn squeeze(arr: &HostArray, options: &MLSqueezeOptions) -> Result<HostArray, GraphError> {
    let input_shape_u32: Vec<u32> = arr.shape.iter().map(|&d| d as u32).collect();
    let axes = if options.axes.is_empty() {
        None
    } else {
        Some(options.axes.as_slice())
    };
    let out_shape_u32 = crate::shape_inference::infer_squeeze_shape(&input_shape_u32, axes)?;
    let out_shape: Vec<usize> = out_shape_u32.iter().map(|&d| d as usize).collect();
    arr.reshape(&out_shape)
}

pub fn unsqueeze(arr: &HostArray, options: &MLUnsqueezeOptions) -> Result<HostArray, GraphError> {
    let input_shape_u32: Vec<u32> = arr.shape.iter().map(|&d| d as u32).collect();
    let out_shape_u32 =
        crate::shape_inference::infer_unsqueeze_shape(&input_shape_u32, &options.axes)?;
    let out_shape: Vec<usize> = out_shape_u32.iter().map(|&d| d as usize).collect();
    arr.reshape(&out_shape)
}

pub fn shape(arr: &HostArray) -> HostArray {
    HostArray {
        shape: vec![arr.rank()],
        data: arr.shape.iter().map(|&d| d as f32).collect(),
        i64_data: None,
        u64_data: None,
    }
}

pub fn tile(arr: &HostArray, repetitions: &[u32]) -> Result<HostArray, GraphError> {
    let input_shape_u32: Vec<u32> = arr.shape.iter().map(|&d| d as u32).collect();
    let out_shape_u32 = crate::shape_inference::infer_tile_shape(&input_shape_u32, repetitions)?;
    let out_shape: Vec<usize> = out_shape_u32.iter().map(|&d| d as usize).collect();
    let out_len = out_shape.iter().product::<usize>().max(1);
    let mut out = vec![0.0f32; out_len];
    for out_flat in 0..out_len {
        let out_coords = unravel(out_flat, &out_shape);
        let mut in_coords = Vec::with_capacity(arr.rank());
        for (i, &rep_coord) in out_coords.iter().enumerate() {
            let in_dim = arr.shape[i];
            in_coords.push(rep_coord % in_dim);
        }
        out[out_flat] = arr.data[linear_index(&in_coords, &arr.shape)];
    }
    Ok(HostArray {
        shape: out_shape,
        data: out,
        i64_data: None,
        u64_data: None,
    })
}

pub fn cumulative_sum(
    arr: &HostArray,
    axis: u32,
    options: &MLCumulativeSumOptions,
) -> Result<HostArray, GraphError> {
    let axis = axis as usize;
    if axis >= arr.rank() {
        return Err(burn_err(format!("cumulativeSum axis {axis} out of range")));
    }
    let mut out = arr.clone();
    let outer: usize = arr.shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = arr.shape[axis + 1..].iter().product::<usize>().max(1);
    let axis_size = arr.shape[axis];
    for o in 0..outer {
        for i in 0..inner {
            let mut sum = 0.0f32;
            let indices: Vec<usize> = if options.reversed {
                (0..axis_size).rev().collect()
            } else {
                (0..axis_size).collect()
            };
            for (step, &a) in indices.iter().enumerate() {
                let idx = o * axis_size * inner + a * inner + i;
                let v = arr.data[idx];
                if options.exclusive {
                    out.data[idx] = sum;
                    sum += v;
                } else {
                    sum += v;
                    out.data[idx] = sum;
                }
                let _ = step;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Binary / compare
// ---------------------------------------------------------------------------

pub fn not_equal(a: &HostArray, b: &HostArray) -> Result<HostArray, GraphError> {
    a.compare(b, |x, y| x != y)
}

pub fn logical_and(a: &HostArray, b: &HostArray) -> Result<HostArray, GraphError> {
    a.binary_broadcast(b, |x, y| if x != 0.0 && y != 0.0 { 1.0 } else { 0.0 })
}

pub fn logical_or(a: &HostArray, b: &HostArray) -> Result<HostArray, GraphError> {
    a.binary_broadcast(b, |x, y| if x != 0.0 || y != 0.0 { 1.0 } else { 0.0 })
}

pub fn logical_xor(a: &HostArray, b: &HostArray) -> Result<HostArray, GraphError> {
    a.binary_broadcast(b, |x, y| {
        let bx = x != 0.0;
        let by = y != 0.0;
        if bx != by { 1.0 } else { 0.0 }
    })
}

pub fn prelu(input: &HostArray, slope: &HostArray) -> Result<HostArray, GraphError> {
    let out_shape = broadcast_shapes(&input.shape, &slope.shape)?;
    let lhs = broadcast_to(input, &out_shape)?;
    let rhs = broadcast_to(slope, &out_shape)?;
    Ok(HostArray {
        shape: out_shape,
        data: lhs
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&x, &s)| if x >= 0.0 { x } else { s * x })
            .collect(),
        i64_data: None,
        u64_data: None,
    })
}

pub fn where_op(
    condition: &HostArray,
    true_value: &HostArray,
    false_value: &HostArray,
) -> Result<HostArray, GraphError> {
    let out_shape = broadcast_shapes(
        &broadcast_shapes(&condition.shape, &true_value.shape)?,
        &false_value.shape,
    )?;
    let cond = broadcast_to(condition, &out_shape)?;
    let t = broadcast_to(true_value, &out_shape)?;
    let f = broadcast_to(false_value, &out_shape)?;
    Ok(HostArray {
        shape: out_shape,
        data: cond
            .data
            .iter()
            .zip(t.data.iter().zip(f.data.iter()))
            .map(|(&c, (&tv, &fv))| if c != 0.0 { tv } else { fv })
            .collect(),
        i64_data: None,
        u64_data: None,
    })
}

// ---------------------------------------------------------------------------
// Gemm
// ---------------------------------------------------------------------------

pub fn gemm(
    a: &HostArray,
    b: &HostArray,
    c: Option<&HostArray>,
    options: &MLGemmOptions,
) -> Result<HostArray, GraphError> {
    let a_t = if options.a_transpose {
        transpose_last_two(a)?
    } else {
        a.clone()
    };
    let b_t = if options.b_transpose {
        transpose_last_two(b)?
    } else {
        b.clone()
    };
    let mut out = matmul(&a_t, &b_t)?;
    let alpha = options.alpha as f32;
    let beta = options.beta as f32;
    if alpha != 1.0 {
        for v in &mut out.data {
            *v *= alpha;
        }
    }
    if let Some(c_arr) = c {
        let c_b = broadcast_to(c_arr, &out.shape)?;
        for (o, &cv) in out.data.iter_mut().zip(c_b.data.iter()) {
            *o += beta * cv;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// ArgMin / ArgMax
// ---------------------------------------------------------------------------

pub fn arg_min(
    arr: &HostArray,
    axis: u32,
    options: &MLArgMinMaxOptions,
) -> Result<HostArray, GraphError> {
    arg_reduce(arr, axis, options, true)
}

pub fn arg_max(
    arr: &HostArray,
    axis: u32,
    options: &MLArgMinMaxOptions,
) -> Result<HostArray, GraphError> {
    arg_reduce(arr, axis, options, false)
}

fn arg_reduce(
    arr: &HostArray,
    axis: u32,
    options: &MLArgMinMaxOptions,
    pick_min: bool,
) -> Result<HostArray, GraphError> {
    let axis = axis as usize;
    if axis >= arr.rank() {
        return Err(burn_err(format!("arg reduce axis {axis} out of range")));
    }
    let input_shape_u32: Vec<u32> = arr.shape.iter().map(|&d| d as u32).collect();
    let out_shape_u32 = crate::shape_inference::infer_arg_reduce_shape(
        &input_shape_u32,
        axis as u32,
        options.keep_dimensions,
    )?;
    let out_shape: Vec<usize> = out_shape_u32.iter().map(|&d| d as usize).collect();
    let out_len = out_shape.iter().product::<usize>().max(1);
    let mut out = vec![0.0f32; out_len];
    let mut best_vals = vec![
        if pick_min {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };
        out_len
    ];
    let mut best_idx = vec![0usize; out_len];

    for flat_in in 0..arr.numel() {
        let in_coords = unravel(flat_in, &arr.shape);
        let mut out_coords = Vec::new();
        for (i, &c) in in_coords.iter().enumerate() {
            if i == axis {
                if options.keep_dimensions {
                    out_coords.push(0);
                }
            } else {
                out_coords.push(c);
            }
        }
        let out_idx = if out_coords.is_empty() {
            0
        } else {
            linear_index(&out_coords, &out_shape)
        };
        let v = arr.data[flat_in];
        let better = if pick_min {
            v < best_vals[out_idx]
        } else {
            v > best_vals[out_idx]
        };
        if better {
            best_vals[out_idx] = v;
            best_idx[out_idx] = in_coords[axis];
        }
    }
    for (o, &idx) in out.iter_mut().zip(best_idx.iter()) {
        *o = idx as f32;
    }
    Ok(HostArray {
        shape: out_shape,
        data: out,
        i64_data: None,
        u64_data: None,
    })
}

// ---------------------------------------------------------------------------
// Pad
// ---------------------------------------------------------------------------

pub fn pad(
    arr: &HostArray,
    beginning_padding: &[u32],
    ending_padding: &[u32],
    options: &MLPadOptions,
) -> Result<HostArray, GraphError> {
    let rank = arr.rank();
    let mut padding = beginning_padding.to_vec();
    padding.extend_from_slice(ending_padding);
    if padding.len() != 2 * rank {
        return Err(burn_err(format!(
            "pad padding length {} must be 2 * rank {rank}",
            padding.len()
        )));
    }
    let input_shape_u32: Vec<u32> = arr.shape.iter().map(|&d| d as u32).collect();
    let out_shape_u32 = infer_pad_shape(&input_shape_u32, &padding)?;
    let out_shape: Vec<usize> = out_shape_u32.iter().map(|&d| d as usize).collect();
    let pad_value = parse_pad_value(options);
    let mode = options.mode.to_ascii_lowercase();
    let out_len = out_shape.iter().product::<usize>().max(1);
    let mut out = vec![pad_value; out_len];

    for out_flat in 0..out_len {
        let out_coords = unravel(out_flat, &out_shape);
        let mut in_coords = Vec::with_capacity(rank);
        let mut valid = true;
        for i in 0..rank {
            let out_c = out_coords[i] as i64;
            let in_c = out_c - beginning_padding[i] as i64;
            match mode.as_str() {
                "edge" => {
                    in_coords.push(in_c.clamp(0, arr.shape[i] as i64 - 1) as usize);
                }
                "reflection" => {
                    if let Some(c) = reflect_index(in_c, arr.shape[i]) {
                        in_coords.push(c);
                    } else {
                        valid = false;
                        break;
                    }
                }
                _ => {
                    if in_c < 0 || in_c >= arr.shape[i] as i64 {
                        valid = false;
                        break;
                    }
                    in_coords.push(in_c as usize);
                }
            }
        }
        if valid {
            out[out_flat] = arr.data[linear_index(&in_coords, &arr.shape)];
        }
    }
    Ok(HostArray {
        shape: out_shape,
        data: out,
        i64_data: None,
        u64_data: None,
    })
}

// ---------------------------------------------------------------------------
// Triangular
// ---------------------------------------------------------------------------

pub fn triangular(arr: &HostArray, options: &MLTriangularOptions) -> Result<HostArray, GraphError> {
    if arr.rank() < 2 {
        return Err(burn_err(format!(
            "triangular requires rank >= 2, got {}",
            arr.rank()
        )));
    }
    let upper = options.upper.unwrap_or(true);
    let diagonal = options.diagonal;
    let rows = arr.shape[arr.rank() - 2];
    let cols = arr.shape[arr.rank() - 1];
    let mut out = arr.clone();
    let batch = arr.numel() / (rows * cols).max(1);
    for b in 0..batch {
        for r in 0..rows {
            for c in 0..cols {
                let keep = if upper {
                    (c as i32) >= (r as i32) + diagonal
                } else {
                    (c as i32) <= (r as i32) + diagonal
                };
                if !keep {
                    let mut coords = unravel(b * rows * cols + r * cols + c, &arr.shape);
                    let idx = linear_index(&coords, &arr.shape);
                    out.data[idx] = 0.0;
                    let _ = &mut coords;
                }
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Gather / scatter
// ---------------------------------------------------------------------------

pub fn gather_elements(
    data: &HostArray,
    indices: &HostArray,
    axis: u32,
) -> Result<HostArray, GraphError> {
    let axis = axis as usize;
    if axis >= data.rank() {
        return Err(burn_err(format!("gatherElements axis {axis} out of range")));
    }
    let out_shape = indices.shape.clone();
    let out_len = out_shape.iter().product::<usize>().max(1);
    let mut out = vec![0.0f32; out_len];
    for out_flat in 0..out_len {
        let mut coords = unravel(out_flat, &out_shape);
        let idx = indices.data[out_flat] as i64;
        coords[axis] = normalize_index(idx, data.shape[axis]);
        if coords.iter().zip(data.shape.iter()).all(|(&c, &s)| c < s) {
            out[out_flat] = data.data[linear_index(&coords, &data.shape)];
        }
    }
    Ok(HostArray {
        shape: out_shape,
        data: out,
        i64_data: None,
        u64_data: None,
    })
}

pub fn gather_nd(data: &HostArray, indices: &HostArray) -> Result<HostArray, GraphError> {
    if indices.rank() < 1 {
        return Err(burn_err("gatherND indices must have rank >= 1".to_string()));
    }
    let k = indices.shape[indices.rank() - 1] as usize;
    if k > data.rank() {
        return Err(burn_err(format!(
            "gatherND index depth {k} exceeds data rank {}",
            data.rank()
        )));
    }
    let index_prefix_shape = &indices.shape[..indices.rank() - 1];
    let mut out_shape = index_prefix_shape.to_vec();
    out_shape.extend_from_slice(&data.shape[k..]);
    let out_len = out_shape.iter().product::<usize>().max(1);
    let mut out = vec![0.0f32; out_len];
    let tail_shape = &data.shape[k..];
    let tail_len = tail_shape.iter().product::<usize>().max(1);

    for out_flat in 0..out_len {
        let tail_flat = out_flat % tail_len;
        let index_flat = out_flat / tail_len;
        let prefix = unravel(index_flat, index_prefix_shape);
        let tail_coords = unravel(tail_flat, tail_shape);
        let mut data_coords = vec![0usize; data.rank()];
        for j in 0..k {
            let idx = read_nd_index(indices, &prefix, j);
            let dim = data.shape[j];
            data_coords[j] = normalize_index(idx, dim);
        }
        for (j, &c) in tail_coords.iter().enumerate() {
            data_coords[k + j] = c;
        }
        if data_coords
            .iter()
            .zip(data.shape.iter())
            .all(|(&c, &s)| c < s)
        {
            out[out_flat] = data.data[linear_index(&data_coords, &data.shape)];
        }
    }
    Ok(HostArray {
        shape: out_shape,
        data: out,
        i64_data: None,
        u64_data: None,
    })
}

pub fn scatter_elements(
    data: &HostArray,
    indices: &HostArray,
    updates: &HostArray,
    options: &MLScatterOptions,
) -> Result<HostArray, GraphError> {
    let axis = options.axis as usize;
    if axis >= data.rank() {
        return Err(burn_err(format!(
            "scatterElements axis {axis} out of range"
        )));
    }
    if indices.shape != updates.shape {
        return Err(burn_err(format!(
            "scatterElements indices shape {:?} != updates shape {:?}",
            indices.shape, updates.shape
        )));
    }
    let mut out = data.clone();
    for flat in 0..indices.numel() {
        let mut coords = unravel(flat, &indices.shape);
        let idx = indices.data[flat];
        let axis_dim = data.shape[axis];
        let idx_i = idx as i64;
        coords[axis] = if idx_i < 0 {
            (axis_dim as i64 + idx_i).max(0) as usize
        } else {
            idx_i as usize
        };
        if coords[axis] < axis_dim {
            let out_idx = linear_index(&coords, &data.shape);
            out.data[out_idx] = updates.data[flat];
        }
    }
    Ok(out)
}

pub fn scatter_nd(
    data: &HostArray,
    indices: &HostArray,
    updates: &HostArray,
) -> Result<HostArray, GraphError> {
    if indices.rank() < 1 {
        return Err(burn_err(
            "scatterND indices must have rank >= 1".to_string(),
        ));
    }
    let k = indices.shape[indices.rank() - 1] as usize;
    if k > data.rank() {
        return Err(burn_err(format!(
            "scatterND index depth {k} exceeds data rank {}",
            data.rank()
        )));
    }
    let mut out = data.clone();
    let index_prefix_shape = &indices.shape[..indices.rank() - 1];
    let index_prefix_len: usize = index_prefix_shape.iter().product::<usize>().max(1);
    let update_tail_shape = &data.shape[k..];
    let update_tail_len: usize = update_tail_shape.iter().product::<usize>().max(1);

    for index_flat in 0..index_prefix_len {
        let prefix = unravel(index_flat, index_prefix_shape);
        let mut base_coords = vec![0usize; data.rank()];
        for j in 0..k {
            let idx = read_nd_index(indices, &prefix, j);
            base_coords[j] = normalize_index(idx, data.shape[j]);
        }
        for update_flat in 0..update_tail_len {
            let tail_coords = unravel(update_flat, update_tail_shape);
            let mut data_coords = base_coords.clone();
            for (i, &c) in tail_coords.iter().enumerate() {
                data_coords[k + i] = c;
            }
            let data_idx = linear_index(&data_coords, &data.shape);
            out.data[data_idx] = updates.data[index_flat * update_tail_len + update_flat];
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Pooling
// ---------------------------------------------------------------------------

pub fn l2_pool2d(input: &HostArray, options: &MLPool2dOptions) -> Result<HostArray, GraphError> {
    pool2d(input, options, PoolKind::L2)
}

pub fn global_average_pool(
    input: &HostArray,
    options: &MLPool2dOptions,
) -> Result<HostArray, GraphError> {
    global_pool(input, options, PoolKind::Average)
}

pub fn global_max_pool(
    input: &HostArray,
    options: &MLPool2dOptions,
) -> Result<HostArray, GraphError> {
    global_pool(input, options, PoolKind::Max)
}

fn global_pool(
    input: &HostArray,
    options: &MLPool2dOptions,
    kind: PoolKind,
) -> Result<HostArray, GraphError> {
    if input.rank() != 4 {
        return Err(burn_err(format!(
            "global pool expects 4D input, got rank {}",
            input.rank()
        )));
    }
    let nchw = !options.layout.eq_ignore_ascii_case("nhwc");
    let (h, w) = if nchw {
        (input.shape[2], input.shape[3])
    } else {
        (input.shape[1], input.shape[2])
    };
    let mut opts = options.clone();
    opts.window_dimensions = Some(vec![h as u32, w as u32]);
    if opts.strides.is_empty() {
        opts.strides = vec![h as u32, w as u32];
    }
    pool2d(input, &opts, kind)
}

// ---------------------------------------------------------------------------
// Quantize / dequantize
// ---------------------------------------------------------------------------

pub fn quantize_linear(
    input: &HostArray,
    scale: &HostArray,
    zero_point: Option<&HostArray>,
    output_dtype: DataType,
) -> Result<HostArray, GraphError> {
    quantize_impl(input, scale, zero_point, output_dtype, true)
}

pub fn dequantize_linear(
    input: &HostArray,
    scale: &HostArray,
    zero_point: Option<&HostArray>,
) -> Result<HostArray, GraphError> {
    quantize_impl(input, scale, zero_point, DataType::Float32, false)
}

fn quantize_clip_bounds(dtype: DataType) -> Option<(f32, f32)> {
    match dtype {
        DataType::Int8 => Some((-128.0, 127.0)),
        DataType::Uint8 => Some((0.0, 255.0)),
        DataType::Int4 => Some((-8.0, 7.0)),
        DataType::Uint4 => Some((0.0, 15.0)),
        DataType::Int32 | DataType::Uint32 => None,
        _ => None,
    }
}

fn quantize_impl(
    input: &HostArray,
    scale: &HostArray,
    zero_point: Option<&HostArray>,
    output_dtype: DataType,
    quantize: bool,
) -> Result<HostArray, GraphError> {
    let out_shape = input.shape.clone();
    let inp = input.clone();
    let sc = expand_quantize_param(&out_shape, scale)?;
    let zp = if let Some(z) = zero_point {
        expand_quantize_param(&out_shape, z)?
    } else {
        HostArray {
            shape: out_shape.clone(),
            data: vec![0.0; out_shape.iter().product::<usize>().max(1)],
            i64_data: None,
            u64_data: None,
        }
    };
    let clip = if quantize {
        quantize_clip_bounds(output_dtype)
    } else {
        None
    };
    let data: Vec<f32> = inp
        .data
        .iter()
        .zip(sc.data.iter().zip(zp.data.iter()))
        .map(|(&x, (&s, &z))| {
            if quantize {
                if s == 0.0 {
                    0.0
                } else {
                    let shifted = round_to_nearest_even(x / s) + z;
                    if let Some((min_v, max_v)) = clip {
                        shifted.clamp(min_v, max_v)
                    } else {
                        shifted
                    }
                }
            } else {
                let q = round_to_nearest_even(x);
                (q - z) * s
            }
        })
        .collect();
    Ok(HostArray {
        shape: out_shape,
        data,
        i64_data: None,
        u64_data: None,
    })
}

// ---------------------------------------------------------------------------
// Resample2d
// ---------------------------------------------------------------------------

pub fn resample2d(
    input: &HostArray,
    options: &MLResample2dOptions,
) -> Result<HostArray, GraphError> {
    if input.rank() != 4 {
        return Err(burn_err(format!(
            "resample2d expects 4D NCHW input, got rank {}",
            input.rank()
        )));
    }
    let axes: [usize; 2] = if options.axes.len() == 2 {
        [options.axes[0] as usize, options.axes[1] as usize]
    } else {
        [2, 3]
    };
    if axes[0] >= 4 || axes[1] >= 4 || axes[0] == axes[1] {
        return Err(burn_err(format!(
            "resample2d invalid axes {:?}",
            options.axes
        )));
    }
    let mut out_shape = input.shape.clone();
    let out_sizes: [usize; 2] = if let Some(ref sizes) = options.sizes {
        if sizes.len() != 2 {
            return Err(burn_err(format!(
                "resample2d sizes must have length 2, got {:?}",
                sizes
            )));
        }
        [sizes[0] as usize, sizes[1] as usize]
    } else {
        let (s0, s1) = if options.scales.is_empty() {
            (1.0_f32, 1.0_f32)
        } else if options.scales.len() == 2 {
            (options.scales[0], options.scales[1])
        } else {
            return Err(burn_err(format!(
                "resample2d scales must have length 2, got {:?}",
                options.scales
            )));
        };
        [
            ((input.shape[axes[0]].max(1) as f32) * s0).round().max(1.0) as usize,
            ((input.shape[axes[1]].max(1) as f32) * s1).round().max(1.0) as usize,
        ]
    };
    out_shape[axes[0]] = out_sizes[0];
    out_shape[axes[1]] = out_sizes[1];

    let mode = options.mode.to_ascii_lowercase();
    let linear = mode == "linear";
    let out_len = out_shape.iter().product::<usize>().max(1);
    let mut out = vec![0.0f32; out_len];

    for out_flat in 0..out_len {
        let out_coords = unravel(out_flat, &out_shape);
        if linear {
            let sy = map_resample_coord(
                out_coords[axes[0]],
                input.shape[axes[0]],
                out_shape[axes[0]],
                true,
            );
            let sx = map_resample_coord(
                out_coords[axes[1]],
                input.shape[axes[1]],
                out_shape[axes[1]],
                true,
            );
            let y0 = sy.floor() as usize;
            let x0 = sx.floor() as usize;
            let y1 = sy.ceil() as usize;
            let x1 = sx.ceil() as usize;
            let wy = sy - y0 as f32;
            let wx = sx - x0 as f32;
            let read = |y: usize, x: usize| -> f32 {
                let mut c = out_coords.clone();
                c[axes[0]] = y.min(input.shape[axes[0]].saturating_sub(1));
                c[axes[1]] = x.min(input.shape[axes[1]].saturating_sub(1));
                input.data[linear_index(&c, &input.shape)]
            };
            let v00 = read(y0, x0);
            let v01 = read(y0, x1);
            let v10 = read(y1, x0);
            let v11 = read(y1, x1);
            out[out_flat] = v00 * (1.0 - wx) * (1.0 - wy)
                + v01 * wx * (1.0 - wy)
                + v10 * (1.0 - wx) * wy
                + v11 * wx * wy;
        } else {
            let mut in_coords = out_coords.clone();
            in_coords[axes[0]] = resample_nearest_index(
                out_coords[axes[0]],
                input.shape[axes[0]],
                out_shape[axes[0]],
            );
            in_coords[axes[1]] = resample_nearest_index(
                out_coords[axes[1]],
                input.shape[axes[1]],
                out_shape[axes[1]],
            );
            out[out_flat] = input.data[linear_index(&in_coords, &input.shape)];
        }
    }

    Ok(HostArray {
        shape: out_shape,
        data: out,
        i64_data: None,
        u64_data: None,
    })
}

// ---------------------------------------------------------------------------
// Reverse
// ---------------------------------------------------------------------------

pub fn reverse(arr: &HostArray, options: &MLReverseOptions) -> Result<HostArray, GraphError> {
    let axes: Vec<usize> = match &options.axes {
        None => (0..arr.rank()).collect(),
        Some(ax) if ax.is_empty() => Vec::new(),
        Some(ax) => ax.iter().map(|&a| a as usize).collect(),
    };
    let mut out = arr.clone();
    for flat in 0..arr.numel() {
        let coords = unravel(flat, &arr.shape);
        let mut out_coords = coords.clone();
        for &ax in &axes {
            if ax < arr.rank() {
                out_coords[ax] = arr.shape[ax] - 1 - coords[ax];
            }
        }
        out.data[linear_index(&out_coords, &arr.shape)] = arr.data[flat];
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// RNN: GRU / LSTM
// ---------------------------------------------------------------------------

pub fn gru_cell(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    hidden_state: &HostArray,
    hidden_size: u32,
    options: &MLGruCellOptions,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
) -> Result<HostArray, GraphError> {
    gru_cell_internal(
        input,
        weight,
        recurrence,
        hidden_state,
        hidden_size as usize,
        options,
        bias,
        recurrent_bias,
        0,
    )
}

pub fn gru(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    steps: u32,
    hidden_size: u32,
    options: &MLGruOptions,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    initial_hidden: Option<&HostArray>,
) -> Result<(Option<HostArray>, HostArray), GraphError> {
    if options.direction.eq_ignore_ascii_case("both") {
        return gru_bidirectional(
            input,
            weight,
            recurrence,
            steps,
            hidden_size,
            options,
            bias,
            recurrent_bias,
            initial_hidden,
        );
    }
    gru_unidirectional(
        input,
        weight,
        recurrence,
        steps,
        hidden_size,
        options,
        bias,
        recurrent_bias,
        initial_hidden,
    )
}

fn gru_unidirectional(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    steps: u32,
    hidden_size: u32,
    options: &MLGruOptions,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    initial_hidden: Option<&HostArray>,
) -> Result<(Option<HostArray>, HostArray), GraphError> {
    let hs = hidden_size as usize;
    let cell_opts = MLGruCellOptions {
        label: options.label.clone(),
        bias: options.bias,
        recurrent_bias: options.recurrent_bias,
        reset_after: options.reset_after,
        layout: options.layout.clone(),
        activations: options.activations.clone(),
    };
    let (step_count, batch, input_size) = gru_sequence_dims(input, steps)?;
    let mut h = initial_hidden
        .map(|t| normalize_hidden(t, batch, hs))
        .transpose()?
        .unwrap_or_else(|| HostArray::new(vec![batch, hs], vec![0.0; batch * hs]).unwrap());
    let mut sequence = if options.return_sequence {
        Some(HostArray::new(
            vec![step_count, batch, hs],
            vec![0.0; step_count * batch * hs],
        )?)
    } else {
        None
    };
    let direction = options.direction.to_ascii_lowercase();
    let dir_idx = match direction.as_str() {
        "backward" => 1,
        _ => 0,
    };
    let step_range: Vec<usize> = if direction == "backward" {
        (0..step_count).rev().collect()
    } else {
        (0..step_count).collect()
    };
    for t in step_range.iter() {
        let x_t = slice_sequence_step(input, *t, step_count, batch, input_size)?;
        h = gru_cell_internal(
            &x_t,
            weight,
            recurrence,
            &h,
            hs,
            &cell_opts,
            bias,
            recurrent_bias,
            dir_idx,
        )?;
        if let Some(seq) = sequence.as_mut() {
            copy_hidden_to_sequence(seq, *t, batch, hs, &h);
        }
    }
    Ok((sequence, h))
}

fn gru_bidirectional(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    steps: u32,
    hidden_size: u32,
    options: &MLGruOptions,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    initial_hidden: Option<&HostArray>,
) -> Result<(Option<HostArray>, HostArray), GraphError> {
    let _hs = hidden_size as usize;
    let (_step_count, _batch, _input_size) = gru_sequence_dims(input, steps)?;
    let mut opts_fwd = options.clone();
    opts_fwd.direction = "forward".to_string();
    let (seq_fwd, h_fwd) = gru_unidirectional(
        input,
        weight,
        recurrence,
        steps,
        hidden_size,
        &opts_fwd,
        bias,
        recurrent_bias,
        initial_hidden,
    )?;
    let mut opts_bwd = options.clone();
    opts_bwd.direction = "backward".to_string();
    let (seq_bwd, h_bwd) = gru_unidirectional(
        input,
        weight,
        recurrence,
        steps,
        hidden_size,
        &opts_bwd,
        bias,
        recurrent_bias,
        initial_hidden,
    )?;

    let final_h = stack_rnn_states(&h_fwd, &h_bwd)?;
    let sequence = if options.return_sequence {
        let fwd = seq_fwd.ok_or_else(|| burn_err("forward GRU missing sequence".to_string()))?;
        let bwd = seq_bwd.ok_or_else(|| burn_err("backward GRU missing sequence".to_string()))?;
        Some(stack_rnn_sequences(&fwd, &bwd)?)
    } else {
        None
    };
    Ok((sequence, final_h))
}

pub fn lstm_cell(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    hidden_state: &HostArray,
    cell_state: &HostArray,
    options: &MLLstmCellOptions,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    peephole: Option<&HostArray>,
) -> Result<(HostArray, HostArray), GraphError> {
    lstm_cell_internal(
        input,
        weight,
        recurrence,
        hidden_state,
        cell_state,
        options,
        bias,
        recurrent_bias,
        peephole,
        0,
    )
}

fn lstm_cell_internal(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    hidden_state: &HostArray,
    cell_state: &HostArray,
    options: &MLLstmCellOptions,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    peephole: Option<&HostArray>,
    direction: usize,
) -> Result<(HostArray, HostArray), GraphError> {
    let hs = infer_lstm_hidden_size_from_weight(weight)?;
    let w = resolve_rnn_matrix(weight, direction)?;
    let r = resolve_rnn_matrix(recurrence, direction)?;
    let (batch, input_size) = matrix_batch_dims(input)?;
    let layout = options.layout.to_ascii_lowercase();
    let activations = options.activations.as_deref();
    let mut h = normalize_hidden(hidden_state, batch, hs)?;
    let mut c = normalize_hidden(cell_state, batch, hs)?;
    let w_bias = if options.bias.is_some() {
        bias.map(|b| resolve_rnn_bias(b, direction, 4 * hs))
            .transpose()?
    } else {
        None
    };
    let r_bias = if options.recurrent_bias.is_some() {
        recurrent_bias
            .map(|b| resolve_rnn_bias(b, direction, 4 * hs))
            .transpose()?
    } else {
        None
    };
    for b in 0..batch {
        let x_row = slice_row(input, b, input_size);
        let h_row = slice_row(&h, b, hs);
        let c_row = slice_row_data(&c, b, hs);
        let w_g = linear_2d_with_bias_vec(&x_row, &w, w_bias.as_deref(), 4 * hs, input_size)?;
        let r_g = linear_2d_with_bias_vec(&h_row, &r, r_bias.as_deref(), 4 * hs, hs)?;
        let mut gates: Vec<f32> = w_g.iter().zip(r_g.iter()).map(|(&a, &b)| a + b).collect();
        if let Some(p) = peephole {
            apply_peephole(&mut gates, p, &c_row, hs, &layout)?;
        }
        let (i, o, f, g, act_c) = split_lstm_gates(&gates, hs, &layout, activations)?;
        let c_new: Vec<f32> = f
            .iter()
            .zip(i.iter().zip(g.iter().zip(c_row.iter())))
            .map(|(&fi, (&ii, (&gi, &ci)))| fi * ci + ii * gi)
            .collect();
        let h_new: Vec<f32> = o
            .iter()
            .zip(c_new.iter())
            .map(|(&oi, &ci)| oi * apply_activation(&act_c, ci))
            .collect();
        write_row(&mut c, b, hs, &c_new);
        write_row(&mut h, b, hs, &h_new);
    }
    Ok((h, c))
}

pub fn lstm(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    options: &MLLstmOptions,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    peephole: Option<&HostArray>,
    initial_hidden: Option<&HostArray>,
    initial_cell: Option<&HostArray>,
) -> Result<(Option<HostArray>, HostArray, Option<HostArray>, HostArray), GraphError> {
    if options.direction.eq_ignore_ascii_case("both") {
        return lstm_bidirectional(
            input,
            weight,
            recurrence,
            options,
            bias,
            recurrent_bias,
            peephole,
            initial_hidden,
            initial_cell,
        );
    }
    lstm_unidirectional(
        input,
        weight,
        recurrence,
        options,
        bias,
        recurrent_bias,
        peephole,
        initial_hidden,
        initial_cell,
    )
}

fn lstm_unidirectional(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    options: &MLLstmOptions,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    peephole: Option<&HostArray>,
    initial_hidden: Option<&HostArray>,
    initial_cell: Option<&HostArray>,
) -> Result<(Option<HostArray>, HostArray, Option<HostArray>, HostArray), GraphError> {
    let hs = infer_lstm_hidden_size_from_weight(weight)?;
    let cell_opts = MLLstmCellOptions {
        label: options.label.clone(),
        bias: options.bias,
        recurrent_bias: options.recurrent_bias,
        peephole_weight: options.peephole_weight,
        layout: options.layout.clone(),
        activations: options.activations.clone(),
    };
    let (step_count, batch, input_size) = lstm_sequence_dims(input)?;
    let direction = options.direction.to_ascii_lowercase();
    let dir_idx = if direction == "backward" { 1 } else { 0 };
    let mut h = resolve_direction_initial(initial_hidden, dir_idx, batch, hs)?
        .unwrap_or_else(|| HostArray::new(vec![batch, hs], vec![0.0; batch * hs]).unwrap());
    let mut c = resolve_direction_initial(initial_cell, dir_idx, batch, hs)?
        .unwrap_or_else(|| HostArray::new(vec![batch, hs], vec![0.0; batch * hs]).unwrap());
    let mut h_seq = if options.return_sequence {
        Some(HostArray::new(
            vec![step_count, batch, hs],
            vec![0.0; step_count * batch * hs],
        )?)
    } else {
        None
    };
    let mut c_seq = if options.return_sequence {
        Some(HostArray::new(
            vec![step_count, batch, hs],
            vec![0.0; step_count * batch * hs],
        )?)
    } else {
        None
    };
    let peephole_dir = peephole
        .map(|p| resolve_rnn_peephole(p, dir_idx, hs))
        .transpose()?;
    let step_range: Vec<usize> = if direction == "backward" {
        (0..step_count).rev().collect()
    } else {
        (0..step_count).collect()
    };
    for t in step_range.iter() {
        let x_t = slice_sequence_step(input, *t, step_count, batch, input_size)?;
        let (h_new, c_new) = lstm_cell_internal(
            &x_t,
            weight,
            recurrence,
            &h,
            &c,
            &cell_opts,
            bias,
            recurrent_bias,
            peephole_dir.as_ref(),
            dir_idx,
        )?;
        h = h_new;
        c = c_new;
        if let Some(seq) = h_seq.as_mut() {
            copy_hidden_to_sequence(seq, *t, batch, hs, &h);
        }
        if let Some(seq) = c_seq.as_mut() {
            copy_hidden_to_sequence(seq, *t, batch, hs, &c);
        }
    }
    Ok((h_seq, h, c_seq, c))
}

fn lstm_bidirectional(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    options: &MLLstmOptions,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    peephole: Option<&HostArray>,
    initial_hidden: Option<&HostArray>,
    initial_cell: Option<&HostArray>,
) -> Result<(Option<HostArray>, HostArray, Option<HostArray>, HostArray), GraphError> {
    let (_step_count, batch, _input_size) = lstm_sequence_dims(input)?;
    let hs = infer_lstm_hidden_size_from_weight(weight)?;
    let h0_fwd = resolve_direction_initial(initial_hidden, 0, batch, hs)?;
    let c0_fwd = resolve_direction_initial(initial_cell, 0, batch, hs)?;

    let mut opts_fwd = options.clone();
    opts_fwd.direction = "forward".to_string();
    let (h_seq_fwd, h_fwd, _c_seq_fwd, c_fwd) = lstm_unidirectional(
        input,
        weight,
        recurrence,
        &opts_fwd,
        bias,
        recurrent_bias,
        peephole,
        h0_fwd.as_ref(),
        c0_fwd.as_ref(),
    )?;

    let h0_bwd = resolve_direction_initial(initial_hidden, 1, batch, hs)?;
    let c0_bwd = resolve_direction_initial(initial_cell, 1, batch, hs)?;
    let mut opts_bwd = options.clone();
    opts_bwd.direction = "backward".to_string();
    let (h_seq_bwd, h_bwd, _c_seq_bwd, c_bwd) = lstm_unidirectional(
        input,
        weight,
        recurrence,
        &opts_bwd,
        bias,
        recurrent_bias,
        peephole,
        h0_bwd.as_ref(),
        c0_bwd.as_ref(),
    )?;

    let final_h = stack_rnn_states(&h_fwd, &h_bwd)?;
    let final_c = stack_rnn_states(&c_fwd, &c_bwd)?;
    let h_seq = if options.return_sequence {
        let fwd = h_seq_fwd.ok_or_else(|| burn_err("forward LSTM missing sequence".to_string()))?;
        let bwd =
            h_seq_bwd.ok_or_else(|| burn_err("backward LSTM missing sequence".to_string()))?;
        Some(stack_rnn_sequences(&fwd, &bwd)?)
    } else {
        None
    };
    Ok((h_seq, final_h, None, final_c))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn read_nd_index(indices: &HostArray, prefix: &[usize], component: usize) -> i64 {
    let mut coord = vec![0usize; indices.rank()];
    for (i, &c) in prefix.iter().enumerate() {
        coord[i] = c;
    }
    coord[indices.rank() - 1] = component;
    indices.data[linear_index(&coord, &indices.shape)] as i64
}

fn normalize_index(idx: i64, dim: usize) -> usize {
    if dim == 0 {
        return 0;
    }
    let mut i = if idx < 0 { dim as i64 + idx } else { idx };
    if i < 0 {
        i = 0;
    } else if i >= dim as i64 {
        i = dim as i64 - 1;
    }
    i as usize
}

fn expand_quantize_param(
    input_shape: &[usize],
    param: &HostArray,
) -> Result<HostArray, GraphError> {
    if param.shape.as_slice() == input_shape {
        return Ok(param.clone());
    }
    if param.shape.is_empty() || param.numel() == 1 {
        return broadcast_to(param, input_shape);
    }
    if param.shape.len() != input_shape.len() {
        return broadcast_to(param, input_shape);
    }
    if param
        .shape
        .iter()
        .zip(input_shape.iter())
        .all(|(&p, &i)| p == i || p == 1)
    {
        return broadcast_to(param, input_shape);
    }
    let block_size: Vec<usize> = input_shape
        .iter()
        .zip(param.shape.iter())
        .map(|(&i, &p)| if p == 0 { 1 } else { i / p })
        .collect();
    let out_len = input_shape.iter().product::<usize>().max(1);
    let mut out = vec![0.0f32; out_len];
    for out_flat in 0..out_len {
        let out_coords = unravel(out_flat, input_shape);
        let mut param_coords = vec![0usize; param.shape.len()];
        for d in 0..param.shape.len() {
            param_coords[d] = out_coords[d] / block_size[d].max(1);
        }
        out[out_flat] = param.data[linear_index(&param_coords, &param.shape)];
    }
    Ok(HostArray {
        shape: input_shape.to_vec(),
        data: out,
        i64_data: None,
        u64_data: None,
    })
}

fn resolve_rnn_matrix(w: &HostArray, direction: usize) -> Result<HostArray, GraphError> {
    match w.rank() {
        2 => Ok(w.clone()),
        3 => {
            let num_dir = w.shape[0];
            let dir = direction.min(num_dir.saturating_sub(1));
            let out_f = w.shape[1];
            let in_f = w.shape[2];
            let start = dir * out_f * in_f;
            HostArray::new(
                vec![out_f, in_f],
                w.data[start..start + out_f * in_f].to_vec(),
            )
        }
        r => Err(burn_err(format!(
            "RNN weight must be rank 2 or 3, got {r} with shape {:?}",
            w.shape
        ))),
    }
}

fn resolve_rnn_bias(
    b: &HostArray,
    direction: usize,
    expected_len: usize,
) -> Result<Vec<f32>, GraphError> {
    match b.rank() {
        1 if b.shape[0] == expected_len => Ok(b.data.clone()),
        1 => Err(burn_err(format!(
            "RNN bias length {} expected {}",
            b.shape[0], expected_len
        ))),
        2 => {
            let dir = direction.min(b.shape[0].saturating_sub(1));
            let row = b.shape[1];
            if row != expected_len {
                return Err(burn_err(format!(
                    "RNN bias row length {} expected {}",
                    row, expected_len
                )));
            }
            let start = dir * row;
            Ok(b.data[start..start + row].to_vec())
        }
        r => Err(burn_err(format!(
            "RNN bias must be rank 1 or 2, got {r} with shape {:?}",
            b.shape
        ))),
    }
}

fn resolve_rnn_peephole(
    p: &HostArray,
    direction: usize,
    hs: usize,
) -> Result<HostArray, GraphError> {
    let expected = 3 * hs;
    match p.rank() {
        1 if p.shape[0] == expected => Ok(p.clone()),
        2 => {
            let dir = direction.min(p.shape[0].saturating_sub(1));
            let row = p.shape[1];
            if row != expected {
                return Err(burn_err(format!(
                    "LSTM peephole row length {row} expected {expected}"
                )));
            }
            let start = dir * row;
            HostArray::new(vec![expected], p.data[start..start + row].to_vec())
        }
        r => Err(burn_err(format!(
            "LSTM peephole must be rank 1 or 2, got {r} with shape {:?}",
            p.shape
        ))),
    }
}

fn resolve_direction_initial(
    hidden: Option<&HostArray>,
    direction: usize,
    batch: usize,
    hs: usize,
) -> Result<Option<HostArray>, GraphError> {
    let Some(h) = hidden else {
        return Ok(None);
    };
    if h.rank() == 3 && h.shape[1] == batch && h.shape[2] == hs {
        let dir = direction.min(h.shape[0].saturating_sub(1));
        let start = dir * batch * hs;
        return HostArray::new(vec![batch, hs], h.data[start..start + batch * hs].to_vec())
            .map(Some);
    }
    normalize_hidden(h, batch, hs).map(Some)
}

fn infer_lstm_hidden_size_from_weight(weight: &HostArray) -> Result<usize, GraphError> {
    let gate_rows = match weight.rank() {
        2 => weight.shape[0],
        3 => weight.shape[1],
        r => {
            return Err(burn_err(format!(
                "LSTM weight must be [4*hidden, input] or [directions, 4*hidden, input], got rank {r} {:?}",
                weight.shape
            )));
        }
    };
    if gate_rows % 4 != 0 {
        return Err(burn_err(format!(
            "LSTM weight gate rows {gate_rows} must be divisible by 4, shape {:?}",
            weight.shape
        )));
    }
    Ok(gate_rows / 4)
}

fn slice_row_data(arr: &HostArray, batch: usize, width: usize) -> Vec<f32> {
    let start = batch * width;
    arr.data[start..start + width].to_vec()
}

fn erf_f32(x: f32) -> f32 {
    // Abramowitz and Stegun approximation 7.1.26
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

pub(crate) fn round_to_nearest_even(x: f32) -> f32 {
    let floor = x.floor();
    let frac = x - floor;
    if frac > 0.5 {
        floor + 1.0
    } else if frac < 0.5 {
        floor
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

fn parse_pad_value(options: &MLPadOptions) -> f32 {
    options
        .value
        .as_ref()
        .and_then(|v| {
            if v.is_null() {
                Some(0.0)
            } else if let Some(s) = v.as_str() {
                match s {
                    "NaN" => Some(f32::NAN),
                    "Infinity" => Some(f32::INFINITY),
                    "-Infinity" => Some(f32::NEG_INFINITY),
                    _ => s.parse::<f64>().ok().map(|f| f as f32),
                }
            } else {
                v.as_f64()
                    .map(|f| f as f32)
                    .or_else(|| v.as_i64().map(|i| i as f32))
            }
        })
        .unwrap_or(0.0)
}

fn reflect_index(coord: i64, size: usize) -> Option<usize> {
    if size == 0 {
        return None;
    }
    if size == 1 {
        return Some(0);
    }
    let period = 2 * (size as i64 - 1);
    let mut i = coord % period;
    if i < 0 {
        i += period;
    }
    if i >= size as i64 {
        i = period - i;
    }
    Some(i as usize)
}

fn transpose_last_two(arr: &HostArray) -> Result<HostArray, GraphError> {
    if arr.rank() < 2 {
        return Err(burn_err("transpose requires rank >= 2".to_string()));
    }
    let mut perm: Vec<usize> = (0..arr.rank()).collect();
    let r = arr.rank();
    perm.swap(r - 2, r - 1);
    transpose(arr, &perm)
}

fn matrix_batch_dims(input: &HostArray) -> Result<(usize, usize), GraphError> {
    match input.rank() {
        1 => Ok((1, input.shape[0])),
        2 => Ok((input.shape[0], input.shape[1])),
        r => Err(burn_err(format!("RNN input must be rank 1 or 2, got {r}"))),
    }
}

fn normalize_hidden(hidden: &HostArray, batch: usize, hs: usize) -> Result<HostArray, GraphError> {
    match hidden.rank() {
        1 if hidden.shape[0] == hs => {
            let mut out = vec![0.0; batch * hs];
            out[..hs].copy_from_slice(&hidden.data[..hs]);
            for b in 1..batch {
                out[b * hs..(b + 1) * hs].copy_from_slice(&hidden.data[..hs]);
            }
            HostArray::new(vec![batch, hs], out)
        }
        2 if hidden.shape[0] == batch && hidden.shape[1] == hs => Ok(hidden.clone()),
        2 if hidden.shape[0] == hs && hidden.shape[1] == batch => {
            let mut out = vec![0.0; batch * hs];
            for b in 0..batch {
                for h in 0..hs {
                    out[b * hs + h] = hidden.data[h * batch + b];
                }
            }
            HostArray::new(vec![batch, hs], out)
        }
        3 if hidden.shape[1] == batch && hidden.shape[2] == hs => {
            HostArray::new(vec![batch, hs], hidden.data[..batch * hs].to_vec())
        }
        r => Err(burn_err(format!(
            "hidden state shape {:?} incompatible with batch {batch}, hidden {hs} (rank {r})",
            hidden.shape
        ))),
    }
}

fn slice_row(arr: &HostArray, batch: usize, width: usize) -> HostArray {
    let start = batch * width;
    HostArray {
        shape: vec![1, width],
        data: arr.data[start..start + width].to_vec(),
        i64_data: None,
        u64_data: None,
    }
}

fn write_row(arr: &mut HostArray, batch: usize, width: usize, row: &[f32]) {
    let start = batch * width;
    arr.data[start..start + width].copy_from_slice(row);
}

fn linear_2d(
    x: &HostArray,
    w: &HostArray,
    b: Option<&HostArray>,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>, GraphError> {
    linear_2d_with_bias(x, w, b, out_features, in_features)
}

fn linear_2d_with_bias(
    x: &HostArray,
    w: &HostArray,
    b: Option<&HostArray>,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>, GraphError> {
    if w.shape.len() != 2
        || w.shape[0] != out_features
        || w.shape[1] != in_features
        || x.shape[x.shape.len() - 1] != in_features
    {
        return Err(burn_err(format!(
            "linear mismatch: x {:?}, w {:?}, expected out {out_features}, in {in_features}",
            x.shape, w.shape
        )));
    }
    let mut out = vec![0.0f32; out_features];
    let x_data = &x.data[x.data.len() - in_features..];
    for o in 0..out_features {
        let mut sum = 0.0f32;
        for i in 0..in_features {
            sum += x_data[i] * w.data[o * in_features + i];
        }
        if let Some(bias) = b {
            sum += bias.data.get(o).copied().unwrap_or(0.0);
        }
        out[o] = sum;
    }
    Ok(out)
}

fn linear_2d_with_bias_vec(
    x: &HostArray,
    w: &HostArray,
    b: Option<&[f32]>,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>, GraphError> {
    if w.shape.len() != 2
        || w.shape[0] != out_features
        || w.shape[1] != in_features
        || x.shape[x.shape.len() - 1] != in_features
    {
        return Err(burn_err(format!(
            "linear mismatch: x {:?}, w {:?}, expected out {out_features}, in {in_features}",
            x.shape, w.shape
        )));
    }
    let mut out = vec![0.0f32; out_features];
    let x_data = &x.data[x.data.len() - in_features..];
    for o in 0..out_features {
        let mut sum = 0.0f32;
        for i in 0..in_features {
            sum += x_data[i] * w.data[o * in_features + i];
        }
        if let Some(bias) = b {
            sum += bias.get(o).copied().unwrap_or(0.0);
        }
        out[o] = sum;
    }
    Ok(out)
}

fn apply_activation(name: &str, x: f32) -> f32 {
    match name.to_ascii_lowercase().as_str() {
        "relu" => x.max(0.0),
        "sigmoid" => 1.0 / (1.0 + (-x).exp()),
        "tanh" => x.tanh(),
        _ => 1.0 / (1.0 + (-x).exp()),
    }
}

fn split_gru_gates(
    w: &[f32],
    r: &[f32],
    h_prev: &[f32],
    recurrence: &HostArray,
    hs: usize,
    layout: &str,
    reset_after: bool,
    activations: Option<&[String]>,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), GraphError> {
    let act_zr = activations
        .and_then(|a| a.first())
        .map(|s| s.as_str())
        .unwrap_or("sigmoid");
    let act_n = activations
        .and_then(|a| a.get(1))
        .map(|s| s.as_str())
        .unwrap_or("tanh");
    let (z_idx, r_idx, n_idx) = if layout == "rzn" {
        (1, 0, 2)
    } else {
        (0, 1, 2)
    };
    let mut z = vec![0.0; hs];
    let mut r_gate = vec![0.0; hs];
    let mut n = vec![0.0; hs];
    for i in 0..hs {
        z[i] = apply_activation(act_zr, w[z_idx * hs + i] + r[z_idx * hs + i]);
        r_gate[i] = apply_activation(act_zr, w[r_idx * hs + i] + r[r_idx * hs + i]);
    }
    for i in 0..hs {
        let pre = if reset_after {
            w[n_idx * hs + i] + r_gate[i] * r[n_idx * hs + i]
        } else {
            let mut rh = 0.0f32;
            for j in 0..hs {
                rh += recurrence.data[(n_idx * hs + i) * hs + j] * h_prev[j];
            }
            let mut r_dot_h = 0.0f32;
            for j in 0..hs {
                r_dot_h += recurrence.data[(n_idx * hs + i) * hs + j] * r_gate[j] * h_prev[j];
            }
            w[n_idx * hs + i] + (r[n_idx * hs + i] - rh) + r_dot_h
        };
        n[i] = apply_activation(act_n, pre);
    }
    Ok((z, r_gate, n))
}

fn split_lstm_gates(
    gates: &[f32],
    hs: usize,
    layout: &str,
    activations: Option<&[String]>,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, String), GraphError> {
    let (i_idx, o_idx, f_idx, g_idx) = if layout == "ifgo" {
        (0, 3, 1, 2)
    } else {
        (0, 1, 2, 3)
    };
    let act_ifo = activations
        .and_then(|a| a.first())
        .map(|s| s.as_str())
        .unwrap_or("sigmoid");
    let act_g = activations
        .and_then(|a| a.get(1))
        .map(|s| s.as_str())
        .unwrap_or("tanh");
    let act_c = activations
        .and_then(|a| a.get(2))
        .map(|s| s.as_str())
        .unwrap_or("tanh");
    let mut i = vec![0.0; hs];
    let mut o = vec![0.0; hs];
    let mut f = vec![0.0; hs];
    let mut g = vec![0.0; hs];
    for j in 0..hs {
        i[j] = apply_activation(act_ifo, gates[i_idx * hs + j]);
        o[j] = apply_activation(act_ifo, gates[o_idx * hs + j]);
        f[j] = apply_activation(act_ifo, gates[f_idx * hs + j]);
        g[j] = apply_activation(act_g, gates[g_idx * hs + j]);
    }
    Ok((i, o, f, g, act_c.to_string()))
}

fn apply_peephole(
    gates: &mut [f32],
    peephole: &HostArray,
    c: &[f32],
    hs: usize,
    layout: &str,
) -> Result<(), GraphError> {
    if peephole.numel() < 3 * hs {
        return Err(burn_err(format!(
            "peephole weight expected at least {} elements, got {}",
            3 * hs,
            peephole.numel()
        )));
    }
    let (i_idx, o_idx, f_idx) = if layout == "ifgo" {
        (0, 3, 1)
    } else {
        (0, 1, 2)
    };
    for j in 0..hs {
        gates[i_idx * hs + j] += peephole.data[0 * hs + j] * c[j];
        gates[o_idx * hs + j] += peephole.data[1 * hs + j] * c[j];
        gates[f_idx * hs + j] += peephole.data[2 * hs + j] * c[j];
    }
    Ok(())
}

fn gru_sequence_dims(input: &HostArray, steps: u32) -> Result<(usize, usize, usize), GraphError> {
    match input.rank() {
        2 => {
            let s = if steps == 0 { 1 } else { steps as usize };
            Ok((s, input.shape[0], input.shape[1]))
        }
        3 => Ok((input.shape[0], input.shape[1], input.shape[2])),
        r => Err(burn_err(format!("gru input must be rank 2 or 3, got {r}"))),
    }
}

fn lstm_sequence_dims(input: &HostArray) -> Result<(usize, usize, usize), GraphError> {
    match input.rank() {
        2 => Ok((1, input.shape[0], input.shape[1])),
        3 => Ok((input.shape[0], input.shape[1], input.shape[2])),
        r => Err(burn_err(format!("lstm input must be rank 2 or 3, got {r}"))),
    }
}

fn slice_sequence_step(
    input: &HostArray,
    step: usize,
    step_count: usize,
    batch: usize,
    input_size: usize,
) -> Result<HostArray, GraphError> {
    if input.rank() == 2 && step_count == 1 {
        Ok(input.clone())
    } else if input.rank() == 3 {
        let start = step * batch * input_size;
        Ok(HostArray::new(
            vec![batch, input_size],
            input.data[start..start + batch * input_size].to_vec(),
        )?)
    } else {
        Err(burn_err(format!(
            "cannot slice sequence step {step} from shape {:?}",
            input.shape
        )))
    }
}

fn copy_hidden_to_sequence(
    seq: &mut HostArray,
    step: usize,
    batch: usize,
    hs: usize,
    h: &HostArray,
) {
    let start = step * batch * hs;
    seq.data[start..start + batch * hs].copy_from_slice(&h.data[..batch * hs]);
}

fn gru_cell_internal(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    hidden_state: &HostArray,
    hs: usize,
    options: &MLGruCellOptions,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    direction: usize,
) -> Result<HostArray, GraphError> {
    let w = resolve_rnn_matrix(weight, direction)?;
    let r = resolve_rnn_matrix(recurrence, direction)?;
    let (batch, input_size) = matrix_batch_dims(input)?;
    let layout = options.layout.to_ascii_lowercase();
    let reset_after = options.reset_after;
    let activations = options.activations.as_deref();
    let w_bias = if options.bias.is_some() {
        bias.map(|b| resolve_rnn_bias(b, direction, 3 * hs))
            .transpose()?
    } else {
        None
    };
    let r_bias = if options.recurrent_bias.is_some() {
        recurrent_bias
            .map(|b| resolve_rnn_bias(b, direction, 3 * hs))
            .transpose()?
    } else {
        None
    };
    let mut h = hidden_state.clone();
    for bidx in 0..batch {
        let x_row = slice_row(input, bidx, input_size);
        let h_row = slice_row(&h, bidx, hs);
        let w_gates = linear_2d_with_bias_vec(&x_row, &w, w_bias.as_deref(), 3 * hs, input_size)?;
        let r_gates = linear_2d_with_bias_vec(&h_row, &r, r_bias.as_deref(), 3 * hs, hs)?;
        let (z, _r, n) = split_gru_gates(
            &w_gates,
            &r_gates,
            &h_row.data,
            &r,
            hs,
            &layout,
            reset_after,
            activations,
        )?;
        let h_new: Vec<f32> = z
            .iter()
            .zip(n.iter().zip(h_row.data.iter()))
            .map(|(&zi, (&ni, &hi))| (1.0 - zi) * ni + zi * hi)
            .collect();
        write_row(&mut h, bidx, hs, &h_new);
    }
    Ok(h)
}

fn stack_rnn_states(dir0: &HostArray, dir1: &HostArray) -> Result<HostArray, GraphError> {
    let batch = dir0.shape[0];
    let hs = dir0.shape[1];
    let mut out = vec![0.0; 2 * batch * hs];
    out[..batch * hs].copy_from_slice(&dir0.data);
    out[batch * hs..].copy_from_slice(&dir1.data);
    HostArray::new(vec![2, batch, hs], out)
}

fn stack_rnn_sequences(fwd: &HostArray, bwd: &HostArray) -> Result<HostArray, GraphError> {
    let steps = fwd.shape[0];
    let batch = fwd.shape[1];
    let hs = fwd.shape[2];
    let mut out = vec![0.0; steps * 2 * batch * hs];
    for t in 0..steps {
        let f_start = t * batch * hs;
        let b_start = t * batch * hs;
        let out_f = t * 2 * batch * hs;
        let out_b = out_f + batch * hs;
        out[out_f..out_f + batch * hs].copy_from_slice(&fwd.data[f_start..f_start + batch * hs]);
        out[out_b..out_b + batch * hs].copy_from_slice(&bwd.data[b_start..b_start + batch * hs]);
    }
    HostArray::new(vec![steps, 2, batch, hs], out)
}

pub fn format_rnn_state_nd(
    state: &HostArray,
    num_directions: usize,
) -> Result<HostArray, GraphError> {
    if state.rank() == 3 {
        return Ok(state.clone());
    }
    if state.rank() == 2 {
        return HostArray::new(
            vec![num_directions, state.shape[0], state.shape[1]],
            state.data.clone(),
        );
    }
    Err(burn_err(format!(
        "RNN state expected rank 2 or 3, got rank {} shape {:?}",
        state.rank(),
        state.shape
    )))
}

pub fn format_rnn_hidden_sequence(
    seq: &HostArray,
    num_directions: usize,
) -> Result<HostArray, GraphError> {
    if seq.rank() == 4 {
        return Ok(seq.clone());
    }
    if seq.rank() == 3 {
        return HostArray::new(
            vec![seq.shape[0], num_directions, seq.shape[1], seq.shape[2]],
            seq.data.clone(),
        );
    }
    Err(burn_err(format!(
        "RNN sequence expected rank 3 or 4, got rank {} shape {:?}",
        seq.rank(),
        seq.shape
    )))
}

fn map_resample_coord(out: usize, in_size: usize, out_size: usize, _linear: bool) -> f32 {
    if in_size == 0 || out_size == 0 {
        return 0.0;
    }
    let scale = out_size as f32 / in_size as f32;
    let unclamped = (out as f32 + 0.5) / scale - 0.5;
    unclamped.clamp(0.0, (in_size.saturating_sub(1)) as f32)
}

fn resample_nearest_index(out: usize, in_size: usize, out_size: usize) -> usize {
    let coord = map_resample_coord(out, in_size, out_size, false);
    (coord - 0.5).ceil().max(0.0) as usize
}

fn bilinear_nchw(
    get: impl Fn(usize, usize, usize, usize) -> f32,
    b: usize,
    ch: usize,
    h: usize,
    w: usize,
    src_y: f32,
    src_x: f32,
) -> f32 {
    let y0 = src_y.floor() as usize;
    let x0 = src_x.floor() as usize;
    let y1 = (y0 + 1).min(h - 1);
    let x1 = (x0 + 1).min(w - 1);
    let wy = src_y - y0 as f32;
    let wx = src_x - x0 as f32;
    let v00 = get(b, ch, y0, x0);
    let v01 = get(b, ch, y0, x1);
    let v10 = get(b, ch, y1, x0);
    let v11 = get(b, ch, y1, x1);
    let top = v00 * (1.0 - wx) + v01 * wx;
    let bot = v10 * (1.0 - wx) + v11 * wx;
    top * (1.0 - wy) + bot * wy
}

fn sample_at_axes(
    input: &HostArray,
    b: usize,
    ch: usize,
    axes: [usize; 2],
    sy: f32,
    sx: f32,
) -> f32 {
    let mut coords = vec![0usize; 4];
    coords[0] = b;
    coords[1] = ch;
    coords[axes[0]] = sy.round() as usize;
    coords[axes[1]] = sx.round() as usize;
    for (i, &dim) in input.shape.iter().enumerate() {
        if coords[i] >= dim {
            coords[i] = dim - 1;
        }
    }
    input.data[linear_index(&coords, &input.shape)]
}

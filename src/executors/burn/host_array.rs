//! Rank-agnostic host tensor utilities for Burn runtime interpretation.

use crate::error::GraphError;
use crate::operator_enums::MLOperandDataType;
use crate::operator_options::{
    MLBatchNormalizationOptions, MLConv2dOptions, MLConvTranspose2dOptions, MLDimension,
    MLInstanceNormalizationOptions, MLLayerNormalizationOptions, MLPool2dOptions,
    mldimensions_static_or_max,
};
use crate::shape_inference::{
    infer_conv_transpose2d_shape, infer_conv2d_shape, infer_pool2d_shape,
};

use super::host_ops_extra::round_to_nearest_even;

/// Host-resident tensor used during Burn plan interpretation.
#[derive(Debug, Clone, PartialEq)]
pub struct HostArray {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub i64_data: Option<Vec<i64>>,
    pub u64_data: Option<Vec<u64>>,
}

impl HostArray {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, GraphError> {
        let expected = shape.iter().product::<usize>();
        if data.len() != expected {
            return Err(burn_err(format!(
                "tensor shape {:?} expects {expected} elements, got {}",
                shape,
                data.len()
            )));
        }
        Ok(Self {
            shape,
            data,
            i64_data: None,
            u64_data: None,
        })
    }

    pub fn from_i64(shape: Vec<usize>, data: Vec<i64>) -> Result<Self, GraphError> {
        let expected = shape.iter().product::<usize>();
        if data.len() != expected {
            return Err(burn_err(format!(
                "tensor shape {:?} expects {expected} elements, got {}",
                shape,
                data.len()
            )));
        }
        Ok(Self {
            shape: shape.clone(),
            data: data.iter().map(|&v| v as f32).collect(),
            i64_data: Some(data),
            u64_data: None,
        })
    }

    pub fn from_u64(shape: Vec<usize>, data: Vec<u64>) -> Result<Self, GraphError> {
        let expected = shape.iter().product::<usize>();
        if data.len() != expected {
            return Err(burn_err(format!(
                "tensor shape {:?} expects {expected} elements, got {}",
                shape,
                data.len()
            )));
        }
        Ok(Self {
            shape: shape.clone(),
            data: data.iter().map(|&v| v as f32).collect(),
            i64_data: None,
            u64_data: Some(data),
        })
    }

    pub fn scalar(value: f32) -> Self {
        Self {
            shape: vec![],
            data: vec![value],
            i64_data: None,
            u64_data: None,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, GraphError> {
        let mut resolved = new_shape.to_vec();
        if resolved.iter().any(|&d| d == 0) {
            let total = self.numel();
            let known: usize = resolved.iter().filter(|&&d| d != 0).product();
            let inferred = if known == 0 { total } else { total / known };
            for dim in &mut resolved {
                if *dim == 0 {
                    *dim = inferred;
                }
            }
        }
        let expected: usize = resolved.iter().product();
        if expected != self.numel() {
            return Err(burn_err(format!(
                "reshape {:?} -> {:?} element count mismatch",
                self.shape, resolved
            )));
        }
        Ok(Self {
            shape: resolved,
            data: self.data.clone(),
            i64_data: None,
            u64_data: None,
        })
    }

    pub fn map_unary<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| f(x)).collect(),
            i64_data: None,
            u64_data: None,
        }
    }

    pub fn binary_broadcast<F>(&self, other: &Self, f: F) -> Result<Self, GraphError>
    where
        F: Fn(f32, f32) -> f32,
    {
        let out_shape = broadcast_shapes(&self.shape, &other.shape)?;
        let lhs = broadcast_to(self, &out_shape)?;
        let rhs = broadcast_to(other, &out_shape)?;
        Ok(Self {
            shape: out_shape,
            data: lhs
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(&a, &b)| f(a, b))
                .collect(),
            i64_data: None,
            u64_data: None,
        })
    }

    pub fn compare<F>(&self, other: &Self, f: F) -> Result<Self, GraphError>
    where
        F: Fn(f32, f32) -> bool,
    {
        self.binary_broadcast(other, |a, b| if f(a, b) { 1.0 } else { 0.0 })
    }
}

pub fn burn_err(reason: String) -> GraphError {
    GraphError::BurnRuntimeFailed { reason }
}

pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, GraphError> {
    let rank = a.len().max(b.len());
    let mut out = vec![1usize; rank];
    for i in 0..rank {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if da != db && da != 1 && db != 1 {
            return Err(burn_err(format!(
                "incompatible broadcast shapes {:?} and {:?}",
                a, b
            )));
        }
        out[rank - 1 - i] = da.max(db);
    }
    Ok(out)
}

pub fn broadcast_to(arr: &HostArray, target: &[usize]) -> Result<HostArray, GraphError> {
    if arr.shape == *target {
        return Ok(arr.clone());
    }
    let out_len: usize = target.iter().product::<usize>().max(1);
    let mut out = vec![0.0f32; out_len];

    fn fill(
        arr: &HostArray,
        target: &[usize],
        out: &mut [f32],
        dim: usize,
        out_index: usize,
        coord_prefix: &[usize],
    ) {
        if dim == target.len() {
            let in_index = linear_index_broadcast(arr, coord_prefix, target);
            out[out_index] = arr.data[in_index];
            return;
        }
        let stride = target[dim + 1..].iter().product::<usize>().max(1);
        for c in 0..target[dim] {
            let mut next = coord_prefix.to_vec();
            next.push(c);
            fill(arr, target, out, dim + 1, out_index + c * stride, &next);
        }
    }

    fill(arr, target, &mut out, 0, 0, &[]);
    Ok(HostArray {
        shape: target.to_vec(),
        data: out,
        i64_data: None,
        u64_data: None,
    })
}

fn linear_index_broadcast(arr: &HostArray, coords: &[usize], target: &[usize]) -> usize {
    let rank_diff = target.len().saturating_sub(arr.rank());
    let mut index = 0usize;
    let arr_strides = strides(&arr.shape);
    for (i, &c) in coords.iter().enumerate() {
        if i < rank_diff {
            continue;
        }
        let arr_axis = i - rank_diff;
        let in_coord = if arr.shape[arr_axis] == 1 { 0 } else { c };
        index += in_coord * arr_strides[arr_axis];
    }
    index
}

pub fn strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![1];
    }
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

pub fn linear_index(coords: &[usize], shape: &[usize]) -> usize {
    let strides = strides(shape);
    coords
        .iter()
        .zip(strides.iter())
        .map(|(&c, &s)| c * s)
        .sum()
}

pub fn transpose(arr: &HostArray, perm: &[usize]) -> Result<HostArray, GraphError> {
    if perm.len() != arr.rank() {
        return Err(burn_err(format!(
            "transpose permutation len {} != rank {}",
            perm.len(),
            arr.rank()
        )));
    }
    let out_shape: Vec<usize> = perm.iter().map(|&i| arr.shape[i]).collect();
    let out_len = out_shape.iter().product::<usize>().max(1);
    let mut out = vec![0.0f32; out_len];
    fn recurse(
        arr: &HostArray,
        perm: &[usize],
        out: &mut [f32],
        out_shape: &[usize],
        dim: usize,
        in_coords: &mut [usize],
        out_flat: &mut usize,
    ) {
        if dim == arr.rank() {
            let in_idx = linear_index(in_coords, &arr.shape);
            out[*out_flat] = arr.data[in_idx];
            *out_flat += 1;
            return;
        }
        for c in 0..arr.shape[dim] {
            in_coords[dim] = c;
            recurse(arr, perm, out, out_shape, dim + 1, in_coords, out_flat);
        }
    }
    let mut in_coords = vec![0usize; arr.rank()];
    let mut out_flat = 0usize;
    // Build output by iterating output coords
    fn out_recurse(
        arr: &HostArray,
        perm: &[usize],
        out: &mut [f32],
        out_shape: &[usize],
        dim: usize,
        out_coords: &mut [usize],
        out_flat: &mut usize,
    ) {
        if dim == out_shape.len() {
            let mut in_coords = vec![0usize; arr.rank()];
            for (out_i, &p) in perm.iter().enumerate() {
                in_coords[p] = out_coords[out_i];
            }
            let in_idx = linear_index(&in_coords, &arr.shape);
            out[*out_flat] = arr.data[in_idx];
            *out_flat += 1;
            return;
        }
        for c in 0..out_shape[dim] {
            out_coords[dim] = c;
            out_recurse(arr, perm, out, out_shape, dim + 1, out_coords, out_flat);
        }
    }
    let mut out_coords = vec![0usize; out_shape.len()];
    out_recurse(
        arr,
        perm,
        &mut out,
        &out_shape,
        0,
        &mut out_coords,
        &mut out_flat,
    );
    let _ = recurse;
    Ok(HostArray {
        shape: out_shape,
        data: out,
        i64_data: None,
        u64_data: None,
    })
}

pub fn concat(arrays: &[HostArray], axis: usize) -> Result<HostArray, GraphError> {
    if arrays.is_empty() {
        return Err(burn_err("concat requires at least one input".to_string()));
    }
    let rank = arrays[0].rank();
    if axis as usize >= rank.max(1) {
        return Err(burn_err(format!("concat axis {axis} out of range")));
    }
    let mut out_shape = arrays[0].shape.clone();
    out_shape[axis as usize] = arrays.iter().map(|a| a.shape[axis as usize]).sum();
    for arr in arrays.iter().skip(1) {
        if arr.rank() != rank {
            return Err(burn_err("concat inputs must have same rank".to_string()));
        }
        for (i, (&a, &b)) in arr.shape.iter().zip(out_shape.iter()).enumerate() {
            if i != axis as usize && a != b {
                return Err(burn_err(format!(
                    "concat shape mismatch at axis {i}: {a} vs {b}"
                )));
            }
        }
    }
    let out_len: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; out_len];
    let mut offset_along_axis = 0usize;
    for arr in arrays {
        copy_into_concat(arr, &mut out, &out_shape, axis as usize, offset_along_axis);
        offset_along_axis += arr.shape[axis as usize];
    }
    Ok(HostArray {
        shape: out_shape,
        data: out,
        i64_data: None,
        u64_data: None,
    })
}

fn copy_into_concat(
    src: &HostArray,
    dst: &mut [f32],
    out_shape: &[usize],
    axis: usize,
    offset: usize,
) {
    fn recurse(
        src: &HostArray,
        dst: &mut [f32],
        out_shape: &[usize],
        axis: usize,
        offset: usize,
        dim: usize,
        src_coords: &mut [usize],
        out_coords: &mut [usize],
    ) {
        if dim == src.rank() {
            out_coords[axis] = offset + src_coords[axis];
            let out_idx = linear_index(out_coords, out_shape);
            let src_idx = linear_index(src_coords, &src.shape);
            dst[out_idx] = src.data[src_idx];
            return;
        }
        for c in 0..src.shape[dim] {
            src_coords[dim] = c;
            out_coords[dim] = c;
            recurse(
                src,
                dst,
                out_shape,
                axis,
                offset,
                dim + 1,
                src_coords,
                out_coords,
            );
        }
    }
    let mut src_coords = vec![0usize; src.rank()];
    let mut out_coords = vec![0usize; out_shape.len()];
    recurse(
        src,
        dst,
        out_shape,
        axis,
        offset,
        0,
        &mut src_coords,
        &mut out_coords,
    );
}

pub fn slice_tensor(
    arr: &HostArray,
    starts: &[u32],
    sizes: &[MLDimension],
    strides: &[u32],
) -> Result<HostArray, GraphError> {
    let sizes_usize: Vec<usize> = mldimensions_static_or_max(sizes)
        .iter()
        .map(|&d| d as usize)
        .collect();
    if starts.len() != arr.rank() || sizes_usize.len() != arr.rank() {
        return Err(burn_err(format!(
            "slice starts/sizes rank mismatch with input rank {}",
            arr.rank()
        )));
    }
    let step: Vec<usize> = if strides.is_empty() {
        vec![1; arr.rank()]
    } else {
        strides.iter().map(|&s| s.max(1) as usize).collect()
    };
    let out_shape: Vec<usize> = sizes_usize
        .iter()
        .zip(step.iter())
        .map(|(&sz, &st)| if st == 1 { sz } else { (sz + st - 1) / st })
        .collect();
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let mut out = vec![0.0f32; out_len];
    fn recurse(
        arr: &HostArray,
        starts: &[u32],
        out_dims: &[usize],
        steps: &[usize],
        out: &mut [f32],
        dim: usize,
        in_coords: &mut [usize],
        out_coords: &mut [usize],
        out_flat: &mut usize,
    ) {
        if dim == arr.rank() {
            if in_coords.iter().zip(arr.shape.iter()).all(|(&c, &s)| c < s) {
                out[*out_flat] = arr.data[linear_index(in_coords, &arr.shape)];
            }
            *out_flat += 1;
            return;
        }
        for c in 0..out_dims[dim] {
            in_coords[dim] = starts[dim] as usize + c * steps[dim];
            out_coords[dim] = c;
            recurse(
                arr,
                starts,
                out_dims,
                steps,
                out,
                dim + 1,
                in_coords,
                out_coords,
                out_flat,
            );
        }
    }
    let mut in_coords = vec![0usize; arr.rank()];
    let mut out_coords = vec![0usize; out_shape.len()];
    let mut out_flat = 0usize;
    recurse(
        arr,
        starts,
        &out_shape,
        &step,
        &mut out,
        0,
        &mut in_coords,
        &mut out_coords,
        &mut out_flat,
    );
    Ok(HostArray {
        shape: out_shape,
        data: out,
        i64_data: None,
        u64_data: None,
    })
}

pub fn split_tensor(
    arr: &HostArray,
    axis: u32,
    splits: &[u32],
    equal_parts: Option<u32>,
) -> Result<Vec<HostArray>, GraphError> {
    let axis = axis as usize;
    if axis >= arr.rank() {
        return Err(burn_err(format!("split axis {axis} out of range")));
    }
    let part_sizes: Vec<usize> = if let Some(n) = equal_parts {
        let dim = arr.shape[axis];
        if dim % n as usize != 0 {
            return Err(burn_err(format!(
                "split equal parts {n} does not divide axis size {dim}"
            )));
        }
        vec![dim / n as usize; n as usize]
    } else {
        splits.iter().map(|&s| s as usize).collect()
    };
    let sum: usize = part_sizes.iter().sum();
    if sum != arr.shape[axis] {
        return Err(burn_err(format!(
            "split sizes sum {sum} != axis dim {}",
            arr.shape[axis]
        )));
    }
    let mut outputs = Vec::with_capacity(part_sizes.len());
    let mut start = 0usize;
    for size in part_sizes {
        let mut out_shape = arr.shape.clone();
        out_shape[axis] = size;
        let out_len: usize = out_shape.iter().product();
        let mut out = vec![0.0f32; out_len.max(1)];
        copy_slice_along_axis(arr, &mut out, &out_shape, axis, start);
        outputs.push(HostArray {
            shape: out_shape,
            data: out,
            i64_data: None,
            u64_data: None,
        });
        start += size;
    }
    Ok(outputs)
}

fn copy_slice_along_axis(
    src: &HostArray,
    dst: &mut [f32],
    out_shape: &[usize],
    axis: usize,
    start: usize,
) {
    for out_flat in 0..dst.len() {
        let mut in_coords = unravel(out_flat, out_shape);
        in_coords[axis] += start;
        dst[out_flat] = src.data[linear_index(&in_coords, &src.shape)];
    }
}

pub fn expand(arr: &HostArray, new_shape: &[MLDimension]) -> Result<HostArray, GraphError> {
    let target: Vec<usize> = mldimensions_static_or_max(new_shape)
        .iter()
        .map(|&d| d as usize)
        .collect();
    broadcast_to(arr, &target)
}

pub fn gather(
    data: &HostArray,
    indices: &HostArray,
    axis: u32,
    batch_dims: u32,
) -> Result<HostArray, GraphError> {
    let axis = axis as usize;
    let batch_dims = batch_dims as usize;
    if axis >= data.rank() {
        return Err(burn_err(format!("gather axis {axis} out of range")));
    }

    let mut out_shape = Vec::new();
    out_shape.extend_from_slice(&data.shape[..axis]);
    out_shape.extend_from_slice(&indices.shape);
    out_shape.extend_from_slice(&data.shape[axis + 1..]);

    let out_len = out_shape.iter().product::<usize>().max(1);
    let mut out = vec![0.0f32; out_len];
    let index_rank = indices.rank();

    for out_flat in 0..out_len {
        let out_coords = unravel(out_flat, &out_shape);
        let mut index_coords = vec![0usize; index_rank];
        let mut data_coords = vec![0usize; data.rank()];

        for i in 0..axis {
            data_coords[i] = out_coords[i];
        }
        for i in 0..index_rank {
            index_coords[i] = out_coords[axis + i];
        }
        for i in (axis + 1)..data.rank() {
            let out_i = axis + index_rank + (i - axis - 1);
            data_coords[i] = out_coords.get(out_i).copied().unwrap_or(0);
        }

        let idx = indices.data[linear_index(&index_coords, &indices.shape)];
        let idx_i = idx as i64;
        let axis_dim = data.shape[axis];
        let tail_shape: Vec<usize> = data.shape[axis + 1..].to_vec();
        let tail_offset = if tail_shape.is_empty() {
            0
        } else {
            let tail_coords: Vec<usize> = (axis + index_rank..out_coords.len())
                .map(|i| out_coords[i])
                .collect();
            linear_index(&tail_coords, &tail_shape)
        };

        if idx_i >= 0 && (idx_i as usize) >= axis_dim {
            let flat_base = idx_i as usize + axis_dim;
            if flat_base + tail_offset < data.data.len() {
                out[out_flat] = data.data[flat_base + tail_offset];
            }
            continue;
        }

        data_coords[axis] = if idx_i < 0 {
            (axis_dim as i64 + idx_i).max(0) as usize
        } else {
            idx_i as usize
        };
        if data_coords
            .iter()
            .zip(data.shape.iter())
            .all(|(&c, &s)| c < s)
        {
            out[out_flat] = data.data[linear_index(&data_coords, &data.shape)];
        }
    }

    let _ = batch_dims;
    Ok(HostArray {
        shape: out_shape,
        data: out,
        i64_data: None,
        u64_data: None,
    })
}

pub fn unravel(flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    let mut rem = flat;
    for i in (0..shape.len()).rev() {
        coords[i] = rem % shape[i];
        rem /= shape[i];
    }
    coords
}

pub fn matmul(a: &HostArray, b: &HostArray) -> Result<HostArray, GraphError> {
    if a.rank() < 2 || b.rank() < 2 {
        return Err(burn_err("matmul requires at least 2D tensors".to_string()));
    }
    let a_rows = a.shape[a.rank() - 2];
    let a_cols = a.shape[a.rank() - 1];
    let b_rows = b.shape[b.rank() - 2];
    let b_cols = b.shape[b.rank() - 1];
    if a_cols != b_rows {
        return Err(burn_err(format!(
            "matmul inner dim mismatch: {a_cols} != {b_rows}"
        )));
    }
    let batch_a = &a.shape[..a.rank() - 2];
    let batch_b = &b.shape[..b.rank() - 2];
    let batch = broadcast_shapes(batch_a, batch_b)?;
    let mut out_shape = batch;
    out_shape.push(a_rows);
    out_shape.push(b_cols);
    let batch_len: usize = out_shape[..out_shape.len() - 2]
        .iter()
        .product::<usize>()
        .max(1);
    let out_mat_len = a_rows * b_cols;
    let mut out = vec![0.0f32; batch_len * out_mat_len];

    for batch_idx in 0..batch_len {
        let a_batch_coords = unravel(batch_idx, &out_shape[..out_shape.len() - 2]);
        let b_batch_coords = broadcast_batch_coords(&a_batch_coords, batch_a, batch_b);
        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = 0.0f32;
                for k in 0..a_cols {
                    let a_idx = batch_offset(a, &a_batch_coords, batch_a) + i * a_cols + k;
                    let b_idx = batch_offset(b, &b_batch_coords, batch_b) + k * b_cols + j;
                    sum += a.data[a_idx] * b.data[b_idx];
                }
                out[batch_idx * out_mat_len + i * b_cols + j] = sum;
            }
        }
    }
    Ok(HostArray {
        shape: out_shape,
        data: out,
        i64_data: None,
        u64_data: None,
    })
}

/// Element-wise `pow` with NumPy-style broadcast and WebNN integer-exponent semantics.
pub fn pow_broadcast(a: &HostArray, b: &HostArray) -> Result<HostArray, GraphError> {
    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let a_bc = broadcast_to(a, &out_shape)?;
    let b_bc = broadcast_to(b, &out_shape)?;
    let data: Vec<f32> = a_bc
        .data
        .iter()
        .zip(b_bc.data.iter())
        .map(|(&base, &exp)| webnn_pow(base, exp))
        .collect();
    HostArray::new(out_shape, data)
}

/// WebNN `pow`: integer exponents use algebraic rules for negative bases; others use `powf`.
fn webnn_pow(base: f32, exp: f32) -> f32 {
    if exp.is_nan() {
        return f32::NAN;
    }
    if base.is_nan() {
        return f32::NAN;
    }
    let rounded = exp.round();
    if (exp - rounded).abs() < 1e-5 {
        return webnn_pow_integer(base, rounded as i32);
    }
    base.powf(exp)
}

/// Element-wise binary op with NumPy-style broadcast (host reference).
#[derive(Clone, Copy, Debug)]
pub enum HostBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

pub fn binary_broadcast(
    a: &HostArray,
    b: &HostArray,
    op: HostBinaryOp,
) -> Result<HostArray, GraphError> {
    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let a_bc = broadcast_to(a, &out_shape)?;
    let b_bc = broadcast_to(b, &out_shape)?;
    let data: Vec<f32> = a_bc
        .data
        .iter()
        .zip(b_bc.data.iter())
        .map(|(&x, &y)| match op {
            HostBinaryOp::Add => x + y,
            HostBinaryOp::Sub => x - y,
            HostBinaryOp::Mul => x * y,
            HostBinaryOp::Div => {
                if y == 0.0 {
                    f32::NAN
                } else {
                    x / y
                }
            }
        })
        .collect();
    HostArray::new(out_shape, data)
}

/// Integer `div` per WebNN WPT: divide in float, then round to nearest even.
pub fn integer_div_broadcast(a: &HostArray, b: &HostArray) -> Result<HostArray, GraphError> {
    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let a_bc = broadcast_to(a, &out_shape)?;
    let b_bc = broadcast_to(b, &out_shape)?;
    let data: Vec<f32> = a_bc
        .data
        .iter()
        .zip(b_bc.data.iter())
        .map(|(&x, &y)| {
            if y == 0.0 {
                f32::NAN
            } else {
                round_to_nearest_even(x / y)
            }
        })
        .collect();
    HostArray::new(out_shape, data)
}

fn webnn_pow_integer(base: f32, exp: i32) -> f32 {
    if exp == 0 {
        return 1.0;
    }
    if base == 0.0 {
        return if exp > 0 { 0.0 } else { f32::INFINITY };
    }
    if base.is_infinite() {
        return base.powi(exp);
    }
    if base < 0.0 {
        let mag = base.abs().powi(exp.unsigned_abs() as i32);
        let mut val = if exp > 0 { mag } else { 1.0 / mag };
        if exp.rem_euclid(2) == 1 {
            val = -val;
        }
        return val;
    }
    base.powi(exp)
}

fn broadcast_batch_coords(coords: &[usize], batch_a: &[usize], batch_b: &[usize]) -> Vec<usize> {
    let rank = batch_a.len().max(batch_b.len());
    let mut out = vec![0usize; rank];
    for i in 0..rank {
        let ca = if i + batch_a.len() >= rank {
            batch_a[i + batch_a.len() - rank]
        } else {
            1
        };
        let cb = if i + batch_b.len() >= rank {
            batch_b[i + batch_b.len() - rank]
        } else {
            1
        };
        let c = coords
            .get(i + coords.len().saturating_sub(rank))
            .copied()
            .unwrap_or(0);
        out[i] = if ca == 1 { 0 } else { c };
        let _ = cb;
    }
    out
}

fn batch_offset(arr: &HostArray, batch_coords: &[usize], batch_shape: &[usize]) -> usize {
    let rank_diff = batch_coords.len().saturating_sub(batch_shape.len());
    let mut coords = vec![0usize; arr.rank()];
    for (i, &c) in batch_coords.iter().enumerate() {
        if i < rank_diff {
            continue;
        }
        let axis = i - rank_diff;
        if axis < batch_shape.len() {
            coords[axis] = if batch_shape[axis] == 1 { 0 } else { c };
        }
    }
    linear_index(&coords, &arr.shape)
}

pub fn reduce_output_shape(shape: &[usize], axes: Option<&[u32]>, keep_dims: bool) -> Vec<usize> {
    let reduce_set: Vec<usize> = match axes {
        None => (0..shape.len()).collect(),
        Some([]) => Vec::new(),
        Some(ax) => ax.iter().map(|&a| a as usize).collect(),
    };
    if reduce_set.is_empty() {
        return shape.to_vec();
    }
    let mut out_shape = shape.to_vec();
    if keep_dims {
        for &ax in &reduce_set {
            if ax < out_shape.len() {
                out_shape[ax] = 1;
            }
        }
    } else {
        out_shape = out_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduce_set.contains(i))
            .map(|(_, &d)| d)
            .collect();
    }
    out_shape
}

pub fn reduce(
    arr: &HostArray,
    axes: Option<&[u32]>,
    keep_dims: bool,
    op: ReduceKind,
) -> Result<HostArray, GraphError> {
    let reduce_set: Vec<usize> = match axes {
        None => (0..arr.rank()).collect(),
        Some([]) => Vec::new(),
        Some(ax) => ax.iter().map(|&a| a as usize).collect(),
    };

    if reduce_set.is_empty() {
        let data: Vec<f32> = arr
            .data
            .iter()
            .map(|&v| match op {
                ReduceKind::L1 => v.abs(),
                ReduceKind::L2 => v.abs(),
                ReduceKind::SumSquare => v * v,
                ReduceKind::LogSum => v.ln(),
                ReduceKind::LogSumExp => v,
                ReduceKind::Sum => v,
                ReduceKind::Mean => v,
                ReduceKind::Max => v,
                ReduceKind::Min => v,
                ReduceKind::Product => v,
            })
            .collect();
        return Ok(HostArray {
            shape: arr.shape.clone(),
            data,
            i64_data: None,
            u64_data: None,
        });
    }

    let mut out_shape = arr.shape.clone();
    if keep_dims {
        for &ax in &reduce_set {
            if ax < out_shape.len() {
                out_shape[ax] = 1;
            }
        }
    } else {
        out_shape = out_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduce_set.contains(i))
            .map(|(_, &d)| d)
            .collect();
    }
    let out_len = out_shape.iter().product::<usize>().max(1);
    let mut out = vec![
        match op {
            ReduceKind::Max => f32::NEG_INFINITY,
            ReduceKind::Min => f32::INFINITY,
            ReduceKind::Product => 1.0,
            _ => 0.0,
        };
        out_len
    ];
    let mut counts = vec![0usize; out_len];

    for flat_in in 0..arr.numel() {
        let in_coords = unravel(flat_in, &arr.shape);
        let mut out_coords = Vec::new();
        for (i, &c) in in_coords.iter().enumerate() {
            if reduce_set.contains(&i) {
                if keep_dims {
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
        apply_reduce(&mut out[out_idx], v, op);
        counts[out_idx] += 1;
    }

    if matches!(op, ReduceKind::Mean) {
        for (v, &c) in out.iter_mut().zip(counts.iter()) {
            if c > 0 {
                *v /= c as f32;
            }
        }
    } else if matches!(op, ReduceKind::L2) {
        for v in &mut out {
            *v = v.sqrt();
        }
    } else if matches!(op, ReduceKind::LogSum) {
        for v in &mut out {
            *v = v.ln();
        }
    } else if matches!(op, ReduceKind::LogSumExp) {
        for out_idx in 0..out_len {
            let mut max_v = f32::NEG_INFINITY;
            for flat_in in 0..arr.numel() {
                let in_coords = unravel(flat_in, &arr.shape);
                let mut out_coords = Vec::new();
                for (i, &c) in in_coords.iter().enumerate() {
                    if reduce_set.contains(&i) {
                        if keep_dims {
                            out_coords.push(0);
                        }
                    } else {
                        out_coords.push(c);
                    }
                }
                let idx = if out_coords.is_empty() {
                    0
                } else {
                    linear_index(&out_coords, &out_shape)
                };
                if idx == out_idx {
                    max_v = max_v.max(arr.data[flat_in]);
                }
            }
            let mut sum = 0.0f32;
            for flat_in in 0..arr.numel() {
                let in_coords = unravel(flat_in, &arr.shape);
                let mut out_coords = Vec::new();
                for (i, &c) in in_coords.iter().enumerate() {
                    if reduce_set.contains(&i) {
                        if keep_dims {
                            out_coords.push(0);
                        }
                    } else {
                        out_coords.push(c);
                    }
                }
                let idx = if out_coords.is_empty() {
                    0
                } else {
                    linear_index(&out_coords, &out_shape)
                };
                if idx == out_idx {
                    sum += (arr.data[flat_in] - max_v).exp();
                }
            }
            out[out_idx] = max_v + sum.ln();
        }
    }

    Ok(HostArray {
        shape: out_shape,
        data: out,
        i64_data: None,
        u64_data: None,
    })
}

#[derive(Clone, Copy)]
pub enum ReduceKind {
    Sum,
    Mean,
    Max,
    Min,
    Product,
    L1,
    L2,
    LogSum,
    LogSumExp,
    SumSquare,
}

fn apply_reduce(acc: &mut f32, v: f32, op: ReduceKind) {
    match op {
        ReduceKind::Sum | ReduceKind::L1 | ReduceKind::SumSquare => {
            *acc += match op {
                ReduceKind::L1 => v.abs(),
                ReduceKind::SumSquare => v * v,
                _ => v,
            }
        }
        ReduceKind::Mean => *acc += v,
        ReduceKind::Max => {
            if v > *acc {
                *acc = v;
            }
        }
        ReduceKind::Min => {
            if v < *acc {
                *acc = v;
            }
        }
        ReduceKind::Product => *acc *= v,
        ReduceKind::L2 => *acc += v * v,
        ReduceKind::LogSum => *acc += v,
        ReduceKind::LogSumExp => *acc += v,
    }
}

pub fn softmax(arr: &HostArray, axis: u32) -> Result<HostArray, GraphError> {
    let axis = axis as usize;
    if axis >= arr.rank() {
        return Err(burn_err(format!("softmax axis {axis} out of range")));
    }
    let mut out = arr.clone();
    let outer: usize = arr.shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = arr.shape[axis + 1..].iter().product::<usize>().max(1);
    let axis_size = arr.shape[axis];
    for o in 0..outer {
        for i in 0..inner {
            let mut max_v = f32::NEG_INFINITY;
            for a in 0..axis_size {
                let idx = o * axis_size * inner + a * inner + i;
                max_v = max_v.max(arr.data[idx]);
            }
            let mut sum = 0.0f32;
            for a in 0..axis_size {
                let idx = o * axis_size * inner + a * inner + i;
                let e = (arr.data[idx] - max_v).exp();
                out.data[idx] = e;
                sum += e;
            }
            for a in 0..axis_size {
                let idx = o * axis_size * inner + a * inner + i;
                out.data[idx] /= sum;
            }
        }
    }
    Ok(out)
}

pub fn cast_values(data: &[f32], to: MLOperandDataType) -> Vec<f32> {
    use MLOperandDataType as T;
    data.iter()
        .map(|&v| match to {
            T::Float32 => v,
            T::Float16 => half::f16::from_f32(v).to_f32(),
            T::Int32 => v.trunc(),
            T::Uint32 => v.trunc().max(0.0),
            T::Int8 => v.trunc().clamp(-128.0, 127.0),
            T::Uint8 => v.trunc().clamp(0.0, 255.0),
            T::Int64 => v.trunc(),
            T::Uint64 => v.trunc().max(0.0),
        })
        .collect()
}

pub fn conv2d(
    input: &HostArray,
    filter: &HostArray,
    bias: Option<&HostArray>,
    options: &MLConv2dOptions,
) -> Result<HostArray, GraphError> {
    let input_shape_u32: Vec<u32> = input.shape.iter().map(|&d| d as u32).collect();
    let filter_shape_u32: Vec<u32> = filter.shape.iter().map(|&d| d as u32).collect();
    let out_shape_u32 = infer_conv2d_shape(&input_shape_u32, &filter_shape_u32, options)?;
    let out_shape: Vec<usize> = out_shape_u32.iter().map(|&d| d as usize).collect();
    let nchw = !options.input_layout.eq_ignore_ascii_case("nhwc");
    let (n, c_in, h, w) = if nchw {
        (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        )
    } else {
        (
            input.shape[0],
            input.shape[3],
            input.shape[1],
            input.shape[2],
        )
    };
    let (c_out, ic_per_filter, kh, kw) =
        conv2d_filter_dims_oihw(&options.filter_layout, &filter.shape)?;
    let filter_oihw = reorder_filter_to_oihw(
        &filter.data,
        &options.filter_layout,
        c_out,
        ic_per_filter,
        kh,
        kw,
    );
    let groups = options.groups.max(1) as usize;
    let strides = pad_to_2(&options.strides, 1);
    let dilations = pad_to_2(&options.dilations, 1);
    let padding = pad_to_4(&options.padding);
    let (out_n, out_c, out_h, out_w) = if nchw {
        (out_shape[0], out_shape[1], out_shape[2], out_shape[3])
    } else {
        (out_shape[0], out_shape[3], out_shape[1], out_shape[2])
    };
    let mut out = vec![0.0f32; out_n * out_c * out_h * out_w];

    let get_input = |b: usize, c: usize, y: usize, x: usize| -> f32 {
        if y >= h || x >= w {
            return 0.0;
        }
        if nchw {
            input.data[b * c_in * h * w + c * h * w + y * w + x]
        } else {
            input.data[b * h * w * c_in + y * w * c_in + x * c_in + c]
        }
    };

    let ic_per_group = c_in / groups.max(1);

    let get_filter = |oc: usize, ic_local: usize, ky: usize, kx: usize| -> f32 {
        filter_oihw[oc * ic_per_filter * kh * kw + ic_local * kh * kw + ky * kw + kx]
    };

    for b in 0..n {
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    let g = oc / (out_c / groups.max(1));
                    let ic_start = g * (c_in / groups.max(1));
                    let ic_end = ic_start + c_in / groups.max(1);
                    for ic in ic_start..ic_end {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oh * strides[0] as usize + ky * dilations[0] as usize;
                                let ix = ow * strides[1] as usize + kx * dilations[1] as usize;
                                let py = iy as i64 - padding[0] as i64;
                                let px = ix as i64 - padding[2] as i64;
                                if py >= 0 && px >= 0 && (py as usize) < h && (px as usize) < w {
                                    let val = get_input(b, ic, py as usize, px as usize);
                                    let f_ic = ic - ic_start;
                                    sum += val * get_filter(oc, f_ic, ky, kx);
                                }
                            }
                        }
                    }
                    if let Some(bias_t) = bias {
                        sum += bias_t.data.get(oc).copied().unwrap_or(0.0);
                    }
                    let out_idx = if nchw {
                        b * out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow
                    } else {
                        b * out_h * out_w * out_c + oh * out_w * out_c + ow * out_c + oc
                    };
                    out[out_idx] = sum;
                }
            }
        }
    }
    Ok(HostArray {
        shape: out_shape,
        data: out,
        i64_data: None,
        u64_data: None,
    })
}

pub fn conv_transpose2d(
    input: &HostArray,
    filter: &HostArray,
    bias: Option<&HostArray>,
    options: &MLConvTranspose2dOptions,
) -> Result<HostArray, GraphError> {
    let input_shape_u32: Vec<u32> = input.shape.iter().map(|&d| d as u32).collect();
    let filter_shape_u32: Vec<u32> = filter.shape.iter().map(|&d| d as u32).collect();
    let out_shape_u32 = infer_conv_transpose2d_shape(&input_shape_u32, &filter_shape_u32, options)?;
    let out_shape: Vec<usize> = out_shape_u32.iter().map(|&d| d as usize).collect();
    let nchw = !options.input_layout.eq_ignore_ascii_case("nhwc");
    let (n, c_in, h, w) = if nchw {
        (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        )
    } else {
        (
            input.shape[0],
            input.shape[3],
            input.shape[1],
            input.shape[2],
        )
    };
    let (filter_in, c_out_per_group, kh, kw) =
        deconv_filter_dims_iohw(&options.filter_layout, &filter.shape)?;
    let filter_iohw = reorder_deconv_filter_to_iohw(
        &filter.data,
        &options.filter_layout,
        filter_in,
        c_out_per_group,
        kh,
        kw,
    );
    let groups = options.groups.max(1) as usize;
    let out_c = c_out_per_group * groups;
    let strides = pad_to_2(&options.strides, 1);
    let dilations = pad_to_2(&options.dilations, 1);
    let padding = pad_to_4(&options.padding);
    let (_, _, out_h, out_w) = if nchw {
        (out_shape[0], out_shape[1], out_shape[2], out_shape[3])
    } else {
        (out_shape[0], out_shape[3], out_shape[1], out_shape[2])
    };
    let mut out = vec![0.0f32; n * out_c * out_h * out_w];

    let get_input = |b: usize, c: usize, y: usize, x: usize| -> f32 {
        if nchw {
            input.data[b * c_in * h * w + c * h * w + y * w + x]
        } else {
            input.data[b * h * w * c_in + y * w * c_in + x * c_in + c]
        }
    };

    let ic_per_group = c_in / groups.max(1);

    let get_filter = |ic: usize, oc: usize, ky: usize, kx: usize| -> f32 {
        let g = ic / ic_per_group.max(1);
        let oc_local = oc - g * c_out_per_group;
        filter_iohw[ic * c_out_per_group * kh * kw + oc_local * kh * kw + ky * kw + kx]
    };

    for b in 0..n {
        for ic in 0..c_in {
            for ih in 0..h {
                for iw in 0..w {
                    let in_val = get_input(b, ic, ih, iw);
                    let g = ic / ic_per_group.max(1);
                    let oc_start = g * c_out_per_group;
                    let oc_end = oc_start + c_out_per_group;
                    for oc in oc_start..oc_end {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let oh = ih * strides[0] as usize + ky * dilations[0] as usize;
                                let ow = iw * strides[1] as usize + kx * dilations[1] as usize;
                                let oh = oh as i64 - padding[0] as i64;
                                let ow = ow as i64 - padding[2] as i64;
                                if oh >= 0
                                    && ow >= 0
                                    && (oh as usize) < out_h
                                    && (ow as usize) < out_w
                                {
                                    let idx = if nchw {
                                        b * out_c * out_h * out_w
                                            + oc * out_h * out_w
                                            + oh as usize * out_w
                                            + ow as usize
                                    } else {
                                        b * out_h * out_w * out_c
                                            + oh as usize * out_w * out_c
                                            + ow as usize * out_c
                                            + oc
                                    };
                                    out[idx] += in_val * get_filter(ic, oc, ky, kx);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(bias_t) = bias {
        for oc in 0..out_c {
            let bval = bias_t.data.get(oc).copied().unwrap_or(0.0);
            for b in 0..n {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let idx = if nchw {
                            b * out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow
                        } else {
                            b * out_h * out_w * out_c + oh * out_w * out_c + ow * out_c + oc
                        };
                        out[idx] += bval;
                    }
                }
            }
        }
    }

    Ok(HostArray {
        shape: out_shape,
        data: out,
        i64_data: None,
        u64_data: None,
    })
}

pub(crate) fn conv2d_filter_dims_oihw(
    layout: &str,
    shape: &[usize],
) -> Result<(usize, usize, usize, usize), GraphError> {
    if shape.len() != 4 {
        return Err(burn_err(format!(
            "conv2d filter must be 4D, got {:?}",
            shape
        )));
    }
    Ok(match layout.to_ascii_lowercase().as_str() {
        "hwio" => (shape[3], shape[2], shape[0], shape[1]),
        "ohwi" => (shape[0], shape[3], shape[1], shape[2]),
        "ihwo" => (shape[3], shape[0], shape[1], shape[2]),
        "hwoi" => (shape[2], shape[3], shape[0], shape[1]),
        _ => (shape[0], shape[1], shape[2], shape[3]),
    })
}

fn deconv_filter_dims_iohw(
    layout: &str,
    shape: &[usize],
) -> Result<(usize, usize, usize, usize), GraphError> {
    if shape.len() != 4 {
        return Err(burn_err(format!(
            "convTranspose2d filter must be 4D, got {:?}",
            shape
        )));
    }
    Ok(match layout.to_ascii_lowercase().as_str() {
        "oihw" => (shape[1], shape[0], shape[2], shape[3]),
        "hwio" => (shape[2], shape[3], shape[0], shape[1]),
        "ohwi" => (shape[3], shape[0], shape[1], shape[2]),
        "ihwo" => (shape[0], shape[3], shape[1], shape[2]),
        "hwoi" => (shape[3], shape[2], shape[0], shape[1]),
        _ => (shape[0], shape[1], shape[2], shape[3]),
    })
}

pub(crate) fn reorder_filter_to_oihw(
    data: &[f32],
    layout: &str,
    o: usize,
    i: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; o * i * h * w];
    for oo in 0..o {
        for ii in 0..i {
            for hh in 0..h {
                for ww in 0..w {
                    let src_idx = match layout.to_ascii_lowercase().as_str() {
                        "oihw" => oo * (i * h * w) + ii * (h * w) + hh * w + ww,
                        "hwio" => hh * (w * i * o) + ww * (i * o) + ii * o + oo,
                        "ohwi" => oo * (h * w * i) + hh * (w * i) + ww * i + ii,
                        "ihwo" => ii * (h * w * o) + hh * (w * o) + ww * o + oo,
                        "hwoi" => hh * (w * o * i) + ww * (o * i) + oo * i + ii,
                        _ => oo * (i * h * w) + ii * (h * w) + hh * w + ww,
                    };
                    let dst_idx = oo * (i * h * w) + ii * (h * w) + hh * w + ww;
                    dst[dst_idx] = data[src_idx];
                }
            }
        }
    }
    dst
}

fn reorder_deconv_filter_to_iohw(
    data: &[f32],
    layout: &str,
    i: usize,
    o: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; i * o * h * w];
    for ii in 0..i {
        for oo in 0..o {
            for hh in 0..h {
                for ww in 0..w {
                    let src_idx = match layout.to_ascii_lowercase().as_str() {
                        "iohw" => ii * (o * h * w) + oo * (h * w) + hh * w + ww,
                        "oihw" => oo * (i * h * w) + ii * (h * w) + hh * w + ww,
                        "hwio" => hh * (w * i * o) + ww * (i * o) + ii * o + oo,
                        "ohwi" => oo * (h * w * i) + hh * (w * i) + ww * i + ii,
                        "ihwo" => ii * (h * w * o) + hh * (w * o) + ww * o + oo,
                        "hwoi" => hh * (w * o * i) + ww * (o * i) + oo * i + ii,
                        _ => ii * (o * h * w) + oo * (h * w) + hh * w + ww,
                    };
                    let dst_idx = ii * (o * h * w) + oo * (h * w) + hh * w + ww;
                    dst[dst_idx] = data[src_idx];
                }
            }
        }
    }
    dst
}

pub fn pool2d(
    input: &HostArray,
    options: &MLPool2dOptions,
    kind: PoolKind,
) -> Result<HostArray, GraphError> {
    let input_shape_u32: Vec<u32> = input.shape.iter().map(|&d| d as u32).collect();
    let out_shape_u32 = infer_pool2d_shape(&input_shape_u32, options)?;
    let out_shape: Vec<usize> = out_shape_u32.iter().map(|&d| d as usize).collect();
    let nchw = !options.layout.eq_ignore_ascii_case("nhwc");
    let strides = pad_to_2(&options.strides, 1);
    let dilations = pad_to_2(&options.dilations, 1);
    let padding = pad_to_4(&options.padding);
    let (n, c, h, w) = if nchw {
        (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        )
    } else {
        (
            input.shape[0],
            input.shape[3],
            input.shape[1],
            input.shape[2],
        )
    };
    let window = options
        .window_dimensions
        .as_ref()
        .map(|w| [w[0] as usize, w[1] as usize])
        .unwrap_or([h, w]);
    let (_, out_c, out_h, out_w) = if nchw {
        (out_shape[0], out_shape[1], out_shape[2], out_shape[3])
    } else {
        (out_shape[0], out_shape[3], out_shape[1], out_shape[2])
    };
    let mut out = vec![0.0f32; n * out_c * out_h * out_w];
    let get = |b: usize, ch: usize, y: usize, x: usize| -> f32 {
        if y >= h || x >= w {
            return f32::NAN;
        }
        if nchw {
            input.data[b * c * h * w + ch * h * w + y * w + x]
        } else {
            input.data[b * h * w * c + y * w * c + x * c + ch]
        }
    };
    for b in 0..n {
        for ch in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut vals = Vec::new();
                    for ky in 0..window[0] {
                        for kx in 0..window[1] {
                            let ih = oh as i64 * strides[0] as i64
                                + ky as i64 * dilations[0] as i64
                                - padding[0] as i64;
                            let iw = ow as i64 * strides[1] as i64
                                + kx as i64 * dilations[1] as i64
                                - padding[2] as i64;
                            if ih >= 0 && iw >= 0 && (ih as usize) < h && (iw as usize) < w {
                                vals.push(get(b, ch, ih as usize, iw as usize));
                            }
                        }
                    }
                    let pooled = match kind {
                        PoolKind::Average => vals.iter().sum::<f32>() / vals.len().max(1) as f32,
                        PoolKind::Max => {
                            if vals.is_empty() {
                                0.0
                            } else {
                                vals.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                            }
                        }
                        PoolKind::L2 => {
                            let sum_sq: f32 = vals.iter().map(|v| v * v).sum();
                            sum_sq.sqrt()
                        }
                    };
                    let idx = if nchw {
                        b * out_c * out_h * out_w + ch * out_h * out_w + oh * out_w + ow
                    } else {
                        b * out_h * out_w * out_c + oh * out_w * out_c + ow * out_c + ch
                    };
                    out[idx] = pooled;
                }
            }
        }
    }
    Ok(HostArray {
        shape: out_shape,
        data: out,
        i64_data: None,
        u64_data: None,
    })
}

#[derive(Clone, Copy)]
pub enum PoolKind {
    Average,
    Max,
    L2,
}

pub fn batch_normalization(
    input: &HostArray,
    mean: &HostArray,
    variance: &HostArray,
    options: &MLBatchNormalizationOptions,
    scale: Option<&HostArray>,
    bias: Option<&HostArray>,
) -> Result<HostArray, GraphError> {
    let axis = options.axis as usize;
    let eps = options.epsilon as f32;
    let mut out = input.clone();
    let channels = mean.numel();
    for flat in 0..input.numel() {
        let coords = unravel(flat, &input.shape);
        let c = coords.get(axis).copied().unwrap_or(0) % channels;
        let m = mean.data.get(c).copied().unwrap_or(0.0);
        let v = variance.data.get(c).copied().unwrap_or(1.0);
        let s = scale
            .map(|t| t.data.get(c).copied().unwrap_or(1.0))
            .unwrap_or(1.0);
        let b = bias
            .map(|t| t.data.get(c).copied().unwrap_or(0.0))
            .unwrap_or(0.0);
        out.data[flat] = s * (input.data[flat] - m) / (v + eps).sqrt() + b;
    }
    Ok(out)
}

pub fn instance_normalization(
    input: &HostArray,
    options: &MLInstanceNormalizationOptions,
    scale: Option<&HostArray>,
    bias: Option<&HostArray>,
) -> Result<HostArray, GraphError> {
    let eps = options.epsilon as f32;
    let nchw = !options.layout.eq_ignore_ascii_case("nhwc");
    if input.rank() != 4 {
        return Err(burn_err(
            "instanceNormalization expects 4D input".to_string(),
        ));
    }
    let (n, c, h, w) = if nchw {
        (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        )
    } else {
        (
            input.shape[0],
            input.shape[3],
            input.shape[1],
            input.shape[2],
        )
    };
    let spatial = h * w;
    let mut out = input.clone();
    for b in 0..n {
        for ch in 0..c {
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            for y in 0..h {
                for x in 0..w {
                    let idx = if nchw {
                        b * c * h * w + ch * h * w + y * w + x
                    } else {
                        b * h * w * c + y * w * c + x * c + ch
                    };
                    sum += input.data[idx];
                    sum_sq += input.data[idx] * input.data[idx];
                }
            }
            let mean = sum / spatial as f32;
            let var = sum_sq / spatial as f32 - mean * mean;
            let s = scale.and_then(|t| t.data.get(ch).copied()).unwrap_or(1.0);
            let bi = bias.and_then(|t| t.data.get(ch).copied()).unwrap_or(0.0);
            for y in 0..h {
                for x in 0..w {
                    let idx = if nchw {
                        b * c * h * w + ch * h * w + y * w + x
                    } else {
                        b * h * w * c + y * w * c + x * c + ch
                    };
                    out.data[idx] = s * (input.data[idx] - mean) / (var + eps).sqrt() + bi;
                }
            }
        }
    }
    Ok(out)
}

pub fn layer_normalization(
    input: &HostArray,
    options: &MLLayerNormalizationOptions,
    scale: Option<&HostArray>,
    bias: Option<&HostArray>,
) -> Result<HostArray, GraphError> {
    let eps = options.epsilon as f32;
    let rank = input.rank();
    let axes: Vec<usize> = options
        .axes
        .as_ref()
        .map(|a| a.iter().map(|&x| x as usize).collect())
        .unwrap_or_else(|| (1..rank).collect());
    let reduce_set: std::collections::HashSet<usize> = axes.iter().copied().collect();
    let norm_count: f32 = axes
        .iter()
        .map(|&a| input.shape.get(a).copied().unwrap_or(1) as f32)
        .product::<f32>()
        .max(1.0);

    let outer_shape: Vec<usize> = input
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| !reduce_set.contains(i))
        .map(|(_, &d)| d)
        .collect();
    let outer_len = outer_shape.iter().product::<usize>().max(1);
    let norm_shape: Vec<usize> = axes
        .iter()
        .map(|&a| input.shape.get(a).copied().unwrap_or(1))
        .collect();

    let mut means = vec![0.0f32; outer_len];
    let mut sum_sq = vec![0.0f32; outer_len];

    for flat_in in 0..input.numel() {
        let coords = unravel(flat_in, &input.shape);
        let outer_coords: Vec<usize> = coords
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduce_set.contains(i))
            .map(|(_, &c)| c)
            .collect();
        let outer_idx = if outer_coords.is_empty() {
            0
        } else {
            linear_index(&outer_coords, &outer_shape)
        };
        means[outer_idx] += input.data[flat_in];
        sum_sq[outer_idx] += input.data[flat_in] * input.data[flat_in];
    }

    for i in 0..outer_len {
        means[i] /= norm_count;
        sum_sq[i] = sum_sq[i] / norm_count - means[i] * means[i];
    }

    let mut out = input.clone();
    for flat_in in 0..input.numel() {
        let coords = unravel(flat_in, &input.shape);
        let outer_coords: Vec<usize> = coords
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduce_set.contains(i))
            .map(|(_, &c)| c)
            .collect();
        let outer_idx = if outer_coords.is_empty() {
            0
        } else {
            linear_index(&outer_coords, &outer_shape)
        };
        let norm_coords: Vec<usize> = axes.iter().map(|&a| coords[a]).collect();
        let norm_idx = linear_index(&norm_coords, &norm_shape);
        let s = scale
            .and_then(|t| t.data.get(norm_idx).copied())
            .unwrap_or(1.0);
        let b = bias
            .and_then(|t| t.data.get(norm_idx).copied())
            .unwrap_or(0.0);
        out.data[flat_in] =
            s * (input.data[flat_in] - means[outer_idx]) / (sum_sq[outer_idx] + eps).sqrt() + b;
    }
    Ok(out)
}

pub(crate) fn pad_to_2(v: &[u32], default: u32) -> [u32; 2] {
    [
        v.first().copied().unwrap_or(default),
        v.get(1).copied().unwrap_or(default),
    ]
}

pub(crate) fn pad_to_4(v: &[u32]) -> [u32; 4] {
    [
        v.first().copied().unwrap_or(0),
        v.get(1).copied().unwrap_or(0),
        v.get(2).copied().unwrap_or(0),
        v.get(3).copied().unwrap_or(0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_allows_zero_sized_dimension() {
        let arr = HostArray::new(vec![1, 3, 0, 64], vec![]).unwrap();
        assert_eq!(arr.numel(), 0);
    }

    #[test]
    fn concat_empty_past_with_present_key() {
        let past = HostArray::new(vec![1, 3, 0, 64], vec![]).unwrap();
        let present = HostArray::new(vec![1, 3, 1, 64], vec![0.0; 192]).unwrap();
        let out = concat(&[past, present], 2).unwrap();
        assert_eq!(out.shape, vec![1, 3, 1, 64]);
        assert_eq!(out.data.len(), 192);
    }

    #[test]
    fn pool2d_default_window_is_full_spatial() {
        use crate::operator_options::MLPool2dOptions;
        let input = HostArray::new(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = pool2d(&input, &MLPool2dOptions::default(), PoolKind::Average).unwrap();
        assert_eq!(out.shape, vec![1, 1, 1, 1]);
        assert!((out.data[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn webnn_pow_negative_base_integer_exponent() {
        let base = HostArray::new(vec![1], vec![-9.8671875]).unwrap();
        let exp = HostArray::new(vec![1], vec![-7.0]).unwrap();
        let out = pow_broadcast(&base, &exp).unwrap();
        assert!(!out.data[0].is_nan());
        assert!(out.data[0] < 0.0);
    }

    #[test]
    fn integer_div_rounds_to_nearest_even() {
        let a = HostArray::new(vec![1], vec![19.0]).unwrap();
        let b = HostArray::new(vec![1], vec![2.0]).unwrap();
        let out = integer_div_broadcast(&a, &b).unwrap();
        assert_eq!(out.data[0] as i32, 10);
    }

    #[test]
    fn matmul_batched_with_lower_rank_rhs() {
        let a = HostArray::new(vec![2, 3, 4], vec![0.0; 24]).unwrap();
        let b = HostArray::new(vec![4, 1], vec![0.0; 4]).unwrap();
        let out = matmul(&a, &b).unwrap();
        assert_eq!(out.shape, vec![2, 3, 1]);
    }
}

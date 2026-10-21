/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! Rank-agnostic host tensor utilities for Burn runtime interpretation.

// Indexed loops express tensor axes and contraction dimensions.
#![allow(clippy::needless_range_loop)]

use super::options::parse_option_or_default as parse_option;
use super::shape::ReductionSpec;
use super::shape::{broadcast_shapes, linear_index, strides, unravel};
use crate::error::GraphError;
use crate::operator_enums::{
    MLConv2dFilterOperandLayout, MLConvTranspose2dFilterOperandLayout, MLInputOperandLayout,
};
use crate::operator_options::{
    MLBatchNormalizationOptions, MLConvTranspose2dOptions, MLInstanceNormalizationOptions,
    MLLayerNormalizationOptions, MLPool2dOptions,
};
use crate::shape_inference::{infer_conv_transpose2d_shape, infer_pool2d_shape};

/// Host-resident tensor used during Burn plan interpretation.
#[derive(Debug, Clone, PartialEq)]
pub struct HostArray {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl HostArray {
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
        })
    }

    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, GraphError> {
        let expected = super::shape::checked_numel(&shape)?;
        if data.len() != expected {
            return Err(burn_err(format!(
                "tensor shape {:?} expects {expected} elements, got {}",
                shape,
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn map_unary<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }
}

pub fn burn_err(reason: String) -> GraphError {
    GraphError::BurnRuntimeFailed { reason }
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

pub fn reduce(
    arr: &HostArray,
    axes: Option<&[u32]>,
    keep_dims: bool,
    op: ReduceKind,
) -> Result<HostArray, GraphError> {
    let spec = ReductionSpec::new(&arr.shape, axes, keep_dims)?;
    let reduce_set = &spec.axes;
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
        });
    }

    let out_shape = spec.output_shape.clone();
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
        let out_coords = spec.output_coords(&in_coords);
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
                let out_coords = spec.output_coords(&in_coords);
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
                let out_coords = spec.output_coords(&in_coords);
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
    let nchw = options.input_layout != crate::operator_enums::MLInputOperandLayout::Nhwc;
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
        deconv_filter_dims_iohw(options.filter_layout, &filter.shape)?;
    let filter_iohw = reorder_deconv_filter_to_iohw(
        &filter.data,
        options.filter_layout,
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
    })
}

pub(crate) fn conv2d_filter_dims_oihw(
    layout: MLConv2dFilterOperandLayout,
    shape: &[usize],
) -> Result<(usize, usize, usize, usize), GraphError> {
    if shape.len() != 4 {
        return Err(burn_err(format!("conv2d filter must be 4D, got {shape:?}")));
    }
    use MLConv2dFilterOperandLayout::*;
    Ok(match layout {
        Oihw => (shape[0], shape[1], shape[2], shape[3]),
        Hwio => (shape[3], shape[2], shape[0], shape[1]),
        Ohwi => (shape[0], shape[3], shape[1], shape[2]),
        Ihwo => (shape[3], shape[0], shape[1], shape[2]),
    })
}
fn deconv_filter_dims_iohw(
    layout: MLConvTranspose2dFilterOperandLayout,
    shape: &[usize],
) -> Result<(usize, usize, usize, usize), GraphError> {
    if shape.len() != 4 {
        return Err(burn_err(format!(
            "convTranspose2d filter must be 4D, got {shape:?}"
        )));
    }
    use MLConvTranspose2dFilterOperandLayout::*;
    Ok(match layout {
        Iohw => (shape[0], shape[1], shape[2], shape[3]),
        Hwoi => (shape[3], shape[2], shape[0], shape[1]),
        Ohwi => (shape[3], shape[0], shape[1], shape[2]),
    })
}
// Flatten one logical four-axis permutation. Layout matches below only select
// physical axis order, keeping the indexing arithmetic in one place.
fn reorder_filter(data: &[f32], dims: [usize; 4], physical_axes: [usize; 4]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    for a in 0..dims[0] {
        for b in 0..dims[1] {
            for c in 0..dims[2] {
                for d in 0..dims[3] {
                    let coords = [a, b, c, d];
                    let index = physical_axes
                        .iter()
                        .fold(0, |index, &axis| index * dims[axis] + coords[axis]);
                    out.push(data[index]);
                }
            }
        }
    }
    out
}
pub(crate) fn reorder_filter_to_oihw(
    data: &[f32],
    layout: MLConv2dFilterOperandLayout,
    o: usize,
    i: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    use MLConv2dFilterOperandLayout::*;
    let axes = match layout {
        Oihw => [0, 1, 2, 3],
        Hwio => [2, 3, 1, 0],
        Ohwi => [0, 2, 3, 1],
        Ihwo => [1, 2, 3, 0],
    };
    reorder_filter(data, [o, i, h, w], axes)
}
fn reorder_deconv_filter_to_iohw(
    data: &[f32],
    layout: MLConvTranspose2dFilterOperandLayout,
    i: usize,
    o: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    use MLConvTranspose2dFilterOperandLayout::*;
    let axes = match layout {
        Iohw => [0, 1, 2, 3],
        Hwoi => [2, 3, 1, 0],
        Ohwi => [1, 2, 3, 0],
    };
    reorder_filter(data, [i, o, h, w], axes)
}

pub fn pool2d(
    input: &HostArray,
    options: &MLPool2dOptions,
    kind: PoolKind,
) -> Result<HostArray, GraphError> {
    let input_shape_u32: Vec<u32> = input.shape.iter().map(|&d| d as u32).collect();
    let out_shape_u32 = infer_pool2d_shape(&input_shape_u32, options)?;
    let out_shape: Vec<usize> = out_shape_u32.iter().map(|&d| d as usize).collect();
    let nchw = parse_option::<MLInputOperandLayout>(&options.layout, "layout")?
        == MLInputOperandLayout::Nchw;
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

// Accumulate centered moments in higher precision so a large common offset does
// not erase a small variance. This also avoids overflowing float32 squares.
#[derive(Clone, Copy, Default)]
struct Moments {
    count: usize,
    mean: f64,
    m2: f64,
}

impl Moments {
    fn add(&mut self, value: f32) {
        self.count += 1;
        let value = f64::from(value);
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (value - self.mean);
    }

    fn normalize(self, value: f32, epsilon: f32, scale: f32, bias: f32) -> f32 {
        let variance = self.m2 / self.count.max(1) as f64;
        (f64::from(scale) * (f64::from(value) - self.mean) / (variance + f64::from(epsilon)).sqrt()
            + f64::from(bias)) as f32
    }
}

pub fn instance_normalization(
    input: &HostArray,
    options: &MLInstanceNormalizationOptions,
    scale: Option<&HostArray>,
    bias: Option<&HostArray>,
) -> Result<HostArray, GraphError> {
    let eps = options.epsilon as f32;
    let nchw = parse_option::<MLInputOperandLayout>(&options.layout, "layout")?
        == MLInputOperandLayout::Nchw;
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
    let mut out = input.clone();
    for b in 0..n {
        for ch in 0..c {
            let mut moments = Moments::default();
            for y in 0..h {
                for x in 0..w {
                    let idx = if nchw {
                        b * c * h * w + ch * h * w + y * w + x
                    } else {
                        b * h * w * c + y * w * c + x * c + ch
                    };
                    moments.add(input.data[idx]);
                }
            }
            let s = scale.and_then(|t| t.data.get(ch).copied()).unwrap_or(1.0);
            let bi = bias.and_then(|t| t.data.get(ch).copied()).unwrap_or(0.0);
            for y in 0..h {
                for x in 0..w {
                    let idx = if nchw {
                        b * c * h * w + ch * h * w + y * w + x
                    } else {
                        b * h * w * c + y * w * c + x * c + ch
                    };
                    out.data[idx] = moments.normalize(input.data[idx], eps, s, bi);
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

    let mut moments = vec![Moments::default(); outer_len];

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
        moments[outer_idx].add(input.data[flat_in]);
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
        out.data[flat_in] = moments[outer_idx].normalize(input.data[flat_in], eps, s, b);
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
}

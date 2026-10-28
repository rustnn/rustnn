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

//! Additional host-side WebNN operators for the Burn executor.

// Indexed loops express the tensor axes used by these floating-point kernels.
#![allow(clippy::needless_range_loop)]

use super::host_array::{HostArray, PoolKind, broadcast_to, burn_err, pool2d};
use super::options::parse_option_or_default as parse_option;
use super::shape::{broadcast_shapes, linear_index, unravel};
use crate::error::GraphError;
use crate::operator_enums::MLInterpolationMode;
use crate::operator_options::{MLCumulativeSumOptions, MLPool2dOptions, MLResample2dOptions};

// ---------------------------------------------------------------------------
// Unary elementwise
// ---------------------------------------------------------------------------

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

pub fn round_even(arr: &HostArray) -> HostArray {
    arr.map_unary(round_to_nearest_even)
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
    let mut out = HostArray::new(arr.shape.clone(), arr.data.clone())?;
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
    })
}

// ---------------------------------------------------------------------------
// Pooling
// ---------------------------------------------------------------------------

pub fn l2_pool2d(input: &HostArray, options: &MLPool2dOptions) -> Result<HostArray, GraphError> {
    pool2d(input, options, PoolKind::L2)
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
            ((input.shape[axes[0]].max(1) as f64) * f64::from(s0))
                .floor()
                .max(1.0) as usize,
            ((input.shape[axes[1]].max(1) as f64) * f64::from(s1))
                .floor()
                .max(1.0) as usize,
        ]
    };
    out_shape[axes[0]] = out_sizes[0];
    out_shape[axes[1]] = out_sizes[1];

    let linear = parse_option::<MLInterpolationMode>(&options.mode, "interpolation mode")?
        == MLInterpolationMode::Linear;
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
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn erf_f32(x: f32) -> f32 {
    // Abramowitz and Stegun approximation 7.1.26
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

pub(crate) fn round_to_nearest_even(x: f32) -> f32 {
    let floor = x.floor();
    let frac = x - floor;
    if frac > 0.5 {
        floor + 1.0
    } else if frac < 0.5 || (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
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

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

//! Exact integer execution and datatype-preserving tensor movement.
//!
//! Kernels read the operand's authoritative enum-backed buffer. Integer operands never
//! allocate a Burn float tensor; float views are produced only at the public output boundary.

use super::host_array::HostArray;
use super::options::parse_option_or_default as parse_option;
use super::outputs::OperationOutputs;
use super::shape::ReductionSpec;
use super::shape::{broadcast_shapes, linear_index, unravel};
use super::tensor_env::TensorEnv;
use crate::operator_enums::{MLInputOperandLayout, MLInterpolationMode};
use crate::operator_options::{MLDimension, MLReduceOptions};
use crate::{error::GraphError, graph::DataType, operators::Operation};
use burn_tensor::backend::Backend;
use std::cmp::Ordering;
pub(super) mod movement;

fn err(message: impl Into<String>) -> GraphError {
    GraphError::BurnRuntimeFailed {
        reason: message.into(),
    }
}

pub(super) fn is_integer(dtype: DataType) -> bool {
    matches!(
        dtype,
        DataType::Int4
            | DataType::Uint4
            | DataType::Int8
            | DataType::Uint8
            | DataType::Int32
            | DataType::Uint32
            | DataType::Int64
            | DataType::Uint64
    )
}

#[derive(Clone, Copy, Debug)]
enum Number {
    Int(i128),
    Float(f32),
}
impl Number {
    fn integer(self) -> Result<i128, GraphError> {
        match self {
            Self::Int(v) => Ok(v),
            _ => Err(err("expected an exact integer value")),
        }
    }
    fn float(self) -> f32 {
        match self {
            Self::Int(v) => v as f32,
            Self::Float(v) => v,
        }
    }
    fn compare(self, other: Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => a.partial_cmp(&b),
            (Self::Float(a), Self::Float(b)) => a.partial_cmp(&b),
            _ => None,
        }
    }
    fn truth(self) -> bool {
        match self {
            Self::Int(v) => v != 0,
            Self::Float(v) => v != 0.0,
        }
    }
}

#[derive(Clone)]
struct Array {
    shape: Vec<usize>,
    values: Vec<Number>,
}
impl Array {
    fn load<B: Backend>(env: &TensorEnv<B>, id: u32) -> Result<Self, GraphError> {
        let value = env.value(id)?;
        let shape = value.shape().to_vec();
        let values = if let Some(data) = value.signed_data() {
            data.iter().map(|&v| Number::Int(i128::from(v))).collect()
        } else if let Some(data) = value.unsigned_data() {
            data.iter().map(|&v| Number::Int(i128::from(v))).collect()
        } else {
            value
                .float()?
                .to_host_array()?
                .data
                .into_iter()
                .map(Number::Float)
                .collect()
        };
        Ok(Self { shape, values })
    }
    fn at(&self, coords: &[usize]) -> Number {
        let offset = coords.len() - self.shape.len();
        let coords: Vec<_> = self
            .shape
            .iter()
            .enumerate()
            .map(|(i, &d)| if d == 1 { 0 } else { coords[offset + i] })
            .collect();
        self.values[linear_index(&coords, &self.shape)]
    }
    fn mapped(
        &self,
        shape: Vec<usize>,
        mut index: impl FnMut(&[usize]) -> Result<Number, GraphError>,
    ) -> Result<Self, GraphError> {
        let values = (0..shape.iter().product())
            .map(|i| index(&unravel(i, &shape)))
            .collect::<Result<_, _>>()?;
        Ok(Self { shape, values })
    }
}

fn domain(dtype: DataType) -> (u32, bool) {
    match dtype {
        DataType::Int4 => (4, true),
        DataType::Uint4 => (4, false),
        DataType::Int8 => (8, true),
        DataType::Uint8 => (8, false),
        DataType::Int32 => (32, true),
        DataType::Uint32 => (32, false),
        DataType::Int64 => (64, true),
        DataType::Uint64 => (64, false),
        _ => unreachable!("integer domain required"),
    }
}
fn bounds(dtype: DataType) -> (i128, i128) {
    let (bits, signed) = domain(dtype);
    if signed {
        (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
    } else {
        (0, (1i128 << bits) - 1)
    }
}
fn wrap(value: i128, dtype: DataType) -> i128 {
    let (bits, signed) = domain(dtype);
    let masked = value & ((1i128 << bits) - 1);
    if signed && masked & (1i128 << (bits - 1)) != 0 {
        masked - (1i128 << bits)
    } else {
        masked
    }
}
fn store<B: Backend>(
    env: &mut OperationOutputs<'_, B>,
    id: u32,
    array: Array,
) -> Result<(), GraphError> {
    let dtype = env.dtype(id)?;
    if is_integer(dtype) {
        let integers = array
            .values
            .into_iter()
            .map(|v| match v {
                Number::Int(v) => wrap(v, dtype),
                Number::Float(v) => {
                    let (lo, hi) = bounds(dtype);
                    (v as i128).clamp(lo, hi)
                }
            })
            .collect::<Vec<_>>();
        let value = if domain(dtype).1 {
            super::value::TensorValue::from_signed(
                dtype,
                array.shape,
                integers.into_iter().map(|v| v as i64).collect(),
            )?
        } else {
            super::value::TensorValue::from_unsigned(
                dtype,
                array.shape,
                integers.into_iter().map(|v| v as u64).collect(),
            )?
        };
        env.write(id, value)?;
    } else {
        let host = HostArray::new(
            array.shape,
            array.values.into_iter().map(Number::float).collect(),
        )?;
        env.write_host(id, host)?;
    }
    Ok(())
}
fn dimensions(dims: &[MLDimension], numel: usize) -> Result<Vec<usize>, GraphError> {
    super::execute::resolve_mldims(dims, numel)
}
fn index(value: Number, dim: usize) -> Result<usize, GraphError> {
    let value = value.integer()?;
    let value = if value < 0 {
        dim as i128 + value
    } else {
        value
    };
    // Match the host kernels' index clamping without narrowing large uint64 indices.
    Ok(value.clamp(0, dim.saturating_sub(1) as i128) as usize)
}

/// Execute exact integer arithmetic, predicates and quantization after explicit routing.
pub(super) fn execute<B: Backend>(
    env: &mut OperationOutputs<'_, B>,
    op: &Operation,
) -> Result<(), GraphError> {
    let first = *op
        .outputs()
        .first()
        .ok_or_else(|| err("operation has no output"))?;
    let dtype = env.dtype(first)?;
    let result = match op {
        Operation::Add { a, b, .. }
        | Operation::Sub { a, b, .. }
        | Operation::Mul { a, b, .. }
        | Operation::Div { a, b, .. }
        | Operation::Min { a, b, .. }
        | Operation::Max { a, b, .. }
        | Operation::Pow { a, b, .. }
        | Operation::Equal { a, b, .. }
        | Operation::NotEqual { a, b, .. }
        | Operation::Greater { a, b, .. }
        | Operation::GreaterOrEqual { a, b, .. }
        | Operation::Lesser { a, b, .. }
        | Operation::LesserOrEqual { a, b, .. }
        | Operation::LogicalAnd { a, b, .. }
        | Operation::LogicalOr { a, b, .. }
        | Operation::LogicalXor { a, b, .. } => {
            let a = Array::load(env, *a)?;
            let b = Array::load(env, *b)?;
            let shape = broadcast_shapes(&a.shape, &b.shape)?;
            a.mapped(shape, |coords| {
                let av = a.at(coords);
                let bv = b.at(coords);
                let cmp = av.compare(bv);
                let boolean = match op {
                    Operation::Equal { .. } => Some(cmp == Some(Ordering::Equal)),
                    Operation::NotEqual { .. } => Some(cmp != Some(Ordering::Equal)),
                    Operation::Greater { .. } => Some(cmp == Some(Ordering::Greater)),
                    Operation::GreaterOrEqual { .. } => {
                        Some(matches!(cmp, Some(Ordering::Greater | Ordering::Equal)))
                    }
                    Operation::Lesser { .. } => Some(cmp == Some(Ordering::Less)),
                    Operation::LesserOrEqual { .. } => {
                        Some(matches!(cmp, Some(Ordering::Less | Ordering::Equal)))
                    }
                    Operation::LogicalAnd { .. } => Some(av.truth() && bv.truth()),
                    Operation::LogicalOr { .. } => Some(av.truth() || bv.truth()),
                    Operation::LogicalXor { .. } => Some(av.truth() != bv.truth()),
                    _ => None,
                };
                if let Some(v) = boolean {
                    return Ok(Number::Int(i128::from(v)));
                }
                let a = av.integer()?;
                let b = bv.integer()?;
                let v = match op {
                    Operation::Add { .. } => a.wrapping_add(b),
                    Operation::Sub { .. } => a.wrapping_sub(b),
                    Operation::Mul { .. } => a.wrapping_mul(b),
                    Operation::Div { .. } => {
                        if b == 0 {
                            0
                        } else {
                            a / b
                        }
                    }
                    Operation::Min { .. } => a.min(b),
                    Operation::Max { .. } => a.max(b),
                    Operation::Pow { .. } => {
                        if b < 0 {
                            if a == 1 {
                                1
                            } else if a == -1 {
                                if b % 2 == 0 { 1 } else { -1 }
                            } else {
                                0
                            }
                        } else {
                            let mut x = a;
                            let mut n = b as u128;
                            let mut acc = 1i128;
                            while n != 0 {
                                if n & 1 != 0 {
                                    acc = wrap(acc.wrapping_mul(x), dtype);
                                }
                                x = wrap(x.wrapping_mul(x), dtype);
                                n >>= 1;
                            }
                            acc
                        }
                    }
                    _ => unreachable!(),
                };
                Ok(Number::Int(v))
            })?
        }
        Operation::Prelu { input, slope, .. } => {
            let a = Array::load(env, *input)?;
            let slope = Array::load(env, *slope)?;
            a.mapped(a.shape.clone(), |c| {
                let v = a.at(c).integer()?;
                Ok(Number::Int(if v < 0 {
                    v.wrapping_mul(slope.at(c).integer()?)
                } else {
                    v
                }))
            })?
        }
        Operation::IsNaN { input, .. } | Operation::IsInfinite { input, .. } => {
            let mut a = Array::load(env, *input)?;
            a.values = a
                .values
                .into_iter()
                .map(|v| {
                    Number::Int(i128::from(match v {
                        Number::Float(v) => {
                            if matches!(op, Operation::IsNaN { .. }) {
                                v.is_nan()
                            } else {
                                v.is_infinite()
                            }
                        }
                        Number::Int(_) => false,
                    }))
                })
                .collect();
            a
        }
        Operation::MaxPool2d { input, options, .. } => {
            integer_max_pool(&Array::load(env, *input)?, options)?
        }
        Operation::Resample2d { input, options, .. } => {
            integer_resample(&Array::load(env, *input)?, options)?
        }
        Operation::Abs { input, .. }
        | Operation::Neg { input, .. }
        | Operation::Relu { input, .. }
        | Operation::Sign { input, .. }
        | Operation::LogicalNot { input, .. }
        | Operation::Clamp { input, .. } => {
            let mut a = Array::load(env, *input)?;
            let clamp = if let Operation::Clamp { options, .. } = op {
                Some(clamp_bounds(options, dtype)?)
            } else {
                None
            };
            a.values = a
                .values
                .into_iter()
                .map(|v| {
                    let v = v.integer()?;
                    Ok(Number::Int(match op {
                        Operation::Abs { .. } => v.abs(),
                        Operation::Neg { .. } => -v,
                        Operation::Relu { .. } => v.max(0),
                        Operation::Sign { .. } => v.signum(),
                        Operation::LogicalNot { .. } => i128::from(v == 0),
                        Operation::Clamp { .. } => {
                            let (lo, hi) = clamp.unwrap();
                            v.max(lo).min(hi)
                        }
                        _ => unreachable!(),
                    }))
                })
                .collect::<Result<_, GraphError>>()?;
            a
        }
        Operation::CumulativeSum {
            input,
            axis,
            options,
            ..
        } => {
            let a = Array::load(env, *input)?;
            let mut out = a.clone();
            let opts = options.clone().unwrap_or_default();
            let axis = super::shape::axis(*axis, a.shape.len())?;
            let inner = a.shape[axis + 1..].iter().product::<usize>();
            let outer = a.shape[..axis].iter().product::<usize>();
            let n = a.shape[axis];
            for o in 0..outer {
                for i in 0..inner {
                    let mut sum = 0i128;
                    for step in 0..n {
                        let j = if opts.reversed { n - 1 - step } else { step };
                        let flat = o * n * inner + j * inner + i;
                        let v = a.values[flat].integer()?;
                        out.values[flat] = Number::Int(if opts.exclusive {
                            let previous = sum;
                            sum = wrap(sum.wrapping_add(v), dtype);
                            previous
                        } else {
                            sum = wrap(sum.wrapping_add(v), dtype);
                            sum
                        });
                    }
                }
            }
            out
        }
        Operation::ReduceSum { input, options, .. }
        | Operation::ReduceProduct { input, options, .. }
        | Operation::ReduceMin { input, options, .. }
        | Operation::ReduceMax { input, options, .. }
        | Operation::ReduceL1 { input, options, .. }
        | Operation::ReduceSumSquare { input, options, .. } => {
            integer_reduce(&Array::load(env, *input)?, options, op, dtype)?
        }
        Operation::ArgMin {
            input,
            axis,
            options,
            ..
        }
        | Operation::ArgMax {
            input,
            axis,
            options,
            ..
        } => {
            let a = Array::load(env, *input)?;
            let axis = super::shape::axis(*axis, a.shape.len())?;
            let keep = options.as_ref().is_some_and(|o| o.keep_dimensions);
            let shape = ReductionSpec::new(&a.shape, Some(&[axis as u32]), keep)?.output_shape;
            a.mapped(shape, |c| {
                let mut src = c.to_vec();
                if keep {
                    src[axis] = 0;
                } else {
                    src.insert(axis, 0);
                }
                let mut best = a.at(&src);
                let mut best_index = 0;
                for j in 1..a.shape[axis] {
                    src[axis] = j;
                    let v = a.at(&src);
                    let better = if matches!(op, Operation::ArgMin { .. }) {
                        v.compare(best) == Some(Ordering::Less)
                    } else {
                        v.compare(best) == Some(Ordering::Greater)
                    };
                    if better {
                        best = v;
                        best_index = j;
                    }
                }
                Ok(Number::Int(best_index as i128))
            })?
        }
        Operation::Shape { input, .. } => {
            let shape = env.shape(*input)?;
            Array {
                shape: vec![shape.len()],
                values: shape.iter().map(|&v| Number::Int(v as i128)).collect(),
            }
        }
        Operation::QuantizeLinear {
            input,
            scale,
            zero_point,
            ..
        }
        | Operation::DequantizeLinear {
            input,
            scale,
            zero_point,
            ..
        } => quantize(
            env,
            *input,
            *scale,
            *zero_point,
            dtype,
            matches!(op, Operation::QuantizeLinear { .. }),
        )?,
        _ => {
            // Integer data must never silently enter a floating-point kernel. Operations
            // with auxiliary integer operands need an explicit implementation here.
            return Err(err(format!(
                "{} has no typed integer implementation",
                op.op_type()
            )));
        }
    };
    store(env, first, result)?;
    Ok(())
}

fn json_integer(value: &serde_json::Value, dtype: DataType) -> Result<Option<i128>, GraphError> {
    if value.is_null() {
        return Ok(None);
    }
    let parsed = if let Some(v) = value.as_i64() {
        Some(i128::from(v))
    } else if let Some(v) = value.as_u64() {
        Some(i128::from(v))
    } else if let Some(s) = value.as_str() {
        s.parse::<i128>().ok()
    } else {
        None
    };
    let (lo, hi) = bounds(dtype);
    Ok(if let Some(v) = parsed {
        Some(v.clamp(lo, hi))
    } else {
        let v = value
            .as_f64()
            .or_else(|| value.as_str().and_then(|s| s.parse::<f64>().ok()))
            .ok_or_else(|| err("invalid integer option"))?;
        if v.is_nan() {
            None
        } else {
            Some((v as i128).clamp(lo, hi))
        }
    })
}
fn clamp_bounds(
    options: &Option<crate::operator_options::MLClampOptions>,
    dtype: DataType,
) -> Result<(i128, i128), GraphError> {
    let (lo, hi) = bounds(dtype);
    let Some(o) = options else {
        return Ok((lo, hi));
    };
    Ok((
        o.min_value
            .as_ref()
            .map(|v| json_integer(v, dtype))
            .transpose()?
            .flatten()
            .unwrap_or(lo),
        o.max_value
            .as_ref()
            .map(|v| json_integer(v, dtype))
            .transpose()?
            .flatten()
            .unwrap_or(hi),
    ))
}

fn integer_reduce(
    a: &Array,
    options: &Option<MLReduceOptions>,
    op: &Operation,
    dtype: DataType,
) -> Result<Array, GraphError> {
    let options = options.clone().unwrap_or_default();
    let spec = ReductionSpec::new(&a.shape, options.axes.as_deref(), options.keep_dimensions)?;
    let shape = spec.output_shape.clone();
    let n = shape.iter().product();
    let mut acc = vec![None::<i128>; n];
    for (flat, &v) in a.values.iter().enumerate() {
        let v = v.integer()?;
        let c = unravel(flat, &a.shape);
        let out = spec.output_coords(&c);
        let i = linear_index(&out, &shape);
        let x = match op {
            Operation::ReduceL1 { .. } => v.abs(),
            Operation::ReduceSumSquare { .. } => v.wrapping_mul(v),
            _ => v,
        };
        acc[i] = Some(if let Some(old) = acc[i] {
            match op {
                Operation::ReduceMin { .. } => old.min(x),
                Operation::ReduceMax { .. } => old.max(x),
                Operation::ReduceProduct { .. } => wrap(old.wrapping_mul(x), dtype),
                _ => wrap(old.wrapping_add(x), dtype),
            }
        } else {
            wrap(x, dtype)
        });
    }
    let values = acc
        .into_iter()
        .map(|v| Number::Int(v.unwrap_or(0)))
        .collect();
    Ok(Array { shape, values })
}

fn integer_max_pool(
    a: &Array,
    options: &Option<crate::operator_options::MLPool2dOptions>,
) -> Result<Array, GraphError> {
    let opts = options.clone().unwrap_or_default();
    let nchw =
        parse_option::<MLInputOperandLayout>(&opts.layout, "layout")? == MLInputOperandLayout::Nchw;
    let (y, x) = if nchw { (2, 3) } else { (1, 2) };
    let shape = crate::shape_inference::infer_pool2d_shape(
        &a.shape.iter().map(|&d| d as u32).collect::<Vec<_>>(),
        &opts,
    )?
    .into_iter()
    .map(|d| d as usize)
    .collect();
    let window = opts
        .window_dimensions
        .clone()
        .unwrap_or_else(|| vec![a.shape[y] as u32, a.shape[x] as u32]);
    a.mapped(shape, |c| {
        let mut best = None;
        for ky in 0..window[0] {
            for kx in 0..window[1] {
                let iy = c[y] as i128 * i128::from(opts.strides.first().copied().unwrap_or(1))
                    + i128::from(ky) * i128::from(opts.dilations.first().copied().unwrap_or(1))
                    - i128::from(opts.padding.first().copied().unwrap_or(0));
                let ix = c[x] as i128 * i128::from(opts.strides.get(1).copied().unwrap_or(1))
                    + i128::from(kx) * i128::from(opts.dilations.get(1).copied().unwrap_or(1))
                    - i128::from(opts.padding.get(2).copied().unwrap_or(0));
                if iy >= 0 && iy < a.shape[y] as i128 && ix >= 0 && ix < a.shape[x] as i128 {
                    let mut src = c.to_vec();
                    src[y] = iy as usize;
                    src[x] = ix as usize;
                    let v = a.at(&src).integer()?;
                    best = Some(best.map_or(v, |b: i128| b.max(v)));
                }
            }
        }
        Ok(Number::Int(best.unwrap_or(0)))
    })
}

fn integer_resample(
    a: &Array,
    options: &Option<crate::operator_options::MLResample2dOptions>,
) -> Result<Array, GraphError> {
    let opts = options.clone().unwrap_or_default();
    let mode = parse_option::<MLInterpolationMode>(&opts.mode, "interpolation mode")?;
    let axes = if opts.axes.is_empty() {
        vec![2, 3]
    } else {
        opts.axes.iter().map(|&v| v as usize).collect()
    };
    let mut shape = a.shape.clone();
    for (i, &axis) in axes.iter().enumerate() {
        shape[axis] = opts
            .sizes
            .as_ref()
            .map(|s| s[i] as usize)
            .unwrap_or_else(|| {
                (a.shape[axis] as f64 * f64::from(opts.scales.get(i).copied().unwrap_or(1.0)))
                    .floor() as usize
            });
    }
    let out_shape = shape.clone();
    a.mapped(shape, |c| {
        let coord = |axis: usize| {
            ((c[axis] as f64 + 0.5) * a.shape[axis] as f64 / out_shape[axis] as f64 - 0.5)
                .clamp(0.0, (a.shape[axis] - 1) as f64)
        };
        if mode == MLInterpolationMode::Linear {
            // Interpolation is intrinsically fractional; integer samples are converted
            // only after exact selection, then the result is converted at the output.
            let sy = coord(axes[0]);
            let sx = coord(axes[1]);
            let wy = sy.fract();
            let wx = sx.fract();
            let mut sum = 0.0;
            for (y, w_y) in [(sy.floor() as usize, 1.0 - wy), (sy.ceil() as usize, wy)] {
                for (x, w_x) in [(sx.floor() as usize, 1.0 - wx), (sx.ceil() as usize, wx)] {
                    let mut src = c.to_vec();
                    src[axes[0]] = y;
                    src[axes[1]] = x;
                    sum += a.at(&src).integer()? as f64 * w_y * w_x;
                }
            }
            Ok(Number::Float(sum as f32))
        } else {
            let mut src = c.to_vec();
            for &axis in &axes {
                src[axis] = (coord(axis) - 0.5).ceil().max(0.0) as usize;
            }
            Ok(a.at(&src))
        }
    })
}

fn quantize<B: Backend>(
    env: &TensorEnv<B>,
    input: u32,
    scale: u32,
    zero_point: Option<u32>,
    dtype: DataType,
    quantize: bool,
) -> Result<Array, GraphError> {
    let a = Array::load(env, input)?;
    let scale = Array::load(env, scale)?;
    let zp = zero_point.map(|id| Array::load(env, id)).transpose()?;
    let param = |a: &Array, c: &[usize]| {
        let offset = c.len() - a.shape.len();
        let coords = a
            .shape
            .iter()
            .enumerate()
            .map(|(i, &d)| c[offset + i] / (env.shape(input).unwrap()[offset + i] / d))
            .collect::<Vec<_>>();
        a.at(&coords)
    };
    a.mapped(a.shape.clone(), |c| {
        let s = f64::from(param(&scale, c).float());
        let z = zp
            .as_ref()
            .map(|a| param(a, c).integer())
            .transpose()?
            .unwrap_or(0);
        if quantize {
            let x = f64::from(a.at(c).float());
            let rounded = if s == 0.0 {
                0
            } else {
                (x / s).round_ties_even() as i128
            };
            let (lo, hi) = bounds(dtype);
            Ok(Number::Int(rounded.saturating_add(z).clamp(lo, hi)))
        } else {
            let delta = a.at(c).integer()? - z;
            Ok(Number::Float((delta as f64 * s) as f32))
        }
    })
}

#[cfg(test)]
mod tests;

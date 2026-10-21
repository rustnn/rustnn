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

//! Datatype-independent tensor selection and index mapping.
use super::super::options::float_number;
use super::super::shape::{SliceSpec, axis, permutation};
use super::{Array, Number, dimensions, err, index, is_integer, json_integer, store};
use super::{Backend, GraphError, Operation, broadcast_shapes, linear_index, unravel};
use crate::operator_enums::MLPaddingMode;

pub(crate) fn handles(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Cast { .. }
            | Operation::Identity { .. }
            | Operation::Reshape { .. }
            | Operation::Expand { .. }
            | Operation::Squeeze { .. }
            | Operation::Unsqueeze { .. }
            | Operation::Transpose { .. }
            | Operation::Slice { .. }
            | Operation::Concat { .. }
            | Operation::Split { .. }
            | Operation::Tile { .. }
            | Operation::Reverse { .. }
            | Operation::Where { .. }
            | Operation::Triangular { .. }
            | Operation::Pad { .. }
            | Operation::Gather { .. }
            | Operation::GatherElements { .. }
            | Operation::GatherND { .. }
            | Operation::ScatterElements { .. }
            | Operation::ScatterND { .. }
    )
}

pub(crate) fn execute<B: Backend>(
    env: &mut super::OperationOutputs<'_, B>,
    op: &Operation,
) -> Result<(), GraphError> {
    let first = *op
        .outputs()
        .first()
        .ok_or_else(|| err("operation has no output"))?;
    let dtype = env.dtype(first)?;
    let result = match op {
        Operation::Cast { input, .. } | Operation::Identity { input, .. } => {
            Array::load(env, *input)?
        }
        Operation::Reshape {
            input, new_shape, ..
        }
        | Operation::Expand {
            input, new_shape, ..
        } => {
            let a = Array::load(env, *input)?;
            let shape = dimensions(new_shape, a.values.len())?;
            if matches!(op, Operation::Reshape { .. }) {
                Array {
                    shape,
                    values: a.values,
                }
            } else {
                broadcast_shapes(&a.shape, &shape)?;
                a.mapped(shape, |c| Ok(a.at(c)))?
            }
        }
        Operation::Squeeze { input, options, .. } => {
            let mut a = Array::load(env, *input)?;
            let axes = options.as_ref().map(|o| o.axes.clone()).unwrap_or_default();
            super::super::shape::axes(&axes, a.shape.len())?;
            a.shape = a
                .shape
                .iter()
                .enumerate()
                .filter(|(i, d)| !(**d == 1 && (axes.is_empty() || axes.contains(&(*i as u32)))))
                .map(|(_, d)| *d)
                .collect();
            a
        }
        Operation::Unsqueeze { input, options, .. } => {
            let mut a = Array::load(env, *input)?;
            let mut axes = options.as_ref().map(|o| o.axes.clone()).unwrap_or_default();
            super::super::shape::axes(&axes, a.shape.len() + axes.len())?;
            axes.sort_unstable();
            for axis in axes {
                a.shape.insert(axis as usize, 1);
            }
            a
        }
        Operation::Transpose { input, options, .. } => {
            let a = Array::load(env, *input)?;
            let perm = permutation(
                options
                    .as_ref()
                    .map(|o| o.permutation.as_slice())
                    .unwrap_or_default(),
                a.shape.len(),
            )?;
            let shape = perm.iter().map(|&p| a.shape[p]).collect();
            a.mapped(shape, |c| {
                let mut src = vec![0; c.len()];
                for (i, &p) in perm.iter().enumerate() {
                    src[p] = c[i];
                }
                Ok(a.at(&src))
            })?
        }
        Operation::Slice {
            input,
            starts,
            sizes,
            options,
            ..
        } => {
            let a = Array::load(env, *input)?;
            let span = dimensions(sizes, a.values.len())?;
            let strides = options
                .as_ref()
                .map(|o| o.strides.clone())
                .unwrap_or_default();
            let spec = SliceSpec::new(&a.shape, starts, &span, &strides)?;
            a.mapped(spec.output_shape.clone(), |c| {
                Ok(a.at(&spec.input_coords(c)))
            })?
        }
        Operation::Concat { inputs, axis, .. } => {
            let arrays = inputs
                .iter()
                .map(|&id| Array::load(env, id))
                .collect::<Result<Vec<_>, _>>()?;
            let a = &arrays[0];
            let axis = super::super::shape::axis(*axis, a.shape.len())?;
            let mut shape = a.shape.clone();
            shape[axis] = arrays.iter().map(|a| a.shape[axis]).sum();
            a.mapped(shape, |c| {
                let mut c = c.to_vec();
                for a in &arrays {
                    if c[axis] < a.shape[axis] {
                        return Ok(a.at(&c));
                    }
                    c[axis] -= a.shape[axis];
                }
                Err(err("concat index out of range"))
            })?
        }
        Operation::Split {
            input,
            splits,
            split_equal_parts,
            options,
            ..
        } => {
            let a = Array::load(env, *input)?;
            let axis = axis(options.as_ref().map(|o| o.axis).unwrap_or(0), a.shape.len())?;
            let sizes = if let Some(parts) = split_equal_parts {
                vec![a.shape[axis] / *parts as usize; *parts as usize]
            } else {
                splits.iter().map(|v| *v as usize).collect()
            };
            let mut start = 0;
            for (&id, size) in op.outputs().iter().zip(sizes) {
                let mut shape = a.shape.clone();
                shape[axis] = size;
                let out = a.mapped(shape, |c| {
                    let mut c = c.to_vec();
                    c[axis] += start;
                    Ok(a.at(&c))
                })?;
                store(env, id, out)?;
                start += size;
            }
            return Ok(());
        }
        Operation::Tile {
            input, repetitions, ..
        } => {
            let a = Array::load(env, *input)?;
            let shape = a
                .shape
                .iter()
                .zip(repetitions)
                .map(|(&d, &n)| d * n as usize)
                .collect();
            a.mapped(shape, |c| {
                Ok(a.at(&c
                    .iter()
                    .zip(&a.shape)
                    .map(|(&c, &d)| c % d)
                    .collect::<Vec<_>>()))
            })?
        }
        Operation::Reverse { input, options, .. } => {
            let a = Array::load(env, *input)?;
            let axes = options
                .as_ref()
                .and_then(|o| o.axes.clone())
                .unwrap_or_else(|| (0..a.shape.len() as u32).collect());
            a.mapped(a.shape.clone(), |c| {
                let mut c = c.to_vec();
                for &axis in &axes {
                    let i = axis as usize;
                    c[i] = a.shape[i] - 1 - c[i];
                }
                Ok(a.at(&c))
            })?
        }
        Operation::Where {
            condition,
            true_value,
            false_value,
            ..
        } => {
            let c = Array::load(env, *condition)?;
            let t = Array::load(env, *true_value)?;
            let f = Array::load(env, *false_value)?;
            let shape = broadcast_shapes(&broadcast_shapes(&c.shape, &t.shape)?, &f.shape)?;
            t.mapped(shape, |coords| {
                Ok(if c.at(coords).truth() {
                    t.at(coords)
                } else {
                    f.at(coords)
                })
            })?
        }
        Operation::Triangular { input, options, .. } => {
            let a = Array::load(env, *input)?;
            let opts = options.clone().unwrap_or_default();
            a.mapped(a.shape.clone(), |c| {
                let offset = c[c.len() - 1] as i64 - c[c.len() - 2] as i64;
                Ok(
                    if if opts.upper.unwrap_or(true) {
                        offset >= i64::from(opts.diagonal)
                    } else {
                        offset <= i64::from(opts.diagonal)
                    } {
                        a.at(c)
                    } else {
                        Number::Int(0)
                    },
                )
            })?
        }
        Operation::Pad {
            input,
            beginning_padding,
            ending_padding,
            options,
            ..
        } => {
            let a = Array::load(env, *input)?;
            let opts = options.clone().unwrap_or_default();
            let shape = a
                .shape
                .iter()
                .enumerate()
                .map(|(i, &d)| d + beginning_padding[i] as usize + ending_padding[i] as usize)
                .collect();
            let fill = if is_integer(dtype) {
                Number::Int(
                    opts.value
                        .as_ref()
                        .map(|v| json_integer(v, dtype))
                        .transpose()?
                        .flatten()
                        .unwrap_or(0),
                )
            } else {
                Number::Float(opts.value.as_ref().and_then(float_number).unwrap_or(0.0))
            };
            a.mapped(shape, |c| {
                let mut src = Vec::new();
                for (i, &v) in c.iter().enumerate() {
                    let x = v as i128 - i128::from(beginning_padding[i]);
                    let n = a.shape[i] as i128;
                    let x = match opts.mode {
                        MLPaddingMode::Edge => x.clamp(0, n - 1),
                        MLPaddingMode::Reflection => {
                            if n <= 1 {
                                0
                            } else {
                                let p = 2 * (n - 1);
                                let r = x.rem_euclid(p);
                                if r >= n { p - r } else { r }
                            }
                        }
                        MLPaddingMode::Constant => {
                            if x < 0 || x >= n {
                                return Ok(fill);
                            }
                            x
                        }
                    };
                    src.push(x as usize);
                }
                Ok(a.at(&src))
            })?
        }
        Operation::Gather {
            input,
            indices,
            options,
            batch_dimensions,
            ..
        } => {
            let a = Array::load(env, *input)?;
            let ix = Array::load(env, *indices)?;
            let axis = axis(options.as_ref().map(|o| o.axis).unwrap_or(0), a.shape.len())?;
            let batch = batch_dimensions.unwrap_or(0) as usize;
            let mut shape = a.shape[..axis].to_vec();
            shape.extend_from_slice(&ix.shape[batch..]);
            shape.extend_from_slice(&a.shape[axis + 1..]);
            a.mapped(shape, |c| {
                let mut ic = c[..batch].to_vec();
                ic.extend_from_slice(&c[axis..axis + ix.shape.len() - batch]);
                let mut src = c[..axis].to_vec();
                src.push(index(ix.at(&ic), a.shape[axis])?);
                src.extend_from_slice(&c[axis + ix.shape.len() - batch..]);
                Ok(a.at(&src))
            })?
        }
        Operation::GatherElements {
            input,
            indices,
            options,
            ..
        } => {
            let a = Array::load(env, *input)?;
            let ix = Array::load(env, *indices)?;
            let axis = axis(options.as_ref().map(|o| o.axis).unwrap_or(0), a.shape.len())?;
            a.mapped(ix.shape.clone(), |c| {
                let mut src = c.to_vec();
                src[axis] = index(ix.at(c), a.shape[axis])?;
                Ok(a.at(&src))
            })?
        }
        Operation::GatherND { input, indices, .. } => {
            let a = Array::load(env, *input)?;
            let ix = Array::load(env, *indices)?;
            let k = *ix.shape.last().unwrap();
            let prefix = ix.shape.len() - 1;
            let mut shape = ix.shape[..prefix].to_vec();
            shape.extend_from_slice(&a.shape[k..]);
            a.mapped(shape, |c| {
                let mut src = Vec::new();
                for j in 0..k {
                    let mut ic = c[..prefix].to_vec();
                    ic.push(j);
                    src.push(index(ix.at(&ic), a.shape[j])?);
                }
                src.extend_from_slice(&c[prefix..]);
                Ok(a.at(&src))
            })?
        }
        Operation::ScatterElements {
            input,
            indices,
            updates,
            options,
            ..
        } => {
            let mut a = Array::load(env, *input)?;
            let ix = Array::load(env, *indices)?;
            let updates = Array::load(env, *updates)?;
            let axis = axis(options.as_ref().map(|o| o.axis).unwrap_or(0), a.shape.len())?;
            for flat in 0..ix.values.len() {
                let mut c = unravel(flat, &ix.shape);
                c[axis] = index(ix.values[flat], a.shape[axis])?;
                let dst = linear_index(&c, &a.shape);
                a.values[dst] = updates.values[flat];
            }
            a
        }
        Operation::ScatterND {
            input,
            indices,
            updates,
            ..
        } => {
            let mut a = Array::load(env, *input)?;
            let ix = Array::load(env, *indices)?;
            let updates = Array::load(env, *updates)?;
            let k = *ix.shape.last().unwrap();
            let prefix = &ix.shape[..ix.shape.len() - 1];
            let tail = &a.shape[k..];
            let tail_len = tail.iter().product::<usize>();
            for flat in 0..prefix.iter().product::<usize>() {
                let mut src = Vec::new();
                for j in 0..k {
                    src.push(index(ix.values[flat * k + j], a.shape[j])?);
                }
                for j in 0..tail_len {
                    let mut c = src.clone();
                    c.extend(unravel(j, tail));
                    let dst = linear_index(&c, &a.shape);
                    a.values[dst] = updates.values[flat * tail_len + j];
                }
            }
            a
        }
        _ => {
            return Err(err(
                "operation was incorrectly routed to movement execution",
            ));
        }
    };
    store(env, first, result)?;
    Ok(())
}

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

//! Checked, datatype-independent shape and coordinate semantics.
use crate::error::GraphError;
fn burn_err(reason: impl Into<String>) -> GraphError {
    GraphError::BurnRuntimeFailed {
        reason: reason.into(),
    }
}

pub(super) fn axis(axis: u32, rank: usize) -> Result<usize, GraphError> {
    axis_index(axis as usize, rank)
}

pub(super) fn axis_index(axis: usize, rank: usize) -> Result<usize, GraphError> {
    if axis >= rank {
        return Err(burn_err(format!(
            "axis {axis} is out of bounds for rank {rank}"
        )));
    }
    Ok(axis)
}
pub(super) fn axes(axes: &[u32], rank: usize) -> Result<Vec<usize>, GraphError> {
    let mut checked = Vec::with_capacity(axes.len());
    for &a in axes {
        let a = axis(a, rank)?;
        if checked.contains(&a) {
            return Err(burn_err(format!("duplicate axis {a}")));
        }
        checked.push(a);
    }
    Ok(checked)
}
pub(super) fn permutation(perm: &[u32], rank: usize) -> Result<Vec<usize>, GraphError> {
    if perm.is_empty() {
        return Ok((0..rank).rev().collect());
    }
    if perm.len() != rank {
        return Err(burn_err("permutation length must match input rank"));
    }
    axes(perm, rank)
}

/// A slice size is the window extent before applying its stride.
pub(super) struct SliceSpec {
    starts: Vec<usize>,
    strides: Vec<usize>,
    pub output_shape: Vec<usize>,
}
impl SliceSpec {
    pub fn new(
        input: &[usize],
        starts: &[u32],
        sizes: &[usize],
        strides: &[u32],
    ) -> Result<Self, GraphError> {
        if starts.len() != input.len()
            || sizes.len() != input.len()
            || (!strides.is_empty() && strides.len() != input.len())
        {
            return Err(burn_err("slice arguments must match the input rank"));
        }
        // Guard addition before calling the shared u32 inference helper.
        let input_dims = dimensions_u32(input)?;
        let size_dims = dimensions_u32(sizes)?;
        for (i, (&start, &size)) in starts.iter().zip(sizes).enumerate() {
            if (start as usize)
                .checked_add(size)
                .is_none_or(|end| end > input[i])
            {
                return Err(burn_err(format!("invalid slice window at axis {i}")));
            }
        }
        let output_shape = crate::shape_inference::infer_slice_shape(
            &input_dims,
            starts,
            &size_dims,
            Some(strides),
        )
        .map_err(|e| burn_err(e.to_string()))?
        .into_iter()
        .map(|d| d as usize)
        .collect();
        let starts = starts.iter().map(|&v| v as usize).collect();
        let strides = (0..input.len())
            .map(|i| strides.get(i).copied().unwrap_or(1) as usize)
            .collect();
        Ok(Self {
            starts,
            strides,
            output_shape,
        })
    }
    pub fn input_coords(&self, output: &[usize]) -> Vec<usize> {
        output
            .iter()
            .enumerate()
            .map(|(i, &v)| self.starts[i] + v * self.strides[i])
            .collect()
    }
}

pub(super) struct ReductionSpec {
    pub axes: Vec<usize>,
    pub output_shape: Vec<usize>,
    keep_dims: bool,
}
impl ReductionSpec {
    pub fn new(
        input: &[usize],
        requested: Option<&[u32]>,
        keep_dims: bool,
    ) -> Result<Self, GraphError> {
        let input_dims = dimensions_u32(input)?;
        let requested =
            requested.map_or_else(|| (0..input.len() as u32).collect(), <[u32]>::to_vec);
        let output_shape = crate::shape_inference::infer_reduce_shape(
            &input_dims,
            &crate::shape_inference::ReduceOptions {
                axes: requested.clone(),
                keep_dimensions: keep_dims,
            },
        )
        .map_err(|e| burn_err(e.to_string()))?
        .into_iter()
        .map(|d| d as usize)
        .collect();
        let axes = requested.into_iter().map(|a| a as usize).collect();
        Ok(Self {
            axes,
            output_shape,
            keep_dims,
        })
    }
    pub fn output_coords(&self, input: &[usize]) -> Vec<usize> {
        input
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                if self.axes.contains(&i) {
                    self.keep_dims.then_some(0)
                } else {
                    Some(v)
                }
            })
            .collect()
    }
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
        out[rank - 1 - i] = if da == 1 { db } else { da };
    }
    Ok(out)
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

pub fn unravel(flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    let mut rem = flat;
    for i in (0..shape.len()).rev() {
        coords[i] = rem % shape[i];
        rem /= shape[i];
    }
    coords
}

/// Dynamic binding dimensions (-1) accept any resolved extent, but never change rank.
pub(super) fn matches_binding(actual: &[usize], declared: &[i64]) -> bool {
    actual.len() == declared.len()
        && actual
            .iter()
            .zip(declared)
            .all(|(&a, &d)| d == -1 || (d >= 0 && a as u64 == d as u64))
}

fn dimensions_u32(shape: &[usize]) -> Result<Vec<u32>, GraphError> {
    checked_numel(shape)?;
    Ok(shape.iter().map(|&d| d as u32).collect())
}

/// Checked element count, including scalars and tensors with zero extents.
pub(super) fn checked_numel(shape: &[usize]) -> Result<usize, GraphError> {
    if shape.len() > 8 || shape.iter().any(|&d| d > u32::MAX as usize) {
        return Err(burn_err(
            "Burn tensors require rank <= 8 and dimensions <= u32::MAX",
        ));
    }
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| burn_err("tensor element count overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn slice_window_and_mapping_share_stride_semantics() {
        let spec = SliceSpec::new(&[3, 7], &[1, 1], &[2, 6], &[1, 4]).unwrap();
        assert_eq!(spec.output_shape, [2, 2]);
        assert_eq!(spec.input_coords(&[1, 1]), [2, 5]);
        assert!(SliceSpec::new(&[7], &[1], &[7], &[]).is_err());
        assert!(SliceSpec::new(&[7], &[1], &[6], &[0]).is_err());
        assert!(SliceSpec::new(&[u32::MAX as usize], &[u32::MAX - 1], &[3], &[]).is_err());
    }
    #[test]
    fn reductions_distinguish_omitted_and_empty_axes_and_validate_axes() {
        let all = ReductionSpec::new(&[2, 3], None, false).unwrap();
        assert!(all.output_shape.is_empty());
        assert!(all.output_coords(&[1, 2]).is_empty());
        let none = ReductionSpec::new(&[2, 3], Some(&[]), false).unwrap();
        assert_eq!(none.output_shape, [2, 3]);
        assert_eq!(none.output_coords(&[1, 2]), [1, 2]);
        let keep = ReductionSpec::new(&[2, 3], Some(&[1]), true).unwrap();
        assert_eq!(keep.output_shape, [2, 1]);
        assert_eq!(keep.output_coords(&[1, 2]), [1, 0]);
        assert!(ReductionSpec::new(&[2, 3], Some(&[2]), false).is_err());
        assert!(ReductionSpec::new(&[2, 3], Some(&[1, 1]), false).is_err());
        assert!(permutation(&[0, 0], 2).is_err());
        assert!(permutation(&[0, 2], 2).is_err());
    }
}

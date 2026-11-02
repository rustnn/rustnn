// SPDX-FileCopyrightText: 2026 NVIDIA
//
// SPDX-License-Identifier: Apache-2.0

//! Tensor shape algebra independent of any graph format or operator naming scheme.

use crate::{Context, Error, Expression, Result, expression::Shape};

/// Position of the channel dimension in a rank-four image tensor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InputLayout {
    /// Batch, channels, height, width.
    #[default]
    Nchw,
    /// Batch, height, width, channels.
    Nhwc,
}

/// Dimension order of a forward convolution filter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Conv2dFilterLayout {
    /// Output channels, input channels, height, width.
    #[default]
    Oihw,
    /// Height, width, input channels, output channels.
    Hwio,
    /// Output channels, height, width, input channels.
    Ohwi,
    /// Input channels, height, width, output channels.
    Ihwo,
}

/// Dimension order of a transposed convolution filter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConvTranspose2dFilterLayout {
    /// Input channels, output channels, height, width.
    #[default]
    Iohw,
    /// Height, width, output channels, input channels.
    Hwoi,
    /// Output channels, height, width, input channels.
    Ohwi,
}

/// Shape parameters for a forward two-dimensional convolution.
#[derive(Clone, Copy, Debug)]
pub struct Conv2dOptions {
    pub input_layout: InputLayout,
    pub filter_layout: Conv2dFilterLayout,
    pub groups: u32,
    /// Beginning height, ending height, beginning width, ending width.
    pub padding: [u32; 4],
    /// Height and width strides.
    pub strides: [u32; 2],
    /// Height and width dilations.
    pub dilations: [u32; 2],
}

impl Default for Conv2dOptions {
    fn default() -> Self {
        Self {
            input_layout: InputLayout::Nchw,
            filter_layout: Conv2dFilterLayout::Oihw,
            groups: 1,
            padding: [0; 4],
            strides: [1; 2],
            dilations: [1; 2],
        }
    }
}

/// Shape parameters for a transposed two-dimensional convolution.
#[derive(Clone, Copy, Debug)]
pub struct ConvTranspose2dOptions {
    pub input_layout: InputLayout,
    pub filter_layout: ConvTranspose2dFilterLayout,
    pub groups: u32,
    /// Beginning height, ending height, beginning width, ending width.
    pub padding: [u32; 4],
    /// Height and width strides.
    pub strides: [u32; 2],
    /// Height and width dilations.
    pub dilations: [u32; 2],
    /// Height and width output padding.
    pub output_padding: [u32; 2],
    /// Explicit height and width of the result, when given.
    pub output_sizes: Option<[u32; 2]>,
}

impl Default for ConvTranspose2dOptions {
    fn default() -> Self {
        Self {
            input_layout: InputLayout::Nchw,
            filter_layout: ConvTranspose2dFilterLayout::Iohw,
            groups: 1,
            padding: [0; 4],
            strides: [1; 2],
            dilations: [1; 2],
            output_padding: [0; 2],
            output_sizes: None,
        }
    }
}

/// Infer the output shape of a two-dimensional convolution and constrain its input channels.
pub fn conv2d(
    context: &mut Context,
    input: &Shape,
    filter: &Shape,
    options: &Conv2dOptions,
) -> Result<Shape> {
    let input: std::vec::Vec<_> = input.iter().cloned().collect();
    let filter: std::vec::Vec<_> = filter.iter().cloned().collect();
    if input.len() != 4 || filter.len() != 4 {
        return Err(Error::ConvolutionRank {
            input: input.len(),
            filter: filter.len(),
        });
    }
    if options.groups == 0 {
        return Err(Error::ZeroConvolutionGroups);
    }
    let (channel, height, width) = match options.input_layout {
        InputLayout::Nchw => (1, 2, 3),
        InputLayout::Nhwc => (3, 1, 2),
    };
    let (output_channels, filter_channels, kernel_h, kernel_w) = match options.filter_layout {
        Conv2dFilterLayout::Oihw => (0, 1, 2, 3),
        Conv2dFilterLayout::Hwio => (3, 2, 0, 1),
        Conv2dFilterLayout::Ohwi => (0, 3, 1, 2),
        Conv2dFilterLayout::Ihwo => (3, 0, 1, 2),
    };
    context.equal_dim(
        &input[channel],
        &(filter[filter_channels].clone() * Expression::new_const(options.groups as i64)),
    )?;
    let mut output = input.clone();
    output[channel] = filter[output_channels].clone();
    output[height] = window_output(
        input[height].clone(),
        filter[kernel_h].clone(),
        [options.padding[0], options.padding[1]],
        options.strides[0],
        options.dilations[0],
        false,
    )?;
    output[width] = window_output(
        input[width].clone(),
        filter[kernel_w].clone(),
        [options.padding[2], options.padding[3]],
        options.strides[1],
        options.dilations[1],
        false,
    )?;
    Ok(Shape::from_vec(output.into()))
}

/// Infer the output shape of a transposed convolution and constrain its input channels.
pub fn conv_transpose2d(
    context: &mut Context,
    input: &Shape,
    filter: &Shape,
    options: &ConvTranspose2dOptions,
) -> Result<Shape> {
    let input: std::vec::Vec<_> = input.iter().cloned().collect();
    let filter: std::vec::Vec<_> = filter.iter().cloned().collect();
    if input.len() != 4 || filter.len() != 4 {
        return Err(Error::ConvolutionRank {
            input: input.len(),
            filter: filter.len(),
        });
    }
    if options.groups == 0 {
        return Err(Error::ZeroConvolutionGroups);
    }
    let (channel, height, width) = match options.input_layout {
        InputLayout::Nchw => (1, 2, 3),
        InputLayout::Nhwc => (3, 1, 2),
    };
    let (filter_channels, output_channels, kernel_h, kernel_w) = match options.filter_layout {
        ConvTranspose2dFilterLayout::Iohw => (0, 1, 2, 3),
        ConvTranspose2dFilterLayout::Hwoi => (3, 2, 0, 1),
        ConvTranspose2dFilterLayout::Ohwi => (3, 0, 1, 2),
    };
    context.equal_dim(&input[channel], &filter[filter_channels])?;
    let mut output = input.clone();
    output[channel] =
        filter[output_channels].clone() * Expression::new_const(options.groups as i64);
    if let Some(sizes) = options.output_sizes {
        output[height] = Expression::new_const(sizes[0] as i64);
        output[width] = Expression::new_const(sizes[1] as i64);
    } else {
        output[height] = transposed_window_output(
            input[height].clone(),
            filter[kernel_h].clone(),
            [options.padding[0], options.padding[1]],
            options.strides[0],
            options.dilations[0],
            options.output_padding[0],
        )?;
        output[width] = transposed_window_output(
            input[width].clone(),
            filter[kernel_w].clone(),
            [options.padding[2], options.padding[3]],
            options.strides[1],
            options.dilations[1],
            options.output_padding[1],
        )?;
    }
    Ok(Shape::from_vec(output.into()))
}

pub fn broadcast(context: &mut Context, lhs: &Shape, rhs: &Shape) -> Result<Shape> {
    let rank = lhs.rank_usize().max(rhs.rank_usize());
    let a: std::vec::Vec<_> = lhs.iter().cloned().collect();
    let b: std::vec::Vec<_> = rhs.iter().cloned().collect();
    let mut result = std::vec::Vec::with_capacity(rank);
    for i in 0..rank {
        let left = a
            .get(a.len().wrapping_sub(i + 1))
            .cloned()
            .unwrap_or_else(|| Expression::new_const(1));
        let right = b
            .get(b.len().wrapping_sub(i + 1))
            .cloned()
            .unwrap_or_else(|| Expression::new_const(1));
        let dimension = match (left.as_const(), right.as_const()) {
            (Some(x), Some(y)) if x != y && x != 1 && y != 1 => {
                return Err(Error::IncompatibleBroadcastDimensions { left: x, right: y });
            }
            (Some(1), _) => right,
            (_, Some(1)) => left,
            (Some(x), Some(_)) => Expression::new_const(x),
            _ => context.broadcast_dim(left, right, "incompatible broadcast dimensions")?,
        };
        result.push(dimension);
    }
    result.reverse();
    Ok(Shape::from_vec(result.into_iter().collect()))
}

pub fn concat(context: &mut Context, shapes: &[Shape], axis: usize) -> Result<Shape> {
    let first = shapes.first().ok_or(Error::EmptyConcat)?;
    let rank = first.rank_usize();
    if axis >= rank {
        return Err(Error::InvalidConcatAxis { axis, rank });
    }
    if let Some(shape) = shapes.iter().find(|s| s.rank_usize() != rank) {
        return Err(Error::ConcatRankMismatch {
            expected: rank,
            actual: shape.rank_usize(),
        });
    }
    let mut result: std::vec::Vec<_> = first.iter().cloned().collect();
    for shape in &shapes[1..] {
        for (i, dim) in shape.iter().enumerate() {
            if i == axis {
                result[i] = result[i].clone() + dim.clone();
            } else {
                context.equal_dim(&result[i], dim)?;
            }
        }
    }
    Ok(Shape::from_vec(result.into_iter().collect()))
}

pub fn transpose(shape: &Shape, permutation: &[usize]) -> Result<Shape> {
    let dims: std::vec::Vec<_> = shape.iter().cloned().collect();
    if permutation.len() != dims.len() {
        return Err(Error::TransposePermutationLength {
            expected: dims.len(),
            actual: permutation.len(),
        });
    }
    if let Some(&axis) = permutation.iter().find(|&&i| i >= dims.len()) {
        return Err(Error::TransposeAxisOutOfRange {
            axis,
            rank: dims.len(),
        });
    }
    let mut seen = vec![false; dims.len()];
    for &i in permutation {
        if seen[i] {
            return Err(Error::DuplicateTransposeAxis { axis: i });
        }
        seen[i] = true;
    }
    Ok(Shape::from_vec(
        permutation.iter().map(|&i| dims[i].clone()).collect(),
    ))
}

pub fn reduce(shape: &Shape, axes: &[usize], keep_dimensions: bool) -> Result<Shape> {
    let rank = shape.rank_usize();
    if let Some(&axis) = axes.iter().find(|&&axis| axis >= rank) {
        return Err(Error::ReductionAxisOutOfRange { axis, rank });
    }
    let mut seen = vec![false; rank];
    for &axis in axes {
        if seen[axis] {
            return Err(Error::DuplicateReductionAxis { axis });
        }
        seen[axis] = true;
    }
    Ok(Shape::from_vec(
        shape
            .iter()
            .enumerate()
            .filter_map(|(i, dim)| {
                if seen[i] {
                    keep_dimensions.then(|| Expression::new_const(1))
                } else {
                    Some(dim.clone())
                }
            })
            .collect(),
    ))
}

pub fn unsqueeze(shape: &Shape, axes: &[usize]) -> Result<Shape> {
    let output_rank = shape.rank_usize() + axes.len();
    if let Some(&axis) = axes.iter().find(|&&axis| axis >= output_rank) {
        return Err(Error::UnsqueezeAxisOutOfRange {
            axis,
            rank: output_rank,
        });
    }
    let mut seen = vec![false; output_rank];
    for &axis in axes {
        if seen[axis] {
            return Err(Error::DuplicateUnsqueezeAxis { axis });
        }
        seen[axis] = true;
    }
    let mut source = shape.iter();
    Ok(Shape::from_vec(
        (0..output_rank)
            .map(|i| {
                if seen[i] {
                    Expression::new_const(1)
                } else {
                    source.next().unwrap().clone()
                }
            })
            .collect(),
    ))
}

pub fn tile(shape: &Shape, repetitions: &[u32]) -> Result<Shape> {
    if shape.rank_usize() != repetitions.len() {
        return Err(Error::TileRankMismatch {
            expected: shape.rank_usize(),
            actual: repetitions.len(),
        });
    }
    Ok(Shape::from_vec(
        shape
            .iter()
            .zip(repetitions)
            .map(|(d, &r)| d.clone() * Expression::new_const(r as i64))
            .collect(),
    ))
}

pub fn matmul(context: &mut Context, lhs: &Shape, rhs: &Shape) -> Result<Shape> {
    let a: std::vec::Vec<_> = lhs.iter().cloned().collect();
    let b: std::vec::Vec<_> = rhs.iter().cloned().collect();
    if a.is_empty() || b.is_empty() {
        return Err(Error::MatmulScalarInput);
    }
    let a_k = a.last().unwrap();
    let b_k = if b.len() == 1 { &b[0] } else { &b[b.len() - 2] };
    context.equal_dim(a_k, b_k)?;
    let a_batch = if a.len() > 2 { &a[..a.len() - 2] } else { &[] };
    let b_batch = if b.len() > 2 { &b[..b.len() - 2] } else { &[] };
    let mut result: std::vec::Vec<_> = broadcast(
        context,
        &Shape::from_vec(a_batch.iter().cloned().collect()),
        &Shape::from_vec(b_batch.iter().cloned().collect()),
    )?
    .iter()
    .cloned()
    .collect();
    if a.len() > 1 {
        result.push(a[a.len() - 2].clone());
    }
    if b.len() > 1 {
        result.push(b[b.len() - 1].clone());
    }
    Ok(Shape::from_vec(result.into_iter().collect()))
}

pub fn squeeze(context: &mut Context, shape: &Shape, axes: &[usize]) -> Result<Shape> {
    let dims: std::vec::Vec<_> = shape.iter().cloned().collect();
    if let Some(&axis) = axes.iter().find(|&&axis| axis >= dims.len()) {
        return Err(Error::SqueezeAxisOutOfRange {
            axis,
            rank: dims.len(),
        });
    }
    let mut selected = vec![false; dims.len()];
    for &axis in axes {
        if selected[axis] {
            return Err(Error::DuplicateSqueezeAxis { axis });
        }
        context.equal_dim(&dims[axis], &Expression::new_const(1))?;
        selected[axis] = true;
    }
    Ok(Shape::from_vec(
        dims.into_iter()
            .enumerate()
            .filter_map(|(i, d)| (!selected[i]).then_some(d))
            .collect(),
    ))
}

pub fn window_output(
    input: Expression,
    kernel: Expression,
    padding: [u32; 2],
    stride: u32,
    dilation: u32,
    round_up: bool,
) -> Result<Expression> {
    if stride == 0 {
        return Err(Error::ZeroWindowStride);
    }
    if dilation == 0 {
        return Err(Error::ZeroWindowDilation);
    }
    let effective_kernel = (kernel - Expression::new_const(1))
        * Expression::new_const(dilation as i64)
        + Expression::new_const(1);
    let numerator =
        input + Expression::new_const((padding[0] as i64) + (padding[1] as i64)) - effective_kernel;
    let numerator = if round_up {
        numerator + Expression::new_const(stride as i64 - 1)
    } else {
        numerator
    };
    Ok(numerator / Expression::new_const(stride as i64) + Expression::new_const(1))
}

pub fn transposed_window_output(
    input: Expression,
    kernel: Expression,
    padding: [u32; 2],
    stride: u32,
    dilation: u32,
    output_padding: u32,
) -> Result<Expression> {
    if stride == 0 {
        return Err(Error::ZeroWindowStride);
    }
    if dilation == 0 {
        return Err(Error::ZeroWindowDilation);
    }
    Ok(
        (input - Expression::new_const(1)) * Expression::new_const(stride as i64)
            - Expression::new_const(padding[0] as i64 + padding[1] as i64)
            + (kernel - Expression::new_const(1)) * Expression::new_const(dilation as i64)
            + Expression::new_const(1 + output_padding as i64),
    )
}

pub fn split_equal(
    context: &mut Context,
    shape: &Shape,
    axis: usize,
    parts: u32,
) -> Result<std::vec::Vec<Shape>> {
    if axis >= shape.rank_usize() {
        return Err(Error::SplitAxisOutOfRange {
            axis,
            rank: shape.rank_usize(),
        });
    }
    if parts == 0 {
        return Err(Error::SplitZeroParts);
    }
    let dims: std::vec::Vec<_> = shape.iter().cloned().collect();
    if let Some(value) = dims[axis].as_const() {
        if value % parts as i64 != 0 {
            return Err(Error::SplitNotDivisible {
                dimension: value,
                parts,
            });
        }
    }
    let width = dims[axis].clone() / Expression::new_const(parts as i64);
    context.ensure_shape(&Shape::from_vec(vec![width.clone()].into()))?;
    Ok((0..parts)
        .map(|_| {
            let mut output = dims.clone();
            output[axis] = width.clone();
            Shape::from_vec(output.into())
        })
        .collect())
}

pub fn split_sizes(
    context: &mut Context,
    shape: &Shape,
    axis: usize,
    sizes: &[u32],
) -> Result<std::vec::Vec<Shape>> {
    if axis >= shape.rank_usize() {
        return Err(Error::SplitAxisOutOfRange {
            axis,
            rank: shape.rank_usize(),
        });
    }
    if sizes.is_empty() {
        return Err(Error::SplitSizesEmpty);
    }
    let dims: std::vec::Vec<_> = shape.iter().cloned().collect();
    let total: u64 = sizes.iter().map(|&size| size as u64).sum();
    let total = i64::try_from(total).map_err(|_| Error::SplitSizesOverflow { total })?;
    context.equal_dim(&dims[axis], &Expression::new_const(total))?;
    Ok(sizes
        .iter()
        .map(|&size| {
            let mut output = dims.clone();
            output[axis] = Expression::new_const(size as i64);
            Shape::from_vec(output.into())
        })
        .collect())
}

pub fn slice(shape: &Shape, starts: &[u32], sizes: &Shape, strides: &[u32]) -> Result<Shape> {
    let rank = shape.rank_usize();
    if starts.len() != rank || sizes.rank_usize() != rank || strides.len() != rank {
        return Err(Error::SliceRankMismatch {
            input: rank,
            starts: starts.len(),
            sizes: sizes.rank_usize(),
            strides: strides.len(),
        });
    }
    let mut output = std::vec::Vec::with_capacity(rank);
    for (axis, ((input, size), (&start, &stride))) in shape
        .iter()
        .zip(sizes.iter())
        .zip(starts.iter().zip(strides))
        .enumerate()
    {
        if stride == 0 {
            return Err(Error::SliceZeroStride { axis });
        }
        if let (Some(input), Some(size)) = (input.as_const(), size.as_const()) {
            if i64::from(start) + size > input {
                return Err(Error::SliceOutOfBounds {
                    axis,
                    start,
                    size,
                    input,
                });
            }
        }
        let extent = if stride == 1 {
            size.clone()
        } else {
            (size.clone() + Expression::new_const(stride as i64 - 1))
                / Expression::new_const(stride as i64)
        };
        output.push(extent);
    }
    Ok(Shape::from_vec(output.into()))
}

pub fn flatten_at_axis(shape: &Shape, axis: usize) -> Result<Shape> {
    if axis > shape.rank_usize() {
        return Err(Error::FlattenAxisOutOfRange {
            axis,
            rank: shape.rank_usize(),
        });
    }
    let dims: std::vec::Vec<_> = shape.iter().cloned().collect();
    let product = |part: &[Expression]| {
        part.iter()
            .cloned()
            .fold(Expression::new_const(1), |a, b| a * b)
    };
    Ok(Shape::from_vec(
        vec![product(&dims[..axis]), product(&dims[axis..])].into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn static_shape(dimensions: &[i64]) -> Shape {
        Shape::from_vec(
            dimensions
                .iter()
                .copied()
                .map(Expression::new_const)
                .collect(),
        )
    }

    #[test]
    fn conv2d_keeps_symbolic_height_and_checks_channels() {
        let mut context = Context::new(&crate::context::ContextOptions::default());
        let input = Shape::from_vec(
            vec![
                Expression::new_const(1),
                Expression::new_const(6),
                Expression::new_dynamic("height"),
                Expression::new_const(10),
            ]
            .into(),
        );
        let filter = static_shape(&[8, 3, 3, 3]);
        let options = Conv2dOptions {
            groups: 2,
            ..Default::default()
        };
        let output = conv2d(&mut context, &input, &filter, &options).unwrap();
        let dimensions: std::vec::Vec<_> = output.iter().map(ToString::to_string).collect();
        assert_eq!(dimensions[1], "8");
        assert_eq!(
            dimensions[2],
            "((((height + 0) - (((3 - 1) * 1) + 1)) / 1) + 1)"
        );
        let invalid = conv2d(
            &mut context,
            &static_shape(&[1, 5, 10, 10]),
            &filter,
            &options,
        )
        .unwrap();
        let bindings = [("height", 10)].into_iter().collect();
        assert!(matches!(
            context.compute_shapes_cel_comcrete(&[invalid], &bindings),
            Err(Error::RuntimeUnequalDimensions { .. })
        ));
    }

    #[test]
    fn conv2d_accepts_nhwc_hwio_layout() {
        let mut context = Context::new(&crate::context::ContextOptions::default());
        let options = Conv2dOptions {
            input_layout: InputLayout::Nhwc,
            filter_layout: Conv2dFilterLayout::Hwio,
            ..Default::default()
        };
        let output = conv2d(
            &mut context,
            &static_shape(&[1, 10, 12, 3]),
            &static_shape(&[3, 3, 3, 8]),
            &options,
        )
        .unwrap();
        let dimensions = context
            .compute_shapes_cel_comcrete(&[output], &Default::default())
            .unwrap();
        assert_eq!(&dimensions[0][..], &[1, 8, 10, 8]);
    }

    #[test]
    fn conv_transpose2d_uses_groups_and_explicit_output_sizes() {
        let mut context = Context::new(&crate::context::ContextOptions::default());
        let options = ConvTranspose2dOptions {
            input_layout: InputLayout::Nhwc,
            filter_layout: ConvTranspose2dFilterLayout::Hwoi,
            groups: 2,
            output_sizes: Some([9, 11]),
            ..Default::default()
        };
        let output = conv_transpose2d(
            &mut context,
            &static_shape(&[1, 5, 6, 4]),
            &static_shape(&[3, 3, 3, 4]),
            &options,
        )
        .unwrap();
        let dimensions = context
            .compute_shapes_cel_comcrete(&[output], &Default::default())
            .unwrap();
        assert_eq!(&dimensions[0][..], &[1, 9, 11, 6]);
    }

    #[test]
    fn conv_transpose2d_computes_symbolic_spatial_extent() {
        let mut context = Context::new(&crate::context::ContextOptions::default());
        let input = Shape::from_vec(
            vec![
                Expression::new_const(1),
                Expression::new_const(4),
                Expression::new_dynamic("height"),
                Expression::new_const(6),
            ]
            .into(),
        );
        let options = ConvTranspose2dOptions {
            strides: [2, 1],
            output_padding: [1, 0],
            ..Default::default()
        };
        let output =
            conv_transpose2d(&mut context, &input, &static_shape(&[4, 3, 3, 3]), &options).unwrap();
        let bindings = [("height", 5)].into_iter().collect();
        let dimensions = context
            .compute_shapes_cel_comcrete(&[output], &bindings)
            .unwrap();
        assert_eq!(&dimensions[0][..], &[1, 3, 12, 8]);
    }

    #[test]
    fn errors_identify_invalid_shape_arguments() {
        let mut context = Context::new(&crate::context::ContextOptions::default());
        let shape = Shape::from_vec(vec![Expression::new_const(5)].into());
        assert!(matches!(
            transpose(&shape, &[1]),
            Err(Error::TransposeAxisOutOfRange { axis: 1, rank: 1 })
        ));
        assert!(matches!(
            split_equal(&mut context, &shape, 0, 2),
            Err(Error::SplitNotDivisible {
                dimension: 5,
                parts: 2
            })
        ));
        assert!(matches!(
            slice(&shape, &[4], &shape, &[1]),
            Err(Error::SliceOutOfBounds {
                axis: 0,
                start: 4,
                size: 5,
                input: 5
            })
        ));
    }

    #[test]
    fn window_upper_bound_accounts_for_variable_kernel() {
        let expression = window_output(
            Expression::new_dynamic("height"),
            Expression::new_dynamic("kernel"),
            [0, 0],
            1,
            1,
            false,
        )
        .unwrap();
        let bounds = [("height", 20), ("kernel", 5)].into_iter().collect();
        assert!(expression.upper_bound(&bounds).unwrap() >= 20);
    }

    #[test]
    fn broadcast_uses_trailing_dimensions() {
        let mut context = Context::new(&crate::context::ContextOptions::default());
        let left = Shape::from_vec(
            vec![Expression::new_dynamic("batch"), Expression::new_const(3)].into(),
        );
        let right = Shape::from_vec(vec![Expression::new_const(3)].into());
        let output = broadcast(&mut context, &left, &right).unwrap();
        assert_eq!(
            output
                .iter()
                .map(ToString::to_string)
                .collect::<std::vec::Vec<_>>(),
            ["batch", "3"]
        );
    }

    #[test]
    fn broadcast_expression_can_be_parsed_again() {
        let mut context = Context::new(&crate::context::ContextOptions::default());
        let output = context
            .broadcast_dim(
                Expression::new_dynamic("left"),
                Expression::new_dynamic("right"),
                "incompatible dimensions",
            )
            .unwrap();
        let text = output.to_string();
        let parsed = Expression::parse(&text).unwrap();
        assert_eq!(parsed.to_string(), text);
    }

    #[test]
    fn rounded_scale_is_evaluable_and_serializable() {
        let expression = Expression::new_dynamic("width").round_scaled(1.5).unwrap();
        let reparsed = Expression::parse(&expression.to_string()).unwrap();
        let shape = Shape::from_vec(vec![reparsed].into());
        let context = Context::new(&crate::context::ContextOptions::default());
        let input = [("width", 3)].into_iter().collect();
        assert_eq!(
            &context
                .compute_shapes_cel_comcrete(&[shape], &input)
                .unwrap()[0][..],
            &[5]
        );
        assert_eq!(
            expression
                .upper_bound(&[("width", 10)].into_iter().collect())
                .unwrap(),
            15
        );
    }

    #[test]
    fn flatten_keeps_symbolic_product() {
        let shape = Shape::from_vec(
            vec![
                Expression::new_const(2),
                Expression::new_dynamic("batch"),
                Expression::new_const(3),
            ]
            .into(),
        );
        let result = flatten_at_axis(&shape, 2).unwrap();
        let bounds = [("batch", 10)].into_iter().collect();
        assert_eq!(
            result.iter().next().unwrap().upper_bound(&bounds).unwrap(),
            20
        );
    }
}

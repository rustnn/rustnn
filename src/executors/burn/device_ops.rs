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

//! Device-side conv, pool, and reduce using Burn kernels.

use burn_tensor::backend::Backend;
use burn_tensor::module::conv2d;
use burn_tensor::ops::{ConvOptions, PaddedConvOptions};
use burn_tensor::{Tensor, TensorData};

use super::options::parse_option_or_default as parse_option;
use super::outputs::OperationOutputs;
use super::shape::ReductionSpec;
use crate::error::GraphError;
use crate::operator_enums::MLInputOperandLayout;
use crate::operator_options::{MLConv2dOptions, MLGemmOptions, MLPool2dOptions, MLReduceOptions};
use crate::shape_inference::infer_conv2d_shape;

use super::host_array::{
    PoolKind, ReduceKind, conv2d_filter_dims_oihw, pad_to_2, pad_to_4, reorder_filter_to_oihw,
};
use super::tensor_env::{RuntimeTensor, TensorEnv, UnaryDeviceOp};

fn burn_err(reason: String) -> GraphError {
    GraphError::BurnRuntimeFailed { reason }
}

fn nhwc_to_nchw<B: Backend>(tensor: Tensor<B, 4>) -> Tensor<B, 4> {
    tensor.swap_dims(1, 3).swap_dims(2, 3)
}

fn nchw_to_nhwc<B: Backend>(tensor: Tensor<B, 4>) -> Tensor<B, 4> {
    tensor.swap_dims(2, 3).swap_dims(1, 3)
}

fn conv_options_2d(opts: &MLConv2dOptions) -> PaddedConvOptions<2> {
    let strides = pad_to_2(&opts.strides, 1);
    let dilation = pad_to_2(&opts.dilations, 1);
    let groups = opts.groups.max(1) as usize;
    let p = pad_to_4(&opts.padding);
    let pad_start = [p[0] as usize, p[2] as usize];
    let pad_end = [p[1] as usize, p[3] as usize];
    if p[0] == p[1] && p[2] == p[3] {
        ConvOptions::new(
            [strides[0] as usize, strides[1] as usize],
            pad_start,
            [dilation[0] as usize, dilation[1] as usize],
            groups,
        )
        .into()
    } else {
        PaddedConvOptions::asymmetric(
            [strides[0] as usize, strides[1] as usize],
            pad_start,
            pad_end,
            [dilation[0] as usize, dilation[1] as usize],
            groups,
        )
    }
}

/// Run a host kernel (sync inputs from device, compute on CPU, upload result once).
pub fn host_kernel<B, F>(env: &TensorEnv<B>, f: F) -> Result<RuntimeTensor<B>, GraphError>
where
    B: Backend,
    F: FnOnce(&TensorEnv<B>) -> Result<super::host_array::HostArray, GraphError>,
{
    RuntimeTensor::from_host_array(f(env)?, &env.device)
}

/// Insert a device tensor as an operation output.
pub fn insert_tensor<B: Backend>(
    env: &mut OperationOutputs<'_, B>,
    outputs: &[u32],
    tensor: RuntimeTensor<B>,
) -> Result<(), GraphError> {
    let id = outputs
        .first()
        .copied()
        .ok_or_else(|| GraphError::BurnRuntimeFailed {
            reason: "operation produced no output operand".to_string(),
        })?;
    env.write_tensor(id, tensor)?;
    Ok(())
}

pub fn conv2d_device<B: Backend>(
    env: &TensorEnv<B>,
    input_id: u32,
    filter_id: u32,
    bias_id: Option<u32>,
    options: &MLConv2dOptions,
) -> Result<RuntimeTensor<B>, GraphError> {
    let input_rt = env.get(input_id)?;
    if input_rt.rank() != 4 {
        return Err(burn_err(format!(
            "conv2d expects 4D input, got rank {}",
            input_rt.rank()
        )));
    }
    let nchw = options.input_layout == MLInputOperandLayout::Nchw;
    let mut x = input_rt.tensor_4()?;
    if !nchw {
        x = nhwc_to_nchw(x);
    }

    let filter_host = env.get_host(filter_id)?;
    let (c_out, ic_per_filter, kh, kw) =
        conv2d_filter_dims_oihw(options.filter_layout, &filter_host.shape)?;
    let filter_oihw = reorder_filter_to_oihw(
        &filter_host.data,
        options.filter_layout,
        c_out,
        ic_per_filter,
        kh,
        kw,
    );
    let weight = Tensor::<B, 4>::from_data(
        TensorData::new(filter_oihw, [c_out, ic_per_filter, kh, kw]),
        &env.device,
    );

    let bias = if let Some(id) = bias_id {
        let b = env.get_host(id)?;
        Some(Tensor::<B, 1>::from_data(
            TensorData::new(b.data, [b.shape[0]]),
            &env.device,
        ))
    } else {
        None
    };

    let mut y = conv2d(x, weight, bias, conv_options_2d(options));
    if !nchw {
        y = nchw_to_nhwc(y);
    }

    let input_shape_u32: Vec<u32> = input_rt.shape().iter().map(|&d| d as u32).collect();
    let filter_shape_u32: Vec<u32> = filter_host.shape.iter().map(|&d| d as u32).collect();
    let out_shape_u32 = infer_conv2d_shape(&input_shape_u32, &filter_shape_u32, options)?;
    let out_shape: Vec<usize> = out_shape_u32.iter().map(|&d| d as usize).collect();
    RuntimeTensor::from_d4(out_shape, y)
}

pub fn pool2d_device<B: Backend>(
    env: &TensorEnv<B>,
    input_id: u32,
    options: &MLPool2dOptions,
    kind: PoolKind,
) -> Result<RuntimeTensor<B>, GraphError> {
    let options = options.clone();
    host_kernel(env, |env| {
        let input = env.get_host(input_id)?;
        if input.rank() != 4 {
            return Err(burn_err(format!(
                "pool2d expects 4D input, got rank {}",
                input.rank()
            )));
        }
        super::host_array::pool2d(&input, &options, kind)
    })
}

pub fn reduce_device<B: Backend>(
    env: &TensorEnv<B>,
    input_id: u32,
    options: &Option<MLReduceOptions>,
    kind: ReduceKind,
) -> Result<RuntimeTensor<B>, GraphError> {
    let opts = options.clone().unwrap_or_default();
    let input = env.get(input_id)?;
    let spec = ReductionSpec::new(input.shape(), opts.axes.as_deref(), opts.keep_dimensions)?;
    let out_shape = &spec.output_shape;
    if spec.axes.is_empty() {
        return host_kernel(env, |env| {
            super::host_array::reduce(
                &env.get_host(input_id)?,
                opts.axes.as_deref(),
                opts.keep_dimensions,
                kind,
            )
        });
    }

    let axes_vec = spec.axes;

    let reduced = match kind {
        ReduceKind::Sum
        | ReduceKind::Mean
        | ReduceKind::Max
        | ReduceKind::Min
        | ReduceKind::Product => input.reduce_dims(&axes_vec, kind)?,
        ReduceKind::SumSquare => input.square()?.reduce_dims(&axes_vec, ReduceKind::Sum)?,
        ReduceKind::L1 => input
            .unary(UnaryDeviceOp::Abs)?
            .reduce_dims(&axes_vec, ReduceKind::Sum)?,
        ReduceKind::L2 => input
            .square()?
            .reduce_dims(&axes_vec, ReduceKind::Sum)?
            .unary(UnaryDeviceOp::Sqrt)?,
        ReduceKind::LogSum | ReduceKind::LogSumExp => {
            return host_kernel(env, |env| {
                super::host_array::reduce(
                    &env.get_host(input_id)?,
                    opts.axes.as_deref(),
                    opts.keep_dimensions,
                    kind,
                )
            });
        }
    };
    reduced.reshape_to(out_shape, &env.device)
}

pub fn gemm_device<B: Backend>(
    env: &TensorEnv<B>,
    a_id: u32,
    b_id: u32,
    c_id: Option<u32>,
    options: &MLGemmOptions,
) -> Result<RuntimeTensor<B>, GraphError> {
    let mut a = env.get(a_id)?.clone();
    let mut b = env.get(b_id)?.clone();
    if options.a_transpose {
        a = a.swap_last_two_dims()?;
    }
    if options.b_transpose {
        b = b.swap_last_two_dims()?;
    }
    let mut out = a.matmul(&b)?;
    let alpha = options.alpha as f32;
    if alpha != 1.0 {
        out = out.mul_scalar(alpha)?;
    }
    if let Some(id) = c_id {
        let beta = options.beta as f32;
        if beta != 0.0 {
            let c = env.get(id)?;
            out = out.add_scaled(c, beta, &env.device)?;
        }
    }
    Ok(out)
}

pub fn global_pool_device<B: Backend>(
    env: &TensorEnv<B>,
    input_id: u32,
    options: &MLPool2dOptions,
    kind: PoolKind,
) -> Result<RuntimeTensor<B>, GraphError> {
    let input_rt = env.get(input_id)?;
    if input_rt.rank() != 4 {
        return Err(burn_err(format!(
            "global pool expects 4D input, got rank {}",
            input_rt.rank()
        )));
    }
    let nchw = parse_option::<MLInputOperandLayout>(&options.layout, "layout")?
        == MLInputOperandLayout::Nchw;
    let (h, w) = if nchw {
        (input_rt.shape()[2], input_rt.shape()[3])
    } else {
        (input_rt.shape()[1], input_rt.shape()[2])
    };
    let mut opts = options.clone();
    opts.window_dimensions = Some(vec![h as u32, w as u32]);
    if opts.strides.is_empty() {
        opts.strides = vec![h as u32, w as u32];
    }
    pool2d_device(env, input_id, &opts, kind)
}

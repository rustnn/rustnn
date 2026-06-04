//! Device-side conv, pool, and reduce using Burn kernels.

use std::collections::HashMap;

use burn::tensor::backend::Backend;
use burn::tensor::module::conv2d;
use burn::tensor::ops::{ConvOptions, PaddedConvOptions};
use burn::tensor::{Tensor, TensorData};

use crate::error::GraphError;
use crate::graph::DataType;
use crate::operator_options::{
    MLConv2dOptions, MLDimension, MLGemmOptions, MLPool2dOptions, MLReduceOptions,
    MLSqueezeOptions, MLUnsqueezeOptions, mldimensions_static_or_max,
};
use crate::shape_inference::infer_conv2d_shape;

use super::host_array::{
    PoolKind, ReduceKind, conv2d_filter_dims_oihw, pad_to_2, pad_to_4, reduce_output_shape,
    reorder_filter_to_oihw,
};
use super::tensor_env::{RuntimeTensor, TensorEnv, UnaryDeviceOp};

fn burn_err(reason: String) -> GraphError {
    GraphError::BurnRuntimeFailed { reason }
}

fn layout_is_nchw(layout: &str) -> bool {
    !layout.eq_ignore_ascii_case("nhwc")
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
    env: &mut TensorEnv<B>,
    operand_types: &HashMap<u32, DataType>,
    outputs: &[u32],
    tensor: RuntimeTensor<B>,
) -> Result<(), GraphError> {
    let id = outputs
        .first()
        .copied()
        .ok_or_else(|| GraphError::BurnRuntimeFailed {
            reason: "operation produced no output operand".to_string(),
        })?;
    let out_dt = operand_types.get(&id).copied().unwrap_or(DataType::Float32);
    env.insert(id, out_dt, tensor);
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
    let nchw = layout_is_nchw(&options.input_layout);
    let mut x = input_rt.tensor_4()?;
    if !nchw {
        x = nhwc_to_nchw(x);
    }

    let filter_host = env.get_host(filter_id)?;
    let (c_out, ic_per_filter, kh, kw) =
        conv2d_filter_dims_oihw(&options.filter_layout, &filter_host.shape)?;
    let filter_oihw = reorder_filter_to_oihw(
        &filter_host.data,
        &options.filter_layout,
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
    let axes: Option<Vec<usize>> = opts
        .axes
        .as_ref()
        .map(|ax| ax.iter().map(|&a| a as usize).collect());
    let axes_ref = axes.as_deref();
    let out_shape = reduce_output_shape(input.shape(), opts.axes.as_deref(), opts.keep_dimensions);

    if axes_ref.is_some_and(|a| a.is_empty()) {
        return host_kernel(env, |env| {
            super::host_array::reduce(
                &env.get_host(input_id)?,
                opts.axes.as_deref(),
                opts.keep_dimensions,
                kind,
            )
        });
    }

    let axes_vec = match axes_ref {
        None => (0..input.rank()).collect::<Vec<_>>(),
        Some(ax) => ax.to_vec(),
    };

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
    reduced.reshape_to(&out_shape, &env.device)
}

pub fn reshape_device<B: Backend>(
    env: &TensorEnv<B>,
    input_id: u32,
    new_shape: Vec<usize>,
) -> Result<RuntimeTensor<B>, GraphError> {
    env.get(input_id)?.reshape_to(&new_shape, &env.device)
}

pub fn transpose_device<B: Backend>(
    env: &TensorEnv<B>,
    input_id: u32,
    perm: Vec<usize>,
) -> Result<RuntimeTensor<B>, GraphError> {
    let input = env.get(input_id)?;
    let perm = if perm.is_empty() && input.rank() > 0 {
        (0..input.rank()).rev().collect()
    } else {
        perm
    };
    input.permute_dims(&perm)
}

pub fn expand_device<B: Backend>(
    env: &TensorEnv<B>,
    input_id: u32,
    new_shape: &[MLDimension],
    numel_hint: usize,
) -> Result<RuntimeTensor<B>, GraphError> {
    let mut target: Vec<usize> = mldimensions_static_or_max(new_shape)
        .iter()
        .map(|&d| d as usize)
        .collect();
    if target.iter().any(|&d| d == 0) {
        let known: usize = target.iter().filter(|&&d| d != 0).product();
        let inferred = if known == 0 {
            numel_hint
        } else {
            numel_hint / known
        };
        for dim in &mut target {
            if *dim == 0 {
                *dim = inferred;
            }
        }
    }
    env.get(input_id)?.broadcast_to_shape(&target, &env.device)
}

pub fn concat_device<B: Backend>(
    env: &TensorEnv<B>,
    input_ids: &[u32],
    axis: usize,
) -> Result<RuntimeTensor<B>, GraphError> {
    let tensors: Vec<RuntimeTensor<B>> = input_ids
        .iter()
        .map(|&id| env.get(id).cloned())
        .collect::<Result<_, _>>()?;
    RuntimeTensor::concat_same_rank(&tensors, axis)
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
    let nchw = layout_is_nchw(&options.layout);
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

pub fn squeeze_device<B: Backend>(
    env: &TensorEnv<B>,
    input_id: u32,
    options: &MLSqueezeOptions,
) -> Result<RuntimeTensor<B>, GraphError> {
    let input = env.get(input_id)?;
    let input_shape_u32: Vec<u32> = input.shape().iter().map(|&d| d as u32).collect();
    let axes = if options.axes.is_empty() {
        None
    } else {
        Some(options.axes.as_slice())
    };
    let out_shape_u32 = crate::shape_inference::infer_squeeze_shape(&input_shape_u32, axes)?;
    let out_shape: Vec<usize> = out_shape_u32.iter().map(|&d| d as usize).collect();
    input.reshape_to(&out_shape, &env.device)
}

pub fn unsqueeze_device<B: Backend>(
    env: &TensorEnv<B>,
    input_id: u32,
    options: &MLUnsqueezeOptions,
) -> Result<RuntimeTensor<B>, GraphError> {
    let input = env.get(input_id)?;
    let input_shape_u32: Vec<u32> = input.shape().iter().map(|&d| d as u32).collect();
    let out_shape_u32 =
        crate::shape_inference::infer_unsqueeze_shape(&input_shape_u32, &options.axes)?;
    let out_shape: Vec<usize> = out_shape_u32.iter().map(|&d| d as usize).collect();
    input.reshape_to(&out_shape, &env.device)
}

pub fn slice_device<B: Backend>(
    env: &TensorEnv<B>,
    input_id: u32,
    starts: &[u32],
    sizes: &[MLDimension],
    strides: &[u32],
) -> Result<RuntimeTensor<B>, GraphError> {
    let sizes_usize: Vec<usize> = mldimensions_static_or_max(sizes)
        .iter()
        .map(|&d| d as usize)
        .collect();
    env.get(input_id)?.slice_dims(starts, &sizes_usize, strides)
}

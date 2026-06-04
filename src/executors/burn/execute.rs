//! Execute WebNN [`Operation`] values on Burn device tensors.

use std::collections::HashMap;

use burn::tensor::backend::Backend;

use crate::error::GraphError;
use crate::graph::DataType;
use crate::operator_enums::MLOperandDataType;
use crate::operator_options::{
    MLClampOptions, MLDimension, MLEluOptions, MLHardSigmoidOptions, MLLinearOptions,
    mldimensions_static_or_max,
};
use crate::operators::Operation;

use super::device_ops::{
    concat_device, conv2d_device, expand_device, gemm_device, global_pool_device, host_kernel,
    insert_tensor, pool2d_device, reduce_device, reshape_device, slice_device, squeeze_device,
    transpose_device, unsqueeze_device,
};
use super::f16::{
    f16_add, f16_div, f16_mul, f16_neg, f16_sqrt, f16_sub, is_integer_element_type, round_f16,
    round_f16_slice, use_f16_arithmetic, use_integer_arithmetic,
};
use super::host_array::{
    HostArray, PoolKind, ReduceKind, batch_normalization, cast_values, conv_transpose2d, gather,
    instance_normalization, integer_div_broadcast, layer_normalization, pow_broadcast,
    split_tensor,
};
use super::host_ops_extra::{
    arg_max, arg_min, cumulative_sum, dequantize_linear, erf, format_rnn_hidden_sequence,
    format_rnn_state_nd, gather_elements, gather_nd, gelu, gru, gru_cell, is_infinite, is_nan,
    l2_pool2d, logical_and, logical_or, logical_xor, lstm, lstm_cell, not_equal, pad, prelu,
    quantize_linear, reciprocal, resample2d, reverse, round_even, scatter_elements, scatter_nd,
    shape, sign, tile, triangular, where_op,
};
use super::tensor_env::{DeviceBinaryOp, DeviceCompareOp, RuntimeTensor, TensorEnv, UnaryDeviceOp};

pub fn execute_operations<B: Backend>(
    env: &mut TensorEnv<B>,
    operations: &[Operation],
    operand_types: &HashMap<u32, DataType>,
) -> Result<(), GraphError> {
    for op in operations {
        execute_one(env, op, operand_types)?;
    }
    Ok(())
}

fn execute_one<B: Backend>(
    env: &mut TensorEnv<B>,
    op: &Operation,
    operand_types: &HashMap<u32, DataType>,
) -> Result<(), GraphError> {
    match op {
        Operation::Add { a, b, outputs, .. } => {
            binary_device_or_f16(
                env,
                operand_types,
                *a,
                *b,
                outputs,
                DeviceBinaryOp::Add,
                |a, b| a.binary_broadcast(b, f16_add),
            )?;
        }
        Operation::Sub { a, b, outputs, .. } => {
            binary_device_or_f16(
                env,
                operand_types,
                *a,
                *b,
                outputs,
                DeviceBinaryOp::Sub,
                |a, b| a.binary_broadcast(b, f16_sub),
            )?;
        }
        Operation::Mul { a, b, outputs, .. } => {
            binary_device_or_f16(
                env,
                operand_types,
                *a,
                *b,
                outputs,
                DeviceBinaryOp::Mul,
                |a, b| a.binary_broadcast(b, f16_mul),
            )?;
        }
        Operation::Div { a, b, outputs, .. } => {
            let output = outputs[0];
            let needs_integer_div =
                use_integer_arithmetic(&env.dtypes, &[*a, *b], output, operand_types)
                    || operand_types
                        .get(&a)
                        .copied()
                        .is_some_and(is_integer_element_type)
                    || operand_types
                        .get(&b)
                        .copied()
                        .is_some_and(is_integer_element_type)
                    || operand_types
                        .get(&output)
                        .copied()
                        .is_some_and(is_integer_element_type);
            if needs_integer_div {
                let out = env.binary_host(*a, *b, integer_div_broadcast)?;
                insert_output(env, operand_types, outputs, out)?;
            } else {
                binary_device_or_f16(
                    env,
                    operand_types,
                    *a,
                    *b,
                    outputs,
                    DeviceBinaryOp::Div,
                    |a, b| a.binary_broadcast(b, f16_div),
                )?;
            }
        }
        Operation::Pow { a, b, outputs, .. } => {
            let out = host_kernel(env, |env| {
                pow_broadcast(&env.get_host(*a)?, &env.get_host(*b)?)
            })?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::Equal { a, b, outputs, .. } => {
            let out = env.compare_broadcast(*a, *b, DeviceCompareOp::Equal)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::Greater { a, b, outputs, .. } => {
            let out = env.compare_broadcast(*a, *b, DeviceCompareOp::Greater)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::GreaterOrEqual { a, b, outputs, .. } => {
            let out = env.compare_broadcast(*a, *b, DeviceCompareOp::GreaterOrEqual)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::Lesser { a, b, outputs, .. } => {
            let out = env.compare_broadcast(*a, *b, DeviceCompareOp::Lesser)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::LesserOrEqual { a, b, outputs, .. } => {
            let out = env.compare_broadcast(*a, *b, DeviceCompareOp::LesserOrEqual)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::Min { a, b, outputs, .. } => {
            let out = env.binary_broadcast(*a, *b, DeviceBinaryOp::Min)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::Max { a, b, outputs, .. } => {
            let out = env.binary_broadcast(*a, *b, DeviceBinaryOp::Max)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::Matmul { a, b, outputs, .. } => {
            let out = env.matmul(*a, *b)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::Abs { input, outputs, .. } => {
            unary_device_or_f16(
                env,
                operand_types,
                *input,
                outputs[0],
                outputs,
                UnaryDeviceOp::Abs,
                f32::abs,
            )?;
        }
        Operation::Neg { input, outputs, .. } => {
            let in_dt = operand_types
                .get(input)
                .copied()
                .unwrap_or(DataType::Float32);
            let out = match in_dt {
                DataType::Int64 => {
                    let values =
                        env.int64_data
                            .get(input)
                            .ok_or_else(|| GraphError::BurnRuntimeFailed {
                                reason: "neg int64 input missing exact integer sidecar".to_string(),
                            })?;
                    let shape = env.get(*input)?.shape().to_vec();
                    HostArray::from_i64(shape, values.iter().map(|&v| v.wrapping_neg()).collect())?
                }
                _ => {
                    if use_f16_arithmetic(&env.dtypes, &[*input], outputs[0], operand_types) {
                        env.get_host(*input)?.map_unary(f16_neg)
                    } else {
                        let tensor = env.unary(*input, UnaryDeviceOp::Neg)?;
                        return insert_tensor(env, operand_types, outputs, tensor);
                    }
                }
            };
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::Exp { input, outputs, .. } => {
            unary_device_or_f16(
                env,
                operand_types,
                *input,
                outputs[0],
                outputs,
                UnaryDeviceOp::Exp,
                f32::exp,
            )?;
        }
        Operation::Log { input, outputs, .. } => {
            unary_device_or_f16(
                env,
                operand_types,
                *input,
                outputs[0],
                outputs,
                UnaryDeviceOp::Log,
                f32::ln,
            )?;
        }
        Operation::Sqrt { input, outputs, .. } => {
            unary_device_or_f16(
                env,
                operand_types,
                *input,
                outputs[0],
                outputs,
                UnaryDeviceOp::Sqrt,
                f16_sqrt,
            )?;
        }
        Operation::Floor { input, outputs, .. } => {
            unary_device_or_f16(
                env,
                operand_types,
                *input,
                outputs[0],
                outputs,
                UnaryDeviceOp::Floor,
                f32::floor,
            )?;
        }
        Operation::Ceil { input, outputs, .. } => {
            unary_device_or_f16(
                env,
                operand_types,
                *input,
                outputs[0],
                outputs,
                UnaryDeviceOp::Ceil,
                f32::ceil,
            )?;
        }
        Operation::Relu { input, outputs, .. } => {
            unary_device_or_f16(
                env,
                operand_types,
                *input,
                outputs[0],
                outputs,
                UnaryDeviceOp::Relu,
                |x| x.max(0.0),
            )?;
        }
        Operation::LeakyRelu {
            input,
            options,
            outputs,
            ..
        } => {
            let alpha = options.as_ref().map(|o| o.alpha).unwrap_or(0.01);
            if use_f16_arithmetic(&env.dtypes, &[*input], outputs[0], operand_types) {
                let alpha_f32 = alpha as f32;
                insert_output(
                    env,
                    operand_types,
                    outputs,
                    unary_output(env, operand_types, *input, outputs[0], move |x| {
                        if x >= 0.0 { x } else { alpha_f32 * x }
                    })?,
                )?;
            } else {
                let out = env.unary(*input, UnaryDeviceOp::LeakyRelu { slope: alpha })?;
                insert_tensor(env, operand_types, outputs, out)?;
            }
        }
        Operation::Elu {
            input,
            options,
            outputs,
            ..
        } => {
            let alpha = options
                .as_ref()
                .map(|o: &MLEluOptions| o.alpha)
                .unwrap_or(1.0);
            if use_f16_arithmetic(&env.dtypes, &[*input], outputs[0], operand_types) {
                let alpha_f32 = alpha as f32;
                insert_output(
                    env,
                    operand_types,
                    outputs,
                    unary_output(env, operand_types, *input, outputs[0], move |x| {
                        if x >= 0.0 {
                            x
                        } else {
                            alpha_f32 * (x.exp() - 1.0)
                        }
                    })?,
                )?;
            } else {
                let out = env.unary(*input, UnaryDeviceOp::Elu { alpha })?;
                insert_tensor(env, operand_types, outputs, out)?;
            }
        }
        Operation::Sigmoid { input, outputs, .. } => {
            unary_device_or_f16(
                env,
                operand_types,
                *input,
                outputs[0],
                outputs,
                UnaryDeviceOp::Sigmoid,
                |x| 1.0 / (1.0 + (-x).exp()),
            )?;
        }
        Operation::Tanh { input, outputs, .. } => {
            unary_device_or_f16(
                env,
                operand_types,
                *input,
                outputs[0],
                outputs,
                UnaryDeviceOp::Tanh,
                f32::tanh,
            )?;
        }
        Operation::HardSigmoid {
            input,
            options,
            outputs,
            ..
        } => {
            let (alpha, beta) = hard_sigmoid_params(options);
            if use_f16_arithmetic(&env.dtypes, &[*input], outputs[0], operand_types) {
                insert_output(
                    env,
                    operand_types,
                    outputs,
                    unary_output(env, operand_types, *input, outputs[0], move |x| {
                        (alpha * x + beta).clamp(0.0, 1.0)
                    })?,
                )?;
            } else {
                let out = env.unary(
                    *input,
                    UnaryDeviceOp::HardSigmoid {
                        alpha: alpha as f64,
                        beta: beta as f64,
                    },
                )?;
                insert_tensor(env, operand_types, outputs, out)?;
            }
        }
        Operation::HardSwish { input, outputs, .. } => {
            unary_device_or_f16(
                env,
                operand_types,
                *input,
                outputs[0],
                outputs,
                UnaryDeviceOp::HardSwish,
                |x| x * (x + 3.0).clamp(0.0, 6.0) / 6.0,
            )?;
        }
        Operation::Linear {
            input,
            options,
            outputs,
            ..
        } => {
            let (alpha, beta) = linear_params(options);
            insert_output(
                env,
                operand_types,
                outputs,
                unary_output(env, operand_types, *input, outputs[0], move |x| {
                    alpha * x + beta
                })?,
            )?;
        }
        Operation::LogicalNot { input, outputs, .. } => {
            let out = host_kernel(env, |env| {
                Ok(env
                    .get_host(*input)?
                    .map_unary(|x| if x == 0.0 { 1.0 } else { 0.0 }))
            })?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::Softmax {
            input,
            axis,
            outputs,
            ..
        } => {
            let out = env.unary(
                *input,
                UnaryDeviceOp::Softmax {
                    axis: *axis as usize,
                },
            )?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::Clamp {
            input,
            options,
            outputs,
            ..
        } => {
            let in_dt = operand_types
                .get(input)
                .copied()
                .unwrap_or(DataType::Float32);
            let out = match in_dt {
                DataType::Int64 => {
                    let values =
                        env.int64_data
                            .get(input)
                            .ok_or_else(|| GraphError::BurnRuntimeFailed {
                                reason: "clamp int64 input missing exact integer sidecar"
                                    .to_string(),
                            })?;
                    let (min_v, max_v) = clamp_i64_bounds(options);
                    let shape = env.get(*input)?.shape().to_vec();
                    HostArray::from_i64(
                        shape,
                        values.iter().map(|&v| clamp_i64(v, min_v, max_v)).collect(),
                    )?
                }
                DataType::Uint64 => {
                    let values = env.uint64_data.get(input).ok_or_else(|| {
                        GraphError::BurnRuntimeFailed {
                            reason: "clamp uint64 input missing exact integer sidecar".to_string(),
                        }
                    })?;
                    let (min_v, max_v) = clamp_u64_bounds(options);
                    let shape = env.get(*input)?.shape().to_vec();
                    HostArray::from_u64(
                        shape,
                        values.iter().map(|&v| clamp_u64(v, min_v, max_v)).collect(),
                    )?
                }
                _ => {
                    let (min_v, max_v) = clamp_bounds(options);
                    unary_output(env, operand_types, *input, outputs[0], move |x| {
                        clamp_f32(x, min_v, max_v)
                    })?
                }
            };
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::Cast {
            input,
            data_type,
            outputs,
            ..
        } => {
            let out_dt = ml_operand_to_data_type(*data_type);
            let input_arr = env.get_host(*input)?;
            if out_dt == DataType::Int64 {
                let out = if let Some(values) = input_arr.i64_data {
                    HostArray::from_i64(input_arr.shape.clone(), values)?
                } else if let Some(values) = env.int64_data.get(input) {
                    HostArray::from_i64(input_arr.shape.clone(), values.clone())?
                } else {
                    let values: Vec<i64> = input_arr
                        .data
                        .iter()
                        .map(|&v| cast_values(&[v], *data_type)[0] as i64)
                        .collect();
                    HostArray::from_i64(input_arr.shape.clone(), values)?
                };
                insert_host_output(env, operand_types, outputs, out, out_dt)?;
            } else if out_dt == DataType::Uint64 {
                let out = if let Some(values) = input_arr.u64_data {
                    HostArray::from_u64(input_arr.shape.clone(), values)?
                } else if let Some(values) = env.uint64_data.get(input) {
                    HostArray::from_u64(input_arr.shape.clone(), values.clone())?
                } else {
                    let values: Vec<u64> = input_arr
                        .data
                        .iter()
                        .map(|&v| cast_values(&[v], *data_type)[0] as u64)
                        .collect();
                    HostArray::from_u64(input_arr.shape.clone(), values)?
                };
                insert_host_output(env, operand_types, outputs, out, out_dt)?;
            } else {
                let values = cast_values(&input_arr.data, *data_type);
                let shape = input_arr.shape.clone();
                let out = HostArray::new(shape, values)?;
                insert_output_typed(env, outputs, out, out_dt)?;
            }
        }
        Operation::Reshape {
            input,
            new_shape,
            outputs,
            ..
        } => {
            let numel = env.get(*input)?.numel();
            let shape = resolve_mldims(new_shape, numel)?;
            let out = reshape_device(env, *input, shape)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::Transpose {
            input,
            options,
            outputs,
            ..
        } => {
            let rank = env.get(*input)?.rank();
            let perm: Vec<usize> = if let Some(opts) = options {
                if opts.permutation.is_empty() {
                    (0..rank).rev().collect()
                } else {
                    opts.permutation.iter().map(|&p| p as usize).collect()
                }
            } else {
                (0..rank).rev().collect()
            };
            let out = transpose_device(env, *input, perm)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::Concat {
            inputs,
            axis,
            outputs,
            ..
        } => {
            let out = concat_device(env, inputs, *axis as usize)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::Slice {
            input,
            starts,
            sizes,
            options,
            outputs,
            ..
        } => {
            let strides = options
                .as_ref()
                .map(|o| o.strides.clone())
                .unwrap_or_default();
            let out = slice_device(env, *input, starts, sizes, &strides)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::Split {
            input,
            splits,
            split_equal_parts,
            options,
            outputs,
            ..
        } => {
            let axis = options.as_ref().map(|o| o.axis).unwrap_or(0);
            let parts = split_tensor(&env.get_host(*input)?, axis, splits, *split_equal_parts)?;
            if parts.len() != outputs.len() {
                return Err(op_err(op, "split output count mismatch"));
            }
            for (id, part) in outputs.iter().zip(parts) {
                let dt = operand_types
                    .get(id)
                    .copied()
                    .or_else(|| env.dtypes.get(id).copied())
                    .unwrap_or(DataType::Float32);
                let tensor = RuntimeTensor::from_host_array(part, &env.device)?;
                env.insert(*id, dt, tensor);
            }
        }
        Operation::Expand {
            input,
            new_shape,
            outputs,
            ..
        } => {
            let numel = env.get(*input)?.numel();
            let out = expand_device(env, *input, new_shape, numel)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::Gather {
            input,
            indices,
            batch_dimensions,
            options,
            outputs,
            ..
        } => {
            let axis = options.as_ref().map(|o| o.axis).unwrap_or(0);
            let input_arr = env.get_host(*input)?;
            let indices_arr = env.get_host(*indices)?;
            let out = gather(
                &input_arr,
                &indices_arr,
                axis,
                batch_dimensions.unwrap_or(0),
            )?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::Conv2d {
            input,
            filter,
            options,
            outputs,
            ..
        } => {
            let opts = options.clone().unwrap_or_default();
            let out = conv2d_device(env, *input, *filter, opts.bias, &opts)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::ConvTranspose2d {
            input,
            filter,
            options,
            outputs,
            ..
        } => {
            let opts = options.clone().unwrap_or_default();
            let input_arr = env.get_host(*input)?;
            let filter_arr = env.get_host(*filter)?;
            let bias = opts.bias.map(|id| env.get_host(id)).transpose()?;
            let out = conv_transpose2d(&input_arr, &filter_arr, bias.as_ref(), &opts)?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::AveragePool2d {
            input,
            options,
            outputs,
            ..
        } => {
            let out = pool2d_device(
                env,
                *input,
                &options.clone().unwrap_or_default(),
                PoolKind::Average,
            )?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::MaxPool2d {
            input,
            options,
            outputs,
            ..
        } => {
            let out = pool2d_device(
                env,
                *input,
                &options.clone().unwrap_or_default(),
                PoolKind::Max,
            )?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::ReduceSum {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::Sum)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::ReduceMean {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::Mean)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::ReduceMax {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::Max)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::ReduceMin {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::Min)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::ReduceProduct {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::Product)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::ReduceL1 {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::L1)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::ReduceL2 {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::L2)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::ReduceLogSum {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::LogSum)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::ReduceLogSumExp {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::LogSumExp)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::ReduceSumSquare {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::SumSquare)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::BatchNormalization {
            input,
            mean,
            variance,
            options,
            outputs,
            ..
        } => {
            let opts = options.clone().unwrap_or_default();
            let scale = opts.scale.map(|id| env.get_host(id)).transpose()?;
            let bias = opts.bias.map(|id| env.get_host(id)).transpose()?;
            let out = batch_normalization(
                &env.get_host(*input)?,
                &env.get_host(*mean)?,
                &env.get_host(*variance)?,
                &opts,
                scale.as_ref(),
                bias.as_ref(),
            )?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::InstanceNormalization {
            input,
            options,
            outputs,
            ..
        } => {
            let opts = options.clone().unwrap_or_default();
            let scale = opts.scale.map(|id| env.get_host(id)).transpose()?;
            let bias = opts.bias.map(|id| env.get_host(id)).transpose()?;
            let out = instance_normalization(
                &env.get_host(*input)?,
                &opts,
                scale.as_ref(),
                bias.as_ref(),
            )?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::LayerNormalization {
            input,
            options,
            outputs,
            ..
        } => {
            let opts = options.clone().unwrap_or_default();
            let scale = opts.scale.map(|id| env.get_host(id)).transpose()?;
            let bias = opts.bias.map(|id| env.get_host(id)).transpose()?;
            let out =
                layer_normalization(&env.get_host(*input)?, &opts, scale.as_ref(), bias.as_ref())?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::Identity { input, outputs, .. } => {
            let out = env.get(*input)?.clone();
            insert_runtime(env, operand_types, outputs, out)?;
        }
        Operation::Cos { input, outputs, .. } => {
            unary_device_or_f16(
                env,
                operand_types,
                *input,
                outputs[0],
                outputs,
                UnaryDeviceOp::Cos,
                f32::cos,
            )?;
        }
        Operation::Sin { input, outputs, .. } => {
            unary_device_or_f16(
                env,
                operand_types,
                *input,
                outputs[0],
                outputs,
                UnaryDeviceOp::Sin,
                f32::sin,
            )?;
        }
        Operation::Tan { input, outputs, .. } => {
            unary_device_or_f16(
                env,
                operand_types,
                *input,
                outputs[0],
                outputs,
                UnaryDeviceOp::Tan,
                f32::tan,
            )?;
        }
        Operation::Erf { input, outputs, .. } => {
            insert_output(env, operand_types, outputs, erf(&env.get_host(*input)?))?;
        }
        Operation::Gelu { input, outputs, .. } => {
            if use_f16_arithmetic(&env.dtypes, &[*input], outputs[0], operand_types) {
                insert_output(env, operand_types, outputs, gelu(&env.get_host(*input)?))?;
            } else {
                let out = env.unary(*input, UnaryDeviceOp::Gelu)?;
                insert_tensor(env, operand_types, outputs, out)?;
            }
        }
        Operation::Reciprocal { input, outputs, .. } => {
            insert_output(
                env,
                operand_types,
                outputs,
                reciprocal(&env.get_host(*input)?),
            )?;
        }
        Operation::Sign { input, outputs, .. } => {
            insert_output(env, operand_types, outputs, sign(&env.get_host(*input)?))?;
        }
        Operation::Softplus { input, outputs, .. } => {
            unary_device_or_f16(
                env,
                operand_types,
                *input,
                outputs[0],
                outputs,
                UnaryDeviceOp::Softplus,
                |x| (1.0 + x.exp()).ln(),
            )?;
        }
        Operation::Softsign { input, outputs, .. } => {
            unary_device_or_f16(
                env,
                operand_types,
                *input,
                outputs[0],
                outputs,
                UnaryDeviceOp::Softsign,
                |x| x / (1.0 + x.abs()),
            )?;
        }
        Operation::IsNaN { input, outputs, .. } => {
            insert_output(env, operand_types, outputs, is_nan(&env.get_host(*input)?))?;
        }
        Operation::IsInfinite { input, outputs, .. } => {
            insert_output(
                env,
                operand_types,
                outputs,
                is_infinite(&env.get_host(*input)?),
            )?;
        }
        Operation::RoundEven { input, outputs, .. } => {
            insert_output(
                env,
                operand_types,
                outputs,
                round_even(&env.get_host(*input)?),
            )?;
        }
        Operation::Squeeze {
            input,
            options,
            outputs,
            ..
        } => {
            let out = squeeze_device(env, *input, &options.clone().unwrap_or_default())?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::Unsqueeze {
            input,
            options,
            outputs,
            ..
        } => {
            let out = unsqueeze_device(env, *input, &options.clone().unwrap_or_default())?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::Shape { input, outputs, .. } => {
            let out = shape(&env.get_host(*input)?);
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::Tile {
            input,
            repetitions,
            outputs,
            ..
        } => {
            let out = tile(&env.get_host(*input)?, repetitions)?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::CumulativeSum {
            input,
            axis,
            options,
            outputs,
            ..
        } => {
            let out = cumulative_sum(
                &env.get_host(*input)?,
                *axis,
                &options.clone().unwrap_or_default(),
            )?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::NotEqual { a, b, outputs, .. } => {
            let out = env.binary_host(*a, *b, not_equal)?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::LogicalAnd { a, b, outputs, .. } => {
            let out = env.binary_host(*a, *b, logical_and)?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::LogicalOr { a, b, outputs, .. } => {
            let out = env.binary_host(*a, *b, logical_or)?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::LogicalXor { a, b, outputs, .. } => {
            let out = env.binary_host(*a, *b, logical_xor)?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::Prelu {
            input,
            slope,
            outputs,
            ..
        } => {
            let out = prelu(&env.get_host(*input)?, &env.get_host(*slope)?)?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::Where {
            condition,
            true_value,
            false_value,
            outputs,
            ..
        } => {
            let out = where_op(
                &env.get_host(*condition)?,
                &env.get_host(*true_value)?,
                &env.get_host(*false_value)?,
            )?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::Gemm {
            a,
            b,
            options,
            outputs,
            ..
        } => {
            let opts = options.clone().unwrap_or_default();
            let out = gemm_device(env, *a, *b, opts.c, &opts)?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::ArgMin {
            input,
            axis,
            options,
            outputs,
            ..
        } => {
            let opts = options.clone().unwrap_or_default();
            let out = arg_min(&env.get_host(*input)?, *axis, &opts)?;
            let out_dt = if opts.output_data_type == MLOperandDataType::Float32
                || opts.output_data_type == MLOperandDataType::Float16
            {
                ml_operand_to_data_type(opts.output_data_type)
            } else {
                operand_types
                    .get(&outputs[0])
                    .copied()
                    .unwrap_or(DataType::Int32)
            };
            insert_output_typed(env, outputs, out, out_dt)?;
        }
        Operation::ArgMax {
            input,
            axis,
            options,
            outputs,
            ..
        } => {
            let opts = options.clone().unwrap_or_default();
            let out = arg_max(&env.get_host(*input)?, *axis, &opts)?;
            let out_dt = if opts.output_data_type == MLOperandDataType::Float32
                || opts.output_data_type == MLOperandDataType::Float16
            {
                ml_operand_to_data_type(opts.output_data_type)
            } else {
                operand_types
                    .get(&outputs[0])
                    .copied()
                    .unwrap_or(DataType::Int32)
            };
            insert_output_typed(env, outputs, out, out_dt)?;
        }
        Operation::Pad {
            input,
            beginning_padding,
            ending_padding,
            options,
            outputs,
            ..
        } => {
            let out = pad(
                &env.get_host(*input)?,
                beginning_padding,
                ending_padding,
                &options.clone().unwrap_or_default(),
            )?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::Triangular {
            input,
            options,
            outputs,
            ..
        } => {
            let out = triangular(&env.get_host(*input)?, &options.clone().unwrap_or_default())?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::GatherElements {
            input,
            indices,
            options,
            outputs,
            ..
        } => {
            let axis = options.as_ref().map(|o| o.axis).unwrap_or(0);
            let out = gather_elements(&env.get_host(*input)?, &env.get_host(*indices)?, axis)?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::GatherND {
            input,
            indices,
            outputs,
            ..
        } => {
            let out = gather_nd(&env.get_host(*input)?, &env.get_host(*indices)?)?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::ScatterElements {
            input,
            indices,
            updates,
            options,
            outputs,
            ..
        } => {
            let out = scatter_elements(
                &env.get_host(*input)?,
                &env.get_host(*indices)?,
                &env.get_host(*updates)?,
                &options.clone().unwrap_or_default(),
            )?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::ScatterND {
            input,
            indices,
            updates,
            outputs,
            ..
        } => {
            let out = scatter_nd(
                &env.get_host(*input)?,
                &env.get_host(*indices)?,
                &env.get_host(*updates)?,
            )?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::L2Pool2d {
            input,
            options,
            outputs,
            ..
        } => {
            let out = l2_pool2d(&env.get_host(*input)?, &options.clone().unwrap_or_default())?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::GlobalAveragePool {
            input,
            options,
            outputs,
            ..
        } => {
            let out = global_pool_device(
                env,
                *input,
                &options.clone().unwrap_or_default(),
                PoolKind::Average,
            )?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::GlobalMaxPool {
            input,
            options,
            outputs,
            ..
        } => {
            let out = global_pool_device(
                env,
                *input,
                &options.clone().unwrap_or_default(),
                PoolKind::Max,
            )?;
            insert_tensor(env, operand_types, outputs, out)?;
        }
        Operation::QuantizeLinear {
            input,
            scale,
            zero_point,
            outputs,
            ..
        } => {
            let output_id =
                outputs
                    .first()
                    .copied()
                    .ok_or_else(|| GraphError::BurnRuntimeFailed {
                        reason: "quantizeLinear produced no output operand".to_string(),
                    })?;
            let output_dtype = operand_types
                .get(&output_id)
                .copied()
                .unwrap_or(DataType::Int8);
            let zp = zero_point.map(|id| env.get_host(id)).transpose()?;
            let out = quantize_linear(
                &env.get_host(*input)?,
                &env.get_host(*scale)?,
                zp.as_ref(),
                output_dtype,
            )?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::DequantizeLinear {
            input,
            scale,
            zero_point,
            outputs,
            ..
        } => {
            let zp = zero_point.map(|id| env.get_host(id)).transpose()?;
            let out =
                dequantize_linear(&env.get_host(*input)?, &env.get_host(*scale)?, zp.as_ref())?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::Resample2d {
            input,
            options,
            outputs,
            ..
        } => {
            let out = resample2d(&env.get_host(*input)?, &options.clone().unwrap_or_default())?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::Reverse {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reverse(&env.get_host(*input)?, &options.clone().unwrap_or_default())?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::GruCell {
            input,
            weight,
            recurrence,
            hidden_state,
            hidden_size,
            options,
            outputs,
            ..
        } => {
            let opts = options.clone().unwrap_or_default();
            let bias = opts.bias.map(|id| env.get_host(id)).transpose()?;
            let recurrent_bias = opts.recurrent_bias.map(|id| env.get_host(id)).transpose()?;
            let out = gru_cell(
                &env.get_host(*input)?,
                &env.get_host(*weight)?,
                &env.get_host(*recurrence)?,
                &env.get_host(*hidden_state)?,
                *hidden_size,
                &opts,
                bias.as_ref(),
                recurrent_bias.as_ref(),
            )?;
            insert_output(env, operand_types, outputs, out)?;
        }
        Operation::Gru {
            input,
            weight,
            recurrence,
            steps,
            hidden_size,
            options,
            outputs,
            ..
        } => {
            let opts = options.clone().unwrap_or_default();
            let bias = opts.bias.map(|id| env.get_host(id)).transpose()?;
            let recurrent_bias = opts.recurrent_bias.map(|id| env.get_host(id)).transpose()?;
            let initial_hidden = opts
                .initial_hidden_state
                .map(|id| env.get_host(id))
                .transpose()?;
            let (sequence, final_h) = gru(
                &env.get_host(*input)?,
                &env.get_host(*weight)?,
                &env.get_host(*recurrence)?,
                *steps,
                *hidden_size,
                &opts,
                bias.as_ref(),
                recurrent_bias.as_ref(),
                initial_hidden.as_ref(),
            )?;
            insert_gru_lstm_outputs(
                env,
                operand_types,
                outputs,
                opts.return_sequence,
                num_rnn_directions(&opts.direction),
                sequence,
                final_h,
            )?;
        }
        Operation::LstmCell {
            input,
            weight,
            recurrence,
            hidden_state,
            cell_state,
            options,
            outputs,
            ..
        } => {
            let opts = options.clone().unwrap_or_default();
            let bias = opts.bias.map(|id| env.get_host(id)).transpose()?;
            let recurrent_bias = opts.recurrent_bias.map(|id| env.get_host(id)).transpose()?;
            let peephole = opts
                .peephole_weight
                .map(|id| env.get_host(id))
                .transpose()?;
            let (h_out, c_out) = lstm_cell(
                &env.get_host(*input)?,
                &env.get_host(*weight)?,
                &env.get_host(*recurrence)?,
                &env.get_host(*hidden_state)?,
                &env.get_host(*cell_state)?,
                &opts,
                bias.as_ref(),
                recurrent_bias.as_ref(),
                peephole.as_ref(),
            )?;
            if outputs.len() >= 2 {
                insert_multi_outputs(env, operand_types, outputs, vec![h_out, c_out])?;
            } else {
                insert_output(env, operand_types, outputs, h_out)?;
            }
        }
        Operation::Lstm {
            input,
            weight,
            recurrence,
            options,
            outputs,
            ..
        } => {
            let opts = options.clone().unwrap_or_default();
            let bias = opts.bias.map(|id| env.get_host(id)).transpose()?;
            let recurrent_bias = opts.recurrent_bias.map(|id| env.get_host(id)).transpose()?;
            let peephole = opts
                .peephole_weight
                .map(|id| env.get_host(id))
                .transpose()?;
            let initial_hidden = opts
                .initial_hidden_state
                .map(|id| env.get_host(id))
                .transpose()?;
            let initial_cell = opts
                .initial_cell_state
                .map(|id| env.get_host(id))
                .transpose()?;
            let (h_seq, final_h, c_seq, final_c) = lstm(
                &env.get_host(*input)?,
                &env.get_host(*weight)?,
                &env.get_host(*recurrence)?,
                &opts,
                bias.as_ref(),
                recurrent_bias.as_ref(),
                peephole.as_ref(),
                initial_hidden.as_ref(),
                initial_cell.as_ref(),
            )?;
            insert_lstm_outputs(
                env,
                operand_types,
                outputs,
                opts.return_sequence,
                num_rnn_directions(&opts.direction),
                h_seq,
                final_h,
                final_c,
            )?;
        }
        Operation::Constant { .. } => {
            return Err(op_err(
                op,
                "constant operands must be loaded from plan constants",
            ));
        }
    }
    Ok(())
}

fn resolve_mldims(dims: &[MLDimension], numel: usize) -> Result<Vec<usize>, GraphError> {
    let mut shape: Vec<usize> = mldimensions_static_or_max(dims)
        .iter()
        .map(|&d| d as usize)
        .collect();
    if shape.iter().any(|&d| d == 0) {
        let known: usize = shape.iter().filter(|&&d| d != 0).product();
        let inferred = if known == 0 { numel } else { numel / known };
        for dim in &mut shape {
            if *dim == 0 {
                *dim = inferred;
            }
        }
    }
    Ok(shape)
}

fn hard_sigmoid_params(options: &Option<MLHardSigmoidOptions>) -> (f32, f32) {
    options
        .as_ref()
        .map(|o| (o.alpha as f32, o.beta as f32))
        .unwrap_or((0.2, 0.5))
}

fn linear_params(options: &Option<MLLinearOptions>) -> (f32, f32) {
    options
        .as_ref()
        .map(|o| (o.alpha as f32, o.beta as f32))
        .unwrap_or((1.0, 0.0))
}

fn clamp_f32(x: f32, min_v: f32, max_v: f32) -> f32 {
    let mut v = x;
    if !min_v.is_nan() {
        v = v.max(min_v);
    }
    if !max_v.is_nan() {
        v = v.min(max_v);
    }
    v
}

fn clamp_bounds(options: &Option<MLClampOptions>) -> (f32, f32) {
    let min_v = options
        .as_ref()
        .and_then(|o| o.min_value.as_ref())
        .and_then(json_to_f32)
        .unwrap_or(f32::NEG_INFINITY);
    let max_v = options
        .as_ref()
        .and_then(|o| o.max_value.as_ref())
        .and_then(json_to_f32)
        .unwrap_or(f32::INFINITY);
    (min_v, max_v)
}

fn clamp_i64_bounds(options: &Option<MLClampOptions>) -> (Option<i64>, Option<i64>) {
    (
        options
            .as_ref()
            .and_then(|o| o.min_value.as_ref())
            .and_then(cast_mlnumber_to_i64),
        options
            .as_ref()
            .and_then(|o| o.max_value.as_ref())
            .and_then(cast_mlnumber_to_i64),
    )
}

fn clamp_u64_bounds(options: &Option<MLClampOptions>) -> (Option<u64>, Option<u64>) {
    (
        options
            .as_ref()
            .and_then(|o| o.min_value.as_ref())
            .and_then(cast_mlnumber_to_u64),
        options
            .as_ref()
            .and_then(|o| o.max_value.as_ref())
            .and_then(cast_mlnumber_to_u64),
    )
}

fn clamp_i64(x: i64, min_v: Option<i64>, max_v: Option<i64>) -> i64 {
    let mut v = x;
    if let Some(min) = min_v {
        v = v.max(min);
    }
    if let Some(max) = max_v {
        v = v.min(max);
    }
    v
}

fn clamp_u64(x: u64, min_v: Option<u64>, max_v: Option<u64>) -> u64 {
    let mut v = x;
    if let Some(min) = min_v {
        v = v.max(min);
    }
    if let Some(max) = max_v {
        v = v.min(max);
    }
    v
}

fn cast_mlnumber_to_i64(v: &serde_json::Value) -> Option<i64> {
    if v.is_null() {
        return None;
    }
    if let Some(s) = v.as_str() {
        if let Ok(i) = s.parse::<i128>() {
            let lower = i128::from(i64::MIN);
            let upper = i128::from(i64::MAX);
            return Some(i.clamp(lower, upper) as i64);
        }
    }
    let x = mlnumber_as_f64(v)?;
    Some(convert_to_int_f64(x, 64, true))
}

fn cast_mlnumber_to_u64(v: &serde_json::Value) -> Option<u64> {
    if v.is_null() {
        return None;
    }
    if let Some(s) = v.as_str() {
        if let Ok(i) = s.parse::<i128>() {
            let clamped = i.clamp(0, i128::from(u64::MAX));
            return Some(clamped as u64);
        }
        if let Ok(u) = s.parse::<u128>() {
            return Some(u.min(u128::from(u64::MAX)) as u64);
        }
    }
    let x = mlnumber_as_f64(v)?;
    Some(convert_to_int_f64(x, 64, false) as u64)
}

fn mlnumber_as_f64(v: &serde_json::Value) -> Option<f64> {
    if let Some(s) = v.as_str() {
        return s.parse::<f64>().ok();
    }
    v.as_f64()
        .or_else(|| v.as_i64().map(|i| i as f64))
        .or_else(|| v.as_u64().map(|u| u as f64))
}

fn convert_to_int_f64(x: f64, bit_length: u32, signed: bool) -> i64 {
    let mut x = if x == 0.0 && x.is_sign_negative() {
        0.0
    } else {
        x
    };
    if x.is_nan() {
        return 0;
    }
    let (lower, upper) = if signed {
        (
            -(2f64.powi((bit_length - 1) as i32)),
            2f64.powi((bit_length - 1) as i32) - 1.0,
        )
    } else {
        (0.0, 2f64.powi(bit_length as i32) - 1.0)
    };
    x = x.clamp(lower, upper);
    x.trunc() as i64
}

fn binary_device_or_f16<B: Backend>(
    env: &mut TensorEnv<B>,
    operand_types: &HashMap<u32, DataType>,
    a: u32,
    b: u32,
    outputs: &[u32],
    device_op: DeviceBinaryOp,
    f16_kernel: impl FnOnce(&HostArray, &HostArray) -> Result<HostArray, GraphError>,
) -> Result<(), GraphError> {
    if use_f16_arithmetic(&env.dtypes, &[a, b], outputs[0], operand_types) {
        let out = env.binary_host(a, b, f16_kernel)?;
        insert_output(env, operand_types, outputs, out)?;
    } else {
        let out = env.binary_broadcast(a, b, device_op)?;
        insert_tensor(env, operand_types, outputs, out)?;
    }
    Ok(())
}

fn unary_device_or_f16<B: Backend>(
    env: &mut TensorEnv<B>,
    operand_types: &HashMap<u32, DataType>,
    input: u32,
    output: u32,
    outputs: &[u32],
    device_op: UnaryDeviceOp,
    f16_map: impl Fn(f32) -> f32,
) -> Result<(), GraphError> {
    if use_f16_arithmetic(&env.dtypes, &[input], output, operand_types) {
        insert_output(
            env,
            operand_types,
            outputs,
            unary_output(env, operand_types, input, output, f16_map)?,
        )?;
    } else {
        let out = env.unary(input, device_op)?;
        insert_tensor(env, operand_types, outputs, out)?;
    }
    Ok(())
}

fn insert_runtime<B: Backend>(
    env: &mut TensorEnv<B>,
    operand_types: &HashMap<u32, DataType>,
    outputs: &[u32],
    tensor: RuntimeTensor<B>,
) -> Result<(), GraphError> {
    insert_tensor(env, operand_types, outputs, tensor)
}

fn json_to_f32(v: &serde_json::Value) -> Option<f32> {
    if v.is_null() {
        return Some(f32::NAN);
    }
    if let Some(s) = v.as_str() {
        return match s {
            "Infinity" => Some(f32::INFINITY),
            "-Infinity" => Some(f32::NEG_INFINITY),
            "NaN" => Some(f32::NAN),
            _ => s.parse::<f64>().ok().map(|f| f as f32),
        };
    }
    v.as_f64()
        .map(|f| f as f32)
        .or_else(|| v.as_i64().map(|i| i as f32))
}

fn ml_operand_to_data_type(dt: MLOperandDataType) -> DataType {
    use MLOperandDataType as T;
    match dt {
        T::Float32 => DataType::Float32,
        T::Float16 => DataType::Float16,
        T::Int32 => DataType::Int32,
        T::Uint32 => DataType::Uint32,
        T::Int8 => DataType::Int8,
        T::Uint8 => DataType::Uint8,
        T::Int64 => DataType::Int64,
        T::Uint64 => DataType::Uint64,
    }
}

fn unary_output<B: Backend>(
    env: &TensorEnv<B>,
    operand_types: &HashMap<u32, DataType>,
    input: u32,
    output: u32,
    f: impl Fn(f32) -> f32,
) -> Result<HostArray, GraphError> {
    let arr = env.get_host(input)?;
    let f16 = use_f16_arithmetic(&env.dtypes, &[input], output, operand_types);
    Ok(if f16 {
        arr.map_unary(|x| round_f16(f(x)))
    } else {
        arr.map_unary(f)
    })
}

fn insert_output<B: Backend>(
    env: &mut TensorEnv<B>,
    operand_types: &HashMap<u32, DataType>,
    outputs: &[u32],
    mut value: HostArray,
) -> Result<(), GraphError> {
    let id = outputs
        .first()
        .copied()
        .ok_or_else(|| GraphError::BurnRuntimeFailed {
            reason: "operation produced no output operand".to_string(),
        })?;
    let out_dt = operand_types.get(&id).copied().unwrap_or(DataType::Float32);
    insert_host_output(env, operand_types, outputs, value, out_dt)
}

fn insert_host_output<B: Backend>(
    env: &mut TensorEnv<B>,
    operand_types: &HashMap<u32, DataType>,
    outputs: &[u32],
    mut value: HostArray,
    data_type: DataType,
) -> Result<(), GraphError> {
    let id = outputs
        .first()
        .copied()
        .ok_or_else(|| GraphError::BurnRuntimeFailed {
            reason: "operation produced no output operand".to_string(),
        })?;
    if data_type == DataType::Float16 {
        round_f16_slice(&mut value.data);
    }
    let int64_data = value.i64_data.take();
    let uint64_data = value.u64_data.take();
    let tensor = RuntimeTensor::from_host_array(value, &env.device)?;
    env.insert_with_integer_sidecar(id, data_type, tensor, int64_data, uint64_data);
    Ok(())
}

fn insert_output_typed<B: Backend>(
    env: &mut TensorEnv<B>,
    outputs: &[u32],
    mut value: HostArray,
    data_type: DataType,
) -> Result<(), GraphError> {
    let id = outputs
        .first()
        .copied()
        .ok_or_else(|| GraphError::BurnRuntimeFailed {
            reason: "operation produced no output operand".to_string(),
        })?;
    if data_type == DataType::Float16 {
        round_f16_slice(&mut value.data);
    }
    let int64_data = value.i64_data.take();
    let uint64_data = value.u64_data.take();
    let tensor = RuntimeTensor::from_host_array(value, &env.device)?;
    env.insert_with_integer_sidecar(id, data_type, tensor, int64_data, uint64_data);
    Ok(())
}

fn insert_multi_outputs<B: Backend>(
    env: &mut TensorEnv<B>,
    operand_types: &HashMap<u32, DataType>,
    outputs: &[u32],
    values: Vec<HostArray>,
) -> Result<(), GraphError> {
    if values.len() != outputs.len() {
        return Err(GraphError::BurnRuntimeFailed {
            reason: format!(
                "operation produced {} values for {} outputs",
                values.len(),
                outputs.len()
            ),
        });
    }
    for (id, value) in outputs.iter().zip(values) {
        let out_dt = operand_types.get(id).copied().unwrap_or(DataType::Float32);
        let int64_data = value.i64_data.clone();
        let uint64_data = value.u64_data.clone();
        let tensor = RuntimeTensor::from_host_array(value, &env.device)?;
        env.insert_with_integer_sidecar(*id, out_dt, tensor, int64_data, uint64_data);
    }
    Ok(())
}

fn num_rnn_directions(direction: &str) -> usize {
    if direction.eq_ignore_ascii_case("both") {
        2
    } else {
        1
    }
}

fn insert_gru_lstm_outputs<B: Backend>(
    env: &mut TensorEnv<B>,
    operand_types: &HashMap<u32, DataType>,
    outputs: &[u32],
    return_sequence: bool,
    num_directions: usize,
    sequence: Option<HostArray>,
    final_state: HostArray,
) -> Result<(), GraphError> {
    if outputs.is_empty() {
        return Err(GraphError::BurnRuntimeFailed {
            reason: "RNN operation produced no output operand".to_string(),
        });
    }
    let final_state = format_rnn_state_nd(&final_state, num_directions)?;
    if return_sequence {
        let seq = sequence.ok_or_else(|| GraphError::BurnRuntimeFailed {
            reason: "RNN returnSequence expected sequence output".to_string(),
        })?;
        let seq = format_rnn_hidden_sequence(&seq, num_directions)?;
        if outputs.len() >= 2 {
            insert_multi_outputs(env, operand_types, &outputs[..2], vec![final_state, seq])
        } else {
            insert_output(env, operand_types, outputs, final_state)
        }
    } else {
        insert_output(env, operand_types, outputs, final_state)
    }
}

fn insert_lstm_outputs<B: Backend>(
    env: &mut TensorEnv<B>,
    operand_types: &HashMap<u32, DataType>,
    outputs: &[u32],
    return_sequence: bool,
    num_directions: usize,
    h_seq: Option<HostArray>,
    final_h: HostArray,
    final_c: HostArray,
) -> Result<(), GraphError> {
    if outputs.is_empty() {
        return Err(GraphError::BurnRuntimeFailed {
            reason: "LSTM operation produced no output operand".to_string(),
        });
    }
    let final_h = format_rnn_state_nd(&final_h, num_directions)?;
    let final_c = format_rnn_state_nd(&final_c, num_directions)?;
    if return_sequence {
        let h_seq = h_seq.ok_or_else(|| GraphError::BurnRuntimeFailed {
            reason: "LSTM returnSequence expected hidden sequence output".to_string(),
        })?;
        let h_seq = format_rnn_hidden_sequence(&h_seq, num_directions)?;
        if outputs.len() >= 3 {
            insert_multi_outputs(
                env,
                operand_types,
                &outputs[..3],
                vec![final_h, final_c, h_seq],
            )
        } else if outputs.len() >= 2 {
            insert_multi_outputs(env, operand_types, &outputs[..2], vec![final_h, final_c])
        } else {
            insert_output(env, operand_types, outputs, final_h)
        }
    } else if outputs.len() >= 2 {
        insert_multi_outputs(env, operand_types, &outputs[..2], vec![final_h, final_c])
    } else {
        insert_output(env, operand_types, outputs, final_h)
    }
}

fn op_err(op: &Operation, reason: &str) -> GraphError {
    GraphError::BurnRuntimeFailed {
        reason: format!("{}: {reason}", op.op_type()),
    }
}

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

//! Execute WebNN [`Operation`] values on Burn device tensors.

use std::collections::HashMap;

use burn_tensor::backend::Backend;

use super::options::float_number;
use super::outputs::OperationOutputs;
use crate::burn::IOBinding;
use crate::error::GraphError;
use crate::graph::DataType;
use crate::operator_options::{
    MLClampOptions, MLDimension, MLEluOptions, MLHardSigmoidOptions, MLLinearOptions,
    mldimensions_static_or_max,
};
use crate::operators::Operation;

use super::device_ops::{
    conv2d_device, gemm_device, host_kernel, insert_tensor, pool2d_device, reduce_device,
};
use super::f16::{
    f16_add, f16_div, f16_mul, f16_neg, f16_sqrt, f16_sub, round_f16, use_f16_arithmetic,
};
use super::float_ops::{
    cumulative_sum, erf, gelu, l2_pool2d, prelu, reciprocal, resample2d, round_even, sign,
};
use super::host_array::{
    HostArray, PoolKind, ReduceKind, batch_normalization, conv_transpose2d, instance_normalization,
    layer_normalization, pow_broadcast,
};
use super::rnn::{format_rnn_hidden_sequence, format_rnn_state_nd, gru, gru_cell, lstm, lstm_cell};
use super::tensor_env::{DeviceBinaryOp, TensorEnv, UnaryDeviceOp};

pub fn execute_operations<B: Backend>(
    env: &mut TensorEnv<B>,
    operations: &[Operation],
    operand_types: &HashMap<u32, DataType>,
    bindings: &[IOBinding],
) -> Result<(), GraphError> {
    for op in operations {
        execute_one(env, op, operand_types, bindings)?;
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum ExecutionKind {
    Movement,
    Integer,
    Float,
}

fn execution_kind(
    op: &Operation,
    types: &HashMap<u32, DataType>,
) -> Result<ExecutionKind, GraphError> {
    let mut integer = false;
    for id in op.all_input_operands().iter().chain(op.outputs()) {
        let dtype = types
            .get(id)
            .ok_or_else(|| op_err(op, &format!("missing datatype for operand {id}")))?;
        integer |= super::typed::is_integer(*dtype);
    }
    Ok(if super::typed::movement::handles(op) {
        ExecutionKind::Movement
    } else if integer {
        ExecutionKind::Integer
    } else {
        ExecutionKind::Float
    })
}

fn execute_one<B: Backend>(
    env: &mut TensorEnv<B>,
    op: &Operation,
    operand_types: &HashMap<u32, DataType>,
    bindings: &[IOBinding],
) -> Result<(), GraphError> {
    let kind = execution_kind(op, operand_types)?;
    let mut outputs = OperationOutputs::new(env, op, operand_types, bindings)?;
    match kind {
        ExecutionKind::Movement => super::typed::movement::execute(&mut outputs, op)?,
        ExecutionKind::Integer => super::typed::execute(&mut outputs, op)?,
        ExecutionKind::Float => execute_float(&mut outputs, op, operand_types)?,
    }
    outputs.finish()
}

fn execute_float<B: Backend>(
    env: &mut OperationOutputs<'_, B>,
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
        Operation::Pow { a, b, outputs, .. } => {
            let out = host_kernel(env, |env| {
                pow_broadcast(&env.get_host(*a)?, &env.get_host(*b)?)
            })?;
            insert_tensor(env, outputs, out)?;
        }

        Operation::Min { a, b, outputs, .. } => {
            let out = env.binary_broadcast(*a, *b, DeviceBinaryOp::Min)?;
            insert_tensor(env, outputs, out)?;
        }
        Operation::Max { a, b, outputs, .. } => {
            let out = env.binary_broadcast(*a, *b, DeviceBinaryOp::Max)?;
            insert_tensor(env, outputs, out)?;
        }
        Operation::Matmul { a, b, outputs, .. } => {
            let out = env.matmul(*a, *b)?;
            insert_tensor(env, outputs, out)?;
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
            unary_device_or_f16(
                env,
                operand_types,
                *input,
                outputs[0],
                outputs,
                UnaryDeviceOp::Neg,
                f16_neg,
            )?;
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
            if use_f16_arithmetic(operand_types, &[*input], outputs[0]) {
                let alpha_f32 = alpha as f32;
                insert_output(
                    env,
                    outputs,
                    unary_output(env, operand_types, *input, outputs[0], move |x| {
                        if x >= 0.0 { x } else { alpha_f32 * x }
                    })?,
                )?;
            } else {
                let out = env.unary(*input, UnaryDeviceOp::LeakyRelu { slope: alpha })?;
                insert_tensor(env, outputs, out)?;
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
            if use_f16_arithmetic(operand_types, &[*input], outputs[0]) {
                let alpha_f32 = alpha as f32;
                insert_output(
                    env,
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
                insert_tensor(env, outputs, out)?;
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
            if use_f16_arithmetic(operand_types, &[*input], outputs[0]) {
                insert_output(
                    env,
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
                insert_tensor(env, outputs, out)?;
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
                outputs,
                unary_output(env, operand_types, *input, outputs[0], move |x| {
                    alpha * x + beta
                })?,
            )?;
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
            insert_tensor(env, outputs, out)?;
        }
        Operation::Clamp {
            input,
            options,
            outputs,
            ..
        } => {
            let (min_v, max_v) = clamp_bounds(options);
            let out = unary_output(env, operand_types, *input, outputs[0], |x| {
                clamp_f32(x, min_v, max_v)
            })?;
            insert_output(env, outputs, out)?;
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
            insert_tensor(env, outputs, out)?;
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
            insert_output(env, outputs, out)?;
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
            insert_tensor(env, outputs, out)?;
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
            insert_tensor(env, outputs, out)?;
        }
        Operation::ReduceSum {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::Sum)?;
            insert_tensor(env, outputs, out)?;
        }
        Operation::ReduceMean {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::Mean)?;
            insert_tensor(env, outputs, out)?;
        }
        Operation::ReduceMax {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::Max)?;
            insert_tensor(env, outputs, out)?;
        }
        Operation::ReduceMin {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::Min)?;
            insert_tensor(env, outputs, out)?;
        }
        Operation::ReduceProduct {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::Product)?;
            insert_tensor(env, outputs, out)?;
        }
        Operation::ReduceL1 {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::L1)?;
            insert_tensor(env, outputs, out)?;
        }
        Operation::ReduceL2 {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::L2)?;
            insert_tensor(env, outputs, out)?;
        }
        Operation::ReduceLogSum {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::LogSum)?;
            insert_tensor(env, outputs, out)?;
        }
        Operation::ReduceLogSumExp {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::LogSumExp)?;
            insert_tensor(env, outputs, out)?;
        }
        Operation::ReduceSumSquare {
            input,
            options,
            outputs,
            ..
        } => {
            let out = reduce_device(env, *input, options, ReduceKind::SumSquare)?;
            insert_tensor(env, outputs, out)?;
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
            insert_output(env, outputs, out)?;
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
            insert_output(env, outputs, out)?;
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
            insert_output(env, outputs, out)?;
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
            insert_output(env, outputs, erf(&env.get_host(*input)?))?;
        }
        Operation::Gelu { input, outputs, .. } => {
            if use_f16_arithmetic(operand_types, &[*input], outputs[0]) {
                insert_output(env, outputs, gelu(&env.get_host(*input)?))?;
            } else {
                let out = env.unary(*input, UnaryDeviceOp::Gelu)?;
                insert_tensor(env, outputs, out)?;
            }
        }
        Operation::Reciprocal { input, outputs, .. } => {
            insert_output(env, outputs, reciprocal(&env.get_host(*input)?))?;
        }
        Operation::Sign { input, outputs, .. } => {
            insert_output(env, outputs, sign(&env.get_host(*input)?))?;
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

        Operation::RoundEven { input, outputs, .. } => {
            insert_output(env, outputs, round_even(&env.get_host(*input)?))?;
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
            insert_output(env, outputs, out)?;
        }

        Operation::Prelu {
            input,
            slope,
            outputs,
            ..
        } => {
            let out = prelu(&env.get_host(*input)?, &env.get_host(*slope)?)?;
            insert_output(env, outputs, out)?;
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
            insert_tensor(env, outputs, out)?;
        }

        Operation::L2Pool2d {
            input,
            options,
            outputs,
            ..
        } => {
            let out = l2_pool2d(&env.get_host(*input)?, &options.clone().unwrap_or_default())?;
            insert_output(env, outputs, out)?;
        }

        Operation::Resample2d {
            input,
            options,
            outputs,
            ..
        } => {
            let out = resample2d(&env.get_host(*input)?, &options.clone().unwrap_or_default())?;
            insert_output(env, outputs, out)?;
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
            insert_output(env, outputs, out)?;
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
                outputs,
                opts.return_sequence,
                num_rnn_directions(&opts.direction)?,
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
                insert_multi_outputs(env, outputs, vec![h_out, c_out])?;
            } else {
                insert_output(env, outputs, h_out)?;
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
            let super::rnn::LstmOutput {
                sequence: h_seq,
                hidden: final_h,
                cell: final_c,
            } = lstm(
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
                outputs,
                opts.return_sequence,
                num_rnn_directions(&opts.direction)?,
                h_seq,
                final_h,
                final_c,
            )?;
        }
        _ => return Err(op_err(op, "operation was not routed to its typed kernel")),
    }
    Ok(())
}

pub(super) fn resolve_mldims(dims: &[MLDimension], numel: usize) -> Result<Vec<usize>, GraphError> {
    let mut shape: Vec<usize> = mldimensions_static_or_max(dims)
        .iter()
        .map(|&d| d as usize)
        .collect();
    if shape.contains(&0) {
        let known: usize = shape.iter().filter(|&&d| d != 0).product();
        let inferred = numel.checked_div(known).unwrap_or(numel);
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
        .and_then(float_number)
        .unwrap_or(f32::NEG_INFINITY);
    let max_v = options
        .as_ref()
        .and_then(|o| o.max_value.as_ref())
        .and_then(float_number)
        .unwrap_or(f32::INFINITY);
    (min_v, max_v)
}

fn binary_device_or_f16<B: Backend>(
    env: &mut OperationOutputs<'_, B>,
    operand_types: &HashMap<u32, DataType>,
    a: u32,
    b: u32,
    outputs: &[u32],
    device_op: DeviceBinaryOp,
    f16_kernel: impl FnOnce(&HostArray, &HostArray) -> Result<HostArray, GraphError>,
) -> Result<(), GraphError> {
    if use_f16_arithmetic(operand_types, &[a, b], outputs[0]) {
        let out = env.binary_host(a, b, f16_kernel)?;
        insert_output(env, outputs, out)?;
    } else {
        let out = env.binary_broadcast(a, b, device_op)?;
        insert_tensor(env, outputs, out)?;
    }
    Ok(())
}

fn unary_device_or_f16<B: Backend>(
    env: &mut OperationOutputs<'_, B>,
    operand_types: &HashMap<u32, DataType>,
    input: u32,
    output: u32,
    outputs: &[u32],
    device_op: UnaryDeviceOp,
    f16_map: impl Fn(f32) -> f32,
) -> Result<(), GraphError> {
    if use_f16_arithmetic(operand_types, &[input], output) {
        insert_output(
            env,
            outputs,
            unary_output(env, operand_types, input, output, f16_map)?,
        )?;
    } else {
        let out = env.unary(input, device_op)?;
        insert_tensor(env, outputs, out)?;
    }
    Ok(())
}

fn unary_output<B: Backend>(
    env: &TensorEnv<B>,
    operand_types: &HashMap<u32, DataType>,
    input: u32,
    output: u32,
    f: impl Fn(f32) -> f32,
) -> Result<HostArray, GraphError> {
    let arr = env.get_host(input)?;
    let f16 = use_f16_arithmetic(operand_types, &[input], output);
    Ok(if f16 {
        arr.map_unary(|x| round_f16(f(x)))
    } else {
        arr.map_unary(f)
    })
}

fn insert_output<B: Backend>(
    env: &mut OperationOutputs<'_, B>,
    outputs: &[u32],
    value: HostArray,
) -> Result<(), GraphError> {
    let id = outputs
        .first()
        .copied()
        .ok_or_else(|| GraphError::BurnRuntimeFailed {
            reason: "operation produced no output operand".to_string(),
        })?;
    env.write_host(id, value)
}

fn insert_multi_outputs<B: Backend>(
    env: &mut OperationOutputs<'_, B>,
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
        env.write_host(*id, value)?;
    }
    Ok(())
}

fn num_rnn_directions(direction: &str) -> Result<usize, GraphError> {
    use crate::operator_enums::MLRecurrentNetworkDirection;
    Ok(
        match super::options::parse_option_or_default(direction, "direction")? {
            MLRecurrentNetworkDirection::Both => 2,
            MLRecurrentNetworkDirection::Forward | MLRecurrentNetworkDirection::Backward => 1,
        },
    )
}

fn insert_gru_lstm_outputs<B: Backend>(
    env: &mut OperationOutputs<'_, B>,
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
            insert_multi_outputs(env, &outputs[..2], vec![final_state, seq])
        } else {
            insert_output(env, outputs, final_state)
        }
    } else {
        insert_output(env, outputs, final_state)
    }
}

// Mirrors the LSTM operation outputs and sequence metadata.
#[allow(clippy::too_many_arguments)]
fn insert_lstm_outputs<B: Backend>(
    env: &mut OperationOutputs<'_, B>,
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
            insert_multi_outputs(env, &outputs[..3], vec![final_h, final_c, h_seq])
        } else if outputs.len() >= 2 {
            insert_multi_outputs(env, &outputs[..2], vec![final_h, final_c])
        } else {
            insert_output(env, outputs, final_h)
        }
    } else if outputs.len() >= 2 {
        insert_multi_outputs(env, &outputs[..2], vec![final_h, final_c])
    } else {
        insert_output(env, outputs, final_h)
    }
}

fn op_err(op: &Operation, reason: &str) -> GraphError {
    GraphError::BurnRuntimeFailed {
        reason: format!("{}: {reason}", op.op_type()),
    }
}

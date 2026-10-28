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

//! Recurrent kernels; gate indices mirror the WebNN tensor layouts.
#![allow(clippy::needless_range_loop)]
use super::host_array::{HostArray, burn_err};
use crate::error::GraphError;
use crate::operator_options::{MLGruCellOptions, MLGruOptions, MLLstmCellOptions, MLLstmOptions};

use super::options::{parse_option, parse_option_or_default};
use crate::operator_enums::{
    MLGruWeightLayout as GruLayout, MLLstmWeightLayout as LstmLayout,
    MLRecurrentNetworkActivation as Activation, MLRecurrentNetworkDirection as Direction,
};

#[derive(Clone, Copy)]
struct GruConfig {
    layout: GruLayout,
    activations: [Activation; 2],
    reset_after: bool,
}
impl GruConfig {
    fn new(
        layout: &str,
        activations: Option<&[String]>,
        reset_after: bool,
    ) -> Result<Self, GraphError> {
        Ok(Self {
            layout: parse_option_or_default(layout, "GRU layout")?,
            activations: parse_activations(activations, [Activation::Sigmoid, Activation::Tanh])?,
            reset_after,
        })
    }
}
#[derive(Clone, Copy)]
struct LstmConfig {
    layout: LstmLayout,
    activations: [Activation; 3],
}
impl LstmConfig {
    fn new(layout: &str, activations: Option<&[String]>) -> Result<Self, GraphError> {
        Ok(Self {
            layout: parse_option_or_default(layout, "LSTM layout")?,
            activations: parse_activations(
                activations,
                [Activation::Sigmoid, Activation::Tanh, Activation::Tanh],
            )?,
        })
    }
}
fn parse_activations<const N: usize>(
    values: Option<&[String]>,
    mut defaults: [Activation; N],
) -> Result<[Activation; N], GraphError> {
    if let Some(values) = values {
        if values.len() > N {
            return Err(burn_err(format!(
                "expected at most {N} recurrent activations"
            )));
        }
        for (target, value) in defaults.iter_mut().zip(values) {
            *target = parse_option(value, "recurrent activation")?;
        }
    }
    Ok(defaults)
}
pub(super) struct LstmOutput {
    pub sequence: Option<HostArray>,
    pub hidden: HostArray,
    pub cell: HostArray,
}
struct LstmGates {
    input: Vec<f32>,
    output: Vec<f32>,
    forget: Vec<f32>,
    cell: Vec<f32>,
    activation: Activation,
}
fn lstm_gate_indices(layout: LstmLayout) -> [usize; 4] {
    match layout {
        LstmLayout::Ifgo => [0, 3, 1, 2],
        LstmLayout::Iofg => [0, 1, 2, 3],
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gru_cell(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    hidden_state: &HostArray,
    hidden_size: u32,
    options: &MLGruCellOptions,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
) -> Result<HostArray, GraphError> {
    let config = GruConfig::new(
        &options.layout,
        options.activations.as_deref(),
        options.reset_after,
    )?;
    gru_cell_internal(
        input,
        weight,
        recurrence,
        hidden_state,
        hidden_size as usize,
        &config,
        bias,
        recurrent_bias,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gru(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    steps: u32,
    hidden_size: u32,
    options: &MLGruOptions,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    initial_hidden: Option<&HostArray>,
) -> Result<(Option<HostArray>, HostArray), GraphError> {
    let config = GruConfig::new(
        &options.layout,
        options.activations.as_deref(),
        options.reset_after,
    )?;
    let direction = parse_option_or_default::<Direction>(&options.direction, "direction")?;
    if direction == Direction::Both {
        return gru_bidirectional(
            input,
            weight,
            recurrence,
            steps,
            hidden_size,
            options,
            &config,
            bias,
            recurrent_bias,
            initial_hidden,
        );
    }
    gru_unidirectional(
        input,
        weight,
        recurrence,
        steps,
        hidden_size,
        options,
        &config,
        direction,
        bias,
        recurrent_bias,
        initial_hidden,
    )
}

#[allow(clippy::too_many_arguments)]
fn gru_unidirectional(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    steps: u32,
    hidden_size: u32,
    options: &MLGruOptions,
    config: &GruConfig,
    direction: Direction,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    initial_hidden: Option<&HostArray>,
) -> Result<(Option<HostArray>, HostArray), GraphError> {
    let hs = hidden_size as usize;

    let (step_count, batch, input_size) = gru_sequence_dims(input, steps)?;
    let dir_idx = usize::from(direction == Direction::Backward);
    let mut h = resolve_direction_initial(initial_hidden, dir_idx, batch, hs)?
        .unwrap_or_else(|| HostArray::new(vec![batch, hs], vec![0.0; batch * hs]).unwrap());
    let mut sequence = if options.return_sequence {
        Some(HostArray::new(
            vec![step_count, batch, hs],
            vec![0.0; step_count * batch * hs],
        )?)
    } else {
        None
    };
    let step_range: Vec<usize> = if direction == Direction::Backward {
        (0..step_count).rev().collect()
    } else {
        (0..step_count).collect()
    };
    for t in step_range.iter() {
        let x_t = slice_sequence_step(input, *t, step_count, batch, input_size)?;
        h = gru_cell_internal(
            &x_t,
            weight,
            recurrence,
            &h,
            hs,
            config,
            bias,
            recurrent_bias,
            dir_idx,
        )?;
        if let Some(seq) = sequence.as_mut() {
            copy_hidden_to_sequence(seq, *t, batch, hs, &h);
        }
    }
    Ok((sequence, h))
}

#[allow(clippy::too_many_arguments)]
fn gru_bidirectional(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    steps: u32,
    hidden_size: u32,
    options: &MLGruOptions,
    config: &GruConfig,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    initial_hidden: Option<&HostArray>,
) -> Result<(Option<HostArray>, HostArray), GraphError> {
    let (seq_fwd, h_fwd) = gru_unidirectional(
        input,
        weight,
        recurrence,
        steps,
        hidden_size,
        options,
        config,
        Direction::Forward,
        bias,
        recurrent_bias,
        initial_hidden,
    )?;
    let (seq_bwd, h_bwd) = gru_unidirectional(
        input,
        weight,
        recurrence,
        steps,
        hidden_size,
        options,
        config,
        Direction::Backward,
        bias,
        recurrent_bias,
        initial_hidden,
    )?;

    let final_h = stack_rnn_states(&h_fwd, &h_bwd)?;
    let sequence = if options.return_sequence {
        let fwd = seq_fwd.ok_or_else(|| burn_err("forward GRU missing sequence".to_string()))?;
        let bwd = seq_bwd.ok_or_else(|| burn_err("backward GRU missing sequence".to_string()))?;
        Some(stack_rnn_sequences(&fwd, &bwd)?)
    } else {
        None
    };
    Ok((sequence, final_h))
}

#[allow(clippy::too_many_arguments)]
pub fn lstm_cell(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    hidden_state: &HostArray,
    cell_state: &HostArray,
    options: &MLLstmCellOptions,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    peephole: Option<&HostArray>,
) -> Result<(HostArray, HostArray), GraphError> {
    let config = LstmConfig::new(&options.layout, options.activations.as_deref())?;
    lstm_cell_internal(
        input,
        weight,
        recurrence,
        hidden_state,
        cell_state,
        &config,
        bias,
        recurrent_bias,
        peephole,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
fn lstm_cell_internal(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    hidden_state: &HostArray,
    cell_state: &HostArray,
    config: &LstmConfig,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    peephole: Option<&HostArray>,
    direction: usize,
) -> Result<(HostArray, HostArray), GraphError> {
    let hs = infer_lstm_hidden_size_from_weight(weight)?;
    let w = resolve_rnn_matrix(weight, direction)?;
    let r = resolve_rnn_matrix(recurrence, direction)?;
    let (batch, input_size) = matrix_batch_dims(input)?;
    let layout = config.layout;
    let activations = config.activations;
    let mut h = normalize_hidden(hidden_state, batch, hs)?;
    let mut c = normalize_hidden(cell_state, batch, hs)?;
    let w_bias = bias
        .map(|b| resolve_rnn_bias(b, direction, 4 * hs))
        .transpose()?;
    let r_bias = recurrent_bias
        .map(|b| resolve_rnn_bias(b, direction, 4 * hs))
        .transpose()?;
    for b in 0..batch {
        let x_row = slice_row(input, b, input_size);
        let h_row = slice_row(&h, b, hs);
        let c_row = slice_row_data(&c, b, hs);
        let w_g = linear_2d_with_bias_vec(&x_row, &w, w_bias.as_deref(), 4 * hs, input_size)?;
        let r_g = linear_2d_with_bias_vec(&h_row, &r, r_bias.as_deref(), 4 * hs, hs)?;
        let mut gates: Vec<f32> = w_g.iter().zip(r_g.iter()).map(|(&a, &b)| a + b).collect();
        if let Some(p) = peephole {
            apply_peephole(&mut gates, p, &c_row, hs, layout)?;
        }
        let LstmGates {
            input: i,
            output: o,
            forget: f,
            cell: g,
            activation: act_c,
        } = split_lstm_gates(&gates, hs, layout, activations)?;
        let c_new: Vec<f32> = f
            .iter()
            .zip(i.iter().zip(g.iter().zip(c_row.iter())))
            .map(|(&fi, (&ii, (&gi, &ci)))| fi * ci + ii * gi)
            .collect();
        let h_new: Vec<f32> = o
            .iter()
            .zip(c_new.iter())
            .map(|(&oi, &ci)| oi * apply_activation(act_c, ci))
            .collect();
        write_row(&mut c, b, hs, &c_new);
        write_row(&mut h, b, hs, &h_new);
    }
    Ok((h, c))
}

#[allow(clippy::too_many_arguments)]
pub fn lstm(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    options: &MLLstmOptions,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    peephole: Option<&HostArray>,
    initial_hidden: Option<&HostArray>,
    initial_cell: Option<&HostArray>,
) -> Result<LstmOutput, GraphError> {
    let config = LstmConfig::new(&options.layout, options.activations.as_deref())?;
    let direction = parse_option_or_default::<Direction>(&options.direction, "direction")?;
    if direction == Direction::Both {
        return lstm_bidirectional(
            input,
            weight,
            recurrence,
            options,
            &config,
            bias,
            recurrent_bias,
            peephole,
            initial_hidden,
            initial_cell,
        );
    }
    lstm_unidirectional(
        input,
        weight,
        recurrence,
        options,
        &config,
        direction,
        bias,
        recurrent_bias,
        peephole,
        initial_hidden,
        initial_cell,
    )
}

#[allow(clippy::too_many_arguments)]
fn lstm_unidirectional(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    options: &MLLstmOptions,
    config: &LstmConfig,
    direction: Direction,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    peephole: Option<&HostArray>,
    initial_hidden: Option<&HostArray>,
    initial_cell: Option<&HostArray>,
) -> Result<LstmOutput, GraphError> {
    let hs = infer_lstm_hidden_size_from_weight(weight)?;

    let (step_count, batch, input_size) = lstm_sequence_dims(input)?;
    let dir_idx = if direction == Direction::Backward {
        1
    } else {
        0
    };
    let mut h = resolve_direction_initial(initial_hidden, dir_idx, batch, hs)?
        .unwrap_or_else(|| HostArray::new(vec![batch, hs], vec![0.0; batch * hs]).unwrap());
    let mut c = resolve_direction_initial(initial_cell, dir_idx, batch, hs)?
        .unwrap_or_else(|| HostArray::new(vec![batch, hs], vec![0.0; batch * hs]).unwrap());
    let mut h_seq = if options.return_sequence {
        Some(HostArray::new(
            vec![step_count, batch, hs],
            vec![0.0; step_count * batch * hs],
        )?)
    } else {
        None
    };
    let peephole_dir = peephole
        .map(|p| resolve_rnn_peephole(p, dir_idx, hs))
        .transpose()?;
    let step_range: Vec<usize> = if direction == Direction::Backward {
        (0..step_count).rev().collect()
    } else {
        (0..step_count).collect()
    };
    for t in step_range.iter() {
        let x_t = slice_sequence_step(input, *t, step_count, batch, input_size)?;
        let (h_new, c_new) = lstm_cell_internal(
            &x_t,
            weight,
            recurrence,
            &h,
            &c,
            config,
            bias,
            recurrent_bias,
            peephole_dir.as_ref(),
            dir_idx,
        )?;
        h = h_new;
        c = c_new;
        if let Some(seq) = h_seq.as_mut() {
            copy_hidden_to_sequence(seq, *t, batch, hs, &h);
        }
    }
    Ok(LstmOutput {
        sequence: h_seq,
        hidden: h,
        cell: c,
    })
}

#[allow(clippy::too_many_arguments)]
fn lstm_bidirectional(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    options: &MLLstmOptions,
    config: &LstmConfig,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    peephole: Option<&HostArray>,
    initial_hidden: Option<&HostArray>,
    initial_cell: Option<&HostArray>,
) -> Result<LstmOutput, GraphError> {
    let (_step_count, batch, _input_size) = lstm_sequence_dims(input)?;
    let hs = infer_lstm_hidden_size_from_weight(weight)?;
    let h0_fwd = resolve_direction_initial(initial_hidden, 0, batch, hs)?;
    let c0_fwd = resolve_direction_initial(initial_cell, 0, batch, hs)?;

    let LstmOutput {
        sequence: h_seq_fwd,
        hidden: h_fwd,
        cell: c_fwd,
    } = lstm_unidirectional(
        input,
        weight,
        recurrence,
        options,
        config,
        Direction::Forward,
        bias,
        recurrent_bias,
        peephole,
        h0_fwd.as_ref(),
        c0_fwd.as_ref(),
    )?;

    let h0_bwd = resolve_direction_initial(initial_hidden, 1, batch, hs)?;
    let c0_bwd = resolve_direction_initial(initial_cell, 1, batch, hs)?;
    let LstmOutput {
        sequence: h_seq_bwd,
        hidden: h_bwd,
        cell: c_bwd,
    } = lstm_unidirectional(
        input,
        weight,
        recurrence,
        options,
        config,
        Direction::Backward,
        bias,
        recurrent_bias,
        peephole,
        h0_bwd.as_ref(),
        c0_bwd.as_ref(),
    )?;

    let final_h = stack_rnn_states(&h_fwd, &h_bwd)?;
    let final_c = stack_rnn_states(&c_fwd, &c_bwd)?;
    let h_seq = if options.return_sequence {
        let fwd = h_seq_fwd.ok_or_else(|| burn_err("forward LSTM missing sequence".to_string()))?;
        let bwd =
            h_seq_bwd.ok_or_else(|| burn_err("backward LSTM missing sequence".to_string()))?;
        Some(stack_rnn_sequences(&fwd, &bwd)?)
    } else {
        None
    };
    Ok(LstmOutput {
        sequence: h_seq,
        hidden: final_h,
        cell: final_c,
    })
}

fn resolve_rnn_matrix(w: &HostArray, direction: usize) -> Result<HostArray, GraphError> {
    match w.rank() {
        2 => Ok(w.clone()),
        3 => {
            let num_dir = w.shape[0];
            let dir = direction.min(num_dir.saturating_sub(1));
            let out_f = w.shape[1];
            let in_f = w.shape[2];
            let start = dir * out_f * in_f;
            HostArray::new(
                vec![out_f, in_f],
                w.data[start..start + out_f * in_f].to_vec(),
            )
        }
        r => Err(burn_err(format!(
            "RNN weight must be rank 2 or 3, got {r} with shape {:?}",
            w.shape
        ))),
    }
}

fn resolve_rnn_bias(
    b: &HostArray,
    direction: usize,
    expected_len: usize,
) -> Result<Vec<f32>, GraphError> {
    match b.rank() {
        1 if b.shape[0] == expected_len => Ok(b.data.clone()),
        1 => Err(burn_err(format!(
            "RNN bias length {} expected {}",
            b.shape[0], expected_len
        ))),
        2 => {
            let dir = direction.min(b.shape[0].saturating_sub(1));
            let row = b.shape[1];
            if row != expected_len {
                return Err(burn_err(format!(
                    "RNN bias row length {} expected {}",
                    row, expected_len
                )));
            }
            let start = dir * row;
            Ok(b.data[start..start + row].to_vec())
        }
        r => Err(burn_err(format!(
            "RNN bias must be rank 1 or 2, got {r} with shape {:?}",
            b.shape
        ))),
    }
}

fn resolve_rnn_peephole(
    p: &HostArray,
    direction: usize,
    hs: usize,
) -> Result<HostArray, GraphError> {
    let expected = 3 * hs;
    match p.rank() {
        1 if p.shape[0] == expected => Ok(p.clone()),
        2 => {
            let dir = direction.min(p.shape[0].saturating_sub(1));
            let row = p.shape[1];
            if row != expected {
                return Err(burn_err(format!(
                    "LSTM peephole row length {row} expected {expected}"
                )));
            }
            let start = dir * row;
            HostArray::new(vec![expected], p.data[start..start + row].to_vec())
        }
        r => Err(burn_err(format!(
            "LSTM peephole must be rank 1 or 2, got {r} with shape {:?}",
            p.shape
        ))),
    }
}

fn resolve_direction_initial(
    hidden: Option<&HostArray>,
    direction: usize,
    batch: usize,
    hs: usize,
) -> Result<Option<HostArray>, GraphError> {
    let Some(h) = hidden else {
        return Ok(None);
    };
    if h.rank() == 3 && h.shape[1] == batch && h.shape[2] == hs {
        let dir = direction.min(h.shape[0].saturating_sub(1));
        let start = dir * batch * hs;
        return HostArray::new(vec![batch, hs], h.data[start..start + batch * hs].to_vec())
            .map(Some);
    }
    normalize_hidden(h, batch, hs).map(Some)
}

fn infer_lstm_hidden_size_from_weight(weight: &HostArray) -> Result<usize, GraphError> {
    let gate_rows = match weight.rank() {
        2 => weight.shape[0],
        3 => weight.shape[1],
        r => {
            return Err(burn_err(format!(
                "LSTM weight must be [4*hidden, input] or [directions, 4*hidden, input], got rank {r} {:?}",
                weight.shape
            )));
        }
    };
    if gate_rows % 4 != 0 {
        return Err(burn_err(format!(
            "LSTM weight gate rows {gate_rows} must be divisible by 4, shape {:?}",
            weight.shape
        )));
    }
    Ok(gate_rows / 4)
}

fn slice_row_data(arr: &HostArray, batch: usize, width: usize) -> Vec<f32> {
    let start = batch * width;
    arr.data[start..start + width].to_vec()
}

fn matrix_batch_dims(input: &HostArray) -> Result<(usize, usize), GraphError> {
    match input.rank() {
        1 => Ok((1, input.shape[0])),
        2 => Ok((input.shape[0], input.shape[1])),
        r => Err(burn_err(format!("RNN input must be rank 1 or 2, got {r}"))),
    }
}

fn normalize_hidden(hidden: &HostArray, batch: usize, hs: usize) -> Result<HostArray, GraphError> {
    match hidden.rank() {
        1 if hidden.shape[0] == hs => {
            let mut out = vec![0.0; batch * hs];
            out[..hs].copy_from_slice(&hidden.data[..hs]);
            for b in 1..batch {
                out[b * hs..(b + 1) * hs].copy_from_slice(&hidden.data[..hs]);
            }
            HostArray::new(vec![batch, hs], out)
        }
        2 if hidden.shape[0] == batch && hidden.shape[1] == hs => Ok(hidden.clone()),
        2 if hidden.shape[0] == hs && hidden.shape[1] == batch => {
            let mut out = vec![0.0; batch * hs];
            for b in 0..batch {
                for h in 0..hs {
                    out[b * hs + h] = hidden.data[h * batch + b];
                }
            }
            HostArray::new(vec![batch, hs], out)
        }
        3 if hidden.shape[1] == batch && hidden.shape[2] == hs => {
            HostArray::new(vec![batch, hs], hidden.data[..batch * hs].to_vec())
        }
        r => Err(burn_err(format!(
            "hidden state shape {:?} incompatible with batch {batch}, hidden {hs} (rank {r})",
            hidden.shape
        ))),
    }
}

fn slice_row(arr: &HostArray, batch: usize, width: usize) -> HostArray {
    let start = batch * width;
    HostArray {
        shape: vec![1, width],
        data: arr.data[start..start + width].to_vec(),
    }
}

fn write_row(arr: &mut HostArray, batch: usize, width: usize, row: &[f32]) {
    let start = batch * width;
    arr.data[start..start + width].copy_from_slice(row);
}

fn linear_2d_with_bias_vec(
    x: &HostArray,
    w: &HostArray,
    b: Option<&[f32]>,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>, GraphError> {
    if w.shape.len() != 2
        || w.shape[0] != out_features
        || w.shape[1] != in_features
        || x.shape[x.shape.len() - 1] != in_features
    {
        return Err(burn_err(format!(
            "linear mismatch: x {:?}, w {:?}, expected out {out_features}, in {in_features}",
            x.shape, w.shape
        )));
    }
    let mut out = vec![0.0f32; out_features];
    let x_data = &x.data[x.data.len() - in_features..];
    for o in 0..out_features {
        let mut sum = 0.0f32;
        for i in 0..in_features {
            sum += x_data[i] * w.data[o * in_features + i];
        }
        if let Some(bias) = b {
            sum += bias.get(o).copied().unwrap_or(0.0);
        }
        out[o] = sum;
    }
    Ok(out)
}

fn apply_activation(activation: Activation, x: f32) -> f32 {
    match activation {
        Activation::Relu => x.max(0.0),
        Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Activation::Tanh => x.tanh(),
    }
}

#[allow(clippy::too_many_arguments)]
fn split_gru_gates(
    w: &[f32],
    r: &[f32],
    h_prev: &[f32],
    recurrence: &HostArray,
    hs: usize,
    layout: GruLayout,
    reset_after: bool,
    activations: [Activation; 2],
) -> Result<(Vec<f32>, Vec<f32>), GraphError> {
    let [act_zr, act_n] = activations;
    let (z_idx, r_idx, n_idx) = match layout {
        GruLayout::Rzn => (1, 0, 2),
        GruLayout::Zrn => (0, 1, 2),
    };
    let mut z = vec![0.0; hs];
    let mut r_gate = vec![0.0; hs];
    let mut n = vec![0.0; hs];
    for i in 0..hs {
        z[i] = apply_activation(act_zr, w[z_idx * hs + i] + r[z_idx * hs + i]);
        r_gate[i] = apply_activation(act_zr, w[r_idx * hs + i] + r[r_idx * hs + i]);
    }
    for i in 0..hs {
        let pre = if reset_after {
            w[n_idx * hs + i] + r_gate[i] * r[n_idx * hs + i]
        } else {
            let mut rh = 0.0f32;
            for j in 0..hs {
                rh += recurrence.data[(n_idx * hs + i) * hs + j] * h_prev[j];
            }
            let mut r_dot_h = 0.0f32;
            for j in 0..hs {
                r_dot_h += recurrence.data[(n_idx * hs + i) * hs + j] * r_gate[j] * h_prev[j];
            }
            w[n_idx * hs + i] + (r[n_idx * hs + i] - rh) + r_dot_h
        };
        n[i] = apply_activation(act_n, pre);
    }
    Ok((z, n))
}

fn split_lstm_gates(
    gates: &[f32],
    hs: usize,
    layout: LstmLayout,
    activations: [Activation; 3],
) -> Result<LstmGates, GraphError> {
    let [i_idx, o_idx, f_idx, g_idx] = lstm_gate_indices(layout);
    let [act_ifo, act_g, act_c] = activations;
    let mut i = vec![0.0; hs];
    let mut o = vec![0.0; hs];
    let mut f = vec![0.0; hs];
    let mut g = vec![0.0; hs];
    for j in 0..hs {
        i[j] = apply_activation(act_ifo, gates[i_idx * hs + j]);
        o[j] = apply_activation(act_ifo, gates[o_idx * hs + j]);
        f[j] = apply_activation(act_ifo, gates[f_idx * hs + j]);
        g[j] = apply_activation(act_g, gates[g_idx * hs + j]);
    }
    Ok(LstmGates {
        input: i,
        output: o,
        forget: f,
        cell: g,
        activation: act_c,
    })
}

fn apply_peephole(
    gates: &mut [f32],
    peephole: &HostArray,
    c: &[f32],
    hs: usize,
    layout: LstmLayout,
) -> Result<(), GraphError> {
    if peephole.numel() < 3 * hs {
        return Err(burn_err(format!(
            "peephole weight expected at least {} elements, got {}",
            3 * hs,
            peephole.numel()
        )));
    }
    let [i_idx, o_idx, f_idx, _] = lstm_gate_indices(layout);
    for j in 0..hs {
        gates[i_idx * hs + j] += peephole.data[j] * c[j];
        gates[o_idx * hs + j] += peephole.data[hs + j] * c[j];
        gates[f_idx * hs + j] += peephole.data[2 * hs + j] * c[j];
    }
    Ok(())
}

fn gru_sequence_dims(input: &HostArray, steps: u32) -> Result<(usize, usize, usize), GraphError> {
    match input.rank() {
        2 => {
            let s = if steps == 0 { 1 } else { steps as usize };
            Ok((s, input.shape[0], input.shape[1]))
        }
        3 => Ok((input.shape[0], input.shape[1], input.shape[2])),
        r => Err(burn_err(format!("gru input must be rank 2 or 3, got {r}"))),
    }
}

fn lstm_sequence_dims(input: &HostArray) -> Result<(usize, usize, usize), GraphError> {
    match input.rank() {
        2 => Ok((1, input.shape[0], input.shape[1])),
        3 => Ok((input.shape[0], input.shape[1], input.shape[2])),
        r => Err(burn_err(format!("lstm input must be rank 2 or 3, got {r}"))),
    }
}

fn slice_sequence_step(
    input: &HostArray,
    step: usize,
    step_count: usize,
    batch: usize,
    input_size: usize,
) -> Result<HostArray, GraphError> {
    if input.rank() == 2 && step_count == 1 {
        Ok(input.clone())
    } else if input.rank() == 3 {
        let start = step * batch * input_size;
        Ok(HostArray::new(
            vec![batch, input_size],
            input.data[start..start + batch * input_size].to_vec(),
        )?)
    } else {
        Err(burn_err(format!(
            "cannot slice sequence step {step} from shape {:?}",
            input.shape
        )))
    }
}

fn copy_hidden_to_sequence(
    seq: &mut HostArray,
    step: usize,
    batch: usize,
    hs: usize,
    h: &HostArray,
) {
    let start = step * batch * hs;
    seq.data[start..start + batch * hs].copy_from_slice(&h.data[..batch * hs]);
}

#[allow(clippy::too_many_arguments)]
fn gru_cell_internal(
    input: &HostArray,
    weight: &HostArray,
    recurrence: &HostArray,
    hidden_state: &HostArray,
    hs: usize,
    config: &GruConfig,
    bias: Option<&HostArray>,
    recurrent_bias: Option<&HostArray>,
    direction: usize,
) -> Result<HostArray, GraphError> {
    let w = resolve_rnn_matrix(weight, direction)?;
    let r = resolve_rnn_matrix(recurrence, direction)?;
    let (batch, input_size) = matrix_batch_dims(input)?;
    let layout = config.layout;
    let reset_after = config.reset_after;
    let activations = config.activations;
    let w_bias = bias
        .map(|b| resolve_rnn_bias(b, direction, 3 * hs))
        .transpose()?;
    let r_bias = recurrent_bias
        .map(|b| resolve_rnn_bias(b, direction, 3 * hs))
        .transpose()?;
    let mut h = hidden_state.clone();
    for bidx in 0..batch {
        let x_row = slice_row(input, bidx, input_size);
        let h_row = slice_row(&h, bidx, hs);
        let w_gates = linear_2d_with_bias_vec(&x_row, &w, w_bias.as_deref(), 3 * hs, input_size)?;
        let r_gates = linear_2d_with_bias_vec(&h_row, &r, r_bias.as_deref(), 3 * hs, hs)?;
        let (z, n) = split_gru_gates(
            &w_gates,
            &r_gates,
            &h_row.data,
            &r,
            hs,
            layout,
            reset_after,
            activations,
        )?;
        let h_new: Vec<f32> = z
            .iter()
            .zip(n.iter().zip(h_row.data.iter()))
            .map(|(&zi, (&ni, &hi))| (1.0 - zi) * ni + zi * hi)
            .collect();
        write_row(&mut h, bidx, hs, &h_new);
    }
    Ok(h)
}

fn stack_rnn_states(dir0: &HostArray, dir1: &HostArray) -> Result<HostArray, GraphError> {
    let batch = dir0.shape[0];
    let hs = dir0.shape[1];
    let mut out = vec![0.0; 2 * batch * hs];
    out[..batch * hs].copy_from_slice(&dir0.data);
    out[batch * hs..].copy_from_slice(&dir1.data);
    HostArray::new(vec![2, batch, hs], out)
}

fn stack_rnn_sequences(fwd: &HostArray, bwd: &HostArray) -> Result<HostArray, GraphError> {
    let steps = fwd.shape[0];
    let batch = fwd.shape[1];
    let hs = fwd.shape[2];
    let mut out = vec![0.0; steps * 2 * batch * hs];
    for t in 0..steps {
        let f_start = t * batch * hs;
        let b_start = t * batch * hs;
        let out_f = t * 2 * batch * hs;
        let out_b = out_f + batch * hs;
        out[out_f..out_f + batch * hs].copy_from_slice(&fwd.data[f_start..f_start + batch * hs]);
        out[out_b..out_b + batch * hs].copy_from_slice(&bwd.data[b_start..b_start + batch * hs]);
    }
    HostArray::new(vec![steps, 2, batch, hs], out)
}

pub fn format_rnn_state_nd(
    state: &HostArray,
    num_directions: usize,
) -> Result<HostArray, GraphError> {
    if state.rank() == 3 {
        return Ok(state.clone());
    }
    if state.rank() == 2 {
        return HostArray::new(
            vec![num_directions, state.shape[0], state.shape[1]],
            state.data.clone(),
        );
    }
    Err(burn_err(format!(
        "RNN state expected rank 2 or 3, got rank {} shape {:?}",
        state.rank(),
        state.shape
    )))
}

pub fn format_rnn_hidden_sequence(
    seq: &HostArray,
    num_directions: usize,
) -> Result<HostArray, GraphError> {
    if seq.rank() == 4 {
        return Ok(seq.clone());
    }
    if seq.rank() == 3 {
        return HostArray::new(
            vec![seq.shape[0], num_directions, seq.shape[1], seq.shape[2]],
            seq.data.clone(),
        );
    }
    Err(burn_err(format!(
        "RNN sequence expected rank 3 or 4, got rank {} shape {:?}",
        seq.rank(),
        seq.shape
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_recurrent_choices_return_errors_instead_of_defaults() {
        assert_eq!(
            GruConfig::new("", None, false).unwrap().layout,
            GruLayout::Zrn
        );
        assert_eq!(LstmConfig::new("", None).unwrap().layout, LstmLayout::Iofg);
        assert_eq!(
            parse_option_or_default::<Direction>("", "direction").unwrap(),
            Direction::Forward
        );
        assert!(GruConfig::new("unknown", None, false).is_err());
        assert!(LstmConfig::new("unknown", None).is_err());
        assert!(parse_option::<Direction>("unknown", "direction").is_err());
        let activations = vec!["sigmiod".to_owned(), "tanh".to_owned()];
        assert!(GruConfig::new("zrn", Some(&activations), false).is_err());
    }
    #[test]
    fn recurrent_gate_layouts_and_activations_use_explicit_variants() {
        let gru = GruConfig::new("rzn", Some(&["relu".into(), "tanh".into()]), true).unwrap();
        assert_eq!(gru.layout, GruLayout::Rzn);
        assert_eq!(gru.activations, [Activation::Relu, Activation::Tanh]);
        assert!(gru.reset_after);
        assert_eq!(lstm_gate_indices(LstmLayout::Ifgo), [0, 3, 1, 2]);
        assert_eq!(lstm_gate_indices(LstmLayout::Iofg), [0, 1, 2, 3]);
        assert_eq!(apply_activation(Activation::Relu, -1.0), 0.0);
        assert_eq!(apply_activation(Activation::Sigmoid, 0.0), 0.5);
        assert_eq!(apply_activation(Activation::Tanh, 0.0), 0.0);
    }
}

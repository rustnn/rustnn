/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * TensorRT lowering for WebNN `lstm` and `lstmCell`, following the ONNX-TensorRT
 * `DEFINE_BUILTIN_OP_IMPORTER(LSTM)` loop decomposition in onnx-tensorrt/onnxOpImporters.cpp.
 */

use std::collections::HashMap;

use trtx::ActivationType;
use trtx::Axes;
use trtx::ElementWiseOperation as Ew;
use trtx::LoopOutput;
use trtx::ReduceOperation;

use super::trtx::TrtxConverter;
use super::trtx_rnn::{
    RnnDirection, rnn_activation, rnn_activation_type, rnn_add_trip_limit, rnn_batch_size,
    rnn_check_bidirectional_activations, rnn_concat_loop_outputs, rnn_err, rnn_err_fmt, rnn_ew,
    rnn_fit_cell_output, rnn_fit_hidden_output, rnn_fit_sequence_output, rnn_initial_state,
    rnn_input_size, rnn_isolate_gate, rnn_iteration_input, rnn_matmul_t,
    rnn_normalize_gate_tensor_2d, rnn_reorder_gates_ifgo_to_iofc, rnn_reshape, rnn_tensor,
    rnn_to_3d, rnn_validate_hidden_size,
};
use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo, get_static_or_max_size};
use crate::operators::Operation;

const LSTM_NUM_GATES: i32 = 4;
/// Peephole weights are always ordered [input, output, forget].
const LSTM_NUM_PEEPHOLE: i32 = 3;

/// Entry point from [`super::trtx::TrtxConverter`].
pub(crate) fn add_lstm_op<'a>(
    graph: &GraphInfo,
    network: &mut trtx::NetworkDefinition<'a>,
    tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
    operation: &Operation,
) -> Result<(), GraphError> {
    TrtxConverter::add_lstm_op_impl(graph, network, tensor_map, operation)
}

/// Entry point from [`super::trtx::TrtxConverter`].
pub(crate) fn add_lstm_cell_op<'a>(
    graph: &GraphInfo,
    network: &mut trtx::NetworkDefinition<'a>,
    tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
    operation: &Operation,
) -> Result<(), GraphError> {
    TrtxConverter::add_lstm_cell_op_impl(graph, network, tensor_map, operation)
}

#[derive(Clone, Copy)]
struct LstmActivations {
    io_gate: ActivationType,
    cell: ActivationType,
    hidden: ActivationType,
}

impl Default for LstmActivations {
    fn default() -> Self {
        Self {
            io_gate: ActivationType::kSIGMOID,
            cell: ActivationType::kTANH,
            hidden: ActivationType::kTANH,
        }
    }
}

struct LstmPreparedWeights<'a> {
    weights: trtx::Tensor<'a>,
    recurrence: trtx::Tensor<'a>,
    combined_bias: Option<trtx::Tensor<'a>>,
    peephole: Option<trtx::Tensor<'a>>,
}

struct LstmLoopParams {
    seq_input_id: u32,
    weight_id: u32,
    recurrence_id: u32,
    bias_id: Option<u32>,
    recurrent_bias_id: Option<u32>,
    peephole_id: Option<u32>,
    initial_hidden_id: Option<u32>,
    initial_cell_id: Option<u32>,
    hidden_size: i32,
    seq_steps: i32,
    direction: RnnDirection,
    layout_ifgo: bool,
    activations: LstmActivations,
    input_dtype: DataType,
    batch_size: u32,
    unsqueeze_sequence_axis: bool,
}

enum LstmOutputMode<'a> {
    Full {
        return_sequence: bool,
        outputs: &'a [u32],
    },
    Cell {
        hidden_output_id: u32,
        cell_output_id: u32,
    },
}

impl TrtxConverter {
    pub(super) fn add_lstm_op_impl<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let Operation::Lstm {
            input,
            weight,
            recurrence,
            steps,
            hidden_size,
            options,
            outputs,
        } = operation
        else {
            return Err(rnn_err("internal: expected Lstm operation"));
        };

        if outputs.is_empty() {
            return Err(rnn_err("lstm requires at least one output"));
        }

        let opts = options.clone().unwrap_or_default();
        let hidden_size = rnn_validate_hidden_size(*hidden_size, "lstm")?;
        let direction = RnnDirection::from_webnn(&opts.direction);
        if direction == RnnDirection::Bidirectional {
            rnn_check_bidirectional_activations(opts.activations.as_deref(), 3, "lstm")?;
        }

        let input_operand = graph
            .operand(*input)
            .ok_or_else(|| rnn_err("lstm input operand"))?;
        let input_rank = input_operand.descriptor.shape.len();
        if input_rank != 2 && input_rank != 3 {
            return Err(rnn_err("lstm input must have rank 2 or 3"));
        }
        let seq_steps = if input_rank == 3 {
            get_static_or_max_size(&input_operand.descriptor.shape[0]) as i32
        } else {
            i32::try_from(*steps).unwrap_or(1)
        };
        if seq_steps <= 0 {
            return Err(rnn_err("lstm sequence length must be positive"));
        }

        let params = LstmLoopParams {
            seq_input_id: *input,
            weight_id: *weight,
            recurrence_id: *recurrence,
            bias_id: opts.bias,
            recurrent_bias_id: opts.recurrent_bias,
            peephole_id: opts.peephole_weight,
            initial_hidden_id: opts.initial_hidden_state,
            initial_cell_id: opts.initial_cell_state,
            hidden_size,
            seq_steps,
            direction,
            layout_ifgo: opts.layout.eq_ignore_ascii_case("ifgo"),
            activations: lstm_parse_activations(opts.activations.as_deref()),
            input_dtype: input_operand.descriptor.data_type,
            batch_size: rnn_batch_size(&input_operand.descriptor.shape),
            unsqueeze_sequence_axis: input_rank == 2,
        };

        lstm_run_loop(
            graph,
            network,
            tensor_map,
            &params,
            LstmOutputMode::Full {
                return_sequence: opts.return_sequence,
                outputs,
            },
        )
    }

    pub(super) fn add_lstm_cell_op_impl<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let Operation::LstmCell {
            input,
            weight,
            recurrence,
            hidden_state,
            cell_state,
            hidden_size,
            options,
            outputs,
        } = operation
        else {
            return Err(rnn_err("internal: expected LstmCell operation"));
        };

        if outputs.len() < 2 {
            return Err(rnn_err("lstmCell requires two outputs (hidden and cell)"));
        }

        let opts = options.clone().unwrap_or_default();
        let hidden_size = rnn_validate_hidden_size(*hidden_size, "lstmCell")?;
        let input_operand = graph
            .operand(*input)
            .ok_or_else(|| rnn_err("lstmCell input operand"))?;

        let params = LstmLoopParams {
            seq_input_id: *input,
            weight_id: *weight,
            recurrence_id: *recurrence,
            bias_id: opts.bias,
            recurrent_bias_id: opts.recurrent_bias,
            peephole_id: opts.peephole_weight,
            initial_hidden_id: Some(*hidden_state),
            initial_cell_id: Some(*cell_state),
            hidden_size,
            seq_steps: 1,
            direction: RnnDirection::Forward,
            layout_ifgo: opts.layout.eq_ignore_ascii_case("ifgo"),
            activations: lstm_parse_activations(opts.activations.as_deref()),
            input_dtype: input_operand.descriptor.data_type,
            batch_size: rnn_batch_size(&input_operand.descriptor.shape),
            unsqueeze_sequence_axis: true,
        };

        lstm_run_loop(
            graph,
            network,
            tensor_map,
            &params,
            LstmOutputMode::Cell {
                hidden_output_id: outputs[0],
                cell_output_id: outputs[1],
            },
        )
    }
}

fn lstm_parse_activations(names: Option<&[String]>) -> LstmActivations {
    let mut acts = LstmActivations::default();
    if let Some(list) = names {
        if let Some(a) = list.first() {
            acts.io_gate = rnn_activation_type(a);
        }
        if let Some(a) = list.get(1) {
            acts.cell = rnn_activation_type(a);
        }
        if let Some(a) = list.get(2) {
            acts.hidden = rnn_activation_type(a);
        }
    }
    acts
}

fn lstm_run_loop<'a>(
    graph: &GraphInfo,
    network: &mut trtx::NetworkDefinition<'a>,
    tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
    params: &LstmLoopParams,
    output_mode: LstmOutputMode<'_>,
) -> Result<(), GraphError> {
    let num_directions = params.direction.num_directions();
    let input_size = rnn_input_size(graph, params.seq_input_id)?;

    let input_tensor = rnn_tensor(tensor_map, params.seq_input_id, "lstm seq input")?;
    let seq_input = if params.unsqueeze_sequence_axis {
        rnn_reshape(
            network,
            &input_tensor,
            &[1, params.batch_size as i64, input_size],
            "lstm unsqueeze seq input",
        )?
    } else {
        input_tensor
    };

    let prepared = lstm_prepare_weights(
        graph,
        network,
        tensor_map,
        params.weight_id,
        params.recurrence_id,
        params.bias_id,
        params.recurrent_bias_id,
        params.peephole_id,
        num_directions,
        params.hidden_size,
        params.layout_ifgo,
    )?;

    let state_shape = [
        num_directions as i64,
        params.batch_size as i64,
        params.hidden_size as i64,
    ];
    let initial_hidden = rnn_initial_state(
        network,
        tensor_map,
        params.initial_hidden_id,
        params.direction,
        &state_shape,
        params.input_dtype,
        "lstm initial hidden",
    )?;
    let initial_cell = rnn_initial_state(
        network,
        tensor_map,
        params.initial_cell_id,
        params.direction,
        &state_shape,
        params.input_dtype,
        "lstm initial cell",
    )?;

    let mut loop_body = network
        .add_loop()
        .map_err(|e| rnn_err_fmt(format!("lstm add_loop: {e}")))?;
    rnn_add_trip_limit(network, &mut loop_body, params.seq_steps, "lstm")?;

    let iteration_input = rnn_iteration_input(
        network,
        &mut loop_body,
        &seq_input,
        params.direction,
        num_directions,
        "lstm",
    )?;

    let mut ht1_layer = loop_body
        .add_recurrence(network, &initial_hidden)
        .map_err(|e| rnn_err_fmt(format!("lstm add_recurrence H: {e}")))?;
    let ht1 = ht1_layer
        .output(network, 0)
        .map_err(|e| rnn_err_fmt(format!("lstm recurrence H output: {e}")))?;

    let mut ct1_layer = loop_body
        .add_recurrence(network, &initial_cell)
        .map_err(|e| rnn_err_fmt(format!("lstm add_recurrence C: {e}")))?;
    let ct1 = ct1_layer
        .output(network, 0)
        .map_err(|e| rnn_err_fmt(format!("lstm recurrence C output: {e}")))?;

    let (ht, ct) = lstm_compute_step(
        network,
        &iteration_input,
        &ht1,
        &ct1,
        params.hidden_size,
        params.activations,
        &prepared,
    )?;

    ht1_layer
        .set_input(network, 1, &ht)
        .map_err(|e| rnn_err_fmt(format!("lstm recurrence H set_input: {e}")))?;
    ct1_layer
        .set_input(network, 1, &ct)
        .map_err(|e| rnn_err_fmt(format!("lstm recurrence C set_input: {e}")))?;

    let yh_layer = loop_body
        .add_loop_output(network, &ht1, LoopOutput::kLAST_VALUE, 0)
        .map_err(|e| rnn_err_fmt(format!("lstm loop last hidden: {e}")))?;
    let yh = yh_layer
        .output(network, 0)
        .map_err(|e| rnn_err_fmt(format!("lstm loop last hidden output: {e}")))?;

    let yc_layer = loop_body
        .add_loop_output(network, &ct1, LoopOutput::kLAST_VALUE, 0)
        .map_err(|e| rnn_err_fmt(format!("lstm loop last cell: {e}")))?;
    let yc = yc_layer
        .output(network, 0)
        .map_err(|e| rnn_err_fmt(format!("lstm loop last cell output: {e}")))?;

    match output_mode {
        LstmOutputMode::Cell {
            hidden_output_id,
            cell_output_id,
        } => {
            let hidden_out = rnn_fit_cell_output(
                network,
                &yh,
                params.batch_size,
                params.hidden_size,
                "lstmCell hidden output",
            )?;
            let cell_out = rnn_fit_cell_output(
                network,
                &yc,
                params.batch_size,
                params.hidden_size,
                "lstmCell cell output",
            )?;
            tensor_map.insert(hidden_output_id, hidden_out);
            tensor_map.insert(cell_output_id, cell_out);
        }
        LstmOutputMode::Full {
            return_sequence,
            outputs,
        } => {
            let (hidden_id, cell_id, seq_id) =
                lstm_resolve_output_ids(graph, outputs, return_sequence);

            let yh_final = rnn_fit_hidden_output(graph, network, hidden_id, &yh, "lstm")?;
            tensor_map.insert(hidden_id, yh_final);

            if outputs.len() >= 2 && cell_id != hidden_id {
                let yc_final = rnn_fit_hidden_output(graph, network, cell_id, &yc, "lstm cell")?;
                tensor_map.insert(cell_id, yc_final);
            }

            if return_sequence && let Some(seq_output_id) = seq_id {
                let single_pass_shape =
                    [1_i64, params.batch_size as i64, params.hidden_size as i64];
                let reverse = params.direction == RnnDirection::Reverse;
                let seq_tensor = rnn_concat_loop_outputs(
                    network,
                    &mut loop_body,
                    &ht,
                    num_directions,
                    &single_pass_shape,
                    params.seq_steps,
                    reverse,
                    "lstm",
                )?;
                let seq_final = rnn_fit_sequence_output(
                    graph,
                    network,
                    seq_output_id,
                    &seq_tensor,
                    params.direction,
                    "lstm",
                )?;
                tensor_map.insert(seq_output_id, seq_final);
            }
        }
    }

    Ok(())
}

fn lstm_resolve_output_ids(
    graph: &GraphInfo,
    outputs: &[u32],
    return_sequence: bool,
) -> (u32, u32, Option<u32>) {
    if outputs.len() >= 3 || return_sequence {
        for &out_id in outputs {
            let name = graph
                .operand(out_id)
                .and_then(|o| o.name.clone())
                .unwrap_or_default();
            if name.contains("Output1") {
                let cell = outputs
                    .iter()
                    .find(|&&id| {
                        graph
                            .operand(id)
                            .and_then(|o| o.name.clone())
                            .unwrap_or_default()
                            .contains("Output2")
                    })
                    .copied()
                    .unwrap_or(outputs.get(1).copied().unwrap_or(out_id));
                let seq = outputs
                    .iter()
                    .find(|&&id| {
                        graph
                            .operand(id)
                            .and_then(|o| o.name.clone())
                            .unwrap_or_default()
                            .contains("Output3")
                    })
                    .copied()
                    .or_else(|| outputs.get(2).copied());
                return (out_id, cell, seq);
            }
        }
        return (
            outputs[0],
            outputs.get(1).copied().unwrap_or(outputs[0]),
            outputs.get(2).copied(),
        );
    }
    if outputs.len() >= 2 {
        return (outputs[0], outputs[1], None);
    }
    (outputs[0], outputs[0], None)
}

fn lstm_maybe_reorder_gates<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    tensor: &trtx::Tensor<'a>,
    gate_axis: i32,
    hidden_size: i32,
    layout_ifgo: bool,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    if layout_ifgo {
        rnn_reorder_gates_ifgo_to_iofc(network, tensor, gate_axis, hidden_size, label)
    } else {
        Ok(*tensor)
    }
}

#[allow(clippy::too_many_arguments)]
fn lstm_prepare_weights<'a>(
    graph: &GraphInfo,
    network: &mut trtx::NetworkDefinition<'a>,
    tensor_map: &HashMap<u32, trtx::Tensor<'a>>,
    weight_id: u32,
    recurrence_id: u32,
    bias_id: Option<u32>,
    recurrent_bias_id: Option<u32>,
    peephole_id: Option<u32>,
    num_directions: i32,
    hidden_size: i32,
    layout_ifgo: bool,
) -> Result<LstmPreparedWeights<'a>, GraphError> {
    let input_size = {
        let w_shape = graph
            .operand(weight_id)
            .ok_or_else(|| rnn_err("lstm weight operand"))?
            .descriptor
            .static_or_max_shape();
        match w_shape.len() {
            2 => w_shape[1] as i32,
            3 => w_shape[2] as i32,
            _ => return Err(rnn_err("lstm weight must be rank 2 or 3")),
        }
    };

    let mut weights = rnn_to_3d(
        network,
        &rnn_tensor(tensor_map, weight_id, "lstm weight")?,
        num_directions,
        LSTM_NUM_GATES * hidden_size,
        input_size,
        "lstm weight 3d",
    )?;
    let mut recurrence = rnn_to_3d(
        network,
        &rnn_tensor(tensor_map, recurrence_id, "lstm recurrence")?,
        num_directions,
        LSTM_NUM_GATES * hidden_size,
        hidden_size,
        "lstm recurrence 3d",
    )?;

    weights = lstm_maybe_reorder_gates(
        network,
        &weights,
        1,
        hidden_size,
        layout_ifgo,
        "lstm weight",
    )?;
    recurrence = lstm_maybe_reorder_gates(
        network,
        &recurrence,
        1,
        hidden_size,
        layout_ifgo,
        "lstm recurrence",
    )?;

    let combined_bias = lstm_prepare_combined_bias(
        network,
        tensor_map,
        bias_id,
        recurrent_bias_id,
        num_directions,
        hidden_size,
        layout_ifgo,
    )?;

    let peephole = if let Some(id) = peephole_id {
        Some(lstm_prepare_peephole(
            network,
            tensor_map,
            id,
            num_directions,
            hidden_size,
        )?)
    } else {
        None
    };

    Ok(LstmPreparedWeights {
        weights,
        recurrence,
        combined_bias,
        peephole,
    })
}

fn lstm_zero_gate_bias<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    num_directions: i32,
    hidden_size: i32,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let n = (num_directions * LSTM_NUM_GATES * hidden_size) as usize;
    let bytes = vec![0u8; n * 4];
    network
        .add_small_constant_copied(
            &[num_directions as i64, (LSTM_NUM_GATES * hidden_size) as i64],
            &bytes,
            trtx::DataType::kFLOAT,
            None,
        )
        .map_err(|e| rnn_err_fmt(format!("{label}: {e}")))?
        .output(&*network, 0)
        .map_err(|e| rnn_err_fmt(format!("{label} output: {e}")))
}

fn lstm_prepare_combined_bias<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    tensor_map: &HashMap<u32, trtx::Tensor<'a>>,
    bias_id: Option<u32>,
    recurrent_bias_id: Option<u32>,
    num_directions: i32,
    hidden_size: i32,
    layout_ifgo: bool,
) -> Result<Option<trtx::Tensor<'a>>, GraphError> {
    if bias_id.is_none() && recurrent_bias_id.is_none() {
        return Ok(None);
    }

    let gate_rows = LSTM_NUM_GATES * hidden_size;
    let mut bias = if let Some(id) = bias_id {
        rnn_tensor(tensor_map, id, "lstm bias")?
    } else {
        lstm_zero_gate_bias(network, num_directions, hidden_size, "lstm bias zero")?
    };
    let mut rbias = if let Some(id) = recurrent_bias_id {
        rnn_tensor(tensor_map, id, "lstm recurrent bias")?
    } else {
        lstm_zero_gate_bias(network, num_directions, hidden_size, "lstm rbias zero")?
    };

    bias = rnn_normalize_gate_tensor_2d(network, &bias, num_directions, gate_rows, "lstm bias 2d")?;
    rbias =
        rnn_normalize_gate_tensor_2d(network, &rbias, num_directions, gate_rows, "lstm rbias 2d")?;

    let gate_axis = 1_i32;

    bias = lstm_maybe_reorder_gates(
        network,
        &bias,
        gate_axis,
        hidden_size,
        layout_ifgo,
        "lstm bias",
    )?;
    rbias = lstm_maybe_reorder_gates(
        network,
        &rbias,
        gate_axis,
        hidden_size,
        layout_ifgo,
        "lstm recurrent bias",
    )?;

    let bias = rnn_to_3d(network, &bias, num_directions, gate_rows, 1, "lstm bias 3d")?;
    let rbias = rnn_to_3d(
        network,
        &rbias,
        num_directions,
        gate_rows,
        1,
        "lstm rbias 3d",
    )?;

    let refs = [&bias, &rbias];
    let mut concat = network
        .add_concatenation(&refs)
        .map_err(|e| rnn_err_fmt(format!("lstm combine bias: {e}")))?;
    concat.set_axis(network, 1);
    let combined = concat
        .output(&*network, 0)
        .map_err(|e| rnn_err_fmt(format!("lstm combine bias output: {e}")))?;

    let combined = rnn_reshape(
        network,
        &combined,
        &[
            num_directions as i64,
            2,
            (LSTM_NUM_GATES * hidden_size) as i64,
        ],
        "lstm bias reshape",
    )?;

    let reduced = network
        .add_reduce(
            &combined,
            ReduceOperation::kSUM,
            Axes::from_bits(0b010),
            true,
        )
        .map_err(|e| rnn_err_fmt(format!("lstm bias reduce: {e}")))?
        .output(&*network, 0)
        .map_err(|e| rnn_err_fmt(format!("lstm bias reduce output: {e}")))?;

    Ok(Some(reduced))
}

fn lstm_prepare_peephole<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    tensor_map: &HashMap<u32, trtx::Tensor<'a>>,
    peephole_id: u32,
    num_directions: i32,
    hidden_size: i32,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let peephole = rnn_tensor(tensor_map, peephole_id, "lstm peephole")?;
    let gate_rows = LSTM_NUM_PEEPHOLE * hidden_size;
    let peephole = rnn_normalize_gate_tensor_2d(
        network,
        &peephole,
        num_directions,
        gate_rows,
        "lstm peephole 2d",
    )?;
    rnn_to_3d(
        network,
        &peephole,
        num_directions,
        gate_rows,
        1,
        "lstm peephole 3d",
    )
}

#[allow(clippy::too_many_arguments)]
fn lstm_add_peephole<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    gate: &trtx::Tensor<'a>,
    cell_state: &trtx::Tensor<'a>,
    peephole: &trtx::Tensor<'a>,
    peephole_gate_index: i32,
    hidden_size: i32,
    num_directions: i32,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    // Peephole is `[numDir, 3*H, 1]`; gate blocks live on axis 1 (ONNX-TensorRT uses 2D `[numDir, 3*H]`).
    let start = [0_i64, (peephole_gate_index * hidden_size) as i64, 0_i64];
    let size = [num_directions as i64, hidden_size as i64, 1_i64];
    let p_weight = network
        .add_slice(peephole, &start, &size, &[1, 1, 1])
        .map_err(|e| rnn_err_fmt(format!("{label} peephole slice: {e}")))?
        .output(&*network, 0)
        .map_err(|e| rnn_err_fmt(format!("{label} peephole slice output: {e}")))?;
    let p_weight = rnn_reshape(
        network,
        &p_weight,
        &[num_directions as i64, 1, hidden_size as i64],
        &format!("{label} peephole broadcast"),
    )?;
    let p_term = rnn_ew(
        network,
        &p_weight,
        cell_state,
        Ew::kPROD,
        &format!("{label} peephole prod"),
    )?;
    rnn_ew(
        network,
        gate,
        &p_term,
        Ew::kSUM,
        &format!("{label} peephole add"),
    )
}

/// One LSTM time step: returns `(H(t), C(t))` in ONNX `iofc` gate order.
fn lstm_compute_step<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    iteration_input: &trtx::Tensor<'a>,
    ht1: &trtx::Tensor<'a>,
    ct1: &trtx::Tensor<'a>,
    hidden_size: i32,
    activations: LstmActivations,
    prepared: &LstmPreparedWeights<'a>,
) -> Result<(trtx::Tensor<'a>, trtx::Tensor<'a>), GraphError> {
    let dims = iteration_input
        .dimensions(&*network)
        .map_err(|e| rnn_err_fmt(e.to_string()))?;
    let num_directions = dims[0] as i32;
    let batch_size = dims[1];

    let xt_w = rnn_matmul_t(network, iteration_input, &prepared.weights, "lstm X W")?;
    let ht_r = rnn_matmul_t(network, ht1, &prepared.recurrence, "lstm H R")?;
    let mut intermediate = rnn_ew(network, &xt_w, &ht_r, Ew::kSUM, "lstm intermediate")?;
    if let Some(bias) = &prepared.combined_bias {
        intermediate = rnn_ew(network, &intermediate, bias, Ew::kSUM, "lstm + bias")?;
    }

    // iofc gate indices: 0=input, 1=output, 2=forget, 3=cell
    let mut it_gate = rnn_isolate_gate(
        network,
        &intermediate,
        0,
        hidden_size,
        num_directions,
        batch_size,
    )?;
    if let Some(p) = &prepared.peephole {
        it_gate = lstm_add_peephole(
            network,
            &it_gate,
            ct1,
            p,
            0,
            hidden_size,
            num_directions,
            "lstm input",
        )?;
    }
    let it_gate = rnn_activation(network, &it_gate, activations.io_gate, "lstm i act")?;

    let mut ft_gate = rnn_isolate_gate(
        network,
        &intermediate,
        2,
        hidden_size,
        num_directions,
        batch_size,
    )?;
    if let Some(p) = &prepared.peephole {
        ft_gate = lstm_add_peephole(
            network,
            &ft_gate,
            ct1,
            p,
            2,
            hidden_size,
            num_directions,
            "lstm forget",
        )?;
    }
    let ft_gate = rnn_activation(network, &ft_gate, activations.io_gate, "lstm f act")?;

    let ct_pre = rnn_isolate_gate(
        network,
        &intermediate,
        3,
        hidden_size,
        num_directions,
        batch_size,
    )?;
    let ct_gate = rnn_activation(network, &ct_pre, activations.cell, "lstm c act")?;

    let operand_ic = rnn_ew(network, &it_gate, &ct_gate, Ew::kPROD, "lstm i*c")?;
    let operand_fc = rnn_ew(network, &ft_gate, ct1, Ew::kPROD, "lstm f*C")?;
    let ct = rnn_ew(network, &operand_fc, &operand_ic, Ew::kSUM, "lstm C(t)")?;

    let mut ot_gate = rnn_isolate_gate(
        network,
        &intermediate,
        1,
        hidden_size,
        num_directions,
        batch_size,
    )?;
    if let Some(p) = &prepared.peephole {
        ot_gate = lstm_add_peephole(
            network,
            &ot_gate,
            &ct,
            p,
            1,
            hidden_size,
            num_directions,
            "lstm output",
        )?;
    }
    let ot_gate = rnn_activation(network, &ot_gate, activations.io_gate, "lstm o act")?;

    let h_act = rnn_activation(network, &ct, activations.hidden, "lstm h(C) act")?;
    let ht = rnn_ew(network, &ot_gate, &h_act, Ew::kPROD, "lstm H(t)")?;

    Ok((ht, ct))
}

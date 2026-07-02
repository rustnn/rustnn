/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * TensorRT lowering for WebNN `gru` and `gruCell`, following the ONNX-TensorRT
 * `DEFINE_BUILTIN_OP_IMPORTER(GRU)` loop decomposition in onnx-tensorrt/onnxOpImporters.cpp.
 */

use std::collections::HashMap;

use half::f16;
use trtx::ActivationType;
use trtx::ElementWiseOperation as Ew;
use trtx::LoopOutput;
use trtx::MatrixOperation as MatOp;
use trtx::TripLimit;

use super::trtx::TrtxConverter;
use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo, get_static_or_max_size};
use crate::operators::Operation;

const GRU_NUM_GATES: i32 = 3;

/// Entry point from [`super::trtx::TrtxConverter`].
pub(crate) fn add_gru_op<'a>(
    graph: &GraphInfo,
    network: &mut trtx::NetworkDefinition<'a>,
    tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
    operation: &Operation,
) -> Result<(), GraphError> {
    TrtxConverter::add_gru_op_impl(graph, network, tensor_map, operation)
}

/// Entry point from [`super::trtx::TrtxConverter`].
pub(crate) fn add_gru_cell_op<'a>(
    graph: &GraphInfo,
    network: &mut trtx::NetworkDefinition<'a>,
    tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
    operation: &Operation,
) -> Result<(), GraphError> {
    TrtxConverter::add_gru_cell_op_impl(graph, network, tensor_map, operation)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GruDirection {
    Forward,
    Reverse,
    Bidirectional,
}

impl GruDirection {
    fn num_directions(self) -> i32 {
        match self {
            Self::Forward | Self::Reverse => 1,
            Self::Bidirectional => 2,
        }
    }

    fn from_webnn(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "backward" | "reverse" => Self::Reverse,
            "both" | "bidirectional" => Self::Bidirectional,
            _ => Self::Forward,
        }
    }
}

#[derive(Clone, Copy)]
struct GruActivations {
    gate: ActivationType,
    hidden: ActivationType,
}

impl Default for GruActivations {
    fn default() -> Self {
        Self {
            gate: ActivationType::kSIGMOID,
            hidden: ActivationType::kTANH,
        }
    }
}

/// Prepared GRU weight tensors in ONNX z-r-n gate order, rank-3 `[numDirections, gates*H, ...]`.
struct GruPreparedWeights<'a> {
    weights_zr: trtx::Tensor<'a>,
    weights_h: trtx::Tensor<'a>,
    recurrence_zr: trtx::Tensor<'a>,
    recurrence_h: trtx::Tensor<'a>,
    bias_zr: Option<trtx::Tensor<'a>>,
    bias_h: Option<trtx::Tensor<'a>>,
    recurrence_bias_zr: Option<trtx::Tensor<'a>>,
    recurrence_bias_h: Option<trtx::Tensor<'a>>,
}

/// Shared parameters for the ONNX-TensorRT GRU loop (`DEFINE_BUILTIN_OP_IMPORTER(GRU)`).
struct GruLoopParams {
    seq_input_id: u32,
    weight_id: u32,
    recurrence_id: u32,
    bias_id: Option<u32>,
    recurrent_bias_id: Option<u32>,
    initial_hidden_id: Option<u32>,
    hidden_size: i32,
    seq_steps: i32,
    direction: GruDirection,
    linear_before_reset: bool,
    activations: GruActivations,
    layout_rzn: bool,
    input_dtype: DataType,
    batch_size: u32,
    /// When `Some`, reshape rank-2 `[B, E]` to `[1, B, E]` before the loop (gru single-step / rank-2).
    unsqueeze_sequence_axis: bool,
}

enum GruOutputMode<'a> {
    /// Full `gru`: optional sequence output + last hidden.
    Full {
        return_sequence: bool,
        outputs: &'a [u32],
    },
    /// `gruCell`: one step, single hidden output `[batch, hidden]`.
    Cell { output_id: u32 },
}

impl TrtxConverter {
    pub(super) fn add_gru_op_impl<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let Operation::Gru {
            input,
            weight,
            recurrence,
            steps,
            hidden_size,
            options,
            outputs,
        } = operation
        else {
            return Err(gru_err("internal: expected Gru operation"));
        };

        if outputs.is_empty() {
            return Err(gru_err("gru requires at least one output"));
        }

        let opts = options.clone().unwrap_or_default();
        let hidden_size = gru_validate_hidden_size(*hidden_size, "gru")?;
        let direction = GruDirection::from_webnn(&opts.direction);
        if direction == GruDirection::Bidirectional {
            gru_check_bidirectional_activations(opts.activations.as_deref())?;
        }

        let input_operand = graph
            .operand(*input)
            .ok_or_else(|| gru_err("gru input operand"))?;
        let input_rank = input_operand.descriptor.shape.len();
        if input_rank != 2 && input_rank != 3 {
            return Err(gru_err("gru input must have rank 2 or 3"));
        }
        let seq_steps = if input_rank == 3 {
            get_static_or_max_size(&input_operand.descriptor.shape[0]) as i32
        } else {
            i32::try_from(*steps).unwrap_or(1)
        };
        if seq_steps <= 0 {
            return Err(gru_err("gru sequence length must be positive"));
        }

        let params = GruLoopParams {
            seq_input_id: *input,
            weight_id: *weight,
            recurrence_id: *recurrence,
            bias_id: opts.bias,
            recurrent_bias_id: opts.recurrent_bias,
            initial_hidden_id: opts.initial_hidden_state,
            hidden_size,
            seq_steps,
            direction,
            linear_before_reset: opts.reset_after,
            activations: gru_parse_activations(opts.activations.as_deref()),
            layout_rzn: opts.layout.eq_ignore_ascii_case("rzn"),
            input_dtype: input_operand.descriptor.data_type,
            batch_size: gru_batch_size(&input_operand.descriptor.shape),
            unsqueeze_sequence_axis: input_rank == 2,
        };

        gru_run_loop(
            graph,
            network,
            tensor_map,
            &params,
            GruOutputMode::Full {
                return_sequence: opts.return_sequence,
                outputs,
            },
        )
    }

    pub(super) fn add_gru_cell_op_impl<'a>(
        graph: &GraphInfo,
        network: &mut trtx::NetworkDefinition<'a>,
        tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
        operation: &Operation,
    ) -> Result<(), GraphError> {
        let Operation::GruCell {
            input,
            weight,
            recurrence,
            hidden_state,
            hidden_size,
            options,
            outputs,
        } = operation
        else {
            return Err(gru_err("internal: expected GruCell operation"));
        };

        let opts = options.clone().unwrap_or_default();
        let hidden_size = gru_validate_hidden_size(*hidden_size, "gruCell")?;
        let input_operand = graph
            .operand(*input)
            .ok_or_else(|| gru_err("gruCell input operand"))?;

        // gruCell is a single-step GRU (ONNX lowers it to GRU with S=1).
        let params = GruLoopParams {
            seq_input_id: *input,
            weight_id: *weight,
            recurrence_id: *recurrence,
            bias_id: opts.bias,
            recurrent_bias_id: opts.recurrent_bias,
            initial_hidden_id: Some(*hidden_state),
            hidden_size,
            seq_steps: 1,
            direction: GruDirection::Forward,
            linear_before_reset: opts.reset_after,
            activations: gru_parse_activations(opts.activations.as_deref()),
            layout_rzn: opts.layout.eq_ignore_ascii_case("rzn"),
            input_dtype: input_operand.descriptor.data_type,
            batch_size: gru_batch_size(&input_operand.descriptor.shape),
            unsqueeze_sequence_axis: true,
        };

        gru_run_loop(
            graph,
            network,
            tensor_map,
            &params,
            GruOutputMode::Cell {
                output_id: outputs[0],
            },
        )
    }
}

/// Shared GRU loop body used by both `gru` and `gruCell`.
fn gru_run_loop<'a>(
    graph: &GraphInfo,
    network: &mut trtx::NetworkDefinition<'a>,
    tensor_map: &mut HashMap<u32, trtx::Tensor<'a>>,
    params: &GruLoopParams,
    output_mode: GruOutputMode<'_>,
) -> Result<(), GraphError> {
    let num_directions = params.direction.num_directions();
    let input_size = gru_input_size(graph, params.seq_input_id)?;

    let input_tensor = gru_tensor(tensor_map, params.seq_input_id, "gru seq input")?;
    let seq_input = if params.unsqueeze_sequence_axis {
        gru_reshape(
            network,
            &input_tensor,
            &[1, params.batch_size as i64, input_size],
            "gru unsqueeze seq input",
        )?
    } else {
        input_tensor
    };

    let prepared = gru_prepare_weights(
        graph,
        network,
        tensor_map,
        params.weight_id,
        params.recurrence_id,
        params.bias_id,
        params.recurrent_bias_id,
        num_directions,
        params.hidden_size,
        params.layout_rzn,
    )?;

    let gate_output_shape = [
        num_directions as i64,
        params.batch_size as i64,
        params.hidden_size as i64,
    ];
    let initial_hidden = gru_initial_hidden_state(
        network,
        tensor_map,
        params.initial_hidden_id,
        params.direction,
        &gate_output_shape,
        params.input_dtype,
    )?;

    let mut loop_body = network
        .add_loop()
        .map_err(|e| gru_err_fmt(format!("add_loop: {e}")))?;
    let trip = gru_i32_scalar(network, params.seq_steps, "gru trip limit")?;
    loop_body
        .add_trip_limit(network, &trip, TripLimit::kCOUNT)
        .map_err(|e| gru_err_fmt(format!("add_trip_limit: {e}")))?;

    let iteration_input = gru_iteration_input(
        network,
        &mut loop_body,
        &seq_input,
        params.direction,
        num_directions,
    )?;

    let mut ht1_layer = loop_body
        .add_recurrence(network, &initial_hidden)
        .map_err(|e| gru_err_fmt(format!("add_recurrence: {e}")))?;
    let ht1 = ht1_layer
        .output(network, 0)
        .map_err(|e| gru_err_fmt(format!("recurrence output: {e}")))?;

    let ht = gru_compute_ht(
        network,
        &iteration_input,
        &ht1,
        params.hidden_size,
        params.linear_before_reset,
        params.activations,
        &prepared,
    )?;

    ht1_layer
        .set_input(network, 1, &ht)
        .map_err(|e| gru_err_fmt(format!("recurrence set_input: {e}")))?;

    let yh_layer = loop_body
        .add_loop_output(network, &ht1, LoopOutput::kLAST_VALUE, 0)
        .map_err(|e| gru_err_fmt(format!("loop last hidden: {e}")))?;
    let yh = yh_layer
        .output(network, 0)
        .map_err(|e| gru_err_fmt(format!("loop last hidden output: {e}")))?;

    match output_mode {
        GruOutputMode::Cell { output_id } => {
            let out = gru_fit_cell_output(network, &yh, params.batch_size, params.hidden_size)?;
            tensor_map.insert(output_id, out);
        }
        GruOutputMode::Full {
            return_sequence,
            outputs,
        } => {
            let single_pass_shape = [1_i64, params.batch_size as i64, params.hidden_size as i64];
            let reverse = params.direction == GruDirection::Reverse;

            if return_sequence {
                let (seq_output_id, last_output_id) =
                    gru_resolve_output_ids(graph, outputs, return_sequence);
                let seq_tensor = gru_concat_loop_outputs(
                    network,
                    &mut loop_body,
                    &ht,
                    num_directions,
                    &single_pass_shape,
                    params.seq_steps,
                    reverse,
                )?;
                let seq_final = gru_fit_sequence_output(
                    graph,
                    network,
                    seq_output_id,
                    &seq_tensor,
                    params.direction,
                )?;
                tensor_map.insert(seq_output_id, seq_final);

                let yh_final = gru_fit_hidden_output(graph, network, last_output_id, &yh)?;
                tensor_map.insert(last_output_id, yh_final);
            } else {
                let last_output_id = outputs[0];
                let yh_final = gru_fit_hidden_output(graph, network, last_output_id, &yh)?;
                tensor_map.insert(last_output_id, yh_final);
            }
        }
    }

    Ok(())
}

fn gru_validate_hidden_size(hidden_size: u32, label: &str) -> Result<i32, GraphError> {
    let h = i32::try_from(hidden_size).map_err(|_| gru_err_fmt(format!("{label} hidden_size")))?;
    if h <= 0 {
        return Err(gru_err_fmt(format!("{label} hidden_size must be positive")));
    }
    Ok(h)
}

/// Squeeze `[1, batch, hidden]` (or `[numDir, batch, hidden]` with numDir=1) to `[batch, hidden]`.
fn gru_fit_cell_output<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    tensor: &trtx::Tensor<'a>,
    batch_size: u32,
    hidden_size: i32,
) -> Result<trtx::Tensor<'a>, GraphError> {
    gru_reshape(
        network,
        tensor,
        &[batch_size as i64, hidden_size as i64],
        "gruCell output",
    )
}

fn gru_err(reason: &str) -> GraphError {
    GraphError::ConversionFailed {
        format: "trtx".to_string(),
        reason: reason.to_string(),
    }
}

fn gru_err_fmt(reason: String) -> GraphError {
    GraphError::ConversionFailed {
        format: "trtx".to_string(),
        reason,
    }
}

fn gru_tensor<'a>(
    tensor_map: &HashMap<u32, trtx::Tensor<'a>>,
    id: u32,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    tensor_map
        .get(&id)
        .cloned()
        .ok_or_else(|| gru_err_fmt(format!("{label} operand {id} not found")))
}

fn gru_batch_size(shape: &[crate::graph::Dimension]) -> u32 {
    match shape.len() {
        2 => get_static_or_max_size(&shape[0]),
        3 => get_static_or_max_size(&shape[1]),
        _ => 1,
    }
}

fn gru_input_size(graph: &GraphInfo, input_id: u32) -> Result<i64, GraphError> {
    let shape = &graph
        .operand(input_id)
        .ok_or_else(|| gru_err("gru input operand"))?
        .descriptor
        .shape;
    let size = match shape.len() {
        2 => get_static_or_max_size(&shape[1]),
        3 => get_static_or_max_size(&shape[2]),
        _ => return Err(gru_err("gru input must have rank 2 or 3")),
    };
    Ok(size as i64)
}

fn gru_parse_activations(names: Option<&[String]>) -> GruActivations {
    let mut acts = GruActivations::default();
    if let Some(list) = names {
        if let Some(a) = list.first() {
            acts.gate = gru_activation_type(a);
        }
        if let Some(a) = list.get(1) {
            acts.hidden = gru_activation_type(a);
        }
    }
    acts
}

fn gru_check_bidirectional_activations(names: Option<&[String]>) -> Result<(), GraphError> {
    if let Some(list) = names {
        if list.len() > 2 {
            return Err(gru_err(
                "bidirectional gru requires the same activations for forward and reverse passes",
            ));
        }
    }
    Ok(())
}

fn gru_activation_type(name: &str) -> ActivationType {
    match name.to_ascii_lowercase().as_str() {
        "tanh" => ActivationType::kTANH,
        "relu" => ActivationType::kRELU,
        "sigmoid" => ActivationType::kSIGMOID,
        _ => ActivationType::kSIGMOID,
    }
}

fn gru_reshape<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    tensor: &trtx::Tensor<'a>,
    dims: &[i64],
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let mut shuffle = network
        .add_shuffle(tensor)
        .map_err(|e| gru_err_fmt(format!("{label}: shuffle: {e}")))?;
    shuffle
        .set_reshape_dimensions(network, dims)
        .map_err(|e| gru_err_fmt(format!("{label}: reshape {dims:?}: {e}")))?;
    shuffle
        .output(&*network, 0)
        .map_err(|e| gru_err_fmt(format!("{label}: output: {e}")))
}

fn gru_ew<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    a: &trtx::Tensor<'a>,
    b: &trtx::Tensor<'a>,
    op: Ew,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    network
        .add_elementwise(a, b, op)
        .map_err(|e| gru_err_fmt(format!("{label}: {e}")))?
        .output(&*network, 0)
        .map_err(|e| gru_err_fmt(format!("{label} output: {e}")))
}

fn gru_matmul_t<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    a: &trtx::Tensor<'a>,
    b: &trtx::Tensor<'a>,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    network
        .add_matrix_multiply(a, MatOp::kNONE, b, MatOp::kTRANSPOSE)
        .map_err(|e| gru_err_fmt(format!("{label}: {e}")))?
        .output(&*network, 0)
        .map_err(|e| gru_err_fmt(format!("{label} output: {e}")))
}

fn gru_activation<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    input: &trtx::Tensor<'a>,
    act: ActivationType,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let layer = network
        .add_activation(input, act)
        .map_err(|e| gru_err_fmt(format!("{label}: {e}")))?;
    layer
        .output(&*network, 0)
        .map_err(|e| gru_err_fmt(format!("{label} output: {e}")))
}

fn gru_i32_scalar<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    value: i32,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    network
        .add_small_constant_copied(&[], &value.to_le_bytes(), trtx::DataType::kINT32, None)
        .map_err(|e| gru_err_fmt(format!("{label}: {e}")))?
        .output(&*network, 0)
        .map_err(|e| gru_err_fmt(format!("{label} output: {e}")))
}

fn gru_f32_one<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    like: &trtx::Tensor<'a>,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let (bytes, dt) = if like.get_type(&*network) == trtx::DataType::kHALF {
        (
            f16::from_f32(1.0).to_le_bytes().to_vec(),
            trtx::DataType::kHALF,
        )
    } else {
        (1.0f32.to_le_bytes().to_vec(), trtx::DataType::kFLOAT)
    };
    network
        .add_small_constant_copied(&[1, 1, 1], &bytes, dt, None)
        .map_err(|e| gru_err_fmt(format!("gru one constant: {e}")))?
        .output(&*network, 0)
        .map_err(|e| gru_err_fmt(format!("gru one constant output: {e}")))
}

fn gru_slice_weights<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    tensor: &trtx::Tensor<'a>,
    gate_start: i32,
    gate_rows: i32,
    num_directions: i32,
    _hidden_size: i32,
    input_or_hidden_cols: i32,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let start = [0_i64, gate_start as i64, 0_i64];
    let size = [
        num_directions as i64,
        gate_rows as i64,
        input_or_hidden_cols as i64,
    ];
    network
        .add_slice(tensor, &start, &size, &[1, 1, 1])
        .map_err(|e| gru_err_fmt(format!("{label}: slice: {e}")))?
        .output(&*network, 0)
        .map_err(|e| gru_err_fmt(format!("{label}: output: {e}")))
}

fn gru_reorder_gates_rzn_to_zrn<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    tensor: &trtx::Tensor<'a>,
    gate_axis: i32,
    hidden_size: i32,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let h = hidden_size as i64;
    let dims = tensor
        .dimensions(&*network)
        .map_err(|e| gru_err_fmt(e.to_string()))?;
    let rank = dims.len() as i32;
    let mut chunks = Vec::with_capacity(3);
    for (gate_idx, gate_name) in [(1, "r"), (0, "z"), (2, "n")] {
        let mut start = vec![0_i64; rank as usize];
        let mut size = dims.to_vec();
        start[gate_axis as usize] = gate_idx as i64 * h;
        size[gate_axis as usize] = h;
        let chunk = network
            .add_slice(tensor, &start, &size, &vec![1_i64; rank as usize])
            .map_err(|e| gru_err_fmt(format!("{label} {gate_name} slice: {e}")))?
            .output(&*network, 0)
            .map_err(|e| gru_err_fmt(format!("{label} {gate_name} output: {e}")))?;
        chunks.push(chunk);
    }
    let refs: Vec<&trtx::Tensor<'a>> = chunks.iter().collect();
    let mut concat = network
        .add_concatenation(&refs)
        .map_err(|e| gru_err_fmt(format!("{label} reorder concat: {e}")))?;
    concat.set_axis(network, gate_axis);
    concat
        .output(&*network, 0)
        .map_err(|e| gru_err_fmt(format!("{label} reorder output: {e}")))
}

fn gru_maybe_reorder_gates<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    tensor: &trtx::Tensor<'a>,
    gate_axis: i32,
    hidden_size: i32,
    layout_rzn: bool,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    if layout_rzn {
        gru_reorder_gates_rzn_to_zrn(network, tensor, gate_axis, hidden_size, label)
    } else {
        Ok(tensor.clone())
    }
}

fn gru_to_3d<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    tensor: &trtx::Tensor<'a>,
    num_directions: i32,
    rows: i32,
    cols: i32,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let dims = tensor
        .dimensions(&*network)
        .map_err(|e| gru_err_fmt(e.to_string()))?;
    if dims.len() == 3 {
        return Ok(tensor.clone());
    }
    if dims.len() == 2 {
        return gru_reshape(
            network,
            tensor,
            &[num_directions as i64, rows as i64, cols as i64],
            label,
        );
    }
    Err(gru_err_fmt(format!(
        "{label}: expected rank 2 or 3, got {}D",
        dims.len()
    )))
}

fn gru_prepare_weights<'a>(
    graph: &GraphInfo,
    network: &mut trtx::NetworkDefinition<'a>,
    tensor_map: &HashMap<u32, trtx::Tensor<'a>>,
    weight_id: u32,
    recurrence_id: u32,
    bias_id: Option<u32>,
    recurrent_bias_id: Option<u32>,
    num_directions: i32,
    hidden_size: i32,
    layout_rzn: bool,
) -> Result<GruPreparedWeights<'a>, GraphError> {
    let input_size = {
        let w_shape = graph
            .operand(weight_id)
            .ok_or_else(|| gru_err("gru weight operand"))?
            .descriptor
            .static_or_max_shape();
        match w_shape.len() {
            2 => w_shape[1] as i32,
            3 => w_shape[2] as i32,
            _ => return Err(gru_err("gru weight must be rank 2 or 3")),
        }
    };

    let w = gru_to_3d(
        network,
        &gru_tensor(tensor_map, weight_id, "gru weight")?,
        num_directions,
        GRU_NUM_GATES * hidden_size,
        input_size,
        "gru weight 3d",
    )?;
    let r = gru_to_3d(
        network,
        &gru_tensor(tensor_map, recurrence_id, "gru recurrence")?,
        num_directions,
        GRU_NUM_GATES * hidden_size,
        hidden_size,
        "gru recurrence 3d",
    )?;

    let w = gru_maybe_reorder_gates(network, &w, 1, hidden_size, layout_rzn, "gru weight")?;
    let r = gru_maybe_reorder_gates(network, &r, 1, hidden_size, layout_rzn, "gru recurrence")?;

    let weights_zr = gru_slice_weights(
        network,
        &w,
        0,
        2 * hidden_size,
        num_directions,
        hidden_size,
        input_size,
        "gru W zr",
    )?;
    let weights_h = gru_slice_weights(
        network,
        &w,
        2 * hidden_size,
        hidden_size,
        num_directions,
        hidden_size,
        input_size,
        "gru W h",
    )?;
    let recurrence_zr = gru_slice_weights(
        network,
        &r,
        0,
        2 * hidden_size,
        num_directions,
        hidden_size,
        hidden_size,
        "gru R zr",
    )?;
    let recurrence_h = gru_slice_weights(
        network,
        &r,
        2 * hidden_size,
        hidden_size,
        num_directions,
        hidden_size,
        hidden_size,
        "gru R h",
    )?;

    let mut bias_zr = None;
    let mut bias_h = None;
    let mut recurrence_bias_zr = None;
    let mut recurrence_bias_h = None;

    if bias_id.is_some() || recurrent_bias_id.is_some() {
        let bias = if let Some(id) = bias_id {
            gru_tensor(tensor_map, id, "gru bias")?
        } else {
            gru_zero_bias(network, num_directions, hidden_size, "gru bias zero")?
        };
        let rbias = if let Some(id) = recurrent_bias_id {
            gru_tensor(tensor_map, id, "gru recurrent bias")?
        } else {
            gru_zero_bias(network, num_directions, hidden_size, "gru rbias zero")?
        };

        let gate_axis = if bias.dimensions(&*network).map(|d| d.len()).unwrap_or(1) == 1 {
            0_i32
        } else {
            1_i32
        };

        let bias = gru_maybe_reorder_gates(
            network,
            &bias,
            gate_axis,
            hidden_size,
            layout_rzn,
            "gru bias",
        )?;
        let rbias = gru_maybe_reorder_gates(
            network,
            &rbias,
            gate_axis,
            hidden_size,
            layout_rzn,
            "gru recurrent bias",
        )?;

        let combined = {
            let refs = [&bias, &rbias];
            let mut concat = network
                .add_concatenation(&refs)
                .map_err(|e| gru_err_fmt(format!("gru combine bias: {e}")))?;
            concat.set_axis(network, gate_axis);
            concat
                .output(&*network, 0)
                .map_err(|e| gru_err_fmt(format!("gru combine bias output: {e}")))?
        };

        let combined = gru_reshape(
            network,
            &combined,
            &[num_directions as i64, 1, (6 * hidden_size) as i64],
            "gru bias 3d",
        )?;

        bias_zr = Some(gru_slice_bias(
            network,
            &combined,
            0,
            2 * hidden_size,
            num_directions,
            "gru Wb zr",
        )?);
        bias_h = Some(gru_slice_bias(
            network,
            &combined,
            2 * hidden_size,
            hidden_size,
            num_directions,
            "gru Wb h",
        )?);
        recurrence_bias_zr = Some(gru_slice_bias(
            network,
            &combined,
            GRU_NUM_GATES * hidden_size,
            2 * hidden_size,
            num_directions,
            "gru Rb zr",
        )?);
        recurrence_bias_h = Some(gru_slice_bias(
            network,
            &combined,
            (GRU_NUM_GATES + 2) * hidden_size,
            hidden_size,
            num_directions,
            "gru Rb h",
        )?);
    }

    Ok(GruPreparedWeights {
        weights_zr,
        weights_h,
        recurrence_zr,
        recurrence_h,
        bias_zr,
        bias_h,
        recurrence_bias_zr,
        recurrence_bias_h,
    })
}

fn gru_zero_bias<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    num_directions: i32,
    hidden_size: i32,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let n = (num_directions * GRU_NUM_GATES * hidden_size) as usize;
    let bytes = vec![0u8; n * 4];
    network
        .add_small_constant_copied(
            &[num_directions as i64, (GRU_NUM_GATES * hidden_size) as i64],
            &bytes,
            trtx::DataType::kFLOAT,
            None,
        )
        .map_err(|e| gru_err_fmt(format!("{label}: {e}")))?
        .output(&*network, 0)
        .map_err(|e| gru_err_fmt(format!("{label} output: {e}")))
}

fn gru_slice_bias<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    tensor: &trtx::Tensor<'a>,
    gate_start: i32,
    gate_rows: i32,
    num_directions: i32,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let start = [0_i64, 0_i64, gate_start as i64];
    let size = [num_directions as i64, 1_i64, gate_rows as i64];
    network
        .add_slice(tensor, &start, &size, &[1, 1, 1])
        .map_err(|e| gru_err_fmt(format!("{label}: {e}")))?
        .output(&*network, 0)
        .map_err(|e| gru_err_fmt(format!("{label} output: {e}")))
}

fn gru_isolate_gate<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    gates: &trtx::Tensor<'a>,
    gate_index: i32,
    hidden_size: i32,
    num_directions: i32,
    batch_size: i64,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let start = [0_i64, 0_i64, (gate_index * hidden_size) as i64];
    let size = [num_directions as i64, batch_size, hidden_size as i64];
    network
        .add_slice(gates, &start, &size, &[1, 1, 1])
        .map_err(|e| gru_err_fmt(format!("gru isolate gate {gate_index}: {e}")))?
        .output(&*network, 0)
        .map_err(|e| gru_err_fmt(format!("gru isolate gate {gate_index} output: {e}")))
}

fn gru_compute_ht<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    iteration_input: &trtx::Tensor<'a>,
    ht1: &trtx::Tensor<'a>,
    hidden_size: i32,
    linear_before_reset: bool,
    activations: GruActivations,
    prepared: &GruPreparedWeights<'a>,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let dims = iteration_input
        .dimensions(&*network)
        .map_err(|e| gru_err_fmt(e.to_string()))?;
    let num_directions = dims[0];
    let batch_size = dims[1];

    // stackedZR = f(X*Wzr^T + H*Rzr^T + Wb + Rb)
    let xt_wzr = gru_matmul_t(network, iteration_input, &prepared.weights_zr, "gru X Wzr")?;
    let ht_rzr = gru_matmul_t(network, ht1, &prepared.recurrence_zr, "gru H Rzr")?;
    let mut stacked_zr = gru_ew(network, &xt_wzr, &ht_rzr, Ew::kSUM, "gru zr matmul sum")?;
    if let (Some(wb), Some(rb)) = (&prepared.bias_zr, &prepared.recurrence_bias_zr) {
        stacked_zr = gru_ew(network, &stacked_zr, wb, Ew::kSUM, "gru + Wb zr")?;
        stacked_zr = gru_ew(network, &stacked_zr, rb, Ew::kSUM, "gru + Rb zr")?;
    }
    let stacked_zr = gru_activation(network, &stacked_zr, activations.gate, "gru zr act")?;

    let zt = gru_isolate_gate(
        network,
        &stacked_zr,
        0,
        hidden_size,
        num_directions as i32,
        batch_size,
    )?;
    let rt = gru_isolate_gate(
        network,
        &stacked_zr,
        1,
        hidden_size,
        num_directions as i32,
        batch_size,
    )?;

    let xt_wh = gru_matmul_t(network, iteration_input, &prepared.weights_h, "gru X Wh")?;
    let ht = if linear_before_reset {
        let ht_rh = gru_matmul_t(network, ht1, &prepared.recurrence_h, "gru H Rh")?;
        let mut inner = ht_rh;
        if let Some(rb) = &prepared.recurrence_bias_h {
            inner = gru_ew(network, &inner, rb, Ew::kSUM, "gru + Rb h")?;
        }
        let rt_inner = gru_ew(network, &rt, &inner, Ew::kPROD, "gru r * (H Rh + Rb)")?;
        let mut act_in = gru_ew(network, &xt_wh, &rt_inner, Ew::kSUM, "gru h act in")?;
        if let Some(wb) = &prepared.bias_h {
            act_in = gru_ew(network, &act_in, wb, Ew::kSUM, "gru + Wb h")?;
        }
        gru_activation(network, &act_in, activations.hidden, "gru h act")?
    } else {
        let rt_ht = gru_ew(network, &rt, ht1, Ew::kPROD, "gru r * H")?;
        let rt_ht_rh = gru_matmul_t(network, &rt_ht, &prepared.recurrence_h, "gru rH * Rh")?;
        let mut act_in = gru_ew(network, &xt_wh, &rt_ht_rh, Ew::kSUM, "gru h act in")?;
        if let (Some(wb), Some(rb)) = (&prepared.bias_h, &prepared.recurrence_bias_h) {
            let bias_sum = gru_ew(network, wb, rb, Ew::kSUM, "gru h bias sum")?;
            act_in = gru_ew(network, &act_in, &bias_sum, Ew::kSUM, "gru + h bias")?;
        }
        gru_activation(network, &act_in, activations.hidden, "gru h act")?
    };

    let one = gru_f32_one(network, &zt)?;
    let one_minus_z = gru_ew(network, &one, &zt, Ew::kSUB, "gru 1-z")?;
    let left = gru_ew(network, &one_minus_z, &ht, Ew::kPROD, "gru (1-z)*h")?;
    let right = gru_ew(network, &zt, ht1, Ew::kPROD, "gru z*H")?;
    gru_ew(network, &left, &right, Ew::kSUM, "gru Ht")
}

fn gru_iteration_input<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    loop_body: &mut trtx::network::Loop<'a>,
    seq_input: &trtx::Tensor<'a>,
    direction: GruDirection,
    num_directions: i32,
) -> Result<trtx::Tensor<'a>, GraphError> {
    match direction {
        GruDirection::Forward => {
            let mut it = loop_body
                .add_iterator(network, seq_input, 0, false)
                .map_err(|e| gru_err_fmt(format!("gru iterator: {e}")))?;
            it.set_axis(network, 0);
            let step = it
                .output(network, 0)
                .map_err(|e| gru_err_fmt(format!("gru iterator output: {e}")))?;
            let dims = step
                .dimensions(&*network)
                .map_err(|e| gru_err_fmt(e.to_string()))?;
            gru_reshape(
                network,
                &step,
                &[num_directions as i64, dims[0], dims[1]],
                "gru iter input",
            )
        }
        GruDirection::Reverse => {
            let mut it = loop_body
                .add_iterator(network, seq_input, 0, true)
                .map_err(|e| gru_err_fmt(format!("gru reverse iterator: {e}")))?;
            it.set_axis(network, 0);
            let step = it
                .output(network, 0)
                .map_err(|e| gru_err_fmt(format!("gru reverse iterator output: {e}")))?;
            let dims = step
                .dimensions(&*network)
                .map_err(|e| gru_err_fmt(e.to_string()))?;
            gru_reshape(
                network,
                &step,
                &[num_directions as i64, dims[0], dims[1]],
                "gru reverse iter input",
            )
        }
        GruDirection::Bidirectional => {
            let mut fwd = loop_body
                .add_iterator(network, seq_input, 0, false)
                .map_err(|e| gru_err_fmt(format!("gru fwd iterator: {e}")))?;
            fwd.set_axis(network, 0);
            let fwd_step = fwd
                .output(network, 0)
                .map_err(|e| gru_err_fmt(format!("gru fwd iterator output: {e}")))?;
            let fwd_dims = fwd_step
                .dimensions(&*network)
                .map_err(|e| gru_err_fmt(e.to_string()))?;
            let fwd_3d = gru_reshape(
                network,
                &fwd_step,
                &[1, fwd_dims[0], fwd_dims[1]],
                "gru fwd 3d",
            )?;

            let mut rev = loop_body
                .add_iterator(network, seq_input, 0, true)
                .map_err(|e| gru_err_fmt(format!("gru rev iterator: {e}")))?;
            rev.set_axis(network, 0);
            let rev_step = rev
                .output(network, 0)
                .map_err(|e| gru_err_fmt(format!("gru rev iterator output: {e}")))?;
            let rev_dims = rev_step
                .dimensions(&*network)
                .map_err(|e| gru_err_fmt(e.to_string()))?;
            let rev_3d = gru_reshape(
                network,
                &rev_step,
                &[1, rev_dims[0], rev_dims[1]],
                "gru rev 3d",
            )?;

            let refs = [&fwd_3d, &rev_3d];
            let mut concat = network
                .add_concatenation(&refs)
                .map_err(|e| gru_err_fmt(format!("gru bidi concat: {e}")))?;
            concat.set_axis(network, 0);
            concat
                .output(&*network, 0)
                .map_err(|e| gru_err_fmt(format!("gru bidi concat output: {e}")))
        }
    }
}

fn gru_initial_hidden_state<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    tensor_map: &HashMap<u32, trtx::Tensor<'a>>,
    initial_id: Option<u32>,
    direction: GruDirection,
    shape: &[i64; 3],
    dtype: DataType,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let base = if let Some(id) = initial_id {
        let t = gru_tensor(tensor_map, id, "gru initial hidden")?;
        gru_reshape(network, &t, shape, "gru initial hidden reshape")?
    } else {
        gru_zeros(network, shape, dtype, "gru initial hidden zero")?
    };

    if direction == GruDirection::Bidirectional {
        let refs = [&base, &base];
        let mut concat = network
            .add_concatenation(&refs)
            .map_err(|e| gru_err_fmt(format!("gru bidi initial hidden: {e}")))?;
        concat.set_axis(network, 0);
        return concat
            .output(&*network, 0)
            .map_err(|e| gru_err_fmt(format!("gru bidi initial hidden output: {e}")));
    }

    if direction == GruDirection::Reverse {
        return gru_reshape(network, &base, shape, "gru reverse initial hidden");
    }

    gru_reshape(network, &base, shape, "gru initial hidden 3d")
}

fn gru_zeros<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    shape: &[i64; 3],
    dtype: DataType,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let count: usize = shape.iter().map(|&d| d as usize).product();
    let (trt_dt, elem) = match dtype {
        DataType::Float16 => (trtx::DataType::kHALF, 2),
        _ => (trtx::DataType::kFLOAT, 4),
    };
    network
        .add_small_constant_copied(shape, &vec![0u8; count * elem], trt_dt, None)
        .map_err(|e| gru_err_fmt(format!("{label}: {e}")))?
        .output(&*network, 0)
        .map_err(|e| gru_err_fmt(format!("{label} output: {e}")))
}

fn gru_concat_loop_outputs<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    loop_body: &mut trtx::network::Loop<'a>,
    ht: &trtx::Tensor<'a>,
    num_directions: i32,
    single_pass_shape: &[i64; 3],
    seq_steps: i32,
    reverse: bool,
) -> Result<trtx::Tensor<'a>, GraphError> {
    if num_directions == 2 {
        let fwd = gru_slice_pass(network, ht, 0, single_pass_shape, "gru fwd pass")?;
        let rev = gru_slice_pass(network, ht, 1, single_pass_shape, "gru rev pass")?;
        let mut fwd_out = loop_body
            .add_loop_output(network, &fwd, LoopOutput::kCONCATENATE, 0)
            .map_err(|e| gru_err_fmt(format!("gru fwd loop out: {e}")))?;
        let trip = gru_i32_scalar(network, seq_steps, "gru seq len")?;
        fwd_out
            .set_input(network, 1, &trip)
            .map_err(|e| gru_err_fmt(format!("gru fwd loop trip: {e}")))?;
        let fwd_tensor = fwd_out
            .output(network, 0)
            .map_err(|e| gru_err_fmt(format!("gru fwd loop tensor: {e}")))?;

        let mut rev_out = loop_body
            .add_loop_output(network, &rev, LoopOutput::kREVERSE, 0)
            .map_err(|e| gru_err_fmt(format!("gru rev loop out: {e}")))?;
        rev_out
            .set_input(network, 1, &trip)
            .map_err(|e| gru_err_fmt(format!("gru rev loop trip: {e}")))?;
        let rev_tensor = rev_out
            .output(network, 0)
            .map_err(|e| gru_err_fmt(format!("gru rev loop tensor: {e}")))?;

        let refs = [&fwd_tensor, &rev_tensor];
        let mut concat = network
            .add_concatenation(&refs)
            .map_err(|e| gru_err_fmt(format!("gru bidi seq concat: {e}")))?;
        concat.set_axis(network, 1);
        return concat
            .output(&*network, 0)
            .map_err(|e| gru_err_fmt(format!("gru bidi seq output: {e}")));
    }

    let kind = if reverse {
        LoopOutput::kREVERSE
    } else {
        LoopOutput::kCONCATENATE
    };
    let mut out = loop_body
        .add_loop_output(network, ht, kind, 0)
        .map_err(|e| gru_err_fmt(format!("gru loop out: {e}")))?;
    let trip = gru_i32_scalar(network, seq_steps, "gru seq len")?;
    out.set_input(network, 1, &trip)
        .map_err(|e| gru_err_fmt(format!("gru loop trip: {e}")))?;
    out.output(network, 0)
        .map_err(|e| gru_err_fmt(format!("gru loop tensor: {e}")))
}

fn gru_slice_pass<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    ht: &trtx::Tensor<'a>,
    pass_index: i32,
    single_pass_shape: &[i64; 3],
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let start = [pass_index as i64, 0_i64, 0_i64];
    network
        .add_slice(ht, &start, single_pass_shape, &[1, 1, 1])
        .map_err(|e| gru_err_fmt(format!("{label}: {e}")))?
        .output(&*network, 0)
        .map_err(|e| gru_err_fmt(format!("{label} output: {e}")))
}

fn gru_resolve_output_ids(graph: &GraphInfo, outputs: &[u32], return_sequence: bool) -> (u32, u32) {
    if return_sequence && outputs.len() >= 2 {
        let name0 = graph
            .operand(outputs[0])
            .and_then(|o| o.name.clone())
            .unwrap_or_default();
        if name0.ends_with('1') {
            return (outputs[1], outputs[0]);
        }
        return (outputs[0], outputs[1]);
    }
    let last = outputs[0];
    let seq = outputs.get(1).copied().unwrap_or(last);
    (seq, last)
}

fn gru_fit_sequence_output<'a>(
    graph: &GraphInfo,
    network: &mut trtx::NetworkDefinition<'a>,
    output_id: u32,
    tensor: &trtx::Tensor<'a>,
    direction: GruDirection,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let expected_rank = graph
        .operand(output_id)
        .map(|o| o.descriptor.shape.len())
        .unwrap_or(4);
    if direction != GruDirection::Bidirectional && expected_rank == 3 {
        // [S, 1, B, H] -> [S, B, H]
        let dims = tensor
            .dimensions(&*network)
            .map_err(|e| gru_err_fmt(e.to_string()))?;
        if dims.len() == 4 && dims[1] == 1 {
            return gru_reshape(
                network,
                tensor,
                &[dims[0], dims[2], dims[3]],
                "gru squeeze direction from sequence",
            );
        }
    }
    Ok(tensor.clone())
}

fn gru_fit_hidden_output<'a>(
    graph: &GraphInfo,
    network: &mut trtx::NetworkDefinition<'a>,
    output_id: u32,
    tensor: &trtx::Tensor<'a>,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let expected_rank = graph
        .operand(output_id)
        .map(|o| o.descriptor.shape.len())
        .unwrap_or(3);
    let dims = tensor
        .dimensions(&*network)
        .map_err(|e| gru_err_fmt(e.to_string()))?;
    if expected_rank == 4 && dims.len() == 3 {
        return gru_reshape(
            network,
            tensor,
            &[1, dims[0], dims[1], dims[2]],
            "gru unsqueeze hidden to 4d",
        );
    }
    Ok(tensor.clone())
}

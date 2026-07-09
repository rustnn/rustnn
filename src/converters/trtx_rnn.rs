/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared TensorRT RNN loop utilities for GRU and LSTM lowering.
 */

use std::collections::HashMap;

use half::f16;
use trtx::ActivationType;
use trtx::ElementWiseOperation as Ew;
use trtx::LoopOutput;
use trtx::MatrixOperation as MatOp;
use trtx::TripLimit;

use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo, get_static_or_max_size};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RnnDirection {
    Forward,
    Reverse,
    Bidirectional,
}

impl RnnDirection {
    pub(crate) fn num_directions(self) -> i32 {
        match self {
            Self::Forward | Self::Reverse => 1,
            Self::Bidirectional => 2,
        }
    }

    pub(crate) fn from_webnn(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "backward" | "reverse" => Self::Reverse,
            "both" | "bidirectional" => Self::Bidirectional,
            _ => Self::Forward,
        }
    }
}

pub(crate) fn rnn_err(reason: &str) -> GraphError {
    GraphError::ConversionFailed {
        format: "trtx".to_string(),
        reason: reason.to_string(),
    }
}

pub(crate) fn rnn_err_fmt(reason: String) -> GraphError {
    GraphError::ConversionFailed {
        format: "trtx".to_string(),
        reason,
    }
}

pub(crate) fn rnn_tensor<'a>(
    tensor_map: &HashMap<u32, trtx::Tensor<'a>>,
    id: u32,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    tensor_map
        .get(&id)
        .cloned()
        .ok_or_else(|| rnn_err_fmt(format!("{label} operand {id} not found")))
}

pub(crate) fn rnn_batch_size(shape: &[crate::graph::Dimension]) -> u32 {
    match shape.len() {
        2 => get_static_or_max_size(&shape[0]),
        3 => get_static_or_max_size(&shape[1]),
        _ => 1,
    }
}

pub(crate) fn rnn_input_size(graph: &GraphInfo, input_id: u32) -> Result<i64, GraphError> {
    let shape = &graph
        .operand(input_id)
        .ok_or_else(|| rnn_err("rnn input operand"))?
        .descriptor
        .shape;
    let size = match shape.len() {
        2 => get_static_or_max_size(&shape[1]),
        3 => get_static_or_max_size(&shape[2]),
        _ => return Err(rnn_err("rnn input must have rank 2 or 3")),
    };
    Ok(size as i64)
}

pub(crate) fn rnn_validate_hidden_size(hidden_size: u32, label: &str) -> Result<i32, GraphError> {
    let h = i32::try_from(hidden_size).map_err(|_| rnn_err_fmt(format!("{label} hidden_size")))?;
    if h <= 0 {
        return Err(rnn_err_fmt(format!("{label} hidden_size must be positive")));
    }
    Ok(h)
}

pub(crate) fn rnn_activation_type(name: &str) -> ActivationType {
    match name.to_ascii_lowercase().as_str() {
        "tanh" => ActivationType::kTANH,
        "relu" => ActivationType::kRELU,
        "sigmoid" => ActivationType::kSIGMOID,
        _ => ActivationType::kSIGMOID,
    }
}

pub(crate) fn rnn_check_bidirectional_activations(
    names: Option<&[String]>,
    max_len: usize,
    label: &str,
) -> Result<(), GraphError> {
    if let Some(list) = names
        && list.len() > max_len
    {
        return Err(rnn_err_fmt(format!(
            "bidirectional {label} requires the same activations for forward and reverse passes"
        )));
    }
    Ok(())
}

pub(crate) fn rnn_reshape<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    tensor: &trtx::Tensor<'a>,
    dims: &[i64],
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let mut shuffle = network
        .add_shuffle(tensor)
        .map_err(|e| rnn_err_fmt(format!("{label}: shuffle: {e}")))?;
    shuffle
        .set_reshape_dimensions(network, dims)
        .map_err(|e| rnn_err_fmt(format!("{label}: reshape {dims:?}: {e}")))?;
    shuffle
        .output(&*network, 0)
        .map_err(|e| rnn_err_fmt(format!("{label}: output: {e}")))
}

pub(crate) fn rnn_ew<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    a: &trtx::Tensor<'a>,
    b: &trtx::Tensor<'a>,
    op: Ew,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    network
        .add_elementwise(a, b, op)
        .map_err(|e| rnn_err_fmt(format!("{label}: {e}")))?
        .output(&*network, 0)
        .map_err(|e| rnn_err_fmt(format!("{label} output: {e}")))
}

pub(crate) fn rnn_matmul_t<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    a: &trtx::Tensor<'a>,
    b: &trtx::Tensor<'a>,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    network
        .add_matrix_multiply(a, MatOp::kNONE, b, MatOp::kTRANSPOSE)
        .map_err(|e| rnn_err_fmt(format!("{label}: {e}")))?
        .output(&*network, 0)
        .map_err(|e| rnn_err_fmt(format!("{label} output: {e}")))
}

pub(crate) fn rnn_activation<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    input: &trtx::Tensor<'a>,
    act: ActivationType,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let layer = network
        .add_activation(input, act)
        .map_err(|e| rnn_err_fmt(format!("{label}: {e}")))?;
    layer
        .output(&*network, 0)
        .map_err(|e| rnn_err_fmt(format!("{label} output: {e}")))
}

pub(crate) fn rnn_i32_scalar<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    value: i32,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    network
        .add_small_constant_copied(&[], &value.to_le_bytes(), trtx::DataType::kINT32, None)
        .map_err(|e| rnn_err_fmt(format!("{label}: {e}")))?
        .output(&*network, 0)
        .map_err(|e| rnn_err_fmt(format!("{label} output: {e}")))
}

pub(crate) fn rnn_f32_one<'a>(
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
        .map_err(|e| rnn_err_fmt(format!("rnn one constant: {e}")))?
        .output(&*network, 0)
        .map_err(|e| rnn_err_fmt(format!("rnn one constant output: {e}")))
}

pub(crate) fn rnn_normalize_gate_tensor_2d<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    tensor: &trtx::Tensor<'a>,
    num_directions: i32,
    gate_rows: i32,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let dims = tensor
        .dimensions(&*network)
        .map_err(|e| rnn_err_fmt(e.to_string()))?;
    match dims.len() {
        1 => {
            let n = dims[0];
            let flat = gate_rows as i64;
            let dir = num_directions as i64;
            if n == flat {
                rnn_reshape(network, tensor, &[1, flat], label)
            } else if n == dir * flat {
                rnn_reshape(network, tensor, &[dir, flat], label)
            } else {
                Err(rnn_err_fmt(format!(
                    "{label}: 1D length {n} does not match [{flat}] or [{dir}, {flat}]"
                )))
            }
        }
        2 | 3 => Ok(*tensor),
        n => Err(rnn_err_fmt(format!("{label}: expected rank 1-3, got {n}D"))),
    }
}

pub(crate) fn rnn_to_3d<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    tensor: &trtx::Tensor<'a>,
    num_directions: i32,
    rows: i32,
    cols: i32,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let dims = tensor
        .dimensions(&*network)
        .map_err(|e| rnn_err_fmt(e.to_string()))?;
    if dims.len() == 3 {
        return Ok(*tensor);
    }
    if dims.len() == 2 {
        return rnn_reshape(
            network,
            tensor,
            &[num_directions as i64, rows as i64, cols as i64],
            label,
        );
    }
    if dims.len() == 1 {
        let flat = rows as i64 * cols as i64;
        if dims[0] == flat {
            let as_2d = rnn_reshape(network, tensor, &[1, flat], &format!("{label} 2d"))?;
            return rnn_reshape(
                network,
                &as_2d,
                &[num_directions as i64, rows as i64, cols as i64],
                label,
            );
        }
        if dims[0] == (num_directions as i64) * flat {
            let as_2d = rnn_reshape(
                network,
                tensor,
                &[num_directions as i64, flat],
                &format!("{label} 2d"),
            )?;
            return rnn_reshape(
                network,
                &as_2d,
                &[num_directions as i64, rows as i64, cols as i64],
                label,
            );
        }
    }
    Err(rnn_err_fmt(format!(
        "{label}: expected rank 2 or 3, got {}D",
        dims.len()
    )))
}

pub(crate) fn rnn_isolate_gate<'a>(
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
        .map_err(|e| rnn_err_fmt(format!("rnn isolate gate {gate_index}: {e}")))?
        .output(&*network, 0)
        .map_err(|e| rnn_err_fmt(format!("rnn isolate gate {gate_index} output: {e}")))
}

pub(crate) fn rnn_iteration_input<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    loop_body: &mut trtx::network::Loop<'a>,
    seq_input: &trtx::Tensor<'a>,
    direction: RnnDirection,
    num_directions: i32,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    match direction {
        RnnDirection::Forward => {
            let mut it = loop_body
                .add_iterator(network, seq_input, 0, false)
                .map_err(|e| rnn_err_fmt(format!("{label} iterator: {e}")))?;
            it.set_axis(network, 0);
            let step = it
                .output(network, 0)
                .map_err(|e| rnn_err_fmt(format!("{label} iterator output: {e}")))?;
            let dims = step
                .dimensions(&*network)
                .map_err(|e| rnn_err_fmt(e.to_string()))?;
            rnn_reshape(
                network,
                &step,
                &[num_directions as i64, dims[0], dims[1]],
                &format!("{label} iter input"),
            )
        }
        RnnDirection::Reverse => {
            let mut it = loop_body
                .add_iterator(network, seq_input, 0, true)
                .map_err(|e| rnn_err_fmt(format!("{label} reverse iterator: {e}")))?;
            it.set_axis(network, 0);
            let step = it
                .output(network, 0)
                .map_err(|e| rnn_err_fmt(format!("{label} reverse iterator output: {e}")))?;
            let dims = step
                .dimensions(&*network)
                .map_err(|e| rnn_err_fmt(e.to_string()))?;
            rnn_reshape(
                network,
                &step,
                &[num_directions as i64, dims[0], dims[1]],
                &format!("{label} reverse iter input"),
            )
        }
        RnnDirection::Bidirectional => {
            let mut fwd = loop_body
                .add_iterator(network, seq_input, 0, false)
                .map_err(|e| rnn_err_fmt(format!("{label} fwd iterator: {e}")))?;
            fwd.set_axis(network, 0);
            let fwd_step = fwd
                .output(network, 0)
                .map_err(|e| rnn_err_fmt(format!("{label} fwd iterator output: {e}")))?;
            let fwd_dims = fwd_step
                .dimensions(&*network)
                .map_err(|e| rnn_err_fmt(e.to_string()))?;
            let fwd_3d = rnn_reshape(
                network,
                &fwd_step,
                &[1, fwd_dims[0], fwd_dims[1]],
                &format!("{label} fwd 3d"),
            )?;

            let mut rev = loop_body
                .add_iterator(network, seq_input, 0, true)
                .map_err(|e| rnn_err_fmt(format!("{label} rev iterator: {e}")))?;
            rev.set_axis(network, 0);
            let rev_step = rev
                .output(network, 0)
                .map_err(|e| rnn_err_fmt(format!("{label} rev iterator output: {e}")))?;
            let rev_dims = rev_step
                .dimensions(&*network)
                .map_err(|e| rnn_err_fmt(e.to_string()))?;
            let rev_3d = rnn_reshape(
                network,
                &rev_step,
                &[1, rev_dims[0], rev_dims[1]],
                &format!("{label} rev 3d"),
            )?;

            let refs = [&fwd_3d, &rev_3d];
            let mut concat = network
                .add_concatenation(&refs)
                .map_err(|e| rnn_err_fmt(format!("{label} bidi concat: {e}")))?;
            concat.set_axis(network, 0);
            concat
                .output(&*network, 0)
                .map_err(|e| rnn_err_fmt(format!("{label} bidi concat output: {e}")))
        }
    }
}

pub(crate) fn rnn_initial_state<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    tensor_map: &HashMap<u32, trtx::Tensor<'a>>,
    initial_id: Option<u32>,
    direction: RnnDirection,
    shape: &[i64; 3],
    dtype: DataType,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let base = if let Some(id) = initial_id {
        let t = rnn_tensor(tensor_map, id, label)?;
        let dims = t
            .dimensions(&*network)
            .map_err(|e| rnn_err_fmt(e.to_string()))?;
        match dims.len() {
            2 => rnn_reshape(
                network,
                &t,
                &[1, shape[1], shape[2]],
                &format!("{label} unsqueeze"),
            )?,
            3 => t,
            _ => {
                return Err(rnn_err_fmt(format!(
                    "{label}: expected rank 2 or 3, got {}D",
                    dims.len()
                )));
            }
        }
    } else {
        // Zeros already use the full `[numDirections, batch, hidden]` shape (ONNX-TensorRT path).
        return rnn_zeros(network, shape, dtype, &format!("{label} zero"));
    };

    // WebNN may supply `[1, batch, hidden]` for bidirectional; ONNX expects `[2, batch, hidden]`.
    if direction == RnnDirection::Bidirectional && shape[0] == 2 {
        let dims = base
            .dimensions(&*network)
            .map_err(|e| rnn_err_fmt(e.to_string()))?;
        if dims[0] == 1 {
            let refs = [&base, &base];
            let mut concat = network
                .add_concatenation(&refs)
                .map_err(|e| rnn_err_fmt(format!("{label} bidi: {e}")))?;
            concat.set_axis(network, 0);
            return concat
                .output(&*network, 0)
                .map_err(|e| rnn_err_fmt(format!("{label} bidi output: {e}")));
        }
    }

    rnn_reshape(network, &base, shape, label)
}

pub(crate) fn rnn_zeros<'a>(
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
        .map_err(|e| rnn_err_fmt(format!("{label}: {e}")))?
        .output(&*network, 0)
        .map_err(|e| rnn_err_fmt(format!("{label} output: {e}")))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rnn_concat_loop_outputs<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    loop_body: &mut trtx::network::Loop<'a>,
    ht: &trtx::Tensor<'a>,
    num_directions: i32,
    single_pass_shape: &[i64; 3],
    seq_steps: i32,
    reverse: bool,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    if num_directions == 2 {
        let fwd = rnn_slice_pass(
            network,
            ht,
            0,
            single_pass_shape,
            &format!("{label} fwd pass"),
        )?;
        let rev = rnn_slice_pass(
            network,
            ht,
            1,
            single_pass_shape,
            &format!("{label} rev pass"),
        )?;
        let mut fwd_out = loop_body
            .add_loop_output(network, &fwd, LoopOutput::kCONCATENATE, 0)
            .map_err(|e| rnn_err_fmt(format!("{label} fwd loop out: {e}")))?;
        let trip = rnn_i32_scalar(network, seq_steps, &format!("{label} seq len"))?;
        fwd_out
            .set_input(network, 1, &trip)
            .map_err(|e| rnn_err_fmt(format!("{label} fwd loop trip: {e}")))?;
        let fwd_tensor = fwd_out
            .output(network, 0)
            .map_err(|e| rnn_err_fmt(format!("{label} fwd loop tensor: {e}")))?;

        let mut rev_out = loop_body
            .add_loop_output(network, &rev, LoopOutput::kREVERSE, 0)
            .map_err(|e| rnn_err_fmt(format!("{label} rev loop out: {e}")))?;
        rev_out
            .set_input(network, 1, &trip)
            .map_err(|e| rnn_err_fmt(format!("{label} rev loop trip: {e}")))?;
        let rev_tensor = rev_out
            .output(network, 0)
            .map_err(|e| rnn_err_fmt(format!("{label} rev loop tensor: {e}")))?;

        let refs = [&fwd_tensor, &rev_tensor];
        let mut concat = network
            .add_concatenation(&refs)
            .map_err(|e| rnn_err_fmt(format!("{label} bidi seq concat: {e}")))?;
        concat.set_axis(network, 1);
        return concat
            .output(&*network, 0)
            .map_err(|e| rnn_err_fmt(format!("{label} bidi seq output: {e}")));
    }

    let kind = if reverse {
        LoopOutput::kREVERSE
    } else {
        LoopOutput::kCONCATENATE
    };
    let mut out = loop_body
        .add_loop_output(network, ht, kind, 0)
        .map_err(|e| rnn_err_fmt(format!("{label} loop out: {e}")))?;
    let trip = rnn_i32_scalar(network, seq_steps, &format!("{label} seq len"))?;
    out.set_input(network, 1, &trip)
        .map_err(|e| rnn_err_fmt(format!("{label} loop trip: {e}")))?;
    out.output(network, 0)
        .map_err(|e| rnn_err_fmt(format!("{label} loop tensor: {e}")))
}

fn rnn_slice_pass<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    ht: &trtx::Tensor<'a>,
    pass_index: i32,
    single_pass_shape: &[i64; 3],
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let start = [pass_index as i64, 0_i64, 0_i64];
    network
        .add_slice(ht, &start, single_pass_shape, &[1, 1, 1])
        .map_err(|e| rnn_err_fmt(format!("{label}: {e}")))?
        .output(&*network, 0)
        .map_err(|e| rnn_err_fmt(format!("{label} output: {e}")))
}

pub(crate) fn rnn_fit_sequence_output<'a>(
    graph: &GraphInfo,
    network: &mut trtx::NetworkDefinition<'a>,
    output_id: u32,
    tensor: &trtx::Tensor<'a>,
    direction: RnnDirection,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let expected_rank = graph
        .operand(output_id)
        .map(|o| o.descriptor.shape.len())
        .unwrap_or(4);
    if direction != RnnDirection::Bidirectional && expected_rank == 3 {
        let dims = tensor
            .dimensions(&*network)
            .map_err(|e| rnn_err_fmt(e.to_string()))?;
        if dims.len() == 4 && dims[1] == 1 {
            return rnn_reshape(
                network,
                tensor,
                &[dims[0], dims[2], dims[3]],
                &format!("{label} squeeze direction from sequence"),
            );
        }
    }
    Ok(*tensor)
}

pub(crate) fn rnn_fit_hidden_output<'a>(
    graph: &GraphInfo,
    network: &mut trtx::NetworkDefinition<'a>,
    output_id: u32,
    tensor: &trtx::Tensor<'a>,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let expected_rank = graph
        .operand(output_id)
        .map(|o| o.descriptor.shape.len())
        .unwrap_or(3);
    let dims = tensor
        .dimensions(&*network)
        .map_err(|e| rnn_err_fmt(e.to_string()))?;
    if expected_rank == 4 && dims.len() == 3 {
        return rnn_reshape(
            network,
            tensor,
            &[1, dims[0], dims[1], dims[2]],
            &format!("{label} unsqueeze hidden to 4d"),
        );
    }
    Ok(*tensor)
}

pub(crate) fn rnn_fit_cell_output<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    tensor: &trtx::Tensor<'a>,
    batch_size: u32,
    hidden_size: i32,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    rnn_reshape(
        network,
        tensor,
        &[batch_size as i64, hidden_size as i64],
        label,
    )
}

/// Reorder stacked gate dimension from WebNN `ifgo` to ONNX `iofc`.
pub(crate) fn rnn_reorder_gates_ifgo_to_iofc<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    tensor: &trtx::Tensor<'a>,
    gate_axis: i32,
    hidden_size: i32,
    label: &str,
) -> Result<trtx::Tensor<'a>, GraphError> {
    let h = hidden_size as i64;
    let dims = tensor
        .dimensions(&*network)
        .map_err(|e| rnn_err_fmt(e.to_string()))?;
    let rank = dims.len() as i32;
    // ifgo[i,f,g,o] -> iofc[i,o,f,c] => source indices 0, 3, 1, 2
    let mut chunks = Vec::with_capacity(4);
    for (src_idx, gate_name) in [(0, "i"), (3, "o"), (1, "f"), (2, "c")] {
        let mut start = vec![0_i64; rank as usize];
        let mut size = dims.to_vec();
        start[gate_axis as usize] = src_idx as i64 * h;
        size[gate_axis as usize] = h;
        let chunk = network
            .add_slice(tensor, &start, &size, &vec![1_i64; rank as usize])
            .map_err(|e| rnn_err_fmt(format!("{label} {gate_name} slice: {e}")))?
            .output(&*network, 0)
            .map_err(|e| rnn_err_fmt(format!("{label} {gate_name} output: {e}")))?;
        chunks.push(chunk);
    }
    let refs: Vec<&trtx::Tensor<'a>> = chunks.iter().collect();
    let mut concat = network
        .add_concatenation(&refs)
        .map_err(|e| rnn_err_fmt(format!("{label} reorder concat: {e}")))?;
    concat.set_axis(network, gate_axis);
    concat
        .output(&*network, 0)
        .map_err(|e| rnn_err_fmt(format!("{label} reorder output: {e}")))
}

pub(crate) fn rnn_add_trip_limit<'a>(
    network: &mut trtx::NetworkDefinition<'a>,
    loop_body: &mut trtx::network::Loop<'a>,
    seq_steps: i32,
    label: &str,
) -> Result<(), GraphError> {
    let trip = rnn_i32_scalar(network, seq_steps, &format!("{label} trip limit"))?;
    loop_body
        .add_trip_limit(network, &trip, TripLimit::kCOUNT)
        .map_err(|e| rnn_err_fmt(format!("{label} add_trip_limit: {e}")))?;
    Ok(())
}

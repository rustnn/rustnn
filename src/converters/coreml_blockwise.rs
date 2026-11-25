/*
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

//! Keeps blockwise-quantized constant weights quantized through CoreML lowering.
//!
//! Quantized language-model graphs author each weight matrix as an explicit dequantization
//! of constants:
//!
//! ```text
//! q        int8 or uint8 constant [n, k]
//! cast     -> float16 or float32 [n, k]
//! sub      (cast, z)       optional; z is a one-element constant zero point
//! reshape  -> [n, g, w]    g * w == k: g blocks of w columns per row
//! mul      (reshape, s)    either operand order; s is a constant scale
//! reshape  -> [n, k]
//! ```
//!
//! Lowered literally, CoreML evaluates the chain when it compiles or loads the model and
//! keeps a float copy of every weight. This prepass, which runs after
//! [`super::coreml_constant_fold`] on the same copy of the [`GraphInfo`], replaces each
//! such chain by one `dequantizeLinear(q, scale [n', g'], zeroPoint [n', g'])` over
//! constants and tells the lowering, through [`BlockwiseConstants`], to emit it as MIL
//! `constexpr_blockwise_shift_scale` (iOS 18 operation set). The stored bytes stay int8 or
//! uint8 in the weights blob and CoreML decompresses them itself. An authored
//! `dequantizeLinear` is not affected: only the operations synthesized here are marked.
//!
//! The converter skips this prepass for the exact float16 lowering of a graph that has
//! float16 operands (see `convert_with_carried_state`): through that lowering's precision
//! pipeline the compressed constant returned `+0` where the literal chain returns `-0`.
//!
//! # What is recognized
//!
//! Operand names and the surrounding architecture do not participate. A chain is rewritten
//! only when all of the following hold; otherwise the graph is left as it is:
//!
//! - `q` is a constant with payload, int8 or uint8, static shape `[n, k]`;
//! - `cast` targets float16 or float32, and every later value of the chain has that type;
//! - `z`, when present, is a constant of the chain's float type with exactly one element,
//!   finite, integral and inside the range of `q`'s type (so that it can be stored as the
//!   offset in that type);
//! - the scale is a constant, or a constant read through at most one `cast` and one
//!   `reshape` in either order, float16 or float32, every value finite, and its shape
//!   broadcasts to `[n, g, w]` with a last dimension of 1: `[n, g, 1]` in the usual form,
//!   and also `[n, 1, 1]`/`[1, g, 1]`/`[g, 1]`/`[1]`/`[]` (one scale per row, per block
//!   column or per tensor);
//! - the casted, centered, blocked and scaled intermediates each have exactly one consumer
//!   and are not graph outputs; the chain's result is not a graph output; no operand has a
//!   pending layout permutation; the operations are recorded in dependency order.
//!
//! A block always covers `w = k / g` whole columns: the operation requires every scale
//! dimension to divide the data dimension, and a `reshape` cannot express a partial block
//! either, so a graph whose shapes disagree is rejected.
//!
//! # Equality with the literal lowering
//!
//! The literal chain computes `fl((q - z) * s)` per element. `cast` is exact: an int8 or
//! uint8 value is an integer of magnitude at most 255, which float16 and float32 both
//! represent. `q - z` is exact for the same reason, because `z` is an integer in the range
//! of the stored type and the difference has magnitude at most 255. The reshapes move no
//! value. `mul` rounds once, to the chain's float type. `constexpr_blockwise_shift_scale`
//! is `scale * (data - offset)` with the result in the scale's type, and the block of an
//! element along axis `i` is `floor(index / (data.shape[i] / scale.shape[i]))`, which for a
//! scale `[n', g']` is the row and the block column the reshape assigned. The difference is
//! the same exact integer and the product is the same real number rounded once. An
//! implementation that forms the product in a wider float first also rounds once: an 8-bit
//! integer times a float16 (11-bit significand) is exact in float32, and times a float32
//! (24-bit significand) is exact in float64. The offset is omitted when `z` is zero, where
//! `q - 0` is `q`. `tests/test_coreml_blockwise_constants.rs` evaluates both lowerings on
//! the CoreML runtime and compares the bits.
//!
//! # Readers of the constant
//!
//! `matmul(x, transpose(W))` is the common reader, and a table may be read both by a
//! `gather` and, transposed, by an output projection. Both read the one constant: the
//! lowering emits MIL `matmul` with `transpose_y = true` over the constant, and omits the
//! `transpose` when every reader of its output is such a `matmul`, so CoreML never produces
//! a transposed float copy of the weight.
//!
//! The constant is bit-identical to the literal chain; what a reader computes from it is
//! not guaranteed to be. CoreML evaluates a `matmul` (or `gemm`) over a constant weight with
//! another kernel than one over a weight computed at run time, and float16 results can
//! differ in rounding from the literal lowering, on the in-memory route and through the URL
//! compiler. A reader of the compressed constant returns what the same reader returns over a
//! plain float constant of those values on that route.

use std::collections::{HashMap, HashSet};

use super::coreml_constant_fold::cast_bytes;
use crate::graph::{
    ConstantData, DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use crate::operator_options::MLDimension;
use crate::operators::Operation;

/// What the CoreML lowering has to know about a graph rewritten by [`apply`]. Lookups only:
/// nothing here is iterated into the emitted model.
#[derive(Default)]
pub(crate) struct BlockwiseConstants {
    /// Output operand of every synthesized `dequantizeLinear`.
    constants: HashSet<u32>,
    /// Output operand of a rank-2 `transpose` of such a constant, with the constant and
    /// whether the `transpose` itself is omitted.
    transposes: HashMap<u32, (u32, bool)>,
}

impl BlockwiseConstants {
    /// Whether `operand` is produced by a `dequantizeLinear` that lowers to
    /// `constexpr_blockwise_shift_scale`.
    pub(crate) fn is_constant(&self, operand: u32) -> bool {
        self.constants.contains(&operand)
    }

    /// The blockwise constant that `operand` is the rank-2 transpose of. A `matmul` whose
    /// second operand it is reads the constant with `transpose_y`.
    pub(crate) fn transposed_source(&self, operand: u32) -> Option<u32> {
        self.transposes.get(&operand).map(|&(source, _)| source)
    }

    /// Whether the `transpose` producing `operand` is not emitted, because every reader is
    /// a `matmul` that reads the constant with `transpose_y`.
    pub(crate) fn omits_transpose(&self, operand: u32) -> bool {
        self.transposes
            .get(&operand)
            .is_some_and(|&(_, omitted)| omitted)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.constants.len()
    }
}

/// One recognized chain, as operation and operand indices of the graph it was found in.
pub(crate) struct Chain {
    /// The `reshape` back to `[n, k]`; replaced by the `dequantizeLinear`.
    anchor: usize,
    /// `cast`, `sub` (when present), the block `reshape` and `mul`.
    dropped: Vec<usize>,
    /// Operations between the scale constant and `mul`; dropped when nothing else reads them.
    scale_operations: Vec<usize>,
    /// Constants that lose their payload when nothing else reads them: the zero point and
    /// the source of the scale.
    released: Vec<u32>,
    data: u32,
    output: u32,
    float_type: DataType,
    /// Scale values in `float_type`, row-major `[scale_shape[0], scale_shape[1]]`.
    scale: Vec<u8>,
    scale_shape: [u32; 2],
    /// Zero point as a byte of the data's storage type; `None` when it is zero.
    zero: Option<u8>,
}

/// Producers and consumer counts of the operands of one graph.
struct Uses {
    producer: Vec<Option<usize>>,
    count: Vec<usize>,
}

impl Uses {
    /// `None` when an operand index is out of range or an operand has two producers: such
    /// a graph is left for the lowering to reject.
    fn new(graph: &GraphInfo) -> Option<Self> {
        let mut producer = vec![None; graph.operands.len()];
        let mut count = vec![0usize; graph.operands.len()];
        for (index, operation) in graph.operations.iter().enumerate() {
            for id in operation
                .input_operands()
                .into_iter()
                .chain(operation.option_operands())
            {
                *count.get_mut(id as usize)? += 1;
            }
            for &id in operation.output_operands() {
                if producer.get_mut(id as usize)?.replace(index).is_some() {
                    return None;
                }
            }
        }
        for &id in &graph.output_operands {
            *count.get_mut(id as usize)? += 1;
        }
        Some(Self { producer, count })
    }

    /// The operation producing `operand`, when `operand` has exactly one consumer. Graph
    /// outputs count as consumers, so an operand read by the chain alone is not an output.
    fn sole_reader_producer(&self, operand: u32) -> Option<usize> {
        (*self.count.get(operand as usize)? == 1)
            .then(|| self.producer[operand as usize])
            .flatten()
    }
}

fn static_dimensions(shape: &[MLDimension]) -> Option<Vec<u32>> {
    shape
        .iter()
        .map(|dimension| match dimension {
            MLDimension::Static(size) => Some(*size),
            MLDimension::Dynamic(_) => None,
        })
        .collect()
}

/// The static shape of `operand` when it has `data_type` and no pending permutation.
fn typed_shape(graph: &GraphInfo, operand: u32, data_type: DataType) -> Option<Vec<u32>> {
    let descriptor = &graph.operand(operand)?.descriptor;
    (descriptor.data_type == data_type && descriptor.pending_permutation.is_empty())
        .then(|| descriptor.static_shape())
        .flatten()
}

fn is_float(data_type: DataType) -> bool {
    matches!(data_type, DataType::Float16 | DataType::Float32)
}

fn read_floats(data: &[u8], data_type: DataType) -> Option<Vec<f64>> {
    match data_type {
        DataType::Float16 if data.len().is_multiple_of(2) => Some(
            data.chunks_exact(2)
                .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f64())
                .collect(),
        ),
        DataType::Float32 if data.len().is_multiple_of(4) => Some(
            data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64)
                .collect(),
        ),
        _ => None,
    }
}

/// The chains of `graph`, in operation order. Analysis only; [`apply`] rewrites.
pub(crate) fn find_chains(graph: &GraphInfo) -> Vec<Chain> {
    let quantized_constant = |operation: &Operation| {
        matches!(operation, Operation::Cast { input, .. }
        if graph.constant_operand_ids_to_handles.contains_key(input)
            && graph.operand(*input).is_some_and(|operand| {
                matches!(operand.descriptor.data_type, DataType::Int8 | DataType::Uint8)
            }))
    };
    if !graph.operations.iter().any(quantized_constant) {
        return Vec::new();
    }
    let Some(uses) = Uses::new(graph) else {
        return Vec::new();
    };
    (0..graph.operations.len())
        .filter_map(|anchor| match_chain(graph, &uses, anchor))
        .collect()
}

fn match_chain(graph: &GraphInfo, uses: &Uses, anchor: usize) -> Option<Chain> {
    let Operation::Reshape {
        input: scaled,
        new_shape,
        outputs,
        ..
    } = &graph.operations[anchor]
    else {
        return None;
    };
    let [output] = outputs.as_slice() else {
        return None;
    };
    let result = graph.operand(*output)?;
    let float_type = result.descriptor.data_type;
    if result.kind != OperandKind::Intermediate
        || !is_float(float_type)
        || graph.output_operands.contains(output)
    {
        return None;
    }
    let matrix = typed_shape(graph, *output, float_type)?;
    let [rows, columns] = *matrix.as_slice() else {
        return None;
    };
    if rows == 0 || columns == 0 || static_dimensions(new_shape)? != matrix {
        return None;
    }

    let mul = uses.sole_reader_producer(*scaled)?;
    let Operation::Mul { a, b, outputs, .. } = &graph.operations[mul] else {
        return None;
    };
    if mul >= anchor || outputs.as_slice() != [*scaled] {
        return None;
    }
    let blocked = typed_shape(graph, *scaled, float_type)?;
    let [block_rows, groups, width] = *blocked.as_slice() else {
        return None;
    };
    // Whole blocks only: `groups` scales cover `groups * width` columns.
    if block_rows != rows || groups.checked_mul(width)? != columns {
        return None;
    }

    [(*a, *b), (*b, *a)]
        .into_iter()
        .find_map(|(blocks, scale)| {
            let reshape = uses.sole_reader_producer(blocks)?;
            let Operation::Reshape {
                input: centered,
                new_shape,
                outputs,
                ..
            } = &graph.operations[reshape]
            else {
                return None;
            };
            if reshape >= mul
                || outputs.as_slice() != [blocks]
                || static_dimensions(new_shape)? != blocked
                || typed_shape(graph, blocks, float_type)? != blocked
                || typed_shape(graph, *centered, float_type)? != matrix
            {
                return None;
            }

            let last = uses.sole_reader_producer(*centered)?;
            if last >= reshape {
                return None;
            }
            let (cast, widened, sub) = match &graph.operations[last] {
                Operation::Sub { a, b, outputs, .. } if outputs.as_slice() == [*centered] => {
                    let cast = uses.sole_reader_producer(*a)?;
                    if cast >= last || typed_shape(graph, *a, float_type)? != matrix {
                        return None;
                    }
                    (cast, *a, Some((last, *b)))
                }
                Operation::Cast { .. } => (last, *centered, None),
                _ => return None,
            };
            let Operation::Cast {
                input: data,
                data_type,
                outputs,
                ..
            } = &graph.operations[cast]
            else {
                return None;
            };
            if *data_type != float_type || outputs.as_slice() != [widened] {
                return None;
            }

            let stored = graph.operand(*data)?;
            let stored_type = stored.descriptor.data_type;
            if stored.kind != OperandKind::Constant
                || !matches!(stored_type, DataType::Int8 | DataType::Uint8)
                || typed_shape(graph, *data, stored_type)? != matrix
                || graph.constant_operand_ids_to_handles.get(data)?.data.len()
                    != (rows as usize).checked_mul(columns as usize)?
            {
                return None;
            }

            let zero = match sub {
                Some((_, zero)) => zero_point(graph, zero, float_type, stored_type)?,
                None => None,
            };
            let scale = scale_values(graph, uses, scale, float_type, mul, [rows, groups])?;

            let mut dropped = vec![cast, reshape, mul];
            let mut released = vec![scale.source];
            if let Some((index, zero)) = sub {
                dropped.push(index);
                released.push(zero);
            }
            Some(Chain {
                anchor,
                dropped,
                scale_operations: scale.operations,
                released,
                data: *data,
                output: *output,
                float_type,
                scale: scale.bytes,
                scale_shape: scale.shape,
                zero,
            })
        })
}

/// The offset byte for the zero point `zero`: `Some(None)` when it is zero, `None` when
/// the operand is not a one-element constant holding an integer of the stored type.
fn zero_point(
    graph: &GraphInfo,
    zero: u32,
    float_type: DataType,
    stored_type: DataType,
) -> Option<Option<u8>> {
    if graph.operand(zero)?.kind != OperandKind::Constant {
        return None;
    }
    // One element in at most the chain's rank: broadcasting it leaves `[n, k]` unchanged.
    let shape = typed_shape(graph, zero, float_type)?;
    if shape.len() > 2 || shape.iter().any(|&size| size != 1) {
        return None;
    }
    let data = &graph.constant_operand_ids_to_handles.get(&zero)?.data;
    let [value] = *read_floats(data, float_type)?.as_slice() else {
        return None;
    };
    if !value.is_finite() || value.fract() != 0.0 {
        return None;
    }
    let byte = match stored_type {
        DataType::Uint8 if (0.0..=255.0).contains(&value) => value as u8,
        DataType::Int8 if (-128.0..=127.0).contains(&value) => value as i8 as u8,
        _ => return None,
    };
    Some((value != 0.0).then_some(byte))
}

struct Scale {
    bytes: Vec<u8>,
    shape: [u32; 2],
    operations: Vec<usize>,
    source: u32,
}

/// The constant scale behind `operand`, converted to `float_type`, with its shape reduced
/// to `[n', g']` for a data tensor blocked as `[rows, groups, width]`.
fn scale_values(
    graph: &GraphInfo,
    uses: &Uses,
    operand: u32,
    float_type: DataType,
    before: usize,
    [rows, groups]: [u32; 2],
) -> Option<Scale> {
    let shape = typed_shape(graph, operand, float_type)?;
    if shape.len() > 3 {
        return None;
    }
    // Right-aligned against `[rows, groups, width]`. The scale may not vary inside a block.
    let mut aligned = [1u32; 3];
    aligned[3 - shape.len()..].copy_from_slice(&shape);
    if aligned[2] != 1
        || (aligned[0] != 1 && aligned[0] != rows)
        || (aligned[1] != 1 && aligned[1] != groups)
    {
        return None;
    }
    let count = (aligned[0] as usize).checked_mul(aligned[1] as usize)?;

    let mut current = operand;
    let mut operations = Vec::new();
    let (mut cast_seen, mut reshape_seen) = (false, false);
    let mut limit = before;
    let source = loop {
        let value = graph.operand(current)?;
        if !value.descriptor.pending_permutation.is_empty()
            || !is_float(value.descriptor.data_type)
            || value
                .descriptor
                .static_shape()?
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d as usize))?
                != count
        {
            return None;
        }
        if value.kind == OperandKind::Constant {
            break current;
        }
        let index = (*uses.producer.get(current as usize)?)?;
        if index >= limit {
            return None;
        }
        limit = index;
        current = match &graph.operations[index] {
            Operation::Cast {
                input,
                data_type,
                outputs,
                ..
            } if !cast_seen
                && outputs.as_slice() == [current]
                && *data_type == value.descriptor.data_type =>
            {
                cast_seen = true;
                *input
            }
            Operation::Reshape {
                input,
                new_shape,
                outputs,
                ..
            } if !reshape_seen
                && outputs.as_slice() == [current]
                && static_dimensions(new_shape)? == value.descriptor.static_shape()?
                && graph.operand(*input)?.descriptor.data_type == value.descriptor.data_type =>
            {
                reshape_seen = true;
                *input
            }
            _ => return None,
        };
        operations.push(index);
    };
    let source_type = graph.operand(source)?.descriptor.data_type;
    let data = &graph.constant_operand_ids_to_handles.get(&source)?.data;
    if data.len() != count.checked_mul(source_type.bytes_per_element())? {
        return None;
    }
    // The cast table of the constant fold: float16 to float32 is exact, float32 to float16
    // rounds to nearest, as the folded `cast` of the same constant does.
    let bytes = cast_bytes(data, source_type, float_type)?;
    if read_floats(&bytes, float_type)?
        .iter()
        .any(|value| !value.is_finite())
    {
        return None;
    }
    Some(Scale {
        bytes,
        shape: [aligned[0], aligned[1]],
        operations,
        source,
    })
}

/// Rewrites `chains` (found in this same `graph`) in place and returns the marks for the
/// lowering. New scale and offset constants are appended to the operand table in chain
/// order, so their place in the weights blob is the same on every conversion.
///
/// A `matmul` whose second operand is the rank-2 `transpose` of a rewritten constant reads
/// the constant itself with `transpose_y` (see the module documentation).
pub(crate) fn apply(graph: &mut GraphInfo, chains: Vec<Chain>) -> BlockwiseConstants {
    let mut plan = BlockwiseConstants::default();
    if chains.is_empty() || graph.operands.len() + 2 * chains.len() >= u32::MAX as usize {
        return plan;
    }
    let mut dropped = vec![false; graph.operations.len()];
    let mut scale_operations = Vec::new();
    let mut released = Vec::new();
    for chain in chains {
        let name = super::operand_name(graph, chain.output);
        let stored_type = graph.operands[chain.data as usize].descriptor.data_type;
        let count = chain.scale_shape[0] as usize * chain.scale_shape[1] as usize;
        let mut constant = |suffix: &str, data_type, data: Vec<u8>| {
            let id = graph.operands.len() as u32;
            graph.operands.push(Operand {
                kind: OperandKind::Constant,
                descriptor: OperandDescriptor {
                    data_type,
                    shape: chain
                        .scale_shape
                        .iter()
                        .map(|&size| Dimension::Static(size))
                        .collect(),
                    pending_permutation: Vec::new(),
                },
                name: Some(format!("{name}_{suffix}")),
            });
            graph
                .constant_operand_ids_to_handles
                .insert(id, ConstantData { data, label: None });
            id
        };
        let scale = constant("scale", chain.float_type, chain.scale);
        let zero_point = chain
            .zero
            .map(|byte| constant("offset", stored_type, vec![byte; count]));
        graph.operations[chain.anchor] = Operation::DequantizeLinear {
            input: chain.data,
            scale,
            zero_point,
            options: None,
            outputs: vec![chain.output],
        };
        for index in chain.dropped {
            dropped[index] = true;
        }
        plan.constants.insert(chain.output);
        scale_operations.extend(chain.scale_operations);
        released.extend(chain.released);
    }

    // Consumers left once the chains are gone. `find_chains` checked every index.
    let mut count = vec![0usize; graph.operands.len()];
    for (operation, _) in graph
        .operations
        .iter()
        .zip(&dropped)
        .filter(|&(_, &gone)| !gone)
    {
        for id in operation
            .input_operands()
            .into_iter()
            .chain(operation.option_operands())
        {
            count[id as usize] += 1;
        }
    }
    for &id in &graph.output_operands {
        count[id as usize] += 1;
    }
    // Scale-side casts and reshapes that only a dropped `mul` read, consumers first.
    scale_operations.sort_unstable_by(|a, b| b.cmp(a));
    scale_operations.dedup();
    for index in scale_operations {
        let operation = &graph.operations[index];
        if let (Some(output), [input]) = (
            operation.output_operand(),
            operation.input_operands().as_slice(),
        ) && count[output as usize] == 0
        {
            dropped[index] = true;
            count[*input as usize] -= 1;
        }
    }
    // A constant nothing reads any more is not emitted, as in the constant fold.
    for id in released {
        if count[id as usize] == 0 {
            graph.constant_operand_ids_to_handles.remove(&id);
        }
    }
    let mut index = 0;
    graph.operations.retain(|_| {
        index += 1;
        !dropped[index - 1]
    });
    plan.transposes = transposed_readers(graph, &plan.constants);
    plan
}

/// Rank-2 transposes of the blockwise constants, and whether each can be omitted because
/// every reader is the second operand of a `matmul`.
fn transposed_readers(graph: &GraphInfo, constants: &HashSet<u32>) -> HashMap<u32, (u32, bool)> {
    let mut transposes: HashMap<u32, (u32, bool)> = HashMap::new();
    for operation in &graph.operations {
        let Operation::Transpose {
            input,
            options,
            outputs,
        } = operation
        else {
            continue;
        };
        let permutation = options.as_ref().map(|o| o.permutation.as_slice());
        let [output] = outputs.as_slice() else {
            continue;
        };
        if !constants.contains(input) || !matches!(permutation, None | Some([]) | Some([1, 0])) {
            continue;
        }
        let (Some(source), Some(result)) = (graph.operand(*input), graph.operand(*output)) else {
            continue;
        };
        let data_type = source.descriptor.data_type;
        if let (Some([rows, columns]), Some([transposed_rows, transposed_columns])) = (
            typed_shape(graph, *input, data_type).as_deref(),
            typed_shape(graph, *output, data_type).as_deref(),
        ) && (rows, columns) == (transposed_columns, transposed_rows)
            && result.kind == OperandKind::Intermediate
        {
            transposes.insert(*output, (*input, true));
        }
    }
    if transposes.is_empty() {
        return transposes;
    }
    for operation in &graph.operations {
        for id in operation
            .input_operands()
            .into_iter()
            .chain(operation.option_operands())
        {
            let through_matmul =
                matches!(operation, Operation::Matmul { a, b, .. } if *b == id && *a != id);
            if !through_matmul && let Some((_, omitted)) = transposes.get_mut(&id) {
                *omitted = false;
            }
        }
    }
    for id in &graph.output_operands {
        if let Some((_, omitted)) = transposes.get_mut(id) {
            *omitted = false;
        }
    }
    transposes
}

/// [`find_chains`] and [`apply`] on a copy of `graph`; `None` when there is no chain.
#[cfg(test)]
fn rewrite(graph: &GraphInfo) -> Option<(GraphInfo, BlockwiseConstants)> {
    let chains = find_chains(graph);
    if chains.is_empty() {
        return None;
    }
    let mut rewritten = graph.clone();
    let plan = apply(&mut rewritten, chains);
    Some((rewritten, plan))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlcontext::{MLNamedOperands, MLOperand, MLOperandDescriptor};
    use crate::mlgraphbuilder::MLGraphBuilder;
    use crate::protos::coreml::mil_spec::{self, argument::binding::Binding};
    use crate::protos::coreml::specification;
    use prost::Message;

    const BLOCKWISE: &str = "constexpr_blockwise_shift_scale";

    type Builder = MLGraphBuilder<'static, 'static>;

    fn dimensions(shape: &[u32]) -> Vec<MLDimension> {
        shape
            .iter()
            .map(|&size| MLDimension::Static(size))
            .collect()
    }

    fn float_bytes(values: &[f32], data_type: DataType) -> Vec<u8> {
        match data_type {
            DataType::Float16 => values
                .iter()
                .flat_map(|&value| half::f16::from_f32(value).to_le_bytes())
                .collect(),
            _ => values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }
    }

    fn constant(
        builder: &mut Builder,
        data_type: DataType,
        shape: &[u32],
        data: Vec<u8>,
    ) -> MLOperand {
        builder
            .constant_from_bytes(
                &MLOperandDescriptor::new(data_type, shape.iter().map(|&d| d as u64).collect()),
                data,
            )
            .unwrap()
    }

    fn float_constant(
        builder: &mut Builder,
        data_type: DataType,
        shape: &[u32],
        values: &[f32],
    ) -> MLOperand {
        constant(builder, data_type, shape, float_bytes(values, data_type))
    }

    fn stored_bytes(rows: u32, columns: u32) -> Vec<u8> {
        (0..rows * columns).map(|i| (i * 7 % 251) as u8).collect()
    }

    fn scale_values(count: u32) -> Vec<f32> {
        (0..count).map(|i| 0.5 + i as f32 / 256.0).collect()
    }

    /// The operands of one chain built with `chain`.
    struct Built {
        data: MLOperand,
        widened: MLOperand,
        centered: MLOperand,
        blocks: MLOperand,
        scale: MLOperand,
        scaled: MLOperand,
        weight: MLOperand,
        zero: Option<MLOperand>,
    }

    /// `reshape(mul(reshape(sub(cast(q), zero), [n, g, w]), scale), [n, k])` for a stored
    /// `[rows, groups * width]` matrix. `scale` is any operand broadcastable to the blocks.
    fn chain_with_scale(
        builder: &mut Builder,
        stored_type: DataType,
        float_type: DataType,
        zero: Option<f32>,
        [rows, groups, width]: [u32; 3],
        scale: MLOperand,
    ) -> Built {
        let columns = groups * width;
        let data = constant(
            builder,
            stored_type,
            &[rows, columns],
            stored_bytes(rows, columns),
        );
        let widened = builder.cast(data, float_type).unwrap();
        let zero = zero.map(|value| float_constant(builder, float_type, &[], &[value]));
        let centered = match zero {
            Some(zero) => builder.sub(widened, zero).unwrap(),
            None => widened,
        };
        let blocks = builder
            .reshape(centered, dimensions(&[rows, groups, width]))
            .unwrap();
        let scaled = builder.mul(blocks, scale).unwrap();
        let weight = builder
            .reshape(scaled, dimensions(&[rows, columns]))
            .unwrap();
        Built {
            data,
            widened,
            centered,
            blocks,
            scale,
            scaled,
            weight,
            zero,
        }
    }

    /// The usual chain: a `[rows, groups, 1]` scale constant in the chain's float type.
    fn chain(
        builder: &mut Builder,
        stored_type: DataType,
        float_type: DataType,
        zero: Option<f32>,
        shape: [u32; 3],
    ) -> Built {
        let scale = float_constant(
            builder,
            float_type,
            &[shape[0], shape[1], 1],
            &scale_values(shape[0] * shape[1]),
        );
        chain_with_scale(builder, stored_type, float_type, zero, shape, scale)
    }

    /// Finishes a graph whose only output is `identity(weight)`.
    fn finish(mut builder: Builder, weight: MLOperand) -> GraphInfo {
        let output = builder.identity(weight).unwrap();
        builder
            .finish_graph_info(&MLNamedOperands::from([("table", output)]))
            .unwrap()
    }

    fn dequantize(graph: &GraphInfo, output: MLOperand) -> (u32, u32, Option<u32>) {
        graph
            .operations
            .iter()
            .find_map(|operation| match operation {
                Operation::DequantizeLinear {
                    input,
                    scale,
                    zero_point,
                    outputs,
                    ..
                } if outputs.as_slice() == [output.rustnn_index()] => {
                    Some((*input, *scale, *zero_point))
                }
                _ => None,
            })
            .expect("the chain became a dequantizeLinear")
    }

    fn operation_types(graph: &GraphInfo) -> Vec<&'static str> {
        graph.operations.iter().map(Operation::op_type).collect()
    }

    fn payload(graph: &GraphInfo, id: u32) -> &[u8] {
        &graph.constant_operand_ids_to_handles[&id].data
    }

    fn shape_of(graph: &GraphInfo, id: u32) -> (DataType, Vec<u32>) {
        let descriptor = &graph.operand(id).unwrap().descriptor;
        (descriptor.data_type, descriptor.static_shape().unwrap())
    }

    #[test]
    fn a_chain_becomes_one_dequantize_linear_over_the_stored_bytes() {
        let (rows, groups, width) = (4u32, 2u32, 32u32);
        let mut builder = MLGraphBuilder::new_uncompiled();
        let built = chain(
            &mut builder,
            DataType::Uint8,
            DataType::Float16,
            Some(128.0),
            [rows, groups, width],
        );
        let graph = finish(builder, built.weight);
        let (rewritten, plan) = rewrite(&graph).expect("the chain is recognized");

        assert_eq!(
            operation_types(&rewritten),
            ["dequantizeLinear", "identity"]
        );
        assert_eq!(plan.len(), 1);
        assert!(plan.is_constant(built.weight.rustnn_index()));
        let (data, scale, zero_point) = dequantize(&rewritten, built.weight);
        assert_eq!(data, built.data.rustnn_index());
        assert_eq!(
            payload(&rewritten, data),
            stored_bytes(rows, groups * width)
        );
        assert_eq!(
            shape_of(&rewritten, scale),
            (DataType::Float16, vec![rows, groups])
        );
        assert_eq!(
            payload(&rewritten, scale),
            float_bytes(&scale_values(rows * groups), DataType::Float16)
        );
        let zero_point = zero_point.expect("a nonzero zero point is an offset");
        assert_eq!(
            shape_of(&rewritten, zero_point),
            (DataType::Uint8, vec![rows, groups])
        );
        assert_eq!(
            payload(&rewritten, zero_point),
            vec![128u8; (rows * groups) as usize]
        );
        // The chain's own scale and zero point constants are not emitted any more.
        for released in [built.scale, built.zero.unwrap()] {
            assert!(
                !rewritten
                    .constant_operand_ids_to_handles
                    .contains_key(&released.rustnn_index())
            );
        }
        // The original graph is untouched.
        assert_eq!(
            operation_types(&graph),
            ["cast", "sub", "reshape", "mul", "reshape", "identity"]
        );
    }

    #[test]
    fn int8_without_a_zero_point_has_no_offset_in_either_operand_order() {
        for swapped in [false, true] {
            let (rows, groups, width) = (3u32, 9u32, 64u32);
            let mut builder = MLGraphBuilder::new_uncompiled();
            let columns = groups * width;
            let data = constant(
                &mut builder,
                DataType::Int8,
                &[rows, columns],
                stored_bytes(rows, columns),
            );
            let widened = builder.cast(data, DataType::Float32).unwrap();
            let blocks = builder
                .reshape(widened, dimensions(&[rows, groups, width]))
                .unwrap();
            let scale = float_constant(
                &mut builder,
                DataType::Float32,
                &[rows, groups, 1],
                &scale_values(rows * groups),
            );
            let scaled = if swapped {
                builder.mul(scale, blocks).unwrap()
            } else {
                builder.mul(blocks, scale).unwrap()
            };
            let weight = builder
                .reshape(scaled, dimensions(&[rows, columns]))
                .unwrap();
            let graph = finish(builder, weight);
            let (rewritten, _) = rewrite(&graph).expect("the chain is recognized");
            let (input, scale, zero_point) = dequantize(&rewritten, weight);
            assert_eq!(input, data.rustnn_index());
            assert_eq!(zero_point, None, "swapped={swapped}");
            assert_eq!(
                shape_of(&rewritten, scale),
                (DataType::Float32, vec![rows, groups])
            );
        }
    }

    #[test]
    fn a_zero_point_of_zero_needs_no_offset() {
        for zero in [0.0f32, -0.0] {
            let mut builder = MLGraphBuilder::new_uncompiled();
            let built = chain(
                &mut builder,
                DataType::Int8,
                DataType::Float16,
                Some(zero),
                [2, 2, 4],
            );
            let graph = finish(builder, built.weight);
            let (rewritten, _) = rewrite(&graph).expect("the chain is recognized");
            assert_eq!(dequantize(&rewritten, built.weight).2, None);
            assert_eq!(
                operation_types(&rewritten),
                ["dequantizeLinear", "identity"]
            );
        }
    }

    #[test]
    fn signed_zero_points_are_stored_in_the_data_type() {
        let mut builder = MLGraphBuilder::new_uncompiled();
        let built = chain(
            &mut builder,
            DataType::Int8,
            DataType::Float32,
            Some(-5.0),
            [2, 2, 4],
        );
        let graph = finish(builder, built.weight);
        let (rewritten, _) = rewrite(&graph).unwrap();
        let zero_point = dequantize(&rewritten, built.weight).2.unwrap();
        assert_eq!(
            shape_of(&rewritten, zero_point),
            (DataType::Int8, vec![2, 2])
        );
        assert_eq!(payload(&rewritten, zero_point), [(-5i8) as u8; 4]);
    }

    #[test]
    fn the_scale_is_read_through_a_cast_and_a_reshape() {
        let (rows, groups, width) = (4u32, 2u32, 8u32);
        let values = [0.1f32, -0.2, 0.3, 1.0e-3, 7.0, 0.333_333_34, 2.5, -1.0];
        // float32 scales reshaped, then narrowed (the form a float16 embedding table uses);
        // float16 scales widened, then reshaped; and the two orders exchanged.
        for (source, target, cast_first) in [
            (DataType::Float32, DataType::Float16, false),
            (DataType::Float32, DataType::Float16, true),
            (DataType::Float16, DataType::Float32, false),
            (DataType::Float16, DataType::Float32, true),
        ] {
            let mut builder = MLGraphBuilder::new_uncompiled();
            let stored = float_constant(&mut builder, source, &[rows, groups], &values);
            let scale = if cast_first {
                let cast = builder.cast(stored, target).unwrap();
                builder
                    .reshape(cast, dimensions(&[rows, groups, 1]))
                    .unwrap()
            } else {
                let reshaped = builder
                    .reshape(stored, dimensions(&[rows, groups, 1]))
                    .unwrap();
                builder.cast(reshaped, target).unwrap()
            };
            let built = chain_with_scale(
                &mut builder,
                DataType::Uint8,
                target,
                Some(128.0),
                [rows, groups, width],
                scale,
            );
            let graph = finish(builder, built.weight);
            let (rewritten, _) = rewrite(&graph).expect("the chain is recognized");
            assert_eq!(
                operation_types(&rewritten),
                ["dequantizeLinear", "identity"]
            );
            let scale = dequantize(&rewritten, built.weight).1;
            assert_eq!(shape_of(&rewritten, scale), (target, vec![rows, groups]));
            // The source values rounded once to the chain's type.
            let expected: Vec<u8> = match (source, target) {
                (DataType::Float32, _) => float_bytes(&values, DataType::Float16),
                _ => values
                    .iter()
                    .flat_map(|&v| half::f16::from_f32(v).to_f32().to_le_bytes())
                    .collect(),
            };
            assert_eq!(
                payload(&rewritten, scale),
                expected,
                "{source:?} -> {target:?}"
            );
            assert!(
                !rewritten
                    .constant_operand_ids_to_handles
                    .contains_key(&stored.rustnn_index())
            );
        }
    }

    #[test]
    fn the_chain_is_also_found_after_the_constant_fold() {
        let mut builder = MLGraphBuilder::new_uncompiled();
        let stored = float_constant(&mut builder, DataType::Float32, &[4, 2], &scale_values(8));
        let reshaped = builder.reshape(stored, dimensions(&[4, 2, 1])).unwrap();
        let scale = builder.cast(reshaped, DataType::Float16).unwrap();
        let built = chain_with_scale(
            &mut builder,
            DataType::Int8,
            DataType::Float16,
            None,
            [4, 2, 8],
            scale,
        );
        let graph = finish(builder, built.weight);
        let folded = super::super::coreml_constant_fold::fold_constant_views(&graph)
            .expect("the scale folds");
        let (after_fold, _) = rewrite(&folded).expect("recognized after the fold");
        let (direct, _) = rewrite(&graph).expect("recognized before the fold");
        for rewritten in [&after_fold, &direct] {
            let scale = dequantize(rewritten, built.weight).1;
            assert_eq!(
                payload(rewritten, scale),
                float_bytes(&scale_values(8), DataType::Float16)
            );
        }
    }

    #[test]
    fn per_row_per_block_column_and_per_tensor_scales_reduce_to_two_dimensions() {
        let (rows, groups, width) = (4u32, 2u32, 8u32);
        for (shape, expected) in [
            (vec![rows, 1, 1], [rows, 1]),
            (vec![1, groups, 1], [1, groups]),
            (vec![groups, 1], [1, groups]),
            (vec![1, 1, 1], [1, 1]),
            (vec![1], [1, 1]),
            (vec![], [1, 1]),
        ] {
            let count: u32 = shape.iter().product();
            let mut builder = MLGraphBuilder::new_uncompiled();
            let scale = float_constant(
                &mut builder,
                DataType::Float16,
                &shape,
                &scale_values(count),
            );
            let built = chain_with_scale(
                &mut builder,
                DataType::Uint8,
                DataType::Float16,
                Some(128.0),
                [rows, groups, width],
                scale,
            );
            let graph = finish(builder, built.weight);
            let (rewritten, _) = rewrite(&graph).unwrap_or_else(|| panic!("{shape:?}"));
            let (_, scale, zero_point) = dequantize(&rewritten, built.weight);
            assert_eq!(
                shape_of(&rewritten, scale),
                (DataType::Float16, expected.to_vec()),
                "{shape:?}"
            );
            assert_eq!(
                shape_of(&rewritten, zero_point.unwrap()).1,
                expected.to_vec()
            );
        }
        // One block per row: `[n, 1, k]` blocks with a `[n, 1, 1]` scale.
        let mut builder = MLGraphBuilder::new_uncompiled();
        let built = chain(
            &mut builder,
            DataType::Int8,
            DataType::Float32,
            None,
            [rows, 1, 16],
        );
        let graph = finish(builder, built.weight);
        let (rewritten, _) = rewrite(&graph).unwrap();
        let scale = dequantize(&rewritten, built.weight).1;
        assert_eq!(shape_of(&rewritten, scale).1, [rows, 1]);
    }

    /// Builds the usual uint8/float16 chain, lets `extra` add readers, and expects the
    /// graph to be left alone.
    fn assert_rejected(label: &str, extra: impl FnOnce(&mut Builder, &Built) -> Vec<MLOperand>) {
        let mut builder = MLGraphBuilder::new_uncompiled();
        let built = chain(
            &mut builder,
            DataType::Uint8,
            DataType::Float16,
            Some(128.0),
            [4, 2, 8],
        );
        let extras = extra(&mut builder, &built);
        let output = builder.identity(built.weight).unwrap();
        let names: Vec<String> = (0..extras.len()).map(|i| format!("extra{i}")).collect();
        let mut outputs = MLNamedOperands::from([("table", output)]);
        for (name, operand) in names.iter().zip(extras) {
            outputs.insert(name, operand);
        }
        let graph = builder.finish_graph_info(&outputs).unwrap();
        assert!(find_chains(&graph).is_empty(), "{label}");
        assert!(rewrite(&graph).is_none(), "{label}");
    }

    #[test]
    fn an_intermediate_with_another_reader_or_a_public_name_keeps_the_literal_chain() {
        assert_rejected("cast read twice", |b, c| {
            vec![b.identity(c.widened).unwrap()]
        });
        assert_rejected("difference read twice", |b, c| {
            vec![b.identity(c.centered).unwrap()]
        });
        assert_rejected("blocks read twice", |b, c| {
            vec![b.identity(c.blocks).unwrap()]
        });
        assert_rejected("product read twice", |b, c| {
            vec![b.identity(c.scaled).unwrap()]
        });
        assert_rejected("cast is a graph output", |_, c| vec![c.widened]);
        assert_rejected("difference is a graph output", |_, c| vec![c.centered]);
        assert_rejected("blocks are a graph output", |_, c| vec![c.blocks]);
        assert_rejected("product is a graph output", |_, c| vec![c.scaled]);
        assert_rejected("result is a graph output", |_, c| vec![c.weight]);
    }

    #[test]
    fn zero_points_outside_the_stored_type_keep_the_literal_chain() {
        for (stored_type, zero) in [
            (DataType::Uint8, 127.5f32),
            (DataType::Uint8, -1.0),
            (DataType::Uint8, 256.0),
            (DataType::Int8, 128.0),
            (DataType::Int8, -129.0),
            (DataType::Uint8, f32::INFINITY),
            (DataType::Uint8, f32::NAN),
        ] {
            let mut builder = MLGraphBuilder::new_uncompiled();
            let built = chain(
                &mut builder,
                stored_type,
                DataType::Float32,
                Some(zero),
                [2, 2, 4],
            );
            let graph = finish(builder, built.weight);
            assert!(rewrite(&graph).is_none(), "{stored_type:?} {zero}");
        }
        // A zero point per row is not a scalar zero point.
        let mut builder = MLGraphBuilder::new_uncompiled();
        let data = constant(&mut builder, DataType::Uint8, &[2, 8], stored_bytes(2, 8));
        let widened = builder.cast(data, DataType::Float32).unwrap();
        let zero = float_constant(&mut builder, DataType::Float32, &[2, 1], &[128.0, 127.0]);
        let centered = builder.sub(widened, zero).unwrap();
        let blocks = builder.reshape(centered, dimensions(&[2, 2, 4])).unwrap();
        let scale = float_constant(
            &mut builder,
            DataType::Float32,
            &[2, 2, 1],
            &scale_values(4),
        );
        let scaled = builder.mul(blocks, scale).unwrap();
        let weight = builder.reshape(scaled, dimensions(&[2, 8])).unwrap();
        let graph = finish(builder, weight);
        assert!(rewrite(&graph).is_none());
    }

    #[test]
    fn scales_that_are_not_finite_block_constants_keep_the_literal_chain() {
        let (rows, groups, width) = (2u32, 2u32, 4u32);
        let build = |scale: &dyn Fn(&mut Builder) -> MLOperand| {
            let mut builder = MLGraphBuilder::new_uncompiled();
            let scale = scale(&mut builder);
            let built = chain_with_scale(
                &mut builder,
                DataType::Uint8,
                DataType::Float32,
                Some(128.0),
                [rows, groups, width],
                scale,
            );
            finish(builder, built.weight)
        };
        let float32 = |shape: Vec<u64>| MLOperandDescriptor::new(DataType::Float32, shape);
        // One scale per element varies inside a block.
        let per_element = build(&|b| {
            float_constant(
                b,
                DataType::Float32,
                &[rows, groups, width],
                &scale_values(rows * groups * width),
            )
        });
        assert!(
            rewrite(&per_element).is_none(),
            "scale varies inside a block"
        );
        let per_column =
            build(&|b| float_constant(b, DataType::Float32, &[width], &scale_values(width)));
        assert!(
            rewrite(&per_column).is_none(),
            "scale varies inside a block"
        );
        for value in [f32::INFINITY, f32::NAN] {
            let graph = build(&|b| {
                float_constant(
                    b,
                    DataType::Float32,
                    &[rows, groups, 1],
                    &[1.0, value, 1.0, 1.0],
                )
            });
            assert!(rewrite(&graph).is_none(), "{value}");
        }
        let runtime = build(&|b| b.input("scale", &float32(vec![2, 2, 1])).unwrap());
        assert!(rewrite(&runtime).is_none(), "scale is a graph input");
        // A float32 scale that overflows float16 when the chain narrows it.
        let mut builder = MLGraphBuilder::new_uncompiled();
        let stored = float_constant(&mut builder, DataType::Float32, &[2, 2, 1], &[1.0e6; 4]);
        let scale = builder.cast(stored, DataType::Float16).unwrap();
        let built = chain_with_scale(
            &mut builder,
            DataType::Uint8,
            DataType::Float16,
            None,
            [2, 2, 4],
            scale,
        );
        assert!(rewrite(&finish(builder, built.weight)).is_none());
        // Two casts are more than the one the scale may be read through.
        let mut builder = MLGraphBuilder::new_uncompiled();
        let stored = float_constant(
            &mut builder,
            DataType::Float32,
            &[2, 2, 1],
            &scale_values(4),
        );
        let narrowed = builder.cast(stored, DataType::Float16).unwrap();
        let scale = builder.cast(narrowed, DataType::Float32).unwrap();
        let built = chain_with_scale(
            &mut builder,
            DataType::Uint8,
            DataType::Float32,
            None,
            [2, 2, 4],
            scale,
        );
        assert!(rewrite(&finish(builder, built.weight)).is_none());
    }

    #[test]
    fn stored_values_that_are_not_narrow_integer_constants_keep_the_literal_chain() {
        for stored_type in [DataType::Int32, DataType::Float32] {
            let mut builder = MLGraphBuilder::new_uncompiled();
            let data = constant(&mut builder, stored_type, &[2, 8], vec![0; 2 * 8 * 4]);
            let widened = builder.cast(data, DataType::Float16).unwrap();
            let blocks = builder.reshape(widened, dimensions(&[2, 2, 4])).unwrap();
            let scale = float_constant(
                &mut builder,
                DataType::Float16,
                &[2, 2, 1],
                &scale_values(4),
            );
            let scaled = builder.mul(blocks, scale).unwrap();
            let weight = builder.reshape(scaled, dimensions(&[2, 8])).unwrap();
            assert!(
                rewrite(&finish(builder, weight)).is_none(),
                "{stored_type:?}"
            );
        }
        let mut builder = MLGraphBuilder::new_uncompiled();
        let data = builder
            .input("q", &MLOperandDescriptor::new(DataType::Uint8, vec![2, 8]))
            .unwrap();
        let widened = builder.cast(data, DataType::Float16).unwrap();
        let blocks = builder.reshape(widened, dimensions(&[2, 2, 4])).unwrap();
        let scale = float_constant(
            &mut builder,
            DataType::Float16,
            &[2, 2, 1],
            &scale_values(4),
        );
        let scaled = builder.mul(blocks, scale).unwrap();
        let weight = builder.reshape(scaled, dimensions(&[2, 8])).unwrap();
        assert!(rewrite(&finish(builder, weight)).is_none(), "runtime data");
        // Rank 3 stored data is not the `[n, k]` form.
        let mut builder = MLGraphBuilder::new_uncompiled();
        let data = constant(&mut builder, DataType::Uint8, &[2, 2, 4], vec![0; 16]);
        let widened = builder.cast(data, DataType::Float16).unwrap();
        let scale = float_constant(
            &mut builder,
            DataType::Float16,
            &[2, 2, 1],
            &scale_values(4),
        );
        let scaled = builder.mul(widened, scale).unwrap();
        let weight = builder.reshape(scaled, dimensions(&[2, 8])).unwrap();
        assert!(rewrite(&finish(builder, weight)).is_none(), "rank 3 data");
    }

    #[test]
    fn a_partial_last_block_keeps_the_literal_chain() {
        // `constexpr_blockwise_shift_scale` needs every scale dimension to divide the data
        // dimension, and a `reshape` cannot leave a remainder either, so only a graph that
        // was not built by the builder can ask for 3 blocks of 2 columns over 8 columns.
        let mut builder = MLGraphBuilder::new_uncompiled();
        let built = chain(
            &mut builder,
            DataType::Uint8,
            DataType::Float16,
            Some(128.0),
            [2, 4, 2],
        );
        let mut graph = finish(builder, built.weight);
        assert!(rewrite(&graph).is_some());
        let partial = crate::graph::to_dimension_vector(&[2, 3, 2]);
        for operand in [built.blocks, built.scaled] {
            graph.operands[operand.rustnn_index() as usize]
                .descriptor
                .shape = partial.clone();
        }
        graph.operands[built.scale.rustnn_index() as usize]
            .descriptor
            .shape = crate::graph::to_dimension_vector(&[2, 3, 1]);
        graph
            .constant_operand_ids_to_handles
            .get_mut(&built.scale.rustnn_index())
            .unwrap()
            .data
            .truncate(2 * 3 * 2);
        for operation in &mut graph.operations {
            if let Operation::Reshape { new_shape, .. } = operation
                && new_shape.len() == 3
            {
                *new_shape = dimensions(&[2, 3, 2]);
            }
        }
        assert!(find_chains(&graph).is_empty());
    }

    #[test]
    fn malformed_graphs_are_left_for_the_lowering_to_reject() {
        let mut builder = MLGraphBuilder::new_uncompiled();
        let built = chain(
            &mut builder,
            DataType::Uint8,
            DataType::Float16,
            Some(128.0),
            [2, 2, 4],
        );
        let graph = finish(builder, built.weight);

        let mut dangling = graph.clone();
        dangling.operations.push(Operation::Identity {
            input: u32::MAX,
            options: None,
            outputs: vec![0],
        });
        assert!(find_chains(&dangling).is_empty(), "operand out of range");

        let mut produced_twice = graph.clone();
        let again = produced_twice.operations[0].clone();
        produced_twice.operations.push(again);
        assert!(find_chains(&produced_twice).is_empty(), "two producers");

        let mut out_of_order = graph.clone();
        out_of_order.operations.swap(0, 1);
        assert!(find_chains(&out_of_order).is_empty(), "sub before its cast");

        let mut short = graph.clone();
        short
            .constant_operand_ids_to_handles
            .get_mut(&built.data.rustnn_index())
            .unwrap()
            .data
            .pop();
        assert!(find_chains(&short).is_empty(), "short payload");

        let mut permuted = graph.clone();
        permuted.operands[built.weight.rustnn_index() as usize]
            .descriptor
            .pending_permutation = vec![1, 0];
        assert!(find_chains(&permuted).is_empty(), "pending permutation");
    }

    #[test]
    fn shared_constants_keep_their_payload_until_the_last_reader_is_gone() {
        let mut builder = MLGraphBuilder::new_uncompiled();
        let zero = float_constant(&mut builder, DataType::Float16, &[], &[128.0]);
        let scale = float_constant(
            &mut builder,
            DataType::Float16,
            &[2, 2, 1],
            &scale_values(4),
        );
        let mut weights = Vec::new();
        for _ in 0..2 {
            let data = constant(&mut builder, DataType::Uint8, &[2, 8], stored_bytes(2, 8));
            let widened = builder.cast(data, DataType::Float16).unwrap();
            let centered = builder.sub(widened, zero).unwrap();
            let blocks = builder.reshape(centered, dimensions(&[2, 2, 4])).unwrap();
            let scaled = builder.mul(blocks, scale).unwrap();
            weights.push(builder.reshape(scaled, dimensions(&[2, 8])).unwrap());
        }
        let sum = builder.add(weights[0], weights[1]).unwrap();
        // A reader of the zero point that is not part of a chain.
        let shifted = builder.add(sum, zero).unwrap();
        let graph = builder
            .finish_graph_info(&MLNamedOperands::from([("sum", shifted)]))
            .unwrap();
        let (rewritten, plan) = rewrite(&graph).expect("both chains are recognized");
        assert_eq!(plan.len(), 2);
        assert_eq!(
            operation_types(&rewritten),
            ["dequantizeLinear", "dequantizeLinear", "add", "add"]
        );
        assert!(
            rewritten
                .constant_operand_ids_to_handles
                .contains_key(&zero.rustnn_index()),
            "the zero point still has a reader"
        );
        assert!(
            !rewritten
                .constant_operand_ids_to_handles
                .contains_key(&scale.rustnn_index()),
            "both readers of the scale are gone"
        );
        // New constants follow the original operands, in chain order.
        let first = dequantize(&rewritten, weights[0]);
        let second = dequantize(&rewritten, weights[1]);
        let original = graph.operands.len() as u32;
        assert_eq!(
            [first.1, first.2.unwrap(), second.1, second.2.unwrap()],
            [original, original + 1, original + 2, original + 3]
        );
    }

    #[test]
    fn a_scale_side_operation_with_another_reader_stays() {
        let mut builder = MLGraphBuilder::new_uncompiled();
        let stored = float_constant(
            &mut builder,
            DataType::Float16,
            &[2, 2, 1],
            &scale_values(4),
        );
        let scale = builder.cast(stored, DataType::Float32).unwrap();
        let built = chain_with_scale(
            &mut builder,
            DataType::Int8,
            DataType::Float32,
            None,
            [2, 2, 4],
            scale,
        );
        let table = builder.identity(built.weight).unwrap();
        let graph = builder
            .finish_graph_info(&MLNamedOperands::from([("table", table), ("scale", scale)]))
            .unwrap();
        let (rewritten, _) = rewrite(&graph).expect("the chain is recognized");
        assert_eq!(
            operation_types(&rewritten),
            ["cast", "dequantizeLinear", "identity"]
        );
        assert!(
            rewritten
                .constant_operand_ids_to_handles
                .contains_key(&stored.rustnn_index())
        );
    }

    /// The projection reader: `matmul(x, transpose(weight))`.
    fn projection(
        builder: &mut Builder,
        weight: MLOperand,
        columns: u32,
    ) -> (MLOperand, MLOperand) {
        let x = builder
            .input(
                "x",
                &MLOperandDescriptor::new(DataType::Float16, vec![1, 1, columns as u64]),
            )
            .unwrap();
        let transposed = builder.transpose(weight).unwrap();
        (transposed, builder.matmul(x, transposed).unwrap())
    }

    #[test]
    fn a_transpose_read_only_by_matmul_is_omitted() {
        let mut builder = MLGraphBuilder::new_uncompiled();
        let built = chain(
            &mut builder,
            DataType::Uint8,
            DataType::Float16,
            Some(128.0),
            [4, 2, 8],
        );
        let (transposed, y) = projection(&mut builder, built.weight, 16);
        let graph = builder
            .finish_graph_info(&MLNamedOperands::from([("y", y)]))
            .unwrap();
        let (rewritten, plan) = rewrite(&graph).unwrap();
        // The graph keeps the transpose (it stays a valid WebNN graph); the lowering skips it.
        assert_eq!(
            operation_types(&rewritten),
            ["dequantizeLinear", "transpose", "matmul"]
        );
        assert_eq!(
            plan.transposed_source(transposed.rustnn_index()),
            Some(built.weight.rustnn_index())
        );
        assert!(plan.omits_transpose(transposed.rustnn_index()));
        assert_eq!(plan.transposed_source(built.weight.rustnn_index()), None);
    }

    #[test]
    fn a_transpose_with_another_reader_is_kept() {
        let mut builder = MLGraphBuilder::new_uncompiled();
        let built = chain(
            &mut builder,
            DataType::Uint8,
            DataType::Float16,
            Some(128.0),
            [4, 2, 8],
        );
        let (transposed, y) = projection(&mut builder, built.weight, 16);
        // The transposed table as the first matmul operand is a second kind of reader.
        let left = builder
            .input(
                "left",
                &MLOperandDescriptor::new(DataType::Float16, vec![4, 3]),
            )
            .unwrap();
        let z = builder.matmul(transposed, left).unwrap();
        let graph = builder
            .finish_graph_info(&MLNamedOperands::from([("y", y), ("z", z)]))
            .unwrap();
        let (_, plan) = rewrite(&graph).unwrap();
        assert_eq!(
            plan.transposed_source(transposed.rustnn_index()),
            Some(built.weight.rustnn_index())
        );
        assert!(!plan.omits_transpose(transposed.rustnn_index()));
    }

    // ---- The emitted model ----

    struct Lowered {
        operations: Vec<mil_spec::Operation>,
        weights: Vec<u8>,
    }

    /// Adds a carried-state pair to the graph being built and returns its output, which the
    /// caller names `next_state`. [`lower`] then converts the graph as a stateful program.
    fn carried_state(builder: &mut Builder) -> MLOperand {
        let state = builder
            .input(
                "state",
                &MLOperandDescriptor::new(DataType::Float16, vec![1, 4]),
            )
            .unwrap();
        builder.add(state, state).unwrap()
    }

    /// The main block and weights of `graph` under the native lowering; a graph built with
    /// [`carried_state`] carries that pair as a CoreML state.
    fn lower(graph: &GraphInfo) -> Lowered {
        let named = |ids: &[u32], name: &str| {
            ids.iter()
                .copied()
                .find(|&id| graph.operand(id).unwrap().name.as_deref() == Some(name))
        };
        let pairs: Vec<_> = named(&graph.input_operands, "state")
            .zip(named(&graph.output_operands, "next_state"))
            .map(|(input, output)| super::super::coreml_state::CarriedStatePair { input, output })
            .into_iter()
            .collect();
        let converted = super::super::CoremlMlProgramConverter
            .convert_with_carried_state(
                graph,
                crate::mlcontextoptions::CoremlFloat16Precision::Native,
                &pairs,
            )
            .expect("the graph lowers");
        let model = specification::Model::decode(converted.data.as_slice()).unwrap();
        let specification::model::Type::MlProgram(program) = model.r#type.unwrap() else {
            panic!("expected an ML program")
        };
        Lowered {
            operations: program.functions["main"].block_specializations["CoreML8"]
                .operations
                .clone(),
            weights: converted.weights_data.unwrap_or_default(),
        }
    }

    impl Lowered {
        fn of_type(&self, kind: &str) -> Vec<&mil_spec::Operation> {
            self.operations
                .iter()
                .filter(|operation| operation.r#type == kind)
                .collect()
        }

        fn producer(&self, name: &str) -> &mil_spec::Operation {
            self.operations
                .iter()
                .find(|operation| operation.outputs.iter().any(|output| output.name == name))
                .unwrap_or_else(|| panic!("{name} has a producer"))
        }

        /// The stored bytes of the `const` operation named `name`.
        fn blob(&self, name: &str) -> &[u8] {
            let constant = self.producer(name);
            assert_eq!(constant.r#type, "const", "{name}");
            let Some(mil_spec::value::Value::BlobFileValue(blob)) =
                &constant.attributes["val"].value
            else {
                panic!("{name} is stored in the weights blob")
            };
            let at = blob.offset as usize;
            let length = u64::from_le_bytes(self.weights[at + 8..at + 16].try_into().unwrap());
            &self.weights[at + 64..at + 64 + length as usize]
        }
    }

    fn bound_name<'a>(operation: &'a mil_spec::Operation, input: &str) -> &'a str {
        match &operation.inputs[input].arguments[0].binding {
            Some(Binding::Name(name)) => name,
            other => panic!("{input} is bound by name, got {other:?}"),
        }
    }

    fn bound_bool(operation: &mil_spec::Operation, input: &str) -> bool {
        use mil_spec::{tensor_value, value};
        let Some(Binding::Value(value)) = &operation.inputs[input].arguments[0].binding else {
            panic!("{input} is an immediate")
        };
        let Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(tensor)),
        })) = &value.value
        else {
            panic!("{input} is an immediate tensor")
        };
        match &tensor.value {
            Some(tensor_value::Value::Bools(bools)) => bools.values[0],
            other => panic!("{input} is a bool, got {other:?}"),
        }
    }

    fn tensor_type(value: &mil_spec::NamedValueType) -> (i32, Vec<u64>) {
        use mil_spec::{dimension, value_type};
        let Some(value_type::Type::TensorType(tensor)) = &value.r#type.as_ref().unwrap().r#type
        else {
            panic!("{} is a tensor", value.name)
        };
        (
            tensor.data_type,
            tensor
                .dimensions
                .iter()
                .map(|d| match &d.dimension {
                    Some(dimension::Dimension::Constant(constant)) => constant.size,
                    other => panic!("static dimension, got {other:?}"),
                })
                .collect(),
        )
    }

    /// Bytes of a weights blob holding `payloads`, each after a 64-byte record header and
    /// padded to 64 bytes, after the 64-byte file header.
    fn blob_size(payloads: &[usize]) -> usize {
        64 + payloads
            .iter()
            .map(|length| 64 + length.div_ceil(64) * 64)
            .sum::<usize>()
    }

    /// The usual chain read by `matmul(x, transpose(weight))`, with or without a carried
    /// state.
    fn projected_chain(stateful: bool, [rows, groups, width]: [u32; 3]) -> Lowered {
        let mut builder = MLGraphBuilder::new_uncompiled();
        let built = chain(
            &mut builder,
            DataType::Uint8,
            DataType::Float16,
            Some(128.0),
            [rows, groups, width],
        );
        let (_, y) = projection(&mut builder, built.weight, groups * width);
        let mut outputs = MLNamedOperands::from([("y", y)]);
        if stateful {
            outputs.insert("next_state", carried_state(&mut builder));
        }
        lower(&builder.finish_graph_info(&outputs).unwrap())
    }

    #[test]
    fn the_lowering_emits_one_compressed_constant() {
        let (rows, groups, width) = (64u32, 2u32, 32u32);
        let columns = groups * width;
        for stateful in [false, true] {
            let lowered = projected_chain(stateful, [rows, groups, width]);
            let [leaf] = lowered.of_type(BLOCKWISE)[..] else {
                panic!("one blockwise constant")
            };
            assert_eq!(
                tensor_type(&leaf.outputs[0]),
                (
                    mil_spec::DataType::Float16 as i32,
                    vec![rows as u64, columns as u64]
                )
            );
            let data = lowered.producer(bound_name(leaf, "data"));
            assert_eq!(
                tensor_type(&data.outputs[0]),
                (
                    mil_spec::DataType::Uint8 as i32,
                    vec![rows as u64, columns as u64]
                )
            );
            assert_eq!(
                lowered.blob(bound_name(leaf, "data")),
                stored_bytes(rows, columns)
            );
            let scale = lowered.producer(bound_name(leaf, "scale"));
            assert_eq!(
                tensor_type(&scale.outputs[0]),
                (
                    mil_spec::DataType::Float16 as i32,
                    vec![rows as u64, groups as u64]
                )
            );
            assert_eq!(
                lowered.blob(bound_name(leaf, "scale")),
                float_bytes(&scale_values(rows * groups), DataType::Float16)
            );
            let offset = lowered.producer(bound_name(leaf, "offset"));
            assert_eq!(
                tensor_type(&offset.outputs[0]),
                (
                    mil_spec::DataType::Uint8 as i32,
                    vec![rows as u64, groups as u64]
                )
            );
            assert_eq!(
                lowered.blob(bound_name(leaf, "offset")),
                vec![128u8; (rows * groups) as usize]
            );
            for literal in ["cast", "sub", "mul", "reshape"] {
                assert!(lowered.of_type(literal).is_empty(), "no {literal} remains");
            }
            // The stored bytes, the scales and the offsets; no float copy of the matrix.
            assert_eq!(
                lowered.weights.len(),
                blob_size(&[
                    (rows * columns) as usize,
                    (rows * groups * 2) as usize,
                    (rows * groups) as usize
                ])
            );
        }
    }

    #[test]
    fn a_transposed_matmul_reads_the_constant_with_transpose_y() {
        for stateful in [false, true] {
            let lowered = projected_chain(stateful, [64, 2, 32]);
            let [leaf] = lowered.of_type(BLOCKWISE)[..] else {
                panic!("one blockwise constant")
            };
            let [product] = lowered.of_type("matmul")[..] else {
                panic!("one matmul")
            };
            assert_eq!(bound_name(product, "y"), leaf.outputs[0].name);
            assert!(bound_bool(product, "transpose_y"));
            assert!(!bound_bool(product, "transpose_x"));
            assert!(lowered.of_type("transpose").is_empty());
        }
    }

    #[test]
    fn a_table_shared_by_a_gather_and_a_projection_is_stored_once() {
        let (rows, groups, width) = (96u32, 2u32, 32u32);
        let columns = groups * width;
        for stateful in [false, true] {
            let mut builder = MLGraphBuilder::new_uncompiled();
            // An int8 table with float32 scales narrowed after the reshape, as a float16
            // graph authors its embedding.
            let stored = float_constant(
                &mut builder,
                DataType::Float32,
                &[rows, groups],
                &scale_values(rows * groups),
            );
            let reshaped = builder
                .reshape(stored, dimensions(&[rows, groups, 1]))
                .unwrap();
            let scale = builder.cast(reshaped, DataType::Float16).unwrap();
            let built = chain_with_scale(
                &mut builder,
                DataType::Int8,
                DataType::Float16,
                None,
                [rows, groups, width],
                scale,
            );
            let token = builder
                .input(
                    "token",
                    &MLOperandDescriptor::new(DataType::Int32, vec![1, 1]),
                )
                .unwrap();
            let embedded = builder.gather(built.weight, token).unwrap();
            let (_, logits) = projection(&mut builder, built.weight, columns);
            let doubled = builder.add(embedded, embedded).unwrap();
            let transposed_again = builder.transpose(built.weight).unwrap();
            let logits_again = builder.matmul(doubled, transposed_again).unwrap();
            let mut outputs = MLNamedOperands::from([
                ("embedded", embedded),
                ("logits", logits),
                ("logits_again", logits_again),
            ]);
            if stateful {
                outputs.insert("next_state", carried_state(&mut builder));
            }
            let lowered = lower(&builder.finish_graph_info(&outputs).unwrap());

            let [leaf] = lowered.of_type(BLOCKWISE)[..] else {
                panic!("one constant for every reader")
            };
            assert!(!leaf.inputs.contains_key("offset"));
            let table = leaf.outputs[0].name.as_str();
            let [lookup] = lowered.of_type("gather")[..] else {
                panic!("one gather")
            };
            assert_eq!(bound_name(lookup, "x"), table);
            let products = lowered.of_type("matmul");
            assert_eq!(products.len(), 2);
            assert!(lowered.of_type("transpose").is_empty());
            for product in products {
                assert_eq!(bound_name(product, "y"), table);
                assert!(bound_bool(product, "transpose_y"));
            }
            assert_eq!(
                lowered.blob(bound_name(leaf, "data")),
                stored_bytes(rows, columns)
            );
            assert_eq!(
                lowered.weights.len(),
                blob_size(&[(rows * columns) as usize, (rows * groups * 2) as usize]),
                "the table is stored once, as int8"
            );
        }
    }

    #[test]
    fn a_kept_transpose_still_reads_the_one_constant() {
        let mut builder = MLGraphBuilder::new_uncompiled();
        let built = chain(
            &mut builder,
            DataType::Uint8,
            DataType::Float16,
            Some(128.0),
            [64, 2, 32],
        );
        let (transposed, y) = projection(&mut builder, built.weight, 64);
        let copy = builder.identity(transposed).unwrap();
        let next_state = carried_state(&mut builder);
        let graph = builder
            .finish_graph_info(&MLNamedOperands::from([
                ("y", y),
                ("copy", copy),
                ("next_state", next_state),
            ]))
            .unwrap();
        let lowered = lower(&graph);
        let [leaf] = lowered.of_type(BLOCKWISE)[..] else {
            panic!("one blockwise constant")
        };
        let [transpose] = lowered.of_type("transpose")[..] else {
            panic!("the transpose with a second reader is emitted")
        };
        assert_eq!(bound_name(transpose, "x"), leaf.outputs[0].name);
        let [product] = lowered.of_type("matmul")[..] else {
            panic!("one matmul")
        };
        assert_eq!(bound_name(product, "y"), leaf.outputs[0].name);
        assert!(bound_bool(product, "transpose_y"));
    }

    #[test]
    fn an_authored_blockwise_dequantize_linear_keeps_its_lowering() {
        let mut builder = MLGraphBuilder::new_uncompiled();
        let data = constant(&mut builder, DataType::Uint8, &[4, 8], stored_bytes(4, 8));
        let scale = float_constant(&mut builder, DataType::Float32, &[4, 2], &scale_values(8));
        let zero_point = constant(&mut builder, DataType::Uint8, &[4, 2], vec![128; 8]);
        let weight = builder
            .dequantize_linear_with_zeropoint(data, scale, zero_point)
            .unwrap();
        let graph = finish(builder, weight);
        assert!(find_chains(&graph).is_empty());
        assert!(lower(&graph).of_type(BLOCKWISE).is_empty());
    }

    #[test]
    fn the_exact_float16_lowering_of_a_float16_graph_keeps_the_literal_chain() {
        use crate::mlcontextoptions::CoremlFloat16Precision;
        let is_blockwise = |float_type, precision| {
            let mut builder = MLGraphBuilder::new_uncompiled();
            let built = chain(
                &mut builder,
                DataType::Uint8,
                float_type,
                Some(128.0),
                [64, 2, 32],
            );
            let x = builder
                .input("x", &MLOperandDescriptor::new(float_type, vec![1, 64]))
                .unwrap();
            let transposed = builder.transpose(built.weight).unwrap();
            let y = builder.matmul(x, transposed).unwrap();
            let graph = builder
                .finish_graph_info(&MLNamedOperands::from([("y", y)]))
                .unwrap();
            let model = super::super::CoremlMlProgramConverter
                .convert_with_float16_precision(&graph, precision)
                .unwrap()
                .data;
            model
                .windows(BLOCKWISE.len())
                .any(|window| window == BLOCKWISE.as_bytes())
        };
        assert!(is_blockwise(
            DataType::Float16,
            CoremlFloat16Precision::Native
        ));
        assert!(!is_blockwise(
            DataType::Float16,
            CoremlFloat16Precision::Exact
        ));
        // Without a float16 operand the exact lowering is the native one.
        assert!(is_blockwise(
            DataType::Float32,
            CoremlFloat16Precision::Exact
        ));
    }

    #[test]
    fn the_weights_blob_is_the_same_on_every_conversion() {
        let mut builder = MLGraphBuilder::new_uncompiled();
        let mut outputs = Vec::new();
        for index in 0..6u32 {
            let built = chain(
                &mut builder,
                if index % 2 == 0 {
                    DataType::Uint8
                } else {
                    DataType::Int8
                },
                DataType::Float16,
                (index % 3 != 0).then_some(index as f32),
                [8 + index, 2, 32],
            );
            outputs.push(builder.identity(built.weight).unwrap());
        }
        let names: Vec<String> = (0..outputs.len()).map(|i| format!("w{i}")).collect();
        let mut named = MLNamedOperands::new();
        for (name, output) in names.iter().zip(outputs) {
            named.insert(name, output);
        }
        let graph = builder.finish_graph_info(&named).unwrap();
        let first = lower(&graph);
        assert_eq!(first.of_type(BLOCKWISE).len(), 6);
        for _ in 0..8 {
            assert_eq!(lower(&graph).weights, first.weights);
        }
    }
}

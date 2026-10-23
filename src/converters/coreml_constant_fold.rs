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

//! Folds `transpose`, `reshape` and `cast` of constants before the CoreML lowering.
//!
//! The Apple Neural Engine admits `matmul(x, const)` but refuses `matmul(x, transpose(const))`
//! (measured with Core ML's compute plan on A12 and A13). GPT-Neo style graphs carry one such
//! transpose per projection, so lowering it literally also costs one run-time transpose of the
//! weight per token on the CPU and GPU paths. This prepass rewrites the three view operations
//! over constants into new constants at convert time, on a clone of the [`GraphInfo`], and
//! leaves everything else untouched:
//!
//! - the input must be an [`OperandKind::Constant`] with payload bytes, a static shape, no
//!   pending layout permutation and a whole-byte element type (4-bit constants are skipped);
//! - the output must be an intermediate operand with a static shape; graph outputs are never
//!   folded, so the copy proofs of the lowering keep working;
//! - `cast` is folded only when every element converts exactly per the WebNN cast table
//!   (integer to integer wraps to the low bits, float to float rounds to nearest or overflows
//!   to infinity, integer to float rounds to nearest, float to integer truncates). A float
//!   whose truncation does not fit the target, or a NaN, is implementation-defined in WebNN,
//!   so such a cast stays in the graph and the CoreML kernel decides;
//! - the folded output keeps its operand index and name and becomes the constant; the
//!   operation is dropped; a source constant with no remaining consumer loses its payload.
//!
//! Chains fold in one ordered pass because operations are recorded topologically: once
//! `cast(const)` has become a constant, a later `transpose` of it is a transpose of a constant.

use std::collections::HashMap;

use crate::graph::{ConstantData, DataType, Dimension, GraphInfo, OperandKind};
use crate::operators::Operation;

/// Returns a copy of `graph` with every foldable view of a constant folded, or `None` when
/// the graph has nothing to fold (the caller then lowers the original without cloning it).
pub(crate) fn fold_constant_views(graph: &GraphInfo) -> Option<GraphInfo> {
    let has_candidate = graph.operations.iter().any(|operation| {
        matches!(
            operation,
            Operation::Transpose { .. } | Operation::Reshape { .. } | Operation::Cast { .. }
        ) && operation
            .output_operand()
            .is_some_and(|output| !graph.output_operands.contains(&output))
            && operation
                .input_operands()
                .first()
                .is_some_and(|&input| graph.constant_operand_ids_to_handles.contains_key(&input))
    });
    if !has_candidate {
        return None;
    }

    let mut folded = graph.clone();
    let mut use_count: HashMap<u32, usize> = HashMap::new();
    for operation in &folded.operations {
        for id in operation
            .input_operands()
            .into_iter()
            .chain(operation.option_operands())
        {
            *use_count.entry(id).or_default() += 1;
        }
    }
    for &id in &folded.output_operands {
        *use_count.entry(id).or_default() += 1;
    }

    let mut changed = false;
    let operations = std::mem::take(&mut folded.operations);
    let mut kept = Vec::with_capacity(operations.len());
    for operation in operations {
        let Some((input, output, data)) = fold_one(&folded, &operation) else {
            kept.push(operation);
            continue;
        };
        changed = true;
        let label = folded
            .constant_operand_ids_to_handles
            .get(&input)
            .and_then(|constant| constant.label.clone());
        folded.operands[output as usize].kind = OperandKind::Constant;
        folded
            .constant_operand_ids_to_handles
            .insert(output, ConstantData { data, label });
        let remaining = use_count.entry(input).or_default();
        *remaining = remaining.saturating_sub(1);
        if *remaining == 0 {
            folded.constant_operand_ids_to_handles.remove(&input);
        }
    }
    folded.operations = kept;
    changed.then_some(folded)
}

/// The folded payload of one operation, with its input and output operand indices, when the
/// operation is a view of a constant that can be folded exactly.
fn fold_one(graph: &GraphInfo, operation: &Operation) -> Option<(u32, u32, Vec<u8>)> {
    let input = match operation {
        Operation::Transpose { input, .. }
        | Operation::Reshape { input, .. }
        | Operation::Cast { input, .. } => *input,
        _ => return None,
    };
    let [output] = operation.outputs() else {
        return None;
    };
    let output = *output;
    if graph.output_operands.contains(&output) || graph.input_operands.contains(&output) {
        return None;
    }
    let source = graph.operand(input)?;
    let destination = graph.operand(output)?;
    if source.kind != OperandKind::Constant || destination.kind != OperandKind::Intermediate {
        return None;
    }
    // Narrow integer constants are quantized weights: their casts, reshapes and transposes
    // are the dequantization chains that later lowering recognizes (and a folded cast would
    // store the widened payload). Leave them for those passes.
    if is_quantized_storage(source.descriptor.data_type) {
        return None;
    }
    if !source.descriptor.pending_permutation.is_empty()
        || !destination.descriptor.pending_permutation.is_empty()
    {
        return None;
    }
    let source_shape = source.descriptor.static_shape()?;
    let destination_shape = destination.descriptor.static_shape()?;
    let source_type = source.descriptor.data_type;
    let destination_type = destination.descriptor.data_type;
    if is_unsupported_scalar_type(source_type) || is_unsupported_scalar_type(destination_type) {
        return None;
    }
    let data = &graph.constant_operand_ids_to_handles.get(&input)?.data;
    let element_count = source.descriptor.element_count()?;
    if data.len() != element_count.checked_mul(source_type.bytes_per_element())?
        || destination.descriptor.element_count()? != element_count
    {
        return None;
    }

    let payload = match operation {
        Operation::Transpose { options, .. } => {
            if source_type != destination_type {
                return None;
            }
            let rank = source_shape.len();
            let permutation: Vec<usize> = match options.as_ref().map(|o| o.permutation.as_slice()) {
                Some(permutation) if !permutation.is_empty() => {
                    permutation.iter().map(|&axis| axis as usize).collect()
                }
                _ => (0..rank).rev().collect(),
            };
            if !is_permutation(&permutation, rank) {
                return None;
            }
            let permuted_shape: Vec<u32> = permutation.iter().map(|&p| source_shape[p]).collect();
            if permuted_shape != destination_shape {
                return None;
            }
            transpose_bytes(
                data,
                &source_shape,
                &permutation,
                source_type.bytes_per_element(),
            )
        }
        Operation::Reshape { new_shape, .. } => {
            if source_type != destination_type {
                return None;
            }
            let requested: Vec<u32> = new_shape
                .iter()
                .map(|dimension| match Dimension::from(dimension.clone()) {
                    Dimension::Static(size) => Some(size),
                    Dimension::Dynamic(_) => None,
                })
                .collect::<Option<_>>()?;
            if requested != destination_shape {
                return None;
            }
            data.clone()
        }
        Operation::Cast { data_type, .. } => {
            if DataType::from(*data_type) != destination_type || source_shape != destination_shape {
                return None;
            }
            // A fold never grows the stored payload: a widening cast (float16 weights read
            // as float32, for example) stays a runtime cast over the original bytes.
            if destination_type.bytes_per_element() > source_type.bytes_per_element() {
                return None;
            }
            cast_bytes(data, source_type, destination_type)?
        }
        _ => return None,
    };
    Some((input, output, payload))
}

fn is_quantized_storage(data_type: DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8 | DataType::Uint8 | DataType::Int4 | DataType::Uint4
    )
}

fn is_unsupported_scalar_type(data_type: DataType) -> bool {
    matches!(
        data_type,
        DataType::Int4 | DataType::Uint4 | DataType::Bfloat16 | DataType::Float8e4m3
    )
}

fn is_permutation(permutation: &[usize], rank: usize) -> bool {
    if permutation.len() != rank {
        return false;
    }
    let mut seen = vec![false; rank];
    permutation
        .iter()
        .all(|&axis| axis < rank && !std::mem::replace(&mut seen[axis], true))
}

/// Reorders the elements of a row-major tensor of `shape` so that output axis `i` is input
/// axis `permutation[i]`. Works for any whole-byte element size.
fn transpose_bytes(
    data: &[u8],
    shape: &[u32],
    permutation: &[usize],
    element_size: usize,
) -> Vec<u8> {
    let rank = shape.len();
    let mut input_strides = vec![0usize; rank];
    let mut stride = 1usize;
    for axis in (0..rank).rev() {
        input_strides[axis] = stride;
        stride *= shape[axis] as usize;
    }
    let output_shape: Vec<usize> = permutation.iter().map(|&p| shape[p] as usize).collect();
    let output_strides: Vec<usize> = permutation.iter().map(|&p| input_strides[p]).collect();

    let mut out = Vec::with_capacity(data.len());
    if data.is_empty() {
        return out;
    }
    let mut index = vec![0usize; rank];
    let mut offset = 0usize;
    loop {
        let start = offset * element_size;
        out.extend_from_slice(&data[start..start + element_size]);
        // Advance the output multi-index, carrying into outer axes.
        let mut axis = rank;
        loop {
            if axis == 0 {
                return out;
            }
            axis -= 1;
            index[axis] += 1;
            offset += output_strides[axis];
            if index[axis] < output_shape[axis] {
                break;
            }
            offset -= output_strides[axis] * output_shape[axis];
            index[axis] = 0;
        }
    }
}

/// Exact intermediate for the WebNN cast table: every integer type fits `i128`, every float
/// type fits `f64` without rounding.
#[derive(Clone, Copy)]
enum Scalar {
    Int(i128),
    Float(f64),
}

fn read_scalars(data: &[u8], data_type: DataType) -> Vec<Scalar> {
    let size = data_type.bytes_per_element();
    data.chunks_exact(size)
        .map(|chunk| match data_type {
            DataType::Float32 => {
                Scalar::Float(f32::from_le_bytes(chunk.try_into().unwrap()) as f64)
            }
            DataType::Float16 => {
                Scalar::Float(half::f16::from_le_bytes(chunk.try_into().unwrap()).to_f64())
            }
            DataType::Int32 => Scalar::Int(i32::from_le_bytes(chunk.try_into().unwrap()) as i128),
            DataType::Uint32 => Scalar::Int(u32::from_le_bytes(chunk.try_into().unwrap()) as i128),
            DataType::Int8 => Scalar::Int(i8::from_le_bytes(chunk.try_into().unwrap()) as i128),
            DataType::Uint8 => Scalar::Int(u8::from_le_bytes(chunk.try_into().unwrap()) as i128),
            DataType::Int64 => Scalar::Int(i64::from_le_bytes(chunk.try_into().unwrap()) as i128),
            DataType::Uint64 => Scalar::Int(u64::from_le_bytes(chunk.try_into().unwrap()) as i128),
            DataType::Bfloat16 | DataType::Float8e4m3 | DataType::Int4 | DataType::Uint4 => {
                unreachable!("unsupported scalar types are rejected earlier")
            }
        })
        .collect()
}

/// A float that truncates to an integer in `[min, max]`; `None` for NaN or out of range,
/// which WebNN leaves implementation-defined.
fn float_to_int(value: f64, min: i128, max: i128) -> Option<i128> {
    if value.is_nan() {
        return None;
    }
    let truncated = value.trunc();
    // Compare in f64 first so that huge magnitudes never overflow the i128 conversion.
    if truncated < min as f64 || truncated > max as f64 {
        return None;
    }
    let integer = truncated as i128;
    (min..=max).contains(&integer).then_some(integer)
}

/// Rounds an exact integer to the nearest `f64`-representable value, then to the target.
/// Values beyond the target's range become infinity, as the cast table specifies.
fn int_to_f32(value: i128) -> f32 {
    value as f32
}

fn int_to_f16(value: i128) -> half::f16 {
    // Every magnitude that is finite in float16 is far below 2^53, so the f64 step is exact
    // there; larger magnitudes overflow to infinity either way.
    half::f16::from_f64(value as f64)
}

fn cast_bytes(data: &[u8], from: DataType, to: DataType) -> Option<Vec<u8>> {
    if from == to {
        return Some(data.to_vec());
    }
    let scalars = read_scalars(data, from);
    let mut out = Vec::with_capacity(scalars.len() * to.bytes_per_element());
    for scalar in scalars {
        match to {
            DataType::Float32 => {
                let value = match scalar {
                    Scalar::Float(f) => f as f32,
                    Scalar::Int(i) => int_to_f32(i),
                };
                out.extend_from_slice(&value.to_le_bytes());
            }
            DataType::Float16 => {
                let value = match scalar {
                    Scalar::Float(f) => half::f16::from_f64(f),
                    Scalar::Int(i) => int_to_f16(i),
                };
                out.extend_from_slice(&value.to_le_bytes());
            }
            DataType::Int32 => {
                let value = int_value(scalar, i32::MIN as i128, i32::MAX as i128)? as i32;
                out.extend_from_slice(&value.to_le_bytes());
            }
            DataType::Uint32 => {
                let value = int_value(scalar, 0, u32::MAX as i128)? as u32;
                out.extend_from_slice(&value.to_le_bytes());
            }
            DataType::Int8 => {
                let value = int_value(scalar, i8::MIN as i128, i8::MAX as i128)? as i8;
                out.extend_from_slice(&value.to_le_bytes());
            }
            DataType::Uint8 => {
                let value = int_value(scalar, 0, u8::MAX as i128)? as u8;
                out.extend_from_slice(&value.to_le_bytes());
            }
            DataType::Int64 => {
                let value = int_value(scalar, i64::MIN as i128, i64::MAX as i128)? as i64;
                out.extend_from_slice(&value.to_le_bytes());
            }
            DataType::Uint64 => {
                let value = int_value(scalar, 0, u64::MAX as i128)? as u64;
                out.extend_from_slice(&value.to_le_bytes());
            }
            DataType::Bfloat16 | DataType::Float8e4m3 | DataType::Int4 | DataType::Uint4 => {
                return None;
            }
        }
    }
    Some(out)
}

/// The integer to store for `scalar` in a target with range `[min, max]`: integers wrap to the
/// target width (the `as` casts above take the low bits of the two's complement value, which
/// is what the WebNN cast table specifies); floats must truncate into range exactly.
fn int_value(scalar: Scalar, min: i128, max: i128) -> Option<i128> {
    match scalar {
        Scalar::Int(i) => Some(i),
        Scalar::Float(f) => float_to_int(f, min, max),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Operand, OperandDescriptor};
    use crate::operator_enums::MLOperandDataType;
    use crate::operator_options::{MLDimension, MLTransposeOptions};

    fn descriptor(data_type: DataType, shape: &[u32]) -> OperandDescriptor {
        OperandDescriptor {
            data_type,
            shape: shape.iter().map(|&d| Dimension::Static(d)).collect(),
            pending_permutation: vec![],
        }
    }

    fn operand(kind: OperandKind, data_type: DataType, shape: &[u32], name: &str) -> Operand {
        Operand {
            kind,
            descriptor: descriptor(data_type, shape),
            name: Some(name.to_string()),
        }
    }

    fn f16_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
        values
            .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
            .collect()
    }

    fn f16_values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    }

    #[test]
    fn widening_casts_keep_the_narrow_payload() {
        // float16 weights read as float32 stay float16 in the weight blob; the narrowing
        // direction still folds.
        let graph = |source: DataType, target: MLOperandDataType, bytes: Vec<u8>| {
            let mut graph = GraphInfo {
                operands: vec![
                    operand(OperandKind::Constant, source, &[2], "w"),
                    operand(
                        OperandKind::Intermediate,
                        DataType::from(target),
                        &[2],
                        "cast",
                    ),
                    operand(OperandKind::Output, DataType::from(target), &[2], "y"),
                ],
                output_operands: vec![2],
                operations: vec![
                    Operation::Cast {
                        input: 0,
                        data_type: target,
                        options: None,
                        outputs: vec![1],
                    },
                    Operation::Identity {
                        input: 1,
                        options: None,
                        outputs: vec![2],
                    },
                ],
                ..Default::default()
            };
            graph.constant_operand_ids_to_handles.insert(
                0,
                ConstantData {
                    data: bytes,
                    label: None,
                },
            );
            graph
        };
        let half: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
            .collect();
        assert!(
            fold_constant_views(&graph(DataType::Float16, MLOperandDataType::Float32, half))
                .is_none()
        );
        let single: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let folded = fold_constant_views(&graph(
            DataType::Float32,
            MLOperandDataType::Float16,
            single,
        ))
        .expect("a narrowing cast of a constant folds");
        assert_eq!(folded.constant_operand_ids_to_handles[&1].data.len(), 4);
    }

    #[test]
    fn narrow_integer_constants_are_left_for_the_dequantization_passes() {
        let mut graph = GraphInfo {
            operands: vec![
                operand(OperandKind::Constant, DataType::Int8, &[4, 2], "w"),
                operand(OperandKind::Intermediate, DataType::Float32, &[4, 2], "w32"),
                operand(OperandKind::Intermediate, DataType::Int8, &[2, 4], "wT"),
                operand(OperandKind::Output, DataType::Float32, &[4, 2], "y"),
            ],
            output_operands: vec![3],
            operations: vec![
                Operation::Cast {
                    input: 0,
                    data_type: MLOperandDataType::Float32,
                    options: None,
                    outputs: vec![1],
                },
                Operation::Transpose {
                    input: 0,
                    options: Some(MLTransposeOptions {
                        label: "t".into(),
                        permutation: vec![1, 0],
                    }),
                    outputs: vec![2],
                },
                Operation::Identity {
                    input: 1,
                    options: None,
                    outputs: vec![3],
                },
            ],
            ..Default::default()
        };
        graph.constant_operand_ids_to_handles.insert(
            0,
            ConstantData {
                data: vec![1, 2, 3, 4, 5, 6, 7, 8],
                label: Some("w".into()),
            },
        );
        assert!(fold_constant_views(&graph).is_none());
    }

    /// `x: [1, 256]` input, `w: [256, 256]` float16 constant, `matmul(x, transpose(w))`.
    fn transposed_projection() -> GraphInfo {
        let n = 256u32;
        let mut graph = GraphInfo {
            operands: vec![
                operand(OperandKind::Input, DataType::Float16, &[1, n], "x"),
                operand(OperandKind::Constant, DataType::Float16, &[n, n], "w"),
                operand(OperandKind::Intermediate, DataType::Float16, &[n, n], "wT"),
                operand(OperandKind::Output, DataType::Float16, &[1, n], "y"),
            ],
            input_operands: vec![0],
            output_operands: vec![3],
            operations: vec![
                Operation::Transpose {
                    input: 1,
                    options: Some(MLTransposeOptions {
                        label: "t".into(),
                        permutation: vec![1, 0],
                    }),
                    outputs: vec![2],
                },
                Operation::Matmul {
                    a: 0,
                    b: 2,
                    options: None,
                    outputs: vec![3],
                },
            ],
            ..Default::default()
        };
        // Distinct, exactly representable float16 values: row * 1/16 + column * 1/256... keep
        // it simple with integers below 2048, which float16 stores exactly.
        let data = f16_bytes((0..n * n).map(|i| ((i * 7) % 2048) as f32));
        graph.constant_operand_ids_to_handles.insert(
            1,
            ConstantData {
                data,
                label: Some("w".into()),
            },
        );
        graph
    }

    #[test]
    fn float16_transpose_feeding_matmul_folds_into_a_constant_operand() {
        let graph = transposed_projection();
        let folded = fold_constant_views(&graph).expect("transpose of a constant folds");

        assert_eq!(folded.operations.len(), 1);
        assert!(matches!(
            folded.operations[0],
            Operation::Matmul { a: 0, b: 2, .. }
        ));
        assert_eq!(folded.operands[2].kind, OperandKind::Constant);
        assert_eq!(folded.operands[2].descriptor.data_type, DataType::Float16);
        assert_eq!(
            folded.operands[2].descriptor.shape,
            descriptor(DataType::Float16, &[256, 256]).shape
        );
        assert_eq!(folded.operands[2].name.as_deref(), Some("wT"));
        // The source constant has no consumer left, so its payload is dropped.
        assert!(!folded.constant_operand_ids_to_handles.contains_key(&1));
        assert_eq!(folded.operands[1].kind, OperandKind::Constant);
        // Inputs, outputs and the rest of the operand table are untouched.
        assert_eq!(folded.input_operands, graph.input_operands);
        assert_eq!(folded.output_operands, graph.output_operands);
        assert_eq!(folded.operands.len(), graph.operands.len());

        let source = f16_values(&graph.constant_operand_ids_to_handles[&1].data);
        let result = &folded.constant_operand_ids_to_handles[&2];
        assert_eq!(result.label.as_deref(), Some("w"));
        let transposed = f16_values(&result.data);
        assert_eq!(transposed.len(), 256 * 256);
        for row in 0..256 {
            for column in 0..256 {
                assert_eq!(
                    transposed[row * 256 + column],
                    source[column * 256 + row],
                    "element [{row}, {column}]"
                );
            }
        }
    }

    #[test]
    fn transpose_permutes_every_whole_byte_element_size() {
        // [2, 3, 2] -> permutation [2, 0, 1] -> [2, 2, 3]
        let shape = [2u32, 3, 2];
        let permutation = [2usize, 0, 1];
        let expected: Vec<u32> = {
            let mut out = Vec::new();
            for k in 0..2u32 {
                for i in 0..2u32 {
                    for j in 0..3u32 {
                        out.push(i * 6 + j * 2 + k);
                    }
                }
            }
            out
        };
        for element_size in [1usize, 2, 4, 8] {
            let data: Vec<u8> = (0..12u32)
                .flat_map(|v| (v as u64).to_le_bytes()[..element_size].to_vec())
                .collect();
            let transposed = transpose_bytes(&data, &shape, &permutation, element_size);
            let values: Vec<u32> = transposed
                .chunks_exact(element_size)
                .map(|chunk| {
                    let mut bytes = [0u8; 8];
                    bytes[..element_size].copy_from_slice(chunk);
                    u64::from_le_bytes(bytes) as u32
                })
                .collect();
            assert_eq!(values, expected, "element size {element_size}");
        }
    }

    #[test]
    fn reshape_of_a_constant_keeps_the_bytes_and_takes_the_new_shape() {
        let data: Vec<u8> = (0..6i32).flat_map(i32::to_le_bytes).collect();
        let mut graph = GraphInfo {
            operands: vec![
                operand(OperandKind::Input, DataType::Int32, &[3, 2], "x"),
                operand(OperandKind::Constant, DataType::Int32, &[2, 3], "c"),
                operand(OperandKind::Intermediate, DataType::Int32, &[3, 2], "r"),
                operand(OperandKind::Output, DataType::Int32, &[3, 2], "y"),
            ],
            input_operands: vec![0],
            output_operands: vec![3],
            operations: vec![
                Operation::Reshape {
                    input: 1,
                    new_shape: vec![MLDimension::Static(3), MLDimension::Static(2)],
                    options: None,
                    outputs: vec![2],
                },
                Operation::Add {
                    a: 0,
                    b: 2,
                    options: None,
                    outputs: vec![3],
                },
            ],
            ..Default::default()
        };
        graph.constant_operand_ids_to_handles.insert(
            1,
            ConstantData {
                data: data.clone(),
                label: None,
            },
        );
        let folded = fold_constant_views(&graph).expect("reshape of a constant folds");
        assert_eq!(folded.operations.len(), 1);
        assert!(matches!(folded.operations[0], Operation::Add { b: 2, .. }));
        assert_eq!(folded.operands[2].kind, OperandKind::Constant);
        assert_eq!(folded.operands[2].descriptor.data_type, DataType::Int32);
        assert_eq!(
            folded.operands[2].descriptor.shape,
            descriptor(DataType::Int32, &[3, 2]).shape
        );
        assert_eq!(folded.constant_operand_ids_to_handles[&2].data, data);
        assert!(!folded.constant_operand_ids_to_handles.contains_key(&1));
    }

    fn cast_graph(from: DataType, to: MLOperandDataType, data: Vec<u8>) -> GraphInfo {
        let to_type = DataType::from(to);
        let mut graph = GraphInfo {
            operands: vec![
                operand(OperandKind::Input, to_type, &[4], "x"),
                operand(OperandKind::Constant, from, &[4], "c"),
                operand(OperandKind::Intermediate, to_type, &[4], "cast"),
                operand(OperandKind::Output, to_type, &[4], "y"),
            ],
            input_operands: vec![0],
            output_operands: vec![3],
            operations: vec![
                Operation::Cast {
                    input: 1,
                    data_type: to,
                    options: None,
                    outputs: vec![2],
                },
                Operation::Add {
                    a: 0,
                    b: 2,
                    options: None,
                    outputs: vec![3],
                },
            ],
            ..Default::default()
        };
        graph
            .constant_operand_ids_to_handles
            .insert(1, ConstantData { data, label: None });
        graph
    }

    #[test]
    fn cast_int32_to_float32_of_a_constant_folds() {
        let values = [0i32, -1, 16_777_217, i32::MIN];
        let graph = cast_graph(
            DataType::Int32,
            MLOperandDataType::Float32,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        );
        let folded = fold_constant_views(&graph).expect("cast of a constant folds");
        assert_eq!(folded.operations.len(), 1);
        assert_eq!(folded.operands[2].kind, OperandKind::Constant);
        let result: Vec<f32> = folded.constant_operand_ids_to_handles[&2]
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // 16_777_217 rounds to the nearest float32 (16_777_216), as the cast table says.
        assert_eq!(result, vec![0.0, -1.0, 16_777_216.0, i32::MIN as f32]);
        assert!(!folded.constant_operand_ids_to_handles.contains_key(&1));
    }

    #[test]
    fn cast_follows_the_webnn_table_for_integers_and_floats() {
        // int8 -> uint8 wraps: -1 becomes 255.
        let out = cast_bytes(&[0xFF, 0x7F, 0x80, 0x00], DataType::Int8, DataType::Uint8).unwrap();
        assert_eq!(out, vec![255, 127, 128, 0]);
        // int32 -> int8 keeps the low 8 bits.
        let out = cast_bytes(&300i32.to_le_bytes(), DataType::Int32, DataType::Int8).unwrap();
        assert_eq!(out, vec![44]);
        // float32 -> int32 truncates toward zero.
        let data: Vec<u8> = [2.75f32, -2.75, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = cast_bytes(&data, DataType::Float32, DataType::Int32).unwrap();
        let ints: Vec<i32> = out
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(ints, vec![2, -2, 0]);
        // float32 -> float16 rounds to nearest and overflows to infinity.
        let data: Vec<u8> = [1.0f32 + 1.0 / 4096.0, 1.0e6]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = cast_bytes(&data, DataType::Float32, DataType::Float16).unwrap();
        let halves = f16_values(&out);
        assert_eq!(halves[0], 1.0);
        assert!(halves[1].is_infinite() && halves[1] > 0.0);
        // float16 -> float32 is exact; int64 -> int32 keeps the low 32 bits.
        let out = cast_bytes(
            &half::f16::from_f32(-0.5).to_le_bytes(),
            DataType::Float16,
            DataType::Float32,
        )
        .unwrap();
        assert_eq!(f32::from_le_bytes(out.try_into().unwrap()), -0.5);
        let out = cast_bytes(
            &((1i64 << 40) | 5).to_le_bytes(),
            DataType::Int64,
            DataType::Int32,
        )
        .unwrap();
        assert_eq!(i32::from_le_bytes(out.try_into().unwrap()), 5);
        // uint32 -> int64 and int32 -> uint64 widen exactly.
        let out = cast_bytes(&u32::MAX.to_le_bytes(), DataType::Uint32, DataType::Int64).unwrap();
        assert_eq!(i64::from_le_bytes(out.try_into().unwrap()), u32::MAX as i64);
        let out = cast_bytes(&(-1i32).to_le_bytes(), DataType::Int32, DataType::Uint64).unwrap();
        assert_eq!(u64::from_le_bytes(out.try_into().unwrap()), u64::MAX);
    }

    #[test]
    fn float_to_integer_casts_that_are_implementation_defined_are_not_folded() {
        for value in [300.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let graph = cast_graph(
                DataType::Float32,
                MLOperandDataType::Uint8,
                [value, 1.0, 2.0, 3.0]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            );
            assert!(
                fold_constant_views(&graph).is_none(),
                "{value} does not fit uint8 exactly"
            );
        }
        // In range it folds.
        let graph = cast_graph(
            DataType::Float32,
            MLOperandDataType::Uint8,
            [255.9f32, 1.5, 2.0, 0.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        );
        let folded = fold_constant_views(&graph).unwrap();
        assert_eq!(
            folded.constant_operand_ids_to_handles[&2].data,
            vec![255, 1, 2, 0]
        );
    }

    #[test]
    fn transpose_of_a_graph_input_is_left_alone() {
        let graph = GraphInfo {
            operands: vec![
                operand(OperandKind::Input, DataType::Float32, &[2, 3], "x"),
                operand(OperandKind::Intermediate, DataType::Float32, &[3, 2], "xT"),
                operand(OperandKind::Output, DataType::Float32, &[3, 2], "y"),
            ],
            input_operands: vec![0],
            output_operands: vec![2],
            operations: vec![
                Operation::Transpose {
                    input: 0,
                    options: None,
                    outputs: vec![1],
                },
                Operation::Relu {
                    input: 1,
                    options: None,
                    outputs: vec![2],
                },
            ],
            ..Default::default()
        };
        assert!(fold_constant_views(&graph).is_none());
    }

    #[test]
    fn constant_with_another_consumer_keeps_its_payload() {
        let mut graph = transposed_projection();
        // A second consumer of `w` itself: y2 = matmul(x, w).
        graph.operands.push(operand(
            OperandKind::Output,
            DataType::Float16,
            &[1, 256],
            "y2",
        ));
        graph.output_operands.push(4);
        graph.operations.push(Operation::Matmul {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![4],
        });
        let folded = fold_constant_views(&graph).unwrap();
        assert_eq!(folded.operations.len(), 2);
        assert!(
            folded
                .operations
                .iter()
                .all(|op| matches!(op, Operation::Matmul { .. }))
        );
        assert_eq!(
            folded.constant_operand_ids_to_handles[&1].data,
            graph.constant_operand_ids_to_handles[&1].data
        );
        assert_eq!(folded.operands[1].kind, OperandKind::Constant);
        assert!(folded.constant_operand_ids_to_handles.contains_key(&2));
    }

    #[test]
    fn a_view_whose_output_is_a_graph_output_is_not_folded() {
        let mut graph = transposed_projection();
        graph.operands[2].kind = OperandKind::Output;
        graph.output_operands.push(2);
        graph.output_operands.sort_unstable();
        assert!(fold_constant_views(&graph).is_none());
    }

    #[test]
    fn chains_fold_in_one_pass() {
        // transpose(cast(c)) with c int32 [2, 3] -> float32 [3, 2]
        let data: Vec<u8> = (0..6i32).flat_map(i32::to_le_bytes).collect();
        let mut graph = GraphInfo {
            operands: vec![
                operand(OperandKind::Input, DataType::Float32, &[3, 2], "x"),
                operand(OperandKind::Constant, DataType::Int32, &[2, 3], "c"),
                operand(OperandKind::Intermediate, DataType::Float32, &[2, 3], "cf"),
                operand(OperandKind::Intermediate, DataType::Float32, &[3, 2], "cfT"),
                operand(OperandKind::Output, DataType::Float32, &[3, 2], "y"),
            ],
            input_operands: vec![0],
            output_operands: vec![4],
            operations: vec![
                Operation::Cast {
                    input: 1,
                    data_type: MLOperandDataType::Float32,
                    options: None,
                    outputs: vec![2],
                },
                Operation::Transpose {
                    input: 2,
                    options: None,
                    outputs: vec![3],
                },
                Operation::Add {
                    a: 0,
                    b: 3,
                    options: None,
                    outputs: vec![4],
                },
            ],
            ..Default::default()
        };
        graph
            .constant_operand_ids_to_handles
            .insert(1, ConstantData { data, label: None });
        let folded = fold_constant_views(&graph).unwrap();
        assert_eq!(folded.operations.len(), 1);
        assert!(matches!(folded.operations[0], Operation::Add { b: 3, .. }));
        assert_eq!(
            folded
                .constant_operand_ids_to_handles
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            vec![3]
        );
        let result: Vec<f32> = folded.constant_operand_ids_to_handles[&3]
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(result, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn packed_four_bit_constants_are_not_folded() {
        let mut graph = GraphInfo {
            operands: vec![
                operand(OperandKind::Constant, DataType::Int4, &[2, 2], "c"),
                operand(OperandKind::Intermediate, DataType::Int4, &[2, 2], "cT"),
                operand(OperandKind::Output, DataType::Float32, &[2, 2], "y"),
            ],
            output_operands: vec![2],
            operations: vec![
                Operation::Transpose {
                    input: 0,
                    options: None,
                    outputs: vec![1],
                },
                Operation::Cast {
                    input: 1,
                    data_type: MLOperandDataType::Float32,
                    options: None,
                    outputs: vec![2],
                },
            ],
            ..Default::default()
        };
        graph.constant_operand_ids_to_handles.insert(
            0,
            ConstantData {
                data: vec![0x21, 0x43],
                label: None,
            },
        );
        assert!(fold_constant_views(&graph).is_none());
    }

    #[test]
    fn the_coreml_lowering_emits_the_folded_weight_as_a_const() {
        use crate::converters::GraphConverter;
        use crate::protos::coreml::specification;
        use prost::Message;

        let graph = transposed_projection();
        let converted = super::super::CoremlMlProgramConverter
            .convert(&graph)
            .expect("the folded graph lowers");
        let model = specification::Model::decode(converted.data.as_slice()).unwrap();
        let specification::model::Type::MlProgram(program) = model.r#type.unwrap() else {
            panic!("expected MLProgram")
        };
        let operations = &program.functions["main"].block_specializations["CoreML7"].operations;
        assert!(
            operations.iter().all(|op| op.r#type != "transpose"),
            "no run-time transpose remains: {:?}",
            operations.iter().map(|op| &op.r#type).collect::<Vec<_>>()
        );
        let matmul = operations
            .iter()
            .find(|op| op.r#type == "matmul")
            .expect("matmul is emitted");
        let weight = match &matmul.inputs["y"].arguments[0].binding {
            Some(crate::protos::coreml::mil_spec::argument::binding::Binding::Name(name)) => {
                name.clone()
            }
            other => panic!("matmul weight must be bound by name, got {other:?}"),
        };
        assert_eq!(weight, "wT");
        let producer = operations
            .iter()
            .find(|op| op.outputs.iter().any(|output| output.name == weight))
            .expect("the weight has a producer");
        assert_eq!(producer.r#type, "const");
        // The transposed float16 payload is in the weight blob, once.
        let source = f16_values(&graph.constant_operand_ids_to_handles[&1].data);
        let weights = converted.weights_data.expect("float16 weights sidecar");
        let expected = f16_bytes((0..256 * 256).map(|i| source[(i % 256) * 256 + i / 256]));
        assert!(
            weights
                .windows(expected.len())
                .any(|window| window == expected.as_slice()),
            "the sidecar carries the transposed payload"
        );
        assert_eq!(
            weights.len(),
            expected.len() + 128,
            "the untransposed original is not emitted"
        );
    }
}

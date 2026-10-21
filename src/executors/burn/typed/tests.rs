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

use super::*;
use crate::operator_options::*;
use burn_ndarray::NdArray;
use std::collections::HashMap;

struct Runtime {
    env: TensorEnv<NdArray<f32>>,
    types: HashMap<u32, DataType>,
}
impl Runtime {
    fn new() -> Self {
        Self {
            env: TensorEnv::new(Default::default()),
            types: HashMap::new(),
        }
    }
    fn integers(&mut self, id: u32, dtype: DataType, shape: &[usize], values: &[i128]) {
        self.types.insert(id, dtype);
        let (_, signed) = domain(dtype);
        let value = if signed {
            super::super::value::TensorValue::from_signed(
                dtype,
                shape.to_vec(),
                values.iter().map(|&v| v as i64).collect(),
            )
        } else {
            super::super::value::TensorValue::from_unsigned(
                dtype,
                shape.to_vec(),
                values.iter().map(|&v| v as u64).collect(),
            )
        }
        .unwrap();
        self.env.insert_value(id, value);
    }
    fn floats(&mut self, id: u32, dtype: DataType, shape: &[usize], values: &[f32]) {
        self.types.insert(id, dtype);
        self.env
            .insert_host(
                id,
                dtype,
                HostArray::new(shape.to_vec(), values.to_vec()).unwrap(),
            )
            .unwrap();
    }
    fn run(&mut self, op: Operation, dtype: DataType) {
        for &id in op.outputs() {
            self.types.insert(id, dtype);
        }
        super::super::execute::execute_operations(&mut self.env, &[op], &self.types, &[]).unwrap();
    }
    fn expect(&self, id: u32, shape: &[usize], values: &[i128]) {
        let a = Array::load(&self.env, id).unwrap();
        assert_eq!(a.shape, shape);
        assert_eq!(
            a.values
                .into_iter()
                .map(|v| v.integer().unwrap())
                .collect::<Vec<_>>(),
            values
        );
    }
}
#[test]
fn integer_arithmetic_comparison_and_width_are_exact() {
    for dtype in [
        DataType::Int32,
        DataType::Uint32,
        DataType::Int64,
        DataType::Uint64,
    ] {
        let mut r = Runtime::new();
        r.integers(0, dtype, &[2], &[16_777_217, 16_777_219]);
        r.integers(1, dtype, &[], &[2]);
        r.run(
            Operation::Add {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![2],
            },
            dtype,
        );
        r.expect(2, &[2], &[16_777_219, 16_777_221]);
        r.run(
            Operation::Sub {
                a: 2,
                b: 1,
                options: None,
                outputs: vec![3],
            },
            dtype,
        );
        r.expect(3, &[2], &[16_777_217, 16_777_219]);
        r.run(
            Operation::Mul {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![4],
            },
            dtype,
        );
        r.expect(4, &[2], &[33_554_434, 33_554_438]);
        r.run(
            Operation::Div {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![5],
            },
            dtype,
        );
        r.expect(5, &[2], &[8_388_608, 8_388_609]);
        r.integers(6, dtype, &[2], &[16_777_216, 16_777_220]);
        r.run(
            Operation::Equal {
                a: 0,
                b: 6,
                options: None,
                outputs: vec![7],
            },
            DataType::Uint8,
        );
        r.expect(7, &[2], &[0, 0]);
        r.run(
            Operation::Greater {
                a: 0,
                b: 6,
                options: None,
                outputs: vec![8],
            },
            DataType::Uint8,
        );
        r.expect(8, &[2], &[1, 0]);
        r.run(
            Operation::Min {
                a: 0,
                b: 6,
                options: None,
                outputs: vec![9],
            },
            dtype,
        );
        r.expect(9, &[2], &[16_777_216, 16_777_219]);
        let (_, max) = bounds(dtype);
        r.integers(10, dtype, &[], &[max]);
        r.integers(11, dtype, &[], &[1]);
        r.run(
            Operation::Add {
                a: 10,
                b: 11,
                options: None,
                outputs: vec![12],
            },
            dtype,
        );
        r.expect(
            12,
            &[],
            &[if domain(dtype).1 { bounds(dtype).0 } else { 0 }],
        );
    }
    let mut r = Runtime::new();
    r.integers(0, DataType::Uint64, &[], &[u64::MAX as i128]);
    r.integers(1, DataType::Uint64, &[], &[2]);
    r.run(
        Operation::Div {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        },
        DataType::Uint64,
    );
    r.expect(2, &[], &[9_223_372_036_854_775_807]);
    r.run(
        Operation::Mul {
            a: 0,
            b: 0,
            options: None,
            outputs: vec![3],
        },
        DataType::Uint64,
    );
    r.expect(3, &[], &[1]);
}
#[test]
fn concat_empty_past_with_present_key() {
    let mut r = Runtime::new();
    r.floats(0, DataType::Float32, &[1, 3, 0, 64], &[]);
    r.floats(1, DataType::Float32, &[1, 3, 1, 64], &[0.0; 192]);
    r.run(
        Operation::Concat {
            inputs: vec![0, 1],
            axis: 2,
            options: None,
            outputs: vec![2],
        },
        DataType::Float32,
    );
    let value = r.env.get_host(2).unwrap();
    assert_eq!(value.shape, [1, 3, 1, 64]);
    assert_eq!(value.data.len(), 192);
}

#[test]
fn integer_pooling_and_nearest_resampling_preserve_samples() {
    let mut r = Runtime::new();
    let max = u64::MAX as i128;
    r.integers(
        0,
        DataType::Uint64,
        &[1, 1, 2, 2],
        &[max - 1, max - 2, 0, max],
    );
    r.run(
        Operation::MaxPool2d {
            input: 0,
            options: Some(MLPool2dOptions {
                window_dimensions: Some(vec![1, 2]),
                ..Default::default()
            }),
            outputs: vec![1],
        },
        DataType::Uint64,
    );
    r.expect(1, &[1, 1, 2, 1], &[max - 1, max]);
    r.run(
        Operation::GlobalMaxPool {
            input: 0,
            options: Some(MLPool2dOptions {
                window_dimensions: Some(vec![1, 1]),
                ..Default::default()
            }),
            outputs: vec![2],
        },
        DataType::Uint64,
    );
    r.expect(2, &[1, 1, 1, 1], &[max]);
    r.integers(3, DataType::Int8, &[1, 1, 1, 2], &[127, -128]);
    r.run(
        Operation::Resample2d {
            input: 3,
            options: Some(MLResample2dOptions {
                sizes: Some(vec![1, 4]),
                ..Default::default()
            }),
            outputs: vec![4],
        },
        DataType::Int8,
    );
    r.expect(4, &[1, 1, 1, 4], &[127, 127, -128, -128]);
}

#[test]
fn zeroed_wide_inputs_have_authoritative_integer_storage() {
    for dtype in [
        DataType::Int32,
        DataType::Uint32,
        DataType::Int64,
        DataType::Uint64,
    ] {
        let descriptors = HashMap::from([(
            "x".to_owned(),
            crate::graph::OperandDescriptor {
                data_type: dtype,
                shape: crate::graph::to_dimension_vector(&[2]),
                pending_permutation: Vec::new(),
            },
        )]);
        let input = super::super::interpreter::zeroed_inputs_from_descriptors(&descriptors)
            .pop()
            .unwrap();
        let mut r = Runtime::new();
        r.types.insert(0, dtype);
        r.env.insert_input(0, dtype, input).unwrap();
        r.run(
            Operation::Identity {
                input: 0,
                options: None,
                outputs: vec![1],
            },
            dtype,
        );
        r.expect(1, &[2], &[0, 0]);
    }
}

#[test]
fn integer_prelu_does_not_round_large_negative_values() {
    let mut r = Runtime::new();
    r.integers(0, DataType::Int64, &[2], &[-16_777_217, i64::MAX as i128]);
    r.integers(1, DataType::Int64, &[], &[-3]);
    r.run(
        Operation::Prelu {
            input: 0,
            slope: 1,
            options: None,
            outputs: vec![2],
        },
        DataType::Int64,
    );
    r.expect(2, &[2], &[50_331_651, i64::MAX as i128]);
}

#[test]
fn movement_preserves_integers_through_a_long_chain() {
    let mut r = Runtime::new();
    let dtype = DataType::Int64;
    r.integers(
        0,
        dtype,
        &[2, 2],
        &[16_777_217, -16_777_219, i64::MAX as i128, i64::MIN as i128],
    );
    r.run(
        Operation::Transpose {
            input: 0,
            options: None,
            outputs: vec![1],
        },
        dtype,
    );
    r.expect(
        1,
        &[2, 2],
        &[16_777_217, i64::MAX as i128, -16_777_219, i64::MIN as i128],
    );
    r.run(
        Operation::Reshape {
            input: 1,
            new_shape: vec![MLDimension::Static(4)],
            options: None,
            outputs: vec![2],
        },
        dtype,
    );
    r.run(
        Operation::Slice {
            input: 2,
            starts: vec![0],
            sizes: vec![MLDimension::Static(4)],
            options: Some(MLSliceOptions {
                strides: vec![2],
                ..Default::default()
            }),
            outputs: vec![3],
        },
        dtype,
    );
    r.expect(3, &[2], &[16_777_217, -16_777_219]);
    r.run(
        Operation::Unsqueeze {
            input: 3,
            options: Some(MLUnsqueezeOptions {
                axes: vec![0],
                ..Default::default()
            }),
            outputs: vec![4],
        },
        dtype,
    );
    r.run(
        Operation::Expand {
            input: 4,
            new_shape: vec![MLDimension::Static(2), MLDimension::Static(2)],
            options: None,
            outputs: vec![5],
        },
        dtype,
    );
    r.expect(
        5,
        &[2, 2],
        &[16_777_217, -16_777_219, 16_777_217, -16_777_219],
    );
    r.run(
        Operation::Split {
            input: 5,
            splits: vec![1, 1],
            split_equal_parts: None,
            options: Some(MLSplitOptions {
                axis: 0,
                ..Default::default()
            }),
            outputs: vec![6, 7],
        },
        dtype,
    );
    r.run(
        Operation::Concat {
            inputs: vec![7, 6],
            axis: 0,
            options: None,
            outputs: vec![8],
        },
        dtype,
    );
    r.run(
        Operation::Reverse {
            input: 8,
            options: None,
            outputs: vec![9],
        },
        dtype,
    );
    r.expect(
        9,
        &[2, 2],
        &[-16_777_219, 16_777_217, -16_777_219, 16_777_217],
    );
    r.run(
        Operation::Slice {
            input: 9,
            starts: vec![0, 0],
            sizes: vec![MLDimension::Static(1), MLDimension::Static(2)],
            options: None,
            outputs: vec![10],
        },
        dtype,
    );
    r.run(
        Operation::Squeeze {
            input: 10,
            options: None,
            outputs: vec![11],
        },
        dtype,
    );
    r.run(
        Operation::Tile {
            input: 11,
            repetitions: vec![2],
            options: None,
            outputs: vec![12],
        },
        dtype,
    );
    r.run(
        Operation::Abs {
            input: 12,
            options: None,
            outputs: vec![13],
        },
        dtype,
    );
    r.expect(13, &[4], &[16_777_219, 16_777_217, 16_777_219, 16_777_217]);
}
#[test]
fn gather_scatter_where_and_padding_keep_payloads() {
    let mut r = Runtime::new();
    let dtype = DataType::Uint64;
    r.integers(0, dtype, &[2], &[u64::MAX as i128 - 1, u64::MAX as i128]);
    r.integers(1, DataType::Int64, &[2], &[1, 0]);
    r.run(
        Operation::Gather {
            input: 0,
            indices: 1,
            batch_dimensions: None,
            options: None,
            outputs: vec![2],
        },
        dtype,
    );
    r.expect(2, &[2], &[u64::MAX as i128, u64::MAX as i128 - 1]);
    r.run(
        Operation::GatherElements {
            batch_dimensions: None,
            input: 0,
            indices: 1,
            options: None,
            outputs: vec![3],
        },
        dtype,
    );
    r.expect(3, &[2], &[u64::MAX as i128, u64::MAX as i128 - 1]);
    r.integers(4, DataType::Int64, &[2, 1], &[1, 0]);
    r.run(
        Operation::GatherND {
            input: 0,
            indices: 4,
            options: None,
            outputs: vec![5],
        },
        dtype,
    );
    r.expect(5, &[2], &[u64::MAX as i128, u64::MAX as i128 - 1]);
    r.run(
        Operation::ScatterElements {
            input: 0,
            indices: 1,
            updates: 0,
            options: None,
            outputs: vec![6],
        },
        dtype,
    );
    r.expect(6, &[2], &[u64::MAX as i128, u64::MAX as i128 - 1]);
    r.run(
        Operation::ScatterND {
            input: 0,
            indices: 4,
            updates: 0,
            options: None,
            outputs: vec![7],
        },
        dtype,
    );
    r.expect(7, &[2], &[u64::MAX as i128, u64::MAX as i128 - 1]);
    r.integers(8, DataType::Uint8, &[2], &[1, 0]);
    r.run(
        Operation::Where {
            condition: 8,
            true_value: 0,
            false_value: 7,
            options: None,
            outputs: vec![9],
        },
        dtype,
    );
    r.expect(9, &[2], &[u64::MAX as i128 - 1, u64::MAX as i128 - 1]);
    r.run(
        Operation::Pad {
            input: 9,
            beginning_padding: vec![1],
            ending_padding: vec![1],
            options: Some(MLPadOptions {
                value: Some(serde_json::json!("18446744073709551615")),
                ..Default::default()
            }),
            outputs: vec![10],
        },
        dtype,
    );
    r.expect(
        10,
        &[4],
        &[
            u64::MAX as i128,
            u64::MAX as i128 - 1,
            u64::MAX as i128 - 1,
            u64::MAX as i128,
        ],
    );
    r.floats(11, DataType::Float32, &[2], &[1.0, 2.0]);
    r.integers(12, DataType::Uint64, &[1], &[u64::MAX as i128]);
    r.run(
        Operation::GatherElements {
            batch_dimensions: None,
            input: 11,
            indices: 12,
            options: None,
            outputs: vec![13],
        },
        DataType::Float32,
    );
    assert_eq!(Array::load(&r.env, 13).unwrap().values[0].float(), 2.0);
}
#[test]
fn casts_preserve_bits_and_signedness() {
    let mut r = Runtime::new();
    r.integers(0, DataType::Int64, &[3], &[-1, 16_777_217, 4_294_967_297]);
    r.run(
        Operation::Cast {
            input: 0,
            data_type: crate::operator_enums::MLOperandDataType::Uint32,
            options: None,
            outputs: vec![1],
        },
        DataType::Uint32,
    );
    r.expect(1, &[3], &[4_294_967_295, 16_777_217, 1]);
    r.run(
        Operation::Cast {
            input: 0,
            data_type: crate::operator_enums::MLOperandDataType::Uint64,
            options: None,
            outputs: vec![2],
        },
        DataType::Uint64,
    );
    r.expect(2, &[3], &[u64::MAX as i128, 16_777_217, 4_294_967_297]);
    r.run(
        Operation::Cast {
            input: 2,
            data_type: crate::operator_enums::MLOperandDataType::Int64,
            options: None,
            outputs: vec![3],
        },
        DataType::Int64,
    );
    r.expect(3, &[3], &[-1, 16_777_217, 4_294_967_297]);
    r.floats(4, DataType::Float32, &[2], &[-1.75, 2.75]);
    r.run(
        Operation::Cast {
            input: 4,
            data_type: crate::operator_enums::MLOperandDataType::Int32,
            options: None,
            outputs: vec![5],
        },
        DataType::Int32,
    );
    r.expect(5, &[2], &[-1, 2]);
}
#[test]
fn reductions_cumulative_sum_and_arg_indices_are_exact() {
    let mut r = Runtime::new();
    let dtype = DataType::Int64;
    r.integers(
        0,
        dtype,
        &[2, 2],
        &[16_777_216, 16_777_217, -16_777_217, -16_777_219],
    );
    r.run(
        Operation::ReduceSum {
            input: 0,
            options: Some(MLReduceOptions {
                axes: Some(vec![1]),
                ..Default::default()
            }),
            outputs: vec![1],
        },
        dtype,
    );
    r.expect(1, &[2], &[33_554_433, -33_554_436]);
    r.run(
        Operation::CumulativeSum {
            input: 0,
            axis: 1,
            options: None,
            outputs: vec![2],
        },
        dtype,
    );
    r.expect(
        2,
        &[2, 2],
        &[16_777_216, 33_554_433, -16_777_217, -33_554_436],
    );
    r.run(
        Operation::ArgMax {
            input: 0,
            axis: 1,
            options: None,
            outputs: vec![3],
        },
        DataType::Int64,
    );
    r.expect(3, &[2], &[1, 0]);
    r.run(
        Operation::ReduceMax {
            input: 0,
            options: Some(MLReduceOptions {
                axes: Some(vec![1]),
                keep_dimensions: true,
                ..Default::default()
            }),
            outputs: vec![4],
        },
        dtype,
    );
    r.expect(4, &[2, 1], &[16_777_217, -16_777_217]);
}
#[test]
fn quantization_uses_exact_large_zero_points_and_differences() {
    let mut r = Runtime::new();
    r.integers(0, DataType::Int32, &[2], &[2_147_483_646, 2_147_483_647]);
    r.integers(1, DataType::Int32, &[], &[2_147_483_645]);
    r.floats(2, DataType::Float32, &[], &[0.5]);
    r.run(
        Operation::DequantizeLinear {
            input: 0,
            scale: 2,
            zero_point: Some(1),
            options: None,
            outputs: vec![3],
        },
        DataType::Float32,
    );
    assert_eq!(
        Array::load(&r.env, 3)
            .unwrap()
            .values
            .iter()
            .map(|v| v.float())
            .collect::<Vec<_>>(),
        [0.5, 1.0]
    );
    r.run(
        Operation::QuantizeLinear {
            input: 3,
            scale: 2,
            zero_point: Some(1),
            options: None,
            outputs: vec![4],
        },
        DataType::Int32,
    );
    r.expect(4, &[2], &[2_147_483_646, 2_147_483_647]);
}
#[test]
fn half_device_outputs_are_rounded_before_the_next_operation() {
    let mut r = Runtime::new();
    let input = half::f16::from_bits(0x3c01).to_f32();
    r.floats(0, DataType::Float16, &[1, 1], &[input]);
    r.run(
        Operation::Matmul {
            a: 0,
            b: 0,
            options: None,
            outputs: vec![1],
        },
        DataType::Float16,
    );
    let expected = half::f16::from_bits(0x3c02).to_f32();
    assert_eq!(r.env.get_host(1).unwrap().data, [expected]);
    r.run(
        Operation::Cast {
            input: 1,
            data_type: crate::operator_enums::MLOperandDataType::Float32,
            options: None,
            outputs: vec![2],
        },
        DataType::Float32,
    );
    assert_eq!(r.env.get_host(2).unwrap().data, [expected]);
}

#[test]
fn native_matmul_broadcasts_a_lower_rank_rhs() {
    let mut r = Runtime::new();
    r.floats(0, DataType::Float32, &[2, 3, 4], &[1.0; 24]);
    r.floats(1, DataType::Float32, &[4, 1], &[2.0; 4]);
    r.run(
        Operation::Matmul {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        },
        DataType::Float32,
    );
    let out = r.env.get_host(2).unwrap();
    assert_eq!(out.shape, [2, 3, 1]);
    assert_eq!(out.data, [8.0; 6]);
}

#[test]
fn shared_padding_preserves_non_finite_float_options() {
    for dtype in [DataType::Float32, DataType::Float16] {
        for (encoded, expected) in [
            ("NaN", f32::NAN),
            ("Infinity", f32::INFINITY),
            ("-Infinity", f32::NEG_INFINITY),
        ] {
            let mut r = Runtime::new();
            r.floats(0, dtype, &[1], &[2.0]);
            r.run(
                Operation::Pad {
                    input: 0,
                    beginning_padding: vec![1],
                    ending_padding: vec![1],
                    options: Some(MLPadOptions {
                        value: Some(serde_json::json!(encoded)),
                        ..Default::default()
                    }),
                    outputs: vec![1],
                },
                dtype,
            );
            let out = r.env.get_host(1).unwrap();
            assert_eq!(out.shape, [3]);
            assert_eq!(out.data[1], 2.0);
            for actual in [out.data[0], out.data[2]] {
                if expected.is_nan() {
                    assert!(actual.is_nan());
                } else {
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}

#[test]
fn float_predicates_route_to_typed_results_then_shared_movement() {
    let mut r = Runtime::new();
    r.floats(0, DataType::Float32, &[2], &[f32::NAN, 2.0]);
    r.run(
        Operation::IsNaN {
            input: 0,
            options: None,
            outputs: vec![1],
        },
        DataType::Uint8,
    );
    r.expect(1, &[2], &[1, 0]);
    r.run(
        Operation::Reverse {
            input: 1,
            options: Some(MLReverseOptions {
                axes: Some(vec![0]),
                ..Default::default()
            }),
            outputs: vec![2],
        },
        DataType::Uint8,
    );
    r.expect(2, &[2], &[0, 1]);
}
#[test]
fn host_half_results_are_rounded_before_float32_consumers() {
    let mut r = Runtime::new();
    r.floats(0, DataType::Float16, &[1], &[0.5]);
    r.run(
        Operation::Erf {
            input: 0,
            options: None,
            outputs: vec![1],
        },
        DataType::Float16,
    );
    r.run(
        Operation::Cast {
            input: 1,
            data_type: crate::operator_enums::MLOperandDataType::Float32,
            options: None,
            outputs: vec![2],
        },
        DataType::Float32,
    );
    // erf(0.5) rounds to this binary16 value; the float32 consumer must see that rounding.
    assert_eq!(r.env.get_host(2).unwrap().data, [0.520_507_8]);
}

#[test]
fn integer_movement_supports_ranks_above_ndarray_float_limit() {
    for rank in 7..=8 {
        let mut r = Runtime::new();
        let shape = vec![1; rank];
        r.integers(0, DataType::Int64, &shape, &[16_777_217]);
        r.run(
            Operation::Transpose {
                input: 0,
                options: None,
                outputs: vec![1],
            },
            DataType::Int64,
        );
        r.expect(1, &shape, &[16_777_217]);
    }
}

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

#![cfg(feature = "burn-runtime-cpu")]

use rustnn::burn::{BurnGraphPlan, ConstantSlot, IOBinding};
use rustnn::executors::burn::run_burn_cpu_with_inputs;
use rustnn::operator_options::{MLDimension, MLSliceOptions};
use rustnn::{DataType, Operation};
use std::collections::HashMap;

#[test]
fn strided_integer_slice_preserves_values_and_uses_window_extents() {
    let data = (0..8i32)
        .flat_map(|i| (16_777_217 + i).to_le_bytes())
        .collect();
    let plan = slice_plan(DataType::Int32, data);
    let output = run_burn_cpu_with_inputs(&plan.serialize().unwrap(), vec![]).unwrap();
    assert_eq!(output[0].shape, vec![2]);
    assert_eq!(
        output[0].int64_data.as_deref(),
        Some([16_777_217i64, 16_777_219].as_slice())
    );
}

#[test]
fn integer_and_float_strided_slices_have_the_same_shape() {
    let data = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let plan = slice_plan(DataType::Float32, data);
    let output = run_burn_cpu_with_inputs(&plan.serialize().unwrap(), vec![]).unwrap();
    assert_eq!(output[0].shape, vec![2]);
    assert_eq!(output[0].data, vec![0.0, 2.0]);
}

fn slice_plan(dtype: DataType, data: Vec<u8>) -> BurnGraphPlan {
    BurnGraphPlan::new(
        vec![],
        vec![IOBinding {
            name: "y".into(),
            operand_id: 1,
            data_type: dtype,
            shape: vec![2],
        }],
        vec![ConstantSlot {
            operand_id: 0,
            data_type: dtype,
            shape: vec![8],
            data,
        }],
        vec![Operation::Slice {
            input: 0,
            starts: vec![0],
            sizes: vec![MLDimension::Static(4)],
            options: Some(MLSliceOptions {
                strides: vec![2],
                ..Default::default()
            }),
            outputs: vec![1],
        }],
        HashMap::from([(0, dtype), (1, dtype)]),
    )
}

#[test]
fn outputless_plan_operation_returns_an_error() {
    let plan = BurnGraphPlan::new(
        vec![],
        vec![],
        vec![],
        vec![Operation::Add {
            a: 0,
            b: 0,
            options: None,
            outputs: vec![],
        }],
        HashMap::new(),
    );
    assert!(run_burn_cpu_with_inputs(&plan.serialize().unwrap(), vec![]).is_err());
}

#[test]
fn standalone_integer_imports_are_validated_before_selection() {
    use rustnn::executors::burn::BurnInput;
    // Put the invalid value outside the selected slice: validation must check
    // the complete imported operand rather than just the values a kernel reads.
    for (dtype, signed, invalid) in [
        (DataType::Int4, true, 8u64),
        (DataType::Int8, true, 128),
        (DataType::Int32, true, i32::MAX as u64 + 1),
        (DataType::Uint4, false, 16),
        (DataType::Uint8, false, 256),
        (DataType::Uint32, false, u64::from(u32::MAX) + 1),
    ] {
        let mut plan = slice_plan(dtype, vec![]);
        plan.constants.clear();
        plan.inputs.push(IOBinding {
            name: "x".into(),
            operand_id: 0,
            data_type: dtype,
            shape: vec![8],
        });
        let bytes = plan.serialize().unwrap();
        let mut input = BurnInput {
            name: "x".into(),
            shape: vec![8],
            data: vec![0.; 8],
            int64_data: signed.then(|| vec![0; 8]),
            uint64_data: (!signed).then(|| vec![0; 8]),
        };
        if signed {
            input.int64_data.as_mut().unwrap()[7] = invalid as i64;
        } else {
            input.uint64_data.as_mut().unwrap()[7] = invalid;
        }
        assert!(
            run_burn_cpu_with_inputs(&bytes, vec![input.clone()]).is_err(),
            "{dtype:?}"
        );
        if signed {
            input.int64_data.as_mut().unwrap()[7] = 0;
        } else {
            input.uint64_data.as_mut().unwrap()[7] = 0;
        }
        let output = run_burn_cpu_with_inputs(&bytes, vec![input]).unwrap();
        assert_eq!(output[0].shape, [2]);
        assert_eq!(output[0].data, [0., 0.]);
        if matches!(
            dtype,
            DataType::Int4 | DataType::Int8 | DataType::Uint4 | DataType::Uint8
        ) {
            for invalid in [0.5, -0.5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut data = vec![0.; 8];
                data[7] = invalid;
                let input = BurnInput {
                    name: "x".into(),
                    shape: vec![8],
                    data,
                    int64_data: None,
                    uint64_data: None,
                };
                assert!(
                    run_burn_cpu_with_inputs(&bytes, vec![input]).is_err(),
                    "{dtype:?}: {invalid}"
                );
            }
        }
    }
}

fn float_plan(
    constants: Vec<(Vec<u32>, Vec<f32>)>,
    op: Operation,
    outputs: Vec<(u32, Vec<i64>)>,
) -> BurnGraphPlan {
    let mut types = HashMap::new();
    let constants = constants
        .into_iter()
        .enumerate()
        .map(|(id, (shape, data))| {
            types.insert(id as u32, DataType::Float32);
            ConstantSlot {
                operand_id: id as u32,
                data_type: DataType::Float32,
                shape,
                data: data.into_iter().flat_map(f32::to_le_bytes).collect(),
            }
        })
        .collect();
    let outputs = outputs
        .into_iter()
        .map(|(id, shape)| {
            types.insert(id, DataType::Float32);
            IOBinding {
                name: format!("y{id}"),
                operand_id: id,
                data_type: DataType::Float32,
                shape,
            }
        })
        .collect();
    BurnGraphPlan::new(vec![], outputs, constants, vec![op], types)
}

#[test]
fn normalization_retains_small_variance_at_large_offsets() {
    use rustnn::operator_options::{MLInstanceNormalizationOptions, MLLayerNormalizationOptions};
    for values in [
        [10000.0f32, 10001.0],
        [1000.0, 1001.0],
        [f32::MAX, f32::MAX],
    ] {
        let mut constants = vec![
            (vec![1, 2], values.to_vec()),
            (vec![2], vec![2., 3.]),
            (vec![2], vec![4., 5.]),
        ];
        let op = Operation::LayerNormalization {
            input: 0,
            options: Some(MLLayerNormalizationOptions {
                scale: Some(1),
                bias: Some(2),
                ..Default::default()
            }),
            outputs: vec![3],
        };
        let plan = float_plan(constants.clone(), op, vec![(3, vec![1, 2])]);
        let result = run_burn_cpu_with_inputs(&plan.serialize().unwrap(), vec![]).unwrap();
        let mean = (f64::from(values[0]) + f64::from(values[1])) / 2.;
        let variance = values
            .iter()
            .map(|&x| (f64::from(x) - mean).powi(2))
            .sum::<f64>()
            / 2.;
        for (i, &x) in values.iter().enumerate() {
            let expected = ((f64::from(x) - mean) / (variance + 1e-5).sqrt() * [2., 3.][i]
                + [4., 5.][i]) as f32;
            assert!(
                (result[0].data[i] - expected).abs() < 1e-5,
                "{:?}",
                result[0].data
            );
        }
        for layout in ["nchw", "nhwc"] {
            constants[0].0 = if layout == "nchw" {
                vec![1, 1, 1, 2]
            } else {
                vec![1, 1, 2, 1]
            };
            let op = Operation::InstanceNormalization {
                input: 0,
                options: Some(MLInstanceNormalizationOptions {
                    layout: layout.into(),
                    ..Default::default()
                }),
                outputs: vec![3],
            };
            let shape = constants[0].0.iter().map(|&d| i64::from(d)).collect();
            let plan = float_plan(constants.clone(), op, vec![(3, shape)]);
            let result = run_burn_cpu_with_inputs(&plan.serialize().unwrap(), vec![]).unwrap();
            for (i, &x) in values.iter().enumerate() {
                let expected = ((f64::from(x) - mean) / (variance + 1e-5).sqrt()) as f32;
                assert!(
                    (result[0].data[i] - expected).abs() < 1e-5,
                    "{layout}: {:?}",
                    result[0].data
                );
            }
        }
    }
}

#[test]
fn bidirectional_gru_selects_initial_state_for_each_direction_and_batch() {
    use rustnn::operator_options::MLGruOptions;
    for layout in ["zrn", "rzn"] {
        for reset_after in [false, true] {
            for return_sequence in [false, true] {
                let constants = vec![
                    (vec![2, 2, 1], vec![0.; 4]),
                    (vec![2, 3, 1], vec![0.; 6]),
                    (vec![2, 3, 1], vec![0.; 6]),
                    (vec![2, 2, 1], vec![2., 4., 6., 8.]),
                ];
                let outputs = if return_sequence { vec![4, 5] } else { vec![4] };
                let op = Operation::Gru {
                    input: 0,
                    weight: 1,
                    recurrence: 2,
                    steps: 2,
                    hidden_size: 1,
                    options: Some(MLGruOptions {
                        initial_hidden_state: Some(3),
                        direction: "both".into(),
                        layout: layout.into(),
                        reset_after,
                        return_sequence,
                        ..Default::default()
                    }),
                    outputs,
                };
                let mut bindings = vec![(4, vec![2, 2, 1])];
                if return_sequence {
                    bindings.push((5, vec![2, 2, 2, 1]));
                }
                let plan = float_plan(constants, op, bindings);
                let result = run_burn_cpu_with_inputs(&plan.serialize().unwrap(), vec![]).unwrap();
                assert_eq!(result[0].data, vec![0.5, 1., 1.5, 2.]);
                if return_sequence {
                    assert_eq!(result[1].data, vec![1., 2., 1.5, 2., 0.5, 1., 3., 4.]);
                }
            }
        }
    }
}

#[test]
fn public_context_strided_slice_and_repeated_dispatch() {
    use rustnn::mlcontext::{
        Backend, MLContext, MLContextOptions, MLGraphBuilder, MLNamedOperands, MLNamedTensors,
        MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType as T;
    for dtype in [
        T::Float32,
        T::Float16,
        T::Int32,
        T::Int64,
        T::Uint64,
        T::Uint8,
    ] {
        for constant in [false, true] {
            let encode = |offset: i32| {
                (0..8)
                    .flat_map(|x| {
                        let x = x + offset;
                        match dtype {
                            T::Float32 => (x as f32).to_le_bytes().to_vec(),
                            T::Float16 => half::f16::from_f32(x as f32)
                                .to_bits()
                                .to_le_bytes()
                                .to_vec(),
                            T::Int32 => x.to_le_bytes().to_vec(),
                            T::Int64 => i64::from(x).to_le_bytes().to_vec(),
                            T::Uint64 => (x as u64).to_le_bytes().to_vec(),
                            T::Uint8 => vec![x as u8],
                            _ => unreachable!(),
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let mut context = MLContext::create(
                &MLContextOptions::new(MLPowerPreference::Default, false)
                    .with_rustnn_backend_hint(Backend::BurnCpu),
            )
            .unwrap();
            let descriptor = MLOperandDescriptor::new(dtype, vec![2, 4]);
            let mut builder = MLGraphBuilder::new(&mut context).unwrap();
            let x = if constant {
                builder.constant_from_bytes(&descriptor, encode(0)).unwrap()
            } else {
                builder.input("x", &descriptor).unwrap()
            };
            let y = builder
                .slice_with_options(
                    x,
                    &[0, 1],
                    &[MLDimension::Static(2), MLDimension::Static(3)],
                    MLSliceOptions {
                        strides: vec![1, 2],
                        ..Default::default()
                    },
                )
                .unwrap();
            let mut graph = builder.build(&MLNamedOperands::from([("y", y)])).unwrap();
            let input = context
                .create_tensor(&MLTensorDescriptor::new(dtype, vec![2, 4]).to_writable())
                .unwrap();
            let output = context
                .create_tensor(&MLTensorDescriptor::new(dtype, vec![2, 2]).to_readable())
                .unwrap();
            for offset in [0, 8] {
                context.write_tensor(&input, &encode(offset)).unwrap();
                let inputs = if constant {
                    MLNamedTensors::new()
                } else {
                    MLNamedTensors::from([("x", &input)])
                };
                context
                    .dispatch(&mut graph, &inputs, &MLNamedTensors::from([("y", &output)]))
                    .unwrap();
                let mut actual = vec![0u8; encode(0).len() / 2];
                context.read_tensor(&output, &mut actual).unwrap();
                let bytes = encode(if constant { 0 } else { offset });
                let size = bytes.len() / 8;
                let expected = [1usize, 3, 5, 7]
                    .into_iter()
                    .flat_map(|i| bytes[i * size..(i + 1) * size].iter().copied())
                    .collect::<Vec<_>>();
                assert_eq!(
                    actual, expected,
                    "{dtype:?} constant={constant} offset={offset}"
                );
            }
        }
    }
}

#[test]
fn strided_slice_rejects_invalid_windows_and_zero_stride() {
    for dtype in [DataType::Float32, DataType::Int32] {
        let data = (0..8)
            .flat_map(|i| {
                if dtype == DataType::Float32 {
                    (i as f32).to_le_bytes()
                } else {
                    i32::to_le_bytes(i)
                }
            })
            .collect();
        let good = slice_plan(dtype, data);
        for (starts, sizes, strides) in [
            (vec![7], vec![2], vec![1]),
            (vec![0], vec![4], vec![0]),
            (vec![0], vec![4], vec![1, 2]),
        ] {
            let mut plan = good.clone();
            plan.operations[0] = Operation::Slice {
                input: 0,
                starts,
                sizes: sizes.into_iter().map(MLDimension::Static).collect(),
                options: Some(MLSliceOptions {
                    strides,
                    ..Default::default()
                }),
                outputs: vec![1],
            };
            assert!(run_burn_cpu_with_inputs(&plan.serialize().unwrap(), vec![]).is_err());
        }
    }
}

#[test]
fn public_context_normalization_is_stable_across_repeated_dispatches() {
    use rustnn::mlcontext::{
        Backend, MLContext, MLContextOptions, MLGraphBuilder, MLNamedOperands, MLNamedTensors,
        MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType as T;
    for dtype in [T::Float32, T::Float16] {
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_rustnn_backend_hint(Backend::BurnCpu),
        )
        .unwrap();
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let x = builder
            .input("x", &MLOperandDescriptor::new(dtype, vec![1, 1, 1, 2]))
            .unwrap();
        let layer = builder.layer_normalization(x).unwrap();
        let instance = builder.instance_normalization(x).unwrap();
        let mut graph = builder
            .build(&MLNamedOperands::from([
                ("layer", layer),
                ("instance", instance),
            ]))
            .unwrap();
        let desc = MLTensorDescriptor::new(dtype, vec![1, 1, 1, 2]);
        let input = context.create_tensor(&desc.clone().to_writable()).unwrap();
        let layer = context.create_tensor(&desc.clone().to_readable()).unwrap();
        let instance = context.create_tensor(&desc.to_readable()).unwrap();
        let pairs = if dtype == T::Float32 {
            [[10000., 10001.], [1e20, 1e20]]
        } else {
            [[1000., 1001.], [65504., 65504.]]
        };
        for pair in pairs {
            let data = pair
                .into_iter()
                .flat_map(|x: f32| {
                    if dtype == T::Float32 {
                        x.to_le_bytes().to_vec()
                    } else {
                        half::f16::from_f32(x).to_bits().to_le_bytes().to_vec()
                    }
                })
                .collect::<Vec<_>>();
            context.write_tensor(&input, &data).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("x", &input)]),
                    &MLNamedTensors::from([("layer", &layer), ("instance", &instance)]),
                )
                .unwrap();
            for tensor in [&layer, &instance] {
                let mut bytes = vec![0u8; data.len()];
                context.read_tensor(tensor, &mut bytes).unwrap();
                let actual = if dtype == T::Float32 {
                    bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect::<Vec<_>>()
                } else {
                    bytes
                        .chunks_exact(2)
                        .map(|b| {
                            half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()
                        })
                        .collect()
                };
                let expected = if pair[0] == pair[1] {
                    [0., 0.]
                } else {
                    [-1., 1.]
                };
                for (actual, expected) in actual.into_iter().zip(expected) {
                    assert!(
                        (actual - expected).abs() < 0.001,
                        "{dtype:?}: {actual} versus {expected}"
                    );
                }
            }
        }
    }
}

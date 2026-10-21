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

//! Float16 arithmetic helpers matching WPT reference semantics.

use crate::graph::DataType;

#[inline]
pub fn round_f16(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

#[inline]
pub fn round_f16_slice(data: &mut [f32]) {
    for v in data {
        *v = round_f16(*v);
    }
}

#[inline]
pub fn f16_add(a: f32, b: f32) -> f32 {
    (half::f16::from_f32(a) + half::f16::from_f32(b)).to_f32()
}

#[inline]
pub fn f16_sub(a: f32, b: f32) -> f32 {
    (half::f16::from_f32(a) - half::f16::from_f32(b)).to_f32()
}

#[inline]
pub fn f16_mul(a: f32, b: f32) -> f32 {
    (half::f16::from_f32(a) * half::f16::from_f32(b)).to_f32()
}

#[inline]
pub fn f16_div(a: f32, b: f32) -> f32 {
    (half::f16::from_f32(a) / half::f16::from_f32(b)).to_f32()
}

#[inline]
pub fn f16_neg(a: f32) -> f32 {
    (-half::f16::from_f32(a)).to_f32()
}

#[inline]
pub fn f16_sqrt(a: f32) -> f32 {
    round_f16(half::f16::from_f32(a).to_f32().sqrt())
}

pub fn use_f16_arithmetic(
    operand_types: &std::collections::HashMap<u32, DataType>,
    inputs: &[u32],
    output: u32,
) -> bool {
    inputs
        .iter()
        .chain(std::iter::once(&output))
        .any(|id| operand_types.get(id) == Some(&DataType::Float16))
}

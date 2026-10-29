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

#![cfg(feature = "burn-cpu-runtime")]

use std::collections::HashMap;

use burn_ndarray::NdArray;

use crate::error::GraphError;
use crate::graph::OperandDescriptor;

use super::interpreter::{
    BurnInput, BurnOutput, BurnOutputWithData, execute_plan, plan_output_metadata,
    zeroed_inputs_from_descriptors,
};

type CpuBackend = NdArray<f32>;

/// Execute a serialized plan on the host CPU.
pub fn run_burn_cpu_with_inputs(
    plan_bytes: &[u8],
    inputs: Vec<BurnInput>,
) -> Result<Vec<BurnOutputWithData>, GraphError> {
    execute_plan::<CpuBackend>(plan_bytes, inputs)
}

/// Execute with zeroed inputs and return output metadata.
pub fn run_burn_cpu_zeroed(
    plan_bytes: &[u8],
    inputs: &HashMap<String, OperandDescriptor>,
) -> Result<Vec<BurnOutput>, GraphError> {
    let zeroed = zeroed_inputs_from_descriptors(inputs);
    let _ = run_burn_cpu_with_inputs(plan_bytes, zeroed)?;
    plan_output_metadata(plan_bytes)
}

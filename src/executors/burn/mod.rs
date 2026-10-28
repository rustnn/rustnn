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

//! Burn NdArray CPU executor and host tensor helpers.
#![cfg(feature = "burn-runtime-cpu")]

#[cfg(feature = "burn-runtime-cpu")]
mod cpu;
mod device_ops;
mod execute;
mod f16;
mod float_ops;
mod host_array;
mod interpreter;
mod options;
mod outputs;
mod prepared;
mod rnn;
mod runtime_tensor;
mod shape;
mod tensor_env;
mod typed;
mod value;

pub use cpu::{run_burn_cpu_with_inputs, run_burn_cpu_zeroed};
pub(crate) use interpreter::bytes_to_host_values;
pub use interpreter::{BurnInput, BurnOutput, BurnOutputWithData, execute_plan};
pub(crate) use prepared::PreparedPlan;

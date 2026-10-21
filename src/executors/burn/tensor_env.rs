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

//! Operand storage and read-only access for executing kernels.
use super::host_array::HostArray;
use super::runtime_tensor::matmul_broadcast_operand_shapes;
pub(super) use super::runtime_tensor::{DeviceBinaryOp, RuntimeTensor, UnaryDeviceOp};
use crate::{error::GraphError, graph::DataType};
use burn_tensor::backend::Backend;
use std::collections::HashMap;
fn burn_err(reason: String) -> GraphError {
    GraphError::BurnRuntimeFailed { reason }
}

/// Execution environment with one authoritative value per operand.
#[derive(Debug)]
pub struct TensorEnv<B: Backend> {
    pub device: B::Device,
    values: HashMap<u32, super::value::TensorValue<B>>,
}
impl<B: Backend> TensorEnv<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            device,
            values: HashMap::new(),
        }
    }
    pub fn remove(&mut self, id: u32) {
        self.values.remove(&id);
    }
    pub fn len(&self) -> usize {
        self.values.len()
    }
    pub fn retain(&mut self, keep: impl Fn(&u32) -> bool) {
        self.values.retain(|id, _| keep(id));
    }
    pub fn value(&self, id: u32) -> Result<&super::value::TensorValue<B>, GraphError> {
        self.values
            .get(&id)
            .ok_or_else(|| burn_err(format!("operand {id} is not available before use")))
    }
    pub fn insert_value(&mut self, id: u32, value: super::value::TensorValue<B>) {
        self.values.insert(id, value);
    }
    pub fn shape(&self, id: u32) -> Result<&[usize], GraphError> {
        Ok(self.value(id)?.shape())
    }
    #[cfg(test)]
    pub fn insert_host(
        &mut self,
        id: u32,
        dtype: DataType,
        host: HostArray,
    ) -> Result<(), GraphError> {
        let value = super::value::TensorValue::from_host(dtype, host, &self.device)?;
        self.insert_value(id, value);
        Ok(())
    }
    pub fn insert_input(
        &mut self,
        id: u32,
        dtype: DataType,
        input: super::interpreter::BurnInput,
    ) -> Result<(), GraphError> {
        let value = super::value::TensorValue::from_input(dtype, input, &self.device)?;
        self.insert_value(id, value);
        Ok(())
    }
    pub fn get(&self, id: u32) -> Result<&RuntimeTensor<B>, GraphError> {
        self.value(id)?.float()
    }
    pub fn get_host(&self, id: u32) -> Result<HostArray, GraphError> {
        self.get(id)?.to_host_array()
    }
    pub fn binary_host<F>(&self, a: u32, b: u32, f: F) -> Result<HostArray, GraphError>
    where
        F: FnOnce(&HostArray, &HostArray) -> Result<HostArray, GraphError>,
    {
        let a_arr = self.get_host(a)?;
        let b_arr = self.get_host(b)?;
        f(&a_arr, &b_arr)
    }

    /// Elementwise binary op on device tensors when shapes match (avoids host round-trip).
    pub fn binary_broadcast(
        &self,
        a_id: u32,
        b_id: u32,
        op: DeviceBinaryOp,
    ) -> Result<RuntimeTensor<B>, GraphError> {
        let a = self.get(a_id)?;
        let b = self.get(b_id)?;
        a.binary_broadcast(b, op, &self.device)
    }

    pub fn unary(&self, input_id: u32, op: UnaryDeviceOp) -> Result<RuntimeTensor<B>, GraphError> {
        self.get(input_id)?.unary(op)
    }

    pub fn matmul(&self, a_id: u32, b_id: u32) -> Result<RuntimeTensor<B>, GraphError> {
        let a = self.get(a_id)?;
        let b = self.get(b_id)?;
        let (a_shape, b_shape) = matmul_broadcast_operand_shapes(a.shape(), b.shape())?;
        let a = a.broadcast_to_shape(&a_shape, &self.device)?;
        let b = b.broadcast_to_shape(&b_shape, &self.device)?;
        a.matmul(&b)
    }
}

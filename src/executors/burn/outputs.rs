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

//! One checked publication boundary for every operation, independent of kernel storage.
use super::{
    host_array::HostArray,
    tensor_env::{RuntimeTensor, TensorEnv},
    value::TensorValue,
};
use crate::{burn::IOBinding, error::GraphError, graph::DataType, operators::Operation};
use burn_tensor::backend::Backend;
use std::{collections::HashMap, ops::Deref};

fn error(reason: impl Into<String>) -> GraphError {
    GraphError::BurnRuntimeFailed {
        reason: reason.into(),
    }
}

/// Read-only access to inputs, with staged, exactly-once output publication.
/// Results become visible together only after `finish` verifies completeness.
pub(super) struct OperationOutputs<'a, B: Backend> {
    env: &'a mut TensorEnv<B>,
    declared: &'a [u32],
    types: &'a HashMap<u32, DataType>,
    bindings: &'a [IOBinding],
    pending: Vec<(u32, TensorValue<B>)>,
}
impl<'a, B: Backend> OperationOutputs<'a, B> {
    pub fn new(
        env: &'a mut TensorEnv<B>,
        op: &'a Operation,
        types: &'a HashMap<u32, DataType>,
        bindings: &'a [IOBinding],
    ) -> Result<Self, GraphError> {
        let declared = op.outputs();
        if declared.is_empty() {
            return Err(error("operation has no output operand"));
        }
        for (i, id) in declared.iter().enumerate() {
            if declared[..i].contains(id) {
                return Err(error(format!("duplicate output operand {id}")));
            }
            if !types.contains_key(id) {
                return Err(error(format!("missing datatype for output operand {id}")));
            }
        }
        Ok(Self {
            env,
            declared,
            types,
            bindings,
            pending: Vec::with_capacity(declared.len()),
        })
    }
    pub fn dtype(&self, id: u32) -> Result<DataType, GraphError> {
        if !self.declared.contains(&id) {
            return Err(error(format!(
                "operand {id} is not an output of this operation"
            )));
        }
        self.types
            .get(&id)
            .copied()
            .ok_or_else(|| error(format!("missing datatype for output operand {id}")))
    }
    pub fn write(&mut self, id: u32, value: TensorValue<B>) -> Result<(), GraphError> {
        if self.dtype(id)? != value.dtype() {
            return Err(error(format!("output operand {id} has the wrong datatype")));
        }
        if self.pending.iter().any(|(written, _)| *written == id) {
            return Err(error(format!("output operand {id} was written twice")));
        }
        for binding in self.bindings.iter().filter(|b| b.operand_id == id) {
            if !super::shape::matches_binding(value.shape(), &binding.shape) {
                return Err(error(format!(
                    "output '{}' does not match its declared shape",
                    binding.name
                )));
            }
        }
        // TensorValue constructors validate shape, storage length and datatype range.
        self.pending.push((id, value));
        Ok(())
    }
    pub fn write_tensor(&mut self, id: u32, tensor: RuntimeTensor<B>) -> Result<(), GraphError> {
        let value = TensorValue::from_float(self.dtype(id)?, tensor, &self.env.device)?;
        self.write(id, value)
    }
    pub fn write_host(&mut self, id: u32, host: HostArray) -> Result<(), GraphError> {
        let value = TensorValue::from_host(self.dtype(id)?, host, &self.env.device)?;
        self.write(id, value)
    }
    pub fn finish(self) -> Result<(), GraphError> {
        if self.pending.len() != self.declared.len() {
            return Err(error("operation did not write every declared output"));
        }
        for (id, value) in self.pending {
            self.env.insert_value(id, value);
        }
        Ok(())
    }
}
impl<B: Backend> Deref for OperationOutputs<'_, B> {
    type Target = TensorEnv<B>;
    fn deref(&self) -> &Self::Target {
        self.env
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    type Cpu = NdArray<f32>;
    fn split(outputs: Vec<u32>) -> Operation {
        Operation::Split {
            input: 0,
            splits: vec![1, 1],
            split_equal_parts: None,
            options: None,
            outputs,
        }
    }
    fn host() -> HostArray {
        HostArray::new(vec![1], vec![1.0003]).unwrap()
    }
    #[test]
    fn incomplete_outputs_are_not_published() {
        let mut env = TensorEnv::<Cpu>::new(Default::default());
        let op = split(vec![1, 2]);
        let types = HashMap::from([(1, DataType::Float32), (2, DataType::Float32)]);
        let mut out = OperationOutputs::new(&mut env, &op, &types, &[]).unwrap();
        out.write_host(1, host()).unwrap();
        assert!(out.value(1).is_err());
        assert!(out.finish().is_err());
        assert_eq!(env.len(), 0);
    }
    #[test]
    fn outputs_reject_duplicate_foreign_and_mistyped_writes() {
        let mut env = TensorEnv::<Cpu>::new(Default::default());
        let op = split(vec![1, 2]);
        let types = HashMap::from([(1, DataType::Float32), (2, DataType::Float32)]);
        let mut out = OperationOutputs::new(&mut env, &op, &types, &[]).unwrap();
        assert!(out.write_host(3, host()).is_err());
        assert!(
            out.write(
                1,
                TensorValue::from_signed(DataType::Int32, vec![1], vec![1]).unwrap()
            )
            .is_err()
        );
        out.write_host(1, host()).unwrap();
        assert!(out.write_host(1, host()).is_err());
        out.write_host(2, host()).unwrap();
        out.finish().unwrap();
        assert_eq!(env.len(), 2);
    }
    #[test]
    fn missing_metadata_and_duplicate_declarations_are_errors() {
        let mut env = TensorEnv::<Cpu>::new(Default::default());
        let types = HashMap::from([(1, DataType::Float32)]);
        assert!(OperationOutputs::new(&mut env, &split(vec![1, 2]), &types, &[]).is_err());
        assert!(OperationOutputs::new(&mut env, &split(vec![1, 1]), &types, &[]).is_err());
    }
    #[test]
    fn bound_shapes_are_checked_and_half_results_rounded_before_publication() {
        let mut env = TensorEnv::<Cpu>::new(Default::default());
        let op = split(vec![1]);
        let types = HashMap::from([(1, DataType::Float16)]);
        let bindings = [IOBinding {
            name: "y".into(),
            operand_id: 1,
            data_type: DataType::Float16,
            shape: vec![-1, 1],
        }];
        let mut out = OperationOutputs::new(&mut env, &op, &types, &bindings).unwrap();
        assert!(out.write_host(1, host()).is_err());
        out.write_host(1, HostArray::new(vec![1, 1], vec![1.0003]).unwrap())
            .unwrap();
        out.finish().unwrap();
        assert_eq!(env.get_host(1).unwrap().data, [1.0]);
    }
}

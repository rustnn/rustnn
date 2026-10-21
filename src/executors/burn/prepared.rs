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

//! Validated, reusable execution state. Constants survive dispatch; activations do not.

use std::collections::{HashMap, HashSet};

use burn_tensor::backend::Backend;

use crate::burn::{BurnGraphPlan, IOBinding};
use crate::{DataType, GraphError, Operation};

use super::execute::execute_operations;
use super::interpreter::{BurnInput, BurnOutputWithData, bytes_to_host_values, validate_input};
use super::shape::checked_numel;
use super::tensor_env::TensorEnv;

fn err(reason: impl Into<String>) -> GraphError {
    GraphError::BurnRuntimeFailed {
        reason: reason.into(),
    }
}

fn binding_type(plan: &BurnGraphPlan, binding: &IOBinding) -> Result<(), GraphError> {
    if binding.name.is_empty()
        || binding.shape.len() > 8
        || binding
            .shape
            .iter()
            .any(|&d| d < -1 || d > i64::from(u32::MAX))
    {
        return Err(err(format!("invalid binding '{}'", binding.name)));
    }
    if plan.operand_types.get(&binding.operand_id) != Some(&binding.data_type) {
        return Err(err(format!(
            "binding '{}' has missing or inconsistent type metadata",
            binding.name
        )));
    }
    if binding.shape.iter().all(|&d| d >= 0) {
        checked_numel(
            &binding
                .shape
                .iter()
                .map(|&d| d as usize)
                .collect::<Vec<_>>(),
        )?;
    }
    Ok(())
}

fn validate_plan(plan: &BurnGraphPlan) -> Result<(), GraphError> {
    if plan.version != crate::burn::BURN_PLAN_VERSION {
        return Err(err(format!(
            "unsupported burn plan version {} (expected {})",
            plan.version,
            crate::burn::BURN_PLAN_VERSION
        )));
    }
    let mut available = HashSet::new();
    let mut names = HashSet::new();
    for input in &plan.inputs {
        binding_type(plan, input)?;
        if !names.insert(&input.name) || !available.insert(input.operand_id) {
            return Err(err("duplicate input name or operand"));
        }
    }
    for constant in &plan.constants {
        if !available.insert(constant.operand_id)
            || plan.operand_types.get(&constant.operand_id) != Some(&constant.data_type)
        {
            return Err(err(
                "duplicate constant operand or inconsistent constant type",
            ));
        }
        let n = checked_numel(
            &constant
                .shape
                .iter()
                .map(|&d| d as usize)
                .collect::<Vec<_>>(),
        )?;
        if constant.data_type.storage_byte_length(n) != Some(constant.data.len()) {
            return Err(err(format!(
                "constant {} has an invalid byte length",
                constant.operand_id
            )));
        }
    }
    for op in &plan.operations {
        let expected_outputs = match op {
            Operation::Split {
                splits,
                split_equal_parts,
                ..
            } => split_equal_parts.map_or(splits.len(), |n| n as usize),
            Operation::Gru { options, .. } => {
                1 + usize::from(options.as_ref().is_some_and(|o| o.return_sequence))
            }
            Operation::Lstm { options, .. } => {
                2 + usize::from(options.as_ref().is_some_and(|o| o.return_sequence))
            }
            Operation::LstmCell { .. } => 2,
            _ => 1,
        };
        if expected_outputs == 0 || op.outputs().len() != expected_outputs {
            return Err(err(format!(
                "{} expects {expected_outputs} output operands, got {}",
                op.op_type(),
                op.outputs().len()
            )));
        }
        for id in op.all_input_operands() {
            if !available.contains(&id) {
                return Err(err(format!(
                    "{} reads unavailable operand {id}",
                    op.op_type()
                )));
            }
        }
        let inputs = op.inputs();
        let first_type = inputs
            .first()
            .and_then(|id| plan.operand_types.get(id))
            .copied()
            .ok_or_else(|| err(format!("{} has no input operand", op.op_type())))?;
        let output_type = match op {
            Operation::Cast { data_type, .. } => (*data_type).into(),
            Operation::Shape { .. } => DataType::Int64,
            Operation::ArgMin { options, .. } | Operation::ArgMax { options, .. } => {
                options.clone().unwrap_or_default().output_data_type.into()
            }
            Operation::Equal { .. }
            | Operation::NotEqual { .. }
            | Operation::Greater { .. }
            | Operation::GreaterOrEqual { .. }
            | Operation::Lesser { .. }
            | Operation::LesserOrEqual { .. }
            | Operation::LogicalAnd { .. }
            | Operation::LogicalOr { .. }
            | Operation::LogicalXor { .. }
            | Operation::LogicalNot { .. }
            | Operation::IsNaN { .. }
            | Operation::IsInfinite { .. } => DataType::Uint8,
            Operation::Where { true_value, .. } => plan.operand_types[true_value],
            Operation::QuantizeLinear { zero_point, .. } => {
                zero_point.map_or(DataType::Uint8, |id| plan.operand_types[&id])
            }
            Operation::DequantizeLinear { scale, .. } => plan.operand_types[scale],
            _ => first_type,
        };
        for &id in op.outputs() {
            if plan.operand_types.get(&id) != Some(&output_type) || !available.insert(id) {
                return Err(err(format!(
                    "{} has a duplicate output or inconsistent type for operand {id}",
                    op.op_type()
                )));
            }
        }
    }
    names.clear();
    for output in &plan.outputs {
        binding_type(plan, output)?;
        if !names.insert(&output.name) || !available.contains(&output.operand_id) {
            return Err(err("duplicate output name or unavailable output operand"));
        }
    }
    Ok(())
}

pub(crate) struct PreparedPlan<B: Backend> {
    pub(crate) plan: BurnGraphPlan,
    env: TensorEnv<B>,
    constants: HashSet<u32>,
    releases: Vec<Vec<u32>>,
    #[cfg(test)]
    peak_activations: usize,
}

impl<B: Backend> std::fmt::Debug for PreparedPlan<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedPlan")
            .field("operations", &self.plan.operations.len())
            .field("constants", &self.constants.len())
            .field("live_tensors", &self.env.len())
            .finish()
    }
}

impl<B: Backend> PreparedPlan<B> {
    /// Prepare a builder-validated graph without repeating operation semantics.
    /// Keep this entry point crate-private: serialized plans use `new` instead.
    pub(crate) fn from_builder_graph(
        graph: &crate::GraphInfo,
        device: B::Device,
    ) -> Result<Self, GraphError> {
        // These are runtime representation limits, not WebNN operation rules.
        // Inspect intermediates too, so unsupported ranks fail at build time.
        for operand in &graph.operands {
            if operand.descriptor.shape.len() > 8 {
                return Err(err("Burn tensors require rank <= 8"));
            }
            if let Some(shape) = operand.descriptor.static_shape() {
                checked_numel(&shape.iter().map(|&d| d as usize).collect::<Vec<_>>())?;
            }
        }
        Self::prepare(crate::converters::BurnConverter::build_plan(graph)?, device)
    }

    /// Validate the envelope and metadata of an externally supplied plan.
    pub(crate) fn new(plan: BurnGraphPlan, device: B::Device) -> Result<Self, GraphError> {
        validate_plan(&plan)?;
        Self::prepare(plan, device)
    }

    fn prepare(plan: BurnGraphPlan, device: B::Device) -> Result<Self, GraphError> {
        let mut env = TensorEnv::new(device);
        for slot in &plan.constants {
            let shape = slot.shape.iter().map(|&d| d as usize).collect::<Vec<_>>();
            let (values, signed, unsigned) =
                bytes_to_host_values(&slot.data, slot.data_type, checked_numel(&shape)?)?;
            env.insert_input(
                slot.operand_id,
                slot.data_type,
                BurnInput {
                    name: String::new(),
                    shape,
                    data: values,
                    int64_data: signed,
                    uint64_data: unsigned,
                },
            )?;
        }
        let constants: HashSet<_> = plan.constants.iter().map(|c| c.operand_id).collect();
        let outputs: HashSet<_> = plan.outputs.iter().map(|b| b.operand_id).collect();
        let mut last_use = HashMap::new();
        for (index, op) in plan.operations.iter().enumerate() {
            // Include dead outputs so they are released immediately after production.
            for id in op
                .all_input_operands()
                .into_iter()
                .chain(op.outputs().iter().copied())
            {
                if !constants.contains(&id) && !outputs.contains(&id) {
                    last_use.insert(id, index);
                }
            }
        }
        let mut releases = vec![Vec::new(); plan.operations.len()];
        for (id, index) in last_use {
            releases[index].push(id);
        }
        Ok(Self {
            plan,
            env,
            constants,
            releases,
            #[cfg(test)]
            peak_activations: 0,
        })
    }

    pub(crate) fn execute(
        &mut self,
        inputs: Vec<BurnInput>,
    ) -> Result<Vec<BurnOutputWithData>, GraphError> {
        // Always reset before and after execution, including the error path.
        self.clear_activations();
        // Burn's shape assertions can unwind for malformed runtime arguments.
        // Keep that failure inside this Result-based API, then discard all run
        // state. Constants are read-only and kernels receive cloned tensor handles.
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.execute_inner(inputs)));
        self.clear_activations();
        result.unwrap_or_else(|payload| {
            let detail = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown kernel failure");
            Err(err(format!("Burn execution panicked: {detail}")))
        })
    }

    fn clear_activations(&mut self) {
        self.env.retain(|id| self.constants.contains(id));
    }

    fn execute_inner(
        &mut self,
        inputs: Vec<BurnInput>,
    ) -> Result<Vec<BurnOutputWithData>, GraphError> {
        let mut input_map = HashMap::new();
        for input in inputs {
            let name = input.name.clone();
            if input_map.insert(name.clone(), input).is_some() {
                return Err(err(format!("duplicate input '{name}'")));
            }
        }
        if input_map.len() != self.plan.inputs.len() {
            return Err(err("input bindings do not match the plan"));
        }
        for binding in &self.plan.inputs {
            let input = input_map
                .remove(&binding.name)
                .ok_or_else(|| err(format!("missing input '{}'", binding.name)))?;
            validate_input(binding, &input)?;
            self.env
                .insert_input(binding.operand_id, binding.data_type, input)?;
        }
        for (op, release) in self.plan.operations.iter().zip(&self.releases) {
            execute_operations(
                &mut self.env,
                std::slice::from_ref(op),
                &self.plan.operand_types,
                &self.plan.outputs,
            )?;
            #[cfg(test)]
            {
                self.peak_activations = self
                    .peak_activations
                    .max(self.env.len() - self.constants.len());
            }
            for &id in release {
                self.env.remove(id);
            }
        }
        self.plan
            .outputs
            .iter()
            .map(|binding| {
                let tensor = self.env.value(binding.operand_id)?;
                if tensor.dtype() != binding.data_type
                    || !super::shape::matches_binding(tensor.shape(), &binding.shape)
                {
                    return Err(err(format!(
                        "output '{}' does not match its declared shape",
                        binding.name
                    )));
                }
                tensor.output(binding.name.clone())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataType;
    use crate::burn::ConstantSlot;
    use burn_ndarray::NdArray;

    fn binding(name: &str, id: u32) -> IOBinding {
        IOBinding {
            name: name.into(),
            operand_id: id,
            data_type: DataType::Int64,
            shape: vec![1],
        }
    }

    fn chain(length: u32) -> BurnGraphPlan {
        BurnGraphPlan::new(
            vec![binding("x", 0)],
            vec![binding("y", length + 1)],
            vec![ConstantSlot {
                operand_id: 1,
                data_type: DataType::Int64,
                shape: vec![1],
                data: 1i64.to_le_bytes().to_vec(),
            }],
            (0..length)
                .map(|i| Operation::Add {
                    a: if i == 0 { 0 } else { i + 1 },
                    b: 1,
                    options: None,
                    outputs: vec![i + 2],
                })
                .collect(),
            (0..length + 2).map(|i| (i, DataType::Int64)).collect(),
        )
    }

    fn input(value: i64) -> BurnInput {
        BurnInput {
            name: "x".into(),
            shape: vec![1],
            data: vec![value as f32],
            int64_data: Some(vec![value]),
            uint64_data: None,
        }
    }

    #[test]
    fn builder_preparation_checks_intermediate_representation_limits() {
        use crate::mlcontext::{MLGraphBuilder, MLNamedOperands, MLOperandDescriptor};
        use crate::operator_enums::MLOperandDataType;
        use crate::operator_options::MLDimension;

        for rank in [8, 9] {
            let mut builder = MLGraphBuilder::new_uncompiled();
            let x = builder
                .input(
                    "x",
                    &MLOperandDescriptor::new(MLOperandDataType::Int64, vec![1]),
                )
                .unwrap();
            let intermediate = builder
                .reshape(x, vec![MLDimension::Static(1); rank])
                .unwrap();
            let y = builder
                .reshape(intermediate, vec![MLDimension::Static(1)])
                .unwrap();
            let graph = builder
                .finish_graph_info(&MLNamedOperands::from([("y", y)]))
                .unwrap();
            let result =
                PreparedPlan::<NdArray<f32>>::from_builder_graph(&graph, Default::default());
            if rank == 9 {
                assert!(result.unwrap_err().to_string().contains("rank <= 8"));
            } else {
                let mut runtime = result.unwrap();
                for value in [16_777_217, i64::MAX] {
                    let output = runtime.execute(vec![input(value)]).unwrap();
                    assert_eq!(output[0].int64_data.as_deref(), Some([value].as_slice()));
                    assert_eq!(runtime.env.len(), 0);
                }
            }
        }
    }

    #[test]
    fn long_chain_releases_exact_integer_payloads_and_reuses_constants() {
        let mut plan = chain(256);
        plan.outputs.push(binding("early", 3));
        let mut runtime = PreparedPlan::<NdArray<f32>>::new(plan, Default::default()).unwrap();
        let constant_pointer = runtime
            .env
            .value(1)
            .unwrap()
            .signed_data()
            .unwrap()
            .as_ptr();
        for x in [16_777_217, 32_000_001] {
            let result = runtime.execute(vec![input(x)]).unwrap();
            assert_eq!(result[0].int64_data.as_deref(), Some([x + 256].as_slice()));
            assert_eq!(result[1].int64_data.as_deref(), Some([x + 2].as_slice()));
            assert_eq!(runtime.env.len(), 1);
            assert_eq!(
                runtime
                    .env
                    .value(1)
                    .unwrap()
                    .signed_data()
                    .unwrap()
                    .as_ptr(),
                constant_pointer
            );
        }
        assert!(
            runtime.peak_activations <= 3,
            "peak {}",
            runtime.peak_activations
        );
    }

    #[test]
    fn fanout_repeated_uses_and_failed_runs_do_not_drop_or_leak_values() {
        let mut plan = chain(3);
        plan.operations[1] = Operation::Mul {
            a: 2,
            b: 2,
            options: None,
            outputs: vec![3],
        };
        plan.operations[2] = Operation::Add {
            a: 2,
            b: 3,
            options: None,
            outputs: vec![4],
        };
        plan.outputs.push(binding("branch", 2));
        let mut runtime = PreparedPlan::<NdArray<f32>>::new(plan, Default::default()).unwrap();
        // Fail after all kernels ran, rather than just during input validation.
        runtime.plan.outputs[0].shape = vec![2];
        assert!(runtime.execute(vec![input(2)]).is_err());
        assert_eq!(runtime.env.len(), 1);
        runtime.plan.outputs[0].shape = vec![1];
        for x in [2, 4] {
            let result = runtime.execute(vec![input(x)]).unwrap();
            assert_eq!(
                result[0].int64_data.as_deref(),
                Some([(x + 1) + (x + 1) * (x + 1)].as_slice())
            );
            assert_eq!(result[1].int64_data.as_deref(), Some([x + 1].as_slice()));
        }
    }

    #[test]
    fn malformed_plans_are_rejected_before_execution() {
        let good = chain(2);
        let mut cases = Vec::new();
        let mut p = good.clone();
        p.version += 1;
        cases.push(p);
        let mut p = good.clone();
        p.inputs.push(p.inputs[0].clone());
        cases.push(p);
        let mut p = good.clone();
        p.outputs.push(p.outputs[0].clone());
        cases.push(p);
        let mut p = good.clone();
        p.constants[0].operand_id = 0;
        cases.push(p);
        let mut p = good.clone();
        p.constants[0].data.pop();
        cases.push(p);
        let mut p = good.clone();
        p.operand_types.remove(&2);
        cases.push(p);
        let mut p = good.clone();
        p.operand_types.insert(2, DataType::Float32);
        cases.push(p);
        let mut p = good.clone();
        p.operand_types.insert(0, DataType::Float32);
        cases.push(p);
        let mut p = good.clone();
        p.operations.swap(0, 1);
        cases.push(p);
        let mut p = good.clone();
        p.operations[0] = Operation::Add {
            a: 99,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        cases.push(p);
        let mut p = good.clone();
        p.operations[1] = Operation::Identity {
            input: 2,
            options: None,
            outputs: vec![2],
        };
        cases.push(p);
        let mut p = good.clone();
        p.operations[0] = Operation::Identity {
            input: 0,
            options: None,
            outputs: vec![2, 3],
        };
        cases.push(p);
        let mut p = good.clone();
        p.operations[0] = Operation::Identity {
            input: 0,
            options: None,
            outputs: vec![],
        };
        cases.push(p);
        let mut p = good.clone();
        p.outputs[0].operand_id = 100;
        cases.push(p);
        let mut p = good.clone();
        p.operations[0] = Operation::Concat {
            inputs: vec![],
            axis: 0,
            options: None,
            outputs: vec![2],
        };
        cases.push(p);
        let mut p = good.clone();
        p.inputs[0].shape = vec![-2];
        cases.push(p);
        for (index, plan) in cases.into_iter().enumerate() {
            assert!(
                super::super::run_burn_cpu_with_inputs(&plan.serialize().unwrap(), vec![]).is_err(),
                "case {index}"
            );
        }
    }

    #[test]
    fn malformed_input_payloads_and_binding_sets_are_rejected() {
        let mut runtime = PreparedPlan::<NdArray<f32>>::new(chain(1), Default::default()).unwrap();
        let good = input(1);
        let mut cases = vec![vec![], vec![good.clone(), good.clone()]];
        let mut x = good.clone();
        x.name = "unknown".into();
        cases.push(vec![x]);
        let mut x = good.clone();
        x.data.clear();
        cases.push(vec![x]);
        let mut x = good.clone();
        x.int64_data = None;
        cases.push(vec![x]);
        let mut x = good.clone();
        x.int64_data = Some(vec![]);
        cases.push(vec![x]);
        let mut x = good.clone();
        x.uint64_data = Some(vec![1]);
        cases.push(vec![x]);
        let mut x = good.clone();
        x.shape = vec![u32::MAX as usize; 8];
        cases.push(vec![x]);
        for case in cases {
            assert!(runtime.execute(case).is_err());
            assert_eq!(runtime.env.len(), 1);
        }
        assert!(runtime.execute(vec![good]).is_ok());
        assert!(checked_numel(&[u32::MAX as usize; 8]).is_err());
        assert_eq!(checked_numel(&[]).unwrap(), 1);
        assert_eq!(checked_numel(&[1, 0, 4]).unwrap(), 0);
    }

    #[test]
    fn graph_outputs_can_be_inputs_or_constants_without_operations() {
        let mut plan = chain(0);
        plan.outputs.push(binding("input", 0));
        let mut runtime = PreparedPlan::<NdArray<f32>>::new(plan, Default::default()).unwrap();
        for value in [16_777_217, 99] {
            let result = runtime.execute(vec![input(value)]).unwrap();
            assert_eq!(result[0].int64_data.as_deref(), Some([1].as_slice()));
            assert_eq!(result[1].int64_data.as_deref(), Some([value].as_slice()));
            assert_eq!(runtime.env.len(), 1);
        }
    }

    #[test]
    fn kernel_shape_panics_become_errors_and_clear_run_state() {
        let mut plan = chain(1);
        plan.operations[0] = Operation::Transpose {
            input: 0,
            options: Some(crate::operator_options::MLTransposeOptions {
                permutation: vec![1],
                ..Default::default()
            }),
            outputs: vec![2],
        };
        let mut runtime = PreparedPlan::<NdArray<f32>>::new(plan, Default::default()).unwrap();
        assert!(runtime.execute(vec![input(3)]).is_err());
        assert_eq!(runtime.env.len(), 1);
        runtime.plan.operations[0] = Operation::Identity {
            input: 0,
            options: None,
            outputs: vec![2],
        };
        let result = runtime.execute(vec![input(4)]).unwrap();
        assert_eq!(result[0].int64_data.as_deref(), Some([4].as_slice()));
    }
}

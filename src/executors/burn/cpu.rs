#![cfg(feature = "burn-runtime-cpu")]

use std::collections::HashMap;

use burn_ndarray::NdArray;

use crate::error::GraphError;
use crate::graph::OperandDescriptor;

use super::interpreter::{
    BurnInput, BurnOutput, BurnOutputWithData, execute_plan, plan_output_metadata,
    zeroed_inputs_from_descriptors,
};

type CpuBackend = NdArray<f32>;

pub fn run_burn_cpu_with_inputs(
    plan_bytes: &[u8],
    inputs: Vec<BurnInput>,
) -> Result<Vec<BurnOutputWithData>, GraphError> {
    execute_plan::<CpuBackend>(plan_bytes, inputs)
}

pub fn run_burn_cpu_zeroed(
    plan_bytes: &[u8],
    inputs: &HashMap<String, OperandDescriptor>,
) -> Result<Vec<BurnOutput>, GraphError> {
    let zeroed = zeroed_inputs_from_descriptors(inputs);
    let _ = run_burn_cpu_with_inputs(plan_bytes, zeroed)?;
    plan_output_metadata(plan_bytes)
}

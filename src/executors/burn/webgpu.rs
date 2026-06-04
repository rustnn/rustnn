#![cfg(feature = "burn-runtime-webgpu")]

use std::collections::HashMap;

use burn_ndarray::NdArray;
use burn_wgpu::Wgpu;

use crate::burn::BurnGraphPlan;
use crate::error::GraphError;
use crate::graph::OperandDescriptor;

use super::interpreter::{
    BurnInput, BurnOutput, BurnOutputWithData, execute_plan, execute_plan_on_device,
    plan_max_tensor_bytes, plan_output_metadata, zeroed_inputs_from_descriptors,
};
use super::webgpu_device::{effective_storage_buffer_binding_bytes, elevated_wgpu_device};

type WebGpuBackend = Wgpu<f32, i32>;
type CpuBackend = NdArray<f32>;

/// Effective [`maxStorageBufferBindingSize`](https://www.w3.org/TR/webgpu/#limits) for the
/// rustnn Burn WGPU device (elevated limits, queried once at init).
pub fn max_storage_buffer_binding_bytes() -> usize {
    effective_storage_buffer_binding_bytes()
}

/// Whether a single tensor exceeds what the elevated WGPU device can bind.
///
/// Tensors larger than the device limit still run on the CPU ndarray backend.
pub fn exceeds_webgpu_tensor_binding_limit(bytes: usize) -> bool {
    bytes > max_storage_buffer_binding_bytes()
}

pub fn run_burn_webgpu_with_inputs(
    plan_bytes: &[u8],
    inputs: Vec<BurnInput>,
) -> Result<Vec<BurnOutputWithData>, GraphError> {
    let plan =
        BurnGraphPlan::deserialize(plan_bytes).map_err(|err| GraphError::BurnRuntimeFailed {
            reason: format!("invalid burn plan bytes: {err}"),
        })?;
    let max_bytes = plan_max_tensor_bytes(&plan, &inputs);
    if exceeds_webgpu_tensor_binding_limit(max_bytes) {
        return execute_plan::<CpuBackend>(plan_bytes, inputs);
    }
    let device = elevated_wgpu_device()?;
    execute_plan_on_device::<WebGpuBackend>(plan_bytes, inputs, device)
}

pub fn run_burn_webgpu_zeroed(
    plan_bytes: &[u8],
    inputs: &HashMap<String, OperandDescriptor>,
) -> Result<Vec<BurnOutput>, GraphError> {
    let zeroed = zeroed_inputs_from_descriptors(inputs);
    let _ = run_burn_webgpu_with_inputs(plan_bytes, zeroed)?;
    plan_output_metadata(plan_bytes)
}

#[cfg(test)]
mod tests {
    use super::super::interpreter::WEBGPU_MIN_STORAGE_BUFFER_BINDING_BYTES;
    use super::super::webgpu_device::DESIRED_STORAGE_BUFFER_BINDING_BYTES;
    use super::*;

    #[test]
    fn desired_limit_exceeds_webgpu_spec_minimum() {
        assert!(
            DESIRED_STORAGE_BUFFER_BINDING_BYTES > WEBGPU_MIN_STORAGE_BUFFER_BINDING_BYTES as u64
        );
    }

    #[test]
    fn wpt_large_tensor_fits_desired_binding_limit() {
        let wpt_large = 6000usize * 6000 * std::mem::size_of::<f32>();
        assert!((wpt_large as u64) < DESIRED_STORAGE_BUFFER_BINDING_BYTES);
    }

    #[test]
    fn queried_storage_buffer_limit_at_least_webgpu_minimum() {
        if let Ok(limit) = std::panic::catch_unwind(max_storage_buffer_binding_bytes) {
            assert!(limit >= WEBGPU_MIN_STORAGE_BUFFER_BINDING_BYTES);
        }
    }
}

#[cfg(all(test, feature = "burn-runtime-webgpu"))]
mod integration {
    use super::*;
    use crate::converters::BurnConverter;
    use crate::converters::GraphConverter;
    use crate::graph::{
        DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, to_dimension_vector,
    };
    use crate::operators::Operation;
    use std::collections::HashMap;

    fn int32_div_graph() -> GraphInfo {
        let shape = to_dimension_vector(&[2, 2, 2, 3]);
        GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Int32,
                        shape: shape.clone(),
                        pending_permutation: Vec::new(),
                    },
                    name: Some("inputA".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Int32,
                        shape: shape.clone(),
                        pending_permutation: Vec::new(),
                    },
                    name: Some("inputB".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Int32,
                        shape: shape.clone(),
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            operations: vec![Operation::Div {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![2],
            }],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        }
    }

    fn large_add_graph(side: usize) -> GraphInfo {
        let shape = to_dimension_vector(&[side as u32, side as u32]);
        GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: shape.clone(),
                        pending_permutation: Vec::new(),
                    },
                    name: Some("inputA".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: shape.clone(),
                        pending_permutation: Vec::new(),
                    },
                    name: Some("inputB".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape,
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            operations: vec![Operation::Add {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![2],
            }],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        }
    }

    /// WPT uses 6000×6000 (~144 MiB); should run on WGPU when the adapter allows >= 256 MiB bindings.
    #[test]
    fn large_add_runs_on_webgpu_when_binding_limit_allows() {
        let side = 6000usize;
        let n = side * side;
        let bytes = n * std::mem::size_of::<f32>();
        if exceeds_webgpu_tensor_binding_limit(bytes) {
            eprintln!(
                "skipping large_add_runs_on_webgpu: device binding limit {} < {} bytes",
                max_storage_buffer_binding_bytes(),
                bytes
            );
            return;
        }
        let graph = large_add_graph(side);
        let plan = BurnConverter.convert(&graph).expect("burn plan");
        let inputs = vec![
            BurnInput {
                name: "inputA".to_string(),
                shape: vec![side, side],
                data: vec![1.0f32; n],
                int64_data: None,
                uint64_data: None,
            },
            BurnInput {
                name: "inputB".to_string(),
                shape: vec![side, side],
                data: vec![2.0f32; n],
                int64_data: None,
                uint64_data: None,
            },
        ];
        let outputs = run_burn_webgpu_with_inputs(&plan.data, inputs).expect("large add on webgpu");
        assert_eq!(outputs[0].data.len(), n);
        assert_eq!(outputs[0].data[0], 3.0);
    }

    #[test]
    fn div_int32_rounds_to_nearest_even_on_webgpu_path() {
        let graph = int32_div_graph();
        let plan = BurnConverter.convert(&graph).expect("burn plan");
        let mut a = vec![0.0f32; 24];
        let mut b = vec![1.0f32; 24];
        a[2] = 19.0;
        b[2] = 2.0;
        let inputs = vec![
            BurnInput {
                name: "inputA".to_string(),
                shape: vec![2, 2, 2, 3],
                data: a,
                int64_data: None,
                uint64_data: None,
            },
            BurnInput {
                name: "inputB".to_string(),
                shape: vec![2, 2, 2, 3],
                data: b,
                int64_data: None,
                uint64_data: None,
            },
        ];
        let outputs =
            run_burn_webgpu_with_inputs(&plan.data, inputs).expect("burn webgpu div int32");
        assert_eq!(outputs[0].data[2], 10.0);
    }
}

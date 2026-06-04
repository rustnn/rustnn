#![cfg(any(feature = "burn-runtime-cpu", feature = "burn-runtime-webgpu"))]

use std::collections::HashMap;
use std::fmt;

use crate::GraphInfo;
use crate::backend_selection::BackendDevice;
use crate::converters::BurnConverter;
use crate::converters::GraphConverter;
use crate::error::Error;
#[cfg(feature = "burn-runtime-webgpu")]
use crate::executors::burn::run_burn_webgpu_with_inputs;
use crate::executors::burn::{BurnInput, BurnOutputWithData, execute_plan};
use crate::graph::DataType;
use crate::mlcontext::{
    ListDevices, MLBackendBuilder, MLBackendContext, MLGraph, MLTensor, MLTensorDescriptor,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BurnRuntimeKind {
    #[cfg(feature = "burn-runtime-cpu")]
    Cpu,
    #[cfg(feature = "burn-runtime-webgpu")]
    WebGpu,
}

#[derive(Debug)]
pub(crate) struct BurnCompiledGraph {
    pub(crate) plan_bytes: Vec<u8>,
}

#[derive(Debug)]
struct BurnHostTensor {
    memory: Vec<u8>,
}

pub(crate) struct BurnContext {
    kind: BurnRuntimeKind,
    tensors: Vec<BurnHostTensor>,
}

impl fmt::Debug for BurnContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BurnContext")
            .field("kind", &self.kind)
            .field("tensor_count", &self.tensors.len())
            .finish()
    }
}

impl BurnContext {
    pub(crate) fn new(kind: BurnRuntimeKind) -> Self {
        Self {
            kind,
            tensors: Vec::new(),
        }
    }
}

pub(crate) struct BurnBuilder;

impl fmt::Debug for BurnBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BurnBuilder").finish()
    }
}

impl<'context, 'builder> MLBackendBuilder<'context, 'builder> for BurnBuilder {
    fn build(&mut self, graph_info: GraphInfo) -> crate::error::Result<MLGraph<'context>> {
        let converted = BurnConverter
            .convert(&graph_info)
            .map_err(|e| Error::GraphBuildError { source: e.into() })?;
        Ok(MLGraph {
            backend: crate::mlcontext::MLBackendGraph::BurnPlan(BurnCompiledGraph {
                plan_bytes: converted.data,
            }),
        })
    }
}

fn tensor_byte_len(descriptor: &MLTensorDescriptor) -> crate::error::Result<usize> {
    let bits = descriptor.data_type().rustnn_element_size_bits();
    let elements: usize = descriptor
        .shape()
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| Error::GraphDispatchError {
            source: "tensor element count overflow".into(),
        })? as usize;
    Ok(bits * elements / 8)
}

fn host_bytes_to_f32(
    bytes: &[u8],
    data_type: DataType,
    tensor: &MLTensor,
) -> crate::error::Result<Vec<f32>> {
    let err = |msg: String| Error::GraphDispatchError { source: msg.into() };
    match data_type {
        DataType::Float32 => {
            let floats: &[f32] = bytemuck::try_cast_slice(bytes).map_err(|e| err(e.to_string()))?;
            Ok(floats.to_vec())
        }
        DataType::Float16 => Err(err(
            "Float16 MLContext tensors are not supported for Burn dispatch yet".into(),
        )),
        DataType::Int64 => {
            let ints: &[i64] = bytemuck::try_cast_slice(bytes).map_err(|e| err(e.to_string()))?;
            Ok(ints.iter().map(|&v| v as f32).collect())
        }
        DataType::Uint64 => {
            let ints: &[u64] = bytemuck::try_cast_slice(bytes).map_err(|e| err(e.to_string()))?;
            Ok(ints.iter().map(|&v| v as f32).collect())
        }
        DataType::Int32 => {
            let ints: &[i32] = bytemuck::try_cast_slice(bytes).map_err(|e| err(e.to_string()))?;
            Ok(ints.iter().map(|&v| v as f32).collect())
        }
        DataType::Uint32 => {
            let ints: &[u32] = bytemuck::try_cast_slice(bytes).map_err(|e| err(e.to_string()))?;
            Ok(ints.iter().map(|&v| v as f32).collect())
        }
        DataType::Int8 => {
            let ints: &[i8] = bytemuck::try_cast_slice(bytes).map_err(|e| err(e.to_string()))?;
            Ok(ints.iter().map(|&v| v as f32).collect())
        }
        DataType::Uint8 => Ok(bytes.iter().map(|&v| v as f32).collect()),
        DataType::Int4 | DataType::Uint4 => Err(err(format!(
            "unsupported Burn dispatch input type {data_type:?} for tensor {}",
            tensor.id
        ))),
    }
}

fn write_f32_output_to_host(
    host: &mut [u8],
    data: &[f32],
    data_type: DataType,
    name: &str,
) -> crate::error::Result<()> {
    let err = |msg: String| Error::GraphDispatchError { source: msg.into() };
    match data_type {
        DataType::Float32 => {
            let bytes = bytemuck::cast_slice(data);
            if bytes.len() != host.len() {
                return Err(err(format!(
                    "output '{name}': byte length mismatch (expected {}, got {})",
                    host.len(),
                    bytes.len()
                )));
            }
            host.copy_from_slice(bytes);
        }
        DataType::Int64 => {
            if data.len() * 8 != host.len() {
                return Err(err(format!(
                    "output '{name}': element count mismatch for int64"
                )));
            }
            let out: Vec<i64> = data.iter().map(|&v| v as i64).collect();
            host.copy_from_slice(bytemuck::cast_slice(&out));
        }
        DataType::Int32 => {
            if data.len() * 4 != host.len() {
                return Err(err(format!(
                    "output '{name}': element count mismatch for int32"
                )));
            }
            let out: Vec<i32> = data.iter().map(|&v| v as i32).collect();
            host.copy_from_slice(bytemuck::cast_slice(&out));
        }
        other => {
            return Err(err(format!(
                "unsupported Burn dispatch output type {other:?} for '{name}'"
            )));
        }
    }
    Ok(())
}

fn run_plan(
    kind: BurnRuntimeKind,
    plan_bytes: &[u8],
    inputs: Vec<BurnInput>,
) -> crate::error::Result<Vec<BurnOutputWithData>> {
    match kind {
        #[cfg(feature = "burn-runtime-cpu")]
        BurnRuntimeKind::Cpu => execute_plan::<burn_ndarray::NdArray<f32>>(plan_bytes, inputs)
            .map_err(|e| Error::GraphDispatchError { source: e.into() }),
        #[cfg(feature = "burn-runtime-webgpu")]
        BurnRuntimeKind::WebGpu => run_burn_webgpu_with_inputs(plan_bytes, inputs)
            .map_err(|e| Error::GraphDispatchError { source: e.into() }),
    }
}

impl ListDevices for BurnContext {
    fn list_devices() -> Vec<BackendDevice> {
        let mut devices = Vec::new();
        #[cfg(feature = "burn-runtime-cpu")]
        devices.push(BackendDevice::BurnCpu);
        #[cfg(feature = "burn-runtime-webgpu")]
        devices.push(BackendDevice::BurnWebGpu);
        devices
    }
}

impl<'context> MLBackendContext<'context> for BurnContext {
    fn accelerated(&self) -> bool {
        match self.kind {
            #[cfg(feature = "burn-runtime-cpu")]
            BurnRuntimeKind::Cpu => false,
            #[cfg(feature = "burn-runtime-webgpu")]
            BurnRuntimeKind::WebGpu => true,
        }
    }

    fn create_builder<'builder>(
        &mut self,
    ) -> crate::error::Result<Box<dyn MLBackendBuilder<'context, 'builder> + 'builder>>
    where
        'context: 'builder,
    {
        Ok(Box::new(BurnBuilder))
    }

    fn create_tensor(&mut self, descriptor: &MLTensorDescriptor) -> crate::error::Result<MLTensor> {
        let n = tensor_byte_len(descriptor)?;
        self.tensors.push(BurnHostTensor {
            memory: vec![0u8; n.max(1)],
        });
        Ok(MLTensor {
            id: self.tensors.len() - 1,
            constant: false,
            descriptor: descriptor.clone(),
        })
    }

    fn create_constant_tensor(
        &mut self,
        descriptor: &MLTensorDescriptor,
        input_data: &[u8],
    ) -> crate::error::Result<MLTensor> {
        let mut tensor = self.create_tensor(descriptor)?;
        tensor.constant = true;
        self.write_tensor(&tensor, input_data)
            .map_err(|e| Error::TensorCreationError {
                source: e.into(),
                descriptor: descriptor.clone(),
            })?;
        Ok(tensor)
    }

    fn read_tensor(&mut self, tensor: &MLTensor, array: &mut [u8]) -> crate::error::Result<()> {
        let host = &self.tensors[tensor.id].memory;
        let logical = tensor_byte_len(tensor.descriptor())?;
        if array.len() < logical {
            return Err(Error::TensorReadError {
                source: format!(
                    "buffer too small: need {logical} logical bytes, got {}",
                    array.len()
                )
                .into(),
                tensor: tensor.clone(),
            });
        }
        let slice = host.get(..logical).ok_or_else(|| Error::TensorReadError {
            source: format!("tensor storage shorter than logical size ({logical} bytes)").into(),
            tensor: tensor.clone(),
        })?;
        array[..logical].copy_from_slice(slice);
        Ok(())
    }

    fn write_tensor(&mut self, tensor: &MLTensor, array: &[u8]) -> crate::error::Result<()> {
        let host = &mut self.tensors[tensor.id].memory;
        if array.len() > host.len() {
            return Err(Error::TensorWriteError {
                source: format!(
                    "write exceeds tensor storage: {} bytes > {}",
                    array.len(),
                    host.len()
                )
                .into(),
                tensor: tensor.clone(),
            });
        }
        let n = array.len();
        host[..n].copy_from_slice(array);
        Ok(())
    }

    fn dispatch(
        &mut self,
        graph: &mut MLGraph,
        inputs: &HashMap<&str, &MLTensor>,
        outputs: &HashMap<&str, &MLTensor>,
    ) -> crate::error::Result<()> {
        let burn_graph =
            graph
                .backend
                .as_burn_plan_mut()
                .ok_or_else(|| Error::GraphDispatchError {
                    source: "MLGraph is not a Burn plan graph".into(),
                })?;

        let plan =
            crate::burn::BurnGraphPlan::deserialize(&burn_graph.plan_bytes).map_err(|err| {
                Error::GraphDispatchError {
                    source: format!("invalid burn plan bytes: {err}").into(),
                }
            })?;

        let mut burn_inputs = Vec::with_capacity(plan.inputs.len());
        for binding in &plan.inputs {
            let tensor =
                inputs
                    .get(binding.name.as_str())
                    .ok_or_else(|| Error::GraphDispatchError {
                        source: format!("missing input '{}' for Burn dispatch", binding.name)
                            .into(),
                    })?;
            let shape: Vec<usize> = tensor.shape().iter().map(|&d| d as usize).collect();
            let full = &self.tensors[tensor.id].memory;
            let logical = tensor_byte_len(tensor.descriptor())?;
            let bytes = full
                .get(..logical)
                .ok_or_else(|| Error::GraphDispatchError {
                    source: format!(
                        "input '{}': tensor storage shorter than logical size ({logical} bytes)",
                        binding.name
                    )
                    .into(),
                })?;
            let data = host_bytes_to_f32(bytes, binding.data_type, tensor)?;
            burn_inputs.push(BurnInput {
                name: binding.name.clone(),
                shape,
                data,
                int64_data: None,
                uint64_data: None,
            });
        }

        let results = run_plan(self.kind, &burn_graph.plan_bytes, burn_inputs)?;

        for binding in &plan.outputs {
            let tensor =
                outputs
                    .get(binding.name.as_str())
                    .ok_or_else(|| Error::GraphDispatchError {
                        source: format!("missing output '{}' for Burn dispatch", binding.name)
                            .into(),
                    })?;
            let result = results
                .iter()
                .find(|out| out.name == binding.name)
                .ok_or_else(|| Error::GraphDispatchError {
                    source: format!("Burn plan produced no output '{}'", binding.name).into(),
                })?;
            let logical = tensor_byte_len(tensor.descriptor())?;
            let host = &mut self.tensors[tensor.id].memory;
            if host.len() < logical {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{}': tensor storage too small (have {}, need {})",
                        binding.name,
                        host.len(),
                        logical
                    )
                    .into(),
                });
            }
            write_f32_output_to_host(
                &mut host[..logical],
                &result.data,
                binding.data_type,
                &binding.name,
            )?;
        }

        Ok(())
    }

    fn rustnn_resize_tensor(
        &mut self,
        tensor: &mut MLTensor,
        new_shape: &[u64],
    ) -> crate::error::Result<()> {
        let mut new_desc = tensor.descriptor().clone();
        new_desc.set_shape(new_shape.to_vec());
        let new_bytes = tensor_byte_len(&new_desc)?;
        let host = &mut self.tensors[tensor.id].memory;
        if new_bytes > host.len() {
            host.resize(new_bytes, 0u8);
        }
        tensor.descriptor = new_desc;
        Ok(())
    }

    fn rustnn_set_tensor_capacity(
        &mut self,
        tensor: &mut MLTensor,
        max_shape: &[u64],
    ) -> crate::error::Result<()> {
        let bits = tensor.data_type().rustnn_element_size_bits();
        let elements: u64 = max_shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| Error::GraphDispatchError {
                source: "rustnn_set_tensor_capacity: shape element count overflow".into(),
            })?;
        let new_bytes = (elements as usize)
            .checked_mul(bits)
            .and_then(|b| b.checked_div(8))
            .ok_or_else(|| Error::GraphDispatchError {
                source: "rustnn_set_tensor_capacity: byte length overflow".into(),
            })?;
        self.tensors[tensor.id].memory = vec![0u8; new_bytes.max(1)];
        Ok(())
    }
}

#[cfg(all(test, feature = "burn-runtime-cpu"))]
mod tests {
    use super::*;
    use crate::mlcontext::{BackendPreference, MLContext, MLContextOptions, MLPowerPreference};
    use crate::webnn_json::from_graph_json;

    #[test]
    fn create_context_with_burn_cpu_hint() {
        let context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_backend_preference(BackendPreference::BurnCpu),
        )
        .expect("burn cpu context");
        assert!(!context.accelerated());
    }

    #[test]
    fn dispatch_add_graph_via_burn_cpu_hint() {
        let contents = r#"
webnn_graph "sample_graph" v1 {
  inputs { lhs: f32[2, 2]; }
  consts { rhs: f32[2, 2] @scalar(1.0); }
  nodes { sum = add(lhs, rhs); }
  outputs { sum; }
}"#;
        let sanitized = crate::loader::sanitize_webnn_identifiers(contents);
        let graph_json = webnn_graph::parser::parse_wg_text(&sanitized).unwrap();
        let graph_info = from_graph_json(&graph_json).unwrap();

        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_backend_preference(BackendPreference::BurnCpu),
        )
        .expect("burn cpu context");

        let mut desc = MLTensorDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            vec![2, 2],
        );
        desc.set_readable(true);
        desc.set_writable(true);

        let mut builder = crate::mlcontext::MLGraphBuilder::new(&mut context).unwrap();
        let mut graph = builder.build_graph_info(graph_info).unwrap();
        drop(builder);

        let tensor = context.create_tensor(&desc).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("lhs", &tensor);
        let mut outputs = HashMap::new();
        outputs.insert("sum", &tensor);

        let upload = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut download = vec![0.0f32; 4];
        context.write_tensor(&tensor, &upload).unwrap();
        context.dispatch(&mut graph, &inputs, &outputs).unwrap();
        context.read_tensor(&tensor, &mut download).unwrap();
        assert_eq!(download, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn write_int64_tensor_with_capacity() {
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_backend_preference(BackendPreference::BurnCpu),
        )
        .expect("burn cpu context");

        let mut desc =
            MLTensorDescriptor::new(crate::operator_enums::MLOperandDataType::Int64, vec![1, 1]);
        desc.set_writable(true);
        desc.set_readable(true);
        let mut tensor = context.create_tensor(&desc).unwrap();
        context
            .rustnn_set_tensor_capacity(&mut tensor, &[1, 512])
            .unwrap();
        context.rustnn_resize_tensor(&mut tensor, &[1, 4]).unwrap();
        context.write_tensor(&tensor, &[1i64, 0, 1, 0]).unwrap();
        let mut out = [0i64; 4];
        context.read_tensor(&tensor, &mut out).unwrap();
        assert_eq!(out, [1, 0, 1, 0]);
    }
}

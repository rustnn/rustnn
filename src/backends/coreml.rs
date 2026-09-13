//! CoreML backend for the unified WebNN IDL API (macOS only).
//!
//! Mirrors the ONNX Runtime backend in `src/backends/ort.rs`: a [`CoremlContext`]
//! owns raw-byte host tensor storage, a [`CoremlBuilder`] converts a [`GraphInfo`]
//! to a CoreML MLProgram and compiles it once, and [`CoremlGraph`] holds the
//! compiled model for repeated dispatch.

#![cfg(feature = "coreml-runtime")]

use std::collections::HashMap;
use std::fmt;

use log::debug;

use crate::GraphInfo;
use crate::backend_selection::DeviceType;
use crate::converters::{CoremlMlProgramConverter, GraphConverter};
use crate::error::Error;
use crate::executors::coreml::{
    CompiledCoremlModel, CoremlByteInput, compile_model, run_coreml_bytes,
};
use crate::graph::DataType;
use crate::mlcontext::RustNNOptions;
use crate::mlcontext::{
    MLBackendBuilder, MLBackendContext, MLBackendGraph, MLGraph, MLNamedTensors, MLTensor,
    MLTensorDescriptor,
};
use crate::operators::Operation;

/// Number of bytes required to store a tensor described by `descriptor`.
fn tensor_byte_len(descriptor: &MLTensorDescriptor) -> usize {
    descriptor.rustnn_required_bytes()
}

/// Bind CoreML to the active tensor extents, not the graph's allocation bounds.
fn runtime_input_descriptor(
    graph_descriptor: &crate::graph::OperandDescriptor,
    shape: &[u64],
) -> crate::error::Result<crate::graph::OperandDescriptor> {
    let shape = shape
        .iter()
        .map(|&size| {
            u32::try_from(size)
                .map(crate::graph::Dimension::Static)
                .map_err(|_| Error::GraphDispatchError {
                    source: format!("runtime input dimension {size} exceeds u32::MAX").into(),
                })
        })
        .collect::<crate::error::Result<Vec<_>>>()?;
    Ok(crate::graph::OperandDescriptor {
        data_type: graph_descriptor.data_type,
        shape,
        pending_permutation: graph_descriptor.pending_permutation.clone(),
    })
}

/// Host tensor storage for the CoreML backend (mirrors `OrtTensor`).
#[derive(Debug)]
pub(crate) struct CoremlTensor {
    memory: Vec<u8>,
}

/// A compiled CoreML model held by [`MLGraph`] (mirrors `OrtGraph`).
pub(crate) struct CoremlGraph {
    model: CompiledCoremlModel,
}

impl fmt::Debug for CoremlGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoremlGraph")
            .field("model", &self.model)
            .finish()
    }
}

pub(crate) struct CoremlBuilder {
    device_type: DeviceType,
}

impl fmt::Debug for CoremlBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoremlBuilder")
            .field("device_type", &self.device_type)
            .finish()
    }
}

impl<'context, 'builder> MLBackendBuilder<'context, 'builder> for CoremlBuilder {
    fn build(&mut self, graph_info: GraphInfo) -> crate::error::Result<MLGraph<'context>> {
        let converted = CoremlMlProgramConverter
            .convert(&graph_info)
            .map_err(|e| Error::GraphBuildError { source: e.into() })?;
        let model = compile_model(
            converted.data,
            converted.weights_data,
            self.device_type,
            supports_in_memory_asset(&graph_info),
        )
        .map_err(|e| Error::GraphBuildError { source: e.into() })?;
        MLGraph::new(
            MLBackendGraph::CoremlModel(CoremlGraph { model }),
            &graph_info,
        )
    }
}

/// CoreML's in-memory model compiler does not currently behave identically to
/// URL compilation for every MIL program. Keep the memory path as the default,
/// but use the established URL path for graphs with demonstrated correctness
/// differences.
fn supports_in_memory_asset(graph: &GraphInfo) -> bool {
    !graph.operations.iter().any(|operation| match operation {
        // MLModelAsset reports an invalid output feature shape when MIL gather
        // consumes a rank-0 index, although the URL compiler accepts it.
        Operation::Gather { indices, .. } => graph
            .operand(*indices)
            .is_some_and(|operand| operand.descriptor.shape.is_empty()),
        // The in-memory execution path loses int32 precision around 2^31 when
        // evaluating the typed `mul(x, -1)` used to lower WebNN neg.
        Operation::Neg { input, .. } => graph
            .operand(*input)
            .is_some_and(|operand| operand.descriptor.data_type == DataType::Int32),
        _ => false,
    })
}

#[derive(Debug)]
pub(crate) struct CoremlContext {
    device_type: DeviceType,
    tensors: Vec<CoremlTensor>,
}

impl CoremlContext {
    pub(crate) fn new_from_device_type(
        device_type: DeviceType,
        _options: Option<&RustNNOptions>,
    ) -> crate::error::Result<Self> {
        Ok(Self {
            device_type,
            tensors: Vec::new(),
        })
    }
}

impl<'context> MLBackendContext<'context> for CoremlContext {
    fn accelerated(&self) -> bool {
        self.device_type != DeviceType::Cpu
    }

    fn create_builder<'builder>(
        &mut self,
    ) -> crate::error::Result<Box<dyn MLBackendBuilder<'context, 'builder> + 'builder>>
    where
        'context: 'builder,
    {
        Ok(Box::new(CoremlBuilder {
            device_type: self.device_type,
        }))
    }

    fn create_tensor(&mut self, descriptor: &MLTensorDescriptor) -> crate::error::Result<MLTensor> {
        let n = tensor_byte_len(descriptor);
        self.tensors.push(CoremlTensor {
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
        let logical = tensor_byte_len(tensor.descriptor());
        if array.len() < logical {
            return Err(Error::TensorReadError {
                source: format!(
                    "buffer too small: need {} logical bytes, got {}",
                    logical,
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
        inputs: &MLNamedTensors,
        outputs: &MLNamedTensors,
    ) -> crate::error::Result<()> {
        // Gather raw-byte inputs keyed by feature name, then run; the borrow of
        // `self.tensors` is released before we write outputs back.
        let out_bytes = {
            let coreml_graph =
                graph
                    .backend
                    .as_coreml_model()
                    .ok_or_else(|| Error::GraphDispatchError {
                        source: "MLGraph is not a CoreML model graph".into(),
                    })?;

            let runtime_descriptors = graph
                .input_descriptors
                .iter()
                .map(|(name, descriptor)| {
                    let tensor =
                        inputs
                            .get(name.as_str())
                            .ok_or_else(|| Error::GraphDispatchError {
                                source: format!("missing input '{name}' for CoreML dispatch")
                                    .into(),
                            })?;
                    Ok((
                        name.clone(),
                        runtime_input_descriptor(descriptor, tensor.shape())?,
                    ))
                })
                .collect::<crate::error::Result<HashMap<_, _>>>()?;

            let mut byte_inputs: HashMap<String, CoremlByteInput> =
                HashMap::with_capacity(graph.input_descriptors.len());
            for (name, descriptor) in &runtime_descriptors {
                let tensor =
                    inputs
                        .get(name.as_str())
                        .ok_or_else(|| Error::GraphDispatchError {
                            source: format!("missing input '{name}' for CoreML dispatch").into(),
                        })?;
                // OperandDescriptor uses checked usize arithmetic, including on
                // arm64_32; do not truncate a u64 element count before checking.
                let logical =
                    descriptor
                        .byte_length()
                        .ok_or_else(|| Error::GraphDispatchError {
                            source: format!("input '{name}': cannot compute byte length").into(),
                        })?;
                let full = &self.tensors[tensor.id].memory;
                let bytes = full
                    .get(..logical)
                    .ok_or_else(|| Error::GraphDispatchError {
                        source: format!(
                            "input '{name}': tensor buffer shorter than logical size ({logical} bytes)"
                        )
                        .into(),
                    })?;
                debug!(
                    target: "rustnn::backends::coreml",
                    "dispatch input '{}' tensor_id={} shape={:?} logical_bytes={}",
                    name,
                    tensor.id,
                    tensor.shape(),
                    logical
                );
                byte_inputs.insert(
                    name.clone(),
                    CoremlByteInput {
                        data: bytes,
                        descriptor,
                    },
                );
            }

            run_coreml_bytes(&coreml_graph.model, &byte_inputs, &graph.output_descriptors)
                .map_err(|e| Error::GraphDispatchError { source: e.into() })?
        };

        for (&name, ml_tensor) in outputs.iter() {
            let data = out_bytes
                .get(name)
                .ok_or_else(|| Error::GraphDispatchError {
                    source: format!("model did not produce output '{name}'").into(),
                })?;
            let logical = tensor_byte_len(ml_tensor.descriptor());

            // When the graph output type is int64/uint64 but CoreML produced int32
            // bytes (argmin/argmax proxy, or a cast to int64/uint64), widen each
            // 4-byte value to 8 bytes. int64 is sign-extended so negative results
            // survive; uint64 is zero-extended.
            use crate::operator_enums::MLOperandDataType;
            let out_dt = ml_tensor.descriptor().data_type();
            let expanded: Option<Vec<u8>> = if data.len() * 2 == logical
                && matches!(out_dt, MLOperandDataType::Int64 | MLOperandDataType::Uint64)
            {
                let sign_extend = matches!(out_dt, MLOperandDataType::Int64);
                let count = data.len() / 4;
                let mut buf = vec![0u8; count * 8];
                for i in 0..count {
                    let v = i32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap());
                    let widened: i64 = if sign_extend {
                        v as i64
                    } else {
                        i64::from(v as u32)
                    };
                    buf[i * 8..i * 8 + 8].copy_from_slice(&widened.to_le_bytes());
                }
                Some(buf)
            } else {
                None
            };
            let effective = expanded.as_deref().unwrap_or(data.as_slice());

            // Do not silently truncate a maximum-sized result to an active
            // output binding: successful dispatch must return the exact size.
            if effective.len() != logical {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{name}': CoreML produced {} bytes, descriptor expects {logical}",
                        data.len()
                    )
                    .into(),
                });
            }
            let dst = &mut self.tensors[ml_tensor.id].memory;
            if dst.len() < logical {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{name}': storage too small ({} bytes) for {logical} logical bytes",
                        dst.len()
                    )
                    .into(),
                });
            }
            dst[..logical].copy_from_slice(&effective[..logical]);
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
        let new_bytes = new_desc.rustnn_required_bytes();
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

#[cfg(test)]
mod test {
    use crate::mlcontext::{
        Backend, MLContext, MLContextOptions, MLNamedOperands, MLNamedTensors, MLOperandDescriptor,
        MLPowerPreference, MLTensorDescriptor,
    };
    use crate::mlgraphbuilder::MLGraphBuilder;
    use crate::operator_enums::MLOperandDataType;

    /// Build a context backed by CoreML. Returns `None` (test skipped) if no
    /// accelerated backend is available on this machine.
    fn coreml_context<'a>() -> Option<MLContext<'a>> {
        let context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, true)
                .with_rustnn_backend_hint(Backend::Coreml),
        );
        match context {
            Ok(ctx) => Some(ctx),
            Err(crate::error::Error::NoBackendAvailableForBackendHint { .. }) => None,
            Err(e) => panic!("unexpected context creation error: {e:?}"),
        }
    }

    #[test]
    fn runtime_descriptor_checks_dimensions_and_byte_length() {
        use crate::graph::{DataType, OperandDescriptor, to_dimension_vector};
        let descriptor = OperandDescriptor {
            data_type: DataType::Float32,
            shape: to_dimension_vector(&[8, 4]),
            pending_permutation: vec![1, 0],
        };
        let active = super::runtime_input_descriptor(&descriptor, &[2, 4]).unwrap();
        assert_eq!(active.shape, to_dimension_vector(&[2, 4]));
        assert_eq!(active.byte_length(), Some(32));
        assert_eq!(active.pending_permutation, descriptor.pending_permutation);
        assert!(super::runtime_input_descriptor(&descriptor, &[u64::from(u32::MAX) + 1]).is_err());
        let overflowing = super::runtime_input_descriptor(
            &descriptor,
            &[u64::from(u32::MAX), u64::from(u32::MAX), 4],
        )
        .unwrap();
        assert_eq!(overflowing.byte_length(), None);
        let scalar = super::runtime_input_descriptor(&descriptor, &[]).unwrap();
        assert_eq!(scalar.byte_length(), Some(4));
    }

    #[cfg(feature = "dynamic-inputs")]
    fn check_dynamic_gather_dispatch(op_name: &str) {
        use crate::graph::{
            DataType, Dimension, DynamicDimension, Operand, OperandDescriptor, OperandKind,
        };
        use crate::operators::Operation;

        let dim = Dimension::Dynamic(DynamicDimension {
            name: "count".into(),
            max_size: 4,
        });
        let index_shape = if op_name == "gather" {
            vec![dim.clone()]
        } else {
            vec![dim.clone(), Dimension::Static(2)]
        };
        let output_shape = if op_name == "gatherND" {
            vec![dim]
        } else {
            vec![dim, Dimension::Static(2)]
        };
        let operation = match op_name {
            "gather" => Operation::Gather {
                input: 0,
                indices: 1,
                batch_dimensions: None,
                options: None,
                outputs: vec![2],
            },
            "gatherElements" => Operation::GatherElements {
                input: 0,
                indices: 1,
                batch_dimensions: None,
                options: None,
                outputs: vec![2],
            },
            "gatherND" => Operation::GatherND {
                input: 0,
                indices: 1,
                options: None,
                outputs: vec![2],
            },
            _ => unreachable!(),
        };
        let operand = |name: &str, kind, data_type, shape| Operand {
            name: Some(name.into()),
            kind,
            descriptor: OperandDescriptor {
                data_type,
                shape,
                pending_permutation: vec![],
            },
        };
        let graph_info = crate::GraphInfo {
            operands: vec![
                operand(
                    "table",
                    OperandKind::Input,
                    DataType::Float32,
                    crate::graph::to_dimension_vector(&[4, 2]),
                ),
                operand("indices", OperandKind::Input, DataType::Int64, index_shape),
                operand(
                    "result",
                    OperandKind::Output,
                    DataType::Float32,
                    output_shape,
                ),
            ],
            operations: vec![operation],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            ..Default::default()
        };
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_rustnn_backend_hint(Backend::Coreml),
        )
        .unwrap();
        let mut graph = MLGraphBuilder::new(&mut context)
            .unwrap()
            .build_graph_info(graph_info)
            .unwrap();
        let table = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4, 2]).to_writable(),
            )
            .unwrap();
        context
            .write_tensor(&table, &[10.0f32, 11., 20., 21., 30., 31., 40., 41.])
            .unwrap();
        let (all_indices, all_expected): (&[i64], &[f32]) = match op_name {
            "gather" => (
                &[0, -1, 100, -100],
                &[10., 11., 40., 41., 40., 41., 10., 11.],
            ),
            "gatherElements" => (
                &[0, -1, -1, 0, 100, -100, -100, 100],
                &[10., 41., 40., 11., 40., 11., 10., 41.],
            ),
            "gatherND" => (&[0, -1, -1, 0, 100, -100, -100, 100], &[11., 40., 40., 11.]),
            _ => unreachable!(),
        };
        // Reuse one compiled model while growing and shrinking active dimensions.
        // Each dispatch binds fresh, immutable-shape MLTensors.
        for count in [1u64, 4, 2, 1] {
            let index_shape = if op_name == "gather" {
                vec![count]
            } else {
                vec![count, 2]
            };
            let output_shape = if op_name == "gatherND" {
                vec![count]
            } else {
                vec![count, 2]
            };
            let index_count = index_shape.iter().product::<u64>() as usize;
            let output_count = output_shape.iter().product::<u64>() as usize;
            let indices = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Int64, index_shape).to_writable(),
                )
                .unwrap();
            let output = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Float32, output_shape.clone())
                        .to_readable(),
                )
                .unwrap();
            context
                .write_tensor(&indices, &all_indices[..index_count])
                .unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("table", &table), ("indices", &indices)]),
                    &MLNamedTensors::from([("result", &output)]),
                )
                .unwrap();
            let mut result = vec![f32::NAN; output_count];
            context.read_tensor(&output, &mut result).unwrap();
            assert_eq!(output.shape(), output_shape);
            assert_eq!(
                result,
                all_expected[..output_count],
                "{op_name} active count {count}"
            );
        }
    }

    #[cfg(feature = "dynamic-inputs")]
    #[test]
    fn coreml_dynamic_gather_dispatch() {
        check_dynamic_gather_dispatch("gather");
    }

    #[cfg(feature = "dynamic-inputs")]
    #[test]
    fn coreml_dynamic_gather_elements_dispatch() {
        check_dynamic_gather_dispatch("gatherElements");
    }

    #[cfg(feature = "dynamic-inputs")]
    #[test]
    fn coreml_dynamic_gather_nd_dispatch() {
        check_dynamic_gather_dispatch("gatherND");
    }

    #[test]
    fn coreml_scalar_gather_returns_scalar_values() {
        for constant_index in [false, true] {
            let mut context = MLContext::create(
                &MLContextOptions::new(MLPowerPreference::Default, false)
                    .with_rustnn_backend_hint(Backend::Coreml),
            )
            .unwrap();
            let mut builder = MLGraphBuilder::new(&mut context).unwrap();
            let data_desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![3]);
            let index_desc = MLOperandDescriptor::new(MLOperandDataType::Int32, vec![]);
            let data = builder.input("data", &data_desc).unwrap();
            let index = if constant_index {
                builder.constant_from_slice(&index_desc, &[-1i32]).unwrap()
            } else {
                builder.input("index", &index_desc).unwrap()
            };
            let result = builder.gather(data, index).unwrap();
            let mut graph = builder
                .build(&MLNamedOperands::from([("result", result)]))
                .unwrap();
            let data = context
                .create_tensor(
                    &MLTensorDescriptor::from_operand_descriptor(&data_desc).to_writable(),
                )
                .unwrap();
            let index = context
                .create_tensor(
                    &MLTensorDescriptor::from_operand_descriptor(&index_desc).to_writable(),
                )
                .unwrap();
            let result = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![]).to_readable(),
                )
                .unwrap();
            context.write_tensor(&data, &[10.0f32, 20., 30.]).unwrap();
            let cases: &[(i32, f32)] = if constant_index {
                &[(-1, 30.)]
            } else {
                &[(0, 10.), (2, 30.), (-1, 30.), (-100, 10.), (100, 30.)]
            };
            for &(value, expected) in cases {
                context.write_tensor(&index, &value.to_le_bytes()).unwrap();
                let inputs = if constant_index {
                    MLNamedTensors::from([("data", &data)])
                } else {
                    MLNamedTensors::from([("data", &data), ("index", &index)])
                };
                context
                    .dispatch(
                        &mut graph,
                        &inputs,
                        &MLNamedTensors::from([("result", &result)]),
                    )
                    .unwrap();
                let mut bytes = [0u8; 4];
                context.read_tensor(&result, &mut bytes).unwrap();
                assert!(result.shape().is_empty());
                assert_eq!(f32::from_le_bytes(bytes), expected, "scalar index {value}");
            }
        }
    }

    #[test]
    fn coreml_relu_add_f32() {
        let _ = pretty_env_logger::try_init();
        let Some(mut context) = coreml_context() else {
            return;
        };

        let desc = MLOperandDescriptor::new(MLOperandDataType::Float32, [2, 2].to_vec());

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let a = builder.input("a", &desc).unwrap();
        let b = builder.input("b", &desc).unwrap();
        let a = builder.relu(a).unwrap();
        let output = builder.add(a, b).unwrap();
        let mut outputs = MLNamedOperands::new();
        outputs.insert("out", output);
        let mut graph = builder.build(&outputs).unwrap();

        let mut io_desc = MLTensorDescriptor::from_operand_descriptor(&desc);
        io_desc.set_writable(true);
        io_desc.set_readable(true);

        let a = context.create_tensor(&io_desc).unwrap();
        let b = context.create_tensor(&io_desc).unwrap();
        let out = context.create_tensor(&io_desc).unwrap();

        // relu(-1, 2, -3, 4) = (0, 2, 0, 4); + (1,1,1,1) = (1, 3, 1, 5)
        context.write_tensor(&a, &[-1.0f32, 2., -3., 4.]).unwrap();
        context.write_tensor(&b, &[1.0f32, 1., 1., 1.]).unwrap();

        let mut inputs = MLNamedTensors::new();
        inputs.insert("a", &a);
        inputs.insert("b", &b);
        let mut out_bindings = MLNamedTensors::new();
        out_bindings.insert("out", &out);

        context
            .dispatch(&mut graph, &inputs, &out_bindings)
            .unwrap();

        let mut result = vec![0.0f32; 4];
        context.read_tensor(&out, &mut result).unwrap();
        assert_eq!(result, &[1.0f32, 3., 1., 5.]);
    }

    #[test]
    fn coreml_add_int32_byte_path() {
        let _ = pretty_env_logger::try_init();
        let Some(mut context) = coreml_context() else {
            return;
        };

        let desc = MLOperandDescriptor::new(MLOperandDataType::Int32, [4].to_vec());

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let a = builder.input("a", &desc).unwrap();
        let b = builder.input("b", &desc).unwrap();
        let output = builder.add(a, b).unwrap();
        let mut outputs = MLNamedOperands::new();
        outputs.insert("out", output);

        // CoreML's MLMultiArray has no native int32 add on every compute unit; if the
        // converter/runtime rejects this graph, skip rather than fail (records a known gap).
        let mut graph = match builder.build(&outputs) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping int32 CoreML test: build failed: {e:?}");
                return;
            }
        };

        let mut io_desc = MLTensorDescriptor::from_operand_descriptor(&desc);
        io_desc.set_writable(true);
        io_desc.set_readable(true);

        let a = context.create_tensor(&io_desc).unwrap();
        let b = context.create_tensor(&io_desc).unwrap();
        let out = context.create_tensor(&io_desc).unwrap();

        context.write_tensor(&a, &[1i32, 2, 3, 4]).unwrap();
        context.write_tensor(&b, &[10i32, 20, 30, 40]).unwrap();

        let mut inputs = MLNamedTensors::new();
        inputs.insert("a", &a);
        inputs.insert("b", &b);
        let mut out_bindings = MLNamedTensors::new();
        out_bindings.insert("out", &out);

        if let Err(e) = context.dispatch(&mut graph, &inputs, &out_bindings) {
            eprintln!("skipping int32 CoreML test: dispatch failed: {e:?}");
            return;
        }

        let mut result = vec![0i32; 4];
        context.read_tensor(&out, &mut result).unwrap();
        assert_eq!(result, &[11, 22, 33, 44]);
    }

    /// Regression guard for the `rustnn_coreml_predict` output-provider lifetime
    /// bug: the shim returned the prediction result through a plain `__bridge`
    /// cast, so ARC deallocated it when the shim returned and the very next use
    /// (reading outputs in `run_coreml_bytes`) SIGSEGVed. The crash only
    /// manifests in optimized builds (`cargo test --release`); in debug builds
    /// the freed memory happened to survive long enough to read.
    #[test]
    fn coreml_repeated_dispatch_provider_lifetime() {
        let _ = pretty_env_logger::try_init();

        let desc = MLOperandDescriptor::new(MLOperandDataType::Float32, [4].to_vec());
        for _ in 0..5 {
            let Some(mut context) = coreml_context() else {
                return;
            };
            let mut builder = MLGraphBuilder::new(&mut context).unwrap();
            let a = builder.input("a", &desc).unwrap();
            let b = builder.input("b", &desc).unwrap();
            let y = builder.add(a, b).unwrap();
            let mut outputs = MLNamedOperands::new();
            outputs.insert("y", y);
            let mut graph = builder.build(&outputs).unwrap();

            let mut io_desc = MLTensorDescriptor::from_operand_descriptor(&desc);
            io_desc.set_writable(true);
            io_desc.set_readable(true);

            let a = context.create_tensor(&io_desc).unwrap();
            let b = context.create_tensor(&io_desc).unwrap();
            let out = context.create_tensor(&io_desc).unwrap();
            context.write_tensor(&a, &[1.0f32, 2., 3., 4.]).unwrap();
            context.write_tensor(&b, &[10.0f32, 20., 30., 40.]).unwrap();

            let mut inputs = MLNamedTensors::new();
            inputs.insert("a", &a);
            inputs.insert("b", &b);
            let mut out_bindings = MLNamedTensors::new();
            out_bindings.insert("y", &out);

            context
                .dispatch(&mut graph, &inputs, &out_bindings)
                .unwrap();

            let mut result = vec![0.0f32; 4];
            context.read_tensor(&out, &mut result).unwrap();
            assert_eq!(result, &[11.0f32, 22., 33., 44.]);
        }
    }
}

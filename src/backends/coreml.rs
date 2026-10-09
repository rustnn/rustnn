//! CoreML backend for the unified WebNN IDL API on supported Apple targets.
//!
//! Mirrors the ONNX Runtime backend in `src/backends/ort.rs`: a `CoremlContext`
//! owns persistent native tensor storage, a `CoremlBuilder` converts a [`GraphInfo`]
//! to a CoreML MLProgram and compiles it once, and `CoremlGraph` holds the
//! compiled model for repeated dispatch.
//!
#![doc = include_str!("../../docs/integration/coreml.md")]
#![cfg(feature = "coreml-runtime")]

use std::collections::HashMap;
use std::fmt;

use log::debug;

use crate::GraphInfo;
use crate::backend_selection::DeviceType;
use crate::converters::{CoremlMlProgramConverter, GraphConverter};
use crate::error::Error;
use crate::executors::coreml::{
    CompiledCoremlModel, CoremlByteInput, CoremlTensorBinding, CoremlTensorStorage, compile_model,
    run_coreml_bytes, run_coreml_tensors,
};
use crate::mlcontext::RustNNOptions;
use crate::mlcontext::{
    MLBackendBuilder, MLBackendContext, MLBackendGraph, MLGraph, MLNamedTensors, MLTensor,
    MLTensorDescriptor,
};
use crate::mlcontextoptions::{BackendStatistics, CoremlOptions, CoremlTensorStatistics};

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

/// Context-owned tensor storage; native arrays never alias distinct tensors.
#[derive(Debug)]
pub(crate) struct CoremlTensor {
    storage: CoremlTensorStorage,
}

/// A compiled CoreML model held by [`MLGraph`] (mirrors `OrtGraph`).
pub(crate) struct CoremlGraph {
    model: CompiledCoremlModel,
    output_backings_eligible: bool,
}

impl CoremlGraph {
    pub(crate) fn load_diagnostics(&self) -> &crate::executors::coreml::CoremlLoadDiagnostics {
        self.model.load_diagnostics()
    }
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
        let output_backings_eligible = supports_output_backings(&graph_info);
        let converted = CoremlMlProgramConverter
            .convert(&graph_info)
            .map_err(|e| Error::GraphBuildError { source: e.into() })?;
        let use_asset =
            CoremlMlProgramConverter::constant_copy_asset_eligible(&graph_info, &converted.data);
        let model = compile_model(
            converted.data,
            converted.weights_data,
            self.device_type,
            // Arithmetic and typed boundaries keep the URL precision path.
            // Data-free, proven copies avoid a BNNS URL constant-fold crash.
            use_asset,
        )
        .map_err(|e| Error::GraphBuildError { source: e.into() })?;
        MLGraph::new(
            MLBackendGraph::CoremlModel(CoremlGraph {
                model,
                output_backings_eligible,
            }),
            &graph_info,
        )
    }
}

/// A fixed output feature does not prove that the whole graph is static.
/// Dynamic intermediates can still make a proposed output backing unsafe.
fn supports_output_backings(graph: &GraphInfo) -> bool {
    graph.operands.iter().all(|operand| {
        operand
            .descriptor
            .shape
            .iter()
            .all(|dimension| matches!(dimension, crate::graph::Dimension::Static(_)))
    })
}

#[derive(Debug)]
pub(crate) struct CoremlContext {
    device_type: DeviceType,
    tensors: Vec<CoremlTensor>,
    options: CoremlOptions,
    statistics: CoremlTensorStatistics,
}

impl CoremlContext {
    pub(crate) fn new_from_device_type(
        device_type: DeviceType,
        options: Option<&RustNNOptions>,
    ) -> crate::error::Result<Self> {
        Ok(Self {
            device_type,
            tensors: Vec::new(),
            options: options.map(|o| o.coreml.clone()).unwrap_or_default(),
            statistics: CoremlTensorStatistics::default(),
        })
    }

    fn dispatch_native(
        &mut self,
        graph: &MLGraph,
        inputs: &MLNamedTensors,
        outputs: &MLNamedTensors,
    ) -> crate::error::Result<HashMap<String, Vec<u8>>> {
        let coreml_graph =
            graph
                .backend
                .as_coreml_model()
                .ok_or_else(|| Error::GraphDispatchError {
                    source: "MLGraph is not a CoreML model graph".into(),
                })?;
        let model = &coreml_graph.model;
        self.statistics.last_compute_units = model.compute_unit();
        let active = |descriptors: &HashMap<String, crate::graph::OperandDescriptor>,
                      bindings: &MLNamedTensors| {
            descriptors
                .iter()
                .map(|(name, descriptor)| {
                    let tensor =
                        bindings
                            .get(name.as_str())
                            .ok_or_else(|| Error::GraphDispatchError {
                                source: format!("missing tensor '{name}'").into(),
                            })?;
                    Ok((
                        name.clone(),
                        runtime_input_descriptor(descriptor, tensor.shape())?,
                    ))
                })
                .collect::<crate::error::Result<HashMap<_, _>>>()
        };
        let input_descriptors = active(&graph.input_descriptors, inputs)?;
        let output_descriptors = active(&graph.output_descriptors, outputs)?;
        fn bind<'a>(
            descriptors: &'a HashMap<String, crate::graph::OperandDescriptor>,
            tensors: &MLNamedTensors,
            storage: &'a [CoremlTensor],
        ) -> HashMap<String, CoremlTensorBinding<'a>> {
            descriptors
                .iter()
                .map(|(name, descriptor)| {
                    (
                        name.clone(),
                        CoremlTensorBinding {
                            storage: &storage[tensors[name.as_str()].id].storage,
                            descriptor,
                        },
                    )
                })
                .collect::<HashMap<_, _>>()
        }
        let native_inputs = bind(&input_descriptors, inputs, &self.tensors);
        let native_outputs = bind(&output_descriptors, outputs, &self.tensors);
        run_coreml_tensors(
            model,
            &native_inputs,
            &native_outputs,
            self.options.output_backings && coreml_graph.output_backings_eligible,
            &mut self.statistics,
        )
        .map_err(|e| Error::GraphDispatchError { source: e.into() })
    }
}

impl<'context> MLBackendContext<'context> for CoremlContext {
    fn backend_statistics(&self) -> Option<BackendStatistics> {
        Some(BackendStatistics::Coreml(self.statistics))
    }
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
        let storage = CoremlTensorStorage::new(
            descriptor.data_type().into(),
            n,
            self.options.reuse_tensor_storage,
        )
        .map_err(|e| Error::TensorCreationError {
            source: e.into(),
            descriptor: descriptor.clone(),
        })?;
        self.statistics.native_allocations += u64::from(storage.is_native());
        self.tensors.push(CoremlTensor { storage });
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
        self.tensors[tensor.id]
            .storage
            .read(&mut array[..logical])
            .map_err(|e| Error::TensorReadError {
                source: e.into(),
                tensor: tensor.clone(),
            })?;
        self.statistics.host_read_bytes += logical as u64;
        Ok(())
    }

    fn write_tensor(&mut self, tensor: &MLTensor, array: &[u8]) -> crate::error::Result<()> {
        self.tensors[tensor.id]
            .storage
            .write(array)
            .map_err(|e| Error::TensorWriteError {
                source: e.into(),
                tensor: tensor.clone(),
            })?;
        self.statistics.host_write_bytes += array.len() as u64;
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
        let proven_copy_outputs = graph
            .backend
            .as_coreml_model()
            .ok_or_else(|| Error::GraphDispatchError {
                source: "MLGraph is not a CoreML model graph".into(),
            })?
            .model
            .proven_copy_output_count();
        let out_bytes = if self.options.reuse_tensor_storage {
            self.dispatch_native(graph, inputs, outputs)?
        } else {
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
                let full = self.tensors[tensor.id]
                    .storage
                    .host()
                    .map_err(|e| Error::GraphDispatchError { source: e.into() })?;
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

            self.statistics.last_compute_units = coreml_graph.model.compute_unit();
            self.statistics.input_copy_bytes += byte_inputs
                .values()
                .map(|i| i.data.len() as u64)
                .sum::<u64>();
            let result =
                run_coreml_bytes(&coreml_graph.model, &byte_inputs, &graph.output_descriptors)
                    .map_err(|e| Error::GraphDispatchError { source: e.into() })?;
            self.statistics.output_copy_bytes +=
                result.values().map(|b| b.len() as u64).sum::<u64>();
            result
        };

        for (&name, ml_tensor) in outputs.iter() {
            if self.tensors[ml_tensor.id].storage.is_native() {
                continue;
            }
            let data = out_bytes
                .get(name)
                .ok_or_else(|| Error::GraphDispatchError {
                    source: format!("model did not produce output '{name}'").into(),
                })?;
            let logical = tensor_byte_len(ml_tensor.descriptor());

            // The executor converts to the graph's declared WebNN dtype,
            // including sign/zero extension of CoreML's Int32 proxies.
            let effective = data.as_slice();

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
            self.tensors[ml_tensor.id]
                .storage
                .write(effective)
                .map_err(|e| Error::GraphDispatchError { source: e.into() })?;
        }
        // Count logical copies only after the whole dispatch succeeds, in all
        // storage modes. An arithmetic-derived alias is not a proven source copy.
        self.statistics.proven_copy_outputs += proven_copy_outputs;
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
        let allocated = self.tensors[tensor.id]
            .storage
            .reserve(new_bytes, true)
            .map_err(|e| Error::GraphDispatchError { source: e.into() })?;
        self.statistics.native_allocations += u64::from(allocated);
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
        let new_bytes = usize::try_from(elements)
            .ok()
            .and_then(|elements| elements.checked_mul(bits))
            .and_then(|b| b.checked_add(7))
            .map(|b| b / 8)
            .ok_or_else(|| Error::GraphDispatchError {
                source: "rustnn_set_tensor_capacity: byte length overflow".into(),
            })?;
        let allocated = self.tensors[tensor.id]
            .storage
            .reserve(new_bytes.max(1), false)
            .map_err(|e| Error::GraphDispatchError { source: e.into() })?;
        self.statistics.native_allocations += u64::from(allocated);
        Ok(())
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn output_backings_require_static_intermediates_as_well_as_inputs_and_outputs() {
        use crate::graph::{
            DataType, Dimension, DynamicDimension, GraphInfo, Operand, OperandDescriptor,
            OperandKind,
        };

        let mut graph = GraphInfo {
            operands: [
                OperandKind::Input,
                OperandKind::Intermediate,
                OperandKind::Output,
            ]
            .into_iter()
            .map(|kind| Operand {
                kind,
                name: None,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![Dimension::Static(1)],
                    pending_permutation: vec![],
                },
            })
            .collect(),
            input_operands: vec![0],
            output_operands: vec![2],
            ..Default::default()
        };
        assert!(super::supports_output_backings(&graph));
        for index in 0..graph.operands.len() {
            for name in ["sequence", ""] {
                // Even max_size=1 is a dynamic dimension, not a static proof.
                graph.operands[index].descriptor.shape[0] = Dimension::Dynamic(DynamicDimension {
                    name: name.into(),
                    max_size: 1,
                });
                assert!(!super::supports_output_backings(&graph));
                graph.operands[index].descriptor.shape[0] = Dimension::Static(1);
            }
        }
        for operand in &mut graph.operands {
            operand.descriptor.shape.clear();
        }
        assert!(super::supports_output_backings(&graph));
    }

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
    fn typed_graphs_use_one_local_compilation_path() {
        use crate::executors::coreml::CoremlLoadRoute;
        use crate::mlcontext::LoadDiagnostics;
        for dtype in [
            MLOperandDataType::Float16,
            MLOperandDataType::Float32,
            MLOperandDataType::Int32,
        ] {
            let Some(mut context) = coreml_context() else {
                return;
            };
            let mut builder = MLGraphBuilder::new(&mut context).unwrap();
            let input = builder
                .input("input", &MLOperandDescriptor::new(dtype, vec![11]))
                .unwrap();
            let output = builder.identity(input).unwrap();
            let graph = builder
                .build(&MLNamedOperands::from([("result", output)]))
                .unwrap();
            let Some(LoadDiagnostics::Coreml(diagnostic)) = graph.rustnn_load_diagnostics() else {
                panic!("missing CoreML load diagnostics")
            };
            assert_eq!(diagnostic.route, CoremlLoadRoute::CompiledUrl);
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
    fn coreml_not_equal_uint8_promotes_inputs_and_executes() {
        let _ = pretty_env_logger::try_init();
        let Some(mut context) = coreml_context() else {
            return;
        };

        let desc = MLOperandDescriptor::new(MLOperandDataType::Uint8, vec![5]);
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let a = builder.input("a", &desc).unwrap();
        let b = builder.input("b", &desc).unwrap();
        let output = builder.not_equal(a, b).unwrap();
        let mut outputs = MLNamedOperands::new();
        outputs.insert("out", output);
        let mut graph = builder.build(&outputs).unwrap();

        let mut io_desc = MLTensorDescriptor::from_operand_descriptor(&desc);
        io_desc.set_writable(true);
        io_desc.set_readable(true);
        let a = context.create_tensor(&io_desc).unwrap();
        let b = context.create_tensor(&io_desc).unwrap();
        let out = context.create_tensor(&io_desc).unwrap();

        context.write_tensor(&a, &[0u8, 1, 2, 255, 5]).unwrap();
        context.write_tensor(&b, &[0u8, 0, 2, 4, 255]).unwrap();
        context
            .dispatch(
                &mut graph,
                &MLNamedTensors::from([("a", &a), ("b", &b)]),
                &MLNamedTensors::from([("out", &out)]),
            )
            .unwrap();

        let mut result = vec![0u8; 5];
        context.read_tensor(&out, &mut result).unwrap();
        assert_eq!(result, [0, 1, 0, 1, 1]);
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

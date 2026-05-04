#![cfg(feature = "web")]

use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

use js_sys::Reflect;
use pollster::FutureExt as _;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;
use web_sys::{MlContext, MlOperandDataType, MlTensor, MlTensorDescriptor};

use crate::{
    GraphInfo,
    converters::webnn::build_ml_graph,
    error::{Error, Result},
    mlcontext::{
        MLBackendBuilder, MLBackendContext, MLBackendGraph, MLGraph, MLOperand, MLPowerPreference,
        MLTensor as RustMLTensor, MLTensorDescriptor as RustMLTensorDescriptor,
    },
    operator_enums::MLOperandDataType,
};

fn rust_dtype_to_ml_dtype(dtype: MLOperandDataType) -> MlOperandDataType {
    match dtype {
        MLOperandDataType::Float32 => MlOperandDataType::Float32,
        MLOperandDataType::Float16 => MlOperandDataType::Float16,
        MLOperandDataType::Int32 => MlOperandDataType::Int32,
        MLOperandDataType::Uint32 => MlOperandDataType::Uint32,
        MLOperandDataType::Int8 => MlOperandDataType::Int8,
        MLOperandDataType::Uint8 => MlOperandDataType::Uint8,
        MLOperandDataType::Int64 => MlOperandDataType::Int64,
        MLOperandDataType::Uint64 => MlOperandDataType::Uint64,
    }
}

/// Wrapper around `web_sys::MlGraph` that implements `Debug`.
pub(crate) struct WebNNGraph(pub(crate) web_sys::MlGraph);

impl fmt::Debug for WebNNGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("WebNNGraph").finish()
    }
}

/// Backend context that delegates to the browser's native WebNN implementation.
pub(crate) struct WebNNContext {
    context: MlContext,
    tensors: Vec<MlTensor>,
    accelerated: bool,
}

impl fmt::Debug for WebNNContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WebNNContext")
            .field("accelerated", &self.accelerated)
            .field("tensor_count", &self.tensors.len())
            .finish()
    }
}

impl WebNNContext {
    pub(crate) fn new(context: MlContext, accelerated: bool) -> Self {
        Self {
            context,
            tensors: Vec::new(),
            accelerated,
        }
    }

    pub(crate) fn from_options(options: &crate::mlcontext::MLContextOptions) -> Result<Self> {
        use web_sys::{MlContextOptions as WebMlContextOptions, MlPowerPreference, window};

        let web_opts = WebMlContextOptions::new();
        web_opts.set_accelerated(options.accelerated);
        let pref = match options.power_preference {
            MLPowerPreference::Default => MlPowerPreference::Default,
            MLPowerPreference::HighPerformance => MlPowerPreference::HighPerformance,
            MLPowerPreference::LowPower => MlPowerPreference::LowPower,
        };
        web_opts.set_power_preference(pref);

        let win = window().ok_or_else(|| Error::ContextCreationError {
            source: "no global window object".into(),
        })?;
        let ml = win.navigator().ml();
        let promise = ml.create_context_with_ml_context_options(&web_opts);
        let context = JsFuture::from(promise)
            .block_on()
            .map(MlContext::from)
            .map_err(|e| Error::ContextCreationError {
                source: format!("WebNN createContext failed: {e:?}").into(),
            })?;

        Ok(Self::new(context, options.accelerated))
    }
}

pub(crate) struct WebNNBuilder<'context> {
    context: MlContext,
    graph_info: Option<&'context GraphInfo>,
}

impl fmt::Debug for WebNNBuilder<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WebNNBuilder")
            .field("has_graph", &self.graph_info.is_some())
            .finish()
    }
}

impl<'context> MLBackendBuilder<'context> for WebNNBuilder<'context> {
    fn build(&mut self, _outputs: &HashMap<&str, MLOperand>) -> Result<MLGraph<'context>> {
        let graph_info = self.graph_info.ok_or_else(|| Error::GraphBuildError {
            source: "build() called before load_graph()".into(),
        })?;

        let ml_graph = build_ml_graph(&self.context, graph_info)
            .block_on()
            .map_err(|e| Error::GraphBuildError { source: e.into() })?;

        Ok(MLGraph {
            backend: MLBackendGraph::WebNNGraph(WebNNGraph(ml_graph), PhantomData),
        })
    }

    fn load_graph(&mut self, graph: &'context GraphInfo) -> Result<()> {
        self.graph_info = Some(graph);
        Ok(())
    }
}

impl<'context> MLBackendContext<'context> for WebNNContext {
    fn accelerated(&self) -> bool {
        self.accelerated
    }

    fn create_builder(&mut self) -> Result<Box<dyn MLBackendBuilder<'context> + 'context>> {
        Ok(Box::new(WebNNBuilder {
            context: self.context.clone(),
            graph_info: None,
        }))
    }

    fn create_tensor(&mut self, descriptor: &RustMLTensorDescriptor) -> Result<RustMLTensor> {
        let shape: Vec<js_sys::Number> = descriptor
            .shape()
            .iter()
            .map(|&d| js_sys::Number::from(d as f64))
            .collect();
        let desc = MlTensorDescriptor::new(rust_dtype_to_ml_dtype(descriptor.data_type()), &shape);
        desc.set_readable(descriptor.readable());
        desc.set_writable(descriptor.writable());

        // create_tensor returns Promise<MlTensor> directly (not Result)
        let promise = self.context.create_tensor(&desc);
        let ml_tensor = JsFuture::from(promise)
            .block_on()
            .map(MlTensor::from)
            .map_err(|e| Error::TensorCreationError {
                source: format!("createTensor rejected: {e:?}").into(),
                descriptor: descriptor.clone(),
            })?;

        self.tensors.push(ml_tensor);
        Ok(RustMLTensor {
            id: self.tensors.len() - 1,
            constant: false,
            descriptor: descriptor.clone(),
        })
    }

    fn create_constant_tensor(
        &mut self,
        descriptor: &RustMLTensorDescriptor,
        input_data: &[u8],
    ) -> Result<RustMLTensor> {
        // Constants need writable=true so we can upload the initial data.
        let mut desc = descriptor.clone();
        desc.set_writable(true);
        let mut tensor = self.create_tensor(&desc)?;
        tensor.constant = true;
        self.write_tensor(&tensor, input_data)
            .map_err(|e| Error::TensorCreationError {
                source: e.into(),
                descriptor: descriptor.clone(),
            })?;
        Ok(tensor)
    }

    fn read_tensor(&mut self, tensor: &RustMLTensor, array: &mut [u8]) -> Result<()> {
        let browser_tensor = &self.tensors[tensor.id];
        // read_tensor returns Promise<ArrayBuffer> directly (not Result)
        let promise = self.context.read_tensor(browser_tensor);
        let result = JsFuture::from(promise)
            .block_on()
            .map_err(|e| Error::TensorReadError {
                source: format!("readTensor rejected: {e:?}").into(),
                tensor: tensor.clone(),
            })?;
        let ab = js_sys::ArrayBuffer::from(result);
        let src = js_sys::Uint8Array::new(&ab);
        let data = src.to_vec();
        let copy_len = data.len().min(array.len());
        array[..copy_len].copy_from_slice(&data[..copy_len]);
        Ok(())
    }

    fn write_tensor(&mut self, tensor: &RustMLTensor, array: &[u8]) -> Result<()> {
        let browser_tensor = &self.tensors[tensor.id];
        let src = js_sys::Uint8Array::new_from_slice(array);
        // write_tensor_with_buffer_source returns () (sync, no Result)
        self.context
            .write_tensor_with_buffer_source(browser_tensor, &src);
        Ok(())
    }

    fn dispatch(
        &mut self,
        graph: &mut MLGraph,
        inputs: &HashMap<&str, &RustMLTensor>,
        outputs: &HashMap<&str, &RustMLTensor>,
    ) -> Result<()> {
        let webnn_graph =
            graph
                .backend
                .as_webnn_graph_mut()
                .ok_or_else(|| Error::GraphDispatchError {
                    source: "MLGraph is not a WebNN graph".into(),
                })?;

        // dispatch() expects Object<MlTensor> for named tensor maps
        let inputs_obj: js_sys::Object<MlTensor> = js_sys::Object::new_typed();
        for (&name, rust_tensor) in inputs {
            let browser_tensor = &self.tensors[rust_tensor.id];
            let _ = Reflect::set(&inputs_obj, &JsValue::from_str(name), browser_tensor).map_err(
                |e| Error::GraphDispatchError {
                    source: format!("failed to set input '{name}': {e:?}").into(),
                },
            )?;
        }

        let outputs_obj: js_sys::Object<MlTensor> = js_sys::Object::new_typed();
        for (&name, rust_tensor) in outputs {
            let browser_tensor = &self.tensors[rust_tensor.id];
            let _ = Reflect::set(&outputs_obj, &JsValue::from_str(name), browser_tensor).map_err(
                |e| Error::GraphDispatchError {
                    source: format!("failed to set output '{name}': {e:?}").into(),
                },
            )?;
        }

        // dispatch returns () (sync, queues work; readTensor after dispatch will block until done)
        self.context
            .dispatch(&webnn_graph.0, &inputs_obj, &outputs_obj);
        Ok(())
    }
}

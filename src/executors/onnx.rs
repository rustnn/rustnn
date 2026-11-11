//! One-shot ONNX Runtime execution of converted models (legacy; see [`crate::executors`]).
//!
//! [`run_onnx_with_inputs`] loads the model bytes (plus optional external weights) into a new
//! session on every call. [`run_onnx_with_inputs_checked`] additionally validates the inputs
//! against WebNN descriptors, and [`run_onnx_zeroed`] feeds zero inputs.
//! [`init_global_ort_env`] installs a custom environment shared with the ONNX backend.

#![cfg(feature = "onnx-runtime")]

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use log::{debug, info, warn};
use ndarray::{ArrayD, IxDyn};
use ort::environment::Environment;
use ort::session::SessionInputValue;

use half;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;

use crate::converters::ONNX_EXTERNAL_WEIGHTS_FILENAME;
use crate::error::GraphError;
use crate::graph::OperandDescriptor;
use crate::runtime_checks::{RuntimeShapeState, TensorKind, validate_shape_data_length};

// Cache initialization failures as well as the process-wide environment.
//
// Having only one environment per process is also a limitation of the ort crate:
// see https://github.com/pykeio/ort/blob/6ed2423cc08ca21d0b07f467b6e4387ef1bee2ce/src/environment.rs#L515-L517
type OrtInitialization = Result<Arc<Environment>, Arc<dyn std::error::Error + Send + Sync>>;
static ENVIRONMENT: RwLock<Option<OrtInitialization>> = RwLock::new(None);

/// An ONNX Runtime environment has already been installed in rustnn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("the global ONNX Runtime environment is already initialized")]
pub struct OrtAlreadyInitialized;

/// Installs a caller-configured ONNX Runtime environment for all rustnn sessions.
///
/// Register any EP libraries on `environment` before calling this function, and call
/// it before device discovery or creating an ONNX context. The first successful
/// initialization wins, including concurrent calls and rustnn's lazy initialization.
/// A cached lazy-initialization failure may be replaced with a supplied environment.
///
/// # Errors
/// Returns [`OrtAlreadyInitialized`] if an environment is already installed.
pub fn init_global_ort_env(environment: Arc<Environment>) -> Result<(), OrtAlreadyInitialized> {
    let mut state = ENVIRONMENT.write().unwrap_or_else(|e| e.into_inner());
    if matches!(*state, Some(Ok(_))) {
        return Err(OrtAlreadyInitialized);
    }
    *state = Some(Ok(environment));
    Ok(())
}

/// Returns whether rustnn has a successfully installed ONNX Runtime environment.
///
/// This does not initialize the runtime. The result is a snapshot; use
/// [`init_global_ort_env`] to atomically install an environment.
pub fn is_ort_initialized() -> bool {
    matches!(
        *ENVIRONMENT.read().unwrap_or_else(|e| e.into_inner()),
        Some(Ok(_))
    )
}

pub(crate) fn ensure_ort_initialized() -> Result<Arc<Environment>, GraphError> {
    let cached = ENVIRONMENT
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    if let Some(result) = cached {
        return result.map_err(|source| GraphError::OnnxRuntimeInitializationFailed { source });
    }
    let mut state = ENVIRONMENT.write().unwrap_or_else(|e| e.into_inner());
    state
        .get_or_insert_with(|| {
            info!("Loading onnxruntime");
            #[cfg(not(target_arch = "wasm32"))]
            let path = std::env::var_os("ORT_DYLIB_PATH")
                .filter(|path| !path.is_empty())
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| {
                    format!(
                        "{}onnxruntime{}",
                        std::env::consts::DLL_PREFIX,
                        std::env::consts::DLL_SUFFIX
                    )
                    .into()
                });
            // Load explicitly before any other ort API: implicit loading can panic.
            let result = (|| {
                #[cfg(not(target_arch = "wasm32"))]
                let builder = ort::init_from(path)
                    .map_err(|error| Arc::new(error) as Arc<dyn std::error::Error + Send + Sync>)?;
                #[cfg(target_arch = "wasm32")]
                let builder = ort::init();
                builder
                    .with_name("rustnn")
                    .with_execution_providers([ort::ep::CPU::default().build()])
                    .with_telemetry(false)
                    .build()
                    .map(Arc::new)
                    .map_err(|error| Arc::new(error) as Arc<dyn std::error::Error + Send + Sync>)
            })();
            match result {
                Ok(env) => {
                    info!("Loaded ONNX runtime");
                    Ok(env)
                }
                Err(error) => {
                    warn!("Error loading onnxruntime: {error}");
                    Err(error)
                }
            }
        })
        .clone()
        .map_err(|source| GraphError::OnnxRuntimeInitializationFailed { source })
}

/// Output metadata reported by [`run_onnx_zeroed`].
#[derive(Debug, Clone)]
pub struct OnnxOutput {
    /// ONNX output name.
    pub name: String,
    /// Shape as reported by ONNX Runtime.
    pub shape: Vec<i64>,
    /// ONNX element type name.
    pub data_type: String,
}

/// Tensor data for different types
pub enum TensorData {
    /// 32-bit floats.
    Float32(Vec<f32>),
    /// 16-bit floats as raw bits.
    Float16(Vec<u16>), // f16 stored as u16 bits
    /// Brain 16-bit floats as raw bits.
    Bfloat16(Vec<u16>),
    /// Finite E4M3 floats as raw bytes (ONNX FLOAT8E4M3FN).
    Float8e4m3(Vec<u8>),
    /// Signed 8-bit integers.
    Int8(Vec<i8>),
    /// Unsigned 8-bit integers.
    Uint8(Vec<u8>),
    /// Signed 32-bit integers.
    Int32(Vec<i32>),
    /// Unsigned 32-bit integers.
    Uint32(Vec<u32>),
    /// Signed 64-bit integers.
    Int64(Vec<i64>),
    /// Unsigned 64-bit integers.
    Uint64(Vec<u64>),
}

impl TensorData {
    fn len(&self) -> usize {
        match self {
            TensorData::Float32(v) => v.len(),
            TensorData::Float16(v) | TensorData::Bfloat16(v) => v.len(),
            TensorData::Float8e4m3(v) => v.len(),
            TensorData::Int8(v) => v.len(),
            TensorData::Uint8(v) => v.len(),
            TensorData::Int32(v) => v.len(),
            TensorData::Uint32(v) => v.len(),
            TensorData::Int64(v) => v.len(),
            TensorData::Uint64(v) => v.len(),
        }
    }
}

/// Input tensor data for ONNX execution
pub struct OnnxInput {
    /// ONNX input name.
    pub name: String,
    /// Shape of the data.
    pub shape: Vec<usize>,
    /// Typed element data.
    pub data: TensorData,
}

/// Output tensor with actual data
pub struct OnnxOutputWithData {
    /// ONNX output name.
    pub name: String,
    /// Shape of the output.
    pub shape: Vec<usize>,
    /// Values converted to `f64` (lossy for 64-bit integers; see the typed fields).
    pub data: Vec<f64>,
    /// Exact values for float32 outputs.
    pub float32_data: Option<Vec<f32>>,
    /// Exact values for boolean outputs.
    pub bool_data: Option<Vec<bool>>,
    /// Exact values for int64 outputs.
    pub int64_data: Option<Vec<i64>>,
    /// Exact values for uint64 outputs.
    pub uint64_data: Option<Vec<u64>>,
}

/// Load an ONNX model and run it once with zero-filled inputs; returns output metadata only.
pub fn run_onnx_zeroed(
    model_bytes: &[u8],
    _inputs: &HashMap<String, OperandDescriptor>,
) -> Result<Vec<OnnxOutput>, GraphError> {
    // Initialize ort global environment (only once per process)
    let env = ensure_ort_initialized()?;

    let mut session = Session::builder(&env)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("session builder failed: {e}"),
        })?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("set opt level failed: {e}"),
        })?
        .commit_from_memory(model_bytes)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("load model failed: {e}"),
        })?;

    // Build zero-filled inputs
    let mut input_values = Vec::new();
    for input_info in session.inputs().iter() {
        // Get shape from input type
        let shape: Vec<usize> = match input_info.dtype() {
            ort::value::ValueType::Tensor {
                ty: _,
                shape,
                dimension_symbols: _,
            } => shape.iter().map(|&d| d.max(1) as usize).collect(),
            _ => {
                return Err(GraphError::OnnxRuntimeFailed {
                    reason: format!("input '{}' is not a tensor", input_info.name()),
                });
            }
        };

        let total: usize = shape.iter().product();
        let zeros = vec![0f32; total.max(1)];

        // Convert shape to Vec<i64> for ort compatibility
        let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();

        let tensor = Value::from_array((shape_i64.as_slice(), zeros)).map_err(|e| {
            GraphError::OnnxRuntimeFailed {
                reason: format!(
                    "failed to create input tensor for {}: {e}",
                    input_info.name()
                ),
            }
        })?;
        input_values.push(tensor.into_dyn());
    }

    // Run inference - convert to Vec of SessionInputValue
    let input_session_values: Vec<SessionInputValue> = input_values
        .into_iter()
        .map(SessionInputValue::from)
        .collect();
    let outputs = session.run(input_session_values.as_slice()).map_err(|e| {
        GraphError::OnnxRuntimeFailed {
            reason: format!("run failed: {e}"),
        }
    })?;

    // Extract output metadata
    let mut results = Vec::new();
    for (idx, (_name, value)) in outputs.iter().enumerate() {
        // Get tensor shape and type
        let (shape, _data) =
            value
                .try_extract_tensor::<f32>()
                .map_err(|e| GraphError::OnnxRuntimeFailed {
                    reason: format!("failed to extract tensor: {e}"),
                })?;

        let shape_vec: Vec<i64> = shape.iter().copied().collect();
        results.push(OnnxOutput {
            name: format!("output_{idx}"),
            shape: shape_vec,
            data_type: "f32".to_string(),
        });
    }
    Ok(results)
}

/// Run ONNX model with actual input tensors and return output tensors with data.
///
/// ONNX Runtime performs its own shape checks. For validation against WebNN operand descriptors
/// (rank, static dimensions, dynamic `maxSize`, linked dynamic names), use
/// [`run_onnx_with_inputs_checked`].
pub fn run_onnx_with_inputs(
    model_bytes: &[u8],
    external_weights: Option<&[u8]>,
    inputs: Vec<OnnxInput>,
) -> Result<Vec<OnnxOutputWithData>, GraphError> {
    run_onnx_with_inputs_impl(model_bytes, external_weights, inputs, None, None)
}

/// Run an ONNX model from disk with actual input tensors.
///
/// Unlike the in-memory entry point, loading from a path lets ONNX Runtime
/// resolve standard external-data files relative to the model location.
pub fn run_onnx_path_with_inputs(
    model_path: impl AsRef<Path>,
    inputs: Vec<OnnxInput>,
) -> Result<Vec<OnnxOutputWithData>, GraphError> {
    let env = ensure_ort_initialized()?;
    let session = Session::builder(&env)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("session builder failed: {e}"),
        })?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("set opt level failed: {e}"),
        })?
        .commit_from_file(model_path)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("load model failed: {e}"),
        })?;
    run_onnx_session_with_inputs(session, inputs, None, None)
}

/// Same as [`run_onnx_with_inputs`], plus optional WebNN operand descriptor validation.
///
/// For each map that is `Some`, validates actual tensor shapes against that map (inputs before the
/// run, outputs after). `None` skips that side. When both maps are `Some`, same-named dynamic
/// dimensions are cross-checked (see `crate::runtime_checks::RuntimeShapeState`).
pub fn run_onnx_with_inputs_checked(
    model_bytes: &[u8],
    external_weights: Option<&[u8]>,
    inputs: Vec<OnnxInput>,
    input_descriptors: &HashMap<String, OperandDescriptor>,
    output_descriptors: &HashMap<String, OperandDescriptor>,
) -> Result<Vec<OnnxOutputWithData>, GraphError> {
    run_onnx_with_inputs_impl(
        model_bytes,
        external_weights,
        inputs,
        Some(input_descriptors),
        Some(output_descriptors),
    )
}

fn run_onnx_with_inputs_impl(
    model_bytes: &[u8],
    external_weights: Option<&[u8]>,
    inputs: Vec<OnnxInput>,
    input_descriptors: Option<&HashMap<String, OperandDescriptor>>,
    output_descriptors: Option<&HashMap<String, OperandDescriptor>>,
) -> Result<Vec<OnnxOutputWithData>, GraphError> {
    // Initialize ort global environment (only once per process)
    let env = ensure_ort_initialized()?;

    let mut builder = Session::builder(&env)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("session builder failed: {e}"),
        })?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("set opt level failed: {e}"),
        })?;
    if let Some(weights) = external_weights {
        builder = builder
            .with_external_initializer_file_in_memory(
                ONNX_EXTERNAL_WEIGHTS_FILENAME,
                Arc::from(weights),
            )
            .map_err(|e| GraphError::OnnxRuntimeFailed {
                reason: format!("set external initializer failed: {e}"),
            })?;
    }
    let session =
        builder
            .commit_from_memory(model_bytes)
            .map_err(|e| GraphError::OnnxRuntimeFailed {
                reason: format!("load model failed: {e}"),
            })?;

    run_onnx_session_with_inputs(session, inputs, input_descriptors, output_descriptors)
}

fn run_onnx_session_with_inputs(
    mut session: Session,
    inputs: Vec<OnnxInput>,
    input_descriptors: Option<&HashMap<String, OperandDescriptor>>,
    output_descriptors: Option<&HashMap<String, OperandDescriptor>>,
) -> Result<Vec<OnnxOutputWithData>, GraphError> {
    let output_names: Vec<String> = session
        // Extract output names for later use.
        .outputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();

    let mut runtime_shape_state = RuntimeShapeState::new();
    let mut actual_input_shapes = HashMap::new();
    for input in &inputs {
        validate_shape_data_length(&input.name, &input.shape, input.data.len())?;
        actual_input_shapes.insert(input.name.clone(), input.shape.clone());
    }
    if let Some(descriptors) = input_descriptors {
        runtime_shape_state.validate_named_shapes(
            &actual_input_shapes,
            descriptors,
            TensorKind::Input,
        )?;
    }

    // Build input tensors in the order the ONNX model expects (session.inputs()), looking up by name.
    // Callers may pass inputs in any order (e.g. alphabetical); matching by name avoids wrong mapping.
    let inputs_by_name: HashMap<String, OnnxInput> =
        inputs.into_iter().map(|i| (i.name.clone(), i)).collect();
    if log::log_enabled!(log::Level::Debug) {
        debug!("[ONNX] session.inputs() order (feed order to runtime):");
        for (idx, input_info) in session.inputs().iter().enumerate() {
            debug!("  {}: {}", idx, input_info.name());
        }
    }
    let mut input_session_values: Vec<SessionInputValue> = Vec::new();
    for input_info in session.inputs().iter() {
        let name = input_info.name().to_string();
        let input = inputs_by_name
            .get(&name)
            .ok_or_else(|| GraphError::OnnxRuntimeFailed {
                reason: format!("model expects input '{name}' but it was not provided"),
            })?;
        if log::log_enabled!(log::Level::Debug) {
            let shape_str = format!("{:?}", input.shape);
            let preview = match &input.data {
                TensorData::Float32(d) if d.len() <= 12 => format!("data={:?}", d),
                TensorData::Float32(d) => {
                    format!("len={} first4={:?}", d.len(), &d[..4.min(d.len())])
                }
                _ => "".to_string(),
            };
            debug!("  [ONNX] feeding {} shape={} {}", name, shape_str, preview);
        }
        let session_value = match &input.data {
            TensorData::Float32(data) => {
                let value =
                    if input.shape.contains(&0) {
                        // ort tuple-shape API rejects 0-sized dimensions; ndarray path supports them.
                        let array = ArrayD::from_shape_vec(IxDyn(&input.shape), data.clone())
                            .map_err(|e| GraphError::OnnxRuntimeFailed {
                                reason: format!(
                                    "failed to create float32 ndarray input tensor for {}: {e}",
                                    input.name
                                ),
                            })?;
                        Value::from_array(array).map_err(|e| GraphError::OnnxRuntimeFailed {
                            reason: format!(
                                "failed to create float32 input tensor for {} via ndarray: {e}",
                                input.name
                            ),
                        })?
                    } else {
                        // Convert shape to i64 for ort compatibility
                        let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                        Value::from_array((shape_i64.clone(), data.clone())).map_err(|e| {
                            GraphError::OnnxRuntimeFailed {
                                reason: format!(
                                    "failed to create float32 input tensor for {}: {e}",
                                    input.name
                                ),
                            }
                        })?
                    };
                SessionInputValue::from(value)
            }
            TensorData::Bfloat16(_) | TensorData::Float8e4m3(_) => {
                return Err(GraphError::OnnxRuntimeFailed {
                    reason: "BF16 and FP8 tensor I/O is not supported yet".into(),
                });
            }
            TensorData::Float16(data) => {
                // Convert u16 bits to half::f16
                let f16_data: Vec<half::f16> = data
                    .iter()
                    .map(|&bits| half::f16::from_bits(bits))
                    .collect();
                // Convert shape to i64 for ort compatibility
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.as_slice(), f16_data)).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create float16 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Int8(data) => {
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.clone(), data.clone())).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create int8 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Uint8(data) => {
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                match input_info.dtype() {
                    ort::value::ValueType::Tensor {
                        ty: ort::value::TensorElementType::Bool,
                        ..
                    } => {
                        let bool_data: Vec<bool> = data.iter().map(|&v| v != 0).collect();
                        Value::from_array((shape_i64.clone(), bool_data))
                            .map_err(|e| GraphError::OnnxRuntimeFailed {
                                reason: format!(
                                    "failed to create bool input tensor for {}: {e}",
                                    input.name
                                ),
                            })?
                            .into()
                    }
                    _ => Value::from_array((shape_i64.clone(), data.clone()))
                        .map_err(|e| GraphError::OnnxRuntimeFailed {
                            reason: format!(
                                "failed to create uint8 input tensor for {}: {e}",
                                input.name
                            ),
                        })?
                        .into(),
                }
            }
            TensorData::Int32(data) => {
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.clone(), data.clone())).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create int32 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Uint32(data) => {
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.clone(), data.clone())).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create uint32 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Int64(data) => {
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.clone(), data.clone())).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create int64 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Uint64(data) => {
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.clone(), data.clone())).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create uint64 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
        };

        input_session_values.push(session_value);
    }

    // Run inference
    let outputs = session.run(input_session_values.as_slice()).map_err(|e| {
        GraphError::OnnxRuntimeFailed {
            reason: format!("run failed: {e}"),
        }
    })?;

    // Extract output tensors with data
    let mut results = Vec::new();
    for (idx, (_name, value)) in outputs.iter().enumerate() {
        let name = output_names
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("output_{}", idx));

        // Try to extract tensor with different types
        // The order matches most common types first for performance
        let (shape_vec, data_vec, float32_data, bool_data, int64_data, uint64_data) =
            if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
                let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data_vec: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                (shape_vec, data_vec, Some(data.to_vec()), None, None, None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<half::f16>() {
                let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data_vec: Vec<f64> = data.iter().map(|&x| x.to_f32() as f64).collect();
                (shape_vec, data_vec, None, None, None, None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<i32>() {
                let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data_vec: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                (shape_vec, data_vec, None, None, None, None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<u32>() {
                let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data_vec: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                (shape_vec, data_vec, None, None, None, None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<i8>() {
                let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data_vec: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                (shape_vec, data_vec, None, None, None, None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<u8>() {
                let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data_vec: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                (shape_vec, data_vec, None, None, None, None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<bool>() {
                let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data_vec: Vec<f64> = data.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect();
                (shape_vec, data_vec, None, Some(data.to_vec()), None, None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
                let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data_vec: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                (shape_vec, data_vec, None, None, Some(data.to_vec()), None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<u64>() {
                let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data_vec: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                (shape_vec, data_vec, None, None, None, Some(data.to_vec()))
            } else {
                return Err(GraphError::OnnxRuntimeFailed {
                    reason: "failed to extract output tensor: unsupported data type".to_string(),
                });
            };

        results.push(OnnxOutputWithData {
            name,
            shape: shape_vec,
            data: data_vec,
            float32_data,
            bool_data,
            int64_data,
            uint64_data,
        });
    }

    if let Some(descriptors) = output_descriptors {
        let mut actual_output_shapes = HashMap::new();
        for output in &results {
            actual_output_shapes.insert(output.name.clone(), output.shape.clone());
        }
        runtime_shape_state.validate_named_shapes(
            &actual_output_shapes,
            descriptors,
            TensorKind::Output,
        )?;
    }

    Ok(results)
}

/// Device-resident tensor implementation for ONNX Runtime
///
/// This enables zero-copy execution by keeping tensors on device (GPU/NPU)
/// across multiple inference steps, eliminating host round-trips for
/// iterative workloads like KV cache in GenAI models.
#[derive(Debug)]
pub struct OrtDeviceTensor {
    /// ONNX Runtime value stored on device
    value: Value,
    /// Session reference to keep it alive
    _session: std::sync::Arc<Session>,
    /// Data type
    dtype: crate::graph::DataType,
    /// Tensor shape
    shape: Vec<usize>,
    /// Device kind (CPU, CUDA, etc.)
    device: crate::tensor::DeviceKind,
}

impl OrtDeviceTensor {
    /// Create a new device tensor with the given shape and data type
    pub fn new(
        session: std::sync::Arc<Session>,
        shape: Vec<usize>,
        dtype: crate::graph::DataType,
        device: crate::tensor::DeviceKind,
    ) -> Result<Self, GraphError> {
        // Convert shape to i64 for ort compatibility
        let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();

        // Create zero-filled tensor on device
        // Currently only supporting f32, will expand to other types
        let total_elements: usize = shape.iter().product();
        let zeros = vec![0.0f32; total_elements.max(1)];

        let value = Value::from_array((shape_i64.as_slice(), zeros))
            .map_err(|e| GraphError::DeviceTensorFailed {
                reason: format!("failed to create device tensor: {e}"),
            })?
            .into();

        Ok(Self {
            value,
            _session: session,
            dtype,
            shape,
            device,
        })
    }

    /// Get a reference to the underlying ORT value
    pub fn value(&self) -> &Value {
        &self.value
    }

    /// Get a mutable reference to the underlying ORT value
    pub fn value_mut(&mut self) -> &mut Value {
        &mut self.value
    }
}

impl crate::tensor::DeviceTensorBackend for OrtDeviceTensor {
    fn dtype(&self) -> crate::graph::DataType {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn device_kind(&self) -> crate::tensor::DeviceKind {
        self.device
    }

    fn backend_kind(&self) -> crate::tensor::BackendKind {
        match self.device {
            crate::tensor::DeviceKind::Cpu => crate::tensor::BackendKind::OnnxCpu,
            crate::tensor::DeviceKind::Cuda => crate::tensor::BackendKind::OnnxGpu,
            crate::tensor::DeviceKind::DirectML => crate::tensor::BackendKind::OnnxGpu,
            crate::tensor::DeviceKind::CoreML => crate::tensor::BackendKind::OnnxCpu,
        }
    }

    fn read_to_host(&self) -> Result<Vec<f32>, GraphError> {
        // Extract tensor data from device to host
        // Currently only supporting f32, will expand to other types
        match self.dtype {
            crate::graph::DataType::Float32 => {
                let (_, data) = self.value.try_extract_tensor::<f32>().map_err(|e| {
                    GraphError::DeviceTensorFailed {
                        reason: format!("failed to read f32 tensor from device: {e}"),
                    }
                })?;
                Ok(data.to_vec())
            }
            crate::graph::DataType::Float16 => {
                let (_, data) = self.value.try_extract_tensor::<half::f16>().map_err(|e| {
                    GraphError::DeviceTensorFailed {
                        reason: format!("failed to read f16 tensor from device: {e}"),
                    }
                })?;
                // Convert f16 to f32
                Ok(data.iter().map(|&x| x.to_f32()).collect())
            }
            crate::graph::DataType::Int32 => {
                let (_, data) = self.value.try_extract_tensor::<i32>().map_err(|e| {
                    GraphError::DeviceTensorFailed {
                        reason: format!("failed to read i32 tensor from device: {e}"),
                    }
                })?;
                // Convert i32 to f32
                Ok(data.iter().map(|&x| x as f32).collect())
            }
            _ => Err(GraphError::DeviceTensorFailed {
                reason: format!("unsupported data type for device tensor: {:?}", self.dtype),
            }),
        }
    }

    fn write_from_host(&mut self, data: &[f32]) -> Result<(), GraphError> {
        // Write tensor data from host to device
        // Note: ORT doesn't support in-place writes, so we recreate the value
        let expected_size: usize = self.shape.iter().product();
        if data.len() != expected_size {
            return Err(GraphError::DeviceTensorFailed {
                reason: format!(
                    "data size mismatch: expected {} elements, got {}",
                    expected_size,
                    data.len()
                ),
            });
        }

        // Convert shape to i64 for ort compatibility
        let shape_i64: Vec<i64> = self.shape.iter().map(|&d| d as i64).collect();

        // Create new value from host data
        // Currently only supporting f32, will expand to other types
        match self.dtype {
            crate::graph::DataType::Float32 => {
                self.value = Value::from_array((shape_i64.as_slice(), data.to_vec()))
                    .map_err(|e| GraphError::DeviceTensorFailed {
                        reason: format!("failed to write f32 tensor to device: {e}"),
                    })?
                    .into();
                Ok(())
            }
            crate::graph::DataType::Float16 => {
                // Convert f32 to f16
                let f16_data: Vec<half::f16> =
                    data.iter().map(|&x| half::f16::from_f32(x)).collect();
                self.value = Value::from_array((shape_i64.as_slice(), f16_data))
                    .map_err(|e| GraphError::DeviceTensorFailed {
                        reason: format!("failed to write f16 tensor to device: {e}"),
                    })?
                    .into();
                Ok(())
            }
            crate::graph::DataType::Int32 => {
                // Convert f32 to i32
                let i32_data: Vec<i32> = data.iter().map(|&x| x as i32).collect();
                self.value = Value::from_array((shape_i64.as_slice(), i32_data))
                    .map_err(|e| GraphError::DeviceTensorFailed {
                        reason: format!("failed to write i32 tensor to device: {e}"),
                    })?
                    .into();
                Ok(())
            }
            _ => Err(GraphError::DeviceTensorFailed {
                reason: format!("unsupported data type for device tensor: {:?}", self.dtype),
            }),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Run ONNX model with device tensor bindings (zero-copy execution)
///
/// This function uses ONNX Runtime IoBinding to execute the model with
/// device-resident tensors, eliminating host-device round-trips.
///
/// Note: This is a placeholder implementation. Full IoBinding support
/// will be added in a future update.
#[allow(dead_code)]
pub fn run_onnx_with_bindings(
    _session: &Session,
    _input_bindings: Vec<(&str, &OrtDeviceTensor)>,
    _output_bindings: Vec<(&str, &mut OrtDeviceTensor)>,
) -> Result<(), GraphError> {
    // Placeholder for future IoBinding implementation
    // Current dispatch() implementation uses regular compute path
    Err(GraphError::DeviceTensorFailed {
        reason: "IoBinding not yet implemented - use dispatch() instead".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_environment_initialization() {
        const CHILD: &str = "RUSTNN_TEST_CUSTOM_ORT_CHILD";
        if let Some(mode) = std::env::var_os(CHILD) {
            let env = if mode == "lazy" {
                ensure_ort_initialized().unwrap()
            } else {
                Arc::new(
                    ort::init_from(std::env::var_os("ORT_DYLIB_PATH").unwrap())
                        .unwrap()
                        .with_name("custom")
                        .build()
                        .unwrap(),
                )
            };
            if mode == "lazy" {
                let original = ensure_ort_initialized().unwrap();
                assert_eq!(init_global_ort_env(env), Err(OrtAlreadyInitialized));
                assert!(Arc::ptr_eq(&original, &ensure_ort_initialized().unwrap()));
            } else {
                if mode == "failure" {
                    *ENVIRONMENT.write().unwrap() = Some(Err(Arc::new(std::io::Error::other(
                        "previous load failure",
                    ))));
                }
                let barrier = std::sync::Barrier::new(8);
                let successes = std::thread::scope(|scope| {
                    let handles: Vec<_> = (0..8)
                        .map(|_| {
                            let env = env.clone();
                            let barrier = &barrier;
                            scope.spawn(move || {
                                barrier.wait();
                                init_global_ort_env(env).is_ok()
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| usize::from(h.join().unwrap()))
                        .sum::<usize>()
                });
                assert_eq!(successes, 1);
                assert!(Arc::ptr_eq(&env, &ensure_ort_initialized().unwrap()));
            }
            return;
        }
        // This test needs a real runtime; the failure test below does not.
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            return;
        }
        for mode in ["custom", "lazy", "failure"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "executors::onnx::tests::test_custom_environment_initialization",
                    "--nocapture",
                ])
                .env(CHILD, mode)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{mode}: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[test]
    fn test_missing_runtime_returns_consistent_errors() {
        const CHILD: &str = "RUSTNN_TEST_MISSING_ORT_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let first = ensure_ort_initialized().expect_err("missing library");
            let second = ensure_ort_initialized().expect_err("cached failure");
            assert_eq!(
                crate::error::format_error_chain(&first),
                crate::error::format_error_chain(&second)
            );
            assert!(crate::error::format_error_chain(&first).contains("missing-onnxruntime"));
            use crate::mlcontext::ListDevices;
            assert!(crate::backends::ort::OrtContext::list_devices().is_empty());
            use crate::mlcontext::{Backend, MLContext, MLContextOptions, MLPowerPreference};
            let options = MLContextOptions::new(MLPowerPreference::Default, false)
                .with_rustnn_backend_hint(Backend::Onnx);
            let error =
                MLContext::create(&options).expect_err("missing runtime must fail selection");
            assert!(matches!(
                error,
                crate::error::Error::NoBackendAvailableForBackendHint { .. }
            ));
            assert!(
                crate::error::format_error_chain(&error).contains("missing-onnxruntime"),
                "{error}"
            );
            let automatic =
                MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, false));
            #[cfg(feature = "burn-cpu-runtime")]
            assert!(
                automatic.is_ok(),
                "automatic selection must fall back to Burn CPU"
            );
            #[cfg(not(any(
                feature = "burn-cpu-runtime",
                feature = "coreml-runtime",
                feature = "litert-runtime"
            )))]
            assert!(
                crate::error::format_error_chain(&automatic.expect_err("no runtime available"))
                    .contains("missing-onnxruntime")
            );
            return;
        }

        // Isolate process-global ort state and avoid mutating this process's environment.
        let dir = tempfile::tempdir().unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "executors::onnx::tests::test_missing_runtime_returns_consistent_errors",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("ORT_DYLIB_PATH", dir.path().join("missing-onnxruntime"))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "child failed: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

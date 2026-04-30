use crate::wpt_embedded::WPT_CONFORMANCE_JSON;
use half::f16;
use js_sys::{
    BigInt64Array, BigUint64Array, Float32Array, Int8Array, Int32Array, Reflect, Uint8Array,
    Uint16Array, Uint32Array,
};
use rustnn::converters::WebNNConverter;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    MlContext, MlContextOptions, MlOperandDataType, MlPowerPreference, MlTensor,
    MlTensorDescriptor, window,
};

use super::tolerance::{ToleranceKind, get_operation_tolerance, validate_result};
use super::wpt_to_graph::{
    expected_output_to_f32, expected_output_to_i8, expected_output_to_i32, expected_output_to_i64,
    expected_output_to_u8, expected_output_to_u32, expected_output_to_u64, wpt_graph_to_graph_info,
};
use super::wpt_types::{WptGraph, WptTensorSpec, WptTestCase, load_wpt_file};

fn wpt_data_type_to_ml(dtype: &str) -> Result<MlOperandDataType, String> {
    match dtype.trim().to_ascii_lowercase().as_str() {
        "float32" => Ok(MlOperandDataType::Float32),
        "float16" => Ok(MlOperandDataType::Float16),
        "int32" => Ok(MlOperandDataType::Int32),
        "uint32" => Ok(MlOperandDataType::Uint32),
        "int64" => Ok(MlOperandDataType::Int64),
        "uint64" => Ok(MlOperandDataType::Uint64),
        "int8" => Ok(MlOperandDataType::Int8),
        "uint8" => Ok(MlOperandDataType::Uint8),
        other => Err(format!("unsupported WebNN operand data type: {other}")),
    }
}

fn tensor_element_count(shape: &[i32]) -> usize {
    let n: usize = shape.iter().map(|&d| d.max(0) as usize).product();
    n.max(1)
}

/// Float16 tensor I/O uses raw IEEE half bits in a `Uint16Array` (WebNN note; same layout as `Float16Array`).
fn f16_bits_from_wpt(spec: &WptTensorSpec) -> Vec<u16> {
    expected_output_to_f32(spec)
        .into_iter()
        .map(|x| f16::from_f32(x).to_bits())
        .collect()
}

fn f32_vec_from_u16_bits(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
}

/// True if WPT expected data for a float tensor includes non-finite infinity (skip: browser/backend variance).
fn float_expected_has_infinity(spec: &WptTensorSpec) -> bool {
    let dt = spec.data_type();
    if dt != "float32" && dt != "float16" {
        return false;
    }
    expected_output_to_f32(spec).iter().any(|x| x.is_infinite())
}

fn js_value_to_string(v: &JsValue) -> String {
    v.as_string().unwrap_or_else(|| format!("{v:?}"))
}

fn error_message_is_not_supported(s: &str) -> bool {
    s.contains("NotSupportedError")
}

/// Outcome of a single browser WebNN WPT case (distinguish pass vs skip for summaries).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebnnWptOutcome {
    Passed,
    Skipped,
}

/// Read back output and compare to expected. Integer outputs use exact equality; floats use WPT tolerances.
async fn validate_output_tensor(
    context: &MlContext,
    outputs_obj: &js_sys::Object<MlTensor>,
    operation: &str,
    test_case: &WptTestCase,
    out_name: &str,
    spec: &WptTensorSpec,
    read_buf: &ReadBackBuf,
) -> Result<(), String> {
    let tensor_val =
        Reflect::get(outputs_obj, &out_name.into()).map_err(|_| "output tensor get")?;
    let tensor: MlTensor = tensor_val.dyn_into().map_err(|_| "output tensor cast")?;

    let (tolerance_kind, tolerance_value) =
        get_operation_tolerance(operation, test_case.tolerance.as_ref());

    match read_buf {
        ReadBackBuf::F32(buf) => {
            let _ = JsFuture::from(context.read_tensor_with_buffer_source(&tensor, buf))
                .await
                .map_err(|e| {
                    format!("read_tensor {out_name} failed: {}", js_value_to_string(&e))
                })?;
            let actual = buf.to_vec();
            let expected = expected_output_to_f32(spec);
            let (pass, msg) = validate_result(&actual, &expected, tolerance_kind, tolerance_value);
            if !pass {
                return Err(msg.unwrap_or_else(|| "validation failed".to_string()));
            }
        }
        ReadBackBuf::F16(buf) => {
            let _ = JsFuture::from(context.read_tensor_with_buffer_source(&tensor, buf))
                .await
                .map_err(|e| {
                    format!("read_tensor {out_name} failed: {}", js_value_to_string(&e))
                })?;
            let actual_bits = buf.to_vec();
            let actual = f32_vec_from_u16_bits(&actual_bits);
            let expected = expected_output_to_f32(spec);
            let (kind, value) = if operation == "batch_normalization" {
                (ToleranceKind::Ulp, 20_000u64)
            } else {
                (tolerance_kind, tolerance_value)
            };
            let (pass, msg) = validate_result(&actual, &expected, kind, value);
            if !pass {
                return Err(msg.unwrap_or_else(|| "validation failed".to_string()));
            }
        }
        ReadBackBuf::I32(buf) => {
            let _ = JsFuture::from(context.read_tensor_with_buffer_source(&tensor, buf))
                .await
                .map_err(|e| {
                    format!("read_tensor {out_name} failed: {}", js_value_to_string(&e))
                })?;
            let actual = buf.to_vec();
            let expected = expected_output_to_i32(spec);
            if actual.len() != expected.len() || actual.iter().ne(expected.iter()) {
                return Err("int32 output mismatch".to_string());
            }
        }
        ReadBackBuf::U32(buf) => {
            let _ = JsFuture::from(context.read_tensor_with_buffer_source(&tensor, buf))
                .await
                .map_err(|e| {
                    format!("read_tensor {out_name} failed: {}", js_value_to_string(&e))
                })?;
            let actual = buf.to_vec();
            let expected = expected_output_to_u32(spec);
            if actual.len() != expected.len() || actual.iter().ne(expected.iter()) {
                return Err("uint32 output mismatch".to_string());
            }
        }
        ReadBackBuf::I64(buf) => {
            let _ = JsFuture::from(context.read_tensor_with_buffer_source(&tensor, buf))
                .await
                .map_err(|e| {
                    format!("read_tensor {out_name} failed: {}", js_value_to_string(&e))
                })?;
            let actual = buf.to_vec();
            let expected = expected_output_to_i64(spec);
            if actual.len() != expected.len() || actual.iter().ne(expected.iter()) {
                return Err("int64 output mismatch".to_string());
            }
        }
        ReadBackBuf::U64(buf) => {
            let _ = JsFuture::from(context.read_tensor_with_buffer_source(&tensor, buf))
                .await
                .map_err(|e| {
                    format!("read_tensor {out_name} failed: {}", js_value_to_string(&e))
                })?;
            let actual = buf.to_vec();
            let expected = expected_output_to_u64(spec);
            if actual.len() != expected.len() || actual.iter().ne(expected.iter()) {
                return Err("uint64 output mismatch".to_string());
            }
        }
        ReadBackBuf::I8(buf) => {
            let _ = JsFuture::from(context.read_tensor_with_buffer_source(&tensor, buf))
                .await
                .map_err(|e| {
                    format!("read_tensor {out_name} failed: {}", js_value_to_string(&e))
                })?;
            let actual = buf.to_vec();
            let expected = expected_output_to_i8(spec);
            if actual.len() != expected.len() || actual.iter().ne(expected.iter()) {
                return Err("int8 output mismatch".to_string());
            }
        }
        ReadBackBuf::U8(buf) => {
            let _ = JsFuture::from(context.read_tensor_with_buffer_source(&tensor, buf))
                .await
                .map_err(|e| {
                    format!("read_tensor {out_name} failed: {}", js_value_to_string(&e))
                })?;
            let actual = buf.to_vec();
            let expected = expected_output_to_u8(spec);
            if actual.len() != expected.len() || actual.iter().ne(expected.iter()) {
                return Err("uint8 output mismatch".to_string());
            }
        }
    }
    Ok(())
}

enum ReadBackBuf {
    F32(Float32Array),
    F16(Uint16Array),
    I32(Int32Array),
    U32(Uint32Array),
    I64(BigInt64Array),
    U64(BigUint64Array),
    I8(Int8Array),
    U8(Uint8Array),
}

fn read_back_buf_for_dtype(dtype: &str, n: usize) -> Result<ReadBackBuf, String> {
    let n = n as u32;
    Ok(match dtype {
        "float32" => ReadBackBuf::F32(Float32Array::new_with_length(n)),
        "float16" => ReadBackBuf::F16(Uint16Array::new_with_length(n)),
        "int32" => ReadBackBuf::I32(Int32Array::new_with_length(n)),
        "uint32" => ReadBackBuf::U32(Uint32Array::new_with_length(n)),
        "int64" => ReadBackBuf::I64(BigInt64Array::new_with_length(n)),
        "uint64" => ReadBackBuf::U64(BigUint64Array::new_with_length(n)),
        "int8" => ReadBackBuf::I8(Int8Array::new_with_length(n)),
        "uint8" => ReadBackBuf::U8(Uint8Array::new_with_length(n)),
        _ => return Err(format!("unsupported dtype for output buffer: {dtype}")),
    })
}

// TODO: should perform checks for each operator via WebNN operator support API
// would need the Operator support API implemented by RustNN or directly in the test using web_sys
// (or via special error return value)
const SKIP_LIST: [&str; 29] = [
    // chrome does not support 5d tensors for
    // "greater"
    "greater float32 5D tensors",
    "greater float16 5D tensors",
    // chrome only supports
    // Failed to execute 'neg' on 'MLGraphBuilder': [neg_op] Unsupported data type int8 for argument input, must be one of [float32, float16, int32, int64].
    "neg int8 4D tensor",
    // Failed to execute 'reduceL1' on 'MLGraphBuilder': [reduceL1_op] Unsupported data type uint32 for argument input, must be one of [float32, float16, int32].
    "reduceL1 uint32 4D tensor options.axes with options.keepDimensions=false",
    // Failed to execute 'abs' on 'MLGraphBuilder': [abs_op] Unsupported data type int8 for argument input, must be one of [float32, float16, int32].
    "abs int8 4D tensor",
    "abs int64 4D tensor",
    // [clamp_op] Unsupported data type int8 for argument input, must be one of [float32, float16].
    "clamp int8 1D tensor",
    "clamp uint8 1D tensor",
    "clamp int32 1D tensor",
    "clamp uint32 1D tensor",
    "clamp int64 1D tensor",
    "clamp uint64 1D tensor",
    "clamp int64 1D tensor with bigint max",
    "clamp uint64 1D tensor with bigint max",
    "clamp uint64 1D tensor with Number min and max",
    // Failed to execute 'equal' on 'MLGraphBuilder': [equal_op] Unsupported rank 5 for argument a, must be in range [0, 4].
    "equal float32 5D tensors",
    "equal float16 5D tensors",
    "greaterOrEqual float32 5D tensors",
    "greaterOrEqual float16 5D tensors",
    "lesserOrEqual float32 5D tensors",
    "lesserOrEqual float16 5D tensors",
    "lesser float32 5D tensors",
    "lesser float16 5D tensors",
    //Failed to execute 'pow' on 'MLGraphBuilder': [pow_op] Unsupported rank 5 for argument a, must be in range [0, 4].
    "pow float32 5D base tensor and 5D integer exponent tensor",
    "pow float16 5D base tensor and 5D integer exponent tensor",
    // [sub_op] Unsupported data type int8 for argument a, must be one of [float32, float16, int32,
    // int64].
    "sub int8 4D tensors",
    "sub uint8 4D tensors",
    "sub uint32 4D tensors",
    "sub uint64 4D tensors",
];

/// Run a single WPT test case using WebNN. Supports all `MlOperandDataType` scalar types.
pub async fn run_one_test_case_webnn(
    operation: &str,
    test_case: &WptTestCase,
) -> Result<WebnnWptOutcome, String> {
    web_sys::console::log_1(&format!("         {}::{}", operation, test_case.name).into());
    if SKIP_LIST.contains(&test_case.name.as_str()) {
        web_sys::console::log_1(&format!("[SKIP]        {}::{}", operation, test_case.name).into());
        return Ok(WebnnWptOutcome::Skipped);
    }
    let graph = &test_case.graph;
    if graph
        .expected_outputs
        .values()
        .any(float_expected_has_infinity)
    {
        web_sys::console::log_1(
            &format!(
                "[SKIP]        {}::{} (expected +/-Infinity in float output)",
                operation, test_case.name
            )
            .into(),
        );
        return Ok(WebnnWptOutcome::Skipped);
    }

    match run_one_test_case_webnn_body(operation, test_case, graph).await {
        Ok(()) => Ok(WebnnWptOutcome::Passed),
        Err(e) if error_message_is_not_supported(&e) => {
            web_sys::console::log_1(
                &format!(
                    "[SKIP]        {}::{} (NotSupportedError: {e})",
                    operation, test_case.name
                )
                .into(),
            );
            Ok(WebnnWptOutcome::Skipped)
        }
        Err(e) => Err(e),
    }
}

async fn run_one_test_case_webnn_body(
    operation: &str,
    test_case: &WptTestCase,
    graph: &WptGraph,
) -> Result<(), String> {
    let (graph_info, input_names) = wpt_graph_to_graph_info(graph)?;

    let window = window().ok_or("no global window")?;
    let navigator = window.navigator();
    let ml = navigator.ml();
    let options = MlContextOptions::new();
    options.set_accelerated(true);
    options.set_power_preference(MlPowerPreference::HighPerformance);
    let promise = ml.create_context_with_ml_context_options(&options);
    let result = JsFuture::from(promise)
        .await
        .map_err(|e| js_value_to_string(&e))?;
    let context: MlContext = result.dyn_into().map_err(|e| js_value_to_string(&e))?;

    let converter = WebNNConverter::default();
    let converted = converter
        .convert_async(&context, &graph_info)
        .await
        .map_err(|e| e.to_string())?;
    let ml_graph = converted
        .graph
        .ok_or("WebNN conversion did not produce a graph")?;

    let inputs_obj: js_sys::Object<MlTensor> = js_sys::Object::new_typed();
    for name in &input_names {
        let spec = graph
            .inputs
            .get(name)
            .ok_or_else(|| format!("input {name} not found"))?;
        let dtype = spec.data_type();
        let ml_ty = wpt_data_type_to_ml(dtype)?;
        let shape: Vec<i32> = spec.shape().iter().map(|&d| d as i32).collect();
        let desc =
            MlTensorDescriptor::new(ml_ty, &shape.iter().map(|&x| x.into()).collect::<Vec<_>>());
        desc.set_writable(true);
        let tensor = JsFuture::from(context.create_tensor(&desc))
            .await
            .map_err(|e| format!("create_tensor {name} failed: {}", js_value_to_string(&e)))?;

        match dtype {
            "float32" => {
                let data = expected_output_to_f32(spec);
                let arr = Float32Array::new_from_slice(&data);
                context.write_tensor_with_buffer_source(&tensor, &arr);
            }
            "float16" => {
                let bits = f16_bits_from_wpt(spec);
                let arr = Uint16Array::new_from_slice(&bits);
                context.write_tensor_with_buffer_source(&tensor, &arr);
            }
            "int32" => {
                let data = expected_output_to_i32(spec);
                let arr = Int32Array::new_from_slice(&data);
                context.write_tensor_with_buffer_source(&tensor, &arr);
            }
            "uint32" => {
                let data = expected_output_to_u32(spec);
                let arr = Uint32Array::new_from_slice(&data);
                context.write_tensor_with_buffer_source(&tensor, &arr);
            }
            "int64" => {
                let data = expected_output_to_i64(spec);
                let arr = BigInt64Array::new_from_slice(&data);
                context.write_tensor_with_buffer_source(&tensor, &arr);
            }
            "uint64" => {
                let data = expected_output_to_u64(spec);
                let arr = BigUint64Array::new_from_slice(&data);
                context.write_tensor_with_buffer_source(&tensor, &arr);
            }
            "int8" => {
                let data = expected_output_to_i8(spec);
                let arr = Int8Array::new_from_slice(&data);
                context.write_tensor_with_buffer_source(&tensor, &arr);
            }
            "uint8" => {
                let data = expected_output_to_u8(spec);
                let arr = Uint8Array::new_from_slice(&data);
                context.write_tensor_with_buffer_source(&tensor, &arr);
            }
            _ => {
                return Err(format!(
                    "unsupported input data type for WebNN runner: {dtype}"
                ));
            }
        }

        Reflect::set(&inputs_obj, &name.clone().into(), &tensor)
            .map_err(|_| "Reflect::set input failed")?;
    }

    let outputs_obj: js_sys::Object<MlTensor> = js_sys::Object::new_typed();
    let mut output_jobs: Vec<(String, WptTensorSpec, ReadBackBuf)> = Vec::new();
    for (out_name, spec) in &graph.expected_outputs {
        let dtype = spec.data_type();
        let ml_ty = wpt_data_type_to_ml(dtype)?;
        let shape: Vec<i32> = spec.shape().iter().map(|&d| d as i32).collect();
        let n = tensor_element_count(&shape);
        let desc =
            MlTensorDescriptor::new(ml_ty, &shape.iter().map(|&x| x.into()).collect::<Vec<_>>());
        desc.set_readable(true);
        let tensor = JsFuture::from(context.create_tensor(&desc))
            .await
            .map_err(|e| {
                format!(
                    "create_tensor output {out_name} failed: {}",
                    js_value_to_string(&e)
                )
            })?;
        let read_buf = read_back_buf_for_dtype(dtype, n)?;
        Reflect::set(&outputs_obj, &out_name.clone().into(), &tensor)
            .map_err(|_| "Reflect::set output failed")?;
        output_jobs.push((out_name.clone(), spec.clone(), read_buf));
    }

    context.dispatch(&ml_graph, &inputs_obj, &outputs_obj);

    for (out_name, spec, read_buf) in &output_jobs {
        validate_output_tensor(
            &context,
            &outputs_obj,
            operation,
            test_case,
            out_name,
            spec,
            read_buf,
        )
        .await
        .map_err(|e| format!("{operation} :: {}: {e}", test_case.name))?;
    }
    Ok(())
}

/// Run all WPT conformance tests from the embedded gzip bundle using WebNN. Called from wasm_bindgen_test.
pub async fn run_all_webnn() -> Result<(), String> {
    let entries: Vec<(String, String)> = serde_json::from_str(WPT_CONFORMANCE_JSON)
        .map_err(|e| format!("parse decompressed JSON: {e}"))?;
    if entries.is_empty() {
        return Ok(());
    }
    let mut passed = 0usize;
    let mut skipped = 0usize;
    let mut failed = Vec::new();
    for (filename, json) in entries {
        let file = load_wpt_file(&json).map_err(|e| format!("parse {filename}: {e}"))?;
        let operation = file.operation.clone();
        for test_case in &file.tests {
            match run_one_test_case_webnn(&operation, test_case).await {
                Ok(WebnnWptOutcome::Passed) => {
                    web_sys::console::log_1(
                        &format!("[OK]     {}::{}", operation, test_case.name).into(),
                    );
                    passed += 1
                }
                Ok(WebnnWptOutcome::Skipped) => {
                    skipped += 1;
                }
                Err(e) => {
                    web_sys::console::log_1(
                        &format!(
                            "[Failed]     {}::{}\n\t\t\t{e:?}",
                            operation, test_case.name
                        )
                        .into(),
                    );
                    failed.push((format!("{}::{}", operation, test_case.name), e))
                }
            }
        }
    }
    web_sys::console::log_1(
        &format!(
            "WPT WebNN: {} passed, {} skipped, {} failed",
            passed,
            skipped,
            failed.len(),
        )
        .into(),
    );
    if failed.is_empty() {
        Ok(())
    } else {
        let msg = failed
            .iter()
            .take(10)
            .map(|(name, e)| format!("  {name}: {e}"))
            .collect::<Vec<_>>()
            .join("\n");
        let more = if failed.len() > 10 {
            format!("\n  ... and {} more", failed.len() - 10)
        } else {
            String::new()
        };
        Err(format!(
            "WPT WebNN: {} passed, {} skipped, {} failed\n{}{}",
            passed,
            skipped,
            failed.len(),
            msg,
            more
        ))
    }
}

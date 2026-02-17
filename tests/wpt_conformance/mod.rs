//! WPT (Web Platform Tests) conformance test runner for WebNN.
//!
//! Runs tests from tests/wpt_data/conformance/*.json using the ONNX backend
//! or the TensorRT (trtx) backend when enabled.

pub mod tolerance;
pub mod wpt_to_graph;
pub mod wpt_types;

use std::fs;

use tolerance::get_operation_tolerance;
use wpt_to_graph::wpt_data_dir;
use wpt_types::load_wpt_file;

#[cfg(feature = "onnx-runtime")]
use rustnn::converters::{GraphConverter, OnnxConverter};
#[cfg(feature = "onnx-runtime")]
use rustnn::run_onnx_with_inputs;
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
use rustnn::converters::{GraphConverter, TrtxConverter};
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
use rustnn::run_trtx_with_inputs;

use tolerance::validate_result;
use wpt_to_graph::{expected_output_to_f32, wpt_graph_to_graph_info};
#[cfg(feature = "onnx-runtime")]
use wpt_to_graph::wpt_graph_to_onnx_inputs;
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
use wpt_to_graph::wpt_graph_to_trtx_inputs;
use wpt_types::WptGraph;

const FAILURE_DISPLAY_LEN: usize = 24;

/// Decode TRTX output bytes to f32 for validation. Handles float32 and float16.
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
fn trtx_output_bytes_to_f32(data: &[u8], data_type: &str) -> Vec<f32> {
    if data_type == "float16" {
        assert!(data.len() % 2 == 0);
        data.chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect()
    } else {
        // float32 or other: interpret as little-endian f32
        assert!(data.len() % 4 == 0);
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

/// Format a slice of f32 for failure output (prefix + first N + suffix if truncated).
fn format_f32_slice_for_failure(slice: &[f32], max_show: usize) -> String {
    if slice.is_empty() {
        return "[]".to_string();
    }
    let head: Vec<String> = slice
        .iter()
        .take(max_show)
        .map(|v| format!("{}", v))
        .collect();
    let s = head.join(", ");
    if slice.len() <= max_show {
        format!("[{}]", s)
    } else {
        format!("[{} ...] (len={})", s, slice.len())
    }
}

/// Format one JSON value for failure output; wrap numbers in Number() for readability.
fn format_input_value(v: &serde_json::Value) -> String {
    if let Some(n) = v.as_i64() {
        return format!("Number({})", n);
    }
    if let Some(n) = v.as_u64() {
        return format!("Number({})", n);
    }
    if let Some(n) = v.as_f64() {
        return format!("Number({})", n);
    }
    format!("{:?}", v)
}

/// Format graph inputs for failure output (only non-constant, by input_names).
fn format_inputs_for_failure(graph: &WptGraph, input_names: &[String]) -> String {
    let parts: Vec<String> = input_names
        .iter()
        .filter_map(|name| {
            graph.inputs.get(name).map(|spec| {
                let data_str = if let Some(arr) = spec.data.as_array() {
                    let head: Vec<String> = arr
                        .iter()
                        .take(FAILURE_DISPLAY_LEN)
                        .map(format_input_value)
                        .collect();
                    if arr.len() <= FAILURE_DISPLAY_LEN {
                        format!("[{}]", head.join(", "))
                    } else {
                        format!("[{} ...] (len={})", head.join(", "), arr.len())
                    }
                } else {
                    format_input_value(&spec.data)
                };
                format!("{}: {}", name, data_str)
            })
        })
        .collect();
    parts.join("; ")
}

/// Discover all operation names that have WPT conformance data.
pub fn discover_operations() -> Vec<String> {
    let dir = wpt_data_dir();
    if !dir.exists() {
        return Vec::new();
    }
    let mut names = Vec::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().map_or(false, |e| e == "json") {
                if let Some(stem) = p.file_stem() {
                    names.push(stem.to_string_lossy().to_string());
                }
            }
        }
    }
    names.sort();
    names
}

/// Run a single WPT test case: build graph, run ONNX, validate outputs.
#[cfg(feature = "onnx-runtime")]
pub fn run_one_test_case(
    operation: &str,
    test_case: &wpt_types::WptTestCase,
) -> Result<(), String> {
    let graph = &test_case.graph;
    let (graph_info, input_names) = wpt_graph_to_graph_info(graph)?;
    let inputs = wpt_graph_to_onnx_inputs(graph, &input_names)?;

    let converter = OnnxConverter;
    let converted = converter.convert(&graph_info).map_err(|e| e.to_string())?;

    let outputs = run_onnx_with_inputs(&converted.data, inputs).map_err(|e| e.to_string())?;

    let (tolerance_kind, tolerance_value) =
        get_operation_tolerance(operation, test_case.tolerance.as_ref());

    for (out_name, expected_spec) in &graph.expected_outputs {
        // Executor returns Vec<f32> only; skip validation for int64 expected output.
        if expected_spec.data_type() == "int64" {
            continue;
        }
        let actual = outputs
            .iter()
            .find(|o| o.name == *out_name)
            .ok_or_else(|| format!("output '{}' not found in results", out_name))?;
        let expected = expected_output_to_f32(expected_spec);
        let (pass, msg) = validate_result(
            &actual.data,
            &expected,
            tolerance_kind,
            tolerance_value,
        );
        if !pass {
            let inputs_str = format_inputs_for_failure(graph, &input_names);
            let expected_str =
                format_f32_slice_for_failure(&expected, FAILURE_DISPLAY_LEN);
            let actual_str =
                format_f32_slice_for_failure(&actual.data, FAILURE_DISPLAY_LEN);
            return Err(format!(
                "{} :: {}: {}\n  inputs: {}\n  expected {}: {}\n  actual {}: {}",
                operation,
                test_case.name,
                msg.unwrap_or_else(|| "validation failed".to_string()),
                inputs_str,
                out_name,
                expected_str,
                out_name,
                actual_str
            ));
        }
    }
    Ok(())
}

/// No tests are skipped; all WPT conformance cases are run.
#[cfg(feature = "onnx-runtime")]
fn is_skipped_test(_operation: &str, _test_name: &str) -> bool {
    false
}

/// Run a single WPT test case: build graph, run TensorRT, validate outputs.
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
pub fn run_one_test_case_trtx(
    operation: &str,
    test_case: &wpt_types::WptTestCase,
) -> Result<(), String> {
    let graph = &test_case.graph;
    let (graph_info, input_names) = wpt_graph_to_graph_info(graph)?;
    let inputs = wpt_graph_to_trtx_inputs(graph, &input_names)?;

    let converter = TrtxConverter;
    let converted = converter
        .convert(&graph_info)
        .map_err(|e: rustnn::GraphError| e.to_string())?;

    let outputs = run_trtx_with_inputs(&converted.data, inputs).map_err(|e| e.to_string())?;

    let (tolerance_kind, tolerance_value) =
        get_operation_tolerance(operation, test_case.tolerance.as_ref());

    for (out_name, expected_spec) in &graph.expected_outputs {
        if expected_spec.data_type() == "int64" {
            continue;
        }
        let actual = outputs
            .iter()
            .find(|o| o.name == *out_name)
            .ok_or_else(|| format!("output '{}' not found in results", out_name))?;
        let actual_f32 = trtx_output_bytes_to_f32(&actual.data, &actual.data_type);
        let expected = expected_output_to_f32(expected_spec);
        let (pass, msg) = validate_result(
            &actual_f32,
            &expected,
            tolerance_kind,
            tolerance_value,
        );
        if !pass {
            let inputs_str = format_inputs_for_failure(graph, &input_names);
            let expected_str =
                format_f32_slice_for_failure(&expected, FAILURE_DISPLAY_LEN);
            let actual_str =
                format_f32_slice_for_failure(&actual_f32, FAILURE_DISPLAY_LEN);
            return Err(format!(
                "{} :: {}: {}\n  inputs: {}\n  expected {}: {}\n  actual {}: {}",
                operation,
                test_case.name,
                msg.unwrap_or_else(|| "validation failed".to_string()),
                inputs_str,
                out_name,
                expected_str,
                out_name,
                actual_str
            ));
        }
    }
    Ok(())
}

/// TRTX executor supports float32 and float16. Skip tests that use int8, int32, or int64 output.
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
fn trtx_skip_reason(test_case: &wpt_types::WptTestCase) -> Option<&'static str> {
    for spec in test_case.graph.inputs.values() {
        let dt = spec.data_type().to_lowercase();
        if dt == "int8" || dt == "uint8" || dt == "int32" || dt == "uint32" {
            return Some("TRTX executor does not support integer input types");
        }
    }
    for spec in test_case.graph.expected_outputs.values() {
        let dt = spec.data_type().to_lowercase();
        if dt == "int8" || dt == "uint8" || dt == "int32" || dt == "uint32" || dt == "int64" {
            return Some("TRTX executor does not support integer output types");
        }
    }
    None
}

/// Run all WPT conformance tests using the TensorRT (trtx) backend.
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
pub fn run_all_trtx() -> Result<(), String> {
    // Reduce TensorRT log noise: only warning and more severe.
    unsafe { std::env::set_var("RUSTNN_TRTX_LOG_VERBOSITY", "error") };

    let dir = wpt_data_dir();
    if !dir.exists() {
        return Err(format!("WPT data dir not found: {}", dir.display()));
    }

    let operations = discover_operations();
    if operations.is_empty() {
        return Err("No WPT conformance JSON files found".to_string());
    }

    println!("[WPT-TRTX] data dir: {}", dir.display());
    println!(
        "[WPT-TRTX] found {} operation(s): {}",
        operations.len(),
        operations.join(", ")
    );

    let mut passed = 0usize;
    let mut skipped = 0usize;
    let mut failed = Vec::new();
    let mut total_cases = 0usize;

    for op in &operations {
        let path = dir.join(format!("{}.json", op));
        let json = fs::read_to_string(&path).map_err(|e| format!("read {}: {}", path.display(), e))?;
        let file = load_wpt_file(&json).map_err(|e| format!("parse {}: {}", path.display(), e))?;

        let num_tests = file.tests.len();
        total_cases += num_tests;
        println!("[WPT-TRTX] operation '{}': {} test case(s)", op, num_tests);

        for test_case in &file.tests {
            println!("  running: {} :: {}", op, test_case.name);
            if let Some(reason) = trtx_skip_reason(test_case) {
                skipped += 1;
                println!("    [SKIP] {}", reason);
                continue;
            }
            match run_one_test_case_trtx(&file.operation, test_case) {
                Ok(()) => {
                    passed += 1;
                    println!("    [OK]");
                }
                Err(e) => {
                    failed.push((format!("{}::{}", op, test_case.name), e.clone()));
                    println!("    [FAIL]\n{}", e);
                }
            }
        }
    }

    println!(
        "[WPT-TRTX] total: {} passed, {} skipped, {} failed (of {} cases)",
        passed, skipped, failed.len(), total_cases
    );

    if failed.is_empty() {
        Ok(())
    } else {
        let msg = failed
            .iter()
            .take(10)
            .map(|(name, e)| format!("  {}: {}", name, e))
            .collect::<Vec<_>>()
            .join("\n");
        let more = if failed.len() > 10 {
            format!("\n  ... and {} more failures", failed.len() - 10)
        } else {
            String::new()
        };
        Err(format!(
            "WPT conformance (TRTX): {} passed, {} skipped, {} failed\n{}{}",
            passed, skipped, failed.len(), msg, more
        ))
    }
}

/// Run all WPT conformance tests (discover from wpt_data/conformance).
#[cfg(feature = "onnx-runtime")]
pub fn run_all() -> Result<(), String> {
    let dir = wpt_data_dir();
    if !dir.exists() {
        return Err(format!("WPT data dir not found: {}", dir.display()));
    }

    let operations = discover_operations();
    if operations.is_empty() {
        return Err("No WPT conformance JSON files found".to_string());
    }

    println!("[WPT] data dir: {}", dir.display());
    println!("[WPT] found {} operation(s): {}", operations.len(), operations.join(", "));

    let mut passed = 0usize;
    let mut skipped = 0usize;
    let mut failed = Vec::new();
    let mut total_cases = 0usize;

    for op in &operations {
        let path = dir.join(format!("{}.json", op));
        let json = fs::read_to_string(&path).map_err(|e| format!("read {}: {}", path.display(), e))?;
        let file = load_wpt_file(&json).map_err(|e| format!("parse {}: {}", path.display(), e))?;

        let num_tests = file.tests.len();
        total_cases += num_tests;
        println!("[WPT] operation '{}': {} test case(s)", op, num_tests);

        for test_case in &file.tests {
            println!("  running: {} :: {}", op, test_case.name);
            if is_skipped_test(&file.operation, &test_case.name) {
                skipped += 1;
                println!("    [SKIP]");
                continue;
            }
            match run_one_test_case(&file.operation, test_case) {
                Ok(()) => {
                    passed += 1;
                    println!("    [OK]");
                }
                Err(e) => {
                    failed.push((format!("{}::{}", op, test_case.name), e.clone()));
                    println!("    [FAIL]\n{}", e);
                }
            }
        }
    }

    println!(
        "[WPT] total: {} passed, {} skipped, {} failed (of {} cases)",
        passed, skipped, failed.len(), total_cases
    );

    if failed.is_empty() {
        Ok(())
    } else {
        let msg = failed
            .iter()
            .take(10)
            .map(|(name, e)| format!("  {}: {}", name, e))
            .collect::<Vec<_>>()
            .join("\n");
        let more = if failed.len() > 10 {
            format!("\n  ... and {} more failures", failed.len() - 10)
        } else {
            String::new()
        };
        Err(format!(
            "WPT conformance: {} passed, {} skipped, {} failed\n{}{}",
            passed, skipped, failed.len(), msg, more
        ))
    }
}

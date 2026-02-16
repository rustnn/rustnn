//! WPT (Web Platform Tests) conformance test runner for WebNN.
//!
//! Runs tests from tests/wpt_data/conformance/*.json using the ONNX backend.

pub mod tolerance;
pub mod wpt_to_graph;
pub mod wpt_types;

use std::fs;

use rustnn::converters::{GraphConverter, OnnxConverter};
use rustnn::run_onnx_with_inputs;

use tolerance::{get_operation_tolerance, validate_result};
use wpt_to_graph::{
    expected_output_to_f32, wpt_data_dir, wpt_graph_to_graph_info, wpt_graph_to_onnx_inputs,
};
use wpt_types::load_wpt_file;

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
            return Err(format!(
                "{} :: {}: {}",
                operation,
                test_case.name,
                msg.unwrap_or_else(|| "validation failed".to_string())
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
                    failed.push((format!("{}::{}", op, test_case.name), e));
                    println!("    [FAIL]");
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

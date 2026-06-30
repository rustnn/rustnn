//! WPT (Web Platform Tests) conformance test runner for WebNN.
//!
//! Loads tests from upstream WPT `.https.any.js` files via the Node.js bridge.
//! All backends run through [`MLGraphBuilder`] and [`MLContext::dispatch`].
//!
//! Backend selection:
//! - `WPT_BACKEND=onnx` or `trtx` limits which backend trials are registered.
//! - Backends without a working [`MLContext`] are omitted at startup (not registered as trials).
//! - Filter trials by name prefix: `cargo test --test run_wpt_conformance -- onnx::relu`

pub mod tolerance;
pub mod wpt_backend;
pub mod wpt_config;
pub mod wpt_context_pool;
pub mod wpt_execute_graph;
pub mod wpt_js_loader;
pub mod wpt_report;
pub mod wpt_tensor;
pub mod wpt_types;

use tolerance::{check_integer_tolerance, get_operation_tolerance, validate_result};
use wpt_backend::WptBackend;
use wpt_tensor::{
    expected_output_to_f32, expected_output_to_i32, expected_output_to_i64, expected_output_to_u64,
};
use wpt_types::WptGraph;

const FAILURE_DISPLAY_LEN: usize = 24;

const SUPPORTED_DTYPES: &[&str] = &[
    "float32", "float16", "int8", "uint8", "int32", "uint32", "int64", "uint64", "int4", "uint4",
];

/// Skip WPT cases whose inputs or expected outputs use unsupported tensor dtypes.
pub fn should_skip_test(graph: &WptGraph) -> Option<String> {
    for spec in graph.inputs.values().chain(graph.expected_outputs.values()) {
        let dt = spec.data_type();
        if !SUPPORTED_DTYPES
            .iter()
            .any(|supported| supported.eq_ignore_ascii_case(dt))
        {
            return Some(format!("unsupported dataType: {dt}"));
        }
    }
    None
}

fn graph_operator_names(graph: &WptGraph) -> Vec<String> {
    graph
        .operators
        .iter()
        .map(|op| wpt_tensor::normalize_wpt_op_name(&op.name))
        .collect()
}

fn runtime_input_names(graph: &WptGraph) -> Vec<String> {
    graph
        .inputs
        .iter()
        .filter(|(_, spec)| !spec.constant || !wpt_execute_graph::should_inline_constant(spec))
        .map(|(name, _)| name.clone())
        .collect()
}

/// Format a flat integer slice as n-dimensional for failure output (exact values, no f32 precision loss).
fn format_int_nd<T: std::fmt::Display>(slice: &[T], shape: &[u32]) -> String {
    if shape.is_empty() {
        return format!(
            "[{}]",
            slice
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let size: usize = shape.iter().map(|&d| d as usize).product();
    if size != slice.len() {
        return format!(
            "[{} ...] (len={}, shape={:?})",
            slice
                .iter()
                .take(24)
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            slice.len(),
            shape
        );
    }
    let s: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let rank = s.len();
    let mut lines = Vec::new();
    if rank == 1 {
        lines.push(format!(
            "[{}]",
            slice
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    } else if rank == 4 {
        let [n, h, w, c] = [s[0], s[1], s[2], s[3]];
        let mut idx = 0;
        for _n in 0..n {
            for _i in 0..h {
                let row: Vec<String> = (0..w)
                    .map(|_| {
                        let cell: Vec<String> = slice[idx..idx + c]
                            .iter()
                            .map(|v| format!("{}", v))
                            .collect();
                        idx += c;
                        format!("[{}]", cell.join(", "))
                    })
                    .collect();
                lines.push(format!("  {}", row.join(" ")));
            }
        }
    } else if rank == 2 {
        let (rows, cols) = (s[0], s[1]);
        let mut idx = 0;
        for _ in 0..rows {
            let row: Vec<String> = slice[idx..idx + cols]
                .iter()
                .map(|v| format!("{}", v))
                .collect();
            idx += cols;
            lines.push(format!("  [{}]", row.join(", ")));
        }
    } else {
        let flat: Vec<String> = slice.iter().map(|v| format!("{}", v)).collect();
        lines.push(format!("[{}]", flat.join(", ")));
    }
    format!("(shape {:?})\n{}", shape, lines.join("\n"))
}

/// Format a flat f32 slice as n-dimensional for failure output (full tensor, shape e.g. [1,6,6,2]).
/// For 4D [N,H,W,C]: prints N*H lines, each line has W cells of C values as [a, b, ...].
fn format_f32_nd(slice: &[f32], shape: &[u32]) -> String {
    if shape.is_empty() {
        return format!(
            "[{}]",
            slice
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let size: usize = shape.iter().map(|&d| d as usize).product();
    if size != slice.len() {
        return format!(
            "[{} ...] (len={}, shape={:?})",
            slice
                .iter()
                .take(24)
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            slice.len(),
            shape
        );
    }
    let s: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let rank = s.len();
    let mut lines = Vec::new();
    if rank == 1 {
        lines.push(format!(
            "[{}]",
            slice
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    } else if rank == 4 {
        let [n, h, w, c] = [s[0], s[1], s[2], s[3]];
        let mut idx = 0;
        for _n in 0..n {
            for _i in 0..h {
                let row: Vec<String> = (0..w)
                    .map(|_| {
                        let cell: Vec<String> = slice[idx..idx + c]
                            .iter()
                            .map(|v| format!("{}", v))
                            .collect();
                        idx += c;
                        format!("[{}]", cell.join(", "))
                    })
                    .collect();
                lines.push(format!("  {}", row.join(" ")));
            }
        }
    } else if rank == 2 {
        let (rows, cols) = (s[0], s[1]);
        let mut idx = 0;
        for _ in 0..rows {
            let row: Vec<String> = slice[idx..idx + cols]
                .iter()
                .map(|v| format!("{}", v))
                .collect();
            idx += cols;
            lines.push(format!("  [{}]", row.join(", ")));
        }
    } else {
        let flat: Vec<String> = slice.iter().map(|v| format!("{}", v)).collect();
        lines.push(format!("[{}]", flat.join(", ")));
    }
    format!("(shape {:?})\n{}", shape, lines.join("\n"))
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

fn format_int_slice_for_failure(slice: &[i64], max_show: usize) -> String {
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

fn format_i32_slice_for_failure(slice: &[i32], max_show: usize) -> String {
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

/// Format one JSON value for failure output (plain numbers, no wrapper).
fn format_input_value(v: &serde_json::Value) -> String {
    if let Some(n) = v.as_i64() {
        return format!("{}", n);
    }
    if let Some(n) = v.as_u64() {
        return format!("{}", n);
    }
    if let Some(n) = v.as_f64() {
        return format!("{}", n);
    }
    format!("{:?}", v)
}

/// Format graph inputs for failure output (non-constant and constant, so constants are visible in debug).
fn format_inputs_for_failure(graph: &WptGraph, input_names: &[String]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for name in input_names {
        if let Some(spec) = graph.inputs.get(name) {
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
            parts.push(format!("{}: {}", name, data_str));
        }
    }
    for (name, spec) in &graph.inputs {
        if spec.constant && !input_names.contains(name) {
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
            parts.push(format!("{}: {} (constant)", name, data_str));
        }
    }
    parts.join("; ")
}

/// Run a single WPT test case: build graph via MLGraphBuilder, dispatch, validate outputs.
pub fn run_one_test_case(
    backend: &WptBackend,
    operation: &str,
    test_case: &wpt_types::WptTestCase,
) -> Result<(), String> {
    let graph = &test_case.graph;
    let graph_op_names = graph_operator_names(graph);
    let graph_op_refs: Vec<&str> = graph_op_names.iter().map(String::as_str).collect();
    let (tolerance_kind, tolerance_value) =
        get_operation_tolerance(operation, test_case.tolerance.as_ref(), &graph_op_refs);
    let wpt_ulp_only = test_case
        .tolerance
        .as_ref()
        .is_some_and(|t| t.metric_type.eq_ignore_ascii_case("ulp"));

    wpt_context_pool::with_context(backend, |context| {
        let artifacts = wpt_execute_graph::execute_wpt_graph(context, graph)?;
        let webnn_text = artifacts.webnn_text.as_deref();
        let outputs = artifacts.outputs;
        let input_names = runtime_input_names(graph);

        for (out_name, expected_spec) in &graph.expected_outputs {
            let actual = outputs
                .get(out_name)
                .ok_or_else(|| format!("output '{out_name}' not found in results"))?;

            let dtype = expected_spec.data_type();
            let (pass, msg, expected_str, actual_str) = match dtype {
                "int64" => {
                    let expected = expected_output_to_i64(expected_spec);
                    let actual_i64 = actual.i64_data().ok_or_else(|| {
                        format!(
                            "output '{out_name}' missing int64 data (type {})",
                            actual.data_type()
                        )
                    })?;
                    let int_tol = test_case
                        .tolerance
                        .as_ref()
                        .and_then(|t| {
                            t.value
                                .as_u64()
                                .or_else(|| t.value.as_f64().map(|f| f as u64))
                        })
                        .unwrap_or(0) as i64;
                    let (pass, msg) = check_integer_tolerance(actual_i64, &expected, int_tol);
                    let expected_str = format_int_slice_for_failure(&expected, FAILURE_DISPLAY_LEN);
                    let actual_str = format_int_slice_for_failure(actual_i64, FAILURE_DISPLAY_LEN);
                    (pass, msg, expected_str, actual_str)
                }
                "uint64" => {
                    let expected = expected_output_to_u64(expected_spec);
                    let actual_i64 = actual.i64_data().ok_or_else(|| {
                        format!(
                            "output '{out_name}' missing uint64 data (type {})",
                            actual.data_type()
                        )
                    })?;
                    let pass = actual_i64.len() == expected.len()
                        && actual_i64
                            .iter()
                            .zip(expected.iter())
                            .all(|(&a, &e)| a as u64 == e);
                    let msg = if pass {
                        None
                    } else {
                        Some("uint64 output mismatch".to_string())
                    };
                    let expected_u64_str: Vec<i64> = expected.iter().map(|&u| u as i64).collect();
                    let expected_str =
                        format_int_slice_for_failure(&expected_u64_str, FAILURE_DISPLAY_LEN);
                    let actual_str = format_int_slice_for_failure(actual_i64, FAILURE_DISPLAY_LEN);
                    (pass, msg, expected_str, actual_str)
                }
                "int4" | "uint4" | "int8" | "uint8" | "int32" | "uint32" => {
                    let expected = expected_output_to_i32(expected_spec);
                    let actual_i32 = actual.i32_data().ok_or_else(|| {
                        format!(
                            "output '{out_name}' missing integer data (type {})",
                            actual.data_type()
                        )
                    })?;
                    let int_tol = test_case
                        .tolerance
                        .as_ref()
                        .and_then(|t| {
                            t.value
                                .as_u64()
                                .or_else(|| t.value.as_f64().map(|f| f as u64))
                        })
                        .unwrap_or(0) as i64;
                    let expected_i64: Vec<i64> = expected.iter().map(|&x| x as i64).collect();
                    let actual_i64: Vec<i64> = actual_i32.iter().map(|&x| x as i64).collect();
                    let (pass, msg) = check_integer_tolerance(&actual_i64, &expected_i64, int_tol);
                    let expected_str = format_i32_slice_for_failure(&expected, FAILURE_DISPLAY_LEN);
                    let actual_str = format_i32_slice_for_failure(actual_i32, FAILURE_DISPLAY_LEN);
                    (pass, msg, expected_str, actual_str)
                }
                _ => {
                    let expected = expected_output_to_f32(expected_spec);
                    let actual_f32 = actual.f32_data().ok_or_else(|| {
                        format!(
                            "output '{out_name}' missing float data (type {})",
                            actual.data_type()
                        )
                    })?;
                    let float16 = dtype.eq_ignore_ascii_case("float16");
                    let (pass, msg) = validate_result(
                        actual_f32,
                        &expected,
                        tolerance_kind,
                        tolerance_value,
                        float16,
                        wpt_ulp_only,
                    );
                    let expected_str = format_f32_slice_for_failure(&expected, FAILURE_DISPLAY_LEN);
                    let actual_str = format_f32_slice_for_failure(actual_f32, FAILURE_DISPLAY_LEN);
                    (pass, msg, expected_str, actual_str)
                }
            };

            if !pass {
                let inputs_str = format_inputs_for_failure(graph, &input_names);
                let shape = expected_spec.shape();
                let nd_suffix = if !shape.is_empty() && shape.iter().all(|&d| d > 0) {
                    if matches!(
                        dtype,
                        "int4" | "uint4" | "int8" | "uint8" | "int32" | "uint32"
                    ) {
                        let expected = expected_output_to_i32(expected_spec);
                        let actual_i32 = actual.i32_data().unwrap_or(&[]);
                        let expected_nd = format_int_nd(
                            &expected.iter().map(|&x| x as i64).collect::<Vec<_>>(),
                            shape,
                        );
                        let actual_nd = format_int_nd(
                            &actual_i32.iter().map(|&x| x as i64).collect::<Vec<_>>(),
                            shape,
                        );
                        format!(
                            "\n  expected {out_name} full nd:\n{expected_nd}\n  actual {out_name} full nd:\n{actual_nd}"
                        )
                    } else {
                        let expected = expected_output_to_f32(expected_spec);
                        let actual_f32 = actual.f32_data().unwrap_or(&[]);
                        let expected_nd = format_f32_nd(&expected, shape);
                        let actual_nd = format_f32_nd(actual_f32, shape);
                        format!(
                            "\n  expected {out_name} full nd:\n{expected_nd}\n  actual {out_name} full nd:\n{actual_nd}"
                        )
                    }
                } else {
                    String::new()
                };
                return Err(wpt_execute_graph::append_webnn_graph_text(
                    format!(
                        "{} :: {}: {}\n  inputs: {}\n  expected {}: {}\n  actual {}: {}{}",
                        operation,
                        test_case.name,
                        msg.unwrap_or_else(|| "validation failed".to_string()),
                        inputs_str,
                        out_name,
                        expected_str,
                        out_name,
                        actual_str,
                        nd_suffix
                    ),
                    webnn_text,
                ));
            }
        }
        Ok(())
    })
}

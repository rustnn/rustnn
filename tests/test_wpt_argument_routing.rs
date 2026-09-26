#![allow(dead_code)] // This focused test imports only the graph-compilation half of the WPT harness.

#[path = "wpt_conformance/wpt_execute_graph.rs"]
mod wpt_execute_graph;
#[path = "wpt_conformance/wpt_tensor.rs"]
mod wpt_tensor;
#[path = "wpt_conformance/wpt_types.rs"]
mod wpt_types;

use serde_json::{Value, json};
use wpt_execute_graph::compile_wpt_graph;
use wpt_types::WptGraph;

fn graph_from_json(input: Value, operator: Value, output: Value) -> WptGraph {
    serde_json::from_value(json!({
        "inputs": { "input": input },
        "operators": [operator],
        "expectedOutputs": { "output": output }
    }))
    .expect("valid WPT graph fixture")
}

fn scalar_tensor(data_type: &str) -> Value {
    json!({
        "data": [1.0],
        "descriptor": { "shape": [], "dataType": data_type }
    })
}

fn one_element_tensor(data_type: &str) -> Value {
    json!({
        "data": [1.0],
        "descriptor": { "shape": [1], "dataType": data_type }
    })
}

fn single_output_operator(name: &str, arguments: Value) -> Value {
    json!({
        "name": name,
        "arguments": arguments,
        "outputs": "output"
    })
}

#[test]
fn empty_array_arguments_compile_as_explicit_values() {
    let cases = [
        graph_from_json(
            scalar_tensor("float32"),
            single_output_operator("expand", json!([{ "input": "input" }, { "newShape": [] }])),
            scalar_tensor("float32"),
        ),
        graph_from_json(
            one_element_tensor("float32"),
            single_output_operator("reshape", json!([{ "input": "input" }, { "newShape": [] }])),
            scalar_tensor("float32"),
        ),
        graph_from_json(
            one_element_tensor("float16"),
            single_output_operator("reshape", json!([{ "input": "input" }, { "newShape": [] }])),
            scalar_tensor("float16"),
        ),
        graph_from_json(
            scalar_tensor("float32"),
            single_output_operator(
                "slice",
                json!([
                    { "input": "input" },
                    { "starts": [] },
                    { "sizes": [] }
                ]),
            ),
            scalar_tensor("float32"),
        ),
        graph_from_json(
            scalar_tensor("float32"),
            single_output_operator("tile", json!([{ "input": "input" }, { "repetitions": [] }])),
            scalar_tensor("float32"),
        ),
    ];

    for graph in cases {
        compile_wpt_graph(&graph).expect("empty array argument should remain explicitly supplied");
    }
}

#[test]
fn nonempty_operand_array_still_routes_positionally() {
    let graph: WptGraph = serde_json::from_value(json!({
        "inputs": {
            "lhs": one_element_tensor("float32"),
            "rhs": one_element_tensor("float32")
        },
        "operators": [single_output_operator(
            "concat",
            json!([{ "inputs": ["lhs", "rhs"] }, { "axis": 0 }])
        )],
        "expectedOutputs": {
            "output": {
                "data": [1.0, 1.0],
                "descriptor": { "shape": [2], "dataType": "float32" }
            }
        }
    }))
    .expect("valid WPT concat fixture");

    compile_wpt_graph(&graph).expect("nonempty operand arrays should remain positional");
}

#[test]
fn missing_required_array_arguments_remain_errors() {
    let cases = [
        ("expand", scalar_tensor("float32"), "requires newShape"),
        (
            "reshape",
            one_element_tensor("float32"),
            "requires newShape",
        ),
        ("slice", scalar_tensor("float32"), "requires starts"),
        ("tile", scalar_tensor("float32"), "requires repetitions"),
    ];

    for (operation, input, expected_error) in cases {
        let graph = graph_from_json(
            input.clone(),
            single_output_operator(operation, json!([{ "input": "input" }])),
            input,
        );
        let error = compile_wpt_graph(&graph).expect_err("missing argument should fail");
        assert!(
            error.contains(expected_error),
            "{operation} returned unexpected error: {error}"
        );
    }

    let graph = graph_from_json(
        scalar_tensor("float32"),
        single_output_operator("slice", json!([{ "input": "input" }, { "starts": [] }])),
        scalar_tensor("float32"),
    );
    let error = compile_wpt_graph(&graph).expect_err("missing sizes should fail");
    assert!(
        error.contains("requires sizes"),
        "slice returned unexpected error: {error}"
    );
}

"""
WPT WebNN Conformance Tests

This module runs W3C Web Platform Tests (WPT) for WebNN conformance against
the rustnn implementation. It loads test data converted from the official WPT
test suite and validates that our implementation produces correct results.

Test data is loaded from tests/wpt_data/conformance/*.json files.

Usage:
    # Run all WPT conformance tests
    pytest tests/test_wpt_conformance.py -v

    # Run tests for specific operation
    pytest tests/test_wpt_conformance.py -k "reduce_sum" -v

    # Run with detailed failure output
    pytest tests/test_wpt_conformance.py -vv --tb=short
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import wpt_utils


# Directory containing WPT test data
WPT_DATA_DIR = Path(__file__).parent / "wpt_data" / "conformance"


def discover_wpt_operations() -> List[str]:
    """Discover all operations that have WPT test data available."""
    if not WPT_DATA_DIR.exists():
        return []

    operations = []
    for json_file in WPT_DATA_DIR.glob("*.json"):
        operations.append(json_file.stem)

    return sorted(operations)


def load_test_cases_for_operation(operation: str) -> List[Dict[str, Any]]:
    """Load all test cases for a given operation."""
    try:
        test_data = wpt_utils.load_wpt_test_data(operation, "conformance")
        return test_data.get("tests", [])
    except FileNotFoundError:
        pytest.skip(f"No WPT test data for {operation}")
        return []


def execute_wpt_test_case(context, test_case: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Execute a single WPT test case using the WebNN API.

    Args:
        context: MLContext instance
        test_case: Test case dictionary from WPT data

    Returns:
        Dictionary mapping output names to NumPy arrays
    """
    builder = context.create_graph_builder()

    # Create operands dictionary to track created operands by name
    operands: Dict[str, Any] = {}

    # Step 1: Create input operands
    inputs_data = test_case.get("inputs", {})
    for input_name, input_spec in inputs_data.items():
        shape = input_spec["shape"]
        dtype = input_spec.get("dataType", "float32")
        operands[input_name] = builder.input(input_name, shape, dtype)

    # Step 2: Execute operators in order
    operators = test_case.get("operators", [])
    for op_spec in operators:
        op_name = op_spec["name"]
        op_args = op_spec.get("arguments", {})
        op_output = op_spec.get("output", "output")

        # Resolve operand references in arguments
        resolved_args = {}
        for arg_name, arg_value in op_args.items():
            if isinstance(arg_value, str) and arg_value in operands:
                # This is a reference to another operand
                resolved_args[arg_name] = operands[arg_value]
            else:
                # This is a literal value
                resolved_args[arg_name] = arg_value

        # Call the appropriate builder method
        result = call_builder_method(builder, op_name, resolved_args)

        # Store result operand
        operands[op_output] = result

    # Step 3: Build graph with outputs
    expected_outputs = test_case.get("expectedOutputs", {})
    graph_outputs = {}
    for output_name in expected_outputs.keys():
        if output_name in operands:
            graph_outputs[output_name] = operands[output_name]

    if not graph_outputs:
        raise ValueError("No outputs specified in test case")

    # Build the graph
    graph = builder.build(graph_outputs)

    # Note: Actual compute() execution is not implemented yet
    # For now, we just validate that the graph builds successfully
    return {}


def call_builder_method(builder, op_name: str, args: Dict[str, Any]) -> Any:
    """
    Call a builder method by name with the given arguments.

    Args:
        builder: MLGraphBuilder instance
        op_name: Operation name (e.g., "reduce_sum", "relu")
        args: Arguments dictionary

    Returns:
        Resulting MLOperand
    """
    # Map operation names to builder method names
    method_name_map = {
        "reduce_sum": "reduce_sum",
        "reduce_mean": "reduce_mean",
        "reduce_max": "reduce_max",
        "reduce_min": "reduce_min",
        "reduce_product": "reduce_product",
        "reduce_l1": "reduce_l1",
        "reduce_l2": "reduce_l2",
        "reduce_log_sum": "reduce_log_sum",
        "reduce_log_sum_exp": "reduce_log_sum_exp",
        "reduce_sum_square": "reduce_sum_square",
        "relu": "relu",
        "sigmoid": "sigmoid",
        "tanh": "tanh",
        "softmax": "softmax",
        "add": "add",
        "sub": "sub",
        "mul": "mul",
        "div": "div",
        "matmul": "matmul",
        "reshape": "reshape",
    }

    method_name = method_name_map.get(op_name, op_name)

    if not hasattr(builder, method_name):
        pytest.skip(f"Operation {op_name} not implemented")

    method = getattr(builder, method_name)

    # Handle different argument patterns
    # For operations with a single input operand
    if "input" in args and len(args) == 1:
        return method(args["input"])

    # For operations with options (like reduction ops)
    if "input" in args:
        input_operand = args["input"]
        options = {k: v for k, v in args.items() if k != "input"}

        # Handle special option name mappings
        if "keepDimensions" in options:
            options["keep_dimensions"] = options.pop("keepDimensions")

        return method(input_operand, **options)

    # For binary operations
    if "a" in args and "b" in args:
        remaining = {k: v for k, v in args.items() if k not in ["a", "b"]}
        return method(args["a"], args["b"], **remaining)

    # Fallback: try calling with all args as kwargs
    return method(**args)


def generate_test_id(operation: str, test_case: Dict[str, Any]) -> str:
    """Generate a pytest test ID for a test case."""
    test_name = test_case.get("name", "unnamed")
    # Sanitize name for pytest
    test_id = f"{operation}::{test_name}".replace(" ", "_")
    return test_id


# Pytest fixtures and test generation
@pytest.fixture(scope="module")
def available_operations():
    """Fixture providing list of operations with WPT test data."""
    operations = discover_wpt_operations()
    if not operations:
        pytest.skip("No WPT test data found. Run: ./scripts/update_wpt_tests.sh")
    return operations


def pytest_generate_tests(metafunc):
    """
    Dynamically generate tests from WPT test data.

    This hook is called by pytest to parameterize test functions.
    """
    if "wpt_test_case" in metafunc.fixturenames:
        # Discover all operations
        operations = discover_wpt_operations()

        if not operations:
            # No test data - create a single skip test
            metafunc.parametrize(
                "wpt_test_case,operation",
                [(None, None)],
                ids=["no_wpt_data"]
            )
            return

        # Generate test parameters for all operations and their test cases
        test_params = []
        test_ids = []

        for operation in operations:
            test_cases = load_test_cases_for_operation(operation)

            if not test_cases:
                # Operation file exists but has no test cases
                test_params.append((None, operation))
                test_ids.append(f"{operation}::no_tests")
                continue

            for test_case in test_cases:
                test_params.append((test_case, operation))
                test_ids.append(generate_test_id(operation, test_case))

        metafunc.parametrize(
            "wpt_test_case,operation",
            test_params,
            ids=test_ids
        )


def test_wpt_conformance(context, wpt_test_case, operation):
    """
    Run a single WPT conformance test.

    This test is parameterized by pytest_generate_tests to run all WPT test cases.
    """
    if wpt_test_case is None and operation is None:
        pytest.skip("No WPT test data found. Run: ./scripts/update_wpt_tests.sh")

    if wpt_test_case is None:
        pytest.skip(f"No test cases for {operation} (may require manual conversion)")

    # Execute test case and get results
    try:
        results = execute_wpt_test_case(context, wpt_test_case)
    except NotImplementedError as e:
        pytest.skip(f"Not implemented: {e}")

    # For now, we just verify the graph builds successfully
    # Full execution and validation will be added once compute() is implemented
    pytest.skip("Graph execution (compute) not yet implemented - graph build validated")

    # TODO: Once compute() is implemented, add validation:
    # expected_outputs = wpt_test_case.get("expectedOutputs", {})
    # for output_name, expected_spec in expected_outputs.items():
    #     actual = results[output_name]
    #     expected = wpt_utils.numpy_array_from_test_data(expected_spec)
    #     tolerance = wpt_utils.get_operation_tolerance(operation, wpt_test_case)
    #     dtype = expected_spec.get("dataType", "float32")
    #
    #     passed, message, failures = wpt_utils.validate_result(
    #         actual, expected, tolerance, dtype
    #     )
    #
    #     if not passed:
    #         failure_msg = wpt_utils.format_test_failure(
    #             wpt_test_case.get("name", "unnamed"),
    #             failures
    #         )
    #         pytest.fail(f"{message}\n{failure_msg}")


# Mark all tests in this module
pytestmark = pytest.mark.wpt

#!/usr/bin/env python3
"""Integration tests for WebNN Python API with ONNX and CoreML conversion.

This module tests the complete workflow of building WebNN graphs and converting
them to both ONNX and CoreML formats. It validates that:
- Graph construction APIs work correctly
- ONNX conversion produces valid models
- CoreML conversion works for supported operations (macOS only)

Usage:
    python tests/test_integration.py [--cleanup]

Arguments:
    --cleanup: Remove generated model files after testing
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import webnn
except ImportError:
    print("Error: webnn module not found. Install with: maturin develop --features python")
    sys.exit(1)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(title)
    print('=' * 60)


def print_step(step: int, description: str) -> None:
    """Print a formatted test step."""
    print(f"\n{step}. {description}")


def print_success(message: str, indent: int = 3) -> None:
    """Print a success message with checkmark."""
    print(f"{' ' * indent}✓ {message}")


def print_info(message: str, indent: int = 5) -> None:
    """Print an info message."""
    print(f"{' ' * indent}- {message}")


def print_warning(message: str, indent: int = 3) -> None:
    """Print a warning message."""
    print(f"{' ' * indent}⚠ {message}")


def print_error(message: str, indent: int = 3) -> None:
    """Print an error message."""
    print(f"{' ' * indent}✗ {message}")


def test_simple_graph(context: "webnn.MLContext") -> "webnn.MLGraph":
    """Test building a simple graph: output = relu(x + y).

    Args:
        context: MLContext for graph building

    Returns:
        Compiled MLGraph
    """
    print_step(1, "Building simple graph: output = relu(x + y)")

    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    y = builder.input("y", [2, 3], "float32")
    z = builder.add(x, y)
    output = builder.relu(z)

    print_success(f"Input x: {x}")
    print_success(f"Input y: {y}")
    print_success(f"Addition result: {z}")
    print_success(f"ReLU output: {output}")

    graph = builder.build({"output": output})
    print_success("Graph compiled:")
    print_info(f"Operands: {graph.operand_count}")
    print_info(f"Operations: {graph.operation_count}")
    print_info(f"Inputs: {graph.get_input_names()}")
    print_info(f"Outputs: {graph.get_output_names()}")

    return graph


def test_complex_graph(context: "webnn.MLContext") -> "webnn.MLGraph":
    """Test building a complex graph with constants: output = sigmoid(input @ weights).

    Args:
        context: MLContext for graph building

    Returns:
        Compiled MLGraph
    """
    print_step(2, "Building complex graph with constants: output = sigmoid(input @ weights)")

    builder = context.create_graph_builder()
    input_tensor = builder.input("input", [1, 4], "float32")

    # Create constant weights
    weights = np.array([[0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                        [0.7, 0.8, 0.9],
                        [1.0, 1.1, 1.2]], dtype=np.float32)
    weights_const = builder.constant(weights)
    print_success(f"Created constant weights: shape {weights_const.shape}")

    # Matrix multiplication and activation
    matmul_result = builder.matmul(input_tensor, weights_const)
    print_success(f"MatMul result: {matmul_result}")

    final_output = builder.sigmoid(matmul_result)
    print_success(f"Sigmoid output: {final_output}")

    graph = builder.build({"output": final_output})
    print_success("Complex graph compiled:")
    print_info(f"Operands: {graph.operand_count}")
    print_info(f"Operations: {graph.operation_count}")

    return graph


def test_onnx_conversion(context: "webnn.MLContext", graph: "webnn.MLGraph",
                        output_path: str) -> bool:
    """Test ONNX conversion.

    Args:
        context: MLContext for conversion
        graph: Graph to convert
        output_path: Path to save ONNX model

    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        context.convert_to_onnx(graph, output_path)
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print_success(f"ONNX model saved: {output_path}")
            print_info(f"Size: {size:,} bytes")
            return True
        else:
            print_error("ONNX conversion failed - file not created")
            return False
    except Exception as e:
        print_error(f"ONNX conversion error: {e}")
        return False


def test_coreml_conversion(context: "webnn.MLContext", graph: "webnn.MLGraph",
                          output_path: str) -> bool:
    """Test CoreML conversion (macOS only).

    Args:
        context: MLContext for conversion
        graph: Graph to convert
        output_path: Path to save CoreML model

    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        context.convert_to_coreml(graph, output_path)
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print_success(f"CoreML model saved: {output_path}")
            print_info(f"Size: {size:,} bytes")
            return True
        else:
            print_error("CoreML conversion failed - file not created")
            return False
    except AttributeError:
        print_warning("CoreML not available (not on macOS or feature not enabled)")
        return False
    except Exception as e:
        print_warning(f"CoreML conversion: {e}")
        return False


def cleanup_files(files: list[str]) -> None:
    """Remove generated test files.

    Args:
        files: List of file paths to remove
    """
    print_step("Cleanup", "Removing generated files...")
    for file_path in files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print_success(f"Removed: {file_path}")


def main() -> int:
    """Run integration tests.

    Returns:
        0 if all tests passed, 1 otherwise
    """
    parser = argparse.ArgumentParser(
        description="Integration tests for WebNN Python API"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove generated model files after testing"
    )
    args = parser.parse_args()

    print_section("WebNN Python API - Integration Tests")

    # Initialize context
    print_step(0, "Initializing ML context...")
    ml = webnn.ML()
    context = ml.create_context(device_type="cpu")
    print_success("Context created")

    # Track generated files for cleanup
    generated_files = []
    test_results = []

    # Test 1: Simple graph
    try:
        simple_graph = test_simple_graph(context)
        test_results.append(("Simple graph build", True))
    except Exception as e:
        print_error(f"Simple graph build failed: {e}")
        test_results.append(("Simple graph build", False))
        return 1

    # Test 2: Complex graph
    try:
        complex_graph = test_complex_graph(context)
        test_results.append(("Complex graph build", True))
    except Exception as e:
        print_error(f"Complex graph build failed: {e}")
        test_results.append(("Complex graph build", False))
        return 1

    # Test 3: ONNX conversions
    print_step(3, "Testing ONNX conversions...")

    onnx_simple = "test_graph.onnx"
    generated_files.append(onnx_simple)
    result1 = test_onnx_conversion(context, simple_graph, onnx_simple)
    test_results.append(("ONNX simple graph", result1))

    onnx_complex = "test_complex_graph.onnx"
    generated_files.append(onnx_complex)
    result2 = test_onnx_conversion(context, complex_graph, onnx_complex)
    test_results.append(("ONNX complex graph", result2))

    # Test 4: CoreML conversions
    print_step(4, "Testing CoreML conversions (macOS only)...")

    coreml_simple = "test_graph.mlmodel"
    generated_files.append(coreml_simple)
    result3 = test_coreml_conversion(context, simple_graph, coreml_simple)
    test_results.append(("CoreML simple graph", result3))

    coreml_complex = "test_complex_graph.mlmodel"
    generated_files.append(coreml_complex)
    result4 = test_coreml_conversion(context, complex_graph, coreml_complex)
    test_results.append(("CoreML complex graph", result4))

    # Print summary
    print_section("Test Summary")
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)

    for name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nResults: {passed}/{total} tests passed")

    # Cleanup if requested
    if args.cleanup:
        cleanup_files(generated_files)
    else:
        print("\nGenerated files:")
        for file_path in generated_files:
            if os.path.exists(file_path):
                print(f"  - {file_path}")
        print("\nRun with --cleanup to remove generated files")

    print_section("Integration Tests Complete")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

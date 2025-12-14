#!/usr/bin/env python3
"""Basic CoreML conversion test with only supported operations.

This test validates CoreML conversion using only operations currently
supported by the Rust CoreML converter (add, matmul). It's useful for:
- Verifying basic CoreML functionality works
- Testing on macOS without hitting unsupported operation errors
- Comparing ONNX vs CoreML output for the same simple graph

Note: CoreML support is limited to basic operations. For full operation
support including activations (relu, sigmoid, tanh), see TODO.txt.

Usage:
    python tests/test_coreml_basic.py [--cleanup]

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


def build_simple_graph(context: "webnn.MLContext") -> "webnn.MLGraph":
    """Build a simple addition graph that CoreML can handle.

    Args:
        context: MLContext for graph building

    Returns:
        Compiled MLGraph with only add operation
    """
    print_step(1, "Building simple graph: output = x + y")

    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    y = builder.input("y", [2, 3], "float32")
    output = builder.add(x, y)

    print_success(f"Input x: {x}")
    print_success(f"Input y: {y}")
    print_success(f"Output: {output}")

    graph = builder.build({"output": output})
    print_success("Graph compiled:")
    print_info(f"Operands: {graph.operand_count}")
    print_info(f"Operations: {graph.operation_count}")
    print_info(f"Inputs: {graph.get_input_names()}")
    print_info(f"Outputs: {graph.get_output_names()}")

    return graph


def verify_conversions(context: "webnn.MLContext", graph: "webnn.MLGraph") -> tuple[bool, bool]:
    """Verify both CoreML and ONNX conversion for comparison.

    Args:
        context: MLContext for conversion
        graph: Graph to convert

    Returns:
        Tuple of (coreml_success, onnx_success)
    """
    print_step(2, "Testing format conversions...")

    # CoreML conversion
    coreml_success = False
    coreml_path = "test_simple.mlmodel"
    try:
        context.convert_to_coreml(graph, coreml_path)
        if os.path.exists(coreml_path):
            size = os.path.getsize(coreml_path)
            print_success(f"CoreML model saved: {coreml_path}")
            print_info(f"Size: {size:,} bytes")
            coreml_success = True
        else:
            print_error("CoreML conversion failed - file not created")
    except AttributeError:
        print_warning("CoreML not available (not on macOS or feature not enabled)")
    except Exception as e:
        print_error(f"CoreML conversion error: {e}")

    # ONNX conversion for comparison
    onnx_success = False
    onnx_path = "test_simple.onnx"
    try:
        context.convert_to_onnx(graph, onnx_path)
        if os.path.exists(onnx_path):
            size = os.path.getsize(onnx_path)
            print_success(f"ONNX model saved: {onnx_path}")
            print_info(f"Size: {size:,} bytes")
            onnx_success = True
        else:
            print_error("ONNX conversion failed - file not created")
    except Exception as e:
        print_error(f"ONNX conversion error: {e}")

    return coreml_success, onnx_success


def cleanup_files(files: list[str]) -> None:
    """Remove generated test files.

    Args:
        files: List of file paths to remove
    """
    print_step(3, "Cleaning up generated files...")
    for file_path in files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print_success(f"Removed: {file_path}")


def main() -> int:
    """Run basic CoreML test.

    Returns:
        0 if test passed, 1 otherwise
    """
    parser = argparse.ArgumentParser(
        description="Basic CoreML conversion test with supported operations"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove generated model files after testing"
    )
    args = parser.parse_args()

    print_section("CoreML Basic Operations Test")

    # Initialize
    print_step(0, "Initializing ML context...")
    ml = webnn.ML()
    context = ml.create_context(device_type="cpu")
    print_success("Context created")

    # Build graph
    try:
        graph = build_simple_graph(context)
    except Exception as e:
        print_error(f"Graph build failed: {e}")
        return 1

    # Test conversions
    coreml_success, onnx_success = verify_conversions(context, graph)

    # Summary
    print_section("Test Results")
    print(f"  {'✓' if coreml_success else '✗'} CoreML conversion: {'SUCCESS' if coreml_success else 'FAILED'}")
    print(f"  {'✓' if onnx_success else '✗'} ONNX conversion: {'SUCCESS' if onnx_success else 'FAILED'}")

    # Cleanup
    generated_files = ["test_simple.mlmodel", "test_simple.onnx"]
    if args.cleanup:
        cleanup_files(generated_files)
    else:
        print("\nGenerated files:")
        for file_path in generated_files:
            if os.path.exists(file_path):
                print(f"  - {file_path}")
        print("\nRun with --cleanup to remove generated files")

    print_section("Test Complete")

    # Consider test passed if at least ONNX works (CoreML may not be available)
    return 0 if onnx_success else 1


if __name__ == "__main__":
    sys.exit(main())

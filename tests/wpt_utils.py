"""
WPT (Web Platform Tests) Utilities for WebNN

This module provides utilities for running W3C Web Platform Tests against the rustnn
implementation. It includes tolerance checking, test data loading, and result validation
compatible with the official WPT WebNN test suite.

Based on: https://github.com/web-platform-tests/wpt/tree/master/webnn
"""

import struct
import numpy as np
from typing import Dict, List, Union, Any, Optional
from pathlib import Path
import json


def ulp_distance(a: float, b: float, dtype: str = "float32") -> int:
    """
    Calculate ULP (Units in Last Place) distance between two floating-point values.

    ULP distance is a measure of floating-point precision that counts the number
    of representable values between two numbers.

    Args:
        a: First value
        b: Second value
        dtype: Data type ("float32" or "float16")

    Returns:
        Integer ULP distance

    Raises:
        ValueError: If dtype is not supported for ULP calculation

    Examples:
        >>> ulp_distance(1.0, 1.0000001, "float32")
        1
        >>> ulp_distance(0.0, 0.0, "float32")
        0
    """
    # Handle special cases
    if np.isnan(a) or np.isnan(b):
        if np.isnan(a) and np.isnan(b):
            return 0
        return float('inf')

    if np.isinf(a) or np.isinf(b):
        if a == b:
            return 0
        return float('inf')

    if dtype == "float32":
        # Convert to int32 bit representation
        # We use unsigned interpretation for ULP distance
        a_bits = struct.unpack('!I', struct.pack('!f', float(a)))[0]
        b_bits = struct.unpack('!I', struct.pack('!f', float(b)))[0]

        # Handle sign bit: if signs differ, distance goes through zero
        if (a_bits ^ b_bits) & 0x80000000:
            # Different signs - distance is sum of distances to zero
            # This matches WPT behavior for crossing zero
            a_dist_to_zero = a_bits & 0x7FFFFFFF
            b_dist_to_zero = b_bits & 0x7FFFFFFF
            return a_dist_to_zero + b_dist_to_zero

        return abs(int(a_bits) - int(b_bits))

    elif dtype == "float16":
        # Use numpy float16
        a_half = np.float16(a)
        b_half = np.float16(b)
        a_bits = a_half.view(np.uint16)
        b_bits = b_half.view(np.uint16)

        # Handle sign bit crossing
        if (int(a_bits) ^ int(b_bits)) & 0x8000:
            a_dist_to_zero = int(a_bits) & 0x7FFF
            b_dist_to_zero = int(b_bits) & 0x7FFF
            return a_dist_to_zero + b_dist_to_zero

        return abs(int(a_bits) - int(b_bits))

    else:
        raise ValueError(f"ULP distance not supported for dtype: {dtype}")


def check_ulp_tolerance(
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: int,
    dtype: str = "float32"
) -> tuple[bool, List[Dict[str, Any]]]:
    """
    Check if actual values are within ULP tolerance of expected values.

    Args:
        actual: Actual output array
        expected: Expected output array
        tolerance: Maximum allowed ULP distance
        dtype: Data type for ULP calculation

    Returns:
        Tuple of (all_pass, failures) where failures is a list of failure details
    """
    if actual.shape != expected.shape:
        return False, [{
            "reason": "shape_mismatch",
            "actual_shape": actual.shape,
            "expected_shape": expected.shape
        }]

    actual_flat = actual.flatten()
    expected_flat = expected.flatten()

    failures = []
    for i, (a, e) in enumerate(zip(actual_flat, expected_flat)):
        ulp_dist = ulp_distance(float(a), float(e), dtype)
        if ulp_dist > tolerance:
            failures.append({
                "index": i,
                "actual": float(a),
                "expected": float(e),
                "ulp_distance": ulp_dist,
                "tolerance": tolerance
            })

    return len(failures) == 0, failures


def check_atol_tolerance(
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: float
) -> tuple[bool, List[Dict[str, Any]]]:
    """
    Check if actual values are within absolute tolerance of expected values.

    Args:
        actual: Actual output array
        expected: Expected output array
        tolerance: Maximum allowed absolute difference

    Returns:
        Tuple of (all_pass, failures) where failures is a list of failure details
    """
    if actual.shape != expected.shape:
        return False, [{
            "reason": "shape_mismatch",
            "actual_shape": actual.shape,
            "expected_shape": expected.shape
        }]

    actual_flat = actual.flatten()
    expected_flat = expected.flatten()

    failures = []
    for i, (a, e) in enumerate(zip(actual_flat, expected_flat)):
        abs_diff = abs(float(a) - float(e))
        if abs_diff > tolerance:
            failures.append({
                "index": i,
                "actual": float(a),
                "expected": float(e),
                "absolute_difference": abs_diff,
                "tolerance": tolerance
            })

    return len(failures) == 0, failures


def get_operation_tolerance(
    operation: str,
    test_case: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get tolerance specification for an operation.

    Returns tolerance from test case if specified, otherwise returns default.

    Args:
        operation: Operation name (e.g., "reduce_sum", "relu")
        test_case: Test case dictionary containing optional tolerance specification

    Returns:
        Tolerance dict with "type" and "value" keys
    """
    # Check if test case specifies tolerance
    if "tolerance" in test_case:
        return test_case["tolerance"]

    # Default tolerances based on operation
    # These match WPT tolerances for common operations
    DEFAULT_TOLERANCES = {
        # Exact operations (no rounding)
        "relu": {"type": "ULP", "value": 0},
        "add": {"type": "ULP", "value": 0},
        "sub": {"type": "ULP", "value": 0},
        "mul": {"type": "ULP", "value": 0},
        "reshape": {"type": "ULP", "value": 0},
        "reduce_sum": {"type": "ULP", "value": 0},
        "reduce_max": {"type": "ULP", "value": 0},
        "reduce_min": {"type": "ULP", "value": 0},

        # Approximate operations (allow small rounding)
        "sigmoid": {"type": "ULP", "value": 34},  # float32
        "tanh": {"type": "ULP", "value": 44},     # float32
        "softmax": {"type": "ULP", "value": 100}, # Accumulates error
        "div": {"type": "ULP", "value": 2},
        "reduce_mean": {"type": "ULP", "value": 2},
        "reduce_product": {"type": "ULP", "value": 10},
        "reduce_l1": {"type": "ULP", "value": 2},
        "reduce_l2": {"type": "ULP", "value": 5},
        "reduce_log_sum": {"type": "ULP", "value": 10},
        "reduce_log_sum_exp": {"type": "ULP", "value": 100},
        "reduce_sum_square": {"type": "ULP", "value": 2},

        # Convolution and pooling (backend-dependent)
        "conv2d": {"type": "ULP", "value": 100},
        "conv_transpose2d": {"type": "ULP", "value": 100},
        "average_pool2d": {"type": "ULP", "value": 2},
        "max_pool2d": {"type": "ULP", "value": 0},
        "global_average_pool": {"type": "ULP", "value": 2},
        "global_max_pool": {"type": "ULP", "value": 0},

        # Normalization (accumulates rounding)
        "batch_normalization": {"type": "ULP", "value": 100},
        "instance_normalization": {"type": "ULP", "value": 100},
        "layer_normalization": {"type": "ULP", "value": 100},

        # Matrix operations
        "matmul": {"type": "ULP", "value": 100},  # Depends on size
    }

    return DEFAULT_TOLERANCES.get(operation, {"type": "ULP", "value": 100})


def validate_result(
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: Dict[str, Any],
    dtype: str = "float32"
) -> tuple[bool, str, Optional[List[Dict[str, Any]]]]:
    """
    Validate actual result against expected with tolerance.

    Args:
        actual: Actual output array
        expected: Expected output array
        tolerance: Tolerance specification dict
        dtype: Data type for ULP calculation

    Returns:
        Tuple of (passed, message, failures)
    """
    tolerance_type = tolerance.get("type", "ULP")
    tolerance_value = tolerance.get("value", 0)

    if tolerance_type == "ULP":
        passed, failures = check_ulp_tolerance(actual, expected, tolerance_value, dtype)
        if passed:
            return True, f"PASS (ULP ≤ {tolerance_value})", None
        else:
            max_ulp = max(f["ulp_distance"] for f in failures if "ulp_distance" in f)
            return False, f"FAIL (max ULP: {max_ulp}, tolerance: {tolerance_value})", failures

    elif tolerance_type == "ATOL":
        passed, failures = check_atol_tolerance(actual, expected, tolerance_value)
        if passed:
            return True, f"PASS (ATOL ≤ {tolerance_value})", None
        else:
            max_diff = max(f["absolute_difference"] for f in failures if "absolute_difference" in f)
            return False, f"FAIL (max diff: {max_diff:.2e}, tolerance: {tolerance_value})", failures

    else:
        raise ValueError(f"Unknown tolerance type: {tolerance_type}")


def load_wpt_test_data(
    operation: str,
    category: str = "conformance",
    wpt_data_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Load WPT test data for an operation.

    Args:
        operation: Operation name (e.g., "reduce_sum", "relu")
        category: Test category ("conformance" or "validation")
        wpt_data_dir: Optional custom WPT data directory path

    Returns:
        Dict containing test data with "tests" key

    Raises:
        FileNotFoundError: If test data file doesn't exist
    """
    if wpt_data_dir is None:
        wpt_data_dir = Path(__file__).parent / "wpt_data"

    test_file = wpt_data_dir / category / f"{operation}.json"

    if not test_file.exists():
        raise FileNotFoundError(
            f"WPT test data not found: {test_file}\n"
            f"Run: python scripts/convert_wpt_tests.py --operation {operation}"
        )

    with open(test_file) as f:
        return json.load(f)


def format_test_failure(
    test_name: str,
    failures: List[Dict[str, Any]],
    max_failures_shown: int = 5
) -> str:
    """
    Format test failure details for human-readable output.

    Args:
        test_name: Name of the test that failed
        failures: List of failure details
        max_failures_shown: Maximum number of failures to show

    Returns:
        Formatted failure message
    """
    lines = [f"\n❌ Test failed: {test_name}"]
    lines.append(f"   Total failures: {len(failures)}")
    lines.append(f"   Showing first {min(len(failures), max_failures_shown)} failures:")

    for i, failure in enumerate(failures[:max_failures_shown]):
        if "ulp_distance" in failure:
            lines.append(
                f"   [{i}] index={failure['index']}: "
                f"actual={failure['actual']:.6f}, expected={failure['expected']:.6f}, "
                f"ULP={failure['ulp_distance']} (tolerance={failure['tolerance']})"
            )
        elif "absolute_difference" in failure:
            lines.append(
                f"   [{i}] index={failure['index']}: "
                f"actual={failure['actual']:.6f}, expected={failure['expected']:.6f}, "
                f"diff={failure['absolute_difference']:.2e} (tolerance={failure['tolerance']})"
            )
        else:
            lines.append(f"   [{i}] {failure}")

    if len(failures) > max_failures_shown:
        lines.append(f"   ... and {len(failures) - max_failures_shown} more failures")

    return "\n".join(lines)


def numpy_array_from_test_data(test_data: Dict[str, Any]) -> np.ndarray:
    """
    Create NumPy array from WPT test data specification.

    Args:
        test_data: Dict with "data", "shape", and "dataType" keys

    Returns:
        NumPy array with specified shape and dtype
    """
    data = test_data["data"]
    shape = test_data["shape"]
    dtype_str = test_data.get("dataType", "float32")

    # Map WebNN data types to NumPy dtypes
    DTYPE_MAP = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "uint32": np.uint32,
        "int8": np.int8,
        "uint8": np.uint8,
        "int64": np.int64,
        "uint64": np.uint64,
    }

    np_dtype = DTYPE_MAP.get(dtype_str, np.float32)
    return np.array(data, dtype=np_dtype).reshape(shape)


# Tolerance presets for different operation categories
TOLERANCE_PRESETS = {
    "exact": {"type": "ULP", "value": 0},
    "low": {"type": "ULP", "value": 2},
    "medium": {"type": "ULP", "value": 10},
    "high": {"type": "ULP", "value": 100},
    "very_high": {"type": "ULP", "value": 1000},
}

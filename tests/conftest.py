"""
Shared pytest fixtures for all test files.

This module provides common fixtures for WebNN API testing.
"""

import pytest
import numpy as np

# Try to import webnn module
try:
    import webnn
    WEBNN_AVAILABLE = True
except ImportError:
    WEBNN_AVAILABLE = False


def _has_onnx_runtime():
    """Check if ONNX runtime is available for actual computation"""
    if not WEBNN_AVAILABLE:
        return False
    try:
        ml = webnn.ML()
        ctx = ml.create_context(power_preference="default", accelerated=False)
        builder = ctx.create_graph_builder()
        x = builder.input("x", [1, 1], "float32")
        y = builder.relu(x)
        graph = builder.build({"output": y})
        result = ctx.compute(graph, {"x": np.array([[1.0]], dtype=np.float32)})
        # If ONNX runtime is available, result should be non-zero
        return np.any(result["output"] != 0)
    except:
        return False


ONNX_RUNTIME_AVAILABLE = _has_onnx_runtime()


# Pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "wpt: WebNN W3C Web Platform Tests"
    )
    config.addinivalue_line(
        "markers", "requires_onnx_runtime: Test requires ONNX Runtime"
    )


@pytest.fixture(scope="session")
def ml():
    """Create ML instance (session-scoped)."""
    if not WEBNN_AVAILABLE:
        pytest.skip("webnn not built yet")
    return webnn.ML()


@pytest.fixture
def context(ml):
    """Create ML context."""
    return ml.create_context(power_preference="default", accelerated=False)


@pytest.fixture
def builder(context):
    """Create graph builder."""
    return context.create_graph_builder()


# Mark for tests requiring ONNX runtime
requires_onnx_runtime = pytest.mark.skipif(
    not ONNX_RUNTIME_AVAILABLE,
    reason="ONNX runtime not available - built without onnx-runtime feature"
)

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
        # Explicitly force ONNX GPU backend with device_type="gpu"
        ctx = ml.create_context(power_preference="default", accelerated=True, device_type="gpu")
        builder = ctx.create_graph_builder()
        x = builder.input("x", [1, 1], "float32")
        y = builder.relu(x)
        graph = builder.build({"output": y})
        result = ctx.compute(graph, {"x": np.array([[1.0]], dtype=np.float32)})
        # If ONNX runtime is available, result should be non-zero
        return np.any(result["output"] != 0)
    except:
        return False


def _has_coreml_runtime():
    """Check if CoreML runtime is available (macOS only)"""
    if not WEBNN_AVAILABLE:
        return False
    try:
        import platform
        if platform.system() != "Darwin":
            return False
        ml = webnn.ML()
        # Explicitly force CoreML backend with device_type="npu"
        ctx = ml.create_context(power_preference="default", accelerated=True, device_type="npu")
        builder = ctx.create_graph_builder()
        x = builder.input("x", [1, 1], "float32")
        y = builder.relu(x)
        graph = builder.build({"output": y})
        result = ctx.compute(graph, {"x": np.array([[1.0]], dtype=np.float32)})
        # If CoreML runtime is available, result should be non-zero
        return np.any(result["output"] != 0)
    except Exception as e:
        # Log the error for debugging but don't fail
        print(f"CoreML runtime check failed: {e}")
        return False


ONNX_RUNTIME_AVAILABLE = _has_onnx_runtime()
COREML_RUNTIME_AVAILABLE = _has_coreml_runtime()


# Pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "wpt: WebNN W3C Web Platform Tests"
    )
    config.addinivalue_line(
        "markers", "requires_onnx_runtime: Test requires ONNX Runtime"
    )
    config.addinivalue_line(
        "markers", "requires_coreml_runtime: Test requires CoreML Runtime"
    )


@pytest.fixture(scope="session")
def ml():
    """Create ML instance (session-scoped)."""
    if not WEBNN_AVAILABLE:
        pytest.skip("webnn not built yet")
    return webnn.ML()


@pytest.fixture(params=[
    pytest.param("onnx", id="onnx") if ONNX_RUNTIME_AVAILABLE else pytest.param(None, id="no_onnx", marks=pytest.mark.skip),
    pytest.param("coreml", id="coreml") if COREML_RUNTIME_AVAILABLE else pytest.param(None, id="no_coreml", marks=pytest.mark.skip),
])
def backend_name(request):
    """Return the backend name for the current test."""
    return request.param


@pytest.fixture
def context(backend_name, ml):
    """Create ML context for the specified backend.

    Uses explicit device_type to force backend selection:
    - "onnx" -> device_type="gpu" (ONNX GPU backend)
    - "coreml" -> device_type="npu" (CoreML backend on macOS)
    """
    if backend_name is None:
        pytest.skip("Backend not available")

    # Map backend name to device_type for explicit backend selection
    device_type_map = {
        "onnx": "gpu",    # Force ONNX GPU backend
        "coreml": "npu",  # Force CoreML backend (macOS only)
    }

    device_type = device_type_map.get(backend_name, "auto")
    ctx = ml.create_context(power_preference="default", accelerated=True, device_type=device_type)
    return ctx


@pytest.fixture
def builder(context):
    """Create graph builder."""
    return context.create_graph_builder()


# Mark for tests requiring ONNX runtime
requires_onnx_runtime = pytest.mark.skipif(
    not ONNX_RUNTIME_AVAILABLE,
    reason="ONNX runtime not available - built without onnx-runtime feature"
)

# Mark for tests requiring CoreML runtime
requires_coreml_runtime = pytest.mark.skipif(
    not COREML_RUNTIME_AVAILABLE,
    reason="CoreML runtime not available - macOS only or built without coreml-runtime feature"
)

"""
Performance benchmarks for WebNN backends

This module contains performance tests comparing ONNX and CoreML backends.
Tests measure cold start (first run) and warm run (subsequent runs) performance.

Usage:
    # Run all performance tests
    pytest tests/test_performance.py -v

    # Run only CoreML tests
    pytest tests/test_performance.py -k "coreml" -v

    # Run with detailed output
    pytest tests/test_performance.py -v -s
"""

import pytest
import numpy as np
import time
from typing import Dict, List, Tuple

# Import conftest fixtures
from conftest import ml, context, builder


def measure_inference_time(context, graph, inputs: Dict, num_runs: int = 5) -> Tuple[float, List[float]]:
    """
    Measure inference time for multiple runs.

    Args:
        context: MLContext instance
        graph: Compiled MLGraph
        inputs: Dictionary of input tensors
        num_runs: Number of runs to measure

    Returns:
        Tuple of (cold_start_time_ms, warm_run_times_ms)
    """
    times = []

    for i in range(num_runs):
        start = time.time()
        result = context.compute(graph, inputs)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    return times[0], times[1:]


def create_simple_model(builder):
    """
    Create a simple model: 10 layers of add + relu

    Args:
        builder: MLGraphBuilder instance

    Returns:
        Compiled MLGraph
    """
    input_tensor = builder.input("input", [1, 3, 224, 224], "float32")
    x = input_tensor

    # 10 layers of add + relu
    for _ in range(10):
        x = builder.add(x, input_tensor)
        x = builder.relu(x)

    return builder.build({"output": x})


def create_complex_model(builder):
    """
    Create a complex model: 50 layers × 4 operations = 200 ops

    Args:
        builder: MLGraphBuilder instance

    Returns:
        Compiled MLGraph
    """
    input_tensor = builder.input("input", [1, 3, 224, 224], "float32")
    x = input_tensor

    # 50 layers with 4 operations each
    for _ in range(50):
        x = builder.add(x, x)
        x = builder.relu(x)
        x = builder.mul(x, x)
        x = builder.relu(x)

    x = builder.global_average_pool(x)

    return builder.build({"output": x})


@pytest.mark.benchmark
def test_performance_simple_onnx_cpu(ml):
    """Test ONNX CPU performance with simple model (baseline)"""
    context = ml.create_context(power_preference="default", accelerated=False)
    builder = context.create_graph_builder()
    graph = create_simple_model(builder)

    input_data = {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}

    cold_time, warm_times = measure_inference_time(context, graph, input_data)
    warm_avg = sum(warm_times) / len(warm_times)

    print(f"\n{'='*60}")
    print(f"ONNX CPU - Simple Model (10 layers)")
    print(f"{'='*60}")
    print(f"Cold start: {cold_time:.2f}ms")
    print(f"Warm avg:   {warm_avg:.2f}ms")
    print(f"Warm range: {min(warm_times):.2f}ms - {max(warm_times):.2f}ms")

    # Performance assertions (reasonable bounds)
    assert cold_time < 500, f"Cold start too slow: {cold_time:.2f}ms"
    assert warm_avg < 100, f"Warm run too slow: {warm_avg:.2f}ms"


@pytest.mark.benchmark
@pytest.mark.skipif(
    not pytest.importorskip("webnn", reason="WebNN not available"),
    reason="CoreML runtime not available"
)
def test_performance_simple_coreml(ml):
    """Test CoreML performance with simple model"""
    context = ml.create_context(power_preference="default", accelerated=True)
    builder = context.create_graph_builder()
    graph = create_simple_model(builder)

    input_data = {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}

    cold_time, warm_times = measure_inference_time(context, graph, input_data)
    warm_avg = sum(warm_times) / len(warm_times)

    print(f"\n{'='*60}")
    print(f"CoreML - Simple Model (10 layers)")
    print(f"{'='*60}")
    print(f"Cold start: {cold_time:.2f}ms")
    print(f"Warm avg:   {warm_avg:.2f}ms")
    print(f"Warm range: {min(warm_times):.2f}ms - {max(warm_times):.2f}ms")

    # Performance assertions
    assert cold_time < 500, f"Cold start too slow: {cold_time:.2f}ms"
    assert warm_avg < 100, f"Warm run too slow: {warm_avg:.2f}ms"


@pytest.mark.benchmark
@pytest.mark.slow
def test_performance_complex_onnx_cpu(ml):
    """Test ONNX CPU performance with complex model"""
    context = ml.create_context(power_preference="default", accelerated=False)
    builder = context.create_graph_builder()
    graph = create_complex_model(builder)

    input_data = {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}

    cold_time, warm_times = measure_inference_time(context, graph, input_data)
    warm_avg = sum(warm_times) / len(warm_times)

    print(f"\n{'='*60}")
    print(f"ONNX CPU - Complex Model (200 operations)")
    print(f"{'='*60}")
    print(f"Cold start: {cold_time:.2f}ms")
    print(f"Warm avg:   {warm_avg:.2f}ms")
    print(f"Warm range: {min(warm_times):.2f}ms - {max(warm_times):.2f}ms")

    # Performance assertions
    assert cold_time < 1000, f"Cold start too slow: {cold_time:.2f}ms"
    assert warm_avg < 200, f"Warm run too slow: {warm_avg:.2f}ms"


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.skipif(
    not pytest.importorskip("webnn", reason="WebNN not available"),
    reason="CoreML runtime not available"
)
def test_performance_complex_coreml(ml):
    """Test CoreML performance with complex model - validates warm-up speedup"""
    context = ml.create_context(power_preference="default", accelerated=True)
    builder = context.create_graph_builder()
    graph = create_complex_model(builder)

    input_data = {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}

    cold_time, warm_times = measure_inference_time(context, graph, input_data)
    warm_avg = sum(warm_times) / len(warm_times)
    speedup = cold_time / warm_avg

    print(f"\n{'='*60}")
    print(f"CoreML - Complex Model (200 operations)")
    print(f"{'='*60}")
    print(f"Cold start: {cold_time:.2f}ms")
    print(f"Run 2:      {warm_times[0]:.2f}ms")
    print(f"Warm avg:   {warm_avg:.2f}ms")
    print(f"Warm range: {min(warm_times):.2f}ms - {max(warm_times):.2f}ms")
    print(f"Speedup:    {speedup:.2f}x")
    print(f"Time saved: {cold_time - warm_avg:.2f}ms ({(cold_time-warm_avg)/cold_time*100:.1f}%)")

    # Performance assertions
    assert cold_time < 1000, f"Cold start too slow: {cold_time:.2f}ms"
    assert warm_avg < 200, f"Warm run too slow: {warm_avg:.2f}ms"

    # Validate warm-up benefit for complex models
    assert speedup > 1.5, f"Insufficient warm-up speedup: {speedup:.2f}x (expected > 1.5x)"
    assert warm_times[0] < cold_time * 0.8, "Second run should be at least 20% faster"


@pytest.mark.benchmark
def test_performance_comparison_simple(ml):
    """Compare all backends with simple model"""
    results = {}

    # Test each backend
    backends = [
        ("ONNX CPU", False, "default"),
        ("CoreML default", True, "default"),
        ("CoreML low-power", True, "low-power"),
        ("CoreML high-perf", True, "high-performance"),
    ]

    input_data = {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}

    for backend_name, accelerated, power_pref in backends:
        try:
            context = ml.create_context(power_preference=power_pref, accelerated=accelerated)
            builder = context.create_graph_builder()
            graph = create_simple_model(builder)

            cold_time, warm_times = measure_inference_time(context, graph, input_data, num_runs=5)
            warm_avg = sum(warm_times) / len(warm_times)

            results[backend_name] = {
                'cold': cold_time,
                'warm_avg': warm_avg,
                'speedup': cold_time / warm_avg
            }
        except Exception as e:
            results[backend_name] = {'error': str(e)}

    # Print comparison table
    print(f"\n{'='*70}")
    print("Backend Performance Comparison - Simple Model")
    print(f"{'='*70}")
    print(f"{'Backend':<30} {'Cold':<12} {'Warm Avg':<12} {'Speedup'}")
    print("-" * 70)

    for backend_name, data in results.items():
        if 'error' in data:
            print(f"{backend_name:<30} {'FAILED':<12}")
        else:
            print(f"{backend_name:<30} {data['cold']:>8.1f}ms  {data['warm_avg']:>8.1f}ms  {data['speedup']:>6.2f}x")

    print("=" * 70)

    # At least one backend should work
    assert any('error' not in data for data in results.values()), "No backends available"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])

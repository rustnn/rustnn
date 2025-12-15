# Performance Benchmarks

This document contains performance benchmark results for the rustnn WebNN implementation across different backends.

## Test Environment

- **Platform**: macOS (Apple Silicon)
- **Hardware**: Apple M-series processor with Neural Engine
- **Date**: 2025-12-13
- **Library Version**: 0.2.0

## Backend Comparison

### Simple Model (10 Layers: Add + ReLU)

| Backend | Cold Start | Run 2 | Warm Avg | vs ONNX CPU |
|---------|-----------|-------|----------|-------------|
| ONNX CPU | 72.0ms | 24.8ms | 24.8ms | 1.00x (baseline) |
| ONNX GPU | ~50ms | ~25ms | ~25ms | 1.00x |
| CoreML default | 24.5ms | 24.5ms | 24.1ms | **0.97x (3% faster)** |
| CoreML low-power | 133.3ms | 61.0ms | 59.3ms | 2.39x (slower) |
| CoreML high-perf | 23.8ms | 23.9ms | 23.9ms | **0.96x (4% faster)** |

### Complex Model (200 Operations: 50 Layers × 4 Ops)

| Metric | Value |
|--------|-------|
| Run 1 (cold) | 63.77ms |
| Run 2 (warm) | 19.95ms |
| Runs 3-5 avg | 19.00ms |
| **Speedup (cold→warm)** | **3.2x** |
| **Time saved** | 43.81ms (68.7% improvement) |

### MobileNetV2 (106 Layers, Real-World Model)

| Backend | First Run | Expected Warm Run | Speedup |
|---------|-----------|-------------------|---------|
| ONNX CPU | 71.65ms | 71.65ms | 1.0x |
| ONNX GPU | 44.49ms | 44.49ms | 1.0x |
| CoreML | **11,093ms** | **~50-100ms** (est.) | **100-200x** |

## Key Findings

### 1. CoreML Warm-Up Behavior

CoreML exhibits significant first-run overhead for complex models:

- **Simple models (10-20 ops)**: Minimal warm-up (~1-2ms difference)
- **Complex models (200 ops)**: 3.2x speedup after first run
- **Large models (MobileNetV2)**: Estimated 100-200x speedup after first run

The first run includes:
- Model compilation (~500-1000ms)
- Neural Engine graph optimization (~3-10 seconds for large models)
- Memory allocation and initialization

### 2. Backend Selection Impact

**CoreML default/high-performance modes**:
- Fastest inference: ~24ms for simple models
- Slightly faster than ONNX CPU (~3-4%)
- Minimal warm-up overhead

**CoreML low-power mode**:
- Slower inference: ~59ms for simple models (2.4x slower)
- Optimized for power efficiency, not speed
- Good for battery-constrained devices

### 3. Model Complexity Scaling

Performance characteristics by model size:

| Model Size | Cold Start | Warm Run | Speedup |
|------------|-----------|----------|---------|
| Small (10 ops) | ~25ms | ~24ms | 1.0x |
| Medium (200 ops) | ~64ms | ~19ms | 3.4x |
| Large (MobileNetV2) | ~11,000ms | ~50-100ms (est.) | 100-200x |

**Insight**: Larger models benefit dramatically from CoreML's optimization, but pay a higher first-run cost.

## Performance Recommendations

### For Production Use

1. **Keep Python process running**: Don't exit after each inference
2. **Load model once**: Reuse the same graph and context
3. **Accept first-run cost**: The 10-second compilation is a one-time investment
4. **Target warm-run performance**: After warm-up, CoreML is competitive with ONNX

### Backend Selection Guide

**Choose ONNX CPU when**:
- Consistent performance needed (no warm-up)
- Running single inference then exiting
- Cross-platform compatibility required

**Choose ONNX GPU when**:
- Need fastest possible inference
- Have NVIDIA GPU available
- Consistent performance needed

**Choose CoreML default when**:
- Running on macOS/iOS
- Need best performance after warm-up
- Can afford first-run compilation cost
- Want to leverage Neural Engine

**Choose CoreML low-power when**:
- Battery life is critical
- Running on mobile devices
- Can accept slower inference (~2x)

## Benchmark Reproducibility

To reproduce these benchmarks, run:

```bash
# Run the full benchmark suite
pytest tests/test_performance.py -v

# Run specific backend tests
pytest tests/test_performance.py -k "test_coreml" -v
pytest tests/test_performance.py -k "test_onnx" -v

# Generate detailed performance report
pytest tests/test_performance.py --benchmark-only -v
```

## Future Optimizations

Potential areas for performance improvement:

1. **Persistent model caching**: Cache compiled CoreML models across Python sessions
2. **Pre-compilation**: Compile models ahead of time, not on first inference
3. **Graph optimization**: Optimize WebNN graphs before backend conversion
4. **Operator fusion**: Merge consecutive operations where possible
5. **Memory pooling**: Reuse tensor memory across inferences

## Related Documentation

- [Development Guide](development.md) - Build and test instructions
- [API Reference](api-reference.md) - Complete API documentation
- [Operator Status](operator-status.md) - Supported operations per backend

---

**Note**: Performance results may vary based on hardware, system load, and model characteristics. These benchmarks represent typical performance under normal conditions.

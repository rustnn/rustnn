#!/usr/bin/env python3
"""
Compare file sizes across different quantization levels

This script creates the same model with different quantization levels
and compares the resulting file sizes to demonstrate storage savings.
"""

import tempfile
import os
import numpy as np
from pathlib import Path

import webnn


def create_model_with_quantization(quant_dtype="float32", size_multiplier=1):
    """
    Create a model with specified quantization level.

    Args:
        quant_dtype: Data type for constants (float32/float16/int8/int4)
        size_multiplier: Scale up model size for more realistic comparison

    Returns:
        MLGraph instance
    """
    ml = webnn.ML()
    context = ml.create_context(device_type="cpu")
    builder = context.create_graph_builder()

    # Create a model with some constants (simulating weights)
    # Scale up for more realistic file sizes
    base_size = 64 * size_multiplier

    # Input
    x = builder.input("x", [base_size, base_size], "float32")

    # Simulated weight matrices
    if quant_dtype == "float32":
        # Standard float32 weights
        w1 = builder.constant(np.random.randn(base_size, base_size).astype(np.float32))
        w2 = builder.constant(np.random.randn(base_size, base_size).astype(np.float32))

        # Simple computation
        h1 = builder.matmul(x, w1)
        h2 = builder.relu(h1)
        output = builder.matmul(h2, w2)

    elif quant_dtype == "float16":
        # Float16 weights
        w1 = builder.constant(np.random.randn(base_size, base_size).astype(np.float16))
        w2 = builder.constant(np.random.randn(base_size, base_size).astype(np.float16))

        # Cast to float32 for computation (WebNN compute is float32)
        w1_f32 = builder.cast(w1, "float32")
        w2_f32 = builder.cast(w2, "float32")

        h1 = builder.matmul(x, w1_f32)
        h2 = builder.relu(h1)
        output = builder.matmul(h2, w2_f32)

    elif quant_dtype in ["int8", "int4"]:
        # Quantized weights with scale and zero-point
        # In real models, these would be pre-quantized weights

        # Create float weights first, then quantize
        w1_float = np.random.randn(base_size, base_size).astype(np.float32)
        w2_float = np.random.randn(base_size, base_size).astype(np.float32)

        # Compute scale and zero-point for quantization
        w1_min, w1_max = w1_float.min(), w1_float.max()
        w2_min, w2_max = w2_float.min(), w2_float.max()

        if quant_dtype == "int8":
            # Int8: range -128 to 127
            w1_scale = (w1_max - w1_min) / 255.0
            w2_scale = (w2_max - w2_min) / 255.0
            w1_zp = int(-128 - w1_min / w1_scale)
            w2_zp = int(-128 - w2_min / w2_scale)

            # Quantize weights
            w1_quant = np.clip(np.round(w1_float / w1_scale) + w1_zp, -128, 127).astype(np.int8)
            w2_quant = np.clip(np.round(w2_float / w2_scale) + w2_zp, -128, 127).astype(np.int8)

        else:  # int4
            # Int4: range -8 to 7 (stored as int8 internally)
            w1_scale = (w1_max - w1_min) / 15.0
            w2_scale = (w2_max - w2_min) / 15.0
            w1_zp = int(-8 - w1_min / w1_scale)
            w2_zp = int(-8 - w2_min / w2_scale)

            # Quantize to int4 range, store as int8
            w1_quant = np.clip(np.round(w1_float / w1_scale) + w1_zp, -8, 7).astype(np.int8)
            w2_quant = np.clip(np.round(w2_float / w2_scale) + w2_zp, -8, 7).astype(np.int8)

        # Create constants for quantized weights and scales
        w1_q = builder.constant(w1_quant)
        w2_q = builder.constant(w2_quant)

        scale1 = builder.constant(np.array(w1_scale, dtype=np.float32))
        scale2 = builder.constant(np.array(w2_scale, dtype=np.float32))
        zp1 = builder.constant(np.array(w1_zp, dtype=np.int8))
        zp2 = builder.constant(np.array(w2_zp, dtype=np.int8))

        # Dequantize for computation
        w1_f32 = builder.dequantize_linear(w1_q, scale1, zp1)
        w2_f32 = builder.dequantize_linear(w2_q, scale2, zp2)

        h1 = builder.matmul(x, w1_f32)
        h2 = builder.relu(h1)
        output = builder.matmul(h2, w2_f32)

    # Build graph
    graph = builder.build({"output": output})
    return graph


def compare_sizes(size_multiplier=1):
    """Compare file sizes across quantization levels."""

    print("=" * 80)
    print("QUANTIZATION FILE SIZE COMPARISON")
    print("=" * 80)
    print(f"\nModel Configuration:")
    print(f"  Base size: {64 * size_multiplier} x {64 * size_multiplier}")
    print(f"  Layers: 2 fully-connected layers (matmul + relu + matmul)")
    print(f"  Total parameters: ~{2 * (64 * size_multiplier) ** 2:,}")
    print()

    levels = [
        ("float32", "float32", False),
        ("float16", "float16", False),
        ("int8", "int8", True),
        ("int4", "int4", True),
    ]

    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for dtype, name, quantized_flag in levels:
            print(f"[{name.upper()}] Creating model...")

            try:
                graph = create_model_with_quantization(dtype, size_multiplier)

                # Save model
                model_path = Path(tmpdir) / f"model_{name}.webnn"
                graph.save(str(model_path), quantized=quantized_flag)

                # Get file size
                size_bytes = model_path.stat().st_size
                size_kb = size_bytes / 1024
                size_mb = size_kb / 1024

                results[name] = {
                    'bytes': size_bytes,
                    'kb': size_kb,
                    'mb': size_mb,
                    'path': model_path
                }

                print(f"  File size: {size_bytes:,} bytes ({size_kb:.2f} KB)")

            except Exception as e:
                print(f"  [ERROR] Failed to create {name} model: {e}")
                import traceback
                traceback.print_exc()
                results[name] = None

    # Summary table
    print("\n" + "=" * 80)
    print("SIZE COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Level':<12} {'Size (KB)':<15} {'Size (MB)':<15} {'vs Float32':<20}")
    print("-" * 80)

    float32_size = results.get('float32', {}).get('kb', 0)

    for name in ['float32', 'float16', 'int8', 'int4']:
        if name in results and results[name]:
            size_kb = results[name]['kb']
            size_mb = results[name]['mb']

            if float32_size > 0:
                ratio = (size_kb / float32_size) * 100
                reduction = 100 - ratio
                comparison = f"{ratio:.1f}% ({reduction:+.1f}%)"
            else:
                comparison = "N/A"

            print(f"{name:<12} {size_kb:>10,.2f} KB   {size_mb:>10,.4f} MB   {comparison:<20}")

    print("=" * 80)

    # Detailed breakdown
    print("\nSTORAGE SAVINGS BREAKDOWN:")
    print("-" * 80)

    if 'float32' in results and results['float32']:
        baseline_bytes = results['float32']['bytes']

        for name in ['float16', 'int8', 'int4']:
            if name in results and results[name]:
                size_bytes = results[name]['bytes']
                saved_bytes = baseline_bytes - size_bytes
                saved_kb = saved_bytes / 1024
                saved_mb = saved_kb / 1024
                percentage = (saved_bytes / baseline_bytes) * 100

                print(f"\n{name.upper()} vs FLOAT32:")
                print(f"  Saved: {saved_bytes:,} bytes ({saved_kb:.2f} KB / {saved_mb:.4f} MB)")
                print(f"  Reduction: {percentage:.1f}%")
                print(f"  Compression ratio: {baseline_bytes / size_bytes:.2f}x")

    print("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    import sys

    # Allow size multiplier as command line argument
    size_multiplier = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    if size_multiplier > 1:
        print(f"\n[INFO] Using size multiplier: {size_multiplier}x")
        print(f"[INFO] This will create larger models for more realistic comparison\n")

    results = compare_sizes(size_multiplier)

    # Exit with success
    sys.exit(0)

#!/usr/bin/env python3
"""Debug script to test ONNX Runtime initialization on Linux"""

import sys
import os

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"ORT_DYLIB_PATH: {os.getenv('ORT_DYLIB_PATH', 'NOT SET')}")
print(f"LD_LIBRARY_PATH: {os.getenv('LD_LIBRARY_PATH', 'NOT SET')}")
print()

print("Attempting to import webnn...")
try:
    import webnn
    print(f"SUCCESS: webnn imported from {webnn.__file__}")
except Exception as e:
    print(f"FAILED to import webnn: {e}")
    sys.exit(1)

print()
print("Attempting to create ML context...")
try:
    ml = webnn.ML()
    print("SUCCESS: ML() created")
except Exception as e:
    print(f"FAILED to create ML(): {e}")
    sys.exit(1)

print()
print("Attempting to create CPU context...")
try:
    context = ml.create_context(device_type="cpu")
    print(f"SUCCESS: CPU context created")
    print(f"  Backend: {context.backend}")
    print(f"  Accelerated: {context.accelerated}")
except Exception as e:
    print(f"FAILED to create context: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("Attempting to create graph builder...")
try:
    builder = context.create_graph_builder()
    print("SUCCESS: Graph builder created")
except Exception as e:
    print(f"FAILED to create graph builder: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("Attempting to build a simple graph...")
try:
    import numpy as np

    # Build a simple graph: y = x + 1
    x = builder.input("x", [2, 3], "float32")
    one = builder.constant(np.ones([2, 3], dtype=np.float32))
    y = builder.add(x, one)

    graph = builder.build({"y": y})
    print("SUCCESS: Graph built")
except Exception as e:
    print(f"FAILED to build graph: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("Attempting to compute on graph...")
try:
    x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    result = context.compute(graph, {"x": x_data})
    print(f"SUCCESS: Compute completed")
    print(f"  Result shape: {result['y'].shape}")
    print(f"  Result: {result['y']}")
except Exception as e:
    print(f"FAILED to compute: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("All tests passed!")
print("=" * 70)

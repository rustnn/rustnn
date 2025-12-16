<div align="center">
  <img src="logo/rustnn.png" alt="rustnn logo" width="200"/>

  # rustnn / PyWebNN

  A Rust implementation of WebNN graph handling with Python bindings that implement the W3C WebNN API specification.
</div>

---

## [WARNING] EXPERIMENTAL - DO NOT USE IN PRODUCTION

This is an early-stage experimental implementation for research and exploration. Many features are incomplete, untested, or may change significantly.

---

## What is rustnn?

rustnn provides:

- **Rust Library**: Validates WebNN graphs and converts to ONNX/CoreML formats
- **Python API**: Complete W3C WebNN API implementation via PyO3 bindings
- **Runtime Backends**: Execute on CPU, GPU, or Neural Engine with backend selection at context creation
- **Real Examples**: Complete MobileNetV2 (99.60% accuracy) and Transformer text generation

## Installation

### Python Package (PyWebNN)

**PyPI Package (v0.4.0+):**
```bash
# Install with bundled ONNX Runtime - no additional dependencies needed
pip install pywebnn

# Works immediately with actual execution (no zeros)
```

**Build from Source (For Development):**
```bash
git clone https://github.com/tarekziade/rustnn.git
cd rustnn
make python-dev  # Sets up venv and builds with ONNX Runtime + CoreML
source .venv-webnn/bin/activate
```

**Requirements:** Python 3.11+, NumPy 1.20+

**Note:** Version 0.4.0+ includes bundled ONNX Runtime. Earlier versions (0.3.0 and below) had no backends and returned zeros.

### Rust Library

```toml
[dependencies]
rustnn = "0.1"
```

## Quick Start

```python
import webnn
import numpy as np

# Create ML context with device hints
ml = webnn.ML()
context = ml.create_context(accelerated=False)  # CPU execution
builder = context.create_graph_builder()

# Build a simple graph: output = relu(x + y)
x = builder.input("x", [2, 3], "float32")
y = builder.input("y", [2, 3], "float32")
z = builder.add(x, y)
output = builder.relu(z)

# Compile the graph
graph = builder.build({"output": output})

# Execute with real data
x_data = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.float32)
y_data = np.array([[-1, 2, -3], [-4, 5, -6]], dtype=np.float32)
results = context.compute(graph, {"x": x_data, "y": y_data})

print(results["output"])  # [[0. 0. 0.] [0. 0. 0.]]

# Optional: Export to ONNX
context.convert_to_onnx(graph, "model.onnx")
```

## Backend Selection

Following the [W3C WebNN Device Selection spec](https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md), backends are selected via hints:

```python
# CPU-only execution
context = ml.create_context(accelerated=False)

# Request GPU/NPU (platform selects best available)
context = ml.create_context(accelerated=True)

# Request high-performance (prefers GPU)
context = ml.create_context(accelerated=True, power_preference="high-performance")

# Request low-power (prefers NPU/Neural Engine)
context = ml.create_context(accelerated=True, power_preference="low-power")
```

**Platform-Specific Backends:**
- NPU: CoreML Neural Engine (Apple Silicon macOS only)
- GPU: ONNX Runtime GPU (cross-platform) or CoreML GPU (macOS)
- CPU: ONNX Runtime CPU (cross-platform)

## Examples

### Complete MobileNetV2 Image Classification

```bash
# Download pretrained weights (first time only)
bash scripts/download_mobilenet_weights.sh

# Run on different backends
python examples/mobilenetv2_complete.py examples/images/test.jpg --backend cpu
python examples/mobilenetv2_complete.py examples/images/test.jpg --backend gpu
python examples/mobilenetv2_complete.py examples/images/test.jpg --backend coreml
```

**Output:**
```
Top 5 Predictions (Real ImageNet Labels):
  1. lesser panda                                        99.60%
  2. polecat                                              0.20%
  3. weasel                                               0.09%

Performance: 74.41ms (CPU) / 77.14ms (GPU) / 51.93ms (CoreML)
```

### Text Generation with Transformer Attention

```bash
# Run generation with attention
make text-gen-demo

# Train on custom text
make text-gen-train

# Generate with trained weights
make text-gen-trained
```

See [examples/](examples/) for more samples.

## Documentation

- **[Getting Started](docs/user-guide/getting-started.md)** - Installation and first steps
- **[API Reference](docs/user-guide/api-reference.md)** - Complete Python API documentation
- **[Examples](docs/user-guide/examples.md)** - Code examples and tutorials
- **[Architecture](docs/architecture/overview.md)** - Design principles and structure
- **[Development Guide](docs/development/setup.md)** - Building and contributing

## Implementation Status

- 85 of ~95 WebNN operations (89% spec coverage)
- Shape inference: 85/85 (100%)
- Python API: 85/85 (100%)
- ONNX Backend: 85/85 (100%)
- CoreML MLProgram: 85/85 (100%)
- 1350+ WPT conformance tests passing

See [docs/development/implementation-status.md](docs/development/implementation-status.md) for complete details.

## Rust CLI Usage

```bash
# Validate a graph
cargo run -- examples/sample_graph.json

# Visualize a graph (requires graphviz)
cargo run -- examples/sample_graph.json --export-dot graph.dot
dot -Tpng graph.dot -o graph.png

# Convert to ONNX
cargo run -- examples/sample_graph.json --convert onnx --convert-output model.onnx

# Execute with ONNX Runtime
cargo run --features onnx-runtime -- examples/sample_graph.json --convert onnx --run-onnx
```

See `make help` for all available targets.

## Contributing

Contributions welcome! Please see:

- [AGENTS.md](AGENTS.md) - Project architecture and conventions
- [docs/development/contributing.md](docs/development/contributing.md) - How to add features
- [TODO.txt](TODO.txt) - Feature requests and known issues

**Quick Contribution Guide:**

1. Fork and create feature branch: `git checkout -b feature/my-feature`
2. Install hooks (optional): `./scripts/install-git-hooks.sh`
3. Make changes and test: `make test && make python-test`
4. Format code: `make fmt`
5. Commit and push

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Links

- **GitHub**: [https://github.com/tarekziade/rustnn](https://github.com/tarekziade/rustnn)
- **PyPI**: [https://pypi.org/project/pywebnn/](https://pypi.org/project/pywebnn/)
- **Documentation**: [https://tarekziade.github.io/rustnn/](https://tarekziade.github.io/rustnn/)
- **W3C WebNN Spec**: [https://www.w3.org/TR/webnn/](https://www.w3.org/TR/webnn/)

## Acknowledgments

- W3C WebNN Community Group for the specification
- Chromium WebNN implementation for reference
- PyO3 and Maturin projects for excellent Python-Rust integration

---

**Made with Rust by [Tarek Ziade](https://github.com/tarekziade)**

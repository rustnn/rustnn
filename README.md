<div align="center">
  <img src="logo/rustnn.png" alt="rustnn logo" width="200"/>

  # rustnn / PyWebNN

  A Rust implementation of WebNN graph handling with Python bindings that implement the W3C WebNN API specification.
</div>

---

## ⚠️ **EXPERIMENTAL - DO NOT USE IN PRODUCTION**

**This project is a proof-of-concept and experimental implementation. It is NOT ready for production use.**

This is an early-stage experiment to explore WebNN graph handling and format conversion. Many features are incomplete, untested, or may change significantly. Use at your own risk for research and experimentation only.

---

**Features:**
- 🦀 **Rust Library**: Validates WebNN graphs and converts to ONNX/CoreML formats
- 🐍 **Python API**: Complete W3C WebNN API implementation via PyO3 bindings
- 🎯 **Runtime Backend Selection**: Choose CPU, GPU, or NPU execution at context creation
- 📊 **Format Conversion**: Export graphs to ONNX (cross-platform) and CoreML (macOS)
- 🚀 **Model Execution**: Run converted models on CPU, GPU, and Neural Engine (macOS)
- ⚡ **Async Support**: Non-blocking execution with Python asyncio integration
- 🔍 **Graph Visualization**: Generate Graphviz diagrams of your neural networks
- ✅ **Validation**: Comprehensive graph validation matching Chromium's WebNN implementation
- 📐 **Shape Inference**: Automatic shape computation with NumPy-style broadcasting

---

## 📦 Installation

### Python Package (PyWebNN)

Install from PyPI:

```bash
pip install pywebnn
```

Or install from source with maturin:

```bash
# Clone the repository
git clone https://github.com/tarekziade/rustnn.git
cd rustnn

# Install in development mode
pip install maturin
maturin develop --features python

# With optional runtime features
maturin develop --features python,onnx-runtime,coreml-runtime
```

**Requirements:** Python 3.11+, NumPy 1.20+

### Rust Library

Add to your `Cargo.toml`:

```toml
[dependencies]
rustnn = "0.1"
```

Or use directly from this repository.

---

## 🚀 Quick Start

### Python API

```python
import webnn
import numpy as np

# Create ML context - use hints for device selection
ml = webnn.ML()
context = ml.create_context(accelerated=False)  # CPU-only execution
# Or: context = ml.create_context(accelerated=True)  # Request GPU/NPU if available

# Create graph builder
builder = context.create_graph_builder()

# Define a simple graph: z = relu(x + y)
x = builder.input("x", [2, 3], "float32")
y = builder.input("y", [2, 3], "float32")
z = builder.add(x, y)
output = builder.relu(z)

# Compile the graph (creates backend-agnostic representation)
graph = builder.build({"output": output})

# Prepare input data
x_data = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.float32)
y_data = np.array([[-1, 2, -3], [-4, 5, -6]], dtype=np.float32)

# Execute: converts to backend-specific format and runs
results = context.compute(graph, {"x": x_data, "y": y_data})
print(results["output"])  # Actual computed values from ONNX Runtime

# Optional: Export the ONNX model to file (for deployment, inspection, etc.)
context.convert_to_onnx(graph, "model.onnx")
```

### Backend Selection

Following the [W3C WebNN Device Selection spec](https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md), device selection uses **hints** rather than explicit device types:

```python
# Request GPU/NPU acceleration (default)
context = ml.create_context(accelerated=True, power_preference="default")
print(f"Accelerated: {context.accelerated}")  # Check if acceleration is available

# Request low-power execution (prefers NPU over GPU)
context = ml.create_context(accelerated=True, power_preference="low-power")

# Request high-performance execution (prefers GPU)
context = ml.create_context(accelerated=True, power_preference="high-performance")

# CPU-only execution (no acceleration)
context = ml.create_context(accelerated=False)
```

**Device Selection Logic:**
- `accelerated=True` + `power_preference="low-power"` → **NPU** > GPU > CPU
- `accelerated=True` + `power_preference="high-performance"` → **GPU** > NPU > CPU
- `accelerated=True` + `power_preference="default"` → **GPU** > NPU > CPU
- `accelerated=False` → **CPU only**

**Platform-Specific Backends:**
- **NPU**: CoreML Neural Engine (Apple Silicon macOS only)
- **GPU**: ONNX Runtime GPU (cross-platform) or CoreML GPU (macOS)
- **CPU**: ONNX Runtime CPU (cross-platform)

**Important:** The `accelerated` property indicates **platform capability**, not a guarantee. Query `context.accelerated` after creation to check if GPU/NPU resources are available. The platform controls actual device allocation based on runtime conditions.

The graph compilation (`builder.build()`) creates a **backend-agnostic representation**. Backend-specific conversion happens automatically during `compute()` based on the context's selected backend.

### Async Execution

WebNN supports asynchronous execution following the W3C specification. Use `AsyncMLContext` for non-blocking operations:

```python
import asyncio
import numpy as np
import webnn

async def main():
    # Create context
    ml = webnn.ML()
    context = ml.create_context(accelerated=False)
    async_context = webnn.AsyncMLContext(context)

    # Build graph
    builder = async_context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    y = builder.input("y", [2, 3], "float32")
    z = builder.add(x, y)
    output = builder.relu(z)
    graph = builder.build({"output": output})

    # Async dispatch (non-blocking execution)
    x_data = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.float32)
    y_data = np.array([[-1, 2, -3], [-4, 5, -6]], dtype=np.float32)
    await async_context.dispatch(graph, {"x": x_data, "y": y_data})

    print("Graph executed asynchronously!")

asyncio.run(main())
```

### Rust Library

```rust
use rustnn::{GraphInfo, GraphValidator, ContextProperties};
use rustnn::converters::{ConverterRegistry, OnnxConverter};

// Load graph from JSON
let graph_info: GraphInfo = serde_json::from_str(&json_data)?;

// Validate the graph
let validator = GraphValidator::new(&graph_info, ContextProperties::default());
let artifacts = validator.validate()?;

// Convert to ONNX
let mut registry = ConverterRegistry::new();
registry.register(Box::new(OnnxConverter));
let converted = registry.convert("onnx", &graph_info)?;

// Save to file
std::fs::write("model.onnx", &converted.data)?;

// Execute with ONNX Runtime (requires "onnx-runtime" feature)
#[cfg(feature = "onnx-runtime")]
{
    use rustnn::executors::onnx::run_onnx_zeroed;

    // Execute model with zeroed inputs
    run_onnx_zeroed(&converted.data)?;
    println!("Model executed successfully with ONNX Runtime");
}

// Execute with CoreML (requires "coreml-runtime" feature, macOS only)
#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
{
    use rustnn::executors::coreml::run_coreml_zeroed_cached;
    use rustnn::converters::CoremlMlProgramConverter;

    // Convert to CoreML MLProgram
    registry.register(Box::new(CoremlMlProgramConverter::default()));
    let coreml = registry.convert("coreml", &graph_info)?;

    // Execute on GPU (0=CPU, 1=GPU, 2=Neural Engine)
    run_coreml_zeroed_cached(&coreml.data, 1)?;
    println!("Model executed successfully with CoreML");
}
```

---

## 📚 Python API Reference
## 📚 Python API Reference

The Python API implements the [W3C WebNN specification](https://www.w3.org/TR/webnn/).

For complete API documentation, see:
- **[API Reference](docs/api-reference.md)** - Full class and method documentation
- **[Getting Started Guide](docs/getting-started.md)** - Quick start tutorial
- **[Examples](examples/)** - Working code examples

---

## 🦀 Rust CLI Usage

The Rust library includes a powerful CLI tool for working with WebNN graphs.

### Validate a Graph

```bash
cargo run -- examples/sample_graph.json
```

### Visualize a Graph

```bash
# Generate DOT file
cargo run -- examples/sample_graph.json --export-dot graph.dot

# Convert to PNG (requires graphviz)
dot -Tpng graph.dot -o graph.png

# Or use the Makefile shortcut (macOS)
make viz
```

### Convert to ONNX

```bash
cargo run -- examples/sample_graph.json \
    --convert onnx \
    --convert-output model.onnx
```

### Convert to CoreML

```bash
cargo run -- examples/sample_graph.json \
    --convert coreml \
    --convert-output model.mlmodel
```

### Execute Models

**ONNX Runtime** (cross-platform):

```bash
cargo run --features onnx-runtime -- \
    examples/sample_graph.json \
    --convert onnx \
    --run-onnx
```

**CoreML Runtime** (macOS only):

```bash
cargo run --features coreml-runtime -- \
    examples/sample_graph.json \
    --convert coreml \
    --run-coreml \
    --device gpu  # or 'cpu', 'ane' for Neural Engine
```

### Makefile Targets

```bash
make help              # Show all available targets
make build             # Build Rust project
make test              # Run Rust tests
make python-dev        # Install Python package in dev mode
make python-test       # Run Python tests
make docs-serve        # Serve documentation locally
make validate-all-env  # Run full test pipeline
```

---

## 🏗️ Architecture
## 🏗️ Architecture

For detailed architecture documentation, see **[Architecture Guide](docs/architecture.md)**.

Key principles:
- **Backend-Agnostic Graph Representation**: Platform-independent `GraphInfo` structure
- **Runtime Backend Selection**: WebNN spec-compliant device selection via hints
- **MLTensor Management**: Explicit tensor control with descriptor flags
- **Rust-First Design**: Pure Rust core with thin Python bindings

---

## 🔧 Development

For detailed development documentation, see **[Development Guide](docs/development.md)**.

Quick start:
```bash
# Clone and build
git clone https://github.com/tarekziade/rustnn.git
cd rustnn
cargo build --release
maturin develop --features python

# Run tests
cargo test
python -m pytest tests/
```

---


## 🧪 Testing

### Python Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_python_api.py -v

# Run integration tests with cleanup
python tests/test_integration.py --cleanup
```

### Rust Tests

```bash
# All tests
cargo test

# Specific module
cargo test converters

# With features
cargo test --features onnx-runtime,coreml-runtime
```

---

## 📋 Roadmap

See [TODO.txt](TODO.txt) for a comprehensive list of planned features.

**Completed:**
- ✅ Python WebNN API implementation
- ✅ Runtime backend selection (WebNN spec-compliant)
- ✅ ONNX conversion with full operation support
- ✅ Actual tensor execution with ONNX Runtime
- ✅ Async execution support (AsyncMLContext)
- ✅ Shape inference and broadcasting
- ✅ Comprehensive documentation

**High Priority:**
- ⬜ PyPI package publishing automation
- ⬜ More operations (conv2d, pooling, normalization)
- ⬜ CoreML execution with actual tensor I/O

**Medium Priority:**
- ⬜ Graph optimization passes
- ⬜ Multi-platform wheel building (manylinux, Windows)
- ⬜ Performance benchmarks

---

## 🤝 Contributing

Contributions are welcome! Please see:

- [CLAUDE.md](CLAUDE.md) - Project architecture and conventions
- [TODO.txt](TODO.txt) - Feature requests and known limitations

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. **Install git hooks** (optional but recommended):
   ```bash
   ./scripts/install-git-hooks.sh
   ```
   This installs a pre-commit hook that automatically checks code formatting before each commit.
4. Make your changes
5. Run tests: `cargo test && pytest tests/`
6. Format code: `cargo fmt` (or let the pre-commit hook handle it)
7. Commit: `git commit -m "Add my feature"`
8. Push and create a pull request

**Note:** The pre-commit hook will prevent commits with unformatted code. If needed, you can bypass it with `git commit --no-verify`, but this is not recommended.

---

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## 🔗 Links

- **GitHub**: [https://github.com/tarekziade/rustnn](https://github.com/tarekziade/rustnn)
- **PyPI**: [https://pypi.org/project/pywebnn/](https://pypi.org/project/pywebnn/)
- **Documentation**: [https://tarekziade.github.io/rustnn/](https://tarekziade.github.io/rustnn/)
- **W3C WebNN Spec**: [https://www.w3.org/TR/webnn/](https://www.w3.org/TR/webnn/)
- **Issues**: [https://github.com/tarekziade/rustnn/issues](https://github.com/tarekziade/rustnn/issues)

---

## 🙏 Acknowledgments

- W3C WebNN Community Group for the specification
- Chromium WebNN implementation for reference
- PyO3 project for excellent Python-Rust bindings
- Maturin for seamless Python package building

---

**Made with ❤️ by [Tarek Ziade](https://github.com/tarekziade)**

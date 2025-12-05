# rust-webnn-graph (rustnn) - Project Guide

## Project Overview

**rustnn** is a standalone Rust crate that mirrors Chromium's WebNN (Web Neural Network) graph handling while adding pluggable format converters and helper tooling to visualize, execute, and validate exported graphs on macOS.

**Core Capabilities:**
- Validates WebNN graph descriptions from JSON files
- Converts WebNN graphs to ONNX and CoreML formats
- Executes converted models on various compute units (CPU, GPU, Neural Engine)
- Visualizes graph structures using Graphviz DOT format
- Provides both a CLI tool and library API
- **Python bindings** via PyO3/maturin implementing the W3C WebNN API specification

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│ CLI (main.rs) / Library API (lib.rs)                        │
└──────────────┬──────────────────────────────────────────────┘
               │
    ┌──────────┴──────────┬──────────────┬─────────────────┐
    ▼                     ▼              ▼                 ▼
┌────────┐     ┌──────────────┐   ┌──────────┐    ┌──────────────┐
│Loader  │────▶│  Validator   │──▶│Converter │───▶│  Executor    │
│(JSON)  │     │(graph.rs)    │   │(Registry)│    │(ONNX/CoreML) │
└────────┘     └──────────────┘   └────┬─────┘    └──────────────┘
                                        │
                              ┌─────────┴─────────┐
                              ▼                   ▼
                        ┌──────────┐       ┌──────────┐
                        │  ONNX    │       │ CoreML   │
                        │Converter │       │Converter │
                        └──────────┘       └──────────┘
```

### Key Modules

#### **graph.rs** - Core Data Model
- `DataType`: Float32, Float16, Int32, Uint32, Int8, Uint8
- `OperandDescriptor`: Shape and type information
- `OperandKind`: Input, Constant, Output
- `Operand`: Graph nodes with descriptors and metadata
- `Operation`: Graph operations with inputs/outputs
- `ConstantData`: Weight/constant storage (base64 encoded)
- `GraphInfo`: Complete graph representation

**Key Convention:** Operands are referenced by their array index (u32) within the graph's operands list.

#### **validator.rs** - Validation Pipeline
- `ContextProperties`: Validation constraints and limits
- `GraphValidator`: Validates graph structure and dependencies
- `ValidationArtifacts`: Results including I/O descriptors and operation dependencies

**Validation Checks:**
1. Operand count limits
2. Tensor byte length limits
3. Valid input/output names
4. Constant data integrity
5. Operation dependency ordering
6. Operand usage consistency

#### **converters/** - Pluggable Format Conversion
- **Registry Pattern**: `ConverterRegistry` manages converters dynamically
- **Trait Interface**: `GraphConverter` defines conversion contract
- **Implementations**:
  - `OnnxConverter` → ONNX protobuf format
  - `CoremlConverter` → CoreML protobuf format

#### **executors/** - Runtime Execution
- **Platform-specific**: Conditional compilation for macOS
- **ONNX Runtime**: `run_onnx_zeroed()` - cross-platform
- **CoreML Runtime**: `run_coreml_zeroed_cached()` - macOS only via Objective-C FFI

#### **graphviz.rs** - Visualization
- Generates DOT format for graph visualization
- Color-coded nodes: inputs (green), outputs (blue), constants (yellow)

## Development Conventions

### Code Style

1. **Naming:**
   - Files: `snake_case.rs`
   - Types: `PascalCase`
   - Functions: `snake_case`
   - Enums: PascalCase variants, snake_case JSON serialization

2. **Error Handling:**
   - All fallible operations return `Result<T, GraphError>`
   - Use `?` operator for error propagation
   - `thiserror` for error type derivation
   - Include contextual information in errors

3. **Serde Integration:**
   - `#[derive(Serialize, Deserialize)]` on all data types
   - `#[serde(rename_all = "snake_case")]` for JSON compatibility
   - `serde_with` for base64 encoding of binary data
   - Optional fields use `Option<T>`

4. **Testing:**
   - Unit tests in `#[cfg(test)]` modules at end of files
   - Use realistic data structures matching actual usage
   - Test examples exist in `graphviz.rs` and `converters/mod.rs`

### Architecture Patterns

1. **Registry Pattern** (converters):
   - Trait objects: `Box<dyn GraphConverter + Send + Sync>`
   - Dynamic registration and lookup
   - Extensible without modifying core code

2. **Builder Pattern** (protobuf construction):
   - Incremental construction of complex structures
   - Used in ONNX and CoreML converters

3. **Validation Pipeline**:
   - Immutable graph input
   - Stateful validator with progressive checks
   - Comprehensive artifacts returned for downstream use

4. **Conditional Compilation**:
   - `#[cfg(target_os = "macos")]` for platform-specific code
   - `#[cfg(feature = "...")]` for optional features
   - Graceful degradation on unsupported platforms

5. **Explicit Dependencies**:
   - No singletons or global state
   - Pass dependencies via function parameters
   - Clear data flow through the system

### File Organization

```
src/
├── lib.rs              # Public API exports
├── main.rs             # CLI entry point
├── graph.rs            # Core data structures
├── error.rs            # Error types
├── validator.rs        # Graph validation
├── loader.rs           # JSON loading
├── graphviz.rs         # DOT export
├── protos.rs           # Protobuf module setup
├── converters/
│   ├── mod.rs          # Registry and trait
│   ├── onnx.rs         # ONNX converter
│   └── coreml.rs       # CoreML converter
├── executors/
│   ├── mod.rs          # Conditional compilation
│   ├── onnx.rs         # ONNX runtime
│   └── coreml.rs       # CoreML runtime
└── python/             # Python bindings (PyO3)
    ├── mod.rs          # Python module definition
    ├── context.rs      # ML and MLContext classes
    ├── graph_builder.rs # MLGraphBuilder class
    ├── graph.rs        # MLGraph class
    └── operand.rs      # MLOperand class

python/webnn/           # Python package
├── __init__.py         # Package exports
└── __init__.pyi        # Type stubs

tests/
└── test_python_api.py  # Python API tests

examples/
├── python_simple.py    # Basic Python example
└── python_matmul.py    # Matrix multiplication example
```

## Adding New Features

### Adding a New Converter

1. **Create converter file** in `src/converters/your_format.rs`
2. **Implement the trait:**
   ```rust
   pub struct YourFormatConverter;

   impl GraphConverter for YourFormatConverter {
       fn name(&self) -> &str { "your-format" }
       fn convert(&self, graph_info: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
           // Implementation
       }
   }
   ```
3. **Register in** `converters/mod.rs` or `main.rs`:
   ```rust
   registry.register(Box::new(YourFormatConverter));
   ```
4. **Add dependencies** to `Cargo.toml` if needed
5. **Add tests** in your converter file

### Adding a New Executor

1. **Create executor file** in `src/executors/your_runtime.rs`
2. **Add feature gate** in `Cargo.toml`:
   ```toml
   [features]
   your-runtime = ["dep:your-runtime-crate"]
   ```
3. **Implement execution function:**
   ```rust
   #[cfg(feature = "your-runtime")]
   pub fn run_your_runtime(model_data: &[u8]) -> Result<(), GraphError> {
       // Implementation
   }
   ```
4. **Add conditional compilation** in `executors/mod.rs`
5. **Wire up in CLI** (`main.rs`) if needed

### Adding New Graph Operations

Currently, operations are validated but not typed. To add operation-specific validation:

1. **Extend** `Operation` struct in `graph.rs` if needed
2. **Add validation logic** in `validator.rs`
3. **Update converters** to handle the new operation
4. **Add test cases** with example graphs

### Adding Protobuf Definitions

1. **Add .proto files** to `protos/your_format/`
2. **Update** `build.rs` to compile them:
   ```rust
   prost_build::compile_protos(&["protos/your_format/schema.proto"], &["protos/"])?;
   ```
3. **Include generated code** in `src/protos.rs`:
   ```rust
   pub mod your_format {
       include!(concat!(env!("OUT_DIR"), "/your.format.namespace.rs"));
   }
   ```

## Common Tasks

### Building the Project
```bash
make build                    # Debug build
cargo build --release         # Release build
cargo build --features coreml-runtime,onnx-runtime  # All features
```

### Running Validation
```bash
cargo run -- validate examples/sample_graph.json
```

### Converting Graphs
```bash
# To ONNX
cargo run -- convert examples/sample_graph.json onnx -o output.onnx

# To CoreML
cargo run -- convert examples/sample_graph.json coreml -o output.mlmodel
```

### Executing Models
```bash
# ONNX (requires onnx-runtime feature)
cargo run --features onnx-runtime -- run-onnx model.onnx

# CoreML (macOS only, requires coreml-runtime feature)
cargo run --features coreml-runtime -- run-coreml model.mlmodel --device gpu
```

### Visualization
```bash
cargo run -- visualize examples/sample_graph.json -o graph.dot
dot -Tpng graph.dot -o graph.png
```

### Running Tests
```bash
cargo test                    # All tests
cargo test --lib              # Library tests only
cargo test converters         # Specific module
```

### Clean Build Artifacts
```bash
make clean
```

## Dependencies

### Core Dependencies
- **clap 4.5** - CLI argument parsing
- **serde 1.0** + **serde_json 1.0** - JSON serialization
- **serde_with 3.8** - Base64 encoding
- **thiserror 1.0** - Error derivation
- **prost 0.12** + **prost-types 0.12** - Protobuf runtime
- **bytes 1.6** - Byte buffer utilities
- **bytemuck 1.15** - Type casting

### Optional Runtime Dependencies
- **objc 0.2** - Objective-C FFI for CoreML (macOS)
- **onnxruntime 0.0.14** - ONNX execution
- **pyo3 0.22** - Python bindings (optional, with `python` feature)

### Build Dependencies
- **prost-build 0.12** - Protobuf code generation
- **maturin** - Python package build system (for Python bindings)

## Platform Support

- **Validation & Conversion**: Cross-platform (Linux, macOS, Windows)
- **ONNX Execution**: Cross-platform with `onnx-runtime` feature
- **CoreML Execution**: macOS only with `coreml-runtime` feature
- **Neural Engine**: macOS with Apple Silicon (via CoreML)
- **Python Bindings**: Cross-platform with `python` feature (Python 3.8+)

## Key Technical Decisions

1. **Protobuf for interop**: Native format for ONNX and CoreML
2. **Compile-time codegen**: Protobufs compiled at build time
3. **Feature flags**: Optional runtimes to minimize dependencies
4. **Objective-C FFI**: Direct CoreML access on macOS
5. **Zero-copy where possible**: `Bytes` type for efficiency
6. **Registry pattern**: Pluggable converters without core changes

## Future Extension Points

- **More converters**: TensorFlow Lite, TensorRT, OpenVINO
- **More executors**: Additional backend runtimes
- **Operation typing**: Strongly-typed operation variants
- **Graph optimization**: Pre-conversion graph transformations
- **Benchmarking**: Performance measurement tools
- **Graph diff**: Compare graphs for equivalence

## Python Integration

### Python Bindings (WebNN API)

The project includes full Python bindings implementing the W3C WebNN API specification.

**Installation:**
```bash
# Development mode
maturin develop --features python

# Or build a wheel
maturin build --release --features python
pip install target/wheels/webnn-*.whl
```

**Quick Example:**
```python
import webnn
import numpy as np

# Create context and builder
ml = webnn.ML()
context = ml.create_context(device_type="cpu")
builder = context.create_graph_builder()

# Build graph: output = relu(x + y)
x = builder.input("x", [2, 3], "float32")
y = builder.input("y", [2, 3], "float32")
z = builder.add(x, y)
output = builder.relu(z)

# Compile and execute
graph = builder.build({"output": output})
context.convert_to_onnx(graph, "model.onnx")
```

**API Classes:**
- `webnn.ML` - Entry point
- `webnn.MLContext` - Execution context
- `webnn.MLGraphBuilder` - Graph construction
- `webnn.MLGraph` - Compiled graph
- `webnn.MLOperand` - Tensor operands

See **README_PYTHON.md** for complete documentation and examples.

### ONNX to WebNN Converter Script

The `scripts/convert_onnx_to_webnn.py` script converts ONNX models to WebNN JSON format:
- Uses `huningxin/onnx2webnn` package
- Includes preprocessing and optimization
- Handles operator normalization

## Claude Code - Approved Permissions

The following operations have been approved for Claude Code to execute without requiring additional user confirmation:

### Build & Development
- `cargo check` - Run Rust type checking
- `cargo build` - Build the Rust project
- `cargo fmt` - Format Rust code according to style guidelines
- `pip install` - Install Python packages
- `maturin develop` - Install Python package in development mode
- `make help` - Display Makefile help
- `make ci-docs` - Build documentation in strict mode

### Python Execution
- `python` - Run Python scripts
- `.venv/bin/python` - Run Python from virtual environment

### Documentation
- `mkdocs build` - Build documentation site

### File Operations
- `find` - Search for files
- `cat` - Read file contents

### Git Operations
- `git add` - Stage files for commit
- `git commit` - Create commits
- `git push` - Push commits to remote
- `git tag` - Create and manage version tags
- `git show` - Display commit information

### GitHub Operations
- `gh run list` - List GitHub Actions workflow runs
- `gh run view` - View details of GitHub Actions runs

### Web Resources
- `WebFetch(domain:www.w3.org)` - Fetch W3C WebNN specifications

These permissions enable Claude Code to efficiently assist with development, testing, documentation, version control, and CI/CD monitoring tasks without interrupting the workflow.

## Resources

- **README.md**: User-facing documentation and usage examples
- **README_PYTHON.md**: Complete Python API documentation
- **examples/**: Sample WebNN graph JSON files and Python examples
- **tests/test_python_api.py**: Python API test suite
- **Makefile**: Common build and validation targets
- **pyproject.toml**: Python package configuration
- **LICENSE**: Apache 2.0 license

---

*This CLAUDE.md evolves with the project. Update it as new patterns emerge or architecture changes.*

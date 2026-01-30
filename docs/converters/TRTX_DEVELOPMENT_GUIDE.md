# TensorRT Converter Development Guide

**Last Updated:** 2026-01-29  
**Status:** 97 WebNN operations implemented (92% coverage), 96 tests passing

---

## Overview

This guide documents the complete implementation of the TensorRT converter for WebNN operations, including all architectural decisions, fixes, and testing strategies used during development.

---

## Quick Start

### Building
```bash
cd c:\git\rustnn-workspace\rustnn
cargo build --release --features trtx-runtime
```

### Testing
```bash
# Run all tests
cargo test --release --test test_trtx_execution --features trtx-runtime

# Run specific test
cargo test --release --test test_trtx_execution --features trtx-runtime test_gather_nd

# Run with verbose output
cargo test --release --test test_trtx_execution --features trtx-runtime -- --nocapture
```

### Troubleshooting Build Issues
```powershell
# If build fails, try aggressive clean
Remove-Item -Recurse -Force target
cargo clean
cargo build --release --features trtx-runtime
```

---

## Recent Implementation (2026-01-29)

### 7 New Operations Implemented

1. **gatherND** - N-dimensional gather using `IGatherLayer` with `GatherMode::kND`
2. **convTranspose2d** - Transposed convolution using `IDeconvolutionLayer`
3. **resample2d** - 2D image resizing using `IResizeLayer` with `InterpolationMode`
4. **scatterElements** - Element-wise scatter using `IScatterLayer` with `ScatterMode::kELEMENT`
5. **scatterND** - N-dimensional scatter using `IScatterLayer` with `ScatterMode::kND`
6. **quantizeLinear** - Float32 → Int8 quantization using `IQuantizeLayer`
7. **dequantizeLinear** - Int8 → Float32 dequantization using `IDequantizeLayer`

### Test Results
- **96 tests passing** (all operations including new ones)
- **0 tests ignored** (QDQ test now uses proper pattern)
- **100% pass rate**

---

## Key Architectural Decisions

### 1. TensorRT Enum Integration

**Problem:** Code used magic constants (e.g., `0`, `1`, `2`) instead of named enum values.

**Solution:** Generated proper Rust enums for all TensorRT C++ enums:
- `ElementWiseOperation`
- `UnaryOperation`
- `ActivationType`
- `ReduceOperation`
- `PoolingType`
- `GatherMode`
- `ScatterMode`
- `MatrixOperation`
- `InterpolationMode` (with `ResizeMode` typedef)
- `ResizeCoordinateTransformation`
- `ResizeSelector`
- `ResizeRoundMode`

**Implementation:**
```rust
// trtx-sys/src/lib.rs
generate!("nvinfer1::ElementWiseOperation");
generate!("nvinfer1::GatherMode");
// etc.

// trtx/src/enum_helpers.rs
pub fn element_wise_operation_name(op: &ElementWiseOperation) -> &'static str {
    match *op {
        ElementWiseOperation::kSUM => "SUM",
        ElementWiseOperation::kPROD => "PROD",
        // ...
    }
}
```

### 2. Typedef Pattern for TensorRT API Compatibility

**Problem:** TensorRT uses C++ typedefs (e.g., `ResizeMode` is typedef for `InterpolationMode`), but `autocxx` can't generate typedefs directly.

**Solution:** Generate the base type and create Rust `type` alias:

```rust
// trtx-sys/src/lib.rs
generate!("nvinfer1::InterpolationMode");  // Generate base type
pub type ResizeMode = InterpolationMode;    // Rust type alias

// trtx/src/lib.rs
pub use trtx_sys::nvinfer1::InterpolationMode;
pub use trtx_sys::ResizeMode;
```

This pattern was also used for `Dims64` (typedef for `Dims`).

### 3. QDQ Testing Pattern

**Problem:** Isolated quantize/dequantize tests failed because TensorRT requires proper QDQ graph patterns.

**Solution:** Created comprehensive roundtrip test:

```rust
#[test]
fn test_quantize_dequantize_roundtrip() {
    // Input → Quantize → Int8 → Dequantize → Float → Add(+1.0) → Output
    // Tests both operations in realistic context
}
```

**Benefits:**
- ✅ Satisfies TensorRT's QDQ graph optimizer
- ✅ Tests both quantize and dequantize operations
- ✅ Mirrors real-world quantized model patterns
- ✅ Includes meaningful computation (add operation)

### 4. Multi-Output Helper Support

**Problem:** QDQ test created intermediate outputs (quantized, dequantized values) that TensorRT exposed as network outputs.

**Solution:** Enhanced `execute_graph_multi_input()` to handle multiple TensorRT outputs but return only the final output:

```rust
// Allocate buffers for ALL TensorRT outputs (including intermediates)
let num_trt_outputs = num_tensors as usize - num_inputs;
for i in 0..num_trt_outputs {
    output_buffers.push(DeviceBuffer::new(output_size)?);
}

// Return only the final output (last buffer)
final_output_buffer.copy_to_host(output_bytes)?;
```

### 5. Weights Lifetime Management

**Critical Requirement:** Constant weights passed to TensorRT must remain valid until engine serialization completes.

**Implementation:**
```rust
pub struct TrtxConverter {
    temp_weights: Vec<Vec<u8>>,  // Keeps weights alive
}

fn add_constant(&mut self, data: &[u8]) -> trtx::Weights {
    let weight_data = data.to_vec();
    self.temp_weights.push(weight_data);
    let stable_ptr = self.temp_weights.last().unwrap().as_ptr();
    trtx::Weights::new(stable_ptr, data.len(), data_type)
}
```

---

## TensorRT API Corrections

### API Signature Fixes (5 fixes)

1. **`IGatherLayer::setGatherMode`**
   - **Wrong:** `setGatherMode(mode)`
   - **Correct:** `setMode(mode)`

2. **`addDeconvolutionNd`**
   - **Wrong:** `addDeconvolutionNd(input, kernel, &kernel_dims)`
   - **Correct:** `addDeconvolutionNd(input, kernel, kernel_dims)` (pass by value)

3. **`addScatter`**
   - **Wrong:** Takes `axis: i32`
   - **Correct:** Takes `mode: ScatterMode`, then call `layer.set_axis(axis)`

4. **`addQuantize`**
   - **Wrong:** Takes 2 arguments
   - **Correct:** Takes 3 arguments: `(input, scale, output_type: DataType)`

5. **`addDequantize`**
   - **Wrong:** Takes 2 arguments
   - **Correct:** Takes 3 arguments: `(input, scale, output_type: DataType)`

### Type Conversion Requirements

**Gather/Scatter Operations:**
- Indices MUST be `Int32` or `Int64`, NOT `Float32`
- Tests updated to use `DataType::Int32` for index tensors
- Helper function converts `f32` → `i32` automatically

**Boolean Operations:**
- Comparison operations output `DataType::Bool`
- Logical operations require `DataType::Bool` inputs
- Use `add_cast()` to convert between types

---

## Test Coverage

### Test Statistics
- **Total Tests:** 96 passing
- **Binary Operations:** 7/7 ✅
- **Unary Operations:** 35/35 ✅
- **Matrix Operations:** 2/2 ✅
- **Convolution:** 2/2 ✅
- **Pooling:** 4/5 ✅
- **Normalization:** 3/3 ✅
- **Reduction:** 10/10 ✅
- **Shape Manipulation:** 8/9 ✅
- **Indexing/Gathering:** 4/7 ✅
- **Comparison:** 6/6 ✅
- **Logical:** 4/4 ✅
- **Quantization:** 1 (roundtrip test) ✅
- **Other:** 7/14 ✅

### Test Helpers

**Single Input:**
```rust
execute_graph(&graph, &input_data)
```

**Multiple Inputs (with automatic type conversion):**
```rust
execute_graph_multi_input(&graph, vec![data, indices, updates])
// Automatically converts f32 → i32 based on operand descriptors
```

**Verification:**
```rust
verify_output(&actual, &expected, tolerance);
```

### Test Patterns

**Basic Operation Test:**
```rust
#[test]
fn test_operation() {
    let graph = GraphInfo { /* graph definition */ };
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let expected = vec![2.0, 3.0, 4.0, 5.0];
    
    let output = execute_graph(&graph, &input).expect("Execution failed");
    verify_output(&output, &expected, 1e-4);
}
```

**Multi-Input Operation Test:**
```rust
#[test]
fn test_scatter_elements() {
    let graph = GraphInfo { /* 3 inputs */ };
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let indices = vec![1.0, 3.0];  // Will be converted to i32
    let updates = vec![10.0, 20.0];
    
    let output = execute_graph_multi_input(&graph, vec![data, indices, updates])
        .expect("Execution failed");
    verify_output(&output, &expected, 1e-4);
}
```

**QDQ Roundtrip Test:**
```rust
#[test]
fn test_quantize_dequantize_roundtrip() {
    // Input → Q → DQ → Add → Output
    let graph = GraphInfo { /* 3 operations */ };
    let output = execute_graph_multi_input(&graph, vec![input, q_scale, dq_scale, add_val])
        .expect("Execution failed");
    verify_output(&output, &expected, 0.1);  // Allow quantization error
}
```

---

## Common Issues and Solutions

### Issue 1: Compilation Fails After Changes

**Symptoms:**
- Old errors persist after fixes
- Incremental compilation issues

**Solution:**
```powershell
Remove-Item -Recurse -Force target
cargo clean
cargo build --release --features trtx-runtime
```

### Issue 2: Test Failures with Type Mismatches

**Symptoms:**
```
[TensorRT Error] indices has type Float but must have type Int64 or Int32
```

**Solution:**
- Change operand descriptor to `DataType::Int32`
- Use `execute_graph_multi_input()` for automatic conversion

### Issue 3: Quantization Tests Fail

**Symptoms:**
```
[TensorRT Error] Node cannot be quantized by input
Value mismatch: actual=0, expected=2
```

**Solution:**
- Don't test quantize/dequantize in isolation
- Use QDQ roundtrip pattern with additional operation
- See `test_quantize_dequantize_roundtrip()` example

### Issue 4: Magic Constants in Code

**Symptoms:**
```rust
layer.set_operation(0);  // What does 0 mean?
```

**Solution:**
```rust
layer.set_operation(ElementWiseOperation::kSUM);  // Clear intent
```

### Issue 5: Weights Freed Too Early

**Symptoms:**
```
[TensorRT Error] Invalid weights pointer
Segmentation fault
```

**Solution:**
- Store weights in `TrtxConverter::temp_weights`
- Use stable pointers from stored `Vec<u8>`
- Don't pass temporary stack arrays

---

## Documentation Organization

### Essential Documents
1. **`trtx-implementation-status.md`** - Current implementation status (DO NOT MODIFY)
2. **`TRTX_DEVELOPMENT_GUIDE.md`** (this file) - Complete development guide

### Archived Documents
All incremental development documents have been consolidated into this guide. Historical documents are kept for reference but this guide is the authoritative source.

---

## Adding New Operations

### 1. Check TensorRT Documentation
- Search NVIDIA TensorRT API docs
- Check Chromium WebNN reference implementation
- Understand layer requirements and parameters

### 2. Add FFI Bindings (if needed)
```rust
// trtx-sys/src/lib.rs
generate!("INewLayer");
generate!("NewLayerEnum");

// trtx/src/network.rs
pub fn add_new_layer(&mut self, ...) -> NewLayer { ... }
```

### 3. Implement Converter
```rust
// src/converters/trtx.rs
fn add_new_op(&mut self, op: &Operation, graph: &GraphInfo, network: &mut NetworkDefinition) 
    -> Result<*mut ITensor, GraphError> {
    // Get inputs
    // Create layer
    // Set parameters
    // Return output tensor
}

// Add to dispatch
"newOp" => self.add_new_op(op, graph, network)?,
```

### 4. Add Tests
```rust
// tests/test_trtx_execution.rs
#[test]
fn test_new_op() {
    let graph = GraphInfo { /* definition */ };
    let input = vec![/* test data */];
    let expected = vec![/* expected output */];
    
    let output = execute_graph(&graph, &input).expect("Execution failed");
    verify_output(&output, &expected, 1e-4);
}
```

### 5. Update Documentation
```bash
# Update operation count in trtx-implementation-status.md
# Run tests to verify
cargo test --release --test test_trtx_execution --features trtx-runtime
```

---

## Future Work

### Remaining Operations (8)

**Not Yet Implemented:**
1. `cumulativeSum` - Cumulative sum along axis
2. `gatherElements` - Gather using index tensor
3. `isInfinite` - Check for infinite values
4. `isNaN` - Check for NaN values
5. `l2Pool2d` - L2 pooling (may require custom implementation)
6. `linear` - Linear transformation (α*x + β)
7. `notEqual` - Element-wise inequality
8. `reverse` - Reverse elements along axes

**Potential Implementations:**
- `l2Pool2d`: Square → avgPool2d → sqrt (multi-layer)
- `linear`: Scale layer with specific parameters
- Others: Check if TensorRT has direct layer support

### Testing Enhancements

1. **Quantized Operation Tests:**
   - Quantized Conv2d: Q → Conv(INT8) → DQ
   - Quantized MatMul: Q → MatMul(INT8) → DQ
   - Mixed precision graphs

2. **Edge Case Tests:**
   - Out-of-bounds indices (should fail gracefully)
   - Empty tensors
   - Maximum dimension sizes
   - Negative strides/dilations

3. **Performance Benchmarks:**
   - Compare TensorRT vs ONNX Runtime
   - Measure quantization speedup
   - Memory usage profiling

---

## Key Learnings

### 1. TensorRT Type Requirements Are Strict
- Indices must be Int32/Int64
- Boolean types required for logical ops
- Quantization requires QDQ patterns
- Type checking happens at graph construction

### 2. Enum Usage Improves Clarity
- Named constants are self-documenting
- Prevents magic number bugs
- Makes code reviewable
- Easier to maintain

### 3. Test Design Matters
- Test realistic patterns, not isolated operations
- Use proper QDQ context for quantization
- Helper functions improve test clarity
- Comprehensive tests prevent regressions

### 4. FFI Binding Generation
- Typedefs need special handling (generate base + alias)
- Some methods have unexpected names (`setMode` vs `setGatherMode`)
- Parameter passing varies (value vs reference)
- Always verify against C++ API docs

### 5. Weights Management
- Temporary data must outlive engine build
- Use `Vec<Vec<u8>>` for stable pointers
- Inner vectors remain at fixed addresses when outer vector grows
- Critical for constants and padding arrays

---

## References

### TensorRT Documentation
- [TensorRT C++ API](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)

### WebNN Specification
- [W3C WebNN Spec](https://www.w3.org/TR/webnn/)
- [WebNN Device Selection](https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md)
- [WebNN MLTensor](https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md)

### Reference Implementations
- [Chromium WebNN](https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/)
- [Chromium ONNX Backend](https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/ort/graph_builder_ort.cc)
- [Chromium CoreML Backend](https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/coreml/graph_builder_coreml.mm)

---

## Changelog

### 2026-01-29
- Implemented 7 new operations (gatherND, convTranspose2d, resample2d, scatterElements, scatterND, quantizeLinear, dequantizeLinear)
- Refactored to use TensorRT enums instead of magic constants
- Added typedef pattern for `ResizeMode` and similar types
- Enhanced QDQ testing with proper roundtrip pattern
- Fixed 5 TensorRT API signature issues
- Added multi-output support to test helpers
- Consolidated 33 documentation files into this guide
- **Final Status:** 96 tests passing, 97/105 operations (92% coverage)

---

*End of TensorRT Development Guide*

# CoreML Weight Files Implementation Plan

## Overview

Implement weight file support for CoreML MLProgram to handle Float16 non-scalar constants, following Chromium's architecture.

## Background

CoreML MLProgram (MIL) requires non-scalar Float16 constants to be stored in external weight files rather than as immediate values in the protobuf. This is an architectural requirement of the format.

**Current Issue:**
- 122 tests (4% of suite) crash with Float16 non-scalar constants
- Tests fail with "Fatal Python error: Aborted" during CoreML execution

**References:**
- Chromium implementation: `chromium/src/services/webnn/coreml/graph_builder_coreml.cc`
- CoreML protobuf: `protos/coreml/MIL.proto` (BlobFileValue message)

## Architecture

### Weight File Structure

```
model.mlpackage/
├── Data/
│   └── com.apple.CoreML/
│       ├── model.mlmodel (protobuf)
│       └── weights/
│           └── weights.bin (binary data with alignment)
```

### BlobFileValue Format

```protobuf
message BlobFileValue {
    string fileName = 1;  // Relative path: "@model_path/weights/weights.bin"
    uint64 offset = 2;    // Byte offset into weights.bin
}
```

### Alignment Requirements

- Each weight entry must be 64-byte aligned
- Metadata format (per Chromium):
  - Sentinel (4 bytes): 0xDEADBEEF
  - Count (8 bytes): number of elements
  - Data (variable): actual bytes
  - Padding: to next 64-byte boundary

## Implementation Phases

### Phase 1: Weight File Builder Infrastructure

**Goal:** Create core weight file management structure

**Files to Create:**
- `src/converters/weight_file_builder.rs`

**Components:**
```rust
pub struct WeightFileBuilder {
    data: Vec<u8>,           // Binary weight data
    offsets: HashMap<u32, u64>,  // operand_id -> file offset
}

impl WeightFileBuilder {
    pub fn new() -> Self;
    pub fn add_weight(&mut self, operand_id: u32, data: &[u8]) -> u64;
    pub fn finalize(self) -> Vec<u8>;
}
```

**Tasks:**
- [TODO] Create `weight_file_builder.rs` with basic structure
- [TODO] Implement 64-byte alignment logic
- [TODO] Add sentinel + count metadata format
- [TODO] Test alignment with various data sizes

**Acceptance Criteria:**
- Can add multiple weight entries
- Each entry is 64-byte aligned
- Returns correct offsets for retrieval

### Phase 2: Integrate with CoreML Converter

**Goal:** Detect Float16 constants and route to weight file

**Files to Modify:**
- `src/converters/coreml_mlprogram.rs`

**Changes:**
```rust
pub struct CoremlMlProgramConverter {
    weight_builder: Option<WeightFileBuilder>,  // New field
}

impl CoremlMlProgramConverter {
    fn create_const_operation() {
        // Detect Float16 non-scalar
        if is_float16 && !is_scalar {
            // Add to weight file
            let offset = self.weight_builder.add_weight(...);
            // Create BlobFileValue instead of immediate
            create_blob_file_value(offset);
        }
    }
}
```

**Tasks:**
- [TODO] Add `weight_builder` field to converter struct
- [TODO] Modify `create_const_operation()` to detect Float16 non-scalars
- [TODO] Implement `create_blob_file_value()` helper
- [TODO] Thread weight builder through conversion pipeline

**Acceptance Criteria:**
- Float16 scalars still use immediate values
- Float16 non-scalars go to weight file
- BlobFileValue has correct offset and filename

### Phase 3: MLPackage File Generation

**Goal:** Generate complete .mlpackage with weights

**Files to Modify:**
- `src/converters/coreml_mlprogram.rs` (convert method)
- `src/executors/coreml.rs` (if needed for file handling)

**Changes:**
```rust
impl GraphConverter for CoremlMlProgramConverter {
    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph> {
        // ... existing protobuf generation ...

        // Generate weights.bin if needed
        let weights_data = self.weight_builder.finalize();

        // Return both model and weights
        ConvertedGraph {
            format: "coreml_mlprogram",
            model_data: protobuf_bytes,
            weights_data: Some(weights_data),  // New field
        }
    }
}
```

**Tasks:**
- [TODO] Add `weights_data` field to `ConvertedGraph`
- [TODO] Update all converters to support optional weights
- [TODO] Modify file writing to create weights/ directory
- [TODO] Write weights.bin with proper permissions

**Acceptance Criteria:**
- .mlpackage contains weights/ directory when needed
- weights.bin file is created with correct data
- Model protobuf references correct weight file path

### Phase 4: CoreML Executor Integration

**Goal:** Ensure CoreML can load models with weight files

**Files to Modify:**
- `src/executors/coreml.rs`
- `src/python/context.rs` (compute_coreml path)

**Changes:**
- Ensure .mlpackage path is used (not individual files)
- CoreML runtime automatically loads weights from standard location
- No explicit weight loading needed (CoreML handles it)

**Tasks:**
- [TODO] Verify .mlpackage directory structure is correct
- [TODO] Test CoreML model loading with weights
- [TODO] Add error handling for weight file issues

**Acceptance Criteria:**
- CoreML successfully loads models with weight files
- Float16 constants are correctly populated
- Tests pass without crashes

### Phase 5: Testing & Validation

**Goal:** Verify Float16 constants work end-to-end

**Test Cases:**
1. Float16 scalar constant (should use immediate value)
2. Float16 1D constant [24] (should use weight file)
3. Float16 2D constant [3, 4] (should use weight file)
4. Multiple Float16 constants in same graph
5. Mixed Float32 immediate + Float16 weight file

**Tasks:**
- [TODO] Run `leakyRelu_float16_1D_constant_tensor_default_options-coreml`
- [TODO] Run full Float16 constant test suite (122 tests)
- [TODO] Verify weights.bin file size and alignment
- [TODO] Check CoreML execution results match expected values

**Acceptance Criteria:**
- All 122 Float16 constant tests pass
- No crashes or segmentation faults
- Results match ONNX backend (within Float16 precision)
- Overall WPT conformance improves from 91.3% to ~95%+

## Technical Details

### Alignment Calculation

```rust
fn align_to_64(offset: usize) -> usize {
    (offset + 63) & !63
}
```

### Weight Entry Format (Chromium-compatible)

```
[Sentinel: 4 bytes] 0xDEADBEEF
[Count: 8 bytes]    Number of elements (e.g., 24 for shape [24])
[Data: N bytes]     Raw Float16 bytes (2 bytes per element)
[Padding: M bytes]  Zero padding to next 64-byte boundary
```

### BlobFileValue Creation

```rust
fn create_blob_file_value(offset: u64, data_type: MilDataType, shape: &[i64]) -> Value {
    Value {
        type: Some(ValueType { /* populate */ }),
        value: Some(value::Value::BlobFileValue(value::BlobFileValue {
            file_name: "@model_path/weights/weights.bin".to_string(),
            offset,
        })),
    }
}
```

## Migration Strategy

1. **Phase 1-2:** Core infrastructure (no user-facing changes)
2. **Phase 3:** File generation (may see .mlpackage with weights/)
3. **Phase 4-5:** Enable and test (Float16 tests start passing)

## Rollout Plan

1. Implement Phases 1-3 without removing error check
2. Test manually with single Float16 constant
3. Enable for all Float16 constants
4. Run full test suite
5. Document any remaining limitations

## Success Metrics

- 122 Float16 constant tests passing (currently crashing)
- WPT conformance: 91.3% → ~95%+ (2700 → ~2820 passing)
- No regressions in existing tests
- Weight file generation adds <50ms overhead

## Future Enhancements

- Support weight files for other data types (Int8, Uint8) if needed
- Optimize by only using weight files when necessary
- Add weight file compression (if CoreML supports it)
- Share weight files across multiple models

## References

- CoreML MLModel format: https://apple.github.io/coremltools/
- Chromium WebNN: https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/coreml/
- W3C WebNN spec: https://www.w3.org/TR/webnn/

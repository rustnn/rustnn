Quantized support plan (int4, int8, fp16)
==========================================

Goal
----
Add full quantized graph support (int4/uint4, int8/uint8, fp16) across validator, converters, Python bindings, and backends, matching Chromium WebNN behavior for QuantizeLinear/DequantizeLinear and quantized ops.

Reference (Chromium lkgr)
-------------------------
- Quantize/Dequantize shape + axis/block handling: `services/webnn/ort/graph_builder_ort.cc` lines ~1097-1211.
- Dtype constraints (dequantize/quantize): `services/webnn/ort/context_impl_ort.cc` lines ~114-175.
- Logical ops bool↔uint8 casts: `services/webnn/ort/graph_builder_ort.cc` around 1230-1250.

Phase 1: Type plumbing and validation
-------------------------------------
- Add DataType variants for int4/uint4; ensure byte_length math handles 4-bit packing (likely treat as 1 byte each; note for follow-up if bit-packed needed).
- Extend ContextProperties allowed IO dtypes to include int4/uint4 (inputs/outputs) and ensure constants accept them.
- Add quantize/dequantize validation:
  - Shapes: scale/zero-point rank must equal input rank. Count non-1 dims; detect per-tensor (all 1 → reshape to scalar), per-axis (single varying dim → axis attr), blockwise (scale smaller than input; divisibility; expand; axis+block_size attr).
  - Dtypes: dequantize input ∈ {int4, uint4, int8, uint8, int32}; scale ∈ {float16, float32}; zero-point ∈ {int4, uint4, int8, uint8, int32}. Quantize input ∈ {float16, float32, int32}; zero-point ∈ {int4, uint4, int8, uint8}; output dtype = zero-point dtype.
  - Boolean convention: logical ops still use uint8; cast to bool for ONNX as needed.

Phase 2: Converter (ONNX first)
-------------------------------
- Emit QuantizeLinear/DequantizeLinear nodes with axis/block_size attributes per Chromium logic.
- Insert reshapes for per-tensor/per-axis; for blockwise, expand scale/zp to input shape.
- If zero-point is int4/uint4, cast to int8/uint8 before expand and cast back after.
- Preserve pending_permutation/layout info so axis is resolved correctly.
- Logical ops: keep uint8↔bool casts (already present).

Phase 3: CoreML/TensorRT adjustments
------------------------------------
- CoreML: propagate quantized tensors where supported; otherwise dequantize→float fallback; ensure channel axis mapping is correct for per-axis scales.
- TensorRT: enable INT8 path when available; fallback to dequantize→float if not.

Phase 4: Python bindings and API surface
----------------------------------------
- Expose int4/uint4/int8/fp16 dtypes in bindings/stubs and dtype mapping to/from NumPy.
- Keep boolean tensors as uint8.
- Consider a small helper/wrapper for quantize_linear that forwards axis/block metadata (optional).

Phase 5: Testing
----------------
- Rust unit tests: validator cases for per-tensor/per-axis/blockwise (int4/int8), converter snapshots for Quantize/Dequantize with axis/block_size and zero-point casts.
- Python tests: end-to-end quantized conv/matmul through ONNX Runtime; verify per-axis scales and uint8 bool handling.
- WPT fixtures: add/update quantizeLinear/dequantizeLinear/int8 cases if available.
- Always run `cargo test --lib` and `make python-test`.

Phase 6: Docs/status
--------------------
- Document supported quantized ops and constraints (scale float, zero-point int4/8/uint4/8, bias int32, per-axis rules, blockwise caveats).
- Update implementation-status and user guide sections.

Notes and open questions
------------------------
- Decide representation for 4-bit tensors in Rust (packed vs byte-per-value); we currently use byte-per-value for simplicity. Packed/buffer-level layout would be a follow-up (affects tensor storage, serialization, executor IO).
- Confirm ONNX backend kernel availability for QLinearConv/QLinearMatMul; fall back to float path when kernels missing.

Implementation Status (as of 2026-01-14)
-----------------------------------------

### COMPLETED ✓

**Phase 1: Type Plumbing and Validation** ✓
- Int4, Uint4, Float16 DataType variants added (src/graph.rs)
- Byte-per-value storage (1 byte per int4/uint4 element, bit-packing deferred)
- ContextProperties allows all quantized types in I/O
- Comprehensive quantize/dequantize validation (src/validator.rs):
  - Scale dtype constraints (float16/float32)
  - Zero-point dtype constraints (int4/uint4/int8/uint8/int32)
  - Shape rank matching
  - Per-tensor/per-axis/blockwise pattern detection
- GraphInfo.quantized flag added

**Phase 4: Python Bindings** ✓
- All quantized types exposed: "int4", "uint4", "float16", etc.
- quantize_linear() and dequantize_linear() methods
- MLGraph.save(path, quantized=True) API
- Type stubs updated (python/webnn/__init__.pyi)
- Int4/Uint4 map to int8/uint8 for NumPy compatibility

**Phase 5: Testing** ✓
- 8 Python quantization tests (tests/test_python_api.py):
  - test_dequantize_linear
  - test_quantize_linear
  - test_quantize_linear_uint8
  - test_dequantize_linear_uint8
  - test_quantize_dequantize_roundtrip
  - test_quantize_linear_int4_per_tensor
  - test_quantize_linear_uint4_per_axis
  - test_dequantize_linear_int4_blockwise
  - test_quantized_roundtrip_save_and_load
- Rust unit tests for quantized flag round-trips (src/webnn_json.rs)
- Validator unit tests for quantization constraints

**Serialization** ✓
- webnn-graph: GraphJson.quantized flag, SerializeOptions, @quantized annotation
- rustnn: webnn_json type conversion, round-trip preservation
- Text format: `webnn_graph "name" v1 @quantized { ... }`
- JSON format: `"quantized": true`

### IN PROGRESS / INCOMPLETE ⚠️

**Phase 2: ONNX Converter** ✓ MOSTLY COMPLETE (with limitations)
- QuantizeLinear/DequantizeLinear node generation implemented (src/converters/onnx.rs:1562-1696)
- Per-tensor reshape logic working ✓
- Per-axis reshape logic working ✓
- Axis attribute correctly emitted for per-axis quantization ✓ (VERIFIED with ONNX inspection)
- Int4/Uint4 type mapping to Int8 working ✓ (at type level)
- **LIMITATION**: Block_size attribute not emitted (requires ONNX opset 21+, currently using opset 14)
  - Blockwise quantization is validated and accepted
  - ONNX conversion works but block_size attribute not included
  - To enable: upgrade opset_import version from 14 to 21+ in src/converters/onnx.rs:3613
- See Chromium reference: services/webnn/ort/graph_builder_ort.cc:1097-1211

**Phase 3: CoreML/TensorRT** ⚠️ DEFERRED
- CoreML: MIL operations mapped, int4/uint4 explicitly rejected with error
- Int8/Uint8 supported through native MIL types
- Full int4/uint4 support deferred until CoreML MLProgram adds native support
- TensorRT: Not yet implemented

### CRITICAL ISSUES

**Repository Synchronization**
- rustnn branch: `quantize-webnn-graph` (uncommitted, 17 files modified)
- webnn-graph branch: `quantize` (uncommitted, 9 files modified)
- Cargo.toml points to GitHub main branches, but local work is on quantize branches
- Changes not pushed to GitHub
- **Fix Required**: Commit, push, and update dependencies

**Cargo.toml Dependency Mismatch** (lines 46-47):
```toml
webnn-graph = { git = "https://github.com/rustnn/webnn-graph", branch = "main" }
webnn-onnx-utils = { git = "https://github.com/rustnn/webnn-onnx-utils", branch = "main" }
```
Should point to `quantize` branches or merge to main first.

Next Immediate Steps (Priority Order)
--------------------------------------

1. **OPTIONAL: Upgrade to ONNX Opset 21** (30 minutes)
   - Change opset version from 14 to 21 in src/converters/onnx.rs:3613
   - Implement block_size attribute emission for blockwise quantization
   - Test with ONNX Runtime to ensure compatibility
   - **OR**: Document opset 14 limitation and proceed (recommended for now)

2. **Repository Synchronization** (CRITICAL - 1 hour)
   - Commit webnn-graph quantize branch changes
   - Push webnn-graph quantize to GitHub
   - Commit rustnn quantize-webnn-graph changes
   - Update rustnn Cargo.toml to use quantize branches
   - Push rustnn quantize-webnn-graph to GitHub

3. **End-to-End Testing** (1 hour)
   - Test quantized graph → ONNX conversion
   - Verify axis attributes in ONNX output
   - Verify block_size attributes for blockwise
   - Run full test suite: cargo test --lib && make python-test

4. **Documentation** (1 hour)
   - Update this plan with final status
   - Add quantization examples to user guide
   - Document per-tensor/per-axis/blockwise patterns
   - Update implementation-status.md

5. **CLI Enhancement** (30 minutes)
   - Add --quantized flag to CLI serialize command
   - Match Python API feature parity

Deferred Items
--------------
- CoreML int4/uint4 support (waiting for MLProgram native support)
- Bit-packing for 4-bit tensors (byte-per-value acceptable for now)
- QLinearConv/QLinearMatMul specialized kernels (use QuantizeLinear/DequantizeLinear fallback)
- TensorRT INT8 quantization path

Testing Before Merge
---------------------
- [ ] cargo test --lib (all Rust tests pass)
- [ ] make python-test (all Python tests pass)
- [ ] cargo fmt (code formatted)
- [ ] cargo clippy (no new warnings)
- [ ] Manual ONNX conversion test with axis/block_size attributes
- [ ] Save/load round-trip with quantized marker
- [ ] Int4/uint4 quantization end-to-end

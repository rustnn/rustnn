> **Archived (2026-09-16).** Historical document kept for reference only. It describes
> plans, investigations or code states that no longer match the repository (removed Python
> bindings, the old executor API, superseded designs, stale status numbers). Do not use it as
> an API or workflow reference; the current documentation starts at `docs/index.md`.

# WebNN Static Shapes Limitation

**Date:** 2026-01-09
**Status:** Documented limitation, not a bug

---

## Summary

WebNN models use **static shapes** compiled at conversion time and cannot handle truly dynamic shapes like ONNX models. This affects variable-length sequence models (e.g., LLMs with different prompt lengths).

---

## Impact

### For All Models
- ✅ **Bug Fix Applied:** Output ordering now ensures "logits" is always at index 0
- ✅ **Bug Fix Applied:** Empty KV cache inputs (past_sequence_length=0) automatically skipped
- ✅ **Converter Working:** WebNN→ONNX conversion faithfully reproduces graph structure

### For Variable-Length Models (LLMs)
- ⚠️ **Limitation:** WebNN models compiled for specific sequence length (e.g., 128 tokens)
- ⚠️ **Requirement:** Inputs must be padded to match compilation length exactly
- ⚠️ **Result:** Incorrect outputs if actual length differs from compiled length

---

## Technical Details

### Root Cause

WebNN specification requires:
1. All dynamic dimensions resolved to static values at conversion time
2. Shape calculations constant-folded for operations like Expand/Reshape
3. This creates "baked" models with fixed input shapes

### Example: SmolLM-135M

**Conversion Parameters:**
```json
{
  "past_sequence_length": 0,
  "past_sequence_length + 1": 128,
  "batch_size": 1,
  "sequence_length": 128
}
```

**What Gets Baked:**
- 218 Shape→Gather operations folded to constants
- Position embeddings computed for indices 0-127
- Attention masks expected shape [1, 128]
- All reshape operations use fixed dimensions

**Runtime Behavior:**
- ✅ 128-token input: Works correctly (matches compilation)
- ❌ 5-token input: Uses position 0-127 embeddings (wrong!)
- ❌ 200-token input: Cannot fit, shape mismatch

---

## Investigation Results

### Bug #1: Output Ordering - FIXED ✅

**Problem:** Logits at output index 60 instead of 0

**Fix:** `src/converters/onnx.rs` lines 783-807
```rust
// Sort outputs: "logits" first, then alphabetically by name
let mut sorted_outputs: Vec<u32> = graph.output_operands.clone();
sorted_outputs.sort_by_key(|&id| {
    let operand = graph.operand(id);
    let name = operand.and_then(|op| op.name.as_deref()).unwrap_or("");
    if name == "logits" {
        (0, String::new())  // Priority 0 = first
    } else {
        (1, name.to_string())  // Priority 1 = alphabetical
    }
});
```

**Benefit:** All models now have predictable output ordering

### Bug #2: Logit Corruption - LIMITATION DOCUMENTED ⚠️

**Initial Hypothesis:** ONNX→WebNN conversion corrupts model
**Actual Finding:** WebNN static shape limitation

**Fix Attempts:**
1. Disabled Gather folding in shape_inference.rs → Still folded
2. Found second location in convert.rs → Disabled both
3. Result: Conversion fails - Expand needs constant shapes
4. Conclusion: **Folding is mandatory for WebNN**

**Evidence:**
- Original ONNX: Dynamic shapes, works with any sequence length
- WebNN model: Static shapes, compiled for 128 tokens
- Converted ONNX: Preserves static compilation from WebNN

---

## Recommendations

### For Users

**When to use WebNN:**
- ✅ Fixed-size inputs (images, fixed-length sequences)
- ✅ Deployment scenarios with known input shapes
- ✅ Cross-platform inference with consistent shapes

**When to use ONNX:**
- ✅ Variable-length inputs (chatbot prompts, translations)
- ✅ Development and validation
- ✅ Models requiring dynamic batching

### For Developers

**Testing WebNN→ONNX Conversion:**
```python
# Correct: Use padding matching original WebNN compilation
input_ids = np.pad(tokens, (0, 128 - len(tokens)), constant_values=pad_token)

# Wrong: Use actual token count without padding
input_ids = np.array(tokens)  # Shape doesn't match compilation
```

**Documentation to Add:**
```python
def convert_to_onnx(self, graph, output_path):
    """
    Convert WebNN graph to ONNX format.

    WARNING: WebNN models use static shapes compiled at conversion time.
    The resulting ONNX model preserves these static shapes and will not
    handle variable-length inputs correctly.

    For models compiled with sequence_length=128, inputs must be padded
    to exactly 128 tokens. Use the original dynamic ONNX model for
    variable-length inference.
    """
```

---

## Files Modified

### Source Code (Ready to Commit)
- ✅ `src/converters/onnx.rs` - Output ordering fix + empty KV cache handling
- ✅ `src/python/context.rs` - Empty KV cache handling in compute methods

### Documentation (Keep for Reference)
- ✅ `docs/webnn_static_shapes_limitation.md` - This document
- ✅ `docs/investigation_complete.md` - Full investigation summary
- ✅ `docs/bug2_investigation_progress.md` - Detailed investigation steps

### Test Scripts (Keep for Debugging)
- ✅ `scripts/compare_onnx_graphs.py` - Graph comparison tool
- ✅ `examples/test_converted_onnx.py` - Test with proper padding

---

## WebNN Specification Context

### Community Discussion: Issue #883

The WebNN specification community is actively discussing dynamic shape support through [Issue #883: "Support Flexible Input Sizes"](https://github.com/webmachinelearning/webnn/issues/883). This issue documents the broader ecosystem need for runtime-determined dimensions.

**Issue Filed:** Multiple stakeholders request dynamic input shapes to handle:
- Vision models with multiple resolutions (e.g., MODNet with `[batch, 3, height, width]`)
- Speech recognition with growing KV cache (e.g., Whisper encoder/decoder)
- Language models with arbitrary sequence lengths (e.g., Qwen2.5-0.5B)
- OCR models requiring dynamic batch and spatial dimensions (e.g., Paddle OCR)

### Current Status: Under Investigation

**Kobe F2F Meeting (November 2025):**
The WebML Working Group resolved to "study more backends and do prototyping before more formally specifying solution." This indicates:
- ✅ **Acknowledged Need:** Dynamic shapes recognized as important use case
- ⏳ **Investigation Phase:** Not yet ready for formal specification
- 🔬 **Prototyping:** Backend implementations being studied for feasibility

**Timeline:** No committed timeline for dynamic shape support in WebNN specification.

### Technical Categories of Dynamic Shapes

Based on community discussion (Markus Tavenrath), dynamic shapes fall into three categories:

#### 1. Completely Unknown Dimensions
```python
# Example: Variable resolution image
input_shape = [batch, 3, ?, ?]  # height/width unknown
```
- **Challenge:** No upper bound for memory allocation
- **Backend Impact:** Requires deferred memory planning

#### 2. Symbolic Dimensions (Shared Symbols)
```python
# Example: Shared sequence length
input_ids = [batch, seq_len]
attention_mask = [batch, seq_len]  # Same symbolic 'seq_len'
```
- **Challenge:** Shape relationships must be maintained
- **Backend Impact:** Symbolic algebra during graph compilation

#### 3. Tensor-Derived Dimensions
```python
# Example: Shape computed from another tensor
indices = some_tensor
output_shape = [batch, indices.shape[0]]
```
- **Challenge:** Dimension values determined by runtime data
- **Backend Impact:** Data-dependent control flow

**Proposed Approach:** Focus on "bound dynamic shapes" where dimensions have maximum limits, enabling:
- Validation at graph build time (detect insufficient resources)
- Pre-allocated memory pools with known upper bounds
- Graceful failure before execution begins

### Proposed Solutions Under Discussion

#### 1. Graph Finalization Phase (Dwayne Robinson)
Introduce intermediate phase between `build()` and `dispatch()`:
```javascript
// Current API
const graph = builder.build(outputs);
const results = await context.compute(graph, inputs);

// Proposed API
const graph = builder.build(outputs);
const instantiatedGraph = graph.instantiate(concreteShapes);  // NEW
const results = await context.compute(instantiatedGraph, inputs);
```

**Benefits:**
- Shape computation and memory planning occur once per shape configuration
- Users control which instantiated shapes persist (memory management)
- Avoids rebuilding entire graph for each inference

**Challenges:**
- API complexity increase
- Unclear interaction with MLTensor lifecycle

#### 2. Bounded Symbolic Dimensions (Markus Tavenrath)
Declare maximum bounds at graph build time:
```javascript
const input = builder.input('tokens', {
    shape: [1, 'seq_len'],  // Symbolic dimension
    bounds: {'seq_len': [1, 2048]}  // Min/max bounds
});
```

**Benefits:**
- Early validation of resource requirements
- Pre-allocate maximum memory upfront
- Fail fast if bounds exceeded

**Challenges:**
- Memory overhead for maximum allocation
- Backend support varies (TensorRT supports, CoreML limited)

### Backend Landscape

**TensorRT (NVIDIA GPU):**
- ✅ Full dynamic shape support via optimization profiles
- ✅ Multiple shape configurations per engine
- 📖 [TensorRT Dynamic Shapes Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-dynamic-shapes.html)

**ONNX Runtime:**
- ✅ Native dynamic shape support in ONNX format
- ⚠️ WebNN EP currently [falls back to CPU for dynamic ops](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/helper.cc#L87)

**CoreML (Apple Neural Engine):**
- ⚠️ Limited dynamic shape support
- ✅ Supports flexible image dimensions with fixed aspect ratios
- ❌ No symbolic algebra for arbitrary dimension relationships

**Chromium WebNN:**
- ❌ No dynamic shape support in current implementation
- ⏳ Waiting for specification guidance before implementation

### Impact on Operations

Dynamic shapes affect multiple WebNN operations:

**Definitely Affected:**
- `input()` - Dimension declarations
- `shape()` - Runtime shape extraction
- `reshape()` - Target shape parameters
- `pooling()` - Output size calculations
- `convTranspose2d()` - Output size computations

**Potentially Affected:**
- All reduction operations with `axes` parameter
- Broadcasting operations (dynamic alignment)
- Slicing/indexing operations with tensor-derived indices

### Rustnn Position

**Current Implementation:**
- ✅ Correctly implements WebNN 1.0 specification (static shapes only)
- ✅ Faithful conversion preserves static compilation behavior
- ✅ No obligation to work around specification limitations

**When Specification Updates:**
If WebNN adds dynamic shape support, rustnn will:
1. Update to implement new specification features
2. Maintain backward compatibility with static shape models
3. Add backend-specific dynamic shape handling where supported
4. Document which backends support which dynamic shape categories

**User Guidance:**
Until WebNN specification adds dynamic shape support:
- Use ONNX format directly for variable-length inference
- Use WebNN for deployment scenarios with known input dimensions
- Compile multiple WebNN models for different shape configurations if needed

---

## Conclusion

**Rustnn Status:**
- ✅ WebNN→ONNX converter working correctly
- ✅ Output ordering fixed for all models
- ✅ Empty KV cache handling for LLM models
- ✅ Not responsible for static shape compilation
- ✅ Correctly implements current WebNN specification (v1.0)

**WebNN Specification (Current):**
- ⚠️ Static shapes only (no dynamic dimensions)
- ⚠️ Requires constant folding for shape operations
- ⚠️ Creates compiled models for specific input sizes
- ℹ️ By design, not a bug or limitation of rustnn

**WebNN Specification (Future):**
- 🔍 Dynamic shape support under active investigation (Issue #883)
- ⏳ No committed timeline for specification
- 🎯 Working Group studying backend capabilities and prototyping
- 📋 Multiple use cases identified (vision, speech, LLMs, OCR)
- 🔬 Proposed solutions: graph finalization phase, bounded symbolic dimensions

**Rustnn Roadmap:**
- ✅ **Now:** Fully compliant with WebNN 1.0 static shape specification
- 📅 **Future:** Will implement dynamic shapes when specification is released
- 🔄 **Migration:** Maintain backward compatibility with static models
- 🎯 **Backend Support:** Prioritize backends with native dynamic shape support (TensorRT, ONNX Runtime)

**Next Steps (Immediate):**
1. Commit Bug #1 fix and empty KV cache handling
2. Document WebNN static shape limitation in user-facing docs
3. Update test expectations for WebNN→ONNX round-trip
4. Add warnings about static shape limitations in API documentation

**Next Steps (Long-term):**
1. Monitor WebNN specification progress on dynamic shapes (Issue #883)
2. Prototype dynamic shape support when specification stabilizes
3. Add backend-specific dynamic shape implementations
4. Provide migration guide for users moving from static to dynamic shapes

---

**Investigation conducted:** 2026-01-09
**Time invested:** ~4 hours
**External repos analyzed:** webnn-wg (no changes needed)
**Specification context added:** 2026-01-09
**Reference:** [WebNN Issue #883 - Support Flexible Input Sizes](https://github.com/webmachinelearning/webnn/issues/883)

# Flexible Input Shapes Plan (Issue #15)

Scope: rustnn + webnn-graph + webnn-onnx-utils
Branch: tarek-flexible-input in all three repos

Guiding model: Chromium DynamicDimension { name, maxSize } and Dimension = static|dynamic. Dynamic dims allowed on inputs/outputs, forbidden on constants. Runtime checks enforce maxSize and same-name equality.

## Phase 1 - webnn-graph
1. [x] Add DynamicDimension and Dimension types.
2. [x] Change operand shape storage to Vec<Dimension>.
3. [x] Add helpers: StaticShape, GetStaticOrMaxSize, ToDimensionVector.
4. [x] Update JSON schema and (de)serialization to accept number or {name,maxSize}.
5. [x] Update wg serializer/JS/HTML emitters to handle Dimension.
6. [ ] Add tests for JSON parsing and StaticShape behavior (optional; pending).

## Phase 2 - webnn-onnx-utils
1. [~] Update shape conversion helpers to accept Dimension.
2. [~] Map dynamic dims to ONNX dim_param or -1 where appropriate.
3. [x] Add tests to confirm ONNX shape emission.

## Phase 3 - rustnn core model
1. [x] Mirror Dimension changes in src/graph.rs and update dependent structs.
2. [x] Update loader/webnn_json to parse dynamic dims and enforce constant shape concreteness.
3. [x] Update validator to allow dynamic dims on inputs/outputs and track known dynamic dims.
4. [~] Implement StaticShape usage where ops require concrete sizes (partial: using static_or_max in converters).
5. [x] Update shape_inference to be Dimension-aware and add ExpandShape rules.
   - Added Dimension-native inference helpers (broadcast/matmul/reduce/transpose/concat/unsqueeze/gather/where/expand).
   - Switched `webnn_json` inference pass to operate on `Vec<Dimension>` directly (no `static_or_max_shape` fallback in the pass).
   - Added explicit dynamic ExpandShape propagation rules and unit tests.

## Phase 4 - rustnn converters
1. [x] ONNX converter: emit dim_param for dynamic dims; avoid over-specifying shapes.
2. [x] Add runtime shape construction for Expand/Reshape when output shape has dynamic dims.
   - ONNX converter now builds runtime shape tensors (`Shape` + `Gather` + `Concat`) for dynamic `newShape`.
   - Applied to both `reshape` and `expand` conversion paths; static shapes still use constant initializers.
3. [x] Add dynamic-aware scale/bias generation for batch/instance/layer norm if needed.
   - ONNX converter now generates default normalization scale/bias using runtime shape vectors when normalized dimensions are dynamic.
   - Uses `Shape` + `Gather` (batch/instance) or `Shape` + `Slice` (layer norm), then `Expand` from scalar defaults.
4. [~] CoreML converter: map dynamic dims to UnknownDimension, keep maxSize for runtime checks only (currently uses static_or_max).

## Phase 5 - runtime checks
1. [x] Enforce actual tensor shape against descriptor.
2. [x] Ensure dynamic dims with same name match across inputs/outputs.
3. [x] Reject actual > maxSize.

Implemented in rustnn via runtime-checked executor entry points:
- `run_onnx_with_inputs_checked(...)`
- `run_trtx_with_inputs_checked(...)`
- `run_coreml_with_inputs_checked(...)`

Notes:
- Existing unchecked `run_*_with_inputs(...)` APIs are preserved for compatibility.
- Checked APIs enforce descriptor rank/static dims, same-name dynamic dim equality, and maxSize.

## Phase 6 - tests and docs
1. [~] Add JSON tests mirroring Chromium dynamic WPT coverage for a small subset of ops.
   - Added rustnn JSON/unit coverage for dynamic dimension parsing/roundtrip and constant-shape rejection.
2. [x] Document dynamic shape encoding and runtime behavior in docs.
   - Added `docs/development/flexible-input-shapes.md`.

Notes:
- TensorRT support deferred (explicitly reject dynamic dims on TRT path or fall back).
- Use Makefile targets for tests in rustnn.

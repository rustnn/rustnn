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
5. [ ] Update shape_inference to be Dimension-aware and add ExpandShape rules.

## Phase 4 - rustnn converters
1. [x] ONNX converter: emit dim_param for dynamic dims; avoid over-specifying shapes.
2. [ ] Add runtime shape construction for Expand/Reshape when output shape has dynamic dims.
3. [ ] Add dynamic-aware scale/bias generation for batch/instance/layer norm if needed.
4. [~] CoreML converter: map dynamic dims to UnknownDimension, keep maxSize for runtime checks only (currently uses static_or_max).

## Phase 5 - runtime checks
1. Enforce actual tensor shape against descriptor.
2. Ensure dynamic dims with same name match across inputs/outputs.
3. Reject actual > maxSize.

## Phase 6 - tests and docs
1. Add JSON tests mirroring Chromium dynamic WPT coverage for a small subset of ops.
2. Document dynamic shape encoding and runtime behavior in docs.

Notes:
- TensorRT support deferred (explicitly reject dynamic dims on TRT path or fall back).
- Use Makefile targets for tests in rustnn.

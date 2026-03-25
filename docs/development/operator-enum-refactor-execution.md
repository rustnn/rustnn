# Operator Enum Refactor — Execution Document

This document is the step-by-step execution guide for the Operator enum refactor. Use it to execute specific steps and track progress.

Please read the WebNN API documentation from here: https://www.w3.org/TR/webnn/.

---

## 1. Goals

- **Single source of truth per op**: One enum variant per WebNN builder, with all builder arguments (positional + options) in one place.
- **Stable, explicit operand roles**: No more "position 0 = input, position 1 = filter"; use named fields (`input`, `filter`, etc.) so meaning does not depend on `op_type` + `input_operands` order.
- **Preserve MLDimension**: Options that use `MLDimension` (e.g. expand `newShape`, reshape `newShape`, slice `sizes`) remain unchanged for dynamic dimensions.
- **Keep options structs**: Each variant still holds the corresponding `ML*Options` where the spec has options; only the container (operation + operands + options) changes.
- **Ignore ONNX when refactoring the RustNN core**: RustNN implements WebNN. ONNX is only of interest during the export and execution phase.

---

## 2. Current Pain Points

- **Operation** is a generic bag: `op_type: String`, `input_operands: Vec<u32>`, `attributes: OperatorOptions`. The meaning of `input_operands[i]` is implicit and op-specific (e.g. conv2d: 0=input, 1=filter; bias in options; layer_norm: 0=input, scale/bias in options).
- **Index fragility**: Reordering operands or building the same logical graph from different paths (WPT JSON vs Python) can change indices; any code that assumes "first input", "second input" is brittle.
- **Split of "positional" vs "options"**: Some builder args live in `input_operands`, others in `OperatorOptions`; that split is duplicated in every converter and easy to get wrong (e.g. bias/scale/recurrent_bias in options only).

---

## 3. Target Design

### 3.1 New Type: `Operator` Enum

- **One variant per WebNN builder** (same set as today: conv2d, add, softmax, batchNormalization, etc.).
- **Each variant** has:
  - Named fields for every **positional** argument (all `MLOperand` / builder inputs).
  - One field for the **options** struct for that op (the existing `ML*Options`), or no options field when the spec only has `MLOperatorOptions` (e.g. label only).

**Conceptual examples:**

```rust
Operator::Conv2d {
  input: OperandIndex,
  filter: OperandIndex,
  options: MLConv2dOptions,  // includes bias: Option<OperandIndex>, padding, strides, ...
}
Operator::Add {
  a: OperandIndex,
  b: OperandIndex,
  options: MLOperatorOptions,
}
Operator::Softmax {
  input: OperandIndex,
  axis: u32,  // required by spec
  options: MLOperatorOptions,
}
Operator::LayerNormalization {
  input: OperandIndex,
  options: MLLayerNormalizationOptions,  // scale, bias, axes, epsilon
}
Operator::Reshape {
  input: OperandIndex,
  options: MLReshapeOptions,  // new_shape: Vec<MLDimension> — keep for dynamic dims
}
```

- **OperandIndex**: Keep as `u32` (index into `graph.operands`). The gain is not changing the ID type but making **which operand is "input" vs "filter" vs "bias"** explicit in the variant, so reordering in the operands list does not change semantics.
- **MLDimension**: No change. Options that already use `MLDimension` (e.g. `MLExpandOptions.new_shape`, `MLReshapeOptions.new_shape`, `MLSliceOptions.sizes`) stay as they are.

### 3.2 `Operation` Struct

Replace `op_type`, `input_operands`, and `attributes` with a single field:

```rust
pub struct Operation {
  pub operator: Operator,
  pub output_operand: Option<u32>,
  pub output_operands: Vec<u32>,
  pub label: Option<String>,
}
```

Outputs stay on `Operation` (single- and multi-output ops unchanged).

### 3.3 `OperatorOptions` Enum

- **Keep** all existing `ML*Options` and the `OperatorOptions` enum for **options only**.
- `Operator` enum holds those same structs inside each variant (e.g. `options: MLConv2dOptions`), not the whole `OperatorOptions` tagged union.
- So: `OperatorOptions` remains the "all options types" union (still useful for deserialization and generic helpers); each `Operator` variant references the **concrete** options type for that op.

---

## 4. Phased Implementation Plan

### Phase 1: Introduce `Operator` and Keep Backward Compatibility

| Step | Action | Status |
|------|--------|--------|
| 1.1 | **New module** (e.g. `src/operators.rs` or under `operator_options.rs`): Define `Operator` enum with one variant per op. Each variant = builder name + named operand fields + options struct (or `MLOperatorOptions`). Reuse all existing `ML*Options` and `MLDimension` from `operator_options.rs`. | [x] |
| 1.2 | **Conversion helpers** (same crate): `Operator -> (op_type: &str, input_operands: Vec<u32>, attributes: OperatorOptions)` so existing code that expects (op_type, input_operands, attributes) can still run. | [x] |
| 1.3 | **Conversion helpers**: `(op_type, input_operands, attributes) -> Operator` for deserializing current JSON and for WPT path. | [x] |
| 1.4 | **`Operation`**: Single representation: `Operation { operator, output_operand, output_operands, label }`. No duplicated legacy fields; `op_type`, `input_operands`, `attributes` derived via `operator.to_legacy()` only at serialization and via accessors `op_type()`, `input_operands()`, `attributes()`. | [x] |
| 1.5 | **GraphInfo**: No change to `operands`, `input_operands`, `output_operands`, `constant_operand_ids_to_handles`; only `operations` now carry `Operator`. | [x] |

### Phase 2: Wire Producers to Build `Operator`

| Step | Action | Status |
|------|--------|--------|
| 2.1 | **WPT to Graph** (`tests/wpt_conformance/wpt_to_graph.rs`): Instead of building `Operation { op_type, input_operands, attributes }`, build `Operation { operator: Operator::X { ... }, output_operand(s), label }`. Use the same name-to-id map and attribute parsing you have now, but assign into the correct variant fields (e.g. conv2d: input, filter from args; bias from attributes into `MLConv2dOptions`). | [ ] |
| 2.2 | **JSON loader / build-from-JSON** (e.g. `webnn_json.rs`, any code that builds `GraphInfo` from JSON): Parse existing JSON (type + inputOperands + attributes) and convert into `Operator` (using the helper from phase 1), then into `Operation`. | [ ] |
| 2.3 | **Python / external builders** (if in this repo): Wherever they currently push `Operation` with `op_type` and `input_operands`, switch to constructing the appropriate `Operator` variant and `Operation { operator, ... }`. If they only produce JSON, phase 1 conversion (JSON -> Operator) is enough. | [ ] |

### Phase 3: Convert Consumers to Use `Operator` Only

| Step | Action | Status |
|------|--------|--------|
| 3.1 | **ONNX converter** (`src/converters/onnx.rs`): Replace `op.op_type` + `op.input_operands` + `op.attributes.as_*()` with a single match on `op.operator` and use variant fields (e.g. `Operator::Conv2d { input, filter, options }` -> use `input`, `filter`, `options.bias`, `options.padding`, etc.). This removes index-position assumptions. | [ ] |
| 3.2 | **CoreML converter** (`src/converters/coreml_mlprogram.rs`): Same: match on `op.operator` and use named fields and options. | [ ] |
| 3.3 | **TensorRT converter** (`src/converters/trtx.rs`): Same: match on `op.operator`, use variant fields. | [ ] |
| 3.4 | **Shape inference** (`src/shape_inference.rs`): Call sites today pass shapes and sometimes options. Have them derive inputs from `op.operator` (e.g. get operand IDs from the variant, then look up shapes in `graph.operands`). Signature of shape functions can stay shape/options-based; only the callers change to pull those from `Operator`. | [ ] |
| 3.5 | **Validator** (if it walks operations): Use `op.operator` for op kind and operand roles instead of `op_type` and `input_operands` indices. | [ ] |

### Phase 4: Drop Legacy Representation and Lock Format

| Step | Action | Status |
|------|--------|--------|
| 4.1 | **Remove** from `Operation`: `op_type`, `input_operands`, `attributes`. | [x] |
| 4.2 | **Serialization**: Custom `Serialize` for `Operation` derives `type`, `input_operands`, `attributes` from `operator.to_legacy()` so existing JSON format is unchanged. | [x] |
| 4.3 | **Deserialization**: Custom `Deserialize` parses legacy JSON (type, input_operands, attributes) and builds `Operator` via `from_legacy()` then `Operation { operator, ... }`. | [x] |

### Phase 5: Cleanup and Docs

| Step | Action | Status |
|------|--------|--------|
| 5.1 | **OperatorOptions**: Still used inside each `Operator` variant and for "get options by op type" helpers. No need to remove; document that it is the "options only" union and that `Operator` is the full "op + operands + options" representation. | [ ] |
| 5.2 | **OperandIndex**: Document that it is an index into `GraphInfo.operands`; semantics of each index are given by the `Operator` variant field name (input, filter, etc.). | [ ] |
| 5.3 | **Tests**: Update all tests that build or match on `Operation` (e.g. in `onnx.rs`, `coreml_mlprogram.rs`, `wpt_to_graph`) to use `Operator` and the new `Operation` shape. Add tests that deserialize legacy JSON and that serialize to the chosen JSON format. | [ ] |

---

## 5. Operand Reference Stability (Optional Later Step)

- **Current**: `OperandIndex = u32` is the index in `graph.operands`; reordering operands changes semantics unless every producer and consumer is updated.
- **With Operator**: Semantics are stable because each reference is named (e.g. `input`, `filter`). The same graph built in different order still has the same `Operator::Conv2d { input: 5, filter: 7, options }`; only the indices 5 and 7 might differ.
- **If you need true ID stability** (e.g. to reorder or deduplicate operands without breaking references): introduce a stable `OperandId` (e.g. newtype over u32 assigned at creation time and never reused) and use it in `Operator` and in `GraphInfo` (e.g. a map `OperandId -> Operand`). That can be a follow-up refactor after this plan is done.

---

## 6. MLDimension and Experimental Dynamic Dimensions

- **Do not change** option structs that use `MLDimension`:
  - `MLExpandOptions.new_shape: Vec<MLDimension>`
  - `MLReshapeOptions.new_shape: Vec<MLDimension>`
  - `MLSliceOptions.sizes: Vec<MLDimension>`
- In the new design they stay as the `options` field of the corresponding variant (e.g. `Operator::Reshape { input, options: MLReshapeOptions }`). No extra "dimension" enum needed; `Operator` just carries the existing options.

---

## 7. Scope and Order Summary

| Area | Action |
|------|--------|
| `operator_options.rs` | Keep as-is (ML*Options, MLDimension, OperatorOptions). Add or use from new `operators.rs`. |
| New `operators.rs` | Define `Operator` enum (~90 variants to match current ops). |
| `graph.rs` | Change `Operation` to hold `operator: Operator`; add (then remove) legacy fields and (de)serialization. |
| `converters/onnx.rs` | Replace op_type/input_operands/attributes with match on `op.operator` (~164 call sites). |
| `converters/coreml_mlprogram.rs` | Same (~68). |
| `converters/trtx.rs` | Same. |
| `shape_inference.rs` | Callers get operand IDs and options from `op.operator`; keep function signatures shape/options-based where possible. |
| `wpt_to_graph.rs` | Build `Operator` variants and `Operation { operator, ... }`. |
| Validator / loader | Use `op.operator` everywhere. |
| Tests | Update to construct and match on `Operator`; add (de)serialization tests. |

Doing the refactor in the order above (introduce Operator + compatibility layer, then producers, then consumers, then remove legacy, then cleanup) keeps the codebase buildable and testable at each step and avoids a single "flag day" change.

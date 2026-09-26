> **Archived (2026-09-16).** Historical document kept for reference only. It describes
> plans, investigations or code states that no longer match the repository (removed Python
> bindings, the old executor API, superseded designs, stale status numbers). Do not use it as
> an API or workflow reference; the current documentation starts at `docs/index.md`.

# Plan: Fix ONNX → WebNN Conversion Divergence

Audience: developer working on the ONNX→WebNN converter in `rust-webnn-graph`.

## Key Artifacts and Paths
- Original ONNX (good): `/Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn/model-static.onnx`
- Alt ONNX (onnxsim, also good): `/Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn/model.onnx`
- WebNN model to inspect: `/Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn/model.webnn` (+ `manifest.json`, `model.weights`)
- WebNN export produced by comparison script: `/Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn/webnn-export.onnx`
- Comparison script (with WebNN export option): `examples/compare_minilm_onnx.py`
  - Example command:
    ```
    DYLD_LIBRARY_PATH=target/onnxruntime/onnxruntime-osx-arm64-1.23.2/lib \
    .venv-webnn/bin/python examples/compare_minilm_onnx.py \
      --alt-onnx /Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn/model.onnx \
      --webnn-dir /Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn \
      --export-webnn-onnx /Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn/webnn-export.onnx
    ```
- Converter plan doc: `/Users/tarekziade/Dev/webnn-wg/webnn-graph_onnx_to_webnn_plan.md`

## Current Symptoms (to eliminate)
- WebNN output is wrong: cosine ≈ 0.005 vs HF reference and vs original ONNX.
- WebNN-export ONNX matches WebNN runtime (cos 1.0) → divergence baked into the converted graph, not runtime.
- Output shape lost: `last_hidden_state []` instead of `[1,128,384]`.
- Graph bloat: nodes 317 → 397; extra ops Shape(49), Concat(24), Clip(3), Slice(1), Unsqueeze(2 vs 1), Gather(3 vs 2).
- Initializers: only 7 shared names; 104 originals missing; 464 new `Constant_*`. Weights exist (bytes similar) but are remapped/renamed.

## Goals (aligned with static WebNN requirement)
1. Preserve static shapes end-to-end (outputs must have `[1,128,384]`).
2. Fold shape/broadcast logic at conversion time; avoid runtime Shape/Concat/Unsqueeze ladders.
3. Preserve or deterministically map initializers/weights to manifest entries; avoid losing original names.
4. Avoid injecting extra ops when they can be resolved statically.

## Step-by-Step Plan

### A) Use the comparison harness
- Run `examples/compare_minilm_onnx.py` with `--export-webnn-onnx` to inspect converted graphs and outputs.
- Targets after fixes: cosine ~1.0 for Transformers vs WebNN and Transformers vs WebNN-export; output shape intact.

#### Latest run (2025-12-28)
Command:
```
DYLD_LIBRARY_PATH=target/onnxruntime/onnxruntime-osx-arm64-1.23.2/lib \
.venv-webnn/bin/python examples/compare_minilm_onnx.py \
  --alt-onnx /Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn/model.onnx \
  --webnn-dir /Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn \
  --export-webnn-onnx /Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn/webnn-export.onnx
```
Notes:
- Hugging Face fetch attempts fail in restricted network; runs may still succeed if the HF cache is warm.
- Export now succeeds, but WebNN still diverges.

Observed output:
- Transformers vs WebNN: average cosine ~0.004845 (bad).
- WebNN vs WebNN-export: cosine 1.0 (export matches runtime).
- ONNX Runtime warning: `last_hidden_state` shape merged from `{1,128,384}` to `{}`.

### B) Analyze current WebNN-export vs original
- Differences observed:
  - Output shape becomes `[]`.
  - Many extra Shape/Concat/Clip/Unsqueeze ops; first nodes show injected unsqueezes with `expand_*_axes`.
  - Initializer names mostly replaced by `Constant_*`.
- Hypothesis: converter materializes dynamic/broadcast shape logic instead of folding it; manifest/name mapping is broken.

### C) Implement concrete fixes
1. **Preserve output shapes**
   - Carry inferred shapes into WebNN graph outputs; ensure `last_hidden_state` is `[1,128,384]`.
   - Verify `MLGraph.load` + `webnn_json::from_graph_json` do not drop shapes; if shapes are missing in AST, propagate from ONNX inference.
2. **Fold shape/broadcast logic (staticization)**
   - Constant-fold shape ops (Shape→Gather→Concat→Reshape, Expand axes) before lowering.
   - For broadcasts (attention_mask, bias): compute broadcasted shapes statically; avoid emitting runtime Shape/Concat/Unsqueeze chains. Use a single static unsqueeze if needed for WebNN op signatures.
   - Reshape: require constant shape; fold `-1/0` rules at conversion time; emit static target shape.
   - Remove/avoid Shape ops in the final graph; only emit static shapes.
3. **Preserve weight mapping**
   - Keep original initializer names when emitting consts; map them consistently to manifest entries (apply the same sanitization to manifest keys and graph references).
   - Avoid the generic `Constant_*` swarm; ensure all originals appear and none are spuriously renamed.
4. **Limit op inflation**
   - Do not insert extra Concats/Clips unless they exist in source or are strictly required; for static shapes, prefer direct const shapes/axes.

### D) Validation loop
1. Re-run `examples/compare_minilm_onnx.py --export-webnn-onnx …`.
2. Expect:
   - Cosine ≈ 1.0 for Transformers vs WebNN and vs WebNN-export.
   - `last_hidden_state` shape `[1,128,384]`.
   - Node/initializer counts close to original; no Shape/Concat explosion.
3. If still diverging, diff first nodes/ops and initializer names to find remaining dynamic/broadcast artifacts.

### E) Reference fixes to check in code
- WebNN loader/parser: `src/python/graph.rs`, `webnn_json::from_graph_json` — ensure shapes/constants are preserved, and manifest name sanitization matches graph references.
- Converter (ONNX→WebNN): add or adjust:
  - Static shape/type inference.
  - Constant folding of shape/broadcast ops.
  - Correct initializer → manifest mapping with stable naming.
  - Emitting WebNN ops without extra shape machinery.

### Quick signatures to eliminate
- Output shape `[]` for `last_hidden_state` (must be `[1,128,384]`).
- Added ops: Shape(49), Concat(24), Clip(3), Slice(1), extra Unsqueeze/Gather.
- Initializers remapped to `Constant_*` (should preserve originals).

Once these are fixed, WebNN and WebNN-export should align numerically with the original ONNX (cosine ~1.0).***

## Progress log

### 2025-12-28: Shape inference improvements in loader path
Work done:
- Updated `src/webnn_json.rs` in rust-webnn-graph to infer shapes for:
  - `expand` (both `axes` and `newShape`)
  - `shape`
  - `gather`
  - `slice`
  - `reduce*` ops
  - `constant` (from `shape` option)
- Added deterministic data type propagation (e.g., `shape` -> int64, `constant` from `dataType`).

Result after update:
- `examples/compare_minilm_onnx.py` still diverges (cosine ~0.004845).
- ONNX Runtime still warns about output shape merge for `last_hidden_state`.
- Sanity check shows many empty shapes remain.

Sanity check command:
```
DYLD_LIBRARY_PATH=target/onnxruntime/onnxruntime-osx-arm64-1.23.2/lib \
.venv-webnn/bin/python - <<'PY'
import webnn
from pathlib import Path
model_dir = Path("/Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn")
graph = webnn.MLGraph.load(
    str(model_dir / "model.webnn"),
    manifest_path=str(model_dir / "manifest.json"),
    weights_path=str(model_dir / "model.weights"),
)
print("output_names:", graph.get_output_names())
print("empty_shapes:", graph.count_empty_shapes())
print("operand_count:", graph.operand_count)
print("operation_count:", graph.operation_count)
for name in graph.get_output_names():
    for i in range(graph.operand_count):
        desc = graph.debug_operand(i)
        if f'name=Some("{name}")' in desc:
            print(desc)
            break
PY
```
Observed:
- output_names: ['last_hidden_state']
- empty_shapes: 637
- operand_count: 742
- operation_count: 445
- Output operand still has `shape=[]`.

### 2025-12-28: Shape propagation + negative axes fix, results now match
Changes:
- Rebuilt Python extension after Rust edits (`make python-dev`) so loader inference runs.
- `webnn_json::infer_output_shapes` now normalizes op types to lowercase to ensure matching.
- ONNX converter now preserves negative axes by parsing `axes` as i64 (not u64).

Sanity check (post-rebuild):
- `empty_shapes` dropped to 268.
- `last_hidden_state` shape resolved to `[1, 128, 384]`.

Comparison run:
```
DYLD_LIBRARY_PATH=target/onnxruntime/onnxruntime-osx-arm64-1.23.2/lib \
.venv-webnn/bin/python examples/compare_minilm_onnx.py \
  --alt-onnx /Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn/model.onnx \
  --webnn-dir /Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn \
  --export-webnn-onnx /Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn/webnn-export.onnx
```
Results:
- Transformers vs WebNN: average cosine 1.0 (match).
- Transformers vs WebNN->ONNX export: average cosine 1.0.
- ONNX (original) vs WebNN: average cosine 1.0.

### 2025-12-28: Trace remaining empty shapes + concat scalar fix
Trace findings:
- `count_empty_shapes` was dominated by scalar constants (shape `[]`) and concat outputs used only for shape-vector construction.
- `shape(...)` outputs are populated (e.g., `/embeddings/Shape_output_0` -> `[2]`).
- `concat` outputs were empty because inputs are rank-0 scalars and `infer_concat_shape` rejects axis 0 for rank 0.

Quick trace script:
```
DYLD_LIBRARY_PATH=target/onnxruntime/onnxruntime-osx-arm64-1.23.2/lib \
.venv-webnn/bin/python - <<'PY'
import re
import webnn
from pathlib import Path
model_dir = Path("/Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn")
graph = webnn.MLGraph.load(
    str(model_dir / "model.webnn"),
    manifest_path=str(model_dir / "manifest.json"),
    weights_path=str(model_dir / "model.weights"),
)
empty = []
for i in range(graph.operand_count):
    desc = graph.debug_operand(i)
    m = re.search(r'name=Some\\(\"([^\"]+)\"\\).*shape=([^)]*)', desc)
    if not m:
        continue
    name = m.group(1)
    shape_str = m.group(2).strip()
    if shape_str == "[]":
        empty.append(name)
print("empty_shapes:", len(empty))
print("concat empty:", len([n for n in empty if "Concat" in n]))
print("constant empty:", len([n for n in empty if "Constant" in n]))
PY
```

Historical fix (superseded by the shared `GraphRecorder` inference path):

> Current invariant: every `OperandDescriptor` in a completed `GraphInfo` has a known shape, and `[]` means only a rank-0 scalar. The empty-shape counting helpers below were diagnostic heuristics for the former loader and must not be used to identify unresolved shapes. Failed inference is now represented before insertion and returned as an error.

- `webnn_json::infer_output_shapes` now treats concat of scalar inputs with `axis=0` as a shape vector with length `N` (number of inputs).
- Added loader helpers to separate empty shapes for constants vs non-constants:
  - `MLGraph.count_unknown_shapes()` counts empty shapes for non-constants (heuristic for unknown shapes).
  - `MLGraph.count_scalar_constants()` counts empty shapes for constant scalars.
  - `MLGraph.count_unknown_shapes_excluding_scalar_ops()` excludes reduction ops that reduce all known dimensions.
  - `MLGraph.debug_unknown_shapes()` lists empty-shape non-constants with producing op and input shapes.
  - `MLGraph.debug_unknown_shapes_structured()` returns structured dicts with `id`, `name`, `op`, and `inputs`.

Sanity check (post-rebuild):
- `empty_shapes` dropped to 244.
- `unknown_shapes` (non-constant empty shapes) = 52.
- `scalar_constants` = 192.
- `unknown_shapes_excluding_scalar_ops` = 0 (all unknowns were scalar `constant` ops).
- concat outputs now have a shape (no empty concat outputs).
- output `last_hidden_state` still `[1, 128, 384]`.

Unknown-shape trace:
```
DYLD_LIBRARY_PATH=target/onnxruntime/onnxruntime-osx-arm64-1.23.2/lib \
.venv-webnn/bin/python - <<'PY'
import re
import webnn
from pathlib import Path
from collections import Counter
model_dir = Path("/Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn")
graph = webnn.MLGraph.load(
    str(model_dir / "model.webnn"),
    manifest_path=str(model_dir / "manifest.json"),
    weights_path=str(model_dir / "model.weights"),
)
items = graph.debug_unknown_shapes()
print("unknown_shapes:", len(items))
op_counts = Counter()
for line in items:
    m = re.search(r'op=([^,]+)', line)
    op = m.group(1) if m else "<none>"
    op_counts[op] += 1
print("by op type:", op_counts.most_common())
print("example:", items[0] if items else "<none>")
PY
```
Observed:
- All 52 unknown-shape entries are produced by `constant` ops (scalar constants), so there are no unknown data tensor shapes remaining.

### 2025-12-28: Demo model name correction (L6, not L12)
Updates:
- The MiniLM demo now targets **all-MiniLM-L6-v2** instead of L12 (typo fix).
- Updated local model path to `/Users/tarekziade/Dev/all-MiniLM-L6-v2-webnn`.
- Updated README and demo strings to L6.

Notes:
- The local L6 WebNN model directory is now present at `/Users/tarekziade/Dev/all-MiniLM-L6-v2-webnn`.

### 2025-12-28: MiniLM L6 demo now matches Transformers
Fixes:
- Transformers embedder now always loads the HF model name (L6), not the local WebNN directory.
- WebNN embedder tokenizer now also loads from HF model name (L6), avoiding invalid config.json in the WebNN export folder.

Demo run:
```
DYLD_LIBRARY_PATH=target/onnxruntime/onnxruntime-osx-arm64-1.23.2/lib \
.venv-webnn/bin/python examples/minilm_embeddings.py
```
Results:
- Transformers vs WebNN cosine similarities: all 1.0 (match).
- Embeddings are VERY SIMILAR (cosine > 0.99).

Structured debug example:
```
DYLD_LIBRARY_PATH=target/onnxruntime/onnxruntime-osx-arm64-1.23.2/lib \
.venv-webnn/bin/python - <<'PY'
import webnn
from pathlib import Path
model_dir = Path("/Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn")
graph = webnn.MLGraph.load(
    str(model_dir / "model.webnn"),
    manifest_path=str(model_dir / "manifest.json"),
    weights_path=str(model_dir / "model.weights"),
)
items = graph.debug_unknown_shapes_structured()
print(type(items), len(items))
print(items[0] if items else None)
PY
```
Example output:
- `{'id': 300, 'name': '/Constant_output_0', 'op': 'constant', 'inputs': []}`

MiniLM comparison rerun (post-structured debug helper):
- All cosine checks remain 1.0 (Transformers vs WebNN, WebNN->ONNX, and original ONNX).

Comparison run:
```
DYLD_LIBRARY_PATH=target/onnxruntime/onnxruntime-osx-arm64-1.23.2/lib \
.venv-webnn/bin/python examples/compare_minilm_onnx.py \
  --alt-onnx /Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn/model.onnx \
  --webnn-dir /Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn \
  --export-webnn-onnx /Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn/webnn-export.onnx
```
Results:
- Transformers vs WebNN: average cosine 1.0.
- Transformers vs WebNN->ONNX export: average cosine 1.0.
- ONNX (original) vs WebNN: average cosine 1.0.

### Next debugging steps (for restart)
1. Treat every empty descriptor shape as a known scalar; inference failures must surface before an operand is inserted.
2. Re-run sanity and comparison commands above if any loader or shape-inference changes are made.

## Environment notes
- Use `DYLD_LIBRARY_PATH=target/onnxruntime/onnxruntime-osx-arm64-1.23.2/lib` when running Python.
- WebNN export writes to `/Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn/webnn-export.onnx`.
- If HF downloads fail, ensure cached HF model is available or run with network access.

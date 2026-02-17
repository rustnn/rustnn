#!/usr/bin/env python3
"""
Generate docs/development/backend-operator-support.md from converter sources.

This script is intentionally source-driven to reduce documentation drift.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass


ROOT = pathlib.Path(__file__).resolve().parents[1]
ONNX_SRC = ROOT / "src/converters/onnx.rs"
COREML_SRC = ROOT / "src/converters/coreml_mlprogram.rs"
TRTX_SRC = ROOT / "src/converters/trtx.rs"
OUTPUT = ROOT / "docs/development/backend-operator-support.md"


EXCLUDED_OPS = {
    # Internal/pseudo ops, not user-facing WebNN operators.
    "constant",
    "shape",
}


DISPLAY_OVERRIDES = {
    "convtranspose2d": "convTranspose2d",
    "averagepool2d": "averagePool2d",
    "maxpool2d": "maxPool2d",
    "globalaveragepool": "globalAveragePool",
    "globalmaxpool": "globalMaxPool",
    "batchnormalization": "batchNormalization",
    "instancenormalization": "instanceNormalization",
    "layernormalization": "layerNormalization",
    "hardsigmoid": "hardSigmoid",
    "hardswish": "hardSwish",
    "leakyrelu": "leakyRelu",
    "logicaland": "logicalAnd",
    "logicalor": "logicalOr",
    "logicalxor": "logicalXor",
    "logicalnot": "logicalNot",
    "greaterorequal": "greaterOrEqual",
    "lesserorequal": "lesserOrEqual",
    "quantizelinear": "quantizeLinear",
    "dequantizelinear": "dequantizeLinear",
    "scatterelements": "scatterElements",
    "scatternd": "scatterND",
    "gatherelements": "gatherElements",
    "gathernd": "gatherND",
    "argmax": "argMax",
    "argmin": "argMin",
    "roundeven": "roundEven",
    "resample2d": "resample2d",
    "cumulativesum": "cumulativeSum",
    "isnan": "isNaN",
    "isinfinite": "isInfinite",
    "notequal": "notEqual",
}


@dataclass
class BackendOps:
    backend: str
    converter_ops: list[str]
    executor_ops: list[str]
    converter_source: str
    executor_source: str


def _normalize(op: str) -> str:
    return re.sub(r"[^a-z0-9]", "", op.lower())


def _display(op: str) -> str:
    n = _normalize(op)
    return DISPLAY_OVERRIDES.get(n, op)


def _extract_quoted_ops(text: str, pattern: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(pattern, text, flags=re.MULTILINE):
        values.append(match.group(1))
    return values


def _collect_from_matches_macro(text: str) -> list[str]:
    values: list[str] = []
    pattern = re.compile(r"matches!\(\s*op\.op_type\.as_str\(\),([\s\S]*?)\)")
    for match in pattern.finditer(text):
        block = match.group(1)
        for q in re.finditer(r'"([A-Za-z0-9_]+)"', block):
            values.append(q.group(1))
    return values


def parse_onnx_ops(text: str) -> list[str]:
    ops: list[str] = []
    ops.extend(
        _extract_quoted_ops(text, r'op\.op_type\s*==\s*"([A-Za-z0-9_]+)"')
    )
    ops.extend(
        _extract_quoted_ops(
            text, r'op\.op_type\.eq_ignore_ascii_case\("([A-Za-z0-9_]+)"\)'
        )
    )
    ops.extend(_collect_from_matches_macro(text))
    return canonicalize_ops(ops)


def parse_coreml_ops(text: str) -> list[str]:
    marker = "let mil_type = match webnn_op.to_lowercase().as_str() {"
    block = extract_brace_block_after_marker(text, marker)
    ops = [q.group(1) for q in re.finditer(r'"([A-Za-z0-9_]+)"\s*=>', block)]
    return canonicalize_ops(ops)


def parse_trtx_ops(text: str) -> list[str]:
    marker = "match op_type {"
    block = extract_brace_block_after_marker(text, marker)
    ops = [q.group(1) for q in re.finditer(r'"([A-Za-z0-9_]+)"\s*=>', block)]
    return canonicalize_ops(ops)


def extract_brace_block_after_marker(text: str, marker: str) -> str:
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"Could not find marker: {marker}")
    brace_open = text.find("{", start)
    if brace_open < 0:
        raise RuntimeError(f"Could not find opening brace after marker: {marker}")

    depth = 0
    i = brace_open
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_open + 1 : i]
        i += 1
    raise RuntimeError(f"Unbalanced braces while parsing marker: {marker}")


def canonicalize_ops(ops: list[str]) -> list[str]:
    by_norm: dict[str, str] = {}
    for raw in ops:
        n = _normalize(raw)
        if not n or n in EXCLUDED_OPS:
            continue
        current = by_norm.get(n)
        if current is None:
            by_norm[n] = raw
            continue
        # Prefer camelCase-ish forms over all-lower where available.
        if any(c.isupper() for c in raw) and not any(c.isupper() for c in current):
            by_norm[n] = raw

    display_ops = [_display(v) for v in by_norm.values()]
    return sorted(display_ops, key=lambda s: s.lower())


def bullet_columns(items: list[str], cols: int = 3) -> str:
    if not items:
        return "- (none)\n"
    rows = (len(items) + cols - 1) // cols
    table: list[list[str]] = [["" for _ in range(cols)] for _ in range(rows)]
    for idx, item in enumerate(items):
        r = idx % rows
        c = idx // rows
        table[r][c] = f"`{item}`"
    lines = []
    for row in table:
        vals = [v for v in row if v]
        lines.append("- " + ", ".join(vals))
    return "\n".join(lines) + "\n"


def render(backends: list[BackendOps]) -> str:
    out: list[str] = []
    out.append("# Backend Operator Support Report")
    out.append("")
    out.append(
        "This file is generated from converter sources by "
        "`scripts/generate_backend_operator_report.py`."
    )
    out.append(
        "Do not edit this file manually. Run `make docs-backend-ops` after backend changes."
    )
    out.append("")
    for b in backends:
        out.append(f"## {b.backend}")
        out.append("")
        out.append(f"- Converter source: `{b.converter_source}`")
        out.append(f"- Executor source: `{b.executor_source}`")
        out.append(f"- Converter operator count: **{len(b.converter_ops)}**")
        out.append(f"- Executor operator count: **{len(b.executor_ops)}**")
        out.append("")
        out.append("### Converter Operators")
        out.append("")
        out.append(bullet_columns(b.converter_ops).rstrip())
        out.append("")
        out.append("### Executor Operators")
        out.append("")
        if b.executor_ops == b.converter_ops:
            out.append(
                "Executor-level operator coverage follows converter coverage for this backend."
            )
            out.append("")
        out.append(bullet_columns(b.executor_ops).rstrip())
        out.append("")
    return "\n".join(out) + "\n"


def build_report() -> str:
    onnx_text = ONNX_SRC.read_text(encoding="utf-8")
    coreml_text = COREML_SRC.read_text(encoding="utf-8")
    trtx_text = TRTX_SRC.read_text(encoding="utf-8")

    onnx_ops = parse_onnx_ops(onnx_text)
    coreml_ops = parse_coreml_ops(coreml_text)
    trtx_ops = parse_trtx_ops(trtx_text)

    backends = [
        BackendOps(
            backend="ONNX Runtime Backend",
            converter_ops=onnx_ops,
            executor_ops=onnx_ops,
            converter_source="src/converters/onnx.rs",
            executor_source="src/executors/onnx.rs",
        ),
        BackendOps(
            backend="CoreML MLProgram Backend",
            converter_ops=coreml_ops,
            executor_ops=coreml_ops,
            converter_source="src/converters/coreml_mlprogram.rs",
            executor_source="src/executors/coreml.rs",
        ),
        BackendOps(
            backend="TensorRT Backend",
            converter_ops=trtx_ops,
            executor_ops=trtx_ops,
            converter_source="src/converters/trtx.rs",
            executor_source="src/executors/trtx.rs",
        ),
    ]
    return render(backends)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if output is stale")
    parser.add_argument(
        "--output",
        default=str(OUTPUT),
        help="Output path (default: docs/development/backend-operator-support.md)",
    )
    args = parser.parse_args()

    output_path = pathlib.Path(args.output)
    generated = build_report()

    if args.check:
        existing = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        if existing != generated:
            sys.stderr.write(
                f"{output_path} is out of date. Run make docs-backend-ops and commit the result.\n"
            )
            return 1
        print(f"{output_path} is up to date.")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generated, encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

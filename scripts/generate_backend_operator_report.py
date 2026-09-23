#!/usr/bin/env python3
"""
Generate docs/development/backend-operator-support.md from the converter sources.

The report is source-driven so that it cannot drift from the code: CI runs this script
with --check and fails when the committed report differs from the generated one.

Operation names come from `Operation::op_type()` in src/operators.rs. Per backend, an
operation counts as supported when:

- ONNX Runtime, LiteRT: the converter source references the `Operation::<Variant>`
  (these converters dispatch on the enum and have no "unsupported operation" fallthrough
  for referenced variants).
- CoreML: as above, plus operation names the converter compares against its lower-cased
  op type (`get_mil_op_type` keys, `op_type_lower == "..."`, `matches!(op_type_lower...)`).
- TensorRT: the operation name is a key of the converter's `match op_type { ... }` table.
- CANN: the variant is listed in `is_supported_op` in src/converters/cann.rs.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
OPERATORS_SRC = ROOT / "src/operators.rs"
ONNX_SRC = ROOT / "src/converters/onnx.rs"
COREML_SRC = ROOT / "src/converters/coreml_mlprogram.rs"
TRTX_SRC = ROOT / "src/converters/trtx.rs"
LITERT_SRC = ROOT / "src/converters/litert.rs"
CANN_SRC = ROOT / "src/converters/cann.rs"
OUTPUT = ROOT / "docs/development/backend-operator-support.md"

# Builder entry points that are not graph operations.
EXCLUDED_OPS = {"constant"}

# Operations rustnn keeps beyond the current WebNN specification.
EXTENSION_OPS = {
    "shape": "rustnn extension used by onnx2webnn exports",
    "squeeze": "removed from the WebNN spec (emulation appendix), kept for onnx2webnn",
    "unsqueeze": "removed from the WebNN spec (emulation appendix), kept for onnx2webnn",
}

OP_TYPE_MARKER = "pub fn op_type(&self) -> &'static str {"
TRTX_MARKER = "match op_type {"
COREML_MARKER = "let mil_type = match webnn_op.to_lowercase().as_str() {"
CANN_MARKER = "pub(crate) fn is_supported_op(op: &Operation) -> bool {"


@dataclass
class Backend:
    name: str
    source: pathlib.Path
    rule: str
    supported: set[str]  # normalized operation names


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def extract_brace_block_after_marker(text: str, marker: str) -> str:
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"Could not find marker: {marker}")
    brace_open = text.find("{", start)
    if brace_open < 0:
        raise RuntimeError(f"Could not find opening brace after marker: {marker}")

    depth = 0
    for i in range(brace_open, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_open + 1 : i]
    raise RuntimeError(f"Unbalanced braces while parsing marker: {marker}")


def parse_operation_names(operators_text: str) -> dict[str, str]:
    """Map `Operation` variant name -> WebNN operation name from `op_type()`."""
    block = extract_brace_block_after_marker(operators_text, OP_TYPE_MARKER)
    names: dict[str, str] = {}
    for match in re.finditer(
        r'Operation::([A-Za-z0-9]+)\s*\{\s*\.\.\s*\}\s*=>\s*"([A-Za-z0-9]+)"', block
    ):
        names[match.group(1)] = match.group(2)
    if not names:
        raise RuntimeError("No Operation variants found in op_type()")
    return names


def parse_variant_references(text: str, variants: dict[str, str]) -> set[str]:
    """Operations whose `Operation::<Variant>` is referenced anywhere in `text`."""
    found: set[str] = set()
    for match in re.finditer(r"Operation::([A-Z][A-Za-z0-9]*)", text):
        variant = match.group(1)
        if variant in variants:
            found.add(normalize(variants[variant]))
    return found


def parse_dispatch_table(text: str, marker: str) -> set[str]:
    """String keys of the match arms in the block after `marker` (e.g. `"add" =>`)."""
    block = extract_brace_block_after_marker(text, marker)
    keys = re.findall(r'"([A-Za-z0-9_]+)"\s*(?=\||=>)', block)
    return {normalize(key) for key in keys}


def parse_variant_list(text: str, marker: str, variants: dict[str, str]) -> set[str]:
    """Variants listed inside the block after `marker` (e.g. a `matches!` gate)."""
    block = extract_brace_block_after_marker(text, marker)
    return parse_variant_references(block, variants)


def extract_paren_block(text: str, start: int) -> str:
    """Text inside the parentheses that open at or after `start`."""
    paren_open = text.find("(", start)
    if paren_open < 0:
        raise RuntimeError("Could not find opening parenthesis")
    depth = 0
    for i in range(paren_open, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[paren_open + 1 : i]
    raise RuntimeError("Unbalanced parentheses")


def parse_op_name_comparisons(text: str, ident: str, known: set[str]) -> set[str]:
    """Operation names compared against the lower-cased op type variable `ident`.

    Covers `ident == "name"` and `matches!(ident.as_str(), "a" | "b")`. Only names that
    are known operations are returned, so unrelated string literals are ignored.
    """
    found: set[str] = set()
    for match in re.finditer(re.escape(ident) + r'\s*==\s*"([a-z0-9_]+)"', text):
        found.add(normalize(match.group(1)))
    for match in re.finditer(r"matches!\(\s*" + re.escape(ident) + r"(?:\.as_str\(\))?\s*,", text):
        block = extract_paren_block(text, match.start())
        for quoted in re.finditer(r'"([a-z0-9_]+)"', block):
            found.add(normalize(quoted.group(1)))
    return found & known


def parse_coreml_ops(text: str, variants: dict[str, str]) -> set[str]:
    """CoreML dispatches on the enum in some passes and on the lower-cased op name in others."""
    known = {normalize(name) for name in variants.values()}
    supported = parse_variant_references(text, variants)
    supported |= parse_dispatch_table(text, COREML_MARKER) & known
    supported |= parse_op_name_comparisons(text, "op_type_lower", known)
    return supported


def build_backends(variants: dict[str, str]) -> list[Backend]:
    return [
        Backend(
            name="ONNX Runtime",
            source=ONNX_SRC,
            rule="`Operation` variants referenced by the converter",
            supported=parse_variant_references(ONNX_SRC.read_text(encoding="utf-8"), variants),
        ),
        Backend(
            name="CoreML",
            source=COREML_SRC,
            rule="`Operation` variants referenced by the converter, plus names in its op-type dispatch",
            supported=parse_coreml_ops(COREML_SRC.read_text(encoding="utf-8"), variants),
        ),
        Backend(
            name="TensorRT",
            source=TRTX_SRC,
            rule="keys of the `match op_type` dispatch table",
            supported=parse_dispatch_table(TRTX_SRC.read_text(encoding="utf-8"), TRTX_MARKER),
        ),
        Backend(
            name="LiteRT",
            source=LITERT_SRC,
            rule="`Operation` variants referenced by the converter",
            supported=parse_variant_references(
                LITERT_SRC.read_text(encoding="utf-8"), variants
            ),
        ),
        ## CANN has changed marker format see "// ── Not supported ───────────────────────────────────────────── "
        # Backend(
            # name="CANN",
            # source=CANN_SRC,
            # rule="variants accepted by `is_supported_op`",
            # supported=parse_variant_list(
                # CANN_SRC.read_text(encoding="utf-8"), CANN_MARKER, variants
            # ),
        # ),
    ]


def render(variants: dict[str, str], backends: list[Backend]) -> str:
    operations = sorted(
        (name for name in variants.values() if name.lower() not in EXCLUDED_OPS),
        key=lambda s: s.lower(),
    )
    keys = {op: normalize(op) for op in operations}

    out: list[str] = []
    out.append("# Backend Operator Support Report")
    out.append("")
    out.append(
        "This file is generated from the converter sources by "
        "`scripts/generate_backend_operator_report.py`."
    )
    out.append(
        "Do not edit it manually. Run `make docs-backend-ops` after backend changes; "
        "CI fails on drift (`make docs-backend-ops-check`)."
    )
    out.append("")
    out.append(
        "Operation names are the WebNN builder names returned by `Operation::op_type()` "
        "in `src/operators.rs`. \"Supported\" means the converter emits a lowering for the "
        "operation. Data type restrictions, dynamic shape limits and known failing cases are "
        "tracked per backend in `tests/wpt_conformance/*_expected_failures.txt` and on the "
        "[WPT conformance dashboard](https://rustnn.github.io/rustnn/wpt-conformance/)."
    )
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append("| Backend | Converter source | Detection rule | Supported |")
    out.append("|---|---|---|---|")
    for b in backends:
        count = sum(1 for op in operations if keys[op] in b.supported)
        rel = b.source.relative_to(ROOT).as_posix()
        out.append(f"| {b.name} | `{rel}` | {b.rule} | {count} of {len(operations)} |")
    out.append("")
    out.append("## Operation matrix")
    out.append("")
    header = "| Operation | " + " | ".join(b.name for b in backends) + " |"
    out.append(header)
    out.append("|---|" + "|".join(":-:" for _ in backends) + "|")
    for op in operations:
        cells = ["yes" if keys[op] in b.supported else "-" for b in backends]
        out.append(f"| `{op}` | " + " | ".join(cells) + " |")
    out.append("")
    out.append("## Unsupported operations per backend")
    out.append("")
    for b in backends:
        missing = [op for op in operations if keys[op] not in b.supported]
        if missing:
            out.append(f"- {b.name}: " + ", ".join(f"`{op}`" for op in missing))
        else:
            out.append(f"- {b.name}: none")
    out.append("")
    out.append("## Notes")
    out.append("")
    for op in operations:
        note = EXTENSION_OPS.get(op.lower())
        if note:
            out.append(f"- `{op}`: {note}.")
    out.append("")
    return "\n".join(out) + "\n"


def build_report() -> str:
    variants = parse_operation_names(OPERATORS_SRC.read_text(encoding="utf-8"))
    return render(variants, build_backends(variants))


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

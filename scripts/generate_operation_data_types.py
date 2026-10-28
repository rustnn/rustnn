#!/usr/bin/env python3
"""Extract per-operand allowed data type rules from WebNN Bikeshed tensor tables.

Uses only the Python standard library. Unknown table syntax is an error rather
than silently producing incomplete validation metadata.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = re.compile(r"\{\{([^{}]+)\}\}")
TYPE = re.compile(r'\{\{MLOperandDataType/"([a-z0-9]+)"\}\}')


def operand_name(text: str) -> str:
    """Normalize a positional operand, option member, or output label."""
    text = text.strip().removesuffix("'s [=list/items=]")
    conditional_output = re.fullmatch(
        r"(\*outputs\[[0-9]+\]\*) if \{\{\w+/returnSequence\}\} is true", text
    )
    if conditional_output:
        text = conditional_output[1]
    reference = REFERENCE.fullmatch(text)
    name = reference[1].rsplit("/", 1)[-1] if reference else text.strip().strip("*")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:\[[0-9]+\])?", name):
        raise ValueError(f"Unknown operand syntax: {text!r}")
    return name


def parse_rule(text: str) -> tuple[str, tuple[str, ...]]:
    """Keep explicit lists, unrestricted types, and operand equality distinct."""
    text = text.strip()
    if text == "[=/any data type|any=]":
        return "Any", ()
    same = re.fullmatch(
        r"\[=/same type as\|same as=\]\s*(\{\{[^{}]+\}\}(?:'s \[=list/items=\])?)", text
    )
    if same:
        return "SameAs", (operand_name(same[1]),)
    types = tuple(TYPE.findall(text))
    if types and not TYPE.sub("", text).replace(",", "").strip():
        if len(set(types)) != len(types):
            raise ValueError(f"Duplicate data type: {text!r}")
        return "Types", types
    raise ValueError(f"Unknown allowed data types: {text!r}")


def parse_spec(source: str) -> dict[str, dict[str, tuple[str, tuple[str, ...]]]]:
    """Read allowed (not required) types from every tensor-limits table."""
    operations = {}
    for attributes, body in re.findall(r"<table\b([^>]*)>(.*?)</table>", source, re.S):
        if not re.search(r"\bid\s*=\s*['\"]?tensor-limits-", attributes):
            continue
        link_for = re.search(r'\blink-for="([^"]+)"', attributes)
        names = re.findall(r"MLGraphBuilder/(\w+)\(", link_for[1] if link_for else "")
        if not names:
            raise ValueError(f"No operation names in tensor table: {attributes}")
        # HTML allows an omitted </tr> before the next row (used by prelu).
        rows = re.findall(r"<tr\b[^>]*>(.*?)(?=</tr>|<tr\b|</table>|\Z)", body, re.S)
        if not rows:
            raise ValueError(f"Empty tensor table: {attributes}")
        headers = re.findall(r"<th\b[^>]*>(.*?)</th>", rows[0], re.S)
        if headers.count("[=/allowed data types=]") != 1 or headers[0] != "operand":
            raise ValueError(f"Unknown tensor table header: {attributes}")
        column = headers.index("[=/allowed data types=]")
        operands = {}
        for row in rows[1:]:
            cells = re.findall(r"<td\b([^>]*)>(.*?)</td>", row, re.S)
            if len(cells) != len(headers) or any(
                re.search(r"\b(?:rowspan|colspan)\b", attrs) for attrs, _ in cells
            ):
                raise ValueError(f"Unknown tensor table row: {row}")
            name = operand_name(cells[0][1])
            if name in operands:
                raise ValueError(f"Duplicate operand: {name}")
            operands[name] = parse_rule(cells[column][1])
        if not operands:
            raise ValueError(f"Empty tensor table: {attributes}")
        for name, (kind, values) in operands.items():
            if kind == "SameAs" and values[0] not in operands:
                raise ValueError(f"Unknown same-type operand for {name}: {values[0]}")
        for name in names:
            if name in operations:
                raise ValueError(f"Duplicate operation: {name}")
            operations[name] = operands.copy()
    if not operations:
        raise ValueError("No tensor-limits tables found")
    # Read declarations independently of table IDs and link-for attributes so
    # changes to either cannot silently remove operations from the metadata.
    idl_operations = set()
    for body in re.findall(r"\binterface\s+MLGraphBuilder\s*\{(.*?)\};", source, re.S):
        body = re.sub(r"/\*.*?\*/|//[^\n]*", "", body, flags=re.S)
        idl_operations.update(
            re.findall(
                r"\b(?:MLOperand|sequence\s*<\s*MLOperand\s*>)\s+(\w+)\s*\(", body
            )
        )
    missing = idl_operations - operations.keys()
    unexpected = operations.keys() - idl_operations
    if missing != {"input", "constant"} or unexpected:
        raise ValueError(
            "Tensor-limits operations do not match MLGraphBuilder IDL "
            "(expected only input and constant without tables): "
            f"IDL operations without tables={sorted(missing)}, "
            f"table operations absent from IDL={sorted(unexpected)}"
        )
    return operations


def operand_data_type_variants() -> dict[str, str]:
    """Map spec spellings to variants implemented by MLOperandDataType."""
    source = (ROOT / "src/operator_enums.rs").read_text(encoding="utf-8")
    enum = re.search(r"pub enum MLOperandDataType\s*\{(.*?)\n\}", source, re.S)
    if not enum:
        raise ValueError("Cannot find MLOperandDataType in src/operator_enums.rs")
    variants = re.findall(r"^\s*([A-Za-z][A-Za-z0-9]*)\s*,", enum[1], re.M)
    if not variants:
        raise ValueError("MLOperandDataType has no variants")

    EXPECTED_RUSTNN_DATA_TYPES = 10
    assert len(variants) == EXPECTED_RUSTNN_DATA_TYPES, (
        f"Detected number of variants of MLOperandDataType is {len(variants)}. In the past it was {EXPECTED_RUSTNN_DATA_TYPES=}."
        "Please review whether this change is expected and then update EXPECTED_RUSTNN_DATA_TYPES"
    )
    return {variant.lower(): variant for variant in variants}


def name_variant(name: str) -> str:
    """Keep camelCase acronyms and turn indexed outputs into unit variants."""
    return name[0].upper() + name[1:].replace("[", "").replace("]", "")


def generate_name_enum(enum_name: str, description: str, names: set[str]) -> list[str]:
    """Emit identifiers with exact, reversible spec spellings via strum."""
    variants = [name_variant(name) for name in sorted(names)]
    if len(variants) != len(set(variants)):
        raise ValueError(f"Conflicting Rust variant names in {enum_name}")
    lines = [
        f"/// {description}",
        "///",
        "/// Strum provides exact spec spelling through `FromStr`, `Display`,",
        "/// `AsRef<str>`, and conversion into `&'static str`.",
        "#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumString, Display, AsRefStr, IntoStaticStr, EnumIter)]",
        f"pub enum {enum_name} {{",
    ]
    for name in sorted(names):
        lines.extend(
            [
                f"    /// `{name}`.",
                f"    #[strum(serialize = {json.dumps(name)})]",
                f"    {name_variant(name)},",
            ]
        )
    return lines + ["}", ""]


def generate(source: str) -> str:
    """Render deterministic Rust metadata, including the input content hash."""
    operations = parse_spec(source)
    variants = operand_data_type_variants()
    lines = [
        "//! WebNN per-operand allowed data types, generated by `make generate-operation-data-types`.",
        "//!",
        "//! Source: <https://github.com/webmachinelearning/webnn/blob/main/index.bs>.",
        "//! These are specification rules, not backend capabilities or runtime validation.",
        "//! Explicit lists contain only types represented by [`MLOperandDataType`].",
        "//! Spec types absent from that enum are recorded in comments beside each list.",
        f"// Source SHA-256: {hashlib.sha256(source.encode()).hexdigest()}",
        "// Do not edit manually.",
        "",
        "use crate::operator_enums::MLOperandDataType;",
        "use strum::{AsRefStr, Display, EnumIter, EnumString, IntoStaticStr};",
        "",
    ]
    lines.extend(
        [
            "/// Every data type represented by rustnn, used for unrestricted spec rules.",
            "pub const ALL_DATA_TYPES: &[MLOperandDataType] = &[",
            *[f"    MLOperandDataType::{variant}," for variant in variants.values()],
            "];",
            "",
        ]
    )
    lines.extend(
        generate_name_enum(
            "OperationName",
            "WebNN operation identifier in the spec data type tables.",
            set(operations),
        )
    )
    lines.extend(
        generate_name_enum(
            "OperandName",
            "Operand identifier shared across the spec data type tables.",
            {operand for operands in operations.values() for operand in operands},
        )
    )
    lines.extend(
        [
            "/// The specification's allowed data types for one operand.",
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]",
            "pub enum AllowedDataTypes {",
            "    /// Any MLOperandDataType, including future additions to the specification.",
            "    Any,",
            "    /// An explicit spec type list restricted to types represented by rustnn.",
            "    Types(&'static [MLOperandDataType]),",
            "    /// Must have the same type as the named operand of this operation.",
            "    SameAs(OperandName),",
            "}",
            "",
            "/// Allowed types for a positional input, option member, or output.",
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]",
            "pub struct OperandDataTypes {",
            "    /// WebNN operation identifier.",
            "    pub operation: OperationName,",
            "    /// Operand name; option members omit the dictionary prefix.",
            "    /// Sequence inputs apply to every item; indexed outputs retain their index.",
            "    /// Optional operands and outputs apply when present.",
            "    pub operand: OperandName,",
            "    /// Allowed type rule, distinct from the specification's required types.",
            "    pub allowed: AllowedDataTypes,",
            "}",
            "",
            "/// Per-operand rules sorted by operation and operand name.",
            "pub const OPERATION_DATA_TYPES: &[OperandDataTypes] = &[",
        ]
    )
    for operation, operands in sorted(operations.items()):
        for operand, (kind, values) in sorted(operands.items()):
            rule = f"AllowedDataTypes::{kind}"
            if kind == "SameAs":
                rule += f"(OperandName::{name_variant(values[0])})"
            elif kind == "Types":
                rule += (
                    "(&["
                    + ", ".join(
                        f"MLOperandDataType::{variants[value]}"
                        for value in values
                        if value in variants
                    )
                    + "])"
                )
            lines.extend(
                [
                    "    OperandDataTypes {",
                    f"        operation: OperationName::{name_variant(operation)},",
                    f"        operand: OperandName::{name_variant(operand)},",
                ]
            )
            omitted = (
                [value for value in values if value not in variants]
                if kind == "Types"
                else []
            )
            if omitted:
                lines.append(
                    "        // Spec types absent from MLOperandDataType: "
                    + ", ".join(omitted)
                )
            lines.extend(
                [
                    f"        allowed: {rule},",
                    "    },",
                ]
            )
    lines.append("];\n")
    result = subprocess.run(
        ["rustfmt", "--edition", "2024", "--emit", "stdout"],
        input="\n".join(lines),
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "target/webnn-index.bs")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "src/operation_data_types.rs"
    )
    parser.add_argument(
        "--check", action="store_true", help="Fail if the generated file differs"
    )
    args = parser.parse_args()
    try:
        result = generate(args.input.read_text(encoding="utf-8"))
        if args.check:
            if (
                not args.output.exists()
                or args.output.read_text(encoding="utf-8") != result
            ):
                print(
                    f"{args.output} is out of date; run make generate-operation-data-types",
                    file=sys.stderr,
                )
                return 1
        else:
            args.output.write_text(result, encoding="utf-8", newline="\n")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(
            f"{error}\nFetch the specification with make fetch-webnn-spec.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

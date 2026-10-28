#!/usr/bin/env python3
"""Offline regressions for specification data type extraction."""

from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_operation_data_types import generate, generate_name_enum, operand_name, operand_data_type_variants, parse_spec


def table(rows, operations="MLGraphBuilder/add(a, b, options), MLGraphBuilder/sub(a, b, options)"):
    names = re.findall(r"MLGraphBuilder/(\w+)\(", operations)
    declarations = "\n".join(f"MLOperand {name}();" for name in names)
    return f'''enum MLOperandDataType {{
    "float32", "float16", "int32", "uint32", "int64", "uint64", "int8", "uint8",
    "bfloat16", "float8e4m3"
    }};
    interface MLGraphBuilder {{
    MLOperand input();
    MLOperand constant();
    MLOperand constant(MLOperandDataType dataType, MLNumber value);
    }};
    partial interface MLGraphBuilder {{
    {declarations}
    }};
    <table id=tensor-limits-test class='data' link-for="{operations}">
    <caption>Tensor limits</caption>
    <thead><tr><th>operand</th><th>[=/allowed data types=]</th>
    <th>[=/required data types=]</th><th>[=/allowed ranks=]</th></tr></thead>
    {rows}</table>'''


def row(operand, allowed):
    # Deliberately different required types: these must never become allowed types.
    return f'''<tr><td>{operand}</td><td>{allowed}</td>
    <td>{{{{MLOperandDataType/"float32"}}}}</td><td>1</td></tr>'''


class OperationDataTypesTests(unittest.TestCase):
    def test_standard_types_come_from_spec_idl(self):
        source = table(row("{{a}}", "[=/any data type|any=]"))
        result = generate(source)
        standard = result.split("pub const ALL_SPEC_DATA_TYPES:", 1)[1].split("];", 1)[0]
        self.assertEqual(re.findall(r"MLOperandDataType::(\w+)", standard), [
            "Float32", "Float16", "Int32", "Uint32", "Int64", "Uint64", "Int8", "Uint8",
        ])
        self.assertIn("// Spec type absent from MLOperandDataType: bfloat16", standard)
        self.assertIn("MLOperandDataType::Int4", result.split("pub const ALL_SPEC_DATA_TYPES:")[0])
        changed = generate(source.replace('"float32", "float16"', '"int4", "float16"'))
        standard = changed.split("pub const ALL_SPEC_DATA_TYPES:", 1)[1].split("];", 1)[0]
        self.assertIn("MLOperandDataType::Int4", standard)
        self.assertNotIn("MLOperandDataType::Float32", standard)

    def test_missing_or_invalid_spec_enum_rejected(self):
        source = table(row("{{a}}", "[=/any data type|any=]"))
        for bad in (
            source.replace("enum MLOperandDataType", "enum Renamed"),
            re.sub(r"enum MLOperandDataType \{.*?\};", "enum MLOperandDataType {};", source, flags=re.S),
            source.replace('"float32", "float16"', '"float32", "float32"'),
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "MLOperandDataType"):
                generate(bad)

    def test_idl_coverage_rejects_missing_or_unexpected_operations(self):
        source = table(row("{{a}}", "[=/any data type|any=]"))
        for bad in (
            source.replace("MLGraphBuilder/sub(a, b, options)", ""),
            source + table(row("{{a}}", "[=/any data type|any=]"),
                           operations="MLGraphBuilder/mul(a, b, options)")
                     .replace("id=tensor-limits-test", "id=renamed-limits"),
            source.replace("MLOperand sub();", ""),
            source.replace("MLOperand input();", ""),
            source.replace("MLOperand constant", "void constant"),
            re.sub(r"(?:partial )?interface MLGraphBuilder \{.*?\};", "", source, flags=re.S),
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "do not match MLGraphBuilder IDL"):
                parse_spec(bad)

    def test_idl_sequence_returns_and_unrelated_interfaces(self):
        source = table(row("{{a}}", "[=/any data type|any=]"))
        source = source.replace("MLOperand sub();", "sequence < MLOperand > sub();")
        source += '''interface Other { MLOperand unrelated(); };
        partial interface MLGraphBuilder {
          // MLOperand commentedOut();
          /* sequence<MLOperand> alsoCommentedOut(); */
          Promise<MLGraph> build();
        };'''
        self.assertEqual(set(parse_spec(source)), {"add", "sub"})

    def test_generated_identifiers_preserve_spec_spelling(self):
        source = table(
            row("{{input}}", "[=/any data type|any=]") +
            row("*outputs[0]*", "[=/same type as|same as=] {{input}}"),
            operations="MLGraphBuilder/isNaN(input, options), MLGraphBuilder/gatherND(input, indices, options)",
        )
        result = generate(source)
        self.assertIn('#[strum(serialize = "isNaN")]\n    IsNaN,', result)
        self.assertIn('#[strum(serialize = "gatherND")]\n    GatherND,', result)
        self.assertIn('#[strum(serialize = "outputs[0]")]\n    Outputs0,', result)
        self.assertIn("operation: OperationName::IsNaN,", result)
        self.assertIn("operand: OperandName::Outputs0,", result)
        self.assertIn("AllowedDataTypes::SameAs(OperandName::Input)", result)

    def test_identifier_collisions_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting Rust variant names"):
            generate_name_enum("OperandName", "Test operands.", {"outputs0", "outputs[0]"})

    def test_enum_variants(self):
        variants = operand_data_type_variants()
        self.assertEqual(variants["float32"], "Float32")
        self.assertEqual(variants["uint4"], "Uint4")
        self.assertEqual(variants["int64"], "Int64")

    def test_generation_uses_enum_variants_and_marks_unrepresented_types(self):
        source = table(row("{{a}}", '{{MLOperandDataType/"float16"}}, {{MLOperandDataType/"futuretype"}}'))
        result = generate(source)
        self.assertIn("Types(&'static [MLOperandDataType])", result)
        self.assertIn("AllowedDataTypes::Types(&[MLOperandDataType::Float16])", result)
        self.assertIn("// Spec types absent from MLOperandDataType: futuretype", result)
        self.assertNotIn('"float16"', result)

    def test_unrepresented_type_only_list_is_empty(self):
        result = generate(table(row("{{a}}", '{{MLOperandDataType/"futuretype"}}')))
        self.assertIn("AllowedDataTypes::Types(&[])", result)
        self.assertIn("// Spec types absent from MLOperandDataType: futuretype", result)

    def test_shared_table_and_allowed_column(self):
        source = table(row("{{a}}", '{{MLOperandDataType/"int64"}}, {{MLOperandDataType/"bfloat16"}}'))
        parsed = parse_spec(source)
        self.assertEqual(set(parsed), {"add", "sub"})
        self.assertEqual(parsed["add"]["a"], ("Types", ("int64", "bfloat16")))
        self.assertEqual(parsed["add"], parsed["sub"])

    def test_any_and_same_as_option_and_output(self):
        source = table(row("{{a}}", "[=/any data type|any=]") +
                       row("{{MLTestOptions/bias}}", "[=/same type as|same as=] {{a}}") +
                       row("*output*", "[=/same type as|same as=] {{MLTestOptions/bias}}"))
        parsed = parse_spec(source)["add"]
        self.assertEqual(parsed["a"], ("Any", ()))
        self.assertEqual(parsed["bias"], ("SameAs", ("a",)))
        self.assertEqual(parsed["output"], ("SameAs", ("bias",)))

    def test_sequences_and_conditional_indexed_outputs(self):
        source = table(row("{{inputs}}'s [=list/items=]", "[=/any data type|any=]") +
                       row("*outputs[1]* if {{MLGruOptions/returnSequence}} is true",
                           "[=/same type as|same as=] {{inputs}}'s [=list/items=]"))
        self.assertEqual(parse_spec(source)["add"]["outputs[1]"], ("SameAs", ("inputs",)))
        self.assertEqual(operand_name("*outputs*"), "outputs")

    def test_omitted_row_end(self):
        source = table(row("{{a}}", "[=/any data type|any=]").removesuffix("</tr>") +
                       row("{{b}}", "[=/same type as|same as=] {{a}}"))
        self.assertEqual(set(parse_spec(source)["add"]), {"a", "b"})

    def test_unrelated_tables_ignored(self):
        source = '<table id=unrelated><tr><td>anything</td></tr></table>'
        with self.assertRaisesRegex(ValueError, "No tensor-limits"):
            parse_spec(source)

    def test_unknown_rules_rejected(self):
        for rule in ("float32", "[=/any data type|any=] or something", '{{MLOperandDataType/"int8"}} or future'):
            with self.subTest(rule=rule), self.assertRaisesRegex(ValueError, "Unknown allowed"):
                parse_spec(table(row("{{a}}", rule)))

    def test_unknown_reference_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown same-type operand"):
            parse_spec(table(row("{{a}}", "[=/same type as|same as=] {{missing}}")))

    def test_duplicate_operation_and_operand_rejected(self):
        source = table(row("{{a}}", "[=/any data type|any=]"))
        with self.assertRaisesRegex(ValueError, "Duplicate operation"):
            parse_spec(source + source)
        with self.assertRaisesRegex(ValueError, "Duplicate operand"):
            parse_spec(table(row("{{a}}", "[=/any data type|any=]") * 2))

    def test_malformed_tables_rejected(self):
        source = table(row("{{a}}", "[=/any data type|any=]"))
        for bad in (source.replace('link-for=', 'other='),
                    source.replace('[=/allowed data types=]', 'unknown'),
                    source.replace('<td>', '<td colspan=2>', 1),
                    source.replace('<td>1</td>', '')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_spec(bad)

    def test_empty_tables_rejected(self):
        for source in (table(""), '<table id=tensor-limits-empty link-for="MLGraphBuilder/add()"></table>'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                parse_spec(source)

    def test_cli_generation_and_drift(self):
        source = table(row("{{a}}", "[=/any data type|any=]"))
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "index.bs"
            output_path = Path(directory) / "generated" / "operation_data_types.rs"
            input_path.write_text(source)
            command = [sys.executable, str(Path(__file__).with_name("generate_operation_data_types.py")),
                       "--input", str(input_path), "--output", str(output_path)]
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(output_path.read_text(), generate(source))
            subprocess.run(command + ["--check"], check=True, capture_output=True)
            output_path.write_text("stale\n")
            result = subprocess.run(command + ["--check"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("out of date", result.stderr)
            self.assertEqual(output_path.read_text(), "stale\n")


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.generate_backend_operator_report import (  # noqa: E402
    normalize,
    parse_dispatch_table,
    parse_op_name_comparisons,
    parse_operation_names,
    parse_variant_list,
    parse_variant_references,
)

OPERATORS_SNIPPET = """
impl Operation {
    pub fn op_type(&self) -> &'static str {
        match self {
            Operation::Add { .. } => "add",
            Operation::ConvTranspose2d { .. } => "convTranspose2d",
            Operation::IsNaN { .. } => "isNaN",
            Operation::Constant { .. } => "constant",
        }
    }
}
"""


class BackendOperatorReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.variants = parse_operation_names(OPERATORS_SNIPPET)

    def test_parse_operation_names(self) -> None:
        self.assertEqual(self.variants["ConvTranspose2d"], "convTranspose2d")
        self.assertEqual(self.variants["IsNaN"], "isNaN")

    def test_variant_references_ignore_associated_functions(self) -> None:
        text = """
        let op = Operation::from_legacy(a, b);
        match op {
            Operation::Add { a, b, .. } => add(a, b),
            Operation::IsNaN { input, .. } => is_nan(input),
        }
        """
        self.assertEqual(
            parse_variant_references(text, self.variants), {"add", "isnan"}
        )

    def test_dispatch_table_handles_or_patterns(self) -> None:
        text = """
        fn add_single_operation(&self) {
            match op_type {
                "add" => foo()?,
                "isNaN" | "isInfinite" => bar()?,
                _ => { return Err(GraphError::ConversionFailed { format: "x".into(), reason: "y".into() }); }
            }
        }
        """
        parsed = parse_dispatch_table(text, "match op_type {")
        self.assertEqual(parsed, {"add", "isnan", "isinfinite"})

    def test_op_name_comparisons_filter_unknown_strings(self) -> None:
        text = """
        if matches!(op_type_lower.as_str(), "equal" | "notequal" | "isnan") { cast_bool(); }
        if op_type_lower == "convtranspose2d" { fix_layout(); }
        let reason = "not supported";
        """
        known = {normalize(name) for name in self.variants.values()}
        parsed = parse_op_name_comparisons(text, "op_type_lower", known)
        self.assertEqual(parsed, {"convtranspose2d", "isnan"})

    def test_variant_list_reads_only_the_gate(self) -> None:
        text = """
        pub(crate) fn is_supported_op(op: &Operation) -> bool {
            matches!(op, Operation::Add { .. } | Operation::ConvTranspose2d { .. })
        }
        fn elsewhere(op: &Operation) { let _ = Operation::IsNaN { input: 0, options: None, outputs: vec![] }; }
        """
        parsed = parse_variant_list(
            text, "pub(crate) fn is_supported_op(op: &Operation) -> bool {", self.variants
        )
        self.assertEqual(parsed, {"add", "convtranspose2d"})


if __name__ == "__main__":
    unittest.main()

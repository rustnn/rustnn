#!/usr/bin/env python3
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.generate_backend_operator_report import (  # noqa: E402
    canonicalize_ops,
    parse_coreml_ops,
    parse_trtx_ops,
)


class BackendOperatorReportTests(unittest.TestCase):
    def test_canonicalize_prefers_camel_case(self) -> None:
        ops = canonicalize_ops(["convtranspose2d", "convTranspose2d"])
        self.assertIn("convTranspose2d", ops)
        self.assertNotIn("convtranspose2d", ops)

    def test_parse_coreml_extracts_known_ops(self) -> None:
        text = """
        fn get_mil_op_type(&self, webnn_op: &str) -> Result<&'static str, GraphError> {
            let mil_type = match webnn_op.to_lowercase().as_str() {
                "add" => mil_ops::ADD,
                "linear" => mil_ops::MUL,
                _ => { return Err(GraphError::ConversionFailed { format: String::new(), reason: String::new()}); }
            };
            Ok(mil_type)
        }
        """
        parsed = parse_coreml_ops(text)
        self.assertIn("add", parsed)
        self.assertIn("linear", parsed)

    def test_parse_trtx_extracts_known_ops(self) -> None:
        text = """
        fn add_single_operation(&self) {
            match op_type {
                "add" => foo()?,
                "isNaN" => bar()?,
                _ => { return Err(GraphError::ConversionFailed { format: String::new(), reason: String::new()}); }
            }
        }
        """
        parsed = parse_trtx_ops(text)
        self.assertIn("add", parsed)
        self.assertIn("isNaN", parsed)


if __name__ == "__main__":
    unittest.main()


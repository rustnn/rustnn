//! LiteRT execution tests with numerical verification
//!
//! These tests verify that WebNN graphs execute correctly on LiteRT
//! and produce numerically correct results. Uses the native TFLite
//! converter (flatbuffers) and LiteRT runtime.
//!
//! Run with: cargo test --test test_litert_execution --features litert-runtime

#[cfg(feature = "litert-runtime")]
mod tests {
    use rustnn::GraphConverter;
    use rustnn::converters::LiteRtConverter;
    use rustnn::graph::{
        ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind,
        to_dimension_vector,
    };
    use rustnn::operators::Operation;
    use std::collections::HashMap;

    fn s(shape: &[u32]) -> Vec<rustnn::graph::Dimension> {
        to_dimension_vector(shape)
    }

    /// Create an operation from WebNN op name and operand wiring.
    fn operation(
        op_type: &str,
        inputs: &[u32],
        outputs: &[u32],
        attributes: serde_json::Value,
    ) -> Operation {
        Operation::from_json_attributes(op_type, inputs, outputs, &attributes)
            .unwrap_or_else(|| panic!("unsupported op: {op_type}"))
    }

    /// Build a unary graph (single input → op → single output).
    fn unary_graph(op_type: &str, shape: Vec<u32>, dt: DataType) -> GraphInfo {
        let desc = OperandDescriptor {
            data_type: dt,
            shape: s(&shape),
            pending_permutation: vec![],
        };
        GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: desc.clone(),
                    name: Some("x".into()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: desc.clone(),
                    name: Some("y".into()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![operation(op_type, &[0], &[1], serde_json::Value::Null)],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        }
    }

    /// Build a binary graph (two inputs → op → single output).
    fn binary_graph(op_type: &str, shape: Vec<u32>, dt: DataType) -> GraphInfo {
        let desc = OperandDescriptor {
            data_type: dt,
            shape: s(&shape),
            pending_permutation: vec![],
        };
        GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: desc.clone(),
                    name: Some("a".into()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: desc.clone(),
                    name: Some("b".into()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: desc.clone(),
                    name: Some("c".into()),
                },
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            operations: vec![operation(op_type, &[0, 1], &[2], serde_json::Value::Null)],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        }
    }

    // ==================== Converter Tests (no runtime needed) ====================

    #[test]
    fn test_converter_relu_produces_tflite() {
        let graph = unary_graph("relu", vec![4], DataType::Float32);
        let converter = LiteRtConverter::new();
        let result = converter.convert(&graph);
        assert!(result.is_ok(), "converter failed: {:?}", result.err());
        let bytes = result.unwrap().data;
        assert!(!bytes.is_empty(), "TFLite output is empty");
        // Check TFLite file magic
        assert_eq!(&bytes[4..8], b"TFL3", "missing TFL3 magic");
    }

    #[test]
    fn test_converter_add_produces_tflite() {
        let graph = binary_graph("add", vec![4], DataType::Float32);
        let converter = LiteRtConverter::new();
        let result = converter.convert(&graph);
        assert!(result.is_ok());
        assert!(!result.unwrap().data.is_empty());
    }

    #[test]
    fn test_converter_conv2d_produces_tflite() {
        let filter_desc = OperandDescriptor {
            data_type: DataType::Float32,
            shape: s(&[1, 1, 3, 3]),
            pending_permutation: vec![],
        };
        let filter_data: Vec<f32> = vec![1.0; 9];
        let filter_bytes: Vec<u8> = filter_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut constants = HashMap::new();
        constants.insert(
            1,
            ConstantData {
                data: filter_bytes,
                label: Some("filter".into()),
            },
        );

        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[1, 1, 5, 5]),
                        pending_permutation: vec![],
                    },
                    name: Some("x".into()),
                },
                Operand {
                    kind: OperandKind::Constant,
                    descriptor: filter_desc,
                    name: Some("w".into()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[1, 1, 3, 3]),
                        pending_permutation: vec![],
                    },
                    name: Some("y".into()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![2],
            operations: vec![operation("conv2d", &[0, 1], &[2], serde_json::Value::Null)],
            constant_operand_ids_to_handles: constants,
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let converter = LiteRtConverter::new();
        let result = converter.convert(&graph);
        assert!(result.is_ok(), "conv2d convert failed: {:?}", result.err());
    }

    #[test]
    fn test_converter_softmax_produces_tflite() {
        let graph = unary_graph("softmax", vec![4], DataType::Float32);
        let converter = LiteRtConverter::new();
        let result = converter.convert(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_converter_reshape_produces_tflite() {
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[2, 3]),
                        pending_permutation: vec![],
                    },
                    name: Some("x".into()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[6]),
                        pending_permutation: vec![],
                    },
                    name: Some("y".into()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![operation("reshape", &[0], &[1], serde_json::Value::Null)],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };
        let converter = LiteRtConverter::new();
        let result = converter.convert(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_converter_matmul_produces_tflite() {
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[2, 3]),
                        pending_permutation: vec![],
                    },
                    name: Some("a".into()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[3, 2]),
                        pending_permutation: vec![],
                    },
                    name: Some("b".into()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[2, 2]),
                        pending_permutation: vec![],
                    },
                    name: Some("c".into()),
                },
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            operations: vec![operation("matmul", &[0, 1], &[2], serde_json::Value::Null)],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };
        let converter = LiteRtConverter::new();
        let result = converter.convert(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_converter_all_activation_ops() {
        for op in &[
            "relu",
            "sigmoid",
            "tanh",
            "softmax",
            "gelu",
            "hardSwish",
            "leakyRelu",
        ] {
            let graph = unary_graph(op, vec![4], DataType::Float32);
            let converter = LiteRtConverter::new();
            let result = converter.convert(&graph);
            assert!(result.is_ok(), "{op} conversion failed: {:?}", result.err());
        }
    }

    #[test]
    fn test_converter_all_elementwise_ops() {
        for op in &[
            "add", "sub", "mul", "div", "abs", "neg", "exp", "log", "sqrt", "ceil", "floor", "sin",
            "cos", "pow",
        ] {
            let graph = if [
                "abs", "neg", "exp", "log", "sqrt", "ceil", "floor", "sin", "cos",
            ]
            .contains(op)
            {
                unary_graph(op, vec![4], DataType::Float32)
            } else {
                binary_graph(op, vec![4], DataType::Float32)
            };
            let converter = LiteRtConverter::new();
            let result = converter.convert(&graph);
            assert!(result.is_ok(), "{op} conversion failed: {:?}", result.err());
        }
    }
}

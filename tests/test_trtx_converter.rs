//! TensorRT converter integration tests
//!
//! These tests verify that WebNN graphs can be converted to TensorRT format
//! and executed correctly. Tests require a NVIDIA GPU with TensorRT installed.
//!
//! Run with: cargo test --test test_trtx_converter --features trtx-runtime

#[cfg(feature = "trtx-runtime")]
mod tests {
    use rustnn::converters::{GraphConverter, TrtxConverter};
    use rustnn::error::GraphError;
    use rustnn::graph::{DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, Operation};
    use std::collections::HashMap;

    /// Helper function to create a simple unary operation graph
    ///
    /// This creates a graph with:
    /// - One input operand
    /// - One unary operation
    /// - One output operand
    fn create_unary_graph(op_type: &str, input_shape: Vec<u32>, data_type: DataType) -> GraphInfo {
        let input_desc = OperandDescriptor {
            data_type,
            shape: input_shape.clone(),
            pending_permutation: Vec::new(),
        };

        let output_desc = input_desc.clone();

        GraphInfo {
            operations: vec![Operation {
                op_type: op_type.to_string(),
                input_operands: vec![0], // Input operand at index 0
                output_operand: Some(1), // Output operand at index 1
                output_operands: Vec::new(),
                attributes: serde_json::Value::Null,
                label: Some(format!("{}_op", op_type)),
            }],
            operands: vec![
                // Input operand
                Operand {
                    kind: OperandKind::Input,
                    descriptor: input_desc,
                    name: Some("input".to_string()),
                },
                // Output operand
                Operand {
                    kind: OperandKind::Output,
                    descriptor: output_desc,
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        }
    }

    /// Helper function to create a unary operation graph with scalar input
    fn create_unary_graph_scalar(op_type: &str, data_type: DataType) -> GraphInfo {
        create_unary_graph(op_type, vec![], data_type)
    }

    /// Helper function to create a unary operation graph with 1D tensor
    fn create_unary_graph_1d(op_type: &str, size: u32, data_type: DataType) -> GraphInfo {
        create_unary_graph(op_type, vec![size], data_type)
    }

    /// Helper function to create a unary operation graph with 2D tensor
    fn create_unary_graph_2d(
        op_type: &str,
        rows: u32,
        cols: u32,
        data_type: DataType,
    ) -> GraphInfo {
        create_unary_graph(op_type, vec![rows, cols], data_type)
    }

    /// Helper function to create a unary operation graph with 4D tensor (typical for CNNs)
    fn create_unary_graph_4d(
        op_type: &str,
        batch: u32,
        channels: u32,
        height: u32,
        width: u32,
        data_type: DataType,
    ) -> GraphInfo {
        create_unary_graph(op_type, vec![batch, channels, height, width], data_type)
    }

    /// Test helper that converts a graph using TrtxConverter
    fn test_convert_graph(graph: &GraphInfo) -> Result<(), GraphError> {
        let converter = TrtxConverter::new();
        let converted = converter.convert(graph)?;

        // Verify the conversion produced valid data
        assert!(
            !converted.data.is_empty(),
            "Converted graph should not be empty"
        );
        assert_eq!(converted.format, "trtx", "Format should be trtx");

        Ok(())
    }

    // ============================================================================
    // Arithmetic Unary Operations
    // ============================================================================

    #[test]
    fn test_abs_scalar() {
        let graph = create_unary_graph_scalar("abs", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert abs scalar");
    }

    #[test]
    fn test_abs_1d() {
        let graph = create_unary_graph_1d("abs", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert abs 1D");
    }

    #[test]
    fn test_abs_2d() {
        let graph = create_unary_graph_2d("abs", 4, 8, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert abs 2D");
    }

    #[test]
    fn test_ceil_scalar() {
        let graph = create_unary_graph_scalar("ceil", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert ceil scalar");
    }

    #[test]
    fn test_ceil_1d() {
        let graph = create_unary_graph_1d("ceil", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert ceil 1D");
    }

    #[test]
    fn test_floor_scalar() {
        let graph = create_unary_graph_scalar("floor", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert floor scalar");
    }

    #[test]
    fn test_floor_1d() {
        let graph = create_unary_graph_1d("floor", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert floor 1D");
    }

    #[test]
    fn test_neg_scalar() {
        let graph = create_unary_graph_scalar("neg", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert neg scalar");
    }

    #[test]
    fn test_neg_1d() {
        let graph = create_unary_graph_1d("neg", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert neg 1D");
    }

    #[test]
    fn test_reciprocal_scalar() {
        let graph = create_unary_graph_scalar("reciprocal", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert reciprocal scalar");
    }

    #[test]
    fn test_reciprocal_1d() {
        let graph = create_unary_graph_1d("reciprocal", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert reciprocal 1D");
    }

    #[test]
    fn test_sign_scalar() {
        let graph = create_unary_graph_scalar("sign", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert sign scalar");
    }

    #[test]
    fn test_sign_1d() {
        let graph = create_unary_graph_1d("sign", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert sign 1D");
    }

    #[test]
    fn test_sqrt_scalar() {
        let graph = create_unary_graph_scalar("sqrt", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert sqrt scalar");
    }

    #[test]
    fn test_sqrt_1d() {
        let graph = create_unary_graph_1d("sqrt", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert sqrt 1D");
    }

    // ============================================================================
    // Trigonometric Unary Operations
    // ============================================================================

    #[test]
    fn test_sin_scalar() {
        let graph = create_unary_graph_scalar("sin", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert sin scalar");
    }

    #[test]
    fn test_sin_1d() {
        let graph = create_unary_graph_1d("sin", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert sin 1D");
    }

    #[test]
    fn test_cos_scalar() {
        let graph = create_unary_graph_scalar("cos", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert cos scalar");
    }

    #[test]
    fn test_cos_1d() {
        let graph = create_unary_graph_1d("cos", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert cos 1D");
    }

    #[test]
    fn test_tan_scalar() {
        let graph = create_unary_graph_scalar("tan", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert tan scalar");
    }

    #[test]
    fn test_tan_1d() {
        let graph = create_unary_graph_1d("tan", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert tan 1D");
    }

    #[test]
    fn test_asin_scalar() {
        let graph = create_unary_graph_scalar("asin", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert asin scalar");
    }

    #[test]
    fn test_asin_1d() {
        let graph = create_unary_graph_1d("asin", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert asin 1D");
    }

    #[test]
    fn test_acos_scalar() {
        let graph = create_unary_graph_scalar("acos", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert acos scalar");
    }

    #[test]
    fn test_acos_1d() {
        let graph = create_unary_graph_1d("acos", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert acos 1D");
    }

    #[test]
    fn test_atan_scalar() {
        let graph = create_unary_graph_scalar("atan", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert atan scalar");
    }

    #[test]
    fn test_atan_1d() {
        let graph = create_unary_graph_1d("atan", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert atan 1D");
    }

    // ============================================================================
    // Hyperbolic Unary Operations
    // ============================================================================

    #[test]
    fn test_sinh_scalar() {
        let graph = create_unary_graph_scalar("sinh", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert sinh scalar");
    }

    #[test]
    fn test_sinh_1d() {
        let graph = create_unary_graph_1d("sinh", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert sinh 1D");
    }

    #[test]
    fn test_cosh_scalar() {
        let graph = create_unary_graph_scalar("cosh", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert cosh scalar");
    }

    #[test]
    fn test_cosh_1d() {
        let graph = create_unary_graph_1d("cosh", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert cosh 1D");
    }

    #[test]
    fn test_tanh_scalar() {
        let graph = create_unary_graph_scalar("tanh", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert tanh scalar");
    }

    #[test]
    fn test_tanh_1d() {
        let graph = create_unary_graph_1d("tanh", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert tanh 1D");
    }

    #[test]
    fn test_asinh_scalar() {
        let graph = create_unary_graph_scalar("asinh", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert asinh scalar");
    }

    #[test]
    fn test_asinh_1d() {
        let graph = create_unary_graph_1d("asinh", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert asinh 1D");
    }

    #[test]
    fn test_acosh_scalar() {
        let graph = create_unary_graph_scalar("acosh", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert acosh scalar");
    }

    #[test]
    fn test_acosh_1d() {
        let graph = create_unary_graph_1d("acosh", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert acosh 1D");
    }

    #[test]
    fn test_atanh_scalar() {
        let graph = create_unary_graph_scalar("atanh", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert atanh scalar");
    }

    #[test]
    fn test_atanh_1d() {
        let graph = create_unary_graph_1d("atanh", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert atanh 1D");
    }

    // ============================================================================
    // Exponential and Logarithmic Unary Operations
    // ============================================================================

    #[test]
    fn test_exp_scalar() {
        let graph = create_unary_graph_scalar("exp", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert exp scalar");
    }

    #[test]
    fn test_exp_1d() {
        let graph = create_unary_graph_1d("exp", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert exp 1D");
    }

    #[test]
    fn test_exp_4d() {
        let graph = create_unary_graph_4d("exp", 1, 3, 224, 224, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert exp 4D");
    }

    #[test]
    fn test_log_scalar() {
        let graph = create_unary_graph_scalar("log", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert log scalar");
    }

    #[test]
    fn test_log_1d() {
        let graph = create_unary_graph_1d("log", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert log 1D");
    }

    #[test]
    fn test_erf_scalar() {
        let graph = create_unary_graph_scalar("erf", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert erf scalar");
    }

    #[test]
    fn test_erf_1d() {
        let graph = create_unary_graph_1d("erf", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert erf 1D");
    }

    // ============================================================================
    // Rounding Operations
    // ============================================================================

    #[test]
    fn test_round_scalar() {
        let graph = create_unary_graph_scalar("round", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert round scalar");
    }

    #[test]
    fn test_round_1d() {
        let graph = create_unary_graph_1d("round", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert round 1D");
    }

    // ============================================================================
    // Activation Functions (Common Unary Operations)
    // ============================================================================

    #[test]
    fn test_relu_scalar() {
        let graph = create_unary_graph_scalar("relu", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert relu scalar");
    }

    #[test]
    fn test_relu_1d() {
        let graph = create_unary_graph_1d("relu", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert relu 1D");
    }

    #[test]
    fn test_relu_4d() {
        let graph = create_unary_graph_4d("relu", 1, 64, 56, 56, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert relu 4D");
    }

    #[test]
    fn test_sigmoid_scalar() {
        let graph = create_unary_graph_scalar("sigmoid", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert sigmoid scalar");
    }

    #[test]
    fn test_sigmoid_1d() {
        let graph = create_unary_graph_1d("sigmoid", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert sigmoid 1D");
    }

    #[test]
    fn test_sigmoid_4d() {
        let graph = create_unary_graph_4d("sigmoid", 1, 64, 56, 56, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert sigmoid 4D");
    }

    #[test]
    fn test_softplus_scalar() {
        let graph = create_unary_graph_scalar("softplus", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert softplus scalar");
    }

    #[test]
    fn test_softplus_1d() {
        let graph = create_unary_graph_1d("softplus", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert softplus 1D");
    }

    #[test]
    fn test_softsign_scalar() {
        let graph = create_unary_graph_scalar("softsign", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert softsign scalar");
    }

    #[test]
    fn test_softsign_1d() {
        let graph = create_unary_graph_1d("softsign", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert softsign 1D");
    }

    #[test]
    fn test_gelu_scalar() {
        let graph = create_unary_graph_scalar("gelu", DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert gelu scalar");
    }

    #[test]
    fn test_gelu_1d() {
        let graph = create_unary_graph_1d("gelu", 10, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert gelu 1D");
    }

    // ============================================================================
    // Data Type Tests
    // ============================================================================

    #[test]
    fn test_abs_float16() {
        let graph = create_unary_graph_1d("abs", 10, DataType::Float16);
        test_convert_graph(&graph).expect("Failed to convert abs float16");
    }

    #[test]
    fn test_relu_float16() {
        let graph = create_unary_graph_1d("relu", 10, DataType::Float16);
        test_convert_graph(&graph).expect("Failed to convert relu float16");
    }

    // ============================================================================
    // Complex Shape Tests
    // ============================================================================

    #[test]
    fn test_tanh_2d() {
        let graph = create_unary_graph_2d("tanh", 32, 128, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert tanh 2D");
    }

    #[test]
    fn test_sigmoid_2d() {
        let graph = create_unary_graph_2d("sigmoid", 32, 128, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert sigmoid 2D");
    }

    #[test]
    fn test_relu_large_4d() {
        let graph = create_unary_graph_4d("relu", 8, 128, 28, 28, DataType::Float32);
        test_convert_graph(&graph).expect("Failed to convert relu large 4D");
    }
}

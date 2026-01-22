//! TensorRT execution tests with numerical verification
//!
//! These tests verify that WebNN graphs execute correctly on TensorRT
//! and produce numerically correct results.
//!
//! Run with: cargo test --test test_trtx_execution --features trtx-runtime

#[cfg(feature = "trtx-runtime")]
mod tests {
    use rustnn::converters::{GraphConverter, TrtxConverter};
    use rustnn::graph::{DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, Operation};
    use std::collections::HashMap;
    use trtx::cuda::DeviceBuffer;
    use trtx::{Logger, Runtime};

    /// Helper to create a simple unary operation graph
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
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Null,
                label: Some(format!("{}_op", op_type)),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: input_desc,
                    name: Some("input".to_string()),
                },
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

    /// Execute a graph with TensorRT and return output
    fn execute_graph(
        graph: &GraphInfo,
        input_data: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Convert graph to TensorRT
        let converter = TrtxConverter::new();
        let converted = converter.convert(graph)?;

        // Create TensorRT runtime
        let logger = Logger::stderr()?;
        let runtime = Runtime::new(&logger)?;
        let engine = runtime.deserialize_cuda_engine(&converted.data)?;
        let mut context = engine.create_execution_context()?;

        // Get tensor info
        let num_tensors = engine.get_nb_io_tensors()?;
        assert_eq!(num_tensors, 2, "Expected 2 tensors (input + output)");

        let input_name = engine.get_tensor_name(0)?;
        let output_name = engine.get_tensor_name(1)?;

        // Allocate device buffers
        let input_size = input_data.len() * std::mem::size_of::<f32>();
        let output_size = input_size;

        let mut input_buffer = DeviceBuffer::new(input_size)?;
        let output_buffer = DeviceBuffer::new(output_size)?;

        // Copy input data to device (convert f32 slice to bytes)
        let input_bytes = unsafe {
            std::slice::from_raw_parts(
                input_data.as_ptr() as *const u8,
                input_data.len() * std::mem::size_of::<f32>(),
            )
        };
        input_buffer.copy_from_host(input_bytes)?;

        // Set tensor addresses
        unsafe {
            context.set_tensor_address(&input_name, input_buffer.as_ptr())?;
            context.set_tensor_address(&output_name, output_buffer.as_ptr())?;
        }

        // Execute inference
        unsafe {
            context.enqueue_v3(trtx::cuda::get_default_stream())?;
        }
        trtx::cuda::synchronize()?;

        // Copy output back to host (convert bytes to f32 slice)
        let mut output_data = vec![0.0f32; input_data.len()];
        let output_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                output_data.as_mut_ptr() as *mut u8,
                output_data.len() * std::mem::size_of::<f32>(),
            )
        };
        output_buffer.copy_to_host(output_bytes)?;

        Ok(output_data)
    }

    /// Helper to verify output within tolerance
    fn verify_output(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "Output length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );

        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff <= tolerance,
                "Value mismatch at index {}: actual={}, expected={}, diff={}, tolerance={}",
                i,
                a,
                e,
                diff,
                tolerance
            );
        }
    }

    // ============================================================================
    // Execution Tests - Arithmetic Operations
    // ============================================================================

    #[test]
    fn test_abs_execution() {
        let graph = create_unary_graph("abs", vec![4], DataType::Float32);
        let input = vec![-2.0, -1.0, 0.0, 1.0];
        let expected = vec![2.0, 1.0, 0.0, 1.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_neg_execution() {
        let graph = create_unary_graph("neg", vec![4], DataType::Float32);
        let input = vec![-2.0, -1.0, 0.0, 1.0];
        let expected = vec![2.0, 1.0, 0.0, -1.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_sqrt_execution() {
        let graph = create_unary_graph("sqrt", vec![4], DataType::Float32);
        let input = vec![0.0, 1.0, 4.0, 9.0];
        let expected = vec![0.0, 1.0, 2.0, 3.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_reciprocal_execution() {
        let graph = create_unary_graph("reciprocal", vec![4], DataType::Float32);
        let input = vec![1.0, 2.0, 4.0, 10.0];
        let expected = vec![1.0, 0.5, 0.25, 0.1];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_ceil_execution() {
        let graph = create_unary_graph("ceil", vec![4], DataType::Float32);
        let input = vec![-1.5, -0.5, 0.5, 1.5];
        let expected = vec![-1.0, 0.0, 1.0, 2.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_floor_execution() {
        let graph = create_unary_graph("floor", vec![4], DataType::Float32);
        let input = vec![-1.5, -0.5, 0.5, 1.5];
        let expected = vec![-2.0, -1.0, 0.0, 1.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_sign_execution() {
        let graph = create_unary_graph("sign", vec![5], DataType::Float32);
        let input = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
        let expected = vec![-1.0, -1.0, 0.0, 1.0, 1.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    // ============================================================================
    // Execution Tests - Exponential and Logarithmic Operations
    // ============================================================================

    #[test]
    fn test_exp_execution() {
        let graph = create_unary_graph("exp", vec![4], DataType::Float32);
        let input = vec![0.0, 1.0, 2.0, -1.0];
        let expected = vec![1.0, 2.718281828, 7.389056099, 0.367879441];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_log_execution() {
        let graph = create_unary_graph("log", vec![4], DataType::Float32);
        let input = vec![1.0, 2.718281828, 7.389056099, 0.367879441];
        let expected = vec![0.0, 1.0, 2.0, -1.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4); // Slightly larger tolerance for log
    }

    // ============================================================================
    // Execution Tests - Trigonometric Operations
    // ============================================================================

    #[test]
    fn test_sin_execution() {
        let graph = create_unary_graph("sin", vec![4], DataType::Float32);
        let input = vec![
            0.0,
            std::f32::consts::PI / 6.0,
            std::f32::consts::PI / 2.0,
            std::f32::consts::PI,
        ];
        let expected = vec![0.0, 0.5, 1.0, 0.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_cos_execution() {
        let graph = create_unary_graph("cos", vec![4], DataType::Float32);
        let input = vec![
            0.0,
            std::f32::consts::PI / 3.0,
            std::f32::consts::PI / 2.0,
            std::f32::consts::PI,
        ];
        let expected = vec![1.0, 0.5, 0.0, -1.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_tan_execution() {
        let graph = create_unary_graph("tan", vec![3], DataType::Float32);
        let input = vec![0.0, std::f32::consts::PI / 4.0, -std::f32::consts::PI / 4.0];
        let expected = vec![0.0, 1.0, -1.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_asin_execution() {
        let graph = create_unary_graph("asin", vec![3], DataType::Float32);
        let input = vec![0.0, 0.5, 1.0];
        let expected = vec![0.0, std::f32::consts::PI / 6.0, std::f32::consts::PI / 2.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_acos_execution() {
        let graph = create_unary_graph("acos", vec![3], DataType::Float32);
        let input = vec![1.0, 0.5, 0.0];
        let expected = vec![0.0, std::f32::consts::PI / 3.0, std::f32::consts::PI / 2.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_atan_execution() {
        let graph = create_unary_graph("atan", vec![3], DataType::Float32);
        let input = vec![0.0, 1.0, -1.0];
        let expected = vec![0.0, std::f32::consts::PI / 4.0, -std::f32::consts::PI / 4.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    // ============================================================================
    // Execution Tests - Hyperbolic Operations
    // ============================================================================

    #[test]
    fn test_sinh_execution() {
        let graph = create_unary_graph("sinh", vec![4], DataType::Float32);
        let input = vec![0.0, 1.0, -1.0, 2.0];
        let expected = vec![0.0, 1.175201194, -1.175201194, 3.626860408];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_cosh_execution() {
        let graph = create_unary_graph("cosh", vec![4], DataType::Float32);
        let input = vec![0.0, 1.0, -1.0, 2.0];
        let expected = vec![1.0, 1.543080635, 1.543080635, 3.762195691];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_asinh_execution() {
        let graph = create_unary_graph("asinh", vec![4], DataType::Float32);
        let input = vec![0.0, 1.0, -1.0, 2.0];
        let expected = vec![0.0, 0.881373587, -0.881373587, 1.443635475];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_acosh_execution() {
        let graph = create_unary_graph("acosh", vec![3], DataType::Float32);
        let input = vec![1.0, 2.0, 3.0];
        let expected = vec![0.0, 1.316957897, 1.762747174];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_atanh_execution() {
        let graph = create_unary_graph("atanh", vec![3], DataType::Float32);
        let input = vec![0.0, 0.5, -0.5];
        let expected = vec![0.0, 0.549306144, -0.549306144];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    // ============================================================================
    // Execution Tests - Rounding and Other Operations
    // ============================================================================

    #[test]
    fn test_round_execution() {
        let graph = create_unary_graph("round", vec![6], DataType::Float32);
        let input = vec![-1.5, -0.5, 0.5, 1.5, 2.5, 3.5];
        // Round to nearest even
        let expected = vec![-2.0, 0.0, 0.0, 2.0, 2.0, 4.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_erf_execution() {
        let graph = create_unary_graph("erf", vec![5], DataType::Float32);
        let input = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let expected = vec![0.0, 0.520499878, 0.842700793, -0.520499878, -0.842700793];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    // ============================================================================
    // Execution Tests - Activation Functions
    // ============================================================================

    #[test]
    fn test_relu_execution() {
        let graph = create_unary_graph("relu", vec![5], DataType::Float32);
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_sigmoid_execution() {
        let graph = create_unary_graph("sigmoid", vec![5], DataType::Float32);
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let expected = vec![0.119202922, 0.268941421, 0.5, 0.731058579, 0.880797078];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_tanh_execution() {
        let graph = create_unary_graph("tanh", vec![5], DataType::Float32);
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let expected = vec![-0.96402758, -0.76159416, 0.0, 0.76159416, 0.96402758];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_elu_execution() {
        let graph = create_unary_graph("elu", vec![5], DataType::Float32);
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        // ELU with alpha=1.0: x if x > 0, else alpha * (exp(x) - 1)
        let expected = vec![-0.864664717, -0.632120559, 0.0, 1.0, 2.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_softsign_execution() {
        let graph = create_unary_graph("softsign", vec![5], DataType::Float32);
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        // softsign: x / (1 + |x|)
        let expected = vec![-0.666666667, -0.5, 0.0, 0.5, 0.666666667];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_softplus_execution() {
        let graph = create_unary_graph("softplus", vec![5], DataType::Float32);
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        // softplus: ln(1 + exp(x))
        let expected = vec![
            0.126928011,
            0.313261688,
            0.693147181,
            1.313261688,
            2.126928011,
        ];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_gelu_execution() {
        let graph = create_unary_graph("gelu", vec![5], DataType::Float32);
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        // GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        let expected = vec![-0.045500263, -0.158655254, 0.0, 0.841344746, 1.954499737];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4); // Slightly larger tolerance for GELU
    }

    // ============================================================================
    // Execution Tests - Multi-dimensional Tensors
    // ============================================================================

    #[test]
    fn test_abs_2d_execution() {
        let graph = create_unary_graph("abs", vec![2, 3], DataType::Float32);
        let input = vec![-1.0, -2.0, -3.0, 4.0, 5.0, 6.0];
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_relu_2d_execution() {
        let graph = create_unary_graph("relu", vec![2, 3], DataType::Float32);
        let input = vec![-1.0, 0.0, 1.0, -2.0, 3.0, -4.0];
        let expected = vec![0.0, 0.0, 1.0, 0.0, 3.0, 0.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_exp_4d_execution() {
        // 4D tensor: 1×1×2×2 (batch × channels × height × width)
        let graph = create_unary_graph("exp", vec![1, 1, 2, 2], DataType::Float32);
        let input = vec![0.0, 1.0, 2.0, -1.0];
        let expected = vec![1.0, 2.718281828, 7.389056099, 0.367879441];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_relu_4d_execution() {
        // 4D tensor: 1×2×2×2 (batch × channels × height × width)
        let graph = create_unary_graph("relu", vec![1, 2, 2, 2], DataType::Float32);
        let input = vec![-1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, -8.0];
        let expected = vec![0.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 0.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_sigmoid_4d_execution() {
        // 4D tensor: 1×1×2×2
        let graph = create_unary_graph("sigmoid", vec![1, 1, 2, 2], DataType::Float32);
        let input = vec![-1.0, 0.0, 1.0, 2.0];
        let expected = vec![0.268941421, 0.5, 0.731058579, 0.880797078];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }

    #[test]
    fn test_tanh_2d_execution() {
        let graph = create_unary_graph("tanh", vec![2, 3], DataType::Float32);
        let input = vec![-1.0, 0.0, 1.0, -2.0, 0.5, 2.0];
        let expected = vec![
            -0.76159416,
            0.0,
            0.76159416,
            -0.96402758,
            0.46211716,
            0.96402758,
        ];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-5);
    }
}

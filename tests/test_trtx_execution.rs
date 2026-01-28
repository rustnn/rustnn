//! TensorRT execution tests with numerical verification
//!
//! These tests verify that WebNN graphs execute correctly on TensorRT
//! and produce numerically correct results.
//!
//! Run with: cargo test --test test_trtx_execution --features trtx-runtime

#[cfg(feature = "trtx-runtime")]
mod tests {
    use rustnn::converters::{GraphConverter, TrtxConverter};
    use rustnn::graph::{
        ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, Operation,
    };
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

    /// Helper to create a simple binary operation graph
    fn create_binary_graph(op_type: &str, input_shape: Vec<u32>, data_type: DataType) -> GraphInfo {
        let input_desc = OperandDescriptor {
            data_type,
            shape: input_shape.clone(),
            pending_permutation: Vec::new(),
        };

        let output_desc = input_desc.clone();

        GraphInfo {
            operations: vec![Operation {
                op_type: op_type.to_string(),
                input_operands: vec![0, 1], // Two inputs
                output_operand: Some(2),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Null,
                label: Some(format!("{}_op", op_type)),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: input_desc.clone(),
                    name: Some("input_a".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: input_desc,
                    name: Some("input_b".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: output_desc,
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        }
    }

    /// Execute a binary operation graph with TensorRT
    fn execute_binary_graph(
        graph: &GraphInfo,
        input_a: &[f32],
        input_b: &[f32],
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
        assert_eq!(num_tensors, 3, "Expected 3 tensors (2 inputs + 1 output)");

        let input_a_name = engine.get_tensor_name(0)?;
        let input_b_name = engine.get_tensor_name(1)?;
        let output_name = engine.get_tensor_name(2)?;

        // Calculate output size from graph's output operand descriptor
        let output_operand_id = graph.output_operands[0];
        let output_operand = &graph.operands[output_operand_id as usize];
        let output_element_count: usize = output_operand.descriptor.shape.iter().map(|&d| d as usize).product();

        // Allocate device buffers
        let input_size = input_a.len() * std::mem::size_of::<f32>();
        let output_size = output_element_count * std::mem::size_of::<f32>();

        let mut input_a_buffer = DeviceBuffer::new(input_size)?;
        let mut input_b_buffer = DeviceBuffer::new(input_size)?;
        let output_buffer = DeviceBuffer::new(output_size)?;

        // Copy input data to device
        let input_a_bytes = unsafe {
            std::slice::from_raw_parts(
                input_a.as_ptr() as *const u8,
                input_a.len() * std::mem::size_of::<f32>(),
            )
        };
        input_a_buffer.copy_from_host(input_a_bytes)?;

        let input_b_bytes = unsafe {
            std::slice::from_raw_parts(
                input_b.as_ptr() as *const u8,
                input_b.len() * std::mem::size_of::<f32>(),
            )
        };
        input_b_buffer.copy_from_host(input_b_bytes)?;

        // Set tensor addresses
        unsafe {
            context.set_tensor_address(&input_a_name, input_a_buffer.as_ptr())?;
            context.set_tensor_address(&input_b_name, input_b_buffer.as_ptr())?;
            context.set_tensor_address(&output_name, output_buffer.as_ptr())?;
        }

        // Execute inference
        unsafe {
            context.enqueue_v3(trtx::cuda::get_default_stream())?;
        }
        trtx::cuda::synchronize()?;

        // Copy output back to host
        let mut output_data = vec![0.0f32; output_element_count];
        let output_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                output_data.as_mut_ptr() as *mut u8,
                output_data.len() * std::mem::size_of::<f32>(),
            )
        };
        output_buffer.copy_to_host(output_bytes)?;

        Ok(output_data)
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

        // Calculate output size from graph's output operand descriptor
        let output_operand_id = graph.output_operands[0];
        let output_operand = &graph.operands[output_operand_id as usize];
        let output_element_count: usize = output_operand.descriptor.shape.iter().map(|&d| d as usize).product();

        // Allocate device buffers
        let input_size = input_data.len() * std::mem::size_of::<f32>();
        let output_size = output_element_count * std::mem::size_of::<f32>();

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
        let mut output_data = vec![0.0f32; output_element_count];
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

    // ============================================================================
    // Execution Tests - Matrix Operations
    // ============================================================================

    /// Helper to create a matmul graph
    fn create_matmul_graph(
        a_shape: Vec<u32>,
        b_shape: Vec<u32>,
        data_type: DataType,
    ) -> GraphInfo {
        let a_desc = OperandDescriptor {
            data_type,
            shape: a_shape.clone(),
            pending_permutation: Vec::new(),
        };

        let b_desc = OperandDescriptor {
            data_type,
            shape: b_shape.clone(),
            pending_permutation: Vec::new(),
        };

        // Output shape calculation for matrix multiply
        let output_shape = if a_shape.len() == 2 && b_shape.len() == 2 {
            vec![a_shape[0], b_shape[1]]
        } else {
            vec![a_shape[0]] // Simplified for tests
        };

        let output_desc = OperandDescriptor {
            data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        GraphInfo {
            operations: vec![Operation {
                op_type: "matmul".to_string(),
                input_operands: vec![0, 1],
                output_operand: Some(2),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Null,
                label: Some("matmul_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: a_desc,
                    name: Some("a".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: b_desc,
                    name: Some("b".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: output_desc,
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        }
    }

    /// Helper to execute a graph with two inputs
    fn execute_graph_two_inputs(
        graph: &GraphInfo,
        input_a: &[f32],
        input_b: &[f32],
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
        assert_eq!(num_tensors, 3, "Expected 3 tensors (2 inputs + 1 output)");

        let input_a_name = engine.get_tensor_name(0)?;
        let input_b_name = engine.get_tensor_name(1)?;
        let output_name = engine.get_tensor_name(2)?;

        // Allocate device buffers
        let input_a_size = input_a.len() * std::mem::size_of::<f32>();
        let input_b_size = input_b.len() * std::mem::size_of::<f32>();

        // Calculate output size based on operation
        let output_size = if graph.operations[0].op_type == "matmul" {
            // For matrix multiply, output size depends on input shapes
            let a_shape = &graph.operands[0].descriptor.shape;
            let b_shape = &graph.operands[1].descriptor.shape;
            if a_shape.len() == 2 && b_shape.len() == 2 {
                (a_shape[0] * b_shape[1]) as usize * std::mem::size_of::<f32>()
            } else {
                input_a_size // Fallback
            }
        } else {
            input_a_size // For element-wise operations
        };

        let mut input_a_buffer = DeviceBuffer::new(input_a_size)?;
        let mut input_b_buffer = DeviceBuffer::new(input_b_size)?;
        let output_buffer = DeviceBuffer::new(output_size)?;

        // Copy input data to device
        let input_a_bytes = unsafe {
            std::slice::from_raw_parts(
                input_a.as_ptr() as *const u8,
                input_a.len() * std::mem::size_of::<f32>(),
            )
        };
        input_a_buffer.copy_from_host(input_a_bytes)?;

        let input_b_bytes = unsafe {
            std::slice::from_raw_parts(
                input_b.as_ptr() as *const u8,
                input_b.len() * std::mem::size_of::<f32>(),
            )
        };
        input_b_buffer.copy_from_host(input_b_bytes)?;

        // Set tensor addresses
        unsafe {
            context.set_tensor_address(&input_a_name, input_a_buffer.as_ptr())?;
            context.set_tensor_address(&input_b_name, input_b_buffer.as_ptr())?;
            context.set_tensor_address(&output_name, output_buffer.as_ptr())?;
        }

        // Execute inference
        unsafe {
            context.enqueue_v3(trtx::cuda::get_default_stream())?;
        }
        trtx::cuda::synchronize()?;

        // Copy output back to host
        let output_elem_count = output_size / std::mem::size_of::<f32>();
        let mut output_data = vec![0.0f32; output_elem_count];
        let output_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                output_data.as_mut_ptr() as *mut u8,
                output_data.len() * std::mem::size_of::<f32>(),
            )
        };
        output_buffer.copy_to_host(output_bytes)?;

        Ok(output_data)
    }

    #[test]
    fn test_matmul_2x2_execution() {
        let graph = create_matmul_graph(vec![2, 2], vec![2, 2], DataType::Float32);

        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // Result = [[19, 22], [43, 50]]
        let input_a = vec![1.0, 2.0, 3.0, 4.0];
        let input_b = vec![5.0, 6.0, 7.0, 8.0];
        let expected = vec![19.0, 22.0, 43.0, 50.0];

        let output =
            execute_graph_two_inputs(&graph, &input_a, &input_b).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_matmul_3x3_execution() {
        let graph = create_matmul_graph(vec![3, 3], vec![3, 3], DataType::Float32);

        // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        // B = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] (identity)
        // Result = A
        let input_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let input_b = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let expected = input_a.clone();

        let output =
            execute_graph_two_inputs(&graph, &input_a, &input_b).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_matmul_2x3_3x2_execution() {
        let graph = create_matmul_graph(vec![2, 3], vec![3, 2], DataType::Float32);

        // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
        // B = [[1, 2], [3, 4], [5, 6]]  (3x2)
        // Result = [[22, 28], [49, 64]]  (2x2)
        let input_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input_b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let expected = vec![22.0, 28.0, 49.0, 64.0];

        let output =
            execute_graph_two_inputs(&graph, &input_a, &input_b).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    // ============================================================================
    // Execution Tests - GEMM Operations
    // ============================================================================

    /// Helper to create a GEMM graph (alpha * A * B + beta * C)
    fn create_gemm_graph(
        a_shape: Vec<u32>,
        b_shape: Vec<u32>,
        c_shape: Vec<u32>,
        alpha: f32,
        beta: f32,
        a_transpose: bool,
        b_transpose: bool,
        data_type: DataType,
    ) -> GraphInfo {
        let a_desc = OperandDescriptor {
            data_type,
            shape: a_shape.clone(),
            pending_permutation: Vec::new(),
        };

        let b_desc = OperandDescriptor {
            data_type,
            shape: b_shape.clone(),
            pending_permutation: Vec::new(),
        };

        let c_desc = OperandDescriptor {
            data_type,
            shape: c_shape.clone(),
            pending_permutation: Vec::new(),
        };

        let output_desc = c_desc.clone();

        let mut attributes = serde_json::Map::new();
        attributes.insert("alpha".to_string(), serde_json::json!(alpha));
        attributes.insert("beta".to_string(), serde_json::json!(beta));
        attributes.insert("aTranspose".to_string(), serde_json::json!(a_transpose));
        attributes.insert("bTranspose".to_string(), serde_json::json!(b_transpose));

        GraphInfo {
            operations: vec![Operation {
                op_type: "gemm".to_string(),
                input_operands: vec![0, 1, 2],
                output_operand: Some(3),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("gemm_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: a_desc,
                    name: Some("a".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: b_desc,
                    name: Some("b".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: c_desc,
                    name: Some("c".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: output_desc,
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0, 1, 2],
            output_operands: vec![3],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        }
    }

    /// Helper to execute a graph with three inputs
    fn execute_graph_three_inputs(
        graph: &GraphInfo,
        input_a: &[f32],
        input_b: &[f32],
        input_c: &[f32],
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
        assert_eq!(num_tensors, 4, "Expected 4 tensors (3 inputs + 1 output)");

        let input_a_name = engine.get_tensor_name(0)?;
        let input_b_name = engine.get_tensor_name(1)?;
        let input_c_name = engine.get_tensor_name(2)?;
        let output_name = engine.get_tensor_name(3)?;

        // Allocate device buffers
        let input_a_size = input_a.len() * std::mem::size_of::<f32>();
        let input_b_size = input_b.len() * std::mem::size_of::<f32>();
        let input_c_size = input_c.len() * std::mem::size_of::<f32>();
        let output_size = input_c_size; // Output has same size as C

        let mut input_a_buffer = DeviceBuffer::new(input_a_size)?;
        let mut input_b_buffer = DeviceBuffer::new(input_b_size)?;
        let mut input_c_buffer = DeviceBuffer::new(input_c_size)?;
        let output_buffer = DeviceBuffer::new(output_size)?;

        // Copy input data to device
        let input_a_bytes = unsafe {
            std::slice::from_raw_parts(
                input_a.as_ptr() as *const u8,
                input_a.len() * std::mem::size_of::<f32>(),
            )
        };
        input_a_buffer.copy_from_host(input_a_bytes)?;

        let input_b_bytes = unsafe {
            std::slice::from_raw_parts(
                input_b.as_ptr() as *const u8,
                input_b.len() * std::mem::size_of::<f32>(),
            )
        };
        input_b_buffer.copy_from_host(input_b_bytes)?;

        let input_c_bytes = unsafe {
            std::slice::from_raw_parts(
                input_c.as_ptr() as *const u8,
                input_c.len() * std::mem::size_of::<f32>(),
            )
        };
        input_c_buffer.copy_from_host(input_c_bytes)?;

        // Set tensor addresses
        unsafe {
            context.set_tensor_address(&input_a_name, input_a_buffer.as_ptr())?;
            context.set_tensor_address(&input_b_name, input_b_buffer.as_ptr())?;
            context.set_tensor_address(&input_c_name, input_c_buffer.as_ptr())?;
            context.set_tensor_address(&output_name, output_buffer.as_ptr())?;
        }

        // Execute inference
        unsafe {
            context.enqueue_v3(trtx::cuda::get_default_stream())?;
        }
        trtx::cuda::synchronize()?;

        // Copy output back to host
        let output_elem_count = output_size / std::mem::size_of::<f32>();
        let mut output_data = vec![0.0f32; output_elem_count];
        let output_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                output_data.as_mut_ptr() as *mut u8,
                output_data.len() * std::mem::size_of::<f32>(),
            )
        };
        output_buffer.copy_to_host(output_bytes)?;

        Ok(output_data)
    }

    #[test]
    fn test_gemm_basic_execution() {
        // C = 1.0 * A * B + 1.0 * C
        let graph = create_gemm_graph(
            vec![2, 2],
            vec![2, 2],
            vec![2, 2],
            1.0,
            1.0,
            false,
            false,
            DataType::Float32,
        );

        // A = [[1, 2], [3, 4]]
        // B = [[1, 0], [0, 1]]
        // C = [[1, 1], [1, 1]]
        // Result = A * B + C = [[1, 2], [3, 4]] + [[1, 1], [1, 1]] = [[2, 3], [4, 5]]
        let input_a = vec![1.0, 2.0, 3.0, 4.0];
        let input_b = vec![1.0, 0.0, 0.0, 1.0];
        let input_c = vec![1.0, 1.0, 1.0, 1.0];
        let expected = vec![2.0, 3.0, 4.0, 5.0];

        let output =
            execute_graph_three_inputs(&graph, &input_a, &input_b, &input_c)
                .expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_gemm_alpha_execution() {
        // C = 2.0 * A * B + 1.0 * C
        let graph = create_gemm_graph(
            vec![2, 2],
            vec![2, 2],
            vec![2, 2],
            2.0,
            1.0,
            false,
            false,
            DataType::Float32,
        );

        // A = [[1, 2], [3, 4]]
        // B = [[1, 0], [0, 1]]
        // C = [[0, 0], [0, 0]]
        // Result = 2 * (A * B) + C = 2 * [[1, 2], [3, 4]] = [[2, 4], [6, 8]]
        let input_a = vec![1.0, 2.0, 3.0, 4.0];
        let input_b = vec![1.0, 0.0, 0.0, 1.0];
        let input_c = vec![0.0, 0.0, 0.0, 0.0];
        let expected = vec![2.0, 4.0, 6.0, 8.0];

        let output =
            execute_graph_three_inputs(&graph, &input_a, &input_b, &input_c)
                .expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_gemm_beta_execution() {
        // C = 1.0 * A * B + 2.0 * C
        let graph = create_gemm_graph(
            vec![2, 2],
            vec![2, 2],
            vec![2, 2],
            1.0,
            2.0,
            false,
            false,
            DataType::Float32,
        );

        // A = [[1, 0], [0, 1]]
        // B = [[1, 0], [0, 1]]
        // C = [[1, 2], [3, 4]]
        // Result = (A * B) + 2 * C = [[1, 0], [0, 1]] + [[2, 4], [6, 8]] = [[3, 4], [6, 9]]
        let input_a = vec![1.0, 0.0, 0.0, 1.0];
        let input_b = vec![1.0, 0.0, 0.0, 1.0];
        let input_c = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![3.0, 4.0, 6.0, 9.0];

        let output =
            execute_graph_three_inputs(&graph, &input_a, &input_b, &input_c)
                .expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    // ============================================================================
    // Execution Tests - Convolution Operations
    // ============================================================================

    /// Helper to create a conv2d graph with constant filter
    fn create_conv2d_graph(
        input_shape: Vec<u32>,  // [batch, channels, height, width]
        filter_shape: Vec<u32>, // [out_channels, in_channels, kernel_h, kernel_w]
        filter_data: Vec<f32>,
        bias_data: Option<Vec<f32>>,
        data_type: DataType,
    ) -> GraphInfo {
        let input_desc = OperandDescriptor {
            data_type,
            shape: input_shape.clone(),
            pending_permutation: Vec::new(),
        };

        let filter_desc = OperandDescriptor {
            data_type,
            shape: filter_shape.clone(),
            pending_permutation: Vec::new(),
        };

        // Calculate output shape: [batch, out_channels, out_h, out_w]
        // For simplicity, assuming no padding, stride=1, dilation=1
        let out_h = input_shape[2] - filter_shape[2] + 1;
        let out_w = input_shape[3] - filter_shape[3] + 1;
        let output_shape = vec![input_shape[0], filter_shape[0], out_h, out_w];

        let output_desc = OperandDescriptor {
            data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        // Convert filter data to bytes
        let filter_bytes: Vec<u8> = filter_data
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();

        let mut constant_map = HashMap::new();
        constant_map.insert(
            1,
            ConstantData {
                data: filter_bytes,
                label: Some("filter".to_string()),
            },
        );

        let mut input_operands = vec![0, 1]; // input and filter
        let mut operands = vec![
            Operand {
                kind: OperandKind::Input,
                descriptor: input_desc,
                name: Some("input".to_string()),
            },
            Operand {
                kind: OperandKind::Constant,
                descriptor: filter_desc,
                name: Some("filter".to_string()),
            },
        ];

        // Add bias if provided
        if let Some(bias) = bias_data {
            let bias_desc = OperandDescriptor {
                data_type,
                shape: vec![filter_shape[0]], // bias shape = [out_channels]
                pending_permutation: Vec::new(),
            };

            let bias_bytes: Vec<u8> = bias.iter().flat_map(|&f| f.to_le_bytes()).collect();

            constant_map.insert(
                2,
                ConstantData {
                    data: bias_bytes,
                    label: Some("bias".to_string()),
                },
            );

            operands.push(Operand {
                kind: OperandKind::Constant,
                descriptor: bias_desc,
                name: Some("bias".to_string()),
            });

            input_operands.push(2);
        }

        // Add output operand
        operands.push(Operand {
            kind: OperandKind::Output,
            descriptor: output_desc,
            name: Some("output".to_string()),
        });

        let output_operand_id = operands.len() as u32 - 1;

        GraphInfo {
            operations: vec![Operation {
                op_type: "conv2d".to_string(),
                input_operands,
                output_operand: Some(output_operand_id),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Null,
                label: Some("conv2d_op".to_string()),
            }],
            operands,
            input_operands: vec![0],
            output_operands: vec![output_operand_id],
            constant_operand_ids_to_handles: constant_map,
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        }
    }

    #[test]
    fn test_conv2d_simple_execution() {
        // Simple 1x1 convolution (channel-wise scaling)
        // Input: [1, 1, 2, 2] (batch=1, channels=1, h=2, w=2)
        // Filter: [1, 1, 1, 1] (out_channels=1, in_channels=1, kh=1, kw=1)
        // Filter weights: [[1.0]]
        // Output: [1, 1, 2, 2]

        let input_shape = vec![1, 1, 2, 2];
        let filter_shape = vec![1, 1, 1, 1];
        let filter_data = vec![2.0]; // Scale by 2

        let graph = create_conv2d_graph(
            input_shape,
            filter_shape,
            filter_data,
            None,
            DataType::Float32,
        );

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![2.0, 4.0, 6.0, 8.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_conv2d_with_bias_execution() {
        // 1x1 convolution with bias
        let input_shape = vec![1, 1, 2, 2];
        let filter_shape = vec![1, 1, 1, 1];
        let filter_data = vec![1.0];
        let bias_data = Some(vec![10.0]);

        let graph = create_conv2d_graph(
            input_shape,
            filter_shape,
            filter_data,
            bias_data,
            DataType::Float32,
        );

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![11.0, 12.0, 13.0, 14.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    // ============================================================================
    // New Operations Tests (2026-01-28)
    // ============================================================================

    // Binary Element-wise Operations

    #[test]
    fn test_max_execution() {
        // Test element-wise max: max([-1, 2, -3, 4], [1, -2, 3, -4])
        let graph = create_binary_graph("max", vec![4], DataType::Float32);
        let input_a = vec![-1.0, 2.0, -3.0, 4.0];
        let input_b = vec![1.0, -2.0, 3.0, -4.0];
        let expected = vec![1.0, 2.0, 3.0, 4.0];

        let output = execute_binary_graph(&graph, &input_a, &input_b).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_min_execution() {
        // Test element-wise min: min([-1, 2, -3, 4], [1, -2, 3, -4])
        let graph = create_binary_graph("min", vec![4], DataType::Float32);
        let input_a = vec![-1.0, 2.0, -3.0, 4.0];
        let input_b = vec![1.0, -2.0, 3.0, -4.0];
        let expected = vec![-1.0, -2.0, -3.0, -4.0];

        let output = execute_binary_graph(&graph, &input_a, &input_b).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    // Unary Activation Operations

    #[test]
    fn test_leaky_relu_execution() {
        // LeakyReLU with default alpha=0.01: x if x > 0, else 0.01*x
        let graph = create_unary_graph("leakyRelu", vec![4], DataType::Float32);
        let input = vec![-2.0, -1.0, 1.0, 2.0];
        let expected = vec![-0.02, -0.01, 1.0, 2.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_prelu_execution() {
        // PReLU: max(0, x) + slope * min(0, x)
        // Create graph with slope constant
        let input_desc = OperandDescriptor {
            data_type: DataType::Float32,
            shape: vec![4],
            pending_permutation: Vec::new(),
        };

        let slope_desc = OperandDescriptor {
            data_type: DataType::Float32,
            shape: vec![1],
            pending_permutation: Vec::new(),
        };

        let slope_data = vec![0.25f32];
        let slope_bytes: Vec<u8> = slope_data
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();

        let mut constants = HashMap::new();
        constants.insert(
            1,
            ConstantData {
                data: slope_bytes,
                label: None,
            },
        );

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "prelu".to_string(),
                input_operands: vec![0, 1], // input and slope
                output_operand: Some(2),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Null,
                label: Some("prelu_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: input_desc.clone(),
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Constant,
                    descriptor: slope_desc,
                    name: Some("slope".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: input_desc,
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![2],
            constant_operand_ids_to_handles: constants,
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![-2.0, -1.0, 1.0, 2.0];
        let expected = vec![-0.5, -0.25, 1.0, 2.0]; // 0.25 * -2, 0.25 * -1, 1, 2

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_hard_sigmoid_execution() {
        // HardSigmoid: clamp(alpha*x + beta, 0, 1) with default alpha=0.2, beta=0.5
        let graph = create_unary_graph("hardSigmoid", vec![5], DataType::Float32);
        let input = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        // alpha=0.2, beta=0.5
        // -3: clamp(-0.6 + 0.5, 0, 1) = clamp(-0.1, 0, 1) = 0
        // -1: clamp(-0.2 + 0.5, 0, 1) = clamp(0.3, 0, 1) = 0.3
        //  0: clamp(0 + 0.5, 0, 1) = 0.5
        //  1: clamp(0.2 + 0.5, 0, 1) = clamp(0.7, 0, 1) = 0.7
        //  3: clamp(0.6 + 0.5, 0, 1) = clamp(1.1, 0, 1) = 1.0
        let expected = vec![0.0, 0.3, 0.5, 0.7, 1.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_hard_swish_execution() {
        // HardSwish: x * hardSigmoid(x)
        let graph = create_unary_graph("hardSwish", vec![4], DataType::Float32);
        let input = vec![-3.0, 0.0, 1.0, 3.0];
        // -3: -3 * 0 = 0
        //  0:  0 * 0.5 = 0
        //  1:  1 * 0.7 = 0.7
        //  3:  3 * 1.0 = 3.0
        let expected = vec![0.0, 0.0, 0.7, 3.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-3); // Slightly higher tolerance for composite op
    }

    // Unary Mathematical Operations

    #[test]
    fn test_identity_execution() {
        // Identity: output = input (no transformation)
        let graph = create_unary_graph("identity", vec![4], DataType::Float32);
        let input = vec![-1.5, 0.0, 1.5, 3.14];
        let expected = input.clone();

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-6);
    }

    #[test]
    fn test_cast_execution() {
        // Cast: type conversion (currently uses identity with implicit conversion)
        let graph = create_unary_graph("cast", vec![4], DataType::Float32);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected = input.clone();

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    // Pooling Operations

    #[test]
    fn test_global_average_pool_execution() {
        // GlobalAveragePool: average over spatial dimensions (H, W)
        // Input: [1, 2, 2, 2] (NCHW format: batch=1, channels=2, height=2, width=2)
        let input_shape = vec![1, 2, 2, 2];
        
        let input_desc = OperandDescriptor {
            data_type: DataType::Float32,
            shape: input_shape.clone(),
            pending_permutation: Vec::new(),
        };

        let output_desc = OperandDescriptor {
            data_type: DataType::Float32,
            shape: vec![1, 2, 1, 1], // Output: [1, 2, 1, 1]
            pending_permutation: Vec::new(),
        };

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "globalAveragePool".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Null,
                label: Some("global_avg_pool_op".to_string()),
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
        };

        // Input: channel 0: [1,2,3,4], channel 1: [5,6,7,8]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // Expected: channel 0 avg=(1+2+3+4)/4=2.5, channel 1 avg=(5+6+7+8)/4=6.5
        let expected = vec![2.5, 6.5];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_global_max_pool_execution() {
        // GlobalMaxPool: max over spatial dimensions (H, W)
        // Input: [1, 2, 2, 2] (NCHW format: batch=1, channels=2, height=2, width=2)
        let input_shape = vec![1, 2, 2, 2];
        
        let input_desc = OperandDescriptor {
            data_type: DataType::Float32,
            shape: input_shape.clone(),
            pending_permutation: Vec::new(),
        };

        let output_desc = OperandDescriptor {
            data_type: DataType::Float32,
            shape: vec![1, 2, 1, 1], // Output: [1, 2, 1, 1]
            pending_permutation: Vec::new(),
        };

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "globalMaxPool".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Null,
                label: Some("global_max_pool_op".to_string()),
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
        };

        // Input: channel 0: [1,2,3,4], channel 1: [5,6,7,8]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // Expected: channel 0 max=4, channel 1 max=8
        let expected = vec![4.0, 8.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }
}

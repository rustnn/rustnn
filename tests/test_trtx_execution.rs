//! TensorRT execution tests with numerical verification
//!
//! These tests verify that WebNN graphs execute correctly on TensorRT
//! and produce numerically correct results.
//!
//! Tests only run with real TensorRT (trtx-runtime). With trtx-runtime-mock, they are excluded.
//! Run with: cargo test --test test_trtx_execution --features trtx-runtime

#[cfg(all(feature = "trtx-runtime", not(feature = "trtx-runtime-mock")))]
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

    /// Helper to create a single-input operation graph with custom output shape
    fn create_single_input_graph(
        input_shape: Vec<u32>,
        output_shape: Vec<u32>,
        op_type: &str,
        data_type: DataType,
    ) -> GraphInfo {
        let input_desc = OperandDescriptor {
            data_type,
            shape: input_shape,
            pending_permutation: Vec::new(),
        };

        let output_desc = OperandDescriptor {
            data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

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
        let output_element_count: usize = output_operand
            .descriptor
            .shape
            .iter()
            .map(|&d| d as usize)
            .product();

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

    /// Execute a graph with TensorRT and return output (single input)
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
        let output_element_count: usize = output_operand
            .descriptor
            .shape
            .iter()
            .map(|&d| d as usize)
            .product();

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

    /// Execute a graph with TensorRT and return output (multiple inputs)
    fn execute_graph_multi_input(
        graph: &GraphInfo,
        input_data_vec: Vec<Vec<f32>>,
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
        let num_inputs = graph.input_operands.len();
        let num_graph_outputs = graph.output_operands.len();
        // Note: TensorRT may have more outputs than graph.output_operands due to intermediate values
        // being marked as OperandKind::Output. We only care about the final outputs.
        assert!(
            num_tensors as usize >= num_inputs + num_graph_outputs,
            "Expected at least {} tensors ({} inputs + {} final outputs), got {}",
            num_inputs + num_graph_outputs,
            num_inputs,
            num_graph_outputs,
            num_tensors
        );
        assert_eq!(
            input_data_vec.len(),
            num_inputs,
            "Expected {} input data arrays",
            num_inputs
        );

        // Calculate output size from graph's output operand descriptor
        let output_operand_id = graph.output_operands[0];
        let output_operand = &graph.operands[output_operand_id as usize];
        let output_element_count: usize = output_operand
            .descriptor
            .shape
            .iter()
            .map(|&d| d as usize)
            .product();

        // Allocate device buffers for all inputs
        let mut input_buffers = Vec::new();
        for (i, input_data) in input_data_vec.iter().enumerate() {
            let input_operand_id = graph.input_operands[i];
            let input_operand = &graph.operands[input_operand_id as usize];
            let data_type = &input_operand.descriptor.data_type;

            // Handle different data types
            match data_type {
                DataType::Int32 => {
                    // Convert f32 to i32
                    let int32_data: Vec<i32> = input_data.iter().map(|&f| f as i32).collect();
                    let input_size = int32_data.len() * std::mem::size_of::<i32>();
                    let mut buffer = DeviceBuffer::new(input_size)?;

                    let input_bytes = unsafe {
                        std::slice::from_raw_parts(
                            int32_data.as_ptr() as *const u8,
                            int32_data.len() * std::mem::size_of::<i32>(),
                        )
                    };
                    buffer.copy_from_host(input_bytes)?;
                    input_buffers.push(buffer);
                }
                _ => {
                    // Float32 or other types - treat as f32
                    let input_size = input_data.len() * std::mem::size_of::<f32>();
                    let mut buffer = DeviceBuffer::new(input_size)?;

                    let input_bytes = unsafe {
                        std::slice::from_raw_parts(
                            input_data.as_ptr() as *const u8,
                            input_data.len() * std::mem::size_of::<f32>(),
                        )
                    };
                    buffer.copy_from_host(input_bytes)?;
                    input_buffers.push(buffer);
                }
            }
        }

        // Allocate output buffers for all TensorRT outputs (including intermediates)
        let num_trt_outputs = num_tensors as usize - num_inputs;
        let mut output_buffers = Vec::new();

        // Allocate buffer for each output tensor
        for _i in 0..num_trt_outputs {
            // For simplicity, allocate max size based on final output
            // (intermediate outputs are usually smaller or same size)
            let output_size = output_element_count * std::mem::size_of::<f32>();
            let buffer = DeviceBuffer::new(output_size)?;
            output_buffers.push(buffer);
        }

        // Set tensor addresses for all inputs
        for (i, buffer) in input_buffers.iter().enumerate() {
            let tensor_name = engine.get_tensor_name(i as i32)?;
            unsafe {
                context.set_tensor_address(&tensor_name, buffer.as_ptr())?;
            }
        }

        // Set tensor addresses for all outputs
        for (i, buffer) in output_buffers.iter().enumerate() {
            let tensor_name = engine.get_tensor_name((num_inputs + i) as i32)?;
            unsafe {
                context.set_tensor_address(&tensor_name, buffer.as_ptr())?;
            }
        }

        // Execute inference
        unsafe {
            context.enqueue_v3(trtx::cuda::get_default_stream())?;
        }
        trtx::cuda::synchronize()?;

        // Copy only the final output back to host
        // The final output is the last output buffer
        let final_output_buffer = &output_buffers[output_buffers.len() - 1];
        let mut output_data = vec![0.0f32; output_element_count];
        let output_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                output_data.as_mut_ptr() as *mut u8,
                output_data.len() * std::mem::size_of::<f32>(),
            )
        };
        final_output_buffer.copy_to_host(output_bytes)?;

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
    fn create_matmul_graph(a_shape: Vec<u32>, b_shape: Vec<u32>, data_type: DataType) -> GraphInfo {
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

        let output = execute_graph_three_inputs(&graph, &input_a, &input_b, &input_c)
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

        let output = execute_graph_three_inputs(&graph, &input_a, &input_b, &input_c)
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

        let output = execute_graph_three_inputs(&graph, &input_a, &input_b, &input_c)
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
        let filter_bytes: Vec<u8> = filter_data.iter().flat_map(|&f| f.to_le_bytes()).collect();

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
        let slope_bytes: Vec<u8> = slope_data.iter().flat_map(|&f| f.to_le_bytes()).collect();

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

    // ============================================================================
    // Reduction Operations Tests (2026-01-29)
    // ============================================================================

    /// Helper to create a reduction operation graph
    fn create_reduce_graph(
        op_type: &str,
        input_shape: Vec<u32>,
        output_shape: Vec<u32>,
        axes: Vec<u32>,
        keep_dims: bool,
        data_type: DataType,
    ) -> GraphInfo {
        let input_desc = OperandDescriptor {
            data_type,
            shape: input_shape,
            pending_permutation: Vec::new(),
        };

        let output_desc = OperandDescriptor {
            data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let mut attributes = serde_json::Map::new();
        attributes.insert(
            "axes".to_string(),
            serde_json::Value::Array(axes.iter().map(|&a| serde_json::Value::from(a)).collect()),
        );
        attributes.insert(
            "keepDimensions".to_string(),
            serde_json::Value::Bool(keep_dims),
        );

        GraphInfo {
            operations: vec![Operation {
                op_type: op_type.to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
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

    #[test]
    fn test_reduce_sum_execution() {
        // ReduceSum: sum along axis 1
        // Input: [[1, 2], [3, 4]] shape [2, 2]
        // Output: [3, 7] shape [2] (axis 1 reduced)
        let graph = create_reduce_graph(
            "reduceSum",
            vec![2, 2],
            vec![2],
            vec![1],
            false,
            DataType::Float32,
        );

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![3.0, 7.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_reduce_mean_execution() {
        // ReduceMean: average along axis 1
        // Input: [[1, 2], [3, 4]] shape [2, 2]
        // Output: [1.5, 3.5] shape [2] (axis 1 reduced)
        let graph = create_reduce_graph(
            "reduceMean",
            vec![2, 2],
            vec![2],
            vec![1],
            false,
            DataType::Float32,
        );

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![1.5, 3.5];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_reduce_max_execution() {
        // ReduceMax: max along axis 1
        // Input: [[1, 2], [3, 4]] shape [2, 2]
        // Output: [2, 4] shape [2] (axis 1 reduced)
        let graph = create_reduce_graph(
            "reduceMax",
            vec![2, 2],
            vec![2],
            vec![1],
            false,
            DataType::Float32,
        );

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![2.0, 4.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_reduce_min_execution() {
        // ReduceMin: min along axis 1
        // Input: [[1, 2], [3, 4]] shape [2, 2]
        // Output: [1, 3] shape [2] (axis 1 reduced)
        let graph = create_reduce_graph(
            "reduceMin",
            vec![2, 2],
            vec![2],
            vec![1],
            false,
            DataType::Float32,
        );

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![1.0, 3.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_reduce_product_execution() {
        // ReduceProduct: product along axis 1
        // Input: [[2, 3], [4, 5]] shape [2, 2]
        // Output: [6, 20] shape [2] (axis 1 reduced)
        let graph = create_reduce_graph(
            "reduceProduct",
            vec![2, 2],
            vec![2],
            vec![1],
            false,
            DataType::Float32,
        );

        let input = vec![2.0, 3.0, 4.0, 5.0];
        let expected = vec![6.0, 20.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_reduce_l1_execution() {
        // ReduceL1: sum(abs(x)) along axis 1
        // Input: [[-1, 2], [-3, 4]] shape [2, 2]
        // Output: [3, 7] shape [2] (sum of absolute values)
        let graph = create_reduce_graph(
            "reduceL1",
            vec![2, 2],
            vec![2],
            vec![1],
            false,
            DataType::Float32,
        );

        let input = vec![-1.0, 2.0, -3.0, 4.0];
        let expected = vec![3.0, 7.0]; // abs(-1)+abs(2)=3, abs(-3)+abs(4)=7

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_reduce_l2_execution() {
        // ReduceL2: sqrt(sum(x^2)) along axis 1
        // Input: [[3, 4], [5, 12]] shape [2, 2]
        // Output: [5, 13] shape [2] (L2 norm)
        let graph = create_reduce_graph(
            "reduceL2",
            vec![2, 2],
            vec![2],
            vec![1],
            false,
            DataType::Float32,
        );

        let input = vec![3.0, 4.0, 5.0, 12.0];
        let expected = vec![5.0, 13.0]; // sqrt(9+16)=5, sqrt(25+144)=13

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_reduce_log_sum_execution() {
        // ReduceLogSum: log(sum(x)) along axis 1
        // Input: [[1, e-1], [e, e^2-e]] shape [2, 2]
        let e = std::f32::consts::E;
        let graph = create_reduce_graph(
            "reduceLogSum",
            vec![2, 2],
            vec![2],
            vec![1],
            false,
            DataType::Float32,
        );

        let input = vec![1.0, e - 1.0, e, e * e - e];
        let expected = vec![1.0, 2.0]; // log(e)=1, log(e^2)=2

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-3);
    }

    #[test]
    fn test_reduce_log_sum_exp_execution() {
        // ReduceLogSumExp: log(sum(exp(x))) along axis 1
        // Input: [[0, 0], [1, 1]] shape [2, 2]
        // Output: [log(2), log(2*e)] shape [2]
        let graph = create_reduce_graph(
            "reduceLogSumExp",
            vec![2, 2],
            vec![2],
            vec![1],
            false,
            DataType::Float32,
        );

        let input = vec![0.0, 0.0, 1.0, 1.0];
        let expected = vec![
            (2.0f32).ln(),       // log(exp(0)+exp(0)) = log(2)
            1.0 + (2.0f32).ln(), // log(exp(1)+exp(1)) = log(2e) = 1 + log(2)
        ];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-3);
    }

    #[test]
    fn test_reduce_sum_square_execution() {
        // ReduceSumSquare: sum(x^2) along axis 1
        // Input: [[3, 4], [1, 2]] shape [2, 2]
        // Output: [25, 5] shape [2] (sum of squares)
        let graph = create_reduce_graph(
            "reduceSumSquare",
            vec![2, 2],
            vec![2],
            vec![1],
            false,
            DataType::Float32,
        );

        let input = vec![3.0, 4.0, 1.0, 2.0];
        let expected = vec![25.0, 5.0]; // 9+16=25, 1+4=5

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_reduce_sum_keep_dims_execution() {
        // ReduceSum with keepDimensions=true
        // Input: [[1, 2], [3, 4]] shape [2, 2]
        // Output: [[3], [7]] shape [2, 1] (axis 1 kept as size 1)
        let graph = create_reduce_graph(
            "reduceSum",
            vec![2, 2],
            vec![2, 1],
            vec![1],
            true, // keepDimensions
            DataType::Float32,
        );

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![3.0, 7.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_reduce_sum_multi_axis_execution() {
        // ReduceSum along multiple axes [0, 1]
        // Input: [[1, 2], [3, 4]] shape [2, 2]
        // Output: [10] shape [] (all axes reduced to scalar)
        let graph = create_reduce_graph(
            "reduceSum",
            vec![2, 2],
            vec![],
            vec![0, 1],
            false,
            DataType::Float32,
        );

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![10.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    // ============================================================================
    // Normalization Operations Tests (2026-01-29)
    // ============================================================================

    #[test]
    fn test_batch_normalization_execution() {
        // BatchNorm: y = (x - mean) / sqrt(variance) * 1.0 + 0.0
        // Input: [[1, 2], [3, 4]] shape [1, 2, 1, 2] (NCHW: batch=1, channels=2, H=1, W=2)
        // Mean: [1, 2, 1, 1] reshaped from [2] for broadcasting
        // Variance: [1, 2, 1, 1] reshaped from [2] for broadcasting
        // Expected: [[(1-1.5)/0.5, (2-1.5)/0.5], [(3-3.5)/0.5, (4-3.5)/0.5]]
        //         = [[-1, 1], [-1, 1]]

        // Note: TensorRT requires mean/variance to have same rank as input for broadcasting
        // So we use [1, 2, 1, 1] instead of [2]
        let input_desc = OperandDescriptor {
            data_type: DataType::Float32,
            shape: vec![1, 2, 1, 2],
            pending_permutation: Vec::new(),
        };

        let stats_desc = OperandDescriptor {
            data_type: DataType::Float32,
            shape: vec![1, 2, 1, 1], // Reshaped for broadcasting
            pending_permutation: Vec::new(),
        };

        let output_desc = OperandDescriptor {
            data_type: DataType::Float32,
            shape: vec![1, 2, 1, 2],
            pending_permutation: Vec::new(),
        };

        let mut attributes = serde_json::Map::new();
        attributes.insert("epsilon".to_string(), serde_json::Value::from(0.0));

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "batchNormalization".to_string(),
                input_operands: vec![0, 1, 2], // input, mean, variance
                output_operand: Some(3),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("batch_norm_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: input_desc,
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: stats_desc.clone(),
                    name: Some("mean".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: stats_desc,
                    name: Some("variance".to_string()),
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
        };

        // Input data in NCHW format
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mean = vec![1.5, 3.5]; // Shape [1,2,1,1] for broadcasting
        let variance = vec![0.25, 0.25]; // sqrt(0.25) = 0.5, shape [1,2,1,1]

        let all_inputs = vec![input, mean, variance];

        let expected = vec![-1.0, 1.0, -1.0, 1.0];

        let output = execute_graph_multi_input(&graph, all_inputs).expect("Execution failed");
        verify_output(&output, &expected, 1e-3);
    }

    /// Helper to create instance normalization graph
    fn create_instance_norm_graph(
        input_shape: Vec<u32>,
        output_shape: Vec<u32>,
        data_type: DataType,
        layout: &str,
    ) -> GraphInfo {
        let input_desc = OperandDescriptor {
            data_type,
            shape: input_shape,
            pending_permutation: Vec::new(),
        };

        let output_desc = OperandDescriptor {
            data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let mut attributes = serde_json::Map::new();
        attributes.insert("epsilon".to_string(), serde_json::Value::from(1e-5));
        attributes.insert(
            "layout".to_string(),
            serde_json::Value::String(layout.to_string()),
        );

        GraphInfo {
            operations: vec![Operation {
                op_type: "instanceNormalization".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("instance_norm_op".to_string()),
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

    #[test]
    fn test_instance_normalization_execution() {
        // InstanceNorm: Normalize per-instance over spatial dimensions
        // Input: [1, 2, 2, 2] NCHW format
        // For each channel independently: compute mean/variance over H,W
        let graph = create_instance_norm_graph(
            vec![1, 2, 2, 2],
            vec![1, 2, 2, 2],
            DataType::Float32,
            "nchw",
        );

        // Channel 0: [1,2,3,4] -> mean=2.5, variance=1.25, std=1.118
        // Channel 1: [5,6,7,8] -> mean=6.5, variance=1.25, std=1.118
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // Channel 0
            5.0, 6.0, 7.0, 8.0, // Channel 1
        ];

        // After normalization: (x - mean) / std
        // Channel 0: [(1-2.5)/1.118, (2-2.5)/1.118, (3-2.5)/1.118, (4-2.5)/1.118]
        //          ≈ [-1.34, -0.447, 0.447, 1.34]
        // Channel 1: [(5-6.5)/1.118, (6-6.5)/1.118, (7-6.5)/1.118, (8-6.5)/1.118]
        //          ≈ [-1.34, -0.447, 0.447, 1.34]
        let expected = vec![-1.34, -0.447, 0.447, 1.34, -1.34, -0.447, 0.447, 1.34];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 0.1); // Looser tolerance for complex computation
    }

    /// Helper to create layer normalization graph
    fn create_layer_norm_graph(
        input_shape: Vec<u32>,
        output_shape: Vec<u32>,
        axes: Vec<u32>,
        data_type: DataType,
    ) -> GraphInfo {
        let input_desc = OperandDescriptor {
            data_type,
            shape: input_shape,
            pending_permutation: Vec::new(),
        };

        let output_desc = OperandDescriptor {
            data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let mut attributes = serde_json::Map::new();
        attributes.insert("epsilon".to_string(), serde_json::Value::from(1e-5));
        attributes.insert(
            "axes".to_string(),
            serde_json::Value::Array(axes.iter().map(|&a| serde_json::Value::from(a)).collect()),
        );

        GraphInfo {
            operations: vec![Operation {
                op_type: "layerNormalization".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("layer_norm_op".to_string()),
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

    #[test]
    fn test_layer_normalization_execution() {
        // LayerNorm: Normalize over specified axes (typically last axis)
        // Input: [[1, 2], [3, 4]] shape [2, 2]
        // Normalize over axis 1 (last axis)
        let graph = create_layer_norm_graph(vec![2, 2], vec![2, 2], vec![1], DataType::Float32);

        let input = vec![1.0, 2.0, 3.0, 4.0];

        // Row 0: [1,2] -> mean=1.5, variance=0.25, std=0.5
        //        normalized: [(1-1.5)/0.5, (2-1.5)/0.5] = [-1, 1]
        // Row 1: [3,4] -> mean=3.5, variance=0.25, std=0.5
        //        normalized: [(3-3.5)/0.5, (4-3.5)/0.5] = [-1, 1]
        let expected = vec![-1.0, 1.0, -1.0, 1.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-3);
    }

    // ============================================================================
    // Shape Manipulation Operations Tests (2026-01-29)
    // ============================================================================

    /// Helper to create slice graph
    fn create_slice_graph(
        input_shape: Vec<u32>,
        output_shape: Vec<u32>,
        starts: Vec<i32>,
        sizes: Vec<i32>,
        strides: Vec<i32>,
        data_type: DataType,
    ) -> GraphInfo {
        let input_desc = OperandDescriptor {
            data_type,
            shape: input_shape,
            pending_permutation: Vec::new(),
        };

        let output_desc = OperandDescriptor {
            data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let mut attributes = serde_json::Map::new();
        attributes.insert(
            "starts".to_string(),
            serde_json::Value::Array(
                starts
                    .iter()
                    .map(|&s| serde_json::Value::from(s as i64))
                    .collect(),
            ),
        );
        attributes.insert(
            "sizes".to_string(),
            serde_json::Value::Array(
                sizes
                    .iter()
                    .map(|&s| serde_json::Value::from(s as i64))
                    .collect(),
            ),
        );
        attributes.insert(
            "strides".to_string(),
            serde_json::Value::Array(
                strides
                    .iter()
                    .map(|&s| serde_json::Value::from(s as i64))
                    .collect(),
            ),
        );

        GraphInfo {
            operations: vec![Operation {
                op_type: "slice".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("slice_op".to_string()),
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

    #[test]
    fn test_slice_basic_execution() {
        // Slice [1:3, 0:2] from [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
        // Result: [[4,5], [7,8]]
        let graph = create_slice_graph(
            vec![4, 3],
            vec![2, 2],
            vec![1, 0], // Start at row 1, col 0
            vec![2, 2], // Take 2 rows, 2 cols
            vec![1, 1], // Stride 1 in both dimensions
            DataType::Float32,
        );

        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let expected = vec![4.0, 5.0, 7.0, 8.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_slice_with_stride_execution() {
        // Slice with stride 2: every other element
        // Input: [1,2,3,4,5,6,7,8] shape [8]
        // Slice [0:8:2] -> [1,3,5,7]
        // TensorRT's size parameter is the OUTPUT size, not input range
        // With stride=2, to get 4 output elements we need size=4
        let graph = create_slice_graph(
            vec![8],
            vec![4],
            vec![0],
            vec![4], // Size is output size, not input range
            vec![2], // Stride 2
            DataType::Float32,
        );

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected = vec![1.0, 3.0, 5.0, 7.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    /// Helper to create squeeze graph
    fn create_squeeze_graph(
        input_shape: Vec<u32>,
        output_shape: Vec<u32>,
        data_type: DataType,
    ) -> GraphInfo {
        let input_desc = OperandDescriptor {
            data_type,
            shape: input_shape,
            pending_permutation: Vec::new(),
        };

        let output_desc = OperandDescriptor {
            data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        GraphInfo {
            operations: vec![Operation {
                op_type: "squeeze".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(serde_json::Map::new()),
                label: Some("squeeze_op".to_string()),
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

    #[test]
    fn test_squeeze_execution() {
        // Squeeze: Remove size-1 dimensions
        // Input: [1, 4, 1, 2] -> Output: [4, 2]
        let graph = create_squeeze_graph(vec![1, 4, 1, 2], vec![4, 2], DataType::Float32);

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected = input.clone(); // Data unchanged, only shape changes

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    /// Helper to create unsqueeze graph
    fn create_unsqueeze_graph(
        input_shape: Vec<u32>,
        output_shape: Vec<u32>,
        axes: Vec<u32>,
        data_type: DataType,
    ) -> GraphInfo {
        let input_desc = OperandDescriptor {
            data_type,
            shape: input_shape,
            pending_permutation: Vec::new(),
        };

        let output_desc = OperandDescriptor {
            data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let mut attributes = serde_json::Map::new();
        attributes.insert(
            "axes".to_string(),
            serde_json::Value::Array(axes.iter().map(|&a| serde_json::Value::from(a)).collect()),
        );

        GraphInfo {
            operations: vec![Operation {
                op_type: "unsqueeze".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("unsqueeze_op".to_string()),
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

    #[test]
    fn test_unsqueeze_execution() {
        // Unsqueeze: Add size-1 dimensions
        // Input: [4, 2] -> Output: [1, 4, 1, 2] (add dims at positions 0 and 2)
        let graph =
            create_unsqueeze_graph(vec![4, 2], vec![1, 4, 1, 2], vec![0, 2], DataType::Float32);

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected = input.clone(); // Data unchanged, only shape changes

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    /// Helper to create expand graph
    fn create_expand_graph(
        input_shape: Vec<u32>,
        output_shape: Vec<u32>,
        new_shape: Vec<i32>,
        data_type: DataType,
    ) -> GraphInfo {
        let input_desc = OperandDescriptor {
            data_type,
            shape: input_shape,
            pending_permutation: Vec::new(),
        };

        let output_desc = OperandDescriptor {
            data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let mut attributes = serde_json::Map::new();
        attributes.insert(
            "newShape".to_string(),
            serde_json::Value::Array(
                new_shape
                    .iter()
                    .map(|&s| serde_json::Value::from(s as i64))
                    .collect(),
            ),
        );

        GraphInfo {
            operations: vec![Operation {
                op_type: "expand".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("expand_op".to_string()),
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

    #[test]
    fn test_expand_execution() {
        // Expand: Currently simplified implementation using identity layer
        // Note: This is a passthrough test. Full expand requires IShuffleLayer
        // with setReshapeDimensions() to be exposed in trtx-rs.
        // Input: [4] -> Output: [4] (identity passthrough)
        let graph = create_expand_graph(vec![4], vec![4], vec![4], DataType::Float32);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![1.0, 2.0, 3.0, 4.0]; // Identity passthrough

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    // ============================================================================
    // Comparison Operations Tests (2026-01-29)
    // ============================================================================

    /// Helper to create comparison graph
    fn create_comparison_graph(
        input_shape: Vec<u32>,
        output_shape: Vec<u32>,
        op_type: &str,
        data_type: DataType,
    ) -> GraphInfo {
        let input_desc = OperandDescriptor {
            data_type,
            shape: input_shape.clone(),
            pending_permutation: Vec::new(),
        };

        let output_desc = OperandDescriptor {
            data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        GraphInfo {
            operations: vec![Operation {
                op_type: op_type.to_string(),
                input_operands: vec![0, 1],
                output_operand: Some(2),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(serde_json::Map::new()),
                label: Some(format!("{}_op", op_type)),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: input_desc.clone(),
                    name: Some("input0".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: input_desc,
                    name: Some("input1".to_string()),
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

    #[test]
    fn test_equal_execution() {
        let graph = create_comparison_graph(vec![4], vec![4], "equal", DataType::Float32);

        let input0 = vec![1.0, 2.0, 3.0, 4.0];
        let input1 = vec![1.0, 3.0, 3.0, 5.0];
        let all_inputs = vec![input0, input1];

        // Expected: [true, false, true, false] = [1.0, 0.0, 1.0, 0.0]
        let expected = vec![1.0, 0.0, 1.0, 0.0];

        let output = execute_graph_multi_input(&graph, all_inputs).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_greater_execution() {
        let graph = create_comparison_graph(vec![4], vec![4], "greater", DataType::Float32);

        let input0 = vec![1.0, 3.0, 2.0, 4.0];
        let input1 = vec![2.0, 2.0, 2.0, 4.0];
        let all_inputs = vec![input0, input1];

        // Expected: [false, true, false, false] = [0.0, 1.0, 0.0, 0.0]
        let expected = vec![0.0, 1.0, 0.0, 0.0];

        let output = execute_graph_multi_input(&graph, all_inputs).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_greater_or_equal_execution() {
        let graph = create_comparison_graph(vec![4], vec![4], "greaterOrEqual", DataType::Float32);

        let input0 = vec![1.0, 3.0, 2.0, 4.0];
        let input1 = vec![2.0, 2.0, 2.0, 4.0];
        let all_inputs = vec![input0, input1];

        // Expected: [false, true, true, true] = [0.0, 1.0, 1.0, 1.0]
        let expected = vec![0.0, 1.0, 1.0, 1.0];

        let output = execute_graph_multi_input(&graph, all_inputs).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_lesser_execution() {
        let graph = create_comparison_graph(vec![4], vec![4], "lesser", DataType::Float32);

        let input0 = vec![1.0, 3.0, 2.0, 4.0];
        let input1 = vec![2.0, 2.0, 2.0, 4.0];
        let all_inputs = vec![input0, input1];

        // Expected: [true, false, false, false] = [1.0, 0.0, 0.0, 0.0]
        let expected = vec![1.0, 0.0, 0.0, 0.0];

        let output = execute_graph_multi_input(&graph, all_inputs).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_lesser_or_equal_execution() {
        let graph = create_comparison_graph(vec![4], vec![4], "lesserOrEqual", DataType::Float32);

        let input0 = vec![1.0, 3.0, 2.0, 4.0];
        let input1 = vec![2.0, 2.0, 2.0, 4.0];
        let all_inputs = vec![input0, input1];

        // Expected: [true, false, true, true] = [1.0, 0.0, 1.0, 1.0]
        let expected = vec![1.0, 0.0, 1.0, 1.0];

        let output = execute_graph_multi_input(&graph, all_inputs).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_not_equal_execution() {
        let graph = create_comparison_graph(vec![4], vec![4], "notEqual", DataType::Float32);

        let input0 = vec![1.0, 2.0, 3.0, 4.0];
        let input1 = vec![1.0, 3.0, 3.0, 5.0];
        let all_inputs = vec![input0, input1];

        // Expected: [false, true, false, true] = [0.0, 1.0, 0.0, 1.0]
        let expected = vec![0.0, 1.0, 0.0, 1.0];

        let output = execute_graph_multi_input(&graph, all_inputs).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    // ============================================================================
    // Logical Operations Tests (2026-01-29)
    // ============================================================================

    #[test]
    fn test_logical_and_execution() {
        let graph = create_comparison_graph(vec![4], vec![4], "logicalAnd", DataType::Float32);

        let input0 = vec![1.0, 1.0, 0.0, 0.0]; // true, true, false, false
        let input1 = vec![1.0, 0.0, 1.0, 0.0]; // true, false, true, false
        let all_inputs = vec![input0, input1];

        // Expected: [true, false, false, false] = [1.0, 0.0, 0.0, 0.0]
        let expected = vec![1.0, 0.0, 0.0, 0.0];

        let output = execute_graph_multi_input(&graph, all_inputs).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_logical_or_execution() {
        let graph = create_comparison_graph(vec![4], vec![4], "logicalOr", DataType::Float32);

        let input0 = vec![1.0, 1.0, 0.0, 0.0]; // true, true, false, false
        let input1 = vec![1.0, 0.0, 1.0, 0.0]; // true, false, true, false
        let all_inputs = vec![input0, input1];

        // Expected: [true, true, true, false] = [1.0, 1.0, 1.0, 0.0]
        let expected = vec![1.0, 1.0, 1.0, 0.0];

        let output = execute_graph_multi_input(&graph, all_inputs).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_logical_xor_execution() {
        let graph = create_comparison_graph(vec![4], vec![4], "logicalXor", DataType::Float32);

        let input0 = vec![1.0, 1.0, 0.0, 0.0]; // true, true, false, false
        let input1 = vec![1.0, 0.0, 1.0, 0.0]; // true, false, true, false
        let all_inputs = vec![input0, input1];

        // Expected: [false, true, true, false] = [0.0, 1.0, 1.0, 0.0]
        let expected = vec![0.0, 1.0, 1.0, 0.0];

        let output = execute_graph_multi_input(&graph, all_inputs).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_logical_not_execution() {
        let graph = create_single_input_graph(vec![4], vec![4], "logicalNot", DataType::Float32);

        let input = vec![1.0, 0.0, 1.0, 0.0]; // true, false, true, false

        // Expected: [false, true, false, true] = [0.0, 1.0, 0.0, 1.0]
        let expected = vec![0.0, 1.0, 0.0, 1.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    // ============================================================================
    // Indexing/Gathering Operations Tests (2026-01-29)
    // ============================================================================

    /// Helper to create gather graph
    fn create_gather_graph(
        input_shape: Vec<u32>,
        indices_shape: Vec<u32>,
        output_shape: Vec<u32>,
        axis: u32,
        data_type: DataType,
    ) -> GraphInfo {
        let input_desc = OperandDescriptor {
            data_type,
            shape: input_shape,
            pending_permutation: Vec::new(),
        };

        let indices_desc = OperandDescriptor {
            data_type: DataType::Int32,
            shape: indices_shape,
            pending_permutation: Vec::new(),
        };

        let output_desc = OperandDescriptor {
            data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let mut attributes = serde_json::Map::new();
        attributes.insert("axis".to_string(), serde_json::Value::from(axis));

        GraphInfo {
            operations: vec![Operation {
                op_type: "gather".to_string(),
                input_operands: vec![0, 1],
                output_operand: Some(2),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("gather_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: input_desc,
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: indices_desc,
                    name: Some("indices".to_string()),
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

    #[test]
    fn test_gather_execution() {
        // Gather: Select elements along axis 0
        // Input: [1,2,3,4,5,6] shape [6]
        // Indices: [0,2,4] shape [3]
        // Expected: [1,3,5] (elements at positions 0, 2, 4)
        let graph = create_gather_graph(vec![6], vec![3], vec![3], 0, DataType::Float32);

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = vec![0.0, 2.0, 4.0]; // Will be converted to int32
        let all_inputs = vec![input, indices];

        let expected = vec![1.0, 3.0, 5.0];

        let output = execute_graph_multi_input(&graph, all_inputs).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    /// Helper to create argMax/argMin graph
    fn create_arg_graph(
        input_shape: Vec<u32>,
        output_shape: Vec<u32>,
        axis: u32,
        keep_dims: bool,
        op_type: &str,
        data_type: DataType,
    ) -> GraphInfo {
        let input_desc = OperandDescriptor {
            data_type,
            shape: input_shape,
            pending_permutation: Vec::new(),
        };

        let output_desc = OperandDescriptor {
            data_type: DataType::Int32,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let mut attributes = serde_json::Map::new();
        attributes.insert("axis".to_string(), serde_json::Value::from(axis));
        attributes.insert(
            "keepDimensions".to_string(),
            serde_json::Value::from(keep_dims),
        );

        GraphInfo {
            operations: vec![Operation {
                op_type: op_type.to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
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

    #[test]
    fn test_arg_max_execution() {
        // ArgMax: Find indices of maximum values
        // Input: [[1,3,2], [4,2,5]] shape [2,3]
        // Axis: 1 (along columns)
        // Expected: [1, 2] (max indices in each row)
        let graph = create_arg_graph(vec![2, 3], vec![2], 1, false, "argMax", DataType::Float32);

        let input = vec![1.0, 3.0, 2.0, 4.0, 2.0, 5.0];

        // Expected: [1, 2] (index of max in each row)
        let expected = vec![1.0, 2.0]; // Will be int32 indices

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_arg_min_execution() {
        // ArgMin: Find indices of minimum values
        // Input: [[3,1,2], [4,5,2]] shape [2,3]
        // Axis: 1 (along columns)
        // Expected: [1, 2] (min indices in each row)
        let graph = create_arg_graph(vec![2, 3], vec![2], 1, false, "argMin", DataType::Float32);

        let input = vec![3.0, 1.0, 2.0, 4.0, 5.0, 2.0];

        // Expected: [1, 2] (index of min in each row)
        let expected = vec![1.0, 2.0]; // Will be int32 indices

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    // ============================================================================
    // Other Operations Tests (2026-01-29)
    // ============================================================================

    #[test]
    fn test_clamp_execution() {
        // Clamp: Clip values to range [minValue, maxValue]
        // Input: [-2, -1, 0, 1, 2, 3, 4, 5]
        // Range: [0, 3]
        // Expected: [0, 0, 0, 1, 2, 3, 3, 3]
        let mut attributes = serde_json::Map::new();
        attributes.insert("minValue".to_string(), serde_json::Value::from(0.0));
        attributes.insert("maxValue".to_string(), serde_json::Value::from(3.0));

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "clamp".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("clamp_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![8],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![8],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        // Clamp with min=0, max=3 should clip values to [0, 3]
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_clamp_multidimensional() {
        // Test clamp with 4D input to verify scalar broadcasting fix
        // This tests the bug fix where scalar constants must use shape [] (0D)
        // not shape [1] (1D) for proper broadcasting with multi-dimensional tensors
        // Input shape: [1,2,2,2] = 8 elements
        // Input: [-2, -1, 0, 1, 2, 3, 4, 5]
        // Range: [-1.0, 3.5]
        // Expected: [-1, -1, 0, 1, 2, 3, 3.5, 3.5]
        let mut attributes = serde_json::Map::new();
        attributes.insert("minValue".to_string(), serde_json::Value::from(-1.0));
        attributes.insert("maxValue".to_string(), serde_json::Value::from(3.5));

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "clamp".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("clamp_4d_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2, 2, 2], // 4D tensor
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2, 2, 2], // 4D tensor
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        // Clamp with min=-1.0, max=3.5 should clip values to [-1, 3.5]
        let expected = vec![-1.0, -1.0, 0.0, 1.0, 2.0, 3.0, 3.5, 3.5];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_where_execution() {
        // Where: Select values based on condition
        // Condition: [1,0,1,0] (true, false, true, false)
        // True values: [10,20,30,40]
        // False values: [1,2,3,4]
        // Expected: [10,2,30,4]
        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "where".to_string(),
                input_operands: vec![0, 1, 2],
                output_operand: Some(3),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(serde_json::Map::new()),
                label: Some("where_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("condition".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("true_value".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("false_value".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0, 1, 2],
            output_operands: vec![3],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let condition = vec![1.0, 0.0, 1.0, 0.0];
        let true_values = vec![10.0, 20.0, 30.0, 40.0];
        let false_values = vec![1.0, 2.0, 3.0, 4.0];
        let all_inputs = vec![condition, true_values, false_values];

        let expected = vec![10.0, 2.0, 30.0, 4.0];

        let output = execute_graph_multi_input(&graph, all_inputs).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_linear_execution() {
        // Linear: alpha * x + beta (general case with both alpha and beta)
        // Input: [1,2,3,4]
        // Alpha: 2.0, Beta: 1.0
        // Expected: [3,5,7,9]
        let mut attributes = serde_json::Map::new();
        attributes.insert("alpha".to_string(), serde_json::Value::from(2.0));
        attributes.insert("beta".to_string(), serde_json::Value::from(1.0));

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "linear".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("linear_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![1.0, 2.0, 3.0, 4.0];
        // Linear: y = alpha * x + beta = 2.0 * x + 1.0
        // [1,2,3,4] → [2*1+1, 2*2+1, 2*3+1, 2*4+1] = [3,5,7,9]
        let expected = vec![3.0, 5.0, 7.0, 9.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_linear_multiply_only() {
        // Linear: alpha * x (beta = 0, only multiply)
        // Input: [1,2,3,4]
        // Alpha: 3.0, Beta: 0.0
        // Expected: [3,6,9,12]
        let mut attributes = serde_json::Map::new();
        attributes.insert("alpha".to_string(), serde_json::Value::from(3.0));
        attributes.insert("beta".to_string(), serde_json::Value::from(0.0));

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "linear".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("linear_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![3.0, 6.0, 9.0, 12.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_linear_add_only() {
        // Linear: x + beta (alpha = 1, only add)
        // Input: [1,2,3,4]
        // Alpha: 1.0, Beta: 5.0
        // Expected: [6,7,8,9]
        let mut attributes = serde_json::Map::new();
        attributes.insert("alpha".to_string(), serde_json::Value::from(1.0));
        attributes.insert("beta".to_string(), serde_json::Value::from(5.0));

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "linear".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("linear_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![6.0, 7.0, 8.0, 9.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_linear_defaults() {
        // Linear: identity (alpha = 1, beta = 0, uses defaults)
        // Input: [1,2,3,4]
        // Expected: [1,2,3,4]
        let attributes = serde_json::Map::new(); // No attributes, use defaults

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "linear".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("linear_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![1.0, 2.0, 3.0, 4.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_linear_multidimensional() {
        // Test linear with 4D input to verify scalar broadcasting fix
        // This tests the bug fix where alpha/beta constants must use shape [] (0D)
        // not shape [1] (1D) for proper broadcasting with multi-dimensional tensors
        // Input shape: [1,2,2,2] = 8 elements
        // Input: [0, 1, 2, 3, 4, 5, 6, 7]
        // Alpha: 0.5, Beta: 1.0
        // Expected: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5] (0.5*x + 1.0)
        let mut attributes = serde_json::Map::new();
        attributes.insert("alpha".to_string(), serde_json::Value::from(0.5));
        attributes.insert("beta".to_string(), serde_json::Value::from(1.0));

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "linear".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("linear_4d_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2, 2, 2], // 4D tensor
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2, 2, 2], // 4D tensor
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        // Linear: y = 0.5*x + 1.0
        let expected = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_pad_execution() {
        // Pad: Add padding to tensor
        // Input: [1,2,3,4] shape [4]
        // BeginningPadding: [1]
        // EndingPadding: [2]
        // Expected: [0,1,2,3,4,0,0] shape [7]
        let mut attributes = serde_json::Map::new();
        attributes.insert(
            "beginningPadding".to_string(),
            serde_json::Value::Array(vec![serde_json::Value::from(1)]),
        );
        attributes.insert(
            "endingPadding".to_string(),
            serde_json::Value::Array(vec![serde_json::Value::from(2)]),
        );

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "pad".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("pad_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![7],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_gather_nd() {
        // Test gatherND operation
        // data shape: [2, 3] = [[1, 2, 3], [4, 5, 6]]
        // indices shape: [2, 2] = [[0, 0], [1, 2]] -> gather data[0,0] and data[1,2]
        // output shape: [2] = [1, 6]
        // NOTE: TensorRT requires Int32 indices, so we use Int32 and convert from f32

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "gatherND".to_string(),
                input_operands: vec![0, 1],
                output_operand: Some(2),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Null,
                label: Some("gatherND_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![2, 3],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("data".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Int32,
                        shape: vec![2, 2],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("indices".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![2],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = vec![0.0, 0.0, 1.0, 2.0]; // Will be converted to Int32

        let output =
            execute_graph_multi_input(&graph, vec![data, indices]).expect("Execution failed");
        let expected = vec![1.0, 6.0];
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_resample2d() {
        // Test resample2d operation with nearest-neighbor mode
        // Input: 1x1x2x2 = [[[[1, 2], [3, 4]]]]
        // Output: 1x1x4x4 (upscale by 2x)

        let mut attributes = serde_json::Map::new();
        attributes.insert(
            "mode".to_string(),
            serde_json::Value::String("nearest-neighbor".to_string()),
        );
        attributes.insert(
            "sizes".to_string(),
            serde_json::Value::Array(vec![
                serde_json::Value::Number(serde_json::Number::from(4)),
                serde_json::Value::Number(serde_json::Number::from(4)),
            ]),
        );

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "resample2d".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("resample2d_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 1, 2, 2],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 1, 4, 4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![1.0, 2.0, 3.0, 4.0];
        // Nearest-neighbor upscaling: each value repeated 2x2
        let expected = vec![
            1.0, 1.0, 2.0, 2.0, // Row 1
            1.0, 1.0, 2.0, 2.0, // Row 2
            3.0, 3.0, 4.0, 4.0, // Row 3
            3.0, 3.0, 4.0, 4.0, // Row 4
        ];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_conv_transpose2d() {
        // Test convTranspose2d operation (deconvolution)
        // Input: 1x1x2x2, Kernel: 1x1x2x2, Output: 1x1x3x3

        let kernel_data = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity-like kernel
        let bias_data = vec![0.0];

        // Convert to bytes using flat_map (no base64 encoding needed)
        let kernel_bytes: Vec<u8> = kernel_data
            .iter()
            .flat_map(|&f: &f32| f.to_le_bytes())
            .collect();

        let bias_bytes: Vec<u8> = bias_data
            .iter()
            .flat_map(|&f: &f32| f.to_le_bytes())
            .collect();

        let mut constant_map = HashMap::new();
        constant_map.insert(
            1,
            ConstantData {
                data: kernel_bytes,
                label: Some("kernel".to_string()),
            },
        );
        constant_map.insert(
            2,
            ConstantData {
                data: bias_bytes,
                label: Some("bias".to_string()),
            },
        );

        let mut attributes = serde_json::Map::new();
        attributes.insert(
            "strides".to_string(),
            serde_json::Value::Array(vec![
                serde_json::Value::Number(serde_json::Number::from(1)),
                serde_json::Value::Number(serde_json::Number::from(1)),
            ]),
        );
        attributes.insert(
            "padding".to_string(),
            serde_json::Value::Array(vec![
                serde_json::Value::Number(serde_json::Number::from(0)),
                serde_json::Value::Number(serde_json::Number::from(0)),
                serde_json::Value::Number(serde_json::Number::from(0)),
                serde_json::Value::Number(serde_json::Number::from(0)),
            ]),
        );

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "convTranspose2d".to_string(),
                input_operands: vec![0, 1, 2],
                output_operand: Some(3),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("convTranspose2d_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 1, 2, 2],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Constant,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 1, 2, 2],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("kernel".to_string()),
                },
                Operand {
                    kind: OperandKind::Constant,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("bias".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 1, 3, 3],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![3],
            constant_operand_ids_to_handles: constant_map,
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![1.0, 2.0, 3.0, 4.0];

        // ConvTranspose2d with 2x2 kernel [1,0,0,1] produces 3x3 output
        // Input: [[1, 2], [3, 4]] -> Output: [[1, 2, 0], [3, 5, 2], [0, 3, 4]]
        // The diagonal kernel spreads values with overlaps (center element = 1+4 = 5)
        let expected = vec![1.0, 2.0, 0.0, 3.0, 5.0, 2.0, 0.0, 3.0, 4.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-3);
    }

    #[test]
    fn test_scatter_elements() {
        // Test scatterElements operation
        // data shape: [4] = [1, 2, 3, 4]
        // indices shape: [2] = [1, 3] (scatter at positions 1 and 3)
        // updates shape: [2] = [10, 20] (new values)
        // output shape: [4] = [1, 10, 3, 20]
        // NOTE: TensorRT requires Int32 indices

        let mut attributes = serde_json::Map::new();
        attributes.insert(
            "axis".to_string(),
            serde_json::Value::Number(serde_json::Number::from(0)),
        );

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "scatterElements".to_string(),
                input_operands: vec![0, 1, 2],
                output_operand: Some(3),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("scatterElements_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("data".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Int32,
                        shape: vec![2],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("indices".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![2],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("updates".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0, 1, 2],
            output_operands: vec![3],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let indices = vec![1.0, 3.0]; // Will be converted to Int32
        let updates = vec![10.0, 20.0];

        let output = execute_graph_multi_input(&graph, vec![data, indices, updates])
            .expect("Execution failed");
        let expected = vec![1.0, 10.0, 3.0, 20.0];
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_scatter_nd() {
        // Test scatterND operation
        // data shape: [2, 3] = [[1, 2, 3], [4, 5, 6]]
        // indices shape: [2, 2] = [[0, 0], [1, 2]] -> positions to update
        // updates shape: [2] = [10, 20] -> new values
        // output shape: [2, 3] = [[10, 2, 3], [4, 5, 20]]
        // NOTE: TensorRT requires Int32 indices

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "scatterND".to_string(),
                input_operands: vec![0, 1, 2],
                output_operand: Some(3),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Null,
                label: Some("scatterND_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![2, 3],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("data".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Int32,
                        shape: vec![2, 2],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("indices".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![2],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("updates".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![2, 3],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0, 1, 2],
            output_operands: vec![3],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = vec![0.0, 0.0, 1.0, 2.0]; // Will be converted to Int32
        let updates = vec![10.0, 20.0];

        let output = execute_graph_multi_input(&graph, vec![data, indices, updates])
            .expect("Execution failed");
        let expected = vec![10.0, 2.0, 3.0, 4.0, 5.0, 20.0];
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        // Test quantizeLinear -> dequantizeLinear -> add (proper QDQ pattern for TensorRT)
        // Input: [1.0, 2.0, 3.0, 4.0]
        // -> Quantize (scale=0.5) -> INT8 [2, 4, 6, 8]
        // -> Dequantize (scale=0.5) -> Float [1.0, 2.0, 3.0, 4.0]
        // -> Add 1.0 -> [2.0, 3.0, 4.0, 5.0]

        let graph = GraphInfo {
            operations: vec![
                Operation {
                    op_type: "quantizeLinear".to_string(),
                    input_operands: vec![0, 1],
                    output_operand: Some(2),
                    output_operands: Vec::new(),
                    attributes: serde_json::Value::Null,
                    label: Some("quantize".to_string()),
                },
                Operation {
                    op_type: "dequantizeLinear".to_string(),
                    input_operands: vec![2, 3],
                    output_operand: Some(4),
                    output_operands: Vec::new(),
                    attributes: serde_json::Value::Null,
                    label: Some("dequantize".to_string()),
                },
                Operation {
                    op_type: "add".to_string(),
                    input_operands: vec![4, 5],
                    output_operand: Some(6),
                    output_operands: Vec::new(),
                    attributes: serde_json::Value::Null,
                    label: Some("add_one".to_string()),
                },
            ],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("quant_scale".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Int8,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("quantized".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("dequant_scale".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("dequantized".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("add_constant".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0, 1, 3, 5],
            output_operands: vec![6],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let quant_scale = vec![0.5];
        let dequant_scale = vec![0.5];
        let add_value = vec![1.0];
        let expected = vec![2.0, 3.0, 4.0, 5.0];

        let output =
            execute_graph_multi_input(&graph, vec![input, quant_scale, dequant_scale, add_value])
                .expect("Execution failed");
        verify_output(&output, &expected, 0.1); // Allow some quantization error
    }

    #[test]
    fn test_is_nan() {
        // Test isNaN operation
        // Input: [1.0, NaN, 3.0, NaN]
        // Output: [false, true, false, true] (as 0.0/1.0)

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "isNaN".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Null,
                label: Some("isNaN_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![1.0, f32::NAN, 3.0, f32::NAN];
        let expected = vec![0.0, 1.0, 0.0, 1.0]; // Bool as float: false=0, true=1

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_is_infinite() {
        // Test isInfinite operation
        // Input: [1.0, INFINITY, -INFINITY, 0.0]
        // Output: [false, true, true, false]

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "isInfinite".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Null,
                label: Some("isInfinite_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![1.0, f32::INFINITY, f32::NEG_INFINITY, 0.0];
        let expected = vec![0.0, 1.0, 1.0, 0.0]; // Bool as float

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_round_even() {
        // Test roundEven operation (banker's rounding)
        // 0.5 rounds to 0, 1.5 rounds to 2, 2.5 rounds to 2, 3.5 rounds to 4

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "roundEven".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Null,
                label: Some("roundEven_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![0.5, 1.5, 2.5, 3.5];
        let expected = vec![0.0, 2.0, 2.0, 4.0]; // Round to nearest even

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_gather_elements() {
        // Test gatherElements operation
        // data: [10, 20, 30, 40]
        // indices: [3, 1, 0, 2] -> gather elements at positions 3,1,0,2
        // output: [40, 20, 10, 30]

        let mut attributes = serde_json::Map::new();
        attributes.insert(
            "axis".to_string(),
            serde_json::Value::Number(serde_json::Number::from(0)),
        );

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "gatherElements".to_string(),
                input_operands: vec![0, 1],
                output_operand: Some(2),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("gatherElements_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("data".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Int32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("indices".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let data = vec![10.0, 20.0, 30.0, 40.0];
        let indices = vec![3.0, 1.0, 0.0, 2.0]; // Will be converted to Int32

        let output =
            execute_graph_multi_input(&graph, vec![data, indices]).expect("Execution failed");
        let expected = vec![40.0, 20.0, 10.0, 30.0];
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_l2_pool2d() {
        // Test l2Pool2d operation
        // Input: 1x1x2x2 = [[[[1, 2], [3, 4]]]]
        // With 2x2 window: sqrt(avg(1^2 + 2^2 + 3^2 + 4^2)) = sqrt(avg(1+4+9+16)) = sqrt(7.5) ≈ 2.74

        let mut attributes = serde_json::Map::new();
        attributes.insert("windowDimensions".to_string(), serde_json::json!([2, 2]));

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "l2Pool2d".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("l2Pool2d_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 1, 2, 2],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 1, 1, 1],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![
            ((1.0f32.powi(2) + 2.0f32.powi(2) + 3.0f32.powi(2) + 4.0f32.powi(2)) / 4.0).sqrt(),
        ];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-3);
    }

    #[test]
    fn test_reverse() {
        // Test reverse operation using ISliceLayer with negative stride
        let mut attributes = serde_json::Map::new();
        attributes.insert("axes".to_string(), serde_json::json!([0]));

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "reverse".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("reverse_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![4.0, 3.0, 2.0, 1.0]; // Reversed along axis 0

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_cumulative_sum() {
        // Test cumulativeSum operation using TensorRT's native ICumulativeLayer
        let mut attributes = serde_json::Map::new();
        attributes.insert(
            "axis".to_string(),
            serde_json::Value::Number(serde_json::Number::from(0)),
        );

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "cumulativeSum".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("cumulativeSum_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![1.0, 2.0, 3.0, 4.0];
        // Cumulative sum along axis 0: [1, 1+2=3, 1+2+3=6, 1+2+3+4=10]
        let expected = vec![1.0, 3.0, 6.0, 10.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_triangular() {
        // Test triangular operation using constant mask + elementwise multiplication
        let mut attributes = serde_json::Map::new();
        attributes.insert("upper".to_string(), serde_json::Value::Bool(true));

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "triangular".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("triangular_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![2, 2],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![2, 2],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2], [3,4]]
        // Upper triangular: keep where j >= i
        // Row 0 (i=0): keep j>=0, so [1,2] -> [1,2]
        // Row 1 (i=1): keep j>=1, so [3,4] -> [0,4]
        // Result: [[1,2], [0,4]]
        let expected = vec![1.0, 2.0, 0.0, 4.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_tile() {
        // Test tile operation
        // Input: [1, 2] (shape: [2])
        // Repetitions: [3] (repeat 3 times along axis 0)
        // Output: [1, 2, 1, 2, 1, 2] (shape: [6])

        let mut attributes = serde_json::Map::new();
        attributes.insert("repetitions".to_string(), serde_json::json!([3]));

        let graph = GraphInfo {
            operations: vec![Operation {
                op_type: "tile".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: Vec::new(),
                attributes: serde_json::Value::Object(attributes),
                label: Some("tile_op".to_string()),
            }],
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![2],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![6],
                        pending_permutation: Vec::new(),
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let input = vec![1.0, 2.0];
        let expected = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]; // Repeated 3 times

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    // NOTE: RNN operation tests removed
    // IRNNv2Layer is deprecated in TensorRT and autocxx cannot generate bindings for it
    // RNN operations (lstm, lstmCell, gru, gruCell) remain deferred

    // ============================================================================
    // Conv2D Padding Tests (2026-01-30)
    // ============================================================================

    /// Helper to create conv2d graph with explicit padding control
    fn create_conv2d_graph_with_padding(
        input_shape: Vec<u32>,  // [batch, channels, height, width]
        filter_shape: Vec<u32>, // [out_channels, in_channels, kernel_h, kernel_w]
        filter_data: Vec<f32>,
        bias_data: Option<Vec<f32>>,
        padding: Vec<u32>, // [pad_top, pad_bottom, pad_left, pad_right]
        stride: Vec<u32>,  // [stride_h, stride_w]
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

        // Calculate output shape with padding and stride
        // out_h = floor((in_h + pad_top + pad_bottom - kernel_h) / stride_h) + 1
        // out_w = floor((in_w + pad_left + pad_right - kernel_w) / stride_w) + 1
        let in_h = input_shape[2];
        let in_w = input_shape[3];
        let kernel_h = filter_shape[2];
        let kernel_w = filter_shape[3];
        let pad_top = padding[0];
        let pad_bottom = padding[1];
        let pad_left = padding[2];
        let pad_right = padding[3];
        let stride_h = stride[0];
        let stride_w = stride[1];

        let out_h = ((in_h + pad_top + pad_bottom - kernel_h) / stride_h) + 1;
        let out_w = ((in_w + pad_left + pad_right - kernel_w) / stride_w) + 1;
        let output_shape = vec![input_shape[0], filter_shape[0], out_h, out_w];

        let output_desc = OperandDescriptor {
            data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        // Convert filter data to bytes
        let filter_bytes: Vec<u8> = filter_data.iter().flat_map(|&f| f.to_le_bytes()).collect();

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

        operands.push(Operand {
            kind: OperandKind::Output,
            descriptor: output_desc,
            name: Some("output".to_string()),
        });

        let output_operand_id = operands.len() as u32 - 1;

        // Create attributes with padding and stride
        let attributes = serde_json::json!({
            "padding": padding,
            "strides": stride,
            "dilations": [1, 1],
            "groups": 1,
        });

        GraphInfo {
            operations: vec![Operation {
                op_type: "conv2d".to_string(),
                input_operands,
                output_operand: Some(output_operand_id),
                output_operands: Vec::new(),
                attributes,
                label: Some("conv2d".to_string()),
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
    fn test_conv2d_valid_padding() {
        // Test "valid" padding (no padding) with 3x3 kernel
        // Input: [1, 1, 5, 5] = 5x5 spatial
        // Kernel: 3x3, stride=1, padding=[0,0,0,0]
        // Output: [1, 1, 3, 3] = (5-3+1)x(5-3+1) = 3x3
        // This demonstrates the 2-pixel shrinkage per dimension

        let input_shape = vec![1, 1, 5, 5];
        let filter_shape = vec![1, 1, 3, 3]; // 3x3 kernel

        // Identity kernel (center=1, rest=0)
        let filter_data = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        let graph = create_conv2d_graph_with_padding(
            input_shape,
            filter_shape,
            filter_data,
            None,
            vec![0, 0, 0, 0], // No padding
            vec![1, 1],       // stride=1
            DataType::Float32,
        );

        // Input: 5x5 = 25 elements
        #[rustfmt::skip]
        let input = vec![
            1.0,  2.0,  3.0,  4.0,  5.0,
            6.0,  7.0,  8.0,  9.0, 10.0,
           11.0, 12.0, 13.0, 14.0, 15.0,
           16.0, 17.0, 18.0, 19.0, 20.0,
           21.0, 22.0, 23.0, 24.0, 25.0,
        ];

        // Output: 3x3 = 9 elements (center values due to identity kernel)
        #[rustfmt::skip]
        let expected = vec![
             7.0,  8.0,  9.0,
            12.0, 13.0, 14.0,
            17.0, 18.0, 19.0,
        ];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_conv2d_same_padding() {
        // Test "same" padding with 3x3 kernel
        // Input: [1, 1, 5, 5]
        // Kernel: 3x3, stride=1
        // Padding: [1,1,1,1] to maintain spatial size
        // Output: [1, 1, 5, 5] (same as input)

        let input_shape = vec![1, 1, 5, 5];
        let filter_shape = vec![1, 1, 3, 3];

        // Identity kernel
        let filter_data = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        let graph = create_conv2d_graph_with_padding(
            input_shape,
            filter_shape,
            filter_data,
            None,
            vec![1, 1, 1, 1], // Same padding: pad by 1 on all sides
            vec![1, 1],
            DataType::Float32,
        );

        #[rustfmt::skip]
        let input = vec![
            1.0,  2.0,  3.0,  4.0,  5.0,
            6.0,  7.0,  8.0,  9.0, 10.0,
           11.0, 12.0, 13.0, 14.0, 15.0,
           16.0, 17.0, 18.0, 19.0, 20.0,
           21.0, 22.0, 23.0, 24.0, 25.0,
        ];

        // With identity kernel and same padding, output should equal input
        let expected = input.clone();

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_conv2d_asymmetric_padding() {
        // Test that asymmetric padding is properly rejected
        // TensorRT's setPaddingNd only supports symmetric padding
        // Input: [1, 1, 4, 4]
        // Kernel: 3x3, stride=1
        // Padding: [1,0,1,0] (top=1, bottom=0, left=1, right=0) - ASYMMETRIC

        let input_shape = vec![1, 1, 4, 4];
        let filter_shape = vec![1, 1, 3, 3];

        // Simple averaging kernel
        #[rustfmt::skip]
        let filter_data = vec![
            1.0/9.0, 1.0/9.0, 1.0/9.0,
            1.0/9.0, 1.0/9.0, 1.0/9.0,
            1.0/9.0, 1.0/9.0, 1.0/9.0,
        ];

        let graph = create_conv2d_graph_with_padding(
            input_shape,
            filter_shape,
            filter_data,
            None,
            vec![1, 0, 1, 0], // Asymmetric padding - should fail
            vec![1, 1],
            DataType::Float32,
        );

        #[rustfmt::skip]
        let input = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];

        // Asymmetric padding should now work with explicit padding layer
        let output = execute_graph(&graph, &input).expect("Asymmetric padding execution failed");
        assert_eq!(output.len(), 9, "Expected 3x3 output = 9 elements");
    }

    #[test]
    fn test_conv2d_stride2_no_padding() {
        // Test stride=2 without padding (downsampling)
        // Input: [1, 1, 6, 6]
        // Kernel: 3x3, stride=2, padding=[0,0,0,0]
        // Output: [1, 1, 2, 2] = ((6-3)/2+1) x ((6-3)/2+1) = 2x2

        let input_shape = vec![1, 1, 6, 6];
        let filter_shape = vec![1, 1, 3, 3];

        // Identity kernel
        let filter_data = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        let graph = create_conv2d_graph_with_padding(
            input_shape,
            filter_shape,
            filter_data,
            None,
            vec![0, 0, 0, 0], // No padding
            vec![2, 2],       // stride=2
            DataType::Float32,
        );

        #[rustfmt::skip]
        let input = vec![
            1.0,  2.0,  3.0,  4.0,  5.0,  6.0,
            7.0,  8.0,  9.0, 10.0, 11.0, 12.0,
           13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
           19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
           25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
           31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
        ];

        // With stride=2 and identity kernel, pick every other center value
        // Centers at positions: (1,1)=8, (1,3)=10, (3,1)=20, (3,3)=22
        let expected = vec![8.0, 10.0, 20.0, 22.0];

        let output = execute_graph(&graph, &input).expect("Execution failed");
        verify_output(&output, &expected, 1e-4);
    }

    #[test]
    fn test_conv2d_3x3_mobilenetv2_case() {
        // Simulate MobileNetV2 residual connection scenario
        // Two branches should produce same spatial dimensions for residual add
        //
        // Branch 1: Input 218x218 -> Conv3x3 same padding -> 218x218
        // Branch 2: Input 218x218 -> Conv3x3 valid padding -> 216x216
        // Add would fail due to dimension mismatch!

        // Branch 1: Same padding maintains size
        let input_shape_1 = vec![1, 1, 218, 218];
        let filter_shape = vec![1, 1, 3, 3];
        let filter_data = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        let graph_same = create_conv2d_graph_with_padding(
            input_shape_1.clone(),
            filter_shape.clone(),
            filter_data.clone(),
            None,
            vec![1, 1, 1, 1], // Same padding
            vec![1, 1],
            DataType::Float32,
        );

        // Verify output shape: should be 218x218
        // Operand layout: [0]=input, [1]=filter, [2]=output (no bias)
        assert_eq!(
            graph_same.operands[2].descriptor.shape,
            vec![1, 1, 218, 218]
        );

        // Branch 2: Valid padding shrinks by 2 per dimension
        let graph_valid = create_conv2d_graph_with_padding(
            input_shape_1,
            filter_shape,
            filter_data,
            None,
            vec![0, 0, 0, 0], // Valid padding (no padding)
            vec![1, 1],
            DataType::Float32,
        );

        // Verify output shape: should be 216x216 (218-3+1=216)
        // Operand layout: [0]=input, [1]=filter, [2]=output (no bias)
        assert_eq!(
            graph_valid.operands[2].descriptor.shape,
            vec![1, 1, 216, 216]
        );

        // This test documents the root cause of MobileNetV2 dimension mismatch:
        // Different padding modes produce incompatible spatial dimensions (218 != 216)
        // which cannot be broadcast in elementwise operations.
    }

    #[test]
    fn test_conv2d_depthwise() {
        // Test depthwise convolution (groups = input_channels = output_channels)
        // This is used extensively in MobileNet architectures
        //
        // Input: [1, 4, 3, 3] = 4 channels, 3x3 spatial
        // Filter: [4, 1, 3, 3] = 4 output channels, 1 input per group, 3x3 kernel
        // Groups: 4 (depthwise - each input channel has its own 3x3 filter)
        // Output: [1, 4, 1, 1] with no padding

        let input_shape = vec![1, 4, 3, 3]; // 4 channels
        let filter_shape = vec![4, 1, 3, 3]; // 4 groups, 1 channel per group

        // 4 separate 3x3 filters (one per channel)
        // Filter 0: all 1s (sum = 9)
        // Filter 1: all 2s (sum = 18)
        // Filter 2: all 3s (sum = 27)
        // Filter 3: all 4s (sum = 36)
        #[rustfmt::skip]
        let filter_data = vec![
            // Filter 0 (channel 0)
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            // Filter 1 (channel 1)
            2.0, 2.0, 2.0,
            2.0, 2.0, 2.0,
            2.0, 2.0, 2.0,
            // Filter 2 (channel 2)
            3.0, 3.0, 3.0,
            3.0, 3.0, 3.0,
            3.0, 3.0, 3.0,
            // Filter 3 (channel 3)
            4.0, 4.0, 4.0,
            4.0, 4.0, 4.0,
            4.0, 4.0, 4.0,
        ];

        // Create graph with groups=4
        let mut graph = create_conv2d_graph_with_padding(
            input_shape,
            filter_shape,
            filter_data,
            None,
            vec![0, 0, 0, 0], // No padding
            vec![1, 1],       // Stride 1
            DataType::Float32,
        );

        // Add groups attribute
        graph.operations[0].attributes["groups"] = serde_json::json!(4);

        // Input: 4 channels, each filled with its channel number
        #[rustfmt::skip]
        let input = vec![
            // Channel 0: all 1s
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            // Channel 1: all 1s
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            // Channel 2: all 1s
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            // Channel 3: all 1s
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ];

        // Expected output: [1, 4, 1, 1]
        // Channel 0: 1*9 = 9
        // Channel 1: 2*9 = 18
        // Channel 2: 3*9 = 27
        // Channel 3: 4*9 = 36
        let expected = vec![9.0, 18.0, 27.0, 36.0];

        let output = execute_graph(&graph, &input).expect("Depthwise conv execution failed");
        verify_output(&output, &expected, 1e-4);
    }
}

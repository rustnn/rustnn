// SPDX-FileCopyrightText: 2026 Shubham Gupta <shubhamg13.work@gmail.com>
//
// SPDX-License-Identifier: Apache-2

//! CANN device test -- exercises the full rustnn CANN pipeline on a real
//! OHOS device with Kirin NPU.
//!
//! Build:
//!   cargo build --example cann_device_test --features cann-runtime \
//!     --target aarch64-unknown-linux-ohos --release
//!
//! Run on device:
//!   LD_LIBRARY_PATH=. ./cann_device_test

use std::collections::HashMap;

use rustnn::backend_selection::Backend;
use rustnn::mlcontext::{
    MLContext, MLContextOptions, MLOperand, MLOperandDescriptor, MLPowerPreference,
    MLTensorDescriptor,
};
use rustnn::mlgraphbuilder::MLGraphBuilder;
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::{
    MLArgMinMaxOptions, MLBatchNormalizationOptions, MLConvTranspose2dOptions, MLPool2dOptions,
};

fn main() {
    println!("=== CANN Device Test ===\n");

    // Shared context
    let options = MLContextOptions::new(MLPowerPreference::Default, true)
        .with_rustnn_backend_hint(Backend::Cann);
    let mut context = match MLContext::create(&options) {
        Ok(ctx) => {
            println!("Context created: {:?}", ctx.rustnn_backend());
            ctx
        }
        Err(err) => {
            eprintln!("Failed to create CANN context: {err:?}");
            std::process::exit(1);
        }
    };

    // ── Test 1: Add ─────────────────────────────────────────────────
    println!("\n--- Op: Add ---");

    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    let desc_2x2 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
    let a: MLOperand = builder.input("a", &desc_2x2).unwrap();
    let b: MLOperand = builder.input("b", &desc_2x2).unwrap();
    let sum: MLOperand = builder.add(a, b).unwrap();
    let mut graph = builder.build(&HashMap::from([("sum", sum)])).unwrap();

    let tdesc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2])
        .to_writable()
        .to_readable();
    let tensor_a = context.create_tensor(&tdesc).unwrap();
    let tensor_b = context.create_tensor(&tdesc).unwrap();
    let tensor_out = context.create_tensor(&tdesc).unwrap();

    context
        .write_tensor(&tensor_a, &vec![1.0f32, 2.0, 3.0, 4.0])
        .unwrap();
    context
        .write_tensor(&tensor_b, &vec![5.0f32, 6.0, 7.0, 8.0])
        .unwrap();
    context
        .dispatch(
            &mut graph,
            &HashMap::from([("a", &tensor_a), ("b", &tensor_b)]),
            &HashMap::from([("sum", &tensor_out)]),
        )
        .unwrap();

    let mut out_data = vec![0.0f32; 4];
    context.read_tensor(&tensor_out, &mut out_data).unwrap();
    println!("  Add([1,2,3,4], [5,6,7,8]) = {:?}", out_data);
    assert_eq!(out_data, [6.0f32, 8.0, 10.0, 12.0]);
    println!("  PASS");

    // ── Test 2: Relu ────────────────────────────────────────────────
    println!("\n--- Op: Relu ---");

    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    let desc_4 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4]);
    let input: MLOperand = builder.input("x", &desc_4).unwrap();
    let output: MLOperand = builder.relu(input).unwrap();
    let mut graph = builder.build(&HashMap::from([("r", output)])).unwrap();

    let tdesc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4])
        .to_writable()
        .to_readable();
    let tensor_input = context.create_tensor(&tdesc).unwrap();
    let tensor_output = context.create_tensor(&tdesc).unwrap();

    context
        .write_tensor(&tensor_input, &vec![-1.0f32, 2.0, -3.0, 4.0])
        .unwrap();
    context
        .dispatch(
            &mut graph,
            &HashMap::from([("x", &tensor_input)]),
            &HashMap::from([("r", &tensor_output)]),
        )
        .unwrap();

    let mut out_data = vec![0.0f32; 4];
    context.read_tensor(&tensor_output, &mut out_data).unwrap();
    println!("  Relu([-1, 2, -3, 4]) = {:?}", out_data);
    assert_eq!(out_data, [0.0f32, 2.0, 0.0, 4.0]);
    println!("  PASS");

    // ── Test 3: Identity ────────────────────────────────────────────
    println!("\n--- Op: Identity ---");

    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    let desc_4 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4]);
    let input: MLOperand = builder.input("x", &desc_4).unwrap();
    let output: MLOperand = builder.identity(input).unwrap();
    let mut graph = builder.build(&HashMap::from([("id", output)])).unwrap();

    let tdesc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4])
        .to_writable()
        .to_readable();
    let tensor_input = context.create_tensor(&tdesc).unwrap();
    let tensor_output = context.create_tensor(&tdesc).unwrap();

    context
        .write_tensor(&tensor_input, &vec![7.0f32, -3.0, 0.0, 5.0])
        .unwrap();
    context
        .dispatch(
            &mut graph,
            &HashMap::from([("x", &tensor_input)]),
            &HashMap::from([("id", &tensor_output)]),
        )
        .unwrap();

    let mut out_data = vec![0.0f32; 4];
    context.read_tensor(&tensor_output, &mut out_data).unwrap();
    println!("  Identity([7, -3, 0, 5]) = {:?}", out_data);
    assert_eq!(out_data, [7.0f32, -3.0, 0.0, 5.0]);
    println!("  PASS");

    // ── Test 4: Conv2d ──────────────────────────────────────────────
    println!("\n--- Op: Conv2d ---");

    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    let input_desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 3, 3]);
    let filter_desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 2, 2]);
    let input: MLOperand = builder.input("x", &input_desc).unwrap();
    let filter: MLOperand = builder
        .constant_from_vec(&filter_desc, vec![1.0f32, 0.0, 0.0, 1.0])
        .unwrap();
    let output: MLOperand = builder.conv2d(input, filter).unwrap();
    let mut graph = builder.build(&HashMap::from([("y", output)])).unwrap();

    let tdesc_input = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 3, 3])
        .to_writable()
        .to_readable();
    let tdesc_output =
        MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 2, 2]).to_readable();
    let tensor_input = context.create_tensor(&tdesc_input).unwrap();
    let tensor_output = context.create_tensor(&tdesc_output).unwrap();

    context
        .write_tensor(
            &tensor_input,
            &vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
    context
        .dispatch(
            &mut graph,
            &HashMap::from([("x", &tensor_input)]),
            &HashMap::from([("y", &tensor_output)]),
        )
        .unwrap();

    let mut out_data = vec![0.0f32; 4];
    context.read_tensor(&tensor_output, &mut out_data).unwrap();
    println!("  Conv2d(3x3, 2x2 identity) = {:?}", out_data);
    assert_eq!(out_data, [6.0f32, 8.0, 12.0, 14.0]);
    println!("  PASS");

    // ── Test 5: MaxPool2d ────────────────────────────────────────────
    println!("\n--- Op: MaxPool2d ---");

    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    let input_desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 4, 4]);
    let input: MLOperand = builder.input("x", &input_desc).unwrap();
    let pool_options = MLPool2dOptions {
        window_dimensions: Some(vec![2, 2]),
        strides: vec![2, 2],
        ..MLPool2dOptions::default()
    };
    let output: MLOperand = builder
        .max_pool2d_with_options(input, pool_options)
        .unwrap();
    let mut graph = builder.build(&HashMap::from([("y", output)])).unwrap();

    let tdesc_input = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 4, 4])
        .to_writable()
        .to_readable();
    let tdesc_output =
        MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 2, 2]).to_readable();
    let tensor_input = context.create_tensor(&tdesc_input).unwrap();
    let tensor_output = context.create_tensor(&tdesc_output).unwrap();

    let input_data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    context.write_tensor(&tensor_input, &input_data).unwrap();
    context
        .dispatch(
            &mut graph,
            &HashMap::from([("x", &tensor_input)]),
            &HashMap::from([("y", &tensor_output)]),
        )
        .unwrap();

    let mut out_data = vec![0.0f32; 4];
    context.read_tensor(&tensor_output, &mut out_data).unwrap();
    println!("  MaxPool2d(4x4, 2x2, stride 2) = {:?}", out_data);
    assert_eq!(out_data, [6.0f32, 8.0, 14.0, 16.0]);
    println!("  PASS");

    // ── Test 6: ConvTranspose2d ──────────────────────────────────────
    println!("\n--- Op: ConvTranspose2d ---");

    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    let input_desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 2, 2]);
    let filter_desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 2, 2]);
    let input: MLOperand = builder.input("x", &input_desc).unwrap();
    let filter: MLOperand = builder
        .constant_from_vec(&filter_desc, vec![1.0f32, 0.0, 0.0, 1.0])
        .unwrap();
    let transpose_options = MLConvTranspose2dOptions {
        strides: vec![2, 2],
        ..MLConvTranspose2dOptions::default()
    };
    let output: MLOperand = builder
        .conv_transpose2d_with_options(input, filter, transpose_options)
        .unwrap();
    let mut graph = builder.build(&HashMap::from([("y", output)])).unwrap();

    let tdesc_input = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 2, 2])
        .to_writable()
        .to_readable();
    let tdesc_output =
        MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 4, 4]).to_readable();
    let tensor_input = context.create_tensor(&tdesc_input).unwrap();
    let tensor_output = context.create_tensor(&tdesc_output).unwrap();

    context
        .write_tensor(&tensor_input, &vec![1.0f32, 2.0, 3.0, 4.0])
        .unwrap();
    context
        .dispatch(
            &mut graph,
            &HashMap::from([("x", &tensor_input)]),
            &HashMap::from([("y", &tensor_output)]),
        )
        .unwrap();

    let mut out_data = vec![0.0f32; 16];
    context.read_tensor(&tensor_output, &mut out_data).unwrap();
    println!(
        "  ConvTranspose2d(2x2, stride 2, identity filter) = {:?}",
        out_data
    );
    assert_eq!(
        out_data,
        [
            1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0,
        ]
    );
    println!("  PASS");

    // ── Test 7: MatMul ──────────────────────────────────────────────
    println!("\n--- Op: MatMul ---");

    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    let desc_2x2 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
    let a: MLOperand = builder.input("a", &desc_2x2).unwrap();
    let b: MLOperand = builder
        .constant_from_vec(&desc_2x2, vec![1.0f32, 0.0, 0.0, 1.0])
        .unwrap();
    let output: MLOperand = builder.matmul(a, b).unwrap();
    let mut graph = builder.build(&HashMap::from([("y", output)])).unwrap();

    let tdesc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2])
        .to_writable()
        .to_readable();
    let tensor_input = context.create_tensor(&tdesc).unwrap();
    let tensor_output = context.create_tensor(&tdesc).unwrap();

    context
        .write_tensor(&tensor_input, &vec![1.0f32, 2.0, 3.0, 4.0])
        .unwrap();
    context
        .dispatch(
            &mut graph,
            &HashMap::from([("a", &tensor_input)]),
            &HashMap::from([("y", &tensor_output)]),
        )
        .unwrap();

    let mut out_data = vec![0.0f32; 4];
    context.read_tensor(&tensor_output, &mut out_data).unwrap();
    println!("  MatMul([1,2;3,4] * I) = {:?}", out_data);
    assert_eq!(out_data, [1.0f32, 2.0, 3.0, 4.0]);
    println!("  PASS");

    // ── Test 8: Cos ─────────────────────────────────────────────────
    println!("\n--- Op: Cos ---");

    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    let desc_2 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2]);
    let input: MLOperand = builder.input("x", &desc_2).unwrap();
    let output: MLOperand = builder.cos(input).unwrap();
    let mut graph = builder.build(&HashMap::from([("y", output)])).unwrap();

    let tdesc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2])
        .to_writable()
        .to_readable();
    let tensor_input = context.create_tensor(&tdesc).unwrap();
    let tensor_output = context.create_tensor(&tdesc).unwrap();

    context
        .write_tensor(&tensor_input, &vec![0.0f32, std::f32::consts::PI])
        .unwrap();
    context
        .dispatch(
            &mut graph,
            &HashMap::from([("x", &tensor_input)]),
            &HashMap::from([("y", &tensor_output)]),
        )
        .unwrap();

    let mut out_data = vec![0.0f32; 2];
    context.read_tensor(&tensor_output, &mut out_data).unwrap();
    println!("  Cos([0, pi]) = {:?}", out_data);
    let cos0 = out_data[0];
    assert!(
        (cos0 - 1.0).abs() < 1e-5,
        "cos(0) should be 1.0, got {cos0}"
    );
    let cos_pi = out_data[1];
    assert!(
        (cos_pi + 1.0).abs() < 1e-5,
        "cos(pi) should be -1.0, got {cos_pi}"
    );
    println!("  PASS");

    // ── Test 9: ArgMax ──────────────────────────────────────────────

    println!("\n--- Op: ArgMax ---");

    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    let desc_4 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4]);
    let input: MLOperand = builder.input("x", &desc_4).unwrap();
    let argmax_options = MLArgMinMaxOptions {
        keep_dimensions: true,
        output_data_type: MLOperandDataType::Int64,
        ..MLArgMinMaxOptions::default()
    };
    let output: MLOperand = builder
        .arg_max_with_options(input, 0, argmax_options)
        .unwrap();
    let mut graph = builder.build(&HashMap::from([("y", output)])).unwrap();

    let tdesc_input = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4])
        .to_writable()
        .to_readable();
    let tdesc_output = MLTensorDescriptor::new(MLOperandDataType::Int64, vec![1]).to_readable();
    let tensor_input = context.create_tensor(&tdesc_input).unwrap();
    let tensor_output = context.create_tensor(&tdesc_output).unwrap();

    context
        .write_tensor(&tensor_input, &vec![3.0f32, 1.0, 4.0, 2.0])
        .unwrap();
    context
        .dispatch(
            &mut graph,
            &HashMap::from([("x", &tensor_input)]),
            &HashMap::from([("y", &tensor_output)]),
        )
        .unwrap();

    let mut out_data = vec![0i64; 1];
    context.read_tensor(&tensor_output, &mut out_data).unwrap();
    println!("  ArgMax([3, 1, 4, 2]) = {:?}", out_data);
    assert_eq!(out_data[0], 2);
    println!("  PASS");

    // ── Test 10: BatchNormalization ──────────────────────────────────
    println!("\n--- Op: BatchNormalization ---");

    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    let input_desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 2, 2]);
    let channel_desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1]);
    let input: MLOperand = builder.input("x", &input_desc).unwrap();
    let mean: MLOperand = builder
        .constant_from_vec(&channel_desc, vec![0.0f32])
        .unwrap();
    let variance: MLOperand = builder
        .constant_from_vec(&channel_desc, vec![1.0f32])
        .unwrap();

    let bn_options = MLBatchNormalizationOptions::default();
    let output: MLOperand = builder
        .batch_normalization_with_options(input, mean, variance, bn_options)
        .unwrap();
    let mut graph = builder.build(&HashMap::from([("y", output)])).unwrap();

    let tdesc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 2, 2])
        .to_writable()
        .to_readable();
    let tensor_input = context.create_tensor(&tdesc).unwrap();
    let tensor_output = context.create_tensor(&tdesc).unwrap();

    context
        .write_tensor(&tensor_input, &vec![1.0f32, 2.0, 3.0, 4.0])
        .unwrap();
    context
        .dispatch(
            &mut graph,
            &HashMap::from([("x", &tensor_input)]),
            &HashMap::from([("y", &tensor_output)]),
        )
        .unwrap();

    let mut out_data = vec![0.0f32; 4];
    context.read_tensor(&tensor_output, &mut out_data).unwrap();
    println!("  BatchNormalization(mean=0, var=1) = {:?}", out_data);
    for (i, val) in out_data.iter().enumerate() {
        let expected = (i + 1) as f32;
        assert!(
            (val - expected).abs() < 1e-4,
            "BatchNormalization[{i}] expected {expected}, got {val}"
        );
    }
    println!("  PASS");

    println!("\n=== All tests passed ===");
}

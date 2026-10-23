#![cfg(feature = "trtx-runtime")]

use rustnn::graph::{ConstantReference, FileRef};
use rustnn::mlcontext::{
    Backend, MLContext, MLContextOptions, MLNamedOperands, MLNamedTensors, MLOperandDescriptor,
    MLPowerPreference, MLTensorDescriptor, RustNNOptions,
};
use rustnn::mlgraphbuilder::MLGraphBuilder;
use rustnn::operator_enums::MLOperandDataType;

fn context() -> MLContext<'static> {
    context_with_cuda_graphs(false)
}

fn context_with_cuda_graphs(cuda_graphs: bool) -> MLContext<'static> {
    let mut tuning = RustNNOptions::default();
    tuning.trtx.cuda_graphs = cuda_graphs;
    MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, true)
            .with_rustnn_backend_hint(Backend::Trtx)
            .with_rustnn_options(tuning),
    )
    .unwrap()
}

#[test]
fn refit_gpu_and_inline_weights_across_cached_builds() {
    let mut context = context();
    let input_desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, 2]);
    let weight_desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
    let mut io_desc = MLTensorDescriptor::from_operand_descriptor(&input_desc);
    io_desc.set_readable(true);
    io_desc.set_writable(true);
    let input = context.create_tensor(&io_desc).unwrap();
    let output = context.create_tensor(&io_desc).unwrap();
    let independent = context.create_tensor(&io_desc).unwrap();
    context.write_tensor(&input, &[1.0f32, 2.0]).unwrap();
    let mut tensor_desc = MLTensorDescriptor::from_operand_descriptor(&weight_desc);
    tensor_desc.set_readable(true);
    tensor_desc.set_writable(true);
    let weights = context.create_tensor(&tensor_desc).unwrap();
    for factor in [1.0f32, 2.0] {
        let values = [factor, 2.0 * factor, 3.0 * factor, 4.0 * factor];
        context.write_tensor(&weights, &values).unwrap();
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let x = builder.input("x", &input_desc).unwrap();
        let w = builder.constant_from_tensor(weights.clone()).unwrap();
        let bias = builder
            .constant_from_slice(&input_desc, &[5.0f32, 6.0])
            .unwrap();
        let scale = builder
            .constant_from_slice(&input_desc, &[2.0f32, 3.0])
            .unwrap();
        let product = builder.matmul(x, w).unwrap();
        let sum = builder.add(product, bias).unwrap();
        let other = builder.mul(x, scale).unwrap();
        let mut graph = builder
            .build(&MLNamedOperands::from([("sum", sum), ("other", other)]))
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &MLNamedTensors::from([("x", &input)]),
                &MLNamedTensors::from([("sum", &output), ("other", &independent)]),
            )
            .unwrap();
        let mut actual = [0.0f32; 2];
        context.read_tensor(&output, &mut actual).unwrap();
        assert_eq!(actual, [7.0 * factor + 5.0, 10.0 * factor + 6.0]);
        context.read_tensor(&independent, &mut actual).unwrap();
        assert_eq!(actual, [2.0, 6.0]);
        let mut source = [0.0f32; 4];
        context.read_tensor(&weights, &mut source).unwrap();
        assert_eq!(source, values, "refitting must retain caller-owned tensors");
    }
}

#[test]
fn refit_file_weights() {
    let mut context = context();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("weights.bin");
    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    std::fs::write(&path, bytemuck::cast_slice(&values)).unwrap();
    let ast = webnn_graph::parser::parse_wg_text(
        r#"
        webnn_graph "file_refit" v1 {
            inputs { x: f32[1, 2]; }
            consts { w: f32[2, 2] @weights("w"); b: f32[1, 2] @weights("b"); }
            nodes {
            product = matmul(x, w);
            sum = add(product, b);
            }
            outputs { sum; }
        }
    "#,
    )
    .unwrap();
    let mut info = rustnn::webnn_json::from_graph_json(&ast).unwrap();
    for (id, reference) in &mut info.constant_operand_ids_to_handles {
        let name = info.operands[*id as usize].name.as_deref().unwrap();
        let (offset, len) = if name == "w" { (0, 16) } else { (16, 8) };
        *reference = ConstantReference::FileRef(FileRef {
            path: path.clone(),
            offset,
            len,
            label: None,
        });
    }
    let mut graph = context.rustnn_build_graph(info).unwrap();
    let mut desc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 2]);
    desc.set_readable(true);
    desc.set_writable(true);
    let input = context.create_tensor(&desc).unwrap();
    let output = context.create_tensor(&desc).unwrap();
    context.write_tensor(&input, &[1.0f32, 2.0]).unwrap();
    context
        .dispatch(
            &mut graph,
            &MLNamedTensors::from([("x", &input)]),
            &MLNamedTensors::from([("sum", &output)]),
        )
        .unwrap();
    let mut actual = [0.0f32; 2];
    context.read_tensor(&output, &mut actual).unwrap();
    assert_eq!(actual, [12.0, 16.0]);
}

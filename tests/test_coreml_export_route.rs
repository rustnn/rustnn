//! The `RustExport` compile route: a bundle written by `mlmodelc-export` loads through the
//! CoreML backend and predicts the same values as Apple's compiler.
#![cfg(all(
    target_vendor = "apple",
    feature = "coreml-runtime",
    feature = "coreml-export"
))]

use std::path::Path;

use rustnn::executors::coreml::CoremlLoadRoute;
use rustnn::mlcontext::{
    Backend, CoremlCompileRoute, LoadDiagnostics, MLContext, MLContextOptions, MLGraphBuilder,
    MLNamedOperands, MLNamedTensors, MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor,
    RustNNOptions,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::{ConverterRegistry, GraphInfo};

const INPUT: [f32; 4] = [0.5, -1.25, 3.0, -0.0625];
const BIAS: [f32; 4] = [1.0, 0.5, -4.0, 0.0625];

fn context(route: CoremlCompileRoute) -> MLContext<'static> {
    let mut options = RustNNOptions::default();
    options.coreml.compile_route = route;
    MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml)
            .with_rustnn_options(options),
    )
    .unwrap()
}

/// `relu(input + bias)`: one input, one weight-blob constant, one activation.
fn save_graph(context: &mut MLContext<'_>, path: &Path) -> GraphInfo {
    let mut builder = MLGraphBuilder::new(context).unwrap();
    let descriptor = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4]);
    let input = builder.input("input", &descriptor).unwrap();
    let bias = builder.constant_from_slice(&descriptor, &BIAS).unwrap();
    let sum = builder.add(input, bias).unwrap();
    let output = builder.relu(sum).unwrap();
    builder
        .rustnn_save_webnn(&MLNamedOperands::from([("output", output)]), path)
        .unwrap();
    rustnn::load_graph_from_path(path).unwrap()
}

fn run(context: &mut MLContext<'_>, graph_info: GraphInfo) -> (CoremlLoadRoute, [f32; 4]) {
    let mut graph = context.rustnn_build_graph(graph_info).unwrap();
    let Some(LoadDiagnostics::Coreml(load)) = graph.rustnn_load_diagnostics() else {
        panic!("CoreML graph did not report CoreML load diagnostics");
    };
    let descriptor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4])
        .to_writable()
        .to_readable();
    let input = context.create_tensor(&descriptor).unwrap();
    let output = context.create_tensor(&descriptor).unwrap();
    context.write_tensor(&input, &INPUT).unwrap();
    context
        .dispatch(
            &mut graph,
            &MLNamedTensors::from([("input", &input)]),
            &MLNamedTensors::from([("output", &output)]),
        )
        .unwrap();
    let mut actual = [0f32; 4];
    context.read_tensor(&output, &mut actual).unwrap();
    (load.route, actual)
}

#[test]
fn rust_export_route_writes_a_bundle_and_matches_the_native_route() {
    let source = tempfile::tempdir().unwrap();
    let path = source.path().join("relu_bias.webnn");

    let mut native = context(CoremlCompileRoute::Native);
    let graph_info = save_graph(&mut native, &path);

    // The same conversion, written as a bundle without Apple's compiler.
    let converted = ConverterRegistry::with_defaults()
        .convert("coreml", &graph_info)
        .unwrap();
    let bundle = source.path().join("relu_bias.mlmodelc");
    converted.write_coreml_bundle(&bundle).unwrap();
    for file in [
        "model.mil",
        "coremldata.bin",
        "metadata.json",
        "analytics/coremldata.bin",
    ] {
        assert!(bundle.join(file).is_file(), "missing {file}");
    }
    assert_eq!(
        std::fs::read(bundle.join("weights/weights.bin")).ok(),
        converted.weights_data,
        "the bundle must carry the converter's weight blob byte for byte"
    );

    let (native_route, expected) = run(&mut native, graph_info.clone());
    assert_ne!(native_route, CoremlLoadRoute::RustExport);
    let reference: Vec<f32> = INPUT
        .iter()
        .zip(BIAS)
        .map(|(x, b)| (x + b).max(0.0))
        .collect();
    assert_eq!(expected.to_vec(), reference);

    let mut exported = context(CoremlCompileRoute::RustExport);
    let (route, actual) = run(&mut exported, graph_info);
    assert_eq!(route, CoremlLoadRoute::RustExport);
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-6 * e.abs().max(1.0),
            "output[{index}]: RustExport {a} vs Native {e}"
        );
    }
}

#[test]
fn auto_route_keeps_the_native_compiler_where_it_exists() {
    // Every Apple target except watchOS has `MLModel.compileModel`; `Auto` must not change
    // the default behavior there even with the exporter compiled in.
    let source = tempfile::tempdir().unwrap();
    let mut context = context(CoremlCompileRoute::Auto);
    let graph_info = save_graph(&mut context, &source.path().join("auto.webnn"));
    let (route, _) = run(&mut context, graph_info);
    if cfg!(target_os = "watchos") {
        assert_eq!(route, CoremlLoadRoute::RustExport);
    } else {
        assert_ne!(route, CoremlLoadRoute::RustExport);
    }
}

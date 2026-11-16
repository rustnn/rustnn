//! `ConvertedGraph::write_coreml_bundle` writes a complete `.mlmodelc` on any host.
//!
//! No Core ML runtime is involved: this checks the bundle layout and the weight blob that
//! the pure-Rust exporter emits from the CoreML converter's output.
#![cfg(feature = "coreml-export")]

use rustnn::{ConverterRegistry, GraphError};

fn convert_sample() -> rustnn::ConvertedGraph {
    let graph = rustnn::load_graph_from_path("examples/sample_graph.webnn").unwrap();
    ConverterRegistry::with_defaults()
        .convert("coreml", &graph)
        .unwrap()
}

#[test]
fn bundle_has_the_four_mlmodelc_files_and_the_weight_blob() {
    let converted = convert_sample();
    let directory = tempfile::tempdir().unwrap();
    let bundle = directory.path().join("sample.mlmodelc");
    let stats = converted.write_coreml_bundle(&bundle).unwrap();

    for file in [
        "model.mil",
        "coremldata.bin",
        "metadata.json",
        "analytics/coremldata.bin",
    ] {
        assert!(bundle.join(file).is_file(), "missing {file}");
    }
    let mil = std::fs::read_to_string(bundle.join("model.mil")).unwrap();
    assert!(mil.contains("program("), "model.mil is not MIL source text");
    assert_eq!(stats.input_bytes, converted.data.len());
    assert_eq!(stats.output_bytes, mil.len());
    assert!(stats.operation_count > 0);

    match &converted.weights_data {
        Some(weights) => {
            let written = std::fs::read(bundle.join("weights/weights.bin")).unwrap();
            assert_eq!(
                &written, weights,
                "weights.bin must carry the converter's blob"
            );
        }
        None => assert!(!bundle.join("weights").exists()),
    }
}

#[test]
fn rewriting_a_bundle_replaces_its_files() {
    let converted = convert_sample();
    let directory = tempfile::tempdir().unwrap();
    let bundle = directory.path().join("twice.mlmodelc");
    let first = converted.write_coreml_bundle(&bundle).unwrap();
    let second = converted.write_coreml_bundle(&bundle).unwrap();
    assert_eq!(first, second);
    assert!(bundle.join("model.mil").is_file());
}

#[test]
fn non_coreml_conversions_are_rejected() {
    let other = rustnn::ConvertedGraph {
        format: "onnx",
        content_type: "application/octet-stream",
        data: vec![0],
        weights_data: None,
    };
    let directory = tempfile::tempdir().unwrap();
    let error = other
        .write_coreml_bundle(directory.path().join("never.mlmodelc"))
        .unwrap_err();
    assert!(matches!(
        error,
        GraphError::ConversionFailed { ref format, .. } if format == "onnx"
    ));
    assert!(!directory.path().join("never.mlmodelc").exists());
}

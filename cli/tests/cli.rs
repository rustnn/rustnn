use std::path::PathBuf;
use std::process::{Command, Output};

fn sample() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../examples/sample_graph.webnn")
}

fn cli(args: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rustnn"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn inspects_and_exports_without_a_runtime() {
    let directory = tempfile::tempdir().unwrap();
    let dot = directory.path().join("graph.dot");
    let output = cli(&[
        sample().as_os_str(),
        "--export-dot".as_ref(),
        dot.as_os_str(),
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(std::fs::read_to_string(dot).unwrap().contains("digraph"));
    assert!(output.stdout.is_empty());
}

#[test]
fn discovery_matches_the_library_registry_and_help() {
    let output = cli(&["--list".as_ref()]);
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    let formats = rustnn::ConverterRegistry::with_defaults().available_formats();
    assert!(text.contains(&format!("Converters: {}", formats.join(", "))));
    let help = cli(&["--help".as_ref()]);
    let help = String::from_utf8(help.stdout).unwrap();
    for backend in rustnn::mlcontext::Backend::compiled() {
        assert!(text.contains(backend.name()));
        assert!(help.contains(backend.name()));
    }
    for format in formats {
        assert!(help.contains(format));
    }
}

#[test]
fn rejects_unregistered_formats() {
    let output = cli(&[
        sample().as_os_str(),
        "--convert".as_ref(),
        "missing".as_ref(),
    ]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid value"));
}

#[cfg(feature = "onnx-converter")]
#[test]
fn binary_stdout_contains_only_the_model_even_with_logging() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("relu.webnn");
    std::fs::write(
        &path,
        r#"webnn_graph "relu" v1 {
      inputs { x: f32[2]; }
      nodes { y = relu(x); }
      outputs { y; }
    }"#,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_rustnn"))
        .arg(&path)
        .args(["--convert", "onnx"])
        .env("RUST_LOG", "rustnn=info")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let graph = rustnn::load_graph_from_path(&path).unwrap();
    let expected = rustnn::ConverterRegistry::with_defaults()
        .convert("onnx", &graph)
        .unwrap();
    assert_eq!(output.stdout, expected.data);
}

#[cfg(feature = "coreml-converter")]
#[test]
fn coreml_export_writes_weights_at_the_referenced_path() {
    let directory = tempfile::tempdir().unwrap();
    let graph = directory.path().join("graph.webnn");
    std::fs::write(
        &graph,
        r#"webnn_graph "blob" v1 {
      inputs { x: f32[2]; }
      consts { c: f32[2] @scalar(1.0); }
      nodes { y = add(x, c); }
      outputs { y; }
    }"#,
    )
    .unwrap();
    let model = directory.path().join("model.mlmodel");
    let output = cli(&[
        graph.as_os_str(),
        "--convert".as_ref(),
        "coreml".as_ref(),
        "--convert-output".as_ref(),
        model.as_os_str(),
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected = rustnn::ConverterRegistry::with_defaults()
        .convert("coreml", &rustnn::load_graph_from_path(&graph).unwrap())
        .unwrap();
    let model = std::fs::read(model).unwrap();
    let reference = b"@model_path/weights/weights.bin";
    assert!(
        model
            .windows(reference.len())
            .any(|window| window == reference)
    );
    assert_eq!(
        std::fs::read(directory.path().join("weights/weights.bin")).unwrap(),
        expected.weights_data.unwrap()
    );
}

#[cfg(feature = "burn-cpu-runtime")]
#[test]
fn executes_through_context_without_requesting_conversion() {
    let output = cli(&[sample().as_os_str(), "--run".as_ref(), "burn-cpu".as_ref()]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stderr).unwrap();
    assert!(text.contains("Runtime: burn-cpu"));
    assert!(text.contains("data=[1, 1, 1, 1]"), "{text}");
    assert!(output.stdout.is_empty());
}

use std::fs;
use std::path::Path;

fn collect_protos(dir: &str) -> Vec<String> {
    let mut files = Vec::new();
    recurse(dir.as_ref(), &mut files);
    files
}

fn recurse(dir: &Path, files: &mut Vec<String>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                recurse(&path, files);
            } else if path.extension().and_then(|e| e.to_str()) == Some("proto") {
                files.push(path.to_string_lossy().to_string());
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = prost_build::Config::new();
    config.bytes(&["."]);

    let coreml_dir = "protos/coreml";
    let onnx_dir = "protos/onnx";

    let mut coreml_files = collect_protos(coreml_dir);
    coreml_files.sort();
    let onnx_files = vec![format!("{}/onnx-ml.proto", onnx_dir)];

    config.compile_protos(&coreml_files, &[coreml_dir])?;
    config.compile_protos(&onnx_files, &[onnx_dir])?;

    println!("cargo:rerun-if-changed=protos");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=OUT_DIR");
    Ok(())
}

#[cfg(feature = "litert-runtime")]
use std::env;
use std::fs;
use std::path::Path;
#[cfg(feature = "litert-runtime")]
use std::process::Command;

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

#[cfg(feature = "litert-runtime")]
fn run_flatc(schema: &str, out_dir: &str) -> Result<(), String> {
    let status = Command::new("flatc")
        .args(["--rust", "-o", out_dir, schema])
        .status()
        .map_err(|e| format!("failed to run flatc: {e}"))?;
    if !status.success() {
        return Err("flatc exited with error".into());
    }

    // Keep warnings in generated bindings from obscuring warnings in handwritten code.
    // flatc does not currently emit Rust 2024-compatible unsafe blocks, among other
    // lints, so scope the allowance to the generated schema rather than the crate.
    let generated_path = Path::new(out_dir).join("schema_generated.rs");
    let generated = fs::read_to_string(&generated_path)
        .map_err(|e| format!("failed to read generated FlatBuffers schema: {e}"))?;
    let wrapped = format!(
        "#[allow(warnings)]\nmod generated_flatc_schema {{\n{generated}\n}}\n\
         pub use generated_flatc_schema::*;\n"
    );
    fs::write(generated_path, wrapped)
        .map_err(|e| format!("failed to wrap generated FlatBuffers schema: {e}"))?;

    Ok(())
}

#[cfg(feature = "litert-runtime")]
fn build_tflite_schema() {
    let out_dir = env::var("OUT_DIR").unwrap();
    run_flatc("protos/tflite/schema.fbs", &out_dir)
        .expect("Running flatc failed. Make sure flatc is installed and in PATH");
    println!("cargo:rerun-if-changed=protos/tflite/schema.fbs");
}

fn build_coreml_protos() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = prost_build::Config::new();
    config.bytes(["."]); // Fix clippy::needless_borrows_for_generic_args

    let coreml_dir = "protos/coreml";
    let mut coreml_files = collect_protos(coreml_dir);
    coreml_files.sort();
    config.compile_protos(&coreml_files, &[coreml_dir])?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Only compile CoreML protos - ONNX protos come from webnn-onnx-utils
    build_coreml_protos()?;

    // Build TFLite flatbuffer schema - Only required for the litert-runtime.
    #[cfg(feature = "litert-runtime")]
    build_tflite_schema();

    println!("cargo:rerun-if-changed=protos");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=OUT_DIR");
    Ok(())
}

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

/// Bundle WPT conformance JSONs as a single JSON array for wasm tests.
fn write_wpt_conformance_gz(out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let conformance_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("wpt_data")
        .join("conformance");
    let mut entries: Vec<(String, String)> = Vec::new();
    if conformance_dir.is_dir() {
        for entry in fs::read_dir(&conformance_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json") {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown.json")
                    .to_string();
                let mut s = String::new();
                fs::File::open(&path)?.read_to_string(&mut s)?;
                entries.push((name, s));
            }
        }
        println!("cargo:rerun-if-changed={}", conformance_dir.display());
    }
    let json = serde_json::to_string(&entries)?;
    let out_path = out_dir.join("wpt_conformance.json");
    let mut file = fs::File::create(&out_path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

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
    config.bytes(["."]); // Fix clippy::needless_borrows_for_generic_args

    let coreml_dir = "protos/coreml";

    let mut coreml_files = collect_protos(coreml_dir);
    coreml_files.sort();

    // Only compile CoreML protos - ONNX protos come from webnn-onnx-utils
    config.compile_protos(&coreml_files, &[coreml_dir])?;

    println!("cargo:rerun-if-changed=protos");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=OUT_DIR");

    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    write_wpt_conformance_gz(&out_dir)?;

    Ok(())
}

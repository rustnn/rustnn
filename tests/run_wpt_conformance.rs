//! WPT WebNN conformance tests (ONNX and/or TensorRT backend).
//!
//! ONNX: cargo test --test run_wpt_conformance --features onnx-runtime [-- run_wpt_conformance_tests]
//! TensorRT: cargo test --test run_wpt_conformance --features trtx-runtime-mock [-- run_wpt_conformance_tests_trtx]
//! Wasm (browser): cargo test --test run_wpt_conformance --target wasm32-unknown-unknown [-- run_wpt_conformance_webnn]
//!   (Test data is embedded as zip in build.rs; wasm cannot read files.)
//!
//! Add -- --nocapture to see which tests are found and run.
//! ONNX requires native library >= 1.23 on PATH; wrong version is skipped with a message.

#![cfg(any(
    feature = "onnx-runtime",
    feature = "trtx-runtime-mock",
    feature = "trtx-runtime",
))]
// A lot of test code is not used for WASM, we just ignore that instead of annotating every non-web
// function
#[cfg(target_arch = "wasm32")]
#[allow(dead_code, unused_imports)]
mod wpt_conformance;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[test]
#[cfg(feature = "onnx-runtime")]
fn run_wpt_conformance_tests() {
    let result = std::panic::catch_unwind(|| wpt_conformance::run_all());
    match result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => panic!("WPT conformance tests failed: {}", e),
        Err(panic_payload) => {
            let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            if msg.contains("ONNX Runtime")
                && (msg.contains("not compatible") || msg.contains("Failed to load"))
            {
                println!(
                    "[SKIP] WPT conformance: ONNX Runtime load/version issue. {}",
                    msg.lines().next().unwrap_or(&msg)
                );
                return;
            }
            panic!("WPT conformance test panicked: {}", msg);
        }
    }
}

/// Embedded WPT conformance JSONs (gzip) for wasm tests; wasm32 cannot read files from disk.
/// Decompressed in wasm via the browser DecompressionStream API.
#[cfg(all(test, target_arch = "wasm32"))]
pub mod wpt_embedded {
    /// Gzip-compressed JSON array of {"name": "add.json", "content": "..."}, produced in build.rs.
    pub static WPT_CONFORMANCE_JSON: &'static str =
        include_str!(concat!(env!("OUT_DIR"), "/wpt_conformance.json"));
}

#[test]
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
fn run_wpt_conformance_tests_trtx() {
    let result = std::panic::catch_unwind(|| wpt_conformance::run_all_trtx());
    match result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => panic!("WPT conformance tests (TRTX) failed: {}", e),
        Err(panic_payload) => {
            let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            panic!("WPT conformance test (TRTX) panicked: {}", msg);
        }
    }
}

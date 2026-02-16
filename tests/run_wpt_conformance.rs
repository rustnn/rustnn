//! WPT WebNN conformance tests (ONNX backend).
//!
//! Run with: cargo test --test run_wpt_conformance --features onnx-runtime
//!
//! To see which tests are found and run (and any skip message), add: -- --nocapture
//!
//! Requires ONNX Runtime native library >= 1.23 (e.g. from onnxruntime.dll on PATH).
//! If the wrong version is found (e.g. 1.17), the test is skipped with a message.

#![cfg(feature = "onnx-runtime")]

mod wpt_conformance;

#[test]
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

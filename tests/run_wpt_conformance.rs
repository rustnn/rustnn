//! WPT WebNN conformance tests (ONNX, TensorRT, and/or Burn backends).
//!
//! ONNX:
//!   cargo test --test run_wpt_conformance --features onnx-runtime -- run_wpt_conformance_tests --nocapture
//!
//! TensorRT:
//!   cargo test --test run_wpt_conformance --features trtx-runtime-mock -- run_wpt_conformance_tests_trtx --nocapture
//!
//! Burn CPU:
//!   cargo test --test run_wpt_conformance --features burn-runtime-cpu -- run_wpt_conformance_tests_burn_cpu --nocapture
//!
//! Burn WebGPU:
//!   cargo test --test run_wpt_conformance --features burn-runtime-webgpu -- run_wpt_conformance_tests_burn_webgpu --nocapture
//!
//! Makefile: `make burn-wpt-cpu`, `make burn-wpt-webgpu`
//!
//! Unsupported Burn ops and non-float32 cases are reported as [SKIP], not failures.

#![cfg(any(
    feature = "onnx-runtime",
    feature = "trtx-runtime-mock",
    feature = "trtx-runtime",
    feature = "burn-runtime-cpu",
    feature = "burn-runtime-webgpu"
))]
mod wpt_conformance;

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

#[test]
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
fn run_wpt_conformance_tests_trtx() {
    let result = std::panic::catch_unwind(|| wpt_conformance::run_all_trtx());
    match result {
        Ok(Ok(())) => {}
        // failure expected for now. Tracked with snapshots (failure on regressions)
        Ok(Err(e)) => println!("WPT conformance tests (TRTX) failed: {}", e),
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

#[test]
#[cfg(feature = "burn-runtime-cpu")]
fn run_wpt_conformance_tests_burn_cpu() {
    let result = std::panic::catch_unwind(|| wpt_conformance::run_all_burn_cpu());
    match result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => panic!("WPT conformance tests (Burn CPU) failed: {}", e),
        Err(panic_payload) => {
            let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            panic!("WPT conformance test (Burn CPU) panicked: {}", msg);
        }
    }
}

#[test]
#[cfg(feature = "burn-runtime-webgpu")]
fn run_wpt_conformance_tests_burn_webgpu() {
    let result = std::panic::catch_unwind(|| wpt_conformance::run_all_burn_webgpu());
    match result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => panic!("WPT conformance tests (Burn WebGPU) failed: {}", e),
        Err(panic_payload) => {
            let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            panic!("WPT conformance test (Burn WebGPU) panicked: {}", msg);
        }
    }
}

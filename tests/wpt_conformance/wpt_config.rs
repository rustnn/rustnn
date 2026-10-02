//! WPT harness compile-time options.

/// Reuse one [`MLContext`](rustnn::mlcontext::MLContext) per [`WptBackend`](super::wpt_backend::WptBackend) per test thread.
///
/// Default is `false`: benchmarks showed no ONNX CPU speedup, `OrtContext` retains tensors across
/// trials when reused, and backends are not validated for concurrent use across libtest threads.
/// When `true`, trials on the same thread share a context via `wpt_context_pool` (thread-local).
pub const REUSE_ML_CONTEXT: bool = true;

/// Opt in to upstream budgets instead of the historical compatibility floors.
pub fn strict_wpt_tolerance() -> bool {
    std::env::var("WPT_STRICT_TOLERANCE").is_ok_and(|value| value == "1")
}

/// Requested CoreML policy, shared by execution and reporting (not placement).
pub fn coreml_requested_device() -> &'static str {
    parse_coreml_device(std::env::var("WPT_COREML_DEVICE").ok().as_deref())
}

fn parse_coreml_device(value: Option<&str>) -> &'static str {
    match value {
        None | Some("cpu") => "cpu",
        Some("gpu") => "gpu",
        Some("npu") => "npu",
        Some(other) => panic!("WPT_COREML_DEVICE must be cpu, gpu, or npu; got {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn coreml_policy_defaults_to_cpu_and_preserves_valid_requests() {
        assert_eq!(super::parse_coreml_device(None), "cpu");
        for value in ["cpu", "gpu", "npu"] {
            assert_eq!(super::parse_coreml_device(Some(value)), value);
        }
    }

    #[test]
    #[should_panic(expected = "WPT_COREML_DEVICE must be cpu, gpu, or npu")]
    fn invalid_coreml_policy_is_not_reported_as_a_valid_request() {
        super::parse_coreml_device(Some("all"));
    }
}

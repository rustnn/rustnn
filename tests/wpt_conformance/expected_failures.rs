//! Backend-specific WPT failures that are allowed without skipping execution.

use std::collections::HashSet;
use std::sync::LazyLock;

static COREML_EXPECTED_FAILURES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    include_str!("coreml_expected_failures.txt")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
});

/// Returns whether a trial is a known failure for the selected backend.
///
/// Expected failures are still executed. Only a failing result is made non-fatal;
/// an unexpected pass remains a normal passing result.
pub fn is_expected_failure(backend: &str, trial_name: &str) -> bool {
    backend == "coreml" && COREML_EXPECTED_FAILURES.contains(trial_name)
}

#[cfg(test)]
mod tests {
    #[test]
    fn expected_failures_are_unique_and_well_formed() {
        let entries: Vec<_> = include_str!("coreml_expected_failures.txt")
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect();

        assert_eq!(entries.len(), super::COREML_EXPECTED_FAILURES.len());
        assert!(entries.iter().all(|entry| entry.starts_with("coreml::")));
    }

    #[test]
    fn expected_failures_are_scoped_to_coreml() {
        let trial = "coreml::abs::abs_float16_1D_constant_tensor";
        assert!(super::is_expected_failure("coreml", trial));
        assert!(!super::is_expected_failure("onnx", trial));
        assert!(!super::is_expected_failure("trtx", trial));
    }
}

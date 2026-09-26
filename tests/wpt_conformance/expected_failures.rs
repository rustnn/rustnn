//! Backend-specific WPT failures that are allowed without skipping execution.
//!
//! Every backend records conformance this way: a trial passes by completing, and fails
//! unless its sanitized libtest id appears in [`is_expected_failure`]'s list. There is no
//! PASS baseline besides these lists, so a listed trial that starts passing is not flagged
//! as such; regenerate the lists with the `wpt-sync-*` Makefile targets.

use std::collections::HashSet;
use std::sync::LazyLock;

static COREML_EXPECTED_FAILURES: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| parse_expected_failures(include_str!("coreml_expected_failures.txt")));

static LITERT_EXPECTED_FAILURES: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| parse_expected_failures(include_str!("litert_expected_failures.txt")));

static CANN_EXPECTED_FAILURES: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| parse_expected_failures(include_str!("cann_expected_failures.txt")));

static ONNX_EXPECTED_FAILURES: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| parse_expected_failures(include_str!("onnx_expected_failures.txt")));

static TRTX_EXPECTED_FAILURES: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| parse_expected_failures(include_str!("trtx_expected_failures.txt")));

fn parse_expected_failures(contents: &'static str) -> HashSet<&'static str> {
    contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
}

/// Returns whether a trial is a known failure for the selected backend.
///
/// Expected failures are still executed. Only a failing result is made non-fatal;
/// an unexpected pass remains a normal passing result.
pub fn is_expected_failure(backend: &str, trial_name: &str) -> bool {
    match backend {
        "coreml" => COREML_EXPECTED_FAILURES.contains(trial_name),
        "litert" => LITERT_EXPECTED_FAILURES.contains(trial_name),
        "cann" => CANN_EXPECTED_FAILURES.contains(trial_name),
        "onnx" => ONNX_EXPECTED_FAILURES.contains(trial_name),
        "trtx" => TRTX_EXPECTED_FAILURES.contains(trial_name),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn expected_failures_are_unique_and_well_formed() {
        let entries = |contents: &'static str| {
            contents
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .collect::<Vec<_>>()
        };
        let coreml_entries = entries(include_str!("coreml_expected_failures.txt"));
        let litert_entries = entries(include_str!("litert_expected_failures.txt"));
        let cann_entries = entries(include_str!("cann_expected_failures.txt"));
        let onnx_entries = entries(include_str!("onnx_expected_failures.txt"));
        let trtx_entries = entries(include_str!("trtx_expected_failures.txt"));

        assert_eq!(coreml_entries.len(), super::COREML_EXPECTED_FAILURES.len());
        assert!(
            coreml_entries
                .iter()
                .all(|entry| entry.starts_with("coreml::"))
        );
        assert_eq!(litert_entries.len(), super::LITERT_EXPECTED_FAILURES.len());
        assert!(
            litert_entries
                .iter()
                .all(|entry| entry.starts_with("litert::"))
        );
        assert_eq!(cann_entries.len(), super::CANN_EXPECTED_FAILURES.len());
        assert!(cann_entries.iter().all(|entry| entry.starts_with("cann::")));
        assert_eq!(onnx_entries.len(), super::ONNX_EXPECTED_FAILURES.len());
        assert!(onnx_entries.iter().all(|entry| entry.starts_with("onnx::")));
        assert_eq!(trtx_entries.len(), super::TRTX_EXPECTED_FAILURES.len());
        assert!(trtx_entries.iter().all(|entry| entry.starts_with("trtx::")));
    }

    #[test]
    fn expected_failures_are_scoped_to_their_backend() {
        let coreml_trial = "coreml::abs::abs_float16_1D_constant_tensor";
        let litert_trial = "litert::abs::abs_int8_4D_tensor";

        assert!(super::is_expected_failure("coreml", coreml_trial));
        assert!(!super::is_expected_failure("litert", coreml_trial));
        assert!(super::is_expected_failure("litert", litert_trial));
        assert!(!super::is_expected_failure("coreml", litert_trial));
        assert!(!super::is_expected_failure("onnx", litert_trial));
        assert!(!super::is_expected_failure("trtx", litert_trial));
    }
}

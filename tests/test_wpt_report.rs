//! Per-trial execution observations must not inherit a reused context's totals.

#[allow(dead_code)]
#[path = "wpt_conformance/wpt_config.rs"]
mod wpt_config;
#[allow(dead_code)]
#[path = "wpt_conformance/wpt_report.rs"]
mod wpt_report;
#[allow(dead_code)]
#[path = "wpt_conformance/wpt_types.rs"]
mod wpt_types;

use rustnn::mlcontext::{BackendStatistics, CoremlTensorStatistics};
use wpt_report::CoremlTrialStatistics;

fn snapshot(copies: u64, bytes: u64) -> Option<BackendStatistics> {
    Some(BackendStatistics::Coreml(CoremlTensorStatistics {
        proven_copy_outputs: copies,
        output_copy_bytes: bytes,
        ..Default::default()
    }))
}

#[test]
fn trial_statistics_distinguish_no_mixed_and_all_copies_after_context_reuse() {
    for (count, outputs) in [(0, 1), (1, 2), (2, 2)] {
        let report =
            CoremlTrialStatistics::between(snapshot(10, 80), snapshot(10 + count, 96), outputs)
                .unwrap();
        assert_eq!(report.logical_outputs, outputs);
        assert_eq!(report.proven_copy_outputs, count);
        assert_eq!(report.output_copy_bytes, 16);
        let json = serde_json::to_value(report).unwrap();
        assert_eq!(json["provenCopyOutputs"], count);
    }
}

#[test]
fn failed_dispatch_does_not_inherit_previous_successful_copy_count() {
    let previous = snapshot(2, 32);
    let report = CoremlTrialStatistics::between(previous, previous, 2).unwrap();
    assert_eq!(report.proven_copy_outputs, 0);
    assert_eq!(report.output_copy_bytes, 0);
    assert!(CoremlTrialStatistics::between(None, None, 1).is_none());
    assert!(CoremlTrialStatistics::between(snapshot(2, 32), snapshot(0, 0), 1).is_none());
}

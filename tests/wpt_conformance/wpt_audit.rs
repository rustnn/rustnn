//! WPT audit: per-pass error metrics vs tolerance (don't trust green).

use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::Serialize;

use super::tolerance::{FloatErrorMetrics, IntegerErrorMetrics, ToleranceKind, merged_ulp_minimum};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AuditCaseMetrics {
    pub test_name: String,
    pub file_name: String,
    pub operation: String,
    pub backend: String,
    pub tolerance_kind: String,
    pub tolerance_value: f64,
    pub tight_ulp_minimum: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_ulp: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_abs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_rtol: Option<f64>,
    pub nonfinite_mismatches: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_int_diff: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub integer_tolerance: Option<u64>,
    pub slack_ratio: Option<f64>,
    pub flagged: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub flag_reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    backend: String,
    strict_tolerance: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    coreml_requested_device: Option<String>,
    passed_cases: u64,
    flagged_cases: u64,
    cases: Vec<AuditCaseMetrics>,
}

#[derive(Clone)]
pub struct WptAuditCollector {
    inner: Arc<Mutex<AuditState>>,
}

#[derive(Debug, Default)]
struct AuditState {
    backend: String,
    cases: Vec<AuditCaseMetrics>,
}

impl WptAuditCollector {
    pub fn new(backend: &str) -> Self {
        Self {
            inner: Arc::new(Mutex::new(AuditState {
                backend: backend.to_string(),
                cases: Vec::new(),
            })),
        }
    }

    pub fn enabled() -> bool {
        std::env::var("WPT_AUDIT")
            .ok()
            .is_some_and(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
    }

    pub fn output_path() -> PathBuf {
        std::env::var("WPT_AUDIT_JSON")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("reports/wpt-trtx-audit.json"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_pass(
        &self,
        file_name: &str,
        test_name: &str,
        operation: &str,
        graph_operator_names: &[&str],
        applied_tolerance: (ToleranceKind, u64),
        float_metrics: Option<FloatErrorMetrics>,
        int_metrics: Option<IntegerErrorMetrics>,
        integer_tolerance: Option<u64>,
    ) {
        // Use the actual comparison budget, including any backend compatibility
        // adjustment. Recomputing it here can misreport a passing case's slack.
        let (kind, value) = applied_tolerance;
        let tight_ulp = merged_ulp_minimum(operation, graph_operator_names);
        let tolerance_value = tolerance_value_f64(kind, value);
        let tolerance_kind = format!("{kind:?}");

        let (max_ulp, max_abs, max_rtol, mut slack_ratio) = match kind {
            ToleranceKind::Ulp => {
                let error = float_metrics
                    .map(|metrics| u64::from(metrics.max_ulp))
                    .or_else(|| int_metrics.map(|metrics| metrics.max_abs_diff));
                let slack = error.map(|error| {
                    if value > 0 {
                        error as f64 / value as f64
                    } else if error == 0 {
                        0.0
                    } else {
                        f64::INFINITY
                    }
                });
                (
                    float_metrics.map(|metrics| metrics.max_ulp),
                    float_metrics.map(|metrics| metrics.max_abs),
                    float_metrics.map(|metrics| metrics.max_rtol),
                    slack,
                )
            }
            ToleranceKind::Atol => {
                let fm = float_metrics.unwrap_or_default();
                let atol = f64::from_bits(value);
                let slack = if atol > 0.0 {
                    Some(fm.max_abs / atol)
                } else {
                    Some(if fm.max_abs == 0.0 {
                        0.0
                    } else {
                        f64::INFINITY
                    })
                };
                (None, Some(fm.max_abs), Some(fm.max_rtol), slack)
            }
            ToleranceKind::Rtol => {
                let fm = float_metrics.unwrap_or_default();
                let rtol = f64::from_bits(value);
                let slack = if rtol > 0.0 {
                    Some(fm.max_rtol / rtol)
                } else {
                    Some(if fm.max_rtol == 0.0 {
                        0.0
                    } else {
                        f64::INFINITY
                    })
                };
                (None, Some(fm.max_abs), Some(fm.max_rtol), slack)
            }
        };

        let max_int_diff = int_metrics.map(|im| im.max_abs_diff);
        if let Some((metrics, budget)) = int_metrics.zip(integer_tolerance) {
            let integer_slack = if budget > 0 {
                metrics.max_abs_diff as f64 / budget as f64
            } else if metrics.max_abs_diff == 0 {
                0.0
            } else {
                f64::INFINITY
            };
            slack_ratio = Some(
                slack_ratio.map_or(integer_slack, |float_slack| float_slack.max(integer_slack)),
            );
        }

        let mut flag_reasons = Vec::new();
        if let Some(fm) = float_metrics {
            if matches!(kind, ToleranceKind::Ulp) && u64::from(fm.max_ulp) > tight_ulp {
                flag_reasons.push(format!(
                    "max_ulp {} exceeds local audit reference {}",
                    fm.max_ulp, tight_ulp
                ));
            }
            if value >= 1_000 && matches!(kind, ToleranceKind::Ulp) {
                flag_reasons.push(format!(
                    "wide ULP tolerance {} (operation minimum {})",
                    value, tight_ulp
                ));
            }
        }
        if let Some(im) = int_metrics
            && im.max_abs_diff > 0
            && integer_tolerance == Some(0)
        {
            flag_reasons.push(format!(
                "integer diff {} with zero tolerance (unexpected pass)",
                im.max_abs_diff
            ));
        }
        if let Some(slack) = slack_ratio
            && slack.is_finite()
            && slack >= 0.5
        {
            flag_reasons.push(format!(
                "uses {:.0}% of tolerance budget (close to edge)",
                slack * 100.0
            ));
        }

        let flagged = !flag_reasons.is_empty();
        let case = AuditCaseMetrics {
            test_name: test_name.to_string(),
            file_name: file_name.to_string(),
            operation: operation.to_string(),
            backend: self.inner.lock().expect("audit lock").backend.clone(),
            tolerance_kind,
            tolerance_value,
            tight_ulp_minimum: tight_ulp,
            max_ulp,
            max_abs,
            max_rtol,
            nonfinite_mismatches: float_metrics.map_or(0, |metrics| metrics.nonfinite_mismatches),
            max_int_diff,
            integer_tolerance,
            slack_ratio,
            flagged,
            flag_reasons,
        };
        self.inner.lock().expect("audit lock").cases.push(case);
    }

    pub fn write_json(&self) -> Result<PathBuf, String> {
        let state = self.inner.lock().expect("audit lock");
        let flagged_cases = state.cases.iter().filter(|c| c.flagged).count() as u64;
        let report = AuditReport {
            strict_tolerance: super::wpt_config::strict_wpt_tolerance(),
            coreml_requested_device: (state.backend == "coreml")
                .then(|| super::wpt_config::coreml_requested_device().to_string()),
            backend: state.backend.clone(),
            passed_cases: state.cases.len() as u64,
            flagged_cases,
            cases: state.cases.clone(),
        };
        let path = Self::output_path();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create audit dir {}: {e}", parent.display()))?;
        }
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| format!("failed to serialize audit report: {e}"))?;
        fs::write(&path, format!("{json}\n"))
            .map_err(|e| format!("failed to write audit report {}: {e}", path.display()))?;
        eprintln!(
            "[WPT audit] {} passed, {} flagged -> {}",
            report.passed_cases,
            report.flagged_cases,
            path.display()
        );
        Ok(path)
    }
}

fn tolerance_value_f64(kind: ToleranceKind, value: u64) -> f64 {
    match kind {
        ToleranceKind::Ulp => value as f64,
        ToleranceKind::Atol | ToleranceKind::Rtol => f64::from_bits(value),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn integer_only_audit_uses_integer_error_for_budget_slack() {
        let collector = super::WptAuditCollector::new("onnx");
        collector.record_pass(
            "cast.https.any.js",
            "integer test",
            "cast",
            &["cast"],
            (super::ToleranceKind::Ulp, 1),
            None,
            Some(super::IntegerErrorMetrics { max_abs_diff: 1 }),
            Some(1),
        );
        let state = collector.inner.lock().unwrap();
        let case = &state.cases[0];
        assert_eq!(case.max_int_diff, Some(1));
        assert_eq!(case.slack_ratio, Some(1.0));
        assert_eq!(case.max_ulp, None);
        assert_eq!(case.max_abs, None);
        assert_eq!(case.max_rtol, None);
        assert!(case.flagged);
        assert!(
            case.flag_reasons
                .iter()
                .any(|reason| reason.contains("100% of tolerance budget"))
        );
    }

    #[test]
    fn audit_records_applied_budget_instead_of_recomputing_local_minimum() {
        let collector = super::WptAuditCollector::new("cann");
        collector.record_pass(
            "gelu.https.any.js",
            "gelu test",
            "gelu",
            &["gelu"],
            (super::ToleranceKind::Ulp, 16_384),
            Some(super::FloatErrorMetrics {
                max_ulp: 8_192,
                ..Default::default()
            }),
            None,
            None,
        );
        let state = collector.inner.lock().unwrap();
        let case = &state.cases[0];
        assert_eq!(case.tolerance_value, 16_384.0);
        assert_eq!(case.slack_ratio, Some(0.5));
        assert!(
            case.flag_reasons
                .iter()
                .all(|reason| !reason.contains("passes only"))
        );
    }

    #[test]
    fn mixed_audit_uses_larger_fraction_with_each_applied_budget() {
        let collector = super::WptAuditCollector::new("cann");
        collector.record_pass(
            "mixed.https.any.js",
            "mixed test",
            "subgraph",
            &["gelu"],
            (super::ToleranceKind::Ulp, 16_384),
            Some(super::FloatErrorMetrics {
                max_ulp: 8_192,
                ..Default::default()
            }),
            Some(super::IntegerErrorMetrics { max_abs_diff: 1 }),
            Some(1),
        );
        let state = collector.inner.lock().unwrap();
        let case = &state.cases[0];
        assert_eq!(case.slack_ratio, Some(1.0));
        assert_eq!(case.tolerance_value, 16_384.0);
        assert_eq!(case.integer_tolerance, Some(1));
        assert_eq!(case.max_ulp, Some(8_192));
    }
}

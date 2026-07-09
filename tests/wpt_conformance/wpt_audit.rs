//! WPT audit: per-pass error metrics vs tolerance (don't trust green).

use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::Serialize;

use super::tolerance::{
    FloatErrorMetrics, IntegerErrorMetrics, ToleranceKind, get_operation_tolerance,
    merged_ulp_minimum,
};
use super::wpt_types::WptTolerance;

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
    pub max_abs: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_rtol: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_int_diff: Option<u64>,
    pub slack_ratio: Option<f64>,
    pub flagged: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub flag_reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    backend: String,
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
        tolerance_override: Option<&WptTolerance>,
        float_metrics: Option<FloatErrorMetrics>,
        int_metrics: Option<IntegerErrorMetrics>,
    ) {
        let (kind, value) =
            get_operation_tolerance(operation, tolerance_override, graph_operator_names);
        let tight_ulp = merged_ulp_minimum(operation, graph_operator_names);
        let tolerance_value = tolerance_value_f64(kind, value);
        let tolerance_kind = format!("{kind:?}");

        let (max_ulp, max_abs, max_rtol, slack_ratio) = match kind {
            ToleranceKind::Ulp => {
                let fm = float_metrics.unwrap_or_default();
                let slack = if value > 0 {
                    Some(fm.max_ulp as f64 / value as f64)
                } else {
                    Some(if fm.max_ulp == 0 { 0.0 } else { f64::INFINITY })
                };
                (Some(fm.max_ulp), Some(fm.max_abs), Some(fm.max_rtol), slack)
            }
            ToleranceKind::Atol => {
                let fm = float_metrics.unwrap_or_default();
                let atol = f64::from_bits(value) as f32;
                let slack = if atol > 0.0 {
                    Some((fm.max_abs / atol) as f64)
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
                let rtol = f64::from_bits(value) as f32;
                let slack = if rtol > 0.0 {
                    Some((fm.max_rtol / rtol) as f64)
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

        let mut flag_reasons = Vec::new();
        if let Some(fm) = float_metrics {
            if matches!(kind, ToleranceKind::Ulp) && fm.max_ulp > tight_ulp as u32 {
                flag_reasons.push(format!(
                    "max_ulp {} exceeds tight minimum {} (passes only with wider tolerance)",
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
            && tolerance_value == 0.0
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
            max_int_diff,
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

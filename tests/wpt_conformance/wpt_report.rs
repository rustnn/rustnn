//! Structured WPT conformance report (JSON), compatible with rustnnpt `reports/conformance.json`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use super::wpt_types::{WptCorpus, WptFileError};

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ReportMeta {
    started_at: String,
    ended_at: Option<String>,
    options: ReportOptions,
    cwd: String,
    rustnn: RustnnMeta,
    runner: &'static str,
}

#[derive(Debug, Serialize)]
struct ReportOptions {
    wpt_dir: String,
    backends: Vec<String>,
    #[serde(rename = "reportJson")]
    report_json: Option<String>,
    filter: Option<String>,
}

#[derive(Debug, Serialize)]
struct RustnnMeta {
    commit: Option<String>,
    #[serde(rename = "commitUrl")]
    commit_url: Option<String>,
}

#[derive(Debug, Default, Clone, Serialize)]
struct Summary {
    passed: u64,
    failed: u64,
    skipped: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    total: Option<u64>,
    #[serde(rename = "passRatePct", skip_serializing_if = "Option::is_none")]
    pass_rate_pct: Option<f64>,
    #[serde(
        rename = "passRateExcludingSkipsPct",
        skip_serializing_if = "Option::is_none"
    )]
    pass_rate_excluding_skips_pct: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FileReport {
    file_name: String,
    selected_tests: u64,
    summary: Summary,
    cases: Vec<CaseReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CaseReport {
    test_name: String,
    backend: String,
    variant: String,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    duration_ms: u64,
}

#[derive(Debug, Serialize)]
struct ConformanceReport {
    meta: ReportMeta,
    summary: Summary,
    files: Vec<FileReport>,
    failures: Vec<String>,
}

#[derive(Debug, Clone)]
struct FileState {
    selected_tests: u64,
    summary: Summary,
    cases: Vec<CaseReport>,
    file_error: Option<String>,
}

/// Thread-safe collector for per-case WPT results.
#[derive(Clone)]
pub struct WptReportCollector {
    inner: Arc<Mutex<CollectorState>>,
}

#[derive(Debug)]
struct CollectorState {
    started_at: String,
    wpt_dir: String,
    backends: Vec<String>,
    report_path: Option<String>,
    filter: Option<String>,
    files: HashMap<String, FileState>,
    failures: Vec<String>,
}

impl WptReportCollector {
    pub fn new(
        wpt_dir: impl Into<String>,
        backends: &[&str],
        corpus: &WptCorpus,
        args_filter: Option<String>,
        report_path: Option<PathBuf>,
    ) -> Self {
        let mut files = HashMap::new();
        for fe in &corpus.file_errors {
            insert_file_error(&mut files, fe);
        }
        for case in &corpus.cases {
            files
                .entry(case.file_name.clone())
                .or_insert_with(empty_file_state);
            let entry = files.get_mut(&case.file_name).expect("file entry");
            entry.selected_tests += 1;
        }

        Self {
            inner: Arc::new(Mutex::new(CollectorState {
                started_at: iso_timestamp_now(),
                wpt_dir: wpt_dir.into(),
                backends: backends.iter().map(|s| (*s).to_string()).collect(),
                report_path: report_path.as_ref().map(|p| p.display().to_string()),
                filter: args_filter,
                files,
                failures: Vec::new(),
            })),
        }
    }

    pub fn record_pass(
        &self,
        file_name: &str,
        test_name: &str,
        backend_prefix: &str,
        duration: Duration,
    ) {
        self.record_case(
            file_name,
            test_name,
            backend_prefix,
            "pass",
            None,
            None,
            duration,
        );
    }

    pub fn record_skip(
        &self,
        file_name: &str,
        test_name: &str,
        backend_prefix: &str,
        reason: impl Into<String>,
        duration: Duration,
    ) {
        self.record_case(
            file_name,
            test_name,
            backend_prefix,
            "skip",
            Some(reason.into()),
            None,
            duration,
        );
    }

    pub fn record_fail(
        &self,
        file_name: &str,
        test_name: &str,
        backend_prefix: &str,
        error: impl Into<String>,
        duration: Duration,
    ) {
        let error = error.into();
        let failure_line = format!(
            "{file_name} :: {backend_prefix}/{} :: {test_name} :: {error}",
            backend_variant(backend_prefix)
        );
        let mut state = self.inner.lock().expect("report collector lock");
        state.failures.push(failure_line);
        drop(state);
        self.record_case(
            file_name,
            test_name,
            backend_prefix,
            "fail",
            None,
            Some(error),
            duration,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn record_case(
        &self,
        file_name: &str,
        test_name: &str,
        backend_prefix: &str,
        status: &'static str,
        reason: Option<String>,
        error: Option<String>,
        duration: Duration,
    ) {
        let (backend, variant) = backend_and_variant(backend_prefix);
        let mut state = self.inner.lock().expect("report collector lock");
        let file = state
            .files
            .entry(file_name.to_string())
            .or_insert_with(empty_file_state);
        match status {
            "pass" => file.summary.passed += 1,
            "fail" => file.summary.failed += 1,
            "skip" => file.summary.skipped += 1,
            _ => {}
        }
        file.cases.push(CaseReport {
            test_name: test_name.to_string(),
            backend,
            variant,
            status,
            reason,
            error,
            duration_ms: duration.as_millis() as u64,
        });
        // Write incremental JSON report so partial results survive SIGABRT mid-suite.
        drop(state);
        self.write_json_incremental();
    }

    fn write_json_incremental(&self) {
        let path = match report_output_path() {
            Some(p) => p,
            None => return,
        };
        let report = self.build_report();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(&report) {
            let _ = fs::write(&path, format!("{json}\n"));
        }
    }

    pub fn write_json(&self) -> Result<Option<PathBuf>, String> {
        let path = match report_output_path() {
            Some(p) => p,
            None => return Ok(None),
        };

        let report = self.build_report();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "failed to create report directory {}: {e}",
                    parent.display()
                )
            })?;
        }
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| format!("failed to serialize WPT report: {e}"))?;
        fs::write(&path, format!("{json}\n"))
            .map_err(|e| format!("failed to write WPT report {}: {e}", path.display()))?;
        eprintln!("[WPT] JSON report written: {}", path.display());
        if let Err(e) = render_html_report(&path) {
            eprintln!("[WPT] warning: failed to write HTML report: {e}");
        }
        Ok(Some(path))
    }

    fn build_report(&self) -> ConformanceReport {
        let state = self.inner.lock().expect("report collector lock");

        let mut summary = Summary::default();
        let mut files: Vec<FileReport> = state
            .files
            .iter()
            .map(|(file_name, file)| {
                summary.passed += file.summary.passed;
                summary.failed += file.summary.failed;
                summary.skipped += file.summary.skipped;
                FileReport {
                    file_name: file_name.clone(),
                    selected_tests: file.selected_tests,
                    summary: file.summary.clone(),
                    cases: file.cases.clone(),
                    file_error: file.file_error.clone(),
                }
            })
            .collect();
        files.sort_by(|a, b| a.file_name.cmp(&b.file_name));

        let total = summary.passed + summary.failed + summary.skipped;
        summary.total = Some(total);
        let pass_rate = if total > 0 {
            (summary.passed as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        let denom_without_skips = summary.passed + summary.failed;
        let pass_rate_excluding_skips = if denom_without_skips > 0 {
            (summary.passed as f64 / denom_without_skips as f64) * 100.0
        } else {
            0.0
        };
        summary.pass_rate_pct = Some((pass_rate * 10.0).round() / 10.0);
        summary.pass_rate_excluding_skips_pct =
            Some((pass_rate_excluding_skips * 10.0).round() / 10.0);

        ConformanceReport {
            meta: ReportMeta {
                started_at: state.started_at.clone(),
                ended_at: Some(iso_timestamp_now()),
                options: ReportOptions {
                    wpt_dir: state.wpt_dir.clone(),
                    backends: state.backends.clone(),
                    report_json: state
                        .report_path
                        .clone()
                        .or_else(|| report_output_path().map(|p| p.display().to_string())),
                    filter: state.filter.clone(),
                },
                cwd: std::env::current_dir()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| ".".to_string()),
                rustnn: RustnnMeta {
                    commit: std::env::var("RUSTNN_GIT_SHA").ok(),
                    commit_url: std::env::var("RUSTNN_GIT_URL").ok(),
                },
                runner: "run_wpt_conformance",
            },
            summary,
            files,
            failures: state.failures.clone(),
        }
    }
}

fn empty_file_state() -> FileState {
    FileState {
        selected_tests: 0,
        summary: Summary::default(),
        cases: Vec::new(),
        file_error: None,
    }
}

fn insert_file_error(files: &mut HashMap<String, FileState>, fe: &WptFileError) {
    files.insert(
        fe.file_name.clone(),
        FileState {
            selected_tests: 0,
            summary: Summary {
                skipped: 1,
                ..Summary::default()
            },
            cases: Vec::new(),
            file_error: Some(fe.error.clone()),
        },
    );
}

fn backend_and_variant(backend_prefix: &str) -> (String, String) {
    match backend_prefix {
        "onnx" => ("onnx".to_string(), "cpu".to_string()),
        "onnx-gpu" => ("onnx".to_string(), "gpu".to_string()),
        "trtx" => ("trtx".to_string(), "trtx".to_string()),
        other => (other.to_string(), "default".to_string()),
    }
}

fn backend_variant(backend_prefix: &str) -> String {
    backend_and_variant(backend_prefix).1
}

/// Output path from `WPT_REPORT_JSON`, or `reports/wpt-conformance.json` when `CI` is set.
pub fn report_output_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("WPT_REPORT_JSON") {
        if path.is_empty() {
            return None;
        }
        return Some(PathBuf::from(path));
    }
    if std::env::var("CI").is_ok() {
        return Some(PathBuf::from("reports/wpt-conformance.json"));
    }
    None
}

fn report_html_output_path(json_path: &Path) -> PathBuf {
    if let Ok(path) = std::env::var("WPT_REPORT_HTML")
        && !path.is_empty()
    {
        return PathBuf::from(path);
    }
    json_path.with_extension("html")
}

fn render_html_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/wpt_bridge/render_conformance_html.mjs")
}

fn render_html_report(json_path: &Path) -> Result<(), String> {
    if std::env::var("WPT_REPORT_HTML").ok().as_deref() == Some("") {
        return Ok(());
    }
    let script = render_html_script();
    if !script.is_file() {
        return Err(format!(
            "HTML render script not found: {}",
            script.display()
        ));
    }
    let html_path = report_html_output_path(json_path);
    let status = Command::new("node")
        .arg(&script)
        .arg(json_path)
        .arg(&html_path)
        .status()
        .map_err(|e| format!("failed to spawn node for HTML report: {e}"))?;
    if !status.success() {
        return Err(format!("HTML report render failed (exit {status})"));
    }
    Ok(())
}

fn iso_timestamp_now() -> String {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    let millis = duration.subsec_millis();
    // ISO-8601 UTC without external chrono dependency.
    format_timestamp_utc(secs, millis)
}

fn format_timestamp_utc(secs: u64, millis: u32) -> String {
    // Reuse a minimal UTC formatter (good enough for reports).
    time_format::format_rfc3339_millis(secs, millis)
}

mod time_format {
    pub fn format_rfc3339_millis(secs: u64, millis: u32) -> String {
        let days = secs / 86_400;
        let rem = secs % 86_400;
        let hours = rem / 3_600;
        let rem = rem % 3_600;
        let minutes = rem / 60;
        let seconds = rem % 60;

        let (year, month, day) = civil_from_days(days as i64);
        format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}.{millis:03}Z")
    }

    fn civil_from_days(days: i64) -> (i64, i64, i64) {
        let z = days + 719_468;
        let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = mp + if mp < 10 { 3 } else { -9 };
        let year = y + if m <= 2 { 1 } else { 0 };
        (year, m, d)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn backend_mapping() {
        assert_eq!(
            super::backend_and_variant("onnx"),
            ("onnx".into(), "cpu".into())
        );
        assert_eq!(
            super::backend_and_variant("onnx-gpu"),
            ("onnx".into(), "gpu".into())
        );
    }
}

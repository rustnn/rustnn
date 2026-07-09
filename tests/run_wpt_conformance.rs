mod wpt_conformance;

use std::time::Instant;

use libtest_mimic::{Arguments, Completion, Failed, Trial};
use wpt_conformance::wpt_audit::WptAuditCollector;
use wpt_conformance::wpt_backend::WptBackend;
use wpt_conformance::wpt_js_loader::{
    default_wpt_dir, load_wpt_corpus, sanitize_test_id, trial_name,
};
use wpt_conformance::wpt_report::{WptReportCollector, report_output_path};
use wpt_conformance::wpt_types::WptLoadedCase;
use wpt_conformance::{
    run_one_test_case_with_audit, should_skip_backend_test, should_skip_test,
    wpt_types::WptTestCase,
};

fn run_trial(
    backend: &WptBackend,
    operation: &str,
    file_name: &str,
    test_case: &WptTestCase,
    audit: Option<&WptAuditCollector>,
) -> Result<Completion, Failed> {
    if let Some(reason) = should_skip_test(&test_case.graph) {
        return Ok(Completion::ignored_with(reason));
    }
    if let Some(reason) = should_skip_backend_test(backend.trial_prefix(), operation) {
        return Ok(Completion::ignored_with(reason));
    }

    run_one_test_case_with_audit(backend, operation, file_name, test_case, audit)
        .map(|()| Completion::Completed)
        .map_err(Failed::from)
}

fn push_backend_trials(
    trials: &mut Vec<Trial>,
    backend: WptBackend,
    cases: &[WptLoadedCase],
    report: &WptReportCollector,
    audit: Option<WptAuditCollector>,
) {
    let prefix = backend.trial_prefix();
    for case in cases {
        let operation = case.operation.clone();
        let test_case = case.as_test_case();
        let name = trial_name(prefix, case);
        let file_name = case.file_name.clone();
        let test_name = case.name.clone();
        let backend_prefix = prefix.to_string();
        let snapshot_name = format!("{backend_prefix}_{}", sanitize_test_id(&test_name));
        let report = report.clone();
        let backend = backend.clone();
        let audit = audit.clone();
        trials.push(Trial::ignorable_test(name, move || {
            let started = Instant::now();
            let result = run_trial(&backend, &operation, &file_name, &test_case, audit.as_ref());
            let duration = started.elapsed();

            match &result {
                Ok(Completion::Completed) => {
                    insta::assert_debug_snapshot!(
                        snapshot_name,
                        (&file_name, &test_name, &backend_prefix, "PASS")
                    );
                    report.record_pass(&file_name, &test_name, &backend_prefix, duration);
                }
                Ok(Completion::Ignored { reason }) => {
                    let reason = reason.as_deref().unwrap_or("ignored").to_string();
                    report.record_skip(&file_name, &test_name, &backend_prefix, reason, duration);
                }
                Err(err) => {
                    let msg = err.message().unwrap_or("test failed").to_string();
                    insta::assert_snapshot!(
                        snapshot_name,
                        format!("{file_name} {test_name}, {backend_prefix}\n {msg}")
                    );
                    report.record_fail(&file_name, &test_name, &backend_prefix, msg, duration);
                }
            }
            result
        }));
    }
}

fn main() {
    let _ = pretty_env_logger::try_init();
    let args = Arguments::from_args();
    let wpt_dir = default_wpt_dir();

    let corpus = match load_wpt_corpus(&wpt_dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            eprintln!();
            eprintln!("Ensure Node.js is on PATH and fetch WPT:");
            eprintln!("  node scripts/fetch_wpt.mjs");
            eprintln!("Or set WPT_DIR to an existing WPT checkout.");
            std::process::exit(2);
        }
    };

    eprintln!(
        "[WPT] loaded {} case(s) from {} via Node bridge",
        corpus.cases.len(),
        if corpus.wpt_dir.is_empty() {
            wpt_dir.display().to_string()
        } else {
            corpus.wpt_dir.clone()
        }
    );

    let backends = WptBackend::selected();
    if backends.is_empty() {
        eprintln!("No WPT backends available (enable onnx-runtime and/or trtx-runtime).");
        eprintln!("Set WPT_BACKEND=onnx or trtx to limit registered trials.");
        std::process::exit(2);
    }
    let backend_prefixes: Vec<&str> = backends.iter().map(|b| b.trial_prefix()).collect();
    eprintln!("[WPT] backends: {}", backend_prefixes.join(", "));

    let skip_eligible = corpus
        .cases
        .iter()
        .filter(|c| should_skip_test(&c.graph).is_some())
        .count()
        * backends.len();
    let trial_count = corpus.cases.len() * backends.len();
    eprintln!(
        "[WPT] registering {trial_count} trial(s) ({skip_eligible} dtype-skipped, {} executed)",
        trial_count.saturating_sub(skip_eligible)
    );
    if wpt_conformance::wpt_config::REUSE_ML_CONTEXT {
        eprintln!("[WPT] MLContext reuse: enabled (one context per backend per thread)");
    }

    let wpt_dir_label = if corpus.wpt_dir.is_empty() {
        wpt_dir.display().to_string()
    } else {
        corpus.wpt_dir.clone()
    };
    let report = WptReportCollector::new(
        wpt_dir_label,
        &backend_prefixes,
        &corpus,
        args.filter.clone(),
        report_output_path(),
    );

    let audit_enabled = WptAuditCollector::enabled();
    let audit = if audit_enabled {
        let collector =
            WptAuditCollector::new(backend_prefixes.first().copied().unwrap_or("unknown"));
        eprintln!(
            "[WPT audit] enabled -> {}",
            WptAuditCollector::output_path().display()
        );
        Some(collector)
    } else {
        None
    };

    let mut trials = Vec::new();
    for backend in backends {
        let backend_audit = audit.clone();
        push_backend_trials(&mut trials, backend, &corpus.cases, &report, backend_audit);
    }

    let conclusion = libtest_mimic::run(&args, trials);
    eprintln!(
        "[WPT] result: {} passed, {} skipped, {} failed; {} filtered out",
        conclusion.num_passed,
        conclusion.num_ignored,
        conclusion.num_failed,
        conclusion.num_filtered_out
    );

    if let Err(e) = report.write_json() {
        eprintln!("[WPT] warning: failed to write JSON report: {e}");
    }

    if let Some(audit) = audit
        && let Err(e) = audit.write_json()
    {
        eprintln!("[WPT] warning: failed to write audit report: {e}");
    }

    conclusion.exit();
}

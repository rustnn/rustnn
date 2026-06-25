mod wpt_conformance;

use libtest_mimic::{Arguments, Completion, Failed, Trial};
use wpt_conformance::wpt_backend::WptBackend;
use wpt_conformance::wpt_js_loader::{default_wpt_dir, load_wpt_corpus, trial_name};
use wpt_conformance::wpt_types::WptLoadedCase;
use wpt_conformance::{should_skip_test, wpt_types::WptTestCase};

#[cfg(any(
    feature = "onnx-runtime",
    feature = "trtx-runtime",
    feature = "trtx-runtime-mock"
))]
use wpt_conformance::run_one_test_case;

fn run_trial(
    backend: WptBackend,
    operation: &str,
    test_case: &WptTestCase,
) -> Result<Completion, Failed> {
    if let Some(reason) = should_skip_test(&test_case.graph) {
        return Ok(Completion::ignored_with(reason));
    }

    run_one_test_case(backend, operation, test_case)
        .map(|()| Completion::Completed)
        .map_err(Failed::from)
}

fn push_backend_trials(
    trials: &mut Vec<Trial>,
    backend: WptBackend,
    cases: &[WptLoadedCase],
) {
    let prefix = backend.trial_prefix();
    for case in cases {
        let operation = case.operation.clone();
        let test_case = case.as_test_case();
        let name = trial_name(prefix, case);
        trials.push(Trial::ignorable_test(name, move || {
            run_trial(backend, &operation, &test_case)
        }));
    }
}

fn main() {
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
        eprintln!(
            "No WPT backends available (enable onnx-runtime and/or trtx-runtime-mock feature)."
        );
        eprintln!("Set WPT_BACKEND=onnx, onnx-gpu, or trtx to limit registered trials.");
        std::process::exit(2);
    }

    eprintln!(
        "[WPT] backends: {}",
        backends
            .iter()
            .map(|b| b.trial_prefix())
            .collect::<Vec<_>>()
            .join(", ")
    );

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

    let mut trials = Vec::new();
    for backend in backends {
        push_backend_trials(&mut trials, backend, &corpus.cases);
    }

    let conclusion = libtest_mimic::run(&args, trials);
    eprintln!(
        "[WPT] result: {} passed, {} skipped, {} failed; {} filtered out",
        conclusion.num_passed,
        conclusion.num_ignored,
        conclusion.num_failed,
        conclusion.num_filtered_out
    );
    conclusion.exit();
}

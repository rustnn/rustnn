#![cfg(all(target_arch = "wasm32", feature = "webnn-wpt-tests"))]
#![allow(dead_code)] // Shared WPT translator exposes helpers used by the native harness.

use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::*;

#[path = "wpt_conformance/webnn_chrome_expected_failures.rs"]
mod webnn_chrome_expected_failures;
#[path = "wpt_conformance/wpt_execute_graph.rs"]
mod wpt_execute_graph;
#[path = "wpt_conformance/wpt_tensor.rs"]
mod wpt_tensor;
#[path = "wpt_conformance/wpt_types.rs"]
mod wpt_types;

wasm_bindgen_test_configure!(run_in_browser);

const WPT_CORPUS: &str = include_str!(concat!(env!("OUT_DIR"), "/webnn-wpt-corpus.json"));

#[wasm_bindgen_test]
fn embedded_wpt_cases_compile_to_graph_info() {
    let corpus: wpt_types::WptCorpus =
        serde_json::from_str(WPT_CORPUS).expect("embedded WPT corpus is JSON");

    assert!(!corpus.cases.is_empty(), "embedded WPT corpus has no cases");
    assert_eq!(corpus.cases.len(), 2482, "embedded WPT corpus case count");

    let failures: Vec<String> = corpus
        .cases
        .iter()
        .filter_map(|case| {
            wpt_execute_graph::compile_wpt_graph(&case.graph)
                .err()
                .map(|error| format!("{}::{}: {error}", case.operation, case.name))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "WPT graph compilation failures:\n{}",
        failures.join("\n")
    );
}

#[wasm_bindgen_test]
async fn embedded_wpt_cases_build_in_browser_webnn() {
    let corpus: wpt_types::WptCorpus =
        serde_json::from_str(WPT_CORPUS).expect("embedded WPT corpus is JSON");

    let Some(context) = create_webnn_context()
        .await
        .expect("read browser WebNN API")
    else {
        log_result("SKIP", "browser", "navigator.ml is unavailable");
        return;
    };
    let mut failures = Vec::new();
    let mut passed = 0usize;
    let mut skipped = 0usize;
    for case in &corpus.cases {
        let case_id = format!("{}::{}", case.operation, case.name);
        if let Some(reason) = webnn_chrome_expected_failures::reason(&case.operation, &case.name) {
            skipped += 1;
            log_result("SKIP", &case_id, reason);
            continue;
        }
        // Log before calling a generated WebNN method: current upstream bindings do
        // not mark those methods `catch`, so a browser exception traps out of Wasm.
        log_result("RUN", &case_id, "");
        let result = match wpt_execute_graph::compile_wpt_graph(&case.graph) {
            Ok(graph) => rustnn::converters::webnn::convert_async(&context, &graph)
                .await
                .map(|_| ())
                .map_err(|error| error.to_string()),
            Err(error) => Err(error),
        };
        if let Err(error) = result {
            log_result("FAIL", &case_id, &error);
            failures.push(format!("{case_id}: {error}"));
        } else {
            passed += 1;
            log_result("PASS", &case_id, "");
        }
    }
    log_result(
        "SUMMARY",
        "browser WebNN WPT",
        &format!(
            "{passed} passed, {skipped} expected failures skipped, {} unexpected failures, {} total",
            failures.len(),
            corpus.cases.len()
        ),
    );
    assert!(
        failures.is_empty(),
        "unexpected browser WebNN WPT failures:\n{}",
        failures.join("\n")
    );
}

fn log_result(status: &str, case: &str, detail: &str) {
    let message = if detail.is_empty() {
        format!("[WEBNN-WPT {status}] {case}")
    } else {
        format!("[WEBNN-WPT {status}] {case}: {detail}")
    };
    web_sys::console::log_1(&message.into());
}

async fn create_webnn_context()
-> Result<Option<rustnn::backends::webnn::api_generated::MlContext>, JsValue> {
    use rustnn::backends::webnn::api_generated::{
        Ml, MlContext, MlContextOptions, MlPowerPreference,
    };

    let navigator = web_sys::window().expect("browser window").navigator();
    let ml_value = js_sys::Reflect::get(&navigator, &"ml".into())?;
    if ml_value.is_undefined() || ml_value.is_null() {
        return Ok(None);
    }
    let ml = ml_value.dyn_into::<Ml>()?;
    let options = MlContextOptions::new();
    options.set_accelerated(true);
    options.set_power_preference(MlPowerPreference::HighPerformance);
    Ok(Some(
        JsFuture::from(ml.create_context_with_ml_context_options(&options))
            .await?
            .dyn_into::<MlContext>()?,
    ))
}

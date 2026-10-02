//! Comparison budgets must distinguish compatibility and upstream-strict runs.

#[allow(dead_code)]
#[path = "wpt_conformance/tolerance.rs"]
mod tolerance;
#[allow(dead_code)]
#[path = "wpt_conformance/wpt_audit.rs"]
mod wpt_audit;
#[allow(dead_code)]
#[path = "wpt_conformance/wpt_backend.rs"]
mod wpt_backend;
#[allow(dead_code)]
#[path = "wpt_conformance/wpt_config.rs"]
mod wpt_config;
#[allow(dead_code)]
#[path = "wpt_conformance/wpt_js_loader.rs"]
mod wpt_js_loader;
#[allow(dead_code)]
#[path = "wpt_conformance/wpt_tensor.rs"]
mod wpt_tensor;
#[allow(dead_code)]
#[path = "wpt_conformance/wpt_types.rs"]
mod wpt_types;

use tolerance::{ToleranceKind, check_ulp_tolerance, get_operation_tolerance_with_mode};
use wpt_types::WptTolerance;

#[test]
fn strict_wpt_budget_does_not_use_local_operator_minimum() {
    let source = WptTolerance {
        metric_type: "ULP".into(),
        value: 2.into(),
    };
    assert_eq!(
        get_operation_tolerance_with_mode("matmul", Some(&source), &["matmul"], false),
        (ToleranceKind::Ulp, 512)
    );
    assert_eq!(
        get_operation_tolerance_with_mode("matmul", Some(&source), &["matmul"], true),
        (ToleranceKind::Ulp, 2)
    );
}

#[test]
fn strict_wpt_budget_rejects_half_gelu_tail_error_hidden_by_absolute_floor() {
    let actual = [-0.003_875_732_4f32];
    let expected = [-0.004_051_208_5f32];
    assert!(check_ulp_tolerance(&actual, &expected, 18, true, true).0);
    assert!(!check_ulp_tolerance(&actual, &expected, 18, true, false).0);
    assert_eq!(tolerance::ulp_distance_f16(actual[0], expected[0]), 54);
}

#[test]
fn strict_wpt_budget_rejects_float_error_hidden_by_absolute_floor() {
    let actual = [1e-6f32];
    let expected = [0.0f32];
    assert!(check_ulp_tolerance(&actual, &expected, 0, false, true).0);
    assert!(!check_ulp_tolerance(&actual, &expected, 0, false, false).0);
}

#[test]
fn strict_wpt_budget_rejects_missing_or_malformed_source_tolerance() {
    assert!(tolerance::validate_strict_source_tolerance(None).is_err());
    for (metric, value) in [
        ("ULP", serde_json::json!(-1)),
        ("ULP", serde_json::json!(0.5)),
        ("other", serde_json::json!(1)),
        ("ATOL", serde_json::json!("unknown")),
        ("RTOL", serde_json::json!(0.1)),
        ("ULP", serde_json::json!(9_007_199_254_740_992u64)),
        ("ULP", serde_json::json!(u64::MAX)),
    ] {
        let source = WptTolerance {
            metric_type: metric.into(),
            value,
        };
        assert!(tolerance::validate_strict_source_tolerance(Some(&source)).is_err());
    }
    let source = WptTolerance {
        metric_type: "ULP".into(),
        value: 18.into(),
    };
    assert!(tolerance::validate_strict_source_tolerance(Some(&source)).is_ok());
}

#[test]
fn strict_half_metric_matches_upstream_raw_bits_and_halfway_rounding() {
    assert_eq!(tolerance::upstream_ulp_distance(-1.0, 1.0, true), 32768);
    assert_eq!(tolerance::upstream_ulp_distance(-0.0, 0.0, true), 0);
    assert_eq!(tolerance::upstream_ulp_distance(0.0, -0.0, true), 0);
    // Upstream toHalf rounds this tie upward; IEEE ties-to-even would give 1.
    assert_eq!(
        tolerance::upstream_ulp_distance(1.0, 1.00048828125, true),
        1
    );
    assert_eq!(
        tolerance::upstream_ulp_distance(-1.0, -1.00048828125, true),
        1
    );
    let subnormal = half::f16::from_bits(1).to_f32();
    assert_eq!(tolerance::upstream_ulp_distance(subnormal, 0.0, true), 1);
    assert_eq!(
        tolerance::upstream_ulp_distance(-subnormal, 0.0, true),
        32769
    );
}

#[test]
fn strict_atol_keeps_source_precision_and_does_not_preround_expected_half() {
    let expected = 1.0 + 2.0f64.powi(-30);
    assert!(
        !tolerance::validate_upstream_result(
            &[1.0],
            &[expected],
            ToleranceKind::Atol,
            0.0f64.to_bits(),
            false
        )
        .0
    );
    assert!(
        tolerance::validate_upstream_result(
            &[1.0],
            &[expected],
            ToleranceKind::Atol,
            2.0f64.powi(-30).to_bits(),
            false
        )
        .0
    );
    assert!(
        !tolerance::validate_upstream_result(
            &[1.0],
            &[1.00048828125],
            ToleranceKind::Atol,
            0.0f64.to_bits(),
            true
        )
        .0
    );
}

#[test]
fn strict_nonfinite_gate_cannot_be_overridden_by_large_budget() {
    for kind in [ToleranceKind::Ulp, ToleranceKind::Atol] {
        let budget = if kind == ToleranceKind::Ulp {
            u64::MAX
        } else {
            f64::MAX.to_bits()
        };
        for (actual, expected) in [
            (f32::NAN, 0.0),
            (f32::INFINITY, 1.0),
            (0.0, f64::NAN),
            (f32::INFINITY, f64::NEG_INFINITY),
        ] {
            for half in [true, false] {
                assert!(
                    !tolerance::validate_upstream_result(
                        &[actual],
                        &[expected],
                        kind,
                        budget,
                        half
                    )
                    .0
                );
            }
        }
        for (actual, expected) in [
            (f32::NAN, f64::NAN),
            (f32::INFINITY, f64::INFINITY),
            (f32::NEG_INFINITY, f64::NEG_INFINITY),
        ] {
            assert!(
                tolerance::validate_upstream_result(&[actual], &[expected], kind, budget, false).0
            );
        }
    }
}

#[test]
fn compatibility_comparators_retain_default_behavior() {
    // Strict nonfinite rejection is opt-in; this PR must not change the
    // historical comparisons used by unqualified backend configurations.
    assert!(tolerance::check_atol_tolerance(&[f32::NAN], &[0.0], 0.1).0);
    assert!(tolerance::check_rtol_tolerance(&[f32::NAN], &[0.0], 0.1).0);
    assert!(!check_ulp_tolerance(&[1.0], &[0.0], 1u64 << 32, false, false).0);
    assert!(!tolerance::check_atol_tolerance(&[1.0], &[0.0], 0.1).0);
    assert!(!tolerance::check_rtol_tolerance(&[1.0], &[0.0], 0.1).0);
}

#[test]
fn strict_large_ulp_budget_is_not_truncated_to_u32() {
    assert!(
        tolerance::validate_upstream_result(&[1.0], &[0.0], ToleranceKind::Ulp, 1u64 << 32, false)
            .0
    );
}

#[test]
fn audit_uses_same_strict_half_metric_and_source_precision() {
    let metrics = tolerance::upstream_float_error_metrics(
        &[-1.0, 1.0],
        &[1.0, 1.00048828125],
        true,
        ToleranceKind::Ulp,
    );
    assert_eq!(metrics.max_ulp, 32768);
    assert_eq!(metrics.max_abs, 2.0);
    let metrics = tolerance::upstream_float_error_metrics(
        &[1.0],
        &[1.0 + 2.0f64.powi(-30)],
        false,
        ToleranceKind::Atol,
    );
    assert_eq!(metrics.max_abs, 2.0f64.powi(-30));
    let metrics =
        tolerance::upstream_float_error_metrics(&[f32::NAN], &[1.0], false, ToleranceKind::Ulp);
    assert_eq!(metrics.nonfinite_mismatches, 1);
    assert!(metrics.max_abs.is_infinite());
    let metrics = tolerance::float_error_metrics(&[f32::NAN], &[1.0], false);
    assert_eq!(metrics.nonfinite_mismatches, 1);
    let metrics = tolerance::upstream_float_error_metrics(
        &[f32::NAN, f32::INFINITY],
        &[f64::NAN, f64::INFINITY],
        true,
        ToleranceKind::Ulp,
    );
    assert_eq!(metrics.max_ulp, 0);
    assert_eq!(metrics.max_abs, 0.0);
    assert_eq!(metrics.nonfinite_mismatches, 0);
}

#[test]
fn strict_expected_values_preserve_numbers_and_reject_malformed_data() {
    let spec = |data| wpt_types::WptTensorSpec {
        data,
        shape: vec![1],
        data_type: "float16".into(),
        constant: false,
        descriptor: None,
    };
    assert_eq!(
        wpt_tensor::expected_output_to_f64(&spec(serde_json::json!([1.00048828125]))).unwrap(),
        vec![1.00048828125]
    );
    for invalid in [
        serde_json::json!([]),
        serde_json::json!([1, 2]),
        serde_json::json!([null]),
        serde_json::json!(["invalid"]),
    ] {
        assert!(wpt_tensor::expected_output_to_f64(&spec(invalid)).is_err());
    }
}

#[test]
fn strict_half_inputs_and_constants_use_source_rounding_without_changing_default() {
    for constant in [false, true] {
        let spec = wpt_types::WptTensorSpec {
            data: serde_json::json!([1.00048828125, -1.00048828125]),
            shape: vec![2],
            data_type: "float16".into(),
            constant,
            descriptor: None,
        };
        let compatibility = wpt_tensor::tensor_f16_bits_with_mode(&spec, false);
        let strict = wpt_tensor::tensor_f16_bits_with_mode(&spec, true);
        assert_eq!(compatibility, [0x3c00, 0xbc00]);
        assert_eq!(strict, [0x3c01, 0xbc01]);
        assert_eq!(
            wpt_tensor::tensor_spec_to_bytes_with_mode(&spec, true).unwrap(),
            strict
                .iter()
                .flat_map(|bits| bits.to_ne_bytes())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            wpt_tensor::tensor_spec_to_bytes_with_mode(&spec, false).unwrap(),
            compatibility
                .iter()
                .flat_map(|bits| bits.to_ne_bytes())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn strict_unsigned_integer_budgets_survive_signed_storage_boundaries() {
    let actual = tolerance::unsigned_32_values(&[i32::MIN]);
    let expected = tolerance::unsigned_32_values(&[i32::MAX]);
    assert!(tolerance::check_unsigned_tolerance(&actual, &expected, 1).0);
    assert!(!tolerance::check_unsigned_tolerance(&actual, &expected, 0).0);
    assert_eq!(
        tolerance::unsigned_error_metrics(&actual, &expected).max_abs_diff,
        1
    );
    for (actual, expected) in [(1u64 << 63, i64::MAX as u64), (u64::MAX, u64::MAX - 1)] {
        assert!(tolerance::check_unsigned_tolerance(&[actual], &[expected], 1).0);
        assert!(!tolerance::check_unsigned_tolerance(&[actual], &[expected], 0).0);
        assert_eq!(
            tolerance::unsigned_error_metrics(&[actual], &[expected]).max_abs_diff,
            1
        );
    }
    assert!(!tolerance::check_unsigned_tolerance(&[0], &[u64::MAX], 1).0);
}

#[test]
fn strict_ulp_classifies_expected_values_after_dtype_conversion() {
    for (actual, expected, float16) in [
        (f32::INFINITY, 65520.0, true),
        (f32::NEG_INFINITY, -65520.0, true),
        (f32::INFINITY, 1e39, false),
        (f32::NEG_INFINITY, -1e39, false),
    ] {
        assert!(
            tolerance::validate_upstream_result(
                &[actual],
                &[expected],
                ToleranceKind::Ulp,
                0,
                float16
            )
            .0
        );
        assert!(
            !tolerance::validate_upstream_result(
                &[actual],
                &[expected],
                ToleranceKind::Atol,
                f64::MAX.to_bits(),
                float16
            )
            .0
        );
        let metrics = tolerance::upstream_float_error_metrics(
            &[actual],
            &[expected],
            float16,
            ToleranceKind::Ulp,
        );
        assert_eq!(metrics.nonfinite_mismatches, 0);
        assert_eq!(metrics.max_ulp, 0);
        let metrics = tolerance::upstream_float_error_metrics(
            &[actual],
            &[expected],
            float16,
            ToleranceKind::Atol,
        );
        assert_eq!(metrics.nonfinite_mismatches, 1);
    }
    assert!(
        !tolerance::validate_upstream_result(
            &[f32::INFINITY],
            &[65504.0],
            ToleranceKind::Ulp,
            u64::MAX,
            true
        )
        .0
    );
}

#[test]
#[cfg(not(feature = "coreml-runtime"))]
fn disabled_coreml_remains_unavailable_without_a_device_hint_panic() {
    assert!(
        wpt_backend::WptBackend::selected()
            .iter()
            .all(|backend| backend.trial_prefix() != "coreml")
    );
}

#[test]
#[ignore = "requires Node.js and the fetched upstream WPT corpus"]
fn strict_comparator_matches_upstream_javascript() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let wpt = std::env::var_os("WPT_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| root.join(".cache/wpt"));
    let utils = wpt.join("webnn/resources/utils.js");
    let bridge = root.join("scripts/wpt_bridge/wpt-tolerance-env.mjs");
    // Expected values are JS Numbers; actual half values are raw Uint16 payloads.
    let mut cases = vec![
        (1.0f32, 1.00048828125, true),
        (-1.0, -1.00048828125, true),
        (-1.0, 1.0, true),
        (-0.0, 0.0, true),
        (0.0, -0.0, true),
        (half::f16::from_bits(1).to_f32(), 0.0, true),
        (-half::f16::from_bits(1).to_f32(), 0.0, true),
        (
            half::f16::from_bits(2).to_f32(),
            3.0 * 2.0f64.powi(-25),
            true,
        ),
        (65504.0, 65504.0, true),
        (f32::INFINITY, 65520.0, true),
        (f32::NEG_INFINITY, -65520.0, true),
        (f32::INFINITY, 1e39, false),
        (f32::NEG_INFINITY, -1e39, false),
        (1.0, 1.0 + 2.0f64.powi(-23), false),
        (-1.0, 1.0, false),
        (-0.0, 0.0, false),
        (f32::from_bits(1), 0.0, false),
        (-f32::from_bits(1), 0.0, false),
    ];
    // Every finite FP16 encoding, including both signs and the entire
    // subnormal range. Adjacent midpoints exercise the source toHalf helper's
    // rounding; opposite signs exercise its raw-bit distance convention.
    for bits in 0..=u16::MAX {
        let value = half::f16::from_bits(bits);
        if !value.is_finite() {
            continue;
        }
        cases.push((value.to_f32(), value.to_f64(), true));
        cases.push((value.to_f32(), -value.to_f64(), true));
        if let Some(next) = bits.checked_add(1).map(half::f16::from_bits)
            && next.is_finite()
        {
            cases.push((value.to_f32(), (value.to_f64() + next.to_f64()) * 0.5, true));
        }
    }
    let input: Vec<_> = cases.iter().map(|&(actual, expected, float16)| {
        serde_json::json!({"actualBits": actual.to_bits(), "halfBits": half::f16::from_f32(actual).to_bits(), "expected": expected, "float16": float16})
    }).collect();
    let atol_cases = [
        (1.0f32, 1.0 + 2.0f64.powi(-30), 0.0),
        (1.0, 1.0 + 2.0f64.powi(-30), 2.0f64.powi(-30)),
        (1.0, 1.00048828125, 0.0004),
        (1.0, 1.00048828125, 0.0005),
    ];
    let atol_input: Vec<_> = atol_cases.iter().map(|&(actual, expected, budget)| {
        serde_json::json!({"actual": actual, "expected": expected, "budget": budget})
    }).collect();
    let script = r#"
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
const {createWptToleranceContext} = await import(pathToFileURL(process.argv[2]));
const {context} = createWptToleranceContext(process.argv[1]);
const input = JSON.parse(readFileSync(0, 'utf8'));
context.cases = input.ulp;
const result = vm.runInContext(`cases.map(c => {
 const actual = c.float16 ? c.halfBits : new Float32Array(new Uint32Array([c.actualBits]).buffer)[0];
 return Number(ulpDistance(actual, c.expected, c.float16 ? 'float16' : 'float32'));
})`, context);
// ATOL is applied by testharness.js, not utils.js: check JS Number arithmetic.
const atol = input.atol.map(c => Math.abs(c.actual - c.expected) <= c.budget);
process.stdout.write(JSON.stringify({ulp: result, atol}));
"#;
    let payload = serde_json::json!({"ulp": input, "atol": atol_input});
    let mut child = std::process::Command::new("node")
        .args(["--input-type=module", "-e", script])
        .arg(&utils)
        .arg(&bridge)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("run Node.js");
    use std::io::Write;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(payload.to_string().as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let results: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    // Optional artifact for running these same vectors through testharness.js
    // and utils.js in a browser, without reimplementing either assertion.
    if let Some(path) = std::env::var_os("WPT_TOLERANCE_PARITY_JSON") {
        std::fs::write(
            path,
            serde_json::to_vec(&serde_json::json!({
                "input": payload, "expected": results
            }))
            .unwrap(),
        )
        .unwrap();
    }
    let distances: Vec<u32> = serde_json::from_value(results["ulp"].clone()).unwrap();
    assert_eq!(cases.len(), distances.len());
    for ((actual, expected, float16), upstream) in cases.into_iter().zip(distances) {
        assert_eq!(
            tolerance::upstream_ulp_distance(actual, expected, float16),
            upstream,
            "actual={actual} expected={expected} float16={float16}"
        );
    }
    let atol_results: Vec<bool> = serde_json::from_value(results["atol"].clone()).unwrap();
    assert_eq!(atol_cases.len(), atol_results.len());
    for ((actual, expected, budget), js_pass) in atol_cases.into_iter().zip(atol_results) {
        assert_eq!(
            tolerance::validate_upstream_result(
                &[actual],
                &[expected],
                ToleranceKind::Atol,
                budget.to_bits(),
                false
            )
            .0,
            js_pass
        );
    }
}

#[test]
#[ignore = "requires Node.js and the fetched WPT corpus; run make test-wpt-tolerance-parity"]
fn strict_comparator_matches_upstream_javascript_intermediate_budgets() {
    use serde_json::json;
    let resolve = |name, descriptors| {
        wpt_js_loader::resolve_source_tolerance("subgraph.https.any.js", name, &descriptors)
    };
    assert!(resolve("gatherElements + matmul", json!({})).is_err());
    let source = resolve(
        "gatherElements + matmul",
        json!({"gatherElementsOutput": {"shape": [2, 3], "dataType": "float32"}}),
    )
    .unwrap();
    assert_eq!(source.metric_type, "ULP");
    assert_eq!(source.value, 6);
    let source = resolve(
        "reshape + conv2d default/ float16",
        json!({"reshapeOutput": {"shape": [1, 1, 3, 3], "dataType": "float16"}}),
    )
    .unwrap();
    assert_eq!(source.value, 18);
    // This source callback omits an int32 budget. Do not substitute the
    // float32 allowance, compatibility default, or an invented exact budget.
    let error = wpt_js_loader::resolve_source_tolerance(
        "div.https.any.js",
        "div int32 4D tensors",
        &json!({}),
    )
    .unwrap_err();
    assert!(
        error.contains("Upstream tolerance callback returned no finite budget"),
        "{error}"
    );
}

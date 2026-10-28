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
#[path = "wpt_conformance/wpt_execute_graph.rs"]
mod wpt_execute_graph;
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
fn strict_half_metric_matches_upstream_signed_bits_and_halfway_rounding() {
    // Signed-magnitude ordering, as for float32: 2 * 0x3c00.
    assert_eq!(tolerance::upstream_ulp_distance(-1.0, 1.0, true), 30720);
    assert_eq!(tolerance::upstream_ulp_distance(-0.0, 0.0, true), 0);
    assert_eq!(tolerance::upstream_ulp_distance(0.0, -0.0, true), 0);
    // Float16Array rounds this tie to even (1.0).
    assert_eq!(
        tolerance::upstream_ulp_distance(1.0, 1.00048828125, true),
        0
    );
    assert_eq!(
        tolerance::upstream_ulp_distance(-1.0, -1.00048828125, true),
        0
    );
    // Just above the tie: direct f64 rounding goes up, while rounding through
    // f32 first would land on the tie and round down.
    let above_tie = 1.0 + 2.0f64.powi(-11) + 2.0f64.powi(-40);
    assert_eq!(tolerance::upstream_ulp_distance(1.0, above_tie, true), 1);
    let subnormal = half::f16::from_bits(1).to_f32();
    assert_eq!(tolerance::upstream_ulp_distance(subnormal, 0.0, true), 1);
    assert_eq!(tolerance::upstream_ulp_distance(-subnormal, 0.0, true), 1);
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
    // This is a RustNN model-validation requirement pending agreement on WPT's
    // nonfinite classification rules, not an additional upstream ULP budget.
    // WPT's raw metric puts maximum finite and same-sign infinity one ULP apart;
    // accepting that transition can poison all subsequent recurrent state.
    // Keep this local until classification is agreed, then contribute the reduced
    // comparator case rather than imposing a model's final-output tolerance.
    for float16 in [false, true] {
        let maximum = if float16 {
            half::f16::MAX.to_f32()
        } else {
            f32::MAX
        };
        let adjacent = if float16 {
            half::f16::from_bits(half::f16::MAX.to_bits() - 1).to_f32()
        } else {
            f32::from_bits(f32::MAX.to_bits() - 1)
        };
        for sign in [-1.0, 1.0] {
            for (actual, expected) in [
                (sign * maximum, f64::from(sign * f32::INFINITY)),
                (sign * f32::INFINITY, f64::from(sign * maximum)),
            ] {
                assert_eq!(
                    tolerance::upstream_ulp_distance(actual, expected, float16),
                    1
                );
                for budget in [1, u64::MAX] {
                    let (passed, diagnostic) = tolerance::validate_upstream_result(
                        &[actual],
                        &[expected],
                        ToleranceKind::Ulp,
                        budget,
                        float16,
                    );
                    assert!(!passed, "{actual} versus {expected}, float16={float16}");
                    assert!(diagnostic.unwrap().contains("nonfinite classification"));
                }
                let metrics = tolerance::upstream_float_error_metrics(
                    &[actual],
                    &[expected],
                    float16,
                    ToleranceKind::Ulp,
                );
                assert_eq!(metrics.nonfinite_mismatches, 1);
            }
            // Adjacent finite values remain eligible for the source ULP budget.
            assert!(
                tolerance::validate_upstream_result(
                    &[sign * adjacent],
                    &[f64::from(sign * maximum)],
                    ToleranceKind::Ulp,
                    1,
                    float16,
                )
                .0
            );
        }
    }
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
    assert_eq!(metrics.max_ulp, 30720);
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
fn fixture_numeric_fields_share_one_contract() {
    if !wpt_js_loader::node_available() {
        eprintln!("[SKIP] fixture field contract requires Node.js");
        return;
    }
    let bridge = wpt_js_loader::bridge_dir();
    let contract_path = bridge.join("numeric-fields.json");
    let contract: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&contract_path).unwrap()).unwrap();
    let floating_options = contract["floatOptions"].as_array().unwrap();
    let tensor_fields = contract["tensorData"].as_array().unwrap();
    let mut unique = std::collections::HashSet::new();
    for field in floating_options.iter().chain(tensor_fields) {
        let field = field.as_str().unwrap();
        assert!(unique.insert(field), "duplicate numeric field: {field}");
        assert!(!["axis", "axes", "label", "unknown"].contains(&field));
    }
    assert!(!floating_options.is_empty());
    assert_eq!(tensor_fields, &[serde_json::json!("data")]);
    let output = std::process::Command::new("node")
        .args([
            "--input-type=module",
            "-e",
            r#"
                import {readFileSync} from 'node:fs';
                import {pathToFileURL} from 'node:url';
                const {normalizeValue} = await import(pathToFileURL(process.argv[1]));
                const fields = JSON.parse(readFileSync(process.argv[2], 'utf8'));
                const options = Object.fromEntries(fields.floatOptions.map(key => [key, -0]));
                Object.assign(options, {axis: -0, axes: [-0], label: '-0', unknown: -0});
                process.stdout.write(JSON.stringify(normalizeValue({
                    data: [-0, 0, Infinity, -Infinity, NaN, 1n], options,
                })));
            "#,
        ])
        .arg(bridge.join("normalize-fixture.mjs"))
        .arg(&contract_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let normalized: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        normalized["data"],
        serde_json::json!(["-0", 0, "Infinity", "-Infinity", "NaN", "1"])
    );
    let operator = wpt_types::WptOperator {
        name: "linear".into(),
        arguments: serde_json::json!([{ "options": normalized["options"] }]),
        outputs: serde_json::json!("output"),
    };
    let call = wpt_execute_graph::build_method_args(
        &operator.name,
        &operator,
        &std::collections::HashMap::new(),
        &std::collections::HashMap::new(),
        &std::collections::HashSet::new(),
    )
    .unwrap();
    for key in floating_options {
        let key = key.as_str().unwrap();
        assert_eq!(normalized["options"][key], "-0", "JS field {key}");
        assert_eq!(
            call.options[key]
                .as_f64()
                .unwrap_or_else(|| panic!("Rust field {key} must be a floating number"))
                .to_bits(),
            (-0.0f64).to_bits(),
            "Rust field {key}"
        );
    }
    assert_eq!(call.options["label"], "-0");
    assert_eq!(call.options["axis"].as_u64(), Some(0));
    assert_eq!(call.options["axes"], serde_json::json!([0]));
    assert_eq!(call.options["unknown"].as_u64(), Some(0));
}

#[test]
fn javascript_fixture_bridge_preserves_signed_zero_in_tensors_and_options() {
    if !wpt_js_loader::node_available() {
        eprintln!("[SKIP] signed-zero fixture bridge requires Node.js");
        return;
    }
    let fixture_root = wpt_js_loader::repo_root().join("tests/fixtures/wpt_signed_zero");
    let fixture = fixture_root.join("webnn/conformance_tests/signed-zero.https.any.js");
    let utils = fixture_root.join("webnn/resources/utils.js");
    for (script, without_native_float16) in [
        ("dump_tests.mjs", false),
        ("dump_corpus.mjs", false),
        ("dump_tests.mjs", true),
        ("dump_corpus.mjs", true),
    ] {
        let mut command = std::process::Command::new("node");
        // Node's --import treats Windows drive letters as URL schemes. Resolve
        // the relative specifier from the fixture directory on every platform.
        command.current_dir(&fixture_root);
        if without_native_float16 {
            command.arg("--import").arg("./without-native-float16.mjs");
        }
        command.arg(wpt_js_loader::bridge_dir().join(script));
        if script == "dump_tests.mjs" {
            command.arg(&fixture).arg("--utils").arg(&utils);
        } else {
            command.arg("--wpt-dir").arg(&fixture_root);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{script}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let payload: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let cases = payload[if script == "dump_tests.mjs" {
            "tests"
        } else {
            "cases"
        }]
        .as_array()
        .unwrap();
        assert_eq!(cases.len(), 23);
        for case in &cases[..20] {
            let test: wpt_types::WptTestCase = serde_json::from_value(case.clone()).unwrap();
            let input = &test.graph.inputs["input"];
            let expected = &test.graph.expected_outputs["output"];
            let source = if test.name.ends_with("scalar") {
                vec![-0.0f64]
            } else if test.name.ends_with("vector") {
                vec![0.0, -0.0, 2.0f64.powi(-24), -2.0f64.powi(-24)]
            } else if test.name.ends_with("negative-fill") {
                vec![-0.0; 3]
            } else if test.name.ends_with("positive-fill") {
                vec![0.0; 3]
            } else {
                vec![0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]
            };
            let expected_f64 = wpt_tensor::expected_output_to_f64(expected).unwrap();
            assert_eq!(expected_f64.len(), source.len());
            for (actual, wanted) in expected_f64.iter().zip(&source) {
                if wanted.is_nan() {
                    assert!(actual.is_nan(), "{script}: {}", test.name);
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        wanted.to_bits(),
                        "{script}: {}",
                        test.name
                    );
                }
            }
            for strict in [false, true] {
                let bytes = wpt_tensor::tensor_spec_to_bytes_with_mode(input, strict).unwrap();
                if input.data_type() == "float16" {
                    let bits = wpt_tensor::tensor_f16_bits_with_mode(input, strict);
                    assert_eq!(bits.len(), source.len());
                    assert_eq!(
                        bytes,
                        bits.iter()
                            .flat_map(|bits| bits.to_ne_bytes())
                            .collect::<Vec<_>>()
                    );
                    for (actual, wanted) in bits.iter().zip(&source) {
                        let actual = half::f16::from_bits(*actual).to_f64();
                        if wanted.is_nan() {
                            assert!(actual.is_nan(), "{script}: {}", test.name);
                        } else {
                            assert_eq!(
                                actual.to_bits(),
                                wanted.to_bits(),
                                "{script}: {}",
                                test.name
                            );
                        }
                    }
                } else {
                    let values = wpt_tensor::tensor_f32_values(input);
                    assert_eq!(values.len(), source.len());
                    assert_eq!(
                        bytes,
                        values
                            .iter()
                            .flat_map(|value| value.to_ne_bytes())
                            .collect::<Vec<_>>()
                    );
                    for (actual, wanted) in values.iter().zip(&source) {
                        if wanted.is_nan() {
                            assert!(actual.is_nan(), "{script}: {}", test.name);
                        } else {
                            assert_eq!(
                                actual.to_bits(),
                                (*wanted as f32).to_bits(),
                                "{script}: {}",
                                test.name
                            );
                        }
                    }
                }
            }
            assert_eq!(tolerance::upstream_ulp_distance(-0.0, 0.0, true), 0);
            assert_eq!(tolerance::upstream_ulp_distance(-0.0, 0.0, false), 0);
        }
        let test: wpt_types::WptTestCase = serde_json::from_value(cases[20].clone()).unwrap();
        let mut operator = test.graph.operators[0].clone();
        operator.arguments.as_array_mut().unwrap().remove(0);
        let call = wpt_execute_graph::build_method_args(
            "hardSigmoid",
            &operator,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .unwrap();
        let options: rustnn::operator_options::MLHardSigmoidOptions =
            serde_json::from_value(serde_json::Value::Object(call.options)).unwrap();
        assert_eq!(
            options.alpha.to_bits(),
            (-0.0f64).to_bits(),
            "{script}: option alpha"
        );
        assert_eq!(
            options.beta.to_bits(),
            0.0f64.to_bits(),
            "{script}: option beta"
        );
        assert_eq!(options.label, "-0");
        for (index, key) in [(21, "axis"), (22, "axes")] {
            let test: wpt_types::WptTestCase =
                serde_json::from_value(cases[index].clone()).unwrap();
            let mut operator = test.graph.operators[0].clone();
            operator.arguments.as_array_mut().unwrap().remove(0);
            let call = wpt_execute_graph::build_method_args(
                &operator.name,
                &operator,
                &std::collections::HashMap::new(),
                &std::collections::HashMap::new(),
                &std::collections::HashSet::new(),
            )
            .unwrap();
            if key == "axis" {
                assert_eq!(
                    serde_json::from_value::<u32>(call.options[key].clone()).unwrap(),
                    0
                );
            } else {
                assert_eq!(
                    serde_json::from_value::<Vec<u32>>(call.options[key].clone()).unwrap(),
                    [0]
                );
            }
            assert_eq!(call.options["label"], "-0");
        }
    }
}

#[test]
fn strict_half_inputs_and_constants_use_source_rounding_without_changing_default() {
    for constant in [false, true] {
        let spec = wpt_types::WptTensorSpec {
            // Just above a float16 tie: rounding through f32 lands on the tie
            // (ties-to-even gives 1.0), while upstream rounds the source directly.
            data: serde_json::json!([
                1.0 + 2.0f64.powi(-11) + 2.0f64.powi(-40),
                -1.0 - 2.0f64.powi(-11) - 2.0f64.powi(-40)
            ]),
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
    // rounding; opposite signs exercise its signed-magnitude distance.
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

//! Tolerance and result validation for WPT conformance tests.
//!
//! Port of pywebnn/tests/wpt_utils.py: ULP/ATOL checks and operation defaults.

use std::collections::HashMap;

/// Tolerance specification: ULP (units in last place) or ATOL (absolute).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ToleranceSpec {
    pub kind: ToleranceKind,
    pub value: u64, // ULP count or f64 for ATOL stored as bits for simplicity; we use value as u64 for ULP, and for ATOL we use a separate field
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToleranceKind {
    Ulp,
    Atol,
    Rtol,
}

/// For ATOL we need a float value.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum ToleranceValue {
    Ulp(u64),
    Atol(f64),
}

#[allow(dead_code)]
impl ToleranceSpec {
    pub fn ulp(v: u64) -> Self {
        Self {
            kind: ToleranceKind::Ulp,
            value: v,
        }
    }
    pub fn atol(v: f64) -> Self {
        Self {
            kind: ToleranceKind::Atol,
            value: v.to_bits(), // store for parsing; we'll use a separate atol_f64 in validation
        }
    }
}

/// Default tolerances per operation (matches WPT/Python wpt_utils).
/// Value: for Ulp = ULP count (u64); for Atol/Rtol = f64 bits (u64).
fn default_tolerances() -> HashMap<String, (ToleranceKind, u64)> {
    let mut m = HashMap::new();
    // Exact (no rounding)
    m.insert("relu".to_string(), (ToleranceKind::Ulp, 0));
    m.insert("add".to_string(), (ToleranceKind::Ulp, 0));
    m.insert("sub".to_string(), (ToleranceKind::Ulp, 0));
    m.insert("mul".to_string(), (ToleranceKind::Ulp, 0));
    m.insert("reshape".to_string(), (ToleranceKind::Ulp, 0));
    m.insert("reduce_sum".to_string(), (ToleranceKind::Ulp, 0));
    m.insert("reduce_max".to_string(), (ToleranceKind::Ulp, 0));
    m.insert("reduce_min".to_string(), (ToleranceKind::Ulp, 0));
    // Approximate
    m.insert("sigmoid".to_string(), (ToleranceKind::Ulp, 34));
    m.insert("tanh".to_string(), (ToleranceKind::Ulp, 44));
    m.insert("softmax".to_string(), (ToleranceKind::Ulp, 100));
    m.insert("div".to_string(), (ToleranceKind::Ulp, 2));
    m.insert("reduce_mean".to_string(), (ToleranceKind::Ulp, 2));
    m.insert("reduce_product".to_string(), (ToleranceKind::Ulp, 10));
    m.insert("reduce_l1".to_string(), (ToleranceKind::Ulp, 2));
    m.insert("reduce_l2".to_string(), (ToleranceKind::Ulp, 5));
    m.insert("reduce_log_sum".to_string(), (ToleranceKind::Ulp, 10));
    m.insert("reduce_log_sum_exp".to_string(), (ToleranceKind::Ulp, 100));
    m.insert("reduce_sum_square".to_string(), (ToleranceKind::Ulp, 2));
    // Convolution / pooling (conv2d/conv_transpose2d: use relative tolerance for TensorRT)
    let rtol_1e3 = 1e-3_f64.to_bits();
    m.insert("conv2d".to_string(), (ToleranceKind::Rtol, rtol_1e3));
    m.insert(
        "conv_transpose2d".to_string(),
        (ToleranceKind::Rtol, rtol_1e3),
    );
    m.insert("average_pool2d".to_string(), (ToleranceKind::Ulp, 2));
    m.insert("max_pool2d".to_string(), (ToleranceKind::Ulp, 0));
    m.insert("global_average_pool".to_string(), (ToleranceKind::Ulp, 2));
    m.insert("global_max_pool".to_string(), (ToleranceKind::Ulp, 0));
    // Normalization
    m.insert("batch_normalization".to_string(), (ToleranceKind::Ulp, 100));
    m.insert(
        "instance_normalization".to_string(),
        (ToleranceKind::Ulp, 100),
    );
    m.insert("layer_normalization".to_string(), (ToleranceKind::Ulp, 100));
    m.insert("matmul".to_string(), (ToleranceKind::Ulp, 100));
    m
}

/// Get tolerance for an operation; test-case override takes precedence.
/// Returns (kind, value): for Ulp, value is the ULP count; for Atol/Rtol, value is f64 bits (use f64::from_bits).
pub fn get_operation_tolerance(
    operation: &str,
    tolerance_override: Option<&super::wpt_types::WptTolerance>,
) -> (ToleranceKind, u64) {
    if let Some(t) = tolerance_override {
        let kind = if t.r#type.eq_ignore_ascii_case("atol") {
            ToleranceKind::Atol
        } else if t.r#type.eq_ignore_ascii_case("rtol") {
            ToleranceKind::Rtol
        } else {
            ToleranceKind::Ulp
        };
        let value = match kind {
            ToleranceKind::Atol => t.value.as_f64().unwrap_or(1e-5).to_bits(),
            ToleranceKind::Rtol => t.value.as_f64().unwrap_or(1e-3).to_bits(),
            ToleranceKind::Ulp => t
                .value
                .as_u64()
                .or_else(|| t.value.as_f64().map(|f| f as u64))
                .unwrap_or(100),
        };
        return (kind, value);
    }
    default_tolerances()
        .get(operation)
        .copied()
        .unwrap_or((ToleranceKind::Ulp, 100u64))
}

/// ULP distance between two f32 values (matches Python/ WPT).
pub fn ulp_distance_f32(a: f32, b: f32) -> u32 {
    if a.is_nan() || b.is_nan() {
        if a.is_nan() && b.is_nan() {
            return 0;
        }
        return u32::MAX;
    }
    if a.is_infinite() || b.is_infinite() {
        if a == b {
            return 0;
        }
        return u32::MAX;
    }
    let a_bits = a.to_bits();
    let b_bits = b.to_bits();
    if (a_bits ^ b_bits) & 0x8000_0000 != 0 {
        let a_dist = a_bits & 0x7FFF_FFFF;
        let b_dist = b_bits & 0x7FFF_FFFF;
        return (a_dist + b_dist) as u32;
    }
    (a_bits as i64 - b_bits as i64).unsigned_abs() as u32
}

/// Check ULP tolerance; returns (pass, optional first failure message).
pub fn check_ulp_tolerance(
    actual: &[f32],
    expected: &[f32],
    tolerance: u64,
) -> (bool, Option<String>) {
    if actual.len() != expected.len() {
        return (
            false,
            Some(format!(
                "shape mismatch: actual len {} expected len {}",
                actual.len(),
                expected.len()
            )),
        );
    }
    let tol = tolerance as u32;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let ulp = ulp_distance_f32(a, e);
        if ulp > tol {
            return (
                false,
                Some(format!(
                    "index {}: actual={} expected={} ulp={} tolerance={}",
                    i, a, e, ulp, tolerance
                )),
            );
        }
    }
    (true, None)
}

/// Check relative tolerance: |actual - expected| / max(|expected|, 1e-6) <= rtol.
pub fn check_rtol_tolerance(actual: &[f32], expected: &[f32], rtol: f64) -> (bool, Option<String>) {
    if actual.len() != expected.len() {
        return (
            false,
            Some(format!(
                "shape mismatch: actual len {} expected len {}",
                actual.len(),
                expected.len()
            )),
        );
    }
    let rtol_f = rtol as f32;
    let eps = 1e-6_f32;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() {
            if !a.is_nan() {
                return (
                    false,
                    Some(format!("index {}: actual={} expected=NaN", i, a)),
                );
            }
            continue;
        }
        if e.is_infinite() {
            if a != e {
                return (
                    false,
                    Some(format!("index {}: actual={} expected={}", i, a, e)),
                );
            }
            continue;
        }
        let denom = e.abs().max(eps);
        let rel = (a - e).abs() / denom;
        if rel > rtol_f {
            return (
                false,
                Some(format!(
                    "index {}: actual={} expected={} rel={} tolerance={}",
                    i, a, e, rel, rtol_f
                )),
            );
        }
    }
    (true, None)
}

/// Check absolute tolerance.
pub fn check_atol_tolerance(
    actual: &[f32],
    expected: &[f32],
    tolerance: f64,
) -> (bool, Option<String>) {
    if actual.len() != expected.len() {
        return (
            false,
            Some(format!(
                "shape mismatch: actual len {} expected len {}",
                actual.len(),
                expected.len()
            )),
        );
    }
    let tol = tolerance as f32;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() {
            if !a.is_nan() {
                return (
                    false,
                    Some(format!("index {}: actual={} expected=NaN", i, a)),
                );
            }
            continue;
        }
        if e.is_infinite() {
            if a != e {
                return (
                    false,
                    Some(format!("index {}: actual={} expected={}", i, a, e)),
                );
            }
            continue;
        }
        let diff = (a - e).abs();
        if diff > tol {
            return (
                false,
                Some(format!(
                    "index {}: actual={} expected={} diff={} tolerance={}",
                    i, a, e, diff, tol
                )),
            );
        }
    }
    (true, None)
}

/// Validate actual vs expected; returns (pass, error message if failed).
pub fn validate_result(
    actual: &[f32],
    expected: &[f32],
    kind: ToleranceKind,
    value: u64,
) -> (bool, Option<String>) {
    match kind {
        ToleranceKind::Ulp => check_ulp_tolerance(actual, expected, value),
        ToleranceKind::Atol => {
            let atol_f = f64::from_bits(value);
            check_atol_tolerance(actual, expected, atol_f)
        }
        ToleranceKind::Rtol => {
            let rtol_f = f64::from_bits(value);
            check_rtol_tolerance(actual, expected, rtol_f)
        }
    }
}

//! Mathematical binary64-to-binary16 reference independent of conversion code.
//!
//! Find adjacent exactly represented finite Half values, compare against their
//! exact dyadic midpoint, and choose the even encoding on a tie. No production
//! narrowing, dependency intrinsic, or intermediate binary32 value is used.

fn positive_finite_half_value(bits: u16) -> f64 {
    assert!(bits <= 0x7bff);
    let exponent = bits / 1024;
    let fraction = bits % 1024;
    if exponent == 0 {
        f64::from(fraction) * 2f64.powi(-24)
    } else {
        f64::from(1024 + fraction) * 2f64.powi(i32::from(exponent) - 25)
    }
}

pub(crate) fn reference_half_bits(value: f64) -> u16 {
    let sign = if value.is_sign_negative() { 0x8000 } else { 0 };
    if value.is_nan() {
        return sign | 0x7e00;
    }
    let magnitude = value.abs();
    // Overflow's midpoint lies between 65504 and the next same-spacing value
    // 65536, whose significand is even.
    if magnitude >= 65520. {
        return sign | 0x7c00;
    }
    if magnitude >= 65504. {
        return sign | 0x7bff;
    }
    if magnitude == 0. {
        return sign;
    }
    let (mut lower, mut upper) = (0u16, 0x7bffu16);
    while upper - lower > 1 {
        let candidate = lower + (upper - lower) / 2;
        if positive_finite_half_value(candidate) > magnitude {
            upper = candidate;
        } else {
            lower = candidate;
        }
    }
    // Each finite Half value and its midpoint is exact in binary64.
    let midpoint = (positive_finite_half_value(lower) + positive_finite_half_value(upper)) / 2.;
    let rounded = if magnitude < midpoint {
        lower
    } else if magnitude > midpoint {
        upper
    } else if lower.is_multiple_of(2) {
        lower
    } else {
        upper
    };
    sign | rounded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_finite_half_encoding_round_trips_with_its_sign() {
        for bits in 0..=0x7bff {
            let value = positive_finite_half_value(bits);
            for (value, expected) in [(value, bits), (-value, bits | 0x8000)] {
                assert_eq!(reference_half_bits(value), expected);
            }
        }
    }

    #[test]
    fn every_midpoint_and_immediate_source_neighbor_uses_even_endpoints() {
        let mut count = 0;
        for lower in 0u16..0x7bff {
            let midpoint =
                (positive_finite_half_value(lower) + positive_finite_half_value(lower + 1)) / 2.;
            for (offset, expected) in [(-1i64, lower), (0, lower + lower % 2), (1, lower + 1)] {
                let double = f64::from_bits((midpoint.to_bits() as i64 + offset) as u64);
                let single =
                    f32::from_bits((i64::from((midpoint as f32).to_bits()) + offset) as u32);
                for value in [double, f64::from(single)] {
                    for (value, expected) in [(value, expected), (-value, expected | 0x8000)] {
                        assert_eq!(reference_half_bits(value), expected, "value={value:?}");
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, 380_916);
    }

    #[test]
    fn literal_extremes_overflow_underflow_zero_and_nonfinite_classes() {
        for (value, expected) in [
            (0., 0),
            (f64::from_bits(1), 0),
            (f64::MIN_POSITIVE, 0),
            (f64::from_bits(2f64.powi(-25).to_bits() - 1), 0),
            (2f64.powi(-25), 0),
            (f64::from_bits(2f64.powi(-25).to_bits() + 1), 1),
            (2f64.powi(-24), 1),
            (f64::from_bits(65520f64.to_bits() - 1), 0x7bff),
            (65520., 0x7c00),
            (f64::from_bits(65520f64.to_bits() + 1), 0x7c00),
            (f64::MAX, 0x7c00),
            (f64::INFINITY, 0x7c00),
        ] {
            for (value, expected) in [(value, expected), (-value, expected | 0x8000)] {
                assert_eq!(reference_half_bits(value), expected);
            }
        }
        for payload in [1, 1 << 31, 1 << 51, (1 << 52) - 1] {
            for sign in [0u64, 1 << 63] {
                let value = f64::from_bits(sign | 0x7ff0_0000_0000_0000 | payload);
                let expected = 0x7e00 | if sign == 0 { 0 } else { 0x8000 };
                assert_eq!(reference_half_bits(value), expected);
            }
        }
    }
}

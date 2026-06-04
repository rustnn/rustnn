//! Float16 arithmetic helpers matching WPT reference semantics.

use crate::graph::DataType;

#[inline]
pub fn round_f16(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

#[inline]
pub fn round_f16_slice(data: &mut [f32]) {
    for v in data {
        *v = round_f16(*v);
    }
}

#[inline]
pub fn f16_add(a: f32, b: f32) -> f32 {
    (half::f16::from_f32(a) + half::f16::from_f32(b)).to_f32()
}

#[inline]
pub fn f16_sub(a: f32, b: f32) -> f32 {
    (half::f16::from_f32(a) - half::f16::from_f32(b)).to_f32()
}

#[inline]
pub fn f16_mul(a: f32, b: f32) -> f32 {
    (half::f16::from_f32(a) * half::f16::from_f32(b)).to_f32()
}

#[inline]
pub fn f16_div(a: f32, b: f32) -> f32 {
    (half::f16::from_f32(a) / half::f16::from_f32(b)).to_f32()
}

#[inline]
pub fn f16_neg(a: f32) -> f32 {
    (-half::f16::from_f32(a)).to_f32()
}

#[inline]
pub fn f16_sqrt(a: f32) -> f32 {
    round_f16(half::f16::from_f32(a).to_f32().sqrt())
}

#[inline]
pub fn is_integer_element_type(dt: DataType) -> bool {
    matches!(
        dt,
        DataType::Int32 | DataType::Uint32 | DataType::Int8 | DataType::Uint8
    )
}

pub fn use_integer_arithmetic(
    dtypes: &std::collections::HashMap<u32, DataType>,
    inputs: &[u32],
    output: u32,
    operand_types: &std::collections::HashMap<u32, DataType>,
) -> bool {
    let is_int = is_integer_element_type;
    let out_dt = operand_types
        .get(&output)
        .copied()
        .or_else(|| dtypes.get(&output).copied())
        .unwrap_or(DataType::Float32);
    if is_int(out_dt) {
        return true;
    }
    inputs.iter().any(|id| {
        dtypes
            .get(id)
            .copied()
            .or_else(|| operand_types.get(id).copied())
            .map(is_int)
            .unwrap_or(false)
    })
}

pub fn use_f16_arithmetic(
    dtypes: &std::collections::HashMap<u32, DataType>,
    inputs: &[u32],
    output: u32,
    operand_types: &std::collections::HashMap<u32, DataType>,
) -> bool {
    let out_dt = operand_types
        .get(&output)
        .copied()
        .or_else(|| dtypes.get(&output).copied())
        .unwrap_or(DataType::Float32);
    if out_dt == DataType::Float16 {
        return true;
    }
    inputs.iter().any(|id| {
        dtypes
            .get(id)
            .copied()
            .or_else(|| operand_types.get(id).copied())
            .unwrap_or(DataType::Float32)
            == DataType::Float16
    })
}

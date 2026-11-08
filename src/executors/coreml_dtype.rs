//! Checked numeric conversions at the MLMultiArray boundary.

use crate::error::GraphError;
use crate::graph::{DataType, pack_int4, pack_uint4_from_i32};

pub(super) fn boundary_error(reason: impl Into<String>) -> GraphError {
    GraphError::CoremlRuntimeFailed {
        reason: reason.into(),
    }
}

/// Canonical values from CoreML's MLMultiArrayDataType enum. There is no Int64
/// variant, and Double is floating point, not an eight-byte integer proxy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeType {
    Float16,
    Float32,
    Double,
    Int32,
    Int8,
}

impl NativeType {
    pub(super) fn from_code(code: i64) -> Result<Self, GraphError> {
        match code {
            65552 => Ok(Self::Float16),
            65568 => Ok(Self::Float32),
            65600 => Ok(Self::Double),
            131104 => Ok(Self::Int32),
            // Reading a returned Int8 array is supported. We do not choose this
            // OS 26+ allocation type for older-platform WebNN byte boundaries.
            131080 => Ok(Self::Int8),
            _ => Err(boundary_error(format!(
                "unsupported MLMultiArray data type code {code}"
            ))),
        }
    }

    pub(super) fn code(self) -> i32 {
        match self {
            Self::Float16 => 65552,
            Self::Float32 => 65568,
            Self::Double => 65600,
            Self::Int32 => 131104,
            Self::Int8 => 131080,
        }
    }

    pub(super) fn element_size(self) -> usize {
        match self {
            Self::Float16 => 2,
            Self::Float32 | Self::Int32 => 4,
            Self::Double => 8,
            Self::Int8 => 1,
        }
    }

    pub(super) fn matches(self, data_type: DataType) -> bool {
        self.storage() == StorageType::Webnn(data_type)
    }

    fn storage(self) -> StorageType {
        match self {
            Self::Float16 => StorageType::Webnn(DataType::Float16),
            Self::Float32 => StorageType::Webnn(DataType::Float32),
            Self::Double => StorageType::Double,
            Self::Int32 => StorageType::Webnn(DataType::Int32),
            Self::Int8 => StorageType::Webnn(DataType::Int8),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StorageType {
    Webnn(DataType),
    Double,
}

fn byte_length(dtype: StorageType, count: usize) -> Result<usize, GraphError> {
    let length = match dtype {
        StorageType::Webnn(DataType::Bfloat16 | DataType::Float8e4m3) => {
            return Err(boundary_error(
                "CoreML boundary does not support BF16 or FP8 storage",
            ));
        }
        StorageType::Webnn(DataType::Int4 | DataType::Uint4) => Some(count.div_ceil(2)),
        StorageType::Webnn(dtype) => count.checked_mul(dtype.bytes_per_element()),
        StorageType::Double => count.checked_mul(8),
    };
    length
        .filter(|&length| length <= isize::MAX as usize)
        .ok_or_else(|| boundary_error("CoreML boundary byte length overflows addressable storage"))
}

pub(super) fn to_native(
    bytes: &[u8],
    source: DataType,
    target: NativeType,
    count: usize,
) -> Result<Vec<u8>, GraphError> {
    convert(bytes, StorageType::Webnn(source), target.storage(), count)
}

pub(super) fn from_native(
    bytes: &[u8],
    source: NativeType,
    target: DataType,
    count: usize,
) -> Result<Vec<u8>, GraphError> {
    // The converter represents unsigned 64-bit graph outputs with an Int32
    // proxy. Preserve the existing zero-extension of that proxy's 32 bits;
    // sign-extending an i32 would turn 0xffff_ffff into u64::MAX instead.
    if source == NativeType::Int32 && target == DataType::Uint64 {
        return convert(
            bytes,
            StorageType::Webnn(DataType::Uint32),
            StorageType::Webnn(target),
            count,
        );
    }
    convert(bytes, source.storage(), StorageType::Webnn(target), count)
}

/// Round directly from binary64 to binary16, retaining every discarded bit.
/// The dependency's software conversion truncates the low 32 significand bits
/// before testing a tie. This also affects binary32 values widened to binary64
/// on devices without the hardware half-conversion instruction.
fn half_bits(value: f64) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    if exponent == 0x7ff {
        return sign | if fraction == 0 { 0x7c00 } else { 0x7e00 };
    }
    // Binary64 subnormals are all below half's smallest rounding midpoint.
    if exponent == 0 {
        return sign;
    }
    let exponent = exponent - 1023;
    if exponent > 15 {
        return sign | 0x7c00;
    }
    let significand = fraction | (1u64 << 52);
    if exponent >= -14 {
        let rounded = round_shift_even(significand, 42);
        // A significand carry advances the exponent, including finite overflow.
        return sign | ((((exponent + 15) as u16) << 10) + (rounded as u16 - 1024));
    }
    if exponent < -25 {
        return sign;
    }
    sign | round_shift_even(significand, (28 - exponent) as u32) as u16
}

fn round_shift_even(value: u64, shift: u32) -> u64 {
    let retained = value >> shift;
    let discarded = value & ((1u64 << shift) - 1);
    let midpoint = 1u64 << (shift - 1);
    retained + u64::from(discarded > midpoint || (discarded == midpoint && retained & 1 != 0))
}

/// Host buffers need not be aligned. Decode byte arrays rather than making
/// typed references into a Vec<u8>. Preserve bits only when the types match.
fn convert(
    bytes: &[u8],
    source: StorageType,
    target: StorageType,
    count: usize,
) -> Result<Vec<u8>, GraphError> {
    let expected = byte_length(source, count)?;
    if bytes.len() != expected {
        return Err(boundary_error(format!(
            "CoreML boundary byte length mismatch: expected {expected}, got {}",
            bytes.len()
        )));
    }
    if source == target {
        return Ok(bytes.to_vec());
    }
    let mut output = Vec::with_capacity(byte_length(target, count)?);
    let mut packed_values = Vec::new();
    for index in 0..count {
        macro_rules! write_value {
            ($value:expr) => {{
                let value = $value;
                match target {
                    StorageType::Webnn(DataType::Bfloat16 | DataType::Float8e4m3) => {
                        return Err(boundary_error(
                            "CoreML boundary does not support BF16 or FP8 storage",
                        ));
                    }
                    StorageType::Double => output.extend_from_slice(&(value as f64).to_ne_bytes()),
                    StorageType::Webnn(DataType::Float32) => {
                        output.extend_from_slice(&(value as f32).to_ne_bytes())
                    }
                    StorageType::Webnn(DataType::Float16) => {
                        output.extend_from_slice(&half_bits(value as f64).to_ne_bytes())
                    }
                    StorageType::Webnn(DataType::Int32) => {
                        output.extend_from_slice(&(value as i32).to_ne_bytes())
                    }
                    StorageType::Webnn(DataType::Uint32) => {
                        output.extend_from_slice(&(value as u32).to_ne_bytes())
                    }
                    StorageType::Webnn(DataType::Int64) => {
                        output.extend_from_slice(&(value as i64).to_ne_bytes())
                    }
                    StorageType::Webnn(DataType::Uint64) => {
                        output.extend_from_slice(&(value as u64).to_ne_bytes())
                    }
                    StorageType::Webnn(DataType::Int8) => output.push((value as i8) as u8),
                    StorageType::Webnn(DataType::Uint8) => output.push(value as u8),
                    StorageType::Webnn(DataType::Int4 | DataType::Uint4) => {
                        // Float-to-integer conversion saturates before the shared
                        // packers clamp to four bits. Narrowing integers directly
                        // would wrap (e.g. 256 -> 0) before that clamp.
                        packed_values.push(value as f64 as i32);
                    }
                }
            }};
        }
        macro_rules! read_value {
            ($type:ty, $width:expr) => {
                <$type>::from_ne_bytes(
                    bytes[index * $width..(index + 1) * $width]
                        .try_into()
                        .unwrap(),
                )
            };
        }
        match source {
            StorageType::Webnn(DataType::Bfloat16 | DataType::Float8e4m3) => {
                return Err(boundary_error(
                    "CoreML boundary does not support BF16 or FP8 storage",
                ));
            }
            StorageType::Double => write_value!(read_value!(f64, 8)),
            StorageType::Webnn(DataType::Float32) => write_value!(read_value!(f32, 4)),
            StorageType::Webnn(DataType::Float16) => {
                write_value!(half::f16::from_bits(read_value!(u16, 2)).to_f32())
            }
            StorageType::Webnn(DataType::Int32) => write_value!(read_value!(i32, 4)),
            StorageType::Webnn(DataType::Uint32) => write_value!(read_value!(u32, 4)),
            StorageType::Webnn(DataType::Int64) => write_value!(read_value!(i64, 8)),
            StorageType::Webnn(DataType::Uint64) => write_value!(read_value!(u64, 8)),
            StorageType::Webnn(DataType::Int8) => write_value!(bytes[index] as i8),
            StorageType::Webnn(DataType::Uint8) => write_value!(bytes[index]),
            StorageType::Webnn(DataType::Int4 | DataType::Uint4) => {
                let nibble = (bytes[index / 2] >> ((index % 2) * 4)) & 0xf;
                let value = if source == StorageType::Webnn(DataType::Int4) {
                    ((nibble << 4) as i8) >> 4
                } else {
                    nibble as i8
                };
                write_value!(value);
            }
        }
    }
    Ok(match target {
        StorageType::Webnn(DataType::Int4) => pack_int4(&packed_values),
        StorageType::Webnn(DataType::Uint4) => pack_uint4_from_i32(&packed_values),
        _ => output,
    })
}

/// Validated positive element strides. The native array owns the storage; this
/// checks its metadata before any pointer arithmetic at the FFI boundary.
pub(super) struct ArrayLayout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    pub(super) count: usize,
    pub(super) byte_length: usize,
    pub(super) storage_byte_length: usize,
    pub(super) contiguous: bool,
    element_size: usize,
}

impl ArrayLayout {
    pub(super) fn new(
        shape: &[i64],
        strides: &[i64],
        count: usize,
        element_size: usize,
    ) -> Result<Self, GraphError> {
        if shape.len() != strides.len() || element_size == 0 {
            return Err(boundary_error(
                "invalid MLMultiArray shape/stride rank or element size",
            ));
        }
        let dimensions = shape
            .iter()
            .map(|&dimension| usize::try_from(dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| boundary_error("negative MLMultiArray dimension"))?;
        let strides = strides
            .iter()
            .map(|&stride| usize::try_from(stride))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| boundary_error("negative MLMultiArray stride"))?;
        let expected = dimensions
            .iter()
            .try_fold(1usize, |product, &dimension| product.checked_mul(dimension));
        if expected != Some(count) {
            return Err(boundary_error(
                "MLMultiArray shape/count mismatch or overflow",
            ));
        }
        let byte_length = count
            .checked_mul(element_size)
            .filter(|&length| length <= isize::MAX as usize)
            .ok_or_else(|| boundary_error("MLMultiArray byte length overflow"))?;
        let storage_byte_length = if count > 0 {
            let span = dimensions
                .iter()
                .zip(&strides)
                .try_fold(0usize, |offset, (&dimension, &stride)| {
                    dimension
                        .saturating_sub(1)
                        .checked_mul(stride)
                        .and_then(|part| offset.checked_add(part))
                })
                .and_then(|last| last.checked_add(1))
                .and_then(|elements| elements.checked_mul(element_size));
            span.filter(|&span| span <= isize::MAX as usize)
                .ok_or_else(|| boundary_error("MLMultiArray strided storage span overflow"))?
        } else {
            0
        };
        let mut contiguous = true;
        let mut expected_stride = 1usize;
        for (&dimension, &stride) in dimensions.iter().zip(&strides).rev() {
            if dimension > 1 && stride != expected_stride {
                contiguous = false;
            }
            expected_stride = expected_stride.checked_mul(dimension).unwrap_or(0);
        }
        Ok(Self {
            shape: dimensions,
            strides,
            count,
            byte_length,
            storage_byte_length,
            contiguous,
            element_size,
        })
    }

    pub(super) fn byte_offset(&self, mut index: usize) -> usize {
        debug_assert!(index < self.count);
        let mut offset = 0;
        for (&dimension, &stride) in self.shape.iter().zip(&self.strides).rev() {
            offset += (index % dimension) * stride;
            index /= dimension;
        }
        offset * self.element_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    #[test]
    fn coreml_dtypes_reject_bf16_and_fp8_storage() {
        for dtype in [DataType::Bfloat16, DataType::Float8e4m3] {
            for count in [0, 1] {
                let bytes = vec![0; dtype.storage_byte_length(count).unwrap()];
                assert!(to_native(&bytes, dtype, NativeType::Float32, count).is_err());
                assert!(
                    from_native(&vec![0; count * 4], NativeType::Float32, dtype, count).is_err()
                );
            }
        }
    }

    #[test]
    fn coreml_dtypes_canonical_codes_reject_legacy_and_unknown_codes() {
        for (code, kind, size) in [
            (65552, NativeType::Float16, 2),
            (65568, NativeType::Float32, 4),
            (65600, NativeType::Double, 8),
            (131104, NativeType::Int32, 4),
            (131080, NativeType::Int8, 1),
        ] {
            assert_eq!(NativeType::from_code(code).unwrap(), kind);
            assert_eq!(kind.code() as i64, code);
            assert_eq!(kind.element_size(), size);
        }
        for code in [0, 1, 3, 4, 16, 32, -1, i64::MAX] {
            assert!(NativeType::from_code(code).is_err());
        }
    }

    #[test]
    fn coreml_dtypes_same_width_mismatch_converts_in_both_directions() {
        let integers: Vec<u8> = [-2i32, 1, 123, 16_777_217]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let expected = f32_bytes(&[-2., 1., 123., 16_777_216.]);
        assert_eq!(
            to_native(&integers, DataType::Int32, NativeType::Float32, 4).unwrap(),
            expected
        );
        assert_eq!(
            from_native(&integers, NativeType::Int32, DataType::Float32, 4).unwrap(),
            expected
        );
        assert_eq!(
            to_native(&integers, DataType::Int32, NativeType::Int32, 4).unwrap(),
            integers
        );
        let floats = f32_bytes(&[-2., 1., 123.]);
        let expected: Vec<u8> = [-2i32, 1, 123]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        assert_eq!(
            to_native(&floats, DataType::Float32, NativeType::Int32, 3).unwrap(),
            expected
        );
        assert_eq!(
            from_native(&floats, NativeType::Float32, DataType::Int32, 3).unwrap(),
            expected
        );
        let unsigned = from_native(
            &f32_bytes(&[3_000_000_000.]),
            NativeType::Float32,
            DataType::Uint32,
            1,
        )
        .unwrap();
        assert_eq!(unsigned, 3_000_000_000u32.to_ne_bytes());
        let proxy = [-1i32, i32::MIN]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        let unsigned = from_native(&proxy, NativeType::Int32, DataType::Uint64, 2).unwrap();
        let expected = [u32::MAX as u64, 1u64 << 31]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        assert_eq!(unsigned, expected);
        let signed = from_native(&proxy, NativeType::Int32, DataType::Int64, 2).unwrap();
        let expected = [-1i64, i32::MIN as i64]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        assert_eq!(signed, expected);
    }

    #[test]
    fn coreml_dtypes_half_double_unaligned_and_same_type_bits() {
        let values = [-2., 0., 0.5, 65504.];
        let float_bytes = f32_bytes(&values);
        let mut unaligned = vec![0];
        unaligned.extend_from_slice(&float_bytes);
        let half_bytes =
            to_native(&unaligned[1..], DataType::Float32, NativeType::Float16, 4).unwrap();
        assert_eq!(half_bytes.len(), 8);
        assert_eq!(
            from_native(&half_bytes, NativeType::Float16, DataType::Float32, 4).unwrap(),
            float_bytes
        );
        let doubles: Vec<u8> = [-2.5f64, 0.125, 256.]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        assert_eq!(
            from_native(&doubles, NativeType::Double, DataType::Float32, 3).unwrap(),
            f32_bytes(&[-2.5, 0.125, 256.])
        );
        let bits = [0x8000_0000u32, 0x7fc0_1234]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            to_native(&bits, DataType::Float32, NativeType::Float32, 2).unwrap(),
            bits
        );
    }

    #[test]
    fn coreml_dtypes_half_rounds_every_midpoint_and_neighbor_in_both_directions() {
        // Build expectations from adjacent binary16 encodings, not from another
        // float-to-half converter. Each midpoint is exactly representable in
        // binary32 and binary64; its immediate neighbors must round oppositely.
        let half_value = |bits: u16| {
            let exponent = bits >> 10;
            let fraction = bits & 1023;
            if exponent == 0 {
                f64::from(fraction) * 2f64.powi(-24)
            } else {
                f64::from(1024 + fraction) * 2f64.powi(i32::from(exponent) - 25)
            }
        };
        let mut singles = Vec::new();
        let mut doubles = Vec::new();
        let mut expected = Vec::new();
        for lower in 0u16..0x7bff {
            let midpoint = (half_value(lower) + half_value(lower + 1)) / 2.;
            let tie = lower + (lower & 1);
            for (offset, want) in [(-1i64, lower), (0, tie), (1, lower + 1)] {
                let single = f32::from_bits(((midpoint as f32).to_bits() as i64 + offset) as u32);
                let double = f64::from_bits((midpoint.to_bits() as i64 + offset) as u64);
                for sign in [false, true] {
                    singles.extend_from_slice(&(if sign { -single } else { single }).to_ne_bytes());
                    doubles.extend_from_slice(&(if sign { -double } else { double }).to_ne_bytes());
                    expected
                        .extend_from_slice(&(want | if sign { 0x8000 } else { 0 }).to_ne_bytes());
                }
            }
        }
        let count = expected.len() / 2;
        assert_eq!(
            to_native(&singles, DataType::Float32, NativeType::Float16, count).unwrap(),
            expected,
            "float32 input boundary"
        );
        assert_eq!(
            from_native(&doubles, NativeType::Double, DataType::Float16, count).unwrap(),
            expected,
            "double output boundary"
        );
    }

    #[test]
    fn coreml_dtypes_half_keeps_sticky_bits_and_handles_special_values() {
        let above_tie = 1. + 2f64.powi(-11) + 2f64.powi(-25);
        for (value, want) in [
            (above_tie, 0x3c01u16),
            (-above_tie, 0xbc01),
            (0., 0),
            (-0., 0x8000),
            (f64::MIN_POSITIVE, 0),
            (-f64::MIN_POSITIVE, 0x8000),
            (65520., 0x7c00),
            (f64::from_bits(65520f64.to_bits() - 1), 0x7bff),
            (-65520., 0xfc00),
            (f64::INFINITY, 0x7c00),
            (f64::NEG_INFINITY, 0xfc00),
        ] {
            assert_eq!(
                from_native(
                    &value.to_ne_bytes(),
                    NativeType::Double,
                    DataType::Float16,
                    1
                )
                .unwrap(),
                want.to_ne_bytes(),
                "value={value:?}"
            );
        }
        for value in [f64::NAN, -f64::NAN] {
            let bytes = from_native(
                &value.to_ne_bytes(),
                NativeType::Double,
                DataType::Float16,
                1,
            )
            .unwrap();
            assert!(half::f16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).is_nan());
        }
        let all_encodings: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_ne_bytes).collect();
        assert_eq!(
            to_native(
                &all_encodings,
                DataType::Float16,
                NativeType::Float16,
                65536
            )
            .unwrap(),
            all_encodings,
            "matching storage must preserve every bit, including NaN payloads"
        );
    }

    #[test]
    fn coreml_dtypes_promoted_and_packed_boundaries() {
        for (dtype, bytes, expected) in [
            (DataType::Int8, vec![128, 255, 127], vec![-128., -1., 127.]),
            (DataType::Uint8, vec![0, 128, 255], vec![0., 128., 255.]),
            (DataType::Int4, vec![0x78, 0x0f], vec![-8., 7., -1.]),
            (DataType::Uint4, vec![0xf0, 0x08], vec![0., 15., 8.]),
        ] {
            let native = to_native(&bytes, dtype, NativeType::Float32, 3).unwrap();
            assert_eq!(native, f32_bytes(&expected));
            assert_eq!(
                from_native(&native, NativeType::Float32, dtype, 3).unwrap(),
                bytes
            );
            let half = to_native(&bytes, dtype, NativeType::Float16, 3).unwrap();
            assert_eq!(
                from_native(&half, NativeType::Float16, dtype, 3).unwrap(),
                bytes
            );
        }
        for dtype in [DataType::Int64, DataType::Uint64] {
            let bytes = [0u64, 123, 65536]
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>();
            let native = to_native(&bytes, dtype, NativeType::Float32, 3).unwrap();
            assert_eq!(native, f32_bytes(&[0., 123., 65536.]));
            assert_eq!(
                from_native(&native, NativeType::Float32, dtype, 3).unwrap(),
                bytes
            );
        }
    }

    #[test]
    fn coreml_dtypes_packed_outputs_saturate_before_narrowing() {
        let values = [i64::MIN, -256, -9, -1, 7, 16, i64::MAX];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        for (dtype, expected) in [
            (DataType::Int4, vec![0x88, 0xf8, 0x77, 0x07]),
            (DataType::Uint4, vec![0x00, 0x00, 0xf7, 0x0f]),
        ] {
            assert_eq!(
                convert(
                    &bytes,
                    StorageType::Webnn(DataType::Int64),
                    StorageType::Webnn(dtype),
                    values.len()
                )
                .unwrap(),
                expected
            );
        }
        let values = [0u64, 15, 16, 255, 256, u64::MAX];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        assert_eq!(
            convert(
                &bytes,
                StorageType::Webnn(DataType::Uint64),
                StorageType::Webnn(DataType::Uint4),
                values.len()
            )
            .unwrap(),
            [0xf0, 0xff, 0xff]
        );
        let values = [
            f64::NEG_INFINITY,
            -8.9,
            -0.9,
            f64::NAN,
            7.9,
            256.0,
            f64::INFINITY,
        ];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        assert_eq!(
            from_native(&bytes, NativeType::Double, DataType::Int4, values.len()).unwrap(),
            [0x88, 0x00, 0x77, 0x07]
        );
        assert_eq!(
            from_native(&bytes, NativeType::Double, DataType::Uint4, values.len()).unwrap(),
            [0x00, 0x00, 0xf7, 0x0f]
        );
        let bytes: Vec<u8> = [-256i32, 256, i32::MAX]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        assert_eq!(
            from_native(&bytes, NativeType::Int32, DataType::Uint4, 3).unwrap(),
            [0xf0, 0x0f]
        );
    }

    #[test]
    fn coreml_dtypes_reject_lengths_and_layout_overflows() {
        assert!(to_native(&[0; 3], DataType::Float32, NativeType::Float16, 1).is_err());
        assert!(to_native(&[], DataType::Int4, NativeType::Float32, 1).is_err());
        assert!(to_native(&[], DataType::Float32, NativeType::Float32, usize::MAX).is_err());
        assert!(ArrayLayout::new(&[2, 3], &[3], 6, 4).is_err());
        assert!(ArrayLayout::new(&[2, -3], &[3, 1], 6, 4).is_err());
        assert!(ArrayLayout::new(&[2, 3], &[3, -1], 6, 4).is_err());
        assert!(ArrayLayout::new(&[2, 3], &[3, 1], 5, 4).is_err());
        assert!(ArrayLayout::new(&[2], &[i64::MAX], 2, 8).is_err());
        assert!(ArrayLayout::new(&[i64::MAX, 3], &[3, 1], 0, 4).is_err());
        let layout = ArrayLayout::new(&[2, 3], &[16, 2], 6, 2).unwrap();
        assert!(!layout.contiguous);
        assert_eq!(layout.byte_length, 12);
        assert_eq!(layout.storage_byte_length, 42);
        assert_eq!(
            (0..6)
                .map(|index| layout.byte_offset(index))
                .collect::<Vec<_>>(),
            [0, 4, 8, 32, 36, 40]
        );
        assert_eq!(
            ArrayLayout::new(&[0, 3], &[3, 1], 0, 4)
                .unwrap()
                .storage_byte_length,
            0
        );
        let scalar = ArrayLayout::new(&[], &[], 1, 4).unwrap();
        assert_eq!(scalar.byte_offset(0), 0);
        assert_eq!(scalar.storage_byte_length, 4);
    }
}

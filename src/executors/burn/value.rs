/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Validated, authoritative values with private storage and integer kinds.
use super::{
    host_array::HostArray,
    interpreter::{BurnInput, BurnOutputWithData},
    tensor_env::RuntimeTensor,
};
use crate::{error::GraphError, graph::DataType};
use burn_tensor::backend::Backend;

#[derive(Debug)]
pub(super) struct TensorValue<B: Backend> {
    storage: ValueStorage<B>,
}

#[derive(Debug)]
enum ValueStorage<B: Backend> {
    Float32(RuntimeTensor<B>),
    Float16(RuntimeTensor<B>),
    Signed(SignedValue),
    Unsigned(UnsignedValue),
}

#[derive(Debug)]
struct SignedValue {
    kind: SignedKind,
    shape: Vec<usize>,
    data: Vec<i64>,
}

#[derive(Debug)]
struct UnsignedValue {
    kind: UnsignedKind,
    shape: Vec<usize>,
    data: Vec<u64>,
}

#[derive(Debug, Clone, Copy)]
enum SignedKind {
    Int4,
    Int8,
    Int32,
    Int64,
}

impl SignedKind {
    fn from_dtype(dtype: DataType) -> Result<Self, GraphError> {
        match dtype {
            DataType::Int4 => Ok(Self::Int4),
            DataType::Int8 => Ok(Self::Int8),
            DataType::Int32 => Ok(Self::Int32),
            DataType::Int64 => Ok(Self::Int64),
            _ => Err(error("signed buffer requires a signed integer datatype")),
        }
    }
    fn dtype(self) -> DataType {
        match self {
            Self::Int4 => DataType::Int4,
            Self::Int8 => DataType::Int8,
            Self::Int32 => DataType::Int32,
            Self::Int64 => DataType::Int64,
        }
    }
    fn bounds(self) -> (i64, i64) {
        match self {
            Self::Int4 => (-8, 7),
            Self::Int8 => (i64::from(i8::MIN), i64::from(i8::MAX)),
            Self::Int32 => (i64::from(i32::MIN), i64::from(i32::MAX)),
            Self::Int64 => (i64::MIN, i64::MAX),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum UnsignedKind {
    Uint4,
    Uint8,
    Uint32,
    Uint64,
}

impl UnsignedKind {
    fn from_dtype(dtype: DataType) -> Result<Self, GraphError> {
        match dtype {
            DataType::Uint4 => Ok(Self::Uint4),
            DataType::Uint8 => Ok(Self::Uint8),
            DataType::Uint32 => Ok(Self::Uint32),
            DataType::Uint64 => Ok(Self::Uint64),
            _ => Err(error(
                "unsigned buffer requires an unsigned integer datatype",
            )),
        }
    }
    fn dtype(self) -> DataType {
        match self {
            Self::Uint4 => DataType::Uint4,
            Self::Uint8 => DataType::Uint8,
            Self::Uint32 => DataType::Uint32,
            Self::Uint64 => DataType::Uint64,
        }
    }
    fn max(self) -> u64 {
        match self {
            Self::Uint4 => 15,
            Self::Uint8 => u64::from(u8::MAX),
            Self::Uint32 => u64::from(u32::MAX),
            Self::Uint64 => u64::MAX,
        }
    }
}

fn validate_len(shape: &[usize], len: usize) -> Result<(), GraphError> {
    let expected = super::shape::checked_numel(shape)?;
    if len != expected {
        return Err(error(&format!(
            "tensor shape {shape:?} expects {expected} elements, got {len}"
        )));
    }
    Ok(())
}
impl<B: Backend> TensorValue<B> {
    pub fn dtype(&self) -> DataType {
        match &self.storage {
            ValueStorage::Float32(_) => DataType::Float32,
            ValueStorage::Float16(_) => DataType::Float16,
            ValueStorage::Signed(value) => value.kind.dtype(),
            ValueStorage::Unsigned(value) => value.kind.dtype(),
        }
    }
    pub fn shape(&self) -> &[usize] {
        match &self.storage {
            ValueStorage::Float32(tensor) | ValueStorage::Float16(tensor) => tensor.shape(),
            ValueStorage::Signed(value) => &value.shape,
            ValueStorage::Unsigned(value) => &value.shape,
        }
    }
    pub fn float(&self) -> Result<&RuntimeTensor<B>, GraphError> {
        match &self.storage {
            ValueStorage::Float32(tensor) | ValueStorage::Float16(tensor) => Ok(tensor),
            _ => Err(error("integer operand reached a floating-point kernel")),
        }
    }
    pub fn signed_data(&self) -> Option<&[i64]> {
        match &self.storage {
            ValueStorage::Signed(value) => Some(&value.data),
            _ => None,
        }
    }
    pub fn unsigned_data(&self) -> Option<&[u64]> {
        match &self.storage {
            ValueStorage::Unsigned(value) => Some(&value.data),
            _ => None,
        }
    }
    // Operation semantics (wrapping, saturation, casts) are applied by kernels.
    // Constructors reject invalid storage rather than silently converting it.
    pub fn from_signed(
        dtype: DataType,
        shape: Vec<usize>,
        data: Vec<i64>,
    ) -> Result<Self, GraphError> {
        let kind = SignedKind::from_dtype(dtype)?;
        validate_len(&shape, data.len())?;
        let (min, max) = kind.bounds();
        if data.iter().any(|&v| v < min || v > max) {
            return Err(error("signed value outside the declared integer range"));
        }
        Ok(Self {
            storage: ValueStorage::Signed(SignedValue { kind, shape, data }),
        })
    }
    pub fn from_unsigned(
        dtype: DataType,
        shape: Vec<usize>,
        data: Vec<u64>,
    ) -> Result<Self, GraphError> {
        let kind = UnsignedKind::from_dtype(dtype)?;
        validate_len(&shape, data.len())?;
        let max = kind.max();
        if data.iter().any(|&v| v > max) {
            return Err(error("unsigned value outside the declared integer range"));
        }
        Ok(Self {
            storage: ValueStorage::Unsigned(UnsignedValue { kind, shape, data }),
        })
    }
    pub fn from_float(
        dtype: DataType,
        tensor: RuntimeTensor<B>,
        device: &B::Device,
    ) -> Result<Self, GraphError> {
        tensor.validate_storage_shape()?;
        match dtype {
            DataType::Float32 => Ok(Self {
                storage: ValueStorage::Float32(tensor),
            }),
            DataType::Float16 => Self::from_host(dtype, tensor.to_host_array()?, device),
            _ => Err(error("float buffer requires a floating-point datatype")),
        }
    }
    pub fn from_input(
        dtype: DataType,
        input: BurnInput,
        device: &B::Device,
    ) -> Result<Self, GraphError> {
        let count = super::shape::checked_numel(&input.shape)?;
        if input.data.len() != count {
            return Err(error("input element count mismatch"));
        }
        let BurnInput {
            shape,
            data,
            int64_data,
            uint64_data,
            ..
        } = input;
        match dtype {
            DataType::Bfloat16 | DataType::Float8e4m3 => {
                Err(error("bfloat16/float8e4m3 inputs are unsupported"))
            }
            DataType::Float32 | DataType::Float16 => {
                if int64_data.is_some() || uint64_data.is_some() {
                    return Err(error("float input has an integer payload"));
                }
                Self::from_host(dtype, HostArray::new(shape, data)?, device)
            }
            DataType::Int4 | DataType::Int8 | DataType::Int32 | DataType::Int64 => {
                if uint64_data.is_some() {
                    return Err(error("signed input has an unsigned payload"));
                }
                let values = match int64_data {
                    Some(values) => values,
                    None if matches!(dtype, DataType::Int4 | DataType::Int8) => {
                        let (min, max) = SignedKind::from_dtype(dtype)?.bounds();
                        // Narrow bounds are exactly representable in float32.
                        if data.iter().any(|v| {
                            !v.is_finite() || v.fract() != 0.0 || *v < min as f32 || *v > max as f32
                        }) {
                            return Err(error(
                                "narrow signed input requires finite, integral, in-range values",
                            ));
                        }
                        data.into_iter().map(|v| v as i64).collect()
                    }
                    None => return Err(error("signed input requires exact integer values")),
                };
                Self::from_signed(dtype, shape, values)
            }
            DataType::Uint4 | DataType::Uint8 | DataType::Uint32 | DataType::Uint64 => {
                if int64_data.is_some() {
                    return Err(error("unsigned input has a signed payload"));
                }
                let values = match uint64_data {
                    Some(values) => values,
                    None if matches!(dtype, DataType::Uint4 | DataType::Uint8) => {
                        let max = UnsignedKind::from_dtype(dtype)?.max();
                        if data.iter().any(|v| {
                            !v.is_finite() || v.fract() != 0.0 || *v < 0.0 || *v > max as f32
                        }) {
                            return Err(error(
                                "narrow unsigned input requires finite, integral, in-range values",
                            ));
                        }
                        data.into_iter().map(|v| v as u64).collect()
                    }
                    None => return Err(error("unsigned input requires exact integer values")),
                };
                Self::from_unsigned(dtype, shape, values)
            }
        }
    }
    pub fn output(&self, name: String) -> Result<BurnOutputWithData, GraphError> {
        let (data, signed, unsigned) = match &self.storage {
            ValueStorage::Float32(tensor) | ValueStorage::Float16(tensor) => {
                (tensor.to_host_array()?.data, None, None)
            }
            ValueStorage::Signed(value) => (
                value.data.iter().map(|&v| v as f32).collect(),
                Some(value.data.clone()),
                None,
            ),
            ValueStorage::Unsigned(value) => (
                value.data.iter().map(|&v| v as f32).collect(),
                None,
                Some(value.data.clone()),
            ),
        };
        Ok(BurnOutputWithData {
            name,
            shape: self.shape().to_vec(),
            data,
            int64_data: signed,
            uint64_data: unsigned,
        })
    }
    pub fn from_host(
        dtype: DataType,
        mut host: HostArray,
        device: &B::Device,
    ) -> Result<Self, GraphError> {
        validate_len(&host.shape, host.data.len())?;
        if !matches!(dtype, DataType::Float16 | DataType::Float32) {
            return Err(error("integer output requires a typed buffer"));
        }
        if dtype == DataType::Float16 {
            super::f16::round_f16_slice(&mut host.data);
        }
        let tensor = RuntimeTensor::from_host_array(host, device)?;
        Ok(Self {
            storage: if dtype == DataType::Float16 {
                ValueStorage::Float16(tensor)
            } else {
                ValueStorage::Float32(tensor)
            },
        })
    }
}
fn error(reason: &str) -> GraphError {
    GraphError::BurnRuntimeFailed {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::tensor_env::TensorEnv;
    use super::*;
    use burn_ndarray::NdArray;

    type Value = TensorValue<NdArray<f32>>;

    fn input(data: Vec<f32>) -> BurnInput {
        BurnInput {
            name: "x".into(),
            shape: vec![data.len()],
            data,
            int64_data: None,
            uint64_data: None,
        }
    }

    #[test]
    fn replacing_a_value_cannot_retain_an_old_integer_payload() {
        let mut env = TensorEnv::<NdArray<f32>>::new(Default::default());
        env.insert_input(
            7,
            DataType::Int64,
            BurnInput {
                name: "x".into(),
                shape: vec![1],
                data: vec![16_777_216.0],
                int64_data: Some(vec![16_777_217]),
                uint64_data: None,
            },
        )
        .unwrap();
        assert_eq!(
            env.value(7).unwrap().signed_data(),
            Some([16_777_217].as_slice())
        );
        assert!(env.get(7).is_err());
        assert!(
            env.insert_input(7, DataType::Uint8, input(vec![-1.]))
                .is_err()
        );
        assert_eq!(env.value(7).unwrap().dtype(), DataType::Int64);
        assert_eq!(
            env.value(7).unwrap().signed_data(),
            Some([16_777_217].as_slice())
        );
        env.insert_host(
            7,
            DataType::Float32,
            HostArray::new(vec![1], vec![2.0]).unwrap(),
        )
        .unwrap();
        assert_eq!(env.len(), 1);
        let result = env.value(7).unwrap().output("y".into()).unwrap();
        assert_eq!(result.data, [2.0]);
        assert!(result.int64_data.is_none());
        assert!(result.uint64_data.is_none());
    }

    #[test]
    fn half_inputs_and_outputs_share_the_same_rounding_boundary() {
        let value = TensorValue::<NdArray<f32>>::from_input(
            DataType::Float16,
            BurnInput {
                name: "x".into(),
                shape: vec![1],
                data: vec![1.0005],
                int64_data: None,
                uint64_data: None,
            },
            &Default::default(),
        )
        .unwrap();
        let result = value.output("y".into()).unwrap();
        assert_eq!(result.data, [half::f16::from_f32(1.0005).to_f32()]);
    }

    #[test]
    fn float_constructors_preserve_rounding_signed_zero_and_non_finite_values() {
        let samples = [
            1.0005,
            -0.,
            0.,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            1e-8,
        ];
        let device = Default::default();
        for dtype in [DataType::Float16, DataType::Float32] {
            let input_value = Value::from_input(dtype, input(samples.to_vec()), &device).unwrap();
            let host_value = Value::from_host(
                dtype,
                HostArray::new(vec![samples.len()], samples.to_vec()).unwrap(),
                &device,
            )
            .unwrap();
            let device_value = Value::from_float(
                dtype,
                RuntimeTensor::from_f32_data(vec![samples.len()], samples.to_vec(), &device)
                    .unwrap(),
                &device,
            )
            .unwrap();
            for value in [input_value, host_value, device_value] {
                assert_eq!(value.dtype(), dtype);
                let result = value.output("y".into()).unwrap();
                for (&actual, &sample) in result.data.iter().zip(&samples) {
                    let expected = if dtype == DataType::Float16 {
                        half::f16::from_f32(sample).to_f32()
                    } else {
                        sample
                    };
                    if expected.is_nan() {
                        assert!(actual.is_nan());
                    } else {
                        assert_eq!(actual.to_bits(), expected.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn constructors_reject_incompatible_storage_kinds_and_lengths() {
        let device = Default::default();
        for dtype in [
            DataType::Float16,
            DataType::Float32,
            DataType::Uint4,
            DataType::Uint8,
            DataType::Uint32,
            DataType::Uint64,
        ] {
            assert!(Value::from_signed(dtype, vec![1], vec![0]).is_err());
        }
        for dtype in [
            DataType::Float16,
            DataType::Float32,
            DataType::Int4,
            DataType::Int8,
            DataType::Int32,
            DataType::Int64,
        ] {
            assert!(Value::from_unsigned(dtype, vec![1], vec![0]).is_err());
        }
        for dtype in [
            DataType::Int4,
            DataType::Int8,
            DataType::Int32,
            DataType::Int64,
            DataType::Uint4,
            DataType::Uint8,
            DataType::Uint32,
            DataType::Uint64,
        ] {
            assert!(
                Value::from_host(dtype, HostArray::new(vec![1], vec![0.]).unwrap(), &device)
                    .is_err()
            );
            let tensor = RuntimeTensor::from_f32_data(vec![1], vec![0.], &device).unwrap();
            assert!(Value::from_float(dtype, tensor, &device).is_err());
        }
        for shape in [vec![], vec![2], vec![1, 0, 2]] {
            let bad_len = super::super::shape::checked_numel(&shape).unwrap() + 1;
            assert!(Value::from_signed(DataType::Int64, shape.clone(), vec![0; bad_len]).is_err());
            assert!(
                Value::from_unsigned(DataType::Uint64, shape.clone(), vec![0; bad_len]).is_err()
            );
            assert!(
                Value::from_host(
                    DataType::Float32,
                    HostArray {
                        shape,
                        data: vec![0.; bad_len]
                    },
                    &device
                )
                .is_err()
            );
        }
        assert!(Value::from_signed(DataType::Int64, vec![1; 9], vec![0]).is_err());
        assert!(
            Value::from_unsigned(DataType::Uint64, vec![u32::MAX as usize; 8], vec![]).is_err()
        );
    }

    #[test]
    fn signed_imports_enforce_width_and_preserve_exact_boundaries() {
        for (dtype, min, max) in [
            (DataType::Int4, -8, 7),
            (DataType::Int8, -128, 127),
            (DataType::Int32, i64::from(i32::MIN), i64::from(i32::MAX)),
            (DataType::Int64, i64::MIN, i64::MAX),
        ] {
            let mut x = input(vec![min as f32, max as f32]);
            x.int64_data = Some(vec![min, max]);
            let value = Value::from_input(dtype, x, &Default::default()).unwrap();
            assert_eq!(value.dtype(), dtype);
            assert_eq!(value.shape(), [2]);
            assert_eq!(value.signed_data(), Some([min, max].as_slice()));
            assert!(value.unsigned_data().is_none());
            assert!(value.float().is_err());
            assert_eq!(
                value.output("y".into()).unwrap().int64_data,
                Some(vec![min, max])
            );
            if dtype != DataType::Int64 {
                for invalid in [min - 1, max + 1] {
                    let mut x = input(vec![invalid as f32]);
                    x.int64_data = Some(vec![invalid]);
                    assert!(Value::from_input(dtype, x, &Default::default()).is_err());
                    assert!(Value::from_signed(dtype, vec![1], vec![invalid]).is_err());
                }
            }
        }
    }

    #[test]
    fn device_values_reject_inconsistent_storage_shapes() {
        let device = Default::default();
        for dtype in [DataType::Float16, DataType::Float32] {
            for shape in [vec![1, 1, 2, 1], vec![1, 1, 1, 3]] {
                let source = RuntimeTensor::<NdArray<f32>>::from_f32_data(
                    vec![1, 1, 1, 2],
                    vec![1., 2.],
                    &device,
                )
                .unwrap();
                let mismatched = RuntimeTensor::from_d4(shape, source.tensor_4().unwrap()).unwrap();
                assert!(Value::from_float(dtype, mismatched, &device).is_err());
            }
        }
    }

    #[test]
    fn unsigned_imports_enforce_width_and_preserve_exact_boundaries() {
        for (dtype, max) in [
            (DataType::Uint4, 15),
            (DataType::Uint8, 255),
            (DataType::Uint32, u64::from(u32::MAX)),
            (DataType::Uint64, u64::MAX),
        ] {
            let mut x = input(vec![0., max as f32]);
            x.uint64_data = Some(vec![0, max]);
            let value = Value::from_input(dtype, x, &Default::default()).unwrap();
            assert_eq!(value.dtype(), dtype);
            assert_eq!(value.unsigned_data(), Some([0, max].as_slice()));
            assert!(value.signed_data().is_none());
            assert!(value.float().is_err());
            assert_eq!(
                value.output("y".into()).unwrap().uint64_data,
                Some(vec![0, max])
            );
            if dtype != DataType::Uint64 {
                let invalid = max + 1;
                let mut x = input(vec![invalid as f32]);
                x.uint64_data = Some(vec![invalid]);
                assert!(Value::from_input(dtype, x, &Default::default()).is_err());
                assert!(Value::from_unsigned(dtype, vec![1], vec![invalid]).is_err());
            }
        }
    }

    #[test]
    fn narrow_float_imports_reject_lossy_casts() {
        for (dtype, min, max) in [
            (DataType::Int4, -8., 7.),
            (DataType::Int8, -128., 127.),
            (DataType::Uint4, 0., 15.),
            (DataType::Uint8, 0., 255.),
        ] {
            for invalid in [
                min - 1.,
                max + 1.,
                0.5,
                -0.5,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ] {
                assert!(
                    Value::from_input(dtype, input(vec![invalid]), &Default::default()).is_err(),
                    "{dtype:?}: {invalid}"
                );
            }
            let value =
                Value::from_input(dtype, input(vec![min, 0., max]), &Default::default()).unwrap();
            assert_eq!(value.output("y".into()).unwrap().data, [min, 0., max]);
        }
    }

    #[test]
    fn scalar_empty_and_wrong_input_payloads() {
        let device = Default::default();
        for shape in [vec![], vec![1, 0, 4]] {
            let n = super::super::shape::checked_numel(&shape).unwrap();
            assert_eq!(
                Value::from_signed(DataType::Int64, shape.clone(), vec![i64::MAX; n])
                    .unwrap()
                    .shape(),
                shape
            );
            assert_eq!(
                Value::from_unsigned(DataType::Uint64, shape.clone(), vec![u64::MAX; n])
                    .unwrap()
                    .shape(),
                shape
            );
            for dtype in [DataType::Float16, DataType::Float32] {
                assert_eq!(
                    Value::from_host(
                        dtype,
                        HostArray {
                            shape: shape.clone(),
                            data: vec![0.; n]
                        },
                        &device
                    )
                    .unwrap()
                    .shape(),
                    shape
                );
                let tensor =
                    RuntimeTensor::from_f32_data(shape.clone(), vec![0.; n], &device).unwrap();
                assert_eq!(
                    Value::from_float(dtype, tensor, &device).unwrap().shape(),
                    shape
                );
            }
        }
        for dtype in [
            DataType::Int32,
            DataType::Int64,
            DataType::Uint32,
            DataType::Uint64,
        ] {
            assert!(Value::from_input(dtype, input(vec![0.]), &device).is_err());
        }
        let mut signed = input(vec![0.]);
        signed.int64_data = Some(vec![0]);
        assert!(Value::from_input(DataType::Uint64, signed.clone(), &device).is_err());
        assert!(Value::from_input(DataType::Float32, signed.clone(), &device).is_err());
        signed.int64_data = Some(vec![]);
        assert!(Value::from_input(DataType::Int64, signed, &device).is_err());
        let mut unsigned = input(vec![0.]);
        unsigned.uint64_data = Some(vec![0]);
        assert!(Value::from_input(DataType::Int64, unsigned.clone(), &device).is_err());
        assert!(Value::from_input(DataType::Float16, unsigned.clone(), &device).is_err());
        unsigned.uint64_data = Some(vec![]);
        assert!(Value::from_input(DataType::Uint64, unsigned, &device).is_err());
    }
}

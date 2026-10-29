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

//! Burn device-resident tensors with runtime (dynamic) shapes.

use burn_tensor::activation::{
    elu, gelu, hard_sigmoid, hard_swish, leaky_relu, relu, sigmoid, softmax, softplus, softsign,
    tanh,
};
use burn_tensor::{Tensor, TensorData, backend::Backend};

use crate::error::GraphError;

use super::host_array::{HostArray, ReduceKind, broadcast_to};
use super::shape::broadcast_shapes;

// These macros only bridge runtime rank to Burn's const generic rank.
// Scalar is a distinct logical rank backed by Tensor<B, 1>.
macro_rules! ranked_ref {
    ($inner:expr, |$t:ident| $body:expr) => {
        match $inner {
            RankedTensor::Scalar($t) => $body,
            RankedTensor::D1($t) => $body,
            RankedTensor::D2($t) => $body,
            RankedTensor::D3($t) => $body,
            RankedTensor::D4($t) => $body,
            RankedTensor::D5($t) => $body,
            RankedTensor::D6($t) => $body,
            RankedTensor::D7($t) => $body,
            RankedTensor::D8($t) => $body,
        }
    };
}
macro_rules! ranked_map {
    ($inner:expr, |$t:ident| $body:expr) => {
        match $inner {
            RankedTensor::Scalar($t) => RankedTensor::Scalar($body),
            RankedTensor::D1($t) => RankedTensor::D1($body),
            RankedTensor::D2($t) => RankedTensor::D2($body),
            RankedTensor::D3($t) => RankedTensor::D3($body),
            RankedTensor::D4($t) => RankedTensor::D4($body),
            RankedTensor::D5($t) => RankedTensor::D5($body),
            RankedTensor::D6($t) => RankedTensor::D6($body),
            RankedTensor::D7($t) => RankedTensor::D7($body),
            RankedTensor::D8($t) => RankedTensor::D8($body),
        }
    };
}
macro_rules! ranked_zip {
    ($left:expr, $right:expr, |$a:ident, $b:ident| $body:expr, $error:expr) => {
        match ($left, $right) {
            (RankedTensor::Scalar($a), RankedTensor::Scalar($b)) => RankedTensor::Scalar($body),
            (RankedTensor::D1($a), RankedTensor::D1($b)) => RankedTensor::D1($body),
            (RankedTensor::D2($a), RankedTensor::D2($b)) => RankedTensor::D2($body),
            (RankedTensor::D3($a), RankedTensor::D3($b)) => RankedTensor::D3($body),
            (RankedTensor::D4($a), RankedTensor::D4($b)) => RankedTensor::D4($body),
            (RankedTensor::D5($a), RankedTensor::D5($b)) => RankedTensor::D5($body),
            (RankedTensor::D6($a), RankedTensor::D6($b)) => RankedTensor::D6($body),
            (RankedTensor::D7($a), RankedTensor::D7($b)) => RankedTensor::D7($body),
            (RankedTensor::D8($a), RankedTensor::D8($b)) => RankedTensor::D8($body),
            _ => return Err($error),
        }
    };
}
macro_rules! dispatch_rank {
    ($rank:expr, $apply:ident) => {
        match $rank {
            1 => $apply!(1, D1),
            2 => $apply!(2, D2),
            3 => $apply!(3, D3),
            4 => $apply!(4, D4),
            5 => $apply!(5, D5),
            6 => $apply!(6, D6),
            7 => $apply!(7, D7),
            8 => $apply!(8, D8),
            rank => return Err(burn_err(format!("unsupported tensor rank {rank}"))),
        }
    };
}

const MAX_RANK: usize = 8;

/// Device-resident tensor whose logical shape may include dynamic axes resolved at runtime.
#[derive(Debug, Clone)]
pub struct RuntimeTensor<B: Backend> {
    shape: Vec<usize>,
    inner: RankedTensor<B>,
}

#[derive(Debug, Clone)]
pub(crate) enum RankedTensor<B: Backend> {
    /// WebNN scalar (`shape == []`, one element).
    Scalar(Tensor<B, 1>),
    D1(Tensor<B, 1>),
    D2(Tensor<B, 2>),
    D3(Tensor<B, 3>),
    D4(Tensor<B, 4>),
    D5(Tensor<B, 5>),
    D6(Tensor<B, 6>),
    D7(Tensor<B, 7>),
    D8(Tensor<B, 8>),
}

impl<B: Backend> RuntimeTensor<B> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product::<usize>()
        }
    }

    pub(super) fn validate_storage_shape(&self) -> Result<(), GraphError> {
        super::shape::checked_numel(&self.shape)?;
        let expected = if self.shape.is_empty() {
            &[1usize][..]
        } else {
            &self.shape
        };
        let check = |actual: &[usize]| {
            if actual != expected {
                return Err(burn_err(format!(
                    "tensor storage shape {actual:?} does not match logical shape {:?}",
                    self.shape
                )));
            }
            Ok(())
        };
        ranked_ref!(&self.inner, |t| check(&t.dims()))
    }

    pub fn from_f32_data(
        shape: Vec<usize>,
        data: Vec<f32>,
        device: &B::Device,
    ) -> Result<Self, GraphError> {
        let expected = super::shape::checked_numel(&shape)?;
        if data.len() != expected {
            return Err(burn_err(format!(
                "tensor shape {shape:?} expects {expected} elements, got {}",
                data.len()
            )));
        }

        let inner = if shape.is_empty() {
            RankedTensor::Scalar(Tensor::from_data(TensorData::new(data, [1usize]), device))
        } else {
            ranked_from_data(&shape, data, device)?
        };

        Ok(Self { shape, inner })
    }

    pub fn from_host_array(array: HostArray, device: &B::Device) -> Result<Self, GraphError> {
        Self::from_f32_data(array.shape, array.data, device)
    }

    pub fn to_host_array(&self) -> Result<HostArray, GraphError> {
        let data = self.read_f32_data()?;
        HostArray::new(self.shape.clone(), data)
    }

    fn read_f32_data(&self) -> Result<Vec<f32>, GraphError> {
        ranked_ref!(&self.inner, |t| read_tensor(t))
    }
}

fn reshape_tensor<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    shape: &[usize],
) -> Result<Tensor<B, D>, GraphError> {
    // Scalars use physical shape [1] while retaining logical shape [].
    let physical = if shape.is_empty() {
        &[1usize][..]
    } else {
        shape
    };
    Ok(tensor.clone().reshape(shape_to_array::<D>(physical)?))
}

fn read_tensor<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Result<Vec<f32>, GraphError> {
    tensor
        .clone()
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| burn_err(format!("failed to read burn tensor data: {err}")))
}

fn ranked_from_data<B: Backend>(
    shape: &[usize],
    data: Vec<f32>,
    device: &B::Device,
) -> Result<RankedTensor<B>, GraphError> {
    if shape.len() > MAX_RANK {
        return Err(burn_err(format!(
            "tensor rank {} exceeds supported maximum {MAX_RANK}",
            shape.len()
        )));
    }

    macro_rules! mk {
        ($d:literal, $variant:ident) => {{
            let burn_shape = shape_to_array::<$d>(shape)?;
            Ok(RankedTensor::$variant(Tensor::from_data(
                TensorData::new(data, burn_shape),
                device,
            )))
        }};
    }

    dispatch_rank!(shape.len(), mk)
}

fn shape_to_array<const D: usize>(shape: &[usize]) -> Result<[usize; D], GraphError> {
    if shape.len() != D {
        return Err(burn_err(format!(
            "internal shape rank mismatch: expected {D}, got {}",
            shape.len()
        )));
    }
    shape
        .try_into()
        .map_err(|_| burn_err(format!("failed to convert shape {shape:?} to array")))
}

fn burn_err(reason: String) -> GraphError {
    GraphError::BurnRuntimeFailed { reason }
}

/// Device-side elementwise binary ops for float tensors (with NumPy-style broadcast).
#[derive(Clone, Copy, Debug)]
pub enum DeviceBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

/// Device-side unary float ops.
#[derive(Clone, Copy, Debug)]
pub enum UnaryDeviceOp {
    Abs,
    Neg,
    Exp,
    Log,
    Sqrt,
    Ceil,
    Floor,
    Cos,
    Sin,
    Tan,
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    Softplus,
    Softsign,
    LeakyRelu { slope: f64 },
    Elu { alpha: f64 },
    HardSigmoid { alpha: f64, beta: f64 },
    HardSwish,
    Softmax { axis: usize },
}

impl<B: Backend> RuntimeTensor<B> {
    pub fn binary_broadcast(
        &self,
        other: &Self,
        op: DeviceBinaryOp,
        device: &B::Device,
    ) -> Result<Self, GraphError> {
        let out_shape = broadcast_shapes(&self.shape, &other.shape)?;
        let lhs = self.broadcast_to_shape(&out_shape, device)?;
        let rhs = other.broadcast_to_shape(&out_shape, device)?;
        lhs.binary_same_shape_force(&rhs, op, &out_shape)
    }

    pub fn unary(&self, op: UnaryDeviceOp) -> Result<Self, GraphError> {
        let inner = ranked_map!(&self.inner, |t| apply_unary(t, op));
        Ok(Self {
            shape: self.shape.clone(),
            inner,
        })
    }

    pub(crate) fn from_d4(shape: Vec<usize>, tensor: Tensor<B, 4>) -> Result<Self, GraphError> {
        if shape.len() != 4 {
            return Err(burn_err(format!(
                "from_d4 expected rank-4 shape, got {:?}",
                shape
            )));
        }
        Ok(Self {
            shape,
            inner: RankedTensor::D4(tensor),
        })
    }

    pub(crate) fn tensor_4(&self) -> Result<Tensor<B, 4>, GraphError> {
        match &self.inner {
            RankedTensor::D4(t) => Ok(t.clone()),
            _ => Err(burn_err(format!(
                "expected 4D tensor, got rank {}",
                self.rank()
            ))),
        }
    }

    pub(crate) fn reduce_dims(&self, axes: &[usize], kind: ReduceKind) -> Result<Self, GraphError> {
        let axes = axes
            .iter()
            .map(|&a| super::shape::axis_index(a, self.rank()).map(|a| a as u32))
            .collect::<Result<Vec<_>, _>>()?;
        let spec = super::shape::ReductionSpec::new(&self.shape, Some(&axes), true)?;
        let shape = spec.output_shape;
        let mut sorted = spec.axes;
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let inner = ranked_map!(&self.inner, |t| reduce_tensor(t.clone(), &sorted, kind)?);
        Ok(Self { shape, inner })
    }

    pub(crate) fn reshape_to(
        &self,
        new_shape: &[usize],
        device: &B::Device,
    ) -> Result<Self, GraphError> {
        if self.shape.as_slice() == new_shape {
            return Ok(self.clone());
        }
        let expected_numel = super::shape::checked_numel(new_shape)?;
        if self.numel() != expected_numel {
            return Err(burn_err(format!(
                "reshape_to numel mismatch: {} vs {expected_numel} for shapes {:?} -> {new_shape:?}",
                self.numel(),
                self.shape
            )));
        }
        if self.rank() != new_shape.len() {
            let data = self.read_f32_data()?;
            return Self::from_f32_data(new_shape.to_vec(), data, device);
        }
        let inner = ranked_map!(&self.inner, |t| reshape_tensor(t, new_shape)?);
        Ok(Self {
            shape: new_shape.to_vec(),
            inner,
        })
    }

    pub fn swap_last_two_dims(&self) -> Result<Self, GraphError> {
        let rank = self.rank();
        if rank < 2 {
            return Err(burn_err(format!(
                "swap_last_two_dims requires rank >= 2, got {rank}"
            )));
        }
        self.swap_dims_axes(rank - 2, rank - 1)
    }

    pub fn swap_dims_axes(&self, dim_a: usize, dim_b: usize) -> Result<Self, GraphError> {
        super::shape::axis_index(dim_a, self.rank())?;
        super::shape::axis_index(dim_b, self.rank())?;
        let mut out_shape = self.shape.clone();
        out_shape.swap(dim_a, dim_b);
        let inner = ranked_map!(&self.inner, |t| t.clone().swap_dims(dim_a, dim_b));
        Ok(Self {
            shape: out_shape,
            inner,
        })
    }

    pub fn square(&self) -> Result<Self, GraphError> {
        let inner = ranked_map!(&self.inner, |t| t.clone() * t.clone());
        Ok(Self {
            shape: self.shape.clone(),
            inner,
        })
    }

    pub fn mul_scalar(&self, factor: f32) -> Result<Self, GraphError> {
        let inner = ranked_map!(&self.inner, |t| t.clone().mul_scalar(factor));
        Ok(Self {
            shape: self.shape.clone(),
            inner,
        })
    }

    pub fn add_scaled(
        &self,
        other: &Self,
        beta: f32,
        device: &B::Device,
    ) -> Result<Self, GraphError> {
        let rhs = other.broadcast_to_shape(self.shape(), device)?;
        let inner = ranked_zip!(
            &self.inner,
            &rhs.inner,
            |a, b| a.clone() + b.clone().mul_scalar(beta),
            burn_err(format!(
                "add_scaled rank mismatch {:?} vs {:?}",
                self.shape, other.shape
            ))
        );
        Ok(Self {
            shape: self.shape.clone(),
            inner,
        })
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, GraphError> {
        if self.rank() < 2 || other.rank() < 2 {
            return Err(burn_err(format!(
                "matmul requires rank >= 2, got {} and {}",
                self.rank(),
                other.rank()
            )));
        }
        let out_shape = matmul_output_shape(&self.shape, &other.shape)?;
        let inner = ranked_zip!(
            &self.inner,
            &other.inner,
            |a, b| a.clone().matmul(b.clone()),
            burn_err(format!(
                "matmul unsupported ranks/shapes {:?} @ {:?}",
                self.shape, other.shape
            ))
        );
        Ok(Self {
            shape: out_shape,
            inner,
        })
    }

    pub fn broadcast_to_shape(
        &self,
        target: &[usize],
        device: &B::Device,
    ) -> Result<Self, GraphError> {
        if self.shape.as_slice() == target {
            return Ok(self.clone());
        }
        if self.numel() == 1
            && let Ok(expanded) = self.expand_singleton_on_device(target)
        {
            return Ok(expanded);
        }
        let host = self.to_host_array()?;
        let expanded = broadcast_to(&host, target)?;
        Self::from_f32_data(expanded.shape, expanded.data, device)
    }

    fn expand_singleton_on_device(&self, target: &[usize]) -> Result<Self, GraphError> {
        let rank = target.len().max(1);
        let padded = pad_leading_ones(&self.shape, rank);
        macro_rules! expand {
            ($d:literal, $variant:ident) => {{
                let t = scalar_or_d1(&self.inner)?;
                let source = shape_to_array::<$d>(&padded)?;
                let target = shape_to_array::<$d>(target)?;
                RankedTensor::$variant(t.reshape(source).expand(target))
            }};
        }
        let inner = dispatch_rank!(rank, expand);
        Ok(Self {
            shape: target.to_vec(),
            inner,
        })
    }

    fn binary_same_shape_force(
        &self,
        other: &Self,
        op: DeviceBinaryOp,
        out_shape: &[usize],
    ) -> Result<Self, GraphError> {
        let inner = ranked_zip!(
            &self.inner,
            &other.inner,
            |a, b| apply_binary(a, b, op),
            burn_err(format!(
                "binary op rank mismatch {:?} vs {:?}",
                self.shape, other.shape
            ))
        );
        Ok(Self {
            shape: out_shape.to_vec(),
            inner,
        })
    }
}

fn pad_leading_ones(shape: &[usize], rank: usize) -> Vec<usize> {
    let mut out = vec![1; rank.saturating_sub(shape.len())];
    out.extend_from_slice(shape);
    out
}

fn scalar_or_d1<B: Backend>(inner: &RankedTensor<B>) -> Result<Tensor<B, 1>, GraphError> {
    match inner {
        RankedTensor::Scalar(t) | RankedTensor::D1(t) => Ok(t.clone()),
        _ => Err(burn_err(
            "expected scalar or 1D tensor for singleton expand".to_string(),
        )),
    }
}

fn matmul_output_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, GraphError> {
    if a.len() < 2 || b.len() < 2 {
        return Err(burn_err("matmul requires rank >= 2".to_string()));
    }
    if a[a.len() - 1] != b[b.len() - 2] {
        return Err(burn_err(format!(
            "matmul inner dim mismatch: {} != {}",
            a[a.len() - 1],
            b[b.len() - 2]
        )));
    }
    let batch = broadcast_shapes(&a[..a.len() - 2], &b[..b.len() - 2])?;
    let mut out = batch;
    out.push(a[a.len() - 2]);
    out.push(b[b.len() - 1]);
    Ok(out)
}

pub(super) fn matmul_broadcast_operand_shapes(
    a: &[usize],
    b: &[usize],
) -> Result<(Vec<usize>, Vec<usize>), GraphError> {
    let _ = matmul_output_shape(a, b)?;
    let batch = broadcast_shapes(&a[..a.len() - 2], &b[..b.len() - 2])?;
    let mut a_out = batch.clone();
    a_out.push(a[a.len() - 2]);
    a_out.push(a[a.len() - 1]);
    let mut b_out = batch;
    b_out.push(b[b.len() - 2]);
    b_out.push(b[b.len() - 1]);
    Ok((a_out, b_out))
}

fn apply_binary<B: Backend, const D: usize>(
    a: &Tensor<B, D>,
    b: &Tensor<B, D>,
    op: DeviceBinaryOp,
) -> Tensor<B, D> {
    let a = a.clone();
    let b = b.clone();
    match op {
        DeviceBinaryOp::Add => a + b,
        DeviceBinaryOp::Sub => a - b,
        DeviceBinaryOp::Mul => a * b,
        DeviceBinaryOp::Div => a / b,
        DeviceBinaryOp::Min => a.min_pair(b),
        DeviceBinaryOp::Max => a.max_pair(b),
    }
}

fn reduce_tensor<B: Backend, const D: usize>(
    mut tensor: Tensor<B, D>,
    axes: &[usize],
    kind: ReduceKind,
) -> Result<Tensor<B, D>, GraphError> {
    for &ax in axes {
        tensor = match kind {
            ReduceKind::Sum => tensor.sum_dim(ax),
            ReduceKind::Mean => tensor.mean_dim(ax),
            ReduceKind::Max => tensor.max_dim(ax),
            ReduceKind::Min => tensor.min_dim(ax),
            ReduceKind::Product => tensor.prod_dim(ax),
            _ => {
                return Err(burn_err(
                    "reduce_tensor does not support this reduction on device".to_string(),
                ));
            }
        };
    }
    Ok(tensor)
}

fn apply_unary<B: Backend, const D: usize>(t: &Tensor<B, D>, op: UnaryDeviceOp) -> Tensor<B, D> {
    let t = t.clone();
    match op {
        UnaryDeviceOp::Abs => t.abs(),
        UnaryDeviceOp::Neg => -t,
        UnaryDeviceOp::Exp => t.exp(),
        UnaryDeviceOp::Log => t.log(),
        UnaryDeviceOp::Sqrt => t.sqrt(),
        UnaryDeviceOp::Ceil => t.ceil(),
        UnaryDeviceOp::Floor => t.floor(),
        UnaryDeviceOp::Cos => t.cos(),
        UnaryDeviceOp::Sin => t.sin(),
        UnaryDeviceOp::Tan => t.tan(),
        UnaryDeviceOp::Relu => relu(t),
        UnaryDeviceOp::Sigmoid => sigmoid(t),
        UnaryDeviceOp::Tanh => tanh(t),
        UnaryDeviceOp::Gelu => gelu(t),
        UnaryDeviceOp::Softplus => softplus(t, 1.0),
        UnaryDeviceOp::Softsign => softsign(t),
        UnaryDeviceOp::LeakyRelu { slope } => leaky_relu(t, slope),
        UnaryDeviceOp::Elu { alpha } => elu(t, alpha),
        UnaryDeviceOp::HardSigmoid { alpha, beta } => hard_sigmoid(t, alpha, beta),
        UnaryDeviceOp::HardSwish => hard_swish(t),
        UnaryDeviceOp::Softmax { axis } => softmax(t, axis),
    }
}

#[cfg(all(test, feature = "burn-cpu-runtime"))]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn round_trip_scalar_and_dynamic_shape() {
        let device = Default::default();
        let scalar =
            RuntimeTensor::<TestBackend>::from_f32_data(vec![], vec![3.5], &device).unwrap();
        assert_eq!(scalar.shape(), &[] as &[usize]);
        assert_eq!(scalar.to_host_array().unwrap().data, vec![3.5]);

        let dynamic =
            RuntimeTensor::<TestBackend>::from_f32_data(vec![1, 0, 4], vec![], &device).unwrap();
        assert_eq!(dynamic.shape(), &[1, 0, 4]);
        assert!(dynamic.to_host_array().unwrap().data.is_empty());
    }
    #[test]
    fn rank_adapter_preserves_scalars_and_ndarray_supported_ranks() {
        let device = Default::default();
        // NdArray native reshape/expansion and batched matmul are limited to six ranks.
        for rank in 0..=6 {
            let shape = vec![2; rank];
            let count = 1usize << rank;
            let a = RuntimeTensor::<TestBackend>::from_f32_data(
                shape.clone(),
                vec![2.0; count],
                &device,
            )
            .unwrap();
            let scalar =
                RuntimeTensor::<TestBackend>::from_f32_data(vec![], vec![3.0], &device).unwrap();
            let sum = a
                .binary_broadcast(&scalar, DeviceBinaryOp::Add, &device)
                .unwrap();
            sum.validate_storage_shape().unwrap();
            assert_eq!(sum.shape(), shape);
            assert_eq!(sum.to_host_array().unwrap().data, vec![5.0; count]);
            let negative = sum.unary(UnaryDeviceOp::Neg).unwrap();
            let squared = negative.square().unwrap();
            assert_eq!(squared.shape(), shape);
            assert_eq!(squared.to_host_array().unwrap().data, vec![25.0; count]);
            let scaled = a
                .mul_scalar(2.0)
                .unwrap()
                .add_scaled(&a, 3.0, &device)
                .unwrap();
            assert_eq!(scaled.to_host_array().unwrap().data, vec![10.0; count]);
            let axes = (0..rank).collect::<Vec<_>>();
            let reduced = a.reduce_dims(&axes, ReduceKind::Sum).unwrap();
            reduced.validate_storage_shape().unwrap();
            assert_eq!(reduced.to_host_array().unwrap().data, [2.0 * count as f32]);
            let reshaped = a.reshape_to(&vec![1; rank], &device);
            assert_eq!(reshaped.is_ok(), rank == 0);
            if rank >= 2 {
                let mut target = shape.clone();
                target[0] = 4;
                target[1] = 1;
                let reshaped = a.reshape_to(&target, &device).unwrap();
                reshaped.validate_storage_shape().unwrap();
                assert_eq!(reshaped.shape(), target);
                let swapped = reshaped.swap_dims_axes(0, 1).unwrap();
                swapped.validate_storage_shape().unwrap();
                target.swap(0, 1);
                assert_eq!(swapped.shape(), target);
            }
        }
    }
    #[test]
    fn rank_adapter_matmul_matches_expected_results_at_every_ndarray_matrix_rank() {
        let device = Default::default();
        for rank in 2..=6 {
            let mut shape = vec![1; rank];
            shape[rank - 2] = 2;
            shape[rank - 1] = 2;
            let a = RuntimeTensor::<TestBackend>::from_f32_data(
                shape.clone(),
                vec![1.0, 2.0, 3.0, 4.0],
                &device,
            )
            .unwrap();
            let out = a.matmul(&a).unwrap();
            out.validate_storage_shape().unwrap();
            assert_eq!(out.shape(), shape);
            assert_eq!(out.to_host_array().unwrap().data, [7.0, 10.0, 15.0, 22.0]);
        }
    }
    #[test]
    fn cpu_executor_reports_ndarray_native_reshape_limit_as_a_runtime_error() {
        use super::super::{BurnInput, run_burn_cpu_with_inputs};
        use crate::burn::{BurnGraphPlan, IOBinding};
        use crate::{DataType, Operation};
        use std::collections::HashMap;
        for rank in 7..=8 {
            let binding = |name: &str, operand_id| IOBinding {
                name: name.into(),
                operand_id,
                data_type: DataType::Float32,
                shape: vec![1; rank],
            };
            let plan = BurnGraphPlan::new(
                vec![binding("x", 0)],
                vec![binding("y", 1)],
                vec![],
                vec![Operation::Matmul {
                    a: 0,
                    b: 0,
                    options: None,
                    outputs: vec![1],
                }],
                HashMap::from([(0, DataType::Float32), (1, DataType::Float32)]),
            );
            let input = BurnInput {
                name: "x".into(),
                shape: vec![1; rank],
                data: vec![1.0],
                int64_data: None,
                uint64_data: None,
            };
            let error =
                run_burn_cpu_with_inputs(&plan.serialize().unwrap(), vec![input]).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("NdArray supports arrays up to 6 dimensions"),
                "{error}"
            );
        }
    }
}

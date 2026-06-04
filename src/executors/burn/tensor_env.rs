//! Burn device-resident tensors with runtime (dynamic) shapes.

use std::collections::HashMap;

use burn::tensor::activation::{
    elu, gelu, hard_sigmoid, hard_swish, leaky_relu, relu, sigmoid, softmax, softplus, softsign,
    tanh,
};
use burn::tensor::{Slice, Tensor, TensorData, backend::Backend};

use crate::error::GraphError;
use crate::graph::DataType;

use super::host_array::{HostArray, ReduceKind, broadcast_shapes, broadcast_to};

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

    pub fn from_f32_data(
        shape: Vec<usize>,
        data: Vec<f32>,
        device: &B::Device,
    ) -> Result<Self, GraphError> {
        let expected = if shape.is_empty() {
            1
        } else {
            shape.iter().product::<usize>()
        };
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
        match &self.inner {
            RankedTensor::Scalar(t) | RankedTensor::D1(t) => read_tensor(t),
            RankedTensor::D2(t) => read_tensor(t),
            RankedTensor::D3(t) => read_tensor(t),
            RankedTensor::D4(t) => read_tensor(t),
            RankedTensor::D5(t) => read_tensor(t),
            RankedTensor::D6(t) => read_tensor(t),
            RankedTensor::D7(t) => read_tensor(t),
            RankedTensor::D8(t) => read_tensor(t),
        }
    }
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

    match shape.len() {
        1 => mk!(1, D1),
        2 => mk!(2, D2),
        3 => mk!(3, D3),
        4 => mk!(4, D4),
        5 => mk!(5, D5),
        6 => mk!(6, D6),
        7 => mk!(7, D7),
        8 => mk!(8, D8),
        other => Err(burn_err(format!("unsupported tensor rank {other}"))),
    }
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

/// Execution environment holding device-resident operand tensors.
pub struct TensorEnv<B: Backend> {
    pub device: B::Device,
    pub tensors: HashMap<u32, RuntimeTensor<B>>,
    pub dtypes: HashMap<u32, DataType>,
    pub int64_data: HashMap<u32, Vec<i64>>,
    pub uint64_data: HashMap<u32, Vec<u64>>,
}

impl<B: Backend> TensorEnv<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            device,
            tensors: HashMap::new(),
            dtypes: HashMap::new(),
            int64_data: HashMap::new(),
            uint64_data: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: u32, dtype: DataType, tensor: RuntimeTensor<B>) {
        self.dtypes.insert(id, dtype);
        self.tensors.insert(id, tensor);
    }

    pub fn insert_with_integer_sidecar(
        &mut self,
        id: u32,
        dtype: DataType,
        tensor: RuntimeTensor<B>,
        int64_data: Option<Vec<i64>>,
        uint64_data: Option<Vec<u64>>,
    ) {
        if let Some(values) = int64_data {
            self.int64_data.insert(id, values);
        }
        if let Some(values) = uint64_data {
            self.uint64_data.insert(id, values);
        }
        self.insert(id, dtype, tensor);
    }

    pub fn get(&self, id: u32) -> Result<&RuntimeTensor<B>, GraphError> {
        self.tensors
            .get(&id)
            .ok_or_else(|| GraphError::BurnRuntimeFailed {
                reason: format!("operand {id} is not available before use"),
            })
    }

    pub fn get_host(&self, id: u32) -> Result<HostArray, GraphError> {
        let mut host = self.get(id)?.to_host_array()?;
        if let Some(values) = self.int64_data.get(&id) {
            host.i64_data = Some(values.clone());
        }
        if let Some(values) = self.uint64_data.get(&id) {
            host.u64_data = Some(values.clone());
        }
        Ok(host)
    }

    pub fn optional_host(&self, id: u32) -> Result<Option<HostArray>, GraphError> {
        if self.tensors.contains_key(&id) {
            Ok(Some(self.get_host(id)?))
        } else {
            Ok(None)
        }
    }

    pub fn binary_host<F>(&self, a: u32, b: u32, f: F) -> Result<HostArray, GraphError>
    where
        F: FnOnce(&HostArray, &HostArray) -> Result<HostArray, GraphError>,
    {
        let a_arr = self.get_host(a)?;
        let b_arr = self.get_host(b)?;
        f(&a_arr, &b_arr)
    }

    /// Elementwise binary op on device tensors when shapes match (avoids host round-trip).
    pub fn binary_device_same_shape(
        &self,
        a_id: u32,
        b_id: u32,
        op: DeviceBinaryOp,
    ) -> Result<Option<RuntimeTensor<B>>, GraphError> {
        let a = self.get(a_id)?;
        let b = self.get(b_id)?;
        a.binary_same_shape(b, op)
    }

    pub fn binary_broadcast(
        &self,
        a_id: u32,
        b_id: u32,
        op: DeviceBinaryOp,
    ) -> Result<RuntimeTensor<B>, GraphError> {
        let a = self.get(a_id)?;
        let b = self.get(b_id)?;
        a.binary_broadcast(b, op, &self.device)
    }

    pub fn compare_broadcast(
        &self,
        a_id: u32,
        b_id: u32,
        op: DeviceCompareOp,
    ) -> Result<RuntimeTensor<B>, GraphError> {
        let a = self.get(a_id)?;
        let b = self.get(b_id)?;
        a.compare_broadcast(b, op, &self.device)
    }

    pub fn unary(&self, input_id: u32, op: UnaryDeviceOp) -> Result<RuntimeTensor<B>, GraphError> {
        self.get(input_id)?.unary(op)
    }

    pub fn matmul(&self, a_id: u32, b_id: u32) -> Result<RuntimeTensor<B>, GraphError> {
        let a = self.get(a_id)?;
        let b = self.get(b_id)?;
        let (a_shape, b_shape) = matmul_broadcast_operand_shapes(a.shape(), b.shape())?;
        let a = a.broadcast_to_shape(&a_shape, &self.device)?;
        let b = b.broadcast_to_shape(&b_shape, &self.device)?;
        a.matmul(&b)
    }
}

/// Device-side elementwise binary ops for float tensors (with NumPy-style broadcast).
#[derive(Clone, Copy, Debug)]
pub enum DeviceBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Min,
    Max,
}

/// Device-side comparison ops (output float 0.0 / 1.0).
#[derive(Clone, Copy, Debug)]
pub enum DeviceCompareOp {
    Equal,
    Greater,
    GreaterOrEqual,
    Lesser,
    LesserOrEqual,
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
    Clamp { min: f32, max: f32 },
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

    pub fn compare_broadcast(
        &self,
        other: &Self,
        op: DeviceCompareOp,
        device: &B::Device,
    ) -> Result<Self, GraphError> {
        let out_shape = broadcast_shapes(&self.shape, &other.shape)?;
        let lhs = self.broadcast_to_shape(&out_shape, device)?;
        let rhs = other.broadcast_to_shape(&out_shape, device)?;
        lhs.compare_same_shape(&rhs, op, &out_shape)
    }

    pub fn unary(&self, op: UnaryDeviceOp) -> Result<Self, GraphError> {
        let inner = match &self.inner {
            RankedTensor::Scalar(t) => RankedTensor::Scalar(apply_unary(t, op)),
            RankedTensor::D1(t) => RankedTensor::D1(apply_unary(t, op)),
            RankedTensor::D2(t) => RankedTensor::D2(apply_unary(t, op)),
            RankedTensor::D3(t) => RankedTensor::D3(apply_unary(t, op)),
            RankedTensor::D4(t) => RankedTensor::D4(apply_unary(t, op)),
            RankedTensor::D5(t) => RankedTensor::D5(apply_unary(t, op)),
            RankedTensor::D6(t) => RankedTensor::D6(apply_unary(t, op)),
            RankedTensor::D7(t) => RankedTensor::D7(apply_unary(t, op)),
            RankedTensor::D8(t) => RankedTensor::D8(apply_unary(t, op)),
        };
        Ok(Self {
            shape: self.shape.clone(),
            inner,
        })
    }

    pub(crate) fn from_inner(shape: Vec<usize>, inner: RankedTensor<B>) -> Self {
        Self { shape, inner }
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
        let mut sorted = axes.to_vec();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let mut shape = self.shape.clone();
        for &ax in axes {
            if ax < shape.len() {
                shape[ax] = 1;
            }
        }
        let inner = match &self.inner {
            RankedTensor::Scalar(t) | RankedTensor::D1(t) => {
                RankedTensor::D1(reduce_tensor(t.clone(), &sorted, kind)?)
            }
            RankedTensor::D2(t) => RankedTensor::D2(reduce_tensor(t.clone(), &sorted, kind)?),
            RankedTensor::D3(t) => RankedTensor::D3(reduce_tensor(t.clone(), &sorted, kind)?),
            RankedTensor::D4(t) => RankedTensor::D4(reduce_tensor(t.clone(), &sorted, kind)?),
            RankedTensor::D5(t) => RankedTensor::D5(reduce_tensor(t.clone(), &sorted, kind)?),
            RankedTensor::D6(t) => RankedTensor::D6(reduce_tensor(t.clone(), &sorted, kind)?),
            RankedTensor::D7(t) => RankedTensor::D7(reduce_tensor(t.clone(), &sorted, kind)?),
            RankedTensor::D8(t) => RankedTensor::D8(reduce_tensor(t.clone(), &sorted, kind)?),
        };
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
        let expected_numel = new_shape.iter().product::<usize>().max(1);
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
        let rank = new_shape.len().max(1);
        let inner = match (rank, &self.inner) {
            (0, RankedTensor::Scalar(t) | RankedTensor::D1(t)) => {
                let data_len = new_shape.iter().product::<usize>().max(1);
                RankedTensor::Scalar(t.clone().reshape([data_len]))
            }
            (1, RankedTensor::Scalar(t) | RankedTensor::D1(t)) => {
                let s = shape_to_array::<1>(new_shape)?;
                RankedTensor::D1(t.clone().reshape(s))
            }
            (2, RankedTensor::D2(t)) => {
                let s = shape_to_array::<2>(new_shape)?;
                RankedTensor::D2(t.clone().reshape(s))
            }
            (3, RankedTensor::D3(t)) => {
                let s = shape_to_array::<3>(new_shape)?;
                RankedTensor::D3(t.clone().reshape(s))
            }
            (4, RankedTensor::D4(t)) => {
                let s = shape_to_array::<4>(new_shape)?;
                RankedTensor::D4(t.clone().reshape(s))
            }
            (5, RankedTensor::D5(t)) => {
                let s = shape_to_array::<5>(new_shape)?;
                RankedTensor::D5(t.clone().reshape(s))
            }
            (6, RankedTensor::D6(t)) => {
                let s = shape_to_array::<6>(new_shape)?;
                RankedTensor::D6(t.clone().reshape(s))
            }
            (7, RankedTensor::D7(t)) => {
                let s = shape_to_array::<7>(new_shape)?;
                RankedTensor::D7(t.clone().reshape(s))
            }
            (8, RankedTensor::D8(t)) => {
                let s = shape_to_array::<8>(new_shape)?;
                RankedTensor::D8(t.clone().reshape(s))
            }
            (other, _) => {
                return Err(burn_err(format!("reshape_to unsupported rank {other}")));
            }
        };
        Ok(Self {
            shape: new_shape.to_vec(),
            inner,
        })
    }

    pub fn permute_dims(&self, perm: &[usize]) -> Result<Self, GraphError> {
        if perm.len() != self.rank() {
            return Err(burn_err(format!(
                "permute length {} != rank {}",
                perm.len(),
                self.rank()
            )));
        }
        if perm.is_empty() {
            return Ok(self.clone());
        }
        let out_shape: Vec<usize> = perm.iter().map(|&i| self.shape[i]).collect();
        let inner = match &self.inner {
            RankedTensor::Scalar(t) | RankedTensor::D1(t) => {
                let axes: [usize; 1] = perm
                    .try_into()
                    .map_err(|_| burn_err("permute expected 1 axis".to_string()))?;
                RankedTensor::D1(t.clone().permute(axes))
            }
            RankedTensor::D2(t) => {
                let axes: [usize; 2] = perm
                    .try_into()
                    .map_err(|_| burn_err("permute expected 2 axes".to_string()))?;
                RankedTensor::D2(t.clone().permute(axes))
            }
            RankedTensor::D3(t) => {
                let axes: [usize; 3] = perm
                    .try_into()
                    .map_err(|_| burn_err("permute expected 3 axes".to_string()))?;
                RankedTensor::D3(t.clone().permute(axes))
            }
            RankedTensor::D4(t) => {
                let axes: [usize; 4] = perm
                    .try_into()
                    .map_err(|_| burn_err("permute expected 4 axes".to_string()))?;
                RankedTensor::D4(t.clone().permute(axes))
            }
            RankedTensor::D5(t) => {
                let axes: [usize; 5] = perm
                    .try_into()
                    .map_err(|_| burn_err("permute expected 5 axes".to_string()))?;
                RankedTensor::D5(t.clone().permute(axes))
            }
            RankedTensor::D6(t) => {
                let axes: [usize; 6] = perm
                    .try_into()
                    .map_err(|_| burn_err("permute expected 6 axes".to_string()))?;
                RankedTensor::D6(t.clone().permute(axes))
            }
            RankedTensor::D7(t) => {
                let axes: [usize; 7] = perm
                    .try_into()
                    .map_err(|_| burn_err("permute expected 7 axes".to_string()))?;
                RankedTensor::D7(t.clone().permute(axes))
            }
            RankedTensor::D8(t) => {
                let axes: [usize; 8] = perm
                    .try_into()
                    .map_err(|_| burn_err("permute expected 8 axes".to_string()))?;
                RankedTensor::D8(t.clone().permute(axes))
            }
        };
        Ok(Self {
            shape: out_shape,
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
        if dim_a >= self.rank() || dim_b >= self.rank() {
            return Err(burn_err(format!(
                "swap_dims_axes out of range for rank {}",
                self.rank()
            )));
        }
        let mut out_shape = self.shape.clone();
        out_shape.swap(dim_a, dim_b);
        let inner = match &self.inner {
            RankedTensor::Scalar(t) | RankedTensor::D1(t) => {
                RankedTensor::D1(t.clone().swap_dims(dim_a, dim_b))
            }
            RankedTensor::D2(t) => RankedTensor::D2(t.clone().swap_dims(dim_a, dim_b)),
            RankedTensor::D3(t) => RankedTensor::D3(t.clone().swap_dims(dim_a, dim_b)),
            RankedTensor::D4(t) => RankedTensor::D4(t.clone().swap_dims(dim_a, dim_b)),
            RankedTensor::D5(t) => RankedTensor::D5(t.clone().swap_dims(dim_a, dim_b)),
            RankedTensor::D6(t) => RankedTensor::D6(t.clone().swap_dims(dim_a, dim_b)),
            RankedTensor::D7(t) => RankedTensor::D7(t.clone().swap_dims(dim_a, dim_b)),
            RankedTensor::D8(t) => RankedTensor::D8(t.clone().swap_dims(dim_a, dim_b)),
        };
        Ok(Self {
            shape: out_shape,
            inner,
        })
    }

    /// WebNN `slice`: `starts`, element counts `sizes`, optional per-axis `strides` (default 1).
    pub fn slice_dims(
        &self,
        starts: &[u32],
        sizes: &[usize],
        strides: &[u32],
    ) -> Result<Self, GraphError> {
        if self.rank() == 0 {
            return Ok(self.clone());
        }
        if starts.len() != self.rank() || sizes.len() != self.rank() {
            return Err(burn_err(format!(
                "slice starts/sizes rank mismatch with input rank {}",
                self.rank()
            )));
        }
        let steps: Vec<isize> = if strides.is_empty() {
            vec![1; self.rank()]
        } else {
            strides.iter().map(|&s| s.max(1) as isize).collect()
        };
        let specs: Vec<Slice> = starts
            .iter()
            .zip(sizes.iter())
            .zip(steps.iter())
            .map(|((&start, &size), &step)| {
                let start = start as isize;
                let end = start + (size as isize) * step;
                Slice::new(start, Some(end), step)
            })
            .collect();
        let out_shape: Vec<usize> = sizes
            .iter()
            .zip(steps.iter())
            .map(|(&sz, &st)| {
                if st == 1 {
                    sz
                } else {
                    (sz + st as usize - 1) / st as usize
                }
            })
            .collect();
        let inner = match &self.inner {
            RankedTensor::Scalar(t) | RankedTensor::D1(t) => {
                let axes: [Slice; 1] = specs
                    .try_into()
                    .map_err(|_| burn_err("slice expected 1 axis".to_string()))?;
                RankedTensor::D1(t.clone().slice(axes))
            }
            RankedTensor::D2(t) => {
                let axes: [Slice; 2] = specs
                    .try_into()
                    .map_err(|_| burn_err("slice expected 2 axes".to_string()))?;
                RankedTensor::D2(t.clone().slice(axes))
            }
            RankedTensor::D3(t) => {
                let axes: [Slice; 3] = specs
                    .try_into()
                    .map_err(|_| burn_err("slice expected 3 axes".to_string()))?;
                RankedTensor::D3(t.clone().slice(axes))
            }
            RankedTensor::D4(t) => {
                let axes: [Slice; 4] = specs
                    .try_into()
                    .map_err(|_| burn_err("slice expected 4 axes".to_string()))?;
                RankedTensor::D4(t.clone().slice(axes))
            }
            RankedTensor::D5(t) => {
                let axes: [Slice; 5] = specs
                    .try_into()
                    .map_err(|_| burn_err("slice expected 5 axes".to_string()))?;
                RankedTensor::D5(t.clone().slice(axes))
            }
            RankedTensor::D6(t) => {
                let axes: [Slice; 6] = specs
                    .try_into()
                    .map_err(|_| burn_err("slice expected 6 axes".to_string()))?;
                RankedTensor::D6(t.clone().slice(axes))
            }
            RankedTensor::D7(t) => {
                let axes: [Slice; 7] = specs
                    .try_into()
                    .map_err(|_| burn_err("slice expected 7 axes".to_string()))?;
                RankedTensor::D7(t.clone().slice(axes))
            }
            RankedTensor::D8(t) => {
                let axes: [Slice; 8] = specs
                    .try_into()
                    .map_err(|_| burn_err("slice expected 8 axes".to_string()))?;
                RankedTensor::D8(t.clone().slice(axes))
            }
        };
        Ok(Self {
            shape: out_shape,
            inner,
        })
    }

    pub fn square(&self) -> Result<Self, GraphError> {
        let inner = match &self.inner {
            RankedTensor::Scalar(t) | RankedTensor::D1(t) => {
                let t = t.clone();
                RankedTensor::D1(t.clone() * t)
            }
            RankedTensor::D2(t) => {
                let t = t.clone();
                RankedTensor::D2(t.clone() * t)
            }
            RankedTensor::D3(t) => {
                let t = t.clone();
                RankedTensor::D3(t.clone() * t)
            }
            RankedTensor::D4(t) => {
                let t = t.clone();
                RankedTensor::D4(t.clone() * t)
            }
            RankedTensor::D5(t) => {
                let t = t.clone();
                RankedTensor::D5(t.clone() * t)
            }
            RankedTensor::D6(t) => {
                let t = t.clone();
                RankedTensor::D6(t.clone() * t)
            }
            RankedTensor::D7(t) => {
                let t = t.clone();
                RankedTensor::D7(t.clone() * t)
            }
            RankedTensor::D8(t) => {
                let t = t.clone();
                RankedTensor::D8(t.clone() * t)
            }
        };
        Ok(Self {
            shape: self.shape.clone(),
            inner,
        })
    }

    pub fn mul_scalar(&self, factor: f32) -> Result<Self, GraphError> {
        let inner = match &self.inner {
            RankedTensor::Scalar(t) | RankedTensor::D1(t) => {
                RankedTensor::D1(t.clone().mul_scalar(factor))
            }
            RankedTensor::D2(t) => RankedTensor::D2(t.clone().mul_scalar(factor)),
            RankedTensor::D3(t) => RankedTensor::D3(t.clone().mul_scalar(factor)),
            RankedTensor::D4(t) => RankedTensor::D4(t.clone().mul_scalar(factor)),
            RankedTensor::D5(t) => RankedTensor::D5(t.clone().mul_scalar(factor)),
            RankedTensor::D6(t) => RankedTensor::D6(t.clone().mul_scalar(factor)),
            RankedTensor::D7(t) => RankedTensor::D7(t.clone().mul_scalar(factor)),
            RankedTensor::D8(t) => RankedTensor::D8(t.clone().mul_scalar(factor)),
        };
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
        let inner = match (&self.inner, &rhs.inner) {
            (RankedTensor::D1(a), RankedTensor::D1(b)) => {
                RankedTensor::D1(a.clone() + b.clone().mul_scalar(beta))
            }
            (RankedTensor::D2(a), RankedTensor::D2(b)) => {
                RankedTensor::D2(a.clone() + b.clone().mul_scalar(beta))
            }
            (RankedTensor::D3(a), RankedTensor::D3(b)) => {
                RankedTensor::D3(a.clone() + b.clone().mul_scalar(beta))
            }
            (RankedTensor::D4(a), RankedTensor::D4(b)) => {
                RankedTensor::D4(a.clone() + b.clone().mul_scalar(beta))
            }
            (RankedTensor::D5(a), RankedTensor::D5(b)) => {
                RankedTensor::D5(a.clone() + b.clone().mul_scalar(beta))
            }
            (RankedTensor::D6(a), RankedTensor::D6(b)) => {
                RankedTensor::D6(a.clone() + b.clone().mul_scalar(beta))
            }
            (RankedTensor::D7(a), RankedTensor::D7(b)) => {
                RankedTensor::D7(a.clone() + b.clone().mul_scalar(beta))
            }
            (RankedTensor::D8(a), RankedTensor::D8(b)) => {
                RankedTensor::D8(a.clone() + b.clone().mul_scalar(beta))
            }
            _ => {
                return Err(burn_err(format!(
                    "add_scaled rank mismatch {:?} vs {:?}",
                    self.shape, other.shape
                )));
            }
        };
        Ok(Self {
            shape: self.shape.clone(),
            inner,
        })
    }

    pub fn concat_same_rank(tensors: &[Self], axis: usize) -> Result<Self, GraphError> {
        if tensors.is_empty() {
            return Err(burn_err("concat requires at least one tensor".to_string()));
        }
        let rank = tensors[0].rank();
        if axis >= rank.max(1) {
            return Err(burn_err(format!(
                "concat axis {axis} out of range for rank {rank}"
            )));
        }
        for t in tensors.iter().skip(1) {
            if t.rank() != rank {
                return Err(burn_err(
                    "concat inputs must have the same rank".to_string(),
                ));
            }
            for (i, (&a, &b)) in t.shape.iter().zip(tensors[0].shape.iter()).enumerate() {
                if i != axis && a != b {
                    return Err(burn_err(format!(
                        "concat shape mismatch at axis {i}: {a} vs {b}"
                    )));
                }
            }
        }
        let mut out_shape = tensors[0].shape.clone();
        out_shape[axis] = tensors.iter().map(|t| t.shape[axis]).sum();
        let inner = match rank {
            1 => {
                let parts: Vec<Tensor<B, 1>> = tensors
                    .iter()
                    .map(|t| scalar_or_d1(&t.inner))
                    .collect::<Result<_, _>>()?;
                RankedTensor::D1(Tensor::cat(parts, axis))
            }
            2 => {
                let parts: Vec<Tensor<B, 2>> = tensors
                    .iter()
                    .map(|t| match &t.inner {
                        RankedTensor::D2(x) => Ok(x.clone()),
                        _ => Err(burn_err("concat rank-2 type mismatch".to_string())),
                    })
                    .collect::<Result<_, _>>()?;
                RankedTensor::D2(Tensor::cat(parts, axis))
            }
            3 => {
                let parts: Vec<Tensor<B, 3>> = tensors
                    .iter()
                    .map(|t| match &t.inner {
                        RankedTensor::D3(x) => Ok(x.clone()),
                        _ => Err(burn_err("concat rank-3 type mismatch".to_string())),
                    })
                    .collect::<Result<_, _>>()?;
                RankedTensor::D3(Tensor::cat(parts, axis))
            }
            4 => {
                let parts: Vec<Tensor<B, 4>> = tensors
                    .iter()
                    .map(|t| match &t.inner {
                        RankedTensor::D4(x) => Ok(x.clone()),
                        _ => Err(burn_err("concat rank-4 type mismatch".to_string())),
                    })
                    .collect::<Result<_, _>>()?;
                RankedTensor::D4(Tensor::cat(parts, axis))
            }
            5 => {
                let parts: Vec<Tensor<B, 5>> = tensors
                    .iter()
                    .map(|t| match &t.inner {
                        RankedTensor::D5(x) => Ok(x.clone()),
                        _ => Err(burn_err("concat rank-5 type mismatch".to_string())),
                    })
                    .collect::<Result<_, _>>()?;
                RankedTensor::D5(Tensor::cat(parts, axis))
            }
            6 => {
                let parts: Vec<Tensor<B, 6>> = tensors
                    .iter()
                    .map(|t| match &t.inner {
                        RankedTensor::D6(x) => Ok(x.clone()),
                        _ => Err(burn_err("concat rank-6 type mismatch".to_string())),
                    })
                    .collect::<Result<_, _>>()?;
                RankedTensor::D6(Tensor::cat(parts, axis))
            }
            7 => {
                let parts: Vec<Tensor<B, 7>> = tensors
                    .iter()
                    .map(|t| match &t.inner {
                        RankedTensor::D7(x) => Ok(x.clone()),
                        _ => Err(burn_err("concat rank-7 type mismatch".to_string())),
                    })
                    .collect::<Result<_, _>>()?;
                RankedTensor::D7(Tensor::cat(parts, axis))
            }
            8 => {
                let parts: Vec<Tensor<B, 8>> = tensors
                    .iter()
                    .map(|t| match &t.inner {
                        RankedTensor::D8(x) => Ok(x.clone()),
                        _ => Err(burn_err("concat rank-8 type mismatch".to_string())),
                    })
                    .collect::<Result<_, _>>()?;
                RankedTensor::D8(Tensor::cat(parts, axis))
            }
            other => return Err(burn_err(format!("concat unsupported rank {other}"))),
        };
        Ok(Self {
            shape: out_shape,
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
        let rank = out_shape.len();
        let inner = match (rank, &self.inner, &other.inner) {
            (2, RankedTensor::D2(a), RankedTensor::D2(b)) => {
                RankedTensor::D2(a.clone().matmul(b.clone()))
            }
            (3, RankedTensor::D3(a), RankedTensor::D3(b)) => {
                RankedTensor::D3(a.clone().matmul(b.clone()))
            }
            (4, RankedTensor::D4(a), RankedTensor::D4(b)) => {
                RankedTensor::D4(a.clone().matmul(b.clone()))
            }
            (5, RankedTensor::D5(a), RankedTensor::D5(b)) => {
                RankedTensor::D5(a.clone().matmul(b.clone()))
            }
            (6, RankedTensor::D6(a), RankedTensor::D6(b)) => {
                RankedTensor::D6(a.clone().matmul(b.clone()))
            }
            (7, RankedTensor::D7(a), RankedTensor::D7(b)) => {
                RankedTensor::D7(a.clone().matmul(b.clone()))
            }
            (8, RankedTensor::D8(a), RankedTensor::D8(b)) => {
                RankedTensor::D8(a.clone().matmul(b.clone()))
            }
            _ => {
                return Err(burn_err(format!(
                    "matmul unsupported ranks/shapes {:?} @ {:?}",
                    self.shape, other.shape
                )));
            }
        };
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
        if self.numel() == 1 {
            if let Ok(expanded) = self.expand_singleton_on_device(target) {
                return Ok(expanded);
            }
        }
        let host = self.to_host_array()?;
        let expanded = broadcast_to(&host, target)?;
        Self::from_f32_data(expanded.shape, expanded.data, device)
    }

    fn expand_singleton_on_device(&self, target: &[usize]) -> Result<Self, GraphError> {
        let rank = target.len().max(1);
        let padded = pad_leading_ones(&self.shape, rank);
        let inner = match rank {
            1 => {
                let t = scalar_or_d1(&self.inner)?;
                let s = shape_to_array::<1>(target)?;
                RankedTensor::D1(t.reshape([1]).expand(s))
            }
            2 => {
                let t = scalar_or_d1(&self.inner)?;
                let s = shape_to_array::<2>(&padded)?;
                let t = t.reshape(s);
                let out_s = shape_to_array::<2>(target)?;
                RankedTensor::D2(t.expand(out_s))
            }
            3 => {
                let t = scalar_or_d1(&self.inner)?;
                let s = shape_to_array::<3>(&padded)?;
                let t = t.reshape(s);
                let out_s = shape_to_array::<3>(target)?;
                RankedTensor::D3(t.expand(out_s))
            }
            4 => {
                let t = scalar_or_d1(&self.inner)?;
                let s = shape_to_array::<4>(&padded)?;
                let t = t.reshape(s);
                let out_s = shape_to_array::<4>(target)?;
                RankedTensor::D4(t.expand(out_s))
            }
            5 => {
                let t = scalar_or_d1(&self.inner)?;
                let s = shape_to_array::<5>(&padded)?;
                let t = t.reshape(s);
                let out_s = shape_to_array::<5>(target)?;
                RankedTensor::D5(t.expand(out_s))
            }
            6 => {
                let t = scalar_or_d1(&self.inner)?;
                let s = shape_to_array::<6>(&padded)?;
                let t = t.reshape(s);
                let out_s = shape_to_array::<6>(target)?;
                RankedTensor::D6(t.expand(out_s))
            }
            7 => {
                let t = scalar_or_d1(&self.inner)?;
                let s = shape_to_array::<7>(&padded)?;
                let t = t.reshape(s);
                let out_s = shape_to_array::<7>(target)?;
                RankedTensor::D7(t.expand(out_s))
            }
            8 => {
                let t = scalar_or_d1(&self.inner)?;
                let s = shape_to_array::<8>(&padded)?;
                let t = t.reshape(s);
                let out_s = shape_to_array::<8>(target)?;
                RankedTensor::D8(t.expand(out_s))
            }
            other => return Err(burn_err(format!("unsupported expand rank {other}"))),
        };
        Ok(Self {
            shape: target.to_vec(),
            inner,
        })
    }

    fn binary_same_shape(
        &self,
        other: &Self,
        op: DeviceBinaryOp,
    ) -> Result<Option<Self>, GraphError> {
        if self.shape != other.shape {
            return Ok(None);
        }
        Ok(Some(self.binary_same_shape_force(
            other,
            op,
            &self.shape,
        )?))
    }

    fn binary_same_shape_force(
        &self,
        other: &Self,
        op: DeviceBinaryOp,
        out_shape: &[usize],
    ) -> Result<Self, GraphError> {
        let inner = match (&self.inner, &other.inner) {
            (RankedTensor::Scalar(a), RankedTensor::Scalar(b)) => {
                RankedTensor::Scalar(apply_binary_scalar(a, b, op))
            }
            (RankedTensor::D1(a), RankedTensor::D1(b)) => RankedTensor::D1(apply_binary(a, b, op)),
            (RankedTensor::D2(a), RankedTensor::D2(b)) => RankedTensor::D2(apply_binary(a, b, op)),
            (RankedTensor::D3(a), RankedTensor::D3(b)) => RankedTensor::D3(apply_binary(a, b, op)),
            (RankedTensor::D4(a), RankedTensor::D4(b)) => RankedTensor::D4(apply_binary(a, b, op)),
            (RankedTensor::D5(a), RankedTensor::D5(b)) => RankedTensor::D5(apply_binary(a, b, op)),
            (RankedTensor::D6(a), RankedTensor::D6(b)) => RankedTensor::D6(apply_binary(a, b, op)),
            (RankedTensor::D7(a), RankedTensor::D7(b)) => RankedTensor::D7(apply_binary(a, b, op)),
            (RankedTensor::D8(a), RankedTensor::D8(b)) => RankedTensor::D8(apply_binary(a, b, op)),
            _ => {
                return Err(burn_err(format!(
                    "binary op rank mismatch {:?} vs {:?}",
                    self.shape, other.shape
                )));
            }
        };
        Ok(Self {
            shape: out_shape.to_vec(),
            inner,
        })
    }

    fn compare_same_shape(
        &self,
        other: &Self,
        op: DeviceCompareOp,
        out_shape: &[usize],
    ) -> Result<Self, GraphError> {
        let inner = match (&self.inner, &other.inner) {
            (RankedTensor::Scalar(a), RankedTensor::Scalar(b)) => {
                RankedTensor::Scalar(apply_compare_scalar(a, b, op))
            }
            (RankedTensor::D1(a), RankedTensor::D1(b)) => RankedTensor::D1(apply_compare(a, b, op)),
            (RankedTensor::D2(a), RankedTensor::D2(b)) => RankedTensor::D2(apply_compare(a, b, op)),
            (RankedTensor::D3(a), RankedTensor::D3(b)) => RankedTensor::D3(apply_compare(a, b, op)),
            (RankedTensor::D4(a), RankedTensor::D4(b)) => RankedTensor::D4(apply_compare(a, b, op)),
            (RankedTensor::D5(a), RankedTensor::D5(b)) => RankedTensor::D5(apply_compare(a, b, op)),
            (RankedTensor::D6(a), RankedTensor::D6(b)) => RankedTensor::D6(apply_compare(a, b, op)),
            (RankedTensor::D7(a), RankedTensor::D7(b)) => RankedTensor::D7(apply_compare(a, b, op)),
            (RankedTensor::D8(a), RankedTensor::D8(b)) => RankedTensor::D8(apply_compare(a, b, op)),
            _ => {
                return Err(burn_err(format!(
                    "compare op rank mismatch {:?} vs {:?}",
                    self.shape, other.shape
                )));
            }
        };
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

fn matmul_broadcast_operand_shapes(
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

fn apply_binary_scalar<B: Backend>(
    a: &Tensor<B, 1>,
    b: &Tensor<B, 1>,
    op: DeviceBinaryOp,
) -> Tensor<B, 1> {
    apply_binary(a, b, op)
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
        DeviceBinaryOp::Pow => a.powf(b),
        DeviceBinaryOp::Min => a.min_pair(b),
        DeviceBinaryOp::Max => a.max_pair(b),
    }
}

fn apply_compare<B: Backend, const D: usize>(
    a: &Tensor<B, D>,
    b: &Tensor<B, D>,
    op: DeviceCompareOp,
) -> Tensor<B, D> {
    let a = a.clone();
    let b = b.clone();
    let mask = match op {
        DeviceCompareOp::Equal => a.equal(b),
        DeviceCompareOp::Greater => a.greater(b),
        DeviceCompareOp::GreaterOrEqual => a.greater_equal(b),
        DeviceCompareOp::Lesser => a.lower(b),
        DeviceCompareOp::LesserOrEqual => a.lower_equal(b),
    };
    mask.float()
}

fn apply_compare_scalar<B: Backend>(
    a: &Tensor<B, 1>,
    b: &Tensor<B, 1>,
    op: DeviceCompareOp,
) -> Tensor<B, 1> {
    apply_compare(a, b, op)
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
        UnaryDeviceOp::Clamp { min, max } => t.clamp(min, max),
        UnaryDeviceOp::Softmax { axis } => softmax(t, axis),
    }
}

#[cfg(all(test, feature = "burn-runtime-cpu"))]
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
}

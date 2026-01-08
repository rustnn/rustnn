//! Device-resident tensor abstractions for zero-copy execution
//!
//! This module provides unified representation for host and device tensors,
//! enabling persistent GPU/NPU tensor storage across inference steps to eliminate
//! host round-trips for iterative GenAI workloads (e.g., KV cache).

use crate::error::GraphError;
use crate::graph::DataType;

/// Unified representation for host and device tensors
#[derive(Debug)]
pub enum TensorValue {
    /// Host-resident tensor stored in CPU memory
    Host(HostTensor),
    /// Device-resident tensor stored in GPU/NPU memory
    Device(DeviceTensorHandle),
}

/// Host-resident tensor stored in CPU memory
#[derive(Debug, Clone)]
pub struct HostTensor {
    /// Tensor data stored as f32 (will expand to support other types later)
    pub data: Vec<f32>,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DataType,
}

impl HostTensor {
    /// Create a new host tensor with the given shape and data type
    pub fn new(shape: Vec<usize>, dtype: DataType) -> Self {
        let total_elements: usize = shape.iter().product();
        let data = vec![0.0f32; total_elements];
        Self { data, shape, dtype }
    }

    /// Create a host tensor from existing data
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>, dtype: DataType) -> Self {
        Self { data, shape, dtype }
    }

    /// Get the total number of elements
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Handle to a device-resident tensor with backend-specific implementation
#[derive(Debug)]
pub struct DeviceTensorHandle {
    /// Backend-specific tensor implementation
    pub inner: Box<dyn DeviceTensorBackend>,
    /// Data type
    pub dtype: DataType,
    /// Tensor shape
    pub shape: Vec<usize>,
}

impl DeviceTensorHandle {
    /// Create a new device tensor handle
    pub fn new(inner: Box<dyn DeviceTensorBackend>) -> Self {
        let dtype = inner.dtype();
        let shape = inner.shape().to_vec();
        Self {
            inner,
            dtype,
            shape,
        }
    }

    /// Get the device kind (CPU, CUDA, CoreML, etc.)
    pub fn device_kind(&self) -> DeviceKind {
        self.inner.device_kind()
    }

    /// Get the backend kind (OnnxCpu, OnnxGpu, TensorRT, etc.)
    pub fn backend_kind(&self) -> BackendKind {
        self.inner.backend_kind()
    }
}

/// Trait for backend-specific device tensor implementations
///
/// Each backend (ONNX Runtime, CoreML, TensorRT) implements this trait
/// to provide device-resident tensor storage and host transfer operations.
pub trait DeviceTensorBackend: Send + Sync + std::fmt::Debug {
    /// Get the data type of the tensor
    fn dtype(&self) -> DataType;

    /// Get the shape of the tensor
    fn shape(&self) -> &[usize];

    /// Get the device kind where this tensor resides
    fn device_kind(&self) -> DeviceKind;

    /// Get the backend that created this tensor
    fn backend_kind(&self) -> BackendKind;

    /// Read tensor data from device to host
    ///
    /// This performs a device-to-host memory transfer.
    /// Returns data as Vec<f32> (will expand to support other types later).
    fn read_to_host(&self) -> Result<Vec<f32>, GraphError>;

    /// Write tensor data from host to device
    ///
    /// This performs a host-to-device memory transfer.
    /// Accepts data as &[f32] (will expand to support other types later).
    fn write_from_host(&mut self, data: &[f32]) -> Result<(), GraphError>;

    /// Get a reference to self as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Device kind where a tensor resides
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// CPU device
    Cpu,
    /// NVIDIA CUDA GPU
    Cuda,
    /// DirectML GPU (Windows)
    DirectML,
    /// Apple CoreML (GPU/Neural Engine)
    CoreML,
}

/// Backend that created a device tensor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// ONNX Runtime CPU backend
    OnnxCpu,
    /// ONNX Runtime GPU backend
    OnnxGpu,
    /// Apple CoreML backend
    CoreML,
    /// NVIDIA TensorRT backend
    TensorRT,
}

impl std::fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceKind::Cpu => write!(f, "cpu"),
            DeviceKind::Cuda => write!(f, "cuda"),
            DeviceKind::DirectML => write!(f, "directml"),
            DeviceKind::CoreML => write!(f, "coreml"),
        }
    }
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendKind::OnnxCpu => write!(f, "onnx_cpu"),
            BackendKind::OnnxGpu => write!(f, "onnx_gpu"),
            BackendKind::CoreML => write!(f, "coreml"),
            BackendKind::TensorRT => write!(f, "tensorrt"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_tensor_creation() {
        let tensor = HostTensor::new(vec![2, 3], DataType::Float32);
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.dtype, DataType::Float32);
        assert_eq!(tensor.element_count(), 6);
        assert_eq!(tensor.data.len(), 6);
    }

    #[test]
    fn test_host_tensor_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = HostTensor::from_data(data.clone(), vec![2, 2], DataType::Float32);
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, data);
    }

    #[test]
    fn test_device_kind_display() {
        assert_eq!(DeviceKind::Cpu.to_string(), "cpu");
        assert_eq!(DeviceKind::Cuda.to_string(), "cuda");
        assert_eq!(DeviceKind::CoreML.to_string(), "coreml");
    }

    #[test]
    fn test_backend_kind_display() {
        assert_eq!(BackendKind::OnnxCpu.to_string(), "onnx_cpu");
        assert_eq!(BackendKind::OnnxGpu.to_string(), "onnx_gpu");
        assert_eq!(BackendKind::CoreML.to_string(), "coreml");
        assert_eq!(BackendKind::TensorRT.to_string(), "tensorrt");
    }
}

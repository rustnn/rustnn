use crate::graph::{DataType, OperandDescriptor};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

/// MLTensor - Represents a tensor with data storage
///
/// MLTensor is used for explicit tensor management in WebNN, allowing
/// pre-allocation of input/output buffers and explicit data transfer.
#[pyclass(name = "MLTensor")]
#[derive(Clone)]
pub struct PyMLTensor {
    pub(crate) descriptor: OperandDescriptor,
    pub(crate) data: Arc<Mutex<Vec<f32>>>,
}

impl PyMLTensor {
    /// Create a new tensor with the given descriptor
    pub fn new(descriptor: OperandDescriptor) -> Self {
        let total_elements: usize = descriptor.shape.iter().map(|&d| d as usize).product();
        let data = vec![0.0f32; total_elements];

        Self {
            descriptor,
            data: Arc::new(Mutex::new(data)),
        }
    }

    /// Create a tensor from existing data
    pub fn from_data(descriptor: OperandDescriptor, data: Vec<f32>) -> Self {
        Self {
            descriptor,
            data: Arc::new(Mutex::new(data)),
        }
    }

    /// Get the data as a vector
    pub fn get_data(&self) -> Vec<f32> {
        self.data.lock().unwrap().clone()
    }

    /// Set the data from a vector
    pub fn set_data(&self, data: Vec<f32>) -> PyResult<()> {
        let expected_size: usize = self.descriptor.shape.iter().map(|&d| d as usize).product();
        if data.len() != expected_size {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Data size mismatch: expected {} elements, got {}",
                expected_size,
                data.len()
            )));
        }
        *self.data.lock().unwrap() = data;
        Ok(())
    }
}

#[pymethods]
impl PyMLTensor {
    /// Get the data type of the tensor
    #[getter]
    fn data_type(&self) -> String {
        match self.descriptor.data_type {
            DataType::Float32 => "float32".to_string(),
            DataType::Float16 => "float16".to_string(),
            DataType::Int32 => "int32".to_string(),
            DataType::Uint32 => "uint32".to_string(),
            DataType::Int8 => "int8".to_string(),
            DataType::Uint8 => "uint8".to_string(),
        }
    }

    /// Get the shape of the tensor
    #[getter]
    fn shape(&self) -> Vec<u32> {
        self.descriptor.shape.clone()
    }

    /// Get the number of elements in the tensor
    #[getter]
    fn size(&self) -> usize {
        self.descriptor.shape.iter().map(|&d| d as usize).product()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "MLTensor(shape={:?}, dtype={})",
            self.descriptor.shape,
            self.data_type()
        )
    }
}

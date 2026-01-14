//! Operand representation for WebNN computational graphs
//!
//! PyO3 macros generate unsafe code that triggers unsafe_op_in_unsafe_fn warnings.
//! This is expected behavior from the macro-generated code.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::graph::{DataType, OperandDescriptor, OperandKind};
use pyo3::prelude::*;

/// Represents an operand in the computational graph
#[pyclass(name = "MLOperand")]
#[derive(Clone)]
pub struct PyMLOperand {
    pub(crate) id: u32,
    pub(crate) descriptor: OperandDescriptor,
    #[allow(dead_code)]
    pub(crate) kind: OperandKind,
    pub(crate) name: Option<String>,
}

#[pymethods]
impl PyMLOperand {
    /// Get the operand's data type
    #[getter]
    fn data_type(&self) -> String {
        match self.descriptor.data_type {
            DataType::Int4 => "int4".to_string(),
            DataType::Uint4 => "uint4".to_string(),
            DataType::Float32 => "float32".to_string(),
            DataType::Float16 => "float16".to_string(),
            DataType::Int32 => "int32".to_string(),
            DataType::Uint32 => "uint32".to_string(),
            DataType::Int8 => "int8".to_string(),
            DataType::Uint8 => "uint8".to_string(),
            DataType::Int64 => "int64".to_string(),
            DataType::Uint64 => "uint64".to_string(),
        }
    }

    /// Get the operand's shape
    #[getter]
    fn shape(&self) -> Vec<u32> {
        self.descriptor.shape.clone()
    }

    /// Get the operand's name (if any)
    #[getter]
    fn name(&self) -> Option<String> {
        self.name.clone()
    }

    fn __repr__(&self) -> String {
        let name = self.name.as_deref().unwrap_or("unnamed");
        let dtype = self.data_type();
        let shape = format!("{:?}", self.shape());
        format!(
            "MLOperand(name='{}', dtype='{}', shape={})",
            name, dtype, shape
        )
    }
}

impl PyMLOperand {
    pub fn new(
        id: u32,
        descriptor: OperandDescriptor,
        kind: OperandKind,
        name: Option<String>,
    ) -> Self {
        Self {
            id,
            descriptor,
            kind,
            name,
        }
    }
}

/// Parse data type string to DataType enum
pub fn parse_data_type(dtype: &str) -> PyResult<DataType> {
    match dtype {
        "int4" => Ok(DataType::Int4),
        "uint4" => Ok(DataType::Uint4),
        "float32" => Ok(DataType::Float32),
        "float16" => Ok(DataType::Float16),
        "int32" => Ok(DataType::Int32),
        "uint32" => Ok(DataType::Uint32),
        "int8" => Ok(DataType::Int8),
        "uint8" => Ok(DataType::Uint8),
        "int64" => Ok(DataType::Int64),
        "uint64" => Ok(DataType::Uint64),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported data type: {}",
            dtype
        ))),
    }
}

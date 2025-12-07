//! Python bindings for the WebNN API

use pyo3::prelude::*;

mod context;
mod graph;
mod graph_builder;
mod operand;
mod tensor;

pub use context::{PyML, PyMLContext};
pub use graph::PyMLGraph;
pub use graph_builder::PyMLGraphBuilder;
pub use operand::PyMLOperand;
pub use tensor::PyMLTensor;

/// WebNN Python module
#[pymodule]
fn _rustnn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyML>()?;
    m.add_class::<PyMLContext>()?;
    m.add_class::<PyMLGraphBuilder>()?;
    m.add_class::<PyMLOperand>()?;
    m.add_class::<PyMLGraph>()?;
    m.add_class::<PyMLTensor>()?;
    Ok(())
}

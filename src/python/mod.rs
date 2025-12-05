//! Python bindings for the WebNN API

use pyo3::prelude::*;

mod operand;
mod graph_builder;
mod context;
mod graph;

pub use operand::PyMLOperand;
pub use graph_builder::PyMLGraphBuilder;
pub use context::{PyML, PyMLContext};
pub use graph::PyMLGraph;

/// WebNN Python module
#[pymodule]
fn _rustnn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyML>()?;
    m.add_class::<PyMLContext>()?;
    m.add_class::<PyMLGraphBuilder>()?;
    m.add_class::<PyMLOperand>()?;
    m.add_class::<PyMLGraph>()?;
    Ok(())
}

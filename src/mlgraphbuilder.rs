use std::collections::HashMap;

use crate::error::Result;
use crate::mlcontext::{MLGraph, MLOperand};
use crate::{
    GraphInfo,
    mlcontext::{MLBackendBuilder, MLContext},
};

#[derive(Debug)]
pub struct MLGraphBuilder<'context> {
    backend: Box<dyn MLBackendBuilder<'context> + 'context>,

    ///[[hasBuilt]] of type boolean
    ///
    /// Whether MLGraphBuilder.build() has been called. Once built, the MLGraphBuilder can no longer create operators or compile MLGraphs.
    has_built: bool,
}

impl<'context> MLGraphBuilder<'context> {
    pub fn new(context: &'_ mut MLContext<'context>) -> Result<Self> {
        let backend = context.backend.create_builder()?;
        Ok(Self {
            backend,
            has_built: false,
        })
    }

    pub fn build_graph_info(&mut self, graph: &'context GraphInfo) -> Result<MLGraph<'context>> {
        self.backend.load_graph(graph)?;
        self.backend.build(&HashMap::new())
    }

    /*async*/
    pub fn build(&mut self, outputs: &HashMap<&str, MLOperand>) -> Result<MLGraph<'context>> {
        if self.has_built {
            panic!("Called MLGraphBuilder::build more than once on a MLGraph");
        }
        self.has_built = true;
        self.backend.build(outputs)
    }
}

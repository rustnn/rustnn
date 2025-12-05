"""
WebNN Python API

This package provides Python bindings for the WebNN (Web Neural Network) API,
allowing you to build, validate, and execute neural network graphs.

Example usage:
    >>> import webnn
    >>> ml = webnn.ML()
    >>> context = ml.create_context(device_type="cpu")
    >>> builder = context.create_graph_builder()
    >>>
    >>> # Build a simple graph
    >>> x = builder.input("x", [2, 3], "float32")
    >>> y = builder.input("y", [2, 3], "float32")
    >>> z = builder.add(x, y)
    >>> output = builder.relu(z)
    >>>
    >>> # Compile the graph
    >>> graph = builder.build({"output": output})
"""

from ._rustnn import (
    ML,
    MLContext,
    MLGraphBuilder,
    MLOperand,
    MLGraph,
)

__all__ = [
    "ML",
    "MLContext",
    "MLGraphBuilder",
    "MLOperand",
    "MLGraph",
]

__version__ = "0.1.0"

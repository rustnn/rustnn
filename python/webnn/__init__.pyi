"""Type stubs for webnn package"""

from typing import Dict, List, Optional, Union
import numpy as np
import numpy.typing as npt

class ML:
    """Entry point for the WebNN API"""

    def __init__(self) -> None: ...

    def create_context(
        self,
        device_type: str = "cpu",
        power_preference: str = "default"
    ) -> MLContext:
        """
        Create a new ML context

        Args:
            device_type: Device type ("cpu", "gpu", or "npu")
            power_preference: Power preference ("default", "high-performance", or "low-power")

        Returns:
            A new MLContext instance
        """
        ...

class MLContext:
    """Execution context for neural network graphs"""

    @property
    def device_type(self) -> str:
        """Get the device type"""
        ...

    @property
    def power_preference(self) -> str:
        """Get the power preference"""
        ...

    def create_graph_builder(self) -> MLGraphBuilder:
        """Create a graph builder for constructing computational graphs"""
        ...

    def compute(
        self,
        graph: MLGraph,
        inputs: Dict[str, npt.ArrayLike],
        outputs: Optional[Dict[str, npt.ArrayLike]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Execute the graph with given inputs

        Args:
            graph: The compiled MLGraph to execute
            inputs: Dictionary mapping input names to numpy arrays
            outputs: Optional pre-allocated output arrays

        Returns:
            Dictionary mapping output names to result numpy arrays
        """
        ...

    def convert_to_onnx(self, graph: MLGraph, output_path: str) -> None:
        """
        Convert graph to ONNX format

        Args:
            graph: The MLGraph to convert
            output_path: Path to save the ONNX model
        """
        ...

    def convert_to_coreml(self, graph: MLGraph, output_path: str) -> None:
        """
        Convert graph to CoreML format (macOS only)

        Args:
            graph: The MLGraph to convert
            output_path: Path to save the CoreML model
        """
        ...

class MLOperand:
    """Represents an operand in the computational graph"""

    @property
    def data_type(self) -> str:
        """Get the operand's data type"""
        ...

    @property
    def shape(self) -> List[int]:
        """Get the operand's shape"""
        ...

    @property
    def name(self) -> Optional[str]:
        """Get the operand's name"""
        ...

class MLGraphBuilder:
    """Builder for constructing WebNN computational graphs"""

    def __init__(self) -> None: ...

    def input(
        self,
        name: str,
        shape: List[int],
        data_type: str = "float32"
    ) -> MLOperand:
        """
        Create an input operand

        Args:
            name: Name of the input
            shape: List of dimensions
            data_type: Data type string (e.g., "float32")

        Returns:
            The created input operand
        """
        ...

    def constant(
        self,
        value: npt.ArrayLike,
        shape: Optional[List[int]] = None,
        data_type: Optional[str] = None
    ) -> MLOperand:
        """
        Create a constant operand from numpy array

        Args:
            value: NumPy array or Python list
            shape: Optional shape override
            data_type: Optional data type string

        Returns:
            The created constant operand
        """
        ...

    # Binary operations
    def add(self, a: MLOperand, b: MLOperand) -> MLOperand:
        """Element-wise addition"""
        ...

    def sub(self, a: MLOperand, b: MLOperand) -> MLOperand:
        """Element-wise subtraction"""
        ...

    def mul(self, a: MLOperand, b: MLOperand) -> MLOperand:
        """Element-wise multiplication"""
        ...

    def div(self, a: MLOperand, b: MLOperand) -> MLOperand:
        """Element-wise division"""
        ...

    def matmul(self, a: MLOperand, b: MLOperand) -> MLOperand:
        """Matrix multiplication"""
        ...

    # Unary operations
    def relu(self, x: MLOperand) -> MLOperand:
        """ReLU activation"""
        ...

    def sigmoid(self, x: MLOperand) -> MLOperand:
        """Sigmoid activation"""
        ...

    def tanh(self, x: MLOperand) -> MLOperand:
        """Tanh activation"""
        ...

    def softmax(self, x: MLOperand) -> MLOperand:
        """Softmax activation"""
        ...

    def reshape(self, x: MLOperand, new_shape: List[int]) -> MLOperand:
        """Reshape operation"""
        ...

    def build(self, outputs: Dict[str, MLOperand]) -> MLGraph:
        """
        Build and compile the computational graph

        Args:
            outputs: Dictionary mapping output names to MLOperand objects

        Returns:
            The compiled graph
        """
        ...

class MLGraph:
    """Compiled computational graph"""

    @property
    def operand_count(self) -> int:
        """Get the number of operands in the graph"""
        ...

    @property
    def operation_count(self) -> int:
        """Get the number of operations in the graph"""
        ...

    def get_input_names(self) -> List[str]:
        """Get list of input names"""
        ...

    def get_output_names(self) -> List[str]:
        """Get list of output names"""
        ...

"""Stub module for the PyO3 extension exports used by the Python wrappers."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt


class ML:
    def __init__(self) -> None: ...

    def create_context(
        self, device_type: str = "cpu", power_preference: str = "default"
    ) -> MLContext: ...


class MLContext:
    @property
    def power_preference(self) -> str: ...

    @property
    def accelerated(self) -> bool: ...

    def create_graph_builder(self) -> MLGraphBuilder: ...

    def compute(
        self,
        graph: MLGraph,
        inputs: Dict[str, npt.ArrayLike],
        outputs: Optional[Dict[str, npt.ArrayLike]] = None,
    ) -> Dict[str, np.ndarray]: ...

    def create_tensor(
        self,
        shape,
        data_type: str,
        readable: bool = True,
        writable: bool = True,
        exportable_to_gpu: bool = False,
    ) -> MLTensor: ...

    def dispatch(
        self, graph: MLGraph, inputs: Dict[str, MLTensor], outputs: Dict[str, MLTensor]
    ) -> None: ...

    def read_tensor(self, tensor: MLTensor) -> np.ndarray: ...

    def write_tensor(self, tensor: MLTensor, data: npt.ArrayLike) -> None: ...

    def op_support_limits(self) -> Dict[str, object]: ...


class MLOperand:
    @property
    def data_type(self) -> str: ...

    @property
    def shape(self) -> List[int]: ...

    @property
    def name(self) -> Optional[str]: ...


class MLGraphBuilder:
    def __init__(self) -> None: ...

    def build(self, outputs): ...


class MLGraph:
    @property
    def operand_count(self) -> int: ...

    @property
    def operation_count(self) -> int: ...

    def get_input_names(self) -> List[str]: ...

    def get_output_names(self) -> List[str]: ...


class MLTensor:
    pass


__all__ = [
    "ML",
    "MLContext",
    "MLGraphBuilder",
    "MLOperand",
    "MLGraph",
    "MLTensor",
]

#!/usr/bin/env python3
"""
Convert an ONNX model to a WebNN-like JSON graph using huningxin/onnx2webnn.

Usage:
  python scripts/convert_onnx_to_webnn.py input.onnx output.json

This script:
  1) Ensures a local virtualenv (.venv-webnn) with onnx2webnn installed.
  2) Invokes onnx2webnn to convert the input ONNX model.
  3) Writes the JSON graph to the requested output path.
"""
import subprocess
import sys
import venv
from pathlib import Path

import onnx
import numpy as np
import onnxruntime as ort

VENV_DIR = Path(".venv-webnn")
GIT_URL = "https://github.com/huningxin/onnx2webnn.git"


def ensure_venv() -> Path:
    if not VENV_DIR.exists():
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(VENV_DIR)
        python_bin = VENV_DIR / "bin" / "python"
        subprocess.run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(python_bin), "-m", "pip", "install", "numpy", "onnx"], check=True)
        subprocess.run([str(python_bin), "-m", "pip", "install", f"git+{GIT_URL}"], check=True)
    else:
        python_bin = VENV_DIR / "bin" / "python"
    patch_onnx2webnn(html_fix_only=True)
    return python_bin


def patch_onnx2webnn(html_fix_only: bool = True) -> None:
    """Patch the installed onnx2webnn to avoid f-string syntax issues."""
    target = VENV_DIR / "lib" / "python3.11" / "site-packages" / "onnx2webnn" / "onnx2webnn.py"
    if not target.exists():
        return
    text = target.read_text()
    if html_fix_only:
        if "html_code = f\"\"\"" in text:
            text = text.replace("html_code = f\"\"\"", "html_code = \"\"\"")
            target.write_text(text)
    else:
        target.write_text(text)


def convert(input_path: Path, output_path: Path) -> None:
    python_bin = ensure_venv()
    simplified = simplify_onnx(input_path)
    patched = preprocess_onnx(simplified)
    try:
        subprocess.run(
            [
                str(python_bin),
                "-m",
                "onnx2webnn",
                "-if",
                str(patched),
                "-oj",
                str(output_path),
            ],
            check=True,
        )
    finally:
        for tmp in {patched, simplified}:
            if tmp != input_path and tmp.exists():
                tmp.unlink()
    print(f"Converted {input_path} -> {output_path}")


def preprocess_onnx(input_path: Path) -> Path:
    """Strip trivial Cast nodes, rewrite Clip->Clamp, normalize aliases, then reject unsupported ops."""
    model = onnx.load(input_path)
    # Try to infer shapes to improve folding of Shape nodes.
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping shape inference: {exc}")
    graph = model.graph
    replacements = {}
    kept_nodes = []
    init_map = {init.name: init for init in graph.initializer}

    for node in graph.node:
        # Strip trivial Cast
        if node.op_type == "Cast" and len(node.input) == 1 and len(node.output) == 1:
            replacements[node.output[0]] = node.input[0]
            continue

        # Lift Constant nodes into initializers.
        if node.op_type == "Constant" and len(node.output) == 1:
            tensor = constant_to_initializer(node)
            if tensor is not None:
                init_map[node.output[0]] = tensor
                graph.initializer.append(tensor)
                replacements[node.output[0]] = node.output[0]
                continue

        # Rewrite Clip -> Clamp
        if node.op_type == "Clip":
            node.op_type = "Clamp"

        # Alias normalizations
        if node.op_type == "Less":
            node.op_type = "Lesser"
        if node.op_type == "LessOrEqual":
            node.op_type = "LesserOrEqual"

        # SimplifiedLayerNormalization -> LayerNormalization
        if node.op_type == "SimplifiedLayerNormalization":
            node.op_type = "LayerNormalization"

        # Fold Range if all inputs are scalar initializers
        if node.op_type == "Range" and len(node.input) == 3:
            start = get_scalar(init_map.get(node.input[0]))
            limit = get_scalar(init_map.get(node.input[1]))
            delta = get_scalar(init_map.get(node.input[2]))
            if start is not None and limit is not None and delta is not None:
                arr = np.arange(start, limit, delta, dtype=np.float32)
                const_name = node.output[0]
                tensor = onnx.helper.make_tensor(
                    name=const_name,
                    data_type=onnx.TensorProto.FLOAT,
                    dims=arr.shape,
                    vals=arr.flatten().tolist(),
                )
                init_map[const_name] = tensor
                graph.initializer.append(tensor)
                replacements[const_name] = const_name
                continue

        # Fold Shape if input shape is static
        if node.op_type == "Shape" and len(node.input) == 1:
            shape_vals = get_static_shape(model, node.input[0])
            if shape_vals is not None:
                arr = np.array(shape_vals, dtype=np.int64)
                const_name = node.output[0]
                tensor = onnx.helper.make_tensor(
                    name=const_name,
                    data_type=onnx.TensorProto.INT64,
                    dims=arr.shape,
                    vals=arr.flatten().tolist(),
                )
                init_map[const_name] = tensor
                graph.initializer.append(tensor)
                replacements[const_name] = const_name
                continue

        kept_nodes.append(node)

    for node in kept_nodes:
        for i, name in enumerate(node.input):
            if name in replacements and replacements[name] != name:
                node.input[i] = replacements[name]
    graph.ClearField("node")
    graph.node.extend(kept_nodes)
    tmp = input_path.with_suffix(".patched.onnx")
    onnx.save(model, tmp)
    check_supported_ops(model)
    return tmp


def simplify_onnx(input_path: Path) -> Path:
    """Run a basic ONNX Runtime optimization pass to fold shapes/ranges if possible."""
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        _ = ort.InferenceSession(str(input_path), so, providers=["CPUExecutionProvider"])
        # If ORT can load the model, it will have applied optimizations internally.
        # Dump the optimized model.
        optimized_path = input_path.with_suffix(".optimized.onnx")
        so.optimized_model_filepath = str(optimized_path)
        _ = ort.InferenceSession(str(input_path), so, providers=["CPUExecutionProvider"])
        if optimized_path.exists():
            return optimized_path
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping ORT simplification: {exc}")
    return input_path


def get_scalar(initializer: onnx.TensorProto | None) -> float | None:
    if initializer is None:
        return None
    if initializer.data_type == onnx.TensorProto.FLOAT and len(initializer.float_data) == 1:
        return float(initializer.float_data[0])
    if initializer.data_type == onnx.TensorProto.INT64 and len(initializer.int64_data) == 1:
        return float(initializer.int64_data[0])
    if initializer.data_type == onnx.TensorProto.INT32 and len(initializer.int32_data) == 1:
        return float(initializer.int32_data[0])
    return None


def get_static_shape(model: onnx.ModelProto, value_name: str) -> list[int] | None:
    def extract(dims):
        vals = []
        for dim in dims:
            if dim.dim_value == 0:
                return None
            if dim.dim_value is None:
                return None
            vals.append(int(dim.dim_value))
        return vals

    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if vi.name == value_name:
            if vi.type.HasField("tensor_type"):
                dims = vi.type.tensor_type.shape.dim
                return extract(dims)
    return None


def constant_to_initializer(node: onnx.NodeProto) -> onnx.TensorProto | None:
    attr_map = {a.name: a for a in node.attribute}
    if "value" in attr_map:
        return attr_map["value"].t
    if "value_float" in attr_map:
        val = attr_map["value_float"].f
        return onnx.helper.make_tensor(
            name=node.output[0],
            data_type=onnx.TensorProto.FLOAT,
            dims=[],
            vals=[val],
        )
    if "value_int" in attr_map:
        val = attr_map["value_int"].i
        return onnx.helper.make_tensor(
            name=node.output[0],
            data_type=onnx.TensorProto.INT64,
            dims=[],
            vals=[val],
        )
    return None


# WebNN ops from https://webmachinelearning.github.io/webnn/#programming-model-operators
SUPPORTED_WEBNN_OPS = {
    # Tensor creation/manipulation
    "Concat",
    "Expand",
    "Gather",
    "GatherElements",
    "ScatterElements",
    "GatherND",
    "ScatterND",
    "Where",
    "Pad",
    "Reshape",
    "Slice",
    "Split",
    "Transpose",
    "Resample2d",
    "Reverse",
    "Tile",
    "Triangular",
    "Unsqueeze",
    "Constant",
    "ConstantOfShape",
    # Quantization / casting
    "QuantizeLinear",
    "DequantizeLinear",
    "Cast",
    # Math
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Max",
    "Min",
    "Clamp",
    "Pow",
    "Abs",
    "Ceil",
    "Cos",
    "Erf",
    "Exp",
    "Floor",
    "Identity",
    "Log",
    "Neg",
    "Reciprocal",
    "Sin",
    "Sqrt",
    "Tan",
    "Tanh",
    "Sign",
    # Logical
    "Equal",
    "NotEqual",
    "Greater",
    "GreaterOrEqual",
    "Lesser",
    "LesserOrEqual",
    "LogicalNot",
    "LogicalAnd",
    "LogicalOr",
    "LogicalXor",
    # Matmul / GEMM
    "MatMul",
    "Gemm",
    # Conv / pooling
    "Conv2d",
    "ConvTranspose2d",
    "AveragePool2d",
    "L2Pool2d",
    "MaxPool2d",
    # Activation
    "Elu",
    "Gelu",
    "HardSigmoid",
    "HardSwish",
    "LeakyRelu",
    "Linear",
    "Prelu",
    "Relu",
    "Sigmoid",
    "Softmax",
    "Softplus",
    "Softsign",
    "Tanh",
    # Normalization
    "BatchNormalization",
    "InstanceNormalization",
    "LayerNormalization",
    # Reduction
    "ArgMin",
    "ArgMax",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceSumSquare",
    "CumulativeSum",
    # RNN
    "GruCell",
    "Gru",
    "LstmCell",
    "Lstm",
}


def check_supported_ops(model: onnx.ModelProto) -> None:
    unsupported = set()
    for node in model.graph.node:
        if node.op_type not in SUPPORTED_WEBNN_OPS and node.op_type != "Cast":
            unsupported.add(node.op_type)
    if unsupported:
        ops_list = ", ".join(sorted(unsupported))
        raise RuntimeError(f"Unsupported ops for WebNN conversion: {ops_list}")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 1
    input_path = Path(argv[1])
    output_path = Path(argv[2])
    if not input_path.exists():
        print(f"Input ONNX file not found: {input_path}")
        return 1
    convert(input_path, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

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
    patched = preprocess_onnx(input_path)
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
        if patched != input_path and patched.exists():
            patched.unlink()
    print(f"Converted {input_path} -> {output_path}")


def preprocess_onnx(input_path: Path) -> Path:
    """Strip trivial Cast nodes and reject unsupported ops before conversion."""
    model = onnx.load(input_path)
    check_supported_ops(model)
    graph = model.graph
    replacements = {}
    kept_nodes = []
    for node in graph.node:
        if node.op_type == "Cast" and len(node.input) == 1 and len(node.output) == 1:
            replacements[node.output[0]] = node.input[0]
            continue
        kept_nodes.append(node)
    if not replacements:
        return input_path
    for node in kept_nodes:
        for i, name in enumerate(node.input):
            if name in replacements:
                node.input[i] = replacements[name]
    graph.ClearField("node")
    graph.node.extend(kept_nodes)
    tmp = input_path.with_suffix(".patched.onnx")
    onnx.save(model, tmp)
    return tmp


# WebNN ops from https://www.w3.org/TR/webnn/#api
SUPPORTED_WEBNN_OPS = {
    # Elementwise
    "Abs",
    "Add",
    "Clamp",
    "Div",
    "Equal",
    "Exp",
    "Floor",
    "LeakyRelu",
    "Log",
    "Max",
    "Min",
    "Mul",
    "Neg",
    "Pow",
    "Relu",
    "Sigmoid",
    "Sqrt",
    "Sub",
    "Tanh",
    # Reductions
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    # Normalization / activation
    "BatchNormalization",
    "InstanceNormalization",
    "LayerNormalization",
    "HardSigmoid",
    "Gelu",
    "Softplus",
    "Softsign",
    # Linear algebra
    "Gemm",
    "MatMul",
    "Linear",  # Dense
    # Pooling / conv
    "AveragePool2d",
    "MaxPool2d",
    "Conv2d",
    "ConvTranspose2d",
    "Resample2d",
    "Pad",
    # Tensor ops
    "Concat",
    "Gather",
    "GatherElements",
    "GatherND",
    "Reshape",
    "Slice",
    "Squeeze",
    "Transpose",
    "Unsqueeze",
    "Where",
    # RNNs
    "Gru",
    "Lstm",
    # Other
    "Softmax",
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

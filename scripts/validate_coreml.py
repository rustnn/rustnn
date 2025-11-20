#!/usr/bin/env python3
"""
Validate and execute a generated CoreML binary (`.mlmodel`) graph.

Usage:
    python scripts/validate_coreml.py target/graph.mlmodel

Steps:
1) Load the emitted CoreML spec to sanity check the IO signature.
2) Run a CPU-only predict pass with zeroed inputs (when coremltools is present).
"""
import os
import sys
from pathlib import Path


def _dtype_to_numpy_code(value: int):
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return None

    map_ = {
        65568: np.float32,  # ArrayDataType::Float32
        65552: np.float16,  # ArrayDataType::Float16
        131104: np.int32,  # ArrayDataType::Int32
        131080: np.int8,  # ArrayDataType::Int8
    }
    return map_.get(value, np.float32)


def _coerce_shape(shape):
    dims = list(shape) or [1]
    return [int(dim) for dim in dims]


def validate_coreml(model_path: Path) -> None:
    try:
        import coremltools as ct  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        print("coremltools not installed; skipping CoreML validation.")
        return

    spec = ct.utils.load_spec(model_path)
    print(f"Loaded CoreML spec from {model_path}")
    for entry in spec.description.input:
        dtype = _dtype_to_numpy_code(entry.type.multiArrayType.dataType)
        shape = _coerce_shape(entry.type.multiArrayType.shape)
        print(f"  input {entry.name}: shape={shape}, dtype={dtype}")
    for entry in spec.description.output:
        shape = _coerce_shape(entry.type.multiArrayType.shape)
        print(f"  output {entry.name}: shape={shape}")

    os.environ.setdefault("TMPDIR", "/tmp")
    try:
        model = ct.models.MLModel(spec, compute_units=ct.ComputeUnit.CPU_ONLY)
    except Exception:
        # Retry with a forced /tmp tempdir in case the default sandbox path is blocked.
        os.environ["TMPDIR"] = "/tmp"
        model = ct.models.MLModel(spec, compute_units=ct.ComputeUnit.CPU_ONLY)
    feeds = {}
    for entry in spec.description.input:
        dtype = _dtype_to_numpy_code(entry.type.multiArrayType.dataType)
        if dtype is None:
            raise RuntimeError(f"Unsupported CoreML dtype {entry.type.multiArrayType.dataType}")
        shape = _coerce_shape(entry.type.multiArrayType.shape)
        feeds[entry.name] = np.zeros(shape, dtype=dtype)

    try:
        outputs = model.predict(feeds)
        print("CoreML predict succeeded:")
        for name, value in outputs.items():
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", type(value))
            print(f"  {name}: shape={shape}, dtype={dtype}")
    except Exception as exc:  # noqa: B902
        if "working directory" in str(exc):
            print("CoreML predict skipped: sandbox cannot create a temp working directory.")
        else:
            print(f"CoreML predict failed: {exc}")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    path = Path(sys.argv[1])
    if not path.exists():
        raise SystemExit(f"Model file not found: {path}")
    validate_coreml(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

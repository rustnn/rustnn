# Backend Operator Support Report

This file is generated from the converter sources by `scripts/generate_backend_operator_report.py`.
Do not edit it manually. Run `make docs-backend-ops` after backend changes; CI fails on drift (`make docs-backend-ops-check`).

Operation names are the WebNN builder names returned by `Operation::op_type()` in `src/operators.rs`. "Supported" means the converter emits a lowering for the operation. Data type restrictions, dynamic shape limits and known failing cases are tracked per backend in `tests/wpt_conformance/*_expected_failures.txt` and on the [WPT conformance dashboard](https://rustnn.github.io/rustnn/wpt-conformance/).

## Summary

| Backend | Converter source | Detection rule | Supported |
|---|---|---|---|
| ONNX Runtime | `src/converters/onnx.rs` | `Operation` variants referenced by the converter | 100 of 100 |
| CoreML | `src/converters/coreml_mlprogram.rs` | `Operation` variants referenced by the converter, plus names in its op-type dispatch | 99 of 100 |
| TensorRT | `src/converters/trtx.rs` | keys of the `match op_type` dispatch table | 99 of 100 |
| LiteRT | `src/converters/litert.rs` | `Operation` variants referenced by the converter | 92 of 100 |
| CANN | `src/converters/cann.rs` | variants accepted by `is_supported_op` | 17 of 100 |

## Operation matrix

| Operation | ONNX Runtime | CoreML | TensorRT | LiteRT | CANN |
|---|:-:|:-:|:-:|:-:|:-:|
| `abs` | yes | yes | yes | yes | - |
| `add` | yes | yes | yes | yes | yes |
| `argMax` | yes | yes | yes | yes | - |
| `argMin` | yes | yes | yes | yes | - |
| `averagePool2d` | yes | yes | yes | yes | - |
| `batchNormalization` | yes | yes | yes | yes | - |
| `cast` | yes | yes | yes | yes | yes |
| `ceil` | yes | yes | yes | yes | - |
| `clamp` | yes | yes | yes | yes | - |
| `concat` | yes | yes | yes | yes | yes |
| `conv2d` | yes | yes | yes | yes | yes |
| `convTranspose2d` | yes | yes | yes | yes | - |
| `cos` | yes | yes | yes | yes | - |
| `cumulativeSum` | yes | yes | yes | yes | - |
| `dequantizeLinear` | yes | yes | yes | yes | - |
| `div` | yes | yes | yes | yes | yes |
| `elu` | yes | yes | yes | yes | - |
| `equal` | yes | yes | yes | yes | - |
| `erf` | yes | yes | yes | yes | - |
| `exp` | yes | yes | yes | yes | - |
| `expand` | yes | yes | yes | yes | - |
| `floor` | yes | yes | yes | yes | - |
| `gather` | yes | yes | yes | yes | - |
| `gatherElements` | yes | yes | yes | yes | - |
| `gatherND` | yes | yes | yes | yes | - |
| `gelu` | yes | yes | yes | yes | - |
| `gemm` | yes | yes | yes | yes | - |
| `globalAveragePool` | yes | yes | yes | - | - |
| `globalMaxPool` | yes | yes | yes | - | - |
| `greater` | yes | yes | yes | yes | - |
| `greaterOrEqual` | yes | yes | yes | yes | - |
| `gru` | yes | yes | yes | - | - |
| `gruCell` | yes | yes | yes | - | - |
| `hardSigmoid` | yes | yes | yes | yes | - |
| `hardSwish` | yes | yes | yes | yes | - |
| `identity` | yes | yes | yes | yes | - |
| `instanceNormalization` | yes | yes | yes | yes | - |
| `isInfinite` | yes | yes | yes | yes | - |
| `isNaN` | yes | yes | yes | yes | - |
| `l2Pool2d` | yes | yes | yes | yes | - |
| `layerNormalization` | yes | yes | yes | yes | - |
| `leakyRelu` | yes | yes | yes | yes | - |
| `lesser` | yes | yes | yes | yes | - |
| `lesserOrEqual` | yes | yes | yes | yes | - |
| `linear` | yes | yes | yes | yes | - |
| `log` | yes | yes | yes | yes | - |
| `logicalAnd` | yes | yes | yes | yes | - |
| `logicalNot` | yes | yes | yes | yes | - |
| `logicalOr` | yes | yes | yes | yes | - |
| `logicalXor` | yes | yes | yes | yes | - |
| `lstm` | yes | yes | yes | - | - |
| `lstmCell` | yes | yes | yes | - | - |
| `matmul` | yes | yes | yes | yes | - |
| `max` | yes | yes | yes | yes | - |
| `maxPool2d` | yes | yes | yes | yes | yes |
| `min` | yes | yes | yes | yes | - |
| `mul` | yes | yes | yes | yes | yes |
| `neg` | yes | yes | yes | yes | - |
| `notEqual` | yes | yes | yes | yes | - |
| `pad` | yes | yes | yes | yes | - |
| `pow` | yes | yes | yes | yes | - |
| `prelu` | yes | yes | yes | yes | yes |
| `quantizeLinear` | yes | yes | yes | yes | - |
| `reciprocal` | yes | yes | yes | yes | - |
| `reduceL1` | yes | yes | yes | yes | - |
| `reduceL2` | yes | yes | yes | yes | - |
| `reduceLogSum` | yes | yes | yes | yes | - |
| `reduceLogSumExp` | yes | yes | yes | yes | - |
| `reduceMax` | yes | yes | yes | yes | - |
| `reduceMean` | yes | yes | yes | yes | - |
| `reduceMin` | yes | yes | yes | yes | - |
| `reduceProduct` | yes | yes | yes | yes | - |
| `reduceSum` | yes | yes | yes | yes | yes |
| `reduceSumSquare` | yes | yes | yes | yes | - |
| `relu` | yes | yes | yes | yes | - |
| `resample2d` | yes | yes | yes | yes | yes |
| `reshape` | yes | yes | yes | yes | yes |
| `reverse` | yes | yes | yes | yes | - |
| `roundEven` | yes | yes | yes | yes | - |
| `scatterElements` | yes | yes | yes | yes | - |
| `scatterND` | yes | yes | yes | yes | - |
| `shape` | yes | - | - | - | - |
| `sigmoid` | yes | yes | yes | yes | yes |
| `sign` | yes | yes | yes | yes | - |
| `sin` | yes | yes | yes | yes | - |
| `slice` | yes | yes | yes | yes | yes |
| `softmax` | yes | yes | yes | yes | yes |
| `softplus` | yes | yes | yes | yes | - |
| `softsign` | yes | yes | yes | yes | - |
| `split` | yes | yes | yes | yes | yes |
| `sqrt` | yes | yes | yes | yes | - |
| `squeeze` | yes | yes | yes | yes | - |
| `sub` | yes | yes | yes | yes | yes |
| `tan` | yes | yes | yes | yes | - |
| `tanh` | yes | yes | yes | yes | - |
| `tile` | yes | yes | yes | yes | - |
| `transpose` | yes | yes | yes | yes | yes |
| `triangular` | yes | yes | yes | yes | - |
| `unsqueeze` | yes | yes | yes | - | - |
| `where` | yes | yes | yes | yes | - |

## Unsupported operations per backend

- ONNX Runtime: none
- CoreML: `shape`
- TensorRT: `shape`
- LiteRT: `globalAveragePool`, `globalMaxPool`, `gru`, `gruCell`, `lstm`, `lstmCell`, `shape`, `unsqueeze`
- CANN: `abs`, `argMax`, `argMin`, `averagePool2d`, `batchNormalization`, `ceil`, `clamp`, `convTranspose2d`, `cos`, `cumulativeSum`, `dequantizeLinear`, `elu`, `equal`, `erf`, `exp`, `expand`, `floor`, `gather`, `gatherElements`, `gatherND`, `gelu`, `gemm`, `globalAveragePool`, `globalMaxPool`, `greater`, `greaterOrEqual`, `gru`, `gruCell`, `hardSigmoid`, `hardSwish`, `identity`, `instanceNormalization`, `isInfinite`, `isNaN`, `l2Pool2d`, `layerNormalization`, `leakyRelu`, `lesser`, `lesserOrEqual`, `linear`, `log`, `logicalAnd`, `logicalNot`, `logicalOr`, `logicalXor`, `lstm`, `lstmCell`, `matmul`, `max`, `min`, `neg`, `notEqual`, `pad`, `pow`, `quantizeLinear`, `reciprocal`, `reduceL1`, `reduceL2`, `reduceLogSum`, `reduceLogSumExp`, `reduceMax`, `reduceMean`, `reduceMin`, `reduceProduct`, `reduceSumSquare`, `relu`, `reverse`, `roundEven`, `scatterElements`, `scatterND`, `shape`, `sign`, `sin`, `softplus`, `softsign`, `sqrt`, `squeeze`, `tan`, `tanh`, `tile`, `triangular`, `unsqueeze`, `where`

## Notes

- `shape`: rustnn extension used by onnx2webnn exports.
- `squeeze`: removed from the WebNN spec (emulation appendix), kept for onnx2webnn.
- `unsqueeze`: removed from the WebNN spec (emulation appendix), kept for onnx2webnn.


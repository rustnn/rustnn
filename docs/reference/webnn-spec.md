# WebNN API Specification Reference

**Source:** https://www.w3.org/TR/webnn/
**Status:** W3C Candidate Recommendation Draft (December 3, 2025)
**Local Copy:** Saved for offline reference and easy parsing

## Overview

The Web Neural Network API (WebNN) defines a dedicated low-level API for neural network inference hardware acceleration. It provides hardware-agnostic access to ML acceleration capabilities across CPU, GPU, and dedicated ML accelerators.

## Core Interfaces

### ML
Entry point for creating ML contexts.

### MLContext
Global execution state managing device resources and graph compilation.

### MLGraphBuilder
Constructs computational graphs using operator methods.

### MLOperand
Represents data flowing through the graph (inputs, constants, intermediate values, outputs).

### MLGraph
Compiled, immutable representation of the computational graph.

### MLTensor
Runtime data binding for graph execution.

## Reduction Operations

Reduction operations reduce input tensor dimensions by applying a reduction function across specified axes.

### Common Parameters (MLReduceOptions)

```webidl
dictionary MLReduceOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> axes;
  boolean keepDimensions = false;
};
```

**Parameters:**
- `axes`: Array of dimension indices to reduce. If not specified, reduces all dimensions.
- `keepDimensions`: If true, retains reduced dimensions with size 1. Default is false.

### reduceSum()

Reduces the input tensor by summing elements along specified axes.

**Formula:** `output = Σ input[i]` for i in reduced dimensions

**Signature:**
```webidl
MLOperand reduceSum(MLOperand input, optional MLReduceOptions options = {});
```

**ONNX Mapping:** `ReduceSum`

### reduceMean()

Reduces the input tensor by computing the arithmetic mean along specified axes.

**Formula:** `output = (Σ input[i]) / n` where n is the number of elements reduced

**Signature:**
```webidl
MLOperand reduceMean(MLOperand input, optional MLReduceOptions options = {});
```

**ONNX Mapping:** `ReduceMean`

### reduceMax()

Reduces the input tensor by computing the maximum value along specified axes.

**Formula:** `output = max(input[i])` for i in reduced dimensions

**Signature:**
```webidl
MLOperand reduceMax(MLOperand input, optional MLReduceOptions options = {});
```

**ONNX Mapping:** `ReduceMax`

### reduceMin()

Reduces the input tensor by computing the minimum value along specified axes.

**Formula:** `output = min(input[i])` for i in reduced dimensions

**Signature:**
```webidl
MLOperand reduceMin(MLOperand input, optional MLReduceOptions options = {});
```

**ONNX Mapping:** `ReduceMin`

### reduceProduct()

Reduces the input tensor by computing the product of elements along specified axes.

**Formula:** `output = Π input[i]` for i in reduced dimensions

**Signature:**
```webidl
MLOperand reduceProduct(MLOperand input, optional MLReduceOptions options = {});
```

**ONNX Mapping:** `ReduceProd`

### reduceL1()

Reduces the input tensor by computing the L1 norm (sum of absolute values) along specified axes.

**Formula:** `output = Σ |input[i]|` for i in reduced dimensions

**Signature:**
```webidl
MLOperand reduceL1(MLOperand input, optional MLReduceOptions options = {});
```

**ONNX Mapping:** `ReduceL1`

### reduceL2()

Reduces the input tensor by computing the L2 norm (Euclidean norm) along specified axes.

**Formula:** `output = sqrt(Σ input[i]²)` for i in reduced dimensions

**Signature:**
```webidl
MLOperand reduceL2(MLOperand input, optional MLReduceOptions options = {});
```

**ONNX Mapping:** `ReduceL2`

### reduceLogSum()

Reduces the input tensor by computing the natural logarithm of the sum along specified axes.

**Formula:** `output = log(Σ input[i])` for i in reduced dimensions

**Signature:**
```webidl
MLOperand reduceLogSum(MLOperand input, optional MLReduceOptions options = {});
```

**ONNX Mapping:** `ReduceLogSum`

### reduceLogSumExp()

Reduces the input tensor by computing the log of the sum of exponentials along specified axes.

**Formula:** `output = log(Σ exp(input[i]))` for i in reduced dimensions

**Signature:**
```webidl
MLOperand reduceLogSumExp(MLOperand input, optional MLReduceOptions options = {});
```

**ONNX Mapping:** `ReduceLogSumExp`

### reduceSumSquare()

Reduces the input tensor by computing the sum of squares along specified axes.

**Formula:** `output = Σ input[i]²` for i in reduced dimensions

**Signature:**
```webidl
MLOperand reduceSumSquare(MLOperand input, optional MLReduceOptions options = {});
```

**ONNX Mapping:** `ReduceSumSquare`

## Shape Inference for Reduction Operations

**Input shape:** `[d0, d1, d2, ..., dn]`

**If keepDimensions = false:**
- Output shape removes the reduced dimensions
- Example: `[2, 3, 4]` with `axes=[1]` → `[2, 4]`

**If keepDimensions = true:**
- Output shape keeps reduced dimensions with size 1
- Example: `[2, 3, 4]` with `axes=[1]` and `keepDimensions=true` → `[2, 1, 4]`

**If axes is empty or not specified:**
- Reduces all dimensions
- Output is a scalar (rank-0 tensor) with `keepDimensions=false`
- Output is `[1, 1, ..., 1]` with `keepDimensions=true`

## Implementation Notes

### Excluded Operations

**localResponseNormalization** - NOT part of WebNN spec as of 2025-12-07
- Decision: Use decomposition in higher layers (e.g., ONNX Runtime's WebNN EP)
- Reason: Rarity in modern models, awkward backend differences
- Source: W3C WebML WG meeting notes (2024-10-31)

### Data Type Support

Reduction operations typically support:
- `float32` (required)
- `float16` (optional)
- `int32` (optional, for min/max operations)
- `int8`/`uint8` (optional, for min/max operations)

### Numerical Stability

**reduceLogSumExp** uses the log-sum-exp trick for numerical stability:
```
output = log(Σ exp(input[i]))
       = max_val + log(Σ exp(input[i] - max_val))
```
where `max_val = max(input[i])` for i in reduced dimensions.

## Additional Operations

For a complete list of all WebNN operations, see:
- Official spec: https://www.w3.org/TR/webnn/
- Implementation status: https://webmachinelearning.github.io/webnn-status/

---

**Last Updated:** 2025-12-07
**Spec Version:** W3C Candidate Recommendation Draft (2025-12-03)

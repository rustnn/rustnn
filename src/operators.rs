/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! WebNN operator enum: one variant per builder with named operand fields and options.
//!
//! This module defines the `Operator` enum as the single source of truth for each WebNN
//! operation: each variant carries the builder name, named operand indices (no positional
//! ambiguity), and the corresponding ML*Options struct. All option types and MLDimension
//! are reused from [crate::operator_options].
//!
//! # Spec reference
//!
//! - [Web Neural Network API](https://www.w3.org/TR/webnn/)
//!
//! # Optional options (per spec)
//!
//! In the WebNN spec, the **options parameter is optional** for every operator:
//! each `MLGraphBuilder` method is defined as `optional ML*Options options = {}`.
//! So the options object itself is optional: each variant's `options` field is
//! `Option<ML*Options>`. When `None`, the operator was created without an options
//! argument (spec defaults apply).

use crate::operator_options::{
    MLArgMinMaxOptions, MLBatchNormalizationOptions, MLCastOptions, MLClampOptions,
    MLConstantOptions, MLConv2dOptions, MLConvTranspose2dOptions, MLConcatOptions,
    MLCumulativeSumOptions, MLEluOptions, MLExpandOptions, MLGatherOptions, MLGemmOptions,
    MLGruCellOptions, MLGruOptions, MLHardSigmoidOptions, MLHardSwishOptions,
    MLInstanceNormalizationOptions, MLLayerNormalizationOptions, MLLeakyReluOptions,
    MLLinearOptions, MLLstmCellOptions, MLLstmOptions, MLOperatorOptions, MLPadOptions,
    MLPool2dOptions, MLReduceOptions, MLResample2dOptions, MLReshapeOptions, MLReverseOptions,
    MLScatterOptions, MLSliceOptions, MLSoftmaxOptions, MLSplitOptions, MLSqueezeOptions,
    MLTileOptions, MLTransposeOptions, MLTriangularOptions, MLUnsqueezeOptions, OperandIndex,
};

// ---------------------------------------------------------------------------
// Operator enum: one variant per WebNN builder
// ---------------------------------------------------------------------------

/// One variant per WebNN graph builder. Each variant has named operand fields and the
/// corresponding options struct, so operand roles are explicit and independent of
/// input_operands order.
#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    // ---------- Binary element-wise (MLOperatorOptions) ----------
    /// [add()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-add)
    Add {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [sub()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-sub)
    Sub {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [mul()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-mul)
    Mul {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [div()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-div)
    Div {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [pow()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-pow)
    Pow {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [matmul()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-matmul)
    Matmul {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
    },

    // ---------- Comparison (MLOperatorOptions) ----------
    /// [equal()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-equal)
    Equal {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [greater()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-greater)
    Greater {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [greaterOrEqual()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-greaterorequal)
    GreaterOrEqual {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [lesser()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-lesser)
    Lesser {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [lesserOrEqual()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-lesserorequal)
    LesserOrEqual {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [notEqual()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-notequal)
    NotEqual {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
    },

    // ---------- Unary element-wise (MLOperatorOptions) ----------
    /// [abs()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-abs)
    Abs {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [ceil()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-ceil)
    Ceil {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [cos()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-cos)
    Cos {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [exp()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-exp)
    Exp {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [floor()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-floor)
    Floor {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [log()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-log)
    Log {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [neg()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-neg)
    Neg {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [relu()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-relu)
    Relu {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [sigmoid()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-sigmoid)
    Sigmoid {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [sin()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-sin)
    Sin {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [sqrt()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-sqrt)
    Sqrt {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [tan()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-tan)
    Tan {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [tanh()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-tanh)
    Tanh {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [asin()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-asin)
    Asin {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [acos()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-acos)
    Acos {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [atan()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-atan)
    Atan {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [sinh()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-sinh)
    Sinh {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [cosh()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-cosh)
    Cosh {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [atanh()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-atanh)
    Atanh {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [erf()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-erf)
    Erf {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [reciprocal()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reciprocal)
    Reciprocal {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [sign()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-sign)
    Sign {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [round()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-round) (spec name may differ; roundEven is separate)
    Round {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },

    // ---------- Logical (MLOperatorOptions) ----------
    /// [logicalAnd()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-logicaland)
    LogicalAnd {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [logicalOr()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-logicalor)
    LogicalOr {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [logicalNot()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-logicalnot)
    LogicalNot {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [logicalXor()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-logicalxor)
    LogicalXor {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
    },

    // ---------- Conditional / identity ----------
    /// [where()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-where)
    Where {
        condition: OperandIndex,
        true_value: OperandIndex,
        false_value: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [identity()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-identity)
    Identity {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },

    // ---------- ArgMin / ArgMax ----------
    /// [argMin()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-argmin)
    ArgMin {
        input: OperandIndex,
        options: Option<MLArgMinMaxOptions>,
    },
    /// [argMax()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-argmax)
    ArgMax {
        input: OperandIndex,
        options: Option<MLArgMinMaxOptions>,
    },

    // ---------- BatchNormalization ----------
    /// [batchNormalization()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-batchnormalization)
    BatchNormalization {
        input: OperandIndex,
        mean: OperandIndex,
        variance: OperandIndex,
        options: Option<MLBatchNormalizationOptions>,
    },

    // ---------- Cast ----------
    /// [cast()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-cast)
    Cast {
        input: OperandIndex,
        options: Option<MLCastOptions>,
    },

    // ---------- Clamp ----------
    /// [clamp()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-clamp)
    Clamp {
        input: OperandIndex,
        options: Option<MLClampOptions>,
    },

    // ---------- Constant (no input operands) ----------
    /// [constant()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-constant)
    Constant {
        options: Option<MLConstantOptions>,
    },

    // ---------- Conv2d ----------
    /// [conv2d()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-conv2d)
    Conv2d {
        input: OperandIndex,
        filter: OperandIndex,
        options: Option<MLConv2dOptions>,
    },

    // ---------- ConvTranspose2d ----------
    /// [convTranspose2d()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-convtranspose2d)
    ConvTranspose2d {
        input: OperandIndex,
        filter: OperandIndex,
        options: Option<MLConvTranspose2dOptions>,
    },

    // ---------- Concat ----------
    /// [concat()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-concat)
    Concat {
        inputs: Vec<OperandIndex>,
        options: Option<MLConcatOptions>,
    },

    // ---------- CumulativeSum ----------
    /// [cumulativeSum()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-cumulativesum)
    CumulativeSum {
        input: OperandIndex,
        options: Option<MLCumulativeSumOptions>,
    },

    // ---------- Expand ----------
    /// [expand()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-expand)
    Expand {
        input: OperandIndex,
        options: Option<MLExpandOptions>,
    },

    // ---------- Elu ----------
    /// [elu()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-elu)
    Elu {
        input: OperandIndex,
        options: Option<MLEluOptions>,
    },

    // ---------- Gather / GatherElements ----------
    /// [gather()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-gather)
    Gather {
        input: OperandIndex,
        indices: OperandIndex,
        options: Option<MLGatherOptions>,
    },
    /// [gatherElements()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-gatherelements)
    GatherElements {
        input: OperandIndex,
        indices: OperandIndex,
        options: Option<MLGatherOptions>,
    },

    // ---------- Gemm ----------
    /// [gemm()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-gemm)
    Gemm {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLGemmOptions>,
    },

    // ---------- GRU ----------
    /// [gru()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-gru)
    Gru {
        input: OperandIndex,
        weight: OperandIndex,
        recurrence: OperandIndex,
        options: Option<MLGruOptions>,
    },
    /// [gruCell()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-grucell)
    GruCell {
        input: OperandIndex,
        weight: OperandIndex,
        recurrence: OperandIndex,
        options: Option<MLGruCellOptions>,
    },

    // ---------- HardSigmoid / HardSwish ----------
    /// [hardSigmoid()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-hardsigmoid)
    HardSigmoid {
        input: OperandIndex,
        options: Option<MLHardSigmoidOptions>,
    },
    /// [hardSwish()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-hardswish)
    HardSwish {
        input: OperandIndex,
        options: Option<MLHardSwishOptions>,
    },

    // ---------- InstanceNormalization ----------
    /// [instanceNormalization()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-instancenormalization)
    InstanceNormalization {
        input: OperandIndex,
        options: Option<MLInstanceNormalizationOptions>,
    },

    // ---------- LayerNormalization ----------
    /// [layerNormalization()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-layernormalization)
    LayerNormalization {
        input: OperandIndex,
        options: Option<MLLayerNormalizationOptions>,
    },

    // ---------- LeakyRelu ----------
    /// [leakyRelu()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-leakyrelu)
    LeakyRelu {
        input: OperandIndex,
        options: Option<MLLeakyReluOptions>,
    },

    // ---------- Linear ----------
    /// [linear()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-linear)
    Linear {
        input: OperandIndex,
        options: Option<MLLinearOptions>,
    },

    // ---------- LSTM ----------
    /// [lstm()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-lstm)
    Lstm {
        input: OperandIndex,
        weight: OperandIndex,
        recurrence: OperandIndex,
        options: Option<MLLstmOptions>,
    },
    /// [lstmCell()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-lstmcell)
    LstmCell {
        input: OperandIndex,
        weight: OperandIndex,
        recurrence: OperandIndex,
        options: Option<MLLstmCellOptions>,
    },

    // ---------- Pad ----------
    /// [pad()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-pad)
    Pad {
        input: OperandIndex,
        options: Option<MLPadOptions>,
    },

    // ---------- Pooling ----------
    /// [averagePool2d()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-averagepool2d)
    AveragePool2d {
        input: OperandIndex,
        options: Option<MLPool2dOptions>,
    },
    /// [maxPool2d()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-maxpool2d)
    MaxPool2d {
        input: OperandIndex,
        options: Option<MLPool2dOptions>,
    },
    /// [l2Pool2d()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-l2pool2d)
    L2Pool2d {
        input: OperandIndex,
        options: Option<MLPool2dOptions>,
    },
    /// Global average pooling (same options as pool2d; see spec table § 7.3).
    GlobalAveragePool {
        input: OperandIndex,
        options: Option<MLPool2dOptions>,
    },
    /// Global max pooling (same options as pool2d; see spec table § 7.3).
    GlobalMaxPool {
        input: OperandIndex,
        options: Option<MLPool2dOptions>,
    },

    // ---------- Reduction ----------
    /// [reduceSum()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducesum)
    ReduceSum {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
    },
    /// [reduceMean()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducemean)
    ReduceMean {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
    },
    /// [reduceMax()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducemax)
    ReduceMax {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
    },
    /// [reduceMin()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducemin)
    ReduceMin {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
    },
    /// [reduceProduct()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reduceproduct)
    ReduceProduct {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
    },
    /// [reduceL1()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducel1)
    ReduceL1 {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
    },
    /// [reduceL2()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducel2)
    ReduceL2 {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
    },
    /// [reduceLogSum()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducelogsum)
    ReduceLogSum {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
    },
    /// [reduceLogSumExp()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducelogsumexp)
    ReduceLogSumExp {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
    },
    /// [reduceSumSquare()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducesumsquare)
    ReduceSumSquare {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
    },

    // ---------- Reshape ----------
    /// [reshape()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reshape)
    Reshape {
        input: OperandIndex,
        options: Option<MLReshapeOptions>,
    },

    // ---------- Resample2d ----------
    /// [resample2d()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-resample2d)
    Resample2d {
        input: OperandIndex,
        options: Option<MLResample2dOptions>,
    },

    // ---------- Reverse ----------
    /// [reverse()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reverse)
    Reverse {
        input: OperandIndex,
        options: Option<MLReverseOptions>,
    },

    // ---------- ScatterElements ----------
    /// [scatterElements()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-scatterelements)
    ScatterElements {
        input: OperandIndex,
        indices: OperandIndex,
        updates: OperandIndex,
        options: Option<MLScatterOptions>,
    },

    // ---------- Softmax ----------
    /// [softmax()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-softmax)
    Softmax {
        input: OperandIndex,
        options: Option<MLSoftmaxOptions>,
    },

    // ---------- Slice ----------
    /// [slice()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-slice)
    Slice {
        input: OperandIndex,
        options: Option<MLSliceOptions>,
    },

    // ---------- Split ----------
    /// [split()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-split)
    Split {
        input: OperandIndex,
        options: Option<MLSplitOptions>,
    },

    // ---------- Transpose ----------
    /// [transpose()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-transpose)
    Transpose {
        input: OperandIndex,
        options: Option<MLTransposeOptions>,
    },

    // ---------- Squeeze / Unsqueeze (emulation) ----------
    /// [squeeze()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-squeeze) (§ 11 Operator Emulation)
    Squeeze {
        input: OperandIndex,
        options: Option<MLSqueezeOptions>,
    },
    /// [unsqueeze()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-unsqueeze) (§ 11 Operator Emulation)
    Unsqueeze {
        input: OperandIndex,
        options: Option<MLUnsqueezeOptions>,
    },

    // ---------- Tile ----------
    /// [tile()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-tile)
    Tile {
        input: OperandIndex,
        options: Option<MLTileOptions>,
    },

    // ---------- Triangular ----------
    /// [triangular()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-triangular)
    Triangular {
        input: OperandIndex,
        options: Option<MLTriangularOptions>,
    },

    // ---------- Prelu (binary, MLOperatorOptions) ----------
    /// [prelu()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-prelu)
    Prelu {
        input: OperandIndex,
        slope: OperandIndex,
        options: Option<MLOperatorOptions>,
    },

    // ---------- QuantizeLinear / DequantizeLinear ----------
    /// [quantizeLinear()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-quantizelinear)
    QuantizeLinear {
        input: OperandIndex,
        scale: OperandIndex,
        zero_point: Option<OperandIndex>,
        options: Option<MLOperatorOptions>,
    },
    /// [dequantizeLinear()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-dequantizelinear)
    DequantizeLinear {
        input: OperandIndex,
        scale: OperandIndex,
        zero_point: Option<OperandIndex>,
        options: Option<MLOperatorOptions>,
    },

    // ---------- Activation (softplus, softsign, gelu - MLOperatorOptions) ----------
    /// [softplus()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-softplus)
    Softplus {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [softsign()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-softsign)
    Softsign {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [gelu()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-gelu)
    Gelu {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },

    // ---------- Shape (interchange / internal) ----------
    /// Shape operator (interchange / internal; see spec § 7.3).
    Shape {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },

    // ---------- Optional / not yet in OperatorOptions ----------
    /// [scatterND()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-scatternd)
    ScatterND {
        input: OperandIndex,
        indices: OperandIndex,
        updates: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [gatherND()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-gathernd)
    GatherND {
        input: OperandIndex,
        indices: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [isNaN()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-isnan)
    IsNaN {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [isInfinite()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-isinfinite)
    IsInfinite {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
    /// [roundEven()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-roundeven)
    RoundEven {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
    },
}

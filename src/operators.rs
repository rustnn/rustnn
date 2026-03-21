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
    MLConcatOptions, MLConstantOptions, MLConv2dOptions, MLConvTranspose2dOptions,
    MLCumulativeSumOptions, MLEluOptions, MLExpandOptions, MLGatherOptions, MLGemmOptions,
    MLGruCellOptions, MLGruOptions, MLHardSigmoidOptions, MLHardSwishOptions,
    MLInstanceNormalizationOptions, MLLayerNormalizationOptions, MLLeakyReluOptions,
    MLLinearOptions, MLLstmCellOptions, MLLstmOptions, MLOperatorOptions, MLPadOptions,
    MLPool2dOptions, MLReduceOptions, MLResample2dOptions, MLReshapeOptions, MLReverseOptions,
    MLScatterOptions, MLSliceOptions, MLSoftmaxOptions, MLSplitOptions, MLSqueezeOptions,
    MLTileOptions, MLTransposeOptions, MLTriangularOptions, MLUnsqueezeOptions, OperandIndex,
    OperatorOptions,
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
        outputs: Vec<OperandIndex>,
    },
    /// [sub()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-sub)
    Sub {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [mul()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-mul)
    Mul {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [div()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-div)
    Div {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [pow()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-pow)
    Pow {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [max()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-max)
    Max {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [min()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-min)
    Min {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [matmul()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-matmul)
    Matmul {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Comparison (MLOperatorOptions) ----------
    /// [equal()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-equal)
    Equal {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [greater()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-greater)
    Greater {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [greaterOrEqual()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-greaterorequal)
    GreaterOrEqual {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [lesser()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-lesser)
    Lesser {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [lesserOrEqual()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-lesserorequal)
    LesserOrEqual {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [notEqual()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-notequal)
    NotEqual {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Unary element-wise (MLOperatorOptions) ----------
    /// [abs()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-abs)
    Abs {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [ceil()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-ceil)
    Ceil {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [cos()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-cos)
    Cos {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [exp()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-exp)
    Exp {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [floor()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-floor)
    Floor {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [log()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-log)
    Log {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [neg()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-neg)
    Neg {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [relu()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-relu)
    Relu {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [sigmoid()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-sigmoid)
    Sigmoid {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [sin()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-sin)
    Sin {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [sqrt()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-sqrt)
    Sqrt {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [tan()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-tan)
    Tan {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [tanh()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-tanh)
    Tanh {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [erf()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-erf)
    Erf {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [reciprocal()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reciprocal)
    Reciprocal {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [sign()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-sign)
    Sign {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    // ---------- Logical (MLOperatorOptions) ----------
    /// [logicalAnd()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-logicaland)
    LogicalAnd {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [logicalOr()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-logicalor)
    LogicalOr {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [logicalNot()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-logicalnot)
    LogicalNot {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [logicalXor()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-logicalxor)
    LogicalXor {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Conditional / identity ----------
    /// [where()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-where)
    Where {
        condition: OperandIndex,
        true_value: OperandIndex,
        false_value: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [identity()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-identity)
    Identity {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- ArgMin / ArgMax ----------
    /// [argMin()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-argmin)
    ArgMin {
        input: OperandIndex,
        options: Option<MLArgMinMaxOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [argMax()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-argmax)
    ArgMax {
        input: OperandIndex,
        options: Option<MLArgMinMaxOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- BatchNormalization ----------
    /// [batchNormalization()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-batchnormalization)
    BatchNormalization {
        input: OperandIndex,
        mean: OperandIndex,
        variance: OperandIndex,
        options: Option<MLBatchNormalizationOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Cast ----------
    /// [cast()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-cast)
    Cast {
        input: OperandIndex,
        options: Option<MLCastOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Clamp ----------
    /// [clamp()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-clamp)
    Clamp {
        input: OperandIndex,
        options: Option<MLClampOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Constant (no input operands) ----------
    /// [constant()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-constant)
    Constant {
        options: Option<MLConstantOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Conv2d ----------
    /// [conv2d()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-conv2d)
    Conv2d {
        input: OperandIndex,
        filter: OperandIndex,
        options: Option<MLConv2dOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- ConvTranspose2d ----------
    /// [convTranspose2d()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-convtranspose2d)
    ConvTranspose2d {
        input: OperandIndex,
        filter: OperandIndex,
        options: Option<MLConvTranspose2dOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Concat ----------
    /// [concat()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-concat)
    Concat {
        inputs: Vec<OperandIndex>,
        options: Option<MLConcatOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- CumulativeSum ----------
    /// [cumulativeSum()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-cumulativesum)
    CumulativeSum {
        input: OperandIndex,
        options: Option<MLCumulativeSumOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Expand ----------
    /// [expand()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-expand)
    Expand {
        input: OperandIndex,
        options: Option<MLExpandOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Elu ----------
    /// [elu()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-elu)
    Elu {
        input: OperandIndex,
        options: Option<MLEluOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Gather / GatherElements ----------
    /// [gather()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-gather)
    Gather {
        input: OperandIndex,
        indices: OperandIndex,
        options: Option<MLGatherOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [gatherElements()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-gatherelements)
    GatherElements {
        input: OperandIndex,
        indices: OperandIndex,
        options: Option<MLGatherOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Gemm ----------
    /// [gemm()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-gemm)
    Gemm {
        a: OperandIndex,
        b: OperandIndex,
        options: Option<MLGemmOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- GRU ----------
    /// [gru()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-gru)
    Gru {
        input: OperandIndex,
        weight: OperandIndex,
        recurrence: OperandIndex,
        options: Option<MLGruOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [gruCell()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-grucell)
    GruCell {
        input: OperandIndex,
        weight: OperandIndex,
        recurrence: OperandIndex,
        hidden_state: OperandIndex,
        options: Option<MLGruCellOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- HardSigmoid / HardSwish ----------
    /// [hardSigmoid()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-hardsigmoid)
    HardSigmoid {
        input: OperandIndex,
        options: Option<MLHardSigmoidOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [hardSwish()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-hardswish)
    HardSwish {
        input: OperandIndex,
        options: Option<MLHardSwishOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- InstanceNormalization ----------
    /// [instanceNormalization()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-instancenormalization)
    InstanceNormalization {
        input: OperandIndex,
        options: Option<MLInstanceNormalizationOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- LayerNormalization ----------
    /// [layerNormalization()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-layernormalization)
    LayerNormalization {
        input: OperandIndex,
        options: Option<MLLayerNormalizationOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- LeakyRelu ----------
    /// [leakyRelu()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-leakyrelu)
    LeakyRelu {
        input: OperandIndex,
        options: Option<MLLeakyReluOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Linear ----------
    /// [linear()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-linear)
    Linear {
        input: OperandIndex,
        options: Option<MLLinearOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- LSTM ----------
    /// [lstm()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-lstm)
    Lstm {
        input: OperandIndex,
        weight: OperandIndex,
        recurrence: OperandIndex,
        options: Option<MLLstmOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [lstmCell()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-lstmcell)
    LstmCell {
        input: OperandIndex,
        weight: OperandIndex,
        recurrence: OperandIndex,
        options: Option<MLLstmCellOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Pad ----------
    /// [pad()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-pad)
    Pad {
        input: OperandIndex,
        options: Option<MLPadOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Pooling ----------
    /// [averagePool2d()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-averagepool2d)
    AveragePool2d {
        input: OperandIndex,
        options: Option<MLPool2dOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [maxPool2d()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-maxpool2d)
    MaxPool2d {
        input: OperandIndex,
        options: Option<MLPool2dOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [l2Pool2d()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-l2pool2d)
    L2Pool2d {
        input: OperandIndex,
        options: Option<MLPool2dOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// Global average pooling (same options as pool2d; see spec table § 7.3).
    GlobalAveragePool {
        input: OperandIndex,
        options: Option<MLPool2dOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// Global max pooling (same options as pool2d; see spec table § 7.3).
    GlobalMaxPool {
        input: OperandIndex,
        options: Option<MLPool2dOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Reduction ----------
    /// [reduceSum()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducesum)
    ReduceSum {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [reduceMean()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducemean)
    ReduceMean {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [reduceMax()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducemax)
    ReduceMax {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [reduceMin()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducemin)
    ReduceMin {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [reduceProduct()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reduceproduct)
    ReduceProduct {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [reduceL1()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducel1)
    ReduceL1 {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [reduceL2()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducel2)
    ReduceL2 {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [reduceLogSum()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducelogsum)
    ReduceLogSum {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [reduceLogSumExp()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducelogsumexp)
    ReduceLogSumExp {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [reduceSumSquare()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducesumsquare)
    ReduceSumSquare {
        input: OperandIndex,
        options: Option<MLReduceOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Reshape ----------
    /// [reshape()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reshape)
    Reshape {
        input: OperandIndex,
        options: Option<MLReshapeOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Resample2d ----------
    /// [resample2d()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-resample2d)
    Resample2d {
        input: OperandIndex,
        options: Option<MLResample2dOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Reverse ----------
    /// [reverse()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reverse)
    Reverse {
        input: OperandIndex,
        options: Option<MLReverseOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- ScatterElements ----------
    /// [scatterElements()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-scatterelements)
    ScatterElements {
        input: OperandIndex,
        indices: OperandIndex,
        updates: OperandIndex,
        options: Option<MLScatterOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Softmax ----------
    /// [softmax()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-softmax)
    Softmax {
        input: OperandIndex,
        options: Option<MLSoftmaxOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Slice ----------
    /// [slice()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-slice)
    Slice {
        input: OperandIndex,
        options: Option<MLSliceOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Split ----------
    /// [split()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-split)
    Split {
        input: OperandIndex,
        options: Option<MLSplitOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Transpose ----------
    /// [transpose()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-transpose)
    Transpose {
        input: OperandIndex,
        options: Option<MLTransposeOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Squeeze / Unsqueeze (emulation) ----------
    /// [squeeze()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-squeeze) (§ 11 Operator Emulation)
    Squeeze {
        input: OperandIndex,
        options: Option<MLSqueezeOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [unsqueeze()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-unsqueeze) (§ 11 Operator Emulation)
    Unsqueeze {
        input: OperandIndex,
        options: Option<MLUnsqueezeOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Tile ----------
    /// [tile()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-tile)
    Tile {
        input: OperandIndex,
        options: Option<MLTileOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Triangular ----------
    /// [triangular()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-triangular)
    Triangular {
        input: OperandIndex,
        options: Option<MLTriangularOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Prelu (binary, MLOperatorOptions) ----------
    /// [prelu()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-prelu)
    Prelu {
        input: OperandIndex,
        slope: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- QuantizeLinear / DequantizeLinear ----------
    /// [quantizeLinear()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-quantizelinear)
    QuantizeLinear {
        input: OperandIndex,
        scale: OperandIndex,
        zero_point: Option<OperandIndex>,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [dequantizeLinear()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-dequantizelinear)
    DequantizeLinear {
        input: OperandIndex,
        scale: OperandIndex,
        zero_point: Option<OperandIndex>,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Activation (softplus, softsign, gelu - MLOperatorOptions) ----------
    /// [softplus()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-softplus)
    Softplus {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [softsign()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-softsign)
    Softsign {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [gelu()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-gelu)
    Gelu {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Shape (interchange / internal) ----------
    /// Shape operator (interchange / internal; see spec § 7.3).
    Shape {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Optional / not yet in OperatorOptions ----------
    /// [scatterND()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-scatternd)
    ScatterND {
        input: OperandIndex,
        indices: OperandIndex,
        updates: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [gatherND()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-gathernd)
    GatherND {
        input: OperandIndex,
        indices: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [isNaN()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-isnan)
    IsNaN {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [isInfinite()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-isinfinite)
    IsInfinite {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [roundEven()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-roundeven)
    RoundEven {
        input: OperandIndex,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },
}

// ---------------------------------------------------------------------------
// Legacy conversion: Operator <-> (op_type, input_operands, attributes)
// ---------------------------------------------------------------------------

impl Operator {
    /// Canonical WebNN operation type string for JSON interchange (e.g. `add`, `batchNormalization`).
    /// Single source of truth shared with [`Operator::to_legacy`] and [`crate::graph::Operation::op_type`].
    pub fn op_type(&self) -> &'static str {
        match self {
            Operator::Add { .. } => "add",
            Operator::Sub { .. } => "sub",
            Operator::Mul { .. } => "mul",
            Operator::Div { .. } => "div",
            Operator::Pow { .. } => "pow",
            Operator::Max { .. } => "max",
            Operator::Min { .. } => "min",
            Operator::Matmul { .. } => "matmul",
            Operator::Equal { .. } => "equal",
            Operator::NotEqual { .. } => "notEqual",
            Operator::Greater { .. } => "greater",
            Operator::GreaterOrEqual { .. } => "greaterOrEqual",
            Operator::Lesser { .. } => "lesser",
            Operator::LesserOrEqual { .. } => "lesserOrEqual",
            Operator::Abs { .. } => "abs",
            Operator::Ceil { .. } => "ceil",
            Operator::Cos { .. } => "cos",
            Operator::Exp { .. } => "exp",
            Operator::Floor { .. } => "floor",
            Operator::Log { .. } => "log",
            Operator::Neg { .. } => "neg",
            Operator::Sin { .. } => "sin",
            Operator::Tan { .. } => "tan",
            Operator::Erf { .. } => "erf",
            Operator::Identity { .. } => "identity",
            Operator::Reciprocal { .. } => "reciprocal",
            Operator::Sign { .. } => "sign",
            Operator::Sqrt { .. } => "sqrt",
            Operator::Tanh { .. } => "tanh",
            Operator::Relu { .. } => "relu",
            Operator::Sigmoid { .. } => "sigmoid",
            Operator::LogicalAnd { .. } => "logicalAnd",
            Operator::LogicalOr { .. } => "logicalOr",
            Operator::LogicalNot { .. } => "logicalNot",
            Operator::LogicalXor { .. } => "logicalXor",
            Operator::Where { .. } => "where",
            Operator::ArgMax { .. } => "argMax",
            Operator::ArgMin { .. } => "argMin",
            Operator::BatchNormalization { .. } => "batchNormalization",
            Operator::Cast { .. } => "cast",
            Operator::Clamp { .. } => "clamp",
            Operator::Constant { .. } => "constant",
            Operator::Conv2d { .. } => "conv2d",
            Operator::ConvTranspose2d { .. } => "convTranspose2d",
            Operator::Concat { .. } => "concat",
            Operator::CumulativeSum { .. } => "cumulativeSum",
            Operator::Expand { .. } => "expand",
            Operator::Elu { .. } => "elu",
            Operator::Gather { .. } => "gather",
            Operator::GatherElements { .. } => "gatherElements",
            Operator::Gemm { .. } => "gemm",
            Operator::Gru { .. } => "gru",
            Operator::GruCell { .. } => "gruCell",
            Operator::HardSigmoid { .. } => "hardSigmoid",
            Operator::HardSwish { .. } => "hardSwish",
            Operator::InstanceNormalization { .. } => "instanceNormalization",
            Operator::LayerNormalization { .. } => "layerNormalization",
            Operator::LeakyRelu { .. } => "leakyRelu",
            Operator::Linear { .. } => "linear",
            Operator::Lstm { .. } => "lstm",
            Operator::LstmCell { .. } => "lstmCell",
            Operator::Pad { .. } => "pad",
            Operator::AveragePool2d { .. } => "averagePool2d",
            Operator::MaxPool2d { .. } => "maxPool2d",
            Operator::L2Pool2d { .. } => "l2Pool2d",
            Operator::GlobalAveragePool { .. } => "globalAveragePool",
            Operator::GlobalMaxPool { .. } => "globalMaxPool",
            Operator::ReduceSum { .. } => "reduceSum",
            Operator::ReduceMean { .. } => "reduceMean",
            Operator::ReduceMax { .. } => "reduceMax",
            Operator::ReduceMin { .. } => "reduceMin",
            Operator::ReduceProduct { .. } => "reduceProduct",
            Operator::ReduceL1 { .. } => "reduceL1",
            Operator::ReduceL2 { .. } => "reduceL2",
            Operator::ReduceLogSum { .. } => "reduceLogSum",
            Operator::ReduceLogSumExp { .. } => "reduceLogSumExp",
            Operator::ReduceSumSquare { .. } => "reduceSumSquare",
            Operator::Reshape { .. } => "reshape",
            Operator::Resample2d { .. } => "resample2d",
            Operator::Reverse { .. } => "reverse",
            Operator::ScatterElements { .. } => "scatterElements",
            Operator::Softmax { .. } => "softmax",
            Operator::Slice { .. } => "slice",
            Operator::Split { .. } => "split",
            Operator::Transpose { .. } => "transpose",
            Operator::Squeeze { .. } => "squeeze",
            Operator::Unsqueeze { .. } => "unsqueeze",
            Operator::Tile { .. } => "tile",
            Operator::Triangular { .. } => "triangular",
            Operator::Prelu { .. } => "prelu",
            Operator::QuantizeLinear { .. } => "quantizeLinear",
            Operator::DequantizeLinear { .. } => "dequantizeLinear",
            Operator::Softplus { .. } => "softplus",
            Operator::Softsign { .. } => "softsign",
            Operator::Gelu { .. } => "gelu",
            Operator::Shape { .. } => "shape",
            Operator::ScatterND { .. } => "scatterND",
            Operator::GatherND { .. } => "gatherND",
            Operator::IsNaN { .. } => "isNaN",
            Operator::IsInfinite { .. } => "isInfinite",
            Operator::RoundEven { .. } => "roundEven",
        }
    }

    /// Output operand id(s) recorded for this operation (same order as WebNN builder results).
    pub fn outputs(&self) -> &[OperandIndex] {
        match self {
            Operator::Add { outputs, .. } => outputs,
            Operator::Sub { outputs, .. } => outputs,
            Operator::Mul { outputs, .. } => outputs,
            Operator::Div { outputs, .. } => outputs,
            Operator::Pow { outputs, .. } => outputs,
            Operator::Max { outputs, .. } => outputs,
            Operator::Min { outputs, .. } => outputs,
            Operator::Matmul { outputs, .. } => outputs,
            Operator::Equal { outputs, .. } => outputs,
            Operator::Greater { outputs, .. } => outputs,
            Operator::GreaterOrEqual { outputs, .. } => outputs,
            Operator::Lesser { outputs, .. } => outputs,
            Operator::LesserOrEqual { outputs, .. } => outputs,
            Operator::NotEqual { outputs, .. } => outputs,
            Operator::Abs { outputs, .. } => outputs,
            Operator::Ceil { outputs, .. } => outputs,
            Operator::Cos { outputs, .. } => outputs,
            Operator::Exp { outputs, .. } => outputs,
            Operator::Floor { outputs, .. } => outputs,
            Operator::Log { outputs, .. } => outputs,
            Operator::Neg { outputs, .. } => outputs,
            Operator::Relu { outputs, .. } => outputs,
            Operator::Sigmoid { outputs, .. } => outputs,
            Operator::Sin { outputs, .. } => outputs,
            Operator::Sqrt { outputs, .. } => outputs,
            Operator::Tan { outputs, .. } => outputs,
            Operator::Tanh { outputs, .. } => outputs,
            Operator::Erf { outputs, .. } => outputs,
            Operator::Reciprocal { outputs, .. } => outputs,
            Operator::Sign { outputs, .. } => outputs,
            Operator::LogicalAnd { outputs, .. } => outputs,
            Operator::LogicalOr { outputs, .. } => outputs,
            Operator::LogicalNot { outputs, .. } => outputs,
            Operator::LogicalXor { outputs, .. } => outputs,
            Operator::Where { outputs, .. } => outputs,
            Operator::Identity { outputs, .. } => outputs,
            Operator::ArgMin { outputs, .. } => outputs,
            Operator::ArgMax { outputs, .. } => outputs,
            Operator::BatchNormalization { outputs, .. } => outputs,
            Operator::Cast { outputs, .. } => outputs,
            Operator::Clamp { outputs, .. } => outputs,
            Operator::Constant { outputs, .. } => outputs,
            Operator::Conv2d { outputs, .. } => outputs,
            Operator::ConvTranspose2d { outputs, .. } => outputs,
            Operator::Concat { outputs, .. } => outputs,
            Operator::CumulativeSum { outputs, .. } => outputs,
            Operator::Expand { outputs, .. } => outputs,
            Operator::Elu { outputs, .. } => outputs,
            Operator::Gather { outputs, .. } => outputs,
            Operator::GatherElements { outputs, .. } => outputs,
            Operator::Gemm { outputs, .. } => outputs,
            Operator::Gru { outputs, .. } => outputs,
            Operator::GruCell { outputs, .. } => outputs,
            Operator::HardSigmoid { outputs, .. } => outputs,
            Operator::HardSwish { outputs, .. } => outputs,
            Operator::InstanceNormalization { outputs, .. } => outputs,
            Operator::LayerNormalization { outputs, .. } => outputs,
            Operator::LeakyRelu { outputs, .. } => outputs,
            Operator::Linear { outputs, .. } => outputs,
            Operator::Lstm { outputs, .. } => outputs,
            Operator::LstmCell { outputs, .. } => outputs,
            Operator::Pad { outputs, .. } => outputs,
            Operator::AveragePool2d { outputs, .. } => outputs,
            Operator::MaxPool2d { outputs, .. } => outputs,
            Operator::L2Pool2d { outputs, .. } => outputs,
            Operator::GlobalAveragePool { outputs, .. } => outputs,
            Operator::GlobalMaxPool { outputs, .. } => outputs,
            Operator::ReduceSum { outputs, .. } => outputs,
            Operator::ReduceMean { outputs, .. } => outputs,
            Operator::ReduceMax { outputs, .. } => outputs,
            Operator::ReduceMin { outputs, .. } => outputs,
            Operator::ReduceProduct { outputs, .. } => outputs,
            Operator::ReduceL1 { outputs, .. } => outputs,
            Operator::ReduceL2 { outputs, .. } => outputs,
            Operator::ReduceLogSum { outputs, .. } => outputs,
            Operator::ReduceLogSumExp { outputs, .. } => outputs,
            Operator::ReduceSumSquare { outputs, .. } => outputs,
            Operator::Reshape { outputs, .. } => outputs,
            Operator::Resample2d { outputs, .. } => outputs,
            Operator::Reverse { outputs, .. } => outputs,
            Operator::ScatterElements { outputs, .. } => outputs,
            Operator::Softmax { outputs, .. } => outputs,
            Operator::Slice { outputs, .. } => outputs,
            Operator::Split { outputs, .. } => outputs,
            Operator::Transpose { outputs, .. } => outputs,
            Operator::Squeeze { outputs, .. } => outputs,
            Operator::Unsqueeze { outputs, .. } => outputs,
            Operator::Tile { outputs, .. } => outputs,
            Operator::Triangular { outputs, .. } => outputs,
            Operator::Prelu { outputs, .. } => outputs,
            Operator::QuantizeLinear { outputs, .. } => outputs,
            Operator::DequantizeLinear { outputs, .. } => outputs,
            Operator::Softplus { outputs, .. } => outputs,
            Operator::Softsign { outputs, .. } => outputs,
            Operator::Gelu { outputs, .. } => outputs,
            Operator::Shape { outputs, .. } => outputs,
            Operator::ScatterND { outputs, .. } => outputs,
            Operator::GatherND { outputs, .. } => outputs,
            Operator::IsNaN { outputs, .. } => outputs,
            Operator::IsInfinite { outputs, .. } => outputs,
            Operator::RoundEven { outputs, .. } => outputs,
        }
    }

    /// WebNN `label` from the typed options for this operator (empty string if unset).
    pub fn label(&self) -> &str {
        macro_rules! opt_label {
            ($opt:expr) => {
                $opt.as_ref().map(|o| o.label.as_str()).unwrap_or("")
            };
        }
        match self {
            Operator::Add { options, .. }
            | Operator::Sub { options, .. }
            | Operator::Mul { options, .. }
            | Operator::Div { options, .. }
            | Operator::Pow { options, .. }
            | Operator::Max { options, .. }
            | Operator::Min { options, .. }
            | Operator::Matmul { options, .. }
            | Operator::Equal { options, .. }
            | Operator::NotEqual { options, .. }
            | Operator::Greater { options, .. }
            | Operator::GreaterOrEqual { options, .. }
            | Operator::Lesser { options, .. }
            | Operator::LesserOrEqual { options, .. }
            | Operator::Abs { options, .. }
            | Operator::Ceil { options, .. }
            | Operator::Cos { options, .. }
            | Operator::Exp { options, .. }
            | Operator::Floor { options, .. }
            | Operator::Log { options, .. }
            | Operator::Neg { options, .. }
            | Operator::Relu { options, .. }
            | Operator::Sigmoid { options, .. }
            | Operator::Sin { options, .. }
            | Operator::Sqrt { options, .. }
            | Operator::Tan { options, .. }
            | Operator::Tanh { options, .. }
            | Operator::Erf { options, .. }
            | Operator::Reciprocal { options, .. }
            | Operator::Sign { options, .. }
            | Operator::LogicalAnd { options, .. }
            | Operator::LogicalOr { options, .. }
            | Operator::LogicalNot { options, .. }
            | Operator::LogicalXor { options, .. }
            | Operator::Where { options, .. }
            | Operator::Identity { options, .. }
            | Operator::Prelu { options, .. }
            | Operator::QuantizeLinear { options, .. }
            | Operator::DequantizeLinear { options, .. }
            | Operator::Softplus { options, .. }
            | Operator::Softsign { options, .. }
            | Operator::Gelu { options, .. }
            | Operator::Shape { options, .. }
            | Operator::ScatterND { options, .. }
            | Operator::GatherND { options, .. }
            | Operator::IsNaN { options, .. }
            | Operator::IsInfinite { options, .. }
            | Operator::RoundEven { options, .. } => opt_label!(options),

            Operator::ArgMin { options, .. } | Operator::ArgMax { options, .. } => {
                opt_label!(options)
            }

            Operator::BatchNormalization { options, .. } => opt_label!(options),
            Operator::Cast { options, .. } => opt_label!(options),
            Operator::Clamp { options, .. } => opt_label!(options),
            Operator::Constant { options, .. } => opt_label!(options),
            Operator::Conv2d { options, .. } => opt_label!(options),
            Operator::ConvTranspose2d { options, .. } => opt_label!(options),
            Operator::Concat { options, .. } => opt_label!(options),
            Operator::CumulativeSum { options, .. } => opt_label!(options),
            Operator::Expand { options, .. } => opt_label!(options),
            Operator::Elu { options, .. } => opt_label!(options),
            Operator::Gather { options, .. } | Operator::GatherElements { options, .. } => {
                opt_label!(options)
            }
            Operator::Gemm { options, .. } => opt_label!(options),
            Operator::Gru { options, .. } => opt_label!(options),
            Operator::GruCell { options, .. } => opt_label!(options),
            Operator::HardSigmoid { options, .. } => opt_label!(options),
            Operator::HardSwish { options, .. } => opt_label!(options),
            Operator::InstanceNormalization { options, .. } => opt_label!(options),
            Operator::LayerNormalization { options, .. } => opt_label!(options),
            Operator::LeakyRelu { options, .. } => opt_label!(options),
            Operator::Linear { options, .. } => opt_label!(options),
            Operator::Lstm { options, .. } => opt_label!(options),
            Operator::LstmCell { options, .. } => opt_label!(options),
            Operator::Pad { options, .. } => opt_label!(options),
            Operator::AveragePool2d { options, .. }
            | Operator::MaxPool2d { options, .. }
            | Operator::L2Pool2d { options, .. }
            | Operator::GlobalAveragePool { options, .. }
            | Operator::GlobalMaxPool { options, .. } => opt_label!(options),
            Operator::ReduceSum { options, .. }
            | Operator::ReduceMean { options, .. }
            | Operator::ReduceMax { options, .. }
            | Operator::ReduceMin { options, .. }
            | Operator::ReduceProduct { options, .. }
            | Operator::ReduceL1 { options, .. }
            | Operator::ReduceL2 { options, .. }
            | Operator::ReduceLogSum { options, .. }
            | Operator::ReduceLogSumExp { options, .. }
            | Operator::ReduceSumSquare { options, .. } => opt_label!(options),
            Operator::Reshape { options, .. } => opt_label!(options),
            Operator::Resample2d { options, .. } => opt_label!(options),
            Operator::Reverse { options, .. } => opt_label!(options),
            Operator::ScatterElements { options, .. } => opt_label!(options),
            Operator::Softmax { options, .. } => opt_label!(options),
            Operator::Slice { options, .. } => opt_label!(options),
            Operator::Split { options, .. } => opt_label!(options),
            Operator::Transpose { options, .. } => opt_label!(options),
            Operator::Squeeze { options, .. } => opt_label!(options),
            Operator::Unsqueeze { options, .. } => opt_label!(options),
            Operator::Tile { options, .. } => opt_label!(options),
            Operator::Triangular { options, .. } => opt_label!(options),
        }
    }

    /// Converts this operator to the legacy triple used by JSON and existing consumers.
    /// Returns `(op_type, input_operands, attributes)`.
    pub fn to_legacy(&self) -> (String, Vec<u32>, OperatorOptions) {
        let tag = self.op_type().to_string();
        use OperatorOptions as OO;
        match self {
            Operator::Add { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Sub { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Mul { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Div { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Pow { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Max { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Min { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Matmul { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Equal { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::NotEqual { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Greater { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::GreaterOrEqual { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Lesser { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::LesserOrEqual { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Abs { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Ceil { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Cos { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Exp { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Floor { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Log { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Neg { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Sin { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Tan { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Erf { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Identity { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Reciprocal { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Sign { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Sqrt { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Tanh { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Relu { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Sigmoid { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::LogicalAnd { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::LogicalOr { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::LogicalNot { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::LogicalXor { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Where {
                condition,
                true_value,
                false_value,
                options,
                ..
            } => (
                tag.clone(),
                vec![*condition, *true_value, *false_value],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::ArgMax { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::ArgMinMax(options.clone().unwrap_or_default()),
            ),
            Operator::ArgMin { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::ArgMinMax(options.clone().unwrap_or_default()),
            ),
            Operator::BatchNormalization {
                input,
                mean,
                variance,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *mean, *variance],
                OO::BatchNormalization(options.clone().unwrap_or_default()),
            ),
            Operator::Cast { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Cast(options.clone().unwrap_or_default()),
            ),
            Operator::Clamp { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Clamp(options.clone().unwrap_or_default()),
            ),
            Operator::Constant { options, .. } => (
                tag.clone(),
                vec![],
                OO::Constant(options.clone().unwrap_or_default()),
            ),
            Operator::Conv2d {
                input,
                filter,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *filter],
                OO::Conv2d(options.clone().unwrap_or_default()),
            ),
            Operator::ConvTranspose2d {
                input,
                filter,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *filter],
                OO::ConvTranspose2d(options.clone().unwrap_or_default()),
            ),
            Operator::Concat {
                inputs, options, ..
            } => (
                tag.clone(),
                inputs.clone(),
                OO::Concat(options.clone().unwrap_or_default()),
            ),
            Operator::CumulativeSum { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::CumulativeSum(options.clone().unwrap_or_default()),
            ),
            Operator::Expand { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Expand(options.clone().unwrap_or_default()),
            ),
            Operator::Elu { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Elu(options.clone().unwrap_or_default()),
            ),
            Operator::Gather {
                input,
                indices,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *indices],
                OO::Gather(options.clone().unwrap_or_default()),
            ),
            Operator::GatherElements {
                input,
                indices,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *indices],
                OO::Gather(options.clone().unwrap_or_default()),
            ),
            Operator::Gemm { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Gemm(options.clone().unwrap_or_default()),
            ),
            Operator::Gru {
                input,
                weight,
                recurrence,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *weight, *recurrence],
                OO::Gru(options.clone().unwrap_or_default()),
            ),
            Operator::GruCell {
                input,
                weight,
                recurrence,
                hidden_state,
                options,
                ..
            } => {
                let o = options.clone().unwrap_or_default();
                let mut ids = vec![*input, *weight, *recurrence, *hidden_state];
                if let Some(id) = o.bias {
                    ids.push(id);
                }
                if let Some(id) = o.recurrent_bias {
                    ids.push(id);
                }
                (tag.clone(), ids, OO::GruCell(o))
            }
            Operator::HardSigmoid { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::HardSigmoid(options.clone().unwrap_or_default()),
            ),
            Operator::HardSwish { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::HardSwish(options.clone().unwrap_or_default()),
            ),
            Operator::InstanceNormalization { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::InstanceNormalization(options.clone().unwrap_or_default()),
            ),
            Operator::LayerNormalization { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::LayerNormalization(options.clone().unwrap_or_default()),
            ),
            Operator::LeakyRelu { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::LeakyRelu(options.clone().unwrap_or_default()),
            ),
            Operator::Linear { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Linear(options.clone().unwrap_or_default()),
            ),
            Operator::Lstm {
                input,
                weight,
                recurrence,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *weight, *recurrence],
                OO::Lstm(options.clone().unwrap_or_default()),
            ),
            Operator::LstmCell {
                input,
                weight,
                recurrence,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *weight, *recurrence],
                OO::LstmCell(options.clone().unwrap_or_default()),
            ),
            Operator::Pad { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Pad(options.clone().unwrap_or_default()),
            ),
            Operator::AveragePool2d { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Pool2d(options.clone().unwrap_or_default()),
            ),
            Operator::MaxPool2d { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Pool2d(options.clone().unwrap_or_default()),
            ),
            Operator::L2Pool2d { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Pool2d(options.clone().unwrap_or_default()),
            ),
            Operator::GlobalAveragePool { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Pool2d(options.clone().unwrap_or_default()),
            ),
            Operator::GlobalMaxPool { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Pool2d(options.clone().unwrap_or_default()),
            ),
            Operator::ReduceSum { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operator::ReduceMean { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operator::ReduceMax { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operator::ReduceMin { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operator::ReduceProduct { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operator::ReduceL1 { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operator::ReduceL2 { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operator::ReduceLogSum { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operator::ReduceLogSumExp { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operator::ReduceSumSquare { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operator::Reshape { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reshape(options.clone().unwrap_or_default()),
            ),
            Operator::Resample2d { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Resample2d(options.clone().unwrap_or_default()),
            ),
            Operator::Reverse { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reverse(options.clone().unwrap_or_default()),
            ),
            Operator::ScatterElements {
                input,
                indices,
                updates,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *indices, *updates],
                OO::ScatterElements(options.clone().unwrap_or_default()),
            ),
            Operator::Softmax { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Softmax(options.clone().unwrap_or_default()),
            ),
            Operator::Slice { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Slice(options.clone().unwrap_or_default()),
            ),
            Operator::Split { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Split(options.clone().unwrap_or_default()),
            ),
            Operator::Transpose { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Transpose(options.clone().unwrap_or_default()),
            ),
            Operator::Squeeze { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Squeeze(options.clone().unwrap_or_default()),
            ),
            Operator::Unsqueeze { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Unsqueeze(options.clone().unwrap_or_default()),
            ),
            Operator::Tile { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Tile(options.clone().unwrap_or_default()),
            ),
            Operator::Triangular { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Triangular(options.clone().unwrap_or_default()),
            ),
            Operator::Prelu {
                input,
                slope,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *slope],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::QuantizeLinear {
                input,
                scale,
                zero_point,
                options,
                ..
            } => {
                let mut inps = vec![*input, *scale];
                if let Some(z) = zero_point {
                    inps.push(*z);
                }
                (
                    tag.clone(),
                    inps,
                    OO::Operator(options.clone().unwrap_or_default()),
                )
            }
            Operator::DequantizeLinear {
                input,
                scale,
                zero_point,
                options,
                ..
            } => {
                let mut inps = vec![*input, *scale];
                if let Some(z) = zero_point {
                    inps.push(*z);
                }
                (
                    tag.clone(),
                    inps,
                    OO::Operator(options.clone().unwrap_or_default()),
                )
            }
            Operator::Softplus { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Softsign { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Gelu { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::Shape { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::ScatterND {
                input,
                indices,
                updates,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *indices, *updates],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::GatherND {
                input,
                indices,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *indices],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::IsNaN { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::IsInfinite { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operator::RoundEven { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
        }
    }

    /// Parses WebNN interchange: builder/JSON `op_type` string (e.g. camelCase `batchNormalization`
    /// or lowercase `add`), operand indices in spec order, and typed [`OperatorOptions`].
    ///
    /// This is the **supported** way to construct an [`Operator`] from serialized graphs (JSON, WPT,
    /// etc.). Prefer constructing enum variants directly only in native Rust graph builders.
    ///
    /// Returns `None` if `op_type` is unknown or operand lengths do not match.
    pub fn from_operator_options(
        op_type: &str,
        input_operands: &[u32],
        attributes: &OperatorOptions,
        outputs: &[OperandIndex],
    ) -> Option<Self> {
        fn at(inputs: &[u32], i: usize) -> Option<u32> {
            inputs.get(i).copied()
        }
        let n = op_type.trim();
        match n {
            "add" if input_operands.len() >= 2 => Some(Operator::Add {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "sub" if input_operands.len() >= 2 => Some(Operator::Sub {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "mul" if input_operands.len() >= 2 => Some(Operator::Mul {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "div" if input_operands.len() >= 2 => Some(Operator::Div {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "pow" if input_operands.len() >= 2 => Some(Operator::Pow {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "max" if input_operands.len() >= 2 => Some(Operator::Max {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "min" if input_operands.len() >= 2 => Some(Operator::Min {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "matmul" if input_operands.len() >= 2 => Some(Operator::Matmul {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "equal" if input_operands.len() >= 2 => Some(Operator::Equal {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "notEqual" if input_operands.len() >= 2 => Some(Operator::NotEqual {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "greater" if input_operands.len() >= 2 => Some(Operator::Greater {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "greaterOrEqual" if input_operands.len() >= 2 => Some(Operator::GreaterOrEqual {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "lesser" if input_operands.len() >= 2 => Some(Operator::Lesser {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "lesserOrEqual" if input_operands.len() >= 2 => Some(Operator::LesserOrEqual {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "abs" | "ceil" | "cos" | "exp" | "floor" | "log" | "neg" | "sin" | "tan" | "erf"
            | "identity" | "reciprocal" | "sign" | "sqrt" | "tanh" | "relu" | "sigmoid"
            | "logicalNot"
                if !input_operands.is_empty() =>
            {
                let input = at(input_operands, 0)?;
                let opts = attributes.as_operator().cloned();
                Some(match n {
                    "abs" => Operator::Abs {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "ceil" => Operator::Ceil {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "cos" => Operator::Cos {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "exp" => Operator::Exp {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "floor" => Operator::Floor {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "log" => Operator::Log {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "neg" => Operator::Neg {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "sin" => Operator::Sin {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "tan" => Operator::Tan {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "erf" => Operator::Erf {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "identity" => Operator::Identity {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reciprocal" => Operator::Reciprocal {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "sign" => Operator::Sign {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "sqrt" => Operator::Sqrt {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "tanh" => Operator::Tanh {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "relu" => Operator::Relu {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "sigmoid" => Operator::Sigmoid {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "logicalNot" => Operator::LogicalNot {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    _ => return None,
                })
            }
            "logicalAnd" if input_operands.len() >= 2 => Some(Operator::LogicalAnd {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "logicalOr" if input_operands.len() >= 2 => Some(Operator::LogicalOr {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "logicalXor" if input_operands.len() >= 2 => Some(Operator::LogicalXor {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "where" if input_operands.len() >= 3 => Some(Operator::Where {
                condition: at(input_operands, 0)?,
                true_value: at(input_operands, 1)?,
                false_value: at(input_operands, 2)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "argMax" if !input_operands.is_empty() => Some(Operator::ArgMax {
                input: at(input_operands, 0)?,
                options: attributes.as_arg_min_max().cloned(),
                outputs: outputs.to_vec(),
            }),
            "argMin" if !input_operands.is_empty() => Some(Operator::ArgMin {
                input: at(input_operands, 0)?,
                options: attributes.as_arg_min_max().cloned(),
                outputs: outputs.to_vec(),
            }),
            "batchNormalization" if input_operands.len() >= 3 => {
                Some(Operator::BatchNormalization {
                    input: at(input_operands, 0)?,
                    mean: at(input_operands, 1)?,
                    variance: at(input_operands, 2)?,
                    options: attributes.as_batch_normalization().cloned(),
                    outputs: outputs.to_vec(),
                })
            }
            "cast" if !input_operands.is_empty() => Some(Operator::Cast {
                input: at(input_operands, 0)?,
                options: attributes.as_cast().cloned(),
                outputs: outputs.to_vec(),
            }),
            "clamp" if !input_operands.is_empty() => Some(Operator::Clamp {
                input: at(input_operands, 0)?,
                options: attributes.as_clamp().cloned(),
                outputs: outputs.to_vec(),
            }),
            "constant" => Some(Operator::Constant {
                options: attributes.as_constant().cloned(),
                outputs: outputs.to_vec(),
            }),
            "conv2d" if input_operands.len() >= 2 => Some(Operator::Conv2d {
                input: at(input_operands, 0)?,
                filter: at(input_operands, 1)?,
                options: attributes.as_conv2d().cloned(),
                outputs: outputs.to_vec(),
            }),
            "convTranspose2d" if input_operands.len() >= 2 => Some(Operator::ConvTranspose2d {
                input: at(input_operands, 0)?,
                filter: at(input_operands, 1)?,
                options: attributes.as_conv_transpose2d().cloned(),
                outputs: outputs.to_vec(),
            }),
            "concat" => Some(Operator::Concat {
                inputs: input_operands.to_vec(),
                options: attributes.as_concat().cloned(),
                outputs: outputs.to_vec(),
            }),
            "cumulativeSum" if !input_operands.is_empty() => Some(Operator::CumulativeSum {
                input: at(input_operands, 0)?,
                options: attributes.as_cumulative_sum().cloned(),
                outputs: outputs.to_vec(),
            }),
            "expand" if !input_operands.is_empty() => Some(Operator::Expand {
                input: at(input_operands, 0)?,
                options: attributes.as_expand().cloned(),
                outputs: outputs.to_vec(),
            }),
            "elu" if !input_operands.is_empty() => Some(Operator::Elu {
                input: at(input_operands, 0)?,
                options: attributes.as_elu().cloned(),
                outputs: outputs.to_vec(),
            }),
            "gather" if input_operands.len() >= 2 => Some(Operator::Gather {
                input: at(input_operands, 0)?,
                indices: at(input_operands, 1)?,
                options: attributes.as_gather().cloned(),
                outputs: outputs.to_vec(),
            }),
            "gatherElements" if input_operands.len() >= 2 => Some(Operator::GatherElements {
                input: at(input_operands, 0)?,
                indices: at(input_operands, 1)?,
                options: attributes.as_gather().cloned(),
                outputs: outputs.to_vec(),
            }),
            "gemm" if input_operands.len() >= 2 => Some(Operator::Gemm {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_gemm().cloned(),
                outputs: outputs.to_vec(),
            }),
            "gru" if input_operands.len() >= 3 => Some(Operator::Gru {
                input: at(input_operands, 0)?,
                weight: at(input_operands, 1)?,
                recurrence: at(input_operands, 2)?,
                options: attributes.as_gru().cloned(),
                outputs: outputs.to_vec(),
            }),
            "gruCell" if input_operands.len() >= 4 => {
                let base = attributes.as_gru_cell().cloned();
                let mut opts = base.clone().unwrap_or_default();
                if input_operands.len() >= 6 {
                    if opts.bias.is_none() {
                        opts.bias = at(input_operands, 4);
                    }
                    if opts.recurrent_bias.is_none() {
                        opts.recurrent_bias = at(input_operands, 5);
                    }
                }
                let options = if base.is_some()
                    || input_operands.len() >= 6
                    || opts != MLGruCellOptions::default()
                {
                    Some(opts)
                } else {
                    None
                };
                Some(Operator::GruCell {
                    input: at(input_operands, 0)?,
                    weight: at(input_operands, 1)?,
                    recurrence: at(input_operands, 2)?,
                    hidden_state: at(input_operands, 3)?,
                    options,
                    outputs: outputs.to_vec(),
                })
            }
            "hardSigmoid" if !input_operands.is_empty() => Some(Operator::HardSigmoid {
                input: at(input_operands, 0)?,
                options: attributes.as_hard_sigmoid().cloned(),
                outputs: outputs.to_vec(),
            }),
            "hardSwish" if !input_operands.is_empty() => Some(Operator::HardSwish {
                input: at(input_operands, 0)?,
                options: attributes.as_hard_swish().cloned(),
                outputs: outputs.to_vec(),
            }),
            "instanceNormalization" if !input_operands.is_empty() => {
                Some(Operator::InstanceNormalization {
                    input: at(input_operands, 0)?,
                    options: attributes.as_instance_normalization().cloned(),
                    outputs: outputs.to_vec(),
                })
            }
            "layerNormalization" if !input_operands.is_empty() => {
                Some(Operator::LayerNormalization {
                    input: at(input_operands, 0)?,
                    options: attributes.as_layer_normalization().cloned(),
                    outputs: outputs.to_vec(),
                })
            }
            "leakyRelu" if !input_operands.is_empty() => Some(Operator::LeakyRelu {
                input: at(input_operands, 0)?,
                options: attributes.as_leaky_relu().cloned(),
                outputs: outputs.to_vec(),
            }),
            "linear" if !input_operands.is_empty() => Some(Operator::Linear {
                input: at(input_operands, 0)?,
                options: attributes.as_linear().cloned(),
                outputs: outputs.to_vec(),
            }),
            "lstm" if input_operands.len() >= 3 => Some(Operator::Lstm {
                input: at(input_operands, 0)?,
                weight: at(input_operands, 1)?,
                recurrence: at(input_operands, 2)?,
                options: attributes.as_lstm().cloned(),
                outputs: outputs.to_vec(),
            }),
            "lstmCell" if input_operands.len() >= 3 => Some(Operator::LstmCell {
                input: at(input_operands, 0)?,
                weight: at(input_operands, 1)?,
                recurrence: at(input_operands, 2)?,
                options: attributes.as_lstm_cell().cloned(),
                outputs: outputs.to_vec(),
            }),
            "pad" if !input_operands.is_empty() => Some(Operator::Pad {
                input: at(input_operands, 0)?,
                options: attributes.as_pad().cloned(),
                outputs: outputs.to_vec(),
            }),
            "averagePool2d" if !input_operands.is_empty() => Some(Operator::AveragePool2d {
                input: at(input_operands, 0)?,
                options: attributes.as_pool2d().cloned(),
                outputs: outputs.to_vec(),
            }),
            "maxPool2d" if !input_operands.is_empty() => Some(Operator::MaxPool2d {
                input: at(input_operands, 0)?,
                options: attributes.as_pool2d().cloned(),
                outputs: outputs.to_vec(),
            }),
            "l2Pool2d" if !input_operands.is_empty() => Some(Operator::L2Pool2d {
                input: at(input_operands, 0)?,
                options: attributes.as_pool2d().cloned(),
                outputs: outputs.to_vec(),
            }),
            "globalAveragePool" if !input_operands.is_empty() => {
                Some(Operator::GlobalAveragePool {
                    input: at(input_operands, 0)?,
                    options: attributes.as_pool2d().cloned(),
                    outputs: outputs.to_vec(),
                })
            }
            "globalMaxPool" if !input_operands.is_empty() => Some(Operator::GlobalMaxPool {
                input: at(input_operands, 0)?,
                options: attributes.as_pool2d().cloned(),
                outputs: outputs.to_vec(),
            }),
            "reduceSum" | "reduceMean" | "reduceMax" | "reduceMin" | "reduceProduct"
            | "reduceL1" | "reduceL2" | "reduceLogSum" | "reduceLogSumExp" | "reduceSumSquare"
                if !input_operands.is_empty() =>
            {
                let input = at(input_operands, 0)?;
                let opts = attributes.as_reduce().cloned();
                Some(match n {
                    "reduceSum" => Operator::ReduceSum {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceMean" => Operator::ReduceMean {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceMax" => Operator::ReduceMax {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceMin" => Operator::ReduceMin {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceProduct" => Operator::ReduceProduct {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceL1" => Operator::ReduceL1 {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceL2" => Operator::ReduceL2 {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceLogSum" => Operator::ReduceLogSum {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceLogSumExp" => Operator::ReduceLogSumExp {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceSumSquare" => Operator::ReduceSumSquare {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    _ => return None,
                })
            }
            "reshape" if !input_operands.is_empty() => Some(Operator::Reshape {
                input: at(input_operands, 0)?,
                options: attributes.as_reshape().cloned(),
                outputs: outputs.to_vec(),
            }),
            "resample2d" if !input_operands.is_empty() => Some(Operator::Resample2d {
                input: at(input_operands, 0)?,
                options: attributes.as_resample2d().cloned(),
                outputs: outputs.to_vec(),
            }),
            "reverse" if !input_operands.is_empty() => Some(Operator::Reverse {
                input: at(input_operands, 0)?,
                options: attributes.as_reverse().cloned(),
                outputs: outputs.to_vec(),
            }),
            "scatterElements" if input_operands.len() >= 3 => Some(Operator::ScatterElements {
                input: at(input_operands, 0)?,
                indices: at(input_operands, 1)?,
                updates: at(input_operands, 2)?,
                options: attributes.as_scatter_elements().cloned(),
                outputs: outputs.to_vec(),
            }),
            "softmax" if !input_operands.is_empty() => Some(Operator::Softmax {
                input: at(input_operands, 0)?,
                options: attributes.as_softmax().cloned(),
                outputs: outputs.to_vec(),
            }),
            "slice" if !input_operands.is_empty() => Some(Operator::Slice {
                input: at(input_operands, 0)?,
                options: attributes.as_slice().cloned(),
                outputs: outputs.to_vec(),
            }),
            "split" if !input_operands.is_empty() => Some(Operator::Split {
                input: at(input_operands, 0)?,
                options: attributes.as_split().cloned(),
                outputs: outputs.to_vec(),
            }),
            "transpose" if !input_operands.is_empty() => Some(Operator::Transpose {
                input: at(input_operands, 0)?,
                options: attributes.as_transpose().cloned(),
                outputs: outputs.to_vec(),
            }),
            "squeeze" if !input_operands.is_empty() => Some(Operator::Squeeze {
                input: at(input_operands, 0)?,
                options: attributes.as_squeeze().cloned(),
                outputs: outputs.to_vec(),
            }),
            "unsqueeze" if !input_operands.is_empty() => Some(Operator::Unsqueeze {
                input: at(input_operands, 0)?,
                options: attributes.as_unsqueeze().cloned(),
                outputs: outputs.to_vec(),
            }),
            "tile" if !input_operands.is_empty() => Some(Operator::Tile {
                input: at(input_operands, 0)?,
                options: attributes.as_tile().cloned(),
                outputs: outputs.to_vec(),
            }),
            "triangular" if !input_operands.is_empty() => Some(Operator::Triangular {
                input: at(input_operands, 0)?,
                options: attributes.as_triangular().cloned(),
                outputs: outputs.to_vec(),
            }),
            "prelu" if input_operands.len() >= 2 => Some(Operator::Prelu {
                input: at(input_operands, 0)?,
                slope: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "quantizeLinear" if input_operands.len() >= 2 => {
                let zero_point = input_operands.get(2).copied();
                Some(Operator::QuantizeLinear {
                    input: at(input_operands, 0)?,
                    scale: at(input_operands, 1)?,
                    zero_point,
                    options: attributes.as_operator().cloned(),
                    outputs: outputs.to_vec(),
                })
            }
            "dequantizeLinear" if input_operands.len() >= 2 => {
                let zero_point = input_operands.get(2).copied();
                Some(Operator::DequantizeLinear {
                    input: at(input_operands, 0)?,
                    scale: at(input_operands, 1)?,
                    zero_point,
                    options: attributes.as_operator().cloned(),
                    outputs: outputs.to_vec(),
                })
            }
            "softplus" | "softsign" | "gelu" | "shape" | "isNaN" | "isInfinite" | "roundEven"
            | "round"
                if !input_operands.is_empty() =>
            {
                let input = at(input_operands, 0)?;
                let opts = attributes.as_operator().cloned();
                Some(match n {
                    "softplus" => Operator::Softplus {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "softsign" => Operator::Softsign {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "gelu" => Operator::Gelu {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "shape" => Operator::Shape {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "isNaN" => Operator::IsNaN {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "isInfinite" => Operator::IsInfinite {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "roundEven" | "round" => Operator::RoundEven {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    _ => return None,
                })
            }
            "scatterND" if input_operands.len() >= 3 => Some(Operator::ScatterND {
                input: at(input_operands, 0)?,
                indices: at(input_operands, 1)?,
                updates: at(input_operands, 2)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "gatherND" if input_operands.len() >= 2 => Some(Operator::GatherND {
                input: at(input_operands, 0)?,
                indices: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            _ => None,
        }
    }

    /// Deprecated: construct [`Operator`] variants directly, or use [`Self::from_operator_options`]
    /// when parsing WebNN JSON (`type` + `input_operands` + tagged `attributes`).
    #[deprecated(
        since = "0.6.0",
        note = "construct Operator variants directly, or call Operator::from_operator_options for JSON interchange"
    )]
    #[inline]
    pub fn from_legacy(
        op_type: &str,
        input_operands: &[u32],
        attributes: &OperatorOptions,
    ) -> Option<Self> {
        Self::from_operator_options(op_type, input_operands, attributes, &[])
    }
}

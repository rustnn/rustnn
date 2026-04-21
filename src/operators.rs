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
//! This module defines the `Operation` enum as the single source of truth for each WebNN
//! operation: each variant carries the builder name, named operand indices (no positional
//! ambiguity), and the corresponding ML*Options struct. All option types and MLDimension
//! are reused from [crate::operator_options].
//!
//! Graph interchange JSON should use [`Operation::from_json_attributes`] (op type string,
//! input/output operand ids, and one attributes object); it parses options and method-level
//! fields and returns a complete [`Operation`].
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

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    operator_enums::MLOperandDataType,
    operator_options::{
        MLArgMinMaxOptions, MLBatchNormalizationOptions, MLClampOptions, MLConstantOptions,
        MLConv2dOptions, MLConvTranspose2dOptions, MLCumulativeSumOptions, MLDimension,
        MLEluOptions, MLGatherOptions, MLGemmOptions, MLGruCellOptions, MLGruOptions,
        MLHardSigmoidOptions, MLInstanceNormalizationOptions, MLLayerNormalizationOptions,
        MLLeakyReluOptions, MLLinearOptions, MLLstmCellOptions, MLLstmOptions, MLOperatorOptions,
        MLPadOptions, MLPool2dOptions, MLReduceOptions, MLResample2dOptions, MLReverseOptions,
        MLScatterOptions, MLSliceOptions, MLSplitOptions, MLSqueezeOptions, MLTransposeOptions,
        MLTriangularOptions, MLUnsqueezeOptions, OperandIndex, OperationExtras, OperatorOptions,
    },
};

// ---------------------------------------------------------------------------
// Operation enum: one variant per WebNN builder
// ---------------------------------------------------------------------------

/// One variant per WebNN graph builder. Each variant has named operand fields and the
/// corresponding options struct, so operand roles are explicit and independent of
/// input_operands order.
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
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
        axis: u32,
        options: Option<MLArgMinMaxOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [argMax()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-argmax)
    ArgMax {
        input: OperandIndex,
        axis: u32,
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
        data_type: MLOperandDataType,
        options: Option<MLOperatorOptions>,
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
    /// [concat()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-concat) —
    /// `axis` is a method parameter; `options` is [`MLOperatorOptions`] (label only).
    Concat {
        inputs: Vec<OperandIndex>,
        axis: u32,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- CumulativeSum ----------
    /// [cumulativeSum()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-cumulativesum)
    CumulativeSum {
        input: OperandIndex,
        axis: u32,
        options: Option<MLCumulativeSumOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Expand ----------
    /// [expand()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-expand)
    Expand {
        input: OperandIndex,
        new_shape: Vec<MLDimension>,
        options: Option<MLOperatorOptions>,
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
        batch_dimensions: Option<u32>,
        options: Option<MLGatherOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [gatherElements()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-gatherelements)
    GatherElements {
        input: OperandIndex,
        indices: OperandIndex,
        batch_dimensions: Option<u32>,
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
        steps: u32,
        hidden_size: u32,
        options: Option<MLGruOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [gruCell()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-grucell)
    GruCell {
        input: OperandIndex,
        weight: OperandIndex,
        recurrence: OperandIndex,
        hidden_state: OperandIndex,
        hidden_size: u32,
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
        options: Option<MLOperatorOptions>,
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
        hidden_state: OperandIndex,
        cell_state: OperandIndex,
        options: Option<MLLstmCellOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Pad ----------
    /// [pad()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-pad)
    Pad {
        input: OperandIndex,
        beginning_padding: Vec<u32>,
        ending_padding: Vec<u32>,
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
        new_shape: Vec<MLDimension>,
        options: Option<MLOperatorOptions>,
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
        axis: u32,
        options: Option<MLOperatorOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Slice ----------
    /// [slice()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-slice)
    Slice {
        input: OperandIndex,
        starts: Vec<u32>,
        sizes: Vec<MLDimension>,
        options: Option<MLSliceOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Split ----------
    /// [split()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-split)
    Split {
        input: OperandIndex,
        splits: Vec<u32>,
        split_equal_parts: Option<u32>,
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
    /// [squeeze()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-squeeze) (§ 11 Operation Emulation)
    Squeeze {
        input: OperandIndex,
        options: Option<MLSqueezeOptions>,
        outputs: Vec<OperandIndex>,
    },
    /// [unsqueeze()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-unsqueeze) (§ 11 Operation Emulation)
    Unsqueeze {
        input: OperandIndex,
        options: Option<MLUnsqueezeOptions>,
        outputs: Vec<OperandIndex>,
    },

    // ---------- Tile ----------
    /// [tile()](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-tile)
    Tile {
        input: OperandIndex,
        repetitions: Vec<u32>,
        options: Option<MLOperatorOptions>,
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

/// Legacy graph JSON used a top-level `"label"` on each operation; WebNN places `label` on options.
/// When deserializing, merge top-level label into `attributes` if `attributes.label` is absent or empty.
fn merge_top_level_label_into_attributes(
    mut attributes: serde_json::Value,
    top_level_label: Option<String>,
) -> serde_json::Value {
    let Some(s) = top_level_label.filter(|x| !x.is_empty()) else {
        return attributes;
    };
    if attributes.is_null() {
        attributes = serde_json::json!({});
    }
    if let Some(obj) = attributes.as_object_mut() {
        let has_nonempty = obj
            .get("label")
            .and_then(|v| v.as_str())
            .map(|t| !t.is_empty())
            .unwrap_or(false);
        if !has_nonempty {
            obj.insert("label".to_string(), serde_json::Value::String(s));
        }
    }
    attributes
}

impl Serialize for Operation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let (op_type, input_operands, _) = self.to_legacy();
        let attributes = self.attributes_json_value();
        let outs = self.outputs();
        let output_operands: Vec<u32> = outs.to_vec();
        let output_operand = outs.first().copied();
        let mut st = serializer.serialize_struct("Operation", 5)?;
        st.serialize_field("type", &op_type)?;
        st.serialize_field("input_operands", &input_operands)?;
        st.serialize_field("attributes", &attributes)?;
        st.serialize_field("output_operand", &output_operand)?;
        st.serialize_field("output_operands", &output_operands)?;
        st.end()
    }
}

impl<'de> Deserialize<'de> for Operation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct OperationHelper {
            #[serde(rename = "type")]
            op_type: String,
            #[serde(default)]
            input_operands: Vec<u32>,
            #[serde(default)]
            output_operand: Option<u32>,
            #[serde(default)]
            output_operands: Vec<u32>,
            #[serde(default)]
            attributes: serde_json::Value,
            #[serde(default)]
            label: Option<String>,
        }
        let h = OperationHelper::deserialize(deserializer)?;
        let attributes_value = merge_top_level_label_into_attributes(h.attributes, h.label);
        let output_ids: Vec<u32> = if !h.output_operands.is_empty() {
            h.output_operands.clone()
        } else if let Some(o) = h.output_operand {
            vec![o]
        } else {
            Vec::new()
        };
        Operation::from_json_attributes(
            &h.op_type,
            &h.input_operands,
            &output_ids,
            &attributes_value,
        )
        .ok_or_else(|| {
            serde::de::Error::custom(format!("unknown or invalid op_type: {}", h.op_type))
        })
    }
}

// ---------------------------------------------------------------------------
// Legacy conversion: Operation <-> (op_type, input_operands, attributes)
// ---------------------------------------------------------------------------

impl Operation {
    /// Canonical WebNN operation type string for JSON interchange (e.g. `add`, `batchNormalization`).
    /// Single source of truth shared with [`Operation::to_legacy`] and [`Self::op_type`].
    pub fn op_type(&self) -> &'static str {
        match self {
            Operation::Add { .. } => "add",
            Operation::Sub { .. } => "sub",
            Operation::Mul { .. } => "mul",
            Operation::Div { .. } => "div",
            Operation::Pow { .. } => "pow",
            Operation::Max { .. } => "max",
            Operation::Min { .. } => "min",
            Operation::Matmul { .. } => "matmul",
            Operation::Equal { .. } => "equal",
            Operation::NotEqual { .. } => "notEqual",
            Operation::Greater { .. } => "greater",
            Operation::GreaterOrEqual { .. } => "greaterOrEqual",
            Operation::Lesser { .. } => "lesser",
            Operation::LesserOrEqual { .. } => "lesserOrEqual",
            Operation::Abs { .. } => "abs",
            Operation::Ceil { .. } => "ceil",
            Operation::Cos { .. } => "cos",
            Operation::Exp { .. } => "exp",
            Operation::Floor { .. } => "floor",
            Operation::Log { .. } => "log",
            Operation::Neg { .. } => "neg",
            Operation::Sin { .. } => "sin",
            Operation::Tan { .. } => "tan",
            Operation::Erf { .. } => "erf",
            Operation::Identity { .. } => "identity",
            Operation::Reciprocal { .. } => "reciprocal",
            Operation::Sign { .. } => "sign",
            Operation::Sqrt { .. } => "sqrt",
            Operation::Tanh { .. } => "tanh",
            Operation::Relu { .. } => "relu",
            Operation::Sigmoid { .. } => "sigmoid",
            Operation::LogicalAnd { .. } => "logicalAnd",
            Operation::LogicalOr { .. } => "logicalOr",
            Operation::LogicalNot { .. } => "logicalNot",
            Operation::LogicalXor { .. } => "logicalXor",
            Operation::Where { .. } => "where",
            Operation::ArgMax { .. } => "argMax",
            Operation::ArgMin { .. } => "argMin",
            Operation::BatchNormalization { .. } => "batchNormalization",
            Operation::Cast { .. } => "cast",
            Operation::Clamp { .. } => "clamp",
            Operation::Constant { .. } => "constant",
            Operation::Conv2d { .. } => "conv2d",
            Operation::ConvTranspose2d { .. } => "convTranspose2d",
            Operation::Concat { .. } => "concat",
            Operation::CumulativeSum { .. } => "cumulativeSum",
            Operation::Expand { .. } => "expand",
            Operation::Elu { .. } => "elu",
            Operation::Gather { .. } => "gather",
            Operation::GatherElements { .. } => "gatherElements",
            Operation::Gemm { .. } => "gemm",
            Operation::Gru { .. } => "gru",
            Operation::GruCell { .. } => "gruCell",
            Operation::HardSigmoid { .. } => "hardSigmoid",
            Operation::HardSwish { .. } => "hardSwish",
            Operation::InstanceNormalization { .. } => "instanceNormalization",
            Operation::LayerNormalization { .. } => "layerNormalization",
            Operation::LeakyRelu { .. } => "leakyRelu",
            Operation::Linear { .. } => "linear",
            Operation::Lstm { .. } => "lstm",
            Operation::LstmCell { .. } => "lstmCell",
            Operation::Pad { .. } => "pad",
            Operation::AveragePool2d { .. } => "averagePool2d",
            Operation::MaxPool2d { .. } => "maxPool2d",
            Operation::L2Pool2d { .. } => "l2Pool2d",
            Operation::GlobalAveragePool { .. } => "globalAveragePool",
            Operation::GlobalMaxPool { .. } => "globalMaxPool",
            Operation::ReduceSum { .. } => "reduceSum",
            Operation::ReduceMean { .. } => "reduceMean",
            Operation::ReduceMax { .. } => "reduceMax",
            Operation::ReduceMin { .. } => "reduceMin",
            Operation::ReduceProduct { .. } => "reduceProduct",
            Operation::ReduceL1 { .. } => "reduceL1",
            Operation::ReduceL2 { .. } => "reduceL2",
            Operation::ReduceLogSum { .. } => "reduceLogSum",
            Operation::ReduceLogSumExp { .. } => "reduceLogSumExp",
            Operation::ReduceSumSquare { .. } => "reduceSumSquare",
            Operation::Reshape { .. } => "reshape",
            Operation::Resample2d { .. } => "resample2d",
            Operation::Reverse { .. } => "reverse",
            Operation::ScatterElements { .. } => "scatterElements",
            Operation::Softmax { .. } => "softmax",
            Operation::Slice { .. } => "slice",
            Operation::Split { .. } => "split",
            Operation::Transpose { .. } => "transpose",
            Operation::Squeeze { .. } => "squeeze",
            Operation::Unsqueeze { .. } => "unsqueeze",
            Operation::Tile { .. } => "tile",
            Operation::Triangular { .. } => "triangular",
            Operation::Prelu { .. } => "prelu",
            Operation::QuantizeLinear { .. } => "quantizeLinear",
            Operation::DequantizeLinear { .. } => "dequantizeLinear",
            Operation::Softplus { .. } => "softplus",
            Operation::Softsign { .. } => "softsign",
            Operation::Gelu { .. } => "gelu",
            Operation::Shape { .. } => "shape",
            Operation::ScatterND { .. } => "scatterND",
            Operation::GatherND { .. } => "gatherND",
            Operation::IsNaN { .. } => "isNaN",
            Operation::IsInfinite { .. } => "isInfinite",
            Operation::RoundEven { .. } => "roundEven",
        }
    }

    /// Output operand id(s) recorded for this operation (same order as WebNN builder results).
    pub fn outputs(&self) -> &[OperandIndex] {
        match self {
            Operation::Add { outputs, .. } => outputs,
            Operation::Sub { outputs, .. } => outputs,
            Operation::Mul { outputs, .. } => outputs,
            Operation::Div { outputs, .. } => outputs,
            Operation::Pow { outputs, .. } => outputs,
            Operation::Max { outputs, .. } => outputs,
            Operation::Min { outputs, .. } => outputs,
            Operation::Matmul { outputs, .. } => outputs,
            Operation::Equal { outputs, .. } => outputs,
            Operation::Greater { outputs, .. } => outputs,
            Operation::GreaterOrEqual { outputs, .. } => outputs,
            Operation::Lesser { outputs, .. } => outputs,
            Operation::LesserOrEqual { outputs, .. } => outputs,
            Operation::NotEqual { outputs, .. } => outputs,
            Operation::Abs { outputs, .. } => outputs,
            Operation::Ceil { outputs, .. } => outputs,
            Operation::Cos { outputs, .. } => outputs,
            Operation::Exp { outputs, .. } => outputs,
            Operation::Floor { outputs, .. } => outputs,
            Operation::Log { outputs, .. } => outputs,
            Operation::Neg { outputs, .. } => outputs,
            Operation::Relu { outputs, .. } => outputs,
            Operation::Sigmoid { outputs, .. } => outputs,
            Operation::Sin { outputs, .. } => outputs,
            Operation::Sqrt { outputs, .. } => outputs,
            Operation::Tan { outputs, .. } => outputs,
            Operation::Tanh { outputs, .. } => outputs,
            Operation::Erf { outputs, .. } => outputs,
            Operation::Reciprocal { outputs, .. } => outputs,
            Operation::Sign { outputs, .. } => outputs,
            Operation::LogicalAnd { outputs, .. } => outputs,
            Operation::LogicalOr { outputs, .. } => outputs,
            Operation::LogicalNot { outputs, .. } => outputs,
            Operation::LogicalXor { outputs, .. } => outputs,
            Operation::Where { outputs, .. } => outputs,
            Operation::Identity { outputs, .. } => outputs,
            Operation::ArgMin { outputs, .. } => outputs,
            Operation::ArgMax { outputs, .. } => outputs,
            Operation::BatchNormalization { outputs, .. } => outputs,
            Operation::Cast { outputs, .. } => outputs,
            Operation::Clamp { outputs, .. } => outputs,
            Operation::Constant { outputs, .. } => outputs,
            Operation::Conv2d { outputs, .. } => outputs,
            Operation::ConvTranspose2d { outputs, .. } => outputs,
            Operation::Concat { outputs, .. } => outputs,
            Operation::CumulativeSum { outputs, .. } => outputs,
            Operation::Expand { outputs, .. } => outputs,
            Operation::Elu { outputs, .. } => outputs,
            Operation::Gather { outputs, .. } => outputs,
            Operation::GatherElements { outputs, .. } => outputs,
            Operation::Gemm { outputs, .. } => outputs,
            Operation::Gru { outputs, .. } => outputs,
            Operation::GruCell { outputs, .. } => outputs,
            Operation::HardSigmoid { outputs, .. } => outputs,
            Operation::HardSwish { outputs, .. } => outputs,
            Operation::InstanceNormalization { outputs, .. } => outputs,
            Operation::LayerNormalization { outputs, .. } => outputs,
            Operation::LeakyRelu { outputs, .. } => outputs,
            Operation::Linear { outputs, .. } => outputs,
            Operation::Lstm { outputs, .. } => outputs,
            Operation::LstmCell { outputs, .. } => outputs,
            Operation::Pad { outputs, .. } => outputs,
            Operation::AveragePool2d { outputs, .. } => outputs,
            Operation::MaxPool2d { outputs, .. } => outputs,
            Operation::L2Pool2d { outputs, .. } => outputs,
            Operation::GlobalAveragePool { outputs, .. } => outputs,
            Operation::GlobalMaxPool { outputs, .. } => outputs,
            Operation::ReduceSum { outputs, .. } => outputs,
            Operation::ReduceMean { outputs, .. } => outputs,
            Operation::ReduceMax { outputs, .. } => outputs,
            Operation::ReduceMin { outputs, .. } => outputs,
            Operation::ReduceProduct { outputs, .. } => outputs,
            Operation::ReduceL1 { outputs, .. } => outputs,
            Operation::ReduceL2 { outputs, .. } => outputs,
            Operation::ReduceLogSum { outputs, .. } => outputs,
            Operation::ReduceLogSumExp { outputs, .. } => outputs,
            Operation::ReduceSumSquare { outputs, .. } => outputs,
            Operation::Reshape { outputs, .. } => outputs,
            Operation::Resample2d { outputs, .. } => outputs,
            Operation::Reverse { outputs, .. } => outputs,
            Operation::ScatterElements { outputs, .. } => outputs,
            Operation::Softmax { outputs, .. } => outputs,
            Operation::Slice { outputs, .. } => outputs,
            Operation::Split { outputs, .. } => outputs,
            Operation::Transpose { outputs, .. } => outputs,
            Operation::Squeeze { outputs, .. } => outputs,
            Operation::Unsqueeze { outputs, .. } => outputs,
            Operation::Tile { outputs, .. } => outputs,
            Operation::Triangular { outputs, .. } => outputs,
            Operation::Prelu { outputs, .. } => outputs,
            Operation::QuantizeLinear { outputs, .. } => outputs,
            Operation::DequantizeLinear { outputs, .. } => outputs,
            Operation::Softplus { outputs, .. } => outputs,
            Operation::Softsign { outputs, .. } => outputs,
            Operation::Gelu { outputs, .. } => outputs,
            Operation::Shape { outputs, .. } => outputs,
            Operation::ScatterND { outputs, .. } => outputs,
            Operation::GatherND { outputs, .. } => outputs,
            Operation::IsNaN { outputs, .. } => outputs,
            Operation::IsInfinite { outputs, .. } => outputs,
            Operation::RoundEven { outputs, .. } => outputs,
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
            Operation::Add { options, .. }
            | Operation::Sub { options, .. }
            | Operation::Mul { options, .. }
            | Operation::Div { options, .. }
            | Operation::Pow { options, .. }
            | Operation::Max { options, .. }
            | Operation::Min { options, .. }
            | Operation::Matmul { options, .. }
            | Operation::Equal { options, .. }
            | Operation::NotEqual { options, .. }
            | Operation::Greater { options, .. }
            | Operation::GreaterOrEqual { options, .. }
            | Operation::Lesser { options, .. }
            | Operation::LesserOrEqual { options, .. }
            | Operation::Abs { options, .. }
            | Operation::Ceil { options, .. }
            | Operation::Cos { options, .. }
            | Operation::Exp { options, .. }
            | Operation::Floor { options, .. }
            | Operation::Log { options, .. }
            | Operation::Neg { options, .. }
            | Operation::Relu { options, .. }
            | Operation::Sigmoid { options, .. }
            | Operation::Sin { options, .. }
            | Operation::Sqrt { options, .. }
            | Operation::Tan { options, .. }
            | Operation::Tanh { options, .. }
            | Operation::Erf { options, .. }
            | Operation::Reciprocal { options, .. }
            | Operation::Sign { options, .. }
            | Operation::LogicalAnd { options, .. }
            | Operation::LogicalOr { options, .. }
            | Operation::LogicalNot { options, .. }
            | Operation::LogicalXor { options, .. }
            | Operation::Where { options, .. }
            | Operation::Identity { options, .. }
            | Operation::Prelu { options, .. }
            | Operation::QuantizeLinear { options, .. }
            | Operation::DequantizeLinear { options, .. }
            | Operation::Softplus { options, .. }
            | Operation::Softsign { options, .. }
            | Operation::Gelu { options, .. }
            | Operation::Shape { options, .. }
            | Operation::ScatterND { options, .. }
            | Operation::GatherND { options, .. }
            | Operation::IsNaN { options, .. }
            | Operation::IsInfinite { options, .. }
            | Operation::RoundEven { options, .. } => opt_label!(options),

            Operation::ArgMin { options, .. } | Operation::ArgMax { options, .. } => {
                opt_label!(options)
            }

            Operation::BatchNormalization { options, .. } => opt_label!(options),
            Operation::Cast { options, .. } => opt_label!(options),
            Operation::Clamp { options, .. } => opt_label!(options),
            Operation::Constant { options, .. } => opt_label!(options),
            Operation::Conv2d { options, .. } => opt_label!(options),
            Operation::ConvTranspose2d { options, .. } => opt_label!(options),
            Operation::Concat { options, .. } => opt_label!(options),
            Operation::CumulativeSum { options, .. } => opt_label!(options),
            Operation::Expand { options, .. } => opt_label!(options),
            Operation::Elu { options, .. } => opt_label!(options),
            Operation::Gather { options, .. } | Operation::GatherElements { options, .. } => {
                opt_label!(options)
            }
            Operation::Gemm { options, .. } => opt_label!(options),
            Operation::Gru { options, .. } => opt_label!(options),
            Operation::GruCell { options, .. } => opt_label!(options),
            Operation::HardSigmoid { options, .. } => opt_label!(options),
            Operation::HardSwish { options, .. } => opt_label!(options),
            Operation::InstanceNormalization { options, .. } => opt_label!(options),
            Operation::LayerNormalization { options, .. } => opt_label!(options),
            Operation::LeakyRelu { options, .. } => opt_label!(options),
            Operation::Linear { options, .. } => opt_label!(options),
            Operation::Lstm { options, .. } => opt_label!(options),
            Operation::LstmCell { options, .. } => opt_label!(options),
            Operation::Pad { options, .. } => opt_label!(options),
            Operation::AveragePool2d { options, .. }
            | Operation::MaxPool2d { options, .. }
            | Operation::L2Pool2d { options, .. }
            | Operation::GlobalAveragePool { options, .. }
            | Operation::GlobalMaxPool { options, .. } => opt_label!(options),
            Operation::ReduceSum { options, .. }
            | Operation::ReduceMean { options, .. }
            | Operation::ReduceMax { options, .. }
            | Operation::ReduceMin { options, .. }
            | Operation::ReduceProduct { options, .. }
            | Operation::ReduceL1 { options, .. }
            | Operation::ReduceL2 { options, .. }
            | Operation::ReduceLogSum { options, .. }
            | Operation::ReduceLogSumExp { options, .. }
            | Operation::ReduceSumSquare { options, .. } => opt_label!(options),
            Operation::Reshape { options, .. } => opt_label!(options),
            Operation::Resample2d { options, .. } => opt_label!(options),
            Operation::Reverse { options, .. } => opt_label!(options),
            Operation::ScatterElements { options, .. } => opt_label!(options),
            Operation::Softmax { options, .. } => opt_label!(options),
            Operation::Slice { options, .. } => opt_label!(options),
            Operation::Split { options, .. } => opt_label!(options),
            Operation::Transpose { options, .. } => opt_label!(options),
            Operation::Squeeze { options, .. } => opt_label!(options),
            Operation::Unsqueeze { options, .. } => opt_label!(options),
            Operation::Tile { options, .. } => opt_label!(options),
            Operation::Triangular { options, .. } => opt_label!(options),
        }
    }

    /// Legacy input operand indices. Derived from this operation.
    pub fn input_operands(&self) -> Vec<u32> {
        self.to_legacy().1
    }

    /// Legacy attributes. Derived from this operation.
    pub fn attributes(&self) -> OperatorOptions {
        self.to_legacy().2
    }

    /// First output operand id, if any (legacy JSON single-output field).
    pub fn output_operand(&self) -> Option<u32> {
        self.outputs().first().copied()
    }

    /// All output operand ids for this operation (legacy JSON multi-output field).
    pub fn output_operands(&self) -> &[u32] {
        self.outputs()
    }

    /// Borrow all output operand IDs (handles single- and multi-output operations)
    pub fn output_operands_slice(&self) -> &[u32] {
        self.outputs()
    }

    /// Get all output operand IDs (handles both single and multi-output operations)
    pub fn get_output_operands(&self) -> Vec<u32> {
        self.outputs().to_vec()
    }

    /// Attributes as a JSON value for code that expects `serde_json::Value` (e.g. parse_json_ints).
    /// Returns `Value::Null` when there are no attributes.
    pub fn attributes_value(&self) -> serde_json::Value {
        self.attributes().to_value()
    }

    /// Serialized `attributes` for JSON interchange: options dictionary plus operation-level parameters
    /// (axis, cast target type, padding lengths, etc.).
    pub fn attributes_json_value(&self) -> serde_json::Value {
        let mut v = self.to_legacy().2.to_value();
        let Some(obj) = v.as_object_mut() else {
            return v;
        };
        match self {
            Operation::ArgMin { axis, .. } | Operation::ArgMax { axis, .. } => {
                obj.insert("axis".to_string(), serde_json::json!(axis));
            }
            Operation::Cast { data_type: to, .. } => {
                if let Ok(v) = serde_json::to_value(to) {
                    obj.insert("to".to_string(), v);
                }
            }
            Operation::CumulativeSum { axis, .. } => {
                obj.insert("axis".to_string(), serde_json::json!(axis));
            }
            Operation::Concat { axis, .. } => {
                obj.insert("axis".to_string(), serde_json::json!(axis));
            }
            Operation::Expand { new_shape, .. } => {
                if !new_shape.is_empty()
                    && let Ok(v) = serde_json::to_value(new_shape)
                {
                    obj.insert("newShape".to_string(), v);
                }
            }
            Operation::Gather {
                batch_dimensions, ..
            }
            | Operation::GatherElements {
                batch_dimensions, ..
            } => {
                if let Some(bd) = batch_dimensions {
                    obj.insert("batchDimensions".to_string(), serde_json::json!(bd));
                }
            }
            Operation::Gru {
                steps, hidden_size, ..
            } => {
                obj.insert("steps".to_string(), serde_json::json!(steps));
                obj.insert("hiddenSize".to_string(), serde_json::json!(hidden_size));
            }
            Operation::GruCell { hidden_size, .. } => {
                obj.insert("hiddenSize".to_string(), serde_json::json!(hidden_size));
            }
            Operation::Pad {
                beginning_padding,
                ending_padding,
                ..
            } => {
                obj.insert(
                    "beginningPadding".to_string(),
                    serde_json::json!(beginning_padding),
                );
                obj.insert(
                    "endingPadding".to_string(),
                    serde_json::json!(ending_padding),
                );
            }
            Operation::Softmax { axis, .. } => {
                obj.insert("axis".to_string(), serde_json::json!(axis));
            }
            Operation::Slice { starts, sizes, .. } => {
                obj.insert("starts".to_string(), serde_json::json!(starts));
                obj.insert("sizes".to_string(), serde_json::json!(sizes));
            }
            Operation::Split {
                splits,
                split_equal_parts,
                ..
            } => {
                if let Some(n) = split_equal_parts {
                    obj.insert("splits".to_string(), serde_json::json!(n));
                } else if !splits.is_empty() {
                    obj.insert("splits".to_string(), serde_json::json!(splits));
                }
            }
            Operation::Tile { repetitions, .. } if !repetitions.is_empty() => {
                obj.insert("repetitions".to_string(), serde_json::json!(repetitions));
            }
            Operation::Reshape { new_shape, .. } => {
                if !new_shape.is_empty()
                    && let Ok(val) = serde_json::to_value(new_shape)
                {
                    obj.insert("newShape".to_string(), val);
                }
            }
            _ => {}
        }
        v
    }

    /// Get a single attribute by key as JSON value. Use for code that still expects key-based lookup.
    pub fn get_attr(&self, key: &str) -> Option<serde_json::Value> {
        self.attributes().get(key)
    }

    pub fn display_name(&self) -> String {
        let l = self.label();
        if !l.is_empty() {
            l.to_string()
        } else {
            self.op_type().to_string()
        }
    }

    /// Converts this operator to the legacy triple used by JSON and existing consumers.
    /// Returns `(op_type, input_operands, attributes)`.
    pub fn to_legacy(&self) -> (String, Vec<u32>, OperatorOptions) {
        let tag = self.op_type().to_string();
        use OperatorOptions as OO;
        match self {
            Operation::Add { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Sub { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Mul { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Div { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Pow { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Max { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Min { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Matmul { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Equal { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::NotEqual { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Greater { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::GreaterOrEqual { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Lesser { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::LesserOrEqual { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Abs { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Ceil { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Cos { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Exp { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Floor { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Log { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Neg { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Sin { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Tan { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Erf { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Identity { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Reciprocal { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Sign { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Sqrt { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Tanh { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Relu { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Sigmoid { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::LogicalAnd { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::LogicalOr { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::LogicalNot { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::LogicalXor { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Where {
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
            Operation::ArgMax { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::ArgMinMax(options.clone().unwrap_or_default()),
            ),
            Operation::ArgMin { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::ArgMinMax(options.clone().unwrap_or_default()),
            ),
            Operation::BatchNormalization {
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
            Operation::Cast { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Clamp { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Clamp(options.clone().unwrap_or_default()),
            ),
            Operation::Constant { options, .. } => (
                tag.clone(),
                vec![],
                OO::Constant(options.clone().unwrap_or_default()),
            ),
            Operation::Conv2d {
                input,
                filter,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *filter],
                OO::Conv2d(options.clone().unwrap_or_default()),
            ),
            Operation::ConvTranspose2d {
                input,
                filter,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *filter],
                OO::ConvTranspose2d(options.clone().unwrap_or_default()),
            ),
            Operation::Concat {
                inputs, options, ..
            } => (
                tag.clone(),
                inputs.clone(),
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::CumulativeSum { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::CumulativeSum(options.clone().unwrap_or_default()),
            ),
            Operation::Expand { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Elu { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Elu(options.clone().unwrap_or_default()),
            ),
            Operation::Gather {
                input,
                indices,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *indices],
                OO::Gather(options.clone().unwrap_or_default()),
            ),
            Operation::GatherElements {
                input,
                indices,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *indices],
                OO::Gather(options.clone().unwrap_or_default()),
            ),
            Operation::Gemm { a, b, options, .. } => (
                tag.clone(),
                vec![*a, *b],
                OO::Gemm(options.clone().unwrap_or_default()),
            ),
            Operation::Gru {
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
            Operation::GruCell {
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
            Operation::HardSigmoid { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::HardSigmoid(options.clone().unwrap_or_default()),
            ),
            Operation::HardSwish { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::InstanceNormalization { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::InstanceNormalization(options.clone().unwrap_or_default()),
            ),
            Operation::LayerNormalization { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::LayerNormalization(options.clone().unwrap_or_default()),
            ),
            Operation::LeakyRelu { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::LeakyRelu(options.clone().unwrap_or_default()),
            ),
            Operation::Linear { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Linear(options.clone().unwrap_or_default()),
            ),
            Operation::Lstm {
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
            Operation::LstmCell {
                input,
                weight,
                recurrence,
                hidden_state,
                cell_state,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *weight, *recurrence, *hidden_state, *cell_state],
                OO::LstmCell(options.clone().unwrap_or_default()),
            ),
            Operation::Pad { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Pad(options.clone().unwrap_or_default()),
            ),
            Operation::AveragePool2d { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Pool2d(options.clone().unwrap_or_default()),
            ),
            Operation::MaxPool2d { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Pool2d(options.clone().unwrap_or_default()),
            ),
            Operation::L2Pool2d { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Pool2d(options.clone().unwrap_or_default()),
            ),
            Operation::GlobalAveragePool { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Pool2d(options.clone().unwrap_or_default()),
            ),
            Operation::GlobalMaxPool { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Pool2d(options.clone().unwrap_or_default()),
            ),
            Operation::ReduceSum { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operation::ReduceMean { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operation::ReduceMax { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operation::ReduceMin { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operation::ReduceProduct { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operation::ReduceL1 { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operation::ReduceL2 { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operation::ReduceLogSum { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operation::ReduceLogSumExp { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operation::ReduceSumSquare { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reduce(options.clone().unwrap_or_default()),
            ),
            Operation::Reshape { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Resample2d { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Resample2d(options.clone().unwrap_or_default()),
            ),
            Operation::Reverse { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Reverse(options.clone().unwrap_or_default()),
            ),
            Operation::ScatterElements {
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
            Operation::Softmax { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Slice { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Slice(options.clone().unwrap_or_default()),
            ),
            Operation::Split { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Split(options.clone().unwrap_or_default()),
            ),
            Operation::Transpose { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Transpose(options.clone().unwrap_or_default()),
            ),
            Operation::Squeeze { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Squeeze(options.clone().unwrap_or_default()),
            ),
            Operation::Unsqueeze { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Unsqueeze(options.clone().unwrap_or_default()),
            ),
            Operation::Tile { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Triangular { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Triangular(options.clone().unwrap_or_default()),
            ),
            Operation::Prelu {
                input,
                slope,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *slope],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::QuantizeLinear {
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
            Operation::DequantizeLinear {
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
            Operation::Softplus { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Softsign { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Gelu { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::Shape { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::ScatterND {
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
            Operation::GatherND {
                input,
                indices,
                options,
                ..
            } => (
                tag.clone(),
                vec![*input, *indices],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::IsNaN { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::IsInfinite { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
            Operation::RoundEven { input, options, .. } => (
                tag.clone(),
                vec![*input],
                OO::Operator(options.clone().unwrap_or_default()),
            ),
        }
    }

    /// Parse WebNN-style attributes JSON (method parameters and options keys in one object),
    /// wire operand indices, and return a fully built [`Operation`].
    ///
    /// This is the preferred entry point for graph interchange: callers do not handle
    /// [`OperatorOptions`] or [`OperationExtras`] separately. Null or empty object attributes
    /// deserialize as default options with no stripped extras.
    ///
    /// Returns `None` if `op_type` is unknown or operand lengths do not match.
    pub fn from_json_attributes(
        op_type: &str,
        input_operands: &[u32],
        outputs: &[OperandIndex],
        attributes: &serde_json::Value,
    ) -> Option<Self> {
        // Null must not map to OperatorOptions::default() (generic MLOperatorOptions): ops like
        // averagePool2d need Pool2d(MLPool2dOptions::default()) so ONNX can infer kernel_shape.
        let empty_obj = serde_json::Value::Object(Default::default());
        let (opts, extras) = if attributes.is_null() {
            OperatorOptions::from_json_with_op_type_and_extras(op_type, &empty_obj)
        } else {
            OperatorOptions::from_json_with_op_type_and_extras(op_type, attributes)
        };
        Self::from_operator_options(op_type, input_operands, &opts, outputs, extras)
    }

    /// Parses WebNN interchange: builder/JSON `op_type` string (e.g. camelCase `batchNormalization`
    /// or lowercase `add`), operand indices in spec order, and typed [`OperatorOptions`].
    ///
    /// For JSON that mixes method-level keys with options, use [`Self::from_json_attributes`]
    /// instead; this function is for callers that already have [`OperatorOptions`] and
    /// [`OperationExtras`] (e.g. custom tooling).
    ///
    /// Returns `None` if `op_type` is unknown or operand lengths do not match.
    pub fn from_operator_options(
        op_type: &str,
        input_operands: &[u32],
        attributes: &OperatorOptions,
        outputs: &[OperandIndex],
        extras: OperationExtras,
    ) -> Option<Self> {
        fn at(inputs: &[u32], i: usize) -> Option<u32> {
            inputs.get(i).copied()
        }
        let n = op_type.trim();
        match n {
            "add" if input_operands.len() >= 2 => Some(Operation::Add {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "sub" if input_operands.len() >= 2 => Some(Operation::Sub {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "mul" if input_operands.len() >= 2 => Some(Operation::Mul {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "div" if input_operands.len() >= 2 => Some(Operation::Div {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "pow" if input_operands.len() >= 2 => Some(Operation::Pow {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "max" if input_operands.len() >= 2 => Some(Operation::Max {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "min" if input_operands.len() >= 2 => Some(Operation::Min {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "matmul" if input_operands.len() >= 2 => Some(Operation::Matmul {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "equal" if input_operands.len() >= 2 => Some(Operation::Equal {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "notEqual" if input_operands.len() >= 2 => Some(Operation::NotEqual {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "greater" if input_operands.len() >= 2 => Some(Operation::Greater {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "greaterOrEqual" if input_operands.len() >= 2 => Some(Operation::GreaterOrEqual {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "lesser" if input_operands.len() >= 2 => Some(Operation::Lesser {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "lesserOrEqual" if input_operands.len() >= 2 => Some(Operation::LesserOrEqual {
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
                    "abs" => Operation::Abs {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "ceil" => Operation::Ceil {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "cos" => Operation::Cos {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "exp" => Operation::Exp {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "floor" => Operation::Floor {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "log" => Operation::Log {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "neg" => Operation::Neg {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "sin" => Operation::Sin {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "tan" => Operation::Tan {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "erf" => Operation::Erf {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "identity" => Operation::Identity {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reciprocal" => Operation::Reciprocal {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "sign" => Operation::Sign {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "sqrt" => Operation::Sqrt {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "tanh" => Operation::Tanh {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "relu" => Operation::Relu {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "sigmoid" => Operation::Sigmoid {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "logicalNot" => Operation::LogicalNot {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    _ => return None,
                })
            }
            "logicalAnd" if input_operands.len() >= 2 => Some(Operation::LogicalAnd {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "logicalOr" if input_operands.len() >= 2 => Some(Operation::LogicalOr {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "logicalXor" if input_operands.len() >= 2 => Some(Operation::LogicalXor {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "where" if input_operands.len() >= 3 => Some(Operation::Where {
                condition: at(input_operands, 0)?,
                true_value: at(input_operands, 1)?,
                false_value: at(input_operands, 2)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "argMax" if !input_operands.is_empty() => Some(Operation::ArgMax {
                input: at(input_operands, 0)?,
                axis: extras.axis.unwrap_or(0),
                options: attributes.as_arg_min_max().cloned(),
                outputs: outputs.to_vec(),
            }),
            "argMin" if !input_operands.is_empty() => Some(Operation::ArgMin {
                input: at(input_operands, 0)?,
                axis: extras.axis.unwrap_or(0),
                options: attributes.as_arg_min_max().cloned(),
                outputs: outputs.to_vec(),
            }),
            "batchNormalization" if input_operands.len() >= 3 => {
                Some(Operation::BatchNormalization {
                    input: at(input_operands, 0)?,
                    mean: at(input_operands, 1)?,
                    variance: at(input_operands, 2)?,
                    options: attributes.as_batch_normalization().cloned(),
                    outputs: outputs.to_vec(),
                })
            }
            "cast" if !input_operands.is_empty() => Some(Operation::Cast {
                input: at(input_operands, 0)?,
                data_type: extras.to_data_type.unwrap_or_default(),
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "clamp" if !input_operands.is_empty() => Some(Operation::Clamp {
                input: at(input_operands, 0)?,
                options: attributes.as_clamp().cloned(),
                outputs: outputs.to_vec(),
            }),
            "constant" => Some(Operation::Constant {
                options: attributes.as_constant().cloned(),
                outputs: outputs.to_vec(),
            }),
            "conv2d" if input_operands.len() >= 2 => {
                let parsed = attributes.as_conv2d().cloned();
                let mut opts = parsed.clone().unwrap_or_default();
                if opts.bias.is_none()
                    && let Some(b) = at(input_operands, 2)
                {
                    opts.bias = Some(b);
                }
                let options = match parsed {
                    None if opts == MLConv2dOptions::default() => None,
                    _ => Some(opts),
                };
                Some(Operation::Conv2d {
                    input: at(input_operands, 0)?,
                    filter: at(input_operands, 1)?,
                    options,
                    outputs: outputs.to_vec(),
                })
            }
            "convTranspose2d" if input_operands.len() >= 2 => {
                let parsed = attributes.as_conv_transpose2d().cloned();
                let mut opts = parsed.clone().unwrap_or_default();
                if opts.bias.is_none()
                    && let Some(b) = at(input_operands, 2)
                {
                    opts.bias = Some(b);
                }
                let options = match parsed {
                    None if opts == MLConvTranspose2dOptions::default() => None,
                    _ => Some(opts),
                };
                Some(Operation::ConvTranspose2d {
                    input: at(input_operands, 0)?,
                    filter: at(input_operands, 1)?,
                    options,
                    outputs: outputs.to_vec(),
                })
            }
            "concat" => Some(Operation::Concat {
                inputs: input_operands.to_vec(),
                axis: extras.axis.unwrap_or(0),
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "cumulativeSum" if !input_operands.is_empty() => Some(Operation::CumulativeSum {
                input: at(input_operands, 0)?,
                axis: extras.axis.unwrap_or(0),
                options: attributes.as_cumulative_sum().cloned(),
                outputs: outputs.to_vec(),
            }),
            "expand" if !input_operands.is_empty() => Some(Operation::Expand {
                input: at(input_operands, 0)?,
                new_shape: extras.expand_new_shape,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "elu" if !input_operands.is_empty() => Some(Operation::Elu {
                input: at(input_operands, 0)?,
                options: attributes.as_elu().cloned(),
                outputs: outputs.to_vec(),
            }),
            "gather" if input_operands.len() >= 2 => Some(Operation::Gather {
                input: at(input_operands, 0)?,
                indices: at(input_operands, 1)?,
                batch_dimensions: extras.batch_dimensions,
                options: attributes.as_gather().cloned(),
                outputs: outputs.to_vec(),
            }),
            "gatherElements" if input_operands.len() >= 2 => Some(Operation::GatherElements {
                input: at(input_operands, 0)?,
                indices: at(input_operands, 1)?,
                batch_dimensions: extras.batch_dimensions,
                options: attributes.as_gather().cloned(),
                outputs: outputs.to_vec(),
            }),
            "gemm" if input_operands.len() >= 2 => Some(Operation::Gemm {
                a: at(input_operands, 0)?,
                b: at(input_operands, 1)?,
                options: attributes.as_gemm().cloned(),
                outputs: outputs.to_vec(),
            }),
            "gru" if input_operands.len() >= 3 => Some(Operation::Gru {
                input: at(input_operands, 0)?,
                weight: at(input_operands, 1)?,
                recurrence: at(input_operands, 2)?,
                steps: extras.steps.unwrap_or(0),
                hidden_size: extras.hidden_size.unwrap_or(0),
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
                Some(Operation::GruCell {
                    input: at(input_operands, 0)?,
                    weight: at(input_operands, 1)?,
                    recurrence: at(input_operands, 2)?,
                    hidden_state: at(input_operands, 3)?,
                    hidden_size: extras.hidden_size.unwrap_or(0),
                    options,
                    outputs: outputs.to_vec(),
                })
            }
            "hardSigmoid" if !input_operands.is_empty() => Some(Operation::HardSigmoid {
                input: at(input_operands, 0)?,
                options: attributes.as_hard_sigmoid().cloned(),
                outputs: outputs.to_vec(),
            }),
            "hardSwish" if !input_operands.is_empty() => Some(Operation::HardSwish {
                input: at(input_operands, 0)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "instanceNormalization" if !input_operands.is_empty() => {
                Some(Operation::InstanceNormalization {
                    input: at(input_operands, 0)?,
                    options: attributes.as_instance_normalization().cloned(),
                    outputs: outputs.to_vec(),
                })
            }
            "layerNormalization" if !input_operands.is_empty() => {
                Some(Operation::LayerNormalization {
                    input: at(input_operands, 0)?,
                    options: attributes.as_layer_normalization().cloned(),
                    outputs: outputs.to_vec(),
                })
            }
            "leakyRelu" if !input_operands.is_empty() => Some(Operation::LeakyRelu {
                input: at(input_operands, 0)?,
                options: attributes.as_leaky_relu().cloned(),
                outputs: outputs.to_vec(),
            }),
            "linear" if !input_operands.is_empty() => Some(Operation::Linear {
                input: at(input_operands, 0)?,
                options: attributes.as_linear().cloned(),
                outputs: outputs.to_vec(),
            }),
            "lstm" if input_operands.len() >= 3 => Some(Operation::Lstm {
                input: at(input_operands, 0)?,
                weight: at(input_operands, 1)?,
                recurrence: at(input_operands, 2)?,
                options: attributes.as_lstm().cloned(),
                outputs: outputs.to_vec(),
            }),
            "lstmCell" if input_operands.len() >= 5 => Some(Operation::LstmCell {
                input: at(input_operands, 0)?,
                weight: at(input_operands, 1)?,
                recurrence: at(input_operands, 2)?,
                hidden_state: at(input_operands, 3)?,
                cell_state: at(input_operands, 4)?,
                options: attributes.as_lstm_cell().cloned(),
                outputs: outputs.to_vec(),
            }),
            "pad" if !input_operands.is_empty() => Some(Operation::Pad {
                input: at(input_operands, 0)?,
                beginning_padding: extras.beginning_padding,
                ending_padding: extras.ending_padding,
                options: attributes.as_pad().cloned(),
                outputs: outputs.to_vec(),
            }),
            "averagePool2d" if !input_operands.is_empty() => Some(Operation::AveragePool2d {
                input: at(input_operands, 0)?,
                options: attributes.as_pool2d().cloned(),
                outputs: outputs.to_vec(),
            }),
            "maxPool2d" if !input_operands.is_empty() => Some(Operation::MaxPool2d {
                input: at(input_operands, 0)?,
                options: attributes.as_pool2d().cloned(),
                outputs: outputs.to_vec(),
            }),
            "l2Pool2d" if !input_operands.is_empty() => Some(Operation::L2Pool2d {
                input: at(input_operands, 0)?,
                options: attributes.as_pool2d().cloned(),
                outputs: outputs.to_vec(),
            }),
            "globalAveragePool" if !input_operands.is_empty() => {
                Some(Operation::GlobalAveragePool {
                    input: at(input_operands, 0)?,
                    options: attributes.as_pool2d().cloned(),
                    outputs: outputs.to_vec(),
                })
            }
            "globalMaxPool" if !input_operands.is_empty() => Some(Operation::GlobalMaxPool {
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
                    "reduceSum" => Operation::ReduceSum {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceMean" => Operation::ReduceMean {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceMax" => Operation::ReduceMax {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceMin" => Operation::ReduceMin {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceProduct" => Operation::ReduceProduct {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceL1" => Operation::ReduceL1 {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceL2" => Operation::ReduceL2 {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceLogSum" => Operation::ReduceLogSum {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceLogSumExp" => Operation::ReduceLogSumExp {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "reduceSumSquare" => Operation::ReduceSumSquare {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    _ => return None,
                })
            }
            "reshape" if !input_operands.is_empty() => Some(Operation::Reshape {
                input: at(input_operands, 0)?,
                new_shape: extras.reshape_new_shape.clone(),
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "resample2d" if !input_operands.is_empty() => Some(Operation::Resample2d {
                input: at(input_operands, 0)?,
                options: attributes.as_resample2d().cloned(),
                outputs: outputs.to_vec(),
            }),
            "reverse" if !input_operands.is_empty() => Some(Operation::Reverse {
                input: at(input_operands, 0)?,
                options: attributes.as_reverse().cloned(),
                outputs: outputs.to_vec(),
            }),
            "scatterElements" if input_operands.len() >= 3 => Some(Operation::ScatterElements {
                input: at(input_operands, 0)?,
                indices: at(input_operands, 1)?,
                updates: at(input_operands, 2)?,
                options: attributes.as_scatter_elements().cloned(),
                outputs: outputs.to_vec(),
            }),
            "softmax" if !input_operands.is_empty() => Some(Operation::Softmax {
                input: at(input_operands, 0)?,
                axis: extras.axis.unwrap_or(0),
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "slice" if !input_operands.is_empty() => Some(Operation::Slice {
                input: at(input_operands, 0)?,
                starts: extras.starts,
                sizes: extras.sizes,
                options: attributes.as_slice().cloned(),
                outputs: outputs.to_vec(),
            }),
            "split" if !input_operands.is_empty() => Some(Operation::Split {
                input: at(input_operands, 0)?,
                splits: extras.splits,
                split_equal_parts: extras.split_equal_parts,
                options: attributes.as_split().cloned(),
                outputs: outputs.to_vec(),
            }),
            "transpose" if !input_operands.is_empty() => Some(Operation::Transpose {
                input: at(input_operands, 0)?,
                options: attributes.as_transpose().cloned(),
                outputs: outputs.to_vec(),
            }),
            "squeeze" if !input_operands.is_empty() => Some(Operation::Squeeze {
                input: at(input_operands, 0)?,
                options: attributes.as_squeeze().cloned(),
                outputs: outputs.to_vec(),
            }),
            "unsqueeze" if !input_operands.is_empty() => Some(Operation::Unsqueeze {
                input: at(input_operands, 0)?,
                options: attributes.as_unsqueeze().cloned(),
                outputs: outputs.to_vec(),
            }),
            "tile" if !input_operands.is_empty() => Some(Operation::Tile {
                input: at(input_operands, 0)?,
                repetitions: extras.repetitions.clone(),
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "triangular" if !input_operands.is_empty() => Some(Operation::Triangular {
                input: at(input_operands, 0)?,
                options: attributes.as_triangular().cloned(),
                outputs: outputs.to_vec(),
            }),
            "prelu" if input_operands.len() >= 2 => Some(Operation::Prelu {
                input: at(input_operands, 0)?,
                slope: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "quantizeLinear" if input_operands.len() >= 2 => {
                let zero_point = input_operands.get(2).copied();
                Some(Operation::QuantizeLinear {
                    input: at(input_operands, 0)?,
                    scale: at(input_operands, 1)?,
                    zero_point,
                    options: attributes.as_operator().cloned(),
                    outputs: outputs.to_vec(),
                })
            }
            "dequantizeLinear" if input_operands.len() >= 2 => {
                let zero_point = input_operands.get(2).copied();
                Some(Operation::DequantizeLinear {
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
                    "softplus" => Operation::Softplus {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "softsign" => Operation::Softsign {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "gelu" => Operation::Gelu {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "shape" => Operation::Shape {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "isNaN" => Operation::IsNaN {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "isInfinite" => Operation::IsInfinite {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    "roundEven" | "round" => Operation::RoundEven {
                        input,
                        options: opts,
                        outputs: outputs.to_vec(),
                    },
                    _ => return None,
                })
            }
            "scatterND" if input_operands.len() >= 3 => Some(Operation::ScatterND {
                input: at(input_operands, 0)?,
                indices: at(input_operands, 1)?,
                updates: at(input_operands, 2)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            "gatherND" if input_operands.len() >= 2 => Some(Operation::GatherND {
                input: at(input_operands, 0)?,
                indices: at(input_operands, 1)?,
                options: attributes.as_operator().cloned(),
                outputs: outputs.to_vec(),
            }),
            _ => None,
        }
    }

    /// Deprecated: construct [`Operation`] variants directly, or use [`Self::from_operator_options`]
    /// when parsing WebNN JSON (`type` + `input_operands` + tagged `attributes`).
    #[deprecated(
        since = "0.6.0",
        note = "construct Operation variants directly, or call Operation::from_operator_options for JSON interchange"
    )]
    #[inline]
    pub fn from_legacy(
        op_type: &str,
        input_operands: &[u32],
        attributes: &OperatorOptions,
    ) -> Option<Self> {
        Self::from_operator_options(
            op_type,
            input_operands,
            attributes,
            &[],
            OperationExtras::default(),
        )
    }
}

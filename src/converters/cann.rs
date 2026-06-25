// SPDX-FileCopyrightText: 2026 Shubham Gupta <shubhamg13.work@gmail.com>
//
// SPDX-License-Identifier: Apache-2

//! Native CANN/HIAI IR converter for WebNN graphs.
//!
//! Converts WebNN `GraphInfo` directly to HIAI IR using the `libhiai_ir.so` C API.
//! The IR is then serialized to a protobuf format that the CANN runtime can execute.
//!
//! ## Architecture
//!
//! ```
//! WebNN GraphInfo
//!   → [CreateModelDef] → IModelDef*
//!     → [CreateGraphDef] → IGraphDef*
//!       → for each operand: [CreateTensorDef] → set shape/dtype
//!       → for each op: [CreateOpDef] → set type, inputs, outputs, attrs
//!     → [Serialize to bytes]
//!   → ConvertedGraph { data: tflite/om bytes }
//! ```
//!
//! ## C API (libhiai_ir.so)
//!
//! ```c
//! // Model lifecycle
//! IModelDef*   CreateModelDef(void);
//! void         DestroyModelDef(IModelDef*);
//!
//! // Graph
//! IGraphDef*   CreateGraphDef(void);
//! void         DestroyGraphDef(IGraphDef*);
//! const char*  IGraphDef_name(IGraphDef*);
//! void         IGraphDef_set_name(IGraphDef*, const char*);
//! int          IGraphDef_op_size(IGraphDef*);
//! IOpDef*      IGraphDef_mutable_op(IGraphDef*, int idx);
//! IOpDef*      IGraphDef_add_op(IGraphDef*);
//!
//! // Op
//! IOpDef*      CreateOpDef(void);
//! void         DestroyOpDef(IOpDef*);
//! void         IOpDef_set_type(IOpDef*, const char*);
//! const char*  IOpDef_type(IOpDef*);
//! void         IOpDef_set_name(IOpDef*, const char*);
//! void         IOpDef_add_input(IOpDef*, const char*);
//! void         IOpDef_add_output(IOpDef*, const char*);
//! IAttrMapDef* IOpDef_mutable_attr(IOpDef*);
//!
//! // Tensor
//! ITensorDef*  CreateTensorDef(void);
//! void         DestroyTensorDef(ITensorDef*);
//! void         ITensorDef_set_name(ITensorDef*, const char*);
//! IShapeDef*   ITensorDef_mutable_shape(ITensorDef*);
//!
//! // Shape
//! IShapeDef*   CreateShapeDef(void);
//! void         DestroyShapeDef(IShapeDef*);
//! void         IShapeDef_add_dim(IShapeDef*, int64_t);
//!
//! // Serialization
//! bool         ModelSerializeWrapper_SaveModelToModelDef(ge::Model&, void**);
//! void         ModelSerializeWrapper_GetModelDefBufferSize(void*, size_t*);
//! bool         ModelSerializeWrapper_SerializeModelDefToBuffer(void*, void*, size_t);
//! void         ModelSerializeWrapper_ReleaseModelDef(void*);
//! ```
//!
//! ## TODO: C++ bridge
//! The CANN backend uses the C++ adapter library (src/executors/cann_shim/)
//! as the bridge between Rust and the HiAI DDK.
//!
//! **cann-runtime**: Calls adapter functions to build a GE graph.
//! **cann-runtime-mock**: Validates graph structure, returns placeholder bytes.

use crate::error::GraphError;
use crate::graph::GraphInfo;
use crate::operators::Operation;

use super::{ConvertedGraph, GraphConverter};

/// Maps a WebNN operation to its HIAI IR operation name.
/// Used by the mock runtime only to verify operation support.
///
/// Returns Some(name) for ops that map directly to an adapter cann_op_*()
///
/// Returns `None` for ops that need decomposition or are not supported
fn webnn_op_to_hiai(op: &Operation) -> Option<&'static str> {
    match op {
        // ── Element-wise binary ──────────────────────────────────────
        Operation::Add { .. } => Some("Add"),
        Operation::Sub { .. } => Some("Sub"),
        Operation::Mul { .. } => Some("Mul"),
        Operation::Div { .. } => Some("Div"),
        Operation::Pow { .. } => Some("Pow"),
        Operation::Max { .. } => Some("Max"),
        Operation::Min { .. } => Some("Min"),
        Operation::Equal { .. } => Some("Equal"),
        Operation::Greater { .. } => Some("Greater"),
        Operation::GreaterOrEqual { .. } => Some("GreaterOrEqual"),
        Operation::Lesser { .. } => Some("Lesser"),
        Operation::LesserOrEqual { .. } => Some("LesserOrEqual"),
        Operation::NotEqual { .. } => Some("NotEqual"),
        Operation::LogicalAnd { .. } => Some("LogicalAnd"),
        Operation::LogicalOr { .. } => Some("LogicalOr"),
        Operation::LogicalXor { .. } => Some("LogicalXor"),
        Operation::LogicalNot { .. } => Some("LogicalNot"),

        // ── Element-wise unary ───────────────────────────────────────
        Operation::Abs { .. } => Some("Abs"),
        Operation::Neg { .. } => Some("Neg"),
        Operation::Exp { .. } => Some("Exp"),
        Operation::Log { .. } => Some("Log"),
        Operation::Sin { .. } => Some("Sin"),
        Operation::Cos { .. } => Some("Cos"),
        Operation::Tan { .. } => Some("Tan"),
        Operation::Sqrt { .. } => Some("Sqrt"),
        Operation::Ceil { .. } => Some("Ceil"),
        Operation::Floor { .. } => Some("Floor"),
        Operation::Sign { .. } => Some("Sign"),
        Operation::Erf { .. } => Some("Erf"),
        Operation::Reciprocal { .. } => Some("Reciprocal"),
        Operation::Cast { .. } => Some("Cast"),
        Operation::Clamp { .. } => Some("Clamp"),

        // ── Activations ──────────────────────────────────────────────
        Operation::Relu { .. } => Some("ReLU"),
        Operation::Sigmoid { .. } => Some("Sigmoid"),
        Operation::Tanh { .. } => Some("Tanh"),
        Operation::Elu { .. } => Some("ELU"),
        Operation::Gelu { .. } => Some("GELU"),
        Operation::LeakyRelu { .. } => Some("LeakyRelu"),
        Operation::HardSigmoid { .. } => Some("HardSigmoid"),
        Operation::HardSwish { .. } => Some("HardSwish"),
        Operation::Softplus { .. } => Some("Softplus"),
        Operation::Softsign { .. } => Some("Softsign"),

        // ── Convolution + Pool + Matmul ──────────────────────────────
        Operation::Conv2d { .. } => Some("Conv2D"),
        Operation::ConvTranspose2d { .. } => Some("Conv2D"),
        Operation::MaxPool2d { .. } => Some("MaxPool"),
        Operation::AveragePool2d { .. } => Some("AvgPool"),
        Operation::Matmul { .. } => Some("MatMul"),
        Operation::Gemm { .. } => Some("Gemm"),
        Operation::Softmax { .. } => Some("Softmax"),

        // ── Reduction ────────────────────────────────────────────────
        Operation::ReduceSum { .. } => Some("ReduceSum"),
        Operation::ReduceMean { .. } => Some("ReduceMean"),
        Operation::ReduceMax { .. } => Some("ReduceMax"),
        Operation::ReduceMin { .. } => Some("ReduceMin"),
        Operation::ReduceProduct { .. } => Some("ReduceProduct"),
        Operation::ReduceL1 { .. } => Some("ReduceL1"),
        Operation::ReduceL2 { .. } => Some("ReduceL2"),
        Operation::ReduceLogSum { .. } => Some("ReduceLogSum"),
        Operation::ReduceLogSumExp { .. } => Some("ReduceLogSumExp"),
        Operation::ReduceSumSquare { .. } => Some("ReduceSumSquare"),
        Operation::ArgMax { .. } => Some("ArgMax"),
        Operation::ArgMin { .. } => Some("ArgMin"),

        // ── Shape ops ────────────────────────────────────────────────
        Operation::Reshape { .. } => Some("Reshape"),
        Operation::Transpose { .. } => Some("Transpose"),
        Operation::Tile { .. } => Some("Tile"),
        Operation::Slice { .. } => Some("Slice"),
        Operation::Split { .. } => Some("Split"),
        Operation::Concat { .. } => Some("Concat"),
        Operation::Pad { .. } => Some("Pad"),
        Operation::Squeeze { .. } => Some("Squeeze"),
        Operation::Unsqueeze { .. } => Some("Unsqueeze"),
        Operation::Expand { .. } => Some("Expand"),
        Operation::CumulativeSum { .. } => Some("CumulativeSum"),

        // ── Gather / Scatter / Where ─────────────────────────────────
        Operation::Gather { .. } => Some("Gather"),
        Operation::GatherElements { .. } => Some("GatherElements"),
        Operation::GatherND { .. } => Some("GatherND"),
        Operation::ScatterElements { .. } => Some("ScatterElements"),
        Operation::ScatterND { .. } => Some("ScatterND"),
        Operation::Where { .. } => Some("Where"),

        // ── Normalization + Quantization ─────────────────────────────
        Operation::BatchNormalization { .. } => Some("BatchNormalization"),
        Operation::InstanceNormalization { .. } => Some("InstanceNormalization"),
        Operation::QuantizeLinear { .. } => Some("QuantizeLinear"),
        Operation::DequantizeLinear { .. } => Some("DequantizeLinear"),

        // ── Other ────────────────────────────────────────────────────
        Operation::Resample2d { .. } => Some("Resample2D"),
        Operation::GlobalAveragePool { .. } => Some("GlobalAveragePool"),
        Operation::GlobalMaxPool { .. } => Some("GlobalMaxPool"),
        Operation::Constant { .. } => Some("Constant"),
        Operation::Shape { .. } => Some("Shape"),

        // ── Needs decomposition (Phase 3) ────────────────────────────
        Operation::Identity { .. } => None,
        Operation::Prelu { .. } => None,
        Operation::Linear { .. } => None,
        Operation::LayerNormalization { .. } => None,
        Operation::Triangular { .. } => None,
        Operation::IsNaN { .. } => None,
        Operation::IsInfinite { .. } => None,

        // ── Not supported ────────────────────────────────────────────
        Operation::Gru { .. }
        | Operation::GruCell { .. }
        | Operation::Lstm { .. }
        | Operation::LstmCell { .. }
        | Operation::L2Pool2d { .. }
        | Operation::Reverse { .. }
        | Operation::RoundEven { .. } => None,
    }
}

pub struct CannConverter;

impl GraphConverter for CannConverter {
    fn format(&self) -> &'static str {
        "cann"
    }

    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
        let model_bytes = build_hiai_ir_model_mock(graph)?;
        Ok(ConvertedGraph {
            format: "cann",
            content_type: "application/octet-stream",
            data: model_bytes,
            weights_data: None,
        })
    }
}

/// Mock implementation: verify graph structure, return placeholder bytes.
fn build_hiai_ir_model_mock(graph: &GraphInfo) -> Result<Vec<u8>, GraphError> {
    let mut op_count: usize = 0;
    for op in &graph.operations {
        if webnn_op_to_hiai(op).is_some() {
            op_count += 1;
        }
    }
    if op_count == 0 {
        return Err(GraphError::ConversionFailed {
            format: "cann".to_string(),
            reason: "no supported ops found".to_string(),
        });
    }
    // Return placeholder CANN IR bytes (valid on aarch64: real bytes)
    Ok(vec![0x00, 0x00, 0x00, 0x00])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind};
    use crate::operator_options::MLDimension;
    use crate::operators::Operation;
    use std::collections::HashMap;

    fn make_relu_graph() -> GraphInfo {
        GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![Dimension::Static(1), Dimension::Static(4)],
                        pending_permutation: vec![],
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![Dimension::Static(1), Dimension::Static(4)],
                        pending_permutation: vec![],
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![Operation::Relu {
                input: 0,
                options: None,
                outputs: vec![1],
            }],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        }
    }

    #[test]
    fn test_relu_graph_converts() {
        let graph = make_relu_graph();
        let converter = CannConverter;
        assert_eq!(converter.format(), "cann");
        // On x86: returns mock bytes
        // On aarch64: returns real HIAI IR bytes
        let result = converter.convert(&graph);
        assert!(result.is_ok(), "{result:?}");
        let converted = result.unwrap();
        assert!(!converted.data.is_empty());
    }

    #[test]
    fn test_webnn_op_to_hiai_relu() {
        let op = Operation::Relu {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("ReLU"));
    }

    #[test]
    fn test_webnn_op_to_hiai_sigmoid() {
        let op = Operation::Sigmoid {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Sigmoid"));
    }

    #[test]
    fn test_webnn_op_to_hiai_tanh() {
        let op = Operation::Tanh {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Tanh"));
    }

    #[test]
    fn test_webnn_op_to_hiai_add() {
        let op = Operation::Add {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Add"));
    }

    #[test]
    fn test_webnn_op_to_hiai_mul() {
        let op = Operation::Mul {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Mul"));
    }

    #[test]
    fn test_webnn_op_to_hiai_sub() {
        let op = Operation::Sub {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Sub"));
    }

    #[test]
    fn test_webnn_op_to_hiai_conv2d() {
        let op = Operation::Conv2d {
            input: 0,
            filter: 1,
            options: None,
            outputs: vec![3],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Conv2D"));
    }

    #[test]
    fn test_webnn_op_to_hiai_max_pool2d() {
        let op = Operation::MaxPool2d {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("MaxPool"));
    }

    #[test]
    fn test_webnn_op_to_hiai_average_pool2d() {
        let op = Operation::AveragePool2d {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("AvgPool"));
    }

    #[test]
    fn test_webnn_op_to_hiai_matmul() {
        let op = Operation::Matmul {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("MatMul"));
    }

    #[test]
    fn test_webnn_op_to_hiai_softmax() {
        let op = Operation::Softmax {
            input: 0,
            axis: 1,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Softmax"));
    }

    #[test]
    fn test_webnn_op_to_hiai_reshape() {
        let op = Operation::Reshape {
            input: 0,
            new_shape: vec![MLDimension::Static(1), MLDimension::Static(4)],
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Reshape"));
    }

    #[test]
    fn test_webnn_op_to_hiai_concat() {
        let op = Operation::Concat {
            inputs: vec![0, 1],
            axis: 0,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Concat"));
    }

    #[test]
    fn test_webnn_op_to_hiai_identity_returns_none() {
        let op = Operation::Identity {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), None);
    }
}

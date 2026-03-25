/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
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

/// CoreML MLProgram (MIL) converter
///
/// This converter generates CoreML MLProgram models using the Model Intermediate Language (MIL).
/// MLProgram is the modern CoreML format (spec v7+, iOS 15+, macOS 12+) that supports:
/// - Flexible MIL operations
/// - Quantization operations
/// - Better optimization
///
/// This replaces the legacy NeuralNetwork format.
use crate::converters::operand_name;
use crate::error::GraphError;
use crate::graph::{DataType, Dimension as GraphDimension, GraphInfo};
use crate::operator_options::MLDimension;
use crate::operators::Operation;
use crate::protos::coreml::mil_spec::{
    Argument, Block, Dimension, Function, NamedValueType, Operation as MilOperation, Program,
    TensorType, ValueType, argument::binding::Binding, dimension,
};
use crate::protos::coreml::specification::Model;
use prost::Message;
use std::collections::HashMap;

/// MIL operation type names (matching Chromium's implementation)
mod mil_ops {
    // Binary operations
    pub const ADD: &str = "add";
    pub const SUB: &str = "sub";
    pub const MUL: &str = "mul";
    pub const DIV: &str = "real_div";
    pub const POW: &str = "pow";
    /// Element-wise maximum (WebNN max).
    pub const MAXIMUM: &str = "maximum";
    /// Element-wise minimum (WebNN min).
    pub const MINIMUM: &str = "minimum";
    pub const MATMUL: &str = "matmul";

    // Activation functions
    pub const RELU: &str = "relu";
    pub const SIGMOID: &str = "sigmoid";
    pub const TANH: &str = "tanh";
    pub const SOFTMAX: &str = "softmax";

    // Convolution and pooling
    pub const CONV: &str = "conv";
    pub const CONV_TRANSPOSE: &str = "conv_transpose";
    pub const AVG_POOL: &str = "avg_pool";
    pub const MAX_POOL: &str = "max_pool";
    pub const GLOBAL_AVG_POOL: &str = "reduce_mean"; // Global pooling via reduction
    pub const GLOBAL_MAX_POOL: &str = "reduce_max"; // Global pooling via reduction

    // Normalization
    pub const BATCH_NORM: &str = "batch_norm";
    pub const INSTANCE_NORM: &str = "instance_norm";
    pub const LAYER_NORM: &str = "layer_norm";

    // Reduction operations
    pub const REDUCE_SUM: &str = "reduce_sum";
    pub const REDUCE_MEAN: &str = "reduce_mean";
    pub const REDUCE_MAX: &str = "reduce_max";
    pub const REDUCE_MIN: &str = "reduce_min";
    pub const REDUCE_PROD: &str = "reduce_prod";
    pub const REDUCE_L1: &str = "reduce_l1_norm";
    pub const REDUCE_L2: &str = "reduce_l2_norm";
    pub const REDUCE_LOG_SUM: &str = "reduce_log_sum";
    pub const REDUCE_LOG_SUM_EXP: &str = "reduce_log_sum_exp";
    pub const REDUCE_SUM_SQUARE: &str = "reduce_sum_square";

    // Element-wise unary operations
    pub const ABS: &str = "abs";
    pub const CEIL: &str = "ceil";
    pub const FLOOR: &str = "floor";
    pub const ROUND_EVEN: &str = "round"; // WebNN roundEven: round to nearest even (MIL "round")
    pub const NEG: &str = "mul"; // Multiply by -1
    pub const IDENTITY: &str = "identity";
    pub const EXP: &str = "exp";
    pub const LOG: &str = "log";
    pub const SQRT: &str = "sqrt";
    pub const SIGN: &str = "sign";
    pub const SIN: &str = "sin";
    pub const COS: &str = "cos";
    pub const TAN: &str = "tan";
    pub const ERF: &str = "erf";
    pub const RECIPROCAL: &str = "inverse";

    // Logic operations
    pub const EQUAL: &str = "equal";
    pub const GREATER: &str = "greater";
    pub const GREATER_EQUAL: &str = "greater_equal";
    pub const LESS: &str = "less";
    pub const LESS_EQUAL: &str = "less_equal";
    pub const LOGICAL_NOT: &str = "logical_not";
    pub const LOGICAL_AND: &str = "logical_and";
    pub const LOGICAL_OR: &str = "logical_or";
    pub const LOGICAL_XOR: &str = "logical_xor";

    // Quantization
    pub const DEQUANTIZE: &str = "dequantize";
    pub const QUANTIZE: &str = "quantize";

    // Shape operations
    pub const RESHAPE: &str = "reshape";

    // Tensor manipulation operations
    pub const TRANSPOSE: &str = "transpose";
    pub const CONCAT: &str = "concat";
    pub const SLICE: &str = "slice_by_size";
    pub const EXPAND: &str = "tile";
    pub const GATHER: &str = "gather";
    pub const GATHER_ALONG_AXIS: &str = "gather_along_axis";
    pub const SPLIT: &str = "split";
    pub const WHERE: &str = "select";
    pub const PAD: &str = "pad";

    // Advanced activation operations
    pub const GELU: &str = "gelu";

    // Specialized activation operations
    pub const PRELU: &str = "prelu";
    pub const ELU: &str = "elu";
    pub const LEAKY_RELU: &str = "leaky_relu";
    pub const SOFTPLUS: &str = "softplus";
    pub const SOFTSIGN: &str = "softsign";
    pub const HARD_SIGMOID: &str = "sigmoid_hard";
    pub const HARD_SWISH: &str = "mul"; // TODO: Implement as x * hardSigmoid(x)

    // Dimension manipulation operations
    pub const SQUEEZE: &str = "squeeze";
    pub const UNSQUEEZE: &str = "expand_dims";

    // Arg reduce operations
    pub const ARG_MAX: &str = "reduce_argmax";
    pub const ARG_MIN: &str = "reduce_argmin";

    // Type conversion operations
    pub const CAST: &str = "cast";

    // Scatter operations
    pub const SCATTER_ELEMENTS: &str = "scatter";
    pub const SCATTER_ND: &str = "scatter_nd";

    // Tile operation
    pub const TILE: &str = "tile";
    pub const REVERSE: &str = "reverse";
    pub const CUM_SUM: &str = "cumsum";

    // Triangular operation
    pub const TRIANGULAR: &str = "band_part";

    // Clamp operation
    pub const CLIP: &str = "clip";
}

// Default epsilon value used by several CoreML operations for numerical stability.
const DEFAULT_EPSILON: f32 = 1e-45;

#[derive(Default)]
pub struct CoremlMlProgramConverter;

impl CoremlMlProgramConverter {
    /// Parse MLNumber values represented as JSON numbers or strings.
    /// Supports non-finite strings used by WPT/interchange JSON.
    fn parse_mlnumber_f64(value: Option<&serde_json::Value>) -> Option<f64> {
        let v = value?;
        if let Some(n) = v.as_f64() {
            return Some(n);
        }
        let s = v.as_str()?.trim().to_ascii_lowercase();
        match s.as_str() {
            "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
            "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
            "nan" => Some(f64::NAN),
            _ => s.parse::<f64>().ok(),
        }
    }

    /// Parse clamp bounds from MLNumber. NaN is treated as "missing bound".
    fn parse_clamp_bound(value: Option<&serde_json::Value>, default: f64) -> f64 {
        Self::parse_mlnumber_f64(value)
            .filter(|v| !v.is_nan())
            .unwrap_or(default)
    }

    fn mil_dimension_from_graph_dim(dim: &GraphDimension) -> Dimension {
        match dim {
            GraphDimension::Static(v) => Dimension {
                dimension: Some(dimension::Dimension::Constant(
                    dimension::ConstantDimension { size: *v as u64 },
                )),
            },
            GraphDimension::Dynamic(_) => Dimension {
                dimension: Some(dimension::Dimension::Unknown(dimension::UnknownDimension {
                    variadic: false,
                })),
            },
        }
    }

    fn mil_dimensions_from_graph_shape(
        shape: &[GraphDimension],
        scalar_as_one_dim: bool,
    ) -> Vec<Dimension> {
        if shape.is_empty() && scalar_as_one_dim {
            return vec![Dimension {
                dimension: Some(dimension::Dimension::Constant(
                    dimension::ConstantDimension { size: 1 },
                )),
            }];
        }
        shape
            .iter()
            .map(Self::mil_dimension_from_graph_dim)
            .collect()
    }

    fn permute_graph_shape(shape: &[GraphDimension], perm: &[u32]) -> Vec<GraphDimension> {
        perm.iter().map(|&i| shape[i as usize].clone()).collect()
    }

    /// Create a MIL Value for a tensor operand
    fn create_value(
        graph: &GraphInfo,
        operand_id: u32,
    ) -> Result<(String, NamedValueType), GraphError> {
        let operand = graph
            .operand(operand_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!("Operand {} not found", operand_id),
            })?;

        let name = operand_name(graph, operand_id);

        let dtype = Self::mil_data_type(&operand.descriptor.data_type)?;
        let value_type =
            Self::create_named_value_type(name.clone(), dtype, &operand.descriptor.shape, true);

        Ok((name, value_type))
    }

    fn create_named_value_type(
        name: String,
        data_type: i32,
        shape: &[GraphDimension],
        scalar_as_one_dim: bool,
    ) -> NamedValueType {
        let dimensions = Self::mil_dimensions_from_graph_shape(shape, scalar_as_one_dim);

        let value_type = ValueType {
            r#type: Some(
                crate::protos::coreml::mil_spec::value_type::Type::TensorType(TensorType {
                    rank: dimensions.len() as i64,
                    data_type,
                    dimensions,
                    attributes: HashMap::new(),
                }),
            ),
        };

        NamedValueType {
            name,
            r#type: Some(value_type),
        }
    }

    fn create_value_with_mil_type(
        graph: &GraphInfo,
        operand_id: u32,
        name: String,
        data_type: i32,
    ) -> Result<NamedValueType, GraphError> {
        let operand = graph
            .operand(operand_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!("Operand {} not found", operand_id),
            })?;

        Ok(Self::create_named_value_type(
            name,
            data_type,
            &operand.descriptor.shape,
            true,
        ))
    }

    fn output_name_for_operand(
        graph: &GraphInfo,
        operand_id: u32,
        operand_name_overrides: &HashMap<u32, String>,
    ) -> String {
        operand_name_overrides
            .get(&operand_id)
            .cloned()
            .unwrap_or_else(|| operand_name(graph, operand_id))
    }

    fn create_output_value(
        graph: &GraphInfo,
        operand_id: u32,
        operand_name_overrides: &HashMap<u32, String>,
    ) -> Result<(String, NamedValueType), GraphError> {
        let name = Self::output_name_for_operand(graph, operand_id, operand_name_overrides);
        let value_type = Self::create_value_with_mil_type(
            graph,
            operand_id,
            name.clone(),
            Self::mil_data_type(
                &graph
                    .operand(operand_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Operand {} not found", operand_id),
                    })?
                    .descriptor
                    .data_type,
            )?,
        )?;
        Ok((name, value_type))
    }

    fn interface_mil_data_type(data_type: &DataType) -> i32 {
        use crate::protos::coreml::mil_spec::DataType as MilDataType;

        match data_type {
            DataType::Float32 => MilDataType::Float32 as i32,
            DataType::Float16 => MilDataType::Float16 as i32,
            DataType::Int32 => MilDataType::Int32 as i32,
            DataType::Int4
            | DataType::Uint4
            | DataType::Int8
            | DataType::Uint8
            | DataType::Uint32
            | DataType::Int64
            | DataType::Uint64 => MilDataType::Float32 as i32,
        }
    }

    fn cast_dtype_string_for_mil_type(data_type: i32) -> Result<&'static str, GraphError> {
        use crate::protos::coreml::mil_spec::DataType as MilDataType;

        match data_type {
            value if value == MilDataType::Float32 as i32 => Ok("fp32"),
            value if value == MilDataType::Float16 as i32 => Ok("fp16"),
            value if value == MilDataType::Int32 as i32 => Ok("int32"),
            value if value == MilDataType::Bool as i32 => Ok("bool"),
            _ => Err(GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!("Unsupported MIL cast dtype {}", data_type),
            }),
        }
    }

    fn cast_dtype_string_for_graph_type(data_type: &DataType) -> Result<&'static str, GraphError> {
        match data_type {
            DataType::Float32 => Ok("fp32"),
            DataType::Float16 => Ok("fp16"),
            DataType::Int32 => Ok("int32"),
            DataType::Uint32 => Ok("uint32"),
            DataType::Int8 => Ok("int8"),
            DataType::Uint8 => Ok("uint8"),
            DataType::Int64 => Ok("int64"),
            DataType::Int4 | DataType::Uint4 | DataType::Uint64 => {
                Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: format!("Unsupported graph cast dtype {:?}", data_type),
                })
            }
        }
    }

    /// Convert WebNN DataType to MIL DataType
    fn mil_data_type(data_type: &DataType) -> Result<i32, GraphError> {
        use crate::protos::coreml::mil_spec::DataType as MilDataType;

        Ok(match data_type {
            DataType::Int4 | DataType::Uint4 => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml".to_string(),
                    reason: "int4/uint4 tensors are not supported in CoreML conversion yet"
                        .to_string(),
                });
            }
            DataType::Float32 => MilDataType::Float32 as i32,
            DataType::Float16 => MilDataType::Float16 as i32,
            DataType::Int32 => MilDataType::Int32 as i32,
            DataType::Int8 => MilDataType::Int8 as i32,
            DataType::Uint32 => MilDataType::Uint32 as i32,
            DataType::Uint8 => MilDataType::Uint8 as i32,
            DataType::Int64 => MilDataType::Int64 as i32,
            DataType::Uint64 => MilDataType::Uint64 as i32,
        })
    }

    /// Create a const operation for a constant operand
    fn create_const_operation(
        graph: &GraphInfo,
        operand_id: u32,
        operand: &crate::graph::Operand,
        constant_data: &crate::graph::ConstantData,
        weight_builder: &mut super::WeightFileBuilder,
    ) -> Result<MilOperation, GraphError> {
        use crate::protos::coreml::mil_spec::{TensorValue, Value, tensor_value, value};

        let (_name, output_type) = Self::create_value(graph, operand_id)?;

        // Create tensor value from constant data
        let tensor_value = match operand.descriptor.data_type {
            crate::graph::DataType::Float32 => {
                // Convert raw bytes to f32 values
                let float_count = constant_data.data.len() / 4;
                let mut floats = Vec::with_capacity(float_count);
                for i in 0..float_count {
                    let bytes = [
                        constant_data.data[i * 4],
                        constant_data.data[i * 4 + 1],
                        constant_data.data[i * 4 + 2],
                        constant_data.data[i * 4 + 3],
                    ];
                    floats.push(f32::from_le_bytes(bytes));
                }
                TensorValue {
                    value: Some(tensor_value::Value::Floats(tensor_value::RepeatedFloats {
                        values: floats,
                    })),
                }
            }
            crate::graph::DataType::Int32 => {
                // Convert raw bytes to i32 values
                let int_count = constant_data.data.len() / 4;
                let mut ints = Vec::with_capacity(int_count);
                for i in 0..int_count {
                    let bytes = [
                        constant_data.data[i * 4],
                        constant_data.data[i * 4 + 1],
                        constant_data.data[i * 4 + 2],
                        constant_data.data[i * 4 + 3],
                    ];
                    ints.push(i32::from_le_bytes(bytes));
                }
                TensorValue {
                    value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                        values: ints,
                    })),
                }
            }
            crate::graph::DataType::Float16 => {
                // CoreML MLProgram (MIL) requires non-scalar Float16 constants to be stored
                // in a separate weight file with BlobFileValue references, not as immediate values.
                // Only scalar (0D) Float16 can be stored as immediate bytes.
                //
                // Chromium's implementation uses WeightsFileHandle::Write() which:
                // - For scalars (empty shape): stores as immediate value
                // - For non-scalars: writes to weights.bin with 64-byte alignment
                //
                // Reference: chromium/src/services/webnn/coreml/graph_builder_coreml.cc

                let is_scalar = operand.descriptor.shape.is_empty();

                if !is_scalar {
                    // Non-scalar Float16: add to weight file and return BlobFileValue
                    let element_count = constant_data.data.len() / 2; // 2 bytes per f16
                    let offset = weight_builder.add_weight(
                        operand_id,
                        element_count,
                        &constant_data.data,
                    )?;

                    // Create BlobFileValue reference
                    let blob_file_value = Value {
                        doc_string: String::new(),
                        r#type: output_type.r#type.clone(),
                        value: Some(value::Value::BlobFileValue(value::BlobFileValue {
                            file_name: "@model_path/weights/weights.bin".to_string(),
                            offset,
                        })),
                    };

                    // Create const operation with BlobFileValue
                    let mut attributes = HashMap::new();
                    attributes.insert("val".to_string(), blob_file_value);

                    return Ok(MilOperation {
                        r#type: "const".to_string(),
                        inputs: HashMap::new(),
                        outputs: vec![output_type],
                        attributes,
                        ..Default::default()
                    });
                }

                // Scalar Float16: store as immediate bytes
                TensorValue {
                    value: Some(tensor_value::Value::Bytes(tensor_value::RepeatedBytes {
                        values: constant_data.data.clone().into(),
                    })),
                }
            }
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: format!(
                        "Unsupported constant data type: {:?}",
                        operand.descriptor.data_type
                    ),
                });
            }
        };

        // Create immediate value
        let immediate_value = Value {
            doc_string: String::new(),
            r#type: output_type.r#type.clone(),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tensor_value)),
            })),
        };

        // Create const operation
        // Note: const operations in CoreML MIL use attributes, not inputs, for the value
        let mut attributes = HashMap::new();
        attributes.insert("val".to_string(), immediate_value);

        Ok(MilOperation {
            r#type: "const".to_string(),
            inputs: HashMap::new(),
            outputs: vec![output_type],
            attributes,
            ..Default::default()
        })
    }

    /// Create a MIL operation
    fn create_mil_operation(
        op_type: &str,
        inputs: HashMap<String, Argument>,
        outputs: Vec<NamedValueType>,
    ) -> MilOperation {
        MilOperation {
            r#type: op_type.to_string(),
            inputs,
            outputs,
            ..Default::default()
        }
    }

    /// Create an Argument from an operand name
    fn create_argument(operand_name: &str) -> Argument {
        Argument {
            arguments: vec![crate::protos::coreml::mil_spec::argument::Binding {
                binding: Some(Binding::Name(operand_name.to_string())),
            }],
        }
    }

    /// Create an Argument from multiple operand names (tuple/list of references)
    /// Used for variadic parameters like concat's 'values'
    fn create_argument_tuple(operand_names: &[String]) -> Argument {
        Argument {
            arguments: operand_names
                .iter()
                .map(|name| crate::protos::coreml::mil_spec::argument::Binding {
                    binding: Some(Binding::Name(name.clone())),
                })
                .collect(),
        }
    }

    /// Create an Argument from an immediate integer array value (int32)
    fn create_immediate_int_array(values: &[u32]) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, dimension,
            tensor_value, value, value_type,
        };

        let int_values: Vec<i32> = values.iter().map(|&v| v as i32).collect();

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                values: int_values,
            })),
        };

        let value = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::Int32 as i32,
                    rank: 1,
                    dimensions: vec![Dimension {
                        dimension: Some(dimension::Dimension::Constant(
                            dimension::ConstantDimension {
                                size: values.len() as u64,
                            },
                        )),
                    }],
                    attributes: HashMap::new(),
                })),
            }),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tensor_value)),
            })),
        };

        Argument {
            arguments: vec![crate::protos::coreml::mil_spec::argument::Binding {
                binding: Some(Binding::Value(value)),
            }],
        }
    }

    /// Create an Argument from an immediate integer scalar value (int32)
    fn create_immediate_int(value: u32) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, tensor_value,
            value, value_type,
        };

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                values: vec![value as i32],
            })),
        };

        let val = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::Int32 as i32,
                    rank: 0, // Scalar
                    dimensions: vec![],
                    attributes: HashMap::new(),
                })),
            }),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tensor_value)),
            })),
        };

        Argument {
            arguments: vec![crate::protos::coreml::mil_spec::argument::Binding {
                binding: Some(Binding::Value(val)),
            }],
        }
    }

    /// Create an Argument from an immediate float scalar value
    fn create_immediate_float(value: f32) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, tensor_value,
            value, value_type,
        };

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::Floats(tensor_value::RepeatedFloats {
                values: vec![value],
            })),
        };

        let val = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::Float32 as i32,
                    rank: 0, // Scalar
                    dimensions: vec![],
                    attributes: HashMap::new(),
                })),
            }),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tensor_value)),
            })),
        };

        Argument {
            arguments: vec![crate::protos::coreml::mil_spec::argument::Binding {
                binding: Some(Binding::Value(val)),
            }],
        }
    }

    /// Create an Argument from an immediate float16 value (scalar)
    fn create_immediate_float16(value: f32) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, tensor_value,
            value, value_type,
        };

        // Convert f32 to f16 bytes
        let f16_bits = half::f16::from_f32(value).to_bits();
        let bytes = f16_bits.to_le_bytes().to_vec();

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::Bytes(tensor_value::RepeatedBytes {
                values: bytes.into(),
            })),
        };

        let val = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::Float16 as i32,
                    rank: 0, // Scalar
                    dimensions: vec![],
                    attributes: HashMap::new(),
                })),
            }),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tensor_value)),
            })),
        };

        Argument {
            arguments: vec![crate::protos::coreml::mil_spec::argument::Binding {
                binding: Some(Binding::Value(val)),
            }],
        }
    }

    /// Create an Argument from an immediate string value
    fn create_immediate_string(value: &str) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, tensor_value,
            value, value_type,
        };

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::Strings(
                tensor_value::RepeatedStrings {
                    values: vec![value.to_string()],
                },
            )),
        };

        let val = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::String as i32,
                    rank: 0, // Scalar string
                    dimensions: vec![],
                    attributes: HashMap::new(),
                })),
            }),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tensor_value)),
            })),
        };

        Argument {
            arguments: vec![crate::protos::coreml::mil_spec::argument::Binding {
                binding: Some(Binding::Value(val)),
            }],
        }
    }

    /// Create an immediate bool argument
    fn create_immediate_bool(value: bool) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, tensor_value,
            value, value_type,
        };

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::Bools(tensor_value::RepeatedBools {
                values: vec![value],
            })),
        };

        let val = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::Bool as i32,
                    rank: 0, // Scalar
                    dimensions: vec![],
                    attributes: HashMap::new(),
                })),
            }),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tensor_value)),
            })),
        };

        Argument {
            arguments: vec![crate::protos::coreml::mil_spec::argument::Binding {
                binding: Some(Binding::Value(val)),
            }],
        }
    }

    /// Create an argument referencing a named value
    fn create_name_argument(name: String) -> Argument {
        use crate::protos::coreml::mil_spec::argument::binding::Binding;

        Argument {
            arguments: vec![crate::protos::coreml::mil_spec::argument::Binding {
                binding: Some(Binding::Name(name)),
            }],
        }
    }

    /// Create an immediate int argument
    fn create_int_argument(value: i32) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, tensor_value,
            value, value_type,
        };

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                values: vec![value],
            })),
        };

        let val = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::Int32 as i32,
                    rank: 0, // Scalar
                    dimensions: vec![],
                    attributes: HashMap::new(),
                })),
            }),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tensor_value)),
            })),
        };

        Argument {
            arguments: vec![crate::protos::coreml::mil_spec::argument::Binding {
                binding: Some(Binding::Value(val)),
            }],
        }
    }

    /// Create an immediate int array argument
    fn create_int_array_argument(values: Vec<i32>) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, tensor_value,
            value, value_type,
        };

        let values_len = values.len();

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                values,
            })),
        };

        let val = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::Int32 as i32,
                    rank: 1, // 1D array
                    dimensions: vec![Dimension {
                        dimension: Some(dimension::Dimension::Constant(
                            dimension::ConstantDimension {
                                size: values_len as u64,
                            },
                        )),
                    }],
                    attributes: HashMap::new(),
                })),
            }),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tensor_value)),
            })),
        };

        Argument {
            arguments: vec![crate::protos::coreml::mil_spec::argument::Binding {
                binding: Some(Binding::Value(val)),
            }],
        }
    }

    /// Map WebNN operation to MIL operation (with optional operand name overrides)
    fn convert_operation_with_overrides(
        &self,
        graph: &GraphInfo,
        op: &Operation,
        operand_name_overrides: &HashMap<u32, String>,
    ) -> Result<MilOperation, GraphError> {
        // Handle multi-output operations separately
        if matches!(&op, Operation::Split { .. }) {
            return self.convert_split_operation(graph, op);
        }

        let mil_op_type = self.get_mil_op_type(op.op_type())?;

        // Get input operand names, using overrides if available
        let input_names = Self::input_names_for_operation(graph, op, operand_name_overrides);

        // Get output operand info
        // Check if this is a single-output or multi-output operation
        let output_id = if let Some(id) = op.output_operand() {
            // Single-output operation
            id
        } else if !op.output_operands().is_empty() {
            // Multi-output operation not handled yet
            return Err(GraphError::ConversionFailed {
                format: "CoreML MLProgram".to_string(),
                reason: format!(
                    "operation '{}' has multiple outputs but is not implemented as multi-output. \
                     Only 'split' is currently supported as multi-output.",
                    op.op_type()
                ),
            });
        } else {
            // No outputs at all - this shouldn't happen but handle gracefully
            return Err(GraphError::ConversionFailed {
                format: "CoreML MLProgram".to_string(),
                reason: format!("operation '{}' has no output operands", op.op_type()),
            });
        };

        let (_output_name, output_type) =
            Self::create_output_value(graph, output_id, operand_name_overrides)?;

        self.convert_operation_with_input_names_and_outputs(
            graph,
            op,
            &input_names,
            vec![output_type],
            mil_op_type,
        )
    }

    fn input_names_for_operation(
        graph: &GraphInfo,
        op: &Operation,
        operand_name_overrides: &HashMap<u32, String>,
    ) -> Vec<String> {
        op.input_operands()
            .iter()
            .map(|&id| {
                operand_name_overrides
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| operand_name(graph, id))
            })
            .collect()
    }

    fn convert_operation_with_input_names_and_outputs(
        &self,
        graph: &GraphInfo,
        op: &Operation,
        input_names: &[String],
        outputs: Vec<NamedValueType>,
        mil_op_type: &str,
    ) -> Result<MilOperation, GraphError> {
        let inputs = self.create_operation_inputs(graph, op, input_names)?;
        Ok(Self::create_mil_operation(mil_op_type, inputs, outputs))
    }

    fn create_cast_operation(
        input_name: String,
        output_type: NamedValueType,
        dtype: &str,
    ) -> MilOperation {
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), Self::create_name_argument(input_name));
        inputs.insert("dtype".to_string(), Self::create_immediate_string(dtype));
        Self::create_mil_operation(mil_ops::CAST, inputs, vec![output_type])
    }

    /// Map WebNN operation to MIL operation (convenience wrapper without overrides)
    #[allow(dead_code)]
    fn convert_operation(
        &self,
        graph: &GraphInfo,
        op: &Operation,
    ) -> Result<MilOperation, GraphError> {
        self.convert_operation_with_overrides(graph, op, &HashMap::new())
    }

    /// Convert split operation (multi-output)
    fn convert_split_operation(
        &self,
        graph: &GraphInfo,
        op: &Operation,
    ) -> Result<MilOperation, GraphError> {
        let Operation::Split {
            input,
            splits,
            options,
            ..
        } = &op
        else {
            return Err(GraphError::ConversionFailed {
                format: "CoreML MLProgram".to_string(),
                reason: "expected Split operator".to_string(),
            });
        };
        let input_id = *input;

        // Get input operand name
        let input_name = operand_name(graph, input_id);

        // Get output types
        let outputs: Vec<NamedValueType> = op
            .output_operands()
            .iter()
            .map(|&id| {
                let (_name, value_type) = Self::create_value(graph, id)?;
                Ok(value_type)
            })
            .collect::<Result<Vec<_>, GraphError>>()?;

        // Create inputs
        let mut inputs: HashMap<String, Argument> = HashMap::new();

        // Add main input (x)
        inputs.insert("x".to_string(), Self::create_name_argument(input_name));

        // num_splits or split_sizes from operation; axis from MLSplitOptions.
        let axis = options.as_ref().map(|o| o.axis).unwrap_or(0);
        if splits.is_empty() {
            inputs.insert(
                "num_splits".to_string(),
                Self::create_int_argument(op.output_operands().len() as i32),
            );
        } else {
            let split_sizes: Vec<i32> = splits.iter().map(|&u| u as i32).collect();
            inputs.insert(
                "split_sizes".to_string(),
                Self::create_int_array_argument(split_sizes),
            );
        }
        inputs.insert("axis".to_string(), Self::create_int_argument(axis as i32));

        Ok(Self::create_mil_operation("split", inputs, outputs))
    }

    /// Get MIL operation type for WebNN operation
    fn get_mil_op_type(&self, webnn_op: &str) -> Result<&'static str, GraphError> {
        let mil_type = match webnn_op.to_lowercase().as_str() {
            // Binary operations
            "add" => mil_ops::ADD,
            "sub" => mil_ops::SUB,
            "mul" => mil_ops::MUL,
            "div" => mil_ops::DIV,
            "pow" => mil_ops::POW,
            "max" => mil_ops::MAXIMUM,
            "min" => mil_ops::MINIMUM,
            "matmul" => mil_ops::MATMUL,
            "gemm" => mil_ops::MATMUL, // Gemm maps to matmul with transpose handling

            // Activation functions
            "relu" => mil_ops::RELU,
            "sigmoid" => mil_ops::SIGMOID,
            "tanh" => mil_ops::TANH,
            "softmax" => mil_ops::SOFTMAX,

            // Convolution and pooling
            "conv2d" => mil_ops::CONV,
            "convtranspose2d" => mil_ops::CONV_TRANSPOSE,
            "averagepool2d" => mil_ops::AVG_POOL,
            "maxpool2d" => mil_ops::MAX_POOL,
            "globalaveragepool" => mil_ops::GLOBAL_AVG_POOL,
            "globalmaxpool" => mil_ops::GLOBAL_MAX_POOL,

            // Normalization
            "batchnormalization" => mil_ops::BATCH_NORM,
            "instancenormalization" => mil_ops::INSTANCE_NORM,
            "layernormalization" => mil_ops::LAYER_NORM,

            // Reduction operations
            "reducesum" => mil_ops::REDUCE_SUM,
            "reducemean" => mil_ops::REDUCE_MEAN,
            "reducemax" => mil_ops::REDUCE_MAX,
            "reducemin" => mil_ops::REDUCE_MIN,
            "reduceproduct" => mil_ops::REDUCE_PROD,
            "reducel1" => mil_ops::REDUCE_L1,
            "reducel2" => mil_ops::REDUCE_L2,
            "reducelogsum" => mil_ops::REDUCE_LOG_SUM,
            "reducelogsumexp" => mil_ops::REDUCE_LOG_SUM_EXP,
            "reducesumsquare" => mil_ops::REDUCE_SUM_SQUARE,

            // Element-wise unary operations
            "abs" => mil_ops::ABS,
            "ceil" => mil_ops::CEIL,
            "floor" => mil_ops::FLOOR,
            "roundeven" => mil_ops::ROUND_EVEN,
            "neg" => mil_ops::NEG,
            "identity" => mil_ops::IDENTITY,
            "exp" => mil_ops::EXP,
            "log" => mil_ops::LOG,
            "sqrt" => mil_ops::SQRT,
            "sign" => mil_ops::SIGN,
            "sin" => mil_ops::SIN,
            "cos" => mil_ops::COS,
            "tan" => mil_ops::TAN,
            "erf" => mil_ops::ERF,
            "reciprocal" => mil_ops::RECIPROCAL,

            // Logic operations
            "equal" => mil_ops::EQUAL,
            "greater" => mil_ops::GREATER,
            "greaterorequal" => mil_ops::GREATER_EQUAL,
            "lesser" => mil_ops::LESS,
            "lesserorequal" => mil_ops::LESS_EQUAL,
            "logicalnot" => mil_ops::LOGICAL_NOT,
            "logicaland" => mil_ops::LOGICAL_AND,
            "logicalor" => mil_ops::LOGICAL_OR,
            "logicalxor" => mil_ops::LOGICAL_XOR,

            // Quantization
            "dequantizelinear" => mil_ops::DEQUANTIZE,
            "quantizelinear" => mil_ops::QUANTIZE,

            // Shape operations
            "reshape" => mil_ops::RESHAPE,

            // Tensor manipulation
            "transpose" => mil_ops::TRANSPOSE,
            "concat" => mil_ops::CONCAT,
            "slice" => mil_ops::SLICE,
            "expand" => mil_ops::EXPAND,
            "gather" => mil_ops::GATHER,
            "gatherelements" => mil_ops::GATHER_ALONG_AXIS,
            "split" => mil_ops::SPLIT,
            "where" => mil_ops::WHERE,
            "pad" => mil_ops::PAD,
            "cumulativesum" => mil_ops::CUM_SUM,
            "cumulative_sum" => mil_ops::CUM_SUM,

            // Advanced operations
            "gelu" => mil_ops::GELU,
            "squeeze" => mil_ops::SQUEEZE,
            "unsqueeze" => mil_ops::UNSQUEEZE,
            "argmax" => mil_ops::ARG_MAX,
            "argmin" => mil_ops::ARG_MIN,
            "cast" => mil_ops::CAST,

            // Specialized activation operations
            "prelu" => mil_ops::PRELU,
            "elu" => mil_ops::ELU,
            "leakyrelu" => mil_ops::LEAKY_RELU,
            "softplus" => mil_ops::SOFTPLUS,
            "softsign" => mil_ops::SOFTSIGN,
            "hardsigmoid" => mil_ops::HARD_SIGMOID,
            "hardswish" => mil_ops::HARD_SWISH,

            // Scatter operations
            "scatterelements" => mil_ops::SCATTER_ELEMENTS,
            "scatternd" => mil_ops::SCATTER_ND,

            // Tile operation
            "tile" => mil_ops::TILE,

            // Reverse operation
            "reverse" => mil_ops::REVERSE,

            // Triangular operation
            "triangular" => mil_ops::TRIANGULAR,

            // Clamp operation
            "clamp" => mil_ops::CLIP,

            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: format!("Unsupported operation: {}", webnn_op),
                });
            }
        };

        Ok(mil_type)
    }

    /// Create inputs map for MIL operation
    fn create_operation_inputs(
        &self,
        _graph: &GraphInfo,
        op: &Operation,
        input_names: &[String],
    ) -> Result<HashMap<String, Argument>, GraphError> {
        let mut inputs = HashMap::new();

        match &op {
            // Binary operations: x, y
            Operation::Add { .. }
            | Operation::Sub { .. }
            | Operation::Mul { .. }
            | Operation::Div { .. }
            | Operation::Pow { .. }
            | Operation::Max { .. }
            | Operation::Min { .. }
            | Operation::Equal { .. }
            | Operation::Greater { .. }
            | Operation::GreaterOrEqual { .. }
            | Operation::Lesser { .. }
            | Operation::LesserOrEqual { .. }
            | Operation::LogicalAnd { .. }
            | Operation::LogicalOr { .. }
            | Operation::LogicalXor { .. } => {
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("y".to_string(), Self::create_argument(&input_names[1]));
                }
            }

            // MatMul operation: x, y, transpose_x, transpose_y
            // CoreML requires transpose parameters, WebNN doesn't have them so default to false
            Operation::Matmul { .. } => {
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("y".to_string(), Self::create_argument(&input_names[1]));
                }

                // Add transpose_x parameter (required by CoreML, defaults to false)
                inputs.insert(
                    "transpose_x".to_string(),
                    Self::create_immediate_bool(false),
                );

                // Add transpose_y parameter (required by CoreML, defaults to false)
                inputs.insert(
                    "transpose_y".to_string(),
                    Self::create_immediate_bool(false),
                );
            }

            // Gemm operation: General Matrix Multiplication
            // Y = alpha * op(A) * op(B) + beta * C
            // CoreML matmul handles: Y = A * B (with transpose options)
            // For now, we support transpose options and basic matmul
            // TODO: Support alpha, beta, and bias (C) by decomposing into mul and add operations
            Operation::Gemm { options, .. } => {
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("y".to_string(), Self::create_argument(&input_names[1]));
                }

                // Add transpose parameters from operator options
                if let Some(opts) = options {
                    inputs.insert(
                        "transpose_x".to_string(),
                        Self::create_immediate_bool(opts.a_transpose),
                    );
                    inputs.insert(
                        "transpose_y".to_string(),
                        Self::create_immediate_bool(opts.b_transpose),
                    );
                }

                // Note: alpha, beta, and bias (C) are not yet supported
                // These would require decomposing gemm into multiple operations:
                // 1. matmul(op(A), op(B))
                // 2. mul by alpha if != 1.0
                // 3. add beta * C if C is provided
            }

            // Global pooling operations (reduce over spatial dimensions)
            Operation::GlobalAveragePool { .. } | Operation::GlobalMaxPool { .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // Global pooling reduces over spatial dimensions (2, 3) for NCHW format
                inputs.insert(
                    "axes".to_string(),
                    Self::create_immediate_int_array(&[2, 3]),
                );
                // Keep dimensions to maintain rank
                inputs.insert("keep_dims".to_string(), Self::create_immediate_bool(true));
            }

            // Softmax operation (axis is required by WebNN spec)
            Operation::Softmax { axis, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                inputs.insert("axis".to_string(), Self::create_immediate_int(*axis));
            }

            // Neg operation: implemented as mul by -1, requires x and y parameters
            // CoreML neg is actually a mul operation, so we need both operands
            Operation::Neg { .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // Add -1.0 as the multiplier (y parameter required by CoreML mul)
                inputs.insert("y".to_string(), Self::create_immediate_float(-1.0));
            }

            // Unary operations: x
            Operation::Relu { .. }
            | Operation::Sigmoid { .. }
            | Operation::Tanh { .. }
            | Operation::Abs { .. }
            | Operation::Ceil { .. }
            | Operation::Floor { .. }
            | Operation::RoundEven { .. }
            | Operation::Sign { .. }
            | Operation::Identity { .. }
            | Operation::Exp { .. }
            | Operation::Sqrt { .. }
            | Operation::Sin { .. }
            | Operation::Cos { .. }
            | Operation::Tan { .. }
            | Operation::Erf { .. }
            | Operation::LogicalNot { .. }
            | Operation::Softplus { .. }
            | Operation::Softsign { .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
            }

            Operation::Reciprocal { .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // Reciprocal requires epsilon parameter (default to 1e-45 for numerical stability)
                inputs.insert(
                    "epsilon".to_string(),
                    Self::create_immediate_float(DEFAULT_EPSILON),
                );
            }

            // Log operation requires epsilon parameter
            Operation::Log { .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // CoreML log requires epsilon parameter (default to 1e-45 for numerical stability)
                inputs.insert(
                    "epsilon".to_string(),
                    Self::create_immediate_float(DEFAULT_EPSILON),
                );
            }

            // Quantization operations: input, scale, zero_point
            Operation::DequantizeLinear { .. } | Operation::QuantizeLinear { .. } => {
                if input_names.len() >= 3 {
                    inputs.insert("input".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("scale".to_string(), Self::create_argument(&input_names[1]));
                    inputs.insert(
                        "zero_point".to_string(),
                        Self::create_argument(&input_names[2]),
                    );
                }
            }

            // Specialized activation: prelu - x, slope (two inputs)
            Operation::Prelu { .. } => {
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("alpha".to_string(), Self::create_argument(&input_names[1]));
                }
            }

            // Specialized activations with alpha parameter: elu, leakyRelu
            Operation::Elu { options, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                let alpha = options.as_ref().map(|o| o.alpha as f32).unwrap_or(1.0);
                inputs.insert("alpha".to_string(), Self::create_immediate_float(alpha));
            }
            Operation::LeakyRelu { options, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                let alpha = options.as_ref().map(|o| o.alpha as f32).unwrap_or(0.01);
                inputs.insert("alpha".to_string(), Self::create_immediate_float(alpha));
            }

            // HardSwish: decomposed in main loop (hardsigmoid + mul)
            // This case should never be reached due to continue in main loop
            Operation::HardSwish { .. } => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: "hardswish should be decomposed in main loop, not here".to_string(),
                });
            }

            // HardSigmoid: x, alpha, beta parameters
            Operation::HardSigmoid { options, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                if let Some(opts) = options {
                    inputs.insert(
                        "alpha".to_string(),
                        Self::create_immediate_float(opts.alpha as f32),
                    );
                    inputs.insert(
                        "beta".to_string(),
                        Self::create_immediate_float(opts.beta as f32),
                    );
                }
            }

            // Clamp operation: x, alpha (min), beta (max)
            Operation::Clamp { options, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // CoreML clip operation requires BOTH alpha and beta parameters
                // WebNN clamp defaults: minValue=-Infinity, maxValue=+Infinity
                let (min_value, max_value) = options
                    .as_ref()
                    .map(|o| {
                        let min =
                            Self::parse_clamp_bound(o.min_value.as_ref(), f64::NEG_INFINITY) as f32;
                        let max =
                            Self::parse_clamp_bound(o.max_value.as_ref(), f64::INFINITY) as f32;
                        (min, max)
                    })
                    .unwrap_or((f32::NEG_INFINITY, f32::INFINITY));

                // Alpha and beta must match input type (CoreML requirement)
                // Check first input operand type and use appropriate immediate value method
                let use_float16 = if !op.input_operands().is_empty() {
                    if let Some(input_operand) = _graph.operand(op.input_operands()[0]) {
                        input_operand.descriptor.data_type == DataType::Float16
                    } else {
                        false
                    }
                } else {
                    false
                };

                if use_float16 {
                    inputs.insert(
                        "alpha".to_string(),
                        Self::create_immediate_float16(min_value),
                    );
                    inputs.insert(
                        "beta".to_string(),
                        Self::create_immediate_float16(max_value),
                    );
                } else {
                    inputs.insert("alpha".to_string(), Self::create_immediate_float(min_value));
                    inputs.insert("beta".to_string(), Self::create_immediate_float(max_value));
                }
            }

            // Transpose operation: x, permutation
            Operation::Transpose { options, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add permutation parameter (required by CoreML)
                // If not specified in WebNN, default is to reverse all dimensions
                if let Some(opts) = options
                    && !opts.permutation.is_empty()
                {
                    inputs.insert(
                        "perm".to_string(),
                        Self::create_immediate_int_array(&opts.permutation),
                    );
                } else if !op.input_operands().is_empty()
                    && let Some(input_operand) = _graph.operand(op.input_operands()[0])
                {
                    let rank = input_operand.descriptor.shape.len();
                    let default_perm: Vec<u32> = (0..rank).rev().map(|i| i as u32).collect();
                    inputs.insert(
                        "perm".to_string(),
                        Self::create_immediate_int_array(&default_perm),
                    );
                }
            }

            // Reshape: x, shape
            Operation::Reshape { new_shape, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                if !new_shape.is_empty() {
                    let shape_values =
                        crate::operator_options::mldimensions_static_or_max(new_shape);
                    inputs.insert(
                        "shape".to_string(),
                        Self::create_immediate_int_array(&shape_values),
                    );
                }
            }

            // Convolution operations: input, filter + parameters
            Operation::Conv2d { options, .. } => {
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("weight".to_string(), Self::create_argument(&input_names[1]));
                }

                // Add optional bias if present (third input)
                if input_names.len() >= 3 {
                    inputs.insert("bias".to_string(), Self::create_argument(&input_names[2]));
                }

                // Add parameters from operator options
                if let Some(opts) = options {
                    if !opts.strides.is_empty() {
                        inputs.insert(
                            "strides".to_string(),
                            Self::create_immediate_int_array(&opts.strides),
                        );
                    }
                    if !opts.dilations.is_empty() {
                        inputs.insert(
                            "dilations".to_string(),
                            Self::create_immediate_int_array(&opts.dilations),
                        );
                    }
                    if !opts.padding.is_empty() {
                        inputs.insert(
                            "pad".to_string(),
                            Self::create_immediate_int_array(&opts.padding),
                        );
                    }
                    inputs.insert(
                        "groups".to_string(),
                        Self::create_immediate_int(opts.groups),
                    );
                }

                // Add pad_type - required parameter in CoreML
                // Use "custom" when explicit padding is provided
                inputs.insert(
                    "pad_type".to_string(),
                    Self::create_immediate_string("custom"),
                );
            }

            // Transposed convolution: input, filter + parameters
            Operation::ConvTranspose2d { options, .. } => {
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("weight".to_string(), Self::create_argument(&input_names[1]));
                }

                // Add optional bias if present (third input)
                if input_names.len() >= 3 {
                    inputs.insert("bias".to_string(), Self::create_argument(&input_names[2]));
                }

                // CoreML requires pad_type parameter (defaults to "custom" for explicit padding)
                inputs.insert(
                    "pad_type".to_string(),
                    Self::create_immediate_string("custom"),
                );

                // Add parameters from operator options
                if let Some(opts) = options {
                    if !opts.strides.is_empty() {
                        inputs.insert(
                            "strides".to_string(),
                            Self::create_immediate_int_array(&opts.strides),
                        );
                    }
                    if !opts.dilations.is_empty() {
                        inputs.insert(
                            "dilations".to_string(),
                            Self::create_immediate_int_array(&opts.dilations),
                        );
                    }
                    if !opts.padding.is_empty() {
                        inputs.insert(
                            "pad".to_string(),
                            Self::create_immediate_int_array(&opts.padding),
                        );
                    }
                    inputs.insert(
                        "groups".to_string(),
                        Self::create_immediate_int(opts.groups),
                    );
                }
                // Handle outputSizes (explicit output spatial dimensions [H, W])
                // Following Chromium: For conv_transpose, CoreML requires output_shape
                // to be the full output tensor dimensions [N, C, H, W] (from output operand),
                // NOT just the spatial dimensions from outputSizes attribute.
                // See: graph_builder_coreml.cc lines 2328-2334
                // When explicit outputSizes is provided, we need to compute full output shape.
                // For now, skip adding output_shape when using padding (custom pad_type).
                // TODO: Compute full output shape from outputSizes + input shape + channels
            }

            // Pooling operations: input + parameters
            Operation::AveragePool2d {
                options: pool_opts, ..
            }
            | Operation::MaxPool2d {
                options: pool_opts, ..
            } => {
                // CoreML MLProgram pooling path currently assumes NCHW input layout.
                // Reject NHWC explicitly to avoid invalid model/runtime crashes.
                let layout = pool_opts
                    .as_ref()
                    .map(|o| {
                        if o.layout.is_empty() {
                            "nchw"
                        } else {
                            o.layout.as_str()
                        }
                    })
                    .unwrap_or("nchw");
                if !layout.eq_ignore_ascii_case("nchw") {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!(
                            "CoreML pooling currently supports only NCHW layout; got '{}' for {}",
                            layout,
                            op.op_type(),
                        ),
                    });
                }

                // WebNN `outputSizes` for pooling is not currently lowered to CoreML
                // pooling parameters in this converter. Reject explicitly to avoid
                // output-shape mismatches that can lead to runtime crashes.
                if let Some(output_sizes) = pool_opts.as_ref().and_then(|o| o.output_sizes.as_ref())
                    && !output_sizes.is_empty()
                {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!(
                            "CoreML pooling with outputSizes is not supported yet; got {:?} for {}",
                            output_sizes,
                            op.op_type()
                        ),
                    });
                }

                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // outputShapeRounding: "floor" (default) or "ceil"
                let ceil_mode = pool_opts
                    .as_ref()
                    .map(|o| o.output_shape_rounding.eq_ignore_ascii_case("ceil"))
                    .unwrap_or(false);
                inputs.insert(
                    "ceil_mode".to_string(),
                    Self::create_immediate_bool(ceil_mode),
                );

                // Only average pooling accepts this parameter.
                if matches!(&op, Operation::AveragePool2d { .. }) {
                    inputs.insert(
                        "exclude_padding_from_average".to_string(),
                        Self::create_immediate_bool(false),
                    );
                }

                // Add parameters from operator options
                if let Some(opts) = pool_opts.as_ref() {
                    if let Some(window_dimensions) = opts.window_dimensions.as_ref()
                        && !window_dimensions.is_empty()
                    {
                        inputs.insert(
                            "kernel_sizes".to_string(),
                            Self::create_immediate_int_array(window_dimensions),
                        );
                    }
                    if !opts.strides.is_empty() {
                        inputs.insert(
                            "strides".to_string(),
                            Self::create_immediate_int_array(&opts.strides),
                        );
                    }
                    if !opts.dilations.is_empty() && opts.dilations.iter().any(|&d| d != 1) {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!(
                                "CoreML pooling does not support non-default dilations; got {:?} for {}",
                                opts.dilations,
                                op.op_type()
                            ),
                        });
                    }
                    if !opts.padding.is_empty() {
                        inputs.insert(
                            "pad".to_string(),
                            Self::create_immediate_int_array(&opts.padding),
                        );
                        inputs.insert(
                            "pad_type".to_string(),
                            Self::create_immediate_string("custom"),
                        );
                    } else {
                        inputs.insert(
                            "pad_type".to_string(),
                            Self::create_immediate_string("same"),
                        );
                    }
                } else {
                    inputs.insert(
                        "pad_type".to_string(),
                        Self::create_immediate_string("same"),
                    );
                }
            }

            // Layer normalization (different from batch/instance normalization)
            Operation::LayerNormalization { options, .. } => {
                // Check if axes is empty - CoreML doesn't support empty axes
                // Following Chromium (graph_builder_coreml.cc:4000-4019):
                // When axes is empty, mean equals input, so output = bias + (scale * 0)
                let axes_vec: Vec<i32> = options
                    .as_ref()
                    .and_then(|o| o.axes.as_ref())
                    .map(|ax| ax.iter().map(|&u| u as i32).collect())
                    .unwrap_or_default();

                if axes_vec.is_empty() {
                    // Empty axes case: use sub operation (input - input = 0)
                    // CoreML doesn't support empty axes, so we emulate it
                    // Note: This will be handled by inserting a sub operation in convert_operation
                    // For now, return error as this needs special multi-operation handling
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: "CoreML layer_norm with empty axes requires special handling (not yet implemented)".to_string(),
                    });
                }

                // Add input operand (only x)
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Scale (gamma) is optional (2nd input)
                // CoreML requires scale/bias to be constant tensors (not graph inputs)
                // Following Chromium: TODO(crbug.com/338529226) - these params must be constant
                if input_names.len() >= 2 && op.input_operands().len() >= 2 {
                    let scale_operand_id = op.input_operands()[1];
                    if let Some(scale_operand) = _graph.operand(scale_operand_id)
                        && scale_operand.kind != crate::graph::OperandKind::Constant
                    {
                        return Err(GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: "CoreML layer_norm requires scale (gamma) parameter to be a constant tensor, not a graph input".to_string(),
                            });
                    }
                    inputs.insert("gamma".to_string(), Self::create_argument(&input_names[1]));
                }

                // Bias (beta) is optional (3rd input)
                if input_names.len() >= 3 && op.input_operands().len() >= 3 {
                    let bias_operand_id = op.input_operands()[2];
                    if let Some(bias_operand) = _graph.operand(bias_operand_id)
                        && bias_operand.kind != crate::graph::OperandKind::Constant
                    {
                        return Err(GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: "CoreML layer_norm requires bias (beta) parameter to be a constant tensor, not a graph input".to_string(),
                            });
                    }
                    inputs.insert("beta".to_string(), Self::create_argument(&input_names[2]));
                }

                // Add axes parameter (REQUIRED by CoreML, must not be empty)
                inputs.insert(
                    "axes".to_string(),
                    Self::create_int_array_argument(axes_vec),
                );

                if let Some(opts) = options {
                    inputs.insert(
                        "epsilon".to_string(),
                        Self::create_immediate_float(opts.epsilon as f32),
                    );
                }
            }

            // Batch/instance normalization (have mean, variance inputs)
            Operation::BatchNormalization { options, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                if input_names.len() >= 2 && op.input_operands().len() >= 2 {
                    let mean_operand_id = op.input_operands()[1];
                    if let Some(mean_operand) = _graph.operand(mean_operand_id)
                        && mean_operand.kind != crate::graph::OperandKind::Constant
                    {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!(
                                "CoreML {} requires mean parameter to be a constant tensor, not a graph input",
                                op.op_type()
                            ),
                        });
                    }
                    inputs.insert("mean".to_string(), Self::create_argument(&input_names[1]));
                }
                if input_names.len() >= 3 && op.input_operands().len() >= 3 {
                    let variance_operand_id = op.input_operands()[2];
                    if let Some(variance_operand) = _graph.operand(variance_operand_id)
                        && variance_operand.kind != crate::graph::OperandKind::Constant
                    {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!(
                                "CoreML {} requires variance parameter to be a constant tensor, not a graph input",
                                op.op_type()
                            ),
                        });
                    }
                    inputs.insert(
                        "variance".to_string(),
                        Self::create_argument(&input_names[2]),
                    );
                }
                if let Some(opts) = options {
                    if let Some(sid) = opts.scale {
                        inputs.insert(
                            "gamma".to_string(),
                            Self::create_argument(&operand_name(_graph, sid)),
                        );
                    } else if input_names.len() >= 4 {
                        inputs.insert("gamma".to_string(), Self::create_argument(&input_names[3]));
                    }
                    if let Some(bid) = opts.bias {
                        inputs.insert(
                            "beta".to_string(),
                            Self::create_argument(&operand_name(_graph, bid)),
                        );
                    } else if input_names.len() >= 5 {
                        inputs.insert("beta".to_string(), Self::create_argument(&input_names[4]));
                    }
                    inputs.insert(
                        "epsilon".to_string(),
                        Self::create_immediate_float(opts.epsilon as f32),
                    );
                }
            }
            Operation::InstanceNormalization { options, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                if input_names.len() >= 2 && op.input_operands().len() >= 2 {
                    let mean_operand_id = op.input_operands()[1];
                    if let Some(mean_operand) = _graph.operand(mean_operand_id)
                        && mean_operand.kind != crate::graph::OperandKind::Constant
                    {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!(
                                "CoreML {} requires mean parameter to be a constant tensor, not a graph input",
                                op.op_type()
                            ),
                        });
                    }
                    inputs.insert("mean".to_string(), Self::create_argument(&input_names[1]));
                }
                if input_names.len() >= 3 && op.input_operands().len() >= 3 {
                    let variance_operand_id = op.input_operands()[2];
                    if let Some(variance_operand) = _graph.operand(variance_operand_id)
                        && variance_operand.kind != crate::graph::OperandKind::Constant
                    {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!(
                                "CoreML {} requires variance parameter to be a constant tensor, not a graph input",
                                op.op_type()
                            ),
                        });
                    }
                    inputs.insert(
                        "variance".to_string(),
                        Self::create_argument(&input_names[2]),
                    );
                }
                if input_names.len() >= 4 {
                    inputs.insert("gamma".to_string(), Self::create_argument(&input_names[3]));
                }
                if input_names.len() >= 5 {
                    inputs.insert("beta".to_string(), Self::create_argument(&input_names[4]));
                }
                if let Some(opts) = options {
                    inputs.insert(
                        "epsilon".to_string(),
                        Self::create_immediate_float(opts.epsilon as f32),
                    );
                }
            }

            Operation::Concat { axis, .. } => {
                // concat: values (variadic list of tensors), axis
                // CoreML expects a single 'values' parameter containing a tuple of all inputs
                if !input_names.is_empty() {
                    inputs.insert(
                        "values".to_string(),
                        Self::create_argument_tuple(input_names),
                    );
                }

                inputs.insert("axis".to_string(), Self::create_immediate_int(*axis));
                inputs.insert("interleave".to_string(), Self::create_immediate_bool(false));
            }

            Operation::Slice {
                starts,
                sizes,
                options,
                ..
            } => {
                // slice_by_size: x, begin, size
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                if !starts.is_empty() {
                    inputs.insert(
                        "begin".to_string(),
                        Self::create_immediate_int_array(starts),
                    );
                }
                if !sizes.is_empty() {
                    let sizes_u32: Vec<u32> = sizes.iter().map(|d| d.static_or_max()).collect();
                    inputs.insert(
                        "size".to_string(),
                        Self::create_immediate_int_array(&sizes_u32),
                    );
                }
                let _ = options;
            }

            Operation::Expand { new_shape, .. } => {
                // CoreML tile operation requires input rank to match reps length
                // If reshape was added before this operation, use reshaped input name
                //  Otherwise use original input

                if let Some(new_shape_u32) = (!new_shape.is_empty()).then(|| {
                    new_shape
                        .iter()
                        .map(MLDimension::static_or_max)
                        .collect::<Vec<u32>>()
                }) {
                    // Get input operand shape
                    if !op.input_operands().is_empty()
                        && let Some(input_operand) = _graph.operand(op.input_operands()[0])
                    {
                        let input_shape = input_operand.descriptor.static_or_max_shape();
                        let input_rank = input_shape.len();
                        let output_rank = new_shape_u32.len();

                        // Determine input name for tile operation
                        let tile_input_name = if input_rank < output_rank {
                            // A reshape was added, use the reshaped output name
                            // The reshape output name is: {input_name}_expand_reshaped
                            format!("{}_expand_reshaped", input_names[0])
                        } else {
                            // No reshape, use original input
                            input_names[0].clone()
                        };

                        inputs.insert("x".to_string(), Self::create_name_argument(tile_input_name));

                        // Create reshaped dimensions (right-aligned, padded with 1s on left)
                        let mut reshaped_dims = vec![1u32; output_rank];
                        for i in 0..input_rank {
                            reshaped_dims[output_rank - i - 1] = input_shape[input_rank - i - 1];
                        }

                        // Calculate reps: reps[i] = output_shape[i] / reshaped_input_shape[i]
                        let reps: Vec<i32> = new_shape_u32
                            .iter()
                            .zip(reshaped_dims.iter())
                            .map(|(&output_dim, &reshaped_dim)| {
                                if reshaped_dim == output_dim {
                                    1
                                } else if reshaped_dim == 1 {
                                    output_dim as i32
                                } else {
                                    // Should not happen - dimensions must match or input must be 1
                                    1
                                }
                            })
                            .collect();

                        inputs.insert("reps".to_string(), Self::create_int_array_argument(reps));
                    }
                }
            }

            Operation::Gather { options, .. } => {
                // gather: x (data), indices, axis, validate_indices
                // CoreML uses 'x' for the data input, not 'params'
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert(
                        "indices".to_string(),
                        Self::create_argument(&input_names[1]),
                    );
                }

                // Add axis parameter (REQUIRED by CoreML, defaults to 0)
                let axis = options.as_ref().map(|o| o.axis).unwrap_or(0);
                inputs.insert("axis".to_string(), Self::create_immediate_int(axis));

                // Add validate_indices parameter (required by CoreML)
                // Chromium sets this to false to avoid validation issues
                // TODO: Handle negative and out-of-bounds indices properly
                inputs.insert(
                    "validate_indices".to_string(),
                    Self::create_immediate_bool(false),
                );
            }

            Operation::GatherElements { options, .. } => {
                // gather_along_axis: x, indices, axis, validate_indices
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert(
                        "indices".to_string(),
                        Self::create_argument(&input_names[1]),
                    );
                }

                let axis = options.as_ref().map(|o| o.axis).unwrap_or(0);
                inputs.insert("axis".to_string(), Self::create_immediate_int(axis));

                inputs.insert(
                    "validate_indices".to_string(),
                    Self::create_immediate_bool(false),
                );
            }

            Operation::Split {
                splits, options, ..
            } => {
                // split: x, num_splits or split_sizes, axis
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                if let Some(opts) = options {
                    if splits.is_empty() {
                        inputs.insert(
                            "num_splits".to_string(),
                            Self::create_immediate_int(op.output_operands().len() as u32),
                        );
                    } else {
                        inputs.insert(
                            "split_sizes".to_string(),
                            Self::create_immediate_int_array(splits),
                        );
                    }
                    inputs.insert("axis".to_string(), Self::create_immediate_int(opts.axis));
                }
            }

            Operation::Where { .. } => {
                // select: cond, a (true_value), b (false_value)
                if input_names.len() >= 3 {
                    inputs.insert("cond".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("a".to_string(), Self::create_argument(&input_names[1]));
                    inputs.insert("b".to_string(), Self::create_argument(&input_names[2]));
                }
            }

            Operation::Pad {
                beginning_padding,
                ending_padding,
                options,
                ..
            } => {
                // pad: x, pad, mode, constant_val
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                if let Some(opts) = options {
                    // CoreML expects pad as [begin_0, end_0, begin_1, end_1, ...]
                    let pad: Vec<u32> = beginning_padding
                        .iter()
                        .zip(ending_padding.iter())
                        .flat_map(|(a, b)| [*a, *b])
                        .collect();
                    if !pad.is_empty() {
                        inputs.insert("pad".to_string(), Self::create_immediate_int_array(&pad));
                    }
                    if let Some(v) = Self::parse_mlnumber_f64(opts.value.as_ref()) {
                        inputs.insert(
                            "constant_val".to_string(),
                            Self::create_immediate_float(v as f32),
                        );
                    }
                }
                // Add mode parameter (defaults to "constant")
                // CoreML pad modes: "constant", "reflect", "replicate"
                // WebNN modes: "constant", "edge", "reflection", "symmetric"
            }

            Operation::Gelu { .. } => {
                // gelu: x (mode is optional, defaults to "EXACT")
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // CoreML GELU supports "EXACT" and "TANH_APPROXIMATION" modes
                // WebNN GELU has no mode parameter (uses exact by default)
            }

            Operation::Squeeze { options, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                if let Some(opts) = options
                    && !opts.axes.is_empty()
                {
                    inputs.insert(
                        "axes".to_string(),
                        Self::create_immediate_int_array(&opts.axes),
                    );
                }
            }

            Operation::Unsqueeze { options, .. } => {
                // expand_dims: x, axes
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                if let Some(opts) = options
                    && !opts.axes.is_empty()
                {
                    inputs.insert(
                        "axes".to_string(),
                        Self::create_immediate_int_array(&opts.axes),
                    );
                }
            }

            Operation::ArgMax { axis, options, .. } | Operation::ArgMin { axis, options, .. } => {
                // reduce_argmax/reduce_argmin: x, axis, keep_dims
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                inputs.insert("axis".to_string(), Self::create_immediate_int(*axis));
                if let Some(opts) = options {
                    inputs.insert(
                        "keep_dims".to_string(),
                        Self::create_immediate_bool(opts.keep_dimensions),
                    );
                }
                // Note: outputDataType is handled by the output tensor's data type
            }

            Operation::Cast { to, .. } => {
                // cast: x, dtype
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add dtype parameter (required)
                if !to.is_empty() {
                    let to_type = to;
                    let dtype_string = match to_type.as_str() {
                        "float32" => "fp32",
                        "float16" => "fp16",
                        "int32" => "int32",
                        "uint32" => "uint32",
                        "int8" => "int8",
                        "uint8" => "uint8",
                        "int64" => "int64",
                        _ => "fp32", // default
                    };
                    inputs.insert(
                        "dtype".to_string(),
                        Self::create_immediate_string(dtype_string),
                    );
                }
            }

            Operation::ScatterElements { options, .. } => {
                // scatter: data, indices, updates, axis
                if input_names.len() >= 3 {
                    inputs.insert("data".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert(
                        "indices".to_string(),
                        Self::create_argument(&input_names[1]),
                    );
                    inputs.insert(
                        "updates".to_string(),
                        Self::create_argument(&input_names[2]),
                    );
                }

                if let Some(opts) = options {
                    inputs.insert("axis".to_string(), Self::create_immediate_int(opts.axis));
                }
            }

            Operation::ScatterND { .. } => {
                // scatter_nd: data, indices, updates
                if input_names.len() >= 3 {
                    inputs.insert("data".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert(
                        "indices".to_string(),
                        Self::create_argument(&input_names[1]),
                    );
                    inputs.insert(
                        "updates".to_string(),
                        Self::create_argument(&input_names[2]),
                    );
                }
            }

            Operation::Tile { repetitions, .. } => {
                // tile: x, reps
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                if !repetitions.is_empty() {
                    inputs.insert(
                        "reps".to_string(),
                        Self::create_immediate_int_array(repetitions),
                    );
                }
            }

            Operation::CumulativeSum { axis, options, .. } => {
                // cumsum: x, axis, exclusive, reverse
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                inputs.insert("axis".to_string(), Self::create_int_argument(*axis as i32));
                if let Some(opts) = options {
                    inputs.insert(
                        "exclusive".to_string(),
                        Self::create_immediate_bool(opts.exclusive),
                    );
                    inputs.insert(
                        "reverse".to_string(),
                        Self::create_immediate_bool(opts.reversed),
                    );
                }
            }

            Operation::Reverse { options, .. } => {
                // reverse: x, axes
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Default behavior: reverse all axes when options.axes is omitted.
                let axes_u32: Vec<u32> = match options.as_ref() {
                    Some(opts) => match opts.axes.as_ref() {
                        Some(axes) => axes.clone(),
                        None => {
                            if let Some(input_id) = op.input_operands().first() {
                                if let Some(input_operand) = _graph.operand(*input_id) {
                                    (0..input_operand.descriptor.shape.len())
                                        .map(|axis| axis as u32)
                                        .collect()
                                } else {
                                    Vec::new()
                                }
                            } else {
                                Vec::new()
                            }
                        }
                    },
                    None => {
                        if let Some(input_id) = op.input_operands().first() {
                            if let Some(input_operand) = _graph.operand(*input_id) {
                                (0..input_operand.descriptor.shape.len())
                                    .map(|axis| axis as u32)
                                    .collect()
                            } else {
                                Vec::new()
                            }
                        } else {
                            Vec::new()
                        }
                    }
                };

                // Always provide axes, including empty arrays (explicit no-op).
                inputs.insert(
                    "axes".to_string(),
                    Self::create_immediate_int_array(&axes_u32),
                );
            }

            Operation::Triangular { options, .. } => {
                // band_part: x, lower, upper
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // CoreML band_part uses lower and upper bounds instead of upper/diagonal
                let is_upper = options.as_ref().and_then(|o| o.upper).unwrap_or(true);
                let diagonal = options.as_ref().map(|o| o.diagonal as i64).unwrap_or(0);

                // Convert WebNN (upper, diagonal) to CoreML (lower, upper)
                // For upper triangle: keep diagonal and above
                // For lower triangle: keep diagonal and below
                let (lower_bound, upper_bound) = if is_upper {
                    // Upper triangle: remove elements below diagonal+k
                    (diagonal as i32, -1) // keep from diagonal+k upward
                } else {
                    // Lower triangle: remove elements above diagonal+k
                    (-1, diagonal as i32) // keep from diagonal+k downward
                };

                inputs.insert(
                    "lower".to_string(),
                    Self::create_immediate_int(lower_bound as u32),
                );
                inputs.insert(
                    "upper".to_string(),
                    Self::create_immediate_int(upper_bound as u32),
                );
            }

            // Reduction operations: reduceSum, reduceMean, reduceMax, etc.
            Operation::ReduceSum { options, .. }
            | Operation::ReduceMean { options, .. }
            | Operation::ReduceMax { options, .. }
            | Operation::ReduceMin { options, .. }
            | Operation::ReduceProduct { options, .. }
            | Operation::ReduceL1 { options, .. }
            | Operation::ReduceL2 { options, .. }
            | Operation::ReduceLogSum { options, .. }
            | Operation::ReduceLogSumExp { options, .. }
            | Operation::ReduceSumSquare { options, .. } => {
                // All reduce operations: x, axes, keep_dims
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                if let Some(opts) = options {
                    if let Some(axes) = opts.axes.as_ref()
                        && !axes.is_empty()
                    {
                        inputs.insert("axes".to_string(), Self::create_immediate_int_array(axes));
                    }
                    inputs.insert(
                        "keep_dims".to_string(),
                        Self::create_immediate_bool(opts.keep_dimensions),
                    );
                }
            }

            _ => {}
        }

        Ok(inputs)
    }

    /// Create a FeatureType for model description from an OperandDescriptor
    fn create_feature_type(
        descriptor: &crate::graph::OperandDescriptor,
    ) -> Result<crate::protos::coreml::specification::FeatureType, GraphError> {
        use crate::protos::coreml::specification::{ArrayFeatureType, FeatureType, feature_type};

        // Map WebNN data type to CoreML array data type
        // CoreML feature descriptions (I/O) ONLY support: DOUBLE, FLOAT32, FLOAT16, INT32
        // Even though Int8 exists in protobuf enum, CoreML runtime rejects it
        let array_data_type = match descriptor.data_type {
            DataType::Float32 => {
                crate::protos::coreml::specification::array_feature_type::ArrayDataType::Float32
            }
            DataType::Float16 => {
                crate::protos::coreml::specification::array_feature_type::ArrayDataType::Float16
            }
            DataType::Int32 => {
                crate::protos::coreml::specification::array_feature_type::ArrayDataType::Int32
            }
            // Unsupported types - assume they have been converted to FLOAT32.
            DataType::Int4
            | DataType::Uint4
            | DataType::Int8
            | DataType::Uint8
            | DataType::Uint32
            | DataType::Int64
            | DataType::Uint64 => {
                crate::protos::coreml::specification::array_feature_type::ArrayDataType::Float32
            }
        };

        // Create array feature type with shape
        let mut array_feature = ArrayFeatureType {
            data_type: array_data_type as i32,
            ..Default::default()
        };

        // Add shape dimensions
        // CoreML requires explicit shape constraints - convert scalars (0D) to 1D [1]
        // Following Chromium's approach for scalar handling
        let shape_to_use = if descriptor.shape.is_empty() {
            vec![1] // Scalar (0D) tensor -> [1] for CoreML compatibility
        } else {
            descriptor.static_or_max_shape()
        };

        for &dim in &shape_to_use {
            array_feature.shape.push(dim as i64);
        }

        Ok(FeatureType {
            r#type: Some(feature_type::Type::MultiArrayType(array_feature)),
            is_optional: false,
        })
    }
}

impl super::GraphConverter for CoremlMlProgramConverter {
    fn format(&self) -> &'static str {
        "coreml"
    }

    fn convert(&self, graph_info: &GraphInfo) -> Result<super::ConvertedGraph, GraphError> {
        if !crate::graph::dynamic_inputs_enabled() && graph_info.has_dynamic_dimensions() {
            return Err(GraphError::DynamicInputsFeatureDisabled);
        }

        // Create weight file builder for Float16 constants
        let mut weight_builder = super::WeightFileBuilder::new();

        // Create MLProgram
        let mut program = Program {
            version: 1,
            ..Default::default()
        };

        // Create main function
        let mut main_function = Function::default();

        // Keep MLProgram boundary types aligned with CoreML feature-description
        // restrictions. Unsupported WebNN I/O types (such as uint8) are exposed
        // as float32 at the function boundary and cast to/from the internal
        // graph representation inside the main block.
        let mut operand_name_overrides: HashMap<u32, String> = HashMap::new();

        for &output_id in &graph_info.output_operands {
            if let Some(operand) = graph_info.operand(output_id) {
                let graph_mil_type = Self::mil_data_type(&operand.descriptor.data_type)?;
                let interface_mil_type =
                    Self::interface_mil_data_type(&operand.descriptor.data_type);
                if graph_mil_type != interface_mil_type {
                    let output_name = operand_name(graph_info, output_id);
                    operand_name_overrides.insert(output_id, format!("{}_graph", output_name));
                }
            }
        }

        // Add function inputs from graph inputs
        for &input_id in &graph_info.input_operands {
            let operand =
                graph_info
                    .operand(input_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Input operand {} not found", input_id),
                    })?;
            let input_name = operand_name(graph_info, input_id);
            let value_type = Self::create_value_with_mil_type(
                graph_info,
                input_id,
                input_name,
                Self::interface_mil_data_type(&operand.descriptor.data_type),
            )?;
            main_function.inputs.push(value_type);
        }

        // Create main block
        let mut main_block = Block::default();

        for &input_id in &graph_info.input_operands {
            let operand =
                graph_info
                    .operand(input_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Input operand {} not found", input_id),
                    })?;
            let graph_mil_type = Self::mil_data_type(&operand.descriptor.data_type)?;
            let interface_mil_type = Self::interface_mil_data_type(&operand.descriptor.data_type);
            if graph_mil_type != interface_mil_type {
                let input_name = operand_name(graph_info, input_id);
                let graph_input_name = format!("{}_graph", input_name);
                operand_name_overrides.insert(input_id, graph_input_name.clone());
                let graph_input_type = Self::create_value_with_mil_type(
                    graph_info,
                    input_id,
                    graph_input_name,
                    graph_mil_type,
                )?;
                main_block.operations.push(Self::create_cast_operation(
                    input_name,
                    graph_input_type,
                    Self::cast_dtype_string_for_graph_type(&operand.descriptor.data_type)?,
                ));
            }
        }

        // Add constant operands as const operations
        for (operand_id, constant_data) in &graph_info.constant_operand_ids_to_handles {
            let operand =
                graph_info
                    .operand(*operand_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Constant operand {} not found", operand_id),
                    })?;

            let const_op = Self::create_const_operation(
                graph_info,
                *operand_id,
                operand,
                constant_data,
                &mut weight_builder,
            )?;
            main_block.operations.push(const_op);
        }

        // First pass: Handle filter layout transformations for conv operations

        for op in &graph_info.operations {
            let op_type_lower = op.op_type().to_lowercase();

            if (op_type_lower == "conv2d" || op_type_lower == "convtranspose2d")
                && op.input_operands().len() >= 2
            {
                let filter_layout = match &op {
                    Operation::Conv2d { options, .. } => options
                        .as_ref()
                        .map(|o| o.filter_layout.as_str())
                        .unwrap_or(""),
                    Operation::ConvTranspose2d { options, .. } => options
                        .as_ref()
                        .map(|o| o.filter_layout.as_str())
                        .unwrap_or(""),
                    _ => "",
                };
                if !filter_layout.is_empty() {
                    let expected_layout = if op_type_lower == "conv2d" {
                        "oihw"
                    } else {
                        "iohw"
                    };

                    if filter_layout != expected_layout {
                        let filter_operand_id = op.input_operands()[1];

                        if let Some(filter_operand) = graph_info.operand(filter_operand_id) {
                            // Calculate transpose permutation
                            let perm = match (op_type_lower.as_str(), filter_layout) {
                                // Conv2d conversions to oihw [O, I, H, W]
                                ("conv2d", "hwio") => vec![3, 2, 0, 1], // [H, W, I, O] -> [O, I, H, W]
                                ("conv2d", "ohwi") => vec![0, 3, 1, 2], // [O, H, W, I] -> [O, I, H, W]
                                ("conv2d", "ihwo") => vec![3, 0, 1, 2], // [I, H, W, O] -> [O, I, H, W]

                                // Conv_transpose2d conversions to iohw [I, O, H, W]
                                ("convtranspose2d", "hwoi") => vec![3, 2, 0, 1], // [H, W, O, I] -> [I, O, H, W]
                                ("convtranspose2d", "ohwi") => vec![3, 0, 1, 2], // [O, H, W, I] -> [I, O, H, W]
                                ("convtranspose2d", "hwio") => vec![2, 3, 0, 1], // [H, W, I, O] -> [I, O, H, W]

                                _ => continue, // Skip unsupported layouts
                            };

                            // Create transpose operation for filter
                            let filter_name = operand_name(graph_info, filter_operand_id);
                            let transposed_filter_name = format!("{}_transposed", filter_name);

                            // Store the override mapping
                            operand_name_overrides
                                .insert(filter_operand_id, transposed_filter_name.clone());

                            let mut transpose_inputs: HashMap<String, Argument> = HashMap::new();
                            transpose_inputs
                                .insert("x".to_string(), Self::create_name_argument(filter_name));
                            transpose_inputs.insert(
                                "perm".to_string(),
                                Self::create_immediate_int_array(&perm),
                            );

                            // Create tensor type for transposed filter
                            let dtype = Self::mil_data_type(&filter_operand.descriptor.data_type)?;
                            let transposed_shape =
                                Self::permute_graph_shape(&filter_operand.descriptor.shape, &perm);
                            let dimensions =
                                Self::mil_dimensions_from_graph_shape(&transposed_shape, false);

                            let value_type = ValueType {
                                r#type: Some(
                                    crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                        TensorType {
                                            rank: dimensions.len() as i64,
                                            data_type: dtype,
                                            dimensions,
                                            attributes: HashMap::new(),
                                        },
                                    ),
                                ),
                            };

                            let transpose_output_type = NamedValueType {
                                name: transposed_filter_name.clone(),
                                r#type: Some(value_type),
                            };

                            let transpose_op = Self::create_mil_operation(
                                "transpose",
                                transpose_inputs,
                                vec![transpose_output_type],
                            );

                            main_block.operations.push(transpose_op);
                        }
                    }
                }

                let input_layout = match &op {
                    Operation::Conv2d { options, .. } => options
                        .as_ref()
                        .map(|o| o.input_layout.as_str())
                        .unwrap_or(""),
                    Operation::ConvTranspose2d { options, .. } => options
                        .as_ref()
                        .map(|o| o.input_layout.as_str())
                        .unwrap_or(""),
                    _ => "",
                };
                if input_layout == "nhwc" && !op.input_operands().is_empty() {
                    let input_operand_id = op.input_operands()[0];

                    // Only transpose if not already transposed
                    if !operand_name_overrides.contains_key(&input_operand_id)
                        && let Some(input_operand) = graph_info.operand(input_operand_id)
                    {
                        // NHWC -> NCHW transposition: [0, 3, 1, 2]
                        let perm = [0, 3, 1, 2];

                        // Create transpose operation for input
                        let input_name = operand_name(graph_info, input_operand_id);
                        let transposed_input_name = format!("{}_nchw", input_name);

                        // Store the override mapping
                        operand_name_overrides
                            .insert(input_operand_id, transposed_input_name.clone());

                        let mut transpose_inputs: HashMap<String, Argument> = HashMap::new();
                        transpose_inputs
                            .insert("x".to_string(), Self::create_name_argument(input_name));
                        transpose_inputs.insert(
                            "perm".to_string(),
                            Self::create_immediate_int_array(perm.as_ref()),
                        );

                        // Create tensor type for transposed input
                        let dtype = Self::mil_data_type(&input_operand.descriptor.data_type)?;
                        let transposed_shape =
                            Self::permute_graph_shape(&input_operand.descriptor.shape, &perm);
                        let dimensions =
                            Self::mil_dimensions_from_graph_shape(&transposed_shape, false);

                        let value_type = ValueType {
                            r#type: Some(
                                crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                    TensorType {
                                        rank: dimensions.len() as i64,
                                        data_type: dtype,
                                        dimensions,
                                        attributes: HashMap::new(),
                                    },
                                ),
                            ),
                        };

                        let transpose_output_type = NamedValueType {
                            name: transposed_input_name.clone(),
                            r#type: Some(value_type),
                        };

                        let transpose_op = Self::create_mil_operation(
                            "transpose",
                            transpose_inputs,
                            vec![transpose_output_type],
                        );

                        main_block.operations.push(transpose_op);
                    }
                }
            }
        }

        // Convert all operations to MIL operations
        for op in &graph_info.operations {
            let op_type_lower = op.op_type().to_lowercase();

            if matches!(
                op_type_lower.as_str(),
                "equal"
                    | "greater"
                    | "greaterorequal"
                    | "lesser"
                    | "lesserorequal"
                    | "logicalnot"
                    | "logicaland"
                    | "logicalor"
                    | "logicalxor"
                    | "notequal"
            ) {
                use crate::protos::coreml::mil_spec::DataType as MilDataType;

                let output_id =
                    op.output_operand()
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "CoreML MLProgram".to_string(),
                            reason: format!("operation '{}' has no output operand", op.op_type()),
                        })?;
                let output_operand =
                    graph_info
                        .operand(output_id)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("Output operand {} not found", output_id),
                        })?;
                if output_operand.descriptor.data_type != DataType::Uint8 {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!(
                            "CoreML logical op '{}' expects uint8 graph output, got {:?}",
                            op.op_type(),
                            output_operand.descriptor.data_type
                        ),
                    });
                }

                let (output_name, output_type) =
                    Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;
                let bool_output_name = format!("{}_bool", output_name);
                let bool_output_type = Self::create_value_with_mil_type(
                    graph_info,
                    output_id,
                    bool_output_name.clone(),
                    MilDataType::Bool as i32,
                )?;

                let mut input_names =
                    Self::input_names_for_operation(graph_info, op, &operand_name_overrides);

                if matches!(
                    op_type_lower.as_str(),
                    "logicalnot" | "logicaland" | "logicalor" | "logicalxor"
                ) {
                    for (index, &input_id) in op.input_operands().iter().enumerate() {
                        let input_operand = graph_info.operand(input_id).ok_or_else(|| {
                            GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: format!("Input operand {} not found", input_id),
                            }
                        })?;
                        if input_operand.descriptor.data_type == DataType::Uint8 {
                            let bool_input_name = format!("{}_bool", input_names[index]);
                            let bool_input_type = Self::create_value_with_mil_type(
                                graph_info,
                                input_id,
                                bool_input_name.clone(),
                                MilDataType::Bool as i32,
                            )?;
                            main_block.operations.push(Self::create_cast_operation(
                                input_names[index].clone(),
                                bool_input_type,
                                "bool",
                            ));
                            input_names[index] = bool_input_name;
                        }
                    }
                }

                if op_type_lower == "notequal" {
                    let equal_output_name = format!("{}_equal", output_name);
                    let equal_output_type = Self::create_value_with_mil_type(
                        graph_info,
                        output_id,
                        equal_output_name.clone(),
                        MilDataType::Bool as i32,
                    )?;

                    let mut equal_inputs = HashMap::new();
                    equal_inputs.insert(
                        "x".to_string(),
                        Self::create_name_argument(input_names[0].clone()),
                    );
                    equal_inputs.insert(
                        "y".to_string(),
                        Self::create_name_argument(input_names[1].clone()),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        mil_ops::EQUAL,
                        equal_inputs,
                        vec![equal_output_type],
                    ));

                    let mut not_inputs = HashMap::new();
                    not_inputs.insert(
                        "x".to_string(),
                        Self::create_name_argument(equal_output_name),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        mil_ops::LOGICAL_NOT,
                        not_inputs,
                        vec![bool_output_type],
                    ));
                } else {
                    let mil_op = self.convert_operation_with_input_names_and_outputs(
                        graph_info,
                        op,
                        &input_names,
                        vec![bool_output_type],
                        self.get_mil_op_type(op.op_type())?,
                    )?;
                    main_block.operations.push(mil_op);
                }

                main_block.operations.push(Self::create_cast_operation(
                    bool_output_name,
                    output_type,
                    "uint8",
                ));
                continue;
            }

            // Special handling for clamp with equal bounds.
            // CoreML clip rejects alpha == beta, while WebNN clamp(min==max) is valid and
            // should produce a constant tensor. Lower as: output = input * 0 + bound.
            if op_type_lower == "clamp" {
                let (min_value, max_value) = match &op {
                    Operation::Clamp { options, .. } => options
                        .as_ref()
                        .map(|o| {
                            (
                                Self::parse_clamp_bound(o.min_value.as_ref(), f64::NEG_INFINITY),
                                Self::parse_clamp_bound(o.max_value.as_ref(), f64::INFINITY),
                            )
                        })
                        .unwrap_or((f64::NEG_INFINITY, f64::INFINITY)),
                    _ => (f64::NEG_INFINITY, f64::INFINITY),
                };

                if min_value == max_value {
                    if op.input_operands().is_empty() || op.output_operand().is_none() {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: "clamp requires input and output operand".to_string(),
                        });
                    }

                    let input_id = op.input_operands()[0];
                    let input_operand = graph_info.operand(input_id).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("Input operand {} not found", input_id),
                        }
                    })?;
                    let output_id = op.output_operand().expect("checked above");
                    let (output_name, output_type) =
                        Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;
                    let input_name = operand_name(graph_info, input_id);
                    let zeroed_name = format!("{}_clamp_zeroed", output_name);

                    let use_float16 = input_operand.descriptor.data_type == DataType::Float16;

                    // zeroed = input * 0
                    let mut mul_inputs: HashMap<String, Argument> = HashMap::new();
                    mul_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
                    if use_float16 {
                        mul_inputs.insert("y".to_string(), Self::create_immediate_float16(0.0));
                    } else {
                        mul_inputs.insert("y".to_string(), Self::create_immediate_float(0.0));
                    }

                    let dtype = Self::mil_data_type(&input_operand.descriptor.data_type)?;
                    let dimensions = Self::mil_dimensions_from_graph_shape(
                        &input_operand.descriptor.shape,
                        false,
                    );
                    let zeroed_type = NamedValueType {
                        name: zeroed_name.clone(),
                        r#type: Some(ValueType {
                            r#type: Some(
                                crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                    TensorType {
                                        rank: dimensions.len() as i64,
                                        data_type: dtype,
                                        dimensions,
                                        attributes: HashMap::new(),
                                    },
                                ),
                            ),
                        }),
                    };
                    main_block.operations.push(Self::create_mil_operation(
                        "mul",
                        mul_inputs,
                        vec![zeroed_type],
                    ));

                    // output = zeroed + min_value
                    let mut add_inputs: HashMap<String, Argument> = HashMap::new();
                    add_inputs.insert("x".to_string(), Self::create_name_argument(zeroed_name));
                    if use_float16 {
                        add_inputs.insert(
                            "y".to_string(),
                            Self::create_immediate_float16(min_value as f32),
                        );
                    } else {
                        add_inputs.insert(
                            "y".to_string(),
                            Self::create_immediate_float(min_value as f32),
                        );
                    }
                    main_block.operations.push(Self::create_mil_operation(
                        "add",
                        add_inputs,
                        vec![output_type],
                    ));

                    continue;
                }
            }

            // Special handling for expand operation (may need reshape first)
            if let Operation::Expand {
                new_shape: expand_shape,
                ..
            } = &op
                && !op.input_operands().is_empty()
                && !expand_shape.is_empty()
                && let Some(input_operand) = graph_info.operand(op.input_operands()[0])
            {
                let new_shape_u32: Vec<u32> = expand_shape
                    .iter()
                    .map(MLDimension::static_or_max)
                    .collect();
                let input_shape = input_operand.descriptor.static_or_max_shape();
                let input_rank = input_shape.len();
                let output_rank = new_shape_u32.len();

                #[allow(clippy::collapsible_if)]
                if input_rank < output_rank {
                    let mut reshaped_dims = vec![1u32; output_rank];
                    for i in 0..input_rank {
                        reshaped_dims[output_rank - i - 1] = input_shape[input_rank - i - 1];
                    }

                    //Create reshape operation
                    let input_name = operand_name(graph_info, op.input_operands()[0]);
                    // Use input name to create unique intermediate name (don't rely on output_operands)
                    let reshape_output_name = format!("{}_expand_reshaped", input_name);

                    let mut reshape_inputs: HashMap<String, Argument> = HashMap::new();
                    reshape_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
                    reshape_inputs.insert(
                        "shape".to_string(),
                        Self::create_int_array_argument(
                            reshaped_dims.iter().map(|&v| v as i32).collect(),
                        ),
                    );

                    // Create tensor type for reshape output
                    let dtype = Self::mil_data_type(&input_operand.descriptor.data_type)?;
                    let dimensions: Vec<Dimension> = reshaped_dims
                        .iter()
                        .map(|&d| Dimension {
                            dimension: Some(dimension::Dimension::Constant(
                                dimension::ConstantDimension { size: d as u64 },
                            )),
                        })
                        .collect();

                    let value_type = ValueType {
                        r#type: Some(
                            crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                TensorType {
                                    rank: dimensions.len() as i64,
                                    data_type: dtype,
                                    dimensions,
                                    attributes: HashMap::new(),
                                },
                            ),
                        ),
                    };

                    let reshape_output_type = NamedValueType {
                        name: reshape_output_name.clone(),
                        r#type: Some(value_type),
                    };

                    let reshape_mil_op = Self::create_mil_operation(
                        "reshape",
                        reshape_inputs,
                        vec![reshape_output_type],
                    );

                    main_block.operations.push(reshape_mil_op);
                }
            }

            // Special handling for hardswish (decompose into hardsigmoid + mul)
            // Following Chromium: hardswish = x * hardsigmoid(x, alpha=1/6, beta=0.5)
            // Note: op_type is "hardSwish" but we normalize to lowercase
            if op_type_lower == "hardswish" {
                // Validate inputs/outputs exist
                // Note: hardswish uses output_operand (singular), not output_operands
                if op.input_operands().is_empty() || op.output_operand().is_none() {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: "hardswish requires input and output operand".to_string(),
                    });
                }

                let input_operand =
                    graph_info.operand(op.input_operands()[0]).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("Input operand {} not found", op.input_operands()[0]),
                        }
                    })?;
                {
                    let input_name = operand_name(graph_info, op.input_operands()[0]);
                    let hardsigmoid_output_name = format!("{}_hardswish_hardsigmoid", input_name);

                    // Create hardsigmoid operation with alpha=1/6, beta=0.5
                    let mut hardsigmoid_inputs: HashMap<String, Argument> = HashMap::new();
                    hardsigmoid_inputs.insert(
                        "x".to_string(),
                        Self::create_name_argument(input_name.clone()),
                    );
                    hardsigmoid_inputs
                        .insert("alpha".to_string(), Self::create_immediate_float(1.0 / 6.0));
                    hardsigmoid_inputs
                        .insert("beta".to_string(), Self::create_immediate_float(0.5));

                    // Create tensor type for hardsigmoid output
                    let dtype = Self::mil_data_type(&input_operand.descriptor.data_type)?;
                    let dimensions = Self::mil_dimensions_from_graph_shape(
                        &input_operand.descriptor.shape,
                        false,
                    );

                    let value_type = ValueType {
                        r#type: Some(
                            crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                TensorType {
                                    rank: dimensions.len() as i64,
                                    data_type: dtype,
                                    dimensions,
                                    attributes: HashMap::new(),
                                },
                            ),
                        ),
                    };

                    let hardsigmoid_output_type = NamedValueType {
                        name: hardsigmoid_output_name.clone(),
                        r#type: Some(value_type),
                    };

                    let hardsigmoid_op = Self::create_mil_operation(
                        "sigmoid_hard",
                        hardsigmoid_inputs,
                        vec![hardsigmoid_output_type],
                    );

                    main_block.operations.push(hardsigmoid_op);

                    // Create mul operation: x * hardsigmoid_output
                    let mut mul_inputs: HashMap<String, Argument> = HashMap::new();
                    mul_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
                    mul_inputs.insert(
                        "y".to_string(),
                        Self::create_name_argument(hardsigmoid_output_name),
                    );

                    // Get output name (using singular output_operand field)
                    let output_operand_id = op.output_operand().unwrap();
                    let output_name = operand_name(graph_info, output_operand_id);
                    let output_operand =
                        graph_info.operand(output_operand_id).ok_or_else(|| {
                            GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: format!("Output operand {} not found", output_operand_id),
                            }
                        })?;

                    let output_dtype = Self::mil_data_type(&output_operand.descriptor.data_type)?;
                    let output_dimensions = Self::mil_dimensions_from_graph_shape(
                        &output_operand.descriptor.shape,
                        false,
                    );

                    let output_value_type = ValueType {
                        r#type: Some(
                            crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                TensorType {
                                    rank: output_dimensions.len() as i64,
                                    data_type: output_dtype,
                                    dimensions: output_dimensions,
                                    attributes: HashMap::new(),
                                },
                            ),
                        ),
                    };

                    let mul_output_type = NamedValueType {
                        name: output_name,
                        r#type: Some(output_value_type),
                    };

                    let mul_op =
                        Self::create_mil_operation("mul", mul_inputs, vec![mul_output_type]);

                    main_block.operations.push(mul_op);
                }

                // Skip normal operation conversion for hardswish
                continue;
            }

            // Special handling for gemm: y = alpha * op(a) * op(b) + beta * c
            // Lower to matmul + optional mul(alpha) + optional mul(beta, c) + add.
            if op_type_lower == "gemm" {
                if op.input_operands().len() < 2 || op.output_operand().is_none() {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: "gemm requires at least 2 input operands and 1 output".to_string(),
                    });
                }

                let output_operand_id = op.output_operand().unwrap();
                let output_operand = graph_info.operand(output_operand_id).ok_or_else(|| {
                    GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Output operand {} not found", output_operand_id),
                    }
                })?;

                let (output_name, output_type) = Self::create_value(graph_info, output_operand_id)?;

                let (alpha, beta) = match &op {
                    Operation::Gemm { options, .. } => (
                        options.as_ref().map(|o| o.alpha as f32).unwrap_or(1.0),
                        options.as_ref().map(|o| o.beta as f32).unwrap_or(1.0),
                    ),
                    _ => (1.0, 1.0),
                };

                let has_bias = op.input_operands().len() >= 3;
                let needs_alpha_mul = (alpha - 1.0).abs() > f32::EPSILON;
                let needs_beta_mul = has_bias && (beta - 1.0).abs() > f32::EPSILON;

                let (alpha_arg, beta_arg) = match output_operand.descriptor.data_type {
                    DataType::Float16 => (
                        Self::create_immediate_float16(alpha),
                        Self::create_immediate_float16(beta),
                    ),
                    DataType::Float32 => (
                        Self::create_immediate_float(alpha),
                        Self::create_immediate_float(beta),
                    ),
                    _ => {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!(
                                "gemm currently supports float32/float16 output, got {:?}",
                                output_operand.descriptor.data_type
                            ),
                        });
                    }
                };

                let mut current_name: String;
                let matmul_output_name = if needs_alpha_mul || has_bias {
                    format!("{}_gemm_matmul", output_name)
                } else {
                    output_name.clone()
                };

                let mut matmul_inputs: HashMap<String, Argument> = HashMap::new();
                matmul_inputs.insert(
                    "x".to_string(),
                    Self::create_name_argument(operand_name(graph_info, op.input_operands()[0])),
                );
                matmul_inputs.insert(
                    "y".to_string(),
                    Self::create_name_argument(operand_name(graph_info, op.input_operands()[1])),
                );
                let (a_transpose, b_transpose) = match &op {
                    Operation::Gemm { options, .. } => (
                        options.as_ref().map(|o| o.a_transpose).unwrap_or(false),
                        options.as_ref().map(|o| o.b_transpose).unwrap_or(false),
                    ),
                    _ => (false, false),
                };
                matmul_inputs.insert(
                    "transpose_x".to_string(),
                    Self::create_immediate_bool(a_transpose),
                );
                matmul_inputs.insert(
                    "transpose_y".to_string(),
                    Self::create_immediate_bool(b_transpose),
                );

                main_block.operations.push(Self::create_mil_operation(
                    "matmul",
                    matmul_inputs,
                    vec![NamedValueType {
                        name: matmul_output_name.clone(),
                        r#type: output_type.r#type.clone(),
                    }],
                ));
                current_name = matmul_output_name;

                if needs_alpha_mul {
                    let alpha_output_name = if has_bias {
                        format!("{}_gemm_alpha", output_name)
                    } else {
                        output_name.clone()
                    };

                    let mut alpha_mul_inputs: HashMap<String, Argument> = HashMap::new();
                    alpha_mul_inputs
                        .insert("x".to_string(), Self::create_name_argument(current_name));
                    alpha_mul_inputs.insert("y".to_string(), alpha_arg);

                    main_block.operations.push(Self::create_mil_operation(
                        "mul",
                        alpha_mul_inputs,
                        vec![NamedValueType {
                            name: alpha_output_name.clone(),
                            r#type: output_type.r#type.clone(),
                        }],
                    ));

                    current_name = alpha_output_name;
                }

                if has_bias {
                    let c_operand_id = op.input_operands()[2];
                    let (c_name, c_type) = Self::create_value(graph_info, c_operand_id)?;
                    let scaled_c_name = if needs_beta_mul {
                        format!("{}_gemm_bias", output_name)
                    } else {
                        c_name.clone()
                    };

                    if needs_beta_mul {
                        let mut beta_mul_inputs: HashMap<String, Argument> = HashMap::new();
                        beta_mul_inputs.insert("x".to_string(), Self::create_name_argument(c_name));
                        beta_mul_inputs.insert("y".to_string(), beta_arg);

                        main_block.operations.push(Self::create_mil_operation(
                            "mul",
                            beta_mul_inputs,
                            vec![NamedValueType {
                                name: scaled_c_name.clone(),
                                r#type: c_type.r#type,
                            }],
                        ));
                    }

                    let mut add_inputs: HashMap<String, Argument> = HashMap::new();
                    add_inputs.insert("x".to_string(), Self::create_name_argument(current_name));
                    add_inputs.insert("y".to_string(), Self::create_name_argument(scaled_c_name));

                    main_block.operations.push(Self::create_mil_operation(
                        "add",
                        add_inputs,
                        vec![NamedValueType {
                            name: output_name,
                            r#type: output_type.r#type,
                        }],
                    ));
                }

                continue;
            }

            // Special handling for linear: y = alpha * x + beta
            // Lower to mul + add primitives for backend parity.
            if op_type_lower == "linear" {
                if op.input_operands().is_empty() || op.output_operand().is_none() {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: "linear requires input and output operand".to_string(),
                    });
                }

                let input_operand =
                    graph_info.operand(op.input_operands()[0]).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("Input operand {} not found", op.input_operands()[0]),
                        }
                    })?;
                let output_operand_id = op.output_operand().unwrap();
                let output_operand = graph_info.operand(output_operand_id).ok_or_else(|| {
                    GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Output operand {} not found", output_operand_id),
                    }
                })?;

                let (alpha, beta) = match &op {
                    Operation::Linear { options, .. } => options
                        .as_ref()
                        .map(|o| (o.alpha as f32, o.beta as f32))
                        .unwrap_or((1.0, 0.0)),
                    _ => (1.0, 0.0),
                };

                let (alpha_arg, beta_arg) = match input_operand.descriptor.data_type {
                    DataType::Float16 => (
                        Self::create_immediate_float16(alpha),
                        Self::create_immediate_float16(beta),
                    ),
                    DataType::Float32 => (
                        Self::create_immediate_float(alpha),
                        Self::create_immediate_float(beta),
                    ),
                    _ => {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!(
                                "linear currently supports float32/float16, got {:?}",
                                input_operand.descriptor.data_type
                            ),
                        });
                    }
                };

                let input_name = operand_name(graph_info, op.input_operands()[0]);
                let output_name = operand_name(graph_info, output_operand_id);
                let mul_output_name = format!("{}_linear_mul", output_name);

                let output_dtype = Self::mil_data_type(&output_operand.descriptor.data_type)?;
                let output_shape = if output_operand.descriptor.shape.is_empty() {
                    vec![1u32]
                } else {
                    output_operand.descriptor.static_or_max_shape()
                };
                let output_dimensions: Vec<Dimension> = output_shape
                    .iter()
                    .map(|&d| Dimension {
                        dimension: Some(dimension::Dimension::Constant(
                            dimension::ConstantDimension { size: d as u64 },
                        )),
                    })
                    .collect();

                let value_type = ValueType {
                    r#type: Some(
                        crate::protos::coreml::mil_spec::value_type::Type::TensorType(TensorType {
                            rank: output_dimensions.len() as i64,
                            data_type: output_dtype,
                            dimensions: output_dimensions.clone(),
                            attributes: HashMap::new(),
                        }),
                    ),
                };

                let mut mul_inputs: HashMap<String, Argument> = HashMap::new();
                mul_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
                mul_inputs.insert("y".to_string(), alpha_arg);
                main_block.operations.push(Self::create_mil_operation(
                    "mul",
                    mul_inputs,
                    vec![NamedValueType {
                        name: mul_output_name.clone(),
                        r#type: Some(value_type.clone()),
                    }],
                ));

                let mut add_inputs: HashMap<String, Argument> = HashMap::new();
                add_inputs.insert("x".to_string(), Self::create_name_argument(mul_output_name));
                add_inputs.insert("y".to_string(), beta_arg);
                main_block.operations.push(Self::create_mil_operation(
                    "add",
                    add_inputs,
                    vec![NamedValueType {
                        name: output_name,
                        r#type: Some(value_type),
                    }],
                ));

                continue;
            }

            // Special handling for neg (decompose into mul(x, -1) with typed constant)
            // Following Chromium: neg = mul(x, -1) with constant matching input dtype
            if op_type_lower == "neg" {
                // Validate inputs/outputs exist
                if op.input_operands().is_empty() || op.output_operand().is_none() {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: "neg requires input and output operand".to_string(),
                    });
                }

                let input_operand =
                    graph_info.operand(op.input_operands()[0]).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("Input operand {} not found", op.input_operands()[0]),
                        }
                    })?;

                let input_name = operand_name(graph_info, op.input_operands()[0]);

                // Create typed -1 constant matching input dtype
                let neg_one_immediate = match input_operand.descriptor.data_type {
                    DataType::Float32 => Self::create_immediate_float(-1.0f32),
                    DataType::Float16 => Self::create_immediate_float16(-1.0f32),
                    DataType::Int32 => {
                        // create_immediate_int accepts u32 but converts to i32 internally
                        // We need to reimplement for -1 value
                        use crate::protos::coreml::mil_spec::{
                            DataType as MilDataType, TensorType, TensorValue, Value, ValueType,
                            argument, tensor_value, value, value_type,
                        };

                        let tensor_value = TensorValue {
                            value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                                values: vec![-1i32],
                            })),
                        };

                        let val = Value {
                            doc_string: String::new(),
                            r#type: Some(ValueType {
                                r#type: Some(value_type::Type::TensorType(TensorType {
                                    data_type: MilDataType::Int32 as i32,
                                    rank: 0, // Scalar
                                    dimensions: vec![],
                                    attributes: HashMap::new(),
                                })),
                            }),
                            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                                value: Some(value::immediate_value::Value::Tensor(tensor_value)),
                            })),
                        };

                        Argument {
                            arguments: vec![argument::Binding {
                                binding: Some(argument::binding::Binding::Value(val)),
                            }],
                        }
                    }
                    _ => {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!(
                                "Unsupported data type for neg: {:?}",
                                input_operand.descriptor.data_type
                            ),
                        });
                    }
                };

                // Create mul operation: x * (-1)
                let mut mul_inputs: HashMap<String, Argument> = HashMap::new();
                mul_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
                mul_inputs.insert("y".to_string(), neg_one_immediate);

                // Get output name
                let output_operand_id = op.output_operand().unwrap();
                let output_name = operand_name(graph_info, output_operand_id);
                let output_operand = graph_info.operand(output_operand_id).ok_or_else(|| {
                    GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Output operand {} not found", output_operand_id),
                    }
                })?;

                let output_dtype = Self::mil_data_type(&output_operand.descriptor.data_type)?;
                let output_dimensions =
                    Self::mil_dimensions_from_graph_shape(&output_operand.descriptor.shape, false);

                let output_value_type = ValueType {
                    r#type: Some(
                        crate::protos::coreml::mil_spec::value_type::Type::TensorType(TensorType {
                            rank: output_dimensions.len() as i64,
                            data_type: output_dtype,
                            dimensions: output_dimensions,
                            attributes: HashMap::new(),
                        }),
                    ),
                };

                let mul_output_type = NamedValueType {
                    name: output_name,
                    r#type: Some(output_value_type),
                };

                let mul_op = Self::create_mil_operation("mul", mul_inputs, vec![mul_output_type]);

                main_block.operations.push(mul_op);

                // Skip normal operation conversion for neg
                continue;
            }

            let mil_op =
                self.convert_operation_with_overrides(graph_info, op, &operand_name_overrides)?;
            main_block.operations.push(mil_op);
        }

        // Add block outputs (output operand names)
        for &output_id in &graph_info.output_operands {
            let operand =
                graph_info
                    .operand(output_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Output operand {} not found", output_id),
                    })?;
            let output_name = operand_name(graph_info, output_id);
            let graph_output_name =
                Self::output_name_for_operand(graph_info, output_id, &operand_name_overrides);
            let graph_mil_type = Self::mil_data_type(&operand.descriptor.data_type)?;
            let interface_mil_type = Self::interface_mil_data_type(&operand.descriptor.data_type);
            if graph_mil_type != interface_mil_type {
                let output_type = Self::create_value_with_mil_type(
                    graph_info,
                    output_id,
                    output_name.clone(),
                    interface_mil_type,
                )?;
                main_block.operations.push(Self::create_cast_operation(
                    graph_output_name,
                    output_type,
                    Self::cast_dtype_string_for_mil_type(interface_mil_type)?,
                ));
            }
            main_block.outputs.push(output_name);
        }

        // Add block to function
        main_function.opset = "CoreML7".to_string(); // Specify the active block specialization
        main_function
            .block_specializations
            .insert("CoreML7".to_string(), main_block);

        // Add function to program
        program.functions.insert("main".to_string(), main_function);

        // Create Model
        let mut model = Model {
            specification_version: 9, // CoreML 9 (iOS 18+, macOS 15+) - required for empty inputs
            ..Default::default()
        };

        // Create ModelDescription with function descriptions
        use crate::protos::coreml::specification::{
            FeatureDescription, FunctionDescription, ModelDescription,
        };

        let mut function_desc = FunctionDescription {
            name: "main".to_string(),
            ..Default::default()
        };

        // Add input descriptions
        for &input_id in &graph_info.input_operands {
            if let Some(operand) = graph_info.operand(input_id) {
                let input_name = operand_name(graph_info, input_id);
                function_desc.input.push(FeatureDescription {
                    name: input_name,
                    r#type: Some(Self::create_feature_type(&operand.descriptor)?),
                    ..Default::default()
                });
            }
        }

        // Add output descriptions
        for &output_id in &graph_info.output_operands {
            if let Some(operand) = graph_info.operand(output_id) {
                let output_name = operand_name(graph_info, output_id);
                function_desc.output.push(FeatureDescription {
                    name: output_name,
                    r#type: Some(Self::create_feature_type(&operand.descriptor)?),
                    ..Default::default()
                });
            }
        }

        model.description = Some(ModelDescription {
            functions: vec![function_desc],
            default_function_name: "main".to_string(),
            ..Default::default()
        });

        // Set MLProgram
        model.r#type = Some(crate::protos::coreml::specification::model::Type::MlProgram(program));

        // Serialize to bytes
        let mut buffer = Vec::new();
        model
            .encode(&mut buffer)
            .map_err(|e| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!("Failed to encode model: {}", e),
            })?;

        // Finalize weight file if any weights were added
        let weights_data = if weight_builder.has_weights() {
            Some(weight_builder.finalize())
        } else {
            None
        };

        Ok(super::ConvertedGraph {
            format: "coreml",
            content_type: "application/x-coreml-model",
            data: buffer,
            weights_data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::GraphConverter;
    #[cfg(feature = "dynamic-inputs")]
    use crate::graph::DynamicDimension;
    use crate::graph::{ConstantData, GraphInfo, Operand, OperandDescriptor, OperandKind};
    use crate::operator_options::{OperationExtras, OperatorOptions};
    use crate::operators::Operation;

    /// Build an `Operation` from WebNN-style `op` name, operand indices, and parsed options (tests).
    fn op_from_operator_options(
        op_type: &str,
        input_operands: Vec<u32>,
        output_operand: Option<u32>,
        output_operands: Vec<u32>,
        attributes: OperatorOptions,
    ) -> Operation {
        let output_ids: Vec<u32> = if !output_operands.is_empty() {
            output_operands
        } else if let Some(o) = output_operand {
            vec![o]
        } else {
            Vec::new()
        };
        let operator = Operation::from_operator_options(
            op_type,
            &input_operands,
            &attributes,
            &output_ids,
            OperationExtras::default(),
        )
        .expect("valid test op");
        operator
    }
    #[cfg(feature = "dynamic-inputs")]
    use crate::protos::coreml::mil_spec::dimension;
    use crate::protos::coreml::specification::Model;
    #[cfg(feature = "dynamic-inputs")]
    use crate::protos::coreml::specification::model::Type;
    use prost::Message;
    use std::collections::HashMap;

    fn s(shape: &[u32]) -> Vec<crate::graph::Dimension> {
        crate::graph::to_dimension_vector(shape)
    }

    /// Helper to create a simple graph with a Float16 constant
    fn create_graph_with_float16_constant(
        shape: Vec<crate::graph::Dimension>,
        data: Vec<u8>,
    ) -> GraphInfo {
        let mut graph = GraphInfo {
            input_operands: vec![],
            output_operands: vec![1], // Output is operand 1
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        // Operand 0: Float16 constant
        graph.operands.push(Operand {
            name: Some("constant".to_string()),
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Float16,
                shape: shape.clone(),
                pending_permutation: vec![],
            },
        });

        // Operand 1: Output (relu result)
        graph.operands.push(Operand {
            name: Some("output".to_string()),
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Float16,
                shape,
                pending_permutation: vec![],
            },
        });

        // Add constant data
        graph
            .constant_operand_ids_to_handles
            .insert(0, ConstantData { data, label: None });

        // Add a simple relu operation
        graph.operations.push(op_from_operator_options(
            "relu",
            vec![0],
            Some(1),
            vec![],
            OperatorOptions::default(),
        ));

        graph
    }

    #[test]
    fn test_parse_mlnumber_f64_non_finite_strings() {
        let pos_inf = serde_json::json!("Infinity");
        let neg_inf = serde_json::json!("-Infinity");
        let nan = serde_json::json!("NaN");
        let finite = serde_json::json!("3.5");

        let parsed_pos =
            CoremlMlProgramConverter::parse_mlnumber_f64(Some(&pos_inf)).expect("parse +inf");
        assert!(parsed_pos.is_infinite());
        assert!(parsed_pos.is_sign_positive());

        let parsed_neg =
            CoremlMlProgramConverter::parse_mlnumber_f64(Some(&neg_inf)).expect("parse -inf");
        assert!(parsed_neg.is_infinite());
        assert!(parsed_neg.is_sign_negative());

        let parsed_nan =
            CoremlMlProgramConverter::parse_mlnumber_f64(Some(&nan)).expect("parse nan");
        assert!(parsed_nan.is_nan());

        let parsed_finite =
            CoremlMlProgramConverter::parse_mlnumber_f64(Some(&finite)).expect("parse finite");
        assert_eq!(parsed_finite, 3.5);
    }

    #[test]
    fn test_parse_clamp_bound_nan_uses_default() {
        let nan = serde_json::json!("NaN");
        let value = CoremlMlProgramConverter::parse_clamp_bound(Some(&nan), 42.0);
        assert_eq!(value, 42.0);
    }

    #[test]
    fn test_float16_scalar_constant_uses_immediate_value() {
        // Create a scalar Float16 constant (shape = [])
        let f16_val = half::f16::from_f32(1.5);
        let data = f16_val.to_le_bytes().to_vec();

        let graph = create_graph_with_float16_constant(s(&[]), data.clone());

        // Convert the graph
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph).unwrap();

        // Verify no weights_data (scalar uses immediate value)
        assert!(
            result.weights_data.is_none(),
            "Scalar Float16 should not use weight file"
        );

        // Verify the model data is valid protobuf
        assert!(!result.data.is_empty(), "Model data should not be empty");
    }

    #[test]
    fn test_float16_1d_constant_uses_weight_file() {
        // Create a 1D Float16 constant [3] - non-scalar
        let data = vec![
            0x00, 0x3C, // f16: 1.0
            0x00, 0x40, // f16: 2.0
            0x00, 0x42, // f16: 3.0
        ];

        let graph = create_graph_with_float16_constant(s(&[3]), data.clone());

        // Convert the graph
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph).unwrap();

        // Verify weights_data is present
        assert!(
            result.weights_data.is_some(),
            "Non-scalar Float16 should use weight file"
        );

        let weights = result.weights_data.unwrap();

        // Verify weight file structure
        // Expected structure:
        // [0-3]: sentinel (0xDEADBEEF)
        // [4-11]: count (3)
        // [12-17]: data (6 bytes)
        // [18-63]: padding (46 bytes)
        assert_eq!(weights.len(), 64, "Weight file should be 64-byte aligned");

        // Verify sentinel
        let sentinel = u32::from_le_bytes([weights[0], weights[1], weights[2], weights[3]]);
        assert_eq!(sentinel, 0xDEADBEEF, "Sentinel should be 0xDEADBEEF");

        // Verify count
        let count = u64::from_le_bytes([
            weights[4],
            weights[5],
            weights[6],
            weights[7],
            weights[8],
            weights[9],
            weights[10],
            weights[11],
        ]);
        assert_eq!(count, 3, "Element count should be 3");

        // Verify data
        assert_eq!(
            &weights[12..18],
            &data[..],
            "Weight data should match input"
        );
    }

    #[test]
    fn test_float16_2d_constant_uses_weight_file() {
        // Create a 2D Float16 constant [2, 2]
        let data = vec![
            0x00, 0x3C, // f16: 1.0
            0x00, 0x40, // f16: 2.0
            0x00, 0x42, // f16: 3.0
            0x00, 0x44, // f16: 4.0
        ];

        let graph = create_graph_with_float16_constant(s(&[2, 2]), data.clone());

        // Convert the graph
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph).unwrap();

        // Verify weights_data is present
        assert!(
            result.weights_data.is_some(),
            "2D Float16 constant should use weight file"
        );

        let weights = result.weights_data.unwrap();

        // Verify count matches 2x2 = 4 elements
        let count = u64::from_le_bytes([
            weights[4],
            weights[5],
            weights[6],
            weights[7],
            weights[8],
            weights[9],
            weights[10],
            weights[11],
        ]);
        assert_eq!(count, 4, "Element count should be 4");
    }

    #[test]
    fn test_multiple_float16_constants_in_weight_file() {
        // Create a graph with TWO Float16 constants
        let mut graph = GraphInfo {
            input_operands: vec![],
            output_operands: vec![2],
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        // Operand 0: First Float16 constant [2]
        let data1 = vec![0x00, 0x3C, 0x00, 0x40]; // 1.0, 2.0
        graph.operands.push(Operand {
            name: Some("constant1".to_string()),
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Float16,
                shape: s(&[2]),
                pending_permutation: vec![],
            },
        });
        graph.constant_operand_ids_to_handles.insert(
            0,
            ConstantData {
                data: data1,
                label: None,
            },
        );

        // Operand 1: Second Float16 constant [2]
        let data2 = vec![0x00, 0x42, 0x00, 0x44]; // 3.0, 4.0
        graph.operands.push(Operand {
            name: Some("constant2".to_string()),
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Float16,
                shape: s(&[2]),
                pending_permutation: vec![],
            },
        });
        graph.constant_operand_ids_to_handles.insert(
            1,
            ConstantData {
                data: data2,
                label: None,
            },
        );

        // Operand 2: Output
        graph.operands.push(Operand {
            name: Some("output".to_string()),
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Float16,
                shape: s(&[2]),
                pending_permutation: vec![],
            },
        });

        // Add operation: output = constant1 + constant2
        graph.operations.push(op_from_operator_options(
            "add",
            vec![0, 1],
            Some(2),
            vec![],
            OperatorOptions::default(),
        ));

        // Convert
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph).unwrap();

        // Verify weights_data is present
        assert!(
            result.weights_data.is_some(),
            "Multiple Float16 constants should use weight file"
        );

        let weights = result.weights_data.unwrap();

        // Should have two entries:
        // Entry 1: offset 0, 64 bytes
        // Entry 2: offset 64, 64 bytes
        // Total: 128 bytes
        assert_eq!(
            weights.len(),
            128,
            "Two Float16 constants should result in 128-byte weight file"
        );

        // Verify first entry sentinel at offset 0
        let sentinel1 = u32::from_le_bytes([weights[0], weights[1], weights[2], weights[3]]);
        assert_eq!(sentinel1, 0xDEADBEEF, "First entry sentinel");

        // Verify second entry sentinel at offset 64
        let sentinel2 = u32::from_le_bytes([weights[64], weights[65], weights[66], weights[67]]);
        assert_eq!(sentinel2, 0xDEADBEEF, "Second entry sentinel");
    }

    #[test]
    fn test_float32_constant_no_weight_file() {
        // Create a graph with Float32 constant (should NOT use weight file)
        let mut graph = GraphInfo {
            input_operands: vec![],
            output_operands: vec![1],
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        // Float32 constant
        let data = vec![0x00, 0x00, 0x80, 0x3F]; // 1.0 as f32
        graph.operands.push(Operand {
            name: Some("constant".to_string()),
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: s(&[1]),
                pending_permutation: vec![],
            },
        });
        graph
            .constant_operand_ids_to_handles
            .insert(0, ConstantData { data, label: None });

        // Output
        graph.operands.push(Operand {
            name: Some("output".to_string()),
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: s(&[1]),
                pending_permutation: vec![],
            },
        });

        // Add relu operation
        graph.operations.push(op_from_operator_options(
            "relu",
            vec![0],
            Some(1),
            vec![],
            OperatorOptions::default(),
        ));

        // Convert
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph).unwrap();

        // Verify NO weights_data (Float32 uses immediate values)
        assert!(
            result.weights_data.is_none(),
            "Float32 constants should not use weight file"
        );
    }

    #[test]
    fn test_int4_data_type_rejected() {
        let mut graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };

        // Int4 input
        graph.operands.push(Operand {
            name: Some("input".to_string()),
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Int4,
                shape: s(&[10, 10]),
                pending_permutation: vec![],
            },
        });

        // Output
        graph.operands.push(Operand {
            name: Some("output".to_string()),
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Int4,
                shape: s(&[10, 10]),
                pending_permutation: vec![],
            },
        });

        // Add relu operation
        graph.operations.push(op_from_operator_options(
            "relu",
            vec![0],
            Some(1),
            vec![],
            OperatorOptions::default(),
        ));

        // Convert should fail with Int4
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph);
        assert!(result.is_err());

        match result.unwrap_err() {
            crate::error::GraphError::ConversionFailed { format, reason } => {
                assert_eq!(format, "coreml");
                assert!(reason.contains("int4/uint4"));
                assert!(reason.contains("not supported"));
            }
            _ => panic!("Expected ConversionFailed error"),
        }
    }

    #[test]
    fn test_uint4_data_type_rejected() {
        let mut graph = GraphInfo {
            input_operands: vec![],
            output_operands: vec![1],
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };

        // Uint4 constant
        let data = vec![0x12, 0x34, 0x56, 0x78];
        graph.operands.push(Operand {
            name: Some("constant".to_string()),
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Uint4,
                shape: s(&[8]),
                pending_permutation: vec![],
            },
        });
        graph
            .constant_operand_ids_to_handles
            .insert(0, ConstantData { data, label: None });

        // Output
        graph.operands.push(Operand {
            name: Some("output".to_string()),
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Uint4,
                shape: s(&[8]),
                pending_permutation: vec![],
            },
        });

        // Add relu operation
        graph.operations.push(op_from_operator_options(
            "relu",
            vec![0],
            Some(1),
            vec![],
            OperatorOptions::default(),
        ));

        // Convert should fail with Uint4
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph);
        assert!(result.is_err());

        match result.unwrap_err() {
            crate::error::GraphError::ConversionFailed { format, reason } => {
                assert_eq!(format, "coreml");
                assert!(reason.contains("int4/uint4"));
                assert!(reason.contains("not supported"));
            }
            _ => panic!("Expected ConversionFailed error"),
        }
    }

    #[test]
    fn test_int4_output_rejected() {
        let mut graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };

        // Float32 input
        graph.operands.push(Operand {
            name: Some("input".to_string()),
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: s(&[10, 10]),
                pending_permutation: vec![],
            },
        });

        // Int4 output (this should be rejected when building value info)
        graph.operands.push(Operand {
            name: Some("output".to_string()),
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Int4,
                shape: s(&[10, 10]),
                pending_permutation: vec![],
            },
        });

        // Add relu operation
        graph.operations.push(op_from_operator_options(
            "relu",
            vec![0],
            Some(1),
            vec![],
            OperatorOptions::default(),
        ));

        // Convert should fail
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph);
        assert!(result.is_err());
    }

    #[test]
    fn test_uint4_input_rejected() {
        let mut graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };

        // Uint4 input
        graph.operands.push(Operand {
            name: Some("input".to_string()),
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Uint4,
                shape: s(&[1, 3, 224, 224]),
                pending_permutation: vec![],
            },
        });

        // Float32 output
        graph.operands.push(Operand {
            name: Some("output".to_string()),
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: s(&[1, 3, 224, 224]),
                pending_permutation: vec![],
            },
        });

        // Add relu operation
        graph.operations.push(op_from_operator_options(
            "relu",
            vec![0],
            Some(1),
            vec![],
            OperatorOptions::default(),
        ));

        // Convert should fail
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_float32_converts_to_mul_add_ops() {
        let graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![
                Operand {
                    name: Some("input".to_string()),
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[2, 3]),
                        pending_permutation: vec![],
                    },
                },
                Operand {
                    name: Some("output".to_string()),
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[2, 3]),
                        pending_permutation: vec![],
                    },
                },
            ],
            operations: vec![op_from_operator_options(
                "linear",
                vec![0],
                Some(1),
                vec![],
                OperatorOptions::from_json_with_op_type(
                    "linear",
                    &serde_json::json!({ "alpha": 2.0, "beta": -1.0 }),
                )
                .expect("linear options"),
            )],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let converted = CoremlMlProgramConverter
            .convert(&graph)
            .expect("coreml linear float32 conversion should succeed");
        let model = Model::decode(converted.data.as_slice()).expect("decode coreml model");
        let program = match model.r#type.expect("model type") {
            crate::protos::coreml::specification::model::Type::MlProgram(program) => program,
            _ => panic!("expected MLProgram model"),
        };
        let main_fn = program.functions.get("main").expect("main function");
        let main_block = main_fn
            .block_specializations
            .get("CoreML7")
            .expect("CoreML7 block");

        assert!(main_block.operations.iter().any(|op| op.r#type == "mul"));
        assert!(main_block.operations.iter().any(|op| op.r#type == "add"));
    }

    #[cfg(not(feature = "dynamic-inputs"))]
    #[test]
    fn test_dynamic_dimensions_require_feature_opt_in() {
        let graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![
                Operand {
                    name: Some("input".to_string()),
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![
                            crate::graph::Dimension::Dynamic(crate::graph::DynamicDimension {
                                name: "batch".to_string(),
                                max_size: 8,
                            }),
                            crate::graph::Dimension::Static(4),
                        ],
                        pending_permutation: vec![],
                    },
                },
                Operand {
                    name: Some("output".to_string()),
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![
                            crate::graph::Dimension::Dynamic(crate::graph::DynamicDimension {
                                name: "batch".to_string(),
                                max_size: 8,
                            }),
                            crate::graph::Dimension::Static(4),
                        ],
                        pending_permutation: vec![],
                    },
                },
            ],
            operations: vec![op_from_operator_options(
                "identity",
                vec![0],
                Some(1),
                vec![],
                OperatorOptions::default(),
            )],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let err = CoremlMlProgramConverter.convert(&graph).unwrap_err();
        assert!(matches!(err, GraphError::DynamicInputsFeatureDisabled));
    }

    #[cfg(feature = "dynamic-inputs")]
    #[test]
    fn test_dynamic_input_dim_maps_to_unknown_mil_dimension() {
        let mut graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        graph.operands.push(Operand {
            name: Some("input".to_string()),
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: vec![
                    crate::graph::Dimension::Dynamic(DynamicDimension {
                        name: "batch".to_string(),
                        max_size: 8,
                    }),
                    crate::graph::Dimension::Static(4),
                ],
                pending_permutation: vec![],
            },
        });

        graph.operands.push(Operand {
            name: Some("output".to_string()),
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: vec![
                    crate::graph::Dimension::Dynamic(DynamicDimension {
                        name: "batch".to_string(),
                        max_size: 8,
                    }),
                    crate::graph::Dimension::Static(4),
                ],
                pending_permutation: vec![],
            },
        });

        graph.operations.push(op_from_operator_options(
            "identity",
            vec![0],
            Some(1),
            vec![],
            OperatorOptions::default(),
        ));

        let converter = CoremlMlProgramConverter;
        let converted = converter.convert(&graph).unwrap();
        let model = Model::decode(converted.data.as_slice()).unwrap();
        let program = match model.r#type.unwrap() {
            Type::MlProgram(p) => p,
            _ => panic!("expected mlProgram"),
        };
        let main = program.functions.get("main").expect("main function");
        let input = main.inputs.first().expect("input");
        let tensor = match input
            .r#type
            .as_ref()
            .and_then(|t| t.r#type.as_ref())
            .expect("input type")
        {
            crate::protos::coreml::mil_spec::value_type::Type::TensorType(t) => t,
            _ => panic!("expected tensor input"),
        };

        match tensor.dimensions[0].dimension.as_ref().expect("dim 0") {
            dimension::Dimension::Unknown(_) => {}
            _ => panic!("expected unknown dimension for dynamic batch"),
        }

        match tensor.dimensions[1].dimension.as_ref().expect("dim 1") {
            dimension::Dimension::Constant(c) => assert_eq!(c.size, 4),
            _ => panic!("expected constant dimension for static axis"),
        }
    }

    #[test]
    fn test_linear_float16_converts_successfully() {
        let graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![
                Operand {
                    name: Some("input".to_string()),
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float16,
                        shape: s(&[4]),
                        pending_permutation: vec![],
                    },
                },
                Operand {
                    name: Some("output".to_string()),
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float16,
                        shape: s(&[4]),
                        pending_permutation: vec![],
                    },
                },
            ],
            operations: vec![op_from_operator_options(
                "linear",
                vec![0],
                Some(1),
                vec![],
                OperatorOptions::default(),
            )],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let converted = CoremlMlProgramConverter
            .convert(&graph)
            .expect("coreml linear float16 conversion should succeed");
        let model = Model::decode(converted.data.as_slice()).expect("decode coreml model");
        assert!(model.r#type.is_some(), "model type should be set");
    }

    #[test]
    fn test_cumulative_sum_converts_to_cumsum_op() {
        let graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![
                Operand {
                    name: Some("input".to_string()),
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![
                            crate::graph::Dimension::Static(2),
                            crate::graph::Dimension::Static(3),
                        ],
                        pending_permutation: vec![],
                    },
                },
                Operand {
                    name: Some("output".to_string()),
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![
                            crate::graph::Dimension::Static(2),
                            crate::graph::Dimension::Static(3),
                        ],
                        pending_permutation: vec![],
                    },
                },
            ],
            operations: vec![op_from_operator_options(
                "cumulativeSum",
                vec![0],
                Some(1),
                vec![],
                OperatorOptions::from_json_with_op_type(
                    "cumulativeSum",
                    &serde_json::json!({ "axis": 1, "exclusive": true, "reversed": true }),
                )
                .expect("cumulativeSum options"),
            )],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let converted = CoremlMlProgramConverter
            .convert(&graph)
            .expect("coreml cumulativeSum conversion should succeed");
        let model = Model::decode(converted.data.as_slice()).expect("decode coreml model");
        let program = match model.r#type.expect("model type") {
            crate::protos::coreml::specification::model::Type::MlProgram(program) => program,
            _ => panic!("expected MLProgram model"),
        };
        let main_fn = program.functions.get("main").expect("main function");
        let main_block = main_fn
            .block_specializations
            .get("CoreML7")
            .expect("CoreML7 block");

        assert!(main_block.operations.iter().any(|op| op.r#type == "cumsum"));
    }
}
